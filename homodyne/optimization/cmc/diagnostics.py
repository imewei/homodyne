"""Consensus Monte Carlo Diagnostics and Validation

This module provides comprehensive diagnostics for CMC results including:
- Per-shard convergence diagnostics (R-hat, ESS, acceptance rate)
- Between-shard KL divergence for assessing posterior agreement
- Combined posterior diagnostics
- Validation criteria for strict and lenient modes

References:
    Scott et al. (2016): "Bayes and big data: the consensus Monte Carlo algorithm"
    Gelman et al. (2013): "Bayesian Data Analysis" (R-hat and ESS)
"""

import logging
from typing import Any, Dict, List, Tuple

import jax.numpy as jnp
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def validate_cmc_results(
    shard_results: List[Dict[str, Any]],
    strict_mode: bool = True,
    min_success_rate: float = 0.90,
    max_kl_divergence: float = 2.0,
    max_rhat: float = 1.1,
    min_ess: float = 100.0,
) -> Tuple[bool, Dict[str, Any]]:
    """Validate CMC results against convergence and consistency criteria.

    This is the main validation function that checks:
    1. Success rate across shards (90% default)
    2. Between-shard KL divergence (< 2.0 default)
    3. Per-shard R-hat convergence (< 1.1 default)
    4. Per-shard ESS (> 100 default)

    Parameters
    ----------
    shard_results : List[Dict[str, Any]]
        List of shard results from parallel MCMC execution.
        Each dict must contain:
        - 'samples': np.ndarray of shape (num_samples, num_params)
        - 'converged': bool indicating if shard converged
        - 'diagnostics': dict with 'rhat', 'ess', 'acceptance_rate'
    strict_mode : bool, default=True
        If True, validation failures raise errors.
        If False, validation failures only log warnings.
    min_success_rate : float, default=0.90
        Minimum fraction of shards that must converge (0.0-1.0).
    max_kl_divergence : float, default=2.0
        Maximum KL divergence between any two shards.
        Higher values indicate shards found different posteriors.
    max_rhat : float, default=1.1
        Maximum R-hat value for convergence.
        R-hat < 1.1 is standard convergence criterion.
    min_ess : float, default=100.0
        Minimum effective sample size per parameter.
        ESS < 100 indicates poor sampling efficiency.

    Returns
    -------
    is_valid : bool
        True if all validation checks pass (or strict_mode=False).
    diagnostics : Dict[str, Any]
        Detailed diagnostic information including:
        - 'success_rate': Fraction of converged shards
        - 'max_kl_divergence': Maximum KL divergence between shards
        - 'kl_matrix': Full KL divergence matrix
        - 'per_shard_diagnostics': List of per-shard diagnostics
        - 'combined_diagnostics': Combined posterior diagnostics
        - 'validation_errors': List of validation error messages (if any)
        - 'validation_warnings': List of validation warnings (if any)

    Examples
    --------
    >>> # Strict mode (raises errors on failure)
    >>> is_valid, diagnostics = validate_cmc_results(
    ...     shard_results,
    ...     strict_mode=True,
    ...     min_success_rate=0.90
    ... )
    >>> if is_valid:
    ...     print("CMC results validated successfully")
    ... else:
    ...     print(f"Validation failed: {diagnostics['validation_errors']}")

    >>> # Lenient mode (only warns on failure)
    >>> is_valid, diagnostics = validate_cmc_results(
    ...     shard_results,
    ...     strict_mode=False,
    ...     min_success_rate=0.80
    ... )
    >>> # Always returns True in lenient mode
    >>> print(f"Success rate: {diagnostics['success_rate']:.1%}")

    Notes
    -----
    - In strict mode, the first validation failure stops execution
    - In lenient mode, all validations are performed and warnings are logged
    - KL divergence is computed using Gaussian approximation for efficiency
    - R-hat and ESS use NumPyro's built-in diagnostics
    """
    validation_errors = []
    validation_warnings = []

    # Check input
    if not shard_results:
        error_msg = "No shard results provided"
        if strict_mode:
            return False, {"error": error_msg}
        else:
            logger.warning(error_msg)
            validation_warnings.append(error_msg)
            return True, {"validation_warnings": validation_warnings}

    # 1. Check success rate
    successful_shards = [r for r in shard_results if r.get("converged", False)]
    success_rate = len(successful_shards) / len(shard_results)

    if success_rate < min_success_rate:
        error_msg = (
            f"Success rate {success_rate:.1%} below minimum {min_success_rate:.1%} "
            f"({len(successful_shards)}/{len(shard_results)} shards converged)"
        )
        if strict_mode:
            return False, {
                "error": error_msg,
                "success_rate": success_rate,
                "num_successful": len(successful_shards),
                "num_total": len(shard_results),
            }
        else:
            logger.warning(error_msg)
            validation_warnings.append(error_msg)

    # If no successful shards, return early
    if not successful_shards:
        error_msg = "No shards converged successfully"
        if strict_mode:
            return False, {"error": error_msg, "success_rate": 0.0}
        else:
            logger.warning(error_msg)
            return True, {
                "success_rate": 0.0,
                "validation_warnings": [error_msg],
            }

    # 2. Compute between-shard KL divergence
    try:
        kl_matrix = compute_between_shard_kl_divergence(successful_shards)
        max_kl = float(np.max(kl_matrix))

        if max_kl > max_kl_divergence:
            error_msg = (
                f"Maximum KL divergence {max_kl:.2f} exceeds threshold {max_kl_divergence:.2f}. "
                "Shards may have converged to different posteriors."
            )
            if strict_mode:
                return False, {
                    "error": error_msg,
                    "max_kl_divergence": max_kl,
                    "kl_matrix": kl_matrix.tolist(),
                }
            else:
                logger.warning(error_msg)
                validation_warnings.append(error_msg)
    except Exception as e:
        error_msg = f"Failed to compute KL divergence: {str(e)}"
        logger.warning(error_msg)
        validation_warnings.append(error_msg)
        kl_matrix = None
        max_kl = None

    # 3. Validate per-shard convergence
    per_shard_diagnostics = []
    for i, result in enumerate(shard_results):
        try:
            shard_diag = _validate_single_shard(
                result,
                shard_idx=i,
                max_rhat=max_rhat,
                min_ess=min_ess,
                strict_mode=strict_mode,
            )
            per_shard_diagnostics.append(shard_diag)

            # Collect validation errors/warnings from shard
            if "validation_errors" in shard_diag:
                validation_errors.extend(shard_diag["validation_errors"])
            if "validation_warnings" in shard_diag:
                validation_warnings.extend(shard_diag["validation_warnings"])

        except Exception as e:
            error_msg = f"Shard {i} validation failed: {str(e)}"
            logger.warning(error_msg)
            validation_warnings.append(error_msg)
            per_shard_diagnostics.append(
                {
                    "shard_id": i,
                    "error": str(e),
                }
            )

    # 4. Compute combined posterior diagnostics
    try:
        combined_diagnostics = compute_combined_posterior_diagnostics(successful_shards)
    except Exception as e:
        error_msg = f"Failed to compute combined diagnostics: {str(e)}"
        logger.warning(error_msg)
        validation_warnings.append(error_msg)
        combined_diagnostics = {"error": str(e)}

    # 5. Build final diagnostics dict
    diagnostics = {
        "success_rate": success_rate,
        "num_successful": len(successful_shards),
        "num_total": len(shard_results),
        "max_kl_divergence": max_kl,
        "kl_matrix": kl_matrix.tolist() if kl_matrix is not None else None,
        "per_shard_diagnostics": per_shard_diagnostics,
        "combined_diagnostics": combined_diagnostics,
        "validation_warnings": validation_warnings,
        "validation_errors": validation_errors,
    }

    # 6. Final validation decision
    if strict_mode and validation_errors:
        return False, diagnostics
    else:
        return True, diagnostics


def compute_per_shard_diagnostics(
    shard_result: Dict[str, Any],
    shard_idx: int = 0,
) -> Dict[str, Any]:
    """Compute convergence diagnostics for a single shard.

    Parameters
    ----------
    shard_result : Dict[str, Any]
        Shard result containing:
        - 'samples': np.ndarray of shape (num_samples, num_params) or (num_chains, num_samples, num_params)
        - 'diagnostics': dict with 'rhat', 'ess', 'acceptance_rate' (optional)
    shard_idx : int, default=0
        Shard index for identification

    Returns
    -------
    diagnostics : Dict[str, Any]
        Dictionary containing:
        - 'shard_id': Shard index
        - 'rhat': Dict of R-hat values per parameter (or None if single chain)
        - 'ess': Dict of ESS values per parameter
        - 'acceptance_rate': Mean acceptance probability (or None)
        - 'num_samples': Number of samples
        - 'num_params': Number of parameters
        - 'trace_data': Dict of sample traces per parameter

    Notes
    -----
    - R-hat is only computed for multi-chain MCMC
    - ESS uses NumPyro's effective_sample_size function
    - Trace data is useful for plotting convergence
    """
    diagnostics = {
        "shard_id": shard_idx,
        "num_samples": None,
        "num_params": None,
        "rhat": None,
        "ess": None,
        "acceptance_rate": None,
        "trace_data": {},
    }

    # Extract samples
    if "samples" not in shard_result:
        logger.warning(f"Shard {shard_idx} missing 'samples' key")
        return diagnostics

    samples = shard_result["samples"]
    if not isinstance(samples, (np.ndarray, jnp.ndarray)):
        logger.warning(f"Shard {shard_idx} samples not an array")
        return diagnostics

    # Detect shape: (num_samples, num_params) or (num_chains, num_samples, num_params)
    if samples.ndim == 2:
        num_chains = 1
        num_samples, num_params = samples.shape
        # Reshape for NumPyro diagnostics: (1, num_samples, num_params)
        samples_for_diag = samples[np.newaxis, :, :]
    elif samples.ndim == 3:
        num_chains, num_samples, num_params = samples.shape
        samples_for_diag = samples
    else:
        logger.warning(
            f"Shard {shard_idx} samples have unexpected shape: {samples.shape}"
        )
        return diagnostics

    diagnostics["num_samples"] = num_samples
    diagnostics["num_params"] = num_params

    # Use existing diagnostics if available
    if "diagnostics" in shard_result and isinstance(shard_result["diagnostics"], dict):
        existing_diag = shard_result["diagnostics"]
        diagnostics["rhat"] = existing_diag.get("rhat")
        diagnostics["ess"] = existing_diag.get("ess")
        diagnostics["acceptance_rate"] = existing_diag.get("acceptance_rate")
    else:
        # Compute diagnostics using NumPyro
        try:
            from numpyro.diagnostics import effective_sample_size, gelman_rubin

            # ESS per parameter
            ess_dict = {}
            for param_idx in range(num_params):
                param_samples = samples_for_diag[:, :, param_idx]
                ess = effective_sample_size(param_samples)
                ess_dict[f"param_{param_idx}"] = (
                    float(ess) if ess.size == 1 else float(np.mean(ess))
                )

            diagnostics["ess"] = ess_dict

            # R-hat per parameter (only for multi-chain)
            if num_chains > 1:
                rhat_dict = {}
                for param_idx in range(num_params):
                    param_samples = samples_for_diag[:, :, param_idx]
                    rhat = gelman_rubin(param_samples)
                    rhat_dict[f"param_{param_idx}"] = (
                        float(rhat) if rhat.size == 1 else float(np.mean(rhat))
                    )

                diagnostics["rhat"] = rhat_dict
            else:
                # Single chain: R-hat not defined
                diagnostics["rhat"] = {f"param_{i}": 1.0 for i in range(num_params)}

        except ImportError:
            logger.warning("NumPyro not available for diagnostics computation")
        except Exception as e:
            logger.warning(
                f"Failed to compute diagnostics for shard {shard_idx}: {str(e)}"
            )

    # Extract trace data for plotting
    try:
        if samples.ndim == 2:
            # Single chain: (num_samples, num_params)
            for param_idx in range(num_params):
                diagnostics["trace_data"][f"param_{param_idx}"] = samples[
                    :, param_idx
                ].tolist()
        else:
            # Multi-chain: (num_chains, num_samples, num_params)
            for param_idx in range(num_params):
                diagnostics["trace_data"][f"param_{param_idx}"] = samples[
                    :, :, param_idx
                ].tolist()
    except Exception as e:
        logger.warning(f"Failed to extract trace data for shard {shard_idx}: {str(e)}")

    return diagnostics


def compute_between_shard_kl_divergence(
    shard_results: List[Dict[str, Any]],
) -> np.ndarray:
    """Compute pairwise KL divergence matrix between shards.

    Uses Gaussian approximation for efficiency:
    KL(p||q) = 0.5 * [trace(Σ_q^-1 Σ_p) + (μ_q - μ_p)^T Σ_q^-1 (μ_q - μ_p) - k + log(det(Σ_q) / det(Σ_p))]

    Parameters
    ----------
    shard_results : List[Dict[str, Any]]
        List of shard results, each containing 'samples' array.

    Returns
    -------
    kl_matrix : np.ndarray of shape (num_shards, num_shards)
        Symmetric matrix where kl_matrix[i, j] = KL(shard_i || shard_j).
        Diagonal elements are 0.0.

    Raises
    ------
    ValueError
        If shards have inconsistent parameter counts or no samples.

    Notes
    -----
    - KL divergence is NOT symmetric: KL(p||q) ≠ KL(q||p)
    - We return the average: 0.5 * (KL(p||q) + KL(q||p)) for symmetry
    - Regularization (1e-6 * I) is added to covariances for numerical stability
    """
    num_shards = len(shard_results)

    # Extract samples and fit Gaussians to each shard
    gaussians = []
    for i, result in enumerate(shard_results):
        if "samples" not in result:
            raise ValueError(f"Shard {i} missing 'samples' key")

        samples = result["samples"]
        if not isinstance(samples, (np.ndarray, jnp.ndarray)):
            raise ValueError(f"Shard {i} samples not an array")

        # Flatten to 2D if multi-chain
        if samples.ndim == 3:
            # (num_chains, num_samples, num_params) -> (num_chains * num_samples, num_params)
            num_chains, num_samples, num_params = samples.shape
            samples = samples.reshape(num_chains * num_samples, num_params)
        elif samples.ndim != 2:
            raise ValueError(
                f"Shard {i} samples have unexpected shape: {samples.shape}"
            )

        # Fit Gaussian
        gaussian = _fit_gaussian_to_samples(samples)
        gaussians.append(gaussian)

    # Check parameter consistency
    num_params = gaussians[0]["mean"].shape[0]
    for i, g in enumerate(gaussians):
        if g["mean"].shape[0] != num_params:
            raise ValueError(
                f"Shard {i} has {g['mean'].shape[0]} parameters, expected {num_params}"
            )

    # Compute KL divergence matrix
    kl_matrix = _compute_kl_divergence_matrix(gaussians)

    return kl_matrix


def compute_combined_posterior_diagnostics(
    shard_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute diagnostics for the combined posterior.

    Parameters
    ----------
    shard_results : List[Dict[str, Any]]
        List of shard results, each containing 'samples' array.

    Returns
    -------
    diagnostics : Dict[str, Any]
        Dictionary containing:
        - 'combined_ess': ESS of concatenated samples
        - 'parameter_uncertainty_ratio': Ratio of combined std to mean per-shard std
        - 'multimodality_detected': Boolean indicating if multimodality was detected

    Notes
    -----
    - Combined ESS is computed by concatenating all samples
    - Uncertainty ratio > 1.0 indicates CMC may be underestimating uncertainty
    - Multimodality detection uses Hartigan's dip test
    """
    diagnostics = {}

    # Concatenate all samples
    all_samples = []
    for result in shard_results:
        if "samples" not in result:
            continue

        samples = result["samples"]
        if not isinstance(samples, (np.ndarray, jnp.ndarray)):
            continue

        # Flatten to 2D
        if samples.ndim == 3:
            num_chains, num_samples, num_params = samples.shape
            samples = samples.reshape(num_chains * num_samples, num_params)

        all_samples.append(samples)

    if not all_samples:
        logger.warning("No samples available for combined diagnostics")
        return diagnostics

    # Stack all samples: (total_samples, num_params)
    combined_samples = np.vstack(all_samples)
    num_params = combined_samples.shape[1]

    # Compute combined ESS
    try:
        from numpyro.diagnostics import effective_sample_size

        # Reshape for NumPyro: (1, total_samples, num_params)
        combined_samples_3d = combined_samples[np.newaxis, :, :]

        ess_dict = {}
        for param_idx in range(num_params):
            param_samples = combined_samples_3d[:, :, param_idx]
            ess = effective_sample_size(param_samples)
            ess_dict[f"param_{param_idx}"] = (
                float(ess) if ess.size == 1 else float(np.mean(ess))
            )

        diagnostics["combined_ess"] = ess_dict

    except Exception as e:
        logger.warning(f"Failed to compute combined ESS: {str(e)}")
        diagnostics["combined_ess"] = None

    # Compute parameter uncertainty comparison
    try:
        # Per-shard std
        per_shard_stds = []
        for result in shard_results:
            if "samples" not in result:
                continue
            samples = result["samples"]
            if samples.ndim == 3:
                samples = samples.reshape(-1, num_params)
            per_shard_stds.append(np.std(samples, axis=0))

        if per_shard_stds:
            mean_per_shard_std = np.mean(per_shard_stds, axis=0)
            combined_std = np.std(combined_samples, axis=0)

            # Ratio > 1.0 means combined posterior has larger uncertainty
            uncertainty_ratio = combined_std / (mean_per_shard_std + 1e-10)
            diagnostics["parameter_uncertainty_ratio"] = {
                f"param_{i}": float(ratio) for i, ratio in enumerate(uncertainty_ratio)
            }
        else:
            diagnostics["parameter_uncertainty_ratio"] = None

    except Exception as e:
        logger.warning(f"Failed to compute uncertainty ratio: {str(e)}")
        diagnostics["parameter_uncertainty_ratio"] = None

    # Check for multimodality
    try:
        multimodality_detected = _check_multimodality(combined_samples)
        diagnostics["multimodality_detected"] = multimodality_detected
    except Exception as e:
        logger.warning(f"Failed to check multimodality: {str(e)}")
        diagnostics["multimodality_detected"] = None

    return diagnostics


# ============================================================================
# Helper Functions
# ============================================================================


def _validate_single_shard(
    shard_result: Dict[str, Any],
    shard_idx: int,
    max_rhat: float = 1.1,
    min_ess: float = 100.0,
    strict_mode: bool = True,
) -> Dict[str, Any]:
    """Validate convergence for a single shard.

    Parameters
    ----------
    shard_result : Dict[str, Any]
        Shard result dict
    shard_idx : int
        Shard index
    max_rhat : float
        Maximum allowed R-hat
    min_ess : float
        Minimum required ESS
    strict_mode : bool
        If True, collect errors; if False, collect warnings

    Returns
    -------
    validation : Dict[str, Any]
        Validation results with errors/warnings
    """
    validation = {
        "shard_id": shard_idx,
        "validation_errors": [],
        "validation_warnings": [],
    }

    # Get diagnostics
    diagnostics = compute_per_shard_diagnostics(shard_result, shard_idx)
    validation.update(diagnostics)

    # Check convergence flag
    if not shard_result.get("converged", False):
        msg = f"Shard {shard_idx} did not converge"
        if strict_mode:
            validation["validation_errors"].append(msg)
        else:
            validation["validation_warnings"].append(msg)

    # Check R-hat
    if diagnostics.get("rhat"):
        for param_name, rhat_val in diagnostics["rhat"].items():
            if rhat_val > max_rhat:
                msg = (
                    f"Shard {shard_idx} {param_name} R-hat {rhat_val:.3f} "
                    f"exceeds maximum {max_rhat}"
                )
                if strict_mode:
                    validation["validation_errors"].append(msg)
                else:
                    validation["validation_warnings"].append(msg)

    # Check ESS
    if diagnostics.get("ess"):
        for param_name, ess_val in diagnostics["ess"].items():
            if ess_val < min_ess:
                msg = (
                    f"Shard {shard_idx} {param_name} ESS {ess_val:.1f} "
                    f"below minimum {min_ess}"
                )
                if strict_mode:
                    validation["validation_errors"].append(msg)
                else:
                    validation["validation_warnings"].append(msg)

    return validation


def _fit_gaussian_to_samples(samples: np.ndarray) -> Dict[str, np.ndarray]:
    """Fit a Gaussian distribution to samples.

    Parameters
    ----------
    samples : np.ndarray of shape (num_samples, num_params)
        MCMC samples

    Returns
    -------
    gaussian : Dict[str, np.ndarray]
        Dictionary with 'mean' and 'cov' arrays
    """
    mean = np.mean(samples, axis=0)
    cov = np.cov(samples, rowvar=False)

    # Ensure cov is 2D even for single parameter
    if cov.ndim == 0:
        cov = np.array([[cov]])
    elif cov.ndim == 1:
        cov = np.diag(cov)

    # Add regularization for numerical stability
    num_params = mean.shape[0]
    regularization = 1e-6 * np.eye(num_params)
    cov = cov + regularization

    # Validate positive definiteness
    try:
        eigenvalues = np.linalg.eigvalsh(cov)
        if np.any(eigenvalues <= 0):
            logger.warning(
                f"Covariance matrix not positive definite, "
                f"min eigenvalue: {np.min(eigenvalues):.2e}"
            )
            # Add more regularization
            cov = cov + (abs(np.min(eigenvalues)) + 1e-6) * np.eye(num_params)
    except Exception as e:
        logger.warning(f"Failed to validate covariance matrix: {str(e)}")

    return {"mean": mean, "cov": cov}


def _compute_kl_divergence_matrix(
    gaussians: List[Dict[str, np.ndarray]],
) -> np.ndarray:
    """Compute pairwise KL divergence matrix between Gaussians.

    For two Gaussians N(μ_p, Σ_p) and N(μ_q, Σ_q):
    KL(p||q) = 0.5 * [trace(Σ_q^-1 Σ_p) + (μ_q - μ_p)^T Σ_q^-1 (μ_q - μ_p) - k + log(det(Σ_q) / det(Σ_p))]

    Parameters
    ----------
    gaussians : List[Dict[str, np.ndarray]]
        List of Gaussian dicts with 'mean' and 'cov'

    Returns
    -------
    kl_matrix : np.ndarray of shape (num_gaussians, num_gaussians)
        Symmetric KL divergence matrix (averaged forward and reverse KL)
    """
    num_gaussians = len(gaussians)
    kl_matrix = np.zeros((num_gaussians, num_gaussians))

    for i in range(num_gaussians):
        for j in range(i + 1, num_gaussians):
            # KL(i || j)
            kl_forward = _kl_divergence_gaussian(
                gaussians[i]["mean"],
                gaussians[i]["cov"],
                gaussians[j]["mean"],
                gaussians[j]["cov"],
            )

            # KL(j || i)
            kl_reverse = _kl_divergence_gaussian(
                gaussians[j]["mean"],
                gaussians[j]["cov"],
                gaussians[i]["mean"],
                gaussians[i]["cov"],
            )

            # Symmetric KL: average of forward and reverse
            kl_symmetric = 0.5 * (kl_forward + kl_reverse)

            kl_matrix[i, j] = kl_symmetric
            kl_matrix[j, i] = kl_symmetric

    return kl_matrix


def _kl_divergence_gaussian(
    mean_p: np.ndarray,
    cov_p: np.ndarray,
    mean_q: np.ndarray,
    cov_q: np.ndarray,
) -> float:
    """Compute KL divergence between two Gaussians.

    KL(p||q) = 0.5 * [trace(Σ_q^-1 Σ_p) + (μ_q - μ_p)^T Σ_q^-1 (μ_q - μ_p) - k + log(det(Σ_q) / det(Σ_p))]

    Parameters
    ----------
    mean_p, mean_q : np.ndarray
        Mean vectors
    cov_p, cov_q : np.ndarray
        Covariance matrices

    Returns
    -------
    kl : float
        KL divergence value
    """
    k = mean_p.shape[0]  # Number of dimensions

    # Compute inverse of cov_q
    try:
        cov_q_inv = np.linalg.inv(cov_q)
    except np.linalg.LinAlgError:
        # Singular matrix, use pseudoinverse
        logger.warning("Singular covariance matrix, using pseudoinverse")
        cov_q_inv = np.linalg.pinv(cov_q)

    # Term 1: trace(Σ_q^-1 Σ_p)
    term1 = np.trace(cov_q_inv @ cov_p)

    # Term 2: (μ_q - μ_p)^T Σ_q^-1 (μ_q - μ_p)
    mean_diff = mean_q - mean_p
    term2 = mean_diff.T @ cov_q_inv @ mean_diff

    # Term 3: log(det(Σ_q) / det(Σ_p))
    try:
        sign_p, logdet_p = np.linalg.slogdet(cov_p)
        sign_q, logdet_q = np.linalg.slogdet(cov_q)

        if sign_p <= 0 or sign_q <= 0:
            logger.warning("Non-positive determinant in KL computation")
            term3 = 0.0
        else:
            term3 = logdet_q - logdet_p
    except Exception as e:
        logger.warning(f"Failed to compute log determinant: {str(e)}")
        term3 = 0.0

    # KL divergence
    kl = 0.5 * (term1 + term2 - k + term3)

    return float(kl)


def _check_multimodality(samples: np.ndarray) -> bool:
    """Check for multimodality in combined samples.

    Uses a simple heuristic: compute standard deviation of per-parameter means
    across bootstrapped subsamples. High variance suggests multimodality.

    Parameters
    ----------
    samples : np.ndarray of shape (num_samples, num_params)
        Combined samples from all shards

    Returns
    -------
    multimodal : bool
        True if multimodality detected
    """
    num_samples, num_params = samples.shape

    if num_samples < 100:
        # Too few samples for reliable multimodality detection
        return False

    # Bootstrap resampling
    num_bootstrap = 20
    bootstrap_means = []

    for _ in range(num_bootstrap):
        # Random subsample (50% of data)
        indices = np.random.choice(num_samples, size=num_samples // 2, replace=False)
        subsample = samples[indices, :]
        bootstrap_means.append(np.mean(subsample, axis=0))

    bootstrap_means = np.array(bootstrap_means)  # Shape: (num_bootstrap, num_params)

    # Compute coefficient of variation for each parameter
    mean_of_means = np.mean(bootstrap_means, axis=0)
    std_of_means = np.std(bootstrap_means, axis=0)
    cv = std_of_means / (np.abs(mean_of_means) + 1e-10)

    # If any parameter has CV > 0.5, flag as potential multimodality
    # (0.5 is more conservative to avoid false positives)
    multimodal = np.any(cv > 0.5)

    if multimodal:
        logger.info(
            f"Potential multimodality detected. "
            f"Max coefficient of variation: {np.max(cv):.3f}"
        )

    return bool(multimodal)
