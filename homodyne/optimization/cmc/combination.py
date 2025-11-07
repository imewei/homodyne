"""Subposterior Combination for Consensus Monte Carlo

This module implements methods to combine subposteriors from parallel MCMC shards
into a single posterior distribution.

Two combination methods are provided:
1. Weighted Gaussian Product (Primary): Fast, closed-form solution assuming
   Gaussian posterior approximation. Optimal for well-behaved posteriors.
2. Simple Averaging (Fallback): Concatenate and resample. More robust to
   non-Gaussian posteriors.

The module supports automatic fallback from weighted to averaging if the
weighted method fails (e.g., due to ill-conditioned covariance matrices).

References
----------
Scott, S. L., et al. (2016). "Bayes and big data: the consensus Monte Carlo
algorithm." International Journal of Management Science and Engineering
Management, 11(2), 78-88.
https://arxiv.org/abs/1411.7435

Examples
--------
Basic usage with automatic method selection:

>>> shard_results = [
...     {'samples': np.random.randn(1000, 5), 'shard_id': 0},
...     {'samples': np.random.randn(1000, 5), 'shard_id': 1},
...     {'samples': np.random.randn(1000, 5), 'shard_id': 2},
... ]
>>> combined = combine_subposteriors(shard_results, method='weighted')
>>> combined['samples'].shape
(1000, 5)
>>> combined['method']
'weighted'

Fallback to averaging on failure:

>>> combined = combine_subposteriors(
...     shard_results,
...     method='weighted',
...     fallback_enabled=True
... )
# If weighted fails, automatically falls back to averaging
"""

import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def combine_subposteriors(
    shard_results: List[Dict[str, Any]],
    method: str = "weighted",
    fallback_enabled: bool = True,
) -> Dict[str, np.ndarray]:
    """Combine subposteriors from all shards into a single posterior.

    This function combines MCMC results from multiple data shards using either
    weighted Gaussian product (fast, assumes Gaussian posteriors) or simple
    averaging (robust, works for non-Gaussian posteriors).

    Parameters
    ----------
    shard_results : list of dict
        Per-shard MCMC results. Each dict must contain:
        - 'samples' : np.ndarray, shape (num_samples, num_params)
            Posterior samples from this shard
        - Additional fields like 'shard_id', 'convergence' are preserved
          but not required for combination
    method : str, default='weighted'
        Combination method to use:
        - 'weighted' : Weighted Gaussian product (assumes Gaussian posterior)
        - 'average' : Simple averaging (concatenate and resample)
    fallback_enabled : bool, default=True
        If True and weighted method fails, automatically fall back to averaging.
        If False, raise exception on failure.

    Returns
    -------
    combined_posterior : dict
        Combined posterior containing:
        - 'samples' : np.ndarray, shape (num_samples, num_params)
            Combined posterior samples
        - 'mean' : np.ndarray, shape (num_params,)
            Posterior mean
        - 'cov' : np.ndarray, shape (num_params, num_params)
            Posterior covariance matrix
        - 'method' : str
            Method actually used ('weighted' or 'average')

    Raises
    ------
    ValueError
        If unknown method specified or if shard results are invalid
    RuntimeError
        If combination fails and fallback is disabled

    Examples
    --------
    Weighted Gaussian product combination:

    >>> shard_results = [
    ...     {'samples': np.random.randn(1000, 3)},
    ...     {'samples': np.random.randn(1000, 3)},
    ... ]
    >>> combined = combine_subposteriors(shard_results, method='weighted')
    >>> combined['samples'].shape
    (1000, 3)
    >>> combined['method']
    'weighted'

    Simple averaging with fallback disabled:

    >>> combined = combine_subposteriors(
    ...     shard_results,
    ...     method='average',
    ...     fallback_enabled=False
    ... )
    >>> combined['method']
    'average'

    Automatic fallback on weighted failure:

    >>> # If weighted fails, falls back to averaging
    >>> combined = combine_subposteriors(shard_results, method='weighted')
    """
    # Validate inputs
    _validate_shard_results(shard_results)

    # Handle single shard edge case
    if len(shard_results) == 1:
        logger.info("Single shard detected, returning shard samples directly")
        samples = shard_results[0]["samples"]
        return {
            "samples": samples,
            "mean": np.mean(samples, axis=0),
            "cov": np.cov(samples.T),
            "method": "single_shard",
        }

    # Try requested method with optional fallback
    try:
        if method == "weighted":
            return _weighted_gaussian_product(shard_results)
        elif method == "average":
            return _simple_averaging(shard_results)
        else:
            raise ValueError(
                f"Unknown combination method: {method}. Must be 'weighted' or 'average'"
            )
    except Exception as e:
        if fallback_enabled and method == "weighted":
            logger.warning(
                f"Weighted combination failed: {e}. Falling back to averaging."
            )
            return _simple_averaging(shard_results)
        else:
            raise


def _weighted_gaussian_product(shard_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine subposteriors using weighted Gaussian product.

    This implements the Scott et al. (2016) algorithm:
    1. Extract samples from each shard
    2. Fit Gaussian to each shard: N(μ_i, Σ_i)
    3. Compute precision matrices: Λ_i = Σ_i⁻¹
    4. Combined precision: Λ = ∑ᵢ Λ_i
    5. Combined covariance: Σ = Λ⁻¹
    6. Combined mean: μ = Σ · (∑ᵢ Λ_i μ_i)
    7. Sample from combined Gaussian: N(μ, Σ)

    Regularization (1e-6 * I) is added to covariance matrices for numerical
    stability to handle ill-conditioned matrices.

    Parameters
    ----------
    shard_results : list of dict
        Per-shard MCMC results containing 'samples' key

    Returns
    -------
    combined_result : dict
        Combined posterior with 'samples', 'mean', 'cov', 'method' keys

    Raises
    ------
    ValueError
        If covariance matrices are not positive definite
    np.linalg.LinAlgError
        If matrix inversion fails

    Notes
    -----
    This method assumes Gaussian posterior approximation and works best for
    well-behaved posteriors. For non-Gaussian or multi-modal posteriors,
    use simple averaging instead.

    References
    ----------
    Scott et al. (2016): https://arxiv.org/abs/1411.7435
    """
    # Extract samples from each shard
    shard_samples = [result["samples"] for result in shard_results]
    num_shards = len(shard_samples)
    num_samples = shard_samples[0].shape[0]
    num_params = shard_samples[0].shape[1]

    logger.info(
        f"Weighted Gaussian product: combining {num_shards} shards, "
        f"{num_samples} samples, {num_params} parameters"
    )

    # Fit Gaussian to each shard
    means = [np.mean(samples, axis=0) for samples in shard_samples]
    covs = [np.cov(samples.T) for samples in shard_samples]

    # Add regularization for numerical stability
    regularization = 1e-6
    identity = np.eye(num_params)

    # Validate and regularize covariance matrices
    for i, cov in enumerate(covs):
        # Check if positive definite (all eigenvalues > 0)
        eigenvalues = np.linalg.eigvalsh(cov)
        if np.any(eigenvalues <= 0):
            logger.warning(
                f"Shard {i} covariance matrix is not positive definite. "
                f"Min eigenvalue: {eigenvalues.min():.2e}. "
                f"Adding regularization."
            )
        # Add regularization
        covs[i] = cov + regularization * identity

    # Compute precision matrices (inverse covariance)
    try:
        precisions = [np.linalg.inv(cov) for cov in covs]
    except np.linalg.LinAlgError as e:
        raise ValueError(
            f"Failed to compute precision matrices: {e}. "
            f"Covariance matrices may be singular."
        ) from e

    # Combined precision (sum of precisions)
    combined_precision = np.sum(precisions, axis=0)

    # Combined covariance (inverse of combined precision)
    try:
        combined_cov = np.linalg.inv(combined_precision)
    except np.linalg.LinAlgError as e:
        raise ValueError(
            f"Failed to compute combined covariance: {e}. "
            f"Combined precision matrix may be singular."
        ) from e

    # Combined mean (weighted average using precisions)
    weighted_means = [prec @ mean for prec, mean in zip(precisions, means)]
    combined_mean = combined_cov @ np.sum(weighted_means, axis=0)

    # Sample from combined Gaussian
    try:
        combined_samples = np.random.multivariate_normal(
            combined_mean, combined_cov, size=num_samples
        )
    except ValueError as e:
        raise ValueError(
            f"Failed to sample from combined Gaussian: {e}. "
            f"Covariance matrix may not be positive definite."
        ) from e

    logger.info("Weighted Gaussian product completed successfully")

    return {
        "samples": combined_samples,
        "mean": combined_mean,
        "cov": combined_cov,
        "method": "weighted",
    }


def _simple_averaging(shard_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine subposteriors using simple averaging.

    This method concatenates all samples from all shards and resamples to
    maintain a uniform sample count. It is more robust to non-Gaussian and
    multi-modal posteriors than the weighted Gaussian product.

    Algorithm:
    1. Concatenate all samples from all shards
    2. Resample to uniform weight (original sample count per shard)
    3. Compute mean and covariance from resampled distribution

    Parameters
    ----------
    shard_results : list of dict
        Per-shard MCMC results containing 'samples' key

    Returns
    -------
    combined_result : dict
        Combined posterior with 'samples', 'mean', 'cov', 'method' keys

    Notes
    -----
    This method works well for:
    - Non-Gaussian posteriors
    - Multi-modal posteriors
    - Posteriors with heavy tails
    - When weighted method fails

    The resampling ensures equal weight for all shards and maintains the
    original sample count from a single shard.
    """
    # Extract samples
    shard_samples = [result["samples"] for result in shard_results]
    num_shards = len(shard_samples)
    num_samples_per_shard = shard_samples[0].shape[0]
    num_params = shard_samples[0].shape[1]

    logger.info(
        f"Simple averaging: combining {num_shards} shards, "
        f"{num_samples_per_shard} samples per shard, {num_params} parameters"
    )

    # Concatenate all samples
    all_samples = np.concatenate(shard_samples, axis=0)

    # Resample to uniform weight (maintain original sample count)
    total_samples = len(all_samples)
    indices = np.random.choice(total_samples, size=num_samples_per_shard, replace=False)
    combined_samples = all_samples[indices]

    # Compute statistics
    combined_mean = np.mean(combined_samples, axis=0)
    combined_cov = np.cov(combined_samples.T)

    logger.info("Simple averaging completed successfully")

    return {
        "samples": combined_samples,
        "mean": combined_mean,
        "cov": combined_cov,
        "method": "average",
    }


def _validate_shard_results(shard_results: List[Dict[str, Any]]) -> None:
    """Validate shard results before combination.

    Parameters
    ----------
    shard_results : list of dict
        Per-shard MCMC results to validate

    Raises
    ------
    ValueError
        If shard results are invalid (missing samples, inconsistent shapes, etc.)
    """
    if not shard_results:
        raise ValueError("shard_results is empty")

    if not isinstance(shard_results, list):
        raise ValueError(f"shard_results must be a list, got {type(shard_results)}")

    # Check all shards have 'samples' key
    for i, result in enumerate(shard_results):
        if not isinstance(result, dict):
            raise ValueError(f"Shard {i} is not a dict, got {type(result)}")
        if "samples" not in result:
            raise ValueError(
                f"Shard {i} missing 'samples' key. "
                f"Available keys: {list(result.keys())}"
            )

    # Extract samples and validate shapes
    shard_samples = [result["samples"] for result in shard_results]

    # Check first shard
    first_samples = shard_samples[0]
    if not isinstance(first_samples, np.ndarray):
        raise ValueError(f"Shard 0 samples is not ndarray, got {type(first_samples)}")
    if first_samples.ndim != 2:
        raise ValueError(
            f"Shard 0 samples must be 2D (num_samples, num_params), "
            f"got shape {first_samples.shape}"
        )

    expected_shape = first_samples.shape
    num_samples_expected, num_params_expected = expected_shape

    # Validate all shards have consistent shapes
    for i, samples in enumerate(shard_samples[1:], start=1):
        if not isinstance(samples, np.ndarray):
            raise ValueError(f"Shard {i} samples is not ndarray, got {type(samples)}")
        if samples.ndim != 2:
            raise ValueError(f"Shard {i} samples must be 2D, got shape {samples.shape}")
        if samples.shape[1] != num_params_expected:
            raise ValueError(
                f"Shard {i} has {samples.shape[1]} parameters, "
                f"expected {num_params_expected}"
            )
        if samples.shape[0] != num_samples_expected:
            raise ValueError(
                f"Shard {i} has {samples.shape[0]} samples, "
                f"expected {num_samples_expected}"
            )

    # Check for NaN/Inf in samples
    for i, samples in enumerate(shard_samples):
        if np.any(np.isnan(samples)):
            raise ValueError(f"Shard {i} contains NaN values")
        if np.any(np.isinf(samples)):
            raise ValueError(f"Shard {i} contains Inf values")

    logger.debug(
        f"Validated {len(shard_results)} shards: "
        f"{num_samples_expected} samples × {num_params_expected} parameters"
    )
