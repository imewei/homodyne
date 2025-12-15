"""Convergence diagnostics for CMC analysis.

This module provides functions for computing MCMC convergence diagnostics
including R-hat, effective sample size (ESS), and divergence checks.
"""

from __future__ import annotations

from typing import Any

import arviz as az
import numpy as np

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)

# Default convergence thresholds
DEFAULT_MAX_RHAT = 1.05
DEFAULT_MIN_ESS = 400
DEFAULT_MAX_DIVERGENCE_RATE = 0.05


def compute_r_hat(
    samples: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute R-hat (Gelman-Rubin diagnostic) for each parameter.

    R-hat measures chain convergence by comparing within-chain and
    between-chain variance. Values close to 1.0 indicate convergence.

    Parameters
    ----------
    samples : dict[str, np.ndarray]
        Parameter samples, {name: (n_chains, n_samples)}.

    Returns
    -------
    dict[str, float]
        R-hat value for each parameter.
    """
    r_hat_dict: dict[str, float] = {}

    for name, arr in samples.items():
        if arr.ndim != 2:
            logger.warning(f"Skipping R-hat for {name}: expected 2D, got {arr.ndim}D")
            continue

        n_chains, n_samples = arr.shape

        if n_chains < 2:
            # Cannot compute R-hat with single chain
            r_hat_dict[name] = np.nan
            continue

        # Between-chain variance
        chain_means = np.mean(arr, axis=1)
        B = n_samples * np.var(chain_means, ddof=1)

        # Within-chain variance
        chain_vars = np.var(arr, axis=1, ddof=1)
        W = np.mean(chain_vars)

        # Pooled variance estimate
        var_plus = ((n_samples - 1) * W + B) / n_samples

        # R-hat
        if W > 0:
            r_hat = np.sqrt(var_plus / W)
        else:
            r_hat = np.nan

        r_hat_dict[name] = float(r_hat)

    return r_hat_dict


def compute_ess(
    samples: dict[str, np.ndarray],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute effective sample size (bulk and tail) for each parameter.

    ESS measures the number of independent samples accounting for
    autocorrelation. Higher is better.

    Parameters
    ----------
    samples : dict[str, np.ndarray]
        Parameter samples, {name: (n_chains, n_samples)}.

    Returns
    -------
    tuple[dict[str, float], dict[str, float]]
        (ess_bulk, ess_tail) dictionaries.
    """
    ess_bulk_dict: dict[str, float] = {}
    ess_tail_dict: dict[str, float] = {}

    # Create ArviZ InferenceData for ESS computation
    try:
        idata = az.from_dict(posterior=samples)

        # Compute ESS using ArviZ
        ess_bulk = az.ess(idata, method="bulk")
        ess_tail = az.ess(idata, method="tail")

        # Extract values
        for name in samples.keys():
            if hasattr(ess_bulk, name):
                ess_bulk_dict[name] = float(getattr(ess_bulk, name).values)
            else:
                ess_bulk_dict[name] = np.nan

            if hasattr(ess_tail, name):
                ess_tail_dict[name] = float(getattr(ess_tail, name).values)
            else:
                ess_tail_dict[name] = np.nan

    except Exception as e:
        logger.warning(f"ArviZ ESS computation failed: {e}, using simple estimate")

        # Fallback: simple ESS estimate
        for name, arr in samples.items():
            n_total = arr.size
            # Very rough estimate: assume moderate autocorrelation
            ess_bulk_dict[name] = float(n_total / 10)
            ess_tail_dict[name] = float(n_total / 10)

    return ess_bulk_dict, ess_tail_dict


def count_divergences(
    extra_fields: dict[str, Any],
) -> int:
    """Count total divergent transitions.

    Parameters
    ----------
    extra_fields : dict[str, Any]
        Extra fields from MCMC sampler.

    Returns
    -------
    int
        Total number of divergent transitions.
    """
    if "diverging" in extra_fields:
        return int(np.sum(extra_fields["diverging"]))
    return 0


def check_convergence(
    r_hat: dict[str, float],
    ess_bulk: dict[str, float],
    divergences: int,
    n_samples: int,
    n_chains: int,
    max_rhat: float = DEFAULT_MAX_RHAT,
    min_ess: float = DEFAULT_MIN_ESS,
    max_divergence_rate: float = DEFAULT_MAX_DIVERGENCE_RATE,
) -> tuple[str, list[str]]:
    """Check convergence and generate warnings.

    Parameters
    ----------
    r_hat : dict[str, float]
        Per-parameter R-hat values.
    ess_bulk : dict[str, float]
        Per-parameter bulk ESS values.
    divergences : int
        Number of divergent transitions.
    n_samples : int
        Samples per chain.
    n_chains : int
        Number of chains.
    max_rhat : float
        Maximum acceptable R-hat.
    min_ess : float
        Minimum acceptable ESS.
    max_divergence_rate : float
        Maximum acceptable divergence rate.

    Returns
    -------
    tuple[str, list[str]]
        (status, warnings) where status is "converged" | "divergences" | "not_converged".
    """
    warnings: list[str] = []

    # Check R-hat
    max_r_hat_value = max(
        (v for v in r_hat.values() if not np.isnan(v)),
        default=1.0,
    )
    if max_r_hat_value > max_rhat:
        bad_params = [k for k, v in r_hat.items() if v > max_rhat]
        warnings.append(
            f"R-hat > {max_rhat} for parameters: {bad_params} (max={max_r_hat_value:.3f})"
        )

    # Check ESS
    min_ess_value = min(
        (v for v in ess_bulk.values() if not np.isnan(v)),
        default=0.0,
    )
    if min_ess_value < min_ess:
        bad_params = [k for k, v in ess_bulk.items() if v < min_ess]
        warnings.append(
            f"ESS < {min_ess} for parameters: {bad_params} (min={min_ess_value:.0f})"
        )

    # Check divergences
    total_transitions = n_samples * n_chains
    divergence_rate = divergences / total_transitions if total_transitions > 0 else 0

    if divergence_rate > max_divergence_rate:
        warnings.append(
            f"Divergence rate {divergence_rate:.1%} exceeds {max_divergence_rate:.1%} "
            f"({divergences}/{total_transitions} transitions)"
        )

    # Determine status
    if divergences > 0 and divergence_rate > max_divergence_rate:
        status = "divergences"
    elif warnings:
        status = "not_converged"
    else:
        status = "converged"

    return status, warnings


def create_diagnostics_dict(
    r_hat: dict[str, float],
    ess_bulk: dict[str, float],
    ess_tail: dict[str, float],
    divergences: int,
    convergence_status: str,
    warnings: list[str],
    n_chains: int,
    n_warmup: int,
    n_samples: int,
    warmup_time: float,
    sampling_time: float,
) -> dict[str, Any]:
    """Create diagnostics dictionary for JSON output.

    Parameters
    ----------
    r_hat : dict[str, float]
        Per-parameter R-hat.
    ess_bulk : dict[str, float]
        Per-parameter bulk ESS.
    ess_tail : dict[str, float]
        Per-parameter tail ESS.
    divergences : int
        Number of divergences.
    convergence_status : str
        Convergence status.
    warnings : list[str]
        Warning messages.
    n_chains : int
        Number of chains.
    n_warmup : int
        Warmup samples.
    n_samples : int
        Posterior samples.
    warmup_time : float
        Warmup time in seconds.
    sampling_time : float
        Sampling time in seconds.

    Returns
    -------
    dict[str, Any]
        Diagnostics dictionary.
    """
    # Compute summary statistics
    r_hat_values = [v for v in r_hat.values() if not np.isnan(v)]
    ess_values = [v for v in ess_bulk.values() if not np.isnan(v)]

    total_samples = n_chains * n_samples
    divergence_rate = divergences / total_samples if total_samples > 0 else 0

    return {
        "convergence_status": convergence_status,
        "total_divergences": divergences,
        "divergence_rate": divergence_rate,
        "max_r_hat": max(r_hat_values) if r_hat_values else np.nan,
        "min_ess_bulk": min(ess_values) if ess_values else np.nan,
        "min_ess_tail": min(ess_tail.values()) if ess_tail else np.nan,
        "all_r_hat_ok": all(v <= DEFAULT_MAX_RHAT for v in r_hat_values),
        "all_ess_ok": all(v >= DEFAULT_MIN_ESS for v in ess_values),
        "warnings": warnings,
        "sampling_config": {
            "n_chains": n_chains,
            "n_warmup": n_warmup,
            "n_samples": n_samples,
        },
        "timing": {
            "warmup_seconds": warmup_time,
            "sampling_seconds": sampling_time,
            "total_seconds": warmup_time + sampling_time,
        },
        "per_parameter": {
            name: {
                "r_hat": r_hat.get(name, np.nan),
                "ess_bulk": ess_bulk.get(name, np.nan),
                "ess_tail": ess_tail.get(name, np.nan),
            }
            for name in r_hat.keys()
        },
    }


def summarize_diagnostics(
    r_hat: dict[str, float],
    ess_bulk: dict[str, float],
    divergences: int,
    n_samples: int,
    n_chains: int,
) -> str:
    """Create human-readable diagnostics summary.

    Parameters
    ----------
    r_hat : dict[str, float]
        R-hat values.
    ess_bulk : dict[str, float]
        ESS values.
    divergences : int
        Divergence count.
    n_samples : int
        Samples per chain.
    n_chains : int
        Number of chains.

    Returns
    -------
    str
        Summary string.
    """
    r_hat_values = [v for v in r_hat.values() if not np.isnan(v)]
    ess_values = [v for v in ess_bulk.values() if not np.isnan(v)]

    max_rhat = max(r_hat_values) if r_hat_values else np.nan
    min_ess = min(ess_values) if ess_values else np.nan

    total = n_samples * n_chains
    div_rate = divergences / total if total > 0 else 0

    return (
        f"Diagnostics: R-hat(max)={max_rhat:.3f}, "
        f"ESS(min)={min_ess:.0f}, "
        f"divergences={divergences} ({div_rate:.1%})"
    )


def log_analysis_summary(
    convergence_status: str,
    r_hat: dict[str, float],
    ess_bulk: dict[str, float],
    divergences: int,
    n_samples: int,
    n_chains: int,
    n_shards: int,
    shards_succeeded: int,
    execution_time: float,
) -> None:
    """Log a comprehensive summary at the end of CMC analysis.

    Parameters
    ----------
    convergence_status : str
        Final convergence status.
    r_hat : dict[str, float]
        Per-parameter R-hat values.
    ess_bulk : dict[str, float]
        Per-parameter bulk ESS values.
    divergences : int
        Total divergent transitions.
    n_samples : int
        Samples per chain.
    n_chains : int
        Number of chains.
    n_shards : int
        Total number of shards.
    shards_succeeded : int
        Number of successful shards.
    execution_time : float
        Total execution time in seconds.
    """
    r_hat_values = [v for v in r_hat.values() if not np.isnan(v)]
    ess_values = [v for v in ess_bulk.values() if not np.isnan(v)]

    max_rhat = max(r_hat_values) if r_hat_values else np.nan
    min_ess = min(ess_values) if ess_values else np.nan

    total_transitions = n_samples * n_chains
    div_rate = divergences / total_transitions if total_transitions > 0 else 0
    success_rate = shards_succeeded / n_shards if n_shards > 0 else 0

    # Visual separator for easy identification in logs
    logger.info("=" * 60)
    logger.info("CMC ANALYSIS SUMMARY")
    logger.info("=" * 60)

    # Status with clear indicator
    if convergence_status == "converged":
        logger.info(f"✓ Status: CONVERGED")
    else:
        logger.warning(f"✗ Status: {convergence_status.upper()}")

    # Key metrics
    logger.info(f"  Shards: {shards_succeeded}/{n_shards} ({success_rate:.0%} success)")
    logger.info(f"  Runtime: {execution_time:.1f}s ({execution_time/60:.1f} min)")
    logger.info(f"  R-hat (max): {max_rhat:.4f} {'✓' if max_rhat <= 1.05 else '✗'}")
    logger.info(f"  ESS (min): {min_ess:.0f} {'✓' if min_ess >= 100 else '✗'}")
    logger.info(f"  Divergences: {divergences} ({div_rate:.1%})")

    # Recommendations if there are issues
    recommendations = get_convergence_recommendations(
        max_rhat, min_ess, divergences, n_samples, n_chains
    )
    if recommendations:
        logger.info("-" * 40)
        logger.info("RECOMMENDATIONS:")
        for rec in recommendations:
            logger.info(f"  → {rec}")

    logger.info("=" * 60)


def get_convergence_recommendations(
    max_rhat: float,
    min_ess: float,
    divergences: int,
    n_samples: int,
    n_chains: int,
) -> list[str]:
    """Generate specific recommendations for convergence issues.

    Parameters
    ----------
    max_rhat : float
        Maximum R-hat value across parameters.
    min_ess : float
        Minimum bulk ESS across parameters.
    divergences : int
        Number of divergent transitions.
    n_samples : int
        Samples per chain.
    n_chains : int
        Number of chains.

    Returns
    -------
    list[str]
        List of recommendation strings.
    """
    recommendations: list[str] = []

    total_transitions = n_samples * n_chains
    div_rate = divergences / total_transitions if total_transitions > 0 else 0

    # R-hat recommendations
    if np.isfinite(max_rhat) and max_rhat > 1.1:
        recommendations.append(
            f"HIGH R-HAT ({max_rhat:.3f}): Chains have not mixed. "
            "Try: increase num_warmup (currently {n_warmup}), "
            "or use more chains (currently {n_chains})."
        )
    elif np.isfinite(max_rhat) and max_rhat > 1.05:
        recommendations.append(
            f"MARGINAL R-HAT ({max_rhat:.3f}): Consider increasing num_samples "
            f"or num_warmup for better convergence."
        )

    # ESS recommendations
    if np.isfinite(min_ess) and min_ess < 100:
        recommendations.append(
            f"LOW ESS ({min_ess:.0f}): High autocorrelation in samples. "
            f"Try: increase num_samples (currently {n_samples}) to at least {int(100 * n_samples / max(min_ess, 1))}."
        )
    elif np.isfinite(min_ess) and min_ess < 400:
        recommendations.append(
            f"MODERATE ESS ({min_ess:.0f}): Consider increasing num_samples "
            f"for more reliable uncertainty estimates."
        )

    # Divergence recommendations
    if div_rate > 0.10:
        recommendations.append(
            f"HIGH DIVERGENCES ({div_rate:.1%}): Model geometry issues. "
            "Try: reduce max_points_per_shard, increase target_accept_prob to 0.95, "
            "or check for data outliers."
        )
    elif div_rate > 0.01:
        recommendations.append(
            f"MODERATE DIVERGENCES ({div_rate:.1%}): Some geometry issues. "
            "Consider increasing target_accept_prob to 0.90."
        )

    # General efficiency recommendations
    if not recommendations and np.isfinite(max_rhat) and max_rhat <= 1.05:
        # Everything looks good - no recommendations needed
        pass

    return recommendations
