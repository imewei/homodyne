"""Convergence diagnostics for CMC analysis.

This module provides functions for computing MCMC convergence diagnostics
including R-hat, effective sample size (ESS), and divergence checks.
"""

from __future__ import annotations

from collections.abc import Callable
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
    num_shards: int = 1,
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
    num_shards : int
        Number of shards (for CMC). Divergences are summed across shards,
        so total transitions = num_shards × n_chains × n_samples.

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
        # T046: Log R-hat warnings for poor convergence
        warning_msg = f"R-hat > {max_rhat} for parameters: {bad_params} (max={max_r_hat_value:.3f})"
        logger.warning(warning_msg)
        warnings.append(warning_msg)

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
    # For CMC, divergences are summed across all shards, so total transitions
    # must account for num_shards to get the correct rate
    total_transitions = num_shards * n_samples * n_chains
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
    num_shards: int = 1,
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
    num_shards : int
        Number of shards (for CMC).

    Returns
    -------
    str
        Summary string.
    """
    r_hat_values = [v for v in r_hat.values() if not np.isnan(v)]
    ess_values = [v for v in ess_bulk.values() if not np.isnan(v)]

    max_rhat = max(r_hat_values) if r_hat_values else np.nan
    min_ess = min(ess_values) if ess_values else np.nan

    total = num_shards * n_samples * n_chains
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

    # For CMC, total transitions = n_shards × n_samples × n_chains
    total_transitions = n_shards * n_samples * n_chains
    div_rate = divergences / total_transitions if total_transitions > 0 else 0
    success_rate = shards_succeeded / n_shards if n_shards > 0 else 0

    # Visual separator for easy identification in logs
    logger.info("=" * 60)
    logger.info("CMC ANALYSIS SUMMARY")
    logger.info("=" * 60)

    # Status with clear indicator
    if convergence_status == "converged":
        logger.info("✓ Status: CONVERGED")
    else:
        logger.warning(f"✗ Status: {convergence_status.upper()}")

    # Key metrics
    logger.info(f"  Shards: {shards_succeeded}/{n_shards} ({success_rate:.0%} success)")
    logger.info(f"  Runtime: {execution_time:.1f}s ({execution_time / 60:.1f} min)")
    logger.info(f"  R-hat (max): {max_rhat:.4f} {'✓' if max_rhat <= 1.05 else '✗'}")
    logger.info(f"  ESS (min): {min_ess:.0f} {'✓' if min_ess >= 100 else '✗'}")
    logger.info(f"  Divergences: {divergences} ({div_rate:.1%})")

    # Recommendations if there are issues
    recommendations = get_convergence_recommendations(
        max_rhat, min_ess, divergences, n_samples, n_chains, n_shards
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
    num_shards: int = 1,
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
    num_shards : int
        Number of shards (for CMC).

    Returns
    -------
    list[str]
        List of recommendation strings.
    """
    recommendations: list[str] = []

    total_transitions = num_shards * n_samples * n_chains
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


# =============================================================================
# PRECISION DIAGNOSTICS (Jan 2026)
# =============================================================================


def compute_posterior_contraction(
    posterior_std: float,
    prior_std: float,
) -> float:
    """Compute Posterior Contraction Ratio (PCR).

    PCR measures how much the data informed the posterior relative to the prior.
    PCR = 1 - (posterior_std / prior_std)

    Interpretation:
    - PCR ≈ 0: Posterior ≈ prior (data didn't constrain the parameter)
    - PCR ≈ 0.5: Posterior half as wide as prior (moderate constraint)
    - PCR ≈ 0.9: Posterior 10% as wide as prior (strong constraint)
    - PCR < 0: Posterior wider than prior (model misspecification or numerical issues)

    Parameters
    ----------
    posterior_std : float
        Standard deviation of the posterior distribution.
    prior_std : float
        Standard deviation of the prior distribution.

    Returns
    -------
    float
        Posterior contraction ratio, typically in [0, 1].
    """
    if prior_std <= 0 or not np.isfinite(prior_std):
        return np.nan
    if posterior_std <= 0 or not np.isfinite(posterior_std):
        return np.nan

    return 1.0 - (posterior_std / prior_std)


def compute_nlsq_comparison_metrics(
    cmc_mean: float,
    cmc_std: float,
    nlsq_value: float,
    nlsq_std: float | None = None,
) -> dict[str, float]:
    """Compute metrics comparing CMC posterior to NLSQ point estimate.

    Parameters
    ----------
    cmc_mean : float
        CMC posterior mean.
    cmc_std : float
        CMC posterior standard deviation.
    nlsq_value : float
        NLSQ point estimate.
    nlsq_std : float | None
        NLSQ standard error. If None, only CMC-based metrics computed.

    Returns
    -------
    dict[str, float]
        Dictionary with comparison metrics:
        - z_score: abs(CMC_mean - NLSQ) / CMC_std (should be < 2 for consistency)
        - uncertainty_ratio: CMC_std / NLSQ_std (should be < 5x ideally)
        - relative_diff: (CMC_mean - NLSQ) / abs(NLSQ) (percent difference)
        - coverage: Whether NLSQ falls within CMC 95% CI
    """
    metrics = {}

    # Z-score: How many CMC standard deviations away is NLSQ?
    if cmc_std > 0 and np.isfinite(cmc_std):
        z_score = abs(cmc_mean - nlsq_value) / cmc_std
        metrics["z_score"] = z_score
        # Coverage: Does 95% CI contain NLSQ?
        metrics["coverage_95"] = float(z_score < 1.96)
    else:
        metrics["z_score"] = np.nan
        metrics["coverage_95"] = np.nan

    # Relative difference (percent)
    if nlsq_value != 0 and np.isfinite(nlsq_value):
        metrics["relative_diff"] = (cmc_mean - nlsq_value) / abs(nlsq_value)
    else:
        metrics["relative_diff"] = np.nan

    # Uncertainty ratio (if NLSQ std available)
    if nlsq_std is not None and nlsq_std > 0 and np.isfinite(nlsq_std):
        metrics["uncertainty_ratio"] = cmc_std / nlsq_std
    else:
        metrics["uncertainty_ratio"] = np.nan

    return metrics


def compute_precision_analysis(
    cmc_result: dict[str, dict],
    nlsq_result: dict[str, float] | None = None,
    nlsq_uncertainties: dict[str, float] | None = None,
    prior_stds: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute comprehensive precision analysis for all parameters.

    Parameters
    ----------
    cmc_result : dict[str, dict]
        CMC posterior statistics, keyed by parameter name.
        Each entry should have "mean" and "std" keys.
    nlsq_result : dict[str, float] | None
        NLSQ point estimates, keyed by parameter name.
    nlsq_uncertainties : dict[str, float] | None
        NLSQ standard errors, keyed by parameter name.
    prior_stds : dict[str, float] | None
        Prior standard deviations, keyed by parameter name.

    Returns
    -------
    dict[str, dict[str, float]]
        Precision metrics for each parameter.
    """
    analysis = {}

    for param_name, stats in cmc_result.items():
        # Skip non-physical parameters
        if param_name in ("sigma", "obs", "n_numerical_issues"):
            continue

        param_metrics = {
            "cmc_mean": stats.get("mean", np.nan),
            "cmc_std": stats.get("std", np.nan),
        }

        # Add posterior contraction if prior_std available
        if prior_stds and param_name in prior_stds:
            pcr = compute_posterior_contraction(
                stats.get("std", np.nan),
                prior_stds[param_name],
            )
            param_metrics["posterior_contraction"] = pcr
            param_metrics["prior_std"] = prior_stds[param_name]

        # Add NLSQ comparison if available
        if nlsq_result and param_name in nlsq_result:
            nlsq_val = nlsq_result[param_name]
            nlsq_std = (
                nlsq_uncertainties.get(param_name) if nlsq_uncertainties else None
            )

            comparison = compute_nlsq_comparison_metrics(
                cmc_mean=stats.get("mean", np.nan),
                cmc_std=stats.get("std", np.nan),
                nlsq_value=nlsq_val,
                nlsq_std=nlsq_std,
            )
            param_metrics.update(comparison)
            param_metrics["nlsq_value"] = nlsq_val
            if nlsq_std is not None:
                param_metrics["nlsq_std"] = nlsq_std

        analysis[param_name] = param_metrics

    return analysis


def log_precision_analysis(
    analysis: dict[str, dict[str, float]],
    log_fn: Callable[[str], None] | None = None,
    tolerance_pct: float = 20.0,
) -> str:
    """Log a comprehensive precision analysis report.

    Parameters
    ----------
    analysis : dict[str, dict[str, float]]
        Output from compute_precision_analysis().
    log_fn : callable | None
        Logging function. If None, uses module logger.
    tolerance_pct : float
        Percentage tolerance threshold for flagging parameters.
        Default 20% - parameters exceeding this are flagged.

    Returns
    -------
    str
        Formatted analysis report.
    """
    if log_fn is None:
        log_fn = logger.info

    lines = ["=" * 80, "CMC vs NLSQ PRECISION ANALYSIS", "=" * 80]

    # Summary statistics
    z_scores = [
        m.get("z_score", np.nan)
        for m in analysis.values()
        if np.isfinite(m.get("z_score", np.nan))
    ]
    rel_diffs = [
        abs(m.get("relative_diff", np.nan) * 100)
        for m in analysis.values()
        if np.isfinite(m.get("relative_diff", np.nan))
    ]
    unc_ratios = [
        m.get("uncertainty_ratio", np.nan)
        for m in analysis.values()
        if np.isfinite(m.get("uncertainty_ratio", np.nan))
    ]
    pcrs = [
        m.get("posterior_contraction", np.nan)
        for m in analysis.values()
        if np.isfinite(m.get("posterior_contraction", np.nan))
    ]

    # Summary section
    lines.append("SUMMARY:")
    if z_scores:
        max_z = max(z_scores)
        mean_z = np.mean(z_scores)
        lines.append(f"  Z-scores: max={max_z:.2f}, mean={mean_z:.2f}")
        high_z = sum(1 for z in z_scores if z > 2)
        very_high_z = sum(1 for z in z_scores if z > 3)
        if very_high_z > 0:
            lines.append(
                f"    ⚠️ CRITICAL: {very_high_z}/{len(z_scores)} params have z > 3 (severe disagreement)"
            )
        elif high_z > 0:
            lines.append(
                f"    ⚠️ WARNING: {high_z}/{len(z_scores)} params have z > 2 (significant disagreement)"
            )
        else:
            lines.append("    ✓ All params have z ≤ 2 (good agreement)")

    if rel_diffs:
        max_diff = max(rel_diffs)
        mean_diff = np.mean(rel_diffs)
        lines.append(f"  Percent differences: max={max_diff:.1f}%, mean={mean_diff:.1f}%")
        over_tolerance = sum(1 for d in rel_diffs if d > tolerance_pct)
        if over_tolerance > 0:
            lines.append(
                f"    ⚠️ {over_tolerance}/{len(rel_diffs)} params exceed {tolerance_pct:.0f}% tolerance"
            )
        else:
            lines.append(f"    ✓ All params within {tolerance_pct:.0f}% tolerance")

    if unc_ratios:
        lines.append(
            f"  Uncertainty ratio (CMC/NLSQ): max={max(unc_ratios):.1f}x, median={np.median(unc_ratios):.1f}x"
        )
        # Flag ratios < 0.5 (CMC too precise - possibly corrupted) or > 10 (CMC too uncertain)
        too_precise = sum(1 for r in unc_ratios if r < 0.5)
        too_uncertain = sum(1 for r in unc_ratios if r > 10)
        if too_precise > 0:
            lines.append(
                f"    ⚠️ {too_precise}/{len(unc_ratios)} params have ratio < 0.5x "
                "(CMC artificially precise - check for shard heterogeneity)"
            )
        if too_uncertain > 0:
            lines.append(
                f"    ℹ️ {too_uncertain}/{len(unc_ratios)} params have ratio > 10x (CMC more uncertain)"
            )

    if pcrs:
        lines.append(
            f"  Posterior contraction: max={max(pcrs):.2f}, mean={np.mean(pcrs):.2f}"
        )
        low_pcr = sum(1 for p in pcrs if p < 0.3)
        if low_pcr > 0:
            lines.append(
                f"    ℹ️ {low_pcr}/{len(pcrs)} params have PCR < 0.3 (weak data constraint)"
            )

    lines.append("-" * 80)
    lines.append(
        f"{'Parameter':<18} {'CMC Mean':>11} {'CMC Std':>10} {'NLSQ':>11} "
        f"{'Diff%':>7} {'Z':>6} {'Ratio':>7}"
    )
    lines.append("-" * 80)

    for param_name, metrics in sorted(analysis.items()):
        cmc_mean = metrics.get("cmc_mean", np.nan)
        cmc_std = metrics.get("cmc_std", np.nan)
        nlsq_val = metrics.get("nlsq_value", np.nan)
        z_score = metrics.get("z_score", np.nan)
        rel_diff = metrics.get("relative_diff", np.nan)
        unc_ratio = metrics.get("uncertainty_ratio", np.nan)

        # Format with appropriate precision
        cmc_mean_str = f"{cmc_mean:.4g}" if np.isfinite(cmc_mean) else "N/A"
        cmc_std_str = f"{cmc_std:.4g}" if np.isfinite(cmc_std) else "N/A"
        nlsq_str = f"{nlsq_val:.4g}" if np.isfinite(nlsq_val) else "N/A"
        z_str = f"{z_score:.2f}" if np.isfinite(z_score) else "N/A"
        diff_str = f"{rel_diff * 100:+.1f}%" if np.isfinite(rel_diff) else "N/A"
        ratio_str = f"{unc_ratio:.1f}x" if np.isfinite(unc_ratio) else "N/A"

        # Add warning markers
        marker = ""
        if np.isfinite(z_score) and z_score > 3:
            marker = " ❌"  # Severe
        elif np.isfinite(z_score) and z_score > 2:
            marker = " ⚠️"  # Warning
        elif np.isfinite(rel_diff) and abs(rel_diff * 100) > tolerance_pct:
            marker = " ⚠️"  # Warning
        elif np.isfinite(unc_ratio) and unc_ratio < 0.5:
            marker = " ⚠️"  # Artificially precise

        lines.append(
            f"{param_name:<18} {cmc_mean_str:>11} {cmc_std_str:>10} "
            f"{nlsq_str:>11} {diff_str:>7} {z_str:>6} {ratio_str:>7}{marker}"
        )

    lines.append("=" * 80)

    report = "\n".join(lines)
    log_fn(report)
    return report
