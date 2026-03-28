"""MCMC Diagnostic Report Generation.

Provides functions to generate comprehensive diagnostic reports (all plots)
and print formatted MCMC summaries to the console.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


def generate_mcmc_diagnostic_report(
    result: Any,
    output_dir: str | Path,
    prefix: str = "mcmc",
    include_heatmaps: bool = True,
    dpi: int = 150,
) -> dict[str, Path]:
    """Generate comprehensive MCMC diagnostic report with all plots.

    Creates a complete set of diagnostic plots for MCMC results:
    1. ArviZ trace plots (trace + posterior side-by-side)
    2. ArviZ posterior distributions with 95% CI
    3. ArviZ pair plots (parameter correlations)
    4. Convergence diagnostics (R-hat, ESS)
    5. CMC-specific: KL divergence matrix, shard comparison

    Parameters
    ----------
    result : MCMCResult
        MCMC or CMC result object.
    output_dir : str or Path
        Directory to save plots.
    prefix : str, default="mcmc"
        Prefix for output filenames.
    include_heatmaps : bool, default=True
        Whether to include C2 heatmap comparisons (requires fitted_data).
    dpi : int, default=150
        DPI for saved figures.

    Returns
    -------
    dict[str, Path]
        Dictionary mapping plot names to file paths.

    Examples
    --------
    >>> paths = generate_mcmc_diagnostic_report(result, "output/mcmc_diagnostics")
    >>> print(paths["trace"])  # Path to trace plot
    >>> print(paths["posterior"])  # Path to posterior plot
    """
    from homodyne.utils.path_validation import PathValidationError, get_safe_output_dir
    from homodyne.viz.mcmc_arviz import (
        plot_arviz_pair,
        plot_arviz_posterior,
        plot_arviz_trace,
    )
    from homodyne.viz.mcmc_dashboard import plot_cmc_summary_dashboard
    from homodyne.viz.mcmc_diagnostics import (
        plot_convergence_diagnostics,
        plot_kl_divergence_matrix,
    )

    try:
        output_dir = get_safe_output_dir(output_dir)
    except (PathValidationError, PermissionError) as e:
        logger.error(f"Invalid MCMC diagnostic output directory: {e}")
        return {}

    paths: dict[str, Path] = {}

    # 1. ArviZ trace plots
    try:
        trace_path = output_dir / f"{prefix}_trace.png"
        plot_arviz_trace(result, save_path=trace_path, dpi=dpi)
        paths["trace"] = trace_path
        logger.info(f"Generated trace plot: {trace_path}")
    except (ValueError, TypeError, OSError) as e:
        logger.warning(f"Failed to generate trace plot: {e}")

    # 2. ArviZ posterior distributions
    try:
        posterior_path = output_dir / f"{prefix}_posterior.png"
        plot_arviz_posterior(result, save_path=posterior_path, dpi=dpi)
        paths["posterior"] = posterior_path
        logger.info(f"Generated posterior plot: {posterior_path}")
    except (ValueError, TypeError, OSError) as e:
        logger.warning(f"Failed to generate posterior plot: {e}")

    # 3. ArviZ pair plots
    try:
        pair_path = output_dir / f"{prefix}_pair.png"
        plot_arviz_pair(result, save_path=pair_path, dpi=dpi)
        paths["pair"] = pair_path
        logger.info(f"Generated pair plot: {pair_path}")
    except (ValueError, TypeError, OSError) as e:
        logger.warning(f"Failed to generate pair plot: {e}")

    # 4. Convergence diagnostics
    try:
        conv_path = output_dir / f"{prefix}_convergence.png"
        plot_convergence_diagnostics(result, save_path=conv_path, dpi=dpi)
        paths["convergence"] = conv_path
        logger.info(f"Generated convergence plot: {conv_path}")
    except (ValueError, TypeError, OSError) as e:
        logger.warning(f"Failed to generate convergence plot: {e}")

    # 4b. BFMI diagnostics
    try:
        from homodyne.viz.mcmc_diagnostics import compute_bfmi

        bfmi_value = None
        # Try to extract energy from extra_fields or inference_data
        if hasattr(result, "extra_fields") and result.extra_fields is not None:
            energy = result.extra_fields.get("potential_energy")
            if energy is not None:
                bfmi_value = compute_bfmi(np.asarray(energy))

        if bfmi_value is not None and np.isfinite(bfmi_value):
            bfmi_status = "GOOD" if bfmi_value >= 0.3 else "LOW (review mass matrix adaptation)"
            logger.info(f"BFMI = {bfmi_value:.4f} ({bfmi_status})")
            paths["bfmi_value"] = bfmi_value  # Store as metadata
        else:
            logger.debug("BFMI: potential_energy not available in result")
    except (ValueError, TypeError, ImportError) as e:
        logger.debug(f"BFMI computation skipped: {e}")

    # 5. CMC-specific plots
    if result.is_cmc_result():
        # KL divergence matrix
        try:
            kl_path = output_dir / f"{prefix}_kl_matrix.png"
            plot_kl_divergence_matrix(result, save_path=kl_path, dpi=dpi)
            paths["kl_matrix"] = kl_path
            logger.info(f"Generated KL matrix plot: {kl_path}")
        except (ValueError, TypeError, OSError) as e:
            logger.warning(f"Failed to generate KL matrix plot: {e}")

        # CMC summary dashboard
        try:
            dashboard_path = output_dir / f"{prefix}_cmc_dashboard.png"
            plot_cmc_summary_dashboard(result, save_path=dashboard_path, dpi=dpi)
            paths["cmc_dashboard"] = dashboard_path
            logger.info(f"Generated CMC dashboard: {dashboard_path}")
        except (ValueError, TypeError, OSError) as e:
            logger.warning(f"Failed to generate CMC dashboard: {e}")

    # 6. Summary statistics (save as CSV)
    try:
        summary_path = output_dir / f"{prefix}_summary.csv"
        summary_df = result.compute_summary()
        summary_df.to_csv(summary_path)
        paths["summary_csv"] = summary_path
        logger.info(f"Generated summary CSV: {summary_path}")
    except (ValueError, TypeError, OSError) as e:
        logger.warning(f"Failed to generate summary CSV: {e}")

    logger.info(f"MCMC diagnostic report generated: {len(paths)} files in {output_dir}")
    return paths


def print_mcmc_summary(result: Any) -> None:  # MCMCResult type
    """Print formatted MCMC summary to console.

    Displays key diagnostics including:
    - Parameter estimates with 95% CI
    - R-hat and ESS for each parameter
    - Convergence status
    - CMC-specific information if applicable

    Parameters
    ----------
    result : MCMCResult
        MCMC or CMC result object.

    Examples
    --------
    >>> print_mcmc_summary(result)
    """
    import math

    logger.info("=" * 60)
    logger.info("MCMC Results Summary")
    logger.info("=" * 60)

    # Basic info
    logger.info("Analysis Mode: %s", result.analysis_mode)
    logger.info("Sampler: %s", result.sampler)
    logger.info("Converged: %s", result.converged)
    logger.info("Computation Time: %.2fs", result.computation_time)

    if result.is_cmc_result():
        logger.info("CMC Information:")
        logger.info("  - Number of Shards: %s", result.num_shards)
        logger.info("  - Combination Method: %s", result.combination_method)

    # Parameter estimates
    logger.info("-" * 60)
    logger.info("Parameter Estimates (95%% CI)")
    logger.info("-" * 60)

    param_names = result.get_param_names()

    # Compute CI if not already computed
    if result.ci_95_lower is None or result.ci_95_upper is None:
        try:
            ci_lower, ci_upper = result.compute_credible_intervals(0.95)
        except ValueError:
            ci_lower = ci_upper = None
    else:
        ci_lower, ci_upper = result.ci_95_lower, result.ci_95_upper

    for i, name in enumerate(param_names):
        mean = result.mean_params[i]
        std = result.std_params[i] if result.std_params is not None else 0.0

        if ci_lower is not None and ci_upper is not None:
            logger.info(
                "  %20s: %12.4f +/- %8.4f  [%.4f, %.4f]",
                name,
                mean,
                std,
                ci_lower[i],
                ci_upper[i],
            )
        else:
            logger.info("  %20s: %12.4f +/- %8.4f", name, mean, std)

    # Convergence diagnostics
    if result.r_hat is not None or result.effective_sample_size is not None:
        logger.info("-" * 60)
        logger.info("Convergence Diagnostics")
        logger.info("-" * 60)

        if result.r_hat is not None:
            logger.info("  R-hat (target < 1.1):")
            for name, value in result.r_hat.items():
                if not math.isfinite(value):
                    status = "N/A"
                elif value < 1.1:
                    status = "pass"
                else:
                    status = "FAIL"
                logger.info("    %20s: %.4f %s", name, value, status)

        if result.effective_sample_size is not None:
            logger.info("  ESS (target > 400):")
            for name, value in result.effective_sample_size.items():
                if not math.isfinite(value):
                    status = "N/A"
                elif value > 400:
                    status = "pass"
                else:
                    status = "FAIL"
                logger.info("    %20s: %.1f %s", name, value, status)

    # BFMI diagnostics
    if hasattr(result, "extra_fields") and result.extra_fields is not None:
        energy = result.extra_fields.get("potential_energy")
        if energy is not None:
            try:
                from homodyne.viz.mcmc_diagnostics import compute_bfmi

                bfmi_value = compute_bfmi(np.asarray(energy))
                if math.isfinite(bfmi_value):
                    bfmi_status = "pass" if bfmi_value >= 0.3 else "LOW"
                    logger.info("  BFMI (target >= 0.3):")
                    logger.info("    %20s: %.4f %s", "pooled", bfmi_value, bfmi_status)
            except (ValueError, TypeError, ImportError):
                pass

    logger.info("=" * 60)
