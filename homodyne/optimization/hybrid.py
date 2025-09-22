"""
Hybrid LSQ→MCMC Optimization for Homodyne v2
=============================================

Optimal hybrid optimization strategy combining Least Squares (LSQ)
for fast parameter estimation with MCMC for accurate uncertainty quantification.
This approach provides the best balance of computational efficiency and
statistical accuracy for most homodyne analysis scenarios.

Hybrid Strategy:
1. LSQ Phase: Fast parameter estimation and initial optimization
2. MCMC Phase: Refined sampling initialized from LSQ results
3. Quality Control: Automatic method selection based on convergence
4. Adaptive Resource Allocation: More MCMC for critical analyses

This is the RECOMMENDED optimization method for routine homodyne analysis.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from homodyne.optimization.mcmc import MCMCJAXSampler, MCMCResult, fit_mcmc_jax
from homodyne.optimization.lsq_wrapper import LSQResult, fit_homodyne_lsq
from homodyne.utils.logging import get_logger, log_performance

logger = get_logger(__name__)


@dataclass
class HybridResult:
    """
    Results from hybrid LSQ→MCMC optimization.

    Contains results from both LSQ and MCMC phases along with
    recommendations for which estimates to use.
    """

    # LSQ results (fast parameter estimation)
    lsq_result: LSQResult

    # MCMC results (refined sampling)
    mcmc_result: Optional[MCMCResult]

    # Recommended estimates (automatically selected)
    recommended_params: np.ndarray  # Recommended parameter estimates
    recommended_errors: np.ndarray  # Recommended parameter uncertainties
    recommended_contrast: float  # Recommended contrast
    recommended_offset: float  # Recommended offset
    recommendation_source: str  # Source of recommendation ("LSQ", "MCMC", "hybrid")

    # Hybrid-specific metrics
    lsq_mcmc_agreement: float  # Agreement between LSQ and MCMC (if both available)
    quality_score: float  # Overall quality score (0-1)

    # Computational summary
    total_computation_time: float  # Total time for both phases
    lsq_time_fraction: float  # Fraction of time spent on LSQ

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive hybrid optimization summary."""
        summary = {
            "recommendations": {
                "parameters": self.recommended_params.tolist(),
                "errors": self.recommended_errors.tolist(),
                "contrast": self.recommended_contrast,
                "offset": self.recommended_offset,
                "source": self.recommendation_source,
                "quality_score": self.quality_score,
            },
            "lsq_phase": self.lsq_result.get_summary(),
            "computational_efficiency": {
                "total_time": self.total_computation_time,
                "lsq_fraction": self.lsq_time_fraction,
                "mcmc_fraction": (
                    1.0 - self.lsq_time_fraction if self.mcmc_result else 0.0
                ),
            },
        }

        if self.mcmc_result:
            summary["mcmc_phase"] = self.mcmc_result.get_summary()
            summary["agreement"] = {
                "lsq_mcmc_agreement": self.lsq_mcmc_agreement,
            }

        return summary

    def get_final_estimates(self) -> Dict[str, Any]:
        """Get final parameter estimates with uncertainties."""
        return {
            "parameters": {
                "values": self.recommended_params,
                "errors": self.recommended_errors,
            },
            "scaling": {
                "contrast": self.recommended_contrast,
                "offset": self.recommended_offset,
            },
            "source": self.recommendation_source,
            "quality_score": self.quality_score,
        }


def fit_hybrid_lsq_mcmc(
    data,
    sigma,
    t1,
    t2,
    phi,
    q,
    L,
    analysis_mode="laminar_flow",
    lsq_max_iterations=10000,
    mcmc_samples=1000,
    mcmc_warmup=1000,
    mcmc_chains=4,
    use_lsq_init=True,
    convergence_threshold=0.1,
    **kwargs
):
    """
    Hybrid LSQ→MCMC optimization for homodyne analysis.

    Args:
        data, sigma, t1, t2, phi, q, L: Standard homodyne analysis inputs
        analysis_mode: Analysis mode ('static_isotropic', 'static_anisotropic', 'laminar_flow')
        lsq_max_iterations: Maximum iterations for LSQ phase
        mcmc_samples: Number of MCMC samples per chain
        mcmc_warmup: Number of MCMC warmup samples
        mcmc_chains: Number of MCMC chains
        use_lsq_init: Whether to initialize MCMC with LSQ results
        convergence_threshold: Quality threshold for recommendation selection
        **kwargs: Additional parameters

    Returns:
        HybridResult: Complete hybrid optimization results
    """
    start_time = time.time()

    logger.info("🔄 Starting Hybrid LSQ→MCMC optimization")
    logger.info(f"  Phase 1: LSQ optimization ({lsq_max_iterations} max iterations)")
    logger.info(f"  Phase 2: MCMC sampling ({mcmc_samples} samples × {mcmc_chains} chains)")

    # Phase 1: LSQ Optimization
    logger.info("⚡ Running LSQ phase...")
    lsq_start = time.time()

    try:
        lsq_result = fit_homodyne_lsq(
            data=data,
            sigma=sigma,
            t1=t1,
            t2=t2,
            phi=phi,
            q=q,
            L=L,
            analysis_mode=analysis_mode,
            max_iterations=lsq_max_iterations,
            **kwargs
        )
        lsq_time = time.time() - lsq_start
        logger.info(f"✓ LSQ phase completed in {lsq_time:.2f}s")

    except Exception as e:
        logger.error(f"❌ LSQ phase failed: {e}")
        raise RuntimeError(f"Hybrid optimization failed in LSQ phase: {e}")

    # Phase 2: MCMC Refinement
    logger.info("🎲 Running MCMC phase...")
    mcmc_start = time.time()

    try:
        # Initialize MCMC with LSQ results if requested
        init_params = None
        if use_lsq_init and hasattr(lsq_result, 'final_params'):
            init_params = lsq_result.final_params
            logger.info("  Using LSQ results for MCMC initialization")

        mcmc_result = fit_mcmc_jax(
            data=data,
            sigma=sigma,
            t1=t1,
            t2=t2,
            phi=phi,
            q=q,
            L=L,
            analysis_mode=analysis_mode,
            n_samples=mcmc_samples,
            n_warmup=mcmc_warmup,
            n_chains=mcmc_chains,
            init_params=init_params,
            **kwargs
        )
        mcmc_time = time.time() - mcmc_start
        logger.info(f"✓ MCMC phase completed in {mcmc_time:.2f}s")

    except Exception as e:
        logger.warning(f"⚠️ MCMC phase failed: {e}")
        logger.info("  Continuing with LSQ-only results")
        mcmc_result = None
        mcmc_time = 0.0

    # Generate recommendations
    total_time = time.time() - start_time
    lsq_time_fraction = lsq_time / total_time

    # Select best estimates based on available results and quality
    if mcmc_result is not None:
        # Compare LSQ and MCMC results
        lsq_params = lsq_result.final_params
        mcmc_params = mcmc_result.mean_params

        # Calculate parameter agreement
        param_agreement = _calculate_parameter_agreement(lsq_params, mcmc_params)

        # Determine recommendation source
        if param_agreement > convergence_threshold:
            # Good agreement - use MCMC for best uncertainties
            recommended_params = mcmc_params
            recommended_errors = mcmc_result.std_params
            recommended_contrast = mcmc_result.final_contrast
            recommended_offset = mcmc_result.final_offset
            recommendation_source = "MCMC"
            quality_score = min(0.9, param_agreement)
        else:
            # Poor agreement - use hybrid approach
            recommended_params = (lsq_params + mcmc_params) / 2
            recommended_errors = mcmc_result.std_params * 1.5  # Conservative uncertainties
            recommended_contrast = (lsq_result.final_contrast + mcmc_result.final_contrast) / 2
            recommended_offset = (lsq_result.final_offset + mcmc_result.final_offset) / 2
            recommendation_source = "hybrid"
            quality_score = param_agreement * 0.7
            logger.warning(f"⚠️ LSQ-MCMC agreement low ({param_agreement:.3f}), using hybrid estimates")
    else:
        # MCMC failed - use LSQ only
        recommended_params = lsq_result.final_params
        recommended_errors = getattr(lsq_result, 'param_errors', np.ones_like(lsq_result.final_params) * 0.1)
        recommended_contrast = lsq_result.final_contrast
        recommended_offset = lsq_result.final_offset
        recommendation_source = "LSQ"
        quality_score = 0.6  # Lower quality without MCMC uncertainties
        param_agreement = 1.0

    result = HybridResult(
        lsq_result=lsq_result,
        mcmc_result=mcmc_result,
        recommended_params=recommended_params,
        recommended_errors=recommended_errors,
        recommended_contrast=recommended_contrast,
        recommended_offset=recommended_offset,
        recommendation_source=recommendation_source,
        lsq_mcmc_agreement=param_agreement,
        quality_score=quality_score,
        total_computation_time=total_time,
        lsq_time_fraction=lsq_time_fraction,
    )

    logger.info(f"🎯 Hybrid optimization completed in {total_time:.2f}s")
    logger.info(f"  Recommendation: {recommendation_source} (quality: {quality_score:.3f})")
    if mcmc_result:
        logger.info(f"  LSQ-MCMC agreement: {param_agreement:.3f}")

    return result


def _calculate_parameter_agreement(lsq_params, mcmc_params):
    """
    Calculate agreement between LSQ and MCMC parameter estimates.

    Returns:
        float: Agreement score (0-1, higher is better)
    """
    if len(lsq_params) != len(mcmc_params):
        return 0.0

    # Normalize differences by parameter magnitudes
    relative_diffs = []
    for lsq_val, mcmc_val in zip(lsq_params, mcmc_params):
        if abs(lsq_val) > 1e-12 or abs(mcmc_val) > 1e-12:
            scale = max(abs(lsq_val), abs(mcmc_val))
            rel_diff = abs(lsq_val - mcmc_val) / scale
            relative_diffs.append(rel_diff)

    if not relative_diffs:
        return 1.0

    # Convert to agreement score (1 - mean relative difference)
    mean_rel_diff = np.mean(relative_diffs)
    agreement = max(0.0, 1.0 - mean_rel_diff)

    return agreement


# Alias for backward compatibility
optimize_hybrid = fit_hybrid_lsq_mcmc