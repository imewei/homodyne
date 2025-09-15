"""
Hybrid VI→MCMC Optimization for Homodyne v2
============================================

Optimal hybrid optimization strategy combining Variational Inference (VI)
for fast exploration with MCMC for accurate uncertainty quantification.
This approach provides the best balance of computational efficiency and
statistical accuracy for most homodyne analysis scenarios.

Hybrid Strategy:
1. VI Phase: Fast parameter exploration and approximate posterior
2. MCMC Phase: Refined sampling initialized from VI results
3. Quality Control: Automatic method selection based on convergence
4. Adaptive Resource Allocation: More MCMC for critical analyses

This is the RECOMMENDED optimization method for routine homodyne analysis.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from homodyne.optimization.mcmc import MCMCJAXSampler, MCMCResult, fit_mcmc_jax
from homodyne.optimization.variational import (VariationalInferenceJAX,
                                               VIResult, fit_vi_jax)
from homodyne.utils.logging import get_logger, log_performance

logger = get_logger(__name__)


@dataclass
class HybridResult:
    """
    Results from hybrid VI→MCMC optimization.

    Contains results from both VI and MCMC phases along with
    recommendations for which estimates to use.
    """

    # VI results (fast exploration)
    vi_result: VIResult

    # MCMC results (refined sampling)
    mcmc_result: Optional[MCMCResult]

    # Recommended estimates (automatically selected)
    recommended_params: np.ndarray  # Recommended parameter estimates
    recommended_errors: np.ndarray  # Recommended parameter uncertainties
    recommended_contrast: float  # Recommended contrast
    recommended_offset: float  # Recommended offset
    recommendation_source: str  # Source of recommendation ("VI", "MCMC", "hybrid")

    # Hybrid-specific metrics
    vi_mcmc_agreement: float  # Agreement between VI and MCMC (if both available)
    quality_score: float  # Overall quality score (0-1)

    # Computational summary
    total_computation_time: float  # Total time for both phases
    vi_time_fraction: float  # Fraction of time spent on VI

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
            "vi_phase": self.vi_result.get_summary(),
            "computational_efficiency": {
                "total_time": self.total_computation_time,
                "vi_fraction": self.vi_time_fraction,
                "mcmc_fraction": (
                    1.0 - self.vi_time_fraction if self.mcmc_result else 0.0
                ),
            },
        }

        if self.mcmc_result:
            summary["mcmc_phase"] = self.mcmc_result.get_summary()
            summary["agreement"] = {
                "vi_mcmc_agreement": self.vi_mcmc_agreement,
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
            "metadata": {
                "source": self.recommendation_source,
                "quality": self.quality_score,
            },
        }


class HybridOptimizer:
    """
    Hybrid VI→MCMC optimizer for balanced speed and accuracy.

    Automatically adapts the optimization strategy based on:
    - Data complexity and signal-to-noise ratio
    - VI convergence quality
    - Available computational resources
    - Required accuracy level
    """

    def __init__(
        self,
        analysis_mode: str = "laminar_flow",
        adaptive_strategy: bool = True,
        mcmc_threshold: float = 0.8,
    ):
        """
        Initialize hybrid optimizer.

        Args:
            analysis_mode: Analysis mode for theory engines
            adaptive_strategy: Enable adaptive MCMC triggering
            mcmc_threshold: VI quality threshold for triggering MCMC
        """
        self.analysis_mode = analysis_mode
        self.adaptive_strategy = adaptive_strategy
        self.mcmc_threshold = mcmc_threshold

        # Initialize optimizers
        self.vi_optimizer = VariationalInferenceJAX(analysis_mode)
        self.mcmc_sampler = MCMCJAXSampler(analysis_mode)

        logger.info(f"Hybrid optimizer initialized for {analysis_mode}")

    @log_performance(threshold=2.0)
    def optimize(
        self,
        data: np.ndarray,
        sigma: np.ndarray,
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
        q: float,
        L: float,
        vi_iterations: int = 2000,
        mcmc_samples: int = 1000,
        mcmc_chains: int = 4,
        force_mcmc: bool = False,
        target_quality: float = 0.9,
    ) -> HybridResult:
        """
        Perform hybrid VI→MCMC optimization.

        Args:
            data: Experimental correlation data
            sigma: Measurement uncertainties
            t1, t2: Time grids
            phi: Angle grid
            q: Wave vector magnitude
            L: Sample-detector distance
            vi_iterations: VI optimization iterations
            mcmc_samples: MCMC samples per chain
            mcmc_chains: Number of MCMC chains
            force_mcmc: Force MCMC even if VI is good
            target_quality: Target quality score for recommendations

        Returns:
            Hybrid optimization result
        """
        start_time = time.time()

        # Phase 1: Variational Inference
        logger.info("Starting VI phase...")
        vi_start = time.time()
        vi_result = self.vi_optimizer.fit(
            data, sigma, t1, t2, phi, q, L, n_iterations=vi_iterations
        )
        vi_time = time.time() - vi_start

        # Assess VI quality
        vi_quality = self._assess_vi_quality(vi_result, data.size)
        logger.info(
            f"VI phase completed: quality={vi_quality:.3f}, time={vi_time:.2f}s"
        )

        # Phase 2: Decide on MCMC
        mcmc_result = None
        mcmc_time = 0.0

        run_mcmc = (
            force_mcmc
            or (self.adaptive_strategy and vi_quality < self.mcmc_threshold)
            or target_quality > 0.95  # High accuracy required
        )

        if run_mcmc:
            logger.info("Starting MCMC phase...")
            mcmc_start = time.time()

            # Initialize MCMC from VI results
            initial_values = {
                "params": vi_result.mean_params,
                "contrast": vi_result.mean_contrast,
                "offset": vi_result.mean_offset,
            }

            mcmc_result = self.mcmc_sampler.sample(
                data,
                sigma,
                t1,
                t2,
                phi,
                q,
                L,
                n_samples=mcmc_samples,
                n_chains=mcmc_chains,
                initial_values=initial_values,
            )
            mcmc_time = time.time() - mcmc_start
            logger.info(f"MCMC phase completed: time={mcmc_time:.2f}s")
        else:
            logger.info("Skipping MCMC phase - VI quality sufficient")

        # Phase 3: Generate recommendations
        recommendations = self._generate_recommendations(
            vi_result, mcmc_result, target_quality
        )

        # Compute final metrics
        total_time = time.time() - start_time
        vi_time_fraction = vi_time / total_time

        vi_mcmc_agreement = (
            self._compute_vi_mcmc_agreement(vi_result, mcmc_result)
            if mcmc_result
            else 1.0
        )

        quality_score = self._compute_overall_quality(
            vi_result, mcmc_result, vi_mcmc_agreement
        )

        result = HybridResult(
            vi_result=vi_result,
            mcmc_result=mcmc_result,
            recommended_params=recommendations["params"],
            recommended_errors=recommendations["errors"],
            recommended_contrast=recommendations["contrast"],
            recommended_offset=recommendations["offset"],
            recommendation_source=recommendations["source"],
            vi_mcmc_agreement=vi_mcmc_agreement,
            quality_score=quality_score,
            total_computation_time=total_time,
            vi_time_fraction=vi_time_fraction,
        )

        logger.info(
            f"Hybrid optimization completed: quality={quality_score:.3f}, "
            f"recommendation_source={recommendations['source']}, "
            f"total_time={total_time:.2f}s"
        )

        return result

    def _assess_vi_quality(self, vi_result: VIResult, n_data_points: int) -> float:
        """
        Assess VI optimization quality (0-1 score).

        Factors:
        - Convergence behavior
        - Parameter uncertainty estimates
        - Fit quality (chi-squared)
        - ELBO stability
        """
        quality_factors = []

        # 1. Convergence quality
        if vi_result.converged:
            # Check ELBO stability in final iterations
            if len(vi_result.elbo_history) > 100:
                final_elbos = vi_result.elbo_history[-100:]
                elbo_stability = 1.0 - np.std(final_elbos) / (
                    np.abs(np.mean(final_elbos)) + 1e-6
                )
                quality_factors.append(np.clip(elbo_stability, 0, 1))
            else:
                quality_factors.append(0.8)  # Moderate if short run
        else:
            quality_factors.append(0.3)  # Poor if not converged

        # 2. Fit quality (chi-squared reasonableness)
        expected_chi2 = n_data_points - len(vi_result.mean_params) - 2
        chi2_ratio = vi_result.chi_squared / expected_chi2
        chi2_quality = np.exp(-0.5 * (chi2_ratio - 1) ** 2)  # Gaussian around 1
        quality_factors.append(np.clip(chi2_quality, 0, 1))

        # 3. Parameter uncertainty reasonableness
        # Uncertainties should be neither too small (overconfident) nor too large (uninformative)
        relative_uncertainties = vi_result.std_params / (
            np.abs(vi_result.mean_params) + 1e-6
        )
        # Good uncertainties are typically 1-50% of parameter values
        uncertainty_quality = np.mean(
            np.exp(-0.5 * (np.log10(relative_uncertainties + 1e-6) + 1) ** 2)
        )
        quality_factors.append(np.clip(uncertainty_quality, 0, 1))

        # Overall quality (weighted average)
        weights = [0.4, 0.4, 0.2]  # Convergence and fit quality are most important
        overall_quality = np.average(quality_factors, weights=weights)

        return float(np.clip(overall_quality, 0, 1))

    def _generate_recommendations(
        self,
        vi_result: VIResult,
        mcmc_result: Optional[MCMCResult],
        target_quality: float,
    ) -> Dict[str, Any]:
        """
        Generate final parameter recommendations based on available results.
        """
        if mcmc_result is None:
            # VI-only recommendations
            return {
                "params": vi_result.mean_params,
                "errors": vi_result.std_params,
                "contrast": vi_result.mean_contrast,
                "offset": vi_result.mean_offset,
                "source": "VI",
            }

        # Both VI and MCMC available - choose best or combine
        vi_mcmc_agreement = self._compute_vi_mcmc_agreement(vi_result, mcmc_result)

        if vi_mcmc_agreement > 0.9:
            # High agreement - can use either, prefer MCMC for uncertainties
            return {
                "params": mcmc_result.mean_params,
                "errors": mcmc_result.std_params,
                "contrast": mcmc_result.mean_contrast,
                "offset": mcmc_result.mean_offset,
                "source": "MCMC",
            }
        elif vi_mcmc_agreement > 0.7:
            # Moderate agreement - weighted combination
            vi_weight = 0.3
            mcmc_weight = 0.7

            combined_params = (
                vi_weight * vi_result.mean_params
                + mcmc_weight * mcmc_result.mean_params
            )
            # Use MCMC uncertainties (typically more reliable)
            combined_errors = mcmc_result.std_params
            combined_contrast = (
                vi_weight * vi_result.mean_contrast
                + mcmc_weight * mcmc_result.mean_contrast
            )
            combined_offset = (
                vi_weight * vi_result.mean_offset
                + mcmc_weight * mcmc_result.mean_offset
            )

            return {
                "params": combined_params,
                "errors": combined_errors,
                "contrast": combined_contrast,
                "offset": combined_offset,
                "source": "hybrid",
            }
        else:
            # Poor agreement - flag for manual inspection, use MCMC as more reliable
            logger.warning(
                f"Poor VI-MCMC agreement ({vi_mcmc_agreement:.3f}) - "
                "results may require manual inspection"
            )
            return {
                "params": mcmc_result.mean_params,
                "errors": mcmc_result.std_params,
                "contrast": mcmc_result.mean_contrast,
                "offset": mcmc_result.mean_offset,
                "source": "MCMC",
            }

    def _compute_vi_mcmc_agreement(
        self, vi_result: VIResult, mcmc_result: MCMCResult
    ) -> float:
        """
        Compute agreement score between VI and MCMC results.

        Agreement is measured as overlap of parameter estimates
        within their respective uncertainties.
        """
        # Parameter agreement (weighted by uncertainties)
        param_agreements = []

        for i in range(len(vi_result.mean_params)):
            vi_mean, vi_std = vi_result.mean_params[i], vi_result.std_params[i]
            mcmc_mean, mcmc_std = mcmc_result.mean_params[i], mcmc_result.std_params[i]

            # Distance in units of combined uncertainty
            combined_std = np.sqrt(vi_std**2 + mcmc_std**2)
            distance = abs(vi_mean - mcmc_mean) / (combined_std + 1e-6)

            # Agreement score (Gaussian-like, 1 at distance=0, ~0 at distance=3)
            agreement = np.exp(-0.5 * distance**2)
            param_agreements.append(agreement)

        # Scaling parameter agreement
        contrast_distance = abs(vi_result.mean_contrast - mcmc_result.mean_contrast) / (
            np.sqrt(vi_result.std_contrast**2 + mcmc_result.std_contrast**2) + 1e-6
        )
        contrast_agreement = np.exp(-0.5 * contrast_distance**2)

        offset_distance = abs(vi_result.mean_offset - mcmc_result.mean_offset) / (
            np.sqrt(vi_result.std_offset**2 + mcmc_result.std_offset**2) + 1e-6
        )
        offset_agreement = np.exp(-0.5 * offset_distance**2)

        # Overall agreement (weighted average)
        all_agreements = param_agreements + [contrast_agreement, offset_agreement]
        overall_agreement = np.mean(all_agreements)

        return float(np.clip(overall_agreement, 0, 1))

    def _compute_overall_quality(
        self,
        vi_result: VIResult,
        mcmc_result: Optional[MCMCResult],
        vi_mcmc_agreement: float,
    ) -> float:
        """
        Compute overall optimization quality score.
        """
        # Base VI quality
        vi_quality = self._assess_vi_quality(vi_result, 1000)  # Placeholder n_data

        if mcmc_result is None:
            return vi_quality

        # MCMC quality factors
        mcmc_converged = mcmc_result.converged
        mcmc_acceptance = mcmc_result.acceptance_rate

        mcmc_quality = 0.5  # Base score
        if mcmc_converged:
            mcmc_quality += 0.3
        if 0.2 <= mcmc_acceptance <= 0.8:  # Reasonable acceptance rate
            mcmc_quality += 0.2

        # Combined quality considering agreement
        agreement_bonus = vi_mcmc_agreement * 0.2  # Up to 20% bonus for good agreement
        overall_quality = max(vi_quality, mcmc_quality) + agreement_bonus

        return float(np.clip(overall_quality, 0, 1))


# Convenience function for direct hybrid optimization
def optimize_hybrid(
    data: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    t1: np.ndarray = None,
    t2: np.ndarray = None,
    phi: np.ndarray = None,
    q: float = None,
    L: float = None,
    analysis_mode: str = "laminar_flow",
    estimate_noise: bool = False,
    noise_model: str = "hierarchical",
    **kwargs,
) -> HybridResult:
    """
    Convenience function for direct hybrid optimization.

    PIPELINE LOGIC:
    - Case 1 (Traditional): optimize_hybrid(data, sigma=provided_sigma)
      → VI(provided σ) → MCMC(provided σ, VI init)
    - Case 2 (Hybrid): optimize_hybrid(data, estimate_noise=True, noise_model="hierarchical")  
      → HybridNumPyro(Adam noise) → VI(estimated σ) → MCMC(estimated σ, VI init)

    Args:
        data: Experimental correlation data
        sigma: Measurement uncertainties (optional if estimate_noise=True)
        t1, t2: Time grids
        phi: Angle grid
        q: Wave vector magnitude
        L: Sample-detector distance
        analysis_mode: Analysis mode
        estimate_noise: Enable hybrid NumPyro noise estimation
        noise_model: Noise model type ("hierarchical", "per_angle", "adaptive")
        **kwargs: Additional arguments for HybridOptimizer

    Returns:
        Hybrid optimization result
        
    Raises:
        ValueError: If neither sigma is provided nor estimate_noise is enabled
        ImportError: If hybrid noise estimation requires JAX/NumPyro but not available
    """
    # Handle hybrid noise estimation if requested
    if sigma is None and estimate_noise:
        logger.info(f"🔄 Hybrid Pipeline: NumPyro noise → VI → MCMC ({noise_model})")
        
        try:
            from homodyne.optimization.hybrid_noise_estimation import HybridNoiseEstimator
            
            # Initialize noise estimator
            noise_estimator = HybridNoiseEstimator(analysis_mode)
            
            # Use hybrid NumPyro pipeline (noise estimation → VI → MCMC)
            return noise_estimator.estimate_noise_for_hybrid(
                data, t1, t2, phi, q, L, noise_model, **kwargs
            )
            
        except ImportError as e:
            logger.error("❌ Hybrid noise estimation requires JAX and NumPyro")
            raise ImportError(
                "Hybrid noise estimation requires JAX and NumPyro. "
                f"Install with: pip install jax numpyro. Error: {e}"
            ) from e
        except Exception as e:
            logger.error(f"❌ Hybrid noise estimation failed: {e}")
            raise RuntimeError(f"Hybrid noise estimation failed: {e}") from e
    
    elif sigma is None and not estimate_noise:
        raise ValueError(
            "Must provide sigma or set estimate_noise=True. "
            "Use estimate_noise=True for automatic noise estimation via hybrid NumPyro."
        )
    
    # Validate required parameters
    if any(param is None for param in [t1, t2, phi, q, L]):
        raise ValueError("t1, t2, phi, q, L are required parameters")
    
    # Traditional hybrid optimization with provided sigma
    logger.info("🔄 Hybrid Pipeline: Traditional mode with provided sigma")
    optimizer = HybridOptimizer(analysis_mode=analysis_mode)
    return optimizer.optimize(data, sigma, t1, t2, phi, q, L, **kwargs)


# Export main classes and functions
__all__ = [
    "HybridResult",
    "HybridOptimizer",
    "optimize_hybrid",
]
