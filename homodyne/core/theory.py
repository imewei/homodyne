"""
Theory Computation Engine for Homodyne v2
==========================================

High-level interface to theoretical calculations for homodyne scattering analysis.
This module provides user-friendly wrappers around the JAX backend functions
with proper error handling, validation, and computational management.

The theory engine handles:
- Model selection and parameter management
- Efficient computation orchestration
- Memory management for large datasets
- Error handling and validation
- Performance monitoring and optimization hints
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from homodyne.utils.logging import get_logger, log_performance

# Import with fallback handling
try:
    from homodyne.core.jax_backend import (batch_chi_squared,
                                           compute_chi_squared,
                                           compute_g1_diffusion,
                                           compute_g1_shear, compute_g1_total,
                                           compute_g2_scaled, jax_available,
                                           jnp, vectorized_g2_computation)
except ImportError:
    jax_available = False
    logger = get_logger(__name__)
    logger.error("Could not import JAX backend - theory computations disabled")

from homodyne.core.models import CombinedModel, create_model
from homodyne.core.physics import (PhysicsConstants, parameter_bounds,
                                   validate_parameters)

logger = get_logger(__name__)


class TheoryEngine:
    """
    High-level interface for theoretical homodyne calculations.

    Manages model selection, parameter validation, and efficient
    computation orchestration for homodyne scattering analysis.
    """

    def __init__(self, analysis_mode: str = "laminar_flow"):
        """
        Initialize theory engine with specified analysis mode.

        Args:
            analysis_mode: "static_isotropic", "static_anisotropic", or "laminar_flow"
        """
        self.analysis_mode = analysis_mode
        self.model = create_model(analysis_mode)
        self._validate_backend()

        logger.info(f"Theory engine initialized for {analysis_mode}")

    def _validate_backend(self):
        """Validate that computational backend is available."""
        if not jax_available:
            logger.warning("JAX backend not available - computations will be slower")

    @log_performance(threshold=0.01)
    def compute_g1(
        self,
        params: np.ndarray,
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
        q: float,
        L: float,
    ) -> np.ndarray:
        """
        Compute g1 correlation function.

        Args:
            params: Physical parameters
            t1, t2: Time grids
            phi: Angle grid
            q: Wave vector magnitude
            L: Sample-detector distance

        Returns:
            g1 correlation function
        """
        # Validate inputs
        self._validate_computation_inputs(params, q, L)

        # Convert to JAX arrays if needed
        if jax_available:
            params = jnp.asarray(params)
            t1 = jnp.asarray(t1)
            t2 = jnp.asarray(t2)
            phi = jnp.asarray(phi)

        return self.model.compute_g1(params, t1, t2, phi, q, L)

    @log_performance(threshold=0.01)
    def compute_g2(
        self,
        params: np.ndarray,
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
        q: float,
        L: float,
        contrast: float,
        offset: float,
    ) -> np.ndarray:
        """
        Compute g2 with scaled fitting: g₂ = offset + contrast × [g₁]²

        This is the core equation for homodyne analysis.

        Args:
            params: Physical parameters
            t1, t2: Time grids
            phi: Angle grid
            q: Wave vector magnitude
            L: Sample-detector distance
            contrast: Contrast parameter
            offset: Baseline offset

        Returns:
            g2 correlation function
        """
        # Validate inputs
        self._validate_computation_inputs(params, q, L)
        self._validate_scaling_parameters(contrast, offset)

        # Convert to JAX arrays if needed
        if jax_available:
            params = jnp.asarray(params)
            t1 = jnp.asarray(t1)
            t2 = jnp.asarray(t2)
            phi = jnp.asarray(phi)

        return self.model.compute_g2(params, t1, t2, phi, q, L, contrast, offset)

    @log_performance(threshold=0.05)
    def compute_chi_squared(
        self,
        params: np.ndarray,
        data: np.ndarray,
        sigma: np.ndarray,
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
        q: float,
        L: float,
        contrast: float,
        offset: float,
    ) -> float:
        """
        Compute chi-squared goodness of fit.

        Args:
            params: Physical parameters
            data: Experimental correlation data
            sigma: Measurement uncertainties
            t1, t2: Time grids
            phi: Angle grid
            q: Wave vector magnitude
            L: Sample-detector distance
            contrast, offset: Scaling parameters

        Returns:
            Chi-squared value
        """
        # Validate inputs
        self._validate_computation_inputs(params, q, L)
        self._validate_scaling_parameters(contrast, offset)
        self._validate_data_inputs(data, sigma, t1, t2, phi)

        # Convert to JAX arrays if needed
        if jax_available:
            params = jnp.asarray(params)
            data = jnp.asarray(data)
            sigma = jnp.asarray(sigma)
            t1 = jnp.asarray(t1)
            t2 = jnp.asarray(t2)
            phi = jnp.asarray(phi)

        return self.model.compute_chi_squared(
            params, data, sigma, t1, t2, phi, q, L, contrast, offset
        )

    @log_performance(threshold=0.1)
    def batch_computation(
        self,
        params_batch: np.ndarray,
        data: np.ndarray,
        sigma: np.ndarray,
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
        q: float,
        L: float,
        contrast: float,
        offset: float,
    ) -> np.ndarray:
        """
        Compute chi-squared for multiple parameter sets efficiently.

        Leverages JAX vectorization for optimal performance.

        Args:
            params_batch: Array of parameter sets (n_sets, n_params)
            data: Experimental correlation data
            sigma: Measurement uncertainties
            t1, t2: Time grids
            phi: Angle grid
            q: Wave vector magnitude
            L: Sample-detector distance
            contrast, offset: Scaling parameters

        Returns:
            Chi-squared values for each parameter set
        """
        # Validate batch input
        if params_batch.ndim != 2:
            raise ValueError("params_batch must be 2D array (n_sets, n_params)")

        n_sets, n_params = params_batch.shape
        if n_params != self.model.n_params:
            raise ValueError(
                f"Expected {self.model.n_params} parameters, got {n_params}"
            )

        logger.debug(f"Batch computation for {n_sets} parameter sets")

        # Convert to JAX arrays if needed
        if jax_available:
            params_batch = jnp.asarray(params_batch)
            data = jnp.asarray(data)
            sigma = jnp.asarray(sigma)
            t1 = jnp.asarray(t1)
            t2 = jnp.asarray(t2)
            phi = jnp.asarray(phi)

            return batch_chi_squared(
                params_batch, data, sigma, t1, t2, phi, q, L, contrast, offset
            )
        else:
            # Fallback: loop over parameter sets
            results = []
            for params in params_batch:
                chi2 = self.compute_chi_squared(
                    params, data, sigma, t1, t2, phi, q, L, contrast, offset
                )
                results.append(chi2)
            return np.array(results)

    def estimate_computation_cost(
        self, t1: np.ndarray, t2: np.ndarray, phi: np.ndarray
    ) -> Dict[str, Any]:
        """
        Estimate computational cost for given data dimensions.

        Helps with performance planning and memory management.

        Args:
            t1, t2: Time grids
            phi: Angle grid

        Returns:
            Cost estimation dictionary
        """
        n_time_pairs = len(t1) * len(t2)
        n_angles = len(phi)
        n_total_points = n_time_pairs * n_angles

        # Rough performance estimates (operations per point)
        ops_per_point = {
            "static_isotropic": 10,  # Diffusion only
            "static_anisotropic": 10,  # Diffusion only
            "laminar_flow": 50,  # Full model with shear
        }

        base_ops = ops_per_point.get(self.analysis_mode, 50)
        total_ops = n_total_points * base_ops

        # Memory estimates (bytes per point, rough)
        memory_per_point = 8 * 4  # ~4 float64 values per point
        total_memory_mb = (n_total_points * memory_per_point) / (1024**2)

        return {
            "n_time_pairs": n_time_pairs,
            "n_angles": n_angles,
            "n_total_points": n_total_points,
            "estimated_operations": total_ops,
            "estimated_memory_mb": total_memory_mb,
            "analysis_mode": self.analysis_mode,
            "backend": "JAX" if jax_available else "NumPy",
            "performance_tier": self._classify_performance_tier(total_ops),
        }

    def _classify_performance_tier(self, operations: int) -> str:
        """Classify computation as light, medium, or heavy."""
        if operations < 1e6:
            return "light"
        elif operations < 1e8:
            return "medium"
        else:
            return "heavy"

    def _validate_computation_inputs(self, params: np.ndarray, q: float, L: float):
        """Validate core computation inputs."""
        # Parameter validation
        if not self.model.validate_parameters(params):
            logger.warning(
                "Parameters outside recommended bounds - results may be unreliable"
            )

        # Experimental setup validation
        if q <= 0:
            raise ValueError(f"Wave vector q must be positive, got {q}")
        if L <= 0:
            raise ValueError(f"Sample-detector distance L must be positive, got {L}")

        # Physical reasonableness checks
        if not (PhysicsConstants.Q_MIN_TYPICAL <= q <= PhysicsConstants.Q_MAX_TYPICAL):
            logger.warning(
                f"q = {q:.2e} outside typical range - check experimental setup"
            )
        if not (10.0 <= L <= 10000.0):
            logger.warning(
                f"L = {L:.1f} mm outside typical range - check experimental setup"
            )

    def _validate_scaling_parameters(self, contrast: float, offset: float):
        """Validate scaling parameters."""
        if contrast <= 0:
            raise ValueError(f"Contrast must be positive, got {contrast}")
        if offset < 0:
            logger.warning(f"Negative offset {offset} - check baseline correction")

    def _validate_data_inputs(
        self,
        data: np.ndarray,
        sigma: np.ndarray,
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
    ):
        """Validate experimental data inputs."""
        # Shape consistency
        expected_shape = (len(phi), len(t1), len(t2))
        if data.shape != expected_shape:
            raise ValueError(
                f"Data shape {data.shape} doesn't match expected {expected_shape}"
            )
        if sigma.shape != expected_shape:
            raise ValueError(
                f"Sigma shape {sigma.shape} doesn't match expected {expected_shape}"
            )

        # Data quality checks
        if np.any(sigma <= 0):
            raise ValueError("All uncertainties must be positive")
        if np.any(~np.isfinite(data)):
            raise ValueError("Data contains non-finite values")
        if np.any(~np.isfinite(sigma)):
            raise ValueError("Uncertainties contain non-finite values")

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model and engine information."""
        info = self.model.get_model_info()
        info.update(
            {
                "theory_engine_version": "2.0",
                "backend_available": jax_available,
                "supports_batch_computation": jax_available,
            }
        )
        return info

    def __repr__(self) -> str:
        backend = "JAX" if jax_available else "NumPy"
        return f"TheoryEngine(mode='{self.analysis_mode}', backend={backend})"


# Convenience functions for direct computation
def compute_g2_theory(
    params: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    phi: np.ndarray,
    q: float,
    L: float,
    contrast: float,
    offset: float,
    analysis_mode: str = "laminar_flow",
) -> np.ndarray:
    """
    Direct computation of g2 theory with minimal overhead.

    Convenience function for simple theory calculations.

    Args:
        params: Physical parameters
        t1, t2: Time grids
        phi: Angle grid
        q: Wave vector magnitude
        L: Sample-detector distance
        contrast, offset: Scaling parameters
        analysis_mode: Analysis mode

    Returns:
        g2 correlation function
    """
    engine = TheoryEngine(analysis_mode)
    return engine.compute_g2(params, t1, t2, phi, q, L, contrast, offset)


def compute_chi2_theory(
    params: np.ndarray,
    data: np.ndarray,
    sigma: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    phi: np.ndarray,
    q: float,
    L: float,
    contrast: float,
    offset: float,
    analysis_mode: str = "laminar_flow",
) -> float:
    """
    Direct computation of chi-squared with minimal overhead.

    Args:
        params: Physical parameters
        data: Experimental data
        sigma: Uncertainties
        t1, t2: Time grids
        phi: Angle grid
        q: Wave vector magnitude
        L: Sample-detector distance
        contrast, offset: Scaling parameters
        analysis_mode: Analysis mode

    Returns:
        Chi-squared value
    """
    engine = TheoryEngine(analysis_mode)
    return engine.compute_chi_squared(
        params, data, sigma, t1, t2, phi, q, L, contrast, offset
    )


# Export main classes and functions
__all__ = [
    "TheoryEngine",
    "compute_g2_theory",
    "compute_chi2_theory",
]
