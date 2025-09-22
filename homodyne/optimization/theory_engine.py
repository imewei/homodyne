"""
Unified Theory Computation Engine
==================================

Centralized theory computation for all optimization methods (LSQ, VI, MCMC).
Eliminates duplication and ensures consistency across methods.
"""

from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass

from homodyne.core.fitting import UnifiedHomodyneEngine
from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TheoryComputationConfig:
    """Configuration for theory computation."""
    engine_type: str = "numpy"  # "numpy" or "jax"
    dt: float = 0.1
    return_g1: bool = False
    return_phi_values: bool = False
    batch_size: Optional[int] = None
    use_sampling: bool = False
    sampling_indices: Optional[np.ndarray] = None


class UnifiedTheoryEngine:
    """
    Unified theory computation engine for all optimization methods.

    This class consolidates theory computation logic that was previously
    duplicated across LSQ, VI, and MCMC modules.
    """

    def __init__(self, config: Optional[TheoryComputationConfig] = None):
        """Initialize the theory engine."""
        self.config = config or TheoryComputationConfig()
        self.engine = None
        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize the computation engine based on configuration."""
        try:
            if self.config.engine_type == "jax":
                from homodyne.core.jax_backend import HeterodyneFitting
                self.engine = HeterodyneFitting()
            else:
                self.engine = UnifiedHomodyneEngine()
        except ImportError:
            logger.warning(f"Could not import {self.config.engine_type} backend, using numpy")
            self.engine = UnifiedHomodyneEngine()

    def compute_theory(
        self,
        params: Union[np.ndarray, Dict[str, float]],
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
        q: Union[float, np.ndarray],
        L: float,
        analysis_mode: str = "laminar_flow",
        contrast: Optional[float] = None,
        offset: Optional[float] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute theoretical correlation function.

        Args:
            params: Physics parameters (array or dict)
            t1, t2: Time grids
            phi: Angular positions
            q: Scattering vector magnitude
            L: System size
            analysis_mode: Analysis mode
            contrast: Optional contrast parameter
            offset: Optional offset parameter

        Returns:
            Theoretical correlation (g2 or c2), optionally with g1
        """
        # Convert params to dict if needed
        if isinstance(params, np.ndarray):
            params = self._array_to_dict(params, analysis_mode)

        # Extract contrast and offset if not provided
        if contrast is None:
            contrast = params.get("contrast", 1.0)
        if offset is None:
            offset = params.get("offset", 0.0)

        # Handle sampling if configured
        if self.config.use_sampling and self.config.sampling_indices is not None:
            return self._compute_sampled(
                params, t1, t2, phi, q, L, analysis_mode, contrast, offset
            )

        # Compute full theory
        physics_params = self._extract_physics_params(params, analysis_mode)

        # Compute g1
        g1 = self.engine.compute_g1_hetero(
            t1, t2, phi, q, L,
            analysis_mode=analysis_mode,
            dt=self.config.dt,
            **physics_params
        )

        # Apply unified model: c2 = |g1|^2 * contrast + offset
        c2_theory = np.abs(g1)**2 * contrast + offset

        if self.config.return_g1:
            return c2_theory, g1
        return c2_theory

    def _compute_sampled(
        self,
        params: Dict[str, float],
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
        q: Union[float, np.ndarray],
        L: float,
        analysis_mode: str,
        contrast: float,
        offset: float,
    ) -> np.ndarray:
        """Compute theory only at sampled points."""
        indices = self.config.sampling_indices

        # Extract sampled coordinates
        i1_sampled = indices[:, 0]
        i2_sampled = indices[:, 1]

        # Get physics parameters
        physics_params = self._extract_physics_params(params, analysis_mode)

        # Compute theory at sampled points
        t1_sampled = t1[i1_sampled]
        t2_sampled = t2[i2_sampled]
        phi_sampled = phi[i1_sampled] if phi.ndim == 1 else phi[i1_sampled, i2_sampled]

        # Handle q array if needed
        if isinstance(q, np.ndarray) and q.size > 1:
            q_sampled = q[i1_sampled] if q.ndim == 1 else q
        else:
            q_sampled = q

        # Compute g1 at sampled points
        g1_sampled = self.engine.compute_g1_hetero(
            t1_sampled, t2_sampled, phi_sampled, q_sampled, L,
            analysis_mode=analysis_mode,
            dt=self.config.dt,
            **physics_params
        )

        # Apply model
        return np.abs(g1_sampled)**2 * contrast + offset

    def _array_to_dict(self, params: np.ndarray, analysis_mode: str) -> Dict[str, float]:
        """Convert parameter array to dictionary."""
        param_names = self._get_param_names(analysis_mode)
        return {name: params[i] for i, name in enumerate(param_names)}

    def _get_param_names(self, analysis_mode: str) -> list:
        """Get parameter names for analysis mode."""
        if analysis_mode == "laminar_flow":
            return ["D0", "alpha", "D_offset", "gamma_dot_0", "beta", "gamma_dot_offset", "phi_0", "contrast", "offset"]
        elif analysis_mode in ["static_isotropic", "static_anisotropic"]:
            return ["D0", "alpha", "D_offset", "contrast", "offset"]
        else:
            raise ValueError(f"Unknown analysis mode: {analysis_mode}. Supported modes: laminar_flow, static_isotropic, static_anisotropic")

    def _extract_physics_params(
        self, params: Dict[str, float], analysis_mode: str
    ) -> Dict[str, float]:
        """Extract physics parameters from full parameter dict."""
        # Base parameters common to all modes
        physics_params = {
            "D0": params.get("D0", 0.0),
            "alpha": params.get("alpha", 0.0),
            "D_offset": params.get("D_offset", 0.0)
        }

        # Add mode-specific parameters
        if analysis_mode == "laminar_flow":
            physics_params.update({
                "gamma_dot_0": params.get("gamma_dot_0", 0.0),
                "beta": params.get("beta", 0.0),
                "gamma_dot_offset": params.get("gamma_dot_offset", 0.0),
                "phi_0": params.get("phi_0", 0.0)
            })
        elif analysis_mode in ["static_isotropic", "static_anisotropic"]:
            # Static modes only use base parameters (D0, alpha, D_offset)
            pass
        else:
            raise ValueError(f"Unknown analysis mode: {analysis_mode}")

        return physics_params

    def compute_chi_squared(
        self,
        params: Union[np.ndarray, Dict[str, float]],
        data: np.ndarray,
        sigma: np.ndarray,
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
        q: Union[float, np.ndarray],
        L: float,
        analysis_mode: str = "laminar_flow",
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute chi-squared value for given parameters.

        This centralizes chi-squared computation logic that was duplicated
        across optimization methods.
        """
        # Compute theory
        theory = self.compute_theory(params, t1, t2, phi, q, L, analysis_mode)

        # Compute residuals
        if self.config.use_sampling and self.config.sampling_indices is not None:
            # Use sampled data
            indices = self.config.sampling_indices
            data_sampled = data.ravel()[indices[:, 0] * data.shape[1] + indices[:, 1]]
            sigma_sampled = sigma.ravel()[indices[:, 0] * sigma.shape[1] + indices[:, 1]]
            residuals = (data_sampled - theory) / sigma_sampled
        else:
            # Use full data
            residuals = (data - theory) / sigma

        # Apply weights if provided
        if weights is not None:
            residuals *= np.sqrt(weights)

        # Return chi-squared
        return np.sum(residuals**2)

    def compute_log_likelihood(
        self,
        params: Union[np.ndarray, Dict[str, float]],
        data: np.ndarray,
        sigma: np.ndarray,
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
        q: Union[float, np.ndarray],
        L: float,
        analysis_mode: str = "laminar_flow",
    ) -> float:
        """
        Compute log-likelihood for given parameters.

        Used by VI and MCMC methods.
        """
        chi2 = self.compute_chi_squared(
            params, data, sigma, t1, t2, phi, q, L, analysis_mode
        )

        # Log-likelihood assuming Gaussian errors
        # L = -0.5 * chi2 - 0.5 * n * log(2*pi) - sum(log(sigma))
        n = data.size if not self.config.use_sampling else len(self.config.sampling_indices)
        log_likelihood = -0.5 * chi2

        # Add normalization terms if needed
        if self.config.use_sampling and self.config.sampling_indices is not None:
            indices = self.config.sampling_indices
            sigma_sampled = sigma.ravel()[indices[:, 0] * sigma.shape[1] + indices[:, 1]]
            log_likelihood -= np.sum(np.log(sigma_sampled))
        else:
            log_likelihood -= np.sum(np.log(sigma))

        return log_likelihood


def create_theory_engine(
    engine_type: str = "numpy",
    use_sampling: bool = False,
    sampling_indices: Optional[np.ndarray] = None,
    **kwargs
) -> UnifiedTheoryEngine:
    """
    Factory function to create a theory engine.

    Args:
        engine_type: "numpy" or "jax"
        use_sampling: Whether to use sampling
        sampling_indices: Indices for sampling
        **kwargs: Additional configuration

    Returns:
        Configured UnifiedTheoryEngine instance
    """
    config = TheoryComputationConfig(
        engine_type=engine_type,
        use_sampling=use_sampling,
        sampling_indices=sampling_indices,
        **kwargs
    )
    return UnifiedTheoryEngine(config)