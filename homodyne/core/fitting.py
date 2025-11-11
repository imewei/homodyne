"""Unified Homodyne Model with JAX-Accelerated Least Squares
=========================================================

Core implementation of the scaled optimization process for homodyne analysis.
This is the central fitting engine that implements:

c2_fitted = c2_theory * contrast + offset

Where both VI+JAX and MCMC+JAX minimize: Exp - Fitted

Key Features:
- Pure least squares implementation (no outlier handling)
- JAX-accelerated computation with automatic differentiation
- Unified parameter space with specified bounds and priors
- Dataset size-aware optimization (<1M, 1-10M, >20M points)
- Mode-aware parameter management (3 vs 7 parameters)
- CPU-primary, GPU-optional architecture
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from homodyne.utils.logging import get_logger, log_performance

# JAX imports with fallback
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

    def jit(f):
        return f  # No-op decorator

    def vmap(f, **kwargs):
        return f

    def grad(f):
        return lambda x: np.zeros_like(x)


try:
    from homodyne.core.theory import TheoryEngine
except ImportError:
    logger = get_logger(__name__)
    logger.error("Could not import core modules - fitting engine disabled")

logger = get_logger(__name__)


@dataclass
class ParameterSpace:
    """Parameter space definition with bounds and priors.

    Implements specified parameter ranges and prior distributions
    for both scaling and physical parameters.
    Supports configuration-based bound override when config_manager is provided.
    """

    # Scaling parameters (always present)
    # ✅ FIXED (Nov 11, 2025): Updated bounds to match homodyne physics g₂ = 1 + β×g₁²
    # - contrast (β): Physical range [0, 1] where 0=no signal, 1=perfect contrast
    # - offset: Deviation from baseline=1.0, range [0.5, 2.5] allows ±50% variation
    contrast_bounds: tuple[float, float] = (0.0, 1.0)  # Physical contrast range
    offset_bounds: tuple[float, float] = (0.5, 2.5)    # Baseline ± significant deviation
    contrast_prior: tuple[float, float] = (0.5, 0.2)   # (mu, sigma) centered at 0.5, wider sigma
    offset_prior: tuple[float, float] = (1.0, 0.3)     # (mu, sigma) centered at 1.0, wider sigma

    # Physical parameter bounds (mode-dependent) - STANDARDIZED VALUES
    D0_bounds: tuple[float, float] = (1.0, 1000000.0)
    alpha_bounds: tuple[float, float] = (-2.0, 2.0)  # ✅ FIXED: Was (-10, 10)
    # Extreme values like alpha=-4.96 cause numerical underflow: exp(-q²*D₀*t^α*dt/2) → 0
    D_offset_bounds: tuple[float, float] = (
        -100000.0,
        100000.0,
    )  # Consistent with physics.py

    # Laminar flow parameters (only for laminar_flow mode) - STANDARDIZED VALUES
    gamma_dot_t0_bounds: tuple[float, float] = (1e-5, 1.0)
    beta_bounds: tuple[float, float] = (-2.0, 2.0)  # ✅ FIXED: Was (-10, 10)
    gamma_dot_t_offset_bounds: tuple[float, float] = (
        -1.0,
        1.0,
    )  # Consistent with physics.py
    phi0_bounds: tuple[float, float] = (-30.0, 30.0)  # Consistent with physics.py

    # Prior means (mu) and standard deviations (sigma) - STANDARDIZED VALUES
    D0_prior: tuple[float, float] = (10000.0, 1000.0)
    alpha_prior: tuple[float, float] = (-1.5, 0.1)
    D_offset_prior: tuple[float, float] = (0.0, 10.0)
    gamma_dot_t0_prior: tuple[float, float] = (0.001, 0.01)
    beta_prior: tuple[float, float] = (0.0, 0.1)
    gamma_dot_t_offset_prior: tuple[float, float] = (0.0, 0.001)
    phi0_prior: tuple[float, float] = (0.0, 5.0)

    # Data ranges
    fitted_range: tuple[float, float] = (0.0, 2.0)
    theory_range: tuple[float, float] = (0.0, 1.0)

    # Optional configuration manager for bound override
    config_manager: Any | None = None

    def get_param_bounds(self, analysis_mode: str) -> list[tuple[float, float]]:
        """Get parameter bounds based on analysis mode with configuration override support.

        Uses ParameterManager for consistent parameter handling and name mapping.

        Parameters
        ----------
        analysis_mode : str
            Analysis mode: "static_isotropic", "static_anisotropic", or "laminar_flow"

        Returns
        -------
        list of tuple
            List of (min, max) bounds tuples for each parameter
        """
        # Strategy 1: Use ParameterManager for full integration (Phase 4.2+)
        if self.config_manager:
            try:
                from homodyne.config.parameter_manager import ParameterManager

                # Get config dict from manager
                config_dict = None
                if hasattr(self.config_manager, "config"):
                    config_dict = self.config_manager.config
                elif isinstance(self.config_manager, dict):
                    config_dict = self.config_manager

                # Create ParameterManager
                param_manager = ParameterManager(config_dict, analysis_mode)

                # Get active parameters (physical only, excludes scaling)
                active_params = param_manager.get_active_parameters()

                # Get bounds as tuples
                bounds = param_manager.get_bounds_as_tuples(active_params)

                logger.info(
                    f"Loaded {len(bounds)} parameter bounds from ParameterManager for {analysis_mode} mode",
                )
                return bounds

            except Exception as e:
                logger.warning(
                    f"Failed to use ParameterManager: {e}, falling back to defaults",
                )

        # Fallback to hardcoded defaults
        logger.debug(f"Using default hardcoded bounds for {analysis_mode} mode")
        bounds = [
            self.D0_bounds,
            self.alpha_bounds,
            self.D_offset_bounds,
        ]

        if analysis_mode == "laminar_flow":
            bounds.extend(
                [
                    self.gamma_dot_t0_bounds,
                    self.beta_bounds,
                    self.gamma_dot_t_offset_bounds,
                    self.phi0_bounds,
                ],
            )

        return bounds

    def _get_default_bound_for_param(self, param_name: str) -> tuple[float, float]:
        """Get default bound for a specific parameter name."""
        bound_map = {
            "D0": self.D0_bounds,
            "alpha": self.alpha_bounds,
            "D_offset": self.D_offset_bounds,
            "gamma_dot_0": self.gamma_dot_t0_bounds,
            "beta": self.beta_bounds,
            "gamma_dot_offset": self.gamma_dot_t_offset_bounds,
            "phi_0": self.phi0_bounds,
        }
        return bound_map.get(param_name, (0.0, 1.0))  # Safe fallback

    def get_param_priors(self, analysis_mode: str) -> list[tuple[float, float]]:
        """Get parameter priors based on analysis mode."""
        priors = [
            self.D0_prior,
            self.alpha_prior,
            self.D_offset_prior,
        ]

        if analysis_mode == "laminar_flow":
            priors.extend(
                [
                    self.gamma_dot_t0_prior,
                    self.beta_prior,
                    self.gamma_dot_t_offset_prior,
                    self.phi0_prior,
                ],
            )

        return priors


class DatasetSize:
    """Dataset size categories for optimization."""

    SMALL = "small"  # <1M points
    MEDIUM = "medium"  # 1-10M points
    LARGE = "large"  # >20M points

    @staticmethod
    def categorize(data_size: int) -> str:
        """Categorize dataset size."""
        if data_size < 1_000_000:
            return DatasetSize.SMALL
        elif data_size < 10_000_000:
            return DatasetSize.MEDIUM
        else:
            return DatasetSize.LARGE


@dataclass
class FitResult:
    """Results from unified homodyne model fitting.

    Contains both physical and scaling parameters with
    comprehensive fit statistics for VI+JAX or MCMC+JAX.
    """

    # Optimized parameters
    params: np.ndarray  # Physical parameters
    contrast: float  # Contrast scaling parameter
    offset: float  # Offset parameter

    # Fit quality metrics
    chi_squared: float  # Chi-squared value
    reduced_chi_squared: float  # Reduced chi-squared
    degrees_of_freedom: int  # Degrees of freedom
    p_value: float  # P-value (if computed)

    # Parameter uncertainties (if computed)
    param_errors: np.ndarray | None = None
    contrast_error: float | None = None
    offset_error: float | None = None

    # Additional statistics
    residual_std: float = 0.0  # Standard deviation of residuals
    max_residual: float = 0.0  # Maximum absolute residual
    fit_iterations: int = 0  # Number of optimization iterations
    converged: bool = True  # Convergence flag

    # Computational metadata
    computation_time: float = 0.0  # Fitting time in seconds
    backend: str = "JAX" if JAX_AVAILABLE else "NumPy"
    dataset_size: str = "unknown"  # Dataset size category
    analysis_mode: str = "unknown"  # Analysis mode used

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive fit summary."""
        return {
            "parameters": {
                "physical": self.params.tolist(),
                "contrast": self.contrast,
                "offset": self.offset,
            },
            "errors": {
                "physical": (
                    self.param_errors.tolist()
                    if self.param_errors is not None
                    else None
                ),
                "contrast": self.contrast_error,
                "offset": self.offset_error,
            },
            "fit_quality": {
                "chi_squared": self.chi_squared,
                "reduced_chi_squared": self.reduced_chi_squared,
                "degrees_of_freedom": self.degrees_of_freedom,
                "p_value": self.p_value,
                "residual_std": self.residual_std,
                "max_residual": self.max_residual,
            },
            "convergence": {
                "converged": self.converged,
                "iterations": self.fit_iterations,
                "computation_time": self.computation_time,
                "backend": self.backend,
                "dataset_size": self.dataset_size,
                "analysis_mode": self.analysis_mode,
            },
        }


# JAX-accelerated least squares implementation
if JAX_AVAILABLE:

    @jit
    def solve_least_squares_jax(
        theory_batch: jnp.ndarray,
        exp_batch: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """JAX-accelerated batch least squares solver.

        Optimized least squares implementation with JAX acceleration
        with JAX acceleration for GPU/CPU optimization.

        Solves: min ||A*x - b||^2 where A = [theory, ones] for each angle.
        Model: c2_fitted = c2_theory * contrast + offset

        Args:
            theory_batch: Theory values, shape (n_angles, n_data_points)
            exp_batch: Experimental values, shape (n_angles, n_data_points)

        Returns:
            Tuple of (contrast_batch, offset_batch), each shape (n_angles,)
        """
        n_angles, n_data = theory_batch.shape

        # Vectorized computation of normal equation components
        sum_theory_sq = jnp.sum(
            theory_batch * theory_batch,
            axis=1,
        )  # shape: (n_angles,)
        sum_theory = jnp.sum(theory_batch, axis=1)  # shape: (n_angles,)
        sum_exp = jnp.sum(exp_batch, axis=1)  # shape: (n_angles,)
        sum_theory_exp = jnp.sum(theory_batch * exp_batch, axis=1)  # shape: (n_angles,)

        # Solve 2x2 system for each angle: AtA * x = Atb
        # [[sum_theory_sq, sum_theory], [sum_theory, n_data]] * [contrast, offset] = [sum_theory_exp, sum_exp]
        det = sum_theory_sq * n_data - sum_theory * sum_theory

        # Handle singular matrix cases
        valid_det = jnp.abs(det) > 1e-12
        safe_det = jnp.where(valid_det, det, 1.0)  # Avoid division by zero

        # Solve normal equations
        contrast = (n_data * sum_theory_exp - sum_theory * sum_exp) / safe_det
        offset = (sum_theory_sq * sum_exp - sum_theory * sum_theory_exp) / safe_det

        # Fallback for singular cases
        contrast = jnp.where(valid_det, contrast, 1.0)
        offset = jnp.where(valid_det, offset, 0.0)

        # Ensure contrast is positive (physical constraint)
        contrast = jnp.maximum(contrast, 1e-6)

        return contrast, offset

else:

    def solve_least_squares_jax(
        theory_batch: np.ndarray,
        exp_batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """NumPy fallback for least squares when JAX unavailable."""
        n_angles, n_data = theory_batch.shape
        contrast_batch = np.zeros(n_angles)
        offset_batch = np.zeros(n_angles)

        for i in range(n_angles):
            theory = theory_batch[i]
            exp = exp_batch[i]

            # Compute normal equation components
            sum_theory_sq = np.sum(theory * theory)
            sum_theory = np.sum(theory)
            sum_exp = np.sum(exp)
            sum_theory_exp = np.sum(theory * exp)

            # Solve 2x2 system
            det = sum_theory_sq * n_data - sum_theory * sum_theory
            if abs(det) > 1e-12:
                contrast_batch[i] = (
                    n_data * sum_theory_exp - sum_theory * sum_exp
                ) / det
                offset_batch[i] = (
                    sum_theory_sq * sum_exp - sum_theory * sum_theory_exp
                ) / det
                contrast_batch[i] = max(contrast_batch[i], 1e-6)  # Ensure positive
            else:
                contrast_batch[i] = 1.0
                offset_batch[i] = 0.0

        return contrast_batch, offset_batch


class UnifiedHomodyneEngine:
    """Unified homodyne fitting engine with JAX acceleration.

    Implements the scaled optimization approach where physical parameters
    are separated from experimental scaling parameters using pure least
    squares (no outlier handling - VI/MCMC handle uncertainty).
    """

    def __init__(
        self,
        analysis_mode: str = "laminar_flow",
        parameter_space: ParameterSpace | None = None,
    ):
        """Initialize unified homodyne engine.

        Args:
            analysis_mode: "static_isotropic", "static_anisotropic", or "laminar_flow"
            parameter_space: Parameter space definition (uses default if None)
        """
        self.analysis_mode = analysis_mode
        self.parameter_space = parameter_space or ParameterSpace()
        self.theory_engine = TheoryEngine(analysis_mode)

        # Get mode-specific parameter configuration
        self.param_bounds = self.parameter_space.get_param_bounds(analysis_mode)
        self.param_priors = self.parameter_space.get_param_priors(analysis_mode)

        logger.info(f"Unified homodyne engine initialized for {analysis_mode}")
        logger.info(f"Parameter count: {len(self.param_bounds)} physical + 2 scaling")
        logger.info(
            f"JAX acceleration: {'enabled' if JAX_AVAILABLE else 'disabled (NumPy fallback)'}",
        )

    @log_performance(threshold=0.1)
    def estimate_scaling_parameters(
        self,
        data: np.ndarray,
        theory: np.ndarray,
        validate_bounds: bool = True,
    ) -> tuple[float, float]:
        """Estimate contrast and offset using pure least squares.

        Uses JAX-accelerated least squares (no outlier handling).
        Both VI and MCMC will handle uncertainty through likelihood.

        Args:
            data: Experimental correlation data
            theory: Theoretical correlation (g1²)
            validate_bounds: Apply parameter space bounds validation

        Returns:
            Tuple of (contrast, offset)
        """
        # Prepare data for batch processing
        if data.ndim == 1 and theory.ndim == 1:
            # Single angle case - add batch dimension
            data_batch = data[np.newaxis, :]
            theory_batch = theory[np.newaxis, :]
        else:
            data_batch = data
            theory_batch = theory

        # Convert to JAX arrays if available
        if JAX_AVAILABLE:
            data_jax = jnp.array(data_batch)
            theory_jax = jnp.array(theory_batch)
        else:
            data_jax = data_batch
            theory_jax = theory_batch

        # Solve least squares
        contrast_batch, offset_batch = solve_least_squares_jax(theory_jax, data_jax)

        # Extract single values (average if multiple angles)
        if JAX_AVAILABLE:
            contrast = float(jnp.mean(contrast_batch))
            offset = float(jnp.mean(offset_batch))
        else:
            contrast = float(np.mean(contrast_batch))
            offset = float(np.mean(offset_batch))

        # Apply parameter space bounds if requested
        if validate_bounds:
            contrast = np.clip(contrast, *self.parameter_space.contrast_bounds)
            offset = np.clip(offset, *self.parameter_space.offset_bounds)

        logger.debug(
            f"Scaling parameters: contrast={contrast:.4f}, offset={offset:.4f}",
        )

        return contrast, offset

    @log_performance(threshold=1.0)
    def compute_likelihood(
        self,
        params: np.ndarray,
        contrast: float,
        offset: float,
        data: np.ndarray,
        sigma: np.ndarray,
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
        q: float,
        L: float,
    ) -> float:
        """Compute likelihood for unified homodyne model.

        This is the core likelihood function used by both VI+JAX and MCMC+JAX
        to minimize: Exp - (contrast * Theory + offset)

        Args:
            params: Physical parameters
            contrast, offset: Scaling parameters
            data: Experimental data
            sigma: Measurement uncertainties
            t1, t2, phi: Time and angle grids
            q, L: Experimental parameters

        Returns:
            Negative log-likelihood value
        """
        try:
            # Compute theoretical g1
            g1_theory = self.theory_engine.compute_g1(params, t1, t2, phi, q, L)
            g1_squared = g1_theory**2

            # Apply scaling: c2_fitted = c2_theory * contrast + offset
            theory_fitted = contrast * g1_squared + offset

            # Compute residuals: Exp - Fitted
            residuals = (data - theory_fitted) / sigma

            # Negative log-likelihood (Gaussian assumption)
            if JAX_AVAILABLE:
                chi_squared = jnp.sum(residuals**2)
                nll = 0.5 * chi_squared + 0.5 * jnp.sum(jnp.log(2 * jnp.pi * sigma**2))
                return float(nll)
            else:
                chi_squared = np.sum(residuals**2)
                nll = 0.5 * chi_squared + 0.5 * np.sum(np.log(2 * np.pi * sigma**2))
                return float(nll)

        except Exception as e:
            logger.warning(f"Likelihood computation failed: {e}")
            return 1e10  # Return large value on failure

    def detect_dataset_size(self, data: np.ndarray) -> str:
        """Detect and categorize dataset size with optimization recommendations."""
        size = data.size
        category = DatasetSize.categorize(size)

        # Calculate memory requirements
        memory_mb = (data.nbytes * 4) / (
            1024 * 1024
        )  # Factor of 4 for intermediate calculations

        logger.info(f"Dataset size: {size:,} points ({category})")
        logger.info(f"Estimated memory: {memory_mb:.1f} MB")

        # Log optimization strategy based on size
        if category == DatasetSize.SMALL:
            logger.info("Small dataset optimization:")
            logger.info("  - In-memory VI+JAX processing for instant fits")
            logger.info("  - Higher iteration counts for better convergence")
            logger.info("  - Full JAX acceleration without chunking")
        elif category == DatasetSize.MEDIUM:
            logger.info("Medium dataset optimization:")
            logger.info("  - Efficient batching with VI+JAX/MCMC+JAX")
            logger.info("  - Balanced iteration counts and memory usage")
            logger.info("  - Moderate chunking for memory efficiency")
        else:
            logger.info("Large dataset optimization:")
            logger.info("  - Distributed processing with intelligent chunking")
            logger.info("  - Conservative iteration counts to manage memory")
            logger.info("  - Progressive loading and compression")

        return category

    def validate_inputs(
        self,
        data: np.ndarray,
        sigma: np.ndarray | None,
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
        q: float,
        L: float,
    ):
        """Validate fitting inputs."""
        if data.size == 0:
            raise ValueError("Data array is empty")

        # Handle sigma validation only if sigma is provided
        if sigma is not None:
            if data.shape != sigma.shape:
                raise ValueError("Data and sigma must have same shape")
            if np.any(sigma <= 0):
                raise ValueError("All uncertainties must be positive")
            if not np.all(np.isfinite(sigma)):
                raise ValueError("Sigma contains non-finite values")

        if q <= 0 or L <= 0:
            raise ValueError("q and L must be positive")
        if not np.all(np.isfinite(data)):
            raise ValueError("Data contains non-finite values")

    def get_parameter_info(self) -> dict[str, Any]:
        """Get parameter space information."""
        return {
            "analysis_mode": self.analysis_mode,
            "parameter_count": len(self.param_bounds),
            "physical_bounds": self.param_bounds,
            "physical_priors": self.param_priors,
            "scaling_bounds": {
                "contrast": self.parameter_space.contrast_bounds,
                "offset": self.parameter_space.offset_bounds,
            },
            "scaling_priors": {
                "contrast": self.parameter_space.contrast_prior,
                "offset": self.parameter_space.offset_prior,
            },
            "data_ranges": {
                "fitted": self.parameter_space.fitted_range,
                "theory": self.parameter_space.theory_range,
            },
        }


ScaledFittingEngine = UnifiedHomodyneEngine

# Enhanced general N-parameter least squares solvers
if JAX_AVAILABLE:

    @jit
    def solve_least_squares_general_jax(
        design_matrix: jnp.ndarray,
        target_vector: jnp.ndarray,
        regularization: float = 1e-10,
    ) -> jnp.ndarray:
        """General N-parameter least squares solver using Normal Equation.

        Extends existing solve_least_squares_jax for arbitrary dimensions.
        Maintains compatibility with existing contrast/offset solver.

        Solves: min ||design_matrix * params - target_vector||²
        Via: (A^T A + λI) params = A^T b

        Args:
            design_matrix: Design matrix A, shape (n_samples, n_params)
            target_vector: Target vector b, shape (n_samples,)
            regularization: Ridge regularization parameter

        Returns:
            Solution vector, shape (n_params,)
        """
        # Compute Gram matrix
        gram_matrix = design_matrix.T @ design_matrix

        # Add regularization for numerical stability
        n_params = gram_matrix.shape[0]
        gram_matrix_reg = gram_matrix + regularization * jnp.eye(n_params)

        # Compute A^T b
        design_T_target = design_matrix.T @ target_vector

        # Check condition number for method selection
        eigenvalues = jnp.linalg.eigvalsh(gram_matrix_reg)
        condition_number = eigenvalues[-1] / (eigenvalues[0] + 1e-15)

        # Use appropriate solver based on conditioning
        def cholesky_solve():
            L = jnp.linalg.cholesky(gram_matrix_reg)
            z = jax.scipy.linalg.solve_triangular(L, design_T_target, lower=True)
            return jax.scipy.linalg.solve_triangular(L.T, z, lower=False)

        def svd_solve():
            return jnp.linalg.lstsq(design_matrix, target_vector, rcond=regularization)[
                0
            ]

        # Use Cholesky for well-conditioned, SVD for ill-conditioned
        params = jax.lax.cond(
            condition_number < 1e10,
            lambda _: cholesky_solve(),
            lambda _: svd_solve(),
            None,
        )

        return params

    @jit
    def solve_least_squares_chunked_jax(
        theory_chunks: jnp.ndarray,
        exp_chunks: jnp.ndarray,
        chunk_indices: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Memory-efficient chunked solver for large datasets.

        Extends existing solve_least_squares_jax with chunking support.
        Maintains model: c2_fitted = c2_theory * contrast + offset

        Args:
            theory_chunks: Theory values, shape (n_chunks, chunk_size)
            exp_chunks: Experimental values, shape (n_chunks, chunk_size)
            chunk_indices: Optional chunk indices for weighted processing

        Returns:
            Tuple of (contrast, offset)
        """

        # Process chunks using scan for memory efficiency
        def process_chunk(carry, chunk_data):
            theory_chunk, exp_chunk = chunk_data
            sum_theory_sq, sum_theory, sum_exp, sum_theory_exp, n_data = carry

            # Accumulate normal equation components
            chunk_size = theory_chunk.shape[0]
            sum_theory_sq += jnp.sum(theory_chunk * theory_chunk)
            sum_theory += jnp.sum(theory_chunk)
            sum_exp += jnp.sum(exp_chunk)
            sum_theory_exp += jnp.sum(theory_chunk * exp_chunk)
            n_data += chunk_size

            return (sum_theory_sq, sum_theory, sum_exp, sum_theory_exp, n_data), None

        # Initialize accumulators
        carry_init = (0.0, 0.0, 0.0, 0.0, 0)

        # Process all chunks
        (
            (
                sum_theory_sq_final,
                sum_theory_final,
                sum_exp_final,
                sum_theory_exp_final,
                n_data_final,
            ),
            _,
        ) = jax.lax.scan(process_chunk, carry_init, (theory_chunks, exp_chunks))

        # Solve 2x2 system (maintaining existing logic)
        det = sum_theory_sq_final * n_data_final - sum_theory_final * sum_theory_final

        # Handle singular matrix cases
        valid_det = jnp.abs(det) > 1e-12
        safe_det = jnp.where(valid_det, det, 1.0)

        # Solve for contrast and offset
        contrast = (
            n_data_final * sum_theory_exp_final - sum_theory_final * sum_exp_final
        ) / safe_det
        offset = (
            sum_theory_sq_final * sum_exp_final
            - sum_theory_final * sum_theory_exp_final
        ) / safe_det

        # Apply constraints
        contrast = jnp.where(valid_det, contrast, 1.0)
        offset = jnp.where(valid_det, offset, 0.0)
        contrast = jnp.maximum(contrast, 1e-6)

        return contrast, offset

else:
    # NumPy fallback versions
    def solve_least_squares_general_jax(
        design_matrix: np.ndarray,
        target_vector: np.ndarray,
        regularization: float = 1e-10,
    ) -> np.ndarray:
        """NumPy fallback for general least squares."""
        # Use numpy.linalg.lstsq as fallback
        solution, _, _, _ = np.linalg.lstsq(
            design_matrix,
            target_vector,
            rcond=regularization,
        )
        return solution

    def solve_least_squares_chunked_jax(
        theory_chunks: np.ndarray,
        exp_chunks: np.ndarray,
        chunk_indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """NumPy fallback for chunked least squares."""
        # Accumulate normal equation components
        sum_theory_sq = 0.0
        sum_theory = 0.0
        sum_exp = 0.0
        sum_theory_exp = 0.0
        n_data = 0

        for theory_chunk, exp_chunk in zip(theory_chunks, exp_chunks, strict=False):
            sum_theory_sq += np.sum(theory_chunk * theory_chunk)
            sum_theory += np.sum(theory_chunk)
            sum_exp += np.sum(exp_chunk)
            sum_theory_exp += np.sum(theory_chunk * exp_chunk)
            n_data += theory_chunk.shape[0]

        # Solve 2x2 system
        det = sum_theory_sq * n_data - sum_theory * sum_theory

        if abs(det) > 1e-12:
            contrast = (n_data * sum_theory_exp - sum_theory * sum_exp) / det
            offset = (sum_theory_sq * sum_exp - sum_theory * sum_theory_exp) / det
            contrast = max(contrast, 1e-6)
        else:
            contrast = 1.0
            offset = 0.0

        return contrast, offset


# Export main classes
__all__ = [
    "FitResult",
    "ParameterSpace",
    "DatasetSize",
    "UnifiedHomodyneEngine",
    "ScaledFittingEngine",
    "solve_least_squares_jax",
    "solve_least_squares_general_jax",
    "solve_least_squares_chunked_jax",
]
