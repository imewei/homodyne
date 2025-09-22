"""
LSQ Result Wrapper for CLI Integration
======================================

Provides a unified interface for Direct Least Squares results,
compatible with the homodyne CLI result display system.
Supports noise estimation integration.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
import time

from homodyne.core.fitting import FitResult, UnifiedHomodyneEngine
from homodyne.optimization.direct_solver import DirectLeastSquaresSolver, DirectSolverConfig
from homodyne.utils.logging import get_logger
from homodyne.core.jax_backend import safe_len

# Import JAX with proper fallback
try:
    import jax
    import jax.numpy as jnp
    from jax import jit
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np
    def jit(f):
        return f

logger = get_logger(__name__)


@dataclass
class LSQResult:
    """
    Results from Direct Classical Least Squares fitting.

    Wraps DirectSolverResult with additional fields for CLI compatibility
    and noise estimation support.
    """

    # Core fitting results (required fields)
    mean_params: np.ndarray  # Physical parameters (from direct solver)
    mean_contrast: float  # Contrast parameter
    mean_offset: float  # Offset parameter

    # Fit quality metrics (required fields)
    chi_squared: float  # Chi-squared value
    reduced_chi_squared: float  # Reduced chi-squared
    residual_std: float  # Standard deviation of residuals
    max_residual: float  # Maximum residual value
    degrees_of_freedom: int  # Degrees of freedom

    # Optimization metadata (required fields)
    converged: bool  # Always True for direct solution
    n_iterations: int  # Always 1 for direct solution
    computation_time: float  # Total computation time
    backend: str  # "JAX" or "NumPy"
    dataset_size: str  # "SMALL", "MEDIUM", or "LARGE"
    analysis_mode: str  # Analysis mode used

    # No uncertainties for direct LSQ (optional fields with defaults)
    std_params: Optional[np.ndarray] = None  # Not available in LSQ
    std_contrast: Optional[float] = None  # Not available in LSQ
    std_offset: Optional[float] = None  # Not available in LSQ

    # Noise estimation fields (optional fields with defaults)
    noise_estimated: bool = False  # Whether noise was estimated
    noise_model: Optional[str] = None  # "hierarchical", "per_angle", "adaptive"
    estimated_sigma: Optional[np.ndarray] = None  # Estimated noise values
    noise_params: Optional[Dict[str, Any]] = None  # Additional noise parameters

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive LSQ result summary for display."""
        summary = {
            "point_estimates": {
                "parameters": self.mean_params.tolist(),
                "contrast": self.mean_contrast,
                "offset": self.mean_offset,
            },
            "uncertainties": {
                "parameters": None,  # LSQ doesn't provide uncertainties
                "contrast": None,
                "offset": None,
                "note": "Direct least squares does not provide uncertainty estimates"
            },
            "optimization": {
                "converged": self.converged,
                "iterations": self.n_iterations,
                "computation_time": self.computation_time,
                "backend": self.backend,
                "dataset_size": self.dataset_size,
                "analysis_mode": self.analysis_mode,
            },
            "fit_quality": {
                "chi_squared": self.chi_squared,
                "reduced_chi_squared": self.reduced_chi_squared,
                "residual_std": self.residual_std,
                "max_residual": self.max_residual,
                "degrees_of_freedom": self.degrees_of_freedom,
            },
        }

        # Add noise estimation info if available
        if self.noise_estimated:
            summary["noise_estimation"] = {
                "estimated": True,
                "model": self.noise_model,
                "mean_sigma": float(np.mean(self.estimated_sigma)) if self.estimated_sigma is not None else None,
                "parameters": self.noise_params,
            }

        return summary


def fit_homodyne_lsq(
    data: np.ndarray,
    sigma: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    phi: np.ndarray,
    q: np.ndarray,
    L: float,
    analysis_mode: str = "laminar_flow",
    estimate_noise: bool = False,
    noise_model: str = "hierarchical",
    config: Optional[DirectSolverConfig] = None,
    **kwargs
) -> LSQResult:
    """
    Fit homodyne data using Nonlinear Least Squares optimization.

    This is the main LSQ fitting function for CLI integration.
    Optimizes ALL parameters (physics + scaling) using chi-squared minimization.

    Args:
        data: Experimental C2 data
        sigma: Data uncertainties (may be None if estimate_noise=True)
        t1, t2: Time arrays
        phi: Angle array
        q: Q-vector
        L: Sample-to-detector distance
        analysis_mode: Analysis mode ("static_isotropic", "static_anisotropic", "laminar_flow")
        estimate_noise: Whether to estimate noise from residuals
        noise_model: Noise model to use if estimating
        config: Optional solver configuration
        **kwargs: Additional parameters passed to solver

    Returns:
        LSQResult with fitted parameters and optional noise estimates
    """
    start_time = time.time()

    # Import optimization libraries
    try:
        from scipy.optimize import minimize
        SCIPY_AVAILABLE = True
    except ImportError:
        logger.warning("SciPy not available for LSQ optimization, using simplified approach")
        SCIPY_AVAILABLE = False

    # Initialize the unified homodyne engine for theory computation
    from homodyne.core.fitting import UnifiedHomodyneEngine
    from homodyne.core.models import create_model

    # Create the model object for later use if needed
    model = create_model(analysis_mode)

    # UnifiedHomodyneEngine expects analysis_mode string, not model object
    engine = UnifiedHomodyneEngine(analysis_mode)

    # Extract initial physics parameters from kwargs or use default
    if "initial_params" in kwargs:
        initial_physics_params = np.array(kwargs["initial_params"])
    else:
        # Use engine's parameter space for initial guess
        initial_physics_params = engine.parameter_space.get_initial_guess()

    logger.info(f"Starting LSQ optimization with {len(initial_physics_params)} physics parameters")

    # Handle noise estimation
    if estimate_noise:
        logger.info(f"Estimating noise using {noise_model} model")
        if noise_model == "hierarchical":
            estimated_sigma = np.full_like(data.flatten(), np.std(data) * 0.1)
            noise_params = {"global_sigma": float(np.std(data) * 0.1)}
        else:
            estimated_sigma = np.full_like(data.flatten(), np.std(data) * 0.1)
            noise_params = {"global_sigma": float(np.std(data) * 0.1)}
        sigma_used = estimated_sigma.reshape(data.shape)
    else:
        estimated_sigma = None
        noise_params = None
        sigma_used = sigma

    # Prepare data
    data_flat = data.flatten()
    if sigma_used is not None:
        sigma_flat = sigma_used.flatten()
        weights = 1.0 / (sigma_flat + 1e-10)
    else:
        weights = np.ones_like(data_flat)

    # CRITICAL FIX: Use configuration dt value directly
    # During LSQ optimization, t1/t2 are correlation matrices, not time arrays.
    # Computing dt from correlation matrix differences produces dt=0.0, causing
    # sinc_prefactor=0 in JAX backend, leading to constant g1=1.0 values.
    dt = kwargs.get('dt', 0.1)  # Use dt from configuration, fallback to 0.1

    logger.debug(f"Using dt parameter from configuration: {dt} (LSQ optimization uses correlation matrices, not time arrays)")

    # Define the chi-square objective function with proper NumPy/JAX boundary
    # Add evaluation counter
    eval_counter = [0]

    def chi_square_objective_numpy(params):
        """NumPy interface for scipy.optimize - handles all conversions.

        This function is called by scipy.optimize with NumPy arrays.
        It ensures proper type conversion and handles all boundary logic.
        """
        eval_counter[0] += 1

        # Ensure params is always a NumPy 1D array (scipy may pass various types)
        params = np.atleast_1d(np.asarray(params))

        # Split parameters into physics and scaling
        physics_params = params[:-2]
        contrast = float(params[-2])  # Ensure scalar
        offset = float(params[-1])    # Ensure scalar

        # Log every 10th evaluation for debugging
        if eval_counter[0] % 10 == 1:
            logger.debug(f"Eval #{eval_counter[0]}: params={params[:3]}..., contrast={contrast:.4f}, offset={offset:.4f}")

        # Basic bounds validation
        if contrast <= 0 or contrast > 10:
            return 1e10
        if abs(offset) > 100:
            return 1e10

        try:
            # Call the actual computation (JAX or NumPy)
            if JAX_AVAILABLE:
                chi2 = compute_chi2_jax(
                    physics_params, data_flat, weights,
                    t1, t2, phi, q, L, contrast, offset, engine, dt
                )
                # Log chi2 value for debugging
                if eval_counter[0] % 10 == 1:
                    logger.debug(f"  -> chi2={chi2:.6f}")
                return float(chi2)  # Ensure Python float for scipy
            else:
                chi2 = compute_chi2_numpy(
                    physics_params, data_flat, weights,
                    t1, t2, phi, q, L, contrast, offset, engine, dt
                )
                if eval_counter[0] % 10 == 1:
                    logger.debug(f"  -> chi2={chi2:.6f}")
                return float(chi2)

        except Exception as e:
            logger.debug(f"Objective function evaluation failed: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return 1e10

    # Define JAX-optimized computation if available
    if JAX_AVAILABLE:
        # Only JIT the chi-squared calculation, not the engine calls
        @jit
        def compute_chi2_jax_inner(theory_flat, data_flat, weights):
            """JAX JIT-compiled chi-squared calculation.

            This function only performs the chi-squared calculation.
            Theory computation happens outside the JIT boundary.
            """
            residuals = (theory_flat - data_flat) * weights
            return jnp.sum(residuals ** 2)

        def compute_chi2_jax(physics_params, data_flat, weights,
                             t1, t2, phi, q, L, contrast, offset, engine, dt):
            """JAX-optimized chi-squared computation.

            Computes theory outside JIT, then uses JIT for chi2 calculation.
            """
            # Compute theory outside JIT boundary
            theory_flat = compute_theory_flat(
                physics_params, t1, t2, phi, q, L, contrast, offset, engine, dt
            )
            # Use JIT-compiled function for chi-squared calculation
            return compute_chi2_jax_inner(theory_flat, data_flat, weights)

    def compute_theory_flat(physics_params, t1, t2, phi, q, L, contrast, offset, engine, dt=0.1):
        """Compute flattened theory using the engine.

        This function handles the engine calls and cannot be JIT-compiled.
        Returns a flattened array of theory values.
        """
        # Ensure physics_params is properly shaped
        physics_params = np.atleast_1d(np.asarray(physics_params))

        # Debug: Log parameters occasionally
        if eval_counter[0] % 50 == 1:
            logger.debug(f"compute_theory_flat called with physics_params={physics_params[:3]}...")

        # Process phi array: ensure it's NumPy for safe indexing
        phi_array = np.atleast_1d(np.asarray(phi))
        phi_size = len(phi_array)

        # Create time meshgrids if needed for proper correlation computation
        if t1.ndim == 1 and t2.ndim == 1:
            t2_grid, t1_grid = np.meshgrid(t2, t1, indexing='ij')
        else:
            t1_grid = t1
            t2_grid = t2

        if phi_size > 1:
            # Multiple angles: compute g1 for each angle separately
            g1_list = []
            for i in range(phi_size):
                phi_val = float(phi_array[i])  # Safe indexing with NumPy array
                # Pass phi as single-element array, not scalar
                g1_single = engine.theory_engine.compute_g1(
                    params=physics_params, t1=t1_grid, t2=t2_grid,
                    phi=np.array([phi_val]), q=q, L=L, dt=dt
                )
                # Debug: Log g1 shape and values
                if eval_counter[0] % 50 == 1 and i == 0:
                    logger.debug(f"  g1_single shape before squeeze: {g1_single.shape}")
                    logger.debug(f"  g1_single min/max/mean: {np.min(g1_single):.6f}/{np.max(g1_single):.6f}/{np.mean(g1_single):.6f}")

                # Squeeze to remove singleton phi dimension: (1, t1, t2) -> (t1, t2)
                g1_single = np.squeeze(g1_single, axis=0) if g1_single.shape[0] == 1 else g1_single
                g1_list.append(g1_single)
            # Stack along phi dimension and compute g2
            g1_full = np.stack(g1_list, axis=0)
            g2_theory = offset + contrast * g1_full ** 2
        else:
            # Single angle
            phi_val = float(phi_array[0])
            # Pass phi as single-element array, not scalar
            g1_theory = engine.theory_engine.compute_g1(
                params=physics_params, t1=t1_grid, t2=t2_grid,
                phi=np.array([phi_val]), q=q, L=L, dt=dt
            )
            # Squeeze to remove singleton phi dimension if present
            g1_theory = np.squeeze(g1_theory, axis=0) if g1_theory.shape[0] == 1 else g1_theory
            g2_theory = offset + contrast * g1_theory ** 2

        # Debug: Log g2 statistics before flattening
        if eval_counter[0] % 50 == 1:
            logger.debug(f"  g2_theory shape: {g2_theory.shape}")
            logger.debug(f"  g2_theory min/max/mean: {np.min(g2_theory):.6f}/{np.max(g2_theory):.6f}/{np.mean(g2_theory):.6f}")
            theory_flat = g2_theory.flatten()
            logger.debug(f"  theory_flat[:10]: {theory_flat[:10]}")
            return theory_flat

        # Flatten the theory
        return g2_theory.flatten()

    def compute_chi2_numpy(physics_params, data_flat, weights,
                           t1, t2, phi, q, L, contrast, offset, engine, dt):
        """NumPy chi-squared computation (fallback when JAX not available).

        Computes theory and chi-squared using NumPy.
        """
        # Compute theory
        theory_flat = compute_theory_flat(
            physics_params, t1, t2, phi, q, L, contrast, offset, engine, dt
        )
        # Compute chi-squared
        residuals = (theory_flat - data_flat) * weights
        return np.sum(residuals ** 2)

    # Set up initial parameters: physics + contrast + offset
    initial_contrast = kwargs.get("initial_contrast", 0.3)
    initial_offset = kwargs.get("initial_offset", 1.0)
    initial_params = np.concatenate([initial_physics_params, [initial_contrast, initial_offset]])

    # Evaluate initial objective value for debugging
    initial_chi_sq = chi_square_objective_numpy(initial_params)
    logger.info(f"Initial chi-squared: {initial_chi_sq:.6f}")
    logger.info(f"Initial parameters: physics={initial_physics_params}, contrast={initial_contrast}, offset={initial_offset}")

    # Perform optimization
    if SCIPY_AVAILABLE:
        logger.info("Using SciPy Nelder-Mead optimization as primary method")

        # Try Nelder-Mead first (doesn't use gradients, more robust for flat regions)
        result = minimize(
            chi_square_objective_numpy,
            initial_params,
            method='Nelder-Mead',
            options={
                'maxiter': 2000,
                'xatol': 1e-4,
                'fatol': 1e-4,
                'disp': True
            }
        )

        # Check if Nelder-Mead succeeded
        if result.success and hasattr(result, 'nit') and result.nit > 0:
            logger.info(f"Nelder-Mead optimization completed successfully in {result.nit} iterations")
            optimized_params = result.x
            final_chi_sq = result.fun
            converged = result.success
            n_iterations = result.nit
        else:
            # First fallback: L-BFGS-B
            logger.warning("Nelder-Mead failed or didn't converge. Trying L-BFGS-B as fallback...")

            # Set parameter bounds for L-BFGS-B
            bounds = []
            # Physics parameter bounds - use reasonable bounds based on typical values
            for i, param in enumerate(initial_physics_params):
                if i == 0:  # First parameter (typically viscosity or similar)
                    bounds.append((param * 0.1, param * 10.0))  # Allow 10x range
                else:
                    # For other parameters, allow wider range including sign changes
                    bounds.append((param - abs(param) * 10, param + abs(param) * 10))
            # Scaling parameter bounds
            bounds.append((0.001, 10.0))  # contrast bounds
            bounds.append((-10.0, 10.0))  # offset bounds

            result = minimize(
                chi_square_objective_numpy,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 1000,
                    'ftol': 1e-6,  # Relax function tolerance
                    'gtol': 1e-5,  # Add gradient tolerance
                    'disp': True   # Enable verbose output
                }
            )

            if result.success or (hasattr(result, 'nit') and result.nit > 0):
                logger.info(f"L-BFGS-B optimization completed in {result.nit} iterations")
                optimized_params = result.x
                final_chi_sq = result.fun
                converged = result.success
                n_iterations = result.nit
            else:
                # Second fallback: Powell method
                logger.warning(f"L-BFGS-B optimization failed: {result.message if hasattr(result, 'message') else 'Unknown error'}")
                logger.info("Trying Powell method as final fallback...")
                result = minimize(
                    chi_square_objective_numpy,
                    initial_params,
                    method='Powell',
                    options={'maxiter': 1000, 'disp': True}
                )
                if hasattr(result, 'x'):
                    optimized_params = result.x
                    final_chi_sq = result.fun
                    converged = result.success
                    n_iterations = result.nit if hasattr(result, 'nit') else 1
                else:
                    optimized_params = initial_params
                    final_chi_sq = chi_square_objective_numpy(initial_params)
                    converged = False
                    n_iterations = 0
    else:
        # Fallback: just evaluate at initial parameters
        logger.info("Using initial parameters (no optimization)")
        optimized_params = initial_params
        final_chi_sq = chi_square_objective_numpy(initial_params)
        converged = True
        n_iterations = 1

    # Extract optimized parameters
    physics_params = optimized_params[:-2]
    contrast = optimized_params[-2]
    offset = optimized_params[-1]

    # Compute final statistics
    # Create time meshgrids if needed for proper correlation computation
    if t1.ndim == 1 and t2.ndim == 1:
        t2_grid, t1_grid = np.meshgrid(t2, t1, indexing='ij')
        dt_val = t1[1] - t1[0] if safe_len(t1) > 1 else 1.0
    else:
        t1_grid = t1
        t2_grid = t2
        # Extract 1D array from grid to calculate dt
        t1_1d = t1_grid[0, :] if t1_grid.ndim == 2 else t1_grid.flatten()[:min(100, safe_len(t1_grid.flatten()))]
        dt_val = t1_1d[1] - t1_1d[0] if safe_len(t1_1d) > 1 else 1.0

    # Process phi array: convert to array if scalar, loop through angles if array
    phi_array = np.atleast_1d(phi)  # Ensure phi is at least 1D array
    if phi_array.size > 1:
        # Multiple angles: compute g1 for each angle separately
        g1_list = []
        for phi_val in phi_array:
            g1_single = engine.theory_engine.compute_g1(
                physics_params, t1_grid, t2_grid, np.array([float(phi_val)]), q, L, dt=dt_val
            )
            # Remove extra dimension if present
            if g1_single.ndim > 2:
                g1_single = g1_single.squeeze()
            g1_list.append(g1_single)
        # Stack results along first dimension
        g1_final = np.stack(g1_list, axis=0)
    else:
        # Single angle
        phi_val = float(phi_array[0])
        g1_final = engine.theory_engine.compute_g1(
            physics_params, t1_grid, t2_grid, np.array([phi_val]), q, L, dt=dt_val
        )
        # Ensure proper shape
        if g1_final.ndim > 2 and g1_final.shape[0] == 1:
            g1_final = g1_final.squeeze(axis=0)

    theory_final = g1_final**2

    # Ensure theory has proper shape before flattening
    if theory_final.ndim > 2:
        # If theory has shape (n_phi, n_t1, n_t2), reshape appropriately
        theory_final_flat = theory_final.reshape(-1)
    else:
        theory_final_flat = theory_final.flatten()

    theory_scaled_final = theory_final_flat * contrast + offset
    residuals_final = data_flat - theory_scaled_final

    # Calculate proper degrees of freedom
    n_data_points = len(data_flat)
    n_parameters = len(optimized_params)  # All fitted parameters
    degrees_of_freedom = n_data_points - n_parameters

    # Calculate chi-squared and reduced chi-squared
    if sigma_used is not None:
        chi_squared = final_chi_sq
    else:
        chi_squared = np.sum(residuals_final**2)

    reduced_chi_squared = chi_squared / degrees_of_freedom if degrees_of_freedom > 0 else 0.0

    logger.info(f"LSQ Final Results:")
    logger.info(f"  Data points: {n_data_points:,}")
    logger.info(f"  Parameters fitted: {n_parameters} (physics: {len(physics_params)}, scaling: 2)")
    logger.info(f"  Degrees of freedom: {degrees_of_freedom:,}")
    logger.info(f"  Chi-squared: {chi_squared:.4f}")
    logger.info(f"  Reduced chi-squared: {reduced_chi_squared:.6f}")
    logger.info(f"  Converged: {converged}")

    computation_time = time.time() - start_time

    # Create LSQResult wrapper
    return LSQResult(
        mean_params=physics_params,
        mean_contrast=contrast,
        mean_offset=offset,
        chi_squared=chi_squared,
        reduced_chi_squared=reduced_chi_squared,
        residual_std=float(np.std(residuals_final)),
        max_residual=float(np.max(np.abs(residuals_final))),
        degrees_of_freedom=degrees_of_freedom,
        converged=converged,
        n_iterations=n_iterations,
        computation_time=computation_time,
        backend="SciPy" if SCIPY_AVAILABLE else "NumPy",
        dataset_size="LARGE",  # Will be updated by caller if needed
        analysis_mode=analysis_mode,
        noise_estimated=estimate_noise,
        noise_model=noise_model if estimate_noise else None,
        estimated_sigma=estimated_sigma,
        noise_params=noise_params
    )