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

    # Define the chi-square objective function
    def chi_square_objective(params):
        """Compute chi-squared for given parameters."""
        try:
            # Split parameters into physics and scaling
            physics_params = params[:-2]
            contrast = params[-2]
            offset = params[-1]

            # Compute theoretical correlation function (g1^2)
            g1_theory = engine.theory_engine.compute_g1(
                physics_params, t1, t2, phi, q, L
            )
            theory = g1_theory**2

            # Apply scaling: data = theory * contrast + offset
            theory_scaled = theory.flatten() * contrast + offset

            # Compute weighted chi-squared
            residuals = data_flat - theory_scaled
            chi_sq = np.sum(weights * residuals**2)

            return chi_sq

        except Exception as e:
            logger.debug(f"Objective function evaluation failed: {e}")
            return 1e10  # Return large value for failed evaluations

    # Set up initial parameters: physics + contrast + offset
    initial_contrast = kwargs.get("initial_contrast", 0.3)
    initial_offset = kwargs.get("initial_offset", 1.0)
    initial_params = np.concatenate([initial_physics_params, [initial_contrast, initial_offset]])

    # Evaluate initial objective value for debugging
    initial_chi_sq = chi_square_objective(initial_params)
    logger.info(f"Initial chi-squared: {initial_chi_sq:.6f}")
    logger.info(f"Initial parameters: physics={initial_physics_params}, contrast={initial_contrast}, offset={initial_offset}")

    # Perform optimization
    if SCIPY_AVAILABLE:
        logger.info("Using SciPy L-BFGS-B optimization")
        # Set parameter bounds
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

        # Use more reasonable tolerances
        result = minimize(
            chi_square_objective,
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

        # Check if optimization actually ran
        if hasattr(result, 'nit') and result.nit == 0:
            logger.warning("L-BFGS-B converged immediately. Trying Nelder-Mead as fallback...")
            # Try Nelder-Mead which doesn't use gradients
            result = minimize(
                chi_square_objective,
                initial_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-4,
                    'fatol': 1e-4,
                    'disp': True
                }
            )
            logger.info(f"Nelder-Mead completed with {result.nit} iterations")

        if result.success or (hasattr(result, 'nit') and result.nit > 0):
            logger.info(f"LSQ optimization completed in {result.nit} iterations")
            optimized_params = result.x
            final_chi_sq = result.fun
            converged = result.success
            n_iterations = result.nit
        else:
            logger.warning(f"LSQ optimization failed: {result.message if hasattr(result, 'message') else 'Unknown error'}")
            # Try one more time with Powell method
            logger.info("Trying Powell method as final fallback...")
            result = minimize(
                chi_square_objective,
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
                final_chi_sq = chi_square_objective(initial_params)
                converged = False
                n_iterations = 0
    else:
        # Fallback: just evaluate at initial parameters
        logger.info("Using initial parameters (no optimization)")
        optimized_params = initial_params
        final_chi_sq = chi_square_objective(initial_params)
        converged = True
        n_iterations = 1

    # Extract optimized parameters
    physics_params = optimized_params[:-2]
    contrast = optimized_params[-2]
    offset = optimized_params[-1]

    # Compute final statistics
    g1_final = engine.theory_engine.compute_g1(physics_params, t1, t2, phi, q, L)
    theory_final = g1_final**2
    theory_scaled_final = theory_final.flatten() * contrast + offset
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