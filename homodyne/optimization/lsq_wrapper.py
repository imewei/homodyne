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
from homodyne.utils.logging import get_logger
from homodyne.core.jax_backend import safe_len
from homodyne.optimization.theory_engine import UnifiedTheoryEngine, create_theory_engine, TheoryComputationConfig
from homodyne.optimization.base_result import LSQResult
from homodyne.utils.progress import OptimizationProgress, track_optimization

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


def create_correlation_samples(t1, t2, data, sigma=None, sampling_config=None):
    """Create intelligent sampling points for correlation matrix fitting.

    Returns:
        sample_indices: List of (i, j) indices for sampling
        sample_weights: Weights for each sample based on error estimates
        sample_data: Flattened array of sampled data values
        sample_sigma: Flattened array of uncertainties (if available)
    """
    n_points = t1.shape[0] if t1.ndim == 1 else t1.shape[0]  # Handle both 1D and 2D time arrays

    # Default sampling configuration
    config = sampling_config or {
        'diagonal_samples': 200,      # Points along diagonal
        'near_diagonal_width': 10,    # Width of near-diagonal band
        'cross_sections': 5,           # Number of fixed-τ cross-sections
        'tau_max_ratio': 0.3,          # Maximum τ as fraction of total time
        'adaptive_density': True,      # Use error-weighted sampling
        'min_samples': 2000,
        'max_samples': 5000
    }

    samples = []

    # 1. Diagonal sampling (τ = 0, most signal)
    diagonal_idx = np.linspace(0, n_points-1, config['diagonal_samples'], dtype=int)
    for idx in diagonal_idx:
        samples.append((idx, idx))

    # 2. Near-diagonal band (small τ, high SNR)
    width = config['near_diagonal_width']
    for offset in range(1, width+1):
        step = max(1, offset * 2)  # Sparser sampling for larger offsets
        for i in range(0, n_points-offset, step):
            samples.append((i, i+offset))
            if i != i+offset:  # Exploit symmetry
                samples.append((i+offset, i))

    # 3. Fixed-τ cross-sections (logarithmically spaced)
    tau_max = int(n_points * config['tau_max_ratio'])
    if tau_max > 1:
        tau_values = np.logspace(0, np.log10(tau_max), config['cross_sections'], dtype=int)
        for tau in tau_values[1:]:  # Skip τ=0 (already in diagonal)
            step = max(1, tau // 20)  # Adaptive step size
            for i in range(0, n_points-tau, step):
                samples.append((i, i+tau))

    # 4. Error-weighted adaptive sampling (if sigma available)
    if sigma is not None and config['adaptive_density']:
        # Add more samples in low-noise regions
        sigma_flat = sigma.flatten()
        low_noise_threshold = np.percentile(sigma_flat, 25)
        low_noise_mask = sigma.reshape(n_points, n_points) < low_noise_threshold

        # Random sampling in low-noise regions
        low_noise_indices = np.argwhere(low_noise_mask)
        n_adaptive = min(500, len(low_noise_indices))
        if n_adaptive > 0:
            np.random.seed(42)  # For reproducibility
            adaptive_idx = np.random.choice(len(low_noise_indices), n_adaptive, replace=False)
            for idx in adaptive_idx:
                samples.append(tuple(low_noise_indices[idx]))

    # Remove duplicates and limit total samples
    samples = list(set(samples))
    if len(samples) > config['max_samples']:
        samples = samples[:config['max_samples']]

    # Extract data and weights
    sample_indices = np.array(samples)
    sample_data = np.array([data[i, j] for i, j in samples])

    # Compute weights based on:
    # 1. Sampling density (to avoid overweighting dense regions)
    # 2. Measurement uncertainty (if available)
    # 3. Physical importance (diagonal > off-diagonal)

    weights = np.ones(len(samples))

    # Density-based weights
    for idx, (i, j) in enumerate(samples):
        tau = abs(i - j)
        if tau == 0:
            weights[idx] *= 2.0  # Diagonal importance
        elif tau < width:
            weights[idx] *= 1.5  # Near-diagonal importance
        else:
            # Reduce weight for densely sampled regions
            density_factor = 1.0 / np.sqrt(1 + tau/10)
            weights[idx] *= density_factor

    # Error-based weights (if available)
    if sigma is not None:
        sample_sigma = np.array([sigma[i, j] for i, j in samples])
        error_weights = 1.0 / (sample_sigma + 1e-10)
        error_weights /= np.mean(error_weights)  # Normalize
        weights *= error_weights
    else:
        sample_sigma = None

    # Normalize weights
    weights /= (np.sum(weights) / len(samples))

    logger.info(f"Created {len(samples)} sampling points from {n_points}x{n_points} matrix")
    diagonal_count = np.sum([s[0]==s[1] for s in samples])
    near_diag_count = np.sum([0<abs(s[0]-s[1])<width for s in samples])
    far_field_count = np.sum([abs(s[0]-s[1])>=width for s in samples])
    logger.debug(f"Sampling distribution: diagonal={diagonal_count}, "
                f"near-diagonal={near_diag_count}, "
                f"far-field={far_field_count}")

    return sample_indices, weights, sample_data, sample_sigma


def compute_theory_sampled(physics_params, sample_indices, t1, t2, phi, q, L,
                          contrast, offset, engine, dt=0.1):
    """Compute theory ONLY at sampled points - no full matrix computation."""

    physics_params = np.atleast_1d(np.asarray(physics_params))
    phi_array = np.atleast_1d(np.asarray(phi))

    # Handle both 1D time arrays and 2D grids
    if t1.ndim == 1 and t2.ndim == 1:
        # Create time arrays for sampled points only
        t1_samples = np.array([t1[i] for i, _ in sample_indices])
        t2_samples = np.array([t2[j] for _, j in sample_indices])
    else:
        # Already 2D grids
        t1_samples = np.array([t1.flat[i * t1.shape[1] + j] for i, j in sample_indices])
        t2_samples = np.array([t2.flat[i * t2.shape[1] + j] for i, j in sample_indices])

    theory_samples = []

    # Check if engine has vectorized method
    if hasattr(engine.theory_engine, 'compute_g1_vectorized'):
        for phi_val in phi_array:
            # Compute g1 ONLY at sample points using vectorized engine call
            g1_samples = engine.theory_engine.compute_g1_vectorized(
                params=physics_params,
                t1=t1_samples,
                t2=t2_samples,
                phi=np.array([phi_val]),
                q=q, L=L, dt=dt
            )
            # Apply correlation function transformation
            g2_samples = offset + contrast * g1_samples**2
            theory_samples.extend(g2_samples)
    else:
        # Fall back to regular compute_g1 but with sampled meshgrids
        t2_grid, t1_grid = np.meshgrid(t2_samples, t1_samples, indexing='ij')

        for phi_val in phi_array:
            g1_samples = engine.theory_engine.compute_g1(
                params=physics_params, t1=t1_grid, t2=t2_grid,
                phi=np.array([phi_val]), q=q, L=L, dt=dt
            )
            # Squeeze and flatten
            if g1_samples.shape[0] == 1:
                g1_samples = np.squeeze(g1_samples, axis=0)
            g1_flat = g1_samples.flatten()

            # Apply correlation function transformation
            g2_samples = offset + contrast * g1_flat**2
            theory_samples.extend(g2_samples)

    return np.array(theory_samples)


# Using LSQResult from base_result.py - duplicate class removed


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
    config: Optional[Dict[str, Any]] = None,
    max_iterations: Optional[int] = None,
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

    # Set default max_iterations for LSQ method if not provided
    if max_iterations is None:
        max_iterations = kwargs.get('max_iterations', 10000)  # CLI default from args_parser.py

    logger.info(f"LSQ optimization will use max_iterations: {max_iterations}")

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
        # Calculate the estimated sigma value
        estimated_sigma_value = float(np.std(data) * 0.1)

        if noise_model == "hierarchical":
            # Store only the scalar value instead of massive array
            estimated_sigma = np.array([estimated_sigma_value])  # Single element array
            noise_params = {"global_sigma": estimated_sigma_value}
        else:
            # Store only the scalar value instead of massive array
            estimated_sigma = np.array([estimated_sigma_value])  # Single element array
            noise_params = {"global_sigma": estimated_sigma_value}

        # Create the sigma array for optimization (same size as data)
        sigma_used = np.full_like(data, estimated_sigma_value)

        logger.info(f"✓ Estimated noise σ: {estimated_sigma_value:.6f} (stored as single value)")
    else:
        estimated_sigma = None
        noise_params = None
        sigma_used = sigma

    # Use intelligent sampling strategy instead of full matrix flattening
    # Get sampling configuration from kwargs if provided
    sampling_config = kwargs.get('sampling_config', None)

    # Create samples at start of optimization
    sample_indices, sample_weights, data_samples, sigma_samples = create_correlation_samples(
        t1, t2, data, sigma_used, sampling_config
    )

    # Store sampling info for later use
    sampling_info = {
        'indices': sample_indices,
        'weights': sample_weights,
        'n_samples': len(sample_indices),
        'full_shape': data.shape
    }

    # CRITICAL FIX: Use configuration dt value directly
    # During LSQ optimization, t1/t2 are correlation matrices, not time arrays.
    # Computing dt from correlation matrix differences produces dt=0.0, causing
    # sinc_prefactor=0 in JAX backend, leading to constant g1=1.0 values.
    dt = kwargs.get('dt', 0.1)  # Use dt from configuration, fallback to 0.1

    logger.debug(f"Using dt parameter from configuration: {dt} (LSQ optimization uses correlation matrices, not time arrays)")

    # Initialize progress tracker if enabled
    show_progress = kwargs.get('show_progress', True)
    if show_progress:
        progress_tracker = track_optimization(
            method="LSQ",
            total_iterations=max_iterations,
            dataset_size=len(data_samples)  # Use sampled size for efficiency
        )
    else:
        progress_tracker = None

    # Define the chi-square objective function with proper NumPy/JAX boundary
    # Add evaluation counter
    eval_counter = [0]

    def chi_square_objective_numpy(params):
        """NumPy interface for scipy.optimize - handles all conversions.

        This function is called by scipy.optimize with NumPy arrays.
        It ensures proper type conversion and handles all boundary logic.
        Uses intelligent sampling for efficient optimization.
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
        if contrast < 1e-10 or contrast > 1.0:
            return 1e10
        if offset < 1e-10 or offset > 2.0:
            return 1e10

        try:
            # Call the actual computation using sampled points (JAX or NumPy)
            if JAX_AVAILABLE:
                chi2 = compute_chi2_sampled_jax(
                    physics_params, data_samples, sample_weights, sample_indices,
                    t1, t2, phi, q, L, contrast, offset, engine, dt
                )
                # Update progress tracker
                if progress_tracker is not None:
                    progress_tracker.update(loss=chi2)
                # Log chi2 value for debugging
                if eval_counter[0] % 10 == 1:
                    logger.debug(f"  -> chi2={chi2:.6f}")
                return float(chi2)  # Ensure Python float for scipy
            else:
                chi2 = compute_chi2_sampled_numpy(
                    physics_params, data_samples, sample_weights, sample_indices,
                    t1, t2, phi, q, L, contrast, offset, engine, dt
                )
                # Update progress tracker
                if progress_tracker is not None:
                    progress_tracker.update(loss=chi2)
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
        def compute_chi2_jax_inner(theory_samples, data_samples, sample_weights):
            """JAX JIT-compiled chi-squared calculation for sampled points.

            This function only performs the chi-squared calculation.
            Theory computation happens outside the JIT boundary.
            """
            residuals = (theory_samples - data_samples) * sample_weights
            return jnp.sum(residuals ** 2)

        def compute_chi2_sampled_jax(physics_params, data_samples, sample_weights, sample_indices,
                                     t1, t2, phi, q, L, contrast, offset, engine, dt):
            """JAX-optimized chi-squared computation using sampled points.

            Computes theory only at sampled points for efficiency.
            """
            # Compute theory at sampled points only
            theory_samples = compute_theory_sampled(
                physics_params, sample_indices, t1, t2, phi, q, L,
                contrast, offset, engine, dt
            )
            # Use JIT-compiled function for chi-squared calculation
            return compute_chi2_jax_inner(theory_samples, data_samples, sample_weights)

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

    def compute_chi2_sampled_numpy(physics_params, data_samples, sample_weights, sample_indices,
                                   t1, t2, phi, q, L, contrast, offset, engine, dt):
        """NumPy chi-squared computation using sampled points.

        Computes theory only at sampled points for efficiency.
        """
        # Compute theory at sampled points only
        theory_samples = compute_theory_sampled(
            physics_params, sample_indices, t1, t2, phi, q, L,
            contrast, offset, engine, dt
        )
        # Compute chi-squared
        residuals = (theory_samples - data_samples) * sample_weights
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
                'maxiter': max_iterations,
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
            bounds.append((1e-10, 1.0))  # contrast bounds
            bounds.append((1e-10, 2.0))  # offset bounds

            result = minimize(
                chi_square_objective_numpy,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': max_iterations,
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
                    options={'maxiter': max_iterations, 'disp': True}
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

    # Close progress tracker if it was used
    if progress_tracker is not None:
        progress_tracker.close()

    # Extract optimized parameters
    physics_params = optimized_params[:-2]
    contrast = optimized_params[-2]
    offset = optimized_params[-1]

    # Compute final statistics using sampled points for efficiency
    logger.info("Computing final statistics from sampled optimization...")

    # Compute final chi-squared on sampled points
    final_chi_sq_sampled = chi_square_objective_numpy(optimized_params)

    # Now compute full matrices ONLY for visualization purposes
    logger.info("Computing full correlation matrices for visualization (post-optimization only)...")

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

    # Compute full fitted g2 matrix for visualization
    theory_final = g1_final**2
    fitted_g2_full = theory_final * contrast + offset

    # Store the full fitted data in original matrix shape for visualization
    if fitted_g2_full.ndim > 2:
        fitted_g2_matrix = fitted_g2_full  # Keep multi-phi shape
    else:
        fitted_g2_matrix = fitted_g2_full.reshape(data.shape)  # Restore original shape

    # Flatten for residual calculation
    if theory_final.ndim > 2:
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

    # Create sampling info dictionary
    sampling_info_dict = {
        'sample_indices': sample_indices,
        'sample_weights': sample_weights,
        'n_samples': len(sample_indices),
        'total_points': len(data_flat),
        'sampling_fraction': len(sample_indices) / len(data_flat)
    }

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
        noise_params=noise_params,
        fitted_g2_matrix=fitted_g2_matrix,
        sampling_info=sampling_info_dict
    )