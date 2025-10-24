"""SVI Initialization Module for Consensus Monte Carlo

This module provides Stochastic Variational Inference (SVI) initialization for MCMC
warmup in the Consensus Monte Carlo pipeline. SVI is MANDATORY because NLSQ does NOT
provide Hessian or covariance matrices.

Key Functions:
    run_svi_initialization: Main SVI training loop to estimate mass matrix
    pool_samples_from_shards: Create balanced pooled dataset from shards
    _get_param_names: Map parameter names between config and code conventions
    _init_loc_fn: Initialize SVI guide with NLSQ parameters

Algorithm:
    1. Pool random samples from all shards (200 points/shard default)
    2. Run SVI with AutoLowRankMultivariateNormal guide (5000 steps)
    3. Extract mean parameters and covariance matrix from trained guide
    4. Invert covariance to get inverse mass matrix (precision matrix)
    5. Return init_params and inv_mass_matrix for NUTS warmup

SVI Rationale:
    - NLSQ provides point estimates only (no Hessian/covariance)
    - SVI provides variational approximation to posterior
    - Covariance from SVI → inverse mass matrix for MCMC
    - Alternative: Identity mass matrix (much slower MCMC convergence, 2-5x longer)

Performance:
    - Expected runtime: 5-10 minutes on typical XPCS datasets
    - Fallback to identity matrix if SVI fails or exceeds timeout (15 min)
    - User can disable SVI via config: cmc.initialization.use_svi: false

References:
    NumPyro SVI Guide: https://num.pyro.ai/en/stable/svi.html
    Hoffman & Gelman (2014): "The No-U-Turn Sampler"
"""

from typing import Dict, Optional, Tuple, Any, Callable
import time

import jax
import jax.numpy as jnp
import numpy as np

try:
    import numpyro
    from numpyro.infer import SVI, Trace_ELBO, init_to_value
    from numpyro.infer.autoguide import AutoLowRankMultivariateNormal
    from numpyro.optim import Adam

    NUMPYRO_AVAILABLE = True
except ImportError:
    NUMPYRO_AVAILABLE = False
    numpyro = None

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


def pool_samples_from_shards(
    shards: list,
    samples_per_shard: int = 200,
    random_seed: int = 42,
) -> Dict[str, jnp.ndarray]:
    """Pool random samples from all shards for SVI initialization.

    Selects random subset of datapoints from each shard to create a balanced
    pooled dataset for SVI training. This ensures efficient initialization
    without requiring the full dataset.

    Parameters
    ----------
    shards : list of dict
        List of shard data dictionaries, each containing:
        - 'data': Experimental c2 values (array)
        - 'sigma': Noise estimates (array)
        - 't1', 't2': Time arrays
        - 'phi': Angle array
        - 'q': Wavevector (scalar)
        - 'L': Sample-detector distance (scalar)
    samples_per_shard : int, default 200
        Number of samples to draw from each shard
    random_seed : int, default 42
        Random seed for reproducibility

    Returns
    -------
    pooled_data : dict
        Pooled dataset dictionary with concatenated samples:
        - 'data': Concatenated experimental values
        - 'sigma': Concatenated noise estimates
        - 't1', 't2': Concatenated time arrays
        - 'phi': Concatenated angles
        - 'q': Wavevector (from first shard)
        - 'L': Sample-detector distance (from first shard)

    Examples
    --------
    >>> shards = [{'data': arr1, ...}, {'data': arr2, ...}]
    >>> pooled = pool_samples_from_shards(shards, samples_per_shard=200)
    >>> print(f"Pooled {len(pooled['data'])} samples from {len(shards)} shards")
    """
    if not shards:
        raise ValueError("Cannot pool samples from empty shards list")

    rng = np.random.RandomState(random_seed)
    pooled_arrays = {}

    total_sampled = 0
    for i, shard in enumerate(shards):
        # Randomly sample indices from this shard
        n_points = len(shard["data"])
        sample_size = min(samples_per_shard, n_points)
        indices = rng.choice(n_points, size=sample_size, replace=False)

        # Pool samples for each array field
        for key in ["data", "sigma", "t1", "t2", "phi"]:
            if key not in shard:
                raise KeyError(f"Shard {i} missing required key '{key}'")

            if key not in pooled_arrays:
                pooled_arrays[key] = []
            pooled_arrays[key].append(shard[key][indices])

        total_sampled += sample_size

    # Concatenate all samples
    pooled_data = {
        key: jnp.concatenate(arrays, axis=0) for key, arrays in pooled_arrays.items()
    }

    # Copy scalar parameters (q, L) from first shard
    # These should be identical across all shards
    pooled_data["q"] = shards[0]["q"]
    pooled_data["L"] = shards[0]["L"]

    logger.info(
        f"Pooled {len(pooled_data['data'])} samples from {len(shards)} shards "
        f"({samples_per_shard} samples/shard target)"
    )
    logger.debug(
        f"Pooled data statistics: "
        f"data range=[{jnp.min(pooled_data['data']):.4f}, {jnp.max(pooled_data['data']):.4f}], "
        f"phi range=[{jnp.min(pooled_data['phi']):.4f}, {jnp.max(pooled_data['phi']):.4f}]"
    )

    return pooled_data


def run_svi_initialization(
    model_fn: Callable,
    pooled_data: Dict[str, jnp.ndarray],
    num_steps: int = 5000,
    learning_rate: float = 0.001,
    rank: int = 5,
    init_params: Optional[Dict[str, float]] = None,
    timeout_minutes: float = 15.0,
    enable_progress_bar: bool = True,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Run SVI to estimate mass matrix for MCMC initialization.

    Uses AutoLowRankMultivariateNormal guide to approximate the posterior
    distribution from pooled samples across all shards. Extracts mean and
    covariance for MCMC initialization.

    Algorithm:
        1. Pool random samples from all shards (e.g., 10k points total)
        2. Run SVI with AutoLowRankMultivariateNormal guide
        3. Extract mean parameters and covariance matrix
        4. Invert covariance to get mass matrix (precision matrix)
        5. Return init_params and inv_mass_matrix for NUTS

    Parameters
    ----------
    model_fn : callable
        NumPyro model function (same as used for MCMC)
        Must accept data dict and return log probability
    pooled_data : dict
        Pooled data dictionary with keys:
        - 'data': Experimental c2 values
        - 'sigma': Noise estimates
        - 't1', 't2': Time arrays
        - 'phi': Angle array
        - 'q': Wavevector
        - 'L': Sample-detector distance
    num_steps : int, default 5000
        Number of SVI optimization steps
    learning_rate : float, default 0.001
        Adam optimizer learning rate
    rank : int, default 5
        Rank for low-rank approximation (5-10 typical)
    init_params : dict, optional
        Optional initial parameters from NLSQ to improve convergence
    timeout_minutes : float, default 15.0
        Maximum runtime in minutes before fallback to identity matrix
    enable_progress_bar : bool, default True
        Show progress bar during SVI optimization

    Returns
    -------
    init_params : dict
        Mean parameter values for MCMC initialization
        Keys are parameter names (e.g., 'D0', 'alpha', 'D_offset', ...)
    inv_mass_matrix : np.ndarray
        Inverse mass matrix (precision matrix) for NUTS
        Shape: (num_params, num_params)

    Raises
    ------
    ImportError
        If NumPyro is not available
    TimeoutError
        If SVI exceeds timeout (fallback to identity matrix)
    ValueError
        If SVI fails to converge or produces invalid mass matrix

    Notes
    -----
    - Expected runtime: 5-10 minutes for typical XPCS datasets
    - Falls back to identity mass matrix if SVI fails
    - Uses only pooled subset of data (not full dataset)
    - ELBO loss should decrease monotonically for proper convergence

    Examples
    --------
    >>> pooled = pool_samples_from_shards(shards)
    >>> init_params, mass_matrix = run_svi_initialization(
    ...     model_fn=numpyro_model,
    ...     pooled_data=pooled,
    ...     init_params=nlsq_params
    ... )
    >>> print(f"Init params: {init_params}")
    >>> print(f"Mass matrix condition number: {np.linalg.cond(mass_matrix):.2e}")
    """
    if not NUMPYRO_AVAILABLE:
        raise ImportError(
            "NumPyro is required for SVI initialization. "
            "Install with: pip install numpyro>=0.18.0"
        )

    logger.info("Starting SVI initialization for CMC...")
    logger.info(
        f"SVI config: num_steps={num_steps}, learning_rate={learning_rate}, "
        f"rank={rank}, timeout={timeout_minutes} min"
    )

    start_time = time.time()

    try:
        # Create variational guide with low-rank structure
        # Low-rank approximation reduces computational cost while maintaining accuracy
        try:
            logger.debug(f"Creating AutoLowRankMultivariateNormal guide with rank={rank}, init_params={'provided' if init_params else 'None'}")
            if init_params:
                init_loc_fn = _init_loc_fn(init_params)
                guide = AutoLowRankMultivariateNormal(
                    model_fn,
                    rank=rank,
                    init_loc_fn=init_loc_fn,
                )
            else:
                # Don't pass init_loc_fn at all if no init_params
                guide = AutoLowRankMultivariateNormal(
                    model_fn,
                    rank=rank,
                )
            logger.debug("Guide created successfully")
        except Exception as e:
            logger.error(f"Failed to create guide: {e}")
            raise

        # Configure Adam optimizer
        optimizer = Adam(learning_rate)

        # Create SVI object with ELBO loss
        svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())
        logger.debug("SVI object created successfully")

        # Initialize SVI state
        # NumPyro SVI requires passing model arguments to init/update
        rng_key = jax.random.PRNGKey(42)
        # Unpack pooled_data to pass as model arguments
        model_args = _extract_model_args(pooled_data)
        logger.debug(f"Model args: {len(model_args)} arguments, types: {[type(x) for x in model_args]}")
        logger.debug(f"Calling svi.init(rng_key, *model_args)...")
        svi_state = svi.init(rng_key, *model_args)
        logger.debug("SVI initialized successfully")

        # Run SVI optimization with progress tracking
        losses = []
        for step in range(num_steps):
            # Check timeout
            elapsed_minutes = (time.time() - start_time) / 60.0
            if elapsed_minutes > timeout_minutes:
                raise TimeoutError(
                    f"SVI exceeded timeout of {timeout_minutes} minutes "
                    f"(current: {elapsed_minutes:.1f} min)"
                )

            # Update SVI state
            svi_state, loss = svi.update(svi_state, *model_args)
            losses.append(loss)

            # Log progress every 1000 steps
            if (step + 1) % 1000 == 0:
                logger.info(
                    f"SVI step {step + 1}/{num_steps}, "
                    f"ELBO loss: {loss:.4f}, "
                    f"time: {elapsed_minutes:.1f} min"
                )

        # Extract guide parameters
        params = svi.get_params(svi_state)

        # Get variational distribution from guide
        base_dist = guide.get_base_dist(params)

        # Extract mean parameters (init_params for MCMC)
        mean_params = base_dist.mean

        # Extract covariance matrix
        cov_matrix = base_dist.covariance_matrix

        # Validate covariance matrix is positive definite
        eigenvalues = jnp.linalg.eigvalsh(cov_matrix)
        min_eigenvalue = jnp.min(eigenvalues)
        if min_eigenvalue <= 0:
            logger.warning(
                f"Covariance matrix has negative eigenvalue: {min_eigenvalue:.2e}. "
                f"Adding regularization."
            )

        # Compute inverse mass matrix (precision matrix)
        # Add regularization for numerical stability
        regularization = 1e-6 * jnp.eye(cov_matrix.shape[0])
        inv_mass_matrix = jnp.linalg.inv(cov_matrix + regularization)

        # Validate positive definiteness of inverse mass matrix
        inv_eigenvalues = jnp.linalg.eigvalsh(inv_mass_matrix)
        min_inv_eigenvalue = jnp.min(inv_eigenvalues)
        if min_inv_eigenvalue <= 0:
            raise ValueError(
                f"Inverse mass matrix has negative eigenvalue: {min_inv_eigenvalue:.2e}"
            )

        # Compute condition number (ratio of largest to smallest eigenvalue)
        condition_number = jnp.linalg.cond(inv_mass_matrix)

        # Log final statistics
        elapsed_time = time.time() - start_time
        logger.info(f"SVI initialization complete in {elapsed_time:.1f} seconds")
        logger.info(f"Final ELBO loss: {losses[-1]:.4f}")
        logger.info(
            f"Mean parameter range: [{jnp.min(mean_params):.4e}, {jnp.max(mean_params):.4e}]"
        )
        logger.info(f"Mass matrix condition number: {condition_number:.2e}")

        # Validate convergence: ELBO should decrease
        if len(losses) >= 100:
            early_loss = np.mean(losses[:100])
            late_loss = np.mean(losses[-100:])
            improvement = (early_loss - late_loss) / early_loss
            logger.info(
                f"ELBO improvement: {improvement:.1%} "
                f"(early={early_loss:.4f}, late={late_loss:.4f})"
            )

            if improvement < 0.01:
                logger.warning(
                    "SVI may not have converged properly (< 1% ELBO improvement). "
                    "Consider increasing num_steps or learning_rate."
                )

        # Convert mean_params array to dict format with parameter names
        num_params = len(mean_params)
        param_names = _get_param_names(num_params)
        init_params_dict = {
            name: float(val) for name, val in zip(param_names, mean_params)
        }

        logger.debug(f"Extracted init_params: {init_params_dict}")

        # Convert to numpy for broader compatibility
        return init_params_dict, np.array(inv_mass_matrix)

    except TimeoutError as e:
        logger.error(f"SVI timeout: {e}")
        logger.warning("Falling back to identity mass matrix (slower MCMC convergence)")
        return _fallback_to_identity_matrix(init_params, pooled_data)

    except Exception as e:
        logger.error(f"SVI initialization failed: {e}")
        logger.warning("Falling back to identity mass matrix (slower MCMC convergence)")
        return _fallback_to_identity_matrix(init_params, pooled_data)


def _init_loc_fn(nlsq_params: Dict[str, float]) -> Callable:
    """Create initialization function for SVI guide using NLSQ parameters.

    This function creates a NumPyro init_loc_fn that initializes the variational
    guide's mean parameters using point estimates from NLSQ optimization.
    Using NLSQ parameters as starting point significantly improves SVI convergence.

    Parameters
    ----------
    nlsq_params : dict
        Parameter dictionary from NLSQ optimization
        Keys: parameter names (e.g., 'D0', 'alpha', 'gamma_dot_t0')
        Values: parameter estimates

    Returns
    -------
    init_fn : callable
        Initialization function for NumPyro guide (init_to_value strategy)

    Examples
    --------
    >>> nlsq_params = {'D0': 1000.0, 'alpha': 0.5, 'D_offset': 10.0}
    >>> init_fn = _init_loc_fn(nlsq_params)
    >>> guide = AutoLowRankMultivariateNormal(model, init_loc_fn=init_fn)
    """
    # Use NumPyro's init_to_value initialization strategy
    # This returns a partial function that initializes specified sites to given values
    logger.debug(f"Creating init_to_value with NLSQ parameters: {list(nlsq_params.keys())}")
    return init_to_value(values=nlsq_params)


def _get_param_names(num_params: int) -> list:
    """Get parameter names based on analysis mode.

    Maps number of parameters to corresponding parameter names following
    Homodyne's parameter naming conventions. Handles config name to code name
    mapping (e.g., gamma_dot_0 → gamma_dot_t0).

    Parameters
    ----------
    num_params : int
        Total number of parameters
        - 5: static_isotropic mode
        - 9: laminar_flow mode
        - Other: generic naming

    Returns
    -------
    param_names : list of str
        Parameter names in canonical order

    Notes
    -----
    Parameter order matches Homodyne optimization conventions:
        Static isotropic (5 params):
            [contrast, offset, D0, alpha, D_offset]

        Laminar flow (9 params):
            [contrast, offset, D0, alpha, D_offset,
             gamma_dot_t0, beta, gamma_dot_t_offset, phi0]

    Examples
    --------
    >>> _get_param_names(5)
    ['contrast', 'offset', 'D0', 'alpha', 'D_offset']

    >>> _get_param_names(9)
    ['contrast', 'offset', 'D0', 'alpha', 'D_offset',
     'gamma_dot_t0', 'beta', 'gamma_dot_t_offset', 'phi0']
    """
    if num_params == 5:
        # Static isotropic: scaling + diffusion parameters
        return ["contrast", "offset", "D0", "alpha", "D_offset"]
    elif num_params == 9:
        # Laminar flow: scaling + diffusion + shear parameters
        # Note: gamma_dot_0 → gamma_dot_t0 (config to code name mapping)
        return [
            "contrast",
            "offset",
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ]
    else:
        # Generic fallback for other parameter counts
        logger.warning(
            f"Unexpected parameter count: {num_params}. "
            f"Using generic parameter names."
        )
        return [f"param_{i}" for i in range(num_params)]


def _extract_model_args(pooled_data: Dict[str, jnp.ndarray]) -> tuple:
    """Extract model arguments from pooled_data dictionary.

    Converts pooled_data dict to positional arguments for NumPyro model.
    The model signature should match the XPCS homodyne model:
        model(data, sigma, t1, t2, phi, q, L)

    Parameters
    ----------
    pooled_data : dict
        Pooled data dictionary

    Returns
    -------
    model_args : tuple
        Positional arguments for model function

    Notes
    -----
    If pooled_data is missing expected keys, this function attempts to
    provide reasonable defaults or extract what's available.
    """
    # Standard XPCS model signature
    try:
        return (
            pooled_data["data"],
            pooled_data["sigma"],
            pooled_data["t1"],
            pooled_data["t2"],
            pooled_data["phi"],
            pooled_data["q"],
            pooled_data["L"],
        )
    except KeyError as e:
        # Fallback: Try minimal signature (data, sigma)
        logger.warning(f"Missing key in pooled_data: {e}. Using minimal model args.")
        return (pooled_data.get("data"), pooled_data.get("sigma"))


def _fallback_to_identity_matrix(
    init_params: Optional[Dict[str, float]],
    pooled_data: Dict[str, jnp.ndarray],
) -> Tuple[Dict[str, float], np.ndarray]:
    """Create fallback initialization with identity mass matrix.

    Used when SVI fails or times out. Returns identity mass matrix which
    provides no information about parameter correlations but allows MCMC
    to proceed (with slower warmup).

    Parameters
    ----------
    init_params : dict or None
        Optional initial parameters from NLSQ
    pooled_data : dict
        Pooled data to infer number of parameters

    Returns
    -------
    init_params : dict
        Initial parameter values (from NLSQ if available, else zeros)
    inv_mass_matrix : np.ndarray
        Identity matrix of appropriate size

    Notes
    -----
    Using identity mass matrix increases MCMC warmup time by 2-5x compared
    to using SVI-derived mass matrix. However, it's a reliable fallback.
    """
    # Infer number of parameters from pooled data
    # This is a rough estimate - actual parameter count determined by model
    # Assume standard parameter counts: 5 or 9
    if init_params:
        num_params = len(init_params)
        fallback_params = init_params.copy()
    else:
        # Default to static_isotropic (5 params)
        num_params = 5
        param_names = _get_param_names(num_params)
        fallback_params = {name: 0.0 for name in param_names}

    # Create identity mass matrix
    identity_matrix = np.eye(num_params)

    logger.warning(
        f"Using identity mass matrix ({num_params}x{num_params}). "
        f"This will increase MCMC warmup time by 2-5x."
    )
    logger.info(f"Fallback init_params: {fallback_params}")

    return fallback_params, identity_matrix
