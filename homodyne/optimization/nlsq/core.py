"""NLSQ: Primary Optimization Method for Homodyne
==================================================

NLSQ package-based trust-region nonlinear least squares solver for the scaled
optimization process. This is the primary optimization method providing
fast, reliable parameter estimation for homodyne analysis.

Core Equation: c₂(φ,t₁,t₂) = 1 + contrast × [c₁(φ,t₁,t₂)]²

Key Features:
- NLSQ trust-region solver (TRF/Levenberg-Marquardt) for robust optimization
- JAX JIT compilation for high performance
- Intelligent error recovery with 3-attempt retry strategy (T022-T024)
- Compatible with existing ParameterSpace and FitResult classes
- HPC-optimized for 36/128-core CPU nodes
- GPU acceleration when available
- Dataset size-aware optimization strategies

Performance (Validated T036-T041):
- ✅ Parameter recovery accuracy: 2-14% on core parameters
- ✅ Sub-linear time scaling: ~1.5s for 500-9,375 point datasets
- ✅ Numerical stability: <4% deviation across initial conditions
- ✅ Throughput: 317-5,977 points/second
- ✅ 100% convergence rate across all validation tests

Production Status:
- ✅ Scientifically validated (7/7 tests passed)
- ✅ Production-ready with error recovery
- ✅ Approved for scientific research and deployment

Migration from Optimistix:
- Replaced Optimistix with NLSQ package (github.com/imewei/NLSQ)
- NLSQWrapper provides unified interface with error recovery
- Maintains backward API compatibility
- User-facing Optimistix references removed from public APIs

References:
- NLSQ Package: https://github.com/imewei/NLSQ
- Validation Report: SCIENTIFIC_VALIDATION_REPORT.md
- Production Report: PRODUCTION_READINESS_REPORT.md
"""

from __future__ import annotations

from typing import Any

import numpy as np

# JAX imports with fallback
try:
    import jax.numpy as jnp
    from jax import grad, jit, vmap

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

    def jit(f):
        return f

    def vmap(f, **kwargs):
        return f

    def grad(f):
        return lambda x: np.zeros_like(x)


# Core homodyne imports
try:
    from homodyne.config.manager import ConfigManager
    from homodyne.core.fitting import ParameterSpace
    from homodyne.utils.logging import get_logger, log_performance

    HAS_CORE_MODULES = True
except ImportError:
    HAS_CORE_MODULES = False
    import logging

    def get_logger(name):
        return logging.getLogger(name)

    def log_performance(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


# Optional ParameterManager import (Phase 4.2)
try:
    from homodyne.config.parameter_manager import ParameterManager

    HAS_PARAMETER_MANAGER = True
except ImportError:
    HAS_PARAMETER_MANAGER = False
    ParameterManager = None

# NLSQWrapper import for new implementation
try:
    from homodyne.optimization.nlsq.wrapper import NLSQWrapper, OptimizationResult

    HAS_NLSQ_WRAPPER = True
except ImportError:
    HAS_NLSQ_WRAPPER = False
    NLSQWrapper = None
    OptimizationResult = None

# Multi-start optimization import (v2.8.0)
try:
    from homodyne.optimization.nlsq.multistart import (
        MultiStartConfig,
        MultiStartResult,
        SingleStartResult,
        run_multistart_nlsq,
    )

    HAS_MULTISTART = True
except ImportError:
    HAS_MULTISTART = False
    MultiStartConfig = None
    MultiStartResult = None
    SingleStartResult = None
    run_multistart_nlsq = None

# Export NLSQ availability for tests and external code
NLSQ_AVAILABLE = HAS_NLSQ_WRAPPER and JAX_AVAILABLE

logger = get_logger(__name__)


class NLSQResult:
    """Result container for NLSQ optimization compatible with FitResult."""

    def __init__(
        self,
        parameters: dict[str, float],
        parameter_errors: dict[str, float],
        chi_squared: float,
        reduced_chi_squared: float,
        success: bool,
        message: str,
        n_iterations: int,
        optimization_time: float,
        method: str = "nlsq",
    ):
        self.parameters = parameters
        self.parameter_errors = parameter_errors
        self.chi_squared = chi_squared
        self.reduced_chi_squared = reduced_chi_squared
        self.success = success
        self.message = message
        self.n_iterations = n_iterations
        self.optimization_time = optimization_time
        self.method = method


@log_performance(threshold=1.0)
def fit_nlsq_jax(
    data: dict[str, Any],
    config: ConfigManager,
    initial_params: dict[str, float] | None = None,
    per_angle_scaling: bool = True,  # REQUIRED: per-angle is physically correct
) -> OptimizationResult:
    """NLSQ trust-region nonlinear least squares optimization with per-angle scaling (NEW IMPLEMENTATION).

    Wrapper around NLSQWrapper that provides the public API.
    Uses the NLSQ package (github.com/imewei/NLSQ) for trust-region optimization.

    Primary optimization method implementing the scaled optimization process:
    c₂(φ,t₁,t₂) = 1 + contrast × [c₁(φ,t₁,t₂)]²

    Parameters
    ----------
    data : dict
        XPCS experimental data. Accepts two formats:

        **Format 1 (CLI/loader format)**:
        - 'phi_angles_list': phi angle array (mapped to 'phi')
        - 'c2_exp': experimental correlation data (n_phi, n_t1, n_t2) (mapped to 'g2')
        - 't1': first delay time array
        - 't2': second delay time array
        - 'wavevector_q_list': q-vector array (first element extracted as scalar 'q')
        - 'sigma': (optional) uncertainty array, defaults to 0.01 * ones_like(g2)
        - 'L': (optional) stator-rotor gap (rheology) or sample-detector distance (standard XPCS), defaults to config value or 2000000 Å (200 µm, typical rheology-XPCS gap)
        - 'dt': (optional) time step, defaults to config value or None

        **Format 2 (Direct format)**:
        - 'phi': phi angle array
        - 'g2': experimental correlation data (n_phi, n_t1, n_t2)
        - 't1': first delay time array
        - 't2': second delay time array
        - 'q': wavevector magnitude (scalar)
        - 'sigma': (optional) uncertainty array
        - 'L': (optional) stator-rotor gap or sample-detector distance [Å]
        - 'dt': (optional) time step [s]

    config : ConfigManager
        Configuration manager with optimization settings
    initial_params : dict, optional
        Initial parameter guesses. If None, uses defaults from config.
    per_angle_scaling : bool, default=True
        MUST be True. Per-angle contrast/offset parameters are physically correct as each
        scattering angle has different optical properties and detector responses.
        Legacy scalar mode (False) is no longer supported (removed Nov 2025).

    Returns
    -------
    OptimizationResult
        Optimization result with parameters, uncertainties, and diagnostics

    Raises
    ------
    ImportError
        If NLSQ package is not available
    ValueError
        If data validation fails
    """

    if not HAS_NLSQ_WRAPPER:
        raise ImportError(
            "NLSQWrapper is required for NLSQ optimization. "
            "Ensure homodyne.optimization.nlsq_wrapper is available.",
        )

    logger.info("Starting NLSQ optimization via NLSQWrapper")

    # Determine analysis mode
    analysis_mode = _get_analysis_mode(config)
    logger.info(f"Analysis mode: {analysis_mode}")

    # Set up initial parameters
    per_angle_scaling_initial: dict[str, list[float]] | None = None
    if initial_params is None:
        # Try to load from config first (pass data for contrast/offset estimation)
        initial_params, per_angle_scaling_initial = _load_initial_params_from_config(
            config, analysis_mode, data
        )
        if initial_params is None:
            # Fallback to defaults (estimate contrast/offset from data if available)
            initial_params = _get_default_initial_params(analysis_mode)
            if data is not None:
                contrast_est, offset_est = _estimate_contrast_offset_from_data(data)
                initial_params["contrast"] = contrast_est
                initial_params["offset"] = offset_est
            logger.info("Using default initial parameters")
        else:
            logger.info("Using initial parameters from configuration")
    else:
        # Make a copy so we don't mutate caller-provided dict
        initial_params = initial_params.copy()
        per_angle_scaling_initial = initial_params.pop("per_angle_scaling", None)

    # Convert initial params dict to array
    x0 = _params_to_array(initial_params, analysis_mode)

    # Set up parameter bounds
    # ✅ FIX: Use ParameterManager to load bounds from config (including custom user bounds)
    if HAS_PARAMETER_MANAGER:
        # Handle both ConfigManager objects and plain dicts
        if hasattr(config, "config"):
            config_dict = config.config  # ConfigManager object
        else:
            config_dict = config  # Already a dict

        # Use ParameterManager to get bounds from config (properly loads custom bounds)
        param_manager = ParameterManager(
            config_dict=config_dict, analysis_mode=analysis_mode
        )
        param_names = _get_param_names(analysis_mode)
        bounds_list = param_manager.get_parameter_bounds(param_names)
        # Convert ParameterManager format (list of dicts) to _bounds_to_arrays format (dict of tuples)
        bounds_dict = {b["name"]: (b["min"], b["max"]) for b in bounds_list}
    else:
        # Fallback to ParameterSpace with hardcoded defaults (for backward compatibility)
        param_space = ParameterSpace()
        bounds_dict = _get_parameter_bounds(analysis_mode, param_space)

    lower_bounds, upper_bounds = _bounds_to_arrays(bounds_dict, analysis_mode)
    bounds = (lower_bounds, upper_bounds)

    def _ensure_positive_sigma(obj: Any) -> None:
        if not hasattr(obj, "sigma"):
            return
        sigma_array = np.asarray(obj.sigma, dtype=np.float64)
        if not np.all(np.isfinite(sigma_array)):
            raise ValueError("sigma values must be finite")
        if np.any(sigma_array <= 0):
            raise ValueError("sigma values must be strictly positive")
        obj.sigma = sigma_array

    # Convert data dict to object if needed (NLSQWrapper expects object attributes)
    if isinstance(data, dict):

        class DataObject:
            pass

        data_obj = DataObject()

        # Map CLI data structure keys to NLSQWrapper expected names
        # CLI provides: phi_angles_list, c2_exp, wavevector_q_list
        # NLSQWrapper needs: phi, g2, t1, t2, sigma, q, L, dt (optional)
        key_mapping = {
            "phi_angles_list": "phi",
            "c2_exp": "g2",
        }

        # Apply key mapping and copy all data
        for key, value in data.items():
            # Use mapped name if available, otherwise keep original
            mapped_key = key_mapping.get(key, key)
            setattr(data_obj, mapped_key, value)

        # Extract scalar q from wavevector_q_list if present
        if hasattr(data_obj, "wavevector_q_list"):
            q_list = np.asarray(data_obj.wavevector_q_list)
            if q_list.size > 0:
                data_obj.q = float(q_list[0])  # Take first q-vector
                logger.debug(f"Extracted q = {data_obj.q:.6f} from wavevector_q_list")

        # Generate default sigma (uncertainty) if missing
        if not hasattr(data_obj, "sigma") and hasattr(data_obj, "g2"):
            g2_array = np.asarray(data_obj.g2)
            # Use 1% relative uncertainty as default
            data_obj.sigma = 0.01 * np.ones_like(g2_array)
            logger.debug(f"Generated default sigma: shape {data_obj.sigma.shape}")
        _ensure_positive_sigma(data_obj)

        # Extract 1D time vectors from 2D meshgrids if needed
        # Data loader returns t1_2d, t2_2d as (N, N) meshgrids, but NLSQWrapper expects 1D vectors
        if hasattr(data_obj, "t1"):
            t1 = np.asarray(data_obj.t1)
            if t1.ndim == 2:
                # t1_2d[i, j] = time[i] (constant along j), so extract first column
                data_obj.t1 = t1[:, 0]
                logger.debug(
                    f"Extracted 1D t1 vector from 2D meshgrid: {t1.shape} → {data_obj.t1.shape}",
                )
            elif t1.ndim != 1:
                raise ValueError(f"t1 must be 1D or 2D array, got shape {t1.shape}")

        if hasattr(data_obj, "t2"):
            t2 = np.asarray(data_obj.t2)
            if t2.ndim == 2:
                # t2_2d[i, j] = time[j] (constant along i), so extract first row
                data_obj.t2 = t2[0, :]
                logger.debug(
                    f"Extracted 1D t2 vector from 2D meshgrid: {t2.shape} → {data_obj.t2.shape}",
                )
            elif t2.ndim != 1:
                raise ValueError(f"t2 must be 1D or 2D array, got shape {t2.shape}")

        # Get characteristic length L from config (stator_rotor_gap or sample_detector_distance)
        if not hasattr(data_obj, "L"):
            # Try to get from config - check multiple possible locations
            # Priority 1: analyzer_parameters.geometry.stator_rotor_gap (for rheology-XPCS)
            # Priority 2: experimental_data.geometry.stator_rotor_gap (alternative location)
            # Priority 3: experimental_data.sample_detector_distance (for standard XPCS)
            # Priority 4: Default 100.0 (fallback)
            try:
                # Try analyzer_parameters.geometry.stator_rotor_gap first
                analyzer_params = config.config.get("analyzer_parameters", {})
                geometry = analyzer_params.get("geometry", {})

                if "stator_rotor_gap" in geometry:
                    data_obj.L = float(geometry["stator_rotor_gap"])
                    logger.debug(
                        f"Using stator_rotor_gap L = {data_obj.L:.1f} Å (from config.analyzer_parameters.geometry)",
                    )
                else:
                    # Try experimental_data.geometry.stator_rotor_gap as alternative
                    exp_config = config.config.get("experimental_data", {})
                    exp_geometry = exp_config.get("geometry", {})

                    if "stator_rotor_gap" in exp_geometry:
                        data_obj.L = float(exp_geometry["stator_rotor_gap"])
                        logger.debug(
                            f"Using stator_rotor_gap L = {data_obj.L:.1f} Å (from config.experimental_data.geometry)",
                        )
                    # Fallback to sample_detector_distance
                    elif "sample_detector_distance" in exp_config:
                        data_obj.L = float(exp_config["sample_detector_distance"])
                        logger.debug(
                            f"Using sample_detector_distance L = {data_obj.L:.1f} Å (from config.experimental_data)",
                        )
                    else:
                        data_obj.L = 2000000.0  # Default: 200 µm stator-rotor gap (typical rheology-XPCS)
                        logger.warning(
                            f"No L parameter found in config, using default L = {data_obj.L:.1f} Å (200 µm, typical rheology-XPCS gap)",
                        )
            except (AttributeError, TypeError, ValueError) as e:
                data_obj.L = 2000000.0  # Default: 200 µm stator-rotor gap (typical rheology-XPCS)
                logger.warning(
                    f"Error reading L from config: {e}, using default L = {data_obj.L:.1f} Å (200 µm)",
                )

        # Get time step dt from config if available
        if not hasattr(data_obj, "dt"):
            try:
                # Try analyzer_parameters first (preferred location)
                analyzer_params = config.config.get("analyzer_parameters", {})
                dt_value = analyzer_params.get("dt")

                # Fallback to experimental_data section
                if dt_value is None:
                    exp_config = config.config.get("experimental_data", {})
                    dt_value = exp_config.get("dt")

                if dt_value is not None:
                    data_obj.dt = float(dt_value)
                    logger.debug(f"Using time step dt = {data_obj.dt:.6f} s")
            except (AttributeError, TypeError, ValueError) as e:
                logger.warning(f"Error reading dt from config: {e}")
                # dt is optional, no problem if missing

        data = data_obj
    else:
        _ensure_positive_sigma(data)

    diagnostics_enabled = _is_nlsq_diagnostics_enabled(config)

    # Create wrapper and run optimization
    # Note: enable_recovery=True provides automatic error recovery for production use
    wrapper = NLSQWrapper(enable_large_dataset=True, enable_recovery=True)

    shear_transform_cfg = _extract_shear_transform_config(config)

    result = wrapper.fit(
        data=data,
        config=config,
        initial_params=x0,
        bounds=bounds,
        analysis_mode=analysis_mode,
        per_angle_scaling=per_angle_scaling,
        diagnostics_enabled=diagnostics_enabled,
        shear_transforms=shear_transform_cfg,
        per_angle_scaling_initial=per_angle_scaling_initial,
    )

    logger.info(f"NLSQ optimization completed in {result.execution_time:.3f}s")
    logger.info(
        f"Final χ² = {result.chi_squared:.6f}, reduced χ² = {result.reduced_chi_squared:.6f}",
    )

    return result


def _validate_data(data: dict[str, Any]) -> None:
    """Validate experimental data structure."""
    required_keys = ["wavevector_q_list", "phi_angles_list", "t1", "t2", "c2_exp"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required data key: {key}")

    if data["c2_exp"].shape[0] == 0:
        raise ValueError("Empty experimental data")


def _get_analysis_mode(config: ConfigManager) -> str:
    """Determine analysis mode from configuration."""
    if hasattr(config, "config") and config.config:
        return config.config.get("analysis_mode", "static_isotropic")
    return "static_isotropic"


def _is_nlsq_diagnostics_enabled(config: ConfigManager | dict[str, Any]) -> bool:
    """Return True if optimization.nlsq.diagnostics.enabled is truthy."""

    config_dict: dict[str, Any] | None = None
    if hasattr(config, "config") and config.config:
        config_dict = config.config
    elif isinstance(config, dict):
        config_dict = config

    if not config_dict:
        return False

    return bool(
        config_dict.get("optimization", {})
        .get("nlsq", {})
        .get("diagnostics", {})
        .get("enabled", False)
    )


def _extract_shear_transform_config(
    config: ConfigManager | dict[str, Any],
) -> dict[str, Any]:
    config_dict: dict[str, Any] | None = None
    if hasattr(config, "config") and config.config:
        config_dict = config.config
    elif isinstance(config, dict):
        config_dict = config

    if not config_dict:
        return {}

    return (
        config_dict.get("optimization", {}).get("nlsq", {}).get("shear_transforms", {})
    )


def _load_initial_params_from_config(
    config: ConfigManager,
    analysis_mode: str,
    data: dict[str, Any] | None = None,
) -> tuple[dict[str, float] | None, dict[str, list[float]] | None]:
    """Load initial parameters from configuration file.

    Handles parameter name mapping between config format and code format.
    Estimates contrast/offset from experimental data if not provided in config.

    Parameters
    ----------
    config : ConfigManager
        Configuration manager with initial_parameters section
    analysis_mode : str
        Analysis mode (static_isotropic or laminar_flow)
    data : dict, optional
        Experimental data used to estimate contrast/offset if not in config

    Returns
    -------
    dict or None
        Dictionary of initial parameters, or None if not found in config
    """
    if not hasattr(config, "config") or not config.config:
        return None, None

    config_dict = config.config
    if "initial_parameters" not in config_dict:
        return None, None

    init_params = config_dict["initial_parameters"]
    if "parameter_names" not in init_params or "values" not in init_params:
        logger.warning(
            "Initial parameters in config missing 'parameter_names' or 'values'",
        )
        return None, None

    names = init_params["parameter_names"]
    values = init_params["values"]

    if len(names) != len(values):
        logger.warning(
            f"Parameter name/value count mismatch: {len(names)} names, {len(values)} values",
        )
        return None, None

    # Map config parameter names to code parameter names
    NAME_MAP = {
        "gamma_dot_0": "gamma_dot_t0",
        "gamma_dot_offset": "gamma_dot_t_offset",
        "phi_0": "phi0",
        "D0": "D0",
        "alpha": "alpha",
        "D_offset": "D_offset",
        "beta": "beta",
    }

    # Build parameter dictionary with name mapping
    params = {}
    for name, value in zip(names, values, strict=False):
        mapped_name = NAME_MAP.get(name, name)
        params[mapped_name] = float(value)

    # Add scaling parameters if missing
    # (config typically only includes physical parameters)
    # ✅ FIX (Nov 14, 2025): Use physically reasonable defaults instead of data estimation
    # PROBLEM: Data estimation from diagonal-corrected g2 gives wrong values
    #   - Estimated: contrast~0.055, offset~1.003 (from percentile + max)
    #   - Actual fitted: contrast~0.26, offset~0.77 (from previous successful runs)
    #   - Mismatch causes optimization to get stuck in wrong parameter space
    # SOLUTION: Use typical XPCS values as defaults
    if "contrast" not in params or "offset" not in params:
        # Use typical homodyne XPCS values (empirically validated)
        contrast_default = 0.3  # Typical range [0.1, 0.5] for homodyne detection
        offset_default = 0.8  # Typical range [0.5, 1.0] for baseline

        if "contrast" not in params:
            params["contrast"] = contrast_default
            logger.info(
                f"Using default contrast={contrast_default:.3f} (typical homodyne XPCS)"
            )
        if "offset" not in params:
            params["offset"] = offset_default
            logger.info(
                f"Using default offset={offset_default:.3f} (typical homodyne XPCS)"
            )

    # Validate parameter count matches analysis mode
    expected_count = 5 if "static" in analysis_mode.lower() else 9
    if len(params) != expected_count:
        logger.warning(
            f"Parameter count mismatch for {analysis_mode}: "
            f"got {len(params)}, expected {expected_count}",
        )
        # Don't return None - let validation/clipping handle it

    per_angle_scaling: dict[str, list[float]] | None = None
    per_angle_cfg = init_params.get("per_angle_scaling")
    if isinstance(per_angle_cfg, dict):
        contrast_vals = per_angle_cfg.get("contrast")
        offset_vals = per_angle_cfg.get("offset")
        try:
            contrast_array = (
                [float(x) for x in contrast_vals]
                if isinstance(contrast_vals, (list, tuple))
                else None
            )
            offset_array = (
                [float(x) for x in offset_vals]
                if isinstance(offset_vals, (list, tuple))
                else None
            )
        except (TypeError, ValueError):
            contrast_array = offset_array = None

        if contrast_array and offset_array and len(contrast_array) == len(offset_array):
            per_angle_scaling = {
                "contrast": contrast_array,
                "offset": offset_array,
            }
        elif contrast_array or offset_array:
            logger.warning(
                "per_angle_scaling in initial_parameters must provide equal-length contrast/offset arrays; ignoring overrides",
            )

    logger.debug(f"Loaded {len(params)} parameters from config: {list(params.keys())}")
    return params, per_angle_scaling


def _estimate_contrast_offset_from_data(
    data: dict[str, Any],
) -> tuple[float, float]:
    """Estimate contrast and offset from experimental g2 data.

    For XPCS correlation function: c₂(φ,t₁,t₂) = offset + contrast × [c₁(φ,t₁,t₂)]²

    Parameters
    ----------
    data : dict
        Experimental data with 'g2' or 'c2_exp' key containing correlation data

    Returns
    -------
    contrast : float
        Estimated contrast parameter (amplitude of correlations)
    offset : float
        Estimated offset parameter (baseline of g2)
    """
    # Extract g2 data (try multiple possible key names)
    # Note: Cannot use `or` operator with numpy arrays as it evaluates truth value
    g2 = data.get("g2")
    if g2 is None:
        g2 = data.get("c2_exp")

    if g2 is None:
        logger.warning(
            "Could not estimate contrast/offset: no 'g2' or 'c2_exp' in data. "
            "Using generic defaults (0.5, 1.0)"
        )
        return 0.5, 1.0

    # Convert to numpy array if needed
    g2_array = np.asarray(g2)

    # Estimate offset from baseline (5th percentile to avoid outliers)
    offset_est = float(np.percentile(g2_array, 5))

    # Estimate contrast from amplitude (max - baseline)
    # For c₂ = offset + contrast × [c₁]², max occurs at c₁²=1
    max_g2 = float(np.max(g2_array))
    contrast_est = max_g2 - offset_est

    # Sanity checks
    if contrast_est <= 0 or offset_est <= 0:
        logger.warning(
            f"Invalid estimated contrast={contrast_est:.3f} or offset={offset_est:.3f}. "
            f"Using generic defaults (0.5, 1.0)"
        )
        return 0.5, 1.0

    logger.info(
        f"Estimated scaling parameters from data: "
        f"contrast={contrast_est:.4f}, offset={offset_est:.4f} "
        f"(g2 range: [{np.min(g2_array):.4f}, {np.max(g2_array):.4f}])"
    )

    return contrast_est, offset_est


def _get_default_initial_params(analysis_mode: str) -> dict[str, float]:
    """Get default initial parameters for analysis mode.

    NOTE: This function provides generic physical parameter defaults.
    Contrast and offset should be estimated from experimental data
    using _estimate_contrast_offset_from_data() before calling this function.
    """
    # Static isotropic mode (3 parameters)
    if "static" in analysis_mode.lower():
        return {
            "contrast": 0.5,  # Generic default - should be replaced with data estimate
            "offset": 1.0,  # Generic default - should be replaced with data estimate
            "D0": 10000.0,
            "alpha": -1.5,
            "D_offset": 0.0,
        }
    # Laminar flow mode (7 parameters)
    else:
        return {
            "contrast": 0.5,  # Generic default - should be replaced with data estimate
            "offset": 1.0,  # Generic default - should be replaced with data estimate
            "D0": 10000.0,
            "alpha": -1.5,
            "D_offset": 0.0,
            "gamma_dot_t0": 0.001,
            "beta": 0.0,
            "gamma_dot_t_offset": 0.0,
            "phi0": 0.0,
        }


def _create_residual_function(
    data: dict[str, Any],
    theory_engine: Any,
    analysis_mode: str,
) -> callable:
    """Create residual function for NLSQ least squares optimization."""

    # Extract q and L as concrete scalars OUTSIDE the residual function
    # This prevents JAX tracing issues
    q_list = data.get("wavevector_q_list", [0.0054])
    q_scalar = float(q_list[0]) if len(q_list) > 0 else 0.0054
    L_scalar = 2000000.0  # Default: 200 µm stator-rotor gap (typical rheology-XPCS)

    def residual_fn(params_array):
        """Residual function: returns residuals vector."""
        params_dict = _array_to_params(params_array, analysis_mode)

        # Extract scaling parameters
        contrast = params_dict["contrast"]
        offset = params_dict["offset"]

        # Create physical params dict (without contrast and offset)
        physical_params = {
            k: v for k, v in params_dict.items() if k not in ["contrast", "offset"]
        }

        # Convert physical params to array format expected by theory engine
        if "static" in analysis_mode.lower():
            params_array_physical = jnp.array(
                [
                    physical_params["D0"],
                    physical_params["alpha"],
                    physical_params["D_offset"],
                ],
            )
        else:
            params_array_physical = jnp.array(
                [
                    physical_params["D0"],
                    physical_params["alpha"],
                    physical_params["D_offset"],
                    physical_params["gamma_dot_t0"],
                    physical_params["beta"],
                    physical_params["gamma_dot_t_offset"],
                    physical_params["phi0"],
                ],
            )

        # Use pre-extracted concrete scalars (extracted outside to avoid JAX tracing)
        c2_theory = theory_engine.compute_g2(
            params_array_physical,
            data["t1"],
            data["t2"],
            data["phi_angles_list"],
            q_scalar,  # Use pre-extracted scalar
            L_scalar,  # Use pre-extracted scalar
            contrast,
            offset,
        )

        # Return residuals (not squared)
        residuals = data["c2_exp"] - c2_theory
        return residuals.flatten()

    return residual_fn


def _get_parameter_bounds(
    analysis_mode: str,
    param_space: ParameterSpace,
) -> dict[str, tuple[float, float]]:
    """Get parameter bounds for analysis mode."""
    bounds = {
        "contrast": param_space.contrast_bounds,
        "offset": param_space.offset_bounds,
        "D0": param_space.D0_bounds,
        "alpha": param_space.alpha_bounds,
        "D_offset": param_space.D_offset_bounds,
    }

    if "laminar" in analysis_mode.lower():
        bounds.update(
            {
                "gamma_dot_t0": param_space.gamma_dot_t0_bounds,
                "beta": param_space.beta_bounds,
                "gamma_dot_t_offset": param_space.gamma_dot_t_offset_bounds,
                "phi0": param_space.phi0_bounds,
            },
        )

    return bounds


def _get_param_names(analysis_mode: str) -> list[str]:
    """Get parameter names for a given analysis mode.

    Parameters
    ----------
    analysis_mode : str
        Analysis mode (e.g., 'static', 'laminar_flow')

    Returns
    -------
    list[str]
        List of parameter names in the order they appear in the parameter array
    """
    if "static" in analysis_mode.lower():
        return ["contrast", "offset", "D0", "alpha", "D_offset"]
    else:
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


def _params_to_array(params: dict[str, float], analysis_mode: str) -> jnp.ndarray:
    """Convert parameter dictionary to array."""
    if "static" in analysis_mode.lower():
        return jnp.array(
            [
                params["contrast"],
                params["offset"],
                params["D0"],
                params["alpha"],
                params["D_offset"],
            ],
        )
    else:
        return jnp.array(
            [
                params["contrast"],
                params["offset"],
                params["D0"],
                params["alpha"],
                params["D_offset"],
                params["gamma_dot_t0"],
                params["beta"],
                params["gamma_dot_t_offset"],
                params["phi0"],
            ],
        )


def _array_to_params(array: jnp.ndarray, analysis_mode: str) -> dict[str, Any]:
    """Convert parameter array to dictionary.

    Returns JAX arrays as-is to avoid tracing issues.
    Conversion to Python floats should only happen at the final step.
    """
    if "static" in analysis_mode.lower():
        return {
            "contrast": array[0],
            "offset": array[1],
            "D0": array[2],
            "alpha": array[3],
            "D_offset": array[4],
        }
    else:
        return {
            "contrast": array[0],
            "offset": array[1],
            "D0": array[2],
            "alpha": array[3],
            "D_offset": array[4],
            "gamma_dot_t0": array[5],
            "beta": array[6],
            "gamma_dot_t_offset": array[7],
            "phi0": array[8],
        }


def _bounds_to_arrays(
    bounds: dict[str, tuple[float, float]],
    analysis_mode: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert bounds dictionary to lower/upper bound arrays."""
    if "static" in analysis_mode.lower():
        param_order = ["contrast", "offset", "D0", "alpha", "D_offset"]
    else:
        param_order = [
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

    lower = jnp.array([bounds[key][0] for key in param_order])
    upper = jnp.array([bounds[key][1] for key in param_order])

    return lower, upper


def _get_optimizer_config(config: ConfigManager) -> dict[str, Any]:
    """Get NLSQ optimizer configuration from config."""
    default_config = {
        "method": "levenberg_marquardt",
        "max_iterations": 10000,
        "tolerance": 1e-8,
        "verbose": False,
    }

    if hasattr(config, "config") and config.config:
        lsq_config = config.config.get("optimization", {}).get("lsq", {})
        default_config.update(lsq_config)

    return default_config


def _check_convergence(result: Any) -> bool:
    """Check if NLSQ optimization converged."""
    return getattr(result, "success", False)


def _get_optimization_message(result: Any) -> str:
    """Get optimization status message from NLSQ result."""
    if hasattr(result, "result_flag") and result.result_flag is not None:
        return str(result.result_flag)
    elif result.success:
        return "Optimization converged successfully"
    else:
        return "Optimization failed to converge"


def _get_iteration_count(result: Any) -> int:
    """Get iteration count from NLSQ result."""
    if hasattr(result, "stats") and "num_steps" in result.stats:
        return int(result.stats["num_steps"])
    return 0


# =============================================================================
# Multi-Start Optimization Entry Point (v2.8.0)
# =============================================================================


@log_performance(threshold=1.0)
def fit_nlsq_multistart(
    data: dict[str, Any],
    config: ConfigManager,
    initial_params: dict[str, float] | None = None,
    per_angle_scaling: bool = True,
) -> MultiStartResult:
    """Multi-start NLSQ optimization with dataset size-based strategy selection.

    This function explores the parameter space using Latin Hypercube Sampling
    to avoid local minima. The strategy is automatically selected based on
    dataset size:

    - < 1M points: Full multi-start (N complete fits)
    - 1M - 100M points: Subsample multi-start (multi-start on 500K subsample)
    - > 100M points: Phase 1 multi-start (parallel warmup, single Gauss-Newton)

    Parameters
    ----------
    data : dict[str, Any]
        XPCS experimental data with keys:
        - wavevector_q_list: Q-vector values
        - phi_angles_list: Azimuthal angles
        - t1, t2: Time coordinates
        - c2_exp: Experimental g2 correlation data
        - sigma (optional): Error weights
    config : ConfigManager
        Configuration manager with optimization.nlsq.multi_start settings.
    initial_params : dict[str, float], optional
        Initial parameter guess. If provided, included as one of the starts.
    per_angle_scaling : bool
        Whether to use per-angle contrast/offset scaling. Default: True.

    Returns
    -------
    MultiStartResult
        Aggregated results including:
        - best: Best result by chi-squared
        - all_results: All optimization attempts
        - strategy_used: "full", "subsample", or "phase1"
        - n_unique_basins: Number of distinct local minima found
        - degeneracy_detected: Whether parameter degeneracy was detected

    Raises
    ------
    ImportError
        If multi-start module is not available.
    ValueError
        If multi-start is not enabled in configuration.

    Examples
    --------
    >>> config = ConfigManager("config.yaml")
    >>> # Ensure multi_start.enable: true in config
    >>> result = fit_nlsq_multistart(data, config)
    >>> print(f"Best chi²: {result.best.chi_squared:.4g}")
    >>> print(f"Strategy used: {result.strategy_used}")
    >>> if result.degeneracy_detected:
    ...     print(f"Warning: {result.n_unique_basins} distinct basins found")
    """
    if not HAS_MULTISTART:
        raise ImportError(
            "Multi-start optimization requires homodyne.optimization.nlsq.multistart. "
            "Ensure the multistart module is properly installed."
        )

    if not HAS_NLSQ_WRAPPER:
        raise ImportError("NLSQWrapper is required for multi-start optimization")

    # Extract multi-start config
    nlsq_dict = config.config.get("optimization", {}).get("nlsq", {})
    multi_start_dict = nlsq_dict.get("multi_start", {})

    if not multi_start_dict.get("enable", False):
        raise ValueError(
            "Multi-start optimization is not enabled. "
            "Set optimization.nlsq.multi_start.enable: true in config."
        )

    from homodyne.optimization.nlsq.config import NLSQConfig

    nlsq_config = NLSQConfig.from_dict(nlsq_dict)
    ms_config = MultiStartConfig.from_nlsq_config(nlsq_config)

    # Validate data
    _validate_data(data)

    # Get analysis mode and parameter setup
    analysis_mode = _get_analysis_mode(config)
    param_space = ParameterSpace() if HAS_CORE_MODULES else None

    # Get bounds
    if HAS_PARAMETER_MANAGER:
        param_manager = ParameterManager(config)
        bounds_list = param_manager.get_all_bounds()
        bounds_dict = {b["name"]: (b["min"], b["max"]) for b in bounds_list}
    else:
        bounds_dict = _get_parameter_bounds(analysis_mode, param_space)

    lower_bounds, upper_bounds = _bounds_to_arrays(bounds_dict, analysis_mode)
    bounds_array = np.column_stack([lower_bounds, upper_bounds])

    # Create single fit function wrapper
    def single_fit_func(
        fit_data: dict[str, Any], start_params: np.ndarray
    ) -> SingleStartResult:
        """Wrapper for single NLSQ fit."""
        import time

        start_time = time.perf_counter()

        # Convert array to dict
        param_names = _get_param_names(analysis_mode)
        params_dict = {
            name: float(start_params[i]) for i, name in enumerate(param_names)
        }

        try:
            result = fit_nlsq_jax(
                data=fit_data,
                config=config,
                initial_params=params_dict,
                per_angle_scaling=per_angle_scaling,
            )

            return SingleStartResult(
                start_idx=0,
                initial_params=start_params,
                final_params=np.array(result.popt),
                chi_squared=result.chi_squared,
                reduced_chi_squared=result.reduced_chi_squared,
                success=result.success,
                status=0,
                message=result.message,
                n_iterations=result.n_iterations,
                n_fev=result.n_fev,
                wall_time=time.perf_counter() - start_time,
                covariance=result.pcov if hasattr(result, "pcov") else None,
            )
        except Exception as e:
            return SingleStartResult(
                start_idx=0,
                initial_params=start_params,
                final_params=start_params,
                chi_squared=np.inf,
                success=False,
                message=str(e),
                wall_time=time.perf_counter() - start_time,
            )

    # Create cost function for screening
    def cost_func(params: np.ndarray) -> float:
        """Quick cost evaluation for screening.

        Uses a heuristic based on distance from bounds center rather than
        full residual evaluation for efficiency during screening phase.
        """
        try:
            # Check if params are at bounds (return large cost)
            for i, (low, high) in enumerate(
                zip(lower_bounds, upper_bounds, strict=True)
            ):
                if params[i] <= low or params[i] >= high:
                    return 1e20

            # Approximate cost from parameter distance to center
            center = (lower_bounds + upper_bounds) / 2
            scale = upper_bounds - lower_bounds
            normalized_dist = np.sum(((params - center) / scale) ** 2)
            return normalized_dist
        except Exception:
            return 1e20

    # Run multi-start optimization
    logger.info(
        f"Starting multi-start NLSQ with {ms_config.n_starts} starts, "
        f"strategy will be auto-selected based on dataset size"
    )

    result = run_multistart_nlsq(
        data=data,
        bounds=bounds_array,
        config=ms_config,
        single_fit_func=single_fit_func,
        cost_func=cost_func if ms_config.use_screening else None,
    )

    logger.info(
        f"Multi-start complete: strategy={result.strategy_used}, "
        f"best χ²={result.best.chi_squared:.4g}, "
        f"basins={result.n_unique_basins}"
    )

    return result
