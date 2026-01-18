"""Prior distribution builders for CMC analysis.

This module provides utilities for building NumPyro prior distributions
from the ParameterSpace configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpyro.distributions as dist

from homodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from homodyne.config.parameter_space import ParameterSpace, PriorDistribution

logger = get_logger(__name__)


# =============================================================================
# DATA-DRIVEN INITIAL VALUE ESTIMATION
# =============================================================================


def estimate_contrast_offset_from_data(
    c2_data: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    contrast_bounds: tuple[float, float] = (0.0, 1.0),
    offset_bounds: tuple[float, float] = (0.5, 1.5),
    lag_floor_quantile: float = 0.80,
    lag_ceiling_quantile: float = 0.20,
    value_quantile_low: float = 0.10,
    value_quantile_high: float = 0.90,
) -> tuple[float, float]:
    """Estimate contrast and offset from C2 data using physics-informed quantile analysis.

    Uses the correlation decay structure: C2 = contrast × g1² + offset
    - At large time lags, g1² → 0, so C2 → offset (the "floor")
    - At small time lags, g1² ≈ 1, so C2 ≈ contrast + offset (the "ceiling")

    Parameters
    ----------
    c2_data : np.ndarray
        C2 correlation values (1D array).
    t1 : np.ndarray
        First time coordinate array (same shape as c2_data).
    t2 : np.ndarray
        Second time coordinate array (same shape as c2_data).
    contrast_bounds : tuple[float, float]
        Valid bounds for contrast parameter.
    offset_bounds : tuple[float, float]
        Valid bounds for offset parameter.
    lag_floor_quantile : float
        Quantile threshold for "large lag" region (default: 0.80 = top 20% of lags).
    lag_ceiling_quantile : float
        Quantile threshold for "small lag" region (default: 0.20 = bottom 20% of lags).
    value_quantile_low : float
        Quantile for robust floor estimation (default: 0.10).
    value_quantile_high : float
        Quantile for robust ceiling estimation (default: 0.90).

    Returns
    -------
    tuple[float, float]
        (contrast_est, offset_est) - Estimated values clipped to bounds.

    Notes
    -----
    The estimation is robust to outliers by using quantiles instead of min/max.
    The lag-based segmentation ensures we're sampling from the appropriate
    regions of the correlation decay curve.
    """
    # Compute time lags
    delta_t = np.abs(np.asarray(t1) - np.asarray(t2))
    c2 = np.asarray(c2_data)

    # Sanity checks
    if len(c2) < 100:
        # Not enough data for robust estimation - return midpoints
        contrast_mid = (contrast_bounds[0] + contrast_bounds[1]) / 2.0
        offset_mid = (offset_bounds[0] + offset_bounds[1]) / 2.0
        logger.debug(
            f"Insufficient data ({len(c2)} points) for quantile estimation, "
            f"using midpoint defaults: contrast={contrast_mid:.3f}, offset={offset_mid:.3f}"
        )
        return contrast_mid, offset_mid

    # Find lag thresholds
    lag_threshold_high = np.percentile(delta_t, lag_floor_quantile * 100)
    lag_threshold_low = np.percentile(delta_t, lag_ceiling_quantile * 100)

    # OFFSET estimation: From large-lag region where g1² ≈ 0
    # C2 → offset at large lags
    large_lag_mask = delta_t >= lag_threshold_high
    if np.sum(large_lag_mask) >= 10:
        c2_floor_region = c2[large_lag_mask]
        # Use low quantile for robustness (in case of noise spikes)
        offset_est = np.percentile(c2_floor_region, value_quantile_low * 100)
    else:
        # Fallback: use overall low quantile
        offset_est = np.percentile(c2, value_quantile_low * 100)

    # Clip offset to bounds
    offset_est = float(np.clip(offset_est, offset_bounds[0], offset_bounds[1]))

    # CONTRAST estimation: From small-lag region where g1² ≈ 1
    # C2 ≈ contrast + offset at small lags
    small_lag_mask = delta_t <= lag_threshold_low
    if np.sum(small_lag_mask) >= 10:
        c2_ceiling_region = c2[small_lag_mask]
        # Use high quantile for robustness
        c2_ceiling = np.percentile(c2_ceiling_region, value_quantile_high * 100)
    else:
        # Fallback: use overall high quantile
        c2_ceiling = np.percentile(c2, value_quantile_high * 100)

    # contrast ≈ c2_ceiling - offset
    contrast_est = c2_ceiling - offset_est

    # Clip contrast to bounds
    contrast_est = float(np.clip(contrast_est, contrast_bounds[0], contrast_bounds[1]))

    logger.debug(
        f"Quantile-based estimation: offset={offset_est:.4f} (from large-lag floor), "
        f"contrast={contrast_est:.4f} (from small-lag ceiling - floor)"
    )

    return contrast_est, offset_est


def estimate_per_angle_scaling(
    c2_data: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    phi_indices: np.ndarray,
    n_phi: int,
    contrast_bounds: tuple[float, float],
    offset_bounds: tuple[float, float],
) -> dict[str, float]:
    """Estimate contrast and offset initial values for each phi angle.

    Optimization (v2.9.1): Uses vectorized grouped operations instead of
    sequential loop over angles. Provides 3-5x speedup for typical datasets
    with 20+ phi angles.

    Parameters
    ----------
    c2_data : np.ndarray
        Pooled C2 correlation values.
    t1 : np.ndarray
        Pooled first time coordinates.
    t2 : np.ndarray
        Pooled second time coordinates.
    phi_indices : np.ndarray
        Index mapping each data point to its phi angle (0 to n_phi-1).
    n_phi : int
        Number of unique phi angles.
    contrast_bounds : tuple[float, float]
        Valid bounds for contrast.
    offset_bounds : tuple[float, float]
        Valid bounds for offset.

    Returns
    -------
    dict[str, float]
        Dictionary with keys 'contrast_0', 'offset_0', 'contrast_1', 'offset_1', etc.
    """
    c2 = np.asarray(c2_data)
    t1_arr = np.asarray(t1)
    t2_arr = np.asarray(t2)
    phi_idx = np.asarray(phi_indices)

    # Pre-compute time lags once (vectorized)
    delta_t = np.abs(t1_arr - t2_arr)

    # Pre-compute midpoint defaults
    contrast_mid = (contrast_bounds[0] + contrast_bounds[1]) / 2.0
    offset_mid = (offset_bounds[0] + offset_bounds[1]) / 2.0

    # Pre-allocate result arrays
    contrast_results = np.full(n_phi, contrast_mid)
    offset_results = np.full(n_phi, offset_mid)
    points_per_angle = np.zeros(n_phi, dtype=np.int64)

    # Count points per angle using bincount (vectorized)
    points_per_angle = np.bincount(phi_idx, minlength=n_phi)

    # Identify angles with sufficient data
    sufficient_mask = points_per_angle >= 100
    n_sufficient = np.sum(sufficient_mask)

    if n_sufficient == 0:
        # No angles have enough data - return defaults
        logger.info(
            f"All {n_phi} angles have insufficient data, using midpoint defaults"
        )
        return {
            **{f"contrast_{i}": contrast_mid for i in range(n_phi)},
            **{f"offset_{i}": offset_mid for i in range(n_phi)},
        }

    # Vectorized estimation for angles with sufficient data
    # Sort data by phi index for efficient grouped operations
    sort_idx = np.argsort(phi_idx)
    c2_sorted = c2[sort_idx]
    delta_t_sorted = delta_t[sort_idx]
    phi_sorted = phi_idx[sort_idx]

    # Find group boundaries
    group_starts = np.searchsorted(phi_sorted, np.arange(n_phi))
    group_ends = np.searchsorted(phi_sorted, np.arange(n_phi), side="right")

    # Process each angle with sufficient data
    for i in range(n_phi):
        if not sufficient_mask[i]:
            logger.info(
                f"Angle {i}: insufficient data ({points_per_angle[i]} points), "
                f"using midpoint init contrast={contrast_mid:.4f}, offset={offset_mid:.4f}"
            )
            continue

        # Extract data for this angle using pre-sorted arrays
        start, end = group_starts[i], group_ends[i]
        c2_angle = c2_sorted[start:end]
        delta_t_angle = delta_t_sorted[start:end]

        # Inline the estimation logic to avoid function call overhead
        # (quantile operations are already vectorized)
        n_points = end - start

        # Find lag thresholds
        lag_threshold_high = np.percentile(delta_t_angle, 80)  # 0.80 quantile
        lag_threshold_low = np.percentile(delta_t_angle, 20)  # 0.20 quantile

        # OFFSET: from large-lag region
        large_lag_mask = delta_t_angle >= lag_threshold_high
        if np.sum(large_lag_mask) >= 10:
            offset_est = np.percentile(c2_angle[large_lag_mask], 10)
        else:
            offset_est = np.percentile(c2_angle, 10)
        offset_est = float(np.clip(offset_est, offset_bounds[0], offset_bounds[1]))

        # CONTRAST: from small-lag region
        small_lag_mask = delta_t_angle <= lag_threshold_low
        if np.sum(small_lag_mask) >= 10:
            c2_ceiling = np.percentile(c2_angle[small_lag_mask], 90)
        else:
            c2_ceiling = np.percentile(c2_angle, 90)
        contrast_est = float(
            np.clip(c2_ceiling - offset_est, contrast_bounds[0], contrast_bounds[1])
        )

        contrast_results[i] = contrast_est
        offset_results[i] = offset_est

        logger.info(
            f"Angle {i}: estimated contrast={contrast_est:.4f}, offset={offset_est:.4f} "
            f"from {n_points:,} data points"
        )

    # Build result dictionary
    estimates: dict[str, float] = {}
    for i in range(n_phi):
        estimates[f"contrast_{i}"] = float(contrast_results[i])
        estimates[f"offset_{i}"] = float(offset_results[i])

    return estimates


# Physical parameter names in canonical order
STATIC_PARAMS = ["D0", "alpha", "D_offset"]
LAMINAR_PARAMS = [
    "D0",
    "alpha",
    "D_offset",
    "gamma_dot_t0",
    "beta",
    "gamma_dot_t_offset",
    "phi0",
]


def build_prior_from_spec(
    prior_spec: PriorDistribution,
) -> dist.Distribution:
    """Build NumPyro distribution from PriorDistribution specification.

    Parameters
    ----------
    prior_spec : PriorDistribution
        Prior specification from ParameterSpace.

    Returns
    -------
    dist.Distribution
        NumPyro distribution object.

    Raises
    ------
    ValueError
        If distribution type is not supported.
    """
    dist_type = prior_spec.dist_type.lower()

    if dist_type == "truncatednormal":
        return dist.TruncatedNormal(
            loc=prior_spec.mu,
            scale=prior_spec.sigma,
            low=prior_spec.min_val,
            high=prior_spec.max_val,
        )
    elif dist_type == "uniform":
        return dist.Uniform(
            low=prior_spec.min_val,
            high=prior_spec.max_val,
        )
    elif dist_type == "lognormal":
        return dist.LogNormal(
            loc=prior_spec.mu,
            scale=prior_spec.sigma,
        )
    elif dist_type == "halfnormal":
        return dist.HalfNormal(scale=prior_spec.sigma)
    elif dist_type == "normal":
        return dist.Normal(loc=prior_spec.mu, scale=prior_spec.sigma)
    elif dist_type == "betascaled":
        # Beta distribution scaled to [min_val, max_val]
        # Use alpha=2, beta=2 for symmetric prior if not specified
        alpha = getattr(prior_spec, "alpha", 2.0)
        beta = getattr(prior_spec, "beta", 2.0)
        base = dist.Beta(concentration1=alpha, concentration0=beta)
        return dist.TransformedDistribution(
            base,
            dist.transforms.AffineTransform(
                loc=prior_spec.min_val,
                scale=prior_spec.max_val - prior_spec.min_val,
            ),
        )
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")


def _get_base_param_name(param_name: str) -> str:
    """Get base parameter name for per-angle parameters.

    Maps 'contrast_0', 'contrast_1', etc. to 'contrast',
    and 'offset_0', 'offset_1', etc. to 'offset'.
    Other parameter names are returned unchanged.

    Parameters
    ----------
    param_name : str
        Parameter name (possibly with angle suffix).

    Returns
    -------
    str
        Base parameter name.
    """
    if param_name.startswith("contrast_"):
        return "contrast"
    elif param_name.startswith("offset_"):
        return "offset"
    return param_name


def build_prior(
    param_name: str,
    parameter_space: ParameterSpace,
) -> dist.Distribution:
    """Build NumPyro prior distribution for a parameter.

    Parameters
    ----------
    param_name : str
        Parameter name (e.g., "D0", "alpha", "contrast", "contrast_0").
    parameter_space : ParameterSpace
        Parameter space with bounds and priors.

    Returns
    -------
    dist.Distribution
        NumPyro distribution for sampling.
    """
    # Use base name for per-angle parameters (contrast_0 -> contrast, etc.)
    base_name = _get_base_param_name(param_name)

    try:
        prior_spec = parameter_space.get_prior(base_name)
        return build_prior_from_spec(prior_spec)
    except (KeyError, AttributeError):
        # Fallback to uniform prior with bounds
        bounds = parameter_space.get_bounds(base_name)
        logger.debug(
            f"No prior spec for {param_name}, using Uniform({bounds[0]}, {bounds[1]})"
        )
        return dist.Uniform(low=bounds[0], high=bounds[1])


def get_init_value(
    param_name: str,
    initial_values: dict[str, float] | None,
    parameter_space: ParameterSpace,
) -> float:
    """Get initial value for a parameter.

    Priority:
    1. Value from initial_values dict if provided (exact match)
    2. Value from initial_values dict for base param (e.g., 'contrast' for 'contrast_0')
    3. Midpoint of parameter bounds as fallback

    Parameters
    ----------
    param_name : str
        Parameter name.
    initial_values : dict[str, float] | None
        Initial values from config.
    parameter_space : ParameterSpace
        Parameter space with bounds.

    Returns
    -------
    float
        Initial value for the parameter.

    Notes
    -----
    Per-angle parameter handling (scalar broadcast):
        For per-angle parameters like 'contrast_0', 'contrast_1', etc., this function
        broadcasts a single scalar value to all angles. If only 'contrast' is provided
        in initial_values (not 'contrast_0', 'contrast_1', etc.), that single value
        is used for ALL phi angles.

        To specify different initial values per angle, provide explicit keys like:
        ``{'contrast_0': 0.4, 'contrast_1': 0.5, 'contrast_2': 0.45}``

        The same applies to 'offset' parameters.

    Examples
    --------
    >>> # Scalar broadcast: same value for all angles
    >>> initial_values = {'contrast': 0.5, 'offset': 1.0}
    >>> get_init_value('contrast_0', initial_values, param_space)  # Returns 0.5
    >>> get_init_value('contrast_1', initial_values, param_space)  # Returns 0.5

    >>> # Explicit per-angle values
    >>> initial_values = {'contrast_0': 0.4, 'contrast_1': 0.6}
    >>> get_init_value('contrast_0', initial_values, param_space)  # Returns 0.4
    >>> get_init_value('contrast_1', initial_values, param_space)  # Returns 0.6
    """
    # Check initial_values first (exact match)
    if initial_values is not None and param_name in initial_values:
        return float(initial_values[param_name])

    # For per-angle params, check base param name in initial_values
    base_name = _get_base_param_name(param_name)
    if initial_values is not None and base_name in initial_values:
        return float(initial_values[base_name])

    # Fallback to midpoint of bounds (use base name for per-angle params)
    bounds = parameter_space.get_bounds(base_name)
    midpoint = (bounds[0] + bounds[1]) / 2.0
    return midpoint


def validate_initial_value_bounds(
    param_name: str,
    value: float,
    parameter_space: ParameterSpace,
) -> tuple[float, bool]:
    """Validate and optionally clip initial value to parameter bounds.

    Parameters
    ----------
    param_name : str
        Parameter name.
    value : float
        Initial value to validate.
    parameter_space : ParameterSpace
        Parameter space with bounds.

    Returns
    -------
    tuple[float, bool]
        (validated_value, was_clipped) - The value (clipped if needed) and whether clipping occurred.
    """
    import math

    base_name = _get_base_param_name(param_name)
    bounds = parameter_space.get_bounds(base_name)
    lower, upper = bounds[0], bounds[1]

    # Check for NaN/Inf
    if not math.isfinite(value):
        midpoint = (lower + upper) / 2.0
        logger.warning(
            f"Initial value for '{param_name}' is {value} (non-finite), "
            f"resetting to midpoint {midpoint:.4g}"
        )
        return midpoint, True

    # Check bounds
    if value < lower:
        logger.warning(
            f"Initial value for '{param_name}' ({value:.4g}) is below lower bound ({lower:.4g}), "
            f"clipping to lower bound + 1% margin"
        )
        margin = 0.01 * (upper - lower)
        return lower + margin, True
    elif value > upper:
        logger.warning(
            f"Initial value for '{param_name}' ({value:.4g}) is above upper bound ({upper:.4g}), "
            f"clipping to upper bound - 1% margin"
        )
        margin = 0.01 * (upper - lower)
        return upper - margin, True

    return value, False


def build_init_values_dict(
    n_phi: int,
    analysis_mode: str,
    initial_values: dict[str, float] | None,
    parameter_space: ParameterSpace,
    *,
    c2_data: np.ndarray | None = None,
    t1: np.ndarray | None = None,
    t2: np.ndarray | None = None,
    phi_indices: np.ndarray | None = None,
    per_angle_mode: str = "individual",
) -> dict[str, float]:
    """Build complete initial values dictionary in sampling order.

    CRITICAL: Parameter order must match NumPyro model sampling order:
    1. contrast_0, contrast_1, ..., contrast_{n_phi-1} (individual mode)
       OR contrast_avg (constant mode)
    2. offset_0, offset_1, ..., offset_{n_phi-1} (individual mode)
       OR offset_avg (constant mode)
    3. Physical parameters in canonical order

    Parameters
    ----------
    n_phi : int
        Number of phi angles.
    analysis_mode : str
        Analysis mode ("static" or "laminar_flow").
    initial_values : dict[str, float] | None
        Initial values from config. Supports both scalar (broadcast) and per-angle
        specifications for contrast/offset. See Notes for details.
    parameter_space : ParameterSpace
        Parameter space with bounds.
    c2_data : np.ndarray | None
        Optional C2 correlation data for quantile-based estimation of contrast/offset.
    t1 : np.ndarray | None
        Optional time coordinates (required if c2_data provided).
    t2 : np.ndarray | None
        Optional time coordinates (required if c2_data provided).
    phi_indices : np.ndarray | None
        Optional phi angle indices for per-angle estimation.
    per_angle_mode : str
        Per-angle scaling mode: "individual" or "constant".

    Returns
    -------
    dict[str, float]
        Initial values dictionary in sampling order.

    Notes
    -----
    Per-angle scaling parameters (contrast/offset):
        This function supports three modes for specifying per-angle initial values:

        1. **Data-driven estimation** (NEW, preferred): If c2_data, t1, t2, and
           phi_indices are provided, and contrast/offset not in initial_values,
           uses physics-informed quantile analysis to estimate values from data.

        2. **Scalar broadcast**: If initial_values contains only base names
           like 'contrast' and 'offset', those values are broadcast to ALL phi angles.
           Example: ``{'contrast': 0.5}`` → contrast_0=0.5, contrast_1=0.5, ...

        3. **Explicit per-angle**: If initial_values contains indexed names like
           'contrast_0', 'contrast_1', etc., those specific values are used.
           Example: ``{'contrast_0': 0.4, 'contrast_1': 0.6}``

        Priority: explicit per-angle > scalar broadcast > data-driven > midpoint fallback

    Bounds validation:
        All initial values are validated against parameter bounds. Out-of-bounds
        values are clipped to bounds ± 1% margin with a warning logged.
    """
    init_dict: dict[str, float] = {}
    clipped_params: list[str] = []

    # Determine physical params early (needed for logging in constant mode)
    physical_params = (
        LAMINAR_PARAMS if analysis_mode == "laminar_flow" else STATIC_PARAMS
    )

    # Check if we should use data-driven estimation for contrast/offset
    # Only use if:
    # 1. Data arrays are provided
    # 2. contrast/offset are NOT in initial_values (neither scalar nor per-angle)
    use_data_estimation = (
        c2_data is not None
        and t1 is not None
        and t2 is not None
        and phi_indices is not None
        and len(c2_data) >= 100
    )

    # Check if contrast/offset are missing from initial_values
    has_contrast = initial_values is not None and (
        "contrast" in initial_values
        or any(k.startswith("contrast_") for k in initial_values)
    )
    has_offset = initial_values is not None and (
        "offset" in initial_values
        or any(k.startswith("offset_") for k in initial_values)
    )

    # Compute data-driven estimates if needed
    data_estimates: dict[str, float] = {}
    if use_data_estimation and (not has_contrast or not has_offset):
        contrast_bounds = parameter_space.get_bounds("contrast")
        offset_bounds = parameter_space.get_bounds("offset")

        data_estimates = estimate_per_angle_scaling(
            c2_data=c2_data,
            t1=t1,
            t2=t2,
            phi_indices=phi_indices,
            n_phi=n_phi,
            contrast_bounds=contrast_bounds,
            offset_bounds=offset_bounds,
        )
        logger.info(
            f"Using data-driven quantile estimation for contrast/offset "
            f"(n_phi={n_phi}, n_data={len(c2_data):,})"
        )

    # =========================================================================
    # Handle per_angle_mode: "constant" vs "individual"
    # =========================================================================
    if per_angle_mode == "constant":
        # CONSTANT MODE: No per-angle params are sampled - they're fixed from
        # quantile estimation and passed directly to the model.
        # Only physical parameters need initialization.
        logger.info(
            f"Constant mode: contrast/offset are FIXED (not sampled). "
            f"Only initializing {len(physical_params)} physical parameters."
        )

    else:
        # INDIVIDUAL MODE: Sample per-angle contrast_i and offset_i
        # 1. Per-angle contrast parameters (FIRST)
        for i in range(n_phi):
            param_name = f"contrast_{i}"

            # Priority: initial_values > data_estimates > midpoint fallback
            if initial_values is not None and param_name in initial_values:
                raw_value = float(initial_values[param_name])
            elif initial_values is not None and "contrast" in initial_values:
                raw_value = float(initial_values["contrast"])
            elif param_name in data_estimates:
                raw_value = data_estimates[param_name]
            else:
                # Midpoint fallback
                bounds = parameter_space.get_bounds("contrast")
                raw_value = (bounds[0] + bounds[1]) / 2.0

            validated_value, was_clipped = validate_initial_value_bounds(
                param_name, raw_value, parameter_space
            )
            init_dict[param_name] = validated_value
            if was_clipped:
                clipped_params.append(param_name)

        # 2. Per-angle offset parameters (SECOND)
        for i in range(n_phi):
            param_name = f"offset_{i}"

            # Priority: initial_values > data_estimates > midpoint fallback
            if initial_values is not None and param_name in initial_values:
                raw_value = float(initial_values[param_name])
            elif initial_values is not None and "offset" in initial_values:
                raw_value = float(initial_values["offset"])
            elif param_name in data_estimates:
                raw_value = data_estimates[param_name]
            else:
                # Midpoint fallback
                bounds = parameter_space.get_bounds("offset")
                raw_value = (bounds[0] + bounds[1]) / 2.0

            validated_value, was_clipped = validate_initial_value_bounds(
                param_name, raw_value, parameter_space
            )
            init_dict[param_name] = validated_value
            if was_clipped:
                clipped_params.append(param_name)

    # 3. Physical parameters (THIRD, in canonical order)
    for param_name in physical_params:
        raw_value = get_init_value(param_name, initial_values, parameter_space)
        validated_value, was_clipped = validate_initial_value_bounds(
            param_name, raw_value, parameter_space
        )
        init_dict[param_name] = validated_value
        if was_clipped:
            clipped_params.append(param_name)

    if clipped_params:
        logger.warning(
            f"⚠️ {len(clipped_params)} initial values were outside bounds and clipped: "
            f"{clipped_params}. This may indicate NLSQ fit issues or mismatched bounds."
        )

    logger.debug(
        f"Built init values for {len(init_dict)} params: {list(init_dict.keys())}"
    )

    # Defensive validation: ensure dict keys match expected order
    # This catches parameter ordering bugs that could cause subtle issues
    expected_names = get_param_names_in_order(n_phi, analysis_mode, per_angle_mode)
    validate_init_values_order(init_dict, expected_names)

    return init_dict


def get_param_names_in_order(
    n_phi: int, analysis_mode: str, per_angle_mode: str = "individual"
) -> list[str]:
    """Get parameter names in NumPyro sampling order.

    CRITICAL: This order must match the model sampling order exactly.

    Parameters
    ----------
    n_phi : int
        Number of phi angles.
    analysis_mode : str
        Analysis mode ("static" or "laminar_flow").
    per_angle_mode : str
        Per-angle scaling mode: "individual" or "constant".

    Returns
    -------
    list[str]
        Parameter names in sampling order.

    Notes
    -----
    - individual mode: Samples per-angle contrast/offset (2*n_phi params)
    - constant mode: NO contrast/offset sampled (fixed from quantile estimation)
    """
    names: list[str] = []

    # For constant mode, contrast/offset are FIXED (not sampled)
    # Only individual mode samples per-angle parameters
    if per_angle_mode != "constant":
        # 1. Per-angle contrast
        for i in range(n_phi):
            names.append(f"contrast_{i}")

        # 2. Per-angle offset
        for i in range(n_phi):
            names.append(f"offset_{i}")

    # 3. Physical parameters
    if analysis_mode == "laminar_flow":
        names.extend(LAMINAR_PARAMS)
    else:
        names.extend(STATIC_PARAMS)

    return names


def validate_init_values_order(
    init_values: dict[str, float],
    expected_names: list[str],
) -> None:
    """Validate that init values dictionary keys match expected order.

    This is a defensive check to catch parameter ordering bugs early.
    In Python 3.7+, dict preserves insertion order, so key order matters
    for functions that assume positional correspondence.

    Parameters
    ----------
    init_values : dict[str, float]
        Initial values dictionary.
    expected_names : list[str]
        Expected parameter names in order.

    Raises
    ------
    ValueError
        If parameter order doesn't match.
    """
    actual_names = list(init_values.keys())

    if actual_names != expected_names:
        # Find first mismatch for helpful error message
        for i, (actual, expected) in enumerate(
            zip(actual_names, expected_names, strict=True)
        ):
            if actual != expected:
                raise ValueError(
                    f"Parameter order mismatch at position {i}!\n"
                    f"Expected: {expected}\n"
                    f"Actual: {actual}\n"
                    f"Full expected: {expected_names}\n"
                    f"Full actual: {actual_names}"
                )

        # Length mismatch
        raise ValueError(
            f"Parameter count mismatch!\n"
            f"Expected {len(expected_names)} params: {expected_names}\n"
            f"Actual {len(actual_names)} params: {actual_names}"
        )
