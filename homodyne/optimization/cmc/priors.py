"""Prior distribution builders for CMC analysis.

This module provides utilities for building NumPyro prior distributions
from the ParameterSpace configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpyro.distributions as dist

from homodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from homodyne.config.parameter_space import ParameterSpace, PriorDistribution

logger = get_logger(__name__)

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
) -> dict[str, float]:
    """Build complete initial values dictionary in sampling order.

    CRITICAL: Parameter order must match NumPyro model sampling order:
    1. contrast_0, contrast_1, ..., contrast_{n_phi-1}
    2. offset_0, offset_1, ..., offset_{n_phi-1}
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

    Returns
    -------
    dict[str, float]
        Initial values dictionary in sampling order.

    Notes
    -----
    Per-angle scaling parameters (contrast/offset):
        This function supports two modes for specifying per-angle initial values:

        1. **Scalar broadcast** (default): If initial_values contains only base names
           like 'contrast' and 'offset', those values are broadcast to ALL phi angles.
           Example: ``{'contrast': 0.5}`` → contrast_0=0.5, contrast_1=0.5, ...

        2. **Explicit per-angle**: If initial_values contains indexed names like
           'contrast_0', 'contrast_1', etc., those specific values are used.
           Example: ``{'contrast_0': 0.4, 'contrast_1': 0.6}``

        Mixed mode is supported: explicit per-angle values take precedence, with
        the base scalar used as fallback for unspecified angles.

    Bounds validation:
        All initial values are validated against parameter bounds. Out-of-bounds
        values are clipped to bounds ± 1% margin with a warning logged.
    """
    init_dict: dict[str, float] = {}
    clipped_params: list[str] = []

    # 1. Per-angle contrast parameters (FIRST)
    for i in range(n_phi):
        param_name = f"contrast_{i}"
        raw_value = get_init_value(param_name, initial_values, parameter_space)
        validated_value, was_clipped = validate_initial_value_bounds(
            param_name, raw_value, parameter_space
        )
        init_dict[param_name] = validated_value
        if was_clipped:
            clipped_params.append(param_name)

    # 2. Per-angle offset parameters (SECOND)
    for i in range(n_phi):
        param_name = f"offset_{i}"
        raw_value = get_init_value(param_name, initial_values, parameter_space)
        validated_value, was_clipped = validate_initial_value_bounds(
            param_name, raw_value, parameter_space
        )
        init_dict[param_name] = validated_value
        if was_clipped:
            clipped_params.append(param_name)

    # 3. Physical parameters (THIRD, in canonical order)
    physical_params = (
        LAMINAR_PARAMS if analysis_mode == "laminar_flow" else STATIC_PARAMS
    )
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
    expected_names = get_param_names_in_order(n_phi, analysis_mode)
    validate_init_values_order(init_dict, expected_names)

    return init_dict


def get_param_names_in_order(n_phi: int, analysis_mode: str) -> list[str]:
    """Get parameter names in NumPyro sampling order.

    CRITICAL: This order must match the model sampling order exactly.

    Parameters
    ----------
    n_phi : int
        Number of phi angles.
    analysis_mode : str
        Analysis mode ("static" or "laminar_flow").

    Returns
    -------
    list[str]
        Parameter names in sampling order.
    """
    names: list[str] = []

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


