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


def build_prior(
    param_name: str,
    parameter_space: ParameterSpace,
) -> dist.Distribution:
    """Build NumPyro prior distribution for a parameter.

    Parameters
    ----------
    param_name : str
        Parameter name (e.g., "D0", "alpha", "contrast").
    parameter_space : ParameterSpace
        Parameter space with bounds and priors.

    Returns
    -------
    dist.Distribution
        NumPyro distribution for sampling.
    """
    try:
        prior_spec = parameter_space.get_prior(param_name)
        return build_prior_from_spec(prior_spec)
    except (KeyError, AttributeError):
        # Fallback to uniform prior with bounds
        bounds = parameter_space.get_bounds(param_name)
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
    1. Value from initial_values dict if provided
    2. Midpoint of parameter bounds as fallback

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
    """
    # Check initial_values first
    if initial_values is not None and param_name in initial_values:
        return float(initial_values[param_name])

    # Fallback to midpoint of bounds
    bounds = parameter_space.get_bounds(param_name)
    midpoint = (bounds[0] + bounds[1]) / 2.0
    return midpoint


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
        Initial values from config.
    parameter_space : ParameterSpace
        Parameter space with bounds.

    Returns
    -------
    dict[str, float]
        Initial values dictionary in sampling order.
    """
    init_dict: dict[str, float] = {}

    # 1. Per-angle contrast parameters (FIRST)
    for i in range(n_phi):
        param_name = f"contrast_{i}"
        init_dict[param_name] = get_init_value(param_name, initial_values, parameter_space)
        # If per-angle not in initial_values, try base "contrast"
        if initial_values is not None and param_name not in initial_values:
            if "contrast" in initial_values:
                init_dict[param_name] = float(initial_values["contrast"])

    # 2. Per-angle offset parameters (SECOND)
    for i in range(n_phi):
        param_name = f"offset_{i}"
        init_dict[param_name] = get_init_value(param_name, initial_values, parameter_space)
        if initial_values is not None and param_name not in initial_values:
            if "offset" in initial_values:
                init_dict[param_name] = float(initial_values["offset"])

    # 3. Physical parameters (THIRD, in canonical order)
    physical_params = LAMINAR_PARAMS if analysis_mode == "laminar_flow" else STATIC_PARAMS
    for param_name in physical_params:
        init_dict[param_name] = get_init_value(param_name, initial_values, parameter_space)

    logger.debug(
        f"Built init values for {len(init_dict)} params: {list(init_dict.keys())}"
    )

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
    """Validate that init values are in correct order.

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
        raise ValueError(
            f"Parameter order mismatch!\n"
            f"Expected: {expected_names}\n"
            f"Actual: {actual_names}"
        )


def build_bounds_arrays(
    n_phi: int,
    analysis_mode: str,
    parameter_space: ParameterSpace,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build lower and upper bounds arrays in parameter order.

    Parameters
    ----------
    n_phi : int
        Number of phi angles.
    analysis_mode : str
        Analysis mode.
    parameter_space : ParameterSpace
        Parameter space with bounds.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        (lower_bounds, upper_bounds) arrays.
    """
    param_names = get_param_names_in_order(n_phi, analysis_mode)

    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    for name in param_names:
        # Handle per-angle parameters
        if name.startswith("contrast_"):
            bounds = parameter_space.get_bounds("contrast")
        elif name.startswith("offset_"):
            bounds = parameter_space.get_bounds("offset")
        else:
            bounds = parameter_space.get_bounds(name)

        lower_bounds.append(bounds[0])
        upper_bounds.append(bounds[1])

    return jnp.array(lower_bounds), jnp.array(upper_bounds)
