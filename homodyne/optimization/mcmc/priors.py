"""Prior distribution sampling for NumPyro MCMC models.

This module extracts prior sampling logic from _create_numpyro_model
to reduce cyclomatic complexity and improve maintainability.

Extracted from mcmc.py as part of technical debt remediation (Dec 2025).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.distributions.transforms as dist_transforms

from homodyne.config.parameter_space import PriorDistribution
from homodyne.utils.logging import get_logger

logger = get_logger(__name__)

# Map distribution type names to NumPyro distribution classes
DIST_TYPE_MAP = {
    "Normal": dist.Normal,
    "TruncatedNormal": dist.TruncatedNormal,
    "Uniform": dist.Uniform,
    "LogNormal": dist.LogNormal,
    "BetaScaled": dist.Beta,
}


def get_prior_spec_with_fallback(
    param_name: str,
    parameter_space: Any,
) -> PriorDistribution:
    """Get prior specification with fallback defaults for scaling parameters.

    Parameters
    ----------
    param_name : str
        Parameter name
    parameter_space : ParameterSpace
        Parameter space with prior configurations

    Returns
    -------
    PriorDistribution
        Prior distribution specification

    Raises
    ------
    KeyError
        If parameter not found and no default available
    """
    try:
        return parameter_space.get_prior(param_name)
    except KeyError:
        # Scaling parameters might not be in config - use sensible defaults
        if param_name == "contrast":
            return PriorDistribution(
                dist_type="TruncatedNormal",
                mu=0.5,
                sigma=0.2,
                min_val=0.0,
                max_val=1.0,
            )
        elif param_name == "offset":
            return PriorDistribution(
                dist_type="TruncatedNormal",
                mu=1.0,
                sigma=0.2,
                min_val=0.5,
                max_val=1.5,
            )
        else:
            raise KeyError(
                f"Parameter '{param_name}' not found in parameter_space "
                f"and no default available. Available parameters: "
                f"{list(parameter_space.priors.keys())}"
            )


def auto_convert_to_bounded_distribution(
    prior_spec: PriorDistribution,
    param_name: str,
    debug: bool = False,
) -> type:
    """Auto-convert unbounded distributions to bounded versions.

    If config specifies Normal/LogNormal with bounds, convert to Truncated version.
    This prevents NumPyro from sampling extreme values that cause physics NaN/inf.

    Parameters
    ----------
    prior_spec : PriorDistribution
        Prior distribution specification
    param_name : str
        Parameter name for logging
    debug : bool, default=False
        Enable debug logging

    Returns
    -------
    type
        NumPyro distribution class
    """
    dist_class = DIST_TYPE_MAP.get(prior_spec.dist_type, dist.TruncatedNormal)

    if hasattr(prior_spec, "min_val") and hasattr(prior_spec, "max_val"):
        if prior_spec.min_val is not None and prior_spec.max_val is not None:
            if prior_spec.dist_type == "Normal":
                dist_class = dist.TruncatedNormal
                if debug:
                    logger.debug(
                        f"Auto-converted {param_name} from Normal to TruncatedNormal "
                        f"with bounds [{prior_spec.min_val}, {prior_spec.max_val}]"
                    )
            elif prior_spec.dist_type == "LogNormal":
                dist_class = dist.TruncatedNormal
                if debug:
                    logger.debug(
                        f"Converted {param_name} from LogNormal to TruncatedNormal "
                        f"with bounds [{prior_spec.min_val}, {prior_spec.max_val}]"
                    )

    return dist_class


def cast_dist_kwargs(
    dist_kwargs: dict[str, Any],
    target_dtype: Any,
) -> dict[str, Any]:
    """Cast distribution kwargs to target dtype.

    Parameters
    ----------
    dist_kwargs : dict
        Distribution keyword arguments
    target_dtype : dtype
        Target JAX dtype

    Returns
    -------
    dict
        Cast keyword arguments
    """
    def _cast_value(value):
        if isinstance(value, (int, float, np.number)):
            return jnp.asarray(value, dtype=target_dtype)
        if isinstance(value, np.ndarray):
            return jnp.asarray(value, dtype=target_dtype)
        return value

    result = {key: _cast_value(val) for key, val in dist_kwargs.items()}

    # Ensure bounds stay as JAX arrays
    if "low" in result and result["low"] is not None:
        result["low"] = jnp.asarray(result["low"], dtype=target_dtype)
    if "high" in result and result["high"] is not None:
        result["high"] = jnp.asarray(result["high"], dtype=target_dtype)

    return result


def prepare_truncated_normal_kwargs(
    dist_kwargs: dict[str, Any],
    target_dtype: Any,
) -> dict[str, Any]:
    """Prepare kwargs for TruncatedNormal distribution.

    Handles broadcasting and ensures valid bounds.

    Parameters
    ----------
    dist_kwargs : dict
        Distribution keyword arguments
    target_dtype : dtype
        Target JAX dtype

    Returns
    -------
    dict
        Prepared keyword arguments
    """
    truncated_keys = ("loc", "scale", "low", "high")

    def _ensure_tensor(val_name: str) -> jnp.ndarray | None:
        value = dist_kwargs.get(val_name)
        if value is None:
            return None
        return jnp.asarray(value, dtype=target_dtype)

    truncated_tensors = {
        key: _ensure_tensor(key)
        for key in truncated_keys
        if dist_kwargs.get(key) is not None
    }

    # Compute broadcast shape
    shapes = [tuple(tensor.shape) for tensor in truncated_tensors.values()]
    broadcast_shape = ()
    for shape in shapes:
        broadcast_shape = np.broadcast_shapes(broadcast_shape, shape)

    def _broadcast(value: jnp.ndarray) -> jnp.ndarray:
        if value is None:
            return value
        if broadcast_shape == ():
            return jnp.asarray(value, dtype=target_dtype).reshape(())
        if value.shape == broadcast_shape:
            return value
        return jnp.broadcast_to(value, broadcast_shape)

    # Ensure valid scale and bounds
    tiny = jnp.asarray(jnp.finfo(target_dtype).tiny, dtype=target_dtype)
    if "scale" in truncated_tensors:
        truncated_tensors["scale"] = jnp.maximum(
            jnp.abs(truncated_tensors["scale"]), tiny
        )
    if "low" in truncated_tensors and "high" in truncated_tensors:
        truncated_tensors["high"] = jnp.maximum(
            truncated_tensors["high"],
            truncated_tensors["low"] + tiny,
        )

    # Apply broadcasting
    result = dict(dist_kwargs)
    for key, tensor in truncated_tensors.items():
        if tensor is not None:
            result[key] = _broadcast(tensor)

    return result


def create_beta_scaled_distribution(
    dist_kwargs: dict[str, Any],
    target_dtype: Any,
) -> dist.TransformedDistribution:
    """Create a BetaScaled distribution.

    Parameters
    ----------
    dist_kwargs : dict
        Distribution keyword arguments (with low, high, concentration1, concentration0)
    target_dtype : dtype
        Target JAX dtype

    Returns
    -------
    dist.TransformedDistribution
        Beta distribution with affine transform
    """
    kwargs = dict(dist_kwargs)
    low = jnp.asarray(kwargs.pop("low"), dtype=target_dtype)
    high = jnp.asarray(kwargs.pop("high"), dtype=target_dtype)
    scale = jnp.maximum(
        high - low,
        jnp.asarray(jnp.finfo(target_dtype).tiny, dtype=target_dtype),
    )
    base_dist = dist.Beta(**kwargs)
    transform = dist_transforms.AffineTransform(loc=low, scale=scale)
    return dist.TransformedDistribution(base_dist, transform)


def sample_parameter(
    param_name: str,
    prior_spec: PriorDistribution,
    target_dtype: Any,
    per_angle_scaling: bool = False,
    n_phi: int = 1,
    debug: bool = False,
) -> jnp.ndarray:
    """Sample a parameter from its prior distribution.

    Parameters
    ----------
    param_name : str
        Parameter name (used as sample site name)
    prior_spec : PriorDistribution
        Prior distribution specification
    target_dtype : dtype
        Target JAX dtype
    per_angle_scaling : bool, default=False
        If True and param is contrast/offset, sample per-angle
    n_phi : int, default=1
        Number of phi angles for per-angle scaling
    debug : bool, default=False
        Enable debug logging

    Returns
    -------
    jnp.ndarray
        Sampled parameter value(s)
    """
    dist_class = auto_convert_to_bounded_distribution(prior_spec, param_name, debug)
    dist_kwargs = cast_dist_kwargs(prior_spec.to_numpyro_kwargs(), target_dtype)

    # Handle BetaScaled specially
    if prior_spec.dist_type == "BetaScaled":
        dist_instance = create_beta_scaled_distribution(dist_kwargs, target_dtype)

        if per_angle_scaling and param_name in ["contrast", "offset"]:
            param_values = [
                jnp.asarray(
                    numpyro.sample(f"{param_name}_{phi_idx}", dist_instance),
                    dtype=target_dtype,
                )
                for phi_idx in range(n_phi)
            ]
            return jnp.stack(param_values, axis=0)
        else:
            return jnp.asarray(
                numpyro.sample(param_name, dist_instance),
                dtype=target_dtype,
            )

    # Prepare TruncatedNormal kwargs if needed
    if dist_class is dist.TruncatedNormal:
        dist_kwargs = prepare_truncated_normal_kwargs(dist_kwargs, target_dtype)

    # Sample with or without per-angle scaling
    if per_angle_scaling and param_name in ["contrast", "offset"]:
        param_values = []
        for phi_idx in range(n_phi):
            param_name_phi = f"{param_name}_{phi_idx}"
            param_value_phi = numpyro.sample(param_name_phi, dist_class(**dist_kwargs))
            param_values.append(jnp.asarray(param_value_phi, dtype=target_dtype))
        return jnp.array(param_values, dtype=target_dtype)
    else:
        param_value = numpyro.sample(param_name, dist_class(**dist_kwargs))
        return jnp.asarray(param_value, dtype=target_dtype)


def sample_scaling_parameters(
    parameter_space: Any,
    n_phi: int,
    per_angle_scaling: bool,
    fixed_overrides: dict[str, float] | None,
    target_dtype: Any,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample contrast and offset parameters.

    Parameters
    ----------
    parameter_space : ParameterSpace
        Parameter space with prior configurations
    n_phi : int
        Number of phi angles
    per_angle_scaling : bool
        If True, sample per-angle parameters
    fixed_overrides : dict or None
        Fixed values for parameters (skip sampling)
    target_dtype : dtype
        Target JAX dtype

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        (contrast, offset) - either scalars or arrays of shape (n_phi,)
    """
    overrides = fixed_overrides or {}

    # Sample or fix contrast
    if "contrast" in overrides and not per_angle_scaling:
        contrast = jnp.asarray(overrides["contrast"], dtype=target_dtype)
        numpyro.deterministic("contrast", contrast)
    else:
        contrast_prior = get_prior_spec_with_fallback("contrast", parameter_space)
        contrast = sample_parameter(
            "contrast",
            contrast_prior,
            target_dtype,
            per_angle_scaling=per_angle_scaling,
            n_phi=n_phi,
        )

    # Sample or fix offset
    if "offset" in overrides and not per_angle_scaling:
        offset = jnp.asarray(overrides["offset"], dtype=target_dtype)
        numpyro.deterministic("offset", offset)
    else:
        offset_prior = get_prior_spec_with_fallback("offset", parameter_space)
        offset = sample_parameter(
            "offset",
            offset_prior,
            target_dtype,
            per_angle_scaling=per_angle_scaling,
            n_phi=n_phi,
        )

    return contrast, offset
