"""
Gradient Diagnostics for Parameter Scaling Optimization
========================================================

This module provides tools to diagnose gradient imbalance issues and compute
optimal parameter scaling factors (x_scale) for NLSQ optimization.

The Problem:
------------
Shear parameters (gamma_dot_t0, beta, gamma_dot_t_offset) can have gradients
100-10,000× larger than diffusion parameters (D0, alpha, D_offset), causing:

- Premature convergence
- Missing fine-scale features (oscillations)
- Poor fit quality despite low chi-squared


The Solution:
-------------
Compute parameter-specific x_scale values inversely proportional to gradient
magnitudes to normalize optimization steps across all parameters.

Usage:
------
.. code-block:: python

    from homodyne.optimization.gradient_diagnostics import compute_optimal_x_scale

    # Compute from fitted parameters
    x_scale_map = compute_optimal_x_scale(
        parameters=result.parameters,
        data=data,
        config=config,
        analysis_mode="laminar_flow"
    )

    # Add to config for next optimization
    config.config["optimization"]["nlsq"]["x_scale_map"] = x_scale_map

Author: Homodyne Development Team
Date: 2025-11-13
Version: 1.0.0
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


def _create_residual_function(
    data: Any,
    config: Any,
    analysis_mode: str,
) -> tuple[Callable, list[str]]:
    """
    Create residual function for gradient computation.

    Args:
        data: Data object with phi, t1, t2, g2, q, L, dt attributes
        config: Configuration object
        analysis_mode: "static_isotropic" or "laminar_flow"

    Returns:
        (residual_fn, param_names): Residual function and parameter names
    """
    from homodyne.core.jax_backend import compute_g1_total

    # Extract data
    phi = jnp.asarray(data.phi)
    t1 = jnp.asarray(data.t1)
    t2 = jnp.asarray(data.t2)
    g2_exp = jnp.asarray(data.g2)
    q = float(data.q)
    L = float(data.L)
    # Ensure dt is a float, defaulting to 1.0 if missing (consistent with other modules)
    dt = float(data.dt) if hasattr(data, "dt") and data.dt is not None else 1.0

    # Get per-angle scaling parameters
    if hasattr(data, "per_angle_scaling_solver"):
        per_angle = np.asarray(data.per_angle_scaling_solver)
        contrasts = jnp.asarray(per_angle[:, 0])
        offsets = jnp.asarray(per_angle[:, 1])
    else:
        # Fallback to defaults
        n_phi = len(np.unique(phi))
        contrasts = jnp.ones(n_phi) * 0.5
        offsets = jnp.ones(n_phi) * 1.0

    # Determine parameter names
    if "static" in analysis_mode.lower():
        param_names = ["D0", "alpha", "D_offset"]
    else:
        param_names = [
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ]

    @jax.jit
    def residual_fn(params: jnp.ndarray) -> jnp.ndarray:
        """Compute residuals for gradient calculation."""
        # Compute g1 using physics model
        g1 = compute_g1_total(params, t1, t2, phi, q, L, dt)

        # Apply per-angle scaling
        g2_theory = offsets[:, None, None] + contrasts[:, None, None] * jnp.square(g1)

        # Compute residuals
        residuals = (g2_theory - g2_exp).reshape(-1)
        return residuals

    return residual_fn, param_names


def compute_gradient_norms(
    parameters: dict[str, float],
    data: Any,
    config: Any,
    analysis_mode: str,
) -> dict[str, float]:
    """
    Compute gradient L2 norms for each parameter at given point.

    Args:
        parameters: Dictionary of parameter values
        data: Data object with experimental data
        config: Configuration object
        analysis_mode: "static_isotropic" or "laminar_flow"

    Returns:
        Dictionary mapping parameter names to gradient norms

    Example:
        >>> gradient_norms = compute_gradient_norms(
        ...     parameters=result.parameters,
        ...     data=data,
        ...     config=config,
        ...     analysis_mode="laminar_flow"
        ... )
        >>> # Output: {'D0': 26.98, 'alpha': 42365.33, ..., 'gamma_dot_t_offset': 346934800.0}
    """
    # Create residual function
    residual_fn, param_names = _create_residual_function(data, config, analysis_mode)

    # Convert parameters dict to array
    param_array = jnp.array([float(parameters[name]) for name in param_names])

    # Compute SSE function
    def sse_fn(params: jnp.ndarray) -> float:
        residuals = residual_fn(params)
        return float(jnp.sum(residuals**2))

    # Compute gradient
    grad_fn = jax.grad(sse_fn)
    gradients = grad_fn(param_array)

    # Create gradient norms dictionary
    gradient_norms = {
        name: float(abs(grad))
        for name, grad in zip(param_names, gradients, strict=False)
    }

    return gradient_norms


def compute_optimal_x_scale(
    parameters: dict[str, float],
    data: Any,
    config: Any,
    analysis_mode: str,
    baseline_params: list[str] | None = None,
    safety_factor: float = 1.0,
    min_scale: float = 1e-8,
    max_scale: float = 1e2,
) -> dict[str, float]:
    """
    Compute optimal x_scale map based on gradient norms.

    The x_scale values are inversely proportional to gradient magnitudes,
    normalized so that baseline parameters have x_scale=1.0.

    Args:
        parameters: Dictionary of parameter values
        data: Data object with experimental data
        config: Configuration object
        analysis_mode: "static_isotropic" or "laminar_flow"
        baseline_params: Parameters to use as baseline (x_scale=1.0).
                        Default: ["D0", "D_offset", "phi0"]
        safety_factor: Multiplicative safety factor (default: 1.0)
                      Increase to make optimization more conservative
        min_scale: Minimum allowed x_scale value (prevents division by zero)
        max_scale: Maximum allowed x_scale value (prevents extreme values)

    Returns:
        Dictionary mapping parameter names to x_scale values

    Example:
        >>> x_scale_map = compute_optimal_x_scale(
        ...     parameters={'D0': 400.0, 'alpha': -0.014, ..., 'gamma_dot_t_offset': 0.0},
        ...     data=data,
        ...     config=config,
        ...     analysis_mode="laminar_flow"
        ... )
        >>> # Output: {'D0': 1.0, 'alpha': 0.001, ..., 'gamma_dot_t_offset': 1e-7}
    """
    # Default baseline parameters
    if baseline_params is None:
        if "static" in analysis_mode.lower():
            baseline_params = ["D0", "D_offset"]
        else:
            baseline_params = ["D0", "D_offset", "phi0"]

    # Compute gradient norms
    gradient_norms = compute_gradient_norms(parameters, data, config, analysis_mode)

    # Compute baseline gradient (geometric mean of baseline parameters)
    baseline_grads = [
        gradient_norms[name] for name in baseline_params if name in gradient_norms
    ]
    if not baseline_grads:
        logger.warning(
            f"No baseline parameters found in gradient norms: {baseline_params}"
        )
        baseline_grads = [1.0]

    baseline_grad = np.exp(np.mean(np.log(np.maximum(baseline_grads, 1e-10))))

    # Compute x_scale values (inversely proportional to gradient)
    x_scale_map = {}
    for name, grad_norm in gradient_norms.items():
        # x_scale = baseline_grad / grad_norm * safety_factor
        raw_scale = baseline_grad / max(grad_norm, 1e-10) * safety_factor

        # Clip to reasonable range
        clipped_scale = np.clip(raw_scale, min_scale, max_scale)

        x_scale_map[name] = float(clipped_scale)

        # Log recommendations
        ratio = grad_norm / baseline_grad
        if ratio > 10:
            logger.info(
                f"Parameter {name:18s}: gradient {ratio:>8.0f}× baseline "
                f"→ x_scale={clipped_scale:.2e}"
            )
        elif ratio < 0.1:
            logger.info(
                f"Parameter {name:18s}: gradient {ratio:>8.2f}× baseline "
                f"→ x_scale={clipped_scale:.2e}"
            )
        else:
            logger.debug(
                f"Parameter {name:18s}: gradient {ratio:>8.2f}× baseline "
                f"→ x_scale={clipped_scale:.2e}"
            )

    return x_scale_map


def diagnose_gradient_imbalance(
    parameters: dict[str, float],
    data: Any,
    config: Any,
    analysis_mode: str,
    threshold: float = 10.0,
) -> dict[str, Any]:
    """
    Diagnose gradient imbalance and provide recommendations.

    Args:
        parameters: Dictionary of parameter values
        data: Data object with experimental data
        config: Configuration object
        analysis_mode: "static_isotropic" or "laminar_flow"
        threshold: Gradient ratio threshold for warning (default: 10.0)

    Returns:
        Dictionary with:
            - gradient_norms: Dict[str, float] - gradient norms for each parameter
            - imbalance_detected: bool - whether imbalance exceeds threshold
            - max_ratio: float - maximum gradient ratio
            - recommendations: Dict[str, Any] - optimization recommendations

    Example:
        >>> diag = diagnose_gradient_imbalance(
        ...     parameters=result.parameters,
        ...     data=data,
        ...     config=config,
        ...     analysis_mode="laminar_flow"
        ... )
        >>> if diag["imbalance_detected"]:
        ...     print(f"Gradient imbalance detected: max ratio = {diag['max_ratio']:.0f}×")
        ...     print("Recommendations:")
        ...     print(diag["recommendations"]["summary"])
    """
    # Compute gradient norms
    gradient_norms = compute_gradient_norms(parameters, data, config, analysis_mode)

    # Compute max/min ratio
    max_grad = max(gradient_norms.values())
    min_grad = min(gradient_norms.values())
    max_ratio = max_grad / max(min_grad, 1e-10)

    # Check for imbalance
    imbalance_detected = max_ratio > threshold

    # Generate recommendations
    recommendations: dict[str, Any] = {}
    if imbalance_detected:
        # Compute optimal x_scale
        x_scale_map = compute_optimal_x_scale(parameters, data, config, analysis_mode)

        recommendations["x_scale_map"] = x_scale_map
        summary = (
            f"Gradient imbalance detected: {max_ratio:.0f}× ratio\n"
            f"Apply parameter-specific scaling by adding to config:\n"
            f"optimization:\n"
            f"  nlsq:\n"
            f"    x_scale_map:\n"
        )
        for name, scale in x_scale_map.items():
            summary += f"      {name}: {scale:.2e}\n"
        recommendations["summary"] = summary

        recommendations["action"] = "apply_x_scale_map"
    else:
        recommendations["summary"] = (
            f"No significant gradient imbalance detected ({max_ratio:.1f}× ratio)"
        )
        recommendations["action"] = "no_action_needed"

    return {
        "gradient_norms": gradient_norms,
        "imbalance_detected": imbalance_detected,
        "max_ratio": max_ratio,
        "max_grad": max_grad,
        "min_grad": min_grad,
        "recommendations": recommendations,
    }


def print_gradient_report(
    parameters: dict[str, float],
    data: Any,
    config: Any,
    analysis_mode: str,
) -> None:
    """
    Print comprehensive gradient diagnostic report.

    Args:
        parameters: Dictionary of parameter values
        data: Data object with experimental data
        config: Configuration object
        analysis_mode: "static_isotropic" or "laminar_flow"

    Example:
        >>> # After NLSQ optimization
        >>> print_gradient_report(
        ...     parameters=result.parameters,
        ...     data=data,
        ...     config=config,
        ...     analysis_mode="laminar_flow"
        ... )
        # Prints detailed gradient analysis and recommendations
    """
    diag = diagnose_gradient_imbalance(parameters, data, config, analysis_mode)

    print("\n" + "=" * 80)
    print("GRADIENT DIAGNOSTIC REPORT")
    print("=" * 80)

    # Print gradient norms
    print("\nGradient Norms (SSE):")
    print("-" * 80)
    baseline_grad = np.median(list(diag["gradient_norms"].values()))
    for name, grad in diag["gradient_norms"].items():
        ratio = grad / baseline_grad
        bar_length = int(min(50, np.log10(max(ratio, 0.1)) * 10 + 25))
        bar = "█" * max(0, bar_length)
        print(f"{name:18s}: {grad:>12.2e}  {ratio:>8.1f}× median  {bar}")

    # Print diagnosis
    print("\n" + "-" * 80)
    if diag["imbalance_detected"]:
        print("⚠ GRADIENT IMBALANCE DETECTED")
        print(f"Maximum ratio: {diag['max_ratio']:.0f}×")
        print("\nThis can cause:")
        print("  - Premature convergence")
        print("  - Missing fine-scale features (oscillations)")
        print("  - Poor fit quality despite low chi-squared")
    else:
        print("✓ No significant gradient imbalance")

    # Print recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print(diag["recommendations"]["summary"])

    if diag["imbalance_detected"]:
        print("\nTo apply these fixes:")
        print("  1. Add x_scale_map to your configuration file")
        print("  2. Re-run optimization with updated config")
        print("  3. Verify improved convergence and fit quality")

    print("=" * 80 + "\n")
