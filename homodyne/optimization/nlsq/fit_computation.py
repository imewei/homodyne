"""Fit Computation Utilities for NLSQ Results.

This module provides functions for computing theoretical fits from NLSQ
optimization results. Extracted from cli/commands.py for better organization.

Extracted from cli/commands.py as part of refactoring (Dec 2025).
"""

from __future__ import annotations

import logging
from typing import Any

import jax.numpy as jnp
import numpy as np

from homodyne.core.jax_backend import compute_g2_scaled

logger = logging.getLogger(__name__)


def normalize_analysis_mode(
    mode: str | None,
    n_params: int,
    n_angles: int,
) -> str:
    """Resolve analysis mode, inferring from parameter counts if needed.

    Args:
        mode: Explicit mode or None
        n_params: Number of parameters
        n_angles: Number of angles

    Returns:
        Normalized mode: 'static' or 'laminar_flow'
    """
    if mode:
        mode_lower = mode.lower()
        if mode_lower in {"static", "static_isotropic"}:
            return "static"
        if mode_lower == "laminar_flow":
            return "laminar_flow"

    # Infer from parameter counts (legacy scalar vs per-angle layout)
    candidates = {
        "static": 3,
        "laminar_flow": 7,
    }
    for candidate_mode, n_phys in candidates.items():
        if n_params in {n_phys + 2, 2 * n_angles + n_phys}:
            return candidate_mode

    # Default to static for backward compatibility
    logger.debug(
        "Unable to infer analysis_mode from params=%s angles=%s; defaulting to static",
        n_params,
        n_angles,
    )
    return "static"


def get_physical_param_count(analysis_mode: str) -> int:
    """Get number of physical parameters for analysis mode.

    Args:
        analysis_mode: 'static' or 'laminar_flow'

    Returns:
        Number of physical parameters

    Raises:
        ValueError: If mode is unknown
    """
    if analysis_mode == "static":
        return 3  # D0, alpha, D_offset
    elif analysis_mode == "laminar_flow":
        return 7  # D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0
    else:
        raise ValueError(
            f"Unknown analysis_mode: '{analysis_mode}'. Expected 'static' or 'laminar_flow'"
        )


def extract_parameters_from_result(
    parameters: np.ndarray,
    n_angles: int,
    analysis_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Extract contrast, offset, and physical parameters from result.

    Handles both per-angle and scalar parameter layouts.

    Args:
        parameters: Full parameter array from optimization
        n_angles: Number of phi angles
        analysis_mode: 'static' or 'laminar_flow'

    Returns:
        Tuple of (contrasts, offsets, physical_params, scalar_expansion_used)

    Raises:
        ValueError: If parameter count doesn't match expected
    """
    n_params = len(parameters)
    n_physical = get_physical_param_count(analysis_mode)
    expected_per_angle = 2 * n_angles + n_physical

    scalar_expansion = False

    if n_params == expected_per_angle:
        # Per-angle layout: [contrast_0, ..., contrast_N, offset_0, ..., offset_N, physical...]
        contrasts = parameters[0:n_angles]
        offsets = parameters[n_angles : 2 * n_angles]
        physical_params = parameters[2 * n_angles :]
    elif n_params == (n_physical + 2):
        # Scalar layout: [contrast, offset, physical...]
        logger.warning(
            "Solver returned scalar contrast/offset (parameter count %d). Expanding "
            "scalars across %d filtered angles for result saving.",
            n_params,
            n_angles,
        )
        scalar_expansion = True
        scalar_contrast = float(parameters[0])
        scalar_offset = float(parameters[1])
        contrasts = np.full(n_angles, scalar_contrast, dtype=float)
        offsets = np.full(n_angles, scalar_offset, dtype=float)
        physical_params = parameters[2:]
    else:
        raise ValueError(
            f"Parameter count mismatch! Expected {expected_per_angle} "
            f"(2×{n_angles} scaling + {n_physical} physical), got {n_params}. "
            f"Per-angle scaling is REQUIRED in v2.4.0+"
        )

    return contrasts, offsets, physical_params, scalar_expansion


def compute_theoretical_fits(
    result: Any,
    data: dict[str, Any],
    metadata: dict[str, Any],
    *,
    analysis_mode: str | None = None,
    include_solver_surface: bool = True,
) -> dict[str, Any]:
    """Compute theoretical fits with per-angle least squares scaling.

    Generates theoretical correlation functions using optimized parameters,
    then applies per-angle scaling (contrast, offset) via least squares fitting
    to match experimental intensities.

    Args:
        result: NLSQ optimization result with physical parameters
        data: Experimental data with phi_angles_list, c2_exp, t1, t2
        metadata: Metadata with L, dt, q for theoretical computation
        analysis_mode: Optional analysis mode override
        include_solver_surface: Whether to include solver surface in output

    Returns:
        Dictionary with keys:
        - 'c2_theoretical_raw': Raw theoretical fits (n_angles, n_t1, n_t2)
        - 'c2_theoretical_scaled': Scaled fits (n_angles, n_t1, n_t2)
        - 'c2_solver_scaled': Solver surface (if requested)
        - 'per_angle_scaling': Post-hoc lstsq scaling params (n_angles, 2)
        - 'per_angle_scaling_solver': Original solver scaling params
        - 'residuals': Exp - scaled fit (n_angles, n_t1, n_t2)
        - 'scalar_per_angle_expansion': Whether scalar expansion was used

    Raises:
        ValueError: If q is missing or parameter count is invalid
    """
    phi_angles = np.asarray(data["phi_angles_list"])
    c2_exp = np.asarray(data["c2_exp"])
    t1 = np.asarray(data["t1"])
    t2 = np.asarray(data["t2"])

    # Convert 2D meshgrids to 1D if needed
    if t1.ndim == 2:
        t1 = t1[:, 0]
    if t2.ndim == 2:
        t2 = t2[0, :]

    n_params = len(result.parameters)
    n_angles = len(phi_angles)

    # Normalize analysis mode
    normalized_mode = normalize_analysis_mode(
        analysis_mode or getattr(result, "analysis_mode", None),
        n_params,
        n_angles,
    )

    # Extract parameters
    fitted_contrasts, fitted_offsets, physical_params, scalar_expansion = (
        extract_parameters_from_result(result.parameters, n_angles, normalized_mode)
    )

    logger.info(
        f"Per-angle scaling: {n_angles} angles, using FITTED scaling parameters from NLSQ optimization"
    )
    logger.debug(
        f"Extracted fitted parameters - "
        f"contrasts: mean={np.mean(fitted_contrasts):.4f}, "
        f"offsets: mean={np.mean(fitted_offsets):.4f}"
    )

    # Extract metadata
    L = metadata["L"]
    dt = metadata.get("dt", 0.1)
    q = metadata["q"]

    if q is None:
        raise ValueError("q (wavevector) is required but was not found")

    logger.info(
        f"Computing theoretical fits for {len(phi_angles)} angles using L={L:.1f} Å, q={q:.6f} Å⁻¹"
    )

    # Sequential per-angle computation
    c2_theoretical_raw_list = []
    c2_theoretical_fitted = []
    solver_surface = []
    per_angle_scaling_posthoc = []
    solver_scaling = np.column_stack((fitted_contrasts, fitted_offsets))

    for i, phi_angle in enumerate(phi_angles):
        # Convert to JAX arrays
        phi_jax = jnp.array([float(phi_angle)])
        t1_jax = jnp.array(t1)
        t2_jax = jnp.array(t2)
        params_jax = jnp.array(physical_params)

        # Compute RAW theory WITHOUT scaling
        c2_theory_raw = compute_g2_scaled(
            params=params_jax,
            t1=t1_jax,
            t2=t2_jax,
            phi=phi_jax,
            q=float(q),
            L=float(L),
            contrast=1.0,
            offset=1.0,
            dt=float(dt),
        )

        # Convert to NumPy and squeeze out phi dimension
        c2_theory_raw_np = np.asarray(c2_theory_raw)
        if c2_theory_raw_np.ndim == 3:
            c2_theory_raw_np = c2_theory_raw_np[0]

        c2_theoretical_raw_list.append(c2_theory_raw_np)

        if include_solver_surface:
            # Evaluate solver surface using original per-angle contrast/offset
            c2_solver = compute_g2_scaled(
                params=params_jax,
                t1=t1_jax,
                t2=t2_jax,
                phi=phi_jax,
                q=float(q),
                L=float(L),
                contrast=float(fitted_contrasts[i]),
                offset=float(fitted_offsets[i]),
                dt=float(dt),
            )
            c2_solver_np = np.asarray(c2_solver)
            if c2_solver_np.ndim == 3:
                c2_solver_np = c2_solver_np[0]
            solver_surface.append(c2_solver_np)

        # Post-hoc least-squares scaling for visualization
        theory_flat_jax = jnp.array(c2_theory_raw_np.flatten())
        exp_flat_jax = jnp.array(c2_exp[i].flatten())

        A_jax = jnp.column_stack([theory_flat_jax, jnp.ones_like(theory_flat_jax)])
        solution_jax, _, _, _ = jnp.linalg.lstsq(A_jax, exp_flat_jax, rcond=None)
        contrast_lstsq = float(solution_jax[0])
        offset_lstsq = float(solution_jax[1])

        c2_theoretical_scaled_angle = contrast_lstsq * c2_theory_raw_np + offset_lstsq
        c2_theoretical_fitted.append(c2_theoretical_scaled_angle)
        per_angle_scaling_posthoc.append([contrast_lstsq, offset_lstsq])

        logger.debug(
            f"Angle {phi_angle:.1f}°: lstsq contrast={contrast_lstsq:.4f}, offset={offset_lstsq:.4f}"
        )

    # Stack arrays
    c2_theoretical_raw = np.array(c2_theoretical_raw_list)
    c2_theoretical_fitted = np.array(c2_theoretical_fitted)
    c2_solver_surface = (
        np.array(solver_surface) if include_solver_surface and solver_surface else None
    )
    per_angle_scaling = np.array(per_angle_scaling_posthoc)

    residuals = c2_exp - c2_theoretical_fitted

    logger.info(
        f"Computed theoretical fits for {len(phi_angles)} angles"
    )

    return {
        "c2_theoretical_raw": c2_theoretical_raw,
        "c2_theoretical_scaled": c2_theoretical_fitted,
        "c2_solver_scaled": c2_solver_surface,
        "per_angle_scaling": per_angle_scaling,
        "per_angle_scaling_solver": solver_scaling,
        "residuals": residuals,
        "scalar_per_angle_expansion": scalar_expansion,
    }
