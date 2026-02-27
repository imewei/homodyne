"""Diagnostic helpers for visual validation of fitted C₂ surfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DiagonalOverlayResult:
    phi_index: int
    raw_diagonal: np.ndarray
    solver_diagonal: np.ndarray
    posthoc_diagonal: np.ndarray
    raw_variance: float
    solver_variance: float
    posthoc_variance: float
    solver_rmse: float
    posthoc_rmse: float


def compute_diagonal_overlay_stats(
    c2_exp: np.ndarray,
    c2_solver: np.ndarray,
    c2_posthoc: np.ndarray,
    *,
    phi_index: int = 0,
) -> DiagonalOverlayResult:
    """Return diagonal traces and simple statistics for a single angle."""

    if c2_solver is None:
        raise ValueError("c2_solver array is required for diagonal overlay diagnostics")

    def _diag(matrix: np.ndarray) -> np.ndarray:
        return np.diag(matrix[phi_index])

    raw_diag = _diag(c2_exp)
    solver_diag = _diag(c2_solver)
    posthoc_diag = _diag(c2_posthoc)

    raw_var = float(np.nanvar(raw_diag))
    solver_var = float(np.nanvar(solver_diag))
    posthoc_var = float(np.nanvar(posthoc_diag))

    solver_rmse = float(np.sqrt(np.nanmean((solver_diag - raw_diag) ** 2)))
    posthoc_rmse = float(np.sqrt(np.nanmean((posthoc_diag - raw_diag) ** 2)))

    return DiagonalOverlayResult(
        phi_index=phi_index,
        raw_diagonal=raw_diag,
        solver_diagonal=solver_diag,
        posthoc_diagonal=posthoc_diag,
        raw_variance=raw_var,
        solver_variance=solver_var,
        posthoc_variance=posthoc_var,
        solver_rmse=solver_rmse,
        posthoc_rmse=posthoc_rmse,
    )
