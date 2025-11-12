"""Unit tests for diagonal overlay diagnostics."""

import numpy as np

from homodyne.viz.diagnostics import compute_diagonal_overlay_stats


def test_solver_overlay_has_lower_rmse_than_posthoc() -> None:
    raw_matrix = np.array(
        [
            [1.0, 1.03, 1.01],
            [1.03, 1.08, 1.04],
            [1.01, 1.04, 1.0],
        ],
    )
    raw = raw_matrix[None, ...]
    solver = (raw * 0.999) + 0.001  # Slight bias but keeps oscillation
    posthoc = np.ones_like(raw) * np.mean(raw)

    overlay = compute_diagonal_overlay_stats(raw, solver, posthoc, phi_index=0)

    assert overlay.solver_rmse < overlay.posthoc_rmse
    assert overlay.raw_diagonal.size == raw.shape[-1]
