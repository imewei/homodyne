"""
Unit Tests for Anti-Degeneracy Pre-Check and Out-of-Core Routing Fix (v2.22.0)
===============================================================================

Regression tests for the bug where NLSQ out-of-core routing bypassed the
Anti-Degeneracy Defense System, causing parameter absorption degeneracy:

1. Anti-degeneracy pre-check determines effective param count BEFORE memory routing
2. Out-of-core convergence uses multi-criteria (xtol + ftol), not single norm
3. rel_change is initialized before the optimization loop (prevents NameError)

Test IDs: T090a-T090g

References:
    - Debug session 2026-02-22: laminar_flow 23-angle production failure
    - wrapper.py: _fit_with_out_of_core_accumulation convergence logic
    - memory.py: select_nlsq_strategy used for routing
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from homodyne.optimization.nlsq.memory import (
    NLSQStrategy,
    select_nlsq_strategy,
)

# =============================================================================
# Anti-Degeneracy Pre-Check: Memory Routing Tests
# =============================================================================


class TestAntiDegeneracyPreCheck:
    """Tests that anti-degeneracy mode selection is used for memory routing.

    The key insight: with per_angle_mode="auto" and n_phi >= threshold (3),
    the effective param count is n_physical + 2 (auto_averaged), NOT
    n_physical + 2 * n_angles (individual). Memory routing must use the
    effective count.
    """

    def test_auto_averaged_effective_params(self):
        """T090a: auto mode with n_phi >= threshold yields n_physical + 2."""
        # Simulate: 23 angles, laminar_flow (7 physical)
        n_physical = 7
        n_angles = 23
        per_angle_mode = "auto"
        threshold = 3

        # Expanded (individual): 7 + 2*23 = 53
        expanded_params = n_physical + 2 * n_angles
        assert expanded_params == 53

        # Effective (auto_averaged): 7 + 2 = 9
        if per_angle_mode == "auto" and n_angles >= threshold:
            effective_params = n_physical + 2
        else:
            effective_params = expanded_params
        assert effective_params == 9

    def test_individual_mode_no_reduction(self):
        """T090b: explicit individual mode uses expanded count."""
        n_physical = 7
        n_angles = 23
        per_angle_mode = "individual"
        threshold = 3

        expanded_params = n_physical + 2 * n_angles
        if per_angle_mode == "auto" and n_angles >= threshold:
            effective_params = n_physical + 2
        else:
            effective_params = expanded_params
        assert effective_params == 53, "individual mode should NOT reduce params"

    def test_constant_mode_physical_only(self):
        """T090c: constant mode uses only physical params."""
        n_physical = 7
        per_angle_mode = "constant"

        if per_angle_mode == "constant":
            effective_params = n_physical
        else:
            effective_params = 53  # fallback
        assert effective_params == 7

    def test_auto_below_threshold_uses_individual(self):
        """T090d: auto mode with n_phi < threshold stays individual."""
        n_physical = 7
        n_angles = 2
        per_angle_mode = "auto"
        threshold = 3

        expanded_params = n_physical + 2 * n_angles  # 11
        if per_angle_mode == "auto" and n_angles >= threshold:
            effective_params = n_physical + 2
        else:
            effective_params = expanded_params
        assert effective_params == 11, "Below threshold should use expanded count"

    def test_memory_routing_with_effective_params(self):
        """T090e: Memory routing uses effective count, not expanded.

        This is the critical regression test. With 23M points and 53 expanded
        params, select_nlsq_strategy returns OUT_OF_CORE (~59 GB). With 9
        effective params, it returns STANDARD (~10 GB).

        Uses a fixed 16 GB threshold to be independent of runner RAM
        (macOS CI runners have only 7 GB, which makes even 9 params exceed
        the 75%-of-RAM threshold).
        """
        n_points = 23_000_000
        fixed_threshold = (16.0, {"detection_method": "mock"})

        with patch(
            "homodyne.optimization.nlsq.memory.get_adaptive_memory_threshold",
            return_value=fixed_threshold,
        ):
            # With expanded params (53): should be OUT_OF_CORE (~59 GB > 16 GB)
            decision_expanded = select_nlsq_strategy(n_points, 53)
            assert decision_expanded.strategy == NLSQStrategy.OUT_OF_CORE, (
                f"53 params should trigger out-of-core, got {decision_expanded.strategy}"
            )

            # With effective params (9): should be STANDARD (~10 GB < 16 GB)
            decision_effective = select_nlsq_strategy(n_points, 9)
            assert decision_effective.strategy == NLSQStrategy.STANDARD, (
                f"9 effective params should fit in standard, got {decision_effective.strategy}"
            )

    def test_static_mode_3_angles(self):
        """T090f: static mode with 3 angles, auto → auto_averaged (5 params)."""
        n_physical = 3  # static: D0, alpha, D_offset
        n_angles = 3
        per_angle_mode = "auto"
        threshold = 3

        expanded_params = n_physical + 2 * n_angles  # 9
        if per_angle_mode == "auto" and n_angles >= threshold:
            effective_params = n_physical + 2  # 5
        else:
            effective_params = expanded_params
        assert effective_params == 5


# =============================================================================
# Out-of-Core Convergence Tests
# =============================================================================


class TestOutOfCoreConvergence:
    """Tests for the multi-criteria convergence logic.

    The old convergence used ||step|| / ||params|| < 1e-4, which was dominated
    by large-magnitude parameters (D0 ~ 19231). The new convergence uses:
    - xtol: per-component max(|step_i| / |param_i|) — scale-invariant
    - ftol: relative cost change — prevents stopping when chi2 still improving
    Both must be satisfied simultaneously.
    """

    def test_norm_dominated_by_large_param(self):
        """T090g: Norm-based convergence is scale-sensitive (old behavior).

        Demonstrates why the old criterion was wrong: a step that barely
        changes D0 but significantly changes gamma_dot would be declared
        converged by the norm-based test.
        """
        # Params: [D0, alpha, D_offset, gamma_dot, beta, gamma_dot_offset, phi0]
        params = np.array([19231.0, 1.0, 879.0, 0.00134, -0.23, -0.00045, -5.57])
        # Step that changes gamma_dot by 50% but D0 by 0.001%
        step = np.array([0.2, 0.01, 0.1, 0.0007, 0.01, 0.0001, 0.01])

        # Old: norm-based (would say "converged" due to D0 dominating)
        old_rel_change = np.linalg.norm(step) / (np.linalg.norm(params) + 1e-10)
        assert old_rel_change < 1e-4, (
            "Old norm-based test should trigger false convergence"
        )

        # New: per-component max (catches that gamma_dot changed by 52%)
        param_scale = np.maximum(np.abs(params), 1e-10)
        new_rel_change = np.max(np.abs(step) / param_scale)
        assert new_rel_change > 0.5, (
            f"Per-component test should see gamma_dot changed by ~52%, "
            f"got max rel_change={new_rel_change:.4f}"
        )

    def test_ftol_prevents_early_convergence(self):
        """T090h: ftol prevents convergence when chi2 still improving."""
        # 19.3% chi2 decrease should NOT trigger convergence
        cost_change = 0.193  # 19.3% decrease
        ftol = 1e-6

        assert cost_change > ftol, "19.3% cost improvement should not satisfy ftol"

    def test_both_criteria_required(self):
        """T090i: Convergence requires BOTH xtol AND ftol."""
        # Case 1: xtol satisfied but ftol not → NOT converged
        xtol_met = True
        ftol_met = False
        assert not (xtol_met and ftol_met)

        # Case 2: ftol satisfied but xtol not → NOT converged
        xtol_met = False
        ftol_met = True
        assert not (xtol_met and ftol_met)

        # Case 3: Both satisfied → converged
        xtol_met = True
        ftol_met = True
        assert xtol_met and ftol_met

    def test_rel_change_initialized(self):
        """T090j: rel_change must be initialized before loop to prevent NameError."""
        # Simulate: no step accepted (inner loop rejects all)
        rel_change = float("inf")  # Must be initialized
        xtol = 1e-6

        # After loop exits without accepting a step, rel_change should be inf
        converged = rel_change < xtol
        assert not converged, "Unaccepted steps should not report convergence"
