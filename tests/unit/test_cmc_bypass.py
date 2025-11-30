"""Unit tests for CMC bypass heuristics and diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from homodyne.optimization.cmc.bypass import evaluate_cmc_bypass
from homodyne.optimization.cmc.combination import combine_subposteriors


class TestCMCBypassHeuristics:
    def test_single_angle_dataset_triggers_bypass(self):
        decision = evaluate_cmc_bypass(
            {},
            num_samples=1,
            dataset_size=20_000,
            phi=[0.1] * 100,
        )

        assert decision.should_bypass
        assert decision.triggered_by == "single_angle"

    def test_force_cmc_override_disables_bypass(self):
        decision = evaluate_cmc_bypass(
            {"bypass_mode": "force_cmc"},
            num_samples=2,
            dataset_size=5_000,
            phi=None,
        )

        assert decision.should_bypass is False

    def test_large_dataset_keeps_cmc_enabled(self):
        phi = np.linspace(0.0, np.pi, 32)
        decision = evaluate_cmc_bypass(
            {},
            num_samples=len(phi),
            dataset_size=2_000_000,
            phi=phi,
        )

        assert decision.should_bypass is False


class TestCMCStrictDiagnostics:
    def test_strict_diagnostics_fail_fast_on_large_kl(self):
        rng = np.random.default_rng(42)
        shard_a = {"samples": rng.normal(loc=0.0, scale=0.05, size=(400, 1))}
        shard_b = {"samples": rng.normal(loc=3.0, scale=0.05, size=(400, 1))}

        with pytest.raises(ValueError):
            combine_subposteriors(
                [shard_a, shard_b],
                diagnostics_config={
                    "strict_mode": True,
                    "max_between_shard_kl": 0.5,
                },
            )

    def test_lenient_mode_logs_warning_but_combines(self):
        rng = np.random.default_rng(7)
        shard_a = {"samples": rng.normal(loc=0.0, scale=0.1, size=(300, 2))}
        shard_b = {"samples": rng.normal(loc=0.5, scale=0.1, size=(300, 2))}

        result = combine_subposteriors(
            [shard_a, shard_b],
            diagnostics_config={
                "strict_mode": False,
                "max_between_shard_kl": 1e-3,
            },
        )

        assert result["samples"].shape[1] == 2
        assert result["method"] in {"weighted", "average", "single_shard"}
