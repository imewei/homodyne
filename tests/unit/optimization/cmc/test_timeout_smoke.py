"""Smoke tests for CMC timeout mitigation settings.

These tests ensure the lighter defaults and shard sizing helpers do not regress
and that a tiny laminar-flow run completes quickly with reduced settings.
"""

from __future__ import annotations

import numpy as np
import pytest

# Require ArviZ for CMC imports; skip module if missing optional dependency
pytest.importorskip("arviz", reason="ArviZ required for CMC unit tests")

from homodyne.optimization.cmc import core  # noqa: E402
from homodyne.optimization.cmc.core import (
    _resolve_max_points_per_shard,
    fit_mcmc_jax,
)
from homodyne.optimization.cmc.results import CMCResult


class _MockParameterSpace:
    """Minimal ParameterSpace substitute for tests."""

    def __init__(self) -> None:
        bounds = {
            "contrast": (0.0, 1.0),
            "offset": (0.5, 1.5),
            "D0": (1e3, 1e5),
            "alpha": (-3.0, 1.0),
            "D_offset": (0.0, 5e3),
            "gamma_dot_t0": (-1.0, 1.0),
            "beta": (-1.0, 1.0),
            "gamma_dot_t_offset": (-1.0, 1.0),
            "phi0": (-180.0, 180.0),
        }
        self._bounds = bounds

    def get_bounds(self, name: str):
        base = name.split("_")[0]
        return self._bounds.get(base, (-1.0, 1.0))

    def get_prior(self, name: str):
        # No custom priors for this smoke test
        raise KeyError(name)


@pytest.mark.unit
def test_resolve_max_points_per_shard_laminar_large_pool():
    # v2.19.0: Angle-aware scaling - with n_phi=1 (default), base 10K is scaled to 3K
    # For n_phi=10+, it would return 10K (angle_factor=1.0)
    assert _resolve_max_points_per_shard("laminar_flow", 3_000_000, "auto") == 3_000
    # Multi-angle dataset should scale differently
    assert _resolve_max_points_per_shard("laminar_flow", 3_000_000, "auto", n_phi=10) == 7_000


@pytest.mark.unit
def test_suggested_timeout_clamps_and_scales():
    # cost: chains=2, warmup+samples=2000, max_per_shard=20k → cost=80M
    cost = 2 * (500 + 1500) * 20_000
    # raw = 5 * 2e-5 * 80,000,000 = 80,000s -> clamped to max_timeout (7200)
    suggested = core._compute_suggested_timeout(cost_per_shard=cost, max_timeout=7200)
    assert suggested == 7200

    # Smaller cost should respect min clamp 600s
    small_cost = 2 * (5 + 10) * 100
    suggested_small = core._compute_suggested_timeout(
        cost_per_shard=small_cost, max_timeout=7200
    )
    assert suggested_small == 600


@pytest.mark.unit
def test_laminar_flow_smoke_run_completes_quickly():
    # Tiny synthetic dataset (single phi angle, off-diagonal time pairs)
    # Note: t1 != t2 required since CMC filters diagonal points (t1==t2)
    t1 = np.linspace(0.1, 0.5, 12)
    t2 = t1 + 0.05  # Off-diagonal: t2 > t1
    data = 1.0 + 0.01 * np.random.randn(len(t1))
    phi = np.zeros_like(t1)

    cmc_cfg = {
        "enable": True,
        "sharding": {"strategy": "stratified", "max_points_per_shard": 1000},
        "per_shard_timeout": 30,
        "per_shard_mcmc": {
            "num_warmup": 5,
            "num_samples": 10,
            "num_chains": 1,
            "target_accept_prob": 0.8,
        },
    }

    result = fit_mcmc_jax(
        data=data,
        t1=t1,
        t2=t2,
        phi=phi,
        q=0.005,
        L=2e6,
        analysis_mode="laminar_flow",
        cmc_config=cmc_cfg,
        parameter_space=_MockParameterSpace(),
        dt=0.1,
        progress_bar=False,
    )

    assert isinstance(result, CMCResult)
    assert result.samples is not None
