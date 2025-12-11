"""Smoke tests for CMC timeout mitigation settings.

These tests ensure the lighter defaults and shard sizing helpers do not regress
and that a tiny laminar-flow run completes quickly with reduced settings.
"""

from __future__ import annotations

import numpy as np
import pytest

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
    assert _resolve_max_points_per_shard("laminar_flow", 3_000_000, "auto") == 250_000


@pytest.mark.unit
def test_laminar_flow_smoke_run_completes_quickly():
    # Tiny synthetic dataset (single phi angle, element-wise paired times)
    t = np.linspace(0.1, 0.5, 12)
    data = 1.0 + 0.01 * np.random.randn(len(t))
    phi = np.zeros_like(t)

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
        t1=t,
        t2=t,
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
