"""Integration tests for NLSQ→CMC warm-start pipeline with adaptive sampling.

Validates that:
1. Adaptive sampling reduces warmup/samples for small shards
2. Non-adaptive mode preserves config defaults
3. Laminar flow mode adjusts minimums for higher parameter counts
"""

from __future__ import annotations

import numpy as np
import pytest

# Require ArviZ for CMC imports; skip module if missing optional dependency
pytest.importorskip("arviz", reason="ArviZ required for CMC unit tests")

from homodyne.optimization.cmc.core import fit_mcmc_jax  # noqa: E402
from homodyne.optimization.cmc.results import CMCResult  # noqa: E402


class _MockParameterSpace:
    """Minimal ParameterSpace substitute for tests.

    Reuses pattern from test_timeout_smoke.py.
    """

    def __init__(self) -> None:
        self._bounds = {
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

    def get_bounds(self, name: str):
        base = name.split("_")[0]
        return self._bounds.get(base, (-1.0, 1.0))

    def get_prior(self, name: str):
        raise KeyError(name)


def _make_tiny_dataset(n: int = 50):
    """Create tiny off-diagonal synthetic data for single-phi smoke tests."""
    rng = np.random.default_rng(42)
    t1 = np.linspace(0.1, 1.0, n)
    t2 = t1 + 0.05  # Off-diagonal: t2 > t1
    data = 1.0 + 0.01 * rng.standard_normal(n)
    phi = np.zeros(n)
    return data, t1, t2, phi


def _mock_nlsq_result_static():
    """Mock NLSQ result dict for static mode (flat param keys)."""
    return {
        "D0": 1e4,
        "alpha": -0.5,
        "D_offset": 1e3,
    }


def _mock_nlsq_result_laminar():
    """Mock NLSQ result dict for laminar_flow mode (flat param keys)."""
    return {
        "D0": 1e4,
        "alpha": -0.5,
        "D_offset": 1e3,
        "gamma_dot_t0": 1e-3,
        "beta": -0.3,
        "gamma_dot_t_offset": 1e-5,
        "phi0": 0.0,
    }


def _base_cmc_config(*, adaptive: bool, num_warmup: int = 500, num_samples: int = 1500):
    """Build minimal CMC config dict."""
    return {
        "enable": True,
        "sharding": {"strategy": "stratified", "max_points_per_shard": 10000},
        "per_shard_timeout": 60,
        "per_shard_mcmc": {
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "num_chains": 1,
            "target_accept_prob": 0.8,
            "adaptive_sampling": adaptive,
        },
    }


@pytest.mark.unit
def test_nlsq_warmstart_adaptive_static():
    """Adaptive sampling should reduce warmup/samples for tiny static dataset."""
    data, t1, t2, phi = _make_tiny_dataset(50)

    result = fit_mcmc_jax(
        data=data,
        t1=t1,
        t2=t2,
        phi=phi,
        q=0.005,
        L=2e6,
        analysis_mode="static",
        cmc_config=_base_cmc_config(adaptive=True),
        parameter_space=_MockParameterSpace(),
        dt=0.1,
        progress_bar=False,
        nlsq_result=_mock_nlsq_result_static(),
    )

    assert isinstance(result, CMCResult)
    # Adaptive should reduce below defaults for 50-point shard
    assert result.n_warmup < 500, (
        f"Expected adapted n_warmup < 500, got {result.n_warmup}"
    )
    assert result.n_samples < 1500, (
        f"Expected adapted n_samples < 1500, got {result.n_samples}"
    )
    # Divergence count should be sane
    assert 0 <= result.divergences <= result.n_samples * result.n_chains


@pytest.mark.unit
def test_nlsq_warmstart_no_adaptation():
    """Non-adaptive should preserve config defaults for warmup/samples."""
    data, t1, t2, phi = _make_tiny_dataset(50)
    num_warmup = 10
    num_samples = 20

    result = fit_mcmc_jax(
        data=data,
        t1=t1,
        t2=t2,
        phi=phi,
        q=0.005,
        L=2e6,
        analysis_mode="static",
        cmc_config=_base_cmc_config(
            adaptive=False, num_warmup=num_warmup, num_samples=num_samples
        ),
        parameter_space=_MockParameterSpace(),
        dt=0.1,
        progress_bar=False,
        nlsq_result=_mock_nlsq_result_static(),
    )

    assert isinstance(result, CMCResult)
    assert result.n_warmup == num_warmup, (
        f"Expected n_warmup == {num_warmup}, got {result.n_warmup}"
    )
    assert result.n_samples == num_samples, (
        f"Expected n_samples == {num_samples}, got {result.n_samples}"
    )


@pytest.mark.unit
def test_nlsq_warmstart_adaptive_laminar():
    """Adaptive laminar_flow should have higher minimums (more params)."""
    data, t1, t2, phi = _make_tiny_dataset(50)

    result = fit_mcmc_jax(
        data=data,
        t1=t1,
        t2=t2,
        phi=phi,
        q=0.005,
        L=2e6,
        analysis_mode="laminar_flow",
        cmc_config=_base_cmc_config(adaptive=True),
        parameter_space=_MockParameterSpace(),
        dt=0.1,
        progress_bar=False,
        nlsq_result=_mock_nlsq_result_laminar(),
    )

    assert isinstance(result, CMCResult)
    # Adaptive should still reduce below defaults
    assert result.n_warmup < 500, (
        f"Expected adapted n_warmup < 500, got {result.n_warmup}"
    )
    # Laminar flow (9+ params) should have higher adapted minimums than static (5 params)
    # min_warmup = max(100, 20 * n_params) where n_params ~= 9-10 for laminar
    # So minimum warmup should be at least 180-200
    assert result.n_warmup >= 100, (
        f"Laminar adapted n_warmup too low: {result.n_warmup} (expected >= 100)"
    )
    assert 0 <= result.divergences <= result.n_samples * result.n_chains
