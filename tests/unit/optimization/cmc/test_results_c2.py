"""Tests for CMC c2_fitted computation wiring."""

from __future__ import annotations

import numpy as np
import pytest

# Require ArviZ for CMC imports; skip module if missing optional dependency
pytest.importorskip("arviz", reason="ArviZ required for CMC unit tests")

from homodyne.optimization.cmc import results  # noqa: E402


class _FakeCMCResult:
    def __init__(self):
        # two phi angles -> per-phi params contrast_0/1, offset_0/1
        self.n_chains = 1
        self.n_samples = 1
        self.samples = {
            "contrast_0": np.array([[1.0]]),
            "contrast_1": np.array([[1.0]]),
            "offset_0": np.array([[0.1]]),
            "offset_1": np.array([[0.2]]),
            # static params: D0, alpha, D_offset
            "D0": np.array([[1.0]]),
            "alpha": np.array([[0.5]]),
            "D_offset": np.array([[0.01]]),
        }

    def get_posterior_stats(self):
        # Return means matching the samples above
        return {
            "contrast_0": {"mean": 1.0},
            "contrast_1": {"mean": 1.0},
            "offset_0": {"mean": 0.1},
            "offset_1": {"mean": 0.2},
            "D0": {"mean": 1.0},
            "alpha": {"mean": 0.5},
            "D_offset": {"mean": 0.01},
        }


def test_compute_fitted_c2_uses_unique_phi(monkeypatch):
    # Phi has duplicates; model should pass unique phi to compute_g1_total
    phi = np.array([0.1, 0.1, 0.5])
    t = np.array([0.1, 0.2, 0.3])

    captured = {}

    def fake_compute_g1_total(
        params, t1, t2, phi_unique, q, L, dt, time_grid=None, _debug=False
    ):
        captured["phi_unique"] = np.array(phi_unique)
        # return ones to keep shapes simple
        return np.ones_like(t1)

    monkeypatch.setattr(
        "homodyne.core.physics_cmc.compute_g1_total", fake_compute_g1_total
    )

    res = _FakeCMCResult()

    c2_mean, c2_std = results.compute_fitted_c2(
        res,
        t1=t,
        t2=t,
        phi=phi,
        q=0.01,
        L=1.0,
        dt=0.01,
        analysis_mode="static",
    )

    # Assert unique phi was passed
    assert np.array_equal(captured["phi_unique"], np.array([0.1, 0.5]))

    # c2_mean should have same length as data
    assert c2_mean.shape == phi.shape
    assert c2_std.shape == phi.shape
