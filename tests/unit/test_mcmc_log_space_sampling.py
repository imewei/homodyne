"""Tests for log-space D0 sampling in single-angle static models."""

from __future__ import annotations

import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp
    from jax import random

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

try:
    import numpyro
    from numpyro import handlers
    from numpyro.infer import MCMC, NUTS

    NUMPYRO_AVAILABLE = True
except ImportError:
    NUMPYRO_AVAILABLE = False

from homodyne.config.parameter_space import ParameterSpace
from homodyne.optimization.mcmc import (
    _build_single_angle_surrogate_settings,
    _create_numpyro_model,
    _process_posterior_samples,
    _sample_single_angle_log_d0,
)


def _build_simple_model_input(n_phi: int = 1, n_points_per_phi: int = 20):
    total = n_phi * n_points_per_phi
    phi_vals = np.linspace(0.0, np.pi / 4, n_phi, endpoint=True)
    phi = np.repeat(phi_vals, n_points_per_phi)
    t1 = np.linspace(0, 1, total)
    t2 = np.linspace(0, 2, total)
    data = np.ones(total) * 1.05
    sigma = np.ones(total) * 0.01
    return {
        "data": data,
        "sigma": sigma,
        "t1": t1,
        "t2": t2,
        "phi": phi,
        "q": 0.01,
        "L": 1.0,
        "dt": 0.1,
    }


@pytest.mark.skipif(not NUMPYRO_AVAILABLE, reason="NumPyro not available")
class TestLogSpaceSurrogateConfiguration:
    def test_all_tiers_enable_log_sampling(self):
        param_space = ParameterSpace.from_defaults("static")
        for tier in ("2", "3", "4"):
            cfg = _build_single_angle_surrogate_settings(param_space, tier)
            assert cfg["sample_log_d0"] is True
            prior = cfg["log_d0_prior"]
            assert prior["high"] > prior["low"]

    def test_tier4_no_deterministic_clamp(self):
        param_space = ParameterSpace.from_defaults("static")
        cfg = _build_single_angle_surrogate_settings(param_space, "4")
        assert cfg["fixed_d0_value"] is None


@pytest.mark.skipif(not NUMPYRO_AVAILABLE, reason="NumPyro not available")
class TestSampleSingleAngleLogD0:
    def test_returns_positive_d0(self):
        prior_cfg = {
            "loc": np.log(1000.0),
            "scale": 0.4,
            "low": np.log(200.0),
            "high": np.log(5000.0),
        }

        def model():
            return _sample_single_angle_log_d0(prior_cfg, jnp.float64)

        seeded = handlers.seed(model, random.PRNGKey(0))
        trace = handlers.trace(seeded).get_trace()
        assert "D0" in trace
        assert "log_D0_latent" in trace
        value = float(trace["D0"]["value"])
        assert 100.0 < value < 10000.0

    def test_log_latent_matches_transform(self):
        prior_cfg = {
            "loc": np.log(500.0),
            "scale": 0.2,
            "low": np.log(100.0),
            "high": np.log(1000.0),
        }

        def model():
            return _sample_single_angle_log_d0(prior_cfg, jnp.float64)

        seeded = handlers.seed(model, random.PRNGKey(1))
        trace = handlers.trace(seeded).get_trace()
        log_latent = trace["log_D0_latent"]["value"]
        d0_value = trace["D0"]["value"]
        assert jnp.allclose(jnp.log(d0_value), log_latent)


@pytest.mark.skipif(not NUMPYRO_AVAILABLE, reason="NumPyro not available")
class TestLogSpaceModelIntegration:
    def test_single_angle_without_surrogate_uses_log_space(self):
        model_input = _build_simple_model_input(n_phi=1)
        param_space = ParameterSpace.from_defaults("static")
        model = _create_numpyro_model(
            data=model_input["data"],
            sigma=model_input["sigma"],
            t1=model_input["t1"],
            t2=model_input["t2"],
            phi=model_input["phi"],
            q=model_input["q"],
            L=model_input["L"],
            analysis_mode="static",
            parameter_space=param_space,
            dt=model_input["dt"],
            per_angle_scaling=False,
            fixed_scaling_overrides={"contrast": 0.05, "offset": 1.0},
        )
        seeded = handlers.seed(model, random.PRNGKey(2))
        trace = handlers.trace(seeded).get_trace()
        assert "log_D0_latent" in trace

    def test_multi_angle_path_uses_linear_sampling(self):
        model_input = _build_simple_model_input(n_phi=3)
        param_space = ParameterSpace.from_defaults("static")
        model = _create_numpyro_model(
            data=model_input["data"],
            sigma=model_input["sigma"],
            t1=model_input["t1"],
            t2=model_input["t2"],
            phi=model_input["phi"],
            q=model_input["q"],
            L=model_input["L"],
            analysis_mode="static",
            parameter_space=param_space,
            dt=model_input["dt"],
        )
        seeded = handlers.seed(model, random.PRNGKey(3))
        trace = handlers.trace(seeded).get_trace()
        assert "log_D0_latent" not in trace


@pytest.mark.skipif(not NUMPYRO_AVAILABLE, reason="NumPyro not available")
class TestLogSpaceDiagnostics:
    def test_process_posterior_samples_handles_log_latent(self):
        samples = {
            "log_D0_latent": jnp.array(
                [
                    np.log(800.0),
                    np.log(900.0),
                    np.log(1000.0),
                ]
            ),
            "alpha": jnp.array([-1.2, -1.15, -1.1]),
            "contrast": jnp.array([0.05, 0.051, 0.049]),
            "offset": jnp.array([1.0, 1.0, 1.0]),
        }
        grouped = {k: jnp.reshape(v, (1,) + v.shape) for k, v in samples.items()}
        dummy_result = type("Dummy", (), {})()
        dummy_result.get_samples = lambda group_by_chain=False: (
            grouped if group_by_chain else samples
        )
        dummy_result.get_extra_fields = lambda: {}
        summary = _process_posterior_samples(
            dummy_result,
            analysis_mode="static",
            diagnostic_settings={
                "max_rhat": 10.0,
                "min_ess": 0,
                "check_hmc_diagnostics": False,
                "expected_params": ["D0", "alpha", "D_offset"],
                "single_angle_surrogate": {
                    "drop_d_offset": True,
                    "fixed_d_offset": 0.0,
                },
            },
        )
        assert "D0" in summary["samples"]
        assert summary["diagnostic_summary"]["per_param_stats"]["D_offset"][
            "deterministic"
        ]


@pytest.mark.skipif(not NUMPYRO_AVAILABLE, reason="NumPyro not available")
class TestLogSpaceMultiChainESS:
    def test_two_chain_run_reports_rhat(self, monkeypatch):
        draws_per_chain = 80
        ground_truth = {
            "contrast": 0.05,
            "offset": 1.0,
            "D0": 1500.0,
            "alpha": -1.05,
        }

        from numpyro import diagnostics as np_diagnostics

        def _fake_rhat(values):
            arr = np.asarray(values)
            chain_means = arr.mean(axis=1)
            b = arr.shape[1] * np.var(chain_means, ddof=0)
            w = np.mean(np.var(arr, axis=1, ddof=0))
            if w <= 0:
                return 1.0
            ratio = ((arr.shape[1] - 1) / arr.shape[1]) + b / (arr.shape[1] * w)
            return float(np.sqrt(max(ratio, 1e-12)))

        def _fake_ess(values):
            arr = np.asarray(values)
            return float(arr.shape[0] * arr.shape[1] * 0.5)

        monkeypatch.setattr(np_diagnostics, "gelman_rubin", lambda arr: _fake_rhat(arr))
        monkeypatch.setattr(
            np_diagnostics, "effective_sample_size", lambda arr: _fake_ess(arr)
        )

        rng = np.random.default_rng(0)
        log_d0_chain_a = rng.normal(
            loc=np.log(ground_truth["D0"]), scale=0.015, size=draws_per_chain
        )
        log_d0_chain_b = rng.normal(
            loc=np.log(ground_truth["D0"] * 1.01), scale=0.015, size=draws_per_chain
        )
        alpha_chain_a = rng.normal(ground_truth["alpha"], 0.01, size=draws_per_chain)
        alpha_chain_b = rng.normal(
            ground_truth["alpha"] * 1.002, 0.01, size=draws_per_chain
        )

        def _stack(a_values, b_values):
            return np.stack([a_values, b_values], axis=0)

        samples_by_chain = {
            "log_D0_latent": _stack(log_d0_chain_a, log_d0_chain_b),
            "alpha": _stack(alpha_chain_a, alpha_chain_b),
            "contrast": _stack(
                np.full(draws_per_chain, ground_truth["contrast"]),
                np.full(draws_per_chain, ground_truth["contrast"] * 1.001),
            ),
            "offset": _stack(
                np.full(draws_per_chain, ground_truth["offset"]),
                np.full(draws_per_chain, ground_truth["offset"] * 0.999),
            ),
        }

        class DummyResult:
            def __init__(self, sample_dict):
                self._by_chain = {
                    name: jnp.asarray(values) for name, values in sample_dict.items()
                }
                self._flat = {
                    name: jnp.reshape(values, (-1,))
                    for name, values in self._by_chain.items()
                }

            def get_samples(self, group_by_chain=False):
                return self._by_chain if group_by_chain else self._flat

            def get_extra_fields(self):
                return {"accept_prob": jnp.full((draws_per_chain * 2,), 0.95)}

        dummy_result = DummyResult(samples_by_chain)
        diagnostic_settings = {
            "max_rhat": 1.2,
            "min_ess": 10,
            "check_hmc_diagnostics": True,
            "expected_params": ["D0", "alpha", "D_offset"],
            "single_angle_surrogate": {
                "drop_d_offset": True,
                "fixed_d_offset": 0.0,
            },
            "deterministic_params": {"contrast", "offset"},
        }
        summary = _process_posterior_samples(
            dummy_result,
            analysis_mode="static",
            diagnostic_settings=diagnostic_settings,
        )
        diag = summary["diagnostic_summary"]
        assert diag["multi_chain"] is True
        d0_stats = diag["per_param_stats"].get("D0")
        assert d0_stats is not None
        assert d0_stats["ess"] is not None and d0_stats["ess"] >= 80
        assert d0_stats["r_hat"] is not None and d0_stats["r_hat"] < 1.02
