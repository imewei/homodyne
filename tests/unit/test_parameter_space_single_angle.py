"""Tests for single-angle prior stabilization utilities."""

import pytest

from homodyne.config.parameter_space import ParameterSpace, PriorDistribution


def _make_parameter_space() -> ParameterSpace:
    bounds = {
        "contrast": (0.0, 1.0),
        "offset": (0.5, 1.5),
        "D0": (10.0, 1000.0),
        "alpha": (-2.0, 2.0),
        "D_offset": (-100.0, 100.0),
    }
    priors = {
        name: PriorDistribution(
            dist_type="TruncatedNormal",
            mu=(lo + hi) / 2.0,
            sigma=(hi - lo) / 4.0,
            min_val=lo,
            max_val=hi,
        )
        for name, (lo, hi) in bounds.items()
    }
    return ParameterSpace(
        model_type="static",
        parameter_names=list(bounds.keys()),
        bounds=bounds,
        priors=priors,
        units={},
    )


def test_single_angle_scalars_tightened():
    space = _make_parameter_space()
    stabilized = space.with_single_angle_stabilization()

    assert stabilized.bounds["contrast"] == (0.3, 0.7)
    assert stabilized.bounds["offset"] == (0.9, 1.1)
    assert stabilized.priors["contrast"].sigma == pytest.approx(0.05)
    assert stabilized.priors["offset"].sigma == pytest.approx(0.05)


def test_single_angle_beta_fallback_applies_to_physics():
    space = _make_parameter_space()
    stabilized = space.with_single_angle_stabilization(enable_beta_fallback=True)

    for param in ("D0", "alpha", "D_offset"):
        prior = stabilized.priors[param]
        assert prior.dist_type == "BetaScaled"
        assert prior.alpha == pytest.approx(2.0)
        assert prior.beta == pytest.approx(2.0)


def test_get_single_angle_fallback_prior_respects_bounds():
    space = _make_parameter_space()
    prior = space.get_single_angle_fallback_prior("D0")
    assert prior.dist_type == "BetaScaled"
    assert prior.min_val == pytest.approx(10.0)
    assert prior.max_val == pytest.approx(1000.0)
    assert prior.alpha == pytest.approx(2.0)
    assert prior.beta == pytest.approx(2.0)


def test_single_angle_geometry_config_defaults():
    space = _make_parameter_space()
    cfg = space.get_single_angle_geometry_config()
    assert cfg["enabled"] is True
    assert cfg["log_center_loc"] > 0
    assert cfg["log_center_scale"] > 0
    assert cfg["delta_floor"] > 0


def test_single_angle_geometry_handles_negative_center():
    space = _make_parameter_space()
    space.priors["D_offset"] = PriorDistribution(
        dist_type="TruncatedNormal",
        mu=-500.0,
        sigma=50.0,
        min_val=-1000.0,
        max_val=-100.0,
    )
    cfg = space.get_single_angle_geometry_config()
    assert cfg["log_center_loc"] > 0  # falls back to positive reference
    assert cfg["delta_loc"] < 5
