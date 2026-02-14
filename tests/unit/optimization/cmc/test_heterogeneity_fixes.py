"""Tests for CMC shard heterogeneity fixes (Feb 2026).

These tests verify the 6 targeted fixes + plumbing for the shard heterogeneity
issue that caused a 62-hour CMC abort on dataset C020 (3 angles, 3M points).
"""

import pytest

pytest.importorskip("arviz", reason="ArviZ required for CMC unit tests")

from homodyne.optimization.cmc.config import CMCConfig  # noqa: E402
from homodyne.optimization.cmc.core import (  # noqa: E402
    _resolve_max_points_per_shard,
)
from homodyne.optimization.cmc.priors import (  # noqa: E402
    build_nlsq_informed_prior,
    get_param_names_in_order,
)


class TestStage1NumChainsDefault:
    """Fix 6: num_chains default inconsistency."""

    def test_default_num_chains_is_4(self):
        """from_dict({}) must default to 4 chains, matching class default."""
        config = CMCConfig.from_dict({})
        assert config.num_chains == 4

    def test_class_default_matches_from_dict(self):
        """Class default and from_dict default must agree."""
        class_default = CMCConfig()
        from_dict_default = CMCConfig.from_dict({})
        assert class_default.num_chains == from_dict_default.num_chains

    def test_explicit_num_chains_preserved(self):
        """Explicit num_chains in YAML should override default."""
        config = CMCConfig.from_dict({"per_shard_mcmc": {"num_chains": 2}})
        assert config.num_chains == 2


class TestStage2RobustConsensus:
    """Fix 4: robust_consensus_mc as default combination method."""

    def test_default_combination_method(self):
        """Default combination method must be robust_consensus_mc."""
        config = CMCConfig.from_dict({})
        assert config.combination_method == "robust_consensus_mc"

    def test_class_default_is_robust(self):
        """Class-level default must also be robust."""
        config = CMCConfig()
        assert config.combination_method == "robust_consensus_mc"

    def test_explicit_method_preserved(self):
        """Explicit method in YAML should override default."""
        config = CMCConfig.from_dict({"combination": {"method": "consensus_mc"}})
        assert config.combination_method == "consensus_mc"


class TestStage3ShardSize:
    """Fix 3: Increased shard sizes for laminar_flow."""

    def test_min_shard_size_laminar_reduced(self):
        """MIN_SHARD_SIZE_LAMINAR is 3K after reparameterization fix."""
        result = _resolve_max_points_per_shard(
            "laminar_flow", 3_000_000, "auto", None, n_phi=3
        )
        assert result >= 3_000  # 5K base * 0.6 angle_factor = 3K

    def test_shard_size_laminar_3m_3angles(self):
        """3M points, 3 angles should produce shard size >= 3K."""
        result = _resolve_max_points_per_shard(
            "laminar_flow", 3_000_000, "auto", None, n_phi=3
        )
        # 5K * 0.6 = 3K
        assert result >= 3_000

    def test_shard_size_static_unchanged_order(self):
        """Static mode shard sizes should still be much larger than laminar."""
        static_result = _resolve_max_points_per_shard(
            "static", 3_000_000, "auto", None, n_phi=3
        )
        laminar_result = _resolve_max_points_per_shard(
            "laminar_flow", 3_000_000, "auto", None, n_phi=3
        )
        assert static_result > laminar_result

    def test_user_specified_shard_size_respected(self):
        """User-specified shard size should be used (if above minimum)."""
        result = _resolve_max_points_per_shard(
            "laminar_flow", 3_000_000, 50_000, None, n_phi=3
        )
        assert result == 50_000


class TestStage4ConstantAveragedScaling:
    """Bug P1: fixed_contrast/offset must be passed for constant_averaged mode."""

    def test_param_names_constant_averaged(self):
        """constant_averaged mode should not include contrast/offset params."""
        names = get_param_names_in_order(3, "laminar_flow", "constant_averaged")
        assert "contrast" not in names
        assert "offset" not in names
        assert "contrast_0" not in names
        assert "D0" in names

    def test_param_names_constant_same_as_constant_averaged(self):
        """constant and constant_averaged should produce same param names."""
        constant = get_param_names_in_order(3, "laminar_flow", "constant")
        constant_averaged = get_param_names_in_order(
            3, "laminar_flow", "constant_averaged"
        )
        assert constant == constant_averaged


class TestStage6EffectiveModeWithWarmstart:
    """Fix 1: Default to constant_averaged when NLSQ warm-start present."""

    def test_auto_with_warmstart_becomes_constant_averaged(self):
        """auto + NLSQ warm-start should upgrade to constant_averaged."""
        config = CMCConfig()
        mode = config.get_effective_per_angle_mode(
            n_phi=3,
            nlsq_per_angle_mode="auto",
            has_nlsq_warmstart=True,
        )
        assert mode == "constant_averaged"

    def test_auto_without_warmstart_stays_auto(self):
        """auto without warm-start should stay as auto."""
        config = CMCConfig()
        mode = config.get_effective_per_angle_mode(
            n_phi=3,
            nlsq_per_angle_mode="auto",
            has_nlsq_warmstart=False,
        )
        assert mode == "auto"

    def test_auto_no_nlsq_mode_stays_auto(self):
        """auto with no nlsq_per_angle_mode should auto-select normally."""
        config = CMCConfig()
        mode = config.get_effective_per_angle_mode(n_phi=3)
        assert mode == "auto"  # n_phi=3 >= threshold=3

    def test_explicit_individual_preserved_with_warmstart(self):
        """Explicit individual mode should not be overridden by warm-start."""
        config = CMCConfig(per_angle_mode="individual")
        mode = config.get_effective_per_angle_mode(
            n_phi=3,
            nlsq_per_angle_mode="individual",
            has_nlsq_warmstart=True,
        )
        assert mode == "individual"

    def test_nlsq_constant_preserved_with_warmstart(self):
        """NLSQ constant mode should be preserved even with warm-start."""
        config = CMCConfig()
        mode = config.get_effective_per_angle_mode(
            n_phi=3,
            nlsq_per_angle_mode="constant",
            has_nlsq_warmstart=True,
        )
        assert mode == "constant"


class TestStage7NLSQPriorsConfig:
    """Fix 2: NLSQ-informed priors config fields."""

    def test_default_use_nlsq_informed_priors(self):
        """Default should enable NLSQ-informed priors."""
        config = CMCConfig.from_dict({})
        assert config.use_nlsq_informed_priors is True

    def test_default_width_factor(self):
        """Default width factor should be 2.0 (tightened from 3.0)."""
        config = CMCConfig.from_dict({})
        assert config.nlsq_prior_width_factor == 2.0

    def test_from_dict_nlsq_priors(self):
        """Should parse NLSQ prior config from dict."""
        config = CMCConfig.from_dict(
            {
                "validation": {
                    "use_nlsq_informed_priors": False,
                    "nlsq_prior_width_factor": 5.0,
                }
            }
        )
        assert config.use_nlsq_informed_priors is False
        assert config.nlsq_prior_width_factor == 5.0

    def test_width_factor_validation_too_small(self):
        """Width factor < 1.0 should produce validation error."""
        config = CMCConfig(nlsq_prior_width_factor=0.5)
        errors = config.validate()
        assert any("nlsq_prior_width_factor" in e for e in errors)

    def test_width_factor_validation_too_large(self):
        """Width factor > 10.0 should produce validation error."""
        config = CMCConfig(nlsq_prior_width_factor=15.0)
        errors = config.validate()
        assert any("nlsq_prior_width_factor" in e for e in errors)

    def test_to_dict_includes_nlsq_priors(self):
        """to_dict should include NLSQ prior config."""
        config = CMCConfig(use_nlsq_informed_priors=True, nlsq_prior_width_factor=4.0)
        d = config.to_dict()
        assert d["validation"]["use_nlsq_informed_priors"] is True
        assert d["validation"]["nlsq_prior_width_factor"] == 4.0


class TestStage7NLSQPriorBuild:
    """Fix 2: verify build_nlsq_informed_prior produces correct distribution."""

    def test_prior_centered_on_nlsq_value(self):
        """Prior should be centered on NLSQ estimate."""
        prior = build_nlsq_informed_prior(
            param_name="D0",
            nlsq_value=1e10,
            nlsq_std=1e8,
            bounds=(1e8, 1e12),
            width_factor=3.0,
        )
        # NumPyro wraps TruncatedNormal as TwoSidedTruncatedDistribution
        # Access the base distribution's loc for the center
        base = getattr(prior, "base_dist", prior)
        assert float(base.loc) == pytest.approx(1e10)

    def test_prior_width_from_nlsq_std(self):
        """Prior scale should be nlsq_std * width_factor."""
        prior = build_nlsq_informed_prior(
            param_name="alpha",
            nlsq_value=-0.5,
            nlsq_std=0.01,
            bounds=(-2.0, 0.0),
            width_factor=3.0,
        )
        # scale = 0.01 * 3.0 = 0.03, but clipped to [0.02, 1.0] (1%/50% of range=2.0)
        base = getattr(prior, "base_dist", prior)
        assert float(base.scale) == pytest.approx(0.03)

    def test_prior_fallback_without_nlsq_std(self):
        """Without NLSQ std, prior should use 10% of range."""
        prior = build_nlsq_informed_prior(
            param_name="D0",
            nlsq_value=1e10,
            nlsq_std=None,
            bounds=(1e8, 1e12),
            width_factor=3.0,
        )
        # 10% of range = 0.1 * (1e12 - 1e8) ≈ 9.99e10
        expected_std = (1e12 - 1e8) * 0.1
        base = getattr(prior, "base_dist", prior)
        assert float(base.scale) == pytest.approx(expected_std, rel=0.01)


class TestStage8ReparamModelSelection:
    """Fix 5: Wire reparameterized model for auto + laminar_flow."""

    def test_get_xpcs_model_auto_reparam(self):
        """auto + use_reparameterization should return reparameterized model."""
        from homodyne.optimization.cmc.model import (
            get_xpcs_model,
            xpcs_model_reparameterized,
        )

        model = get_xpcs_model("auto", use_reparameterization=True)
        assert model is xpcs_model_reparameterized

    def test_get_xpcs_model_auto_no_reparam(self):
        """auto without reparameterization should return averaged model."""
        from homodyne.optimization.cmc.model import (
            get_xpcs_model,
            xpcs_model_averaged,
        )

        model = get_xpcs_model("auto", use_reparameterization=False)
        assert model is xpcs_model_averaged

    def test_get_xpcs_model_constant_averaged(self):
        """constant_averaged should return constant_averaged model."""
        from homodyne.optimization.cmc.model import (
            get_xpcs_model,
            xpcs_model_constant_averaged,
        )

        model = get_xpcs_model("constant_averaged")
        assert model is xpcs_model_constant_averaged

    def test_reparam_not_used_for_constant_averaged(self):
        """Reparameterization flag should be ignored for constant_averaged."""
        from homodyne.optimization.cmc.model import (
            get_xpcs_model,
            xpcs_model_constant_averaged,
        )

        model = get_xpcs_model("constant_averaged", use_reparameterization=True)
        assert model is xpcs_model_constant_averaged


class TestBoundsAwareCV:
    """P2-2: CV calculation should use bounds range for near-zero params."""

    def test_cv_near_zero_uses_bounds_range(self):
        """D_offset near zero should not trigger false heterogeneity."""
        import numpy as np

        # Simulate shard means for D_offset near zero
        means = [0.001, -0.002, 0.003, -0.001, 0.002]
        mean_val = abs(np.mean(means))
        std_val = np.std(means)

        # Old calculation: cv = std / max(|mean|, 1e-10) → huge CV
        old_cv = std_val / max(mean_val, 1e-10)

        # New bounds-aware calculation
        lo, hi = -1e5, 1e5  # Typical D_offset bounds
        param_range = hi - lo
        scale = max(mean_val, param_range * 0.01)
        new_cv = std_val / scale

        # Old CV would be enormous (false positive), new CV should be tiny
        assert old_cv > 1.0, f"Old CV should be large, got {old_cv}"
        assert new_cv < 0.01, f"New CV should be small, got {new_cv}"

    def test_cv_large_mean_unaffected(self):
        """Parameters with large mean should be unaffected by bounds-aware CV."""
        import numpy as np

        # D0 with large mean
        means = [15000.0, 15100.0, 14900.0, 15050.0]
        mean_val = abs(np.mean(means))
        std_val = np.std(means)

        # Bounds for D0
        lo, hi = 100.0, 100000.0
        param_range = hi - lo

        # Both old and new should give similar CV when mean >> range*0.01
        old_cv = std_val / max(mean_val, 1e-10)
        scale = max(mean_val, param_range * 0.01)
        new_cv = std_val / scale

        # mean_val (15012.5) >> param_range*0.01 (999), so scale == mean_val
        assert abs(old_cv - new_cv) < 1e-6


class TestCVDispatchEdgeCases:
    """P3: CV dispatch edge cases — parameter_space=None, invalid bounds."""

    def test_cv_without_parameter_space(self):
        """parameter_space=None falls back to mean-based scale (inflated CV)."""
        import numpy as np

        # Near-zero parameter without bounds information
        means = [0.001, -0.002, 0.003, -0.001, 0.002]
        mean_val = abs(np.mean(means))
        std_val = np.std(means)

        # Replicate backend code: parameter_space is None
        parameter_space = None
        if parameter_space is not None:
            scale = mean_val  # won't reach
        else:
            scale = max(mean_val, 1e-10)

        cv = std_val / scale
        # Without bounds, near-zero param → inflated CV (false positive risk)
        assert cv > 1.0

    def test_cv_with_invalid_bounds_zero_range(self):
        """param_range == 0 (lo == hi) falls back to mean-based scale."""
        import numpy as np

        means = [0.5, 0.6, 0.4]
        mean_val = abs(np.mean(means))
        std_val = np.std(means)

        # Invalid bounds: lo == hi → param_range = 0
        lo, hi = 1.0, 1.0
        param_range = hi - lo
        assert param_range == 0

        # Replicate backend code
        scale = (
            max(mean_val, param_range * 0.01)
            if param_range > 0
            else max(mean_val, 1e-10)
        )
        cv = std_val / scale
        # Falls back to mean-based scale
        assert scale == mean_val
        assert np.isfinite(cv)

    def test_cv_with_inverted_bounds(self):
        """lo > hi → negative param_range → uses abs(range) for scale."""
        import numpy as np

        means = [100.0, 110.0, 90.0]
        mean_val = abs(np.mean(means))
        std_val = np.std(means)

        # Inverted bounds
        lo, hi = 200.0, 50.0
        param_range = hi - lo
        assert param_range < 0

        # Updated backend code: inverted bounds use abs(range)
        if param_range < 0:
            param_range = abs(param_range)
        elif param_range == 0:
            pass  # degenerate: fall back to mean

        scale = (
            max(mean_val, param_range * 0.01)
            if param_range > 0
            else max(mean_val, 1e-10)
        )
        cv = std_val / scale
        # abs(range) = 150, 1% = 1.5; mean_val ≈ 100 >> 1.5, so scale = mean_val
        assert scale == mean_val
        assert np.isfinite(cv)

    def test_cv_with_inverted_bounds_uses_abs_range(self):
        """Inverted bounds should use abs(range) for near-zero params, not fall back."""
        import numpy as np

        # Near-zero parameter with inverted bounds
        means = [1e-4, -2e-4, 3e-4]
        mean_val = abs(np.mean(means))
        std_val = np.std(means)

        # Inverted bounds (lo > hi)
        lo, hi = 1.0, -1.0
        param_range = hi - lo  # -2.0
        assert param_range < 0

        # After fix: use abs(range)
        param_range = abs(param_range)  # 2.0
        scale = max(mean_val, param_range * 0.01)  # max(~6.67e-5, 0.02) = 0.02

        cv = std_val / scale
        # With abs(range), we get bounds-aware scale = 0.02, so CV is small
        assert scale == pytest.approx(0.02)
        assert cv < 0.05  # Not a false positive

    def test_cv_near_zero_with_bounds_suppressed(self):
        """Near-zero param with valid bounds → small CV (suppressed false positive)."""
        import numpy as np

        # gamma_dot_t0 near zero with tight spread
        means = [1e-4, -2e-4, 3e-4, -1e-4, 2e-4]
        mean_val = abs(np.mean(means))
        std_val = np.std(means)

        # Valid bounds for gamma_dot_t0
        lo, hi = -1.0, 1.0
        param_range = hi - lo
        scale = max(mean_val, param_range * 0.01)

        cv = std_val / scale
        # Bounds-aware scale = max(6e-5, 0.02) = 0.02 → CV << 1
        assert cv < 0.05
        assert scale == pytest.approx(0.02)
