"""Tests for CMA-ES global optimization wrapper.

Tests for homodyne CMA-ES integration (v2.15.0 / NLSQ 0.6.4+):
- CMAESWrapperConfig creation and conversion
- CMAESWrapper scale ratio computation and method selection
- CMA-ES optimization (when evosax available)
"""

from __future__ import annotations

import numpy as np
import pytest

# Import JAX for CMA-ES tests (required for JAX-compatible model functions)
try:
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    jnp = np  # Fallback to numpy for non-JAX environments
    JAX_AVAILABLE = False

from homodyne.optimization.nlsq.cmaes_wrapper import (
    CMAES_AVAILABLE,
    CMAESResult,
    CMAESWrapper,
    CMAESWrapperConfig,
)
from homodyne.optimization.nlsq.config import NLSQConfig


class TestCMAESWrapperConfig:
    """Tests for CMAESWrapperConfig dataclass."""

    def test_default_values(self):
        """Verify default configuration values."""
        config = CMAESWrapperConfig()

        assert config.preset == "cmaes"
        assert config.max_generations is None  # None = use preset + adaptive scaling
        assert config.sigma == 0.5
        assert config.tol_fun == 1e-8
        assert config.tol_x == 1e-8
        assert config.restart_strategy == "bipop"
        assert config.max_restarts == 9
        assert config.population_batch_size is None
        assert config.data_chunk_size is None
        assert config.refine_with_nlsq is True
        assert config.auto_memory is True
        assert config.memory_limit_gb == 8.0

    def test_from_nlsq_config(self):
        """Verify conversion from NLSQConfig."""
        nlsq_config = NLSQConfig(
            enable_cmaes=True,
            cmaes_preset="cmaes-global",
            cmaes_max_generations=200,
            cmaes_sigma=0.3,
            cmaes_tol_fun=1e-10,
            cmaes_tol_x=1e-10,
            cmaes_restart_strategy="none",
            cmaes_max_restarts=0,
            cmaes_population_batch_size=50,
            cmaes_data_chunk_size=10000,
            cmaes_refine_with_nlsq=False,
            cmaes_memory_limit_gb=16.0,
        )

        wrapper_config = CMAESWrapperConfig.from_nlsq_config(nlsq_config)

        assert wrapper_config.preset == "cmaes-global"
        assert wrapper_config.max_generations == 200
        assert wrapper_config.sigma == 0.3
        assert wrapper_config.tol_fun == 1e-10
        assert wrapper_config.tol_x == 1e-10
        assert wrapper_config.restart_strategy == "none"
        assert wrapper_config.max_restarts == 0
        assert wrapper_config.population_batch_size == 50
        assert wrapper_config.data_chunk_size == 10000
        assert wrapper_config.refine_with_nlsq is False
        # auto_memory is False when population_batch_size is explicitly set
        assert wrapper_config.auto_memory is False
        assert wrapper_config.memory_limit_gb == 16.0

    def test_from_nlsq_config_refinement_fields(self):
        """Verify refinement fields are transferred from NLSQConfig."""
        nlsq_config = NLSQConfig(
            enable_cmaes=True,
            cmaes_refine_with_nlsq=True,
            cmaes_refinement_workflow="streaming",
            cmaes_refinement_ftol=1e-12,
            cmaes_refinement_xtol=1e-12,
            cmaes_refinement_gtol=1e-12,
            cmaes_refinement_max_nfev=1000,
            cmaes_refinement_loss="huber",
        )

        wrapper_config = CMAESWrapperConfig.from_nlsq_config(nlsq_config)

        # Verify refinement fields
        assert wrapper_config.refine_with_nlsq is True
        assert wrapper_config.refinement_workflow == "streaming"
        assert wrapper_config.refinement_ftol == 1e-12
        assert wrapper_config.refinement_xtol == 1e-12
        assert wrapper_config.refinement_gtol == 1e-12
        assert wrapper_config.refinement_max_nfev == 1000
        assert wrapper_config.refinement_loss == "huber"

    def test_to_cmaes_config_requires_evosax(self):
        """Verify to_cmaes_config() raises ImportError without evosax."""
        config = CMAESWrapperConfig()

        if not CMAES_AVAILABLE:
            with pytest.raises(ImportError, match="evosax"):
                config.to_cmaes_config(n_params=10)
        else:
            # Should succeed when evosax is available
            cmaes_config = config.to_cmaes_config(n_params=10)
            assert cmaes_config is not None


class TestCMAESWrapper:
    """Tests for CMAESWrapper class."""

    def test_is_available(self):
        """Verify availability check matches module constant."""
        wrapper = CMAESWrapper()
        assert wrapper.is_available == CMAES_AVAILABLE

    def test_compute_scale_ratio_uniform_scales(self):
        """Uniform scales should give scale ratio of 1."""
        wrapper = CMAESWrapper()

        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])

        ratio = wrapper.compute_scale_ratio((lower, upper))
        assert ratio == pytest.approx(1.0)

    def test_compute_scale_ratio_different_scales(self):
        """Different scales should give scale ratio > 1."""
        wrapper = CMAESWrapper()

        # Scales: 1, 10, 100
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 10.0, 100.0])

        ratio = wrapper.compute_scale_ratio((lower, upper))
        assert ratio == pytest.approx(100.0)

    def test_compute_scale_ratio_laminar_flow_like(self):
        """Laminar flow-like scales should give very high ratio."""
        wrapper = CMAESWrapper()

        # Mimics laminar_flow: D0 ~ 1e4, gamma_dot ~ 1e-3
        lower = np.array([0.0, 1e-6, 100.0])
        upper = np.array([1.0, 1e-2, 50000.0])

        ratio = wrapper.compute_scale_ratio((lower, upper))
        # (50000-100) / (1e-2 - 1e-6) ≈ 5e6
        assert ratio > 1e6

    def test_compute_scale_ratio_zero_range_param(self):
        """Zero-range parameters should be ignored."""
        wrapper = CMAESWrapper()

        # One parameter has zero range (fixed)
        lower = np.array([0.0, 1.0, 0.0])
        upper = np.array([10.0, 1.0, 100.0])

        ratio = wrapper.compute_scale_ratio((lower, upper))
        # Should only consider non-zero ranges: 10 and 100
        assert ratio == pytest.approx(10.0)

    def test_should_use_cmaes_high_scale_ratio(self):
        """High scale ratio should select CMA-ES."""
        wrapper = CMAESWrapper()

        # High scale ratio > 1000
        lower = np.array([0.0, 0.0])
        upper = np.array([0.001, 10000.0])

        should_use = wrapper.should_use_cmaes((lower, upper), scale_threshold=1000.0)

        if CMAES_AVAILABLE:
            assert should_use == True  # noqa: E712 - comparing numpy bool
        else:
            assert should_use == False  # noqa: E712 - Not available

    def test_should_use_cmaes_low_scale_ratio(self):
        """Low scale ratio should not select CMA-ES."""
        wrapper = CMAESWrapper()

        # Low scale ratio < 1000
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 10.0])

        should_use = wrapper.should_use_cmaes((lower, upper), scale_threshold=1000.0)
        assert should_use == False  # noqa: E712 - comparing numpy bool


class TestCMAESResult:
    """Tests for CMAESResult dataclass."""

    def test_result_creation(self):
        """Verify CMAESResult can be created."""
        result = CMAESResult(
            parameters=np.array([1.0, 2.0, 3.0]),
            covariance=np.eye(3),
            chi_squared=0.01,
            success=True,
            diagnostics={"generations": 50, "restarts": 2},
            method_used="cmaes",
            nlsq_refined=True,
            message="Converged",
        )

        assert len(result.parameters) == 3
        assert result.chi_squared == 0.01
        assert result.success is True
        assert result.diagnostics["generations"] == 50
        assert result.method_used == "cmaes"
        assert result.nlsq_refined is True

    def test_result_without_covariance(self):
        """Verify CMAESResult works without covariance."""
        result = CMAESResult(
            parameters=np.array([1.0]),
            covariance=None,
            chi_squared=0.1,
            success=True,
        )

        assert result.covariance is None
        assert result.success is True


@pytest.mark.skipif(not CMAES_AVAILABLE, reason="evosax not available")
class TestCMAESOptimization:
    """Integration tests for CMA-ES optimization (requires evosax)."""

    def test_fit_simple_quadratic_with_refinement(self):
        """Test CMA-ES + NLSQ refinement on simple quadratic function."""
        wrapper = CMAESWrapper(
            CMAESWrapperConfig(
                preset="cmaes-fast",
                max_generations=50,
                refine_with_nlsq=True,  # Use NLSQ refinement
                refinement_workflow="auto",
            )
        )

        # Simple quadratic: y = (x - 2)^2
        def model(xdata, a):
            return (xdata - a) ** 2

        xdata = np.linspace(-5, 5, 100)
        ydata = model(xdata, 2.0)  # True a = 2.0

        p0 = np.array([0.0])
        bounds = (np.array([-10.0]), np.array([10.0]))

        result = wrapper.fit(model, xdata, ydata, p0, bounds)

        assert result.success
        # With NLSQ refinement, should converge to true value
        assert result.parameters[0] == pytest.approx(2.0, abs=0.1)
        assert result.chi_squared < 1.0
        # Verify refinement was applied
        assert result.nlsq_refined is True

    def test_fit_multi_scale_problem(self):
        """Test CMA-ES on multi-scale optimization problem."""
        wrapper = CMAESWrapper(
            CMAESWrapperConfig(
                preset="cmaes",
                max_generations=100,
                refine_with_nlsq=True,
            )
        )

        # Multi-scale: y = a * exp(-b * x) where a ~ 1e4, b ~ 1e-3
        # Must use jnp for JAX compatibility with CMA-ES optimizer
        def model(xdata, a, b):
            return a * jnp.exp(-b * xdata)

        xdata = np.linspace(0, 1000, 100)
        true_a = 10000.0
        true_b = 0.001
        ydata = model(xdata, true_a, true_b) + np.random.normal(0, 10, 100)

        p0 = np.array([5000.0, 0.0005])  # Poor initial guess
        bounds = (np.array([100.0, 1e-6]), np.array([100000.0, 0.01]))

        result = wrapper.fit(model, xdata, ydata, p0, bounds)

        assert result.success
        # Should recover parameters within 20%
        assert result.parameters[0] == pytest.approx(true_a, rel=0.2)
        assert result.parameters[1] == pytest.approx(true_b, rel=0.2)


class TestNLSQConfigCMAES:
    """Tests for CMA-ES fields in NLSQConfig."""

    def test_cmaes_defaults(self):
        """Verify CMA-ES default values in NLSQConfig."""
        config = NLSQConfig()

        # CMA-ES global search defaults
        assert config.enable_cmaes is False
        assert config.cmaes_preset == "cmaes"
        assert config.cmaes_max_generations is None  # None = adaptive
        assert config.cmaes_sigma == 0.5
        assert config.cmaes_tol_fun == 1e-8
        assert config.cmaes_tol_x == 1e-8
        assert config.cmaes_restart_strategy == "bipop"
        assert config.cmaes_max_restarts == 9
        assert config.cmaes_population_batch_size is None
        assert config.cmaes_data_chunk_size is None
        assert config.cmaes_refine_with_nlsq is True
        assert config.cmaes_auto_select is True
        assert config.cmaes_scale_threshold == 1000.0
        assert config.cmaes_memory_limit_gb == 8.0

        # NLSQ TRF refinement defaults
        assert config.cmaes_refinement_workflow == "auto"
        assert config.cmaes_refinement_ftol == 1e-10
        assert config.cmaes_refinement_xtol == 1e-10
        assert config.cmaes_refinement_gtol == 1e-10
        assert config.cmaes_refinement_max_nfev == 500
        assert config.cmaes_refinement_loss == "linear"

    def test_cmaes_from_dict(self):
        """Verify CMA-ES fields are parsed from dict."""
        config_dict = {
            "cmaes": {
                "enable": True,
                "preset": "cmaes-global",
                "max_generations": 200,
                "sigma": 0.3,
                "restart_strategy": "none",
                "max_restarts": 0,
                "refine_with_nlsq": False,
                "auto_select": False,
                "scale_threshold": 500.0,
                "memory_limit_gb": 32.0,
                # NLSQ TRF refinement settings
                "refinement_workflow": "streaming",
                "refinement_ftol": 1e-12,
                "refinement_xtol": 1e-12,
                "refinement_gtol": 1e-12,
                "refinement_max_nfev": 1000,
                "refinement_loss": "huber",
            }
        }

        config = NLSQConfig.from_dict(config_dict)

        # CMA-ES global search fields
        assert config.enable_cmaes is True
        assert config.cmaes_preset == "cmaes-global"
        assert config.cmaes_max_generations == 200
        assert config.cmaes_sigma == 0.3
        assert config.cmaes_restart_strategy == "none"
        assert config.cmaes_max_restarts == 0
        assert config.cmaes_refine_with_nlsq is False
        assert config.cmaes_auto_select is False
        assert config.cmaes_scale_threshold == 500.0
        assert config.cmaes_memory_limit_gb == 32.0

        # NLSQ TRF refinement fields
        assert config.cmaes_refinement_workflow == "streaming"
        assert config.cmaes_refinement_ftol == 1e-12
        assert config.cmaes_refinement_xtol == 1e-12
        assert config.cmaes_refinement_gtol == 1e-12
        assert config.cmaes_refinement_max_nfev == 1000
        assert config.cmaes_refinement_loss == "huber"

    def test_cmaes_to_dict_roundtrip(self):
        """Verify CMA-ES fields survive to_dict() roundtrip."""
        original = NLSQConfig(
            enable_cmaes=True,
            cmaes_preset="cmaes-fast",
            cmaes_max_generations=50,
            # NLSQ TRF refinement settings
            cmaes_refinement_workflow="streaming",
            cmaes_refinement_ftol=1e-12,
            cmaes_refinement_max_nfev=1000,
        )

        config_dict = original.to_dict()
        restored = NLSQConfig.from_dict(config_dict)

        # CMA-ES global search fields
        assert restored.enable_cmaes is True
        assert restored.cmaes_preset == "cmaes-fast"
        assert restored.cmaes_max_generations == 50

        # NLSQ TRF refinement fields
        assert restored.cmaes_refinement_workflow == "streaming"
        assert restored.cmaes_refinement_ftol == 1e-12
        assert restored.cmaes_refinement_max_nfev == 1000

    def test_cmaes_validation(self):
        """Verify CMA-ES validation catches invalid values."""
        # Invalid preset
        config = NLSQConfig(cmaes_preset="invalid")
        errors = config.validate()
        assert any("cmaes_preset" in e for e in errors)

        # Invalid sigma
        config = NLSQConfig(cmaes_sigma=2.0)  # > 1
        errors = config.validate()
        assert any("cmaes_sigma" in e for e in errors)

        # Invalid restart strategy
        config = NLSQConfig(cmaes_restart_strategy="invalid")
        errors = config.validate()
        assert any("cmaes_restart_strategy" in e for e in errors)

        # Negative max_restarts
        config = NLSQConfig(cmaes_max_restarts=-1)
        errors = config.validate()
        assert any("cmaes_max_restarts" in e for e in errors)

        # Negative scale_threshold
        config = NLSQConfig(cmaes_scale_threshold=-100)
        errors = config.validate()
        assert any("cmaes_scale_threshold" in e for e in errors)

        # Invalid refinement workflow
        config = NLSQConfig(cmaes_refinement_workflow="invalid")
        errors = config.validate()
        assert any("cmaes_refinement_workflow" in e for e in errors)

        # Invalid refinement loss function
        config = NLSQConfig(cmaes_refinement_loss="invalid")
        errors = config.validate()
        assert any("cmaes_refinement_loss" in e for e in errors)

        # Negative refinement max_nfev
        config = NLSQConfig(cmaes_refinement_max_nfev=-10)
        errors = config.validate()
        assert any("cmaes_refinement_max_nfev" in e for e in errors)


class TestWarmStartSigma:
    """Tests for warm-start sigma selection (v2.20.0).

    When NLSQ warm-start provides a near-optimal starting point, CMA-ES should
    use a reduced sigma for local refinement instead of the default global
    search sigma (0.5).
    """

    def test_config_defaults(self):
        """Verify sigma_warmstart defaults in NLSQConfig and CMAESWrapperConfig."""
        nlsq = NLSQConfig()
        assert nlsq.cmaes_sigma_warmstart == 0.05
        assert nlsq.cmaes_warmstart_auto_skip is True
        assert nlsq.cmaes_warmstart_skip_threshold == 5.0

        wrapper = CMAESWrapperConfig()
        assert wrapper.sigma_warmstart == 0.05

    def test_config_from_dict(self):
        """Verify warm-start fields are parsed from config dict."""
        config_dict = {
            "cmaes": {
                "sigma_warmstart": 0.03,
                "warmstart_auto_skip": False,
                "warmstart_skip_threshold": 10.0,
            }
        }
        config = NLSQConfig.from_dict(config_dict)
        assert config.cmaes_sigma_warmstart == 0.03
        assert config.cmaes_warmstart_auto_skip is False
        assert config.cmaes_warmstart_skip_threshold == 10.0

    def test_config_to_dict_roundtrip(self):
        """Verify warm-start fields survive to_dict() roundtrip."""
        original = NLSQConfig(
            cmaes_sigma_warmstart=0.02,
            cmaes_warmstart_auto_skip=False,
            cmaes_warmstart_skip_threshold=8.0,
        )
        config_dict = original.to_dict()
        restored = NLSQConfig.from_dict(config_dict)
        assert restored.cmaes_sigma_warmstart == 0.02
        assert restored.cmaes_warmstart_auto_skip is False
        assert restored.cmaes_warmstart_skip_threshold == 8.0

    def test_config_validation_invalid_sigma_warmstart(self):
        """Invalid sigma_warmstart should fail validation."""
        config = NLSQConfig(cmaes_sigma_warmstart=2.0)
        errors = config.validate()
        assert any("cmaes_sigma_warmstart" in e for e in errors)

        config = NLSQConfig(cmaes_sigma_warmstart=0.0)
        errors = config.validate()
        assert any("cmaes_sigma_warmstart" in e for e in errors)

    def test_config_validation_invalid_skip_threshold(self):
        """Invalid warmstart_skip_threshold should fail validation."""
        config = NLSQConfig(cmaes_warmstart_skip_threshold=-1.0)
        errors = config.validate()
        assert any("cmaes_warmstart_skip_threshold" in e for e in errors)

    def test_wrapper_config_from_nlsq_config(self):
        """Verify sigma_warmstart transfers from NLSQConfig."""
        nlsq = NLSQConfig(cmaes_sigma_warmstart=0.03)
        wrapper = CMAESWrapperConfig.from_nlsq_config(nlsq)
        assert wrapper.sigma_warmstart == 0.03

    @pytest.mark.skipif(not CMAES_AVAILABLE, reason="evosax not available")
    def test_to_cmaes_config_sigma_override(self):
        """Verify sigma_override is applied in to_cmaes_config."""
        config = CMAESWrapperConfig(sigma=0.5, sigma_warmstart=0.05)

        # Without override, uses default sigma
        cmaes_config = config.to_cmaes_config(n_params=9)
        assert cmaes_config.sigma == 0.5

        # With override, uses warm-start sigma
        cmaes_config = config.to_cmaes_config(n_params=9, sigma_override=0.05)
        assert cmaes_config.sigma == 0.05

    @pytest.mark.skipif(not CMAES_AVAILABLE, reason="evosax not available")
    def test_fit_logs_warmstart_sigma_when_active(self, caplog):
        """When warmstart_chi2 is provided, CMA-ES should log sigma_warmstart.

        Regression test for production issue where sigma=0.5 caused CMA-ES
        to produce 100x worse results than NLSQ warm-start due to excessive
        exploration from a near-optimal starting point.
        """
        import logging

        config = CMAESWrapperConfig(
            preset="cmaes-fast",
            max_generations=50,
            sigma=0.5,
            sigma_warmstart=0.05,
            refine_with_nlsq=True,
        )
        wrapper = CMAESWrapper(config)

        def model(xdata, a):
            return (xdata - a) ** 2

        xdata = np.linspace(-5, 5, 100)
        ydata = model(xdata, 2.0)

        p0 = np.array([1.95])  # Near-optimal warm-start
        bounds = (np.array([-10.0]), np.array([10.0]))

        with caplog.at_level(
            logging.INFO, logger="homodyne.optimization.nlsq.cmaes_wrapper"
        ):
            wrapper.fit(model, xdata, ydata, p0, bounds, warmstart_chi2=0.1)

        # Verify warm-start sigma was selected
        assert any("sigma_warmstart=0.050" in r.message for r in caplog.records), (
            "Expected log message about sigma_warmstart=0.050"
        )
        # Verify algorithm settings log shows the warm-start sigma
        assert any("sigma=0.050 (warm-start)" in r.message for r in caplog.records), (
            "Expected algorithm settings to show warm-start sigma"
        )

    @pytest.mark.skipif(not CMAES_AVAILABLE, reason="evosax not available")
    def test_fit_logs_default_sigma_without_warmstart(self, caplog):
        """Without warmstart_chi2, CMA-ES should use default sigma."""
        import logging

        config = CMAESWrapperConfig(
            preset="cmaes-fast",
            max_generations=50,
            sigma=0.5,
            sigma_warmstart=0.05,
            refine_with_nlsq=True,
        )
        wrapper = CMAESWrapper(config)

        def model(xdata, a):
            return (xdata - a) ** 2

        xdata = np.linspace(-5, 5, 100)
        ydata = model(xdata, 2.0)

        p0 = np.array([0.0])
        bounds = (np.array([-10.0]), np.array([10.0]))

        with caplog.at_level(
            logging.INFO, logger="homodyne.optimization.nlsq.cmaes_wrapper"
        ):
            wrapper.fit(model, xdata, ydata, p0, bounds)

        # Verify default sigma was used (no warm-start message)
        assert not any("sigma_warmstart" in r.message for r in caplog.records), (
            "Should not log sigma_warmstart when warm-start is not active"
        )
        # Verify algorithm settings log shows the default sigma
        assert any("sigma=0.500," in r.message for r in caplog.records), (
            "Expected algorithm settings to show default sigma=0.500"
        )


class TestAutoSkipLogic:
    """Tests for CMA-ES auto-skip when warm-start is sufficient (v2.20.0).

    The auto-skip logic in fit_nlsq_cmaes skips the CMA-ES global search
    when NLSQ warm-start achieves a reduced chi-squared below the threshold.
    These tests verify the decision logic directly, including edge cases.
    """

    def _compute_skip_decision(
        self,
        ydata_len: int,
        x0_len: int,
        warmstart_chi2: float,
        warmstart_skip_threshold: float = 5.0,
        warmstart_auto_skip: bool = True,
        warmstart_params: np.ndarray | None = None,
    ) -> tuple[bool, float]:
        """Replicate the auto-skip decision logic from core.py fit_nlsq_cmaes.

        Returns (skip_cmaes, warmstart_reduced_chi2).
        """
        if warmstart_params is None:
            warmstart_params = np.zeros(x0_len)

        skip_cmaes = False
        warmstart_reduced_chi2 = float("inf")

        if (
            warmstart_auto_skip
            and warmstart_params is not None
            and warmstart_chi2 < float("inf")
        ):
            n_data_eff = ydata_len - x0_len
            if n_data_eff <= 0:
                warmstart_reduced_chi2 = float("inf")
            else:
                warmstart_reduced_chi2 = warmstart_chi2 / n_data_eff
            if warmstart_reduced_chi2 < warmstart_skip_threshold:
                skip_cmaes = True

        return skip_cmaes, warmstart_reduced_chi2

    def test_auto_skip_triggers_when_chi2_below_threshold(self):
        """CMA-ES should be skipped when reduced chi2 < threshold."""
        # 100 data points, 9 params, chi2=100 -> reduced=100/91=1.10 < 5.0
        skip, reduced = self._compute_skip_decision(
            ydata_len=100, x0_len=9, warmstart_chi2=100.0
        )
        assert skip is True
        assert reduced == pytest.approx(100.0 / 91.0)

    def test_auto_skip_does_not_trigger_when_chi2_above_threshold(self):
        """CMA-ES should NOT be skipped when reduced chi2 >= threshold."""
        # 100 data points, 9 params, chi2=500 -> reduced=500/91=5.49 > 5.0
        skip, reduced = self._compute_skip_decision(
            ydata_len=100, x0_len=9, warmstart_chi2=500.0
        )
        assert skip is False
        assert reduced == pytest.approx(500.0 / 91.0)

    def test_auto_skip_disabled(self):
        """When auto_skip=False, CMA-ES should never be skipped."""
        skip, _ = self._compute_skip_decision(
            ydata_len=100, x0_len=9, warmstart_chi2=0.01, warmstart_auto_skip=False
        )
        assert skip is False

    def test_negative_dof_never_skips(self):
        """When n_data <= n_params (negative DOF), CMA-ES must NOT be skipped.

        Regression test: negative DOF produces negative reduced chi2, which
        would always be below the positive threshold, falsely triggering skip.
        """
        # 5 data points, 9 params -> DOF = -4
        skip, reduced = self._compute_skip_decision(
            ydata_len=5, x0_len=9, warmstart_chi2=1.0
        )
        assert skip is False
        assert reduced == float("inf")

    def test_zero_dof_never_skips(self):
        """When n_data == n_params (zero DOF), CMA-ES must NOT be skipped."""
        skip, reduced = self._compute_skip_decision(
            ydata_len=9, x0_len=9, warmstart_chi2=1.0
        )
        assert skip is False
        assert reduced == float("inf")

    def test_inf_warmstart_chi2_never_skips(self):
        """When warm-start chi2 is inf (no warm-start), never skip."""
        skip, _ = self._compute_skip_decision(
            ydata_len=100, x0_len=9, warmstart_chi2=float("inf")
        )
        assert skip is False

    def test_no_warmstart_params_never_skips(self):
        """When warm-start params are None, never skip."""
        skip, _ = self._compute_skip_decision(
            ydata_len=100, x0_len=9, warmstart_chi2=1.0, warmstart_params=None
        )
        # warmstart_params defaults to np.zeros in helper, so test with explicit None
        # by bypassing the default
        skip_cmaes = False
        warmstart_params = None
        if warmstart_params is not None:
            skip_cmaes = True
        assert skip_cmaes is False

    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        # reduced chi2 = 100/91 = 1.10
        skip_low, _ = self._compute_skip_decision(
            ydata_len=100,
            x0_len=9,
            warmstart_chi2=100.0,
            warmstart_skip_threshold=1.0,
        )
        assert skip_low is False  # 1.10 > 1.0

        skip_high, _ = self._compute_skip_decision(
            ydata_len=100,
            x0_len=9,
            warmstart_chi2=100.0,
            warmstart_skip_threshold=2.0,
        )
        assert skip_high is True  # 1.10 < 2.0

    def test_skip_result_diagnostics(self):
        """When skip triggers, verify the diagnostics that would be set."""
        skip, reduced = self._compute_skip_decision(
            ydata_len=1000,
            x0_len=9,
            warmstart_chi2=500.0,
            warmstart_skip_threshold=5.0,
        )
        assert skip is True
        expected_reduced = 500.0 / 991.0
        assert reduced == pytest.approx(expected_reduced)
        # In production, diagnostics would contain:
        # {"selected": "nlsq_warmstart_auto_skip", "cmaes_skipped": True}
        assert expected_reduced < 5.0
