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
        assert config.max_generations == 100
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
        assert config.cmaes_max_generations == 100
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
