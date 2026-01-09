"""
Unit Tests for Anti-Degeneracy Defense System v2.9.0
=====================================================

Tests cover:
- Layer 1: FourierReparameterizer (Fourier transformation)
- Layer 2: HierarchicalOptimizer (Two-stage optimization)
- Layer 3: AdaptiveRegularizer (CV-based regularization)
- Layer 4: GradientCollapseMonitor (Runtime detection)

Test IDs: T050-T059
"""

import numpy as np
import pytest

from homodyne.optimization.nlsq.adaptive_regularization import (
    AdaptiveRegularizationConfig,
    AdaptiveRegularizer,
)
from homodyne.optimization.nlsq.anti_degeneracy_controller import (
    AntiDegeneracyConfig,
    AntiDegeneracyController,
)
from homodyne.optimization.nlsq.fourier_reparam import (
    FourierReparamConfig,
    FourierReparameterizer,
)
from homodyne.optimization.nlsq.gradient_monitor import (
    GradientCollapseMonitor,
    GradientMonitorConfig,
)
from homodyne.optimization.nlsq.hierarchical import (
    HierarchicalConfig,
    HierarchicalOptimizer,
    HierarchicalResult,
)

# =============================================================================
# Layer 1: Fourier Reparameterization Tests
# =============================================================================


class TestFourierReparamConfig:
    """Test FourierReparamConfig dataclass."""

    def test_default_values(self):
        """T050a: Test default config values."""
        config = FourierReparamConfig()
        assert config.mode == "auto"
        assert config.fourier_order == 2
        assert config.auto_threshold == 6

    def test_from_dict(self):
        """T050b: Test config creation from dictionary."""
        config_dict = {
            "per_angle_mode": "fourier",
            "fourier_order": 3,
            "fourier_auto_threshold": 10,
        }
        config = FourierReparamConfig.from_dict(config_dict)
        assert config.mode == "fourier"
        assert config.fourier_order == 3
        assert config.auto_threshold == 10


class TestFourierReparameterizer:
    """Test FourierReparameterizer class (T050)."""

    @pytest.fixture
    def phi_angles_small(self):
        """Small set of phi angles (< threshold)."""
        return np.linspace(-np.pi, np.pi, 5)

    @pytest.fixture
    def phi_angles_large(self):
        """Large set of phi angles (> threshold)."""
        return np.linspace(-np.pi, np.pi, 23)

    def test_auto_mode_selects_independent_for_small(self, phi_angles_small):
        """T050c: Auto mode selects independent for n_phi < threshold."""
        config = FourierReparamConfig(mode="auto", auto_threshold=6)
        fourier = FourierReparameterizer(phi_angles_small, config)
        assert not fourier.use_fourier
        assert fourier.n_coeffs == 2 * len(phi_angles_small)

    def test_auto_mode_selects_fourier_for_large(self, phi_angles_large):
        """T050d: Auto mode selects Fourier for n_phi > threshold."""
        config = FourierReparamConfig(mode="auto", auto_threshold=6)
        fourier = FourierReparameterizer(phi_angles_large, config)
        assert fourier.use_fourier
        # order=2 -> 5 coeffs per group -> 10 total
        assert fourier.n_coeffs == 10
        assert fourier.n_coeffs_per_param == 5

    def test_fourier_transformation_roundtrip(self, phi_angles_large):
        """T050e: Fourier transformation is approximately invertible."""
        config = FourierReparamConfig(mode="fourier", fourier_order=2)
        fourier = FourierReparameterizer(phi_angles_large, config)

        # Create smooth per-angle values (should be well-represented by Fourier)
        contrast = 0.3 + 0.1 * np.cos(phi_angles_large)
        offset = 1.0 + 0.05 * np.sin(phi_angles_large)

        # Transform to Fourier and back
        coeffs = fourier.per_angle_to_fourier(contrast, offset)
        contrast_out, offset_out = fourier.fourier_to_per_angle(coeffs)

        # Should be close (not exact due to truncation)
        np.testing.assert_allclose(contrast, contrast_out, atol=0.01)
        np.testing.assert_allclose(offset, offset_out, atol=0.01)

    def test_to_from_fourier_single_group(self, phi_angles_large):
        """T050f: Single-group to_fourier/from_fourier methods."""
        config = FourierReparamConfig(mode="fourier", fourier_order=2)
        fourier = FourierReparameterizer(phi_angles_large, config)

        # Create smooth per-angle values
        values = 0.5 + 0.1 * np.cos(phi_angles_large)

        # Transform to Fourier and back
        coeffs = fourier.to_fourier(values)
        values_out = fourier.from_fourier(coeffs)

        assert len(coeffs) == fourier.n_coeffs_per_param
        np.testing.assert_allclose(values, values_out, atol=0.01)

    def test_array_validation_wrong_size(self, phi_angles_large):
        """T050g: Array validation catches wrong input size."""
        config = FourierReparamConfig(mode="fourier", fourier_order=2)
        fourier = FourierReparameterizer(phi_angles_large, config)

        # Wrong size array
        wrong_size = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Expected"):
            fourier.to_fourier(wrong_size)

        with pytest.raises(ValueError, match="Expected"):
            fourier.from_fourier(wrong_size)

    def test_array_validation_wrong_dims(self, phi_angles_large):
        """T050h: Array validation catches wrong dimensions."""
        config = FourierReparamConfig(mode="fourier", fourier_order=2)
        fourier = FourierReparameterizer(phi_angles_large, config)

        # 2D array instead of 1D
        wrong_dims = np.ones((5, 5))

        with pytest.raises(ValueError, match="must be 1D"):
            fourier.to_fourier(wrong_dims)

    def test_parameter_reduction_ratio(self, phi_angles_large):
        """T050i: Parameter reduction is significant for large n_phi."""
        config = FourierReparamConfig(mode="fourier", fourier_order=2)
        fourier = FourierReparameterizer(phi_angles_large, config)

        # 23 angles -> 46 per-angle params -> 10 Fourier coeffs
        n_original = 2 * len(phi_angles_large)
        reduction = fourier.n_coeffs / n_original

        assert reduction < 0.25  # At least 75% reduction
        assert fourier.get_diagnostics()["reduction_ratio"] == reduction


# =============================================================================
# Layer 2: Hierarchical Optimization Tests
# =============================================================================


class TestHierarchicalConfig:
    """Test HierarchicalConfig dataclass."""

    def test_default_values(self):
        """T051a: Test default config values."""
        config = HierarchicalConfig()
        assert config.enable is True
        assert config.max_outer_iterations == 5
        assert config.outer_tolerance == 1e-6

    def test_from_dict_safe_conversion(self):
        """T051b: Test safe type conversion in from_dict."""
        config_dict = {
            "enable": True,
            "max_outer_iterations": "invalid",  # Should use default
            "outer_tolerance": None,  # Should use default
        }
        config = HierarchicalConfig.from_dict(config_dict)
        assert config.max_outer_iterations == 5  # Default
        assert config.outer_tolerance == 1e-6  # Default


class TestHierarchicalOptimizer:
    """Test HierarchicalOptimizer class (T051)."""

    @pytest.fixture
    def simple_optimizer(self):
        """Create a simple hierarchical optimizer."""
        config = HierarchicalConfig(
            enable=True,
            max_outer_iterations=3,
            outer_tolerance=1e-4,
        )
        return HierarchicalOptimizer(
            config=config,
            n_phi=5,
            n_physical=3,
            fourier_reparameterizer=None,
        )

    def test_parameter_indices(self, simple_optimizer):
        """T051c: Test parameter index computation."""
        opt = simple_optimizer
        # 5 phi -> 10 per-angle params (5 contrast + 5 offset)
        assert opt.n_per_angle == 10
        # Indices are now numpy arrays for JAX compatibility
        np.testing.assert_array_equal(opt.per_angle_indices, np.arange(10))
        np.testing.assert_array_equal(opt.physical_indices, np.arange(10, 13))

    def test_fit_convergence(self, simple_optimizer):
        """T051d: Test hierarchical fit converges on simple problem."""
        opt = simple_optimizer

        # Simple quadratic loss
        def loss_fn(params):
            return np.sum((params - 1.0) ** 2)

        def grad_fn(params):
            return 2 * (params - 1.0)

        p0 = np.zeros(13)  # 10 per-angle + 3 physical
        bounds = (np.full(13, -10.0), np.full(13, 10.0))

        result = opt.fit(loss_fn, grad_fn, p0, bounds)

        assert isinstance(result, HierarchicalResult)
        assert result.success
        assert result.fun < 1e-3  # Should converge close to optimum

    def test_diagnostics(self, simple_optimizer):
        """T051e: Test diagnostics output."""
        diag = simple_optimizer.get_diagnostics()

        assert "enabled" in diag
        assert "n_phi" in diag
        assert "n_physical" in diag
        assert diag["n_phi"] == 5
        assert diag["n_physical"] == 3


# =============================================================================
# Layer 3: Adaptive Regularization Tests
# =============================================================================


class TestAdaptiveRegularizationConfig:
    """Test AdaptiveRegularizationConfig dataclass."""

    def test_default_values(self):
        """T052a: Test default config values."""
        config = AdaptiveRegularizationConfig()
        assert config.enable is True
        assert config.mode == "relative"
        assert config.lambda_base == 1.0
        assert config.target_cv == 0.10

    def test_from_dict_safe_conversion(self):
        """T052b: Test safe type conversion in from_dict."""
        config_dict = {
            "enable": True,
            "lambda": "invalid",  # Should use default
            "target_cv": None,  # Should use default
        }
        config = AdaptiveRegularizationConfig.from_dict(config_dict)
        assert config.lambda_base == 1.0  # Default
        assert config.target_cv == 0.10  # Default


class TestAdaptiveRegularizer:
    """Test AdaptiveRegularizer class (T052)."""

    @pytest.fixture
    def regularizer(self):
        """Create a regularizer for testing."""
        config = AdaptiveRegularizationConfig(
            enable=True,
            mode="relative",
            target_cv=0.10,
            target_contribution=0.10,
            auto_tune_lambda=True,
        )
        return AdaptiveRegularizer(config, n_phi=10)

    def test_auto_tuned_lambda(self, regularizer):
        """T052c: Test auto-tuned lambda calculation."""
        # λ = target_contribution / target_cv² = 0.10 / 0.01 = 10
        assert regularizer.lambda_value == pytest.approx(10.0)

    def test_cv_based_regularization(self, regularizer):
        """T052d: Test CV-based regularization computation."""
        # Params: [10 contrast, 10 offset, 3 physical]
        # Create params with 10% CV in per-angle groups
        contrast = np.full(10, 0.5) + np.random.randn(10) * 0.05
        offset = np.full(10, 1.0) + np.random.randn(10) * 0.10
        physical = np.array([1000.0, 0.5, 10.0])
        params = np.concatenate([contrast, offset, physical])

        mse = 0.04
        n_points = 1000000

        reg_term = regularizer.compute_regularization(params, mse, n_points)

        # Regularization should be non-zero for varying per-angle params
        assert reg_term > 0
        # Should be approximately 10% of SSE (since CV ~ target_cv)
        sse = mse * n_points
        contribution = reg_term / (sse + reg_term)
        assert 0.01 < contribution < 0.50  # Reasonable range

    def test_regularization_gradient(self, regularizer):
        """T052e: Test regularization gradient computation."""
        params = np.concatenate(
            [
                np.full(10, 0.5),  # contrast
                np.full(10, 1.0),  # offset
                np.array([1000.0, 0.5, 10.0]),  # physical
            ]
        )

        mse = 0.04
        n_points = 1000000

        grad = regularizer.compute_regularization_gradient(params, mse, n_points)

        assert grad.shape == params.shape
        # Physical params should have zero gradient (not regularized)
        np.testing.assert_array_equal(grad[-3:], 0.0)

    def test_n_group_validation(self):
        """T052f: Test n_group validation prevents division by zero."""
        config = AdaptiveRegularizationConfig(
            enable=True,
            group_indices=[(0, 1)],  # Only 1 element - should be skipped
        )
        regularizer = AdaptiveRegularizer(config, n_phi=10)

        params = np.ones(20)
        mse = 0.04
        n_points = 1000

        # Should not crash, should skip the group
        reg_term = regularizer.compute_regularization(params, mse, n_points)
        assert reg_term == 0.0  # No valid groups

    def test_constraint_violation_check(self, regularizer):
        """T052g: Test CV constraint violation detection."""
        # Create params with high CV (> max_cv of 0.20)
        contrast = np.linspace(0.2, 0.8, 10)  # High variance
        offset = np.full(10, 1.0)
        physical = np.array([1000.0, 0.5, 10.0])
        params = np.concatenate([contrast, offset, physical])

        violations = regularizer.check_constraint_violation(params)

        # Contrast group should violate (CV > 0.20)
        assert len(violations) > 0
        assert "group_0_contrast" in violations


# =============================================================================
# Layer 4: Gradient Collapse Monitor Tests
# =============================================================================


class TestGradientMonitorConfig:
    """Test GradientMonitorConfig dataclass."""

    def test_default_values(self):
        """T053a: Test default config values."""
        config = GradientMonitorConfig()
        assert config.enable is True
        assert config.ratio_threshold == 0.01
        assert config.consecutive_triggers == 5
        assert config.response_mode == "hierarchical"

    def test_from_dict_safe_conversion(self):
        """T053b: Test safe type conversion in from_dict."""
        config_dict = {
            "ratio_threshold": "invalid",  # Should use default
            "consecutive_triggers": None,  # Should use default
        }
        config = GradientMonitorConfig.from_dict(config_dict)
        assert config.ratio_threshold == 0.01  # Default
        assert config.consecutive_triggers == 5  # Default


class TestGradientCollapseMonitor:
    """Test GradientCollapseMonitor class (T053)."""

    @pytest.fixture
    def monitor(self):
        """Create a monitor for testing."""
        config = GradientMonitorConfig(
            enable=True,
            ratio_threshold=0.01,
            consecutive_triggers=3,
            response_mode="hierarchical",
        )
        return GradientCollapseMonitor(
            config=config,
            physical_indices=[10, 11, 12],
            per_angle_indices=list(range(10)),
        )

    def test_ok_status_for_healthy_gradients(self, monitor):
        """T053c: Test OK status for healthy gradient ratio."""
        # Physical gradients comparable to per-angle
        gradients = np.ones(13)  # Equal gradients -> ratio = 1.0

        status = monitor.check(gradients, iteration=0)
        assert status == "OK"
        assert not monitor.collapse_detected

    def test_warning_status_for_low_ratio(self, monitor):
        """T053d: Test WARNING status when ratio drops."""
        # Physical gradients much smaller than per-angle
        gradients = np.ones(13)
        gradients[10:13] = 0.001  # Physical grads very small

        status = monitor.check(gradients, iteration=0)
        assert status == "WARNING"
        assert monitor.consecutive_count == 1

    def test_collapse_detection_after_consecutive_triggers(self, monitor):
        """T053e: Test collapse detection after N consecutive triggers."""
        # Physical gradients much smaller than per-angle
        gradients = np.ones(13)
        gradients[10:13] = 0.001  # Physical grads very small -> ratio < 0.01

        # First 2 triggers: WARNING
        for i in range(2):
            status = monitor.check(gradients, iteration=i)
            assert status == "WARNING"
            assert not monitor.collapse_detected

        # Third trigger: COLLAPSE_DETECTED
        status = monitor.check(gradients, iteration=2)
        assert status == "COLLAPSE_DETECTED"
        assert monitor.collapse_detected
        assert len(monitor.collapse_events) == 1

    def test_consecutive_count_resets_on_healthy_gradient(self, monitor):
        """T053f: Test consecutive count resets on healthy gradient."""
        gradients_bad = np.ones(13)
        gradients_bad[10:13] = 0.001

        gradients_good = np.ones(13)

        # Build up consecutive count
        monitor.check(gradients_bad, iteration=0)
        monitor.check(gradients_bad, iteration=1)
        assert monitor.consecutive_count == 2

        # Reset with healthy gradient
        monitor.check(gradients_good, iteration=2)
        assert monitor.consecutive_count == 0

    def test_memory_leak_prevention(self, monitor):
        """T053g: Test history size is capped to prevent memory leaks."""
        gradients = np.ones(13)

        # Add more entries than MAX_HISTORY_SIZE
        for i in range(1500):
            monitor.check(gradients, iteration=i)

        # History should be capped
        assert len(monitor.history) <= GradientCollapseMonitor.MAX_HISTORY_SIZE
        assert len(monitor.history) == GradientCollapseMonitor.MAX_HISTORY_SIZE

    def test_get_response_after_collapse(self, monitor):
        """T053h: Test get_response returns action after collapse."""
        gradients = np.ones(13)
        gradients[10:13] = 0.001

        # Trigger collapse
        for i in range(5):
            monitor.check(gradients, iteration=i)

        response = monitor.get_response()
        assert response is not None
        assert response["mode"] == "hierarchical"
        assert "collapse_events" in response

    def test_disabled_monitor_always_returns_ok(self):
        """T053i: Test disabled monitor always returns OK."""
        config = GradientMonitorConfig(enable=False)
        monitor = GradientCollapseMonitor(
            config=config,
            physical_indices=[10, 11, 12],
            per_angle_indices=list(range(10)),
        )

        gradients = np.ones(13)
        gradients[10:13] = 0.0  # Zero physical gradient

        status = monitor.check(gradients, iteration=0)
        assert status == "OK"

    def test_diagnostics(self, monitor):
        """T053j: Test diagnostics output after checks."""
        gradients = np.ones(13)
        monitor.check(gradients, iteration=0)
        monitor.check(gradients, iteration=1)

        diag = monitor.get_diagnostics()

        assert "enabled" in diag
        assert "n_checks" in diag
        assert diag["n_checks"] == 2
        assert "min_ratio" in diag
        assert "max_ratio" in diag

    def test_reset_clears_state(self, monitor):
        """T053k: Test reset clears all state."""
        gradients = np.ones(13)
        gradients[10:13] = 0.001

        # Build up state
        for i in range(5):
            monitor.check(gradients, iteration=i)

        assert monitor.collapse_detected
        assert len(monitor.history) > 0

        # Reset
        monitor.reset()

        assert not monitor.collapse_detected
        assert len(monitor.history) == 0
        assert monitor.consecutive_count == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestAntiDegeneracyIntegration:
    """Integration tests for anti-degeneracy components working together."""

    def test_fourier_with_hierarchical(self):
        """T054a: Test Fourier reparameterizer works with hierarchical optimizer."""
        phi_angles = np.linspace(-np.pi, np.pi, 23)
        fourier_config = FourierReparamConfig(mode="fourier", fourier_order=2)
        fourier = FourierReparameterizer(phi_angles, fourier_config)

        hier_config = HierarchicalConfig(enable=True, max_outer_iterations=2)
        hier = HierarchicalOptimizer(
            config=hier_config,
            n_phi=23,
            n_physical=7,
            fourier_reparameterizer=fourier,
        )

        # With Fourier: n_per_angle should be n_coeffs (10), not 2*n_phi (46)
        assert hier.n_per_angle == fourier.n_coeffs

    def test_regularizer_with_gradient_monitor(self):
        """T054b: Test regularizer and gradient monitor work together."""
        # Setup regularizer
        reg_config = AdaptiveRegularizationConfig(enable=True, target_cv=0.10)
        regularizer = AdaptiveRegularizer(reg_config, n_phi=10)

        # Setup gradient monitor
        mon_config = GradientMonitorConfig(enable=True, consecutive_triggers=3)
        monitor = GradientCollapseMonitor(
            config=mon_config,
            physical_indices=list(range(20, 27)),
            per_angle_indices=list(range(20)),
        )

        # Create params and compute regularization gradient
        params = np.concatenate(
            [
                np.full(10, 0.5),
                np.full(10, 1.0),
                np.array([1000.0, 0.5, 10.0, 0.001, 0.5, 0.0, 0.0]),
            ]
        )

        grad = regularizer.compute_regularization_gradient(
            params, mse=0.04, n_points=1000
        )

        # Check gradient with monitor
        status = monitor.check(grad, iteration=0)
        assert status in ["OK", "WARNING"]


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_fourier_with_minimum_angles(self):
        """T055a: Test Fourier with minimum viable angle count."""
        # order=2 requires at least 5 angles (1 + 2*order)
        phi_angles = np.linspace(-np.pi, np.pi, 5)
        config = FourierReparamConfig(mode="fourier", fourier_order=2)
        fourier = FourierReparameterizer(phi_angles, config)

        assert fourier.use_fourier
        assert fourier.n_coeffs_per_param == 5

    def test_fourier_fallback_for_insufficient_angles(self):
        """T055b: Test Fourier falls back to independent for too few angles."""
        # order=2 requires 5 angles, provide only 3
        phi_angles = np.linspace(-np.pi, np.pi, 3)
        config = FourierReparamConfig(mode="fourier", fourier_order=2)
        fourier = FourierReparameterizer(phi_angles, config)

        # Should fall back to independent
        assert not fourier.use_fourier

    def test_regularizer_with_empty_groups(self):
        """T055c: Test regularizer handles empty group indices."""
        config = AdaptiveRegularizationConfig(
            enable=True,
            group_indices=[],  # Empty groups
        )
        regularizer = AdaptiveRegularizer(config, n_phi=10)

        params = np.ones(20)
        reg_term = regularizer.compute_regularization(params, mse=0.04, n_points=1000)

        assert reg_term == 0.0  # No groups to regularize

    def test_monitor_with_zero_gradient(self):
        """T055d: Test monitor handles zero gradient gracefully."""
        config = GradientMonitorConfig(enable=True)
        monitor = GradientCollapseMonitor(
            config=config,
            physical_indices=[3, 4, 5],
            per_angle_indices=[0, 1, 2],
        )

        # Zero gradients everywhere
        gradients = np.zeros(6)
        status = monitor.check(gradients, iteration=0)

        # Should not crash, ratio should be 0
        assert status == "WARNING"  # Zero physical / small per-angle triggers warning


# =============================================================================
# Full 4-Layer Integration Test with Synthetic 23-Angle Data
# =============================================================================


class TestAntiDegeneracyFull23Angle:
    """Full integration test for 23-angle laminar flow scenario."""

    @pytest.fixture
    def synthetic_23angle_setup(self):
        """Create synthetic 23-angle laminar flow data and components."""
        # 23 phi angles (typical XPCS setup)
        n_phi = 23
        phi_angles = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)

        # Physical parameters
        n_physical = (
            7  # D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_offset, phi0
        )

        # Layer 1: Fourier Reparameterization
        fourier_config = FourierReparamConfig(mode="fourier", fourier_order=2)
        fourier = FourierReparameterizer(phi_angles, fourier_config)

        # Layer 2: Hierarchical Optimizer
        hier_config = HierarchicalConfig(
            enable=True,
            max_outer_iterations=3,
            outer_tolerance=1e-4,
        )
        hier = HierarchicalOptimizer(
            config=hier_config,
            n_phi=n_phi,
            n_physical=n_physical,
            fourier_reparameterizer=fourier,
        )

        # Layer 3: Adaptive Regularizer
        reg_config = AdaptiveRegularizationConfig(
            enable=True,
            mode="relative",
            target_cv=0.10,
            target_contribution=0.10,
        )
        regularizer = AdaptiveRegularizer(reg_config, n_phi=n_phi)

        # Layer 4: Gradient Monitor (using Fourier indices)
        # Use numpy arrays for JAX compatibility
        n_fourier_per_angle = fourier.n_coeffs
        per_angle_indices = np.arange(n_fourier_per_angle, dtype=np.intp)
        physical_indices = np.arange(
            n_fourier_per_angle, n_fourier_per_angle + n_physical, dtype=np.intp
        )

        monitor_config = GradientMonitorConfig(
            enable=True,
            ratio_threshold=0.01,
            consecutive_triggers=3,
        )
        monitor = GradientCollapseMonitor(
            config=monitor_config,
            physical_indices=physical_indices,
            per_angle_indices=per_angle_indices,
        )

        return {
            "n_phi": n_phi,
            "n_physical": n_physical,
            "phi_angles": phi_angles,
            "fourier": fourier,
            "hierarchical": hier,
            "regularizer": regularizer,
            "monitor": monitor,
        }

    def test_fourier_reduces_params_from_46_to_10(self, synthetic_23angle_setup):
        """T056a: Fourier reparameterization reduces 46 per-angle params to 10 coeffs."""
        fourier = synthetic_23angle_setup["fourier"]

        # 23 angles * 2 groups (contrast + offset) = 46 per-angle params
        # With order=2: 2 * 5 = 10 Fourier coefficients
        assert fourier.n_coeffs == 10
        assert fourier.n_coeffs_per_param == 5

        # Reduction ratio
        reduction = fourier.n_coeffs / (2 * synthetic_23angle_setup["n_phi"])
        assert reduction < 0.25  # >75% reduction

    def test_hierarchical_uses_fourier_indices(self, synthetic_23angle_setup):
        """T056b: Hierarchical optimizer correctly uses Fourier-adjusted indices."""
        hier = synthetic_23angle_setup["hierarchical"]
        fourier = synthetic_23angle_setup["fourier"]

        # Hierarchical should use Fourier n_coeffs for per-angle count
        assert hier.n_per_angle == fourier.n_coeffs
        assert len(hier.per_angle_indices) == 10
        assert len(hier.physical_indices) == 7

    def test_regularizer_lambda_autotuned(self, synthetic_23angle_setup):
        """T056c: Regularizer auto-tunes lambda for 10% CV target."""
        regularizer = synthetic_23angle_setup["regularizer"]

        # λ = target_contribution / target_cv² = 0.10 / 0.01 = 10
        assert regularizer.lambda_value == pytest.approx(10.0)

    def test_full_pipeline_synthetic_data(self, synthetic_23angle_setup):
        """T056d: Full 4-layer defense on synthetic 23-angle data."""
        fourier = synthetic_23angle_setup["fourier"]
        regularizer = synthetic_23angle_setup["regularizer"]
        monitor = synthetic_23angle_setup["monitor"]
        n_phi = synthetic_23angle_setup["n_phi"]

        # Create synthetic per-angle values with known pattern
        # Smooth variation (should be well-captured by Fourier)
        phi_angles = synthetic_23angle_setup["phi_angles"]
        contrast = 0.3 + 0.05 * np.cos(phi_angles)  # 5% variation
        offset = 1.0 + 0.02 * np.sin(phi_angles)  # 2% variation

        # Transform to Fourier
        contrast_coeffs = fourier.to_fourier(contrast)
        offset_coeffs = fourier.to_fourier(offset)
        physical_params = np.array([1000.0, 0.5, 10.0, 0.01, 0.5, 0.0, 0.0])

        # Combine into full Fourier-space parameter vector
        fourier_params = np.concatenate(
            [contrast_coeffs, offset_coeffs, physical_params]
        )

        # Transform back to per-angle
        contrast_out = fourier.from_fourier(contrast_coeffs)
        offset_out = fourier.from_fourier(offset_coeffs)

        # Verify roundtrip preserves smooth variation
        np.testing.assert_allclose(contrast, contrast_out, atol=0.01)
        np.testing.assert_allclose(offset, offset_out, atol=0.01)

        # Test regularizer on reconstructed per-angle params
        full_params = np.concatenate([contrast_out, offset_out, physical_params])
        mse = 0.04
        n_points = 1000000

        reg_term = regularizer.compute_regularization(full_params, mse, n_points)
        assert reg_term > 0  # Should have non-zero regularization

        # Test gradient monitor with typical gradient magnitudes
        # Physical gradients should be comparable to per-angle for healthy optimization
        gradients = np.ones(len(fourier_params))
        gradients[10:17] = 0.5  # Physical gradients slightly smaller but still healthy

        status = monitor.check(gradients, iteration=0)
        assert status == "OK"  # Healthy gradient ratio

    def test_gradient_collapse_detection_in_fourier_space(
        self, synthetic_23angle_setup
    ):
        """T056e: Gradient monitor detects collapse with Fourier parameters."""
        monitor = synthetic_23angle_setup["monitor"]

        # Simulate collapsed physical gradients (typical degeneracy symptom)
        # 10 Fourier coeffs for per-angle, 7 physical
        n_total = 17
        gradients = np.ones(n_total)
        gradients[10:17] = 0.001  # Physical gradients very small -> ratio < 0.01

        # First check: WARNING
        status = monitor.check(gradients, iteration=0)
        assert status == "WARNING"

        # After N consecutive warnings: COLLAPSE_DETECTED
        for i in range(1, 3):  # 2 more to trigger
            status = monitor.check(gradients, iteration=i)

        assert status == "COLLAPSE_DETECTED"
        assert monitor.collapse_detected

    def test_cv_constraint_violation_detection(self, synthetic_23angle_setup):
        """T056f: Regularizer detects CV constraint violation."""
        regularizer = synthetic_23angle_setup["regularizer"]
        n_phi = synthetic_23angle_setup["n_phi"]

        # Create params with high variance (CV > max_cv of 0.20)
        contrast = np.linspace(0.1, 0.9, n_phi)  # High variance
        offset = np.full(n_phi, 1.0)
        physical = np.array([1000.0, 0.5, 10.0, 0.01, 0.5, 0.0, 0.0])
        params = np.concatenate([contrast, offset, physical])

        violations = regularizer.check_constraint_violation(params)

        # Contrast group should violate CV constraint
        assert len(violations) > 0
        assert "group_0_contrast" in violations

    def test_diagnostics_output(self, synthetic_23angle_setup):
        """T056g: All components provide diagnostic output."""
        fourier = synthetic_23angle_setup["fourier"]
        hier = synthetic_23angle_setup["hierarchical"]
        regularizer = synthetic_23angle_setup["regularizer"]
        monitor = synthetic_23angle_setup["monitor"]

        # Check diagnostics are available
        fourier_diag = fourier.get_diagnostics()
        assert "use_fourier" in fourier_diag
        assert "n_coeffs" in fourier_diag
        assert "reduction_ratio" in fourier_diag

        hier_diag = hier.get_diagnostics()
        assert "enabled" in hier_diag
        assert "n_per_angle" in hier_diag
        assert "fourier_enabled" in hier_diag

        reg_diag = regularizer.get_diagnostics()
        assert "enabled" in reg_diag
        assert "lambda" in reg_diag  # Key is 'lambda' not 'lambda_value'
        assert "mode" in reg_diag

        # Monitor needs at least one check for diagnostics
        gradients = np.ones(17)
        monitor.check(gradients, iteration=0)

        monitor_diag = monitor.get_diagnostics()
        assert "enabled" in monitor_diag
        assert "n_checks" in monitor_diag


# =============================================================================
# AntiDegeneracyController Tests
# =============================================================================


class TestAntiDegeneracyConfig:
    """Test AntiDegeneracyConfig dataclass."""

    def test_default_values(self):
        """T057a: Test default config values."""
        config = AntiDegeneracyConfig()
        assert config.enable is True
        assert config.per_angle_mode == "auto"
        assert config.fourier_order == 2
        assert config.hierarchical_enable is True

    def test_from_dict(self):
        """T057b: Test config creation from nested dictionary."""
        config_dict = {
            "enable": True,
            "per_angle_mode": "fourier",
            "fourier_order": 3,
            "hierarchical": {
                "enable": True,
                "max_outer_iterations": 10,
            },
            "regularization": {
                "mode": "relative",
                "lambda": 5.0,
            },
            "gradient_monitoring": {
                "enable": True,
                "ratio_threshold": 0.05,
            },
        }
        config = AntiDegeneracyConfig.from_dict(config_dict)
        assert config.per_angle_mode == "fourier"
        assert config.fourier_order == 3
        assert config.hierarchical_max_outer_iterations == 10
        assert config.regularization_lambda == 5.0
        assert config.gradient_ratio_threshold == 0.05


class TestAntiDegeneracyController:
    """Test AntiDegeneracyController orchestrator."""

    @pytest.fixture
    def controller_23angle(self):
        """Create controller for 23-angle laminar flow using Fourier mode.

        Note: We set constant_scaling_threshold=100 to force Fourier mode selection
        for this test, since the default constant_scaling_threshold=3 would otherwise
        select constant mode first.
        """
        config_dict = {
            "enable": True,
            "per_angle_mode": "auto",
            "fourier_order": 2,
            "fourier_auto_threshold": 6,
            "constant_scaling_threshold": 100,  # Disable constant mode for this test
            "hierarchical": {"enable": True},
            "regularization": {"mode": "relative"},
            "gradient_monitoring": {"enable": True},
        }
        phi_angles = np.linspace(-np.pi, np.pi, 23, endpoint=False)
        return AntiDegeneracyController.from_config(
            config_dict=config_dict,
            n_phi=23,
            phi_angles=phi_angles,
            n_physical=7,
            per_angle_scaling=True,
            is_laminar_flow=True,
        )

    @pytest.fixture
    def controller_5angle(self):
        """Create controller for 5-angle (below Fourier threshold) using independent mode.

        Note: We set constant_scaling_threshold=100 to force independent mode selection
        for this test, since the default constant_scaling_threshold=3 would otherwise
        select constant mode first.
        """
        config_dict = {
            "enable": True,
            "per_angle_mode": "auto",
            "fourier_order": 2,
            "fourier_auto_threshold": 6,
            "constant_scaling_threshold": 100,  # Disable constant mode for this test
            "hierarchical": {"enable": True},
            "regularization": {"mode": "relative"},
            "gradient_monitoring": {"enable": True},
        }
        phi_angles = np.linspace(-np.pi, np.pi, 5, endpoint=False)
        return AntiDegeneracyController.from_config(
            config_dict=config_dict,
            n_phi=5,
            phi_angles=phi_angles,
            n_physical=7,
            per_angle_scaling=True,
            is_laminar_flow=True,
        )

    def test_controller_enabled_for_23angle(self, controller_23angle):
        """T057c: Controller is enabled for 23-angle laminar flow."""
        assert controller_23angle.is_enabled
        assert controller_23angle.use_fourier  # Auto selects Fourier for n_phi > 6
        assert controller_23angle.use_hierarchical

    def test_auto_selects_fourier_for_23angle(self, controller_23angle):
        """T057d: Auto mode selects Fourier for n_phi > threshold."""
        assert controller_23angle.per_angle_mode_actual == "fourier"
        assert controller_23angle.n_per_angle_params == 10  # 2 * 5 coeffs

    def test_auto_selects_independent_for_5angle(self, controller_5angle):
        """T057e: Auto mode selects individual for n_phi < threshold."""
        assert controller_5angle.per_angle_mode_actual == "individual"
        assert not controller_5angle.use_fourier
        assert controller_5angle.n_per_angle_params == 10  # 2 * 5 direct params

    def test_components_initialized(self, controller_23angle):
        """T057f: All 4 layers are properly initialized."""
        assert controller_23angle.fourier is not None
        assert controller_23angle.hierarchical is not None
        assert controller_23angle.regularizer is not None
        assert controller_23angle.monitor is not None

    def test_transform_params_to_fourier(self, controller_23angle):
        """T057g: Parameter transformation to Fourier space."""
        # Create per-angle params: [23 contrast, 23 offset, 7 physical]
        params = np.concatenate(
            [
                np.full(23, 0.3),  # contrast
                np.full(23, 1.0),  # offset
                np.array([1000.0, 0.5, 10.0, 0.01, 0.5, 0.0, 0.0]),  # physical
            ]
        )

        fourier_params, _ = controller_23angle.transform_params_to_fourier(params)

        # Should have: [5 contrast coeffs, 5 offset coeffs, 7 physical] = 17
        assert len(fourier_params) == 17

    def test_transform_params_roundtrip(self, controller_23angle):
        """T057h: Fourier transformation roundtrip preserves params."""
        # Create smooth per-angle params
        phi = controller_23angle.phi_angles
        params = np.concatenate(
            [
                0.3 + 0.05 * np.cos(phi),  # smooth contrast
                1.0 + 0.02 * np.sin(phi),  # smooth offset
                np.array([1000.0, 0.5, 10.0, 0.01, 0.5, 0.0, 0.0]),
            ]
        )

        fourier_params, _ = controller_23angle.transform_params_to_fourier(params)
        restored_params = controller_23angle.transform_params_from_fourier(
            fourier_params
        )

        # Smooth functions should be well-preserved
        np.testing.assert_allclose(params, restored_params, atol=0.01)

    def test_group_variance_indices_fourier(self, controller_23angle):
        """T057i: Group variance indices correct for Fourier mode."""
        indices = controller_23angle.get_group_variance_indices()

        # With 5 Fourier coeffs per group:
        # Group 0 (contrast): [0, 5)
        # Group 1 (offset): [5, 10)
        assert indices == [(0, 5), (5, 10)]

    def test_group_variance_indices_independent(self, controller_5angle):
        """T057j: Group variance indices correct for independent mode."""
        indices = controller_5angle.get_group_variance_indices()

        # With 5 direct params per group:
        # Group 0 (contrast): [0, 5)
        # Group 1 (offset): [5, 10)
        assert indices == [(0, 5), (5, 10)]

    def test_diagnostics(self, controller_23angle):
        """T057k: Controller provides comprehensive diagnostics."""
        diag = controller_23angle.get_diagnostics()

        assert diag["version"] == "2.14.0"
        assert diag["enabled"] is True
        assert diag["per_angle_mode"] == "fourier"
        assert "fourier" in diag
        assert "hierarchical" in diag
        assert "regularization" in diag
        assert "gradient_monitor" in diag

    def test_disabled_for_static_mode(self):
        """T057l: Controller not initialized for static mode."""
        config_dict = {"enable": True}
        phi_angles = np.linspace(-np.pi, np.pi, 10)

        controller = AntiDegeneracyController.from_config(
            config_dict=config_dict,
            n_phi=10,
            phi_angles=phi_angles,
            n_physical=3,
            per_angle_scaling=True,
            is_laminar_flow=False,  # Static mode
        )

        assert not controller.is_enabled
        assert controller.fourier is None
        assert controller.hierarchical is None

    def test_reset_monitor(self, controller_23angle):
        """T057m: Monitor reset clears state."""
        monitor = controller_23angle.monitor

        # Add some state
        gradients = np.ones(17)
        monitor.check(gradients, iteration=0)
        assert len(monitor.history) > 0

        # Reset
        controller_23angle.reset_monitor()
        assert len(monitor.history) == 0


# =============================================================================
# JAX Array Indexing Compatibility Tests
# =============================================================================


class TestJAXArrayIndexingCompatibility:
    """Test that all anti-degeneracy components work correctly with JAX arrays.

    This test class verifies the fix for the "non-tuple sequence for
    multidimensional indexing" error that occurs when Python lists are used
    to index JAX arrays. All components should use numpy arrays for indices.

    Test IDs: T058
    """

    @pytest.fixture
    def jax_available(self):
        """Check if JAX is available."""
        try:
            import jax.numpy as jnp

            return True
        except ImportError:
            pytest.skip("JAX not available")

    def test_hierarchical_indices_are_numpy_arrays(self):
        """T058a: HierarchicalOptimizer indices are numpy arrays."""
        config = HierarchicalConfig()
        opt = HierarchicalOptimizer(config, n_phi=10, n_physical=3)

        assert isinstance(opt.per_angle_indices, np.ndarray)
        assert isinstance(opt.physical_indices, np.ndarray)
        assert opt.per_angle_indices.dtype == np.intp
        assert opt.physical_indices.dtype == np.intp

    def test_gradient_monitor_indices_are_numpy_arrays(self):
        """T058b: GradientCollapseMonitor indices are numpy arrays."""
        config = GradientMonitorConfig()
        monitor = GradientCollapseMonitor(
            config=config,
            physical_indices=range(10, 17),
            per_angle_indices=range(10),
        )

        assert isinstance(monitor.per_angle_indices, np.ndarray)
        assert isinstance(monitor.physical_indices, np.ndarray)
        assert monitor.per_angle_indices.dtype == np.intp
        assert monitor.physical_indices.dtype == np.intp

    def test_hierarchical_indices_work_with_jax_arrays(self, jax_available):
        """T058c: HierarchicalOptimizer indices can index JAX arrays."""
        import jax.numpy as jnp

        config = HierarchicalConfig()
        opt = HierarchicalOptimizer(config, n_phi=10, n_physical=3)

        # Create JAX array
        jax_array = jnp.arange(23)  # 20 per-angle + 3 physical

        # This should not raise "non-tuple sequence" error
        per_angle_subset = jax_array[opt.per_angle_indices]
        physical_subset = jax_array[opt.physical_indices]

        assert per_angle_subset.shape == (20,)
        assert physical_subset.shape == (3,)

    def test_gradient_monitor_indices_work_with_jax_arrays(self, jax_available):
        """T058d: GradientCollapseMonitor indices can index JAX arrays."""
        import jax.numpy as jnp

        config = GradientMonitorConfig()
        monitor = GradientCollapseMonitor(
            config=config,
            physical_indices=range(10, 17),
            per_angle_indices=range(10),
        )

        # Create JAX gradient array
        jax_grad = jnp.ones(17) * 0.1

        # This should not raise "non-tuple sequence" error
        phys_grad = jax_grad[monitor.physical_indices]
        per_angle_grad = jax_grad[monitor.per_angle_indices]

        assert phys_grad.shape == (7,)
        assert per_angle_grad.shape == (10,)

    def test_gradient_monitor_check_with_jax_gradient(self, jax_available):
        """T058e: GradientCollapseMonitor.check() works with JAX gradients."""
        import jax.numpy as jnp

        config = GradientMonitorConfig(enable=True)
        monitor = GradientCollapseMonitor(
            config=config,
            physical_indices=range(10, 17),
            per_angle_indices=range(10),
        )

        # Simulate JAX gradient
        jax_grad = jnp.ones(17) * 0.5

        # Should not raise error
        status = monitor.check(jax_grad, iteration=0)
        assert status in ["OK", "WARNING"]

    def test_hierarchical_with_fourier_uses_correct_indices(self, jax_available):
        """T058f: HierarchicalOptimizer with Fourier uses Fourier indices."""
        import jax.numpy as jnp

        # Create Fourier reparameterizer
        phi_angles = np.linspace(-np.pi, np.pi, 23)
        fourier_config = FourierReparamConfig(mode="fourier", fourier_order=2)
        fourier = FourierReparameterizer(phi_angles, fourier_config)

        # Create hierarchical optimizer with Fourier
        config = HierarchicalConfig()
        opt = HierarchicalOptimizer(
            config, n_phi=23, n_physical=7, fourier_reparameterizer=fourier
        )

        # Should have 10 Fourier coefficients, not 46 per-angle
        assert opt.n_per_angle == 10
        assert len(opt.per_angle_indices) == 10
        assert len(opt.physical_indices) == 7

        # Indices should work with JAX arrays
        jax_params = jnp.arange(17)
        per_angle_subset = jax_params[opt.per_angle_indices]
        physical_subset = jax_params[opt.physical_indices]

        assert per_angle_subset.shape == (10,)
        assert physical_subset.shape == (7,)

    def test_full_fourier_hierarchical_jax_gradient_flow(self, jax_available):
        """T058g: Full integration test with JAX gradient through all components."""
        import jax
        import jax.numpy as jnp

        # Setup
        n_phi = 23
        n_physical = 7
        phi_angles = np.linspace(-np.pi, np.pi, n_phi)

        # Layer 1: Fourier
        fourier_config = FourierReparamConfig(mode="fourier", fourier_order=2)
        fourier = FourierReparameterizer(phi_angles, fourier_config)

        # Layer 2: Hierarchical
        hier_config = HierarchicalConfig()
        hier = HierarchicalOptimizer(
            hier_config, n_phi, n_physical, fourier_reparameterizer=fourier
        )

        # Layer 3: Regularizer with Fourier-aware group indices
        n_coeffs_per_param = fourier.n_coeffs_per_param
        fourier_group_indices = [
            (0, n_coeffs_per_param),
            (n_coeffs_per_param, 2 * n_coeffs_per_param),
        ]
        reg_config = AdaptiveRegularizationConfig(
            enable=True,
            mode="relative",
            group_indices=fourier_group_indices,
        )
        regularizer = AdaptiveRegularizer(reg_config, n_phi)

        # Layer 4: Gradient monitor with Fourier indices
        monitor_config = GradientMonitorConfig(enable=True)
        monitor = GradientCollapseMonitor(
            config=monitor_config,
            physical_indices=hier.physical_indices,
            per_angle_indices=hier.per_angle_indices,
        )

        # Create a simple JAX loss function
        def loss_fn(params):
            # Simple quadratic loss
            return jnp.sum((params - 1.0) ** 2)

        # Compute gradient using JAX
        grad_fn = jax.grad(loss_fn)
        params = jnp.zeros(17)  # 10 Fourier + 7 physical
        jax_grad = grad_fn(params)

        # All components should work with JAX gradient
        # Hierarchical indices
        physical_grad = jax_grad[hier.physical_indices]
        per_angle_grad = jax_grad[hier.per_angle_indices]
        assert physical_grad.shape == (7,)
        assert per_angle_grad.shape == (10,)

        # Gradient monitor
        status = monitor.check(jax_grad, iteration=0)
        assert status in ["OK", "WARNING"]

        # Regularizer (uses numpy internally, so convert)
        numpy_params = np.asarray(params)
        reg_term = regularizer.compute_regularization(
            numpy_params, mse=0.04, n_points=1000
        )
        assert reg_term >= 0
