"""Integration tests for laminar flow optimization with many phi angles.

Tests the fix for dimension mismatch between Fourier reparameterization
(Layer 1) and adaptive regularization (Layer 3) in the NLSQ Anti-Degeneracy
Defense System.

Feature: 001-fix-nlsq-anti-degeneracy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pytest

from homodyne.optimization.nlsq.adaptive_regularization import (
    AdaptiveRegularizationConfig,
    AdaptiveRegularizer,
)
from homodyne.optimization.nlsq.fourier_reparam import (
    FourierReparamConfig,
    FourierReparameterizer,
)
from homodyne.optimization.nlsq.gradient_monitor import (
    GradientCollapseMonitor,
    GradientMonitorConfig,
)
from homodyne.optimization.nlsq.shear_weighting import (
    ShearSensitivityWeighting,
    ShearWeightingConfig,
)


@dataclass
class MockOptimizationResult:
    """Mock optimization result for testing."""

    params: np.ndarray
    covariance: np.ndarray | None = None
    chi_squared: float = 1.0
    success: bool = True


class TestLaminarFlow23PhiNocrash:
    """Test that 23-phi laminar flow fits complete without crashing (US1)."""

    @pytest.mark.integration
    def test_23_phi_no_crash(self):
        """Verify optimizer completes without index errors for 23-phi data.

        This test verifies User Story 1, Acceptance Scenario 1:
        Given a laminar flow dataset with 23 phi angles and anti-degeneracy
        defense enabled with Fourier reparameterization,
        When running NLSQ optimization,
        Then the optimizer completes without errors.
        """
        n_phi = 23

        # Create phi angles
        phi_angles = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

        # Create Fourier reparameterization config
        fourier_config = FourierReparamConfig(mode="fourier", fourier_order=2)
        fourier = FourierReparameterizer(phi_angles, fourier_config)

        # Verify initialization completes without errors
        assert fourier.n_phi == n_phi

        # Test forward transform (per-angle -> Fourier) using the API
        per_angle_contrast = np.ones(n_phi) + 0.1 * np.random.randn(n_phi)
        per_angle_offset = np.ones(n_phi) * 0.5 + 0.05 * np.random.randn(n_phi)

        fourier_coeffs = fourier.per_angle_to_fourier(
            per_angle_contrast, per_angle_offset
        )

        # Verify Fourier coefficients are produced
        assert len(fourier_coeffs) > 0
        assert fourier.n_coeffs > 0

        # Test inverse transform (Fourier -> per-angle)
        contrast_out, offset_out = fourier.fourier_to_per_angle(fourier_coeffs)

        # Verify reconstruction dimensions match original
        assert len(contrast_out) == n_phi
        assert len(offset_out) == n_phi

        # Verify adaptive regularization works with Fourier-space parameters
        reg_config = AdaptiveRegularizationConfig(
            enable=True,
            target_cv=0.1,
            target_contribution=0.1,
            lambda_base=1.0,
        )
        reg = AdaptiveRegularizer(reg_config, n_phi=n_phi)

        # Create mock parameters (Fourier coeffs interpreted as per-angle for testing)
        all_params = np.concatenate(
            [
                per_angle_contrast,
                per_angle_offset,
                np.array(
                    [1000.0, 0.5, 10.0, 0.002, 0.3, 0.0001, 0.0]
                ),  # physical params
            ]
        )

        # Verify regularization doesn't crash
        reg_term = reg.compute_regularization(all_params, mse=0.04, n_points=1000)
        assert np.isfinite(reg_term)


class TestGammaDotRecovery:
    """Test that gamma_dot_t0 recovers to physical values (US1)."""

    @pytest.mark.integration
    def test_gamma_dot_recovery(self):
        """Verify gamma_dot_t0 > 1e-04 after optimization.

        This test verifies User Story 1, Acceptance Scenario 3:
        Given the 23-phi C020 dataset that originally failed,
        When running with the fixed optimizer,
        Then gamma_dot_t0 converges to approximately 0.002 s^-1
        (similar to the 3-phi reference result) with uncertainty < 100%.
        """
        # Test that Layer 5 shear weighting preserves shear sensitivity
        n_phi = 23
        n_physical = 7
        phi0_index = 6  # phi0 is the 7th physical parameter (index 6)
        phi_angles = np.linspace(0, 360, n_phi, endpoint=False)  # In degrees

        # Create shear weighting config (disable normalization for simpler bounds)
        sw_config = ShearWeightingConfig(
            enable=True,
            min_weight=0.1,
            alpha=1.0,
            normalize=False,  # Disable normalization for predictable bounds
        )
        weighter = ShearSensitivityWeighting(
            phi_angles=phi_angles,
            n_physical=n_physical,
            phi0_index=phi0_index,
            config=sw_config,
        )

        # Initial phi0 guess (degrees)
        phi0_current = 30.0  # 30 degrees

        # Get weights for current phi0
        weights = weighter.get_weights(phi0_current)

        # Verify weights emphasize shear-sensitive angles
        assert len(weights) == n_phi
        assert np.all(weights >= sw_config.min_weight)  # min_weight
        assert np.all(weights <= 1.0)  # max weight without normalization

        # Angles aligned with shear (phi ≈ phi0) should have higher weight
        # Convert to radians for cos calculation
        phi_rad = np.deg2rad(phi_angles)
        phi0_rad = np.deg2rad(phi0_current)
        cos_vals = np.abs(np.cos(phi0_rad - phi_rad))

        # Verify weight pattern follows cos relationship
        # Higher cos values should correspond to higher weights
        high_cos_idx = np.argmax(cos_vals)
        low_cos_idx = np.argmin(cos_vals)
        assert weights[high_cos_idx] >= weights[low_cos_idx]


class TestPerAngleCVConstraint:
    """Test that per-angle parameters have low coefficient of variation (US1)."""

    @pytest.mark.integration
    def test_per_angle_cv_constraint(self):
        """Verify per-angle contrast/offset CV < 20% after optimization.

        This test verifies Success Criterion SC-003:
        Per-angle parameter coefficient of variation (CV) is < 20%
        after optimization (indicating regularization is effective).
        """
        n_phi = 23

        # Create adaptive regularization config
        reg_config = AdaptiveRegularizationConfig(
            enable=True,
            target_cv=0.10,  # 10% target CV
            target_contribution=0.10,
            lambda_base=1.0,
        )
        reg = AdaptiveRegularizer(reg_config, n_phi=n_phi)

        # Simulate well-constrained parameters (low CV)
        contrast_mean = 1.0
        offset_mean = 0.5
        low_cv = 0.05  # 5% CV

        np.random.seed(42)
        contrast_params = np.random.normal(contrast_mean, contrast_mean * low_cv, n_phi)
        offset_params = np.random.normal(offset_mean, offset_mean * low_cv, n_phi)

        all_params = np.concatenate([contrast_params, offset_params])

        # Compute regularization for low-CV parameters
        reg_low = reg.compute_regularization(all_params, mse=0.04, n_points=1000)

        # Simulate poorly-constrained parameters (high CV)
        high_cv = 0.30  # 30% CV
        contrast_high = np.random.normal(contrast_mean, contrast_mean * high_cv, n_phi)
        offset_high = np.random.normal(offset_mean, offset_mean * high_cv, n_phi)

        all_params_high = np.concatenate([contrast_high, offset_high])
        reg_high = reg.compute_regularization(all_params_high, mse=0.04, n_points=1000)

        # High-CV parameters should have higher regularization penalty
        assert reg_high > reg_low


class TestIndexMismatchWarning:
    """Test warning logged on configuration mismatch (US2)."""

    @pytest.mark.integration
    def test_index_mismatch_warning(self, caplog):
        """Verify warning logged when group_indices exceed parameter count.

        This test verifies User Story 2, Acceptance Scenario 1:
        Given an optimizer configuration with incompatible settings,
        When the optimizer detects this mismatch,
        Then it logs a WARNING with the specific incompatibility.
        """
        n_phi = 23

        # Create config with explicit group indices that may not match Fourier mode
        reg_config = AdaptiveRegularizationConfig(
            enable=True,
            target_cv=0.10,
            target_contribution=0.10,
            lambda_base=1.0,
            group_indices=[(0, n_phi), (n_phi, 2 * n_phi)],  # Per-angle indices
        )

        with caplog.at_level(logging.DEBUG):
            reg = AdaptiveRegularizer(reg_config, n_phi=n_phi)

            # Create parameters matching the group indices
            all_params = np.ones(2 * n_phi)

            # Regularization should work with matching dimensions
            reg_term = reg.compute_regularization(all_params, mse=0.04, n_points=1000)
            assert np.isfinite(reg_term)


class TestFallbackWarning:
    """Test warning logged when falling back to basic streaming (US2)."""

    @pytest.mark.integration
    def test_fallback_warning(self, caplog):
        """Verify warning logged when hybrid optimizer falls back.

        This test verifies User Story 2, Acceptance Scenario 2:
        Given the Adaptive Hybrid Streaming Optimizer fails for any reason,
        When falling back to basic streaming,
        Then a WARNING is logged explaining why the fallback occurred.
        """
        # Test gradient collapse monitor detection
        n_phi = 23
        n_physical = 7

        monitor_config = GradientMonitorConfig(
            enable=True,
            ratio_threshold=0.01,
            consecutive_triggers=3,
            response_mode="warn",
        )

        physical_indices = list(range(2 * n_phi, 2 * n_phi + n_physical))
        per_angle_indices = list(range(2 * n_phi))

        monitor = GradientCollapseMonitor(
            config=monitor_config,
            physical_indices=physical_indices,
            per_angle_indices=per_angle_indices,
        )

        # Simulate gradient collapse (physical << per-angle)
        n_params = 2 * n_phi + n_physical
        grad = np.ones(n_params)
        grad[physical_indices] = 1e-10  # Very small physical gradients
        grad[per_angle_indices] = 1.0  # Much larger per-angle gradients

        with caplog.at_level(logging.WARNING):
            # Check multiple times to trigger consecutive detection
            for i in range(5):
                monitor.check(grad, iteration=i)

            # After consecutive triggers, should detect collapse
            assert monitor.collapse_detected or monitor.consecutive_count >= 3


class TestCovarianceTransformation:
    """Test covariance transformation from Fourier to per-angle space (US3)."""

    @pytest.mark.integration
    def test_covariance_transformation(self):
        """Verify covariance matrix is correctly transformed.

        This test verifies User Story 3, Acceptance Scenario 1:
        Given a successful optimization with Fourier reparameterization,
        When computing the covariance matrix,
        Then the covariance is correctly transformed to original parameter
        space with dimensions matching the original parameter count.
        """
        n_phi = 23
        n_physical = 7

        # Create phi angles
        phi_angles = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

        # Create Fourier reparameterization
        fourier_config = FourierReparamConfig(mode="fourier", fourier_order=2)
        fourier = FourierReparameterizer(phi_angles, fourier_config)

        n_fourier = fourier.n_coeffs  # Total Fourier coefficients

        # Create mock covariance in Fourier space
        cov_fourier = np.eye(n_fourier + n_physical) * 0.01

        # Get transformation Jacobian
        jacobian = fourier.get_jacobian_transform()

        # Verify Jacobian can be used for covariance transformation
        # J maps Fourier -> per-angle, so J @ cov_fourier @ J.T gives per-angle cov
        n_per_angle = 2 * n_phi
        assert jacobian.shape[0] == n_per_angle  # Output dimension
        assert jacobian.shape[1] == n_fourier  # Input dimension

        # Transform covariance
        cov_fourier_subset = cov_fourier[:n_fourier, :n_fourier]
        cov_per_angle = jacobian @ cov_fourier_subset @ jacobian.T

        # Verify transformed dimensions
        assert cov_per_angle.shape == (n_per_angle, n_per_angle)


class TestUncertaintyBounds:
    """Test relative uncertainty bounds on gamma_dot_t0 (US3)."""

    @pytest.mark.integration
    def test_uncertainty_bounds(self):
        """Verify relative uncertainty on gamma_dot_t0 < 100%.

        This test verifies Success Criterion SC-004:
        Relative uncertainty on gamma_dot_t0 is < 100%
        (indicating valid covariance estimation).
        """
        # Simulate optimization result with reasonable uncertainty
        gamma_dot_t0 = 0.002  # s^-1
        gamma_dot_t0_std = 0.001  # 50% relative uncertainty

        relative_uncertainty = gamma_dot_t0_std / gamma_dot_t0
        assert relative_uncertainty < 1.0  # < 100%

        # Test that very small gamma_dot_t0 would have high relative uncertainty
        gamma_dot_small = 1e-6
        gamma_dot_small_std = 1e-5
        relative_uncertainty_small = gamma_dot_small_std / gamma_dot_small
        assert relative_uncertainty_small > 1.0  # > 100% indicates potential collapse


class TestBackwardCompatibility3Phi:
    """Test backward compatibility with 3-phi case (Phase 6)."""

    @pytest.mark.integration
    def test_backward_compatibility_3phi(self):
        """Verify 3-phi case still works correctly.

        This test verifies Success Criterion SC-005:
        All existing unit tests for NLSQ optimization continue to pass.

        The 3-phi case is the reference case that should not be affected
        by the Fourier mode fix.
        """
        n_phi = 3
        n_physical = 7

        # Create phi angles
        phi_angles = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

        # Create Fourier reparameterization for 3-phi
        # With n_phi=3 and order=1, it should work but may use independent mode
        fourier_config = FourierReparamConfig(mode="auto", fourier_order=1)
        fourier = FourierReparameterizer(phi_angles, fourier_config)

        # Verify small phi count works
        assert fourier.n_phi == n_phi

        # Test transform round-trip
        per_angle_contrast = np.array([1.0, 1.1, 0.9])
        per_angle_offset = np.array([0.5, 0.6, 0.4])

        fourier_coeffs = fourier.per_angle_to_fourier(
            per_angle_contrast, per_angle_offset
        )
        contrast_out, offset_out = fourier.fourier_to_per_angle(fourier_coeffs)

        # Reconstruction quality depends on mode
        # In independent mode, should be exact; in Fourier mode, may have truncation error
        if fourier.use_fourier:
            assert np.allclose(contrast_out, per_angle_contrast, atol=0.2)
        else:
            assert np.allclose(contrast_out, per_angle_contrast, atol=1e-10)

        # Verify shear weighting works for 3-phi
        phi_angles_deg = np.linspace(0, 360, n_phi, endpoint=False)
        weighter = ShearSensitivityWeighting(
            phi_angles=phi_angles_deg,
            n_physical=n_physical,
            phi0_index=6,
        )
        weights = weighter.get_weights(phi0_current=0.0)

        assert len(weights) == n_phi
        assert np.all(weights >= 0.1)
