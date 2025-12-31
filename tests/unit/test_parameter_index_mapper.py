"""Unit tests for ParameterIndexMapper.

Tests the centralized index mapping for anti-degeneracy layers,
verifying correct behavior in both Fourier and non-Fourier modes.

Feature: 001-fix-nlsq-anti-degeneracy
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest

from homodyne.optimization.nlsq.parameter_index_mapper import ParameterIndexMapper


# Mock FourierReparameterizer for testing
@dataclass
class MockFourierReparameterizer:
    """Mock FourierReparameterizer for unit testing."""

    n_coeffs_per_param: int = 5  # order=2 -> 1 + 2*2 = 5
    n_coeffs: int = 10  # 2 groups * 5 coeffs
    use_fourier: bool = True


class TestParameterIndexMapperNonFourier:
    """Tests for ParameterIndexMapper in non-Fourier mode."""

    def test_init_valid_inputs(self):
        """Test mapper initializes with valid inputs."""
        mapper = ParameterIndexMapper(n_phi=23, n_physical=7, fourier=None)
        assert mapper.n_phi == 23
        assert mapper.n_physical == 7
        assert mapper.fourier is None

    def test_init_invalid_n_phi(self):
        """Test mapper rejects n_phi < 1."""
        with pytest.raises(ValueError, match="n_phi must be >= 1"):
            ParameterIndexMapper(n_phi=0, n_physical=7, fourier=None)

    def test_init_invalid_n_physical(self):
        """Test mapper rejects n_physical < 1."""
        with pytest.raises(ValueError, match="n_physical must be >= 1"):
            ParameterIndexMapper(n_phi=23, n_physical=0, fourier=None)

    def test_use_fourier_false_when_no_fourier(self):
        """Test use_fourier is False when fourier is None."""
        mapper = ParameterIndexMapper(n_phi=23, n_physical=7, fourier=None)
        assert mapper.use_fourier is False

    def test_n_per_group_equals_n_phi(self):
        """Test n_per_group equals n_phi in non-Fourier mode."""
        mapper = ParameterIndexMapper(n_phi=23, n_physical=7, fourier=None)
        assert mapper.n_per_group == 23

    def test_n_per_angle_total_equals_2n_phi(self):
        """Test n_per_angle_total equals 2*n_phi in non-Fourier mode."""
        mapper = ParameterIndexMapper(n_phi=23, n_physical=7, fourier=None)
        assert mapper.n_per_angle_total == 46

    def test_total_params_correct(self):
        """Test total_params is sum of per-angle and physical."""
        mapper = ParameterIndexMapper(n_phi=23, n_physical=7, fourier=None)
        assert mapper.total_params == 53  # 46 + 7

    def test_get_group_indices_non_fourier(self):
        """Test group indices in non-Fourier mode (23 phi angles)."""
        mapper = ParameterIndexMapper(n_phi=23, n_physical=7, fourier=None)
        indices = mapper.get_group_indices()
        assert indices == [(0, 23), (23, 46)]

    def test_get_group_indices_3phi(self):
        """Test group indices for 3-phi case (reference case)."""
        mapper = ParameterIndexMapper(n_phi=3, n_physical=7, fourier=None)
        indices = mapper.get_group_indices()
        assert indices == [(0, 3), (3, 6)]

    def test_get_physical_indices(self):
        """Test physical indices start after per-angle params."""
        mapper = ParameterIndexMapper(n_phi=23, n_physical=7, fourier=None)
        physical = mapper.get_physical_indices()
        assert physical == [46, 47, 48, 49, 50, 51, 52]

    def test_get_per_angle_indices(self):
        """Test per-angle indices cover contrast + offset."""
        mapper = ParameterIndexMapper(n_phi=3, n_physical=7, fourier=None)
        per_angle = mapper.get_per_angle_indices()
        assert per_angle == [0, 1, 2, 3, 4, 5]


class TestParameterIndexMapperFourier:
    """Tests for ParameterIndexMapper in Fourier mode."""

    def test_use_fourier_true_when_fourier_active(self):
        """Test use_fourier is True when Fourier reparameterizer is active."""
        fourier = MockFourierReparameterizer(use_fourier=True)
        mapper = ParameterIndexMapper(n_phi=23, n_physical=7, fourier=fourier)
        assert mapper.use_fourier is True

    def test_use_fourier_false_when_fourier_disabled(self):
        """Test use_fourier is False when Fourier reparameterizer is disabled."""
        fourier = MockFourierReparameterizer(use_fourier=False)
        mapper = ParameterIndexMapper(n_phi=23, n_physical=7, fourier=fourier)
        assert mapper.use_fourier is False

    def test_n_per_group_uses_fourier_coeffs(self):
        """Test n_per_group uses Fourier coefficient count."""
        fourier = MockFourierReparameterizer(n_coeffs_per_param=5, use_fourier=True)
        mapper = ParameterIndexMapper(n_phi=23, n_physical=7, fourier=fourier)
        assert mapper.n_per_group == 5

    def test_n_per_angle_total_uses_fourier_coeffs(self):
        """Test n_per_angle_total uses total Fourier coefficients."""
        fourier = MockFourierReparameterizer(n_coeffs=10, use_fourier=True)
        mapper = ParameterIndexMapper(n_phi=23, n_physical=7, fourier=fourier)
        assert mapper.n_per_angle_total == 10

    def test_total_params_with_fourier(self):
        """Test total_params with Fourier mode active."""
        fourier = MockFourierReparameterizer(n_coeffs=10, use_fourier=True)
        mapper = ParameterIndexMapper(n_phi=23, n_physical=7, fourier=fourier)
        assert mapper.total_params == 17  # 10 + 7

    def test_get_group_indices_fourier_mode(self):
        """Test group indices in Fourier mode.

        With order=2, n_coeffs_per_param = 5, so:
        - Contrast: [0, 5)
        - Offset: [5, 10)
        """
        fourier = MockFourierReparameterizer(
            n_coeffs_per_param=5, n_coeffs=10, use_fourier=True
        )
        mapper = ParameterIndexMapper(n_phi=23, n_physical=7, fourier=fourier)
        indices = mapper.get_group_indices()
        assert indices == [(0, 5), (5, 10)]

    def test_get_physical_indices_fourier_mode(self):
        """Test physical indices in Fourier mode."""
        fourier = MockFourierReparameterizer(n_coeffs=10, use_fourier=True)
        mapper = ParameterIndexMapper(n_phi=23, n_physical=7, fourier=fourier)
        physical = mapper.get_physical_indices()
        assert physical == [10, 11, 12, 13, 14, 15, 16]


class TestParameterIndexMapperValidation:
    """Tests for ParameterIndexMapper.validate_indices()."""

    def test_validate_indices_success(self):
        """Test validation passes with correct parameter count."""
        mapper = ParameterIndexMapper(n_phi=3, n_physical=7, fourier=None)
        params = np.zeros(13)  # 6 per-angle + 7 physical
        assert mapper.validate_indices(params) is True

    def test_validate_indices_too_few_params(self):
        """Test validation fails when params are too short."""
        mapper = ParameterIndexMapper(n_phi=23, n_physical=7, fourier=None)
        params = np.zeros(17)  # Should be 53 in non-Fourier mode

        with pytest.raises(ValueError, match="exceeds parameter count"):
            mapper.validate_indices(params)

    def test_validate_indices_with_fourier(self):
        """Test validation passes in Fourier mode with correct count."""
        fourier = MockFourierReparameterizer(n_coeffs=10, use_fourier=True)
        mapper = ParameterIndexMapper(n_phi=23, n_physical=7, fourier=fourier)
        params = np.zeros(17)  # 10 Fourier + 7 physical
        assert mapper.validate_indices(params) is True

    def test_validate_indices_fourier_mismatch(self):
        """Test validation catches Fourier/regularization mode mismatch.

        This is the exact bug this fix addresses: when Fourier mode
        creates 17 params but non-Fourier indices expect 53.
        """
        # Simulate the bug: mapper thinks non-Fourier, but params are Fourier-sized
        mapper = ParameterIndexMapper(n_phi=23, n_physical=7, fourier=None)
        params = np.zeros(17)  # Fourier-sized params

        with pytest.raises(ValueError, match="Fourier/regularization mode mismatch"):
            mapper.validate_indices(params)


class TestParameterIndexMapperDiagnostics:
    """Tests for ParameterIndexMapper.get_diagnostics()."""

    def test_diagnostics_non_fourier(self):
        """Test diagnostics output in non-Fourier mode."""
        mapper = ParameterIndexMapper(n_phi=3, n_physical=7, fourier=None)
        diag = mapper.get_diagnostics()

        assert diag["use_fourier"] is False
        assert diag["n_phi"] == 3
        assert diag["n_physical"] == 7
        assert diag["n_per_group"] == 3
        assert diag["n_per_angle_total"] == 6
        assert diag["total_params"] == 13
        assert diag["group_indices"] == [(0, 3), (3, 6)]
        assert diag["physical_indices"] == [6, 7, 8, 9, 10, 11, 12]

    def test_diagnostics_fourier(self):
        """Test diagnostics output in Fourier mode."""
        fourier = MockFourierReparameterizer(
            n_coeffs_per_param=5, n_coeffs=10, use_fourier=True
        )
        mapper = ParameterIndexMapper(n_phi=23, n_physical=7, fourier=fourier)
        diag = mapper.get_diagnostics()

        assert diag["use_fourier"] is True
        assert diag["n_phi"] == 23
        assert diag["n_physical"] == 7
        assert diag["n_per_group"] == 5
        assert diag["n_per_angle_total"] == 10
        assert diag["total_params"] == 17
        assert diag["group_indices"] == [(0, 5), (5, 10)]
        assert diag["physical_indices"] == [10, 11, 12, 13, 14, 15, 16]


class TestParameterIndexMapperCovarianceSlice:
    """Tests for covariance slice index methods."""

    def test_get_covariance_slice_indices_non_fourier(self):
        """Test covariance slice indices in non-Fourier mode."""
        mapper = ParameterIndexMapper(n_phi=3, n_physical=7, fourier=None)
        per_angle_slice, physical_slice = mapper.get_covariance_slice_indices()

        assert per_angle_slice == slice(0, 6)
        assert physical_slice == slice(6, 13)

    def test_get_covariance_slice_indices_fourier(self):
        """Test covariance slice indices in Fourier mode."""
        fourier = MockFourierReparameterizer(n_coeffs=10, use_fourier=True)
        mapper = ParameterIndexMapper(n_phi=23, n_physical=7, fourier=fourier)
        per_angle_slice, physical_slice = mapper.get_covariance_slice_indices()

        assert per_angle_slice == slice(0, 10)
        assert physical_slice == slice(10, 17)


class TestParameterIndexMapperNonFourierModeUnchanged:
    """Tests ensuring non-Fourier mode behavior is unchanged (backward compat)."""

    def test_3phi_indices_unchanged(self):
        """Verify 3-phi case produces expected indices (reference case)."""
        mapper = ParameterIndexMapper(n_phi=3, n_physical=7, fourier=None)

        # From data-model.md: n_phi=3, non-Fourier
        # n_per_angle_total = 2 * 3 = 6
        # group_indices = [(0, 3), (3, 6)]
        # physical_indices = [6, 7, 8, 9, 10, 11, 12]
        # total_params = 6 + 7 = 13

        assert mapper.n_per_angle_total == 6
        assert mapper.get_group_indices() == [(0, 3), (3, 6)]
        assert mapper.get_physical_indices() == [6, 7, 8, 9, 10, 11, 12]
        assert mapper.total_params == 13

    def test_23phi_indices_unchanged(self):
        """Verify 23-phi case produces expected indices without Fourier."""
        mapper = ParameterIndexMapper(n_phi=23, n_physical=7, fourier=None)

        # Non-Fourier mode with 23 angles
        # n_per_angle_total = 2 * 23 = 46
        # group_indices = [(0, 23), (23, 46)]
        # total_params = 46 + 7 = 53

        assert mapper.n_per_angle_total == 46
        assert mapper.get_group_indices() == [(0, 23), (23, 46)]
        assert mapper.total_params == 53
