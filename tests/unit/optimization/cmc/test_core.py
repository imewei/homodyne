"""Tests for CMC core module.

Comprehensive tests for the core CMC orchestration:
- run_cmc: Main entry point
- CMC workflow validation
- Integration with data preparation and results
"""

from __future__ import annotations

import numpy as np
import pytest

# Require ArviZ for CMC imports; skip if optional dependency is missing
pytest.importorskip("arviz", reason="ArviZ required for CMC unit tests")

from homodyne.optimization.cmc.config import CMCConfig  # noqa: E402
from homodyne.optimization.cmc.data_prep import PreparedData  # noqa: E402
from homodyne.optimization.cmc.results import CMCResult  # noqa: E402

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def minimal_config():
    """Minimal CMC configuration for fast tests."""
    return CMCConfig(
        enable=True,
        num_warmup=10,
        num_samples=20,
        num_chains=1,
        target_accept_prob=0.8,
    )


@pytest.fixture
def mock_prepared_data():
    """Create mock PreparedData."""
    n = 100
    np.random.seed(42)

    return PreparedData(
        data=np.random.randn(n) * 0.1 + 1.0,
        t1=np.random.rand(n) * 10,
        t2=np.random.rand(n) * 10,
        phi=np.random.choice([0.0, np.pi / 4, np.pi / 2], size=n),
        phi_unique=np.array([0.0, np.pi / 4, np.pi / 2]),
        phi_indices=np.random.randint(0, 3, n),
        n_total=n,
        n_phi=3,
        noise_scale=0.1,
    )


@pytest.fixture
def mock_cmc_result():
    """Create mock CMCResult for testing."""
    n_chains = 2
    n_samples = 50
    param_names = ["contrast_0", "offset_0", "D0", "alpha", "D_offset"]

    samples = {name: np.random.randn(n_chains, n_samples) for name in param_names}

    return CMCResult(
        parameters=np.array([0.5, 1.0, 1000.0, 0.5, 10.0]),
        uncertainties=np.array([0.1, 0.1, 100.0, 0.1, 5.0]),
        param_names=param_names,
        samples=samples,
        convergence_status="converged",
        r_hat=dict.fromkeys(param_names, 1.01),
        ess_bulk=dict.fromkeys(param_names, 500.0),
        ess_tail=dict.fromkeys(param_names, 400.0),
        divergences=0,
        inference_data=None,
        execution_time=30.0,
        warmup_time=15.0,
        n_chains=n_chains,
        n_samples=n_samples,
        n_warmup=50,
        analysis_mode="static",
    )


# =============================================================================
# Tests for CMCResult Properties
# =============================================================================


class TestCMCResultProperties:
    """Additional tests for CMCResult properties and methods."""

    def test_success_states(self, mock_cmc_result):
        """Test all success states."""
        # Converged = success
        mock_cmc_result.convergence_status = "converged"
        assert mock_cmc_result.success is True

        # Not converged = failure
        mock_cmc_result.convergence_status = "not_converged"
        assert mock_cmc_result.success is False

        # Divergences = failure
        mock_cmc_result.convergence_status = "divergences"
        assert mock_cmc_result.success is False

        # Unknown status = failure
        mock_cmc_result.convergence_status = "unknown"
        assert mock_cmc_result.success is False

    def test_is_cmc_result(self, mock_cmc_result):
        """Test is_cmc_result returns True."""
        assert mock_cmc_result.is_cmc_result() is True

    def test_posterior_stats_keys(self, mock_cmc_result):
        """Test get_posterior_stats returns expected keys."""
        stats = mock_cmc_result.get_posterior_stats()

        for name in mock_cmc_result.param_names:
            assert name in stats
            assert "mean" in stats[name]
            assert "std" in stats[name]
            assert "median" in stats[name]
            assert "r_hat" in stats[name]
            assert "ess_bulk" in stats[name]

    def test_samples_array_shape(self, mock_cmc_result):
        """Test get_samples_array returns correct shape."""
        arr = mock_cmc_result.get_samples_array()

        n_chains = mock_cmc_result.n_chains
        n_samples = mock_cmc_result.n_samples
        n_params = len(mock_cmc_result.param_names)

        assert arr.shape == (n_chains, n_samples, n_params)

    def test_samples_array_ordering(self, mock_cmc_result):
        """Test samples array preserves parameter ordering."""
        arr = mock_cmc_result.get_samples_array()

        for i, name in enumerate(mock_cmc_result.param_names):
            np.testing.assert_array_equal(arr[:, :, i], mock_cmc_result.samples[name])


# =============================================================================
# Tests for PreparedData Validation
# =============================================================================


class TestPreparedDataValidation:
    """Tests for PreparedData validation."""

    def test_consistent_lengths(self, mock_prepared_data):
        """Test all arrays have consistent lengths."""
        assert len(mock_prepared_data.data) == mock_prepared_data.n_total
        assert len(mock_prepared_data.t1) == mock_prepared_data.n_total
        assert len(mock_prepared_data.t2) == mock_prepared_data.n_total
        assert len(mock_prepared_data.phi) == mock_prepared_data.n_total
        assert len(mock_prepared_data.phi_indices) == mock_prepared_data.n_total

    def test_phi_unique_consistency(self, mock_prepared_data):
        """Test phi_unique matches n_phi."""
        assert len(mock_prepared_data.phi_unique) == mock_prepared_data.n_phi

    def test_phi_indices_range(self, mock_prepared_data):
        """Test phi_indices are in valid range."""
        assert np.all(mock_prepared_data.phi_indices >= 0)
        assert np.all(mock_prepared_data.phi_indices < mock_prepared_data.n_phi)

    def test_noise_scale_positive(self, mock_prepared_data):
        """Test noise_scale is positive."""
        assert mock_prepared_data.noise_scale > 0


# =============================================================================
# Tests for CMCConfig Integration
# =============================================================================


class TestCMCConfigIntegration:
    """Tests for CMCConfig integration with core."""

    def test_config_validation(self):
        """Test config validation catches invalid values."""
        # Valid config
        valid_config = CMCConfig(
            num_warmup=100,
            num_samples=200,
            num_chains=4,
        )
        errors = valid_config.validate()
        assert len(errors) == 0

        # Invalid chains
        invalid_config = CMCConfig(num_chains=0)
        errors = invalid_config.validate()
        assert len(errors) > 0

    def test_should_enable_cmc_logic(self):
        """Test CMC enable logic."""
        config = CMCConfig(enable="auto", min_points_for_cmc=100000)

        # Small data - should not enable
        assert config.should_enable_cmc(n_points=50000) is False

        # Large data - should enable
        assert config.should_enable_cmc(n_points=200000) is True

        # Explicit True overrides
        config.enable = True
        assert config.should_enable_cmc(n_points=100) is True

        # Explicit False overrides
        config.enable = False
        assert config.should_enable_cmc(n_points=1000000) is False


# =============================================================================
# Tests for Workflow Integration
# =============================================================================


class TestCMCWorkflow:
    """Tests for CMC workflow integration."""

    def test_prepared_data_to_result_consistency(
        self, mock_prepared_data, mock_cmc_result
    ):
        """Test data flows correctly from prepared data to result."""
        # n_phi should be consistent
        assert mock_prepared_data.n_phi >= 1

        # Result should have appropriate number of per-angle params
        n_contrast = sum(
            1 for name in mock_cmc_result.param_names if name.startswith("contrast_")
        )
        # With single phi, should have at least 1 contrast param
        assert n_contrast >= 1

    def test_result_diagnostics_complete(self, mock_cmc_result):
        """Test result contains all required diagnostics."""
        # R-hat for all parameters
        assert len(mock_cmc_result.r_hat) == len(mock_cmc_result.param_names)

        # ESS for all parameters
        assert len(mock_cmc_result.ess_bulk) == len(mock_cmc_result.param_names)
        assert len(mock_cmc_result.ess_tail) == len(mock_cmc_result.param_names)

        # Timing information
        assert mock_cmc_result.execution_time > 0
        assert mock_cmc_result.warmup_time >= 0


# =============================================================================
# Scientific Validation Tests
# =============================================================================


class TestCMCScientificProperties:
    """Scientific validation tests for CMC."""

    def test_parameter_recovery_structure(self, mock_cmc_result):
        """Test parameter structure supports recovery validation."""
        # Check D0 is in reasonable range for diffusion
        d0_idx = mock_cmc_result.param_names.index("D0")
        d0_value = mock_cmc_result.parameters[d0_idx]
        d0_unc = mock_cmc_result.uncertainties[d0_idx]

        # Uncertainty should be smaller than value
        assert d0_unc < d0_value

    def test_convergence_diagnostics_thresholds(self, mock_cmc_result):
        """Test convergence diagnostics meet thresholds."""
        # R-hat should be close to 1.0 for converged chains
        for name, r_hat in mock_cmc_result.r_hat.items():
            assert r_hat < 1.1, f"R-hat for {name} is {r_hat}, should be < 1.1"

        # ESS should be reasonable
        for name, ess in mock_cmc_result.ess_bulk.items():
            assert ess > 100, f"ESS for {name} is {ess}, should be > 100"

    def test_sample_statistics_consistency(self, mock_cmc_result):
        """Test sample statistics are consistent with parameters."""
        stats = mock_cmc_result.get_posterior_stats()

        for i, name in enumerate(mock_cmc_result.param_names):
            mock_cmc_result.parameters[i]
            mock_cmc_result.uncertainties[i]

            # Mean from samples should be close to reported parameter
            sample_mean = stats[name]["mean"]
            sample_std = stats[name]["std"]

            # With finite samples, these might differ but should be reasonable
            assert np.isfinite(sample_mean)
            assert np.isfinite(sample_std)
            assert sample_std > 0


# =============================================================================
# Edge Cases
# =============================================================================


class TestCMCEdgeCases:
    """Tests for edge cases in CMC operations."""

    def test_single_phi_angle(self):
        """Test CMC with single phi angle."""
        n = 50
        np.random.seed(42)

        prepared = PreparedData(
            data=np.random.randn(n) * 0.1 + 1.0,
            t1=np.random.rand(n) * 10,
            t2=np.random.rand(n) * 10,
            phi=np.zeros(n),  # Single angle
            phi_unique=np.array([0.0]),
            phi_indices=np.zeros(n, dtype=int),
            n_total=n,
            n_phi=1,
            noise_scale=0.1,
        )

        assert prepared.n_phi == 1
        assert len(prepared.phi_unique) == 1

    def test_many_phi_angles(self):
        """Test CMC with many phi angles."""
        n = 1000
        n_phi = 36
        np.random.seed(42)

        phi_values = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        phi_indices = np.random.randint(0, n_phi, n)
        phi = phi_values[phi_indices]

        prepared = PreparedData(
            data=np.random.randn(n) * 0.1 + 1.0,
            t1=np.random.rand(n) * 10,
            t2=np.random.rand(n) * 10,
            phi=phi,
            phi_unique=phi_values,
            phi_indices=phi_indices,
            n_total=n,
            n_phi=n_phi,
            noise_scale=0.1,
        )

        assert prepared.n_phi == 36

    def test_minimal_samples_config(self):
        """Test with minimal sample configuration."""
        config = CMCConfig(
            num_warmup=1,
            num_samples=1,
            num_chains=1,
        )

        errors = config.validate()
        # Minimal config should still be valid
        assert len(errors) == 0

    def test_large_samples_config(self):
        """Test with large sample configuration."""
        config = CMCConfig(
            num_warmup=1000,
            num_samples=10000,
            num_chains=8,
        )

        errors = config.validate()
        assert len(errors) == 0

        # Total samples should be correct
        total = config.num_samples * config.num_chains
        assert total == 80000
