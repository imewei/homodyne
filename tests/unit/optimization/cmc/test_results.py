"""Tests for CMC results module."""

import numpy as np
import pytest

pytest.importorskip("arviz", reason="ArviZ required for CMC unit tests")

from homodyne.optimization.cmc.results import CMCResult  # noqa: E402


class TestCMCResult:
    """Tests for CMCResult dataclass."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample CMCResult for testing."""
        n_chains = 4
        n_samples = 100
        param_names = ["contrast_0", "offset_0", "D0", "alpha", "D_offset"]

        # Create sample arrays
        samples = {name: np.random.randn(n_chains, n_samples) for name in param_names}

        return CMCResult(
            parameters=np.array([0.5, 1.0, 1000.0, 0.5, 0.0]),
            uncertainties=np.array([0.1, 0.1, 100.0, 0.1, 10.0]),
            param_names=param_names,
            samples=samples,
            convergence_status="converged",
            r_hat=dict.fromkeys(param_names, 1.001),
            ess_bulk=dict.fromkeys(param_names, 500.0),
            ess_tail=dict.fromkeys(param_names, 400.0),
            divergences=0,
            inference_data=None,  # Would be ArviZ object
            execution_time=60.0,
            warmup_time=30.0,
            n_chains=n_chains,
            n_samples=n_samples,
            n_warmup=100,
            analysis_mode="static",
        )

    def test_is_cmc_result(self, sample_result):
        """Test is_cmc_result returns True."""
        assert sample_result.is_cmc_result() is True

    def test_success_when_converged(self, sample_result):
        """Test success property returns True when converged."""
        assert sample_result.success is True

    def test_success_when_not_converged(self, sample_result):
        """Test success property returns False when not converged."""
        sample_result.convergence_status = "not_converged"
        assert sample_result.success is False

    def test_success_with_divergences(self, sample_result):
        """Test success property returns False with divergences."""
        sample_result.convergence_status = "divergences"
        assert sample_result.success is False

    def test_param_names_immutable(self, sample_result):
        """Test param_names is a list."""
        assert isinstance(sample_result.param_names, list)
        assert len(sample_result.param_names) == 5

    def test_samples_dict_structure(self, sample_result):
        """Test samples dict has correct structure."""
        for name in sample_result.param_names:
            assert name in sample_result.samples
            assert sample_result.samples[name].shape == (4, 100)

    def test_r_hat_values(self, sample_result):
        """Test r_hat contains values for all parameters."""
        for name in sample_result.param_names:
            assert name in sample_result.r_hat
            assert sample_result.r_hat[name] < 1.1

    def test_ess_values(self, sample_result):
        """Test ESS contains values for all parameters."""
        for name in sample_result.param_names:
            assert name in sample_result.ess_bulk
            assert sample_result.ess_bulk[name] > 0

    def test_timing_values(self, sample_result):
        """Test timing values are positive."""
        assert sample_result.execution_time > 0
        assert sample_result.warmup_time > 0
        assert sample_result.warmup_time < sample_result.execution_time


class TestCMCResultMethods:
    """Tests for CMCResult methods."""

    @pytest.fixture
    def result_with_samples(self):
        """Create result with real sample structure."""
        n_chains = 2
        n_samples = 50
        param_names = ["contrast_0", "D0", "alpha"]

        samples = {
            "contrast_0": np.random.uniform(0.3, 0.7, (n_chains, n_samples)),
            "D0": np.random.uniform(800, 1200, (n_chains, n_samples)),
            "alpha": np.random.uniform(-0.5, 0.5, (n_chains, n_samples)),
        }

        return CMCResult(
            parameters=np.array([0.5, 1000.0, 0.0]),
            uncertainties=np.array([0.1, 100.0, 0.2]),
            param_names=param_names,
            samples=samples,
            convergence_status="converged",
            r_hat={"contrast_0": 1.01, "D0": 1.02, "alpha": 1.01},
            ess_bulk={"contrast_0": 100.0, "D0": 100.0, "alpha": 100.0},
            ess_tail={"contrast_0": 80.0, "D0": 80.0, "alpha": 80.0},
            divergences=0,
            inference_data=None,
            execution_time=30.0,
            warmup_time=15.0,
            n_chains=n_chains,
            n_samples=n_samples,
            n_warmup=50,
            analysis_mode="static",
        )

    def test_get_posterior_stats(self, result_with_samples):
        """Test get_posterior_stats returns statistics."""
        stats = result_with_samples.get_posterior_stats()

        assert isinstance(stats, dict)
        for name in result_with_samples.param_names:
            assert name in stats
            assert "mean" in stats[name]
            assert "std" in stats[name]
            assert "median" in stats[name]

    def test_get_samples_array_shape(self, result_with_samples):
        """Test get_samples_array returns correct shape."""
        arr = result_with_samples.get_samples_array()

        # Shape should be (n_chains, n_samples, n_params)
        assert arr.shape == (2, 50, 3)

    def test_get_samples_array_order(self, result_with_samples):
        """Test get_samples_array preserves parameter order."""
        arr = result_with_samples.get_samples_array()

        # First param (contrast_0) should match
        np.testing.assert_array_almost_equal(
            arr[:, :, 0], result_with_samples.samples["contrast_0"]
        )

        # Second param (D0) should match
        np.testing.assert_array_almost_equal(
            arr[:, :, 1], result_with_samples.samples["D0"]
        )


class TestCMCResultValidation:
    """Tests for CMCResult.validate_parameters method."""

    @pytest.fixture
    def complete_static_result(self):
        """Create a complete static result for validation testing."""
        n_chains = 2
        n_samples = 100
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

    def test_validate_complete_static_result(self, complete_static_result):
        """Test validation passes for complete static result."""
        warnings = complete_static_result.validate_parameters(n_phi=1)
        assert len(warnings) == 0

    def test_validate_missing_physical_parameter(self, complete_static_result):
        """Test validation catches missing physical parameter."""
        del complete_static_result.samples["D0"]
        warnings = complete_static_result.validate_parameters()
        assert any("Missing required parameter: D0" in w for w in warnings)

    def test_validate_non_finite_values(self, complete_static_result):
        """Test validation catches non-finite values."""
        complete_static_result.samples["D0"][0, 0] = np.nan
        warnings = complete_static_result.validate_parameters()
        assert any("Non-finite values" in w for w in warnings)

    def test_validate_high_r_hat(self, complete_static_result):
        """Test validation warns about high R-hat."""
        complete_static_result.r_hat["D0"] = 1.5
        warnings = complete_static_result.validate_parameters()
        assert any("High R-hat" in w for w in warnings)

    def test_validate_low_ess(self, complete_static_result):
        """Test validation warns about low ESS."""
        complete_static_result.ess_bulk = dict.fromkeys(
            complete_static_result.param_names, 50.0
        )
        warnings = complete_static_result.validate_parameters()
        assert any("Low ESS" in w for w in warnings)

    def test_validate_high_divergence_rate(self, complete_static_result):
        """Test validation warns about high divergence rate."""
        complete_static_result.divergences = 50  # 25% rate with 200 samples
        warnings = complete_static_result.validate_parameters()
        assert any("divergence rate" in w for w in warnings)

    def test_validate_infers_n_phi_from_samples(self, complete_static_result):
        """Test validation can infer n_phi from sample names."""
        # Already has contrast_0 and offset_0
        warnings = complete_static_result.validate_parameters(n_phi=None)
        # Should pass since contrast_0 and offset_0 exist
        assert not any("Missing per-angle" in w for w in warnings)

    def test_validate_missing_per_angle_parameter(self, complete_static_result):
        """Test validation catches missing per-angle parameters."""
        del complete_static_result.samples["contrast_0"]
        warnings = complete_static_result.validate_parameters(n_phi=1)
        assert any("Missing per-angle parameter: contrast_0" in w for w in warnings)
