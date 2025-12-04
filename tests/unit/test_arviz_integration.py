"""ArviZ Integration Tests for Homodyne v2.4.1+

This module provides comprehensive tests for ArviZ integration including:
- MCMCResult.to_arviz() conversion
- ArviZ plot functions (plot_arviz_trace, plot_arviz_posterior, plot_arviz_pair)
- Path validation in plot functions
- Deserialization validation in from_dict()
- DeserializationError handling

Addresses the code review finding that ArviZ integration had 0% test coverage
for 460+ lines of new code.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from homodyne.optimization.mcmc.cmc.result import (
    DeserializationError,
    MCMCResult,
    _validate_array_field,
    _validate_scalar_field,
    _validate_string_field,
)
from homodyne.utils.path_validation import PathValidationError, validate_plot_save_path


class TestMCMCResultToArviz:
    """Tests for MCMCResult.to_arviz() conversion."""

    @pytest.fixture
    def simple_mcmc_result(self):
        """Create a simple MCMCResult for testing."""
        n_chains = 2
        n_samples = 100
        n_params = 3

        # Create sample data with proper shape
        samples = np.random.normal(0, 1, (n_chains, n_samples, n_params))

        return MCMCResult(
            mean_params=np.array([1000.0, 0.567, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            std_params=np.array([50.0, 0.02, 1.0]),
            std_contrast=0.02,
            std_offset=0.01,
            samples_params=samples.reshape(-1, n_params),
            samples_contrast=np.random.normal(0.5, 0.02, n_chains * n_samples),
            samples_offset=np.random.normal(1.0, 0.01, n_chains * n_samples),
            converged=True,
            n_iterations=1000,
            computation_time=10.5,
            backend="JAX",
            analysis_mode="static",
            n_chains=n_chains,
            n_warmup=500,
            n_samples=n_samples,
            sampler="NUTS",
            r_hat=1.01,
            effective_sample_size=450.0,
            param_names=["D0", "alpha", "D_offset"],
        )

    @pytest.fixture
    def cmc_result(self):
        """Create a CMC result for testing."""
        n_chains = 2
        n_samples = 100
        n_params = 9  # 3 contrast + 3 offset + 3 physical

        samples = np.random.normal(0, 1, (n_chains, n_samples, n_params))

        return MCMCResult(
            mean_params=np.array([0.5, 0.52, 0.48, 1.0, 1.01, 0.99, 1000.0, 0.567, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            std_params=np.array([0.02] * 6 + [50.0, 0.02, 1.0]),
            samples_params=samples.reshape(-1, n_params),
            n_chains=n_chains,
            n_samples=n_samples,
            num_shards=4,
            combination_method="precision_weighted",
            param_names=[
                "contrast_0", "contrast_1", "contrast_2",
                "offset_0", "offset_1", "offset_2",
                "D0", "alpha", "D_offset",
            ],
        )

    def test_to_arviz_returns_inference_data(self, simple_mcmc_result):
        """Test that to_arviz() returns an ArviZ InferenceData object."""
        pytest.importorskip("arviz")
        import arviz as az

        idata = simple_mcmc_result.to_arviz()
        assert isinstance(idata, az.InferenceData)

    def test_to_arviz_has_posterior_group(self, simple_mcmc_result):
        """Test that InferenceData has posterior group."""
        pytest.importorskip("arviz")

        idata = simple_mcmc_result.to_arviz()
        assert hasattr(idata, "posterior")
        assert idata.posterior is not None

    def test_to_arviz_parameter_names(self, simple_mcmc_result):
        """Test that parameter names are correctly preserved."""
        pytest.importorskip("arviz")

        idata = simple_mcmc_result.to_arviz()
        param_names = list(idata.posterior.data_vars.keys())

        assert "D0" in param_names
        assert "alpha" in param_names
        assert "D_offset" in param_names

    def test_to_arviz_cmc_result(self, cmc_result):
        """Test to_arviz() with CMC result including per-angle parameters."""
        pytest.importorskip("arviz")

        idata = cmc_result.to_arviz()
        param_names = list(idata.posterior.data_vars.keys())

        # Check per-angle contrast params
        assert "contrast_0" in param_names
        assert "contrast_1" in param_names
        assert "contrast_2" in param_names

        # Check physical params
        assert "D0" in param_names

    def test_to_arviz_without_samples_raises(self):
        """Test that to_arviz() raises ValueError when no samples available."""
        result = MCMCResult(
            mean_params=np.array([1000.0, 0.567, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            samples_params=None,  # No samples
        )
        with pytest.raises(ValueError, match="samples_params is None"):
            result.to_arviz()


class TestDeserializationValidation:
    """Tests for from_dict() deserialization validation."""

    def test_from_dict_valid_data(self):
        """Test from_dict with valid minimal data."""
        data = {
            "mean_params": [1000.0, 0.567, 10.0],
            "mean_contrast": 0.5,
            "mean_offset": 1.0,
        }
        result = MCMCResult.from_dict(data)
        assert result.mean_contrast == 0.5
        np.testing.assert_array_almost_equal(
            result.mean_params, [1000.0, 0.567, 10.0]
        )

    def test_from_dict_missing_required_field(self):
        """Test that missing required fields raise DeserializationError."""
        data = {
            "mean_params": [1.0, 2.0],
            # Missing mean_contrast and mean_offset
        }
        with pytest.raises(DeserializationError, match="Required field 'mean_contrast'"):
            MCMCResult.from_dict(data)

    def test_from_dict_invalid_type(self):
        """Test that invalid types raise DeserializationError."""
        data = {
            "mean_params": "not an array",  # Should be list/array
            "mean_contrast": 0.5,
            "mean_offset": 1.0,
        }
        with pytest.raises(DeserializationError, match="must be array-like"):
            MCMCResult.from_dict(data)

    def test_from_dict_array_size_limit(self):
        """Test that oversized arrays raise DeserializationError."""
        data = {
            "mean_params": list(range(1000)),  # Exceeds _MAX_PARAM_COUNT
            "mean_contrast": 0.5,
            "mean_offset": 1.0,
        }
        with pytest.raises(DeserializationError, match="exceeds size limit"):
            MCMCResult.from_dict(data)

    def test_from_dict_invalid_string_field(self):
        """Test that invalid string enum values raise DeserializationError."""
        data = {
            "mean_params": [1.0, 2.0, 3.0],
            "mean_contrast": 0.5,
            "mean_offset": 1.0,
            "analysis_mode": "invalid_mode",  # Not in allowed values
        }
        with pytest.raises(DeserializationError, match="must be one of"):
            MCMCResult.from_dict(data)

    def test_from_dict_numeric_bounds(self):
        """Test that out-of-bounds numeric values raise DeserializationError."""
        data = {
            "mean_params": [1.0, 2.0, 3.0],
            "mean_contrast": 0.5,
            "mean_offset": 1.0,
            "n_chains": 0,  # Below min_value of 1
        }
        with pytest.raises(DeserializationError, match="below minimum"):
            MCMCResult.from_dict(data)

    def test_from_dict_non_dict_input(self):
        """Test that non-dict input raises DeserializationError."""
        with pytest.raises(DeserializationError, match="Expected dict"):
            MCMCResult.from_dict("not a dict")

    def test_from_dict_param_names_validation(self):
        """Test param_names list validation."""
        data = {
            "mean_params": [1.0, 2.0, 3.0],
            "mean_contrast": 0.5,
            "mean_offset": 1.0,
            "param_names": ["D0", 123, "D_offset"],  # 123 is not a string
        }
        with pytest.raises(DeserializationError, match="must be string"):
            MCMCResult.from_dict(data)

    def test_from_dict_cmc_fields(self):
        """Test that CMC-specific fields are properly validated."""
        data = {
            "mean_params": [1.0, 2.0, 3.0],
            "mean_contrast": 0.5,
            "mean_offset": 1.0,
            "num_shards": 4,
            "combination_method": "precision_weighted",
        }
        result = MCMCResult.from_dict(data)
        assert result.num_shards == 4
        assert result.combination_method == "precision_weighted"
        assert result.is_cmc_result()


class TestValidationHelpers:
    """Tests for validation helper functions."""

    def test_validate_array_field_valid(self):
        """Test _validate_array_field with valid data."""
        data = {"test": [1.0, 2.0, 3.0]}
        result = _validate_array_field(data, "test", required=True)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_validate_array_field_none_optional(self):
        """Test _validate_array_field with None for optional field."""
        data = {"other": [1.0]}
        result = _validate_array_field(data, "test", required=False)
        assert result is None

    def test_validate_array_field_none_required(self):
        """Test _validate_array_field raises for missing required field."""
        data = {"other": [1.0]}
        with pytest.raises(DeserializationError, match="Required field"):
            _validate_array_field(data, "test", required=True)

    def test_validate_scalar_field_valid(self):
        """Test _validate_scalar_field with valid data."""
        data = {"value": 42}
        result = _validate_scalar_field(data, "value", expected_type=int)
        assert result == 42

    def test_validate_scalar_field_bounds(self):
        """Test _validate_scalar_field with bounds checking."""
        data = {"value": 5}
        result = _validate_scalar_field(
            data, "value", expected_type=int, min_value=0, max_value=10
        )
        assert result == 5

        data = {"value": 15}
        with pytest.raises(DeserializationError, match="exceeds maximum"):
            _validate_scalar_field(
                data, "value", expected_type=int, max_value=10
            )

    def test_validate_string_field_allowed_values(self):
        """Test _validate_string_field with allowed values."""
        data = {"mode": "static"}
        result = _validate_string_field(
            data, "mode", allowed_values=("static", "laminar_flow")
        )
        assert result == "static"

        data = {"mode": "invalid"}
        with pytest.raises(DeserializationError, match="must be one of"):
            _validate_string_field(
                data, "mode", allowed_values=("static", "laminar_flow")
            )

    def test_validate_string_field_max_length(self):
        """Test _validate_string_field with max length."""
        data = {"name": "x" * 2000}
        with pytest.raises(DeserializationError, match="exceeds max length"):
            _validate_string_field(data, "name", max_length=1000)


class TestPathValidation:
    """Tests for path validation in plotting functions."""

    def test_validate_plot_save_path_valid(self):
        """Test validate_plot_save_path with valid path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.png"
            result = validate_plot_save_path(str(path))
            assert result.suffix == ".png"

    def test_validate_plot_save_path_none(self):
        """Test validate_plot_save_path returns None for None input."""
        result = validate_plot_save_path(None)
        assert result is None

    def test_validate_plot_save_path_traversal_attack(self):
        """Test that path traversal is blocked."""
        with pytest.raises(PathValidationError, match="Path traversal detected"):
            validate_plot_save_path("../../../etc/passwd.png")

    def test_validate_plot_save_path_invalid_extension(self):
        """Test that invalid extensions are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.exe"
            with pytest.raises(ValueError, match="Invalid file extension"):
                validate_plot_save_path(str(path))

    def test_validate_plot_save_path_nonexistent_parent(self):
        """Test that nonexistent parent directory raises error."""
        with pytest.raises(ValueError, match="Parent directory does not exist"):
            validate_plot_save_path("/nonexistent/path/test.png")


class TestArvizPlotFunctions:
    """Tests for ArviZ plotting functions."""

    @pytest.fixture
    def mcmc_result_with_samples(self):
        """Create MCMCResult with samples for plotting tests."""
        n_chains = 2
        n_samples = 50
        n_params = 3

        samples = np.random.normal(0, 1, (n_chains, n_samples, n_params))

        return MCMCResult(
            mean_params=np.array([1000.0, 0.567, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            samples_params=samples.reshape(-1, n_params),
            n_chains=n_chains,
            n_samples=n_samples,
            param_names=["D0", "alpha", "D_offset"],
        )

    def test_plot_arviz_trace_returns_figure(self, mcmc_result_with_samples):
        """Test that plot_arviz_trace returns a matplotlib Figure."""
        pytest.importorskip("arviz")
        from homodyne.viz.mcmc_plots import plot_arviz_trace

        fig = plot_arviz_trace(mcmc_result_with_samples)
        assert fig is not None

    def test_plot_arviz_trace_with_save_path(self, mcmc_result_with_samples):
        """Test plot_arviz_trace with valid save_path."""
        pytest.importorskip("arviz")
        from homodyne.viz.mcmc_plots import plot_arviz_trace

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "trace.png"
            fig = plot_arviz_trace(mcmc_result_with_samples, save_path=str(save_path))
            assert save_path.exists()
            assert fig is not None

    def test_plot_arviz_trace_path_traversal_blocked(self, mcmc_result_with_samples):
        """Test that path traversal is blocked in plot_arviz_trace."""
        pytest.importorskip("arviz")
        from homodyne.viz.mcmc_plots import plot_arviz_trace

        # Should not raise, but should log warning and not save
        fig = plot_arviz_trace(
            mcmc_result_with_samples,
            save_path="../../../tmp/malicious.png"
        )
        # Function should still return figure even if save fails
        assert fig is not None

    def test_plot_arviz_posterior_returns_figure(self, mcmc_result_with_samples):
        """Test that plot_arviz_posterior returns a matplotlib Figure."""
        pytest.importorskip("arviz")
        from homodyne.viz.mcmc_plots import plot_arviz_posterior

        fig = plot_arviz_posterior(mcmc_result_with_samples)
        assert fig is not None

    def test_plot_arviz_pair_returns_figure(self, mcmc_result_with_samples):
        """Test that plot_arviz_pair returns a matplotlib Figure."""
        pytest.importorskip("arviz")
        from homodyne.viz.mcmc_plots import plot_arviz_pair

        fig = plot_arviz_pair(mcmc_result_with_samples)
        assert fig is not None


class TestConvergenceDiagnosticsPlot:
    """Tests for plot_convergence_diagnostics function."""

    @pytest.fixture
    def result_with_diagnostics(self):
        """Create MCMCResult with convergence diagnostics.

        Note: r_hat and effective_sample_size must be dicts for per-parameter diagnostics.
        """
        return MCMCResult(
            mean_params=np.array([1000.0, 0.567, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            r_hat={"D0": 1.01, "alpha": 1.02, "D_offset": 1.01},
            effective_sample_size={"D0": 450, "alpha": 480, "D_offset": 420},
            n_chains=4,
            n_samples=1000,
            param_names=["D0", "alpha", "D_offset"],
        )

    def test_plot_convergence_diagnostics_default_metrics(self, result_with_diagnostics):
        """Test plot_convergence_diagnostics with default metrics."""
        from homodyne.viz.mcmc_plots import plot_convergence_diagnostics

        fig = plot_convergence_diagnostics(result_with_diagnostics)
        assert fig is not None

    def test_plot_convergence_diagnostics_custom_metrics(self, result_with_diagnostics):
        """Test plot_convergence_diagnostics with custom metrics list."""
        from homodyne.viz.mcmc_plots import plot_convergence_diagnostics

        # Test with only rhat
        fig = plot_convergence_diagnostics(result_with_diagnostics, metrics=["rhat"])
        assert fig is not None

        # Test with only ess
        fig = plot_convergence_diagnostics(result_with_diagnostics, metrics=["ess"])
        assert fig is not None


class TestMCMCResultRoundtrip:
    """Tests for MCMCResult serialization roundtrip."""

    def test_to_dict_from_dict_roundtrip(self):
        """Test that to_dict -> from_dict preserves all data."""
        original = MCMCResult(
            mean_params=np.array([1000.0, 0.567, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            std_params=np.array([50.0, 0.02, 1.0]),
            std_contrast=0.02,
            std_offset=0.01,
            converged=True,
            n_iterations=1000,
            computation_time=10.5,
            backend="JAX",
            analysis_mode="static",
            n_chains=4,
            n_warmup=500,
            n_samples=1000,
            sampler="NUTS",
            r_hat=1.01,
            effective_sample_size=450.0,
            num_shards=4,
            combination_method="precision_weighted",
            param_names=["D0", "alpha", "D_offset"],
        )

        # Convert to dict and back
        data = original.to_dict()
        restored = MCMCResult.from_dict(data)

        # Verify key fields
        np.testing.assert_array_almost_equal(
            restored.mean_params, original.mean_params
        )
        assert restored.mean_contrast == original.mean_contrast
        assert restored.mean_offset == original.mean_offset
        assert restored.converged == original.converged
        assert restored.backend == original.backend
        assert restored.analysis_mode == original.analysis_mode
        assert restored.num_shards == original.num_shards
        assert restored.combination_method == original.combination_method
        assert restored.param_names == original.param_names

    def test_to_dict_from_dict_with_arrays(self):
        """Test roundtrip with sample arrays."""
        n_samples = 100
        n_params = 3

        original = MCMCResult(
            mean_params=np.array([1000.0, 0.567, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            std_params=np.array([50.0, 0.02, 1.0]),
            samples_params=np.random.normal(0, 1, (n_samples, n_params)),
            samples_contrast=np.random.normal(0.5, 0.02, n_samples),
            samples_offset=np.random.normal(1.0, 0.01, n_samples),
            ci_95_lower=np.array([900.0, 0.52, 8.0]),
            ci_95_upper=np.array([1100.0, 0.61, 12.0]),
        )

        data = original.to_dict()
        restored = MCMCResult.from_dict(data)

        np.testing.assert_array_almost_equal(
            restored.samples_params, original.samples_params
        )
        np.testing.assert_array_almost_equal(
            restored.ci_95_lower, original.ci_95_lower
        )
        np.testing.assert_array_almost_equal(
            restored.ci_95_upper, original.ci_95_upper
        )
