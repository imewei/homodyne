"""Integration tests for CMC result saving and loading with v2.1.0 metadata.

This module tests the new config-driven metadata fields added to MCMCResult in v2.1.0:
- parameter_space_metadata
- initial_values_metadata
- selection_decision_metadata

Test Coverage
-------------
- Result saving/loading with new metadata
- Backward compatibility with old result files (pre-v2.1.0)
- JSON serialization of metadata dictionaries
- Integration with save_mcmc_results() function

Backward Compatibility
----------------------
Old result files (pre-v2.1.0) without new metadata fields should load successfully
with the new fields defaulting to None.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from homodyne.optimization.cmc.result import MCMCResult


# ==============================================================================
# Test Class: Result Structure with New Metadata
# ==============================================================================


class TestMCMCResultMetadata:
    """Test MCMCResult class with v2.1.0 config-driven metadata fields."""

    def test_result_with_parameter_space_metadata(self):
        """Verify parameter_space_metadata field works correctly."""
        # Create parameter space metadata (typical from ParameterSpace.from_config())
        param_space_metadata = {
            "bounds": {
                "D0": [50.0, 500.0],
                "alpha": [0.1, 2.0],
                "D_offset": [1.0, 50.0],
            },
            "priors": {
                "D0": {"type": "TruncatedNormal", "mu": 200.0, "sigma": 50.0},
                "alpha": {"type": "TruncatedNormal", "mu": 1.0, "sigma": 0.3},
                "D_offset": {"type": "TruncatedNormal", "mu": 10.0, "sigma": 5.0},
            },
            "model_type": "static",
        }

        # Create result with metadata
        result = MCMCResult(
            mean_params=np.array([200.0, 1.0, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            parameter_space_metadata=param_space_metadata,
        )

        # Verify metadata stored correctly
        assert result.parameter_space_metadata is not None
        assert result.parameter_space_metadata["model_type"] == "static"
        assert "D0" in result.parameter_space_metadata["bounds"]
        assert "D0" in result.parameter_space_metadata["priors"]
        assert (
            result.parameter_space_metadata["priors"]["D0"]["type"] == "TruncatedNormal"
        )

    def test_result_with_initial_values_metadata(self):
        """Verify initial_values_metadata field works correctly."""
        # Create initial values metadata (from config or NLSQ results)
        initial_values_metadata = {
            "D0": 1234.5,
            "alpha": 0.567,
            "D_offset": 12.34,
        }

        # Create result with metadata
        result = MCMCResult(
            mean_params=np.array([1250.0, 0.58, 12.5]),
            mean_contrast=0.5,
            mean_offset=1.0,
            initial_values_metadata=initial_values_metadata,
        )

        # Verify metadata stored correctly
        assert result.initial_values_metadata is not None
        assert result.initial_values_metadata["D0"] == 1234.5
        assert result.initial_values_metadata["alpha"] == 0.567
        assert result.initial_values_metadata["D_offset"] == 12.34

    def test_result_with_selection_decision_metadata(self):
        """Verify selection_decision_metadata field works correctly."""
        # Create selection decision metadata (from automatic NUTS/CMC selection)
        selection_decision_metadata = {
            "selected_method": "CMC",
            "num_samples": 50,
            "parallelism_criterion_met": True,
            "memory_criterion_met": False,
            "min_samples_for_cmc": 15,
            "memory_threshold_pct": 0.30,
            "estimated_memory_fraction": 0.15,
        }

        # Create result with metadata
        result = MCMCResult(
            mean_params=np.array([200.0, 1.0, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            num_shards=10,  # CMC result
            selection_decision_metadata=selection_decision_metadata,
        )

        # Verify metadata stored correctly
        assert result.selection_decision_metadata is not None
        assert result.selection_decision_metadata["selected_method"] == "CMC"
        assert result.selection_decision_metadata["parallelism_criterion_met"] is True
        assert result.selection_decision_metadata["memory_criterion_met"] is False
        assert result.is_cmc_result()  # Should be detected as CMC

    def test_result_with_all_metadata_fields(self):
        """Verify all three metadata fields can coexist."""
        # Create all metadata fields
        param_space_metadata = {
            "bounds": {"D0": [50.0, 500.0]},
            "priors": {"D0": {"type": "TruncatedNormal", "mu": 200.0, "sigma": 50.0}},
            "model_type": "static",
        }
        initial_values_metadata = {"D0": 1234.5, "alpha": 0.567, "D_offset": 12.34}
        selection_decision_metadata = {
            "selected_method": "NUTS",
            "num_samples": 10,
            "parallelism_criterion_met": False,
            "memory_criterion_met": False,
        }

        # Create result with all metadata
        result = MCMCResult(
            mean_params=np.array([1250.0, 0.58, 12.5]),
            mean_contrast=0.5,
            mean_offset=1.0,
            parameter_space_metadata=param_space_metadata,
            initial_values_metadata=initial_values_metadata,
            selection_decision_metadata=selection_decision_metadata,
        )

        # Verify all metadata fields present
        assert result.parameter_space_metadata is not None
        assert result.initial_values_metadata is not None
        assert result.selection_decision_metadata is not None


# ==============================================================================
# Test Class: Serialization and Deserialization
# ==============================================================================


class TestResultSerialization:
    """Test to_dict() and from_dict() with new metadata fields."""

    def test_to_dict_includes_new_metadata(self):
        """Verify to_dict() includes v2.1.0 metadata fields."""
        # Create result with metadata
        result = MCMCResult(
            mean_params=np.array([200.0, 1.0, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            parameter_space_metadata={"model_type": "static"},
            initial_values_metadata={"D0": 1234.5},
            selection_decision_metadata={"selected_method": "NUTS"},
        )

        # Convert to dict
        data = result.to_dict()

        # Verify new fields in dictionary
        assert "parameter_space_metadata" in data
        assert "initial_values_metadata" in data
        assert "selection_decision_metadata" in data
        assert data["parameter_space_metadata"]["model_type"] == "static"
        assert data["initial_values_metadata"]["D0"] == 1234.5
        assert data["selection_decision_metadata"]["selected_method"] == "NUTS"

    def test_from_dict_with_new_metadata(self):
        """Verify from_dict() reconstructs result with v2.1.0 metadata."""
        # Create dictionary with new metadata fields
        data = {
            "mean_params": [200.0, 1.0, 10.0],
            "mean_contrast": 0.5,
            "mean_offset": 1.0,
            "parameter_space_metadata": {
                "bounds": {"D0": [50.0, 500.0]},
                "model_type": "static",
            },
            "initial_values_metadata": {"D0": 1234.5, "alpha": 0.567},
            "selection_decision_metadata": {
                "selected_method": "CMC",
                "parallelism_criterion_met": True,
            },
        }

        # Reconstruct result
        result = MCMCResult.from_dict(data)

        # Verify metadata reconstructed correctly
        assert result.parameter_space_metadata is not None
        assert result.parameter_space_metadata["model_type"] == "static"
        assert result.initial_values_metadata["D0"] == 1234.5
        assert result.selection_decision_metadata["selected_method"] == "CMC"

    def test_from_dict_backward_compatibility(self):
        """Verify from_dict() handles old results without new metadata (backward compatibility)."""
        # Create dictionary WITHOUT new metadata fields (simulates old v2.0.0 result file)
        data = {
            "mean_params": [200.0, 1.0, 10.0],
            "mean_contrast": 0.5,
            "mean_offset": 1.0,
            # No parameter_space_metadata
            # No initial_values_metadata
            # No selection_decision_metadata
        }

        # Reconstruct result
        result = MCMCResult.from_dict(data)

        # Verify new metadata fields default to None (backward compatible)
        assert result.parameter_space_metadata is None
        assert result.initial_values_metadata is None
        assert result.selection_decision_metadata is None

        # Verify standard fields still work
        assert np.allclose(result.mean_params, [200.0, 1.0, 10.0])
        assert result.mean_contrast == 0.5
        assert result.mean_offset == 1.0

    def test_round_trip_serialization(self):
        """Verify save → load preserves all metadata."""
        # Create result with all metadata
        original = MCMCResult(
            mean_params=np.array([200.0, 1.0, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            std_params=np.array([5.0, 0.1, 1.0]),
            parameter_space_metadata={
                "bounds": {"D0": [50.0, 500.0]},
                "priors": {"D0": {"type": "TruncatedNormal", "mu": 200.0}},
                "model_type": "static",
            },
            initial_values_metadata={"D0": 1234.5, "alpha": 0.567, "D_offset": 12.34},
            selection_decision_metadata={
                "selected_method": "NUTS",
                "num_samples": 10,
                "parallelism_criterion_met": False,
            },
        )

        # Round-trip: to_dict → from_dict
        data = original.to_dict()
        reconstructed = MCMCResult.from_dict(data)

        # Verify all metadata preserved
        assert (
            reconstructed.parameter_space_metadata == original.parameter_space_metadata
        )
        assert reconstructed.initial_values_metadata == original.initial_values_metadata
        assert (
            reconstructed.selection_decision_metadata
            == original.selection_decision_metadata
        )

        # Verify standard fields preserved
        assert np.allclose(reconstructed.mean_params, original.mean_params)
        assert np.allclose(reconstructed.std_params, original.std_params)
        assert reconstructed.mean_contrast == original.mean_contrast


# ==============================================================================
# Test Class: JSON File Saving/Loading Integration
# ==============================================================================


class TestResultFileSaving:
    """Test integration with save_mcmc_results() function."""

    def test_json_serialization_of_metadata(self):
        """Verify metadata fields can be serialized to JSON."""
        # Create result with complex metadata
        result = MCMCResult(
            mean_params=np.array([200.0, 1.0, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            parameter_space_metadata={
                "bounds": {
                    "D0": [50.0, 500.0],
                    "alpha": [0.1, 2.0],
                    "D_offset": [1.0, 50.0],
                },
                "priors": {
                    "D0": {"type": "TruncatedNormal", "mu": 200.0, "sigma": 50.0},
                    "alpha": {"type": "TruncatedNormal", "mu": 1.0, "sigma": 0.3},
                },
                "model_type": "static",
            },
            initial_values_metadata={"D0": 1234.5, "alpha": 0.567, "D_offset": 12.34},
            selection_decision_metadata={
                "selected_method": "CMC",
                "num_samples": 50,
                "parallelism_criterion_met": True,
                "memory_criterion_met": False,
                "min_samples_for_cmc": 15,
                "memory_threshold_pct": 0.30,
            },
        )

        # Convert to dict
        data = result.to_dict()

        # Verify JSON serialization works (no errors)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f, indent=2)
            temp_path = f.name

        try:
            # Load back and verify
            with open(temp_path, "r") as f:
                loaded_data = json.load(f)

            # Verify metadata preserved through JSON serialization
            assert "parameter_space_metadata" in loaded_data
            assert "initial_values_metadata" in loaded_data
            assert "selection_decision_metadata" in loaded_data
            assert (
                loaded_data["parameter_space_metadata"]["model_type"]
                == "static"
            )
            assert loaded_data["initial_values_metadata"]["D0"] == 1234.5
            assert (
                loaded_data["selection_decision_metadata"]["selected_method"] == "CMC"
            )
        finally:
            # Clean up
            Path(temp_path).unlink()

    def test_old_result_file_loads_without_new_metadata(self):
        """Verify old result files (without new metadata) can still be loaded."""
        # Simulate old result file (v2.0.0) - JSON without new metadata fields
        old_result_json = {
            "mean_params": [200.0, 1.0, 10.0],
            "mean_contrast": 0.5,
            "mean_offset": 1.0,
            "std_params": [5.0, 0.1, 1.0],
            "converged": True,
            "n_iterations": 2000,
            "computation_time": 45.3,
            "backend": "JAX",
            "analysis_mode": "static",
            "n_chains": 4,
            "n_warmup": 1000,
            "n_samples": 1000,
            "sampler": "NUTS",
            # No new v2.1.0 fields here (simulates old file)
        }

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(old_result_json, f, indent=2)
            temp_path = f.name

        try:
            # Load and reconstruct result
            with open(temp_path, "r") as f:
                loaded_data = json.load(f)

            result = MCMCResult.from_dict(loaded_data)

            # Verify result loads successfully
            assert np.allclose(result.mean_params, [200.0, 1.0, 10.0])
            assert result.converged is True

            # Verify new metadata fields default to None (backward compatible)
            assert result.parameter_space_metadata is None
            assert result.initial_values_metadata is None
            assert result.selection_decision_metadata is None
        finally:
            # Clean up
            Path(temp_path).unlink()


# ==============================================================================
# Test Class: CMC-Specific Integration
# ==============================================================================


class TestCMCResultMetadata:
    """Test CMC results with combined CMC and config-driven metadata."""

    def test_cmc_result_with_selection_metadata(self):
        """Verify CMC results include selection decision metadata."""
        # Create CMC result with selection decision
        result = MCMCResult(
            mean_params=np.array([200.0, 1.0, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            num_shards=10,  # CMC indicator
            combination_method="weighted",
            cmc_diagnostics={"combination_success": True, "n_shards_converged": 10},
            selection_decision_metadata={
                "selected_method": "CMC",
                "num_samples": 50,
                "parallelism_criterion_met": True,
                "memory_criterion_met": False,
                "min_samples_for_cmc": 15,
            },
        )

        # Verify CMC detection works
        assert result.is_cmc_result() is True
        assert result.num_shards == 10

        # Verify selection metadata shows why CMC was chosen
        assert result.selection_decision_metadata["selected_method"] == "CMC"
        assert result.selection_decision_metadata["parallelism_criterion_met"] is True

    def test_nuts_result_with_selection_metadata(self):
        """Verify NUTS results include selection decision metadata."""
        # Create NUTS result with selection decision
        result = MCMCResult(
            mean_params=np.array([200.0, 1.0, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            num_shards=None,  # Not CMC
            selection_decision_metadata={
                "selected_method": "NUTS",
                "num_samples": 10,
                "parallelism_criterion_met": False,
                "memory_criterion_met": False,
                "min_samples_for_cmc": 15,
            },
        )

        # Verify NUTS detection works
        assert result.is_cmc_result() is False

        # Verify selection metadata shows why NUTS was chosen
        assert result.selection_decision_metadata["selected_method"] == "NUTS"
        assert result.selection_decision_metadata["parallelism_criterion_met"] is False
        assert result.selection_decision_metadata["memory_criterion_met"] is False
