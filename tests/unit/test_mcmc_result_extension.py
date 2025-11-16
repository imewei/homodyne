"""Tests for extended MCMCResult class with CMC support.

Tests backward compatibility and CMC-specific functionality:
1. Backward compatibility with existing code
2. is_cmc_result() method correctness
3. CMC-specific fields preservation
4. Serialization/deserialization with CMC data
5. None defaults for non-CMC results
6. Edge cases (num_shards=1, missing fields)
"""

import json

import numpy as np
import pytest

from homodyne.optimization.cmc.result import MCMCResult


class TestBackwardCompatibility:
    """Test that existing code continues to work without modification."""

    def test_standard_mcmc_result_creation(self):
        """Test creating standard MCMC result without CMC fields."""
        # Create result with only standard fields (as existing code does)
        result = MCMCResult(
            mean_params=np.array([100.0, 1.5, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            std_params=np.array([5.0, 0.1, 0.5]),
            converged=True,
            n_iterations=1000,
        )

        # Verify standard fields work as before
        assert result.mean_params.shape == (3,)
        assert result.mean_contrast == 0.5
        assert result.mean_offset == 1.0
        assert result.converged is True
        assert result.n_iterations == 1000

        # Verify CMC fields default to None
        assert result.per_shard_diagnostics is None
        assert result.cmc_diagnostics is None
        assert result.combination_method is None
        assert result.num_shards is None

    def test_old_results_still_load(self):
        """Test that results without CMC fields can be deserialized."""
        # Simulate old result dictionary (no CMC fields)
        old_data = {
            "mean_params": [100.0, 1.5, 10.0],
            "mean_contrast": 0.5,
            "mean_offset": 1.0,
            "std_params": [5.0, 0.1, 0.5],
            "std_contrast": 0.05,
            "std_offset": 0.02,
            "samples_params": None,
            "samples_contrast": None,
            "samples_offset": None,
            "converged": True,
            "n_iterations": 1000,
            "computation_time": 45.5,
            "backend": "JAX",
            "analysis_mode": "static",
            "dataset_size": "medium",
            "n_chains": 4,
            "n_warmup": 500,
            "n_samples": 1000,
            "sampler": "NUTS",
            "acceptance_rate": 0.85,
            "r_hat": {"D0": 1.01, "alpha": 1.02},
            "effective_sample_size": {"D0": 800, "alpha": 750},
            # No CMC fields
        }

        # Should load without errors
        result = MCMCResult.from_dict(old_data)

        # Standard fields preserved
        assert np.allclose(result.mean_params, [100.0, 1.5, 10.0])
        assert result.mean_contrast == 0.5
        assert result.converged is True

        # CMC fields default to None
        assert result.num_shards is None
        assert result.combination_method is None
        assert result.is_cmc_result() is False


class TestIsCMCResult:
    """Test the is_cmc_result() method."""

    def test_is_cmc_result_false_when_none(self):
        """Test is_cmc_result() returns False when num_shards is None."""
        result = MCMCResult(
            mean_params=np.array([1.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
        )
        assert result.is_cmc_result() is False

    def test_is_cmc_result_false_when_one(self):
        """Test is_cmc_result() returns False when num_shards=1."""
        result = MCMCResult(
            mean_params=np.array([1.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            num_shards=1,  # Single shard is standard MCMC
        )
        assert result.is_cmc_result() is False

    def test_is_cmc_result_true_when_multiple(self):
        """Test is_cmc_result() returns True when num_shards > 1."""
        result = MCMCResult(
            mean_params=np.array([1.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            num_shards=10,
        )
        assert result.is_cmc_result() is True

    def test_is_cmc_result_with_all_cmc_fields(self):
        """Test is_cmc_result() with complete CMC data."""
        result = MCMCResult(
            mean_params=np.array([1.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            num_shards=5,
            combination_method="weighted",
            per_shard_diagnostics=[
                {"shard_id": 0, "converged": True},
                {"shard_id": 1, "converged": True},
            ],
            cmc_diagnostics={"combination_success": True},
        )
        assert result.is_cmc_result() is True
        assert result.combination_method == "weighted"


class TestCMCFieldsPreservation:
    """Test that CMC-specific fields are preserved correctly."""

    def test_per_shard_diagnostics_storage(self):
        """Test per_shard_diagnostics field storage."""
        diagnostics = [
            {
                "shard_id": 0,
                "converged": True,
                "acceptance_rate": 0.85,
                "n_samples": 1000,
            },
            {
                "shard_id": 1,
                "converged": True,
                "acceptance_rate": 0.82,
                "n_samples": 1000,
            },
            {
                "shard_id": 2,
                "converged": False,
                "acceptance_rate": 0.45,
                "n_samples": 1000,
            },
        ]

        result = MCMCResult(
            mean_params=np.array([1.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            num_shards=3,
            per_shard_diagnostics=diagnostics,
        )

        assert result.per_shard_diagnostics is not None
        assert len(result.per_shard_diagnostics) == 3
        assert result.per_shard_diagnostics[0]["shard_id"] == 0
        assert result.per_shard_diagnostics[2]["converged"] is False

    def test_cmc_diagnostics_storage(self):
        """Test cmc_diagnostics field storage."""
        cmc_diag = {
            "combination_success": True,
            "n_shards_converged": 8,
            "n_shards_total": 10,
            "weighted_product_std": 0.15,
            "combination_time": 2.3,
        }

        result = MCMCResult(
            mean_params=np.array([1.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            num_shards=10,
            cmc_diagnostics=cmc_diag,
        )

        assert result.cmc_diagnostics is not None
        assert result.cmc_diagnostics["combination_success"] is True
        assert result.cmc_diagnostics["n_shards_converged"] == 8
        assert result.cmc_diagnostics["combination_time"] == 2.3

    def test_combination_method_storage(self):
        """Test combination_method field storage."""
        for method in ["weighted", "average", "hierarchical"]:
            result = MCMCResult(
                mean_params=np.array([1.0]),
                mean_contrast=0.5,
                mean_offset=1.0,
                num_shards=5,
                combination_method=method,
            )
            assert result.combination_method == method


class TestSerialization:
    """Test serialization and deserialization with CMC data."""

    def test_cmc_result_serialization(self):
        """Test to_dict() preserves CMC-specific data."""
        result = MCMCResult(
            mean_params=np.array([100.0, 1.5, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            std_params=np.array([5.0, 0.1, 0.5]),
            num_shards=5,
            combination_method="weighted",
            per_shard_diagnostics=[
                {"shard_id": 0, "converged": True},
                {"shard_id": 1, "converged": True},
            ],
            cmc_diagnostics={"combination_success": True, "n_shards_total": 5},
        )

        data = result.to_dict()

        # Standard fields
        assert data["mean_params"] == [100.0, 1.5, 10.0]
        assert data["mean_contrast"] == 0.5

        # CMC fields
        assert data["num_shards"] == 5
        assert data["combination_method"] == "weighted"
        assert len(data["per_shard_diagnostics"]) == 2
        assert data["cmc_diagnostics"]["combination_success"] is True

    def test_cmc_result_deserialization(self):
        """Test from_dict() reconstructs CMC-specific data."""
        data = {
            "mean_params": [100.0, 1.5, 10.0],
            "mean_contrast": 0.5,
            "mean_offset": 1.0,
            "std_params": [5.0, 0.1, 0.5],
            "std_contrast": 0.05,
            "std_offset": 0.02,
            "samples_params": None,
            "samples_contrast": None,
            "samples_offset": None,
            "converged": True,
            "n_iterations": 2000,
            "computation_time": 120.5,
            "backend": "JAX",
            "analysis_mode": "laminar_flow",
            "dataset_size": "large",
            "n_chains": 4,
            "n_warmup": 500,
            "n_samples": 2000,
            "sampler": "NUTS",
            "acceptance_rate": 0.83,
            "r_hat": None,
            "effective_sample_size": None,
            # CMC fields
            "num_shards": 10,
            "combination_method": "weighted",
            "per_shard_diagnostics": [
                {"shard_id": 0, "converged": True, "acceptance_rate": 0.85},
                {"shard_id": 1, "converged": True, "acceptance_rate": 0.82},
            ],
            "cmc_diagnostics": {
                "combination_success": True,
                "n_shards_converged": 9,
                "n_shards_total": 10,
            },
        }

        result = MCMCResult.from_dict(data)

        # Standard fields
        assert np.allclose(result.mean_params, [100.0, 1.5, 10.0])
        assert result.mean_contrast == 0.5
        assert result.analysis_mode == "laminar_flow"

        # CMC fields
        assert result.is_cmc_result() is True
        assert result.num_shards == 10
        assert result.combination_method == "weighted"
        assert len(result.per_shard_diagnostics) == 2
        assert result.cmc_diagnostics["n_shards_converged"] == 9

    def test_serialization_roundtrip(self):
        """Test full serialization roundtrip preserves all data."""
        original = MCMCResult(
            mean_params=np.array([100.0, 1.5, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            std_params=np.array([5.0, 0.1, 0.5]),
            samples_params=np.array([[99.0, 1.4, 9.8], [101.0, 1.6, 10.2]]),
            samples_contrast=np.array([0.49, 0.51]),
            samples_offset=np.array([0.98, 1.02]),
            num_shards=5,
            combination_method="average",
            per_shard_diagnostics=[
                {"shard_id": i, "converged": True} for i in range(5)
            ],
            cmc_diagnostics={"combination_success": True},
        )

        # Serialize
        data = original.to_dict()

        # Deserialize
        reconstructed = MCMCResult.from_dict(data)

        # Verify all fields match
        assert np.allclose(reconstructed.mean_params, original.mean_params)
        assert reconstructed.mean_contrast == original.mean_contrast
        assert reconstructed.num_shards == original.num_shards
        assert reconstructed.combination_method == original.combination_method
        assert len(reconstructed.per_shard_diagnostics) == 5
        assert reconstructed.is_cmc_result() == original.is_cmc_result()

        # Verify samples preserved
        assert np.allclose(reconstructed.samples_params, original.samples_params)
        assert np.allclose(reconstructed.samples_contrast, original.samples_contrast)

    def test_json_serialization_compatibility(self):
        """Test that results can be serialized to JSON."""
        result = MCMCResult(
            mean_params=np.array([100.0, 1.5]),
            mean_contrast=0.5,
            mean_offset=1.0,
            num_shards=3,
            combination_method="weighted",
            cmc_diagnostics={"combination_success": True},
        )

        # Convert to dict and serialize to JSON
        data = result.to_dict()
        json_str = json.dumps(data)

        # Deserialize from JSON
        loaded_data = json.loads(json_str)
        reconstructed = MCMCResult.from_dict(loaded_data)

        # Verify reconstruction
        assert np.allclose(reconstructed.mean_params, result.mean_params)
        assert reconstructed.num_shards == result.num_shards
        assert reconstructed.combination_method == result.combination_method


class TestNoneDefaults:
    """Test that CMC fields default to None for non-CMC results."""

    def test_minimal_result_creation(self):
        """Test creating result with minimal required fields."""
        result = MCMCResult(
            mean_params=np.array([1.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
        )

        # All CMC fields should be None
        assert result.num_shards is None
        assert result.combination_method is None
        assert result.per_shard_diagnostics is None
        assert result.cmc_diagnostics is None

        # Standard fields should have defaults
        assert result.converged is True  # default
        assert result.n_iterations == 0  # default
        assert result.backend == "JAX"  # default

    def test_partial_cmc_fields(self):
        """Test providing only some CMC fields."""
        # Only num_shards provided
        result1 = MCMCResult(
            mean_params=np.array([1.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            num_shards=5,
        )
        assert result1.is_cmc_result() is True
        assert result1.combination_method is None  # Other fields still None

        # Only combination_method provided (but no num_shards)
        result2 = MCMCResult(
            mean_params=np.array([1.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            combination_method="weighted",
        )
        assert result2.is_cmc_result() is False  # Not CMC without num_shards
        assert result2.combination_method == "weighted"  # But field is set


class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_empty_per_shard_diagnostics(self):
        """Test with empty per_shard_diagnostics list."""
        result = MCMCResult(
            mean_params=np.array([1.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            num_shards=5,
            per_shard_diagnostics=[],  # Empty list
        )
        assert result.is_cmc_result() is True
        assert result.per_shard_diagnostics == []

    def test_large_num_shards(self):
        """Test with large number of shards."""
        result = MCMCResult(
            mean_params=np.array([1.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            num_shards=1000,  # Large number
        )
        assert result.is_cmc_result() is True
        assert result.num_shards == 1000

    def test_zero_num_shards(self):
        """Test with num_shards=0 (invalid but should handle gracefully)."""
        result = MCMCResult(
            mean_params=np.array([1.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            num_shards=0,
        )
        # num_shards=0 should be treated as non-CMC
        assert result.is_cmc_result() is False

    def test_deserialization_with_extra_fields(self):
        """Test deserialization ignores unknown fields."""
        data = {
            "mean_params": [1.0],
            "mean_contrast": 0.5,
            "mean_offset": 1.0,
            "num_shards": 5,
            "unknown_field": "should be ignored",  # Extra field
            "another_unknown": 123,
        }

        # Should not raise error
        result = MCMCResult.from_dict(data)
        assert result.num_shards == 5
        assert result.is_cmc_result() is True
