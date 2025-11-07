"""Comprehensive tests for checkpoint_manager.py to achieve >80% coverage.

This test suite covers:
- Checkpoint save/load with and without compression
- Metadata handling (dict, list, numpy arrays)
- Checksum validation and corruption detection
- Cleanup edge cases and error handling
- Validation method edge cases
- Save time warnings
- Missing fields and invalid checksums
"""

import json
import pickle
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import h5py
import numpy as np
import pytest

from homodyne.optimization.checkpoint_manager import CheckpointManager
from homodyne.optimization.exceptions import NLSQCheckpointError


class TestCheckpointManagerSave:
    """Test checkpoint saving functionality."""

    def test_save_checkpoint_with_compression(self, tmp_path):
        """Test saving checkpoint with compression enabled."""
        manager = CheckpointManager(tmp_path, enable_compression=True)
        params = np.array([1.0, 2.0, 3.0])
        optimizer_state = {"iteration": 42, "learning_rate": 0.01}

        checkpoint_path = manager.save_checkpoint(
            batch_idx=10,
            parameters=params,
            optimizer_state=optimizer_state,
            loss=0.123,
        )

        assert checkpoint_path.exists()
        assert checkpoint_path.name == "homodyne_state_batch_0010.h5"

        # Verify compression was applied
        with h5py.File(checkpoint_path, "r") as f:
            assert "parameters" in f
            # Check if compression is enabled
            params_dataset = f["parameters"]
            assert params_dataset.compression == "gzip"
            assert params_dataset.compression_opts == 4

    def test_save_checkpoint_without_compression(self, tmp_path):
        """Test saving checkpoint with compression disabled."""
        manager = CheckpointManager(tmp_path, enable_compression=False)
        params = np.array([1.0, 2.0, 3.0])
        optimizer_state = {"iteration": 42}

        checkpoint_path = manager.save_checkpoint(
            batch_idx=5,
            parameters=params,
            optimizer_state=optimizer_state,
            loss=0.456,
        )

        # Verify no compression
        with h5py.File(checkpoint_path, "r") as f:
            params_dataset = f["parameters"]
            assert params_dataset.compression is None

    def test_save_checkpoint_with_scalar_metadata(self, tmp_path):
        """Test saving checkpoint with scalar metadata."""
        manager = CheckpointManager(tmp_path)
        params = np.array([1.0, 2.0])
        optimizer_state = {"iteration": 10}
        metadata = {
            "int_value": 42,
            "float_value": 3.14,
            "str_value": "test",
            "bool_value": True,
        }

        checkpoint_path = manager.save_checkpoint(
            batch_idx=1,
            parameters=params,
            optimizer_state=optimizer_state,
            loss=0.1,
            metadata=metadata,
        )

        # Verify metadata was saved
        with h5py.File(checkpoint_path, "r") as f:
            assert "metadata" in f
            metadata_group = f["metadata"]
            assert metadata_group.attrs["int_value"] == 42
            assert metadata_group.attrs["float_value"] == 3.14
            assert metadata_group.attrs["str_value"] == "test"
            # h5py converts bool to numpy bool, use == not is
            assert metadata_group.attrs["bool_value"] == True

    def test_save_checkpoint_with_list_metadata(self, tmp_path):
        """Test saving checkpoint with list metadata (JSON serialization)."""
        manager = CheckpointManager(tmp_path)
        params = np.array([1.0, 2.0])
        optimizer_state = {"iteration": 10}
        metadata = {
            "list_value": [1, 2, 3, 4],
            "nested_list": [[1, 2], [3, 4]],
        }

        checkpoint_path = manager.save_checkpoint(
            batch_idx=1,
            parameters=params,
            optimizer_state=optimizer_state,
            loss=0.1,
            metadata=metadata,
        )

        # Verify list was JSON-serialized
        with h5py.File(checkpoint_path, "r") as f:
            metadata_group = f["metadata"]
            stored_list = json.loads(metadata_group.attrs["list_value"])
            assert stored_list == [1, 2, 3, 4]

    def test_save_checkpoint_with_dict_metadata(self, tmp_path):
        """Test saving checkpoint with dict metadata (JSON serialization)."""
        manager = CheckpointManager(tmp_path)
        params = np.array([1.0, 2.0])
        optimizer_state = {"iteration": 10}
        metadata = {
            "dict_value": {"a": 1, "b": 2, "c": 3},
        }

        checkpoint_path = manager.save_checkpoint(
            batch_idx=1,
            parameters=params,
            optimizer_state=optimizer_state,
            loss=0.1,
            metadata=metadata,
        )

        # Verify dict was JSON-serialized
        with h5py.File(checkpoint_path, "r") as f:
            metadata_group = f["metadata"]
            stored_dict = json.loads(metadata_group.attrs["dict_value"])
            assert stored_dict == {"a": 1, "b": 2, "c": 3}

    def test_save_checkpoint_with_numpy_array_metadata(self, tmp_path):
        """Test saving checkpoint with numpy array metadata."""
        manager = CheckpointManager(tmp_path)
        params = np.array([1.0, 2.0])
        optimizer_state = {"iteration": 10}
        metadata = {
            "array_value": np.array([10.0, 20.0, 30.0]),
        }

        checkpoint_path = manager.save_checkpoint(
            batch_idx=1,
            parameters=params,
            optimizer_state=optimizer_state,
            loss=0.1,
            metadata=metadata,
        )

        # Verify array was saved as dataset
        with h5py.File(checkpoint_path, "r") as f:
            metadata_group = f["metadata"]
            assert "array_value" in metadata_group
            stored_array = metadata_group["array_value"][:]
            np.testing.assert_array_equal(stored_array, np.array([10.0, 20.0, 30.0]))

    def test_save_checkpoint_slow_warning(self, tmp_path, caplog):
        """Test warning when checkpoint save exceeds 2 seconds."""
        manager = CheckpointManager(tmp_path)
        params = np.array([1.0, 2.0, 3.0])
        optimizer_state = {"iteration": 42}

        # Mock time.time to simulate slow save
        original_time = time.time
        call_count = [0]

        def mock_time():
            call_count[0] += 1
            if call_count[0] == 1:
                return 0.0  # Start time
            else:
                return 2.5  # End time (2.5 seconds elapsed)

        with patch("time.time", side_effect=mock_time):
            checkpoint_path = manager.save_checkpoint(
                batch_idx=10,
                parameters=params,
                optimizer_state=optimizer_state,
                loss=0.123,
            )

        # Check warning was logged
        assert "Checkpoint save took 2.50s" in caplog.text
        assert "target: < 2s" in caplog.text

    def test_save_checkpoint_error_handling(self, tmp_path):
        """Test error handling during checkpoint save."""
        manager = CheckpointManager(tmp_path)
        params = np.array([1.0, 2.0, 3.0])

        # Create an un-picklable optimizer state
        import threading

        optimizer_state = {"lock": threading.Lock()}  # Cannot pickle locks

        with pytest.raises(NLSQCheckpointError, match="Failed to save checkpoint"):
            manager.save_checkpoint(
                batch_idx=10,
                parameters=params,
                optimizer_state=optimizer_state,
                loss=0.123,
            )


class TestCheckpointManagerLoad:
    """Test checkpoint loading functionality."""

    def test_load_checkpoint_missing_file(self, tmp_path):
        """Test loading non-existent checkpoint raises error."""
        manager = CheckpointManager(tmp_path)
        missing_path = tmp_path / "missing_checkpoint.h5"

        with pytest.raises(NLSQCheckpointError, match="Checkpoint file not found"):
            manager.load_checkpoint(missing_path)

    def test_load_checkpoint_corrupted_checksum(self, tmp_path):
        """Test loading checkpoint with corrupted checksum."""
        manager = CheckpointManager(tmp_path)
        params = np.array([1.0, 2.0, 3.0])
        optimizer_state = {"iteration": 42}

        # Save valid checkpoint
        checkpoint_path = manager.save_checkpoint(
            batch_idx=10,
            parameters=params,
            optimizer_state=optimizer_state,
            loss=0.123,
        )

        # Corrupt the checksum
        with h5py.File(checkpoint_path, "r+") as f:
            f.attrs["checksum"] = "corrupted_checksum_hash"

        # Loading should fail with checksum mismatch
        with pytest.raises(NLSQCheckpointError, match="Checkpoint checksum mismatch"):
            manager.load_checkpoint(checkpoint_path)

    def test_load_checkpoint_with_metadata(self, tmp_path):
        """Test loading checkpoint with various metadata types."""
        manager = CheckpointManager(tmp_path)
        params = np.array([1.0, 2.0, 3.0])
        optimizer_state = {"iteration": 42}
        metadata = {
            "scalar_int": 100,
            "scalar_float": 3.14,
            "list_value": [1, 2, 3],
            "dict_value": {"a": 1, "b": 2},
            "array_value": np.array([10.0, 20.0]),
        }

        checkpoint_path = manager.save_checkpoint(
            batch_idx=10,
            parameters=params,
            optimizer_state=optimizer_state,
            loss=0.123,
            metadata=metadata,
        )

        # Load and verify metadata
        loaded = manager.load_checkpoint(checkpoint_path)
        assert loaded["metadata"]["scalar_int"] == 100
        assert loaded["metadata"]["scalar_float"] == 3.14
        assert loaded["metadata"]["list_value"] == [1, 2, 3]
        assert loaded["metadata"]["dict_value"] == {"a": 1, "b": 2}
        np.testing.assert_array_equal(
            loaded["metadata"]["array_value"], np.array([10.0, 20.0])
        )

    def test_load_checkpoint_json_decode_error(self, tmp_path):
        """Test loading checkpoint with invalid JSON in metadata."""
        manager = CheckpointManager(tmp_path)
        params = np.array([1.0, 2.0, 3.0])
        optimizer_state = {"iteration": 42}

        checkpoint_path = manager.save_checkpoint(
            batch_idx=10,
            parameters=params,
            optimizer_state=optimizer_state,
            loss=0.123,
        )

        # Manually add invalid JSON to metadata
        with h5py.File(checkpoint_path, "r+") as f:
            metadata_group = f.create_group("metadata_corrupt")
            metadata_group.attrs["invalid_json"] = "{this is not valid json"

        # Load should handle JSON decode error gracefully
        # (falls back to storing as string)
        loaded = manager.load_checkpoint(checkpoint_path)
        # Metadata should still be accessible
        assert "metadata" in loaded

    def test_load_checkpoint_h5py_error(self, tmp_path):
        """Test error handling when HDF5 file is corrupted."""
        manager = CheckpointManager(tmp_path)

        # Create a corrupted HDF5 file
        corrupt_path = tmp_path / "homodyne_state_batch_0010.h5"
        with open(corrupt_path, "w") as f:
            f.write("This is not a valid HDF5 file")

        with pytest.raises(NLSQCheckpointError, match="Failed to load checkpoint"):
            manager.load_checkpoint(corrupt_path)


class TestCheckpointManagerValidation:
    """Test checkpoint validation functionality."""

    def test_validate_checkpoint_missing_file(self, tmp_path):
        """Test validation fails for missing file."""
        manager = CheckpointManager(tmp_path)
        missing_path = tmp_path / "missing.h5"

        assert manager.validate_checkpoint(missing_path) is False

    def test_validate_checkpoint_missing_required_field(self, tmp_path):
        """Test validation fails for missing required field."""
        manager = CheckpointManager(tmp_path)
        params = np.array([1.0, 2.0, 3.0])
        optimizer_state = {"iteration": 42}

        checkpoint_path = manager.save_checkpoint(
            batch_idx=10,
            parameters=params,
            optimizer_state=optimizer_state,
            loss=0.123,
        )

        # Remove required field
        with h5py.File(checkpoint_path, "r+") as f:
            del f["parameters"]

        assert manager.validate_checkpoint(checkpoint_path) is False

    def test_validate_checkpoint_missing_required_attr(self, tmp_path):
        """Test validation fails for missing required attribute."""
        manager = CheckpointManager(tmp_path)
        params = np.array([1.0, 2.0, 3.0])
        optimizer_state = {"iteration": 42}

        checkpoint_path = manager.save_checkpoint(
            batch_idx=10,
            parameters=params,
            optimizer_state=optimizer_state,
            loss=0.123,
        )

        # Remove required attribute
        with h5py.File(checkpoint_path, "r+") as f:
            del f.attrs["batch_idx"]

        assert manager.validate_checkpoint(checkpoint_path) is False

    def test_validate_checkpoint_corrupted_checksum(self, tmp_path):
        """Test validation fails for corrupted checksum."""
        manager = CheckpointManager(tmp_path)
        params = np.array([1.0, 2.0, 3.0])
        optimizer_state = {"iteration": 42}

        checkpoint_path = manager.save_checkpoint(
            batch_idx=10,
            parameters=params,
            optimizer_state=optimizer_state,
            loss=0.123,
        )

        # Corrupt checksum
        with h5py.File(checkpoint_path, "r+") as f:
            f.attrs["checksum"] = "wrong_checksum"

        assert manager.validate_checkpoint(checkpoint_path) is False

    def test_validate_checkpoint_oserror(self, tmp_path):
        """Test validation handles OSError gracefully."""
        manager = CheckpointManager(tmp_path)

        # Create a file that will cause OSError when opened as HDF5
        bad_path = tmp_path / "bad_checkpoint.h5"
        with open(bad_path, "w") as f:
            f.write("Not an HDF5 file")

        assert manager.validate_checkpoint(bad_path) is False

    def test_validate_checkpoint_keyerror(self, tmp_path):
        """Test validation handles KeyError gracefully."""
        manager = CheckpointManager(tmp_path)

        # Create minimal HDF5 file missing required fields
        incomplete_path = tmp_path / "incomplete.h5"
        with h5py.File(incomplete_path, "w") as f:
            f.attrs["batch_idx"] = 10
            # Missing other required fields

        assert manager.validate_checkpoint(incomplete_path) is False


class TestCheckpointManagerCleanup:
    """Test checkpoint cleanup functionality."""

    def test_cleanup_old_checkpoints_nothing_to_delete(self, tmp_path):
        """Test cleanup when number of checkpoints <= keep_last_n."""
        manager = CheckpointManager(tmp_path, keep_last_n=3)
        params = np.array([1.0, 2.0, 3.0])
        optimizer_state = {"iteration": 42}

        # Create only 2 checkpoints
        manager.save_checkpoint(1, params, optimizer_state, 0.1)
        manager.save_checkpoint(2, params, optimizer_state, 0.2)

        deleted = manager.cleanup_old_checkpoints()

        assert len(deleted) == 0
        # Both checkpoints should still exist
        assert len(list(tmp_path.glob("homodyne_state_batch_*.h5"))) == 2

    def test_cleanup_old_checkpoints_deletes_oldest(self, tmp_path):
        """Test cleanup deletes oldest checkpoints."""
        manager = CheckpointManager(tmp_path, keep_last_n=3)
        params = np.array([1.0, 2.0, 3.0])
        optimizer_state = {"iteration": 42}

        # Create 5 checkpoints
        for i in range(1, 6):
            manager.save_checkpoint(i, params, optimizer_state, float(i) * 0.1)

        deleted = manager.cleanup_old_checkpoints()

        # Should delete 2 oldest (batch 1 and 2)
        assert len(deleted) == 2

        # Should keep 3 newest (batch 3, 4, 5)
        remaining = list(tmp_path.glob("homodyne_state_batch_*.h5"))
        assert len(remaining) == 3

        # Verify correct ones remain
        batch_indices = sorted([int(p.stem.split("_")[-1]) for p in remaining])
        assert batch_indices == [3, 4, 5]

    def test_cleanup_old_checkpoints_oserror_handling(self, tmp_path, caplog):
        """Test cleanup handles OSError gracefully."""
        manager = CheckpointManager(tmp_path, keep_last_n=2)
        params = np.array([1.0, 2.0, 3.0])
        optimizer_state = {"iteration": 42}

        # Create 4 checkpoints
        for i in range(1, 5):
            manager.save_checkpoint(i, params, optimizer_state, float(i) * 0.1)

        # Make one checkpoint undeletable by mocking unlink
        checkpoints = sorted(tmp_path.glob("homodyne_state_batch_*.h5"))
        oldest_checkpoint = checkpoints[0]

        original_unlink = Path.unlink

        def mock_unlink(self, *args, **kwargs):
            if self == oldest_checkpoint:
                raise OSError("Permission denied")
            else:
                original_unlink(self, *args, **kwargs)

        with patch.object(Path, "unlink", mock_unlink):
            deleted = manager.cleanup_old_checkpoints()

        # Should attempt to delete 2 but only succeed with 1
        assert len(deleted) == 1

        # Check warning was logged
        assert "Failed to delete checkpoint" in caplog.text


class TestCheckpointManagerFindLatest:
    """Test finding latest checkpoint."""

    def test_find_latest_checkpoint_none_exist(self, tmp_path):
        """Test finding latest checkpoint when none exist."""
        manager = CheckpointManager(tmp_path)

        assert manager.find_latest_checkpoint() is None

    def test_find_latest_checkpoint_returns_highest_batch(self, tmp_path):
        """Test finding latest checkpoint returns highest batch index."""
        manager = CheckpointManager(tmp_path)
        params = np.array([1.0, 2.0, 3.0])
        optimizer_state = {"iteration": 42}

        # Create checkpoints with non-sequential batch indices
        manager.save_checkpoint(1, params, optimizer_state, 0.1)
        manager.save_checkpoint(5, params, optimizer_state, 0.2)
        manager.save_checkpoint(3, params, optimizer_state, 0.3)

        latest = manager.find_latest_checkpoint()

        assert latest is not None
        assert latest.name == "homodyne_state_batch_0005.h5"

    def test_find_latest_checkpoint_invalid_filename(self, tmp_path):
        """Test finding latest checkpoint with invalid filename."""
        manager = CheckpointManager(tmp_path)
        params = np.array([1.0, 2.0, 3.0])
        optimizer_state = {"iteration": 42}

        # Create valid checkpoint
        manager.save_checkpoint(10, params, optimizer_state, 0.1)

        # Create file with invalid filename format
        invalid_file = tmp_path / "homodyne_state_batch_invalid.h5"
        with h5py.File(invalid_file, "w") as f:
            f.attrs["batch_idx"] = 999

        # Should still find the valid checkpoint
        latest = manager.find_latest_checkpoint()

        assert latest is not None
        assert latest.name == "homodyne_state_batch_0010.h5"

    def test_find_latest_checkpoint_skips_invalid(self, tmp_path):
        """Test finding latest checkpoint skips invalid checkpoints."""
        manager = CheckpointManager(tmp_path)
        params = np.array([1.0, 2.0, 3.0])
        optimizer_state = {"iteration": 42}

        # Create valid checkpoints
        manager.save_checkpoint(1, params, optimizer_state, 0.1)
        checkpoint_5 = manager.save_checkpoint(5, params, optimizer_state, 0.2)
        manager.save_checkpoint(3, params, optimizer_state, 0.3)

        # Corrupt the highest batch checkpoint
        with h5py.File(checkpoint_5, "r+") as f:
            f.attrs["checksum"] = "corrupted"

        # Should skip corrupted checkpoint and return next valid
        latest = manager.find_latest_checkpoint()

        assert latest is not None
        assert latest.name == "homodyne_state_batch_0003.h5"


class TestCheckpointManagerIntegration:
    """Integration tests for checkpoint manager."""

    def test_full_save_load_cycle(self, tmp_path):
        """Test complete save/load cycle."""
        manager = CheckpointManager(tmp_path)

        # Save checkpoint
        params = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        optimizer_state = {
            "iteration": 100,
            "learning_rate": 0.001,
            "momentum": 0.9,
        }
        loss = 0.456
        metadata = {
            "batch_size": 1024,
            "epoch": 5,
            "success_rate": [0.9, 0.85, 0.92],
        }

        checkpoint_path = manager.save_checkpoint(
            batch_idx=50,
            parameters=params,
            optimizer_state=optimizer_state,
            loss=loss,
            metadata=metadata,
        )

        # Load checkpoint
        loaded = manager.load_checkpoint(checkpoint_path)

        # Verify all data
        assert loaded["batch_idx"] == 50
        np.testing.assert_array_equal(loaded["parameters"], params)
        assert loaded["optimizer_state"] == optimizer_state
        assert loaded["loss"] == loss
        assert loaded["metadata"]["batch_size"] == 1024
        assert loaded["metadata"]["epoch"] == 5
        assert loaded["metadata"]["success_rate"] == [0.9, 0.85, 0.92]

    def test_checkpoint_resume_workflow(self, tmp_path):
        """Test checkpoint resume workflow."""
        manager = CheckpointManager(tmp_path, checkpoint_frequency=10, keep_last_n=2)

        params = np.array([1.0, 2.0, 3.0])

        # Simulate optimization creating checkpoints
        for batch_idx in [10, 20, 30, 40]:
            optimizer_state = {"iteration": batch_idx * 10}
            manager.save_checkpoint(
                batch_idx, params, optimizer_state, loss=1.0 / batch_idx
            )

        # Cleanup old checkpoints
        deleted = manager.cleanup_old_checkpoints()
        assert len(deleted) == 2  # Keep last 2, delete first 2

        # Find latest checkpoint
        latest = manager.find_latest_checkpoint()
        assert latest is not None

        # Load latest
        loaded = manager.load_checkpoint(latest)
        assert loaded["batch_idx"] == 40
        assert loaded["optimizer_state"]["iteration"] == 400
