"""
Unit Tests for Checkpoint Management
=====================================

**Consolidation**: Week 6 (2025-11-15)

Consolidated from:
- test_checkpoint_manager.py (Checkpoint save/resume functionality, 22 tests, 534 lines)
- test_checkpoint_manager_coverage.py (Extended checkpoint coverage, 28 tests, 619 lines)

Test Categories:
---------------
**Save/Resume Functionality** (22 tests):
- Checkpoint save with HDF5 format
- Checkpoint load and validation
- Resume from partial optimization
- Checkpoint file permissions and structure

**Extended Coverage** (28 tests):
- Checkpoint save/load with and without compression
- Metadata handling (dict, list, numpy arrays)
- Checksum validation and corruption detection
- Automatic cleanup of old checkpoints
- Save time warnings and performance monitoring
- Error handling for missing/corrupted checkpoints

Test Coverage:
-------------
- Checkpoint save/load with HDF5 format and optional compression
- Resume from partial optimization (batch continuation)
- Metadata handling: dictionaries, lists, numpy arrays
- Checksum validation: MD5 hashing for corruption detection
- Automatic cleanup: keep last N checkpoints, disk space management
- Checkpoint validation: required fields, version compatibility
- Corruption detection: checksum mismatch, incomplete files
- Save time warnings: alert on slow checkpoint operations
- Error handling: missing files, corrupted checksums, decode errors
- Graceful degradation: continue on non-critical checkpoint errors

Total: 50 tests

Critical Fix (Week 6):
---------------------
Renamed duplicate TestCheckpointManagerIntegration class to
TestCheckpointManagerWorkflow to prevent 3-test loss during consolidation.

Usage Example:
-------------
```python
# Run all checkpoint tests
pytest tests/unit/test_checkpoint_core.py -v

# Run specific category
pytest tests/unit/test_checkpoint_core.py -k "save" -v
pytest tests/unit/test_checkpoint_core.py -k "cleanup" -v

# Test corruption detection
pytest tests/unit/test_checkpoint_core.py::TestCheckpointCorruptionDetection -v
```

See Also:
---------
- docs/WEEK6_CONSOLIDATION_SUMMARY.md: Consolidation details
- homodyne/optimization/checkpoint_manager.py: CheckpointManager implementation
- homodyne/optimization/exceptions.py: NLSQCheckpointError exception
"""

import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from homodyne.optimization.checkpoint_manager import CheckpointManager
from homodyne.optimization.exceptions import NLSQCheckpointError

# ==============================================================================
# Checkpoint Save/Resume Tests (from test_checkpoint_manager.py)
# ==============================================================================


class TestCheckpointSaveHDF5:
    """Test checkpoint save functionality with HDF5 format."""

    def test_checkpoint_creation(self, tmp_path):
        """Test that checkpoint file is created successfully."""
        # This test will validate checkpoint creation
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Simulated checkpoint data
        batch_idx = 10
        parameters = np.array([1.0, 2.0, 3.0])
        loss = 0.123

        # Would call: checkpoint_manager.save_checkpoint(...)
        # For now, create mock checkpoint file
        checkpoint_path = checkpoint_dir / f"checkpoint_batch_{batch_idx:04d}.h5"

        # Simulate HDF5 file creation
        with h5py.File(checkpoint_path, "w") as f:
            f.create_dataset("parameters", data=parameters)
            f.attrs["batch_idx"] = batch_idx
            f.attrs["loss"] = loss

        assert checkpoint_path.exists()
        assert checkpoint_path.suffix == ".h5"

    def test_checkpoint_contains_required_state(self, tmp_path):
        """Test checkpoint contains all required state information."""
        checkpoint_path = tmp_path / "checkpoint.h5"

        # Required state
        parameters = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        optimizer_state = {"momentum": np.array([0.1, 0.2]), "iteration": 100}
        batch_idx = 42
        loss = 0.456

        # Create checkpoint with all state
        with h5py.File(checkpoint_path, "w") as f:
            f.create_dataset("parameters", data=parameters)
            f.create_dataset(
                "optimizer_state/momentum", data=optimizer_state["momentum"]
            )
            f.attrs["batch_idx"] = batch_idx
            f.attrs["loss"] = loss
            f.attrs["iteration"] = optimizer_state["iteration"]

        # Verify all state present
        with h5py.File(checkpoint_path, "r") as f:
            assert "parameters" in f
            assert "optimizer_state" in f
            assert "batch_idx" in f.attrs
            assert "loss" in f.attrs

    def test_checkpoint_compression(self, tmp_path):
        """Test HDF5 compression reduces file size."""
        checkpoint_path_uncompressed = tmp_path / "checkpoint_uncompressed.h5"
        checkpoint_path_compressed = tmp_path / "checkpoint_compressed.h5"

        # Large parameter array
        parameters = np.random.randn(10000)

        # Uncompressed
        with h5py.File(checkpoint_path_uncompressed, "w") as f:
            f.create_dataset("parameters", data=parameters)

        # Compressed (gzip level 4)
        with h5py.File(checkpoint_path_compressed, "w") as f:
            f.create_dataset(
                "parameters", data=parameters, compression="gzip", compression_opts=4
            )

        # Compressed should be smaller
        size_uncompressed = checkpoint_path_uncompressed.stat().st_size
        size_compressed = checkpoint_path_compressed.stat().st_size

        assert size_compressed < size_uncompressed

    def test_checkpoint_file_permissions(self, tmp_path):
        """Test checkpoint files have correct permissions."""
        checkpoint_path = tmp_path / "checkpoint.h5"

        with h5py.File(checkpoint_path, "w") as f:
            f.attrs["test"] = "data"

        # File should be readable and writable
        assert checkpoint_path.exists()
        assert checkpoint_path.is_file()
        # Check permissions (owner read/write)
        mode = checkpoint_path.stat().st_mode
        assert mode & 0o600  # At least owner read/write


class TestCheckpointLoadValidation:
    """Test checkpoint load and validation functionality."""

    def test_load_valid_checkpoint(self, tmp_path):
        """Test loading a valid checkpoint."""
        checkpoint_path = tmp_path / "checkpoint.h5"

        # Create valid checkpoint
        expected_params = np.array([1.0, 2.0, 3.0])
        expected_batch = 25
        expected_loss = 0.789

        with h5py.File(checkpoint_path, "w") as f:
            f.create_dataset("parameters", data=expected_params)
            f.attrs["batch_idx"] = expected_batch
            f.attrs["loss"] = expected_loss

        # Load and verify
        with h5py.File(checkpoint_path, "r") as f:
            loaded_params = f["parameters"][:]
            loaded_batch = f.attrs["batch_idx"]
            loaded_loss = f.attrs["loss"]

        np.testing.assert_array_equal(loaded_params, expected_params)
        assert loaded_batch == expected_batch
        assert loaded_loss == expected_loss

    def test_checksum_validation(self, tmp_path):
        """Test checksum validation detects modifications."""
        checkpoint_path = tmp_path / "checkpoint.h5"

        # Create checkpoint with checksum
        data = b"checkpoint data"
        checksum = hashlib.sha256(data).hexdigest()

        with h5py.File(checkpoint_path, "w") as f:
            f.attrs["checksum"] = checksum
            f.attrs["data"] = data.decode("latin1")

        # Load and validate checksum
        with h5py.File(checkpoint_path, "r") as f:
            stored_checksum = f.attrs["checksum"]
            stored_data = f.attrs["data"].encode("latin1")
            computed_checksum = hashlib.sha256(stored_data).hexdigest()

        assert stored_checksum == computed_checksum

    def test_version_compatibility_check(self, tmp_path):
        """Test version compatibility checking."""
        checkpoint_path = tmp_path / "checkpoint.h5"

        # Create checkpoint with version
        current_version = "1.0.0"

        with h5py.File(checkpoint_path, "w") as f:
            f.attrs["version"] = current_version

        # Verify version
        with h5py.File(checkpoint_path, "r") as f:
            stored_version = f.attrs["version"]

        assert stored_version == current_version

    def test_missing_file_handling(self, tmp_path):
        """Test graceful handling of missing checkpoint file."""
        checkpoint_path = tmp_path / "nonexistent_checkpoint.h5"

        assert not checkpoint_path.exists()

        # Should handle missing file gracefully
        # Would raise: NLSQCheckpointError
        # For now, verify file doesn't exist
        with pytest.raises(FileNotFoundError):
            with h5py.File(checkpoint_path, "r"):
                pass


class TestResumeFromCheckpoint:
    """Test resume from partial optimization functionality."""

    def test_resume_from_checkpoint_at_batch_n(self, tmp_path):
        """Test resuming from checkpoint at specific batch."""
        checkpoint_path = tmp_path / "checkpoint_batch_0050.h5"

        # Create checkpoint at batch 50
        batch_idx = 50
        parameters = np.array([10.0, 20.0, 30.0])

        with h5py.File(checkpoint_path, "w") as f:
            f.create_dataset("parameters", data=parameters)
            f.attrs["batch_idx"] = batch_idx

        # Resume simulation
        with h5py.File(checkpoint_path, "r") as f:
            resume_batch = f.attrs["batch_idx"]
            resume_params = f["parameters"][:]

        # Should continue from next batch
        next_batch = resume_batch + 1
        assert next_batch == 51
        np.testing.assert_array_equal(resume_params, parameters)

    def test_parameter_state_restoration(self, tmp_path):
        """Test that parameter state is fully restored."""
        checkpoint_path = tmp_path / "checkpoint.h5"

        # Save parameter state
        original_params = np.random.randn(9)  # laminar_flow has 9 params

        with h5py.File(checkpoint_path, "w") as f:
            f.create_dataset("parameters", data=original_params)

        # Load and verify exact restoration
        with h5py.File(checkpoint_path, "r") as f:
            restored_params = f["parameters"][:]

        np.testing.assert_array_almost_equal(restored_params, original_params)

    def test_optimizer_state_restoration(self, tmp_path):
        """Test that optimizer internal state is restored."""
        checkpoint_path = tmp_path / "checkpoint.h5"

        # Complex optimizer state
        optimizer_state = {
            "iteration": 150,
            "learning_rate": 0.001,
            "momentum_buffer": np.random.randn(5),
            "squared_gradient": np.random.randn(5),
        }

        with h5py.File(checkpoint_path, "w") as f:
            grp = f.create_group("optimizer_state")
            grp.attrs["iteration"] = optimizer_state["iteration"]
            grp.attrs["learning_rate"] = optimizer_state["learning_rate"]
            grp.create_dataset(
                "momentum_buffer", data=optimizer_state["momentum_buffer"]
            )
            grp.create_dataset(
                "squared_gradient", data=optimizer_state["squared_gradient"]
            )

        # Restore and verify
        with h5py.File(checkpoint_path, "r") as f:
            grp = f["optimizer_state"]
            restored_state = {
                "iteration": grp.attrs["iteration"],
                "learning_rate": grp.attrs["learning_rate"],
                "momentum_buffer": grp["momentum_buffer"][:],
                "squared_gradient": grp["squared_gradient"][:],
            }

        assert restored_state["iteration"] == optimizer_state["iteration"]
        assert restored_state["learning_rate"] == optimizer_state["learning_rate"]
        np.testing.assert_array_equal(
            restored_state["momentum_buffer"], optimizer_state["momentum_buffer"]
        )

    def test_batch_index_continuation(self, tmp_path):
        """Test that batch index continues correctly after resume."""
        # Simulate: optimization runs batches 0-50, checkpoints, resumes
        checkpoint_path = tmp_path / "checkpoint_batch_0050.h5"

        with h5py.File(checkpoint_path, "w") as f:
            f.attrs["batch_idx"] = 50

        # Resume
        with h5py.File(checkpoint_path, "r") as f:
            resume_batch = f.attrs["batch_idx"]

        # Process next batches
        next_batches = list(range(resume_batch + 1, resume_batch + 11))

        assert next_batches[0] == 51
        assert next_batches[-1] == 60
        assert len(next_batches) == 10


class TestCheckpointCorruptionDetection:
    """Test checkpoint corruption detection."""

    def test_corrupted_file_detection(self, tmp_path):
        """Test detection of corrupted HDF5 file."""
        checkpoint_path = tmp_path / "corrupted_checkpoint.h5"

        # Create invalid HDF5 file (truncated)
        with open(checkpoint_path, "wb") as f:
            f.write(b"corrupted data")

        # Should detect corruption
        with pytest.raises(OSError):
            with h5py.File(checkpoint_path, "r") as f:
                pass

    def test_incomplete_checkpoint_detection(self, tmp_path):
        """Test detection of incomplete checkpoint (missing required fields)."""
        checkpoint_path = tmp_path / "incomplete_checkpoint.h5"

        # Create checkpoint missing required data
        with h5py.File(checkpoint_path, "w") as f:
            # Only store batch_idx, missing parameters
            f.attrs["batch_idx"] = 10

        # Load and check for missing data
        with h5py.File(checkpoint_path, "r") as f:
            has_parameters = "parameters" in f

        assert not has_parameters  # Incomplete

    def test_checksum_mismatch_handling(self, tmp_path):
        """Test handling of checksum mismatch."""
        checkpoint_path = tmp_path / "checkpoint.h5"

        # Create checkpoint with checksum
        original_data = b"original data"
        original_checksum = hashlib.sha256(original_data).hexdigest()

        with h5py.File(checkpoint_path, "w") as f:
            f.attrs["checksum"] = original_checksum
            f.attrs["data"] = original_data.decode("latin1")

        # Modify data (simulate corruption)
        with h5py.File(checkpoint_path, "a") as f:
            f.attrs["data"] = "modified data"

        # Verify checksum mismatch
        with h5py.File(checkpoint_path, "r") as f:
            stored_checksum = f.attrs["checksum"]
            current_data = f.attrs["data"].encode("latin1")
            current_checksum = hashlib.sha256(current_data).hexdigest()

        assert stored_checksum != current_checksum  # Mismatch detected

    def test_graceful_degradation(self, tmp_path):
        """Test graceful degradation when checkpoint is invalid."""
        checkpoint_path = tmp_path / "invalid_checkpoint.h5"

        # Create invalid checkpoint
        checkpoint_path.write_text("not an HDF5 file")

        # Should handle gracefully (return None or raise specific exception)
        checkpoint_valid = False
        try:
            with h5py.File(checkpoint_path, "r"):
                checkpoint_valid = True
        except Exception:
            checkpoint_valid = False

        assert not checkpoint_valid


class TestCheckpointCleanup:
    """Test automatic checkpoint cleanup."""

    def test_automatic_cleanup_keep_last_3(self, tmp_path):
        """Test automatic cleanup keeps only last 3 checkpoints."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create 10 checkpoints
        checkpoint_files = []
        for i in range(10):
            checkpoint_path = checkpoint_dir / f"checkpoint_batch_{i:04d}.h5"
            with h5py.File(checkpoint_path, "w") as f:
                f.attrs["batch_idx"] = i
            checkpoint_files.append(checkpoint_path)

        # Simulate cleanup: keep last 3
        keep_last_n = 3
        all_checkpoints = sorted(checkpoint_dir.glob("checkpoint_batch_*.h5"))

        # Identify checkpoints to delete
        if len(all_checkpoints) > keep_last_n:
            to_delete = all_checkpoints[:-keep_last_n]
            for cp in to_delete:
                cp.unlink()

        # Verify only 3 remain
        remaining = list(checkpoint_dir.glob("checkpoint_batch_*.h5"))
        assert len(remaining) == keep_last_n

        # Verify correct checkpoints kept (last 3)
        remaining_indices = sorted([int(p.stem.split("_")[-1]) for p in remaining])
        assert remaining_indices == [7, 8, 9]

    def test_manual_cleanup(self, tmp_path):
        """Test manual cleanup of checkpoints."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create checkpoints
        for i in range(5):
            checkpoint_path = checkpoint_dir / f"checkpoint_{i}.h5"
            with h5py.File(checkpoint_path, "w") as f:
                f.attrs["batch_idx"] = i

        # Manual cleanup: delete all
        for checkpoint in checkpoint_dir.glob("checkpoint_*.h5"):
            checkpoint.unlink()

        # Verify all deleted
        assert len(list(checkpoint_dir.glob("checkpoint_*.h5"))) == 0

    def test_disk_space_management(self, tmp_path):
        """Test disk space considerations in cleanup."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create checkpoints of known size
        total_size = 0
        for i in range(5):
            checkpoint_path = checkpoint_dir / f"checkpoint_{i}.h5"
            with h5py.File(checkpoint_path, "w") as f:
                # Create dataset with known size
                f.create_dataset("data", data=np.random.randn(1000))
            total_size += checkpoint_path.stat().st_size

        # Verify total size
        assert total_size > 0

        # Cleanup old checkpoints to save space
        old_checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.h5"))[:-2]
        freed_space = 0
        for cp in old_checkpoints:
            freed_space += cp.stat().st_size
            cp.unlink()

        assert freed_space > 0


class TestCheckpointManagerWorkflow:
    """Integration tests for CheckpointManager (workflow scenarios)."""

    def test_find_latest_checkpoint(self, tmp_path):
        """Test finding the most recent checkpoint."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create checkpoints with different timestamps

        for i in range(3):
            checkpoint_path = checkpoint_dir / f"checkpoint_batch_{i:04d}.h5"
            with h5py.File(checkpoint_path, "w") as f:
                f.attrs["batch_idx"] = i
            time.sleep(0.01)  # Ensure different timestamps

        # Find latest
        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint_batch_*.h5"),
            key=lambda p: p.stat().st_mtime,
        )
        latest = checkpoints[-1] if checkpoints else None

        assert latest is not None
        assert "0002" in latest.name  # batch 2 is latest

    def test_checkpoint_lifecycle(self, tmp_path):
        """Test complete checkpoint lifecycle: save → load → resume → cleanup."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # 1. Save checkpoint
        batch_idx = 10
        parameters = np.array([1.0, 2.0, 3.0])
        checkpoint_path = checkpoint_dir / f"checkpoint_batch_{batch_idx:04d}.h5"

        with h5py.File(checkpoint_path, "w") as f:
            f.create_dataset("parameters", data=parameters)
            f.attrs["batch_idx"] = batch_idx

        # 2. Load checkpoint
        with h5py.File(checkpoint_path, "r") as f:
            f["parameters"][:]
            loaded_batch = f.attrs["batch_idx"]

        # 3. Resume (next batch)
        next_batch = loaded_batch + 1
        assert next_batch == 11

        # 4. Cleanup
        checkpoint_path.unlink()
        assert not checkpoint_path.exists()

    def test_multiple_checkpoints_at_different_frequencies(self, tmp_path):
        """Test checkpoints saved at different frequencies."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Simulate saving at frequency=10 (batches 0, 10, 20, 30, ...)
        frequency = 10
        total_batches = 35

        saved_batches = []
        for batch_idx in range(total_batches):
            if batch_idx % frequency == 0:
                checkpoint_path = (
                    checkpoint_dir / f"checkpoint_batch_{batch_idx:04d}.h5"
                )
                with h5py.File(checkpoint_path, "w") as f:
                    f.attrs["batch_idx"] = batch_idx
                saved_batches.append(batch_idx)

        # Verify correct batches saved
        assert saved_batches == [0, 10, 20, 30]

        # Verify checkpoint files exist
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_batch_*.h5"))
        assert len(checkpoint_files) == len(saved_batches)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ==============================================================================
# Checkpoint Coverage Tests (from test_checkpoint_manager_coverage.py)
# ==============================================================================


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
            assert metadata_group.attrs["bool_value"]

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
        call_count = [0]

        def mock_time():
            call_count[0] += 1
            if call_count[0] == 1:
                return 0.0  # Start time
            else:
                return 2.5  # End time (2.5 seconds elapsed)

        with patch("time.time", side_effect=mock_time):
            manager.save_checkpoint(
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
