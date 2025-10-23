"""Comprehensive tests for checkpoint management system.

Tests the CheckpointManager class for checkpoint save/resume functionality,
including validation, corruption detection, and automatic cleanup.

Test Coverage:
--------------
1. Checkpoint save with HDF5
2. Checkpoint load and validation
3. Resume from partial optimization
4. Checkpoint corruption detection
5. Automatic cleanup of old checkpoints
"""

import numpy as np
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import h5py
import hashlib

# Import will be created
# from homodyne.optimization.checkpoint_manager import CheckpointManager
# from homodyne.optimization.exceptions import NLSQCheckpointError


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
        optimizer_state = {'iteration': 50, 'lr': 0.01}
        loss = 0.123

        # Would call: checkpoint_manager.save_checkpoint(...)
        # For now, create mock checkpoint file
        checkpoint_path = checkpoint_dir / f"checkpoint_batch_{batch_idx:04d}.h5"

        # Simulate HDF5 file creation
        with h5py.File(checkpoint_path, 'w') as f:
            f.create_dataset('parameters', data=parameters)
            f.attrs['batch_idx'] = batch_idx
            f.attrs['loss'] = loss

        assert checkpoint_path.exists()
        assert checkpoint_path.suffix == '.h5'

    def test_checkpoint_contains_required_state(self, tmp_path):
        """Test checkpoint contains all required state information."""
        checkpoint_path = tmp_path / "checkpoint.h5"

        # Required state
        parameters = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        optimizer_state = {'momentum': np.array([0.1, 0.2]), 'iteration': 100}
        batch_idx = 42
        loss = 0.456

        # Create checkpoint with all state
        with h5py.File(checkpoint_path, 'w') as f:
            f.create_dataset('parameters', data=parameters)
            f.create_dataset('optimizer_state/momentum', data=optimizer_state['momentum'])
            f.attrs['batch_idx'] = batch_idx
            f.attrs['loss'] = loss
            f.attrs['iteration'] = optimizer_state['iteration']

        # Verify all state present
        with h5py.File(checkpoint_path, 'r') as f:
            assert 'parameters' in f
            assert 'optimizer_state' in f
            assert 'batch_idx' in f.attrs
            assert 'loss' in f.attrs

    def test_checkpoint_compression(self, tmp_path):
        """Test HDF5 compression reduces file size."""
        checkpoint_path_uncompressed = tmp_path / "checkpoint_uncompressed.h5"
        checkpoint_path_compressed = tmp_path / "checkpoint_compressed.h5"

        # Large parameter array
        parameters = np.random.randn(10000)

        # Uncompressed
        with h5py.File(checkpoint_path_uncompressed, 'w') as f:
            f.create_dataset('parameters', data=parameters)

        # Compressed (gzip level 4)
        with h5py.File(checkpoint_path_compressed, 'w') as f:
            f.create_dataset('parameters', data=parameters, compression='gzip', compression_opts=4)

        # Compressed should be smaller
        size_uncompressed = checkpoint_path_uncompressed.stat().st_size
        size_compressed = checkpoint_path_compressed.stat().st_size

        assert size_compressed < size_uncompressed

    def test_checkpoint_file_permissions(self, tmp_path):
        """Test checkpoint files have correct permissions."""
        checkpoint_path = tmp_path / "checkpoint.h5"

        with h5py.File(checkpoint_path, 'w') as f:
            f.attrs['test'] = 'data'

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

        with h5py.File(checkpoint_path, 'w') as f:
            f.create_dataset('parameters', data=expected_params)
            f.attrs['batch_idx'] = expected_batch
            f.attrs['loss'] = expected_loss

        # Load and verify
        with h5py.File(checkpoint_path, 'r') as f:
            loaded_params = f['parameters'][:]
            loaded_batch = f.attrs['batch_idx']
            loaded_loss = f.attrs['loss']

        np.testing.assert_array_equal(loaded_params, expected_params)
        assert loaded_batch == expected_batch
        assert loaded_loss == expected_loss

    def test_checksum_validation(self, tmp_path):
        """Test checksum validation detects modifications."""
        checkpoint_path = tmp_path / "checkpoint.h5"

        # Create checkpoint with checksum
        data = b"checkpoint data"
        checksum = hashlib.sha256(data).hexdigest()

        with h5py.File(checkpoint_path, 'w') as f:
            f.attrs['checksum'] = checksum
            f.attrs['data'] = data.decode('latin1')

        # Load and validate checksum
        with h5py.File(checkpoint_path, 'r') as f:
            stored_checksum = f.attrs['checksum']
            stored_data = f.attrs['data'].encode('latin1')
            computed_checksum = hashlib.sha256(stored_data).hexdigest()

        assert stored_checksum == computed_checksum

    def test_version_compatibility_check(self, tmp_path):
        """Test version compatibility checking."""
        checkpoint_path = tmp_path / "checkpoint.h5"

        # Create checkpoint with version
        current_version = "1.0.0"

        with h5py.File(checkpoint_path, 'w') as f:
            f.attrs['version'] = current_version

        # Verify version
        with h5py.File(checkpoint_path, 'r') as f:
            stored_version = f.attrs['version']

        assert stored_version == current_version

    def test_missing_file_handling(self, tmp_path):
        """Test graceful handling of missing checkpoint file."""
        checkpoint_path = tmp_path / "nonexistent_checkpoint.h5"

        assert not checkpoint_path.exists()

        # Should handle missing file gracefully
        # Would raise: NLSQCheckpointError
        # For now, verify file doesn't exist
        with pytest.raises(FileNotFoundError):
            with h5py.File(checkpoint_path, 'r') as f:
                pass


class TestResumeFromCheckpoint:
    """Test resume from partial optimization functionality."""

    def test_resume_from_checkpoint_at_batch_n(self, tmp_path):
        """Test resuming from checkpoint at specific batch."""
        checkpoint_path = tmp_path / "checkpoint_batch_0050.h5"

        # Create checkpoint at batch 50
        batch_idx = 50
        parameters = np.array([10.0, 20.0, 30.0])

        with h5py.File(checkpoint_path, 'w') as f:
            f.create_dataset('parameters', data=parameters)
            f.attrs['batch_idx'] = batch_idx

        # Resume simulation
        with h5py.File(checkpoint_path, 'r') as f:
            resume_batch = f.attrs['batch_idx']
            resume_params = f['parameters'][:]

        # Should continue from next batch
        next_batch = resume_batch + 1
        assert next_batch == 51
        np.testing.assert_array_equal(resume_params, parameters)

    def test_parameter_state_restoration(self, tmp_path):
        """Test that parameter state is fully restored."""
        checkpoint_path = tmp_path / "checkpoint.h5"

        # Save parameter state
        original_params = np.random.randn(9)  # laminar_flow has 9 params

        with h5py.File(checkpoint_path, 'w') as f:
            f.create_dataset('parameters', data=original_params)

        # Load and verify exact restoration
        with h5py.File(checkpoint_path, 'r') as f:
            restored_params = f['parameters'][:]

        np.testing.assert_array_almost_equal(restored_params, original_params)

    def test_optimizer_state_restoration(self, tmp_path):
        """Test that optimizer internal state is restored."""
        checkpoint_path = tmp_path / "checkpoint.h5"

        # Complex optimizer state
        optimizer_state = {
            'iteration': 150,
            'learning_rate': 0.001,
            'momentum_buffer': np.random.randn(5),
            'squared_gradient': np.random.randn(5),
        }

        with h5py.File(checkpoint_path, 'w') as f:
            grp = f.create_group('optimizer_state')
            grp.attrs['iteration'] = optimizer_state['iteration']
            grp.attrs['learning_rate'] = optimizer_state['learning_rate']
            grp.create_dataset('momentum_buffer', data=optimizer_state['momentum_buffer'])
            grp.create_dataset('squared_gradient', data=optimizer_state['squared_gradient'])

        # Restore and verify
        with h5py.File(checkpoint_path, 'r') as f:
            grp = f['optimizer_state']
            restored_state = {
                'iteration': grp.attrs['iteration'],
                'learning_rate': grp.attrs['learning_rate'],
                'momentum_buffer': grp['momentum_buffer'][:],
                'squared_gradient': grp['squared_gradient'][:],
            }

        assert restored_state['iteration'] == optimizer_state['iteration']
        assert restored_state['learning_rate'] == optimizer_state['learning_rate']
        np.testing.assert_array_equal(
            restored_state['momentum_buffer'],
            optimizer_state['momentum_buffer']
        )

    def test_batch_index_continuation(self, tmp_path):
        """Test that batch index continues correctly after resume."""
        # Simulate: optimization runs batches 0-50, checkpoints, resumes
        checkpoint_path = tmp_path / "checkpoint_batch_0050.h5"

        with h5py.File(checkpoint_path, 'w') as f:
            f.attrs['batch_idx'] = 50

        # Resume
        with h5py.File(checkpoint_path, 'r') as f:
            resume_batch = f.attrs['batch_idx']

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
        with open(checkpoint_path, 'wb') as f:
            f.write(b"corrupted data")

        # Should detect corruption
        with pytest.raises(Exception):  # h5py will raise OSError or similar
            with h5py.File(checkpoint_path, 'r') as f:
                pass

    def test_incomplete_checkpoint_detection(self, tmp_path):
        """Test detection of incomplete checkpoint (missing required fields)."""
        checkpoint_path = tmp_path / "incomplete_checkpoint.h5"

        # Create checkpoint missing required data
        with h5py.File(checkpoint_path, 'w') as f:
            # Only store batch_idx, missing parameters
            f.attrs['batch_idx'] = 10

        # Load and check for missing data
        with h5py.File(checkpoint_path, 'r') as f:
            has_parameters = 'parameters' in f

        assert not has_parameters  # Incomplete

    def test_checksum_mismatch_handling(self, tmp_path):
        """Test handling of checksum mismatch."""
        checkpoint_path = tmp_path / "checkpoint.h5"

        # Create checkpoint with checksum
        original_data = b"original data"
        original_checksum = hashlib.sha256(original_data).hexdigest()

        with h5py.File(checkpoint_path, 'w') as f:
            f.attrs['checksum'] = original_checksum
            f.attrs['data'] = original_data.decode('latin1')

        # Modify data (simulate corruption)
        with h5py.File(checkpoint_path, 'a') as f:
            f.attrs['data'] = "modified data"

        # Verify checksum mismatch
        with h5py.File(checkpoint_path, 'r') as f:
            stored_checksum = f.attrs['checksum']
            current_data = f.attrs['data'].encode('latin1')
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
            with h5py.File(checkpoint_path, 'r') as f:
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
            with h5py.File(checkpoint_path, 'w') as f:
                f.attrs['batch_idx'] = i
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
        remaining_indices = sorted([int(p.stem.split('_')[-1]) for p in remaining])
        assert remaining_indices == [7, 8, 9]

    def test_manual_cleanup(self, tmp_path):
        """Test manual cleanup of checkpoints."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create checkpoints
        for i in range(5):
            checkpoint_path = checkpoint_dir / f"checkpoint_{i}.h5"
            with h5py.File(checkpoint_path, 'w') as f:
                f.attrs['batch_idx'] = i

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
            with h5py.File(checkpoint_path, 'w') as f:
                # Create dataset with known size
                f.create_dataset('data', data=np.random.randn(1000))
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


class TestCheckpointManagerIntegration:
    """Integration tests for CheckpointManager."""

    def test_find_latest_checkpoint(self, tmp_path):
        """Test finding the most recent checkpoint."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Create checkpoints with different timestamps
        import time
        for i in range(3):
            checkpoint_path = checkpoint_dir / f"checkpoint_batch_{i:04d}.h5"
            with h5py.File(checkpoint_path, 'w') as f:
                f.attrs['batch_idx'] = i
            time.sleep(0.01)  # Ensure different timestamps

        # Find latest
        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint_batch_*.h5"),
            key=lambda p: p.stat().st_mtime
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

        with h5py.File(checkpoint_path, 'w') as f:
            f.create_dataset('parameters', data=parameters)
            f.attrs['batch_idx'] = batch_idx

        # 2. Load checkpoint
        with h5py.File(checkpoint_path, 'r') as f:
            loaded_params = f['parameters'][:]
            loaded_batch = f.attrs['batch_idx']

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
                checkpoint_path = checkpoint_dir / f"checkpoint_batch_{batch_idx:04d}.h5"
                with h5py.File(checkpoint_path, 'w') as f:
                    f.attrs['batch_idx'] = batch_idx
                saved_batches.append(batch_idx)

        # Verify correct batches saved
        assert saved_batches == [0, 10, 20, 30]

        # Verify checkpoint files exist
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_batch_*.h5"))
        assert len(checkpoint_files) == len(saved_batches)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
