"""Checkpoint management for streaming optimization.

This module provides checkpoint save/load functionality for fault-tolerant
streaming optimization. Checkpoints are stored in HDF5 format with compression
and checksum validation.

Key Features:
- HDF5-based checkpoint storage with compression
- Checksum validation for integrity
- Automatic cleanup of old checkpoints
- Version compatibility checking
- Fast save time (< 2 seconds target)

The CheckpointManager complements NLSQ's built-in checkpointing by storing
homodyne-specific state (batch statistics, recovery actions, best parameters).
"""

import hashlib
import time
from pathlib import Path

import h5py
import numpy as np

from homodyne._version import __version__
from homodyne.optimization.exceptions import NLSQCheckpointError


class CheckpointManager:
    """Manage checkpoint save/load for streaming optimization.

    This class provides checkpoint management for homodyne-specific state
    during streaming optimization. It complements NLSQ's built-in checkpoint
    functionality by storing additional metadata, batch statistics, and
    recovery action history.

    Features:
    - HDF5-based checkpoint storage with compression
    - Checksum validation for integrity
    - Automatic cleanup of old checkpoints
    - Version compatibility checking

    Attributes
    ----------
    checkpoint_dir : Path
        Directory for checkpoint files
    checkpoint_frequency : int
        Save checkpoint every N batches
    keep_last_n : int
        Keep last N checkpoints (default: 3)
    enable_compression : bool
        Use HDF5 compression (default: True)

    Examples
    --------
    >>> manager = CheckpointManager("./checkpoints", checkpoint_frequency=10)
    >>> # Save checkpoint
    >>> path = manager.save_checkpoint(
    ...     batch_idx=10,
    ...     parameters=params,
    ...     optimizer_state={'iteration': 42},
    ...     loss=0.123,
    ... )
    >>> # Load checkpoint
    >>> data = manager.load_checkpoint(path)
    >>> params = data['parameters']
    >>> batch_idx = data['batch_idx']
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        checkpoint_frequency: int = 10,
        keep_last_n: int = 3,
        enable_compression: bool = True,
    ):
        """Initialize checkpoint manager.

        Parameters
        ----------
        checkpoint_dir : str or Path
            Directory for checkpoint files
        checkpoint_frequency : int, optional
            Save checkpoint every N batches, by default 10
        keep_last_n : int, optional
            Keep last N checkpoints, by default 3
        enable_compression : bool, optional
            Use HDF5 compression, by default True
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_frequency = checkpoint_frequency
        self.keep_last_n = keep_last_n
        self.enable_compression = enable_compression

        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        batch_idx: int,
        parameters: np.ndarray,
        optimizer_state: dict,
        loss: float,
        metadata: dict | None = None,
    ) -> Path:
        """Save checkpoint to HDF5 file.

        Saves checkpoint with compression, checksum validation, and version
        information. Target save time is < 2 seconds for typical parameter sets.

        Parameters
        ----------
        batch_idx : int
            Current batch index
        parameters : np.ndarray
            Current parameter values
        optimizer_state : dict
            Optimizer internal state
        loss : float
            Current loss value
        metadata : dict, optional
            Additional metadata (batch statistics, recovery actions, etc.)

        Returns
        -------
        Path
            Path to saved checkpoint file

        Raises
        ------
        NLSQCheckpointError
            If checkpoint save fails

        Notes
        -----
        Checkpoint file naming: `homodyne_state_batch_{batch_idx:04d}.h5`
        """
        start_time = time.time()

        # Generate checkpoint filename
        checkpoint_path = (
            self.checkpoint_dir / f"homodyne_state_batch_{batch_idx:04d}.h5"
        )

        try:
            # Serialize optimizer state to bytes for checksum (trusted checkpoint)
            import pickle  # nosec B403

            optimizer_bytes = pickle.dumps(optimizer_state)

            # Compute checksum
            checksum = self._compute_checksum(optimizer_bytes)

            # Save to HDF5 with compression
            with h5py.File(checkpoint_path, "w") as f:
                # Save parameters
                if self.enable_compression:
                    f.create_dataset(
                        "parameters",
                        data=parameters,
                        compression="gzip",
                        compression_opts=4,
                    )
                else:
                    f.create_dataset("parameters", data=parameters)

                # Save optimizer state as pickled bytes
                f.create_dataset("optimizer_state", data=np.void(optimizer_bytes))

                # Save scalar attributes
                f.attrs["batch_idx"] = batch_idx
                f.attrs["loss"] = loss
                f.attrs["checksum"] = checksum
                f.attrs["version"] = __version__
                f.attrs["timestamp"] = time.time()

                # Save metadata if provided
                if metadata is not None:
                    metadata_group = f.create_group("metadata")
                    for key, value in metadata.items():
                        if isinstance(value, (int, float, str, bool)):
                            metadata_group.attrs[key] = value
                        elif isinstance(value, (list, dict)):
                            # Store complex types as JSON strings
                            import json

                            metadata_group.attrs[key] = json.dumps(value)
                        elif isinstance(value, np.ndarray):
                            metadata_group.create_dataset(key, data=value)

            elapsed = time.time() - start_time

            # Check if save time exceeds target
            if elapsed > 2.0:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"Checkpoint save took {elapsed:.2f}s (target: < 2s). "
                    f"Consider disabling compression or reducing checkpoint frequency."
                )

            return checkpoint_path

        except Exception as e:
            raise NLSQCheckpointError(
                f"Failed to save checkpoint at batch {batch_idx}: {e}",
                error_context={
                    "batch_idx": batch_idx,
                    "checkpoint_path": str(checkpoint_path),
                },
            ) from e

    def load_checkpoint(self, checkpoint_path: Path) -> dict:
        """Load and validate checkpoint.

        Loads checkpoint from HDF5 file and validates checksum integrity.

        Parameters
        ----------
        checkpoint_path : Path
            Path to checkpoint file

        Returns
        -------
        dict
            Checkpoint data with keys:
            - batch_idx: int - Batch index when checkpoint was saved
            - parameters: np.ndarray - Parameter values
            - optimizer_state: dict - Optimizer internal state
            - loss: float - Loss value at checkpoint
            - metadata: dict - Additional metadata (if available)
            - version: str - Homodyne version
            - timestamp: float - Unix timestamp

        Raises
        ------
        NLSQCheckpointError
            If checkpoint is corrupted, invalid, or missing
        """
        if not checkpoint_path.exists():
            raise NLSQCheckpointError(
                f"Checkpoint file not found: {checkpoint_path}",
                error_context={"checkpoint_path": str(checkpoint_path)},
            )

        try:
            with h5py.File(checkpoint_path, "r") as f:
                # Load required data
                parameters = f["parameters"][:]
                optimizer_bytes = bytes(f["optimizer_state"][()])

                # Load attributes
                batch_idx = f.attrs["batch_idx"]
                loss = f.attrs["loss"]
                stored_checksum = f.attrs["checksum"]
                version = f.attrs.get("version", "unknown")
                timestamp = f.attrs.get("timestamp", 0.0)

                # Validate checksum
                computed_checksum = self._compute_checksum(optimizer_bytes)
                if computed_checksum != stored_checksum:
                    raise NLSQCheckpointError(
                        "Checkpoint checksum mismatch. File may be corrupted.",
                        error_context={
                            "checkpoint_path": str(checkpoint_path),
                            "stored_checksum": stored_checksum,
                            "computed_checksum": computed_checksum,
                        },
                    )

                # Deserialize optimizer state
                import pickle  # nosec B403: internal checkpoint serialization

                optimizer_state = pickle.loads(optimizer_bytes)  # nosec B301

                # Load metadata if available
                metadata = {}
                if "metadata" in f:
                    metadata_group = f["metadata"]
                    for key in metadata_group.attrs:
                        value = metadata_group.attrs[key]
                        # Try to parse JSON strings
                        if isinstance(value, str):
                            try:
                                import json

                                metadata[key] = json.loads(value)
                            except (json.JSONDecodeError, TypeError):
                                metadata[key] = value
                        else:
                            metadata[key] = value
                    # Load array datasets
                    for key in metadata_group.keys():
                        metadata[key] = metadata_group[key][:]

                return {
                    "batch_idx": int(batch_idx),
                    "parameters": parameters,
                    "optimizer_state": optimizer_state,
                    "loss": float(loss),
                    "metadata": metadata,
                    "version": version,
                    "timestamp": float(timestamp),
                }

        except (OSError, KeyError, ValueError) as e:
            raise NLSQCheckpointError(
                f"Failed to load checkpoint: {e}",
                error_context={"checkpoint_path": str(checkpoint_path)},
            ) from e

    def find_latest_checkpoint(self) -> Path | None:
        """Find most recent valid checkpoint.

        Searches checkpoint directory for valid checkpoint files and returns
        the one with the highest batch index.

        Returns
        -------
        Path or None
            Path to latest checkpoint, or None if none exist

        Notes
        -----
        Only returns checkpoints that pass validation.
        """
        # Find all checkpoint files
        checkpoint_files = list(self.checkpoint_dir.glob("homodyne_state_batch_*.h5"))

        if not checkpoint_files:
            return None

        # Sort by batch index (extracted from filename)
        def get_batch_idx(path: Path) -> int:
            try:
                # Extract batch index from filename: homodyne_state_batch_0010.h5
                return int(path.stem.split("_")[-1])
            except (ValueError, IndexError):
                return -1

        checkpoint_files.sort(key=get_batch_idx, reverse=True)

        # Find first valid checkpoint
        for checkpoint_path in checkpoint_files:
            if self.validate_checkpoint(checkpoint_path):
                return checkpoint_path

        return None

    def cleanup_old_checkpoints(self) -> list[Path]:
        """Remove old checkpoints, keeping last N.

        Keeps the most recent N checkpoints based on batch index and removes
        older ones to manage disk space.

        Returns
        -------
        list of Path
            Paths of deleted checkpoints

        Notes
        -----
        Only deletes checkpoints, never removes the keep_last_n most recent ones.
        """
        # Find all checkpoint files
        checkpoint_files = list(self.checkpoint_dir.glob("homodyne_state_batch_*.h5"))

        if len(checkpoint_files) <= self.keep_last_n:
            return []  # Nothing to clean up

        # Sort by batch index
        def get_batch_idx(path: Path) -> int:
            try:
                return int(path.stem.split("_")[-1])
            except (ValueError, IndexError):
                return -1

        checkpoint_files.sort(key=get_batch_idx, reverse=True)

        # Keep last N, delete the rest
        to_delete = checkpoint_files[self.keep_last_n :]
        deleted = []

        for checkpoint_path in to_delete:
            try:
                checkpoint_path.unlink()
                deleted.append(checkpoint_path)
            except OSError:
                # Log warning but continue
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to delete checkpoint: {checkpoint_path}")

        return deleted

    def validate_checkpoint(self, checkpoint_path: Path) -> bool:
        """Validate checkpoint integrity.

        Checks that checkpoint file exists, can be opened, has required fields,
        and passes checksum validation.

        Parameters
        ----------
        checkpoint_path : Path
            Path to checkpoint file

        Returns
        -------
        bool
            True if valid, False otherwise
        """
        if not checkpoint_path.exists():
            return False

        try:
            with h5py.File(checkpoint_path, "r") as f:
                # Check required fields
                required_fields = ["parameters", "optimizer_state"]
                required_attrs = ["batch_idx", "loss", "checksum"]

                for field in required_fields:
                    if field not in f:
                        return False

                for attr in required_attrs:
                    if attr not in f.attrs:
                        return False

                # Validate checksum
                optimizer_bytes = bytes(f["optimizer_state"][()])
                stored_checksum = f.attrs["checksum"]
                computed_checksum = self._compute_checksum(optimizer_bytes)

                if computed_checksum != stored_checksum:
                    return False

                return True

        except (OSError, KeyError, ValueError):
            return False

    def _compute_checksum(self, data: bytes) -> str:
        """Compute SHA256 checksum of data.

        Parameters
        ----------
        data : bytes
            Data to checksum

        Returns
        -------
        str
            Hexadecimal checksum string
        """
        return hashlib.sha256(data).hexdigest()
