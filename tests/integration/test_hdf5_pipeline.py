"""Integration Tests for HDF5 Pipeline
======================================

End-to-end tests: synthetic HDF5 -> XPCSDataLoader -> data validation.

Tests cover:
- Synthetic HDF5 generation in APS old format
- Loading via XPCSDataLoader with config dict
- Verifying loaded data structure and integrity
- Round-trip data integrity checks

These tests require h5py and are marked as integration tests.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

h5py = pytest.importorskip("h5py")

from homodyne.data.xpcs_loader import (  # noqa: E402
    XPCSConfigurationError,
    XPCSDataLoader,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_synthetic_hdf5_aps_old(
    path: Path,
    n_phi: int = 5,
    n_frames: int = 50,
    q_value: float = 0.0054,
) -> dict[str, np.ndarray]:
    """Write a minimal APS-old-format HDF5 file and return ground-truth arrays.

    APS old format structure:
        xpcs/dqlist        (1, N)  -- wavevector magnitudes
        xpcs/dphilist      (1, N)  -- phi angles in degrees
        exchange/C2T_all/  -- group with one key per (q, phi) pair
            c2_00001       (n_frames, n_frames) upper-triangle half-matrix
    """
    rng = np.random.default_rng(42)

    # All entries share the same q value (typical for a single q-ring)
    dqlist = np.full(n_phi, q_value)
    dphilist = np.linspace(0.0, 180.0, n_phi)

    # Generate realistic-looking half-matrices (upper triangular)
    half_matrices = []
    for _ in range(n_phi):
        full = 1.0 + 0.3 * np.exp(
            -0.01 * np.abs(
                np.arange(n_frames)[:, None] - np.arange(n_frames)[None, :]
            )
        )
        full += rng.normal(0, 0.005, full.shape)
        # Store as upper triangle (APS convention)
        half = np.triu(full)
        half_matrices.append(half)

    with h5py.File(path, "w") as f:
        xpcs = f.create_group("xpcs")
        xpcs.create_dataset("dqlist", data=dqlist.reshape(1, -1))
        xpcs.create_dataset("dphilist", data=dphilist.reshape(1, -1))

        exchange = f.create_group("exchange")
        c2t = exchange.create_group("C2T_all")
        for i, hm in enumerate(half_matrices):
            c2t.create_dataset(f"c2_{i + 1:05d}", data=hm)

    return {
        "dqlist": dqlist,
        "dphilist": dphilist,
        "half_matrices": np.array(half_matrices),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_hdf5_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture()
def synthetic_hdf5(tmp_hdf5_dir: Path) -> tuple[Path, dict[str, np.ndarray]]:
    """Create a synthetic APS-old HDF5 and return (path, ground_truth)."""
    hdf5_path = tmp_hdf5_dir / "synthetic_experiment.hdf"
    ground_truth = _create_synthetic_hdf5_aps_old(hdf5_path)
    return hdf5_path, ground_truth


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestHDF5ToNLSQStaticMode:
    """Verify the HDF5 -> XPCSDataLoader path for static mode."""

    def test_loader_detects_aps_old_format(
        self,
        synthetic_hdf5: tuple[Path, dict[str, np.ndarray]],
    ) -> None:
        """XPCSDataLoader should auto-detect APS old format."""
        hdf5_path, _ = synthetic_hdf5

        config_dict = {
            "experimental_data": {
                "data_folder_path": str(hdf5_path.parent),
                "data_file_name": hdf5_path.name,
            },
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": 50,
            },
        }

        loader = XPCSDataLoader(config_dict=config_dict)
        fmt = loader._detect_format(str(hdf5_path))
        assert fmt == "aps_old", f"Expected 'aps_old', got '{fmt}'"

    def test_loaded_data_has_required_keys(
        self,
        synthetic_hdf5: tuple[Path, dict[str, np.ndarray]],
    ) -> None:
        """Loaded data dict must contain all keys consumed by NLSQ."""
        hdf5_path, _ = synthetic_hdf5

        config_dict = {
            "experimental_data": {
                "data_folder_path": str(hdf5_path.parent),
                "data_file_name": hdf5_path.name,
            },
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": 50,
                "scattering": {"wavevector_q": 0.0054},
            },
        }

        loader = XPCSDataLoader(config_dict=config_dict)
        data = loader.load_experimental_data()

        required_keys = {
            "wavevector_q_list",
            "phi_angles_list",
            "t1",
            "t2",
            "c2_exp",
        }
        assert required_keys.issubset(
            data.keys()
        ), f"Missing keys: {required_keys - data.keys()}"

    def test_loaded_shapes_are_consistent(
        self,
        synthetic_hdf5: tuple[Path, dict[str, np.ndarray]],
    ) -> None:
        """Array dimensions must be internally consistent."""
        hdf5_path, _ = synthetic_hdf5
        n_frames = 50

        config_dict = {
            "experimental_data": {
                "data_folder_path": str(hdf5_path.parent),
                "data_file_name": hdf5_path.name,
            },
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": n_frames,
                "scattering": {"wavevector_q": 0.0054},
            },
        }

        loader = XPCSDataLoader(config_dict=config_dict)
        data = loader.load_experimental_data()

        n_phi_loaded = len(data["phi_angles_list"])
        c2_shape = np.asarray(data["c2_exp"]).shape

        # c2_exp should be (n_phi, n_t, n_t)
        assert c2_shape[0] == n_phi_loaded, (
            f"c2_exp leading dim {c2_shape[0]} != n_phi {n_phi_loaded}"
        )
        assert c2_shape[1] == c2_shape[2], "c2_exp time dims must be square"

        # t1 and t2 should be 1-D with length == time dim of c2_exp
        t1 = np.asarray(data["t1"])
        t2 = np.asarray(data["t2"])
        assert t1.ndim == 1, f"t1 should be 1-D, got {t1.ndim}-D"
        assert t2.ndim == 1, f"t2 should be 1-D, got {t2.ndim}-D"
        assert len(t1) == c2_shape[1], "t1 length must match c2_exp time dim"

    def test_time_arrays_start_at_zero(
        self,
        synthetic_hdf5: tuple[Path, dict[str, np.ndarray]],
    ) -> None:
        """Per loader contract, 1-D time arrays start at t=0."""
        hdf5_path, _ = synthetic_hdf5

        config_dict = {
            "experimental_data": {
                "data_folder_path": str(hdf5_path.parent),
                "data_file_name": hdf5_path.name,
            },
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": 50,
                "scattering": {"wavevector_q": 0.0054},
            },
        }

        loader = XPCSDataLoader(config_dict=config_dict)
        data = loader.load_experimental_data()

        t1 = np.asarray(data["t1"])
        assert_allclose(t1[0], 0.0, atol=1e-12, err_msg="t1 must start at 0")

    def test_c2_values_are_finite(
        self,
        synthetic_hdf5: tuple[Path, dict[str, np.ndarray]],
    ) -> None:
        """No NaN or Inf should appear in loaded correlation data."""
        hdf5_path, _ = synthetic_hdf5

        config_dict = {
            "experimental_data": {
                "data_folder_path": str(hdf5_path.parent),
                "data_file_name": hdf5_path.name,
            },
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": 50,
                "scattering": {"wavevector_q": 0.0054},
            },
        }

        loader = XPCSDataLoader(config_dict=config_dict)
        data = loader.load_experimental_data()
        c2 = np.asarray(data["c2_exp"])
        assert np.all(np.isfinite(c2)), "c2_exp contains NaN or Inf values"


@pytest.mark.integration
class TestHDF5RoundTripDataIntegrity:
    """Verify that data written to HDF5 survives the load round-trip."""

    def test_phi_angles_preserved(
        self,
        synthetic_hdf5: tuple[Path, dict[str, np.ndarray]],
    ) -> None:
        """Phi angles loaded must match those written to HDF5."""
        hdf5_path, ground_truth = synthetic_hdf5

        config_dict = {
            "experimental_data": {
                "data_folder_path": str(hdf5_path.parent),
                "data_file_name": hdf5_path.name,
            },
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": 50,
                "scattering": {"wavevector_q": 0.0054},
            },
        }

        loader = XPCSDataLoader(config_dict=config_dict)
        data = loader.load_experimental_data()

        loaded_phi = np.asarray(data["phi_angles_list"])
        expected_phi = ground_truth["dphilist"]

        assert_allclose(
            np.sort(loaded_phi),
            np.sort(expected_phi),
            atol=1e-6,
            err_msg="Phi angles differ after round-trip",
        )

    def test_q_values_preserved(
        self,
        synthetic_hdf5: tuple[Path, dict[str, np.ndarray]],
    ) -> None:
        """Wavevector q values loaded must match those written."""
        hdf5_path, ground_truth = synthetic_hdf5

        config_dict = {
            "experimental_data": {
                "data_folder_path": str(hdf5_path.parent),
                "data_file_name": hdf5_path.name,
            },
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": 50,
                "scattering": {"wavevector_q": 0.0054},
            },
        }

        loader = XPCSDataLoader(config_dict=config_dict)
        data = loader.load_experimental_data()

        loaded_q = np.asarray(data["wavevector_q_list"])
        expected_q = ground_truth["dqlist"]

        # Loaded q should be a subset or equal to expected
        for q_val in loaded_q:
            assert np.any(
                np.abs(expected_q - q_val) < 1e-8
            ), f"Loaded q={q_val} not found in ground truth"

    def test_correlation_matrix_symmetry(
        self,
        synthetic_hdf5: tuple[Path, dict[str, np.ndarray]],
    ) -> None:
        """After half-matrix reconstruction, c2 matrices should be symmetric."""
        hdf5_path, _ = synthetic_hdf5

        config_dict = {
            "experimental_data": {
                "data_folder_path": str(hdf5_path.parent),
                "data_file_name": hdf5_path.name,
            },
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": 50,
                "scattering": {"wavevector_q": 0.0054},
            },
        }

        loader = XPCSDataLoader(config_dict=config_dict)
        data = loader.load_experimental_data()

        c2 = np.asarray(data["c2_exp"])
        for i in range(c2.shape[0]):
            # Diagonal correction may break exact symmetry, but off-diag
            # elements should still be symmetric.
            mat = c2[i]
            # Check off-diagonal symmetry only
            n = mat.shape[0]
            for row in range(n):
                for col in range(row + 1, n):
                    assert_allclose(
                        mat[row, col],
                        mat[col, row],
                        atol=1e-10,
                        err_msg=(
                            f"Asymmetry at phi_idx={i}, "
                            f"({row},{col}) vs ({col},{row})"
                        ),
                    )

    def test_missing_hdf5_file_raises(self, tmp_hdf5_dir: Path) -> None:
        """Attempting to load a non-existent HDF5 must raise FileNotFoundError."""
        config_dict = {
            "experimental_data": {
                "data_folder_path": str(tmp_hdf5_dir),
                "data_file_name": "nonexistent.hdf",
            },
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": 50,
            },
        }

        loader = XPCSDataLoader(config_dict=config_dict)
        with pytest.raises(FileNotFoundError):
            loader.load_experimental_data()

    def test_invalid_config_raises(self) -> None:
        """Missing required config keys must raise XPCSConfigurationError."""
        config_dict = {
            "experimental_data": {
                # Missing data_folder_path and data_file_name
            },
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": 50,
            },
        }

        with pytest.raises(XPCSConfigurationError):
            XPCSDataLoader(config_dict=config_dict)
