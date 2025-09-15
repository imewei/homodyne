"""
Synthetic Test Data Generator for Homodyne v2 Testing
=====================================================

Generates realistic synthetic HDF5 datasets for comprehensive testing of the enhanced
data loading system. Supports both APS old and APS-U formats with configurable
characteristics for testing different scenarios.

Key Features:
- Generate datasets with various sizes (MB to GB)
- Simulate realistic XPCS correlation matrices
- Support both APS old and APS-U formats
- Generate datasets with different quality levels
- Create edge case scenarios (missing data, corruption, etc.)
- Memory-efficient generation for large datasets
"""

import os
import shutil
import tempfile
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

# JAX fallback
try:
    import jax.numpy as jnp
    from jax import random

    HAS_JAX = True
    jax_key = random.PRNGKey(42)
except ImportError:
    import numpy as jnp

    HAS_JAX = False
    jax_key = None


class DatasetSize(Enum):
    """Dataset size categories for testing"""

    TINY = "tiny"  # ~10MB - Quick unit tests
    SMALL = "small"  # ~100MB - Integration tests
    MEDIUM = "medium"  # ~500MB - Performance tests
    LARGE = "large"  # ~2GB - Scalability tests
    MASSIVE = "massive"  # ~10GB - Stress tests


class DataQuality(Enum):
    """Data quality levels for testing"""

    PERFECT = "perfect"  # No noise, perfect correlations
    GOOD = "good"  # Realistic noise levels
    NOISY = "noisy"  # Higher noise, some artifacts
    POOR = "poor"  # High noise, missing sections
    CORRUPTED = "corrupted"  # Intentional data corruption


class DatasetFormat(Enum):
    """HDF5 format types"""

    APS_OLD = "aps_old"
    APS_U = "aps_u"


@dataclass
class SyntheticDatasetConfig:
    """Configuration for synthetic dataset generation"""

    name: str
    size: DatasetSize = DatasetSize.SMALL
    quality: DataQuality = DataQuality.GOOD
    format: DatasetFormat = DatasetFormat.APS_U
    num_q: int = 50
    num_phi: int = 36
    num_frames: int = 10000
    num_delays: int = 100
    output_dir: Optional[Path] = None

    # Physics parameters for realistic data
    diffusion_coefficient: float = 1e-12  # m²/s
    temperature: float = 300  # K
    viscosity: float = 1e-3  # Pa·s

    # Data characteristics
    signal_to_noise: float = 10.0
    baseline_level: float = 1.0
    contrast: float = 0.8

    # Advanced options
    include_metadata: bool = True
    add_processing_artifacts: bool = False
    simulate_detector_issues: bool = False
    memory_efficient: bool = True


class SyntheticDataGenerator:
    """Generates synthetic HDF5 datasets for testing"""

    def __init__(self, config: SyntheticDatasetConfig):
        self.config = config
        self.output_dir = (
            config.output_dir or Path(tempfile.gettempdir()) / "homodyne_test_data"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate dataset parameters based on size
        self._adjust_parameters_for_size()

    def _adjust_parameters_for_size(self):
        """Adjust dataset parameters based on target size"""
        size_multipliers = {
            DatasetSize.TINY: 0.1,
            DatasetSize.SMALL: 1.0,
            DatasetSize.MEDIUM: 5.0,
            DatasetSize.LARGE: 20.0,
            DatasetSize.MASSIVE: 100.0,
        }

        multiplier = size_multipliers[self.config.size]

        # Adjust parameters to achieve target size
        self.config.num_frames = int(self.config.num_frames * multiplier)
        if multiplier < 1.0:  # For tiny datasets
            self.config.num_q = max(10, int(self.config.num_q * multiplier))
            self.config.num_phi = max(8, int(self.config.num_phi * multiplier))
            self.config.num_delays = max(20, int(self.config.num_delays * multiplier))

    def generate_realistic_correlation_matrix(
        self, q_value: float, phi_angle: float
    ) -> np.ndarray:
        """Generate realistic g2 correlation function"""
        # Time delays (logarithmic spacing)
        delays = np.logspace(-6, 1, self.config.num_delays)  # 1µs to 10s

        # Physical parameters
        D = self.config.diffusion_coefficient
        k_B = 1.380649e-23  # Boltzmann constant
        T = self.config.temperature

        # Calculate correlation function g1
        # For Brownian motion: g1(t) = exp(-Dq²t)
        gamma = D * (q_value**2)
        g1 = np.exp(-gamma * delays)

        # Add angular dependence for anisotropic systems
        if self.config.format == DatasetFormat.APS_U:
            anisotropy_factor = 1.0 + 0.3 * np.cos(2 * phi_angle)
            g1 = g1 * anisotropy_factor

        # Calculate g2 from Siegert relation: g2 = 1 + β|g1|²
        beta = self.config.contrast
        g2 = self.config.baseline_level + beta * (g1**2)

        # Add realistic noise based on quality
        noise_levels = {
            DataQuality.PERFECT: 0.0,
            DataQuality.GOOD: 0.05,
            DataQuality.NOISY: 0.15,
            DataQuality.POOR: 0.30,
            DataQuality.CORRUPTED: 0.50,
        }

        noise_level = noise_levels[self.config.quality]
        if noise_level > 0:
            if HAS_JAX and jax_key is not None:
                global jax_key
                jax_key, subkey = random.split(jax_key)
                noise = random.normal(subkey, shape=g2.shape) * noise_level
            else:
                noise = np.random.normal(0, noise_level, g2.shape)

            g2 = g2 + noise

            # Ensure g2 >= 1 (physical constraint)
            g2 = np.maximum(g2, 1.0)

        return g2, delays

    def add_data_artifacts(self, data: np.ndarray) -> np.ndarray:
        """Add realistic data artifacts based on quality level"""
        if self.config.quality == DataQuality.CORRUPTED:
            # Randomly corrupt some data points
            corrupt_fraction = 0.05
            corrupt_mask = np.random.random(data.shape) < corrupt_fraction
            data[corrupt_mask] = np.nan

            # Add some outliers
            outlier_mask = np.random.random(data.shape) < 0.01
            data[outlier_mask] = data[outlier_mask] * 10

        elif self.config.quality == DataQuality.POOR:
            # Add missing sections
            section_length = data.shape[-1] // 10
            start_idx = np.random.randint(0, data.shape[-1] - section_length)
            data[..., start_idx : start_idx + section_length] = np.nan

        if self.config.add_processing_artifacts:
            # Add diagonal artifacts (common in XPCS data)
            if data.ndim >= 2:
                for i in range(min(data.shape[0], data.shape[1])):
                    data[i, i] = data[i, i] * 1.1  # Slight diagonal enhancement

        if self.config.simulate_detector_issues:
            # Simulate dead pixels or bad detectors
            bad_detector_fraction = 0.02
            num_bad = int(data.shape[0] * bad_detector_fraction)
            bad_indices = np.random.choice(data.shape[0], num_bad, replace=False)
            data[bad_indices] = 0

        return data

    def create_aps_old_format(self, filepath: Path) -> None:
        """Create HDF5 file in APS old format"""
        with h5py.File(filepath, "w") as f:
            # Create main groups
            exchange_group = f.create_group("/exchange")

            # Q values
            q_values = np.logspace(-4, -2, self.config.num_q)  # Typical XPCS q range
            exchange_group.create_dataset("q", data=q_values)

            # Phi angles
            phi_angles = np.linspace(0, 2 * np.pi, self.config.num_phi, endpoint=False)
            exchange_group.create_dataset("phi", data=phi_angles)

            # Generate correlation matrices
            print(f"Generating correlation matrices for {self.config.name}...")

            if self.config.memory_efficient:
                # Create dataset and fill chunk by chunk
                corr_shape = (
                    self.config.num_q,
                    self.config.num_phi,
                    self.config.num_delays,
                )
                corr_dataset = exchange_group.create_dataset(
                    "g2",
                    shape=corr_shape,
                    dtype=np.float64,
                    chunks=True,
                    compression="gzip",
                )

                delays_dataset = exchange_group.create_dataset(
                    "delays", shape=(self.config.num_delays,), dtype=np.float64
                )

                # Generate data chunk by chunk
                for q_idx in range(self.config.num_q):
                    for phi_idx in range(self.config.num_phi):
                        g2, delays = self.generate_realistic_correlation_matrix(
                            q_values[q_idx], phi_angles[phi_idx]
                        )
                        corr_dataset[q_idx, phi_idx, :] = g2
                        if q_idx == 0 and phi_idx == 0:
                            delays_dataset[:] = delays

                    # Progress indicator
                    if (q_idx + 1) % 10 == 0:
                        print(f"  Generated {q_idx + 1}/{self.config.num_q} q-values")

            else:
                # Generate all data at once (for smaller datasets)
                corr_data = np.zeros(
                    (self.config.num_q, self.config.num_phi, self.config.num_delays)
                )

                for q_idx, q_val in enumerate(q_values):
                    for phi_idx, phi_val in enumerate(phi_angles):
                        g2, delays = self.generate_realistic_correlation_matrix(
                            q_val, phi_val
                        )
                        corr_data[q_idx, phi_idx, :] = g2

                # Add artifacts
                corr_data = self.add_data_artifacts(corr_data)

                # Save to HDF5
                exchange_group.create_dataset("g2", data=corr_data, compression="gzip")
                exchange_group.create_dataset("delays", data=delays)

            # Add metadata
            if self.config.include_metadata:
                self._add_metadata(f, "aps_old")

    def create_aps_u_format(self, filepath: Path) -> None:
        """Create HDF5 file in APS-U format (enhanced structure)"""
        with h5py.File(filepath, "w") as f:
            # Create enhanced group structure
            measurement_group = f.create_group("/measurement")
            analysis_group = f.create_group("/analysis")
            instrument_group = f.create_group("/instrument")

            # Q values with enhanced metadata
            q_values = np.logspace(-4, -2, self.config.num_q)
            q_dataset = measurement_group.create_dataset("q", data=q_values)
            q_dataset.attrs["units"] = "m^-1"
            q_dataset.attrs["long_name"] = "Scattering vector magnitude"

            # Phi angles with metadata
            phi_angles = np.linspace(0, 2 * np.pi, self.config.num_phi, endpoint=False)
            phi_dataset = measurement_group.create_dataset("phi", data=phi_angles)
            phi_dataset.attrs["units"] = "radians"
            phi_dataset.attrs["long_name"] = "Azimuthal angle"

            # Enhanced correlation data structure
            print(f"Generating enhanced correlation matrices for {self.config.name}...")

            # Create correlation matrix with enhanced structure
            corr_shape = (
                self.config.num_q,
                self.config.num_phi,
                self.config.num_delays,
            )

            g2_dataset = analysis_group.create_dataset(
                "correlation/g2",
                shape=corr_shape,
                dtype=np.float64,
                chunks=True,
                compression="gzip",
                shuffle=True,
            )
            g2_dataset.attrs["long_name"] = "Intensity correlation function g2"
            g2_dataset.attrs["units"] = "dimensionless"

            # Time delays with enhanced metadata
            delays_dataset = analysis_group.create_dataset(
                "correlation/delays", shape=(self.config.num_delays,), dtype=np.float64
            )
            delays_dataset.attrs["units"] = "seconds"
            delays_dataset.attrs["long_name"] = "Time delays"

            # Generate enhanced data
            for q_idx in range(self.config.num_q):
                for phi_idx in range(self.config.num_phi):
                    g2, delays = self.generate_realistic_correlation_matrix(
                        q_values[q_idx], phi_angles[phi_idx]
                    )
                    g2_dataset[q_idx, phi_idx, :] = g2
                    if q_idx == 0 and phi_idx == 0:
                        delays_dataset[:] = delays

                # Progress indicator
                if (q_idx + 1) % 10 == 0:
                    print(f"  Generated {q_idx + 1}/{self.config.num_q} q-values")

            # Add quality metrics
            quality_group = analysis_group.create_group("quality")
            self._add_quality_metrics(quality_group, g2_dataset)

            # Add instrument information
            self._add_instrument_info(instrument_group)

            # Add metadata
            if self.config.include_metadata:
                self._add_metadata(f, "aps_u")

    def _add_quality_metrics(self, group: h5py.Group, g2_data: h5py.Dataset) -> None:
        """Add quality metrics to the dataset"""
        # Calculate basic quality metrics
        # Note: For large datasets, we calculate these from a sample
        sample_data = g2_data[
            :: max(1, g2_data.shape[0] // 10), :: max(1, g2_data.shape[1] // 10), :
        ]

        signal_to_noise = self.config.signal_to_noise
        if self.config.quality == DataQuality.GOOD:
            signal_to_noise *= np.random.uniform(0.8, 1.2)
        elif self.config.quality == DataQuality.NOISY:
            signal_to_noise *= np.random.uniform(0.3, 0.7)
        elif self.config.quality == DataQuality.POOR:
            signal_to_noise *= np.random.uniform(0.1, 0.4)

        group.create_dataset("signal_to_noise_ratio", data=signal_to_noise)
        group.create_dataset(
            "data_completeness",
            data=0.95 if self.config.quality != DataQuality.CORRUPTED else 0.85,
        )
        group.create_dataset("baseline_stability", data=np.random.uniform(0.8, 1.0))

        # Add quality flags
        quality_flags = {
            DataQuality.PERFECT: 0,
            DataQuality.GOOD: 1,
            DataQuality.NOISY: 2,
            DataQuality.POOR: 3,
            DataQuality.CORRUPTED: 4,
        }
        group.create_dataset("quality_flag", data=quality_flags[self.config.quality])

    def _add_instrument_info(self, group: h5py.Group) -> None:
        """Add instrument information"""
        group.create_dataset("detector_name", data=b"Pilatus 1M")
        group.create_dataset("beam_energy", data=8.0)  # keV
        group.create_dataset("detector_distance", data=5.0)  # meters
        group.create_dataset("pixel_size", data=172e-6)  # meters

        group.attrs["facility"] = "Advanced Photon Source"
        group.attrs["beamline"] = "8-ID-I"

    def _add_metadata(self, hdf5_file: h5py.File, format_type: str) -> None:
        """Add comprehensive metadata to HDF5 file"""
        hdf5_file.attrs["file_format"] = format_type
        hdf5_file.attrs["created_by"] = "Homodyne v2 Test Data Generator"
        hdf5_file.attrs["dataset_size"] = self.config.size.value
        hdf5_file.attrs["quality_level"] = self.config.quality.value
        hdf5_file.attrs["num_q_values"] = self.config.num_q
        hdf5_file.attrs["num_phi_angles"] = self.config.num_phi
        hdf5_file.attrs["num_delays"] = self.config.num_delays
        hdf5_file.attrs["num_frames"] = self.config.num_frames

        # Physics parameters
        physics_group = hdf5_file.create_group("/physics_parameters")
        physics_group.create_dataset(
            "diffusion_coefficient", data=self.config.diffusion_coefficient
        )
        physics_group.create_dataset("temperature", data=self.config.temperature)
        physics_group.create_dataset("viscosity", data=self.config.viscosity)
        physics_group.create_dataset("contrast", data=self.config.contrast)
        physics_group.create_dataset("baseline", data=self.config.baseline_level)

    def generate_dataset(self) -> Path:
        """Generate synthetic dataset and return filepath"""
        filename = f"{self.config.name}_{self.config.size.value}_{self.config.quality.value}_{self.config.format.value}.h5"
        filepath = self.output_dir / filename

        print(f"Generating synthetic dataset: {filename}")
        print(f"  Size: {self.config.size.value}")
        print(f"  Quality: {self.config.quality.value}")
        print(f"  Format: {self.config.format.value}")
        print(
            f"  Dimensions: {self.config.num_q} × {self.config.num_phi} × {self.config.num_delays}"
        )

        if self.config.format == DatasetFormat.APS_OLD:
            self.create_aps_old_format(filepath)
        else:
            self.create_aps_u_format(filepath)

        print(f"Dataset generated: {filepath}")
        print(f"File size: {filepath.stat().st_size / 1024 / 1024:.1f} MB")

        return filepath


def generate_test_dataset_suite(output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Generate a comprehensive suite of test datasets"""
    if output_dir is None:
        output_dir = Path(tempfile.gettempdir()) / "homodyne_test_data"

    datasets = {}

    # Standard test datasets
    configs = [
        # Small datasets for unit testing
        SyntheticDatasetConfig(
            "unit_test_perfect",
            DatasetSize.TINY,
            DataQuality.PERFECT,
            DatasetFormat.APS_U,
        ),
        SyntheticDatasetConfig(
            "unit_test_good", DatasetSize.TINY, DataQuality.GOOD, DatasetFormat.APS_U
        ),
        SyntheticDatasetConfig(
            "unit_test_aps_old",
            DatasetSize.TINY,
            DataQuality.GOOD,
            DatasetFormat.APS_OLD,
        ),
        # Integration test datasets
        SyntheticDatasetConfig(
            "integration_small",
            DatasetSize.SMALL,
            DataQuality.GOOD,
            DatasetFormat.APS_U,
        ),
        SyntheticDatasetConfig(
            "integration_noisy",
            DatasetSize.SMALL,
            DataQuality.NOISY,
            DatasetFormat.APS_U,
        ),
        SyntheticDatasetConfig(
            "integration_aps_old",
            DatasetSize.SMALL,
            DataQuality.GOOD,
            DatasetFormat.APS_OLD,
        ),
        # Performance test datasets
        SyntheticDatasetConfig(
            "performance_medium",
            DatasetSize.MEDIUM,
            DataQuality.GOOD,
            DatasetFormat.APS_U,
        ),
        SyntheticDatasetConfig(
            "performance_large",
            DatasetSize.LARGE,
            DataQuality.GOOD,
            DatasetFormat.APS_U,
        ),
        # Edge case datasets
        SyntheticDatasetConfig(
            "edge_case_poor", DatasetSize.SMALL, DataQuality.POOR, DatasetFormat.APS_U
        ),
        SyntheticDatasetConfig(
            "edge_case_corrupted",
            DatasetSize.SMALL,
            DataQuality.CORRUPTED,
            DatasetFormat.APS_U,
        ),
    ]

    for config in configs:
        config.output_dir = output_dir
        generator = SyntheticDataGenerator(config)
        try:
            filepath = generator.generate_dataset()
            datasets[config.name] = filepath
        except Exception as e:
            print(f"Failed to generate {config.name}: {e}")
            continue

    print(f"\nGenerated {len(datasets)} test datasets in {output_dir}")
    return datasets


if __name__ == "__main__":
    # Generate test datasets when run directly
    test_datasets = generate_test_dataset_suite()

    print("\nTest datasets generated:")
    for name, path in test_datasets.items():
        print(f"  {name}: {path}")
