"""Large Dataset Mock Generators for NLSQ API Alignment Testing.

This module provides efficient mock data generators for testing optimization
strategies at various scales without consuming excessive memory or compute.

Design Principles:
- Deterministic generation with fixed random seeds
- Memory-efficient: Only allocates data when needed
- Scalable: Supports 1M to 1B+ point datasets
- Realistic: Mimics real XPCS data structure and characteristics

Test Dataset Sizes:
- SMALL: < 1M points → STANDARD strategy
- MEDIUM: 1M - 10M points → LARGE strategy
- LARGE: 10M - 100M points → CHUNKED strategy
- XLARGE: > 100M points → STREAMING strategy

Author: Testing Engineer (Task Group 6.1)
Date: 2025-10-22
"""

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DatasetMetadata:
    """Metadata for generated dataset."""

    n_points: int
    n_phi: int
    n_t1: int
    n_t2: int
    strategy_expected: str
    memory_estimate_mb: float
    seed: int


class LargeDatasetFactory:
    """Factory for generating large-scale mock XPCS datasets."""

    # Strategy thresholds (matching homodyne/optimization/strategy.py)
    THRESHOLD_STANDARD = 1_000_000  # 1M
    THRESHOLD_LARGE = 10_000_000  # 10M
    THRESHOLD_CHUNKED = 100_000_000  # 100M

    def __init__(self, seed: int = 42):
        """Initialize factory with deterministic random seed.

        Parameters
        ----------
        seed : int
            Random seed for reproducible data generation
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _estimate_memory_mb(self, n_points: int, n_params: int = 5) -> float:
        """Estimate memory requirements in MB.

        Parameters
        ----------
        n_points : int
            Total number of data points
        n_params : int
            Number of parameters

        Returns
        -------
        float
            Estimated memory in MB
        """
        # Data arrays (float64): g2, sigma, phi, t1, t2
        data_bytes = n_points * 8 * 5
        # Jacobian (n_points x n_params)
        jacobian_bytes = n_points * n_params * 8
        # Working memory (factor of 3 for intermediate calculations)
        total_bytes = (data_bytes + jacobian_bytes) * 3
        return total_bytes / (1024**2)

    def _determine_expected_strategy(self, n_points: int) -> str:
        """Determine expected optimization strategy based on dataset size.

        Parameters
        ----------
        n_points : int
            Total number of data points

        Returns
        -------
        str
            Expected strategy: STANDARD, LARGE, CHUNKED, or STREAMING
        """
        if n_points < self.THRESHOLD_STANDARD:
            return "STANDARD"
        elif n_points < self.THRESHOLD_LARGE:
            return "LARGE"
        elif n_points < self.THRESHOLD_CHUNKED:
            return "CHUNKED"
        else:
            return "STREAMING"

    def create_mock_dataset(
        self,
        n_phi: int,
        n_t1: int,
        n_t2: int,
        true_params: dict[str, float] | None = None,
        allocate_data: bool = True,
    ) -> tuple[Any, DatasetMetadata]:
        """Create mock XPCS dataset with specified dimensions.

        Parameters
        ----------
        n_phi : int
            Number of phi angles
        n_t1 : int
            Number of t1 time points
        n_t2 : int
            Number of t2 time points
        true_params : dict, optional
            True physical parameters for synthetic data
        allocate_data : bool
            If False, return metadata only (for large datasets)

        Returns
        -------
        data : MockXPCSData or None
            Mock XPCS data object (None if allocate_data=False)
        metadata : DatasetMetadata
            Dataset metadata and characteristics
        """
        n_points = n_phi * n_t1 * n_t2

        # Create metadata
        metadata = DatasetMetadata(
            n_points=n_points,
            n_phi=n_phi,
            n_t1=n_t1,
            n_t2=n_t2,
            strategy_expected=self._determine_expected_strategy(n_points),
            memory_estimate_mb=self._estimate_memory_mb(n_points),
            seed=self.seed,
        )

        if not allocate_data:
            # Return metadata only for memory efficiency
            return None, metadata

        # Allocate and generate data
        if true_params is None:
            true_params = {
                "contrast_0": 0.3,
                "offset_0": 1.0,
                "D0": 1000.0,
                "alpha": 0.5,
                "D_offset": 10.0,
            }

        data = self._generate_synthetic_data(n_phi, n_t1, n_t2, true_params)

        return data, metadata

    def _generate_synthetic_data(
        self,
        n_phi: int,
        n_t1: int,
        n_t2: int,
        true_params: dict[str, float],
    ) -> Any:
        """Generate synthetic XPCS correlation data.

        Parameters
        ----------
        n_phi : int
            Number of phi angles
        n_t1 : int
            Number of t1 time points
        n_t2 : int
            Number of t2 time points
        true_params : dict
            True physical parameters

        Returns
        -------
        MockXPCSData
            Mock data object with realistic structure
        """

        class MockXPCSData:
            pass

        data = MockXPCSData()

        # Generate coordinate arrays
        data.phi = np.linspace(0, 90, n_phi)
        data.t1 = np.logspace(-2, 1, n_t1)  # Log-spaced: 0.01 to 10 seconds
        data.t2 = np.logspace(-2, 1, n_t2)

        # Generate correlation function g2(phi, t1, t2)
        # Use simplified model: g2 = offset + contrast * exp(-rate * t)
        contrast = true_params.get("contrast", true_params.get("contrast_0", 0.3))
        offset = true_params.get("offset", true_params.get("offset_0", 1.0))
        D0 = true_params["D0"]

        # Create meshgrid
        phi_grid, t1_grid, t2_grid = np.meshgrid(
            data.phi, data.t1, data.t2, indexing="ij"
        )

        # Simplified decay rate (angle-dependent)
        decay_rate = D0 * (1 + 0.1 * np.sin(np.deg2rad(phi_grid)))

        # Generate g2 with realistic structure
        data.g2 = offset + contrast * np.exp(-decay_rate * (t1_grid + t2_grid) / 1000.0)

        # Add realistic noise (Poisson-like)
        noise_level = 0.01
        data.g2 += self.rng.normal(0, noise_level, data.g2.shape)

        # Generate uncertainties (sigma)
        data.sigma = np.ones_like(data.g2) * noise_level

        # Metadata
        data.q = 0.01  # Scattering vector (1/nm)
        data.L = 1000.0  # Sample thickness (nm)
        data.dt = 0.1  # Time step (s)

        return data

    # ========================================================================
    # Convenience Methods for Standard Dataset Sizes
    # ========================================================================

    def create_1m_dataset(
        self, allocate_data: bool = True
    ) -> tuple[Any, DatasetMetadata]:
        """Create 1M point dataset (boundary: STANDARD → LARGE).

        Dataset: 100 phi × 100 t1 × 100 t2 = 1,000,000 points
        Expected strategy: LARGE (at threshold)
        Memory estimate: ~120 MB

        Parameters
        ----------
        allocate_data : bool
            If False, return metadata only

        Returns
        -------
        data : MockXPCSData or None
        metadata : DatasetMetadata
        """
        return self.create_mock_dataset(
            n_phi=100,
            n_t1=100,
            n_t2=100,
            allocate_data=allocate_data,
        )

    def create_10m_dataset(
        self, allocate_data: bool = False
    ) -> tuple[Any, DatasetMetadata]:
        """Create 10M point dataset (boundary: LARGE → CHUNKED).

        Dataset: 200 phi × 250 t1 × 200 t2 = 10,000,000 points
        Expected strategy: CHUNKED (at threshold)
        Memory estimate: ~1.2 GB

        WARNING: By default, does NOT allocate data (allocate_data=False)
        to avoid memory issues in tests. Set allocate_data=True only if
        you have sufficient memory and need real data.

        Parameters
        ----------
        allocate_data : bool
            If True, allocate full dataset (requires ~1.2 GB)

        Returns
        -------
        data : MockXPCSData or None
        metadata : DatasetMetadata
        """
        if allocate_data:
            warnings.warn(
                "Allocating 10M point dataset (~1.2 GB). "
                "This may cause memory issues on low-memory systems.",
                ResourceWarning,
                stacklevel=2,
            )

        return self.create_mock_dataset(
            n_phi=200,
            n_t1=250,
            n_t2=200,
            allocate_data=allocate_data,
        )

    def create_100m_dataset(
        self, allocate_data: bool = False
    ) -> tuple[Any, DatasetMetadata]:
        """Create 100M point dataset (boundary: CHUNKED → STREAMING).

        Dataset: 400 phi × 500 t1 × 500 t2 = 100,000,000 points
        Expected strategy: STREAMING (at threshold)
        Memory estimate: ~12 GB

        WARNING: Does NOT allocate data by default to prevent memory exhaustion.
        Only metadata is generated. Use allocate_data=True with extreme caution.

        Parameters
        ----------
        allocate_data : bool
            If True, allocate full dataset (requires ~12 GB) - NOT RECOMMENDED

        Returns
        -------
        data : None (unless allocate_data=True)
        metadata : DatasetMetadata
        """
        if allocate_data:
            warnings.warn(
                "Allocating 100M point dataset (~12 GB). "
                "This will likely cause out-of-memory errors on most systems. "
                "Consider using metadata-only mode (allocate_data=False).",
                ResourceWarning,
                stacklevel=2,
            )

        return self.create_mock_dataset(
            n_phi=400,
            n_t1=500,
            n_t2=500,
            allocate_data=allocate_data,
        )

    def create_1b_dataset(
        self, allocate_data: bool = False
    ) -> tuple[Any, DatasetMetadata]:
        """Create 1B point dataset (extreme STREAMING test).

        Dataset: 1000 phi × 1000 t1 × 1000 t2 = 1,000,000,000 points
        Expected strategy: STREAMING
        Memory estimate: ~120 GB

        WARNING: This is metadata-only by design. NEVER set allocate_data=True.
        This dataset exists only for testing strategy selection and metadata handling.

        Parameters
        ----------
        allocate_data : bool
            Should ALWAYS be False. Included only for API consistency.

        Returns
        -------
        data : None (always)
        metadata : DatasetMetadata
        """
        if allocate_data:
            raise MemoryError(
                "Cannot allocate 1B point dataset (~120 GB). "
                "This method is metadata-only for testing strategy selection. "
                "Use streaming optimization with batch processing instead."
            )

        return self.create_mock_dataset(
            n_phi=1000,
            n_t1=1000,
            n_t2=1000,
            allocate_data=False,
        )

    # ========================================================================
    # Batch Data Generation for Streaming Tests
    # ========================================================================

    def create_batch_generator(
        self,
        total_points: int,
        batch_size: int,
        n_params: int = 5,
    ):
        """Create a generator for batch-wise data processing.

        This generator yields synthetic data in batches without allocating
        the full dataset, enabling streaming optimization tests.

        Parameters
        ----------
        total_points : int
            Total number of data points
        batch_size : int
            Number of points per batch
        n_params : int
            Number of parameters (for residual shape)

        Yields
        ------
        batch_data : dict
            Dictionary with keys: 'indices', 'xdata', 'ydata'
        """
        n_batches = (total_points + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_points)
            n_points_batch = end_idx - start_idx

            # Generate synthetic batch data
            xdata = np.arange(start_idx, end_idx, dtype=np.int32)
            ydata = self.rng.normal(1.0, 0.01, n_points_batch)

            yield {
                "batch_idx": batch_idx,
                "indices": xdata,
                "xdata": xdata,
                "ydata": ydata,
                "n_points": n_points_batch,
            }


# ============================================================================
# Convenience Functions
# ============================================================================


def create_test_dataset(
    size: str,
    seed: int = 42,
    allocate_data: bool = True,
) -> tuple[Any, DatasetMetadata]:
    """Create test dataset by size specification.

    Parameters
    ----------
    size : str
        Dataset size: '1m', '10m', '100m', '1b'
    seed : int
        Random seed
    allocate_data : bool
        Whether to allocate full dataset

    Returns
    -------
    data : MockXPCSData or None
    metadata : DatasetMetadata

    Examples
    --------
    >>> # Create 1M dataset with data
    >>> data, meta = create_test_dataset('1m', allocate_data=True)
    >>> print(meta.strategy_expected)
    LARGE

    >>> # Create 100M dataset (metadata only)
    >>> data, meta = create_test_dataset('100m', allocate_data=False)
    >>> print(data)  # None
    >>> print(meta.memory_estimate_mb)
    12288.0
    """
    factory = LargeDatasetFactory(seed=seed)

    size = size.lower()
    if size == "1m":
        return factory.create_1m_dataset(allocate_data=allocate_data)
    elif size == "10m":
        return factory.create_10m_dataset(allocate_data=allocate_data)
    elif size == "100m":
        return factory.create_100m_dataset(allocate_data=allocate_data)
    elif size == "1b":
        return factory.create_1b_dataset(allocate_data=False)
    else:
        raise ValueError(
            f"Unknown size: {size}. Valid options: '1m', '10m', '100m', '1b'"
        )


def estimate_test_duration(n_points: int, points_per_second: float = 10000.0) -> float:
    """Estimate test duration for dataset size.

    Parameters
    ----------
    n_points : int
        Number of data points
    points_per_second : float
        Throughput estimate (points/second)

    Returns
    -------
    float
        Estimated duration in seconds
    """
    return n_points / points_per_second
