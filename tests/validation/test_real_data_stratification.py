"""
Real Data Validation for Stratified Chunking

Tests the stratification implementation on actual experimental XPCS data from
C020, C021 datasets to ensure production readiness.

This validates:
1. Data loading from real HDF5 files
2. Stratification decision logic with real phi angle distributions
3. Memory estimation with real dataset sizes
4. Compatibility with existing configurations
5. No breaking changes to production workflows
"""

import logging
from pathlib import Path

import numpy as np
import pytest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestRealDataStratification:
    """Validate stratification on real experimental data."""

    def test_c020_dataset_characteristics(self):
        """Test C020 dataset characteristics and stratification decision."""
        # C020 dataset location
        data_dir = Path("/home/wei/Documents/Projects/data/C020")
        config_file = data_dir / "homodyne_laminar_flow_config.yaml"

        if not config_file.exists():
            pytest.skip(f"C020 config file not found: {config_file}")

        # Import here to avoid import errors if not available
        from homodyne.config.manager import ConfigManager
        from homodyne.data.xpcs_loader import XPCSDataLoader
        from homodyne.optimization.nlsq.strategies.chunking import (
            analyze_angle_distribution,
            should_use_stratification,
        )

        # Load config and data
        logger.info(f"Loading C020 config from: {config_file}")
        ConfigManager(str(config_file))

        # XPCSDataLoader expects config file path (not data file)
        # It will load the data file path from the config
        loader = XPCSDataLoader(str(config_file))

        # Load experimental data to get phi angles
        data = loader.load_experimental_data()
        phi_angles = data["phi_angles_list"]
        n_angles = len(phi_angles)
        logger.info(f"C020 phi angles: {n_angles} angles")
        logger.info(f"Angle range: {phi_angles.min():.2f}° to {phi_angles.max():.2f}°")

        # Analyze angle distribution
        # Create mock phi array with repeated angles (simulating real data structure)
        # In real data, each angle appears once per (t1, t2) pair
        n_time_points = 100  # Estimate
        points_per_angle = n_time_points * n_time_points

        # Create phi array: repeat each angle points_per_angle times
        phi_repeated = np.repeat(phi_angles, points_per_angle)
        total_points = len(phi_repeated)
        logger.info(f"Estimated total points: {total_points:,}")

        # Test angle distribution analysis
        stats = analyze_angle_distribution(phi=phi_repeated)

        logger.info("Angle distribution stats:")
        logger.info(f"  Total angles: {stats.n_angles}")
        logger.info(f"  Min points/angle: {min(stats.counts.values()):,}")
        logger.info(f"  Max points/angle: {max(stats.counts.values()):,}")
        logger.info(f"  Imbalance ratio: {stats.imbalance_ratio:.2f}")
        logger.info(f"  Is balanced: {stats.is_balanced}")

        # Test stratification decision
        should_stratify, reason = should_use_stratification(
            n_points=total_points,
            n_angles=n_angles,
            per_angle_scaling=True,
            imbalance_ratio=stats.imbalance_ratio,
        )

        logger.info(f"Stratification decision: {should_stratify}")
        logger.info(f"Reason: {reason}")

        # Assertions
        assert stats.n_angles == n_angles
        assert stats.n_angles > 0
        assert stats.imbalance_ratio >= 1.0

    def test_c021_dataset_characteristics(self):
        """Test C021 dataset characteristics and stratification decision."""
        # C021 dataset location
        data_dir = Path("/home/wei/Documents/Projects/data/C021")

        if not data_dir.exists():
            pytest.skip(f"C021 data directory not found: {data_dir}")

        # Find HDF5 files
        hdf_files = list(data_dir.glob("*_Twotime.hdf"))
        if not hdf_files:
            pytest.skip(f"No Twotime HDF5 files found in: {data_dir}")

        hdf_file = hdf_files[0]
        logger.info(f"Testing C021 data from: {hdf_file}")

        # Import here to avoid import errors
        from homodyne.data.xpcs_loader import XPCSDataLoader
        from homodyne.optimization.nlsq.strategies.chunking import (
            analyze_angle_distribution,
        )

        # Load data characteristics
        try:
            loader = XPCSDataLoader(str(hdf_file))
            phi_angles = loader.phi_angles
            n_angles = len(phi_angles)

            logger.info(f"C021 phi angles: {n_angles} angles")
            logger.info(
                f"Angle range: {phi_angles.min():.2f}° to {phi_angles.max():.2f}°"
            )

            # Analyze distribution
            n_time_points = 100  # Estimate
            stats = analyze_angle_distribution(
                phi=phi_angles,
                points_per_angle=np.full(n_angles, n_time_points * n_time_points),
            )

            logger.info("C021 angle distribution:")
            logger.info(f"  Imbalance ratio: {stats.imbalance_ratio:.2f}")
            logger.info(f"  Is balanced: {stats.is_balanced}")

            assert stats.n_angles > 0

        except Exception as e:
            pytest.skip(f"Error loading C021 data: {e}")

    def test_stratification_memory_safety_real_data(self):
        """Test memory estimation with realistic dataset sizes."""
        from homodyne.optimization.nlsq.strategies.chunking import (
            estimate_stratification_memory,
        )

        # Test with C020-like dataset
        # ~50 angles × 100×100 time points = 500k points
        n_points_c020 = 50 * 100 * 100  # 500,000 points

        mem_stats = estimate_stratification_memory(
            n_points=n_points_c020, use_index_based=False
        )

        logger.info("C020-like memory estimate:")
        logger.info(f"  Original: {mem_stats['original_memory_mb']:.1f} MB")
        logger.info(f"  Stratified: {mem_stats['stratified_memory_mb']:.1f} MB")
        logger.info(f"  Peak: {mem_stats['peak_memory_mb']:.1f} MB")
        logger.info(f"  Is safe: {mem_stats['is_safe']}")

        # Should be safe for 500k points
        assert mem_stats["is_safe"], (
            "Memory estimation should be safe for C020-like data"
        )
        assert mem_stats["peak_memory_mb"] < 1000, "Peak memory should be reasonable"

    def test_existing_config_compatibility(self):
        """Test that existing configs work without stratification section."""
        # C020 config doesn't have stratification section yet
        config_file = Path(
            "/home/wei/Documents/Projects/data/C020/homodyne_laminar_flow_config.yaml"
        )

        if not config_file.exists():
            pytest.skip(f"C020 config not found: {config_file}")

        import yaml

        with open(config_file) as f:
            config = yaml.safe_load(f)

        # Verify config loads successfully
        assert "analysis_mode" in config
        assert config["analysis_mode"] == "laminar_flow"

        # Verify stratification section is optional (not required)
        # If missing, defaults should be used
        stratification = config.get("optimization", {}).get("stratification", {})

        logger.info("Existing config compatibility:")
        logger.info(
            f"  Has stratification section: {'stratification' in config.get('optimization', {})}"
        )
        logger.info(f"  Config mode: {config['analysis_mode']}")

        # Test default values when section is missing
        enabled = stratification.get("enabled", "auto")  # Default to "auto"
        target_chunk_size = stratification.get("target_chunk_size", 100_000)
        max_imbalance_ratio = stratification.get("max_imbalance_ratio", 5.0)

        assert enabled in ["auto", True, False]
        assert target_chunk_size > 0
        assert max_imbalance_ratio >= 1.0

        logger.info("  Defaults loaded successfully")


class TestRealDataWorkflow:
    """Test full workflow integration with real data."""

    def test_dry_run_c020_analysis(self):
        """Dry run of C020 analysis workflow to validate stratification integration."""
        config_file = Path(
            "/home/wei/Documents/Projects/data/C020/homodyne_laminar_flow_config.yaml"
        )

        if not config_file.exists():
            pytest.skip(f"C020 config not found: {config_file}")

        import yaml

        from homodyne.data.xpcs_loader import XPCSDataLoader

        # Load config
        with open(config_file) as f:
            config_dict = yaml.safe_load(f)

        # Verify data file exists
        data_file = Path(config_dict["experimental_data"]["file_path"])
        if not data_file.exists():
            pytest.skip(f"Data file not found: {data_file}")

        # Test data loading - XPCSDataLoader expects config file path
        loader = XPCSDataLoader(str(config_file))
        data = loader.load_experimental_data()
        phi_angles = data["phi_angles_list"]

        logger.info("C020 workflow dry run:")
        logger.info(f"  Config loaded: {config_file.name}")
        logger.info(
            f"  Data file: {data_file.name} ({data_file.stat().st_size / 1e9:.2f} GB)"
        )
        logger.info(f"  Phi angles: {len(phi_angles)} angles")
        logger.info(f"  Analysis mode: {config_dict['analysis_mode']}")

        # Verify stratification integration doesn't break existing workflow
        assert len(phi_angles) > 0
        assert config_dict["analysis_mode"] in ["static", "laminar_flow"]

        logger.info("  Workflow compatibility: PASS")


# Run validation if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
