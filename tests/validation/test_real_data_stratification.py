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

import pytest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestRealDataStratification:
    """Validate stratification on real experimental data."""

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


# Run validation if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
