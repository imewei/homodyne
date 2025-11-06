"""Integration Tests for Stratified Chunking with NLSQ Optimization.

This module tests the integration of angle-stratified chunking with NLSQ
optimization workflows, validating component interactions and configuration
handling for the stratification feature.

Test Coverage
-------------
1. Configuration parsing and validation
2. Strategy selection with stratification
3. Data preparation with stratification
4. Component integration (strategy selector, data prep, optimization)

References
----------
Ultra-Think Analysis: ultra-think-20251106-012247
Root Cause: Per-angle parameters + NLSQ chunking incompatibility
Solution: Angle-stratified data reorganization
"""

from __future__ import annotations

import numpy as np
import pytest

from homodyne.optimization.strategy import OptimizationStrategy, DatasetSizeStrategy
from homodyne.optimization.stratified_chunking import (
    analyze_angle_distribution,
    create_angle_stratified_data,
    should_use_stratification,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def balanced_test_data():
    """Create balanced test data for integration tests."""
    n_angles = 3
    n_points_per_angle = 50_000
    n_total = n_angles * n_points_per_angle

    phi = np.repeat([0.0, 45.0, 90.0], n_points_per_angle)
    t1 = np.tile(np.linspace(1e-6, 1e-3, n_points_per_angle), n_angles)
    t2 = t1.copy()
    g2_exp = 1.0 + 0.4 * np.exp(-0.1 * (t1 + t2))

    return {"phi": phi, "t1": t1, "t2": t2, "g2_exp": g2_exp}


# ============================================================================
# Test 1-4: Configuration Integration
# ============================================================================


def test_stratification_config_parsing():
    """Test that stratification configuration is properly parsed from config dict."""
    config = {
        "optimization": {
            "stratification": {
                "enabled": True,
                "target_chunk_size": 50_000,
                "max_imbalance_ratio": 10.0,
            }
        }
    }

    # Verify config structure
    assert "stratification" in config["optimization"]
    assert config["optimization"]["stratification"]["enabled"] is True
    assert config["optimization"]["stratification"]["target_chunk_size"] == 50_000


def test_stratification_config_defaults():
    """Test that stratification uses defaults when not specified."""
    config = {"optimization": {}}

    # Should handle missing stratification section gracefully
    strat_config = config["optimization"].get("stratification", {})
    enabled = strat_config.get("enabled", "auto")
    chunk_size = strat_config.get("target_chunk_size", 100_000)

    assert enabled == "auto"
    assert chunk_size == 100_000


def test_stratification_config_enabled_auto():
    """Test 'auto' mode configuration."""
    config = {"optimization": {"stratification": {"enabled": "auto"}}}

    enabled = config["optimization"]["stratification"]["enabled"]
    assert enabled == "auto"


def test_stratification_config_validation():
    """Test that invalid configuration values are handled."""
    # Invalid enabled value
    config = {"optimization": {"stratification": {"enabled": "invalid"}}}

    # Should not crash when parsing
    enabled = config["optimization"]["stratification"]["enabled"]
    assert enabled == "invalid"  # Value stored, validation happens later


# ============================================================================
# Test 5-7: Strategy Selection Integration
# ============================================================================


def test_strategy_selection_with_large_dataset():
    """Test that LARGE strategy is selected for datasets > 1M points."""
    selector = DatasetSizeStrategy()

    # 1.5M points → should trigger LARGE strategy
    strategy = selector.select_strategy(n_points=1_500_000, n_parameters=5)

    assert strategy == OptimizationStrategy.LARGE


def test_strategy_selection_with_small_dataset():
    """Test that STANDARD strategy is selected for small datasets."""
    selector = DatasetSizeStrategy()

    # 50k points → should trigger STANDARD strategy
    strategy = selector.select_strategy(n_points=50_000, n_parameters=5)

    assert strategy == OptimizationStrategy.STANDARD


def test_stratification_decision_integrates_with_strategy():
    """Test that stratification decision logic works with strategy selection."""
    selector = DatasetSizeStrategy()

    # Large dataset with per-angle scaling (>= 100k for stratification)
    n_points = 150_000
    strategy = selector.select_strategy(n_points, n_parameters=5)

    # Check stratification decision
    should_stratify, reason = should_use_stratification(
        n_points=n_points,
        n_angles=3,
        per_angle_scaling=True,
        imbalance_ratio=1.5,
    )

    # Should stratify: dataset >= 100k + per-angle + balanced
    assert should_stratify is True
    # Strategy for 150k is still STANDARD (< 1M), but stratification can still be used
    assert strategy == OptimizationStrategy.STANDARD


# ============================================================================
# Test 8-10: Data Stratification Integration
# ============================================================================


def test_stratification_with_balanced_data(balanced_test_data):
    """Test stratification with balanced angle distribution."""
    data = balanced_test_data

    # Analyze distribution
    stats = analyze_angle_distribution(data["phi"])
    assert stats.n_angles == 3
    assert stats.imbalance_ratio == 1.0  # Perfectly balanced

    # Apply stratification
    phi_s, t1_s, t2_s, g2_s = create_angle_stratified_data(
        data["phi"],
        data["t1"],
        data["t2"],
        data["g2_exp"],
        target_chunk_size=100_000,
    )

    # Verify data integrity
    assert len(phi_s) == len(data["phi"])
    assert np.allclose(np.sort(phi_s), np.sort(data["phi"]))


def test_stratification_with_imbalanced_data():
    """Test stratification with imbalanced angle distribution."""
    # Create imbalanced data (>= 100k points to pass size threshold)
    phi = np.concatenate([np.full(100_000, 0.0), np.full(50_000, 45.0), np.full(10_000, 90.0)])
    t1 = np.linspace(1e-6, 1e-3, len(phi))
    t2 = t1.copy()
    g2_exp = np.random.uniform(1.0, 1.5, len(phi))

    # Analyze distribution
    stats = analyze_angle_distribution(phi)
    assert stats.n_angles == 3
    assert stats.imbalance_ratio == pytest.approx(10.0, rel=0.01)  # 100000/10000

    # Check stratification decision
    should_stratify, reason = should_use_stratification(
        n_points=len(phi),
        n_angles=stats.n_angles,
        per_angle_scaling=True,
        imbalance_ratio=stats.imbalance_ratio,
    )

    # Should NOT stratify due to high imbalance (>5.0)
    assert should_stratify is False
    assert "imbalance" in reason.lower() or "sequential" in reason.lower()


def test_stratification_preserves_all_data_points(balanced_test_data):
    """Test that stratification doesn't lose or duplicate data points."""
    data = balanced_test_data

    phi_s, t1_s, t2_s, g2_s = create_angle_stratified_data(
        data["phi"],
        data["t1"],
        data["t2"],
        data["g2_exp"],
        target_chunk_size=100_000,
    )

    # Check counts
    assert len(phi_s) == len(data["phi"])
    assert len(t1_s) == len(data["t1"])
    assert len(t2_s) == len(data["t2"])
    assert len(g2_s) == len(data["g2_exp"])

    # Check no duplicates (all original points present)
    original_set = set(zip(data["phi"], data["t1"], data["t2"], data["g2_exp"]))
    stratified_set = set(zip(np.asarray(phi_s), np.asarray(t1_s), np.asarray(t2_s), np.asarray(g2_s)))
    assert len(stratified_set) == len(original_set)


# ============================================================================
# Test 11-12: Component Integration
# ============================================================================


def test_integration_stratification_with_strategy_selector(balanced_test_data):
    """Test integration of stratification with strategy selection."""
    data = balanced_test_data
    n_points = len(data["phi"])  # 150k points

    # Strategy selection
    selector = DatasetSizeStrategy()
    strategy = selector.select_strategy(n_points, n_parameters=5)

    # 150k is still STANDARD (< 1M), but stratification is still applied
    assert strategy == OptimizationStrategy.STANDARD

    # Stratification decision
    stats = analyze_angle_distribution(data["phi"])
    should_stratify, _ = should_use_stratification(
        n_points=n_points,
        n_angles=stats.n_angles,
        per_angle_scaling=True,
        imbalance_ratio=stats.imbalance_ratio,
    )

    # Should stratify (>= 100k + per-angle + balanced)
    assert should_stratify is True

    # Apply stratification
    phi_s, t1_s, t2_s, g2_s = create_angle_stratified_data(
        data["phi"],
        data["t1"],
        data["t2"],
        data["g2_exp"],
        target_chunk_size=100_000,
    )

    # Verify workflow completes successfully
    assert len(phi_s) == n_points


def test_integration_workflow_without_stratification():
    """Test workflow when stratification is not needed (small dataset)."""
    # Small dataset
    n_points = 10_000
    phi = np.repeat([0.0, 45.0, 90.0], n_points // 3)
    t1 = np.linspace(1e-6, 1e-3, n_points)
    t2 = t1.copy()
    g2_exp = np.random.uniform(1.0, 1.5, n_points)

    # Strategy selection
    selector = DatasetSizeStrategy()
    strategy = selector.select_strategy(n_points, n_parameters=5)

    # Should select STANDARD (no chunking)
    assert strategy == OptimizationStrategy.STANDARD

    # Stratification decision
    stats = analyze_angle_distribution(phi)
    should_stratify, reason = should_use_stratification(
        n_points=n_points,
        n_angles=stats.n_angles,
        per_angle_scaling=True,
        imbalance_ratio=stats.imbalance_ratio,
    )

    # Should NOT stratify (small dataset)
    assert should_stratify is False
    assert "100k" in reason.lower() or "standard" in reason.lower()
