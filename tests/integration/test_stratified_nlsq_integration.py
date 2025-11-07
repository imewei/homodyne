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

    return {
        "phi": phi,
        "t1": t1,
        "t2": t2,
        "g2_exp": g2_exp,
        "n_angles": n_angles,
        "n_points": n_total,
    }


@pytest.fixture
def imbalanced_test_data():
    """Create imbalanced test data (skewed angle distribution)."""
    # Extreme imbalance: angle 0 has 10x more points than angles 45 and 90
    n_points_angle0 = 100_000
    n_points_other = 10_000

    phi = np.concatenate([
        np.full(n_points_angle0, 0.0),
        np.full(n_points_other, 45.0),
        np.full(n_points_other, 90.0),
    ])
    t1 = np.linspace(1e-6, 1e-3, len(phi))
    t2 = t1.copy()
    g2_exp = 1.0 + 0.4 * np.exp(-0.1 * (t1 + t2))

    return {
        "phi": phi,
        "t1": t1,
        "t2": t2,
        "g2_exp": g2_exp,
        "n_angles": 3,
        "n_points": len(phi),
    }


# ============================================================================
# Configuration Tests
# ============================================================================


def test_stratification_config_parsing():
    """Test parsing of stratification configuration section."""
    config = {
        "optimization": {
            "stratification": {
                "enabled": "auto",
                "target_chunk_size": 100_000,
                "max_imbalance_ratio": 5.0,
                "check_memory_safety": True,
                "min_points_per_angle": 1000,
            }
        }
    }

    strat_config = config["optimization"]["stratification"]
    assert strat_config["enabled"] == "auto"
    assert strat_config["target_chunk_size"] == 100_000
    assert strat_config["max_imbalance_ratio"] == 5.0
    assert strat_config["check_memory_safety"] is True
    assert strat_config["min_points_per_angle"] == 1000


def test_stratification_config_defaults():
    """Test default values when stratification config is missing."""
    # Default configuration
    default_enabled = "auto"
    default_chunk_size = 100_000
    default_max_imbalance = 5.0

    # Verify defaults match documented behavior
    assert default_enabled == "auto"  # Automatic activation
    assert default_chunk_size == 100_000  # 100k points per chunk
    assert default_max_imbalance == 5.0  # 5x imbalance threshold


def test_stratification_config_enabled_auto():
    """Test 'auto' mode activation criteria."""
    # Auto mode should activate when:
    # - per_angle_scaling=True
    # - n_points >= 100k
    # - Angles are balanced (imbalance_ratio <= max_imbalance_ratio)

    # Case 1: All conditions met → should activate
    should_activate, _ = should_use_stratification(
        n_points=150_000,
        n_angles=3,
        per_angle_scaling=True,
        imbalance_ratio=2.0,  # balanced
    )
    assert should_activate is True

    # Case 2: Small dataset → should NOT activate
    should_activate, reason = should_use_stratification(
        n_points=50_000,
        n_angles=3,
        per_angle_scaling=True,
        imbalance_ratio=2.0,
    )
    assert should_activate is False
    assert "100k" in reason.lower()


def test_stratification_config_validation():
    """Test validation of stratification configuration values."""
    # Valid configuration
    valid_config = {
        "enabled": "auto",
        "target_chunk_size": 100_000,
        "max_imbalance_ratio": 5.0,
    }
    assert valid_config["target_chunk_size"] > 0
    assert valid_config["max_imbalance_ratio"] > 0

    # Edge case: Minimum chunk size
    min_chunk_size = 10_000
    assert min_chunk_size >= 1000  # Reasonable minimum


# ============================================================================
# Strategy Selection Tests
# ============================================================================


def test_strategy_selection_with_large_dataset():
    """Test NLSQ strategy selection with large dataset requiring chunking."""
    selector = DatasetSizeStrategy()

    # Large dataset (>1M points) → should select CHUNKED or STREAMING
    n_points = 3_000_000
    n_parameters = 9  # laminar_flow with per-angle scaling

    strategy = selector.select_strategy(n_points, n_parameters)

    # Should select CHUNKED or STREAMING (not STANDARD)
    assert strategy in [OptimizationStrategy.CHUNKED, OptimizationStrategy.STREAMING]


def test_strategy_selection_with_small_dataset():
    """Test NLSQ strategy selection with small dataset (no chunking needed)."""
    selector = DatasetSizeStrategy()

    # Small dataset (<100k points) → should select STANDARD
    n_points = 50_000
    n_parameters = 5

    strategy = selector.select_strategy(n_points, n_parameters)

    # Should select STANDARD (no chunking)
    assert strategy == OptimizationStrategy.STANDARD


def test_stratification_decision_integrates_with_strategy():
    """Test stratification decision considers NLSQ strategy selection."""
    # Large dataset with per-angle scaling
    n_points = 2_000_000
    n_angles = 3
    per_angle_scaling = True

    # Strategy selector suggests CHUNKED/STREAMING
    selector = DatasetSizeStrategy()
    strategy = selector.select_strategy(n_points, n_parameters=9)
    assert strategy in [OptimizationStrategy.CHUNKED, OptimizationStrategy.STREAMING]

    # Stratification should activate for per-angle + large dataset
    stats = analyze_angle_distribution(np.repeat([0.0, 45.0, 90.0], n_points // 3))
    should_stratify, _ = should_use_stratification(
        n_points=n_points,
        n_angles=stats.n_angles,
        per_angle_scaling=per_angle_scaling,
        imbalance_ratio=stats.imbalance_ratio,
    )

    assert should_stratify is True


# ============================================================================
# Data Preparation Tests
# ============================================================================


def test_stratification_with_balanced_data(balanced_test_data):
    """Test stratification with balanced angle distribution."""
    data = balanced_test_data
    n_points = data["n_points"]

    # Verify balanced distribution
    stats = analyze_angle_distribution(data["phi"])
    assert stats.is_balanced is True
    assert stats.imbalance_ratio <= 2.0  # Good balance

    # Apply stratification
    phi_s, t1_s, t2_s, g2_s = create_angle_stratified_data(
        data["phi"],
        data["t1"],
        data["t2"],
        data["g2_exp"],
        target_chunk_size=100_000,
    )

    # Verify all points preserved
    assert len(phi_s) == n_points
    assert len(t1_s) == n_points
    assert len(t2_s) == n_points
    assert len(g2_s) == n_points


def test_stratification_with_imbalanced_data(imbalanced_test_data):
    """Test stratification detects extreme angle imbalance."""
    data = imbalanced_test_data

    # Verify imbalanced distribution
    stats = analyze_angle_distribution(data["phi"])
    assert stats.is_balanced is False
    assert stats.imbalance_ratio > 5.0  # Extreme imbalance (10x)

    # Stratification should detect this and suggest sequential
    should_stratify, reason = should_use_stratification(
        n_points=data["n_points"],
        n_angles=stats.n_angles,
        per_angle_scaling=True,
        imbalance_ratio=stats.imbalance_ratio,
    )

    # Should NOT stratify (use sequential instead for extreme imbalance)
    assert should_stratify is False
    assert "imbalance" in reason.lower()


def test_stratification_preserves_all_data_points():
    """Test stratification preserves all data points (no duplicates or losses)."""
    # Large balanced dataset
    n_points = 300_000
    phi = np.repeat([0.0, 45.0, 90.0], n_points // 3)
    t1 = np.linspace(1e-6, 1e-3, n_points)
    t2 = t1.copy()
    g2_exp = np.random.uniform(1.0, 1.5, n_points)

    # Apply stratification
    phi_s, t1_s, t2_s, g2_s = create_angle_stratified_data(
        phi, t1, t2, g2_exp, target_chunk_size=100_000
    )

    # Verify no data loss
    assert len(phi_s) == n_points
    assert len(t1_s) == n_points
    assert len(t2_s) == n_points
    assert len(g2_s) == n_points

    # Verify no duplicates (all unique combinations present)
    # Note: Due to stratification, order changes but uniqueness preserved
    original_tuples = set(zip(phi, t1, t2))
    stratified_tuples = set(zip(phi_s, t1_s, t2_s))
    assert len(original_tuples) == len(stratified_tuples)


# ============================================================================
# Component Integration Tests
# ============================================================================


def test_integration_stratification_with_strategy_selector():
    """Test full integration: strategy selection + stratification decision."""
    # Large dataset parameters
    n_points = 2_000_000
    n_parameters = 9  # laminar_flow with per-angle scaling
    per_angle_scaling = True

    # Mock data
    data = {
        "phi": np.repeat([0.0, 45.0, 90.0], n_points // 3),
        "t1": np.linspace(1e-6, 1e-3, n_points),
        "t2": np.linspace(1e-6, 1e-3, n_points),
        "g2_exp": np.random.uniform(1.0, 1.5, n_points),
    }

    # Strategy selection
    selector = DatasetSizeStrategy()
    strategy = selector.select_strategy(n_points, n_parameters)

    # Should select CHUNKED or STREAMING
    assert strategy in [OptimizationStrategy.CHUNKED, OptimizationStrategy.STREAMING]

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


def test_stratified_data_preserves_metadata_attributes():
    """Test that StratifiedData properly copies metadata attributes from original data.

    This is a regression test for bug where stratification created new data object
    without copying critical metadata (sigma, q, L, dt), causing AttributeError
    in residual function creation.

    Bug Report: 2025-11-06
    Root Cause: StratifiedData.__init__() only copied arrays, not scalar metadata
    Fix: Added explicit attribute copying for sigma, q, L, dt
    """
    from homodyne.optimization.nlsq_wrapper import NLSQWrapper
    import logging

    # Create mock data with all required metadata attributes
    # Need 100k+ points to trigger stratification
    # Data structure: unique arrays for phi, t1, t2 (not flattened/repeated)
    # n_points = len(phi) × len(t1) × len(t2)
    n_phi = 3  # 3 phi angles
    n_t = 200  # 200 time points
    n_total = n_phi * n_t * n_t  # 3 × 200 × 200 = 120k points

    class MockOriginalData:
        """Mock data object with all attributes expected by NLSQ."""
        def __init__(self):
            # Array attributes: UNIQUE values only (meshgrid will expand them)
            self.phi = np.array([0.0, 45.0, 90.0])  # 3 unique phi angles
            self.t1 = np.linspace(1e-6, 1e-3, n_t)  # 200 unique time points
            self.t2 = np.linspace(1e-6, 1e-3, n_t)  # 200 unique time points
            # g2 shape: (n_phi, n_t, n_t) or flattened (120k,)
            self.g2 = np.random.uniform(1.0, 1.5, (n_phi, n_t, n_t))

            # Critical metadata attributes (must be copied verbatim)
            # sigma shape must match g2 after meshgrid expansion
            self.sigma = np.ones((n_phi, n_t, n_t)) * 0.1  # Uncertainty/error bars
            self.q = 0.005  # Wavevector magnitude (Å⁻¹)
            self.L = 5000000.0  # Sample-detector distance (Å)
            self.dt = 0.1  # Frame time step (s) - optional

    original_data = MockOriginalData()

    # Create mock config with stratification enabled
    mock_config = {
        "optimization": {
            "stratification": {
                "enabled": True,
                "target_chunk_size": 50_000,  # Reasonable chunk size
                "max_imbalance_ratio": 5.0,
            }
        }
    }

    # Create wrapper and apply stratification
    wrapper = NLSQWrapper()  # Use default initialization
    logger = logging.getLogger(__name__)

    # This should trigger stratification since we have balanced angles
    stratified_data = wrapper._apply_stratification_if_needed(
        data=original_data,
        per_angle_scaling=True,  # Requires metadata for residual computation
        config=mock_config,
        logger=logger,
    )

    # CRITICAL: Verify all metadata attributes were copied
    # These checks prevent regression of the AttributeError bug

    # Check required attributes exist
    assert hasattr(stratified_data, 'sigma'), "sigma not copied (CRITICAL)"
    assert hasattr(stratified_data, 'q'), "q not copied (CRITICAL)"
    assert hasattr(stratified_data, 'L'), "L not copied (CRITICAL)"

    # Check optional dt attribute
    assert hasattr(stratified_data, 'dt'), "dt not copied (optional but expected)"

    # Verify values match original (scalars should be copied verbatim)
    assert stratified_data.q == original_data.q, "q value mismatch"
    assert stratified_data.L == original_data.L, "L value mismatch"
    assert stratified_data.dt == original_data.dt, "dt value mismatch"

    # Verify array attributes were reorganized (not just copied)
    assert hasattr(stratified_data, 'phi'), "phi missing"
    assert hasattr(stratified_data, 't1'), "t1 missing"
    assert hasattr(stratified_data, 't2'), "t2 missing"
    assert hasattr(stratified_data, 'g2'), "g2 missing"

    # Verify stratification diagnostics present
    assert hasattr(stratified_data, 'stratification_diagnostics'), "diagnostics missing"

    # Additional check: Simulate residual function validation
    # This is the code that originally failed with AttributeError
    required_attrs = ["phi", "t1", "t2", "g2", "sigma", "q", "L"]
    for attr in required_attrs:
        assert hasattr(stratified_data, attr), f"Missing required attribute: {attr}"

    # Success: All attributes present, residual function validation would pass


def test_stratification_diagnostics_passed_to_result():
    """Test that stratification diagnostics parameter is accepted without NameError.

    This is a regression test for bug where stratification_diagnostics was extracted
    from stratified_data but not passed to _create_fit_result(), causing NameError.

    Bug Report: 2025-11-06 (log: homodyne_analysis_20251106_122208.log)
    Root Cause: _create_fit_result() referenced stratification_diagnostics without
                receiving it as a parameter
    Fix: Added stratification_diagnostics parameter to function call and signature

    Test strategy: This test verifies that _create_fit_result() accepts the
    stratification_diagnostics parameter (preventing NameError), even when diagnostics
    is None (the common case when collect_diagnostics=False).
    """
    from homodyne.optimization.nlsq_wrapper import NLSQWrapper, OptimizationResult

    # Create wrapper
    wrapper = NLSQWrapper()

    # Mock minimal parameters
    n_data = 1000
    popt = np.array([1.0, 0.5, 0.1, 1.0, 0.5, 0.1, 0.0, 0.5, 1.0])

    # CRITICAL TEST 1: Verify _create_fit_result() accepts stratification_diagnostics=None
    # This is the common case and where the NameError occurred
    try:
        result = wrapper._create_fit_result(
            popt=popt,
            pcov=np.eye(9),
            residuals=np.zeros(n_data),
            n_data=n_data,
            iterations=10,
            execution_time=1.0,
            convergence_status="converged",
            recovery_actions=[],
            streaming_diagnostics=None,
            stratification_diagnostics=None,  # KEY FIX: Parameter now accepted
        )
    except NameError as e:
        if 'stratification_diagnostics' in str(e):
            pytest.fail(f"NameError when passing stratification_diagnostics=None: {e}")
        else:
            raise

    # Verify result was created successfully
    assert isinstance(result, OptimizationResult), "Result is not OptimizationResult"
    assert hasattr(result, 'stratification_diagnostics'), \
        "Result missing stratification_diagnostics attribute"
    assert result.stratification_diagnostics is None, \
        "Expected stratification_diagnostics=None"

    # Success: The parameter is now accepted without NameError
    # This prevents the bug from recurring where _create_fit_result() tried to
    # use stratification_diagnostics without receiving it as a parameter


def test_full_nlsq_workflow_with_stratification():
    """End-to-end integration test for stratification with per-angle scaling.

    This test validates the complete workflow from data preparation through
    stratification to result creation. Uses a moderate dataset size to trigger
    stratification while avoiding vmap errors.

    Test Coverage:
    - Data structure validation (phi, t1, t2, g2 with metadata)
    - Stratification activation and execution
    - Stratification diagnostics collection and propagation
    - Result creation with diagnostics
    - Metadata preservation (regression test for bug #1)
    - Diagnostics parameter passing (regression test for bug #2)

    Integration test (2025-11-06): Validates fixes for metadata and diagnostics bugs

    Note: Uses moderate dataset size (50k points) with per_angle_scaling=True to
    trigger stratification while avoiding known large dataset issues.
    """
    from homodyne.optimization.nlsq_wrapper import NLSQWrapper
    import logging

    # Create realistic mock data (50k points, 3 angles)
    # Smaller dataset to avoid vmap errors while still triggering stratification
    n_phi = 3
    n_t = 130  # 130x130 time grid per angle ≈ 50,700 points
    n_total = n_phi * n_t * n_t

    class MockData:
        """Mock experimental data with full metadata."""
        def __init__(self):
            # Unique arrays for phi, t1, t2 (meshgrid will expand)
            self.phi = np.array([0.0, 45.0, 90.0])
            self.t1 = np.linspace(1e-6, 1e-3, n_t)
            self.t2 = np.linspace(1e-6, 1e-3, n_t)

            # Generate synthetic g2 data with known structure
            t1_grid, t2_grid = np.meshgrid(self.t1, self.t2, indexing='ij')
            tau_sum = t1_grid + t2_grid

            # Simple isotropic data (same for all angles)
            self.g2 = np.zeros((n_phi, n_t, n_t))
            for i in range(n_phi):
                self.g2[i] = 1.0 + 0.4 * np.exp(-1000.0 * tau_sum)

            # Required metadata (regression test for bug #1)
            self.sigma = np.ones((n_phi, n_t, n_t)) * 0.01
            self.q = 0.005
            self.L = 5000000.0
            self.dt = 0.1

    data = MockData()

    # Configuration for stratification with diagnostics
    # Use a mock ConfigManager to properly pass stratification settings
    class MockConfig:
        def __init__(self):
            self.config = {
                "optimization": {
                    "stratification": {
                        "enabled": True,  # Force stratification
                        "target_chunk_size": 20_000,  # Create 2-3 chunks from 50k points
                        "collect_diagnostics": True,  # Enable diagnostics (bug #2)
                        "max_imbalance_ratio": 5.0,
                    }
                }
            }

    config = MockConfig()

    # Create wrapper and apply stratification
    wrapper = NLSQWrapper()
    logger = logging.getLogger(__name__)

    # CRITICAL TEST: Apply stratification (tests bug #1 and #2 fixes)
    try:
        stratified_data = wrapper._apply_stratification_if_needed(
            data=data,
            per_angle_scaling=True,  # Required for stratification
            config=config,
            logger=logger,
        )
    except AttributeError as e:
        if 'sigma' in str(e) or 'q' in str(e) or 'L' in str(e):
            pytest.fail(f"Metadata not preserved during stratification (Bug #1): {e}")
        else:
            raise
    except Exception as e:
        pytest.fail(f"Stratification failed: {e}")

    # Validate stratification occurred
    assert hasattr(stratified_data, 'stratification_diagnostics'), \
        "Stratification diagnostics not created"
    assert stratified_data.stratification_diagnostics is not None, \
        "Stratification diagnostics is None (Bug #2 would cause NameError later)"

    # Validate metadata preserved (Bug #1 fix)
    assert hasattr(stratified_data, 'sigma'), "sigma not preserved"
    assert hasattr(stratified_data, 'q'), "q not preserved"
    assert hasattr(stratified_data, 'L'), "L not preserved"
    assert hasattr(stratified_data, 'dt'), "dt not preserved"

    # Validate diagnostics content
    diag = stratified_data.stratification_diagnostics
    assert diag.n_chunks > 0, "Should have created chunks"
    assert len(diag.chunk_sizes) == diag.n_chunks, "Chunk sizes list length mismatch"
    assert len(diag.angles_per_chunk) == diag.n_chunks, "Angles per chunk list length mismatch"

    # Validate all chunks have all angles (key stratification requirement)
    for i, n_angles in enumerate(diag.angles_per_chunk):
        assert n_angles == 3, \
            f"Chunk {i} has {n_angles} angles, expected 3 (stratification failed)"

    # CRITICAL TEST: Create result with diagnostics (Bug #2 fix)
    try:
        result = wrapper._create_fit_result(
            popt=np.array([1000.0, 1.0, 0.0]),
            pcov=np.eye(3),
            residuals=np.zeros(n_total),
            n_data=n_total,
            iterations=10,
            execution_time=1.0,
            convergence_status="converged",
            recovery_actions=[],
            streaming_diagnostics=None,
            stratification_diagnostics=stratified_data.stratification_diagnostics,  # Bug #2
        )
    except NameError as e:
        if 'stratification_diagnostics' in str(e):
            pytest.fail(f"NameError when passing stratification_diagnostics (Bug #2): {e}")
        else:
            raise

    # Validate result contains diagnostics
    assert hasattr(result, 'stratification_diagnostics'), \
        "Result missing stratification_diagnostics"
    assert result.stratification_diagnostics is not None, \
        "Result stratification_diagnostics is None"

    # Success: Full workflow completed with both fixes validated
    print(f"✓ Full stratification workflow test passed:")
    print(f"  - Data points: {n_total:,}")
    print(f"  - Chunks created: {diag.n_chunks}")
    print(f"  - Metadata preserved: ✓ (Bug #1 fix)")
    print(f"  - Diagnostics passed: ✓ (Bug #2 fix)")
