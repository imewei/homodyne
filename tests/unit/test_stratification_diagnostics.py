"""
Unit tests for stratification diagnostics functionality.

Tests:
- StratificationDiagnostics dataclass construction
- compute_stratification_diagnostics() with various datasets
- format_diagnostics_report() output validation
- Diagnostic accuracy verification
"""

import jax.numpy as jnp
import pytest

from homodyne.optimization.nlsq.strategies.chunking import (
    StratificationDiagnostics,
    compute_stratification_diagnostics,
    create_angle_stratified_data,
    create_angle_stratified_indices,
    format_diagnostics_report,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def balanced_data():
    """Create balanced dataset with 3 angles, 300 points each."""
    n_per_angle = 300
    phi = jnp.array([0.0] * n_per_angle + [1.0] * n_per_angle + [2.0] * n_per_angle)
    t1 = jnp.linspace(1e-6, 1e-3, 900)
    t2 = jnp.linspace(1e-5, 1e-2, 900)
    g2 = jnp.ones(900) * 1.5
    return phi, t1, t2, g2


@pytest.fixture
def imbalanced_data():
    """Create imbalanced dataset with 3 angles, varying points."""
    phi = jnp.array([0.0] * 100 + [1.0] * 300 + [2.0] * 500)
    t1 = jnp.linspace(1e-6, 1e-3, 900)
    t2 = jnp.linspace(1e-5, 1e-2, 900)
    g2 = jnp.ones(900) * 1.5
    return phi, t1, t2, g2


@pytest.fixture
def large_balanced_data():
    """Create large balanced dataset for performance testing."""
    n_per_angle = 50000  # 150k total
    phi = jnp.array([0.0] * n_per_angle + [1.0] * n_per_angle + [2.0] * n_per_angle)
    t1 = jnp.linspace(1e-6, 1e-3, 150000)
    t2 = jnp.linspace(1e-5, 1e-2, 150000)
    g2 = jnp.ones(150000) * 1.5
    return phi, t1, t2, g2


# ============================================================================
# StratificationDiagnostics Dataclass Tests
# ============================================================================


def test_stratification_diagnostics_dataclass_construction():
    """Test StratificationDiagnostics dataclass can be constructed."""
    diagnostics = StratificationDiagnostics(
        n_chunks=5,
        chunk_sizes=[100, 100, 100, 100, 100],
        chunk_balance={"mean": 100.0, "std": 0.0, "min": 100, "max": 100, "cv": 0.0},
        angles_per_chunk=[3, 3, 3, 3, 3],
        angle_coverage={
            "mean_angles": 3.0,
            "std_angles": 0.0,
            "min_coverage_ratio": 1.0,
            "perfect_coverage_chunks": 5,
        },
        execution_time_ms=10.5,
        memory_overhead_mb=5.2,
        memory_efficiency=0.95,
        throughput_points_per_sec=47619.0,
        use_index_based=False,
    )

    assert diagnostics.n_chunks == 5
    assert diagnostics.chunk_balance["mean"] == 100.0
    assert diagnostics.angle_coverage["min_coverage_ratio"] == 1.0
    assert diagnostics.throughput_points_per_sec == 47619.0
    assert diagnostics.use_index_based is False


def test_stratification_diagnostics_all_fields():
    """Test all fields are accessible."""
    diagnostics = StratificationDiagnostics(
        n_chunks=3,
        chunk_sizes=[100, 100, 100],
        chunk_balance={"mean": 100.0, "std": 0.0, "min": 100, "max": 100, "cv": 0.0},
        angles_per_chunk=[3, 3, 3],
        angle_coverage={
            "mean_angles": 3.0,
            "std_angles": 0.0,
            "min_coverage_ratio": 1.0,
            "perfect_coverage_chunks": 3,
        },
        execution_time_ms=5.0,
        memory_overhead_mb=2.0,
        memory_efficiency=0.98,
        throughput_points_per_sec=60000.0,
        use_index_based=True,
    )

    # Access all fields to ensure they exist
    assert hasattr(diagnostics, "n_chunks")
    assert hasattr(diagnostics, "chunk_sizes")
    assert hasattr(diagnostics, "chunk_balance")
    assert hasattr(diagnostics, "angles_per_chunk")
    assert hasattr(diagnostics, "angle_coverage")
    assert hasattr(diagnostics, "execution_time_ms")
    assert hasattr(diagnostics, "memory_overhead_mb")
    assert hasattr(diagnostics, "memory_efficiency")
    assert hasattr(diagnostics, "throughput_points_per_sec")
    assert hasattr(diagnostics, "use_index_based")


# ============================================================================
# compute_stratification_diagnostics() Tests
# ============================================================================


def test_compute_diagnostics_balanced_data(balanced_data):
    """Test diagnostic computation with balanced dataset."""
    phi, t1, t2, g2 = balanced_data

    # Create stratified data
    phi_stratified, _, _, _, _ = create_angle_stratified_data(
        phi, t1, t2, g2, target_chunk_size=250
    )

    # Compute diagnostics
    diagnostics = compute_stratification_diagnostics(
        phi_original=phi,
        phi_stratified=phi_stratified,
        execution_time_ms=10.0,
        use_index_based=False,
        target_chunk_size=250,
    )

    # Validate structure
    assert isinstance(diagnostics, StratificationDiagnostics)
    assert diagnostics.n_chunks > 0
    assert len(diagnostics.chunk_sizes) == diagnostics.n_chunks
    assert len(diagnostics.angles_per_chunk) == diagnostics.n_chunks


def test_compute_diagnostics_chunk_balance(balanced_data):
    """Test chunk balance statistics are computed correctly."""
    phi, t1, t2, g2 = balanced_data

    phi_stratified, _, _, _, _ = create_angle_stratified_data(
        phi, t1, t2, g2, target_chunk_size=250
    )

    diagnostics = compute_stratification_diagnostics(
        phi_original=phi,
        phi_stratified=phi_stratified,
        execution_time_ms=10.0,
        use_index_based=False,
        target_chunk_size=250,
    )

    # Chunk balance should have all required keys
    assert "mean" in diagnostics.chunk_balance
    assert "std" in diagnostics.chunk_balance
    assert "min" in diagnostics.chunk_balance
    assert "max" in diagnostics.chunk_balance
    assert "cv" in diagnostics.chunk_balance

    # Values should be reasonable
    assert diagnostics.chunk_balance["mean"] > 0
    assert diagnostics.chunk_balance["std"] >= 0
    assert diagnostics.chunk_balance["min"] > 0
    assert diagnostics.chunk_balance["max"] >= diagnostics.chunk_balance["min"]
    assert diagnostics.chunk_balance["cv"] >= 0


def test_compute_diagnostics_angle_coverage(balanced_data):
    """Test angle coverage statistics are computed correctly."""
    phi, t1, t2, g2 = balanced_data

    phi_stratified, _, _, _, _ = create_angle_stratified_data(
        phi, t1, t2, g2, target_chunk_size=250
    )

    diagnostics = compute_stratification_diagnostics(
        phi_original=phi,
        phi_stratified=phi_stratified,
        execution_time_ms=10.0,
        use_index_based=False,
        target_chunk_size=250,
    )

    # Angle coverage should have all required keys
    assert "mean_angles" in diagnostics.angle_coverage
    assert "std_angles" in diagnostics.angle_coverage
    assert "min_coverage_ratio" in diagnostics.angle_coverage
    assert "perfect_coverage_chunks" in diagnostics.angle_coverage

    # Values should be reasonable
    assert diagnostics.angle_coverage["mean_angles"] > 0
    assert diagnostics.angle_coverage["std_angles"] >= 0
    assert 0.0 <= diagnostics.angle_coverage["min_coverage_ratio"] <= 1.0
    assert (
        0
        <= diagnostics.angle_coverage["perfect_coverage_chunks"]
        <= diagnostics.n_chunks
    )


def test_compute_diagnostics_perfect_coverage_balanced(balanced_data):
    """Test that balanced data achieves perfect coverage in all chunks."""
    phi, t1, t2, g2 = balanced_data

    phi_stratified, _, _, _, chunk_sizes = create_angle_stratified_data(
        phi, t1, t2, g2, target_chunk_size=250
    )

    diagnostics = compute_stratification_diagnostics(
        phi_original=phi,
        phi_stratified=phi_stratified,
        execution_time_ms=10.0,
        use_index_based=False,
        target_chunk_size=250,
        chunk_sizes=chunk_sizes,  # Use actual chunk boundaries from stratification
    )

    # With balanced data, all chunks should have all 3 angles (except possibly last chunk)
    # min_coverage_ratio should be high (>0.9)
    assert diagnostics.angle_coverage["min_coverage_ratio"] >= 0.9
    # Most chunks should have perfect coverage
    assert (
        diagnostics.angle_coverage["perfect_coverage_chunks"]
        >= diagnostics.n_chunks - 1
    )


def test_compute_diagnostics_performance_metrics(balanced_data):
    """Test performance metrics are computed correctly."""
    phi, t1, t2, g2 = balanced_data

    phi_stratified, _, _, _, _ = create_angle_stratified_data(
        phi, t1, t2, g2, target_chunk_size=250
    )

    execution_time_ms = 15.5
    diagnostics = compute_stratification_diagnostics(
        phi_original=phi,
        phi_stratified=phi_stratified,
        execution_time_ms=execution_time_ms,
        use_index_based=False,
        target_chunk_size=250,
    )

    # Execution time should match input
    assert diagnostics.execution_time_ms == execution_time_ms

    # Throughput should be reasonable (points per second)
    n_points = len(phi)
    expected_throughput = (n_points / execution_time_ms) * 1000.0
    assert abs(diagnostics.throughput_points_per_sec - expected_throughput) < 1.0


def test_compute_diagnostics_memory_metrics(balanced_data):
    """Test memory metrics are computed correctly."""
    phi, t1, t2, g2 = balanced_data

    phi_stratified, _, _, _, _ = create_angle_stratified_data(
        phi, t1, t2, g2, target_chunk_size=250
    )

    diagnostics = compute_stratification_diagnostics(
        phi_original=phi,
        phi_stratified=phi_stratified,
        execution_time_ms=10.0,
        use_index_based=False,
        target_chunk_size=250,
    )

    # Memory overhead should be positive for full copy
    assert diagnostics.memory_overhead_mb > 0

    # Memory efficiency should be between 0 and 1
    assert 0.0 < diagnostics.memory_efficiency <= 1.0

    # For full copy, efficiency should be ~0.5 (2x memory)
    assert 0.4 <= diagnostics.memory_efficiency <= 0.6


def test_compute_diagnostics_index_based_memory(balanced_data):
    """Test memory metrics for index-based stratification."""
    phi, t1, t2, g2 = balanced_data

    indices, _ = create_angle_stratified_indices(phi, target_chunk_size=250)
    phi_stratified = phi[indices]

    diagnostics = compute_stratification_diagnostics(
        phi_original=phi,
        phi_stratified=phi_stratified,
        execution_time_ms=10.0,
        use_index_based=True,
        target_chunk_size=250,
    )

    # Memory overhead should be lower for index-based
    assert diagnostics.memory_overhead_mb > 0

    # Memory efficiency should be higher (>0.7 for index-based)
    assert diagnostics.memory_efficiency > 0.7

    # use_index_based flag should be set
    assert diagnostics.use_index_based is True


def test_compute_diagnostics_imbalanced_data(imbalanced_data):
    """Test diagnostic computation with imbalanced dataset."""
    phi, t1, t2, g2 = imbalanced_data

    phi_stratified, _, _, _, _ = create_angle_stratified_data(
        phi, t1, t2, g2, target_chunk_size=250
    )

    diagnostics = compute_stratification_diagnostics(
        phi_original=phi,
        phi_stratified=phi_stratified,
        execution_time_ms=10.0,
        use_index_based=False,
        target_chunk_size=250,
    )

    # Chunk sizes may vary more with imbalanced data
    assert diagnostics.chunk_balance["cv"] >= 0

    # Coverage may be less perfect
    assert diagnostics.angle_coverage["min_coverage_ratio"] > 0


def test_compute_diagnostics_large_dataset(large_balanced_data):
    """Test diagnostic computation with large dataset."""
    phi, t1, t2, g2 = large_balanced_data

    phi_stratified, _, _, _, _ = create_angle_stratified_data(
        phi, t1, t2, g2, target_chunk_size=50000
    )

    diagnostics = compute_stratification_diagnostics(
        phi_original=phi,
        phi_stratified=phi_stratified,
        execution_time_ms=100.0,
        use_index_based=False,
        target_chunk_size=50000,
    )

    # Should have ~3 chunks for 150k points with 50k chunk size
    assert 2 <= diagnostics.n_chunks <= 4

    # Memory overhead should scale with dataset size
    # For full copy: 150k × 4 arrays × 8 bytes = 4.6 MB overhead
    assert diagnostics.memory_overhead_mb > 4.0  # >4 MB for 150k points


# ============================================================================
# format_diagnostics_report() Tests
# ============================================================================


def test_format_diagnostics_report_basic():
    """Test basic report formatting."""
    diagnostics = StratificationDiagnostics(
        n_chunks=5,
        chunk_sizes=[100, 100, 100, 100, 100],
        chunk_balance={"mean": 100.0, "std": 0.0, "min": 100, "max": 100, "cv": 0.0},
        angles_per_chunk=[3, 3, 3, 3, 3],
        angle_coverage={
            "mean_angles": 3.0,
            "std_angles": 0.0,
            "min_coverage_ratio": 1.0,
            "perfect_coverage_chunks": 5,
        },
        execution_time_ms=10.5,
        memory_overhead_mb=5.2,
        memory_efficiency=0.95,
        throughput_points_per_sec=47619.0,
        use_index_based=False,
    )

    report = format_diagnostics_report(diagnostics)

    # Report should be a string
    assert isinstance(report, str)

    # Report should contain key sections
    assert "STRATIFICATION DIAGNOSTICS REPORT" in report
    assert "Chunking:" in report
    assert "Chunk Balance:" in report
    assert "Angle Coverage:" in report
    assert "Performance:" in report
    assert "Memory:" in report


def test_format_diagnostics_report_contains_values():
    """Test report contains actual diagnostic values."""
    diagnostics = StratificationDiagnostics(
        n_chunks=7,
        chunk_sizes=[100] * 7,
        chunk_balance={"mean": 100.0, "std": 0.0, "min": 100, "max": 100, "cv": 0.0},
        angles_per_chunk=[3] * 7,
        angle_coverage={
            "mean_angles": 3.0,
            "std_angles": 0.0,
            "min_coverage_ratio": 1.0,
            "perfect_coverage_chunks": 7,
        },
        execution_time_ms=12.3,
        memory_overhead_mb=6.5,
        memory_efficiency=0.92,
        throughput_points_per_sec=56910.0,
        use_index_based=True,
    )

    report = format_diagnostics_report(diagnostics)

    # Check specific values appear in report
    assert "7" in report  # n_chunks
    assert "100" in report  # chunk size
    assert "3.0" in report or "3" in report  # angles
    assert "coverage" in report.lower()  # coverage section exists
    assert "12.3" in report  # execution time
    assert "index-based" in report.lower()  # stratification type


def test_format_diagnostics_report_formatting():
    """Test report is properly formatted."""
    diagnostics = StratificationDiagnostics(
        n_chunks=3,
        chunk_sizes=[100, 100, 100],
        chunk_balance={"mean": 100.0, "std": 0.0, "min": 100, "max": 100, "cv": 0.0},
        angles_per_chunk=[3, 3, 3],
        angle_coverage={
            "mean_angles": 3.0,
            "std_angles": 0.0,
            "min_coverage_ratio": 1.0,
            "perfect_coverage_chunks": 3,
        },
        execution_time_ms=10.0,
        memory_overhead_mb=5.0,
        memory_efficiency=0.95,
        throughput_points_per_sec=50000.0,
        use_index_based=False,
    )

    report = format_diagnostics_report(diagnostics)

    # Report should have multiple lines
    lines = report.split("\n")
    assert len(lines) > 5

    # Report should have header separator
    assert "=" in lines[0]

    # Lines should not be excessively long (70 char width target)
    for line in lines:
        assert len(line) <= 80  # Allow some margin


def test_format_diagnostics_report_index_based_indicator():
    """Test report correctly indicates index-based stratification."""
    diagnostics_full = StratificationDiagnostics(
        n_chunks=3,
        chunk_sizes=[100, 100, 100],
        chunk_balance={"mean": 100.0, "std": 0.0, "min": 100, "max": 100, "cv": 0.0},
        angles_per_chunk=[3, 3, 3],
        angle_coverage={
            "mean_angles": 3.0,
            "std_angles": 0.0,
            "min_coverage_ratio": 1.0,
            "perfect_coverage_chunks": 3,
        },
        execution_time_ms=10.0,
        memory_overhead_mb=10.0,
        memory_efficiency=0.5,
        throughput_points_per_sec=50000.0,
        use_index_based=False,
    )

    diagnostics_index = StratificationDiagnostics(
        n_chunks=3,
        chunk_sizes=[100, 100, 100],
        chunk_balance={"mean": 100.0, "std": 0.0, "min": 100, "max": 100, "cv": 0.0},
        angles_per_chunk=[3, 3, 3],
        angle_coverage={
            "mean_angles": 3.0,
            "std_angles": 0.0,
            "min_coverage_ratio": 1.0,
            "perfect_coverage_chunks": 3,
        },
        execution_time_ms=10.0,
        memory_overhead_mb=2.0,
        memory_efficiency=0.8,
        throughput_points_per_sec=50000.0,
        use_index_based=True,
    )

    report_full = format_diagnostics_report(diagnostics_full)
    report_index = format_diagnostics_report(diagnostics_index)

    # Full copy report
    assert "full-copy" in report_full.lower() or "full copy" in report_full.lower()

    # Index-based report
    assert "index-based" in report_index.lower() or "zero-copy" in report_index.lower()


# ============================================================================
# Integration Tests
# ============================================================================


def test_diagnostics_end_to_end_balanced(balanced_data):
    """Test full diagnostic workflow with balanced data."""
    phi, t1, t2, g2 = balanced_data

    # Stratify data
    phi_stratified, _, _, _, _ = create_angle_stratified_data(
        phi, t1, t2, g2, target_chunk_size=250
    )

    # Compute diagnostics
    diagnostics = compute_stratification_diagnostics(
        phi_original=phi,
        phi_stratified=phi_stratified,
        execution_time_ms=10.0,
        use_index_based=False,
        target_chunk_size=250,
    )

    # Format report
    report = format_diagnostics_report(diagnostics)

    # Validate complete workflow
    assert isinstance(diagnostics, StratificationDiagnostics)
    assert isinstance(report, str)
    assert len(report) > 100
    assert "STRATIFICATION DIAGNOSTICS REPORT" in report


def test_diagnostics_end_to_end_index_based(balanced_data):
    """Test full diagnostic workflow with index-based stratification."""
    phi, t1, t2, g2 = balanced_data

    # Stratify using index-based
    indices, _ = create_angle_stratified_indices(phi, target_chunk_size=250)
    phi_stratified = phi[indices]

    # Compute diagnostics
    diagnostics = compute_stratification_diagnostics(
        phi_original=phi,
        phi_stratified=phi_stratified,
        execution_time_ms=10.0,
        use_index_based=True,
        target_chunk_size=250,
    )

    # Format report
    report = format_diagnostics_report(diagnostics)

    # Validate
    assert diagnostics.use_index_based is True
    assert diagnostics.memory_efficiency > 0.7
    assert "index-based" in report.lower() or "zero-copy" in report.lower()


def test_diagnostics_comparison_full_vs_index(balanced_data):
    """Test diagnostic comparison between full copy and index-based."""
    phi, t1, t2, g2 = balanced_data
    target_chunk_size = 250

    # Full copy
    phi_full, _, _, _, _ = create_angle_stratified_data(
        phi, t1, t2, g2, target_chunk_size=target_chunk_size
    )
    diag_full = compute_stratification_diagnostics(
        phi_original=phi,
        phi_stratified=phi_full,
        execution_time_ms=20.0,
        use_index_based=False,
        target_chunk_size=target_chunk_size,
    )

    # Index-based
    indices, _ = create_angle_stratified_indices(
        phi, target_chunk_size=target_chunk_size
    )
    phi_index = phi[indices]
    diag_index = compute_stratification_diagnostics(
        phi_original=phi,
        phi_stratified=phi_index,
        execution_time_ms=15.0,
        use_index_based=True,
        target_chunk_size=target_chunk_size,
    )

    # Index-based should have better memory efficiency
    assert diag_index.memory_efficiency > diag_full.memory_efficiency
    assert diag_index.memory_overhead_mb < diag_full.memory_overhead_mb

    # Both should have similar chunking structure, but index-based coverage can drop
    assert diag_full.n_chunks == diag_index.n_chunks
    assert diag_index.angle_coverage["min_coverage_ratio"] >= 0.0
    assert (
        diag_full.angle_coverage["min_coverage_ratio"]
        >= diag_index.angle_coverage["min_coverage_ratio"]
    )
