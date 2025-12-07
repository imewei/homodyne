"""
Unit tests for StratifiedResidualFunction.

Tests the angle-stratified residual function used for solving the NLSQ
double-chunking problem with per-angle scaling parameters.

Author: Homodyne Development Team
Date: 2025-11-06
Version: 2.2.0
"""

import numpy as np
import pytest

from homodyne.optimization.nlsq.strategies.residual import (
    StratifiedResidualFunction,
    create_stratified_residual_function,
)


# Test fixtures
@pytest.fixture
def mock_stratified_data_small():
    """Create small mock stratified data for testing (3 angles, 100 points per angle)."""
    n_phi = 3
    n_points_per_angle = 100
    n_phi * n_points_per_angle

    # Create mock chunks (2 chunks)
    n_chunks = 2

    class MockChunk:
        def __init__(self, phi, t1, t2, g2, q, L, dt):
            self.phi = phi
            self.t1 = t1
            self.t2 = t2
            self.g2 = g2
            self.q = q
            self.L = L
            self.dt = dt

    # Generate data
    phi_vals = np.array([0.0, np.pi / 4, np.pi / 2])  # 3 angles
    t1_vals = np.linspace(0.001, 1.0, 10)
    t2_vals = np.linspace(0.001, 1.0, 10)

    # Create meshgrid
    phi_grid, t1_grid, t2_grid = np.meshgrid(phi_vals, t1_vals, t2_vals, indexing="ij")

    # Create sigma (3D array: n_phi × n_t1 × n_t2) - stored at parent level
    sigma = np.ones((n_phi, len(t1_vals), len(t2_vals))) * 0.1

    # Create angle-stratified chunks (each chunk has all phi angles)
    # Split the (t1, t2) grid into chunks while keeping all phi angles
    n_t_pairs = len(t1_vals) * len(t2_vals)  # 100 pairs
    t_pairs_per_chunk = n_t_pairs // n_chunks  # 50 pairs per chunk

    chunks = []
    for i in range(n_chunks):
        # For this chunk, take a subset of (t1, t2) pairs but ALL phi angles
        t_start = i * t_pairs_per_chunk
        t_end = min(t_start + t_pairs_per_chunk, n_t_pairs)

        # Create arrays with all phi angles for this subset of (t1, t2) pairs
        chunk_phi = []
        chunk_t1 = []
        chunk_t2 = []
        chunk_g2 = []

        for phi_idx, _phi_val in enumerate(phi_vals):
            # Flatten the (t1, t2) grid for this phi
            phi_slice = phi_grid[phi_idx].flatten()
            t1_slice = t1_grid[phi_idx].flatten()
            t2_slice = t2_grid[phi_idx].flatten()

            # Take the subset for this chunk
            chunk_phi.extend(phi_slice[t_start:t_end])
            chunk_t1.extend(t1_slice[t_start:t_end])
            chunk_t2.extend(t2_slice[t_start:t_end])
            chunk_g2.extend([1.5] * (t_end - t_start))

        chunk = MockChunk(
            phi=np.array(chunk_phi),
            t1=np.array(chunk_t1),
            t2=np.array(chunk_t2),
            g2=np.array(chunk_g2),
            q=0.02,  # 1/Å
            L=1000.0,  # mm
            dt=0.001,  # seconds
        )
        chunks.append(chunk)

    class MockStratifiedData:
        def __init__(self, chunks, sigma):
            self.chunks = chunks
            self.sigma = sigma  # Store sigma at parent level

    return MockStratifiedData(chunks, sigma)


@pytest.fixture
def physical_param_names_static():
    """Physical parameter names for static_mode mode."""
    return ["D0", "alpha", "D_offset"]


@pytest.fixture
def physical_param_names_laminar():
    """Physical parameter names for laminar_flow mode."""
    return [
        "D0",
        "alpha",
        "D_offset",
        "gamma_dot_0",
        "beta",
        "gamma_dot_offset",
        "phi0",
    ]


# Test 1: Initialization
def test_initialization_per_angle_scaling(
    mock_stratified_data_small, physical_param_names_static
):
    """Test StratifiedResidualFunction initialization with per-angle scaling."""
    residual_fn = StratifiedResidualFunction(
        stratified_data=mock_stratified_data_small,
        per_angle_scaling=True,
        physical_param_names=physical_param_names_static,
    )

    assert residual_fn.n_chunks == 2
    assert residual_fn.n_total_points == 300  # 3 angles × 100 points
    assert residual_fn.n_phi == 3
    assert residual_fn.n_scaling_params == 6  # 2 * n_phi (contrast + offset per angle)
    assert residual_fn.n_physical_params == 3
    assert residual_fn.n_total_params == 9  # 6 scaling + 3 physical


def test_initialization_laminar_flow(
    mock_stratified_data_small, physical_param_names_laminar
):
    """Test initialization with laminar_flow parameters."""
    residual_fn = StratifiedResidualFunction(
        stratified_data=mock_stratified_data_small,
        per_angle_scaling=True,
        physical_param_names=physical_param_names_laminar,
    )

    assert residual_fn.n_physical_params == 7
    assert residual_fn.n_total_params == 13  # 6 scaling + 7 physical


def test_initialization_empty_chunks_raises():
    """Test that empty chunks raise ValueError."""

    class EmptyData:
        def __init__(self):
            self.chunks = []
            self.sigma = np.array([])  # Add sigma attribute

    with pytest.raises(ValueError, match="stratified_data.chunks is empty"):
        StratifiedResidualFunction(
            stratified_data=EmptyData(),
            per_angle_scaling=True,
            physical_param_names=["D0", "alpha", "D_offset"],
        )


# Test 2: Chunk Structure Validation
def test_validate_chunk_structure_success(
    mock_stratified_data_small, physical_param_names_static
):
    """Test successful chunk structure validation."""
    residual_fn = StratifiedResidualFunction(
        stratified_data=mock_stratified_data_small,
        per_angle_scaling=True,
        physical_param_names=physical_param_names_static,
    )

    # Should not raise
    result = residual_fn.validate_chunk_structure()
    assert result is True


def test_validate_chunk_structure_missing_angles():
    """Test validation failure when chunks missing angles."""
    n_phi = 3
    phi_vals = np.array([0.0, np.pi / 4, np.pi / 2])
    t1_vals = np.linspace(0.001, 1.0, 10)
    t2_vals = np.linspace(0.001, 1.0, 10)

    # Create chunk 1 with all angles
    phi_grid, t1_grid, t2_grid = np.meshgrid(phi_vals, t1_vals, t2_vals, indexing="ij")

    class MockChunk:
        def __init__(self, phi, t1, t2, g2, q, L, dt):
            self.phi = phi
            self.t1 = t1
            self.t2 = t2
            self.g2 = g2
            self.q = q
            self.L = L
            self.dt = dt

    sigma = np.ones((n_phi, len(t1_vals), len(t2_vals))) * 0.1

    # Create chunk 1 with all 3 angles (first half of (t1, t2) pairs)
    chunk1_phi = []
    chunk1_t1 = []
    chunk1_t2 = []
    for phi_val in phi_vals:
        # Add first 50 (t1, t2) pairs for this angle
        chunk1_phi.extend([phi_val] * 50)
        t1_flat = t1_vals.repeat(len(t2_vals))[:50]
        t2_flat = np.tile(t2_vals, len(t1_vals))[:50]
        chunk1_t1.extend(t1_flat)
        chunk1_t2.extend(t2_flat)

    chunk1 = MockChunk(
        phi=np.array(chunk1_phi),
        t1=np.array(chunk1_t1),
        t2=np.array(chunk1_t2),
        g2=np.ones(150) * 1.5,
        q=0.02,
        L=1000.0,
        dt=0.001,
    )

    # Create chunk 2 with missing angle (only first 2 angles - missing π/2)
    chunk2_phi = []
    chunk2_t1 = []
    chunk2_t2 = []
    for phi_val in [0.0, np.pi / 4]:  # Missing π/2
        # Add second half of (t1, t2) pairs for this angle
        chunk2_phi.extend([phi_val] * 50)
        t1_flat = t1_vals.repeat(len(t2_vals))[50:100]
        t2_flat = np.tile(t2_vals, len(t1_vals))[50:100]
        chunk2_t1.extend(t1_flat)
        chunk2_t2.extend(t2_flat)

    chunk2 = MockChunk(
        phi=np.array(chunk2_phi),
        t1=np.array(chunk2_t1),
        t2=np.array(chunk2_t2),
        g2=np.ones(100) * 1.5,
        q=0.02,
        L=1000.0,
        dt=0.001,
    )

    class MockData:
        def __init__(self, chunks, sigma):
            self.chunks = chunks
            self.sigma = sigma

    residual_fn = StratifiedResidualFunction(
        stratified_data=MockData([chunk1, chunk2], sigma),
        per_angle_scaling=True,
        physical_param_names=["D0", "alpha", "D_offset"],
    )

    with pytest.raises(ValueError, match="Chunk .* has inconsistent angles"):
        residual_fn.validate_chunk_structure()


# Test 3: Residual Computation
def test_call_returns_correct_shape_per_angle(
    mock_stratified_data_small, physical_param_names_static
):
    """Test that __call__ returns residuals with correct shape (per-angle mode)."""
    residual_fn = StratifiedResidualFunction(
        stratified_data=mock_stratified_data_small,
        per_angle_scaling=True,
        physical_param_names=physical_param_names_static,
    )

    # Create mock parameters: [contrast_0, contrast_1, contrast_2, offset_0, offset_1, offset_2, D0, alpha, D_offset]
    params = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1e-10, 1.0, 0.0])

    residuals = residual_fn(params)

    # The implementation can return either NumPy or JAX arrays; normalize to NumPy
    residuals_np = np.asarray(residuals)
    assert residuals_np.shape == (300,)  # Total points across all chunks
    assert np.all(np.isfinite(residuals_np))


def test_residuals_are_finite_and_reasonable(
    mock_stratified_data_small, physical_param_names_static
):
    """Test that residuals are finite and within reasonable range."""
    residual_fn = StratifiedResidualFunction(
        stratified_data=mock_stratified_data_small,
        per_angle_scaling=True,
        physical_param_names=physical_param_names_static,
    )

    # Use parameters that don't perfectly match mock data (g2=1.5)
    # contrast=0.4 gives g2_theory ≈ 0.4*1^2 + 1.0 = 1.4 ≠ 1.5
    params = np.array([0.4, 0.4, 0.4, 1.0, 1.0, 1.0, 1e-10, 1.0, 0.0])
    residuals = residual_fn(params)

    # Check all residuals are finite
    assert np.all(np.isfinite(residuals))

    # Check residuals are not all zero (optimization is doing something)
    assert not np.allclose(residuals, 0.0)

    # Check residuals are weighted (divided by sigma)
    # With sigma=0.1, residuals should be scaled up
    assert np.abs(residuals).mean() > 0.1  # Some residuals should be non-negligible


# Test 4: Diagnostics
def test_get_diagnostics(mock_stratified_data_small, physical_param_names_static):
    """Test get_diagnostics returns expected information."""
    residual_fn = StratifiedResidualFunction(
        stratified_data=mock_stratified_data_small,
        per_angle_scaling=True,
        physical_param_names=physical_param_names_static,
    )

    diagnostics = residual_fn.get_diagnostics()

    assert diagnostics["n_chunks"] == 2
    assert diagnostics["n_total_points"] == 300
    assert diagnostics["n_angles"] == 3
    assert diagnostics["per_angle_scaling"] is True
    assert len(diagnostics["chunk_sizes"]) == 2
    assert diagnostics["min_chunk_size"] == 150
    assert diagnostics["max_chunk_size"] == 150
    assert "chunk_angle_counts" in diagnostics


def test_log_diagnostics_no_error(
    mock_stratified_data_small, physical_param_names_static
):
    """Test that log_diagnostics doesn't raise errors."""
    residual_fn = StratifiedResidualFunction(
        stratified_data=mock_stratified_data_small,
        per_angle_scaling=True,
        physical_param_names=physical_param_names_static,
    )

    # Should not raise
    residual_fn.log_diagnostics()


# Test 5: Factory Function
def test_create_factory_function(
    mock_stratified_data_small, physical_param_names_static
):
    """Test create_stratified_residual_function factory."""
    residual_fn = create_stratified_residual_function(
        stratified_data=mock_stratified_data_small,
        per_angle_scaling=True,
        physical_param_names=physical_param_names_static,
        validate=True,
    )

    assert isinstance(residual_fn, StratifiedResidualFunction)
    assert residual_fn.n_total_params == 9  # 6 scaling + 3 physical


def test_factory_validation_can_be_disabled(
    mock_stratified_data_small, physical_param_names_static
):
    """Test that factory validation can be disabled."""
    residual_fn = create_stratified_residual_function(
        stratified_data=mock_stratified_data_small,
        per_angle_scaling=True,
        physical_param_names=physical_param_names_static,
        validate=False,
    )

    assert isinstance(residual_fn, StratifiedResidualFunction)


# Test 6: Parameter Structure
def test_per_angle_parameter_structure():
    """Test that per-angle parameters are structured correctly."""
    # For 3 angles, static_mode mode:
    # params = [contrast_0, contrast_1, contrast_2, offset_0, offset_1, offset_2, D0, alpha, D_offset]
    n_phi = 3
    n_scaling = 2 * n_phi  # 6
    n_physical = 3
    n_total = n_scaling + n_physical  # 9

    assert n_total == 9
    assert n_scaling == 6


def test_legacy_parameter_structure():
    """Test that legacy parameters are structured correctly."""
    # Legacy mode:
    # params = [contrast, offset, D0, alpha, D_offset]
    n_scaling = 2  # Single contrast + offset
    n_physical = 3
    n_total = n_scaling + n_physical  # 5

    assert n_total == 5


# Test 7: JIT Compilation
def test_jit_compilation_works(mock_stratified_data_small, physical_param_names_static):
    """Test that JIT compilation is set up correctly."""
    residual_fn = StratifiedResidualFunction(
        stratified_data=mock_stratified_data_small,
        per_angle_scaling=True,
        physical_param_names=physical_param_names_static,
    )

    # Check that JIT function exists
    assert hasattr(residual_fn, "compute_chunk_jit")
    assert callable(residual_fn.compute_chunk_jit)


def test_multiple_calls_consistent(
    mock_stratified_data_small, physical_param_names_static
):
    """Test that multiple calls with same parameters give consistent results."""
    residual_fn = StratifiedResidualFunction(
        stratified_data=mock_stratified_data_small,
        per_angle_scaling=True,
        physical_param_names=physical_param_names_static,
    )

    params = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1e-10, 1.0, 0.0])

    residuals1 = residual_fn(params)
    residuals2 = residual_fn(params)

    np.testing.assert_allclose(residuals1, residuals2, rtol=1e-10)


# Test 8: Edge Cases
def test_single_chunk(physical_param_names_static):
    """Test with single chunk."""
    n_phi = 3
    phi_vals = np.array([0.0, np.pi / 4, np.pi / 2])
    t1_vals = np.linspace(0.001, 1.0, 10)
    t2_vals = np.linspace(0.001, 1.0, 10)

    phi_grid, t1_grid, t2_grid = np.meshgrid(phi_vals, t1_vals, t2_vals, indexing="ij")

    class MockChunk:
        def __init__(self, phi, t1, t2, g2, sigma, q, L, dt):
            self.phi = phi
            self.t1 = t1
            self.t2 = t2
            self.g2 = g2
            self.sigma = sigma
            self.q = q
            self.L = L
            self.dt = dt

    sigma = np.ones((n_phi, len(t1_vals), len(t2_vals))) * 0.1

    chunk = MockChunk(
        phi=phi_grid.flatten(),
        t1=t1_grid.flatten(),
        t2=t2_grid.flatten(),
        g2=np.ones(300) * 1.5,
        sigma=sigma,
        q=0.02,
        L=1000.0,
        dt=0.001,
    )

    class MockData:
        def __init__(self, chunks, sigma):
            self.chunks = chunks
            self.sigma = sigma

    residual_fn = StratifiedResidualFunction(
        stratified_data=MockData([chunk], sigma),
        per_angle_scaling=True,
        physical_param_names=physical_param_names_static,
    )

    params = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1e-10, 1.0, 0.0])
    residuals = residual_fn(params)

    assert residuals.shape == (300,)
    assert np.all(np.isfinite(residuals))


def test_large_parameter_array(
    mock_stratified_data_small, physical_param_names_laminar
):
    """Test with larger parameter array (laminar_flow)."""
    residual_fn = StratifiedResidualFunction(
        stratified_data=mock_stratified_data_small,
        per_angle_scaling=True,
        physical_param_names=physical_param_names_laminar,
    )

    # 6 scaling + 7 physical = 13 parameters
    params = np.array(
        [
            0.5,
            0.5,
            0.5,  # contrast per angle
            1.0,
            1.0,
            1.0,  # offset per angle
            1e-10,
            1.0,
            0.0,  # D0, alpha, D_offset
            1.0,
            1.0,
            0.0,
            0.0,  # gamma_dot_0, beta, gamma_dot_offset, phi0
        ]
    )

    residuals = residual_fn(params)

    assert residuals.shape == (300,)
    assert np.all(np.isfinite(residuals))


# Test 9: Integration with NLSQ
def test_compatible_with_nlsq_least_squares(
    mock_stratified_data_small, physical_param_names_static
):
    """Test that residual function is compatible with NLSQ's least_squares signature."""
    residual_fn = StratifiedResidualFunction(
        stratified_data=mock_stratified_data_small,
        per_angle_scaling=True,
        physical_param_names=physical_param_names_static,
    )

    # NLSQ's least_squares expects: fun(params) -> residuals
    params = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1e-10, 1.0, 0.0])

    # Test callable signature
    assert callable(residual_fn)

    # Test return type - v2.2.1: Returns NumPy arrays (converted from JAX to prevent device memory accumulation)
    residuals = residual_fn(params)
    # StratifiedResidualFunction returns NumPy arrays, not JAX arrays
    # (JAX → NumPy conversion happens in __call__ to prevent memory leaks)
    assert isinstance(residuals, np.ndarray)
    assert residuals.ndim == 1  # 1D array required by least_squares
    # Verify it's compatible with least_squares (accepts both NumPy and JAX arrays)
    assert len(residuals) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
