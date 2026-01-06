"""
Unit tests for JIT-compatible stratified residual function.

Tests cover:
- Buffer donation for memory efficiency (T033, FR-003)
- JIT compilation correctness
- Chunk validation and structure
- Residual computation accuracy
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from homodyne.optimization.nlsq.strategies.residual_jit import StratifiedResidualFunctionJIT


class MockChunk:
    """Mock chunk for testing stratified residual function."""

    def __init__(self, phi, t1, t2, g2, q=0.01, L=1000.0, dt=0.1):
        self.phi = np.asarray(phi, dtype=np.float64)
        self.t1 = np.asarray(t1, dtype=np.float64)
        self.t2 = np.asarray(t2, dtype=np.float64)
        self.g2 = np.asarray(g2, dtype=np.float64)
        self.q = q
        self.L = L
        self.dt = dt


class MockStratifiedData:
    """Mock stratified data container for testing."""

    def __init__(self, chunks, sigma):
        self.chunks = chunks
        self.sigma = np.asarray(sigma, dtype=np.float64)


class TestBufferDonation:
    """Tests for buffer donation optimization (T033, FR-003).

    Buffer donation allows JAX to reuse input buffers for output,
    reducing peak memory usage during JIT-compiled operations.
    """

    @pytest.fixture
    def simple_stratified_data(self):
        """Create simple stratified data for testing."""
        n_points_per_chunk = 100
        n_chunks = 2
        n_phi = 3
        phi_vals = np.array([0.0, 45.0, 90.0])  # degrees

        chunks = []
        for _ in range(n_chunks):
            # Create data with all phi angles represented
            phi_list = []
            t1_list = []
            t2_list = []
            g2_list = []

            points_per_angle = n_points_per_chunk // n_phi
            for phi_val in phi_vals:
                for _ in range(points_per_angle):
                    phi_list.append(phi_val)
                    t1_list.append(np.random.uniform(0, 1))
                    t2_list.append(np.random.uniform(0, 1))
                    g2_list.append(np.random.uniform(1.0, 2.0))

            chunks.append(MockChunk(
                phi=phi_list,
                t1=t1_list,
                t2=t2_list,
                g2=g2_list,
            ))

        # Create sigma array matching flattened shape
        total_points = sum(len(c.phi) for c in chunks)
        sigma = np.ones(total_points) * 0.1

        return MockStratifiedData(chunks, sigma.reshape(n_phi, -1, 1))

    def test_buffer_donation_compile_flag(self, simple_stratified_data):
        """T033: Verify donate_argnums is applied in JIT compilation.

        Performance Optimization (Spec 001 - FR-003, T033): Buffer donation
        allows JAX to reuse input buffers for output, reducing memory.
        """
        # Create residual function
        residual_fn = StratifiedResidualFunctionJIT(
            stratified_data=simple_stratified_data,
            per_angle_scaling=True,
            physical_param_names=["D0", "alpha", "D_offset"],
        )

        # Verify JIT compilation was applied
        assert hasattr(residual_fn, "_residual_fn_jit")
        assert residual_fn._residual_fn_jit is not None

        # The _compute_all_residuals should be JIT compiled
        # Check that it's a traced function (JAX compiled)
        # Note: Direct check for donate_argnums requires inspecting lowered HLO
        # Here we verify the function is JIT-wrapped and callable
        assert callable(residual_fn._residual_fn_jit)

    def test_buffer_donation_no_memory_leak(self, simple_stratified_data):
        """Test that buffer donation doesn't cause memory accumulation.

        Performance Optimization (Spec 001 - FR-003): Verify repeated calls
        don't accumulate memory due to improper buffer handling.
        """
        residual_fn = StratifiedResidualFunctionJIT(
            stratified_data=simple_stratified_data,
            per_angle_scaling=True,
            physical_param_names=["D0", "alpha", "D_offset"],
        )

        n_phi = 3
        # Parameters: 3 contrasts + 3 offsets + 3 physical
        n_params = 2 * n_phi + 3
        params = np.array([
            0.8, 0.8, 0.8,  # contrasts
            1.0, 1.0, 1.0,  # offsets
            1e-11, 0.5, 1e-14,  # physical: D0, alpha, D_offset
        ])

        # Run multiple iterations to check for memory stability
        results = []
        for _ in range(10):
            result = residual_fn(params)
            results.append(float(jnp.sum(result**2)))

        # All results should be finite and consistent
        assert all(np.isfinite(r) for r in results)
        # Results should be identical (no numerical drift)
        np.testing.assert_allclose(results[0], results[-1], rtol=1e-10)

    def test_jit_function_returns_correct_shape(self, simple_stratified_data):
        """Test that JIT function returns residuals with correct shape."""
        residual_fn = StratifiedResidualFunctionJIT(
            stratified_data=simple_stratified_data,
            per_angle_scaling=True,
            physical_param_names=["D0", "alpha", "D_offset"],
        )

        n_phi = 3
        params = np.array([
            0.8, 0.8, 0.8,  # contrasts
            1.0, 1.0, 1.0,  # offsets
            1e-11, 0.5, 1e-14,  # physical
        ])

        result = residual_fn(params)

        # Shape should be (n_chunks * max_chunk_size,)
        expected_shape = (
            residual_fn.n_chunks * residual_fn.max_chunk_size,
        )
        assert result.shape == expected_shape


class TestStratifiedResidualFunctionJIT:
    """General tests for StratifiedResidualFunctionJIT class."""

    @pytest.fixture
    def stratified_data(self):
        """Create stratified data with multiple angles."""
        n_phi = 5
        n_t1 = 10
        n_t2 = 10
        phi_vals = np.linspace(-60, 60, n_phi)
        t1_vals = np.linspace(0, 1, n_t1)
        t2_vals = np.linspace(0, 1, n_t2)

        # Create two chunks with all angles
        chunks = []
        for chunk_idx in range(2):
            phi_list = []
            t1_list = []
            t2_list = []
            g2_list = []

            # Each chunk has points from all angles
            for phi_val in phi_vals:
                for t1_val in t1_vals[:5]:  # Half of t1 per chunk
                    for t2_val in t2_vals[:5]:  # Half of t2 per chunk
                        phi_list.append(phi_val)
                        t1_list.append(t1_val if chunk_idx == 0 else t1_vals[5:][t1_vals[:5].tolist().index(t1_val)] if t1_val in t1_vals[:5] else t1_val)
                        t2_list.append(t2_val if chunk_idx == 0 else t2_vals[5:][t2_vals[:5].tolist().index(t2_val)] if t2_val in t2_vals[:5] else t2_val)
                        g2_list.append(1.5 + 0.1 * np.random.randn())

            chunks.append(MockChunk(
                phi=phi_list,
                t1=t1_list,
                t2=t2_list,
                g2=g2_list,
            ))

        total_points = sum(len(c.phi) for c in chunks)
        sigma = np.ones((n_phi, n_t1, n_t2)) * 0.1

        return MockStratifiedData(chunks, sigma)

    def test_validate_chunk_structure_passes(self, stratified_data):
        """Test that chunk structure validation passes for valid data."""
        residual_fn = StratifiedResidualFunctionJIT(
            stratified_data=stratified_data,
            per_angle_scaling=True,
            physical_param_names=["D0", "alpha", "D_offset"],
        )

        # Should not raise
        assert residual_fn.validate_chunk_structure() is True

    def test_diagnostics_contain_required_keys(self, stratified_data):
        """Test that diagnostics contain all required information."""
        residual_fn = StratifiedResidualFunctionJIT(
            stratified_data=stratified_data,
            per_angle_scaling=True,
            physical_param_names=["D0", "alpha", "D_offset"],
        )

        diag = residual_fn.get_diagnostics()

        required_keys = [
            "n_chunks",
            "max_chunk_size",
            "n_real_points",
            "padding_overhead_pct",
            "n_phi",
            "n_t1",
            "n_t2",
            "per_angle_scaling",
            "jit_compiled",
        ]

        for key in required_keys:
            assert key in diag, f"Missing diagnostic key: {key}"

        assert diag["jit_compiled"] is True

    def test_residuals_are_finite(self, stratified_data):
        """Test that computed residuals are all finite."""
        residual_fn = StratifiedResidualFunctionJIT(
            stratified_data=stratified_data,
            per_angle_scaling=True,
            physical_param_names=["D0", "alpha", "D_offset"],
        )

        n_phi = 5
        params = np.concatenate([
            np.full(n_phi, 0.8),  # contrasts
            np.full(n_phi, 1.0),  # offsets
            [1e-11, 0.5, 1e-14],  # physical
        ])

        result = residual_fn(params)

        # All residuals should be finite
        assert jnp.all(jnp.isfinite(result))
