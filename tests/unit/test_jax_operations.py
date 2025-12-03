"""
Unit tests for JAX operations and indexing behavior.

This test suite verifies that JAX fancy indexing works correctly with traced indices,
addressing the misdiagnosed TracerArrayConversionError from the NLSQ zero-iteration issue.

Historical Context:
- Original investigation blamed "TracerArrayConversionError from dynamic indexing"
- These tests prove JAX fancy indexing works correctly with JIT compilation
- Root cause was actually per-angle scaling incompatibility with NLSQ chunking

See: .ultra-think/ROOT_CAUSE_FOUND.md for full investigation
"""

import jax
import jax.numpy as jnp
import pytest


class TestJAXFancyIndexing:
    """Test JAX fancy indexing with traced indices in JIT-compiled functions."""

    def test_small_array_traced_indices(self):
        """Test fancy indexing with small array (1k elements) and traced indices."""

        @jax.jit
        def small_array_indexing(indices):
            arr = jnp.arange(1000)
            return arr[indices]

        indices = jnp.array([0, 100, 500, 999])
        result = small_array_indexing(indices)

        # Verify indexing worked correctly
        assert jnp.array_equal(result, jnp.array([0, 100, 500, 999]))
        assert result.shape == (4,)

    def test_large_array_traced_indices(self):
        """Test fancy indexing with large array (3M elements) simulating g2_theory_flat."""

        @jax.jit
        def large_array_indexing(indices):
            arr = jnp.arange(3_000_000)  # 3M elements like g2_theory_flat
            return arr[indices]

        # Simulate chunk of 100k indices
        indices = jnp.arange(100000)
        result = large_array_indexing(indices)

        # Verify indexing worked correctly
        assert result.shape == (100000,)
        assert jnp.array_equal(result[:5], jnp.array([0, 1, 2, 3, 4]))
        assert result[-1] == 99999

    def test_very_large_array_traced_indices(self):
        """Test fancy indexing with very large array (23M elements) simulating real dataset."""

        @jax.jit
        def very_large_array_indexing(indices):
            arr = jnp.arange(23_000_000)  # 23M elements
            return arr[indices]

        indices = jnp.arange(100000)
        result = very_large_array_indexing(indices)

        # Verify indexing worked correctly
        assert result.shape == (100000,)
        assert jnp.array_equal(result[:5], jnp.array([0, 1, 2, 3, 4]))

    def test_type_conversion_with_indexing(self):
        """Test type conversion followed by indexing (original code pattern)."""

        @jax.jit
        def type_conversion_indexing(xdata):
            arr = jnp.arange(3_000_000, dtype=jnp.float64)  # Simulate g2_theory_flat
            indices = xdata.astype(jnp.int32)  # Original code pattern
            return arr[indices]

        xdata = jnp.arange(100000, dtype=jnp.float64)  # Simulate xdata
        result = type_conversion_indexing(xdata)

        # Verify type conversion + indexing worked
        assert result.shape == (100000,)
        assert result.dtype == jnp.float64
        assert jnp.array_equal(result[:5], jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]))

    def test_nested_computation_with_indexing(self):
        """Test nested computation + indexing (real scenario simulation)."""

        def compute_g2_mock(params):
            """Mock computation that creates large array."""
            # Simulate: g2_theory = compute_g2_scaled_vmap(phi)
            # Shape: (23, 1001, 1001) = 23M points
            g2 = jnp.ones((23, 1001, 1001)) * params[0]
            return g2.flatten()

        @jax.jit
        def model_function_mock(xdata, param):
            g2_theory_flat = compute_g2_mock(jnp.array([param]))
            indices = xdata.astype(jnp.int32)
            return g2_theory_flat[indices]

        xdata = jnp.arange(100000, dtype=jnp.float64)
        param = 1.5
        result = model_function_mock(xdata, param)

        # Verify nested computation + indexing worked
        assert result.shape == (100000,)
        assert jnp.allclose(result, 1.5)  # All values should be ~1.5

    def test_random_indices_out_of_order(self):
        """Test fancy indexing with random out-of-order indices."""

        @jax.jit
        def random_index_access(indices):
            arr = jnp.arange(10000)
            return arr[indices]

        # Test with non-sequential indices
        indices = jnp.array([9999, 5000, 100, 7, 4567])
        result = random_index_access(indices)

        assert jnp.array_equal(result, indices)

    def test_boolean_mask_indexing(self):
        """
        Test boolean mask indexing (another form of fancy indexing).

        Note: Boolean indexing with traced arrays is not supported in JIT-compiled
        functions due to shape uncertainty. This test uses non-JIT version.
        """
        # Boolean indexing works without JIT
        arr = jnp.arange(1000)
        mask = jnp.arange(1000) % 2 == 0
        result = arr[mask]

        assert result.shape == (500,)
        assert jnp.array_equal(result[:5], jnp.array([0, 2, 4, 6, 8]))


class TestJAXGradientFlow:
    """Test JAX gradient computation through indexed operations."""

    def test_gradient_through_fancy_indexing(self):
        """Verify gradients flow correctly through fancy indexing operations."""

        def loss_fn(param):
            # Create array that depends on param
            arr = jnp.arange(1000, dtype=jnp.float32) * param
            # Index with fixed indices
            indices = jnp.array([0, 100, 500])
            selected = arr[indices]
            return jnp.sum(selected)

        # Compute gradient
        grad_fn = jax.grad(loss_fn)
        gradient = grad_fn(2.0)

        # Gradient should be sum of selected indices: 0 + 100 + 500 = 600
        expected_gradient = 0.0 + 100.0 + 500.0
        assert jnp.isclose(gradient, expected_gradient)

    def test_gradient_with_parameter_dependent_computation(self):
        """Test gradient when both array values and indexing depend on parameters."""

        def loss_fn(param):
            # Array values depend on param
            arr = jnp.ones(1000) * param
            # Fixed indices
            indices = jnp.array([0, 10, 20, 30])
            selected = arr[indices]
            return jnp.sum(selected**2)  # Nonlinear to test gradient

        grad_fn = jax.grad(loss_fn)
        gradient = grad_fn(2.0)

        # d/dp sum((p)^2) for 4 elements = d/dp (4p^2) = 8p
        # At p=2.0: 8*2.0 = 16.0
        assert jnp.isclose(gradient, 16.0)


if __name__ == "__main__":
    # Allow running directly for quick debugging
    pytest.main([__file__, "-v"])
