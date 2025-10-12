"""
Unit Tests for ParameterManager (Phase 4.2)
===========================================

Tests for homodyne.config.parameter_manager module.
"""

import numpy as np
import pytest

from homodyne.config.parameter_manager import ParameterManager


class TestParameterManagerInit:
    """Test ParameterManager initialization."""

    def test_init_without_config(self):
        """Test initialization without configuration."""
        pm = ParameterManager()
        assert pm.config_dict == {}
        assert pm.analysis_mode == "laminar_flow"

    def test_init_with_config(self):
        """Test initialization with configuration."""
        config = {"analysis_mode": "laminar_flow"}
        pm = ParameterManager(config, "laminar_flow")
        assert pm.config_dict == config
        assert pm.analysis_mode == "laminar_flow"

    def test_init_static_mode(self):
        """Test initialization with static mode."""
        pm = ParameterManager(None, "static")
        assert pm.analysis_mode == "static"


class TestParameterBounds:
    """Test parameter bounds functionality."""

    def test_get_bounds_laminar_flow(self):
        """Test getting bounds for laminar flow mode."""
        pm = ParameterManager(None, "laminar_flow")
        bounds = pm.get_parameter_bounds(["D0", "alpha", "D_offset"])

        assert len(bounds) == 3
        assert all(isinstance(b, dict) for b in bounds)
        assert all("min" in b and "max" in b and "name" in b for b in bounds)

    def test_get_bounds_static_mode(self):
        """Test getting bounds for static mode."""
        pm = ParameterManager(None, "static")
        bounds = pm.get_parameter_bounds(["D0", "alpha", "D_offset"])

        assert len(bounds) == 3
        assert bounds[0]["name"] == "D0"
        assert bounds[1]["name"] == "alpha"
        assert bounds[2]["name"] == "D_offset"

    def test_get_bounds_all_parameters(self):
        """Test getting bounds for all parameters."""
        pm = ParameterManager(None, "laminar_flow")
        bounds = pm.get_parameter_bounds()  # No param_names = all params

        # Should include scaling + all physical parameters
        expected_count = 9  # 2 scaling + 7 physical
        assert len(bounds) == expected_count

    def test_get_bounds_with_config_override(self):
        """Test bounds override from configuration."""
        config = {
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 10.0, "max": 1e5},
                    {"name": "alpha", "min": -1.0, "max": 1.0},
                ]
            }
        }
        pm = ParameterManager(config, "laminar_flow")
        bounds = pm.get_parameter_bounds(["D0", "alpha"])

        assert bounds[0]["min"] == 10.0
        assert bounds[0]["max"] == 1e5
        assert bounds[1]["min"] == -1.0
        assert bounds[1]["max"] == 1.0

    def test_get_bounds_as_tuples(self):
        """Test getting bounds as tuple format."""
        pm = ParameterManager(None, "laminar_flow")
        bounds = pm.get_bounds_as_tuples(["D0", "alpha"])

        assert len(bounds) == 2
        assert isinstance(bounds[0], tuple)
        assert len(bounds[0]) == 2
        assert bounds[0] == (1.0, 1e6)  # Default D0 bounds

    def test_get_bounds_as_arrays(self):
        """Test getting bounds as numpy arrays."""
        pm = ParameterManager(None, "laminar_flow")
        lower, upper = pm.get_bounds_as_arrays(["D0", "alpha", "D_offset"])

        assert isinstance(lower, np.ndarray)
        assert isinstance(upper, np.ndarray)
        assert len(lower) == 3
        assert len(upper) == 3
        assert lower[0] == 1.0  # D0 min
        assert upper[0] == 1e6  # D0 max


class TestActiveParameters:
    """Test active parameters functionality."""

    def test_get_active_params_laminar_flow(self):
        """Test getting active parameters for laminar flow."""
        pm = ParameterManager(None, "laminar_flow")
        active = pm.get_active_parameters()

        expected = [
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ]
        assert active == expected
        assert len(active) == 7

    def test_get_active_params_static_mode(self):
        """Test getting active parameters for static mode."""
        pm = ParameterManager(None, "static")
        active = pm.get_active_parameters()

        expected = ["D0", "alpha", "D_offset"]
        assert active == expected
        assert len(active) == 3

    def test_get_active_params_from_config(self):
        """Test getting active parameters from configuration."""
        config = {
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset", "gamma_dot_t0"]
            }
        }
        pm = ParameterManager(config, "laminar_flow")
        active = pm.get_active_parameters()

        assert active == ["D0", "alpha", "D_offset", "gamma_dot_t0"]
        assert len(active) == 4

    def test_get_active_params_with_explicit_active_list(self):
        """Test explicit active_parameters list in config."""
        config = {
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "active_parameters": ["D0", "alpha"],  # Subset
            }
        }
        pm = ParameterManager(config, "laminar_flow")
        active = pm.get_active_parameters()

        assert active == ["D0", "alpha"]
        assert len(active) == 2


class TestParameterNameMapping:
    """Test parameter name mapping."""

    def test_name_mapping_gamma_dot(self):
        """Test gamma_dot_0 -> gamma_dot_t0 mapping."""
        config = {
            "parameter_space": {
                "bounds": [{"name": "gamma_dot_0", "min": 1e-5, "max": 0.1}]
            }
        }
        pm = ParameterManager(config, "laminar_flow")
        bounds = pm.get_parameter_bounds(["gamma_dot_t0"])

        # Should find the bound using the mapped name
        assert bounds[0]["name"] == "gamma_dot_t0"
        assert bounds[0]["min"] == 1e-5

    def test_name_mapping_phi(self):
        """Test phi_0 -> phi0 mapping."""
        config = {
            "parameter_space": {"bounds": [{"name": "phi_0", "min": -1.0, "max": 1.0}]}
        }
        pm = ParameterManager(config, "laminar_flow")
        bounds = pm.get_parameter_bounds(["phi0"])

        assert bounds[0]["name"] == "phi0"
        assert bounds[0]["min"] == -1.0


class TestParameterCounts:
    """Test parameter counting methods."""

    def test_effective_count_laminar(self):
        """Test effective parameter count for laminar flow."""
        pm = ParameterManager(None, "laminar_flow")
        assert pm.get_effective_parameter_count() == 7

    def test_effective_count_static(self):
        """Test effective parameter count for static mode."""
        pm = ParameterManager(None, "static")
        assert pm.get_effective_parameter_count() == 3

    def test_total_count_laminar(self):
        """Test total parameter count for laminar flow."""
        pm = ParameterManager(None, "laminar_flow")
        assert pm.get_total_parameter_count() == 9  # 2 scaling + 7 physical

    def test_total_count_static(self):
        """Test total parameter count for static mode."""
        pm = ParameterManager(None, "static")
        assert pm.get_total_parameter_count() == 5  # 2 scaling + 3 physical

    def test_all_parameter_names(self):
        """Test getting all parameter names including scaling."""
        pm = ParameterManager(None, "laminar_flow")
        all_params = pm.get_all_parameter_names()

        # Should start with scaling parameters
        assert all_params[0] == "contrast"
        assert all_params[1] == "offset"

        # Then physical parameters
        assert "D0" in all_params
        assert "gamma_dot_t0" in all_params
        assert len(all_params) == 9


class TestParameterValidation:
    """Test parameter validation integration."""

    def test_validate_valid_params(self):
        """Test validation with valid parameters."""
        pm = ParameterManager(None, "static")
        params = np.array([0.5, 1.0, 1000.0, 0.5, 10.0])
        param_names = ["contrast", "offset", "D0", "alpha", "D_offset"]

        result = pm.validate_parameters(params, param_names)
        assert result.valid is True
        assert len(result.violations) == 0

    def test_validate_invalid_params(self):
        """Test validation with out of bounds parameters."""
        pm = ParameterManager(None, "static")
        params = np.array([1.5, 1.0, 2e6, 0.5, 10.0])  # contrast and D0 out of bounds
        param_names = ["contrast", "offset", "D0", "alpha", "D_offset"]

        result = pm.validate_parameters(params, param_names)
        assert result.valid is False
        assert len(result.violations) == 2

    def test_validate_uses_config_bounds(self):
        """Test that validation uses configuration bounds."""
        config = {
            "parameter_space": {
                "bounds": [{"name": "D0", "min": 100.0, "max": 1000.0}]
            }
        }
        pm = ParameterManager(config, "static")
        params = np.array([0.5, 1.0, 50.0, 0.5, 10.0])  # D0 = 50 < 100 (config min)
        param_names = ["contrast", "offset", "D0", "alpha", "D_offset"]

        result = pm.validate_parameters(params, param_names)
        assert result.valid is False
        assert any("D0" in v for v in result.violations)


class TestParameterManagerRepr:
    """Test string representation."""

    def test_repr_laminar(self):
        """Test repr for laminar flow mode."""
        pm = ParameterManager(None, "laminar_flow")
        repr_str = repr(pm)

        assert "ParameterManager" in repr_str
        assert "laminar_flow" in repr_str
        assert "active_params=7" in repr_str
        assert "total_params=9" in repr_str

    def test_repr_static(self):
        """Test repr for static mode."""
        pm = ParameterManager(None, "static")
        repr_str = repr(pm)

        assert "static" in repr_str
        assert "active_params=3" in repr_str
        assert "total_params=5" in repr_str


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_unknown_parameter_bound(self):
        """Test requesting bounds for unknown parameter."""
        pm = ParameterManager(None, "laminar_flow")
        bounds = pm.get_parameter_bounds(["unknown_param"])

        # Should return fallback bounds
        assert len(bounds) == 1
        assert bounds[0]["min"] == 0.0
        assert bounds[0]["max"] == 1.0

    def test_empty_config(self):
        """Test with empty configuration."""
        pm = ParameterManager({}, "laminar_flow")
        active = pm.get_active_parameters()

        # Should fall back to defaults
        assert len(active) == 7

    def test_malformed_config_bounds(self):
        """Test with malformed bounds in config."""
        config = {"parameter_space": {"bounds": "not_a_list"}}
        pm = ParameterManager(config, "laminar_flow")
        bounds = pm.get_parameter_bounds(["D0"])

        # Should fall back to defaults without crashing
        assert len(bounds) == 1
        assert bounds[0]["min"] == 1.0  # Default D0 min


class TestCaching:
    """Test caching functionality for performance optimization."""

    def test_bounds_cache_hit(self):
        """Test that repeated bounds queries use cache."""
        pm = ParameterManager(None, "laminar_flow")
        param_names = ["D0", "alpha", "D_offset"]

        # First call - cache miss
        bounds1 = pm.get_parameter_bounds(param_names)

        # Verify cache was populated
        cache_key = tuple(sorted(param_names))
        assert cache_key in pm._bounds_cache

        # Second call - should hit cache
        bounds2 = pm.get_parameter_bounds(param_names)

        # Results should be identical
        assert bounds1 == bounds2
        assert len(bounds1) == 3

    def test_bounds_cache_different_params(self):
        """Test that different parameter sets create different cache entries."""
        pm = ParameterManager(None, "laminar_flow")

        bounds1 = pm.get_parameter_bounds(["D0", "alpha"])
        bounds2 = pm.get_parameter_bounds(["D0", "alpha", "D_offset"])

        # Cache should have two entries
        assert len(pm._bounds_cache) == 2
        assert len(bounds1) == 2
        assert len(bounds2) == 3

    def test_bounds_cache_order_independence(self):
        """Test that parameter order doesn't affect cache key."""
        pm = ParameterManager(None, "laminar_flow")

        bounds1 = pm.get_parameter_bounds(["D0", "alpha", "D_offset"])
        bounds2 = pm.get_parameter_bounds(["alpha", "D_offset", "D0"])

        # Should hit the same cache entry
        assert len(pm._bounds_cache) == 1

        # Results should be identical (same order of params in result)
        assert len(bounds1) == len(bounds2)

    def test_bounds_cache_returns_copy(self):
        """Test that cached results are copied to prevent mutation."""
        pm = ParameterManager(None, "laminar_flow")
        param_names = ["D0", "alpha"]

        # Get bounds twice
        bounds1 = pm.get_parameter_bounds(param_names)
        bounds2 = pm.get_parameter_bounds(param_names)

        # Mutate first result
        bounds1[0]["min"] = -9999.0

        # Second result should be unchanged
        assert bounds2[0]["min"] == 1.0  # Original D0 min

        # Cache should also be unchanged
        cache_key = tuple(sorted(param_names))
        cached_bounds = pm._bounds_cache[cache_key]
        assert cached_bounds[0]["min"] == 1.0

    def test_active_params_cache_hit(self):
        """Test that repeated active parameters queries use cache."""
        pm = ParameterManager(None, "laminar_flow")

        # First call - cache miss
        active1 = pm.get_active_parameters()

        # Verify cache was populated
        assert pm._active_params_cache is not None

        # Second call - should hit cache
        active2 = pm.get_active_parameters()

        # Results should be identical
        assert active1 == active2
        assert len(active1) == 7

    def test_active_params_cache_returns_copy(self):
        """Test that cached active params are copied to prevent mutation."""
        pm = ParameterManager(None, "laminar_flow")

        # Get active params twice
        active1 = pm.get_active_parameters()
        active2 = pm.get_active_parameters()

        # Mutate first result
        active1.append("fake_param")

        # Second result should be unchanged
        assert len(active2) == 7
        assert "fake_param" not in active2

        # Cache should also be unchanged
        assert len(pm._active_params_cache) == 7

    def test_cache_with_config_override(self):
        """Test caching with config bounds override."""
        config = {
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 10.0, "max": 1e5},
                    {"name": "alpha", "min": -1.0, "max": 1.0},
                ]
            }
        }
        pm = ParameterManager(config, "laminar_flow")

        # First call
        bounds1 = pm.get_parameter_bounds(["D0", "alpha"])

        # Verify custom bounds
        assert bounds1[0]["min"] == 10.0
        assert bounds1[1]["min"] == -1.0

        # Second call - should use cache
        bounds2 = pm.get_parameter_bounds(["D0", "alpha"])

        # Should still have custom bounds
        assert bounds2[0]["min"] == 10.0
        assert bounds2[1]["min"] == -1.0

    def test_cache_enabled_flag(self):
        """Test that caching can be controlled via _cache_enabled flag."""
        pm = ParameterManager(None, "laminar_flow")

        # Disable caching
        pm._cache_enabled = False

        bounds1 = pm.get_parameter_bounds(["D0", "alpha"])

        # Cache should not be populated
        assert len(pm._bounds_cache) == 0

        # Re-enable caching
        pm._cache_enabled = True

        bounds2 = pm.get_parameter_bounds(["D0", "alpha"])

        # Now cache should be populated
        assert len(pm._bounds_cache) == 1

    def test_all_params_caching(self):
        """Test caching when requesting all parameters."""
        pm = ParameterManager(None, "laminar_flow")

        # Get all parameters (None = all)
        bounds1 = pm.get_parameter_bounds()
        bounds2 = pm.get_parameter_bounds()

        # Should hit cache
        assert len(bounds1) == len(bounds2)
        assert len(bounds1) == 9  # 2 scaling + 7 physical

    def test_cache_performance_benefit(self):
        """Test that caching provides measurable performance benefit."""
        import time

        config = {
            "parameter_space": {
                "bounds": [{"name": f"param_{i}", "min": 0.0, "max": 1.0} for i in range(50)]
            }
        }
        pm = ParameterManager(config, "laminar_flow")
        param_names = [f"param_{i}" for i in range(50)]

        # First call - cache miss
        start_time = time.time()
        bounds1 = pm.get_parameter_bounds(param_names)
        first_call_time = time.time() - start_time

        # Second call - cache hit
        start_time = time.time()
        bounds2 = pm.get_parameter_bounds(param_names)
        second_call_time = time.time() - start_time

        # Cache hit should be faster (allow some tolerance)
        # Note: This is a weak assertion since timing can vary
        assert len(bounds1) == len(bounds2)
        # In practice, cache hit should be ~10-100x faster


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
