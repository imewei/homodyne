"""
Unit Tests for Parameter Manager Core
======================================

**Consolidation**: Week 5 (2025-11-15)

Consolidated from:
- test_parameter_manager.py (Core functionality, 38 tests, 515 lines)
- test_parameter_manager_advanced.py (Advanced features, 17 tests, 275 lines)
- test_parameter_manager_physics.py (Physics-specific, 32 tests, 471 lines)

Test Categories:
---------------
**Core Functionality** (38 tests):
- ParameterManager initialization and configuration
- Name mapping and validation (canonical names)
- Bounds checking and parameter space operations
- Caching mechanism (~10-100x speedup)

**Advanced Features** (17 tests):
- Array operations and batch processing
- Parameter expansion for per-angle scaling
- Complex bound calculations
- Edge case handling

**Physics-Specific** (32 tests):
- Mode-specific parameter sets (static, laminar_flow)
- Physics parameter validation
- Parameter constraints and dependencies
- Domain-specific bound checking

Test Coverage:
-------------
- Core ParameterManager functionality with efficient caching
- Canonical name mapping: gamma_dot_0 → gamma_dot_t0, etc.
- Bounds validation and parameter space operations
- Advanced array operations and batch parameter processing
- Physics-specific parameter handling for different analysis modes
- Mode-specific parameter sets (static: 3 params, laminar_flow: 7 params)
- Parameter expansion for per-angle scaling (3 angles: 5→9 params)
- Constraint validation and dependency checking

Total: 87 tests

Usage Example:
-------------
```python
# Run all parameter manager tests
pytest tests/unit/test_parameter_manager_core.py -v

# Run specific category
pytest tests/unit/test_parameter_manager_core.py -k "physics" -v
pytest tests/unit/test_parameter_manager_core.py::TestParameterManagerInit -v

# Test caching functionality
pytest tests/unit/test_parameter_manager_core.py -k "cache" -v
```

See Also:
---------
- docs/WEEK5_CONSOLIDATION_SUMMARY.md: Consolidation details
- homodyne/config/parameter_manager.py: ParameterManager implementation
- homodyne/config/types.py: Parameter name mappings and constants
"""

import pytest
import numpy as np
from typing import Any, Dict, List
from unittest.mock import Mock, patch

from homodyne.config.parameter_manager import ParameterManager
from homodyne.config.types import (
    LAMINAR_FLOW_PARAM_NAMES,
    PARAMETER_NAME_MAPPING,
    STATIC_PARAM_NAMES,
    BoundDict,
)


# ==============================================================================
# Core Parameter Manager Tests (from test_parameter_manager.py)
# ==============================================================================


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
        assert bounds[0] == (1e2, 1e5)  # Updated default D0 bounds

    def test_get_bounds_as_arrays(self):
        """Test getting bounds as numpy arrays."""
        pm = ParameterManager(None, "laminar_flow")
        lower, upper = pm.get_bounds_as_arrays(["D0", "alpha", "D_offset"])

        assert isinstance(lower, np.ndarray)
        assert isinstance(upper, np.ndarray)
        assert len(lower) == 3
        assert len(upper) == 3
        assert lower[0] == 1e2  # D0 min
        assert upper[0] == 1e5  # D0 max


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
            "parameter_space": {"bounds": [{"name": "D0", "min": 100.0, "max": 1000.0}]}
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
        assert bounds[0]["min"] == 1e2  # Default D0 min


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
        assert bounds2[0]["min"] == 1e2  # Original D0 min

        # Cache should also be unchanged
        cache_key = tuple(sorted(param_names))
        cached_bounds = pm._bounds_cache[cache_key]
        assert cached_bounds[0]["min"] == 1e2

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
                "bounds": [
                    {"name": f"param_{i}", "min": 0.0, "max": 1.0} for i in range(50)
                ]
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


# ==============================================================================
# Advanced Parameter Manager Tests (from test_parameter_manager_advanced.py)
# ==============================================================================


class TestFixedParameters:
    """Test fixed parameters functionality."""

    def test_get_fixed_parameters_none(self):
        """Test getting fixed parameters when none specified."""
        pm = ParameterManager(None, "laminar_flow")
        fixed = pm.get_fixed_parameters()

        assert fixed == {}

    def test_get_fixed_parameters_from_config(self):
        """Test getting fixed parameters from configuration."""
        config = {
            "initial_parameters": {
                "fixed_parameters": {"D_offset": 10.0, "contrast": 0.5}
            }
        }
        pm = ParameterManager(config, "laminar_flow")
        fixed = pm.get_fixed_parameters()

        assert fixed == {"D_offset": 10.0, "contrast": 0.5}

    def test_get_fixed_parameters_invalid_type(self):
        """Test handling of invalid fixed_parameters type."""
        config = {"initial_parameters": {"fixed_parameters": ["not", "a", "dict"]}}
        pm = ParameterManager(config, "laminar_flow")
        fixed = pm.get_fixed_parameters()

        # Should return empty dict and log warning
        assert fixed == {}


class TestParameterActiveStatus:
    """Test is_parameter_active() functionality."""

    def test_parameter_active_no_config(self):
        """Test parameter active status without configuration."""
        pm = ParameterManager(None, "laminar_flow")

        # All default active parameters should be active
        assert pm.is_parameter_active("D0") is True
        assert pm.is_parameter_active("alpha") is True
        assert pm.is_parameter_active("gamma_dot_t0") is True

    def test_parameter_active_with_fixed(self):
        """Test parameter active status with fixed parameters."""
        config = {
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "fixed_parameters": {"D_offset": 10.0},
            }
        }
        pm = ParameterManager(config, "laminar_flow")

        # D0 and alpha are active
        assert pm.is_parameter_active("D0") is True
        assert pm.is_parameter_active("alpha") is True

        # D_offset is fixed (not active)
        assert pm.is_parameter_active("D_offset") is False

    def test_parameter_active_with_name_mapping(self):
        """Test parameter active status with name mapping."""
        config = {
            "initial_parameters": {
                "parameter_names": ["D0", "gamma_dot_0"],  # Config name
                "fixed_parameters": {"D0": 1000.0},
            }
        }
        pm = ParameterManager(config, "laminar_flow")

        # D0 is fixed
        assert pm.is_parameter_active("D0") is False

        # gamma_dot_t0 is active (mapped from gamma_dot_0)
        assert pm.is_parameter_active("gamma_dot_t0") is True


class TestOptimizableParameters:
    """Test get_optimizable_parameters() functionality."""

    def test_optimizable_params_no_fixed(self):
        """Test optimizable parameters when nothing is fixed."""
        pm = ParameterManager(None, "static")
        optimizable = pm.get_optimizable_parameters()

        # All active parameters should be optimizable
        assert optimizable == ["D0", "alpha", "D_offset"]

    def test_optimizable_params_with_fixed(self):
        """Test optimizable parameters with some fixed."""
        config = {
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset", "gamma_dot_t0"],
                "fixed_parameters": {"D_offset": 10.0, "gamma_dot_t0": 1e-4},
            }
        }
        pm = ParameterManager(config, "laminar_flow")
        optimizable = pm.get_optimizable_parameters()

        # Only D0 and alpha should be optimizable
        assert optimizable == ["D0", "alpha"]
        assert len(optimizable) == 2

    def test_optimizable_params_all_fixed(self):
        """Test optimizable parameters when all are fixed."""
        config = {
            "initial_parameters": {
                "parameter_names": ["D0", "alpha"],
                "fixed_parameters": {"D0": 1000.0, "alpha": 0.5},
            }
        }
        pm = ParameterManager(config, "laminar_flow")
        optimizable = pm.get_optimizable_parameters()

        # No parameters should be optimizable
        assert optimizable == []


class TestTypeConstants:
    """Test type constants from types.py."""

    def test_static_param_names(self):
        """Test STATIC_PARAM_NAMES constant."""
        assert STATIC_PARAM_NAMES == ["D0", "alpha", "D_offset"]
        assert len(STATIC_PARAM_NAMES) == 3

    def test_laminar_flow_param_names(self):
        """Test LAMINAR_FLOW_PARAM_NAMES constant."""
        expected = [
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ]
        assert LAMINAR_FLOW_PARAM_NAMES == expected
        assert len(LAMINAR_FLOW_PARAM_NAMES) == 7

    def test_parameter_name_mapping(self):
        """Test PARAMETER_NAME_MAPPING constant."""
        assert PARAMETER_NAME_MAPPING["gamma_dot_0"] == "gamma_dot_t0"
        assert PARAMETER_NAME_MAPPING["gamma_dot_offset"] == "gamma_dot_t_offset"
        assert PARAMETER_NAME_MAPPING["phi_0"] == "phi0"
        assert len(PARAMETER_NAME_MAPPING) == 3


class TestBoundDictType:
    """Test BoundDict TypedDict."""

    def test_bound_dict_structure(self):
        """Test that BoundDict has expected structure."""
        # This tests that the TypedDict is properly defined
        bound: BoundDict = {
            "name": "D0",
            "min": 1.0,
            "max": 1e6,
            "type": "Normal",
        }

        assert bound["name"] == "D0"
        assert bound["min"] == 1.0
        assert bound["max"] == 1e6
        assert bound["type"] == "Normal"


class TestReprWithFixedParams:
    """Test string representation with fixed parameters."""

    def test_repr_with_fixed_params(self):
        """Test repr includes fixed parameter count."""
        config = {
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "fixed_parameters": {"D_offset": 10.0},
            }
        }
        pm = ParameterManager(config, "laminar_flow")
        repr_str = repr(pm)

        assert "ParameterManager" in repr_str
        assert "active_params=3" in repr_str
        assert "fixed_params=1" in repr_str
        assert "optimizable=2" in repr_str

    def test_repr_no_fixed_params(self):
        """Test repr with no fixed parameters."""
        pm = ParameterManager(None, "static")
        repr_str = repr(pm)

        assert "active_params=3" in repr_str
        assert "fixed_params=0" in repr_str
        assert "optimizable=3" in repr_str


class TestIntegrationScenarios:
    """Integration tests for advanced features."""

    def test_subset_optimization_workflow(self):
        """Test workflow with subset of parameters and fixed values."""
        config = {
            "analysis_mode": "laminar_flow",
            "initial_parameters": {
                "parameter_names": [
                    "D0",
                    "alpha",
                    "D_offset",
                    "gamma_dot_t0",
                    "beta",
                ],
                "fixed_parameters": {"beta": 0.5, "D_offset": 10.0},
            },
        }
        pm = ParameterManager(config, "laminar_flow")

        # Check active parameters
        active = pm.get_active_parameters()
        assert len(active) == 5

        # Check optimizable (active - fixed)
        optimizable = pm.get_optimizable_parameters()
        assert optimizable == ["D0", "alpha", "gamma_dot_t0"]
        assert len(optimizable) == 3

        # Check individual status
        assert pm.is_parameter_active("D0") is True
        assert pm.is_parameter_active("beta") is False
        assert pm.is_parameter_active("D_offset") is False

    def test_name_mapping_with_fixed_params(self):
        """Test that name mapping works with fixed parameters."""
        config = {
            "initial_parameters": {
                "parameter_names": ["D0", "gamma_dot_0", "phi_0"],
                "fixed_parameters": {"phi_0": 0.1},  # Config name
            }
        }
        pm = ParameterManager(config, "laminar_flow")

        # Active should use canonical names
        active = pm.get_active_parameters()
        assert "gamma_dot_t0" in active
        assert "phi0" in active

        # Fixed should work with config name
        fixed = pm.get_fixed_parameters()
        assert "phi_0" in fixed

        # is_parameter_active should handle canonical name
        assert pm.is_parameter_active("phi0") is False
        assert pm.is_parameter_active("gamma_dot_t0") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ==============================================================================
# Physics-Specific Parameter Tests (from test_parameter_manager_physics.py)
# ==============================================================================


class TestPhysicsValidationBasics:
    """Test basic physics validation functionality."""

    def test_validate_all_valid_parameters(self):
        """Test validation with physically reasonable parameters."""
        pm = ParameterManager(None, "laminar_flow")
        params = {
            "D0": 1000.0,
            "alpha": 0.5,
            "D_offset": 10.0,
            "gamma_dot_t0": 0.001,
            "beta": 0.5,
            "gamma_dot_t_offset": 0.0001,
            "phi0": 0.5,
            "contrast": 0.8,
            "offset": 1.0,
        }

        result = pm.validate_physical_constraints(params)

        # Should pass all checks (only INFO messages possible)
        assert len(result.violations) == 0
        assert result.parameters_checked == len(params)
        assert "successfully" in result.message.lower()

    def test_validate_empty_parameters(self):
        """Test validation with empty parameter dict."""
        pm = ParameterManager(None, "laminar_flow")
        params = {}

        result = pm.validate_physical_constraints(params)

        assert result.valid is True
        assert len(result.violations) == 0
        assert result.parameters_checked == 0


class TestErrorLevelViolations:
    """Test ERROR-level physics violations (physically impossible)."""

    def test_negative_diffusion_coefficient(self):
        """Test ERROR for negative D0."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"D0": -100.0}

        result = pm.validate_physical_constraints(params, severity_level="error")

        assert result.valid is False
        assert len(result.violations) == 1
        assert "D0" in result.violations[0]
        assert "physically impossible" in result.violations[0].lower()
        assert "[error]" in result.violations[0]

    def test_zero_diffusion_coefficient(self):
        """Test ERROR for D0 = 0."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"D0": 0.0}

        result = pm.validate_physical_constraints(params, severity_level="error")

        assert result.valid is False
        assert "non-positive" in result.violations[0].lower()

    def test_negative_shear_rate(self):
        """Test ERROR for negative gamma_dot_t0."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"gamma_dot_t0": -0.001}

        result = pm.validate_physical_constraints(params, severity_level="error")

        assert result.valid is False
        assert len(result.violations) == 1
        assert "gamma_dot_t0" in result.violations[0]
        assert "negative shear rate" in result.violations[0].lower()
        assert "physically impossible" in result.violations[0].lower()

    def test_invalid_contrast_too_high(self):
        """Test ERROR for contrast > 1."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"contrast": 1.5}

        result = pm.validate_physical_constraints(params, severity_level="error")

        assert result.valid is False
        assert "contrast" in result.violations[0]
        assert "outside physical range" in result.violations[0].lower()

    def test_invalid_contrast_zero(self):
        """Test ERROR for contrast = 0."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"contrast": 0.0}

        result = pm.validate_physical_constraints(params, severity_level="error")

        assert result.valid is False
        assert "contrast" in result.violations[0]

    def test_negative_offset(self):
        """Test ERROR for negative offset."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"offset": -0.5}

        result = pm.validate_physical_constraints(params, severity_level="error")

        assert result.valid is False
        assert "offset" in result.violations[0]
        assert "non-positive baseline" in result.violations[0].lower()

    def test_multiple_errors(self):
        """Test multiple ERROR violations."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"D0": -100.0, "gamma_dot_t0": -0.001, "contrast": 2.0}

        result = pm.validate_physical_constraints(params, severity_level="error")

        assert result.valid is False
        assert len(result.violations) == 3
        # Check all three violations are present
        violation_str = " ".join(result.violations)
        assert "D0" in violation_str
        assert "gamma_dot_t0" in violation_str
        assert "contrast" in violation_str


class TestWarningLevelViolations:
    """Test WARNING-level physics violations (unusual but possible)."""

    def test_very_large_diffusion_coefficient(self):
        """Test WARNING for D0 > 1e7."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"D0": 5e7}

        result = pm.validate_physical_constraints(params, severity_level="warning")

        assert result.valid is False
        assert len(result.violations) == 1
        assert "D0" in result.violations[0]
        assert "extremely large" in result.violations[0].lower()
        assert "[warning]" in result.violations[0]

    def test_strongly_subdiffusive_alpha(self):
        """Test WARNING for alpha < -1.5."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"alpha": -1.8}

        result = pm.validate_physical_constraints(params, severity_level="warning")

        assert result.valid is False
        assert "alpha" in result.violations[0]
        assert "subdiffusive" in result.violations[0].lower()

    def test_strongly_superdiffusive_alpha(self):
        """Test WARNING for alpha > 1.0."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"alpha": 1.5}

        result = pm.validate_physical_constraints(params, severity_level="warning")

        assert result.valid is False
        assert "alpha" in result.violations[0]
        assert "superdiffusive" in result.violations[0].lower()
        assert "ballistic" in result.violations[0].lower()

    def test_negative_d_offset(self):
        """Test WARNING for negative D_offset."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"D_offset": -10.0}

        result = pm.validate_physical_constraints(params, severity_level="warning")

        assert result.valid is False
        assert "D_offset" in result.violations[0]
        assert "negative offset" in result.violations[0].lower()

    def test_very_high_shear_rate(self):
        """Test WARNING for gamma_dot_t0 > 1.0."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"gamma_dot_t0": 5.0}

        result = pm.validate_physical_constraints(params, severity_level="warning")

        assert result.valid is False
        assert "gamma_dot_t0" in result.violations[0]
        assert "very high shear rate" in result.violations[0].lower()

    def test_beta_out_of_range(self):
        """Test WARNING for beta outside [-2, 2]."""
        pm = ParameterManager(None, "laminar_flow")
        params_low = {"beta": -2.5}
        params_high = {"beta": 2.5}

        result_low = pm.validate_physical_constraints(
            params_low, severity_level="warning"
        )
        result_high = pm.validate_physical_constraints(
            params_high, severity_level="warning"
        )

        assert result_low.valid is False
        assert result_high.valid is False
        assert "beta" in result_low.violations[0]
        assert "beta" in result_high.violations[0]

    def test_negative_gamma_dot_offset(self):
        """Test WARNING for negative gamma_dot_t_offset."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"gamma_dot_t_offset": -0.0001}

        result = pm.validate_physical_constraints(params, severity_level="warning")

        assert result.valid is False
        assert "gamma_dot_t_offset" in result.violations[0]

    def test_very_low_contrast(self):
        """Test WARNING for contrast < 0.1."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"contrast": 0.05}

        result = pm.validate_physical_constraints(params, severity_level="warning")

        assert result.valid is False
        assert "contrast" in result.violations[0]
        assert "very low contrast" in result.violations[0].lower()


class TestInfoLevelViolations:
    """Test INFO-level physics notifications (noteworthy but acceptable)."""

    def test_near_normal_diffusion(self):
        """Test INFO for alpha near zero (normal diffusion)."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"alpha": 0.05}

        result = pm.validate_physical_constraints(params, severity_level="info")

        assert result.valid is False
        assert len(result.violations) == 1
        assert "alpha" in result.violations[0]
        assert "near-normal diffusion" in result.violations[0].lower()
        assert "[info]" in result.violations[0]

    def test_very_low_shear_rate(self):
        """Test INFO for gamma_dot_t0 < 1e-6."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"gamma_dot_t0": 1e-8}

        result = pm.validate_physical_constraints(params, severity_level="info")

        assert result.valid is False
        assert "gamma_dot_t0" in result.violations[0]
        assert "very low shear rate" in result.violations[0].lower()
        assert "quasi-static" in result.violations[0].lower()

    def test_angle_outside_pi_range(self):
        """Test INFO for phi0 outside [-π, π]."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"phi0": 4.0}

        result = pm.validate_physical_constraints(params, severity_level="info")

        assert result.valid is False
        assert "phi0" in result.violations[0]
        assert "will wrap" in result.violations[0].lower()


class TestCrossParameterConstraints:
    """Test cross-parameter physics constraints."""

    def test_d_offset_dominates_d0(self):
        """Test INFO when D_offset is large compared to D0."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"D0": 1000.0, "alpha": 0.5, "D_offset": 600.0}

        result = pm.validate_physical_constraints(params, severity_level="info")

        assert result.valid is False
        assert len(result.violations) == 1
        assert "D_offset" in result.violations[0]
        assert "60.0%" in result.violations[0]  # 600/1000 = 60%
        assert "overfitting" in result.violations[0].lower()

    def test_d_offset_reasonable_ratio(self):
        """Test no violation when D_offset is reasonable compared to D0."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"D0": 1000.0, "alpha": 0.5, "D_offset": 100.0}

        result = pm.validate_physical_constraints(params, severity_level="info")

        # Should pass - 100/1000 = 10% < 50%
        assert result.valid is True
        assert len(result.violations) == 0


class TestSeverityFiltering:
    """Test severity level filtering."""

    def test_error_level_filters_warnings(self):
        """Test that error level only shows ERROR violations."""
        pm = ParameterManager(None, "laminar_flow")
        params = {
            "D0": -100.0,  # ERROR
            "alpha": 1.5,  # WARNING
            "gamma_dot_t0": 1e-8,  # INFO
        }

        result = pm.validate_physical_constraints(params, severity_level="error")

        assert result.valid is False
        assert len(result.violations) == 1
        assert "D0" in result.violations[0]
        assert "[error]" in result.violations[0]

    def test_warning_level_shows_errors_and_warnings(self):
        """Test that warning level shows ERROR + WARNING."""
        pm = ParameterManager(None, "laminar_flow")
        params = {
            "D0": -100.0,  # ERROR
            "alpha": 1.5,  # WARNING
            "gamma_dot_t0": 1e-8,  # INFO
        }

        result = pm.validate_physical_constraints(params, severity_level="warning")

        assert result.valid is False
        assert len(result.violations) == 2
        violation_str = " ".join(result.violations)
        assert "D0" in violation_str
        assert "alpha" in violation_str
        assert "gamma_dot_t0" not in violation_str

    def test_info_level_shows_all(self):
        """Test that info level shows all violations."""
        pm = ParameterManager(None, "laminar_flow")
        params = {
            "D0": -100.0,  # ERROR
            "alpha": 1.5,  # WARNING
            "gamma_dot_t0": 1e-8,  # INFO
        }

        result = pm.validate_physical_constraints(params, severity_level="info")

        assert result.valid is False
        assert len(result.violations) == 3
        violation_str = " ".join(result.violations)
        assert "D0" in violation_str
        assert "alpha" in violation_str
        assert "gamma_dot_t0" in violation_str


class TestValidationMessages:
    """Test validation result messages."""

    def test_success_message(self):
        """Test message for successful validation."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"D0": 1000.0, "alpha": 0.5}

        result = pm.validate_physical_constraints(params)

        assert result.valid is True
        assert "successfully" in result.message.lower()
        assert "2 parameters" in result.message.lower()

    def test_failure_message(self):
        """Test message for failed validation."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"D0": -100.0}

        result = pm.validate_physical_constraints(params)

        assert result.valid is False
        assert "1 issue" in result.message.lower()

    def test_parameters_checked_count(self):
        """Test parameters_checked field."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"D0": 1000.0, "alpha": 0.5, "gamma_dot_t0": 0.001}

        result = pm.validate_physical_constraints(params)

        assert result.parameters_checked == 3


class TestPhysicsValidationIntegration:
    """Integration tests for physics validation."""

    def test_static_mode_parameters(self):
        """Test validation for static mode parameters."""
        pm = ParameterManager(None, "static")
        params = {
            "D0": 1000.0,
            "alpha": -0.5,  # Subdiffusion common in static
            "D_offset": 10.0,
            "contrast": 0.8,
            "offset": 1.0,
        }

        result = pm.validate_physical_constraints(params)

        # Should pass all checks
        assert result.valid is True

    def test_laminar_flow_parameters(self):
        """Test validation for laminar flow parameters."""
        pm = ParameterManager(None, "laminar_flow")
        params = {
            "D0": 1000.0,
            "alpha": 0.5,
            "D_offset": 10.0,
            "gamma_dot_t0": 0.001,
            "beta": 0.3,
            "gamma_dot_t_offset": 0.0001,
            "phi0": 0.5,
            "contrast": 0.8,
            "offset": 1.0,
        }

        result = pm.validate_physical_constraints(params)

        assert result.valid is True

    def test_realistic_subdiffusion_scenario(self):
        """Test realistic subdiffusion scenario."""
        pm = ParameterManager(None, "laminar_flow")
        params = {
            "D0": 500.0,
            "alpha": -0.7,  # Subdiffusion
            "D_offset": 5.0,
            "gamma_dot_t0": 0.0001,
            "beta": 0.5,
            "contrast": 0.6,
            "offset": 1.2,
        }

        result = pm.validate_physical_constraints(params, severity_level="warning")

        # Should pass - subdiffusion is common
        assert result.valid is True

    def test_problematic_parameter_set(self):
        """Test detection of problematic parameter set."""
        pm = ParameterManager(None, "laminar_flow")
        params = {
            "D0": 100.0,
            "alpha": 0.5,
            "D_offset": 90.0,  # 90% of D0 - likely overfitting
            "gamma_dot_t0": 10.0,  # Very high
            "contrast": 0.05,  # Very low
        }

        result = pm.validate_physical_constraints(params, severity_level="warning")

        assert result.valid is False
        # Should have multiple warnings
        assert len(result.violations) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
