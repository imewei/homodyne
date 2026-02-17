"""Unit tests for configuration parameter management.

Tests for ParameterManager, parameter bounds, and validation.
"""

from homodyne.config.parameter_manager import ParameterManager
from homodyne.config.types import (
    LAMINAR_FLOW_PARAM_NAMES,
    SCALING_PARAM_NAMES,
    STATIC_PARAM_NAMES,
)


class TestParameterManagerInit:
    """Tests for ParameterManager initialization."""

    def test_init_with_no_config(self):
        """Test initialization without config dict."""
        pm = ParameterManager()
        assert pm.config_dict == {}
        assert pm.analysis_mode == "laminar_flow"

    def test_init_with_static_mode(self):
        """Test initialization with static mode."""
        pm = ParameterManager(analysis_mode="static")
        assert pm.analysis_mode == "static"

    def test_init_with_laminar_flow_mode(self):
        """Test initialization with laminar_flow mode."""
        pm = ParameterManager(analysis_mode="laminar_flow")
        assert pm.analysis_mode == "laminar_flow"

    def test_init_with_config_dict(self):
        """Test initialization with config dictionary."""
        config = {"analysis_mode": "static"}
        pm = ParameterManager(config_dict=config)
        assert pm.config_dict == config

    def test_cache_enabled_by_default(self):
        """Test that caching is enabled by default."""
        pm = ParameterManager()
        assert pm._cache_enabled is True


class TestParameterBounds:
    """Tests for parameter bounds management."""

    def test_default_bounds_for_contrast(self):
        """Test default bounds for contrast parameter."""
        pm = ParameterManager()
        assert "contrast" in pm._default_bounds
        bounds = pm._default_bounds["contrast"]
        assert bounds["min"] == 0.0
        assert bounds["max"] == 1.0

    def test_default_bounds_for_offset(self):
        """Test default bounds for offset parameter."""
        pm = ParameterManager()
        assert "offset" in pm._default_bounds
        bounds = pm._default_bounds["offset"]
        assert bounds["min"] == 0.5
        assert bounds["max"] == 1.5

    def test_default_bounds_for_d0(self):
        """Test default bounds for D0 parameter."""
        pm = ParameterManager()
        assert "D0" in pm._default_bounds
        bounds = pm._default_bounds["D0"]
        assert bounds["min"] == 1e2
        assert bounds["max"] == 1e5

    def test_default_bounds_for_alpha(self):
        """Test default bounds for alpha parameter."""
        pm = ParameterManager()
        assert "alpha" in pm._default_bounds
        bounds = pm._default_bounds["alpha"]
        assert bounds["min"] == -2.0
        assert bounds["max"] == 2.0

    def test_default_bounds_for_d_offset(self):
        """Test default bounds for D_offset parameter."""
        pm = ParameterManager()
        assert "D_offset" in pm._default_bounds
        bounds = pm._default_bounds["D_offset"]
        assert bounds["min"] == -1e5
        assert bounds["max"] == 1e5

    def test_default_bounds_for_gamma_dot_t0(self):
        """Test default bounds for gamma_dot_t0 parameter."""
        pm = ParameterManager()
        assert "gamma_dot_t0" in pm._default_bounds
        bounds = pm._default_bounds["gamma_dot_t0"]
        assert bounds["min"] == 1e-6
        assert bounds["max"] == 0.5

    def test_default_bounds_for_beta(self):
        """Test default bounds for beta parameter."""
        pm = ParameterManager()
        assert "beta" in pm._default_bounds
        bounds = pm._default_bounds["beta"]
        assert bounds["min"] == -2.0
        assert bounds["max"] == 2.0

    def test_default_bounds_for_phi0(self):
        """Test default bounds for phi0 parameter."""
        pm = ParameterManager()
        assert "phi0" in pm._default_bounds
        bounds = pm._default_bounds["phi0"]
        assert bounds["min"] == -10.0
        assert bounds["max"] == 10.0

    def test_bounds_have_correct_structure(self):
        """Test that bounds have required keys."""
        pm = ParameterManager()
        bounds = pm._default_bounds["D0"]
        assert "min" in bounds
        assert "max" in bounds
        assert "name" in bounds
        assert "type" in bounds


class TestParameterNameMapping:
    """Tests for parameter name mapping and extraction."""

    def test_extract_base_param_name_contrast(self):
        """Test extracting base name from indexed contrast parameter."""
        pm = ParameterManager()
        base_name = pm._extract_base_param_name("contrast[0]")
        assert base_name == "contrast"

    def test_extract_base_param_name_offset(self):
        """Test extracting base name from indexed offset parameter."""
        pm = ParameterManager()
        base_name = pm._extract_base_param_name("offset[15]")
        assert base_name == "offset"

    def test_extract_base_param_name_non_indexed(self):
        """Test extracting base name from non-indexed parameter returns None."""
        pm = ParameterManager()
        base_name = pm._extract_base_param_name("D0")
        assert base_name is None

    def test_extract_base_param_name_invalid_pattern(self):
        """Test invalid indexed parameter patterns return None."""
        pm = ParameterManager()
        assert pm._extract_base_param_name("contrast[]") is None
        assert pm._extract_base_param_name("contrast[a]") is None
        assert pm._extract_base_param_name("D0[0]") is None


class TestConfigBoundsLoading:
    """Tests for loading bounds from configuration."""

    def test_load_bounds_from_config(self):
        """Test loading bounds from config overrides defaults."""
        config = {
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 500.0, "max": 5000.0},
                ]
            }
        }
        pm = ParameterManager(config_dict=config)
        bounds = pm._default_bounds["D0"]
        assert bounds["min"] == 500.0
        assert bounds["max"] == 5000.0

    def test_load_bounds_preserves_defaults_for_unspecified(self):
        """Test that unspecified bounds keep default values."""
        config = {
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 500.0},
                ]
            }
        }
        pm = ParameterManager(config_dict=config)
        bounds = pm._default_bounds["D0"]
        assert bounds["min"] == 500.0
        assert bounds["max"] == 1e5

    def test_load_bounds_handles_missing_bounds_section(self):
        """Test handling of missing bounds section in config."""
        config = {"parameter_space": {}}
        pm = ParameterManager(config_dict=config)
        assert "D0" in pm._default_bounds

    def test_load_bounds_handles_empty_config(self):
        """Test handling of empty config."""
        pm = ParameterManager(config_dict={})
        assert "D0" in pm._default_bounds

    def test_load_bounds_converts_string_numbers(self):
        """Test that string numbers in YAML are converted to floats."""
        config = {
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": "1e3", "max": "1e4"},
                ]
            }
        }
        pm = ParameterManager(config_dict=config)
        bounds = pm._default_bounds["D0"]
        assert bounds["min"] == 1e3
        assert bounds["max"] == 1e4


class TestStaticVsLaminarFlowParameters:
    """Tests for static vs laminar_flow parameter sets."""

    def test_static_params_subset_of_laminar_flow(self):
        """Test that static params are subset of laminar_flow params."""
        static_set = set(STATIC_PARAM_NAMES)
        laminar_set = set(LAMINAR_FLOW_PARAM_NAMES)
        assert static_set.issubset(laminar_set)

    def test_static_params_count(self):
        """Test number of static parameters."""
        assert len(STATIC_PARAM_NAMES) == 3

    def test_laminar_flow_params_count(self):
        """Test number of laminar_flow parameters."""
        assert len(LAMINAR_FLOW_PARAM_NAMES) == 7

    def test_scaling_params_count(self):
        """Test number of scaling parameters."""
        assert len(SCALING_PARAM_NAMES) == 2

    def test_static_params_content(self):
        """Test that static params contain expected parameters."""
        assert "D0" in STATIC_PARAM_NAMES
        assert "alpha" in STATIC_PARAM_NAMES
        assert "D_offset" in STATIC_PARAM_NAMES

    def test_laminar_flow_params_content(self):
        """Test that laminar_flow params contain expected parameters."""
        expected = [
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ]
        for param in expected:
            assert param in LAMINAR_FLOW_PARAM_NAMES

    def test_scaling_params_content(self):
        """Test that scaling params contain contrast and offset."""
        assert "contrast" in SCALING_PARAM_NAMES
        assert "offset" in SCALING_PARAM_NAMES


class TestParameterBoundsOrdering:
    """Tests for parameter bounds ordering consistency."""

    def test_bounds_ordering_for_static(self):
        """Test that bounds for static mode maintain consistent ordering."""
        pm = ParameterManager(analysis_mode="static")
        static_bounds = [pm._default_bounds[name] for name in STATIC_PARAM_NAMES]
        assert len(static_bounds) == len(STATIC_PARAM_NAMES)
        assert static_bounds[0]["name"] == "D0"
        assert static_bounds[1]["name"] == "alpha"
        assert static_bounds[2]["name"] == "D_offset"

    def test_bounds_ordering_for_laminar_flow(self):
        """Test that bounds for laminar_flow mode maintain consistent ordering."""
        pm = ParameterManager(analysis_mode="laminar_flow")
        flow_bounds = [pm._default_bounds[name] for name in LAMINAR_FLOW_PARAM_NAMES]
        assert len(flow_bounds) == len(LAMINAR_FLOW_PARAM_NAMES)
        assert flow_bounds[0]["name"] == "D0"
        assert flow_bounds[-1]["name"] == "phi0"
