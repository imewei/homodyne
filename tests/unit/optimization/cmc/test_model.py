"""Tests for CMC NumPyro model module.

Comprehensive tests for:
- xpcs_model: Main NumPyro probabilistic model
- xpcs_model_single_chain: Single-chain model with log-space D0
- validate_model_output: Physics validation
- get_model_param_count: Parameter counting
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import jax.numpy as jnp
import numpy as np
import pytest

from homodyne.optimization.cmc.model import (
    get_model_param_count,
    validate_model_output,
    xpcs_model,
    xpcs_model_single_chain,
)


# =============================================================================
# Fixtures
# =============================================================================


@dataclass
class MockPriorSpec:
    """Mock PriorDistribution-like object."""

    dist_type: str
    min_val: float
    max_val: float
    mu: float
    sigma: float


class MockParameterSpace:
    """Mock ParameterSpace for testing without full config system."""

    def __init__(self, analysis_mode: str = "static"):
        self.analysis_mode = analysis_mode
        self._priors = {
            "contrast": MockPriorSpec("TruncatedNormal", 0.0, 1.0, 0.5, 0.2),
            "offset": MockPriorSpec("TruncatedNormal", 0.5, 1.5, 1.0, 0.2),
            "D0": MockPriorSpec("TruncatedNormal", 100.0, 100000.0, 1000.0, 500.0),
            "alpha": MockPriorSpec("TruncatedNormal", 0.1, 2.0, 0.5, 0.3),
            "D_offset": MockPriorSpec("TruncatedNormal", 0.0, 1000.0, 10.0, 50.0),
            "gamma_dot_t0": MockPriorSpec("TruncatedNormal", 1.0, 10000.0, 100.0, 200.0),
            "beta": MockPriorSpec("TruncatedNormal", 0.1, 2.0, 0.5, 0.3),
            "gamma_dot_t_offset": MockPriorSpec("TruncatedNormal", -100.0, 100.0, 0.0, 20.0),
            "phi0": MockPriorSpec("TruncatedNormal", -np.pi, np.pi, 0.0, 1.0),
        }
        self._bounds = {
            "contrast": (0.0, 1.0),
            "offset": (0.5, 1.5),
            "D0": (100.0, 100000.0),
            "alpha": (0.1, 2.0),
            "D_offset": (0.0, 1000.0),
            "gamma_dot_t0": (1.0, 10000.0),
            "beta": (0.1, 2.0),
            "gamma_dot_t_offset": (-100.0, 100.0),
            "phi0": (-np.pi, np.pi),
        }

    def get_prior(self, param_name: str) -> MockPriorSpec:
        if param_name not in self._priors:
            raise KeyError(f"Unknown parameter: {param_name}")
        return self._priors[param_name]

    def get_bounds(self, param_name: str) -> tuple[float, float]:
        if param_name not in self._bounds:
            raise KeyError(f"Unknown parameter: {param_name}")
        return self._bounds[param_name]


@pytest.fixture
def mock_parameter_space_static():
    """Mock ParameterSpace for static mode."""
    return MockParameterSpace("static")


@pytest.fixture
def mock_parameter_space_laminar():
    """Mock ParameterSpace for laminar flow mode."""
    return MockParameterSpace("laminar_flow")


@pytest.fixture
def synthetic_pooled_data():
    """Generate synthetic pooled data for model testing.

    Note: phi must be the UNIQUE phi values for compute_g1_total.
    The model expects phi_indices to map data points to these unique values.
    """
    np.random.seed(42)
    n_total = 500
    n_phi = 3

    # Generate pooled time coordinates
    t1 = np.random.uniform(0, 10, n_total)
    t2 = np.random.uniform(0, 10, n_total)

    # Generate UNIQUE phi angles (this is what compute_g1_total expects)
    phi_unique = np.array([0.0, np.pi / 4, np.pi / 2])
    phi_indices = np.random.randint(0, n_phi, n_total)

    # For the model, we pass phi_unique as phi (not the replicated version)
    # This matches how compute_g1_total works after Nov 2025 fix

    # Generate synthetic C2 data (realistic range 1.0 - 1.5)
    tau = np.abs(t1 - t2) + 0.1
    contrast = 0.5
    offset = 1.0
    g1 = np.exp(-tau / 5.0)  # Simple decay
    data = contrast * g1**2 + offset + np.random.normal(0, 0.01, n_total)

    return {
        "data": jnp.array(data),
        "t1": jnp.array(t1),
        "t2": jnp.array(t2),
        "phi": jnp.array(phi_unique),  # UNIQUE phi values
        "phi_indices": jnp.array(phi_indices),
        "n_phi": n_phi,
        "n_total": n_total,
    }


# =============================================================================
# Tests for get_model_param_count
# =============================================================================


class TestGetModelParamCount:
    """Tests for get_model_param_count function."""

    def test_static_mode_single_phi(self):
        """Test parameter count for static mode with 1 phi angle."""
        count = get_model_param_count(n_phi=1, analysis_mode="static")

        # 1 contrast + 1 offset + 3 physical (D0, alpha, D_offset) + 1 sigma = 6
        expected = 1 * 2 + 3 + 1
        assert count == expected

    def test_static_mode_multi_phi(self):
        """Test parameter count for static mode with 3 phi angles."""
        count = get_model_param_count(n_phi=3, analysis_mode="static")

        # 3 contrast + 3 offset + 3 physical + 1 sigma = 10
        expected = 3 * 2 + 3 + 1
        assert count == expected

    def test_laminar_flow_mode_single_phi(self):
        """Test parameter count for laminar_flow mode with 1 phi angle."""
        count = get_model_param_count(n_phi=1, analysis_mode="laminar_flow")

        # 1 contrast + 1 offset + 7 physical + 1 sigma = 10
        expected = 1 * 2 + 7 + 1
        assert count == expected

    def test_laminar_flow_mode_multi_phi(self):
        """Test parameter count for laminar_flow mode with 3 phi angles."""
        count = get_model_param_count(n_phi=3, analysis_mode="laminar_flow")

        # 3 contrast + 3 offset + 7 physical + 1 sigma = 14
        expected = 3 * 2 + 7 + 1
        assert count == expected

    @pytest.mark.parametrize("n_phi", [1, 2, 5, 10, 36])
    def test_scaling_with_n_phi(self, n_phi):
        """Test that parameter count scales correctly with n_phi."""
        static_count = get_model_param_count(n_phi=n_phi, analysis_mode="static")
        laminar_count = get_model_param_count(n_phi=n_phi, analysis_mode="laminar_flow")

        # Static: 2*n_phi + 3 + 1
        assert static_count == 2 * n_phi + 4

        # Laminar: 2*n_phi + 7 + 1
        assert laminar_count == 2 * n_phi + 8

        # Laminar always has 4 more parameters than static
        assert laminar_count - static_count == 4


# =============================================================================
# Tests for validate_model_output
# =============================================================================


class TestValidateModelOutput:
    """Tests for validate_model_output function."""

    def test_valid_output(self):
        """Test validation passes for valid output."""
        c2_theory = jnp.array([1.0, 1.2, 1.5, 1.3, 1.1])
        params = jnp.array([1000.0, 0.5, 10.0])

        result = validate_model_output(c2_theory, params)

        assert result is True

    def test_nan_in_c2(self):
        """Test validation fails for NaN values."""
        c2_theory = jnp.array([1.0, jnp.nan, 1.5])
        params = jnp.array([1000.0, 0.5, 10.0])

        result = validate_model_output(c2_theory, params)

        assert result is False

    def test_inf_in_c2(self):
        """Test validation fails for infinite values."""
        c2_theory = jnp.array([1.0, jnp.inf, 1.5])
        params = jnp.array([1000.0, 0.5, 10.0])

        result = validate_model_output(c2_theory, params)

        assert result is False

    def test_negative_c2(self):
        """Test validation fails for negative C2 (unphysical)."""
        c2_theory = jnp.array([1.0, -1.5, 1.5])  # Below -1.0 threshold
        params = jnp.array([1000.0, 0.5, 10.0])

        result = validate_model_output(c2_theory, params)

        assert result is False

    def test_very_high_c2(self):
        """Test validation fails for very high C2 values."""
        c2_theory = jnp.array([1.0, 15.0, 1.5])  # Above 10.0 threshold
        params = jnp.array([1000.0, 0.5, 10.0])

        result = validate_model_output(c2_theory, params)

        assert result is False

    def test_edge_of_valid_range(self):
        """Test validation at edge of valid range."""
        # Just within valid range
        c2_lower_edge = jnp.array([0.0, 1.0, 1.5])  # >= -1.0
        c2_upper_edge = jnp.array([1.0, 9.0, 1.5])  # <= 10.0

        assert validate_model_output(c2_lower_edge, jnp.array([1.0])) is True
        assert validate_model_output(c2_upper_edge, jnp.array([1.0])) is True

    def test_empty_array(self):
        """Test validation with empty array."""
        c2_theory = jnp.array([])
        params = jnp.array([])

        # Empty arrays should be valid (no invalid values)
        result = validate_model_output(c2_theory, params)

        assert result is True


# =============================================================================
# Tests for xpcs_model - Trace only (no sampling)
# =============================================================================


class TestXPCSModelStructure:
    """Structural tests for xpcs_model without running MCMC."""

    def test_model_callable(self, mock_parameter_space_static, synthetic_pooled_data):
        """Test that xpcs_model is callable with correct arguments."""
        import numpyro

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                xpcs_model(
                    data=synthetic_pooled_data["data"],
                    t1=synthetic_pooled_data["t1"],
                    t2=synthetic_pooled_data["t2"],
                    phi=synthetic_pooled_data["phi"],
                    phi_indices=synthetic_pooled_data["phi_indices"],
                    q=0.01,
                    L=2000000.0,
                    dt=0.1,
                    analysis_mode="static",
                    parameter_space=mock_parameter_space_static,
                    n_phi=synthetic_pooled_data["n_phi"],
                    noise_scale=0.1,
                )

        # Check trace contains expected parameters
        sampled_params = [k for k, v in trace.items() if v.get("type") == "sample"]
        assert "D0" in sampled_params
        assert "alpha" in sampled_params
        assert "D_offset" in sampled_params
        assert "sigma" in sampled_params
        assert "contrast_0" in sampled_params
        assert "offset_0" in sampled_params

    def test_model_laminar_flow_params(
        self, mock_parameter_space_laminar, synthetic_pooled_data
    ):
        """Test laminar_flow mode includes additional parameters."""
        import numpyro

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                xpcs_model(
                    data=synthetic_pooled_data["data"],
                    t1=synthetic_pooled_data["t1"],
                    t2=synthetic_pooled_data["t2"],
                    phi=synthetic_pooled_data["phi"],
                    phi_indices=synthetic_pooled_data["phi_indices"],
                    q=0.01,
                    L=2000000.0,
                    dt=0.1,
                    analysis_mode="laminar_flow",
                    parameter_space=mock_parameter_space_laminar,
                    n_phi=synthetic_pooled_data["n_phi"],
                    noise_scale=0.1,
                )

        sampled_params = [k for k, v in trace.items() if v.get("type") == "sample"]

        # Check laminar-specific parameters
        assert "gamma_dot_t0" in sampled_params
        assert "beta" in sampled_params
        assert "gamma_dot_t_offset" in sampled_params
        assert "phi0" in sampled_params

    def test_parameter_ordering(self, mock_parameter_space_static, synthetic_pooled_data):
        """Test parameters are sampled in NumPyro-required order."""
        import numpyro
        from numpyro.handlers import Messenger

        sampled_order = []

        class OrderTracker(Messenger):
            """Track order of sampled parameters."""

            def process_message(self, msg):
                if msg.get("type") == "sample" and not msg.get("is_observed", False):
                    sampled_order.append(msg["name"])

        with numpyro.handlers.seed(rng_seed=42):
            with OrderTracker():
                xpcs_model(
                    data=synthetic_pooled_data["data"],
                    t1=synthetic_pooled_data["t1"],
                    t2=synthetic_pooled_data["t2"],
                    phi=synthetic_pooled_data["phi"],
                    phi_indices=synthetic_pooled_data["phi_indices"],
                    q=0.01,
                    L=2000000.0,
                    dt=0.1,
                    analysis_mode="static",
                    parameter_space=mock_parameter_space_static,
                    n_phi=3,
                    noise_scale=0.1,
                )

        # Verify ordering: contrast_*, offset_*, physical, sigma
        # Find indices
        contrast_indices = [i for i, n in enumerate(sampled_order) if n.startswith("contrast_")]
        offset_indices = [i for i, n in enumerate(sampled_order) if n.startswith("offset_")]
        d0_index = sampled_order.index("D0")

        # All contrast before all offset
        assert max(contrast_indices) < min(offset_indices), (
            f"Contrast indices {contrast_indices} should all come before offset indices {offset_indices}"
        )

        # All offset before physical params
        assert max(offset_indices) < d0_index, (
            f"Offset indices {offset_indices} should all come before D0 at {d0_index}"
        )


# =============================================================================
# Tests for xpcs_model_single_chain
# =============================================================================


class TestXPCSModelSingleChain:
    """Tests for xpcs_model_single_chain with log-space D0."""

    def test_log_d0_parameter(self, mock_parameter_space_static, synthetic_pooled_data):
        """Test that log_D0 is sampled when use_log_d0=True."""
        import numpyro

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                xpcs_model_single_chain(
                    data=synthetic_pooled_data["data"],
                    t1=synthetic_pooled_data["t1"],
                    t2=synthetic_pooled_data["t2"],
                    phi=synthetic_pooled_data["phi"],
                    phi_indices=synthetic_pooled_data["phi_indices"],
                    q=0.01,
                    L=2000000.0,
                    dt=0.1,
                    analysis_mode="static",
                    parameter_space=mock_parameter_space_static,
                    n_phi=synthetic_pooled_data["n_phi"],
                    noise_scale=0.1,
                    use_log_d0=True,
                )

        sampled_params = [k for k, v in trace.items() if v.get("type") == "sample"]

        # log_D0 should be sampled, D0 should be deterministic
        assert "log_D0" in sampled_params

    def test_no_log_d0_parameter(self, mock_parameter_space_static, synthetic_pooled_data):
        """Test that D0 is sampled directly when use_log_d0=False."""
        import numpyro

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                xpcs_model_single_chain(
                    data=synthetic_pooled_data["data"],
                    t1=synthetic_pooled_data["t1"],
                    t2=synthetic_pooled_data["t2"],
                    phi=synthetic_pooled_data["phi"],
                    phi_indices=synthetic_pooled_data["phi_indices"],
                    q=0.01,
                    L=2000000.0,
                    dt=0.1,
                    analysis_mode="static",
                    parameter_space=mock_parameter_space_static,
                    n_phi=synthetic_pooled_data["n_phi"],
                    noise_scale=0.1,
                    use_log_d0=False,
                )

        sampled_params = [k for k, v in trace.items() if v.get("type") == "sample"]

        # D0 should be sampled directly
        assert "D0" in sampled_params
        assert "log_D0" not in sampled_params


# =============================================================================
# Scientific Validation Tests
# =============================================================================


class TestModelPhysics:
    """Scientific validation tests for physics consistency."""

    def test_c2_range_realistic(self, mock_parameter_space_static, synthetic_pooled_data):
        """Test that model produces physically realistic C2 values."""
        import numpyro

        # Get the theoretical C2 from a model trace
        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                xpcs_model(
                    data=synthetic_pooled_data["data"],
                    t1=synthetic_pooled_data["t1"],
                    t2=synthetic_pooled_data["t2"],
                    phi=synthetic_pooled_data["phi"],
                    phi_indices=synthetic_pooled_data["phi_indices"],
                    q=0.01,
                    L=2000000.0,
                    dt=0.1,
                    analysis_mode="static",
                    parameter_space=mock_parameter_space_static,
                    n_phi=synthetic_pooled_data["n_phi"],
                    noise_scale=0.1,
                )

        # Get the obs distribution mean (c2_theory in the model)
        obs_dist = trace["obs"]["fn"]
        c2_theory = obs_dist.loc

        # C2 should be in physically reasonable range
        assert jnp.all(jnp.isfinite(c2_theory)), "C2 contains non-finite values"
        # Allow for noise: realistic C2 is typically 1.0 - 2.0
        assert jnp.all(c2_theory >= 0.5), f"C2 too low: min={jnp.min(c2_theory)}"
        assert jnp.all(c2_theory <= 3.0), f"C2 too high: max={jnp.max(c2_theory)}"

    def test_n_phi_consistency(self, mock_parameter_space_static, synthetic_pooled_data):
        """Test that n_phi matches number of per-angle parameters."""
        import numpyro

        n_phi = synthetic_pooled_data["n_phi"]

        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                xpcs_model(
                    data=synthetic_pooled_data["data"],
                    t1=synthetic_pooled_data["t1"],
                    t2=synthetic_pooled_data["t2"],
                    phi=synthetic_pooled_data["phi"],
                    phi_indices=synthetic_pooled_data["phi_indices"],
                    q=0.01,
                    L=2000000.0,
                    dt=0.1,
                    analysis_mode="static",
                    parameter_space=mock_parameter_space_static,
                    n_phi=n_phi,
                    noise_scale=0.1,
                )

        sampled_params = [k for k, v in trace.items() if v.get("type") == "sample"]

        # Count contrast and offset parameters
        n_contrast = sum(1 for p in sampled_params if p.startswith("contrast_"))
        n_offset = sum(1 for p in sampled_params if p.startswith("offset_"))

        assert n_contrast == n_phi, f"Expected {n_phi} contrast params, got {n_contrast}"
        assert n_offset == n_phi, f"Expected {n_phi} offset params, got {n_offset}"
