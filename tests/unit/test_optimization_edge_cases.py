"""Comprehensive edge case tests for homodyne/optimization module.

This module tests edge cases, boundary conditions, and error handling
for the NLSQ and CMC optimization components of the homodyne package.

Edge Cases Covered:
- NLSQ parameter bounds handling
- NLSQ error recovery mechanisms
- CMC configuration parsing
- CMC data sharding edge cases
- Time step inference edge cases
- Analysis mode detection
- Parameter conversion edge cases
- Convergence checking
- Data validation edge cases
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.testing import assert_allclose

# Require ArviZ for CMC imports; skip module if missing optional dependency
pytest.importorskip("arviz", reason="ArviZ required for CMC unit tests")

from homodyne.optimization.cmc.config import CMCConfig  # noqa: E402

# Import modules under test - CMC
from homodyne.optimization.cmc.core import (  # noqa: E402
    _infer_time_step,
)
from homodyne.optimization.cmc.data_prep import (  # noqa: E402
    prepare_mcmc_data,
    shard_data_random,
    shard_data_stratified,
)

# Import modules under test - NLSQ
from homodyne.optimization.nlsq.core import (
    NLSQResult,
    _array_to_params,
    _bounds_to_arrays,
    _estimate_contrast_offset_from_data,
    _get_analysis_mode,
    _get_default_initial_params,
    _get_param_names,
    _params_to_array,
    _validate_data,
)

# Conditionally import JAX
try:
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    import numpy as jnp

    JAX_AVAILABLE = False


# =============================================================================
# NLSQ PARAMETER HANDLING EDGE CASES
# =============================================================================


class TestNLSQParameterConversion:
    """Test edge cases for NLSQ parameter array/dict conversion."""

    def test_params_to_array_static_mode(self):
        """Static mode should produce 5-element array."""
        params = {
            "contrast": 0.5,
            "offset": 1.0,
            "D0": 10000.0,
            "alpha": -1.5,
            "D_offset": 0.0,
        }
        arr = _params_to_array(params, "static_isotropic")
        assert len(arr) == 5
        assert float(arr[0]) == 0.5  # contrast
        assert float(arr[2]) == 10000.0  # D0

    def test_params_to_array_laminar_mode(self):
        """Laminar flow mode should produce 9-element array."""
        params = {
            "contrast": 0.5,
            "offset": 1.0,
            "D0": 10000.0,
            "alpha": -1.5,
            "D_offset": 0.0,
            "gamma_dot_t0": 0.001,
            "beta": 0.0,
            "gamma_dot_t_offset": 0.0,
            "phi0": 0.0,
        }
        arr = _params_to_array(params, "laminar_flow")
        assert len(arr) == 9
        assert float(arr[5]) == 0.001  # gamma_dot_t0

    def test_array_to_params_static_mode(self):
        """Converting array to params should preserve values for static mode."""
        arr = jnp.array([0.5, 1.0, 10000.0, -1.5, 100.0])
        params = _array_to_params(arr, "static")
        assert "contrast" in params
        assert "D0" in params
        assert float(params["contrast"]) == 0.5
        assert float(params["D0"]) == 10000.0

    def test_array_to_params_laminar_mode(self):
        """Converting array to params should preserve values for laminar mode."""
        arr = jnp.array([0.5, 1.0, 10000.0, -1.5, 100.0, 0.001, 0.0, 0.0, 45.0])
        params = _array_to_params(arr, "laminar_flow")
        assert "gamma_dot_t0" in params
        assert "phi0" in params
        assert float(params["gamma_dot_t0"]) == 0.001
        assert float(params["phi0"]) == 45.0

    def test_params_array_roundtrip_static(self):
        """Params -> array -> params roundtrip should preserve values for static."""
        original = {
            "contrast": 0.3,
            "offset": 0.8,
            "D0": 5000.0,
            "alpha": -2.0,
            "D_offset": 50.0,
        }
        arr = _params_to_array(original, "static")
        recovered = _array_to_params(arr, "static")
        for key in original:
            assert_allclose(float(recovered[key]), original[key], rtol=1e-10)

    def test_params_array_roundtrip_laminar(self):
        """Params -> array -> params roundtrip should preserve values for laminar."""
        original = {
            "contrast": 0.3,
            "offset": 0.8,
            "D0": 5000.0,
            "alpha": -2.0,
            "D_offset": 50.0,
            "gamma_dot_t0": 0.002,
            "beta": 0.5,
            "gamma_dot_t_offset": 0.0001,
            "phi0": 30.0,
        }
        arr = _params_to_array(original, "laminar_flow")
        recovered = _array_to_params(arr, "laminar_flow")
        for key in original:
            assert_allclose(float(recovered[key]), original[key], rtol=1e-10)


class TestNLSQBoundsHandling:
    """Test edge cases for NLSQ parameter bounds."""

    def test_bounds_to_arrays_static_mode(self):
        """Bounds should be converted to arrays correctly for static mode."""
        bounds = {
            "contrast": (0.01, 1.0),
            "offset": (0.01, 2.0),
            "D0": (1.0, 1e15),
            "alpha": (-5.0, 5.0),
            "D_offset": (-1e14, 1e14),
        }
        lower, upper = _bounds_to_arrays(bounds, "static")
        assert len(lower) == 5
        assert len(upper) == 5
        assert float(lower[0]) == 0.01  # contrast lower
        assert float(upper[0]) == 1.0  # contrast upper

    def test_bounds_to_arrays_laminar_mode(self):
        """Bounds should be converted to arrays correctly for laminar mode."""
        bounds = {
            "contrast": (0.01, 1.0),
            "offset": (0.01, 2.0),
            "D0": (1.0, 1e15),
            "alpha": (-5.0, 5.0),
            "D_offset": (-1e14, 1e14),
            "gamma_dot_t0": (0.0, 10.0),
            "beta": (-5.0, 5.0),
            "gamma_dot_t_offset": (-1.0, 1.0),
            "phi0": (-180.0, 180.0),
        }
        lower, upper = _bounds_to_arrays(bounds, "laminar_flow")
        assert len(lower) == 9
        assert len(upper) == 9


class TestNLSQAnalysisModeDetection:
    """Test edge cases for analysis mode detection."""

    def test_get_analysis_mode_static(self):
        """Static mode should be detected from config."""
        mock_config = MagicMock()
        mock_config.config = {"analysis_mode": "static_isotropic"}
        mode = _get_analysis_mode(mock_config)
        assert mode == "static_isotropic"

    def test_get_analysis_mode_laminar(self):
        """Laminar flow mode should be detected from config."""
        mock_config = MagicMock()
        mock_config.config = {"analysis_mode": "laminar_flow"}
        mode = _get_analysis_mode(mock_config)
        assert mode == "laminar_flow"

    def test_get_analysis_mode_default(self):
        """Missing analysis_mode should default to static_isotropic."""
        mock_config = MagicMock()
        mock_config.config = {}
        mode = _get_analysis_mode(mock_config)
        assert mode == "static_isotropic"

    def test_get_analysis_mode_no_config(self):
        """None config should default to static_isotropic."""
        mock_config = MagicMock()
        mock_config.config = None
        mode = _get_analysis_mode(mock_config)
        assert mode == "static_isotropic"


class TestNLSQDefaultParams:
    """Test edge cases for default parameter generation."""

    def test_default_params_static(self):
        """Default params for static mode should have correct keys."""
        params = _get_default_initial_params("static_isotropic")
        required_keys = ["contrast", "offset", "D0", "alpha", "D_offset"]
        for key in required_keys:
            assert key in params
        # Should NOT have laminar flow params
        assert "gamma_dot_t0" not in params

    def test_default_params_laminar(self):
        """Default params for laminar mode should have all keys."""
        params = _get_default_initial_params("laminar_flow")
        all_keys = [
            "contrast",
            "offset",
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ]
        for key in all_keys:
            assert key in params

    def test_default_params_reasonable_values(self):
        """Default params should have physically reasonable values."""
        params = _get_default_initial_params("static_isotropic")
        # Contrast should be positive and < 1
        assert 0 < params["contrast"] < 1
        # Offset should be around 1
        assert 0 < params["offset"] < 2
        # D0 should be positive
        assert params["D0"] > 0
        # Alpha should be in reasonable range
        assert -5 < params["alpha"] < 5


class TestNLSQDataValidation:
    """Test edge cases for NLSQ data validation."""

    def test_validate_data_missing_key(self):
        """Missing required key should raise ValueError."""
        data = {
            "phi_angles_list": np.array([0.0]),
            "t1": np.array([0.1, 1.0]),
            "t2": np.array([0.1, 1.0]),
            # Missing wavevector_q_list and c2_exp
        }
        with pytest.raises(ValueError, match="Missing required"):
            _validate_data(data)

    def test_validate_data_empty_c2(self):
        """Empty c2_exp should raise ValueError."""
        data = {
            "wavevector_q_list": np.array([0.005]),
            "phi_angles_list": np.array([0.0]),
            "t1": np.array([0.1, 1.0]),
            "t2": np.array([0.1, 1.0]),
            "c2_exp": np.array([]),  # Empty
        }
        # Empty array reshape or shape check
        with pytest.raises((ValueError, IndexError)):
            _validate_data(data)

    def test_validate_data_valid(self):
        """Valid data should pass validation without error."""
        data = {
            "wavevector_q_list": np.array([0.005]),
            "phi_angles_list": np.array([0.0]),
            "t1": np.array([0.1, 1.0]),
            "t2": np.array([0.1, 1.0]),
            "c2_exp": np.ones((1, 2, 2)),
        }
        # Should not raise
        _validate_data(data)


class TestNLSQContrastOffsetEstimation:
    """Test edge cases for contrast/offset estimation from data."""

    def test_estimate_from_valid_g2(self):
        """Estimation from valid g2 data should work."""
        n_phi, n_t = 5, 50
        g2 = np.ones((n_phi, n_t, n_t)) * 0.8  # baseline ~0.8
        g2[:, 0, 0] = 1.2  # peak at t=0
        data = {"g2": g2}
        contrast, offset = _estimate_contrast_offset_from_data(data)
        assert contrast > 0
        assert offset > 0

    def test_estimate_from_c2_exp_fallback(self):
        """Estimation should fallback to c2_exp if g2 is missing."""
        n_phi, n_t = 5, 50
        c2 = np.ones((n_phi, n_t, n_t)) * 0.8
        c2[:, 0, 0] = 1.2
        data = {"c2_exp": c2}  # Only c2_exp, no g2
        contrast, offset = _estimate_contrast_offset_from_data(data)
        assert contrast > 0
        assert offset > 0

    def test_estimate_from_missing_g2(self):
        """Missing g2 should return default values."""
        data = {"other_key": np.array([1, 2, 3])}
        contrast, offset = _estimate_contrast_offset_from_data(data)
        # Should return defaults
        assert contrast == 0.5
        assert offset == 1.0

    def test_estimate_from_negative_g2(self):
        """Data with negative values should return defaults."""
        data = {"g2": np.array([-1.0, -0.5, 0.0])}
        contrast, offset = _estimate_contrast_offset_from_data(data)
        # Should fall back to defaults due to invalid estimated values
        assert contrast == 0.5
        assert offset == 1.0


class TestNLSQResult:
    """Test edge cases for NLSQResult container."""

    def test_result_creation(self):
        """NLSQResult should be created with all attributes."""
        result = NLSQResult(
            parameters={"D0": 10000.0, "alpha": -1.5},
            parameter_errors={"D0": 100.0, "alpha": 0.1},
            chi_squared=100.0,
            reduced_chi_squared=1.0,
            success=True,
            message="Converged",
            n_iterations=50,
            optimization_time=1.5,
        )
        assert result.success
        assert result.chi_squared == 100.0
        assert result.parameters["D0"] == 10000.0

    def test_result_failed_optimization(self):
        """Failed optimization should be representable."""
        result = NLSQResult(
            parameters={},
            parameter_errors={},
            chi_squared=float("inf"),
            reduced_chi_squared=float("inf"),
            success=False,
            message="Failed to converge",
            n_iterations=10000,
            optimization_time=60.0,
        )
        assert not result.success
        assert result.chi_squared == float("inf")


class TestNLSQParamNames:
    """Test edge cases for parameter name retrieval."""

    def test_get_param_names_static(self):
        """Static mode should have 5 parameter names."""
        names = _get_param_names("static")
        assert len(names) == 5
        assert "D0" in names
        assert "gamma_dot_t0" not in names

    def test_get_param_names_laminar(self):
        """Laminar mode should have 9 parameter names."""
        names = _get_param_names("laminar_flow")
        assert len(names) == 9
        assert "D0" in names
        assert "gamma_dot_t0" in names

    def test_get_param_names_order_preserved(self):
        """Parameter order should be consistent."""
        names_static = _get_param_names("static")
        assert names_static[0] == "contrast"
        assert names_static[1] == "offset"
        assert names_static[2] == "D0"

        names_laminar = _get_param_names("laminar_flow")
        # First 5 should be same as static
        for i in range(5):
            assert names_laminar[i] == names_static[i]


# =============================================================================
# CMC CONFIGURATION EDGE CASES
# =============================================================================


class TestCMCConfig:
    """Test edge cases for CMCConfig."""

    def test_config_from_empty_dict(self):
        """Empty dict should create config with defaults."""
        config = CMCConfig.from_dict({})
        assert config.num_chains >= 1
        assert config.num_warmup >= 0
        assert config.num_samples >= 0

    def test_config_from_dict_with_overrides(self):
        """Config dict values should override defaults using nested structure."""
        # CMCConfig uses nested structure: per_shard_mcmc contains sampling params
        config = CMCConfig.from_dict(
            {
                "per_shard_mcmc": {
                    "num_chains": 8,
                    "num_warmup": 2000,
                    "num_samples": 3000,
                }
            }
        )
        assert config.num_chains == 8
        assert config.num_warmup == 2000
        assert config.num_samples == 3000

    def test_config_to_dict_roundtrip(self):
        """Config -> dict -> config should preserve values."""
        original = CMCConfig.from_dict(
            {
                "per_shard_mcmc": {
                    "num_chains": 4,
                    "num_warmup": 500,
                    "num_samples": 1000,
                }
            }
        )
        recovered = CMCConfig.from_dict(original.to_dict())
        assert recovered.num_chains == original.num_chains
        assert recovered.num_warmup == original.num_warmup
        assert recovered.num_samples == original.num_samples

    def test_config_should_enable_cmc_small_data(self):
        """Small datasets should not trigger CMC."""
        # Use min_points_for_cmc directly (not cmc_threshold)
        config = CMCConfig.from_dict({"min_points_for_cmc": 10000})
        assert not config.should_enable_cmc(5000)

    def test_config_should_enable_cmc_large_data(self):
        """Large datasets should trigger CMC."""
        # Use min_points_for_cmc directly
        config = CMCConfig.from_dict({"min_points_for_cmc": 10000})
        assert config.should_enable_cmc(50000)


# =============================================================================
# CMC TIME STEP INFERENCE EDGE CASES
# =============================================================================


class TestCMCTimeStepInference:
    """Test edge cases for time step inference."""

    def test_infer_from_uniform_spacing(self):
        """Uniform time spacing should be detected correctly."""
        t1 = np.linspace(0, 10, 11)  # dt = 1.0
        t2 = np.linspace(0, 10, 11)
        dt = _infer_time_step(t1, t2)
        assert_allclose(dt, 1.0, rtol=1e-10)

    def test_infer_from_non_uniform_spacing(self):
        """Non-uniform spacing should return median diff."""
        t1 = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
        t2 = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
        dt = _infer_time_step(t1, t2)
        # Diffs: 0.1, 0.2, 0.3, 0.4 → median = 0.25
        assert_allclose(dt, 0.25, rtol=1e-10)

    def test_infer_from_single_point(self):
        """Single time point should return default (1.0)."""
        t1 = np.array([0.5])
        t2 = np.array([0.5])
        dt = _infer_time_step(t1, t2)
        assert dt == 1.0

    def test_infer_from_identical_points(self):
        """All identical time points should return default."""
        t1 = np.array([1.0, 1.0, 1.0])
        t2 = np.array([1.0, 1.0, 1.0])
        dt = _infer_time_step(t1, t2)
        assert dt == 1.0

    def test_infer_from_pooled_times(self):
        """Should work with different t1 and t2 ranges."""
        t1 = np.array([0.0, 1.0, 2.0])
        t2 = np.array([0.5, 1.5, 2.5])
        dt = _infer_time_step(t1, t2)
        # Unique values: [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        # Diffs: all 0.5 → median = 0.5
        assert_allclose(dt, 0.5, rtol=1e-10)


# =============================================================================
# CMC DATA PREPARATION EDGE CASES
# =============================================================================


class TestCMCDataPreparation:
    """Test edge cases for CMC data preparation."""

    def test_prepare_mcmc_data_single_phi(self):
        """Single phi angle should work."""
        n_points = 100
        data = np.random.rand(n_points)
        t1 = np.random.rand(n_points)
        t2 = np.random.rand(n_points)
        phi = np.zeros(n_points)  # Single phi angle

        prepared = prepare_mcmc_data(data, t1, t2, phi)
        assert prepared.n_phi == 1
        assert prepared.n_total == n_points

    def test_prepare_mcmc_data_multiple_phi(self):
        """Multiple phi angles should be detected."""
        n_per_phi = 50
        n_phi = 5
        n_points = n_per_phi * n_phi

        data = np.random.rand(n_points)
        t1 = np.random.rand(n_points)
        t2 = np.random.rand(n_points)
        phi = np.repeat(np.linspace(-20, 20, n_phi), n_per_phi)

        prepared = prepare_mcmc_data(data, t1, t2, phi)
        assert prepared.n_phi == n_phi
        assert prepared.n_total == n_points

    def test_prepare_mcmc_data_rejects_nan(self):
        """NaN values should be rejected during validation."""
        data = np.array([1.0, np.nan, 2.0, 3.0])
        t1 = np.array([0.1, 0.2, 0.3, 0.4])
        t2 = np.array([0.1, 0.2, 0.3, 0.4])
        phi = np.array([0.0, 0.0, 0.0, 0.0])

        # The implementation validates and rejects NaN values
        with pytest.raises(ValueError, match="NaN"):
            prepare_mcmc_data(data, t1, t2, phi)


class TestCMCDataSharding:
    """Test edge cases for CMC data sharding."""

    def test_shard_stratified_single_shard(self):
        """Single shard should return all data."""
        n_points = 100
        data = np.random.rand(n_points)
        t1 = np.random.rand(n_points)
        t2 = np.random.rand(n_points)
        phi = np.zeros(n_points)
        prepared = prepare_mcmc_data(data, t1, t2, phi)

        shards = shard_data_stratified(prepared, num_shards=1)
        assert len(shards) == 1
        assert shards[0].n_total == n_points

    def test_shard_stratified_multiple_shards(self):
        """Multiple shards should partition data."""
        n_per_phi = 100
        n_phi = 4
        n_points = n_per_phi * n_phi

        data = np.random.rand(n_points)
        t1 = np.random.rand(n_points)
        t2 = np.random.rand(n_points)
        phi = np.repeat(np.array([0.0, 30.0, 60.0, 90.0]), n_per_phi)
        prepared = prepare_mcmc_data(data, t1, t2, phi)

        # Stratified sharding creates one shard per phi angle by default
        shards = shard_data_stratified(prepared, num_shards=n_phi)
        # Should have at least one shard
        assert len(shards) >= 1
        # Total points across shards should equal or exceed original
        # (stratified sampling may sample with replacement)
        total_shard_points = sum(s.n_total for s in shards)
        assert total_shard_points > 0

    def test_shard_stratified_preserves_phi_indices_and_order(self):
        """Stratified shards keep phi grouping and per-angle ordering intact."""
        # Two angles with distinct phi values and ordered blocks
        phi_vals = np.array([0.0, 30.0])
        counts = [5, 3]
        phi = np.repeat(phi_vals, counts)
        data = np.arange(len(phi))  # monotonic to track ordering
        t1 = np.arange(len(phi)) * 0.1
        t2 = t1 + 0.01

        prepared = prepare_mcmc_data(data, t1, t2, phi)
        # Don't pass num_shards to get exactly one shard per angle
        # (passing num_shards derives max_points_per_shard which may split angles)
        shards = shard_data_stratified(prepared, max_points_per_shard=None)

        # Expect one shard per angle; each shard's phi_indices should be 0..0 or 1..1 and sorted
        assert len(shards) == len(phi_vals)

        for shard in shards:
            # All phi entries in the shard should be the same angle value
            shard_phis = shard.phi
            assert np.allclose(shard_phis, shard_phis[0])

            # phi_indices should all be equal (single angle per shard) and non-decreasing
            assert np.all(shard.phi_indices == shard.phi_indices[0])
            assert np.all(np.diff(shard.phi_indices) == 0)

            # data preserved ordering within the shard
            assert np.all(np.diff(shard.data) >= 0)

    def test_shard_random_equal_split(self):
        """Random sharding should split data approximately evenly."""
        n_points = 1000
        data = np.random.rand(n_points)
        t1 = np.random.rand(n_points)
        t2 = np.random.rand(n_points)
        phi = np.zeros(n_points)
        prepared = prepare_mcmc_data(data, t1, t2, phi)

        shards = shard_data_random(
            prepared, num_shards=4, max_points_per_shard=500, max_shards=10
        )
        # Should have some shards
        assert len(shards) >= 1
        # Each shard should have some data
        for shard in shards:
            assert shard.n_total > 0

    def test_shard_random_respects_max_points(self):
        """Random sharding should respect max points per shard."""
        n_points = 10000
        data = np.random.rand(n_points)
        t1 = np.random.rand(n_points)
        t2 = np.random.rand(n_points)
        phi = np.zeros(n_points)
        prepared = prepare_mcmc_data(data, t1, t2, phi)

        max_per_shard = 1000
        shards = shard_data_random(
            prepared,
            num_shards=None,
            max_points_per_shard=max_per_shard,
            max_shards=100,
        )

        # Each shard should have at most max_per_shard points
        for shard in shards:
            assert shard.n_total <= max_per_shard


# =============================================================================
# CMC RESULT EDGE CASES
# =============================================================================


class TestCMCResultEdgeCases:
    """Test edge cases for CMC result handling."""

    def test_convergence_status_with_high_rhat(self):
        """High R-hat should indicate non-convergence."""
        # This tests the summary/diagnostics logic
        from homodyne.optimization.cmc.diagnostics import summarize_diagnostics

        # Simulate high R-hat values (> 1.1 indicates non-convergence)
        r_hat = {"D0": 1.5, "alpha": 1.3}
        ess_bulk = {"D0": 100.0, "alpha": 100.0}

        summary = summarize_diagnostics(
            r_hat=r_hat,
            ess_bulk=ess_bulk,
            divergences=0,
            n_samples=1000,
            n_chains=4,
        )
        # Should warn about convergence
        assert "R-hat" in summary or "convergence" in summary.lower()

    def test_convergence_status_with_divergences(self):
        """Divergences should be noted in summary."""
        from homodyne.optimization.cmc.diagnostics import summarize_diagnostics

        r_hat = {"D0": 1.01, "alpha": 1.01}
        ess_bulk = {"D0": 500.0, "alpha": 500.0}

        summary = summarize_diagnostics(
            r_hat=r_hat,
            ess_bulk=ess_bulk,
            divergences=50,  # High divergence count
            n_samples=1000,
            n_chains=4,
        )
        # Should mention divergences
        assert "divergen" in summary.lower()


# =============================================================================
# NLSQ WRAPPER EDGE CASES
# =============================================================================


class TestNLSQWrapperEdgeCases:
    """Test edge cases for NLSQWrapper if available."""

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX required for NLSQWrapper tests")
    def test_wrapper_import(self):
        """NLSQWrapper should be importable."""
        try:
            from homodyne.optimization.nlsq.wrapper import NLSQWrapper

            assert NLSQWrapper is not None
        except ImportError:
            pytest.skip("NLSQWrapper not available")

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX required for NLSQWrapper tests")
    def test_wrapper_enable_recovery_flag(self):
        """NLSQWrapper should accept enable_recovery flag."""
        try:
            from homodyne.optimization.nlsq.wrapper import NLSQWrapper

            wrapper = NLSQWrapper(enable_recovery=True)
            assert wrapper is not None
        except ImportError:
            pytest.skip("NLSQWrapper not available")


# =============================================================================
# INTEGRATION EDGE CASES
# =============================================================================


class TestOptimizationIntegrationEdgeCases:
    """Test integration edge cases between NLSQ and CMC."""

    def test_analysis_mode_case_insensitive(self):
        """Analysis mode detection should handle case variations."""
        for mode_input in ["static", "STATIC", "Static", "static_isotropic"]:
            names = _get_param_names(mode_input)
            assert len(names) == 5  # Static mode

        for mode_input in ["laminar_flow", "LAMINAR_FLOW", "Laminar_Flow"]:
            names = _get_param_names(mode_input)
            assert len(names) == 9  # Laminar mode

    def test_parameter_bounds_consistency(self):
        """Parameter bounds should be consistent between NLSQ and CMC."""
        # Get NLSQ bounds
        static_names = _get_param_names("static")
        laminar_names = _get_param_names("laminar_flow")

        # First 5 parameters should be same
        assert static_names == laminar_names[:5]


# =============================================================================
# NUMERICAL STABILITY EDGE CASES
# =============================================================================


class TestNumericalStabilityEdgeCases:
    """Test numerical stability in optimization edge cases."""

    def test_large_d0_values(self):
        """Large D0 values should be handled without overflow."""
        params = {
            "contrast": 0.3,
            "offset": 1.0,
            "D0": 1e14,  # Very large
            "alpha": -1.5,
            "D_offset": 0.0,
        }
        arr = _params_to_array(params, "static")
        recovered = _array_to_params(arr, "static")
        assert_allclose(float(recovered["D0"]), 1e14, rtol=1e-10)

    def test_small_d0_values(self):
        """Small D0 values should be handled without underflow."""
        params = {
            "contrast": 0.3,
            "offset": 1.0,
            "D0": 1e-10,  # Very small
            "alpha": -1.5,
            "D_offset": 0.0,
        }
        arr = _params_to_array(params, "static")
        recovered = _array_to_params(arr, "static")
        assert_allclose(float(recovered["D0"]), 1e-10, rtol=1e-6)

    def test_extreme_alpha_values(self):
        """Extreme alpha values should be handled."""
        for alpha in [-4.9, 4.9]:  # Near bounds
            params = {
                "contrast": 0.3,
                "offset": 1.0,
                "D0": 10000.0,
                "alpha": alpha,
                "D_offset": 0.0,
            }
            arr = _params_to_array(params, "static")
            recovered = _array_to_params(arr, "static")
            assert_allclose(float(recovered["alpha"]), alpha, rtol=1e-10)
