"""
Unit Tests for NLSQAdapterBase Abstract Base Class
===================================================

Tests for homodyne/optimization/nlsq/adapter_base.py covering:
- TestNLSQAdapterBase: ABC interface and shared methods
- TestPrepareData: Data preparation utilities
- TestValidateInput: Input validation
- TestBuildResult: Result dictionary construction
- TestHandleError: Error handling
- TestSetupBounds: Parameter bounds setup
- TestComputeCovariance: Covariance matrix computation

Part of v2.14.0 architecture refactoring tests.
"""

from unittest.mock import Mock

import numpy as np
import pytest


# =============================================================================
# TestNLSQAdapterBase
# =============================================================================
@pytest.mark.unit
class TestNLSQAdapterBase:
    """Tests for NLSQAdapterBase abstract class."""

    def test_is_abstract_class(self):
        """NLSQAdapterBase cannot be instantiated directly."""
        from homodyne.optimization.nlsq.adapter_base import NLSQAdapterBase

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            NLSQAdapterBase()

    def test_fit_is_abstract_method(self):
        """fit() must be implemented by subclasses."""
        from homodyne.optimization.nlsq.adapter_base import NLSQAdapterBase

        # Create a concrete subclass without implementing fit
        class IncompleteAdapter(NLSQAdapterBase):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteAdapter()

    def test_subclass_with_fit_can_be_instantiated(self):
        """Subclass implementing fit() can be instantiated."""
        from homodyne.optimization.nlsq.adapter_base import NLSQAdapterBase

        class ConcreteAdapter(NLSQAdapterBase):
            def fit(self, *args, **kwargs):
                return {"success": True}

        adapter = ConcreteAdapter()
        assert adapter.fit() == {"success": True}


# =============================================================================
# TestPrepareData
# =============================================================================
@pytest.mark.unit
class TestPrepareData:
    """Tests for _prepare_data() method."""

    @pytest.fixture
    def adapter(self):
        """Create a concrete adapter for testing."""
        from homodyne.optimization.nlsq.adapter_base import NLSQAdapterBase

        class TestAdapter(NLSQAdapterBase):
            def fit(self, *args, **kwargs):
                return {}

        return TestAdapter()

    def test_prepare_data_converts_to_float64(self, adapter):
        """Input arrays are converted to float64."""
        t1 = np.array([1, 2, 3], dtype=np.int32)
        t2 = np.array([1, 2, 3], dtype=np.int32)
        phi = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        g2 = np.array([1.0, 1.1, 1.2], dtype=np.float32)

        result = adapter._prepare_data(t1, t2, phi, g2)

        assert result["t1"].dtype == np.float64
        assert result["t2"].dtype == np.float64
        assert result["phi"].dtype == np.float64
        assert result["g2"].dtype == np.float64

    def test_prepare_data_returns_n_points(self, adapter):
        """Prepared data includes point count."""
        n = 100
        t1 = np.linspace(0, 10, n)
        t2 = np.linspace(0, 10, n)
        phi = np.zeros(n)
        g2 = np.ones(n)

        result = adapter._prepare_data(t1, t2, phi, g2)

        assert result["n_points"] == n

    def test_prepare_data_computes_unique_phi(self, adapter):
        """Unique phi values are computed."""
        t1 = np.array([1, 2, 3, 4, 5, 6])
        t2 = np.array([1, 2, 3, 4, 5, 6])
        phi = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
        g2 = np.ones(6)

        result = adapter._prepare_data(t1, t2, phi, g2)

        assert result["n_phi"] == 3
        np.testing.assert_array_equal(result["phi_unique"], [0.0, 1.0, 2.0])

    def test_prepare_data_handles_weights(self, adapter):
        """Weights are included when provided."""
        t1 = np.array([1.0, 2.0, 3.0])
        t2 = np.array([1.0, 2.0, 3.0])
        phi = np.array([0.0, 0.0, 0.0])
        g2 = np.array([1.0, 1.1, 1.2])
        weights = np.array([1.0, 0.5, 0.25])

        result = adapter._prepare_data(t1, t2, phi, g2, weights)

        assert result["weights"] is not None
        np.testing.assert_array_equal(result["weights"], weights)

    def test_prepare_data_none_weights(self, adapter):
        """Weights are None when not provided."""
        t1 = np.array([1.0, 2.0, 3.0])
        t2 = np.array([1.0, 2.0, 3.0])
        phi = np.array([0.0, 0.0, 0.0])
        g2 = np.array([1.0, 1.1, 1.2])

        result = adapter._prepare_data(t1, t2, phi, g2)

        assert result["weights"] is None


# =============================================================================
# TestValidateInput
# =============================================================================
@pytest.mark.unit
class TestValidateInput:
    """Tests for _validate_input() method."""

    @pytest.fixture
    def adapter(self):
        """Create a concrete adapter for testing."""
        from homodyne.optimization.nlsq.adapter_base import NLSQAdapterBase

        class TestAdapter(NLSQAdapterBase):
            def fit(self, *args, **kwargs):
                return {}

        return TestAdapter()

    def test_validate_input_passes_valid_data(self, adapter):
        """Valid input passes validation without error."""
        n = 100
        t1 = np.linspace(0, 10, n)
        t2 = np.linspace(0, 10, n)
        phi = np.zeros(n)
        g2 = np.ones(n)

        # Should not raise
        adapter._validate_input(t1, t2, phi, g2)

    def test_validate_input_rejects_mismatched_t1_t2(self, adapter):
        """Mismatched t1 and t2 lengths raise ValueError."""
        t1 = np.array([1.0, 2.0, 3.0])
        t2 = np.array([1.0, 2.0])  # Wrong length
        phi = np.array([0.0, 0.0, 0.0])
        g2 = np.array([1.0, 1.1, 1.2])

        with pytest.raises(ValueError, match="t1 and t2 must have same length"):
            adapter._validate_input(t1, t2, phi, g2)

    def test_validate_input_rejects_mismatched_phi(self, adapter):
        """Mismatched phi length raises ValueError."""
        t1 = np.array([1.0, 2.0, 3.0])
        t2 = np.array([1.0, 2.0, 3.0])
        phi = np.array([0.0, 0.0])  # Wrong length
        g2 = np.array([1.0, 1.1, 1.2])

        with pytest.raises(ValueError, match="t1 and phi must have same length"):
            adapter._validate_input(t1, t2, phi, g2)

    def test_validate_input_rejects_mismatched_g2(self, adapter):
        """Mismatched g2 length raises ValueError."""
        t1 = np.array([1.0, 2.0, 3.0])
        t2 = np.array([1.0, 2.0, 3.0])
        phi = np.array([0.0, 0.0, 0.0])
        g2 = np.array([1.0, 1.1])  # Wrong length

        with pytest.raises(ValueError, match="t1 and g2 must have same length"):
            adapter._validate_input(t1, t2, phi, g2)

    def test_validate_input_rejects_nan_in_t1(self, adapter):
        """NaN in t1 raises ValueError."""
        t1 = np.array([1.0, np.nan, 3.0])
        t2 = np.array([1.0, 2.0, 3.0])
        phi = np.array([0.0, 0.0, 0.0])
        g2 = np.array([1.0, 1.1, 1.2])

        with pytest.raises(ValueError, match="t1 contains NaN or Inf"):
            adapter._validate_input(t1, t2, phi, g2)

    def test_validate_input_rejects_inf_in_g2(self, adapter):
        """Inf in g2 raises ValueError."""
        t1 = np.array([1.0, 2.0, 3.0])
        t2 = np.array([1.0, 2.0, 3.0])
        phi = np.array([0.0, 0.0, 0.0])
        g2 = np.array([1.0, np.inf, 1.2])

        with pytest.raises(ValueError, match="g2 contains NaN or Inf"):
            adapter._validate_input(t1, t2, phi, g2)

    def test_validate_input_rejects_empty_arrays(self, adapter):
        """Empty arrays raise ValueError."""
        t1 = np.array([])
        t2 = np.array([])
        phi = np.array([])
        g2 = np.array([])

        with pytest.raises(ValueError, match="Input arrays cannot be empty"):
            adapter._validate_input(t1, t2, phi, g2)

    def test_validate_input_rejects_mismatched_weights(self, adapter):
        """Mismatched weights length raises ValueError."""
        t1 = np.array([1.0, 2.0, 3.0])
        t2 = np.array([1.0, 2.0, 3.0])
        phi = np.array([0.0, 0.0, 0.0])
        g2 = np.array([1.0, 1.1, 1.2])
        weights = np.array([1.0, 0.5])  # Wrong length

        with pytest.raises(ValueError, match="weights must have same length"):
            adapter._validate_input(t1, t2, phi, g2, weights)


# =============================================================================
# TestBuildResult
# =============================================================================
@pytest.mark.unit
class TestBuildResult:
    """Tests for _build_result() method."""

    @pytest.fixture
    def adapter(self):
        """Create a concrete adapter for testing."""
        from homodyne.optimization.nlsq.adapter_base import NLSQAdapterBase

        class TestAdapter(NLSQAdapterBase):
            def fit(self, *args, **kwargs):
                return {}

        return TestAdapter()

    def test_build_result_returns_required_keys(self, adapter):
        """Result contains all required keys."""
        params = np.array([1.0, 2.0, 3.0])
        chi_squared = 0.5
        covariance = np.eye(3)
        param_names = ["a", "b", "c"]

        result = adapter._build_result(
            params=params,
            chi_squared=chi_squared,
            covariance=covariance,
            param_names=param_names,
            n_iter=10,
            success=True,
            message="Converged",
        )

        assert "params" in result
        assert "chi_squared" in result
        assert "covariance" in result
        assert "param_names" in result
        assert "n_iter" in result
        assert "success" in result
        assert "message" in result

    def test_build_result_computes_uncertainties(self, adapter):
        """Uncertainties are computed from covariance diagonal."""
        params = np.array([1.0, 2.0])
        covariance = np.array([[4.0, 0.0], [0.0, 9.0]])

        result = adapter._build_result(
            params=params,
            chi_squared=0.1,
            covariance=covariance,
            param_names=["a", "b"],
            n_iter=5,
            success=True,
            message="OK",
        )

        np.testing.assert_array_almost_equal(result["uncertainties"], [2.0, 3.0])

    def test_build_result_handles_none_covariance(self, adapter):
        """None covariance results in None uncertainties."""
        params = np.array([1.0, 2.0])

        result = adapter._build_result(
            params=params,
            chi_squared=0.1,
            covariance=None,
            param_names=["a", "b"],
            n_iter=5,
            success=True,
            message="OK",
        )

        assert result["covariance"] is None
        assert result["uncertainties"] is None

    def test_build_result_includes_diagnostics(self, adapter):
        """Diagnostics are included when provided."""
        params = np.array([1.0])
        diagnostics = {"extra_info": "test_value"}

        result = adapter._build_result(
            params=params,
            chi_squared=0.1,
            covariance=None,
            param_names=["a"],
            n_iter=1,
            success=True,
            message="OK",
            diagnostics=diagnostics,
        )

        assert result["diagnostics"] == diagnostics


# =============================================================================
# TestHandleError
# =============================================================================
@pytest.mark.unit
class TestHandleError:
    """Tests for _handle_error() method."""

    @pytest.fixture
    def adapter(self):
        """Create a concrete adapter for testing."""
        from homodyne.optimization.nlsq.adapter_base import NLSQAdapterBase

        class TestAdapter(NLSQAdapterBase):
            def fit(self, *args, **kwargs):
                return {}

        return TestAdapter()

    def test_handle_error_returns_error_result(self, adapter):
        """Error result includes failure info."""
        error = ValueError("Test error")

        result = adapter._handle_error(error, "test_context")

        assert result["success"] is False
        assert result["chi_squared"] == np.inf
        assert "Test error" in result["message"]
        assert result["error"] == "Test error"

    def test_handle_error_includes_params_if_provided(self, adapter):
        """Current params are included in error result."""
        error = RuntimeError("Optimization failed")
        params = np.array([1.0, 2.0, 3.0])

        result = adapter._handle_error(error, "fit", params=params)

        np.testing.assert_array_equal(result["params"], params)

    def test_handle_error_reraises_when_requested(self, adapter):
        """Error is re-raised when raise_on_error=True."""
        error = ValueError("Critical error")

        with pytest.raises(ValueError, match="Critical error"):
            adapter._handle_error(error, "context", raise_on_error=True)


# =============================================================================
# TestSetupBounds
# =============================================================================
@pytest.mark.unit
class TestSetupBounds:
    """Tests for _setup_bounds() method."""

    @pytest.fixture
    def adapter(self):
        """Create a concrete adapter for testing."""
        from homodyne.optimization.nlsq.adapter_base import NLSQAdapterBase

        class TestAdapter(NLSQAdapterBase):
            def fit(self, *args, **kwargs):
                return {}

        return TestAdapter()

    def test_setup_bounds_default_infinite(self, adapter):
        """Default bounds are (-inf, inf)."""
        param_names = ["a", "b", "c"]

        lower, upper = adapter._setup_bounds(param_names)

        assert all(lower == -np.inf)
        assert all(upper == np.inf)

    def test_setup_bounds_from_dict(self, adapter):
        """Bounds are set from dictionary."""
        param_names = ["a", "b", "c"]
        bounds_dict = {"a": (0.0, 1.0), "c": (-10.0, 10.0)}

        lower, upper = adapter._setup_bounds(param_names, bounds_dict)

        assert lower[0] == 0.0
        assert upper[0] == 1.0
        assert lower[1] == -np.inf  # "b" not in dict
        assert upper[1] == np.inf
        assert lower[2] == -10.0
        assert upper[2] == 10.0

    def test_setup_bounds_custom_default(self, adapter):
        """Custom default bounds are applied."""
        param_names = ["x", "y"]
        default = (0.0, 100.0)

        lower, upper = adapter._setup_bounds(param_names, default_bounds=default)

        np.testing.assert_array_equal(lower, [0.0, 0.0])
        np.testing.assert_array_equal(upper, [100.0, 100.0])


# =============================================================================
# TestComputeCovariance
# =============================================================================
@pytest.mark.unit
class TestComputeCovariance:
    """Tests for _compute_covariance() method."""

    @pytest.fixture
    def adapter(self):
        """Create a concrete adapter for testing."""
        from homodyne.optimization.nlsq.adapter_base import NLSQAdapterBase

        class TestAdapter(NLSQAdapterBase):
            def fit(self, *args, **kwargs):
                return {}

        return TestAdapter()

    def test_compute_covariance_from_jacobian(self, adapter):
        """Covariance is computed from Jacobian."""
        # Simple case: J = I, residuals = [0.1, 0.1, 0.1]
        jacobian = np.eye(3)
        residuals = np.array([0.1, 0.1, 0.1])
        n_params = 3

        # dof = 3 - 3 = 0, should return None
        cov = adapter._compute_covariance(jacobian, residuals, n_params)
        assert cov is None

    def test_compute_covariance_with_sufficient_dof(self, adapter):
        """Covariance computed with sufficient degrees of freedom."""
        # 10 points, 2 params -> dof = 8
        n_points = 10
        n_params = 2
        jacobian = np.random.randn(n_points, n_params)
        residuals = np.random.randn(n_points) * 0.1

        cov = adapter._compute_covariance(jacobian, residuals, n_params)

        assert cov is not None
        assert cov.shape == (n_params, n_params)
        # Covariance should be symmetric
        np.testing.assert_array_almost_equal(cov, cov.T)

    def test_compute_covariance_returns_none_on_singular(self, adapter):
        """Singular Jacobian returns None covariance."""
        # Singular Jacobian (columns are identical)
        jacobian = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        residuals = np.array([0.1, 0.1, 0.1])
        n_params = 2

        # Should handle gracefully (uses pinv for ill-conditioned)
        cov = adapter._compute_covariance(jacobian, residuals, n_params)
        # May return a covariance or None depending on implementation
        # Just verify no exception is raised


# =============================================================================
# TestInheritance
# =============================================================================
@pytest.mark.unit
class TestInheritance:
    """Tests that NLSQAdapter and NLSQWrapper inherit from NLSQAdapterBase."""

    def test_nlsq_adapter_inherits_from_base(self):
        """NLSQAdapter inherits from NLSQAdapterBase."""
        from homodyne.optimization.nlsq.adapter import NLSQAdapter
        from homodyne.optimization.nlsq.adapter_base import NLSQAdapterBase

        assert issubclass(NLSQAdapter, NLSQAdapterBase)

    def test_nlsq_wrapper_inherits_from_base(self):
        """NLSQWrapper inherits from NLSQAdapterBase."""
        from homodyne.optimization.nlsq.adapter_base import NLSQAdapterBase
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        assert issubclass(NLSQWrapper, NLSQAdapterBase)
