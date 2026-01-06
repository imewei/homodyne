"""Tests for NLSQAdapterBase (T057).

Tests the abstract base class for NLSQ adapters.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestNLSQAdapterBase:
    """Tests for T057: NLSQAdapterBase interface."""

    def test_adapter_base_exists(self):
        """Test that NLSQAdapterBase ABC exists."""
        from homodyne.optimization.nlsq.adapter_base import NLSQAdapterBase

        assert hasattr(NLSQAdapterBase, "_prepare_data")
        assert hasattr(NLSQAdapterBase, "_validate_input")
        assert hasattr(NLSQAdapterBase, "_build_result")
        assert hasattr(NLSQAdapterBase, "_handle_error")
        assert hasattr(NLSQAdapterBase, "_setup_bounds")
        assert hasattr(NLSQAdapterBase, "_compute_covariance")

    def test_prepare_data_signature(self):
        """Test that _prepare_data has expected signature."""
        from homodyne.optimization.nlsq.adapter_base import NLSQAdapterBase
        import inspect

        sig = inspect.signature(NLSQAdapterBase._prepare_data)
        params = list(sig.parameters.keys())
        assert "self" in params
        # Should accept data-related parameters

    def test_validate_input_signature(self):
        """Test that _validate_input has expected signature."""
        from homodyne.optimization.nlsq.adapter_base import NLSQAdapterBase
        import inspect

        sig = inspect.signature(NLSQAdapterBase._validate_input)
        params = list(sig.parameters.keys())
        assert "self" in params

    def test_build_result_signature(self):
        """Test that _build_result has expected signature."""
        from homodyne.optimization.nlsq.adapter_base import NLSQAdapterBase
        import inspect

        sig = inspect.signature(NLSQAdapterBase._build_result)
        params = list(sig.parameters.keys())
        assert "self" in params

    def test_handle_error_signature(self):
        """Test that _handle_error has expected signature."""
        from homodyne.optimization.nlsq.adapter_base import NLSQAdapterBase
        import inspect

        sig = inspect.signature(NLSQAdapterBase._handle_error)
        params = list(sig.parameters.keys())
        assert "self" in params

    def test_setup_bounds_signature(self):
        """Test that _setup_bounds has expected signature."""
        from homodyne.optimization.nlsq.adapter_base import NLSQAdapterBase
        import inspect

        sig = inspect.signature(NLSQAdapterBase._setup_bounds)
        params = list(sig.parameters.keys())
        assert "self" in params

    def test_compute_covariance_signature(self):
        """Test that _compute_covariance has expected signature."""
        from homodyne.optimization.nlsq.adapter_base import NLSQAdapterBase
        import inspect

        sig = inspect.signature(NLSQAdapterBase._compute_covariance)
        params = list(sig.parameters.keys())
        assert "self" in params


class TestAdapterInheritance:
    """Test that adapters inherit from NLSQAdapterBase."""

    def test_adapter_inherits_from_base(self):
        """Test that NLSQAdapter inherits from NLSQAdapterBase."""
        from homodyne.optimization.nlsq.adapter import NLSQAdapter
        from homodyne.optimization.nlsq.adapter_base import NLSQAdapterBase

        assert issubclass(NLSQAdapter, NLSQAdapterBase)

    def test_wrapper_inherits_from_base(self):
        """Test that NLSQWrapper inherits from NLSQAdapterBase."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper
        from homodyne.optimization.nlsq.adapter_base import NLSQAdapterBase

        assert issubclass(NLSQWrapper, NLSQAdapterBase)


class TestSharedMethodImplementation:
    """Test that shared methods work correctly."""

    def test_prepare_data_returns_expected_structure(self):
        """Test that _prepare_data returns structured data."""
        from homodyne.optimization.nlsq.adapter_base import NLSQAdapterBase

        # Create a concrete subclass for testing
        class TestAdapter(NLSQAdapterBase):
            def fit(self, *args, **kwargs):
                pass

        adapter = TestAdapter()

        # Test with minimal input
        t1 = np.array([1.0, 2.0, 3.0])
        t2 = np.array([1.0, 2.0, 3.0])
        phi = np.array([0.0, 0.0, 0.0])
        g2 = np.array([1.0, 0.9, 0.8])

        result = adapter._prepare_data(t1, t2, phi, g2)
        assert result is not None

    def test_validate_input_checks_dimensions(self):
        """Test that _validate_input validates array dimensions."""
        from homodyne.optimization.nlsq.adapter_base import NLSQAdapterBase

        class TestAdapter(NLSQAdapterBase):
            def fit(self, *args, **kwargs):
                pass

        adapter = TestAdapter()

        # Valid input
        t1 = np.array([1.0, 2.0])
        t2 = np.array([1.0, 2.0])
        phi = np.array([0.0, 0.0])
        g2 = np.array([1.0, 0.9])

        # Should not raise
        adapter._validate_input(t1, t2, phi, g2)

    def test_validate_input_rejects_mismatched_arrays(self):
        """Test that _validate_input rejects mismatched array lengths."""
        from homodyne.optimization.nlsq.adapter_base import NLSQAdapterBase

        class TestAdapter(NLSQAdapterBase):
            def fit(self, *args, **kwargs):
                pass

        adapter = TestAdapter()

        # Mismatched input
        t1 = np.array([1.0, 2.0])
        t2 = np.array([1.0])  # Wrong length
        phi = np.array([0.0, 0.0])
        g2 = np.array([1.0, 0.9])

        with pytest.raises(ValueError):
            adapter._validate_input(t1, t2, phi, g2)
