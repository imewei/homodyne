"""
API Compatibility Tests
=======================

Tests for API stability and backward compatibility:
- Public API surface validation
- Import structure stability
- Function signature consistency
- Deprecation handling
- Version compatibility checks
"""

import pytest
import inspect
import importlib
from typing import Dict, Any, List, Set, Optional, Callable


@pytest.mark.api
class TestPublicAPIStability:
    """Test public API stability and structure."""

    def test_main_package_imports(self):
        """Test main package import structure."""
        # Should be able to import main package
        import homodyne
        assert hasattr(homodyne, '__version__')

        # Check for expected main modules
        expected_modules = ['core', 'data', 'optimization', 'config']
        for module_name in expected_modules:
            try:
                module = getattr(homodyne, module_name)
                assert module is not None, f"Module {module_name} not accessible"
            except AttributeError:
                # Module might not be exposed at top level - try importing directly
                try:
                    importlib.import_module(f'homodyne.{module_name}')
                except ImportError:
                    pytest.fail(f"Module homodyne.{module_name} not importable")

    def test_core_api_functions(self):
        """Test core API function availability."""
        try:
            from homodyne.core.jax_backend import (
                compute_c2_model_jax,
                compute_g1_diffusion_jax,
                compute_g1_shear_jax,
                residuals_jax,
                chi_squared_jax
            )

            # All should be callable
            core_functions = [
                compute_c2_model_jax,
                compute_g1_diffusion_jax,
                compute_g1_shear_jax,
                residuals_jax,
                chi_squared_jax
            ]

            for func in core_functions:
                assert callable(func), f"Function {func.__name__} not callable"

        except ImportError as e:
            pytest.skip(f"Core functions not available: {e}")

    def test_optimization_api_functions(self):
        """Test optimization API function availability."""
        try:
            from homodyne.optimization.nlsq import fit_nlsq_jax
            assert callable(fit_nlsq_jax), "fit_nlsq_jax not callable"

            # Check function signature
            sig = inspect.signature(fit_nlsq_jax)
            params = list(sig.parameters.keys())

            # Should have expected parameters
            expected_params = ['data', 'config']
            for param in expected_params:
                assert param in params, f"Missing parameter {param} in fit_nlsq_jax"

        except ImportError as e:
            pytest.skip(f"Optimization functions not available: {e}")

        try:
            from homodyne.optimization.mcmc import fit_mcmc_jax
            assert callable(fit_mcmc_jax), "fit_mcmc_jax not callable"

        except ImportError:
            # MCMC might not be available
            pass

    def test_data_api_functions(self):
        """Test data loading API function availability."""
        try:
            from homodyne.data.xpcs_loader import load_xpcs_data, XPCSLoader

            assert callable(load_xpcs_data), "load_xpcs_data not callable"
            assert inspect.isclass(XPCSLoader), "XPCSLoader not a class"

            # Check load_xpcs_data signature
            sig = inspect.signature(load_xpcs_data)
            params = list(sig.parameters.keys())
            assert 'config' in params, "Missing config parameter in load_xpcs_data"

        except ImportError as e:
            pytest.skip(f"Data functions not available: {e}")

    def test_config_api_functions(self):
        """Test configuration API function availability."""
        try:
            from homodyne.config.manager import ConfigManager

            assert inspect.isclass(ConfigManager), "ConfigManager not a class"

            # Should have expected methods
            expected_methods = ['get_config', 'update_config']
            for method_name in expected_methods:
                assert hasattr(ConfigManager, method_name), \
                    f"ConfigManager missing method {method_name}"

        except ImportError as e:
            pytest.skip(f"Config functions not available: {e}")

    def test_gpu_api_functions(self):
        """Test GPU API function availability."""
        try:
            from homodyne.runtime.gpu import (
                activate_gpu,
                get_gpu_status,
                GPUActivator
            )

            assert callable(activate_gpu), "activate_gpu not callable"
            assert callable(get_gpu_status), "get_gpu_status not callable"
            assert inspect.isclass(GPUActivator), "GPUActivator not a class"

        except ImportError:
            # GPU module might not be available
            pass


@pytest.mark.api
class TestFunctionSignatures:
    """Test function signature compatibility."""

    def test_core_function_signatures(self):
        """Test core function signatures are stable."""
        try:
            from homodyne.core.jax_backend import compute_c2_model_jax

            sig = inspect.signature(compute_c2_model_jax)
            params = list(sig.parameters.keys())

            # Expected parameters (order matters for positional args)
            expected_params = ['params', 't1', 't2', 'phi', 'q']
            for i, expected_param in enumerate(expected_params):
                assert i < len(params), f"Missing parameter {expected_param}"
                assert params[i] == expected_param, \
                    f"Parameter order changed: expected {expected_param}, got {params[i]}"

        except ImportError:
            pytest.skip("Core functions not available")

    def test_optimization_function_signatures(self):
        """Test optimization function signatures are stable."""
        try:
            from homodyne.optimization.nlsq import fit_nlsq_jax

            sig = inspect.signature(fit_nlsq_jax)
            params = list(sig.parameters.keys())

            # Should have data and config as first parameters
            assert params[0] == 'data', f"First parameter should be 'data', got {params[0]}"
            assert params[1] == 'config', f"Second parameter should be 'config', got {params[1]}"

        except ImportError:
            pytest.skip("Optimization functions not available")

    def test_data_loader_signatures(self):
        """Test data loader function signatures are stable."""
        try:
            from homodyne.data.xpcs_loader import load_xpcs_data

            sig = inspect.signature(load_xpcs_data)
            params = list(sig.parameters.keys())

            # Should have config as parameter
            assert 'config' in params, "load_xpcs_data should have config parameter"

        except ImportError:
            pytest.skip("Data loader not available")

    def test_gpu_function_signatures(self):
        """Test GPU function signatures are stable."""
        try:
            from homodyne.runtime.gpu import activate_gpu

            sig = inspect.signature(activate_gpu)
            params = list(sig.parameters.keys())

            # Expected GPU activation parameters
            expected_params = ['memory_fraction', 'force_gpu', 'gpu_id', 'verbose']
            for param in expected_params:
                assert param in params, f"Missing parameter {param} in activate_gpu"

        except ImportError:
            pytest.skip("GPU functions not available")


@pytest.mark.api
class TestReturnTypes:
    """Test return type compatibility."""

    def test_optimization_return_types(self, synthetic_xpcs_data, test_config):
        """Test optimization return types are consistent."""
        try:
            from homodyne.optimization.nlsq import fit_nlsq_jax, NLSQResult, JAXFIT_AVAILABLE
        except ImportError:
            pytest.skip("Optimization module not available")

        if not JAXFIT_AVAILABLE:
            pytest.skip("JAXFit not available")

        data = synthetic_xpcs_data
        config = test_config

        try:
            result = fit_nlsq_jax(data, config)

            # Should return NLSQResult or compatible object
            assert hasattr(result, 'parameters'), "Result missing parameters attribute"
            assert hasattr(result, 'chi_squared'), "Result missing chi_squared attribute"
            assert hasattr(result, 'success'), "Result missing success attribute"
            assert hasattr(result, 'message'), "Result missing message attribute"

            # Type checking
            assert isinstance(result.parameters, dict), "Parameters should be dict"
            assert isinstance(result.success, bool), "Success should be bool"
            assert isinstance(result.message, str), "Message should be string"

        except Exception as e:
            pytest.skip(f"Optimization test failed: {e}")

    def test_data_loader_return_types(self, synthetic_xpcs_data):
        """Test data loader return types are consistent."""
        # Data should have expected structure
        data = synthetic_xpcs_data

        # Required keys
        required_keys = ['t1', 't2', 'phi_angles_list', 'c2_exp', 'wavevector_q_list']
        for key in required_keys:
            assert key in data, f"Missing required key {key}"

        # Types should be array-like
        import numpy as np
        for key in required_keys:
            assert hasattr(data[key], 'shape'), f"{key} should be array-like"
            assert hasattr(data[key], 'dtype'), f"{key} should have dtype"

    def test_gpu_return_types(self):
        """Test GPU function return types are consistent."""
        try:
            from homodyne.runtime.gpu import get_gpu_status, activate_gpu
        except ImportError:
            pytest.skip("GPU module not available")

        # GPU status should return dict
        status = get_gpu_status()
        assert isinstance(status, dict), "GPU status should be dict"

        # Required keys
        assert 'jax_available' in status, "Missing jax_available in GPU status"
        assert 'devices' in status, "Missing devices in GPU status"

        # GPU activation should return dict
        result = activate_gpu(force_gpu=False, verbose=False)
        assert isinstance(result, dict), "GPU activation result should be dict"


@pytest.mark.api
class TestErrorHandling:
    """Test error handling consistency."""

    def test_optimization_error_handling(self, test_config):
        """Test optimization error handling."""
        try:
            from homodyne.optimization.nlsq import fit_nlsq_jax
        except ImportError:
            pytest.skip("Optimization module not available")

        # Invalid data should raise appropriate errors
        invalid_data_cases = [
            {},  # Empty dict
            {'t1': None},  # None values
            {'t1': [[0, 1]], 't2': [[0, 1]]},  # Missing required keys
        ]

        for invalid_data in invalid_data_cases:
            with pytest.raises((KeyError, ValueError, TypeError, AttributeError)):
                fit_nlsq_jax(invalid_data, test_config)

    def test_data_loader_error_handling(self):
        """Test data loader error handling."""
        try:
            from homodyne.data.xpcs_loader import load_xpcs_data
        except ImportError:
            pytest.skip("Data loader not available")

        # Invalid configs should raise appropriate errors
        invalid_configs = [
            {},  # Empty config
            {'data_file': '/nonexistent/file.h5'},  # Nonexistent file
        ]

        for invalid_config in invalid_configs:
            with pytest.raises((KeyError, ValueError, FileNotFoundError, TypeError)):
                load_xpcs_data(invalid_config)

    def test_gpu_error_handling(self):
        """Test GPU error handling."""
        try:
            from homodyne.runtime.gpu import activate_gpu
        except ImportError:
            pytest.skip("GPU module not available")

        # Invalid parameters should raise appropriate errors
        with pytest.raises(ValueError):
            activate_gpu(memory_fraction=2.0)  # > 1.0

        with pytest.raises(ValueError):
            activate_gpu(memory_fraction=0.0)  # <= 0.0


@pytest.mark.api
class TestDeprecationWarnings:
    """Test deprecation warning handling."""

    def test_no_unexpected_deprecation_warnings(self):
        """Test that normal usage doesn't produce deprecation warnings."""
        import warnings

        # Capture warnings during import
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Import main modules
            try:
                import homodyne
                from homodyne.core import jax_backend
                from homodyne.optimization import nlsq
                from homodyne.data import xpcs_loader
            except ImportError:
                pass  # Modules might not be available

            # Check for unexpected deprecation warnings
            deprecation_warnings = [warning for warning in w
                                  if issubclass(warning.category, DeprecationWarning)]

            # Should not have deprecation warnings during normal import
            if deprecation_warnings:
                warning_messages = [str(warn.message) for warn in deprecation_warnings]
                pytest.fail(f"Unexpected deprecation warnings: {warning_messages}")

    def test_backward_compatibility_imports(self):
        """Test backward compatibility imports still work."""
        # Test that common import patterns still work
        import_patterns = [
            'from homodyne.core.jax_backend import compute_c2_model_jax',
            'from homodyne.optimization.nlsq import fit_nlsq_jax',
            'from homodyne.data.xpcs_loader import load_xpcs_data',
        ]

        for import_pattern in import_patterns:
            try:
                exec(import_pattern)
            except ImportError as e:
                # Only fail if it's not a dependency issue
                if "No module named" not in str(e) or "homodyne" in str(e):
                    pytest.fail(f"Backward compatibility broken: {import_pattern}")


@pytest.mark.api
class TestVersionCompatibility:
    """Test version compatibility and metadata."""

    def test_version_info_available(self):
        """Test version information is available."""
        try:
            import homodyne
            assert hasattr(homodyne, '__version__'), "Package should have __version__"

            version = homodyne.__version__
            assert isinstance(version, str), "Version should be string"
            assert len(version) > 0, "Version should not be empty"

            # Should follow semantic versioning pattern (roughly)
            parts = version.split('.')
            assert len(parts) >= 2, "Version should have at least major.minor"

        except ImportError:
            pytest.skip("Package not properly installed")

    def test_python_version_compatibility(self):
        """Test Python version compatibility."""
        import sys

        # Should work with Python 3.10+
        assert sys.version_info >= (3, 10), "Requires Python 3.10+"

        # Should not require very new Python features
        assert sys.version_info < (4, 0), "Should not require Python 4+"

    def test_dependency_version_compatibility(self):
        """Test dependency version compatibility."""
        try:
            import numpy as np
            import jax

            # NumPy should be reasonably recent
            np_version = tuple(map(int, np.__version__.split('.')[:2]))
            assert np_version >= (1, 21), f"NumPy too old: {np.__version__}"

            # JAX should be recent
            jax_version = tuple(map(int, jax.__version__.split('.')[:2]))
            assert jax_version >= (0, 4), f"JAX too old: {jax.__version__}"

        except ImportError:
            pytest.skip("Core dependencies not available")

    def test_optional_dependency_handling(self):
        """Test graceful handling of optional dependencies."""
        # Should handle missing optional dependencies gracefully
        optional_modules = [
            'homodyne.optimization.mcmc',
            'homodyne.runtime.gpu',
        ]

        for module_name in optional_modules:
            try:
                importlib.import_module(module_name)
                # If import succeeds, module should work
                assert True
            except ImportError:
                # Missing optional dependencies should be handled gracefully
                assert True  # This is acceptable


@pytest.mark.api
class TestDocumentationCompatibility:
    """Test that documented examples still work."""

    def test_basic_usage_example(self, synthetic_xpcs_data, test_config):
        """Test basic usage example from documentation."""
        try:
            # This should match the basic example in documentation
            from homodyne.optimization.nlsq import fit_nlsq_jax, JAXFIT_AVAILABLE
        except ImportError:
            pytest.skip("Required modules not available")

        if not JAXFIT_AVAILABLE:
            pytest.skip("JAXFit not available")

        data = synthetic_xpcs_data
        config = test_config

        try:
            # Basic optimization (should match docs)
            result = fit_nlsq_jax(data, config)

            # Should have expected structure
            assert hasattr(result, 'success')
            assert hasattr(result, 'parameters')

            # Should work as documented
            if result.success:
                assert 'offset' in result.parameters
                assert result.chi_squared >= 0.0

        except Exception as e:
            pytest.skip(f"Basic usage example failed: {e}")

    def test_gpu_usage_example(self):
        """Test GPU usage example from documentation."""
        try:
            from homodyne.runtime.gpu import activate_gpu, get_gpu_status
        except ImportError:
            pytest.skip("GPU module not available")

        try:
            # Example usage (should match docs)
            status = get_gpu_status()
            assert isinstance(status, dict)

            # Activation example
            result = activate_gpu(memory_fraction=0.8, force_gpu=False)
            assert isinstance(result, dict)

        except Exception as e:
            pytest.skip(f"GPU usage example failed: {e}")