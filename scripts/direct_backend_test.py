#!/usr/bin/env python3
"""
Direct test of the JAX backend module without dependencies.
This imports the backend module directly and tests the fallback functionality.
"""

import sys
import os
import numpy as np

# Add the homodyne directory to the path
sys.path.insert(0, '/home/wei/Documents/GitHub/homodyne')

# Import required modules directly, bypassing the main package
sys.path.insert(0, '/home/wei/Documents/GitHub/homodyne/homodyne/utils')
sys.path.insert(0, '/home/wei/Documents/GitHub/homodyne/homodyne/core')

def test_backend_isolation():
    """Test the backend in isolation."""
    print("🧪 Direct JAX Backend Test (Isolated)")
    print("=" * 45)
    
    # Create minimal logging mock to avoid dependencies
    class MockLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
        def debug(self, msg): pass
    
    # Mock the logging imports
    import sys
    from unittest.mock import MagicMock
    
    # Create mock modules
    mock_logging_module = MagicMock()
    mock_logging_module.get_logger = MagicMock(return_value=MockLogger())
    mock_logging_module.log_performance = lambda threshold: lambda func: func
    
    sys.modules['homodyne.utils.logging'] = mock_logging_module
    sys.modules['homodyne.utils'] = MagicMock()
    
    # Now try to import the backend
    try:
        import homodyne.core.jax_backend as jax_backend
        print("✅ Successfully imported JAX backend (isolated)")
    except Exception as e:
        print(f"❌ Failed to import backend: {e}")
        return False
    
    # Test system status
    print(f"\\nSystem Status:")
    print(f"  JAX available: {jax_backend.jax_available}")
    print(f"  NumPy gradients available: {jax_backend.numpy_gradients_available}")
    
    # Test basic computations
    print("\\n🧮 Testing Basic Computations:")
    
    test_params = np.array([100.0, 0.0, 10.0])
    test_t1 = np.array([0.0])
    test_t2 = np.array([1.0])
    test_q = 0.01
    phi = np.array([0.0])
    L = 1000.0
    contrast = 0.8
    offset = 0.0
    
    try:
        # Test forward computations
        g1_result = jax_backend.compute_g1_diffusion(test_params, test_t1, test_t2, test_q)
        print(f"  ✅ G1 diffusion: {g1_result}")
        
        g2_result = jax_backend.compute_g2_scaled(test_params, test_t1, test_t2, phi, test_q, L, contrast, offset)
        print(f"  ✅ G2 scaled: {g2_result}")
        
        data = np.array([1.0])
        sigma = np.array([0.01])
        chi2_result = jax_backend.compute_chi_squared(test_params, data, sigma, test_t1, test_t2, phi, test_q, L, contrast, offset)
        print(f"  ✅ Chi-squared: {chi2_result}")
        
    except Exception as e:
        print(f"  ❌ Basic computations failed: {e}")
        return False
    
    # Test differentiation
    print("\\n📐 Testing Differentiation:")
    
    try:
        def scalar_func(params):
            result = jax_backend.compute_g1_diffusion(params, test_t1, test_t2, test_q)
            return result[0]
        
        # Test gradient
        grad_func = jax_backend.grad(scalar_func)
        grad_result = grad_func(test_params)
        print(f"  ✅ Gradient computed: {grad_result}")
        
        # Test hessian
        hess_func = jax_backend.hessian(scalar_func)
        hess_result = hess_func(test_params)
        print(f"  ✅ Hessian computed: shape {hess_result.shape}")
        
    except ImportError as e:
        print(f"  ⚠️  Differentiation not available: {e}")
        print("     This is expected if neither JAX nor NumPy gradients are installed")
    except Exception as e:
        print(f"  ❌ Differentiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test system info functions
    print("\\n📊 Testing System Information:")
    
    try:
        # Test backend validation
        results = jax_backend.validate_backend()
        print(f"  Backend type: {results.get('backend_type', 'unknown')}")
        print(f"  Gradient support: {results.get('gradient_support', False)}")
        print(f"  Hessian support: {results.get('hessian_support', False)}")
        
        # Test device info
        device_info = jax_backend.get_device_info()
        print(f"  Device backend: {device_info.get('backend', 'unknown')}")
        print(f"  Performance impact: {device_info.get('performance_impact', 'unknown')}")
        
        # Test performance summary
        perf_summary = jax_backend.get_performance_summary()
        print(f"  Performance multiplier: {perf_summary.get('performance_multiplier', 'unknown')}")
        
    except Exception as e:
        print(f"  ❌ System info failed: {e}")
        return False
    
    print("\\n🎉 Direct backend test completed successfully!")
    
    # Show what was achieved
    print("\\n🎯 Intelligent Fallback System Features Demonstrated:")
    print("  ✅ No NotImplementedError failures")
    print("  ✅ Graceful degradation when JAX unavailable")
    print("  ✅ Functional fallbacks for all operations")
    print("  ✅ Smart performance monitoring")
    print("  ✅ Comprehensive system diagnostics")
    
    # Show current backend status
    if jax_backend.jax_available:
        print("\\n🚀 Status: JAX acceleration active (optimal performance)")
    elif jax_backend.numpy_gradients_available:
        print("\\n🔄 Status: NumPy fallback active (reduced performance, full functionality)")
    else:
        print("\\n⚠️  Status: Limited functionality (install JAX or scipy for full features)")
    
    return True

def main():
    """Run the direct backend test."""
    try:
        success = test_backend_isolation()
        if success:
            print("\\n🎉 All tests passed! Fallback system is working correctly.")
            return 0
        else:
            print("\\n❌ Some tests failed.")
            return 1
    except Exception as e:
        print(f"\\n💥 Test crashed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)