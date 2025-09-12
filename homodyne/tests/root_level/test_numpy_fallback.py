#!/usr/bin/env python3
"""
Test script that specifically tests the NumPy fallback system by 
temporarily making JAX unavailable.
"""

import sys
import os
import importlib
from unittest.mock import patch

# Add the homodyne directory to the path
sys.path.insert(0, '/home/wei/Documents/GitHub/homodyne')

def test_fallback_with_mock():
    """Test fallback system by mocking JAX as unavailable."""
    print("🧪 Testing NumPy Fallback System (JAX Mocked as Unavailable)")
    print("=" * 70)
    
    # Mock JAX import failure
    with patch.dict('sys.modules', {'jax': None, 'jax.numpy': None}):
        with patch('builtins.__import__', side_effect=lambda name, *args: 
                  ImportError(f"No module named '{name}'") if name.startswith('jax') else __import__(name, *args)):
            
            # Force reload the module to trigger fallback
            if 'homodyne.core.jax_backend' in sys.modules:
                del sys.modules['homodyne.core.jax_backend']
            if 'homodyne.core' in sys.modules:
                del sys.modules['homodyne.core']
            if 'homodyne' in sys.modules:
                del sys.modules['homodyne']
            
            try:
                from homodyne.core import jax_backend
                
                print(f"✅ Successfully imported with JAX mocked unavailable")
                print(f"JAX available: {jax_backend.jax_available}")
                print(f"NumPy gradients available: {jax_backend.numpy_gradients_available}")
                
                # Test backend validation
                print("\n🧪 Testing backend validation...")
                results = jax_backend.validate_backend()
                print(f"Backend type: {results['backend_type']}")
                print(f"Performance estimate: {results['performance_estimate']}")
                print(f"Gradient support: {results['gradient_support']}")
                print(f"Hessian support: {results['hessian_support']}")
                
                # Test basic computations
                print("\n🧪 Testing basic computations...")
                import numpy as np
                test_params = np.array([100.0, 0.0, 10.0])
                test_t1 = np.array([0.0])
                test_t2 = np.array([1.0])
                test_q = 0.01
                
                result = jax_backend.compute_g1_diffusion(test_params, test_t1, test_t2, test_q)
                print(f"✅ compute_g1_diffusion: {result}")
                
                # Test gradient computation
                if jax_backend.numpy_gradients_available:
                    print("\n🧪 Testing gradient computation with NumPy fallback...")
                    
                    def test_func(params):
                        return jax_backend.compute_g1_diffusion(params, test_t1, test_t2, test_q)[0]
                    
                    try:
                        grad_func = jax_backend.grad(test_func)
                        grad_result = grad_func(test_params)
                        print(f"✅ Gradient computation successful: {grad_result}")
                    except Exception as e:
                        print(f"❌ Gradient computation failed: {e}")
                else:
                    print("\n❌ NumPy gradients not available - gradient computation disabled")
                
                # Test performance warnings
                print("\n🧪 Testing performance warnings...")
                device_info = jax_backend.get_device_info()
                print(f"Backend: {device_info['backend']}")
                print(f"Performance impact: {device_info['performance_impact']}")
                if device_info.get('recommendations'):
                    for rec in device_info['recommendations']:
                        print(f"  - {rec}")
                
                return True
                
            except Exception as e:
                print(f"❌ Failed to test fallback system: {e}")
                import traceback
                traceback.print_exc()
                return False

def test_actual_fallback():
    """Test with actual system state (without mocking)."""
    print("\n🧪 Testing Actual System State")
    print("=" * 50)
    
    try:
        # Clean import
        if 'homodyne.core.jax_backend' in sys.modules:
            del sys.modules['homodyne.core.jax_backend']
        
        from homodyne.core import jax_backend
        
        print(f"JAX available: {jax_backend.jax_available}")
        print(f"NumPy gradients available: {jax_backend.numpy_gradients_available}")
        
        # Test that the system works
        import numpy as np
        test_params = np.array([100.0, 0.0, 10.0])
        test_t1 = np.array([0.0])
        test_t2 = np.array([1.0])
        test_q = 0.01
        phi = np.array([0.0])
        
        # Test forward computation
        result = jax_backend.compute_g1_diffusion(test_params, test_t1, test_t2, test_q)
        print(f"✅ Forward computation works: {result}")
        
        # Test gradient with scalar function
        def scalar_func(params):
            return jax_backend.compute_g1_diffusion(params, test_t1, test_t2, test_q)[0]
        
        try:
            grad_func = jax_backend.grad(scalar_func)
            grad_result = grad_func(test_params)
            print(f"✅ Gradient computation works: {grad_result}")
        except Exception as e:
            print(f"❌ Gradient computation failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Actual system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run fallback system tests."""
    print("🚀 Testing Intelligent Fallback Architecture")
    print("=" * 80)
    
    tests_passed = 0
    total_tests = 2
    
    # Test 1: Mock JAX unavailable
    if test_fallback_with_mock():
        tests_passed += 1
        print("✅ Mock fallback test: PASSED")
    else:
        print("❌ Mock fallback test: FAILED")
    
    # Test 2: Actual system state
    if test_actual_fallback():
        tests_passed += 1
        print("✅ Actual system test: PASSED")
    else:
        print("❌ Actual system test: FAILED")
    
    # Summary
    print("\n" + "=" * 80)
    print(f"📊 SUMMARY: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All fallback tests passed!")
        return 0
    else:
        print("⚠️  Some fallback tests failed.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)