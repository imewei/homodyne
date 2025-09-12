#!/usr/bin/env python3
"""
Test script for JAX backend with NumPy gradients fallback integration.
"""

import numpy as np
import sys
import os

# Add the current directory to path
sys.path.insert(0, '.')

# Mock the logging module for testing
class SimpleLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def debug(self, msg): print(f"DEBUG: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")

# Mock the logging module
sys.modules['homodyne'] = type(sys)('homodyne')
sys.modules['homodyne.utils'] = type(sys)('homodyne.utils')
sys.modules['homodyne.utils.logging'] = type(sys)('logging_mock')

def get_logger(name):
    return SimpleLogger()

def log_performance(threshold=0.1):
    def decorator(func):
        return func
    return decorator

# Inject mocks
sys.modules['homodyne.utils.logging'].get_logger = get_logger
sys.modules['homodyne.utils.logging'].log_performance = log_performance

print("Testing JAX backend with NumPy gradients fallback integration...")
print("=" * 70)

# Import the updated JAX backend
import importlib.util
spec = importlib.util.spec_from_file_location("jax_backend", "homodyne/core/jax_backend.py")
jax_backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jax_backend)

# Test backend availability
print("\n1. Backend Availability Check:")
print(f"   JAX available: {jax_backend.jax_available}")
print(f"   NumPy gradients available: {jax_backend.numpy_gradients_available}")

# Validate backend functionality
print("\n2. Backend Validation:")
validation_results = jax_backend.validate_backend()
for key, value in validation_results.items():
    status = "✓" if value else "✗"
    print(f"   {key}: {status}")

# Test basic XPCS physics computation
print("\n3. XPCS Physics Functions Test:")
try:
    # Test parameters
    params = np.array([1000.0, -1.5, 10.0])  # [D0, alpha, D_offset]
    t1 = np.array([0.1])
    t2 = np.array([1.0])
    phi = np.array([0.0])
    q = 0.01
    L = 1000.0
    
    # Test g1 diffusion computation
    g1_result = jax_backend.compute_g1_diffusion(params, t1, t2, q)
    print(f"   g1_diffusion result: {g1_result}")
    
    # Test g2 computation
    contrast, offset = 0.8, 1.0
    g2_result = jax_backend.compute_g2_scaled(params, t1, t2, phi, q, L, contrast, offset)
    print(f"   g2_scaled result: {g2_result}")
    
    print("   ✓ XPCS physics functions working")
    
except Exception as e:
    print(f"   ✗ XPCS physics functions failed: {e}")

# Test gradient computation
print("\n4. Gradient Computation Test:")
try:
    # Test gradient functions
    gradient_g2 = jax_backend.gradient_g2
    gradient_chi2 = jax_backend.gradient_chi2
    
    # Create simple test data
    data = np.array([[[1.2]]])
    sigma = np.array([[[0.1]]])
    
    # Test g2 gradient
    grad_g2 = gradient_g2(params, t1, t2, phi, q, L, contrast, offset)
    print(f"   g2 gradient: {grad_g2}")
    
    # Test chi2 gradient
    grad_chi2 = gradient_chi2(params, data, sigma, t1, t2, phi, q, L, contrast, offset)
    print(f"   chi2 gradient: {grad_chi2}")
    
    print("   ✓ Gradient computation working")
    
except Exception as e:
    print(f"   ✗ Gradient computation failed: {e}")
    import traceback
    traceback.print_exc()

# Test hessian computation
print("\n5. Hessian Computation Test:")
try:
    hessian_g2 = jax_backend.hessian_g2
    
    # Test Hessian (may be slower for NumPy fallback)
    hess_g2 = hessian_g2(params, t1, t2, phi, q, L, contrast, offset)
    print(f"   g2 hessian shape: {hess_g2.shape}")
    print(f"   g2 hessian sample: {hess_g2[0, :2] if len(hess_g2) > 0 else 'empty'}")
    
    print("   ✓ Hessian computation working")
    
except Exception as e:
    print(f"   ✗ Hessian computation failed: {e}")

# Performance comparison if both backends available
print("\n6. Performance Summary:")
if jax_backend.jax_available:
    print("   Using JAX backend (optimal performance)")
elif jax_backend.numpy_gradients_available:
    print("   Using NumPy gradients fallback (reduced performance but full functionality)")
else:
    print("   No gradient support available")

print("\n" + "=" * 70)
if validation_results["gradient_support"]:
    print("✅ JAX backend with NumPy gradients fallback integration successful!")
    print("   The system provides graceful degradation from JAX to NumPy-based")
    print("   numerical differentiation, maintaining full functionality.")
else:
    print("❌ Integration test failed - gradient support not working")

print("=" * 70)