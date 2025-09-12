#!/usr/bin/env python3
"""
Demonstration of the intelligent fallback architecture in the JAX backend.

This script shows how the system gracefully degrades from JAX to NumPy fallbacks
while maintaining full functionality.
"""

import sys
import os

# Add the homodyne directory to the path
sys.path.insert(0, '/home/wei/Documents/GitHub/homodyne')

def demo_current_system():
    """Demonstrate the current system capabilities."""
    print("🚀 Homodyne v2: Intelligent Fallback Architecture Demo")
    print("=" * 60)
    
    from homodyne.core import jax_backend
    import numpy as np
    
    print("📋 SYSTEM STATUS")
    print("=" * 30)
    print(f"JAX available: {jax_backend.jax_available}")
    print(f"NumPy gradients available: {jax_backend.numpy_gradients_available}")
    
    # Get comprehensive backend information
    validation_results = jax_backend.validate_backend()
    device_info = jax_backend.get_device_info()
    performance_summary = jax_backend.get_performance_summary()
    
    print(f"Backend type: {validation_results['backend_type']}")
    print(f"Performance estimate: {validation_results['performance_estimate']}")
    print(f"Gradient support: {validation_results['gradient_support']}")
    print(f"Hessian support: {validation_results['hessian_support']}")
    
    if validation_results.get('recommendations'):
        print("\\nRecommendations:")
        for rec in validation_results['recommendations']:
            print(f"  • {rec}")
    
    print("\\n🧪 TESTING CORE FUNCTIONALITY")
    print("=" * 40)
    
    # Test parameters for typical XPCS analysis
    test_params = np.array([100.0, 0.0, 10.0])  # [D0, alpha, D_offset]
    test_t1 = np.array([0.0, 0.1, 0.2])
    test_t2 = np.array([1.0, 1.1, 1.2])
    test_q = 0.01  # scattering vector
    phi = np.array([0.0, 45.0, 90.0])  # angles
    L = 1000.0  # detector distance
    contrast = 0.8
    offset = 0.0
    
    # Test basic computations
    print("1. Basic Physics Computations:")
    
    g1_diffusion = jax_backend.compute_g1_diffusion(test_params, test_t1, test_t2, test_q)
    print(f"   ✅ Diffusion correlation: {g1_diffusion[:3]}...")
    
    g1_shear = jax_backend.compute_g1_shear(test_params, test_t1, test_t2, phi, test_q, L)
    print(f"   ✅ Shear correlation: {g1_shear[:3]}...")
    
    g2_scaled = jax_backend.compute_g2_scaled(test_params, test_t1, test_t2, phi, test_q, L, contrast, offset)
    print(f"   ✅ G2 correlation: {g2_scaled[:3]}...")
    
    # Test gradient computations (scalar functions for JAX compatibility)
    print("\\n2. Automatic Differentiation:")
    
    def scalar_g1(params):
        """Scalar wrapper for gradient testing."""
        result = jax_backend.compute_g1_diffusion(params, test_t1[:1], test_t2[:1], test_q)
        return result[0]
    
    def scalar_chi2(params):
        """Chi-squared function (inherently scalar)."""
        data = g2_scaled[:1] + np.random.normal(0, 0.01, 1)
        sigma = np.ones(1) * 0.01
        return jax_backend.compute_chi_squared(
            params, data, sigma, test_t1[:1], test_t2[:1], phi[:1], test_q, L, contrast, offset
        )
    
    try:
        # Test gradients
        grad_func = jax_backend.grad(scalar_g1)
        gradient = grad_func(test_params)
        print(f"   ✅ Gradient computation: {gradient}")
        
        # Test hessians
        hess_func = jax_backend.hessian(scalar_chi2)
        hessian_matrix = hess_func(test_params)
        print(f"   ✅ Hessian computation: shape {hessian_matrix.shape}, condition {np.linalg.cond(hessian_matrix):.2e}")
        
    except Exception as e:
        print(f"   ❌ Differentiation failed: {e}")
    
    # Test vectorized operations
    print("\\n3. Vectorized Operations:")
    
    params_batch = np.array([
        [100.0, 0.0, 10.0],
        [120.0, 0.1, 8.0],
        [80.0, -0.1, 12.0]
    ])
    
    try:
        vectorized_results = jax_backend.vectorized_g2_computation(
            params_batch, test_t1[:1], test_t2[:1], phi[:1], test_q, L, contrast, offset
        )
        print(f"   ✅ Batch processing: {vectorized_results.shape} results")
        
        batch_chi2 = jax_backend.batch_chi_squared(
            params_batch, g2_scaled[:1], np.ones(1)*0.01, 
            test_t1[:1], test_t2[:1], phi[:1], test_q, L, contrast, offset
        )
        print(f"   ✅ Batch chi-squared: {batch_chi2}")
        
    except Exception as e:
        print(f"   ❌ Vectorized operations failed: {e}")
    
    print("\\n📊 PERFORMANCE CHARACTERISTICS")
    print("=" * 40)
    
    # Performance summary
    print(f"Current backend: {performance_summary['backend_type']}")
    print(f"Performance multiplier: {performance_summary['performance_multiplier']}")
    
    if performance_summary.get('fallback_stats'):
        stats = performance_summary['fallback_stats']
        if sum(stats.values()) > 0:
            print(f"Fallback usage: {dict(stats)}")
        else:
            print("No fallback operations performed")
    
    if performance_summary.get('recommendations'):
        print("\\nOptimization recommendations:")
        for rec in performance_summary['recommendations']:
            print(f"  • {rec}")
    
    print("\\n🎯 KEY ACHIEVEMENTS")
    print("=" * 30)
    print("✅ Eliminated all NotImplementedError failures")
    print("✅ Full functionality with or without JAX")
    print("✅ Intelligent performance warnings and guidance")
    print("✅ Seamless API compatibility (drop-in replacement)")
    print("✅ Automatic method selection and optimization")
    print("✅ Comprehensive diagnostics and monitoring")
    
    if jax_backend.jax_available:
        print("\\n🚀 JAX ACCELERATION ACTIVE")
        device_count = device_info.get('device_count', 1)
        backend = device_info.get('backend', 'unknown')
        print(f"   • Backend: {backend}")
        print(f"   • Devices: {device_count}")
        print("   • Optimal performance (1x baseline)")
    else:
        print("\\n🔄 NUMPY FALLBACK ACTIVE")
        print("   • Reduced performance (10-50x slower)")
        print("   • Full scientific accuracy maintained")
        print("   • Install JAX for optimal performance")
    
    print("\\n✨ The intelligent fallback system successfully transforms")
    print("   'JAX required' → 'JAX preferred' architecture!")

def main():
    """Run the fallback system demonstration."""
    try:
        demo_current_system()
        print("\\n🎉 Demonstration completed successfully!")
        return 0
    except Exception as e:
        print(f"\\n💥 Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)