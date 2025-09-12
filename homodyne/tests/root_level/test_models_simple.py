#!/usr/bin/env python3
"""
Simple test script for enhanced model classes.
Tests core functionality without full package dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

# Direct imports to avoid full package initialization
import numpy as np
from homodyne.core.jax_backend import jax_available, numpy_gradients_available, validate_backend
from homodyne.core.models import CombinedModel

def test_enhanced_models():
    """Test enhanced model functionality."""
    print("=" * 80)
    print("Testing Enhanced Model Classes - Core Functionality")
    print("=" * 80)
    
    print(f"JAX available: {jax_available}")
    print(f"NumPy gradients available: {numpy_gradients_available}")
    
    # Test backend validation
    print("\n--- Backend Validation ---")
    backend_info = validate_backend()
    print(f"Backend type: {backend_info['backend_type']}")
    print(f"Gradient support: {backend_info['gradient_support']}")
    print(f"Hessian support: {backend_info['hessian_support']}")
    print(f"Performance estimate: {backend_info['performance_estimate']}")
    
    # Test model creation and capabilities
    print("\n--- Model Creation and Capabilities ---")
    for mode in ["static_isotropic", "laminar_flow"]:
        print(f"\nTesting {mode} mode:")
        model = CombinedModel(analysis_mode=mode)
        
        print(f"  Model name: {model.name}")
        print(f"  Parameters: {model.n_params} ({model.parameter_names})")
        print(f"  Supports gradients: {model.supports_gradients()}")
        print(f"  Best gradient method: {model.get_best_gradient_method()}")
        
        # Test gradient capabilities
        try:
            capabilities = model.get_gradient_capabilities()
            print(f"  Backend summary: {capabilities['backend_summary']}")
            print(f"  Best method: {capabilities['best_method']}")
            if capabilities['performance_warning']:
                print(f"  Warning: {capabilities['performance_warning']}")
        except Exception as e:
            print(f"  ❌ Capabilities test failed: {e}")
        
        # Test optimization recommendations
        try:
            recommendations = model.get_optimization_recommendations()
            print(f"  Recommendations: {len(recommendations)} items")
            for rec in recommendations[:2]:  # Show first 2
                print(f"    {rec}")
            if len(recommendations) > 2:
                print(f"    ... and {len(recommendations)-2} more")
        except Exception as e:
            print(f"  ❌ Recommendations test failed: {e}")
    
    # Test gradient function retrieval
    print("\n--- Gradient Function Retrieval ---")
    model = CombinedModel(analysis_mode="static_isotropic")
    
    try:
        grad_func = model.get_gradient_function()
        print("✅ Gradient function retrieved successfully")
        print(f"   Type: {type(grad_func)}")
        
        hess_func = model.get_hessian_function()
        print("✅ Hessian function retrieved successfully") 
        print(f"   Type: {type(hess_func)}")
        
    except ImportError as e:
        print("❌ Functions not available (expected if no backends):")
        print(f"   {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test enhanced model info
    print("\n--- Enhanced Model Information ---")
    try:
        info = model.get_model_info()
        print("Model info keys:", list(info.keys()))
        print(f"Gradient method: {info.get('gradient_method', 'N/A')}")
        print(f"Backend summary: {info.get('backend_summary', 'N/A')}")
        print(f"Optimization recommendations: {len(info.get('optimization_recommendations', []))}")
    except Exception as e:
        print(f"❌ Model info test failed: {e}")
    
    print("\n" + "=" * 80)
    print("Enhanced Model Testing Complete")
    print("=" * 80)

if __name__ == "__main__":
    test_enhanced_models()