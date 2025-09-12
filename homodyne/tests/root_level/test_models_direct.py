#!/usr/bin/env python3
"""
Direct test script for enhanced model classes.
Tests core functionality by importing modules directly.
"""

import sys
import os

# Add homodyne directory to path for direct imports
homodyne_dir = os.path.join(os.path.dirname(__file__), 'homodyne')
sys.path.insert(0, homodyne_dir)

# Import logging first to set up basic infrastructure
try:
    from utils.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    # Fallback logging if utils not available
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Direct imports of core modules
try:
    from core.jax_backend import (
        jax_available, numpy_gradients_available, 
        validate_backend, get_device_info, get_performance_summary
    )
    from core.models import CombinedModel
    from core.physics import PhysicsConstants
    
    print("✅ Successfully imported core modules")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_backend_capabilities():
    """Test backend capability detection."""
    print("\n" + "=" * 60)
    print("Backend Capabilities Test")
    print("=" * 60)
    
    print(f"JAX available: {jax_available}")
    print(f"NumPy gradients available: {numpy_gradients_available}")
    
    try:
        backend_info = validate_backend()
        print(f"Backend type: {backend_info['backend_type']}")
        print(f"Gradient support: {backend_info['gradient_support']}")
        print(f"Hessian support: {backend_info['hessian_support']}")
        print(f"Performance estimate: {backend_info['performance_estimate']}")
        
        device_info = get_device_info()
        print(f"Device info available: {device_info['available']}")
        if device_info['available']:
            print(f"Backend: {device_info['backend']}")
            print(f"Devices: {device_info.get('devices', ['unknown'])}")
        
        performance_info = get_performance_summary()
        print(f"Performance multiplier: {performance_info['performance_multiplier']}")
        
    except Exception as e:
        print(f"❌ Backend test failed: {e}")

def test_model_creation():
    """Test enhanced model creation and basic functionality."""
    print("\n" + "=" * 60)
    print("Model Creation Test")
    print("=" * 60)
    
    analysis_modes = ["static_isotropic", "static_anisotropic", "laminar_flow"]
    
    for mode in analysis_modes:
        print(f"\n--- {mode.upper()} MODE ---")
        
        try:
            model = CombinedModel(analysis_mode=mode)
            print(f"✅ Model created: {model.name}")
            print(f"   Parameters: {model.n_params} ({model.parameter_names})")
            print(f"   Supports gradients: {model.supports_gradients()}")
            
            # Test new enhanced methods
            best_method = model.get_best_gradient_method()
            print(f"   Best gradient method: {best_method}")
            
        except Exception as e:
            print(f"❌ Model creation failed for {mode}: {e}")

def test_gradient_capabilities():
    """Test gradient capability introspection."""
    print("\n" + "=" * 60)
    print("Gradient Capabilities Test")
    print("=" * 60)
    
    try:
        model = CombinedModel(analysis_mode="laminar_flow")
        capabilities = model.get_gradient_capabilities()
        
        print("Capability Summary:")
        print(f"  Gradient available: {capabilities['gradient_available']}")
        print(f"  Hessian available: {capabilities['hessian_available']}")
        print(f"  Best method: {capabilities['best_method']}")
        print(f"  Backend type: {capabilities['backend_type']}")
        print(f"  Performance estimate: {capabilities['performance_estimate']}")
        
        if capabilities['performance_warning']:
            print(f"  ⚠️  {capabilities['performance_warning']}")
        
        print(f"  Backend summary: {capabilities['backend_summary']}")
        
        print("\nRecommendations:")
        for i, rec in enumerate(capabilities['recommendations'][:3], 1):
            print(f"  {i}. {rec}")
            
    except Exception as e:
        print(f"❌ Capabilities test failed: {e}")

def test_gradient_function_retrieval():
    """Test gradient function retrieval with enhanced error handling."""
    print("\n" + "=" * 60)
    print("Gradient Function Retrieval Test")
    print("=" * 60)
    
    model = CombinedModel(analysis_mode="static_isotropic")
    
    # Test gradient function
    print("Testing gradient function retrieval...")
    try:
        grad_func = model.get_gradient_function()
        print("✅ Gradient function retrieved successfully")
        print(f"   Function type: {type(grad_func)}")
        print(f"   Function name: {getattr(grad_func, '__name__', 'unnamed')}")
        
    except ImportError as e:
        print("❌ Gradient function not available (expected if no backends):")
        print(f"   {e}")
        
    except Exception as e:
        print(f"❌ Unexpected error in gradient retrieval: {e}")
    
    # Test hessian function
    print("\nTesting Hessian function retrieval...")
    try:
        hess_func = model.get_hessian_function()
        print("✅ Hessian function retrieved successfully")
        print(f"   Function type: {type(hess_func)}")
        print(f"   Function name: {getattr(hess_func, '__name__', 'unnamed')}")
        
    except ImportError as e:
        print("❌ Hessian function not available (expected if no backends):")
        print(f"   {e}")
        
    except Exception as e:
        print(f"❌ Unexpected error in Hessian retrieval: {e}")

def test_optimization_recommendations():
    """Test optimization recommendations."""
    print("\n" + "=" * 60)
    print("Optimization Recommendations Test")
    print("=" * 60)
    
    modes = ["static_isotropic", "laminar_flow"]
    
    for mode in modes:
        print(f"\n--- {mode.upper()} MODE ---")
        
        try:
            model = CombinedModel(analysis_mode=mode)
            recommendations = model.get_optimization_recommendations()
            
            print(f"Found {len(recommendations)} recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
                
        except Exception as e:
            print(f"❌ Recommendations test failed for {mode}: {e}")

def test_enhanced_model_info():
    """Test enhanced model information."""
    print("\n" + "=" * 60)
    print("Enhanced Model Information Test")
    print("=" * 60)
    
    try:
        model = CombinedModel(analysis_mode="laminar_flow")
        info = model.get_model_info()
        
        print("Enhanced Model Info Keys:")
        for key in sorted(info.keys()):
            if key in ['parameter_bounds', 'default_parameters', 'gradient_capabilities', 'device_info']:
                print(f"  {key}: <complex data>")
            else:
                print(f"  {key}: {info[key]}")
                
        print(f"\nKey capabilities:")
        print(f"  Supports gradients: {info['supports_gradients']}")
        print(f"  Gradient method: {info['gradient_method']}")
        print(f"  Backend summary: {info['backend_summary']}")
        
    except Exception as e:
        print(f"❌ Enhanced model info test failed: {e}")

def main():
    """Run all tests."""
    print("=" * 80)
    print("Enhanced Model Classes - Direct Testing")
    print("=" * 80)
    
    test_backend_capabilities()
    test_model_creation()
    test_gradient_capabilities()
    test_gradient_function_retrieval()
    test_optimization_recommendations()
    test_enhanced_model_info()
    
    print("\n" + "=" * 80)
    print("Direct Testing Complete")
    print("=" * 80)

if __name__ == "__main__":
    main()