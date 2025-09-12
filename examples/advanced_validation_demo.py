#!/usr/bin/env python3
"""
Advanced Validation System Demonstration
========================================

This script demonstrates the enhanced validation capabilities of the Homodyne v2
configuration system, including:

1. Multi-method workflow validation (VI → MCMC → Hybrid)
2. Physics constraint validation for extreme parameters
3. HPC-specific configuration validation (PBS, SLURM)
4. GPU memory validation with hardware detection
5. Advanced scenario validation (large datasets, batch processing)
6. Complex phi angle filtering validation

Run this script to see comprehensive validation in action.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add homodyne to path if running from examples directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from homodyne.config import (
    ParameterValidator,
    ModeResolver,
    HPCValidator,
    GPUValidator,
    AdvancedScenarioValidator,
    ConfigManager
)
from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


def demonstrate_parameter_validation():
    """Demonstrate enhanced parameter validation."""
    print("\n" + "="*60)
    print("1. PARAMETER VALIDATION DEMONSTRATION")
    print("="*60)
    
    validator = ParameterValidator()
    
    # Test configuration with various validation scenarios
    test_config = {
        'analysis_mode': 'laminar_flow',
        'optimization': {
            'vi': {
                'n_iterations': 3000,
                'learning_rate': 0.01,
                'convergence_tol': 1e-6
            },
            'mcmc': {
                'n_samples': 2000,
                'n_warmup': 1500,
                'n_chains': 4,
                'target_accept_prob': 0.8
            },
            'hybrid': {
                'use_vi_init': True,
                'convergence_threshold': 0.05
            }
        },
        'hardware': {
            'gpu_memory_fraction': 0.85,
            'force_cpu': False
        },
        'data': {
            'file_path': '/nonexistent/path/data.hdf',
            'custom_phi_angles': [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        }
    }
    
    print("Validating comprehensive configuration...")
    result = validator.validate_config(test_config)
    
    print(f"\nValidation Result: {'PASSED' if result.is_valid else 'FAILED'}")
    print(f"Hardware Info Available: {bool(result.hardware_info)}")
    
    if result.errors:
        print(f"\n❌ Errors ({len(result.errors)}):")
        for error in result.errors:
            print(f"   • {error}")
    
    if result.warnings:
        print(f"\n⚠️  Warnings ({len(result.warnings)}):")
        for warning in result.warnings:
            print(f"   • {warning}")
    
    if result.suggestions:
        print(f"\n💡 Suggestions ({len(result.suggestions)}):")
        for suggestion in result.suggestions:
            print(f"   • {suggestion}")
    
    if result.info:
        print(f"\nℹ️  Information ({len(result.info)}):")
        for info in result.info:
            print(f"   • {info}")
    
    # Test physics validation with extreme parameters
    print("\n" + "-"*50)
    print("Testing Physics Constraint Validation")
    print("-"*50)
    
    extreme_params = {
        'D0': 1e-15,  # Very small
        'alpha': 5.0,  # Outside typical range
        'gamma_dot_0': 1e6,  # Very large
        'contrast': -0.1,  # Invalid (negative)
        'offset': 5.0  # Very large
    }
    
    physics_result = validator.validate_analysis_parameters('laminar_flow', extreme_params)
    print(f"Physics Validation: {'PASSED' if physics_result.is_valid else 'FAILED'}")
    
    for error in physics_result.errors:
        print(f"   ❌ {error}")
    for warning in physics_result.warnings:
        print(f"   ⚠️  {warning}")


def demonstrate_mode_resolution():
    """Demonstrate enhanced mode resolution and compatibility analysis."""
    print("\n" + "="*60)
    print("2. MODE RESOLUTION & COMPATIBILITY ANALYSIS")
    print("="*60)
    
    resolver = ModeResolver()
    
    # Test data scenarios
    test_scenarios = [
        {
            'name': 'Single Angle Dataset',
            'data': {
                'phi_angles': [0.0],
                'c2_exp': [1.0] * 10000  # 10K points
            },
            'config': {}
        },
        {
            'name': 'Multi-Angle Small Dataset',
            'data': {
                'phi_angles': [0, 45, 90, 135],
                'c2_exp': [1.0] * 50000  # 50K points
            },
            'config': {'optimization_config': {'mcmc_sampling': {'enabled': True}}}
        },
        {
            'name': 'Wide-Angle Large Dataset',
            'data': {
                'phi_angles': [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
                'c2_exp': [1.0] * 10000000  # 10M points
            },
            'config': {'optimization_config': {'mcmc_sampling': {'draws': 5000}}}
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
        print("-" * len(f"📊 Scenario: {scenario['name']}"))
        
        # Get mode suggestions
        suggestions = resolver.suggest_mode_for_data(scenario['data'], scenario['config'])
        
        print(f"Primary Suggestion: {suggestions['primary_suggestion']}")
        print(f"Confidence: {suggestions['confidence']}")
        
        if suggestions['alternatives']:
            print(f"Alternatives: {', '.join(suggestions['alternatives'])}")
        
        print("Reasoning:")
        for reason in suggestions['reasoning']:
            print(f"   • {reason}")
        
        if suggestions.get('performance_notes'):
            print("Performance Notes:")
            for note in suggestions['performance_notes']:
                print(f"   📈 {note}")
        
        if suggestions.get('warnings'):
            print("Warnings:")
            for warning in suggestions['warnings']:
                print(f"   ⚠️  {warning}")
        
        # Test mode transition feasibility
        if scenario['name'] == 'Wide-Angle Large Dataset':
            print("\n🔄 Mode Transition Analysis:")
            transition = resolver.analyze_mode_transition_feasibility(
                'static_anisotropic', 'laminar_flow', 
                scenario['data'], scenario['config']
            )
            
            print(f"   Feasible: {'Yes' if transition['feasible'] else 'No'}")
            print(f"   Confidence: {transition['confidence']:.2f}")
            print(f"   Performance Impact: {transition['performance_impact']}")
            
            if transition['recommended_changes']:
                print("   Recommended Changes:")
                for change in transition['recommended_changes']:
                    print(f"     • {change}")


def demonstrate_hpc_validation():
    """Demonstrate HPC configuration validation."""
    print("\n" + "="*60)
    print("3. HPC CONFIGURATION VALIDATION")
    print("="*60)
    
    hpc_validator = HPCValidator()
    
    # Test HPC configurations
    test_configs = [
        {
            'name': 'PBS Professional Setup',
            'config': {
                'pbs': {
                    'nodes': 2,
                    'ppn': 24,
                    'mem': '96gb',
                    'walltime': '24:00:00',
                    'queue': 'normal'
                }
            }
        },
        {
            'name': 'SLURM Configuration',
            'config': {
                'slurm': {
                    'ntasks': 48,
                    'cpus-per-task': 1,
                    'mem-per-cpu': '4G',
                    'time': '12:00:00',
                    'partition': 'compute'
                }
            }
        },
        {
            'name': 'Overprovisioned Resources',
            'config': {
                'pbs': {
                    'nodes': 10,
                    'ppn': 48,
                    'mem': '2tb',
                    'walltime': '168:00:00',  # 1 week
                    'queue': 'debug'  # Debug queue with large resources
                }
            }
        }
    ]
    
    for test in test_configs:
        print(f"\n🖥️  Testing: {test['name']}")
        print("-" * len(f"🖥️  Testing: {test['name']}"))
        
        result = hpc_validator.validate_hpc_configuration(test['config'])
        
        print(f"Valid: {'Yes' if result.is_valid else 'No'}")
        print(f"Scheduler: {result.scheduler_type or 'Not detected'}")
        print(f"Resource Efficiency: {result.resource_efficiency:.2f}")
        
        if result.estimated_walltime:
            print(f"Estimated Walltime: {result.estimated_walltime}")
        
        if result.warnings:
            print("Warnings:")
            for warning in result.warnings:
                print(f"   ⚠️  {warning}")
        
        if result.recommendations:
            print("Recommendations:")
            for rec in result.recommendations:
                print(f"   💡 {rec}")


def demonstrate_gpu_validation():
    """Demonstrate GPU configuration validation."""
    print("\n" + "="*60)
    print("4. GPU HARDWARE & CONFIGURATION VALIDATION")
    print("="*60)
    
    gpu_validator = GPUValidator()
    
    # Test GPU configurations
    test_configs = [
        {
            'name': 'Conservative GPU Settings',
            'hardware_config': {
                'gpu_memory_fraction': 0.6,
                'force_cpu': False
            },
            'optimization_config': {
                'mcmc_sampling': {
                    'backend_specific': {
                        'gpu_backend': {
                            'device_memory_fraction': 0.6
                        }
                    }
                }
            }
        },
        {
            'name': 'Aggressive GPU Settings',
            'hardware_config': {
                'gpu_memory_fraction': 0.95,
                'force_cpu': False
            },
            'optimization_config': {
                'mcmc_sampling': {
                    'backend_specific': {
                        'gpu_backend': {
                            'device_memory_fraction': 0.9
                        }
                    }
                }
            }
        },
        {
            'name': 'CPU Fallback Mode',
            'hardware_config': {
                'force_cpu': True,
                'gpu_memory_fraction': 0.8  # Should be ignored
            },
            'optimization_config': {}
        }
    ]
    
    for test in test_configs:
        print(f"\n🎮 Testing: {test['name']}")
        print("-" * len(f"🎮 Testing: {test['name']}"))
        
        result = gpu_validator.validate_gpu_configuration(
            test['hardware_config'], 
            test['optimization_config']
        )
        
        print(f"GPU Available: {'Yes' if result.gpu_available else 'No'}")
        print(f"GPU Compatible: {'Yes' if result.gpu_compatible else 'No'}")
        print(f"Total Memory: {result.total_memory_gb:.1f} GB")
        print(f"Recommended Memory Fraction: {result.recommended_memory_fraction:.2f}")
        
        if result.warnings:
            print("Warnings:")
            for warning in result.warnings:
                print(f"   ⚠️  {warning}")
        
        if result.optimization_suggestions:
            print("Optimization Suggestions:")
            for suggestion in result.optimization_suggestions:
                print(f"   💡 {suggestion}")


def demonstrate_advanced_scenarios():
    """Demonstrate advanced scenario validation."""
    print("\n" + "="*60)
    print("5. ADVANCED SCENARIO VALIDATION")
    print("="*60)
    
    scenario_validator = AdvancedScenarioValidator()
    
    # Test large dataset scenario
    print("\n📊 Large Dataset Validation")
    print("-" * 30)
    
    large_dataset_config = {
        'analysis_mode': 'laminar_flow',
        'optimization_config': {
            'mcmc_sampling': {
                'enabled': True,
                'draws': 5000
            }
        },
        'performance_settings': {
            'caching': {'enable_disk_cache': False},
            'numba_optimization': {'enable_numba': False}
        }
    }
    
    large_result = scenario_validator.validate_large_dataset_scenario(
        large_dataset_config, 
        estimated_data_size=150_000_000  # 150M points
    )
    
    print(f"Suitable for Large Dataset: {'Yes' if large_result['is_suitable'] else 'No'}")
    if large_result['estimated_runtime_hours']:
        print(f"Estimated Runtime: {large_result['estimated_runtime_hours']:.1f} hours")
    
    if large_result['memory_warnings']:
        print("Memory Warnings:")
        for warning in large_result['memory_warnings']:
            print(f"   ⚠️  {warning}")
    
    if large_result['optimization_recommendations']:
        print("Optimization Recommendations:")
        for rec in large_result['optimization_recommendations']:
            print(f"   💡 {rec}")
    
    # Test batch processing scenario
    print("\n🔄 Batch Processing Validation")
    print("-" * 32)
    
    batch_config = {
        'batch_size': 8,
        'parallel_jobs': 3,
        'output_directory': './test_batch_results',
        'estimated_output_size_gb': 0.2
    }
    
    batch_result = scenario_validator.validate_batch_processing_scenario(batch_config)
    
    print(f"Batch Configuration Valid: {'Yes' if batch_result['is_valid'] else 'No'}")
    
    if batch_result['resource_requirements']:
        req = batch_result['resource_requirements']
        print(f"Storage Required: {req['estimated_storage_gb']:.1f} GB")
        print(f"Recommended Cores: {req['recommended_cores']}")
    
    if batch_result['warnings']:
        print("Warnings:")
        for warning in batch_result['warnings']:
            print(f"   ⚠️  {warning}")
    
    # Test complex phi angle filtering
    print("\n📐 Complex Phi Angle Filtering Validation")
    print("-" * 42)
    
    angle_config = {
        'enabled': True,
        'target_ranges': [
            {'min_angle': 0, 'max_angle': 15},
            {'min_angle': 45, 'max_angle': 75},
            {'min_angle': 105, 'max_angle': 135},
            {'min_angle': 165, 'max_angle': 195},
            {'min_angle': 225, 'max_angle': 255},
            {'min_angle': 285, 'max_angle': 315}
        ]
    }
    
    angle_result = scenario_validator.validate_complex_phi_filtering(angle_config)
    
    print(f"Angle Configuration Valid: {'Yes' if angle_result['is_valid'] else 'No'}")
    
    coverage = angle_result['coverage_analysis']
    print(f"Total Coverage: {coverage['total_coverage_degrees']:.1f}°")
    print(f"Valid Ranges: {coverage['valid_ranges']}/{coverage['total_ranges']}")
    
    if coverage['overlapping_ranges']:
        print(f"Overlapping Ranges: {len(coverage['overlapping_ranges'])}")
    
    if angle_result['warnings']:
        print("Warnings:")
        for warning in angle_result['warnings']:
            print(f"   ⚠️  {warning}")
    
    if angle_result['recommendations']:
        print("Recommendations:")
        for rec in angle_result['recommendations']:
            print(f"   💡 {rec}")


def main():
    """Run comprehensive validation demonstration."""
    print("HOMODYNE v2 ADVANCED VALIDATION SYSTEM DEMONSTRATION")
    print("=" * 60)
    print("This demonstration shows enhanced validation capabilities")
    print("for production-ready scientific computing environments.")
    
    try:
        # Run all demonstrations
        demonstrate_parameter_validation()
        demonstrate_mode_resolution()
        demonstrate_hpc_validation()
        demonstrate_gpu_validation()
        demonstrate_advanced_scenarios()
        
        print("\n" + "="*60)
        print("✅ VALIDATION DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nKey Validation Features Demonstrated:")
        print("• Multi-method workflow compatibility checking")
        print("• Enhanced physics constraint validation")
        print("• HPC resource allocation validation")
        print("• GPU hardware detection and memory validation")
        print("• Large dataset processing feasibility analysis")
        print("• Complex phi angle filtering validation")
        print("• Intelligent mode-data compatibility assessment")
        print("\nThe enhanced validation system makes Homodyne v2 production-ready")
        print("for complex scientific computing environments with comprehensive")
        print("error detection, performance optimization, and resource validation.")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())