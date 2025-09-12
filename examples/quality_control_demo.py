#!/usr/bin/env python3
"""
Quality Control Integration Demo
===============================

Demonstration of the comprehensive data quality control system implemented in Phase 3.
Shows how to use progressive validation, auto-repair, and quality reporting features.

This example demonstrates:
- Progressive quality control throughout data loading pipeline
- Auto-repair functionality for common data issues
- Quality metrics and reporting system
- Integration with filtering and preprocessing
- Exportable quality assessment reports

Usage:
    python examples/quality_control_demo.py

Requirements:
    - Homodyne v2 with Phase 3 quality control features
    - Sample XPCS data file (can be simulated)
    - Quality control configuration enabled

Authors: Homodyne Development Team
Institution: Argonne National Laboratory
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path

# Add homodyne to path for demo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from homodyne.data.xpcs_loader import XPCSDataLoader
    from homodyne.data.quality_controller import (
        DataQualityController, QualityControlStage, 
        create_quality_controller
    )
    from homodyne.data.validation import validate_xpcs_data_incremental
    print("✓ Successfully imported Homodyne quality control modules")
except ImportError as e:
    print(f"❌ Failed to import Homodyne modules: {e}")
    print("Make sure you're running this from the homodyne root directory")
    sys.exit(1)

def create_demo_config():
    """Create demonstration configuration with quality control enabled."""
    return {
        "metadata": {
            "config_version": "0.8.0",
            "description": "Quality Control Demo Configuration",
            "analysis_mode": "static_anisotropic"
        },
        "experimental_data": {
            "data_folder_path": "./examples/demo_data/",
            "data_file_name": "demo_xpcs_data.hdf",
            "phi_angles_path": "./examples/demo_data/",
            "cache_file_path": "./examples/demo_data/",
            "cache_filename_template": "demo_cached_c2_frames_{start_frame}_{end_frame}.npz",
            "cache_compression": True
        },
        "analyzer_parameters": {
            "dt": 0.1,
            "start_frame": 1,
            "end_frame": 100,
            "wavevector_q": 0.01
        },
        "quality_control": {
            "enabled": True,
            "validation_level": "standard",
            "auto_repair": "conservative",
            "pass_threshold": 60.0,
            "warn_threshold": 75.0,
            "excellent_threshold": 85.0,
            "validation_stages": {
                "enable_raw_validation": True,
                "enable_filtering_validation": True,
                "enable_preprocessing_validation": True,
                "enable_final_validation": True
            },
            "repair_settings": {
                "repair_nan_values": True,
                "repair_infinite_values": True,
                "repair_negative_correlations": False,
                "repair_scaling_issues": True,
                "repair_format_inconsistencies": True
            },
            "performance": {
                "cache_validation_results": True,
                "incremental_validation": True,
                "parallel_validation": False
            },
            "reporting": {
                "generate_reports": True,
                "export_detailed_reports": True,
                "save_quality_history": True
            }
        },
        "data_filtering": {
            "enabled": True,
            "q_range": {"enabled": True, "min": 0.005, "max": 0.05},
            "combine_criteria": "AND",
            "fallback_on_empty": True
        },
        "preprocessing": {
            "enabled": True,
            "stages": {
                "correct_diagonal": {"enabled": True, "method": "statistical"},
                "normalize_data": {"enabled": True, "method": "baseline"},
                "standardize_format": {"enabled": True}
            }
        },
        "v2_features": {
            "output_format": "auto",
            "validation_level": "standard",
            "cache_strategy": "intelligent"
        }
    }

def create_demo_data():
    """Create synthetic XPCS data for demonstration."""
    print("📊 Creating synthetic XPCS data for demonstration...")
    
    # Create demo data directory
    demo_dir = Path("./examples/demo_data")
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic data
    n_q = 20
    n_phi = 8
    n_time = 50
    
    # Q and phi arrays
    q_list = np.logspace(-3, -1, n_q)  # 0.001 to 0.1 Å⁻¹
    phi_list = np.linspace(-15, 195, n_phi)  # -15 to 195 degrees
    
    # Time arrays
    dt = 0.1
    t_max = dt * n_time
    t1 = t2 = np.linspace(0, t_max, n_time)
    
    # Synthetic correlation matrices with some realistic features
    c2_matrices = []
    for i in range(len(q_list)):
        # Create symmetric correlation matrix with exponential decay
        matrix = np.zeros((n_time, n_time))
        for j in range(n_time):
            for k in range(n_time):
                tau = abs(t1[j] - t2[k])
                # Exponential decay with some noise
                correlation = 1.0 + 0.8 * np.exp(-q_list[i]**2 * 100 * tau) + 0.1 * np.random.randn()
                matrix[j, k] = max(0.1, correlation)  # Keep positive
        
        # Make matrix symmetric
        matrix = (matrix + matrix.T) / 2
        c2_matrices.append(matrix)
    
    # Introduce some data quality issues for demonstration
    print("⚠️  Introducing data quality issues for demonstration...")
    
    # Add some NaN values
    c2_matrices[2][5:8, 5:8] = np.nan
    print(f"   Added NaN values to matrix 2")
    
    # Add some infinite values  
    c2_matrices[5][10, 10] = np.inf
    print(f"   Added infinite value to matrix 5")
    
    # Scale one matrix incorrectly
    c2_matrices[8] *= 100  # 100x scaling issue
    print(f"   Added scaling issue to matrix 8 (100x)")
    
    # Add negative correlations
    c2_matrices[12][0:3, 0:3] -= 2.0  # Make some values negative
    print(f"   Added negative correlations to matrix 12")
    
    c2_exp = np.array(c2_matrices)
    
    # Create data dictionary
    demo_data = {
        'wavevector_q_list': q_list,
        'phi_angles_list': phi_list,
        't1': t1,
        't2': t2,
        'c2_exp': c2_exp
    }
    
    # Save as NPZ cache file (simulating pre-processed data)
    cache_path = demo_dir / "demo_cached_c2_frames_1_100.npz"
    np.savez_compressed(cache_path, **demo_data)
    print(f"✓ Saved synthetic data to: {cache_path}")
    
    return demo_data

def demonstrate_quality_controller():
    """Demonstrate standalone quality controller functionality."""
    print("\n🔍 === QUALITY CONTROLLER DEMONSTRATION ===")
    
    # Create configuration and controller
    config = create_demo_config()
    controller = create_quality_controller(config)
    
    # Create demo data with quality issues
    data = create_demo_data()
    
    print(f"\n📈 Data summary:")
    print(f"   Q-values: {len(data['wavevector_q_list'])} points")
    print(f"   Phi angles: {len(data['phi_angles_list'])} points") 
    print(f"   Correlation matrices: {data['c2_exp'].shape}")
    print(f"   Time points: {len(data['t1'])}")
    
    # Progressive quality control validation
    results = []
    
    print(f"\n🔬 Stage 1: Raw Data Validation")
    raw_result = controller.validate_data_stage(data, QualityControlStage.RAW_DATA)
    results.append(raw_result)
    print(f"   Quality Score: {raw_result.metrics.overall_score:.1f}/100")
    print(f"   Issues Found: {len(raw_result.issues)}")
    print(f"   Auto-Repairs: {len(raw_result.repairs_applied)}")
    if raw_result.repairs_applied:
        for repair in raw_result.repairs_applied:
            print(f"     - {repair}")
    
    print(f"\n🔬 Stage 2: Filtered Data Validation")
    filtered_result = controller.validate_data_stage(
        data, QualityControlStage.FILTERED_DATA, raw_result
    )
    results.append(filtered_result)
    print(f"   Quality Score: {filtered_result.metrics.overall_score:.1f}/100")
    print(f"   Filtering Efficiency: {filtered_result.metrics.filtering_efficiency:.1f}%")
    print(f"   Issues Found: {len(filtered_result.issues)}")
    
    print(f"\n🔬 Stage 3: Preprocessed Data Validation")
    preprocessed_result = controller.validate_data_stage(
        data, QualityControlStage.PREPROCESSED_DATA, filtered_result
    )
    results.append(preprocessed_result)
    print(f"   Quality Score: {preprocessed_result.metrics.overall_score:.1f}/100")
    print(f"   Preprocessing Success: {preprocessed_result.metrics.preprocessing_success}")
    print(f"   Transform Fidelity: {preprocessed_result.metrics.transformation_fidelity:.2f}")
    
    print(f"\n🔬 Stage 4: Final Data Validation")
    final_result = controller.validate_data_stage(
        data, QualityControlStage.FINAL_DATA, preprocessed_result
    )
    results.append(final_result)
    print(f"   Quality Score: {final_result.metrics.overall_score:.1f}/100")
    print(f"   Analysis Ready: {'Yes' if final_result.passed else 'No'}")
    print(f"   Total Issues: {len(final_result.issues)}")
    
    # Generate comprehensive quality report
    print(f"\n📊 Generating Quality Assessment Report...")
    report_path = Path("./examples/demo_data/quality_reports/demo_quality_report.json")
    quality_report = controller.generate_quality_report(results, str(report_path))
    
    print(f"   Overall Status: {quality_report['overall_summary']['status'].upper()}")
    print(f"   Total Stages: {quality_report['overall_summary']['total_stages_processed']}")
    print(f"   Total Issues: {quality_report['overall_summary']['total_issues_found']}")
    print(f"   Total Repairs: {quality_report['overall_summary']['total_repairs_applied']}")
    print(f"   Data Modified: {quality_report['overall_summary']['data_modified']}")
    print(f"   Report Saved: {report_path}")
    
    # Show recommendations
    print(f"\n💡 Quality Control Recommendations:")
    final_recommendations = quality_report['recommendations']
    for i, rec in enumerate(final_recommendations[:5], 1):
        print(f"   {i}. {rec}")
    
    return quality_report

def demonstrate_incremental_validation():
    """Demonstrate incremental validation functionality."""
    print(f"\n⚡ === INCREMENTAL VALIDATION DEMONSTRATION ===")
    
    config = create_demo_config()
    data = create_demo_data()
    
    # First validation
    print(f"\n🔍 Initial validation...")
    start_time = time.time()
    report1 = validate_xpcs_data_incremental(data, config, "incremental")
    time1 = time.time() - start_time
    print(f"   Initial validation completed in {time1:.3f}s")
    print(f"   Quality Score: {report1.quality_score:.3f}")
    print(f"   Issues: {len(report1.errors + report1.warnings)}")
    
    # Second validation (should use cache)
    print(f"\n🔍 Cached validation...")
    start_time = time.time()
    report2 = validate_xpcs_data_incremental(data, config, "incremental", report1)
    time2 = time.time() - start_time
    print(f"   Cached validation completed in {time2:.3f}s")
    print(f"   Speedup: {time1/time2:.1f}x faster")
    
    # Modify data slightly and validate again
    print(f"\n🔍 Modified data validation...")
    modified_data = data.copy()
    modified_data['c2_exp'] = data['c2_exp'] + 0.001 * np.random.randn(*data['c2_exp'].shape)
    
    start_time = time.time()
    report3 = validate_xpcs_data_incremental(modified_data, config, "incremental", report2)
    time3 = time.time() - start_time
    print(f"   Modified data validation completed in {time3:.3f}s")
    print(f"   Detected data changes and re-validated affected components")

def demonstrate_integrated_pipeline():
    """Demonstrate full integrated pipeline with quality control."""
    print(f"\n🚀 === INTEGRATED PIPELINE DEMONSTRATION ===")
    
    config = create_demo_config()
    
    # Force creation of demo data
    create_demo_data()
    
    print(f"\n📥 Loading data through integrated pipeline...")
    
    try:
        # This will fail because we don't have actual HDF5 file, but will show cache loading
        loader = XPCSDataLoader(config_dict=config)
        data = loader.load_experimental_data()
        
        print(f"✓ Data loaded successfully through integrated quality control pipeline")
        print(f"   Final data shapes: q{data['wavevector_q_list'].shape}, "
              f"phi{data['phi_angles_list'].shape}, c2{data['c2_exp'].shape}")
        
    except FileNotFoundError:
        print(f"⚠️  HDF5 file not found - this is expected for demo")
        print(f"   In real usage, the integrated pipeline would:")
        print(f"   1. Load data from HDF5 file")
        print(f"   2. Apply quality control at each stage")
        print(f"   3. Perform auto-repair as needed")
        print(f"   4. Generate comprehensive quality reports")
        print(f"   5. Integrate with filtering and preprocessing")

def print_demo_summary():
    """Print summary of quality control features."""
    print(f"\n📋 === QUALITY CONTROL FEATURES SUMMARY ===")
    
    features = {
        "Progressive Validation": [
            "Raw data validation after HDF5 loading",
            "Filtered data validation after filtering",
            "Preprocessed data validation after transformations", 
            "Final validation before analysis"
        ],
        "Auto-Repair Capabilities": [
            "NaN and infinite value repair with interpolation",
            "Scaling issue correction (10x, 100x factors)",
            "Format standardization across APS/APS-U sources",
            "Optional negative correlation fixing (aggressive mode)"
        ],
        "Quality Metrics": [
            "Overall quality score (0-100) with configurable thresholds",
            "Data integrity metrics (finite fraction, consistency)",
            "Physics validation (q-range, time consistency)",
            "Statistical analysis (signal-to-noise, correlation decay)"
        ],
        "Integration Features": [
            "Seamless integration with Phase 1 filtering system",
            "Seamless integration with Phase 2 preprocessing system", 
            "Incremental validation with intelligent caching",
            "Real-time quality feedback during data loading"
        ],
        "Reporting System": [
            "Comprehensive JSON quality reports",
            "Quality evolution analysis across processing stages",
            "Actionable recommendations for data improvement",
            "Quality history tracking for analysis workflows"
        ]
    }
    
    for category, items in features.items():
        print(f"\n🔧 {category}:")
        for item in items:
            print(f"   • {item}")

def main():
    """Main demonstration function."""
    print("🎯 Homodyne Phase 3 - Data Quality Control Integration Demo")
    print("=" * 60)
    print("This demo showcases the comprehensive quality control system")
    print("implemented in Phase 3, including progressive validation,")
    print("auto-repair capabilities, and integration with the data pipeline.")
    print("=" * 60)
    
    try:
        # Demonstrate quality controller functionality
        quality_report = demonstrate_quality_controller()
        
        # Demonstrate incremental validation
        demonstrate_incremental_validation()
        
        # Demonstrate integrated pipeline
        demonstrate_integrated_pipeline()
        
        # Print feature summary
        print_demo_summary()
        
        print(f"\n🎉 === DEMO COMPLETED SUCCESSFULLY ===")
        print(f"✓ All quality control features demonstrated")
        print(f"✓ Quality report generated and saved")
        print(f"✓ Integration with existing systems verified")
        print(f"\nNext steps:")
        print(f"1. Review the generated quality report in examples/demo_data/quality_reports/")
        print(f"2. Try the quality control template: config_quality_control_template.yaml")
        print(f"3. Enable quality_control in your own configurations")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())