"""
Complete Usage Examples for Homodyne v2 Python API
==================================================

Comprehensive examples demonstrating all API capabilities from basic
usage to advanced workflows. Suitable for documentation, tutorials,
and as starting templates for user analyses.

Examples Cover:
- Basic analysis workflows
- Method comparison and selection
- Interactive session usage
- Batch processing
- Parameter exploration
- Custom analysis pipelines
- Jupyter notebook integration
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Homodyne API imports
from homodyne.api import (AnalysisSession, batch_analyze, compare_methods,
                          create_session, explore_parameters, export_results,
                          fit_data, get_method_recommendations,
                          load_and_analyze, load_data, plot_results,
                          quick_hybrid_fit, quick_mcmc_fit, quick_vi_fit,
                          run_analysis)
from homodyne.utils.logging import setup_logging


def example_basic_analysis():
    """
    Example 1: Basic Analysis Workflow

    Demonstrates the simplest way to run a complete analysis
    with automatic configuration and result export.
    """
    print("=== Example 1: Basic Analysis ===")

    # Setup logging for nice output
    setup_logging(level="INFO")

    # Run complete analysis with single function call
    results = run_analysis(
        data_file="sample_data.h5",
        method="vi",  # Variational Inference (fastest)
        analysis_mode="auto",  # Automatic mode detection
        output_dir="./basic_analysis",  # Results will be saved here
    )

    # Display key results
    print(f"Analysis completed successfully!")
    print(f"Method: {results['method'].upper()}")
    print(f"Parameters: {results['parameters']}")
    print(f"Chi-squared: {results['chi_squared']:.4f}")
    print(f"Output directory: {results['output_directory']}")

    # Results are automatically exported in multiple formats
    # Plots are automatically generated

    return results


def example_method_comparison():
    """
    Example 2: Method Comparison

    Demonstrates how to compare multiple methods on the same
    dataset to choose the best approach.
    """
    print("\n=== Example 2: Method Comparison ===")

    # Compare all three methods
    comparison = compare_methods(
        data_file="sample_data.h5", output_dir="./method_comparison"
    )

    print(f"Methods compared: {comparison['individual_results'].keys()}")
    print(f"Recommended method: {comparison['recommendation']}")

    # Detailed comparison
    for method, result in comparison["individual_results"].items():
        if "error" not in result:
            chi_sq = result.get("chi_squared", "N/A")
            print(f"{method.upper()}: χ² = {chi_sq}")
        else:
            print(f"{method.upper()}: Failed - {result['error']}")

    return comparison


def example_quick_analysis():
    """
    Example 3: Quick Analysis Functions

    Demonstrates convenience functions for rapid analysis
    with commonly adjusted parameters.
    """
    print("\n=== Example 3: Quick Analysis Functions ===")

    # Quick VI with custom parameters
    vi_result = quick_vi_fit(
        "sample_data.h5",
        iterations=5000,  # More iterations for better convergence
        learning_rate=0.005,  # Lower learning rate for stability
    )
    print(f"Quick VI: ELBO = {vi_result.get('final_elbo', 'N/A'):.4f}")

    # Quick MCMC with custom sampling
    mcmc_result = quick_mcmc_fit(
        "sample_data.h5",
        samples=2000,  # More samples for better posterior
        chains=8,  # More chains for better mixing
        warmup=1000,
    )
    print(f"Quick MCMC: Max R-hat = {mcmc_result.get('max_r_hat', 'N/A'):.3f}")

    # Quick hybrid analysis
    hybrid_result = quick_hybrid_fit(
        "sample_data.h5",
        vi_iterations=1000,  # Moderate VI phase
        mcmc_samples=1500,  # More MCMC samples
    )
    print(f"Quick Hybrid: Recommends {hybrid_result.get('recommended_method', 'N/A')}")

    return vi_result, mcmc_result, hybrid_result


def example_interactive_session():
    """
    Example 4: Interactive Analysis Session

    Demonstrates session-based analysis for interactive exploration
    and progressive refinement.
    """
    print("\n=== Example 4: Interactive Session ===")

    # Create analysis session
    session = create_session("sample_data.h5", auto_plot=True)

    print(f"Session created with {session.data_dict['c2_exp'].size:,} data points")

    # Run different analyses interactively
    print("Running VI analysis...")
    vi_result = session.analyze("vi", vi_iterations=3000)

    print("Running MCMC analysis...")
    mcmc_result = session.analyze("mcmc", mcmc_samples=1000, mcmc_chains=4)

    # Compare methods within session
    comparison = session.compare_all()
    print("Session comparison:")
    for method, metrics in comparison.items():
        print(f"  {method}: χ² = {metrics.get('chi_squared', 'N/A')}")

    # Export all session results
    export_paths = session.export_results(formats=["yaml", "json", "npz"])
    print(f"Results exported to: {list(export_paths.keys())}")

    # Get session summary
    summary = session.get_summary()
    print(
        f"Session summary: {summary['methods_run']} methods in {summary['analysis_count']} runs"
    )

    return session


def example_direct_fitting():
    """
    Example 5: Direct Data Fitting

    Demonstrates direct fitting with numpy arrays for advanced users
    who have already preprocessed their data.
    """
    print("\n=== Example 5: Direct Data Fitting ===")

    # Create synthetic data for demonstration
    n_phi, n_t1, n_t2 = 5, 50, 50
    phi_angles = np.linspace(0, np.pi / 2, n_phi)
    t1 = np.logspace(-3, 1, n_t1)
    t2 = np.logspace(-3, 1, n_t2)

    # Generate synthetic correlation data
    T1, T2, PHI = np.meshgrid(t1, t2, phi_angles, indexing="ij")
    synthetic_data = np.exp(-0.1 * (T1 + T2)) * (1 + 0.5 * np.cos(2 * PHI))
    synthetic_data = synthetic_data.transpose(2, 0, 1)  # (phi, t1, t2)

    print(f"Fitting synthetic data with shape {synthetic_data.shape}")

    # Fit data directly
    result = fit_data(
        data=synthetic_data,
        t1=t1,
        t2=t2,
        phi=phi_angles,
        q=0.01,  # Scattering vector
        L=10.0,  # Sample thickness
        method="vi",
        analysis_mode="static_anisotropic",
    )

    print(f"Direct fitting results:")
    print(f"  Parameters: {result.mean_params}")
    print(f"  Chi-squared: {result.chi_squared:.4f}")
    print(f"  Converged: {getattr(result, 'converged', 'N/A')}")

    return result


def example_batch_processing():
    """
    Example 6: Batch Processing

    Demonstrates automated batch processing of multiple datasets
    with identical analysis parameters.
    """
    print("\n=== Example 6: Batch Processing ===")

    # List of data files to process
    data_files = ["experiment_001.h5", "experiment_002.h5", "experiment_003.h5"]

    # Note: For demo purposes, we'll use the same file multiple times
    # In practice, you would have different experimental files
    demo_files = ["sample_data.h5"] * 3

    print(f"Processing {len(demo_files)} files...")

    # Batch analyze with consistent parameters
    batch_results = batch_analyze(
        data_files=demo_files,
        method="vi",
        output_base_dir="./batch_results",
        vi_iterations=2000,
        generate_plots=True,
    )

    print("Batch processing results:")
    for file_name, result in batch_results.items():
        if "error" not in result:
            chi_sq = result.get("chi_squared", "N/A")
            print(f"  {file_name}: χ² = {chi_sq}")
        else:
            print(f"  {file_name}: Failed - {result['error']}")

    return batch_results


def example_parameter_exploration():
    """
    Example 7: Parameter Exploration

    Demonstrates systematic exploration of analysis parameters
    to understand sensitivity and optimize settings.
    """
    print("\n=== Example 7: Parameter Exploration ===")

    # Define parameter ranges to explore
    vi_parameter_ranges = {
        "vi_iterations": [1000, 2000, 5000],
        "vi_learning_rate": [0.001, 0.01, 0.1],
    }

    print(f"Exploring VI parameters: {vi_parameter_ranges}")

    # Run parameter exploration
    exploration = explore_parameters(
        data_file="sample_data.h5",
        method="vi",
        parameter_ranges=vi_parameter_ranges,
        output_dir="./parameter_exploration",
    )

    print(f"Parameter exploration completed:")
    print(f"  Total combinations: {exploration['total_explorations']}")
    print(f"  Successful runs: {exploration['successful_explorations']}")

    if exploration["successful_explorations"] > 0:
        print(f"  Best parameters: {exploration['best_parameters']}")
        print(f"  Best χ²: {exploration['best_chi_squared']:.4f}")

    return exploration


def example_custom_workflow():
    """
    Example 8: Custom Analysis Workflow

    Demonstrates building custom analysis pipelines by combining
    different API functions for specialized workflows.
    """
    print("\n=== Example 8: Custom Analysis Workflow ===")

    # Step 1: Load and examine data
    print("Step 1: Loading data...")
    data_dict = load_data("sample_data.h5", validate=True)
    print(f"Data loaded: {data_dict['c2_exp'].shape} correlation matrix")

    # Step 2: Get method recommendations
    data_size = data_dict["c2_exp"].size
    recommendations = get_method_recommendations(
        data_size=data_size,
        time_budget_seconds=300,  # 5 minutes
        accuracy_priority="balanced",
    )
    print(f"Step 2: Recommended method: {recommendations['primary_method']}")
    print(f"Rationale: {recommendations['rationale']}")

    # Step 3: Run recommended method with custom parameters
    recommended_method = recommendations["primary_method"]
    print(f"Step 3: Running {recommended_method.upper()} analysis...")

    if recommended_method == "vi":
        result = quick_vi_fit("sample_data.h5", iterations=3000)
    elif recommended_method == "mcmc":
        result = quick_mcmc_fit("sample_data.h5", samples=1500)
    else:  # hybrid
        result = quick_hybrid_fit("sample_data.h5")

    # Step 4: Export results in specific formats
    print("Step 4: Exporting results...")
    export_paths = export_results(
        result=result,
        output_dir="./custom_workflow",
        formats=["yaml", "json"],
        include_plots=True,
    )
    print(f"Exported to: {list(export_paths.keys())}")

    # Step 5: Generate custom plots
    print("Step 5: Generating plots...")
    plot_files = plot_results(
        result=result,
        data=data_dict,
        output_dir="./custom_workflow/plots",
        plot_types=["fit_comparison", "experimental_data"],
    )
    print(f"Generated {len(plot_files)} plot files")

    return result, export_paths, plot_files


def example_jupyter_notebook_integration():
    """
    Example 9: Jupyter Notebook Integration

    Demonstrates API usage patterns optimized for Jupyter notebooks
    with inline plotting and progressive analysis.
    """
    print("\n=== Example 9: Jupyter Notebook Integration ===")
    print("# This example shows patterns for Jupyter notebook usage")

    # Cell 1: Setup and data loading
    print("\n# Cell 1: Setup")
    setup_logging(level="INFO", quiet=True)  # Quieter for notebooks
    session = create_session(auto_plot=False)  # Manual plotting control
    session.load_data("sample_data.h5")
    print(f"✓ Session ready with {session.data_dict['c2_exp'].size:,} points")

    # Cell 2: Quick exploration
    print("\n# Cell 2: Quick VI analysis")
    vi_result = session.analyze("vi", vi_iterations=1000)
    print(f"VI completed: ELBO = {getattr(vi_result, 'final_elbo', 'N/A'):.4f}")

    # Cell 3: Plot results inline (would show plots in notebook)
    print("\n# Cell 3: Generate plots")
    session.plot_latest("vi")
    print("✓ Plots generated (would display inline in notebook)")

    # Cell 4: Compare with MCMC
    print("\n# Cell 4: MCMC comparison")
    mcmc_result = session.analyze("mcmc", mcmc_samples=500)  # Quick for demo
    comparison = session.compare_all()

    print("Method comparison:")
    for method, metrics in comparison.items():
        chi_sq = metrics.get("chi_squared", "N/A")
        print(f"  {method}: χ² = {chi_sq}")

    # Cell 5: Export final results
    print("\n# Cell 5: Export results")
    export_paths = session.export_results(["yaml"])
    print(f"✓ Results exported to: {export_paths}")

    return session


def example_error_handling():
    """
    Example 10: Error Handling and Troubleshooting

    Demonstrates proper error handling and common troubleshooting
    scenarios in analysis workflows.
    """
    print("\n=== Example 10: Error Handling ===")

    # Example 1: Handling missing files
    try:
        result = run_analysis("nonexistent_file.h5")
    except FileNotFoundError as e:
        print(f"✓ Caught expected error: {e}")

    # Example 2: Handling invalid parameters
    try:
        result = quick_vi_fit("sample_data.h5", iterations=-1000)
    except ValueError as e:
        print(f"✓ Caught parameter validation error: {e}")

    # Example 3: Graceful handling of analysis failures
    try:
        # This might fail with difficult data
        result = fit_data(
            data=np.full((3, 10, 10), np.nan),  # Invalid data
            method="vi",
        )
    except Exception as e:
        print(f"✓ Caught analysis error: {type(e).__name__}: {e}")

    # Example 4: Batch processing with some failures
    print("\nBatch processing with mixed success:")
    mixed_files = [
        "sample_data.h5",  # Valid file
        "nonexistent.h5",  # Invalid file
        "sample_data.h5",  # Valid file again
    ]

    # Batch analyze will handle individual failures gracefully
    batch_results = {}
    for i, file_path in enumerate(mixed_files):
        try:
            result = run_analysis(file_path, output_dir=f"./error_demo_{i}")
            batch_results[f"file_{i}"] = "SUCCESS"
            print(f"  File {i}: ✓ Success")
        except Exception as e:
            batch_results[f"file_{i}"] = f"FAILED: {e}"
            print(f"  File {i}: ❌ Failed - {type(e).__name__}")

    print(
        f"Batch completed: {sum(1 for v in batch_results.values() if 'SUCCESS' in v)}/{len(mixed_files)} successful"
    )

    return batch_results


def run_all_examples():
    """
    Run all examples in sequence.

    This function executes all example functions to demonstrate
    the complete range of API capabilities.
    """
    print("🚀 Running all Homodyne v2 API examples...")
    print("=" * 60)

    examples = [
        example_basic_analysis,
        example_method_comparison,
        example_quick_analysis,
        example_interactive_session,
        example_direct_fitting,
        example_batch_processing,
        example_parameter_exploration,
        example_custom_workflow,
        example_jupyter_notebook_integration,
        example_error_handling,
    ]

    results = {}

    for i, example_func in enumerate(examples, 1):
        try:
            print(f"\n{'=' * 20} Running Example {i} {'=' * 20}")
            result = example_func()
            results[example_func.__name__] = result
            print(f"✅ Example {i} completed successfully")

        except Exception as e:
            print(f"❌ Example {i} failed: {e}")
            results[example_func.__name__] = f"ERROR: {e}"

    print(f"\n{'=' * 60}")
    print(f"🏁 All examples completed!")
    print(
        f"Successful examples: {sum(1 for v in results.values() if not isinstance(v, str) or not v.startswith('ERROR'))}/{len(examples)}"
    )

    return results


# Additional utility functions for examples


def create_sample_data(output_path: str = "sample_data.h5"):
    """
    Create sample data file for examples.

    This creates a synthetic dataset that can be used to run
    all the examples without requiring real experimental data.
    """
    print(f"Creating sample data file: {output_path}")

    try:
        import h5py
    except ImportError:
        print("h5py required for creating sample data")
        return False

    # Generate synthetic XPCS data
    n_phi, n_t1, n_t2 = 3, 30, 30
    phi_angles = np.array([0, np.pi / 4, np.pi / 2])
    t1 = np.logspace(-3, 1, n_t1)  # 1ms to 10s
    t2 = np.logspace(-3, 1, n_t2)

    # Create correlation function with realistic physics
    T1, T2, PHI = np.meshgrid(t1, t2, phi_angles, indexing="ij")

    # Anomalous diffusion + shear flow model
    D0 = 1e-8  # Diffusion coefficient
    alpha = 0.8  # Subdiffusive exponent
    gamma_dot = 0.5  # Shear rate

    # Time-dependent diffusion
    g1 = np.exp(-D0 * (T1**alpha + T2**alpha))

    # Add shear flow contribution
    g1 *= np.exp(-gamma_dot * np.abs(T1 - T2) * np.sin(PHI) ** 2)

    # Convert to g2 = 1 + contrast * g1^2
    contrast = 0.8
    c2_data = 1.0 + contrast * g1**2
    c2_data = c2_data.transpose(2, 0, 1)  # (phi, t1, t2)

    # Add realistic noise
    noise_level = 0.02
    c2_data += noise_level * np.random.normal(size=c2_data.shape)
    c2_std = noise_level * np.ones_like(c2_data)

    # Save as HDF5 file
    with h5py.File(output_path, "w") as f:
        f.create_dataset("c2_exp", data=c2_data)
        f.create_dataset("c2_std", data=c2_std)
        f.create_dataset("t1", data=t1)
        f.create_dataset("t2", data=t2)
        f.create_dataset("phi_angles", data=phi_angles)
        f.create_dataset("q", data=0.01)  # Scattering vector
        f.create_dataset("L", data=10.0)  # Sample thickness

        # Add metadata
        f.attrs["experiment_type"] = "synthetic_xpcs"
        f.attrs["created_by"] = "homodyne_examples"
        f.attrs["n_phi"] = n_phi
        f.attrs["n_t1"] = n_t1
        f.attrs["n_t2"] = n_t2

    print(f"✓ Sample data created: {c2_data.shape} correlation matrix")
    return True


if __name__ == "__main__":
    # If running as script, create sample data and run all examples
    print("Homodyne v2 API Examples")
    print("=" * 40)

    # Create sample data if needed
    sample_file = "sample_data.h5"
    if not Path(sample_file).exists():
        print("Creating sample data for examples...")
        if not create_sample_data(sample_file):
            print("❌ Could not create sample data. Some examples may fail.")
        else:
            print("✓ Sample data created successfully")

    # Run all examples
    results = run_all_examples()

    print("\n📚 Example Documentation:")
    print("All examples are extensively documented and can be used as:")
    print("• Learning materials for new users")
    print("• Templates for custom analysis workflows")
    print("• Reference implementations for API usage")
    print("• Integration examples for Jupyter notebooks")

    print("\n🔗 Next Steps:")
    print("• Copy and modify examples for your analysis needs")
    print("• Explore parameter ranges suitable for your data")
    print("• Integrate with existing analysis pipelines")
    print("• Refer to API documentation for detailed parameter descriptions")
