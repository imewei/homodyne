"""
Convenience Functions for Homodyne v2 API
=========================================

Collection of convenience functions and shortcuts for common workflows,
providing even simpler interfaces for routine analysis tasks.

Key Features:
- One-line fitting functions for each method
- Quick data loading utilities
- Simple result export functions
- Plotting shortcuts
- Parameter exploration helpers
- Batch processing utilities
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from homodyne.api.high_level import AnalysisSession, fit_data, run_analysis
from homodyne.data.xpcs_loader import XPCSDataLoader
from homodyne.optimization.hybrid import HybridResult
from homodyne.optimization.mcmc import MCMCResult
from homodyne.optimization.variational import VIResult
from homodyne.results.exporters import MultiFormatExporter
from homodyne.utils.logging import get_logger
from homodyne.workflows.plotting_controller import PlottingController

logger = get_logger(__name__)

ResultType = Union[VIResult, MCMCResult, HybridResult]


def quick_vi_fit(
    data_file: Union[str, Path],
    iterations: int = 2000,
    learning_rate: float = 0.01,
    **kwargs,
) -> Dict[str, Any]:
    """
    Quick VI fit with sensible defaults.

    Simplified interface for Variational Inference with the most
    commonly adjusted parameters exposed.

    Args:
        data_file: Path to experimental data
        iterations: Number of VI iterations (default: 2000)
        learning_rate: VI learning rate (default: 0.01)
        **kwargs: Additional parameters passed to run_analysis

    Returns:
        Analysis results dictionary

    Example:
        >>> result = quick_vi_fit('data.h5', iterations=5000)
        >>> print(f"ELBO: {result['final_elbo']:.4f}")
        >>> params = result['parameters']
    """
    logger.info(f"🎯 Quick VI fit: {iterations} iterations, lr={learning_rate}")

    return run_analysis(
        data_file=data_file,
        method="vi",
        vi_iterations=iterations,
        vi_learning_rate=learning_rate,
        **kwargs,
    )


def quick_mcmc_fit(
    data_file: Union[str, Path],
    samples: int = 1000,
    chains: int = 4,
    warmup: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Quick MCMC fit with sensible defaults.

    Simplified interface for MCMC sampling with the most commonly
    adjusted parameters exposed.

    Args:
        data_file: Path to experimental data
        samples: Number of MCMC samples per chain (default: 1000)
        chains: Number of MCMC chains (default: 4)
        warmup: Warmup samples (default: same as samples)
        **kwargs: Additional parameters passed to run_analysis

    Returns:
        Analysis results dictionary

    Example:
        >>> result = quick_mcmc_fit('data.h5', samples=2000, chains=8)
        >>> print(f"Max R-hat: {result['max_r_hat']:.3f}")
        >>> params = result['parameters']
    """
    if warmup is None:
        warmup = samples

    logger.info(f"🎰 Quick MCMC fit: {samples} samples × {chains} chains")

    return run_analysis(
        data_file=data_file,
        method="mcmc",
        mcmc_samples=samples,
        mcmc_chains=chains,
        mcmc_warmup=warmup,
        **kwargs,
    )


def quick_hybrid_fit(
    data_file: Union[str, Path],
    vi_iterations: int = 1000,
    mcmc_samples: int = 1000,
    **kwargs,
) -> Dict[str, Any]:
    """
    Quick hybrid fit with sensible defaults.

    Simplified interface for hybrid VI→MCMC analysis with balanced
    computational allocation between both phases.

    Args:
        data_file: Path to experimental data
        vi_iterations: VI phase iterations (default: 1000)
        mcmc_samples: MCMC phase samples (default: 1000)
        **kwargs: Additional parameters passed to run_analysis

    Returns:
        Analysis results dictionary

    Example:
        >>> result = quick_hybrid_fit('data.h5')
        >>> print(f"Recommended: {result['recommended_method']}")
        >>> params = result['parameters']
    """
    logger.info(f"🔄 Quick hybrid fit: VI({vi_iterations}) → MCMC({mcmc_samples})")

    return run_analysis(
        data_file=data_file,
        method="hybrid",
        vi_iterations=vi_iterations,
        mcmc_samples=mcmc_samples,
        **kwargs,
    )


def load_data(
    data_file: Union[str, Path],
    config_file: Optional[Union[str, Path]] = None,
    validate: bool = True,
) -> Dict[str, Any]:
    """
    Load experimental data with automatic validation.

    Simple data loading utility that handles XPCS data with
    optional validation and basic preprocessing.

    Args:
        data_file: Path to experimental data file
        config_file: Optional configuration file
        validate: Perform data validation (default: True)

    Returns:
        Data dictionary with loaded arrays

    Example:
        >>> data = load_data('experiment.h5')
        >>> print(f"Data shape: {data['c2_exp'].shape}")
        >>> print(f"Q value: {data['q']:.4f}")
    """
    logger.info(f"📊 Loading data from {data_file}")

    data_file = Path(data_file)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    # Create minimal config for data loading
    config = {"data": {"file_path": str(data_file)}}

    if config_file:
        config_file = Path(config_file)
        if config_file.exists():
            # Load and merge with file config
            from homodyne.config.manager import ConfigManager

            config_manager = ConfigManager(str(config_file))
            config = config_manager.config
        else:
            warnings.warn(f"Config file not found: {config_file}")

    # Load data
    loader = XPCSDataLoader(config_dict=config)
    data_dict = loader.load_experimental_data()

    # Basic validation if requested
    if validate:
        _validate_data_dict(data_dict)

    logger.info(f"✓ Data loaded successfully")
    logger.info(f"  Shape: {data_dict['c2_exp'].shape}")
    logger.info(f"  Phi angles: {len(data_dict['phi_angles'])}")
    logger.info(f"  Q: {data_dict['q']:.4f}, L: {data_dict['L']:.4f}")

    return data_dict


def export_results(
    result: Union[ResultType, Dict[str, Any]],
    output_dir: Union[str, Path],
    formats: List[str] = ["yaml", "json", "npz"],
    include_plots: bool = False,
) -> Dict[str, Path]:
    """
    Export analysis results in multiple formats.

    Convenient function for exporting results with automatic
    format selection and file naming.

    Args:
        result: Analysis result or result dictionary
        output_dir: Output directory for exported files
        formats: List of export formats
        include_plots: Include analysis plots (requires data)

    Returns:
        Dictionary mapping formats to export file paths

    Example:
        >>> export_paths = export_results(result, './exports', ['yaml', 'npz'])
        >>> print(f"YAML saved to: {export_paths['yaml']}")
    """
    logger.info(f"💾 Exporting results to {len(formats)} formats")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle result dictionary (from API) vs result object
    if isinstance(result, dict):
        # Convert API result dict back to result object (simplified)
        logger.warning(
            "Result dictionary export not fully implemented - using simplified export"
        )

        # Save as JSON for now
        import json

        json_file = output_dir / "result_summary.json"
        with open(json_file, "w") as f:
            json.dump(result, f, indent=2, default=str)

        return {"json": json_file}

    # Export result object
    exporter = MultiFormatExporter(output_dir)
    export_paths = exporter.export_all_formats(result, formats)

    logger.info(f"✓ Results exported to {len(export_paths)} formats")
    return export_paths


def plot_results(
    result: Union[ResultType, Dict[str, Any]],
    data: Optional[Dict[str, Any]] = None,
    output_dir: Union[str, Path] = "./plots",
    plot_types: List[str] = ["fit_comparison", "residuals"],
) -> List[Path]:
    """
    Generate plots for analysis results.

    Simple plotting interface that creates standard visualization
    plots for analysis results.

    Args:
        result: Analysis result or result dictionary
        data: Original experimental data (required for some plots)
        output_dir: Output directory for plots
        plot_types: Types of plots to generate

    Returns:
        List of generated plot file paths

    Example:
        >>> plot_files = plot_results(result, data, plot_types=['fit_comparison'])
        >>> print(f"Generated {len(plot_files)} plots")
    """
    logger.info(f"🎨 Generating {len(plot_types)} plot types")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(result, dict):
        logger.warning("Plotting from result dictionary not fully implemented")
        return []

    if data is None:
        logger.warning("No data provided - limited plotting capability")
        return []

    # Generate plots
    plotter = PlottingController(output_dir=output_dir)

    plot_files = []

    if "fit_comparison" in plot_types:
        plotter.plot_fit_results(result, data)
        plot_files.extend(
            [output_dir / "c2_fitted_heatmap.png", output_dir / "residuals_heatmap.png"]
        )

    if "experimental_data" in plot_types and data:
        plotter.plot_experimental_data(data)
        plot_files.append(output_dir / "c2_experimental_heatmap.png")

    # Filter for actually created files
    actual_files = [f for f in plot_files if f.exists()]

    logger.info(f"✓ Generated {len(actual_files)} plots")
    return actual_files


def batch_analyze(
    data_files: List[Union[str, Path]],
    method: str = "vi",
    output_base_dir: Union[str, Path] = "./batch_analysis",
    **kwargs,
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze multiple data files with the same method.

    Convenient batch processing for analyzing multiple datasets
    with identical analysis parameters.

    Args:
        data_files: List of data file paths
        method: Analysis method to use for all files
        output_base_dir: Base directory for all outputs
        **kwargs: Analysis parameters applied to all files

    Returns:
        Dictionary mapping file names to analysis results

    Example:
        >>> files = ['exp1.h5', 'exp2.h5', 'exp3.h5']
        >>> results = batch_analyze(files, method='mcmc', mcmc_samples=2000)
        >>> for name, result in results.items():
        >>>     print(f"{name}: χ² = {result['chi_squared']:.4f}")
    """
    logger.info(f"🔄 Batch analyzing {len(data_files)} files with {method.upper()}")

    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for i, data_file in enumerate(data_files):
        data_file = Path(data_file)
        file_name = data_file.stem

        logger.info(f"Processing {i + 1}/{len(data_files)}: {file_name}")

        # Create individual output directory
        file_output_dir = output_base_dir / file_name

        try:
            result = run_analysis(
                data_file=data_file, method=method, output_dir=file_output_dir, **kwargs
            )
            results[file_name] = result
            logger.info(f"✓ {file_name} completed")

        except Exception as e:
            logger.error(f"❌ {file_name} failed: {e}")
            results[file_name] = {"error": str(e)}

    # Create batch summary
    successful = len([r for r in results.values() if "error" not in r])
    logger.info(
        f"✓ Batch analysis completed: {successful}/{len(data_files)} successful"
    )

    return results


def explore_parameters(
    data_file: Union[str, Path],
    method: str = "vi",
    parameter_ranges: Dict[str, List] = None,
    output_dir: Union[str, Path] = "./parameter_exploration",
) -> Dict[str, Any]:
    """
    Explore parameter sensitivity by running multiple analyses.

    Systematic parameter exploration for understanding sensitivity
    and optimization landscape.

    Args:
        data_file: Path to experimental data
        method: Analysis method to use
        parameter_ranges: Dictionary of parameter names to value lists
        output_dir: Output directory for exploration results

    Returns:
        Parameter exploration results

    Example:
        >>> ranges = {'vi_iterations': [1000, 2000, 5000],
        >>>           'vi_learning_rate': [0.001, 0.01, 0.1]}
        >>> exploration = explore_parameters('data.h5', 'vi', ranges)
        >>> best_params = exploration['best_parameters']
    """
    if parameter_ranges is None:
        # Default parameter exploration for VI
        if method == "vi":
            parameter_ranges = {
                "vi_iterations": [1000, 2000, 5000],
                "vi_learning_rate": [0.001, 0.01, 0.1],
            }
        elif method == "mcmc":
            parameter_ranges = {
                "mcmc_samples": [500, 1000, 2000],
                "mcmc_chains": [2, 4, 8],
            }
        else:
            raise ValueError("Must specify parameter_ranges for exploration")

    logger.info(
        f"🔍 Exploring {len(parameter_ranges)} parameters with {method.upper()}"
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate parameter combinations
    import itertools

    param_names = list(parameter_ranges.keys())
    param_values = list(parameter_ranges.values())

    exploration_results = []

    for i, combination in enumerate(itertools.product(*param_values)):
        param_dict = dict(zip(param_names, combination))

        logger.info(f"Exploration {i + 1}: {param_dict}")

        try:
            result = run_analysis(
                data_file=data_file,
                method=method,
                output_dir=output_dir / f"exploration_{i:03d}",
                **param_dict,
            )

            exploration_results.append(
                {
                    "parameters": param_dict,
                    "chi_squared": result.get("chi_squared"),
                    "result": result,
                    "success": True,
                }
            )

        except Exception as e:
            logger.error(f"Exploration {i + 1} failed: {e}")
            exploration_results.append(
                {"parameters": param_dict, "error": str(e), "success": False}
            )

    # Analyze exploration results
    successful_results = [r for r in exploration_results if r["success"]]

    if successful_results:
        # Find best parameters by chi-squared
        best_result = min(
            successful_results, key=lambda r: r["chi_squared"] or float("inf")
        )

        exploration_summary = {
            "total_explorations": len(exploration_results),
            "successful_explorations": len(successful_results),
            "best_parameters": best_result["parameters"],
            "best_chi_squared": best_result["chi_squared"],
            "all_results": exploration_results,
        }
    else:
        exploration_summary = {
            "total_explorations": len(exploration_results),
            "successful_explorations": 0,
            "error": "All parameter explorations failed",
        }

    logger.info(
        f"✓ Parameter exploration completed: {len(successful_results)}/{len(exploration_results)} successful"
    )

    return exploration_summary


def create_session(
    data_file: Optional[Union[str, Path]] = None, **kwargs
) -> AnalysisSession:
    """
    Create an interactive analysis session.

    Convenience function for creating analysis sessions with
    common default settings.

    Args:
        data_file: Data file to load (optional)
        **kwargs: Session parameters

    Returns:
        Initialized analysis session

    Example:
        >>> session = create_session('data.h5')
        >>> vi_result = session.analyze('vi', vi_iterations=3000)
        >>> mcmc_result = session.analyze('mcmc', mcmc_samples=2000)
        >>> comparison = session.compare_all()
    """
    logger.info(f"🎯 Creating interactive analysis session")

    session = AnalysisSession(data_file=data_file, **kwargs)

    if data_file:
        logger.info(f"✓ Session ready with data from {data_file}")
    else:
        logger.info(f"✓ Session ready - use session.load_data() to load data")

    return session


def get_method_recommendations(
    data_size: int,
    time_budget_seconds: float = 300.0,
    accuracy_priority: str = "balanced",
) -> Dict[str, Any]:
    """
    Get method recommendations based on data characteristics and constraints.

    Provides intelligent method selection based on dataset size,
    time constraints, and accuracy requirements.

    Args:
        data_size: Total number of data points
        time_budget_seconds: Available computation time budget
        accuracy_priority: 'speed', 'accuracy', or 'balanced'

    Returns:
        Method recommendations with rationale

    Example:
        >>> recs = get_method_recommendations(data_size=5000000, time_budget_seconds=600)
        >>> print(f"Primary recommendation: {recs['primary_method']}")
        >>> print(f"Rationale: {recs['rationale']}")
    """
    logger.info(f"🧠 Generating method recommendations")
    logger.info(f"  Data size: {data_size:,} points")
    logger.info(f"  Time budget: {time_budget_seconds:.0f}s")
    logger.info(f"  Priority: {accuracy_priority}")

    # Simple heuristic-based recommendations
    recommendations = {
        "data_size": data_size,
        "time_budget": time_budget_seconds,
        "accuracy_priority": accuracy_priority,
    }

    # Primary method selection
    if accuracy_priority == "speed":
        primary_method = "vi"
        rationale = "VI selected for speed priority - fastest convergence"
    elif accuracy_priority == "accuracy":
        if time_budget_seconds > 600:  # 10+ minutes
            primary_method = "mcmc"
            rationale = (
                "MCMC selected for accuracy priority with sufficient time budget"
            )
        else:
            primary_method = "hybrid"
            rationale = "Hybrid selected - best accuracy within time constraints"
    else:  # balanced
        if data_size < 1_000_000:
            primary_method = "hybrid"
            rationale = "Hybrid selected for balanced accuracy/speed on medium dataset"
        elif data_size < 10_000_000:
            primary_method = "vi"
            rationale = "VI selected for large dataset - MCMC may be too slow"
        else:
            primary_method = "vi"
            rationale = "VI selected for very large dataset - only practical option"

    recommendations.update(
        {
            "primary_method": primary_method,
            "rationale": rationale,
            "alternative_methods": [
                m for m in ["vi", "mcmc", "hybrid"] if m != primary_method
            ],
            "estimated_time": _estimate_analysis_time(data_size, primary_method),
            "confidence": "medium",  # Would be based on more sophisticated analysis
        }
    )

    logger.info(f"✓ Recommendation: {primary_method.upper()} - {rationale}")

    return recommendations


# Helper functions


def _validate_data_dict(data_dict: Dict[str, Any]) -> None:
    """Validate loaded data dictionary."""
    required_keys = ["c2_exp", "t1", "t2", "phi_angles", "q", "L"]

    for key in required_keys:
        if key not in data_dict:
            raise ValueError(f"Missing required data key: {key}")

    # Check data shape consistency
    c2_shape = data_dict["c2_exp"].shape
    if len(c2_shape) != 3:
        raise ValueError(f"Expected 3D correlation data, got shape {c2_shape}")

    n_phi, n_t1, n_t2 = c2_shape

    if len(data_dict["phi_angles"]) != n_phi:
        raise ValueError(
            f"Phi angles length {len(data_dict['phi_angles'])} doesn't match data shape {n_phi}"
        )

    if len(data_dict["t1"]) != n_t1:
        raise ValueError(
            f"t1 length {len(data_dict['t1'])} doesn't match data shape {n_t1}"
        )

    if len(data_dict["t2"]) != n_t2:
        raise ValueError(
            f"t2 length {len(data_dict['t2'])} doesn't match data shape {n_t2}"
        )

    # Check for NaN/infinite values
    if not np.isfinite(data_dict["c2_exp"]).all():
        warnings.warn("Correlation data contains NaN or infinite values")

    logger.debug("✓ Data validation passed")


def _estimate_analysis_time(data_size: int, method: str) -> Dict[str, float]:
    """Estimate analysis time for given data size and method."""
    # Rough estimates based on typical performance (would be calibrated from benchmarks)
    base_rates = {
        "vi": 50000,  # points per second
        "mcmc": 5000,  # points per second
        "hybrid": 15000,  # points per second
    }

    rate = base_rates.get(method, 10000)
    estimated_seconds = data_size / rate

    return {
        "seconds": estimated_seconds,
        "minutes": estimated_seconds / 60,
        "human_readable": (
            f"{estimated_seconds / 60:.1f} minutes"
            if estimated_seconds > 60
            else f"{estimated_seconds:.0f} seconds"
        ),
    }
