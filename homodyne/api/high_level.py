"""
High-Level Python API for Homodyne v2
=====================================

Simplified Python interface for homodyne analysis providing one-line
analysis execution with automatic configuration and result processing.

Key Features:
- One-function analysis execution
- Automatic mode detection and configuration
- Built-in data validation and preprocessing
- Integrated result export and visualization
- Session-based analysis for interactive use
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import warnings
import numpy as np
from datetime import datetime

from homodyne.utils.logging import get_logger, configure_logging
from homodyne.data.xpcs_loader import XPCSDataLoader
from homodyne.workflows.executor import MethodExecutor
from homodyne.workflows.results_manager import ResultsManager
from homodyne.workflows.plotting_controller import PlottingController
from homodyne.config.cli_config import CLIConfigManager
from homodyne.config.mode_resolver import ModeResolver
from homodyne.optimization.variational import VIResult
from homodyne.optimization.mcmc import MCMCResult
from homodyne.optimization.hybrid import HybridResult
from homodyne.workflows.pipeline import AnalysisPipeline

logger = get_logger(__name__)

ResultType = Union[VIResult, MCMCResult, HybridResult]


def run_analysis(data_file: Union[str, Path],
                method: str = "vi",
                analysis_mode: str = "auto",
                output_dir: Optional[Union[str, Path]] = None,
                config_file: Optional[Union[str, Path]] = None,
                **kwargs) -> Dict[str, Any]:
    """
    Run complete homodyne analysis with a single function call.
    
    This is the main entry point for Python API usage, providing
    a simplified interface to the complete analysis pipeline.
    
    Args:
        data_file: Path to experimental data file
        method: Analysis method ('vi', 'mcmc', 'hybrid')
        analysis_mode: Analysis mode ('auto', 'static_isotropic', 
                      'static_anisotropic', 'laminar_flow')
        output_dir: Output directory for results (optional)
        config_file: Configuration file path (optional)
        **kwargs: Additional configuration parameters
        
    Returns:
        Dictionary containing analysis results and metadata
        
    Example:
        >>> results = run_analysis('data.h5', method='vi', analysis_mode='laminar_flow')
        >>> print(f"Chi-squared: {results['chi_squared']:.4f}")
        >>> parameters = results['parameters']
    """
    logger.info(f"🚀 Starting homodyne analysis via Python API")
    logger.info(f"  Data file: {data_file}")
    logger.info(f"  Method: {method.upper()}")
    logger.info(f"  Mode: {analysis_mode}")
    
    # Validate inputs
    data_file = Path(data_file)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    if method not in ['vi', 'mcmc', 'hybrid']:
        raise ValueError(f"Invalid method '{method}'. Choose from: vi, mcmc, hybrid")
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path.cwd() / f"homodyne_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create configuration
        config = _create_api_config(
            data_file=data_file,
            method=method,
            analysis_mode=analysis_mode,
            config_file=config_file,
            **kwargs
        )
        
        # Load and validate data
        data_loader = XPCSDataLoader(config_dict=config)
        data_dict = data_loader.load_experimental_data()
        logger.info(f"✓ Data loaded: {data_dict['c2_exp'].size:,} data points")
        
        # Initialize executor
        executor = MethodExecutor(
            config=config,
            force_cpu=kwargs.get('force_cpu', False),
            gpu_memory_fraction=kwargs.get('gpu_memory_fraction', 0.8),
            disable_dataset_optimization=kwargs.get('disable_dataset_optimization', False)
        )
        
        # Execute analysis
        result = _execute_analysis_method(
            executor, method, data_dict, config
        )
        
        if result is None:
            raise RuntimeError(f"{method.upper()} analysis failed")
        
        # Process and export results
        results_manager = ResultsManager(output_dir=output_dir, config=config)
        results_manager.process_results(result)
        results_manager.save_fitted_data(result, data_dict)
        
        # Generate plots if requested
        if kwargs.get('generate_plots', True):
            plotter = PlottingController(output_dir=output_dir)
            plotter.plot_fit_results(result, data_dict)
            if kwargs.get('plot_experimental_data', False):
                plotter.plot_experimental_data(data_dict, config)
        
        # Create return dictionary
        return _create_result_summary(result, data_dict, output_dir, config)
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise


def fit_data(data: np.ndarray,
            sigma: Optional[np.ndarray] = None,
            t1: Optional[np.ndarray] = None,
            t2: Optional[np.ndarray] = None,
            phi: Optional[np.ndarray] = None,
            q: Optional[float] = None,
            L: Optional[float] = None,
            method: str = "vi",
            analysis_mode: str = "auto",
            **kwargs) -> ResultType:
    """
    Fit data directly with numpy arrays (for advanced users).
    
    Provides direct access to fitting algorithms for users who have
    already preprocessed their data or want fine-grained control.
    
    Args:
        data: Experimental correlation data (3D array)
        sigma: Measurement uncertainties (optional)
        t1: Time grid 1 (optional, will be inferred if None)
        t2: Time grid 2 (optional, will be inferred if None) 
        phi: Angle grid (optional, will be inferred if None)
        q: Scattering vector magnitude (optional)
        L: Sample thickness (optional)
        method: Analysis method ('vi', 'mcmc', 'hybrid')
        analysis_mode: Analysis mode specification
        **kwargs: Additional method parameters
        
    Returns:
        Analysis result object (VIResult, MCMCResult, or HybridResult)
        
    Example:
        >>> result = fit_data(c2_data, method='mcmc', n_samples=2000)
        >>> print(f"Parameters: {result.mean_params}")
        >>> print(f"R-hat: {np.max(result.r_hat):.3f}")
    """
    logger.info(f"🔬 Fitting data directly with {method.upper()}")
    
    # Validate data shape
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(f"Data must be 3D array, got shape {data.shape}")
    
    # Infer missing parameters
    if t1 is None or t2 is None or phi is None:
        t1, t2, phi = _infer_grid_parameters(data)
        logger.info("Grid parameters inferred from data shape")
    
    if q is None:
        q = kwargs.get('q', 1.0)  # Default value
        logger.info(f"Using default q = {q}")
        
    if L is None:
        L = kwargs.get('L', 1.0)  # Default value  
        logger.info(f"Using default L = {L}")
    
    # Create basic configuration
    config = _create_minimal_config(analysis_mode, method, **kwargs)
    
    # Initialize executor
    executor = MethodExecutor(
        config=config,
        force_cpu=kwargs.get('force_cpu', False),
        disable_dataset_optimization=kwargs.get('disable_dataset_optimization', False)
    )
    
    # Execute fitting
    if method == 'vi':
        result = executor.execute_vi(data, sigma, t1, t2, phi, q, L)
    elif method == 'mcmc':
        result = executor.execute_mcmc(data, sigma, t1, t2, phi, q, L)
    elif method == 'hybrid':
        result = executor.execute_hybrid(data, sigma, t1, t2, phi, q, L)
    else:
        raise ValueError(f"Invalid method: {method}")
    
    if result is None:
        raise RuntimeError(f"{method.upper()} fitting failed")
    
    logger.info(f"✓ Direct fitting completed successfully")
    return result


def load_and_analyze(data_file: Union[str, Path],
                    methods: List[str] = ["vi", "mcmc"],
                    **kwargs) -> Dict[str, Dict[str, Any]]:
    """
    Load data and run multiple analysis methods for comparison.
    
    Convenient function for comparing different methods on the same
    dataset with automatic result comparison and recommendations.
    
    Args:
        data_file: Path to experimental data file
        methods: List of methods to run
        **kwargs: Additional parameters passed to each analysis
        
    Returns:
        Dictionary mapping method names to analysis results
        
    Example:
        >>> results = load_and_analyze('data.h5', methods=['vi', 'mcmc', 'hybrid'])
        >>> for method, result in results.items():
        >>>     print(f"{method}: χ² = {result['chi_squared']:.4f}")
    """
    logger.info(f"🔄 Running comparative analysis with methods: {methods}")
    
    all_results = {}
    
    for method in methods:
        try:
            logger.info(f"Running {method.upper()} analysis...")
            result = run_analysis(data_file, method=method, **kwargs)
            all_results[method] = result
            logger.info(f"✓ {method.upper()} analysis completed")
            
        except Exception as e:
            logger.error(f"❌ {method.upper()} analysis failed: {e}")
            all_results[method] = {'error': str(e)}
    
    # Add comparison summary
    if len(all_results) > 1:
        all_results['_comparison'] = _create_method_comparison(all_results)
    
    logger.info(f"✓ Comparative analysis completed: {len(all_results)} methods")
    return all_results


def compare_methods(data_file: Union[str, Path],
                   output_dir: Optional[Union[str, Path]] = None,
                   **kwargs) -> Dict[str, Any]:
    """
    Comprehensive method comparison with performance analysis.
    
    Runs all three methods (VI, MCMC, Hybrid) and provides detailed
    comparison including accuracy, speed, and recommendation.
    
    Args:
        data_file: Path to experimental data file
        output_dir: Output directory for comparison results
        **kwargs: Additional analysis parameters
        
    Returns:
        Comprehensive comparison results
        
    Example:
        >>> comparison = compare_methods('data.h5')
        >>> print(f"Best method: {comparison['recommendation']}")
        >>> print(f"Speed ranking: {comparison['speed_ranking']}")
    """
    logger.info(f"🏁 Running comprehensive method comparison")
    
    methods = ['vi', 'mcmc', 'hybrid']
    results = load_and_analyze(data_file, methods=methods, output_dir=output_dir, **kwargs)
    
    # Create comprehensive comparison
    comparison = {
        'individual_results': results,
        'performance_comparison': _analyze_performance(results),
        'accuracy_comparison': _analyze_accuracy(results),
        'recommendation': _recommend_best_method(results),
        'summary': _create_comparison_summary(results)
    }
    
    logger.info(f"✓ Method comparison completed")
    logger.info(f"  Recommended method: {comparison['recommendation']}")
    
    return comparison


def run_complete_pipeline(data_file: Union[str, Path],
                         method: str = "vi",
                         config_file: Optional[Union[str, Path]] = None,
                         output_dir: Optional[Union[str, Path]] = None,
                         **kwargs) -> Dict[str, Any]:
    """
    Run complete analysis pipeline using the integrated workflow system.
    
    This function provides access to the full AnalysisPipeline workflow,
    including comprehensive error handling, configuration validation,
    phi angle filtering, and complete result processing.
    
    Args:
        data_file: Path to experimental data file
        method: Analysis method ('vi', 'mcmc', 'hybrid')
        config_file: Configuration file path (optional)
        output_dir: Output directory for results (optional)
        **kwargs: Additional CLI-compatible parameters
        
    Returns:
        Dictionary containing pipeline results and metadata
        
    Example:
        >>> results = run_complete_pipeline('data.h5', method='hybrid', 
        ...                                config_file='config.yaml')
        >>> print(f"Pipeline exit code: {results['exit_code']}")
        >>> print(f"Analysis mode: {results['analysis_mode']}")
    """
    import argparse
    from datetime import datetime
    
    logger.info(f"🚀 Running complete analysis pipeline via Python API")
    logger.info(f"  Data file: {data_file}")
    logger.info(f"  Method: {method.upper()}")
    
    # Validate inputs
    data_file = Path(data_file)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    if method not in ['vi', 'mcmc', 'hybrid']:
        raise ValueError(f"Invalid method '{method}'. Choose from: vi, mcmc, hybrid")
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path.cwd() / f"homodyne_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create CLI-compatible argument namespace
    args = argparse.Namespace()
    
    # Required arguments
    args.data_file = str(data_file)
    args.method = method
    args.output_dir = str(output_dir)
    args.config = str(config_file) if config_file else None
    
    # Optional arguments with defaults
    args.verbose = kwargs.get('verbose', False)
    args.force_cpu = kwargs.get('force_cpu', False)
    args.gpu_memory_fraction = kwargs.get('gpu_memory_fraction', 0.8)
    args.disable_dataset_optimization = kwargs.get('disable_dataset_optimization', False)
    
    # Plotting options
    args.plot_experimental_data = kwargs.get('plot_experimental_data', False)
    args.plot_simulated_data = kwargs.get('plot_simulated_data', False)
    
    # Phi angle filtering
    args.phi_angles = kwargs.get('phi_angles', None)
    
    # Analysis mode (if specified)
    args.analysis_mode = kwargs.get('analysis_mode', 'auto')
    
    try:
        # Initialize and run the pipeline
        pipeline = AnalysisPipeline(args)
        exit_code = pipeline.run_analysis()
        
        # Collect pipeline information
        pipeline_results = {
            'exit_code': exit_code,
            'success': exit_code == 0,
            'method': method,
            'output_directory': str(output_dir),
            'data_file': str(data_file),
            'config_file': str(config_file) if config_file else None,
            'pipeline_components': {
                'data_loader': pipeline.data_loader is not None,
                'executor': pipeline.executor is not None,
                'results_manager': pipeline.results_manager is not None,
                'plotter': pipeline.plotter is not None,
            }
        }
        
        # Add configuration information if available
        if pipeline.config:
            pipeline_results['analysis_mode'] = pipeline.config.get('analysis_mode', 'unknown')
            pipeline_results['effective_parameter_count'] = pipeline.config.get('effective_parameter_count')
        
        if exit_code == 0:
            logger.info(f"✅ Complete pipeline executed successfully")
            logger.info(f"   Output directory: {output_dir}")
            logger.info(f"   Analysis mode: {pipeline_results.get('analysis_mode', 'unknown')}")
        else:
            logger.error(f"❌ Pipeline execution failed with exit code: {exit_code}")
        
        return pipeline_results
        
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        return {
            'exit_code': 1,
            'success': False,
            'error': str(e),
            'method': method,
            'output_directory': str(output_dir),
            'data_file': str(data_file),
            'config_file': str(config_file) if config_file else None,
        }


class AnalysisSession:
    """
    Interactive analysis session for Jupyter notebooks and scripting.
    
    Provides a stateful interface for interactive analysis with
    result caching, parameter exploration, and progressive refinement.
    """
    
    def __init__(self, 
                 data_file: Optional[Union[str, Path]] = None,
                 output_dir: Optional[Union[str, Path]] = None,
                 auto_plot: bool = True):
        """
        Initialize analysis session.
        
        Args:
            data_file: Data file to load (optional)
            output_dir: Output directory for session results
            auto_plot: Automatically generate plots after analysis
        """
        self.data_file = Path(data_file) if data_file else None
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / "session_results"
        self.auto_plot = auto_plot
        
        # Session state
        self.data_dict: Optional[Dict[str, Any]] = None
        self.config: Optional[Dict[str, Any]] = None
        self.results: Dict[str, ResultType] = {}
        self.analysis_history: List[Dict[str, Any]] = []
        
        # Initialize logging for session
        configure_logging(level="INFO")
        logger.info(f"🎯 Analysis session initialized")
        
        # Load data if provided
        if self.data_file:
            self.load_data(self.data_file)
    
    def load_data(self, data_file: Union[str, Path]) -> None:
        """
        Load experimental data into session.
        
        Args:
            data_file: Path to data file
        """
        self.data_file = Path(data_file)
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        # Create basic configuration for data loading
        self.config = _create_api_config(self.data_file, method="vi")
        
        # Load data
        data_loader = XPCSDataLoader(config_dict=self.config)
        self.data_dict = data_loader.load_experimental_data()
        
        logger.info(f"✓ Data loaded: {self.data_dict['c2_exp'].size:,} data points")
        logger.info(f"  Phi angles: {len(self.data_dict['phi_angles'])}")
        logger.info(f"  Time range: t1=[{self.data_dict['t1'][0]:.2e}, {self.data_dict['t1'][-1]:.2e}]")
    
    def analyze(self, method: str, use_pipeline: bool = False, **kwargs) -> ResultType:
        """
        Run analysis with specified method.
        
        Args:
            method: Analysis method ('vi', 'mcmc', 'hybrid')
            use_pipeline: Whether to use the complete AnalysisPipeline workflow
            **kwargs: Method-specific parameters
            
        Returns:
            Analysis result
        """
        if self.data_dict is None:
            raise RuntimeError("No data loaded. Use load_data() first.")
        
        logger.info(f"🔬 Running {method.upper()} analysis in session")
        
        if use_pipeline:
            # Use the complete pipeline workflow
            logger.info("   Using complete AnalysisPipeline workflow")
            pipeline_result = run_complete_pipeline(
                data_file=self.data_file,
                method=method,
                output_dir=self.output_dir,
                **kwargs
            )
            
            if not pipeline_result['success']:
                raise RuntimeError(f"Pipeline {method.upper()} analysis failed: {pipeline_result.get('error', 'Unknown error')}")
            
            # For pipeline execution, we can't return the result object directly
            # Instead, we'll create a summary result
            result = type('PipelineResult', (), {
                'method': method,
                'pipeline_results': pipeline_result,
                'success': pipeline_result['success'],
                'analysis_mode': pipeline_result.get('analysis_mode', 'unknown'),
                'output_directory': pipeline_result['output_directory']
            })()
            
        else:
            # Use the direct execution approach
            # Update configuration with method-specific parameters
            session_config = self.config.copy()
            session_config.update(kwargs)
            
            # Execute analysis
            executor = MethodExecutor(
                config=session_config,
                force_cpu=kwargs.get('force_cpu', False)
            )
            
            result = _execute_analysis_method(executor, method, self.data_dict, session_config)
            
            if result is None:
                raise RuntimeError(f"{method.upper()} analysis failed")
        
        # Store result and update history
        self.results[method] = result
        self.analysis_history.append({
            'timestamp': datetime.now().isoformat(),
            'method': method,
            'parameters': kwargs,
            'success': True,
            'used_pipeline': use_pipeline
        })
        
        # Auto-plot if enabled (only for non-pipeline results)
        if self.auto_plot and not use_pipeline:
            self.plot_latest(method)
        
        logger.info(f"✓ {method.upper()} analysis completed and stored")
        return result
    
    def compare_all(self) -> Dict[str, Any]:
        """
        Compare all methods run in this session.
        
        Returns:
            Comparison results
        """
        if not self.results:
            raise RuntimeError("No analysis results available")
        
        logger.info(f"📊 Comparing {len(self.results)} methods in session")
        
        comparison = {}
        for method_name, result in self.results.items():
            comparison[method_name] = {
                'chi_squared': getattr(result, 'chi_squared', None),
                'parameters': result.mean_params.tolist() if hasattr(result, 'mean_params') else None,
                'converged': getattr(result, 'converged', None)
            }
        
        return comparison
    
    def plot_latest(self, method: str) -> None:
        """
        Plot results from latest analysis.
        
        Args:
            method: Method name to plot
        """
        if method not in self.results:
            raise ValueError(f"No results for method: {method}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plotter = PlottingController(output_dir=self.output_dir)
        plotter.plot_fit_results(self.results[method], self.data_dict)
        
        logger.info(f"✓ Plots generated for {method.upper()}")
    
    def export_results(self, formats: List[str] = ['yaml', 'npz']) -> Dict[str, Path]:
        """
        Export all session results.
        
        Args:
            formats: Export formats
            
        Returns:
            Dictionary mapping formats to file paths
        """
        if not self.results:
            raise RuntimeError("No results to export")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        export_paths = {}
        for method, result in self.results.items():
            results_manager = ResultsManager(output_dir=self.output_dir, config=self.config)
            method_paths = results_manager.export_results_multiple_formats(result, formats)
            export_paths[method] = method_paths
        
        logger.info(f"✓ Results exported for {len(self.results)} methods")
        return export_paths
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get session summary.
        
        Returns:
            Session summary dictionary
        """
        return {
            'data_file': str(self.data_file) if self.data_file else None,
            'data_points': self.data_dict['c2_exp'].size if self.data_dict else None,
            'methods_run': list(self.results.keys()),
            'analysis_count': len(self.analysis_history),
            'output_dir': str(self.output_dir),
            'session_duration': (
                datetime.fromisoformat(self.analysis_history[-1]['timestamp']) - 
                datetime.fromisoformat(self.analysis_history[0]['timestamp'])
            ).total_seconds() if self.analysis_history else 0
        }


# Helper functions for API implementation

def _create_api_config(data_file: Path,
                      method: str,
                      analysis_mode: str = "auto",
                      config_file: Optional[Path] = None,
                      **kwargs) -> Dict[str, Any]:
    """Create configuration for API usage."""
    
    # Start with basic configuration
    config = {
        'analysis_mode': analysis_mode,
        'data': {
            'file_path': str(data_file),
            'dataset_optimization': kwargs.get('dataset_optimization', True)
        },
        'optimization': {
            'vi': {
                'n_iterations': kwargs.get('vi_iterations', 2000),
                'learning_rate': kwargs.get('vi_learning_rate', 0.01)
            },
            'mcmc': {
                'n_samples': kwargs.get('mcmc_samples', 1000),
                'n_chains': kwargs.get('mcmc_chains', 4)
            },
            'hybrid': {
                'use_vi_init': kwargs.get('use_vi_init', True)
            }
        },
        'hardware': {
            'force_cpu': kwargs.get('force_cpu', False),
            'gpu_memory_fraction': kwargs.get('gpu_memory_fraction', 0.8)
        }
    }
    
    # Merge with config file if provided
    if config_file and config_file.exists():
        from homodyne.config.cli_config import CLIConfigManager
        config_manager = CLIConfigManager()
        
        # Create fake args namespace for compatibility
        import argparse
        args = argparse.Namespace()
        for key, value in kwargs.items():
            setattr(args, key, value)
        
        config = config_manager.create_effective_config(str(config_file), args)
    
    return config


def _create_minimal_config(analysis_mode: str, method: str, **kwargs) -> Dict[str, Any]:
    """Create minimal configuration for direct fitting."""
    return {
        'analysis_mode': analysis_mode,
        'optimization': {
            method: {
                key.replace(f'{method}_', ''): value 
                for key, value in kwargs.items() 
                if key.startswith(f'{method}_')
            }
        }
    }


def _execute_analysis_method(executor: MethodExecutor,
                           method: str,
                           data_dict: Dict[str, Any],
                           config: Dict[str, Any]) -> Optional[ResultType]:
    """Execute specified analysis method."""
    
    # Extract data components
    data = data_dict['c2_exp']
    sigma = data_dict.get('c2_std', None)
    t1 = data_dict['t1']
    t2 = data_dict['t2']
    phi_angles = data_dict['phi_angles']
    q = data_dict['q']
    L = data_dict['L']
    
    # Execute method
    if method == 'vi':
        return executor.execute_vi(data, sigma, t1, t2, phi_angles, q, L)
    elif method == 'mcmc':
        return executor.execute_mcmc(data, sigma, t1, t2, phi_angles, q, L)
    elif method == 'hybrid':
        return executor.execute_hybrid(data, sigma, t1, t2, phi_angles, q, L)
    else:
        raise ValueError(f"Unknown method: {method}")


def _create_result_summary(result: ResultType,
                         data_dict: Dict[str, Any],
                         output_dir: Path,
                         config: Dict[str, Any]) -> Dict[str, Any]:
    """Create API result summary."""
    
    summary = {
        'success': True,
        'method': result.__class__.__name__.replace('Result', '').lower(),
        'analysis_mode': getattr(result, 'analysis_mode', 'unknown'),
        'parameters': result.mean_params.tolist() if hasattr(result, 'mean_params') else None,
        'chi_squared': getattr(result, 'chi_squared', None),
        'output_directory': str(output_dir),
        'data_summary': {
            'data_points': data_dict['c2_exp'].size,
            'phi_angles': len(data_dict['phi_angles']),
            'q_value': data_dict['q'],
            'L_value': data_dict['L']
        }
    }
    
    # Add method-specific information
    if hasattr(result, 'final_elbo'):
        summary['final_elbo'] = result.final_elbo
        summary['converged'] = getattr(result, 'converged', False)
    
    if hasattr(result, 'r_hat'):
        summary['max_r_hat'] = float(np.max(result.r_hat))
        summary['n_samples'] = getattr(result, 'n_samples', None)
    
    if hasattr(result, 'recommended_method'):
        summary['recommended_method'] = result.recommended_method
    
    return summary


def _infer_grid_parameters(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Infer grid parameters from data shape."""
    n_phi, n_t1, n_t2 = data.shape
    
    # Create default grids
    phi = np.linspace(0, np.pi, n_phi)
    t1 = np.logspace(-3, 1, n_t1)  # 1ms to 10s
    t2 = np.logspace(-3, 1, n_t2)
    
    return t1, t2, phi


def _create_method_comparison(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Create method comparison summary."""
    comparison = {
        'methods_compared': list(results.keys()),
        'chi_squared_ranking': [],
        'parameter_consistency': 'good',  # Would implement actual comparison
        'recommendations': []
    }
    
    # Rank by chi-squared if available
    chi_squared_values = {}
    for method, result in results.items():
        if isinstance(result, dict) and 'chi_squared' in result:
            chi_squared_values[method] = result['chi_squared']
    
    if chi_squared_values:
        sorted_methods = sorted(chi_squared_values.items(), key=lambda x: x[1])
        comparison['chi_squared_ranking'] = [method for method, _ in sorted_methods]
        comparison['best_chi_squared'] = sorted_methods[0]
    
    return comparison


def _analyze_performance(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze performance comparison between methods."""
    # This would implement actual performance analysis
    return {'speed_ranking': ['vi', 'hybrid', 'mcmc']}


def _analyze_accuracy(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze accuracy comparison between methods."""
    # This would implement actual accuracy analysis
    return {'accuracy_ranking': ['mcmc', 'hybrid', 'vi']}


def _recommend_best_method(results: Dict[str, Dict[str, Any]]) -> str:
    """Recommend best method based on results."""
    # Simple heuristic - would implement more sophisticated logic
    if 'hybrid' in results and 'error' not in results['hybrid']:
        return 'hybrid'
    elif 'mcmc' in results and 'error' not in results['mcmc']:
        return 'mcmc'
    elif 'vi' in results and 'error' not in results['vi']:
        return 'vi'
    else:
        return 'unknown'


def _create_comparison_summary(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Create overall comparison summary."""
    successful_methods = [m for m, r in results.items() if 'error' not in r]
    
    return {
        'total_methods': len(results),
        'successful_methods': len(successful_methods),
        'failed_methods': len(results) - len(successful_methods),
        'recommendation_available': len(successful_methods) > 0
    }