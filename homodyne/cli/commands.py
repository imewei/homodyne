"""
Command Routing and Validation for Homodyne v2 CLI
==================================================

Routes CLI commands to appropriate handlers and manages execution flow.
Coordinates between configuration, data loading, and analysis execution.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from homodyne.utils.logging import get_logger
from homodyne.cli.validators import validate_args, ValidationError

logger = get_logger(__name__)


def dispatch_command(args: argparse.Namespace) -> int:
    """
    Dispatch CLI command to appropriate handler.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Exit code: 0 for success, 1 for error, 2 for invalid arguments
    """
    try:
        logger.debug("Dispatching command based on CLI arguments")
        
        # Check for plotting-only modes first
        if args.plot_simulated_data and not has_analysis_flags(args):
            return handle_plot_simulated_data(args)
        
        if args.plot_experimental_data and not has_analysis_flags(args):
            return handle_plot_experimental_data(args)
        
        # Main analysis workflow
        return handle_analysis_workflow(args)
        
    except ValidationError as e:
        logger.error(f"❌ Validation error: {e}")
        return 2
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error in command dispatch: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def has_analysis_flags(args: argparse.Namespace) -> bool:
    """
    Check if arguments indicate analysis should be performed.
    
    Args:
        args: Parsed arguments
        
    Returns:
        True if analysis should be performed
    """
    # Check for explicit analysis mode flags
    if any([
        args.static_isotropic, 
        args.static_anisotropic, 
        args.laminar_flow
    ]):
        return True
    
    # Check for non-default method selection (indicates explicit analysis intent)
    if args.method in ['mcmc', 'hybrid']:
        return True
        
    # If plotting flag is specified, check if it's plotting-only
    if args.plot_simulated_data or args.plot_experimental_data:
        return False  # Plotting-only mode
    
    # Default to analysis if config exists and no plotting flags
    return args.config.exists()


def handle_plot_simulated_data(args: argparse.Namespace) -> int:
    """
    Handle plotting simulated data without experimental data.
    
    Args:
        args: CLI arguments
        
    Returns:
        Exit code
    """
    try:
        logger.info("🎨 Plotting simulated data mode")
        
        # Import plotting controller
        from homodyne.workflows.plotting_controller import PlottingController
        from homodyne.config.cli_config import CLIConfigManager
        
        # Load and process configuration
        config_manager = CLIConfigManager()
        config = config_manager.create_effective_config(args.config, args)
        
        # Initialize plotting controller
        plotter = PlottingController(args.output_dir)
        
        # Generate simulated plots
        plotter.plot_simulated_data(
            config,
            contrast=args.contrast,
            offset=args.offset,
            phi_angles_str=args.phi_angles
        )
        
        logger.info("✓ Simulated data plots generated successfully")
        return 0
        
    except ImportError as e:
        logger.error(f"❌ Missing dependencies for plotting: {e}")
        return 1
    except Exception as e:
        logger.error(f"❌ Error generating simulated plots: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def handle_plot_experimental_data(args: argparse.Namespace) -> int:
    """
    Handle plotting experimental data for validation.
    
    Args:
        args: CLI arguments
        
    Returns:
        Exit code
    """
    try:
        logger.info("🎨 Plotting experimental data mode")
        
        # Import required modules
        from homodyne.workflows.plotting_controller import PlottingController
        from homodyne.config.cli_config import CLIConfigManager
        from homodyne.data.xpcs_loader import XPCSDataLoader
        
        # Load and process configuration
        config_manager = CLIConfigManager()
        config = config_manager.create_effective_config(args.config, args)
        
        # Load experimental data
        data_loader = XPCSDataLoader(config_dict=config)
        data_dict = data_loader.load_experimental_data()
        
        # Initialize plotting controller
        plotter = PlottingController(args.output_dir)
        
        # Generate experimental data plots
        plotter.plot_experimental_data(
            data_dict,
            config
        )
        
        logger.info("✓ Experimental data plots generated successfully")
        return 0
        
    except ImportError as e:
        logger.error(f"❌ Missing dependencies for plotting: {e}")
        return 1
    except Exception as e:
        logger.error(f"❌ Error plotting experimental data: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def handle_analysis_workflow(args: argparse.Namespace) -> int:
    """
    Handle main analysis workflow.
    
    Args:
        args: CLI arguments
        
    Returns:
        Exit code
    """
    try:
        logger.info(f"🔬 Starting {args.method.upper()} analysis workflow")
        
        # Import the main analysis pipeline
        from homodyne.workflows.pipeline import AnalysisPipeline
        
        # Create and run analysis pipeline
        pipeline = AnalysisPipeline(args)
        exit_code = pipeline.run_analysis()
        
        if exit_code == 0:
            logger.info("✓ Analysis workflow completed successfully")
        else:
            logger.error(f"❌ Analysis workflow failed with exit code {exit_code}")
        
        return exit_code
        
    except ImportError as e:
        logger.error(f"❌ Missing dependencies for analysis: {e}")
        logger.error("Please ensure all required packages are installed:")
        logger.error("  pip install 'homodyne[full]'")
        return 1
    except Exception as e:
        logger.error(f"❌ Error in analysis workflow: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def validate_mutual_exclusions(args: argparse.Namespace) -> None:
    """
    Validate mutually exclusive argument combinations.
    
    Args:
        args: Parsed arguments
        
    Raises:
        ValidationError: If mutually exclusive args are used together
    """
    # Verbose and quiet are mutually exclusive
    if args.verbose and args.quiet:
        raise ValidationError("Cannot use --verbose and --quiet together")
    
    # Scaling parameters only with simulated data plotting
    if (args.contrast != 1.0 or args.offset != 0.0) and not args.plot_simulated_data:
        raise ValidationError(
            "--contrast and --offset can only be used with --plot-simulated-data"
        )
    
    # GPU options with force CPU
    if args.force_cpu and args.gpu_memory_fraction != 0.8:
        logger.warning(
            "GPU memory fraction ignored when --force-cpu is specified"
        )


def print_method_info(method: str) -> None:
    """
    Print information about the selected method.
    
    Args:
        method: Selected optimization method
    """
    method_info = {
        "vi": {
            "name": "Variational Inference + JAX", 
            "speed": "Fast (10-100x speedup)",
            "accuracy": "Good approximate posterior",
            "use_case": "Routine analysis, parameter screening"
        },
        "mcmc": {
            "name": "MCMC + JAX (NumPyro/BlackJAX)",
            "speed": "Slower but thorough",
            "accuracy": "Full posterior sampling", 
            "use_case": "Publication-quality results"
        },
        "hybrid": {
            "name": "Hybrid VI → MCMC Pipeline",
            "speed": "Balanced (VI init + MCMC refinement)", 
            "accuracy": "Best of both approaches",
            "use_case": "Comprehensive analysis"
        }
    }
    
    if method in method_info:
        info = method_info[method]
        logger.info(f"Selected method: {info['name']}")
        logger.info(f"  Speed: {info['speed']}")
        logger.info(f"  Accuracy: {info['accuracy']}")
        logger.info(f"  Best for: {info['use_case']}")


def setup_output_directory(output_dir: Path) -> None:
    """
    Setup output directory structure.
    
    Args:
        output_dir: Path to output directory
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organization
        subdirs = ['plots', 'results', 'logs', 'configs']
        for subdir in subdirs:
            (output_dir / subdir).mkdir(exist_ok=True)
        
        logger.debug(f"✓ Output directory structure created: {output_dir}")
        
    except Exception as e:
        logger.warning(f"Could not create output directory structure: {e}")


def log_system_info(args: argparse.Namespace) -> None:
    """
    Log system and configuration information.
    
    Args:
        args: CLI arguments
    """
    try:
        import platform
        import homodyne
        
        logger.debug("=== System Information ===")
        logger.debug(f"Homodyne version: {homodyne.__version__}")
        logger.debug(f"Python version: {platform.python_version()}")
        logger.debug(f"Platform: {platform.platform()}")
        
        # JAX availability
        try:
            import jax
            logger.debug(f"JAX version: {jax.__version__}")
            logger.debug(f"JAX devices: {jax.devices()}")
        except ImportError:
            logger.debug("JAX not available - using NumPy fallback")
        
        # GPU status
        if not args.force_cpu:
            try:
                from homodyne.runtime.gpu.wrapper import check_gpu_availability
                gpu_info = check_gpu_availability()
                logger.debug(f"GPU availability: {gpu_info}")
            except ImportError:
                logger.debug("GPU detection not available")
        
        logger.debug("=== Configuration ===")
        logger.debug(f"Config file: {args.config}")
        logger.debug(f"Output directory: {args.output_dir}")
        logger.debug(f"Analysis method: {args.method}")
        
    except Exception as e:
        logger.debug(f"Error logging system info: {e}")


def handle_graceful_shutdown(signum, frame):
    """
    Handle graceful shutdown on signal.
    
    Args:
        signum: Signal number
        frame: Current stack frame
    """
    logger.info("Received shutdown signal - cleaning up...")
    sys.exit(1)