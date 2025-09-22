"""
Argument Parser for Homodyne v2 CLI
===================================

Comprehensive argument parsing system for modern CLI interface.
Enforces global constraints: no --method all, exact help strings.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from homodyne import __version__


def create_parser() -> argparse.ArgumentParser:
    """
    Create the main argument parser for Homodyne CLI.

    Returns:
        Configured ArgumentParser with all CLI options
    """
    parser = argparse.ArgumentParser(
        prog="homodyne",
        description="JAX-accelerated homodyne scattering analysis for XPCS under nonequilibrium conditions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  %(prog)s                                    # Run with default LSQ method
  %(prog)s --method lsq                       # Direct least squares fitting (default)
  %(prog)s --method mcmc                      # MCMC sampling for high accuracy
  %(prog)s --method hybrid                    # LSQ → MCMC pipeline (best of both)
  %(prog)s --config my_config.yaml            # Use custom config file
  %(prog)s --output-dir ./results --log-level DEBUG   # Custom output with debug logging
  %(prog)s --log-file                         # Save log messages to homodyne_results/logs/homodyne.log
  %(prog)s --no-console --log-file            # File logging only (no console output)
  %(prog)s --static-isotropic                 # Force static mode (3 parameters)
  %(prog)s --laminar-flow --method mcmc       # Force laminar flow (7 parameters) with MCMC
  %(prog)s --plot-experimental-data          # Generate data validation plots
  %(prog)s --plot-simulated-data              # Plot theoretical heatmaps
  %(prog)s --plot-simulated-data --contrast 1.5 --offset 0.1  # Custom scaling
  %(prog)s --phi-angles "0,45,90,135"         # Custom phi angles
  %(prog)s --estimate-noise                   # Automatic noise estimation (no sigma required)
  %(prog)s --estimate-noise --noise-model per_angle  # Per-angle noise for anisotropic data
  %(prog)s --method mcmc --estimate-noise     # Full Bayesian with joint noise+physics sampling
  %(prog)s --method lsq                       # Direct least squares (fastest, no uncertainty)
  %(prog)s --method lsq --estimate-noise      # LSQ with automatic noise estimation
  %(prog)s --method lsq --estimate-noise --noise-model per_angle  # Per-angle noise model
  %(prog)s --method lsq --max-iterations 5000 # LSQ with custom iteration limit
  %(prog)s --method hybrid --max-iterations 500 # Quick hybrid LSQ→MCMC optimization

Method Performance Characteristics:
  LSQ:     Direct classical least squares (fastest, good initial estimates)
          Use for: quick parameter estimation, real-time analysis, large datasets

  MCMC:    Full posterior sampling with uncertainty quantification
          Use for: publication-quality results, critical analysis

  Hybrid:  LSQ → MCMC pipeline combining speed and accuracy
          Use for: comprehensive analysis requiring both speed and precision
          
Physical Models:
  g₂(φ,t₁,t₂) = offset + contrast × [g₁(φ,t₁,t₂)]²
  
  Static Isotropic:     3 parameters [D₀, α, D_offset]
  Static Anisotropic:   3 parameters [D₀, α, D_offset] + angle filtering
  Laminar Flow:         7 parameters [D₀, α, D_offset, γ̇₀, β, γ̇_offset, φ₀] + filtering

Homodyne v{__version__} - Wei Chen, Hongrui He (Argonne National Laboratory)
        """,
    )

    # Version information
    parser.add_argument(
        "--version", action="version", version=f"Homodyne v{__version__}"
    )

    # Method selection - ONLY lsq, mcmc, hybrid (NO all per global constraints)
    parser.add_argument(
        "--method",
        choices=["lsq", "mcmc", "hybrid"],
        default="lsq",
        help="Optimization method: lsq (direct least squares), mcmc (accurate MCMC+JAX), hybrid (LSQ→MCMC pipeline) (default: %(default)s)",
    )

    # Configuration and I/O
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("./homodyne_config.yaml"),
        help="Path to configuration file (YAML or JSON) (default: %(default)s)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./homodyne_results"),
        help="Output directory for results and plots (default: %(default)s)",
    )

    # Logging control
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: %(default)s)",
    )

    parser.add_argument(
        "--no-console",
        action="store_true",
        help="Disable console logging (file logging remains enabled if --log-file specified)",
    )

    parser.add_argument(
        "--log-file",
        action="store_true",
        help="Save running log messages to homodyne.log in the results directory",
    )

    # Analysis mode selection - mutually exclusive group
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--static-isotropic",
        action="store_true",
        help="Force static isotropic analysis (3 parameters, no angle filtering)",
    )
    mode_group.add_argument(
        "--static-anisotropic",
        action="store_true",
        help="Force static anisotropic analysis (3 parameters + angle filtering)",
    )
    mode_group.add_argument(
        "--laminar-flow",
        action="store_true",
        help="7 parameters + filtering",  # Exact help string per global constraints
    )

    # Data visualization options
    parser.add_argument(
        "--plot-experimental-data",
        action="store_true",
        help="Generate validation plots of experimental data for quality checking",
    )

    parser.add_argument(
        "--plot-simulated-data",
        action="store_true",
        help="Plot theoretical C₂ heatmaps using parameters from config (no experimental data required)",
    )

    # Scaling parameters (only valid with --plot-simulated-data)
    parser.add_argument(
        "--contrast",
        type=float,
        default=0.3,
        help="Contrast parameter for scaling: fitted = contrast × theory + offset (default: %(default)s, requires --plot-simulated-data)",
    )

    parser.add_argument(
        "--offset",
        type=float,
        default=1.0,
        help="Offset parameter for scaling: fitted = contrast × theory + offset (default: %(default)s, requires --plot-simulated-data)",
    )

    # Custom phi angles
    parser.add_argument(
        "--phi-angles",
        type=str,
        help="Comma-separated list of phi angles in degrees (e.g., '0,45,90,135'). Default uses config file values",
    )

    # Cache generation mode
    parser.add_argument(
        "--save-cache-data",
        action="store_true",
        help="Generate optimized cache data from HDF5 file without analysis (selective q-vector + frame slicing)",
    )

    # Dataset optimization control
    parser.add_argument(
        "--disable-dataset-optimization",
        action="store_true",
        help="Disable automatic dataset size optimization (advanced users only)",
    )

    # GPU control
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU-only processing (disable GPU acceleration)",
    )

    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=0.8,
        help="Fraction of GPU memory to use (0.1-1.0, default: %(default)s)",
    )

    # Optimization control
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum optimization iterations (method-specific defaults: LSQ=10000, MCMC=2000, Hybrid=1000)",
    )

    # Noise estimation options
    parser.add_argument(
        "--estimate-noise",
        action="store_true",
        help="Enable hybrid NumPyro noise estimation instead of requiring sigma in data files",
    )

    parser.add_argument(
        "--noise-model",
        choices=["hierarchical", "per_angle", "adaptive"],
        default="hierarchical",
        help="Noise model for estimation: hierarchical (single global), per_angle (multi-angle), adaptive (heteroscedastic) (default: %(default)s)",
    )

    return parser


def add_development_args(parser: argparse.ArgumentParser) -> None:
    """
    Add development and debugging arguments.
    Only available in development builds.
    """
    dev_group = parser.add_argument_group("Development Options")

    dev_group.add_argument(
        "--profile", action="store_true", help="Enable performance profiling"
    )

    dev_group.add_argument(
        "--debug-jax",
        action="store_true",
        help="Enable JAX debugging and error checking",
    )

    dev_group.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate computation results",
    )


if __name__ == "__main__":
    # Quick test of parser
    parser = create_parser()
    args = parser.parse_args(["--help"] if len(sys.argv) == 1 else sys.argv[1:])
    print(f"Parsed args: {vars(args)}")
