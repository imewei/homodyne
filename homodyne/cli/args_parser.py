"""Argument Parser for Homodyne v2 CLI
===================================

Simplified argument parsing system for the minimal CLI interface.
Focuses on essential commands while maintaining compatibility.
"""

import argparse
from pathlib import Path

# Import version from package (with fallback for development)
try:
    from homodyne._version import __version__
except ImportError:
    __version__ = "unknown"


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser for Homodyne CLI.

    Returns:
        Configured ArgumentParser with essential CLI options
    """
    parser = argparse.ArgumentParser(
        prog="homodyne",
        description="JAX-first homodyne scattering analysis for XPCS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  %(prog)s                                    # Run with default NLSQ method
  %(prog)s --method nlsq                      # NLSQ trust-region least squares (default)
  %(prog)s --method mcmc                      # MCMC with automatic NUTS/CMC selection
  %(prog)s --method auto                      # Auto-select NUTS or CMC (same as mcmc)
  %(prog)s --method nuts                      # Force standard NUTS (for datasets < 500k points)
  %(prog)s --method cmc                       # Force CMC (for datasets > 500k points)
  %(prog)s --config my_config.yaml            # Use custom config file
  %(prog)s --output-dir ./results             # Custom output directory
  %(prog)s --force-cpu                        # Force CPU-only computation
  %(prog)s --verbose                          # Enable verbose logging
  %(prog)s --plot-experimental-data          # Generate data validation plots
  %(prog)s --plot-simulated-data              # Plot theoretical heatmaps
  %(prog)s --plot-simulated-data --contrast 0.5 --offset 1.05  # Custom contrast/offset
  %(prog)s --phi-angles "0,45,90,135"         # Custom phi angles for simulated data
  %(prog)s --static-mode                      # Force static mode (3 parameters)
  %(prog)s --laminar-flow --method mcmc       # Force laminar flow (7 parameters) with MCMC

CMC Examples:
  %(prog)s --method cmc --cmc-num-shards 20                    # CMC with 20 shards
  %(prog)s --method cmc --cmc-backend multiprocessing          # CMC with multiprocessing backend
  %(prog)s --method cmc --cmc-plot-diagnostics                 # CMC with diagnostic plots
  %(prog)s --method cmc --cmc-num-shards 16 --cmc-backend pjit # CMC on multi-GPU system

Optimization Methods:
  nlsq:    NLSQ trust-region nonlinear least squares (PRIMARY)
          Use for: Fast, reliable parameter estimation

  mcmc:    Alias for 'auto' - automatic NUTS/CMC selection (SECONDARY)
          Use for: Uncertainty quantification, publication-quality analysis

  auto:    Automatic selection between NUTS and CMC based on dataset size
          < 500k points → NUTS (single-device MCMC)
          > 500k points → CMC (distributed MCMC with data sharding)

  nuts:    Force standard NUTS MCMC (NumPyro/BlackJAX)
          Use for: Datasets < 500k points, when CMC overhead is unnecessary

  cmc:     Force Consensus Monte Carlo (distributed Bayesian inference)
          Use for: Large datasets (> 500k points), multi-GPU/HPC systems

Physical Model:
  c₂(φ,t₁,t₂) = 1 + contrast × [c₁(φ,t₁,t₂)]²

  Static Mode:     3 parameters [D₀, α, D_offset]
  Laminar Flow:    7 parameters [D₀, α, D_offset, γ̇₀, β, γ̇_offset, φ₀]

Homodyne v{__version__} - JAX-First Architecture
        """,
    )

    # Version information
    parser.add_argument(
        "--version",
        action="version",
        version=f"Homodyne v{__version__}",
    )

    # Method selection - JAX-first methods only
    parser.add_argument(
        "--method",
        choices=["nlsq", "mcmc", "nuts", "cmc", "auto"],
        default="nlsq",
        help=(
            "Optimization method: nlsq (NLSQ trust-region), mcmc (alias for auto), "
            "auto (automatic NUTS/CMC selection), nuts (force NUTS), cmc (force CMC) "
            "(default: %(default)s)"
        ),
    )

    # Configuration and I/O
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("./homodyne_config.yaml"),
        help="Path to configuration file (YAML) (default: %(default)s)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./homodyne_results"),
        help="Output directory for results (default: %(default)s)",
    )

    # Data file
    parser.add_argument(
        "--data-file",
        type=Path,
        help="Path to experimental data file (overrides config)",
    )

    # Device configuration
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU-only computation (disable GPU acceleration)",
    )

    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use (0.1-1.0) (default: %(default)s)",
    )

    # Analysis mode
    parser.add_argument(
        "--static-mode",
        action="store_true",
        help="Force static analysis mode (3 parameters)",
    )

    parser.add_argument(
        "--laminar-flow",
        action="store_true",
        help="Force laminar flow analysis mode (7 parameters)",
    )

    # NLSQ-specific options
    nlsq_group = parser.add_argument_group("NLSQ Options")
    nlsq_group.add_argument(
        "--max-iterations",
        type=int,
        default=10000,
        help="Maximum NLSQ iterations (default: %(default)s)",
    )

    nlsq_group.add_argument(
        "--tolerance",
        type=float,
        default=1e-8,
        help="NLSQ convergence tolerance (default: %(default)s)",
    )

    # MCMC-specific options
    mcmc_group = parser.add_argument_group("MCMC Options")
    mcmc_group.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of MCMC samples (default: %(default)s)",
    )

    mcmc_group.add_argument(
        "--n-warmup",
        type=int,
        default=1000,
        help="Number of MCMC warmup samples (default: %(default)s)",
    )

    mcmc_group.add_argument(
        "--n-chains",
        type=int,
        default=4,
        help="Number of MCMC chains (default: %(default)s)",
    )

    # CMC-specific options
    cmc_group = parser.add_argument_group("Consensus Monte Carlo (CMC) Options")
    cmc_group.add_argument(
        "--cmc-num-shards",
        type=int,
        default=None,
        help="Number of data shards for CMC (overrides config, default: auto-detect based on dataset size)",
    )

    cmc_group.add_argument(
        "--cmc-backend",
        choices=["auto", "pjit", "multiprocessing", "pbs"],
        default=None,
        help="CMC backend for parallel execution (overrides config, default: auto-detect based on hardware)",
    )

    cmc_group.add_argument(
        "--cmc-plot-diagnostics",
        action="store_true",
        help="Generate CMC diagnostic plots (per-shard convergence, between-shard consistency)",
    )

    # Output options
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save result plots to output directory",
    )

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

    parser.add_argument(
        "--plotting-backend",
        type=str,
        choices=["auto", "matplotlib", "datashader"],
        default="auto",
        help="Plotting backend: auto (use Datashader if available), matplotlib (slower), datashader (5-10x faster, requires datashader package) (default: %(default)s)",
    )

    parser.add_argument(
        "--parallel-plots",
        action="store_true",
        help="Generate plots in parallel using multiprocessing (faster for multiple angles, requires Datashader backend)",
    )

    # Simulated data parameters (only valid with --plot-simulated-data)
    parser.add_argument(
        "--contrast",
        type=float,
        default=0.3,
        help="Contrast parameter for simulated data: c₂ = 1 + contrast × c₁² (default: %(default)s, requires --plot-simulated-data)",
    )

    parser.add_argument(
        "--offset",
        type=float,
        default=1.0,
        help="Offset parameter for simulated data: c₂ = offset + contrast × c₁² (default: %(default)s, requires --plot-simulated-data)",
    )

    parser.add_argument(
        "--phi-angles",
        type=str,
        help="Comma-separated list of phi angles in degrees (e.g., '0,45,90,135'). Default uses config file values or evenly spaced angles",
    )

    parser.add_argument(
        "--output-format",
        choices=["yaml", "json", "npz"],
        default="yaml",
        help="Output format for results (default: %(default)s)",
    )

    # Logging
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all output except errors",
    )

    return parser


def validate_args(args) -> bool:
    """Validate parsed command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments

    Returns
    -------
    bool
        True if arguments are valid, False otherwise
    """
    # Check for conflicting analysis modes
    if args.static_mode and args.laminar_flow:
        print("Error: Cannot specify both --static-mode and --laminar-flow")
        return False

    # Check for conflicting logging options
    if args.verbose and args.quiet:
        print("Error: Cannot specify both --verbose and --quiet")
        return False

    # Validate numeric ranges
    if not (0.1 <= args.gpu_memory_fraction <= 1.0):
        print("Error: GPU memory fraction must be between 0.1 and 1.0")
        return False

    if args.max_iterations <= 0:
        print("Error: Maximum iterations must be positive")
        return False

    if args.tolerance <= 0:
        print("Error: Tolerance must be positive")
        return False

    if args.n_samples <= 0 or args.n_warmup <= 0 or args.n_chains <= 0:
        print("Error: MCMC parameters (samples, warmup, chains) must be positive")
        return False

    # Validate CMC parameters
    if args.cmc_num_shards is not None and args.cmc_num_shards <= 0:
        print("Error: CMC num_shards must be positive")
        return False

    # Warn if CMC arguments provided with non-MCMC method
    if args.method not in ["mcmc", "auto", "nuts", "cmc"]:
        if args.cmc_num_shards is not None:
            print(
                f"Warning: --cmc-num-shards ignored (not applicable for method={args.method})"
            )
        if args.cmc_backend is not None:
            print(
                f"Warning: --cmc-backend ignored (not applicable for method={args.method})"
            )
        if args.cmc_plot_diagnostics:
            print(
                f"Warning: --cmc-plot-diagnostics ignored (not applicable for method={args.method})"
            )

    # Check config file exists if provided and not default
    if args.config != Path("./homodyne_config.yaml") and not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}")
        return False

    # Check data file exists if provided
    if args.data_file and not args.data_file.exists():
        print(f"Error: Data file not found: {args.data_file}")
        return False

    return True
