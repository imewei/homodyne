"""
Argument Parser for Homodyne v2 CLI
===================================

Simplified argument parsing system for the minimal CLI interface.
Focuses on essential commands while maintaining compatibility.
"""

import argparse
from pathlib import Path

# Version placeholder - would normally import from main package
__version__ = "2.1.0"


def create_parser() -> argparse.ArgumentParser:
    """
    Create the main argument parser for Homodyne CLI.

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
  %(prog)s --method nlsq                      # Optimistix nonlinear least squares (default)
  %(prog)s --method mcmc                      # MCMC sampling for uncertainty quantification
  %(prog)s --config my_config.yaml            # Use custom config file
  %(prog)s --output-dir ./results             # Custom output directory
  %(prog)s --force-cpu                        # Force CPU-only computation
  %(prog)s --verbose                          # Enable verbose logging

Optimization Methods:
  nlsq:    Optimistix trust-region nonlinear least squares (PRIMARY)
          Use for: Fast, reliable parameter estimation

  mcmc:    NumPyro/BlackJAX NUTS sampling (SECONDARY)
          Use for: Uncertainty quantification, publication-quality analysis

Physical Model:
  c₂(φ,t₁,t₂) = 1 + contrast × [c₁(φ,t₁,t₂)]²

  Static Mode:     3 parameters [D₀, α, D_offset]
  Laminar Flow:    7 parameters [D₀, α, D_offset, γ̇₀, β, γ̇_offset, φ₀]

Homodyne v{__version__} - JAX-First Architecture
        """,
    )

    # Version information
    parser.add_argument(
        "--version", action="version", version=f"Homodyne v{__version__}"
    )

    # Method selection - JAX-first methods only
    parser.add_argument(
        "--method",
        choices=["nlsq", "mcmc"],
        default="nlsq",
        help="Optimization method: nlsq (Optimistix trust-region), mcmc (NumPyro/BlackJAX NUTS) (default: %(default)s)",
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

    # Output options
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save result plots to output directory",
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
    """
    Validate parsed command-line arguments.

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

    # Check config file exists if provided and not default
    if args.config != Path("./homodyne_config.yaml") and not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}")
        return False

    # Check data file exists if provided
    if args.data_file and not args.data_file.exists():
        print(f"Error: Data file not found: {args.data_file}")
        return False

    return True
