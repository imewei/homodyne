"""Argument Parser for Homodyne v2 CLI
===================================

Simplified argument parsing system for the minimal CLI interface.
Focuses on essential commands while maintaining compatibility.

Note: GPU support removed in v2.3.0 - CPU-only execution.
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
        Configured ArgumentParser with essential CLI options (CPU-only)
    """
    # Create epilog with version interpolation
    epilog_text = f"""
Examples:
  %(prog)s                                    # Run with default NLSQ method
  %(prog)s --method nlsq                      # NLSQ trust-region least squares (default)
  %(prog)s --method mcmc                      # MCMC with automatic NUTS/CMC selection
  %(prog)s --config my_config.yaml            # Use custom config file
  %(prog)s --output-dir ./results             # Custom output directory
  %(prog)s --verbose                          # Enable verbose logging
  %(prog)s --plot-experimental-data          # Generate data validation plots
  %(prog)s --plot-simulated-data              # Plot theoretical heatmaps
  %(prog)s --plot-simulated-data --contrast 0.5 --offset 1.05  # Custom contrast/offset
  %(prog)s --phi-angles "0,45,90,135"         # Custom phi angles for simulated data
  %(prog)s --static-mode                      # Force static mode (3 parameters)
  %(prog)s --laminar-flow --method mcmc       # Force laminar flow (7 parameters) with MCMC

Manual NLSQ → MCMC Workflow:
  Step 1: Run NLSQ to get point estimates
    %(prog)s --method nlsq --config config.yaml

  Step 2: Manually copy best-fit parameters from NLSQ output

  Step 3: Update config.yaml with NLSQ results:
    initial_parameters:
      values: [1234.5, -1.234, 567.8]  # From NLSQ output

  Step 4: Run MCMC with initialized parameters
    %(prog)s --method mcmc --config config.yaml

Optimization Methods:
  nlsq:    NLSQ trust-region nonlinear least squares (PRIMARY)
          Use for: Fast, reliable parameter estimation

  mcmc:    Automatic NUTS/CMC selection based on dataset characteristics (SECONDARY)
          Use for: Uncertainty quantification, publication-quality analysis
          Automatic selection: (num_samples >= 15) OR (memory > 30%%) → CMC, else NUTS
          Configure thresholds in config file: min_samples_for_cmc, memory_threshold_pct

Physical Model:
  c₂(φ,t₁,t₂) = 1 + contrast × [c₁(φ,t₁,t₂)]²

  Static Mode:     3 parameters [D₀, α, D_offset]
  Laminar Flow:    7 parameters [D₀, α, D_offset, γ̇₀, β, γ̇_offset, φ₀]

Homodyne v{__version__} - CPU-Optimized JAX Architecture
        """

    parser = argparse.ArgumentParser(
        prog="homodyne",
        description="CPU-optimized homodyne scattering analysis for XPCS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog_text,
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
        choices=["nlsq", "mcmc"],
        default="nlsq",
        help=(
            "Optimization method: nlsq (NLSQ trust-region), mcmc (automatic NUTS/CMC selection). "
            "For MCMC, selection uses dual criteria: (num_samples >= min_samples_for_cmc) OR "
            "(memory > memory_threshold_pct). Control via config: optimization.mcmc.min_samples_for_cmc "
            "and optimization.mcmc.memory_threshold_pct (default: %(default)s)"
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
        default=None,
        help="Number of MCMC samples (default: from config or 1000)",
    )

    mcmc_group.add_argument(
        "--n-warmup",
        type=int,
        default=None,
        help="Number of MCMC warmup samples (default: from config or 500)",
    )

    mcmc_group.add_argument(
        "--n-chains",
        type=int,
        default=None,
        help="Number of MCMC chains (default: from config or 4)",
    )

    # CMC-specific options
    cmc_group = parser.add_argument_group(
        "Consensus Monte Carlo (CMC) Options",
        description="Options for CMC when automatically selected by --method mcmc. "
        "CMC is selected when: (num_samples >= 15) OR (memory > 30%). "
        "These options control CMC behavior when automatic selection chooses CMC.",
    )
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

    # Parameter override options
    override_group = parser.add_argument_group(
        "Parameter Override Options",
        description="Override initial parameter values and MCMC thresholds from config file. "
        "Priority: CLI args > config file > package defaults. "
        "Useful for exploratory analysis without modifying config files.",
    )

    # Initial parameter overrides (static mode: 3 parameters)
    override_group.add_argument(
        "--initial-d0",
        type=float,
        default=None,
        help="Override initial D0 (diffusion coefficient at t=1s, nm²/s) from config",
    )

    override_group.add_argument(
        "--initial-alpha",
        type=float,
        default=None,
        help="Override initial alpha (time-dependent diffusion exponent) from config",
    )

    override_group.add_argument(
        "--initial-d-offset",
        type=float,
        default=None,
        help="Override initial D_offset (constant offset diffusion, nm²/s) from config",
    )

    # Initial parameter overrides (laminar flow mode: 4 additional parameters)
    override_group.add_argument(
        "--initial-gamma-dot-t0",
        type=float,
        default=None,
        help="Override initial gamma_dot_t0 (shear rate at t=1s, s⁻¹) from config (laminar flow only)",
    )

    override_group.add_argument(
        "--initial-beta",
        type=float,
        default=None,
        help="Override initial beta (time-dependent shear exponent) from config (laminar flow only)",
    )

    override_group.add_argument(
        "--initial-gamma-dot-offset",
        type=float,
        default=None,
        help="Override initial gamma_dot_t_offset (constant offset shear rate, s⁻¹) from config (laminar flow only)",
    )

    override_group.add_argument(
        "--initial-phi0",
        type=float,
        default=None,
        help="Override initial phi0 (flow direction angle, radians) from config (laminar flow only)",
    )

    # MCMC threshold overrides
    override_group.add_argument(
        "--min-samples-cmc",
        type=int,
        default=None,
        help="Override min_samples_for_cmc threshold for CMC selection (default: 15)",
    )

    override_group.add_argument(
        "--memory-threshold-pct",
        type=float,
        default=None,
        help="Override memory_threshold_pct for CMC selection (0.0-1.0, default: 0.30)",
    )

    # MCMC/CMC mass matrix option
    override_group.add_argument(
        "--dense-mass-matrix",
        action="store_true",
        help="Use dense mass matrix for NUTS/CMC (default: diagonal). May improve sampling for correlated parameters.",
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
    if args.max_iterations <= 0:
        print("Error: Maximum iterations must be positive")
        return False

    if args.tolerance <= 0:
        print("Error: Tolerance must be positive")
        return False

    # Validate MCMC parameters if provided via CLI
    if args.n_samples is not None and args.n_samples <= 0:
        print("Error: MCMC samples must be positive")
        return False
    if args.n_warmup is not None and args.n_warmup <= 0:
        print("Error: MCMC warmup must be positive")
        return False
    if args.n_chains is not None and args.n_chains <= 0:
        print("Error: MCMC chains must be positive")
        return False

    # Validate CMC parameters
    if args.cmc_num_shards is not None and args.cmc_num_shards <= 0:
        print("Error: CMC num_shards must be positive")
        return False

    # Warn if CMC arguments provided with non-MCMC method
    if args.method not in ["mcmc"]:
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

    # Validate parameter override values
    if args.initial_d0 is not None and args.initial_d0 <= 0:
        print("Error: --initial-d0 must be positive")
        return False

    if args.min_samples_cmc is not None and args.min_samples_cmc <= 0:
        print("Error: --min-samples-cmc must be positive")
        return False

    if args.memory_threshold_pct is not None:
        if not (0.0 <= args.memory_threshold_pct <= 1.0):
            print("Error: --memory-threshold-pct must be between 0.0 and 1.0")
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
