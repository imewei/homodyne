"""Configuration handling for Homodyne CLI.

Manages device configuration, config file loading, CLI overrides,
and MCMC runtime kwargs construction.
"""

from __future__ import annotations

import argparse
from typing import Any

try:
    from homodyne._version import __version__ as _pkg_version
except ImportError:
    _pkg_version = "unknown"

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)

# Import core modules with fallback
try:
    from homodyne.config.manager import ConfigManager
    from homodyne.device import configure_optimal_device

    HAS_CORE_MODULES = True
except ImportError as e:
    HAS_CORE_MODULES = False
    logger.error(f"Core modules not available: {e}")


def _configure_device(args: argparse.Namespace) -> dict[str, Any]:
    """Configure optimal device based on CLI arguments."""

    logger.info("Configuring computational device...")

    # Configure CPU-only device (GPU support removed in v2.3.0)
    device_config = configure_optimal_device()

    if device_config.get("configuration_successful"):
        device_type = device_config.get("device_type", "")
        logger.info(f"OK: Device configured: {str(device_type).upper()}")
    else:
        logger.warning("Device configuration failed, using defaults")

    return device_config


def _load_configuration(args: argparse.Namespace) -> ConfigManager:
    """Load configuration from file or create default."""
    import yaml

    logger.info(f"Loading configuration from: {args.config}")

    try:
        # Try to load from file
        if args.config.exists():
            config = ConfigManager(str(args.config))
            logger.info(f"OK: Configuration loaded: {args.config}")
        else:
            # Create default configuration
            logger.info("Configuration file not found, using defaults")
            config = ConfigManager(config_override=_get_default_config(args))

        # Apply CLI overrides
        _apply_cli_overrides(config, args)

        return config

    except (ValueError, KeyError, OSError, yaml.YAMLError) as e:
        logger.warning(f"Configuration loading failed: {e}, using defaults")
        return ConfigManager(config_override=_get_default_config(args))


def _get_default_config(args: argparse.Namespace) -> dict[str, Any]:
    """Create default configuration from CLI arguments."""
    # Determine analysis mode
    if args.static_mode:
        analysis_mode = "static_isotropic"
    elif args.laminar_flow:
        analysis_mode = "laminar_flow"
    else:
        analysis_mode = "auto_detect"

    config = {
        "metadata": {
            "config_version": _pkg_version,
            "description": "CLI-generated configuration",
        },
        "analysis_mode": analysis_mode,
        "experimental_data": {
            "file_path": str(args.data_file) if args.data_file else None,
        },
        "optimization": {
            "method": args.method,
            "lsq": {
                "max_iterations": args.max_iterations,
                "tolerance": args.tolerance,
            },
            "mcmc": {
                "n_samples": args.n_samples,
                "n_warmup": args.n_warmup,
                "n_chains": args.n_chains,
            },
        },
        "hardware": {},
        "output": {
            "formats": [args.output_format],
            "save_plots": args.save_plots,
            "output_dir": str(args.output_dir),
        },
    }

    return config


def _apply_cli_overrides(config: ConfigManager, args: argparse.Namespace) -> None:
    """Apply CLI argument overrides to configuration.

    Implements precedence: CLI args > Config file > Code defaults
    For MCMC parameters, config uses 'num_*' prefix, args use 'n_*' prefix

    Supports overriding:
    - Data file path
    - Analysis mode
    - MCMC sampling parameters (n_samples, n_warmup, n_chains)
    - CMC sharding/backend parameters
    - Initial parameter values (D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0)
    - Mass matrix type (dense_mass_matrix flag)
    """
    if not hasattr(config, "config") or not config.config:
        return

    # Override data file if provided
    if args.data_file:
        config.config.setdefault("experimental_data", {})
        config.config["experimental_data"]["file_path"] = str(args.data_file)

    # Override analysis mode if specified
    if args.static_mode:
        config.config["analysis_mode"] = "static_isotropic"
    elif args.laminar_flow:
        config.config["analysis_mode"] = "laminar_flow"

    # Override optimization parameters
    if "optimization" not in config.config:
        config.config["optimization"] = {}

    config.config["optimization"]["method"] = args.method

    # Load MCMC parameters from config if not provided via CLI
    mcmc_config = config.config.get("optimization", {}).get("mcmc", {})

    if args.n_samples is None:
        args.n_samples = mcmc_config.get("num_samples", 1000)
    if args.n_warmup is None:
        args.n_warmup = mcmc_config.get("num_warmup", 500)
    if args.n_chains is None:
        args.n_chains = mcmc_config.get("num_chains", 4)

    # CMC-only: ensure sections exist and CLI overrides take precedence
    optimization_section = config.config.setdefault("optimization", {})
    optimization_section.setdefault("mcmc", {})
    cmc_section = optimization_section.setdefault("cmc", {})
    sharding_section = cmc_section.setdefault("sharding", {})
    per_shard = cmc_section.setdefault("per_shard_mcmc", {})

    if args.cmc_num_shards is not None:
        sharding_section["num_shards"] = args.cmc_num_shards

    per_shard["num_samples"] = args.n_samples
    per_shard["num_warmup"] = args.n_warmup
    per_shard["num_chains"] = args.n_chains

    # Override dense mass matrix flag
    if args.dense_mass_matrix:
        old_value = config.config["optimization"]["mcmc"].get("dense_mass", False)
        config.config["optimization"]["mcmc"]["dense_mass"] = True
        logger.info(
            f"Overriding config dense_mass={old_value} with CLI flag dense_mass=True"
        )

    # Override initial parameter values
    param_overrides = {
        "initial_d0": "D0",
        "initial_alpha": "alpha",
        "initial_d_offset": "D_offset",
        "initial_gamma_dot_t0": "gamma_dot_t0",
        "initial_beta": "beta",
        "initial_gamma_dot_offset": "gamma_dot_t_offset",
        "initial_phi0": "phi0",
    }

    cli_param_values = {}
    for arg_name, param_name in param_overrides.items():
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            cli_param_values[param_name] = arg_value

    if cli_param_values:
        if "initial_parameters" not in config.config:
            config.config["initial_parameters"] = {}

        param_names = config.config["initial_parameters"].get("parameter_names", [])
        param_values = config.config["initial_parameters"].get("values", [])

        if param_values is None:
            param_values = []

        if param_names and param_values and len(param_names) == len(param_values):
            from homodyne.config.parameter_manager import ParameterManager

            pm = ParameterManager(
                config.config, config.config.get("analysis_mode", "laminar_flow")
            )
            current_params = {}
            for pname, pval in zip(param_names, param_values, strict=False):
                canonical_name = pm._param_name_mapping.get(pname, pname)
                current_params[canonical_name] = pval
        else:
            current_params = {}

        for param_name, new_value in cli_param_values.items():
            old_value = current_params.get(param_name, None)
            current_params[param_name] = new_value

            if old_value is not None:
                logger.info(
                    f"Overriding config {param_name}={old_value:.6g} with CLI value {param_name}={new_value:.6g}"
                )
            else:
                logger.info(
                    f"Setting {param_name}={new_value:.6g} from CLI (not in config)"
                )

        analysis_mode = config.config.get("analysis_mode", "laminar_flow")
        if "static" in analysis_mode.lower():
            expected_params = ["D0", "alpha", "D_offset"]
        else:
            expected_params = [
                "D0",
                "alpha",
                "D_offset",
                "gamma_dot_t0",
                "beta",
                "gamma_dot_t_offset",
                "phi0",
            ]

        final_param_names = []
        final_param_values = []
        for param in expected_params:
            if param in current_params:
                final_param_names.append(param)
                final_param_values.append(current_params[param])

        config.config["initial_parameters"]["parameter_names"] = final_param_names
        config.config["initial_parameters"]["values"] = final_param_values

    if "hardware" not in config.config:
        config.config["hardware"] = {}


def _build_mcmc_runtime_kwargs(
    args: argparse.Namespace, config: ConfigManager
) -> dict[str, Any]:
    """Collect runtime kwargs for fit_mcmc_jax from CLI args and YAML config."""

    cfg_dict = config.config if hasattr(config, "config") else {}
    optimization_cfg = cfg_dict.get("optimization", {}) if cfg_dict else {}
    mcmc_cfg = optimization_cfg.get("mcmc", {}) if optimization_cfg else {}

    runtime_kwargs: dict[str, Any] = {
        "n_samples": args.n_samples,
        "n_warmup": args.n_warmup,
        "n_chains": args.n_chains,
        "run_id": getattr(args, "run_id", None),
    }

    if cfg_dict:
        runtime_kwargs["config"] = cfg_dict

    def _set_runtime_value(dest: str, *aliases: str) -> None:
        for key in (dest, *aliases):
            if key in mcmc_cfg and mcmc_cfg[key] is not None:
                runtime_kwargs[dest] = mcmc_cfg[key]
                return

    _set_runtime_value("target_accept_prob")
    _set_runtime_value("max_tree_depth")
    _set_runtime_value("dense_mass_matrix", "dense_mass")
    _set_runtime_value("rng_key")
    _set_runtime_value("stable_prior_fallback")
    _set_runtime_value("min_ess")
    _set_runtime_value("max_rhat")
    _set_runtime_value("n_retries")
    _set_runtime_value("check_hmc_diagnostics")

    if args.dense_mass_matrix:
        runtime_kwargs["dense_mass_matrix"] = True

    return runtime_kwargs
