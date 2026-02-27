#!/usr/bin/env python3
"""Post-installation hook for Homodyne with Optional Shell Completion System
========================================================================

This script provides optional setup for:
1. Shell completion system (bash, zsh, fish) - user choice
2. Virtual environment integration (conda, mamba, venv, virtualenv)

Version 2.3.0: CPU-only architecture (GPU support removed)

Features:
- Safe completion scripts that don't interfere with system commands
- Cross-platform support: bash, zsh, fish
- Optional installation - user can choose what to install
- Easy removal with homodyne-cleanup
- Robust error handling and graceful degradation
"""

import argparse
import logging
import os
import platform
import sys
from pathlib import Path

logger = logging.getLogger("homodyne.post_install")


def is_linux() -> bool:
    """Check if running on Linux."""
    return platform.system() == "Linux"


def detect_shell_type() -> str:
    """Detect the current shell type."""
    shell = os.environ.get("SHELL", "")
    if "zsh" in shell:
        return "zsh"
    elif "bash" in shell:
        return "bash"
    elif "fish" in shell:
        return "fish"
    else:
        return "bash"  # Default fallback


def is_virtual_environment() -> bool:
    """Check if running in a virtual environment."""
    return (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        or os.environ.get("CONDA_DEFAULT_ENV") is not None
        or os.environ.get("MAMBA_ROOT_PREFIX") is not None
        or os.environ.get("VIRTUAL_ENV") is not None
    )


def is_conda_environment(venv_path: Path) -> bool:
    """Check if the environment is a conda/mamba environment."""
    # Check for conda directory structure
    conda_meta = venv_path / "conda-meta"
    # Check if path contains conda/mamba/miniforge/mambaforge
    path_indicators = ["conda", "mamba", "miniforge", "mambaforge"]
    return conda_meta.exists() or any(
        indicator in str(venv_path).lower() for indicator in path_indicators
    )


def create_unified_zsh_completion(venv_path: Path) -> Path:
    """Create the unified zsh completion file."""
    zsh_dir = venv_path / "etc" / "zsh"
    zsh_dir.mkdir(parents=True, exist_ok=True)

    completion_file = zsh_dir / "homodyne-completion.zsh"
    completion_content = """#!/usr/bin/env zsh
# Homodyne Zsh aliases

# Only load if not already loaded
if [[ -z "$_HOMODYNE_ZSH_COMPLETION_LOADED" ]]; then
    export _HOMODYNE_ZSH_COMPLETION_LOADED=1

    # Base command aliases
    alias hm='homodyne'
    alias hconfig='homodyne-config'

    # Method aliases (hm- prefix)
    alias hm-nlsq='homodyne --method nlsq'    # NLSQ trust-region optimization (primary)
    alias hm-cmc='homodyne --method cmc'    # Consensus Monte Carlo

    # Config mode aliases (hc- prefix)
    alias hc-stat='homodyne-config --mode static'         # Generate static mode config
    alias hc-flow='homodyne-config --mode laminar_flow'   # Generate laminar_flow mode config

    # Plotting shortcuts
    alias hexp='homodyne --plot-experimental-data'
    alias hsim='homodyne --plot-simulated-data'

    # Helper function
    homodyne_help() {
        echo "Homodyne command shortcuts:"
        echo "  hm       = homodyne"
        echo "  hconfig  = homodyne-config"
        echo ""
        echo "Method shortcuts (hm- prefix):"
        echo "  hm-nlsq  = homodyne --method nlsq  # NLSQ trust-region (primary)"
        echo "  hm-cmc  = homodyne --method cmc  # Consensus Monte Carlo"
        echo ""
        echo "Config mode shortcuts (hc- prefix):"
        echo "  hc-stat  = homodyne-config --mode static"
        echo "  hc-flow  = homodyne-config --mode laminar_flow"
        echo ""
        echo "Plotting shortcuts:"
        echo "  hexp     = homodyne --plot-experimental-data"
        echo "  hsim     = homodyne --plot-simulated-data"
    }
fi"""

    completion_file.write_text(completion_content)
    return completion_file


def copy_full_completion_script(venv_path: Path) -> Path | None:
    """Copy the full completion.sh into the venv for sourcing on activation.

    This provides tab completion, aliases, and interactive functions — not just
    the minimal aliases from create_unified_zsh_completion().
    """
    try:
        import homodyne

        homodyne_pkg = Path(homodyne.__file__).parent
        src = homodyne_pkg / "runtime" / "shell" / "completion.sh"

        if not src.exists():
            return None

        dest_dir = venv_path / "etc" / "homodyne" / "shell"
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest = dest_dir / "completion.sh"
        dest.write_text(src.read_text())
        dest.chmod(0o644)
        return dest

    except OSError:
        return None


def create_fish_completion(venv_path: Path) -> Path | None:
    """Create a fish-specific completion file with proper fish syntax."""
    try:
        fish_dir = venv_path / "etc" / "homodyne" / "shell"
        fish_dir.mkdir(parents=True, exist_ok=True)

        fish_file = fish_dir / "completion.fish"
        fish_content = """\
# Homodyne Fish Shell Completion
# Auto-generated by homodyne-post-install

if not set -q _HOMODYNE_FISH_COMPLETION_LOADED
    set -gx _HOMODYNE_FISH_COMPLETION_LOADED 1

    # Base command aliases
    alias hm 'homodyne'
    alias hconfig 'homodyne-config'

    # Method aliases (hm- prefix)
    alias hm-nlsq 'homodyne --method nlsq'
    alias hm-cmc 'homodyne --method cmc'

    # Config mode aliases (hc- prefix)
    alias hc-stat 'homodyne-config --mode static'
    alias hc-flow 'homodyne-config --mode laminar_flow'

    # Plotting shortcuts
    alias hexp 'homodyne --plot-experimental-data'
    alias hsim 'homodyne --plot-simulated-data'

    # Tool aliases
    alias hxla 'homodyne-config-xla'
    alias hsetup 'homodyne-post-install'
    alias hclean 'homodyne-cleanup'

    # Help function
    function homodyne_help
        echo "Homodyne Shell Shortcuts"
        echo "========================"
        echo ""
        echo "Base Commands:"
        echo "  hm                    homodyne"
        echo "  hconfig               homodyne-config"
        echo ""
        echo "Method Aliases (hm- prefix):"
        echo "  hm-nlsq               homodyne --method nlsq (primary)"
        echo "  hm-cmc                homodyne --method cmc (uncertainty)"
        echo ""
        echo "Config Aliases (hc- prefix):"
        echo "  hc-stat               homodyne-config --mode static"
        echo "  hc-flow               homodyne-config --mode laminar_flow"
        echo ""
        echo "Utility Aliases:"
        echo "  hexp                  homodyne --plot-experimental-data"
        echo "  hsim                  homodyne --plot-simulated-data"
        echo ""
        echo "Tool Aliases:"
        echo "  hxla                  homodyne-config-xla"
        echo "  hsetup                homodyne-post-install"
        echo "  hclean                homodyne-cleanup"
    end

    # Tab completion for homodyne
    complete -c homodyne -l help -d 'Show help message'
    complete -c homodyne -l version -d 'Show version information'
    complete -c homodyne -l method -xa 'nlsq cmc' -d 'Optimization method'
    complete -c homodyne -l config -rF -d 'Configuration file (YAML)'
    complete -c homodyne -l output-dir -xa '(__fish_complete_directories)' -d 'Output directory'
    complete -c homodyne -l data-file -rF -d 'Experimental data file'
    complete -c homodyne -l output-format -xa 'yaml json npz' -d 'Output format'
    complete -c homodyne -l verbose -d 'Enable verbose logging'
    complete -c homodyne -l quiet -d 'Suppress all output except errors'
    complete -c homodyne -l static-mode -d 'Force static analysis mode'
    complete -c homodyne -l laminar-flow -d 'Force laminar flow mode'
    complete -c homodyne -l max-iterations -x -d 'Maximum NLSQ iterations'
    complete -c homodyne -l tolerance -x -d 'NLSQ convergence tolerance'
    complete -c homodyne -l n-samples -x -d 'CMC samples per chain'
    complete -c homodyne -l n-warmup -x -d 'CMC warmup samples'
    complete -c homodyne -l n-chains -x -d 'Number of CMC chains'
    complete -c homodyne -l cmc-num-shards -xa '4 8 10 16 20 32' -d 'Data shards for CMC'
    complete -c homodyne -l cmc-backend -xa 'auto pjit multiprocessing pbs' -d 'CMC parallel backend'
    complete -c homodyne -l no-nlsq-warmstart -d 'Disable NLSQ warm-start for CMC'
    complete -c homodyne -l nlsq-result -xa '(__fish_complete_directories)' -d 'Pre-computed NLSQ results'
    complete -c homodyne -l dense-mass-matrix -d 'Use dense mass matrix for NUTS'
    complete -c homodyne -l save-plots -d 'Save result plots'
    complete -c homodyne -l plot-experimental-data -d 'Generate data validation plots'
    complete -c homodyne -l plot-simulated-data -d 'Plot theoretical C2 heatmaps'
    complete -c homodyne -l plotting-backend -xa 'auto matplotlib datashader' -d 'Plotting backend'
    complete -c homodyne -l parallel-plots -d 'Generate plots in parallel'

    # Tab completion for homodyne-config
    complete -c homodyne-config -l help -d 'Show help message'
    complete -c homodyne-config -s m -l mode -xa 'static laminar_flow' -d 'Configuration mode'
    complete -c homodyne-config -s o -l output -rF -d 'Output file'
    complete -c homodyne-config -s i -l interactive -d 'Interactive configuration builder'
    complete -c homodyne-config -s v -l validate -rF -d 'Validate configuration file'
    complete -c homodyne-config -s f -l force -d 'Force overwrite'

    # Tab completion for homodyne-config-xla
    complete -c homodyne-config-xla -l help -d 'Show help message'
    complete -c homodyne-config-xla -l mode -xa 'cmc cmc-hpc nlsq auto' -d 'XLA mode'
    complete -c homodyne-config-xla -l show -d 'Show current XLA configuration'

    # Tab completion for homodyne-post-install
    complete -c homodyne-post-install -l help -d 'Show help message'
    complete -c homodyne-post-install -s i -l interactive -d 'Interactive setup'
    complete -c homodyne-post-install -l shell -xa 'bash zsh fish' -d 'Shell type'
    complete -c homodyne-post-install -l xla-mode -xa 'cmc cmc-hpc nlsq auto' -d 'XLA mode'
    complete -c homodyne-post-install -l advanced -d 'Install advanced features'
    complete -c homodyne-post-install -s f -l force -d 'Force setup'

    # Tab completion for homodyne-cleanup
    complete -c homodyne-cleanup -l help -d 'Show help message'
    complete -c homodyne-cleanup -s i -l interactive -d 'Interactive cleanup'
    complete -c homodyne-cleanup -s n -l dry-run -d 'Show what would be removed'
    complete -c homodyne-cleanup -s f -l force -d 'Skip confirmation'
end
"""
        fish_file.write_text(fish_content)
        fish_file.chmod(0o644)
        return fish_file

    except OSError:
        return None


def integrate_with_venv_activate(venv_path: Path) -> list[Path]:
    """Integrate homodyne completion with virtual environment's activate scripts.

    This modifies the venv's own activate scripts (not user's global shell configs)
    so completion is automatically available when the environment is activated.

    Parameters
    ----------
    venv_path : Path
        Path to virtual environment

    Returns
    -------
    list[Path]
        List of modified activation script paths
    """
    bin_dir = venv_path / "bin"
    modified_scripts = []

    # Bash/Zsh activation integration
    bash_activate = bin_dir / "activate"
    if bash_activate.exists():
        try:
            content = bash_activate.read_text()

            # Check if already integrated
            if "homodyne-completion" not in content:
                # Source full completion.sh (tab completion + aliases + functions),
                # fall back to minimal zsh aliases file
                completion_code = """
# Homodyne shell completion (auto-added by homodyne-post-install)
if [[ -f "$VIRTUAL_ENV/etc/homodyne/shell/completion.sh" ]]; then
    source "$VIRTUAL_ENV/etc/homodyne/shell/completion.sh"
elif [[ -f "$VIRTUAL_ENV/etc/zsh/homodyne-completion.zsh" ]]; then
    source "$VIRTUAL_ENV/etc/zsh/homodyne-completion.zsh"
fi
"""
                with bash_activate.open("a") as f:
                    f.write(completion_code)

                modified_scripts.append(bash_activate)
                logger.info("Integrated with %s", bash_activate.name)
            else:
                logger.info("Already integrated with %s", bash_activate.name)

        except OSError as e:
            logger.warning("Could not integrate with %s: %s", bash_activate.name, e)

    # Fish activation integration
    fish_activate = bin_dir / "activate.fish"
    if fish_activate.exists():
        try:
            content = fish_activate.read_text()

            # Check if already integrated
            if "homodyne-completion" not in content:
                # Source fish-specific completion file
                completion_code = """
# Homodyne shell completion (auto-added by homodyne-post-install)
if test -f "$VIRTUAL_ENV/etc/homodyne/shell/completion.fish"
    source "$VIRTUAL_ENV/etc/homodyne/shell/completion.fish"
end
"""
                with fish_activate.open("a") as f:
                    f.write(completion_code)

                modified_scripts.append(fish_activate)
                logger.info("Integrated with %s", fish_activate.name)
            else:
                logger.info("Already integrated with %s", fish_activate.name)

        except OSError as e:
            logger.warning("Could not integrate with %s: %s", fish_activate.name, e)

    return modified_scripts


def create_xla_activation_scripts(venv_path: Path) -> list[Path]:
    """Create XLA_FLAGS activation scripts for bash/zsh and fish.

    Parameters
    ----------
    venv_path : Path
        Path to virtual environment

    Returns
    -------
    list[Path]
        List of created XLA activation script paths
    """
    created_scripts: list[Path] = []

    try:
        # Find homodyne package to copy activation scripts
        import homodyne

        homodyne_pkg = Path(homodyne.__file__).parent

        # Source XLA activation scripts
        xla_bash_src = (
            homodyne_pkg / "runtime" / "shell" / "activation" / "xla_config.bash"
        )
        xla_fish_src = (
            homodyne_pkg / "runtime" / "shell" / "activation" / "xla_config.fish"
        )

        if not xla_bash_src.exists() or not xla_fish_src.exists():
            logger.warning("XLA activation scripts not found in homodyne package")
            return created_scripts

        # Create activation directory
        activation_dir = venv_path / "etc" / "homodyne" / "activation"
        activation_dir.mkdir(parents=True, exist_ok=True)

        # Copy bash/zsh activation script
        xla_bash_dest = activation_dir / "xla_config.bash"
        xla_bash_dest.write_text(xla_bash_src.read_text())
        xla_bash_dest.chmod(0o644)
        created_scripts.append(xla_bash_dest)

        # Copy fish activation script
        xla_fish_dest = activation_dir / "xla_config.fish"
        xla_fish_dest.write_text(xla_fish_src.read_text())
        xla_fish_dest.chmod(0o644)
        created_scripts.append(xla_fish_dest)

        logger.info("Created XLA activation scripts in %s", activation_dir)

    except (ImportError, OSError) as e:
        logger.warning("Could not create XLA activation scripts: %s", e)

    return created_scripts


def integrate_xla_with_venv_activate(venv_path: Path) -> list[Path]:
    """Integrate XLA_FLAGS configuration with virtual environment's activate scripts.

    Parameters
    ----------
    venv_path : Path
        Path to virtual environment

    Returns
    -------
    list[Path]
        List of modified activation script paths
    """
    bin_dir = venv_path / "bin"
    modified_scripts = []

    # First create the XLA activation scripts
    create_xla_activation_scripts(venv_path)

    # Bash/Zsh activation integration
    bash_activate = bin_dir / "activate"
    if bash_activate.exists():
        try:
            content = bash_activate.read_text()

            # Check if already integrated
            if "homodyne XLA configuration" not in content:
                # Append XLA configuration sourcing
                xla_code = """
# Homodyne XLA configuration (auto-added by homodyne-post-install)
if [[ -f "$VIRTUAL_ENV/etc/homodyne/activation/xla_config.bash" ]]; then
    source "$VIRTUAL_ENV/etc/homodyne/activation/xla_config.bash"
fi
"""
                with bash_activate.open("a") as f:
                    f.write(xla_code)

                modified_scripts.append(bash_activate)
                logger.info("Integrated XLA config with %s", bash_activate.name)
            else:
                logger.info("XLA config already integrated with %s", bash_activate.name)

        except OSError as e:
            logger.warning(
                "Could not integrate XLA config with %s: %s", bash_activate.name, e
            )

    # Fish activation integration
    fish_activate = bin_dir / "activate.fish"
    if fish_activate.exists():
        try:
            content = fish_activate.read_text()

            # Check if already integrated
            if "homodyne XLA configuration" not in content:
                # Append XLA configuration sourcing
                xla_code = """
# Homodyne XLA configuration (auto-added by homodyne-post-install)
if test -f "$VIRTUAL_ENV/etc/homodyne/activation/xla_config.fish"
    source "$VIRTUAL_ENV/etc/homodyne/activation/xla_config.fish"
end
"""
                with fish_activate.open("a") as f:
                    f.write(xla_code)

                modified_scripts.append(fish_activate)
                logger.info("Integrated XLA config with %s", fish_activate.name)
            else:
                logger.info("XLA config already integrated with %s", fish_activate.name)

        except OSError as e:
            logger.warning(
                "Could not integrate XLA config with %s: %s", fish_activate.name, e
            )

    return modified_scripts


def configure_xla_mode(xla_mode: str | None = None) -> bool:
    """Configure XLA_FLAGS mode for the current environment.

    Parameters
    ----------
    xla_mode : str, optional
        XLA mode to configure (cmc, cmc-hpc, nlsq, auto)
        If None, prompts user interactively

    Returns
    -------
    bool
        True if configuration succeeded, False otherwise
    """
    from homodyne.cli.xla_config import VALID_MODES, set_mode

    if xla_mode is None:
        logger.info("XLA Configuration")
        logger.info("Select XLA device mode:")
        logger.info("  cmc        - 4 devices (multi-core workstations)")
        logger.info("  cmc-hpc    - 8 devices (HPC with 36+ cores)")
        logger.info("  nlsq       - 1 device (NLSQ-only workflows)")
        logger.info("  auto       - Auto-detect based on CPU cores")

        xla_mode = input("\nXLA mode [cmc]: ").strip() or "cmc"

    if xla_mode not in VALID_MODES:
        logger.error("Invalid XLA mode: %s", xla_mode)
        logger.error("Valid modes: %s", ", ".join(VALID_MODES))
        return False

    try:
        if set_mode(xla_mode):
            logger.info("XLA mode configured successfully")
            return True
        else:
            logger.error("Failed to configure XLA mode")
            return False
    except Exception as e:
        logger.error("XLA configuration failed: %s", e)
        return False


def install_shell_completion(
    shell_type: str | None = None, force: bool = False
) -> bool:
    """Install unified shell completion system.

    Integrates completion directly into the virtual environment's activate scripts.
    Does NOT modify user's global shell configuration files (~/.bashrc, ~/.zshrc).

    Parameters
    ----------
    shell_type : str, optional
        Shell type (bash, zsh, fish). Auto-detected if not provided.
    force : bool, default False
        Force installation even if not in virtual environment
    """
    if not is_virtual_environment() and not force:
        logger.warning("Shell completion recommended only in virtual environments")
        return False

    venv_path = Path(sys.prefix)

    try:
        # Create minimal zsh aliases as fallback
        create_unified_zsh_completion(venv_path)

        # Copy full completion.sh (tab completion + aliases + interactive builder)
        completion_sh = copy_full_completion_script(venv_path)
        if completion_sh:
            logger.info(
                "Full completion script: %s", completion_sh.relative_to(venv_path)
            )
        else:
            logger.warning(
                "Could not copy full completion.sh, using aliases-only fallback"
            )

        # Create fish-specific completion file
        fish_completion = create_fish_completion(venv_path)
        if fish_completion:
            logger.info("Fish completion: %s", fish_completion.relative_to(venv_path))

        # Environment-specific setup
        is_conda = is_conda_environment(venv_path)

        if is_conda:
            # Create conda activation script (automatic on conda activate)
            activate_dir = venv_path / "etc" / "conda" / "activate.d"
            activate_dir.mkdir(parents=True, exist_ok=True)

            # Create XLA activation scripts
            create_xla_activation_scripts(venv_path)

            completion_script = activate_dir / "homodyne-completion.sh"
            completion_content = f"""#!/bin/bash
# Homodyne completion activation

# Full completion (tab completion + aliases + functions)
if [[ -f "{venv_path}/etc/homodyne/shell/completion.sh" ]]; then
    source "{venv_path}/etc/homodyne/shell/completion.sh"
elif [[ -f "{venv_path}/etc/zsh/homodyne-completion.zsh" ]]; then
    source "{venv_path}/etc/zsh/homodyne-completion.zsh"
fi

# XLA configuration
if [[ -f "{venv_path}/etc/homodyne/activation/xla_config.bash" ]]; then
    source "{venv_path}/etc/homodyne/activation/xla_config.bash"
fi
"""
            completion_script.write_text(completion_content)
            completion_script.chmod(0o755)

            logger.info("Shell completion installed (conda/mamba)")
            logger.info("  Auto-activated on environment activation")
            logger.info("  Tab completion for all commands and options")
            logger.info("  Aliases: hm, hconfig, hm-nlsq, hm-cmc")
            logger.info("  Config: hc-stat, hc-flow")
            logger.info("  Plotting: hexp, hsim")
            logger.info(
                "Completion activates automatically when you activate this environment"
            )
        else:
            # Integrate with venv/uv/virtualenv activation scripts
            modified_scripts = integrate_with_venv_activate(venv_path)

            # Also integrate XLA configuration
            xla_modified = integrate_xla_with_venv_activate(venv_path)
            modified_scripts.extend(xla_modified)

            logger.info("Shell completion installed (uv/venv/virtualenv)")
            logger.info("  Aliases: hm, hconfig, hm-nlsq, hm-cmc")
            logger.info("  Config: hc-stat, hc-flow")
            logger.info("  Plotting: hexp, hsim")
            logger.info("  XLA_FLAGS: Auto-configured on activation")
            logger.info(
                "Completion activates automatically when you activate this environment"
            )
            logger.info("  Modified: %s", ", ".join(s.name for s in modified_scripts))
            logger.info("To test:")
            logger.info("  1. Deactivate and reactivate this environment")
            logger.info("  2. Try: hm --help")

        return True

    except Exception as e:
        logger.error("Shell completion installation failed: %s", e)
        return False


# GPU acceleration removed in v2.3.0 - function removed for CPU-only architecture


def install_macos_shell_completion() -> None:
    """Install shell completion for macOS."""
    import sys
    from pathlib import Path

    venv_path = Path(sys.prefix)
    config_dir = venv_path / "etc" / "homodyne"
    config_dir.mkdir(parents=True, exist_ok=True)

    # NOTE (Dec 2025): macOS shell aliases feature is not yet implemented.
    # This function is a placeholder for future macOS shell completion support.
    # Users can manually add aliases to their ~/.zshrc or ~/.bashrc:
    #
    #   alias hm='homodyne --method cmc'
    #   alias hconfig='homodyne --config'
    #
    # See docs/README.md for CLI usage documentation.
    pass


def install_advanced_features() -> bool:
    """Install advanced features: completion caching and system validation."""
    logger.info("Installing Advanced Features...")

    try:
        venv_path = Path(sys.prefix)

        # Find the homodyne source directory
        try:
            import homodyne

            homodyne_src_dir = Path(homodyne.__file__).parent.parent
        except ImportError:
            logger.error("Homodyne package not found")
            return False

        # Check if advanced features files exist (CPU-only)
        required_files = [
            homodyne_src_dir / "homodyne" / "runtime" / "shell" / "completion.sh",
            homodyne_src_dir / "homodyne" / "runtime" / "utils" / "system_validator.py",
        ]

        missing_files = [f for f in required_files if not f.exists()]
        if missing_files:
            logger.warning(
                "Advanced features files not found: %s",
                [f.name for f in missing_files],
            )
            logger.info("Run from development environment or upgrade to latest version")
            return False

        # Install CLI commands for advanced features
        bin_dir = venv_path / "bin"

        # System validator command — use the installed package rather than a
        # hardcoded absolute path so the script survives venv relocation.
        validator_cmd = bin_dir / "homodyne-validate"
        validator_content = """#!/usr/bin/env python3
from homodyne.runtime.utils.system_validator import main
if __name__ == "__main__":
    main()
"""
        validator_cmd.write_text(validator_content)
        validator_cmd.chmod(0o755)

        # Note: Full completion.sh is now always copied by install_shell_completion().
        # The separate advanced-completion conda script is no longer needed.

        logger.info("Advanced features installed successfully")
        logger.info("  homodyne-validate - Comprehensive system validation")

        return True

    except OSError as e:
        logger.error("Advanced features installation failed: %s", e)
        return False


def interactive_setup() -> tuple[bool, list[str]]:
    """Interactive setup allowing user to choose what to install (CPU-only in v2.3.0)."""
    logger.info("Homodyne Optional Setup (v2.3.0 - CPU-only)")
    logger.info("Choose what to install (you can run this again later):")

    # Shell completion
    logger.info("1. Shell Completion (bash/zsh/fish)")
    logger.info("   - Adds tab completion for homodyne commands")
    logger.info("   - Adds convenient aliases (hm, hconfig, hm-nlsq, hm-cmc)")
    logger.info("   - Safe: doesn't interfere with system commands")

    install_completion = (
        input("   Install shell completion? [y/N]: ").lower().startswith("y")
    )

    shell_type = None
    if install_completion:
        logger.info("Detected shells:")
        current_shell = os.environ.get("SHELL", "").split("/")[-1]
        if current_shell:
            logger.info("   - Current: %s", current_shell)
        logger.info("   - Available: bash, zsh, fish")

        shell_input = input(f"   Shell type [{current_shell or 'bash'}]: ").strip()
        shell_type = (
            shell_input
            if shell_input in ["bash", "zsh", "fish"]
            else (current_shell or "bash")
        )

    # XLA configuration
    logger.info("2. XLA_FLAGS Configuration")
    logger.info("   - Auto-configures JAX CPU devices on venv activation")
    logger.info(
        "   - Modes: cmc (4 devices), cmc-hpc (8 devices), nlsq (1 device), auto"
    )
    logger.info("   - Essential for MCMC parallelization and optimal CPU usage")

    configure_xla = input("   Configure XLA_FLAGS? [Y/n]: ").lower() != "n"

    xla_mode = None
    if configure_xla:
        xla_mode = None  # Will be prompted in configure_xla_mode()

    # Advanced features
    logger.info("3. Advanced Features")
    logger.info("   - Smart shell completion with config file caching")
    logger.info("   - System validation (homodyne-validate command)")
    logger.info("   - Adds homodyne-validate command")

    install_advanced = (
        input("   Install advanced features? [y/N]: ").lower().startswith("y")
    )

    # Perform installations
    results = []

    if install_completion and shell_type:
        if install_shell_completion(shell_type):
            results.append(f"[OK] {shell_type.title()} completion")
        else:
            results.append(f"[FAIL] {shell_type.title()} completion failed")

    if configure_xla:
        if configure_xla_mode(xla_mode):
            results.append("[OK] XLA configuration")
        else:
            results.append("[FAIL] XLA configuration failed")

    if install_advanced:
        if install_advanced_features():
            results.append("[OK] Advanced features")
        else:
            results.append("[FAIL] Advanced features failed")

    return len([r for r in results if r.startswith("[OK]")]) > 0, results


def show_installation_summary(interactive_results: list[str] | None = None) -> None:
    """Show installation summary with available commands."""
    logger.info("Quick Start Commands:")
    logger.info("   homodyne --method nlsq --config config.yaml")
    logger.info(
        "   homodyne --method cmc --config config.yaml  # Automatic NUTS/CMC selection"
    )
    logger.info("   homodyne-config --mode static -o my_config.yaml")
    logger.info("Available Shortcuts (after shell restart):")
    logger.info("   Base commands:")
    logger.info("     hm       = homodyne")
    logger.info("     hconfig  = homodyne-config")
    logger.info("   Method shortcuts (hm- prefix):")
    logger.info("     hm-nlsq  = homodyne --method nlsq  # NLSQ trust-region (primary)")
    logger.info("     hm-cmc  = homodyne --method cmc  # Consensus Monte Carlo")
    logger.info("   Config mode shortcuts (hc- prefix):")
    logger.info("     hc-stat  = homodyne-config --mode static")
    logger.info("     hc-flow  = homodyne-config --mode laminar_flow")
    logger.info("Help:")
    logger.info("   homodyne --help")
    logger.info("   homodyne-config --help")
    logger.info("   homodyne_help               # View all shortcuts")


def main() -> int:
    """Main post-installation routine with optional shell completion system."""
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("=" * 70)
    logger.info("Homodyne Post-Installation Setup")
    logger.info("=" * 70)

    # Detect environment and platform
    is_venv = is_virtual_environment()
    system = platform.system()

    logger.info("Platform: %s", system)
    logger.info(
        "Environment: %s", "Virtual Environment" if is_venv else "System Python"
    )

    if not is_venv and not args.force:
        logger.warning("Virtual environment recommended for optimal setup")
        logger.info("   Run in conda/mamba/venv for full functionality")
        logger.info("   Use --force to install anyway")
        logger.info("Basic usage (no setup needed):")
        logger.info("   homodyne --help")
        logger.info("   homodyne-config --help")
        return 0

    if args.interactive:
        success, results = interactive_setup()
        logger.info("=" * 70)
        if success:
            logger.info("Setup completed!")
            for result in results:
                logger.info("   %s", result)
            logger.info("Next steps:")
            logger.info("   1. Restart your shell or reactivate environment")
            logger.info("   2. Test: homodyne --help")
            logger.info("   3. Try shortcuts: hm --help")
        else:
            logger.warning("Setup completed with issues")
            for result in results:
                logger.info("   %s", result)
        logger.info("=" * 70)
        return 0 if success else 1

    # Non-interactive mode - install based on arguments
    results = []
    success = True

    # Determine what to install
    if args.shell or (not args.shell and not args.advanced and not args.xla_mode):
        # Install shell completion by default or if specified
        logger.info("Installing shell completion...")
        shell_type = args.shell if args.shell else None
        if install_shell_completion(shell_type, force=args.force):
            results.append("Shell completion: OK")
        else:
            results.append("Shell completion: FAILED")
            success = False

    if args.xla_mode:
        # Configure XLA if requested
        logger.info("Configuring XLA_FLAGS...")
        if configure_xla_mode(args.xla_mode):
            results.append("XLA configuration: OK")
        else:
            results.append("XLA configuration: FAILED")
            success = False

    if args.advanced:
        # Install advanced features if requested
        logger.info("Installing Advanced Features...")
        if install_advanced_features():
            results.append("Advanced features: OK")
        else:
            results.append("Advanced features: FAILED")
            success = False

    logger.info("=" * 70)
    if results:
        logger.info("Installation results:")
        for result in results:
            logger.info("   %s", result)

    if success:
        logger.info("Setup completed!")
        logger.info("Next steps:")
        logger.info("   1. Restart shell or reactivate environment:")
        logger.info("      conda deactivate && conda activate $CONDA_DEFAULT_ENV")
        logger.info("   2. Test commands:")
        logger.info("      hm --help  # Should work after reactivation")
    else:
        logger.warning("Setup had some issues")
        logger.info("   Try: homodyne-post-install --interactive")
    logger.info("=" * 70)

    return 0 if success else 1


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="homodyne-post-install",
        description="Set up optional Homodyne shell completion (v2.3.0 - CPU-only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  homodyne-post-install --interactive              # Interactive setup (recommended)
  homodyne-post-install                            # Install shell completion only
  homodyne-post-install --xla-mode auto            # Configure XLA with auto-detection
  homodyne-post-install --shell bash --xla-mode cmc   # Shell completion + XLA
  homodyne-post-install --force                    # Force install outside venv

The script provides optional installation of:
- Shell completion (bash/zsh/fish) with safe aliases
- XLA_FLAGS auto-configuration for JAX CPU devices
- Advanced features: completion caching, system validation
- Virtual environment integration

Version 2.3.0: GPU support removed - CPU-only architecture
        """,
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive setup - choose what to install",
    )

    parser.add_argument(
        "--shell",
        choices=["bash", "zsh", "fish"],
        help="Specify shell type for completion",
    )

    parser.add_argument(
        "--xla-mode",
        choices=["cmc", "cmc-hpc", "nlsq", "auto"],
        help="Configure XLA_FLAGS mode (cmc, cmc-hpc, nlsq, auto)",
    )

    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Install advanced features (completion caching, system validation)",
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force setup even if not in virtual environment",
    )

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
