"""Post-installation setup for homodyne package.

This module provides interactive setup for:
- Shell completion installation (bash/zsh/fish)
- XLA_FLAGS configuration
- Virtual environment integration

CLI Entry Point: homodyne-post-install
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Literal


def detect_shell_type() -> Literal["bash", "zsh", "fish", "unknown"]:
    """Detect the current shell type.

    Returns:
        Shell type string or "unknown" if detection fails.
    """
    # Check SHELL environment variable
    shell_path = os.environ.get("SHELL", "")
    shell_name = os.path.basename(shell_path)

    if "zsh" in shell_name:
        return "zsh"
    elif "bash" in shell_name:
        return "bash"
    elif "fish" in shell_name:
        return "fish"

    # Fallback: check parent process name
    try:
        import psutil

        parent = psutil.Process().parent()
        if parent:
            pname = parent.name().lower()
            if "zsh" in pname:
                return "zsh"
            elif "bash" in pname:
                return "bash"
            elif "fish" in pname:
                return "fish"
    except (ImportError, OSError, AttributeError):
        pass

    return "unknown"


def is_virtual_environment() -> bool:
    """Check if running in a virtual environment.

    Returns:
        True if in a venv, conda env, or similar.
    """
    # Standard venv check
    if sys.prefix != sys.base_prefix:
        return True

    # Conda environment check
    if os.environ.get("CONDA_PREFIX"):
        return True

    # Check for VIRTUAL_ENV marker
    if os.environ.get("VIRTUAL_ENV"):
        return True

    return False


def is_conda_environment() -> bool:
    """Check if running in a conda/mamba environment.

    Returns:
        True if in a conda environment.
    """
    return bool(os.environ.get("CONDA_PREFIX"))


def get_venv_path() -> Path:
    """Get the virtual environment path.

    Returns:
        Path to the virtual environment directory.
    """
    # Prefer VIRTUAL_ENV if set
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        return Path(venv)

    # Conda environment
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        return Path(conda_prefix)

    # Fallback to sys.prefix
    return Path(sys.prefix)


def get_completion_source_path() -> Path:
    """Get the path to the completion script in the package.

    Returns:
        Path to completion.sh in the installed package.
    """
    try:
        from homodyne.runtime.shell import COMPLETION_SCRIPT

        return COMPLETION_SCRIPT
    except ImportError:
        # Fallback: find relative to this file
        return Path(__file__).parent / "runtime" / "shell" / "completion.sh"


def get_xla_config_source_path(shell: str) -> Path:
    """Get the path to the XLA config script.

    Args:
        shell: Shell type ("bash", "zsh", or "fish")

    Returns:
        Path to the XLA config script.
    """
    try:
        from homodyne.runtime.shell import XLA_CONFIG_BASH, XLA_CONFIG_FISH

        if shell == "fish":
            return XLA_CONFIG_FISH
        return XLA_CONFIG_BASH
    except ImportError:
        # Fallback
        base = Path(__file__).parent / "runtime" / "shell" / "activation"
        if shell == "fish":
            return base / "xla_config.fish"
        return base / "xla_config.bash"


def install_bash_completion(venv_path: Path, verbose: bool = False) -> bool:
    """Install bash completion script.

    Args:
        venv_path: Path to virtual environment.
        verbose: Print verbose output.

    Returns:
        True if installation succeeded.
    """
    source = get_completion_source_path()
    if not source.exists():
        if verbose:
            print(f"Completion script not found: {source}")
        return False

    # Install to venv/etc/bash_completion.d/
    dest_dir = venv_path / "etc" / "bash_completion.d"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "homodyne"

    try:
        shutil.copy2(source, dest)
        if verbose:
            print(f"Installed bash completion to: {dest}")
        return True
    except (OSError, shutil.Error) as e:
        if verbose:
            print(f"Failed to install bash completion: {e}")
        return False


def install_zsh_completion(venv_path: Path, verbose: bool = False) -> bool:
    """Install zsh completion script.

    Args:
        venv_path: Path to virtual environment.
        verbose: Print verbose output.

    Returns:
        True if installation succeeded.
    """
    source = get_completion_source_path()
    if not source.exists():
        if verbose:
            print(f"Completion script not found: {source}")
        return False

    # Install to venv/etc/zsh/
    dest_dir = venv_path / "etc" / "zsh"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "homodyne-completion.zsh"

    try:
        # Use the installed copy in the venv rather than the package source
        # path, so the wrapper survives wheel-based relocations.
        installed_bash = venv_path / "etc" / "bash_completion.d" / "homodyne"
        if not installed_bash.exists():
            # Ensure the bash completion is installed first
            install_bash_completion(venv_path, verbose=False)

        # Prefer the venv-local copy; fall back to source if install failed
        completion_path = installed_bash if installed_bash.exists() else source
        content = f"""# Zsh completion for homodyne (generated)
# Source the bash completion in zsh-compatible mode

autoload -Uz bashcompinit
bashcompinit

source "{completion_path}"
"""
        dest.write_text(content, encoding="utf-8")
        if verbose:
            print(f"Installed zsh completion to: {dest}")
        return True
    except OSError as e:
        if verbose:
            print(f"Failed to install zsh completion: {e}")
        return False


def install_fish_completion(venv_path: Path, verbose: bool = False) -> bool:
    """Install fish completion (basic support).

    Args:
        venv_path: Path to virtual environment.
        verbose: Print verbose output.

    Returns:
        True if installation succeeded.
    """
    # Fish completions go to a specific location
    dest_dir = venv_path / "share" / "fish" / "vendor_completions.d"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "homodyne.fish"

    try:
        content = """# Fish completion for homodyne (generated)

# homodyne
complete -c homodyne -s c -l config -d 'Configuration file' -F
complete -c homodyne -s d -l data-file -d 'Input data file' -F
complete -c homodyne -s m -l method -d 'Optimization method' -a 'nlsq cmc'
complete -c homodyne -s o -l output-dir -d 'Output directory' -F
complete -c homodyne -s v -l verbose -d 'Verbose output'
complete -c homodyne -s q -l quiet -d 'Quiet output'
complete -c homodyne -l log-level -d 'Log level' -a 'DEBUG INFO WARNING ERROR'
complete -c homodyne -s h -l help -d 'Show help'
complete -c homodyne -l version -d 'Show version'
complete -c homodyne -l static-mode -d 'Force static analysis mode'
complete -c homodyne -l laminar-flow -d 'Force laminar flow mode'
complete -c homodyne -l output-format -d 'Output format' -a 'yaml json npz'
complete -c homodyne -l save-plots -d 'Save result plots'
complete -c homodyne -l plot-experimental-data -d 'Generate data validation plots'
complete -c homodyne -l plot-simulated-data -d 'Plot theoretical C2 heatmaps'
complete -c homodyne -l plotting-backend -d 'Plotting backend' -a 'auto matplotlib datashader'
complete -c homodyne -l parallel-plots -d 'Generate plots in parallel'
complete -c homodyne -l max-iterations -d 'Maximum NLSQ iterations'
complete -c homodyne -l tolerance -d 'NLSQ convergence tolerance'
complete -c homodyne -l n-samples -d 'CMC samples per chain'
complete -c homodyne -l n-warmup -d 'CMC warmup samples'
complete -c homodyne -l n-chains -d 'Number of CMC chains'
complete -c homodyne -l cmc-num-shards -d 'Data shards for CMC' -a '4 8 10 16 20 32'
complete -c homodyne -l cmc-backend -d 'CMC parallel backend' -a 'auto pjit multiprocessing pbs'
complete -c homodyne -l no-nlsq-warmstart -d 'Disable NLSQ warm-start for CMC'
complete -c homodyne -l nlsq-result -d 'Pre-computed NLSQ results' -F
complete -c homodyne -l dense-mass-matrix -d 'Use dense mass matrix for NUTS'

# homodyne-config
complete -c homodyne-config -s m -l mode -d 'Configuration mode' -a 'static laminar_flow'
complete -c homodyne-config -s o -l output -d 'Output file' -F
complete -c homodyne-config -s i -l interactive -d 'Interactive configuration builder'
complete -c homodyne-config -s v -l validate -d 'Validate configuration file' -F
complete -c homodyne-config -s f -l force -d 'Force overwrite'
complete -c homodyne-config -s h -l help -d 'Show help'

# homodyne-config-xla
complete -c homodyne-config-xla -l mode -d 'XLA mode' -a 'auto nlsq cmc cmc-hpc'
complete -c homodyne-config-xla -l show -d 'Show current XLA configuration'
complete -c homodyne-config-xla -s h -l help -d 'Show help'

# homodyne-post-install
complete -c homodyne-post-install -s i -l interactive -d 'Interactive setup'
complete -c homodyne-post-install -s s -l shell -d 'Shell type' -a 'bash zsh fish'
complete -c homodyne-post-install -l no-completion -d 'Skip shell completion'
complete -c homodyne-post-install -l no-xla -d 'Skip XLA configuration'
complete -c homodyne-post-install -l xla-mode -d 'XLA mode' -a 'auto nlsq cmc cmc-hpc'
complete -c homodyne-post-install -s v -l verbose -d 'Verbose output'
complete -c homodyne-post-install -s h -l help -d 'Show help'

# homodyne-cleanup
complete -c homodyne-cleanup -s n -l dry-run -d 'Show what would be removed'
complete -c homodyne-cleanup -s f -l force -d 'Force cleanup without confirmation'
complete -c homodyne-cleanup -s i -l interactive -d 'Interactive cleanup'
complete -c homodyne-cleanup -s v -l verbose -d 'Verbose output'
complete -c homodyne-cleanup -s h -l help -d 'Show help'

# Short aliases (hm = homodyne)
complete -c hm -w homodyne
complete -c hconfig -w homodyne-config
complete -c hm-nlsq -w homodyne
complete -c hm-cmc -w homodyne
complete -c hc-stat -w homodyne-config
complete -c hc-flow -w homodyne-config
complete -c hxla -w homodyne-config-xla
complete -c hsetup -w homodyne-post-install
complete -c hclean -w homodyne-cleanup

# Plotting aliases
alias hm 'homodyne'
alias hconfig 'homodyne-config'
alias hm-nlsq 'homodyne --method nlsq'
alias hm-cmc 'homodyne --method cmc'
alias hc-stat 'homodyne-config --mode static'
alias hc-flow 'homodyne-config --mode laminar_flow'
alias hexp 'homodyne --plot-experimental-data'
alias hsim 'homodyne --plot-simulated-data'
alias hxla 'homodyne-config-xla'
alias hsetup 'homodyne-post-install'
alias hclean 'homodyne-cleanup'
complete -c hexp -w homodyne
complete -c hsim -w homodyne
"""
        dest.write_text(content, encoding="utf-8")
        if verbose:
            print(f"Installed fish completion to: {dest}")
        return True
    except OSError as e:
        if verbose:
            print(f"Failed to install fish completion: {e}")
        return False


def install_shell_completion(
    shell: str | None = None,
    verbose: bool = False,
) -> bool:
    """Install shell completion for the detected or specified shell.

    Args:
        shell: Shell type or None for auto-detection.
        verbose: Print verbose output.

    Returns:
        True if installation succeeded.
    """
    if not is_virtual_environment():
        if verbose:
            print("Not in a virtual environment, skipping completion install")
        return False

    venv_path = get_venv_path()
    detected_shell = shell or detect_shell_type()

    if detected_shell == "unknown":
        if verbose:
            print("Could not detect shell type, trying bash completion")
        detected_shell = "bash"

    if verbose:
        print(f"Installing {detected_shell} completion to {venv_path}")

    if detected_shell == "zsh":
        return install_zsh_completion(venv_path, verbose)
    elif detected_shell == "fish":
        return install_fish_completion(venv_path, verbose)
    else:
        return install_bash_completion(venv_path, verbose)


def install_xla_activation(
    shell: str | None = None,
    mode: str = "auto",
    verbose: bool = False,
) -> bool:
    """Install XLA configuration to venv activation script.

    Args:
        shell: Shell type or None for auto-detection.
        mode: XLA mode (auto, nlsq, cmc, cmc-hpc).
        verbose: Print verbose output.

    Returns:
        True if installation succeeded.
    """
    if not is_virtual_environment():
        if verbose:
            print("Not in a virtual environment, skipping XLA activation install")
        return False

    venv_path = get_venv_path()
    detected_shell = shell or detect_shell_type()

    if detected_shell in ("bash", "zsh", "unknown"):
        return _install_xla_bash_activation(venv_path, mode, verbose)
    elif detected_shell == "fish":
        return _install_xla_fish_activation(venv_path, mode, verbose)
    else:
        return False


def _install_xla_bash_activation(
    venv_path: Path,
    mode: str,
    verbose: bool,
) -> bool:
    """Install XLA config to bash/zsh activate script."""
    activate_script = venv_path / "bin" / "activate"
    if not activate_script.exists():
        if verbose:
            print(f"Activate script not found: {activate_script}")
        return False

    # Check if already installed
    content = activate_script.read_text(encoding="utf-8")
    marker = "# homodyne XLA configuration"

    if marker in content:
        if verbose:
            print("XLA activation already installed in activate script")
        return True

    # Get source script path
    xla_script = get_xla_config_source_path("bash")

    # Append XLA configuration sourcing
    addition = f"""
{marker}
if [ -f "{xla_script}" ]; then
    source "{xla_script}" {mode}
fi
"""

    try:
        with open(activate_script, "a", encoding="utf-8") as f:
            f.write(addition)
        if verbose:
            print(f"Added XLA activation to: {activate_script}")
        return True
    except OSError as e:
        if verbose:
            print(f"Failed to modify activate script: {e}")
        return False


def _install_xla_fish_activation(
    venv_path: Path,
    mode: str,
    verbose: bool,
) -> bool:
    """Install XLA config to fish activate script."""
    activate_script = venv_path / "bin" / "activate.fish"
    if not activate_script.exists():
        if verbose:
            print(f"Fish activate script not found: {activate_script}")
        return False

    # Check if already installed
    content = activate_script.read_text(encoding="utf-8")
    marker = "# homodyne XLA configuration"

    if marker in content:
        if verbose:
            print("XLA activation already installed in fish activate script")
        return True

    # Get source script path
    xla_script = get_xla_config_source_path("fish")

    # Append XLA configuration sourcing
    addition = f"""
{marker}
if test -f "{xla_script}"
    source "{xla_script}" {mode}
end
"""

    try:
        with open(activate_script, "a", encoding="utf-8") as f:
            f.write(addition)
        if verbose:
            print(f"Added XLA activation to: {activate_script}")
        return True
    except OSError as e:
        if verbose:
            print(f"Failed to modify fish activate script: {e}")
        return False


def get_xla_mode_path() -> Path:
    """Get the path for the XLA mode configuration file.

    Uses the virtual environment if active, otherwise XDG config directory.
    Priority: $VIRTUAL_ENV or $CONDA_PREFIX > $XDG_CONFIG_HOME/homodyne.

    Returns:
        Path to the XLA mode file.
    """
    # Prefer per-environment config
    venv = os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX")
    if venv:
        return Path(venv) / "etc" / "homodyne" / "xla_mode"

    # Fall back to XDG config directory
    xdg_config = os.environ.get("XDG_CONFIG_HOME", "")
    if not xdg_config:
        xdg_config = str(Path.home() / ".config")
    return Path(xdg_config) / "homodyne" / "xla_mode"


def _migrate_legacy_xla_mode(new_path: Path) -> None:
    """Migrate legacy ~/.homodyne_xla_mode to new location if it exists."""
    legacy = Path.home() / ".homodyne_xla_mode"
    if legacy.exists() and not new_path.exists():
        try:
            new_path.parent.mkdir(parents=True, exist_ok=True)
            mode = legacy.read_text(encoding="utf-8").strip()
            new_path.write_text(mode, encoding="utf-8")
            legacy.unlink()
        except OSError:
            pass  # Best-effort migration


def configure_xla_mode(mode: str = "auto", verbose: bool = False) -> bool:
    """Configure the XLA mode.

    Stores in the virtual environment (if active) or XDG config directory.

    Args:
        mode: XLA mode (auto, nlsq, cmc, cmc-hpc, or a number).
        verbose: Print verbose output.

    Returns:
        True if configuration succeeded.
    """
    config_file = get_xla_mode_path()

    try:
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(mode, encoding="utf-8")
        if verbose:
            print(f"Set XLA mode to '{mode}' in {config_file}")

        # Clean up legacy file if it exists
        legacy = Path.home() / ".homodyne_xla_mode"
        if legacy.exists():
            legacy.unlink(missing_ok=True)
            if verbose:
                print(f"Removed legacy config: {legacy}")

        return True
    except OSError as e:
        if verbose:
            print(f"Failed to write XLA mode config: {e}")
        return False


def interactive_setup() -> None:
    """Run interactive post-installation setup."""
    print("=" * 60)
    print("Homodyne Post-Installation Setup")
    print("=" * 60)
    print()

    # Detect environment
    shell = detect_shell_type()
    in_venv = is_virtual_environment()
    is_conda = is_conda_environment()

    print(f"Detected shell: {shell}")
    print(f"Virtual environment: {in_venv}")
    if is_conda:
        print(f"Conda environment: {os.environ.get('CONDA_PREFIX', '')}")
    elif in_venv:
        print(f"Venv path: {get_venv_path()}")
    print()

    if not in_venv:
        print("WARNING: Not running in a virtual environment.")
        print("Shell completion and XLA activation require a virtual environment.")
        print()
        response = input("Continue anyway? [y/N]: ").strip().lower()
        if response != "y":
            print("Aborted.")
            return

    # Shell completion
    print("\n--- Shell Completion ---")
    response = input(f"Install {shell} shell completion? [Y/n]: ").strip().lower()
    if response != "n":
        success = install_shell_completion(shell, verbose=True)
        if success:
            print("Shell completion installed successfully!")
            env_var = "$CONDA_PREFIX" if is_conda else "$VIRTUAL_ENV"
            if shell == "zsh":
                print(
                    f"Add to ~/.zshrc: source {env_var}/etc/zsh/homodyne-completion.zsh"
                )
            elif shell == "bash":
                print(
                    f"Add to ~/.bashrc: source {env_var}/etc/bash_completion.d/homodyne"
                )
        else:
            print("Shell completion installation failed.")
    print()

    # XLA Configuration
    print("\n--- XLA Configuration ---")
    print("XLA modes control how many CPU devices JAX uses:")
    print("  auto    - Auto-detect based on CPU cores (recommended)")
    print("  nlsq    - Single device for NLSQ fitting")
    print("  cmc     - 4 devices for CMC sampling")
    print("  cmc-hpc - 8 devices for HPC CMC")
    print()

    mode = input("Select XLA mode [auto]: ").strip().lower() or "auto"
    if mode not in ("auto", "nlsq", "cmc", "cmc-hpc"):
        # Check if it's a number
        try:
            int(mode)
        except ValueError:
            print(f"Invalid mode: {mode}, using 'auto'")
            mode = "auto"

    success = configure_xla_mode(mode, verbose=True)
    if success:
        print(f"XLA mode set to '{mode}'")

    # Install XLA activation
    response = (
        input("\nAdd XLA config to venv activate script? [Y/n]: ").strip().lower()
    )
    if response != "n":
        success = install_xla_activation(shell, mode, verbose=True)
        if success:
            print("XLA activation installed!")
            print("Deactivate and reactivate your venv to apply.")
        else:
            print("XLA activation installation failed.")

    print("\n" + "=" * 60)
    print("Setup complete!")
    print()
    print("To verify installation, run: homodyne-validate")
    print("=" * 60)


def main() -> int:
    """CLI entry point for homodyne-post-install."""
    parser = argparse.ArgumentParser(
        description="Post-installation setup for homodyne",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  homodyne-post-install                  # Interactive setup
  homodyne-post-install --shell zsh      # Install zsh completion
  homodyne-post-install --no-xla         # Skip XLA configuration
""",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run interactive setup (default if no options)",
    )
    parser.add_argument(
        "--shell",
        "-s",
        choices=["bash", "zsh", "fish"],
        help="Shell type for completion installation",
    )
    parser.add_argument(
        "--no-completion",
        action="store_true",
        help="Skip shell completion installation",
    )
    parser.add_argument(
        "--no-xla",
        action="store_true",
        help="Skip XLA configuration",
    )
    parser.add_argument(
        "--xla-mode",
        choices=["auto", "nlsq", "cmc", "cmc-hpc"],
        default="auto",
        help="XLA configuration mode (default: auto)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Run interactive setup if no specific options given
    if args.interactive or (
        not args.no_completion and not args.no_xla and not args.shell
    ):
        interactive_setup()
        return 0

    # Non-interactive mode
    success = True

    if not args.no_completion:
        result = install_shell_completion(args.shell, args.verbose)
        if not result:
            print("Shell completion installation failed")
            success = False

    if not args.no_xla:
        result = configure_xla_mode(args.xla_mode, args.verbose)
        if not result:
            print("XLA mode configuration failed")
            success = False

        result = install_xla_activation(args.shell, args.xla_mode, args.verbose)
        if not result:
            print("XLA activation installation failed")
            success = False

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
