"""Shell completion and activation scripts for homodyne.

This subpackage contains shell scripts for:
- Bash/Zsh completion for homodyne CLI commands
- XLA_FLAGS auto-configuration scripts

The scripts are installed by the post_install module and can be
sourced in shell startup files or virtual environment activation scripts.
"""

from pathlib import Path

# Path to shell scripts
SHELL_DIR = Path(__file__).parent
COMPLETION_SCRIPT = SHELL_DIR / "completion.sh"
ACTIVATION_DIR = SHELL_DIR / "activation"
XLA_CONFIG_BASH = ACTIVATION_DIR / "xla_config.bash"
XLA_CONFIG_FISH = ACTIVATION_DIR / "xla_config.fish"


def get_completion_script() -> str:
    """Get the path to the bash completion script.

    Returns:
        Absolute path to completion.sh
    """
    return str(COMPLETION_SCRIPT.resolve())


def get_xla_config_script(shell: str = "bash") -> str:
    """Get the path to the XLA configuration script.

    Args:
        shell: Shell type ("bash", "zsh", or "fish")

    Returns:
        Absolute path to the appropriate XLA config script.
    """
    if shell in ("bash", "zsh"):
        return str(XLA_CONFIG_BASH.resolve())
    elif shell == "fish":
        return str(XLA_CONFIG_FISH.resolve())
    else:
        raise ValueError(f"Unsupported shell: {shell}")


__all__ = [
    "SHELL_DIR",
    "COMPLETION_SCRIPT",
    "ACTIVATION_DIR",
    "XLA_CONFIG_BASH",
    "XLA_CONFIG_FISH",
    "get_completion_script",
    "get_xla_config_script",
]
