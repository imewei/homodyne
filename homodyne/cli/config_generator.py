"""Configuration Generator for Homodyne Analysis.

This module provides the homodyne-config command-line tool for:
- Generating configuration files from templates
- Interactive configuration building
- Validating existing configurations

Usage:
    homodyne-config --mode static --output config.yaml
    homodyne-config --interactive
    homodyne-config --validate my_config.yaml
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    from ruamel.yaml import YAML

    HAS_RUAMEL = True
except ImportError:
    HAS_RUAMEL = False

from homodyne.config.manager import ConfigManager
from homodyne.utils.path_validation import PathValidationError, validate_save_path


def _yaml_escape_string(s: str) -> str:
    """Escape a string for safe insertion into YAML double-quoted values."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for homodyne-config.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog="homodyne-config",
        description="Generate, build, and validate Homodyne configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from template
  homodyne-config --mode static --output my_config.yaml
  homodyne-config --mode laminar_flow --output flow_config.yaml

  # Interactive configuration builder
  homodyne-config --interactive

  # Validate existing configuration
  homodyne-config --validate my_config.yaml

Modes:
  static        Generic static diffusion (static_isotropic)
  laminar_flow  Flow analysis with shear dynamics

Aliases:
  hconfig       homodyne-config
  hc-stat       homodyne-config --mode static
  hc-flow       homodyne-config --mode laminar_flow
        """,
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        "-m",
        choices=["static", "laminar_flow"],
        help="Configuration mode (static or laminar_flow)",
    )

    # Output path
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output configuration file path (default: homodyne_config.yaml)",
    )

    # Interactive mode
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive configuration builder",
    )

    # Validation mode
    parser.add_argument(
        "--validate",
        "-v",
        type=Path,
        metavar="CONFIG_FILE",
        help="Validate existing configuration file",
    )

    # Force overwrite
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force overwrite existing configuration file",
    )

    return parser


def get_template_path(mode: str) -> Path:
    """Get template file path for given mode.

    Parameters
    ----------
    mode : str
        Configuration mode ('static' or 'laminar_flow')

    Returns
    -------
    Path
        Path to template file

    Raises
    ------
    FileNotFoundError
        If template file not found
    """
    # Map mode to template filename
    template_map = {
        "static": "homodyne_static.yaml",
        "laminar_flow": "homodyne_laminar_flow.yaml",
    }

    template_name = template_map[mode]

    # Find template in package using homodyne.config module
    import homodyne.config

    config_dir = Path(homodyne.config.__file__).parent
    templates_dir = config_dir / "templates"
    template_path = templates_dir / template_name

    if not template_path.exists():
        raise FileNotFoundError(
            f"Template not found: {template_path}\n"
            f"Expected template: {template_name} for mode '{mode}'"
        )

    return template_path


def generate_config(
    mode: str, output_path: Path, force: bool = False
) -> dict[str, Any]:
    """Generate configuration from template.

    Parameters
    ----------
    mode : str
        Configuration mode ('static' or 'laminar_flow')
    output_path : Path
        Output file path
    force : bool, optional
        Force overwrite if file exists (default: False)

    Returns
    -------
    dict
        Generated configuration dictionary

    Raises
    ------
    FileExistsError
        If output file exists and force=False
    FileNotFoundError
        If template not found
    """
    # Check if output exists
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Configuration file already exists: {output_path}\n"
            f"Use --force to overwrite"
        )

    # Get template
    template_path = get_template_path(mode)

    # Copy template directly to preserve all comments and formatting
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(template_path, output_path)

    # Load config for return value (without modifying file)
    with open(template_path, encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    print(f"OK: Generated {mode} configuration: {output_path}")
    print(f"  Template: {template_path.name}")
    print(f"  Mode: {mode}")
    print("\nNext steps:")
    print(f"  1. Edit configuration: {output_path}")
    print("  2. Update data file path")
    print("  3. Adjust parameters and bounds")
    print(f"  4. Run analysis: homodyne --config {output_path}")
    print("\nNote: All template comments and instructions are preserved.")

    return config


def interactive_builder() -> dict[str, Any]:
    """Interactive configuration builder.

    Returns
    -------
    dict
        Built configuration dictionary
    """
    print("=" * 70)
    print("Homodyne Interactive Configuration Builder")
    print("=" * 70)
    print()

    # Mode selection
    print("Select analysis mode:")
    print("  1. static        - Generic static diffusion")
    print("  2. laminar_flow  - Flow analysis with shear dynamics")
    print()

    while True:
        mode_choice = input("Mode [1/2]: ").strip()
        if mode_choice == "1":
            mode = "static"
            break
        elif mode_choice == "2":
            mode = "laminar_flow"
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    print(f"\nOK: Selected mode: {mode}")
    print()

    # Sample information
    sample_name = input("Sample name (e.g., 'my_sample'): ").strip() or "my_sample"
    experiment_id = (
        input("Experiment ID (e.g., 'exp_001'): ").strip() or "experiment_001"
    )

    # Data file
    print()
    print("Data file configuration:")
    data_file = (
        input("  HDF5 data file path (e.g., './data/experiment.hdf'): ").strip()
        or "./data/experiment.hdf"
    )

    # Output directory
    print()
    output_dir = input("Output directory (default: './output'): ").strip() or "./output"

    # Output path
    print()
    suggested_filename = f"homodyne_{mode}_{sample_name}.yaml"
    output_path_str = (
        input(f"Save configuration to (default: {suggested_filename}): ").strip()
        or suggested_filename
    )
    output_path = Path(output_path_str)

    # Check overwrite
    if output_path.exists():
        overwrite = (
            input(f"\nWARNING: File exists: {output_path}\nOverwrite? [y/N]: ")
            .strip()
            .lower()
        )
        if overwrite != "y":
            print("Configuration not saved.")
            # Still load and return config for reference
            template_path = get_template_path(mode)
            with open(template_path, encoding="utf-8") as f:
                result: dict[str, Any] = yaml.safe_load(f)
                return result

    # Get template
    template_path = get_template_path(mode)

    # Validate output path before creating directories to prevent path traversal.
    try:
        validated_path = validate_save_path(
            output_path,
            allowed_extensions=(".yaml", ".yml"),
            require_parent_exists=False,
            allow_absolute=True,
        )
    except (PathValidationError, ValueError) as e:
        print(f"Error: Invalid output path - {e}", file=sys.stderr)
        return {}
    if validated_path is None:
        print("Error: Invalid output path", file=sys.stderr)
        return {}
    output_path = validated_path

    # Save configuration with comment preservation
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_config: dict[str, Any]

    try:
        if HAS_RUAMEL:
            # Use ruamel.yaml to preserve comments while modifying values
            yaml_handler = YAML()
            yaml_handler.preserve_quotes = True
            yaml_handler.width = 4096  # Prevent line wrapping

            with open(template_path, encoding="utf-8") as f:
                config = yaml_handler.load(f)

            # Customize configuration
            config["experimental_data"]["file_path"] = data_file

            if "output" not in config:
                config["output"] = {}
            config["output"]["output_folder"] = output_dir
            config["output"]["sample_name"] = sample_name
            config["output"]["experiment_id"] = experiment_id

            with open(output_path, "w", encoding="utf-8") as f:
                yaml_handler.dump(config, f)

            # Load for return value with proper type
            with open(output_path, encoding="utf-8") as f:
                result_config = yaml.safe_load(f)
        else:
            # Fallback: use text replacement to preserve comments
            with open(template_path, encoding="utf-8") as f:
                content = f.read()

            # Replace key values using simple text substitution
            # Escape user inputs to prevent YAML injection (e.g., colons, quotes, hashes)
            safe_data_file = _yaml_escape_string(data_file)
            safe_output_dir = _yaml_escape_string(output_dir)
            safe_sample_name = _yaml_escape_string(sample_name)
            safe_experiment_id = _yaml_escape_string(experiment_id)

            content = content.replace(
                'file_path: "./data/sample/experiment.hdf"',
                f'file_path: "{safe_data_file}"',
            )
            content = content.replace(
                'directory: "./results"', f'directory: "{safe_output_dir}"'
            )

            # Add sample name and experiment_id if not present
            if "sample_name:" not in content:
                # Insert after directory line in output section
                content = content.replace(
                    f'directory: "{safe_output_dir}"',
                    f'directory: "{safe_output_dir}"\n  sample_name: "{safe_sample_name}"\n  experiment_id: "{safe_experiment_id}"',
                )

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

            # Load config for return value
            with open(template_path, encoding="utf-8") as f:
                result_config = yaml.safe_load(f)
    except OSError as e:
        print(
            f"Error: Cannot write configuration to {output_path}: {e}", file=sys.stderr
        )
        return {}

    print()
    print("=" * 70)
    print(f"OK: Configuration saved: {output_path}")
    print("=" * 70)
    print()
    print("Configuration summary:")
    print(f"  Mode:         {mode}")
    print(f"  Sample:       {sample_name}")
    print(f"  Experiment:   {experiment_id}")
    print(f"  Data file:    {data_file}")
    print(f"  Output dir:   {output_dir}")
    print()
    print("Next steps:")
    print(f"  1. Review configuration: {output_path}")
    print(f"  2. Validate: homodyne-config --validate {output_path}")
    print(f"  3. Run analysis: homodyne --config {output_path}")
    print()
    print("Note: All template comments and instructions are preserved.")

    return result_config


def validate_config(config_path: Path) -> bool:
    """Validate configuration file.

    Parameters
    ----------
    config_path : Path
        Path to configuration file

    Returns
    -------
    bool
        True if valid, False otherwise
    """
    print(f"Validating configuration: {config_path}")
    print("=" * 70)

    # Check file exists
    if not config_path.exists():
        print(f"ERROR: Configuration file not found: {config_path}")
        return False

    # Try loading with ConfigManager
    try:
        config_mgr = ConfigManager(str(config_path))
        config = config_mgr.config

        if config is None:
            print("ERROR: Configuration is empty or failed to load")
            return False

        print("OK: Configuration loaded successfully")
        print()

        # Display key information
        print("Configuration summary:")
        print(f"  Version:      {config.get('config_version', 'unknown')}")
        print(f"  Mode:         {config.get('analysis_mode', 'unknown')}")

        # Data file
        experimental_data = config.get("experimental_data")
        if experimental_data is not None:
            data_file = experimental_data.get("file_path", "not specified")
            print(f"  Data file:    {data_file}")

            # Check if data file exists
            data_path = Path(data_file)
            if data_path.exists():
                print("    OK: Data file exists")
            else:
                print("    WARNING: Data file not found")

        # Parameters
        initial_parameters = config.get("initial_parameters")
        if initial_parameters is not None:
            params = initial_parameters.get("parameter_names", [])
            print(f"  Parameters:   {len(params)} parameters")
            print(f"                {', '.join(params)}")

        # Optimization method
        optimization = config.get("optimization")
        if optimization is not None:
            method = optimization.get("method", "not specified")
            print(f"  Method:       {method}")

        print()
        print("=" * 70)
        print("OK: Configuration is valid")
        print()
        print("Next steps:")
        print(f"  Run analysis: homodyne --config {config_path}")
        print()

        return True

    except Exception as e:
        print("ERROR: Configuration validation failed:")
        print(f"  {type(e).__name__}: {e}")
        print()
        print("Common issues:")
        print("  - Check YAML syntax")
        print("  - Verify required sections exist")
        print("  - Check parameter names match mode")
        print("  - Ensure file paths are correct")
        print()
        return False


def main() -> int:
    """Main entry point for homodyne-config.

    Returns
    -------
    int
        Exit code (0 for success, 1 for error)
    """
    parser = create_parser()
    args = parser.parse_args()

    # Validation mode
    if args.validate:
        success = validate_config(args.validate)
        return 0 if success else 1

    # Interactive mode
    if args.interactive:
        try:
            interactive_builder()
            return 0
        except KeyboardInterrupt:
            print("\n\nInteractive builder cancelled.")
            return 1
        except Exception as e:
            print(f"\nError: {e}")
            return 1

    # Generate mode (requires --mode)
    if args.mode:
        # Default output path
        if args.output is None:
            args.output = Path(f"homodyne_{args.mode}_config.yaml")

        try:
            generate_config(args.mode, args.output, force=args.force)
            return 0
        except FileExistsError as e:
            print(f"ERROR: {e}")
            return 1
        except Exception as e:
            print(f"Error generating configuration: {e}")
            return 1

    # No mode specified
    parser.print_help()
    print()
    print("Error: Please specify --mode, --interactive, or --validate")
    print()
    print("Quick start:")
    print("  homodyne-config --mode static")
    print("  homodyne-config --interactive")
    return 1


if __name__ == "__main__":
    sys.exit(main())
