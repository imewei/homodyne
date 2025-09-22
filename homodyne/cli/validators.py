"""
Input Validation for Homodyne v2 CLI
====================================

Comprehensive validation of CLI arguments with clear error messages.
Validates file paths, parameter ranges, and argument consistency.
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


class ValidationError(ValueError):
    """Custom exception for CLI validation errors."""

    pass


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate all CLI arguments for consistency and correctness.

    Args:
        args: Parsed command line arguments

    Raises:
        ValidationError: If any validation fails
    """
    logger.debug("Starting CLI argument validation")

    # Note: verbose and quiet arguments have been removed from CLI
    # Any legacy references are handled via backwards compatibility

    # Validate configuration file
    validate_config_file(args.config)

    # Validate output directory
    validate_output_directory(args.output_dir)

    # Validate scaling parameters usage
    validate_scaling_parameters(args)

    # Validate phi angles if provided
    if args.phi_angles:
        validate_phi_angles(args.phi_angles)

    # Validate GPU memory fraction
    if args.gpu_memory_fraction:
        validate_gpu_memory_fraction(args.gpu_memory_fraction)

    # Method-specific validations
    validate_method_specific_args(args)

    logger.debug("✓ All CLI arguments validated successfully")


def validate_config_file(config_path: Path) -> None:
    """
    Validate configuration file exists and is readable.

    Args:
        config_path: Path to configuration file

    Raises:
        ValidationError: If config file is invalid
    """
    if not config_path.exists():
        raise ValidationError(
            f"Configuration file not found: {config_path.absolute()}\n"
            f"Please check the file path and ensure the configuration file exists.\n"
            f"You can create a default config with: homodyne --create-config"
        )

    if not config_path.is_file():
        raise ValidationError(
            f"Configuration path is not a file: {config_path.absolute()}"
        )

    if not os.access(config_path, os.R_OK):
        raise ValidationError(
            f"Configuration file is not readable: {config_path.absolute()}\n"
            f"Please check file permissions."
        )

    # Validate file extension
    valid_extensions = {".yaml", ".yml", ".json"}
    if config_path.suffix.lower() not in valid_extensions:
        logger.warning(
            f"Configuration file extension '{config_path.suffix}' not in {valid_extensions}. "
            f"Will attempt to parse as YAML/JSON."
        )


def validate_output_directory(output_dir: Path) -> None:
    """
    Validate output directory is writable.

    Args:
        output_dir: Path to output directory

    Raises:
        ValidationError: If output directory is not writable
    """
    try:
        # Create directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Test writeability
        test_file = output_dir / ".homodyne_write_test"
        try:
            test_file.touch()
            test_file.unlink()
        except (PermissionError, OSError) as e:
            raise ValidationError(
                f"Output directory is not writable: {output_dir.absolute()}\n"
                f"Error: {e}\n"
                f"Please check directory permissions or choose a different output directory."
            )

    except (PermissionError, OSError) as e:
        raise ValidationError(
            f"Cannot create output directory: {output_dir.absolute()}\n"
            f"Error: {e}\n"
            f"Please check parent directory permissions or choose a different path."
        )


def validate_scaling_parameters(args: argparse.Namespace) -> None:
    """
    Validate scaling parameters are only used with appropriate flags.

    Args:
        args: Parsed arguments

    Raises:
        ValidationError: If scaling parameters used incorrectly
    """
    contrast_specified = args.contrast != 0.3  # Match actual default from args_parser
    offset_specified = args.offset != 1.0  # Match actual default from args_parser

    if (contrast_specified or offset_specified) and not args.plot_simulated_data:
        raise ValidationError(
            "The --contrast and --offset parameters can only be used with --plot-simulated-data.\n"
            f"Current values: contrast={args.contrast}, offset={args.offset}\n"
            f"Either add --plot-simulated-data or remove the scaling parameters."
        )

    # Validate contrast range
    if args.contrast <= 0:
        raise ValidationError(
            f"Contrast must be positive, got: {args.contrast}\n"
            f"Typical range: 0.1 to 2.0"
        )

    if args.contrast > 10:
        logger.warning(
            f"Contrast value {args.contrast} is unusually high (typical range: 0.1-2.0). "
            f"Please verify this is intentional."
        )


def validate_phi_angles(phi_angles_str: str) -> List[float]:
    """
    Validate and parse phi angles string.

    Args:
        phi_angles_str: Comma-separated angles string

    Returns:
        List of validated phi angles in degrees

    Raises:
        ValidationError: If angles are invalid
    """
    try:
        # Parse comma-separated values
        angle_strs = [s.strip() for s in phi_angles_str.split(",")]
        angles = []

        for angle_str in angle_strs:
            if not angle_str:
                continue

            try:
                angle = float(angle_str)
                angles.append(angle)
            except ValueError:
                raise ValidationError(
                    f"Invalid phi angle: '{angle_str}' is not a valid number\n"
                    f"Expected format: '0,45,90,135' (comma-separated degrees)"
                )

        if not angles:
            raise ValidationError(
                "No valid phi angles found in the provided string.\n"
                f"Input: '{phi_angles_str}'\n"
                f"Expected format: '0,45,90,135' (comma-separated degrees)"
            )

        # Validate angle ranges
        for angle in angles:
            if angle < 0 or angle >= 360:
                raise ValidationError(
                    f"Phi angle {angle}° is outside valid range [0, 360)\n"
                    f"All angles must be between 0 and 360 degrees"
                )

        # Check for duplicates
        unique_angles = list(set(angles))
        if len(unique_angles) != len(angles):
            logger.warning(
                f"Duplicate phi angles detected: {angles}\n"
                f"Will use unique values: {sorted(unique_angles)}"
            )

        logger.debug(f"✓ Validated phi angles: {angles}")
        return angles

    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(
            f"Error parsing phi angles: {e}\n"
            f"Input: '{phi_angles_str}'\n"
            f"Expected format: '0,45,90,135' (comma-separated degrees)"
        )


def validate_gpu_memory_fraction(fraction: float) -> None:
    """
    Validate GPU memory fraction is in valid range.

    Args:
        fraction: GPU memory fraction

    Raises:
        ValidationError: If fraction is invalid
    """
    if not (0.1 <= fraction <= 1.0):
        raise ValidationError(
            f"GPU memory fraction must be between 0.1 and 1.0, got: {fraction}\n"
            f"Recommended values: 0.7-0.9 for shared GPUs, 0.8-0.95 for dedicated GPUs"
        )

    if fraction < 0.3:
        logger.warning(
            f"GPU memory fraction {fraction} is quite low. "
            f"This may limit performance for large datasets."
        )


def validate_method_specific_args(args: argparse.Namespace) -> None:
    """
    Validate method-specific argument combinations.

    Args:
        args: Parsed arguments

    Raises:
        ValidationError: If method-specific validation fails
    """
    # GPU-specific validations
    if args.force_cpu and args.gpu_memory_fraction != 0.8:
        logger.warning(
            "GPU memory fraction specified with --force-cpu. "
            "The memory fraction will be ignored in CPU-only mode."
        )

    # Analysis mode warnings
    if args.static_isotropic and args.method == "mcmc":
        logger.info(
            "Using MCMC with static isotropic mode (3 parameters). "
            "Consider LSQ for faster analysis of simple models."
        )



def validate_file_format(file_path: Path, expected_formats: List[str]) -> None:
    """
    Validate file has expected format.

    Args:
        file_path: Path to file
        expected_formats: List of valid extensions (e.g., ['.yaml', '.json'])

    Raises:
        ValidationError: If file format is invalid
    """
    if file_path.suffix.lower() not in expected_formats:
        raise ValidationError(
            f"File '{file_path}' has unsupported format '{file_path.suffix}'\n"
            f"Supported formats: {expected_formats}"
        )


def validate_parameter_range(
    value: Union[int, float],
    name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    warn_outside: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Validate parameter is within acceptable range.

    Args:
        value: Parameter value
        name: Parameter name for error messages
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        warn_outside: Tuple of (min, max) for warning range

    Raises:
        ValidationError: If value is outside allowed range
    """
    if min_val is not None and value < min_val:
        raise ValidationError(f"{name} must be >= {min_val}, got: {value}")

    if max_val is not None and value > max_val:
        raise ValidationError(f"{name} must be <= {max_val}, got: {value}")

    if warn_outside:
        warn_min, warn_max = warn_outside
        if value < warn_min or value > warn_max:
            logger.warning(
                f"{name} value {value} is outside typical range [{warn_min}, {warn_max}]. "
                f"Please verify this is intentional."
            )
