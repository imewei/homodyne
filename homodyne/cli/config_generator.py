#!/usr/bin/env python3
"""
Configuration File Generator for Homodyne v2
============================================

CLI tool for generating configuration files from templates.
Allows users to create customized YAML config files for analysis.

Usage:
    homodyne-config                                  # Interactive mode
    homodyne-config --mode static_isotropic         # Generate specific template
    homodyne-config --output my_config.yaml         # Custom output file
    homodyne-config --sample "protein_gel" --author "John Doe"  # Custom metadata
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)

# Template mapping - Consolidated to 4 comprehensive templates
TEMPLATE_MAP = {
    # Core templates (comprehensive, production-ready)
    "default": "homodyne_default_comprehensive.yaml",
    "static_isotropic": "homodyne_static_isotropic.yaml",
    "static_anisotropic": "homodyne_static_anisotropic.yaml", 
    "laminar_flow": "homodyne_laminar_flow.yaml",
}

# Dataset size presets - Updated for new template structure
DATASET_SIZE_PRESETS = {
    "small": {
        "performance": {
            "memory_optimization": {
                "max_memory_usage_gb": 2.0,
                "chunk_size": 5000,
                "enable_caching": False,
                "cache_strategy": "conservative"
            }
        },
        "optimization": {
            "vi": {
                "max_iterations": 500,
                "learning_rate": 0.01
            }
        }
    },
    "standard": {
        "performance": {
            "memory_optimization": {
                "max_memory_usage_gb": 8.0,
                "chunk_size": 10000,
                "enable_caching": True,
                "cache_strategy": "adaptive"
            }
        },
        "optimization": {
            "vi": {
                "max_iterations": 1000,
                "learning_rate": 0.01
            }
        }
    },
    "large": {
        "performance": {
            "memory_optimization": {
                "max_memory_usage_gb": 16.0,
                "chunk_size": 20000,
                "enable_caching": True,
                "cache_strategy": "aggressive"
            },
            "performance_engine_enabled": True,
            "io_optimization": {
                "memory_mapped_io": True,
                "parallel_loading": True
            }
        },
        "optimization": {
            "vi": {
                "max_iterations": 2000,
                "learning_rate": 0.005
            }
        }
    },
}


def get_template_path(template_name: str) -> Path:
    """Get path to template file."""
    # Get the homodyne package directory
    homodyne_dir = Path(__file__).parent.parent
    templates_dir = homodyne_dir / "config" / "templates"  # Simplified path

    if template_name in TEMPLATE_MAP:
        template_file = TEMPLATE_MAP[template_name]
    else:
        template_file = f"config_{template_name}.yaml"

    template_path = templates_dir / template_file

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    return template_path


def load_template(template_name: str) -> Dict[str, Any]:
    """Load and parse template file."""
    template_path = get_template_path(template_name)

    try:
        with open(template_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise RuntimeError(f"Error loading template {template_name}: {e}")


def update_config_metadata(
    config: Dict[str, Any],
    sample: Optional[str] = None,
    experiment: Optional[str] = None,
    author: Optional[str] = None,
) -> None:
    """Update config with custom metadata."""
    if not config.get("metadata"):
        config["metadata"] = {}

    # Update timestamp
    config["metadata"]["generated_at"] = datetime.now().isoformat()
    config["metadata"]["generated_by"] = "homodyne-config"

    # Update custom fields if provided
    if sample:
        config["metadata"]["sample_name"] = sample
    if experiment:
        config["metadata"]["experiment_description"] = experiment
    if author:
        config["metadata"]["author"] = author


def apply_dataset_size_preset(config: Dict[str, Any], size: str) -> None:
    """Apply dataset size optimizations to config."""
    if size not in DATASET_SIZE_PRESETS:
        logger.warning(f"Unknown dataset size: {size}")
        return

    preset = DATASET_SIZE_PRESETS[size]

    # Deep merge preset into config
    for section, settings in preset.items():
        if section not in config:
            config[section] = {}
        config[section].update(settings)


def save_config(config: Dict[str, Any], output_path: Path) -> None:
    """Save config to YAML file."""
    try:
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)

        logger.info(f"✅ Configuration saved to: {output_path}")

    except Exception as e:
        raise RuntimeError(f"Error saving config to {output_path}: {e}")


def interactive_mode() -> Dict[str, Any]:
    """Interactive configuration generator with comprehensive template guidance."""
    print("🔧 Homodyne v2 Configuration Generator")
    print("=" * 45)
    print("Generate optimized YAML configurations for XPCS analysis")

    # Mode selection with detailed descriptions
    print("\nSelect analysis mode:")
    mode_descriptions = {
        "default": "Comprehensive reference template with all available options",
        "static_isotropic": "3-parameter single-angle diffusion analysis (D₀, α, D_offset)",
        "static_anisotropic": "3-parameter multi-angle analysis with phi filtering", 
        "laminar_flow": "7-parameter nonequilibrium analysis with time-dependent shear"
    }
    
    modes = list(TEMPLATE_MAP.keys())
    for i, mode in enumerate(modes, 1):
        description = mode_descriptions.get(mode, "")
        print(f"  {i}. {mode}")
        print(f"     {description}")
        print()

    while True:
        try:
            choice = input(
                "Enter mode number (default: 2 for static_isotropic): "
            ).strip()
            if not choice:
                selected_mode = "static_isotropic"
                break
            mode_idx = int(choice) - 1
            if 0 <= mode_idx < len(modes):
                selected_mode = modes[mode_idx]
                break
            print("❌ Invalid selection. Please try again.")
        except ValueError:
            print("❌ Please enter a number.")

    print(f"✅ Selected mode: {selected_mode}")
    
    # Show mode-specific guidance
    if selected_mode == "static_isotropic":
        print("   → 3-parameter model for single-angle diffusion analysis")
        print("   → Best for: routine XPCS analysis, isotropic samples")
    elif selected_mode == "static_anisotropic":
        print("   → 3-parameter model with multi-angle analysis")
        print("   → Best for: anisotropic samples, directional effects")
    elif selected_mode == "laminar_flow":
        print("   → 7-parameter nonequilibrium model with shear")
        print("   → Best for: samples under flow, time-dependent shear")
    elif selected_mode == "default":
        print("   → Comprehensive template with all features documented")
        print("   → Best for: learning all options, custom configurations")

    # Dataset size with descriptions
    print("\nSelect dataset size optimization:")
    size_descriptions = {
        "small": "< 1GB data, 2GB RAM, basic optimization",
        "standard": "1-10GB data, 8GB RAM, balanced performance",
        "large": "> 10GB data, 16GB+ RAM, advanced optimization"
    }
    
    sizes = list(DATASET_SIZE_PRESETS.keys())
    for i, size in enumerate(sizes, 1):
        description = size_descriptions.get(size, "")
        print(f"  {i}. {size}")
        print(f"     {description}")
        print()

    size_choice = input("Enter size number (default: 2 for standard): ").strip()
    if not size_choice:
        selected_size = "standard"
    else:
        try:
            size_idx = int(size_choice) - 1
            if 0 <= size_idx < len(sizes):
                selected_size = sizes[size_idx]
            else:
                selected_size = "standard"
        except ValueError:
            selected_size = "standard"

    print(f"✅ Selected dataset size: {selected_size}")

    # Custom metadata
    print("\nOptional metadata (press Enter to skip):")
    sample = input("Sample name: ").strip() or None
    experiment = input("Experiment description: ").strip() or None
    author = input("Author name: ").strip() or None

    # Output file
    default_output = f"homodyne_{selected_mode}_config.yaml"
    output = input(f"Output file (default: {default_output}): ").strip()
    if not output:
        output = default_output

    # Generate config
    print(f"\n🚀 Generating configuration...")
    config = load_template(selected_mode)
    apply_dataset_size_preset(config, selected_size)
    update_config_metadata(config, sample, experiment, author)

    # Show what was generated
    print(f"   📄 Template: {selected_mode}")
    print(f"   🗂️  Dataset size: {selected_size}")
    print(f"   📁 Output file: {output}")
    
    # Provide usage guidance based on mode
    print(f"\n📋 Next steps:")
    print(f"   1. Edit {output} to customize paths and parameters")
    if selected_mode in ["static_anisotropic", "laminar_flow"]:
        print(f"   2. Ensure your data contains multiple phi angles")
    if selected_mode == "laminar_flow":
        print(f"   3. Consider starting with VI method, then MCMC for full analysis")
    print(f"   4. Run: homodyne --method vi --config {output}")

    return {
        "config": config,
        "output": Path(output),
        "mode": selected_mode,
        "size": selected_size,
    }


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for homodyne-config."""
    parser = argparse.ArgumentParser(
        prog="homodyne-config",
        description="Generate configuration files for Homodyne XPCS analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  homodyne-config                                    # Interactive mode (recommended)
  homodyne-config --mode static_isotropic            # Generate 3-parameter static config
  homodyne-config --mode laminar_flow --output flow_config.yaml
  homodyne-config --sample "protein_gel" --author "John Doe"
  homodyne-config --dataset-size large --mode static_anisotropic

Available templates (consolidated from 8 to 4 comprehensive templates):
  default                - Master reference with all available options and documentation
  static_isotropic       - 3-parameter single-angle diffusion analysis (production-ready)
  static_anisotropic     - 3-parameter multi-angle analysis with phi filtering
  laminar_flow           - 7-parameter nonequilibrium analysis with time-dependent shear

Template Features:
  • All templates include proper instrumental parameters (stator_rotor_gap preserved)
  • Physically reasonable default parameter values and bounds
  • Comprehensive documentation and usage guidance
  • Optimized settings for each analysis mode
  • Production-ready configurations with performance optimization
""",
    )

    parser.add_argument(
        "--mode",
        "-m",
        choices=list(TEMPLATE_MAP.keys()),
        help="Analysis mode template to generate",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output configuration file path (default: homodyne_config.yaml)",
    )

    parser.add_argument(
        "--dataset-size",
        "-d",
        choices=["small", "standard", "large"],
        default="standard",
        help="Dataset size optimization (default: %(default)s)",
    )

    parser.add_argument("--sample", "-s", help="Sample name for metadata")

    parser.add_argument(
        "--experiment", "-e", help="Experiment description for metadata"
    )

    parser.add_argument("--author", "-a", help="Author name for metadata")

    parser.add_argument(
        "--list-templates",
        action="store_true",
        help="List available templates and exit",
    )

    return parser


def main() -> int:
    """Main entry point for homodyne-config."""
    try:
        parser = create_parser()
        args = parser.parse_args()

        # List templates and exit
        if args.list_templates:
            print("Available configuration templates:")
            print("=" * 40)
            for name, filename in TEMPLATE_MAP.items():
                template_path = (
                    Path(__file__).parent.parent / "config" / "templates" / filename
                )
                if template_path.exists():
                    print(f"  ✅ {name:<20} - {filename}")
                else:
                    print(f"  ❌ {name:<20} - {filename} (missing)")
            return 0

        # Interactive mode if no mode specified
        if not args.mode:
            result = interactive_mode()
            config = result["config"]
            output_path = result["output"]
        else:
            # Command line mode
            config = load_template(args.mode)
            apply_dataset_size_preset(config, args.dataset_size)
            update_config_metadata(config, args.sample, args.experiment, args.author)

            if args.output:
                output_path = args.output
            else:
                output_path = Path(f"homodyne_{args.mode}_config.yaml")

        # Save configuration
        save_config(config, output_path)

        print(f"\n✅ Configuration generated successfully!")
        print(f"📄 File: {output_path.absolute()}")
        print(f"\n🚀 Usage:")
        print(f"   homodyne --config {output_path}")

        return 0

    except KeyboardInterrupt:
        print("\n❌ Configuration generation cancelled")
        return 1
    except Exception as e:
        logger.error(f"❌ Error generating configuration: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
