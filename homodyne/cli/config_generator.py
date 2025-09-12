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
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)

# Template mapping - Updated for flattened structure
TEMPLATE_MAP = {
    # Minimal templates (new, clean, working)
    "static_isotropic": "config_minimal_static_isotropic.yaml",
    "static_anisotropic": "config_minimal_static_anisotropic.yaml", 
    "laminar_flow": "config_minimal_laminar_flow.yaml",
    
    # Standard template (comprehensive but clean)
    "default": "config_default_template.yaml",
    
    # Legacy templates removed - no longer available
    
    # Other advanced templates
    "enhanced_logging": "config_enhanced_logging_template.yaml",
    "quality_control": "config_quality_control_template.yaml",
    "performance_optimized": "config_performance_optimized_template.yaml",
    "advanced_validation": "config_advanced_validation_demo.yaml",
}

# Dataset size presets
DATASET_SIZE_PRESETS = {
    "small": {
        "data_processing": {
            "max_memory_usage_gb": 2,
            "chunk_size": 1000,
            "enable_caching": False
        }
    },
    "standard": {
        "data_processing": {
            "max_memory_usage_gb": 8,
            "chunk_size": 5000,
            "enable_caching": True
        }
    },
    "large": {
        "data_processing": {
            "max_memory_usage_gb": 32,
            "chunk_size": 10000,
            "enable_caching": True,
            "enable_memory_mapping": True
        }
    }
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
        with open(template_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise RuntimeError(f"Error loading template {template_name}: {e}")


def update_config_metadata(config: Dict[str, Any], 
                         sample: Optional[str] = None,
                         experiment: Optional[str] = None, 
                         author: Optional[str] = None) -> None:
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
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
        
        logger.info(f"✅ Configuration saved to: {output_path}")
        
    except Exception as e:
        raise RuntimeError(f"Error saving config to {output_path}: {e}")


def interactive_mode() -> Dict[str, Any]:
    """Interactive configuration generator."""
    print("🔧 Homodyne Configuration Generator")
    print("=" * 40)
    
    # Mode selection
    print("\nSelect analysis mode:")
    modes = list(TEMPLATE_MAP.keys())
    for i, mode in enumerate(modes, 1):
        print(f"  {i}. {mode}")
    
    while True:
        try:
            choice = input("\nEnter mode number (default: 1 for static_isotropic): ").strip()
            if not choice:
                selected_mode = "static_isotropic"
                break
            mode_idx = int(choice) - 1
            if 0 <= mode_idx < len(modes):
                selected_mode = modes[mode_idx]
                break
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a number.")
    
    print(f"Selected mode: {selected_mode}")
    
    # Dataset size
    print("\nSelect dataset size optimization:")
    sizes = list(DATASET_SIZE_PRESETS.keys())
    for i, size in enumerate(sizes, 1):
        print(f"  {i}. {size}")
    
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
    
    print(f"Selected dataset size: {selected_size}")
    
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
    
    return {
        "config": config,
        "output": Path(output),
        "mode": selected_mode,
        "size": selected_size
    }


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for homodyne-config."""
    parser = argparse.ArgumentParser(
        prog="homodyne-config",
        description="Generate configuration files for Homodyne XPCS analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  homodyne-config                                    # Interactive mode
  homodyne-config --mode static_isotropic            # Generate basic static config
  homodyne-config --mode laminar_flow --output flow_config.yaml
  homodyne-config --sample "protein_gel" --author "John Doe"
  homodyne-config --dataset-size large --mode static_anisotropic

Available templates:
  static_isotropic       - 3 parameters, no angle filtering
  static_anisotropic     - 3 parameters + angle filtering  
  laminar_flow           - 7 parameters + filtering
  default                - General purpose template
  enhanced_logging       - Enhanced logging and debugging
  quality_control        - Advanced data quality validation
  performance_optimized  - Memory and performance optimized
  advanced_validation    - Comprehensive validation demo
"""
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=list(TEMPLATE_MAP.keys()),
        help="Analysis mode template to generate"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output configuration file path (default: homodyne_config.yaml)"
    )
    
    parser.add_argument(
        "--dataset-size", "-d",
        choices=["small", "standard", "large"],
        default="standard",
        help="Dataset size optimization (default: %(default)s)"
    )
    
    parser.add_argument(
        "--sample", "-s",
        help="Sample name for metadata"
    )
    
    parser.add_argument(
        "--experiment", "-e", 
        help="Experiment description for metadata"
    )
    
    parser.add_argument(
        "--author", "-a",
        help="Author name for metadata"
    )
    
    parser.add_argument(
        "--list-templates",
        action="store_true",
        help="List available templates and exit"
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
                template_path = Path(__file__).parent.parent / "config" / "templates" / filename
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