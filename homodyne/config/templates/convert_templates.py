#!/usr/bin/env python3
"""
Template Conversion Script: JSON to YAML
=======================================

Converts v1 JSON configuration templates to v2 YAML format templates.
Uses existing migration utilities to preserve all configuration structure
while adding v2 features and enhancements.

Usage:
    python convert_templates.py [--verbose]
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List
import logging

# Add homodyne to path for imports
homodyne_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(homodyne_root))

try:
    from homodyne.data.migration import migrate_json_to_yaml, MigrationResult
    from homodyne.data.config import save_yaml_config
    from homodyne.utils.logging import get_logger
except ImportError as e:
    print(f"Error importing homodyne modules: {e}")
    print("Make sure you're running from the homodyne package root directory")
    sys.exit(1)

# Logging setup
logger = get_logger(__name__)


def convert_all_templates(source_dir: str, target_dir: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Convert all JSON templates in source directory to YAML format in target directory.
    
    Args:
        source_dir: Path to v1_json templates directory
        target_dir: Path to v2_yaml templates directory  
        verbose: Enable verbose logging
        
    Returns:
        Dictionary with conversion results and statistics
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    if verbose:
        logging.basicConfig(level=logging.INFO)
    
    logger.info(f"Converting templates from {source_path} to {target_path}")
    
    # Ensure target directory exists
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Find all JSON template files
    json_files = list(source_path.glob("*.json"))
    
    if not json_files:
        logger.warning(f"No JSON files found in {source_path}")
        return {"converted": 0, "failed": 0, "results": []}
    
    results = {
        "converted": 0,
        "failed": 0,  
        "results": [],
        "files": []
    }
    
    for json_file in json_files:
        logger.info(f"Processing {json_file.name}")
        
        # Generate YAML filename (preserve basename, change extension)
        yaml_filename = json_file.stem + ".yaml"
        yaml_file = target_path / yaml_filename
        
        try:
            # Use migration utility to convert
            migration_result = migrate_json_to_yaml(
                json_path=json_file,
                yaml_path=yaml_file,
                add_v2_features=True  # Add v2 enhancements
            )
            
            if migration_result.success:
                results["converted"] += 1
                logger.info(f"✓ Successfully converted {json_file.name} → {yaml_filename}")
                
                # Log any warnings from conversion
                if migration_result.warnings:
                    for warning in migration_result.warnings:
                        logger.warning(f"  Warning: {warning}")
            else:
                results["failed"] += 1
                logger.error(f"✗ Failed to convert {json_file.name}: {migration_result.error}")
            
            results["results"].append(migration_result)
            results["files"].append({
                "source": str(json_file),
                "target": str(yaml_file), 
                "success": migration_result.success
            })
            
        except Exception as e:
            results["failed"] += 1
            logger.error(f"✗ Exception converting {json_file.name}: {str(e)}")
            results["results"].append(MigrationResult(
                success=False,
                error=str(e),
                source_path=str(json_file),
                target_path=str(yaml_file)
            ))
    
    return results


def add_template_metadata(yaml_files: List[Path]) -> None:
    """
    Add template-specific metadata and comments to converted YAML files.
    
    Args:
        yaml_files: List of converted YAML template files
    """
    logger.info("Adding template metadata and documentation")
    
    for yaml_file in yaml_files:
        if not yaml_file.exists():
            continue
            
        # Read existing content
        with open(yaml_file, 'r') as f:
            content = f.read()
        
        # Determine template type from filename
        template_type = "general"
        if "static_isotropic" in yaml_file.name:
            template_type = "static_isotropic"
        elif "static_anisotropic" in yaml_file.name:
            template_type = "static_anisotropic"
        elif "laminar_flow" in yaml_file.name:
            template_type = "laminar_flow"
        
        # Create header comment based on template type
        header_comments = {
            "general": """# Homodyne v2 Configuration Template
# ===================================
# General purpose template with all available configuration options.
# This template demonstrates the full range of v2 features and settings.

""",
            "static_isotropic": """# Homodyne v2 Template: Static Isotropic Analysis
# ==============================================
# Template optimized for static isotropic diffusion analysis.
# Models: D₀, α, D_offset (3 parameters)

""",
            "static_anisotropic": """# Homodyne v2 Template: Static Anisotropic Analysis  
# ================================================
# Template optimized for static anisotropic analysis with angle filtering.
# Models: D₀, α, D_offset with directional dependencies (3 parameters)

""",
            "laminar_flow": """# Homodyne v2 Template: Laminar Flow Analysis
# ==========================================
# Template optimized for laminar flow analysis under shear.
# Models: D₀, α, D_offset, γ̇₀, β, γ̇_offset, φ₀ (7 parameters)

"""
        }
        
        # Prepend header and write back
        with open(yaml_file, 'w') as f:
            f.write(header_comments.get(template_type, header_comments["general"]))
            f.write(content)
        
        logger.info(f"Added metadata to {yaml_file.name}")


def main():
    """Main conversion script entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert v1 JSON templates to v2 YAML templates"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--source", "-s",
        default="v1_json",
        help="Source directory for JSON templates (relative to script location)"
    )
    parser.add_argument(
        "--target", "-t", 
        default="v2_yaml",
        help="Target directory for YAML templates (relative to script location)"
    )
    
    args = parser.parse_args()
    
    # Make paths relative to script location
    script_dir = Path(__file__).parent
    source_path = script_dir / args.source
    target_path = script_dir / args.target
    
    print("🔄 Homodyne Template Converter: JSON → YAML")
    print("=" * 50)
    
    # Convert templates
    results = convert_all_templates(str(source_path), str(target_path), args.verbose)
    
    # Add template-specific metadata
    if results["converted"] > 0:
        yaml_files = list(target_path.glob("*.yaml"))
        add_template_metadata(yaml_files)
    
    # Print summary
    print(f"\n📊 Conversion Summary:")
    print(f"  ✓ Converted: {results['converted']} templates")
    print(f"  ✗ Failed: {results['failed']} templates")
    
    if args.verbose and results["results"]:
        print(f"\n📋 Detailed Results:")
        for result in results["results"]:
            status = "✓" if result.success else "✗"
            source_name = Path(result.source_path).name if result.source_path else "unknown"
            print(f"  {status} {source_name}")
            if result.error:
                print(f"    Error: {result.error}")
            if result.warnings:
                for warning in result.warnings:
                    print(f"    Warning: {warning}")
    
    if results["converted"] > 0:
        print(f"\n🎉 Templates successfully converted to {target_path}")
    
    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())