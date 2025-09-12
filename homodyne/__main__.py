"""
Main Entry Point for Homodyne v2
================================

Primary entry point when homodyne is called as a module:
    python -m homodyne [args...]

This provides the same functionality as the CLI entry point
but allows execution as a module for various deployment scenarios.

Key Features:
- Module execution support (python -m homodyne)
- Identical functionality to CLI entry point
- Error handling for missing dependencies
- Version checking and dependency validation
"""

import sys
from pathlib import Path

def main() -> int:
    """
    Main entry point for module execution.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Import the CLI main function
        from homodyne.cli.main import main as cli_main
        
        # Execute the CLI with current arguments
        return cli_main()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Ensure homodyne is properly installed with all dependencies")
        return 1
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("   Please report this issue at: https://github.com/your-org/homodyne/issues")
        return 1


if __name__ == "__main__":
    # This allows: python -m homodyne
    sys.exit(main())