"""
Setup Configuration for Homodyne v2
===================================

Enhanced setup.py with comprehensive dependency management,
installation options, and entry point configuration.

Key Features:
- Tiered dependency groups (minimal, standard, full)
- Multiple installation options (pip install homodyne[full])
- CLI entry point registration  
- Platform-specific optimizations
- Development tooling integration
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Get the directory containing setup.py
HERE = Path(__file__).parent.resolve()

# Read the long description from README
def read_readme():
    """Read README file for long description."""
    readme_path = HERE / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return "Advanced X-ray Photon Correlation Spectroscopy (XPCS) analysis with JAX acceleration"

# Read version from homodyne/__init__.py
def read_version():
    """Read version from package __init__.py."""
    init_path = HERE / "homodyne" / "__init__.py"
    if init_path.exists():
        with open(init_path, 'r') as f:
            for line in f:
                if line.startswith('__version__'):
                    # Extract version string
                    version = line.split('=')[1].strip().strip('"\'')
                    return version
    return "2.0.0"  # Fallback version

# Define dependency groups
INSTALL_REQUIRES = [
    # Core dependencies (always required)
    "numpy>=1.21.0",
    "scipy>=1.7.0", 
    "pyyaml>=5.4.0",
    "pathlib2>=2.3.0;python_version<'3.4'",
]

EXTRAS_REQUIRE = {
    # Minimal installation (core functionality only)
    "minimal": [],
    
    # Standard installation (recommended for most users)
    "standard": [
        "jax>=0.4.0",
        "jaxlib>=0.4.0",
        "h5py>=3.1.0",
        "matplotlib>=3.3.0",
        "pandas>=1.3.0",
    ],
    
    # Full installation (all features)
    "full": [
        "jax>=0.4.0",
        "jaxlib>=0.4.0", 
        "h5py>=3.1.0",
        "matplotlib>=3.3.0",
        "pandas>=1.3.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "jupyter>=1.0.0",
        "notebook>=6.4.0",
        "ipywidgets>=7.6.0",
    ],
    
    # GPU support (CUDA-enabled JAX)
    "gpu": [
        "jax[cuda]>=0.4.0",
        "jaxlib[cuda]>=0.4.0",
    ],
    
    # Development dependencies
    "dev": [
        "pytest>=6.2.0",
        "pytest-cov>=2.12.0",
        "pytest-mock>=3.6.0",
        "black>=21.0.0",
        "ruff>=0.0.290",
        "mypy>=0.910",
        "pre-commit>=2.15.0",
        "sphinx>=4.0.0",
        "sphinx-rtd-theme>=0.5.0",
    ],
    
    # Performance testing
    "perf": [
        "pytest-benchmark>=3.4.0",
        "memory_profiler>=0.58.0",
        "psutil>=5.8.0",
    ],
    
    # Documentation
    "docs": [
        "sphinx>=4.0.0",
        "sphinx-rtd-theme>=0.5.0",
        "myst-parser>=0.15.0",
        "sphinx-autodoc-typehints>=1.12.0",
        "nbsphinx>=0.8.0",
    ],
}

# Combine extras for convenience
EXTRAS_REQUIRE["all"] = list(set(sum(EXTRAS_REQUIRE.values(), [])))

# Platform-specific dependencies
if sys.platform.startswith("win"):
    EXTRAS_REQUIRE["standard"].append("colorama>=0.4.0")

# Python version-specific dependencies
if sys.version_info < (3, 8):
    INSTALL_REQUIRES.append("typing_extensions>=3.7.0")

# Entry points for console scripts
ENTRY_POINTS = {
    "console_scripts": [
        "homodyne=homodyne.cli.main:main",
        "homodyne-config=homodyne.cli.config_generator:main",
    ]
}

# Package metadata
PACKAGE_DATA = {
    "homodyne": [
        "config/templates/**/*.yaml",
        "config/templates/**/*.json",
        "data/examples/**/*",
        "docs/**/*.md",
        "tests/data/**/*",
    ]
}

# Classifiers for PyPI
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Physics",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9", 
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Operating System :: OS Independent",
    "Framework :: Jupyter",
]

# Keywords for PyPI search
KEYWORDS = [
    "xpcs", "x-ray", "photon correlation spectroscopy", "scattering",
    "anomalous diffusion", "laminar flow", "jax", "bayesian",
    "variational inference", "mcmc", "scientific computing"
]


def check_python_version():
    """Check if Python version is supported."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        sys.exit(1)


def post_install_message():
    """Display post-installation information."""
    print("\n" + "="*60)
    print("🎉 Homodyne v2 installation completed successfully!")
    print("="*60)
    
    print("\n📋 Quick Start:")
    print("   homodyne --help                    # Show help")
    print("   homodyne data.h5 --method vi      # Run VI analysis")
    print("   homodyne data.h5 --method mcmc    # Run MCMC analysis")
    print("   homodyne data.h5 --method hybrid  # Run hybrid analysis")
    
    print("\n🐍 Python API:")
    print("   from homodyne.api import run_analysis")
    print("   result = run_analysis('data.h5', method='vi')")
    
    print("\n🚀 Optional Features:")
    print("   pip install homodyne[gpu]         # GPU acceleration")
    print("   pip install homodyne[full]        # All features")
    print("   pip install homodyne[dev]         # Development tools")
    
    print("\n📚 Documentation:")
    print("   https://homodyne.readthedocs.io")
    
    print("\n🐛 Issues & Support:")
    print("   https://github.com/your-org/homodyne/issues")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # Check Python version before installation
    check_python_version()
    
    # Run setup
    setup(
        # Basic package information
        name="homodyne",
        version=read_version(),
        description="Advanced X-ray Photon Correlation Spectroscopy (XPCS) analysis with JAX acceleration",
        long_description=read_readme(),
        long_description_content_type="text/markdown",
        
        # Author and contact information
        author="Homodyne Development Team",
        author_email="homodyne-dev@example.com",
        maintainer="Homodyne Development Team",
        maintainer_email="homodyne-dev@example.com",
        
        # URLs and links
        url="https://github.com/your-org/homodyne",
        project_urls={
            "Documentation": "https://homodyne.readthedocs.io",
            "Bug Reports": "https://github.com/your-org/homodyne/issues",
            "Source": "https://github.com/your-org/homodyne",
            "Changelog": "https://github.com/your-org/homodyne/blob/main/CHANGELOG.md",
        },
        
        # Package discovery and inclusion
        packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
        package_data=PACKAGE_DATA,
        include_package_data=True,
        
        # Dependencies
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        python_requires=">=3.8",
        
        # Entry points
        entry_points=ENTRY_POINTS,
        
        # Metadata for PyPI
        classifiers=CLASSIFIERS,
        keywords=KEYWORDS,
        license="MIT",
        
        # Build configuration
        zip_safe=False,  # Required for proper package data access
        platforms=["any"],
        
        # Additional options
        options={
            "bdist_wheel": {
                "universal": False,  # Python 3 only
            }
        },
    )
    
    # Show post-installation message
    post_install_message()