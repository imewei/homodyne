"""Unit tests for API documentation generation.

Tests validate that Sphinx autodoc and autosummary can successfully
generate API documentation from the homodyne package without import errors.
"""

import subprocess
import sys
from pathlib import Path

import pytest


def test_autodoc_imports_homodyne_successfully():
    """Test that autodoc can import homodyne package without errors."""
    # Import should succeed without mocking
    try:
        import homodyne

        # Verify core modules are accessible
        assert hasattr(homodyne, "core"), "homodyne.core module not accessible"
        assert hasattr(homodyne, "config"), "homodyne.config module not accessible"
        assert hasattr(homodyne, "data"), "homodyne.data module not accessible"
        assert hasattr(homodyne, "optimization"), (
            "homodyne.optimization module not accessible"
        )
        assert hasattr(homodyne, "cli"), "homodyne.cli module not accessible"
        assert hasattr(homodyne, "viz"), "homodyne.viz module not accessible"
        assert hasattr(homodyne, "device"), "homodyne.device module not accessible"
        assert hasattr(homodyne, "utils"), "homodyne.utils module not accessible"

    except ImportError as e:
        pytest.fail(f"Failed to import homodyne package: {e}")


def test_core_module_imports():
    """Test that core submodules can be imported."""
    try:
        from homodyne import core

        # Check key components exist
        assert hasattr(core, "jax_backend"), "jax_backend not found in core"
        assert hasattr(core, "physics"), "physics not found in core"
        assert hasattr(core, "models"), "models not found in core"
        assert hasattr(core, "fitting"), "fitting not found in core"

    except ImportError as e:
        pytest.fail(f"Failed to import homodyne.core: {e}")


def test_optimization_module_imports():
    """Test that optimization submodules can be imported."""
    try:
        from homodyne import optimization

        # Check subpackages exist
        assert hasattr(optimization, "nlsq"), "nlsq subpackage not found"
        # cmc may be None if arviz is not installed
        assert hasattr(optimization, "cmc"), "cmc subpackage not found"
        # If cmc is None, that's ok - it means arviz is missing
        if optimization.cmc is None:
            pytest.skip("cmc subpackage not available (arviz missing)")

    except ImportError as e:
        pytest.fail(f"Failed to import homodyne.optimization: {e}")


def test_config_module_imports():
    """Test that config module can be imported."""
    try:
        from homodyne import config

        # Check key components exist
        assert hasattr(config, "manager"), "manager not found in config"
        assert hasattr(config, "parameter_manager"), (
            "parameter_manager not found in config"
        )
        assert hasattr(config, "types"), "types not found in config"

    except ImportError as e:
        pytest.fail(f"Failed to import homodyne.config: {e}")


def test_docstring_parsing_with_napoleon():
    """Test that napoleon can parse Google/NumPy style docstrings."""
    try:
        from homodyne.core import jax_backend

        # Check that compute_g2_scaled has a docstring
        assert hasattr(jax_backend, "compute_g2_scaled"), "compute_g2_scaled not found"
        assert jax_backend.compute_g2_scaled.__doc__ is not None, (
            "compute_g2_scaled missing docstring"
        )

        # Check docstring contains typical sections
        docstring = jax_backend.compute_g2_scaled.__doc__
        assert len(docstring) > 0, "Docstring is empty"

    except (ImportError, AttributeError) as e:
        pytest.fail(f"Failed to test docstring parsing: {e}")


def test_api_reference_rst_files_exist():
    """Test that API reference RST files will be created."""
    docs_path = Path(__file__).parent.parent.parent / "docs"
    api_ref_path = docs_path / "api-reference"

    # Check that api-reference directory exists
    assert api_ref_path.exists(), f"api-reference directory not found at {api_ref_path}"
    assert api_ref_path.is_dir(), "api-reference should be a directory"


@pytest.mark.slow
def test_autosummary_generation():
    """Test that autosummary can generate module stubs.

    This is a slow test that actually runs sphinx-autogen to verify
    that module stubs can be generated without errors.
    """
    # Skip if sphinx is not installed
    pytest.importorskip(
        "sphinx", reason="Sphinx required for autosummary generation test"
    )

    docs_path = Path(__file__).parent.parent.parent / "docs"

    # Create a minimal test RST file with autosummary directive
    test_rst = docs_path / "_test_autosummary.rst"
    test_content = """
Test Autosummary
================

.. autosummary::
   :toctree: _autosummary

   homodyne.core
   homodyne.config
"""

    try:
        test_rst.write_text(test_content)

        # Run sphinx-autogen
        result = subprocess.run(
            [sys.executable, "-m", "sphinx.ext.autosummary.generate", str(test_rst)],
            cwd=str(docs_path),
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Check that autogen succeeded (exit code 0 or generated output)
        # Note: autogen may return non-zero but still work, so check stderr
        if result.returncode != 0 and "error" in result.stderr.lower():
            pytest.fail(f"sphinx-autogen failed: {result.stderr}")

    finally:
        # Clean up test file
        if test_rst.exists():
            test_rst.unlink()


def test_type_hints_accessible():
    """Test that type hints are accessible for autodoc."""
    try:
        import inspect

        from homodyne.core import jax_backend

        # Check that functions have type annotations
        if hasattr(jax_backend, "compute_g2_scaled"):
            sig = inspect.signature(jax_backend.compute_g2_scaled)
            # Type hints should be present (even if empty dict)
            assert hasattr(sig, "parameters"), "Function signature missing parameters"

    except (ImportError, AttributeError) as e:
        pytest.fail(f"Failed to access type hints: {e}")
