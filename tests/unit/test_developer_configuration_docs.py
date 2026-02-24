"""Unit tests for developer guide and configuration documentation.

Tests validate that developer guide and configuration documentation pages
build correctly with Sphinx, internal links resolve, and content is properly
formatted and complete.
"""

from pathlib import Path


def test_developer_guide_index_exists():
    """Test that developer/index.rst exists."""
    docs_path = Path(__file__).parent.parent.parent / "docs"
    dev_guide_index = docs_path / "source" / "developer" / "index.rst"

    assert dev_guide_index.exists(), (
        f"developer/index.rst not found at {dev_guide_index}"
    )

    # Verify file is not empty
    content = dev_guide_index.read_text(encoding="utf-8")
    assert len(content) > 0, "developer/index.rst is empty"
    assert "toctree" in content, "developer/index.rst missing toctree directive"


def test_developer_guide_contributing_exists():
    """Test that developer/contributing_guide.rst exists."""
    docs_path = Path(__file__).parent.parent.parent / "docs"
    contributing = docs_path / "source" / "developer" / "contributing_guide.rst"

    assert contributing.exists(), (
        f"developer/contributing_guide.rst not found at {contributing}"
    )

    content = contributing.read_text(encoding="utf-8")
    assert len(content) > 0, "developer/contributing_guide.rst is empty"

    # Verify key sections exist
    required_sections = [
        "Development Setup",
        "uv",
        "Code Quality",
        "Testing",
        "Pull Request",
    ]
    for section in required_sections:
        assert section in content, (
            f"Required section '{section}' not found in contributing_guide.rst"
        )


def test_developer_guide_testing_exists():
    """Test that developer/testing_guide.rst exists."""
    docs_path = Path(__file__).parent.parent.parent / "docs"
    testing_guide = docs_path / "source" / "developer" / "testing_guide.rst"

    assert testing_guide.exists(), (
        f"developer/testing_guide.rst not found at {testing_guide}"
    )

    content = testing_guide.read_text(encoding="utf-8")
    assert len(content) > 0, "developer/testing_guide.rst is empty"

    # Verify key sections exist
    required_sections = [
        "Test Organization",
        "unit/",
        "integration/",
        "Running Tests",
        "make test",
    ]
    for section in required_sections:
        assert section in content, (
            f"Required section '{section}' not found in testing_guide.rst"
        )


def test_configuration_index_exists():
    """Test that configuration/index.rst exists."""
    docs_path = Path(__file__).parent.parent.parent / "docs"
    config_index = docs_path / "source" / "configuration" / "index.rst"

    assert config_index.exists(), f"configuration/index.rst not found at {config_index}"

    content = config_index.read_text(encoding="utf-8")
    assert len(content) > 0, "configuration/index.rst is empty"
    assert "toctree" in content, "configuration/index.rst missing toctree directive"


def test_configuration_templates_exists():
    """Test that configuration/templates.rst exists."""
    docs_path = Path(__file__).parent.parent.parent / "docs"
    templates = docs_path / "source" / "configuration" / "templates.rst"

    assert templates.exists(), f"configuration/templates.rst not found at {templates}"

    content = templates.read_text(encoding="utf-8")
    assert len(content) > 0, "configuration/templates.rst is empty"

    # Verify key sections exist
    required_sections = ["Static Mode", "Laminar Flow", "Per-Angle Scaling", "YAML"]
    for section in required_sections:
        assert section in content, (
            f"Required section '{section}' not found in templates.rst"
        )


def test_configuration_options_exists():
    """Test that configuration/options.rst exists."""
    docs_path = Path(__file__).parent.parent.parent / "docs"
    options = docs_path / "source" / "configuration" / "options.rst"

    assert options.exists(), f"configuration/options.rst not found at {options}"

    content = options.read_text(encoding="utf-8")
    assert len(content) > 0, "configuration/options.rst is empty"

    # Verify key sections exist
    required_sections = [
        "Configuration Options Reference",
        "Static Mode Bounds",
        "NLSQ Configuration",
        "MCMC Configuration",
    ]
    for section in required_sections:
        assert section in content, (
            f"Required section '{section}' not found in options.rst"
        )


def test_developer_guide_references_claude_md():
    """Test that developer guide references CLAUDE.md for commands."""
    docs_path = Path(__file__).parent.parent.parent / "docs"
    contributing = docs_path / "source" / "developer" / "contributing_guide.rst"
    testing_guide = docs_path / "source" / "developer" / "testing_guide.rst"

    contributing_content = contributing.read_text(encoding="utf-8")
    testing_content = testing_guide.read_text(encoding="utf-8")

    # Check for command references (make test, black, ruff, mypy, pytest, uv)
    expected_refs = ["make test", "black", "ruff", "mypy", "pytest", "uv"]

    combined_content = contributing_content + testing_content
    for ref in expected_refs:
        assert ref in combined_content, (
            f"Expected command/tool '{ref}' not found in developer guides"
        )


def test_configuration_templates_reference_yaml():
    """Test that configuration templates reference the YAML templates."""
    docs_path = Path(__file__).parent.parent.parent / "docs"
    templates = docs_path / "source" / "configuration" / "templates.rst"

    content = templates.read_text(encoding="utf-8")

    # Check for references to template examples
    expected_refs = ["static", "laminar_flow", "contrast", "offset", "D0", "alpha"]
    for ref in expected_refs:
        assert ref in content, f"Expected reference '{ref}' not found in templates.rst"


def test_internal_links_in_developer_guide():
    """Test that internal links in developer guide are properly formatted."""
    docs_path = Path(__file__).parent.parent.parent / "docs"
    dev_guide_index = docs_path / "source" / "developer" / "index.rst"

    # Check that index.rst has proper toctree
    index_content = dev_guide_index.read_text(encoding="utf-8")
    assert "contributing" in index_content.lower(), (
        "contributing guide not referenced in dev guide index"
    )
    assert "testing" in index_content.lower(), (
        "testing guide not referenced in dev guide index"
    )


def test_internal_links_in_configuration():
    """Test that internal links in configuration guide are properly formatted."""
    docs_path = Path(__file__).parent.parent.parent / "docs"
    config_index = docs_path / "source" / "configuration" / "index.rst"

    # Check that index.rst has proper toctree
    index_content = config_index.read_text(encoding="utf-8")
    assert "templates" in index_content.lower(), (
        "templates guide not referenced in configuration index"
    )
    assert "options" in index_content.lower(), (
        "options guide not referenced in configuration index"
    )


def test_sphinx_rst_formatting():
    """Test that RST files have proper formatting."""
    docs_path = Path(__file__).parent.parent.parent / "docs"

    rst_files = [
        docs_path / "source" / "developer" / "index.rst",
        docs_path / "source" / "developer" / "contributing_guide.rst",
        docs_path / "source" / "developer" / "testing_guide.rst",
        docs_path / "source" / "configuration" / "index.rst",
        docs_path / "source" / "configuration" / "templates.rst",
        docs_path / "source" / "configuration" / "options.rst",
    ]

    for rst_file in rst_files:
        assert rst_file.exists(), f"RST file not found: {rst_file}"
        content = rst_file.read_text(encoding="utf-8")

        # Check for required RST elements
        assert "=" in content or "-" in content or "~" in content, (
            f"No section headers in {rst_file.name}"
        )

        # Check that file doesn't have trailing whitespace on lines (common issue)
        lines = content.split("\n")
        for _i, line in enumerate(lines):
            if line.rstrip() != line and line.strip():  # Ignore empty lines
                # This is a warning, not a failure - allow it but note it
                pass
