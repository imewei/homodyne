"""Unit tests for Sphinx documentation configuration.

Tests validate that docs/conf.py is properly configured for building
comprehensive documentation with all required extensions and settings.
"""

import sys
from pathlib import Path


def test_conf_py_loads_without_errors():
    """Test that conf.py can be loaded without import errors."""
    docs_path = Path(__file__).parent.parent.parent / "docs"
    conf_path = docs_path / "conf.py"

    assert conf_path.exists(), f"conf.py not found at {conf_path}"

    # Add docs directory to path temporarily
    sys.path.insert(0, str(docs_path))

    try:
        # Import conf.py as module
        import importlib.util

        spec = importlib.util.spec_from_file_location("conf", conf_path)
        assert spec is not None, "Failed to create module spec"

        conf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(conf)

        # Basic smoke test - check that key attributes exist
        assert hasattr(conf, "project")
        assert hasattr(conf, "extensions")
        assert hasattr(conf, "html_theme")

    finally:
        # Clean up sys.path
        sys.path.pop(0)


def test_required_extensions_enabled():
    """Test that all required Sphinx extensions are enabled."""
    docs_path = Path(__file__).parent.parent.parent / "docs"
    conf_path = docs_path / "conf.py"

    # Add docs directory to path temporarily
    sys.path.insert(0, str(docs_path))

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("conf", conf_path)
        conf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(conf)

        required_extensions = [
            "sphinx.ext.autodoc",
            "sphinx.ext.autosummary",
            "sphinx.ext.napoleon",
            "sphinx.ext.viewcode",
            "sphinx.ext.intersphinx",
            "sphinx.ext.mathjax",
            "myst_parser",
        ]

        for ext in required_extensions:
            assert (
                ext in conf.extensions
            ), f"Required extension '{ext}' not found in extensions list"

    finally:
        sys.path.pop(0)


def test_intersphinx_mappings_resolve():
    """Test that intersphinx mappings are properly configured."""
    docs_path = Path(__file__).parent.parent.parent / "docs"
    conf_path = docs_path / "conf.py"

    sys.path.insert(0, str(docs_path))

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("conf", conf_path)
        conf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(conf)

        # Check that intersphinx_mapping exists and has required projects
        assert hasattr(
            conf, "intersphinx_mapping"
        ), "intersphinx_mapping not configured"

        required_mappings = ["python", "jax", "numpy", "numpyro", "scipy"]

        for project in required_mappings:
            assert (
                project in conf.intersphinx_mapping
            ), f"Required intersphinx mapping '{project}' not found"

            # Check that mapping has a valid URL
            url = conf.intersphinx_mapping[project][0]
            assert url.startswith("http"), f"Invalid URL for {project}: {url}"

    finally:
        sys.path.pop(0)


def test_mathjax3_configuration():
    """Test that MathJax 3 is properly configured."""
    docs_path = Path(__file__).parent.parent.parent / "docs"
    conf_path = docs_path / "conf.py"

    sys.path.insert(0, str(docs_path))

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("conf", conf_path)
        conf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(conf)

        # Check MathJax 3 path
        assert hasattr(conf, "mathjax_path"), "mathjax_path not configured"
        assert "mathjax@3" in conf.mathjax_path, "MathJax 3 CDN not configured"

        # Check MathJax 3 configuration
        assert hasattr(conf, "mathjax3_config"), "mathjax3_config not configured"
        assert "tex" in conf.mathjax3_config, "MathJax tex config missing"

        # Check delimiters are configured
        tex_config = conf.mathjax3_config["tex"]
        assert "inlineMath" in tex_config, "Inline math delimiters not configured"
        assert "displayMath" in tex_config, "Display math delimiters not configured"

        # Verify standard delimiters are present
        assert ["$", "$"] in tex_config["inlineMath"], "Dollar sign delimiter missing"
        assert ["$$", "$$"] in tex_config[
            "displayMath"
        ], "Double dollar delimiter missing"

    finally:
        sys.path.pop(0)


def test_autodoc_mock_imports_allowed():
    """Test that autodoc_mock_imports only contains expected optional dependencies."""
    docs_path = Path(__file__).parent.parent.parent / "docs"
    conf_path = docs_path / "conf.py"

    # Optional dependencies that are allowed to be mocked
    allowed_mocks = {"arviz"}

    sys.path.insert(0, str(docs_path))

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("conf", conf_path)
        conf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(conf)

        # Check that autodoc_mock_imports exists
        assert hasattr(conf, "autodoc_mock_imports"), "autodoc_mock_imports not set"
        # Check that only allowed optional dependencies are mocked
        mock_set = set(conf.autodoc_mock_imports)
        unexpected = mock_set - allowed_mocks
        assert not unexpected, (
            f"Unexpected packages in autodoc_mock_imports: {unexpected}. "
            f"Only optional dependencies {allowed_mocks} should be mocked."
        )

    finally:
        sys.path.pop(0)


def test_sphinx_rtd_theme_configuration():
    """Test that sphinx_rtd_theme is properly configured."""
    docs_path = Path(__file__).parent.parent.parent / "docs"
    conf_path = docs_path / "conf.py"

    sys.path.insert(0, str(docs_path))

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("conf", conf_path)
        conf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(conf)

        # Check theme is set
        assert conf.html_theme == "sphinx_rtd_theme", "Theme should be sphinx_rtd_theme"

        # Check theme options
        assert hasattr(conf, "html_theme_options"), "html_theme_options not configured"
        theme_opts = conf.html_theme_options

        # Verify required theme options
        assert theme_opts.get("navigation_depth") == 3, "navigation_depth should be 3"
        assert (
            theme_opts.get("sticky_navigation") is True
        ), "sticky_navigation should be True"
        assert (
            theme_opts.get("collapse_navigation") is True
        ), "collapse_navigation should be True"
        assert (
            theme_opts.get("style_nav_header_background") == "#2980b9"
        ), "Argonne branding color should be #2980b9"
        assert (
            theme_opts.get("prev_next_buttons_location") == "bottom"
        ), "prev_next_buttons_location should be 'bottom'"

        # Check sourcelink is disabled
        assert hasattr(conf, "html_show_sourcelink"), "html_show_sourcelink not set"
        assert (
            conf.html_show_sourcelink is False
        ), "html_show_sourcelink should be False"

    finally:
        sys.path.pop(0)


def test_myst_parser_extensions():
    """Test that MyST parser extensions are properly configured."""
    docs_path = Path(__file__).parent.parent.parent / "docs"
    conf_path = docs_path / "conf.py"

    sys.path.insert(0, str(docs_path))

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("conf", conf_path)
        conf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(conf)

        # Check MyST extensions
        assert hasattr(
            conf, "myst_enable_extensions"
        ), "myst_enable_extensions not configured"

        required_myst_extensions = ["dollarmath", "amsmath", "deflist", "colon_fence"]

        for ext in required_myst_extensions:
            assert (
                ext in conf.myst_enable_extensions
            ), f"Required MyST extension '{ext}' not enabled"

        # Check heading anchors
        assert hasattr(conf, "myst_heading_anchors"), "myst_heading_anchors not set"
        assert conf.myst_heading_anchors == 3, "myst_heading_anchors should be 3"

    finally:
        sys.path.pop(0)


def test_project_metadata():
    """Test that project metadata is correctly configured."""
    docs_path = Path(__file__).parent.parent.parent / "docs"
    conf_path = docs_path / "conf.py"

    sys.path.insert(0, str(docs_path))

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("conf", conf_path)
        conf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(conf)

        # Check basic project info
        assert (
            conf.project == "Homodyne"
        ), f"Project should be 'Homodyne', got '{conf.project}'"
        assert (
            conf.version == "2.4.3"
        ), f"Version should be '2.4.3', got '{conf.version}'"
        assert (
            conf.release == "2.4.3"
        ), f"Release should be '2.4.3', got '{conf.release}'"

        # Check autodoc settings
        assert (
            conf.autodoc_typehints == "description"
        ), "autodoc_typehints should be 'description'"
        assert conf.autosummary_generate is True, "autosummary_generate should be True"

    finally:
        sys.path.pop(0)


def test_warning_suppression():
    """Test that expected warnings are suppressed."""
    docs_path = Path(__file__).parent.parent.parent / "docs"
    conf_path = docs_path / "conf.py"

    sys.path.insert(0, str(docs_path))

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("conf", conf_path)
        conf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(conf)

        # Check warning suppression
        assert hasattr(conf, "suppress_warnings"), "suppress_warnings not configured"

        expected_suppressions = ["misc.highlighting_failure", "myst.xref_missing"]

        for warning in expected_suppressions:
            assert (
                warning in conf.suppress_warnings
            ), f"Warning '{warning}' should be suppressed"

    finally:
        sys.path.pop(0)
