# Configuration file for the Sphinx documentation builder.
# Homodyne Documentation

import os
import sys
from pathlib import Path

# Add the parent directory (containing homodyne package) to the Python path
sys.path.insert(0, str(Path("../..").resolve()))

# -- Project information -----------------------------------------------------
project = "Homodyne"
copyright = "2025, Wei Chen, Hongrui He - Argonne National Laboratory"
author = "Wei Chen, Hongrui He"

# Get version dynamically from package; fall back to hardcoded value
try:
    from homodyne import __version__ as _pkg_version

    # Strip any .post or +dirty suffixes to get the clean base version
    version = _pkg_version.split(".post")[0].split("+")[0]
except ImportError:
    version = "2.22.2"

release = version

# Project metadata
project_description = (
    "JAX-first high-performance XPCS analysis for nonequilibrium soft matter systems"
)
github_url = "https://github.com/imewei/homodyne"
doi_2024 = "10.1073/pnas.2401162121"
doi_2025 = "10.1073/pnas.2514216122"
institution = "Argonne National Laboratory"

# -- General configuration ---------------------------------------------------
source_encoding = "utf-8"

extensions = [
    # Sphinx built-ins
    "sphinx.ext.autodoc",  # Auto-generate API docs from docstrings
    "sphinx.ext.autosummary",  # Summary tables for modules/classes
    "sphinx.ext.napoleon",  # NumPy/Google-style docstring support
    "sphinx.ext.viewcode",  # Source code links in API pages
    "sphinx.ext.intersphinx",  # Cross-project hyperlinks
    "sphinx.ext.mathjax",  # LaTeX math rendering via MathJax
    "sphinx.ext.doctest",  # Doctest validation in documentation
    "sphinx.ext.todo",  # TODO items support
    "sphinx.ext.coverage",  # API coverage reporting
    # Third-party
    "myst_parser",  # Markdown (.md) file support
    "sphinx_autodoc_typehints",  # Type hint rendering in API docs
    "sphinx_copybutton",  # Copy-to-clipboard button on code blocks
]

# Template paths (relative to this conf.py directory)
templates_path = ["_templates"]

# Patterns to exclude from source discovery
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "_autosummary",
]

# Default syntax-highlighting language for code blocks
highlight_language = "python3"

# -- Options for autodoc extension -------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
    "ignore-module-all": True,
    "no-index": True,
}

# Show __init__ signature separately from the class docstring
autodoc_class_signature = "separated"

# Render type hints in the parameter description section
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_preserve_defaults = True

# Mock optional dependencies that may be absent in CI environments
autodoc_mock_imports: list[str] = ["arviz"]

# -- Options for autosummary extension ---------------------------------------
autosummary_generate = True
autosummary_imported_members = False

# -- Options for sphinx_autodoc_typehints ------------------------------------
always_use_bars_union = True
typehints_fully_qualified = False
simplify_optional_unions = True

# -- Options for sphinx-copybutton -------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- Options for Napoleon extension ------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# -- Options for intersphinx extension ---------------------------------------
_online_intersphinx = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "numpyro": ("https://num.pyro.ai/en/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# Default to offline builds in CI/air-gapped environments.
# Set SPHINX_OFFLINE=0 to enable live cross-project links.
intersphinx_mapping = _online_intersphinx.copy()
if os.environ.get("SPHINX_OFFLINE", "1") == "0":
    intersphinx_disabled_domains: list[str] = []
else:
    intersphinx_disabled_domains = list(_online_intersphinx.keys())
intersphinx_timeout = 1

# -- Options for MathJax extension -------------------------------------------
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    }
}

# -- Options for MyST parser -------------------------------------------------
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "html_admonition",
    "html_image",
    "colon_fence",
    "smartquotes",
    "replacements",
]
myst_heading_anchors = 3

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"

html_theme_options = {
    "navigation_with_keys": True,
    # Source repository link (edit-on-GitHub button)
    "source_repository": github_url,
    "source_branch": "main",
    "source_directory": "docs/source/",
    # Footer icon: GitHub
    "footer_icons": [
        {
            "name": "GitHub",
            "url": github_url,
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16">'
                '<path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 '
                "6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37"
                "-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01"
                "-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87"
                ".51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2"
                "-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 "
                "1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12"
                ".51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54"
                " 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 "
                '0016 8c0-4.42-3.58-8-8-8z"></path>'
                "</svg>"
            ),
            "class": "",
        },
    ],
    # Brand colours (light / dark)
    "light_css_variables": {
        # Primary brand
        "--color-brand-primary": "#2962FF",
        "--color-brand-content": "#2962FF",
        # Admonition: tip / note
        "--color-admonition-background-tip": "#E8F5E9",
        "--color-admonition-title-tip": "#00897B",
        "--color-admonition-background-note": "#E3F2FD",
        "--color-admonition-title-note": "#1565C0",
        # Typography
        "--font-stack": (
            "Inter, -apple-system, BlinkMacSystemFont, "
            "Segoe UI, Helvetica Neue, Arial, sans-serif"
        ),
        "--font-stack--monospace": (
            "JetBrains Mono, SFMono-Regular, Menlo, "
            "Consolas, Liberation Mono, monospace"
        ),
    },
    "dark_css_variables": {
        # Primary brand (accessible on dark backgrounds)
        "--color-brand-primary": "#82B1FF",
        "--color-brand-content": "#82B1FF",
        # Admonition: tip / note
        "--color-admonition-background-tip": "#1B3A36",
        "--color-admonition-title-tip": "#4DB6AC",
        "--color-admonition-background-note": "#0D2A45",
        "--color-admonition-title-note": "#64B5F6",
        # Typography
        "--font-stack": (
            "Inter, -apple-system, BlinkMacSystemFont, "
            "Segoe UI, Helvetica Neue, Arial, sans-serif"
        ),
        "--font-stack--monospace": (
            "JetBrains Mono, SFMono-Regular, Menlo, "
            "Consolas, Liberation Mono, monospace"
        ),
    },
}

# Custom static files (copied after Sphinx builtins)
html_static_path = ["_static"]

# Extra CSS files loaded after the theme
html_css_files = [
    "custom.css",
]

# Sidebar: no extra items beyond Furo defaults
# html_logo = "_static/logo.png"
# html_favicon = "_static/favicon.ico"

html_show_sphinx = True
html_show_copyright = True
html_show_sourcelink = False

htmlhelp_basename = "homodynedoc"

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": r"""
\usepackage{amsmath}
\usepackage{amssymb}
""",
}

latex_documents = [
    (
        "index",
        "homodyne.tex",
        "Homodyne Documentation",
        "Wei Chen, Hongrui He",
        "manual",
    ),
]

# -- Options for manual page output ------------------------------------------
man_pages = [("index", "homodyne", "Homodyne Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    (
        "index",
        "homodyne",
        "Homodyne Documentation",
        author,
        "homodyne",
        "JAX-first high-performance XPCS analysis for nonequilibrium soft matter systems",
        "Miscellaneous",
    ),
]

# -- Options for Epub output -------------------------------------------------
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright
epub_exclude_files = ["search.html"]

# -- Suppress known harmless warnings ----------------------------------------
suppress_warnings = [
    "misc.highlighting_failure",
    "myst.xref_missing",
    "autosummary.import_cycle",
    "toc.not_included",
]

# -- Options for linkcheck ---------------------------------------------------
linkcheck_ignore = [
    r"https://doi\.org/.*",  # DOI redirects block bots (403)
    r"https://github\.com/imewei/homodyne.*",  # Private repository (404)
]

# -- Todo extension ----------------------------------------------------------
todo_include_todos = True
