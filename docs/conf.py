# Configuration file for the Sphinx documentation builder.
# Homodyne v2.0+ Documentation

import os
import sys
from pathlib import Path

# Add the parent directory (containing homodyne package) to Python path
sys.path.insert(0, str(Path("..").resolve()))

# -- Project information -----------------------------------------------------
project = "Homodyne"
copyright = "2025, Wei Chen, Hongrui He - Argonne National Laboratory"
author = "Wei Chen, Hongrui He"
release = "2.4.1"
version = "2.4.1"

# Project metadata
project_description = (
    "JAX-first high-performance XPCS analysis for nonequilibrium soft matter systems"
)
github_url = "https://github.com/imewei/homodyne"
doi = "10.1073/pnas.2401162121"
institution = "Argonne National Laboratory"

# -- General configuration ---------------------------------------------------
source_encoding = "utf-8"

extensions = [
    "sphinx.ext.autodoc",  # Auto-generate API docs
    "sphinx.ext.autosummary",  # Summary tables
    "sphinx.ext.napoleon",  # NumPy/Google docstrings
    "sphinx.ext.viewcode",  # Source code links
    "sphinx.ext.intersphinx",  # Cross-project links
    "sphinx.ext.mathjax",  # Math rendering
    "sphinx.ext.doctest",  # Doctest support
    "sphinx.ext.todo",  # Todo items
    "sphinx.ext.coverage",  # Coverage tracking
    "myst_parser",  # Markdown support
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The default language to highlight source code in.
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

# Include __init__ methods in class documentation
autodoc_class_signature = "separated"

# Type hints configuration
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_preserve_defaults = True

# Mock imports for dependencies that might not be available during build
autodoc_mock_imports = []

# -- Options for autosummary extension ---------------------------------------
autosummary_generate = True
autosummary_imported_members = False

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
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "numpyro": ("https://num.pyro.ai/en/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# Default to offline builds in CI/air-gapped environments; opt in with SPHINX_OFFLINE=0
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
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 3,
    "includehidden": True,
    "titles_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    "style_nav_header_background": "#2980b9",  # Argonne branding
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Custom CSS files
html_css_files = [
    "custom.css",
]

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = "_static/logo.png"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = "_static/favicon.ico"

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# If true, links to source code are shown. We disable this for cleaner docs.
html_show_sourcelink = False

# Output file base name for HTML help builder.
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

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto, manual, or own class]).
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

# -- Extension configuration -------------------------------------------------
# Suppress specific warnings to reduce noise
suppress_warnings = [
    "misc.highlighting_failure",
    "myst.xref_missing",
    "autosummary.import_cycle",
]

# -- Todo extension configuration --------------------------------------------
todo_include_todos = True
