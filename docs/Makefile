# Makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Custom targets for homodyne documentation

# Clean build artifacts
clean:
	rm -rf $(BUILDDIR)/*
	@echo "Build artifacts cleaned."

# Build HTML documentation
html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

# Serve documentation locally
serve: html
	@echo "Serving documentation at http://localhost:8000"
	@echo "Press Ctrl+C to stop."
	python3 -m http.server -d $(BUILDDIR)/html 8000

# Check for broken links
linkcheck:
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS) $(O)
	@echo
	@echo "Link check complete. Results are in $(BUILDDIR)/linkcheck."

# Run doctests
doctest:
	@$(SPHINXBUILD) -b doctest "$(SOURCEDIR)" "$(BUILDDIR)/doctest" $(SPHINXOPTS) $(O)
	@echo
	@echo "Doctest complete. Results are in $(BUILDDIR)/doctest."

# Build with strict mode (treat warnings as errors)
strict:
	@$(SPHINXBUILD) -b html -W --keep-going "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)
	@echo
	@echo "Strict build finished. The HTML pages are in $(BUILDDIR)/html."

# Watch for changes and rebuild automatically (requires sphinx-autobuild)
watch:
	@command -v sphinx-autobuild >/dev/null 2>&1 || { echo "sphinx-autobuild not found. Install with: pip install sphinx-autobuild"; exit 1; }
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
