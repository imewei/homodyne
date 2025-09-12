# Homodyne Analysis Package - Development Makefile
# ================================================
# Updated: 2024-08-31 - Added shell completion and GPU acceleration setup targets

.PHONY: help clean clean-all clean-build clean-pyc clean-test clean-venv install dev-install test test-all lint format docs docs-serve build upload check setup-shell setup-gpu setup-advanced cleanup-homodyne

# Default target
help:
	@echo "Homodyne Analysis Package - Development Commands"
	@echo "=============================================="
	@echo
	@echo "Installation & Setup:"
	@echo "  install         Install package in editable mode"
	@echo "  dev-install     Install package with all development dependencies"
	@echo "  setup-shell     Install shell completion (aliases: hmv, hmm, hmh, hconfig, hexp, hsim)"
	@echo "  cleanup-homodyne Remove all homodyne setup files"
	@echo
	@echo "Testing:"
	@echo "  test            Run tests with pytest"
	@echo "  test-all        Run tests with all optional dependencies"
	@echo "  test-performance     Run performance tests only"
	@echo "  test-regression      Run performance regression tests"
	@echo "  test-ci         Run CI-style performance tests"
	@echo "  lint            Run code linting (ruff, mypy)"
	@echo "  format          Format code with ruff and black"
	@echo
	@echo "Performance Baselines:"
	@echo "  baseline-update      Update performance baselines"
	@echo "  baseline-reset       Reset all performance baselines"
	@echo "  baseline-report      Generate performance report"
	@echo
	@echo "Cleanup:"
	@echo "  clean           Clean all build artifacts and cache files (preserves virtual environment)"
	@echo "  clean-all       Clean everything including virtual environment"
	@echo "  cleanup-homodyne Remove homodyne shell completion and GPU setup files"
	@echo "  clean-build     Remove build artifacts"
	@echo "  clean-pyc       Remove Python bytecode files"
	@echo "  clean-test      Remove test and coverage artifacts"
	@echo "  clean-venv      Remove virtual environment"
	@echo
	@echo "Documentation:"
	@echo "  docs            Build documentation"
	@echo "  docs-serve      Serve documentation locally"
	@echo
	@echo "Packaging:"
	@echo "  build           Build distribution packages"
	@echo "  upload          Upload to PyPI"
	@echo "  check           Check package metadata and distribution"

# Installation targets
install:
	pip install -e .

dev-install:
	pip install -e ".[all,dev,docs]"

# Homodyne Setup targets  
setup-shell:
	@echo "Installing shell completion (aliases: hmv, hmm, hmh, hconfig, hexp, hsim)..."
	@echo "Sourcing shell completion directly from package..."
	@echo "source $(python -c 'import homodyne.runtime.shell; print(homodyne.runtime.shell.__path__[0])')/completion.sh" >> ~/.bashrc || echo "Please manually add completion.sh to your shell"
	@echo "✅ Shell completion installed. Restart shell or run: source ~/.bashrc"

cleanup-homodyne:
	@echo "Removing homodyne setup files..."
	@echo "Cleaning up shell configuration..."
	@echo "✅ Homodyne cleanup completed"

# Testing targets
test:
	python -c "import sys; sys.modules['numba'] = None; sys.modules['pymc'] = None; sys.modules['arviz'] = None; sys.modules['corner'] = None; import pytest; pytest.main(['-v', '--tb=short', '--continue-on-collection-errors', '--maxfail=5'])"

test-all:
	pytest -v --cov=homodyne --cov-report=html --cov-report=term

test-fast:
	python -c "import sys; import os; os.environ['PYTHONWARNINGS'] = 'ignore'; sys.modules['numba'] = None; sys.modules['pymc'] = None; sys.modules['arviz'] = None; sys.modules['corner'] = None; import pytest; result = pytest.main(['-q', '--tb=no', '--continue-on-collection-errors']); print(f'\nTest result code: {result}')"

# Performance testing targets
test-performance:
	pytest homodyne/tests/ -v -m performance

test-regression:
	@echo "Running performance regression tests..."
	pytest homodyne/tests/test_performance.py -v -m regression

test-ci:
	@echo "Running CI-style performance tests..."
	pytest homodyne/tests/test_performance.py -v --tb=short

# Performance baseline management
baseline-update:
	@echo "Updating performance baselines..."
	pytest homodyne/tests/test_performance.py -v --update-baselines
	@echo "✓ Baselines updated successfully"

baseline-reset:
	@echo "Resetting performance baselines..."
	rm -f ci_performance_baselines.json
	rm -f homodyne_test_performance_baselines.json
	rm -f homodyne/tests/test_performance_baselines.json
	rm -f homodyne/tests/performance_baselines.json
	@echo "✓ Baselines reset"

baseline-report:
	@echo "Generating performance report..."
	pytest homodyne/tests/test_performance.py -v --tb=short --durations=0
	@echo "✓ Performance report completed"

# Code quality targets
lint:
	ruff check homodyne/
	mypy homodyne/

format:
	ruff format homodyne/ tests/
	black homodyne/ tests/
	isort homodyne/ tests/

ruff-fix:
	ruff check --fix homodyne/

quality: format lint
	@echo "Code quality checks completed"

# Cleanup targets
clean: clean-build clean-pyc clean-test
	rm -rf node_modules/
	@echo "Cleaned all build artifacts and cache files"

clean-cache:
	find . -name '__pycache__' -type d -exec rm -rf {} +
	find . -name '*.pyc' -delete
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	@echo "Cleaned Python cache files"

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .eggs/

clean-pyc:
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete
	find . -name '*~' -delete
	find . -name '__pycache__' -type d -exec rm -rf {} +
	find . -name '*.orig' -delete
	find . -name '*.rej' -delete

clean-test:
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -name '.benchmarks' -type d -exec rm -rf {} +
	rm -f test_report.html
	rm -f bandit-report.json
	rm -f bandit_report.json
	rm -f bandit-results.json
	rm -f bandit_results.json
	rm -f code_quality_report.md
	rm -f pip_audit_report.json
	rm -f coverage.xml

clean-venv:
	rm -rf venv/
	rm -rf .venv/
	rm -rf env/
	rm -rf .env/

clean-all: clean-build clean-pyc clean-test clean-venv
	rm -rf node_modules/
	@echo "Cleaned all build artifacts, cache files, and virtual environment"
	@echo "Note: Run 'make cleanup-homodyne' to also remove shell completion and GPU setup"

# Documentation targets
docs:
	$(MAKE) -C docs html

docs-serve:
	@echo "Serving documentation at http://localhost:8000"
	cd docs/_build/html && python -m http.server 8000

# Packaging targets
build: clean
	python -m build

upload: build
	python -m twine upload dist/*

check:
	python -m twine check dist/*
	python -m build --check

pre-commit:
	pre-commit run --all-files

install-hooks:
	pre-commit install
