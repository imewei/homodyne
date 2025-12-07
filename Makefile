# Homodyne Analysis Package - Development Makefile
# JAX-accelerated XPCS analysis with NLSQ and MCMC (CPU-only)

.PHONY: help clean clean-all clean-build clean-pyc clean-test clean-cache install dev test test-all test-parallel test-all-parallel test-unit test-integration test-performance lint format type-check quality docs build check benchmark env-info

# Variables
PYTHON := python
PYTEST := pytest
RUFF := ruff

# Platform detection
UNAME_S := $(shell uname -s 2>/dev/null || echo "Windows")
ifeq ($(UNAME_S),Linux)
    PLATFORM := linux
else ifeq ($(UNAME_S),Darwin)
    PLATFORM := macos
else
    PLATFORM := windows
endif

# Package manager detection (prioritize uv > conda/mamba > pip)
UV_AVAILABLE := $(shell command -v uv 2>/dev/null)
CONDA_PREFIX := $(shell echo $$CONDA_PREFIX)
MAMBA_AVAILABLE := $(shell command -v mamba 2>/dev/null)

ifdef UV_AVAILABLE
    PKG_MANAGER := uv
    PIP := uv pip
    INSTALL_CMD := uv pip install
else ifdef CONDA_PREFIX
    ifdef MAMBA_AVAILABLE
        PKG_MANAGER := mamba (using pip for JAX)
    else
        PKG_MANAGER := conda (using pip for JAX)
    endif
    PIP := pip
    INSTALL_CMD := pip install
else
    PKG_MANAGER := pip
    PIP := pip
    INSTALL_CMD := pip install
endif

# Default target
help:
	@echo "Homodyne Analysis Package - Development Commands"
	@echo "================================================"
	@echo
	@echo "Environment Detection:"
	@echo "  Platform: $(PLATFORM)"
	@echo "  Package manager: $(PKG_MANAGER)"
	@echo "  Python: $(shell $(PYTHON) --version 2>&1 || echo 'not found')"
	@echo
	@echo "Installation & Setup:"
	@echo "  install         Install package in production mode (CPU-only)"
	@echo "  dev             Install package with development dependencies"
	@echo "  env-info        Show detailed environment info"
	@echo "  deps-check      Check all dependencies status"
	@echo
	@echo "Testing:"
	@echo "  test            Run core unit tests"
	@echo "  test-all        Run all tests with coverage"
	@echo "  test-parallel   Run tests in parallel (requires pytest-xdist)"
	@echo "  test-unit       Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-performance Run performance benchmarks"
	@echo "  test-nlsq       Test NLSQ optimization"
	@echo "  test-mcmc       Test MCMC optimization"
	@echo
	@echo "Code Quality:"
	@echo "  lint            Run code linting (ruff)"
	@echo "  format          Auto-format code (ruff)"
	@echo "  type-check      Run type checking (mypy)"
	@echo "  quality         Run all quality checks"
	@echo "  pre-commit      Run pre-commit hooks"
	@echo
	@echo "Performance:"
	@echo "  benchmark       Run performance benchmarks"
	@echo "  profile-nlsq    Profile NLSQ optimization"
	@echo "  profile-mcmc    Profile MCMC optimization"
	@echo
	@echo "Cleanup:"
	@echo "  clean           Clean build artifacts and cache"
	@echo "  clean-all       Clean everything including test data"
	@echo
	@echo "Documentation:"
	@echo "  docs            Build documentation"
	@echo "  docs-serve      Serve documentation locally"
	@echo
	@echo "Packaging:"
	@echo "  build           Build distribution packages"
	@echo "  check           Check package metadata"
	@echo "  release         Prepare for release"

# Installation targets
install:
	$(PIP) install -e .
	@echo "Package installed (CPU-only)"

dev:
	$(PIP) install -e ".[dev,docs]"
	@echo "Development environment ready (CPU-only)"
	@echo "  Platform: $(PLATFORM)"
	@echo "  Python: $(shell $(PYTHON) --version 2>&1)"
	@echo "  JAX: $(shell $(PYTHON) -c 'import jax; print(jax.__version__)' 2>/dev/null || echo 'not installed')"

env-info:
	@echo "Homodyne Environment Information"
	@echo "================================="
	@echo
	@echo "Platform: $(UNAME_S) ($(PLATFORM))"
	@echo "Python: $(shell $(PYTHON) --version 2>&1 || echo 'not found')"
	@echo "Package manager: $(PKG_MANAGER)"
	@echo
	@echo "JAX: $(shell $(PYTHON) -c 'import jax; print(jax.__version__)' 2>/dev/null || echo 'not installed')"
	@echo "NLSQ: $(shell $(PYTHON) -c 'import nlsq; print(nlsq.__version__)' 2>/dev/null || echo 'not installed')"
	@echo "NumPyro: $(shell $(PYTHON) -c 'import numpyro; print(numpyro.__version__)' 2>/dev/null || echo 'not installed')"

deps-check:
	@echo "Checking Homodyne Dependencies..."
	@$(PYTHON) -c "from homodyne import get_package_info; info = get_package_info(); \
		deps = info['dependencies']; \
		print('Core: JAX=' + ('yes' if deps.get('jax') else 'no') + \
		      ', NumPy=' + ('yes' if deps.get('numpy') else 'no') + \
		      ', SciPy=' + ('yes' if deps.get('scipy') else 'no')); \
		print('Optimization: NLSQ=' + ('yes' if deps.get('nlsq') else 'no') + \
		      ', NumPyro=' + ('yes' if deps.get('numpyro') else 'no')); \
		"

# Testing targets
test:
	$(PYTEST) tests/unit -v --tb=short

test-all:
	$(PYTEST) tests/ -v --cov=homodyne --cov-report=html --cov-report=term

test-parallel:
	$(PYTEST) tests/unit -v --tb=short -n auto

test-all-parallel:
	$(PYTEST) tests/ -v --cov=homodyne --cov-report=html --cov-report=term -n auto

test-unit:
	$(PYTEST) tests/unit -v --tb=short

test-integration:
	$(PYTEST) tests/integration -v --tb=short

test-performance:
	$(PYTEST) tests/performance -v -m performance

test-nlsq:
	$(PYTEST) tests/unit/test_optimization_nlsq.py -v

test-mcmc:
	$(PYTEST) tests/unit/test_optimization_mcmc.py -v 2>/dev/null || echo "MCMC tests not found"

test-quick:
	$(PYTEST) tests/unit -v -x --tb=no -q

# Code quality targets
lint:
	$(RUFF) check homodyne/

format:
	$(RUFF) format homodyne/ tests/
	$(RUFF) check --fix homodyne/ tests/

type-check:
	mypy homodyne/ --show-error-codes

quality: format lint type-check
	@echo "All quality checks passed"

pre-commit:
	pre-commit run --all-files

install-hooks:
	pre-commit install

# Performance targets
benchmark:
	$(PYTEST) tests/performance/test_benchmarks.py -v --tb=short

profile-nlsq:
	$(PYTHON) -m cProfile -s cumulative scripts/nlsq/static_isotropic_nlsq.py 2>/dev/null || \
	$(PYTHON) -c "from homodyne.optimization import fit_nlsq_jax; print('NLSQ profiling ready')"

profile-mcmc:
	$(PYTHON) -m cProfile -s cumulative scripts/mcmc/mcmc_uncertainty.py 2>/dev/null || \
	$(PYTHON) -c "from homodyne.optimization import fit_mcmc_jax; print('MCMC profiling ready')"

# Cleanup targets
clean: clean-build clean-pyc clean-test
	rm -rf node_modules/

clean-all: clean clean-cache
	rm -rf data/test_*
	rm -rf scripts/output/

clean-build:
	rm -rf build/ dist/ *.egg-info/ .eggs/ homodyne.egg-info/

clean-pyc:
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type f -name '*~' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

clean-test:
	rm -rf .pytest_cache/ .coverage htmlcov/ .mypy_cache/ .ruff_cache/ .benchmarks/
	rm -f coverage.xml .coverage.* test-results.xml
	rm -rf .nlsq_cache/ .hypothesis/ homodyne_results/
	rm -f bandit*report.json
	rm -rf .homodyne_cache/ cmc_temp_*/ tmp/ .ultra-think/
	rm -f phi_angles_list.txt TASK_GROUP_*.md

clean-cache:
	find . -type d -name '.cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.nlsq_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.mypy_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.ruff_cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '.pytest_cache' -exec rm -rf {} + 2>/dev/null || true

# Documentation targets
docs:
	cd docs && $(MAKE) html

docs-serve:
	cd docs/_build/html && $(PYTHON) -m http.server 8000

docs-clean:
	cd docs && $(MAKE) clean

# Packaging targets
build: clean
	$(PYTHON) -m build

check:
	$(PYTHON) -m twine check dist/* 2>/dev/null || echo "Run 'make build' first"

release: test quality build check
	@echo "Package ready for release"

# Example/Demo targets
run-example:
	$(PYTHON) scripts/nlsq/static_isotropic_nlsq.py

# CI targets
ci: clean test lint

ci-full: clean install test-all quality check

# Development shortcuts
dev-install: dev install-hooks

quick: test-quick lint

# Watch for changes (requires entr)
watch:
	find homodyne tests -name "*.py" | entr -c make test-quick

# Project statistics
stats:
	@echo "Homodyne Project Statistics"
	@echo "==========================="
	@echo "Lines of code:"
	@find homodyne -name "*.py" -exec wc -l {} + | tail -n 1
	@echo "Python files: $(shell find homodyne -name '*.py' | wc -l)"
	@echo "Test files: $(shell find tests -name 'test_*.py' | wc -l)"

# Verify NLSQ integration
verify-nlsq:
	@$(PYTHON) -c "from homodyne.optimization.nlsq import NLSQ_AVAILABLE; print(f'NLSQ available: {NLSQ_AVAILABLE}')"
	@$(PYTHON) -c "from homodyne.optimization import fit_nlsq_jax; print('fit_nlsq_jax imported')"
	@$(PYTHON) -c "import nlsq; print(f'NLSQ version: {nlsq.__version__}')" 2>/dev/null || echo "NLSQ version check failed"

.DEFAULT_GOAL := help
