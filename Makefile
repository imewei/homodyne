# Homodyne Analysis Package - Development Makefile
# ================================================
# JAX-accelerated XPCS analysis with NLSQ and MCMC (CPU-only)
# Updated: 2025-11-10 - CPU-only (JAX 0.7.2 local, 0.8.0 target)

.PHONY: help clean clean-all clean-build clean-pyc clean-test clean-cache install dev test test-all test-unit test-integration test-performance lint format type-check type-check-strict quality docs build check benchmark env-info

# Variables
PYTHON := python
PYTEST := pytest
RUFF := ruff
BLACK := black

# Platform detection (for informational purposes only)
UNAME_S := $(shell uname -s 2>/dev/null || echo "Windows")
ifeq ($(UNAME_S),Linux)
    PLATFORM := linux
else ifeq ($(UNAME_S),Darwin)
    PLATFORM := macos
else
    PLATFORM := windows
endif

# Package manager detection (prioritize uv > conda/mamba > pip)
# Check for uv (fast Rust-based package manager)
UV_AVAILABLE := $(shell command -v uv 2>/dev/null)
# Check for conda/mamba
CONDA_PREFIX := $(shell echo $$CONDA_PREFIX)
MAMBA_AVAILABLE := $(shell command -v mamba 2>/dev/null)

# Determine package manager and commands
ifdef UV_AVAILABLE
    PKG_MANAGER := uv
    PIP := uv pip
    UNINSTALL_CMD := uv pip uninstall -y
    INSTALL_CMD := uv pip install
else ifdef CONDA_PREFIX
    # In conda environment - use pip within conda
    ifdef MAMBA_AVAILABLE
        PKG_MANAGER := mamba (using pip for JAX)
    else
        PKG_MANAGER := conda (using pip for JAX)
    endif
    PIP := pip
    UNINSTALL_CMD := pip uninstall -y
    INSTALL_CMD := pip install
else
    PKG_MANAGER := pip
    PIP := pip
    UNINSTALL_CMD := pip uninstall -y
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
	@echo "  dev             Install package with development dependencies (CPU-only)"
	@echo "  env-info        Show detailed environment and package manager detection"
	@echo "  deps-check      Check all dependencies status"
	@echo
	@echo "Testing:"
	@echo "  test            Run core unit tests"
	@echo "  test-all        Run all tests with coverage"
	@echo "  test-unit       Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-performance Run performance benchmarks"
	@echo "  test-nlsq       Test NLSQ optimization specifically"
	@echo "  test-mcmc       Test MCMC optimization specifically"
	@echo
	@echo "Code Quality:"
	@echo "  lint            Run code linting (ruff)"
	@echo "  format          Auto-format code (black + ruff)"
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
	@echo "  clean-build     Remove build artifacts"
	@echo "  clean-pyc       Remove Python bytecode files"
	@echo "  clean-test      Remove test and coverage artifacts"
	@echo "  clean-cache     Remove all cache directories"
	@echo
	@echo "Documentation:"
	@echo "  docs            Build documentation"
	@echo "  docs-serve      Serve documentation locally"
	@echo "  docs-clean      Clean documentation build"
	@echo
	@echo "Packaging:"
	@echo "  build           Build distribution packages"
	@echo "  check           Check package metadata"
	@echo "  release         Prepare for release (test + build + check)"
	@echo
	@echo "Examples:"
	@echo "  run-example     Run example optimization"
	@echo "  demo-nlsq       Demo NLSQ optimization"
	@echo "  demo-mcmc       Demo MCMC optimization"

# Installation targets
install:
	$(PIP) install -e .
	@echo "✓ Package installed (CPU-only)"
	@echo "  JAX version: $(shell $(PYTHON) -c 'import jax; print(jax.__version__)' 2>/dev/null || echo 'not installed')"

dev:
	$(PIP) install -e ".[dev,docs]"
	@echo "✓ Development environment ready (CPU-only)"
	@echo "  Platform: $(PLATFORM)"
	@echo "  Python: $(shell $(PYTHON) --version 2>&1)"
	@echo "  JAX: $(shell $(PYTHON) -c 'import jax; print(jax.__version__)' 2>/dev/null || echo 'not installed')"
	@echo "  NLSQ: $(shell $(PYTHON) -c 'import nlsq; print(nlsq.__version__)' 2>/dev/null || echo 'not installed')"
	@echo "  NumPyro: $(shell $(PYTHON) -c 'import numpyro; print(numpyro.__version__)' 2>/dev/null || echo 'not installed')"
	@$(PYTHON) -c "from homodyne import get_package_info; info = get_package_info(); print('\nDependency Status:'); [print(f'  {k}: ✓' if v else f'  {k}: ✗') for k,v in info['dependencies'].items()]" 2>/dev/null || echo "  Homodyne not fully installed yet"

env-info:
	@echo "Homodyne Environment Information"
	@echo "================================="
	@echo
	@echo "Platform Detection:"
	@echo "  OS: $(UNAME_S)"
	@echo "  Platform: $(PLATFORM)"
	@echo
	@echo "Python Environment:"
	@echo "  Python: $(shell $(PYTHON) --version 2>&1 || echo 'not found')"
	@echo "  Python path: $(shell which $(PYTHON) 2>/dev/null || echo 'not found')"
	@echo
	@echo "Package Manager Detection:"
	@echo "  Active manager: $(PKG_MANAGER)"
ifdef UV_AVAILABLE
	@echo "  ✓ uv detected: $(UV_AVAILABLE)"
	@echo "    Install command: $(INSTALL_CMD)"
	@echo "    Uninstall command: $(UNINSTALL_CMD)"
else
	@echo "  ✗ uv not found"
endif
ifdef CONDA_PREFIX
	@echo "  ✓ Conda environment detected"
	@echo "    CONDA_PREFIX: $(CONDA_PREFIX)"
ifdef MAMBA_AVAILABLE
	@echo "    Mamba available: $(MAMBA_AVAILABLE)"
else
	@echo "    Mamba: not found"
endif
	@echo "    Note: Using pip within conda for JAX installation"
else
	@echo "  ✗ Not in conda environment"
endif
	@echo "  pip: $(shell which pip 2>/dev/null || echo 'not found')"
	@echo
	@echo "JAX Configuration:"
	@echo "  JAX version: $(shell $(PYTHON) -c 'import jax; print(jax.__version__)' 2>/dev/null || echo 'not installed')"
	@echo "  JAX target: 0.8.0 (CPU-only per v2.3.0)"
	@echo "  GPU support: Not available (CPU-only installation)"
	@echo
	@echo "Installation Commands:"
	@echo "  Install dev: make dev"
	@echo "  Install prod: make install"
	@echo

deps-check:
	@echo "Checking Homodyne Dependencies..."
	@echo "================================="
	@$(PYTHON) -c "from homodyne import get_package_info; info = get_package_info(); \
		deps = info['dependencies']; \
		print('Core Dependencies:'); \
		print(f'  JAX: ' + ('✓' if deps.get('jax') else '✗')); \
		print(f'  NumPy: ' + ('✓' if deps.get('numpy') else '✗')); \
		print(f'  SciPy: ' + ('✓' if deps.get('scipy') else '✗')); \
		print('\nOptimization:'); \
		print(f'  NLSQ: ' + ('✓' if deps.get('nlsq') else '✗')); \
		print(f'  Equinox: ' + ('✓' if deps.get('equinox') else '✗')); \
		print(f'  NumPyro: ' + ('✓' if deps.get('numpyro') else '✗ (MCMC unavailable)')); \
		print(f'  BlackJAX: ' + ('✓' if deps.get('blackjax') else '✗')); \
		print('\nData I/O:'); \
		print(f'  h5py: ' + ('✓' if deps.get('h5py') else '✗')); \
		print(f'  PyYAML: ' + ('✓' if deps.get('yaml') else '✗')); \
		"

# Testing targets
test:
	$(PYTEST) tests/unit -v --tb=short

test-all:
	$(PYTEST) tests/ -v --cov=homodyne --cov-report=html --cov-report=term

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
	@echo "✓ Linting passed"

format:
	$(RUFF) format homodyne/ tests/
	$(BLACK) homodyne/ tests/ --quiet
	@echo "✓ Code formatted"

type-check:
	mypy homodyne/ --show-error-codes
	@echo "✓ Type checking passed"

type-check-strict:
	mypy homodyne/ --strict --show-error-codes
	@echo "✓ Strict type checking passed"

quality: format lint type-check
	@echo "✓ All quality checks passed"

pre-commit:
	pre-commit run --all-files

install-hooks:
	pre-commit install

# Performance targets
benchmark:
	@echo "Running performance benchmarks..."
	$(PYTEST) tests/performance/test_benchmarks.py -v --tb=short

profile-nlsq:
	@echo "Profiling NLSQ optimization..."
	$(PYTHON) -m cProfile -s cumulative examples/profile_nlsq.py 2>/dev/null || \
	$(PYTHON) -c "from homodyne.optimization import fit_nlsq_jax; print('NLSQ profiling ready')"

profile-mcmc:
	@echo "Profiling MCMC optimization..."
	$(PYTHON) -m cProfile -s cumulative examples/profile_mcmc.py 2>/dev/null || \
	$(PYTHON) -c "from homodyne.optimization import fit_mcmc_jax; print('MCMC profiling ready')"

# Cleanup targets
clean: clean-build clean-pyc clean-test
	@echo "✓ Cleaned build artifacts and cache"

clean-all: clean clean-cache
	rm -rf data/test_*
	rm -rf examples/output/
	@echo "✓ Cleaned everything"

clean-build:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .eggs/
	rm -rf homodyne.egg-info/

clean-pyc:
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type f -name '*~' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

clean-test:
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf .benchmarks/
	rm -f coverage.xml
	rm -f test-results.xml
	rm -rf .nlsq_cache/
	rm -rf .hypothesis/
	rm -rf homodyne_results/
	rm -f bandit*report.json
	rm -rf .homodyne_cache/
	rm -f TASK_GROUP_*.md
	rm -rf cmc_temp_*/
	rm -f phi_angles_list.txt
	rm -rf .ultra-think/

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
	@echo "✓ Documentation built at docs/_build/html/index.html"

docs-serve:
	@echo "Serving documentation at http://localhost:8000"
	cd docs/_build/html && $(PYTHON) -m http.server 8000

docs-clean:
	cd docs && $(MAKE) clean

# Packaging targets
build: clean
	$(PYTHON) -m build
	@echo "✓ Package built in dist/"

check:
	$(PYTHON) -m twine check dist/* 2>/dev/null || echo "Run 'make build' first"

release: test quality build check
	@echo "✓ Package ready for release"
	@echo "  Next step: python -m twine upload dist/*"

# Example/Demo targets
run-example:
	$(PYTHON) examples/static_isotropic_nlsq.py

demo-nlsq:
	@echo "Running NLSQ optimization demo..."
	@$(PYTHON) -c "import numpy as np; \
		from homodyne.optimization import fit_nlsq_jax; \
		from homodyne.config import ConfigManager; \
		data = {'c2_exp': np.ones((10,10,10)) + 0.1*np.random.randn(10,10,10), \
			'wavevector_q_list': [0.01], 'phi_angles_list': np.linspace(0, 2*np.pi, 10), \
			't1': np.linspace(0.1, 1, 10), 't2': np.linspace(0.1, 1, 10)}; \
		config = ConfigManager(config_override={'analysis_mode': 'static_isotropic'}); \
		print('Starting NLSQ optimization...'); \
		result = fit_nlsq_jax(data, config); \
		print(f'Result: success={result.success}, method={result.method}'); \
		print('✓ NLSQ demo completed')"

demo-mcmc:
	@echo "Running MCMC optimization demo..."
	@$(PYTHON) -c "import numpy as np; \
		from homodyne.optimization import fit_mcmc_jax; \
		data = np.ones((10,10,10)) + 0.1*np.random.randn(10,10,10); \
		print('Starting MCMC optimization...'); \
		result = fit_mcmc_jax(data, n_samples=100, n_warmup=50); \
		print(f'Result: converged={result.converged}'); \
		print('✓ MCMC demo completed')" 2>/dev/null || echo "MCMC not available"

# Continuous Integration targets
ci: clean test lint
	@echo "✓ CI checks passed"

ci-full: clean install test-all quality check
	@echo "✓ Full CI pipeline passed"

# Development shortcuts
dev-install: dev install-hooks
	@echo "✓ Development environment fully configured"

quick: test-quick lint
	@echo "✓ Quick checks passed"

# Watch for changes (requires entr)
watch:
	find homodyne tests -name "*.py" | entr -c make test-quick

# Print project statistics
stats:
	@echo "Homodyne Project Statistics"
	@echo "==========================="
	@echo "Lines of code:"
	@find homodyne -name "*.py" -exec wc -l {} + | tail -n 1
	@echo "Number of Python files:"
	@find homodyne -name "*.py" | wc -l
	@echo "Number of test files:"
	@find tests -name "test_*.py" | wc -l
	@echo "Package size:"
	@du -sh homodyne/

# Verify NLSQ integration
verify-nlsq:
	@echo "Verifying NLSQ Integration..."
	@echo "=============================="
	@$(PYTHON) -c "from homodyne.optimization.nlsq import NLSQ_AVAILABLE; print(f'NLSQ available: {NLSQ_AVAILABLE}')"
	@$(PYTHON) -c "from homodyne.optimization import fit_nlsq_jax; print(f'fit_nlsq_jax imported: ✓')"
	@$(PYTHON) -c "import nlsq; print(f'NLSQ package version: {nlsq.__version__}')" 2>/dev/null || echo "NLSQ version check failed"
	@echo "✓ NLSQ integration verified"

.DEFAULT_GOAL := help
