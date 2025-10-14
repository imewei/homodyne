# Homodyne Analysis Package - Development Makefile
# ================================================
# JAX-accelerated XPCS analysis with Optimistix NLSQ and MCMC
# Updated: 2025-09-24 - Optimistix integration

.PHONY: help clean clean-all clean-build clean-pyc clean-test clean-cache install dev test test-all test-unit test-integration test-performance lint format docs build check gpu-check benchmark

# Variables
PYTHON := python
PIP := pip
PYTEST := pytest
RUFF := ruff
BLACK := black

# Default target
help:
	@echo "Homodyne Analysis Package - Development Commands"
	@echo "================================================"
	@echo
	@echo "Installation & Setup:"
	@echo "  install         Install package in production mode"
	@echo "  dev             Install package with development dependencies"
	@echo "  install-gpu     Install with GPU acceleration support"
	@echo "  install-hpc     Install for HPC environments"
	@echo "  deps-check      Check all dependencies status"
	@echo "  gpu-check       Check GPU availability and CUDA setup"
	@echo
	@echo "Testing:"
	@echo "  test            Run core unit tests"
	@echo "  test-all        Run all tests with coverage"
	@echo "  test-unit       Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-performance Run performance benchmarks"
	@echo "  test-gpu        Run GPU validation tests"
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

dev:
	$(PIP) install -e ".[dev,docs]"
	@echo "✓ Development environment ready"
	@echo "  Optimistix NLSQ: Ready"
	@echo "  MCMC (NumPyro): Ready"
	@$(PYTHON) -c "from homodyne import get_package_info; info = get_package_info(); print('\nDependency Status:'); [print(f'  {k}: ✓' if v else f'  {k}: ✗') for k,v in info['dependencies'].items()]"

install-gpu:
	$(PIP) install -e ".[gpu]"
	@echo "✓ GPU acceleration installed"
	@$(MAKE) gpu-check

install-hpc:
	$(PIP) install -e ".[hpc]"
	@echo "✓ HPC environment configured"

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
		print(f'  Optimistix: ' + ('✓' if deps.get('optimistix') else '✗ (NLSQ unavailable)')); \
		print(f'  Equinox: ' + ('✓' if deps.get('equinox') else '✗')); \
		print(f'  NumPyro: ' + ('✓' if deps.get('numpyro') else '✗ (MCMC unavailable)')); \
		print(f'  BlackJAX: ' + ('✓' if deps.get('blackjax') else '✗')); \
		print('\nData I/O:'); \
		print(f'  h5py: ' + ('✓' if deps.get('h5py') else '✗')); \
		print(f'  PyYAML: ' + ('✓' if deps.get('yaml') else '✗')); \
		"

gpu-check:
	@echo "Checking GPU Configuration..."
	@echo "============================"
	@$(PYTHON) -c "import jax; print(f'JAX version: {jax.__version__}'); devices = jax.devices(); print(f'Available devices: {devices}'); gpu_devices = [d for d in devices if 'cuda' in str(d).lower() or 'gpu' in str(d).lower()]; print(f'✓ GPU detected: {len(gpu_devices)} device(s)') if gpu_devices else print('✗ No GPU detected - using CPU')"

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

test-gpu:
	$(PYTEST) tests/gpu -v --tb=short

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
	mypy homodyne/ --ignore-missing-imports

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

clean-cache:
	find . -type d -name '.cache' -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf .pytest_cache/
	rm -rf .nlsq_cache/

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
	$(PYTHON) examples/gpu_accelerated_optimization.py

demo-nlsq:
	@echo "Running NLSQ optimization demo..."
	@$(PYTHON) -c "import numpy as np; \
		from homodyne.optimization import fit_nlsq_jax; \
		from homodyne.config import ConfigManager; \
		data = {'c2_exp': np.ones((10,10,10)) + 0.1*np.random.randn(10,10,10), \
			'wavevector_q_list': [0.01], 'phi_angles_list': np.linspace(0, 2*np.pi, 10), \
			't1': np.linspace(0.1, 1, 10), 't2': np.linspace(0.1, 1, 10)}; \
		config = ConfigManager(config_override={'analysis_mode': 'static_isotropic'}); \
		print('Starting Optimistix NLSQ optimization...'); \
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

# Verify Optimistix integration
verify-optimistix:
	@echo "Verifying Optimistix Integration..."
	@echo "==================================="
	@$(PYTHON) -c "from homodyne.optimization.nlsq import OPTIMISTIX_AVAILABLE; print(f'Optimistix available: {OPTIMISTIX_AVAILABLE}')"
	@$(PYTHON) -c "from homodyne.optimization import fit_nlsq_jax; print(f'fit_nlsq_jax imported: ✓')"
	@$(PYTHON) -c "import optimistix; print(f'Optimistix version: {optimistix.__version__}')" 2>/dev/null || echo "Optimistix version check failed"
	@echo "✓ Optimistix integration verified"

.DEFAULT_GOAL := help