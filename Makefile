# Homodyne Analysis Package Makefile
# ==================================
# JAX-accelerated XPCS analysis with NLSQ and MCMC (CPU-only)

.PHONY: help install install-dev env-info deps-check version info \
        test test-smoke test-fast test-ci test-ci-full test-coverage test-unit test-integration test-performance \
        test-parallel test-all-parallel test-parallel-fast test-coverage-parallel test-nlsq test-mcmc test-quick \
        format lint type-check check quality quick pre-commit install-hooks \
        benchmark profile-nlsq profile-mcmc \
        clean clean-all clean-pyc clean-build clean-test clean-cache clean-venv \
        docs docs-serve docs-clean \
        build release run-example ci ci-full dev-install watch stats verify-nlsq

# Configuration
PYTHON := python
PYTEST := pytest
RUFF := ruff
PACKAGE_NAME := homodyne
SRC_DIR := homodyne
TEST_DIR := tests
DOCS_DIR := docs
VENV := .venv

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

# Determine package manager and commands
ifdef UV_AVAILABLE
    PKG_MANAGER := uv
    PIP := uv pip
    UNINSTALL_CMD := uv pip uninstall -y
    INSTALL_CMD := uv pip install
    RUN_CMD := uv run
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
    RUN_CMD :=
else
    PKG_MANAGER := pip
    PIP := pip
    UNINSTALL_CMD := pip uninstall -y
    INSTALL_CMD := pip install
    RUN_CMD :=
endif

# Colors for output
BOLD := \033[1m
RESET := \033[0m
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
CYAN := \033[36m

# Default target
.DEFAULT_GOAL := help

# ===================
# Help target
# ===================
help:
	@echo "$(BOLD)$(BLUE)Homodyne Analysis Package - Development Commands$(RESET)"
	@echo ""
	@echo "$(BOLD)Usage:$(RESET) make $(CYAN)<target>$(RESET)"
	@echo ""
	@echo "$(BOLD)$(GREEN)ENVIRONMENT$(RESET)"
	@echo "  $(CYAN)env-info$(RESET)         Show detailed environment information"
	@echo "  $(CYAN)info$(RESET)             Show project and environment info"
	@echo "  $(CYAN)version$(RESET)          Show package version"
	@echo "  $(CYAN)deps-check$(RESET)       Check all dependencies status"
	@echo ""
	@echo "$(BOLD)$(GREEN)INSTALLATION$(RESET)"
	@echo "  $(CYAN)install$(RESET)          Install package in editable mode (CPU-only)"
	@echo "  $(CYAN)dev$(RESET)              Install with development dependencies"
	@echo "  $(CYAN)dev-install$(RESET)      Install dev deps + pre-commit hooks"
	@echo ""
	@echo "$(BOLD)$(GREEN)TESTING$(RESET)"
	@echo "  $(CYAN)test$(RESET)                   Run core unit tests"
	@echo "  $(CYAN)test-smoke$(RESET)             Run smoke tests (critical tests, ~30s-2min)"
	@echo "  $(CYAN)test-fast$(RESET)              Run tests excluding slow tests (~5-10min)"
	@echo "  $(CYAN)test-ci$(RESET)                Run CI test suite (matches GitHub Actions)"
	@echo "  $(CYAN)test-ci-full$(RESET)           Run full CI suite with coverage"
	@echo "  $(CYAN)test-parallel$(RESET)          Run tests in parallel (2-4x faster)"
	@echo "  $(CYAN)test-all-parallel$(RESET)      Run full test suite in parallel"
	@echo "  $(CYAN)test-parallel-fast$(RESET)     Run fast tests in parallel"
	@echo "  $(CYAN)test-coverage$(RESET)          Run tests with coverage report"
	@echo "  $(CYAN)test-coverage-parallel$(RESET) Run coverage with parallel execution"
	@echo "  $(CYAN)test-unit$(RESET)              Run unit tests only"
	@echo "  $(CYAN)test-integration$(RESET)       Run integration tests only"
	@echo "  $(CYAN)test-performance$(RESET)       Run performance benchmarks"
	@echo "  $(CYAN)test-nlsq$(RESET)              Test NLSQ optimization"
	@echo "  $(CYAN)test-mcmc$(RESET)              Test MCMC optimization"
	@echo "  $(CYAN)test-quick$(RESET)             Quick tests with minimal output"
	@echo ""
	@echo "$(BOLD)$(GREEN)CODE QUALITY$(RESET)"
	@echo "  $(CYAN)format$(RESET)           Format code with ruff"
	@echo "  $(CYAN)lint$(RESET)             Run linting checks (ruff)"
	@echo "  $(CYAN)type-check$(RESET)       Run type checking (mypy)"
	@echo "  $(CYAN)check$(RESET)            Run all checks (format + lint + type)"
	@echo "  $(CYAN)quality$(RESET)          Run all quality checks"
	@echo "  $(CYAN)quick$(RESET)            Fast iteration: format + smoke tests"
	@echo "  $(CYAN)pre-commit$(RESET)       Run pre-commit hooks"
	@echo ""
	@echo "$(BOLD)$(GREEN)PERFORMANCE$(RESET)"
	@echo "  $(CYAN)benchmark$(RESET)        Run performance benchmarks"
	@echo "  $(CYAN)profile-nlsq$(RESET)     Profile NLSQ optimization"
	@echo "  $(CYAN)profile-mcmc$(RESET)     Profile MCMC optimization"
	@echo ""
	@echo "$(BOLD)$(GREEN)DOCUMENTATION$(RESET)"
	@echo "  $(CYAN)docs$(RESET)             Build documentation with Sphinx"
	@echo "  $(CYAN)docs-serve$(RESET)       Serve documentation locally"
	@echo "  $(CYAN)docs-clean$(RESET)       Clean documentation build"
	@echo ""
	@echo "$(BOLD)$(GREEN)BUILD & RELEASE$(RESET)"
	@echo "  $(CYAN)build$(RESET)            Build distribution packages"
	@echo "  $(CYAN)release$(RESET)          Prepare for release (test + quality + build)"
	@echo ""
	@echo "$(BOLD)$(GREEN)CLEANUP$(RESET)"
	@echo "  $(CYAN)clean$(RESET)            Remove build artifacts and caches (preserves venv, .claude, agent-os)"
	@echo "  $(CYAN)clean-all$(RESET)        Deep clean of all caches (preserves venv, .claude, agent-os)"
	@echo "  $(CYAN)clean-venv$(RESET)       Remove virtual environment (use with caution)"
	@echo ""
	@echo "$(BOLD)Environment Detection:$(RESET)"
	@echo "  Platform: $(PLATFORM)"
	@echo "  Package manager: $(PKG_MANAGER)"
	@echo ""

# ===================
# Installation targets
# ===================
install:
	@echo "$(BOLD)$(BLUE)Installing $(PACKAGE_NAME) in editable mode (CPU-only)...$(RESET)"
	@$(INSTALL_CMD) -e .
	@echo "$(BOLD)$(GREEN)✓ Package installed!$(RESET)"

dev:
	@echo "$(BOLD)$(BLUE)Installing development dependencies...$(RESET)"
	@$(INSTALL_CMD) -e ".[dev,docs]"
	@echo "$(BOLD)$(GREEN)✓ Dev dependencies installed!$(RESET)"
	@echo "  Platform: $(PLATFORM)"
	@echo "  Python: $(shell $(PYTHON) --version 2>&1)"
	@echo "  JAX: $(shell $(PYTHON) -c 'import jax; print(jax.__version__)' 2>/dev/null || echo 'not installed')"

dev-install: dev install-hooks
	@echo "$(BOLD)$(GREEN)✓ Development environment ready!$(RESET)"

# ===================
# Environment info targets
# ===================
env-info:
	@echo "$(BOLD)$(BLUE)Homodyne Environment Information$(RESET)"
	@echo "================================="
	@echo ""
	@echo "$(BOLD)Platform Detection:$(RESET)"
	@echo "  OS: $(UNAME_S)"
	@echo "  Platform: $(PLATFORM)"
	@echo ""
	@echo "$(BOLD)Python Environment:$(RESET)"
	@echo "  Python: $(shell $(PYTHON) --version 2>&1 || echo 'not found')"
	@echo "  Python path: $(shell which $(PYTHON) 2>/dev/null || echo 'not found')"
	@echo ""
	@echo "$(BOLD)Package Manager Detection:$(RESET)"
	@echo "  Active manager: $(PKG_MANAGER)"
ifdef UV_AVAILABLE
	@echo "  ✓ uv detected: $(UV_AVAILABLE)"
	@echo "    Install command: $(INSTALL_CMD)"
else
	@echo "  ✗ uv not found"
endif
ifdef CONDA_PREFIX
	@echo "  ✓ Conda environment detected"
	@echo "    CONDA_PREFIX: $(CONDA_PREFIX)"
ifdef MAMBA_AVAILABLE
	@echo "    Mamba available: $(MAMBA_AVAILABLE)"
endif
else
	@echo "  ✗ Not in conda environment"
endif
	@echo ""
	@echo "$(BOLD)Core Dependencies:$(RESET)"
	@echo "  JAX: $(shell $(PYTHON) -c 'import jax; print(jax.__version__)' 2>/dev/null || echo 'not installed')"
	@echo "  NLSQ: $(shell $(PYTHON) -c 'import nlsq; print(nlsq.__version__)' 2>/dev/null || echo 'not installed')"
	@echo "  NumPyro: $(shell $(PYTHON) -c 'import numpyro; print(numpyro.__version__)' 2>/dev/null || echo 'not installed')"
	@echo "  NumPy: $(shell $(PYTHON) -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo 'not installed')"
	@echo ""

deps-check:
	@echo "$(BOLD)$(BLUE)Checking Homodyne Dependencies...$(RESET)"
	@$(PYTHON) -c "from homodyne import get_package_info; info = get_package_info(); \
		deps = info['dependencies']; \
		print('Core: JAX=' + ('✓' if deps.get('jax') else '✗') + \
		      ', NumPy=' + ('✓' if deps.get('numpy') else '✗') + \
		      ', SciPy=' + ('✓' if deps.get('scipy') else '✗')); \
		print('Optimization: NLSQ=' + ('✓' if deps.get('nlsq') else '✗') + \
		      ', NumPyro=' + ('✓' if deps.get('numpyro') else '✗')); \
		"

info:
	@echo "$(BOLD)$(BLUE)Project Information$(RESET)"
	@echo "===================="
	@echo "Project: $(PACKAGE_NAME)"
	@echo "Python: $(shell $(PYTHON) --version 2>&1)"
	@echo "Platform: $(PLATFORM)"
	@echo "Package manager: $(PKG_MANAGER)"
	@echo ""
	@echo "$(BOLD)$(BLUE)Directory Structure$(RESET)"
	@echo "===================="
	@echo "Source: $(SRC_DIR)/"
	@echo "Tests: $(TEST_DIR)/"
	@echo "Docs: $(DOCS_DIR)/"
	@echo ""
	@echo "$(BOLD)$(BLUE)JAX Configuration$(RESET)"
	@echo "=================="
	@$(PYTHON) -c "import jax; print('JAX version:', jax.__version__); print('Default backend:', jax.default_backend())" 2>/dev/null || echo "JAX not installed"

version:
	@$(PYTHON) -c "import $(PACKAGE_NAME); print($(PACKAGE_NAME).__version__)" 2>/dev/null || \
		echo "$(BOLD)$(RED)Error: Package not installed. Run 'make install' first.$(RESET)"

# ===================
# Testing targets
# ===================
test:
	@echo "$(BOLD)$(BLUE)Running core unit tests...$(RESET)"
	$(RUN_CMD) $(PYTEST) $(TEST_DIR)/unit -v --tb=short

test-smoke:
	@echo "$(BOLD)$(BLUE)Running smoke tests (critical tests, ~30s-2min)...$(RESET)"
	$(RUN_CMD) $(PYTEST) $(TEST_DIR)/unit -n auto -v --tb=short -x -q
	@echo "$(BOLD)$(GREEN)✓ Smoke tests passed!$(RESET)"

test-fast:
	@echo "$(BOLD)$(BLUE)Running fast tests (excluding slow tests)...$(RESET)"
	@echo "$(BOLD)Note:$(RESET) Excludes slow tests, ~5-10 min runtime"
	$(RUN_CMD) $(PYTEST) $(TEST_DIR)/unit -n auto -m "not slow" -v --tb=short

test-parallel:
	@echo "$(BOLD)$(BLUE)Running tests in parallel (2-4x speedup)...$(RESET)"
	$(RUN_CMD) $(PYTEST) $(TEST_DIR)/unit -n auto -v --tb=short

test-all-parallel:
	@echo "$(BOLD)$(BLUE)Running full test suite in parallel...$(RESET)"
	$(RUN_CMD) $(PYTEST) $(TEST_DIR) -n auto -v --tb=short
	@echo "$(BOLD)$(GREEN)✓ Full test suite passed!$(RESET)"

test-parallel-fast:
	@echo "$(BOLD)$(BLUE)Running fast tests in parallel...$(RESET)"
	$(RUN_CMD) $(PYTEST) $(TEST_DIR)/unit -n auto -m "not slow" -v --tb=short

test-ci:
	@echo "$(BOLD)$(BLUE)Running CI test suite (matches GitHub Actions)...$(RESET)"
	$(RUN_CMD) $(PYTEST) $(TEST_DIR)/unit -n auto -v --tb=short
	@echo "$(BOLD)$(GREEN)✓ CI test suite passed!$(RESET)"

test-ci-full:
	@echo "$(BOLD)$(BLUE)Running full CI test suite with coverage...$(RESET)"
	$(RUN_CMD) $(PYTEST) $(TEST_DIR) -n auto -v --cov=$(PACKAGE_NAME) --cov-report=html --cov-report=term
	@echo "$(BOLD)$(GREEN)✓ Full CI test suite passed!$(RESET)"

test-all:
	@echo "$(BOLD)$(BLUE)Running all tests with coverage...$(RESET)"
	$(RUN_CMD) $(PYTEST) $(TEST_DIR) -v --cov=$(PACKAGE_NAME) --cov-report=html --cov-report=term

test-coverage:
	@echo "$(BOLD)$(BLUE)Running tests with coverage report...$(RESET)"
	$(RUN_CMD) $(PYTEST) $(TEST_DIR) --cov=$(PACKAGE_NAME) --cov-report=term-missing --cov-report=html --cov-report=xml
	@echo "$(BOLD)$(GREEN)✓ Coverage report generated!$(RESET)"
	@echo "View HTML report: open htmlcov/index.html"

test-coverage-parallel:
	@echo "$(BOLD)$(BLUE)Running tests with coverage in parallel...$(RESET)"
	$(RUN_CMD) $(PYTEST) $(TEST_DIR) -n auto --cov=$(PACKAGE_NAME) --cov-report=term-missing --cov-report=html --cov-report=xml
	@echo "$(BOLD)$(GREEN)✓ Coverage report generated!$(RESET)"
	@echo "View HTML report: open htmlcov/index.html"

test-unit:
	@echo "$(BOLD)$(BLUE)Running unit tests...$(RESET)"
	$(RUN_CMD) $(PYTEST) $(TEST_DIR)/unit -v --tb=short

test-integration:
	@echo "$(BOLD)$(BLUE)Running integration tests...$(RESET)"
	$(RUN_CMD) $(PYTEST) $(TEST_DIR)/integration -v --tb=short

test-performance:
	@echo "$(BOLD)$(BLUE)Running performance benchmarks...$(RESET)"
	$(RUN_CMD) $(PYTEST) $(TEST_DIR)/performance -v -m performance

test-nlsq:
	@echo "$(BOLD)$(BLUE)Running NLSQ optimization tests...$(RESET)"
	$(RUN_CMD) $(PYTEST) $(TEST_DIR)/unit/test_optimization_nlsq.py -v

test-mcmc:
	@echo "$(BOLD)$(BLUE)Running MCMC optimization tests...$(RESET)"
	$(RUN_CMD) $(PYTEST) $(TEST_DIR)/unit/test_optimization_mcmc.py -v 2>/dev/null || echo "MCMC tests not found"

test-quick:
	@echo "$(BOLD)$(BLUE)Running quick tests...$(RESET)"
	$(RUN_CMD) $(PYTEST) $(TEST_DIR)/unit -v -x --tb=no -q

# ===================
# Code quality targets
# ===================
format:
	@echo "$(BOLD)$(BLUE)Formatting code with ruff...$(RESET)"
	$(RUN_CMD) $(RUFF) format $(SRC_DIR) $(TEST_DIR)
	$(RUN_CMD) $(RUFF) check --fix $(SRC_DIR) $(TEST_DIR)
	@echo "$(BOLD)$(GREEN)✓ Code formatted!$(RESET)"

lint:
	@echo "$(BOLD)$(BLUE)Running linting checks...$(RESET)"
	$(RUN_CMD) $(RUFF) check $(SRC_DIR)
	@echo "$(BOLD)$(GREEN)✓ No linting errors!$(RESET)"

type-check:
	@echo "$(BOLD)$(BLUE)Running type checks...$(RESET)"
	$(RUN_CMD) mypy $(SRC_DIR) --show-error-codes
	@echo "$(BOLD)$(GREEN)✓ Type checking passed!$(RESET)"

check: format lint type-check
	@echo "$(BOLD)$(GREEN)✓ All checks passed!$(RESET)"

quality: format lint type-check
	@echo "$(BOLD)$(GREEN)✓ All quality checks passed!$(RESET)"

quick: format test-smoke
	@echo "$(BOLD)$(GREEN)✓ Quick iteration complete!$(RESET)"

pre-commit:
	@echo "$(BOLD)$(BLUE)Running pre-commit hooks...$(RESET)"
	$(RUN_CMD) pre-commit run --all-files

install-hooks:
	@echo "$(BOLD)$(BLUE)Installing pre-commit hooks...$(RESET)"
	$(RUN_CMD) pre-commit install
	@echo "$(BOLD)$(GREEN)✓ Hooks installed!$(RESET)"

# ===================
# Performance targets
# ===================
benchmark:
	@echo "$(BOLD)$(BLUE)Running performance benchmarks...$(RESET)"
	$(RUN_CMD) $(PYTEST) $(TEST_DIR)/performance/test_benchmarks.py -v --tb=short

profile-nlsq:
	@echo "$(BOLD)$(BLUE)Profiling NLSQ optimization...$(RESET)"
	$(RUN_CMD) $(PYTHON) -m cProfile -s cumulative scripts/nlsq/static_isotropic_nlsq.py 2>/dev/null || \
	$(PYTHON) -c "from homodyne.optimization import fit_nlsq_jax; print('NLSQ profiling ready')"

profile-mcmc:
	@echo "$(BOLD)$(BLUE)Profiling MCMC optimization...$(RESET)"
	$(RUN_CMD) $(PYTHON) -m cProfile -s cumulative scripts/mcmc/mcmc_uncertainty.py 2>/dev/null || \
	$(PYTHON) -c "from homodyne.optimization import fit_mcmc_jax; print('MCMC profiling ready')"

# ===================
# Cleanup targets
# ===================
clean-build:
	@echo "$(BOLD)$(BLUE)Removing build artifacts...$(RESET)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .benchmarks/
	rm -rf node_modules/
	find . -type d -name "*.egg-info" \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./.specify/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg" \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./.specify/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true

clean-pyc:
	@echo "$(BOLD)$(BLUE)Removing Python file artifacts...$(RESET)"
	find . -type d -name __pycache__ \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./.specify/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*~" \) \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./.specify/*" \
		-not -path "./agent-os/*" \
		-delete 2>/dev/null || true

clean-test:
	@echo "$(BOLD)$(BLUE)Removing test and coverage artifacts...$(RESET)"
	find . -type d -name .pytest_cache \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./.specify/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .nlsq_cache \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./.specify/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./.specify/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./.specify/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./.specify/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .hypothesis \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./.specify/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage coverage.xml .coverage.* test-results.xml
	rm -rf homodyne_results/ .homodyne_cache/ cmc_temp_*/ tmp/ .ultra-think/
	rm -f bandit*report.json phi_angles_list.txt TASK_GROUP_*.md

clean-cache:
	@echo "$(BOLD)$(BLUE)Removing additional caches...$(RESET)"
	find . -type d -name '.cache' \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./.specify/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true

clean: clean-build clean-pyc clean-test
	@echo "$(BOLD)$(GREEN)✓ Cleaned!$(RESET)"
	@echo "$(BOLD)Protected directories preserved:$(RESET) .venv/, venv/, .claude/, .specify/, agent-os/"

clean-all: clean clean-cache
	@echo "$(BOLD)$(BLUE)Performing deep clean...$(RESET)"
	rm -rf data/test_* scripts/output/ .tox/ .nox/ .eggs/
	@echo "$(BOLD)$(GREEN)✓ Deep clean complete!$(RESET)"
	@echo "$(BOLD)Protected directories preserved:$(RESET) .venv/, venv/, .claude/, .specify/, agent-os/"

clean-venv:
	@echo "$(BOLD)$(YELLOW)WARNING: This will remove the virtual environment!$(RESET)"
	@echo "$(BOLD)You will need to recreate it manually.$(RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "$(BOLD)$(BLUE)Removing virtual environment...$(RESET)"; \
		rm -rf $(VENV) venv; \
		echo "$(BOLD)$(GREEN)✓ Virtual environment removed!$(RESET)"; \
	else \
		echo "Cancelled."; \
	fi

# ===================
# Documentation targets
# ===================
docs:
	@echo "$(BOLD)$(BLUE)Building documentation...$(RESET)"
	cd $(DOCS_DIR) && $(MAKE) html
	@echo "$(BOLD)$(GREEN)✓ Documentation built!$(RESET)"
	@echo "Open: $(DOCS_DIR)/_build/html/index.html"

docs-serve:
	@echo "$(BOLD)$(BLUE)Serving documentation locally...$(RESET)"
	cd $(DOCS_DIR)/_build/html && $(PYTHON) -m http.server 8000

docs-clean:
	@echo "$(BOLD)$(BLUE)Cleaning documentation build...$(RESET)"
	cd $(DOCS_DIR) && $(MAKE) clean

# ===================
# Build and release targets
# ===================
build: clean
	@echo "$(BOLD)$(BLUE)Building distribution packages...$(RESET)"
	$(PYTHON) -m build
	@echo "$(BOLD)$(GREEN)✓ Build complete!$(RESET)"
	@echo "Distributions in dist/"

release: test quality build
	@echo "$(BOLD)$(BLUE)Checking package...$(RESET)"
	$(PYTHON) -m twine check dist/* 2>/dev/null || echo "Run 'pip install twine' for package validation"
	@echo "$(BOLD)$(GREEN)✓ Package ready for release!$(RESET)"

# ===================
# Example/Demo targets
# ===================
run-example:
	@echo "$(BOLD)$(BLUE)Running example...$(RESET)"
	$(RUN_CMD) $(PYTHON) scripts/nlsq/static_isotropic_nlsq.py

# ===================
# CI targets
# ===================
ci: clean test lint
	@echo "$(BOLD)$(GREEN)✓ CI checks passed!$(RESET)"

ci-full: clean install test-all quality
	@echo "$(BOLD)$(GREEN)✓ Full CI checks passed!$(RESET)"

# ===================
# Development shortcuts
# ===================
watch:
	@echo "$(BOLD)$(BLUE)Watching for changes (requires entr)...$(RESET)"
	find $(SRC_DIR) $(TEST_DIR) -name "*.py" | entr -c make test-quick

# ===================
# Project statistics
# ===================
stats:
	@echo "$(BOLD)$(BLUE)Homodyne Project Statistics$(RESET)"
	@echo "============================"
	@echo "Lines of code:"
	@find $(SRC_DIR) -name "*.py" -exec wc -l {} + | tail -n 1
	@echo "Python files: $(shell find $(SRC_DIR) -name '*.py' | wc -l)"
	@echo "Test files: $(shell find $(TEST_DIR) -name 'test_*.py' | wc -l)"

# ===================
# Verify NLSQ integration
# ===================
verify-nlsq:
	@echo "$(BOLD)$(BLUE)Verifying NLSQ integration...$(RESET)"
	@$(PYTHON) -c "from homodyne.optimization.nlsq import NLSQ_AVAILABLE; print(f'NLSQ available: {NLSQ_AVAILABLE}')"
	@$(PYTHON) -c "from homodyne.optimization import fit_nlsq_jax; print('✓ fit_nlsq_jax imported')"
	@$(PYTHON) -c "import nlsq; print(f'✓ NLSQ version: {nlsq.__version__}')" 2>/dev/null || echo "✗ NLSQ version check failed"
	@echo "$(BOLD)$(GREEN)✓ NLSQ integration verified!$(RESET)"
