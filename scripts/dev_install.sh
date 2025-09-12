#!/bin/bash
#
# Development Installation Script for Homodyne v2
# ===============================================
#
# Complete development environment setup with all dependencies,
# pre-commit hooks, and shell completion.
#
# Usage:
#   ./dev_install.sh [--gpu] [--minimal] [--force]
#

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Installation options
INSTALL_GPU=false
MINIMAL_INSTALL=false
FORCE_INSTALL=false
VERBOSE=false

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

log_step() {
    echo -e "${CYAN}➤${NC} $1"
}

# Usage information
show_usage() {
    cat << EOF
Homodyne v2 Development Environment Setup

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --gpu           Install GPU-accelerated JAX (CUDA)
    --minimal       Minimal development setup (no optional tools)
    --force         Force reinstall even if already installed
    --verbose       Show detailed output
    --help          Show this help message

WHAT THIS INSTALLS:
    ✓ Homodyne in development mode (pip install -e .)
    ✓ All development dependencies (pytest, black, ruff, mypy)
    ✓ Pre-commit hooks for code quality
    ✓ Shell completion (bash)
    ✓ GPU support (with --gpu)
    ✓ Documentation tools (with full install)

EXAMPLES:
    $0                      # Standard development install
    $0 --gpu                # With GPU support
    $0 --minimal            # Minimal setup for CI/testing

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --gpu)
                INSTALL_GPU=true
                shift
                ;;
            --minimal)
                MINIMAL_INSTALL=true
                shift
                ;;
            --force)
                FORCE_INSTALL=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Check system requirements
check_requirements() {
    log_step "Checking system requirements"
    
    # Check Python version
    if ! python3 --version &> /dev/null; then
        log_error "Python 3 not found. Please install Python 3.8 or later"
        exit 1
    fi
    
    local python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log_info "Python version: $python_version"
    
    # Check pip
    if ! pip3 --version &> /dev/null; then
        log_error "pip3 not found. Please install pip"
        exit 1
    fi
    
    # Check git
    if ! git --version &> /dev/null; then
        log_warning "git not found. Some features may not work"
    fi
    
    # Check if we're in the right directory
    if [[ ! -f "$PROJECT_ROOT/setup.py" ]]; then
        log_error "setup.py not found. Make sure you're running from homodyne/scripts/"
        exit 1
    fi
    
    log_success "System requirements check passed"
}

# Create virtual environment
setup_virtual_env() {
    local venv_path="$PROJECT_ROOT/venv"
    
    if [[ -d "$venv_path" ]] && [[ "$FORCE_INSTALL" == false ]]; then
        log_info "Virtual environment already exists: $venv_path"
        log_info "Use --force to recreate"
    else
        log_step "Creating virtual environment"
        
        if [[ -d "$venv_path" ]]; then
            log_info "Removing existing virtual environment"
            rm -rf "$venv_path"
        fi
        
        python3 -m venv "$venv_path"
        log_success "Virtual environment created: $venv_path"
    fi
    
    # Activate virtual environment
    log_info "Activating virtual environment"
    source "$venv_path/bin/activate"
    
    # Upgrade pip
    log_info "Upgrading pip"
    pip install --upgrade pip setuptools wheel
    
    log_success "Virtual environment setup complete"
}

# Install homodyne in development mode
install_homodyne() {
    log_step "Installing Homodyne in development mode"
    
    cd "$PROJECT_ROOT"
    
    local install_cmd="pip install -e ."
    
    # Add dependency groups
    local extras=""
    if [[ "$MINIMAL_INSTALL" == true ]]; then
        extras="[dev]"
        log_info "Minimal development install"
    else
        if [[ "$INSTALL_GPU" == true ]]; then
            extras="[full,dev,gpu,docs,perf]"
            log_info "Full development install with GPU support"
        else
            extras="[full,dev,docs,perf]"
            log_info "Full development install"
        fi
    fi
    
    install_cmd="$install_cmd$extras"
    
    log_info "Running: $install_cmd"
    eval "$install_cmd"
    
    log_success "Homodyne installed in development mode"
}

# Setup pre-commit hooks
setup_pre_commit() {
    if [[ "$MINIMAL_INSTALL" == true ]]; then
        log_info "Skipping pre-commit setup (minimal install)"
        return
    fi
    
    log_step "Setting up pre-commit hooks"
    
    cd "$PROJECT_ROOT"
    
    # Check if .pre-commit-config.yaml exists
    if [[ ! -f ".pre-commit-config.yaml" ]]; then
        log_warning ".pre-commit-config.yaml not found, creating basic config"
        create_pre_commit_config
    fi
    
    # Install pre-commit hooks
    pre-commit install
    
    # Run pre-commit on all files (optional)
    log_info "Running pre-commit on all files (this may take a moment)..."
    pre-commit run --all-files || log_warning "Some pre-commit checks failed (this is normal for first run)"
    
    log_success "Pre-commit hooks installed"
}

# Create basic pre-commit configuration
create_pre_commit_config() {
    cat > "$PROJECT_ROOT/.pre-commit-config.yaml" << EOF
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict

  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.0.290
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-PyYAML, types-requests]
EOF
    
    log_info "Created basic pre-commit configuration"
}

# Install shell completion
install_shell_completion() {
    log_step "Installing shell completion"
    
    local completion_script="$SCRIPT_DIR/install_completion.sh"
    
    if [[ -x "$completion_script" ]]; then
        "$completion_script" --user --shell bash
        log_success "Shell completion installed"
    else
        log_warning "Shell completion script not found or not executable"
    fi
}

# Verify installation
verify_installation() {
    log_step "Verifying installation"
    
    # Test homodyne CLI
    if homodyne --version &> /dev/null; then
        local version=$(homodyne --version)
        log_success "Homodyne CLI working: $version"
    else
        log_error "Homodyne CLI not working"
        return 1
    fi
    
    # Test Python API
    if python3 -c "import homodyne; print(f'Homodyne v{homodyne.__version__} imported successfully')" 2>/dev/null; then
        log_success "Python API working"
    else
        log_error "Python API import failed"
        return 1
    fi
    
    # Test JAX installation
    if python3 -c "import jax; print(f'JAX {jax.__version__} available')" 2>/dev/null; then
        log_success "JAX available"
        
        # Check GPU support
        if [[ "$INSTALL_GPU" == true ]]; then
            if python3 -c "import jax; print(f'GPUs: {len(jax.devices(\"gpu\"))}')" 2>/dev/null; then
                log_success "GPU support configured"
            else
                log_warning "GPU support installed but no GPUs detected"
            fi
        fi
    else
        log_warning "JAX not available (optional dependency)"
    fi
    
    log_success "Installation verification complete"
}

# Show development workflow information
show_dev_info() {
    echo
    log_success "Development environment setup complete!"
    echo
    log_info "Development Workflow:"
    echo "  📁 Project root: $PROJECT_ROOT"
    echo "  🐍 Virtual env: $PROJECT_ROOT/venv"
    echo "  🔧 Editable install: Changes take effect immediately"
    echo
    log_info "Useful Commands:"
    echo "  source $PROJECT_ROOT/venv/bin/activate  # Activate environment"
    echo "  homodyne --help                         # Test CLI"
    echo "  python -m homodyne --help               # Alternative CLI"
    echo "  python -c 'import homodyne; print(homodyne.__version__)'  # Test API"
    echo
    log_info "Code Quality Tools:"
    echo "  black .                    # Format code"
    echo "  ruff .                     # Lint code"
    echo "  mypy homodyne             # Type check"
    echo "  pytest                    # Run tests"
    echo "  pre-commit run --all-files # Run all checks"
    echo
    log_info "Testing:"
    echo "  pytest homodyne/tests/     # Run test suite"
    echo "  pytest -v                  # Verbose output"
    echo "  pytest -k 'test_vi'        # Run specific tests"
    echo
    
    if [[ "$MINIMAL_INSTALL" == false ]]; then
        log_info "Documentation:"
        echo "  cd docs && make html       # Build documentation"
        echo "  python -m http.server 8000 # Serve docs locally"
        echo
    fi
    
    log_info "Next Steps:"
    echo "  1. Activate the virtual environment"
    echo "  2. Run the test suite to verify everything works"
    echo "  3. Try running an analysis: homodyne --help"
    echo "  4. Start developing! 🚀"
}

# Main execution
main() {
    echo "Homodyne v2 Development Environment Setup"
    echo "========================================="
    echo
    
    # Parse arguments
    parse_args "$@"
    
    # Show configuration
    log_info "Installation Configuration:"
    echo "  GPU support: $([ "$INSTALL_GPU" == true ] && echo "✓ Enabled" || echo "○ Disabled")"
    echo "  Minimal install: $([ "$MINIMAL_INSTALL" == true ] && echo "✓ Yes" || echo "○ No")"
    echo "  Force reinstall: $([ "$FORCE_INSTALL" == true ] && echo "✓ Yes" || echo "○ No")"
    echo
    
    # Execute installation steps
    check_requirements
    setup_virtual_env
    install_homodyne
    
    if [[ "$MINIMAL_INSTALL" == false ]]; then
        setup_pre_commit
        install_shell_completion
    fi
    
    verify_installation
    show_dev_info
}

# Run main function with all arguments
main "$@"