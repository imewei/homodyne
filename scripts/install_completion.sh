#!/bin/bash
#
# Installation Script for Homodyne v2 Shell Completion
# ====================================================
#
# Installs bash completion scripts system-wide or user-specific.
# Supports multiple shell environments and installation methods.
#
# Usage:
#   ./install_completion.sh [--system|--user] [--shell bash|zsh|fish]
#

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPLETION_FILE="bash_completion.sh"
COMPLETION_PATH="${SCRIPT_DIR}/${COMPLETION_FILE}"

# Default settings
INSTALL_TYPE="user"
SHELL_TYPE="bash"
VERBOSE=false

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Usage information
show_usage() {
    cat << EOF
Homodyne v2 Shell Completion Installer

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --system        Install system-wide (requires sudo)
    --user          Install for current user only (default)
    --shell SHELL   Target shell (bash, zsh, fish)
    --verbose       Show detailed output
    --help          Show this help message

EXAMPLES:
    $0                          # Install for current user (bash)
    $0 --system                 # Install system-wide (bash)
    $0 --user --shell zsh       # Install for user (zsh)

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --system)
                INSTALL_TYPE="system"
                shift
                ;;
            --user)
                INSTALL_TYPE="user"
                shift
                ;;
            --shell)
                SHELL_TYPE="$2"
                shift 2
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

# Detect current shell
detect_shell() {
    if [[ -n "$ZSH_VERSION" ]]; then
        echo "zsh"
    elif [[ -n "$BASH_VERSION" ]]; then
        echo "bash"
    elif [[ -n "$FISH_VERSION" ]]; then
        echo "fish"
    else
        echo "bash"  # Default fallback
    fi
}

# Validate shell type
validate_shell() {
    case $SHELL_TYPE in
        bash|zsh|fish)
            ;;
        *)
            log_error "Unsupported shell: $SHELL_TYPE"
            log_info "Supported shells: bash, zsh, fish"
            exit 1
            ;;
    esac
}

# Check if completion file exists
check_completion_file() {
    if [[ ! -f "$COMPLETION_PATH" ]]; then
        log_error "Completion file not found: $COMPLETION_PATH"
        log_info "Make sure you're running this script from the homodyne/scripts directory"
        exit 1
    fi
}

# Install bash completion
install_bash_completion() {
    local target_dir
    local target_file
    
    if [[ "$INSTALL_TYPE" == "system" ]]; then
        # System-wide installation
        target_dir="/etc/bash_completion.d"
        target_file="${target_dir}/homodyne"
        
        log_info "Installing bash completion system-wide..."
        
        if [[ ! -d "$target_dir" ]]; then
            log_error "System bash completion directory not found: $target_dir"
            log_info "Install bash-completion package first"
            exit 1
        fi
        
        if [[ $EUID -ne 0 ]]; then
            log_error "System installation requires sudo privileges"
            exit 1
        fi
        
        cp "$COMPLETION_PATH" "$target_file"
        chmod 644 "$target_file"
        
    else
        # User-specific installation
        target_dir="$HOME/.local/share/bash-completion/completions"
        target_file="${target_dir}/homodyne"
        
        log_info "Installing bash completion for current user..."
        
        mkdir -p "$target_dir"
        cp "$COMPLETION_PATH" "$target_file"
        chmod 644 "$target_file"
        
        # Add to .bashrc if not already present
        local bashrc="$HOME/.bashrc"
        local completion_line="source $target_file"
        
        if [[ -f "$bashrc" ]] && ! grep -q "$completion_line" "$bashrc"; then
            log_info "Adding completion source to .bashrc..."
            echo "" >> "$bashrc"
            echo "# Homodyne completion" >> "$bashrc"
            echo "$completion_line" >> "$bashrc"
        fi
    fi
    
    log_success "Bash completion installed: $target_file"
}

# Install zsh completion
install_zsh_completion() {
    log_warning "Zsh completion not yet implemented"
    log_info "For now, you can source the bash completion directly:"
    log_info "  echo 'source $COMPLETION_PATH' >> ~/.zshrc"
}

# Install fish completion
install_fish_completion() {
    log_warning "Fish completion not yet implemented"
    log_info "Fish completion would go in ~/.config/fish/completions/"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    case $SHELL_TYPE in
        bash)
            if command -v homodyne &> /dev/null; then
                log_success "Homodyne command found"
                
                # Test completion (this is tricky to automate)
                log_info "To test completion, run: homodyne <TAB><TAB>"
                log_info "You may need to restart your terminal or run: source ~/.bashrc"
            else
                log_warning "Homodyne command not found in PATH"
                log_info "Make sure homodyne is installed: pip install homodyne"
            fi
            ;;
        *)
            log_info "Manual verification required for $SHELL_TYPE"
            ;;
    esac
}

# Show post-installation instructions
show_post_install() {
    log_success "Installation completed!"
    echo
    log_info "Next steps:"
    
    case $SHELL_TYPE in
        bash)
            if [[ "$INSTALL_TYPE" == "user" ]]; then
                echo "  1. Restart your terminal OR run: source ~/.bashrc"
            else
                echo "  1. Restart your terminal"
            fi
            echo "  2. Test completion: homodyne <TAB><TAB>"
            ;;
        *)
            echo "  1. Restart your terminal"
            echo "  2. Test completion manually"
            ;;
    esac
    
    echo
    log_info "Completion features:"
    echo "  • Command options and flags"
    echo "  • Method names (vi, mcmc, hybrid)"
    echo "  • File path completion for data files"
    echo "  • Parameter value suggestions"
    echo "  • Configuration file completion"
}

# Main execution
main() {
    echo "Homodyne v2 Shell Completion Installer"
    echo "======================================"
    echo
    
    # Parse arguments
    parse_args "$@"
    
    # Auto-detect shell if not specified
    if [[ "$SHELL_TYPE" == "auto" ]]; then
        SHELL_TYPE=$(detect_shell)
        log_info "Detected shell: $SHELL_TYPE"
    fi
    
    # Validate inputs
    validate_shell
    check_completion_file
    
    # Show installation info
    log_info "Installation settings:"
    echo "  Target: $INSTALL_TYPE"
    echo "  Shell: $SHELL_TYPE"
    echo "  Source: $COMPLETION_PATH"
    echo
    
    # Perform installation
    case $SHELL_TYPE in
        bash)
            install_bash_completion
            ;;
        zsh)
            install_zsh_completion
            ;;
        fish)
            install_fish_completion
            ;;
    esac
    
    # Verify and show next steps
    verify_installation
    show_post_install
}

# Execute main function with all arguments
main "$@"