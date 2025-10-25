#!/bin/bash
#
# Example 8: Shell Completion Setup
#
# Sets up bash/zsh shell completion for faster homodyne CLI usage.
#
# Usage: bash setup_shell_completion.sh [bash|zsh]
#

set -e

SHELL_TYPE="${1:-bash}"

echo "Homodyne Shell Completion Setup"
echo "================================"

# Detect shell if not provided
if [ -z "$1" ]; then
    if [ -n "$BASH_VERSION" ]; then
        SHELL_TYPE="bash"
    elif [ -n "$ZSH_VERSION" ]; then
        SHELL_TYPE="zsh"
    else
        echo "Could not detect shell. Please specify: bash or zsh"
        echo "Usage: bash setup_shell_completion.sh [bash|zsh]"
        exit 1
    fi
    echo "Detected shell: $SHELL_TYPE"
fi

case "$SHELL_TYPE" in
    bash)
        echo ""
        echo "Setting up Bash completion..."

        # Generate completion script
        COMPLETION_FILE="$HOME/.homodyne-completion.bash"
        echo "Generating completion to $COMPLETION_FILE..."

        # Create or update completion script
        cat > "$COMPLETION_FILE" << 'EOF'
# Homodyne bash completion
_homodyne_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    opts="--config --data-file --method --output-dir --plot-experimental-data --verbose --quiet --help --version"

    if [[ ${cur} == -* ]] ; then
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
        return 0
    fi

    # Complete filenames after --config
    if [[ ${prev} == "--config" ]] || [[ ${prev} == "--data-file" ]]; then
        COMPREPLY=( $(compgen -f -- ${cur}) )
        return 0
    fi

    # Complete method names after --method
    if [[ ${prev} == "--method" ]]; then
        COMPREPLY=( $(compgen -W "nlsq mcmc" -- ${cur}) )
        return 0
    fi
}

complete -o bashdefault -o default -o nospace -F _homodyne_completion homodyne
EOF

        # Add to ~/.bashrc if not already there
        if ! grep -q "homodyne-completion.bash" "$HOME/.bashrc" 2>/dev/null; then
            echo ""
            echo "Adding to ~/.bashrc..."
            cat >> "$HOME/.bashrc" << 'EOF'

# Homodyne bash completion
if [ -f ~/.homodyne-completion.bash ]; then
    source ~/.homodyne-completion.bash
fi
EOF
            echo "✓ Added to ~/.bashrc"
        else
            echo "✓ Already in ~/.bashrc"
        fi

        echo ""
        echo "Reload bash to enable completion:"
        echo "  source ~/.bashrc"
        echo ""
        echo "Test completion:"
        echo "  homodyne --<TAB>"
        ;;

    zsh)
        echo ""
        echo "Setting up Zsh completion..."

        # Create completion directory
        COMPLETION_DIR="$HOME/.zsh/completions"
        mkdir -p "$COMPLETION_DIR"

        # Generate completion script
        COMPLETION_FILE="$COMPLETION_DIR/_homodyne"
        cat > "$COMPLETION_FILE" << 'EOF'
#compdef homodyne

_homodyne() {
    local -a subcommands
    local -a options

    options=(
        '--config[Configuration YAML file]:file:_files'
        '--data-file[Override data file path]:file:_files'
        '--method[Optimization method]:method:(nlsq mcmc)'
        '--output-dir[Output directory]:dir:_files -/'
        '--plot-experimental-data[Create plots]'
        '--verbose[Verbose output]'
        '--quiet[Quiet output]'
        '--help[Show help]'
        '--version[Show version]'
    )

    _arguments -s "$options[@]"
}

_homodyne "$@"
EOF

        # Add to ~/.zshrc if not already there
        if ! grep -q "homodyne" "$HOME/.zshrc" 2>/dev/null; then
            echo ""
            echo "Adding to ~/.zshrc..."
            cat >> "$HOME/.zshrc" << 'EOF'

# Homodyne zsh completion
fpath=(~/.zsh/completions $fpath)
autoload -Uz compinit && compinit
EOF
            echo "✓ Added to ~/.zshrc"
        else
            echo "✓ Already in ~/.zshrc"
        fi

        echo ""
        echo "Reload zsh to enable completion:"
        echo "  exec zsh"
        echo ""
        echo "Test completion:"
        echo "  homodyne --<TAB>"
        ;;

    *)
        echo "Unknown shell: $SHELL_TYPE"
        echo "Please specify: bash or zsh"
        exit 1
        ;;
esac

echo ""
echo "Setup complete!"
echo ""
echo "Benefits of shell completion:"
echo "  - Faster CLI typing"
echo "  - Automatic file completion"
echo "  - Method suggestions"
echo "  - Option discovery"
