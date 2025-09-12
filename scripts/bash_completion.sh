#!/bin/bash
#
# Bash completion script for Homodyne v2 CLI
# ==========================================
#
# Provides intelligent tab completion for homodyne CLI commands,
# options, file paths, and parameter values.
#
# Installation:
#   sudo cp bash_completion.sh /etc/bash_completion.d/homodyne
#   # OR
#   source bash_completion.sh  # In your .bashrc
#
# Features:
# - Command and option completion
# - File path completion for data files
# - Method name completion (vi, mcmc, hybrid)
# - Analysis mode completion
# - Configuration file completion
#

_homodyne_complete() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # Main options and flags
    opts="--help --version --verbose --quiet --config --output-dir --method
          --static-isotropic --static-anisotropic --laminar-flow
          --force-cpu --gpu-memory-fraction --disable-dataset-optimization
          --plot-experimental-data --plot-simulated-data
          --phi-angles --vi-iterations --vi-learning-rate
          --mcmc-samples --mcmc-chains --mcmc-warmup"

    # Method completion
    if [[ ${prev} == "--method" ]]; then
        COMPREPLY=($(compgen -W "vi mcmc hybrid" -- ${cur}))
        return 0
    fi

    # Config file completion
    if [[ ${prev} == "--config" ]]; then
        COMPREPLY=($(compgen -f -X '!*.@(yaml|yml|json)' -- ${cur}))
        return 0
    fi

    # Output directory completion
    if [[ ${prev} == "--output-dir" ]]; then
        COMPREPLY=($(compgen -d -- ${cur}))
        return 0
    fi

    # GPU memory fraction completion
    if [[ ${prev} == "--gpu-memory-fraction" ]]; then
        COMPREPLY=($(compgen -W "0.2 0.4 0.6 0.8 0.9" -- ${cur}))
        return 0
    fi

    # VI iterations completion
    if [[ ${prev} == "--vi-iterations" ]]; then
        COMPREPLY=($(compgen -W "1000 2000 5000 10000" -- ${cur}))
        return 0
    fi

    # VI learning rate completion
    if [[ ${prev} == "--vi-learning-rate" ]]; then
        COMPREPLY=($(compgen -W "0.001 0.01 0.1" -- ${cur}))
        return 0
    fi

    # MCMC samples completion
    if [[ ${prev} == "--mcmc-samples" ]]; then
        COMPREPLY=($(compgen -W "500 1000 2000 5000" -- ${cur}))
        return 0
    fi

    # MCMC chains completion
    if [[ ${prev} == "--mcmc-chains" ]]; then
        COMPREPLY=($(compgen -W "2 4 8 16" -- ${cur}))
        return 0
    fi

    # MCMC warmup completion
    if [[ ${prev} == "--mcmc-warmup" ]]; then
        COMPREPLY=($(compgen -W "500 1000 2000" -- ${cur}))
        return 0
    fi

    # Phi angles completion (common values)
    if [[ ${prev} == "--phi-angles" ]]; then
        COMPREPLY=($(compgen -W "0 0.785 1.571 2.356 3.142" -- ${cur}))
        return 0
    fi

    # Default completion for options and file paths
    if [[ ${cur} == -* ]]; then
        COMPREPLY=($(compgen -W "${opts}" -- ${cur}))
        return 0
    fi

    # File completion for data files (first argument)
    if [[ ${COMP_CWORD} -eq 1 ]]; then
        COMPREPLY=($(compgen -f -X '!*.@(h5|hdf5|nc|dat)' -- ${cur}))
        return 0
    fi

    # Default file completion
    COMPREPLY=($(compgen -f -- ${cur}))
    return 0
}

# Register completion function
complete -F _homodyne_complete homodyne
complete -F _homodyne_complete homodyne-analyze

# Also provide completion for python module execution
_homodyne_module_complete() {
    local cur prev
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # If we're completing after "python -m homodyne"
    if [[ ${COMP_WORDS[2]} == "homodyne" ]]; then
        # Shift the completion context
        COMP_WORDS=("homodyne" "${COMP_WORDS[@]:3}")
        COMP_CWORD=$((COMP_CWORD - 2))
        _homodyne_complete
    fi
}

# Handle python -m homodyne completion
complete -F _homodyne_module_complete python python3