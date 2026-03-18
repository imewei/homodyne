#!/bin/bash
# shellcheck disable=SC2034  # words/cword set by _init_completion convention
# Bash completion for homodyne CLI
#
# Installation:
#   Source this file in your .bashrc or copy to /etc/bash_completion.d/
#
# Features:
#   - Context-aware completions for options
#   - Config file caching (5-minute TTL)
#   - Method suggestions based on workflow

# Cache directory for completions
_HOMODYNE_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/homodyne"
_HOMODYNE_CACHE_TTL=300  # 5 minutes

# Ensure cache directory exists
_homodyne_ensure_cache() {
    [[ -d "$_HOMODYNE_CACHE_DIR" ]] || mkdir -p "$_HOMODYNE_CACHE_DIR"
}

# Fallback for _init_completion when bash-completion is not loaded
# (common in conda/mamba environments)
if ! type _init_completion &>/dev/null; then
    _init_completion() {
        COMPREPLY=()
        cur="${COMP_WORDS[COMP_CWORD]}"
        prev="${COMP_WORDS[COMP_CWORD-1]}"
        words=("${COMP_WORDS[@]}")
        cword=$COMP_CWORD
    }

    # Fallback for _filedir (directory completion)
    if ! type _filedir &>/dev/null; then
        _filedir() {
            if [[ "$1" == "-d" ]]; then
                mapfile -t COMPREPLY < <(compgen -d -- "${cur}")
            else
                mapfile -t COMPREPLY < <(compgen -f -- "${cur}")
            fi
        }
    fi
fi

# Get cached config files or update cache
_homodyne_get_config_files() {
    _homodyne_ensure_cache
    local cache_file="$_HOMODYNE_CACHE_DIR/config_files"
    local now
    now=$(date +%s)

    # Check cache validity
    if [[ -f "$cache_file" ]]; then
        local cache_time
        cache_time=$(stat -f %m "$cache_file" 2>/dev/null || stat -c %Y "$cache_file" 2>/dev/null)
        if [[ $((now - cache_time)) -lt $_HOMODYNE_CACHE_TTL ]]; then
            cat "$cache_file"
            return
        fi
    fi

    # Refresh cache: find YAML files in current and config directories
    {
        find . -maxdepth 2 \( -name "*.yaml" -o -name "*.yml" \) -type f 2>/dev/null
        [[ -d "config" ]] && find config \( -name "*.yaml" -o -name "*.yml" \) -type f 2>/dev/null
        [[ -d "configs" ]] && find configs \( -name "*.yaml" -o -name "*.yml" \) -type f 2>/dev/null
    } | sort -u | tee "$cache_file"
}

# Get HDF5 data files
_homodyne_get_data_files() {
    find . -maxdepth 3 \( -name "*.h5" -o -name "*.hdf5" -o -name "*.nxs" \) -type f 2>/dev/null
}

# Main homodyne completion
_homodyne() {
    local cur prev words cword
    _init_completion -s || return

    # Global options
    local global_opts="--config --data-file --method --output-dir --verbose --quiet --help --version"
    local mode_opts="--static-mode --laminar-flow"
    local output_opts="--output-format --save-plots --plot-experimental-data --plot-simulated-data --plotting-backend --parallel-plots"
    local nlsq_opts="--max-iterations --tolerance"
    local cmc_opts="--n-samples --n-warmup --n-chains --cmc-num-shards --cmc-backend --no-nlsq-warmstart --nlsq-result --dense-mass-matrix"

    # Method options
    local methods="nlsq cmc"

    case "$prev" in
        --config|-c)
            mapfile -t COMPREPLY < <(compgen -W "$(_homodyne_get_config_files)" -- "${cur}")
            return
            ;;
        --data-file|-d)
            mapfile -t COMPREPLY < <(compgen -W "$(_homodyne_get_data_files)" -- "${cur}")
            return
            ;;
        --method|-m)
            mapfile -t COMPREPLY < <(compgen -W "${methods}" -- "${cur}")
            return
            ;;
        --output-dir|-o)
            _filedir -d
            return
            ;;
        --nlsq-result)
            _filedir -d
            return
            ;;
        --output-format)
            mapfile -t COMPREPLY < <(compgen -W "yaml json npz" -- "${cur}")
            return
            ;;
        --plotting-backend)
            mapfile -t COMPREPLY < <(compgen -W "auto matplotlib datashader" -- "${cur}")
            return
            ;;
        --cmc-backend)
            mapfile -t COMPREPLY < <(compgen -W "auto pjit multiprocessing pbs" -- "${cur}")
            return
            ;;
        --cmc-num-shards)
            mapfile -t COMPREPLY < <(compgen -W "4 8 10 16 20 32" -- "${cur}")
            return
            ;;
        --log-level)
            mapfile -t COMPREPLY < <(compgen -W "DEBUG INFO WARNING ERROR" -- "${cur}")
            return
            ;;
    esac

    # If current word starts with -, complete options
    if [[ "$cur" == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "${global_opts} ${mode_opts} ${output_opts} ${nlsq_opts} ${cmc_opts}" -- "${cur}")
        return
    fi

    # Default: complete with config files
    mapfile -t COMPREPLY < <(compgen -W "$(_homodyne_get_config_files) ${global_opts}" -- "${cur}")
}

# homodyne-config completion
_homodyne_config() {
    local cur prev words cword
    _init_completion -s || return

    local opts="--mode --output --interactive --validate --force --help"

    case "$prev" in
        --mode|-m)
            mapfile -t COMPREPLY < <(compgen -W "static laminar_flow" -- "${cur}")
            return
            ;;
        --output|-o)
            _filedir
            return
            ;;
        --validate|-v)
            _filedir
            return
            ;;
    esac

    if [[ "$cur" == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "${opts}" -- "${cur}")
    fi
}

# homodyne-config-xla completion
_homodyne_config_xla() {
    local cur prev words cword
    _init_completion -s || return

    local opts="--mode --show --help"

    case "$prev" in
        --mode)
            mapfile -t COMPREPLY < <(compgen -W "auto nlsq cmc cmc-hpc" -- "${cur}")
            return
            ;;
    esac

    if [[ "$cur" == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "${opts}" -- "${cur}")
    fi
}

# homodyne-post-install completion
_homodyne_post_install() {
    local cur prev words cword
    _init_completion -s || return

    local opts="--interactive --shell --no-completion --no-xla --xla-mode --help"

    case "$prev" in
        --shell|-s)
            mapfile -t COMPREPLY < <(compgen -W "bash zsh fish" -- "${cur}")
            return
            ;;
        --xla-mode)
            mapfile -t COMPREPLY < <(compgen -W "auto nlsq cmc cmc-hpc" -- "${cur}")
            return
            ;;
    esac

    if [[ "$cur" == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "${opts}" -- "${cur}")
    fi
}

# homodyne-cleanup completion
_homodyne_cleanup() {
    local cur prev words cword
    _init_completion -s || return

    local opts="--dry-run --force --interactive --help"

    if [[ "$cur" == -* ]]; then
        mapfile -t COMPREPLY < <(compgen -W "${opts}" -- "${cur}")
    fi
}

# Register completions
complete -F _homodyne homodyne
complete -F _homodyne_config homodyne-config
complete -F _homodyne_config_xla homodyne-config-xla
complete -F _homodyne_post_install homodyne-post-install
complete -F _homodyne_cleanup homodyne-cleanup

# Short aliases (hm = homodyne)
complete -F _homodyne hm
complete -F _homodyne hm-nlsq
complete -F _homodyne hm-cmc
complete -F _homodyne_config hconfig
complete -F _homodyne_config hc-stat
complete -F _homodyne_config hc-flow
complete -F _homodyne_config_xla hxla
complete -F _homodyne_post_install hsetup
complete -F _homodyne_cleanup hclean

# Plotting aliases
alias hm='homodyne'
alias hconfig='homodyne-config'
alias hm-nlsq='homodyne --method nlsq'
alias hm-cmc='homodyne --method cmc'
alias hc-stat='homodyne-config --mode static'
alias hc-flow='homodyne-config --mode laminar_flow'
alias hexp='homodyne --plot-experimental-data'
alias hsim='homodyne --plot-simulated-data'
alias hxla='homodyne-config-xla'
alias hsetup='homodyne-post-install'
alias hclean='homodyne-cleanup'
complete -F _homodyne hexp
complete -F _homodyne hsim
