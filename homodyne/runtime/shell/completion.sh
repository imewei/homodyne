#!/bin/bash
# Advanced Homodyne Shell Completion System
# Provides intelligent completion with context awareness

# Cache for faster completion
HOMODYNE_COMPLETION_CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/homodyne"
HOMODYNE_COMPLETION_CACHE_FILE="$HOMODYNE_COMPLETION_CACHE_DIR/completion_cache"

# Initialize cache directory
_homodyne_init_cache() {
    [[ ! -d "$HOMODYNE_COMPLETION_CACHE_DIR" ]] && mkdir -p "$HOMODYNE_COMPLETION_CACHE_DIR"
}

# Get recent config files (cached for 5 minutes)
_homodyne_get_recent_configs() {
    local cache_age=300  # 5 minutes
    local current_time=$(date +%s)

    _homodyne_init_cache

    # Check if cache exists and is fresh
    if [[ -f "$HOMODYNE_COMPLETION_CACHE_FILE" ]]; then
        local cache_time=$(stat -c %Y "$HOMODYNE_COMPLETION_CACHE_FILE" 2>/dev/null || stat -f %m "$HOMODYNE_COMPLETION_CACHE_FILE" 2>/dev/null)
        if [[ $((current_time - cache_time)) -lt $cache_age ]]; then
            cat "$HOMODYNE_COMPLETION_CACHE_FILE"
            return
        fi
    fi

    # Rebuild cache
    {
        # Find JSON files in current and parent directories
        find . -maxdepth 2 -name "*.json" -type f 2>/dev/null | head -20
        # Add commonly used config names
        echo "homodyne_config.json"
        echo "config.json"
        echo "analysis_config.json"
    } | sort -u > "$HOMODYNE_COMPLETION_CACHE_FILE"

    cat "$HOMODYNE_COMPLETION_CACHE_FILE"
}

# Smart method completion based on config mode
_homodyne_smart_method_completion() {
    local config_file="$1"
    local methods="nlsq mcmc auto nuts cmc"

    # If config file exists, try to detect mode and suggest appropriate methods
    if [[ -f "$config_file" ]] && command -v python3 >/dev/null 2>&1; then
        local mode=$(python3 -c "
import json, yaml
try:
    with open('$config_file') as f:
        try:
            config = yaml.safe_load(f)
        except:
            config = json.load(f)
        mode = config.get('mode', '')
        if 'static' in mode:
            print('nlsq auto')
        elif 'laminar' in mode:
            print('mcmc auto cmc')
        else:
            print('$methods')
except:
    print('$methods')
" 2>/dev/null)
        echo "${mode:-$methods}"
    else
        echo "$methods"
    fi
}

# Advanced bash completion for homodyne
_homodyne_advanced_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # Main options
    local main_opts="--help --method --config --output-dir --verbose --quiet"
    local mode_opts="--static-isotropic --static-anisotropic --laminar-flow"
    local plot_opts="--plot-experimental-data --plot-simulated-data"
    local param_opts="--contrast --offset --phi-angles"

    case $prev in
        --method)
            # Smart method completion based on config
            local config_file=""
            for ((i=1; i<COMP_CWORD; i++)); do
                if [[ "${COMP_WORDS[i]}" == "--config" ]] && [[ $((i+1)) -lt $COMP_CWORD ]]; then
                    config_file="${COMP_WORDS[i+1]}"
                    break
                fi
            done

            local methods=$(_homodyne_smart_method_completion "$config_file")
            COMPREPLY=($(compgen -W "$methods" -- "$cur"))
            return 0
            ;;
        --config)
            # Use cached recent configs
            local configs=$(_homodyne_get_recent_configs)
            COMPREPLY=($(compgen -W "$configs" -- "$cur"))
            # Also add file completion
            COMPREPLY+=($(compgen -f -X '!*.json' -- "$cur"))
            return 0
            ;;
        --output-dir)
            # Smart directory completion with suggestions
            local common_dirs="./results ./output ./homodyne_results ./analysis"
            COMPREPLY=($(compgen -W "$common_dirs" -- "$cur"))
            COMPREPLY+=($(compgen -d -- "$cur"))
            return 0
            ;;
        --phi-angles)
            # Common angle sets
            local angles="0,45,90,135 0,36,72,108,144 0,30,60,90,120,150"
            COMPREPLY=($(compgen -W "$angles" -- "$cur"))
            return 0
            ;;
        --contrast|--offset)
            # Common values
            local values="0.0 0.5 1.0 1.5 2.0"
            COMPREPLY=($(compgen -W "$values" -- "$cur"))
            return 0
            ;;
        --cmc-num-shards)
            # Common shard counts for CMC
            local counts="4 8 10 16 20 32"
            COMPREPLY=($(compgen -W "$counts" -- "$cur"))
            return 0
            ;;
        --cmc-backend)
            # CMC backend options
            local backends="auto pjit multiprocessing pbs"
            COMPREPLY=($(compgen -W "$backends" -- "$cur"))
            return 0
            ;;
    esac

    # Check for incompatible options
    local has_mode=false
    for word in "${COMP_WORDS[@]}"; do
        if [[ "$word" =~ ^--(static-isotropic|static-anisotropic|laminar-flow)$ ]]; then
            has_mode=true
            break
        fi
    done

    if [[ $cur == -* ]]; then
        local all_opts="$main_opts $plot_opts $param_opts"
        [[ "$has_mode" == false ]] && all_opts="$all_opts $mode_opts"
        COMPREPLY=($(compgen -W "$all_opts" -- "$cur"))
    else
        # Default to config files
        COMPREPLY=($(compgen -f -X '!*.json' -- "$cur"))
    fi
}

# Bash completion for homodyne-config
_homodyne_config_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # homodyne-config options
    local main_opts="--help -h --mode -m --output -o --interactive -i --validate -v --force -f"

    case $prev in
        --mode|-m)
            # Updated modes: static (generic) and laminar_flow
            local modes="static laminar_flow"
            COMPREPLY=($(compgen -W "$modes" -- "$cur"))
            return 0
            ;;
        --output|-o)
            # Suggest common YAML config file names
            local suggestions="config.yaml homodyne_config.yaml my_config.yaml analysis_config.yaml"
            COMPREPLY=($(compgen -W "$suggestions" -- "$cur"))
            # Also add general YAML file completion
            COMPREPLY+=($(compgen -f -X '!*.yaml' -- "$cur"))
            COMPREPLY+=($(compgen -f -X '!*.yml' -- "$cur"))
            return 0
            ;;
        --validate|-v)
            # Complete with existing YAML files for validation
            COMPREPLY=($(compgen -f -X '!*.yaml' -- "$cur"))
            COMPREPLY+=($(compgen -f -X '!*.yml' -- "$cur"))
            return 0
            ;;
    esac

    if [[ $cur == -* ]]; then
        COMPREPLY=($(compgen -W "$main_opts" -- "$cur"))
    else
        # Default to YAML file completion for output
        COMPREPLY=($(compgen -f -X '!*.yaml' -- "$cur"))
        COMPREPLY+=($(compgen -f -X '!*.yml' -- "$cur"))
    fi
}

# Advanced zsh completion
if [[ -n "$ZSH_VERSION" ]]; then
    _homodyne_advanced_zsh() {
        local -a args
        args=(
            '(--help -h)'{--help,-h}'[Show help message]'
            '--method[Analysis method]:method:->methods'
            '--config[Configuration file]:file:->configs'
            '--output-dir[Output directory]:dir:_directories'
            '--verbose[Enable verbose logging]'
            '--quiet[Disable console logging]'
            '(--static-isotropic --static-anisotropic --laminar-flow)--static-isotropic[Static isotropic mode]'
            '(--static-isotropic --static-anisotropic --laminar-flow)--static-anisotropic[static mode]'
            '(--static-isotropic --static-anisotropic --laminar-flow)--laminar-flow[Laminar flow mode]'
            '--plot-experimental-data[Plot experimental data]'
            '--plot-simulated-data[Plot simulated data]'
            '--contrast[Contrast parameter]:value:'
            '--offset[Offset parameter]:value:'
            '--phi-angles[Phi angles (comma-separated)]:angles:'
            '*:config:_files -g "*.json"'
        )

        _arguments -C $args

        case $state in
            methods)
                local methods="nlsq mcmc auto nuts cmc"
                _values 'method' ${(z)methods}
                ;;
            configs)
                # Show recent configs first
                local -a configs
                configs=($(_homodyne_get_recent_configs))
                _describe 'recent configs' configs
                _files -g '*.json'
                ;;
        esac
    }

    # Zsh completion for homodyne-config
    _homodyne_config_zsh() {
        local -a args
        args=(
            '(--help -h)'{--help,-h}'[Show help message]'
            '(--mode -m)'{--mode,-m}'[Analysis mode]:mode:(static laminar_flow)'
            '(--output -o)'{--output,-o}'[Output configuration file]:file:_files -g "*.yaml"'
            '(--interactive -i)'{--interactive,-i}'[Interactive configuration builder]'
            '(--validate -v)'{--validate,-v}'[Validate configuration file]:file:_files -g "*.yaml"'
            '(--force -f)'{--force,-f}'[Force overwrite existing file]'
            '*:output:_files -g "*.yaml"'
        )

        _arguments -C $args
    }

    # Register completion for all homodyne commands
    compdef _homodyne_advanced_zsh homodyne 2>/dev/null || true
    compdef _homodyne_config_zsh homodyne-config 2>/dev/null || true

    # Register completion for method aliases (hm- prefix)
    compdef _homodyne_advanced_zsh hm 2>/dev/null || true         # homodyne
    compdef _homodyne_advanced_zsh hm-nlsq 2>/dev/null || true    # homodyne --method nlsq
    compdef _homodyne_advanced_zsh hm-mcmc 2>/dev/null || true    # homodyne --method mcmc
    compdef _homodyne_advanced_zsh hm-auto 2>/dev/null || true    # homodyne --method auto
    compdef _homodyne_advanced_zsh hm-nuts 2>/dev/null || true    # homodyne --method nuts
    compdef _homodyne_advanced_zsh hm-cmc 2>/dev/null || true     # homodyne --method cmc

    # Register completion for config aliases (hc- prefix)
    compdef _homodyne_config_zsh hconfig 2>/dev/null || true      # homodyne-config
    compdef _homodyne_config_zsh hc-stat 2>/dev/null || true      # homodyne-config --mode static
    compdef _homodyne_config_zsh hc-flow 2>/dev/null || true      # homodyne-config --mode laminar_flow

    # Register completion for utility aliases
    compdef _homodyne_advanced_zsh hexp 2>/dev/null || true       # homodyne --plot-experimental-data
    compdef _homodyne_advanced_zsh hsim 2>/dev/null || true       # homodyne --plot-simulated-data
fi

# Register bash completion
if [[ -n "$BASH_VERSION" ]]; then
    # Register completion for all homodyne commands
    complete -F _homodyne_advanced_completion homodyne 2>/dev/null || true
    complete -F _homodyne_config_completion homodyne-config 2>/dev/null || true

    # Register completion for method aliases (hm- prefix)
    complete -F _homodyne_advanced_completion hm 2>/dev/null || true         # homodyne
    complete -F _homodyne_advanced_completion hm-nlsq 2>/dev/null || true    # homodyne --method nlsq
    complete -F _homodyne_advanced_completion hm-mcmc 2>/dev/null || true    # homodyne --method mcmc
    complete -F _homodyne_advanced_completion hm-auto 2>/dev/null || true    # homodyne --method auto
    complete -F _homodyne_advanced_completion hm-nuts 2>/dev/null || true    # homodyne --method nuts
    complete -F _homodyne_advanced_completion hm-cmc 2>/dev/null || true     # homodyne --method cmc

    # Register completion for config aliases (hc- prefix)
    complete -F _homodyne_config_completion hconfig 2>/dev/null || true      # homodyne-config
    complete -F _homodyne_config_completion hc-stat 2>/dev/null || true      # homodyne-config --mode static
    complete -F _homodyne_config_completion hc-flow 2>/dev/null || true      # homodyne-config --mode laminar_flow

    # Register completion for utility aliases
    complete -F _homodyne_advanced_completion hexp 2>/dev/null || true       # homodyne --plot-experimental-data
    complete -F _homodyne_advanced_completion hsim 2>/dev/null || true       # homodyne --plot-simulated-data
fi

# Define aliases for convenience
# These aliases provide quick access to different analysis methods and configurations
if [[ -n "$BASH_VERSION" ]] || [[ -n "$ZSH_VERSION" ]]; then
    # Base command aliases
    alias hm='homodyne'                            # Homodyne base command
    alias hconfig='homodyne-config'                # Configuration generator

    # Method aliases (hm- prefix)
    alias hm-nlsq='homodyne --method nlsq'         # NLSQ trust-region optimization (primary)
    alias hm-mcmc='homodyne --method mcmc'         # Alias for auto (NUTS/CMC based on dataset size)
    alias hm-auto='homodyne --method auto'         # Auto-select NUTS (<500k) or CMC (>500k)
    alias hm-nuts='homodyne --method nuts'         # Standard NUTS MCMC
    alias hm-cmc='homodyne --method cmc'           # Consensus Monte Carlo for large datasets

    # Config mode aliases (hc- prefix)
    alias hc-stat='homodyne-config --mode static'         # Generate static mode config
    alias hc-flow='homodyne-config --mode laminar_flow'   # Generate laminar_flow mode config

    # Utility aliases
    alias hexp='homodyne --plot-experimental-data'        # Plot experimental data
    alias hsim='homodyne --plot-simulated-data'           # Plot simulated data
fi

# Quick command builder function
homodyne_build() {
    local cmd="homodyne"
    local method=""
    local config=""
    local output=""

    echo "🔧 Homodyne Command Builder"
    echo ""

    # Select method
    PS3="Select analysis method: "
    select m in "nlsq" "mcmc" "auto" "nuts" "cmc" "skip"; do
        [[ -n "$m" ]] && [[ "$m" != "skip" ]] && method="--method $m"
        break
    done

    # Select config
    echo ""
    echo "Available config files:"
    local configs=($(_homodyne_get_recent_configs))
    if [[ ${#configs[@]} -gt 0 ]]; then
        PS3="Select config file: "
        select c in "${configs[@]}" "manual" "skip"; do
            if [[ "$c" == "manual" ]]; then
                read -p "Enter config path: " config
                config="--config $config"
            elif [[ "$c" != "skip" ]] && [[ -n "$c" ]]; then
                config="--config $c"
            fi
            break
        done
    else
        read -p "Enter config file path (or press Enter to skip): " c
        [[ -n "$c" ]] && config="--config $c"
    fi

    # No GPU options needed - homodyne handles GPU automatically via JAX

    # Build and show command
    echo ""
    echo "📋 Generated command:"
    echo "  $cmd $method $config"
    echo ""
    read -p "Run this command? (y/N): " run_it

    if [[ "$run_it" =~ ^[Yy] ]]; then
        echo "🚀 Running..."
        eval "$cmd $method $config"
    else
        echo "💡 Command copied to clipboard (if available)"
        echo "$cmd $method $config" | xclip -selection clipboard 2>/dev/null || \
        echo "$cmd $method $config" | pbcopy 2>/dev/null || \
        echo "Copy command: $cmd $method $config"
    fi
}
