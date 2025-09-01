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
    local methods="classical mcmc robust all"
    
    # If config file exists, try to detect mode and suggest appropriate methods
    if [[ -f "$config_file" ]] && command -v python3 >/dev/null 2>&1; then
        local mode=$(python3 -c "
import json
try:
    with open('$config_file') as f:
        config = json.load(f)
        mode = config.get('mode', '')
        if 'static' in mode:
            print('classical robust')
        elif 'laminar' in mode:
            print('mcmc robust all')
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
            '(--static-isotropic --static-anisotropic --laminar-flow)--static-anisotropic[Static anisotropic mode]'
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
                local methods="classical mcmc robust all"
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
    
    # Register advanced completion
    compdef _homodyne_advanced_zsh homodyne 2>/dev/null || true
    compdef _homodyne_advanced_zsh homodyne-gpu 2>/dev/null || true
fi

# Register bash completion
if [[ -n "$BASH_VERSION" ]]; then
    complete -F _homodyne_advanced_completion homodyne 2>/dev/null || true
    complete -F _homodyne_advanced_completion homodyne-gpu 2>/dev/null || true
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
    select m in "classical" "mcmc" "robust" "all" "skip"; do
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
    
    # GPU acceleration?
    if [[ "$(uname -s)" == "Linux" ]] && command -v homodyne-gpu >/dev/null 2>&1; then
        echo ""
        read -p "Use GPU acceleration? (y/N): " use_gpu
        [[ "$use_gpu" =~ ^[Yy] ]] && cmd="homodyne-gpu"
    fi
    
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