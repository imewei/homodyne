#!/bin/bash
# Advanced Homodyne Shell Completion System
# Provides intelligent completion with context awareness
# Supports NLSQ (primary) and CMC (secondary) optimization methods

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

    # Rebuild cache - find YAML files (primary config format)
    {
        find . -maxdepth 2 -name "*.yaml" -type f 2>/dev/null | head -20
        find . -maxdepth 2 -name "*.yml" -type f 2>/dev/null | head -10
        # Add commonly used config names
        echo "homodyne_config.yaml"
        echo "config.yaml"
        echo "analysis_config.yaml"
    } | sort -u > "$HOMODYNE_COMPLETION_CACHE_FILE"

    cat "$HOMODYNE_COMPLETION_CACHE_FILE"
}

# Smart method completion based on config mode
_homodyne_smart_method_completion() {
    local config_file="$1"
    local methods="nlsq cmc"

    # If config file exists, try to detect mode and suggest appropriate methods
    if [[ -f "$config_file" ]] && command -v python3 >/dev/null 2>&1; then
        local mode=$(python3 -c "
import yaml
try:
    with open('$config_file') as f:
        config = yaml.safe_load(f)
        mode = config.get('analysis_mode', '')
        if 'static' in mode:
            print('nlsq cmc')
        elif 'laminar' in mode:
            print('nlsq cmc')
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

    # Main options (from args_parser.py)
    local main_opts="--help --version --method --config --output-dir --data-file --verbose --quiet"

    # Analysis mode options (simplified in v2.4+)
    local mode_opts="--static-mode --laminar-flow"

    # NLSQ-specific options
    local nlsq_opts="--max-iterations --tolerance"

    # CMC-specific options
    local cmc_opts="--n-samples --n-warmup --n-chains --cmc-num-shards --cmc-backend --cmc-plot-diagnostics --dense-mass-matrix"

    # Parameter override options
    local override_opts="--initial-d0 --initial-alpha --initial-d-offset --initial-gamma-dot-t0 --initial-beta --initial-gamma-dot-offset --initial-phi0"

    # Output and plotting options
    local output_opts="--output-format --save-plots --plot-experimental-data --plot-simulated-data --plotting-backend --parallel-plots"
    local plot_param_opts="--contrast --offset --phi-angles"

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
            # Use cached recent configs (YAML files)
            local configs=$(_homodyne_get_recent_configs)
            COMPREPLY=($(compgen -W "$configs" -- "$cur"))
            # Also add file completion for YAML files
            COMPREPLY+=($(compgen -f -X '!*.yaml' -- "$cur"))
            COMPREPLY+=($(compgen -f -X '!*.yml' -- "$cur"))
            return 0
            ;;
        --output-dir)
            # Smart directory completion with suggestions
            local common_dirs="./results ./output ./homodyne_results ./analysis"
            COMPREPLY=($(compgen -W "$common_dirs" -- "$cur"))
            COMPREPLY+=($(compgen -d -- "$cur"))
            return 0
            ;;
        --data-file)
            # HDF5 data file completion
            COMPREPLY=($(compgen -f -X '!*.hdf' -- "$cur"))
            COMPREPLY+=($(compgen -f -X '!*.h5' -- "$cur"))
            COMPREPLY+=($(compgen -f -X '!*.hdf5' -- "$cur"))
            return 0
            ;;
        --phi-angles)
            # Common angle sets
            local angles="0,45,90,135 0,36,72,108,144 0,30,60,90,120,150"
            COMPREPLY=($(compgen -W "$angles" -- "$cur"))
            return 0
            ;;
        --contrast)
            # Common contrast values
            local values="0.1 0.2 0.3 0.4 0.5"
            COMPREPLY=($(compgen -W "$values" -- "$cur"))
            return 0
            ;;
        --offset)
            # Common offset values
            local values="0.9 0.95 1.0 1.05 1.1"
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
        --output-format)
            # Output format options
            local formats="yaml json npz"
            COMPREPLY=($(compgen -W "$formats" -- "$cur"))
            return 0
            ;;
        --plotting-backend)
            # Plotting backend options
            local backends="auto matplotlib datashader"
            COMPREPLY=($(compgen -W "$backends" -- "$cur"))
            return 0
            ;;
        --max-iterations)
            # Common iteration counts
            local counts="1000 5000 10000 50000"
            COMPREPLY=($(compgen -W "$counts" -- "$cur"))
            return 0
            ;;
        --tolerance)
            # Common tolerance values
            local values="1e-6 1e-7 1e-8 1e-9 1e-10"
            COMPREPLY=($(compgen -W "$values" -- "$cur"))
            return 0
            ;;
        --n-samples)
            # Common MCMC sample counts
            local counts="500 1000 2000 5000 10000"
            COMPREPLY=($(compgen -W "$counts" -- "$cur"))
            return 0
            ;;
        --n-warmup)
            # Common warmup counts
            local counts="250 500 1000 2000"
            COMPREPLY=($(compgen -W "$counts" -- "$cur"))
            return 0
            ;;
        --n-chains)
            # Common chain counts
            local counts="2 4 6 8"
            COMPREPLY=($(compgen -W "$counts" -- "$cur"))
            return 0
            ;;
        --initial-d0)
            # Common D0 values (nm^2/s)
            local values="100 500 1000 5000 10000"
            COMPREPLY=($(compgen -W "$values" -- "$cur"))
            return 0
            ;;
        --initial-alpha)
            # Common alpha values
            local values="-2.0 -1.5 -1.0 -0.5 0.0"
            COMPREPLY=($(compgen -W "$values" -- "$cur"))
            return 0
            ;;
        --initial-d-offset)
            # Common D_offset values
            local values="0 100 500 1000"
            COMPREPLY=($(compgen -W "$values" -- "$cur"))
            return 0
            ;;
        --initial-gamma-dot-t0)
            # Common gamma_dot values (s^-1)
            local values="0.001 0.01 0.1 1.0"
            COMPREPLY=($(compgen -W "$values" -- "$cur"))
            return 0
            ;;
        --initial-beta)
            # Common beta values
            local values="-2.0 -1.0 0.0 1.0"
            COMPREPLY=($(compgen -W "$values" -- "$cur"))
            return 0
            ;;
        --initial-gamma-dot-offset)
            # Common gamma_dot_offset values
            local values="0 0.001 0.01"
            COMPREPLY=($(compgen -W "$values" -- "$cur"))
            return 0
            ;;
        --initial-phi0)
            # Common phi0 values (radians)
            local values="0.0 0.785 1.571 2.356 3.142"
            COMPREPLY=($(compgen -W "$values" -- "$cur"))
            return 0
            ;;
    esac

    # Check for incompatible options
    local has_mode=false
    for word in "${COMP_WORDS[@]}"; do
        if [[ "$word" =~ ^--(static-mode|laminar-flow)$ ]]; then
            has_mode=true
            break
        fi
    done

    # Check current method for context-aware completions
    local current_method=""
    for ((i=1; i<COMP_CWORD; i++)); do
        if [[ "${COMP_WORDS[i]}" == "--method" ]] && [[ $((i+1)) -lt ${#COMP_WORDS[@]} ]]; then
            current_method="${COMP_WORDS[i+1]}"
            break
        fi
    done

    if [[ $cur == -* ]]; then
        local all_opts="$main_opts $output_opts $plot_param_opts $nlsq_opts"
        [[ "$has_mode" == false ]] && all_opts="$all_opts $mode_opts"

        # Add CMC options if method is cmc
        if [[ "$current_method" == "cmc" ]]; then
            all_opts="$all_opts $cmc_opts $override_opts"
        else
            # Always show override options (useful for any method)
            all_opts="$all_opts $override_opts"
        fi

        COMPREPLY=($(compgen -W "$all_opts" -- "$cur"))
    else
        # Default to config files (YAML)
        COMPREPLY=($(compgen -f -X '!*.yaml' -- "$cur"))
        COMPREPLY+=($(compgen -f -X '!*.yml' -- "$cur"))
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
            # Updated modes: static and laminar_flow
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
            '--version[Show version information]'
            '--method[Optimization method]:method:(nlsq cmc)'
            '--config[Configuration file (YAML)]:file:_files -g "*.yaml" -g "*.yml"'
            '--output-dir[Output directory]:dir:_directories'
            '--data-file[Experimental data file (overrides config)]:file:_files -g "*.hdf" -g "*.h5" -g "*.hdf5"'
            '--output-format[Output format]:format:(yaml json npz)'
            '--verbose[Enable verbose logging]'
            '--quiet[Suppress all output except errors]'
            # Analysis modes
            '(--static-mode --laminar-flow)--static-mode[Force static analysis mode (3 parameters)]'
            '(--static-mode --laminar-flow)--laminar-flow[Force laminar flow mode (7 parameters)]'
            # NLSQ options
            '--max-iterations[Maximum NLSQ iterations]:iterations:'
            '--tolerance[NLSQ convergence tolerance]:tolerance:'
            # CMC options
            '--n-samples[CMC samples per chain]:samples:'
            '--n-warmup[CMC warmup samples]:warmup:'
            '--n-chains[Number of CMC chains]:chains:'
            '--cmc-num-shards[Data shards for CMC]:shards:(4 8 10 16 20 32)'
            '--cmc-backend[CMC parallel backend]:backend:(auto pjit multiprocessing pbs)'
            '--cmc-plot-diagnostics[DEPRECATED - ArviZ plots always generated]'
            '--dense-mass-matrix[Use dense mass matrix for NUTS/CMC]'
            # Parameter overrides
            '--initial-d0[Override initial D0 (nm^2/s)]:value:'
            '--initial-alpha[Override initial alpha]:value:'
            '--initial-d-offset[Override initial D_offset (nm^2/s)]:value:'
            '--initial-gamma-dot-t0[Override gamma_dot_t0 (s^-1, laminar flow)]:value:'
            '--initial-beta[Override beta (laminar flow)]:value:'
            '--initial-gamma-dot-offset[Override gamma_dot_offset (s^-1, laminar flow)]:value:'
            '--initial-phi0[Override phi0 (radians, laminar flow)]:value:'
            # Output and plotting
            '--save-plots[Save result plots to output directory]'
            '--plot-experimental-data[Generate data validation plots]'
            '--plot-simulated-data[Plot theoretical C2 heatmaps]'
            '--plotting-backend[Plotting backend]:backend:(auto matplotlib datashader)'
            '--parallel-plots[Generate plots in parallel]'
            '--contrast[Contrast parameter for simulated data]:contrast:'
            '--offset[Offset parameter for simulated data]:offset:'
            '--phi-angles[Phi angles (comma-separated degrees)]:angles:'
            '*:config:_files -g "*.yaml" -g "*.yml"'
        )

        _arguments -C $args
    }

    # Zsh completion for homodyne-config
    _homodyne_config_zsh() {
        local -a args
        args=(
            '(--help -h)'{--help,-h}'[Show help message]'
            '(--mode -m)'{--mode,-m}'[Configuration mode]:mode:(static laminar_flow)'
            '(--output -o)'{--output,-o}'[Output configuration file]:file:_files -g "*.yaml" -g "*.yml"'
            '(--interactive -i)'{--interactive,-i}'[Interactive configuration builder]'
            '(--validate -v)'{--validate,-v}'[Validate configuration file]:file:_files -g "*.yaml" -g "*.yml"'
            '(--force -f)'{--force,-f}'[Force overwrite existing file]'
            '*:output:_files -g "*.yaml" -g "*.yml"'
        )

        _arguments -C $args
    }

    # Register completion for all homodyne commands
    compdef _homodyne_advanced_zsh homodyne 2>/dev/null || true
    compdef _homodyne_config_zsh homodyne-config 2>/dev/null || true

    # Register completion for method aliases (hm- prefix)
    compdef _homodyne_advanced_zsh hm 2>/dev/null || true         # homodyne
    compdef _homodyne_advanced_zsh hm-nlsq 2>/dev/null || true    # homodyne --method nlsq
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
    alias hm-cmc='homodyne --method cmc'           # Consensus Monte Carlo (CMC)

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

    echo "Homodyne Command Builder"
    echo "========================"
    echo ""

    # Select method
    PS3="Select analysis method: "
    select m in "nlsq (primary)" "cmc (uncertainty)" "skip"; do
        case $REPLY in
            1) method="--method nlsq"; break;;
            2) method="--method cmc"; break;;
            3) break;;
            *) echo "Invalid selection";;
        esac
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

    # Build and show command
    echo ""
    echo "Generated command:"
    echo "  $cmd $method $config"
    echo ""
    read -p "Run this command? (y/N): " run_it

    if [[ "$run_it" =~ ^[Yy] ]]; then
        echo "Running..."
        eval "$cmd $method $config"
    else
        echo "Command copied to clipboard (if available)"
        echo "$cmd $method $config" | xclip -selection clipboard 2>/dev/null || \
        echo "$cmd $method $config" | pbcopy 2>/dev/null || \
        echo "Copy command: $cmd $method $config"
    fi
}

# Help function showing all aliases and shortcuts
homodyne_help() {
    echo "Homodyne Shell Shortcuts"
    echo "========================"
    echo ""
    echo "Base Commands:"
    echo "  hm                    homodyne"
    echo "  hconfig               homodyne-config"
    echo ""
    echo "Method Aliases (hm- prefix):"
    echo "  hm-nlsq               homodyne --method nlsq (primary)"
    echo "  hm-cmc                homodyne --method cmc (uncertainty)"
    echo ""
    echo "Config Aliases (hc- prefix):"
    echo "  hc-stat               homodyne-config --mode static"
    echo "  hc-flow               homodyne-config --mode laminar_flow"
    echo ""
    echo "Utility Aliases:"
    echo "  hexp                  homodyne --plot-experimental-data"
    echo "  hsim                  homodyne --plot-simulated-data"
    echo ""
    echo "Interactive:"
    echo "  homodyne_build        Interactive command builder"
    echo "  homodyne_help         Show this help"
    echo ""
    echo "Tab Completion:"
    echo "  All commands support intelligent tab completion."
    echo "  Try: hm --<TAB> or hm --method <TAB>"
}
