#!/bin/bash
# homodyne/runtime/shell/activation/xla_config.bash
# XLA_FLAGS configuration for Homodyne CMC/NLSQ optimization
# Automatically sourced when virtual environment is activated

# ============================================================================
# CRITICAL: Respect user's existing XLA_FLAGS
# ============================================================================
if [ -n "$XLA_FLAGS" ]; then
    # User has manually set XLA_FLAGS before activation
    # Respect their choice and skip automatic configuration
    if [ -n "$HOMODYNE_VERBOSE" ]; then
        echo "[homodyne] Using existing XLA_FLAGS: $XLA_FLAGS" >&2
    fi
    return 0
fi

# ============================================================================
# Auto-Detection Function
# ============================================================================
_homodyne_detect_xla_devices() {
    local cores=4

    # Try multiple methods to detect CPU cores (cross-platform)
    if command -v nproc >/dev/null 2>&1; then
        # Linux
        cores=$(nproc 2>/dev/null || echo 4)
    elif command -v sysctl >/dev/null 2>&1; then
        # macOS/BSD
        cores=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
    elif [ -f /proc/cpuinfo ]; then
        # Linux fallback
        cores=$(grep -c ^processor /proc/cpuinfo 2>/dev/null || echo 4)
    fi

    # Intelligent device count based on core count
    # Reserve 2+ cores for OS and other processes
    if [ "$cores" -le 7 ]; then
        echo 2   # Small workstations: 4-7 cores → 2 devices
    elif [ "$cores" -le 15 ]; then
        echo 4   # Medium workstations: 8-15 cores → 4 devices
    elif [ "$cores" -le 35 ]; then
        echo 6   # Large workstations: 16-35 cores → 6 devices
    else
        echo 8   # HPC nodes: 36+ cores → 8 devices
    fi
}

# ============================================================================
# Read User Preference
# ============================================================================
HOMODYNE_XLA_MODE="${HOMODYNE_XLA_MODE:-}"

# If not set via environment, read from config file
if [ -z "$HOMODYNE_XLA_MODE" ] && [ -f "$HOME/.homodyne_xla_mode" ]; then
    HOMODYNE_XLA_MODE=$(cat "$HOME/.homodyne_xla_mode" 2>/dev/null | tr -d '[:space:]')
fi

# Validate mode from config file
case "$HOMODYNE_XLA_MODE" in
    cmc|cmc-hpc|nlsq|auto)
        # Valid mode, proceed
        ;;
    "")
        # No mode set, use default
        HOMODYNE_XLA_MODE="cmc"
        ;;
    *)
        # Invalid mode in file, warn and use default
        if [ -n "$HOMODYNE_VERBOSE" ]; then
            echo "[homodyne] Warning: Invalid mode '$HOMODYNE_XLA_MODE' in ~/.homodyne_xla_mode" >&2
            echo "[homodyne] Using default mode: cmc" >&2
        fi
        HOMODYNE_XLA_MODE="cmc"
        ;;
esac

# ============================================================================
# Set XLA_FLAGS Based on Mode
# ============================================================================
case "$HOMODYNE_XLA_MODE" in
    cmc)
        export XLA_FLAGS="--xla_force_host_platform_device_count=4"
        if [ -n "$HOMODYNE_VERBOSE" ]; then
            echo "[homodyne] XLA: cmc mode (4 devices for parallel CMC chains)" >&2
        fi
        ;;

    cmc-hpc)
        export XLA_FLAGS="--xla_force_host_platform_device_count=8"
        if [ -n "$HOMODYNE_VERBOSE" ]; then
            echo "[homodyne] XLA: cmc-hpc mode (8 devices for HPC clusters)" >&2
        fi
        ;;

    nlsq)
        export XLA_FLAGS="--xla_force_host_platform_device_count=1"
        if [ -n "$HOMODYNE_VERBOSE" ]; then
            echo "[homodyne] XLA: nlsq mode (1 device, NLSQ doesn't need parallelism)" >&2
        fi
        ;;

    auto)
        _devices=$(_homodyne_detect_xla_devices)
        export XLA_FLAGS="--xla_force_host_platform_device_count=$_devices"

        # Always show auto-detection result (informative)
        _cores=4
        if command -v nproc >/dev/null 2>&1; then
            _cores=$(nproc 2>/dev/null || echo 4)
        elif command -v sysctl >/dev/null 2>&1; then
            _cores=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
        fi
        echo "[homodyne] XLA: auto mode → $_devices devices (detected $_cores CPU cores)" >&2
        unset _cores _devices
        ;;
esac

# ============================================================================
# Cleanup
# ============================================================================
unset -f _homodyne_detect_xla_devices

# Export mode for introspection (useful for debugging)
export HOMODYNE_XLA_MODE
