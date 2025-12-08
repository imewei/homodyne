#!/usr/bin/env fish
# homodyne/runtime/shell/activation/xla_config.fish
# XLA_FLAGS configuration for Homodyne (Fish shell)

# ============================================================================
# CRITICAL: Respect user's existing XLA_FLAGS
# ============================================================================
if set -q XLA_FLAGS
    if set -q HOMODYNE_VERBOSE
        echo "[homodyne] Using existing XLA_FLAGS: $XLA_FLAGS" >&2
    end
    exit 0
end

# ============================================================================
# Auto-Detection Function
# ============================================================================
function _homodyne_detect_xla_devices
    set -l cores 4

    if command -v nproc >/dev/null 2>&1
        set cores (nproc 2>/dev/null; or echo 4)
    else if command -v sysctl >/dev/null 2>&1
        set cores (sysctl -n hw.ncpu 2>/dev/null; or echo 4)
    else if test -f /proc/cpuinfo
        set cores (grep -c ^processor /proc/cpuinfo 2>/dev/null; or echo 4)
    end

    if test $cores -le 7
        echo 2
    else if test $cores -le 15
        echo 4
    else if test $cores -le 35
        echo 6
    else
        echo 8
    end
end

# ============================================================================
# Read User Preference
# ============================================================================
set -l mode $HOMODYNE_XLA_MODE

# If not set, read from config file
if test -z "$mode"; and test -f $HOME/.homodyne_xla_mode
    set mode (cat $HOME/.homodyne_xla_mode 2>/dev/null | tr -d '[:space:]'; or echo "cmc")
end

# Validate and default
switch $mode
    case cmc cmc-hpc nlsq auto
        # Valid mode
    case ''
        set mode cmc
    case '*'
        if set -q HOMODYNE_VERBOSE
            echo "[homodyne] Warning: Invalid mode '$mode', using cmc" >&2
        end
        set mode cmc
end

# ============================================================================
# Set XLA_FLAGS Based on Mode
# ============================================================================
switch $mode
    case cmc
        set -gx XLA_FLAGS "--xla_force_host_platform_device_count=4"
        if set -q HOMODYNE_VERBOSE
            echo "[homodyne] XLA: cmc mode (4 devices)" >&2
        end

    case cmc-hpc
        set -gx XLA_FLAGS "--xla_force_host_platform_device_count=8"
        if set -q HOMODYNE_VERBOSE
            echo "[homodyne] XLA: cmc-hpc mode (8 devices)" >&2
        end

    case nlsq
        set -gx XLA_FLAGS "--xla_force_host_platform_device_count=1"
        if set -q HOMODYNE_VERBOSE
            echo "[homodyne] XLA: nlsq mode (1 device)" >&2
        end

    case auto
        set -l devices (_homodyne_detect_xla_devices)
        set -gx XLA_FLAGS "--xla_force_host_platform_device_count=$devices"

        set -l cores 4
        if command -v nproc >/dev/null 2>&1
            set cores (nproc 2>/dev/null; or echo 4)
        else if command -v sysctl >/dev/null 2>&1
            set cores (sysctl -n hw.ncpu 2>/dev/null; or echo 4)
        end
        echo "[homodyne] XLA: auto mode → $devices devices (detected $cores CPU cores)" >&2
end

# Export mode
set -gx HOMODYNE_XLA_MODE $mode

# Cleanup
functions -e _homodyne_detect_xla_devices
