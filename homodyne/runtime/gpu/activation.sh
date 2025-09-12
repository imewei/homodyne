#!/bin/bash
# Smart GPU Activation Script with Auto-Optimization
# Detects hardware and applies optimal settings automatically

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# GPU activation with smart detection
homodyne_gpu_activate_smart() {
    # Check if already activated
    if [[ "$HOMODYNE_GPU_ACTIVATED" == "1" ]]; then
        echo -e "${GREEN}✅ GPU already activated${NC}"
        return 0
    fi

    # Platform check
    if [[ "$(uname -s)" != "Linux" ]]; then
        echo -e "${YELLOW}⚠️  GPU acceleration only available on Linux${NC}"
        return 1
    fi

    # Find CUDA installation
    local cuda_found=false
    local cuda_paths=("/usr/local/cuda" "/opt/cuda" "$HOME/cuda")

    for cuda_path in "${cuda_paths[@]}"; do
        if [[ -d "$cuda_path" ]]; then
            export CUDA_ROOT="$cuda_path"
            export CUDA_HOME="$cuda_path"
            cuda_found=true
            break
        fi
    done

    if [[ "$cuda_found" == false ]]; then
        echo -e "${RED}❌ CUDA not found. GPU acceleration unavailable${NC}"
        echo "   Install CUDA from: https://developer.nvidia.com/cuda-downloads"
        return 1
    fi

    # Set basic CUDA paths
    export PATH="$CUDA_ROOT/bin:$PATH"

    # Add CUDA libraries to LD_LIBRARY_PATH if not already present
    if [[ ":$LD_LIBRARY_PATH:" != *":$CUDA_ROOT/lib64:"* ]]; then
        export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:${LD_LIBRARY_PATH}"
    fi

    # Check for NVIDIA GPU
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  nvidia-smi not found. GPU may not be available${NC}"
    else
        # Get GPU info
        local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        local gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)

        if [[ -n "$gpu_name" ]]; then
            echo -e "${GREEN}🎮 GPU Detected: $gpu_name ($gpu_memory)${NC}"
        fi
    fi

    # Smart XLA flags based on available memory
    local xla_flags="--xla_gpu_cuda_data_dir=$CUDA_ROOT"

    # Check available GPU memory and set memory fraction
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_memory_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)

        if [[ -n "$gpu_memory_mb" ]]; then
            if [[ $gpu_memory_mb -lt 4096 ]]; then
                export XLA_PYTHON_CLIENT_MEM_FRACTION="0.7"
                echo -e "${YELLOW}   Memory: Low (<4GB) - Using 70% allocation${NC}"
            elif [[ $gpu_memory_mb -lt 8192 ]]; then
                export XLA_PYTHON_CLIENT_MEM_FRACTION="0.8"
                echo -e "${BLUE}   Memory: Medium (4-8GB) - Using 80% allocation${NC}"
            else
                export XLA_PYTHON_CLIENT_MEM_FRACTION="0.9"
                echo -e "${GREEN}   Memory: High (>8GB) - Using 90% allocation${NC}"
            fi
        fi

        # Enable modern GPU optimizations (only use supported flags)
        xla_flags="$xla_flags --xla_gpu_triton_gemm_any=true"
        xla_flags="$xla_flags --xla_gpu_enable_latency_hiding_scheduler=true"
    fi

    export XLA_FLAGS="$xla_flags"

    # JAX settings for NumPyro GPU compatibility
    unset JAX_PLATFORMS  # Let JAX auto-detect platforms
    export JAX_ENABLE_X64="0"  # Use float32 for better GPU performance
    export XLA_PYTHON_CLIENT_PREALLOCATE="false"  # Dynamic memory allocation

    # Set memory fraction if not already configured by smart detection above
    if [[ -z "$XLA_PYTHON_CLIENT_MEM_FRACTION" ]]; then
        export XLA_PYTHON_CLIENT_MEM_FRACTION="0.8"  # Default 80% allocation
    fi

    # NumPyro GPU routing (NOTE: HOMODYNE_GPU_INTENT should only be set by homodyne-gpu wrapper, not permanently)
    # Do not export HOMODYNE_GPU_INTENT here - it's set by homodyne-gpu command wrapper only

    # PyTensor settings (CPU mode to avoid conflicts)
    export PYTENSOR_FLAGS="device=cpu,floatX=float64,on_opt_error=ignore"

    # Mark as activated
    export HOMODYNE_GPU_ACTIVATED="1"

    echo -e "${GREEN}✅ GPU environment activated successfully${NC}"

    # Show optimization tips
    if [[ -z "$HOMODYNE_GPU_TIPS_SHOWN" ]]; then
        export HOMODYNE_GPU_TIPS_SHOWN="1"
        echo ""
        echo -e "${BLUE}💡 GPU Optimization Tips:${NC}"
        echo "   • Use larger batch sizes for better GPU utilization"
        echo "   • Run 'homodyne-gpu-optimize' for hardware-specific tuning"
        echo "   • Monitor GPU usage with 'nvidia-smi -l 1'"
    fi

    return 0
}

# Enhanced GPU status function
homodyne_gpu_status_detailed() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}🚀 Homodyne GPU Status${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Activation status
    if [[ "$HOMODYNE_GPU_ACTIVATED" == "1" ]]; then
        echo -e "${GREEN}✅ GPU Environment: Activated${NC}"
    else
        echo -e "${RED}❌ GPU Environment: Not Activated${NC}"
    fi

    # CUDA status
    if [[ -n "$CUDA_HOME" ]]; then
        echo -e "${GREEN}✅ CUDA Path: $CUDA_HOME${NC}"

        # Check CUDA version
        if command -v nvcc >/dev/null 2>&1; then
            local cuda_version=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
            echo -e "${GREEN}   CUDA Version: $cuda_version${NC}"
        fi
    else
        echo -e "${RED}❌ CUDA: Not configured${NC}"
    fi

    # GPU hardware
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo ""
        echo -e "${BLUE}📊 GPU Hardware:${NC}"
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
                  --format=csv,noheader | while IFS=',' read -r idx name mem_total mem_used util; do
            echo -e "   GPU $idx: $name"
            echo -e "      Memory: $mem_used / $mem_total"
            echo -e "      Utilization:$util"
        done
    else
        echo -e "${YELLOW}⚠️  nvidia-smi not available${NC}"
    fi

    # JAX devices
    echo ""
    echo -e "${BLUE}🔧 JAX Configuration:${NC}"

    # JAX availability check
    if python3 -c "import jax; print('JAX_CHECK_OK')" >/dev/null 2>&1; then
        python3 -c "
import jax
devices = jax.devices()
print('   Devices: ' + str(devices))

gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
if gpu_devices:
    print('   \033[0;32m✅ GPU Available: ' + str(len(gpu_devices)) + ' device(s)\033[0m')
    print('   \033[0;32m✅ GPU Computation: Working\033[0m')
else:
    print('   \033[1;33m⚠️  No GPU devices found by JAX\033[0m')
    print('   Try: pip install jax[cuda12-local]')
"
    else
        echo -e "   ${RED}❌ JAX not available${NC}"
        echo "   Install with: pip install jax[cuda12-local]"
    fi

    # Environment variables
    echo ""
    echo -e "${BLUE}🔧 Environment Variables:${NC}"
    [[ -n "$XLA_FLAGS" ]] && echo "   XLA_FLAGS: $(echo $XLA_FLAGS | cut -c1-50)..."
    [[ -n "$XLA_PYTHON_CLIENT_MEM_FRACTION" ]] && echo "   Memory Fraction: $XLA_PYTHON_CLIENT_MEM_FRACTION"
    [[ -n "$JAX_PLATFORMS" ]] && echo "   JAX_PLATFORMS: $JAX_PLATFORMS" || echo "   JAX_PLATFORMS: (auto-detect)"
    [[ -n "$HOMODYNE_GPU_INTENT" ]] && echo "   GPU Intent: $HOMODYNE_GPU_INTENT" || echo "   GPU Intent: (not set)"

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Quick GPU benchmark
homodyne_gpu_benchmark() {
    echo -e "${BLUE}⏱️  Running GPU Benchmark...${NC}"

    if [[ "$HOMODYNE_GPU_ACTIVATED" != "1" ]]; then
        echo -e "${YELLOW}Activating GPU first...${NC}"
        homodyne_gpu_activate_smart
    fi

    python3 -c "
import time
import sys

print('\\n📊 Matrix Multiplication Benchmark:')
print('-' * 40)

try:
    import jax
    import jax.numpy as jnp
    from jax import jit

    # Check for GPU
    devices = jax.devices()
    gpu_available = any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices)

    if not gpu_available:
        print('❌ No GPU available for benchmarking')
        sys.exit(1)

    sizes = [100, 500, 1000, 2000]

    for size in sizes:
        # Create random matrices
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (size, size))

        # JIT compile
        @jit
        def matmul(a, b):
            return jnp.dot(a, b)

        # Warmup
        _ = matmul(x, x).block_until_ready()

        # Benchmark
        start = time.perf_counter()
        for _ in range(10):
            _ = matmul(x, x).block_until_ready()
        elapsed = (time.perf_counter() - start) / 10

        gflops = (2 * size**3) / (elapsed * 1e9)
        print(f'   {size:4d}x{size:4d}: {elapsed*1000:6.2f}ms ({gflops:6.1f} GFLOPS)')

    print('\\n✅ GPU Benchmark Complete')

except ImportError:
    print('❌ JAX not installed')
    print('   Install with: pip install jax[cuda12-local]')
except Exception as e:
    print(f'❌ Benchmark failed: {e}')
" 2>/dev/null || echo -e "${RED}❌ Benchmark failed${NC}"
}

# Aliases for quick access
alias gpu-on='homodyne_gpu_activate_smart'
alias gpu-status='homodyne_gpu_status_detailed'
alias gpu-bench='homodyne_gpu_benchmark'
alias homodyne_gpu_status='homodyne_gpu_status_detailed'

# Auto-activate if in conda environment
if [[ -n "$CONDA_DEFAULT_ENV" ]] || [[ -n "$VIRTUAL_ENV" ]]; then
    homodyne_gpu_activate_smart >/dev/null 2>&1
fi

# Export functions (only in bash)
if [[ -n "$BASH_VERSION" ]]; then
    export -f homodyne_gpu_activate_smart
    export -f homodyne_gpu_status_detailed
    export -f homodyne_gpu_benchmark
fi
