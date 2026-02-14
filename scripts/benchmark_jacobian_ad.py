#!/usr/bin/env python3
"""Benchmark: jacfwd vs jacrev for XPCS Jacobian computation.

Measures performance of JAX automatic differentiation modes for the typical
XPCS regime where m (residuals) >> n (parameters).

AD Theory:
- jacfwd (JVP-based): O(n × cost_f) — one forward pass per input dimension
- jacrev (VJP-based): O(m × cost_f) — one reverse pass per output dimension

For XPCS: m ~ 5K-5M, n ~ 3-9, so jacfwd should be O(9 × cost_f) vs
jacrev's O(5M × cost_f). But jacrev can be faster when the function
involves operations that are more efficient in reverse mode.

Usage:
    uv run python scripts/benchmark_jacobian_ad.py
"""

from __future__ import annotations

import gc
import time
import tracemalloc

import jax
import jax.numpy as jnp
import numpy as np


def create_xpcs_residual_fn(
    m: int, n: int
) -> tuple[callable, jnp.ndarray, jnp.ndarray]:
    """Create a synthetic XPCS-like residual function.

    Mimics the structure of homodyne's model evaluation:
    - Exponential decay (diffusion)
    - Sinc-squared modulation (shear, if n >= 7)
    - Per-angle contrast/offset scaling

    Parameters
    ----------
    m : int
        Number of residual points (data points).
    n : int
        Number of parameters (3 for static, 9 for laminar_flow).

    Returns
    -------
    tuple
        (residual_fn, params, y_target) where residual_fn(p) -> residuals.
    """
    rng = np.random.default_rng(42)

    # Synthetic time delays and angles
    t1 = jnp.array(rng.uniform(0.01, 10.0, size=m))
    t2 = jnp.array(rng.uniform(0.01, 10.0, size=m))
    phi = jnp.array(rng.uniform(0, 2 * np.pi, size=m))
    q = 0.01  # typical q value

    if n == 3:
        # Static mode: [D0, alpha, D_offset]
        true_params = jnp.array([1000.0, 0.5, 10.0])
    elif n == 9:
        # Laminar flow with averaged scaling: [contrast, offset, D0, alpha, D_offset, gamma0, beta, gamma_offset, phi0]
        true_params = jnp.array([0.3, 1.0, 1000.0, 0.5, 10.0, 1e-3, 0.5, 0.0, 0.0])
    else:
        true_params = jnp.array(rng.uniform(0.1, 10.0, size=n))

    # Generate synthetic target data
    def model_fn(params: jnp.ndarray) -> jnp.ndarray:
        """Simplified XPCS model mimicking compute_g1 structure."""
        if n >= 7:
            contrast = params[0]
            offset = params[1]
            D0 = params[2]
            alpha = params[3]
            D_offset = params[4]
            gamma0 = params[5]
            phi0 = params[7] if n > 7 else 0.0
        else:
            contrast = 0.3
            offset = 1.0
            D0 = params[0]
            alpha = params[1]
            D_offset = params[2]

        # Diffusion: exp(-q^2 * D(t) * |dt|)
        dt = jnp.abs(t2 - t1)
        D_eff = D0 * jnp.power(dt + 1e-10, alpha) + D_offset
        g1_diff = jnp.exp(-q**2 * D_eff * dt)

        if n >= 7:
            # Shear: sinc^2 modulation
            phase = q * gamma0 * jnp.cos(phi0 - phi) * dt
            g1_shear = jnp.where(
                jnp.abs(phase) > 1e-12,
                (jnp.sin(phase) / phase) ** 2,
                1.0,
            )
            g1 = g1_diff * g1_shear
            g2 = offset + contrast * g1**2
        else:
            g2 = offset + contrast * g1_diff**2

        return g2

    # Generate target with noise
    y_target = model_fn(true_params) + jnp.array(rng.normal(0, 0.01, size=m))
    sigma = jnp.ones(m) * 0.01

    def residual_fn(params: jnp.ndarray) -> jnp.ndarray:
        """Weighted residuals: (model - data) / sigma."""
        return (model_fn(params) - y_target) / sigma

    return residual_fn, true_params, y_target


def benchmark_ad_mode(
    mode: str,
    residual_fn: callable,
    params: jnp.ndarray,
    n_warmup: int = 2,
    n_trials: int = 5,
) -> dict:
    """Benchmark a single AD mode.

    Parameters
    ----------
    mode : str
        'jacfwd' or 'jacrev'.
    residual_fn : callable
        Function to differentiate.
    params : jnp.ndarray
        Parameter point.
    n_warmup : int
        JIT warmup iterations.
    n_trials : int
        Timed iterations.

    Returns
    -------
    dict
        Timing and memory results.
    """
    jac_fn = jax.jacfwd if mode == "jacfwd" else jax.jacrev

    # JIT-cold: first call includes tracing + compilation
    gc.collect()
    t0 = time.perf_counter()
    J_cold = jac_fn(residual_fn)(params)
    J_cold.block_until_ready()
    cold_time = time.perf_counter() - t0

    # Warmup (JIT-warm)
    for _ in range(n_warmup):
        J = jac_fn(residual_fn)(params)
        J.block_until_ready()

    # Timed runs
    gc.collect()
    tracemalloc.start()
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        J = jac_fn(residual_fn)(params)
        J.block_until_ready()
        times.append(time.perf_counter() - t0)

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "mode": mode,
        "cold_time_s": cold_time,
        "warm_mean_s": np.mean(times),
        "warm_std_s": np.std(times),
        "warm_min_s": np.min(times),
        "warm_max_s": np.max(times),
        "peak_memory_mb": peak_mem / 1024 / 1024,
        "jacobian": np.asarray(J),
    }


def run_benchmark(m: int, n: int) -> dict:
    """Run jacfwd vs jacrev benchmark for given dimensions.

    Parameters
    ----------
    m : int
        Number of residual points.
    n : int
        Number of parameters.

    Returns
    -------
    dict
        Results for both modes.
    """
    print(f"\n{'='*70}")
    print(f"  Benchmark: m={m:,} residuals, n={n} params (ratio={m/n:.0f}:1)")
    print(f"{'='*70}")

    residual_fn, params, _ = create_xpcs_residual_fn(m, n)

    # Perturb params slightly from true values (realistic optimization scenario)
    rng = np.random.default_rng(123)
    params = params * (1.0 + jnp.array(rng.normal(0, 0.1, size=n)))

    results = {}
    for mode in ["jacfwd", "jacrev"]:
        print(f"\n  {mode}...")
        try:
            res = benchmark_ad_mode(mode, residual_fn, params)
            results[mode] = res
            print(f"    JIT-cold:  {res['cold_time_s']:.4f}s")
            print(f"    JIT-warm:  {res['warm_mean_s']:.4f}s +/- {res['warm_std_s']:.4f}s")
            print(f"    Peak mem:  {res['peak_memory_mb']:.1f} MB")
        except Exception as e:
            print(f"    FAILED: {e}")
            results[mode] = {"error": str(e)}

    # Numerical equivalence check
    if "jacobian" in results.get("jacfwd", {}) and "jacobian" in results.get("jacrev", {}):
        J_fwd = results["jacfwd"]["jacobian"]
        J_rev = results["jacrev"]["jacobian"]
        try:
            max_diff = np.max(np.abs(J_fwd - J_rev))
            rel_diff = np.max(np.abs(J_fwd - J_rev) / (np.abs(J_rev) + 1e-30))
            is_close = np.allclose(J_fwd, J_rev, rtol=1e-6)
            print(f"\n  Numerical equivalence:")
            print(f"    Max abs diff:  {max_diff:.2e}")
            print(f"    Max rel diff:  {rel_diff:.2e}")
            print(f"    np.allclose(rtol=1e-6): {is_close}")
        except Exception as e:
            print(f"    Equivalence check failed: {e}")

    # Speed comparison
    if all("warm_mean_s" in results.get(m_, {}) for m_ in ["jacfwd", "jacrev"]):
        fwd_t = results["jacfwd"]["warm_mean_s"]
        rev_t = results["jacrev"]["warm_mean_s"]
        speedup = rev_t / fwd_t if fwd_t > 0 else float("inf")
        winner = "jacfwd" if fwd_t < rev_t else "jacrev"
        print(f"\n  Winner: {winner} ({speedup:.2f}x {'faster' if winner == 'jacfwd' else 'slower'} than jacfwd)")

    return results


def main() -> None:
    """Run full benchmark suite."""
    print("Jacobian AD Mode Benchmark for XPCS")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Platform: {jax.default_backend()}")

    # Test configurations: (m, n) pairs
    # Note: Dense Jacobian computation is memory-intensive — JAX needs to store
    # the full AD trace. 50K+ causes multi-GB usage. The trend from smaller
    # sizes (up to 20K) is sufficient to determine the winner for m >> n.
    configs = [
        # Static mode (n=3)
        (500, 3),
        (1_000, 3),
        (5_000, 3),
        (10_000, 3),
        (20_000, 3),
        # Laminar flow mode (n=9)
        (500, 9),
        (1_000, 9),
        (5_000, 9),
        (10_000, 9),
        (20_000, 9),
    ]

    all_results = {}
    summary = []

    for m, n in configs:
        try:
            results = run_benchmark(m, n)
            all_results[(m, n)] = results

            if all("warm_mean_s" in results.get(mode, {}) for mode in ["jacfwd", "jacrev"]):
                fwd_t = results["jacfwd"]["warm_mean_s"]
                rev_t = results["jacrev"]["warm_mean_s"]
                winner = "jacfwd" if fwd_t < rev_t else "jacrev"
                speedup = max(fwd_t, rev_t) / min(fwd_t, rev_t)
                summary.append((m, n, winner, speedup, fwd_t, rev_t))
        except Exception as e:
            print(f"\n  BENCHMARK FAILED for m={m}, n={n}: {e}")

    # Summary table
    print(f"\n\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'m':>10s}  {'n':>4s}  {'Winner':>8s}  {'Speedup':>8s}  {'jacfwd':>10s}  {'jacrev':>10s}")
    print(f"  {'-'*10}  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}")
    for m, n, winner, speedup, fwd_t, rev_t in summary:
        print(f"  {m:>10,}  {n:>4}  {winner:>8s}  {speedup:>7.2f}x  {fwd_t:>9.4f}s  {rev_t:>9.4f}s")

    # Decision
    print(f"\n  DECISION GATE:")
    if summary:
        # Count wins for typical XPCS regime (m >= 50K)
        typical = [(m, n, w, s) for m, n, w, s, _, _ in summary if m >= 50_000]
        if typical:
            fwd_wins = sum(1 for _, _, w, _ in typical if w == "jacfwd")
            rev_wins = sum(1 for _, _, w, _ in typical if w == "jacrev")
            overall = "jacfwd" if fwd_wins >= rev_wins else "jacrev"
            print(f"  For typical XPCS (m >= 50K): jacfwd wins {fwd_wins}/{len(typical)}, jacrev wins {rev_wins}/{len(typical)}")
            print(f"  Recommendation: Use {overall} across all call sites")
        else:
            print("  No results for typical XPCS regime")
    else:
        print("  No benchmark results available")


if __name__ == "__main__":
    main()
