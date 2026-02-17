#!/usr/bin/env python3
"""End-to-end benchmark for performance optimizations.

Measures the impact of each optimization on key metrics:
1. Jacobian computation time (jacfwd unified)
2. Model function evaluation time (xdata caching)
3. Covariance expansion time (vectorized)
4. Debug logging overhead (guarded)
5. SharedDataManager creation/reconstruction (CMC shared memory)

Usage:
    uv run python scripts/benchmark_perf_optimizations.py
"""

from __future__ import annotations

import gc
import time

import jax
import jax.numpy as jnp
import numpy as np


def benchmark_jacobian_forward_mode() -> dict:
    """Benchmark unified jacfwd performance across typical XPCS sizes."""
    print("\n1. Jacobian (jacfwd) Performance")
    print("-" * 50)

    rng = np.random.default_rng(42)
    results = {}

    for m in [1_000, 5_000, 10_000]:
        for n in [3, 9]:
            t = jnp.array(rng.uniform(0.01, 10.0, size=m))
            target = jnp.array(rng.normal(0, 1, size=m))
            params = jnp.array(rng.uniform(0.1, 10.0, size=n))

            def residual(
                p: jnp.ndarray,
                t: jnp.ndarray = t,
                n: int = n,
                target: jnp.ndarray = target,
            ) -> jnp.ndarray:
                return (
                    jnp.sum(
                        p[:, None] * jnp.power(t[None, :], jnp.arange(n)[:, None]),
                        axis=0,
                    )
                    - target
                )

            # Warmup
            J = jax.jacfwd(residual)(params)
            J.block_until_ready()

            gc.collect()
            times = []
            for _ in range(5):
                t0 = time.perf_counter()
                J = jax.jacfwd(residual)(params)
                J.block_until_ready()
                times.append(time.perf_counter() - t0)

            mean_t = np.mean(times)
            results[(m, n)] = mean_t
            print(f"  m={m:>6,}, n={n}: {mean_t:.4f}s")

    return results


def benchmark_xdata_caching() -> dict:
    """Benchmark xdata JAX conversion with caching vs without."""
    print("\n2. xdata JAX Conversion (Caching)")
    print("-" * 50)

    rng = np.random.default_rng(42)
    results = {}

    for m in [10_000, 100_000, 1_000_000]:
        xdata = rng.uniform(0, 10, size=(m, 3))

        # Without cache: convert every call
        gc.collect()
        times_nocache = []
        for _ in range(10):
            t0 = time.perf_counter()
            t1 = jnp.array(xdata[:, 0])
            t2 = jnp.array(xdata[:, 1])
            phi = jnp.array(xdata[:, 2]).astype(jnp.int32)
            t1.block_until_ready()
            t2.block_until_ready()
            phi.block_until_ready()
            times_nocache.append(time.perf_counter() - t0)

        # With cache: convert once, then retrieve
        cache: dict[int, tuple] = {}
        gc.collect()
        times_cache = []
        for _ in range(10):
            t0 = time.perf_counter()
            xdata_id = id(xdata)
            if xdata_id in cache:
                t1, t2, phi = cache[xdata_id]
            else:
                t1 = jnp.array(xdata[:, 0])
                t2 = jnp.array(xdata[:, 1])
                phi = jnp.array(xdata[:, 2]).astype(jnp.int32)
                cache[xdata_id] = (t1, t2, phi)
            t1.block_until_ready()
            times_cache.append(time.perf_counter() - t0)

        nc = np.mean(times_nocache)
        wc = np.mean(times_cache)
        speedup = nc / wc if wc > 0 else float("inf")
        results[m] = {"nocache": nc, "cache": wc, "speedup": speedup}
        print(
            f"  m={m:>10,}: nocache={nc:.4f}s, cache={wc:.6f}s, speedup={speedup:.1f}x"
        )

    return results


def benchmark_covariance_expansion() -> dict:
    """Benchmark vectorized vs loop-based covariance expansion."""
    print("\n3. Covariance Expansion (Vectorized)")
    print("-" * 50)

    results = {}

    for n_phi in [5, 23, 50]:
        n_physical = 7
        n_total = 2 + n_physical  # Original layout: [contrast, offset, 7 physical]
        rng = np.random.default_rng(42)
        final_cov = rng.uniform(-0.1, 0.1, (n_total, n_total))
        final_cov = final_cov @ final_cov.T  # Make positive semi-definite

        # Loop-based (original)
        gc.collect()
        times_loop = []
        for _ in range(100):
            t0 = time.perf_counter()
            n_expanded = 2 * n_phi + n_physical
            expanded = np.zeros((n_expanded, n_expanded))
            cv = final_cov[0, 0]
            for i in range(n_phi):
                for j in range(n_phi):
                    expanded[i, j] = cv
            ov = final_cov[1, 1]
            for i in range(n_phi):
                for j in range(n_phi):
                    expanded[n_phi + i, n_phi + j] = ov
            coc = final_cov[0, 1]
            for i in range(n_phi):
                for j in range(n_phi):
                    expanded[i, n_phi + j] = coc
                    expanded[n_phi + j, i] = coc
            for i in range(n_physical):
                for j in range(n_physical):
                    expanded[2 * n_phi + i, 2 * n_phi + j] = final_cov[2 + i, 2 + j]
            for i in range(n_physical):
                for k in range(n_phi):
                    expanded[k, 2 * n_phi + i] = final_cov[0, 2 + i]
                    expanded[2 * n_phi + i, k] = final_cov[0, 2 + i]
                    expanded[n_phi + k, 2 * n_phi + i] = final_cov[1, 2 + i]
                    expanded[2 * n_phi + i, n_phi + k] = final_cov[1, 2 + i]
            times_loop.append(time.perf_counter() - t0)

        # Vectorized (new)
        gc.collect()
        times_vec = []
        for _ in range(100):
            t0 = time.perf_counter()
            n_expanded = 2 * n_phi + n_physical
            expanded_v = np.zeros((n_expanded, n_expanded))
            expanded_v[:n_phi, :n_phi] = final_cov[0, 0]
            expanded_v[n_phi : 2 * n_phi, n_phi : 2 * n_phi] = final_cov[1, 1]
            expanded_v[:n_phi, n_phi : 2 * n_phi] = final_cov[0, 1]
            expanded_v[n_phi : 2 * n_phi, :n_phi] = final_cov[0, 1]
            expanded_v[2 * n_phi :, 2 * n_phi :] = final_cov[
                2 : 2 + n_physical, 2 : 2 + n_physical
            ]
            for i in range(n_physical):
                expanded_v[:n_phi, 2 * n_phi + i] = final_cov[0, 2 + i]
                expanded_v[2 * n_phi + i, :n_phi] = final_cov[0, 2 + i]
                expanded_v[n_phi : 2 * n_phi, 2 * n_phi + i] = final_cov[1, 2 + i]
                expanded_v[2 * n_phi + i, n_phi : 2 * n_phi] = final_cov[1, 2 + i]
            times_vec.append(time.perf_counter() - t0)

        # Verify equivalence
        assert np.allclose(expanded, expanded_v), (
            "Vectorized result differs from loop-based!"
        )

        lp = np.mean(times_loop)
        vp = np.mean(times_vec)
        speedup = lp / vp if vp > 0 else float("inf")
        results[n_phi] = {"loop": lp, "vectorized": vp, "speedup": speedup}
        print(
            f"  n_phi={n_phi:>3}: loop={lp:.6f}s, vec={vp:.6f}s, speedup={speedup:.1f}x"
        )

    return results


def benchmark_shared_memory() -> dict:
    """Benchmark SharedDataManager creation and reconstruction."""
    print("\n4. SharedDataManager (CMC Shared Memory)")
    print("-" * 50)

    from homodyne.optimization.cmc.backends.multiprocessing import (
        SharedDataManager,
        _load_shared_array,
        _load_shared_dict,
    )

    rng = np.random.default_rng(42)
    results = {}

    # Simulate typical shared data
    config_dict = {f"key_{i}": f"value_{i}" for i in range(50)}
    ps_dict = {f"param_{i}": {"lower": 0.0, "upper": 10.0} for i in range(9)}
    time_grid = rng.uniform(0, 100, size=10_000)

    # Measure creation time
    gc.collect()
    times_create = []
    for _ in range(20):
        mgr = SharedDataManager()
        t0 = time.perf_counter()
        cr = mgr.create_shared_dict("config", config_dict)
        pr = mgr.create_shared_dict("ps", ps_dict)
        tr = mgr.create_shared_array("tg", time_grid)
        times_create.append(time.perf_counter() - t0)
        mgr.cleanup()

    # Measure reconstruction time (simulates worker side)
    mgr = SharedDataManager()
    cr = mgr.create_shared_dict("config", config_dict)
    pr = mgr.create_shared_dict("ps", ps_dict)
    tr = mgr.create_shared_array("tg", time_grid)

    gc.collect()
    times_recon = []
    for _ in range(20):
        t0 = time.perf_counter()
        c = _load_shared_dict(cr)
        p = _load_shared_dict(pr)
        t = _load_shared_array(tr)
        times_recon.append(time.perf_counter() - t0)

    mgr.cleanup()

    # Verify reconstruction
    assert c == config_dict
    assert p == ps_dict
    assert np.allclose(t, time_grid)

    ct = np.mean(times_create)
    rt = np.mean(times_recon)
    results["create"] = ct
    results["reconstruct"] = rt
    print(f"  Create (3 blocks):       {ct:.6f}s")
    print(f"  Reconstruct (3 blocks):  {rt:.6f}s")
    print(
        f"  Savings per 500 shards:  ~{500 * rt:.3f}s reconstruct vs ~{500 * ct:.3f}s create"
    )

    return results


def benchmark_logging_guard() -> dict:
    """Benchmark guarded vs unguarded debug logging."""
    import logging

    print("\n5. Debug Logging Guard")
    print("-" * 50)

    logger = logging.getLogger("benchmark_test")
    logger.setLevel(logging.WARNING)  # Disable DEBUG

    results: dict[str, float] = {}

    # Simulate hot-path data
    shape = (1000,)
    params = np.random.default_rng(42).normal(0, 1, shape)

    # Unguarded: f-string evaluated even though DEBUG is disabled
    gc.collect()
    times_unguarded = []
    for _ in range(10000):
        t0 = time.perf_counter()
        logger.debug(
            f"params.shape={params.shape}, min={params.min():.6e}, max={params.max():.6e}"
        )
        times_unguarded.append(time.perf_counter() - t0)

    # Guarded: isEnabledFor short-circuits
    gc.collect()
    times_guarded = []
    for _ in range(10000):
        t0 = time.perf_counter()
        if logger.isEnabledFor(10):
            logger.debug(
                "params.shape=%s, min=%s, max=%s",
                params.shape,
                params.min(),
                params.max(),
            )
        times_guarded.append(time.perf_counter() - t0)

    ug = float(np.mean(times_unguarded))
    gd = float(np.mean(times_guarded))
    speedup = ug / gd if gd > 0 else float("inf")
    results["unguarded"] = ug
    results["guarded"] = gd
    results["speedup"] = speedup
    print(f"  Unguarded:  {ug * 1e6:.2f} us/call")
    print(f"  Guarded:    {gd * 1e6:.2f} us/call")
    print(f"  Speedup:    {speedup:.1f}x")

    return results


def main() -> None:
    """Run all benchmarks."""
    print("=" * 60)
    print("  Performance Optimization Benchmark Suite")
    print("=" * 60)
    print(f"JAX version: {jax.__version__}")
    print(f"Platform: {jax.default_backend()}")

    all_results = {}
    all_results["jacobian"] = benchmark_jacobian_forward_mode()
    all_results["xdata_cache"] = benchmark_xdata_caching()
    all_results["covariance"] = benchmark_covariance_expansion()
    all_results["shared_memory"] = benchmark_shared_memory()
    all_results["logging"] = benchmark_logging_guard()

    print("\n" + "=" * 60)
    print("  ALL BENCHMARKS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
