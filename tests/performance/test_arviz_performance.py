"""
ArviZ Performance Benchmarks
============================

Performance benchmarks for ArviZ integration features added in v2.4.1.
Tests measure conversion speed, plotting performance, and summary computation.

Implements Phase 2.2 of TEST_ACTION_PLAN.md.

Benchmarks:
- MCMCResult.to_arviz() conversion speed
- ArviZ trace plot generation
- Summary statistics computation
- Large dataset handling
"""

import gc
import time

import numpy as np
import pytest

# Handle ArviZ imports
try:
    import arviz as az

    HAS_ARVIZ = True
except ImportError:
    HAS_ARVIZ = False
    az = None

# Handle matplotlib for plotting tests
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for CI
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None


@pytest.fixture
def mcmc_result_small():
    """Small MCMCResult for fast benchmarks."""
    from homodyne.optimization.mcmc.cmc.result import MCMCResult

    n_samples = 500
    n_params = 9

    np.random.seed(42)
    samples = np.random.normal(1.0, 0.2, (n_samples, n_params))

    return MCMCResult(
        mean_params=np.mean(samples, axis=0),
        mean_contrast=0.5,
        mean_offset=1.0,
        std_params=np.std(samples, axis=0),
        samples_params=samples,
        ci_95_lower=np.percentile(samples, 2.5, axis=0),
        ci_95_upper=np.percentile(samples, 97.5, axis=0),
        param_names=[
            "contrast_0",
            "contrast_1",
            "contrast_2",
            "offset_0",
            "offset_1",
            "offset_2",
            "D0",
            "alpha",
            "D_offset",
        ],
        n_chains=1,
        n_samples=n_samples,
        analysis_mode="static",
    )


@pytest.fixture
def mcmc_result_medium():
    """Medium MCMCResult for standard benchmarks."""
    from homodyne.optimization.mcmc.cmc.result import MCMCResult

    n_chains = 2
    n_samples = 1000
    n_params = 9

    np.random.seed(42)
    # Generate chain-structured samples
    samples = np.random.normal(1.0, 0.2, (n_chains * n_samples, n_params))

    return MCMCResult(
        mean_params=np.mean(samples, axis=0),
        mean_contrast=0.5,
        mean_offset=1.0,
        std_params=np.std(samples, axis=0),
        samples_params=samples,
        ci_95_lower=np.percentile(samples, 2.5, axis=0),
        ci_95_upper=np.percentile(samples, 97.5, axis=0),
        param_names=[
            "contrast_0",
            "contrast_1",
            "contrast_2",
            "offset_0",
            "offset_1",
            "offset_2",
            "D0",
            "alpha",
            "D_offset",
        ],
        n_chains=n_chains,
        n_samples=n_samples,
        analysis_mode="static",
    )


@pytest.fixture
def mcmc_result_large():
    """Large MCMCResult for stress testing."""
    from homodyne.optimization.mcmc.cmc.result import MCMCResult

    n_chains = 4
    n_samples = 5000
    n_params = 13  # Laminar flow mode

    np.random.seed(42)
    samples = np.random.normal(1.0, 0.2, (n_chains * n_samples, n_params))

    return MCMCResult(
        mean_params=np.mean(samples, axis=0),
        mean_contrast=0.5,
        mean_offset=1.0,
        std_params=np.std(samples, axis=0),
        samples_params=samples,
        ci_95_lower=np.percentile(samples, 2.5, axis=0),
        ci_95_upper=np.percentile(samples, 97.5, axis=0),
        param_names=[
            "contrast_0",
            "contrast_1",
            "contrast_2",
            "offset_0",
            "offset_1",
            "offset_2",
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ],
        n_chains=n_chains,
        n_samples=n_samples,
        analysis_mode="laminar_flow",
    )


@pytest.mark.performance
@pytest.mark.arviz
@pytest.mark.skipif(not HAS_ARVIZ, reason="ArviZ not available")
class TestArviZConversionPerformance:
    """Benchmark ArviZ InferenceData conversion."""

    def test_to_arviz_small_dataset(self, mcmc_result_small):
        """Benchmark to_arviz() with small dataset (500 samples)."""
        result = mcmc_result_small

        # Warmup
        for _ in range(3):
            idata = result.to_arviz()
            del idata
            gc.collect()

        # Benchmark
        times = []
        for _ in range(10):
            gc.collect()
            start = time.perf_counter()
            idata = result.to_arviz()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            del idata

        avg_time = np.mean(times)
        std_time = np.std(times)

        # Performance assertions
        assert avg_time < 0.5, f"to_arviz() too slow: {avg_time:.3f}s (small dataset)"
        assert std_time < avg_time, f"High variance: std={std_time:.3f}s"

        print(f"\nto_arviz (small): {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")

    def test_to_arviz_medium_dataset(self, mcmc_result_medium):
        """Benchmark to_arviz() with medium dataset (2000 samples)."""
        result = mcmc_result_medium

        # Warmup
        for _ in range(2):
            idata = result.to_arviz()
            del idata
            gc.collect()

        # Benchmark
        times = []
        for _ in range(5):
            gc.collect()
            start = time.perf_counter()
            idata = result.to_arviz()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            del idata

        avg_time = np.mean(times)

        # Performance assertions
        assert avg_time < 1.0, f"to_arviz() too slow: {avg_time:.3f}s (medium dataset)"

        print(f"\nto_arviz (medium): {avg_time*1000:.2f}ms")

    def test_to_arviz_large_dataset(self, mcmc_result_large):
        """Benchmark to_arviz() with large dataset (20000 samples)."""
        result = mcmc_result_large

        # Single warmup
        idata = result.to_arviz()
        del idata
        gc.collect()

        # Benchmark
        times = []
        for _ in range(3):
            gc.collect()
            start = time.perf_counter()
            idata = result.to_arviz()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            del idata

        avg_time = np.mean(times)

        # Performance assertions
        assert avg_time < 5.0, f"to_arviz() too slow: {avg_time:.3f}s (large dataset)"

        print(f"\nto_arviz (large): {avg_time*1000:.2f}ms")

    def test_to_arviz_scaling(self, mcmc_result_small, mcmc_result_medium, mcmc_result_large):
        """Verify to_arviz() scales reasonably with dataset size."""
        results = {
            "small": mcmc_result_small,
            "medium": mcmc_result_medium,
            "large": mcmc_result_large,
        }

        times = {}
        sizes = {}

        for name, result in results.items():
            gc.collect()
            start = time.perf_counter()
            idata = result.to_arviz()
            elapsed = time.perf_counter() - start
            times[name] = elapsed
            sizes[name] = result.samples_params.shape[0] if result.samples_params is not None else 0
            del idata

        # Check scaling is sub-linear or linear
        size_ratio = sizes["large"] / sizes["small"]
        time_ratio = times["large"] / times["small"]

        # Should scale at most linearly (time_ratio <= size_ratio * 2)
        assert time_ratio < size_ratio * 3, f"Poor scaling: {time_ratio:.2f}x time for {size_ratio:.2f}x size"

        print(f"\nScaling: {size_ratio:.1f}x data → {time_ratio:.1f}x time")


@pytest.mark.performance
@pytest.mark.arviz
@pytest.mark.visualization
@pytest.mark.skipif(not HAS_ARVIZ, reason="ArviZ not available")
@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")
class TestArviZPlottingPerformance:
    """Benchmark ArviZ plotting functions."""

    def test_plot_trace_speed(self, mcmc_result_medium):
        """Benchmark trace plot generation."""
        from homodyne.viz.mcmc_plots import plot_trace_plots

        result = mcmc_result_medium

        # Warmup
        fig = plot_trace_plots(result, max_params=3)
        plt.close(fig)
        gc.collect()

        # Benchmark
        times = []
        for _ in range(3):
            gc.collect()
            start = time.perf_counter()
            fig = plot_trace_plots(result, max_params=6)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            plt.close(fig)

        avg_time = np.mean(times)

        # Performance assertions
        assert avg_time < 5.0, f"plot_trace_plots() too slow: {avg_time:.3f}s"

        print(f"\nplot_trace_plots: {avg_time*1000:.2f}ms")

    def test_plot_arviz_trace_speed(self, mcmc_result_medium):
        """Benchmark ArviZ-native trace plot."""
        from homodyne.viz.mcmc_plots import plot_arviz_trace

        result = mcmc_result_medium

        # Warmup
        fig = plot_arviz_trace(result, var_names=["D0", "alpha"])
        plt.close(fig)
        gc.collect()

        # Benchmark
        times = []
        for _ in range(3):
            gc.collect()
            start = time.perf_counter()
            fig = plot_arviz_trace(result, var_names=["D0", "alpha", "D_offset"])
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            plt.close(fig)

        avg_time = np.mean(times)

        # Performance assertions
        assert avg_time < 10.0, f"plot_arviz_trace() too slow: {avg_time:.3f}s"

        print(f"\nplot_arviz_trace: {avg_time*1000:.2f}ms")

    def test_plot_arviz_posterior_speed(self, mcmc_result_medium):
        """Benchmark ArviZ posterior plot."""
        from homodyne.viz.mcmc_plots import plot_arviz_posterior

        result = mcmc_result_medium

        # Warmup
        fig = plot_arviz_posterior(result, var_names=["D0"])
        plt.close(fig)
        gc.collect()

        # Benchmark
        times = []
        for _ in range(3):
            gc.collect()
            start = time.perf_counter()
            fig = plot_arviz_posterior(result, var_names=["D0", "alpha", "D_offset"])
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            plt.close(fig)

        avg_time = np.mean(times)

        # Performance assertions
        assert avg_time < 5.0, f"plot_arviz_posterior() too slow: {avg_time:.3f}s"

        print(f"\nplot_arviz_posterior: {avg_time*1000:.2f}ms")

    def test_plot_arviz_pair_speed(self, mcmc_result_small):
        """Benchmark ArviZ pair plot (uses small dataset due to O(n^2) complexity)."""
        import warnings

        from homodyne.viz.mcmc_plots import plot_arviz_pair

        result = mcmc_result_small

        # Warmup with 2 params only (suppress tight_layout warnings)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*tight_layout.*")
            fig = plot_arviz_pair(result, var_names=["D0", "alpha"])
            plt.close(fig)
        gc.collect()

        # Benchmark with 3 params
        times = []
        for _ in range(2):
            gc.collect()
            start = time.perf_counter()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*tight_layout.*")
                fig = plot_arviz_pair(result, var_names=["D0", "alpha", "D_offset"])
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            plt.close(fig)

        avg_time = np.mean(times)

        # Performance assertions (pair plots are slow)
        assert avg_time < 15.0, f"plot_arviz_pair() too slow: {avg_time:.3f}s"

        print(f"\nplot_arviz_pair (3 params): {avg_time*1000:.2f}ms")


@pytest.mark.performance
@pytest.mark.arviz
@pytest.mark.skipif(not HAS_ARVIZ, reason="ArviZ not available")
class TestArviZSummaryPerformance:
    """Benchmark ArviZ summary statistics."""

    def test_arviz_summary_speed(self, mcmc_result_medium):
        """Benchmark az.summary() computation."""
        result = mcmc_result_medium
        idata = result.to_arviz()

        # Warmup
        summary = az.summary(idata, var_names=["D0"])
        del summary
        gc.collect()

        # Benchmark
        times = []
        for _ in range(5):
            gc.collect()
            start = time.perf_counter()
            summary = az.summary(idata, var_names=["D0", "alpha", "D_offset"])
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            del summary

        avg_time = np.mean(times)

        # Performance assertions
        assert avg_time < 2.0, f"az.summary() too slow: {avg_time:.3f}s"

        print(f"\naz.summary (3 params): {avg_time*1000:.2f}ms")

    def test_arviz_rhat_speed(self, mcmc_result_medium):
        """Benchmark az.rhat() computation."""
        result = mcmc_result_medium
        idata = result.to_arviz()

        # Warmup
        rhat = az.rhat(idata, var_names=["D0"])
        del rhat
        gc.collect()

        # Benchmark
        times = []
        for _ in range(5):
            gc.collect()
            start = time.perf_counter()
            rhat = az.rhat(idata, var_names=["D0", "alpha", "D_offset"])
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            del rhat

        avg_time = np.mean(times)

        # Performance assertions
        assert avg_time < 1.0, f"az.rhat() too slow: {avg_time:.3f}s"

        print(f"\naz.rhat (3 params): {avg_time*1000:.2f}ms")

    def test_arviz_ess_speed(self, mcmc_result_medium):
        """Benchmark az.ess() computation."""
        result = mcmc_result_medium
        idata = result.to_arviz()

        # Warmup
        ess = az.ess(idata, var_names=["D0"])
        del ess
        gc.collect()

        # Benchmark
        times = []
        for _ in range(5):
            gc.collect()
            start = time.perf_counter()
            ess = az.ess(idata, var_names=["D0", "alpha", "D_offset"])
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            del ess

        avg_time = np.mean(times)

        # Performance assertions
        assert avg_time < 1.0, f"az.ess() too slow: {avg_time:.3f}s"

        print(f"\naz.ess (3 params): {avg_time*1000:.2f}ms")


@pytest.mark.performance
@pytest.mark.arviz
@pytest.mark.slow
@pytest.mark.skipif(not HAS_ARVIZ, reason="ArviZ not available")
class TestArviZMemoryPerformance:
    """Benchmark ArviZ memory usage."""

    def test_memory_usage_to_arviz(self, mcmc_result_large):
        """Check memory overhead of to_arviz() conversion."""
        try:
            import psutil

            HAS_PSUTIL = True
        except ImportError:
            pytest.skip("psutil not available for memory profiling")

        process = psutil.Process()
        result = mcmc_result_large

        # Measure baseline memory
        gc.collect()
        baseline_mb = process.memory_info().rss / (1024 * 1024)

        # Convert to ArviZ
        idata = result.to_arviz()

        # Measure memory after conversion
        gc.collect()
        after_mb = process.memory_info().rss / (1024 * 1024)

        memory_increase_mb = after_mb - baseline_mb

        # Should not more than double memory usage
        input_size_mb = result.samples_params.nbytes / (1024 * 1024) if result.samples_params is not None else 0

        # Memory increase should be reasonable (less than 10x input size)
        assert memory_increase_mb < input_size_mb * 10, (
            f"Excessive memory: {memory_increase_mb:.1f}MB for {input_size_mb:.1f}MB input"
        )

        print(f"\nMemory: +{memory_increase_mb:.1f}MB for {input_size_mb:.1f}MB input ({memory_increase_mb/input_size_mb:.1f}x)")

        del idata


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
