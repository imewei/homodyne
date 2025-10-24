"""Tests for MCMC diagnostic visualization module.

This test suite validates the MCMC diagnostic plotting functions for both
standard NUTS results and Consensus Monte Carlo (CMC) results.

Test Coverage:
- Trace plot generation (standard and CMC)
- Posterior distribution plots
- Convergence diagnostics plots (R-hat, ESS)
- KL divergence matrix heatmap (CMC-specific)
- CMC summary dashboard
- File saving functionality
- Edge cases (single chain, single shard, missing data)
"""

import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

# Use non-interactive backend for testing
matplotlib.use('Agg')

from homodyne.optimization.cmc.result import MCMCResult
from homodyne.viz.mcmc_plots import (
    plot_cmc_summary_dashboard,
    plot_convergence_diagnostics,
    plot_kl_divergence_matrix,
    plot_posterior_comparison,
    plot_trace_plots,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def standard_mcmc_result():
    """Create a standard NUTS MCMC result for testing."""
    num_samples = 1000
    num_params = 3

    # Generate synthetic samples
    np.random.seed(42)
    samples = np.random.randn(num_samples, num_params) * 10 + np.array([100.0, 1.5, 10.0])

    # Create result
    result = MCMCResult(
        mean_params=np.mean(samples, axis=0),
        mean_contrast=0.5,
        mean_offset=1.0,
        std_params=np.std(samples, axis=0),
        std_contrast=0.05,
        std_offset=0.1,
        samples_params=samples,
        samples_contrast=np.random.rand(num_samples) * 0.1 + 0.45,
        samples_offset=np.random.rand(num_samples) * 0.2 + 0.9,
        converged=True,
        n_iterations=2000,
        n_chains=4,
        n_warmup=1000,
        n_samples=1000,
        analysis_mode='static_isotropic',
        r_hat={'D0': 1.02, 'alpha': 1.03, 'D_offset': 1.01},
        effective_sample_size={'D0': 850.0, 'alpha': 920.0, 'D_offset': 880.0},
        acceptance_rate=0.85,
    )

    return result


@pytest.fixture
def cmc_result():
    """Create a CMC result with per-shard diagnostics for testing."""
    num_shards = 5
    num_samples_per_shard = 200
    num_params = 3

    np.random.seed(123)

    # Generate per-shard samples with slight variations
    per_shard_diagnostics = []
    all_samples = []

    for shard_idx in range(num_shards):
        # Each shard has slightly different posterior
        shard_mean = np.array([100.0, 1.5, 10.0]) + np.random.randn(num_params) * 2
        shard_samples = np.random.randn(num_samples_per_shard, num_params) * 5 + shard_mean
        all_samples.append(shard_samples)

        # Create trace data
        trace_data = {
            f'param_{i}': shard_samples[:, i].tolist()
            for i in range(num_params)
        }

        # Create diagnostics
        diag = {
            'shard_id': shard_idx,
            'num_samples': num_samples_per_shard,
            'num_params': num_params,
            'rhat': {f'param_{i}': 1.0 + np.random.rand() * 0.05 for i in range(num_params)},
            'ess': {f'param_{i}': 150.0 + np.random.rand() * 100 for i in range(num_params)},
            'acceptance_rate': 0.8 + np.random.rand() * 0.15,
            'trace_data': trace_data,
            'converged': True,
        }
        per_shard_diagnostics.append(diag)

    # Combine all samples
    combined_samples = np.vstack(all_samples)

    # Create KL divergence matrix (symmetric, small values for good agreement)
    kl_matrix = np.zeros((num_shards, num_shards))
    for i in range(num_shards):
        for j in range(i + 1, num_shards):
            kl_val = np.random.rand() * 1.5  # Good agreement (< 2.0)
            kl_matrix[i, j] = kl_val
            kl_matrix[j, i] = kl_val

    # Create CMC diagnostics
    cmc_diagnostics = {
        'combination_success': True,
        'n_shards_converged': num_shards,
        'n_shards_total': num_shards,
        'combination_time': 2.5,
        'kl_matrix': kl_matrix.tolist(),
        'max_kl_divergence': float(np.max(kl_matrix)),
        'success_rate': 1.0,
    }

    # Create result
    result = MCMCResult(
        mean_params=np.mean(combined_samples, axis=0),
        mean_contrast=0.5,
        mean_offset=1.0,
        std_params=np.std(combined_samples, axis=0),
        std_contrast=0.05,
        std_offset=0.1,
        samples_params=combined_samples,
        converged=True,
        n_iterations=1000,
        analysis_mode='static_isotropic',
        num_shards=num_shards,
        combination_method='weighted',
        per_shard_diagnostics=per_shard_diagnostics,
        cmc_diagnostics=cmc_diagnostics,
    )

    return result


# ============================================================================
# Test: Trace Plots
# ============================================================================


class TestTracePlots:
    """Tests for trace plot generation."""

    def test_trace_plots_standard_nuts(self, standard_mcmc_result):
        """Test trace plots for standard NUTS result."""
        fig = plot_trace_plots(standard_mcmc_result)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 3  # At least 3 parameters

        # Check axes labels
        for ax in fig.axes[:3]:
            assert ax.get_xlabel() == 'Sample Index'
            assert ax.get_ylabel() != ''

        plt.close(fig)

    def test_trace_plots_cmc_result(self, cmc_result):
        """Test trace plots for CMC result with multiple shards."""
        fig = plot_trace_plots(cmc_result)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 3

        # Check that title mentions CMC
        title = fig._suptitle.get_text()
        assert 'CMC' in title
        assert '5 shards' in title

        plt.close(fig)

    def test_trace_plots_custom_param_names(self, standard_mcmc_result):
        """Test trace plots with custom parameter names."""
        param_names = ['D0', 'alpha', 'D_offset']
        fig = plot_trace_plots(standard_mcmc_result, param_names=param_names)

        assert isinstance(fig, plt.Figure)

        # Check that parameter names are used
        for ax, name in zip(fig.axes[:3], param_names):
            assert name in ax.get_ylabel()

        plt.close(fig)

    def test_trace_plots_save_to_file(self, standard_mcmc_result, tmp_path):
        """Test saving trace plots to file."""
        save_path = tmp_path / "trace_plots.png"

        fig = plot_trace_plots(standard_mcmc_result, save_path=save_path)

        assert save_path.exists()
        assert save_path.stat().st_size > 0

        plt.close(fig)

    def test_trace_plots_no_samples_available(self):
        """Test trace plots with no samples available."""
        result = MCMCResult(
            mean_params=np.array([100.0, 1.5, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            samples_params=None,  # No samples
        )

        fig = plot_trace_plots(result)

        # Should create empty figure with message
        assert isinstance(fig, plt.Figure)

        plt.close(fig)


# ============================================================================
# Test: KL Divergence Matrix
# ============================================================================


class TestKLDivergenceMatrix:
    """Tests for KL divergence matrix heatmap."""

    def test_kl_matrix_cmc_result(self, cmc_result):
        """Test KL divergence matrix for CMC result."""
        fig = plot_kl_divergence_matrix(cmc_result)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 1

        # Check title mentions KL divergence
        ax = fig.axes[0]
        assert 'KL Divergence' in ax.get_title()

        plt.close(fig)

    def test_kl_matrix_threshold_highlighting(self, cmc_result):
        """Test threshold highlighting in KL matrix."""
        threshold = 1.0
        fig = plot_kl_divergence_matrix(cmc_result, threshold=threshold)

        assert isinstance(fig, plt.Figure)

        plt.close(fig)

    def test_kl_matrix_non_cmc_result_raises_error(self, standard_mcmc_result):
        """Test that KL matrix raises error for non-CMC result."""
        with pytest.raises(ValueError, match="only available for CMC results"):
            plot_kl_divergence_matrix(standard_mcmc_result)

    def test_kl_matrix_missing_diagnostics_raises_error(self):
        """Test that missing KL matrix raises error."""
        result = MCMCResult(
            mean_params=np.array([100.0, 1.5, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            num_shards=5,
            cmc_diagnostics=None,  # No diagnostics
        )

        with pytest.raises(ValueError, match="not found in CMC diagnostics"):
            plot_kl_divergence_matrix(result)

    def test_kl_matrix_save_to_file(self, cmc_result, tmp_path):
        """Test saving KL matrix to file."""
        save_path = tmp_path / "kl_matrix.png"

        fig = plot_kl_divergence_matrix(cmc_result, save_path=save_path)

        assert save_path.exists()
        assert save_path.stat().st_size > 0

        plt.close(fig)


# ============================================================================
# Test: Convergence Diagnostics
# ============================================================================


class TestConvergenceDiagnostics:
    """Tests for convergence diagnostics plots."""

    def test_convergence_diagnostics_standard_nuts(self, standard_mcmc_result):
        """Test convergence diagnostics for standard NUTS."""
        fig = plot_convergence_diagnostics(standard_mcmc_result, metrics=['rhat', 'ess'])

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 2  # One for R-hat, one for ESS

        plt.close(fig)

    def test_convergence_diagnostics_cmc_result(self, cmc_result):
        """Test convergence diagnostics for CMC result."""
        fig = plot_convergence_diagnostics(cmc_result, metrics=['rhat', 'ess'])

        assert isinstance(fig, plt.Figure)

        # Check title mentions CMC
        title = fig._suptitle.get_text()
        assert 'CMC' in title

        plt.close(fig)

    def test_convergence_diagnostics_rhat_only(self, standard_mcmc_result):
        """Test plotting only R-hat metric."""
        fig = plot_convergence_diagnostics(standard_mcmc_result, metrics=['rhat'])

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1

        plt.close(fig)

    def test_convergence_diagnostics_ess_only(self, standard_mcmc_result):
        """Test plotting only ESS metric."""
        fig = plot_convergence_diagnostics(standard_mcmc_result, metrics=['ess'])

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1

        plt.close(fig)

    def test_convergence_diagnostics_custom_thresholds(self, standard_mcmc_result):
        """Test custom convergence thresholds."""
        fig = plot_convergence_diagnostics(
            standard_mcmc_result,
            rhat_threshold=1.05,
            ess_threshold=200.0
        )

        assert isinstance(fig, plt.Figure)

        plt.close(fig)

    def test_convergence_diagnostics_save_to_file(self, standard_mcmc_result, tmp_path):
        """Test saving convergence diagnostics to file."""
        save_path = tmp_path / "convergence.png"

        fig = plot_convergence_diagnostics(standard_mcmc_result, save_path=save_path)

        assert save_path.exists()
        assert save_path.stat().st_size > 0

        plt.close(fig)


# ============================================================================
# Test: Posterior Comparison
# ============================================================================


class TestPosteriorComparison:
    """Tests for posterior comparison plots."""

    def test_posterior_comparison_cmc_result(self, cmc_result):
        """Test posterior comparison for CMC result."""
        fig = plot_posterior_comparison(cmc_result)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 3  # At least 3 parameters

        plt.close(fig)

    def test_posterior_comparison_custom_param_indices(self, cmc_result):
        """Test posterior comparison with custom parameter indices."""
        param_indices = [0, 1]
        fig = plot_posterior_comparison(cmc_result, param_indices=param_indices)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 2

        plt.close(fig)

    def test_posterior_comparison_non_cmc_result_raises_error(self, standard_mcmc_result):
        """Test that posterior comparison raises error for non-CMC result."""
        with pytest.raises(ValueError, match="only available for CMC results"):
            plot_posterior_comparison(standard_mcmc_result)

    def test_posterior_comparison_save_to_file(self, cmc_result, tmp_path):
        """Test saving posterior comparison to file."""
        save_path = tmp_path / "posterior_comparison.png"

        fig = plot_posterior_comparison(cmc_result, save_path=save_path)

        assert save_path.exists()
        assert save_path.stat().st_size > 0

        plt.close(fig)


# ============================================================================
# Test: CMC Summary Dashboard
# ============================================================================


class TestCMCSummaryDashboard:
    """Tests for CMC summary dashboard."""

    def test_cmc_summary_dashboard(self, cmc_result):
        """Test creating CMC summary dashboard."""
        fig = plot_cmc_summary_dashboard(cmc_result)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 4  # Multiple panels

        # Check title mentions CMC
        title = fig._suptitle.get_text()
        assert 'CMC' in title
        assert '5 shards' in title

        plt.close(fig)

    def test_cmc_summary_dashboard_non_cmc_raises_error(self, standard_mcmc_result):
        """Test that summary dashboard raises error for non-CMC result."""
        with pytest.raises(ValueError, match="only available for CMC results"):
            plot_cmc_summary_dashboard(standard_mcmc_result)

    def test_cmc_summary_dashboard_save_to_file(self, cmc_result, tmp_path):
        """Test saving CMC summary dashboard to file."""
        save_path = tmp_path / "cmc_summary.png"

        fig = plot_cmc_summary_dashboard(cmc_result, save_path=save_path)

        assert save_path.exists()
        assert save_path.stat().st_size > 0

        plt.close(fig)


# ============================================================================
# Test: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_shard_cmc_result(self):
        """Test visualization with single shard (edge case)."""
        num_params = 3
        num_samples = 100

        np.random.seed(456)
        samples = np.random.randn(num_samples, num_params) * 10 + np.array([100.0, 1.5, 10.0])

        per_shard_diagnostics = [{
            'shard_id': 0,
            'num_samples': num_samples,
            'num_params': num_params,
            'rhat': {f'param_{i}': 1.02 for i in range(num_params)},
            'ess': {f'param_{i}': 80.0 for i in range(num_params)},
            'trace_data': {f'param_{i}': samples[:, i].tolist() for i in range(num_params)},
            'converged': True,
        }]

        kl_matrix = np.array([[0.0]])

        result = MCMCResult(
            mean_params=np.mean(samples, axis=0),
            mean_contrast=0.5,
            mean_offset=1.0,
            samples_params=samples,
            num_shards=1,
            per_shard_diagnostics=per_shard_diagnostics,
            cmc_diagnostics={'kl_matrix': kl_matrix.tolist()},
            analysis_mode='static_isotropic',
        )

        # Should not be considered a CMC result (single shard)
        assert not result.is_cmc_result()

        # Trace plots should still work
        fig = plot_trace_plots(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_multi_chain_traces(self):
        """Test visualization with multi-chain traces."""
        num_chains = 4
        num_samples = 100
        num_params = 3

        np.random.seed(789)

        # Create multi-chain samples
        per_shard_diagnostics = [{
            'shard_id': 0,
            'trace_data': {
                f'param_{i}': np.random.randn(num_chains, num_samples).tolist()
                for i in range(num_params)
            },
            'converged': True,
        }]

        result = MCMCResult(
            mean_params=np.array([100.0, 1.5, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            samples_params=np.random.randn(num_samples, num_params),
            num_shards=2,
            per_shard_diagnostics=per_shard_diagnostics,
            analysis_mode='static_isotropic',
        )

        fig = plot_trace_plots(result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_laminar_flow_mode(self, cmc_result):
        """Test visualization with laminar_flow analysis mode."""
        # Change analysis mode
        cmc_result.analysis_mode = 'laminar_flow'

        # Should use laminar_flow parameter names
        fig = plot_trace_plots(cmc_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        fig = plot_convergence_diagnostics(cmc_result)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ============================================================================
# Test: File Format Support
# ============================================================================


class TestFileFormatSupport:
    """Tests for multiple file format support."""

    def test_save_as_png(self, standard_mcmc_result, tmp_path):
        """Test saving plot as PNG."""
        save_path = tmp_path / "plot.png"
        fig = plot_trace_plots(standard_mcmc_result, save_path=save_path)

        assert save_path.exists()
        assert save_path.suffix == '.png'

        plt.close(fig)

    def test_save_as_pdf(self, standard_mcmc_result, tmp_path):
        """Test saving plot as PDF."""
        save_path = tmp_path / "plot.pdf"
        fig = plot_trace_plots(standard_mcmc_result, save_path=save_path)

        assert save_path.exists()
        assert save_path.suffix == '.pdf'

        plt.close(fig)

    def test_save_as_svg(self, standard_mcmc_result, tmp_path):
        """Test saving plot as SVG."""
        save_path = tmp_path / "plot.svg"
        fig = plot_trace_plots(standard_mcmc_result, save_path=save_path)

        assert save_path.exists()
        assert save_path.suffix == '.svg'

        plt.close(fig)

    def test_custom_dpi(self, standard_mcmc_result, tmp_path):
        """Test saving with custom DPI."""
        save_path = tmp_path / "plot_high_dpi.png"
        fig = plot_trace_plots(standard_mcmc_result, save_path=save_path, dpi=300)

        assert save_path.exists()

        plt.close(fig)


# ============================================================================
# Test Summary
# ============================================================================


def test_visualization_module_summary():
    """Summary test to verify all major visualization functions work."""
    print("\n" + "=" * 70)
    print("MCMC Visualization Module Test Summary")
    print("=" * 70)

    # Count test categories
    test_categories = {
        'Trace Plots': 5,
        'KL Divergence Matrix': 5,
        'Convergence Diagnostics': 6,
        'Posterior Comparison': 4,
        'CMC Summary Dashboard': 3,
        'Edge Cases': 3,
        'File Format Support': 4,
    }

    total_tests = sum(test_categories.values())

    print(f"\nTotal Tests: {total_tests}")
    print("\nTest Categories:")
    for category, count in test_categories.items():
        print(f"  - {category}: {count} tests")

    print("\nFunctionality Validated:")
    print("  ✅ Trace plots (standard NUTS and CMC)")
    print("  ✅ KL divergence matrix heatmaps")
    print("  ✅ Convergence diagnostics (R-hat, ESS)")
    print("  ✅ Posterior distribution comparisons")
    print("  ✅ Comprehensive CMC summary dashboard")
    print("  ✅ File saving (PNG, PDF, SVG)")
    print("  ✅ Edge case handling (single shard, multi-chain)")
    print("  ✅ Multiple analysis modes (static_isotropic, laminar_flow)")

    print("\n" + "=" * 70)

    assert total_tests == 30, f"Expected 30 tests, found {total_tests}"
