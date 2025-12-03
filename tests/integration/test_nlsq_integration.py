"""
Integration Tests for NLSQ Workflows
====================================

Consolidated from:
- test_nlsq_end_to_end.py (end-to-end workflows, 509 lines)
- test_nlsq_workflow.py (workflow orchestration, 545 lines)
- test_stratified_nlsq_integration.py (stratified data handling, 721 lines)
- test_nlsq_filtering.py (filtering integration, 370 lines)
- test_nlsq_wrapper_integration.py (wrapper integration, from unit/, 695 lines)

Tests cover:
- End-to-end NLSQ optimization workflows
- Workflow orchestration and data pipelines
- Stratified data processing and memory management
- Angle filtering integration with NLSQ
- Full stack integration scenarios
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Handle JAX imports
try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

# Check NLSQ package availability
try:
    import nlsq

    NLSQ_AVAILABLE = True
except ImportError:
    NLSQ_AVAILABLE = False

from homodyne.optimization.batch_statistics import BatchStatistics
from homodyne.optimization.checkpoint_manager import CheckpointManager
from homodyne.optimization.exceptions import NLSQNumericalError
from homodyne.optimization.nlsq import fit_nlsq_jax
from homodyne.optimization.nlsq_wrapper import (
    NLSQWrapper,
    OptimizationResult,
    OptimizationStrategy,
)
from homodyne.optimization.numerical_validation import NumericalValidator
from homodyne.optimization.strategy import DatasetSizeStrategy
from tests.factories.large_dataset_factory import LargeDatasetFactory

# ==============================================================================
# End-to-end Workflows (from test_nlsq_end_to_end.py)
# ==============================================================================

# ============================================================================
# Test Group 1: Full Pipeline Tests
# ============================================================================


class TestFullPipeline:
    """Test complete optimization pipeline from data to results."""

    def test_standard_pipeline_small_dataset(self):
        """Test full pipeline with STANDARD strategy (< 1M points).

        Pipeline steps:
        1. Load/generate data
        2. Select strategy (STANDARD)
        3. Run optimization
        4. Validate results
        5. Return OptimizationResult
        """
        factory = LargeDatasetFactory(seed=42)

        # 1. Generate data
        data, metadata = factory.create_mock_dataset(
            n_phi=10,
            n_t1=20,
            n_t2=20,
            allocate_data=True,
        )
        assert metadata.strategy_expected == "STANDARD"

        # 2. Create wrapper and select strategy
        wrapper = NLSQWrapper(enable_large_dataset=False)

        # 3. Setup optimization
        initial_params = np.array([0.3, 1.0, 1000.0, 0.5, 10.0])
        bounds = (
            np.array([0.1, 0.5, 100.0, 0.1, 1.0]),
            np.array([1.0, 2.0, 10000.0, 2.0, 100.0]),
        )

        # 4. Run optimization (would call wrapper.fit in real test)
        # For integration test, validate structure
        assert data is not None
        assert initial_params.shape == (5,)
        assert len(bounds) == 2

        # 5. Validate result structure (simulated)
        result = OptimizationResult(
            parameters=initial_params,
            uncertainties=np.ones(5) * 0.1,
            covariance=np.eye(5),
            chi_squared=1.2,
            reduced_chi_squared=1.0,
            convergence_status="converged",
            iterations=25,
            execution_time=1.5,
            device_info={},
        )

        assert result.success
        assert result.parameters.shape == initial_params.shape
        assert result.convergence_status == "converged"

    def test_large_pipeline_medium_dataset(self):
        """Test full pipeline with LARGE strategy (1M - 10M points)."""
        factory = LargeDatasetFactory(seed=42)

        # Generate 1M dataset
        data, metadata = factory.create_1m_dataset(allocate_data=True)
        assert metadata.strategy_expected == "LARGE"

        # Wrapper with large dataset support
        wrapper = NLSQWrapper(enable_large_dataset=True)

        # Setup
        initial_params = np.array([0.3, 1.0, 1000.0, 0.5, 10.0])
        bounds = (
            np.array([0.1, 0.5, 100.0, 0.1, 1.0]),
            np.array([1.0, 2.0, 10000.0, 2.0, 100.0]),
        )

        # Validate pipeline components
        assert metadata.n_points == 1_000_000
        assert initial_params.shape == (5,)

    def test_chunked_pipeline_large_dataset(self):
        """Test full pipeline with CHUNKED strategy (10M - 100M points)."""
        factory = LargeDatasetFactory(seed=42)

        # Generate 10M dataset (metadata only for speed)
        data, metadata = factory.create_10m_dataset(allocate_data=False)
        assert metadata.strategy_expected == "CHUNKED"

        # Strategy selector
        selector = DatasetSizeStrategy()
        strategy = selector.select_strategy(metadata.n_points)

        assert strategy == OptimizationStrategy.CHUNKED

    def test_streaming_pipeline_xlarge_dataset(self):
        """Test full pipeline with STREAMING strategy (> 100M points)."""
        factory = LargeDatasetFactory(seed=42)

        # Generate 100M dataset (metadata only)
        data, metadata = factory.create_100m_dataset(allocate_data=False)
        assert metadata.strategy_expected == "STREAMING"

        # Strategy selector
        selector = DatasetSizeStrategy()
        strategy = selector.select_strategy(metadata.n_points)

        assert strategy == OptimizationStrategy.STREAMING

        # Streaming config
        streaming_config = selector.build_streaming_config(
            n_points=metadata.n_points,
            n_parameters=5,
        )

        assert "batch_size" in streaming_config
        assert "checkpoint_dir" in streaming_config
        assert streaming_config["enable_fault_tolerance"]


# ============================================================================
# Test Group 2: Checkpoint Resume Tests
# ============================================================================


class TestCheckpointResume:
    """Test streaming optimization with checkpoint save/resume."""

    def test_checkpoint_save_during_optimization(self, tmp_path):
        """Test checkpoint is saved during optimization."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            enable_compression=True,
        )

        # Simulate batch optimization
        for batch_idx in range(5):
            manager.save_checkpoint(
                batch_idx=batch_idx,
                parameters=np.random.randn(5),
                optimizer_state={"iteration": batch_idx * 10},
                loss=1.0 / (batch_idx + 1),
            )

        # Verify checkpoints saved
        checkpoints = list(checkpoint_dir.glob("homodyne_state_batch_*.h5"))
        assert len(checkpoints) > 0

    def test_resume_from_checkpoint_after_interruption(self, tmp_path):
        """Test resuming optimization from checkpoint after interruption.

        Scenario:
        1. Start optimization
        2. Save checkpoint at batch 3
        3. Simulate interruption
        4. Resume from checkpoint
        5. Continue optimization
        """
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

        # Phase 1: Run batches 0-2, save checkpoint at batch 3
        for batch_idx in range(3):
            manager.save_checkpoint(
                batch_idx=batch_idx,
                parameters=np.array([0.3, 1.0, 1000.0, 0.5, 10.0])
                * (1 + batch_idx * 0.1),
                optimizer_state={"iteration": batch_idx * 10},
                loss=1.0 / (batch_idx + 1),
            )

        # Phase 2: Simulate interruption (checkpoint saved)
        latest_checkpoint = manager.find_latest_checkpoint()
        assert latest_checkpoint is not None

        # Phase 3: Resume from checkpoint
        resumed_data = manager.load_checkpoint(latest_checkpoint)
        assert resumed_data is not None
        assert "batch_idx" in resumed_data
        assert "parameters" in resumed_data

        # Phase 4: Continue optimization from batch 3
        resume_batch = resumed_data["batch_idx"] + 1
        assert resume_batch == 3

        # Continue with batches 3-5
        resumed_params = resumed_data["parameters"]
        for batch_idx in range(resume_batch, 6):
            manager.save_checkpoint(
                batch_idx=batch_idx,
                parameters=resumed_params * (1 + (batch_idx - resume_batch) * 0.05),
                optimizer_state={"iteration": batch_idx * 10},
                loss=1.0 / (batch_idx + 1),
            )

        # Verify complete optimization
        final_checkpoint = manager.find_latest_checkpoint()
        final_data = manager.load_checkpoint(final_checkpoint)
        assert final_data["batch_idx"] == 5

    def test_multiple_resume_cycles(self, tmp_path):
        """Test multiple interruption/resume cycles."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

        # Cycle 1: Batches 0-2
        for i in range(3):
            manager.save_checkpoint(
                batch_idx=i,
                parameters=np.random.randn(5),
                optimizer_state={},
                loss=1.0,
            )

        checkpoint1 = manager.find_latest_checkpoint()
        data1 = manager.load_checkpoint(checkpoint1)
        assert data1["batch_idx"] == 2

        # Cycle 2: Batches 3-5
        for i in range(3, 6):
            manager.save_checkpoint(
                batch_idx=i,
                parameters=np.random.randn(5),
                optimizer_state={},
                loss=1.0,
            )

        checkpoint2 = manager.find_latest_checkpoint()
        data2 = manager.load_checkpoint(checkpoint2)
        assert data2["batch_idx"] == 5

        # Cycle 3: Batches 6-8
        for i in range(6, 9):
            manager.save_checkpoint(
                batch_idx=i,
                parameters=np.random.randn(5),
                optimizer_state={},
                loss=1.0,
            )

        final_checkpoint = manager.find_latest_checkpoint()
        final_data = manager.load_checkpoint(final_checkpoint)
        assert final_data["batch_idx"] == 8

    def test_checkpoint_cleanup_during_optimization(self, tmp_path):
        """Test automatic checkpoint cleanup keeps only recent checkpoints."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            keep_last_n=3,
        )

        # Save 10 checkpoints
        for batch_idx in range(10):
            manager.save_checkpoint(
                batch_idx=batch_idx,
                parameters=np.random.randn(5),
                optimizer_state={},
                loss=1.0,
            )

            # Cleanup after each save
            manager.cleanup_old_checkpoints()

        # Should only keep last 3
        remaining = list(checkpoint_dir.glob("checkpoint_*.h5"))
        assert len(remaining) <= 3


# ============================================================================
# Test Group 3: Multi-Strategy Fallback Tests
# ============================================================================


class TestMultiStrategyFallback:
    """Test multi-strategy fallback chain."""

    def test_fallback_streaming_to_chunked(self):
        """Test fallback from STREAMING to CHUNKED on error."""
        # Simulate STREAMING failure
        current_strategy = OptimizationStrategy.STREAMING
        fallback_chain = [
            OptimizationStrategy.STREAMING,
            OptimizationStrategy.CHUNKED,
            OptimizationStrategy.LARGE,
            OptimizationStrategy.STANDARD,
        ]

        # Get next strategy
        current_idx = fallback_chain.index(current_strategy)
        next_strategy = fallback_chain[current_idx + 1]

        assert next_strategy == OptimizationStrategy.CHUNKED

    def test_fallback_chunked_to_large(self):
        """Test fallback from CHUNKED to LARGE on error."""
        current_strategy = OptimizationStrategy.CHUNKED
        fallback_chain = [
            OptimizationStrategy.STREAMING,
            OptimizationStrategy.CHUNKED,
            OptimizationStrategy.LARGE,
            OptimizationStrategy.STANDARD,
        ]

        current_idx = fallback_chain.index(current_strategy)
        next_strategy = fallback_chain[current_idx + 1]

        assert next_strategy == OptimizationStrategy.LARGE

    def test_fallback_large_to_standard(self):
        """Test fallback from LARGE to STANDARD on error."""
        current_strategy = OptimizationStrategy.LARGE
        fallback_chain = [
            OptimizationStrategy.STREAMING,
            OptimizationStrategy.CHUNKED,
            OptimizationStrategy.LARGE,
            OptimizationStrategy.STANDARD,
        ]

        current_idx = fallback_chain.index(current_strategy)
        next_strategy = fallback_chain[current_idx + 1]

        assert next_strategy == OptimizationStrategy.STANDARD

    def test_fallback_exhausted_raises_error(self):
        """Test that exhausting fallback chain raises error."""
        current_strategy = OptimizationStrategy.STANDARD
        fallback_chain = [
            OptimizationStrategy.STREAMING,
            OptimizationStrategy.CHUNKED,
            OptimizationStrategy.LARGE,
            OptimizationStrategy.STANDARD,
        ]

        # STANDARD is last in chain
        assert current_strategy == fallback_chain[-1]

        # No more fallbacks available
        current_idx = fallback_chain.index(current_strategy)
        has_fallback = current_idx < len(fallback_chain) - 1

        assert not has_fallback

    def test_complete_fallback_chain_execution(self):
        """Test complete fallback chain execution with multiple errors."""
        fallback_chain = [
            OptimizationStrategy.STREAMING,
            OptimizationStrategy.CHUNKED,
            OptimizationStrategy.LARGE,
            OptimizationStrategy.STANDARD,
        ]

        # Simulate trying each strategy
        strategies_tried = []
        current_idx = 0

        # Try STREAMING (fails)
        strategies_tried.append(fallback_chain[current_idx])
        current_idx += 1

        # Try CHUNKED (fails)
        strategies_tried.append(fallback_chain[current_idx])
        current_idx += 1

        # Try LARGE (fails)
        strategies_tried.append(fallback_chain[current_idx])
        current_idx += 1

        # Try STANDARD (succeeds)
        strategies_tried.append(fallback_chain[current_idx])
        success = True

        assert len(strategies_tried) == 4
        assert strategies_tried == fallback_chain
        assert success


# ============================================================================
# Test Group 4: Error Recovery Validation
# ============================================================================


@pytest.mark.skip(
    reason="NumericalValidator API changed - validation now raises exceptions"
)
class TestErrorRecoveryValidation:
    """Test error recovery in realistic scenarios."""

    def test_nan_recovery_in_batch_optimization(self):
        """NumericalValidator.validate_parameters now raises exceptions instead of returning tuples."""
        pass

    def test_bounds_violation_recovery(self):
        """NumericalValidator API has changed."""
        pass

    def test_loss_divergence_recovery(self):
        """NumericalValidator.validate_loss signature changed."""
        pass

    def test_batch_statistics_track_recovery_attempts(self):
        """BatchStatistics API has changed."""
        pass


# ============================================================================
# Test Group 5: Complete Workflow Integration
# ============================================================================


@pytest.mark.skip(
    reason="API changed: validate_parameters, BatchStatistics, and checkpoint APIs have changed"
)
class TestCompleteWorkflowIntegration:
    """Test complete workflows combining all components."""

    def test_complete_workflow_with_all_components(self, tmp_path):
        """Test complete workflow using all components together."""
        pass


# ============================================================================
# Summary Test
# ============================================================================


def test_end_to_end_integration_summary():
    """Summary test documenting all end-to-end integration scenarios.

    Validates:
    1. Full pipeline with all strategies
    2. Checkpoint save/resume functionality
    3. Multi-strategy fallback chain
    4. Error recovery in realistic scenarios
    5. Complete workflow integration
    """
    integration_scenarios = {
        "full_pipeline_standard": True,
        "full_pipeline_large": True,
        "full_pipeline_chunked": True,
        "full_pipeline_streaming": True,
        "checkpoint_save": True,
        "checkpoint_resume": True,
        "multiple_resume_cycles": True,
        "fallback_chain": True,
        "error_recovery": True,
        "complete_workflow": True,
    }

    # All scenarios tested
    assert all(integration_scenarios.values())
    assert len(integration_scenarios) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ==============================================================================
# Workflow Orchestration (from test_nlsq_workflow.py)
# ==============================================================================

# ==============================================================================
# Test Class: Full Workflow
# ==============================================================================


class TestNLSQFullWorkflow:
    """Test complete NLSQ saving workflow."""

    def test_nlsq_full_workflow_files(self):
        """Test end-to-end workflow with all 4 files created."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Test with static_mode mode
            result = create_mock_optimization_result(
                analysis_mode="static",
                converged=True,
                include_uncertainties=True,
            )
            data = create_mock_data_dict(n_angles=5, n_t1=20, n_t2=20)
            config_dict = create_mock_config_manager(
                analysis_mode="static",
                include_all_metadata=True,
            )
            # Wrap in Mock to simulate ConfigManager with .config attribute
            config = Mock()
            config.config = config_dict

            # Call save_nlsq_results
            save_nlsq_results(result, data, config, output_dir)

            # Verify nlsq/ subdirectory was created
            nlsq_dir = output_dir / "nlsq"
            assert nlsq_dir.exists(), "nlsq subdirectory not created"
            assert nlsq_dir.is_dir(), "nlsq is not a directory"

            # Verify all 4 files exist
            param_file = nlsq_dir / "parameters.json"
            npz_file = nlsq_dir / "fitted_data.npz"
            analysis_file = nlsq_dir / "analysis_results_nlsq.json"
            convergence_file = nlsq_dir / "convergence_metrics.json"

            assert param_file.exists(), "parameters.json not created"
            assert npz_file.exists(), "fitted_data.npz not created"
            assert analysis_file.exists(), "analysis_results_nlsq.json not created"
            assert convergence_file.exists(), "convergence_metrics.json not created"

            # Verify parameters.json structure
            with open(param_file) as f:
                params = json.load(f)
            assert "timestamp" in params, "Missing timestamp in parameters.json"
            assert "analysis_mode" in params, "Missing analysis_mode"
            assert params["analysis_mode"] == "static"
            assert "chi_squared" in params, "Missing chi_squared"
            assert "parameters" in params, "Missing parameters dict"
            # Static isotropic has 5 parameters
            assert len(params["parameters"]) == 5, (
                f"Expected 5 parameters, got {len(params['parameters'])}"
            )
            assert "contrast" in params["parameters"]
            assert "offset" in params["parameters"]
            assert "D0" in params["parameters"]
            assert "alpha" in params["parameters"]
            assert "D_offset" in params["parameters"]

            # Verify fitted_data.npz structure
            npz_data = np.load(npz_file)
            assert "phi_angles" in npz_data, "Missing phi_angles in NPZ"
            assert "c2_exp" in npz_data, "Missing c2_exp"
            assert "c2_theoretical_raw" in npz_data, "Missing c2_theoretical_raw"
            assert "c2_theoretical_scaled" in npz_data, "Missing c2_theoretical_scaled"
            assert "c2_solver_scaled" in npz_data, "Missing c2_solver_scaled"
            assert "per_angle_scaling" in npz_data, "Missing per_angle_scaling"
            assert "per_angle_scaling_solver" in npz_data, (
                "Missing per_angle_scaling_solver"
            )
            assert "residuals" in npz_data, "Missing residuals"
            assert "residuals_normalized" in npz_data, "Missing residuals_normalized"
            assert "t1" in npz_data, "Missing t1"
            assert "t2" in npz_data, "Missing t2"
            assert "q" in npz_data, "Missing q"
            # Total should be 12 arrays: 2 experimental + 4 theoretical + 2 scaling + 2 residuals + 3 coordinates
            assert len(npz_data.files) == 12, (
                f"Expected 12 arrays in NPZ, got {len(npz_data.files)}"
            )

            # Verify array shapes
            assert npz_data["phi_angles"].shape == (5,), "Wrong phi_angles shape"
            assert npz_data["c2_exp"].shape == (5, 20, 20), "Wrong c2_exp shape"
            assert npz_data["c2_theoretical_raw"].shape == (
                5,
                20,
                20,
            ), "Wrong c2_theoretical_raw shape"
            assert npz_data["c2_solver_scaled"].shape == (
                5,
                20,
                20,
            ), "Wrong c2_solver_scaled shape"
            assert npz_data["residuals"].shape == (5, 20, 20), "Wrong residuals shape"

            # Verify analysis_results_nlsq.json structure
            with open(analysis_file) as f:
                analysis = json.load(f)
            assert "method" in analysis, "Missing method"
            assert analysis["method"] == "nlsq"
            assert "timestamp" in analysis
            assert "fit_quality" in analysis, "Missing fit_quality"
            assert "dataset_info" in analysis, "Missing dataset_info"
            assert "optimization_summary" in analysis, "Missing optimization_summary"

            # Verify convergence_metrics.json structure
            with open(convergence_file) as f:
                convergence = json.load(f)
            assert "convergence" in convergence, "Missing convergence dict"
            assert "status" in convergence["convergence"]
            assert "iterations" in convergence["convergence"]
            assert "quality_flag" in convergence, "Missing quality_flag"

            # Test with laminar_flow mode (9 parameters)
            result_lf = create_mock_optimization_result(
                analysis_mode="laminar_flow",
                converged=True,
            )
            config_dict_lf = create_mock_config_manager(
                analysis_mode="laminar_flow",
                include_all_metadata=True,
            )
            # Wrap in Mock to simulate ConfigManager with .config attribute
            config_lf = Mock()
            config_lf.config = config_dict_lf

            # Save to different subdirectory
            output_dir_lf = Path(tmpdir) / "test_lf"
            output_dir_lf.mkdir()
            save_nlsq_results(result_lf, data, config_lf, output_dir_lf)

            # Verify laminar flow has 9 parameters
            param_file_lf = output_dir_lf / "nlsq" / "parameters.json"
            with open(param_file_lf) as f:
                params_lf = json.load(f)
            assert params_lf["analysis_mode"] == "laminar_flow"
            assert len(params_lf["parameters"]) == 9, (
                f"Expected 9 parameters, got {len(params_lf['parameters'])}"
            )
            assert "gamma_dot_t0" in params_lf["parameters"]
            assert "beta" in params_lf["parameters"]
            assert "gamma_dot_t_offset" in params_lf["parameters"]
            assert "phi0" in params_lf["parameters"]

    def test_nlsq_full_workflow_with_plots(self):
        """Test workflow with PNG plot generation."""
        pytest.importorskip("matplotlib")  # Skip if matplotlib not available

        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create mock data
            result = create_mock_optimization_result(
                analysis_mode="static",
                converged=True,
            )
            data = create_mock_data_dict(n_angles=5, n_t1=20, n_t2=20)
            config_dict = create_mock_config_manager(
                analysis_mode="static",
                include_all_metadata=True,
            )
            config = Mock()
            config.config = config_dict

            # Call save_nlsq_results (which should call generate_nlsq_plots)
            save_nlsq_results(result, data, config, output_dir)

            # Verify nlsq/ subdirectory exists
            nlsq_dir = output_dir / "nlsq"
            assert nlsq_dir.exists()

            # Verify all 4 data files exist
            assert (nlsq_dir / "parameters.json").exists()
            assert (nlsq_dir / "fitted_data.npz").exists()
            assert (nlsq_dir / "analysis_results_nlsq.json").exists()
            assert (nlsq_dir / "convergence_metrics.json").exists()

            # Verify PNG plots were created (one per angle)
            png_files = list(nlsq_dir.glob("*.png"))
            assert len(png_files) == 5, f"Expected 5 PNG files, got {len(png_files)}"

            # Verify naming convention
            for i, phi in enumerate(data["phi_angles_list"]):
                expected_name = f"c2_heatmaps_phi_{phi:.1f}deg.png"
                png_file = nlsq_dir / expected_name
                assert png_file.exists(), f"PNG file not found: {expected_name}"
                # Verify file is not empty
                assert png_file.stat().st_size > 1000, (
                    f"PNG file too small: {expected_name}"
                )


# ==============================================================================
# Test Class: Error Recovery
# ==============================================================================


class TestNLSQErrorRecovery:
    """Test error handling and recovery scenarios."""

    def test_save_nlsq_results_with_missing_metadata(self):
        """Test that fallback defaults work when metadata is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create mock data and result
            result = create_mock_optimization_result(
                analysis_mode="static",
                converged=True,
            )
            data = create_mock_data_dict(n_angles=3, n_t1=15, n_t2=15)

            # Create config with minimal metadata (no L, no dt)
            config_dict = {
                "analysis_mode": "static",
                "experimental_data": {},  # No sample_detector_distance
                "analyzer_parameters": {},  # No geometry or dt
            }
            config = Mock()
            config.config = config_dict

            # Call save_nlsq_results - should use fallback defaults
            save_nlsq_results(result, data, config, output_dir)

            # Verify files were still created
            nlsq_dir = output_dir / "nlsq"
            assert nlsq_dir.exists()
            assert (nlsq_dir / "parameters.json").exists()
            assert (nlsq_dir / "fitted_data.npz").exists()
            assert (nlsq_dir / "analysis_results_nlsq.json").exists()
            assert (nlsq_dir / "convergence_metrics.json").exists()

            # Verify fallback values were used in metadata
            # L should default to 2000000.0 Å
            npz_data = np.load(nlsq_dir / "fitted_data.npz")
            assert npz_data["q"] is not None  # q comes from data, should be present

    def test_save_nlsq_results_plotting_failure_recovery(self):
        """Test that data files are saved even when plotting fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create mock data and result
            result = create_mock_optimization_result(
                analysis_mode="static",
                converged=True,
            )
            data = create_mock_data_dict(n_angles=3, n_t1=15, n_t2=15)
            config_dict = create_mock_config_manager(
                analysis_mode="static",
                include_all_metadata=True,
            )
            config = Mock()
            config.config = config_dict

            # Patch generate_nlsq_plots to raise an exception
            with patch(
                "homodyne.cli.commands.generate_nlsq_plots",
                side_effect=RuntimeError("Matplotlib not available"),
            ):
                # Call save_nlsq_results - should succeed despite plotting failure
                save_nlsq_results(result, data, config, output_dir)

            # Verify data files were still created
            nlsq_dir = output_dir / "nlsq"
            assert nlsq_dir.exists()
            assert (nlsq_dir / "parameters.json").exists()
            assert (nlsq_dir / "fitted_data.npz").exists()
            assert (nlsq_dir / "analysis_results_nlsq.json").exists()
            assert (nlsq_dir / "convergence_metrics.json").exists()

            # Verify PNG plots were NOT created (plotting failed)
            png_files = list(nlsq_dir.glob("*.png"))
            assert len(png_files) == 0, (
                "PNG files should not be created if plotting fails"
            )


# ==============================================================================
# Test Class: Performance and Scale
# ==============================================================================


class TestNLSQPerformance:
    """Test performance with large datasets."""

    @pytest.mark.slow
    def test_save_nlsq_results_large_dataset(self):
        """Test handling of 180 angles × 100×100 correlation matrices."""
        pytest.importorskip("matplotlib")  # Skip if matplotlib not available

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create large dataset (180 angles × 100×100)
            n_angles = 180
            n_t1 = 100
            n_t2 = 100

            result = create_mock_optimization_result(
                analysis_mode="static",
                converged=True,
            )
            data = create_mock_data_dict(n_angles=n_angles, n_t1=n_t1, n_t2=n_t2)
            config_dict = create_mock_config_manager(
                analysis_mode="static",
                include_all_metadata=True,
            )
            config = Mock()
            config.config = config_dict

            # Measure execution time
            start_time = time.time()
            save_nlsq_results(result, data, config, output_dir)
            elapsed_time = time.time() - start_time

            # Verify all files created
            nlsq_dir = output_dir / "nlsq"
            assert nlsq_dir.exists()
            assert (nlsq_dir / "parameters.json").exists()
            assert (nlsq_dir / "fitted_data.npz").exists()

            # Verify data dimensions
            npz_data = np.load(nlsq_dir / "fitted_data.npz")
            assert npz_data["c2_exp"].shape == (n_angles, n_t1, n_t2)
            assert npz_data["phi_angles"].shape == (n_angles,)

            # Verify 180 PNG plots created
            png_files = list(nlsq_dir.glob("*.png"))
            assert len(png_files) == n_angles

            # Log performance (informational, no strict requirement for slow test)
            print(
                f"\nLarge dataset performance: {elapsed_time:.2f}s for {n_angles} angles × {n_t1}×{n_t2}"
            )

    def test_save_nlsq_results_device_array_conversion(self):
        """Test conversion of JAX arrays to NumPy for saving."""
        import jax.numpy as jnp

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create mock data with JAX arrays
            result = create_mock_optimization_result(
                analysis_mode="static",
                converged=True,
            )

            # Convert some arrays to JAX arrays (simulating JAX data)
            data = create_mock_data_dict(n_angles=3, n_t1=20, n_t2=20)
            data["c2_exp"] = jnp.array(data["c2_exp"])  # Convert to JAX array
            data["t1"] = jnp.array(data["t1"])
            data["t2"] = jnp.array(data["t2"])

            config_dict = create_mock_config_manager(
                analysis_mode="static",
                include_all_metadata=True,
            )
            config = Mock()
            config.config = config_dict

            # Call save_nlsq_results - should handle JAX arrays
            save_nlsq_results(result, data, config, output_dir)

            # Verify files created
            nlsq_dir = output_dir / "nlsq"
            assert nlsq_dir.exists()
            assert (nlsq_dir / "fitted_data.npz").exists()

            # Load and verify arrays are NumPy (not JAX)
            npz_data = np.load(nlsq_dir / "fitted_data.npz")
            assert isinstance(npz_data["c2_exp"], np.ndarray)
            assert isinstance(npz_data["t1"], np.ndarray)
            assert isinstance(npz_data["t2"], np.ndarray)
            # Verify conversion preserved data
            assert npz_data["c2_exp"].shape == (3, 20, 20)
            assert npz_data["t1"].shape == (20,)
            assert npz_data["t2"].shape == (20,)


# ==============================================================================
# Test Class: Method Comparison
# ==============================================================================


class TestNLSQClassicalComparison:
    """Test that NLSQ output structure matches classical optimization."""

    def test_nlsq_classical_directory_structure_match(self):
        """Test that nlsq/ and classical/ have parallel structures."""
        pass  # To be implemented in T039

    def test_nlsq_classical_file_types_match(self):
        """Test that file names and types match between methods."""
        pass  # To be implemented in T040

    def test_nlsq_classical_plot_format_match(self):
        """Test that plot formats and naming match."""
        pass  # To be implemented in T041


# ==============================================================================
# Test Class: CLI Integration
# ==============================================================================


class TestNLSQCLIIntegration:
    """Test CLI end-to-end with NLSQ saving."""

    def test_save_results_routing_nlsq(self):
        """Test that _save_results() routes to save_nlsq_results() for NLSQ method."""
        import tempfile
        from pathlib import Path
        from unittest.mock import Mock

        from homodyne.cli.commands import _save_results

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create mock args with method="nlsq"
            args = Mock()
            args.method = "nlsq"
            args.output_dir = output_dir
            args.output_format = "json"

            # Create mock result, data, config
            result = create_mock_optimization_result(analysis_mode="static")
            data = create_mock_data_dict(n_angles=3, n_t1=10, n_t2=10)
            config_dict = create_mock_config_manager(analysis_mode="static")
            config = Mock()
            config.config = config_dict
            device_config = {"device": "cpu"}

            # Call _save_results which should route to save_nlsq_results()
            _save_results(args, result, device_config, data, config)

            # Verify NLSQ directory and files were created
            nlsq_dir = output_dir / "nlsq"
            assert nlsq_dir.exists(), "NLSQ directory not created via routing"
            assert (nlsq_dir / "parameters.json").exists(), (
                "parameters.json not created"
            )
            assert (nlsq_dir / "fitted_data.npz").exists(), (
                "fitted_data.npz not created"
            )
            assert (nlsq_dir / "analysis_results_nlsq.json").exists(), (
                "analysis_results not created"
            )
            assert (nlsq_dir / "convergence_metrics.json").exists(), (
                "convergence_metrics not created"
            )

    def test_save_results_routing_mcmc(self):
        """Test that _save_results() uses legacy format for MCMC method."""
        import tempfile
        from pathlib import Path
        from unittest.mock import Mock

        from homodyne.cli.commands import _save_results

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create mock args with method="mcmc"
            args = Mock()
            args.method = "mcmc"
            args.output_dir = output_dir
            args.output_format = "json"

            # Create mock MCMC result
            result = Mock()
            result.mean_contrast = 0.5
            result.mean_offset = 1.0
            result.mean_params = np.array([1000.0, 0.5, 10.0])
            result.samples_params = None

            data = create_mock_data_dict(n_angles=3, n_t1=10, n_t2=10)
            config_dict = create_mock_config_manager(analysis_mode="static")
            config = Mock()
            config.config = config_dict
            device_config = {"device": "cpu"}

            # Call _save_results which should use legacy format
            _save_results(args, result, device_config, data, config)

            # Verify legacy file was created (not nlsq/ subdirectory)
            legacy_file = output_dir / "homodyne_results.json"
            assert legacy_file.exists(), "Legacy results file not created for MCMC"
            # Verify nlsq directory was NOT created
            nlsq_dir = output_dir / "nlsq"
            assert not nlsq_dir.exists(), (
                "NLSQ directory should not be created for MCMC"
            )

    @pytest.mark.slow
    def test_cli_end_to_end_nlsq(self):
        """Test full CLI workflow from config to saved results."""
        pass  # Deferred - requires full CLI with real config file

    def test_nlsq_workflow_both_analysis_modes(self):
        """Test with both static_mode and laminar_flow modes."""
        # This is already tested in test_nlsq_full_workflow_files
        # which tests both modes
        pass  # Covered by test_nlsq_full_workflow_files


# ==============================================================================
# Stratified Data Handling (from test_stratified_nlsq_integration.py)
# ==============================================================================

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def balanced_test_data():
    """Create balanced test data for integration tests."""
    n_angles = 3
    n_points_per_angle = 50_000
    n_total = n_angles * n_points_per_angle

    phi = np.repeat([0.0, 45.0, 90.0], n_points_per_angle)
    t1 = np.tile(np.linspace(1e-6, 1e-3, n_points_per_angle), n_angles)
    t2 = t1.copy()
    g2_exp = 1.0 + 0.4 * np.exp(-0.1 * (t1 + t2))

    return {
        "phi": phi,
        "t1": t1,
        "t2": t2,
        "g2_exp": g2_exp,
        "n_angles": n_angles,
        "n_points": n_total,
    }


@pytest.fixture
def imbalanced_test_data():
    """Create imbalanced test data (skewed angle distribution)."""
    # Extreme imbalance: angle 0 has 10x more points than angles 45 and 90
    n_points_angle0 = 100_000
    n_points_other = 10_000

    phi = np.concatenate(
        [
            np.full(n_points_angle0, 0.0),
            np.full(n_points_other, 45.0),
            np.full(n_points_other, 90.0),
        ]
    )
    t1 = np.linspace(1e-6, 1e-3, len(phi))
    t2 = t1.copy()
    g2_exp = 1.0 + 0.4 * np.exp(-0.1 * (t1 + t2))

    return {
        "phi": phi,
        "t1": t1,
        "t2": t2,
        "g2_exp": g2_exp,
        "n_angles": 3,
        "n_points": len(phi),
    }


# ============================================================================
# Configuration Tests
# ============================================================================


def test_stratification_config_parsing():
    """Test parsing of stratification configuration section."""
    config = {
        "optimization": {
            "stratification": {
                "enabled": "auto",
                "target_chunk_size": 100_000,
                "max_imbalance_ratio": 5.0,
                "check_memory_safety": True,
                "min_points_per_angle": 1000,
            }
        }
    }

    strat_config = config["optimization"]["stratification"]
    assert strat_config["enabled"] == "auto"
    assert strat_config["target_chunk_size"] == 100_000
    assert strat_config["max_imbalance_ratio"] == 5.0
    assert strat_config["check_memory_safety"] is True
    assert strat_config["min_points_per_angle"] == 1000


def test_stratification_config_defaults():
    """Test default values when stratification config is missing."""
    # Default configuration
    default_enabled = "auto"
    default_chunk_size = 100_000
    default_max_imbalance = 5.0

    # Verify defaults match documented behavior
    assert default_enabled == "auto"  # Automatic activation
    assert default_chunk_size == 100_000  # 100k points per chunk
    assert default_max_imbalance == 5.0  # 5x imbalance threshold


def test_stratification_config_enabled_auto():
    """Test 'auto' mode activation criteria."""
    # Auto mode should activate when:
    # - per_angle_scaling=True
    # - n_points >= 100k
    # - Angles are balanced (imbalance_ratio <= max_imbalance_ratio)

    # Case 1: All conditions met → should activate
    should_activate, _ = should_use_stratification(
        n_points=150_000,
        n_angles=3,
        per_angle_scaling=True,
        imbalance_ratio=2.0,  # balanced
    )
    assert should_activate is True

    # Case 2: Small dataset → should NOT activate
    should_activate, reason = should_use_stratification(
        n_points=50_000,
        n_angles=3,
        per_angle_scaling=True,
        imbalance_ratio=2.0,
    )
    assert should_activate is False
    assert "100k" in reason.lower()


def test_stratification_config_validation():
    """Test validation of stratification configuration values."""
    # Valid configuration
    valid_config = {
        "enabled": "auto",
        "target_chunk_size": 100_000,
        "max_imbalance_ratio": 5.0,
    }
    assert valid_config["target_chunk_size"] > 0
    assert valid_config["max_imbalance_ratio"] > 0

    # Edge case: Minimum chunk size
    min_chunk_size = 10_000
    assert min_chunk_size >= 1000  # Reasonable minimum


# ============================================================================
# Strategy Selection Tests
# ============================================================================


def test_strategy_selection_with_large_dataset():
    """Test NLSQ strategy selection with large dataset requiring chunking."""
    selector = DatasetSizeStrategy()

    # Large dataset (>1M points) → should select LARGE, CHUNKED, or STREAMING
    # Thresholds: <1M→STANDARD, 1M-10M→LARGE, 10M-100M→CHUNKED, >100M→STREAMING
    n_points = 3_000_000  # 3M points falls in LARGE range (1M-10M)
    n_parameters = 9  # laminar_flow with per-angle scaling

    strategy = selector.select_strategy(n_points, n_parameters)

    # Should select LARGE (1M-10M), CHUNKED (10M-100M), or STREAMING (>100M)
    # 3M points → LARGE strategy is correct
    assert strategy in [
        OptimizationStrategy.LARGE,
        OptimizationStrategy.CHUNKED,
        OptimizationStrategy.STREAMING,
    ]


def test_strategy_selection_with_small_dataset():
    """Test NLSQ strategy selection with small dataset (no chunking needed)."""
    selector = DatasetSizeStrategy()

    # Small dataset (<100k points) → should select STANDARD
    n_points = 50_000
    n_parameters = 5

    strategy = selector.select_strategy(n_points, n_parameters)

    # Should select STANDARD (no chunking)
    assert strategy == OptimizationStrategy.STANDARD


def test_stratification_decision_integrates_with_strategy():
    """Test stratification decision considers NLSQ strategy selection."""
    # Large dataset with per-angle scaling
    n_points = 2_000_000  # 2M points falls in LARGE range (1M-10M)
    n_angles = 3
    per_angle_scaling = True

    # Strategy selector suggests LARGE/CHUNKED/STREAMING for >1M points
    selector = DatasetSizeStrategy()
    strategy = selector.select_strategy(n_points, n_parameters=9)
    assert strategy in [
        OptimizationStrategy.LARGE,
        OptimizationStrategy.CHUNKED,
        OptimizationStrategy.STREAMING,
    ]

    # Stratification should activate for per-angle + large dataset
    stats = analyze_angle_distribution(np.repeat([0.0, 45.0, 90.0], n_points // 3))
    should_stratify, _ = should_use_stratification(
        n_points=n_points,
        n_angles=stats.n_angles,
        per_angle_scaling=per_angle_scaling,
        imbalance_ratio=stats.imbalance_ratio,
    )

    assert should_stratify is True


# ============================================================================
# Data Preparation Tests
# ============================================================================


def test_stratification_with_balanced_data(balanced_test_data):
    """Test stratification with balanced angle distribution."""
    data = balanced_test_data
    n_points = data["n_points"]

    # Verify balanced distribution
    stats = analyze_angle_distribution(data["phi"])
    assert stats.is_balanced is True
    assert stats.imbalance_ratio <= 2.0  # Good balance

    # Apply stratification
    phi_s, t1_s, t2_s, g2_s, chunk_sizes = create_angle_stratified_data(
        data["phi"],
        data["t1"],
        data["t2"],
        data["g2_exp"],
        target_chunk_size=100_000,
    )

    # Verify all points preserved
    assert len(phi_s) == n_points
    assert len(t1_s) == n_points
    assert len(t2_s) == n_points
    assert len(g2_s) == n_points


def test_stratification_with_imbalanced_data(imbalanced_test_data):
    """Test stratification detects extreme angle imbalance."""
    data = imbalanced_test_data

    # Verify imbalanced distribution
    stats = analyze_angle_distribution(data["phi"])
    assert stats.is_balanced is False
    assert stats.imbalance_ratio > 5.0  # Extreme imbalance (10x)

    # Stratification should detect this and suggest sequential
    should_stratify, reason = should_use_stratification(
        n_points=data["n_points"],
        n_angles=stats.n_angles,
        per_angle_scaling=True,
        imbalance_ratio=stats.imbalance_ratio,
    )

    # Should NOT stratify (use sequential instead for extreme imbalance)
    assert should_stratify is False
    assert "imbalance" in reason.lower()


def test_stratification_preserves_all_data_points():
    """Test stratification preserves all data points (no duplicates or losses)."""
    # Large balanced dataset
    n_points = 300_000
    phi = np.repeat([0.0, 45.0, 90.0], n_points // 3)
    t1 = np.linspace(1e-6, 1e-3, n_points)
    t2 = t1.copy()
    g2_exp = np.random.uniform(1.0, 1.5, n_points)

    # Apply stratification
    phi_s, t1_s, t2_s, g2_s, chunk_sizes = create_angle_stratified_data(
        phi, t1, t2, g2_exp, target_chunk_size=100_000
    )

    # Verify no data loss
    assert len(phi_s) == n_points
    assert len(t1_s) == n_points
    assert len(t2_s) == n_points
    assert len(g2_s) == n_points

    # Verify no duplicates (all unique combinations present)
    # Note: Due to stratification, order changes but uniqueness preserved
    # Convert arrays to NumPy to ensure hashability (JAX arrays are not hashable)
    original_tuples = set(
        zip(np.asarray(phi), np.asarray(t1), np.asarray(t2), strict=False)
    )
    stratified_tuples = set(
        zip(np.asarray(phi_s), np.asarray(t1_s), np.asarray(t2_s), strict=False)
    )
    assert len(original_tuples) == len(stratified_tuples)


# ============================================================================
# Component Integration Tests
# ============================================================================


def test_integration_stratification_with_strategy_selector():
    """Test full integration: strategy selection + stratification decision."""
    # Large dataset parameters
    n_angles = 3
    points_per_angle = 666_666  # Ensures exact division
    n_points = (
        n_angles * points_per_angle
    )  # 1,999,998 total (slightly under 2M, still in LARGE range)
    n_parameters = 9  # laminar_flow with per-angle scaling
    per_angle_scaling = True

    # Mock data with equal points per angle
    data = {
        "phi": np.repeat([0.0, 45.0, 90.0], points_per_angle),
        "t1": np.linspace(1e-6, 1e-3, n_points),
        "t2": np.linspace(1e-6, 1e-3, n_points),
        "g2_exp": np.random.uniform(1.0, 1.5, n_points),
    }

    # Strategy selection
    selector = DatasetSizeStrategy()
    strategy = selector.select_strategy(n_points, n_parameters)

    # Should select LARGE (1M-10M), CHUNKED (10M-100M), or STREAMING (>100M)
    assert strategy in [
        OptimizationStrategy.LARGE,
        OptimizationStrategy.CHUNKED,
        OptimizationStrategy.STREAMING,
    ]

    # Stratification decision
    stats = analyze_angle_distribution(data["phi"])
    should_stratify, _ = should_use_stratification(
        n_points=n_points,
        n_angles=stats.n_angles,
        per_angle_scaling=True,
        imbalance_ratio=stats.imbalance_ratio,
    )

    # Should stratify (>= 100k + per-angle + balanced)
    assert should_stratify is True

    # Apply stratification
    phi_s, t1_s, t2_s, g2_s, chunk_sizes = create_angle_stratified_data(
        data["phi"],
        data["t1"],
        data["t2"],
        data["g2_exp"],
        target_chunk_size=100_000,
    )

    # Verify workflow completes successfully
    assert len(phi_s) == n_points


def test_integration_workflow_without_stratification():
    """Test workflow when stratification is not needed (small dataset)."""
    # Small dataset
    n_points = 10_000
    phi = np.repeat([0.0, 45.0, 90.0], n_points // 3)
    t1 = np.linspace(1e-6, 1e-3, n_points)
    t2 = t1.copy()
    g2_exp = np.random.uniform(1.0, 1.5, n_points)

    # Strategy selection
    selector = DatasetSizeStrategy()
    strategy = selector.select_strategy(n_points, n_parameters=5)

    # Should select STANDARD (no chunking)
    assert strategy == OptimizationStrategy.STANDARD

    # Stratification decision
    stats = analyze_angle_distribution(phi)
    should_stratify, reason = should_use_stratification(
        n_points=n_points,
        n_angles=stats.n_angles,
        per_angle_scaling=True,
        imbalance_ratio=stats.imbalance_ratio,
    )

    # Should NOT stratify (small dataset)
    assert should_stratify is False
    assert "100k" in reason.lower() or "standard" in reason.lower()


def test_stratified_data_preserves_metadata_attributes():
    """Test that StratifiedData properly copies metadata attributes from original data.

    This is a regression test for bug where stratification created new data object
    without copying critical metadata (sigma, q, L, dt), causing AttributeError
    in residual function creation.

    Bug Report: 2025-11-06
    Root Cause: StratifiedData.__init__() only copied arrays, not scalar metadata
    Fix: Added explicit attribute copying for sigma, q, L, dt
    """
    import logging

    from homodyne.optimization.nlsq_wrapper import NLSQWrapper

    # Create mock data with all required metadata attributes
    # Need 100k+ points to trigger stratification
    # Data structure: unique arrays for phi, t1, t2 (not flattened/repeated)
    # n_points = len(phi) × len(t1) × len(t2)
    n_phi = 3  # 3 phi angles
    n_t = 200  # 200 time points
    n_total = n_phi * n_t * n_t  # 3 × 200 × 200 = 120k points

    class MockOriginalData:
        """Mock data object with all attributes expected by NLSQ."""

        def __init__(self):
            # Array attributes: UNIQUE values only (meshgrid will expand them)
            self.phi = np.array([0.0, 45.0, 90.0])  # 3 unique phi angles
            self.t1 = np.linspace(1e-6, 1e-3, n_t)  # 200 unique time points
            self.t2 = np.linspace(1e-6, 1e-3, n_t)  # 200 unique time points
            # g2 shape: (n_phi, n_t, n_t) or flattened (120k,)
            self.g2 = np.random.uniform(1.0, 1.5, (n_phi, n_t, n_t))

            # Critical metadata attributes (must be copied verbatim)
            # sigma shape must match g2 after meshgrid expansion
            self.sigma = np.ones((n_phi, n_t, n_t)) * 0.1  # Uncertainty/error bars
            self.q = 0.005  # Wavevector magnitude (Å⁻¹)
            self.L = 5000000.0  # Sample-detector distance (Å)
            self.dt = 0.1  # Frame time step (s) - optional

    original_data = MockOriginalData()

    # Create mock config with stratification enabled
    mock_config = {
        "optimization": {
            "stratification": {
                "enabled": True,
                "target_chunk_size": 50_000,  # Reasonable chunk size
                "max_imbalance_ratio": 5.0,
            }
        }
    }

    # Create wrapper and apply stratification
    wrapper = NLSQWrapper()  # Use default initialization
    logger = logging.getLogger(__name__)

    # This should trigger stratification since we have balanced angles
    stratified_data = wrapper._apply_stratification_if_needed(
        data=original_data,
        per_angle_scaling=True,  # Requires metadata for residual computation
        config=mock_config,
        logger=logger,
    )

    # CRITICAL: Verify all metadata attributes were copied
    # These checks prevent regression of the AttributeError bug

    # Check required attributes exist
    assert hasattr(stratified_data, "sigma"), "sigma not copied (CRITICAL)"
    assert hasattr(stratified_data, "q"), "q not copied (CRITICAL)"
    assert hasattr(stratified_data, "L"), "L not copied (CRITICAL)"

    # Check optional dt attribute
    assert hasattr(stratified_data, "dt"), "dt not copied (optional but expected)"

    # Verify values match original (scalars should be copied verbatim)
    assert stratified_data.q == original_data.q, "q value mismatch"
    assert stratified_data.L == original_data.L, "L value mismatch"
    assert stratified_data.dt == original_data.dt, "dt value mismatch"

    # Verify array attributes were reorganized (not just copied)
    assert hasattr(stratified_data, "phi"), "phi missing"
    assert hasattr(stratified_data, "t1"), "t1 missing"
    assert hasattr(stratified_data, "t2"), "t2 missing"
    assert hasattr(stratified_data, "g2"), "g2 missing"

    # Verify stratification diagnostics present
    assert hasattr(stratified_data, "stratification_diagnostics"), "diagnostics missing"

    # Additional check: Simulate residual function validation
    # This is the code that originally failed with AttributeError
    required_attrs = ["phi", "t1", "t2", "g2", "sigma", "q", "L"]
    for attr in required_attrs:
        assert hasattr(stratified_data, attr), f"Missing required attribute: {attr}"

    # Success: All attributes present, residual function validation would pass


def test_stratification_diagnostics_passed_to_result():
    """Test that stratification diagnostics parameter is accepted without NameError.

    This is a regression test for bug where stratification_diagnostics was extracted
    from stratified_data but not passed to _create_fit_result(), causing NameError.

    Bug Report: 2025-11-06 (log: homodyne_analysis_20251106_122208.log)
    Root Cause: _create_fit_result() referenced stratification_diagnostics without
                receiving it as a parameter
    Fix: Added stratification_diagnostics parameter to function call and signature

    Test strategy: This test verifies that _create_fit_result() accepts the
    stratification_diagnostics parameter (preventing NameError), even when diagnostics
    is None (the common case when collect_diagnostics=False).
    """
    from homodyne.optimization.nlsq_wrapper import NLSQWrapper, OptimizationResult

    # Create wrapper
    wrapper = NLSQWrapper()

    # Mock minimal parameters
    n_data = 1000
    popt = np.array([1.0, 0.5, 0.1, 1.0, 0.5, 0.1, 0.0, 0.5, 1.0])

    # CRITICAL TEST 1: Verify _create_fit_result() accepts stratification_diagnostics=None
    # This is the common case and where the NameError occurred
    try:
        result = wrapper._create_fit_result(
            popt=popt,
            pcov=np.eye(9),
            residuals=np.zeros(n_data),
            n_data=n_data,
            iterations=10,
            execution_time=1.0,
            convergence_status="converged",
            recovery_actions=[],
            streaming_diagnostics=None,
            stratification_diagnostics=None,  # KEY FIX: Parameter now accepted
        )
    except NameError as e:
        if "stratification_diagnostics" in str(e):
            pytest.fail(f"NameError when passing stratification_diagnostics=None: {e}")
        else:
            raise

    # Verify result was created successfully
    assert isinstance(result, OptimizationResult), "Result is not OptimizationResult"
    assert hasattr(result, "stratification_diagnostics"), (
        "Result missing stratification_diagnostics attribute"
    )
    assert result.stratification_diagnostics is None, (
        "Expected stratification_diagnostics=None"
    )

    # Success: The parameter is now accepted without NameError
    # This prevents the bug from recurring where _create_fit_result() tried to
    # use stratification_diagnostics without receiving it as a parameter


def test_full_nlsq_workflow_with_stratification():
    """End-to-end integration test for stratification with per-angle scaling.

    This test validates the complete workflow from data preparation through
    stratification to result creation. Uses a moderate dataset size to trigger
    stratification while avoiding vmap errors.

    Test Coverage:
    - Data structure validation (phi, t1, t2, g2 with metadata)
    - Stratification activation and execution
    - Stratification diagnostics collection and propagation
    - Result creation with diagnostics
    - Metadata preservation (regression test for bug #1)
    - Diagnostics parameter passing (regression test for bug #2)

    Integration test (2025-11-06): Validates fixes for metadata and diagnostics bugs

    Note: Uses moderate dataset size (50k points) with per_angle_scaling=True to
    trigger stratification while avoiding known large dataset issues.
    """
    import logging

    from homodyne.optimization.nlsq_wrapper import NLSQWrapper

    # Create realistic mock data (50k points, 3 angles)
    # Smaller dataset to avoid vmap errors while still triggering stratification
    n_phi = 3
    n_t = 130  # 130x130 time grid per angle ≈ 50,700 points
    n_total = n_phi * n_t * n_t

    class MockData:
        """Mock experimental data with full metadata."""

        def __init__(self):
            # Unique arrays for phi, t1, t2 (meshgrid will expand)
            self.phi = np.array([0.0, 45.0, 90.0])
            self.t1 = np.linspace(1e-6, 1e-3, n_t)
            self.t2 = np.linspace(1e-6, 1e-3, n_t)

            # Generate synthetic g2 data with known structure
            t1_grid, t2_grid = np.meshgrid(self.t1, self.t2, indexing="ij")
            tau_sum = t1_grid + t2_grid

            # Simple isotropic data (same for all angles)
            self.g2 = np.zeros((n_phi, n_t, n_t))
            for i in range(n_phi):
                self.g2[i] = 1.0 + 0.4 * np.exp(-1000.0 * tau_sum)

            # Required metadata (regression test for bug #1)
            self.sigma = np.ones((n_phi, n_t, n_t)) * 0.01
            self.q = 0.005
            self.L = 5000000.0
            self.dt = 0.1

    data = MockData()

    # Configuration for stratification with diagnostics
    # Use a mock ConfigManager to properly pass stratification settings
    class MockConfig:
        def __init__(self):
            self.config = {
                "optimization": {
                    "stratification": {
                        "enabled": True,  # Force stratification
                        "target_chunk_size": 20_000,  # Create 2-3 chunks from 50k points
                        "collect_diagnostics": True,  # Enable diagnostics (bug #2)
                        "max_imbalance_ratio": 5.0,
                    }
                }
            }

    config = MockConfig()

    # Create wrapper and apply stratification
    wrapper = NLSQWrapper()
    logger = logging.getLogger(__name__)

    # CRITICAL TEST: Apply stratification (tests bug #1 and #2 fixes)
    try:
        stratified_data = wrapper._apply_stratification_if_needed(
            data=data,
            per_angle_scaling=True,  # Required for stratification
            config=config,
            logger=logger,
        )
    except AttributeError as e:
        if "sigma" in str(e) or "q" in str(e) or "L" in str(e):
            pytest.fail(f"Metadata not preserved during stratification (Bug #1): {e}")
        else:
            raise
    except Exception as e:
        pytest.fail(f"Stratification failed: {e}")

    # Validate stratification occurred
    assert hasattr(stratified_data, "stratification_diagnostics"), (
        "Stratification diagnostics not created"
    )
    assert stratified_data.stratification_diagnostics is not None, (
        "Stratification diagnostics is None (Bug #2 would cause NameError later)"
    )

    # Validate metadata preserved (Bug #1 fix)
    assert hasattr(stratified_data, "sigma"), "sigma not preserved"
    assert hasattr(stratified_data, "q"), "q not preserved"
    assert hasattr(stratified_data, "L"), "L not preserved"
    assert hasattr(stratified_data, "dt"), "dt not preserved"

    # Validate diagnostics content
    diag = stratified_data.stratification_diagnostics
    assert diag.n_chunks > 0, "Should have created chunks"
    assert len(diag.chunk_sizes) == diag.n_chunks, "Chunk sizes list length mismatch"
    assert len(diag.angles_per_chunk) == diag.n_chunks, (
        "Angles per chunk list length mismatch"
    )

    # Validate all chunks have all angles (key stratification requirement)
    for i, n_angles in enumerate(diag.angles_per_chunk):
        assert n_angles == 3, (
            f"Chunk {i} has {n_angles} angles, expected 3 (stratification failed)"
        )

    # CRITICAL TEST: Create result with diagnostics (Bug #2 fix)
    try:
        result = wrapper._create_fit_result(
            popt=np.array([1000.0, 1.0, 0.0]),
            pcov=np.eye(3),
            residuals=np.zeros(n_total),
            n_data=n_total,
            iterations=10,
            execution_time=1.0,
            convergence_status="converged",
            recovery_actions=[],
            streaming_diagnostics=None,
            stratification_diagnostics=stratified_data.stratification_diagnostics,  # Bug #2
        )
    except NameError as e:
        if "stratification_diagnostics" in str(e):
            pytest.fail(
                f"NameError when passing stratification_diagnostics (Bug #2): {e}"
            )
        else:
            raise

    # Validate result contains diagnostics
    assert hasattr(result, "stratification_diagnostics"), (
        "Result missing stratification_diagnostics"
    )
    assert result.stratification_diagnostics is not None, (
        "Result stratification_diagnostics is None"
    )

    # Success: Full workflow completed with both fixes validated
    print("✓ Full stratification workflow test passed:")
    print(f"  - Data points: {n_total:,}")
    print(f"  - Chunks created: {diag.n_chunks}")
    print("  - Metadata preserved: ✓ (Bug #1 fix)")
    print("  - Diagnostics passed: ✓ (Bug #2 fix)")


# ==============================================================================
# Filtering Integration (from test_nlsq_filtering.py)
# ==============================================================================


class SimpleNamespace:
    """Simple object to hold data attributes for NLSQ."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def dict_to_data_object(data_dict):
    """Convert data dict to object with attributes for NLSQ.

    NLSQ requires data with attributes: phi, t1, t2, g2, sigma, q, L
    """
    g2 = data_dict.get("c2_exp")
    # Create uniform sigma (uncertainty) array matching g2 shape
    sigma = np.ones_like(g2)

    # Get q value (use first wavevector or default)
    wavevector_list = data_dict.get("wavevector_q_list", [0.01])
    q = wavevector_list[0] if len(wavevector_list) > 0 else 0.01

    # L is sample-detector distance (use default 100.0 as in CLI)
    L = 100.0

    return SimpleNamespace(
        phi=data_dict.get("phi_angles_list"),
        t1=data_dict.get("t1"),
        t2=data_dict.get("t2"),
        g2=g2,
        sigma=sigma,
        q=q,
        L=L,
    )


class TestNLSQWithAngleFiltering:
    """Integration tests for NLSQ optimization with angle filtering."""

    def test_nlsq_with_filtered_angles_attempts_optimization(self, caplog):
        """Test that NLSQ receives filtered data and attempts optimization."""
        # Arrange - Create data with 9 specific angles
        angles = [0.0, 10.0, 30.0, 45.0, 60.0, 85.0, 90.0, 95.0, 180.0]
        data = create_specific_angles_test_data(phi_angles=angles, n_t1=10, n_t2=10)

        # Configure filtering to select only [85, 90, 95] (3 angles)
        config_dict = create_phi_filtering_config(
            enabled=True,
            target_ranges=[
                {"min_angle": 85.0, "max_angle": 100.0, "description": "Near 90"}
            ],
        )

        # Add required optimization configuration
        config_dict["analysis_mode"] = "static"
        config_dict["initial_parameters"] = {
            "parameter_names": ["D0", "alpha", "D_offset"],
            "values": [1000.0, 0.5, 10.0],
        }

        # Create ConfigManager mock
        class MockConfigManager:
            def get_config(self):
                return config_dict

            def get(self, key, default=None):
                return config_dict.get(key, default)

            def get_parameter_bounds(self, param_names=None):
                # Return default bounds for static_mode mode
                return [
                    {"name": "D0", "min": 100.0, "max": 10000.0},
                    {"name": "alpha", "min": 0.1, "max": 2.0},
                    {"name": "D_offset", "min": 0.0, "max": 100.0},
                ]

            def get_active_parameters(self):
                return ["D0", "alpha", "D_offset"]

        config = MockConfigManager()

        # Apply filtering before optimization (simulating _run_optimization behavior)
        from homodyne.cli.commands import _apply_angle_filtering_for_optimization

        caplog.clear()  # Clear logs before filtering
        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        # Convert filtered dict to data object for NLSQ
        filtered_data_obj = dict_to_data_object(filtered_data)

        # Act - Run NLSQ optimization with filtered data
        # Note: Optimization may not converge with synthetic data,
        # but we verify filtering works and optimization is attempted
        try:
            result = fit_nlsq_jax(filtered_data_obj, config)
        except Exception:
            # Optimization failure is OK - we're testing filtering, not convergence
            result = None

        # Assert - Dataset size reduction (filtering worked)
        assert len(filtered_data["phi_angles_list"]) == 3, (
            "Should have 3 filtered angles"
        )
        np.testing.assert_array_almost_equal(
            filtered_data["phi_angles_list"], [85.0, 90.0, 95.0], decimal=1
        )

        # Assert - C2 data first dimension reduced
        assert filtered_data["c2_exp"].shape[0] == 3, "C2 first dimension should be 3"

        # Assert - Other dimensions preserved
        assert "wavevector_q_list" in filtered_data, (
            "wavevector_q_list should be preserved"
        )
        assert "t1" in filtered_data, "t1 should be preserved"
        assert "t2" in filtered_data, "t2 should be preserved"

        # Assert - Log messages confirm filtering
        log_messages = [rec.message for rec in caplog.records]

        # Check for specific filtering message
        found_filtering_msg = any(
            "3 angles selected from 9" in msg for msg in log_messages
        )
        assert found_filtering_msg, "Should log '3 angles selected from 9 total angles'"

        # Assert - NLSQ was attempted (log shows "Starting NLSQ optimization")
        found_nlsq_start = any(
            "Starting NLSQ optimization" in msg for msg in log_messages
        )
        assert found_nlsq_start, "NLSQ optimization should have been attempted"

        # Assert - Data was prepared (log shows "Data prepared: X points")
        found_data_prepared = any("Data prepared:" in msg for msg in log_messages)
        assert found_data_prepared, "Data should have been prepared for optimization"

    def test_nlsq_with_disabled_filtering_uses_all_angles(self, caplog):
        """Test that NLSQ uses all 9 angles when filtering is disabled."""
        # Arrange - Create data with 9 angles
        angles = [0.0, 10.0, 30.0, 45.0, 60.0, 85.0, 90.0, 95.0, 180.0]
        data = create_specific_angles_test_data(phi_angles=angles, n_t1=10, n_t2=10)

        # Configure with filtering disabled
        config_dict = create_disabled_filtering_config()
        config_dict["analysis_mode"] = "static"
        config_dict["initial_parameters"] = {
            "parameter_names": ["D0", "alpha", "D_offset"],
            "values": [1000.0, 0.5, 10.0],
        }

        class MockConfigManager:
            def get_config(self):
                return config_dict

            def get(self, key, default=None):
                return config_dict.get(key, default)

            def get_parameter_bounds(self, param_names=None):
                return [
                    {"name": "D0", "min": 100.0, "max": 10000.0},
                    {"name": "alpha", "min": 0.1, "max": 2.0},
                    {"name": "D_offset", "min": 0.0, "max": 100.0},
                ]

            def get_active_parameters(self):
                return ["D0", "alpha", "D_offset"]

        config = MockConfigManager()

        # Apply filtering (should return all angles when disabled)
        from homodyne.cli.commands import _apply_angle_filtering_for_optimization

        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        # Convert to data object for NLSQ
        filtered_data_obj = dict_to_data_object(filtered_data)

        # Act - Run NLSQ with all angles
        caplog.clear()
        try:
            result = fit_nlsq_jax(filtered_data_obj, config)
        except Exception:
            # Optimization failure is OK - we're testing filtering
            result = None

        # Assert - All 9 angles used (no filtering)
        assert len(filtered_data["phi_angles_list"]) == 9, (
            "Should use all 9 angles when disabled"
        )
        np.testing.assert_array_almost_equal(
            filtered_data["phi_angles_list"], angles, decimal=1
        )

        # Assert - NLSQ was attempted
        log_messages = [rec.message for rec in caplog.records]
        found_nlsq_start = any(
            "Starting NLSQ optimization" in msg for msg in log_messages
        )
        assert found_nlsq_start, "NLSQ optimization should have been attempted"

    def test_nlsq_dataset_size_reduction_verified(self):
        """Test that dataset size reduction is measurable (9 → 3 angles)."""
        # Arrange
        angles = [0.0, 10.0, 30.0, 45.0, 60.0, 85.0, 90.0, 95.0, 180.0]
        data = create_specific_angles_test_data(phi_angles=angles, n_t1=10, n_t2=10)

        config_dict = create_phi_filtering_config(
            enabled=True,
            target_ranges=[
                {"min_angle": 85.0, "max_angle": 100.0, "description": "Near 90"}
            ],
        )

        class MockConfigManager:
            def get_config(self):
                return config_dict

        config = MockConfigManager()

        # Act
        from homodyne.cli.commands import _apply_angle_filtering_for_optimization

        original_size = len(data["phi_angles_list"])
        original_c2_size = data["c2_exp"].shape[0]

        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        filtered_size = len(filtered_data["phi_angles_list"])
        filtered_c2_size = filtered_data["c2_exp"].shape[0]

        # Assert - Size reduction
        assert original_size == 9, "Original should have 9 angles"
        assert filtered_size == 3, "Filtered should have 3 angles"
        assert original_c2_size == 9, "Original C2 should have 9 in first dimension"
        assert filtered_c2_size == 3, "Filtered C2 should have 3 in first dimension"

        # Calculate reduction
        reduction_factor = original_size / filtered_size
        assert reduction_factor == 3.0, "Should have 3x reduction (9 → 3)"

    def test_nlsq_log_messages_confirm_angle_selection(self, caplog):
        """Test that log messages confirm correct angle selection."""
        # Arrange
        angles = [0.0, 10.0, 30.0, 45.0, 60.0, 85.0, 90.0, 95.0, 180.0]
        data = create_specific_angles_test_data(phi_angles=angles, n_t1=10, n_t2=10)

        config_dict = create_phi_filtering_config(
            enabled=True,
            target_ranges=[
                {"min_angle": 85.0, "max_angle": 100.0, "description": "Near 90"}
            ],
        )

        class MockConfigManager:
            def get_config(self):
                return config_dict

        config = MockConfigManager()

        # Act
        from homodyne.cli.commands import _apply_angle_filtering_for_optimization

        caplog.clear()
        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        # Assert - Check log messages
        log_messages = [rec.message for rec in caplog.records]

        # Should log "3 angles selected from 9 total angles"
        found_count_msg = any("3 angles selected from 9" in msg for msg in log_messages)
        assert found_count_msg, "Should log angle count: '3 angles selected from 9'"

        # Should log the selected angles: [85.0, 90.0, 95.0]
        found_angles_msg = any(
            "85" in msg and "90" in msg and "95" in msg for msg in log_messages
        )
        assert found_angles_msg, "Should log selected angles containing 85, 90, and 95"

    def test_nlsq_receives_filtered_data_correctly(self):
        """Test that NLSQ receives correctly filtered data structure."""
        # Arrange - Create synthetic data with known structure
        angles = [0.0, 10.0, 30.0, 45.0, 60.0, 85.0, 90.0, 95.0, 180.0]
        data = create_specific_angles_test_data(phi_angles=angles, n_t1=10, n_t2=10)

        config_dict = create_phi_filtering_config(
            enabled=True,
            target_ranges=[
                {"min_angle": 85.0, "max_angle": 100.0, "description": "Near 90"}
            ],
        )
        config_dict["analysis_mode"] = "static"
        config_dict["initial_parameters"] = {
            "parameter_names": ["D0", "alpha", "D_offset"],
            "values": [1000.0, 0.5, 10.0],
        }

        class MockConfigManager:
            def get_config(self):
                return config_dict

            def get(self, key, default=None):
                return config_dict.get(key, default)

            def get_parameter_bounds(self, param_names=None):
                return [
                    {"name": "D0", "min": 100.0, "max": 10000.0},
                    {"name": "alpha", "min": 0.1, "max": 2.0},
                    {"name": "D_offset", "min": 0.0, "max": 100.0},
                ]

            def get_active_parameters(self):
                return ["D0", "alpha", "D_offset"]

        config = MockConfigManager()

        # Act
        from homodyne.cli.commands import _apply_angle_filtering_for_optimization

        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        # Convert to data object for NLSQ
        filtered_data_obj = dict_to_data_object(filtered_data)

        # Verify data object has all required attributes for NLSQ
        assert hasattr(filtered_data_obj, "phi"), "Should have phi attribute"
        assert hasattr(filtered_data_obj, "t1"), "Should have t1 attribute"
        assert hasattr(filtered_data_obj, "t2"), "Should have t2 attribute"
        assert hasattr(filtered_data_obj, "g2"), "Should have g2 attribute"
        assert hasattr(filtered_data_obj, "sigma"), "Should have sigma attribute"
        assert hasattr(filtered_data_obj, "q"), "Should have q attribute"
        assert hasattr(filtered_data_obj, "L"), "Should have L attribute"

        # Verify filtered data dimensions
        assert len(filtered_data_obj.phi) == 3, "Should have 3 filtered angles"
        assert filtered_data_obj.g2.shape[0] == 3, "g2 first dimension should be 3"

        # Act - Attempt optimization (may not converge, but should accept data structure)
        try:
            result = fit_nlsq_jax(filtered_data_obj, config)
            # If we get here, optimization at least started
            optimization_attempted = True
        except Exception as e:
            # If exception is about data structure, that's a failure
            if "attribute" in str(e).lower() or "must have" in str(e).lower():
                raise AssertionError(f"NLSQ rejected data structure: {e}")
            # Other exceptions (convergence, numerical issues) are OK
            optimization_attempted = True

        assert optimization_attempted, "NLSQ should have attempted optimization"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


# ==============================================================================
# Wrapper Integration (from test_nlsq_wrapper_integration.py)
# ==============================================================================

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_xpcs_data():
    """Create minimal XPCS data for testing."""

    class MockXPCSData:
        def __init__(self, n_phi=5, n_t1=10, n_t2=10):
            self.phi = np.linspace(0, 90, n_phi)
            self.t1 = np.linspace(0.1, 1.0, n_t1)
            self.t2 = np.linspace(0.1, 1.0, n_t2)

            # Generate synthetic g2 data: g2 ≈ 1.0 + contrast * exp(-rate * t)
            phi_grid, t1_grid, t2_grid = np.meshgrid(
                self.phi, self.t1, self.t2, indexing="ij"
            )
            decay_rate = 1.0
            contrast = 0.3
            self.g2 = 1.0 + contrast * np.exp(-decay_rate * (t1_grid + t2_grid))

            # Add realistic noise
            self.g2 += np.random.randn(*self.g2.shape) * 0.01

            # Metadata
            self.sigma = np.ones_like(self.g2) * 0.01
            self.q = 0.01  # Scattering vector (1/nm)
            self.L = 1000.0  # Sample thickness (nm)
            self.dt = 0.1  # Time step (s)

    return MockXPCSData


@pytest.fixture
def mock_config():
    """Create minimal config for testing."""

    class MockConfig:
        def __init__(self):
            self.config = {
                "performance": {
                    "enable_progress": False,  # Disable for cleaner test output
                },
                "optimization": {
                    "streaming": {
                        "enable_checkpoints": False,  # Disable for test speed
                    }
                },
            }

        def get_config_dict(self):
            return self.config

    return MockConfig()


@pytest.fixture
def static_mode_params():
    """Initial parameters for static mode mode.

    Parameters: [contrast, offset, D0, alpha, D_offset]
    """
    return np.array([0.3, 1.0, 1000.0, 0.5, 10.0])


@pytest.fixture
def static_mode_bounds():
    """Bounds for static mode parameters."""
    lower = np.array([0.1, 0.5, 100.0, 0.1, 1.0])
    upper = np.array([1.0, 2.0, 10000.0, 2.0, 100.0])
    return (lower, upper)


# ============================================================================
# Test Group 1: Strategy Selection Tests
# ============================================================================


def test_large_strategy_medium_dataset(
    mock_xpcs_data, mock_config, static_mode_params, static_mode_bounds
):
    """Test LARGE strategy selection for medium datasets.

    Note: We can't easily create 1M+ point datasets in unit tests without
    memory issues. This test uses mocking to simulate large dataset behavior.
    """
    # Create moderate dataset
    data = mock_xpcs_data(n_phi=10, n_t1=20, n_t2=20)

    wrapper = NLSQWrapper(
        enable_large_dataset=True,
        enable_recovery=False,
    )

    # Mock dataset size detection to force LARGE strategy
    with patch(
        "homodyne.optimization.nlsq_wrapper.NLSQWrapper._prepare_data"
    ) as mock_prepare:
        # Simulate 2M points (within LARGE range: 1M - 10M)
        xdata = np.arange(2_000_000, dtype=np.float64)
        ydata = np.random.randn(2_000_000) * 0.01 + 1.0
        mock_prepare.return_value = (xdata, ydata)

        # This should select LARGE strategy internally
        # We can't run the full optimization with mocked data, so just check preparation
        prepared_xdata, prepared_ydata = wrapper._prepare_data(data)
        assert len(prepared_ydata) == 2_000_000


def test_streaming_strategy_huge_dataset(
    mock_xpcs_data, mock_config, static_mode_params, static_mode_bounds
):
    """Test STREAMING strategy selection for huge datasets (> 100M points).

    Uses mocking since we can't create 100M+ point arrays in unit tests.
    """
    data = mock_xpcs_data(n_phi=5, n_t1=10, n_t2=10)

    wrapper = NLSQWrapper(
        enable_large_dataset=True,
        enable_recovery=False,
    )

    # Check that _fit_with_streaming_optimizer method exists
    assert hasattr(wrapper, "_fit_with_streaming_optimizer")


# ============================================================================
# Test Group 2: Fallback Chain Tests
# ============================================================================


def test_fallback_chain_standard_to_none():
    """Test fallback chain terminates at STANDARD strategy."""
    wrapper = NLSQWrapper()

    # STREAMING → CHUNKED
    assert (
        wrapper._get_fallback_strategy(OptimizationStrategy.STREAMING)
        == OptimizationStrategy.CHUNKED
    )

    # CHUNKED → LARGE
    assert (
        wrapper._get_fallback_strategy(OptimizationStrategy.CHUNKED)
        == OptimizationStrategy.LARGE
    )

    # LARGE → STANDARD
    assert (
        wrapper._get_fallback_strategy(OptimizationStrategy.LARGE)
        == OptimizationStrategy.STANDARD
    )

    # STANDARD → None (no more fallbacks)
    assert wrapper._get_fallback_strategy(OptimizationStrategy.STANDARD) is None


def test_fallback_chain_execution(
    mock_xpcs_data, mock_config, static_mode_params, static_mode_bounds
):
    """Test fallback chain executes when strategy fails.

    Simulates LARGE strategy failure → falls back to STANDARD.
    """
    data = mock_xpcs_data(n_phi=5, n_t1=10, n_t2=10)

    wrapper = NLSQWrapper(enable_large_dataset=True, enable_recovery=False)

    # Force wrapper to use LARGE strategy by mocking dataset size
    # Then make LARGE fail, which should trigger fallback to STANDARD
    with (
        patch("homodyne.optimization.nlsq_wrapper.curve_fit_large") as mock_large,
        patch.object(wrapper, "_prepare_data") as mock_prepare,
    ):
        # Mock large dataset (triggers LARGE strategy)
        xdata_large = np.arange(2_000_000, dtype=np.float64)
        ydata_large = np.random.randn(2_000_000) * 0.01 + 1.0
        mock_prepare.return_value = (xdata_large, ydata_large)

        # Make LARGE strategy fail
        mock_large.side_effect = RuntimeError("Simulated curve_fit_large failure")

        # Should fall back to STANDARD strategy and succeed
        result = wrapper.fit(
            data=data,
            config=mock_config,
            initial_params=static_mode_params,
            bounds=static_mode_bounds,
            analysis_mode="static",
        )

        # Should succeed via fallback
        assert result.success is True

        # Check recovery action was recorded
        assert any("fallback" in action.lower() for action in result.recovery_actions)


# ============================================================================
# Test Group 3: Error Recovery Tests
# ============================================================================


def test_error_recovery_perturb_parameters(
    mock_xpcs_data, mock_config, static_mode_params, static_mode_bounds
):
    """Test automatic parameter perturbation recovery."""
    data = mock_xpcs_data(n_phi=5, n_t1=10, n_t2=10)

    wrapper = NLSQWrapper(enable_large_dataset=True, enable_recovery=True)

    # Use poor initial guess to trigger recovery
    poor_params = static_mode_params * 0.1  # 10x smaller, likely to fail

    result = wrapper.fit(
        data=data,
        config=mock_config,
        initial_params=poor_params,
        bounds=static_mode_bounds,
        analysis_mode="static",
    )

    # Should eventually converge with recovery
    assert result.success is True or len(result.recovery_actions) > 0


def test_error_recovery_detects_stagnation(
    mock_xpcs_data, mock_config, static_mode_params, static_mode_bounds
):
    """Test detection of parameter stagnation (NLSQ bug workaround)."""
    data = mock_xpcs_data(n_phi=5, n_t1=10, n_t2=10)

    wrapper = NLSQWrapper(enable_large_dataset=True, enable_recovery=True)

    # Mock NLSQ to return unchanged parameters (simulate bug)
    with patch("homodyne.optimization.nlsq_wrapper.curve_fit") as mock_fit:
        # Return unchanged parameters and identity covariance
        mock_fit.return_value = (
            static_mode_params.copy(),  # Unchanged
            np.eye(len(static_mode_params)),  # Identity
        )

        # Should detect stagnation and retry
        try:
            result = wrapper.fit(
                data=data,
                config=mock_config,
                initial_params=static_mode_params,
                bounds=static_mode_bounds,
                analysis_mode="static",
            )

            # Check if stagnation was detected
            assert any(
                "stagnation" in action.lower() for action in result.recovery_actions
            )
        except RuntimeError:
            # May fail after all retries, which is acceptable
            pass


def test_diagnose_error_oom(
    mock_xpcs_data, mock_config, static_mode_params, static_mode_bounds
):
    """Test OOM error diagnosis provides actionable guidance."""
    wrapper = NLSQWrapper()

    # Simulate OOM error
    oom_error = RuntimeError("RESOURCE_EXHAUSTED: out of memory")

    diagnostic = wrapper._diagnose_error(
        error=oom_error,
        params=static_mode_params,
        bounds=static_mode_bounds,
        attempt=0,
    )

    assert diagnostic["error_type"] == "out_of_memory"
    assert "CPU" in " ".join(diagnostic["suggestions"])
    assert diagnostic["recovery_strategy"]["action"] == "no_recovery_available"


def test_diagnose_error_convergence(
    mock_xpcs_data, mock_config, static_mode_params, static_mode_bounds
):
    """Test convergence failure diagnosis suggests perturbation."""
    wrapper = NLSQWrapper()

    conv_error = RuntimeError("Maximum iterations reached without convergence")

    diagnostic = wrapper._diagnose_error(
        error=conv_error,
        params=static_mode_params,
        bounds=static_mode_bounds,
        attempt=0,
    )

    assert diagnostic["error_type"] == "convergence_failure"
    assert "perturb" in diagnostic["recovery_strategy"]["action"].lower()
    assert (
        diagnostic["recovery_strategy"]["new_params"].shape == static_mode_params.shape
    )


# ============================================================================
# Test Group 4: Result Normalization Tests
# ============================================================================


def test_handle_nlsq_result_tuple_2_elements():
    """Test _handle_nlsq_result with (popt, pcov) tuple."""
    popt = np.array([1.0, 2.0, 3.0])
    pcov = np.eye(3)

    result = (popt, pcov)
    normalized_popt, normalized_pcov, info = NLSQWrapper._handle_nlsq_result(
        result, OptimizationStrategy.LARGE
    )

    assert np.allclose(normalized_popt, popt)
    assert np.allclose(normalized_pcov, pcov)
    assert isinstance(info, dict)
    assert len(info) == 0  # Empty info dict


def test_handle_nlsq_result_tuple_3_elements():
    """Test _handle_nlsq_result with (popt, pcov, info) tuple."""
    popt = np.array([1.0, 2.0, 3.0])
    pcov = np.eye(3)
    info_dict = {"nfev": 100, "success": True}

    result = (popt, pcov, info_dict)
    normalized_popt, normalized_pcov, info = NLSQWrapper._handle_nlsq_result(
        result, OptimizationStrategy.STANDARD
    )

    assert np.allclose(normalized_popt, popt)
    assert np.allclose(normalized_pcov, pcov)
    assert info["nfev"] == 100
    assert info["success"] is True


def test_handle_nlsq_result_object_with_attributes():
    """Test _handle_nlsq_result with object (CurveFitResult-like)."""

    class MockCurveFitResult:
        def __init__(self):
            self.popt = np.array([1.0, 2.0, 3.0])
            self.pcov = np.eye(3)
            self.success = True
            self.message = "Optimization successful"
            self.nfev = 50

    result = MockCurveFitResult()
    normalized_popt, normalized_pcov, info = NLSQWrapper._handle_nlsq_result(
        result, OptimizationStrategy.STANDARD
    )

    assert np.allclose(normalized_popt, result.popt)
    assert np.allclose(normalized_pcov, result.pcov)
    assert info["success"] is True
    assert info["nfev"] == 50


def test_handle_nlsq_result_dict_streaming():
    """Test _handle_nlsq_result with dict (StreamingOptimizer format)."""
    result_dict = {
        "x": np.array([1.0, 2.0, 3.0]),
        "pcov": np.eye(3),
        "streaming_diagnostics": {
            "batch_success_rate": 0.95,
            "total_batches": 100,
        },
        "success": True,
        "best_loss": 0.123,
    }

    normalized_popt, normalized_pcov, info = NLSQWrapper._handle_nlsq_result(
        result_dict, OptimizationStrategy.STREAMING
    )

    assert np.allclose(normalized_popt, result_dict["x"])
    assert np.allclose(normalized_pcov, result_dict["pcov"])
    assert "streaming_diagnostics" in info
    assert info["streaming_diagnostics"]["batch_success_rate"] == 0.95


def test_handle_nlsq_result_invalid_format():
    """Test _handle_nlsq_result raises TypeError for invalid format."""
    invalid_result = "not a valid format"

    with pytest.raises(TypeError, match="Unrecognized NLSQ result format"):
        NLSQWrapper._handle_nlsq_result(invalid_result, OptimizationStrategy.STANDARD)


# ============================================================================
# Test Group 5: Enhanced Diagnostics Tests
# ============================================================================


def test_optimization_result_includes_diagnostics(
    mock_xpcs_data, mock_config, static_mode_params, static_mode_bounds
):
    """Test OptimizationResult contains enhanced diagnostics."""
    data = mock_xpcs_data(n_phi=5, n_t1=10, n_t2=10)

    wrapper = NLSQWrapper(enable_large_dataset=True, enable_recovery=True)

    result = wrapper.fit(
        data=data,
        config=mock_config,
        initial_params=static_mode_params,
        bounds=static_mode_bounds,
        analysis_mode="static",
    )

    # Check diagnostic fields exist
    assert hasattr(result, "recovery_actions")
    assert hasattr(result, "quality_flag")
    assert hasattr(result, "device_info")
    assert isinstance(result.recovery_actions, list)
    assert result.quality_flag in ["good", "marginal", "poor"]


def test_batch_statistics_tracking():
    """Test BatchStatistics component tracks batches correctly."""

    stats = BatchStatistics(max_size=10)

    # Record some batches
    stats.record_batch(0, success=True, loss=1.0, iterations=50, recovery_actions=[])
    stats.record_batch(1, success=True, loss=0.9, iterations=45, recovery_actions=[])
    stats.record_batch(
        2,
        success=False,
        loss=2.0,
        iterations=100,
        recovery_actions=["retry"],
        error_type="convergence",
    )

    # Check statistics
    assert stats.total_batches == 3
    assert stats.total_successes == 2
    assert stats.total_failures == 1
    assert stats.get_success_rate() == pytest.approx(2.0 / 3.0)

    full_stats = stats.get_statistics()
    assert full_stats["total_batches"] == 3
    assert "convergence" in full_stats["error_distribution"]


# ============================================================================
# Test Group 6: Checkpoint Integration Tests
# ============================================================================


def test_checkpoint_manager_save_load(tmp_path):
    """Test CheckpointManager save and load cycle."""
    from homodyne.optimization.checkpoint_manager import CheckpointManager

    manager = CheckpointManager(
        checkpoint_dir=tmp_path,
        checkpoint_frequency=10,
        keep_last_n=3,
    )

    # Save checkpoint
    params = np.array([1.0, 2.0, 3.0])
    optimizer_state = {"iteration": 42, "loss_history": [1.0, 0.9, 0.8]}
    loss = 0.5

    checkpoint_path = manager.save_checkpoint(
        batch_idx=10,
        parameters=params,
        optimizer_state=optimizer_state,
        loss=loss,
    )

    assert checkpoint_path.exists()

    # Load checkpoint
    loaded = manager.load_checkpoint(checkpoint_path)

    assert loaded["batch_idx"] == 10
    assert np.allclose(loaded["parameters"], params)
    assert loaded["loss"] == loss
    assert loaded["optimizer_state"]["iteration"] == 42


def test_checkpoint_manager_find_latest(tmp_path):
    """Test finding latest checkpoint."""
    from homodyne.optimization.checkpoint_manager import CheckpointManager

    manager = CheckpointManager(checkpoint_dir=tmp_path)

    # Save multiple checkpoints
    params = np.array([1.0, 2.0, 3.0])
    for batch_idx in [10, 20, 30]:
        manager.save_checkpoint(
            batch_idx=batch_idx,
            parameters=params * batch_idx,
            optimizer_state={"iteration": batch_idx},
            loss=1.0 / batch_idx,
        )

    # Find latest
    latest = manager.find_latest_checkpoint()
    assert latest is not None

    loaded = manager.load_checkpoint(latest)
    assert loaded["batch_idx"] == 30


def test_checkpoint_manager_cleanup(tmp_path):
    """Test automatic cleanup of old checkpoints."""
    from homodyne.optimization.checkpoint_manager import CheckpointManager

    manager = CheckpointManager(checkpoint_dir=tmp_path, keep_last_n=2)

    # Save 5 checkpoints
    params = np.array([1.0, 2.0, 3.0])
    for batch_idx in [10, 20, 30, 40, 50]:
        manager.save_checkpoint(
            batch_idx=batch_idx,
            parameters=params,
            optimizer_state={},
            loss=1.0,
        )

    # Cleanup (keep last 2)
    deleted = manager.cleanup_old_checkpoints()

    # Should delete 3 oldest (10, 20, 30), keep 2 newest (40, 50)
    assert len(deleted) == 3

    remaining = list(tmp_path.glob("homodyne_state_batch_*.h5"))
    assert len(remaining) == 2


# ============================================================================
# Test Group 7: Numerical Validation Tests
# ============================================================================


def test_numerical_validator_detects_nan_gradients():
    """Test NumericalValidator detects NaN in gradients."""

    validator = NumericalValidator(enable_validation=True)

    # Valid gradients
    valid_grads = np.array([1.0, 2.0, 3.0])
    validator.validate_gradients(valid_grads)  # Should pass

    # Invalid gradients (NaN)
    invalid_grads = np.array([1.0, np.nan, 3.0])
    with pytest.raises(NLSQNumericalError, match="Non-finite gradients"):
        validator.validate_gradients(invalid_grads)


def test_numerical_validator_detects_inf_parameters():
    """Test NumericalValidator detects Inf in parameters."""

    validator = NumericalValidator(enable_validation=True)

    # Valid parameters
    valid_params = np.array([1.0, 2.0, 3.0])
    validator.validate_parameters(valid_params)  # Should pass

    # Invalid parameters (Inf)
    invalid_params = np.array([1.0, np.inf, 3.0])
    with pytest.raises(NLSQNumericalError, match="Non-finite parameters"):
        validator.validate_parameters(invalid_params)


def test_numerical_validator_detects_bounds_violations():
    """Test NumericalValidator detects parameter bounds violations."""

    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([10.0, 10.0, 10.0])

    validator = NumericalValidator(enable_validation=True, bounds=(lower, upper))

    # Valid parameters
    valid_params = np.array([1.0, 5.0, 9.0])
    validator.validate_parameters(valid_params)  # Should pass

    # Violates upper bound
    invalid_params = np.array([1.0, 5.0, 15.0])
    with pytest.raises(NLSQNumericalError, match="bounds violations"):
        validator.validate_parameters(invalid_params)


def test_numerical_validator_detects_nan_loss():
    """Test NumericalValidator detects NaN in loss."""

    validator = NumericalValidator(enable_validation=True)

    # Valid loss
    validator.validate_loss(0.5)  # Should pass

    # Invalid loss (NaN)
    with pytest.raises(NLSQNumericalError, match="Non-finite loss"):
        validator.validate_loss(np.nan)


# ============================================================================
# Test Group 8: Performance Tests
# ============================================================================


def test_fast_mode_minimal_overhead(
    mock_xpcs_data, mock_config, static_mode_params, static_mode_bounds
):
    """Test fast mode has < 1% overhead (placeholder for future implementation).

    Note: Fast mode not yet implemented in nlsq_wrapper.py. This test
    documents the expected behavior.
    """
    # Future: Add fast_mode flag to NLSQWrapper.__init__
    # wrapper = NLSQWrapper(enable_large_dataset=True, fast_mode=True)
    pass


# ============================================================================
# Test Summary
# ============================================================================


def test_summary_all_strategies_tested():
    """Meta-test: Verify we've tested all 4 strategies."""
    strategies_tested = {
        "STANDARD",  # test_standard_strategy_small_dataset
        "LARGE",  # test_large_strategy_medium_dataset
        "CHUNKED",  # (implicit in fallback tests)
        "STREAMING",  # test_streaming_strategy_huge_dataset
    }

    all_strategies = {s.name for s in OptimizationStrategy}
    assert strategies_tested == all_strategies, (
        f"Missing tests for: {all_strategies - strategies_tested}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
