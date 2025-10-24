"""Tests for MCMC integration with Consensus Monte Carlo (CMC).

This test suite validates Task Group 9: MCMC Integration (replace fit_mcmc_jax).

Test Coverage
-------------
1. Automatic method selection with small dataset (<500k) → NUTS
2. Automatic method selection with large dataset (>1M) → CMC
3. Forced NUTS method
4. Forced CMC method
5. Invalid method raises error
6. Backward compatibility (no method parameter)
7. Parameter passing to coordinator
8. MCMCResult format consistency
9. Hardware detection fallback
10. Warning logging for suboptimal choices

Integration Points
------------------
- fit_mcmc_jax() with method parameter
- Hardware detection (detect_hardware, should_use_cmc)
- CMC coordinator execution
- Extended MCMCResult compatibility

Author: Claude Code (Task Group 9)
Date: 2025-10-24
"""

import numpy as np
import pytest

from homodyne.optimization.mcmc import fit_mcmc_jax, MCMCResult


class TestMethodSelection:
    """Test automatic method selection logic."""

    def test_auto_selection_small_dataset_uses_nuts(self, mocker):
        """Test automatic selection uses NUTS for small datasets (<500k)."""
        # Mock hardware detection at the point of import
        mock_hardware = mocker.MagicMock()
        mock_hardware.platform = "cpu"
        mock_hardware.num_devices = 1

        mocker.patch(
            'homodyne.device.config.detect_hardware',
            return_value=mock_hardware
        )
        mocker.patch(
            'homodyne.device.config.should_use_cmc',
            return_value=False  # Small dataset → NUTS
        )

        # Mock _run_standard_nuts to avoid actual execution
        mock_nuts_result = MCMCResult(
            mean_params=np.array([100.0, 1.5, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            converged=True,
        )
        mock_nuts = mocker.patch(
            'homodyne.optimization.mcmc._run_standard_nuts',
            return_value=mock_nuts_result
        )

        # Create small dataset (100k points)
        data = np.random.randn(100_000)
        t1 = np.random.rand(100_000)
        t2 = np.random.rand(100_000)
        phi = np.random.rand(100_000)

        # Call with method='auto' (default)
        result = fit_mcmc_jax(
            data=data,
            t1=t1,
            t2=t2,
            phi=phi,
            q=0.01,
            L=3.5,
            analysis_mode='static_isotropic',
        )

        # Verify NUTS was called
        mock_nuts.assert_called_once()

        # Verify result is not CMC
        assert not result.is_cmc_result()
        assert result.num_shards is None

    def test_auto_selection_large_dataset_uses_cmc(self, mocker):
        """Test automatic selection uses CMC for large datasets (>1M)."""
        # Mock hardware detection
        mock_hardware = mocker.MagicMock()
        mock_hardware.platform = "gpu"
        mock_hardware.num_devices = 4

        mocker.patch(
            'homodyne.device.config.detect_hardware',
            return_value=mock_hardware
        )
        mocker.patch(
            'homodyne.device.config.should_use_cmc',
            return_value=True  # Large dataset → CMC
        )

        # Mock CMC coordinator
        mock_cmc_result = MCMCResult(
            mean_params=np.array([100.0, 1.5, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            converged=True,
            num_shards=10,
            combination_method='weighted',
        )

        mock_coordinator = mocker.MagicMock()
        mock_coordinator.run_cmc.return_value = mock_cmc_result

        mock_coordinator_class = mocker.patch(
            'homodyne.optimization.cmc.coordinator.CMCCoordinator',
            return_value=mock_coordinator
        )

        # Create large dataset (2M points)
        data = np.random.randn(2_000_000)
        t1 = np.random.rand(2_000_000)
        t2 = np.random.rand(2_000_000)
        phi = np.random.rand(2_000_000)

        # Call with method='auto' (default)
        result = fit_mcmc_jax(
            data=data,
            t1=t1,
            t2=t2,
            phi=phi,
            q=0.01,
            L=3.5,
            analysis_mode='static_isotropic',
        )

        # Verify CMC coordinator was created
        mock_coordinator_class.assert_called_once()

        # Verify CMC was executed
        mock_coordinator.run_cmc.assert_called_once()

        # Verify result is CMC
        assert result.is_cmc_result()
        assert result.num_shards == 10
        assert result.combination_method == 'weighted'

    def test_auto_selection_fallback_without_hardware_detection(self, mocker):
        """Test automatic selection works even if hardware detection fails."""
        # Mock hardware detection to raise ImportError
        mocker.patch(
            'homodyne.device.config.detect_hardware',
            side_effect=ImportError("Hardware detection unavailable")
        )

        # Mock _run_standard_nuts for small dataset
        mock_nuts_result = MCMCResult(
            mean_params=np.array([100.0, 1.5, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            converged=True,
        )
        mock_nuts = mocker.patch(
            'homodyne.optimization.mcmc._run_standard_nuts',
            return_value=mock_nuts_result
        )

        # Create small dataset (500k points, below 1M fallback threshold)
        data = np.random.randn(500_000)
        t1 = np.random.rand(500_000)
        t2 = np.random.rand(500_000)
        phi = np.random.rand(500_000)

        # Call with method='auto'
        result = fit_mcmc_jax(
            data=data,
            t1=t1,
            t2=t2,
            phi=phi,
            q=0.01,
            L=3.5,
            analysis_mode='static_isotropic',
        )

        # Verify NUTS was used (fallback threshold)
        mock_nuts.assert_called_once()
        assert not result.is_cmc_result()


class TestForcedMethodSelection:
    """Test forced method selection (method='nuts' or method='cmc')."""

    def test_forced_nuts_method(self, mocker):
        """Test forcing NUTS method bypasses automatic selection."""
        # Mock _run_standard_nuts
        mock_nuts_result = MCMCResult(
            mean_params=np.array([100.0, 1.5, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            converged=True,
        )
        mock_nuts = mocker.patch(
            'homodyne.optimization.mcmc._run_standard_nuts',
            return_value=mock_nuts_result
        )

        # Create dataset
        data = np.random.randn(100_000)
        t1 = np.random.rand(100_000)
        t2 = np.random.rand(100_000)
        phi = np.random.rand(100_000)

        # Force NUTS method
        result = fit_mcmc_jax(
            data=data,
            t1=t1,
            t2=t2,
            phi=phi,
            q=0.01,
            L=3.5,
            method='nuts',
        )

        # Verify NUTS was called
        mock_nuts.assert_called_once()
        assert not result.is_cmc_result()

    def test_forced_cmc_method(self, mocker):
        """Test forcing CMC method bypasses automatic selection."""
        # Mock CMC coordinator
        mock_cmc_result = MCMCResult(
            mean_params=np.array([100.0, 1.5, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            converged=True,
            num_shards=5,
            combination_method='weighted',
        )

        mock_coordinator = mocker.MagicMock()
        mock_coordinator.run_cmc.return_value = mock_cmc_result

        mock_coordinator_class = mocker.patch(
            'homodyne.optimization.cmc.coordinator.CMCCoordinator',
            return_value=mock_coordinator
        )

        # Create dataset
        data = np.random.randn(100_000)
        t1 = np.random.rand(100_000)
        t2 = np.random.rand(100_000)
        phi = np.random.rand(100_000)

        # Force CMC method
        result = fit_mcmc_jax(
            data=data,
            t1=t1,
            t2=t2,
            phi=phi,
            q=0.01,
            L=3.5,
            method='cmc',
        )

        # Verify CMC was used
        mock_coordinator_class.assert_called_once()
        mock_coordinator.run_cmc.assert_called_once()
        assert result.is_cmc_result()
        assert result.num_shards == 5

    def test_forced_nuts_on_large_dataset_logs_warning(self, mocker, caplog):
        """Test forcing NUTS on large dataset logs warning."""
        # Mock _run_standard_nuts
        mock_nuts_result = MCMCResult(
            mean_params=np.array([100.0, 1.5, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            converged=True,
        )
        mocker.patch(
            'homodyne.optimization.mcmc._run_standard_nuts',
            return_value=mock_nuts_result
        )

        # Create large dataset (6M points)
        data = np.random.randn(6_000_000)
        t1 = np.random.rand(6_000_000)
        t2 = np.random.rand(6_000_000)
        phi = np.random.rand(6_000_000)

        # Force NUTS on large dataset
        result = fit_mcmc_jax(
            data=data,
            t1=t1,
            t2=t2,
            phi=phi,
            q=0.01,
            L=3.5,
            method='nuts',
        )

        # Verify warning was logged
        assert "Risk of OOM errors" in caplog.text
        assert ">5M points" in caplog.text

    def test_forced_cmc_on_small_dataset_logs_warning(self, mocker, caplog):
        """Test forcing CMC on small dataset logs warning."""
        # Mock CMC coordinator
        mock_cmc_result = MCMCResult(
            mean_params=np.array([100.0, 1.5, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            converged=True,
            num_shards=2,
        )

        mock_coordinator = mocker.MagicMock()
        mock_coordinator.run_cmc.return_value = mock_cmc_result

        mocker.patch(
            'homodyne.optimization.cmc.coordinator.CMCCoordinator',
            return_value=mock_coordinator
        )

        # Create small dataset (100k points)
        data = np.random.randn(100_000)
        t1 = np.random.rand(100_000)
        t2 = np.random.rand(100_000)
        phi = np.random.rand(100_000)

        # Force CMC on small dataset
        result = fit_mcmc_jax(
            data=data,
            t1=t1,
            t2=t2,
            phi=phi,
            q=0.01,
            L=3.5,
            method='cmc',
        )

        # Verify warning was logged
        assert "CMC adds overhead" in caplog.text
        assert "<500k points" in caplog.text


class TestMethodValidation:
    """Test method parameter validation."""

    def test_invalid_method_raises_error(self):
        """Test invalid method parameter raises ValueError."""
        data = np.random.randn(1000)
        t1 = np.random.rand(1000)
        t2 = np.random.rand(1000)
        phi = np.random.rand(1000)

        with pytest.raises(ValueError, match="Invalid method"):
            fit_mcmc_jax(
                data=data,
                t1=t1,
                t2=t2,
                phi=phi,
                q=0.01,
                L=3.5,
                method='invalid_method',
            )

    def test_method_must_be_string(self):
        """Test method parameter must be string."""
        data = np.random.randn(1000)
        t1 = np.random.rand(1000)
        t2 = np.random.rand(1000)
        phi = np.random.rand(1000)

        # This should raise ValueError (not in allowed list)
        with pytest.raises(ValueError):
            fit_mcmc_jax(
                data=data,
                t1=t1,
                t2=t2,
                phi=phi,
                q=0.01,
                L=3.5,
                method=123,  # Invalid type
            )


class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_no_method_parameter_uses_auto(self, mocker):
        """Test omitting method parameter defaults to 'auto'."""
        # Mock hardware detection
        mock_hardware = mocker.MagicMock()
        mock_hardware.platform = "cpu"

        mocker.patch(
            'homodyne.device.config.detect_hardware',
            return_value=mock_hardware
        )
        mocker.patch(
            'homodyne.device.config.should_use_cmc',
            return_value=False
        )

        # Mock NUTS execution
        mock_nuts_result = MCMCResult(
            mean_params=np.array([100.0, 1.5, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            converged=True,
        )
        mock_nuts = mocker.patch(
            'homodyne.optimization.mcmc._run_standard_nuts',
            return_value=mock_nuts_result
        )

        # Create dataset
        data = np.random.randn(10_000)
        t1 = np.random.rand(10_000)
        t2 = np.random.rand(10_000)
        phi = np.random.rand(10_000)

        # Call without method parameter (backward compatibility)
        result = fit_mcmc_jax(
            data=data,
            t1=t1,
            t2=t2,
            phi=phi,
            q=0.01,
            L=3.5,
        )

        # Verify automatic selection worked
        mock_nuts.assert_called_once()
        assert not result.is_cmc_result()

    def test_existing_kwargs_still_work(self, mocker):
        """Test existing kwargs (n_samples, n_warmup, etc.) still work."""
        # Mock NUTS execution
        mock_nuts_result = MCMCResult(
            mean_params=np.array([100.0, 1.5, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            converged=True,
        )
        mock_nuts = mocker.patch(
            'homodyne.optimization.mcmc._run_standard_nuts',
            return_value=mock_nuts_result
        )

        # Create dataset
        data = np.random.randn(10_000)
        t1 = np.random.rand(10_000)
        t2 = np.random.rand(10_000)
        phi = np.random.rand(10_000)

        # Call with existing kwargs
        result = fit_mcmc_jax(
            data=data,
            t1=t1,
            t2=t2,
            phi=phi,
            q=0.01,
            L=3.5,
            method='nuts',
            n_samples=2000,
            n_warmup=1000,
            n_chains=4,
        )

        # Verify kwargs were passed through
        call_kwargs = mock_nuts.call_args[1]
        assert 'n_samples' in call_kwargs or 'kwargs' in call_kwargs


class TestParameterPassing:
    """Test parameter passing to CMC coordinator."""

    def test_cmc_config_passed_to_coordinator(self, mocker):
        """Test cmc_config kwarg is passed to CMCCoordinator."""
        # Mock CMC coordinator
        mock_cmc_result = MCMCResult(
            mean_params=np.array([100.0, 1.5, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            converged=True,
            num_shards=10,
        )

        mock_coordinator = mocker.MagicMock()
        mock_coordinator.run_cmc.return_value = mock_cmc_result

        mock_coordinator_class = mocker.patch(
            'homodyne.optimization.cmc.coordinator.CMCCoordinator',
            return_value=mock_coordinator
        )

        # Create dataset
        data = np.random.randn(100_000)
        t1 = np.random.rand(100_000)
        t2 = np.random.rand(100_000)
        phi = np.random.rand(100_000)

        # Custom CMC config
        cmc_config = {
            'sharding': {'num_shards': 15, 'strategy': 'stratified'},
            'initialization': {'use_svi': True, 'svi_steps': 10000},
        }

        # Call with CMC method and config
        result = fit_mcmc_jax(
            data=data,
            t1=t1,
            t2=t2,
            phi=phi,
            q=0.01,
            L=3.5,
            method='cmc',
            cmc_config=cmc_config,
        )

        # Verify coordinator was created with config
        created_config = mock_coordinator_class.call_args[0][0]
        assert 'sharding' in created_config
        assert created_config['sharding']['num_shards'] == 15

    def test_initial_params_passed_to_cmc(self, mocker):
        """Test initial_params are passed to CMC as nlsq_params."""
        # Mock CMC coordinator
        mock_cmc_result = MCMCResult(
            mean_params=np.array([100.0, 1.5, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            converged=True,
            num_shards=5,
        )

        mock_coordinator = mocker.MagicMock()
        mock_coordinator.run_cmc.return_value = mock_cmc_result

        mocker.patch(
            'homodyne.optimization.cmc.coordinator.CMCCoordinator',
            return_value=mock_coordinator
        )

        # Create dataset
        data = np.random.randn(100_000)
        t1 = np.random.rand(100_000)
        t2 = np.random.rand(100_000)
        phi = np.random.rand(100_000)

        # Initial params (from NLSQ)
        initial_params = {
            'D0': 1000.0,
            'alpha': 1.5,
            'D_offset': 10.0,
        }

        # Call with CMC method
        result = fit_mcmc_jax(
            data=data,
            t1=t1,
            t2=t2,
            phi=phi,
            q=0.01,
            L=3.5,
            method='cmc',
            initial_params=initial_params,
        )

        # Verify initial_params were passed as nlsq_params
        call_kwargs = mock_coordinator.run_cmc.call_args[1]
        assert 'nlsq_params' in call_kwargs
        assert call_kwargs['nlsq_params'] == initial_params


class TestMCMCResultFormat:
    """Test MCMCResult format consistency between NUTS and CMC."""

    def test_nuts_result_has_standard_fields(self, mocker):
        """Test NUTS result has all standard MCMCResult fields."""
        # Mock NUTS execution
        mock_nuts_result = MCMCResult(
            mean_params=np.array([100.0, 1.5, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            std_params=np.array([5.0, 0.1, 0.5]),
            std_contrast=0.05,
            std_offset=0.02,
            converged=True,
            n_iterations=3000,
            computation_time=45.2,
        )
        mocker.patch(
            'homodyne.optimization.mcmc._run_standard_nuts',
            return_value=mock_nuts_result
        )

        # Create dataset
        data = np.random.randn(10_000)
        t1 = np.random.rand(10_000)
        t2 = np.random.rand(10_000)
        phi = np.random.rand(10_000)

        # Run NUTS
        result = fit_mcmc_jax(
            data=data,
            t1=t1,
            t2=t2,
            phi=phi,
            q=0.01,
            L=3.5,
            method='nuts',
        )

        # Verify standard fields
        assert hasattr(result, 'mean_params')
        assert hasattr(result, 'mean_contrast')
        assert hasattr(result, 'mean_offset')
        assert hasattr(result, 'std_params')
        assert hasattr(result, 'converged')
        assert not result.is_cmc_result()

    def test_cmc_result_has_extended_fields(self, mocker):
        """Test CMC result has extended CMC-specific fields."""
        # Mock CMC coordinator
        mock_cmc_result = MCMCResult(
            mean_params=np.array([100.0, 1.5, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            converged=True,
            num_shards=10,
            combination_method='weighted',
            per_shard_diagnostics=[
                {'shard_id': 0, 'converged': True, 'acceptance_rate': 0.85},
                {'shard_id': 1, 'converged': True, 'acceptance_rate': 0.82},
            ],
            cmc_diagnostics={
                'combination_success': True,
                'n_shards_converged': 10,
                'n_shards_total': 10,
            },
        )

        mock_coordinator = mocker.MagicMock()
        mock_coordinator.run_cmc.return_value = mock_cmc_result

        mocker.patch(
            'homodyne.optimization.cmc.coordinator.CMCCoordinator',
            return_value=mock_coordinator
        )

        # Create dataset
        data = np.random.randn(100_000)
        t1 = np.random.rand(100_000)
        t2 = np.random.rand(100_000)
        phi = np.random.rand(100_000)

        # Run CMC
        result = fit_mcmc_jax(
            data=data,
            t1=t1,
            t2=t2,
            phi=phi,
            q=0.01,
            L=3.5,
            method='cmc',
        )

        # Verify CMC-specific fields
        assert result.is_cmc_result()
        assert result.num_shards == 10
        assert result.combination_method == 'weighted'
        assert hasattr(result, 'per_shard_diagnostics')
        assert hasattr(result, 'cmc_diagnostics')


# Run all tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
