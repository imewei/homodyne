
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from homodyne.optimization.cmc.data_prep import (
    PreparedData,
    shard_data_stratified,
    shard_data_random,
)
from homodyne.optimization.cmc.core import fit_mcmc_jax
from homodyne.optimization.cmc.config import CMCConfig

class TestShardingLogic:
    """Test CMC sharding strategies and their enforcement."""

    @pytest.fixture
    def multi_angle_data(self):
        """Create synthetic multi-angle data."""
        n_points = 1000
        # 500 points for angle 0, 500 for angle 1
        phi_indices = np.concatenate([np.zeros(500), np.ones(500)]).astype(int)
        phi_values = np.concatenate([np.full(500, 0.1), np.full(500, 0.2)])
        
        t1 = np.linspace(0, 1, n_points)
        t2 = t1 + 0.1 # Ensure off-diagonal
        
        return PreparedData(
            data=np.abs(np.random.randn(n_points)), # positive data to avoid warnings
            t1=t1,
            t2=t2,
            phi=phi_values,
            phi_unique=np.array([0.1, 0.2]),
            phi_indices=phi_indices,
            n_total=n_points,
            n_phi=2,
            noise_scale=0.1
        )

    def test_stratified_sharding_purity(self, multi_angle_data):
        """Verify stratified sharding creates angle-pure shards."""
        # 2 shards (one per angle)
        shards = shard_data_stratified(multi_angle_data, num_shards=2)
        
        assert len(shards) == 2
        # Shard 0 should be all angle 0
        assert np.all(shards[0].phi_indices == 0)
        # Shard 1 should be all angle 1 (which becomes index 0 in the shard's local scope? 
        # distinct phi values. Let's check phi values directly)
        assert np.all(np.isclose(shards[0].phi, 0.1))
        assert np.all(np.isclose(shards[1].phi, 0.2))

    def test_random_sharding_mixture(self, multi_angle_data):
        """Verify random sharding creates mixed-angle shards."""
        # 2 shards
        shards = shard_data_random(multi_angle_data, num_shards=2, seed=42)
        
        assert len(shards) == 2
        # Both shards should contain BOTH angles (statistically highly probable)
        for s in shards:
            unique_sub = np.unique(s.phi)
            assert len(unique_sub) == 2, "Shard should contain mixed angles"
            assert np.isclose(unique_sub[0], 0.1)
            assert np.isclose(unique_sub[1], 0.2)

    @patch("homodyne.optimization.cmc.core.select_backend")
    @patch("homodyne.optimization.cmc.core.CMCResult") # Mock result creation
    @patch("homodyne.optimization.cmc.core.shard_data_random")
    @patch("homodyne.optimization.cmc.core.shard_data_stratified")
    def test_force_random_sharding_override(
        self, mock_stratified, mock_random, mock_result_cls, mock_select_backend, multi_angle_data
    ):
        """Verify core.py forces random sharding for multi-angle data even if stratified requested."""
        
        # Setup mocks
        mock_random.return_value = [multi_angle_data, multi_angle_data] # Return 2 shards
        mock_stratified.return_value = [multi_angle_data, multi_angle_data]
        
        mock_backend = MagicMock()
        mock_backend.get_name.return_value = "mock_backend"
        mock_samples = MagicMock()
        mock_samples.extra_fields = {"diverging": np.array([0])}
        mock_backend.run.return_value = mock_samples
        mock_select_backend.return_value = mock_backend
        
        mock_result = MagicMock()
        mock_result.convergence_status = "converged"
        mock_result.r_hat = {"param1": 1.01}
        mock_result.ess_bulk = {"param1": 1000}
        mock_result.divergences = 0
        mock_result.n_samples = 1500
        mock_result.n_chains = 4
        mock_result.num_shards = 2
        mock_result.get_posterior_stats.return_value = {}
        mock_result_cls.from_mcmc_samples.return_value = mock_result
        
        # Manually create config demanding stratified
        config_dict = {
            "sharding": {"strategy": "stratified", "num_shards": 2},
            "enable": True,
            "backend": {"name": "auto"} # Ensure backend selection logic works
        }
        
        # Call fit with multi-angle data (n_phi=2)
        fit_mcmc_jax(
            data=multi_angle_data.data,
            t1=multi_angle_data.t1,
            t2=multi_angle_data.t2,
            phi=multi_angle_data.phi,
            q=0.01,
            L=1.0,
            analysis_mode="laminar_flow",
            cmc_config=config_dict
        )
        
        # Verify shard_data_random WAS called
        assert mock_random.called, "Should have switched to random sharding"
        # Verify shard_data_stratified WAS NOT called
        assert not mock_stratified.called, "Should not have used stratified sharding"

    @patch("homodyne.optimization.cmc.core.select_backend")
    @patch("homodyne.optimization.cmc.core.CMCResult")
    @patch("homodyne.optimization.cmc.core.shard_data_random")
    @patch("homodyne.optimization.cmc.core.shard_data_stratified")
    def test_stratified_allowed_for_single_angle(
        self, mock_stratified, mock_random, mock_result_cls, mock_select_backend, multi_angle_data
    ):
        """Verify stratified is STILL allowed for single-angle data (no override)."""
        
        # Single angle data
        n = 100
        data = np.abs(np.random.randn(n))
        phi = np.full(n, 0.1) # single angle
        t1 = np.linspace(0, 1, n)
        t2 = t1 + 0.1
        
        # Setup mocks
        mock_stratified.return_value = [MagicMock(n_total=50), MagicMock(n_total=50)]
        
        mock_backend = MagicMock()
        mock_backend.get_name.return_value = "mock_backend"
        mock_samples = MagicMock()
        mock_samples.extra_fields = {"diverging": np.array([0])}
        mock_backend.run.return_value = mock_samples
        mock_select_backend.return_value = mock_backend
        
        mock_result = MagicMock()
        mock_result.convergence_status = "converged"
        mock_result.r_hat = {"param1": 1.01}
        mock_result.ess_bulk = {"param1": 1000}
        mock_result.divergences = 0
        mock_result.n_samples = 1500
        mock_result.n_chains = 4
        mock_result.num_shards = 2
        mock_result.get_posterior_stats.return_value = {}
        mock_result_cls.from_mcmc_samples.return_value = mock_result
        
        config_dict = {
            "sharding": {"strategy": "stratified", "num_shards": 2},
            "enable": True
        }
        
        fit_mcmc_jax(
            data=data,
            t1=t1,
            t2=t2,
            phi=phi,
            q=0.01,
            L=1.0,
            analysis_mode="laminar_flow",
            cmc_config=config_dict
        )
        
        # Verify stratified WAS called (since n_phi=1)
        assert mock_stratified.called, "Should use stratified when only 1 angle"
        assert not mock_random.called
