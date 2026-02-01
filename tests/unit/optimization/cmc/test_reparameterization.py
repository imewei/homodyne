# tests/unit/optimization/cmc/test_reparameterization.py
"""Tests for CMC reparameterization module."""

import numpy as np
import pytest


class TestReparamConfig:
    """Tests for ReparamConfig dataclass."""

    def test_default_config_enables_all_transforms(self):
        """Default config enables D_total and log_gamma transforms."""
        from homodyne.optimization.cmc.reparameterization import ReparamConfig

        config = ReparamConfig()
        assert config.enable_d_total is True
        assert config.enable_log_gamma is True
        assert config.t_ref == 1.0

    def test_config_can_disable_transforms(self):
        """Config can selectively disable transforms."""
        from homodyne.optimization.cmc.reparameterization import ReparamConfig

        config = ReparamConfig(enable_d_total=False, enable_log_gamma=False)
        assert config.enable_d_total is False
        assert config.enable_log_gamma is False
