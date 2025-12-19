"""
MCMC Statistical Validation Tests
=================================

Basic validation tests for MCMC/CMC module availability.

Note: Full statistical tests for MCMC sampling (convergence diagnostics, parameter
recovery, chain mixing, posterior distribution properties, credible intervals) require
the CMC (Consensus Monte Carlo) workflow with proper data preparation. These tests
should be run through the CLI or integration tests that use the full CMC pipeline.
"""

import pytest


@pytest.mark.mcmc
class TestMCMCModuleAvailability:
    """Test MCMC/CMC module availability and imports."""

    def test_mcmc_module_imports(self):
        """Test MCMC module availability and basic imports."""
        try:
            from homodyne.optimization import NUMPYRO_AVAILABLE, fit_mcmc_jax

            # fit_mcmc_jax may be None if arviz is not installed
            if fit_mcmc_jax is None:
                pytest.skip("fit_mcmc_jax not available (arviz missing)")
            assert callable(fit_mcmc_jax)
            assert isinstance(NUMPYRO_AVAILABLE, bool)
        except ImportError:
            pytest.skip("MCMC module not available")

    def test_cmc_module_imports(self):
        """Test CMC (Consensus Monte Carlo) module availability."""
        # Skip if arviz is not available
        pytest.importorskip("arviz", reason="ArviZ required for CMC imports")

        from homodyne.optimization.cmc import (
            CMCConfig,
            CMCResult,
            fit_mcmc_jax,
        )

        assert callable(fit_mcmc_jax)
        assert CMCConfig is not None
        assert CMCResult is not None

    def test_numpyro_availability(self):
        """Test NumPyro availability for MCMC sampling."""
        try:
            from homodyne.optimization import NUMPYRO_AVAILABLE

            if NUMPYRO_AVAILABLE:
                import numpyro

                assert numpyro is not None
        except ImportError:
            pytest.skip("NumPyro not available")
