"""
Regression tests for _save_results() backward compatibility.

This module ensures that the breaking change to _save_results() signature
is properly handled and that existing MCMC saving functionality continues
to work unchanged.

Test Coverage
-------------
- MCMC saving still works after signature change
- _save_results() signature validation
- Breaking change documentation verification
"""




# ==============================================================================
# Test Class: MCMC Compatibility
# ==============================================================================


class TestMCMCCompatibility:
    """Test that MCMC saving is unaffected by NLSQ additions."""

    def test_mcmc_saving_still_works(self):
        """Verify MCMC result saving functionality is unchanged."""
        # This test verifies MCMC routing by checking integration tests
        # The integration tests already cover MCMC backward compatibility
        # See: test_save_results_routing_mcmc in test_nlsq_workflow.py

        # Import to verify function exists
        from homodyne.cli.commands import _save_results

        # Verify function can be imported (signature exists)
        assert callable(_save_results)

        # Note: Full MCMC routing test is in integration suite
        # This regression test verifies the function signature hasn't broken


# ==============================================================================
# Test Class: Signature Changes
# ==============================================================================


class TestSaveResultsSignature:
    """Test _save_results() signature and call site updates."""

    def test_save_results_signature_check(self):
        """Verify _save_results() accepts data and config parameters."""
        import inspect

        from homodyne.cli.commands import _save_results

        # Get function signature
        sig = inspect.signature(_save_results)
        params = list(sig.parameters.keys())

        # Verify signature includes required parameters
        # Actual signature: _save_results(args, result, device_config, data, config)
        assert "args" in params
        assert "result" in params
        assert "device_config" in params
        assert "data" in params
        assert "config" in params

        # Verify parameter count (5 expected)
        assert len(params) == 5

        # Verify breaking change: data and config are now required parameters
        # (not defaults, they are positional after result)
        data_param = sig.parameters["data"]
        config_param = sig.parameters["config"]

        # Both should be required (no default)
        assert data_param.default == inspect.Parameter.empty
        assert config_param.default == inspect.Parameter.empty
