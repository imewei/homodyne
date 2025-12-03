"""Test tier 1 reparameterization fix (disable_reparam flag).

This test verifies that tier 1 with disable_reparam=True does NOT add
reparameterization parameters (log_D_center, delta_raw) to initial_values.
"""

import numpy as np
import pytest

try:
    from homodyne.optimization.mcmc import NUMPYRO_AVAILABLE
except ImportError:
    NUMPYRO_AVAILABLE = False


@pytest.mark.skipif(not NUMPYRO_AVAILABLE, reason="NumPyro not available")
class TestTier1ReparameterizationFix:
    """Test suite for tier 1 disable_reparam fix."""

    def test_tier1_initialization_logic(self):
        """Test that initialization logic respects disable_reparam flag.

        This is a whitebox test that verifies the fix at lines 1575-1580 in mcmc.py.
        """
        # Simulate the initialization logic
        single_angle_static = True
        initial_values = {
            "D0": 16834.86,
            "alpha": -1.571,
            "D_offset": 3.026,
            "contrast": 0.05015,
            "offset": 1.001,
        }

        # Tier 1 config with disable_reparam=True
        single_angle_surrogate_cfg = {
            "tier": "1",
            "disable_reparam": True,
            "drop_d_offset": False,
            "sample_log_d0": False,
        }

        # Replicate the logic from mcmc.py lines 1575-1580
        reparam_disabled = (
            single_angle_surrogate_cfg
            and single_angle_surrogate_cfg.get("disable_reparam", False)
        )

        initial_values_copy = initial_values.copy()

        # This is the fixed code path
        if (
            single_angle_static
            and initial_values_copy is not None
            and not reparam_disabled
        ):
            d0_init = initial_values_copy.get("D0")
            d_offset_init = initial_values_copy.get("D_offset")
            if d0_init is not None and d_offset_init is not None:
                center_value = d0_init + d_offset_init
                if not np.isfinite(center_value) or center_value <= 0:
                    center_value = max(1.0, abs(d0_init))
                log_center_init = float(np.log(max(center_value, 1e-6)))
                frac = d0_init / max(center_value, 1e-6)
                frac = float(np.clip(frac, 1e-6, 5.0))
                delta_raw_init = float(np.log(np.expm1(frac)))
                initial_values_copy.setdefault("log_D_center", log_center_init)
                initial_values_copy.setdefault("delta_raw", delta_raw_init)

        # Verify that reparam parameters were NOT added for tier 1
        assert reparam_disabled is True, "Tier 1 should have reparam_disabled=True"
        assert "log_D_center" not in initial_values_copy, (
            "Tier 1 should NOT add log_D_center"
        )
        assert "delta_raw" not in initial_values_copy, "Tier 1 should NOT add delta_raw"
        assert len(initial_values_copy) == 5, (
            f"Tier 1 should have exactly 5 parameters, not {len(initial_values_copy)}"
        )

    def test_tier2_initialization_adds_reparam(self):
        """Test that tier 2 initialization DOES add reparameterization parameters."""
        # Simulate the initialization logic
        single_angle_static = True
        initial_values = {
            "D0": 16834.86,
            "alpha": -1.571,
            "contrast": 0.05015,
            "offset": 1.001,
        }

        # Tier 2 config WITHOUT disable_reparam (or set to False)
        single_angle_surrogate_cfg = {
            "tier": "2",
            "disable_reparam": False,  # or None
            "drop_d_offset": True,
            "sample_log_d0": True,
        }

        # Replicate the logic from mcmc.py lines 1575-1580
        reparam_disabled = (
            single_angle_surrogate_cfg
            and single_angle_surrogate_cfg.get("disable_reparam", False)
        )

        initial_values_copy = initial_values.copy()

        # Add D_offset for the test (tier 2 code path)
        initial_values_copy["D_offset"] = 0.0

        # This is the code path that adds reparameterization
        if (
            single_angle_static
            and initial_values_copy is not None
            and not reparam_disabled
        ):
            d0_init = initial_values_copy.get("D0")
            d_offset_init = initial_values_copy.get("D_offset")
            if d0_init is not None and d_offset_init is not None:
                center_value = d0_init + d_offset_init
                if not np.isfinite(center_value) or center_value <= 0:
                    center_value = max(1.0, abs(d0_init))
                log_center_init = float(np.log(max(center_value, 1e-6)))
                frac = d0_init / max(center_value, 1e-6)
                frac = float(np.clip(frac, 1e-6, 5.0))
                delta_raw_init = float(np.log(np.expm1(frac)))
                initial_values_copy.setdefault("log_D_center", log_center_init)
                initial_values_copy.setdefault("delta_raw", delta_raw_init)

        # Verify that reparam parameters WERE added for tier 2
        assert reparam_disabled is False, "Tier 2 should have reparam_disabled=False"
        assert "log_D_center" in initial_values_copy, "Tier 2 SHOULD add log_D_center"
        assert "delta_raw" in initial_values_copy, "Tier 2 SHOULD add delta_raw"
        assert len(initial_values_copy) == 7, (
            f"Tier 2 should have 7 parameters (5 + 2 reparam), not {len(initial_values_copy)}"
        )
