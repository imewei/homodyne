import logging

import pytest

# Require ArviZ for CMC imports; skip module if missing optional dependency
pytest.importorskip("arviz", reason="ArviZ required for CMC unit tests")

from homodyne.optimization.cmc import core  # noqa: E402


def test_cap_laminar_max_points_caps_high_values(caplog):
    caplog.set_level(logging.WARNING)

    # Implementation now caps at 3M, not 50K (runtime fix v2.4.1)
    # Values under the cap should pass through unchanged
    capped = core._cap_laminar_max_points(200_000, core.logger)
    assert capped == 200_000  # 200K is under the 3M cap
    assert not any("capping" in msg for msg in caplog.messages)

    # Test actual capping behavior at the 3M threshold
    capped_high = core._cap_laminar_max_points(5_000_000, core.logger)
    assert capped_high == 3_000_000
    assert any("capping to 3,000,000" in msg for msg in caplog.messages)


def test_cap_laminar_max_points_preserves_safe_values(caplog):
    caplog.set_level(logging.WARNING)

    capped = core._cap_laminar_max_points(40_000, core.logger)

    assert capped == 40_000
    assert not caplog.messages
