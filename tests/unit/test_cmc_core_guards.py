import logging

from homodyne.optimization.cmc import core


def test_cap_laminar_max_points_caps_high_values(caplog):
    caplog.set_level(logging.WARNING)

    capped = core._cap_laminar_max_points(200_000, core.logger)

    assert capped == 50_000
    assert any("capping to 50,000" in msg for msg in caplog.messages)


def test_cap_laminar_max_points_preserves_safe_values(caplog):
    caplog.set_level(logging.WARNING)

    capped = core._cap_laminar_max_points(40_000, core.logger)

    assert capped == 40_000
    assert not caplog.messages

