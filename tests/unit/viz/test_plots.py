"""Tests for visualization validation and basic plot functions."""

import numpy as np


class TestValidatePlotArrays:
    """Tests for validate_plot_arrays utility."""

    def test_clean_data_returns_true(self):
        """Arrays with no NaN/Inf should return True."""
        from homodyne.viz.validation import validate_plot_arrays

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert validate_plot_arrays(a, b) is True

    def test_nan_data_returns_false(self):
        """Arrays with NaN should return False and log warning."""
        from homodyne.viz.validation import validate_plot_arrays

        a = np.array([1.0, np.nan, 3.0])
        assert validate_plot_arrays(a, names=["test_arr"]) is False

    def test_inf_data_returns_false(self):
        """Arrays with Inf should return False and log warning."""
        from homodyne.viz.validation import validate_plot_arrays

        a = np.array([1.0, np.inf, -np.inf])
        assert validate_plot_arrays(a, names=["inf_arr"]) is False

    def test_mixed_clean_and_dirty(self):
        """Mix of clean and dirty arrays should return False."""
        from homodyne.viz.validation import validate_plot_arrays

        clean = np.array([1.0, 2.0])
        dirty = np.array([np.nan])
        assert validate_plot_arrays(clean, dirty, names=["clean", "dirty"]) is False

    def test_empty_array_is_clean(self):
        """Empty arrays should be considered clean."""
        from homodyne.viz.validation import validate_plot_arrays

        empty = np.array([])
        assert validate_plot_arrays(empty) is True

    def test_names_optional(self):
        """Should work without names parameter."""
        from homodyne.viz.validation import validate_plot_arrays

        a = np.array([np.nan])
        assert validate_plot_arrays(a) is False

    def test_no_arrays_returns_true(self):
        """No arrays should return True (vacuously)."""
        from homodyne.viz.validation import validate_plot_arrays

        assert validate_plot_arrays() is True


class TestNlsqPlotsContrastWarning:
    """Test that low contrast logs a warning instead of overriding."""

    def test_low_contrast_logs_warning(self, caplog, tmp_path):
        """Very low contrast should log a warning, not silently override."""
        import logging

        from homodyne.viz.nlsq_plots import plot_simulated_data

        # Use small end_frame to avoid creating huge meshgrid (default 8000x8000).
        config = {
            "analysis_mode": "static",
            "parameters": {},
            "analyzer_parameters": {"end_frame": 10, "start_frame": 1, "dt": 0.1},
        }
        with caplog.at_level(logging.WARNING, logger="homodyne.viz.nlsq_plots"):
            try:
                plot_simulated_data(
                    config=config,
                    contrast=0.05,
                    offset=0.0,
                    phi_angles_str=None,
                    plots_dir=tmp_path,
                )
            except Exception:
                pass  # May fail due to missing model deps; warning is emitted first

        assert any("contrast" in r.message.lower() for r in caplog.records), (
            "Expected a warning about low contrast"
        )
