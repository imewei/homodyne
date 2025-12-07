"""Tests for CMC diagnostics module."""

import pytest

pytest.importorskip("arviz", reason="ArviZ required for CMC unit tests")

from homodyne.optimization.cmc.diagnostics import (
    check_convergence,
    create_diagnostics_dict,
    summarize_diagnostics,
)


def compute_divergence_rate(divergences: int, n_samples: int, n_chains: int) -> float:
    """Compute divergence rate (helper for tests)."""
    total_samples = n_samples * n_chains
    if total_samples == 0:
        return 0.0
    return divergences / total_samples


class TestCheckConvergence:
    """Tests for check_convergence function."""

    def test_converged_good_diagnostics(self):
        """Test convergence check passes with good diagnostics."""
        r_hat = {"D0": 1.001, "alpha": 1.002}
        ess_bulk = {"D0": 1000.0, "alpha": 900.0}
        divergences = 0

        status, warnings = check_convergence(
            r_hat=r_hat,
            ess_bulk=ess_bulk,
            divergences=divergences,
            n_samples=2000,
            n_chains=4,
            max_rhat=1.1,
            min_ess=100.0,
        )

        assert status == "converged"
        assert len(warnings) == 0

    def test_not_converged_high_rhat(self):
        """Test convergence fails with high R-hat."""
        r_hat = {"D0": 1.2, "alpha": 1.001}  # D0 R-hat too high
        ess_bulk = {"D0": 1000.0, "alpha": 900.0}
        divergences = 0

        status, warnings = check_convergence(
            r_hat=r_hat,
            ess_bulk=ess_bulk,
            divergences=divergences,
            n_samples=2000,
            n_chains=4,
            max_rhat=1.1,
            min_ess=100.0,
        )

        assert status == "not_converged"
        assert any(
            "R-hat" in w or "r_hat" in w.lower() or "rhat" in w.lower()
            for w in warnings
        )

    def test_not_converged_low_ess(self):
        """Test convergence fails with low ESS."""
        r_hat = {"D0": 1.001, "alpha": 1.002}
        ess_bulk = {"D0": 50.0, "alpha": 900.0}  # D0 ESS too low
        divergences = 0

        status, warnings = check_convergence(
            r_hat=r_hat,
            ess_bulk=ess_bulk,
            divergences=divergences,
            n_samples=2000,
            n_chains=4,
            max_rhat=1.1,
            min_ess=100.0,
        )

        assert status == "not_converged"
        assert any("ESS" in w or "ess" in w.lower() for w in warnings)

    def test_divergences_status(self):
        """Test convergence reports divergences."""
        r_hat = {"D0": 1.001, "alpha": 1.002}
        ess_bulk = {"D0": 1000.0, "alpha": 900.0}
        divergences = 1000  # Many divergences (12.5% rate)

        status, warnings = check_convergence(
            r_hat=r_hat,
            ess_bulk=ess_bulk,
            divergences=divergences,
            n_samples=2000,
            n_chains=4,
            max_rhat=1.1,
            min_ess=100.0,
        )

        assert status == "divergences"
        assert any("divergen" in w.lower() for w in warnings)


class TestComputeDivergenceRate:
    """Tests for compute_divergence_rate function."""

    def test_no_divergences(self):
        """Test rate is 0 with no divergences."""
        rate = compute_divergence_rate(0, n_samples=1000, n_chains=4)
        assert rate == 0.0

    def test_some_divergences(self):
        """Test rate calculation with divergences."""
        # 100 divergences out of 4000 total samples
        rate = compute_divergence_rate(100, n_samples=1000, n_chains=4)
        assert rate == pytest.approx(0.025)

    def test_all_divergences(self):
        """Test rate is 1.0 when all samples diverge."""
        rate = compute_divergence_rate(4000, n_samples=1000, n_chains=4)
        assert rate == 1.0


class TestSummarizeDiagnostics:
    """Tests for summarize_diagnostics function."""

    def test_summary_structure(self):
        """Test summary returns expected string format."""
        r_hat = {"D0": 1.01, "alpha": 1.02}
        ess_bulk = {"D0": 500.0, "alpha": 600.0}

        summary = summarize_diagnostics(
            r_hat=r_hat,
            ess_bulk=ess_bulk,
            divergences=0,
            n_samples=2000,
            n_chains=4,
        )

        # Check that it returns a string
        assert isinstance(summary, str)
        # Should contain key diagnostic info
        assert "R-hat" in summary
        assert "ESS" in summary

    def test_summary_values(self):
        """Test summary includes correct values."""
        r_hat = {"D0": 1.01, "alpha": 1.05}
        ess_bulk = {"D0": 500.0, "alpha": 200.0}

        summary = summarize_diagnostics(
            r_hat=r_hat,
            ess_bulk=ess_bulk,
            divergences=10,
            n_samples=2000,
            n_chains=4,
        )

        # Should contain divergence info
        assert "divergences" in summary.lower()
        assert "10" in summary


class TestCreateDiagnosticsDict:
    """Tests for create_diagnostics_dict function."""

    def test_diagnostics_dict_structure(self):
        """Test diagnostics dict has all required fields."""
        r_hat = {"D0": 1.01, "alpha": 1.02}
        ess_bulk = {"D0": 500.0, "alpha": 600.0}
        ess_tail = {"D0": 400.0, "alpha": 500.0}

        diag = create_diagnostics_dict(
            r_hat=r_hat,
            ess_bulk=ess_bulk,
            ess_tail=ess_tail,
            divergences=5,
            convergence_status="converged",
            warnings=[],
            n_chains=4,
            n_warmup=500,
            n_samples=2000,
            warmup_time=60.0,
            sampling_time=60.0,
        )

        required_fields = [
            "convergence_status",
            "total_divergences",
        ]

        for field in required_fields:
            assert field in diag, f"Missing field: {field}"

    def test_diagnostics_dict_values(self):
        """Test diagnostics dict has correct values."""
        r_hat = {"D0": 1.01, "alpha": 1.02}
        ess_bulk = {"D0": 500.0, "alpha": 600.0}
        ess_tail = {"D0": 400.0, "alpha": 500.0}

        diag = create_diagnostics_dict(
            r_hat=r_hat,
            ess_bulk=ess_bulk,
            ess_tail=ess_tail,
            divergences=5,
            convergence_status="converged",
            warnings=[],
            n_chains=4,
            n_warmup=500,
            n_samples=2000,
            warmup_time=60.0,
            sampling_time=60.0,
        )

        assert diag["convergence_status"] == "converged"
        assert diag["total_divergences"] == 5

    def test_diagnostics_dict_timing(self):
        """Test timing section has correct values."""
        r_hat = {"D0": 1.01, "alpha": 1.02}
        ess_bulk = {"D0": 500.0, "alpha": 600.0}
        ess_tail = {"D0": 400.0, "alpha": 500.0}

        diag = create_diagnostics_dict(
            r_hat=r_hat,
            ess_bulk=ess_bulk,
            ess_tail=ess_tail,
            divergences=5,
            convergence_status="converged",
            warnings=[],
            n_chains=4,
            n_warmup=500,
            n_samples=2000,
            warmup_time=60.0,
            sampling_time=60.0,
        )

        assert "timing" in diag
        assert diag["timing"]["warmup_seconds"] == 60.0
