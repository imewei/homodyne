"""Tests for CMC diagnostics module."""

import numpy as np
import pytest

pytest.importorskip("arviz", reason="ArviZ required for CMC unit tests")

from homodyne.optimization.cmc.diagnostics import (  # noqa: E402
    check_convergence,
    create_diagnostics_dict,
    get_convergence_recommendations,
    log_analysis_summary,
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


class TestGetConvergenceRecommendations:
    """Tests for get_convergence_recommendations function."""

    def test_no_recommendations_when_converged(self):
        """Test no recommendations for good convergence."""
        recommendations = get_convergence_recommendations(
            max_rhat=1.01,
            min_ess=500.0,
            divergences=0,
            n_samples=2000,
            n_chains=4,
        )
        assert len(recommendations) == 0

    def test_high_rhat_recommendation(self):
        """Test recommendation generated for high R-hat."""
        recommendations = get_convergence_recommendations(
            max_rhat=1.15,
            min_ess=500.0,
            divergences=0,
            n_samples=2000,
            n_chains=4,
        )
        assert len(recommendations) > 0
        assert any("R-HAT" in r for r in recommendations)

    def test_marginal_rhat_recommendation(self):
        """Test recommendation for marginal R-hat (1.05 < rhat <= 1.1)."""
        recommendations = get_convergence_recommendations(
            max_rhat=1.07,
            min_ess=500.0,
            divergences=0,
            n_samples=2000,
            n_chains=4,
        )
        assert len(recommendations) > 0
        assert any("MARGINAL" in r for r in recommendations)

    def test_low_ess_recommendation(self):
        """Test recommendation for low ESS."""
        recommendations = get_convergence_recommendations(
            max_rhat=1.01,
            min_ess=50.0,
            divergences=0,
            n_samples=2000,
            n_chains=4,
        )
        assert len(recommendations) > 0
        assert any("ESS" in r for r in recommendations)

    def test_moderate_ess_recommendation(self):
        """Test recommendation for moderate ESS (100 <= ess < 400)."""
        recommendations = get_convergence_recommendations(
            max_rhat=1.01,
            min_ess=200.0,
            divergences=0,
            n_samples=2000,
            n_chains=4,
        )
        assert len(recommendations) > 0
        assert any("MODERATE ESS" in r for r in recommendations)

    def test_high_divergence_recommendation(self):
        """Test recommendation for high divergence rate."""
        # 15% divergence rate
        recommendations = get_convergence_recommendations(
            max_rhat=1.01,
            min_ess=500.0,
            divergences=1200,
            n_samples=2000,
            n_chains=4,
        )
        assert len(recommendations) > 0
        assert any("DIVERGENCE" in r for r in recommendations)

    def test_moderate_divergence_recommendation(self):
        """Test recommendation for moderate divergence rate."""
        # 2% divergence rate
        recommendations = get_convergence_recommendations(
            max_rhat=1.01,
            min_ess=500.0,
            divergences=160,
            n_samples=2000,
            n_chains=4,
        )
        assert len(recommendations) > 0
        assert any("DIVERGENCE" in r for r in recommendations)


class TestLogAnalysisSummary:
    """Tests for log_analysis_summary function."""

    def test_log_analysis_summary_runs(self):
        """Test that log_analysis_summary runs without error."""
        r_hat = {"D0": 1.01, "alpha": 1.02}
        ess_bulk = {"D0": 500.0, "alpha": 600.0}

        # Should not raise any errors
        log_analysis_summary(
            convergence_status="converged",
            r_hat=r_hat,
            ess_bulk=ess_bulk,
            divergences=0,
            n_samples=2000,
            n_chains=4,
            n_shards=3,
            shards_succeeded=3,
            execution_time=120.0,
        )

    def test_log_analysis_summary_with_warnings(self):
        """Test log_analysis_summary with convergence issues."""
        r_hat = {"D0": 1.15, "alpha": 1.02}  # High R-hat
        ess_bulk = {"D0": 50.0, "alpha": 600.0}  # Low ESS

        # Should not raise any errors
        log_analysis_summary(
            convergence_status="not_converged",
            r_hat=r_hat,
            ess_bulk=ess_bulk,
            divergences=100,
            n_samples=2000,
            n_chains=4,
            n_shards=5,
            shards_succeeded=4,
            execution_time=300.0,
        )


# =============================================================================
# Bimodal Detection Tests
# =============================================================================


class TestBimodalDetection:
    """Tests for GMM-based bimodal detection."""

    def test_unimodal_samples_not_detected_as_bimodal(self):
        """Unimodal samples should not be flagged as bimodal."""
        from homodyne.optimization.cmc.diagnostics import detect_bimodal

        # Generate unimodal samples
        rng = np.random.default_rng(42)
        samples = rng.normal(loc=100.0, scale=10.0, size=1000)

        result = detect_bimodal(samples)

        assert result.is_bimodal == False  # noqa: E712 (numpy bool)

    def test_bimodal_samples_detected(self):
        """Clearly bimodal samples should be detected."""
        from homodyne.optimization.cmc.diagnostics import detect_bimodal

        # Generate bimodal samples (two well-separated modes)
        rng = np.random.default_rng(42)
        mode1 = rng.normal(loc=-50.0, scale=5.0, size=500)
        mode2 = rng.normal(loc=50.0, scale=5.0, size=500)
        samples = np.concatenate([mode1, mode2])

        result = detect_bimodal(samples)

        assert result.is_bimodal == True  # noqa: E712 (numpy bool)
        assert min(result.weights) > 0.2  # Both modes significant
        assert result.separation > 50.0  # Modes well separated

    def test_bimodal_result_contains_expected_fields(self):
        """BimodalResult has all expected fields."""
        from homodyne.optimization.cmc.diagnostics import detect_bimodal

        rng = np.random.default_rng(42)
        samples = rng.normal(loc=0.0, scale=1.0, size=100)

        result = detect_bimodal(samples)

        assert hasattr(result, "is_bimodal")
        assert hasattr(result, "weights")
        assert hasattr(result, "means")
        assert hasattr(result, "separation")
        assert hasattr(result, "relative_separation")
        assert len(result.weights) == 2
        assert len(result.means) == 2

    def test_unbalanced_bimodal_not_detected(self):
        """Unbalanced modes (one < 20%) should not be flagged."""
        from homodyne.optimization.cmc.diagnostics import detect_bimodal

        # 90% from mode1, 10% from mode2
        rng = np.random.default_rng(42)
        mode1 = rng.normal(loc=0.0, scale=1.0, size=900)
        mode2 = rng.normal(loc=10.0, scale=1.0, size=100)
        samples = np.concatenate([mode1, mode2])

        result = detect_bimodal(samples, min_weight=0.2)

        assert result.is_bimodal == False  # noqa: E712 (numpy bool)

    def test_bimodal_result_contains_stds(self):
        """BimodalResult should include per-component standard deviations."""
        from homodyne.optimization.cmc.diagnostics import detect_bimodal

        rng = np.random.default_rng(42)
        mode1 = rng.normal(loc=-50.0, scale=5.0, size=500)
        mode2 = rng.normal(loc=50.0, scale=5.0, size=500)
        samples = np.concatenate([mode1, mode2])

        result = detect_bimodal(samples)

        assert hasattr(result, "stds")
        assert len(result.stds) == 2
        # Each component std should be roughly 5.0 (the generating std)
        assert all(1.0 < s < 20.0 for s in result.stds)


class TestCheckShardBimodality:
    """Tests for check_shard_bimodality function."""

    def test_checks_multiple_parameters(self):
        """Function checks all specified parameters."""
        from homodyne.optimization.cmc.diagnostics import check_shard_bimodality

        rng = np.random.default_rng(42)
        samples = {
            "D0": rng.normal(20000, 1000, size=500),
            "alpha": rng.normal(-1.0, 0.1, size=500),
            "D_offset": rng.normal(1000, 100, size=500),
        }

        results = check_shard_bimodality(samples, params_to_check=["D0", "alpha"])

        assert "D0" in results
        assert "alpha" in results
        assert "D_offset" not in results  # Not in params_to_check


class TestModeClusterTypes:
    """Tests for ModeCluster and BimodalConsensusResult dataclasses."""

    def test_mode_cluster_fields(self):
        """ModeCluster has all required fields."""
        from homodyne.optimization.cmc.diagnostics import ModeCluster

        cluster = ModeCluster(
            mean={"D0": 19000.0, "alpha": -1.5},
            std={"D0": 1200.0, "alpha": 0.12},
            weight=0.55,
            n_shards=85,
            samples={"D0": np.ones((2, 100)), "alpha": np.ones((2, 100))},
        )
        assert cluster.weight == 0.55
        assert cluster.n_shards == 85
        assert cluster.mean["D0"] == 19000.0

    def test_bimodal_consensus_result_fields(self):
        """BimodalConsensusResult has all required fields."""
        from homodyne.optimization.cmc.diagnostics import (
            BimodalConsensusResult,
            ModeCluster,
        )

        mode_a = ModeCluster(
            mean={"D0": 19000.0},
            std={"D0": 1200.0},
            weight=0.55,
            n_shards=85,
            samples={"D0": np.ones((2, 100))},
        )
        mode_b = ModeCluster(
            mean={"D0": 32000.0},
            std={"D0": 2100.0},
            weight=0.45,
            n_shards=70,
            samples={"D0": np.ones((2, 100))},
        )
        result = BimodalConsensusResult(
            modes=[mode_a, mode_b],
            modal_params=["D0"],
            co_occurrence={"d0_alpha_fraction": 0.67},
        )
        assert len(result.modes) == 2
        assert result.modal_params == ["D0"]
        assert abs(result.modes[0].weight + result.modes[1].weight - 1.0) < 0.01


class TestClusterShardModes:
    """Tests for joint mode clustering of shard posteriors."""

    def _make_shard_samples(
        self,
        n_shards: int,
        bimodal_shards: set[int],
        mode1_center: float = 19000.0,
        mode2_center: float = 32000.0,
    ) -> list:
        """Create mock MCMCSamples-like objects for testing."""
        from types import SimpleNamespace

        rng = np.random.default_rng(42)
        shards = []
        for i in range(n_shards):
            if i in bimodal_shards:
                d0 = np.concatenate(
                    [
                        rng.normal(mode1_center, 1200, size=500),
                        rng.normal(mode2_center, 2100, size=500),
                    ]
                )
                alpha_lo, alpha_hi = -1.5, -0.4
                alpha = np.concatenate(
                    [
                        rng.normal(alpha_lo, 0.12, size=500),
                        rng.normal(alpha_hi, 0.09, size=500),
                    ]
                )
            else:
                center = mode1_center if i % 3 != 0 else mode2_center
                d0 = rng.normal(center, 1200, size=1000)
                alpha_center = -1.5 if center == mode1_center else -0.4
                alpha = rng.normal(alpha_center, 0.12, size=1000)
            shards.append(
                SimpleNamespace(
                    samples={"D0": d0.reshape(2, 500), "alpha": alpha.reshape(2, 500)},
                )
            )
        return shards

    def test_cluster_assigns_all_shards(self):
        """Every shard should be assigned to at least one cluster."""
        from homodyne.optimization.cmc.diagnostics import cluster_shard_modes

        shards = self._make_shard_samples(20, bimodal_shards={3, 7, 12})
        detections = [
            {
                "shard": 3,
                "param": "D0",
                "mode1": 19200,
                "mode2": 31800,
                "std1": 1200,
                "std2": 2100,
                "weights": (0.5, 0.5),
                "separation": 12600,
            },
            {
                "shard": 7,
                "param": "D0",
                "mode1": 18800,
                "mode2": 32200,
                "std1": 1100,
                "std2": 2000,
                "weights": (0.48, 0.52),
                "separation": 13400,
            },
            {
                "shard": 12,
                "param": "D0",
                "mode1": 19500,
                "mode2": 31500,
                "std1": 1300,
                "std2": 2200,
                "weights": (0.51, 0.49),
                "separation": 12000,
            },
        ]
        summary = {
            "per_param": {
                "D0": {
                    "lower_mean": 19200,
                    "upper_mean": 31800,
                    "lower_std": 300,
                    "upper_std": 400,
                    "bimodal_fraction": 0.15,
                    "n_detections": 3,
                },
            },
            "co_occurrence": {},
        }
        bounds = {"D0": (5000.0, 50000.0), "alpha": (-3.0, 0.0)}

        assignments = cluster_shard_modes(
            bimodal_detections=detections,
            successful_samples=shards,
            bimodal_summary=summary,
            param_bounds=bounds,
        )

        # Every shard should appear in the union of both clusters
        all_assigned = set(assignments[0]) | set(assignments[1])
        assert all_assigned == set(range(20))

    def test_bimodal_shards_split_between_clusters(self):
        """Bimodal shards should contribute components to both clusters."""
        from homodyne.optimization.cmc.diagnostics import cluster_shard_modes

        shards = self._make_shard_samples(10, bimodal_shards={2, 5})
        detections = [
            {
                "shard": 2,
                "param": "D0",
                "mode1": 19000,
                "mode2": 32000,
                "std1": 1200,
                "std2": 2100,
                "weights": (0.5, 0.5),
                "separation": 13000,
            },
            {
                "shard": 5,
                "param": "D0",
                "mode1": 19100,
                "mode2": 31900,
                "std1": 1100,
                "std2": 2000,
                "weights": (0.52, 0.48),
                "separation": 12800,
            },
        ]
        summary = {
            "per_param": {
                "D0": {
                    "lower_mean": 19050,
                    "upper_mean": 31950,
                    "lower_std": 70,
                    "upper_std": 70,
                    "bimodal_fraction": 0.2,
                    "n_detections": 2,
                },
            },
            "co_occurrence": {},
        }
        bounds = {"D0": (5000.0, 50000.0)}

        assignments = cluster_shard_modes(
            bimodal_detections=detections,
            successful_samples=shards,
            bimodal_summary=summary,
            param_bounds=bounds,
        )

        # Bimodal shards 2 and 5 should appear in BOTH clusters
        assert 2 in assignments[0] and 2 in assignments[1]
        assert 5 in assignments[0] and 5 in assignments[1]

    def test_empty_detections_returns_single_cluster(self):
        """With no bimodal detections, all shards go in one cluster."""
        from homodyne.optimization.cmc.diagnostics import cluster_shard_modes

        shards = self._make_shard_samples(5, bimodal_shards=set())
        assignments = cluster_shard_modes(
            bimodal_detections=[],
            successful_samples=shards,
            bimodal_summary={"per_param": {}, "co_occurrence": {}},
            param_bounds={"D0": (5000.0, 50000.0)},
        )

        # All shards in cluster 0, none in cluster 1
        assert set(assignments[0]) == set(range(5))
        assert len(assignments[1]) == 0


def test_mcmc_samples_has_bimodal_consensus_field():
    """MCMCSamples should have an optional bimodal_consensus field."""
    from homodyne.optimization.cmc.sampler import MCMCSamples

    samples = MCMCSamples(
        samples={"D0": np.ones((2, 100))},
        param_names=["D0"],
        n_chains=2,
        n_samples=100,
    )
    # Default should be None
    assert samples.bimodal_consensus is None
