# Mode-Aware Consensus MC Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix consensus combination to handle bimodal per-shard posteriors by extracting per-component GMM statistics, jointly clustering shards into mode populations, and running separate consensus MC per mode.

**Architecture:** Enrich `BimodalResult` with per-component stds from GMM. Add joint mode clustering in `diagnostics.py`. Add bimodal-aware combination in `backends/base.py`. Wire the mode-aware path in `multiprocessing.py` before the existing combine step. Attach `BimodalConsensusResult` metadata to `MCMCSamples`.

**Tech Stack:** NumPy, scikit-learn (GaussianMixture — already used), dataclasses

---

### Task 1: Add `stds` field to `BimodalResult` and extract from GMM

**Files:**
- Modify: `homodyne/optimization/cmc/diagnostics.py:853-921`
- Test: `tests/unit/optimization/cmc/test_diagnostics.py`

**Step 1: Write the failing test**

Add to the `TestBimodalDetection` class in `tests/unit/optimization/cmc/test_diagnostics.py` (after line 440):

```python
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/optimization/cmc/test_diagnostics.py::TestBimodalDetection::test_bimodal_result_contains_stds -v`
Expected: FAIL with `AttributeError: 'BimodalResult' object has no attribute 'stds'`

**Step 3: Implement — add `stds` to `BimodalResult` and `detect_bimodal()`**

In `homodyne/optimization/cmc/diagnostics.py`:

1. Add field to `BimodalResult` dataclass (after `means`, before `separation`):

```python
@dataclass
class BimodalResult:
    """Result of bimodal detection for a single parameter.

    Attributes
    ----------
    is_bimodal : bool
        Whether the posterior appears bimodal.
    weights : tuple[float, float]
        Component weights from GMM.
    means : tuple[float, float]
        Component means from GMM.
    stds : tuple[float, float]
        Component standard deviations from GMM.
    separation : float
        Absolute distance between means.
    relative_separation : float
        Separation relative to scale (separation / |mean(means)|).
    """

    is_bimodal: bool
    weights: tuple[float, float]
    means: tuple[float, float]
    stds: tuple[float, float]
    separation: float
    relative_separation: float
```

2. Extract stds in `detect_bimodal()` (after `means = ...`, before `separation = ...`):

```python
    stds = tuple(np.sqrt(gmm.covariances_.flatten()).tolist())
```

3. Pass `stds` to `BimodalResult` constructor:

```python
    return BimodalResult(
        is_bimodal=is_bimodal,
        weights=weights,
        means=means,
        stds=stds,
        separation=separation,
        relative_separation=relative_separation,
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/optimization/cmc/test_diagnostics.py::TestBimodalDetection -v`
Expected: All 5 tests PASS (including the new one)

**Step 5: Run full diagnostics test suite to check for regressions**

Run: `uv run pytest tests/unit/optimization/cmc/test_diagnostics.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add homodyne/optimization/cmc/diagnostics.py tests/unit/optimization/cmc/test_diagnostics.py
git commit -m "feat(diagnostics): add per-component stds to BimodalResult

Extract GMM component standard deviations (from gmm.covariances_)
and store in BimodalResult.stds. Required for mode-aware consensus
which needs per-component variance, not inflated full-posterior variance."
```

---

### Task 2: Add `ModeCluster` and `BimodalConsensusResult` dataclasses

**Files:**
- Modify: `homodyne/optimization/cmc/diagnostics.py` (after `BimodalResult`, ~line 876)
- Test: `tests/unit/optimization/cmc/test_diagnostics.py`

**Step 1: Write the failing test**

Add new test class in `tests/unit/optimization/cmc/test_diagnostics.py`:

```python
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
            mean={"D0": 19000.0}, std={"D0": 1200.0},
            weight=0.55, n_shards=85, samples={"D0": np.ones((2, 100))},
        )
        mode_b = ModeCluster(
            mean={"D0": 32000.0}, std={"D0": 2100.0},
            weight=0.45, n_shards=70, samples={"D0": np.ones((2, 100))},
        )
        result = BimodalConsensusResult(
            modes=[mode_a, mode_b],
            modal_params=["D0"],
            co_occurrence={"d0_alpha_fraction": 0.67},
        )
        assert len(result.modes) == 2
        assert result.modal_params == ["D0"]
        assert abs(result.modes[0].weight + result.modes[1].weight - 1.0) < 0.01
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/optimization/cmc/test_diagnostics.py::TestModeClusterTypes -v`
Expected: FAIL with `ImportError: cannot import name 'ModeCluster'`

**Step 3: Implement — add dataclasses after `BimodalResult`**

In `homodyne/optimization/cmc/diagnostics.py`, after the `BimodalResult` dataclass (after line ~876, before `detect_bimodal`):

```python
@dataclass
class ModeCluster:
    """A single mode from bimodal consensus combination.

    Attributes
    ----------
    mean : dict[str, float]
        Per-parameter consensus mean for this mode.
    std : dict[str, float]
        Per-parameter consensus std for this mode.
    weight : float
        Fraction of shards supporting this mode (0-1).
    n_shards : int
        Number of shards in this cluster.
    samples : dict[str, np.ndarray]
        Generated samples from N(mean, std^2), shape (n_chains, n_samples).
    """

    mean: dict[str, float]
    std: dict[str, float]
    weight: float
    n_shards: int
    samples: dict[str, np.ndarray]


@dataclass
class BimodalConsensusResult:
    """Result of mode-aware consensus combination.

    Attached to MCMCSamples when bimodal posteriors are detected and
    per-mode consensus is used instead of standard combination.

    Attributes
    ----------
    modes : list[ModeCluster]
        Mode clusters (typically 2) with per-mode consensus statistics.
    modal_params : list[str]
        Parameter names that triggered bimodal detection.
    co_occurrence : dict[str, Any]
        Cross-parameter co-occurrence info (e.g., D0-alpha correlation).
    """

    modes: list[ModeCluster]
    modal_params: list[str]
    co_occurrence: dict[str, Any]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/optimization/cmc/test_diagnostics.py::TestModeClusterTypes -v`
Expected: PASS

**Step 5: Commit**

```bash
git add homodyne/optimization/cmc/diagnostics.py tests/unit/optimization/cmc/test_diagnostics.py
git commit -m "feat(diagnostics): add ModeCluster and BimodalConsensusResult types

Dataclasses for mode-aware consensus output. ModeCluster holds
per-mode consensus statistics; BimodalConsensusResult holds the
full bimodal analysis attached to MCMCSamples."
```

---

### Task 3: Implement `cluster_shard_modes()` in `diagnostics.py`

**Files:**
- Modify: `homodyne/optimization/cmc/diagnostics.py` (after `summarize_cross_shard_bimodality`, ~line 1062)
- Test: `tests/unit/optimization/cmc/test_diagnostics.py`

**Step 1: Write the failing tests**

Add new test class in `tests/unit/optimization/cmc/test_diagnostics.py`:

```python
class TestClusterShardModes:
    """Tests for joint mode clustering of shard posteriors."""

    def _make_shard_samples(
        self, n_shards: int, bimodal_shards: set[int],
        mode1_center: float = 19000.0, mode2_center: float = 32000.0,
    ) -> list:
        """Create mock MCMCSamples-like objects for testing."""
        from types import SimpleNamespace

        rng = np.random.default_rng(42)
        shards = []
        for i in range(n_shards):
            if i in bimodal_shards:
                # Bimodal: 50/50 split
                d0 = np.concatenate([
                    rng.normal(mode1_center, 1200, size=500),
                    rng.normal(mode2_center, 2100, size=500),
                ])
                alpha_lo, alpha_hi = -1.5, -0.4
                alpha = np.concatenate([
                    rng.normal(alpha_lo, 0.12, size=500),
                    rng.normal(alpha_hi, 0.09, size=500),
                ])
            else:
                # Unimodal: converged to mode 1 or mode 2
                center = mode1_center if i % 3 != 0 else mode2_center
                d0 = rng.normal(center, 1200, size=1000)
                alpha_center = -1.5 if center == mode1_center else -0.4
                alpha = rng.normal(alpha_center, 0.12, size=1000)
            shards.append(SimpleNamespace(
                samples={"D0": d0.reshape(2, 500), "alpha": alpha.reshape(2, 500)},
            ))
        return shards

    def test_cluster_assigns_all_shards(self):
        """Every shard should be assigned to exactly one cluster."""
        from homodyne.optimization.cmc.diagnostics import cluster_shard_modes

        shards = self._make_shard_samples(20, bimodal_shards={3, 7, 12})
        detections = [
            {"shard": 3, "param": "D0", "mode1": 19200, "mode2": 31800,
             "std1": 1200, "std2": 2100, "weights": (0.5, 0.5), "separation": 12600},
            {"shard": 7, "param": "D0", "mode1": 18800, "mode2": 32200,
             "std1": 1100, "std2": 2000, "weights": (0.48, 0.52), "separation": 13400},
            {"shard": 12, "param": "D0", "mode1": 19500, "mode2": 31500,
             "std1": 1300, "std2": 2200, "weights": (0.51, 0.49), "separation": 12000},
        ]
        summary = {
            "per_param": {
                "D0": {"lower_mean": 19200, "upper_mean": 31800,
                       "lower_std": 300, "upper_std": 400,
                       "bimodal_fraction": 0.15, "n_detections": 3},
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

        # Every shard should appear in exactly one cluster
        all_assigned = set(assignments[0]) | set(assignments[1])
        assert all_assigned == set(range(20))
        # No overlap
        assert len(set(assignments[0]) & set(assignments[1])) == 0

    def test_bimodal_shards_split_between_clusters(self):
        """Bimodal shards should contribute components to both clusters."""
        from homodyne.optimization.cmc.diagnostics import cluster_shard_modes

        shards = self._make_shard_samples(10, bimodal_shards={2, 5})
        detections = [
            {"shard": 2, "param": "D0", "mode1": 19000, "mode2": 32000,
             "std1": 1200, "std2": 2100, "weights": (0.5, 0.5), "separation": 13000},
            {"shard": 5, "param": "D0", "mode1": 19100, "mode2": 31900,
             "std1": 1100, "std2": 2000, "weights": (0.52, 0.48), "separation": 12800},
        ]
        summary = {
            "per_param": {
                "D0": {"lower_mean": 19050, "upper_mean": 31950,
                       "lower_std": 70, "upper_std": 70,
                       "bimodal_fraction": 0.2, "n_detections": 2},
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
        # (lower component in one, upper in the other)
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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/optimization/cmc/test_diagnostics.py::TestClusterShardModes -v`
Expected: FAIL with `ImportError: cannot import name 'cluster_shard_modes'`

**Step 3: Implement `cluster_shard_modes()`**

In `homodyne/optimization/cmc/diagnostics.py`, after `summarize_cross_shard_bimodality()`:

```python
def cluster_shard_modes(
    bimodal_detections: list[dict[str, Any]],
    successful_samples: list[Any],
    bimodal_summary: dict[str, Any],
    param_bounds: dict[str, tuple[float, float]],
) -> tuple[list[int], list[int]]:
    """Jointly cluster shards into two mode populations.

    Uses range-normalized feature vectors from modal parameters to assign
    each shard to the nearest mode centroid. Bimodal shards contribute
    one component to each cluster.

    Parameters
    ----------
    bimodal_detections : list[dict[str, Any]]
        Per-detection records with keys: "shard", "param", "mode1", "mode2",
        "std1", "std2", "weights", "separation".
    successful_samples : list[Any]
        List of MCMCSamples (or similar with .samples dict attribute).
    bimodal_summary : dict[str, Any]
        Output from summarize_cross_shard_bimodality().
    param_bounds : dict[str, tuple[float, float]]
        Parameter bounds for range-based normalization, {param: (lo, hi)}.

    Returns
    -------
    tuple[list[int], list[int]]
        (cluster_0_shards, cluster_1_shards) where cluster_0 is "lower" and
        cluster_1 is "upper". Bimodal shards appear in both lists.
    """
    per_param = bimodal_summary.get("per_param", {})
    modal_params = sorted(per_param.keys())
    n_shards = len(successful_samples)

    if not modal_params:
        # No significant bimodality — all shards in cluster 0
        return list(range(n_shards)), []

    # Build centroids from cross-shard summary (lower/upper means)
    centroid_lower = []
    centroid_upper = []
    scales = []
    for param in modal_params:
        stats = per_param[param]
        centroid_lower.append(stats["lower_mean"])
        centroid_upper.append(stats["upper_mean"])
        # Range-based normalization
        lo, hi = param_bounds.get(param, (0.0, 1.0))
        param_range = abs(hi - lo)
        scales.append(max(param_range, 1e-10))

    scales_arr = np.array(scales)
    centroid_lower_norm = np.array(centroid_lower) / scales_arr
    centroid_upper_norm = np.array(centroid_upper) / scales_arr

    # Index bimodal detections by shard for fast lookup
    bimodal_by_shard: dict[int, dict[str, dict[str, Any]]] = {}
    for det in bimodal_detections:
        shard_idx = det["shard"]
        param = det["param"]
        if param in modal_params:
            bimodal_by_shard.setdefault(shard_idx, {})[param] = det

    cluster_lower: list[int] = []
    cluster_upper: list[int] = []

    for shard_idx in range(n_shards):
        shard_bimodal = bimodal_by_shard.get(shard_idx, {})

        if shard_bimodal:
            # Bimodal shard: assign to BOTH clusters
            cluster_lower.append(shard_idx)
            cluster_upper.append(shard_idx)
        else:
            # Unimodal shard: compute feature vector and assign to nearest centroid
            feature = []
            for param in modal_params:
                if param in successful_samples[shard_idx].samples:
                    mean_val = float(
                        np.mean(successful_samples[shard_idx].samples[param])
                    )
                else:
                    # Fallback: use midpoint of centroids
                    idx = modal_params.index(param)
                    mean_val = (centroid_lower[idx] + centroid_upper[idx]) / 2
                feature.append(mean_val)

            feature_norm = np.array(feature) / scales_arr
            dist_lower = np.linalg.norm(feature_norm - centroid_lower_norm)
            dist_upper = np.linalg.norm(feature_norm - centroid_upper_norm)

            if dist_lower <= dist_upper:
                cluster_lower.append(shard_idx)
            else:
                cluster_upper.append(shard_idx)

    return cluster_lower, cluster_upper
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/optimization/cmc/test_diagnostics.py::TestClusterShardModes -v`
Expected: All 3 tests PASS

**Step 5: Run full diagnostics suite**

Run: `uv run pytest tests/unit/optimization/cmc/test_diagnostics.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add homodyne/optimization/cmc/diagnostics.py tests/unit/optimization/cmc/test_diagnostics.py
git commit -m "feat(diagnostics): add cluster_shard_modes() for joint mode clustering

Clusters shards into two mode populations using range-normalized
feature vectors from modal parameters. Bimodal shards contribute
one component to each cluster. Unimodal shards assigned by nearest
centroid distance."
```

---

### Task 4: Implement `combine_shard_samples_bimodal()` in `backends/base.py`

**Files:**
- Modify: `homodyne/optimization/cmc/backends/base.py` (after `_combine_shard_chunk`, ~line 441)
- Test: `tests/unit/optimization/cmc/test_backends.py`

**Step 1: Write the failing tests**

Add new test class in `tests/unit/optimization/cmc/test_backends.py`:

```python
class TestBimodalCombination:
    """Tests for mode-aware bimodal consensus combination."""

    def _make_samples(self, mean: float, std: float, n: int = 1000):
        """Create MCMCSamples-like object."""
        from homodyne.optimization.cmc.sampler import MCMCSamples

        rng = np.random.default_rng(42)
        return MCMCSamples(
            samples={"D0": rng.normal(mean, std, size=(2, n // 2))},
            param_names=["D0"],
            n_chains=2,
            n_samples=n // 2,
        )

    def test_bimodal_combine_produces_two_modes(self):
        """Bimodal combination should produce a BimodalConsensusResult with 2 modes."""
        from homodyne.optimization.cmc.backends.base import (
            combine_shard_samples_bimodal,
        )

        # 10 shards: 6 near mode1 (19K), 4 near mode2 (32K)
        rng = np.random.default_rng(123)
        shards = []
        for i in range(10):
            center = 19000.0 if i < 6 else 32000.0
            shards.append(self._make_samples(center, 1200.0))

        # Assignments: shards 0-5 in cluster 0 (lower), shards 6-9 in cluster 1 (upper)
        cluster_lower = list(range(6))
        cluster_upper = list(range(6, 10))

        # No bimodal shards in this simple case
        bimodal_detections = []

        combined, bimodal_result = combine_shard_samples_bimodal(
            shard_samples=shards,
            cluster_assignments=(cluster_lower, cluster_upper),
            bimodal_detections=bimodal_detections,
            modal_params=["D0"],
            co_occurrence={},
        )

        assert bimodal_result is not None
        assert len(bimodal_result.modes) == 2
        # Mode 0 should be near 19K, mode 1 near 32K
        assert abs(bimodal_result.modes[0].mean["D0"] - 19000) < 2000
        assert abs(bimodal_result.modes[1].mean["D0"] - 32000) < 2000
        # Weights should reflect shard counts
        assert abs(bimodal_result.modes[0].weight - 0.6) < 0.05
        assert abs(bimodal_result.modes[1].weight - 0.4) < 0.05

    def test_bimodal_combine_uses_component_stats_for_bimodal_shards(self):
        """For bimodal shards, the combination should use component-level stats."""
        from homodyne.optimization.cmc.backends.base import (
            combine_shard_samples_bimodal,
        )
        from homodyne.optimization.cmc.sampler import MCMCSamples

        rng = np.random.default_rng(42)
        # 5 shards, shard 2 is bimodal (both modes)
        shards = []
        for i in range(5):
            if i == 2:
                # Bimodal shard: 50/50 mix
                d0 = np.concatenate([
                    rng.normal(19000, 1200, size=250),
                    rng.normal(32000, 2100, size=250),
                ])
            else:
                center = 19000 if i < 3 else 32000
                d0 = rng.normal(center, 1200, size=500)
            shards.append(MCMCSamples(
                samples={"D0": d0.reshape(2, 250)},
                param_names=["D0"], n_chains=2, n_samples=250,
            ))

        # Shard 2 appears in both clusters
        cluster_lower = [0, 1, 2]
        cluster_upper = [2, 3, 4]
        detections = [
            {"shard": 2, "param": "D0", "mode1": 19000, "mode2": 32000,
             "std1": 1200, "std2": 2100, "weights": (0.5, 0.5), "separation": 13000},
        ]

        combined, bimodal_result = combine_shard_samples_bimodal(
            shard_samples=shards,
            cluster_assignments=(cluster_lower, cluster_upper),
            bimodal_detections=detections,
            modal_params=["D0"],
            co_occurrence={},
        )

        # Lower mode consensus should be near 19K (not pulled toward 25.5K)
        assert abs(bimodal_result.modes[0].mean["D0"] - 19000) < 3000
        # Upper mode consensus should be near 32K
        assert abs(bimodal_result.modes[1].mean["D0"] - 32000) < 3000

    def test_bimodal_combine_empty_cluster_fallback(self):
        """If one cluster has <3 shards, fall back to simple mean for it."""
        from homodyne.optimization.cmc.backends.base import (
            combine_shard_samples_bimodal,
        )

        shards = [self._make_samples(19000.0, 1200.0) for _ in range(8)]
        # Only 1 shard in upper cluster
        cluster_lower = list(range(7))
        cluster_upper = [7]

        combined, bimodal_result = combine_shard_samples_bimodal(
            shard_samples=shards,
            cluster_assignments=(cluster_lower, cluster_upper),
            bimodal_detections=[],
            modal_params=["D0"],
            co_occurrence={},
        )

        # Should still produce 2 modes (upper with simple mean)
        assert len(bimodal_result.modes) == 2
        assert bimodal_result.modes[1].n_shards == 1
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/optimization/cmc/test_backends.py::TestBimodalCombination -v`
Expected: FAIL with `ImportError: cannot import name 'combine_shard_samples_bimodal'`

**Step 3: Implement `combine_shard_samples_bimodal()`**

In `homodyne/optimization/cmc/backends/base.py`, after `_combine_shard_chunk()` (after line 441):

```python
def combine_shard_samples_bimodal(
    shard_samples: list[MCMCSamples],
    cluster_assignments: tuple[list[int], list[int]],
    bimodal_detections: list[dict[str, Any]],
    modal_params: list[str],
    co_occurrence: dict[str, Any],
    method: str = "consensus_mc",
) -> tuple[MCMCSamples, BimodalConsensusResult]:
    """Combine shard samples using mode-aware consensus.

    For bimodal shards, uses per-component GMM statistics instead of
    full-posterior statistics to avoid density-trough corruption.

    Parameters
    ----------
    shard_samples : list[MCMCSamples]
        All successful shard samples.
    cluster_assignments : tuple[list[int], list[int]]
        (lower_cluster_shards, upper_cluster_shards) from cluster_shard_modes().
        Bimodal shards may appear in both lists.
    bimodal_detections : list[dict[str, Any]]
        Per-detection records with "shard", "param", "mode1", "mode2",
        "std1", "std2", "weights".
    modal_params : list[str]
        Parameters that triggered bimodal detection.
    co_occurrence : dict[str, Any]
        Cross-parameter co-occurrence info.
    method : str
        Base combination method for non-modal params.

    Returns
    -------
    tuple[MCMCSamples, BimodalConsensusResult]
        (combined_samples, bimodal_result) where combined_samples has
        mixture-drawn primary samples and bimodal_result has per-mode details.
    """
    import numpy as np

    from homodyne.optimization.cmc.diagnostics import (
        BimodalConsensusResult,
        ModeCluster,
    )
    from homodyne.optimization.cmc.sampler import MCMCSamples

    cluster_lower, cluster_upper = cluster_assignments
    n_total = len(shard_samples)
    param_names = shard_samples[0].param_names
    n_chains = shard_samples[0].n_chains
    n_samples = shard_samples[0].n_samples
    rng = np.random.default_rng(42)

    # Index bimodal detections by (shard, param) for fast lookup
    bimodal_index: dict[tuple[int, str], dict[str, Any]] = {}
    for det in bimodal_detections:
        bimodal_index[(det["shard"], det["param"])] = det

    modal_set = set(modal_params)

    def _consensus_for_cluster(
        cluster_shards: list[int],
        is_lower: bool,
    ) -> dict[str, tuple[float, float]]:
        """Compute consensus (mean, std) for each param in a cluster.

        Returns dict of {param: (combined_mean, combined_std)}.
        """
        result: dict[str, tuple[float, float]] = {}

        for name in param_names:
            shard_means: list[float] = []
            shard_variances: list[float] = []

            for shard_idx in cluster_shards:
                key = (shard_idx, name)
                if name in modal_set and key in bimodal_index:
                    # Bimodal shard + modal param: use component-level stats
                    det = bimodal_index[key]
                    m1, m2 = det["mode1"], det["mode2"]
                    s1, s2 = det["std1"], det["std2"]
                    lo, hi = sorted([(m1, s1), (m2, s2)], key=lambda x: x[0])
                    if is_lower:
                        shard_means.append(lo[0])
                        shard_variances.append(lo[1] ** 2)
                    else:
                        shard_means.append(hi[0])
                        shard_variances.append(hi[1] ** 2)
                else:
                    # Unimodal shard or non-modal param: use full posterior
                    samples = shard_samples[shard_idx].samples[name].flatten()
                    shard_means.append(float(np.mean(samples)))
                    shard_variances.append(float(np.var(samples)))

            if len(shard_means) < 3:
                # Too few shards: use simple mean
                combined_mean = float(np.mean(shard_means)) if shard_means else 0.0
                combined_std = float(np.std(shard_means)) if len(shard_means) > 1 else 1e-6
            else:
                # Precision-weighted consensus
                precisions = [1.0 / max(v, 1e-10) for v in shard_variances]
                combined_precision = sum(precisions)
                combined_variance = 1.0 / combined_precision
                weighted_mean_sum = sum(
                    p * m for p, m in zip(precisions, shard_means)
                )
                combined_mean = combined_variance * weighted_mean_sum
                combined_std = np.sqrt(combined_variance)

            result[name] = (float(combined_mean), float(combined_std))

        return result

    # Run per-mode consensus
    lower_stats = _consensus_for_cluster(cluster_lower, is_lower=True)
    upper_stats = _consensus_for_cluster(cluster_upper, is_lower=False)

    # Build mode weights
    # For bimodal shards that appear in both clusters, count them once total
    unique_shards = set(cluster_lower) | set(cluster_upper)
    w_lower = len(cluster_lower) / max(len(unique_shards), 1)
    w_upper = len(cluster_upper) / max(len(unique_shards), 1)
    # Normalize (bimodal shards counted in both lists inflate the sum)
    total_w = w_lower + w_upper
    w_lower /= total_w
    w_upper /= total_w

    # Generate per-mode samples
    n_lower_samples = int(round(w_lower * n_samples))
    n_upper_samples = n_samples - n_lower_samples

    lower_samples: dict[str, np.ndarray] = {}
    upper_samples: dict[str, np.ndarray] = {}
    combined_samples: dict[str, np.ndarray] = {}

    for name in param_names:
        lo_mean, lo_std = lower_stats[name]
        up_mean, up_std = upper_stats[name]

        lower_samples[name] = rng.normal(
            loc=lo_mean, scale=max(lo_std, 1e-10),
            size=(n_chains, n_lower_samples),
        )
        upper_samples[name] = rng.normal(
            loc=up_mean, scale=max(up_std, 1e-10),
            size=(n_chains, n_upper_samples),
        )
        # Mixture-draw: concatenate and shuffle within each chain
        mixed = np.concatenate([lower_samples[name], upper_samples[name]], axis=1)
        for c in range(n_chains):
            rng.shuffle(mixed[c])
        combined_samples[name] = mixed

    # Build ModeCluster objects (with full n_samples per mode for metadata)
    mode_lower = ModeCluster(
        mean={n: lower_stats[n][0] for n in param_names},
        std={n: lower_stats[n][1] for n in param_names},
        weight=w_lower,
        n_shards=len(cluster_lower),
        samples={n: rng.normal(
            loc=lower_stats[n][0], scale=max(lower_stats[n][1], 1e-10),
            size=(n_chains, n_samples),
        ) for n in param_names},
    )
    mode_upper = ModeCluster(
        mean={n: upper_stats[n][0] for n in param_names},
        std={n: upper_stats[n][1] for n in param_names},
        weight=w_upper,
        n_shards=len(cluster_upper),
        samples={n: rng.normal(
            loc=upper_stats[n][0], scale=max(upper_stats[n][1], 1e-10),
            size=(n_chains, n_samples),
        ) for n in param_names},
    )

    bimodal_result = BimodalConsensusResult(
        modes=[mode_lower, mode_upper],
        modal_params=modal_params,
        co_occurrence=co_occurrence,
    )

    # Combine extra fields from all shards
    combined_extra: dict[str, Any] = {}
    for key in shard_samples[0].extra_fields.keys():
        all_extra = [
            s.extra_fields.get(key) for s in shard_samples if key in s.extra_fields
        ]
        if all_extra:
            try:
                if all_extra[0].ndim == 0:
                    combined_extra[key] = np.stack(all_extra, axis=0)
                else:
                    combined_extra[key] = np.concatenate(all_extra, axis=0)
            except Exception as e:
                logger.warning(f"Failed to combine extra field '{key}': {e}")
                combined_extra[key] = all_extra[0]

    combined = MCMCSamples(
        samples=combined_samples,
        param_names=param_names,
        n_chains=n_chains,
        n_samples=n_samples,
        extra_fields=combined_extra,
        num_shards=n_total,
    )

    return combined, bimodal_result
```

Also add the necessary import at the top of `base.py`:

```python
from typing import Any
```

(If not already present — check before adding.)

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/optimization/cmc/test_backends.py::TestBimodalCombination -v`
Expected: All 3 tests PASS

**Step 5: Run full backends test suite**

Run: `uv run pytest tests/unit/optimization/cmc/test_backends.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add homodyne/optimization/cmc/backends/base.py tests/unit/optimization/cmc/test_backends.py
git commit -m "feat(backends): add combine_shard_samples_bimodal()

Mode-aware consensus combination that uses per-component GMM
statistics for bimodal shards. Produces mixture-drawn output
samples and BimodalConsensusResult metadata."
```

---

### Task 5: Add `bimodal_consensus` field to `MCMCSamples`

**Files:**
- Modify: `homodyne/optimization/cmc/sampler.py` (MCMCSamples dataclass, ~line 287)
- Test: `tests/unit/optimization/cmc/test_diagnostics.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/optimization/cmc/test_diagnostics.py::test_mcmc_samples_has_bimodal_consensus_field -v`
Expected: FAIL with `AttributeError`

**Step 3: Add the field to MCMCSamples**

In `homodyne/optimization/cmc/sampler.py`, add to the `MCMCSamples` dataclass after `shard_adapted_n_warmup`:

```python
    bimodal_consensus: Any = None  # BimodalConsensusResult when mode-aware consensus used
```

Use `Any` to avoid circular import (BimodalConsensusResult is in diagnostics.py). The type annotation in the docstring is sufficient for documentation.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/optimization/cmc/test_diagnostics.py::test_mcmc_samples_has_bimodal_consensus_field -v`
Expected: PASS

**Step 5: Commit**

```bash
git add homodyne/optimization/cmc/sampler.py tests/unit/optimization/cmc/test_diagnostics.py
git commit -m "feat(sampler): add bimodal_consensus field to MCMCSamples

Optional field (default None) to carry BimodalConsensusResult
when mode-aware consensus is used. Uses Any type to avoid
circular import with diagnostics module."
```

---

### Task 6: Enrich bimodal detections with `std1`/`std2` in `multiprocessing.py`

**Files:**
- Modify: `homodyne/optimization/cmc/backends/multiprocessing.py:1432-1453`

**Step 1: Update the detection collection block**

In `multiprocessing.py`, modify the bimodal detection loop (lines 1432-1453) to include `std1`/`std2`:

Change the dict construction from:

```python
                    bimodal_detections.append(
                        {
                            "shard": i,
                            "param": param,
                            "mode1": result.means[0],
                            "mode2": result.means[1],
                            "weights": result.weights,
                            "separation": result.separation,
                        }
                    )
```

To:

```python
                    bimodal_detections.append(
                        {
                            "shard": i,
                            "param": param,
                            "mode1": result.means[0],
                            "mode2": result.means[1],
                            "std1": result.stds[0],
                            "std2": result.stds[1],
                            "weights": result.weights,
                            "separation": result.separation,
                        }
                    )
```

**Step 2: Run lint and type checks**

Run: `uv run ruff check homodyne/optimization/cmc/backends/multiprocessing.py`
Run: `uv run mypy homodyne/optimization/cmc/backends/multiprocessing.py`
Expected: Both pass

**Step 3: Commit**

```bash
git add homodyne/optimization/cmc/backends/multiprocessing.py
git commit -m "feat(multiprocessing): include component stds in bimodal detections

Add std1/std2 from BimodalResult.stds to the detection dicts.
Required for mode-aware consensus which needs per-component
variance instead of inflated full-posterior variance."
```

---

### Task 7: Wire mode-aware consensus path in `multiprocessing.py`

**Files:**
- Modify: `homodyne/optimization/cmc/backends/multiprocessing.py:1455-1477`
- Test: Run existing CMC test suite for regression

**Step 1: Update imports**

Add `cluster_shard_modes` to the import from diagnostics (line 28):

```python
from homodyne.optimization.cmc.diagnostics import (
    check_shard_bimodality,
    cluster_shard_modes,
    summarize_cross_shard_bimodality,
)
```

Add `combine_shard_samples_bimodal` to the import from base (line 27):

```python
from homodyne.optimization.cmc.backends.base import (
    CMCBackend,
    combine_shard_samples,
    combine_shard_samples_bimodal,
)
```

**Step 2: Replace the combine block**

Replace the block from the `if bimodal_detections:` section through `combined = combine_shard_samples(...)` (lines 1455-1477) with:

```python
        if bimodal_detections:
            # Compute pre-combine consensus means from per-shard posteriors
            consensus_means: dict[str, float] = {}
            key_params = ["D0", "alpha", "D_offset", "gamma_dot_t0", "beta"]
            for param in key_params:
                if param in successful_samples[0].samples:
                    means = [
                        float(np.mean(s.samples[param])) for s in successful_samples
                    ]
                    consensus_means[param] = float(np.mean(means))

            bimodal_summary = summarize_cross_shard_bimodality(
                bimodal_detections,
                n_shards=len(successful_samples),
                consensus_means=consensus_means,
            )
            _log_bimodality_summary(run_logger, bimodal_summary)

            # Mode-aware consensus if significant bimodality detected
            if bimodal_summary["per_param"]:
                modal_params = sorted(bimodal_summary["per_param"].keys())
                # Get parameter bounds for range normalization
                param_bounds: dict[str, tuple[float, float]] = {}
                if parameter_space is not None:
                    for param in modal_params:
                        try:
                            param_bounds[param] = parameter_space.get_bounds(param)
                        except (KeyError, ValueError):
                            pass

                mode_assignments = cluster_shard_modes(
                    bimodal_detections=bimodal_detections,
                    successful_samples=successful_samples,
                    bimodal_summary=bimodal_summary,
                    param_bounds=param_bounds,
                )

                run_logger.info(
                    f"Mode-aware consensus: cluster sizes = "
                    f"{len(mode_assignments[0])}, {len(mode_assignments[1])}"
                )

                combined, bimodal_result = combine_shard_samples_bimodal(
                    shard_samples=successful_samples,
                    cluster_assignments=mode_assignments,
                    bimodal_detections=bimodal_detections,
                    modal_params=modal_params,
                    co_occurrence=bimodal_summary.get("co_occurrence", {}),
                    method=config.combination_method,
                )
                combined.bimodal_consensus = bimodal_result

                # Log mode summary
                for i, mode in enumerate(bimodal_result.modes):
                    mode_means = ", ".join(
                        f"{p}={mode.mean[p]:.4g}" for p in modal_params
                        if p in mode.mean
                    )
                    run_logger.info(
                        f"  Mode {i}: weight={mode.weight:.2f}, "
                        f"n_shards={mode.n_shards}, {mode_means}"
                    )
            else:
                # Bimodal detections exist but below significance threshold
                combined = combine_shard_samples(
                    successful_samples,
                    method=config.combination_method,
                )
        else:
            # No bimodality detected — standard path
            combined = combine_shard_samples(
                successful_samples,
                method=config.combination_method,
            )
```

**Step 3: Run lint and type checks**

Run: `uv run ruff check homodyne/optimization/cmc/backends/multiprocessing.py`
Run: `uv run mypy homodyne/optimization/cmc/backends/multiprocessing.py`
Expected: Both pass

**Step 4: Run full CMC test suite for regression**

Run: `uv run pytest tests/unit/optimization/cmc/ -q --tb=short`
Expected: All 443+ tests PASS

**Step 5: Commit**

```bash
git add homodyne/optimization/cmc/backends/multiprocessing.py
git commit -m "feat(multiprocessing): wire mode-aware consensus path

When significant bimodality is detected, uses cluster_shard_modes()
and combine_shard_samples_bimodal() instead of standard consensus.
Falls back to standard combination when bimodality is below threshold
or absent."
```

---

### Task 8: Quality checks and final verification

**Step 1: Run full project lint**

Run: `uv run ruff check homodyne/optimization/cmc/diagnostics.py homodyne/optimization/cmc/backends/multiprocessing.py homodyne/optimization/cmc/backends/base.py homodyne/optimization/cmc/sampler.py`
Expected: All checks passed

**Step 2: Run full project type check**

Run: `uv run mypy homodyne/optimization/cmc/diagnostics.py homodyne/optimization/cmc/backends/multiprocessing.py homodyne/optimization/cmc/backends/base.py homodyne/optimization/cmc/sampler.py`
Expected: No issues found

**Step 3: Run full CMC test suite**

Run: `uv run pytest tests/unit/optimization/cmc/ -v --tb=short`
Expected: All tests PASS (443 original + ~10 new = ~453)

**Step 4: Run full project tests**

Run: `uv run pytest tests/unit/ -q --tb=short`
Expected: All tests PASS

---

## File Change Summary

| File | Changes |
|------|---------|
| `homodyne/optimization/cmc/diagnostics.py` | Add `stds` to `BimodalResult`; add `ModeCluster`, `BimodalConsensusResult` dataclasses; add `cluster_shard_modes()` |
| `homodyne/optimization/cmc/backends/base.py` | Add `combine_shard_samples_bimodal()` |
| `homodyne/optimization/cmc/backends/multiprocessing.py` | Enrich detections with `std1`/`std2`; wire mode-aware consensus branch |
| `homodyne/optimization/cmc/sampler.py` | Add `bimodal_consensus` field to `MCMCSamples` |
| `tests/unit/optimization/cmc/test_diagnostics.py` | Tests for `stds`, `ModeCluster`, `BimodalConsensusResult`, `cluster_shard_modes()` |
| `tests/unit/optimization/cmc/test_backends.py` | Tests for `combine_shard_samples_bimodal()` |
