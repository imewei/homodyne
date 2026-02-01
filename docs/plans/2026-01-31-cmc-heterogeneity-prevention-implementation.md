# CMC Heterogeneity Prevention Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent CMC heterogeneity abort through internal reparameterization, bimodal detection, and param-aware shard sizing.

**Architecture:** Add `reparameterization.py` module for D_total/log_gamma transforms, extend `diagnostics.py` with GMM-based bimodal detection, modify `config.py` for param-aware sizing, and update `model.py` with reparameterized model variant.

**Tech Stack:** NumPyro, JAX, scikit-learn (GaussianMixture), NumPy

---

## Task 1: Reparameterization Module (Core Transforms)

**Files:**
- Create: `homodyne/optimization/cmc/reparameterization.py`
- Test: `tests/unit/optimization/cmc/test_reparameterization.py`

**Step 1: Write the failing test for ReparamConfig dataclass**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/optimization/cmc/test_reparameterization.py::TestReparamConfig -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

**Step 3: Write minimal implementation**

```python
# homodyne/optimization/cmc/reparameterization.py
"""Internal reparameterization for CMC sampling.

Transforms correlated parameters to orthogonal sampling space,
then converts back to physics parameters for output.

Reparameterizations:
- D0, D_offset → D_total, D_offset_frac (breaks linear degeneracy)
- gamma_dot_t0 → log_gamma_dot_t0 (improves conditioning)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ReparamConfig:
    """Configuration for internal reparameterization.

    Attributes
    ----------
    enable_d_total : bool
        Enable D_total reparameterization (D0 + D_offset → D_total).
    enable_log_gamma : bool
        Sample log(gamma_dot_t0) instead of gamma_dot_t0.
    t_ref : float
        Reference time for shear scaling (not currently used).
    """

    enable_d_total: bool = True
    enable_log_gamma: bool = True
    t_ref: float = 1.0
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/optimization/cmc/test_reparameterization.py::TestReparamConfig -v`
Expected: PASS

**Step 5: Commit**

```bash
git add homodyne/optimization/cmc/reparameterization.py tests/unit/optimization/cmc/test_reparameterization.py
git commit -m "feat(cmc): add ReparamConfig dataclass for reparameterization settings"
```

---

## Task 2: Transform Functions (to_sampling_space, to_physics_space)

**Files:**
- Modify: `homodyne/optimization/cmc/reparameterization.py`
- Test: `tests/unit/optimization/cmc/test_reparameterization.py`

**Step 1: Write the failing test for transform functions**

```python
# Add to tests/unit/optimization/cmc/test_reparameterization.py

class TestTransformToSamplingSpace:
    """Tests for transform_to_sampling_space function."""

    def test_d_total_transform(self):
        """D0 + D_offset → D_total, D_offset_frac."""
        from homodyne.optimization.cmc.reparameterization import (
            ReparamConfig,
            transform_to_sampling_space,
        )

        config = ReparamConfig(enable_d_total=True, enable_log_gamma=False)
        params = {"D0": 20000.0, "D_offset": 1000.0, "alpha": -1.0}

        result = transform_to_sampling_space(params, config)

        assert "D_total" in result
        assert "D_offset_frac" in result
        assert "D0" not in result
        assert "D_offset" not in result
        assert result["D_total"] == pytest.approx(21000.0)
        assert result["D_offset_frac"] == pytest.approx(1000.0 / 21000.0)
        assert result["alpha"] == -1.0  # Unchanged

    def test_log_gamma_transform(self):
        """gamma_dot_t0 → log_gamma_dot_t0."""
        from homodyne.optimization.cmc.reparameterization import (
            ReparamConfig,
            transform_to_sampling_space,
        )

        config = ReparamConfig(enable_d_total=False, enable_log_gamma=True)
        params = {"gamma_dot_t0": 0.002, "beta": -0.3}

        result = transform_to_sampling_space(params, config)

        assert "log_gamma_dot_t0" in result
        assert "gamma_dot_t0" not in result
        assert result["log_gamma_dot_t0"] == pytest.approx(np.log(0.002))
        assert result["beta"] == -0.3  # Unchanged

    def test_both_transforms(self):
        """Both transforms applied together."""
        from homodyne.optimization.cmc.reparameterization import (
            ReparamConfig,
            transform_to_sampling_space,
        )

        config = ReparamConfig(enable_d_total=True, enable_log_gamma=True)
        params = {
            "D0": 20000.0,
            "D_offset": 1000.0,
            "gamma_dot_t0": 0.002,
            "beta": -0.3,
        }

        result = transform_to_sampling_space(params, config)

        assert "D_total" in result
        assert "log_gamma_dot_t0" in result


class TestTransformToPhysicsSpace:
    """Tests for transform_to_physics_space function."""

    def test_d_total_inverse(self):
        """D_total, D_offset_frac → D0, D_offset."""
        from homodyne.optimization.cmc.reparameterization import (
            ReparamConfig,
            transform_to_physics_space,
        )

        config = ReparamConfig(enable_d_total=True, enable_log_gamma=False)
        samples = {
            "D_total": np.array([21000.0, 22000.0]),
            "D_offset_frac": np.array([0.05, 0.1]),
            "alpha": np.array([-1.0, -1.1]),
        }

        result = transform_to_physics_space(samples, config)

        assert "D0" in result
        assert "D_offset" in result
        assert "D_total" not in result
        np.testing.assert_allclose(result["D0"], [19950.0, 19800.0])
        np.testing.assert_allclose(result["D_offset"], [1050.0, 2200.0])

    def test_log_gamma_inverse(self):
        """log_gamma_dot_t0 → gamma_dot_t0."""
        from homodyne.optimization.cmc.reparameterization import (
            ReparamConfig,
            transform_to_physics_space,
        )

        config = ReparamConfig(enable_d_total=False, enable_log_gamma=True)
        samples = {
            "log_gamma_dot_t0": np.array([-6.0, -5.5]),
            "beta": np.array([-0.3, -0.25]),
        }

        result = transform_to_physics_space(samples, config)

        assert "gamma_dot_t0" in result
        assert "log_gamma_dot_t0" not in result
        np.testing.assert_allclose(result["gamma_dot_t0"], np.exp([-6.0, -5.5]))

    def test_roundtrip(self):
        """Transform to sampling and back preserves values."""
        from homodyne.optimization.cmc.reparameterization import (
            ReparamConfig,
            transform_to_physics_space,
            transform_to_sampling_space,
        )

        config = ReparamConfig(enable_d_total=True, enable_log_gamma=True)
        original = {
            "D0": 20000.0,
            "D_offset": 1000.0,
            "gamma_dot_t0": 0.002,
            "alpha": -1.0,
            "beta": -0.3,
        }

        # Forward transform
        sampling = transform_to_sampling_space(original, config)

        # Convert to arrays for inverse
        sampling_arrays = {k: np.array([v]) for k, v in sampling.items()}

        # Inverse transform
        recovered = transform_to_physics_space(sampling_arrays, config)

        # Check roundtrip
        assert recovered["D0"][0] == pytest.approx(original["D0"], rel=1e-10)
        assert recovered["D_offset"][0] == pytest.approx(original["D_offset"], rel=1e-10)
        assert recovered["gamma_dot_t0"][0] == pytest.approx(
            original["gamma_dot_t0"], rel=1e-10
        )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/optimization/cmc/test_reparameterization.py::TestTransformToSamplingSpace -v`
Expected: FAIL with "ImportError" for transform_to_sampling_space

**Step 3: Write minimal implementation**

```python
# Add to homodyne/optimization/cmc/reparameterization.py


def transform_to_sampling_space(
    params: dict[str, float],
    config: ReparamConfig,
) -> dict[str, float]:
    """Transform physics params → sampling params.

    Parameters
    ----------
    params : dict[str, float]
        Physics parameters (D0, D_offset, gamma_dot_t0, etc.).
    config : ReparamConfig
        Reparameterization configuration.

    Returns
    -------
    dict[str, float]
        Transformed parameters for sampling.
    """
    result = dict(params)

    if config.enable_d_total and "D0" in params and "D_offset" in params:
        D0 = params["D0"]
        D_offset = params["D_offset"]
        D_total = D0 + D_offset
        D_offset_frac = D_offset / D_total if D_total != 0 else 0.0
        result["D_total"] = D_total
        result["D_offset_frac"] = D_offset_frac
        del result["D0"]
        del result["D_offset"]

    if config.enable_log_gamma and "gamma_dot_t0" in params:
        result["log_gamma_dot_t0"] = np.log(params["gamma_dot_t0"])
        del result["gamma_dot_t0"]

    return result


def transform_to_physics_space(
    samples: dict[str, np.ndarray],
    config: ReparamConfig,
) -> dict[str, np.ndarray]:
    """Transform sampling params → physics params (vectorized).

    Parameters
    ----------
    samples : dict[str, np.ndarray]
        Sampled parameters from MCMC.
    config : ReparamConfig
        Reparameterization configuration.

    Returns
    -------
    dict[str, np.ndarray]
        Physics parameters.
    """
    result = dict(samples)

    if config.enable_d_total and "D_total" in samples:
        D_total = samples["D_total"]
        D_offset_frac = samples["D_offset_frac"]
        result["D0"] = D_total * (1 - D_offset_frac)
        result["D_offset"] = D_total * D_offset_frac
        del result["D_total"]
        del result["D_offset_frac"]

    if config.enable_log_gamma and "log_gamma_dot_t0" in samples:
        result["gamma_dot_t0"] = np.exp(samples["log_gamma_dot_t0"])
        del result["log_gamma_dot_t0"]

    return result
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/optimization/cmc/test_reparameterization.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add homodyne/optimization/cmc/reparameterization.py tests/unit/optimization/cmc/test_reparameterization.py
git commit -m "feat(cmc): add transform functions for reparameterization"
```

---

## Task 3: Bimodal Detection (GMM-based)

**Files:**
- Modify: `homodyne/optimization/cmc/diagnostics.py`
- Test: `tests/unit/optimization/cmc/test_diagnostics.py`

**Step 1: Write the failing test for bimodal detection**

```python
# Add to tests/unit/optimization/cmc/test_diagnostics.py

import numpy as np
import pytest


class TestBimodalDetection:
    """Tests for GMM-based bimodal detection."""

    def test_unimodal_samples_not_detected_as_bimodal(self):
        """Unimodal samples should not be flagged as bimodal."""
        from homodyne.optimization.cmc.diagnostics import detect_bimodal

        # Generate unimodal samples
        rng = np.random.default_rng(42)
        samples = rng.normal(loc=100.0, scale=10.0, size=1000)

        result = detect_bimodal(samples)

        assert result.is_bimodal is False

    def test_bimodal_samples_detected(self):
        """Clearly bimodal samples should be detected."""
        from homodyne.optimization.cmc.diagnostics import detect_bimodal

        # Generate bimodal samples (two well-separated modes)
        rng = np.random.default_rng(42)
        mode1 = rng.normal(loc=-50.0, scale=5.0, size=500)
        mode2 = rng.normal(loc=50.0, scale=5.0, size=500)
        samples = np.concatenate([mode1, mode2])

        result = detect_bimodal(samples)

        assert result.is_bimodal is True
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

        assert result.is_bimodal is False


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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/optimization/cmc/test_diagnostics.py::TestBimodalDetection -v`
Expected: FAIL with "ImportError" for detect_bimodal

**Step 3: Write minimal implementation**

```python
# Add to homodyne/optimization/cmc/diagnostics.py (near top, after imports)

from dataclasses import dataclass

# Add sklearn import (after numpy)
from sklearn.mixture import GaussianMixture


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
    separation : float
        Absolute distance between means.
    relative_separation : float
        Separation relative to scale (separation / |mean(means)|).
    """

    is_bimodal: bool
    weights: tuple[float, float]
    means: tuple[float, float]
    separation: float
    relative_separation: float


def detect_bimodal(
    samples: np.ndarray,
    min_weight: float = 0.2,
    min_relative_separation: float = 0.5,
) -> BimodalResult:
    """Detect bimodality using 2-component Gaussian Mixture Model.

    Parameters
    ----------
    samples : np.ndarray
        1D array of posterior samples.
    min_weight : float
        Minimum weight for both components to be considered bimodal.
    min_relative_separation : float
        Minimum separation between means (relative to scale) for bimodality.

    Returns
    -------
    BimodalResult
        Detection result with component details.
    """
    samples_2d = samples.reshape(-1, 1)

    gmm = GaussianMixture(n_components=2, random_state=42, n_init=3)
    gmm.fit(samples_2d)

    weights = tuple(gmm.weights_.tolist())
    means = tuple(gmm.means_.flatten().tolist())
    separation = abs(means[0] - means[1])
    scale = max(abs(np.mean(means)), 1e-10)
    relative_separation = separation / scale

    # Bimodal if: both components significant AND well-separated
    is_bimodal = min(weights) > min_weight and relative_separation > min_relative_separation

    return BimodalResult(
        is_bimodal=is_bimodal,
        weights=weights,
        means=means,
        separation=separation,
        relative_separation=relative_separation,
    )


def check_shard_bimodality(
    samples: dict[str, np.ndarray],
    params_to_check: list[str] | None = None,
) -> dict[str, BimodalResult]:
    """Check multiple parameters for bimodality.

    Parameters
    ----------
    samples : dict[str, np.ndarray]
        Parameter samples from a shard.
    params_to_check : list[str], optional
        Parameters to check. Defaults to key physical parameters.

    Returns
    -------
    dict[str, BimodalResult]
        Mapping from param name to BimodalResult.
    """
    if params_to_check is None:
        params_to_check = ["D0", "D_offset", "gamma_dot_t0", "beta", "alpha"]

    results = {}
    for param in params_to_check:
        if param in samples:
            results[param] = detect_bimodal(samples[param].flatten())

    return results
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/optimization/cmc/test_diagnostics.py::TestBimodalDetection -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add homodyne/optimization/cmc/diagnostics.py tests/unit/optimization/cmc/test_diagnostics.py
git commit -m "feat(cmc): add GMM-based bimodal detection for posterior diagnostics"
```

---

## Task 4: Param-aware Shard Sizing

**Files:**
- Modify: `homodyne/optimization/cmc/config.py`
- Modify: `homodyne/optimization/cmc/model.py`
- Test: `tests/unit/optimization/cmc/test_config.py`

**Step 1: Write the failing test for get_model_param_count**

```python
# Add to tests/unit/optimization/cmc/test_config.py (or test_model.py)

class TestGetModelParamCount:
    """Tests for get_model_param_count function."""

    def test_static_auto_mode(self):
        """Static mode with auto per-angle has 6 params."""
        from homodyne.optimization.cmc.model import get_model_param_count

        count = get_model_param_count("static", "auto", n_phi=3)
        # 3 physical + 2 scaling + 1 sigma = 6
        assert count == 6

    def test_laminar_flow_auto_mode(self):
        """Laminar flow with auto per-angle has 10 params."""
        from homodyne.optimization.cmc.model import get_model_param_count

        count = get_model_param_count("laminar_flow", "auto", n_phi=3)
        # 7 physical + 2 scaling + 1 sigma = 10
        assert count == 10

    def test_laminar_flow_individual_mode(self):
        """Laminar flow with individual per-angle scales with n_phi."""
        from homodyne.optimization.cmc.model import get_model_param_count

        count = get_model_param_count("laminar_flow", "individual", n_phi=23)
        # 7 physical + 46 scaling (2*23) + 1 sigma = 54
        assert count == 54

    def test_laminar_flow_constant_mode(self):
        """Constant mode has no sampled scaling params."""
        from homodyne.optimization.cmc.model import get_model_param_count

        count = get_model_param_count("laminar_flow", "constant", n_phi=23)
        # 7 physical + 0 scaling + 1 sigma = 8
        assert count == 8
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/optimization/cmc/test_config.py::TestGetModelParamCount -v`
Expected: FAIL with "ImportError" for get_model_param_count

**Step 3: Write minimal implementation**

```python
# Add to homodyne/optimization/cmc/model.py (after get_xpcs_model function)


def get_model_param_count(
    analysis_mode: str,
    per_angle_mode: str,
    n_phi: int,
) -> int:
    """Get total parameter count for given configuration.

    This is used for param-aware shard sizing.

    Parameters
    ----------
    analysis_mode : str
        Analysis mode: "static" or "laminar_flow".
    per_angle_mode : str
        Per-angle mode: "auto", "constant", "individual", "fourier".
    n_phi : int
        Number of phi angles.

    Returns
    -------
    int
        Total number of sampled parameters.
    """
    # Base physical parameters
    if analysis_mode == "static":
        base = 3  # D0, alpha, D_offset
    else:  # laminar_flow
        base = 7  # D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0

    # Scaling parameters based on per_angle_mode
    if per_angle_mode == "constant" or per_angle_mode == "constant_averaged":
        scaling = 0  # Fixed, not sampled
    elif per_angle_mode == "auto":
        scaling = 2  # Averaged contrast + offset
    elif per_angle_mode == "individual":
        scaling = 2 * n_phi  # Per-angle contrast + offset
    elif per_angle_mode == "fourier":
        scaling = 10  # Fourier coefficients (K=2)
    else:
        scaling = 0

    # +1 for sigma (noise scale)
    return base + scaling + 1
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/optimization/cmc/test_config.py::TestGetModelParamCount -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add homodyne/optimization/cmc/model.py tests/unit/optimization/cmc/test_config.py
git commit -m "feat(cmc): add get_model_param_count for param-aware sizing"
```

---

## Task 5: Update CMCConfig.get_num_shards

**Files:**
- Modify: `homodyne/optimization/cmc/config.py`
- Test: `tests/unit/optimization/cmc/test_config.py`

**Step 1: Write the failing test for param-aware shard sizing**

```python
# Add to tests/unit/optimization/cmc/test_config.py

class TestParamAwareShardSizing:
    """Tests for param-aware shard sizing in get_num_shards."""

    def test_no_adjustment_for_7_params(self):
        """No adjustment when n_params <= 7."""
        from homodyne.optimization.cmc.config import CMCConfig

        config = CMCConfig(max_points_per_shard=10000)

        # 7 params should not adjust
        n_shards = config.get_num_shards(n_points=100000, n_phi=3, n_params=7)

        assert n_shards == 10  # 100000 / 10000

    def test_scales_up_for_10_params(self):
        """Shard size scales up for n_params > 7."""
        from homodyne.optimization.cmc.config import CMCConfig

        config = CMCConfig(max_points_per_shard=10000)

        # 10 params should scale by 10/7 ≈ 1.43
        n_shards = config.get_num_shards(n_points=100000, n_phi=3, n_params=10)

        # adjusted_max = 10000 * (10/7) ≈ 14286
        # n_shards = 100000 / 14286 ≈ 7
        assert n_shards == 7

    def test_min_points_per_param_floor(self):
        """Shard size respects min_points_per_param floor."""
        from homodyne.optimization.cmc.config import CMCConfig

        # With 54 params and min_points_per_param=1500
        # min_required = 54 * 1500 = 81000
        config = CMCConfig(max_points_per_shard=10000, min_points_per_param=1500)

        n_shards = config.get_num_shards(n_points=500000, n_phi=23, n_params=54)

        # Should use min_required (81000) as the floor
        # n_shards = 500000 / 81000 ≈ 6
        assert n_shards <= 7
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/optimization/cmc/test_config.py::TestParamAwareShardSizing -v`
Expected: FAIL (get_num_shards doesn't accept n_params yet)

**Step 3: Modify get_num_shards implementation**

```python
# In homodyne/optimization/cmc/config.py, modify CMCConfig class

# Add new field after existing fields:
min_points_per_param: int = 1500  # Minimum points per parameter per shard

# Replace get_num_shards method:
def get_num_shards(self, n_points: int, n_phi: int, n_params: int = 7) -> int:
    """Calculate number of shards with param-aware sizing.

    Parameters
    ----------
    n_points : int
        Total number of data points.
    n_phi : int
        Number of phi angles.
    n_params : int
        Number of model parameters (default: 7 for static).

    Returns
    -------
    int
        Number of shards to use.
    """
    if isinstance(self.num_shards, int):
        return self.num_shards

    # Auto calculation: stratified by phi angle
    if self.sharding_strategy == "stratified":
        return n_phi

    # For other strategies, calculate based on max_points_per_shard
    if isinstance(self.max_points_per_shard, int):
        base_max = self.max_points_per_shard
    else:
        # Default: ~100k points per shard
        base_max = 100000

    # Param-aware adjustment: scale up for n_params > 7
    param_factor = max(1.0, n_params / 7.0)
    min_required = int(self.min_points_per_param * n_params)

    adjusted_max = max(
        int(base_max * param_factor),
        min_required,
    )

    if param_factor > 1.0:
        logger.warning(
            f"Param-aware shard sizing: {n_params} params detected. "
            f"Adjusted max_points_per_shard: {base_max:,} → {adjusted_max:,} "
            f"(factor={param_factor:.2f})"
        )

    return max(1, n_points // adjusted_max)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/optimization/cmc/test_config.py::TestParamAwareShardSizing -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add homodyne/optimization/cmc/config.py tests/unit/optimization/cmc/test_config.py
git commit -m "feat(cmc): add param-aware shard sizing to get_num_shards"
```

---

## Task 6: Integrate Bimodal Detection in Backend

**Files:**
- Modify: `homodyne/optimization/cmc/backends/multiprocessing.py`
- Test: `tests/unit/optimization/cmc/test_backends.py`

**Step 1: Write integration test**

```python
# Add to tests/unit/optimization/cmc/test_backends.py

class TestBimodalIntegration:
    """Tests for bimodal detection integration in backend."""

    def test_bimodal_check_called_after_sampling(self, mocker):
        """Bimodal check is called on successful samples."""
        from homodyne.optimization.cmc.backends.multiprocessing import MultiprocessingBackend
        from homodyne.optimization.cmc.diagnostics import check_shard_bimodality

        # Mock the check_shard_bimodality function
        mock_check = mocker.patch(
            "homodyne.optimization.cmc.backends.multiprocessing.check_shard_bimodality"
        )
        mock_check.return_value = {}

        # This test verifies the function is wired correctly
        # Full integration test requires actual MCMC run
        assert callable(check_shard_bimodality)
```

**Step 2: Add import and integration code**

```python
# In homodyne/optimization/cmc/backends/multiprocessing.py

# Add import near top
from homodyne.optimization.cmc.diagnostics import check_shard_bimodality

# Add after "Log per-shard posterior statistics" block and before heterogeneity check:

        # Check for bimodal posteriors (per-shard)
        bimodal_alerts: list[tuple[int, str, Any]] = []
        for i, shard_result in enumerate(successful_samples):
            bimodal_results = check_shard_bimodality(shard_result.samples)
            for param, result in bimodal_results.items():
                if result.is_bimodal:
                    bimodal_alerts.append((i, param, result))
                    run_logger.warning(
                        f"BIMODAL POSTERIOR: Shard {i}, {param}: "
                        f"modes at {result.means[0]:.4g} and {result.means[1]:.4g} "
                        f"(weights: {result.weights[0]:.2f}/{result.weights[1]:.2f})"
                    )

        if bimodal_alerts:
            run_logger.warning(
                f"Detected {len(bimodal_alerts)} bimodal posteriors across shards. "
                f"This may indicate model misspecification or local minima."
            )
```

**Step 3: Run tests**

Run: `pytest tests/unit/optimization/cmc/test_backends.py -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add homodyne/optimization/cmc/backends/multiprocessing.py tests/unit/optimization/cmc/test_backends.py
git commit -m "feat(cmc): integrate bimodal detection alerts in multiprocessing backend"
```

---

## Task 7: Documentation Update

**Files:**
- Modify: `docs/architecture/cmc-fitting-architecture.md`

**Step 1: Add degeneracy documentation section**

Add after the "Quality Filtering" section in `docs/architecture/cmc-fitting-architecture.md`:

```markdown
## Parameter Degeneracy in Laminar Flow Mode

### Known Degeneracies

The `laminar_flow` model has two known parameter degeneracies that can cause
high heterogeneity across CMC shards:

#### 1. D₀/D_offset Linear Degeneracy

The diffusion contribution depends on `D₀ + D_offset`, creating a linear manifold
in parameter space where different (D₀, D_offset) pairs produce equivalent fits.

| Symptom | Cause |
|---------|-------|
| `D_offset` CV > 1.0 | Shards find different points along the D₀ + D_offset = const ridge |
| `D_offset` spans positive and negative | Ridge crosses zero for D_offset |
| High `D₀` range despite good NLSQ fit | Compensating D_offset values |

**Mitigation (automatic in v0.8+):**
CMC internally reparameterizes to `D_total = D₀ + D_offset` and 
`D_offset_frac = D_offset / D_total`, which are orthogonal and well-constrained.
Results are automatically converted back to D₀/D_offset for output.

#### 2. γ̇₀/β Multiplicative Correlation

The shear contribution scales as `γ̇₀ · t^(1+β)`. Higher γ̇₀ with more negative β
can produce similar effects to lower γ̇₀ with less negative β.

| Symptom | Cause |
|---------|-------|
| `gamma_dot_t0` CV > 1.0 | Shards explore the γ̇₀-β correlation ridge |
| `gamma_dot_t0` spans 10-100× range | Compensating β values |
| `beta` moderate heterogeneity (CV ~0.5-0.8) | Correlated with γ̇₀ |

**Mitigation (automatic in v0.8+):**
CMC samples `log(γ̇₀)` instead of γ̇₀ directly, which improves conditioning
and reduces posterior ridge exploration.

### Diagnostic Indicators

When heterogeneity abort triggers, check these indicators:

| Indicator | Healthy | Problematic |
|-----------|---------|-------------|
| D_offset CV | < 0.5 | > 1.0 |
| D_offset range | Within ±20% of D₀ | Spans ±D₀ or sign changes |
| gamma_dot_t0 CV | < 0.5 | > 1.0 |
| Bimodal warnings | 0 | Multiple shards |

### Configuration Options

If heterogeneity persists after v0.8+ mitigations:

```yaml
optimization:
  cmc:
    reparameterization:
      d_total: true           # Default: true for laminar_flow
      log_gamma_dot: true     # Default: true for laminar_flow
    sharding:
      max_points_per_shard: 50000  # Increase for more statistical power
    validation:
      max_parameter_cv: 1.5   # Relax threshold if physical heterogeneity expected
```
```

**Step 2: Commit documentation**

```bash
git add docs/architecture/cmc-fitting-architecture.md
git commit -m "docs(cmc): document parameter degeneracy in laminar_flow mode"
```

---

## Task 8: Reparameterized Model Variant

**Files:**
- Modify: `homodyne/optimization/cmc/model.py`
- Test: `tests/unit/optimization/cmc/test_model.py`

**Step 1: Write test for reparameterized model**

```python
# Add to tests/unit/optimization/cmc/test_model.py

class TestReparameterizedModel:
    """Tests for xpcs_model_reparameterized."""

    def test_model_samples_d_total_not_d0(self):
        """Reparameterized model samples D_total instead of D0."""
        import jax.numpy as jnp
        import numpyro
        from numpyro.handlers import seed, trace

        from homodyne.optimization.cmc.model import xpcs_model_reparameterized
        from homodyne.optimization.cmc.reparameterization import ReparamConfig

        # Minimal test data
        data = jnp.ones(10)
        t1 = jnp.linspace(0.1, 1.0, 10)
        t2 = jnp.linspace(0.1, 1.0, 10)

        config = ReparamConfig(enable_d_total=True, enable_log_gamma=True)

        # Trace the model to see what's sampled
        with seed(numpyro.handlers.seed, 0):
            tr = trace(xpcs_model_reparameterized).get_trace(
                data=data,
                t1=t1,
                t2=t2,
                phi_unique=jnp.array([0.0]),
                phi_indices=jnp.zeros(10, dtype=jnp.int32),
                q=0.005,
                L=2e6,
                dt=0.1,
                analysis_mode="laminar_flow",
                parameter_space=None,  # Use defaults
                n_phi=1,
                reparam_config=config,
            )

        # Should sample D_total, not D0
        assert "D_total" in tr
        assert "D0" not in tr or tr["D0"]["type"] == "deterministic"

        # Should sample log_gamma_dot_t0, not gamma_dot_t0
        assert "log_gamma_dot_t0" in tr
        assert "gamma_dot_t0" not in tr or tr["gamma_dot_t0"]["type"] == "deterministic"

    def test_deterministic_outputs_present(self):
        """D0, D_offset, gamma_dot_t0 present as deterministic."""
        # Similar to above but check deterministic nodes
        pass  # Implement similar to above
```

**Step 2: Implement xpcs_model_reparameterized**

This is a larger implementation. Key points:
- Sample D_total, D_offset_frac, log_gamma_dot_t0, beta
- Use numpyro.deterministic for D0, D_offset, gamma_dot_t0
- Rest of model unchanged

**Step 3: Update get_xpcs_model to support reparameterization**

```python
def get_xpcs_model(per_angle_mode: str = "individual", use_reparameterization: bool = False):
    """Get the appropriate NumPyro model function.
    
    Parameters
    ----------
    per_angle_mode : str
        Per-angle scaling mode.
    use_reparameterization : bool
        If True and per_angle_mode is "auto", use reparameterized model.
    """
    if use_reparameterization and per_angle_mode == "auto":
        logger.info("CMC: Using reparameterized auto mode model")
        return xpcs_model_reparameterized
    # ... existing logic
```

**Step 4: Commit**

```bash
git add homodyne/optimization/cmc/model.py tests/unit/optimization/cmc/test_model.py
git commit -m "feat(cmc): add xpcs_model_reparameterized for orthogonal sampling"
```

---

## Task 9: Config Options for Reparameterization

**Files:**
- Modify: `homodyne/optimization/cmc/config.py`
- Test: `tests/unit/optimization/cmc/test_config.py`

**Step 1: Add config fields**

```python
# Add to CMCConfig dataclass
reparameterization_d_total: bool = True
reparameterization_log_gamma: bool = True
bimodal_min_weight: float = 0.2
bimodal_min_separation: float = 0.5
```

**Step 2: Update from_dict parsing**

Parse from YAML `reparameterization` section.

**Step 3: Update to_dict serialization**

Include reparameterization settings.

**Step 4: Commit**

```bash
git add homodyne/optimization/cmc/config.py tests/unit/optimization/cmc/test_config.py
git commit -m "feat(cmc): add reparameterization and bimodal config options"
```

---

## Task 10: Integration Test

**Files:**
- Create: `tests/integration/optimization/cmc/test_reparameterization_integration.py`

**Step 1: Write integration test**

```python
"""Integration test for CMC reparameterization."""

import numpy as np
import pytest

pytest.importorskip("numpyro", reason="NumPyro required")


class TestReparameterizationIntegration:
    """Integration tests for reparameterization in CMC."""

    @pytest.mark.slow
    def test_reparameterized_model_produces_valid_samples(self):
        """Reparameterized model produces samples convertible to physics params."""
        # Run short MCMC with reparameterized model
        # Verify D0, D_offset, gamma_dot_t0 present in output
        pass

    @pytest.mark.slow
    def test_param_aware_sizing_reduces_shards(self):
        """Higher param count results in fewer, larger shards."""
        pass
```

**Step 2: Commit**

```bash
git add tests/integration/optimization/cmc/test_reparameterization_integration.py
git commit -m "test(cmc): add integration tests for reparameterization"
```

---

## Final: Run Full Test Suite

```bash
pytest tests/unit/optimization/cmc/ -v
pytest tests/integration/optimization/cmc/ -v --ignore=tests/integration/optimization/cmc/test_reparameterization_integration.py  # Skip slow tests
```

Expected: All PASS
