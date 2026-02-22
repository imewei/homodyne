# Changelog

All notable changes to the Homodyne project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

______________________________________________________________________

## [Unreleased]

*No unreleased changes*

______________________________________________________________________

## [2.22.2] - 2026-02-22

### Performance, Reliability, and Out-of-Core Routing

Performance optimization release with shared-memory shard transport, LPT scheduling, and
persistent JIT caching for CMC; plus critical NLSQ out-of-core routing fix and broad
reliability hardening across both optimization backends.

#### Performance

**CMC shared memory and scheduling:**

- **perf(cmc)**: Place per-shard data arrays in `SharedMemory` via
  `SharedDataManager.create_shared_shard_arrays()`, eliminating per-process serialization
  overhead through the spawn mechanism (`multiprocessing.py`)

- **perf(cmc)**: Pack all shard arrays per key into a single `SharedMemory` segment
  (5 segments total regardless of shard count), preventing `Too many open files` OS limits
  (`multiprocessing.py`)

- **perf(cmc)**: Add noise-weighted LPT (Longest Processing Time first) shard scheduling —
  dispatches largest/noisiest shards first to minimize tail latency (`multiprocessing.py`)

- **perf(cmc)**: Enable persistent JIT compilation cache in workers via
  `jax.config.update()` with `min_compile_time_secs=0` (CMC functions compile in
  0.07–0.15s, below the default 1.0s threshold). First worker compiles; subsequent workers
  load from disk cache (`multiprocessing.py`)

**NLSQ memory:**

- **perf(nlsq)**: Free per-chunk precomputed indices and numpy sigma copy after
  concatenation into device-side arrays; frees ~160+ MB for 10M-point datasets
  (`residual.py`)

- **perf(nlsq)**: Remove dead `_call_jax_chunked` body (replaced with `RuntimeError`
  guard); the vectorized path is used exclusively (`residual.py`)

**Core JAX cleanup:**

- **refactor(core)**: Replace `safe_exp()` with direct `jnp.exp`/`np.exp` — all call sites
  pre-clip to `[-700, 0]`, making the wrapper redundant in JIT hot paths

- **refactor(core)**: Replace `safe_len()` with `.shape[0]` — always concrete in JAX,
  avoids Python dispatch overhead

- **refactor(core)**: Replace phi while-loop reshape with `.reshape(-1)` to handle all
  input shapes uniformly without JIT retracing

- **refactor(core)**: Make `trapezoid_cumsum` unconditional — remove if-branch on array
  size (n==1 naturally produces `[0.0]`), eliminating JIT retracing

#### Fixed

**NLSQ out-of-core routing (critical):**

- **fix(nlsq)**: Base memory routing on effective parameter count instead of expanded
  count — auto-averaged mode (9 params) was being routed to out-of-core (which lacks
  anti-degeneracy) because memory was estimated with 53 expanded params (`wrapper.py`)

- **fix(nlsq)**: Replace scale-sensitive norm-based out-of-core convergence
  (`||step||/||params|| < 1e-4`) with multi-criteria `xtol=1e-6` (per-component max) +
  `ftol=1e-6` (cost change), both required. Prevents false convergence from large-magnitude
  parameters like D₀ ~ 19231 (`wrapper.py`)

**Physics and numerical correctness:**

- **fix(physics)**: Replace `jnp.maximum`/`jnp.clip` with `jnp.where` for floor operations
  on traced parameters (contrast, cos_factor, g1, g2) across NLSQ and NUTS hot paths —
  `jnp.maximum` zeros the gradient below the floor (`physics_nlsq.py`, `fitting.py`,
  `shear_weighting.py`)

- **fix(physics)**: Raise `epsilon_abs` from 1e-20 to 1e-12 in `sqrt(diff² + ε)` smooth-abs
  — 1e-20 is below float32 precision, producing NaN gradients (`wrapper.py`)

- **fix(physics)**: Remove `jnp.clip(g2, 0.5, 2.5)` in per-angle model — kills gradients at
  boundaries; bounds enforced via optimizer constraints (`wrapper.py`)

- **fix(physics)**: Enforce 2D meshgrid in `compute_g2_scaled` public API when
  `get_cached_meshgrid` returns 1D arrays for >2000 elements (`jax_backend.py`)

- **fix(physics)**: Use `dt_value` (resolved config dt) instead of bare `dt` in
  `_compute_g2_scaled_core` call (`jax_backend.py`)

- **fix(core)**: Replace fragile `+1e-15` offset with `jnp.where` for condition number
  calculation in `solve_gram_system` — near-singular systems now correctly route to SVD
  instead of Cholesky (`jax_backend.py`)

**NLSQ optimizer:**

- **fix(nlsq)**: Mask zero-sigma points to 0 residual instead of inflating with `+EPS` —
  prevents invalid/excluded data from dominating the cost function (`residual.py`)

- **fix(nlsq)**: Move chunk structure validation inline into `__init__` (before freeing
  numpy chunks) so construction fails fast on inconsistent angles (`residual.py`)

- **fix(nlsq)**: Derive dt from t1 minimum spacing when `data.dt` is missing instead of
  silently using `None` (`wrapper.py`)

- **fix(nlsq)**: Fix gradient estimate perturbation to target physical params (index 2 in
  auto_averaged mode) instead of always perturbing contrast at `[0]` (`wrapper.py`)

- **fix(nlsq)**: Seed recovery perturbation RNG (`np.random.default_rng(42+attempt)`) for
  reproducible retry sequences (`wrapper.py`)

- **fix(nlsq)**: Track `sigma_is_default` flag in `OptimizationResult` so fit quality
  validator can distinguish "bad fit" from "meaningless chi-squared due to uniform sigma"
  (`wrapper.py`, `results.py`)

**CMC reliability:**

- **fix(cmc)**: Tighten NLSQ-informed prior `width_factor` from 3.0 to 2.0 (~95% coverage)
  for sharper NUTS initialization (`priors.py`, `model.py`)

- **fix(cmc)**: Cap `D_offset_frac` tempering scale at 0.24 and clip to `[0, 0.5]` in
  reparameterization transforms to match prior support bounds (`reparameterization.py`)

- **fix(cmc)**: Fix shard-local angle remapping via `argmin` in `xpcs_model_averaged` and
  `xpcs_model_constant_averaged` to prevent cross-angle bias (`model.py`)

- **fix(cmc)**: Add `_SINGLE_SHARD_HARD_LIMIT = 100K` to prevent NUTS running O(n)
  leapfrog on excessively large single shards; auto-fallback to random sharding
  (`core.py`)

- **fix(cmc)**: Harden multiprocessing backend — pre-generate PRNG keys in batch before
  spawning workers, guard duplicate results from timed-out shards, handle non-finite CV in
  heterogeneity detection (`multiprocessing.py`)

- **fix(cmc)**: Map NumPyro `potential_energy` field to ArviZ `energy` convention and
  sanitize `extra_fields` keys (replace dots with underscores) for xarray compatibility
  (`sampler.py`)

**Configuration:**

- **fix(config)**: Tune NLSQ hybrid streaming defaults — warmup iterations 100→200,
  Gauss-Newton max iterations 50→100, chunk size 50000→10000 (`nlsq/config.py`)

- **fix(config)**: Fix CMC config serialization — `to_dict()` missing
  `min_points_per_param`, wrong key `backend` → `backend_config` (`cmc/config.py`)

- **refactor(config)**: Unify CMC auto-enable threshold to 100K for all modes; remove
  mode-specific threshold logic (`config.py`, templates)

**Shell:**

- **fix(shell)**: Replace `COMPREPLY=($(compgen ...))` with `mapfile -t` pattern to prevent
  word-splitting and globbing on filenames with spaces (SC2207) (`completion.sh`)

#### Changed

**Code quality:**

- **refactor**: Add explicit `jnp.ndarray` type annotations to JIT return values and remove
  `type: ignore` suppressions across `jax_backend.py`, `physics_cmc.py`, `physics_nlsq.py`,
  and CLI modules

- **style**: Sort imports alphabetically and remove unused `log_phase` import in
  `residual.py`

**Documentation:**

- **docs(cmc)**: Document shared memory architecture, LPT scheduling heuristic, and
  persistent JIT compilation cache with JAX 0.8+ `config.update()` requirement
  (`cmc_backends.rst`, `performance_tuning.rst`)

- **docs(physics)**: Correct t1/t2 docstrings from "frame indices" to "physical time in
  seconds" (`physics_nlsq.py`)

**Dependencies:**

- **chore(deps)**: Bump ruff 0.15.1 → 0.15.2
- **chore(deps)**: Update pyproject.toml — ruff ≥0.15.2, sphinx ≥9.1.0, myst-parser ≥5.0.0
- **chore(deps)**: Update pre-commit hooks — ruff-pre-commit v0.15.2, bandit 1.9.3
- **chore(deps)**: Align mypy additional dependencies with project requirements (jax ≥0.8.2,
  numpy ≥2.3, matplotlib ≥3.10)

**Tests:**

- **fix(tests)**: Align documentation tests with restructured docs layout (research/ →
  theory/, user-guide/ → new structure, docs/ → docs/source/)
- **fix(tests)**: Harden factory param keys with `.get()` fallback for per-angle naming
- **fix(tests)**: Stabilize performance benchmarks for CI (relax pool startup threshold,
  measure `.nbytes` instead of RSS)
- **fix(tests)**: Use ArviZ `energy` field name instead of `potential_energy`
- **chore(tests)**: Add `regression` pytest marker and suppress fork/dt warnings

______________________________________________________________________

## [2.22.1] - 2026-02-19

### Numerical Correctness and Gradient Safety

Bug-fix release addressing gradient-killing numerical floors, configuration access errors,
and documentation accuracy. All fixes target the NLSQ Jacobian and NUTS leapfrog hot paths.

#### Fixed

**Gradient-safe numerical floors (P1-C, P1-D, P2-C):**

- **fix(physics)**: Replace `jnp.maximum(x, eps)` with `jnp.where(x > eps, x, eps)` in
  `calculate_diffusion_coefficient`, `calculate_shear_rate`, and `calculate_shear_rate_cmc`
  (`physics_utils.py`). `jnp.maximum` zeros the gradient below the floor, stalling both
  NLSQ Jacobian computation and NUTS leapfrog integration.

- **fix(cmc)**: Replace `jnp.maximum(1 - D_offset_frac, 1e-10)` with `jnp.where` in
  `xpcs_model_reparameterized` D_offset back-transform (`model.py`)

**Configuration and validation (P1-A, P1-B, P1-E, P2-A, P2-B, P2-D):**

- **fix(cmc)**: Fix `require_warmstart` attribute access in `fit_mcmc_jax` — was calling
  `getattr(cmc_config, ...)` on a raw dict (always returned `False`); now reads
  `config.require_nlsq_warmstart` (`core.py`)

- **fix(nlsq)**: Remove hardcoded `dtype=jnp.float64` inside `@jit` shear meshgrid;
  now uses `jnp.result_type(phi)` to preserve caller dtype (`physics_nlsq.py`)

- **fix(cli)**: Read `stator_rotor_gap` from config instead of hardcoded `L=2000000.0`
  in CMC-only CLI path (`commands.py`)

- **fix(init)**: Set `JAX_ENABLE_X64=1` explicitly in `__init__.py` and `cli/main.py`
  before any JAX import, rather than relying on NLSQ import side-effect

- **fix(cmc)**: Change `CMCConfig.from_dict` sharding default from `"stratified"` to
  `"random"` to match the dataclass field default (`config.py`)

- **fix(cmc)**: Add `min_points_per_param` to `CMCConfig.from_dict` — was not
  configurable from YAML despite being a dataclass field (`config.py`)

- **fix(nlsq)**: Align `hybrid_normalization_strategy` default in
  `NLSQConfig.from_dict` to `"auto"` — was `"bounds"`, mismatching the
  dataclass default and `NLSQWrapper` internal default (`nlsq/config.py`)

**Code quality (P2-F):**

- **refactor(cmc)**: Move `_calc_diff` and `_calc_shear` imports from inside deprecated
  `@jit` functions to module level in `physics_cmc.py`

- **fix(multiprocessing)**: Correct misleading comment about JAX_ENABLE_X64 propagation
  in worker spawn (`multiprocessing.py`)

#### Changed

- **docs(api)**: Update `cmc.rst` YAML reference — fix stale `strategy: "stratified"`
  default, add `min_points_per_param` field
- **docs(api)**: Add gradient-safe numerical floors admonition and shear rate functions
  to `core.rst` physics_utils section
- **docs(api)**: Expand XLA configuration documentation in `cmc_backends.rst` to cover
  parent-process `JAX_ENABLE_X64` setup
- **docs(claude)**: Add Critical Rules 7 (gradient-safe floors) and 8 (float64 before
  JAX import) to `CLAUDE.md`

______________________________________________________________________

## [2.22.0] - 2026-02-02

### CMC Adaptive Sampling and Performance Profiling

Performance optimization release adding adaptive NUTS sampling for small datasets and
JAX profiler integration for XLA-level performance analysis.

#### Added

**Adaptive Sampling:**

- **feat(cmc)**: Add adaptive sample count based on shard size

  - Automatically reduces warmup/samples for small datasets
  - Reference: 10K points → full samples; smaller → proportionally fewer
  - 50 points → 140 warmup, 350 samples (75% reduction)
  - 5000 points → 250 warmup, 750 samples (50% reduction)
  - New `CMCConfig.get_adaptive_sample_counts(shard_size, n_params)` method

- **feat(cmc)**: Add `adaptive_sampling` config option (default: `true`)

  - Enable/disable automatic sample count reduction
  - `min_warmup: int = 100` - minimum warmup even for tiny datasets
  - `min_samples: int = 200` - minimum samples for statistical validity

- **feat(cmc)**: Add `max_tree_depth` config option (default: `10`)

  - Limits NUTS leapfrog steps to 2^depth per sample
  - Reduces overhead for posteriors requiring many integration steps
  - Passed directly to NumPyro NUTS kernel

**JAX Profiler Integration:**

- **feat(cmc)**: Add JAX profiler tracing for XLA-level performance analysis
  - py-spy only profiles Python frames; XLA runs native code invisible to it
  - New `enable_jax_profiling: bool = False` config option
  - New `jax_profile_dir: str = "./profiles/jax"` config option
  - Output compatible with TensorBoard and Perfetto

**Configuration Templates:**

- **docs(config)**: Add adaptive sampling section to all YAML templates
  - `homodyne_master_template.yaml`
  - `homodyne_laminar_flow.yaml`
  - `homodyne_static.yaml`

**Performance Benchmarks:**

- **test(perf)**: Add CMC profiling infrastructure
  (`tests/performance/benchmark_cmc.py`)

  - `benchmark_nuts_single_shard()` - profiles NUTS sampling on single shard
  - `benchmark_multiprocessing_overhead()` - measures IPC serialization
  - `benchmark_consensus_aggregation()` - measures posterior aggregation
  - `generate_flamegraph()` - generates py-spy flamegraphs

- **test(perf)**: Add regression test suite (`tests/performance/test_cmc_benchmarks.py`)

  - 18 benchmark tests for multiprocessing, memory, throughput, scaling
  - CI/CD integration tests for performance monitoring

#### Changed

- **refactor(cmc)**: Use adaptive sample counts in `run_nuts_sampling()`

  - Automatically adjusts warmup/samples based on shard size
  - Logs adaptive adjustment: `"(adaptive: 50 pts)"`

- **refactor(cmc)**: Pass `max_tree_depth` to NUTS kernel

  - Limits maximum leapfrog steps per NUTS sample

#### Performance

- **perf(cmc)**: Profiling revealed XLA JIT as dominant bottleneck
  - NUTS sampling: ~2.3 samples/sec, 437ms per sample
  - IPC overhead negligible: 0.07ms/shard serialization
  - Adaptive sampling reduces small-dataset overhead by 60-80%

______________________________________________________________________

## [2.20.0] - 2026-02-01

### CMC Reparameterization and Heterogeneity Prevention

Major release adding reparameterization transforms to break parameter degeneracies and
improve NUTS sampling efficiency in CMC.

#### Added

**Reparameterization Transforms:**

- **feat(cmc)**: Add `xpcs_model_reparameterized` for transformed sampling space

  - Samples `D_total = D0 + D_offset` instead of D0 directly (breaks linear degeneracy)
  - Samples `log(gamma_dot_t0)` instead of gamma_dot_t0 (improves NUTS for small shear
    rates)
  - Transforms back to physical parameters after sampling
  - New `get_xpcs_model()` factory with `use_reparameterization` parameter

- **feat(cmc)**: Add `ReparamConfig` dataclass for reparameterization settings

  - `enable_d_total`: Enable D_total reparameterization (default: True)
  - `enable_log_gamma`: Enable log-gamma reparameterization (default: True)
  - Transform functions: `to_reparameterized()`, `from_reparameterized()`

- **feat(cmc)**: Add reparameterization config options to `CMCConfig`

  - `reparameterization_d_total: bool = True`
  - `reparameterization_log_gamma: bool = True`
  - `bimodal_min_weight: float = 0.2`
  - `bimodal_min_separation: float = 0.5`

**Bimodal Detection:**

- **feat(cmc)**: Add GMM-based bimodal detection for posterior diagnostics

  - Detects bimodal posteriors that may indicate model identifiability issues
  - Configurable via `bimodal_min_weight` and `bimodal_min_separation`
  - Alerts logged when bimodality detected

- **feat(cmc)**: Integrate bimodal detection alerts in multiprocessing backend

  - Per-shard bimodal detection during CMC execution
  - Summary alerts in combined results

**Param-Aware Shard Sizing:**

- **feat(cmc)**: Add param-aware shard sizing to `get_num_shards()`
  - Scales `max_points_per_shard` based on model parameter count
  - `min_points_per_param` ensures adequate data per parameter
  - Prevents data-starved shards for complex models

**CLI Enhancements:**

- **feat(cli)**: Enable automatic NLSQ warm-start for CMC
  - `homodyne --method cmc` now runs NLSQ first automatically
  - Pass `--no-nlsq-warmstart` to disable (not recommended)
  - Reduces divergence rate from ~28% to \<5%

**Configuration Templates:**

- **docs(config)**: Add reparameterization section to all YAML templates
  - `homodyne_master_template.yaml`
  - `homodyne_laminar_flow.yaml`
  - `homodyne_static.yaml`

#### Changed

- **refactor(cmc)**: Optimize sharding logic and enforce NLSQ parity

  - CMC per-angle mode now matches NLSQ behavior exactly
  - `constant_averaged` mode uses fixed averaged scaling

- **refactor(core)**: Modernize type hints and imports

  - Use `collections.abc` for type hints
  - Cleanup imports and type annotations across modules

- **docs(cmc)**: Document parameter degeneracy in laminar_flow mode

  - Comprehensive explanation of D0/D_offset degeneracy
  - Reparameterization rationale and usage

#### Fixed

- **fix(lint)**: Resolve ruff lint errors across codebase
- **fix(types)**: Resolve mypy type errors across codebase
- **fix(tests)**: Resolve deprecation and runtime warnings

#### Testing

- **test(cmc)**: Add tests for `get_model_param_count` verification
- **test(cmc)**: Add reparameterization integration tests
  - MCMC sampling with reparameterized model
  - Transform roundtrip verification

#### Configuration

**New CMC Reparameterization Options:**

```yaml
optimization:
  cmc:
    # Reparameterization (v2.20.0)
    reparameterization:
      enable_d_total: true            # Sample D_total = D0 + D_offset
      enable_log_gamma: true          # Sample log(gamma_dot_t0)
      bimodal_min_weight: 0.2         # GMM component weight threshold
      bimodal_min_separation: 0.5     # Relative separation threshold
```

**When to Use Reparameterization:**

- laminar_flow mode with D0/D_offset degeneracy
- Small shear rates spanning orders of magnitude (1e-6 to 0.5 s⁻¹)
- Posteriors showing linear correlation between D0 and D_offset

**Files Modified:**

- `homodyne/optimization/cmc/model.py` - Reparameterized model
- `homodyne/optimization/cmc/config.py` - Reparameterization config options
- `homodyne/optimization/cmc/reparameterization.py` - Transform functions
- `homodyne/optimization/cmc/backends/multiprocessing.py` - Bimodal alerts
- `homodyne/config/templates/*.yaml` - Reparameterization section

______________________________________________________________________

## [2.19.0] - 2026-01-23

### CMC Convergence and Precision Fixes

Major release addressing catastrophic CMC failures on multi-angle datasets. Previously,
3-angle laminar_flow analysis showed 94% shard timeout, 28.4% divergence rate, and
33-43x uncertainty inflation compared to NLSQ. This release implements comprehensive
fixes across the CMC pipeline.

#### Added

**Angle-Aware Shard Sizing:**

- **feat(cmc)**: Implement angle-aware shard size scaling in
  `_resolve_max_points_per_shard()`
  - `n_phi <= 3`: 30% of base size (prevents timeout on few-angle datasets)
  - `n_phi <= 5`: 50% of base size
  - `n_phi <= 10`: 70% of base size
  - `n_phi > 10`: 100% of base size (full capacity)
  - New `n_phi` parameter propagated through CMC pipeline

**Angle-Balanced Sharding:**

- **feat(cmc)**: Add `shard_data_angle_balanced()` function in `data_prep.py`
  - Ensures proportional angle coverage per shard (default: 80% minimum)
  - Samples proportionally from each angle group
  - Logs angle coverage statistics per shard
  - Configurable via `min_angle_coverage` parameter

**NLSQ Warm-Start Priors:**

- **feat(cmc)**: Add NLSQ-informed prior builders in `priors.py`

  - `build_nlsq_informed_prior()`: TruncatedNormal centered on NLSQ estimate
  - `build_nlsq_informed_priors()`: Build informative priors for all physical parameters
  - `extract_nlsq_values_for_cmc()`: Extract values from various NLSQ result formats
  - Priors use `width_factor=3.0` by default (3σ = NLSQ uncertainty)

- **feat(cmc)**: Add `nlsq_result` parameter to `fit_mcmc_jax()`

  - Automatically builds informative priors from NLSQ results
  - Logs warm-start information when enabled

**Per-Angle Mode Alignment:**

- **feat(cmc)**: Add `xpcs_model_constant_averaged()` for NLSQ parity

  - Uses FIXED averaged contrast/offset (not sampled)
  - Exactly matches NLSQ "auto" behavior
  - 8 parameters (7 physical + sigma) instead of 10

- **feat(cmc)**: Add "constant_averaged" to valid per-angle modes

**Precision Diagnostics:**

- **feat(cmc)**: Add precision analysis functions in `diagnostics.py`
  - `compute_posterior_contraction()`: PCR = 1 - (posterior_std / prior_std)
  - `compute_nlsq_comparison_metrics()`: z-score, uncertainty ratio, coverage
  - `compute_precision_analysis()`: Comprehensive precision analysis
  - `log_precision_analysis()`: Formatted diagnostic report

**Early Abort Mechanism:**

- **feat(cmc)**: Add early termination in multiprocessing backend
  - Tracks failure categories: timeout, heartbeat, crash, numerical, convergence
  - Aborts if >50% of first 10 shards fail
  - Prevents hour-long waits on systematic failures

**NUTS Convergence Improvements:**

- **feat(cmc)**: Elevate `target_accept_prob` to 0.9 for laminar_flow mode
- **feat(cmc)**: Add divergence rate checking with severity levels
  - > 30%: CRITICAL (logged as error)
  - > 10%: WARNING
  - > 5%: ELEVATED (info)

**Shard Quality Filtering:**

- **feat(cmc)**: Add `max_divergence_rate` config option (default: 0.10)

  - Filters shards with >10% divergence rate before consensus combination
  - Prevents corrupted posteriors from biasing combined results
  - Logs filtered shard count and reasons

- **feat(cmc)**: Add `require_nlsq_warmstart` config option (default: false)

  - When true, raises ValueError if laminar_flow mode lacks NLSQ warm-start
  - When false (default), logs warning for laminar_flow without warm-start
  - Helps prevent high divergence rates from poor initialization

#### Changed

- **refactor(cmc)**: Reduce default `per_shard_timeout` from 7200s to 3600s
- **refactor(cmc)**: Tighten sigma prior from `noise_scale * 3.0` to `noise_scale * 1.5`
- **refactor(cmc)**: Update `get_xpcs_model()` to support "constant_averaged" mode

#### Configuration

**New CMC Configuration Options:**

```yaml
optimization:
  cmc:
    sharding:
      max_points_per_shard: "auto"  # Angle-aware scaling (recommended)
      strategy: "angle_balanced"    # Ensure coverage per shard
      min_angle_coverage: 0.8       # 80% of angles per shard minimum
    sampler:
      target_accept_prob: 0.9       # Higher for multi-scale problems
    execution:
      per_shard_timeout: 3600       # 1 hour (reduced from 2 hours)
    per_angle_mode: "constant_averaged"  # Match NLSQ "auto" behavior
    # Quality filtering (v2.19.0)
    max_divergence_rate: 0.10       # Filter shards with >10% divergence rate
    require_nlsq_warmstart: false   # Require NLSQ warm-start for laminar_flow
```

**NLSQ Warm-Start Usage:**

```python
from homodyne.optimization.nlsq import fit_nlsq_jax
from homodyne.optimization.cmc import fit_mcmc_jax

# Step 1: Run NLSQ
nlsq_result = fit_nlsq_jax(data, config)

# Step 2: Run CMC with NLSQ warm-start
cmc_result = fit_mcmc_jax(data, config, nlsq_result=nlsq_result)
```

#### Performance Impact

| Metric | Before | After | Improvement | |--------|--------|-------|-------------| |
Shard success rate | 6% | >90% | 15x | | Divergence rate | 28.4% | \<5% | 5.7x | |
D_offset CV | 1.58 | \<0.5 | 3x | | Uncertainty ratio vs NLSQ | 33-43x | \<5x | 7-9x |

#### Files Modified

- `homodyne/optimization/cmc/core.py` - Angle-aware sizing, NLSQ warm-start
- `homodyne/optimization/cmc/data_prep.py` - Angle-balanced sharding
- `homodyne/optimization/cmc/sampler.py` - NUTS convergence improvements
- `homodyne/optimization/cmc/model.py` - constant_averaged model, sigma prior
- `homodyne/optimization/cmc/priors.py` - NLSQ-informed prior builders
- `homodyne/optimization/cmc/config.py` - Mode validation, timeout changes
- `homodyne/optimization/cmc/diagnostics.py` - Precision analysis functions
- `homodyne/optimization/cmc/backends/multiprocessing.py` - Early termination

______________________________________________________________________

## [2.18.0] - 2026-01-18

### CMC Per-Angle Scaling Modes

Minor release adding per-angle scaling mode support to CMC (Consensus Monte Carlo),
ensuring parity with NLSQ's anti-degeneracy system for consistent dimensionality
reduction.

#### Added

- **feat(cmc)**: Implement per-angle scaling modes for CMC anti-degeneracy

  - `auto` mode (default): Single averaged contrast/offset SAMPLED (10 params)
  - `constant` mode: Per-angle contrast/offset from quantile estimation, FIXED (8
    params)
  - `individual` mode: Per-angle contrast/offset SAMPLED (54 params)
  - Configuration via `per_angle_mode` and `constant_scaling_threshold` fields

- **feat(core)**: Add shared scaling utilities (`homodyne/core/scaling_utils.py`)

  - Quantile-based per-angle contrast/offset estimation
  - Shared between NLSQ and CMC backends

- **feat(cmc)**: Add `xpcs_model_constant()` with fixed_contrast/fixed_offset arrays

  - Factory function `get_xpcs_model()` for mode-based model selection
  - Updated `get_param_names_in_order()` and `build_init_values_dict()` for mode support

#### Changed

- **refactor(nlsq)**: Distinguish auto_averaged from fixed_constant modes

  - Unified per-angle mode semantics across NLSQ and CMC backends
  - `auto` mode estimates per-angle values → AVERAGES → broadcasts, OPTIMIZED
  - `constant` mode uses estimated per-angle values DIRECTLY, NOT optimized

- **docs(cmc)**: Document per-angle mode configuration and architecture

- **docs(config)**: Update templates for v2.18.0 per-angle modes

#### Fixed

- **fix(cmc)**: Resolve linting errors and add security annotations

- **fix**: Add type annotations and remove unnecessary f-strings

- **docs**: Fix 19 Sphinx build warnings

#### Testing

- **test(cmc)**: Add per-angle mode unit tests (24 tests in
  `tests/unit/optimization/cmc/test_per_angle_modes.py`)

______________________________________________________________________

## [2.17.0] - 2026-01-16

### Quantile-Based Per-Angle Scaling

Minor release with improved per-angle parameter initialization for robust NLSQ
optimization.

#### Added

- **feat(nlsq)**: Implement quantile-based per-angle scaling

  - Per-angle contrast/offset computed using robust quantile statistics
  - Improves initialization for laminar_flow mode with many phi angles
  - Reduces sensitivity to outliers in experimental data

- **feat(nlsq)**: Integrate anti-degeneracy constant mode into CMA-ES

  - Constant (global) scaling mode now works with CMA-ES global optimization
  - Automatic mode selection based on phi angle count

#### Fixed

- **fix(imports)**: Resolve broken internal module imports

  - Fixed circular import issues in optimization submodules

- **fix(config)**: Handle per-angle parameter names in bounds lookup

  - Correctly resolves bounds for `contrast_0`, `offset_0`, etc.
  - Prevents KeyError when using per-angle scaling with custom bounds

- **fix(nlsq)**: Resolve constant mode parameter mismatch and diagonal masking

  - Fixed parameter count mismatch when using constant scaling mode
  - Corrected diagonal masking for proper residual computation

#### Removed

- **refactor(nlsq)**: Remove unused anti-degeneracy layer interface
  - Removed `homodyne/optimization/nlsq/anti_degeneracy_layer.py` (507 lines of dead
    code)
  - Removed abstract classes: `AntiDegeneracyLayer`, `AntiDegeneracyChain`,
    `OptimizationState`
  - Removed layer implementations: `FourierReparamLayer`, `HierarchicalLayer`, etc.
  - Production code uses concrete implementations in `wrapper.py` and `core.py` directly
  - Removed associated test files: `test_anti_degeneracy_layer.py`,
    `test_nlsq_anti_degeneracy_layer.py`

______________________________________________________________________

## [2.16.0] - 2026-01-15

### CMA-ES Configuration Enhancements and Visualization Fixes

Minor release with CMA-ES configuration improvements and visualization bug fixes.

#### Added

- **feat(nlsq)**: Add `cmaes_popsize` configuration field to `NLSQConfig`
  - Previously, configured `popsize` value in YAML was ignored (always computed default)
  - New field wires through `CMAESWrapperConfig` to override auto-computed value
  - Validation ensures popsize is positive or None (auto)

#### Fixed

- **fix(viz)**: Use combined data range for adaptive color scaling in C2 heatmaps

  - Block artifacts appeared when fit data had narrower range than experimental data
  - Now computes adaptive color scale from BOTH exp AND fit data
  - Changed `adaptive` default from `False` to `True` in `plot_c2_comparison_fast()`

- **fix(cli)**: Correct analysis mode display in `AnalysisSummaryLogger`

  - Previously showed hardcoded mode instead of actual config value

- **fix(tests)**: Resolve 6 failing unit tests related to CMA-ES JIT tracing

- **fix(nlsq)**: Make model functions JAX-traceable for CMA-ES JIT compilation

#### Changed

- **chore(templates)**: Update CMA-ES defaults for better convergence
  - `max_generations`: 200 → 300
  - `sigma`: 0.3 → 0.5
  - `popsize`: 30 → 40
  - `max_restarts`: 15 → 20
  - Added comments noting multi-start is redundant when CMA-ES is enabled

______________________________________________________________________

## [2.15.0] - 2026-01-10

### CMA-ES Global Optimization Integration

Major feature release adding CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
for robust global optimization of multi-scale parameter spaces.

#### Added

**CMA-ES Global Optimization:**

- **feat(nlsq)**: Add CMA-ES global optimization wrapper (`cmaes_wrapper.py`)

  - Population-based evolution with BIPOP restart strategy
  - Automatic scale detection via `MethodSelector`
  - Memory batching for large datasets (100M+ points)
  - Two-phase architecture: CMA-ES global search → NLSQ TRF refinement

- **feat(nlsq)**: Integrate CMA-ES into config and core optimization flow

  - New config fields: `enable_cmaes`, `cmaes_preset`, `cmaes_max_generations`, etc.
  - Auto-selection when scale ratio > threshold (D₀/γ̇₀ > 1000)
  - Presets: "cmaes-fast" (50 gen), "cmaes" (100), "cmaes-global" (200)

- **feat(nlsq)**: Add bounds-based parameter normalization for CMA-ES

  - Maps parameters to [0,1] space for equal-scale optimization
  - Proper covariance transformation via Jacobian
  - Improves convergence for laminar_flow mode (scale ratio > 1e6)

- **feat(nlsq)**: Add post-optimization fit quality validation

  - Checks: reduced χ² threshold, CMA-ES convergence, parameters at bounds
  - Quality report saved in `convergence_metrics.json`
  - Configurable thresholds via `quality_*` config fields

- **feat(nlsq)**: Integrate global optimization selection into `fit_nlsq_jax()`

  - Unified entry point delegates to CMA-ES or multi-start based on config
  - Availability checks before delegation

**Enhanced Logging System (001-enhance-logging-system):**

- **feat(logging)**: Implement structured logging with rotation
  - `get_logger()`, `with_context()`, `log_phase()`, `log_exception()` APIs
  - `LogConfiguration` for CLI integration
  - Timestamped log files with configurable rotation (10 MB, 5 backups)

**Documentation:**

- **docs(nlsq)**: Document CMA-ES global optimization in CLAUDE.md
- **docs(architecture)**: Add NLSQ and CMC fitting flow documentation
- **docs**: Switch to Furo theme with CSS variables

#### Changed

- **refactor(nlsq)**: Replace scipy.optimize with JAX-native stack

  - Uses jaxopt.LBFGSB for bounded L-BFGS in hierarchical optimization
  - JAX autodiff replaces scipy's approx_derivative

- **refactor(nlsq)**: Enhance CMA-ES logging with phase timing and structured messages

- **refactor(nlsq)**: Update integration for NLSQ 0.6.4 compatibility

  - Removed deprecated WorkflowSelector, WorkflowTier APIs
  - Uses homodyne's own `select_nlsq_strategy()` for memory-based selection

- **chore(config)**: Update templates with CMA-ES settings

- **chore(deps)**: Add evosax and update dependencies for CMA-ES support

- **build(pre-commit)**: Update ruff to v0.14.11

- **build(deps)**: Update dependencies and add furo documentation theme

#### Fixed

- **fix(cli)**: Use `fit_nlsq_jax` as unified entry point for global optimization

- **fix(nlsq)**: Prevent infinite recursion in multistart single fit worker

- **fix(nlsq)**: Add availability checks before global optimization delegation

- **fix(nlsq)**: Make CMA-ES model function JAX-traceable

- **fix(cli)**: Handle constant scaling format in result saving

- **fix(stratification)**: Return tuple from `create_angle_stratified_indices()`

- **fix(memory)**: Increase Jacobian overhead factor from 3.0× to 6.5× for accurate
  large dataset memory estimation

#### Performance

- **perf(jax)**: Improve meshgrid cache and data loader efficiency

#### Configuration

**New CMA-ES Configuration:**

```yaml
optimization:
  nlsq:
    cmaes:
      enable: false                 # Enable CMA-ES global optimization
      preset: "cmaes-global"        # "cmaes-fast" | "cmaes" | "cmaes-global"
      max_generations: 200
      sigma: 0.3
      popsize: 30
      tol_fun: 1.0e-8
      tol_x: 1.0e-8
      restart_strategy: "bipop"
      max_restarts: 15
      auto_select: true             # Auto-select based on scale ratio
      scale_threshold: 1000.0
      refine_with_nlsq: true        # Post-CMA-ES TRF refinement
```

**When to Use CMA-ES:**

- laminar_flow mode with vastly different parameter scales (D₀ ~ 1e4 vs γ̇₀ ~ 1e-3)
- When multi-start local optimization finds inconsistent minima
- Scale ratio > 1000 (auto-detected when `auto_select: true`)

______________________________________________________________________

## [2.14.0] - 2026-01-06

### NLSQ Module Optimization - Architecture Refactoring and Config Consolidation

Major refactoring release implementing the NLSQ Module Optimization specification
(001-nlsq-optimization). Improves code organization, reduces duplication, and
streamlines configuration.

#### Added

**Architecture (Phase 5 - User Story 3)**:

- `homodyne/optimization/nlsq/adapter_base.py` - Abstract base class `NLSQAdapterBase`
  providing shared functionality:

  - `_prepare_data()`: Data flattening and validation
  - `_validate_input()`: Input shape and type validation
  - `_build_result()`: Result object construction
  - `_handle_error()`: Error handling with recovery
  - `_setup_bounds()`: Bounds configuration
  - `_compute_covariance()`: Covariance from Jacobian

- `homodyne/optimization/nlsq/anti_degeneracy_layer.py` - Layer interface for
  anti-degeneracy defense:

  - `OptimizationState` dataclass for state passing between layers
  - `AntiDegeneracyLayer` ABC for layer implementations
  - `FourierReparamLayer`, `HierarchicalLayer`, `AdaptiveRegularizationLayer`
  - `GradientMonitorLayer`, `ShearWeightingLayer`
  - `AntiDegeneracyChain` executor for layer composition

- `homodyne/optimization/nlsq/validation/` - Extracted validation utilities:

  - `InputValidator` class with `validate_all()` method
  - `ResultValidator` class with result validation
  - Standalone functions: `validate_array_dimensions`, `validate_no_nan_inf`,
    `validate_bounds_consistency`

**Configuration (Phase 6 - User Story 4)**:

- `NLSQConfig.from_yaml()` - Single entry point for YAML configuration loading
- `safe_float()`, `safe_int()`, `safe_bool()` utilities in `config.py`
- Deprecation warnings for `config_utils.py` module (v2.14.0)

**Performance (Phase 3 - User Story 1)**:

- JIT-compiled shear weight computation with underflow protection
- LRU meshgrid cache with proper eviction (64-entry limit)
- Static shape annotations for JIT compilation optimization
- CPU thread configuration for HPC environments

**Memory (Phase 4 - User Story 2)**:

- Buffer donation for residual JIT functions
- Vectorized model computation with `compute_g1_batch()` using vmap
- Index-based stratification for memory efficiency
- QR-based J^T J fallback for ill-conditioned matrices
- Explicit rcond for Fourier lstsq calls

**Tests**:

- `tests/unit/test_adapter_base.py` - NLSQAdapterBase unit tests
- `tests/unit/test_anti_degeneracy_layer.py` - Layer interface tests
- `tests/unit/test_chunking.py` - Index-based chunking tests
- `tests/unit/test_residual_jit.py` - Buffer donation tests
- `tests/performance/test_nlsq_memory.py` - Memory usage tests

**Documentation**:

- Updated `docs/api-reference/optimization.rst` with new modules
- Added Quick Links in `docs/index.rst` for new features
- GitHub Actions workflow for documentation CI/CD

#### Changed

- `homodyne/optimization/nlsq/adapter.py` - Now inherits from `NLSQAdapterBase`
- `homodyne/optimization/nlsq/wrapper.py` - Now inherits from `NLSQAdapterBase`
- `homodyne/optimization/nlsq/__init__.py` - Added exports for new modules
- `homodyne/core/jax_backend.py` - LRU cache eviction with `OrderedDict`
- `homodyne/core/models.py` - Added `compute_g1_batch()` vectorized method
- `homodyne/optimization/nlsq/shear_weighting.py` - JIT-compiled
  `_compute_weights_jax()`
- `homodyne/optimization/nlsq/fit_computation.py` - Static shape annotations
- `homodyne/device/cpu.py` - CPU threading configuration

#### Deprecated

- `homodyne.optimization.nlsq.config_utils` module - Use
  `homodyne.optimization.nlsq.config` instead
  - `safe_float()`, `safe_int()`, `safe_bool()` moved to `config.py`
  - Module import now raises `DeprecationWarning`

#### Documentation

- Version updated to 2.14.0 in `docs/conf.py`, `docs/index.rst`, `README.md`
- API reference expanded with new module documentation
- Added usage examples for `OptimizationState`, `AntiDegeneracyChain`, validators

______________________________________________________________________

## [2.13.0] - 2026-01-05

### Deprecation Removal - Unified Memory-Based Strategy Selection

Removed deprecated `DatasetSizeStrategy` class and replaced with unified memory-based
strategy selection API. The new API uses adaptive memory thresholds instead of fixed
dataset size thresholds.

#### Removed

- `homodyne/optimization/nlsq/strategies/selection.py` - Entire module removed
  - `DatasetSizeStrategy` class
  - `OptimizationStrategy` enum (moved to `wrapper.py` as internal)
  - `estimate_memory_requirements()` function

#### Added

New unified strategy selection API in `homodyne.optimization.nlsq.memory`:

```python
from homodyne.optimization.nlsq.memory import (
    NLSQStrategy,           # Enum: STANDARD, OUT_OF_CORE, HYBRID_STREAMING
    StrategyDecision,       # Dataclass with strategy, threshold_gb, peak_memory_gb, reason
    select_nlsq_strategy,   # Unified strategy selection function
)

# Usage
decision = select_nlsq_strategy(n_points=100_000_000, n_params=53)
print(decision.strategy.value)  # 'standard', 'out_of_core', or 'hybrid_streaming'
print(decision.reason)          # Human-readable explanation
```

**Strategy Decision Tree:**

1. If index array (n_points × 8 bytes) > 75% RAM → `HYBRID_STREAMING`
1. Elif peak memory (Jacobian + autodiff) > 75% RAM → `OUT_OF_CORE`
1. Else → `STANDARD`

#### Changed

- `homodyne/optimization/nlsq/memory.py` - Updated docstring, now primary strategy
  selection module
- `homodyne/optimization/nlsq/wrapper.py` - Added local `OptimizationStrategy` enum and
  `_get_strategy_info()` helper
- `homodyne/optimization/nlsq/__init__.py` - Added exports for `NLSQStrategy`,
  `StrategyDecision`, `select_nlsq_strategy`
- `homodyne/runtime/utils/system_validator.py` - Updated to use new memory-based
  strategy selection

#### Tests Updated

- `tests/regression/test_nlsq_regression.py` - Replaced `DatasetSizeStrategy` with
  `select_nlsq_strategy`
- `tests/unit/test_streaming_optimizer.py` - Updated strategy tests
- `tests/integration/test_nlsq_integration.py` - Updated all strategy selection tests
- Removed obsolete `filterwarnings` for `DatasetSizeStrategy` deprecation from:
  - `tests/unit/test_nlsq_core.py`
  - `tests/unit/test_nlsq_wrapper.py`
  - `tests/unit/test_optimization_edge_cases.py`
  - `tests/unit/test_residual_function.py`
  - `tests/performance/test_wrapper_overhead.py`
  - `tests/performance/test_nlsq_performance.py`

______________________________________________________________________

## [2.12.0] - 2026-01-04

### Deprecated Code Cleanup and CMC Defaults Update

Major cleanup release removing deprecated code and updating CMC (Consensus Monte Carlo)
default settings.

#### Removed

- Deprecated `StreamingOptimizer` legacy compatibility code
- Deprecated `should_use_streaming()` function references
- Legacy test files for deprecated streaming functionality
- Unused config path warnings for `performance.subsampling`

#### Changed

- **BREAKING**: CMC default combination method changed from `weighted_gaussian` to
  `consensus_mc`
  - `weighted_gaussian` was mathematically incorrect for posterior combination
  - Add `combination_method: weighted_gaussian` to config to preserve old behavior
    (deprecated)
- Updated config defaults for anti-degeneracy system

#### Deprecated

- CMC combination methods `weighted_gaussian` and `simple_average`
  - Will be removed in v3.0
  - Use `consensus_mc` (now default)

______________________________________________________________________

## [2.11.0] - 2026-01-03

### NLSQAdapter with Model Caching and CurveFit Integration

Added `NLSQAdapter` as the recommended interface for NLSQ optimization, featuring model
caching for significant performance improvements in multi-start optimization.

#### Added

- **NLSQAdapter**: New adapter class using NLSQ's `CurveFit` class
  - Model caching: 3-5× speedup for multi-start optimization (avoids rebuilding models)
  - JIT compilation caching: 2-3× speedup from compiled model reuse
  - Automatic fallback to `NLSQWrapper` on failure
  - Simplified API with `AdapterConfig` dataclass

```python
from homodyne.optimization.nlsq import NLSQAdapter, AdapterConfig

config = AdapterConfig(
    use_jit=True,
    cache_models=True,
    fallback_to_wrapper=True,
)
adapter = NLSQAdapter(config)
result = adapter.fit(data, params)
```

- **Model caching utilities**:
  - `get_or_create_model()` - Get cached model or create new one
  - `clear_model_cache()` - Clear model cache
  - `get_cache_stats()` - Get cache hit/miss statistics

#### Fixed

- **fix(nlsq)**: Handle constant scaling mode in anti-degeneracy layers
- **fix(nlsq)**: Auto-enable hierarchical optimizer when shear weighting is configured

#### Testing

- Updated error handling tests for graceful degradation behavior

______________________________________________________________________

## [2.10.0] - 2026-01-02

### Anti-Degeneracy Defense System - Layer 5

Added **shear-sensitivity weighting** (Layer 5) to complete the 5-layer defense against
structural degeneracy in laminar_flow mode.

#### Layer 5: Shear-Sensitivity Weighting

**Problem**: The shear term gradient `∂L/∂γ̇₀ ∝ Σ cos(φ₀-φ)` suffers from 94.6%
cancellation when summing across 23 angles spanning 360° (11 positive, 12 negative cos
contributions).

**Solution**: Weight residuals by `|cos(φ₀-φ)|` to prevent gradient cancellation:

- Angles perpendicular to flow (cos≈0) contribute less to loss
- Angles parallel to flow (|cos|≈1) contribute more
- Preserves gradient signal for shear parameters

**Configuration:**

```yaml
optimization:
  nlsq:
    anti_degeneracy:
      shear_weighting:
        enable: true              # Enable angle-dependent loss weighting
        min_weight: 0.3           # Minimum weight for perpendicular angles
        alpha: 1.0                # Shear sensitivity exponent
        update_frequency: 1       # Update weights every N outer iterations
        normalize: true           # Normalize weights so mean = 1.0
```

**New File:** `homodyne/optimization/nlsq/shear_weighting.py`

#### Updated 5-Layer Defense System

| Layer | Solution | Effect | |-------|----------|--------| | 1 | Fourier
Reparameterization | Reduces 2×n_phi per-angle params to 10 Fourier coefficients | | 2 |
Hierarchical Optimization | Alternates physical/per-angle stages | | 3 | Adaptive CV
Regularization | Auto-tunes λ to contribute ~10% to loss | | 4 | Gradient Collapse
Monitor | Runtime detection with automatic response | | **5** | **Shear-Sensitivity
Weighting** | **Weights residuals by |cos(φ₀-φ)|** |

### Constant Scaling Mode

Added constant (global) scaling mode as alternative to per-angle scaling for datasets
where per-angle parameters are not needed.

**Features:**

- Auto-selection in anti-degeneracy controller based on phi angle count
- Parameter transformation in wrapper for seamless integration
- Config option: `per_angle_mode: "constant"` or `"auto"`

### Performance Optimizations

- **perf(core)**: Meshgrid cache hit/miss statistics for monitoring
- **perf(core)**: Parallelize Richardson gradient computation
- **perf(data)**: Batch diagonal correction with pre-allocation
- **perf(nlsq)**: Vectorize computation and pre-allocate buffers

### Refactoring

#### 4-Phase Codebase Modernization

- **Phase 1**: Legacy modernization quick wins
- **Phase 2**: Complexity reduction with validator extraction
- **Phase 3**: Wrapper decomposition - extract memory and parameter utilities
- **Phase 4**: Configuration consolidation - improve module exports

#### Streaming Optimizer Migration

- Migrated from legacy `StreamingOptimizer` to `AdaptiveHybridStreamingOptimizer`
- Replaced cyclic stratification with interleaved stratification for better angle
  coverage

#### L-BFGS Terminology Transition

- Renamed all "Adam warmup" references to "L-BFGS warmup" to reflect actual algorithm
- Updated config templates and documentation

### Added

- **feat(nlsq)**: `ParameterIndexMapper` for correct Fourier coefficient indexing
- **feat(nlsq)**: `watch_parameters` option in `GradientCollapseMonitor`
- **feat(nlsq)**: Residual weighting support for anti-degeneracy Layer 5
- **feat(config)**: `shear_weighting` section in laminar flow templates

### Fixed

- **fix(nlsq)**: Fourier coefficient indexing and user bounds propagation
- **fix(nlsq)**: Use `jnp.mean` instead of `np.mean` in hierarchical loss function
- **fix(nlsq)**: JAX-compatible regularization for hybrid streaming
- **fix(nlsq)**: Require consecutive triggers for watched parameter gradient warnings
- **fix(nlsq)**: Use numpy arrays for indices to support JAX array indexing
- **fix(nlsq)**: Define `phi_unique` before shear weighting initialization

### Testing

- **test(nlsq)**: Anti-degeneracy integration tests
- **test(nlsq)**: Phase result and TRF dataclass tests for NLSQ 0.4.x compatibility
- **test(perf)**: Relax thresholds and suppress numpyro warnings

### Documentation

- Simplified anti-degeneracy templates for v2.10.0
- Layer 5 shear-sensitivity weighting documentation
- Updated StreamingOptimizer references to hybrid

______________________________________________________________________

## [2.9.0] - 2025-12-31

### Anti-Degeneracy Defense System

Major feature release addressing **structural degeneracy** in laminar_flow mode with
many phi angles. When using hybrid streaming with n_phi > 6 angles, the optimizer could
incorrectly collapse shear parameters (gamma_dot_t0) to zero because per-angle
parameters absorbed the angle-dependent signal.

#### 4-Layer Defense System

| Layer | Solution | Effect | |-------|----------|--------| | 1 | Fourier
Reparameterization | Reduces 2×n_phi per-angle params to 10 Fourier coefficients | | 2 |
Hierarchical Optimization | Alternates physical/per-angle stages to break gradient
cancellation | | 3 | Adaptive CV Regularization | Auto-tunes λ to contribute ~10% to
loss | | 4 | Gradient Collapse Monitor | Runtime detection with automatic response |

#### New Files

- `homodyne/optimization/nlsq/fourier_reparam.py` - Layer 1: Fourier basis expansion
- `homodyne/optimization/nlsq/hierarchical.py` - Layer 2: Alternating optimization
- `homodyne/optimization/nlsq/adaptive_regularization.py` - Layer 3: CV-based tuning
- `homodyne/optimization/nlsq/gradient_monitor.py` - Layer 4: Collapse detection
- `homodyne/optimization/nlsq/anti_degeneracy_controller.py` - Unified controller
- `docs/specs/anti-degeneracy-defense-v2.9.0.md` - Full specification

#### Configuration

```yaml
optimization:
  nlsq:
    anti_degeneracy:
      per_angle_mode: "auto"     # "independent", "fourier", or "auto"
      fourier_order: 2           # 5 coeffs per param group
      hierarchical:
        enable: true
        max_outer_iterations: 5
      regularization:
        mode: "relative"
        target_cv: 0.10          # 10% variation target
```

#### When to Use

- laminar_flow mode with n_phi > 6 angles
- gamma_dot_t0 collapsing to lower bound
- Per-angle contrast/offset showing high variance (CV > 20%)

### Performance Optimizations

- **perf(nlsq)**: Parallelized multi-start screening for faster global optimization
- **perf(data)**: Selective HDF5 loading with memory-mapped arrays
- **perf(cmc)**: Vectorized per-angle scaling estimation
- **perf(cmc)**: Batch PRNG generation and adaptive polling
- **perf(jax)**: Meshgrid caching for repeated time arrays

### Refactoring

- **refactor(nlsq)**: Extract shared config utilities to `config_utils.py`
- **refactor(core)**: Extract model mixins for gradient and benchmarking

### Testing

- **test(cmc)**: Update multiprocessing tests for psutil detection
- **test(nlsq)**: Add anti-degeneracy defense unit tests

______________________________________________________________________

## [2.7.2] - 2025-12-15

### Group Variance Regularization (NLSQ 0.3.8)

Added regularization to prevent per-angle parameter absorption in laminar_flow mode.

**Problem**: With 23 angles, 46 per-angle params (contrast + offset) vs 7 physical
params. Per-angle params have larger gradients and can "absorb" shear signal.

**Solution**: Add penalty term constraining per-angle parameter variance:

```
L = MSE + λ × [Var(contrast_per_angle) + Var(offset_per_angle)]
```

**Configuration**:

```yaml
optimization:
  nlsq:
    hybrid_streaming:
      enable_group_variance_regularization: true
      group_variance_lambda: 0.01
```

______________________________________________________________________

## [2.7.1] - 2025-12-12

### Per-Angle Initialization Fix

**Fixed shear parameter collapse** in laminar_flow mode with many angles (n_phi > 3).

**Root Cause**: When expanding parameters to per-angle form, code used uniform
initialization (`np.full(n_phi, contrast_single)`), creating physics mismatch with
nonzero gamma_dot_t0. The optimizer found it easier to absorb shear signal into
per-angle params, collapsing gamma_dot_t0 to zero.

**Fix**: New `_compute_consistent_per_angle_init()` computes per-angle contrast/offset
consistent with initial physical parameters via linear regression per angle.

______________________________________________________________________

## [2.7.0] - 2025-12-10

### Adaptive Memory Threshold

Added adaptive memory threshold based on system RAM for NLSQ streaming mode selection.

- Default: 75% of total system RAM as threshold
- Fallback: 16 GB if memory detection fails
- Override via config: `memory_threshold_gb` (explicit) or `memory_fraction` (adaptive)
- Override via env var: `NLSQ_MEMORY_FRACTION` (0.1-0.9)

### Dependency Updates

- NLSQ: 0.3.6 → 0.3.7 → 0.4.2 compatibility
- Added numba/llvmlite for Python 3.13 compatibility
- Relaxed version upper bounds for better compatibility

______________________________________________________________________

## [2.6.0] - 2025-12-05

### Multi-Start NLSQ Optimization

Added multi-start optimization with Latin Hypercube Sampling for global optimization and
parameter degeneracy detection.

#### Dataset Size-Based Strategy Selection

| Dataset Size | Strategy | Approach | Overhead |
|--------------|----------|----------|----------| | < 1M points | **full** | Run N
complete fits in parallel | N× | | 1M - 100M | **subsample** | Multi-start on 500K
subsample, full fit from best | ~1.1× | | > 100M | **phase1** | Parallel L-BFGS warmup,
streaming from best | ~1.2× |

#### Configuration

```yaml
optimization:
  nlsq:
    multi_start:
      enable: false               # User opt-in
      n_starts: 10                # Number of starting points
      seed: 42                    # Random seed for reproducibility
      sampling_strategy: latin_hypercube
      use_screening: true         # Pre-filter poor starting points
      screen_keep_fraction: 0.5   # Keep top 50% after screening
```

#### Features

- **Latin Hypercube Sampling**: Better space-filling than random sampling
- **Screening Phase**: Filters poor starting points before expensive optimization
- **Parallel Execution**: Uses ProcessPoolExecutor for multi-core parallelism
- **Basin Clustering**: Identifies unique local minima in parameter space
- **Degeneracy Detection**: Warns when multiple solutions have similar chi-squared
- **Progress Tracking**: Rich progress bar and phase-based logging

#### CLI Integration

```bash
homodyne --config config.yaml --method nlsq --multi-start
```

### Progress Bar and Logging

Added progress bar for NLSQ optimization with real-time parameter updates and
convergence monitoring.

______________________________________________________________________

## [2.5.0] - 2025-11-25

### Adaptive Hybrid Streaming Optimizer

For large datasets (>10M points), NLSQ can use streaming optimization to avoid OOM
errors.

#### Four Phases

1. **Phase 0**: Parameter normalization setup (bounds-based)
1. **Phase 1**: L-BFGS warmup with adaptive switching (fast initial exploration)
1. **Phase 2**: Streaming Gauss-Newton with exact J^T J accumulation
1. **Phase 3**: Denormalization and covariance transform

#### Memory-Based Auto-Selection

- Estimates peak memory: Jacobian (n_points × n_params × 8 bytes) + 3× intermediates
- Switches to streaming if estimated memory > 75% available RAM
- Default threshold: 75% of total system RAM (adaptive)

#### Configuration

```yaml
optimization:
  nlsq:
    memory_fraction: 0.75
    hybrid_streaming:
      enable: true
      normalize: true
      warmup_iterations: 100
      gauss_newton_max_iterations: 50
      chunk_size: 50000
```

#### Performance Characteristics

| Mode | Memory | Convergence | Covariance |
|------|--------|-------------|------------| | Stratified LS | ~30+ GB | Exact (L-M) |
Exact | | Old Streaming | ~2 GB | Slow | Crude | | **Hybrid Streaming** | ~2 GB | Fast
(Hybrid) | Exact |

### Fixed

- NLSQ integration fixes for cumulative trapezoid integration matching CMC physics
- Searchsorted index bounds checking

______________________________________________________________________

## [2.4.3] - 2025-11-20

### Fixed

- **fix(cmc)**: Use cumulative trapezoid integration to match CMC physics exactly
- **fix(config)**: Sync CMC template settings with CMCConfig defaults
- **fix(cmc)**: Exclude z-space parameters from legacy stats extraction

### Added

- **feat(cmc)**: Hierarchical shard combination with memory caps (max 2000 shards)
- **feat(cmc)**: Add configurable timeouts, diagnostics, and runtime estimation

### Documentation

- Expanded optimization and core module API documentation

______________________________________________________________________

## [2.4.0] - 2025-11-15

### Added

#### Automatic XLA_FLAGS Configuration System

**Automatic JAX CPU device optimization** for MCMC and NLSQ workflows with intelligent
hardware detection.

**New CLI Commands:**

- `homodyne-config-xla` - Utility for XLA mode configuration and status checking
  - `--mode {mcmc,mcmc-hpc,nlsq,auto}` - Set XLA device mode
  - `--show` - Display current XLA configuration and JAX devices

**Features:**

- **Auto-detection**: Intelligent device count based on CPU core count
  - ≤7 cores → 2 devices (small workstations)
  - 8-15 cores → 4 devices (medium workstations)
  - 16-35 cores → 6 devices (large workstations)
  - 36+ cores → 8 devices (HPC nodes)
- **Configuration modes**:
  - `mcmc`: 4 devices for parallel MCMC chains
  - `mcmc-hpc`: 8 devices for HPC clusters
  - `nlsq`: 1 device for optimal NLSQ performance
  - `auto`: Automatic detection based on hardware
- **Persistent configuration**: Saves mode preference to `~/.homodyne_xla_mode`
- **Shell integration**: Works with bash, zsh, and fish shells

**Performance Impact:**

- **MCMC (4 chains, 14-core CPU)**: 1.4x speedup with 4 devices
- **MCMC (8 chains, 36-core HPC)**: 1.8x speedup with 8 devices
- **NLSQ optimization**: Optimal performance with 1 device

#### CMC v3.0 with ArviZ-Native Output

Complete rewrite of Consensus Monte Carlo with ArviZ-native output format.

- **refactor(optimization)**: Replace legacy mcmc with CMC v3.0
- **feat(cmc)**: Add CMC v3.0 implementation with ArviZ InferenceData output
- **refactor(mcmc)**: Remove tier system for single-angle sampling
- **refactor(config)**: Add centralized parameter registry

______________________________________________________________________

## [2.3.0] - 2025-11-07

### Breaking Changes

#### 🚨 GPU Support Removed - CPU-Only Architecture

**CRITICAL BREAKING CHANGE**: Homodyne.3.0 removes all GPU acceleration support. This is
a **hard break** with no migration path.

**Decision:**

- **Stay on v2.2.1**: If you need GPU acceleration (last GPU-supporting version)
- **Upgrade to v2.3.0**: For simplified CPU-only workflows on multi-core systems

**Rationale:**

- Simplify maintenance and reduce complexity
- Focus on reliable HPC CPU optimization for 36/128-core nodes
- Eliminate GPU OOM errors and CUDA compatibility issues
- Enable better cross-platform support (Linux/macOS/Windows)

### Removed

#### CLI Flags

- `--force-cpu` - No longer needed (CPU-only by default)
- `--gpu-memory-fraction` - GPU memory management removed

#### Configuration Keys

- `hardware.force_cpu` - Removed from YAML templates
- `hardware.gpu_memory_fraction` - Removed from YAML templates
- `hardware.cuda_device_id` - Removed from YAML templates
- `performance.computation.gpu_acceleration` - Removed from YAML templates
- `performance.device.preferred_device` - Removed from YAML templates
- `plotting.datashader.gpu_acceleration` - Removed from YAML templates

**Note**: Old configs will gracefully ignore these keys (no errors).

#### API Functions (homodyne.device module)

- `configure_system_cuda()` - GPU device configuration
- `detect_system_cuda()` - GPU detection
- `get_gpu_memory_info()` - GPU memory queries
- `is_gpu_active()` - GPU status checks
- `switch_to_cpu()` - GPU fallback logic
- `benchmark_gpu_performance()` - GPU benchmarking
- Plus 3 additional internal GPU functions

**New API**: `configure_optimal_device()` now CPU-only, no parameters needed.

#### Makefile Targets

- `install-jax-gpu` - GPU JAX installation
- `gpu-check` - GPU validation
- `test-gpu` - GPU test suite

#### Examples/Scripts

- `examples/gpu_accelerated_optimization.py` - Deleted
- `examples/gpu_acceleration.py` - Deleted

**New Scripts (relocated under `scripts/`):**

- `scripts/nlsq/cpu_optimization.py` - HPC CPU optimization guide
- `scripts/nlsq/multi_core_batch_processing.py` - Parallel CPU workflows

#### Test Infrastructure

- `tests/gpu/` directory - All GPU tests removed
- GPU test markers from pytest configuration

#### Runtime Modules

- `homodyne/runtime/gpu/` - Entire GPU runtime module deleted
  - `activation.py`
  - `gpu_wrapper.py`
  - `optimizer.py`
  - `activation.sh`

#### Device Modules

- `homodyne/device/gpu.py` - GPU device management (637 lines removed)

#### Documentation Sections

- GPU installation instructions from README.md
- GPU acceleration sections from all YAML templates
- CUDA setup guides from documentation
- GPU troubleshooting sections

### Modified

#### Core Modules

- `homodyne/cli/commands.py` - Removed GPU OOM fallback logic (~60 lines)
- `homodyne/cli/args_parser.py` - Removed GPU CLI flags (~17 lines)
- `homodyne/cli/main.py` - Removed GPU availability check function
- `homodyne/config/manager.py` - GPU config keys now gracefully ignored
- `homodyne/optimization/nlsq_wrapper.py` - Removed GPU chunk size adaptation
- `homodyne/device/__init__.py` - Simplified to CPU-only device configuration

#### Configuration Templates

- `homodyne/config/templates/homodyne_static.yaml` - CPU-only updates
- `homodyne/config/templates/homodyne_laminar_flow.yaml` - CPU-only updates
- `homodyne/config/templates/homodyne_master_template.yaml` - CPU-only updates

**Changes:**

- Removed `gpu_acceleration` and `gpu_memory_fraction` settings
- Removed `preferred_device` options (CPU-only now)
- Updated platform support notes (CPU-only, all platforms)
- Updated installation instructions (removed GPU steps)

#### System Validator

- `homodyne/runtime/utils/system_validator.py` - Reduced from 10 to 9 tests
  - Removed "GPU Setup" test (test #9, 124 lines)
  - Redistributed test weights (Integration: 2% → 5%)
  - Updated health score calculation for 9 tests

### Added

#### CPU Optimization

- Enhanced multi-core CPU support for HPC clusters
- NUMA-aware thread allocation recommendations
- Slurm job script examples for 36/128-core nodes

#### Documentation

- `docs/migration/v2.2-to-v2.3-gpu-removal.md` - Comprehensive migration guide
- `scripts/README.md` - Updated with CPU-focused scripts
- CPU optimization sections in README.md and CLAUDE.md

#### Scripts

- `scripts/nlsq/cpu_optimization.py` (relocated) - HPC CPU setup guide

  - Multi-core thread configuration
  - NUMA awareness
  - Slurm/PBS/LSF job scripts
  - Performance profiling

- `scripts/nlsq/multi_core_batch_processing.py` (relocated) - Parallel CPU workflows

  - ProcessPoolExecutor patterns
  - Automatic worker scaling
  - Memory-efficient batch processing

### Changed

#### Dependencies

- JAX/jaxlib locked to 0.8.0 (CPU-only, exact version)
- Removed all CUDA-related optional dependencies
- Updated platform support: Linux/macOS/Windows (all CPU-only)

#### Performance

- Optimized for 14+ core CPUs (desktop/workstation)
- HPC cluster support: 36/128-core nodes
- Memory-efficient processing for datasets up to 100M+ points
- No GPU OOM errors (CPU memory management more predictable)

#### Platform Support

- ✅ Linux: Full CPU-only support
- ✅ macOS: Full CPU-only support (Apple Silicon + Intel)
- ✅ Windows: Full CPU-only support

### Migration Guide

**From v2.2.x → v2.3.0:**

1. **Remove GPU JAX** (if installed):

   ```bash
   pip uninstall -y jax jaxlib
   ```

1. **Install v2.3.0**:

   ```bash
   pip install homodyne==2.3.0
   ```

1. **Update configs** (optional - graceful degradation):

   - Remove `--force-cpu` and `--gpu-memory-fraction` from CLI commands
   - GPU settings in YAML files are automatically ignored

1. **Verify**:

   ```bash
   python -c "import jax; print(jax.devices())"
   # Expected: [CpuDevice(id=0)]
   ```

**For GPU Users:**

Stay on v2.2.1 (last GPU-supporting version):

```bash
pip install homodyne==2.2.1
```

### Statistics

**Code Removal:**

- 5 test files deleted (tests/gpu/)
- 8 test files modified (GPU markers removed)
- 4 source files deleted (device/gpu.py, runtime/gpu/)
- 2 example files deleted (GPU examples)
- 3 Makefile targets removed
- 9 API functions removed
- 2 CLI flags removed
- 6 configuration keys deprecated

**Code Added:**

- 2 new CPU-focused examples (963 lines)
- 1 migration guide (450 lines)
- Enhanced CPU optimization documentation

**Overall Impact:**

- ~2,000 lines of GPU code removed
- ~1,400 lines of CPU optimization code added
- Net reduction: ~600 lines
- Simplified architecture and maintenance

### References

- Migration Guide: `docs/migration/v2.2-to-v2.3-gpu-removal.md`
- CPU Optimization: `scripts/nlsq/cpu_optimization.py`
- Multi-Core Workflows: `scripts/nlsq/multi_core_batch_processing.py`
- System Validator: Now 9 tests (was 10)

______________________________________________________________________

## [2.2.0] - 2025-11-06

### Added

#### 🎯 Angle-Stratified Chunking (Critical Fix for Large Datasets)

**Problem Solved:** Silent NLSQ optimization failures on datasets >1M points with
per-angle scaling.

**Root Cause:** NLSQ's arbitrary chunking created chunks missing certain phi angles,
resulting in zero gradients for per-angle parameters and silent optimization failures (0
iterations, unchanged parameters).

**Solution:** Automatic data reorganization BEFORE optimization to ensure every chunk
contains all phi angles.

**New Modules:**

- `homodyne/optimization/stratified_chunking.py` - Core stratification engine (530
  lines)

  - `reorganize_data_stratified()` - Angle-stratified data reorganization
  - `sequential_per_angle_optimization()` - Fallback for extreme imbalance
  - `StratificationDiagnostics` - Performance monitoring

- `homodyne/optimization/sequential_angle.py` - Sequential per-angle optimization
  fallback

**Configuration (Auto-activates):**

```yaml
optimization:
  stratification:
    enabled: "auto"  # Auto: True when per_angle_scaling=True AND n_points>=100k
    target_chunk_size: 100000
    max_imbalance_ratio: 5.0  # Use sequential if max/min angle count > 5.0
    force_sequential_fallback: false
    check_memory_safety: true
    use_index_based: false  # Future: zero-copy optimization
    collect_diagnostics: false
    log_diagnostics: false
```

**Performance:**

- Overhead: \<1% (0.15s for 3M points)
- Scaling: O(n^1.01) sub-linear
- Memory: 2x peak (temporary during reorganization)

**Testing:**

- 47/47 tests passing (100%)
- Zero regressions on existing workflows

**References:**

- Release Notes: `docs/releases/v2.2-stratification-release-notes.md`
- Ultra-Think Analysis: `ultra-think-20251106-012247`
- Investigation: `docs/troubleshooting/nlsq-zero-iterations-investigation.md`

#### ⚠️ CMC Per-Angle Compatibility Safeguards

**Added validation** to warn users if non-stratified CMC sharding is used with per-angle
scaling:

- Updated `homodyne/optimization/cmc/coordinator.py` with validation check
- Added comprehensive troubleshooting section in
  `docs/troubleshooting/cmc_troubleshooting.md`
- Added warning in `docs/user-guide/cmc_guide.md` about stratified sharding requirement
- Added test `test_stratified_sharding_per_angle_parameter_compatibility()` to verify
  angle coverage

**Why:** CMC always uses `per_angle_scaling=True`. Random/contiguous sharding may create
shards with incomplete phi angle coverage, causing zero gradients and silent failures.

**Default:** Stratified sharding (safe for per-angle scaling) is already the CMC
default.

### 🎯 Per-Angle Contrast/Offset Feature

Homodyne now implements **per-angle contrast and offset parameters**, allowing each
scattering angle (phi) to have independent scaling parameters. This is the **physically
correct behavior** as different scattering angles can have different optical properties
and detector responses.

______________________________________________________________________

### Breaking Changes

#### Default Behavior Change: per_angle_scaling=True

**CRITICAL:** Both MCMC and NLSQ now default to per-angle scaling mode
(`per_angle_scaling=True`). This is a breaking change in parameter structure and naming.

**Parameter Structure Changes:**

**MCMC Model Parameters:**

```python
# OLD (v2.1.0): Global contrast/offset
samples = {
    "contrast": [0.5],      # Single value for all angles
    "offset": [1.0],        # Single value for all angles
    "D0": [1000.0],
    "alpha": [0.5],
    ...
}

# NEW (Unreleased): Per-angle contrast/offset
samples = {
    "contrast_0": [0.5],    # Contrast for phi angle 0
    "contrast_1": [0.6],    # Contrast for phi angle 1
    "contrast_2": [0.7],    # Contrast for phi angle 2
    "offset_0": [1.0],      # Offset for phi angle 0
    "offset_1": [1.1],      # Offset for phi angle 1
    "offset_2": [1.2],      # Offset for phi angle 2
    "D0": [1000.0],         # Physical params shared across angles
    "alpha": [0.5],
    ...
}
```

**NLSQ Optimization Parameters:**

```python
# OLD (v2.1.0): 5 parameters for static mode
parameters = [contrast, offset, D0, alpha, D_offset]

# NEW (Unreleased): (2*n_phi + 3) parameters for static mode
# For n_phi=3:
parameters = [
    contrast_0, contrast_1, contrast_2,  # Per-angle contrasts
    offset_0, offset_1, offset_2,        # Per-angle offsets
    D0, alpha, D_offset                  # Physical parameters
]
```

**Total Parameter Counts:**

- **Static Mode (OLD)**: 5 params (2 scaling + 3 physical)
- **Static Mode (NEW with n_phi=3)**: 9 params (6 scaling + 3 physical)
- **Laminar Flow (OLD)**: 9 params (2 scaling + 7 physical)
- **Laminar Flow (NEW with n_phi=3)**: 15 params (6 scaling + 7 physical)

**Formula:** `total_params = (2 × n_phi) + n_physical`

#### API Changes

**MCMC (`_create_numpyro_model`):**

```python
# NEW parameter (default=True)
def _create_numpyro_model(
    ...,
    per_angle_scaling: bool = True,  # NEW: Physically correct default
):
    """
    Parameters
    ----------
    per_angle_scaling : bool, default=True
        If True (default), sample contrast and offset as arrays of shape (n_phi,)
        for per-angle scaling. This is the physically correct behavior.

        If False, use legacy behavior with scalar contrast/offset (shared across
        all angles). This mode is provided for backward compatibility testing only.
    """
```

**NLSQ (`fit_nlsq_jax`):**

```python
# NEW parameter (default=True)
def fit_nlsq_jax(
    data: dict,
    config: ConfigManager,
    initial_params: dict | None = None,
    per_angle_scaling: bool = True,  # NEW: Physically correct default
) -> OptimizationResult:
    """
    Parameters
    ----------
    per_angle_scaling : bool, default=True
        If True (default), use per-angle contrast/offset parameters. This is the
        physically correct behavior as each scattering angle can have different
        optical properties and detector responses.
    """
```

#### Migration Guide

**For Existing Code:**

If you have code that expects global `contrast` and `offset` parameters, you have two
options:

**Option 1: Update to Per-Angle (Recommended)**

```python
# Update your code to handle per-angle parameters
result = fit_mcmc_jax(data, config)  # per_angle_scaling=True by default

# Access per-angle parameters
for i in range(n_phi):
    contrast_i = result.samples[f"contrast_{i}"]
    offset_i = result.samples[f"offset_{i}"]
```

**Option 2: Use Legacy Mode (Backward Compatibility)**

```python
# Explicitly request legacy behavior
result = fit_mcmc_jax(data, config, per_angle_scaling=False)

# Access global parameters (old behavior)
contrast = result.samples["contrast"]
offset = result.samples["offset"]
```

**For Tests:**

Update test expectations to check for per-angle parameter names:

```python
# OLD
assert "contrast" in samples
assert "offset" in samples

# NEW (for n_phi=1)
assert "contrast_0" in samples
assert "offset_0" in samples

# NEW (for n_phi=3)
assert "contrast_0" in samples
assert "contrast_1" in samples
assert "contrast_2" in samples
assert "offset_0" in samples
assert "offset_1" in samples
assert "offset_2" in samples
```

#### Rationale

**Physical Correctness:**

- Different scattering angles probe different length scales in the sample
- Detector response varies across the detector surface
- Optical path differences affect signal intensity
- Each angle can have different contrast and baseline levels

**Scientific Validation:**

- Allows fitting data where different angles have genuinely different properties
- Improves fit quality for heterogeneous samples
- Enables detection of angle-dependent artifacts

**Implementation:**

- MCMC: Per-angle parameters sampled independently via NumPyro
- NLSQ: Per-angle parameters optimized via JAX vmap
- Closure pattern used to avoid JAX concretization errors

______________________________________________________________________

### Added

- **Per-angle scaling for MCMC**: Each phi angle has independent contrast/offset
  parameters
- **Per-angle scaling for NLSQ**: Trust-region optimization with per-angle parameters
- **Comprehensive per-angle tests**: 8 new tests covering multiple angles, independence,
  and backward compatibility
  - `tests/unit/test_per_angle_scaling.py`: Full test suite for per-angle functionality
- **JAX concretization fix**: Pre-compute phi_unique before JIT tracing to avoid
  abstract tracer errors
  - Location: `homodyne/optimization/mcmc.py:1347-1360`

### Changed

- **Default behavior**: `per_angle_scaling=True` is now the default for both MCMC and
  NLSQ
- **Parameter naming**: Contrast/offset now named as `contrast_0`, `offset_0`, etc. by
  default
- **Test expectations**: Updated all MCMC unit tests to expect per-angle parameter names

### Fixed

- **JAX concretization error**: Fixed ConcretizationTypeError when calling
  `jnp.unique()` inside JIT-traced MCMC model
- **MCMC model parameter structure**: Properly handles variable number of phi angles

______________________________________________________________________

## [2.1.0] - 2025-10-31

### 🎉 MCMC/CMC Simplification Release

Homodyne.1.0 significantly simplifies the MCMC API by removing manual method selection
and implementing automatic NUTS/CMC selection based on dataset characteristics. This
release introduces **breaking changes** to the MCMC interface that require configuration
updates.

📖 **[Read the Migration Guide](docs/migration/v2.0-to-v2.1.md)** for step-by-step
upgrade instructions.

______________________________________________________________________

### Breaking Changes

#### API Changes: fit_mcmc_jax()

**Removed Parameters:**

- `method` (str) - No longer accepts `"nuts"`, `"cmc"`, or `"auto"` arguments
- `initial_params` (dict) - Renamed to `initial_values`

**Added Parameters:**

- `parameter_space` (ParameterSpace | None) - Config-driven bounds and prior
  distributions
- `initial_values` (dict[str, float] | None) - Renamed from `initial_params`
- `min_samples_for_cmc` (int, default=15) - Parallelism threshold for CMC selection
- `memory_threshold_pct` (float, default=0.30) - Memory threshold for CMC selection
- `dense_mass_matrix` (bool, default=False) - Use dense vs diagonal mass matrix

**Migration Example:**

```python
# OLD (v2.0.0)
result = fit_mcmc_jax(
    method="nuts",
    initial_params={"D0": 1000.0, "alpha": 0.5},
    ...
)

# NEW (v2.1.0)
from homodyne.config.parameter_space import ParameterSpace
from homodyne.config.manager import ConfigManager

config_mgr = ConfigManager("config.yaml")
parameter_space = ParameterSpace.from_config(config_mgr.config)
initial_values = config_mgr.get_initial_parameters()

result = fit_mcmc_jax(
    parameter_space=parameter_space,
    initial_values=initial_values,
    ...
)
```

#### CLI Changes

**Removed CLI Flags:**

- `--method nuts` → Use `--method mcmc` (automatic selection)
- `--method cmc` → Use `--method mcmc` (automatic selection)
- `--method auto` → Use `--method mcmc` (now default behavior)

**Supported Methods (v2.1.0):**

- `--method nlsq` - Nonlinear least squares optimization
- `--method mcmc` - MCMC with automatic NUTS/CMC selection

**Migration Example:**

```bash
# OLD
homodyne --config config.yaml --method nuts

# NEW
homodyne --config config.yaml --method mcmc
```

#### Configuration Changes

**Removed from YAML:**

```yaml
# REMOVED in v2.1.0
mcmc:
  initialization:
    run_nlsq_init: true
    use_svi: false
    svi_steps: 1000
    svi_timeout: 300
```

**Added to YAML:**

```yaml
# NEW in v2.1.0
optimization:
  mcmc:
    min_samples_for_cmc: 15        # Parallelism threshold
    memory_threshold_pct: 0.30     # Memory threshold (30%)
    dense_mass_matrix: false       # Diagonal vs full covariance

parameter_space:
  bounds:
    - name: D0
      min: 100.0
      max: 10000.0
  priors:  # NEW: Prior distributions for MCMC
    D0:
      type: TruncatedNormal
      mu: 1000.0
      sigma: 500.0

initial_parameters:
  parameter_names: [D0, alpha, D_offset]
  values: [1234.5, 0.567, 12.34]  # From NLSQ results (manual copy)
```

#### Workflow Changes

**OLD Workflow (v2.0):** Automatic initialization

```bash
homodyne --config config.yaml --method mcmc
```

**NEW Workflow (v2.1.0):** Manual NLSQ → MCMC

```bash
# Step 1: Run NLSQ optimization
homodyne --config config.yaml --method nlsq

# Step 2: Manually copy best-fit results to config.yaml
# Edit initial_parameters.values: [D0_result, alpha_result, D_offset_result]

# Step 3: Run MCMC with initialized parameters
homodyne --config config.yaml --method mcmc
```

**Rationale:**

- **Transparency**: Clear separation between NLSQ and MCMC methods
- **User control**: Explicit parameter transfer ensures understanding
- **Simplification**: Removes complex initialization logic from codebase

______________________________________________________________________

### Added

#### Automatic NUTS/CMC Selection

**Dual-Criteria OR Logic:**

CMC is automatically selected when **EITHER** criterion is met:

1. **Parallelism criterion**: `num_samples >= min_samples_for_cmc` (default: 15)

   - Triggers CMC for CPU parallelization with many independent samples
   - Example: 50 phi angles → ~3x speedup on 14-core CPU

1. **Memory criterion**: `estimated_memory > memory_threshold_pct` (default: 0.30)

   - Triggers CMC for memory management with large datasets
   - Example: 10M+ points → prevent OOM errors

**Decision Logic**: CMC if **(Criterion 1 OR Criterion 2)**, otherwise NUTS

**Configurable Thresholds:**

```yaml
optimization:
  mcmc:
    min_samples_for_cmc: 15        # Adjust parallelism trigger
    memory_threshold_pct: 0.30     # Adjust memory trigger (0-1)
```

**Metadata in Results:**

```python
result = fit_mcmc_jax(...)
print(f"Method used: {result.metadata.get('method_used')}")  # 'NUTS' or 'CMC'
print(f"Selection reason: {result.metadata.get('selection_decision_metadata')}")
```

#### Configuration-Driven Parameter Management

**New Classes:**

- `ParameterSpace` class in `homodyne.config.parameter_space`

  - `ParameterSpace.from_config(config_dict)` - Load from YAML config
  - Stores parameter bounds and prior distributions
  - Supports TruncatedNormal, Uniform, and LogNormal priors

- `ConfigManager.get_initial_parameters()` method

  - Loads initial values from `initial_parameters.values` in YAML
  - Falls back to mid-point of bounds: `(min + max) / 2`
  - Supports CLI overrides

**Usage Example:**

```python
from homodyne.config import ConfigManager
from homodyne.config.parameter_space import ParameterSpace
from homodyne.optimization import fit_mcmc_jax

config_mgr = ConfigManager("config.yaml")
parameter_space = ParameterSpace.from_config(config_mgr.config)
initial_values = config_mgr.get_initial_parameters()

result = fit_mcmc_jax(
    data=data, t1=t1, t2=t2, phi=phi, q=0.01,
    parameter_space=parameter_space,
    initial_values=initial_values
)
```

#### Auto-Retry Mechanism

**MCMC Convergence Failures:**

- Automatic retry with different random seeds (max 3 attempts)
- Helps recover from poor initialization or transient numerical issues
- Configurable retry limit: `max_retries` parameter

**Metadata Tracking:**

```python
result = fit_mcmc_jax(...)
print(f"Number of retries: {result.metadata.get('num_retries')}")
```

#### Enhanced Diagnostics

**MCMCResult Metadata Fields:**

- `method_used` - 'NUTS' or 'CMC'
- `selection_decision_metadata` - Why NUTS/CMC was selected
- `parameter_space_metadata` - Bounds and priors used
- `initial_values_metadata` - Initial parameter values used
- `num_retries` - Number of retry attempts (if any)
- `convergence_diagnostics` - R-hat, ESS, acceptance rate, divergences

______________________________________________________________________

### Changed

#### CMC Selection Thresholds (Performance Optimization)

**Threshold Evolution:**

- `min_samples_for_cmc`: 100 → 20 → **15** (October 28, 2025)

  - More aggressive parallelism for multi-core CPUs
  - 20-sample experiment on 14-core CPU now triggers CMC (~1.4x speedup)

- `memory_threshold_pct`: 0.50 → 0.40 → **0.30** (October 28, 2025)

  - More conservative OOM prevention
  - Triggers CMC earlier for large datasets

**Impact:**

- Better CPU utilization on HPC systems (14-128 cores)
- Safer memory management for large datasets (>1M points)
- Hardware-adaptive decision making

#### Documentation Updates

**Updated API References:**

- `docs/api-reference/optimization.rst` - fit_mcmc_jax() v2.1.0 API
- `docs/api-reference/config.rst` - ParameterSpace and ConfigManager
- `docs/advanced-topics/mcmc-uncertainty.rst` - v2.1.0 workflow changes

**New Documentation:**

- `docs/migration/v2.0-to-v2.1.md` - Comprehensive migration guide
- YAML configuration examples with v2.1.0 structure
- API reference for ParameterSpace class

**Updated Architecture Docs:**

- `docs/architecture/cmc-dual-mode-strategy.md` - Updated thresholds
- `CLAUDE.md` - v2.1.0 changes and workflows

______________________________________________________________________

### Deprecated

**No Deprecation Warnings (Hard Break):**

The following features were removed without deprecation warnings due to acknowledged
breaking change:

- `method` parameter in `fit_mcmc_jax()`
- `initial_params` parameter (use `initial_values`)
- `--method nuts/cmc/auto` CLI flags
- `mcmc.initialization` configuration section

**Rationale:** Simplification release with clear migration path documented.

______________________________________________________________________

### Removed

**Automatic Initialization:**

- No automatic NLSQ/SVI initialization before MCMC
- Removed `run_nlsq_init`, `use_svi`, `svi_steps`, `svi_timeout` config options
- Manual workflow required for NLSQ → MCMC transfer

**Method Selection:**

- Removed manual NUTS/CMC/auto method specification
- All method selection is now automatic based on dual criteria

______________________________________________________________________

### Fixed

**MCMC Numerical Stability:**

- Improved convergence with auto-retry mechanism
- Better handling of poor initialization
- Enhanced error messages with recovery suggestions

**Configuration Validation:**

- Better error messages for missing parameter_space
- Validation of prior distribution types
- Clear warnings when initial_values not provided

______________________________________________________________________

### Performance

**CMC Optimization:**

- 20-sample experiment on 14-core CPU: ~1.4x speedup (CMC vs NUTS)
- 50-sample experiment on 14-core CPU: ~3x speedup (CMC vs NUTS)
- Large datasets (>1M points): ~30% memory threshold prevents OOM

**Auto-Retry Overhead:**

- Typical: 0 retries (no overhead)
- Poor initialization: 1-2 retries (~2-4x initial warmup time)
- Max 3 retries before failure

______________________________________________________________________

### Testing

**New Tests:**

- `tests/mcmc/test_mcmc_simplified.py` - Simplified API tests
- `tests/integration/test_mcmc_simplified_workflow.py` - Workflow tests
- `tests/unit/test_cli_args.py` - CLI argument validation

**Test Coverage:**

- Automatic selection logic (15 test cases)
- Configuration-driven parameter management (8 test cases)
- Auto-retry mechanism (5 test cases)
- Breaking change detection (10 test cases)

______________________________________________________________________

### Migration Resources

**Documentation:**

- [Migration Guide](docs/migration/v2.0-to-v2.1.md) - Step-by-step upgrade instructions
- [API Reference](docs/api-reference/optimization.rst) - Updated fit_mcmc_jax() docs
- [Configuration Guide](docs/api-reference/config.rst) - ParameterSpace and YAML
  structure
- [CLAUDE.md](CLAUDE.md) - Developer guide with v2.1.0 workflows

**Support:**

- GitHub Issues: https://github.com/imewei/homodyne/issues
- Migration Questions: Tag with `migration-v2.1`

______________________________________________________________________

## [2.0.0] - 2025-10-12

### 🎉 Major Release: Optimistix → NLSQ Migration

Homodyne.0 represents a major architectural upgrade, migrating from Optimistix to the
**NLSQ** package for trust-region nonlinear least squares optimization. **Good news**:
The migration is **99% backward compatible** - most existing code works without
modifications!

📖 **[Read the Migration Guide](docs/MIGRATION_OPTIMISTIX_TO_NLSQ.md)** for detailed
upgrade instructions.

______________________________________________________________________

### Added

#### **Core Optimization**

- ✅ **NLSQ Package Integration** - Replaced Optimistix with NLSQ
  (github.com/imewei/NLSQ) for JAX-native trust-region optimization
- ✅ **NLSQWrapper Adapter** - New adapter layer providing seamless integration with
  homodyne's existing API
- ✅ **Automatic Error Recovery** - Intelligent retry system with parameter perturbation
  on convergence failures (enabled by default)
- ✅ **Large Dataset Support** - Automatic selection of memory-efficient algorithms for
  datasets >1M points via `curve_fit_large()`
- ✅ **Enhanced Device Reporting** - `OptimizationResult.device_info` now includes
  detailed GPU/CPU information

#### **Testing & Validation**

- ✅ **Scientific Validation Suite** - 7/7 validation tests passing (ground truth
  recovery, numerical stability, performance benchmarks)
- ✅ **Error Recovery Tests** - Comprehensive tests for auto-retry and diagnostics
  (T022/T022b)
- ✅ **Performance Overhead Benchmarks** - Validated \<5% wrapper overhead per NFR-003
  (T031)
- ✅ **GPU Performance Benchmarks** - US2 acceptance tests for GPU acceleration
  validation
- ✅ **Synthetic Data Factory** - Realistic XPCS data generation for testing
  (`tests/factories/synthetic_data.py`)

#### **Documentation**

- ✅ **Migration Guide** - Comprehensive 300+ line guide covering upgrade path,
  troubleshooting, FAQ
- ✅ **Updated README.md** - Prominent migration notice, NLSQ references throughout
- ✅ **Updated CLAUDE.md** - Developer guidance for NLSQ architecture and GPU status
- ✅ **Performance Documentation** - Benchmarks for wrapper overhead and throughput

#### **New Files**

- `homodyne/optimization/nlsq_wrapper.py` (423 lines) - Core adapter implementation
- `tests/factories/synthetic_data.py` - Ground-truth XPCS data generation
- `tests/unit/test_nlsq_public_api.py` - Backward compatibility validation (T020)
- `tests/unit/test_nlsq_wrapper.py` - Wrapper functionality tests (T014-T016, T022)
- `tests/performance/test_wrapper_overhead.py` - Performance benchmarks (T031)
- `tests/gpu/test_gpu_performance_benchmarks.py` - GPU acceleration tests (US2)
- `tests/integration/test_parameter_recovery.py` - Scientific validation
- `tests/validation/` - Validation test suite directory
- `docs/MIGRATION_OPTIMISTIX_TO_NLSQ.md` - User migration guide

______________________________________________________________________

### Changed

#### **Optimization Engine**

- **BREAKING (internal only)**: Replaced Optimistix with NLSQ package
- **API**: `fit_nlsq_jax()` signature **unchanged** (99% backward compatible)
- **Result Format**: `OptimizationResult` attributes **unchanged** (chi_squared,
  parameters, success, etc.)
- **Error Messages**: Enhanced with actionable diagnostics and recovery suggestions
- **Performance**: 10-30% improvement in optimization speed depending on dataset size

#### **GPU Acceleration**

- **GPU Support**: Now automatic via JAX (no configuration needed)
- **Device Selection**: Transparent GPU detection and usage
- **Fallback**: Graceful CPU fallback on GPU memory exhaustion
- **Status**: Functional via JAX, formal benchmarking deferred to future work

#### **Configuration**

- **Config Files**: All existing YAML configs work without modification
- **Parameter Bounds**: Same format, enhanced validation with physics-based constraints
- **Initial Parameters**: Same format, automatic loading from config preserved

______________________________________________________________________

### Deprecated

- **Optimistix**: No longer used (replaced with NLSQ)
- **VI Optimization**: Variational Inference method removed (MCMC remains fully
  supported)
- **Direct Optimistix Usage**: Internal Optimistix APIs no longer available (use public
  `fit_nlsq_jax()` API)

______________________________________________________________________

### Removed

- ❌ `homodyne/optimization/error_recovery.py` - Stub file removed (error recovery
  integrated into NLSQWrapper)
- ❌ Optimistix dependency from `pyproject.toml`
- ❌ All internal Optimistix references

______________________________________________________________________

### Fixed

- 🐛 **Parameter Validation Bug** - Fixed crash with "Parameter count mismatch: got 9,
  expected 12" (T003 aftermath)
- 🐛 **Import Errors** - Fixed `OPTIMISTIX_AVAILABLE` references in tests (replaced with
  `NLSQ_AVAILABLE`)
- 🐛 **Convergence Issues** - Improved convergence for difficult optimizations via
  auto-retry
- 🐛 **Bounds Clipping** - Fixed parameter bounds violations causing crashes

______________________________________________________________________

### Security

- ✅ All dependencies updated to latest stable versions (October 2025)
- ✅ No known security vulnerabilities in NLSQ or JAX dependencies

______________________________________________________________________

## Migration Impact

### For Users

**Action Required**: ✅ **None for 99% of users!**

If you're using the documented public API (`fit_nlsq_jax`), your code will work without
changes.

```python
# This code works in both v1.x and v2.0+
from homodyne.optimization.nlsq import fit_nlsq_jax

result = fit_nlsq_jax(data, config)
print(f"Chi-squared: {result.chi_squared}")  # Same API!
```

**Exception**: If you were directly importing Optimistix internals (undocumented),
you'll need to update to use the public API.

### For Developers

**Action Required**: ✅ **Minimal changes needed**

- Update imports: Replace `from optimistix import ...` with `from nlsq import ...`
- Update documentation: Replace Optimistix references with NLSQ
- Run tests: Verify backward compatibility with `pytest tests/`

See [MIGRATION_OPTIMISTIX_TO_NLSQ.md](docs/MIGRATION_OPTIMISTIX_TO_NLSQ.md) for detailed
upgrade instructions.

______________________________________________________________________

## Performance Improvements

### Optimization Speed

| Dataset Size | v1.x (Optimistix) | v2.0 (NLSQ) | Improvement |
|--------------|-------------------|-------------|-------------| | Small (\<1K) | ~500ms
| ~500ms | ~0% (similar) | | Medium (1-100K) | ~5s | ~4s | ~20% faster | | Large (>1M) |
~60s | ~45s | ~25% faster |

### Wrapper Overhead (NFR-003)

| Dataset | Throughput | Overhead | Status |
|---------|------------|----------|--------| | Medium (9K pts) | >1,000 pts/s | \<10% |
✅ PASS | | Large (50K pts) | >2,000 pts/s | \<5% | ✅ PASS |

### GPU Acceleration (US2)

- **Auto-detection**: ✅ Working via JAX
- **Speedup**: 2-3x for datasets >100K points
- **Fallback**: Graceful CPU fallback on GPU OOM

______________________________________________________________________

## Scientific Validation

### Ground Truth Recovery (T036)

| Difficulty | D0 Error | Alpha Error | Status |
|------------|----------|-------------|--------| | Easy | 1.88-8.61% | \<5% | ✅
Excellent | | Medium | 2.31-12.34% | \<10% | ✅ Good | | Hard | 3.45-14.23% | \<15% | ✅
Acceptable |

All parameter recovery within XPCS community standards.

### Numerical Stability (T037)

- **5 different initial conditions** → all converge to identical solution
- **Chi-squared consistency**: 0.00% deviation
- **Max parameter deviation**: 3.56%

### Physics Validation (T040)

- **6/6 physics constraints satisfied** (100% pass rate)
- Contrast, offset, D0, alpha, D_offset, reduced χ² all valid

______________________________________________________________________

## Known Issues

### Non-Blocking

1. **Test Convergence Tuning** - Some synthetic data tests need parameter tuning for
   reliable convergence (test infrastructure correct, just needs tuning)
1. **GPU Benchmarking** - Formal performance benchmarks (US2 full-scale 50M+ points)
   deferred to future work

### Resolved

- ✅ ErrorRecoveryManager stub removed (no longer needed)
- ✅ Import errors fixed in test_optimization_nlsq.py
- ✅ T020 public API test implemented with realistic synthetic data
- ✅ T022/T022b error recovery tests implemented with mocking

______________________________________________________________________

## Upgrade Instructions

### Quick Upgrade (Most Users)

```bash
# 1. Upgrade homodyne
pip install --upgrade homodyne>=2.0

# 2. Verify NLSQ installed
python -c "import nlsq; print('✓')"

# 3. Run your existing code (no changes needed!)
python your_analysis_script.py
```

### Detailed Upgrade

See [MIGRATION_OPTIMISTIX_TO_NLSQ.md](docs/MIGRATION_OPTIMISTIX_TO_NLSQ.md) for:

- Step-by-step upgrade instructions
- Troubleshooting common issues
- Performance comparison benchmarks
- FAQ and support resources

______________________________________________________________________

## Contributors

Special thanks to all contributors who made v2.0 possible:

- **Core Team**: Migration architecture, implementation, testing
- **Scientific Validation**: Parameter recovery validation, physics checks
- **Documentation**: Migration guide, user documentation, examples
- **Testing**: Comprehensive test suite, performance benchmarks

______________________________________________________________________

## Links

- **Homepage**: https://github.com/your-org/homodyne
- **Documentation**: [README.md](README.md), [CLAUDE.md](CLAUDE.md)
- **Migration Guide**:
  [MIGRATION_OPTIMISTIX_TO_NLSQ.md](docs/MIGRATION_OPTIMISTIX_TO_NLSQ.md)
- **Issue Tracker**: https://github.com/your-org/homodyne/issues
- **Discussions**: https://github.com/your-org/homodyne/discussions

______________________________________________________________________

## [1.x.x] - Previous Versions

For changelog entries prior to v2.0, please see the git history or GitHub releases page.

______________________________________________________________________

**Note**: This is the first formal CHANGELOG for Homodyne. Previous versions (v1.x) did
not maintain a structured changelog. Going forward, all notable changes will be
documented here following Keep a Changelog conventions.
