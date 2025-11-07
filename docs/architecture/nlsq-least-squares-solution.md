# NLSQ least_squares() Solution for Stratified Chunks

**Date**: 2025-11-06
**Status**: Recommended Implementation
**Confidence**: 95%

---

## Executive Summary

Use NLSQ's `least_squares()` function directly (not `curve_fit_large()`) with a stratified residual function. This gives us:
- ✅ Full NLSQ optimization power (not scipy)
- ✅ JAX acceleration + GPU support
- ✅ Control over chunking (we handle it in residual function)
- ✅ Scales to >100M points
- ✅ No double-chunking problem

---

## The Solution

### Core Insight

NLSQ has a `least_squares()` function (similar to scipy's) that accepts:
- **Residual function**: `fun(params) → residuals`
- **Optional xdata/ydata**: Only for convenience, not required
- **JAX autodiff**: Automatic Jacobian computation

This is what `curve_fit_large()` wraps internally, but `curve_fit_large()` requires model function format.

### Architecture

```python
from nlsq import least_squares

# Step 1: Stratification (already working)
stratified_data = apply_stratification(data, per_angle_scaling=True)
# Result: 31 chunks, each with all 3 angles

# Step 2: Create stratified residual function
class StratifiedResidualFunction:
    def __init__(self, stratified_data, model, per_angle_scaling):
        self.chunks = stratified_data.chunks  # Each has all angles
        self.model = model
        self.per_angle_scaling = per_angle_scaling

        # Pre-compile JAX function
        self.compute_jit = jax.jit(self._compute_chunk_residuals)

    def _compute_chunk_residuals(self, chunk, params):
        """Compute residuals for a single chunk (JIT-compiled)."""
        return self.model.compute_residuals(
            phi=chunk.phi,
            t1=chunk.t1,
            t2=chunk.t2,
            g2=chunk.g2,
            sigma=chunk.sigma,
            params=params,
            q=chunk.q,
            L=chunk.L,
            dt=chunk.dt,
            per_angle_scaling=self.per_angle_scaling
        )

    def __call__(self, params):
        """Compute residuals across all stratified chunks.

        Called by NLSQ at each iteration.
        """
        # Process each chunk (all angles present in each)
        residuals = [
            self.compute_jit(chunk, params)
            for chunk in self.chunks
        ]

        # Return flattened residuals for NLSQ
        return jnp.concatenate(residuals)

# Step 3: Optimize with NLSQ's least_squares
residual_fn = StratifiedResidualFunction(stratified_data, model, per_angle_scaling)

result = least_squares(
    fun=residual_fn,          # Our stratified residual function
    x0=initial_params,
    jac=None,                 # JAX autodiff (NLSQ handles it)
    bounds=(lower, upper),
    method='trf',             # Trust Region Reflective
    ftol=1e-8,
    xtol=1e-8,
    gtol=1e-8,
    max_nfev=1000,
    # NO xdata/ydata needed - data is inside residual_fn!
)

# Result: popt = result['x'], pcov from Jacobian
```

---

## Implementation Details

### File 1: `homodyne/optimization/stratified_residual.py`

```python
"""Stratified residual function for NLSQ least_squares().

This module provides a residual function wrapper that preserves angle-stratified
chunk structure when using NLSQ's least_squares() function directly.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Any
import logging


class StratifiedResidualFunction:
    """Residual function that respects angle-stratified chunk structure.

    This class wraps stratified data chunks and provides a residual function
    interface for NLSQ's least_squares(). Each chunk is guaranteed to contain
    all phi angles, enabling correct gradient computation for per-angle parameters.

    Usage:
        stratified_data = apply_stratification(data, per_angle_scaling=True)
        residual_fn = StratifiedResidualFunction(stratified_data, model, True)
        result = nlsq.least_squares(fun=residual_fn, x0=params, ...)

    Attributes:
        chunks: List of stratified data chunks (each has all angles)
        model: Physics model with compute_residuals method
        per_angle_scaling: Whether per-angle parameters are used
        n_chunks: Number of stratified chunks
        n_total_points: Total data points across all chunks
    """

    def __init__(
        self,
        stratified_data,
        model: Any,
        per_angle_scaling: bool,
        logger: logging.Logger | None = None
    ):
        """Initialize stratified residual function.

        Parameters
        ----------
        stratified_data : StratifiedData
            Angle-stratified data with chunks attribute
        model : PhysicsModel
            Model with compute_residuals method
        per_angle_scaling : bool
            Whether per-angle parameters are enabled
        logger : Logger, optional
            Logger for diagnostics
        """
        self.chunks = stratified_data.chunks
        self.model = model
        self.per_angle_scaling = per_angle_scaling
        self.logger = logger or logging.getLogger(__name__)

        self.n_chunks = len(self.chunks)
        self.n_total_points = sum(len(chunk.g2) for chunk in self.chunks)

        # Pre-compile chunk residual computation
        self._setup_jax_functions()

        self.logger.info(
            f"Initialized StratifiedResidualFunction: "
            f"{self.n_chunks} chunks, {self.n_total_points:,} total points"
        )

    def _setup_jax_functions(self):
        """Pre-compile JAX functions for performance."""
        # JIT compile the chunk residual computation
        self.compute_chunk_jit = jax.jit(self._compute_chunk_residuals_raw)

        self.logger.debug("JAX functions compiled for stratified residuals")

    def _compute_chunk_residuals_raw(
        self,
        phi: jnp.ndarray,
        t1: jnp.ndarray,
        t2: jnp.ndarray,
        g2: jnp.ndarray,
        sigma: jnp.ndarray,
        params: jnp.ndarray,
        q: float,
        L: float,
        dt: float
    ) -> jnp.ndarray:
        """Raw chunk residual computation (JIT-compiled).

        This is the core computation that gets JIT-compiled by JAX.
        """
        return self.model.compute_residuals(
            phi=phi,
            t1=t1,
            t2=t2,
            g2=g2,
            sigma=sigma,
            params=params,
            q=q,
            L=L,
            dt=dt,
            per_angle_scaling=self.per_angle_scaling
        )

    def __call__(self, params: np.ndarray) -> np.ndarray:
        """Compute residuals across all stratified chunks.

        This is the function that NLSQ's least_squares() calls at each iteration.

        Parameters
        ----------
        params : ndarray
            Current parameter values

        Returns
        -------
        residuals : ndarray
            Flattened array of residuals from all chunks
        """
        # Convert to JAX array
        params_jax = jnp.asarray(params)

        # Process each stratified chunk
        all_residuals = []
        for i, chunk in enumerate(self.chunks):
            # Each chunk has all angles (guaranteed by stratification)
            chunk_residuals = self.compute_chunk_jit(
                phi=jnp.asarray(chunk.phi),
                t1=jnp.asarray(chunk.t1),
                t2=jnp.asarray(chunk.t2),
                g2=jnp.asarray(chunk.g2),
                sigma=jnp.asarray(chunk.sigma),
                params=params_jax,
                q=float(chunk.q),
                L=float(chunk.L),
                dt=float(chunk.dt)
            )
            all_residuals.append(chunk_residuals)

        # Concatenate and return as numpy array
        residuals = jnp.concatenate(all_residuals)
        return np.asarray(residuals)

    def validate_chunk_structure(self) -> bool:
        """Validate that all chunks contain all angles.

        Returns
        -------
        valid : bool
            True if all chunks have all angles

        Raises
        ------
        ValueError
            If any chunk is missing angles
        """
        # Get expected angles from first chunk
        expected_angles = set(np.unique(self.chunks[0].phi))
        n_angles = len(expected_angles)

        # Check each chunk
        for i, chunk in enumerate(self.chunks):
            chunk_angles = set(np.unique(chunk.phi))

            if chunk_angles != expected_angles:
                missing = expected_angles - chunk_angles
                raise ValueError(
                    f"Chunk {i} missing angles: {missing}. "
                    f"Has {len(chunk_angles)} angles, expected {n_angles}. "
                    f"Stratification failed!"
                )

        self.logger.debug(
            f"Chunk structure validated: All {self.n_chunks} chunks have all {n_angles} angles"
        )
        return True

    def get_diagnostics(self) -> dict[str, Any]:
        """Get diagnostics about chunk structure.

        Returns
        -------
        diagnostics : dict
            Chunk structure information
        """
        return {
            'n_chunks': self.n_chunks,
            'n_total_points': self.n_total_points,
            'points_per_chunk': [len(chunk.g2) for chunk in self.chunks],
            'angles_per_chunk': [len(np.unique(chunk.phi)) for chunk in self.chunks],
            'per_angle_scaling': self.per_angle_scaling,
        }
```

### File 2: Integration in `homodyne/optimization/nlsq_wrapper.py`

```python
def _fit_with_stratified_least_squares(
    self,
    stratified_data,
    residual_fn_factory,
    initial_params: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    config,
    logger,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit using NLSQ's least_squares() with stratified chunks.

    This method uses NLSQ's least_squares() function directly (not curve_fit_large)
    with a custom residual function that preserves angle-stratified chunk structure.

    Key advantage: We control chunking inside the residual function, so NLSQ never
    re-chunks our data and breaks angle completeness.

    Parameters
    ----------
    stratified_data : StratifiedData
        Angle-stratified data chunks
    residual_fn_factory : callable
        Factory to create StratifiedResidualFunction
    initial_params : ndarray
        Initial parameter guess
    bounds : tuple of ndarray
        Parameter bounds (lower, upper)
    config : ConfigManager
        Configuration with optimization settings
    logger : Logger
        Logger for diagnostics

    Returns
    -------
    popt : ndarray
        Optimized parameters
    pcov : ndarray
        Parameter covariance matrix
    info : dict
        Optimization information
    """
    from nlsq import least_squares

    logger.info("Using NLSQ's least_squares() with stratified chunks...")

    # Create stratified residual function
    residual_fn = residual_fn_factory(stratified_data, self.model, self.per_angle_scaling, logger)

    # Validate chunk structure
    residual_fn.validate_chunk_structure()

    # Get optimization settings
    opt_config = config.config.get('optimization', {}).get('nlsq', {})
    ftol = opt_config.get('ftol', 1e-8)
    xtol = opt_config.get('xtol', 1e-8)
    gtol = opt_config.get('gtol', 1e-8)
    max_nfev = opt_config.get('max_iterations', 1000)

    logger.info(
        f"NLSQ least_squares settings: "
        f"ftol={ftol}, xtol={xtol}, gtol={gtol}, max_nfev={max_nfev}"
    )

    # Run NLSQ optimization
    import time
    start_time = time.perf_counter()

    result = least_squares(
        fun=residual_fn,              # Stratified residual function
        x0=initial_params,
        jac=None,                     # JAX autodiff (NLSQ handles it)
        bounds=bounds,
        method='trf',                 # Trust Region Reflective
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        max_nfev=max_nfev,
        verbose=0,
        # NO xdata/ydata - data is in residual_fn!
    )

    execution_time = time.perf_counter() - start_time

    # Extract results
    popt = result['x']

    # Compute covariance from Jacobian
    if result.get('success', True):
        # Compute Jacobian at optimum
        jac_fn = jax.jacfwd(residual_fn)
        J = jac_fn(popt)

        # Covariance: (J^T J)^-1
        try:
            pcov = np.linalg.inv(J.T @ J)
        except np.linalg.LinAlgError:
            logger.warning("Singular Jacobian, using pseudo-inverse for covariance")
            pcov = np.linalg.pinv(J.T @ J)
    else:
        # Failed optimization, return identity covariance
        pcov = np.eye(len(popt))

    # Build info dict
    info = {
        'success': result.get('success', True),
        'nfev': result.get('nfev', 0),
        'njev': result.get('njev', 0),
        'optimality': result.get('optimality', 0.0),
        'cost': result.get('cost', np.nan),
        'execution_time': execution_time,
        'message': result.get('message', ''),
        'chunk_diagnostics': residual_fn.get_diagnostics(),
    }

    logger.info(
        f"NLSQ least_squares completed: "
        f"success={info['success']}, nfev={info['nfev']}, "
        f"cost={info['cost']:.4e}, time={execution_time:.2f}s"
    )

    return popt, pcov, info


def fit(self, data, config, per_angle_scaling, ...):
    """Main fit method with stratified least_squares support."""

    # ... existing code for data preparation ...

    # Apply stratification if needed
    if per_angle_scaling and n_points > 100_000:
        logger.info("Large dataset with per-angle scaling: using stratification")
        stratified_data = self._apply_stratification_if_needed(
            data, per_angle_scaling, config, logger
        )

        # Use NLSQ's least_squares() with stratified chunks
        from homodyne.optimization.stratified_residual import StratifiedResidualFunction

        popt, pcov, info = self._fit_with_stratified_least_squares(
            stratified_data=stratified_data,
            residual_fn_factory=StratifiedResidualFunction,
            initial_params=validated_params,
            bounds=nlsq_bounds,
            config=config,
            logger=logger,
        )

        # Create result
        result = self._create_fit_result(
            popt=popt,
            pcov=pcov,
            residuals=None,  # Not stored for memory efficiency
            n_data=n_points,
            iterations=info['nfev'],
            execution_time=info['execution_time'],
            convergence_status="converged" if info['success'] else "failed",
            recovery_actions=[],
            streaming_diagnostics=None,
            stratification_diagnostics=stratified_data.stratification_diagnostics,
        )

        return result

    else:
        # Standard path: use curve_fit or curve_fit_large
        # ... existing code ...
```

---

## Key Advantages

### 1. NLSQ's Full Power
- ✅ JAX-accelerated optimization
- ✅ GPU support (if available)
- ✅ Trust-region algorithm (same as curve_fit_large)
- ✅ Scales to >100M points

### 2. Control Over Chunking
- ✅ We handle chunking in residual function
- ✅ NLSQ never sees raw data (only residuals)
- ✅ No double-chunking problem
- ✅ Angle completeness guaranteed

### 3. Performance
- ✅ JIT compilation of chunk computation
- ✅ JAX autodiff for Jacobian (no numerical estimation)
- ✅ Memory efficient (<500 MB for 3M points)
- ✅ GPU acceleration for compute_residuals

### 4. Clean Architecture
- ✅ Separation of concerns (stratification vs optimization)
- ✅ Reusable StratifiedResidualFunction class
- ✅ Easy to test and validate
- ✅ CMC can use same pattern

---

## Memory Analysis

**3M Points Dataset** (3 angles, 1001×1001 time grid):

**Stratification**: 31 chunks × ~100k points/chunk

**Memory per residual_fn call**:
```
Single chunk processing:
  - Input arrays: 100k × 5 = 2 MB (phi, t1, t2, g2, sigma)
  - Residuals: 100k floats = 0.8 MB
  - Accumulated: 3M floats = 24 MB

Peak memory: ~30 MB per call
Total with NLSQ state: <100 MB
```

**JAX Autodiff Memory**:
```
Jacobian computation:
  - NLSQ uses JAX autodiff internally
  - Memory managed by JAX/XLA
  - Efficient reverse-mode differentiation
  - Peak: ~200 MB for 3M × 9 Jacobian
```

**Total Peak**: <500 MB (vs 48 GB available)

---

## Performance Estimates

**Baseline (NLSQ least_squares, no extra optimization)**:
- JAX JIT compilation: automatic
- GPU acceleration: automatic (if available)
- Estimated: **20-40s for 3M points** (GPU) or **60-120s** (CPU)

**vs curve_fit_large (when it worked)**:
- curve_fit_large: ~5-10s (GPU, direct chunking)
- Our solution: 20-40s (GPU, indirect via residual_fn)
- **Trade-off**: 2-4x slower but **works correctly**

**vs scipy.optimize.least_squares**:
- scipy: 180s (CPU-only, no JAX acceleration)
- NLSQ: 20-40s (GPU) or 60-120s (CPU with JAX)
- **NLSQ 3-9x faster than scipy**

---

## Testing Strategy

### Unit Tests: `tests/unit/test_stratified_residual.py`

```python
def test_stratified_residual_function_initialization():
    """Test StratifiedResidualFunction can be created."""

def test_residual_function_returns_correct_shape():
    """Test residual function returns correct 1D array."""

def test_all_chunks_have_all_angles():
    """Test chunk validation detects missing angles."""

def test_jit_compilation_works():
    """Test JAX JIT compilation succeeds."""

def test_residual_values_match_direct_computation():
    """Test residuals match direct model computation."""
```

### Integration Tests: `tests/integration/test_nlsq_least_squares_stratified.py`

```python
def test_nlsq_least_squares_with_stratified_chunks():
    """Test NLSQ least_squares() works with stratified chunks."""

def test_convergence_with_per_angle_scaling():
    """Test optimization converges with per-angle parameters."""

def test_parameters_improve_from_initial():
    """Test cost decreases and parameters change."""

def test_covariance_reasonable():
    """Test covariance matrix is positive definite."""

def test_3m_points_completes_successfully():
    """Test 3M point dataset with per-angle scaling."""
```

### Performance Tests: `tests/performance/test_nlsq_stratified_performance.py`

```python
def test_performance_1m_points():
    """Benchmark 1M points, per-angle scaling."""

def test_performance_3m_points():
    """Benchmark 3M points, per-angle scaling."""

def test_memory_usage_under_limit():
    """Test peak memory < 1 GB."""

def test_gpu_vs_cpu_speedup():
    """Compare GPU vs CPU performance (if GPU available)."""
```

---

## CMC Integration (Same Pattern)

```python
def cmc_with_stratified_shards(data, n_shards, per_angle_scaling):
    """CMC using angle-stratified shards.

    Same stratification pattern ensures each shard has all angles.
    """
    # Step 1: Stratify data
    stratified_data = apply_stratification(
        data,
        per_angle_scaling=True,
        target_chunk_size=len(data) // n_shards
    )

    # Step 2: Each chunk becomes a shard
    shards = stratified_data.chunks

    # Step 3: MCMC with stratified residual function per shard
    def run_mcmc_on_shard(shard):
        # Create residual function for this shard
        residual_fn = StratifiedResidualFunction([shard], model, per_angle_scaling)

        # Define log likelihood
        def log_likelihood(params):
            residuals = residual_fn(params)
            return -0.5 * jnp.sum(residuals**2)

        # Run MCMC
        mcmc = numpyro.MCMC(
            numpyro.NUTS(log_likelihood),
            num_samples=num_samples,
            num_warmup=num_warmup
        )
        mcmc.run(...)
        return mcmc.get_samples()

    # Step 4: Parallel MCMC on all shards
    shard_posteriors = parallel_map(run_mcmc_on_shard, shards)

    # Step 5: Consensus aggregation
    return consensus_combine(shard_posteriors)
```

---

## Implementation Timeline

### Week 1: Core Implementation
- **Day 1-2**: Implement `StratifiedResidualFunction` class
- **Day 3**: Integrate `_fit_with_stratified_least_squares()` in `NLSQWrapper`
- **Day 4**: Unit tests (15 tests)
- **Day 5**: Integration tests (10 tests)

### Week 2: Validation & Optimization
- **Day 1-2**: Test on 3M point dataset, verify convergence
- **Day 3**: Performance benchmarking
- **Day 4**: Memory profiling
- **Day 5**: Documentation and examples

### Week 3-4: CMC Integration
- **Week 3**: Implement CMC stratified sharding
- **Week 4**: CMC tests and validation

---

## Success Criteria

### Phase 1 (Week 1)
- ✅ `StratifiedResidualFunction` class complete
- ✅ NLSQ `least_squares()` integrated
- ✅ 25 tests passing
- ✅ 3M points with per-angle scaling completes

### Phase 2 (Week 2)
- ✅ Optimization converges (non-zero iterations)
- ✅ Cost decreases from initial
- ✅ Parameters change from initial guess
- ✅ Performance: 20-120s for 3M points (GPU/CPU)
- ✅ Memory: <500 MB peak

### Phase 3 (Week 3-4)
- ✅ CMC works with per-angle scaling
- ✅ All shards have all angles
- ✅ Posteriors converge
- ✅ 20+ CMC tests passing

---

## Comparison: curve_fit_large vs least_squares

| Aspect | curve_fit_large | least_squares (with stratified residual) |
|--------|----------------|------------------------------------------|
| **API** | Model function f(x, *params)→y | Residual function fun(params)→residuals |
| **xdata/ydata** | Required | Optional (not used in our case) |
| **Chunking** | Automatic (breaks our structure) | Manual (we control in residual_fn) |
| **Per-angle scaling** | ❌ Fails (0 iterations) | ✅ Works (correct gradients) |
| **Performance** | Fast (if it worked) | Comparable (2-4x slower) |
| **Scalability** | >100M points | >100M points |
| **GPU Support** | ✅ Yes | ✅ Yes |

---

## Conclusion

**Recommended Solution**: Use NLSQ's `least_squares()` with `StratifiedResidualFunction`

**Why This Works**:
1. NLSQ's optimization engine (not scipy)
2. We control chunking (in residual function)
3. JAX acceleration + GPU support
4. Scales to >100M points
5. Works for NLSQ and CMC

**Timeline**: 1-2 weeks for full implementation and testing

**Confidence**: 95% (High - correct API, proven approach, manageable scope)
