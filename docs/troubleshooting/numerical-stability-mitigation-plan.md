# Physics-Induced NaN/Inf Mitigation Plan

_Last updated: 2025-11-13_

## Objectives

- Catalogue every laminar_flow/c2_theory/c2_fitted operation that can emit NaN/Inf.
- Define mathematically sound clamps/epsilons/bounded transforms to keep values in physical ranges.
- Provide HOMODYNE_DEBUG_INIT-gated diagnostics that log the first non-finite along the chain with enough context to reproduce the issue, while keeping the default (debug disabled) path cost-free.

## High-Risk Operations

| Operation                             | Why It Blows Up                            | Existing Guard | Needed Action |
|---------------------------------------|--------------------------------------------|----------------|---------------|
| `contrast / sigma`, `1 / sigma^2`     | sigma can be ~0 after config mistakes      | partial        | `safe_div`    |
| `log(sigma^2)`                        | sigma^2 ≤ 0                                | ❌             | `safe_log`    |
| `sqrt(velocity^2 - shear)`            | subtraction goes negative after rounding   | ❌             | `safe_sqrt`   |
| `exp(z)` in kernels                   | overflow to Inf when z > ~700 (float64)    | ❌             | clip before exp |
| Power laws `t^alpha`                  | t=0 with alpha<0                           | ❌             | clamp t, softplus alpha |
| Phase wraps feeding `sin/cos`         | runaway arguments → catastrophic cancellation | partial     | clamp phase to [-50π, 50π] |
| `c2_fitted = contrast * theory + offset` | upstream errors propagate directly     | weak clamp     | staged clipping (see below) |

## Mitigation Pillars

### 1. Mathematical Safeguards

- **Bounded priors**: prefer TruncatedNormal/BetaScaled for parameters with physical ranges (already in-progress for MCMC).
- **Positive transforms**: represent strictly-positive parameters as `softplus(raw) + eps`; keep `eps = max(1e-12, 10 * finfo(dtype).tiny)`.
- **Phase clamping**: normalize/snap laminar flow phases into `[-50π, 50π]` before calling `sin/cos` to avoid runaway rounding error.
- **Safe c2_fitted calculation**:
  1. `c2_theory_safe = clip(c2_theory, 0.9, 2.2)` to keep within expected physics envelope.
  2. `scaled = clip(contrast * c2_theory_safe, -10, 10)`.
  3. `c2_fitted_safe = clip(scaled + offset, 0.5, 2.5)`.
- **Safe math helpers (`homodyne/core/safe_math.py`)**:
  - `safe_div(n, d, name=None)` (signed epsilon on denominator).
  - `safe_log(x, name=None)` (lower-bound x by epsilon).
  - `safe_sqrt(x, name=None)` (ensure x ≥ 0 + eps).
  - `safe_exp(z, name=None)` (clip input to `[log(tiny)+2, log(max)-2]`).
  - `safe_logsumexp(x, axis, name=None)`.

### 2. Diagnostics (HOMODYNE_DEBUG_INIT)

- **Environment flag**: `HOMODYNE_DEBUG_INIT=1[,first_only=1,min_size=10000,sample=8]`.
- **Debugger API (`homodyne/core/debug.py`)**:
  - `dbg.enabled` boolean, parsed once from the env.
  - `dbg.mark(tag, array, meta=None)`: if enabled, lazily scans the array (respecting `sample`/`min_size`) and logs the first non-finite with min/max/mean/count + optional first index.
  - `dbg.summary()` registered via `atexit` to print per-tag counts.
- **Instrumentation sites** (initial 15): data ingestion (sigma, t1/t2), physics intermediates (Reynolds numbers, shear terms), `c2_theory`, `contrast`, `offset`, `c2_fitted`, likelihood terms, and any per-angle scaling arrays.
- **Cost control**: when debug is off the API is a no-op; when on it short-circuits after the first logged failure per tag by default.

### 3. Implementation Staging

| Phase | Scope                                    | Risk | Success Criteria |
|-------|------------------------------------------|------|------------------|
| 1     | Drop-in diagnostics + `safe_math` helpers | Low  | Debug logs show first non-finite without affecting production perf |
| 2     | Apply clamps in laminar_flow/c2_fitted    | Med  | No regressions on golden datasets; NaN repro case now finite |
| 3     | Reparameterize/transform risky priors     | Med+ | Stable inference for laminar_flow config C020 |

## Example Usage

```python
from homodyne.core import safe_math as sm, debug as dbg

def log_likelihood(y, mu, log_sigma2):
    sigma2 = sm.safe_exp(log_sigma2, name="sigma2")
    resid = dbg.mark("residual", y - mu)
    term = sm.safe_div(resid * resid, sigma2, name="resid_over_sigma2")
    log_norm = sm.safe_log(2.0 * np.pi * sigma2, name="log2pi_sigma2")
    ll = -0.5 * (term + log_norm)
    return ll.sum()
```

## Testing & Monitoring

- **Unit tests**: feed `safe_*` helpers adversarial values (negative variances, zero denominators, large log arguments) to ensure they stay finite and monotonic in valid regions.
- **Integration tests**: 
  - Regression fixture for `c2_fitted_safe` verifying clamps keep values within `[0.5, 2.5]` even when `contrast/offset` are extreme.
  - Smoke test with `HOMODYNE_DEBUG_INIT=1` to ensure first non-finite logging works and summarises tags at exit.
- **Metrics**: count of diagnostic warnings per tag, time until first non-finite, coverage of instrumentation (arrays seen vs with issues).

## Next Steps

1. Land the `safe_math` + `debug` modules and wire the highest-risk laminar_flow call sites (contrast/offset scaling, c2_fitted) through them.
2. Enable diagnostics in the consensus-MCMC worker so shard failures dump the first offending tag before retry logic triggers.
3. Iterate on clamps/epsilons using real repro logs; relax bounds only once the physics team confirms the safe envelope.
