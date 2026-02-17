# Mode-Aware Consensus Monte Carlo

**Date:** 2026-02-14 **Status:** Approved

## Problem

Standard Consensus MC (Scott et al., 2016) assumes per-shard posteriors are
approximately Gaussian. When shards have bimodal posteriors — e.g., D0 modes at ~19K and
~32K with 50/50 weights — the per-shard mean falls in the density trough between modes:

```
E[D0] = 0.5 × 19K + 0.5 × 32K = 25.5K   (low-density region)
Var[D0] includes 0.25 × (13K)² = 42.25M   (inflated ~10x by inter-modal term)
```

The consensus combination then produces a combined mean in the trough (~27.4K in the
production run), yielding a corrupted posterior that doesn't represent either physical
mode. NLSQ found D0~19.7K, leading to z-score=126.7 disagreement.

**Root cause:** The product-of-Gaussians identity underlying Consensus MC doesn't apply
to mixture distributions.

## Solution

Detect bimodal shards, extract per-component GMM statistics, jointly cluster shards into
mode populations, and run consensus MC separately per mode. Output a mixture
distribution.

## Data Flow

```
Per-shard MCMC results
    |
Bimodal detection (enriched with per-component stds)
    |
Cross-shard bimodality summary
    |
[If bimodal params exist]
    |
Joint clustering of shard means (k=2, range-normalized, seeded from summary)
    |
Per-mode consensus MC (using component-level mu, sigma^2 for bimodal shards)
    |
MCMCSamples with mixture-drawn primary samples + BimodalConsensusResult metadata
```

## Theory

### Why Consensus MC Breaks

Consensus MC exploits the product-of-Gaussians identity:

```
If p(theta | D_k) ~ N(mu_k, sigma_k^2), then:
  combined_precision = sum(1/sigma_k^2)
  combined_mean = (sum(mu_k / sigma_k^2)) / combined_precision
```

When a shard posterior is bimodal `p = w1 N(mu1, s1^2) + w2 N(mu2, s2^2)`:

1. `np.mean(samples) = w1*mu1 + w2*mu2` falls between modes
1. `np.var(samples) = w1*s1^2 + w2*s2^2 + w1*w2*(mu1-mu2)^2` is inflated by the
   inter-modal term
1. The Gaussian product identity doesn't apply to mixtures

### Mode-Aware Fix

For bimodal shards, use GMM **component** statistics `(mu_component, sigma_component^2)`
instead of full-posterior statistics. This removes the inter-modal inflation and places
per-shard contributions at the actual mode locations.

## Design

### New Types

```python
@dataclass
class ModeCluster:
    mean: dict[str, float]              # Per-param consensus mean for this mode
    std: dict[str, float]               # Per-param consensus std for this mode
    weight: float                       # Fraction of shards supporting this mode
    n_shards: int                       # Number of shards in this cluster
    samples: dict[str, np.ndarray]      # Generated samples from N(mean, std^2)

@dataclass
class BimodalConsensusResult:
    modes: list[ModeCluster]            # Typically 2 modes
    modal_params: list[str]             # Which parameters are bimodal
    co_occurrence: dict[str, Any]       # D0-alpha correlation info
```

### Joint Mode Clustering

1. Identify modal parameters (bimodal fraction > 5% of shards)
1. Build per-shard feature vectors from modal parameter means
1. Normalize features by parameter_space bounds range
1. Seed 2 centroids from cross-shard lower/upper mode means
1. Assign shards to nearest centroid (1 iteration of k-means)
   - Bimodal shards: each component assigned independently
   - Unimodal shards: full posterior assigned to nearest cluster

### Per-Mode Consensus

For each mode cluster:

1. Collect per-shard contributions:
   - Unimodal shards: use full posterior (mu, sigma^2)
   - Bimodal shards: use component-level (mu_component, sigma_component^2)
1. Run standard precision-weighted consensus MC on these statistics
1. Compute mode weight = n_shards_in_cluster / n_total_shards

### Output Integration

Primary `MCMCSamples.samples` contains mixture-drawn samples:

- Draw `w_mode * n_total_samples` from each mode's Gaussian
- Concatenate and shuffle

`MCMCSamples.bimodal_consensus: BimodalConsensusResult | None` carries full mode
details.

## Changes by File

| File | Change | |------|--------| | `diagnostics.py` | Add `stds` to `BimodalResult`;
add `ModeCluster`, `BimodalConsensusResult`, `cluster_shard_modes()` | |
`backends/base.py` | Add `combine_shard_samples_bimodal()` | |
`backends/multiprocessing.py` | Enrich detections with `std1`/`std2`; add mode-aware
consensus branch | | `sampler.py` | Add
`bimodal_consensus: BimodalConsensusResult | None = None` to `MCMCSamples` |

## Edge Cases

- No bimodal params: standard consensus (unchanged)
- All shards bimodal: both clusters have full data
- Only 1 bimodal parameter: joint clustering reduces to 1D nearest-mode
- \<3 shards in a cluster: fall back to simple mean (insufficient for consensus)

## What Does NOT Change

- Per-shard MCMC sampling (NUTS, warmup, etc.)
- Bimodal detection thresholds (min_weight=0.2, min_separation=0.5)
- Divergence-based quality filtering
- Heterogeneity abort logic
- Downstream consumers that only use `MCMCSamples.samples`
