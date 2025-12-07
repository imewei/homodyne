# Homodyne Scripts & Demos

Organized, ready-to-run scripts for Homodyne. All are CPU-only and grouped by workflow.

## Layout

- `nlsq/` – NLSQ optimization demos (static, laminar flow, streaming, overlays, CPU
  tuning)
- `mcmc/` – CMC/NUTS uncertainty quantification demos (small and large datasets)
- `benchmarks/` – Performance checks (e.g., oneDNN)
- `utils/` – Utilities (angle filtering helper, shell completion setup)
- `notebooks/` – Interactive tutorials (`nlsq_optimization_example.ipynb`)

## Quick starts

### NLSQ (static isotropic)

```bash
python scripts/nlsq/static_isotropic_nlsq.py
```

### NLSQ (laminar flow)

```bash
python scripts/nlsq/laminar_flow_nlsq.py
```

### CPU optimization guide

```bash
python scripts/nlsq/cpu_optimization.py
```

### Batch processing

```bash
python scripts/nlsq/multi_core_batch_processing.py
```

### Streaming 100M points

```bash
python scripts/nlsq/streaming_100m_points.py
```

### MCMC/CMC uncertainty

```bash
python scripts/mcmc/mcmc_uncertainty.py
python scripts/mcmc/mcmc_integration_demo.py
python scripts/mcmc/cmc_large_dataset.py
```

### Diagnostics & overlays

```bash
python scripts/nlsq/overlay_solver_vs_posthoc.py
```

### oneDNN benchmark

```bash
python scripts/benchmarks/benchmark_onednn.py
```

### Utilities

```bash
python scripts/utils/angle_filtering.py
bash scripts/utils/setup_shell_completion.sh
```

### Notebook

```bash
jupyter notebook scripts/notebooks/nlsq_optimization_example.ipynb
```
