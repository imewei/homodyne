# Homodyne

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![Documentation](https://img.shields.io/badge/docs-sphinx-blue.svg)](https://homodyne.readthedocs.io)
[![ReadTheDocs](https://readthedocs.org/projects/homodyne/badge/?version=latest)](https://homodyne.readthedocs.io/en/latest/)
[![DOI](https://zenodo.org/badge/DOI/10.1073/pnas.2401162121.svg)](https://doi.org/10.1073/pnas.2401162121)

CPU-optimized JAX package for X-ray Photon Correlation Spectroscopy (XPCS) analysis,
implementing the theoretical framework from
[He et al. PNAS 2024](https://doi.org/10.1073/pnas.2401162121) and
[He et al. PNAS 2025](https://doi.org/10.1073/pnas.2514216122) for characterizing
transport properties in flowing soft matter systems.

## Correlation Model

```
c2(phi, t1, t2) = offset + contrast * c1_diff(t1, t2) * [c1_shear(phi, t1, t2)]^2

c1_diff(t1, t2)        = exp[-q^2 * integral_{t1}^{t2} D(t') dt']
c1_shear(phi, t1, t2)  = sinc(q * L * cos(phi0 - phi) / (2*pi) * integral_{t1}^{t2} gamma_dot(t') dt')

D(t)          = D0 * t^alpha + D_offset
gamma_dot(t)  = gamma_dot_0 * t^beta + gamma_dot_offset
```

| Mode | Parameters | Count |
|------|------------|-------|
| **static** | D0, alpha, D_offset | 3 |
| **laminar_flow** | D0, alpha, D_offset, gamma_dot_0, beta, gamma_dot_offset, phi0 | 7 |

Per-angle contrast and offset are added automatically based on the number of azimuthal angles.

## Installation

```bash
pip install homodyne
```

For development:

```bash
git clone https://github.com/imewei/homodyne.git
cd homodyne
make dev    # or: uv sync --extra dev
```

**Requirements:** Python 3.12+, CPU-only (no GPU). Runs on Linux, macOS, and Windows.

## Quick Start

### CLI

```bash
# Generate a config template
homodyne-config --mode static --output config.yaml

# Run NLSQ optimization
homodyne --method nlsq --config config.yaml

# Run Consensus Monte Carlo for uncertainty quantification
homodyne --method cmc --config config.yaml
```

### Python API

```python
from homodyne.optimization import fit_nlsq_jax, fit_mcmc_jax
from homodyne.data import load_xpcs_data
from homodyne.config import ConfigManager

data = load_xpcs_data("config.yaml")
config = ConfigManager("config.yaml")

# NLSQ trust-region optimization
nlsq_result = fit_nlsq_jax(data, config)

# CMC with NLSQ warm-start for Bayesian uncertainty
cmc_result = fit_mcmc_jax(data, config, nlsq_result=nlsq_result)
```

### Data Flow

```
YAML config --> XPCSDataLoader(HDF5) --> HomodyneModel --> NLSQ or CMC --> Results (JSON + NPZ)
```

## Optimization Methods

**NLSQ** (primary) -- JAX-native trust-region Levenberg-Marquardt with automatic
anti-degeneracy defense, CMA-ES global search for multi-scale problems, and memory-aware
routing for large datasets.

**CMC** (secondary) -- Consensus Monte Carlo using NumPyro NUTS sampling with automatic
sharding, NLSQ warm-start priors, and multiprocessing across CPU cores. Produces
publication-quality posterior distributions with ArviZ diagnostics.

## Configuration

Homodyne uses YAML configuration files. Generate a template:

```bash
homodyne-config --mode laminar_flow --output config.yaml
```

Key sections:

```yaml
analysis_mode: "laminar_flow"
experimental_data:
  file_path: "data.h5"
optimization:
  method: "nlsq"
  nlsq:
    anti_degeneracy:
      per_angle_mode: "auto"   # auto, constant, individual, fourier
  cmc:
    sharding:
      max_points_per_shard: "auto"
```

See the [User Guide](https://homodyne.readthedocs.io/en/latest/user-guide/index.html)
for full configuration reference.

## CLI Commands

| Command | Purpose |
|---------|---------|
| `homodyne` | Run XPCS analysis (NLSQ/CMC) |
| `homodyne-config` | Generate and validate config files |
| `homodyne-config-xla` | Configure XLA device settings |
| `homodyne-post-install` | Install shell completion (bash/zsh/fish) |
| `homodyne-cleanup` | Remove shell completion files |

Shell completion and aliases are available after running `homodyne-post-install --interactive`.

## Development

```bash
make test       # Unit tests
make test-all   # Full suite + coverage
make quality    # Format + lint + type-check
```

## Documentation

- [User Guide](https://homodyne.readthedocs.io/en/latest/user-guide/index.html)
- [API Reference](https://homodyne.readthedocs.io/en/latest/api-reference/index.html)
- [Changelog](CHANGELOG.md)

## Citation

If you use Homodyne in your research, please cite:

```bibtex
@article{He2024,
  author  = {He, Hongrui and Liang, Hao and Chu, Miaoqi and Jiang, Zhang and
             de Pablo, Juan J and Tirrell, Matthew V and Narayanan, Suresh
             and Chen, Wei},
  title   = {Transport coefficient approach for characterizing nonequilibrium
             dynamics in soft matter},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {121},
  number  = {31},
  year    = {2024},
  doi     = {10.1073/pnas.2401162121}
}
```

```bibtex
@article{He2025,
  author  = {He, Hongrui and Liang, Heyi and Chu, Miaoqi and Jiang, Zhang and
             de Pablo, Juan J and Tirrell, Matthew V and Narayanan, Suresh
             and Chen, Wei},
  title   = {Bridging microscopic dynamics and rheology in the yielding
             of charged colloidal suspensions},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {122},
  number  = {42},
  year    = {2025},
  doi     = {10.1073/pnas.2514216122}
}
```

## License

MIT License -- see [LICENSE](LICENSE) for details.

## Authors

- Wei Chen (weichen@anl.gov) -- Argonne National Laboratory
- Hongrui He (hhe@anl.gov) -- Argonne National Laboratory
