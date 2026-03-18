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

## Homodyne Model

The package implements a single-component scattering model where correlation decay
encodes diffusion and shear dynamics. The scattered intensity auto-correlates to
produce a two-time correlation function whose shape reveals transport coefficients.

### Two-Time Correlation

$$c_2(\vec{q}, t_1, t_2) = 1 + \beta \times [c_1(\vec{q}, t_1, t_2)]^2$$

where $\beta$ the optical contrast and the intermediate scattering function combines diffusion and shear:

$$c_1(\vec{q}, t_1, t_2) = \exp\left(-q^2 \int_{t_1}^{t_2} J(t')\ dt'\right) \times \mathrm{sinc}\left(\frac{qL\cos(\phi)}{2\pi} \int_{t_1}^{t_2} \dot{\gamma}(t')\ dt'\right)$$

where $\phi$ is the angle between the scattering vector and the flow direction.

### Rate Functions

Transport coefficients follow power-law forms:

$$J(t) = D_0 \cdot t^{\alpha} + D_{\text{offset}} \qquad \dot{\gamma}(t) = \dot{\gamma}_0 \cdot t^{\beta} + \dot{\gamma}_{\text{offset}}$$

All time integrals are evaluated **numerically** via cumulative trapezoid on the discrete
time grid -- no analytical antiderivatives are ever used, ensuring correctness for the
general power-law form.

### Parameters

The model has up to 7 physics parameters organized into three groups, plus 2 per-angle
scaling parameters:

**Diffusion transport** -- $J(t) = D_0 t^\alpha + D_{\text{offset}}$

| Parameter | Description | Default | Units |
|-----------|-------------|---------|-------|
| `D0` | Diffusion prefactor | 1e4 | Å²/s |
| `alpha` | Transport exponent (0 = Wiener, 1 = ballistic) | 0.0 | -- |
| `D_offset` | Transport rate offset | 0.0 | Å²/s |

**Shear rate** (laminar_flow only) -- $\dot\gamma(t) = \dot\gamma_0 t^\beta + \dot\gamma_{\text{offset}}$

| Parameter | Description | Default | Units |
|-----------|-------------|---------|-------|
| `gamma_dot_0` | Shear rate prefactor | 1e-3 | s⁻¹ |
| `beta` | Shear rate exponent (0 = constant shear) | 0.0 | -- |
| `gamma_dot_offset` | Shear rate offset | 0.0 | s⁻¹ |

**Flow angle**

| Parameter | Description | Default | Units |
|-----------|-------------|---------|-------|
| `phi0` | Flow angle offset relative to q-vector | 0.0 | degrees |

**Per-angle scaling** (2 parameters per detector angle)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `contrast` | Optical contrast (speckle contrast) | 0.5 |
| `offset` | Baseline offset | 1.0 |

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
make dev    # or: uv sync
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
from homodyne.data import load_xpcs_data
from homodyne.config import ConfigManager
from homodyne.optimization import fit_nlsq_jax
from homodyne.optimization.cmc import fit_mcmc_jax

# Load data and config
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
