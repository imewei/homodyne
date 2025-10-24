# Deprecated Configuration Templates

These templates have been **deprecated** as of Homodyne v2.0 and replaced with model-based templates.

## Migration Guide

**Old (v1.0) - Method-based templates:**
- `homodyne_default_comprehensive.yaml` → Use `homodyne_master_template.yaml`
- `homodyne_cmc_config.yaml` → Use `homodyne_static_isotropic.yaml` or `homodyne_laminar_flow.yaml` with `optimization.cmc.enable: true`
- `homodyne_streaming_config.yaml` → Use `homodyne_static_isotropic.yaml` or `homodyne_laminar_flow.yaml` with `optimization.streaming` settings

**New (v2.0+) - Model-based templates:**
- `homodyne_master_template.yaml` - Comprehensive reference with ALL methods
- `homodyne_static_isotropic.yaml` - 3+2n parameters (ALL methods included)
- `homodyne_laminar_flow.yaml` - 7+2n parameters (ALL methods included)

## Key Changes

1. **Organization:** Templates now organized by **physical model** (static vs laminar), not by **analysis method** (CMC, streaming, etc.)

2. **Completeness:** Each new template includes configuration for **ALL optimization methods**:
   - NLSQ (trust-region solver)
   - MCMC (uncertainty quantification)
   - Streaming (large datasets >100M points)
   - CMC (multi-angle combination)

3. **Parameter Counting:** Templates clearly document parameter count formulas:
   - Static Isotropic: 3 physical + 2 per angle
   - Laminar Flow: 7 physical + 2 per angle

## Why the Change?

The old method-based organization created confusion:
- Users had to choose a template based on optimization method, not physical system
- Each template only showed one method, hiding other options
- Unclear parameter counting across different templates

The new model-based organization:
- Choose template based on your physical system (static vs flowing)
- See all optimization methods in one place
- Clear parameter counting explanation

## Do Not Use These Templates

These deprecated templates are kept for reference only. They may be removed in future versions.

**Use the new templates instead:**
```bash
# For static systems
cp homodyne/config/templates/homodyne_static_isotropic.yaml my_config.yaml

# For flow systems
cp homodyne/config/templates/homodyne_laminar_flow.yaml my_config.yaml
```

## Documentation

See comprehensive documentation at:
- Configuration templates: `docs/configuration-templates/index.rst`
- Master template: `docs/configuration-templates/master-template.rst`
- Static isotropic: `docs/configuration-templates/static-isotropic.rst`
- Laminar flow: `docs/configuration-templates/laminar-flow.rst`
