CMC-Only Migration (v3.0)
=========================

Summary
-------
- MCMC now always runs Consensus Monte Carlo (CMC); NUTS-only paths and auto-selection flags are removed.
- Per-phi initial values are taken from config when provided, otherwise derived from per-phi percentiles; shards preserve phi stratification.
- Single-shard runs still use NumPyro/BlackJAX NUTS inside CMC.

User actions
------------
1. Remove deprecated CLI flags: ``--min-samples-cmc`` and ``--memory-threshold-pct``.
2. Update configs to the new CMC-only fields:
   - ``optimization.mcmc.sharding``: ``num_shards``, ``seed_base``.
   - ``optimization.mcmc.initial_values.phi``: per-phi contrast/offset (fallback computed automatically).
3. Regenerate configs from templates in ``homodyne/config/templates`` for the updated schema.

Removed/changed
---------------
- ``homodyne.device.config.should_use_cmc`` deprecated shim (always True, slated for removal).
- Auto-selection docs/examples/tests deleted (NUTS-only flows removed).
- NUTS/selection integration tests replaced by CMC-only coverage.

Testing
-------
- Run ``make test`` (unit) and ``make test-all`` after updating configs.
- Spot-check small and large datasets; single-angle datasets should run as single-shard CMC (NUTS per shard).
