# Test Suite Cleanup Analysis
**Date**: 2025-11-15
**Scope**: `/home/wei/Documents/GitHub/homodyne/tests`

## Executive Summary

The homodyne test suite has grown to **110 files** with **1,576 test functions** (49,165 lines of code). Analysis reveals significant opportunities for consolidation and cleanup, particularly after the v2.3.0 GPU removal and mode standardization work.

**Key Findings**:
- ✅ No obsolete mode terminology remaining (cleaned up in previous work)
- ⚠️ GPU-related tests still present (2 test functions, 13 files with GPU references)
- ⚠️ 23 duplicate test function names across different files
- ⚠️ 73+ skipped/xfailed tests (potential obsolete tests)
- ⚠️ High test fragmentation: 21 CMC-related files, 12 NLSQ files, 10 parameter files
- ⚠️ 2 PjitBackend tests failing due to parameter ordering issues

**Recommended Reduction**: 20-30% (targeting ~75-85 files, ~1200 functions)

---

## Current Test Suite Structure

### By Category
| Category | Files | Functions | Notes |
|----------|-------|-----------|-------|
| **unit** | 63 | 1,226 | Largest category, high fragmentation |
| **integration** | 15 | 169 | Some overlap with unit tests |
| **performance** | 5 | 48 | 11 skipped tests |
| **validation** | 3 | 32 | Scientific validation |
| **mcmc** | 3 | 32 | MCMC-specific |
| **regression** | 2 | 22 | Quality regression |
| **property** | 1 | 12 | Mathematical properties |
| **api** | 1 | 19 | 17 skipped tests |
| **self_consistency** | 1 | 16 | 6 skipped tests |
| **factories** | 5 | 0 | Test data generators |
| **Total** | **110** | **1,576** | |

### Largest Files (>750 lines)
1. `test_cli_validation.py` - 881 lines
2. `test_angle_filtering.py` - 859 lines
3. `test_sharding.py` - 814 lines
4. `test_automatic_cmc_selection.py` - 807 lines
5. `test_parameter_space_config.py` - 789 lines
6. `test_cmc_integration.py` - 777 lines
7. `test_backend_implementations.py` - 772 lines

---

## Identified Issues

### 1. GPU-Related Obsolete Tests

**Status**: Should be removed (GPU support removed in v2.3.0)

**GPU Test Functions** (2 total):
- `tests/unit/test_hardware_detection.py::test_detect_gpu_system`
- `tests/unit/test_hardware_detection.py::test_gpu_memory_detection_fallback`

**Files with GPU References** (13 files):
```
tests/performance/__init__.py
tests/performance/test_benchmarks.py
tests/__init__.py
tests/api/test_compatibility.py
tests/conftest.py
tests/unit/test_coordinator.py
tests/unit/test_jax_backend.py
tests/unit/test_backend_infrastructure.py
tests/unit/test_hardware_detection.py
tests/unit/test_backend_implementations.py
tests/mcmc/test_nuts_validation.py
tests/integration/test_nlsq_workflow.py
tests/integration/test_parameter_recovery.py
```

**Recommendation**:
- Remove 2 GPU-specific test functions from `test_hardware_detection.py`
- Update remaining 11 files to remove GPU references in comments/docstrings
- Keep CPU fallback paths for backward compatibility

---

### 2. Duplicate Test Function Names

**Status**: 23 duplicate names found

**Critical Duplicates** (same functionality):

**CLI Validation Duplicates** (6 pairs):
```python
# test_cli_validation.py vs test_cli_args.py
- test_cmc_backend_accepted_with_mcmc
- test_cmc_num_shards_accepted_with_mcmc
- test_default_method_is_nlsq
- test_method_mcmc_accepted
- test_method_nlsq_accepted
- test_dense_mass_matrix_override (also in test_mcmc_model.py)
```

**Parameter Management Duplicates** (3 sets):
```python
# test_parameter_manager_advanced.py vs test_config_initial_params.py vs test_config_manager_parameters.py
- test_parameter_name_mapping

# test_physics.py vs test_parameter_manager_physics.py
- test_laminar_flow_parameters
- test_static_mode_parameters
```

**Backend Testing Duplicates** (2 pairs):
```python
# test_coordinator.py vs test_backend_implementations.py
- test_backend_selection_cpu

# test_backend_infrastructure.py vs test_backend_implementations.py
- test_backend_selection_pbs_cluster
```

**Integration Testing Duplicates** (2 pairs):
```python
# test_parameter_transformation.py vs test_cmc_integration.py
- test_empty_data_handling

# test_cmc_config.py vs test_cmc_integration.py
- test_invalid_num_shards
```

**Model Testing Duplicates** (4 sets):
```python
# test_mcmc_model.py vs test_parameter_space_config.py
- test_to_numpyro_kwargs_normal
- test_to_numpyro_kwargs_truncated_normal
- test_to_numpyro_kwargs_uniform
- test_repr (also in test_homodyne_model.py)
```

**Recommendation**:
- Consolidate each duplicate set into a single canonical location
- Delete redundant test instances
- Estimated reduction: 23 test functions

---

### 3. Skipped/XFailed Tests

**Status**: 73+ tests skipped across 20 files

**Files with Skipped Tests**:
```
tests/performance/test_wrapper_overhead.py: 6 skipped
tests/performance/test_benchmarks.py: 11 skipped
tests/performance/test_nlsq_performance.py: 3 skipped
tests/performance/test_stratified_chunking_performance.py: 1 skipped
tests/api/test_compatibility.py: 17 skipped
tests/self_consistency/test_cmc_consistency.py: 6 skipped
tests/conftest.py: 3 skipped
tests/unit/test_cmc_coordinator.py: 1 skipped
tests/unit/test_coordinator.py: 1 skipped
tests/unit/test_nlsq_saving.py: 2 skipped
tests/unit/test_cmc_backend_laminar_flow.py: 1 skipped
tests/unit/test_cmc_config.py: 1 skipped
tests/unit/test_angle_filtering_jax.py: 1 skipped
tests/unit/test_diagonal_correction.py: 7 skipped
tests/unit/test_data_loader.py: 13 skipped
tests/unit/test_mcmc_model.py: 4 skipped
tests/unit/test_jax_backend.py: 2 skipped
tests/unit/test_failure_injection.py: 5 skipped
tests/unit/test_hardware_detection.py: 1 skipped
tests/unit/test_optimization_nlsq.py: 10 skipped
```

**Recommendation**:
- Review each skipped test
- Delete permanently obsolete tests (e.g., GPU-only, deprecated features)
- Re-enable or fix tests that should pass
- Keep intentionally skipped tests (e.g., slow benchmarks) with clear rationale

---

### 4. High Test Fragmentation

**Status**: Multiple files testing same components

**Angle Filtering** (4 files):
```
tests/performance/test_angle_filtering_performance.py
tests/unit/test_angle_filtering_jax.py
tests/unit/test_angle_filtering.py
tests/unit/test_angle_filtering_consistency.py
```
**Consolidation Target**: Merge into 2 files (unit + performance)

**NLSQ** (12 files):
```
tests/performance/test_nlsq_performance.py
tests/regression/test_nlsq_quality_regression.py
tests/unit/test_nlsq_wrapper_integration.py
tests/unit/test_nlsq_saving.py
tests/unit/test_nlsq_api_handling.py
tests/unit/test_optimization_nlsq.py
tests/unit/test_nlsq_public_api.py
tests/unit/test_nlsq_wrapper.py
tests/integration/test_nlsq_end_to_end.py
tests/integration/test_nlsq_filtering.py
tests/integration/test_nlsq_workflow.py
tests/integration/test_stratified_nlsq_integration.py
```
**Consolidation Target**: Merge into 4-5 files (unit core, integration, performance, regression, API)

**MCMC/CMC** (21 files):
```
tests/self_consistency/test_cmc_consistency.py
tests/unit/test_cmc_coordinator.py
tests/unit/test_cmc_backend_laminar_flow.py
tests/unit/test_cmc_config.py
tests/unit/test_mcmc_selection.py
tests/unit/test_mcmc_model.py
tests/unit/test_mcmc_integration.py
tests/unit/test_mcmc_init.py
tests/unit/test_mcmc_result_extension.py
tests/unit/test_mcmc_visualization.py
tests/mcmc/test_statistical_validation.py
tests/mcmc/test_nuts_validation.py
tests/mcmc/test_mcmc_simplified.py
tests/integration/test_cmc_integration.py
tests/integration/test_cmc_results.py
tests/integration/test_config_driven_mcmc.py
tests/integration/test_mcmc_filtering.py
tests/integration/test_mcmc_regression.py
tests/integration/test_mcmc_simplified_workflow.py
tests/integration/test_automatic_cmc_selection.py
tests/validation/test_cmc_accuracy.py
```
**Consolidation Target**: Merge into 6-7 files (unit core, backends, integration, validation, nuts, statistical)

**Parameter Management** (10 files):
```
tests/unit/test_parameter_gradients.py
tests/unit/test_parameter_manager_advanced.py
tests/unit/test_parameter_manager_physics.py
tests/unit/test_parameter_expansion.py
tests/unit/test_parameter_manager.py
tests/unit/test_parameter_names_consistency.py
tests/unit/test_parameter_space_config.py
tests/unit/test_config_manager_parameters.py
tests/unit/test_config_initial_params.py
tests/integration/test_parameter_recovery.py
```
**Consolidation Target**: Merge into 4 files (manager, space, physics, integration)

**Config Management** (8 files):
```
tests/unit/test_cmc_config.py
tests/unit/test_config_initial_params.py
tests/unit/test_device_config.py
tests/unit/test_parameter_space_config.py
tests/unit/test_config_manager_parameters.py
tests/unit/test_edge_cases_config_validation.py
tests/integration/test_config_driven_mcmc.py
tests/integration/test_cli_config_integration.py
```
**Consolidation Target**: Merge into 3-4 files (manager, validation, integration)

**CLI Testing** (6 files):
```
tests/unit/test_cli_overrides.py
tests/unit/test_cli_integration.py
tests/unit/test_cli_validation.py
tests/unit/test_cli_data_loading.py
tests/unit/test_cli_args.py
tests/integration/test_cli_config_integration.py
```
**Consolidation Target**: Merge into 2-3 files (args+validation, integration, overrides)

**Backend Testing** (4 files):
```
tests/unit/test_cmc_backend_laminar_flow.py
tests/unit/test_jax_backend.py
tests/unit/test_backend_infrastructure.py
tests/unit/test_backend_implementations.py
```
**Consolidation Target**: Merge into 2 files (implementations, infrastructure)

---

### 5. Current Test Failures

**Status**: 2 PjitBackend tests failing

**Failed Tests**:
- `test_pjit_backend_single_shard`
- `test_pjit_backend_multiple_shards`

**Root Cause**: Parameter ordering mismatch
```
NumPyro expects per-angle params (contrast_*, offset_*) FIRST, then physical params
Got: ['D0', 'alpha', 'D_offset', 'contrast_0', 'offset_0']...
Expected: ['contrast_0', 'contrast_1', 'contrast_2', 'contrast_3', 'contrast_4']...
```

**Recommendation**: This is a **code bug**, not a test issue. The test correctly catches the problem. Fix the parameter ordering in the CMC coordinator before cleanup.

---

## Consolidation Strategy

### Phase 1: Remove Obsolete Tests (Immediate)
**Target**: Remove ~15-20 files

1. **GPU Tests** (2 functions in `test_hardware_detection.py`)
   - Delete: `test_detect_gpu_system`, `test_gpu_memory_detection_fallback`
   - Remove GPU references from 11 other files

2. **Permanently Skipped Tests** (~30-40 tests)
   - Review each @pytest.mark.skip
   - Delete tests marked as "obsolete", "deprecated", "GPU-only"
   - Keep slow benchmarks with clear documentation

3. **Duplicate Tests** (23 functions)
   - Consolidate duplicates into canonical locations
   - Delete redundant instances

**Estimated Reduction**: 55-65 test functions, ~3,000 lines of code

### Phase 2: Consolidate Fragmented Tests (1-2 weeks)
**Target**: Reduce by ~20-25 files

1. **Angle Filtering**: 4 files → 2 files
   - `test_angle_filtering.py` (unit: core functionality)
   - `test_angle_filtering_performance.py` (benchmarks)
   - Merge: consistency + JAX tests into core
   - **Reduction**: 2 files

2. **NLSQ**: 12 files → 5 files
   - `test_nlsq_core.py` (wrapper, optimization, API)
   - `test_nlsq_integration.py` (end-to-end, workflow, stratified)
   - `test_nlsq_saving.py` (result saving)
   - `test_nlsq_performance.py` (benchmarks)
   - `test_nlsq_regression.py` (quality regression)
   - **Reduction**: 7 files

3. **MCMC/CMC**: 21 files → 7 files
   - `test_cmc_core.py` (coordinator, config, selection)
   - `test_cmc_backends.py` (pjit, multiprocessing, PBS, laminar_flow)
   - `test_cmc_integration.py` (workflows, filtering, automatic selection)
   - `test_mcmc_model.py` (model, initialization, visualization)
   - `test_nuts_validation.py` (NUTS-specific)
   - `test_statistical_validation.py` (statistical tests)
   - `test_cmc_accuracy.py` (validation)
   - **Reduction**: 14 files

4. **Parameter Management**: 10 files → 4 files
   - `test_parameter_manager.py` (core, advanced, names)
   - `test_parameter_space.py` (space, config)
   - `test_parameter_physics.py` (physics, gradients)
   - `test_parameter_integration.py` (recovery, expansion, transformation)
   - **Reduction**: 6 files

5. **Config Management**: 8 files → 4 files
   - `test_config_manager.py` (core, parameters)
   - `test_config_validation.py` (validation, edge cases)
   - `test_config_initial_params.py` (initial parameters)
   - `test_config_integration.py` (CLI integration, device config, CMC config)
   - **Reduction**: 4 files

6. **CLI Testing**: 6 files → 3 files
   - `test_cli_args_validation.py` (args, validation merged)
   - `test_cli_integration.py` (integration, config)
   - `test_cli_overrides.py` (overrides, data loading)
   - **Reduction**: 3 files

7. **Backend Testing**: 4 files → 2 files
   - `test_backends.py` (implementations, infrastructure, CMC backend)
   - `test_jax_backend.py` (JAX-specific)
   - **Reduction**: 2 files

**Total Phase 2 Reduction**: ~38 files

### Phase 3: Quality Improvements (Optional)
**Target**: Improve test organization

1. **Standardize Test Structure**
   - Consistent naming: `test_<component>_<aspect>.py`
   - Group related tests with classes
   - Clear docstrings for each test

2. **Improve Test Factories**
   - Consolidate `optimization_factory.py` and `data_factory.py`
   - Add comprehensive fixture documentation

3. **Performance Test Management**
   - Create `@pytest.mark.slow` for long-running tests
   - Skip by default, run in CI nightly

---

## Proposed Final Structure

```
tests/
├── conftest.py
├── __init__.py
├── test_runner.py
│
├── unit/ (40-45 files, down from 63)
│   ├── test_angle_filtering.py (consolidated)
│   ├── test_backends.py (consolidated)
│   ├── test_checkpoint_manager.py
│   ├── test_cli_args_validation.py (consolidated)
│   ├── test_cli_integration.py
│   ├── test_cli_overrides.py (consolidated)
│   ├── test_cmc_core.py (consolidated)
│   ├── test_cmc_backends.py (consolidated)
│   ├── test_combination.py
│   ├── test_config_manager.py (consolidated)
│   ├── test_config_validation.py (consolidated)
│   ├── test_config_initial_params.py
│   ├── test_data_loader.py
│   ├── test_device_config.py
│   ├── test_diagnostics.py
│   ├── test_diagonal_correction.py
│   ├── test_failure_injection.py
│   ├── test_hardware_detection.py (GPU tests removed)
│   ├── test_homodyne_model.py
│   ├── test_jax_backend.py
│   ├── test_jax_operations.py
│   ├── test_mcmc_model.py (consolidated)
│   ├── test_nlsq_core.py (consolidated)
│   ├── test_nlsq_saving.py
│   ├── test_parameter_manager.py (consolidated)
│   ├── test_parameter_space.py (consolidated)
│   ├── test_parameter_physics.py (consolidated)
│   ├── test_per_angle_scaling.py
│   ├── test_physics.py
│   ├── test_recovery_strategies.py
│   ├── test_residual_function.py
│   ├── test_sequential_angle.py
│   ├── test_sharding.py
│   ├── test_simulated_data_fixes.py
│   ├── test_stable_prior_fallback.py
│   ├── test_strategy_selection.py
│   ├── test_stratification_diagnostics.py
│   ├── test_stratified_chunking.py
│   ├── test_stratified_residual.py
│   └── test_streaming_optimizer.py
│
├── integration/ (8-10 files, down from 15)
│   ├── test_cmc_integration.py (consolidated)
│   ├── test_config_integration.py (consolidated)
│   ├── test_nlsq_integration.py (consolidated)
│   ├── test_parameter_integration.py (consolidated)
│   ├── test_optimization_edge_cases.py
│   └── test_workflows.py
│
├── performance/ (5 files, unchanged)
│   ├── test_angle_filtering_performance.py (from consolidation)
│   ├── test_benchmarks.py
│   ├── test_nlsq_performance.py
│   ├── test_stratified_chunking_performance.py
│   └── test_wrapper_overhead.py
│
├── mcmc/ (3 files, unchanged)
│   ├── test_nuts_validation.py
│   ├── test_statistical_validation.py
│   └── test_mcmc_simplified.py
│
├── validation/ (3 files, unchanged)
│   ├── test_cmc_accuracy.py
│   ├── test_real_data_stratification.py
│   └── test_scientific_validation.py
│
├── regression/ (2 files, unchanged)
│   ├── test_nlsq_regression.py (from consolidation)
│   └── test_save_results_compat.py
│
├── property/ (1 file, unchanged)
│   └── test_mathematical_properties.py
│
├── api/ (1 file, unchanged)
│   └── test_compatibility.py
│
├── self_consistency/ (1 file, unchanged)
│   └── test_cmc_consistency.py
│
└── factories/ (4-5 files, consolidated)
    ├── __init__.py
    ├── config_factory.py
    ├── data_factory.py (consolidated with optimization_factory)
    ├── large_dataset_factory.py
    └── synthetic_data.py
```

**Final Count**: ~75-85 files (down from 110), ~1,150-1,250 functions (down from 1,576)
**Reduction**: 25-35 files (23-32%), 325-425 functions (21-27%)

---

## Implementation Plan

### Week 1: Quick Wins (Phase 1)
- [ ] Remove 2 GPU test functions from `test_hardware_detection.py`
- [ ] Clean GPU references from 11 files (comments/docstrings only)
- [ ] Review and delete permanently skipped tests (30-40 tests)
- [ ] Consolidate 23 duplicate test functions
- [ ] Create PR: "test: remove obsolete GPU tests and duplicates"

### Week 2-3: NLSQ + Angle Filtering (Phase 2a)
- [ ] Consolidate angle filtering: 4 files → 2 files
- [ ] Consolidate NLSQ: 12 files → 5 files
- [ ] Update import paths
- [ ] Verify all tests pass
- [ ] Create PR: "test: consolidate NLSQ and angle filtering tests"

### Week 4-5: MCMC/CMC (Phase 2b)
- [ ] Consolidate MCMC/CMC: 21 files → 7 files
- [ ] Update import paths
- [ ] Verify all tests pass
- [ ] Create PR: "test: consolidate MCMC/CMC tests"

### Week 6: Parameters + Config (Phase 2c)
- [ ] Consolidate parameter tests: 10 files → 4 files
- [ ] Consolidate config tests: 8 files → 4 files
- [ ] Update import paths
- [ ] Verify all tests pass
- [ ] Create PR: "test: consolidate parameter and config tests"

### Week 7: CLI + Backends (Phase 2d)
- [ ] Consolidate CLI tests: 6 files → 3 files
- [ ] Consolidate backend tests: 4 files → 2 files
- [ ] Update import paths
- [ ] Verify all tests pass
- [ ] Create PR: "test: consolidate CLI and backend tests"

### Week 8: Quality + Documentation (Phase 3, optional)
- [ ] Standardize test structure and naming
- [ ] Add comprehensive docstrings
- [ ] Update test documentation in docs/
- [ ] Create PR: "test: improve test organization and documentation"

---

## Success Metrics

**Quantitative**:
- Test suite size: 110 → 75-85 files (23-32% reduction)
- Test count: 1,576 → 1,150-1,250 functions (21-27% reduction)
- Lines of code: 49,165 → ~35,000-40,000 lines (20-29% reduction)
- Test execution time: Baseline → TBD (target <10% increase)

**Qualitative**:
- Clearer test organization and discoverability
- Reduced duplication and maintenance burden
- Easier onboarding for new contributors
- Better test coverage documentation

---

## Risks and Mitigation

**Risk**: Breaking existing CI workflows
- **Mitigation**: Run full test suite before/after each PR

**Risk**: Accidentally removing critical test coverage
- **Mitigation**: Review test coverage reports (pytest-cov) before/after

**Risk**: Import path updates causing failures
- **Mitigation**: Use automated refactoring tools (e.g., rope, bowler)

**Risk**: Merge conflicts during multi-week effort
- **Mitigation**: Work in small PRs, coordinate with team

---

## References

- **CLAUDE.md**: Test suite structure and conventions
- **pytest.ini**: Test configuration
- **Makefile**: Test execution targets
- **v2.3.0 Migration**: GPU removal context
- **Recent Work**: Mode terminology standardization

---

**Next Steps**: Review this analysis with the team and prioritize phases based on urgency and available resources.
