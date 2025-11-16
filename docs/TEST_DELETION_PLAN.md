# Test Deletion Plan - Week 1, Task 3

## Summary

**Total Tests to Delete:** 16 permanently skipped tests
**Estimated Line Reduction:** ~400-600 lines
**Reason:** These tests are for deprecated/removed features, unimplemented functionality, or infeasible test approaches

---

## Category 1: Deprecated/Removed Features (v2.1.0 API Changes)

### 1.1 test_optimization_nlsq.py (2 tests)
- **Line 55-75:** `test_nlsq_result_structure` - NLSQResult deprecated, replaced by OptimizationResult
- **Line 366-395:** `test_nlsq_result_serialization` - NLSQResult deprecated

### 1.2 test_coordinator.py (1 test)
- **Line 282:** `test_svi_initialization_from_config` - v2.1.0 removed SVI initialization

### 1.3 test_cmc_config.py (1 test)
- **Line 313:** `test_invalid_initialization_method` - v2.1.0 removed mcmc.initialization section

### 1.4 test_cmc_consistency.py (1 test)
- **Line 533:** Test for imbalanced shards - v2.1.0 enforces balanced shards

---

## Category 2: Deprecated Checkpoint/Recovery APIs

### 2.1 test_failure_injection.py (4 tests)
- **Line 44:** Class-level skip on TestFailureInjection
- **Line 201:** `test_checkpoint_save_on_timeout` - deprecated APIs
- **Line 255:** `test_resume_from_checkpoint_after_crash` - deprecated APIs
- **Line 373:** `test_failed_shard_recovery` - deprecated recovery APIs
- **Line 502:** `test_catastrophic_failure_handling` - deprecated failure handling APIs

**Note:** Entire test class appears to be obsolete. Consider deleting entire file.

---

## Category 3: Unimplemented Features

### 3.1 test_jax_backend.py (2 tests)
- **Line 489:** `test_dispatcher_element_wise_path` - dispatcher feature not implemented
- **Line 548:** `test_dispatcher_meshgrid_path` - dispatcher feature not implemented

### 3.2 test_nlsq_saving.py (2 tests)
- **Line 526:** `test_plot_saving_integration_real_data` - Datashader backend incompatible with test
- **Line 630:** `test_plot_generation_with_datashader` - shape mismatch, TODO to fix

---

## Category 4: Missing Optional Dependencies

### 4.1 test_diagonal_correction.py (2 tests)
- **Line 488:** `test_diagonal_correction_benchmark_small` - requires pytest-benchmark plugin
- **Line 497:** `test_diagonal_correction_benchmark_large` - requires pytest-benchmark plugin

**Decision:** DELETE - pytest-benchmark not in pyproject.toml dependencies

---

## Category 5: Infeasible Test Approach

### 5.1 test_hardware_detection.py (1 class)
- **Line 68:** `TestDetectHardware` class - JAX import-time mocking not feasible

**Note:** Entire class (likely ~50-100 lines) can be deleted

---

## Deletion Strategy

1. **File-by-file approach:** Delete tests from each file systematically
2. **Verification:** Run `pytest --collect-only` after each file to ensure no syntax errors
3. **Priority order:**
   - Start with deprecated features (Categories 1 & 2)
   - Then unimplemented features (Category 3)
   - Then missing dependencies (Category 4)
   - Finally infeasible tests (Category 5)

4. **Special cases:**
   - **test_failure_injection.py**: Consider deleting entire file (all tests skipped)
   - **test_hardware_detection.py**: Delete entire TestDetectHardware class

---

## Expected Impact

- **Before:** 1,576 test functions
- **After:** ~1,560 test functions (-16)
- **Line reduction:** ~400-600 lines
- **Files modified:** 9 files
- **Files potentially deleted:** 1 file (test_failure_injection.py)

---

## Validation

After deletions:
1. Run `pytest tests/ --collect-only` to verify no collection errors
2. Run `make test` to ensure remaining tests still pass
3. Check git diff to review all deletions
4. Update TEST_CLEANUP_PROGRESS.md with results
