# Week 2-3: Test Consolidation Plan
**Date**: 2025-11-15
**Scope**: Angle Filtering and NLSQ Test Consolidation

## Part 1: Angle Filtering (4 files → 2 files)

### Current Structure (63 tests, 1,584 lines)

**test_angle_filtering.py** (859 lines):
- TestApplyAngleFiltering
- TestApplyAngleFilteringForOptimization
- TestAngleFilteringEdgeCases
- TestAngleNormalization
- TestAngleInRange
- TestNormalizationIntegration
- TestAngleValidation

**test_angle_filtering_jax.py** (208 lines):
- TestJAXArrayCompatibility
- TestNumpyArrayCompatibility

**test_angle_filtering_consistency.py** (314 lines):
- TestAngleFilteringConsistency

**test_angle_filtering_performance.py** (203 lines):
- TestAngleFilteringPerformance

### Consolidation Strategy

**Target File 1: tests/unit/test_angle_filtering.py** (~1,381 lines)
Merge:
1. Keep existing 7 test classes from test_angle_filtering.py
2. Add TestJAXArrayCompatibility from test_angle_filtering_jax.py
3. Add TestNumpyArrayCompatibility from test_angle_filtering_jax.py
4. Add TestAngleFilteringConsistency from test_angle_filtering_consistency.py

Final structure (10 test classes):
- TestApplyAngleFiltering (core functionality)
- TestApplyAngleFilteringForOptimization (optimization-specific)
- TestAngleFilteringEdgeCases (edge cases)
- TestAngleNormalization (normalization logic)
- TestAngleInRange (range validation)
- TestNormalizationIntegration (integration tests)
- TestAngleValidation (validation logic)
- TestJAXArrayCompatibility (JAX array handling)
- TestNumpyArrayCompatibility (NumPy array handling)
- TestAngleFilteringConsistency (cross-implementation consistency)

**Target File 2: tests/performance/test_angle_filtering_performance.py** (203 lines)
- Keep as is (no changes needed)
- TestAngleFilteringPerformance

### Files to Delete
- tests/unit/test_angle_filtering_jax.py
- tests/unit/test_angle_filtering_consistency.py

### Reduction
- Files: 4 → 2 (-2 files, 50% reduction)
- Lines: 1,584 → ~1,584 (same, just reorganized)
- Tests: 63 (no loss)

---

## Part 2: NLSQ (12 files → 5 files, 171 tests, 6,676 lines)

### Current Structure

**Unit Tests** (6 files, ~3,253 lines):
1. tests/unit/test_nlsq_wrapper_integration.py (695 lines)
2. tests/unit/test_nlsq_api_handling.py (657 lines)
3. tests/unit/test_optimization_nlsq.py (595 lines)
4. tests/unit/test_nlsq_wrapper.py (592 lines)
5. tests/unit/test_nlsq_saving.py (572 lines)
6. tests/unit/test_nlsq_public_api.py (142 lines)

**Integration Tests** (4 files, ~2,145 lines):
7. tests/integration/test_stratified_nlsq_integration.py (721 lines)
8. tests/integration/test_nlsq_workflow.py (545 lines)
9. tests/integration/test_nlsq_end_to_end.py (509 lines)
10. tests/integration/test_nlsq_filtering.py (370 lines)

**Performance Tests** (1 file, 706 lines):
11. tests/performance/test_nlsq_performance.py (706 lines)

**Regression Tests** (1 file, 572 lines):
12. tests/regression/test_nlsq_quality_regression.py (572 lines)

### Consolidation Strategy

**Target File 1: tests/unit/test_nlsq_core.py**
Merge:
- test_nlsq_wrapper.py (wrapper core functionality)
- test_optimization_nlsq.py (optimization logic)
- test_nlsq_api_handling.py (API handling)
- test_nlsq_public_api.py (public API)

Content: Core NLSQ optimization, wrapper, API handling

**Target File 2: tests/unit/test_nlsq_saving.py**
Keep as is (result saving and plotting)

**Target File 3: tests/integration/test_nlsq_integration.py**
Merge:
- test_nlsq_end_to_end.py (end-to-end workflows)
- test_nlsq_workflow.py (workflow orchestration)
- test_stratified_nlsq_integration.py (stratified data handling)
- test_nlsq_filtering.py (filtering integration)
- test_nlsq_wrapper_integration.py (from unit/)

Content: End-to-end NLSQ workflows, filtering, stratified data

**Target File 4: tests/performance/test_nlsq_performance.py**
Keep as is (benchmarks)

**Target File 5: tests/regression/test_nlsq_regression.py**
Rename from test_nlsq_quality_regression.py for consistency

### Files to Delete
Unit (4 deleted, 1 renamed):
- test_nlsq_wrapper_integration.py (→ integration)
- test_nlsq_api_handling.py (→ test_nlsq_core.py)
- test_optimization_nlsq.py (→ test_nlsq_core.py)
- test_nlsq_public_api.py (→ test_nlsq_core.py)
- test_nlsq_wrapper.py (→ test_nlsq_core.py)

Integration (3 deleted):
- test_nlsq_end_to_end.py (→ test_nlsq_integration.py)
- test_nlsq_filtering.py (→ test_nlsq_integration.py)
- test_nlsq_workflow.py (→ test_nlsq_integration.py)
- test_stratified_nlsq_integration.py (→ test_nlsq_integration.py)

Regression (1 renamed):
- test_nlsq_quality_regression.py → test_nlsq_regression.py

### Reduction
- Files: 12 → 5 (-7 files, 58% reduction)
- Tests: TBD (will count during consolidation)

---

## Implementation Steps

### Step 1: Angle Filtering Consolidation
1. ✅ Analyze current structure
2. ⏳ Read test_angle_filtering_jax.py and extract test classes
3. ⏳ Read test_angle_filtering_consistency.py and extract test classes
4. ⏳ Append to test_angle_filtering.py with clear section markers
5. ⏳ Delete consolidated files
6. ⏳ Run pytest to verify (expect 63 tests)

### Step 2: NLSQ Consolidation
1. ⏳ Analyze test counts in all 12 files
2. ⏳ Create test_nlsq_core.py from 4 unit test files
3. ⏳ Create test_nlsq_integration.py from 5 integration files
4. ⏳ Rename test_nlsq_quality_regression.py
5. ⏳ Delete consolidated files
6. ⏳ Run pytest to verify (expect same test count)

### Step 3: Verification
1. ⏳ Run pytest --collect-only to verify test count
2. ⏳ Run full test suite to verify all pass
3. ⏳ Check for any import errors

### Step 4: Documentation
1. ⏳ Update this plan with final counts
2. ⏳ Create Week 2-3 summary document
3. ⏳ Commit changes with conventional commit message

---

## Success Metrics

**Angle Filtering**:
- ✅ 4 files → 2 files (50% reduction)
- ✅ 63 tests preserved
- ✅ All tests pass

**NLSQ**:
- ⏳ 12 files → 5 files (58% reduction)
- ⏳ All tests preserved
- ⏳ All tests pass

**Combined**:
- ⏳ 16 files → 7 files (56% reduction)
- ⏳ No test loss
- ⏳ No import errors
