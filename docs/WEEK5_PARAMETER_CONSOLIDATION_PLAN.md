# Week 5: Parameter Tests Consolidation Plan
**Date**: 2025-11-15
**Target**: 11 files → 4 files
**Total Tests**: 201 tests (must preserve all)

## Current State Analysis

### Parameter Test Files (11 files, 4,248 lines, 201 tests)

**Unit Tests** (10 files, 3,987 lines, 199 tests):
1. `test_parameter_manager.py` (515 lines, 38 tests) - Core parameter manager
2. `test_parameter_manager_advanced.py` (275 lines, 17 tests) - Advanced features
3. `test_parameter_manager_physics.py` (471 lines, 32 tests) - Physics-specific
4. `test_config_initial_params.py` (591 lines, 26 tests) - Initial parameter config
5. `test_config_manager_parameters.py` (246 lines, 8 tests) - Config manager integration
6. `test_parameter_space_config.py` (789 lines, 35 tests) - Parameter space config
7. `test_parameter_expansion.py` (317 lines, 11 tests) - Parameter expansion
8. `test_parameter_transformation.py` (147 lines, 6 tests) - Transformations
9. `test_parameter_gradients.py` (315 lines, 4 tests) - Gradient computations
10. `test_parameter_names_consistency.py` (321 lines, 22 tests) - Name consistency

**Integration Tests** (1 file, 261 lines, 2 tests):
11. `test_parameter_recovery.py` (261 lines, 2 tests) - Parameter recovery validation

---

## Proposed Consolidation (11 → 4 files)

### Group 1: Parameter Manager Core (3 files → 1 file)
**Target**: `tests/unit/test_parameter_manager_core.py` (87 tests)

**Merge**:
- test_parameter_manager.py (38 tests) - Core functionality
- test_parameter_manager_advanced.py (17 tests) - Advanced features
- test_parameter_manager_physics.py (32 tests) - Physics-specific

**Rationale**: All ParameterManager class tests in one location

### Group 2: Parameter Configuration (3 files → 1 file)
**Target**: `tests/unit/test_parameter_config.py` (69 tests)

**Merge**:
- test_config_initial_params.py (26 tests) - Initial params
- test_config_manager_parameters.py (8 tests) - Config manager integration
- test_parameter_space_config.py (35 tests) - Parameter space config

**Rationale**: All configuration-related parameter tests

### Group 3: Parameter Operations (4 files → 1 file)
**Target**: `tests/unit/test_parameter_operations.py` (43 tests)

**Merge**:
- test_parameter_expansion.py (11 tests) - Expansion logic
- test_parameter_transformation.py (6 tests) - Transformations
- test_parameter_gradients.py (4 tests) - Gradient calculations
- test_parameter_names_consistency.py (22 tests) - Name consistency

**Rationale**: Parameter manipulation and validation operations

### Group 4: Parameter Recovery (keep as-is)
**Target**: `tests/integration/test_parameter_recovery.py` (2 tests)

**Keep as-is**: Integration test, already focused

**Rationale**: Small integration test, no consolidation needed

---

## Expected Metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Files | 11 | 4 | **-7 files (64% reduction)** |
| Lines | 4,248 | ~3,600 | -15% (estimated) |
| Tests | 201 | 201 | **No loss** |

---

## Execution Plan

**Phase 1**: Parameter Manager Core
1. Create test_parameter_manager_core.py
2. Merge 3 files (87 tests)
3. Verify syntax and test count
4. Delete source files

**Phase 2**: Parameter Configuration
1. Create test_parameter_config.py
2. Merge 3 files (69 tests)
3. Verify syntax and test count
4. Delete source files

**Phase 3**: Parameter Operations
1. Create test_parameter_operations.py
2. Merge 4 files (43 tests)
3. Verify syntax and test count
4. Delete source files

**Phase 4**: Verification & Commit
1. Verify total test suite (1,567 tests)
2. Create summary document
3. Commit with conventional message

---

## Risk Assessment

**Low Risk**: All groups have clear boundaries and similar imports

**Mitigation**: Same approach as Week 4 (Python scripts, section markers, careful line skipping)

---

**Estimated Time**: 1.5-2 hours
**Next**: Execute Phase 1
