# Week 6: Config & Infrastructure Consolidation Plan

**Date**: 2025-11-15
**Target**: 5 files → 3 files (101 tests)
**Status**: Planning

---

## Overview

Week 6 consolidates configuration validation and infrastructure tests into logical groups. Many config tests were already consolidated in Weeks 4-5 (parameter config, ConfigManager integration), leaving a focused set of infrastructure tests.

---

## File Analysis

### Current State (5 files, 101 tests)

**Config Validation Tests** (2 files, 44 tests):
1. `test_device_config.py` (10 tests, 307 lines)
   - Hardware detection and CMC/NUTS selection logic
   - Dual-criteria OR logic: `(num_samples >= 15) OR (memory > 30%)`
   - Platform-specific configurations

2. `test_edge_cases_config_validation.py` (34 tests, 513 lines)
   - Malformed config file handling
   - Parameter value validation edge cases
   - Threshold validation and extreme values
   - Invalid data types and boundary conditions

**Checkpoint Management Tests** (2 files, 50 tests):
3. `test_checkpoint_manager.py` (22 tests, 534 lines)
   - Checkpoint save/load with HDF5
   - Resume from partial optimization
   - Corruption detection
   - Automatic cleanup

4. `test_checkpoint_manager_coverage.py` (28 tests, 619 lines)
   - Extended checkpoint coverage
   - Edge cases in checkpoint handling
   - Checkpoint validation scenarios

**Integration Test** (1 file, 7 tests):
5. `test_cli_config_integration.py` (7 tests, 395 lines)
   - CLI-config integration workflow
   - End-to-end config loading via CLI

---

## Consolidation Strategy

### Phase 1: Config Validation (2 → 1 file, 44 tests)
**Target**: `tests/unit/test_config_validation.py`

**Merge**:
- `test_device_config.py` (10 tests)
- `test_edge_cases_config_validation.py` (34 tests)

**Rationale**: Both test configuration validation - one focuses on hardware/device config, the other on parameter validation edge cases. Natural grouping.

### Phase 2: Checkpoint Management (2 → 1 file, 50 tests)
**Target**: `tests/unit/test_checkpoint_core.py`

**Merge**:
- `test_checkpoint_manager.py` (22 tests)
- `test_checkpoint_manager_coverage.py` (28 tests)

**Rationale**: Both test CheckpointManager class. Coverage file extends base functionality. Single consolidated file reduces redundancy.

### Phase 3: Keep As-Is (1 file, 7 tests)
**Keep**: `tests/integration/test_cli_config_integration.py`

**Rationale**: Integration test, small file, separate concern (CLI-config workflow).

---

## Final Structure

```
tests/
├── unit/
│   ├── test_config_validation.py (NEW, 44 tests)
│   └── test_checkpoint_core.py (NEW, 50 tests)
└── integration/
    └── test_cli_config_integration.py (KEPT, 7 tests)
```

**Summary**: 5 files → 3 files (40% reduction), 101 tests preserved

---

## Execution Plan

### Phase 1: Config Validation
1. Create consolidation script at `/tmp/consolidate_config_validation.py`
2. Merge device_config + edge_cases_config_validation
3. Verify syntax: `python -m py_compile tests/unit/test_config_validation.py`
4. Verify tests: `pytest tests/unit/test_config_validation.py --collect-only -q` (expect 44)
5. Delete source files

### Phase 2: Checkpoint Management
1. Create consolidation script at `/tmp/consolidate_checkpoint_core.py`
2. Merge checkpoint_manager + checkpoint_manager_coverage
3. Verify syntax: `python -m py_compile tests/unit/test_checkpoint_core.py`
4. Verify tests: `pytest tests/unit/test_checkpoint_core.py --collect-only -q` (expect 50)
5. Delete source files

### Phase 3: Verification
1. Verify consolidated files: 44 + 50 + 7 = 101 tests
2. Verify total suite: `pytest tests/ --collect-only -q` (expect 1,567 tests, no regression)
3. Commit: `test: consolidate config & infrastructure tests (Week 6 complete)`

---

## Risk Assessment

**Low Risk**:
- Small file count (5 files)
- Clear separation of concerns
- Established consolidation pattern from Weeks 4-5

**Potential Issues**:
- Import variations between files (need careful header construction)
- Checkpoint tests may have complex fixture dependencies
- Device config tests may have platform-specific logic

**Mitigation**:
- Careful analysis of imports before consolidation
- Verify fixture compatibility
- Test collection after each phase

---

## Estimated Effort

- **Phase 1**: 15-20 minutes (config validation)
- **Phase 2**: 15-20 minutes (checkpoint management)
- **Phase 3**: 10 minutes (verification and commit)
- **Total**: 40-50 minutes

---

## Success Criteria

1. ✅ All 101 tests preserved (zero loss)
2. ✅ Syntax valid for all consolidated files
3. ✅ Total suite unchanged (1,567 tests)
4. ✅ Clear section markers identifying source files
5. ✅ Conventional commit with detailed breakdown
6. ✅ Professional documentation (this plan + summary)

---

## Notes

- Week 6 is smaller than originally estimated (~8 files) because many config tests were consolidated in Weeks 4-5
- Checkpoint tests are infrastructure-related, not purely "config", but natural fit for this week
- CLI-config integration kept separate as it's an end-to-end workflow test
- Weeks 2-5 reduced 43 → 15 files (65% reduction); Week 6 will achieve 48 → 18 files (62.5% reduction overall)

---

**Next**: Execute Phase 1 (Config Validation consolidation)
