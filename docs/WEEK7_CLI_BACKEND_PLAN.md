# Week 7: CLI & Backend Consolidation Plan

**Date**: 2025-11-15
**Target**: 8 files → 4 files (176 tests)
**Status**: Planning

---

## Overview

Week 7 consolidates CLI argument handling and backend infrastructure tests into logical groups. This completes the core test consolidation effort (Weeks 2-7), achieving significant file reduction while preserving all tests.

---

## File Analysis

### Current State (8 files, 176 tests)

**CLI Tests** (5 files, 124 tests):
1. `test_cli_args.py` (17 tests, 273 lines)
   - CLI argument parsing
   - Argument validation and defaults

2. `test_cli_data_loading.py` (14 tests, 345 lines)
   - Data loading via CLI
   - File path handling and validation

3. `test_cli_integration.py` (22 tests, 570 lines)
   - End-to-end CLI integration
   - Command workflows

4. `test_cli_overrides.py` (25 tests, 744 lines)
   - Config overrides via CLI
   - Parameter override logic

5. `test_cli_validation.py` (46 tests, 881 lines)
   - CLI input validation
   - Error handling and edge cases

**Backend Tests** (3 files, 52 tests):
6. `test_backend_implementations.py` (22 tests, 772 lines)
   - Backend implementation tests
   - Different backend modes

7. `test_backend_infrastructure.py` (15 tests, 494 lines)
   - Backend infrastructure
   - Setup and teardown

8. `test_coordinator.py` (15 tests, 601 lines)
   - Coordinator pattern tests
   - Multi-backend coordination

---

## Consolidation Strategy

### Phase 1: CLI Core (3 → 1 file, 83 tests)
**Target**: `tests/unit/test_cli_core.py`

**Merge**:
- `test_cli_args.py` (17 tests)
- `test_cli_validation.py` (46 tests)
- `test_cli_data_loading.py` (14 tests)
- `test_cli_integration.py` (6 tests, pending check)

**Rationale**: Core CLI functionality - argument parsing, validation, and data loading. These are fundamental CLI operations that belong together.

### Phase 2: CLI Workflows (2 → 1 file, 41 tests)
**Target**: `tests/unit/test_cli_workflows.py`

**Merge**:
- `test_cli_integration.py` (remaining tests after Phase 1 split)
- `test_cli_overrides.py` (25 tests)

**Rationale**: Higher-level CLI workflows including integration tests and config override logic.

### Phase 3: Backend Infrastructure (3 → 1 file, 52 tests)
**Target**: `tests/unit/test_backend_core.py`

**Merge**:
- `test_backend_implementations.py` (22 tests)
- `test_backend_infrastructure.py` (15 tests)
- `test_coordinator.py` (15 tests)

**Rationale**: All backend-related tests consolidated - implementations, infrastructure, and coordination logic.

### Phase 4: Keep As-Is (1 file)
**Keep**: `tests/integration/test_cli_config_integration.py` (7 tests)

**Rationale**: Integration test, kept separate in Week 6.

---

## Final Structure

```
tests/
├── unit/
│   ├── test_cli_core.py (NEW, ~83 tests)
│   ├── test_cli_workflows.py (NEW, ~41 tests)
│   └── test_backend_core.py (NEW, 52 tests)
└── integration/
    └── test_cli_config_integration.py (KEPT, 7 tests)
```

**Summary**: 8 files → 4 files (50% reduction), 176 tests preserved

---

## Execution Plan

### Phase 1: CLI Core
1. Analyze test_cli_integration.py to determine split (core vs workflow tests)
2. Create consolidation script at `/tmp/consolidate_cli_core.py`
3. Merge cli_args + cli_validation + cli_data_loading + (cli_integration core tests)
4. Verify syntax and test count
5. Delete source files (partial for cli_integration if split)

### Phase 2: CLI Workflows
1. Create consolidation script at `/tmp/consolidate_cli_workflows.py`
2. Merge cli_integration (workflow tests) + cli_overrides
3. Verify syntax and test count
4. Delete source files

### Phase 3: Backend Infrastructure
1. Create consolidation script at `/tmp/consolidate_backend_core.py`
2. Merge backend_implementations + backend_infrastructure + coordinator
3. Verify syntax and test count
4. Delete source files

### Phase 4: Verification
1. Verify consolidated files: 83 + 41 + 52 = 176 tests
2. Verify total suite: `pytest tests/ --collect-only -q` (expect 1,567 tests)
3. Commit: `test: consolidate CLI & backend tests (Week 7 complete)`

---

## Risk Assessment

**Medium Risk**:
- Large file consolidation (8 files, 4,680 lines)
- test_cli_integration.py may need splitting between phases
- Complex import dependencies in CLI tests

**Potential Issues**:
- Import variations between CLI test files
- test_cli_integration.py classification (core vs workflow)
- Backend test fixture dependencies
- Mock/patch conflicts when merging

**Mitigation**:
- Careful analysis of test_cli_integration.py before splitting
- Verify all imports are compatible
- Test collection after each phase
- Check for duplicate test names across files

---

## Estimated Effort

- **Phase 1**: 25-30 minutes (CLI core, includes integration analysis)
- **Phase 2**: 15-20 minutes (CLI workflows)
- **Phase 3**: 15-20 minutes (Backend core)
- **Phase 4**: 10 minutes (verification and commit)
- **Total**: 65-80 minutes

---

## Success Criteria

1. ✅ All 176 tests preserved (zero loss)
2. ✅ Syntax valid for all consolidated files
3. ✅ Total suite unchanged (1,567 tests)
4. ✅ Clear section markers identifying source files
5. ✅ Conventional commit with detailed breakdown
6. ✅ Professional documentation (this plan + summary)

---

## Weeks 2-7 Target Metrics

| Metric | Before | After (projected) | Change |
|--------|--------|-------------------|--------|
| **Total files** | 56 | 22 | **-34 files (61% reduction)** |
| **Tests preserved** | 878+ | 878+ | **Zero loss** |
| **Total suite** | 1,567 | 1,567 | **No regressions** |

Week 7 specifically:
- **Files**: 8 → 4 (50% reduction)
- **Tests**: 176 preserved (100%)
- **Lines**: 4,680 → ~4,000 (-15%)

---

**Next**: Analyze test_cli_integration.py structure to determine core vs workflow split
