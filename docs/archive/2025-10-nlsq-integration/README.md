# NLSQ Integration Troubleshooting Archive (October 2025)

**Status**: RESOLVED - All fixes implemented as of Oct 17, 2025

## Summary
This archive contains diagnostic files from NLSQ curve_fit_large integration debugging session on October 17, 2025. All issues have been fixed, validated, and integrated into the main codebase. Files are preserved for historical reference and future troubleshooting context.

## Issues Resolved

### 1. curve_fit_large API Unpacking Error
- **File**: `NLSQ_UNPACKING_ERROR_FIX.md`
- **Issue**: ValueError - not enough values to unpack (expected 3, got 2)
- **Root Cause**: curve_fit_large() returns only (popt, pcov), not (popt, pcov, info)
- **Fix**: Changed unpacking to 2 values, create empty info dict for consistency
- **Code**: `homodyne/optimization/nlsq_wrapper.py` lines 397-414

### 2. Model Function Chunking Incompatibility
- **File**: `NLSQ_FIX_PLAN.md`
- **Issue**: Model function returned fixed 23M array regardless of chunk size
- **Root Cause**: Ignored xdata indices → shape mismatch in residual computation
- **Fix**: Added indexing `g2_theory_flat[xdata.astype(jnp.int32)]`
- **Code**: `homodyne/optimization/nlsq_wrapper.py` lines 922-923

### 3. Integration Testing Validation
- **File**: `INTEGRATION_TEST_GUIDE.md`
- **Purpose**: Comprehensive validation checklist for combined homodyne + NLSQ fixes
- **Result**: All validation tests passed, 23M point dataset optimization working
- **Tests**: 72/72 angle filtering tests passing, NLSQ integration validated

## Quick Links to Current Documentation

- **CLAUDE.md**: Current project status and comprehensive documentation
- **Known Issues**: See `docs/troubleshooting/` for reusable guides
- **Git commits**: Search for "Oct 17, 2025" in git log for implementation details
- **Tests**: `tests/unit/test_nlsq_*.py`, `tests/integration/test_nlsq_*.py`

## Files in This Archive

1. **INTEGRATION_TEST_GUIDE.md** (584 lines)
   - Step-by-step testing guide for NLSQ fixes
   - Validation checklist and success criteria
   - Troubleshooting procedures
   - Expected vs actual behavior comparisons

2. **NLSQ_FIX_PLAN.md** (326 lines)
   - Root cause analysis of chunking bug
   - Implementation plan and code changes
   - Testing plan and validation checklist
   - Memory considerations and alternatives

3. **NLSQ_UNPACKING_ERROR_FIX.md** (462 lines)
   - API incompatibility diagnosis
   - Unpacking error fix with code diffs
   - Testing and verification procedures
   - API comparison (curve_fit vs curve_fit_large)

## Why These Files Were Archived (Not Deleted)

These files contain:
- **Unique diagnostic methodology** for future similar issues
- **Complete context** of the debugging session
- **Integration testing approach** that may be reusable
- **Architecture decision rationale** for the fixes

While the fixes are fully implemented and the issues resolved, these files preserve the "why" and "how" of the troubleshooting process for future reference.

## Archive Retention Policy

- **Review date**: April 2026 (6 months from creation)
- **Retention criteria**: If no recurring issues and fixes remain stable, eligible for deletion
- **Preservation priority**: MEDIUM - useful historical context but not critical for daily operations

## Related Documentation

### Permanent Documentation Created from This Episode
- `docs/troubleshooting/silent-failure-diagnosis.md` - Extracted from NLSQ_0_ITERATIONS_DIAGNOSIS.md
- `docs/troubleshooting/imshow-transpose-pitfalls.md` - Extracted from PLOTTING_TRANSPOSE_FIX.md
- CLAUDE.md "Known Issues" section - Summary of NLSQ quirks and gotchas

### Files Not Archived (Deleted as Redundant)
- **NLSQ_FIXES_APPLIED.md** - Fully covered in CLAUDE.md "Recent Major Changes"
- **NLSQ_FIX_COMPLETE.md** - Completion status, no longer needed

### Files Still in Root (Will be cleaned in Phase 3)
- NLSQ_0_ITERATIONS_DIAGNOSIS.md → Will extract to troubleshooting guide
- NLSQ_VALIDATION_REPORT.md → Partially documented in CLAUDE.md
- PLOTTING_TRANSPOSE_FIX.md → Will extract to troubleshooting guide

## For Future Developers

If you encounter similar NLSQ integration issues:

1. **Check current documentation first**: `docs/troubleshooting/` and CLAUDE.md
2. **Review this archive** if issues seem familiar but not documented elsewhere
3. **Search git history**: `git log --all --grep="NLSQ" --since="2025-10-01"`
4. **Run existing tests**: `make test-nlsq` to verify current implementation
5. **Consult NLSQ docs**: https://nlsq.readthedocs.io/en/latest/

---

**Archive Created**: November 17, 2025
**Completion of**: October 17, 2025 NLSQ integration debugging
**Next Review**: April 2026
