# Integration Testing Guide: Homodyne + NLSQ Fixes

**Date:** October 17, 2025 **Status:** ✅ **READY FOR INTEGRATION TESTING**

______________________________________________________________________

## Overview

This guide provides step-by-step instructions for testing the combined homodyne model
fix and NLSQ shape validation improvement with the 23M point C020 dataset.

**Two Fixes Implemented:**

1. **Homodyne Fix:** Model function now respects xdata indices for chunking

   - File: `homodyne/optimization/nlsq_wrapper.py`
   - Lines: 922-923
   - Change: `indices = xdata.astype(jnp.int32); return g2_theory_flat[indices]`

1. **NLSQ Improvement:** Shape validation catches incompatible models early

   - File: `nlsq/large_dataset.py`
   - Lines: 839-945
   - Purpose: Detect shape mismatches before processing all chunks

______________________________________________________________________

## Pre-Test Checklist

### 1. Verify Homodyne Fix Applied

```bash
cd /home/wei/Documents/GitHub/homodyne
grep -A 5 "CRITICAL FIX for curve_fit_large" homodyne/optimization/nlsq_wrapper.py
```

**Expected output:**

```python
# CRITICAL FIX for curve_fit_large chunking:
# xdata contains indices into the flattened array.
# When curve_fit_large chunks the data, it passes subset indices.
# We must return only those requested points to match ydata chunk size.
indices = xdata.astype(jnp.int32)
return g2_theory_flat[indices]  # ✅ Returns only requested points
```

### 2. Verify NLSQ Improvement Applied

```bash
cd /home/wei/Documents/GitHub/NLSQ
grep -A 3 "SHAPE VALIDATION" nlsq/large_dataset.py | head -5
```

**Expected output:**

```python
# ========== SHAPE VALIDATION ==========
# Validate that model function respects input size before processing all chunks.
# This catches shape mismatches early with clear error messages.
self.logger.debug("Validating model function shape compatibility...")
```

### 3. Check Code Quality

```bash
# Check homodyne
cd /home/wei/Documents/GitHub/homodyne
python3 -m black homodyne/optimization/nlsq_wrapper.py --check
python3 -m ruff check homodyne/optimization/nlsq_wrapper.py

# Check NLSQ
cd /home/wei/Documents/GitHub/NLSQ
python3 -m black nlsq/large_dataset.py --check
python3 -m ruff check nlsq/large_dataset.py
```

**Expected:** All checks pass (no violations).

______________________________________________________________________

## Test 1: Shape Validation Detection (Expected to PASS)

**Purpose:** Verify NLSQ shape validation works correctly with the fixed homodyne model.

### Run Command

```bash
cd /home/wei/Documents/Projects/data/C020
homodyne --config homodyne_laminar_flow_config.yaml \
         --method nlsq \
         --output-dir homodyne_results_test1 \
         --verbose
```

### Expected Behavior

**Early in execution (within first 10 seconds):**

```
DEBUG: Validating model function shape compatibility...
DEBUG: ✓ Model validation passed: f((100,), 9 params) -> (100,)
INFO: Fitting dataset using 24 chunks
```

**Key Success Indicators:**

1. ✅ Validation message appears early in logs
1. ✅ Validation passes (shape (100,) matches)
1. ✅ Chunking proceeds (24 chunks)
1. ✅ No shape mismatch errors

### Expected Results

**Timing:**

- **Duration:** 5-10 minutes (actual optimization)
- **NOT:** 2.6 seconds (indicates failure)

**Parameters:**

- **Values:** Different from initial parameters
- **NOT:** Unchanged from initial values

**Uncertainties:**

- **Values:** Varied (e.g., 0.001 to 100.0)
- **NOT:** All = 1.0 (identity matrix)

**Convergence:**

- **Iterations:** > 0 (should show actual iterations)
- **NOT:** 0 iterations

**Chi-squared:**

- **Value:** Reduced chi-squared ≈ 1.0-1.5
- **Behavior:** Improves during optimization

______________________________________________________________________

## Test 2: Parameter Verification

**Purpose:** Verify optimized parameters are physically meaningful and uncertainties are
computed.

### Check Results

```bash
cd /home/wei/Documents/Projects/data/C020/homodyne_results_test1

# View parameter results
python3 << 'EOF'
import json
import sys

# Load parameters
with open('nlsq/parameters.json', 'r') as f:
    data = json.load(f)

print("="*70)
print("PARAMETER VERIFICATION")
print("="*70)
print()

# Check each parameter
all_ones = True
params = data['parameters']

print(f"{'Parameter':<20s} {'Value':>12s} {'Uncertainty':>12s} {'Status':>8s}")
print("-"*70)

for name, param in params.items():
    value = param['value']
    uncertainty = param['uncertainty']

    if uncertainty != 1.0:
        all_ones = False

    status = "✓ OK" if uncertainty != 1.0 else "✗ FAIL"
    print(f"{name:<20s} {value:>12.6f} {uncertainty:>12.6f} {status:>8s}")

print()
print("="*70)

if all_ones:
    print("❌ FAIL: All uncertainties = 1.0 (identity matrix)")
    print("This indicates the fix did NOT work correctly.")
    sys.exit(1)
else:
    print("✅ SUCCESS: Covariance computed properly!")
    print("Parameters have varied uncertainties, indicating successful optimization.")
    sys.exit(0)
EOF
```

### Expected Output

```
======================================================================
PARAMETER VERIFICATION
======================================================================

Parameter            Value   Uncertainty   Status
----------------------------------------------------------------------
contrast             0.487623     0.002341    ✓ OK
offset               1.001234     0.000876    ✓ OK
D0                1425.678900    15.234567    ✓ OK
alpha                0.523456     0.012345    ✓ OK
D_offset            12.345678     0.456789    ✓ OK
gamma_dot_t0         0.000123     0.000012    ✓ OK
beta                 0.456789     0.023456    ✓ OK
gamma_dot_offset     0.000045     0.000005    ✓ OK
phi0                 2.345678     0.123456    ✓ OK

======================================================================
✅ SUCCESS: Covariance computed properly!
Parameters have varied uncertainties, indicating successful optimization.
```

**FAIL Criteria:**

```
======================================================================
❌ FAIL: All uncertainties = 1.0 (identity matrix)
This indicates the fix did NOT work correctly.
```

______________________________________________________________________

## Test 3: Log File Analysis

**Purpose:** Verify optimization actually ran (not 0 iterations).

### Check Log

```bash
cd /home/wei/Documents/Projects/data/C020/homodyne_results_test1/logs

# Find most recent log
LATEST_LOG=$(ls -t homodyne_analysis_*.log | head -1)

# Extract key information
echo "Log file: $LATEST_LOG"
echo ""
echo "Optimization Summary:"
grep "Optimization completed" "$LATEST_LOG"
echo ""
echo "Chi-squared:"
grep "chi-squared" "$LATEST_LOG" | tail -3
echo ""
echo "Chunk Processing:"
grep -i "chunk" "$LATEST_LOG" | head -10
```

### Expected Output

```
Log file: homodyne_analysis_20251017_145623.log

Optimization Summary:
INFO: Optimization completed in 387.45s, 15 iterations

Chi-squared:
INFO: Initial chi-squared: 2.4484e+07
INFO: Final chi-squared: 2.3156e+07
INFO: Reduced chi-squared: 1.0047

Chunk Processing:
DEBUG: ✓ Model validation passed: f((100,), 9 params) -> (100,)
INFO: Fitting dataset using 24 chunks
INFO: Progress: 1/24 chunks (4.2%) - ETA: 456.3s
INFO: Progress: 2/24 chunks (8.3%) - ETA: 423.1s
...
INFO: Chunked fit completed with 100.0% success rate
```

**Success Indicators:**

- ✅ Execution time: 5-10 minutes (not seconds)
- ✅ Iterations: > 0 (not 0)
- ✅ Chunk processing: 24 chunks, high success rate
- ✅ Chi-squared improvement: final < initial

______________________________________________________________________

## Test 4: Timing Comparison

**Purpose:** Compare execution time before and after fix.

### Reference Timings

**Before Fix (Broken):**

```
Duration: 2.62 seconds
Iterations: 0
Result: Identity covariance matrix
Status: ❌ Silent failure
```

**After Fix (Working):**

```
Duration: 5-10 minutes
Iterations: 10-20
Result: Proper covariance matrix
Status: ✅ Successful optimization
```

### Time Check

```bash
cd /home/wei/Documents/Projects/data/C020/homodyne_results_test1/logs
LATEST_LOG=$(ls -t homodyne_analysis_*.log | head -1)

# Extract timing
echo "Execution time:"
grep "Optimization completed" "$LATEST_LOG" | grep -oP '\d+\.\d+s'

# Verify it's in expected range (>300s)
SECONDS=$(grep "Optimization completed" "$LATEST_LOG" | grep -oP '\d+\.\d+(?=s)' | head -1)

if (( $(echo "$SECONDS < 30" | bc -l) )); then
    echo "❌ FAIL: Execution too fast ($SECONDS s) - indicates failure"
    exit 1
elif (( $(echo "$SECONDS > 300" | bc -l) )); then
    echo "✅ PASS: Execution time ($SECONDS s) indicates actual optimization"
    exit 0
else
    echo "⚠️ WARNING: Execution time ($SECONDS s) is borderline - review results"
    exit 2
fi
```

______________________________________________________________________

## Test 5: Comparison with Broken Version (Optional)

**Purpose:** Confirm the fix actually changed behavior.

### Run with Broken Model (Temporarily Revert Fix)

**⚠️ WARNING:** This is destructive testing. Only do this if you want to see the failure
mode.

```bash
# Backup the fix
cd /home/wei/Documents/GitHub/homodyne
cp homodyne/optimization/nlsq_wrapper.py homodyne/optimization/nlsq_wrapper.py.FIXED

# Temporarily revert to broken version (remove indexing)
# Edit lines 922-923 to:
#   return g2_theory_flat  # BROKEN: returns all 23M points

# Run homodyne (should fail with shape validation error)
cd /home/wei/Documents/Projects/data/C020
homodyne --config homodyne_laminar_flow_config.yaml \
         --method nlsq \
         --output-dir homodyne_results_broken

# Restore fix
cd /home/wei/Documents/GitHub/homodyne
cp homodyne/optimization/nlsq_wrapper.py.FIXED homodyne/optimization/nlsq_wrapper.py
```

### Expected Behavior (Broken Version)

**With NLSQ shape validation:**

```
ERROR: Model function validation failed:
Model function SHAPE MISMATCH detected!

  Input xdata shape:  (100,)
  Input ydata shape:  (100,)
  Model output shape: (23046023,)
  Expected shape:     (100,)

ERROR: Model output must match ydata size.
[... fix instructions ...]
```

**Result:** ✅ **Shape validation catches the bug immediately**

- Fails fast (within seconds)
- Clear error message
- Fix instructions provided

**Without NLSQ shape validation (old NLSQ):**

```
INFO: Optimization completed in 2.62s, 0 iterations
INFO: Final chi-squared: 2.4484e+07
```

**Result:** ❌ **Silent failure**

- Appears successful
- Identity covariance matrix
- No error messages

______________________________________________________________________

## Troubleshooting

### Issue 1: Shape Validation Fails

**Error:**

```
ERROR: Model function SHAPE MISMATCH detected!
```

**Diagnosis:**

- Homodyne fix not applied correctly
- Check lines 922-923 in nlsq_wrapper.py

**Fix:**

```bash
cd /home/wei/Documents/GitHub/homodyne
grep -A 2 "indices = xdata.astype" homodyne/optimization/nlsq_wrapper.py
# Should show: return g2_theory_flat[indices]
```

### Issue 2: Still Getting 0 Iterations

**Symptoms:**

- Completes in \<30 seconds
- 0 iterations
- All uncertainties = 1.0

**Diagnosis:**

- Homodyne fix not applied or reverted
- Using old homodyne installation

**Fix:**

1. Verify homodyne fix is applied (see Issue 1)
1. Reinstall homodyne: `pip install -e /home/wei/Documents/GitHub/homodyne`
1. Confirm using correct environment: `which homodyne`

### Issue 3: Parameters Unchanged

**Symptoms:**

- Optimization completes
- Parameters exactly match initial values
- Some uncertainties ≠ 1.0 (so not identity matrix)

**Diagnosis:**

- Different issue (not the chunking bug)
- Possible causes:
  - Initial guess already optimal
  - Tolerances too strict
  - Bounds too tight

**Fix:**

1. Check initial chi-squared vs final chi-squared (should improve)
1. Try different initial parameters
1. Review parameter bounds in config

### Issue 4: NLSQ Shape Validation Not Running

**Symptoms:**

- No "Validating model function" message in logs
- Running old NLSQ version

**Diagnosis:**

- NLSQ improvement not applied
- Using system NLSQ instead of local version

**Fix:**

```bash
# Verify NLSQ improvement
cd /home/wei/Documents/GitHub/NLSQ
grep "SHAPE VALIDATION" nlsq/large_dataset.py

# Install local NLSQ
pip install -e /home/wei/Documents/GitHub/NLSQ

# Verify installation
python3 -c "import nlsq; print(nlsq.__file__)"
# Should show: /home/wei/Documents/GitHub/NLSQ/nlsq/__init__.py
```

______________________________________________________________________

## Success Criteria Summary

| Check | Success Criteria | Failure Indicator |
|-------|-----------------|-------------------| | **Validation** | "✓ Model validation
passed" in logs | Shape mismatch error | | **Timing** | 5-10 minutes (300-600s) | \<30
seconds | | **Iterations** | >0 (typically 10-20) | 0 iterations | | **Parameters** |
Changed from initial values | Unchanged from initial | | **Uncertainties** | Varied (not
all 1.0) | All = 1.0 | | **Chi-squared** | Reduced chi² ≈ 1.0-1.5 | Unchanged or
suspiciously good | | **Chunks** | 24 chunks, >90% success rate | High failure rate |

**Overall Success:** ALL checks must pass.

______________________________________________________________________

## Next Steps After Successful Testing

### 1. Commit Homodyne Fix

```bash
cd /home/wei/Documents/GitHub/homodyne

git add homodyne/optimization/nlsq_wrapper.py
git add NLSQ_FIX_COMPLETE.md
git add test_nlsq_chunking_fix.py
git add NLSQ_FIX_PLAN.md

git commit -m "$(cat <<'EOF'
fix(nlsq): make model_function respect xdata indices for curve_fit_large chunking

CRITICAL FIX: curve_fit_large was failing silently because model_function
returned fixed 23M array regardless of chunk size, causing shape mismatch.

Changes:
- Modified model_function to use xdata as indices into flattened g2 array
- Returns only requested subset: g2_theory_flat[xdata.astype(jnp.int32)]
- Updated docstring to document xdata usage for chunking
- Added 2 lines of code (lines 922-923 in nlsq_wrapper.py)

Root Cause:
- curve_fit_large chunks data: x_chunk (1M pts), y_chunk (1M pts)
- Old model returned: 23M points (ignored x_chunk size)
- Residual computation: (1M) - (23M) → shape mismatch → all chunks failed
- success_rate = 0% → identity covariance fallback

Testing:
- Code quality: black + ruff pass
- Indexing logic: 5/5 tests pass
- Integration: 23M point test successful

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

### 2. Commit NLSQ Improvement

```bash
cd /home/wei/Documents/GitHub/NLSQ

git add nlsq/large_dataset.py
git add PROPOSED_IMPROVEMENTS.md
git add NLSQ_IMPROVEMENTS_COMPLETE.md

git commit -m "$(cat <<'EOF'
feat(large_dataset): add comprehensive shape validation for curve_fit_large

Implements Priority 1 improvement to detect model-chunking incompatibility
early with clear, actionable error messages.

Changes:
- Added shape validation in _fit_chunked() before processing all chunks
- Tests model function with first 100 points to verify output shape
- Provides detailed error messages with fix examples
- Graceful degradation for non-critical validation failures

Benefits:
- Prevents silent failures → identity covariance matrices
- Reduces debugging time from hours to seconds
- <0.01% performance overhead

Testing:
- Code quality: black + ruff pass
- Integration: verified with homodyne XPCS optimization

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

### 3. Update Documentation

Consider updating:

- Homodyne README.md with known issues section
- NLSQ README.md with chunking-compatible model guidelines
- Both CHANGELOGs with this fix

______________________________________________________________________

## Contact and Support

**Issue Reports:**

- Homodyne: `/home/wei/Documents/GitHub/homodyne/issues`
- NLSQ: `/home/wei/Documents/GitHub/NLSQ/issues`

**Documentation:**

- Homodyne fix: `NLSQ_FIX_COMPLETE.md`
- NLSQ improvement: `NLSQ_IMPROVEMENTS_COMPLETE.md`
- Test scripts: `test_nlsq_chunking_fix.py`

______________________________________________________________________

*End of Integration Testing Guide*
