# Matplotlib imshow() Transpose Pitfalls for Two-Time Correlation

**Problem**: Two-time correlation C2(t1, t2) displays with wrong diagonal orientation

This guide explains a common pitfall when plotting two-time correlation matrices with
matplotlib's `imshow()` and how to fix it.

## The Problem

### Symptom

Yellow correlation strip (diagonal) oriented **incorrectly**:

- **Wrong**: Diagonal from top-left to bottom-right
- **Correct**: Diagonal from bottom-left to top-right

### Visual Comparison

```
❌ WRONG (without transpose):          ✅ CORRECT (with transpose):
t2 ↑                                   t2 ↑
   │ ╲                                    │   ╱
   │  ╲  Low correlation                 │  ╱  High correlation
   │   ╲                                  │ ╱
   │    ╲                                 │╱
   └─────→ t1                             └─────→ t1
   Backward orientation                   Correct physics!
```

## Root Cause

### Data Structure

```python
# HDF5 storage and memory layout:
c2_exp.shape = (n_phi, n_t1, n_t2)  # e.g., (23, 1001, 1001)

# Time grid creation:
t1_2d, t2_2d = np.meshgrid(time_1d, time_1d, indexing="ij")
```

With `indexing="ij"` (matrix indexing):

- `t1_2d[i, j] = time_1d[i]` → rows represent t1 index
- `t2_2d[i, j] = time_1d[j]` → columns represent t2 index
- **Data**: `c2[i, j]` = correlation at (t1[i], t2[j])

### matplotlib imshow() Behavior

**Key Insight**: `imshow(array)` interprets `array[row, col]` as pixel at position
**(x=col, y=row)**

For our data structure `c2[t1_idx, t2_idx]`:

**Without transpose**:

```python
ax.imshow(c2, origin='lower')
# array[row, col] → c2[t1_idx, t2_idx]
# Display: x-axis = col = t2_idx, y-axis = row = t1_idx
# Result: x=t2, y=t1 (BACKWARDS!)
```

**With transpose**:

```python
ax.imshow(c2.T, origin='lower')
# array[row, col] → c2.T[t2_idx, t1_idx] = c2[t1_idx, t2_idx]
# Display: x-axis = col = t1_idx, y-axis = row = t2_idx
# Result: x=t1, y=t2 (CORRECT!)
```

## Solution

### Always Use `.T` Transpose

```python
# ❌ WRONG: Diagonal from top-left to bottom-right
ax.imshow(
    c2,  # Missing transpose!
    aspect='equal',
    cmap='viridis',
    origin='lower',
    extent=[t1[0], t1[-1], t2[0], t2[-1]]
)

# ✅ CORRECT: Diagonal from bottom-left to top-right
ax.imshow(
    c2.T,  # Transpose applied
    aspect='equal',
    cmap='viridis',
    origin='lower',
    extent=[t1[0], t1[-1], t2[0], t2[-1]]
)
ax.set_xlabel('t₁ (s)')  # x-axis = first time
ax.set_ylabel('t₂ (s)')  # y-axis = second time
```

### Why Extent Must Match Transpose

The `extent` parameter maps array indices to real coordinates:

```python
extent = [x_min, x_max, y_min, y_max]
```

With `.T` transpose:

- **x-range** `[t1[0], t1[-1]]` maps to **columns** (which are t1 values after
  transpose)
- **y-range** `[t2[0], t2[-1]]` maps to **rows** (which are t2 values after transpose)
- Perfect alignment! ✓

## Physical Validation

### Expected Behavior for C2(t1, t2)

**Physics**: Two-time correlation function shows how system evolves between times t1 and
t2

**Key features**:

- **Maximum correlation** when t1 ≈ t2 (diagonal line)
- **Decorrelation** as |t2 - t1| increases (away from diagonal)
- **Visual signature**: Bright yellow/green strip from (0,0) to (tmax, tmax)

**Coordinate system**:

- **x-axis (horizontal)**: t1 (first measurement time)
- **y-axis (vertical)**: t2 (second measurement time)
- **Diagonal**: t1 = t2 (simultaneous measurements → perfect correlation)

With `origin='lower'`:

- Point (0,0) at **bottom-left** corner
- Diagonal extends to **top-right** corner ✓
- Matches standard mathematical convention

### Validation Checklist

After applying transpose, verify:

- [ ] Diagonal stripe from **bottom-left to top-right** ✓
- [ ] x-axis labeled **"t₁ (s)"** or similar ✓
- [ ] y-axis labeled **"t₂ (s)"** or similar ✓
- [ ] Bright yellow/green along diagonal (high correlation) ✓
- [ ] Darker colors off-diagonal (decorrelation) ✓
- [ ] Roughly symmetric about diagonal (for equilibrium systems) ✓

## When to Transpose

### Decision Tree

```
Is your data indexed as data[x_index, y_index]?
│
├─ YES → Do you want x on horizontal axis?
│         │
│         ├─ YES → Use data.T with extent=[x_min, x_max, y_min, y_max]
│         └─ NO  → Use data with extent=[y_min, y_max, x_min, x_max]
│
└─ NO (data is [y_index, x_index])
          │
          └─ Want x on horizontal? → Use data without transpose
```

### Common Cases

**Case 1: meshgrid with indexing='ij'** (our case)

```python
# Data: c2[t1_idx, t2_idx]
# Want: x=t1, y=t2
# Solution: Use c2.T
t1_2d, t2_2d = np.meshgrid(t1, t2, indexing='ij')
ax.imshow(c2.T, ...)  # ✓ Transpose needed
```

**Case 2: meshgrid with indexing='xy'** (default)

```python
# Data: c2[t2_idx, t1_idx]  # Already transposed!
# Want: x=t1, y=t2
# Solution: Use c2 directly
t1_2d, t2_2d = np.meshgrid(t1, t2, indexing='xy')
ax.imshow(c2, ...)  # ✓ No transpose needed
```

**Case 3: Manual array creation**

```python
# Data: c2[i, j] where i=x_index, j=y_index
# Want: x on horizontal axis
# Solution: Use c2.T
c2 = np.zeros((nx, ny))
for i in range(nx):
    for j in range(ny):
        c2[i, j] = compute_value(x[i], y[j])

ax.imshow(c2.T, extent=[x[0], x[-1], y[0], y[-1]])  # ✓
```

## Debugging Transpose Issues

### Quick Test

```python
# Create simple test matrix with known diagonal
n = 100
c2_test = np.diag(np.ones(n))  # Identity matrix (diagonal of 1s)

# Plot both versions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Without transpose
ax1.imshow(c2_test, origin='lower')
ax1.set_title('Without .T')
ax1.set_xlabel('Column index')
ax1.set_ylabel('Row index')

# With transpose
ax2.imshow(c2_test.T, origin='lower')
ax2.set_title('With .T')
ax2.set_xlabel('Column index')
ax2.set_ylabel('Row index')

plt.show()
```

Expected result: Both show diagonal from bottom-left to top-right (because identity
matrix is symmetric). But for non-symmetric data, you'll see the difference!

### Asymmetric Test

```python
# Create asymmetric matrix
c2_test = np.triu(np.ones((100, 100)))  # Upper triangular

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.imshow(c2_test, origin='lower')
ax1.set_title('Without .T')

ax2.imshow(c2_test.T, origin='lower')
ax2.set_title('With .T (mirrored)')

# The difference is obvious for asymmetric data!
```

## Implementation in Homodyne

### Correct Usage (Fixed in Oct 2025)

```python
# homodyne/cli/commands.py

# Experimental C2 individual plots (line 1289)
im = ax.imshow(
    angle_data.T,  # ✓ Transpose applied
    aspect='equal',
    cmap='viridis',
    origin='lower',
    extent=extent,
)

# Simulated C2 plots (line 1607)
im = ax.imshow(
    c2_simulated[idx].T,  # ✓ Transpose applied
    extent=[t_vals[0], t_vals[-1], t_vals[0], t_vals[-1]],
    aspect='equal',
    cmap='viridis',
    origin='lower',
)
ax.set_xlabel('t₁ (s)', fontsize=11)  # ✓ Correct label
ax.set_ylabel('t₂ (s)', fontsize=11)  # ✓ Correct label

# Fitted C2 plots (line 1885)
im = ax.imshow(
    c2_fitted[i].T,  # ✓ Transpose applied
    aspect='equal',
    cmap='viridis',
    origin='lower',
    extent=extent,
)
```

### Historical Context

Prior to October 17, 2025 fix, 4 plotting functions were missing `.T`:

- Experimental C2 individual plots
- Experimental C2 single matrix
- Simulated C2 plots (also had swapped axis labels)
- Fitted C2 plots

Only the 3-panel matplotlib comparison plots were correct (added `.T` in earlier fix).

## Advanced: origin='lower' vs origin='upper'

### origin='lower' (recommended for scientific data)

```python
ax.imshow(data, origin='lower')
# (0, 0) at bottom-left corner
# y-axis increases upward (mathematical convention)
# Matches standard x-y plots
```

### origin='upper' (image processing default)

```python
ax.imshow(data, origin='upper')
# (0, 0) at top-left corner
# y-axis increases downward (image row convention)
# Natural for photographs, but confusing for scientific data
```

**Homodyne uses** `origin='lower'` for all C2 plots to match mathematical physics
convention.

## Summary

### Key Takeaways

1. **Always transpose** when data is indexed as `data[x_index, y_index]` but you want x
   on horizontal axis
1. **Use** `origin='lower'` for scientific data (mathematical convention)
1. **Validate** by checking diagonal orientation after plotting
1. **Test** with asymmetric matrices to verify transpose direction

### One-Line Rule

> **If meshgrid uses `indexing='ij'` and you want x=horizontal, y=vertical, then use
> `.T`**

### Fix Template

```python
# Before (WRONG):
ax.imshow(c2, origin='lower', extent=[x_min, x_max, y_min, y_max])

# After (CORRECT):
ax.imshow(c2.T, origin='lower', extent=[x_min, x_max, y_min, y_max])
ax.set_xlabel('x_label')
ax.set_ylabel('y_label')
```

## References

- **Full fix report**: `docs/archive/2025-10-nlsq-integration/PLOTTING_TRANSPOSE_FIX.md`
- **Fixed files**: `homodyne/cli/commands.py` (4 locations, Oct 17 2025)
- **Matplotlib docs**:
  https://matplotlib.org/stable/api/\_as_gen/matplotlib.pyplot.imshow.html
- **NumPy meshgrid**:
  https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html

______________________________________________________________________

**Last Updated**: November 17, 2025 **Applies to**: homodyne v2.x with matplotlib >=
3.8.0
