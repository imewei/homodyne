# Shear Gradient Check (2025-11-12)

Data source: `/home/wei/Documents/Projects/data/C020/homodyne_results/nlsq` from the
2025-11-12 laminar-flow run. Script executed from repo root:

```python
import json
import numpy as np
import jax
import jax.numpy as jnp
from homodyne.core.jax_backend import compute_g1_total

with open('/home/wei/Documents/Projects/data/C020/homodyne_results/nlsq/parameters.json') as f:
    params_json = json.load(f)
order = ['D0','alpha','D_offset','gamma_dot_t0','beta','gamma_dot_t_offset','phi0']
params = jnp.array([params_json['parameters'][k]['value'] for k in order])
npz = np.load('/home/wei/Documents/Projects/data/C020/homodyne_results/nlsq/fitted_data.npz')
t1 = jnp.array(npz['t1'])
t2 = jnp.array(npz['t2'])
phi = jnp.array(npz['phi_angles'])
q = float(npz['q'][0])
L = 2_000_000.0
dt = float(npz['t1'][1] - npz['t1'][0])
exp_c2 = jnp.array(npz['c2_exp'])
per_angle = jnp.array(npz['per_angle_scaling_solver'])
contrasts, offsets = per_angle[:,0], per_angle[:,1]
@jax.jit
def residuals_vec(p):
    g1 = compute_g1_total(p, t1, t2, phi, q, L, dt)
    pred = offsets[:, None, None] + contrasts[:, None, None] * jnp.square(g1)
    return (pred - exp_c2).reshape(-1)
sse_fn = lambda p: jnp.sum(residuals_vec(p)**2)
grad = jax.grad(sse_fn)(params)
for name, val, g in zip(order, params, grad):
    print(f"{name:18s} value={float(val):+.6e} grad={float(g):+.6e}")
```

Output:

```
SSE 1.659705e+05
D0                 value=+4.007580e+02 grad=+2.698292e+01
alpha              value=-1.400000e-02 grad=+4.236533e+04
D_offset           value=-6.742710e-01 grad=+2.850510e+01
gamma_dot_t0       value=+3.000000e-03 grad=+8.684888e+06
beta               value=-9.090000e-01 grad=+1.028153e+05
gamma_dot_t_offset value=+0.000000e+00 grad=+3.469348e+08
phi0               value=-4.529225e-02 grad=-4.363216e+01
```

All three shear parameters report multi-order-of-magnitude larger gradients than the
diffusion terms, confirming the optimizer stopped while the cost function still has
steep descent directions in the shear subspace.
