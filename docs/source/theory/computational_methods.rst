.. _theory_computational_methods:

Computational Methods
======================

Homodyne solves two distinct estimation problems: (1) **non-linear least squares** (NLSQ)
fitting of the two-time correlation function to recover point estimates with uncertainties,
and (2) **Bayesian posterior sampling** via Consensus Monte Carlo (CMC) with NUTS to obtain
full posterior distributions. This page describes the mathematical foundations of both
methods and the numerical techniques used to make them fast and reliable on CPU hardware.

.. contents:: Contents
   :local:
   :depth: 2


Non-Linear Least Squares (NLSQ)
---------------------------------

**Problem formulation**: Given measured two-time correlation data
:math:`\{c_2^{ij,\mathrm{meas}}\}` and the theoretical model :math:`c_2^{ij}(\theta)`,
minimize the weighted sum of squared residuals:

.. math::
   :label: nlsq_objective

   \mathcal{L}(\theta) \;=\; \sum_{(i,j)\in\mathcal{S}}
     w_{ij}\!\left[c_2^{ij,\mathrm{meas}} - c_2^{ij}(\theta)\right]^2

where :math:`\mathcal{S}` is the set of sampled time-pairs and :math:`w_{ij}` are
measurement weights (typically :math:`\propto 1/\sigma_{ij}^2`).


Levenberg-Marquardt Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Homodyne uses the **Trust-Region Levenberg-Marquardt** algorithm, which is the gold
standard for non-linear least squares. The update step at iteration :math:`k` solves:

.. math::
   :label: LM_step

   \left(J_k^\top W J_k + \lambda_k \mathrm{diag}(J_k^\top W J_k)\right)\delta\theta_k
   \;=\; J_k^\top W r_k

where :math:`J_k = \partial r_k / \partial\theta` is the Jacobian matrix of residuals,
:math:`W = \mathrm{diag}(w_{ij})` is the weight matrix, :math:`r_k` is the residual vector,
and :math:`\lambda_k \geq 0` is the damping parameter.

- When :math:`\lambda_k \to 0`: reduces to **Gauss-Newton** (fast convergence near minimum).
- When :math:`\lambda_k \to \infty`: reduces to **gradient descent** (stable far from minimum).

The trust-region variant adaptively controls :math:`\lambda_k` based on the ratio of
actual vs predicted reduction, providing global convergence guarantees.

**Implementation**: Uses the NLSQ library (version :math:`\geq` 0.6.4) via
:class:`homodyne.optimization.nlsq.adapter.NLSQAdapter`. The library provides
GPU-accelerated non-linear least squares; homodyne uses its CPU mode.


Cumulative Trapezoid Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The diffusion and shear integrals :math:`\mathcal{D}(t_1, t_2)` and :math:`\Gamma(t_1, t_2)`
are evaluated analytically via the closed-form power-law formulas (see
:ref:`theory_transport_coefficient`). For general non-power-law models, the package falls
back to **cumulative trapezoid integration**:

.. math::

   \int_{t_1}^{t_2} f(t)\,dt \;\approx\;
   \sum_{k=\lfloor t_1/\Delta t\rfloor}^{\lfloor t_2/\Delta t\rfloor - 1}
   \frac{f(t_k) + f(t_{k+1})}{2}\,\Delta t_k

This is second-order accurate and numerically stable for monotone integrands.

The two-time integral is vectorized over the full :math:`(t_1, t_2)` grid using
:func:`jax.numpy.cumsum` and broadcasting, giving :math:`O(N_t^2)` evaluations in
:math:`O(N_t)` operations via the prefix-sum trick:

.. math::

   \int_{t_1}^{t_2} f\,dt = F(t_2) - F(t_1), \qquad F(t) = \int_0^t f(t')\,dt'

where :math:`F(t)` is computed once as a cumulative sum.


Jacobian Computation
~~~~~~~~~~~~~~~~~~~~~

The Jacobian :math:`\partial c_2 / \partial\theta` is computed by **JAX automatic
differentiation**:

.. code-block:: python

   from jax import jacfwd

   def residuals(theta):
       c2_model = compute_c2(theta, t1, t2, q, phi, h)
       return c2_model.ravel() - c2_measured.ravel()

   J = jacfwd(residuals)(theta_current)

Forward-mode AD is used because :math:`n_\mathrm{params} \ll n_\mathrm{residuals}`,
making forward mode more efficient than reverse mode (which would require
:math:`n_\mathrm{residuals}` backward passes).

The Jacobian is JIT-compiled, so the first call pays a compilation overhead (typically
0.5–2 seconds) but subsequent calls are fast.


Memory Management
~~~~~~~~~~~~~~~~~~

For large datasets (:math:`> 10^7` data points), the full Jacobian matrix may exceed
available RAM. Homodyne uses two strategies:

**Stratified LS** (default, :math:`< 10^7` points):
Full Jacobian fits in memory. Parameters selected by the NLSQ library's internal strategy.

**Hybrid Streaming** (:math:`> 10^7` points):
The data is partitioned into shards that fit in memory. The normal equations
:math:`J^\top W J` and :math:`J^\top W r` are accumulated shard-by-shard:

.. math::

   A \;=\; \sum_s J_s^\top W_s J_s, \qquad b \;=\; \sum_s J_s^\top W_s r_s

The LM step then solves :math:`(A + \lambda\,\mathrm{diag}(A))\delta\theta = b` using
the accumulated matrices. This keeps memory usage at :math:`O(N_\mathrm{params}^2)`
regardless of dataset size.

Configure via:

.. code-block:: yaml

   optimization:
     nlsq:
       memory_fraction: 0.75     # 75% of RAM triggers streaming


CMA-ES for Multi-Scale Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the physical parameters span many orders of magnitude (scale ratio :math:`> 10^6`,
e.g., :math:`D_0 \sim 10^4` and :math:`\dot{\gamma}_0 \sim 10^{-3}`), gradient-based NLSQ
can stagnate in poor local minima. The package provides **CMA-ES** (Covariance Matrix
Adaptation Evolution Strategy) as a global pre-optimizer:

- CMA-ES is a **derivative-free** evolutionary algorithm.
- The covariance matrix adapts to the objective landscape, handling ill-conditioning.
- After CMA-ES finds a good basin, NLSQ refines to high precision.

Configure via:

.. code-block:: yaml

   optimization:
     nlsq:
       cmaes:
         enable: true
         preset: "cmaes-global"  # 200 generations
         refine_with_nlsq: true


Consensus Monte Carlo (CMC)
-----------------------------

Homodyne uses **Consensus Monte Carlo** (Scott et al. 2016) to scale MCMC to large datasets
that are too expensive for a single NUTS chain.

**Core idea**: Partition the data into :math:`M` shards of size :math:`n_s`. Run independent
NUTS chains on each shard, obtaining posterior samples :math:`\{\theta^{(k)}_s\}_{k=1}^K`
from the shard-specific posterior:

.. math::
   :label: shard_posterior

   p_s(\theta | \mathcal{D}_s) \;\propto\;
   p(\theta) \cdot \prod_{(i,j)\in\mathcal{D}_s}
   p\!\left(c_2^{ij,\mathrm{meas}} | c_2^{ij}(\theta)\right)

**Consensus step**: Combine the :math:`M` shard posteriors into the global posterior
estimate. For the consensus algorithm, each shard posterior is treated as a weighted
contribution to the full posterior via the Weierstrass consensus rule:

.. math::
   :label: consensus_combine

   p(\theta | \mathcal{D}) \;\approx\;
   \frac{\prod_s p_s(\theta | \mathcal{D}_s)}{p(\theta)^{M-1}}

This is exact when the likelihood factors over shards (which it does for independent
measurements) and the prior is Gaussian.


NUTS (No-U-Turn Sampler)
~~~~~~~~~~~~~~~~~~~~~~~~~

Each shard runs NUTS, an adaptive HMC algorithm that automatically tunes:

- **Step size** :math:`\epsilon`: controlled by dual averaging to achieve target acceptance
  rate :math:`\delta_\mathrm{target} = 0.8`.
- **Trajectory length**: uses the No-U-Turn criterion to stop leapfrog integration before
  the trajectory doubles back, avoiding wasted computation.

The NUTS update is:

.. math::

   (\theta^*, \rho^*) \sim \mathrm{HMC}(\theta^{(k)}, \rho^{(k)}; \epsilon, L)

where :math:`\rho \sim \mathcal{N}(0, M)` is the auxiliary momentum (with mass matrix
:math:`M` estimated during warmup), and :math:`L` is the number of leapfrog steps
determined adaptively.

**Cost per sample**: :math:`O(n_s \cdot L)` gradient evaluations, where :math:`L \sim
O(\log n_s)` for typical problems. This is implemented by NumPyro, which JIT-compiles the
NUTS kernel via JAX.


Adaptive Sampling
~~~~~~~~~~~~~~~~~~

To reduce overhead for small shards, homodyne uses **adaptive sampling**: warmup and sample
counts are scaled with shard size:

.. math::
   :label: adaptive_sampling

   n_\mathrm{warmup}(n_s) \;=\;
   \max\!\left(n_\mathrm{warmup}^\mathrm{min},\;
   n_\mathrm{warmup}^\mathrm{default} \cdot \min\!\left(1,\, \frac{n_s}{n_s^\mathrm{ref}}\right)\right)

with a similar formula for :math:`n_\mathrm{samples}`. This prevents spending most of the
wall-time on NUTS warmup for small shards.

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Shard size
     - Warmup
     - Samples
     - Reduction
   * - 50 pts
     - 140
     - 350
     - 75%
   * - 5K pts
     - 250
     - 750
     - 50%
   * - 50K+ pts
     - 500
     - 1,500
     - None (full)


Reparameterization (Log-Space Priors)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameters like :math:`D_0` span several orders of magnitude, making the posterior
geometry challenging for NUTS. Homodyne uses **log-space reparameterization**:

.. math::

   \tilde{D}_0 \;=\; \log\!\left(\frac{D_0}{t_\mathrm{ref}^\alpha}\right)

where :math:`t_\mathrm{ref} = \sqrt{\Delta t \cdot t_\mathrm{max}}` is a reference timescale
computed from the data. In reparameterized space, the posterior is approximately Gaussian,
improving NUTS efficiency (fewer leapfrog steps, higher acceptance rate).

The reference timescale is computed in :func:`homodyne.optimization.cmc.reparameterization.compute_t_ref`,
which raises :class:`ValueError` for invalid inputs; the caller catches this and falls back
to :math:`t_\mathrm{ref} = 1.0`.


Diagnostics
~~~~~~~~~~~~

Convergence is assessed using standard ArviZ diagnostics:

.. math::

   \hat{R} \;=\; \sqrt{\frac{\hat{V}}{W}}

where :math:`\hat{V}` is the between-chain variance estimate and :math:`W` is the
within-chain variance. Good convergence: :math:`\hat{R} < 1.01`.

**Effective sample size**:

.. math::

   \mathrm{ESS} \;=\; \frac{K}{\sum_{t=-\infty}^\infty \rho_t}

where :math:`\rho_t` is the lag-:math:`t` autocorrelation. Adequate sampling:
:math:`\mathrm{ESS} > 400` per parameter.

**BFMI** (Bayesian Fraction of Missing Information):

.. math::

   \mathrm{BFMI} \;=\; \frac{\langle(\Delta H)^2\rangle}{\langle H^2\rangle}

where :math:`H = -\log p(\theta) - \log p(y|\theta)` is the Hamiltonian. Good mixing:
:math:`\mathrm{BFMI} > 0.3`.

Quality filtering removes shards with divergence rate :math:`> 10\%` before consensus
(configurable via ``max_divergence_rate``).


.. seealso::

   - :ref:`theory_anti_degeneracy` — per-angle scaling modes
   - :mod:`homodyne.optimization.nlsq.adapter` — NLSQAdapter implementation
   - :mod:`homodyne.optimization.cmc.sampler` — SamplingPlan and NUTS execution
   - :mod:`homodyne.optimization.cmc.reparameterization` — log-space priors
   - :mod:`homodyne.optimization.cmc.backends.multiprocessing` — parallel CMC backend
