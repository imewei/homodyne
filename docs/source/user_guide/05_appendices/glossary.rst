.. _glossary:

Glossary
========

This glossary defines terms used in homodyne documentation and XPCS analysis.

---

.. glossary::
   :sorted:

   APS
      Advanced Photon Source — a synchrotron X-ray facility at Argonne National
      Laboratory (Illinois, USA). The primary source of XPCS data for which
      homodyne was developed.

   APS-U
      Advanced Photon Source Upgrade — the upgraded APS facility with a new
      multi-bend achromat storage ring providing dramatically higher brightness
      and coherent flux.

   anomalous diffusion
      Diffusion in which the mean-squared displacement scales as
      :math:`\langle r^2(t) \rangle \propto t^\alpha` with :math:`\alpha \neq 1`.
      Sub-diffusion (:math:`\alpha < 1`) is seen in gels, glasses, and crowded
      environments. Super-diffusion (:math:`\alpha > 1`) occurs in active systems.

   ArviZ
      A Python package for exploratory analysis of Bayesian models. Provides
      diagnostics (R-hat, ESS, BFMI), posterior visualization, and model
      comparison tools. Used in homodyne for CMC result analysis.

   Brownian motion
      Random thermal motion of particles suspended in a fluid, described by
      the Langevin equation. Characterized by normal diffusion (:math:`\alpha = 1`).

   chi-squared
      A statistic measuring goodness of fit: :math:`\chi^2 = \sum_i (y_i - f_i)^2 / \sigma_i^2`.
      The **reduced** chi-squared :math:`\chi^2_\nu = \chi^2 / (n - p)` normalizes by
      degrees of freedom. Values near 1 indicate a good fit.

   CMC
      Consensus Monte Carlo — a divide-and-conquer MCMC method that partitions data
      into shards, runs NUTS sampling on each shard independently, then combines
      posteriors using precision-weighted averaging. Enables Bayesian inference on
      large datasets that would be infeasible with full-data MCMC.

   coherence length
      The spatial extent over which X-rays maintain a definite phase relationship.
      The transverse coherence length determines the speckle size; the longitudinal
      coherence length determines the energy resolution of the correlation.

   contrast
      The amplitude of the correlation function at zero lag time, denoted :math:`\beta`
      (also called speckle contrast or coherence factor). Values range from 0 (fully
      incoherent) to 1 (fully coherent). Typical experimental values: 0.01–0.5.

   Couette geometry
      A concentric-cylinder shear cell geometry used in rheology-XPCS experiments.
      The inner cylinder (rotor) rotates while the outer cylinder (stator) is fixed,
      creating simple shear flow in the gap.

   decorrelation
      The process by which the normalized correlation function :math:`g_2(t)` decays
      from 1+:math:`\beta` to the baseline (offset) as lag time increases. Fast
      decorrelation means fast dynamics.

   DLS
      Dynamic Light Scattering — the optical (visible light) analogue of XPCS. Uses
      the same physical principles but with visible laser light instead of X-rays.
      Limited to dilute, transparent samples and larger length scales.

   D_offset
      The baseline (time-independent) diffusion coefficient in homodyne's model.
      Adds a linear contribution to the diffusion kernel:
      :math:`J \supset D_\text{offset} (t_2 - t_1)`. Physically represents a fast
      background diffusion component.

   D0
      The reference diffusion coefficient :math:`D_0` in homodyne's model. Controls
      the amplitude of the diffusion kernel and is the primary parameter of interest
      for characterizing particle dynamics. Units: Å²/s.

   divergent transitions
      In NUTS/HMC sampling, transitions where the numerical integrator produces
      energy errors exceeding a threshold. High divergence rates (> 10%) indicate
      problems with the posterior geometry or sampler configuration.

   effective sample size (ESS)
      The number of truly independent samples a Markov chain is equivalent to,
      accounting for chain autocorrelation. A rule of thumb: ESS > 400 for reliable
      posterior summaries.

   ergodic
      A system is ergodic if time averages equal ensemble averages. Standard XPCS
      analysis assumes ergodicity; XPCS with two-time correlations can detect and
      accommodate non-ergodic behavior (aging, non-stationary dynamics).

   g1
      The normalized first-order (field) correlation function:
      :math:`g_1(q, t) = \langle E^*(q, 0) E(q, t) \rangle / \langle |E(q)|^2 \rangle`.
      For Brownian particles: :math:`g_1(q, t) = \exp(-q^2 D_0 t^\alpha)`.

   g2
      The normalized second-order (intensity) correlation function:
      :math:`g_2(q, t) = \langle I(q, 0) I(q, t) \rangle / \langle I(q) \rangle^2`.
      Related to :math:`g_1` by the Siegert relation: :math:`g_2 = 1 + \beta |g_1|^2`.

   GAP
      See: gap distance

   gap distance
      The stator-rotor separation in a Couette shear cell, denoted :math:`h`. Configured
      as ``gap_distance`` in YAML (in µm). Internally stored in Å (1 µm = 10⁴ Å).

   gamma_dot_0
      The reference shear rate :math:`\dot\gamma_0` (units: s⁻¹). Amplitude of the
      time-dependent shear rate in homodyne's laminar flow model.

   HDF5
      Hierarchical Data Format 5 — a file format for large numerical datasets, widely
      used at synchrotron facilities for storing raw and reduced XPCS data.

   HMC
      Hamiltonian Monte Carlo — a family of MCMC algorithms that use Hamiltonian
      dynamics to propose distant moves, reducing random-walk behavior. NUTS is an
      adaptive variant of HMC.

   homodyne detection
      Detection of scattered X-rays without mixing with a reference beam. Produces
      :math:`C_2 \propto |g_1|^2` via the Siegert relation. Contrast with heterodyne
      detection, where scattered and reference beams are mixed.

   intermediate scattering function (ISF)
      See: g1

   JAX
      A Python library for high-performance numerical computing with automatic
      differentiation and JIT compilation via XLA. Homodyne uses JAX for all
      numerical computations.

   JIT compilation
      Just-in-Time compilation — compiling code at runtime rather than ahead-of-time.
      JAX's JIT compilation traces Python functions and compiles them to optimized
      XLA code on first invocation; subsequent calls are fast.

   laminar_flow
      Analysis mode for samples under laminar shear (e.g., Couette geometry). Uses
      7 physical parameters (D₀, α, D_offset, γ̇₀, β, γ̇_offset, φ₀) and includes
      the sinc² factor for shear-induced decorrelation.

   Latin Hypercube Sampling (LHS)
      A stratified random sampling method that ensures even coverage of parameter
      space. Used in homodyne's multi-start NLSQ to generate diverse starting points.

   Levenberg-Marquardt
      A damped least-squares algorithm that interpolates between gradient descent
      (far from minimum) and Gauss-Newton (near minimum). The basis of homodyne's
      trust-region NLSQ solver.

   NLSQ
      Non-Linear Least Squares — a class of optimization methods for minimizing a
      sum of squared residuals with respect to nonlinear parameters. Homodyne uses
      the NLSQ package with a trust-region LM solver.

   NUTS
      No-U-Turn Sampler — an adaptive extension of HMC that automatically chooses
      the trajectory length, avoiding the need to tune it manually. Used in homodyne's
      CMC backend (via NumPyro).

   NumPyro
      A probabilistic programming library built on JAX and NumPy, providing NUTS/HMC
      samplers with automatic differentiation. Homodyne's CMC backend uses NumPyro.

   offset
      The baseline level of the correlation function at large lag times, where
      :math:`C_2 \to 1` in the ideal case. Deviations from 1 indicate incoherent
      background contributions.

   per-angle mode
      Controls how homodyne handles angle-to-angle variations in speckle contrast and
      background offset. Options: ``auto`` (recommended), ``constant``, ``individual``,
      ``fourier``.

   phi_0
      The angular offset :math:`\phi_0` (degrees) between the shear flow direction
      and the detector coordinate system. Used in laminar_flow mode to orient the
      sinc² angular dependence.

   R-hat
      The Gelman-Rubin convergence statistic. Values near 1.0 indicate that multiple
      Markov chains have converged to the same distribution. Rule of thumb: R-hat < 1.05.

   relaxation time
      The characteristic timescale at which the correlation function decays to 1/e
      of its initial value. For normal diffusion: :math:`\tau_q \sim (q^2 D_0)^{-1}`.

   shard
      In CMC, a subset of the data assigned to one worker process. The shard size
      (``max_points_per_shard``) controls the trade-off between NUTS accuracy
      (larger shards) and computation time (smaller shards).

   shear rate
      The rate of deformation in a shear flow, :math:`\dot\gamma = dv_x/dy` (s⁻¹).
      In Couette geometry, the average shear rate equals the rotor angular velocity
      times the ratio of rotor radius to gap width.

   Siegert relation
      The relation :math:`g_2(q,t) = 1 + \beta |g_1(q,t)|^2`, connecting the
      measurable intensity correlation function :math:`g_2` to the intermediate
      scattering function :math:`g_1`.

   sinc
      :math:`\mathrm{sinc}(x) = \sin(\pi x) / (\pi x)`. The shear-induced term in
      homodyne's laminar flow model contains :math:`\mathrm{sinc}^2(\cdot)`, which
      creates zeros (nulls) at specific lag times when the shear displacement equals
      multiples of :math:`1/q`.

   speckle
      A random, granular intensity pattern produced when coherent radiation
      (light or X-rays) is scattered by a disordered sample. Speckle patterns
      change as scatterers move, encoding the dynamics in their temporal fluctuations.

   speckle contrast
      See: contrast

   static mode
      Analysis mode for equilibrium samples with pure diffusive dynamics. Uses 3
      physical parameters (D₀, α, D_offset). No angular dependence expected.

   Stokes-Einstein equation
      Relates the diffusion coefficient to particle size:
      :math:`D_0 = k_BT / (6\pi\eta R_h)`, where :math:`k_BT` is thermal energy,
      :math:`\eta` is viscosity, and :math:`R_h` is the hydrodynamic radius.

   sub-diffusion
      Diffusion with :math:`\alpha < 1` in the mean-squared displacement
      :math:`\langle r^2 \rangle \propto t^\alpha`. Characteristic of caged motion in
      dense suspensions, gels, and glasses.

   super-diffusion
      Diffusion with :math:`\alpha > 1`. Seen in active particle systems, anomalous
      transport in heterogeneous media, and driven systems.

   Taylor-Couette geometry
      See: Couette geometry

   two-time correlation function (C2)
      The main observable in XPCS: :math:`C_2(t_1, t_2)` correlates scattering
      intensities at two absolute times rather than a single lag. Captures
      non-stationary dynamics. Shape: (n_phi, n_t1, n_t2).

   uv
      A fast Python package manager and project tool used to manage homodyne's
      virtual environment and dependencies. All project commands use ``uv run``.

   wavevector
      The scattering vector :math:`q = (4\pi/\lambda) \sin(\theta/2)`, where
      :math:`\lambda` is the X-ray wavelength and :math:`\theta` is the scattering
      angle. In XPCS, :math:`q` selects the length scale probed: large :math:`q`
      → short distances.

   XLA
      Accelerated Linear Algebra — a domain-specific compiler for linear algebra
      operations, used by JAX as its execution backend. XLA JIT-compiles and
      optimizes computational graphs for CPU (and GPU/TPU).

   XPCS
      X-ray Photon Correlation Spectroscopy. A technique using coherent X-rays to
      measure the dynamics of materials via temporal correlations of scattered
      intensities.

   YML / YAML
      YAML Ain't Markup Language — a human-readable data serialization format.
      All homodyne configuration is written in YAML files loaded by ``ConfigManager``.
