"""
JAX JIT-compatible stratified residual function for NLSQ optimization.

This module provides a JIT-compatible version of StratifiedResidualFunction that uses
static shapes and vmap for vectorization, solving the JAX tracing incompatibility.

Key Improvements over original StratifiedResidualFunction:
- Uses jax.vmap for parallel chunk processing (no Python loops)
- Pads chunks to uniform size for static shapes (JIT-compatible)
- Fully JIT-compiled for maximum performance
- Maintains angle stratification guarantee
- Buffer donation for memory efficiency (FR-003)

Author: Homodyne Development Team
Date: 2025-11-13
Version: 2.4.0 (updated with buffer donation in 2.14.0)
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from homodyne.core.physics_nlsq import compute_g2_scaled
from homodyne.core.physics_utils import apply_diagonal_correction
from homodyne.utils.logging import get_logger, log_phase


class StratifiedResidualFunctionJIT:
    """
    JIT-compatible stratified residual function using padded vmap.

    This class solves the JAX JIT incompatibility by:
    1. Padding all chunks to uniform size (static shapes)
    2. Using jax.vmap for vectorized parallel processing
    3. Masking padded values in the final residuals

    The function maintains angle stratification (all chunks contain all angles)
    while being fully JIT-compilable.

    Attributes:
        phi_padded: Padded phi arrays (n_chunks, max_chunk_size)
        t1_padded: Padded t1 arrays (n_chunks, max_chunk_size)
        t2_padded: Padded t2 arrays (n_chunks, max_chunk_size)
        g2_padded: Padded g2 observations (n_chunks, max_chunk_size)
        mask: Boolean mask for real vs padded data (n_chunks, max_chunk_size)
        n_chunks: Number of stratified chunks
        max_chunk_size: Maximum points per chunk (for padding)
        n_real_points: Total number of real (non-padded) data points
    """

    def __init__(
        self,
        stratified_data: any,
        per_angle_scaling: bool,
        physical_param_names: list[str],
        logger: logging.Logger | None = None,
    ):
        """
        Initialize JIT-compatible stratified residual function.

        Args:
            stratified_data: Object with .chunks attribute containing angle-stratified chunks
            per_angle_scaling: Whether per-angle scaling parameters are used
            physical_param_names: List of physical parameter names
            logger: Optional logger for diagnostics
        """
        self.logger = logger or get_logger(__name__)
        self.chunks = stratified_data.chunks
        self.per_angle_scaling = per_angle_scaling
        self.physical_param_names = physical_param_names

        if not self.chunks:
            raise ValueError("stratified_data.chunks is empty")

        self.n_chunks = len(self.chunks)

        # Extract global metadata (same across all chunks)
        self.q, self.L, self.dt = self._extract_global_metadata()
        self.phi_unique, self.t1_unique, self.t2_unique = self._extract_unique_values()
        self.n_phi = len(self.phi_unique)

        # Prepare sigma array
        sigma_array = np.asarray(stratified_data.sigma, dtype=np.float64)
        self.sigma_jax = jnp.asarray(sigma_array)

        # Create padded arrays with static shapes
        self.logger.info(f"Creating padded arrays for {self.n_chunks} chunks...")
        (
            self.phi_padded,
            self.t1_padded,
            self.t2_padded,
            self.g2_padded,
            self.mask,
            self.max_chunk_size,
            self.n_real_points,
        ) = self._create_padded_arrays()

        self.logger.info(
            f"Padded arrays created: shape ({self.n_chunks}, {self.max_chunk_size}), "
            f"real points: {self.n_real_points:,}, "
            f"padding overhead: {(1 - self.n_real_points / (self.n_chunks * self.max_chunk_size)) * 100:.2f}%"
        )

        # JIT-compile the main residual computation with buffer donation
        # Performance Optimization (Spec 001 - FR-003, T038): Buffer donation allows
        # JAX to reuse input buffers for output, reducing peak memory by avoiding
        # the need to allocate new buffers for intermediate results.
        # donate_argnums=(0,) tells JAX the params array can be reused after the call.
        self.logger.info("JIT-compiling residual function with buffer donation...")
        # T035: Add log_phase for JIT compilation timing with memory tracking
        with log_phase(
            "jit_residual_compilation", logger=self.logger, track_memory=True
        ) as phase:
            self._residual_fn_jit = jax.jit(
                self._compute_all_residuals,
                donate_argnums=(0,),  # FR-003: Donate params buffer to reduce memory
            )
        self.logger.info(
            f"✓ JIT compilation setup complete in {phase.duration:.3f}s "
            f"(buffer donation enabled)"
        )

    def _extract_global_metadata(self) -> tuple[float, float, float | None]:
        """Extract q, L, dt from chunks (should be same for all chunks)."""
        q_values = [float(chunk.q) for chunk in self.chunks]
        L_values = [float(chunk.L) for chunk in self.chunks]
        dt_values = [
            float(chunk.dt) if chunk.dt is not None else None for chunk in self.chunks
        ]

        # Validate consistency
        if not all(abs(q - q_values[0]) < 1e-9 for q in q_values):
            raise ValueError("Inconsistent q values across chunks")
        if not all(abs(L - L_values[0]) < 1e-6 for L in L_values):
            raise ValueError("Inconsistent L values across chunks")

        q = q_values[0]
        L = L_values[0]
        dt = dt_values[0] if dt_values[0] is not None else None

        self.logger.debug(f"Global metadata: q={q:.6f}, L={L:.1f}, dt={dt}")
        return q, L, dt

    def _extract_unique_values(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Extract unique phi, t1, t2 values from ALL chunks.

        CRITICAL: Must extract from all chunks, not just first chunk, because
        stratified chunking may distribute different subsets of t1/t2 values
        across different chunks.
        """
        # Concatenate values from all chunks to get complete set
        all_phi = np.concatenate([chunk.phi for chunk in self.chunks])
        all_t1 = np.concatenate([chunk.t1 for chunk in self.chunks])
        all_t2 = np.concatenate([chunk.t2 for chunk in self.chunks])

        # Extract unique values across all chunks
        phi_unique = jnp.sort(jnp.unique(jnp.asarray(all_phi)))
        t1_unique = jnp.sort(jnp.unique(jnp.asarray(all_t1)))
        t2_unique = jnp.sort(jnp.unique(jnp.asarray(all_t2)))

        self.logger.debug(
            f"Unique values (from all chunks): {len(phi_unique)} phi, {len(t1_unique)} t1, {len(t2_unique)} t2"
        )

        # Validation: check if we missed any values by comparing with first chunk
        first_chunk = self.chunks[0]
        _phi_first = jnp.sort(jnp.unique(jnp.asarray(first_chunk.phi)))  # noqa: F841
        t1_first = jnp.sort(jnp.unique(jnp.asarray(first_chunk.t1)))
        t2_first = jnp.sort(jnp.unique(jnp.asarray(first_chunk.t2)))

        if len(t1_unique) != len(t1_first) or len(t2_unique) != len(t2_first):
            self.logger.debug(
                f"Stratified chunking: chunks have different time point subsets "
                f"(first chunk: {len(t1_first)} t1, all chunks: {len(t1_unique)} t1) - "
                f"using complete set from all chunks"
            )

        return phi_unique, t1_unique, t2_unique

    def _create_padded_arrays(
        self,
    ) -> tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int, int
    ]:
        """
        Create padded arrays with uniform size across all chunks.

        Returns:
            phi_padded, t1_padded, t2_padded, g2_padded, mask, max_chunk_size, n_real_points
        """
        # Determine max chunk size
        chunk_sizes = [len(chunk.phi) for chunk in self.chunks]
        max_chunk_size = max(chunk_sizes)
        n_real_points = sum(chunk_sizes)

        self.logger.debug(
            f"Max chunk size: {max_chunk_size:,}, total real points: {n_real_points:,}"
        )

        # Initialize padded arrays
        phi_padded = np.zeros((self.n_chunks, max_chunk_size), dtype=np.float64)
        t1_padded = np.zeros((self.n_chunks, max_chunk_size), dtype=np.float64)
        t2_padded = np.zeros((self.n_chunks, max_chunk_size), dtype=np.float64)
        g2_padded = np.zeros((self.n_chunks, max_chunk_size), dtype=np.float64)
        mask = np.zeros((self.n_chunks, max_chunk_size), dtype=bool)

        # Fill arrays with data and create mask
        for i, chunk in enumerate(self.chunks):
            n_points = len(chunk.phi)

            # Copy real data
            phi_padded[i, :n_points] = chunk.phi
            t1_padded[i, :n_points] = chunk.t1
            t2_padded[i, :n_points] = chunk.t2
            g2_padded[i, :n_points] = chunk.g2
            mask[i, :n_points] = True

            # Pad with last valid value (prevents out-of-bounds indexing)
            if n_points < max_chunk_size:
                phi_padded[i, n_points:] = chunk.phi[-1]
                t1_padded[i, n_points:] = chunk.t1[-1]
                t2_padded[i, n_points:] = chunk.t2[-1]
                g2_padded[i, n_points:] = chunk.g2[-1]
                # mask already False for padding

        # Convert to JAX arrays
        phi_padded_jax = jnp.asarray(phi_padded)
        t1_padded_jax = jnp.asarray(t1_padded)
        t2_padded_jax = jnp.asarray(t2_padded)
        g2_padded_jax = jnp.asarray(g2_padded)
        mask_jax = jnp.asarray(mask)

        return (
            phi_padded_jax,
            t1_padded_jax,
            t2_padded_jax,
            g2_padded_jax,
            mask_jax,
            max_chunk_size,
            n_real_points,
        )

    def _compute_single_chunk_residuals(
        self,
        phi_chunk: jnp.ndarray,
        t1_chunk: jnp.ndarray,
        t2_chunk: jnp.ndarray,
        g2_obs_chunk: jnp.ndarray,
        mask_chunk: jnp.ndarray,
        params_all: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute residuals for a single padded chunk.

        This function is designed to be vmapped over the chunk dimension.

        Args:
            phi_chunk: Phi values for this chunk (max_chunk_size,)
            t1_chunk: T1 values for this chunk (max_chunk_size,)
            t2_chunk: T2 values for this chunk (max_chunk_size,)
            g2_obs_chunk: Observed g2 for this chunk (max_chunk_size,)
            mask_chunk: Mask for real vs padded data (max_chunk_size,)
            params_all: All parameters [scaling_params, physical_params]

        Returns:
            Masked residuals (max_chunk_size,) - padded values are zeros
        """
        # Extract scaling and physical parameters
        if self.per_angle_scaling:
            contrast = params_all[: self.n_phi]
            offset = params_all[self.n_phi : 2 * self.n_phi]
            physical_params = params_all[2 * self.n_phi :]
        else:
            contrast = params_all[0]
            offset = params_all[1]
            physical_params = params_all[2:]

        # Compute theoretical g2 using vectorized computation
        if self.per_angle_scaling:
            # Vectorize over phi with corresponding contrast/offset
            compute_g2_vmap = jax.vmap(
                lambda phi_val, contrast_val, offset_val: jnp.squeeze(
                    compute_g2_scaled(
                        params=physical_params,
                        t1=self.t1_unique,
                        t2=self.t2_unique,
                        phi=phi_val,
                        q=self.q,
                        L=self.L,
                        contrast=contrast_val,
                        offset=offset_val,
                        dt=self.dt,
                    ),
                    axis=0,
                ),
                in_axes=(0, 0, 0),
            )
            g2_theory_grid = compute_g2_vmap(self.phi_unique, contrast, offset)
        else:
            # Legacy: single contrast/offset
            compute_g2_vmap = jax.vmap(
                lambda phi_val: jnp.squeeze(
                    compute_g2_scaled(
                        params=physical_params,
                        t1=self.t1_unique,
                        t2=self.t2_unique,
                        phi=phi_val,
                        q=self.q,
                        L=self.L,
                        contrast=contrast,
                        offset=offset,
                        dt=self.dt,
                    ),
                    axis=0,
                ),
                in_axes=0,
            )
            g2_theory_grid = compute_g2_vmap(self.phi_unique)

        # Apply diagonal correction (vmap over phi dimension)
        apply_diagonal_vmap = jax.vmap(apply_diagonal_correction, in_axes=0)
        g2_theory_grid = apply_diagonal_vmap(g2_theory_grid)

        # Flatten theory grid for indexing
        g2_theory_flat = g2_theory_grid.flatten()

        # Find indices of (phi, t1, t2) in the full grid
        _n_phi = len(self.phi_unique)  # noqa: F841 - Dimensions for grid validation
        n_t1 = len(self.t1_unique)
        n_t2 = len(self.t2_unique)

        # Note: clip removed - stratified LS data comes from same chunks that build
        # unique arrays, so all values are guaranteed to be in range. The clip was
        # causing optimization to converge to wrong local minima (D0=91342 vs 19253).
        # Original clip added in ae4848c for streaming optimizer, but not needed here.
        phi_indices = jnp.searchsorted(self.phi_unique, phi_chunk)
        t1_indices = jnp.searchsorted(self.t1_unique, t1_chunk)
        t2_indices = jnp.searchsorted(self.t2_unique, t2_chunk)

        # Compute flat indices
        flat_indices = phi_indices * (n_t1 * n_t2) + t1_indices * n_t2 + t2_indices

        # Extract theory values for chunk points
        g2_theory_chunk = g2_theory_flat[flat_indices]

        # Get sigma values for chunk points
        sigma_flat = self.sigma_jax.flatten()
        sigma_chunk = sigma_flat[flat_indices]

        # Compute weighted residuals
        EPS = 1e-10
        residuals_raw = (g2_obs_chunk - g2_theory_chunk) / (sigma_chunk + EPS)

        # v2.14.2+: Mask out both padded values AND diagonal values (t1 == t2)
        # Diagonal points are autocorrelation artifacts, not physics
        # CRITICAL FIX (2026-01-15): Compare actual time VALUES, not indices.
        # t1_indices and t2_indices reference DIFFERENT arrays (t1_unique vs t2_unique),
        # so comparing indices is wrong. Must compare the actual t1_chunk and t2_chunk values.
        non_diagonal = t1_chunk != t2_chunk
        residuals_masked = jnp.where(mask_chunk & non_diagonal, residuals_raw, 0.0)

        return residuals_masked

    def _compute_all_residuals(self, params: jnp.ndarray) -> jnp.ndarray:
        """
        Compute residuals for all chunks using vmap (JIT-compiled).

        Args:
            params: All parameters (scaling + physical)

        Returns:
            Flattened residuals INCLUDING padding (will be filtered in __call__)
            Shape: (n_chunks * max_chunk_size,) with zeros for padded values
        """
        # Vectorize over chunks (first dimension)
        vmap_fn = jax.vmap(
            lambda phi, t1, t2, g2, mask: self._compute_single_chunk_residuals(
                phi, t1, t2, g2, mask, params
            ),
            in_axes=(0, 0, 0, 0, 0),  # vmap over first dim of all arrays
        )

        # Compute residuals for all chunks in parallel
        residuals_padded = vmap_fn(
            self.phi_padded,
            self.t1_padded,
            self.t2_padded,
            self.g2_padded,
            self.mask,
        )  # Shape: (n_chunks, max_chunk_size)

        # Flatten residuals (padding is already masked to zero in _compute_single_chunk_residuals)
        residuals_flat = (
            residuals_padded.flatten()
        )  # Shape: (n_chunks * max_chunk_size,)

        # Return full array (filtering happens in __call__ to avoid JIT boolean indexing)
        return residuals_flat

    def __call__(self, params):
        """
        Compute residuals (interface for NLSQ least_squares).

        This method is JIT-traced by NLSQ, so it must use JAX operations only.
        Padded values are already masked to zero, so they don't contribute to
        the optimization objective (sum of squared residuals).

        Args:
            params: Parameters (numpy or JAX array)

        Returns:
            Residuals as JAX array (n_chunks * max_chunk_size,) with zeros for padding
            Note: Padding zeros don't affect optimization but increase array size
        """
        params_jax = jnp.asarray(params, dtype=jnp.float64)
        residuals_jax = self._residual_fn_jit(params_jax)
        return residuals_jax  # Keep as JAX array for JIT compatibility

    def validate_chunk_structure(self) -> bool:
        """
        Validate that all chunks contain all phi angles.

        Returns:
            True if validation passes

        Raises:
            ValueError: If validation fails
        """
        expected_angles = set(
            np.unique(np.round(np.asarray(self.phi_unique), decimals=6))
        )
        n_expected = len(expected_angles)

        self.logger.info(
            f"Validating chunk structure: {self.n_chunks} chunks, "
            f"{n_expected} expected angles per chunk"
        )

        for i, _chunk in enumerate(self.chunks):
            # Only check real data (not padding)
            n_real = int(np.sum(self.mask[i]))
            phi_real = self.phi_padded[i, :n_real]
            chunk_angles = set(np.unique(np.round(np.asarray(phi_real), decimals=6)))

            if chunk_angles != expected_angles:
                missing = expected_angles - chunk_angles
                extra = chunk_angles - expected_angles
                raise ValueError(
                    f"Chunk {i} has invalid angle distribution:\n"
                    f"  Missing angles: {sorted(missing)}\n"
                    f"  Extra angles: {sorted(extra)}\n"
                    f"  Expected {n_expected} angles, got {len(chunk_angles)}"
                )

        self.logger.info(
            "✓ Chunk structure validation passed: all chunks angle-complete"
        )
        return True

    def get_diagnostics(self) -> dict:
        """Get diagnostic information about the residual function."""
        return {
            "n_chunks": self.n_chunks,
            "max_chunk_size": self.max_chunk_size,
            "n_real_points": self.n_real_points,
            "padding_overhead_pct": (
                1 - self.n_real_points / (self.n_chunks * self.max_chunk_size)
            )
            * 100,
            "n_phi": self.n_phi,
            "n_t1": len(self.t1_unique),
            "n_t2": len(self.t2_unique),
            "per_angle_scaling": self.per_angle_scaling,
            "jit_compiled": True,
        }

    def log_diagnostics(self):
        """Log diagnostic information about the residual function."""
        diag = self.get_diagnostics()
        self.logger.info("Stratified Residual Function Diagnostics:")
        self.logger.info(f"  Chunks: {diag['n_chunks']}")
        self.logger.info(f"  Max chunk size: {diag['max_chunk_size']:,}")
        self.logger.info(f"  Real points: {diag['n_real_points']:,}")
        self.logger.info(f"  Padding overhead: {diag['padding_overhead_pct']:.2f}%")
        self.logger.info(f"  Angles (phi): {diag['n_phi']}")
        self.logger.info(f"  Time points (t1): {diag['n_t1']}")
        self.logger.info(f"  Time points (t2): {diag['n_t2']}")
        self.logger.info(f"  Per-angle scaling: {diag['per_angle_scaling']}")
        self.logger.info(f"  JIT compiled: {diag['jit_compiled']}")
