"""
Stratified residual function for NLSQ optimization with per-angle scaling.

This module provides a residual function that preserves angle-stratified chunk structure
during NLSQ optimization, solving the double-chunking problem that occurs when using
curve_fit_large() with per-angle scaling on large datasets.

Key Features:

- Maintains angle completeness across chunks (required for per-angle parameter gradients)
- JAX JIT-compiled chunk computation for performance
- Compatible with NLSQ's least_squares() function
- Supports both CPU and GPU execution
- Provides diagnostics for monitoring and validation

Usage:

.. code-block:: python

    from nlsq import least_squares
    from homodyne.optimization.nlsq.strategies.residual import StratifiedResidualFunction

    # Create stratified data (each chunk contains all angles)
    stratified_data = create_stratified_chunks(data, target_chunk_size=100000)

    # Create residual function
    residual_fn = StratifiedResidualFunction(
        stratified_data=stratified_data,
        model=model,
        per_angle_scaling=True,
        logger=logger
    )

    # Validate chunk structure
    residual_fn.validate_chunk_structure()

    # Use with NLSQ's least_squares()
    result = least_squares(
        fun=residual_fn,
        x0=initial_params,
        jac=None,  # JAX autodiff
        bounds=(lower, upper),
        method='trf'
    )

Author: Homodyne Development Team
Date: 2025-11-06
Version: 2.2.0
"""

import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from homodyne.core.physics_nlsq import compute_g2_scaled
from homodyne.core.physics_utils import apply_diagonal_correction


class StratifiedResidualFunction:
    """
    Residual function that respects angle-stratified chunk structure.

    This class wraps the model's residual computation to work with stratified chunks,
    ensuring that each chunk contains all phi angles. This is critical for per-angle
    scaling parameters to have non-zero gradients.

    The function is designed to work with NLSQ's least_squares() function, which calls
    the residual function at each optimization iteration.

    Attributes:
        chunks: List of angle-stratified data chunks
        model: TheoryEngine instance for computing residuals
        per_angle_scaling: Whether per-angle scaling is enabled
        logger: Logger instance for diagnostics
        n_chunks: Number of stratified chunks
        n_total_points: Total number of data points across all chunks
        compute_chunk_jit: JIT-compiled chunk residual computation
    """

    def __init__(
        self,
        stratified_data: Any,
        per_angle_scaling: bool,
        physical_param_names: list[str],
        logger: logging.Logger | None = None,
    ):
        """
        Initialize the stratified residual function.

        Args:
            stratified_data: Object with .chunks attribute containing angle-stratified chunks.
                Each chunk must have: phi, t1, t2, g2, q, L, dt attributes.
                stratified_data.sigma contains the full 3D sigma array (metadata).
            per_angle_scaling: Whether per-angle scaling parameters are used.
            physical_param_names: List of physical parameter names (e.g., ['D0', 'alpha', 'D_offset'])
            logger: Optional logger for diagnostics.

        Raises:
            ValueError: If stratified_data.chunks is empty or invalid.
        """
        self.chunks = stratified_data.chunks
        sigma_array = np.asarray(stratified_data.sigma, dtype=np.float64)
        self.sigma = sigma_array  # Keep numpy view for legacy paths
        self._sigma_jax = jnp.asarray(sigma_array)
        self.per_angle_scaling = per_angle_scaling
        self.physical_param_names = physical_param_names
        self.logger = logger or logging.getLogger(__name__)

        if not self.chunks:
            raise ValueError("stratified_data.chunks is empty")

        self.n_chunks = len(self.chunks)
        self.n_total_points = sum(len(chunk.g2) for chunk in self.chunks)

        # Determine number of unique angles from first chunk
        self.n_phi = len(np.unique(self.chunks[0].phi))

        # Determine expected parameter structure
        # Per-angle: [contrast_0, ..., contrast_{n-1}, offset_0, ..., offset_{n-1}, *physical]
        # Legacy: [contrast, offset, *physical]
        if per_angle_scaling:
            self.n_scaling_params = 2 * self.n_phi
        else:
            self.n_scaling_params = 2

        self.n_physical_params = len(physical_param_names)
        self.n_total_params = self.n_scaling_params + self.n_physical_params

        # Pre-compute unique values for each chunk (avoid jnp.unique in JIT)
        self._precompute_chunk_metadata()

        # Setup JIT-compiled functions
        self._setup_jax_functions()

        # Pre-convert chunk arrays to JAX (avoid jnp.asarray in loop)
        self._preconvert_chunk_arrays()

        self.logger.info(
            f"StratifiedResidualFunction initialized: "
            f"{self.n_chunks} chunks, {self.n_total_points:,} total points, "
            f"n_phi={self.n_phi}, per_angle_scaling={self.per_angle_scaling}, "
            f"n_scaling_params={self.n_scaling_params}, n_physical_params={self.n_physical_params}"
        )

    def _precompute_chunk_metadata(self):
        """
        Pre-compute GLOBAL unique values from ALL chunks to avoid jnp.unique() in JIT.

        This method extracts unique phi, t1, t2 values from ALL chunks combined
        and stores them as metadata. Each chunk gets the SAME global unique arrays
        to ensure correct flat indexing when accessing sigma_full array.

        This avoids ConcretizationTypeError when using jnp.unique() inside
        JIT-compiled functions.

        CRITICAL: Must use global unique values, not per-chunk subsets, because
        sigma_full dimensions are based on ALL data points across all chunks.
        """
        # Extract GLOBAL unique values from ALL chunks combined
        # This ensures grid dimensions match sigma_full dimensions
        all_phi = np.concatenate([chunk.phi for chunk in self.chunks])
        all_t1 = np.concatenate([chunk.t1 for chunk in self.chunks])
        all_t2 = np.concatenate([chunk.t2 for chunk in self.chunks])

        global_phi_unique = jnp.sort(jnp.unique(jnp.asarray(all_phi)))
        global_t1_unique = jnp.sort(jnp.unique(jnp.asarray(all_t1)))
        global_t2_unique = jnp.sort(jnp.unique(jnp.asarray(all_t2)))

        self.logger.debug(
            f"Global unique values extracted from all chunks: "
            f"{len(global_phi_unique)} phi, "
            f"{len(global_t1_unique)} t1, "
            f"{len(global_t2_unique)} t2"
        )

        # Store SAME global unique arrays for ALL chunks
        # This ensures flat indexing calculations use correct dimensions
        self.chunk_metadata = []
        for _chunk in self.chunks:
            metadata = {
                "phi_unique": global_phi_unique,  # Same for all chunks
                "t1_unique": global_t1_unique,  # Same for all chunks
                "t2_unique": global_t2_unique,  # Same for all chunks
            }
            self.chunk_metadata.append(metadata)

    def _setup_jax_functions(self):
        """
        Pre-compile JAX functions for performance.

        This method sets up JIT-compiled versions of the residual computation
        to maximize performance during optimization.
        """
        self.compute_chunk_jit = jax.jit(self._compute_chunk_residuals_raw)
        self.logger.debug("JAX functions JIT-compiled successfully")

    def _preconvert_chunk_arrays(self):
        """
        Pre-convert chunk arrays to JAX arrays during initialization.

        This avoids repeated jnp.asarray() calls inside the optimization loop,
        providing ~10-15% speedup by eliminating array conversion overhead.
        """
        self.chunks_jax = []
        for chunk in self.chunks:
            chunk_jax = {
                "phi": jnp.asarray(chunk.phi),
                "t1": jnp.asarray(chunk.t1),
                "t2": jnp.asarray(chunk.t2),
                "g2": jnp.asarray(chunk.g2),
                "q": float(chunk.q),
                "L": float(chunk.L),
                "dt": float(chunk.dt) if chunk.dt is not None else None,
            }
            self.chunks_jax.append(chunk_jax)
        self.logger.debug(f"Pre-converted {len(self.chunks_jax)} chunks to JAX arrays")

    def _compute_chunk_residuals_raw(
        self,
        phi: jnp.ndarray,
        t1: jnp.ndarray,
        t2: jnp.ndarray,
        g2_obs: jnp.ndarray,
        sigma_full: jnp.ndarray,
        params_all: jnp.ndarray,
        phi_unique: jnp.ndarray,
        t1_unique: jnp.ndarray,
        t2_unique: jnp.ndarray,
        q: float,
        L: float,
        dt: float | None,
    ) -> jnp.ndarray:
        """
        Raw chunk residual computation (JIT-compiled).

        This method is JIT-compiled and called for each chunk during optimization.
        It computes weighted residuals: (g2_obs - g2_theory) / sigma

        The computation follows the same logic as _create_residual_function in nlsq_wrapper.py,
        but works with a single chunk at a time.

        Args:
            phi: Angular positions for this chunk (radians)
            t1: Time delays for this chunk (seconds)
            t2: Time delays for this chunk (seconds)
            g2_obs: Observed g2 values for this chunk
            sigma_full: Full sigma array (3D: phi × t1 × t2) - will be indexed
            params_all: All parameters [scaling_params, physical_params]
            phi_unique: Pre-computed unique phi values (avoid jnp.unique in JIT)
            t1_unique: Pre-computed unique t1 values (avoid jnp.unique in JIT)
            t2_unique: Pre-computed unique t2 values (avoid jnp.unique in JIT)
            q: Wave vector magnitude (1/Å)
            L: Sample-to-detector distance (mm)
            dt: Time step (seconds), optional

        Returns:
            Weighted residuals: (g2_obs - g2_theory) / sigma
        """

        # Extract scaling and physical parameters
        if self.per_angle_scaling:
            # Per-angle mode: params = [contrast_0, ..., contrast_{n-1}, offset_0, ..., offset_{n-1}, *physical]
            contrast = params_all[: self.n_phi]  # Shape: (n_phi,)
            offset = params_all[self.n_phi : 2 * self.n_phi]  # Shape: (n_phi,)
            physical_params = params_all[2 * self.n_phi :]
        else:
            # Legacy mode: params = [contrast, offset, *physical]
            contrast = params_all[0]  # Scalar
            offset = params_all[1]  # Scalar
            physical_params = params_all[2:]

        # Compute theoretical g2 using vectorized computation over phi angles
        if self.per_angle_scaling:
            # Vectorize over phi with corresponding contrast/offset
            compute_g2_vmap = jax.vmap(
                lambda phi_val, contrast_val, offset_val: jnp.squeeze(
                    compute_g2_scaled(
                        params=physical_params,
                        t1=t1_unique,
                        t2=t2_unique,
                        phi=phi_val,
                        q=q,
                        L=L,
                        contrast=contrast_val,
                        offset=offset_val,
                        dt=dt,
                    ),
                    axis=0,
                ),
                in_axes=(0, 0, 0),
            )
            g2_theory_grid = compute_g2_vmap(
                phi_unique, contrast, offset
            )  # Shape: (n_phi, n_t1, n_t2)
        else:
            # Legacy: single contrast/offset for all angles
            compute_g2_vmap = jax.vmap(
                lambda phi_val: jnp.squeeze(
                    compute_g2_scaled(
                        params=physical_params,
                        t1=t1_unique,
                        t2=t2_unique,
                        phi=phi_val,
                        q=q,
                        L=L,
                        contrast=contrast,
                        offset=offset,
                        dt=dt,
                    ),
                    axis=0,
                ),
                in_axes=0,
            )
            g2_theory_grid = compute_g2_vmap(phi_unique)  # Shape: (n_phi, n_t1, n_t2)

        # Apply diagonal correction (matches experimental data preprocessing)
        apply_diagonal_vmap = jax.vmap(apply_diagonal_correction, in_axes=0)
        g2_theory_grid = apply_diagonal_vmap(g2_theory_grid)

        # Flatten theory grid
        g2_theory_flat = g2_theory_grid.flatten()

        # Create index mapping from chunk points to grid points
        # For each (phi, t1, t2) in chunk, find its index in the flattened grid
        # CRITICAL FIX: Clip indices to valid range to prevent out-of-bounds access
        # searchsorted returns len(array) when value >= max, which is out of bounds
        phi_indices = jnp.clip(
            jnp.searchsorted(phi_unique, phi), 0, len(phi_unique) - 1
        )
        t1_indices = jnp.clip(
            jnp.searchsorted(t1_unique, t1), 0, len(t1_unique) - 1
        )
        t2_indices = jnp.clip(
            jnp.searchsorted(t2_unique, t2), 0, len(t2_unique) - 1
        )

        # Convert to flat grid indices
        n_t1 = len(t1_unique)
        n_t2 = len(t2_unique)
        flat_indices = phi_indices * (n_t1 * n_t2) + t1_indices * n_t2 + t2_indices

        # Extract theory values for chunk points
        g2_theory_chunk = g2_theory_flat[flat_indices]

        # Get sigma values for chunk points (same indexing)
        sigma_flat = sigma_full.flatten()
        sigma_chunk = sigma_flat[flat_indices]

        # Compute weighted residuals
        EPS = 1e-10
        residuals = (g2_obs - g2_theory_chunk) / (sigma_chunk + EPS)

        return residuals

    def _call_jax(self, params: jnp.ndarray) -> jnp.ndarray:
        """JAX-native residuals for use in JIT/Jacobian contexts."""
        params_jax = jnp.asarray(params)
        sigma_full = self._sigma_jax
        residuals = []
        for i, chunk_jax in enumerate(self.chunks_jax):
            metadata = self.chunk_metadata[i]
            # Use pre-converted JAX arrays (avoid jnp.asarray overhead in loop)
            chunk_residuals = self.compute_chunk_jit(
                phi=chunk_jax["phi"],
                t1=chunk_jax["t1"],
                t2=chunk_jax["t2"],
                g2_obs=chunk_jax["g2"],
                sigma_full=sigma_full,
                params_all=params_jax,
                phi_unique=metadata["phi_unique"],
                t1_unique=metadata["t1_unique"],
                t2_unique=metadata["t2_unique"],
                q=chunk_jax["q"],
                L=chunk_jax["L"],
                dt=chunk_jax["dt"],
            )
            residuals.append(chunk_residuals)
        return jnp.concatenate(residuals, axis=0)

    def jax_residual(self, params: jnp.ndarray) -> jnp.ndarray:
        return self._call_jax(params)

    def __call__(self, params: np.ndarray) -> np.ndarray:
        params_jax = jnp.asarray(params)
        residuals_jax = self._call_jax(params_jax)
        return np.asarray(np.array(residuals_jax))

    def validate_chunk_structure(self) -> bool:
        """
        Validate that all chunks contain all phi angles.

        This is a critical validation to ensure per-angle parameter gradients
        will be non-zero. If any chunk is missing an angle, the gradient for
        that angle's parameters will be zero, causing optimization failure.

        Returns:
            True if validation passes

        Raises:
            ValueError: If any chunk is missing angles or has inconsistent structure
        """
        if not self.chunks:
            raise ValueError("No chunks to validate")

        # Get expected angles from first chunk
        expected_angles = set(np.unique(np.round(self.chunks[0].phi, decimals=6)))
        n_expected = len(expected_angles)

        self.logger.info(
            f"Validating chunk structure: {self.n_chunks} chunks, "
            f"{n_expected} expected angles per chunk"
        )

        # Validate each chunk
        for i, chunk in enumerate(self.chunks):
            chunk_angles = set(np.unique(np.round(chunk.phi, decimals=6)))

            # Check angle completeness
            if chunk_angles != expected_angles:
                missing = expected_angles - chunk_angles
                extra = chunk_angles - expected_angles
                error_msg = f"Chunk {i} has inconsistent angles:\n"
                if missing:
                    error_msg += f"  Missing: {missing}\n"
                if extra:
                    error_msg += f"  Extra: {extra}\n"
                raise ValueError(error_msg)

            # Check for valid data
            if len(chunk.g2) == 0:
                raise ValueError(f"Chunk {i} has no data points")

            # Check array shapes match
            # Note: sigma is stored at parent level (self.sigma), not in chunks
            n_points = len(chunk.g2)
            if not (len(chunk.phi) == len(chunk.t1) == len(chunk.t2) == n_points):
                raise ValueError(
                    f"Chunk {i} has inconsistent array shapes: "
                    f"phi={len(chunk.phi)}, t1={len(chunk.t1)}, "
                    f"t2={len(chunk.t2)}, g2={len(chunk.g2)}"
                )

        self.logger.info("✓ Chunk structure validation passed")
        return True

    def get_diagnostics(self) -> dict[str, Any]:
        """
        Get diagnostic information about the residual function.

        Returns:
            Dictionary containing:
                - n_chunks: Number of chunks
                - n_total_points: Total data points
                - n_angles: Number of unique phi angles
                - per_angle_scaling: Whether per-angle scaling is enabled
                - chunk_sizes: List of points per chunk
                - chunk_angle_counts: List of angles per chunk
                - min_chunk_size: Minimum chunk size
                - max_chunk_size: Maximum chunk size
                - mean_chunk_size: Mean chunk size
        """
        chunk_sizes = [len(chunk.g2) for chunk in self.chunks]
        chunk_angle_counts = [len(np.unique(chunk.phi)) for chunk in self.chunks]

        diagnostics = {
            "n_chunks": self.n_chunks,
            "n_total_points": self.n_total_points,
            "n_angles": len(np.unique(self.chunks[0].phi)),
            "per_angle_scaling": self.per_angle_scaling,
            "chunk_sizes": chunk_sizes,
            "chunk_angle_counts": chunk_angle_counts,
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "mean_chunk_size": np.mean(chunk_sizes),
        }

        return diagnostics

    def log_diagnostics(self):
        """Log diagnostic information for monitoring."""
        diag = self.get_diagnostics()
        self.logger.info(
            f"StratifiedResidualFunction diagnostics:\n"
            f"  Chunks: {diag['n_chunks']}\n"
            f"  Total points: {diag['n_total_points']:,}\n"
            f"  Angles: {diag['n_angles']}\n"
            f"  Per-angle scaling: {diag['per_angle_scaling']}\n"
            f"  Chunk sizes: min={diag['min_chunk_size']:,}, "
            f"max={diag['max_chunk_size']:,}, mean={diag['mean_chunk_size']:.0f}\n"
            f"  Angle counts per chunk: {set(diag['chunk_angle_counts'])}"
        )


def create_stratified_residual_function(
    stratified_data: Any,
    per_angle_scaling: bool,
    physical_param_names: list[str],
    logger: logging.Logger | None = None,
    validate: bool = True,
) -> StratifiedResidualFunction:
    """
    Factory function to create and validate a stratified residual function.

    This is a convenience function that creates a StratifiedResidualFunction,
    optionally validates its structure, and logs diagnostics.

    Args:
        stratified_data: Object with .chunks attribute containing angle-stratified chunks
        per_angle_scaling: Whether per-angle scaling parameters are used
        physical_param_names: List of physical parameter names (e.g., ['D0', 'alpha', 'D_offset'])
        logger: Optional logger for diagnostics
        validate: Whether to validate chunk structure (recommended)

    Returns:
        Validated StratifiedResidualFunction instance

    Raises:
        ValueError: If validation fails

    Example:
        >>> residual_fn = create_stratified_residual_function(
        ...     stratified_data=stratified_data,
        ...     per_angle_scaling=True,
        ...     physical_param_names=['D0', 'alpha', 'D_offset'],
        ...     validate=True
        ... )
        >>> residual_fn.log_diagnostics()
    """
    residual_fn = StratifiedResidualFunction(
        stratified_data=stratified_data,
        per_angle_scaling=per_angle_scaling,
        physical_param_names=physical_param_names,
        logger=logger,
    )

    if validate:
        residual_fn.validate_chunk_structure()

    residual_fn.log_diagnostics()

    return residual_fn
