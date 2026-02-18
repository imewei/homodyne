.. _data-loading-snippet:

Loading XPCS Data
~~~~~~~~~~~~~~~~~

:class:`~homodyne.data.XPCSDataLoader` reads HDF5 files and validates the
data before returning a typed container:

.. code-block:: python

   from homodyne.data import XPCSDataLoader
   from homodyne.config import ConfigManager

   config = ConfigManager.from_yaml("my_config.yaml")
   loader = XPCSDataLoader(config)
   data   = loader.load()

   # Validated fields on the returned XPCSData object:
   print(data.c2.shape)       # (n_phi, n_t, n_t)  — two-time correlation matrix
   print(data.t_grid.shape)   # (n_t,)              — time grid in seconds
   print(data.q_values.shape) # (n_q,)              — q-values in nm^-1
   print(data.phi_values)     # (n_phi,)            — azimuthal angles in radians

The loader raises ``ValueError`` if:

- ``c2`` contains NaN or Inf values.
- The time grid is not monotonically increasing.
- Array shapes are inconsistent with the configuration.

.. note::

   The loader never subsamples data. Full-precision HDF5 arrays are always
   read into memory as ``float64`` JAX arrays. See
   :class:`~homodyne.data.XPCSDataLoader` for the full API.

.. _end-data-loading-snippet:
