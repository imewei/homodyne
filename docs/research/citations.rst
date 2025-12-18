Citations
=========

This section provides citation information for the Homodyne package and the
underlying theoretical framework.

Primary Research Citation
-------------------------

If you use Homodyne in your research, please cite the following publication:

**He, H., Liang, H., Chu, M., Jiang, Z., de Pablo, J.J., Tirrell, M.V., Narayanan, S., & Chen, W.**
*Transport coefficient approach for characterizing nonequilibrium dynamics in soft matter.*
Proceedings of the National Academy of Sciences, 121(31), e2401162121 (2024).

**DOI**: `10.1073/pnas.2401162121 <https://doi.org/10.1073/pnas.2401162121>`_

BibTeX Entry
~~~~~~~~~~~~

For reference manager import:

.. code-block:: bibtex

   @article{he2024transport,
     title={Transport coefficient approach for characterizing nonequilibrium
            dynamics in soft matter},
     author={He, Hongrui and Liang, Hao and Chu, Miaoqi and Jiang, Zhang and
             de Pablo, Juan J and Tirrell, Matthew V and Narayanan, Suresh
             and Chen, Wei},
     journal={Proceedings of the National Academy of Sciences},
     volume={121},
     number={31},
     pages={e2401162121},
     year={2024},
     publisher={National Academy of Sciences},
     doi={10.1073/pnas.2401162121}
   }

Software Package Citation
-------------------------

To cite the Homodyne software package directly:

.. code-block:: bibtex

   @software{homodyne,
     title={Homodyne: JAX-first high-performance XPCS analysis},
     author={Chen, Wei and He, Hongrui},
     year={2024},
     url={https://github.com/imewei/homodyne},
     version={2.5.0},
     institution={Argonne National Laboratory}
   }

Citation Guidelines
-------------------

When citing Homodyne, we recommend:

1. **Always cite the primary PNAS paper** for the theoretical framework
2. **Optionally cite the software package** if describing computational methods
3. **Include the version number** (e.g., v2.5.0) for reproducibility
4. **Mention the DOI** in supplementary materials for persistent identification

Example Citation Text
~~~~~~~~~~~~~~~~~~~~~

In your methods section, you might write:

   "XPCS data were analyzed using the transport coefficient approach [1] as
   implemented in the Homodyne software package (v2.5.0) [2]. The analysis
   employed the laminar flow mode with per-angle scaling to account for
   instrumental variations across detector positions."

   [1] He et al., PNAS 121(31), e2401162121 (2024). DOI: 10.1073/pnas.2401162121
   [2] https://github.com/imewei/homodyne

Acknowledgments
---------------

If you use Homodyne in your research, please consider acknowledging:

**Funding Sources:**

* U.S. Department of Energy, Office of Science, Basic Energy Sciences
* Advanced Photon Source User Facility at Argonne National Laboratory

**Collaborating Institutions:**

* Argonne National Laboratory, X-ray Science Division
* University of Chicago, Pritzker School of Molecular Engineering

Contact Information
-------------------

For questions about the theoretical framework or software:

* **Principal Investigator**: Wei Chen (wchen@anl.gov)
* **Lead Developer**: Hongrui He
* **Technical Support**: GitHub Issues at https://github.com/imewei/homodyne/issues
* **Research Collaboration**: Argonne National Laboratory, X-ray Science Division

Related Publications
--------------------

The following publications provide additional context for XPCS analysis:

**XPCS Fundamentals:**

* Shpyrko, O.G. "X-ray photon correlation spectroscopy."
  Journal of Synchrotron Radiation 21.5 (2014): 1057-1064.

**Nonequilibrium Dynamics:**

* Cipelletti, L., & Ramos, L. "Slow dynamics in glassy soft matter."
  Journal of Physics: Condensed Matter 17.6 (2005): R253.

**Synchrotron Methods:**

* Sutton, M. "A review of X-ray intensity fluctuation spectroscopy."
  Comptes Rendus Physique 9.5-6 (2008): 657-667.

License
-------

Homodyne is distributed under the MIT License. See the LICENSE file in the
repository for full terms.

When redistributing or modifying the software, please maintain attribution
to the original authors and include the appropriate citations.
