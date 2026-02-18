.. _contributing-overview:

Contributing to Homodyne
=========================

Thank you for your interest in contributing to Homodyne.

This page provides a brief overview.  For the full contribution guide,
development environment setup, code style standards, testing requirements,
and pull request guidelines, see the :doc:`developer/index` section.

----

Quick Links
-----------

- :doc:`developer/contributing` — development environment, commit conventions,
  pull request checklist
- :doc:`developer/testing` — test organization, running the suite, coverage
  requirements
- :doc:`developer/architecture` — code layout, design decisions, ADRs

----

How to Contribute
-----------------

1. **Report a bug** — open an issue on
   `GitHub Issues <https://github.com/imewei/homodyne/issues>`_.
2. **Request a feature** — start a
   `GitHub Discussion <https://github.com/imewei/homodyne/discussions>`_
   first so the design can be agreed upon before implementation.
3. **Submit a fix or feature** — fork the repo, create a branch, and open a
   pull request against ``main``.

All contributions must:

- Pass ``make quality`` (Black, Ruff, MyPy).
- Include tests covering the new or modified behaviour.
- Update documentation if any public API changes.

----

Code of Conduct
---------------

Homodyne follows the `Contributor Covenant Code of Conduct
<https://www.contributor-covenant.org/version/2/1/code_of_conduct/>`_.
All participants are expected to uphold these standards.
