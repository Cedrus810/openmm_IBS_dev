"""Local-residual path potential: the production-facing subset.

This package intentionally exposes **no** re-exports.  Import the module you
need explicitly, e.g.::

    from local_residual.openmm_plugin import LocalManyBodyResidualPlugin
    from local_residual import em_no_residual

Rationale (2026-08-31 release cleanup): this ``__init__`` previously did
``from .schema import *`` / ``.mm_ledger`` / ``.ledger_audit`` / ``.metrics``
/ ``.softlift*``.  Because ``runabfe`` imports ``local_residual.openmm_plugin``,
Python executed this file first and dragged the whole EXP-012 research stack
(softlift training/deploy, dataset builders, and the ``exp012_xed`` namespace)
into the production start-up path.  None of it is reachable from the ABFE
entry point.  Keeping this file empty is what makes that research code
separable from the shipped product.
"""
