"""Standalone EXP-012 research code.

This package must remain isolated from the production ABFE modules until the
pre-registered feature, force, and deployment gates have passed.
"""

from .schema import (
    Exp012IntegrityError,
    Exp012ProtocolError,
    Exp012Preregistration,
    load_preregistration,
    validate_preregistration,
)
from .mm_ledger import (
    analytic_ibs_bias_kj_mol,
    compose_mm_ledger_arrays,
    relabel_mm_ledger,
)
from .metrics import assess_whole_run_data_support

__all__ = [
    "Exp012IntegrityError",
    "Exp012ProtocolError",
    "Exp012Preregistration",
    "load_preregistration",
    "validate_preregistration",
    "analytic_ibs_bias_kj_mol",
    "compose_mm_ledger_arrays",
    "relabel_mm_ledger",
    "assess_whole_run_data_support",
]
