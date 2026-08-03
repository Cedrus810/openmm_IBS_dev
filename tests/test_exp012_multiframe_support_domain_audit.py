"""DEC-030 step (a): report-only multi-frame support-domain audit.

Only ``audit_frame`` is exercised here with a synthetic system -- the CLI's
preregistration/trajectory plumbing is identical to
scripts/run_exp012_mm_ledger.py and scripts/smoke_exp012_mace_latent.py and is
not re-tested. The behavior worth pinning down is that a fixed atom set
omitting a real cutoff-graph neighbor is *reported*, never raised.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

# ``sys.modules`` is populated before ``exec_module`` (the documented pattern
# for importing a source file directly) so a function defined in this module
# -- e.g. its ProcessPoolExecutor worker -- can be pickled by module+qualname
# and found by a child process, should a future test exercise it directly.
_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_exp012_multiframe_support_domain.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "audit_exp012_multiframe_support_domain", _MODULE_PATH
)
_audit_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _audit_module
_SPEC.loader.exec_module(_audit_module)

audit_frame = _audit_module.audit_frame

pytestmark = pytest.mark.cpu_only


def _line_positions(count: int, spacing: float):
    return torch.tensor(
        [[index * spacing, 0.0, 0.0] for index in range(count)], dtype=torch.float64
    )


def _cell(side: float):
    return torch.eye(3, dtype=torch.float64) * side


def test_no_violation_when_fixed_set_covers_true_closure():
    # 0 = ligand seed; 1 is one hop away; 2 is a second hop away from 1 (not
    # directly reachable from 0 within the cutoff).
    positions = _line_positions(3, 2.0)
    cell = _cell(30.0)
    result = audit_frame(
        positions, cell,
        ligand_indices=[0],
        fixed_atom_indices={0, 1, 2},
        edge_cutoff_angstrom=3.0,
        interaction_layers=2,
    )
    assert result["closure_atom_count"] == 3
    assert result["omitted_atom_count"] == 0
    assert result["omitted_atom_indices"] == []


def test_violation_is_reported_not_raised():
    # Same true closure as above, but the fixed set omits atom 2 -- exactly
    # the DEC-030(a) scenario a frame0-derived manifest could hit at a later
    # frame. This must be reported, never raised: the audit is non-blocking.
    positions = _line_positions(3, 2.0)
    cell = _cell(30.0)
    result = audit_frame(
        positions, cell,
        ligand_indices=[0],
        fixed_atom_indices={0, 1},
        edge_cutoff_angstrom=3.0,
        interaction_layers=2,
    )
    assert result["closure_atom_count"] == 3
    assert result["omitted_atom_count"] == 1
    assert result["omitted_atom_indices"] == [2]
    assert result["omitted_atom_hops"] == [2]
