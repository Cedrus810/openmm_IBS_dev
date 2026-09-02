from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "abfe_pipeline.py"


class _Ledger:
    def __init__(self, seed, leg):
        self.seed, self.leg = seed, leg

    def derive(self, *parts):
        return self.seed + len(parts)

    def snapshot(self):
        return {"seed": self.seed, "leg": self.leg}


from abfe_pipeline import (  # noqa: E402
    _preopt_boresch_protocol_payload as _real_preopt_boresch_protocol_payload,
)


def _traditional_class(tmp_path, charge=0.0):
    tree = ast.parse(PIPELINE_PATH.read_text(encoding="utf-8"), filename=str(PIPELINE_PATH))
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TraditionalABFEPipeline"
    )
    module = ast.Module(body=[class_node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "openmm": SimpleNamespace(System=object),
        "app": SimpleNamespace(Topology=object),
        "unit": SimpleNamespace(),
        "List": List,
        "Optional": Optional,
        "Dict": Dict,
        "Any": Any,
        "np": np,
        "os": os,
        "json": json,
        "Exp019SeedLedger": _Ledger,
        # 2026-09-01：`run_leg` 的 sampling fingerprint 改用收窄口径的 boresch_params
        # （诊断字段不得进缓存键，见 test_decharging_seed_contract 里那两条）。
        # 这个合成命名空间只 exec 了类体，模块级函数要显式注入。
        "_preopt_boresch_protocol_payload": _real_preopt_boresch_protocol_payload,
        "LIGAND_NET_CHARGE_INTEGER_TOLERANCE_E": 1.0e-3,
        "_compute_ligand_net_charge": lambda _system, _indices: charge,
    }
    exec(compile(module, str(PIPELINE_PATH), "exec"), namespace)
    return namespace["TraditionalABFEPipeline"], namespace


def test_neutral_traditional_fresh_and_resume_have_complete_interfaces(tmp_path):
    cls, ns = _traditional_class(tmp_path, charge=0.0)
    pipeline = cls(
        object(), object(), object(), None, [0], output_dir=str(tmp_path),
        repeat_seed=11, leg_name="complex",
    )
    assert pipeline.resolve_co_alchemical_ion_spec() is None
    assert pipeline._seed_for("charging", "stage", "exchange", "numpy") is not None
    assert pipeline.seed_contract_snapshot() == {"seed": 11, "leg": "complex"}

    calls = []

    class _REMD:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def run(self, **_kwargs):
            return [str(tmp_path / "r0.dcd"), str(tmp_path / "r1.dcd")]

    class _Analyzer:
        def __init__(self, temperature):
            self.temperature = temperature
            self._last_n_k = np.array([2, 2])
            self._last_lj_lrc_metadata = {"enabled": False}

        def compute_u_kn(self, **_kwargs):
            return np.zeros((2, 4))

        def solve(self, _u_kn):
            return {"total_delta_G": 0.0, "diagnostics": {}}

    def _fingerprint(payload):
        encoded = json.dumps(payload, sort_keys=True, default=str)
        return {"sha256": str(abs(hash(encoded))), "payload": payload}

    ns.update(
        {
            "print": lambda *args, **kwargs: None,
            "REMDManager": _REMD,
            "TraditionalMBARAnalyzer": _Analyzer,
            "_expected_remd_traj_files": lambda directory, stage, n: [
                os.path.join(directory, f"{stage}_rep{i}.dcd") for i in range(n)
            ],
            "_expected_remd_frame_count": lambda _steps: 1,
            "_all_remd_trajs_valid": lambda *args, **kwargs: False,
            "_protocol_fingerprint": _fingerprint,
            "_protocol_fingerprint_ignoring_code_hash": lambda value: value,
            "_system_xml_hash": lambda _value: "system",
            "_topology_hash": lambda _value: "topology",
            "_positions_hash": lambda _value: "positions",
            "_lambda_signature": lambda value: list(value),
            "TRADITIONAL_LJ_LRC_PROTOCOL_VERSION": 1,
            "PME_DECHARGE_MODEL_VERSION": "test",
            "constraint_identity_fingerprint": lambda *_: {"test": True},
        }
    )

    fresh = pipeline.run_leg("stage", [1.0, 0.0], [1.0, 0.0], n_steps=10)
    assert fresh["total_delta_G"] == 0.0
    assert calls[0]["co_alchemical_ion_spec"] is None
    assert calls[0]["seed_ledger"] is pipeline.seed_ledger
    assert calls[0]["seed_leg"] == "complex"

    resumed = pipeline.run_leg(
        "stage", [1.0, 0.0], [1.0, 0.0], n_steps=10, resume=True
    )
    assert resumed["total_delta_G"] == 0.0
    assert len(calls) == 1, "valid u_kn resume should not start a second REMD run"


def test_charged_traditional_route_fails_before_any_context_can_be_created(tmp_path):
    cls, _ = _traditional_class(tmp_path, charge=1.0)
    with pytest.raises(RuntimeError, match="只支持中性配体"):
        cls(object(), object(), object(), None, [0], output_dir=str(tmp_path / "charged"))
    assert not (tmp_path / "charged").exists()
