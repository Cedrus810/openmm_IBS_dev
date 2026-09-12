#!/usr/bin/env python
"""把一份训练好的 R1 payload/weights 封成可部署的 resource manifest。

这一步以前没有脚本 —— Atenolol 那份 `resources/outer_lambda_local_residual/
manifest.json` 是手写的。换配体重训之后必须重新生成，而 manifest 里三样东西
手算都容易错：配体的原子序数序列、内部键图、以及两者的 canonical-JSON 指纹。
这里直接复用 loader 自己的 `ligand_chemical_identity()` 算，保证"写进去的"和
"运行时比对的"永远是同一份实现。

用法::

    python scripts/write_local_residual_resource_manifest.py \\
        --payload  resources/outer_lambda_local_residual/r1_model_payload_v1.json \\
        --weights  resources/outer_lambda_local_residual/r1_model_weights_f64.bin \\
        --topology output/topology.cif \\
        --ligand-indices output/ligand_indices.json \\
        --system   output/system_native.xml \\
        --ligand-name Atenolol \\
        --output   resources/outer_lambda_local_residual/manifest.json

`--system` 可省但强烈建议给：mmCIF 对非标准残基不保留键，缺 System 时内部键图
只能从拓扑图推，可能少键（见 `ligand_chemical_identity` 的说明）。

写完会立刻用 loader 的 `_load_resource_manifest()` 回读校验一遍，任何一项对不上
就 fail-closed 删掉半成品，不留一份"能写出来但加载不了"的 manifest。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_residual.openmm_plugin import (  # noqa: E402
    FEATURE_NAME,
    FROZEN_CANDIDATE_LIST_CAPACITY,
    FROZEN_SKIN_ANGSTROM,
    KNOWN_PLUGIN_SOURCE_SHA256,
    LIGAND_IDENTITY_PROTOCOL,
    RESOURCE_MANIFEST_VERSION,
    SCHEMA_VERSION,
    _load_resource_manifest,
    ligand_chemical_identity,
    load_r1_payload,
    sha256_file,
)


def _read_ligand_indices(path: Path) -> list[int]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    indices = doc["ligand_indices"] if isinstance(doc, dict) else doc
    return [int(value) for value in indices]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--topology", required=True, type=Path, help="PDBx/mmCIF")
    parser.add_argument("--ligand-indices", required=True, type=Path)
    parser.add_argument("--system", type=Path, help="serialized System XML；不给则内部键图只能从拓扑推")
    parser.add_argument("--ligand-name", required=True)
    parser.add_argument("--experiment-id", default="EXP-020")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    from openmm import XmlSerializer, app

    payload = load_r1_payload(args.payload, args.weights)
    topology = app.PDBxFile(str(args.topology)).topology
    ligand_indices = _read_ligand_indices(args.ligand_indices)
    system = (
        XmlSerializer.deserialize(args.system.read_text(encoding="utf-8"))
        if args.system is not None
        else None
    )

    identity = ligand_chemical_identity(topology, ligand_indices, system=system)
    if len(ligand_indices) != len(payload.ligand_topology_indices):
        raise SystemExit(
            f"配体原子数与 payload 不一致: topology={len(ligand_indices)} "
            f"payload={len(payload.ligand_topology_indices)}"
        )

    manifest = {
        "manifest_version": RESOURCE_MANIFEST_VERSION,
        "feature": FEATURE_NAME,
        "model_name": f"exp025_local_manybody_residual_r1_{args.ligand_name.lower()}",
        "supported_ligand": {
            "name": args.ligand_name,
            "identity_protocol": LIGAND_IDENTITY_PROTOCOL,
            "atom_count": identity["atom_count"],
            "atomic_numbers": identity["atomic_numbers"],
            "internal_bonds": identity["internal_bonds"],
            "fingerprint_sha256": identity["fingerprint_sha256"],
        },
        "payload": {
            "path": args.payload.name,
            "sha256": sha256_file(args.payload),
        },
        "weights": {
            "path": args.weights.name,
            "sha256": sha256_file(args.weights),
        },
        "training": {
            "experiment_id": args.experiment_id,
            "source_checkpoint_sha256": payload.source_checkpoint_sha256,
        },
        "plugin": {
            "source_sha256": KNOWN_PLUGIN_SOURCE_SHA256,
            "schema_version": SCHEMA_VERSION,
            "skin_angstrom": FROZEN_SKIN_ANGSTROM,
            "candidate_list_capacity": FROZEN_CANDIDATE_LIST_CAPACITY,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # payload/weights 的相对路径是相对 manifest 所在目录解析的，所以回读校验
    # 顺带把"三个文件放在同一个目录里"这件事也验了。
    try:
        _load_resource_manifest(args.output)
    except Exception:
        args.output.unlink(missing_ok=True)
        raise

    print(f"写入并回读校验通过: {args.output}")
    print(f"  配体 {args.ligand_name}: {identity['atom_count']} 原子, "
          f"指纹 {identity['fingerprint_sha256'][:16]}...")


if __name__ == "__main__":
    main()
