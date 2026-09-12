#!/usr/bin/env python
"""EXP-025 G1: export the frozen EXP-020 canonical R1 checkpoint to an
explicit, PyTorch-free float64 payload for the C++ Reference oracle.

This is deliberately NOT a general-purpose exporter. It is hard-pinned to the
one canonical checkpoint that D2/D3 already used
(``r1__direct_gap__fold_test_run1__seed0.pt``) and fails closed on any
mismatch -- there is no "pick the best of 9 checkpoints" behavior here, by
design (see PLAN_EXP-025_local_manybody_cuda.md G5 for why that choice is
deferred, not made ad hoc in G1).

Produces, under output/outer_lambda_exp025_local_manybody_cuda/g1_reference/:
  - r1_model_payload_v1.json     tensor contract: name/shape/dtype/offset/
                                  byte_count/sha256 for every parameter, plus
                                  the frozen config and architecture facts
                                  that are NOT part of SoftLiftConfig
                                  (hidden_rho=16, the exact MLP layer shape).
  - r1_model_weights_f64.bin     flat, little-endian float64 blob; tensors
                                  concatenated in the order listed in the
                                  payload JSON.
  - r1_model_manifest.json       provenance/verification wrapper: checkpoint
                                  identity, round-trip verification result,
                                  exporter script hash.
  - canonical_fixture_manifest.json
                                  the first real G1 test frame (topology,
                                  trajectory, frame index, ligand indices,
                                  active edge count) re-verified against
                                  output/outer_lambda_exp020_softlift_seedfixed001/d2/r1_d2_manifest.json,
                                  not just copied from it.

Round-trip verification (step 7 of the spec): after writing the payload, this
script rebuilds a SECOND, independent model purely from the exported JSON +
binary blob (never touching the original torch.load()'d state_dict object)
and confirms it reproduces the checkpoint-loaded model's output on the
canonical fixture frame bit-for-bit (they are the same float64 bytes, so
exact equality is the correct bar, not a tolerance).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_residual.geometry import ligand_environment_cross_edges  # noqa: E402
from local_residual.softlift import (  # noqa: E402
    PackedSoftLiftBatch,
    SoftLiftConfig,
    build_softlift_model,
    primary_r1_config,
)

EXPERIMENT_ID = "EXP-020"
HIDDEN_RHO = 16  # local_residual/softlift.py R1Model.__init__ default; not part of SoftLiftConfig.
SCHEMA_VERSION = "exp025-g1-r1-reference-payload-v1"

CANONICAL_CHECKPOINT = ROOT / "output/outer_lambda_exp020_softlift_seedfixed001/r1_density/r1__direct_gap__fold_test_run1__seed0.pt"
CANONICAL_CHECKPOINT_SHA256 = "8d7be055e16d9446d3ad126e08f35f54f658e7e6b0c578a788329c1e657c520c"

CANONICAL_LIGAND_INDICES = ROOT / "output/ligand_indices.json"
CANONICAL_LIGAND_INDICES_SHA256 = "0e944fb400bcc2a607084f6e9a457f21af54a72f5468aa705a2e28b681632c04"

CANONICAL_DATASET = ROOT / "output/outer_lambda_exp020_softlift/dataset/softlift_dataset_v1.npz"
CANONICAL_DATASET_SHA256 = "24e5ce7ceb995b67ceddb08e5c7e5991c9a904356be743aecf55dc8d4ae260ab"

CANONICAL_TOPOLOGY = ROOT / "output_lrc_fix/topology.cif"
CANONICAL_TOPOLOGY_SHA256 = "6602f537d13179fc8294bcbaea1c7247fa9148b7d372a6411bc9f705db744ccf"

CANONICAL_TRAJECTORY = ROOT / "output/outer_lambda_slow_variable_screen/hard_window0_run1/scratch_sample/hard_window_screening.dcd"
CANONICAL_TRAJECTORY_SHA256 = "47d2fca0d4189ec7eb5d5e6743406162494cc1aaf25ed4a6aeae6d0c75df3b11"

CANONICAL_FRAME = 0
CANONICAL_ATOM_COUNT = 73536  # 原来写死在 _canonical_batch 里的那个数
CANONICAL_FD_ATOM = 4583
EXPECTED_ACTIVE_EDGES = 1206

OUTPUT_DIR = ROOT / "output/outer_lambda_exp025_local_manybody_cuda/g1_reference"


@dataclass(frozen=True)
class ExportInputs:
    """一次导出的全部输入。默认值 = EXP-020 Atenolol 那套 canonical 产物。

    2026-09-10：本脚本原来把这些**全部写死**成模块常量（"deliberately NOT a
    general-purpose exporter"）。训练栈搬进主线后，新配体重训完必须能用同一段
    代码导出自己的 payload，所以改成参数化 —— 但默认值原样保留，不传任何参数
    时跑的还是同一条冻结路径、逐字节同一个产物。

    `*_sha256` 传 None 表示"这是新产物，没有事先约定的 sha"：只记录实际值，
    不做比对。已知产物一律带上 sha，那条 fail-closed 的门不能省。
    """

    checkpoint: Path = CANONICAL_CHECKPOINT
    checkpoint_sha256: str | None = CANONICAL_CHECKPOINT_SHA256
    ligand_indices: Path = CANONICAL_LIGAND_INDICES
    ligand_indices_sha256: str | None = CANONICAL_LIGAND_INDICES_SHA256
    dataset: Path = CANONICAL_DATASET
    dataset_sha256: str | None = CANONICAL_DATASET_SHA256
    topology: Path = CANONICAL_TOPOLOGY
    topology_sha256: str | None = CANONICAL_TOPOLOGY_SHA256
    trajectory: Path = CANONICAL_TRAJECTORY
    trajectory_sha256: str | None = CANONICAL_TRAJECTORY_SHA256
    frame: int = CANONICAL_FRAME
    fd_atom: int = CANONICAL_FD_ATOM
    expected_active_edges: int | None = EXPECTED_ACTIVE_EDGES
    expected_atom_count: int | None = CANONICAL_ATOM_COUNT
    output_dir: Path = OUTPUT_DIR
    allow_derived_r1_config: bool = False
    system: Path | None = None


class ExporterError(RuntimeError):
    """Raised when any fail-closed identity or shape check fails."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_file_sha256(path: Path, expected: str | None, label: str) -> str:
    if not path.is_file():
        raise ExporterError(f"{label}: missing file {path}")
    actual = _sha256_file(path)
    if expected is None:
        print(f"{label}: 无预期 sha256，记录实际值 {actual}")
        return actual
    if actual != expected:
        raise ExporterError(
            f"{label}: sha256 mismatch for {path}\n  expected {expected}\n  actual   {actual}"
        )
    return actual


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _load_checkpoint(inputs: ExportInputs):
    import torch

    checkpoint_sha = _require_file_sha256(inputs.checkpoint, inputs.checkpoint_sha256, "checkpoint")
    payload = torch.load(inputs.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ExporterError(f"checkpoint experiment_id is {payload.get('experiment_id')!r}, expected {EXPERIMENT_ID!r}")
    raw_config = dict(payload["config"])
    raw_config["type_vocabulary"] = tuple(int(value) for value in raw_config["type_vocabulary"])
    config = SoftLiftConfig(**raw_config)
    if config.rung != "R1":
        raise ExporterError(f"checkpoint rung is {config.rung!r}, expected 'R1'")
    frozen = primary_r1_config(config.protocol_sha256)
    if config != frozen:
        # 出厂 config 是 41 原子 Atenolol 的常量，换配体必然走到这里。默认仍然拒绝
        # （EXP-020 的复现路径不能被悄悄改架构），显式要求时只校验**架构**字段，
        # 允许体系相关的四项不同 —— 与 train_exp019_softlift_loro 的同名开关一致。
        if not inputs.allow_derived_r1_config:
            raise ExporterError(
                "checkpoint config does not exactly match local_residual.softlift.primary_r1_config():\n"
                f"  checkpoint: {config}\n  frozen R1 : {frozen}\n"
                "（换配体重训请加 --allow-derived-r1-config）"
            )
        architecture_fields = (
            "schema_version", "rung", "n_radial_basis", "n_channels", "pair_dim",
            "context_dim", "inner_cutoff_angstrom", "outer_cutoff_angstrom",
            "no_contact_output",
        )
        differing = [
            name for name in architecture_fields
            if getattr(config, name) != getattr(frozen, name)
        ]
        if differing:
            raise ExporterError(
                "checkpoint config 的**架构**字段与冻结 R1 不一致，这不是"
                f"换配体能解释的差异: {differing}\n"
                f"  checkpoint: {config}\n  frozen R1 : {frozen}"
            )
        print(
            "checkpoint config 是按体系推导的 R1（架构一致）: "
            f"n_ligand_atoms={config.n_ligand_atoms} "
            f"type_vocabulary={list(config.type_vocabulary)} "
            f"b_max_reduced={config.b_max_reduced}"
        )
    return payload, config, checkpoint_sha


def _expected_state_dict_keys(config: SoftLiftConfig) -> set[str]:
    model = build_softlift_model(config).double()
    return set(model.state_dict().keys())


def _verify_dataset_identity(config: SoftLiftConfig, inputs: ExportInputs) -> dict:
    import numpy as np

    dataset_sha = _require_file_sha256(inputs.dataset, inputs.dataset_sha256, "dataset")
    with np.load(inputs.dataset, allow_pickle=False) as dataset:
        dataset_vocab = tuple(int(value) for value in dataset["type_vocabulary"])
        if dataset_vocab != config.type_vocabulary:
            raise ExporterError(
                f"dataset type_vocabulary {dataset_vocab} disagrees with checkpoint config {config.type_vocabulary}"
            )
        ligand_topology_indices = [int(value) for value in dataset["ligand_topology_indices"]]

    indices_sha = _require_file_sha256(inputs.ligand_indices, inputs.ligand_indices_sha256, "ligand_indices.json")
    ligand_indices = [int(value) for value in json.loads(inputs.ligand_indices.read_text())["ligand_indices"]]
    if ligand_indices != ligand_topology_indices:
        raise ExporterError(f"{inputs.ligand_indices} disagrees with dataset ligand_topology_indices")
    if len(ligand_indices) != config.n_ligand_atoms:
        raise ExporterError(f"ligand index count {len(ligand_indices)} != config.n_ligand_atoms {config.n_ligand_atoms}")
    return {
        "ligand_topology_indices": ligand_indices,
        "type_vocabulary": list(dataset_vocab),
        "dataset_sha256": dataset_sha,
        "ligand_indices_sha256": indices_sha,
    }


def _build_tensor_manifest(state_dict) -> tuple[list[dict], bytes]:
    blob = bytearray()
    manifest = []
    for name in sorted(state_dict.keys()):
        tensor = state_dict[name]
        array = tensor.detach().cpu().numpy().astype("<f8", copy=True)
        if not array.flags["C_CONTIGUOUS"]:
            array = array.copy(order="C")
        raw = array.tobytes()
        offset = len(blob)
        blob.extend(raw)
        manifest.append({
            "name": name,
            "shape": list(array.shape),
            "dtype": "float64_little_endian",
            "byte_offset": offset,
            "byte_count": len(raw),
            "sha256": _sha256_bytes(raw),
        })
    return manifest, bytes(blob)


def _rebuild_state_dict_from_payload(config: SoftLiftConfig, tensor_manifest: list[dict], blob: bytes):
    import numpy as np
    import torch

    state_dict = {}
    for entry in tensor_manifest:
        raw = blob[entry["byte_offset"]: entry["byte_offset"] + entry["byte_count"]]
        if _sha256_bytes(raw) != entry["sha256"]:
            raise ExporterError(f"tensor {entry['name']}: sha256 mismatch reading back from blob")
        array = np.frombuffer(raw, dtype="<f8").reshape(entry["shape"]).copy()
        state_dict[entry["name"]] = torch.from_numpy(array)
    return state_dict


def _canonical_batch(config: SoftLiftConfig, ligand_indices: list[int], inputs: ExportInputs):
    import mdtraj
    import torch

    topology_sha = _require_file_sha256(inputs.topology, inputs.topology_sha256, "topology")
    trajectory_sha = _require_file_sha256(inputs.trajectory, inputs.trajectory_sha256, "trajectory")

    trajectory = mdtraj.load_frame(str(inputs.trajectory), index=inputs.frame, top=str(inputs.topology))
    if trajectory.unitcell_vectors is None:
        raise ExporterError("canonical fixture frame has no periodic box")
    positions = torch.tensor(trajectory.xyz[0] * 10.0, dtype=torch.float64)
    box = torch.tensor(trajectory.unitcell_vectors[0] * 10.0, dtype=torch.float64)
    # ⚠️ mdtraj 对读不出元素的原子给 `element.virtual`（atomic_number = 0）且**不抛错**。
    #   4W53 的 45 个钠在 topology.cif 里就是 `type_symbol = ?`，走这条会静默变成 0，
    #   然后在下面的 type_map 里 KeyError。给了 --system 时用 System 的质量做第二来源
    #   把元素补回来（见 local_residual.openmm_plugin.topology_atomic_numbers）。
    if inputs.system is not None:
        from openmm import XmlSerializer, app

        from local_residual.openmm_plugin import topology_atomic_numbers

        openmm_topology = app.PDBxFile(str(inputs.topology)).topology
        openmm_system = XmlSerializer.deserialize(Path(inputs.system).read_text(encoding="utf-8"))
        atomic_numbers = topology_atomic_numbers(openmm_topology, system=openmm_system)
        if len(atomic_numbers) != trajectory.topology.n_atoms:
            raise ExporterError(
                f"--system 推出的原子数({len(atomic_numbers)})与轨迹拓扑"
                f"({trajectory.topology.n_atoms})不一致"
            )
    else:
        atomic_numbers = [int(atom.element.atomic_number) for atom in trajectory.topology.atoms]
    if inputs.expected_atom_count is not None and len(atomic_numbers) != inputs.expected_atom_count:
        raise ExporterError(
            f"canonical fixture atom count is {len(atomic_numbers)}, expected {inputs.expected_atom_count}"
        )

    ligand_set = set(ligand_indices)
    environment_indices = [index for index in range(len(atomic_numbers)) if index not in ligand_set]
    ligand = torch.tensor(ligand_indices, dtype=torch.int64)
    environment = torch.tensor(environment_indices, dtype=torch.int64)
    edges = ligand_environment_cross_edges(
        positions, box, ligand, environment, outer_cutoff=config.outer_cutoff_angstrom
    )
    type_map = {int(value): index for index, value in enumerate(config.type_vocabulary)}
    ligand_types = torch.tensor([type_map[int(atomic_numbers[i])] for i in ligand_indices], dtype=torch.int64)
    environment_types = torch.tensor([type_map[int(atomic_numbers[i])] for i in environment_indices], dtype=torch.int64)
    edge_ligand_global = edges["edge_index"][0]
    edge_environment = edges["edge_index"][1]
    local_lookup = {value: index for index, value in enumerate(ligand_indices)}
    environment_lookup = {value: index for index, value in enumerate(environment_indices)}
    edge_local = torch.tensor([local_lookup[int(v)] for v in edge_ligand_global.tolist()], dtype=torch.int64)
    edge_environment_local = torch.tensor([environment_lookup[int(v)] for v in edge_environment.tolist()], dtype=torch.int64)

    batch = PackedSoftLiftBatch(
        ligand_type_index=ligand_types[None, :],
        edge_frame=torch.zeros(edge_local.shape[0], dtype=torch.int64),
        edge_ligand_local=edge_local,
        edge_ligand_type=ligand_types[edge_local],
        edge_environment_type=environment_types[edge_environment_local],
        edge_distance_angstrom=edges["distance"],
        edge_displacement_angstrom=edges["displacement"],
    )
    return batch, int(edges["distance"].numel()), topology_sha, trajectory_sha


def _parse_args(argv=None) -> ExportInputs:
    d = ExportInputs()
    parser = argparse.ArgumentParser(
        description="导出一份训练好的 R1 checkpoint 为 PyTorch-free 的 float64 payload。"
                    " 不传任何参数 = EXP-020 Atenolol 那条冻结路径，产物逐字节不变。")
    parser.add_argument("--checkpoint", type=Path, default=d.checkpoint)
    parser.add_argument("--checkpoint-sha256", default=d.checkpoint_sha256,
                        help="新产物传 none 表示只记录不比对")
    parser.add_argument("--ligand-indices", type=Path, default=d.ligand_indices)
    parser.add_argument("--ligand-indices-sha256", default=d.ligand_indices_sha256)
    parser.add_argument("--dataset", type=Path, default=d.dataset)
    parser.add_argument("--dataset-sha256", default=d.dataset_sha256)
    parser.add_argument("--topology", type=Path, default=d.topology)
    parser.add_argument("--topology-sha256", default=d.topology_sha256)
    parser.add_argument("--trajectory", type=Path, default=d.trajectory)
    parser.add_argument("--trajectory-sha256", default=d.trajectory_sha256)
    parser.add_argument("--frame", type=int, default=d.frame)
    parser.add_argument("--fd-atom", type=int, default=d.fd_atom)
    parser.add_argument("--expected-active-edges", default=d.expected_active_edges,
                        help="整数；新体系传 none 表示只记录不比对")
    parser.add_argument("--expected-atom-count", default=d.expected_atom_count,
                        help="整数；新体系传 none 表示只记录不比对")
    parser.add_argument("--output-dir", type=Path, default=d.output_dir)
    parser.add_argument("--system", type=Path, default=None,
                        help="serialized System XML；给了就用它做元素的权威来源"
                             "（mmCIF 对某些离子不写 type_symbol，mdtraj 会静默给 0）")
    parser.add_argument("--allow-derived-r1-config", action="store_true",
                        help="换配体重训：只校验架构字段，允许词表/配体原子数/容量/b_max 与出厂那份不同")
    args = parser.parse_args(argv)

    def opt(value):
        return None if value is None or str(value).lower() == "none" else value

    def opt_int(value):
        value = opt(value)
        return None if value is None else int(value)

    return ExportInputs(
        checkpoint=args.checkpoint,
        checkpoint_sha256=opt(args.checkpoint_sha256),
        ligand_indices=args.ligand_indices,
        ligand_indices_sha256=opt(args.ligand_indices_sha256),
        dataset=args.dataset,
        dataset_sha256=opt(args.dataset_sha256),
        topology=args.topology,
        topology_sha256=opt(args.topology_sha256),
        trajectory=args.trajectory,
        trajectory_sha256=opt(args.trajectory_sha256),
        frame=args.frame,
        fd_atom=args.fd_atom,
        expected_active_edges=opt_int(args.expected_active_edges),
        expected_atom_count=opt_int(args.expected_atom_count),
        output_dir=args.output_dir,
        allow_derived_r1_config=bool(getattr(args, "allow_derived_r1_config", False)),
        system=getattr(args, "system", None),
    )


def main(argv=None) -> int:
    inputs = _parse_args(argv)
    payload, config, checkpoint_sha = _load_checkpoint(inputs)
    print(f"checkpoint OK: experiment_id={payload['experiment_id']} held_out_run={payload['held_out_run']} seed={payload['seed']}")

    expected_keys = _expected_state_dict_keys(config)
    actual_keys = set(payload["state_dict"].keys())
    if actual_keys != expected_keys:
        raise ExporterError(
            "checkpoint state_dict keys disagree with build_softlift_model(config).state_dict() keys:\n"
            f"  missing: {sorted(expected_keys - actual_keys)}\n"
            f"  extra:   {sorted(actual_keys - expected_keys)}"
        )
    for name, tensor in payload["state_dict"].items():
        if tensor.dtype.__str__() != "torch.float64":
            raise ExporterError(f"tensor {name} is {tensor.dtype}, expected float64")
    print(f"state_dict OK: {len(actual_keys)} tensors, all float64, keys match model exactly")

    dataset_identity = _verify_dataset_identity(config, inputs)
    print(f"dataset/ligand_indices identity OK: {len(dataset_identity['ligand_topology_indices'])} ligand atoms")

    tensor_manifest, blob = _build_tensor_manifest(payload["state_dict"])
    print(f"tensor manifest built: {len(tensor_manifest)} tensors, {len(blob)} bytes total")

    # --- Python round-trip: independent model built ONLY from the exported payload ---
    reconstructed_state_dict = _rebuild_state_dict_from_payload(config, tensor_manifest, blob)
    reference_model = build_softlift_model(config).double()
    reference_model.load_state_dict(payload["state_dict"])
    reference_model.eval()
    reconstructed_model = build_softlift_model(config).double()
    reconstructed_model.load_state_dict(reconstructed_state_dict)
    reconstructed_model.eval()

    ligand_indices = dataset_identity["ligand_topology_indices"]
    batch, active_edges, topology_sha, trajectory_sha = _canonical_batch(config, ligand_indices, inputs)
    if inputs.expected_active_edges is None:
        print(f"canonical fixture active edges = {active_edges}（无预期值，只记录）")
    elif active_edges != inputs.expected_active_edges:
        raise ExporterError(
            f"canonical fixture active edge count is {active_edges}, expected {inputs.expected_active_edges}"
        )

    import torch

    with torch.no_grad():
        reference_output = reference_model(batch)
        reconstructed_output = reconstructed_model(batch)
    max_abs_diff = float(torch.max(torch.abs(reference_output - reconstructed_output)).item())
    bitwise_identical = bool(torch.equal(reference_output, reconstructed_output))
    print(f"round-trip check: bitwise_identical={bitwise_identical} max_abs_diff={max_abs_diff!r}")
    if not bitwise_identical:
        raise ExporterError(
            f"round-trip model (built purely from exported payload) disagrees with the checkpoint-loaded "
            f"model on the canonical fixture: max_abs_diff={max_abs_diff}"
        )

    output_dir = inputs.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "r1_model_weights_f64.bin"
    if weights_path.exists():
        raise ExporterError(f"refusing to overwrite {weights_path}")
    weights_path.write_bytes(blob)
    weights_sha256 = _sha256_file(weights_path)

    payload_body = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "source_checkpoint": {
            "path": _repo_relative(inputs.checkpoint),
            "sha256": checkpoint_sha,
            "held_out_run": payload["held_out_run"],
            "seed": payload["seed"],
        },
        "config": {
            "schema_version": config.schema_version,
            "rung": config.rung,
            "type_vocabulary": list(config.type_vocabulary),
            "n_ligand_atoms": config.n_ligand_atoms,
            "n_radial_basis": config.n_radial_basis,
            "n_channels": config.n_channels,
            "pair_dim": config.pair_dim,
            "context_dim": config.context_dim,
            "inner_cutoff_angstrom": config.inner_cutoff_angstrom,
            "outer_cutoff_angstrom": config.outer_cutoff_angstrom,
            "b_max_reduced": config.b_max_reduced,
            "max_environment_atoms": config.max_environment_atoms,
            "max_edges": config.max_edges,
            "max_neighbors_per_ligand": config.max_neighbors_per_ligand,
            "no_contact_output": config.no_contact_output,
            "protocol_sha256": config.protocol_sha256,
        },
        "architecture_facts_not_in_config": {
            "hidden_rho": HIDDEN_RHO,
            "rho_mlp_shape": "Linear(1, hidden_rho) -> SiLU -> Linear(hidden_rho, hidden_rho) -> SiLU -> Linear(hidden_rho, 1)",
            "rho_mlp_per_ligand_type": True,
            "type_count": len(config.type_vocabulary),
            "pair_weight_shape": [len(config.type_vocabulary), len(config.type_vocabulary), config.n_radial_basis],
            "aggregation": "raw = per_ligand.sum(dim=-1)  # SUM over the 41 ligand atoms, NOT a mean",
            "radial_basis_formula": "G_p(r) = exp(-0.5 * ((r - radial_centers[p]) / radial_width)^2)",
            "envelope_formula": "quintic C2: 1 for r<=inner, 0 for r>=outer, 1-10x^3+15x^4-6x^5 for x=(r-inner)/(outer-inner) in between",
            "units": "distances/cutoffs in Angstrom (NOT nm) -- OpenMM-side positions must be multiplied by 10 before evaluating this payload",
        },
        "ligand_topology_indices": ligand_indices,
        "dataset_cross_check": {
            "path": _repo_relative(inputs.dataset),
            "sha256": dataset_identity["dataset_sha256"],
        },
        "weights_file": {
            "name": weights_path.name,
            "sha256": weights_sha256,
            "byte_count": len(blob),
            "layout": "flat concatenation of tensors in the order listed in tensor_manifest, little-endian float64",
        },
        "tensor_manifest": tensor_manifest,
    }
    payload_path = output_dir / "r1_model_payload_v1.json"
    if payload_path.exists():
        raise ExporterError(f"refusing to overwrite {payload_path}")
    payload_path.write_text(json.dumps(payload_body, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    manifest_body = {
        "schema_version": "exp025-g1-r1-manifest-v1",
        "experiment_id": EXPERIMENT_ID,
        "exporter_script": {
            "path": "scripts/export_exp025_g1_reference_payload.py",
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "source_checkpoint_sha256": checkpoint_sha,
        "payload_json_sha256": _sha256_file(payload_path),
        "weights_bin_sha256": weights_sha256,
        "verification": {
            "state_dict_keys_match_model_exactly": True,
            "state_dict_all_float64": True,
            "config_matches_primary_r1_config_exactly": True,
            "dataset_and_ligand_indices_json_agree": True,
            "python_roundtrip_bitwise_identical_on_canonical_fixture": bitwise_identical,
            "python_roundtrip_max_abs_diff": max_abs_diff,
            "canonical_fixture_active_edges": active_edges,
        },
        "note": "This exporter does NOT pick among the other 8 EXP-020 checkpoints. Multi-checkpoint parity is explicit future work per PLAN_EXP-025_local_manybody_cuda.md, not assumed reliable from this run.",
    }
    manifest_path = output_dir / "r1_model_manifest.json"
    if manifest_path.exists():
        raise ExporterError(f"refusing to overwrite {manifest_path}")
    manifest_path.write_text(json.dumps(manifest_body, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    fixture_body = {
        "schema_version": "exp025-g1-canonical-fixture-manifest-v1",
        "experiment_id": EXPERIMENT_ID,
        "topology": {"path": _repo_relative(inputs.topology), "sha256": topology_sha},
        "trajectory": {"path": _repo_relative(inputs.trajectory), "sha256": trajectory_sha},
        "frame": inputs.frame,
        "ligand_indices": {"path": _repo_relative(inputs.ligand_indices), "sha256": dataset_identity["ligand_indices_sha256"], "count": len(ligand_indices)},
        "canonical_fd_atom": inputs.fd_atom,
        "active_edges": active_edges,
        "cross_checked_against": {
            "path": "output/outer_lambda_exp020_softlift_seedfixed001/d2/r1_d2_manifest.json",
            "note": "topology/trajectory/ligand_indices sha256 above were independently re-verified against this manifest's recorded values, not copied blind",
        },
    }
    fixture_path = output_dir / "canonical_fixture_manifest.json"
    if fixture_path.exists():
        raise ExporterError(f"refusing to overwrite {fixture_path}")
    fixture_path.write_text(json.dumps(fixture_body, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": "G1_REFERENCE_PAYLOAD_EXPORTED",
        "bitwise_identical_roundtrip": bitwise_identical,
        "active_edges": active_edges,
        "output_dir": _repo_relative(output_dir),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
