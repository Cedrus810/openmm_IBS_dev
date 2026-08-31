"""Canonical OpenMM loader for the opt-in outer-lambda residual Hamiltonian.

This module is the mainline adapter around the existing
``LocalManyBodyResidual`` plugin.  The residual implementation remains in the
plugin; Python only validates frozen artifacts, loads the plugin before XML
deserialization, and creates one fresh Force for each System.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from outer_lambda_neural_basis import (
    NeuralBasisModelSpec,
    NeuralPathSafety,
    OuterLambdaController,
)


SCHEMA_VERSION = 2
FROZEN_SKIN_ANGSTROM = 1.0
FROZEN_CANDIDATE_LIST_CAPACITY = 8192
KNOWN_PLUGIN_SOURCE_SHA256 = (
    "10afff53ef85aba99b024e4ce5f9a66927cbc0bf9bb392537663934327b5b0be"
)
FEATURE_NAME = "Outer-Lambda Local Residual for IBS"
EM_POLICY = "no_residual_twin"
RESOURCE_MANIFEST_VERSION = 1
LIGAND_IDENTITY_PROTOCOL = "local_atomic_numbers_and_internal_bond_graph_v1"


@dataclass(frozen=True)
class R1Payload:
    ligand_topology_indices: tuple[int, ...]
    type_vocabulary: tuple[int, ...]
    inner_cutoff_angstrom: float
    outer_cutoff_angstrom: float
    b_max_reduced: float
    max_edges: int
    max_neighbors_per_ligand: int
    max_environment_atoms: int
    pair_weight: np.ndarray
    radial_centers: np.ndarray
    radial_width: float
    rho: list[dict[str, np.ndarray]]
    source_checkpoint_sha256: str


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_r1_payload(payload_json_path: str | Path, weights_bin_path: str | Path) -> R1Payload:
    payload_path = Path(payload_json_path)
    weights_path = Path(weights_bin_path)
    doc = json.loads(payload_path.read_text())
    blob = weights_path.read_bytes()
    tensors: dict[str, np.ndarray] = {}
    for entry in doc["tensor_manifest"]:
        if entry["dtype"] != "float64_little_endian":
            raise ValueError(f"unexpected tensor dtype: {entry['dtype']!r}")
        offset = int(entry["byte_offset"])
        count = int(entry["byte_count"])
        if offset < 0 or count <= 0 or offset + count > len(blob) or count % 8:
            raise ValueError(f"invalid tensor bounds for {entry['name']!r}")
        shape = tuple(int(v) for v in entry["shape"]) or (1,)
        values = np.frombuffer(blob[offset:offset + count], dtype="<f8")
        if values.size != int(np.prod(shape)):
            raise ValueError(f"tensor shape/count mismatch for {entry['name']!r}")
        tensors[entry["name"]] = values.reshape(shape)

    cfg = doc["config"]
    type_vocab = tuple(int(v) for v in cfg["type_vocabulary"])
    n_types = len(type_vocab)
    n_radial = int(cfg["n_radial_basis"])
    rho = []
    for t in range(n_types):
        rho.append({
            "w0": tensors[f"rho.{t}.0.weight"].reshape(16),
            "b0": tensors[f"rho.{t}.0.bias"].reshape(16),
            "w2": tensors[f"rho.{t}.2.weight"].reshape(16, 16),
            "b2": tensors[f"rho.{t}.2.bias"].reshape(16),
            "w4": tensors[f"rho.{t}.4.weight"].reshape(16),
            "b4": float(tensors[f"rho.{t}.4.bias"].reshape(())),
        })
    source = doc.get("source_checkpoint")
    source_sha = source.get("sha256") if isinstance(source, Mapping) else source
    if not isinstance(source_sha, str) or len(source_sha) != 64:
        raise ValueError("R1 payload 缺少有效 source_checkpoint SHA-256")
    return R1Payload(
        ligand_topology_indices=tuple(int(v) for v in doc["ligand_topology_indices"]),
        type_vocabulary=type_vocab,
        inner_cutoff_angstrom=float(cfg["inner_cutoff_angstrom"]),
        outer_cutoff_angstrom=float(cfg["outer_cutoff_angstrom"]),
        b_max_reduced=float(cfg["b_max_reduced"]),
        max_edges=int(cfg["max_edges"]),
        max_neighbors_per_ligand=int(cfg["max_neighbors_per_ligand"]),
        max_environment_atoms=int(cfg["max_environment_atoms"]),
        pair_weight=tensors["pair_weight"].reshape(n_types, n_types, n_radial),
        radial_centers=tensors["radial_centers"].reshape(n_radial),
        radial_width=float(tensors["radial_width"].reshape(())),
        rho=rho,
        source_checkpoint_sha256=source_sha,
    )


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _topology_atomic_numbers(topology, ligand_ids: Sequence[int]) -> list[int]:
    atoms = list(topology.atoms())
    n_atoms = len(atoms)
    ids = [int(value) for value in ligand_ids]
    if not ids or len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= n_atoms:
        raise RuntimeError(
            "LocalManyBodyResidual 的 ligand_indices 必须是拓扑范围内不重复的原子序号"
        )
    atomic_numbers = []
    for index in ids:
        atom = atoms[index]
        if atom.element is None or atom.element.atomic_number is None:
            raise RuntimeError("LocalManyBodyResidual 要求配体每个原子都有元素序号")
        atomic_numbers.append(int(atom.element.atomic_number))
    return atomic_numbers


def _internal_bonds_from_topology(topology, ligand_ids: Sequence[int]) -> set[tuple[int, int]]:
    local = {int(global_index): local_index for local_index, global_index in enumerate(ligand_ids)}
    pairs: set[tuple[int, int]] = set()
    bonds = topology.bonds()
    for bond in bonds:
        atom1 = getattr(bond, "atom1", None)
        atom2 = getattr(bond, "atom2", None)
        if atom1 is None or atom2 is None:
            continue
        first = int(atom1.index)
        second = int(atom2.index)
        if first in local and second in local and first != second:
            pairs.add(tuple(sorted((local[first], local[second]))))
    return pairs


def _internal_bonds_from_system(system, ligand_ids: Sequence[int]) -> set[tuple[int, int]]:
    """Recover the actual bonded graph when a cached mmCIF dropped nonstandard bonds."""
    from openmm import HarmonicBondForce

    local = {int(global_index): local_index for local_index, global_index in enumerate(ligand_ids)}
    pairs: set[tuple[int, int]] = set()
    for force in system.getForces():
        if not isinstance(force, HarmonicBondForce):
            continue
        for bond_index in range(force.getNumBonds()):
            first, second, _length, _stiffness = force.getBondParameters(bond_index)
            first, second = int(first), int(second)
            if first in local and second in local and first != second:
                pairs.add(tuple(sorted((local[first], local[second]))))
    for constraint_index in range(system.getNumConstraints()):
        first, second, _distance = system.getConstraintParameters(constraint_index)
        first, second = int(first), int(second)
        if first in local and second in local and first != second:
            pairs.add(tuple(sorted((local[first], local[second]))))
    return pairs


def ligand_chemical_identity(
    topology,
    ligand_indices: Sequence[int],
    *,
    system=None,
) -> dict[str, Any]:
    """Return a topology-local identity, independent of global atom numbering.

    Cached mmCIF files do not preserve bonds for nonstandard ``MOL`` residues.
    When the native System is available, its HarmonicBondForce/constraints are
    therefore the authoritative internal graph; otherwise the topology graph is
    used and an incomplete graph will fail the frozen-model comparison.
    """
    ligand_ids = tuple(int(value) for value in ligand_indices)
    atomic_numbers = _topology_atomic_numbers(topology, ligand_ids)
    topology_bonds = _internal_bonds_from_topology(topology, ligand_ids)
    system_bonds = _internal_bonds_from_system(system, ligand_ids) if system is not None else set()
    internal_bonds = system_bonds or topology_bonds
    identity = {
        "protocol": LIGAND_IDENTITY_PROTOCOL,
        "atom_count": len(ligand_ids),
        "atomic_numbers": atomic_numbers,
        "internal_bonds": [list(pair) for pair in sorted(internal_bonds)],
    }
    identity["fingerprint_sha256"] = _canonical_json_sha256({
        "atomic_numbers": atomic_numbers,
        "internal_bonds": identity["internal_bonds"],
    })
    return identity


#: 缺少冻结模型资源时的说明。资源只对 Atenolol 有效（manifest 硬绑 41 个原子
#: 和具体键图），换体系用不上，因此 2026-08-31 发布整理时没有随工程区分支分发。
RESOURCE_MISSING_HINT = (
    "Outer-Lambda Local Residual for IBS 的冻结 R1 模型资源不在本仓库中：\n"
    "  {path}\n"
    "这份资源只对 Atenolol 有效（manifest 硬绑 41 个原子与具体内部键图），"
    "换体系用不上，所以不随本工程区分支分发。\n"
    "要在 Atenolol 上启用 outer_lambda_local_residual_ibs，从 Atenolol-rank11 "
    "工作区取回 resources/outer_lambda_local_residual/ 整个目录，或用 "
    "resource_manifest= 显式指定一份 manifest。\n"
    "换成别的配体不能只换 manifest：R1 是按配体训练的模型，必须重训，"
    "而训练/部署栈（softlift*、student*、teacher_graph、loss、atom_mapping 等）"
    "同样不随本分支分发，也在 Atenolol-rank11。\n"
    "不需要该功能时保持该开关为 false 即可（默认值）。"
)


def _load_resource_manifest(manifest_path: str | Path) -> tuple[dict[str, Any], Path, Path]:
    manifest = Path(manifest_path).resolve()
    # fail closed，并且说清楚"为什么没有"——裸 FileNotFoundError 只给一个路径，
    # 读起来像装坏了，而实际是这份资源本来就不随包发布。
    if not manifest.is_file():
        raise FileNotFoundError(RESOURCE_MISSING_HINT.format(path=manifest))
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    if doc.get("manifest_version") != RESOURCE_MANIFEST_VERSION:
        raise RuntimeError("LocalManyBodyResidual resource manifest 版本不匹配")
    if doc.get("feature") != FEATURE_NAME:
        raise RuntimeError("LocalManyBodyResidual resource manifest feature 不匹配")
    supported = doc.get("supported_ligand")
    if not isinstance(supported, Mapping) or supported.get("name") != "Atenolol":
        raise RuntimeError("冻结 R1 模型的 supported_ligand 必须明确声明为 Atenolol")
    expected_fingerprint = _canonical_json_sha256({
        "atomic_numbers": supported.get("atomic_numbers"),
        "internal_bonds": supported.get("internal_bonds"),
    })
    if supported.get("identity_protocol") != LIGAND_IDENTITY_PROTOCOL:
        raise RuntimeError("冻结 R1 模型的 ligand identity protocol 不匹配")
    if supported.get("fingerprint_sha256") != expected_fingerprint:
        raise RuntimeError("冻结 R1 模型 manifest 的 Atenolol chemical fingerprint 损坏")
    plugin = doc.get("plugin")
    if not isinstance(plugin, Mapping) or plugin.get("source_sha256") != KNOWN_PLUGIN_SOURCE_SHA256:
        raise RuntimeError("冻结 R1 模型 manifest 的插件源码身份不匹配")
    payload_info = doc.get("payload")
    weights_info = doc.get("weights")
    if not isinstance(payload_info, Mapping) or not isinstance(weights_info, Mapping):
        raise RuntimeError("冻结 R1 模型 resource manifest 缺少 payload/weights")
    payload_path = (manifest.parent / str(payload_info["path"])).resolve()
    weights_path = (manifest.parent / str(weights_info["path"])).resolve()
    if not payload_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError("冻结 R1 模型 resource manifest 指向的文件不存在")
    if sha256_file(payload_path) != str(payload_info.get("sha256")):
        raise RuntimeError("冻结 R1 payload SHA-256 不匹配")
    if sha256_file(weights_path) != str(weights_info.get("sha256")):
        raise RuntimeError("冻结 R1 weights SHA-256 不匹配")
    return doc, payload_path, weights_path


def atom_type_index_for_topology(
    atomic_numbers: Sequence[int], type_vocabulary: Sequence[int]
) -> list[int]:
    type_map = {int(value): index for index, value in enumerate(type_vocabulary)}
    try:
        return [type_map[int(number)] for number in atomic_numbers]
    except KeyError as exc:
        raise ValueError(
            f"拓扑包含不在 LocalManyBodyResidual 固定词表中的元素: {exc.args[0]}"
        ) from exc


def _encode_double_array(values: np.ndarray) -> str:
    return " ".join(repr(float(value)) for value in np.asarray(values).ravel())


def _encode_int_array(values: Sequence[int]) -> str:
    return " ".join(str(int(value)) for value in values)


def build_local_manybody_residual_force_xml(
    payload: R1Payload,
    *,
    atom_type_index: Sequence[int],
    temperature_kelvin: float,
    ligand_topology_indices: Sequence[int],
    skin_angstrom: float = FROZEN_SKIN_ANGSTROM,
    candidate_list_capacity: int = FROZEN_CANDIDATE_LIST_CAPACITY,
    force_group: int = 0,
) -> str:
    if not atom_type_index:
        raise ValueError("atom_type_index must be non-empty")
    if len(ligand_topology_indices) != len(payload.ligand_topology_indices):
        raise ValueError("当前拓扑的配体原子数与 R1 payload 不一致")
    n_types = len(payload.type_vocabulary)
    mlps = []
    for t in range(n_types):
        r = payload.rho[t]
        mlps.append(
            f'\t\t<MLP b0="{_encode_double_array(r["b0"])}" '
            f'b2="{_encode_double_array(r["b2"])}" b4="{repr(float(r["b4"]))}" '
            f'w0="{_encode_double_array(r["w0"])}" '
            f'w2="{_encode_double_array(r["w2"])}" '
            f'w4="{_encode_double_array(r["w4"])}"/>'
        )
    return (
        '<?xml version="1.0" ?>\n'
        f'<Force atomTypeIndex="{_encode_int_array(atom_type_index)}" '
        f'bMaxReduced="{repr(float(payload.b_max_reduced))}" '
        f'candidateListCapacity="{int(candidate_list_capacity)}" forceGroup="{int(force_group)}" '
        f'innerCutoffAngstrom="{repr(float(payload.inner_cutoff_angstrom))}" '
        f'ligandTopologyIds="{_encode_int_array(ligand_topology_indices)}" '
        f'maxEdges="{int(payload.max_edges)}" maxEnvironmentAtoms="{int(payload.max_environment_atoms)}" '
        f'maxNeighborsPerLigand="{int(payload.max_neighbors_per_ligand)}" name="" '
        f'numTypes="{n_types}" outerCutoffAngstrom="{repr(float(payload.outer_cutoff_angstrom))}" '
        f'pairWeight="{_encode_double_array(payload.pair_weight)}" '
        f'radialCenters="{_encode_double_array(payload.radial_centers)}" '
        f'radialWidthAngstrom="{repr(float(payload.radial_width))}" schema_version="{SCHEMA_VERSION}" '
        f'skinAngstrom="{repr(float(skin_angstrom))}" '
        f'sourceCheckpointSha256="{payload.source_checkpoint_sha256}" '
        f'temperatureKelvin="{repr(float(temperature_kelvin))}" '
        'type="LocalManyBodyResidualForce" '
        f'typeVocabulary="{_encode_int_array(payload.type_vocabulary)}">\n'
        '\t<TypedMLPs>\n' + '\n'.join(mlps) + '\n\t</TypedMLPs>\n</Force>'
    )


def load_plugin_libraries(
    openmm_module, plugin_build_dir: str | Path, *, include_cuda: bool = True
) -> None:
    plugin_dir = Path(plugin_build_dir)
    required = (
        "libOpenMMLocalManyBodyResidual.so",
        "libOpenMMLocalManyBodyResidualReference.so",
    )
    if include_cuda:
        required += ("libOpenMMLocalManyBodyResidualCUDA.so",)
    for name in required:
        path = plugin_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"缺少 LocalManyBodyResidual 插件库: {path}")
    for name in required:
        openmm_module.Platform.loadPluginLibrary(str(plugin_dir / name))


def _verify_plugin_source(plugin_build_dir: Path) -> tuple[Path, str, str]:
    cuda_source = plugin_build_dir.parent / "platforms/cuda/src/CudaLocalManyBodyResidualKernels.cpp"
    if not cuda_source.is_file():
        raise FileNotFoundError(f"缺少插件固定源码身份文件: {cuda_source}")
    source_sha = sha256_file(cuda_source)
    if source_sha != KNOWN_PLUGIN_SOURCE_SHA256:
        raise RuntimeError(
            "LocalManyBodyResidual 插件源码身份不匹配；拒绝启用正式开关: "
            f"actual={source_sha}, expected={KNOWN_PLUGIN_SOURCE_SHA256}"
        )
    binary = plugin_build_dir / "libOpenMMLocalManyBodyResidualCUDA.so"
    return binary, source_sha, sha256_file(binary)


@dataclass(frozen=True)
class OuterLambdaLocalResidualRuntime:
    payload: R1Payload
    atom_type_index: tuple[int, ...]
    ligand_topology_indices: tuple[int, ...]
    temperature_kelvin: float
    controller: OuterLambdaController
    plugin_identity: dict[str, Any]
    sampling_score_sha256: str
    ligand_identity: dict[str, Any]
    ligand_indices_sha256: str
    leg_name: str

    @property
    def energy_offset_kj_mol(self) -> float:
        return 0.0

    @property
    def em_policy(self) -> str:
        return EM_POLICY

    def force_factory(self):
        import openmm

        return openmm.XmlSerializer.deserialize(
            build_local_manybody_residual_force_xml(
                self.payload,
                atom_type_index=self.atom_type_index,
                ligand_topology_indices=self.ligand_topology_indices,
                temperature_kelvin=self.temperature_kelvin,
            )
        )

    def state_coefficients_factory(self, lc_win, lv_win):
        if len(lc_win) != len(lv_win):
            raise ValueError("lc_win/lv_win 长度不一致，无法生成 A_k")
        return [row[0] for row in self.controller.coefficient_matrix(lv_win)]

    def provenance_payload(self) -> dict[str, Any]:
        return {
            "feature": FEATURE_NAME,
            "em_policy": self.em_policy,
            "plugin": dict(self.plugin_identity),
            "model": {
                "source_checkpoint_sha256": self.payload.source_checkpoint_sha256,
                "supported_ligand": "Atenolol",
                "ligand_identity_protocol": LIGAND_IDENTITY_PROTOCOL,
                "trained_ligand_topology_indices": list(
                    self.payload.ligand_topology_indices
                ),
                "type_vocabulary": list(self.payload.type_vocabulary),
                "ligand_topology_indices": list(self.ligand_topology_indices),
                "ligand_indices_sha256": self.ligand_indices_sha256,
                "leg_name": self.leg_name,
                "chemical_identity": dict(self.ligand_identity),
                "atom_type_index_sha256": hashlib.sha256(
                    json.dumps(list(self.atom_type_index), separators=(",", ":")).encode()
                ).hexdigest(),
            },
            "controller": self.controller.protocol_payload(),
            "sampling_score_sha256": self.sampling_score_sha256,
        }


def build_outer_lambda_local_residual_runtime(
    *,
    topology,
    ligand_indices: Sequence[int],
    temperature_kelvin: float,
    potential_type: str,
    output_dir: str | Path | None = None,
    platform_name: str = "CUDA",
    plugin_build_dir: str | Path | None = None,
    payload_json: str | Path | None = None,
    weights_bin: str | Path | None = None,
    coefficient: float = 1.0,
    system=None,
    ligand_indices_path: str | Path | None = None,
    leg_name: str = "complex",
    resource_manifest: str | Path | None = None,
) -> OuterLambdaLocalResidualRuntime:
    """Load the frozen plugin/model and bind it to one concrete topology."""
    import openmm

    repo_root = Path(__file__).resolve().parents[1]
    plugin_dir = Path(plugin_build_dir or repo_root / "plugins/LocalManyBodyResidual/build")
    manifest_path = Path(
        resource_manifest
        or repo_root / "resources/outer_lambda_local_residual/manifest.json"
    )
    manifest, frozen_payload_path, frozen_weights_path = _load_resource_manifest(
        manifest_path
    )
    if (payload_json is None) != (weights_bin is None):
        raise ValueError("payload_json 与 weights_bin 必须同时提供")
    if payload_json is not None:
        raise ValueError(
            "正式 residual 只允许从稳定 resource manifest 加载冻结模型；"
            "不支持第二套运行时 artifact 选择路径"
        )
    payload_path = frozen_payload_path
    weights_path = frozen_weights_path
    binary_path, source_sha, binary_sha = _verify_plugin_source(plugin_dir)
    normalized_platform = str(platform_name).strip().upper()
    if normalized_platform not in {"CPU", "CUDA"}:
        raise RuntimeError(
            "Outer-Lambda Local Residual for IBS 当前只支持 CUDA 或 CPU(Reference)；"
            f"不支持 platform={platform_name!r}"
        )
    load_plugin_libraries(
        openmm,
        plugin_dir,
        include_cuda=normalized_platform == "CUDA",
    )
    expected_ligand = manifest["supported_ligand"]
    payload = load_r1_payload(payload_path, weights_path)
    training = manifest.get("training")
    if not isinstance(training, Mapping) or training.get(
        "source_checkpoint_sha256"
    ) != payload.source_checkpoint_sha256:
        raise RuntimeError("冻结 R1 payload 与 resource manifest 的训练 checkpoint 身份不一致")
    if len(payload.ligand_topology_indices) != int(expected_ligand["atom_count"]):
        raise RuntimeError("冻结 R1 payload 的 ligand atom count 与 Atenolol manifest 不一致")
    ligand_ids = tuple(int(value) for value in ligand_indices)
    if len(ligand_ids) != len(payload.ligand_topology_indices):
        raise RuntimeError(
            "当前拓扑的配体原子数与冻结 R1 payload 不一致；"
            "拒绝把模型静默用于不兼容的新体系"
        )
    ligand_identity = ligand_chemical_identity(
        topology, ligand_ids, system=system
    )
    if (
        ligand_identity["atomic_numbers"] != expected_ligand["atomic_numbers"]
        or ligand_identity["internal_bonds"] != expected_ligand["internal_bonds"]
    ):
        raise RuntimeError(
            "冻结 LocalManyBodyResidual R1 模型只支持 Atenolol；"
            "当前配体的局部原子序列或内部键图与 Atenolol 不一致，"
            "拒绝把模型静默用于任意新配体。"
            f" expected_fingerprint={expected_ligand['fingerprint_sha256']},"
            f" actual_fingerprint={ligand_identity['fingerprint_sha256']}"
        )
    atomic_numbers = []
    for atom in topology.atoms():
        if atom.element is None or atom.element.atomic_number is None:
            raise RuntimeError("LocalManyBodyResidual 要求拓扑每个原子都有元素序号")
        atomic_numbers.append(int(atom.element.atomic_number))
    atom_types = tuple(atom_type_index_for_topology(atomic_numbers, payload.type_vocabulary))
    if ligand_indices_path is None:
        if output_dir is None:
            raise ValueError("未提供 ligand_indices_path 或 output_dir，无法绑定配体身份")
        index_filename = (
            "ligand_indices_solvent.json" if str(leg_name) == "solvent"
            else "ligand_indices.json"
        )
        indices_path = Path(output_dir) / index_filename
    else:
        indices_path = Path(ligand_indices_path)
    if not indices_path.is_file():
        raise FileNotFoundError(f"缺少正式 residual 的配体索引身份文件: {indices_path}")
    indices_doc = json.loads(indices_path.read_text(encoding="utf-8"))
    recorded_ids = tuple(int(value) for value in indices_doc.get("ligand_indices", ()))
    if recorded_ids != ligand_ids:
        raise RuntimeError(
            f"{leg_name} residual 的 ligand_indices 文件与当前拓扑不一致；"
            "拒绝把另一条腿的索引身份绑定进模型"
        )
    ligand_indices_sha256 = sha256_file(indices_path)
    basis_spec = NeuralBasisModelSpec(
        name="outer_lambda_local_manybody_residual_r1",
        backend="existing_openmmml",
        model_path=str(payload_path.resolve()),
        sha256=payload.source_checkpoint_sha256,
        energy_offset_kj_mol=0.0,
        atom_selection="fixed_indices",
        atom_indices_path=str(indices_path.resolve()),
        atom_indices_sha256=ligand_indices_sha256,
        output_unit="kJ_per_mol",
        precision="single",
        periodic=True,
    )
    kt = 8.31446261815324e-3 * float(temperature_kelvin)
    max_abs_basis = kt * payload.b_max_reduced * 1.2
    controller = OuterLambdaController(
        enabled=True,
        stage="vanishing",
        baseline_potential=str(potential_type),
        endpoint_tolerance=1.0e-12,
        coefficients=(float(coefficient),),
        max_abs_coefficient=max(10.0, abs(float(coefficient)) * 2.0),
        bases=(basis_spec,),
        safety=NeuralPathSafety(
            max_abs_basis_energy_kj_mol=max_abs_basis,
            max_abs_path_energy_kj_mol=max_abs_basis * abs(float(coefficient)) * 2.0 + 1.0,
            max_force_norm_kj_mol_nm=1.0e5,
            fail_on_support_domain_violation=False,
        ),
    )
    payload_identity = {
        "payload_json_sha256": sha256_file(payload_path),
        "weights_sha256": sha256_file(weights_path),
        "source_checkpoint_sha256": payload.source_checkpoint_sha256,
        "resource_manifest": str(manifest_path.resolve()),
        "resource_manifest_version": int(manifest["manifest_version"]),
        "supported_ligand": expected_ligand["name"],
        "ligand_identity_protocol": LIGAND_IDENTITY_PROTOCOL,
        "ligand_identity_fingerprint": ligand_identity["fingerprint_sha256"],
        "leg_name": str(leg_name),
        "plugin_source_sha256": source_sha,
        "plugin_binary_sha256": binary_sha,
        "plugin_binary": str(binary_path.resolve()),
        "schema_version": SCHEMA_VERSION,
        "skin_angstrom": FROZEN_SKIN_ANGSTROM,
        "candidate_list_capacity": FROZEN_CANDIDATE_LIST_CAPACITY,
    }
    identity_without_score = {
        "feature": FEATURE_NAME,
        "em_policy": EM_POLICY,
        "plugin": payload_identity,
        "model": controller.protocol_payload(),
        "atom_type_index": list(atom_types),
        "ligand_topology_indices": list(ligand_ids),
        "ligand_indices_sha256": ligand_indices_sha256,
        "ligand_chemical_identity": ligand_identity,
        "temperature_kelvin": float(temperature_kelvin),
    }
    score_sha = hashlib.sha256(
        json.dumps(identity_without_score, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return OuterLambdaLocalResidualRuntime(
        payload=payload,
        atom_type_index=atom_types,
        ligand_topology_indices=ligand_ids,
        temperature_kelvin=float(temperature_kelvin),
        controller=controller,
        plugin_identity=payload_identity,
        sampling_score_sha256=score_sha,
        ligand_identity=ligand_identity,
        ligand_indices_sha256=ligand_indices_sha256,
        leg_name=str(leg_name),
    )
