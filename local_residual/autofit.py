"""⚠️ 未接线，留档。不要重新接进 runabfe / pipeline —— 见文件末尾的"为什么撤"。

换配体时自动重训 R1 局部残差模型（原设计：在 Stage-2 vdW 采样之前跑完）。

与 `docs/RETRAIN_LOCAL_RESIDUAL.md` 那条四步手工链共用同一批实现
（`build_dataset_v1` → `scripts/train_exp019_softlift_loro.py` →
`scripts/export_exp025_g1_reference_payload.py` →
`scripts/write_local_residual_resource_manifest.py`）。只有两处不同：

1. **训练帧来自本次 run 自己现采的探针轨迹**，而不是三条冻结的 EXP-012 run。
   探针就是 Stage-2 那套 dual-lambda 软核系统，沿 λ_vdw 梯子走一遍，每个 λ 采几帧。
2. **词表 / 容量 / `n_ligand_atoms` 从当前体系推**（`derived_r1_config`），
   而不是 `primary_r1_config` 里那份 41 原子 Atenolol 常量 —— 换配体时那份必然
   fail-closed。

这条链上**没有 ML teacher**：R1 的训练信号是纯 MM 的 adjacent gap
（`local_residual/loss.py` 的 `bidirectional_gap_variance_loss` 只吃
`adjacent_gap_reduced`，见 EXP-033 §2.1）。`teacher_z_table` 只用来在开跑前判
"这个体系的元素下游 ML 势能不能表示"，不参与拟合。

⚠️ 产出是**这次 run 私有**的冻结产物（写在 `<output>/resources/…`），不覆盖仓库
里那份出厂 Atenolol manifest。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import shutil
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: 探针梯子默认态数。只用来产训练帧，跟生产 λ 布点无关（模型是几何的函数，
#: 不是 λ 的函数），所以不必等 Stage-2 pilot 把真实梯子解出来。
DEFAULT_PROBE_STATES = 8
#: 探针 λ 区间。**必须是窗口形状的，不能横跨整条梯子**（2026-09-11 实测教训）：
#: ledger 把每一帧重加权到区间内**每一个**态上，而深度解耦端采到的是鬼影构型
#: ——把它评到 λ≈1 的耦合态上就是 r⁻¹² 爆炸，实测 |log_importance| 到过 1e13，
#: 权重退化到 ESS≈1.5，最后死在 loss 的 `normalized_weights must sum to one`。
#: 生产从不这么做：它只在窗口内（相邻 5~6 个 λ）重加权，EXP-020 那份冻结数据集
#: 也是 `hard_window0` 的窗口切片。默认区间取窗口 0（λ 1.0→0.5），与冻结 R1
#: 的训练分布同口径。
DEFAULT_PROBE_LAMBDA_MAX = 1.0
DEFAULT_PROBE_LAMBDA_MIN = 0.5
#: 每个 λ 态采几帧。三个 LORO 分区按 round-robin 取帧，所以要是 3 的倍数。
#: 8 态 × 60 帧 = 480 帧，与 EXP-020 冻结那份数据集（1500 帧 / 3 条 run）同一个
#: 量级。**别再往下砍**：2026-09-11 用 8 态 × 18 帧（54 帧）实测，held-out 改善
#: 只有 +0.000 —— 链子跑得通，但学不出东西。
DEFAULT_FRAMES_PER_STATE = 60
#: 每帧之间跑多少步（2 fs → 2 ps/帧）。
DEFAULT_STEPS_BETWEEN_FRAMES = 1000
#: 每个 λ 态换点后先跑多少步再开始取帧。
DEFAULT_EQUILIBRATION_STEPS = 2000

#: 找 MACE 模型文件的根目录，按顺序试。`ABFE_MACE_MODEL_DIR` 是给"模型不在默认
#: 缓存里"的机器用的（本机是 /home/ruigengji/MLP/mace）——不把个人路径写进代码。
MACE_MODEL_SEARCH_ENV = "ABFE_MACE_MODEL_DIR"
MACE_MODEL_DEFAULT_DIRS = ("~/.cache/mace",)

#: 只放**没法自报家门**的模型。MACE 不在这里：它的 z-table 直接从 .model 文件读
#: （见 `mace_z_table`），抄一份常量迟早跟模型对不上——同一批 MACE-omol 里
#: `-1024` 是 83 个元素、`-4M` 是 82 个，抄错一个就白挡或漏挡。
KNOWN_TEACHER_Z_TABLES: dict[str, tuple[int, ...]] = {
    # UBio-MolFM（/home/ruigengji/UBio-MolFM_OpenMM-ML）head=omol25。
    # ⚠️ 它**没有**通常意义上的 z-table：元素嵌入是 nn.Embedding(256)，词表外的 Z
    # 不会报错，只会静默返回一个没训过的 embedding。仓库唯一机器可读的覆盖声明是
    # 每个 head 的原子参考能表：
    #     REF = np.asarray(molfm.data.utils.get_data_defult_config("omol25")[0])
    #     covered = np.nonzero(np.abs(REF) > 1e-3)[0]      # 实测 Z = 1..83 连续无缺口
    # Z >= 100 会在适配层抛，Z 84..99 参考能为 0 **静默通过** —— 这里挡的就是那段。
    # "参考能非零"只是训练覆盖的代理，不是保证（该仓库能给出的最强信号）。
    # 数据来源：ubio-molfm-openmm-ml 会话 2026-09-11 在 /home/ruigengji/UBio-MolFM_OpenMM-ML
    # 上的实测。这是一份**拷贝**，两个仓库之间没有链接：参考能表/默认 head/checkpoint
    # 变了这里不会自己跟着变，用上面那段 get_data_defult_config 重算一次即可核对。
    # 不从 .pt 直接读是因为那需要 import molfm，等于把那个项目接进本仓库的启动路径。
    "ubio-molfm-omol25": tuple(range(1, 84)),
}


class AutofitError(RuntimeError):
    """自动重训链上的契约失败。一律 fail-closed，不回退到出厂权重。"""


@dataclass(frozen=True)
class AutofitResult:
    manifest_path: Path
    payload_path: Path
    weights_path: Path
    report_path: Path
    reused_existing: bool


def _log_to(log: Callable[[str], None] | None) -> Callable[[str], None]:
    return log if callable(log) else (lambda message: print(message))


def resolve_mace_model(model: str | Path) -> Path:
    """把 `mace-off24-medium` 这样的名字或一个路径解析成具体的 .model 文件。

    换模型就是换这个参数——`/home/ruigengji/MLP/mace` 下那一堆（MACE-OFF24 /
    MACE-omol / MACE-POLAR / mace-mpa）都能直接给路径，或者把那个目录设成
    `ABFE_MACE_MODEL_DIR` 后按文件名给。
    """

    import os

    candidate = Path(model).expanduser()
    if candidate.is_file():
        return candidate
    roots = [os.environ.get(MACE_MODEL_SEARCH_ENV)] + list(MACE_MODEL_DEFAULT_DIRS)
    searched: list[str] = []
    for root in roots:
        if not root:
            continue
        directory = Path(root).expanduser()
        searched.append(str(directory))
        if not directory.is_dir():
            continue
        for name in (str(model), f"{model}.model"):
            hit = directory / name
            if hit.is_file():
                return hit
        # 名字大小写/分隔符不一致很常见（mace-off24-medium ↔ MACE-OFF24_medium）
        normalized = str(model).replace("-", "").replace("_", "").lower()
        for hit in sorted(directory.glob("*.model")):
            if hit.stem.replace("-", "").replace("_", "").lower() == normalized:
                return hit
    raise AutofitError(
        f"找不到 MACE 模型 {model!r}；已找过 {searched}。"
        f"给绝对路径，或把模型目录设进 ${MACE_MODEL_SEARCH_ENV}。"
    )


@lru_cache(maxsize=8)
def _mace_z_table_cached(resolved: str) -> tuple[int, ...]:
    import torch

    module = torch.load(resolved, map_location="cpu", weights_only=False)
    numbers = getattr(module, "atomic_numbers", None)
    if numbers is None:
        raise AutofitError(f"{resolved} 不像 MACE 模型：没有 atomic_numbers")
    return tuple(sorted(int(value) for value in numbers))


def mace_z_table(model: str | Path) -> tuple[int, ...]:
    """从 MACE 模型文件**读**它自己的元素表，不抄常量。

    代价是要 `torch.load` 整个模型（extra-large 那几个 ~400 MB），所以按解析后的
    路径做了缓存，一次 run 只读一遍。
    """

    return _mace_z_table_cached(str(resolve_mace_model(model)))


def teacher_z_table(
    teacher: str, *, extra_z_tables: Mapping[str, Sequence[int]] | None = None
) -> tuple[int, ...]:
    """teacher 名字 → 元素表。显式传的优先，其次能自报家门的 MACE，最后常量表。"""

    if extra_z_tables and teacher in extra_z_tables:
        return tuple(sorted(int(value) for value in extra_z_tables[teacher]))
    if teacher in KNOWN_TEACHER_Z_TABLES:
        return KNOWN_TEACHER_Z_TABLES[teacher]
    try:
        return mace_z_table(teacher)
    except AutofitError as exc:
        raise AutofitError(
            f"无法确定 teacher {teacher!r} 的元素覆盖：{exc}. "
            f"常量表里只有 {sorted(KNOWN_TEACHER_Z_TABLES)}（那些是没法自报家门的），"
            "MACE 模型直接给路径或文件名即可。"
        ) from exc


def check_element_coverage(
    atomic_numbers: Iterable[int], teacher: str | None, *, extra_z_tables: Mapping[str, Sequence[int]] | None = None
) -> tuple[int, ...]:
    """体系用到的元素 → 排序去重；给了 teacher 就同时校验它能不能表示这些元素。

    R1 自己的词表就是这一串（`softlift_dataset` 也是这么推的），所以"元素种类
    超了"对 MM 链来说不是问题——词表跟着体系走。会出问题的是下游 ML 势：
    `openmm_plugin.atom_type_index_for_topology` 要求**整个拓扑每个原子**都在词表里，
    而 MACE-OFF24 连 Na 都表示不了。teacher 不匹配时在这里就给出缺哪几个元素。
    """

    vocabulary = tuple(sorted({int(value) for value in atomic_numbers}))
    if not vocabulary:
        raise AutofitError("体系元素列表为空")
    if teacher is None:
        return vocabulary
    table = teacher_z_table(teacher, extra_z_tables=extra_z_tables)
    missing = [z for z in vocabulary if z not in set(table)]
    if missing:
        raise AutofitError(
            f"teacher {teacher!r} 表示不了体系里的元素 {missing}（它覆盖 "
            f"{len(table)} 种元素）。换一个覆盖更广的模型，或者不要用 teacher "
            "口径——R1 的训练信号本来就是纯 MM 的。"
        )
    return vocabulary


def derived_capacities(n_ligand_atoms: int) -> dict[str, int]:
    """容量常量按配体尺寸缩放；41 原子时与出厂那组 (320/2048/80) 逐值相同。

    `max_neighbors_per_ligand` 是**每个配体原子**的邻居上限，不随配体大小变，
    所以不缩放。
    """

    from .softlift import R1_REFERENCE_LIGAND_ATOM_COUNT

    n = int(n_ligand_atoms)
    if n <= 0:
        raise AutofitError("n_ligand_atoms 必须为正")
    scale = max(1.0, n / float(R1_REFERENCE_LIGAND_ATOM_COUNT))
    return {
        "max_environment_atoms": int(math.ceil(320 * scale)),
        "max_edges": int(math.ceil(2048 * scale)),
        "max_neighbors_per_ligand": 80,
    }


def _strip_barostats(system) -> int:
    """探针必须 NVT：`TraditionalMBARAnalyzer.compute_u_kn` 的传统 LRC 路径要求
    盒体积相对波动 < 1e-3，带恒压器一定过不了那道门（而且它会直接抛，不是静默）。"""

    removed = 0
    for index in reversed(range(system.getNumForces())):
        if "Barostat" in type(system.getForce(index)).__name__:
            system.removeForce(index)
            removed += 1
    return removed


def probe_sweep(
    *,
    system,
    topology,
    positions,
    box_vectors,
    ligand_indices: Sequence[int],
    temperature_kelvin: float,
    platform_name: str,
    lambdas_vdw: Sequence[float],
    out_dir: Path,
    frames_per_state: int = DEFAULT_FRAMES_PER_STATE,
    steps_between_frames: int = DEFAULT_STEPS_BETWEEN_FRAMES,
    equilibration_steps: int = DEFAULT_EQUILIBRATION_STEPS,
    seed: int | None = None,
    n_partitions: int = 3,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """沿 λ_vdw 梯子采训练帧，写成 `n_partitions` 条 DCD。

    每个 λ 的帧按 round-robin 分给各分区 ⟹ **每条分区都横跨整条梯子**。
    LORO 折的本意是"留一条独立 run"，这里只有一次连续轨迹，留一条"λ 子集"会让
    held-out 折落在训练折没见过的 λ 区间上，评的就不是泛化了。这是本模块相对
    EXP-020 三条独立 run 的**已知降级**。
    """

    _log = _log_to(log)
    import numpy as np
    import openmm
    from openmm import app, unit

    sys.path.insert(0, str(_REPO_ROOT)) if str(_REPO_ROOT) not in sys.path else None
    from abfe_preoptimizer import ACESoftcorePotential, build_aces_probe_system_dual_lambda

    if int(n_partitions) < 3:
        raise AutofitError("build_dataset_v1 要求至少三个分区")
    if int(frames_per_state) < n_partitions:
        raise AutofitError("frames_per_state 必须不少于分区数，否则有分区拿不到帧")

    softcore = ACESoftcorePotential.from_dict(
        ACESoftcorePotential.optimize_alpha(len(ligand_indices))
    )
    probe_system = build_aces_probe_system_dual_lambda(
        system,
        list(ligand_indices),
        softcore,
        fixed_lam_coul=0.0,
        fixed_lam_vdw=1.0,
        topology=topology,
        positions=positions,
        box_vectors=box_vectors,
    )
    removed = _strip_barostats(probe_system)
    if removed:
        _log(f"  [autofit] 探针系统移除了 {removed} 个恒压器（训练帧必须 NVT）")

    integrator = openmm.LangevinMiddleIntegrator(
        float(temperature_kelvin) * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picosecond,
    )
    if seed is not None:
        integrator.setRandomNumberSeed(int(seed))
    context = openmm.Context(
        probe_system, integrator, openmm.Platform.getPlatformByName(str(platform_name))
    )
    context.setPositions(positions)
    if box_vectors is not None:
        context.setPeriodicBoxVectors(*box_vectors)
    openmm.LocalEnergyMinimizer.minimize(context, maxIterations=500)
    context.setVelocitiesToTemperature(
        float(temperature_kelvin) * unit.kelvin, int(seed) if seed is not None else 0
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"probe_partition{index}.dcd" for index in range(n_partitions)]
    handles = [path.open("wb") for path in paths]
    writers = [
        app.DCDFile(handle, topology, 0.002 * unit.picosecond) for handle in handles
    ]
    frame_states: list[list[int]] = [[] for _ in range(n_partitions)]
    try:
        for state_index, lam in enumerate(lambdas_vdw):
            context.setParameter("lam_vdw", float(lam))
            context.setParameter("lam_coul", 0.0)
            integrator.step(int(equilibration_steps))
            for frame_index in range(int(frames_per_state)):
                integrator.step(int(steps_between_frames))
                state = context.getState(getPositions=True, enforcePeriodicBox=False)
                partition = frame_index % n_partitions
                writers[partition].writeModel(
                    state.getPositions(),
                    periodicBoxVectors=state.getPeriodicBoxVectors(),
                )
                frame_states[partition].append(state_index)
            _log(
                f"  [autofit] 探针 λ_vdw={float(lam):.4f} 采了 {frames_per_state} 帧"
            )
    finally:
        for handle in handles:
            handle.close()
        del context, integrator

    return {
        "trajectory_paths": [str(path) for path in paths],
        "frame_states": [np.asarray(states, dtype=np.int64) for states in frame_states],
        "lambdas_vdw": [float(value) for value in lambdas_vdw],
    }


def write_ledgers(
    *,
    system,
    topology,
    positions,
    box_vectors,
    ligand_indices: Sequence[int],
    temperature_kelvin: float,
    platform_name: str,
    sweep: Mapping[str, Any],
    out_dir: Path,
    log: Callable[[str], None] | None = None,
) -> list[dict[str, str]]:
    """把探针帧变成 EXP-012 口径的 MM ledger（每个分区一份 npz + 一份 report）。

    `TraditionalMBARAnalyzer.compute_u_kn` 给的就是目标态的**约化**势能
    `βU_k(x_n)`（`_compute_u_kn_chunk` 里 `e / kt`），所以：

        target_u   = u_kn.T                      (F, S)
        sampling_u = target_u[n, k(n)]           探针没有 IBS 偏置，采样态就是目标态
        adjacent_gap_reduced        = diff(target_u, axis=1)
        log_importance_unnormalized = sampling_u[:, None] - target_u

    与 `exp012_xed/mm_ledger.py:113-118` 同一套定义（那里 base 会在 diff 里抵消）。
    """

    _log = _log_to(log)
    import numpy as np

    sys.path.insert(0, str(_REPO_ROOT)) if str(_REPO_ROOT) not in sys.path else None
    from ibs_engine import TraditionalMBARAnalyzer

    lambdas_vdw = list(sweep["lambdas_vdw"])
    traj_files = list(sweep["trajectory_paths"])
    analyzer = TraditionalMBARAnalyzer(float(temperature_kelvin))
    u_kn = analyzer.compute_u_kn(
        traj_files=traj_files,
        system_template=system,
        ligand_indices=list(ligand_indices),
        lambdas_coul=[0.0] * len(lambdas_vdw),
        lambdas_vdw=lambdas_vdw,
        platform_name=str(platform_name),
        topology=topology,
        reference_positions=positions,
        reference_box_vectors=box_vectors,
    )
    n_k = np.asarray(analyzer._last_n_k, dtype=int)
    if n_k.size != len(traj_files):
        raise AutofitError("compute_u_kn 返回的每文件帧数与分区数不一致")

    runs: list[dict[str, str]] = []
    offset = 0
    for partition, frames in enumerate(n_k.tolist()):
        target_u = np.asarray(u_kn[:, offset : offset + frames], dtype=np.float64).T
        offset += frames
        states = np.asarray(sweep["frame_states"][partition], dtype=np.int64)
        if states.size != frames:
            raise AutofitError(
                f"分区 {partition} 的帧数({frames})与采样态记录({states.size})对不上"
            )
        if not np.all(np.isfinite(target_u)):
            raise AutofitError(f"分区 {partition} 的约化势能里有非有限值")
        sampling_u = target_u[np.arange(frames), states]
        ledger_path = out_dir / f"ledger_partition{partition}.npz"
        report_path = out_dir / f"ledger_partition{partition}_report.json"
        np.savez(
            ledger_path,
            frame_index=np.arange(frames, dtype=np.int64),
            adjacent_gap_reduced=np.diff(target_u, axis=1),
            log_importance_unnormalized=sampling_u[:, None] - target_u,
            sampled_state_index=states,
        )
        report_path.write_text(
            json.dumps(
                {
                    "frame_count": int(frames),
                    "state_count": int(len(lambdas_vdw)),
                    "lambdas_vdw": lambdas_vdw,
                    "temperature_kelvin": float(temperature_kelvin),
                    "source": "local_residual.autofit.probe_sweep",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        runs.append(
            {
                "run_id": f"autofit_partition{partition}",
                "trajectory_path": traj_files[partition],
                "ledger_path": str(ledger_path),
                "ledger_report_path": str(report_path),
            }
        )
        _log(f"  [autofit] 分区 {partition}: {frames} 帧 ledger 已落盘")
    return runs


def a_k_schedule(lambdas_vdw: Sequence[float]) -> tuple[list[float], list[float]]:
    """`A_k = sin^2(pi*lambda)`，端点严格 0 —— 与运行时
    `OuterLambdaController.envelope`（`outer_lambda_neural_basis.py:841`）
    逐字同一个定义。训练用的系数必须与运行时用的是同一条，否则学的不是同一个量。
    """

    values = [float(value) for value in lambdas_vdw]
    a_k = [
        0.0 if (lam <= 0.0 or lam >= 1.0) else math.sin(math.pi * lam) ** 2
        for lam in values
    ]
    delta = [a_k[index + 1] - a_k[index] for index in range(len(a_k) - 1)]
    return delta, a_k


def _protocol_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _existing_manifest_matches(
    manifest_path: Path, topology, ligand_indices: Sequence[int], system
) -> bool:
    """已有 manifest 是不是就是这个配体的。不是就重训，是就直接复用（可 resume）。"""

    if not manifest_path.is_file():
        return False
    from .openmm_plugin import _load_resource_manifest, ligand_chemical_identity

    try:
        manifest, _payload, _weights = _load_resource_manifest(manifest_path)
    except Exception:
        return False
    identity = ligand_chemical_identity(topology, ligand_indices, system=system)
    return manifest["supported_ligand"].get("fingerprint_sha256") == identity[
        "fingerprint_sha256"
    ]


def _best_checkpoint(train_report: Mapping[str, Any]) -> Path:
    """按 held-out 相对改善挑一份 checkpoint。

    LORO 的每一折都是一个独立模型，出厂链最后也是**挑一份**去导出。这里挑
    held-out 改善最高的那个 seed —— 注意这是**选型**不是验收：真正的验收是上机
    A/B（EXP-033 §4），离线 gap-variance 只是"值得上机的信号"。
    """

    best: tuple[float, str] | None = None
    for fold in train_report.get("folds", []):
        for seed_report in fold.get("seeds", []):
            improvement = float(seed_report["relative_improvement"])
            if best is None or improvement > best[0]:
                best = (improvement, str(seed_report["checkpoint"]))
    if best is None:
        raise AutofitError("训练报告里没有任何 checkpoint")
    if best[0] <= 0.0:
        raise AutofitError(
            f"所有折的 held-out gap-variance 都没有改善（最好 {best[0]:+.3f}）；"
            "拒绝冻结一个连离线信号都没有的模型。"
        )
    return Path(best[1])


def autofit_r1(
    *,
    system,
    topology,
    positions,
    box_vectors,
    ligand_indices: Sequence[int],
    temperature_kelvin: float,
    platform_name: str,
    output_dir: str | Path,
    ligand_name: str,
    topology_cif: str | Path,
    ligand_indices_path: str | Path,
    system_xml: str | Path | None = None,
    n_probe_states: int = DEFAULT_PROBE_STATES,
    probe_lambda_max: float = DEFAULT_PROBE_LAMBDA_MAX,
    probe_lambda_min: float = DEFAULT_PROBE_LAMBDA_MIN,
    frames_per_state: int = DEFAULT_FRAMES_PER_STATE,
    steps_between_frames: int = DEFAULT_STEPS_BETWEEN_FRAMES,
    equilibration_steps: int = DEFAULT_EQUILIBRATION_STEPS,
    seed: int | None = None,
    teacher: str | None = None,
    extra_z_tables: Mapping[str, Sequence[int]] | None = None,
    max_epochs: int = 500,
    patience: int = 30,
    train_seeds: Sequence[int] = (0, 1, 2),
    log: Callable[[str], None] | None = None,
) -> AutofitResult:
    """整条链：探针采样 → ledger → dataset → 训练 → 导出 → 部署 manifest。

    返回的 manifest 路径可以直接喂给
    `build_outer_lambda_local_residual_runtime(resource_manifest=...)`。
    """

    _log = _log_to(log)
    import numpy as np

    output_dir = Path(output_dir)
    resources_dir = output_dir / "resources" / "outer_lambda_local_residual"
    manifest_path = resources_dir / "manifest.json"
    if _existing_manifest_matches(manifest_path, topology, ligand_indices, system):
        _log(f"  [autofit] 复用已有的 run 内冻结模型: {manifest_path}")
        return AutofitResult(
            manifest_path=manifest_path,
            payload_path=resources_dir / "r1_model_payload_v1.json",
            weights_path=resources_dir / "r1_model_weights_f64.bin",
            report_path=output_dir / "autofit" / "autofit_report.json",
            reused_existing=True,
        )

    from .openmm_plugin import topology_atomic_numbers

    atomic_numbers = topology_atomic_numbers(topology, system=system)
    vocabulary = check_element_coverage(
        atomic_numbers, teacher, extra_z_tables=extra_z_tables
    )
    n_ligand = len(list(ligand_indices))
    capacities = derived_capacities(n_ligand)
    _log(
        f"  [autofit] 配体 {n_ligand} 原子；体系元素词表 {list(vocabulary)}；"
        f"容量 {capacities}"
    )

    work_dir = output_dir / "autofit"
    work_dir.mkdir(parents=True, exist_ok=True)
    if not 0.0 <= float(probe_lambda_min) < float(probe_lambda_max) <= 1.0:
        raise AutofitError(
            f"探针 λ 区间非法: [{probe_lambda_min}, {probe_lambda_max}]"
        )
    lambdas_vdw = np.linspace(
        float(probe_lambda_max), float(probe_lambda_min), int(n_probe_states)
    ).tolist()
    delta_A, A_k = a_k_schedule(lambdas_vdw)
    protocol_sha = _protocol_sha256(
        {
            "chain": "local_residual.autofit",
            "version": 1,
            "lambdas_vdw": lambdas_vdw,
            "frames_per_state": int(frames_per_state),
            "steps_between_frames": int(steps_between_frames),
            "equilibration_steps": int(equilibration_steps),
            "capacities": capacities,
            "type_vocabulary": list(vocabulary),
            "A_definition": "sin_squared_pi_lambda_vdw",
        }
    )

    dataset_path = work_dir / "dataset" / "softlift_dataset_v1.npz"
    dataset_report_path = dataset_path.with_name(dataset_path.stem + "_report.json")
    trajectory_paths = sorted(str(path) for path in work_dir.glob("probe_partition*.dcd"))
    # 断点续跑：上一次跑到一半（比如导出那步挂了）时，探针帧和数据集都还在。
    # `build_dataset_v1` 拒绝覆盖已有产物，所以这里必须要么复用、要么说清楚怎么清。
    # 复用的前提是**协议身份一致**——protocol_sha 是由 λ 表/采样参数/容量/词表算出来
    # 的，不含任何自产文件的字节，改了参数就一定对不上。
    if dataset_path.is_file() and dataset_report_path.is_file():
        existing = json.loads(dataset_report_path.read_text(encoding="utf-8"))
        if existing.get("protocol_sha256") != protocol_sha:
            raise AutofitError(
                f"{work_dir} 里已有一份用不同参数产出的数据集"
                f"（protocol_sha256 {existing.get('protocol_sha256')} != {protocol_sha}）。"
                f"要用新参数重跑请先删掉 {work_dir}。"
            )
        if not trajectory_paths:
            raise AutofitError(f"{work_dir} 有数据集却没有探针轨迹；删掉该目录重跑")
        _log(f"  [autofit] 复用上次的数据集，跳过探针采样: {dataset_path}")
        return _autofit_from_dataset(
            dataset_path=dataset_path,
            first_trajectory=trajectory_paths[0],
            work_dir=work_dir,
            resources_dir=resources_dir,
            manifest_path=manifest_path,
            topology_cif=topology_cif,
            ligand_indices_path=ligand_indices_path,
            system_xml=system_xml,
            ligand_name=ligand_name,
            protocol_sha=protocol_sha,
            vocabulary=vocabulary,
            capacities=capacities,
            n_ligand=n_ligand,
            lambdas_vdw=lambdas_vdw,
            teacher=teacher,
            max_epochs=max_epochs,
            patience=patience,
            train_seeds=train_seeds,
            log=log,
        )

    sweep = probe_sweep(
        system=system,
        topology=topology,
        positions=positions,
        box_vectors=box_vectors,
        ligand_indices=ligand_indices,
        temperature_kelvin=temperature_kelvin,
        platform_name=platform_name,
        lambdas_vdw=lambdas_vdw,
        out_dir=work_dir,
        frames_per_state=frames_per_state,
        steps_between_frames=steps_between_frames,
        equilibration_steps=equilibration_steps,
        seed=seed,
        log=log,
    )
    runs = write_ledgers(
        system=system,
        topology=topology,
        positions=positions,
        box_vectors=box_vectors,
        ligand_indices=ligand_indices,
        temperature_kelvin=temperature_kelvin,
        platform_name=platform_name,
        sweep=sweep,
        out_dir=work_dir,
        log=log,
    )

    _check_reweighting_sanity(runs, log=log)

    from .softlift_dataset import build_dataset_v1

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    build_dataset_v1(
        runs=runs,
        ligand_topology_indices=list(ligand_indices),
        topology_path=topology_cif,
        output_path=dataset_path,
        delta_A=delta_A,
        A_k_window=A_k,
        protocol_sha256=protocol_sha,
        # 显式给元素：mdtraj 从 cif 读不到元素时给的是 `element.virtual`
        # （atomic_number = 0），不抛错，会把 0 静默混进类型词表。
        atomic_numbers_override=atomic_numbers,
        # 深度解耦端偶尔会有环境原子压到鬼影配体上（真实构型，但落在模型支撑域外）。
        # 跳过这种帧，跳多了才报错——那说明采样区间选错了。
        skip_unsupported_frames=True,
        **capacities,
    )
    _log(f"  [autofit] 数据集已落盘: {dataset_path}")
    _check_skipped_frame_budget(
        dataset_path.with_name(dataset_path.stem + "_report.json"), log=log
    )

    return _autofit_from_dataset(
        dataset_path=dataset_path,
        first_trajectory=sweep["trajectory_paths"][0],
        work_dir=work_dir,
        resources_dir=resources_dir,
        manifest_path=manifest_path,
        topology_cif=topology_cif,
        ligand_indices_path=ligand_indices_path,
        system_xml=system_xml,
        ligand_name=ligand_name,
        protocol_sha=protocol_sha,
        vocabulary=vocabulary,
        capacities=capacities,
        n_ligand=n_ligand,
        lambdas_vdw=lambdas_vdw,
        teacher=teacher,
        max_epochs=max_epochs,
        patience=patience,
        train_seeds=train_seeds,
        log=log,
    )


def _autofit_from_dataset(
    *,
    dataset_path: Path,
    first_trajectory: str,
    work_dir: Path,
    resources_dir: Path,
    manifest_path: Path,
    topology_cif,
    ligand_indices_path,
    system_xml,
    ligand_name: str,
    protocol_sha: str,
    vocabulary,
    capacities,
    n_ligand: int,
    lambdas_vdw,
    teacher,
    max_epochs: int,
    patience: int,
    train_seeds,
    log,
) -> AutofitResult:
    """数据集之后的三步：训练 → 导出 → manifest。单独拆出来是为了能从这里续跑。"""

    _log = _log_to(log)
    train_root = work_dir / "training"
    train_report_path = train_root / "r1_density" / "r1__direct_gap__d1_report.json"
    if train_report_path.is_file():
        # 训练脚本同样拒绝覆盖 checkpoint，续跑时不能再训一遍。
        _log(f"  [autofit] 复用上次的训练结果: {train_report_path}")
    elif (train_root / "r1_density").is_dir():
        raise AutofitError(
            f"{train_root / 'r1_density'} 里有上次没跑完的 checkpoint 但没有报告；"
            "训练脚本拒绝覆盖 checkpoint，删掉该目录后重跑"
        )
    else:
        _run_script(
            "train_exp019_softlift_loro",
            [
                "--dataset", str(dataset_path),
                "--rung", "R1",
                "--allow-derived-r1-config",
                "--max-epochs", str(int(max_epochs)),
                "--patience", str(int(patience)),
                "--seeds", *[str(int(value)) for value in train_seeds],
                "--output-root", str(train_root),
            ],
        )
    train_report = json.loads(train_report_path.read_text(encoding="utf-8"))
    checkpoint = _best_checkpoint(train_report)
    _log(
        f"  [autofit] 选中 {checkpoint.name}（held-out 平均改善 "
        f"{train_report['mean_relative_improvement']:+.3f}）"
    )

    export_dir = work_dir / "export"
    if (export_dir / "r1_model_payload_v1.json").is_file():
        _log(f"  [autofit] 复用上次的导出产物: {export_dir}")
    else:
        _run_script(
            "export_exp025_g1_reference_payload",
            [
                "--allow-derived-r1-config",
                "--checkpoint", str(checkpoint), "--checkpoint-sha256", "none",
                "--dataset", str(dataset_path), "--dataset-sha256", "none",
                "--ligand-indices", str(ligand_indices_path), "--ligand-indices-sha256", "none",
                "--topology", str(topology_cif), "--topology-sha256", "none",
                *(["--system", str(system_xml)] if system_xml is not None else []),
                "--trajectory", first_trajectory, "--trajectory-sha256", "none",
                "--expected-active-edges", "none", "--expected-atom-count", "none",
                "--output-dir", str(export_dir),
            ],
        )

    # manifest 里记的是 payload/weights 的**文件名**，相对 manifest 所在目录解析
    # （`write_local_residual_resource_manifest.py:102/106`），所以三个文件必须同目录。
    # 留在 export/ 下的话，生成器最后那道回读校验当场就失败。
    resources_dir.mkdir(parents=True, exist_ok=True)
    payload_path = resources_dir / "r1_model_payload_v1.json"
    weights_path = resources_dir / "r1_model_weights_f64.bin"
    for source, destination in (
        (export_dir / payload_path.name, payload_path),
        (export_dir / weights_path.name, weights_path),
    ):
        if not source.is_file():
            raise AutofitError(f"导出步骤没有产出 {source}")
        shutil.copyfile(source, destination)
    manifest_argv = [
        "--payload", str(payload_path),
        "--weights", str(weights_path),
        "--topology", str(topology_cif),
        "--ligand-indices", str(ligand_indices_path),
        "--ligand-name", str(ligand_name),
        "--experiment-id", "AUTOFIT",
        "--output", str(manifest_path),
    ]
    if system_xml is not None:
        manifest_argv += ["--system", str(system_xml)]
    _run_script("write_local_residual_resource_manifest", manifest_argv)

    report_path = work_dir / "autofit_report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "local-residual-autofit-v1",
                "protocol_sha256": protocol_sha,
                "ligand_name": str(ligand_name),
                "n_ligand_atoms": n_ligand,
                "type_vocabulary": list(vocabulary),
                "capacities": capacities,
                "teacher_checked": teacher,
                "lambdas_vdw": lambdas_vdw,
                "selected_checkpoint": str(checkpoint),
                "training_report": str(train_report_path),
                "mean_relative_improvement": train_report["mean_relative_improvement"],
                "qualification_offline_only": train_report["qualification"],
                "manifest": str(manifest_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _log(f"  [autofit] 冻结产物已就位: {manifest_path}")
    return AutofitResult(
        manifest_path=manifest_path,
        payload_path=payload_path,
        weights_path=weights_path,
        report_path=report_path,
        reused_existing=False,
    )



#: 支撑域外的帧最多能跳掉多大比例。超过就不是个别构型的问题，是探针 λ 区间选错了。
MAX_SKIPPED_FRAME_FRACTION = 0.05


def _check_skipped_frame_budget(
    dataset_report_path: Path, *, log: Callable[[str], None] | None = None
) -> None:
    _log = _log_to(log)
    report = json.loads(Path(dataset_report_path).read_text(encoding="utf-8"))
    runs = report.get("inputs", {})
    total = sum(int(item.get("frame_count", 0)) for item in runs.values())
    skipped = sum(int(item.get("skipped_unsupported_frames", 0)) for item in runs.values())
    if total <= 0:
        raise AutofitError("数据集报告里没有任何帧")
    fraction = skipped / total
    if fraction > MAX_SKIPPED_FRAME_FRACTION:
        raise AutofitError(
            f"{skipped}/{total} 帧（{fraction:.1%}）落在模型支撑域之外被跳过，"
            f"超过上限 {MAX_SKIPPED_FRAME_FRACTION:.0%}。这不是个别构型的问题："
            "探针 λ 区间在深度解耦端待得太久，或者配体尺寸/容量常量不对。"
        )
    if skipped:
        _log(f"  [autofit] 跳过 {skipped}/{total} 帧（支撑域外，{fraction:.1%}）")



#: 单帧 |log_importance| 的上限。超过它的帧权重是 exp(-1e4)=0，只会污染数值：
#: 归一化后列和的舍入误差会超过 loss 里 32*eps 的严格判据，报出一句跟真因毫无
#: 关系的 `normalized_weights must sum to one`。
MAX_ABS_LOG_IMPORTANCE = 1.0e4
#: 允许超限的帧比例。超过说明是**区间选错**，不是个别构型。
MAX_DEGENERATE_FRAME_FRACTION = 0.05


def _check_reweighting_sanity(
    runs: Sequence[Mapping[str, str]], *, log: Callable[[str], None] | None = None
) -> None:
    """探针区间是不是窗口形状的，用 ledger 自己的数字判，别等死在 loss 里。

    鬼影构型被重加权到耦合态上会给出天文数字的约化势能（实测 1e13 kT）。这种帧
    权重恒为 0、对训练毫无贡献，却会把归一化的数值精度毁掉。它们成片出现只有一个
    原因：探针 λ 区间横跨了整条梯子，而不是一个窗口。
    """

    _log = _log_to(log)
    import numpy as np

    total = 0
    degenerate = 0
    worst = 0.0
    for run in runs:
        with np.load(run["ledger_path"]) as ledger:
            values = np.abs(np.asarray(ledger["log_importance_unnormalized"], dtype=np.float64))
        per_frame = values.max(axis=1) if values.size else np.zeros(0)
        total += int(per_frame.size)
        degenerate += int((per_frame > MAX_ABS_LOG_IMPORTANCE).sum())
        worst = max(worst, float(per_frame.max()) if per_frame.size else 0.0)
    if total <= 0:
        raise AutofitError("ledger 里没有任何帧")
    fraction = degenerate / total
    if fraction > MAX_DEGENERATE_FRAME_FRACTION:
        raise AutofitError(
            f"{degenerate}/{total} 帧（{fraction:.1%}）的 |log_importance| 超过 "
            f"{MAX_ABS_LOG_IMPORTANCE:.0e}（最大 {worst:.3g}），超过上限 "
            f"{MAX_DEGENERATE_FRAME_FRACTION:.0%}。"
            "这说明探针 λ 区间不是窗口形状的：深度解耦端采到的鬼影构型被重加权到"
            "耦合态上，r⁻¹² 直接爆炸。把 probe_lambda_min 提上来（默认 "
            f"{DEFAULT_PROBE_LAMBDA_MIN}），别让区间横跨整条梯子。"
        )
    if degenerate:
        _log(f"  [autofit] {degenerate}/{total} 帧权重退化（最大 {worst:.3g}），在允许范围内")


def _run_script(module_name: str, argv: list[str]) -> None:
    """调 `scripts/` 下那三个已有生成器的 `main(argv)`，不 subprocess。

    它们本来就是 argparse 入口，出厂那条手工链跑的也是同一批代码；这里只是把
    命令行换成参数列表，避免重写一份并行实现。
    """

    import importlib.util

    path = _REPO_ROOT / "scripts" / f"{module_name}.py"
    if not path.is_file():
        raise AutofitError(f"缺少重训链脚本: {path}")
    spec = importlib.util.spec_from_file_location(f"_autofit_{module_name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # 🔑 必须先注册进 sys.modules 再 exec：`@dataclass` 在处理类时会去
    # `sys.modules[cls.__module__].__dict__` 找注解用的命名空间（CPython
    # `dataclasses._is_type`），模块不在表里就是
    # `AttributeError: 'NoneType' object has no attribute '__dict__'`。
    # 这三个脚本里都有 frozen dataclass，所以必踩。
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        status = module.main(argv)
    finally:
        sys.modules.pop(spec.name, None)
    if status not in (None, 0):
        raise AutofitError(f"{module_name} 退出码 {status}")


def startup_element_gate(
    *,
    topology,
    autofit_enabled: bool,
    system=None,
    teacher: str | None = None,
    resource_manifest: str | Path | None = None,
    extra_z_tables: Mapping[str, Sequence[int]] | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[int, ...]:
    """拿到输入拓扑的第一时间就判"元素超了没有"，不等到建窗口才炸。

    元素集合是拓扑的性质，不需要跑任何 MD、不需要建 System 就能知道，所以这道门
    应该在最前面。两种情形：

    * `autofit_enabled` —— 词表由本体系推出来，**元素不可能超**；这里只在指定了
      teacher 时校验那个模型覆不覆盖得住。
    * 否则 —— 用的是冻结 manifest 里那份固定词表，逐个原子比对，缺哪些一次报全。
      不比对的话，同样的失败会推迟到 `atom_type_index_for_topology`（建 runtime
      时）才发生，而那时 System 已经建完并落盘了。
    """

    _log = _log_to(log)
    from .openmm_plugin import topology_atomic_numbers

    vocabulary = tuple(sorted(set(topology_atomic_numbers(topology, system=system))))
    if teacher is not None:
        check_element_coverage(vocabulary, teacher, extra_z_tables=extra_z_tables)
        _log(f"  [autofit] teacher {teacher!r} 覆盖体系全部 {len(vocabulary)} 种元素")
    if autofit_enabled:
        return vocabulary

    from .openmm_plugin import _load_resource_manifest, load_r1_payload

    manifest_path = (
        Path(resource_manifest)
        if resource_manifest is not None
        else _REPO_ROOT / "resources/outer_lambda_local_residual/manifest.json"
    )
    _manifest, payload_path, weights_path = _load_resource_manifest(manifest_path)
    frozen = tuple(int(value) for value in load_r1_payload(payload_path, weights_path).type_vocabulary)
    missing = [z for z in vocabulary if z not in set(frozen)]
    if missing:
        raise AutofitError(
            f"冻结 R1 模型的词表 {sorted(frozen)} 覆盖不了本体系的元素 {missing}。"
            "加 --outer-lambda-autofit 按本体系重训一份（词表跟着体系走），"
            "或者关掉 residual 开关。"
        )
    return vocabulary


# =============================================================================
# 为什么撤（2026-09-11，与 atenolol-rank11 档案会话核对后）
# =============================================================================
# 这条链一度接在 `runabfe.py`（开关 `--outer-lambda-autofit`）上，已摘除。三条理由，
# 从浅到深：
#
# 1. **违反预注册。** EXP-030_MAINLINE_INTEGRATION_AND_NEW_SYSTEM_PLAN_2026-08-30.md
#    §6.2：「如果模型不覆盖新体系，停止；不能拿 production 结果反向调模型后继续沿用
#    同一预注册。」用 A/B 那条臂自己的数据拟合那条臂要用的模型，比的就不再是方法。
#
# 2. **身份链变成 run-dependent。** B_φ 进 sampling Hamiltonian，f_k 是在线对着那个
#    Hamiltonian 学的。B_φ 按本次 run 重拟之后，`sampling_score_sha256` / 冻结 manifest
#    不再是两臂共用的同一把尺子 ⟹ 两臂不可配对、resume 身份校验会被自己新生成的
#    manifest 骗过去、跨 repeat/跨 seed 的比较失去意义。
#
# 3. **最要命的一层：拿来重训的帧本身就缺构型。** stage2 每个窗口只跑一条轨迹、窗内
#    所有 λ 态靠重加权覆盖，配体消失时空腔塌缩+水灌入这个结构性慢模态根本没被采到
#    （docs/STAGE2_ROOT_CAUSE_2026-08-28.md）。用这批帧拟 B_φ 等于把缺掉的那个构象态
#    焊进模型：模型学会在已采到的区域上把混合做得更漂亮，没采到的态连梯度都拿不到。
#    然后 ESS / overlap / split-half / MBAR-BAR-TI 三方一致**全都会更绿**——它们只问
#    "这批样本内部自洽吗"。实测参照：门全绿（mixture overlap 0.4684、converged=True）
#    时 ΔG 可以错 42 kJ/mol（raw overlap 只有 0.0196）。
#    ⟹ 自适应重训 + 对慢模态失明的门 = 一个自我确认的闭环。
#
# 定位纠正：B_φ **不是**"这个配体的物理模型"，而是提高 λ 态之间混合、以及相邻窗口
# 共享态那条缝上衔接的**采样增强项**。按体系学的只有 f_k（在线，`ibs_engine.py:8593`，
# IBS_BIAS_PROTOCOL_VERSION 33）。模型不覆盖新体系 ⟹ 停下来报告，不是就地重训。
#
# 另记一条判据层级错误：本模块用「所有 LORO 折的 held-out gap-variance 都没改善才
# 拒绝冻结」当硬门。那个量只能证明"B_φ 在给它的那批帧上泛化得动"，不能证明"那批帧
# 代表真实系综"，因此只配当过拟合的内部诊断，永远不能当冻结/放行判据。
