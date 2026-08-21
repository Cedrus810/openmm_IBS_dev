"""C3：真实体系 λ=1/λ=0 端点能量与力恒等式验证。

对应 `memtodolist.md` §「C3：真实体系 λ=1/λ=0 端点能量和力」。C3 不跑新的
自由能、不重新采样——它是一套 Hamiltonian 端点恒等式测试：

## Protocol v2（2026-08-11，用户审阅批准；取代 v1 的单层力门）

v1 只有一层门：production（真实平台）vs reference，energy/force 都是硬门。
2026-08-11 用 `diagnose_coion_parameteroffset_mixed_precision.py` 做归因
诊断后发现：production 与独立 reference 在 **Reference 平台（双精度）上
逐位完全相同**（同一帧的 `E_B==E_C` 到小数点后六位，力差精确为 `0.0`）——
证明两侧构造的确实是同一个 Hamiltonian，不存在真实的构造差异。差异只在
**CUDA mixed precision** 下出现（某些帧力差远超 1e-3 门），且集中在配体内部
exception 数量较多的原子——production 与 reference 给 `NonbondedForce` 添加
L-L exception 的**顺序不同**（数值相同），CUDA mixed precision 的
direct-space exception 核算对这种顺序差异敏感，是**平台数值路径问题**，
不是物理构造错误。这个差异的大小依赖于"哪两种等价的 exception 排列碰在
一起"，没有稳定的物理含义，**不能靠从已观测到的最大值反推一个新的绝对力
容差**（那是事后定门）。

因此 v2 把验收拆成两层，物理正确性判据完全不放宽：

1. **独立 Hamiltonian 恒等性门（Reference 平台，权威）**：production（显式
   设 global parameter）vs 独立 reference。energy 相对差 `≤1e-5`、力差
   `≤1e-3 kJ/mol/nm`，两者都是硬门——这是 A/B/C/D 构造是否正确的唯一权威
   判据。
2. **真实 CUDA mixed 门**（验证生产实际会用的精度配置本身没有问题）：
   - **live ParameterOffset vs baked production**（`A` vs
     `bake_global_parameter_into_fixed_nonbonded_force(A)`）：energy/force
     都仍是硬门——这条已经被归因诊断证明干净（5 帧 direct-space 精确为
     `0.0`），如果这条都不过，说明烘焙函数本身出了问题，必须当场拦下。
   - **production vs 独立 reference**（`A` vs `C`，与门 1 同一对 System，
     只是换到 CUDA mixed 平台）：energy 相对差 `≤1e-5` 仍是硬门；力差改成
     **诊断项，不参与 `passed` 判定**（`force_gate_mode="diagnostic"`）——
     取消的只是"CUDA mixed 下两种数值相同但排列不同的 exception 必须打到
     `1e-3` 力差"这一条已经证明不成立的要求，不是取消力差检查本身（数值
     仍然全程记录、仍然落盘）。
   - 无论哪一层，**所有力必须是有限值**（NaN/Inf 直接 fail，与 force_gate_mode
     无关）；D 的严格零门（λ_vdw=0 时 `|E_ligand-environment|≤1e-6`）不受
     影响，继续是硬门（它是代数结构性的零，不依赖数值精度）。

v1 的 FAIL 记录（`validation/c3_real_endpoints_v1/`）保留不覆盖——v2 是重新
用同一批已有的 100 帧数据后处理，不是重新采样、不是推翻 v1 的发现。

## Protocol v2 的 C/D 应用（2026-08-11，用户明确指示，不新跑 vanishing MD）

C/D 曾经因为"带净电的 charge-transfer 配体接入 vanishing 阶段在生产代码里
尚未实现"而只能在合成 fixture 上验证（见下一节）。Stage2 handoff 落地之后
（`abfe_core.bake_global_parameter_into_fixed_nonbonded_force` +
`abfe_pipeline.py` 的 vanishing 分支接线），这个限制已经解除——真实 C1/C2
带电探针配体现在可以走完整的"charging 配置 → 烘焙 → 喂给
`build_ibs_dual_system`"链路。**C/D 是同坐标下的 Hamiltonian 端点恒等式，
不是新的自由能采样**：直接复用 charging λ_coul=0 已经采样过的真实轨迹帧
（`C1_LAMBDA0_FRAME_INDICES`/`C2_LAMBDA0_FRAME_INDICES`，与 A/B 的 B 端点
共用同一组 10 帧/case），不需要跑新的 vanishing MD（`run_protocol_v2_matrix_cd`）。

对每一帧：

    C：baked charging λ_coul=0（`charging0_baked`）
       vs production vanishing λ_vdw=1（`vanishing_one`）
       ——两侧都是生产代码路径，测两阶段接缝自洽性，不是独立构造。
    D：production vanishing λ_vdw=0（`vanishing_zero`）
       vs 独立构造的 `reference_vanishing_zero_system`
       ——额外跑 `compare_vanishing_zero_endpoint` 的严格零门。

同样套用两层门原则：Reference 平台上能量/力都是硬门；CUDA mixed 下能量硬门、
"两种等价 exception 排列导致的力差"降级为诊断。**但 C/D 有一处 A/B 没有的
结构性差异**：`production_vanishing_fixed_hamiltonian_systems` 把每个
λ_vdw 态的软核 CV 构造成固定表达式（不是运行时可设的 GlobalParameter），
charging 侧在喂给 vanishing 之前已经被烘焙掉了——比较发生的时刻，C 和 D
两侧都**没有活的 GlobalParameter**。这意味着"live ParameterOffset vs baked
production"（gate2）在这里没有对应的比较对象，标记为
`applicable=False`、不参与 `passed`，不为了凑够三层门而制造一个无意义比较。

D 的严格零门（`strict_zero_reference`/`strict_zero_mixed`）在 Reference 和
CUDA mixed 下都**继续是硬门**，不受 force-gate 诊断化影响——它检验的是
λ_vdw≡0 时软核 CV 系数结构性归零，这是代数事实，不依赖数值精度，跟 gate3
诊断化针对的"两种等价 exception 排列在 mixed precision 下的力差"是完全
不同的两件事。

## MEM-00h 双边归一化（2026-08-11，用户对第一版消融实验的修正）

第一次跑真实 C/D 数据时，C（seam）在 C2 四格上大量失败（Gate1，Reference
双精度硬门），力差最坏到 0.64 kJ/mol/nm。**根因**：C2 自己的 raw
`system.xml` 的 `NonbondedForce` 带一个 C2 专用的 `[0.995,1.0]nm` LJ
switch（`validate_charge_transfer_lipid_slab.py` v2→v3，2026-08-07，为修
`MonteCarloMembraneBarostat` 造成的膜面内人工压缩而加的；当时代码注释已经
写明"范围只到 C2 自己的 System 构建，不改全局 MEM-00h 常量……需要独立决策"），
而 vanishing 阶段的配体–环境软核 CV 一直遵循全局 MEM-00h 的无 switch 约定
（`ibs_engine.SOFTCORE_CUTOFF_NM`/`use_switching=False`）。逐帧统计证实：
只要该帧配体–环境有至少一对 LJ-active 原子对落在这个窗口内，C 就出现非零
力差；零对时精确为 0。

**第一版试图"关掉一侧的 switch 再比"的消融实验是错的**（用户指出）：只对
`charging0_baked` 一侧关闭 switch，`vanishing_one` 的 Group0（环境–环境，
继承自同一份 raw System）却仍然带着 switch——制造了一个新的、更大范围的
环境–环境不一致，而不是移除原来那个局部的配体–环境不一致，所以所有帧都
变得更差，这个结果本身不能反驳"switch 不一致是根因"这个结论。

**正确修法（用户指定）**：`mem00h_normalized_raw_system()` 在分支出
charging/baked/vanishing/reference 之前，先对**共同的** raw System clone
做一次统一转换——`NonbondedForce.setUseSwitchingFunction(False)`（cutoff
只做核验、不强制改写，不等于 `MEM00H_CUTOFF_NM` 就直接 fail closed）。A/B/
C/D 全部从这同一份归一化 clone 分别构造，因此下游任何一步都不会再各自继承
彼此不一致的 switch 设置。`assert_mem00h_switching_convention()` 是配套的
结构核验，在关键构造节点核对 `NonbondedForce` 确实是 `cutoff=1.0nm,
switching=False`——不能只信"我调用过归一化函数"，要核实它真的传导到了最终
喂给 Context 的 System 上。**不改 C2 已有的 raw 文件/轨迹本身**——归一化
只发生在 C3 评估工具内部的一份内存 clone 上，C2 生产采样用的仍然是它自己
原来的 switch 惯例（继续解决膜压缩问题），两者互不影响。`run_protocol_v2_
matrix`（A/B）和 `run_protocol_v2_matrix_cd`（C/D）都在 `load_case_raw_
inputs()` 之后立即调用这个归一化——C1 的 raw System 本来就没有局部
switch，归一化是 no-op，A/B/C1-C/D 的既有干净结果不受影响；只有 C2 的 C
（seam）在归一化后从大量失败变回机器精度量级。

## 四条恒等式

    生产 builder 构造出的端点   vs   完全独立、直接改粒子参数构造的 reference 端点

详见 memtodolist.md 的表：

    A：charging λ_coul=1        vs 原始物理体系（配体满电、co-ion 中性 dummy）
    B：charging λ_coul=0        vs 直接把配体电荷置零、co-ion 充满
    C：vanishing λ_vdw=1        vs charging λ_coul=0（两阶段接缝）
    D：vanishing λ_vdw=0        vs 独立构造的 ligand-environment 全删 reference

C/D 对**净中性配体**直接吃 `raw_system`（`production_vanishing_fixed_
hamiltonian_systems` 的 docstring）：`build_ibs_dual_system` 自己的静态电
中性防御要求配体在喂给它的这份电荷上净和为 0。带净电的 charge-transfer
配体（C1/C2 用的单原子探针）不能直接吃原始电荷——**2026-08-11 Stage2
handoff 落地后**，改吃"charging 配置 → `bake_global_parameter_into_fixed_
nonbonded_force` 结构性烘焙成 λ_coul=0 的静态电荷"之后的 System（此时配体
电荷已经转移给 co-ion、净和重新为 0，满足静态防御），与
`abfe_pipeline.py::ABFEPipeline._run_dual_lambda_stage` 的真实生产接线逐字
一致。见上一节"Protocol v2 的 C/D 应用"和 `run_protocol_v2_matrix_cd`——
C1/C2 的真实带电探针数据现在可以直接跑 C/D，不再局限于合成中性配体。
合成 fixture（下面 `_build_neutral_system`/`_case` 系列 CPU 契约测试）继续
保留，作为不依赖真实数据、跑得快的结构性回归。

## Reference 侧的独立性（硬约束）

Reference builder 只允许读取 raw `system.xml`、`topology.cif`、
`ligand_indices.json`、冻结的 `coalchemical_ion_spec.json` 和当前帧的
positions/box。**禁止调用**（见 `FORBIDDEN_REFERENCE_CALLABLES`）：

    abfe_core.co_alchemical_charge_offset_plan
    abfe_core.create_ligand_internal_force
    ibs_engine.configure_charge_transfer_decharging
    ibs_engine.configure_pme_ligand_charge_offsets
    ibs_engine.configure_coalchemical_neutral_decharging
    ibs_engine.select_co_alchemical_ion_once
    ibs_engine.build_ibs_dual_system

`forbidden_calls_disabled()` 把这些函数换成"调用即抛异常"，供测试证明 reference
builder 真正独立（memtodolist.md §10 的建议）。`verify_co_alchemical_ion_identity`
不在禁止列表里——它是只读核对，不产生任何电荷/restraint 副作用（已核实：函数体内
没有任何 `setParticleParameters`/`addForce`/`addParameterOffset` 调用）。

## 生产侧允许（也应该）调用被测 builder

生产侧的职责恰恰是正常调用上面列出的函数——它们就是被测对象。`production_*`
系列函数都是薄封装：clone 一份 System，然后调用真实的 `ibs_engine`/`abfe_core`
入口。

## 数值门（与 memtodolist.md 完全一致，直接复用 `abfe_core` 里已冻结的常量）

    energy relative difference = abs(Eprod-Eref) / max(1 kJ/mol, abs(Eref)) ≤ 1e-5
    max |ΔF_atom_component| ≤ 1e-3 kJ/mol/nm
    vanishing λ=0 额外要求：|E_ligand-environment| ≤ 1e-6 kJ/mol，
        max|F_ligand-environment| ≤ 1e-3 kJ/mol/nm，LRC coefficient = 0

## 求值纪律（memtodolist.md §9）

每次求值都是：`setPeriodicBoxVectors()` → `setPositions()` →
（若有 virtual site）`computeVirtualSites()` → 显式设置所有 global parameter →
零步直接取 energy/forces。禁止 minimize / applyConstraints / 积分一步 /
用初始 `box_vectors_nm.npy` 代替逐帧盒。
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import openmm
from openmm import app, unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import abfe_core as core  # noqa: E402
import ibs_engine as ie  # noqa: E402

# ---------------------------------------------------------------------------
# 协议版本与容差（与 memtodolist.md §C3-0 逐字一致，全部直接复用
# abfe_core 里已经冻结的常量——不重新定义一份，写歪了会自己对上自己）。
# ---------------------------------------------------------------------------

PROTOCOL_VERSION = 2  # v1→v2（2026-08-11）：拆成"Reference 权威恒等性门" +
# "CUDA mixed 门（力差诊断化，energy/finite 仍硬门）"两层，见模块 docstring。

ENERGY_RELATIVE_TOLERANCE = core.ENDPOINT_ENERGY_RELATIVE_TOLERANCE
FORCE_ABS_TOLERANCE_KJ_MOL_NM = core.ENDPOINT_FORCE_MAX_ABS_TOLERANCE_KJ_PER_MOL_NM
DECOUPLED_ABS_TOLERANCE_KJ_MOL = core.DECOUPLED_ENDPOINT_ENERGY_ABS_TOLERANCE_KJ_PER_MOL

# MEM-00h 生产 baseline（C3 硬验收基线，与 C2 专用的窄 switching 不是一回事——
# 见 memtodolist.md §7）。
MEM00H_CUTOFF_NM = 1.0
MEM00H_SWITCHING_ENABLED = False

# Reference builder 绝对不能调用的生产函数：一处集中声明，供
# `forbidden_calls_disabled()` 与文档共用同一份真相。
FORBIDDEN_REFERENCE_CALLABLES: Tuple[Tuple[str, str], ...] = (
    ("abfe_core", "co_alchemical_charge_offset_plan"),
    ("abfe_core", "create_ligand_internal_force"),
    ("ibs_engine", "configure_charge_transfer_decharging"),
    ("ibs_engine", "configure_pme_ligand_charge_offsets"),
    ("ibs_engine", "configure_coalchemical_neutral_decharging"),
    ("ibs_engine", "select_co_alchemical_ion_once"),
    ("ibs_engine", "build_ibs_dual_system"),
)


# ---------------------------------------------------------------------------
# 哈希 / manifest 约定（与 tools/validation/validate_charge_transfer_*.py 共用
# 同一套：_sha256_file 逐块读文件，_canonical_fingerprint 走 abfe_core）。
# ---------------------------------------------------------------------------


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(paths: Iterable[str]) -> Dict[str, str]:
    return {
        str(Path(p).resolve().relative_to(_REPO_ROOT)): sha256_file(p) for p in paths
    }


# ---------------------------------------------------------------------------
# 独立性防护：把禁止列表里的函数换成"调用即抛异常"。
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def forbidden_calls_disabled():
    """在这个上下文里，`FORBIDDEN_REFERENCE_CALLABLES` 里任何一个被调用就抛
    `AssertionError`。测试用它包住 reference builder 调用，证明其真正独立于生产
    planner——不是"看代码大概没调"，是"调了就当场炸"。
    """
    saved: List[Tuple[Any, str, Any]] = []
    for mod_name, attr_name in FORBIDDEN_REFERENCE_CALLABLES:
        module = sys.modules.get(mod_name) or importlib.import_module(mod_name)
        original = getattr(module, attr_name)

        def _forbidden(*_args, __mod=mod_name, __attr=attr_name, **_kwargs):
            raise AssertionError(
                f"reference builder 调用了被禁止的生产函数 {__mod}.{__attr}()"
                "——reference 必须只读取 raw system/topology/spec/frame 并自己"
                "重新实现电荷/restraint 映射，不能借用被测 builder。"
            )

        saved.append((module, attr_name, original))
        setattr(module, attr_name, _forbidden)
    try:
        yield
    finally:
        for module, attr_name, original in saved:
            setattr(module, attr_name, original)


# ---------------------------------------------------------------------------
# System 操作的小工具
# ---------------------------------------------------------------------------


def _clone_system(system: openmm.System) -> openmm.System:
    """深拷贝一份 System，绝不与调用方共享任何 OpenMM 对象。"""
    return openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))


def _find_nonbonded_force(system: openmm.System) -> openmm.NonbondedForce:
    nb = next(
        (f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)), None
    )
    if nb is None:
        raise RuntimeError("System 中没有 NonbondedForce。")
    return nb


def _ion_indices_from_spec(spec: Dict[str, Any]) -> List[int]:
    return sorted(int(ion["atom_index"]) for ion in spec["ions"])


def _freeze_ligand_internal_pairs(
    nb: openmm.NonbondedForce,
    ligand_indices: Sequence[int],
    physical_params: Dict[int, Tuple[float, float, float]],
) -> None:
    """把配体每一对没有既有 exception 的原子对，冻结成显式的物理 exception。

    与生产侧（`configure_charge_transfer_decharging` 里的同名逻辑）算的是同一件
    事，但这里是独立重新实现——不调用、不 import 那段代码。已经存在的 exception
    （通常是力场自带的 1-2/1-3/1-4）原样保留，不去覆盖。
    """
    ligand_list = sorted(int(i) for i in ligand_indices)
    ligand_set = set(ligand_list)
    existing_ll_pairs = set()
    for exc_idx in range(nb.getNumExceptions()):
        p1, p2, _cp, _s, _e = nb.getExceptionParameters(exc_idx)
        p1, p2 = int(p1), int(p2)
        if p1 in ligand_set and p2 in ligand_set:
            existing_ll_pairs.add((min(p1, p2), max(p1, p2)))

    for offset_i, p1 in enumerate(ligand_list):
        q1, sig1, eps1 = physical_params[p1]
        for p2 in ligand_list[offset_i + 1 :]:
            key = (p1, p2)
            if key in existing_ll_pairs:
                continue
            q2, sig2, eps2 = physical_params[p2]
            nb.addException(
                p1,
                p2,
                (q1 * q2) * unit.elementary_charge**2,
                0.5 * (sig1 + sig2) * unit.nanometer,
                math.sqrt(max(eps1 * eps2, 0.0)) * unit.kilojoule_per_mole,
                True,
            )


# ---------------------------------------------------------------------------
# Reference builders（A/B/D）—— 只读 raw system + 冻结 spec，独立重新实现。
# ---------------------------------------------------------------------------


def reference_charging_endpoint_system(
    raw_system: openmm.System,
    ligand_indices: Sequence[int],
    spec: Dict[str, Any],
    lam: float,
) -> openmm.System:
    """charging 端点（A: λ=1 / B: λ=0）的独立参照 System。

    λ 的映射直接抄 spec 里已经算好的端点电荷（`charge_at_lambda1_e`/
    `charge_at_lambda0_e`），不重新调用 `co_alchemical_charge_offset_plan`——
    那份 plan 本身就在禁止列表里。这里只是把"λ=1/λ=0 时每个粒子的电荷是多少"
    这件事，用 spec 记录的数、手写进一个干净的 System。

    λ=1：配体保持物理电荷（就是原始物理体系本身，不需要改动），co-ion 电荷设为
    `charge_at_lambda1_e`（charge-transfer 路线下应为 0，与 raw system 一致）。
    λ=0：配体电荷清零，co-ion 电荷设为 `charge_at_lambda0_e`（charge-transfer
    路线下是配体净电荷的那一份），配体内部按物理电荷冻结成显式 exception，
    配体/co-ion 与纯环境之间若存在跨组 exception 则把 chargeProd 清零。
    """
    lam = float(lam)
    if lam not in (0.0, 1.0):
        raise ValueError(f"charging 端点参照只定义在 λ∈{{0,1}}，收到 λ={lam}")

    system = _clone_system(raw_system)
    nb = _find_nonbonded_force(system)
    ligand_set = sorted(int(i) for i in ligand_indices)
    ion_indices = _ion_indices_from_spec(spec)
    if set(ligand_set) & set(ion_indices):
        raise ValueError("ligand_indices 与 co-ion atom_index 重叠。")
    alchemical_set = set(ligand_set) | set(ion_indices)

    # co-ion 只支持单原子（§2.2）：任何 exception 都说明它不是单原子离子。
    for exc_idx in range(nb.getNumExceptions()):
        p1, p2, _cp, _s, _e = nb.getExceptionParameters(exc_idx)
        if int(p1) in ion_indices or int(p2) in ion_indices:
            raise RuntimeError(
                f"co-ion 粒子 {int(p1)}/{int(p2)} 之一带有 NonbondedForce "
                "exception——这个独立参照 builder 只实现了单原子 co-ion 的情形。"
            )

    physical_ligand_params: Dict[int, Tuple[float, float, float]] = {}
    for idx in ligand_set:
        q, sigma, epsilon = nb.getParticleParameters(idx)
        physical_ligand_params[idx] = (
            q.value_in_unit(unit.elementary_charge),
            sigma.value_in_unit(unit.nanometer),
            epsilon.value_in_unit(unit.kilojoule_per_mole),
        )

    if lam == 0.0:
        for idx in ligand_set:
            _q, sigma, epsilon = nb.getParticleParameters(idx)
            nb.setParticleParameters(idx, 0.0 * unit.elementary_charge, sigma, epsilon)

    charge_key = "charge_at_lambda1_e" if lam == 1.0 else "charge_at_lambda0_e"
    for ion in spec["ions"]:
        idx = int(ion["atom_index"])
        target_q = float(ion[charge_key])
        _q, sigma, epsilon = nb.getParticleParameters(idx)
        nb.setParticleParameters(idx, target_q * unit.elementary_charge, sigma, epsilon)

    if lam == 0.0:
        for exc_idx in range(nb.getNumExceptions()):
            p1, p2, charge_prod, sig, eps = nb.getExceptionParameters(exc_idx)
            p1, p2 = int(p1), int(p2)
            if (p1 in alchemical_set) ^ (p2 in alchemical_set):
                nb.setExceptionParameters(
                    exc_idx, p1, p2, 0.0 * unit.elementary_charge**2, sig, eps
                )

    # 配体内部对的冻结与 λ 无关——生产侧（`configure_charge_transfer_decharging`）
    # 在配置时就无条件把每一对配体原子都变成显式 exception，不管随后会不会把
    # λ 设成 1 还是 0。这里必须原样跟上：λ=1 的参照不是"什么都不用改的原始
    # System"，它仍然要带上这个结构性变化（哪怕数值上 PME 的 ordinary-pair
    # 处理与显式 exception 在这种紧凑的分子内对上几乎给出同一个能量，两者在
    # 结构上仍然是两个不同的力——前面 test_ligand_internal_pairs_are_identical_
    # between_production_and_reference 就是靠这个结构性差异钉出来的）。
    _freeze_ligand_internal_pairs(nb, ligand_set, physical_ligand_params)

    return system


def reference_vanishing_zero_system(
    raw_system: openmm.System,
    ligand_indices: Sequence[int],
    spec: Dict[str, Any],
) -> openmm.System:
    """D 的独立参照：charging λ=0 之上，再把配体–环境的 LJ 也删掉。

    "删掉"是结构性的（epsilon 设为 0），不是"代入 λ_vdw=0 算一遍软核公式再看
    是否接近 0"——D 的意义正在于这两件事必须给出同一个答案：如果只对了后者、
    没对上前者，说明生产端的软核表达式在 λ_vdw=0 处有残留（比如指数用错、
    符号用错），而不是真的把相互作用关掉了。

    配体内部 LJ 不受影响：它已经在 `reference_charging_endpoint_system` 里被
    冻结成带真实 epsilon 的显式 exception，这里只改主 NonbondedForce 里配体
    粒子自身的 epsilon，不碰那些 exception。
    """
    system = reference_charging_endpoint_system(raw_system, ligand_indices, spec, lam=0.0)
    nb = _find_nonbonded_force(system)
    ligand_set = set(int(i) for i in ligand_indices)

    for idx in sorted(ligand_set):
        q, sigma, _epsilon = nb.getParticleParameters(idx)
        nb.setParticleParameters(idx, q, sigma, 0.0 * unit.kilojoule_per_mole)

    for exc_idx in range(nb.getNumExceptions()):
        p1, p2, charge_prod, sig, eps = nb.getExceptionParameters(exc_idx)
        p1, p2 = int(p1), int(p2)
        if (p1 in ligand_set) ^ (p2 in ligand_set):
            nb.setExceptionParameters(exc_idx, p1, p2, charge_prod, sig, 0.0 * unit.kilojoule_per_mole)

    return system


# ---------------------------------------------------------------------------
# Production builders（A/B/C/D 的"被测"那一侧）—— 正常调用真实生产函数。
# ---------------------------------------------------------------------------


def production_charging_system(
    raw_system: openmm.System,
    ligand_indices: Sequence[int],
    topology: Any,
    spec: Dict[str, Any],
    *,
    lambda_name: str = "lam_coul",
    positions=None,
    box_vectors=None,
) -> openmm.System:
    """调用真实的 `ibs_engine.configure_pme_ligand_charge_offsets`（被测对象）。

    返回的 System 的 `lambda_name` global parameter 默认值是 **1.0**
    （`addGlobalParameter` 的既有约定），不是 0——评价 λ=0 端点前必须显式
    `context.setParameter(lambda_name, 0.0)`，不能依赖默认值。
    """
    system = _clone_system(raw_system)
    ie.configure_pme_ligand_charge_offsets(
        system,
        list(int(i) for i in ligand_indices),
        lambda_name=lambda_name,
        allow_charged_ligand=True,
        topology=topology,
        positions=positions,
        box_vectors=box_vectors,
        co_alchemical_ion_spec=spec,
    )
    return system


def production_vanishing_fixed_hamiltonian_systems(
    raw_system: openmm.System,
    ligand_indices: Sequence[int],
    lambdas_vdw: Sequence[float],
    box_vectors_nm: np.ndarray,
) -> List[openmm.System]:
    """构造固定物理 Hamiltonian `U_common + CV(λ_vdw=k)`（memtodolist.md §4）。

    `raw_system` 必须是**原始物理 System**，配体粒子在其中的电荷是**真实物理
    电荷**——不是 charging 配置完之后 base=0 的那份。

    2026-08-11 用真实 Stage1→Stage2 调用链核实过（`abfe_pipeline.py:3380`
    经 `IBSWindowManagerDualLambda._build_window_system`，一路追溯到
    `self.system`，从未被 `configure_pme_ligand_charge_offsets`/
    `configure_charge_transfer_decharging` 改写过），`build_ibs_dual_system`
    在真实生产里吃的从来是原始 System：Group 2（`create_ligand_internal_force`）
    直接用**这次调用一开始快照的 `all_params`** 重建配体内部 Coulomb
    （`138.935456*q1*q2/r`）——如果传入的是已经把配体 base 电荷置零的
    charging-configured System，Group 2 算出来的配体内部静电会被静默腰斩成 0，
    而不是保持"逐 λ 恒定的物理值"（`U_common` 的定义要求它必须是物理值）。
    这一版之前把 `charging_lambda0_system` 当参数喂进来是错的——已经被 C3-1
    的 seam 测试（`test_vanishing_lambda_one_seam_matches_charging_lambda_zero`）
    实测炸出来（0.71 kJ/mol 的差，远超 1e-5 相对容差），修复后改吃 `raw_system`。

    这也意味着：`build_ibs_dual_system` 自身的静态电中性防御
    （`abs(lig_net_charge) > 0.01` 就 raise）在**原始、未经 charging 配置**的
    电荷上核对——所以只有配体在原始拓扑里本来就净中性（可以有非零的单原子
    partial charge，只要求净和为 0）时，vanishing 阶段才能这样直接吃 `raw_system`。
    带净电的 charge-transfer 配体接入 vanishing 阶段，在当前生产代码里
    **尚未实现**（`abfe_config.json` 就地注明该路线是"Phase B3，尚未实现"）；
    C3 的 comparison C/D 因此只对净中性配体成立，不能拿本函数验证带电配体的
    charge-transfer + vanishing 组合——那个组合本身还不存在。

    不直接查询 `IBSBiasForce` 的总能量：Group 1 是混合态 bias，Group 4 还有
    WCA sampling shell，两者都不是物理 Hamiltonian 的一部分。这里用
    `ibs_engine._serialize_ibs_common_system` 拿到严格的 `U_common`
    （Group 0/2/3/5），再逐个 λ_vdw 状态把对应的单态软核 CV
    （`wrapper._int_cv_force_xmls[k]`）接到 Group 1 上，构成 C3 要求的
    "输出完整能量和力，不只查询 CV energy" 的固定态 System。

    返回的 System 列表与 `lambdas_vdw` 逐一对应；每个 System 内部只有一个
    Group 1 力（我们自己接的那个 CV），没有 IBS 混合 bias、没有 Group 4。
    """
    alchemical_params = core.ACESoftcorePotential.from_dict(
        core.ACESoftcorePotential.optimize_alpha(len(list(ligand_indices)))
    )
    n_states = len(lambdas_vdw)
    new_sys, wrapper = ie.build_ibs_dual_system(
        raw_system,
        topology=None,
        perturbed_indices=list(int(i) for i in ligand_indices),
        lambdas_coul=[0.0] * n_states,
        lambdas_vdw=[float(v) for v in lambdas_vdw],
        alchemical_params=alchemical_params,
        potential_type="softcore",
        box_vectors=np.asarray(box_vectors_nm, dtype=float) * unit.nanometer,
    )
    common_xml = ie._serialize_ibs_common_system(new_sys)
    int_cv_xmls = list(wrapper._int_cv_force_xmls)
    if len(int_cv_xmls) != n_states:
        raise RuntimeError(
            f"build_ibs_dual_system 返回了 {len(int_cv_xmls)} 个 per-state CV，"
            f"但请求了 {n_states} 个 λ_vdw 状态。"
        )

    systems = []
    for k in range(n_states):
        common = openmm.XmlSerializer.deserialize(common_xml)
        cv = openmm.XmlSerializer.deserialize(int_cv_xmls[k])
        cv.setForceGroup(1)
        common.addForce(cv)
        systems.append(common)
    return systems


# ---------------------------------------------------------------------------
# 求值纪律（memtodolist.md §9）：box → positions → virtual sites →
# 显式 global parameters → 零步取 energy/forces。
# ---------------------------------------------------------------------------


def evaluate_with_platform_info(
    system: openmm.System,
    positions_nm: np.ndarray,
    box_vectors_nm: np.ndarray,
    *,
    global_parameters: Optional[Dict[str, float]] = None,
    groups: Optional[Iterable[int]] = None,
    platform_name: str = "Reference",
    platform_properties: Optional[Dict[str, str]] = None,
) -> Tuple[float, np.ndarray, Dict[str, Any]]:
    """`evaluate()` 的完整版本：额外核验**实际 resolved** 的 platform 与
    property 值，不是只看"我请求了什么"。CUDA 的
    `Precision=double`/`DeterministicForces=true` 必须靠这个核验，请求了
    不代表真的生效（OpenMM 有的属性名打错/平台不支持会静默忽略）。
    """
    positions_nm = np.asarray(positions_nm, dtype=np.float64)
    box_vectors_nm = np.asarray(box_vectors_nm, dtype=np.float64)
    if not np.all(np.isfinite(positions_nm)):
        raise ValueError("positions 含非有限值（NaN/Inf），拒绝求值。")
    if not np.all(np.isfinite(box_vectors_nm)):
        raise ValueError("box vectors 含非有限值（NaN/Inf），拒绝求值。")

    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    platform = openmm.Platform.getPlatformByName(platform_name)
    properties = {str(k): str(v) for k, v in (platform_properties or {}).items()}
    context = (
        openmm.Context(system, integrator, platform, properties)
        if properties
        else openmm.Context(system, integrator, platform)
    )
    try:
        context.setPeriodicBoxVectors(*(box_vectors_nm * unit.nanometer))
        context.setPositions(positions_nm * unit.nanometer)
        has_virtual_site = any(
            system.isVirtualSite(i) for i in range(system.getNumParticles())
        )
        if has_virtual_site:
            context.computeVirtualSites()
        for name, value in (global_parameters or {}).items():
            context.setParameter(name, float(value))
        kwargs: Dict[str, Any] = dict(getEnergy=True, getForces=True)
        if groups is not None:
            kwargs["groups"] = set(int(g) for g in groups)
        state = context.getState(**kwargs)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        forces = np.asarray(
            state.getForces().value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        )

        resolved_platform = context.getPlatform()
        resolved_properties: Dict[str, str] = {}
        for key in properties:
            try:
                resolved_properties[key] = resolved_platform.getPropertyValue(context, key)
            except Exception as exc:  # noqa: BLE001 -- 报出来，不吞掉
                resolved_properties[key] = f"<无法读取: {exc}>"
        platform_info = {
            "platform_requested": platform_name,
            "platform_resolved": resolved_platform.getName(),
            "properties_requested": dict(properties),
            "properties_resolved": resolved_properties,
        }
    finally:
        del context, integrator
    return float(energy), forces, platform_info


def evaluate(
    system: openmm.System,
    positions_nm: np.ndarray,
    box_vectors_nm: np.ndarray,
    *,
    global_parameters: Optional[Dict[str, float]] = None,
    groups: Optional[Iterable[int]] = None,
    platform_name: str = "Reference",
    platform_properties: Optional[Dict[str, str]] = None,
) -> Tuple[float, np.ndarray]:
    """薄封装：只要 energy/forces，不要 platform 核验信息，用这个。"""
    energy, forces, _platform_info = evaluate_with_platform_info(
        system, positions_nm, box_vectors_nm,
        global_parameters=global_parameters, groups=groups,
        platform_name=platform_name, platform_properties=platform_properties,
    )
    return energy, forces


# ---------------------------------------------------------------------------
# 比较与报告
# ---------------------------------------------------------------------------


def energy_relative_difference(e_production: float, e_reference: float) -> float:
    return abs(e_production - e_reference) / max(1.0, abs(e_reference))


_VALID_FORCE_GATE_MODES = ("hard", "diagnostic")


def _compare_energy_and_forces(
    label: str,
    e_prod: float,
    f_prod: np.ndarray,
    e_ref: float,
    f_ref: np.ndarray,
    *,
    energy_rel_tol: float,
    force_abs_tol: float,
    force_gate_mode: str = "hard",
) -> Dict[str, Any]:
    """纯函数版本的比较逻辑，不碰 OpenMM——独立于 `evaluate()`，方便直接用
    手造的数组单测 fail-closed 分支（`evaluate()`/OpenMM 自己的
    `setPositions()` 会在粒子数不一致时先炸，用真实 System 测不到这一段）。

    `force_gate_mode`（Protocol v2，2026-08-11）：
    - `"hard"`（默认，v1 行为不变）：energy 与 force 都参与 `passed`。
    - `"diagnostic"`：力差仍然全程计算、记录（`force_within_tolerance` 字段
      如实反映有没有过 `force_abs_tol`），但**不计入 `passed`**——只有
      energy 和"力必须是有限值"两条决定 `passed`。用于 CUDA mixed
      precision 下"production vs 独立 reference"这一条比较：已经用
      Reference 平台证明两侧是同一个 Hamiltonian，CUDA mixed 下的力差是
      平台数值路径的产物（见模块 docstring），不代表构造错误。

    无论哪种模式，非有限力（NaN/Inf）都直接 fail——这条不受 `force_gate_mode`
    影响，任何情况下都是硬门。
    """
    if force_gate_mode not in _VALID_FORCE_GATE_MODES:
        raise ValueError(
            f"force_gate_mode={force_gate_mode!r} 不合法，只接受 {_VALID_FORCE_GATE_MODES}。"
        )

    f_prod = np.asarray(f_prod)
    f_ref = np.asarray(f_ref)
    if f_prod.shape != f_ref.shape:
        raise RuntimeError(
            f"[{label}] production 力数组形状 {f_prod.shape} 与 reference "
            f"{f_ref.shape} 不一致——两个 System 的粒子数/顺序必须完全相同。"
        )

    forces_finite = bool(np.all(np.isfinite(f_prod)) and np.all(np.isfinite(f_ref)))

    abs_diff = float(np.max(np.abs(e_prod - e_ref)))
    rel_diff = energy_relative_difference(e_prod, e_ref)
    force_diff = np.abs(f_prod - f_ref)
    max_force_diff = float(np.max(force_diff)) if force_diff.size else 0.0
    if force_diff.size:
        flat_idx = int(np.argmax(force_diff))
        worst_atom, worst_component = divmod(flat_idx, 3)
    else:
        worst_atom, worst_component = -1, -1

    energy_ok = rel_diff <= energy_rel_tol
    force_within_tolerance = max_force_diff <= force_abs_tol
    if force_gate_mode == "hard":
        passed = energy_ok and force_within_tolerance and forces_finite
    else:  # "diagnostic"
        passed = energy_ok and forces_finite
    return {
        "label": str(label),
        "e_production_kj_mol": e_prod,
        "e_reference_kj_mol": e_ref,
        "abs_delta_e_kj_mol": abs_diff,
        "rel_delta_e": rel_diff,
        "max_abs_force_component_diff_kj_mol_nm": max_force_diff,
        "worst_atom_index": worst_atom,
        "worst_component": ["x", "y", "z"][worst_component] if worst_component >= 0 else None,
        "worst_atom_force_production_kj_mol_nm": (
            f_prod[worst_atom].tolist() if worst_atom >= 0 else None
        ),
        "worst_atom_force_reference_kj_mol_nm": (
            f_ref[worst_atom].tolist() if worst_atom >= 0 else None
        ),
        "energy_rel_tol": energy_rel_tol,
        "force_abs_tol": force_abs_tol,
        "force_gate_mode": force_gate_mode,
        "forces_finite": forces_finite,
        "force_within_tolerance": bool(force_within_tolerance),
        "passed": bool(passed),
    }


def _platform_info_matches_request(platform_info: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """请求了某个 platform/property 不代表真的生效——OpenMM 对拼错的属性名
    或者当前平台不支持的属性会静默忽略。这里逐项核对 resolved 值，返回
    `(是否完全匹配, 不匹配原因列表)`；不匹配就该让上层判定失败，不能只是
    记在报告里没人看。
    """
    reasons: List[str] = []
    if platform_info["platform_resolved"] != platform_info["platform_requested"]:
        reasons.append(
            f"platform_resolved={platform_info['platform_resolved']!r} != "
            f"platform_requested={platform_info['platform_requested']!r}"
        )
    for key, requested_value in platform_info["properties_requested"].items():
        resolved_value = platform_info["properties_resolved"].get(key)
        if str(resolved_value).strip().lower() != str(requested_value).strip().lower():
            reasons.append(
                f"property {key!r} requested={requested_value!r} but resolved={resolved_value!r}"
            )
    return (len(reasons) == 0, reasons)


def compare_endpoint(
    label: str,
    production_system: openmm.System,
    reference_system: openmm.System,
    positions_nm: np.ndarray,
    box_vectors_nm: np.ndarray,
    *,
    production_globals: Optional[Dict[str, float]] = None,
    reference_globals: Optional[Dict[str, float]] = None,
    production_groups: Optional[Iterable[int]] = None,
    reference_groups: Optional[Iterable[int]] = None,
    platform_name: str = "Reference",
    platform_properties: Optional[Dict[str, str]] = None,
    energy_rel_tol: Optional[float] = None,
    force_abs_tol: Optional[float] = None,
    force_gate_mode: str = "hard",
) -> Dict[str, Any]:
    """求值 + 比较 + 打包成一份可落盘的 report，不 raise——由调用方按
    `passed` 判定并决定要不要 assert（这样一次调用既能喂 pytest，也能喂
    CLI 报告）。`platform_properties`（如 CUDA 的
    `{"Precision": "double", "DeterministicForces": "true"}`）两侧共用同一份；
    report 里附带**实际 resolved** 的 platform/property 值，不是只记请求值。

    resolved platform/property 与请求值不一致会让 `passed` 直接变 `False`
    （`platform_verified=False`，`platform_mismatch_reasons` 列出具体哪项
    不对）——2026-08-11 用户审阅指出：只记录 resolved 值但不拿它判定结果，
    等于权威门形同虚设（比如请求 CUDA 双精度，实际却悄悄跑成了 mixed，
    report 里能查到但不会让这次比较失败）。

    `force_gate_mode`（Protocol v2，2026-08-11，见模块 docstring）：默认
    `"hard"`（v1 行为，力差参与 `passed`）；`"diagnostic"` 时力差仍计算/
    落盘但不参与 `passed`——只用于 CUDA mixed precision 下"production vs
    独立 reference"这一条已经被证明与真实构造无关的力差。
    """
    energy_rel_tol = ENERGY_RELATIVE_TOLERANCE if energy_rel_tol is None else energy_rel_tol
    force_abs_tol = FORCE_ABS_TOLERANCE_KJ_MOL_NM if force_abs_tol is None else force_abs_tol

    e_prod, f_prod, prod_platform_info = evaluate_with_platform_info(
        production_system,
        positions_nm,
        box_vectors_nm,
        global_parameters=production_globals,
        groups=production_groups,
        platform_name=platform_name,
        platform_properties=platform_properties,
    )
    e_ref, f_ref, ref_platform_info = evaluate_with_platform_info(
        reference_system,
        positions_nm,
        box_vectors_nm,
        global_parameters=reference_globals,
        groups=reference_groups,
        platform_name=platform_name,
        platform_properties=platform_properties,
    )
    result = _compare_energy_and_forces(
        label, e_prod, f_prod, e_ref, f_ref,
        energy_rel_tol=energy_rel_tol, force_abs_tol=force_abs_tol,
        force_gate_mode=force_gate_mode,
    )
    result["production_platform_info"] = prod_platform_info
    result["reference_platform_info"] = ref_platform_info

    prod_ok, prod_reasons = _platform_info_matches_request(prod_platform_info)
    ref_ok, ref_reasons = _platform_info_matches_request(ref_platform_info)
    result["platform_verified"] = bool(prod_ok and ref_ok)
    result["platform_mismatch_reasons"] = [f"production: {r}" for r in prod_reasons] + [
        f"reference: {r}" for r in ref_reasons
    ]
    result["passed"] = bool(result["passed"] and result["platform_verified"])
    return result


def compare_vanishing_zero_endpoint(
    production_system: openmm.System,
    positions_nm: np.ndarray,
    box_vectors_nm: np.ndarray,
    *,
    ligand_environment_groups: Iterable[int],
    platform_name: str = "Reference",
    platform_properties: Optional[Dict[str, str]] = None,
    abs_tol_kj_mol: Optional[float] = None,
    force_abs_tol: Optional[float] = None,
) -> Dict[str, Any]:
    """D 专用的额外硬门：单独隔离出配体–环境分量，要求严格零
    （`|E|≤1e-6 kJ/mol`），不是"很小"（memtodolist.md §8）。

    没有 `reference_system` 参数：D 的参照在这个隔离出来的 force group 里就是
    解析意义上严格的 0（结构性删除，不是"数值上很小"），不需要另建一个 System
    去求值再减法——直接检验这一个 group 的能量/力是否满足严格零阈值即可。

    `ligand_environment_groups` 应该只包含"纯配体–环境软核 CV"所在的那一个
    force group（生产侧接上去的单态 CV，见
    `production_vanishing_fixed_hamiltonian_systems`）——这个 group 里没有
    别的力，所以查询它就是直接拿到配体–环境相互作用能量，不需要再做减法。

    `platform_properties`（Protocol v2 C/D，2026-08-11）：D 的严格零门在
    CUDA mixed precision 下**仍然是硬门**，不因为"这是 CUDA mixed"而放宽——
    它是代数结构性的零（λ_vdw≡0 直接让软核 CV 系数为零），不依赖数值精度，
    跟 A/B/C 那条"两种等价 exception 排列在 mixed precision 下力差只能诊断"
    完全是两件事。因此这里同样核验 **resolved** platform/property 是否真的
    等于请求值（不核验就等于门形同虚设，见 `compare_endpoint` 同一处理由）。
    """
    abs_tol_kj_mol = DECOUPLED_ABS_TOLERANCE_KJ_MOL if abs_tol_kj_mol is None else abs_tol_kj_mol
    force_abs_tol = FORCE_ABS_TOLERANCE_KJ_MOL_NM if force_abs_tol is None else force_abs_tol

    e_lig_env, f_lig_env, platform_info = evaluate_with_platform_info(
        production_system,
        positions_nm,
        box_vectors_nm,
        groups=ligand_environment_groups,
        platform_name=platform_name,
        platform_properties=platform_properties,
    )
    max_force = float(np.max(np.abs(f_lig_env))) if f_lig_env.size else 0.0
    platform_ok, platform_reasons = _platform_info_matches_request(platform_info)
    passed = abs(e_lig_env) <= abs_tol_kj_mol and max_force <= force_abs_tol and platform_ok
    return {
        "label": "D_ligand_environment_strict_zero",
        "e_ligand_environment_kj_mol": e_lig_env,
        "max_abs_force_ligand_environment_kj_mol_nm": max_force,
        "abs_tol_kj_mol": abs_tol_kj_mol,
        "force_abs_tol": force_abs_tol,
        "platform_info": platform_info,
        "platform_verified": bool(platform_ok),
        "platform_mismatch_reasons": platform_reasons,
        "passed": bool(passed),
    }


# ---------------------------------------------------------------------------
# 静态 fail-closed 检查（memtodolist.md §10）。
# ---------------------------------------------------------------------------


def assert_system_not_alchemically_configured(system: openmm.System, *, context: str) -> None:
    """raw system.xml 不应该已经带 charging offsets——防止误把
    `system_prepared.xml` 当成 raw 输入喂进来。
    """
    nb = _find_nonbonded_force(system)
    if nb.getNumParticleParameterOffsets() > 0 or nb.getNumExceptionParameterOffsets() > 0:
        raise RuntimeError(
            f"[{context}] 输入 System 已经带有 ParameterOffset——这应该是 raw "
            "system.xml，但看起来是已配置过 charging 的 system_prepared.xml。"
            "禁止把 system_prepared.xml 再传给 charging configure 或当成 raw 输入。"
        )


def assert_protocol_version(manifest: Dict[str, Any], *, expected: int, path: str) -> None:
    got = manifest.get("protocol_version")
    if int(got) != int(expected):
        raise RuntimeError(
            f"{path} 的 protocol_version={got!r}，C3 期望 {expected!r}。"
            "版本不符：可能是尚未按最新协议重建的旧产物，拒绝继续。"
        )


def assert_lambda_is_exact_endpoint(lam: float, *, context: str) -> None:
    lam = float(lam)
    if lam not in (0.0, 1.0):
        raise RuntimeError(f"[{context}] λ={lam} 不是精确的 0 或 1，C3 只比较端点。")


def assert_finite(array: np.ndarray, *, context: str) -> None:
    if not np.all(np.isfinite(np.asarray(array, dtype=np.float64))):
        raise RuntimeError(f"[{context}] 数组包含非有限值（NaN/Inf）。")


# ---------------------------------------------------------------------------
# MEM-00h 双边归一化（Protocol v2 的 C/D 应用，2026-08-11，用户明确指正后的
# 修法——不是我最初提的三个选项里的任何一个）。
#
# 根因：C2 raw System 自己的 `NonbondedForce` 带一个 C2 专用的
# `[0.995,1.0]nm` LJ switch（`validate_charge_transfer_lipid_slab.py`
# v2→v3，2026-08-07，为修 `MonteCarloMembraneBarostat` 造成的膜面内人工
# 压缩问题特意加的，当时代码注释已明确声明"范围只到 C2 自己的 System 构建，
# 不改全局 MEM-00h 常量"）；vanishing 阶段的配体–环境软核 CV 则一直遵循全局
# MEM-00h 的无 switch 约定（`ibs_engine.SOFTCORE_CUTOFF_NM`/
# `use_switching=False`）。C3 的端点恒等式测的是"同一个 MEM-00h Hamiltonian
# 在不同构造路径下是否给出同一个答案"——C2 采样脚本的轨迹只提供构象和周期盒，
# 不代表 C3 求值时也要继续背着 C2 自己的采样期 switch 惯例。
#
# 修法（用户指定，不是"三个候选选项"里任何一个）：C3 的 loader 在读到 raw
# System 之后，先建立一份**求值专用**的归一化 clone（`mem00h_normalized_
# raw_system`），把它的 `NonbondedForce` 强制转到 MEM-00h 的
# `cutoff=1.0nm, switching=False`；之后 A（charging λ=1）/B（charging λ=0）/
# C（baked charging λ=0 + vanishing λ_vdw=1）/D（vanishing λ_vdw=0 + 独立
# reference）**全部从这同一份归一化 clone 分别构造**，不再各自继承 raw
# System 原本的、彼此不一致的 switch 设置。**不改 C2 已有的 raw 文件/轨迹
# 本身**——归一化只发生在 C3 评估工具内部的一份内存 clone 上，C2 生产采样
# 用的仍然是它自己的原始 switch 惯例，两者互不影响。
#
# `assert_mem00h_switching_convention` 是配套的 fail-closed 结构核验：每次
# 构造完 production/reference 侧的 System 之后都核对它们的 `NonbondedForce`
# 确实是 cutoff=1.0nm/switching=False——防止某个下游构造函数（比如
# `build_ibs_dual_system` 内部）悄悄从别的地方重新引入一个不同的 switch
# 惯例而没被发现。
# ---------------------------------------------------------------------------


def mem00h_normalized_raw_system(raw_system: openmm.System) -> openmm.System:
    """把 raw System 的 `NonbondedForce` 强制归一化到 MEM-00h 的
    `cutoff=1.0nm, switching=False`，用于 C3 v2 的评估（不改传入的原始
    System，也不改磁盘上的任何 raw 文件）。

    cutoff 只做**核验**、不强制改写——如果 raw System 的 cutoff 不是
    `MEM00H_CUTOFF_NM`，说明构造这份输入时用了完全不同的协议，直接 fail
    closed，而不是悄悄拿一个跟原意图不同的 cutoff 继续跑。switching 则是
    **强制**关掉——这正是本次归一化要修的那条不一致（某些 case，比如 C2，
    raw System 自带一个局部 switch；某些 case，比如 C1，raw System 本来就
    没有 switch，这里是 no-op）。dispersion correction 原样保留，不属于
    这次要修的范围。
    """
    system = _clone_system(raw_system)
    found_any = False
    for force in system.getForces():
        if not isinstance(force, openmm.NonbondedForce):
            continue
        found_any = True
        cutoff_nm = force.getCutoffDistance().value_in_unit(unit.nanometer)
        if abs(cutoff_nm - MEM00H_CUTOFF_NM) > 1e-9:
            raise RuntimeError(
                f"mem00h_normalized_raw_system: NonbondedForce cutoff={cutoff_nm}nm，"
                f"与 MEM00H_CUTOFF_NM={MEM00H_CUTOFF_NM}nm 不符，拒绝继续"
                "（这不是一次'顺手改成 1.0nm'——cutoff 不符说明输入协议本身就不对，"
                "应该先查清楚，不是被这个归一化函数悄悄掩盖）。"
            )
        force.setUseSwitchingFunction(bool(MEM00H_SWITCHING_ENABLED))
    if not found_any:
        raise RuntimeError("mem00h_normalized_raw_system: System 中没有 NonbondedForce。")
    return system


def assert_mem00h_switching_convention(system: openmm.System, *, context: str) -> None:
    """核验 System 里每个 `NonbondedForce` 确实是 MEM-00h 的
    `cutoff=1.0nm, switching=False`——归一化之后的每一次构造都应该核对，
    不能只信"我调用过 `mem00h_normalized_raw_system`"，要核实它真的传导到了
    最终喂给 Context 的这个 System 上（下游构造函数完全可能另外 clone 一份
    带着自己的设置）。
    """
    for force in system.getForces():
        if not isinstance(force, openmm.NonbondedForce):
            continue
        cutoff_nm = force.getCutoffDistance().value_in_unit(unit.nanometer)
        if abs(cutoff_nm - MEM00H_CUTOFF_NM) > 1e-9:
            raise RuntimeError(
                f"[{context}] NonbondedForce cutoff={cutoff_nm}nm，"
                f"应为 MEM00H_CUTOFF_NM={MEM00H_CUTOFF_NM}nm。"
            )
        if bool(force.getUseSwitchingFunction()) != bool(MEM00H_SWITCHING_ENABLED):
            raise RuntimeError(
                f"[{context}] NonbondedForce.getUseSwitchingFunction()="
                f"{force.getUseSwitchingFunction()}，应为 {MEM00H_SWITCHING_ENABLED}"
                "（MEM-00h 约定）——归一化没有正确传导到这个 System。"
            )


# ---------------------------------------------------------------------------
# C3-2：真实 C1/C2 raw 输入读取 + 单帧 wiring smoke。
#
# 只做"逐 force-group 账目对不对、生产/参照两侧能不能求出同一个数"这件事，
# 不是 C3-3/C3-4 的权威数值门——那要求 CUDA + double precision + 完整
# 100 帧矩阵（memtodolist.md §9），留给用户在计算节点上跑。这里默认用 CPU
# 平台，能覆盖"两个 System 的构造是不是接对了"，覆盖不了"GPU mixed
# precision 下数值门还过不过"。
# ---------------------------------------------------------------------------


def load_case_raw_inputs(case_dir) -> Dict[str, Any]:
    """只读 raw `system.xml`/`topology.cif`/`ligand_indices.json`/
    `coalchemical_ion_spec.json`/`build_manifest.json`——不读
    `system_prepared.xml`（那是已经配置过 charging 的产物，见
    `assert_system_not_alchemically_configured`）。
    """
    case_dir = Path(case_dir)
    with open(case_dir / "system.xml", "r", encoding="utf-8") as fh:
        system = openmm.XmlSerializer.deserialize(fh.read())
    topology = app.PDBxFile(str(case_dir / "topology.cif")).topology
    with open(case_dir / "ligand_indices.json", "r", encoding="utf-8") as fh:
        ligand_indices = json.load(fh)["ligand_indices"]
    with open(case_dir / "coalchemical_ion_spec.json", "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    with open(case_dir / "build_manifest.json", "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    assert_system_not_alchemically_configured(system, context=str(case_dir / "system.xml"))
    expected_fp = manifest.get("coalchemical_ion_fingerprint")
    if expected_fp is not None and str(spec.get("fingerprint")) != str(expected_fp):
        raise RuntimeError(
            f"{case_dir}/coalchemical_ion_spec.json 的 fingerprint 与 "
            f"build_manifest.json 记录的 {expected_fp!r} 不一致。"
        )
    return {
        "system": system,
        "topology": topology,
        "ligand_indices": ligand_indices,
        "spec": spec,
        "manifest": manifest,
        "case_dir": case_dir,
    }


def read_dcd_frame(
    dcd_path,
    topology: Any,
    frame_index: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """读一帧的坐标 + **这一帧自己的**完整周期盒矢量（不是初始
    `box_vectors_nm.npy`），不做 `image_molecules`（memtodolist.md §9 禁止在
    求值前修改坐标）。返回 `(positions_nm, box_vectors_nm)`，均是裸 ndarray。
    """
    import mdtraj as md

    md_top = md.Topology.from_openmm(topology)
    traj = md.load(str(dcd_path), top=md_top)
    if frame_index >= traj.n_frames:
        raise RuntimeError(
            f"{dcd_path} 只有 {traj.n_frames} 帧，请求的 frame_index={frame_index} 越界。"
        )
    if traj.unitcell_vectors is None:
        raise RuntimeError(f"{dcd_path} 没有记录周期盒矢量，拒绝用初始盒代替。")
    positions_nm = np.asarray(traj.xyz[frame_index], dtype=np.float64)
    box_nm = np.asarray(traj.unitcell_vectors[frame_index], dtype=np.float64)
    assert_finite(positions_nm, context=f"{dcd_path}[{frame_index}] positions")
    assert_finite(box_nm, context=f"{dcd_path}[{frame_index}] box")
    return positions_nm, box_nm


def wiring_smoke_report(
    case_dir,
    *,
    lambda1_dcd_name: str,
    lambda0_dcd_name: str,
    frame_index: int = 10,
    platform_name: str = "CPU",
    dynamics_dir: Optional[Any] = None,
) -> Dict[str, Any]:
    """C3-2：对 A/B 两个 charging 端点各取一帧，打印逐 force-group 账目并跑
    完整的 production vs reference 比较。只验证"接线对不对"，不是权威数值门。

    `dynamics_dir` 用于 C2：raw 输入在 `validation/c2_lipid_slab_v11/<case>/`，
    真实轨迹在**另一个目录** `validation/c2_lipid_slab_v11_full11/<case>/dynamics/`
    （`c2_lipid_slab_v11/<case>/dynamics/` 只是占位，是空的——memtodolist.md §6）。
    不传时默认 `case_dir/dynamics`（C1 的布局）。
    """
    inputs = load_case_raw_inputs(case_dir)
    system = inputs["system"]
    topology = inputs["topology"]
    ligand_indices = inputs["ligand_indices"]
    spec = inputs["spec"]
    case_dir = inputs["case_dir"]
    dynamics_dir = Path(case_dir) / "dynamics" if dynamics_dir is None else Path(dynamics_dir)

    endpoints = []
    for lam, dcd_name in ((1.0, lambda1_dcd_name), (0.0, lambda0_dcd_name)):
        positions_nm, box_nm = read_dcd_frame(
            dynamics_dir / dcd_name, topology, frame_index
        )
        production = production_charging_system(system, ligand_indices, topology, spec)
        reference = reference_charging_endpoint_system(system, ligand_indices, spec, lam=lam)

        comparison = compare_endpoint(
            f"lambda_coul={lam}",
            production,
            reference,
            positions_nm,
            box_nm,
            production_globals={"lam_coul": lam},
            production_groups={0},
            reference_groups={0},
            platform_name=platform_name,
        )

        group_energies = {}
        max_group = max(
            (
                production.getForce(i).getForceGroup()
                for i in range(production.getNumForces())
            ),
            default=0,
        )
        for group in range(max_group + 1):
            e_group, _f_group = evaluate(
                production, positions_nm, box_nm,
                global_parameters={"lam_coul": lam}, groups={group},
                platform_name=platform_name,
            )
            group_energies[str(group)] = e_group

        endpoints.append(
            {
                "lambda_coul": lam,
                "dcd": dcd_name,
                "frame_index": frame_index,
                "comparison": comparison,
                "production_force_group_energies_kj_mol": group_energies,
            }
        )

    return {
        "case_dir": str(case_dir),
        "platform_name": platform_name,
        "protocol_version": PROTOCOL_VERSION,
        "endpoints": endpoints,
        "passed": all(e["comparison"]["passed"] for e in endpoints),
    }


# ---------------------------------------------------------------------------
# 权威多帧 runner：固定帧索引表（memtodolist.md §5），跑 A/B 全矩阵。
#
# C（seam）/D（严格零）目前没有接入真实 C1/C2 数据——2026-08-11 实测确认
# vanishing 阶段的 charging→vanishing handoff 对带电 charge-transfer 配体尚未
# 实现（见 memtodolist.md C3-1 小节），这里先只跑 A/B；C/D 仍只在
# `tests/test_charge_transfer_real_endpoints.py` 的合成中性配体上做契约测试。
# ---------------------------------------------------------------------------

C1_LAMBDA1_FRAME_INDICES: Tuple[int, ...] = (10, 20, 30, 40, 50, 60, 70, 80, 90, 99)
C1_LAMBDA0_FRAME_INDICES: Tuple[int, ...] = (10, 20, 30, 40, 50, 60, 70, 80, 90, 99)
C2_LAMBDA1_FRAME_INDICES: Tuple[int, ...] = (10, 20, 30, 40, 50, 60, 70, 80, 90, 99)
C2_LAMBDA0_FRAME_INDICES: Tuple[int, ...] = (20, 40, 60, 80, 100, 120, 140, 160, 180, 199)


def run_matrix(
    case_dir,
    *,
    lambda1_dcd_name: str,
    lambda0_dcd_name: str,
    lambda1_frame_indices: Sequence[int],
    lambda0_frame_indices: Sequence[int],
    dynamics_dir=None,
    platform_name: str = "CUDA",
    platform_properties: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """A/B 端点在固定帧索引表上的权威判定——不是单帧 smoke。

    每一帧独立判定、独立记录；`passed` = 所有帧都过，`failed_frames` 显式列出
    没过的帧，不用平均值掩盖任何一帧的失败（memtodolist.md §8/§9）。生产/参照
    两个 System 只构造一次（不依赖帧），逐帧只换 positions/box——这与真实
    Stage2 的"System 结构固定、只有坐标随采样变化"一致，也避免重复构造的
    开销随帧数线性放大。
    """
    inputs = load_case_raw_inputs(case_dir)
    system = inputs["system"]
    topology = inputs["topology"]
    ligand_indices = inputs["ligand_indices"]
    spec = inputs["spec"]
    case_dir = inputs["case_dir"]
    dynamics_dir = Path(case_dir) / "dynamics" if dynamics_dir is None else Path(dynamics_dir)

    production = production_charging_system(system, ligand_indices, topology, spec)
    reference_by_lambda = {
        1.0: reference_charging_endpoint_system(system, ligand_indices, spec, lam=1.0),
        0.0: reference_charging_endpoint_system(system, ligand_indices, spec, lam=0.0),
    }

    frames: List[Dict[str, Any]] = []
    for lam, dcd_name, frame_indices in (
        (1.0, lambda1_dcd_name, lambda1_frame_indices),
        (0.0, lambda0_dcd_name, lambda0_frame_indices),
    ):
        dcd_path = dynamics_dir / dcd_name
        reference = reference_by_lambda[lam]
        for frame_index in frame_indices:
            positions_nm, box_nm = read_dcd_frame(dcd_path, topology, int(frame_index))
            comparison = compare_endpoint(
                f"lambda_coul={lam}_frame{frame_index}",
                production, reference, positions_nm, box_nm,
                production_globals={"lam_coul": lam},
                production_groups={0},
                reference_groups={0},
                platform_name=platform_name,
                platform_properties=platform_properties,
            )
            frames.append(
                {
                    "lambda_coul": lam,
                    "dcd": dcd_name,
                    "frame_index": int(frame_index),
                    "comparison": comparison,
                }
            )

    failed_frames = [
        {"lambda_coul": f["lambda_coul"], "dcd": f["dcd"], "frame_index": f["frame_index"]}
        for f in frames
        if not f["comparison"]["passed"]
    ]
    return {
        "case_dir": str(case_dir),
        "platform_name": platform_name,
        "platform_properties": dict(platform_properties or {}),
        "protocol_version": PROTOCOL_VERSION,
        "n_frames": len(frames),
        "n_failed": len(failed_frames),
        "failed_frames": failed_frames,
        "frames": frames,
        "passed": len(failed_frames) == 0,
    }


# ---------------------------------------------------------------------------
# Protocol v2（2026-08-11，见模块 docstring）：三层门的权威 runner。
#
#   gate1_reference_identity        production(显式设参数) vs 独立reference，
#                                    Reference 平台，energy+force 都硬门。
#   gate2_mixed_live_vs_baked       production(显式设参数) vs
#                                    bake(production)，CUDA 平台（默认 mixed，
#                                    与生产一致），energy+force 都硬门——已
#                                    用归因诊断证明这条本来就干净，这里是
#                                    结构性地把它钉进权威 runner，不是新增
#                                    风险。
#   gate3_mixed_production_vs_reference   production(显式设参数) vs 独立
#                                    reference，CUDA 平台，energy 硬门，
#                                    force 只诊断（`force_gate_mode=
#                                    "diagnostic"`），力仍须有限。
#
# 每帧独立判定：三个 gate 都 passed 才算这一帧过；`failed_frames` 记录哪一帧
# 卡在哪一个 gate，不用总数掩盖。
# ---------------------------------------------------------------------------


def run_protocol_v2_matrix(
    case_dir,
    *,
    lambda1_dcd_name: str,
    lambda0_dcd_name: str,
    lambda1_frame_indices: Sequence[int],
    lambda0_frame_indices: Sequence[int],
    dynamics_dir=None,
    lambda_name: str = "lam_coul",
    mixed_platform_name: str = "CUDA",
    mixed_platform_properties: Optional[Dict[str, str]] = None,
    reference_platform_name: str = "Reference",
) -> Dict[str, Any]:
    """Protocol v2 的权威 runner——A/B 端点，固定帧索引表，三层门。"""
    mixed_platform_properties = (
        {"Precision": "mixed"} if mixed_platform_properties is None else mixed_platform_properties
    )

    inputs = load_case_raw_inputs(case_dir)
    system = mem00h_normalized_raw_system(inputs["system"])
    topology = inputs["topology"]
    ligand_indices = inputs["ligand_indices"]
    spec = inputs["spec"]
    case_dir = inputs["case_dir"]
    dynamics_dir = Path(case_dir) / "dynamics" if dynamics_dir is None else Path(dynamics_dir)

    production = production_charging_system(
        system, ligand_indices, topology, spec, lambda_name=lambda_name
    )
    reference_by_lambda = {
        1.0: reference_charging_endpoint_system(system, ligand_indices, spec, lam=1.0),
        0.0: reference_charging_endpoint_system(system, ligand_indices, spec, lam=0.0),
    }
    baked_by_lambda = {
        lam: core.bake_global_parameter_into_fixed_nonbonded_force(production, lambda_name, lam)
        for lam in (1.0, 0.0)
    }
    assert_mem00h_switching_convention(production, context="run_protocol_v2_matrix:production")
    for lam, ref_sys in reference_by_lambda.items():
        assert_mem00h_switching_convention(
            ref_sys, context=f"run_protocol_v2_matrix:reference(lambda={lam})"
        )
    for lam, baked_sys in baked_by_lambda.items():
        assert_mem00h_switching_convention(
            baked_sys, context=f"run_protocol_v2_matrix:baked(lambda={lam})"
        )

    frames: List[Dict[str, Any]] = []
    for lam, dcd_name, frame_indices in (
        (1.0, lambda1_dcd_name, lambda1_frame_indices),
        (0.0, lambda0_dcd_name, lambda0_frame_indices),
    ):
        dcd_path = dynamics_dir / dcd_name
        reference = reference_by_lambda[lam]
        baked = baked_by_lambda[lam]
        for frame_index in frame_indices:
            positions_nm, box_nm = read_dcd_frame(dcd_path, topology, int(frame_index))
            production_globals = {lambda_name: lam}

            gate1 = compare_endpoint(
                f"gate1_reference_identity_lambda={lam}_frame{frame_index}",
                production, reference, positions_nm, box_nm,
                production_globals=production_globals,
                production_groups={0}, reference_groups={0},
                platform_name=reference_platform_name,
                force_gate_mode="hard",
            )
            gate2 = compare_endpoint(
                f"gate2_mixed_live_vs_baked_lambda={lam}_frame{frame_index}",
                production, baked, positions_nm, box_nm,
                production_globals=production_globals,
                production_groups={0}, reference_groups={0},
                platform_name=mixed_platform_name,
                platform_properties=mixed_platform_properties,
                force_gate_mode="hard",
            )
            gate3 = compare_endpoint(
                f"gate3_mixed_production_vs_reference_lambda={lam}_frame{frame_index}",
                production, reference, positions_nm, box_nm,
                production_globals=production_globals,
                production_groups={0}, reference_groups={0},
                platform_name=mixed_platform_name,
                platform_properties=mixed_platform_properties,
                force_gate_mode="diagnostic",
            )

            frame_passed = gate1["passed"] and gate2["passed"] and gate3["passed"]
            frames.append(
                {
                    "lambda_coul": lam,
                    "dcd": dcd_name,
                    "frame_index": int(frame_index),
                    "gate1_reference_identity": gate1,
                    "gate2_mixed_live_vs_baked": gate2,
                    "gate3_mixed_production_vs_reference": gate3,
                    "passed": bool(frame_passed),
                }
            )

    failed_frames = []
    for f in frames:
        if f["passed"]:
            continue
        failing_gates = [
            gname
            for gname in (
                "gate1_reference_identity",
                "gate2_mixed_live_vs_baked",
                "gate3_mixed_production_vs_reference",
            )
            if not f[gname]["passed"]
        ]
        failed_frames.append(
            {
                "lambda_coul": f["lambda_coul"], "dcd": f["dcd"], "frame_index": f["frame_index"],
                "failing_gates": failing_gates,
            }
        )

    return {
        "case_dir": str(case_dir),
        "protocol_version": PROTOCOL_VERSION,
        "lambda_name": lambda_name,
        "reference_platform_name": reference_platform_name,
        "mixed_platform_name": mixed_platform_name,
        "mixed_platform_properties": dict(mixed_platform_properties),
        "n_frames": len(frames),
        "n_failed": len(failed_frames),
        "failed_frames": failed_frames,
        "frames": frames,
        "passed": len(failed_frames) == 0,
    }


# ---------------------------------------------------------------------------
# Protocol v2 —— C/D 端点，复用真实 charging λ_coul=0 轨迹帧（2026-08-11，
# 用户明确指示：C/D 是同坐标下的 Hamiltonian 端点恒等式，不需要新跑
# vanishing MD；直接拿 charging λ=0 已经采样过的真实帧算）。
#
#   C（seam）      baked charging λ_coul=0（`charging0_baked`）
#                  vs production vanishing λ_vdw=1（`vanishing_one`）——
#                  两侧都是生产代码路径，测两阶段接缝自洽性，不是独立构造。
#   D（strict zero + vs 独立 reference）
#                  production vanishing λ_vdw=0（`vanishing_zero`）
#                  vs 独立构造的 `reference_vanishing_zero_system`；额外跑
#                  `compare_vanishing_zero_endpoint` 的严格零门。
#
# 两个恒等式在比较发生的时刻，两侧都**没有活的 GlobalParameter**——
# `production_vanishing_fixed_hamiltonian_systems` 把每个 λ_vdw 态的 CV
# 构造成固定表达式（不是运行时可设的 GlobalParameter），charging 侧在喂给
# vanishing 之前已经被 `bake_global_parameter_into_fixed_nonbonded_force`
# 结构性烘焙掉了。所以 gate2（"live ParameterOffset vs baked production"）
# 在这里没有对应的比较对象，标记 `applicable=False`，不参与 `passed`，
# 不制造一个无意义比较（用户 2026-08-11 明确指示）。
#
# D 的严格零门（`strict_zero_reference`/`strict_zero_mixed`）在 Reference
# 和 CUDA mixed 下都是硬门——它是代数结构性的零，不是"两种等价构造在 mixed
# precision 下的力差"，跟 gate3 的诊断化不是一回事。
# ---------------------------------------------------------------------------


def run_protocol_v2_matrix_cd(
    case_dir,
    *,
    lambda0_dcd_name: str,
    lambda0_frame_indices: Sequence[int],
    dynamics_dir=None,
    lambda_name: str = "lambda_coul",
    mixed_platform_name: str = "CUDA",
    mixed_platform_properties: Optional[Dict[str, str]] = None,
    reference_platform_name: str = "Reference",
) -> Dict[str, Any]:
    """Protocol v2 的权威 runner——C/D 端点，复用现有 charging λ_coul=0 真实帧。

    只对**净电荷非零、走 charge-transfer 路线**的配体有意义（`spec` 必须是
    带 co-ion 的 charge-transfer spec）——C1/C2 的单原子 Na+ 探针正是这种
    情形。System 只构造一次（与 A/B 的 `run_protocol_v2_matrix` 同一纪律：
    结构固定，只有坐标逐帧变化）。
    """
    mixed_platform_properties = (
        {"Precision": "mixed"} if mixed_platform_properties is None else mixed_platform_properties
    )

    inputs = load_case_raw_inputs(case_dir)
    system = mem00h_normalized_raw_system(inputs["system"])
    topology = inputs["topology"]
    ligand_indices = inputs["ligand_indices"]
    spec = inputs["spec"]
    case_dir = inputs["case_dir"]
    dynamics_dir = Path(case_dir) / "dynamics" if dynamics_dir is None else Path(dynamics_dir)

    charging0 = production_charging_system(
        system, ligand_indices, topology, spec, lambda_name=lambda_name
    )
    charging0_baked = core.bake_global_parameter_into_fixed_nonbonded_force(
        charging0, lambda_name, 0.0
    )
    vanishing_input = core.bake_global_parameter_into_fixed_nonbonded_force(
        charging0, lambda_name, 0.0
    )
    box_vectors_nm = np.load(str(case_dir / "box_vectors_nm.npy"))
    vanishing_one, vanishing_zero = production_vanishing_fixed_hamiltonian_systems(
        vanishing_input, ligand_indices, [1.0, 0.0], box_vectors_nm
    )
    reference_vzero = reference_vanishing_zero_system(system, ligand_indices, spec)

    # 归一化必须真的传导到这四个系统——A（charging）已经在 run_protocol_v2_
    # matrix 里核验过，这里核验 C/D 链路特有的几个（charging0_baked 是
    # C 的 reference 侧；vanishing_one/vanishing_zero 的 Group0 继承自
    # `vanishing_input`；reference_vzero 是 D 的独立 reference 侧）。
    for label, sys_to_check in (
        ("charging0_baked", charging0_baked),
        ("vanishing_one", vanishing_one),
        ("vanishing_zero", vanishing_zero),
        ("reference_vzero", reference_vzero),
    ):
        assert_mem00h_switching_convention(
            sys_to_check, context=f"run_protocol_v2_matrix_cd:{label}"
        )

    dcd_path = dynamics_dir / lambda0_dcd_name
    not_applicable_gate2 = {
        "applicable": False,
        "reason": (
            "vanishing 侧每个 λ_vdw 态的 CV 是固定构造（不是运行时可设的 "
            "GlobalParameter），charging 侧在喂给 vanishing 前已被结构性烘焙——"
            "没有可比较的 live-vs-baked 对象。"
        ),
        "passed": True,
    }

    frames: List[Dict[str, Any]] = []
    for frame_index in lambda0_frame_indices:
        positions_nm, box_nm = read_dcd_frame(dcd_path, topology, int(frame_index))

        c_gate1 = compare_endpoint(
            f"C_seam_gate1_reference_frame{frame_index}",
            vanishing_one, charging0_baked, positions_nm, box_nm,
            production_groups={0, 1, 2}, reference_groups={0},
            platform_name=reference_platform_name,
            force_gate_mode="hard",
        )
        c_gate3 = compare_endpoint(
            f"C_seam_gate3_mixed_production_vs_reference_frame{frame_index}",
            vanishing_one, charging0_baked, positions_nm, box_nm,
            production_groups={0, 1, 2}, reference_groups={0},
            platform_name=mixed_platform_name,
            platform_properties=mixed_platform_properties,
            force_gate_mode="diagnostic",
        )
        c_passed = bool(c_gate1["passed"] and c_gate3["passed"])

        d_gate1 = compare_endpoint(
            f"D_vs_reference_gate1_reference_frame{frame_index}",
            vanishing_zero, reference_vzero, positions_nm, box_nm,
            production_groups={0, 1, 2}, reference_groups={0},
            platform_name=reference_platform_name,
            force_gate_mode="hard",
        )
        d_gate3 = compare_endpoint(
            f"D_vs_reference_gate3_mixed_production_vs_reference_frame{frame_index}",
            vanishing_zero, reference_vzero, positions_nm, box_nm,
            production_groups={0, 1, 2}, reference_groups={0},
            platform_name=mixed_platform_name,
            platform_properties=mixed_platform_properties,
            force_gate_mode="diagnostic",
        )
        d_strict_zero_reference = compare_vanishing_zero_endpoint(
            vanishing_zero, positions_nm, box_nm,
            ligand_environment_groups={1},
            platform_name=reference_platform_name,
        )
        d_strict_zero_mixed = compare_vanishing_zero_endpoint(
            vanishing_zero, positions_nm, box_nm,
            ligand_environment_groups={1},
            platform_name=mixed_platform_name,
            platform_properties=mixed_platform_properties,
        )
        d_passed = bool(
            d_gate1["passed"] and d_gate3["passed"]
            and d_strict_zero_reference["passed"] and d_strict_zero_mixed["passed"]
        )

        frames.append(
            {
                "lambda_coul": 0.0,
                "dcd": lambda0_dcd_name,
                "frame_index": int(frame_index),
                "C": {
                    "gate1_reference_identity": c_gate1,
                    "gate2_mixed_live_vs_baked": dict(not_applicable_gate2),
                    "gate3_mixed_production_vs_reference": c_gate3,
                    "passed": c_passed,
                },
                "D": {
                    "gate1_reference_identity": d_gate1,
                    "gate2_mixed_live_vs_baked": dict(not_applicable_gate2),
                    "gate3_mixed_production_vs_reference": d_gate3,
                    "strict_zero_reference": d_strict_zero_reference,
                    "strict_zero_mixed": d_strict_zero_mixed,
                    "passed": d_passed,
                },
                "passed": bool(c_passed and d_passed),
            }
        )

    failed_frames = []
    for f in frames:
        if f["passed"]:
            continue
        failing = []
        if not f["C"]["passed"]:
            failing.append(
                "C:" + ",".join(
                    g for g in ("gate1_reference_identity", "gate3_mixed_production_vs_reference")
                    if not f["C"][g]["passed"]
                )
            )
        if not f["D"]["passed"]:
            failing.append(
                "D:" + ",".join(
                    g for g in (
                        "gate1_reference_identity",
                        "gate3_mixed_production_vs_reference",
                        "strict_zero_reference",
                        "strict_zero_mixed",
                    )
                    if not f["D"][g]["passed"]
                )
            )
        failed_frames.append(
            {
                "lambda_coul": f["lambda_coul"], "dcd": f["dcd"], "frame_index": f["frame_index"],
                "failing": failing,
            }
        )

    return {
        "case_dir": str(case_dir),
        "protocol_version": PROTOCOL_VERSION,
        "lambda_name": lambda_name,
        "reference_platform_name": reference_platform_name,
        "mixed_platform_name": mixed_platform_name,
        "mixed_platform_properties": dict(mixed_platform_properties),
        "n_frames": len(frames),
        "n_failed": len(failed_frames),
        "failed_frames": failed_frames,
        "frames": frames,
        "passed": len(failed_frames) == 0,
    }


# ---------------------------------------------------------------------------
# Reference / CPU / CUDA(double, deterministic) 三平台归因。
#
# 只回答一个问题："production 与 reference 的力差，是同一个哈密顿量在不同
# 平台上的浮点求和噪声，还是两侧真的构造了不一样的东西？" 判据：如果是浮点
# 噪声，差值的**大小**应该随精度提高（Reference/CPU 本身已是双精度；CUDA 请求
# double+deterministic 后应与它们同量级）而不是随平台"随机"变化；如果三个
# 平台在同一个 atom/component 上给出几乎相同的差值（不只是量级接近，是几乎
# 相同的数），那更像是两侧真的算的是不同的东西，而不是噪声。这里不代为下
# 结论，只把三个平台的原始数字、worst atom/component 摆在一起。
# ---------------------------------------------------------------------------


def three_platform_attribution(
    production_system: openmm.System,
    reference_system: openmm.System,
    positions_nm: np.ndarray,
    box_vectors_nm: np.ndarray,
    *,
    label: str,
    production_globals: Optional[Dict[str, float]] = None,
    reference_globals: Optional[Dict[str, float]] = None,
    production_groups: Optional[Iterable[int]] = None,
    reference_groups: Optional[Iterable[int]] = None,
    cuda_properties: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """同一帧在 `Reference`/`CPU`/`CUDA`（双精度+deterministic）三个平台各跑
    一次完整比较，原始结果全部保留，不做任何"哪个才对"的取舍。
    """
    cuda_properties = (
        {"Precision": "double", "DeterministicForces": "true"}
        if cuda_properties is None
        else dict(cuda_properties)
    )
    platform_specs = [
        ("Reference", None),
        ("CPU", None),
        ("CUDA", cuda_properties),
    ]
    per_platform = {}
    for platform_name, properties in platform_specs:
        comparison = compare_endpoint(
            f"{label}::{platform_name}",
            production_system, reference_system, positions_nm, box_vectors_nm,
            production_globals=production_globals,
            reference_globals=reference_globals,
            production_groups=production_groups,
            reference_groups=reference_groups,
            platform_name=platform_name,
            platform_properties=properties,
        )
        per_platform[platform_name] = comparison

    max_force_diffs = {name: c["max_abs_force_component_diff_kj_mol_nm"] for name, c in per_platform.items()}
    worst_atoms = {name: c["worst_atom_index"] for name, c in per_platform.items()}
    return {
        "label": label,
        "per_platform": per_platform,
        "max_abs_force_component_diff_kj_mol_nm_by_platform": max_force_diffs,
        "worst_atom_index_by_platform": worst_atoms,
        "worst_atom_index_stable_across_platforms": len(set(worst_atoms.values())) == 1,
    }


def _build_arg_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    smoke = sub.add_parser(
        "wiring-smoke",
        help="C3-2：对 C1/C2 raw 输入的单帧做 production vs reference 比较 + "
        "逐 force-group 账目（默认 CPU 平台，不是权威数值门）。",
    )
    smoke.add_argument("--case-dir", required=True, help="raw 输入所在目录（含 system.xml 等）")
    smoke.add_argument("--lambda1-dcd", required=True, help="λ_coul=1 状态的 dcd 文件名")
    smoke.add_argument("--lambda0-dcd", required=True, help="λ_coul=0 状态的 dcd 文件名")
    smoke.add_argument("--frame-index", type=int, default=10)
    smoke.add_argument("--platform", default="CPU")
    smoke.add_argument(
        "--dynamics-dir", default=None,
        help="真实轨迹所在目录，不给则默认 <case-dir>/dynamics（C1 布局）。"
        "C2 需要显式传 validation/c2_lipid_slab_v11_full11/<case>/dynamics"
        "——c2_lipid_slab_v11/<case>/dynamics 只是空占位。",
    )
    smoke.add_argument("--out", default=None, help="把 report JSON 写到这个路径")

    def _add_common_case_args(p):
        p.add_argument("--case-dir", required=True, help="raw 输入所在目录（含 system.xml 等）")
        p.add_argument("--lambda1-dcd", required=True, help="λ_coul=1 状态的 dcd 文件名")
        p.add_argument("--lambda0-dcd", required=True, help="λ_coul=0 状态的 dcd 文件名")
        p.add_argument(
            "--dynamics-dir", default=None,
            help="真实轨迹所在目录，不给则默认 <case-dir>/dynamics（C1 布局）。"
            "C2 需要显式传 validation/c2_lipid_slab_v11_full11/<case>/dynamics。",
        )
        p.add_argument("--out", default=None, help="把 report JSON 写到这个路径")

    def _add_platform_property_args(p, *, default_platform):
        p.add_argument("--platform", default=default_platform)
        p.add_argument(
            "--platform-property", action="append", default=[],
            metavar="KEY=VALUE",
            help="可重复传，例如 --platform-property Precision=double "
            "--platform-property DeterministicForces=true。",
        )

    matrix = sub.add_parser(
        "run-matrix",
        help="A/B 端点在固定帧索引表上的权威判定（不是单帧 smoke）。",
    )
    _add_common_case_args(matrix)
    _add_platform_property_args(matrix, default_platform="CUDA")
    matrix.add_argument(
        "--frame-set", choices=["c1", "c2"], required=True,
        help="c1：λ=1/λ=0 都用 [10,...,99]；c2：λ=1 用 [10,...,99]，"
        "λ=0 用 [20,...,199]（memtodolist.md §5）。",
    )

    matrix_v2 = sub.add_parser(
        "run-matrix-v2",
        help="Protocol v2 权威判定：Reference 恒等性门 + CUDA mixed"
        "（live-vs-baked 硬门 + production-vs-reference energy 硬门/force 诊断）"
        "三层门，见模块 docstring。",
    )
    _add_common_case_args(matrix_v2)
    matrix_v2.add_argument(
        "--frame-set", choices=["c1", "c2"], required=True,
        help="c1：λ=1/λ=0 都用 [10,...,99]；c2：λ=1 用 [10,...,99]，"
        "λ=0 用 [20,...,199]（memtodolist.md §5）。",
    )
    matrix_v2.add_argument("--lambda-name", default="lam_coul")
    matrix_v2.add_argument("--reference-platform", default="Reference")
    matrix_v2.add_argument("--mixed-platform", default="CUDA")
    matrix_v2.add_argument(
        "--mixed-platform-property", action="append", default=[],
        metavar="KEY=VALUE",
        help="默认 Precision=mixed（与生产一致）；可重复传覆盖。",
    )

    matrix_v2_cd = sub.add_parser(
        "run-matrix-v2-cd",
        help="Protocol v2 权威判定：C/D 端点，复用现有 charging λ_coul=0 "
        "真实帧（不重跑 vanishing MD）。",
    )
    matrix_v2_cd.add_argument("--case-dir", required=True, help="raw 输入所在目录（含 system.xml 等）")
    matrix_v2_cd.add_argument("--lambda0-dcd", required=True, help="λ_coul=0 状态的 dcd 文件名")
    matrix_v2_cd.add_argument(
        "--dynamics-dir", default=None,
        help="真实轨迹所在目录，不给则默认 <case-dir>/dynamics（C1 布局）。"
        "C2 需要显式传 validation/c2_lipid_slab_v11_full11/<case>/dynamics。",
    )
    matrix_v2_cd.add_argument("--out", default=None, help="把 report JSON 写到这个路径")
    matrix_v2_cd.add_argument(
        "--frame-set", choices=["c1", "c2"], required=True,
        help="c1/c2 的 λ=0 帧索引表与 run-matrix-v2 共用同一份常量。",
    )
    matrix_v2_cd.add_argument("--lambda-name", default="lambda_coul")
    matrix_v2_cd.add_argument("--reference-platform", default="Reference")
    matrix_v2_cd.add_argument("--mixed-platform", default="CUDA")
    matrix_v2_cd.add_argument(
        "--mixed-platform-property", action="append", default=[],
        metavar="KEY=VALUE",
        help="默认 Precision=mixed（与生产一致）；可重复传覆盖。",
    )

    attribute = sub.add_parser(
        "attribute",
        help="同一帧在 Reference/CPU/CUDA(double+deterministic) 三平台的力差归因，"
        "不预设结论，只把三个平台的原始数字摆在一起。",
    )
    _add_common_case_args(attribute)
    attribute.add_argument("--lambda-coul", type=float, required=True, choices=[0.0, 1.0])
    attribute.add_argument("--frame-index", type=int, required=True)
    attribute.add_argument(
        "--cuda-property", action="append", default=[],
        metavar="KEY=VALUE",
        help="覆盖 CUDA 属性，默认 Precision=double、DeterministicForces=true。",
    )

    return parser


def _parse_key_value_pairs(pairs: List[str]) -> Dict[str, str]:
    result = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"期望 KEY=VALUE，收到 {pair!r}")
        key, value = pair.split("=", 1)
        result[key] = value
    return result


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.stage == "wiring-smoke":
        report = wiring_smoke_report(
            args.case_dir,
            lambda1_dcd_name=args.lambda1_dcd,
            lambda0_dcd_name=args.lambda0_dcd,
            dynamics_dir=args.dynamics_dir,
            frame_index=args.frame_index,
            platform_name=args.platform,
        )
        text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
        print(text)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        return 0 if report["passed"] else 1

    if args.stage == "run-matrix":
        if args.frame_set == "c1":
            lambda1_indices, lambda0_indices = C1_LAMBDA1_FRAME_INDICES, C1_LAMBDA0_FRAME_INDICES
        else:
            lambda1_indices, lambda0_indices = C2_LAMBDA1_FRAME_INDICES, C2_LAMBDA0_FRAME_INDICES
        report = run_matrix(
            args.case_dir,
            lambda1_dcd_name=args.lambda1_dcd,
            lambda0_dcd_name=args.lambda0_dcd,
            lambda1_frame_indices=lambda1_indices,
            lambda0_frame_indices=lambda0_indices,
            dynamics_dir=args.dynamics_dir,
            platform_name=args.platform,
            platform_properties=_parse_key_value_pairs(args.platform_property),
        )
        text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
        print(text)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        return 0 if report["passed"] else 1

    if args.stage == "run-matrix-v2":
        if args.frame_set == "c1":
            lambda1_indices, lambda0_indices = C1_LAMBDA1_FRAME_INDICES, C1_LAMBDA0_FRAME_INDICES
        else:
            lambda1_indices, lambda0_indices = C2_LAMBDA1_FRAME_INDICES, C2_LAMBDA0_FRAME_INDICES
        mixed_props = _parse_key_value_pairs(args.mixed_platform_property) or None
        report = run_protocol_v2_matrix(
            args.case_dir,
            lambda1_dcd_name=args.lambda1_dcd,
            lambda0_dcd_name=args.lambda0_dcd,
            lambda1_frame_indices=lambda1_indices,
            lambda0_frame_indices=lambda0_indices,
            dynamics_dir=args.dynamics_dir,
            lambda_name=args.lambda_name,
            reference_platform_name=args.reference_platform,
            mixed_platform_name=args.mixed_platform,
            mixed_platform_properties=mixed_props,
        )
        text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
        print(text)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        return 0 if report["passed"] else 1

    if args.stage == "run-matrix-v2-cd":
        lambda0_indices = C1_LAMBDA0_FRAME_INDICES if args.frame_set == "c1" else C2_LAMBDA0_FRAME_INDICES
        mixed_props = _parse_key_value_pairs(args.mixed_platform_property) or None
        report = run_protocol_v2_matrix_cd(
            args.case_dir,
            lambda0_dcd_name=args.lambda0_dcd,
            lambda0_frame_indices=lambda0_indices,
            dynamics_dir=args.dynamics_dir,
            lambda_name=args.lambda_name,
            reference_platform_name=args.reference_platform,
            mixed_platform_name=args.mixed_platform,
            mixed_platform_properties=mixed_props,
        )
        text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
        print(text)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        return 0 if report["passed"] else 1

    if args.stage == "attribute":
        inputs = load_case_raw_inputs(args.case_dir)
        system = inputs["system"]
        topology = inputs["topology"]
        ligand_indices = inputs["ligand_indices"]
        spec = inputs["spec"]
        case_dir = inputs["case_dir"]
        dynamics_dir = (
            Path(case_dir) / "dynamics" if args.dynamics_dir is None else Path(args.dynamics_dir)
        )
        dcd_name = args.lambda1_dcd if args.lambda_coul == 1.0 else args.lambda0_dcd
        positions_nm, box_nm = read_dcd_frame(
            dynamics_dir / dcd_name, topology, args.frame_index
        )
        production = production_charging_system(system, ligand_indices, topology, spec)
        reference = reference_charging_endpoint_system(
            system, ligand_indices, spec, lam=args.lambda_coul
        )
        report = three_platform_attribution(
            production, reference, positions_nm, box_nm,
            label=f"{case_dir}::lambda={args.lambda_coul}::frame{args.frame_index}",
            production_globals={"lam_coul": args.lambda_coul},
            production_groups={0},
            reference_groups={0},
            cuda_properties=_parse_key_value_pairs(args.cuda_property) or None,
        )
        text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
        print(text)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        return 0

    raise AssertionError(f"未知 stage={args.stage!r}")


if __name__ == "__main__":
    sys.exit(main())
