"""P1-18：decharging REMD 必须接入 repeat-seed contract，且采样缓存身份含 seed。

## 缺陷是什么（已修，本文件现在是契约测试）

dual_lambda Stage 1（decharging，默认 PME decharging）的 REMDManager 构造
没有传 `random_seed`/`seed_ledger`/`seed_stage`/`seed_leg`——随机源未在
EXP-019 ledger 登记；`_remd_sampling_fingerprint` 的 payload 里也没有任何
repeat-seed/seed-contract 身份。后果：

- 所谓"独立重复"（repeat_seed=101/102）可能共享同一段 decharging 采样；
- 在同一输出目录改 repeat_seed 再 resume，sampling fingerprint 不变 ⟹
  命中并复用旧 DCD/u_kn；
- provenance 里即使有 seed ledger，也证明不了 Stage 1 真的消费了它。

## 修法（2026-08-30）

- decharging 分支与已接线的 single/2D、traditional 分支共用同一推导入口：
  `random_seed=self._seed_for("charging", stage_name, "exchange", "numpy")`，
  并传 `seed_ledger`/`seed_stage`/`seed_leg`；
- `_remd_sampling_fingerprint` 新增 `seed_contract` 字段，三处调用点
  （decharging、single/2D、traditional 的手写 payload）都传入
  `seed_contract_snapshot()` —— repeat_seed 一变，采样缓存立即失效。

## 不要这样让本文件变绿

把 seed_contract 从指纹 payload 里拿掉、或在 decharging 分支恢复"不传 seed"
的构造——那等于重新打开独立重复可能共享采样这条洞。
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

openmm = pytest.importorskip("openmm")
from openmm import app  # noqa: E402

import abfe_pipeline as pl  # noqa: E402
from ibs_engine import Exp019SeedLedger  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "abfe_pipeline.py"


# ---------------------------------------------------------------------------
# 1. 指纹行为：seed contract 进入采样缓存身份
# ---------------------------------------------------------------------------


class _EmptySystem:
    def getForces(self):
        return []


class _EmptyTopology:
    pass


def _fingerprint(seed_contract, leg_hint="complex"):
    return pl._remd_sampling_fingerprint(
        stage_name="decharging",
        system=pl.openmm.System(),
        topology=app.Topology(),
        ligand_indices=[0],
        lambdas_coul=[1.0, 0.0],
        lambdas_vdw=[1.0, 1.0],
        temperature_K=300.0,
        n_steps=100,
        exchange_interval=10,
        boresch_params=None,
        potential_type="softcore",
        platform_name="CPU",
        coion_identity=None,
        max_resident_contexts=None,
        seed_contract=seed_contract,
    )


def test_seed_contract_changes_the_sampling_fingerprint():
    fp_none = _fingerprint(None)
    fp_101 = _fingerprint({"repeat_seed": 101, "leg": "complex"})
    fp_102 = _fingerprint({"repeat_seed": 102, "leg": "complex"})
    assert fp_none["sha256"] != fp_101["sha256"]
    assert fp_101["sha256"] != fp_102["sha256"], (
        "repeat_seed=101/102 的 sampling fingerprint 必须不同（P1-18），"
        "否则独立重复会复用旧 DCD/u_kn"
    )


def test_real_seed_ledger_snapshots_change_the_fingerprint():
    snap_101 = Exp019SeedLedger(101, "complex").snapshot()
    snap_102 = Exp019SeedLedger(102, "complex").snapshot()
    assert snap_101 != snap_102
    assert _fingerprint(snap_101)["sha256"] != _fingerprint(snap_102)["sha256"]


# ---------------------------------------------------------------------------
# 2. decharging 分支接线：manager 消费 seed、指纹包含 contract（静态守护）
# ---------------------------------------------------------------------------


def _abfe_pipeline_class_source() -> str:
    src = PIPELINE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    pipeline_class = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ABFEPipeline"
    )
    return ast.get_source_segment(src, pipeline_class)


def test_decharging_remd_manager_receives_the_seed_contract():
    src = _abfe_pipeline_class_source()
    # decharging 分支的 REMDManager 构造必须带全套 seed 参数（与 single/2D、
    # traditional 分支同一约定）。
    assert src.count("random_seed=self._seed_for(") >= 2, (
        "decharging 与 single/2D 两个分支都必须把 random_seed 接进 REMDManager"
    )
    assert "seed_ledger=self.seed_ledger" in src
    assert "seed_stage=stage_name" in src
    assert "seed_leg=self.leg_name" in src


def test_both_dual_lambda_fingerprint_call_sites_pass_seed_contract():
    src = _abfe_pipeline_class_source()
    assert src.count("seed_contract=self.seed_contract_identity()") >= 2, (
        "decharging 与 single/2D 两处 sampling fingerprint 都必须包含 seed contract"
    )


# ---------------------------------------------------------------------------
# 3. complex/solvent 的 Stage 1 seed 确定且互不相同
# ---------------------------------------------------------------------------


def test_stage1_seeds_differ_between_legs_and_follow_repeat_seed():
    """mock 视角：complex/solvent 各自的 Stage 1 seed 确定且不同。"""

    def seed_for(repeat_seed, leg):
        ledger = Exp019SeedLedger(repeat_seed, leg)
        return ledger.derive("charging", "decharging", "exchange", "numpy", 0)

    assert seed_for(101, "complex") is not None
    assert seed_for(101, "complex") == seed_for(101, "complex")  # 确定
    assert seed_for(101, "complex") != seed_for(101, "solvent"), (
        "complex/solvent 必须拿到不同的 Stage 1 seed，否则两条腿共享随机流"
    )
    assert seed_for(101, "complex") != seed_for(102, "complex"), (
        "repeat_seed 改变必须改变 Stage 1 seed"
    )


def test_remd_manager_rejects_leg_mismatch():
    """REMDManager 的 seed_leg 与 ledger.leg 不一致时 fail closed。"""
    import inspect

    src = (ROOT / "ibs_engine.py").read_text(encoding="utf-8")
    assert 'seed_leg 与 EXP-019 seed ledger 不一致' in src, (
        "REMDManager 丢失了 seed_leg/ledger 一致性检查，P1-18 的接线就少了半边"
    )


# ---------------------------------------------------------------------------
# 5. [2026-08-31 P1] attachment 腿与三个外层聚合缓存键接入 repeat-seed contract
#
# 0831issue.md 两条 P1：
#   · Boresch attachment（stage0）此前用写死的 attachment_seed=20260728，
#     所有"独立重复"的 ΔG_attach 逐位相同 → 重复间经验 σ 被系统性低估。
#   · _stage_protocol_key / _build_top_level_protocol_key /
#     _build_sampling_protocol_key 都不含 seed_contract，而**外层早退先于内层
#     _remd_sampling_fingerprint 执行** → 同目录换 repeat_seed 后 --resume
#     会把旧 repeat 的聚合 ΔG 当本次结果返回。
# P1-18 只修了 _remd_sampling_fingerprint 那条路径，这两处在修复面之外。
# ---------------------------------------------------------------------------


def test_attachment_seed_is_derived_from_the_ledger_not_a_fixed_constant():
    src = _abfe_pipeline_class_source()
    assert 'self._seed_for("attachment", "stage0", "global", "integrator")' in src, (
        "attachment_config 的 seed 必须由 seed ledger 派生"
    )
    # 旧常量只允许作为"没有 ledger"时的兜底出现，不能是无条件取值
    assert 'if self.seed_ledger is not None' in src


def test_stage0_protocol_key_contains_the_seed_contract():
    src = _abfe_pipeline_class_source()
    i = src.index('"kind": "boresch_attachment_stage0"')
    block = src[i:i + 2000]
    assert '"seed_contract": self.seed_contract_identity()' in block, (
        "换 repeat_seed 必须让 stage0 attachment 缓存失配"
    )


def test_all_three_outer_aggregate_keys_carry_the_seed_contract():
    """外层三个键都必须条件插入 seed_contract（ledger 为 None 时不插入）。"""
    src = _abfe_pipeline_class_source()
    tree = ast.parse(PIPELINE_PATH.read_text(encoding="utf-8"))
    wanted = {
        "_stage_protocol_key",
        "_build_top_level_protocol_key",
        "_build_sampling_protocol_key",
    }
    seen = {}
    lines = PIPELINE_PATH.read_text(encoding="utf-8").split("\n")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            body = "\n".join(lines[node.lineno - 1:node.end_lineno])
            seen[node.name] = body
    assert seen.keys() == wanted, f"没找齐三个函数: {sorted(seen)}"
    for name, body in seen.items():
        assert "_seed_contract = self.seed_contract_identity()" in body, name
        assert 'payload["seed_contract"] = _seed_contract' in body, name
        # 必须是条件插入，否则未启用 repeat-seed 的旧指纹会无谓失效
        assert "if _seed_contract is not None:" in body, name


def test_attachment_seed_actually_changes_with_repeat_seed():
    """真实派生：换 repeat_seed 必须给出不同的 attachment 种子。"""

    def attach_seed(repeat_seed, leg="complex"):
        ledger = Exp019SeedLedger(repeat_seed, leg)
        return ledger.derive("attachment", "stage0", "global", "integrator", 0)

    a = attach_seed(20260905)
    b = attach_seed(20260906)
    c = attach_seed(20260907)
    assert None not in (a, b, c)
    assert len({a, b, c}) == 3, (a, b, c)
    # 同一 repeat_seed 必须确定可复现
    assert attach_seed(20260905) == a
    # 两条腿也不能撞
    assert attach_seed(20260905, "solvent") != a


# ---------------------------------------------------------------------------
# [2026-09-01] 缓存键只能带**稳定身份**，不能带 derived_seed_map
#
# 事故（4W53 resume_v3.log）：Stage 1 结果缓存被拒，原因之一是
#   seed_contract.derived_seed_map.equilibration/pre_equilibration/global/velocity/0:
#   缓存=285721177 -> 当前='<missing>'
# derived_seed_map 记的是"本进程到此刻为止派生过哪些种子"，随执行路径增长：
# 首跑时预平衡真跑过、派生了 equilibration/* 的种子；resume 时预平衡走 checkpoint
# 没派生，同一份配置下快照就不同 ⇒ 每次 resume 都判失配、整段重算。
# 缓存键必须只依赖**配置身份**，不能依赖**执行历史**。
# ---------------------------------------------------------------------------

def test_identity_is_stable_across_derivations_but_snapshot_is_not():
    a = Exp019SeedLedger(20260908, "complex")
    before = a.identity()
    a.derive("equilibration", "pre_equilibration", "global", "velocity", 0)
    a.derive("equilibration", "boresch_rebalance", "global", "velocity", 0)
    assert a.identity() == before, "identity 不得随已派生种子变化"
    # 对照：snapshot 会变——这正是它不能进缓存键的原因
    assert a.snapshot() != Exp019SeedLedger(20260908, "complex").snapshot()
    assert "derived_seed_map" not in a.identity()


def test_identity_still_distinguishes_repeat_seed_and_leg():
    base = Exp019SeedLedger(20260908, "complex").identity()
    assert Exp019SeedLedger(20260909, "complex").identity() != base
    assert Exp019SeedLedger(20260908, "solvent").identity() != base


def test_no_cache_key_uses_the_path_dependent_snapshot():
    """协议指纹/缓存键一律走 identity；snapshot 只留给 provenance。"""
    src = PIPELINE_PATH.read_text(encoding="utf-8")
    lines = [
        l for l in src.split("\n")
        if "self.seed_contract_snapshot()" in l
        and not l.strip().startswith("#")
        and "`" not in l          # 排除 docstring / 注释里的文字引用
    ]
    assert lines == [], f"这些行仍在用路径依赖的 snapshot 做键：{lines}"
    # 且 identity 确实被用上了
    assert src.count("seed_contract_identity()") >= 5


def test_identity_works_with_a_ledger_that_only_has_snapshot():
    """ledger 是 duck-typed 的：只实现 snapshot() 的替身也必须能取到稳定身份。

    2026-09-01 踩过：`seed_contract_identity()` 直接调 `_ledger.identity()`，
    在只实现 snapshot() 的测试替身上抛 AttributeError。
    语义保证（键里没有 derived_seed_map）必须与 ledger 实现无关。
    """
    import abfe_pipeline

    class _SnapshotOnlyLedger:
        def snapshot(self):
            return {
                "protocol_version": 1,
                "repeat_seed": 20260908,
                "leg": "complex",
                "derived_seed_map": {"equilibration/pre_equilibration/x/y/0": 123},
            }

    stub = object.__new__(abfe_pipeline.ABFEPipeline)
    stub.seed_ledger = _SnapshotOnlyLedger()
    got = abfe_pipeline.ABFEPipeline.seed_contract_identity(stub)
    assert got == {"protocol_version": 1, "repeat_seed": 20260908, "leg": "complex"}
    assert "derived_seed_map" not in got


# ---------------------------------------------------------------------------
# [2026-09-01] 缓存键里的 boresch_params 必须走收窄口径
#
# 事故（4W53 resume_v3.log）：Stage 1 结果缓存被拒，原因之一是
#   boresch_params.last_frame_geometry_diagnostic: 缓存={...} -> 当前='<missing>'
# 首次提交路径 `_commit_ensemble_boresch_equilibrium` 会往 boresch_params 里塞这个
# **诊断**字段，而 resume 复用已提交值的分支不会再补 —— 两条路径产出的字典形状永远
# 对不上，即使真正决定 Hamiltonian 的锚点/平衡值/力常数逐位相同。
# stage0 在 2026-08-27 就改用了收窄函数，其余键当时没跟上。
# ---------------------------------------------------------------------------

def test_every_cache_key_narrows_boresch_params():
    """任何进缓存键的 boresch_params 都必须先过 `_preopt_boresch_protocol_payload`。"""
    src = PIPELINE_PATH.read_text(encoding="utf-8")
    raw = [
        (i + 1, l) for i, l in enumerate(src.split("\n"))
        if '"boresch_params": boresch_params' in l
    ]
    assert raw == [], f"这些键仍放未收窄的 boresch_params：{raw}"
    # 且收窄口径确实在用（stage0 / stage / 顶层 / 两处 sampling fingerprint …）
    assert src.count('"boresch_params": _preopt_boresch_protocol_payload') >= 6


def test_narrowing_drops_diagnostics_but_keeps_the_hamiltonian():
    """收窄必须扔掉诊断、保住四项 Hamiltonian 身份。"""
    from abfe_pipeline import _preopt_boresch_protocol_payload

    params = {
        "receptor_indices": [1, 2, 3],
        "ligand_indices": [4, 5, 6],
        "equilibrium_values": {
            "r0": 0.36, "thetaA0": 1.47, "thetaB0": 1.42,
            "phiA0": 1.44, "phiB0": -2.32, "phiC0": -1.41,
        },
        "force_constants": {
            "kr": 2000.0, "kthetaA": 200.0, "kthetaB": 200.0,
            "kphiA": 100.0, "kphiB": 100.0, "kphiC": 100.0,
        },
    }
    bare = _preopt_boresch_protocol_payload(dict(params))
    with_diag = _preopt_boresch_protocol_payload(
        {**params, "last_frame_geometry_diagnostic": {"status": "DIAGNOSTIC_ONLY"},
         "total_score": 0.87, "provenance": {"when": "now"}}
    )
    assert bare == with_diag, "加一段诊断不得改变缓存身份"
    # 而真正决定 Hamiltonian 的量变了，身份必须变
    moved = _preopt_boresch_protocol_payload(
        {**params, "force_constants": {**params["force_constants"], "kr": 1000.0}}
    )
    assert moved != bare
