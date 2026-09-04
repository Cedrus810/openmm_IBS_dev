"""RBFE R0 契约（`docs/design/PLAN_rbfe_interface_and_implementation.md` §2/§3/§8）。

R0 的验收标准（计划 §8）：**错误输入被拒绝；合成两腿数据的符号／单位／误差传播
正确；不启动 GPU。** 本文件按这三条组织，不 import openmm、不建任何 System。
"""

from __future__ import annotations

import math

import pytest

import rbfe_core as rc

_HASH_A = "a" * 64
_HASH_B = "b" * 64


def _endpoint(name, charge=0, **kw):
    kw.setdefault("structure", "[H]c1ccccc1")
    kw.setdefault("input_path", f"/tmp/{name}.sdf")
    kw.setdefault("input_sha256", _HASH_A if name == "A" else _HASH_B)
    kw.setdefault("protonation_state", "neutral_pH7")
    kw.setdefault("stereochemistry", "S")
    kw.setdefault("partial_charge_source", "am1bcc")
    return rc.LigandEndpoint(name=name, formal_charge=charge, **kw)


def _spec(**kw):
    ligand_a = kw.pop("ligand_a", _endpoint("A"))
    ligand_b = kw.pop("ligand_b", _endpoint("B"))
    env = kw.pop(
        "environment",
        rc.EnvironmentSpec(
            receptor_name="4W53",
            receptor_path="/tmp/rec.pdb",
            receptor_sha256="c" * 64,
            force_field="amber14sb",
            water_model="tip3p",
            ion_model="joung_cheatham",
        ),
    )
    protocol = kw.pop(
        "protocol",
        rc.ProtocolSpec(
            temperature_kelvin=298.15,
            pressure_bar=1.0,
            n_lambda_states=12,
            n_steps_per_state=500_000,
            seed=20260901,
            lambda_schedule_name="uniform_12",
        ),
    )
    return rc.EdgeSpec(
        edge_id=kw.pop("edge_id", "A_to_B"),
        ligand_a=ligand_a,
        ligand_b=ligand_b,
        environment=env,
        protocol=protocol,
        output_dir=kw.pop("output_dir", "/tmp/out"),
        **kw,
    )


def _leg(phase, dg, se, *, edge_id="E1", ok=True, unit=rc.KJ_PER_MOL):
    return rc.LegResult(
        phase=phase,
        edge_id=edge_id,
        ligand_a_name="A",
        ligand_b_name="B",
        delta_g=dg,
        stderr=se,
        energy_unit=unit,
        uncertainty_method="mbar",
        n_effective_samples=500,
        quality_gate_passed=ok,
        artifacts_fingerprint="deadbeef",
    )


# ---------------------------------------------------------------------------
# 符号约定（计划 §3）—— R0 最重要的一条
# ---------------------------------------------------------------------------


def test_ddg_is_complex_minus_solvent():
    """ΔΔG_bind(B-A) = ΔG_complex(A→B) - ΔG_solvent(A→B)（计划 §3）。"""
    result = rc.combine_rbfe(_leg("complex", -10.0, 0.0), _leg("solvent", -2.0, 0.0))
    assert result.ddg_bind == pytest.approx(-8.0)


def test_negative_ddg_means_b_binds_better():
    """计划 §3：负的 ΔΔG 表示 B 的结合自由能更低。"""
    result = rc.combine_rbfe(_leg("complex", -10.0, 0.0), _leg("solvent", -2.0, 0.0))
    assert result.ddg_bind < 0
    assert "B" in result.interpretation()


def test_positive_ddg_means_a_binds_better():
    result = rc.combine_rbfe(_leg("complex", -2.0, 0.0), _leg("solvent", -10.0, 0.0))
    assert result.ddg_bind > 0
    assert result.interpretation().startswith("A")


def test_sign_is_opposite_to_abfe_convention():
    """🔑 计划 §3 加粗警告：ABFE 用 solvent−complex，RBFE 用 complex−solvent。

    这个测试存在的唯一目的，是让「有人把 ABFE 的汇总函数搬过来用」立刻变红。
    两种口径给出的数值互为相反数，量级看着都合理，事后极难发现。
    """
    dg_complex, dg_solvent = -10.0, -2.0
    rbfe = rc.combine_rbfe(
        _leg("complex", dg_complex, 0.0), _leg("solvent", dg_solvent, 0.0)
    ).ddg_bind
    abfe_style = dg_solvent - dg_complex
    assert rbfe == pytest.approx(-abfe_style)
    assert rbfe != pytest.approx(abfe_style)


def test_legs_must_be_passed_in_the_right_order():
    with pytest.raises(ValueError, match="complex"):
        rc.combine_rbfe(_leg("solvent", -2.0, 1.0), _leg("complex", -10.0, 1.0))


def test_direction_is_locked():
    with pytest.raises(ValueError, match="direction"):
        rc.LegResult(
            phase="complex",
            edge_id="E1",
            ligand_a_name="A",
            ligand_b_name="B",
            delta_g=-1.0,
            stderr=0.1,
            energy_unit=rc.KJ_PER_MOL,
            uncertainty_method="mbar",
            n_effective_samples=1,
            quality_gate_passed=True,
            artifacts_fingerprint="x",
            direction="B_to_A",
        )


# ---------------------------------------------------------------------------
# 误差传播（计划 §7）
# ---------------------------------------------------------------------------


def test_independent_legs_add_variances():
    """独立两腿时方差**相加**——差的方差仍然是和，不是差。"""
    result = rc.combine_rbfe(_leg("complex", 0.0, 3.0), _leg("solvent", 0.0, 4.0))
    assert result.ddg_stderr == pytest.approx(5.0)  # sqrt(9+16)


def test_covariance_is_subtracted_twice():
    """Var(X−Y) = Var(X) + Var(Y) − 2Cov(X,Y)。"""
    result = rc.combine_rbfe(
        _leg("complex", 0.0, 3.0), _leg("solvent", 0.0, 4.0), covariance=6.0
    )
    assert result.ddg_stderr == pytest.approx(math.sqrt(25.0 - 12.0))


def test_negative_covariance_increases_uncertainty():
    result = rc.combine_rbfe(
        _leg("complex", 0.0, 3.0), _leg("solvent", 0.0, 4.0), covariance=-6.0
    )
    assert result.ddg_stderr == pytest.approx(math.sqrt(25.0 + 12.0))


def test_inconsistent_covariance_raises_instead_of_clamping():
    """负方差说明输入自相矛盾。clamp 到 0 会把它伪装成「误差很小」。"""
    with pytest.raises(ValueError, match="方差为负"):
        rc.combine_rbfe(
            _leg("complex", 0.0, 3.0), _leg("solvent", 0.0, 4.0), covariance=100.0
        )


def test_zero_stderr_legs_give_zero_uncertainty():
    result = rc.combine_rbfe(_leg("complex", -5.0, 0.0), _leg("solvent", -1.0, 0.0))
    assert result.ddg_stderr == pytest.approx(0.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_energies_rejected(bad):
    with pytest.raises(ValueError):
        _leg("complex", bad, 1.0)


def test_negative_stderr_rejected():
    with pytest.raises(ValueError):
        _leg("complex", -1.0, -0.5)


# ---------------------------------------------------------------------------
# 身份校验（计划 §4.1：EdgeResult 是「经过身份校验的两腿结果」）
# ---------------------------------------------------------------------------


def test_legs_from_different_edges_cannot_be_combined():
    with pytest.raises(ValueError, match="edge_id"):
        rc.combine_rbfe(
            _leg("complex", -10.0, 1.0, edge_id="E1"),
            _leg("solvent", -2.0, 1.0, edge_id="E2"),
        )


def test_legs_with_different_units_cannot_be_combined():
    with pytest.raises(ValueError, match="energy_unit"):
        rc.combine_rbfe(
            _leg("complex", -10.0, 1.0),
            _leg("solvent", -2.0, 1.0, unit=rc.KCAL_PER_MOL),
        )


def test_failed_quality_gate_propagates_to_unqualified():
    result = rc.combine_rbfe(_leg("complex", -10.0, 1.0, ok=False), _leg("solvent", -2.0, 1.0))
    assert not result.qualified
    assert any("complex" in r for r in result.qualification_reasons)


def test_both_gates_passing_qualifies():
    assert rc.combine_rbfe(_leg("complex", -10.0, 1.0), _leg("solvent", -2.0, 1.0)).qualified


# ---------------------------------------------------------------------------
# 单位
# ---------------------------------------------------------------------------


def test_unit_conversion_round_trips():
    assert rc.convert_energy(4.184, rc.KJ_PER_MOL, rc.KCAL_PER_MOL) == pytest.approx(1.0)
    assert rc.convert_energy(1.0, rc.KCAL_PER_MOL, rc.KJ_PER_MOL) == pytest.approx(4.184)
    assert rc.convert_energy(7.0, rc.KJ_PER_MOL, rc.KJ_PER_MOL) == 7.0


def test_unknown_unit_rejected():
    with pytest.raises(ValueError):
        rc.convert_energy(1.0, "hartree", rc.KJ_PER_MOL)


def test_result_dict_states_unit_and_direction():
    """计划 §3：输出必须同时记录 A、B、方向、单位和每条腿的值。"""
    payload = rc.combine_rbfe(_leg("complex", -10.0, 1.0), _leg("solvent", -2.0, 1.0)).to_dict()
    assert payload["direction"] == rc.RBFE_DIRECTION
    assert payload["energy_unit"] == rc.KJ_PER_MOL
    assert payload["ligand_A"] == "A" and payload["ligand_B"] == "B"
    assert set(payload["legs"]) == {"complex", "solvent"}
    assert "ddG_bind_B_minus_A" in payload  # 字段名自带方向，读的人不用猜


# ---------------------------------------------------------------------------
# 首版范围拒绝（计划 §2）
# ---------------------------------------------------------------------------


def test_valid_edge_passes():
    assert rc.validate_edge(_spec()).ok


def test_net_charge_change_rejected():
    report = rc.validate_edge(_spec(ligand_b=_endpoint("B", charge=-1)))
    assert not report.ok
    assert any("净电荷变化" in e for e in report.errors)


def test_charged_but_equal_ligands_also_rejected_for_a_different_reason():
    """净电荷变化和「同电荷带电配体」是两条独立判据，不能合并。"""
    report = rc.validate_edge(
        _spec(ligand_a=_endpoint("A", charge=-1), ligand_b=_endpoint("B", charge=-1))
    )
    assert not report.ok
    assert any("非中性" in e for e in report.errors)
    assert not any("净电荷变化" in e for e in report.errors)


def test_rejection_message_does_not_claim_the_method_cannot_do_it():
    """计划 §2：这些是本项目首版的范围限制，**不代表 RBFE 方法普遍不支持**。

    错误信息必须保留这个区分，否则以后会有人拿它当「做不到」的结论引用。
    """
    report = rc.validate_edge(_spec(ligand_b=_endpoint("B", charge=-1)))
    assert any("不是 RBFE 方法本身不支持" in e for e in report.errors)


def test_membrane_rejected():
    env = rc.EnvironmentSpec(
        receptor_name="GPCR",
        receptor_path="/tmp/r.pdb",
        receptor_sha256="c" * 64,
        force_field="charmm36",
        water_model="tip3p",
        ion_model="default",
        is_membrane=True,
    )
    report = rc.validate_edge(_spec(environment=env))
    assert not report.ok
    assert any("膜体系" in e for e in report.errors)


def test_protonation_change_rejected():
    report = rc.validate_edge(_spec(ligand_b=_endpoint("B", protonation_state="protonated_amine")))
    assert not report.ok
    assert any("质子化态改变" in e for e in report.errors)


def test_self_edge_rejected():
    """A→A 是 R3 的验收用例，不是一条可以直接跑的生产边。"""
    report = rc.validate_edge(_spec(ligand_b=_endpoint("A")))
    assert not report.ok


@pytest.mark.parametrize(
    "field,value",
    [("protonation_state", ""), ("stereochemistry", ""), ("partial_charge_source", "")],
)
def test_undeclared_identity_fields_rejected(field, value):
    """沉默的默认值正是这类项目最容易出错的地方——未声明就是拒绝。"""
    report = rc.validate_edge(_spec(ligand_b=_endpoint("B", **{field: value})))
    assert not report.ok


def test_bad_input_hash_rejected():
    report = rc.validate_edge(_spec(ligand_b=_endpoint("B", input_sha256="short")))
    assert not report.ok
    assert any("sha256" in e for e in report.errors)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature_kelvin": 0.0},
        {"temperature_kelvin": -1.0},
        {"pressure_bar": -1.0},
        {"n_lambda_states": 1},
        {"n_steps_per_state": 0},
        {"lambda_schedule_name": ""},
    ],
)
def test_invalid_protocol_rejected(kwargs):
    base = dict(
        temperature_kelvin=298.15,
        pressure_bar=1.0,
        n_lambda_states=12,
        n_steps_per_state=1000,
        seed=1,
        lambda_schedule_name="uniform_12",
    )
    base.update(kwargs)
    assert not rc.validate_edge(_spec(protocol=rc.ProtocolSpec(**base))).ok


def test_lambda_schedule_must_be_declared_not_inherited_from_abfe():
    """计划 §2：λ 与温度／压力／约束协议显式配置，不沿用 ABFE 的隐藏默认值。"""
    protocol = rc.ProtocolSpec(
        temperature_kelvin=298.15,
        pressure_bar=1.0,
        n_lambda_states=12,
        n_steps_per_state=1000,
        seed=1,
        lambda_schedule_name="",
    )
    report = rc.validate_edge(_spec(protocol=protocol))
    assert any("ABFE" in e for e in report.errors)


def test_validation_declares_what_it_did_not_check():
    """🔑 R0 没有原子映射，环变化/手性/共价这些**查不了**。

    必须显式声明「没查」，不能让 PASS 被读成「全都查过了」——这正是计划 §2
    那张拒绝清单里最容易被误以为已经生效的部分。
    """
    report = rc.validate_edge(_spec())
    assert report.ok
    joined = " ".join(report.unchecked)
    for topic in ("环断裂", "手性", "共价", "互变异构"):
        assert topic in joined, f"未检查清单里缺 {topic}"


def test_raise_if_failed_reports_every_error():
    report = rc.validate_edge(
        _spec(ligand_b=_endpoint("B", charge=-1, protonation_state="protonated"))
    )
    with pytest.raises(rc.RBFEUnsupportedTransformationError) as excinfo:
        report.raise_if_failed()
    assert "净电荷变化" in str(excinfo.value)
    assert "质子化态改变" in str(excinfo.value)


def test_manifest_carries_full_identity():
    """计划 §7：元数据包含 A/B 化学及输入身份、力场、方向、单位。"""
    manifest = _spec().manifest()
    assert manifest["direction"] == rc.RBFE_DIRECTION
    assert manifest["ligand_A"]["input_sha256"] == _HASH_A
    assert manifest["environment"]["force_field"] == "amber14sb"
    assert manifest["protocol"]["lambda_schedule_name"] == "uniform_12"


# ---------------------------------------------------------------------------
# 未实现的部分必须抛错，不能返回假成功（计划 §4.1）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn,args",
    [
        (rc.prepare_edge, (None,)),
        (rc.build_hybrid_leg, (None, "complex")),
    ],
)
def test_unimplemented_stages_raise(fn, args):
    """「不提供尚未实现但返回成功的占位 sampler」——接口存在 ≠ 科学计算可用。

    ⚠ `analyze_leg` **已于 2026-09-03（R2）实现**，从这张表里移走了。
    它此前是第三个参数化用例；实现之后 `analyze_leg(None, None)` 会先因为缺
    keyword-only 参数抛 `TypeError`，而不是 `NotImplementedError`——留着的话
    这条测试测的就不再是「未实现的必须抛错」，而是「调用签名写错了会报错」。
    真正实现了的东西不该继续挂在"未实现"清单里。
    """
    with pytest.raises(NotImplementedError):
        fn(*args)


def test_core_has_no_module_level_openmm_or_abfe_import():
    """`rbfe_core` **模块顶层**不得 import openmm / ABFE / ibs_engine / rbfe_pipeline。

    ## 这条测的是什么，以及为什么口径变了

    原本的意图是「rbfe_core 不得反向 import ABFE」。2026-09-03 用户明确改口：
    **RBFE 直接 import `abfe_core` 复用，但不改 ABFE 一行代码。** 所以「不许 import」
    不再成立，剩下的约束是**位置**：

      * 顶层 import 会让 `import rbfe_core` 无条件拉进 openmm/pymbar——R0 与 R1a
        那两层「不 import openmm、不启动 GPU」的性质连同它们上百条测试会一起失效；
      * 函数体内的惰性 import 只在真正用到 R1b/R2 时付出代价，两个性质都保住。

    判据因此是「行首（第 0 列）不得出现这些 import」。缩进的惰性 import 是**刻意的**，
    不算违规。`rbfe_pipeline` 仍然一概不许——那是真正的反向依赖，跟位置无关，
    另有下面一条专门守它。
    """
    from pathlib import Path

    source = Path(rc.__file__).read_text(encoding="utf-8")
    offenders = [
        line
        for line in source.splitlines()
        if (line.startswith("import ") or line.startswith("from "))
        and any(
            mod in line
            for mod in ("openmm", "abfe_core", "abfe_pipeline", "ibs_engine", "rbfe_pipeline")
        )
    ]
    assert not offenders, f"这些 import 必须挪进函数体（惰性）：{offenders}"


def test_core_never_imports_rbfe_pipeline_even_lazily():
    """反向依赖是硬禁：`rbfe_core` 任何位置都不许 import `rbfe_pipeline`。

    与上一条不同——那条管的是"放哪里"，这条管的是"能不能"。依赖方向
    `runrbfe -> rbfe_pipeline -> rbfe_core` 是单向的，反过来就成环。
    """
    from pathlib import Path

    source = Path(rc.__file__).read_text(encoding="utf-8")
    offenders = [
        line.strip()
        for line in source.splitlines()
        # 只看**真正的 import 语句**。此前这里是裸的 `"import " in line`，
        # 结果把模块文档串里"本模块不 import rbfe_pipeline"那句话当成了违规。
        if "rbfe_pipeline" in line
        and (line.strip().startswith("import ") or line.strip().startswith("from "))
    ]
    assert not offenders, f"rbfe_core 反向 import 了 rbfe_pipeline：{offenders}"


def test_lazy_imports_are_actually_lazy():
    """确认惰性 import 真的惰性：单独 import rbfe_core 不应把 openmm 拉进来。"""
    import subprocess
    import sys as _sys
    from pathlib import Path

    repo_root = Path(rc.__file__).resolve().parent
    probe = (
        "import sys; sys.path.insert(0, %r); import rbfe_core; "
        "print('openmm' in sys.modules, 'ibs_engine' in sys.modules)" % str(repo_root)
    )
    out = subprocess.run(
        [_sys.executable, "-c", probe], capture_output=True, text=True, timeout=300
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False False", out.stdout
