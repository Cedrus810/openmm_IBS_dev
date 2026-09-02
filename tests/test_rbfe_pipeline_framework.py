"""RBFE 编排层框架（`docs/design/PLAN_rbfe_interface_and_implementation.md` §4/§7）。

覆盖的是**不依赖 R1** 的那几层：身份指纹与续跑规则、运行目录、两腿编排的身份校验、
独立重复聚合、边网络、ABFE 锚点换算。全部纯计算，不建 System、不启动 GPU。
"""

from __future__ import annotations

import json
import math

import pytest

import rbfe_core as rc
import rbfe_pipeline as rp

_H = "a" * 64


def _endpoint(name, **kw):
    kw.setdefault("structure", "[H]c1ccccc1")
    kw.setdefault("input_path", f"/tmp/{name}.sdf")
    kw.setdefault("input_sha256", _H)
    kw.setdefault("protonation_state", "neutral_pH7")
    kw.setdefault("stereochemistry", "S")
    kw.setdefault("partial_charge_source", "am1bcc")
    kw.setdefault("formal_charge", 0)
    return rc.LigandEndpoint(name=name, **kw)


def _spec(tmp_path=None, **kw):
    env = kw.pop("environment", None) or rc.EnvironmentSpec(
        receptor_name="R", receptor_path="/tmp/r.pdb", receptor_sha256="c" * 64,
        force_field="amber14sb", water_model="tip3p", ion_model="jc",
    )
    proto = kw.pop("protocol", None) or rc.ProtocolSpec(
        temperature_kelvin=298.15, pressure_bar=1.0, n_lambda_states=12,
        n_steps_per_state=1000, seed=1, lambda_schedule_name="uniform_12",
    )
    return rc.EdgeSpec(
        edge_id=kw.pop("edge_id", "A_to_B"),
        ligand_a=kw.pop("ligand_a", None) or _endpoint("A"),
        ligand_b=kw.pop("ligand_b", None) or _endpoint("B"),
        environment=env, protocol=proto,
        output_dir=str(tmp_path) if tmp_path else "/tmp/out",
        **kw,
    )


def _leg(phase, dg, se, *, edge_id="A_to_B", a="A", b="B", ok=True):
    return rc.LegResult(
        phase=phase, edge_id=edge_id, ligand_a_name=a, ligand_b_name=b,
        delta_g=dg, stderr=se, energy_unit=rc.KJ_PER_MOL, uncertainty_method="mbar",
        n_effective_samples=100, quality_gate_passed=ok, artifacts_fingerprint="f",
    )


def _edge_result(dg_complex=-10.0, dg_solvent=-2.0, se=1.0, **kw):
    return rc.combine_rbfe(_leg("complex", dg_complex, se, **kw),
                           _leg("solvent", dg_solvent, se, **kw))


# ---------------------------------------------------------------------------
# 身份指纹：只放身份，不放执行历史
# ---------------------------------------------------------------------------


def test_identity_excludes_execution_history():
    """🔑 本仓库在 ABFE 那边为「缓存键里混进执行历史」付过大代价。

    指纹里出现代码哈希/时间戳/步数，任何无关改动都会让 resume 被迫重跑 GPU。
    """
    identity = rp.edge_identity(_spec())
    forbidden = ("code_sha256", "timestamp", "created_at", "elapsed", "steps_done", "hostname")
    for key in identity:
        assert not any(f in key for f in forbidden), f"身份里混进了执行历史：{key}"


def test_identity_contains_what_the_plan_requires():
    """计划 §7：A/B 化学及输入身份、映射和参数 hash、builder 版本、λ 路径、方向、单位。"""
    identity = rp.edge_identity(
        _spec(), atom_mapping_hash="m" * 64, hybrid_builder_version="builder-1"
    )
    for key in (
        "ligand_A", "ligand_B", "direction", "energy_unit", "force_field",
        "lambda_schedule_name", "atom_mapping_sha256", "hybrid_builder_version",
        "temperature_kelvin", "receptor_sha256",
    ):
        assert key in identity, f"身份缺 {key}"


def test_platform_and_backend_are_not_in_edge_identity():
    """换平台不该让映射和建系那一步作废——后端进的是**腿**的身份。"""
    identity = rp.edge_identity(_spec())
    assert not any("backend" in k or "platform" in k for k in identity)


def test_backend_is_in_leg_identity():
    """§4.3：禁止跨后端向未完成轨迹追加帧——所以后端必须进腿的身份。"""
    import free_energy_engine as fee

    ident = rp.leg_identity(_spec(), "complex", backend=fee.resolve_remd_backend("legacy"))
    assert ident["backend"] == "legacy"
    assert ident["exchange_scheme"] == fee.EXCHANGE_SCHEME_LEGACY


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: {"ligand_b": _endpoint("B", input_sha256="b" * 64)},
        lambda s: {"environment": rc.EnvironmentSpec(
            receptor_name="R", receptor_path="/tmp/r.pdb", receptor_sha256="c" * 64,
            force_field="charmm36", water_model="tip3p", ion_model="jc")},
        lambda s: {"protocol": rc.ProtocolSpec(
            temperature_kelvin=310.0, pressure_bar=1.0, n_lambda_states=12,
            n_steps_per_state=1000, seed=1, lambda_schedule_name="uniform_12")},
        lambda s: {"protocol": rc.ProtocolSpec(
            temperature_kelvin=298.15, pressure_bar=1.0, n_lambda_states=12,
            n_steps_per_state=1000, seed=1, lambda_schedule_name="other_schedule")},
    ],
)
def test_fingerprint_changes_when_identity_changes(mutate):
    base = _spec()
    assert rp.edge_fingerprint(base) != rp.edge_fingerprint(_spec(**mutate(base)))


def test_fingerprint_is_stable_across_calls():
    assert rp.edge_fingerprint(_spec()) == rp.edge_fingerprint(_spec())


def test_atom_mapping_change_invalidates_identity():
    """§7：禁止跨映射直接追加。"""
    a = rp.edge_fingerprint(_spec(), atom_mapping_hash="1" * 64)
    b = rp.edge_fingerprint(_spec(), atom_mapping_hash="2" * 64)
    assert a != b


# ---------------------------------------------------------------------------
# 运行目录与续跑
# ---------------------------------------------------------------------------


def test_prepare_creates_layout_and_manifest(tmp_path):
    spec = _spec(tmp_path)
    layout = rp.prepare_run_directory(spec, repeat_index=1)
    assert layout.manifest_path.exists()
    for phase in rc.RBFE_PHASES:
        assert layout.leg_dir(phase).is_dir()
    manifest = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    assert manifest["edge_fingerprint"] == rp.edge_fingerprint(spec)
    assert manifest["direction"] == rc.RBFE_DIRECTION


def test_invalid_edge_creates_no_directories(tmp_path):
    """先验证再建目录——别留一堆注定跑不成的空目录。"""
    spec = _spec(tmp_path, ligand_b=_endpoint("B", formal_charge=-1))
    with pytest.raises(rc.RBFEUnsupportedTransformationError):
        rp.prepare_run_directory(spec)
    assert not (tmp_path / "repeat_01").exists()


def test_reprepare_with_same_identity_is_allowed(tmp_path):
    spec = _spec(tmp_path)
    rp.prepare_run_directory(spec)
    rp.prepare_run_directory(spec)  # 幂等，不该报错


def test_reprepare_with_changed_force_field_is_refused(tmp_path):
    """§7：禁止跨力场直接追加。"""
    rp.prepare_run_directory(_spec(tmp_path))
    changed = _spec(tmp_path, environment=rc.EnvironmentSpec(
        receptor_name="R", receptor_path="/tmp/r.pdb", receptor_sha256="c" * 64,
        force_field="charmm36", water_model="tip3p", ion_model="jc"))
    with pytest.raises(rp.RBFEResumeError, match="force_field"):
        rp.prepare_run_directory(changed)


def test_refuses_directory_without_identity(tmp_path):
    layout = rp.RunLayout(root=tmp_path, repeat_index=1)
    layout.manifest_path.parent.mkdir(parents=True)
    layout.manifest_path.write_text('{"edge_id": "A_to_B"}', encoding="utf-8")
    with pytest.raises(rp.RBFEResumeError, match="edge_identity"):
        rp.prepare_run_directory(_spec(tmp_path))


def test_refuses_unreadable_manifest(tmp_path):
    layout = rp.RunLayout(root=tmp_path, repeat_index=1)
    layout.manifest_path.parent.mkdir(parents=True)
    layout.manifest_path.write_text("{ truncated", encoding="utf-8")
    with pytest.raises(rp.RBFEResumeError):
        rp.prepare_run_directory(_spec(tmp_path))


def test_refusal_does_not_delete_existing_artifacts(tmp_path):
    """拒绝，不是清理——不自动删别人的产物。"""
    rp.prepare_run_directory(_spec(tmp_path))
    victim = tmp_path / "repeat_01" / "complex" / "precious.dcd"
    victim.write_bytes(b"data")
    changed = _spec(tmp_path, environment=rc.EnvironmentSpec(
        receptor_name="R", receptor_path="/tmp/r.pdb", receptor_sha256="c" * 64,
        force_field="charmm36", water_model="tip3p", ion_model="jc"))
    with pytest.raises(rp.RBFEResumeError):
        rp.prepare_run_directory(changed)
    assert victim.exists() and victim.read_bytes() == b"data"


def test_repeats_get_separate_directories(tmp_path):
    spec = _spec(tmp_path)
    l1 = rp.prepare_run_directory(spec, 1)
    l2 = rp.prepare_run_directory(spec, 2)
    assert l1.repeat_dir != l2.repeat_dir
    assert l1.repeat_dir.name == "repeat_01" and l2.repeat_dir.name == "repeat_02"


# ---------------------------------------------------------------------------
# 两腿编排
# ---------------------------------------------------------------------------


def test_combine_legs_writes_result(tmp_path):
    spec = _spec(tmp_path)
    layout = rp.prepare_run_directory(spec)
    result = rp.combine_legs(spec, _leg("complex", -10.0, 3.0), _leg("solvent", -2.0, 4.0), layout)
    assert result.ddg_bind == pytest.approx(-8.0)
    assert result.ddg_stderr == pytest.approx(5.0)
    payload = json.loads(layout.result_path.read_text(encoding="utf-8"))
    assert payload["ddG_bind_B_minus_A"] == pytest.approx(-8.0)
    assert payload["repeat_index"] == 1


def test_combine_legs_checks_legs_belong_to_this_spec():
    """两腿彼此自洽还不够——它们得确实是这条 spec 的。"""
    spec = _spec()
    with pytest.raises(rp.RBFEPipelineError, match="edge_id"):
        rp.combine_legs(spec, _leg("complex", -10.0, 1.0, edge_id="OTHER"),
                        _leg("solvent", -2.0, 1.0, edge_id="OTHER"))


def test_combine_legs_checks_ligand_identity():
    spec = _spec()
    with pytest.raises(rp.RBFEPipelineError, match="配体身份"):
        rp.combine_legs(spec, _leg("complex", -10.0, 1.0, a="X", b="Y"),
                        _leg("solvent", -2.0, 1.0, a="X", b="Y"))


def test_pipeline_does_not_compute_science_itself():
    """§4：编排层不自行构建相互作用公式。ΔΔG 只能来自 rbfe_core。"""
    from pathlib import Path

    source = Path(rp.__file__).read_text(encoding="utf-8")
    assert "combine_rbfe" in source
    assert "complex_result.delta_g -" not in source


# ---------------------------------------------------------------------------
# 独立重复
# ---------------------------------------------------------------------------


def test_between_repeat_sd_is_none_for_a_single_repeat():
    """🔑 n=1 时重复间散布**无法估计**，返回 None 而不是 0。

    0 会被读成「重复之间完全一致」——那是只跑了一次的运行绝对给不出的结论。
    """
    agg = rp.aggregate_repeats([_edge_result()])
    assert agg.between_repeat_sd is None
    assert not agg.qualified
    assert any("无法估计" in r for r in agg.qualification_reasons)


def test_repeat_aggregate_uses_sample_sd_and_stderr_of_mean():
    results = [_edge_result(dg_complex=-10.0 + d, se=0.0) for d in (-1.0, 0.0, 1.0)]
    agg = rp.aggregate_repeats(results)
    assert agg.mean_ddg == pytest.approx(-8.0)
    assert agg.between_repeat_sd == pytest.approx(1.0)          # 样本 sd，n-1
    assert agg.stderr_of_mean == pytest.approx(1.0 / math.sqrt(3))


def test_aggregate_keeps_both_uncertainty_estimates():
    """重复间散布与单次传播误差必须并排留着，否则读的人无从判断单次是否被低估。"""
    results = [_edge_result(dg_complex=-10.0 + d, se=0.05) for d in (-2.0, 0.0, 2.0)]
    agg = rp.aggregate_repeats(results)
    assert agg.between_repeat_sd > 10 * agg.mean_reported_stderr
    payload = agg.to_dict()
    assert "between_repeat_sd" in payload and "mean_reported_stderr" in payload


def test_aggregate_refuses_mixed_edges():
    with pytest.raises(rp.RBFEPipelineError, match="edge_id"):
        rp.aggregate_repeats([_edge_result(), _edge_result(edge_id="OTHER")])


def test_aggregate_propagates_failed_qualification():
    bad = rc.combine_rbfe(_leg("complex", -10.0, 1.0, ok=False), _leg("solvent", -2.0, 1.0))
    agg = rp.aggregate_repeats([_edge_result(), bad, _edge_result()])
    assert not agg.qualified
    assert any("repeat_02" in r for r in agg.qualification_reasons)


def test_aggregate_rejects_empty():
    with pytest.raises(rp.RBFEPipelineError):
        rp.aggregate_repeats([])


# ---------------------------------------------------------------------------
# 边网络
# ---------------------------------------------------------------------------


E = rp.NetworkEdge


def test_perfect_triangle_closes():
    report = rp.analyze_network([E("A", "B", 2.0, 0.5), E("B", "C", 3.0, 0.5), E("A", "C", 5.0, 0.5)])
    assert report.connected
    assert len(report.cycles) == 1
    assert report.cycles[0].residual == pytest.approx(0.0)
    assert report.all_cycles_close


def test_broken_triangle_fails_closure():
    report = rp.analyze_network([E("A", "B", 2.0, 0.5), E("B", "C", 3.0, 0.5), E("A", "C", 9.0, 0.5)])
    cycle = report.cycles[0]
    assert abs(cycle.residual) == pytest.approx(4.0)
    assert abs(cycle.z_score) > 2.0
    assert not cycle.passes()
    assert not report.all_cycles_close


def test_direction_matters_when_walking_a_cycle():
    """ΔΔG 是有向量：反向走必须取负号，否则完美闭合的环会被判成不闭合。"""
    report = rp.analyze_network([E("A", "B", 2.0, 0.1), E("C", "B", -3.0, 0.1), E("A", "C", 5.0, 0.1)])
    assert report.cycles[0].residual == pytest.approx(0.0)


def test_disconnected_network_is_reported():
    report = rp.analyze_network([E("A", "B", 1.0, 0.1), E("C", "D", 1.0, 0.1)])
    assert not report.connected
    assert len(report.components) == 2


def test_duplicate_edge_rejected():
    with pytest.raises(rp.RBFEPipelineError, match="重复边"):
        rp.analyze_network([E("A", "B", 1.0, 0.1), E("B", "A", -1.0, 0.1)])


def test_self_loop_rejected():
    with pytest.raises(rp.RBFEPipelineError, match="自环"):
        E("A", "A", 0.0, 0.1)


def test_mixed_units_rejected():
    with pytest.raises(rp.RBFEPipelineError, match="单位"):
        rp.analyze_network([E("A", "B", 1.0, 0.1),
                            E("B", "C", 1.0, 0.1, energy_unit=rc.KCAL_PER_MOL)])


def test_zero_uncertainty_cycle_is_not_faked_as_passing():
    """误差为 0 时做不了 z 检验。残差非 0 就是不通过，不能因为"判不了"而放行。"""
    report = rp.analyze_network([E("A", "B", 2.0, 0.0), E("B", "C", 3.0, 0.0), E("A", "C", 9.0, 0.0)])
    assert not report.cycles[0].passes()


def test_report_carries_the_closure_caveat():
    """§7：闭合通过不是所有系统误差均消失的证明——这句必须跟着结果走。"""
    report = rp.analyze_network([E("A", "B", 2.0, 0.5), E("B", "C", 3.0, 0.5), E("A", "C", 5.0, 0.5)])
    assert "系统误差" in report.to_dict()["caveat"]


# ---------------------------------------------------------------------------
# ABFE 锚点
# ---------------------------------------------------------------------------


_TRIANGLE = [E("A", "B", 2.0, 0.5), E("B", "C", 3.0, 0.5), E("A", "C", 5.0, 0.5)]


def _anchor(**kw):
    kw.setdefault("comparability_statement", "同受体状态/化学身份/力场/温度/自由能定义")
    return rp.AbsoluteAnchor(
        ligand=kw.pop("ligand", "A"), delta_g_bind=kw.pop("delta_g_bind", -40.0),
        stderr=kw.pop("stderr", 1.0), energy_unit=rc.KJ_PER_MOL,
        source=kw.pop("source", "ABFE"), **kw,
    )


def test_anchor_requires_an_explicit_comparability_statement():
    """§7：只有受体状态、化学身份、力场、温度和自由能定义一致时才可换算。

    这个前提机器验不了，所以要求调用方**显式声明**——没有声明就拒绝，
    而不是默默换算。
    """
    with pytest.raises(rp.RBFEPipelineError, match="comparability_statement"):
        rp.absolute_from_anchor(_TRIANGLE, rp.AbsoluteAnchor(
            "A", -40.0, 1.0, rc.KJ_PER_MOL, "ABFE"))


def test_absolute_values_follow_the_edges():
    out = rp.absolute_from_anchor(_TRIANGLE, _anchor())
    assert out["absolute"]["A"]["delta_g_bind"] == pytest.approx(-40.0)
    assert out["absolute"]["B"]["delta_g_bind"] == pytest.approx(-38.0)
    assert out["absolute"]["C"]["delta_g_bind"] == pytest.approx(-35.0)


def test_anchor_error_is_always_included_and_grows_with_distance():
    """离锚点越远误差越大，这一点必须在数字上看得见。"""
    out = rp.absolute_from_anchor(_TRIANGLE, _anchor(stderr=1.0))
    a, b = out["absolute"]["A"], out["absolute"]["B"]
    assert a["stderr"] == pytest.approx(1.0)                       # 锚点自身误差
    assert b["stderr"] == pytest.approx(math.sqrt(1.0 + 0.25))     # 锚点 ⊕ 一条边
    assert b["stderr"] > a["stderr"]
    assert b["hops_from_anchor"] == 1


def test_unreachable_ligands_are_reported_not_silently_dropped():
    edges = _TRIANGLE + [E("X", "Y", 1.0, 0.1)]
    out = rp.absolute_from_anchor(edges, _anchor())
    assert set(out["unreachable_from_anchor"]) == {"X", "Y"}


def test_anchor_not_in_network_rejected():
    with pytest.raises(rp.RBFEPipelineError, match="不在网络里"):
        rp.absolute_from_anchor(_TRIANGLE, _anchor(ligand="Z"))


def test_anchor_unit_mismatch_rejected():
    bad = rp.AbsoluteAnchor("A", -10.0, 1.0, rc.KCAL_PER_MOL, "exp",
                            comparability_statement="ok")
    with pytest.raises(rp.RBFEPipelineError, match="单位"):
        rp.absolute_from_anchor(_TRIANGLE, bad)


def test_absolute_output_carries_the_precision_caveat():
    out = rp.absolute_from_anchor(_TRIANGLE, _anchor())
    assert "不可当作等精度" in out["caveat"]


# ---------------------------------------------------------------------------
# 依赖方向与未实现边界
# ---------------------------------------------------------------------------


def test_pipeline_never_imports_abfe():
    """ABFE 与 RBFE 是两条独立入口（计划 §1）。"""
    from pathlib import Path

    source = Path(rp.__file__).read_text(encoding="utf-8")
    lines = [l.strip() for l in source.splitlines()
             if l.lstrip().startswith(("import ", "from "))]
    for banned in ("abfe_core", "abfe_pipeline", "ibs_engine", "runabfe"):
        assert not [l for l in lines if banned in l], f"不得 import {banned}"


def test_run_leg_fails_closed_because_r1_is_missing():
    """R1 没有 hybrid builder，就没有可采样的 Hamiltonian——必须抛错，不能假装跑完。"""
    with pytest.raises(NotImplementedError):
        rp.run_leg(_spec(), "complex", rp.RunLayout(root=__import__("pathlib").Path("/tmp"), repeat_index=1))
