"""2026-09-23：生产盒 = NPT 均值盒；盒体积变了的 stage-2 窗口缓存不能复用。

cyclod_ligand1 实测：溶剂腿用 NPT 末帧盒（rep 间差 0.9% ⟹ Δs2 差 1.79 kJ）；
复合物腿 resume 退回输入盒（323.17 vs NPT 308.0）。
"""
import types

import numpy as np
import pytest

pytest.importorskip("openmm")

import abfe_pipeline as ap
import test_resume_reuse_contracts as t
from test_resume_reuse_contracts import _gate, _matching_conv


def _fake_pipeline(tmp_path, vols):
    (tmp_path / "pre_equilibration_convergence_monitor.csv").write_text(
        "step,volume_nm3\n" + "".join(f"{i},{v}\n" for i, v in enumerate(vols)),
        encoding="utf-8",
    )
    return types.SimpleNamespace(
        output_dir=str(tmp_path), environment_type="soluble", _log=lambda *_: None
    )


def test_box_rescaled_to_second_half_mean_volume_rigid_molecules(tmp_path):
    # 前半段在弛豫（体积大），后半段均值 50.0；末帧 50.385（rep2 溶剂腿的实测值）。
    fake = _fake_pipeline(tmp_path, [60.0, 58.0, 49.9, 50.1, 50.0, 49.95, 50.05, 50.0])
    box = np.eye(3) * 50.385 ** (1 / 3)
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, box[0, 0], size=(9, 3))
    mols = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    new_pos, new_box = ap.ABFEPipeline._rescale_to_npt_mean_volume(fake, pos, box, mols)

    assert abs(np.linalg.det(new_box)) == pytest.approx(50.0, rel=1e-12)
    for m in mols:  # 分子内几何不变，只移动质心（与 MC barostat 一致）
        assert np.allclose(new_pos[m] - new_pos[m].mean(0), pos[m] - pos[m].mean(0))
    s = (50.0 / 50.385) ** (1 / 3)
    assert np.allclose(new_pos[mols[1]].mean(0), s * pos[mols[1]].mean(0))


def test_box_unchanged_without_enough_blocks(tmp_path):
    fake = _fake_pipeline(tmp_path, [50.0, 50.1, 49.9])
    box = np.eye(3) * 3.7
    pos = np.zeros((3, 3))
    new_pos, new_box = ap.ABFEPipeline._rescale_to_npt_mean_volume(fake, pos, box, [[0, 1, 2]])
    assert new_box is box and new_pos is pos


def _gate_with_box(conv, current_box_volume_nm3):
    import ibs_engine as ie

    return ie._resume_cached_window_gate_status(
        conv, t.GOOD_SHAPE, t.LC_WIN, t.LV_WIN, t.REPAIR_POLICY, t.LSE_TOL, False,
        dict(t.EARLY_STOP_CONFIG), t.TARGET_STEPS,
        current_box_volume_nm3=current_box_volume_nm3,
    )


def test_window_cache_rejected_when_box_volume_changed():
    assert _gate(_matching_conv())["usable"] is True  # 基线：不传当前盒 ⟹ 不设门

    stale = _matching_conv(lj_tail_lrc_box_volume_nm3=323.16780367462405)
    status = _gate_with_box(stale, 308.0)
    assert status["usable"] is False and status["box_volume_match"] is False
    assert "盒体积不符" in status["reason"]

    assert _gate_with_box(stale, 323.16780367462405)["usable"] is True
    # 缺字段（旧缓存）+ 当前盒已知 ⟹ 不可复用（本函数"缺字段一律保守"的惯例）
    assert _gate_with_box(_matching_conv(), 308.0)["box_volume_match"] is False


def test_adopt_reads_back_checkpoint_state_on_real_openmm_water_box(tmp_path):
    """端到端（CPU）：旧 run 目录只有 pre_equil.chk 时，读回的是 checkpoint 里的
    NPT 盒/坐标（再缩到均值体积），而不是输入盒——复合物腿 09-11 那个 bug 的正面。"""
    import functools
    import json

    import openmm
    from openmm import app, unit

    from abfe_core import resolve_membrane_protocol

    modeller = app.Modeller(app.Topology(), [])
    ff = app.ForceField("amber14/tip3p.xml")
    modeller.addSolvent(ff, boxSize=openmm.Vec3(2.0, 2.0, 2.0) * unit.nanometer)
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=0.8 * unit.nanometer, constraints=app.HBonds)
    input_box = modeller.topology.getPeriodicBoxVectors()

    chk_dir = tmp_path / "checkpoints"
    chk_dir.mkdir()
    # 造一份"预平衡 checkpoint"：盒子故意改成与输入不同，才能区分读回的是谁。
    npt_box = [v * 0.97 for v in input_box]
    work = openmm.XmlSerializer.clone(system)
    work.addForce(openmm.MonteCarloBarostat(1.0 * unit.bar, 300 * unit.kelvin, 25))
    sim = app.Simulation(modeller.topology, work,
                         openmm.LangevinMiddleIntegrator(300, 1, 0.002),
                         openmm.Platform.getPlatformByName("CPU"))
    sim.context.setPeriodicBoxVectors(*npt_box)
    sim.context.setPositions([p * 0.97 for p in modeller.positions.value_in_unit(unit.nanometer)] * unit.nanometer)
    sim.saveCheckpoint(str(chk_dir / "pre_equil.chk"))
    chk_volume = 0.97 ** 3 * float(np.linalg.det(np.asarray(input_box.value_in_unit(unit.nanometer))))
    del sim

    (tmp_path / "pre_equilibration_fingerprint.json").write_text(
        json.dumps({"identity_fingerprint": "ID"}), encoding="utf-8")
    # NPT 后半段均值 = checkpoint 体积 × 1.01
    vols = [chk_volume * 1.2] * 4 + [chk_volume * 1.01] * 4
    (tmp_path / "pre_equilibration_convergence_monitor.csv").write_text(
        "step,volume_nm3\n" + "".join(f"{i},{v}\n" for i, v in enumerate(vols)), encoding="utf-8")

    fake = types.SimpleNamespace(
        system=system, topology=modeller.topology,
        positions=modeller.positions, box_vectors=input_box,
        barostat_protocol=resolve_membrane_protocol("soluble", None),
        temperature=300 * unit.kelvin, pressure=1.0 * unit.bar, platform_name="CPU",
        checkpoint_dir=str(chk_dir), output_dir=str(tmp_path), environment_type="soluble",
        PRE_EQUILIBRATION_FINAL_STATE_FILENAME=ap.ABFEPipeline.PRE_EQUILIBRATION_FINAL_STATE_FILENAME,
        pre_equilibration_identity_fingerprint=lambda n: "ID", _log=lambda *_: None,
    )
    for name in ("_rescale_to_npt_mean_volume", "_set_pre_equilibrated_state"):
        setattr(fake, name, functools.partial(getattr(ap.ABFEPipeline, name), fake))

    assert ap.ABFEPipeline.adopt_pre_equilibrated_state(fake) is True
    got = abs(np.linalg.det(np.asarray(
        [v.value_in_unit(unit.nanometer) for v in fake.box_vectors])))
    assert got == pytest.approx(chk_volume * 1.01, rel=1e-9)
    assert fake._positions_from_pre_equilibration is True
    assert len(fake.positions) == system.getNumParticles()

    # 身份不符 ⟹ 不读回
    fake.pre_equilibration_identity_fingerprint = lambda n: "OTHER"
    assert ap.ABFEPipeline.adopt_pre_equilibrated_state(fake) is False


def test_main_window_checkpoint_manifest_binds_box_volume(tmp_path):
    """BOX-02：修复前写下的主窗口 checkpoint（manifest 无盒体积 / 盒不同）不得被续验恢复。"""
    import json

    import ibs_engine as ie

    kw = dict(stage_type="vdw", window_idx=0, K=2, win_sys_xml="<System/>",
              lambdas_coul=[1.0, 1.0], lambdas_vdw=[1.0, 0.9], lambda_shield=None,
              temperature_K=300.0, platform_name="CPU")
    old = ie._build_main_window_checkpoint_manifest(**kw)  # 修复前：无 box_volume_nm3
    new = ie._build_main_window_checkpoint_manifest(**kw, box_volume_nm3=308.0)
    assert new["box_volume_nm3"] == 308.0 and "box_volume_nm3" not in old

    ckpt, manifest = ie._main_window_checkpoint_paths(str(tmp_path), "vdw", 0)
    import os
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    os.makedirs(os.path.dirname(manifest), exist_ok=True)
    open(ckpt, "wb").write(b"x")
    for written, ok in ((old, False),
                        (ie._build_main_window_checkpoint_manifest(**kw, box_volume_nm3=323.1678), False),
                        (new, True)):
        open(manifest, "w", encoding="utf-8").write(json.dumps(written))
        assert ie._main_window_checkpoint_is_usable(str(tmp_path), "vdw", 0, new) is ok


def test_stage2_forwards_every_config_key_the_stage_reads():
    """REPRO-01 判据 3：`_run_stage2_once` 是逐个具名转发，漏一个键 ⟹ config 开关恒取默认值
    （`stage2_independent_endpoint` / `stage2_frozen_layout` /
    `stage2_accept_recalibrated_f_k_without_gate` 三次同一形状）。"""
    import ast
    from pathlib import Path

    repo = Path(ap.__file__).resolve().parent
    tree = ast.parse((repo / "abfe_pipeline.py").read_text(encoding="utf-8"))
    fn = lambda name: next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)

    def kwargs_keys(node):
        return {
            n.args[0].value for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr in ("get", "pop") and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "kwargs" and n.args and isinstance(n.args[0], ast.Constant)
        }

    read = kwargs_keys(fn("_run_dual_lambda_stage"))
    # runabfe 真正会往 run_full_pipeline kwargs 里放的键：`out["..."] = ...`
    rtree = ast.parse((repo / "runabfe.py").read_text(encoding="utf-8"))
    set_by_config = {
        t.slice.value for n in ast.walk(rtree) if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
        and t.value.id == "out" and isinstance(t.slice, ast.Constant)
    }
    call = next(
        n for n in ast.walk(fn("_run_stage2_once"))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_run_dual_lambda_stage"
    )
    forwarded = {k.arg for k in call.keywords}
    assert (read & set_by_config) - forwarded == set()
