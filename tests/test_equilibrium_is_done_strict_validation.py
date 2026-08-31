"""Pre-equilibration completion requires a parsed DCD and a loadable checkpoint.

A missing/unavailable target Simulation fails closed. These cases cover corrupt
DCDs, garbage checkpoints, absent probes, incomplete metadata and a real valid
OpenMM checkpoint; file size alone must never authorize skipping equilibration.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

openmm = pytest.importorskip("openmm")
from openmm import app, unit  # noqa: E402

import runabfe  # noqa: E402
from abfe_pipeline import _is_traj_valid  # noqa: E402


def _write_fingerprint_and_state(output_dir: Path, *, completed: bool, n_steps: int = 1000):
    """把弱判据需要的一切元数据都摆好：指纹一致 + 状态 completed + 步数达标。

    这样测试失败只可能来自严格校验本身，而不是元数据缺失的保守分支。
    """
    chk_dir = output_dir / "checkpoints"
    chk_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "pre_equilibration_fingerprint.json", "w", encoding="utf-8") as fh:
        json.dump({"fingerprint": "fp-test", "n_steps": n_steps}, fh)
    with open(chk_dir / "pipeline_state.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "stages": {
                    "equilibration": {
                        "status": "completed" if completed else "running",
                        "total_steps": n_steps,
                    }
                }
            },
            fh,
        )


def _make_dcd(path: Path, *, truncate: bool):
    """写一条 300 帧的极小 DCD（>10 KB，能通过弱短路阈值）；truncate=True 时
    把文件尾部砍掉 16 字节 —— 无论 parser 按什么帧边界读都必须读到截断。"""
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("LIG", chain)
    topology.addAtom("C1", app.element.carbon, residue)
    system = openmm.System()
    system.addParticle(12.011 * unit.dalton)
    box_vec = (openmm.Vec3(3.0, 0.0, 0.0), openmm.Vec3(0.0, 3.0, 0.0), openmm.Vec3(0.0, 0.0, 3.0))
    system.setDefaultPeriodicBoxVectors(*[v * unit.nanometer for v in box_vec])
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions([[0.5, 0.5, 0.5] * unit.nanometer])
    reporter = app.DCDReporter(str(path), 1, enforcePeriodicBox=False)
    simulation.reporters.append(reporter)
    simulation.step(300)
    del reporter, simulation, integrator  # 关闭文件句柄
    if truncate:
        size = path.stat().st_size
        # 帧记录是 3*N*(8) + 4 字节（double 坐标 + 记录长度标记），砍掉整条
        # 末帧再加一半 —— 无论 parser 按什么边界读都必须失败。
        with open(path, "r+b") as fh:
            fh.truncate(size - 16)
    return path


def _prepare_case(tmp_path: Path, *, truncate: bool, random_chk: bool, completed: bool = True):
    output_dir = tmp_path / ("case_" + "".join(str(b) for b in (truncate, random_chk, completed)))
    output_dir.mkdir(parents=True)
    _write_fingerprint_and_state(output_dir, completed=completed)
    _make_dcd(output_dir / "pre_equilibration.dcd", truncate=truncate)
    chk = output_dir / "checkpoints" / "pre_equil.chk"
    if random_chk:
        chk.write_bytes(os.urandom(512))
    else:
        chk.write_bytes(os.urandom(512))
    return output_dir


def test_truncated_dcd_is_not_done(tmp_path):
    """截断末帧的 DCD ⟹ 未完成，即使指纹/完成标记/大小阈值全过。"""
    output_dir = _prepare_case(tmp_path, truncate=True, random_chk=False)
    size = (output_dir / "pre_equilibration.dcd").stat().st_size
    assert size > 10000  # 弱判据的大小阈值不是本次被测对象
    runabfe._TRAJ_STRICT_VALIDITY_CACHE.clear()  # 确保走真实 parser
    assert not runabfe.equilibrium_is_done(str(output_dir), expected_fingerprint="fp-test")


def test_random_bytes_checkpoint_with_valid_dcd_is_not_done_when_target_given(tmp_path):
    """随机字节的 checkpoint 在有目标 Simulation 时 ⟹ 未完成（真实 loadCheckpoint 失败）。"""
    output_dir = _prepare_case(tmp_path, truncate=False, random_chk=True)
    # 先确认 DCD 本身是完好的（否则测试没测到 checkpoint 这一环）
    assert _is_traj_valid(str(output_dir / "pre_equilibration.dcd"), min_frames=1)

    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("LIG", chain)
    topology.addAtom("C1", app.element.carbon, residue)
    system = openmm.System()
    system.addParticle(12.011 * unit.dalton)
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    simulation = app.Simulation(topology, system, integrator)

    assert not runabfe.equilibrium_is_done(
        str(output_dir), expected_fingerprint="fp-test", simulation=simulation
    )


def test_valid_complete_case_is_done(tmp_path):
    """完整 DCD + 可加载的真实 checkpoint + 全部元数据 ⟹ 完成。"""
    output_dir = _prepare_case(tmp_path, truncate=False, random_chk=False)
    topology = app.Topology()
    residue = topology.addResidue("LIG", topology.addChain())
    topology.addAtom("C1", app.element.carbon, residue)
    system = openmm.System()
    system.addParticle(12.011)
    simulation = app.Simulation(
        topology, system, openmm.VerletIntegrator(0.001),
        openmm.Platform.getPlatformByName("Reference"),
    )
    simulation.context.setPositions([[0.5, 0.5, 0.5]])
    simulation.saveCheckpoint(str(output_dir / "checkpoints" / "pre_equil.chk"))
    assert runabfe.equilibrium_is_done(
        str(output_dir), expected_fingerprint="fp-test", simulation=simulation
    )


def test_missing_probe_cannot_fall_back_to_nonempty_checkpoint(tmp_path):
    output_dir = _prepare_case(tmp_path, truncate=False, random_chk=True)
    assert not runabfe.equilibrium_is_done(str(output_dir), expected_fingerprint="fp-test")


def test_not_completed_state_is_not_done(tmp_path):
    """严格校验之外的既有守门（status != completed）继续生效。"""
    output_dir = _prepare_case(tmp_path, truncate=False, random_chk=False, completed=False)
    assert not runabfe.equilibrium_is_done(str(output_dir), expected_fingerprint="fp-test")


def test_real_dcd_parser_is_the_authority_not_size(tmp_path):
    """一条"大于阈值"但 mdtraj 拒读的垃圾文件 ⟹ 未完成。"""
    output_dir = tmp_path / "garbage"
    output_dir.mkdir()
    _write_fingerprint_and_state(output_dir, completed=True)
    garbage = output_dir / "pre_equilibration.dcd"
    garbage.write_bytes(np.random.default_rng(0).integers(0, 256, 64 * 1024, dtype=np.uint8).tobytes())
    (output_dir / "checkpoints" / "pre_equil.chk").write_bytes(os.urandom(512))
    assert not runabfe.equilibrium_is_done(str(output_dir), expected_fingerprint="fp-test")
