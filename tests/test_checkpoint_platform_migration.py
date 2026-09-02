"""跨 platform 的 checkpoint 必须能迁移，损坏/System 不符的必须仍被拒。

背景（4W53，2026-09-01）：`checkpoints/pre_equil.chk` 文件头是 `CPUB`，OpenMM 报
`loadCheckpoint: Checkpoint was created with a different Platform: CPU`。成因是那次
运行开始时 CUDA 没被检测到、被强制降级到 CPU 跑完了预平衡（该误降级同日已修）。
现在 CUDA 认得出来，于是这份**完好**的 checkpoint 载不进 CUDA context，
`equilibrium_is_done` 判成"没做完"，要重跑 5,000,000 步预平衡。

「别的 platform 写的」和「坏了」此前被同一个 `except Exception: return False`
吞成一件事。这组测试钉住拆分后的行为。
"""
import openmm
from openmm import app, unit
import pytest

import abfe_pipeline as ap

pytestmark = pytest.mark.cpu_only


def _system():
    sysm = openmm.System()
    for _ in range(3):
        sysm.addParticle(12.0 * unit.dalton)
    f = openmm.CustomExternalForce("k_probe*(x*x+y*y+z*z)")
    f.addGlobalParameter("k_probe", 1.0)
    for i in range(3):
        f.addParticle(i, [])
    sysm.addForce(f)
    sysm.setDefaultPeriodicBoxVectors(*(openmm.Vec3(3, 0, 0), openmm.Vec3(0, 3, 0), openmm.Vec3(0, 0, 3)))
    return sysm


def _topology():
    top = app.Topology()
    chain = top.addChain()
    res = top.addResidue("PRB", chain)
    for i in range(3):
        top.addAtom(f"C{i}", app.element.carbon, res)
    return top


def _sim(platform_name, *, with_param=True):
    sysm = _system()
    if not with_param:
        sysm.removeForce(0)
    integ = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond
    )
    return app.Simulation(
        _topology(), sysm, integ, openmm.Platform.getPlatformByName(platform_name)
    )


@pytest.fixture()
def reference_checkpoint(tmp_path):
    """用 Reference platform 写一份 checkpoint（跑过几步，带非平凡状态）。"""
    sim = _sim("Reference")
    sim.context.setPositions([openmm.Vec3(0.1, 0.2, 0.3)] * 3)
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin, 12345)
    sim.context.setParameter("k_probe", 2.5)
    sim.step(7)
    path = tmp_path / "probe.chk"
    with open(path, "wb") as handle:
        sim.saveCheckpoint(handle)
    return str(path), sim


def test_same_platform_load_reports_native(reference_checkpoint):
    path, _ = reference_checkpoint
    assert ap.load_checkpoint_with_platform_migration(_sim("Reference"), path) == "native"


def test_cross_platform_load_migrates_full_physical_state(reference_checkpoint):
    """坐标/速度/盒子/全局参数/步数/时间 全部要搬过去。"""
    path, source = reference_checkpoint
    target = _sim("CPU")
    assert ap.load_checkpoint_with_platform_migration(target, path) == "migrated:Reference"

    a = target.context.getState(getPositions=True, getVelocities=True, getParameters=True)
    b = source.context.getState(getPositions=True, getVelocities=True, getParameters=True)
    pa = a.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    pb = b.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    va = a.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
    vb = b.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
    assert abs(pa - pb).max() < 1e-6
    assert abs(va - vb).max() < 1e-6
    # 全局参数不是可有可无的装饰：λ_vdw / Boresch scale 都走这条路
    assert dict(a.getParameters())["k_probe"] == pytest.approx(2.5)
    # 步数决定"还剩多少步"
    assert target.context.getStepCount() == source.context.getStepCount() == 7
    assert target.currentStep == 7


def test_missing_global_parameter_fails_closed_instead_of_dropping_it(reference_checkpoint):
    """目标 System 少一个全局参数时必须报错 —— 静默丢 λ 是正确性 bug。"""
    path, _ = reference_checkpoint
    with pytest.raises(RuntimeError, match="k_probe"):
        ap.load_checkpoint_with_platform_migration(_sim("CPU", with_param=False), path)


def test_corrupt_checkpoint_still_raises(reference_checkpoint, tmp_path):
    path, _ = reference_checkpoint
    bad = tmp_path / "bad.chk"
    data = bytearray(open(path, "rb").read())
    data[40:200] = b"\x00" * 160
    bad.write_bytes(bytes(data))
    with pytest.raises(Exception):
        ap.load_checkpoint_with_platform_migration(_sim("CPU"), str(bad))


def test_is_checkpoint_valid_accepts_cross_platform_but_rejects_corrupt(
    reference_checkpoint, tmp_path
):
    path, _ = reference_checkpoint
    assert ap._is_checkpoint_valid(path, simulation=_sim("CPU")) is True
    assert ap._is_checkpoint_valid(path, simulation=_sim("CPU", with_param=False)) is False
    bad = tmp_path / "bad2.chk"
    data = bytearray(open(path, "rb").read())
    data[40:200] = b"\x00" * 160
    bad.write_bytes(bytes(data))
    assert ap._is_checkpoint_valid(str(bad), simulation=_sim("CPU")) is False


def test_is_checkpoint_valid_stays_fail_closed_without_a_simulation(reference_checkpoint):
    """只给裸 load_checkpoint 函数时无法迁移，必须维持原来的 fail-closed。"""
    path, _ = reference_checkpoint

    def _loader(_p):
        raise openmm.OpenMMException(
            "loadCheckpoint: Checkpoint was created with a different Platform: Reference"
        )

    assert ap._is_checkpoint_valid(path, load_checkpoint=_loader) is False


def test_non_platform_errors_are_not_swallowed_as_migration():
    """只有 platform 不匹配才走迁移；别的异常原样抛。"""
    assert ap._checkpoint_source_platform(RuntimeError("disk is on fire")) is None
    assert ap._checkpoint_source_platform(
        openmm.OpenMMException(
            "loadCheckpoint: Checkpoint was created with a different Platform: CUDA"
        )
    ) == "CUDA"


def test_window_level_production_resume_is_deliberately_excluded():
    """window 级 production 续算**不得**走跨平台迁移，必须保持 fail-closed。

    [abfe-ibs-d5 复核，2026-09-01] 那条路径把「loadCheckpoint 成功」本身当作
    「这段轨迹没有被打断」的判据（`ibs_engine.py` 附近注释明写 "only a successful
    ``loadCheckpoint`` here counts as a true, uninterrupted ..."，且续算契约**含
    积分器状态**）。State 迁移搬不走积分器内部状态，一旦在那里启用，就会在积分器
    状态其实已经断掉的情况下报告"未打断" —— 那是字段说谎，比重跑一个窗口糟得多。

    预平衡/再平衡是另一回事：那是消费一段**已完成**片段的末态，物理状态搬全了就够。
    """
    import pathlib

    repo = pathlib.Path(__file__).resolve().parent.parent
    engine = (repo / "ibs_engine.py").read_text(encoding="utf-8")
    assert "load_checkpoint_with_platform_migration" not in engine

    pipeline = (repo / "abfe_pipeline.py").read_text(encoding="utf-8")
    # 只允许出现在：1 处定义 + `_is_checkpoint_valid` 的校验 + 预平衡 resume +
    # 再平衡 resume。这三处消费的都是"已完成片段的末态"。多出任何一处都要先回答
    # "那条路径是不是把积分器状态当契约"。
    assert pipeline.count("load_checkpoint_with_platform_migration(") == 4


def test_migration_documents_what_it_cannot_carry():
    """搬不走的东西必须写在代码里，不能只活在某个人的脑子里。"""
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parent.parent / "abfe_pipeline.py"
    ).read_text(encoding="utf-8")
    i = src.index("def load_checkpoint_with_platform_migration")
    head = src[max(0, i - 3000) : i]
    assert "随机数流" in head          # Langevin RNG
    assert "逐位可复现" in head        # 代价说明
