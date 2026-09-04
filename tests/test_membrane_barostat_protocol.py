"""B1：`system_type=membrane` + 膜恒压器（docs/status/memtodolist.md §3.1 / §3.2 / §7.4 / §7.7）。

对应清单条目：MEM-00i（"没有任何 MonteCarloMembraneBarostat"）、§3.1 膜体系识别、
§3.2 膜恒压器、§7.4 barostat 测试组、§7.7 原有体系回归。

## 本文件守的三件事

1. **纯增量**：不声明 `system_type` 时，预平衡 fingerprint 与本次改动前**逐位相同**，
   加的仍是 `MonteCarloBarostat(pressure, temperature, 25)`。已有生产预平衡
   checkpoint 不会因为这次改动失效（§7.7）。
2. **fail closed 而不是叠加**：输入 System 已带不兼容 barostat 时报错，绝不再加一个。
   旧代码只用 `isinstance(f, openmm.MonteCarloBarostat)` 判断，而
   `MonteCarloMembraneBarostat` / `MonteCarloAnisotropicBarostat` **不是**它的子类，
   所以旧逻辑会漏检并叠加出两个同时生效的 barostat。
3. **两个 `system_type` 不能混**：本仓库里 `system_type` 同时被
   环境类型（soluble/membrane）与腿身份（complex/solvent）使用，
   把腿身份传进膜协议解析必须报错。

全部 CPU 可跑：只构建 System/Topology 与 Force 对象，不创建 Context、不跑动力学。
"""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

openmm = pytest.importorskip("openmm")
from openmm import app, unit

import abfe_core as core

ROOT = Path(__file__).absolute().parents[1]


# ---------------------------------------------------------------------------
# 构造器
# ---------------------------------------------------------------------------


def _topology(residue_names):
    """按残基名列表建一个每残基一原子的最小 Topology。"""
    topology = app.Topology()
    chain = topology.addChain()
    for name in residue_names:
        residue = topology.addResidue(name, chain)
        topology.addAtom("C", app.element.carbon, residue)
    return topology


def _soluble_topology():
    """与生产体系同口径的残基名：水 / 氨基酸 / 离子 / 配体，无任何脂质。"""
    return _topology(
        ["SOL"] * 20 + ["LEU", "ILE", "VAL", "ALA", "ASH"] + ["CL", "NA"] + ["MOL"]
    )


def _membrane_topology(n_lipids=64):
    return _topology(["POPC"] * n_lipids + ["SOL"] * 20 + ["ALA"] * 5 + ["MOL"])


def _empty_system(n_particles=1):
    system = openmm.System()
    for _ in range(n_particles):
        system.addParticle(12.0)
    return system


# ---------------------------------------------------------------------------
# §3.1 体系识别
# ---------------------------------------------------------------------------


def test_environment_type_defaults_to_soluble_when_not_declared():
    assert core.resolve_environment_type(None) == core.ENVIRONMENT_TYPE_SOLUBLE
    assert core.resolve_environment_type("") == core.ENVIRONMENT_TYPE_SOLUBLE
    assert core.resolve_environment_type("MEMBRANE") == core.ENVIRONMENT_TYPE_MEMBRANE


def test_misspelled_environment_type_fails_instead_of_falling_back():
    with pytest.raises(ValueError, match="不是合法的环境类型"):
        core.resolve_environment_type("membrame")


@pytest.mark.parametrize("leg_identity", ["complex", "solvent"])
def test_leg_identity_passed_as_environment_type_is_rejected(leg_identity):
    """`system_type` 在本仓库被两个轴共用，混用必须响亮地失败。"""
    with pytest.raises(ValueError, match="腿身份"):
        core.resolve_environment_type(leg_identity)


def test_membrane_without_lipid_residues_fails_closed():
    with pytest.raises(ValueError, match="找不到任何已知脂质残基名"):
        core.resolve_membrane_protocol("membrane", topology=_soluble_topology())


def test_soluble_with_many_lipids_requires_explicit_confirmation():
    top = _membrane_topology(n_lipids=64)
    with pytest.raises(ValueError, match="显式传 confirm_soluble_with_lipids"):
        core.resolve_membrane_protocol("soluble", topology=top)

    protocol = core.resolve_membrane_protocol(
        "soluble", topology=top, confirm_soluble_with_lipids=True
    )
    assert protocol["system_type"] == "soluble"
    assert protocol["soluble_with_lipids_confirmed"] is True
    assert protocol["lipid_residue_total_unambiguous"] == 64


def test_production_soluble_topology_is_not_flagged_as_lipidic():
    """实测生产体系残基名与两套脂质集合都无交集——本节新增检查零误伤。"""
    protocol = core.resolve_membrane_protocol("soluble", topology=_soluble_topology())
    assert protocol["lipid_residue_total"] == 0
    assert protocol["lipid_residue_total_unambiguous"] == 0
    assert protocol["barostat_class"] == "MonteCarloBarostat"


def test_ambiguous_amber_modular_names_do_not_block_a_soluble_run():
    """`PC`/`OL`/`ST` 这类短 token 只进宽集合，不参与"拦下 soluble"的判据。

    否则一个配体或非标准残基恰好叫 `PC`，就会挡住一个完全合法的可溶体系运行。
    """
    top = _topology(["PC"] * 40 + ["OL"] * 40 + ["SOL"] * 10)
    protocol = core.resolve_membrane_protocol("soluble", topology=top)
    assert protocol["lipid_residue_total"] == 80          # 宽集合认得
    assert protocol["lipid_residue_total_unambiguous"] == 0  # 窄集合不认，因此不拦人
    # 反方向：同一份拓扑声明成 membrane 时，宽集合让它通过（不 fail closed）。
    membrane_protocol = core.resolve_membrane_protocol("membrane", topology=top)
    assert membrane_protocol["barostat_class"] == "MonteCarloMembraneBarostat"


def test_soluble_with_membrane_config_is_rejected():
    with pytest.raises(ValueError, match="却提供了 membrane"):
        core.resolve_membrane_protocol(
            "soluble",
            membrane_config={"xy_mode": "isotropic"},
            topology=_soluble_topology(),
        )


def test_unknown_membrane_field_is_rejected():
    with pytest.raises(ValueError, match="未知字段"):
        core.resolve_membrane_protocol(
            "membrane",
            membrane_config={"surfacetension": 0.0},
            topology=_membrane_topology(),
        )


def test_non_z_membrane_normal_axis_fails_closed_because_openmm_hardcodes_z():
    with pytest.raises(ValueError, match="硬编码为 z"):
        core.resolve_membrane_protocol(
            "membrane",
            membrane_config={"normal_axis": "x"},
            topology=_membrane_topology(),
        )


@pytest.mark.parametrize(
    "bad_config, pattern",
    [
        ({"xy_mode": "diagonal"}, "xy_mode"),
        ({"z_mode": "clamped"}, "z_mode"),
        ({"surface_tension_bar_nm": "zero"}, "不是数值"),
        ({"barostat_frequency": 0}, "必须为正整数"),
        ({"barostat_frequency": "many"}, "不是整数"),
    ],
)
def test_invalid_membrane_values_fail_closed(bad_config, pattern):
    with pytest.raises(ValueError, match=pattern):
        core.resolve_membrane_protocol(
            "membrane", membrane_config=bad_config, topology=_membrane_topology()
        )


def test_membrane_protocol_defaults_match_the_frozen_plan():
    """§3.1 的配置样例：法向 z、表面张力 0、XY 等比例、Z 独立、频率 25。"""
    protocol = core.resolve_membrane_protocol(
        "membrane", topology=_membrane_topology()
    )
    assert protocol["system_type"] == "membrane"
    assert protocol["barostat_class"] == "MonteCarloMembraneBarostat"
    assert protocol["barostat_frequency"] == 25
    assert protocol["membrane"] == {
        "normal_axis": "z",
        "surface_tension_bar_nm": 0.0,
        "xy_mode": "isotropic",
        "z_mode": "free",
    }
    # 协议必须能直接进 provenance / fingerprint。
    json.dumps(protocol)


# ---------------------------------------------------------------------------
# §3.2 / §7.4 barostat
# ---------------------------------------------------------------------------


def test_membrane_complex_gets_membrane_barostat_with_declared_modes():
    system = _empty_system()
    protocol = core.resolve_membrane_protocol(
        "membrane",
        membrane_config={"surface_tension_bar_nm": 12.5, "barostat_frequency": 50},
        topology=_membrane_topology(),
    )
    result = core.ensure_barostat_for_protocol(
        system, protocol, temperature=310.0, pressure=1.0
    )
    assert result["action"] == "added"

    barostats = core.detect_barostats(system)
    assert [name for _, name in barostats] == ["MonteCarloMembraneBarostat"]

    force = system.getForce(barostats[0][0])
    assert force.getDefaultTemperature().value_in_unit(unit.kelvin) == pytest.approx(310.0)
    assert force.getDefaultPressure().value_in_unit(unit.bar) == pytest.approx(1.0)
    assert force.getDefaultSurfaceTension().value_in_unit(
        unit.bar * unit.nanometer
    ) == pytest.approx(12.5)
    assert force.getXYMode() == openmm.MonteCarloMembraneBarostat.XYIsotropic
    assert force.getZMode() == openmm.MonteCarloMembraneBarostat.ZFree
    assert force.getFrequency() == 50


def test_solvent_leg_gets_plain_isotropic_barostat():
    """§3.2：溶剂腿继续使用普通 MonteCarloBarostat。"""
    system = _empty_system()
    protocol = core.resolve_membrane_protocol("soluble", topology=_soluble_topology())
    core.ensure_barostat_for_protocol(
        system, protocol, temperature=300.0, pressure=1.0
    )
    barostats = core.detect_barostats(system)
    assert [name for _, name in barostats] == ["MonteCarloBarostat"]
    assert system.getForce(barostats[0][0]).getFrequency() == 25


def test_existing_compatible_barostat_is_reused_not_duplicated():
    system = _empty_system()
    system.addForce(openmm.MonteCarloBarostat(1.0 * unit.bar, 300.0 * unit.kelvin, 25))
    protocol = core.resolve_membrane_protocol("soluble", topology=_soluble_topology())
    result = core.ensure_barostat_for_protocol(
        system, protocol, temperature=300.0, pressure=1.0
    )
    assert result["action"] == "reused_existing"
    assert len(core.detect_barostats(system)) == 1


def test_existing_membrane_barostat_is_not_shadowed_by_an_isotropic_one():
    """旧代码的漏检点：isinstance(MonteCarloBarostat) 认不出膜 barostat。

    结果是给一个已经带膜 barostat 的输入 System 再叠一个各向同性的，
    两个同时做体积移动，集合定义错误且完全不报错。现在必须 fail closed。
    """
    system = _empty_system()
    system.addForce(
        openmm.MonteCarloMembraneBarostat(
            1.0 * unit.bar,
            0.0 * unit.bar * unit.nanometer,
            300.0 * unit.kelvin,
            openmm.MonteCarloMembraneBarostat.XYIsotropic,
            openmm.MonteCarloMembraneBarostat.ZFree,
            25,
        )
    )
    # 旧判据认不出它——这正是 MEM-00i 之外的隐藏坑。
    assert not any(
        isinstance(f, openmm.MonteCarloBarostat) for f in system.getForces()
    )
    # 新判据认得出。
    assert [name for _, name in core.detect_barostats(system)] == [
        "MonteCarloMembraneBarostat"
    ]

    soluble = core.resolve_membrane_protocol("soluble", topology=_soluble_topology())
    with pytest.raises(RuntimeError, match="fail closed"):
        core.ensure_barostat_for_protocol(
            system, soluble, temperature=300.0, pressure=1.0
        )
    # 失败之后不许留下第二个 barostat。
    assert len(core.detect_barostats(system)) == 1


def test_membrane_protocol_rejects_preexisting_isotropic_barostat():
    system = _empty_system()
    system.addForce(openmm.MonteCarloBarostat(1.0 * unit.bar, 300.0 * unit.kelvin, 25))
    protocol = core.resolve_membrane_protocol(
        "membrane", topology=_membrane_topology()
    )
    with pytest.raises(RuntimeError, match="fail closed"):
        core.ensure_barostat_for_protocol(
            system, protocol, temperature=300.0, pressure=1.0
        )


def test_multiple_preexisting_barostats_fail_closed():
    system = _empty_system()
    system.addForce(openmm.MonteCarloBarostat(1.0 * unit.bar, 300.0 * unit.kelvin, 25))
    system.addForce(
        openmm.MonteCarloAnisotropicBarostat(
            openmm.Vec3(1.0, 1.0, 1.0) * unit.bar, 300.0 * unit.kelvin
        )
    )
    protocol = core.resolve_membrane_protocol("soluble", topology=_soluble_topology())
    with pytest.raises(RuntimeError, match="已有 2 个 barostat"):
        core.ensure_barostat_for_protocol(
            system, protocol, temperature=300.0, pressure=1.0
        )


# ---------------------------------------------------------------------------
# §3.2 fingerprint 与 §7.7 逐位一致
# ---------------------------------------------------------------------------


def test_legacy_soluble_protocol_contributes_nothing_to_the_fingerprint():
    """§7.7 的核心保证：不声明 system_type → 指纹与改动前逐位相同。"""
    soluble = core.resolve_membrane_protocol("soluble", topology=_soluble_topology())
    assert core.barostat_fingerprint_payload(soluble) is None
    assert core.barostat_fingerprint_payload(None) is None


def test_membrane_protocol_changes_the_fingerprint_payload():
    membrane = core.resolve_membrane_protocol(
        "membrane", topology=_membrane_topology()
    )
    payload = core.barostat_fingerprint_payload(membrane)
    assert payload is not None
    assert payload["barostat_class"] == "MonteCarloMembraneBarostat"
    assert payload["protocol_version"] == core.MEMBRANE_BAROSTAT_PROTOCOL_VERSION


@pytest.mark.parametrize(
    "changed",
    [
        {"surface_tension_bar_nm": 5.0},
        {"xy_mode": "anisotropic"},
        {"z_mode": "fixed"},
        {"barostat_frequency": 100},
    ],
)
def test_changing_any_membrane_parameter_invalidates_the_fingerprint(changed):
    """§3.2：表面张力 / XY-Z 模式 / 频率任一改变都必须让旧 checkpoint 失效。"""
    top = _membrane_topology()
    baseline = core.barostat_fingerprint_payload(
        core.resolve_membrane_protocol("membrane", topology=top)
    )
    modified = core.barostat_fingerprint_payload(
        core.resolve_membrane_protocol("membrane", membrane_config=changed, topology=top)
    )
    assert baseline != modified


def test_pre_equilibration_fingerprint_is_bit_identical_without_system_type():
    """直接对 `_pre_equilibration_fingerprint` 验证 §7.7，而不只是验证 payload。"""
    pytest.importorskip("pymbar")
    from abfe_pipeline import _pre_equilibration_fingerprint

    import numpy as np

    system = _empty_system()
    # 与 test_todo_verified_fixes.py 同口径：坐标用 (N,3) 数组，盒子用 Vec3 列表。
    positions = np.asarray([[0.1, 0.2, 0.3]]) * unit.nanometer
    box = [
        openmm.Vec3(2.0, 0.0, 0.0),
        openmm.Vec3(0.0, 2.0, 0.0),
        openmm.Vec3(0.0, 0.0, 2.0),
    ] * unit.nanometer
    common = dict(
        system=system,
        ligand_indices=[0],
        temperature=300.0,
        pressure=1.0,
        positions=positions,
        box_vectors=box,
        requested_steps=1000,
    )

    legacy = _pre_equilibration_fingerprint(**common)
    soluble = core.resolve_membrane_protocol("soluble", topology=_soluble_topology())
    assert _pre_equilibration_fingerprint(**common, barostat_protocol=soluble) == legacy

    membrane = core.resolve_membrane_protocol("membrane", topology=_membrane_topology())
    assert _pre_equilibration_fingerprint(**common, barostat_protocol=membrane) != legacy


# ---------------------------------------------------------------------------
# 接线契约：溶剂腿不许接膜协议
# ---------------------------------------------------------------------------


def test_pipeline_exposes_environment_type_not_a_second_system_type():
    """`ABFEPipeline` 的新形参必须叫 `environment_type`，不能占用 `system_type`。

    `system_type` 已被 `run_full_pipeline` 用作腿身份（complex/solvent），
    占用它会静默改掉 Boresch 与盒子逻辑。
    """
    pytest.importorskip("pymbar")
    import inspect

    from abfe_pipeline import ABFEPipeline

    init_params = inspect.signature(ABFEPipeline.__init__).parameters
    assert "environment_type" in init_params
    assert "system_type" not in init_params
    assert init_params["environment_type"].default is None

    run_params = inspect.signature(ABFEPipeline.run_full_pipeline).parameters
    assert run_params["system_type"].default == "complex"


def _abfe_pipeline_construction_sites():
    """用 AST 列出 runabfe.py 里所有 `X = ABFEPipeline(...)` 及其关键字参数。

    刻意用 AST 而不是数字符串出现次数：本测试最初写成
    `source.count('environment_type=config.get("system_type")') == 1`，
    而 B2/B6 落地后同一个写法也被用来把环境类型传给
    `resolve_charge_treatment()` / `resolve_dispersion_protocol()`，
    计数从 1 变成 3，测试红了但代码是对的——契约写错了，不是代码坏了。
    真正要守的是"每个 pipeline 构造点都对膜协议做过明确决定"。
    """
    import ast

    source = (ROOT / "runabfe.py").read_text(encoding="utf-8")
    sites = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        callee = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        # 只看 ABFEPipeline；TraditionalABFEPipeline 是另一个类，不在本契约内。
        if callee != "ABFEPipeline":
            continue
        target = node.targets[0]
        sites.append(
            {
                "name": target.id if isinstance(target, ast.Name) else ast.dump(target),
                "lineno": node.lineno,
                "kwargs": {kw.arg for kw in node.value.keywords},
            }
        )
    return sites


def test_every_pipeline_construction_makes_an_explicit_membrane_decision():
    """§3.2：复合物腿必须接膜协议，溶剂腿必须不接——每个构造点都要有明确决定。

    新增一个 `ABFEPipeline(...)` 构造点却忘了决定走哪条，会在这里失败。
    这比"计数等于 N"稳健：加构造点时它强迫你做决定，而不是让你去改一个数字。
    """
    sites = _abfe_pipeline_construction_sites()
    assert len(sites) >= 5, f"ABFEPipeline 构造点只找到 {len(sites)} 个，契约需重新核对"

    solvent = [s for s in sites if s["name"] == "pipeline_solv"]
    assert len(solvent) == 1, "溶剂腿 pipeline 构造点不见了或出现多个"
    # §3.2：溶剂腿是配体在体相水里，与膜无关，永远用普通各向同性 MonteCarloBarostat。
    # 不要"为了一致"把 config 的 system_type 传进来——那会给纯水盒装上膜 barostat。
    assert "environment_type" not in solvent[0]["kwargs"]
    assert "membrane" not in solvent[0]["kwargs"]

    for site in sites:
        if site["name"] == "pipeline_solv":
            continue
        assert "environment_type" in site["kwargs"], (
            f"runabfe.py:{site['lineno']} 的 {site['name']} 是复合物腿 pipeline，"
            "必须显式传 environment_type，否则膜体系下会静默用各向同性 barostat"
        )
        assert "membrane" in site["kwargs"], (
            f"runabfe.py:{site['lineno']} 的 {site['name']} 缺 membrane 子配置"
        )


def test_dispersion_protocol_is_wired_to_every_leg_including_solvent():
    """§1.3 路线 A / §7.5：复合物腿与溶剂腿必须用**同一套** ligand–environment 非键定义。

    这与恒压器的规矩**方向相反**，容易搞反：
      - 恒压器：溶剂腿刻意不接膜协议（纯水盒不能装膜 barostat）；
      - 色散路线：溶剂腿必须接，且与复合物腿相同。

    若复合物腿按 §1.3 关掉了炼金均匀密度 LRC 而溶剂腿还留着，两腿的
    ligand–environment 口径就不一致，ΔG_bind = ΔG_solv − ΔG_cplx 的差值里会混进
    一个纯协议差——而且不会有任何报错。
    """
    sites = _abfe_pipeline_construction_sites()
    for site in sites:
        assert "dispersion_protocol" in site["kwargs"], (
            f"runabfe.py:{site['lineno']} 的 {site['name']} 缺 dispersion_protocol；"
            "两腿色散路线不一致会在 ΔG_bind 里混进协议差"
        )
    # 溶剂腿拿到的必须是同一个已解析值，而不是重新解析或留空。
    source = (ROOT / "runabfe.py").read_text(encoding="utf-8")
    _, marker, tail = source.partition("pipeline_solv = ABFEPipeline(")
    assert marker
    solvent_call, closer, _ = tail.partition("\n    )")
    assert closer
    assert 'dispersion_protocol=_dispersion_protocol["dispersion_protocol"]' in solvent_call


def test_membrane_equilibration_initializes_velocities_but_soluble_does_not():
    """膜体系必须在最小化后初始化速度；可溶路径**不能**动（§7.7 / R7）。

    `pre_equilibrate()` 原来从不调 `setVelocitiesToTemperature`（`ibs_engine` 里有
    10 处都调，唯独它没有），于是体系从 **0 K** 起跑而 barostat 已在做体积移动。
    冷体系下 |ΔE| 很小、`P·ΔV` 主导 → 倾向接受大幅压缩；纯水盒扛得住（可溶路径
    一直如此），但膜在法向上没有横向支撑，Z 自由那一维可能被压塌。
    实测（`memtest/diagnose_nan.py` 的 [F]/[G]）：2 ps 内不初始化速度时 Z 收缩
    0.55%，初始化后 0.00%。

    但给可溶路径加速度初始化会改变现有生产基线的轨迹，所以必须**只对膜生效**。
    """
    source = (ROOT / "abfe_pipeline.py").read_text(encoding="utf-8")
    body = source.split("def pre_equilibrate")[1]

    guard = body.find("if self.environment_type == ENVIRONMENT_TYPE_MEMBRANE:")
    set_velocities = body.find("setVelocitiesToTemperature")
    assert guard > 0, "pre_equilibrate 里找不到膜守卫"
    assert set_velocities > guard, (
        "setVelocitiesToTemperature 必须在膜守卫**之内**——"
        "无条件初始化速度会改变可溶体系的生产基线（§7.7 / R7）"
    )
    # 种子固定，保证同一输入可复现。
    assert "MEMBRANE_EQUILIBRATION_VELOCITY_SEED = 20260730" in source
    assert "self.temperature, MEMBRANE_EQUILIBRATION_VELOCITY_SEED" in source


def test_membrane_equilibration_writes_a_state_monitor_and_soluble_does_not():
    """膜预平衡必须周期性落盘体积/温度/能量；可溶路径的 reporter 组合不变。

    首跑的 NaN 出现在 0.4 ns 之后（离线诊断已排除输入坏点、预处理破坏、原子重叠、
    以及"barostat 压塌盒子"），所以只能**在跑中**留下轨迹才能知道崩的时候是体积
    失控、温度失控还是能量先发散。
    """
    source = (ROOT / "abfe_pipeline.py").read_text(encoding="utf-8")
    body = source.split("def pre_equilibrate")[1]

    guard = body.find(
        "self.environment_type == ENVIRONMENT_TYPE_MEMBRANE\n            and save_traj"
    )
    reporter = body.find("app.StateDataReporter(")
    assert guard > 0, "找不到膜专属的监控守卫"
    assert reporter > guard, (
        "StateDataReporter 必须在膜守卫之内——可溶路径不必要就不动（§7.7）"
    )
    assert "MEMBRANE_EQUILIBRATION_MONITOR_INTERVAL = 5000" in source
    # 崩的时候要能分辨是哪一项先失控，这几项都得记。
    for field in ("potentialEnergy=True", "temperature=True", "volume=True"):
        assert field in body[reporter : reporter + 800], field


def test_membrane_requires_a_bonded_topology_not_the_mmcif_cache():
    """mmCIF 缓存会丢掉非标准残基的键；膜体系必须从 `.top` 重建拓扑。

    实测根因：`app.PDBxFile.writeFile` **不写任何键记录**（写入端没有
    `struct_conn` / `chem_comp_bond`），读取端只能靠 `createStandardBonds()` 补
    **标准残基**（氨基酸/核酸/水）。于是 `POPC`(PA/PC/OL) / `MOL` / 离子的键全丢。

    而 `pre_equilibrate` 之前的「PBC 分子完整性修复」靠 topology 的键判断
    "什么算一个分子" —— 键丢了就把跨周期边界的脂质**逐段撕开**：
    最小化后 PE = 4.1e13 kJ/mol、max|F| = 3.7e9（落在 `PA334/H8S`），
    几千步后变成一个没有上下文的 `Particle coordinate is NaN`。
    从 `.top` 重建（有键）时同一体系 PE ≈ −6.5e5，完全正常。

    ⚠️ 关键：`runabfe` 在**全新构建**之后也会立刻 `load_native_system` 重新读回
    （"确保后续所有对象都来自落盘文件"），所以首跑与缓存命中**都会**中招。
    这正是"直接用 build_system_from_gromacs 复现不出 NaN"的原因。
    """
    source = (ROOT / "runabfe.py").read_text(encoding="utf-8")

    # 参数存在且默认关闭（可溶路径不变，§7.7 / R7）。
    import inspect

    import runabfe

    params = inspect.signature(runabfe.load_native_system).parameters
    assert "require_bonded_topology" in params
    assert params["require_bonded_topology"].default is False

    # 复合物腿两处加载点都要接上（缓存命中 + 全新构建后重新读回）。
    assert source.count("require_bonded_topology=_require_bonded_topology,") == 2

    # 环境类型必须在**加载 System 之前**解析，否则 NameError。
    lines = source.splitlines()
    define = next(
        i for i, line in enumerate(lines)
        if "_environment_type = resolve_environment_type(" in line
    )
    uses = [
        i for i, line in enumerate(lines)
        if "require_bonded_topology=_require_bonded_topology," in line
    ]
    assert uses and all(define < u for u in uses), "定义必须先于使用"

    # 两处环境类型解析必须有一致性断言，避免"带键拓扑"与"用哪种 barostat"分叉。
    assert 'if _barostat_protocol["system_type"] != _environment_type:' in source

    # mmCIF 分支要校验键数，不能只看原子数。
    assert "mmCIF 拓扑只有 %d 个键" in source


def test_topology_must_carry_box_vectors_before_a_membrane_run():
    """DCD 的 unitcell 取自 **topology** 的盒矢量，缺了要在开跑前拦住。

    实测教训：`app.DCDFile` 写每帧 unitcell 时读的是
    `self._topology.getPeriodicBoxVectors()`（`dcdfile.py:155`），为 None 就整段不写、
    header 的 `boxFlag` 也是 0。而 §9 膜质量门要从这条轨迹算 APL 与盒序列 ——
    拿到没有 unitcell 的 DCD 时只能报错，**那时 10 ns 已经烧完了**。

    根因是一处 fail-open：从 `.top` 重建拓扑时 `GromacsTopFile` 只在显式传
    `periodicBoxVectors` 时才设盒，少传一个参数就静默产出缺盒信息的轨迹。
    （这个坑正是"修好 mmCIF 丢键"那一版引入的。）
    """
    pipeline_src = (ROOT / "abfe_pipeline.py").read_text(encoding="utf-8")
    runabfe_src = (ROOT / "runabfe.py").read_text(encoding="utf-8")
    core_src = (ROOT / "abfe_core.py").read_text(encoding="utf-8")

    # 预平衡开跑前必须拦住，而不是等质量门。
    body = pipeline_src.split("def pre_equilibrate")[1]
    guard = body.find("self.topology.getPeriodicBoxVectors() is None")
    assert guard > 0, "pre_equilibrate 里缺少「拓扑无盒矢量」的守卫"
    assert "膜体系的 topology 没有周期盒矢量" in body
    # 守卫必须在 step() 之前。
    assert guard < body.find("simulation.step(steps_remaining)")

    # 三条拓扑恢复路径都要补盒矢量（重建时传参 + 统一兜底）。
    assert (
        "top_file, includeDir=inc_dir, periodicBoxVectors=box_vectors" in runabfe_src
    )
    assert "topology.setPeriodicBoxVectors(box_vectors)" in runabfe_src

    # 提取器的报错要指明根因位置，否则又要从头查一轮。
    assert "dcdfile.py:155" in core_src


def test_bad_starting_state_fails_closed_after_minimization():
    """最小化后受力异常必须**当场**报错，不许跑到 NaN 才发现。

    实测教训：一个坏掉的 System（缓存里的 `system_native.xml`）最小化后
    PE = 4.1e13 kJ/mol、max|F| = 3.7e9（落在脂质尾链氢 PA334/H8S），
    但当时代码什么都不报，几千步后只给出一个**没有上下文**的
    `Particle coordinate is NaN`，于是花了好几轮去猜。
    正常时 PE ≈ −6.5e5、max|F| ≈ 2.5e3 且落在水上 —— 相差 6 个数量级。

    2026-08-02（MEM-06）起这道门收敛为**唯一实现**
    `abfe_core.assert_starting_state_is_sane()`，Boresch attachment 腿的起点体检
    调的是同一个函数 —— 同一道门写两遍迟早有一处漏改。
    """
    import abfe_core as core

    source = (ROOT / "abfe_pipeline.py").read_text(encoding="utf-8")
    core_src = (ROOT / "abfe_core.py").read_text(encoding="utf-8")
    assert "MEMBRANE_POST_MINIMIZATION_MAX_FORCE_KJ_PER_MOL_NM = 1.0e6" in source
    # 调用点的阈值别名不许与共用实现的默认值漂开。
    assert core.STARTING_STATE_MAX_FORCE_KJ_PER_MOL_NM == 1.0e6

    gate_body = core_src.split("def assert_starting_state_is_sane")[1].split(
        "\ndef "
    )[0]
    # 必须先报数、再判门；门要 raise 而不是 warning。
    assert "{label}: PE =" in gate_body
    assert "raise RuntimeError(" in gate_body
    assert "起始态就是坏的" in gate_body
    # NaN/Inf 受力要单独更早地拦住。
    assert "个原子受力为 NaN/Inf" in gate_body
    # 报错要指出最可能的原因与验证方法，否则用户仍然只能猜——提示由调用点提供。
    equil_body = source.split("def pre_equilibrate")[1]
    assert 'label="最小化后"' in equil_body
    assert "compare_systems.py" in equil_body

    # attachment 腿必须用同一个函数体检，且体检要在第一次 step 之前。
    engine_src = (ROOT / "ibs_engine.py").read_text(encoding="utf-8")
    leg_body = engine_src.split("def run_boresch_attachment_leg")[1].split("\ndef ")[0]
    assert "assert_starting_state_is_sane(" in leg_body, (
        "attachment 腿必须复用同一道起始态门 —— 2026-08-02 的 NaN 就是因为"
        "它开跑前什么都不量，只留下一个没有上下文的 Particle coordinate is NaN"
    )
    # 锚点用"第一次 simulation.step("，而不是某个具体的步数表达式：
    # 平衡段 2026-08-03 起改成分块步进（为了在崩之前留下监控行），
    # 原先那个 `simulation.step(int(equil_steps_per_state))` 字面量已不存在。
    assert leg_body.index("assert_starting_state_is_sane(") < leg_body.index(
        "simulation.step("
    ), "起点体检必须在第一次 step 之前"
    # 监控必须在崩之前就有行落盘，否则又是"只有一个 traceback"。
    assert "stage0_attachment_monitor.csv" in leg_body
    assert "stage0_attachment_inputs.json" in leg_body
    # 扫描顺序是从全强度端往下：日志必须说清第一个跑的是哪个 λ，
    # 否则会像 2026-08-02 那样按"λ 列表第一个是 0.0"误判成"限制力为零时炸的"。
    assert "第一个实际跑的态是" in leg_body


def test_rerun_pipelines_inherit_the_source_protocol_instead_of_rereading_config():
    """增量重跑必须与被复用的那次用同一个恒压器集合。

    `--only-complex-charging` / `--only-boresch-attachment` 复用冻结的
    stage1/stage2/solvent 结果。如果 rerun 重读 config 而 config 已经改过
    （比如加了 `--system-type membrane`），就会出现"被复用的结果是各向同性下测的、
    增量那段是膜恒压下测的"这种不可比拼接，且不会报错。
    """
    source = (ROOT / "runabfe.py").read_text(encoding="utf-8")
    rerun_wiring = "environment_type=source_pipeline.environment_type"
    # 两个 rerun 入口各一处。
    assert source.count(rerun_wiring) == 2, (
        f"rerun 继承接线出现 {source.count(rerun_wiring)} 处，预期 2 "
        "（_run_boresch_attachment_only 与 _run_complex_charging_only 各一）"
    )
    assert 'environment_type=config.get("system_type")' not in source.split(
        "def _run_boresch_attachment_only"
    )[1].split("def ")[0], "rerun 不得重读 config 的 system_type"


def test_interrupted_pre_equilibration_is_not_mistaken_for_a_finished_one(tmp_path):
    """MEM-09：被中断的预平衡不许被当成"已完成"复用。

    `equilibrium_is_done()` 原先只查「轨迹存在 + checkpoint 存在 + 指纹相符」，
    而 `pre_equilibration_fingerprint.json` 是 `pre_equilibrate()` 在**第一步之前**
    就写下的（为的是让被中断的运行下次能认出自己），它记的 `n_steps` 是**目标**步数。

    于是一次 100 ns 跑到 40 ns 被杀掉的运行，下次会被判成"已完成"→
    `pre_equilibrate()` 整段跳过 → 连它内部的 §9 膜质量门一起跳过，
    而 provenance 与指纹都写着 100 ns。这是短平衡冒充长平衡，`enforce` 拦不到。
    真正的完成标记是 `pipeline_state.json` 的 `equilibration.status=completed`。
    """
    import runabfe

    fp = "deadbeef"

    def _write_real_minimal_dcd(path, chk_path):
        """[P1-07] 写一条真实的极小 DCD。

        `equilibrium_is_done()` 现在会用真实 DCD parser 校验轨迹——垃圾字节
        （旧行为能骗过的大小阈值用例）必须被拒。这里用 openmm 的 DCDReporter
        产出一条 300 帧、>10 KB 的合法轨迹，让本测试继续聚焦 MEM-09 的
        "完成状态 vs 目标步数"判断。
        """
        import openmm
        import openmm.app as _app

        topology = _app.Topology()
        chain = topology.addChain()
        residue = topology.addResidue("LIG", chain)
        topology.addAtom("C1", _app.element.carbon, residue)
        system = openmm.System()
        system.addParticle(12.011 * openmm.unit.dalton)
        vec = (
            openmm.Vec3(3.0, 0.0, 0.0),
            openmm.Vec3(0.0, 3.0, 0.0),
            openmm.Vec3(0.0, 0.0, 3.0),
        )
        system.setDefaultPeriodicBoxVectors(*[v * openmm.unit.nanometer for v in vec])
        integrator = openmm.VerletIntegrator(0.001 * openmm.unit.picoseconds)
        simulation = _app.Simulation(topology, system, integrator)
        simulation.context.setPositions([[0.5, 0.5, 0.5] * openmm.unit.nanometer])
        reporter = _app.DCDReporter(str(path), 1, enforcePeriodicBox=False)
        simulation.reporters.append(reporter)
        simulation.step(300)
        simulation.saveCheckpoint(str(chk_path))
        del reporter, simulation, integrator

    def _probe_simulation():
        """[P1-07] 与上面写 DCD/checkpoint 用的是同一个 System/Topology。

        `equilibrium_is_done()` 现在把"目标 Simulation 能真的 loadCheckpoint"
        当硬前提（缺 simulation 一律 return False）。本测试要判的是 MEM-09 的
        "完成状态 vs 目标步数"状态机，所以必须给它一个**真的能加载**的探针：
        checkpoint 由 `simulation.saveCheckpoint()` 真实写出（原来写的是
        `b"y" * 1024` 这种假字节，加上严格校验后必然被拒，正例就退化成恒 False、
        四个分支全部失去区分力）。
        """
        import openmm
        import openmm.app as _app

        topology = _app.Topology()
        chain = topology.addChain()
        residue = topology.addResidue("LIG", chain)
        topology.addAtom("C1", _app.element.carbon, residue)
        system = openmm.System()
        system.addParticle(12.011 * openmm.unit.dalton)
        vec = (
            openmm.Vec3(3.0, 0.0, 0.0),
            openmm.Vec3(0.0, 3.0, 0.0),
            openmm.Vec3(0.0, 0.0, 3.0),
        )
        system.setDefaultPeriodicBoxVectors(*[v * openmm.unit.nanometer for v in vec])
        sim = _app.Simulation(
            topology, system, openmm.VerletIntegrator(0.001 * openmm.unit.picoseconds)
        )
        sim.context.setPositions([[0.5, 0.5, 0.5] * openmm.unit.nanometer])
        return sim

    _probe = _probe_simulation()

    def _make(status, achieved, target=50_000_000, with_state=True):
        d = tmp_path / f"{status}-{achieved}-{target}-{with_state}"
        (d / "checkpoints").mkdir(parents=True)
        _write_real_minimal_dcd(
            d / "pre_equilibration.dcd", d / "checkpoints" / "pre_equil.chk"
        )
        (d / "pre_equilibration_fingerprint.json").write_text(
            json.dumps({"fingerprint": fp, "n_steps": target}), encoding="utf-8"
        )
        if with_state:
            (d / "checkpoints" / "pipeline_state.json").write_text(
                json.dumps(
                    {"stages": {"equilibration": {
                        "status": status, "total_steps": achieved}}}
                ),
                encoding="utf-8",
            )
        return str(d)

    # 跑完了 → 复用（既有行为，output_lrc_fix 那类目录不受影响）
    assert runabfe.equilibrium_is_done(
        _make("completed", 50_000_000), expected_fingerprint=fp, simulation=_probe
    ) is True

    # 被中断（状态还是 running）→ 不许复用
    assert runabfe.equilibrium_is_done(
        _make("running", 20_000_000), expected_fingerprint=fp, simulation=_probe
    ) is False

    # 声称 completed 但步数不够目标（例如目标从 10 ns 调到 100 ns）→ 不许复用
    assert runabfe.equilibrium_is_done(
        _make("completed", 5_000_000), expected_fingerprint=fp, simulation=_probe
    ) is False

    # 连 pipeline_state.json 都没有 → 无法证明跑完，保守视为未完成
    assert runabfe.equilibrium_is_done(
        _make("completed", 50_000_000, with_state=False), expected_fingerprint=fp, simulation=_probe
    ) is False


def test_pre_equilibration_checkpoint_interval_bounds_the_work_lost_on_resume():
    """续跑最多丢多少：CheckpointReporter 的间隔 × 步长。"""
    import abfe_core as core

    source = (ROOT / "abfe_pipeline.py").read_text(encoding="utf-8")
    body = source.split("def pre_equilibrate")[1]
    assert "CheckpointReporter(chk_file, 100000)" in body
    # 100000 步 × 2 fs = 200 ps。若有人改小步长/改大间隔，这条会提醒重新算。
    assert 100000 * core.PRE_EQUILIBRATION_TIMESTEP_PS == pytest.approx(200.0)
    # 续跑不能猜 checkpoint 对应的 DCD 字节偏移；必须开启新 segment，
    # 由 manifest 保存旧轨迹证据，canonical reporter 不得盲目 append。
    assert "append=False" in body


def test_quality_gate_cannot_be_bypassed_by_rerunning(tmp_path):
    """MEM-14：门必须在**每个消费预平衡的入口**判，不能只在 pre_equilibrate 里判。

    此前 `_update_stage_status("equilibration","completed")` 与预平衡指纹写在门
    **之前**，所以门失败后原样重跑一次 → `equilibrium_is_done()` 为真 →
    `pre_equilibrate()` 整段跳过 → **门也一起被跳过** → 直接进 Stage 0。
    `enforce` 的语义被控制流击穿了。实测触发过（memtest 100 ns 那轮）。
    """
    import inspect
    import abfe_pipeline

    pipeline_src = (ROOT / "abfe_pipeline.py").read_text(encoding="utf-8")
    runabfe_src = (ROOT / "runabfe.py").read_text(encoding="utf-8")

    assert hasattr(abfe_pipeline.ABFEPipeline, "ensure_membrane_quality_gate_passed")

    # 完成状态确实写在门之前——这正是绕过路径成立的前提，写成断言以免有人
    # 以为"调换顺序就够了"（调换顺序会让中断的运行留下未完成状态，是另一个坑）。
    equil_body = pipeline_src.split("def pre_equilibrate")[1]
    assert equil_body.index('"equilibration",') < equil_body.index(
        "_evaluate_membrane_quality_gate_after_equilibration"
    )

    # run_full_pipeline 必须在开头就调，且在任何采样之前。
    full_body = pipeline_src.split("def run_full_pipeline")[1].split("\n    def ")[0]
    assert "ensure_membrane_quality_gate_passed()" in full_body, (
        "run_full_pipeline 必须先过 §9 门——否则门失败后重跑就直接进 Stage 0"
    )
    assert full_body.index("ensure_membrane_quality_gate_passed()") < full_body.index(
        "Stage 0"
    ) if "Stage 0" in full_body else True

    # 两个增量重跑入口同样在消费那次预平衡。
    for entry in ("_run_boresch_attachment_only", "_run_complex_charging_only"):
        body = runabfe_src.split(f"def {entry}")[1].split("\ndef ")[0]
        assert "ensure_membrane_quality_gate_passed()" in body, (
            f"{entry} 也在消费预平衡，必须先过 §9 门，否则它就是另一条绕过路径"
        )

    # 幂等 + 可溶体系短路：签名与实现里都要有对应分支。
    gate_src = inspect.getsource(
        abfe_pipeline.ABFEPipeline.ensure_membrane_quality_gate_passed
    )
    assert "ENVIRONMENT_TYPE_MEMBRANE" in gate_src
    assert "_membrane_quality_gate_report" in gate_src


def test_torn_rigid_water_is_caught_before_dynamics(tmp_path):
    """MEM-15：跨镜像的刚性水必须在**开跑前**被抓住，而不是几百步后 NaN。

    2026-08-03 实测（memtest 100 ns，Stage 0）：PBC 分子完整性修复按 topology 的
    **键**归组分子，而刚性水的 O–H 只以**约束**存在（`topology.bonds()` 里涉及水的
    键数 = **0**，约束 28626 个 = 9542 水 × 3）。于是跨边界的 243 个水被逐原子回卷、
    O 与 H 落到不同镜像 → 729 个 PME 排除对跨盒（最远 13.76 nm）。

    这类损坏对**所有**常规诊断隐形：
      * 水没有键力项 ⟹ 键能（9525.72，两边逐位相同）与最大键长（0.19 nm）正常；
      * PME 误差是平滑长程项 ⟹ 势能只是偏移 −30.9 MJ/mol，`max|F|` = 5292 正常；
      * 崩的是**约束求解器** ⟹ 不到 1 ps 就 `Particle coordinate is NaN`。
    """
    import numpy as np
    import abfe_core as core

    # 三点刚性水 ×2：O–H/H–H 只做约束，**不加任何键力**（与真实拓扑一致）。
    system = openmm.System()
    topology = app.Topology()
    chain = topology.addChain()
    box = 2.0
    for _ in range(2):
        res = topology.addResidue("HOH", chain)
        for name, element, mass in (
            ("O", app.element.oxygen, 15.999),
            ("H1", app.element.hydrogen, 1.008),
            ("H2", app.element.hydrogen, 1.008),
        ):
            topology.addAtom(name, element, res)
            system.addParticle(mass)
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb.setCutoffDistance(0.9 * unit.nanometer)
    for i in range(6):
        q = -0.834 if i % 3 == 0 else 0.417
        nb.addParticle(q, 0.315 * unit.nanometer, 0.636 * unit.kilojoule_per_mole)
    for base in (0, 3):
        for a, b in ((0, 1), (0, 2), (1, 2)):
            nb.addException(base + a, base + b, 0.0, 0.1, 0.0)
    system.addForce(nb)
    for base in (0, 3):
        system.addConstraint(base + 0, base + 1, 0.09572 * unit.nanometer)
        system.addConstraint(base + 0, base + 2, 0.09572 * unit.nanometer)
        system.addConstraint(base + 1, base + 2, 0.15139 * unit.nanometer)
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box, 0, 0) * unit.nanometer,
        openmm.Vec3(0, box, 0) * unit.nanometer,
        openmm.Vec3(0, 0, box) * unit.nanometer,
    )
    assert topology.getNumBonds() == 0, "刚性水本来就没有键——这正是问题的前提"

    context = openmm.Context(
        system, openmm.VerletIntegrator(0.001),
        openmm.Platform.getPlatformByName("Reference"),
    )
    context.setPeriodicBoxVectors(
        openmm.Vec3(box, 0, 0) * unit.nanometer,
        openmm.Vec3(0, box, 0) * unit.nanometer,
        openmm.Vec3(0, 0, box) * unit.nanometer,
    )

    intact = np.array([
        [0.50, 0.50, 0.50], [0.59, 0.52, 0.50], [0.46, 0.58, 0.50],
        [1.50, 1.50, 1.50], [1.59, 1.52, 1.50], [1.46, 1.58, 1.50],
    ])
    context.setPositions(intact * unit.nanometer)
    report = core.assert_starting_state_is_sane(context, topology, label="完好")
    assert report["periodic_image_consistency"]["checked"] is True
    assert report["periodic_image_consistency"]["constraints"]["n_over_limit"] == 0

    # 把第二个水的 H2 逐原子回卷（正是 image_molecules 在没有键时干的事）。
    torn = intact.copy()
    torn[5] -= np.array([box, 0.0, 0.0])
    context.setPositions(torn * unit.nanometer)
    with pytest.raises(RuntimeError, match="跨了周期镜像"):
        core.assert_starting_state_is_sane(context, topology, label="撕开的水")


def test_pbc_repair_promotes_constraints_to_bonds_for_grouping():
    """契约：PBC 修复必须把约束补成键，否则刚性水会被 image_molecules 撕开。"""
    source = (ROOT / "abfe_pipeline.py").read_text(encoding="utf-8")
    body = source.split("def repair_pbc_molecule_integrity")[1].split("\n    def ")[0]
    assert "getConstraintParameters" in body, (
        "必须把 System 的约束补成键再交给 image_molecules() —— "
        "刚性水的 O–H 只以约束存在（实测 topology.bonds() 里 0 个水键）"
    )
    assert "add_bond" in body
    # 锚点用**实际调用** `traj.image_molecules(`，不是裸名字——docstring 里就提到了
    # `image_molecules()`，用裸名字会命中说明文字而不是调用点。
    assert body.index("getConstraintParameters") < body.index("traj.image_molecules(")
