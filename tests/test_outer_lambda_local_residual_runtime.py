"""CPU-only contracts for the formal Outer-Lambda Local Residual switch.

2026-08-31 发布整理：冻结 R1 模型资源 `resources/outer_lambda_local_residual/`
不随本工程区分支分发（它硬绑 Atenolol 的 41 个原子与内部键图，换体系用不上，
而重训需要的训练栈同样不在本分支）。因此下面所有真正去加载模型的用例改成
**缺资源时 skip**，而不是删掉——把资源拷回来它们就该重新变绿。
开关本身的配置契约（不需要资源那几条）继续无条件运行。

2026-09-10：资源已从 Atenolol-rank11 接回本仓，`requires_frozen_r1_resource`
因此全部通过；标记保留，资源再被移走时它们会重新变成 skip 而不是红。
"""

import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.cpu_only

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RESOURCE_MANIFEST = _REPO_ROOT / "resources/outer_lambda_local_residual/manifest.json"

#: 用在需要加载冻结 R1 模型的用例上。
requires_frozen_r1_resource = pytest.mark.skipif(
    not _RESOURCE_MANIFEST.is_file(),
    reason=(
        "冻结 R1 模型资源不随本工程区分支分发（只对 Atenolol 有效）；"
        "从 Atenolol-rank11 取回 resources/outer_lambda_local_residual/ 后本用例恢复"
    ),
)


def test_run_config_has_one_formal_boolean_and_accepts_legacy_alias(tmp_path, monkeypatch):
    import runabfe

    config_path = tmp_path / "legacy.json"
    config_path.write_text(json.dumps({"residual_sampling_enabled": True}))
    monkeypatch.setattr(
        "sys.argv",
        ["runabfe.py", "--config", str(config_path)],
    )
    cfg = runabfe.RunConfig(runabfe.parse_arguments())
    assert cfg.outer_lambda_local_residual_ibs is True
    assert "residual_sampling_enabled" not in cfg.data


def test_run_config_rejects_conflicting_switch_spellings(tmp_path, monkeypatch):
    import runabfe

    config_path = tmp_path / "conflict.json"
    config_path.write_text(
        json.dumps(
            {
                "outer_lambda_local_residual_ibs": True,
                "residual_sampling_enabled": False,
            }
        )
    )
    monkeypatch.setattr(
        "sys.argv",
        ["runabfe.py", "--config", str(config_path)],
    )
    with pytest.raises(ValueError, match="取值冲突"):
        runabfe.RunConfig(runabfe.parse_arguments())


@requires_frozen_r1_resource
def test_formal_loader_builds_fresh_cpu_plugin_force_without_writing_output():
    openmm = pytest.importorskip("openmm")
    from openmm import XmlSerializer, app
    from local_residual.openmm_plugin import (
        FEATURE_NAME,
        build_outer_lambda_local_residual_runtime,
    )

    # 用 fixtures 里的溶剂腿体系：复合物腿那三份冻结产物（topology.cif 9 MB +
    # system_native.xml 17 MB）不适合当仓库 fixture，而这条断言的是"加载器从
    # resources/ 取模型、不碰 output/"，跟哪条腿的体系无关。索引直接显式传，
    # 于是连 output_dir 都不需要——本用例名字里的 "without writing output"。
    root = Path(__file__).resolve().parents[1]
    fixtures = root / "tests/fixtures/output"
    topology = app.PDBxFile(str(fixtures / "topology_solvent.cif")).topology
    system = XmlSerializer.deserialize((fixtures / "system_solvent.xml").read_text())
    indices_path = fixtures / "ligand_indices_solvent.json"
    ligand_indices = json.loads(indices_path.read_text())["ligand_indices"]
    runtime = build_outer_lambda_local_residual_runtime(
        topology=topology,
        ligand_indices=ligand_indices,
        system=system,
        temperature_kelvin=300.0,
        potential_type="softcore",
        ligand_indices_path=indices_path,
        platform_name="CPU",
    )
    assert FEATURE_NAME in runtime.provenance_payload()["feature"]
    assert runtime.provenance_payload()["model"]["supported_ligand"] == "Atenolol"
    assert "output/outer_lambda_exp025" not in runtime.controller.bases[0].model_path
    assert "resources/outer_lambda_local_residual" in runtime.controller.bases[0].model_path
    assert len(runtime.atom_type_index) == topology.getNumAtoms()
    assert runtime.state_coefficients_factory([0.0, 0.5, 1.0], [0.0, 0.5, 1.0]) == [
        0.0,
        1.0,
        0.0,
    ]
    force = runtime.force_factory()
    assert isinstance(force, openmm.Force)


@requires_frozen_r1_resource
def test_formal_loader_binds_solvent_indices_to_the_solvent_leg():
    openmm = pytest.importorskip("openmm")
    from openmm import XmlSerializer, app
    from local_residual.openmm_plugin import build_outer_lambda_local_residual_runtime

    root = Path(__file__).resolve().parents[1]
    topology = app.PDBxFile(str(root / "tests/fixtures/output/topology_solvent.cif")).topology
    system = XmlSerializer.deserialize((root / "tests/fixtures/output/system_solvent.xml").read_text())
    indices_path = root / "tests/fixtures/output/ligand_indices_solvent.json"
    ligand_indices = json.loads(indices_path.read_text())["ligand_indices"]
    runtime = build_outer_lambda_local_residual_runtime(
        topology=topology,
        ligand_indices=ligand_indices,
        system=system,
        temperature_kelvin=300.0,
        potential_type="softcore",
        # 索引文件跟拓扑/System 一样取自 fixtures：仓库的 output/ 在发布整理时
        # 已经清空，而这条断言只关心 loader 会不会把腿名拼进文件名。
        output_dir=root / "tests/fixtures/output",
        platform_name="CPU",
        leg_name="solvent",
    )
    basis = runtime.controller.bases[0]
    assert basis.atom_indices_path.endswith("output/ligand_indices_solvent.json")
    assert runtime.provenance_payload()["model"]["leg_name"] == "solvent"


@requires_frozen_r1_resource
def test_formal_loader_rejects_a_same_size_but_different_ligand(tmp_path):
    openmm = pytest.importorskip("openmm")
    from openmm import app
    from local_residual.openmm_plugin import build_outer_lambda_local_residual_runtime

    root = Path(__file__).resolve().parents[1]
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("NEW", chain)
    for index in range(41):
        topology.addAtom("C", app.element.carbon, residue)
        if index:
            topology.addBond(list(residue.atoms())[-2], list(residue.atoms())[-1])
    indices_path = tmp_path / "ligand_indices.json"
    indices_path.write_text(json.dumps({"ligand_indices": list(range(41))}))
    with pytest.raises(RuntimeError, match="只支持 Atenolol"):
        build_outer_lambda_local_residual_runtime(
            topology=topology,
            ligand_indices=list(range(41)),
            temperature_kelvin=300.0,
            potential_type="softcore",
            ligand_indices_path=indices_path,
            platform_name="CPU",
            plugin_build_dir=root / "plugins/LocalManyBodyResidual/build",
        )


def test_formal_pipeline_em_scope_is_canonical_and_exception_safe(monkeypatch):
    import runabfe
    from local_residual import em_no_residual

    assert runabfe.install_outer_lambda_em_policy is em_no_residual.install
    assert runabfe.uninstall_outer_lambda_em_policy is em_no_residual.uninstall

    events = []
    monkeypatch.setattr(runabfe, "install_outer_lambda_em_policy", lambda: events.append("install"))
    monkeypatch.setattr(runabfe, "uninstall_outer_lambda_em_policy", lambda: events.append("uninstall"))

    calls = []

    class FakePipeline:
        def run_full_pipeline(self, **kwargs):
            calls.append((self.name, kwargs))
            if self.fail:
                raise RuntimeError("synthetic pipeline failure")
            return self.name

    complex_pipeline = FakePipeline()
    complex_pipeline.name = "complex"
    complex_pipeline.fail = False
    solvent_pipeline = FakePipeline()
    solvent_pipeline.name = "solvent"
    solvent_pipeline.fail = False

    original_complex = complex_pipeline.run_full_pipeline
    runabfe._scope_pipeline_with_optional_outer_lambda_em(complex_pipeline, False)
    assert complex_pipeline.run_full_pipeline is not None
    assert complex_pipeline.run_full_pipeline() == "complex"
    assert events == []

    runabfe._scope_pipeline_with_optional_outer_lambda_em(complex_pipeline, True)
    assert complex_pipeline.run_full_pipeline() == "complex"
    assert events == ["install", "uninstall"]
    assert complex_pipeline.run_full_pipeline.__func__ is original_complex.__func__

    runabfe._scope_pipeline_with_optional_outer_lambda_em(solvent_pipeline, True)
    solvent_pipeline.fail = True
    with pytest.raises(RuntimeError, match="synthetic pipeline failure"):
        solvent_pipeline.run_full_pipeline()
    assert events == ["install", "uninstall", "install", "uninstall"]
    assert solvent_pipeline.run_full_pipeline.__func__ is FakePipeline.run_full_pipeline
    assert [name for name, _kwargs in calls] == ["complex", "complex", "solvent"]


def test_missing_frozen_resource_fails_closed_with_an_actionable_message(tmp_path):
    """资源不在时必须说清"为什么没有、怎么拿回来"，不能只抛一个裸路径。

    真实场景：2026-08-31 把 `resources/` 移出工程区分支后，
    `_load_resource_manifest` 原来会在 `manifest.read_text()` 上抛
    `FileNotFoundError: [Errno 2] ... manifest.json`——只有一个路径，
    读起来像"装坏了"，而实际是这份资源本来就不随包发布。
    """
    from local_residual.openmm_plugin import (
        RESOURCE_MISSING_HINT,
        _load_resource_manifest,
    )

    # 指一个确定不存在的路径，而不是仓库里那份：这条测的是"缺资源时的错误
    # 文案"，跟资源当下在不在 checkout 里无关（2026-09-10 资源接回来之后，
    # 原来那个 `if missing.is_file(): skip` 让它永久失效了）。
    missing = tmp_path / "resources/outer_lambda_local_residual/manifest.json"

    with pytest.raises(FileNotFoundError) as excinfo:
        _load_resource_manifest(missing)

    message = str(excinfo.value)
    # 说清"是什么、为什么没有、怎么拿回来、不用时怎么办"。
    # 2026-09-10 训练栈搬进主线后改了口径：不再说"去 rank11 拿训练栈"，改成
    # 指向仓库内的重训手册；模型的适用范围也不再写死 Atenolol。
    assert "只对它训练的那个配体有效" in message
    assert "resource_manifest=" in message
    assert "docs/RETRAIN_LOCAL_RESIDUAL.md" in message
    assert "false" in message
    assert (ROOT_DOCS := _REPO_ROOT / "docs/RETRAIN_LOCAL_RESIDUAL.md").is_file(), ROOT_DOCS
    assert "{path}" not in RESOURCE_MISSING_HINT.format(path=str(missing))
