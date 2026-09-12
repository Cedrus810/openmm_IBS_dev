"""`local_residual.autofit` 的离线契约。

⚠️ **这条链已于 2026-09-11 从 runabfe 摘除**（理由见 `local_residual/autofit.py`
末尾"为什么撤"）。模块留档，这些测试跟着留，用来锁住"如果将来有人在离线场景下
再用它，这些不变量还成立"。**它们不构成"这条链可以接回生产"的任何背书。**

真正的验收只能上机（EXP-033 §4：离线 gap-variance 不是验收口径）。这里钉住的是
不上机也必须成立的四件事：
  1. 训练用的 A_k 与运行时 `OuterLambdaController.envelope` 是**同一条**曲线；
  2. 41 原子时推导出来的 config/容量与出厂那份逐字段相同（新链不改老路径）；
  3. 元素覆盖检查对两个 teacher 都 fail-closed；
  4. ledger 的形状/定义满足 `build_dataset_v1` 的入口校验。
"""

import json
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parents[1]
_FAKE_SHA = "0" * 64


def test_training_a_k_matches_the_runtime_envelope():
    from local_residual.autofit import a_k_schedule
    from outer_lambda_neural_basis import OuterLambdaController

    lambdas = [1.0, 0.9, 0.5, 0.1, 0.0]
    delta, a_k = a_k_schedule(lambdas)
    controller = OuterLambdaController(
        # enabled=False 只是为了免掉 bases/safety 那套构造；envelope() 是同一个方法，
        # 与运行时（openmm_plugin.py:554 那份 enabled=True 的 controller）逐字一致。
        enabled=False,
        stage="vanishing",
        baseline_potential="softcore",
        endpoint_tolerance=1.0e-12,
        coefficients=(1.0,),
        max_abs_coefficient=10.0,
    )
    expected = [controller.envelope(value) for value in lambdas]
    assert a_k == expected
    assert delta == [expected[i + 1] - expected[i] for i in range(len(expected) - 1)]
    # 端点必须严格 0，否则残差会在完全解耦/完全耦合端点上改哈密尔顿量。
    assert a_k[0] == 0.0 and a_k[-1] == 0.0


def test_derived_config_reproduces_the_frozen_atenolol_config():
    from local_residual.autofit import derived_capacities
    from local_residual.softlift import derived_r1_config, primary_r1_config

    frozen = primary_r1_config(_FAKE_SHA)
    capacities = derived_capacities(frozen.n_ligand_atoms)
    assert capacities == {
        "max_environment_atoms": 320,
        "max_edges": 2048,
        "max_neighbors_per_ligand": 80,
    }
    derived = derived_r1_config(
        _FAKE_SHA,
        type_vocabulary=frozen.type_vocabulary,
        n_ligand_atoms=frozen.n_ligand_atoms,
        **capacities,
    )
    assert derived == frozen


def test_derived_config_scales_capacity_and_b_max_for_a_bigger_ligand():
    from local_residual.autofit import derived_capacities
    from local_residual.softlift import derived_r1_config

    capacities = derived_capacities(82)
    assert capacities["max_environment_atoms"] == 640
    assert capacities["max_edges"] == 4096
    # 每个配体原子的邻居上限与配体大小无关，不缩放。
    assert capacities["max_neighbors_per_ligand"] == 80
    derived = derived_r1_config(
        _FAKE_SHA,
        type_vocabulary=(1, 6, 7, 8, 9),
        n_ligand_atoms=82,
        **capacities,
    )
    # 输出头是"对原子求和后再套全局 tanh"，b_max 不跟着尺寸走就直接进死区。
    assert derived.b_max_reduced == pytest.approx(20.0)
    assert derived.rung == "R1" and derived.n_channels == 1 and derived.pair_dim == 0


@pytest.mark.parametrize(
    "teacher,missing_z",
    [("mace-off24-medium", 11), ("ubio-molfm-omol25", 92)],
)
@pytest.mark.needs_gpu  # 需要从真实 MACE 模型文件读 z_table（GPU 节点才有）
def test_element_coverage_fails_closed_per_teacher(teacher, missing_z):
    from local_residual.autofit import AutofitError, check_element_coverage

    assert check_element_coverage([8, 1, 6, 6], None) == (1, 6, 8)
    with pytest.raises(AutofitError, match=str(missing_z)):
        check_element_coverage([1, 6, missing_z], teacher)


def test_unknown_teacher_is_refused_rather_than_guessed():
    from local_residual.autofit import AutofitError, check_element_coverage

    with pytest.raises(AutofitError, match="无法确定 teacher"):
        check_element_coverage([1, 6], "some-model-nobody-measured")
    # 显式给了 z-table 就能用——新 teacher 靠传值接入，不靠猜。
    assert check_element_coverage([1, 6], "x", extra_z_tables={"x": [1, 6, 7]}) == (1, 6)


def test_ledger_shapes_and_definitions_match_the_dataset_entry_contract(tmp_path, monkeypatch):
    """ledger 三个数组的形状/定义正是 `build_dataset_v1` 进门校验的那三条。"""

    import ibs_engine
    from local_residual import autofit

    n_states, per_partition = 4, 5
    frames_total = per_partition * 3
    rng = np.random.default_rng(20260911)
    fake_u_kn = rng.normal(size=(n_states, frames_total))

    class _FakeAnalyzer:
        def __init__(self, temperature):
            self.temperature = temperature
            self._last_n_k = np.array([per_partition] * 3, dtype=int)

        def compute_u_kn(self, **kwargs):
            assert kwargs["lambdas_coul"] == [0.0] * n_states
            return fake_u_kn

    monkeypatch.setattr(ibs_engine, "TraditionalMBARAnalyzer", _FakeAnalyzer)
    sweep = {
        "trajectory_paths": [str(tmp_path / f"p{i}.dcd") for i in range(3)],
        "frame_states": [
            np.array([0, 1, 2, 3, 0], dtype=np.int64) for _ in range(3)
        ],
        "lambdas_vdw": np.linspace(1.0, 0.0, n_states).tolist(),
    }
    runs = autofit.write_ledgers(
        system=None,
        topology=None,
        positions=None,
        box_vectors=None,
        ligand_indices=[0, 1],
        temperature_kelvin=300.0,
        platform_name="CPU",
        sweep=sweep,
        out_dir=tmp_path,
        log=lambda message: None,
    )
    assert len(runs) == 3
    for partition, run in enumerate(runs):
        with np.load(run["ledger_path"]) as ledger:
            gaps = ledger["adjacent_gap_reduced"]
            weights = ledger["log_importance_unnormalized"]
            frame_index = ledger["frame_index"]
        report = json.loads(Path(run["ledger_report_path"]).read_text(encoding="utf-8"))
        assert report["frame_count"] == per_partition
        # build_dataset_v1 的三条硬校验
        assert np.array_equal(frame_index, np.arange(per_partition))
        assert gaps.shape == (per_partition, n_states - 1)
        assert weights.shape == (per_partition, n_states)
        # 定义本身：gap 是目标态之差，log_importance 在采样态那一列恒为 0
        target_u = fake_u_kn[:, partition * per_partition : (partition + 1) * per_partition].T
        assert np.allclose(gaps, np.diff(target_u, axis=1))
        sampled = sweep["frame_states"][partition]
        assert np.allclose(weights[np.arange(per_partition), sampled], 0.0)


def test_frame_count_mismatch_fails_closed(tmp_path, monkeypatch):
    import ibs_engine
    from local_residual import autofit

    class _FakeAnalyzer:
        def __init__(self, temperature):
            self._last_n_k = np.array([4, 4, 4], dtype=int)

        def compute_u_kn(self, **kwargs):
            return np.zeros((2, 12))

    monkeypatch.setattr(ibs_engine, "TraditionalMBARAnalyzer", _FakeAnalyzer)
    sweep = {
        "trajectory_paths": [str(tmp_path / f"p{i}.dcd") for i in range(3)],
        "frame_states": [np.zeros(3, dtype=np.int64) for _ in range(3)],  # 3 != 4
        "lambdas_vdw": [1.0, 0.0],
    }
    with pytest.raises(autofit.AutofitError, match="对不上"):
        autofit.write_ledgers(
            system=None, topology=None, positions=None, box_vectors=None,
            ligand_indices=[0], temperature_kelvin=300.0, platform_name="CPU",
            sweep=sweep, out_dir=tmp_path, log=lambda message: None,
        )


def test_trainer_still_refuses_a_foreign_ligand_without_the_opt_in():
    """老路径不能被新链弄松：不加 --allow-derived-r1-config 仍然拒绝。"""

    import importlib.util

    path = REPO_ROOT / "abfe_scripts" / "train_exp019_softlift_loro.py"
    spec = importlib.util.spec_from_file_location("_trainer_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    budgets = {
        "unique_environment_atoms_hard": 320,
        "directed_edges_hard": 2048,
        "neighbors_per_ligand_hard": 80,
    }
    foreign = np.array([1, 6, 7, 8, 9], dtype=np.int64)
    with pytest.raises(module.SoftLiftTrainError):
        module._config("R1", _FAKE_SHA, foreign, 60, budgets)
    derived = module._config("R1", _FAKE_SHA, foreign, 60, budgets, allow_derived=True)
    assert derived.type_vocabulary == (1, 6, 7, 8, 9)
    assert derived.n_ligand_atoms == 60
class _FakeAtom:
    def __init__(self, z):
        self.element = type("_E", (), {"atomic_number": z})()


class _FakeTopology:
    def __init__(self, zs):
        self._atoms = [_FakeAtom(z) for z in zs]

    def atoms(self):
        return iter(self._atoms)


def test_startup_gate_needs_no_manifest_when_autofitting():
    """autofit 时词表由体系推出来，元素**不可能**超——这道门不该去读冻结产物。"""

    from local_residual.autofit import startup_element_gate

    vocabulary = startup_element_gate(
        topology=_FakeTopology([1, 6, 15, 19, 12, 6]),
        autofit_enabled=True,
        resource_manifest="/nonexistent/manifest.json",
        log=lambda message: None,
    )
    assert vocabulary == (1, 6, 12, 15, 19)


def test_startup_gate_rejects_a_foreign_element_against_the_frozen_vocabulary():
    from local_residual.autofit import AutofitError, startup_element_gate

    manifest = REPO_ROOT / "resources/outer_lambda_local_residual/manifest.json"
    if not manifest.is_file():
        pytest.skip("冻结 R1 资源不在本树上")
    # 出厂词表是 (1,6,7,8,11,16,17)；膜里的 P(15) 不在其中。
    with pytest.raises(AutofitError, match=r"\[15\]"):
        startup_element_gate(
            topology=_FakeTopology([1, 6, 7, 8, 15]),
            autofit_enabled=False,
            log=lambda message: None,
        )
    assert startup_element_gate(
        topology=_FakeTopology([1, 6, 7, 8]), autofit_enabled=False, log=lambda m: None
    ) == (1, 6, 7, 8)


def test_atom_type_index_reports_every_missing_element_at_once():
    from local_residual.openmm_plugin import atom_type_index_for_topology

    with pytest.raises(ValueError, match=r"\[12, 15, 19\]"):
        atom_type_index_for_topology([1, 6, 15, 19, 12], [1, 6, 7, 8])
    assert atom_type_index_for_topology([1, 6, 6], [1, 6, 7]) == [0, 1, 1]


@pytest.mark.needs_gpu  # torch/mace（GPU 版构建）
def test_mace_z_table_is_read_from_the_model_not_from_a_constant():
    """换模型 = 换路径。z-table 从 .model 里读，所以不会跟模型对不上。"""

    import os

    from local_residual.autofit import mace_z_table, resolve_mace_model

    try:
        path = resolve_mace_model("mace-off24-medium")
    except Exception:
        pytest.skip(
            f"本机没有 MACE-OFF24 模型（给 ${'ABFE_MACE_MODEL_DIR'} 或放进 ~/.cache/mace）"
        )
    pytest.importorskip("torch")
    pytest.importorskip("mace")
    assert path.is_file()
    assert mace_z_table(path) == (1, 6, 7, 8, 9, 15, 16, 17, 35, 53)
    # 常量表里**不该**有 MACE —— 有就说明又抄了一份。
    from local_residual.autofit import KNOWN_TEACHER_Z_TABLES

    assert not any("mace" in name.lower() for name in KNOWN_TEACHER_Z_TABLES)
def _tiny_topology_with_elementless_ion(mass_dalton):
    """一个 2 原子拓扑：一个正常的 O，一个**没有元素**、名字是 NA 的粒子。

    4W53 的真实形态：`GromacsTopFile` 没给 45 个钠赋元素，写进 topology.cif 就是
    `type_symbol = ?`。
    """

    import openmm
    from openmm import app, unit

    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("HOH", chain)
    topology.addAtom("O", app.element.oxygen, residue)
    ion_residue = topology.addResidue("NA", chain)
    topology.addAtom("NA", None, ion_residue)
    system = openmm.System()
    system.addParticle(15.999 * unit.dalton)
    system.addParticle(mass_dalton * unit.dalton)
    return topology, system


def test_elementless_ion_is_recovered_from_name_and_mass_agreement():
    from local_residual.openmm_plugin import topology_atomic_numbers

    topology, system = _tiny_topology_with_elementless_ion(22.99)
    assert topology_atomic_numbers(topology, system=system) == [8, 11]


def test_elementless_atom_fails_closed_when_the_two_sources_disagree():
    from local_residual.openmm_plugin import topology_atomic_numbers

    # 名字说钠、质量说别的：两个来源不一致就不许猜。
    topology, system = _tiny_topology_with_elementless_ion(40.08)
    with pytest.raises(RuntimeError, match="无法由"):
        topology_atomic_numbers(topology, system=system)
    # 没有 System 就没有第二个来源，同样不许猜。
    topology, _ = _tiny_topology_with_elementless_ion(22.99)
    with pytest.raises(RuntimeError, match="无法由"):
        topology_atomic_numbers(topology, system=None)


def test_dataset_builder_takes_an_atomic_number_override():
    """mdtraj 对未知元素给 atomic_number=0 且不抛错，所以要能显式覆盖。"""

    import inspect

    from local_residual.softlift_dataset import build_dataset_v1

    signature = inspect.signature(build_dataset_v1)
    assert "atomic_numbers_override" in signature.parameters
    assert signature.parameters["atomic_numbers_override"].default is None


def test_run_script_registers_the_module_before_executing_it(tmp_path, monkeypatch):
    """`@dataclass` 需要 `sys.modules[cls.__module__]`，不注册就炸在 import 期。

    重训链那三个脚本都是 `from __future__ import annotations` + frozen dataclass：
    注解全是字符串，`dataclasses._is_type` 于是要去 sys.modules 取命名空间来解析，
    模块不在表里就 `AttributeError: 'NoneType' object has no attribute '__dict__'`。
    """

    import sys

    from local_residual import autofit

    script = tmp_path / "abfe_scripts" / "fake_stage.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "from dataclasses import dataclass",
                "from pathlib import Path",
                "",
                "@dataclass(frozen=True)",
                "class _Frozen:",
                "    value: Path | None = None",
                "",
                "def main(argv=None):",
                "    open(argv[0], 'w').write('ran')",
                "    return 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(autofit, "_REPO_ROOT", tmp_path)
    marker = tmp_path / "marker.txt"
    autofit._run_script("fake_stage", [str(marker)])
    assert marker.read_text() == "ran"
    # 跑完要摘干净，别在 sys.modules 里留一份
    assert "_autofit_fake_stage" not in sys.modules


@pytest.mark.needs_gpu  # torch（GPU 版构建）
def test_resume_refuses_a_dataset_built_with_different_parameters(tmp_path):
    """续跑只在协议身份一致时复用；参数变了就说清楚怎么清，不静默用旧数据。"""

    from local_residual.autofit import AutofitError, autofit_r1

    topology, system = _tiny_topology_with_elementless_ion(22.99)
    work_dir = tmp_path / "autofit"
    (work_dir / "dataset").mkdir(parents=True)
    (work_dir / "probe_partition0.dcd").write_bytes(b"")
    (work_dir / "dataset" / "softlift_dataset_v1.npz").write_bytes(b"")
    (work_dir / "dataset" / "softlift_dataset_v1_report.json").write_text(
        json.dumps({"protocol_sha256": "f" * 64}), encoding="utf-8"
    )
    with pytest.raises(AutofitError, match="不同参数"):
        autofit_r1(
            system=system,
            topology=topology,
            positions=None,
            box_vectors=None,
            ligand_indices=[0],
            temperature_kelvin=300.0,
            platform_name="CPU",
            output_dir=tmp_path,
            ligand_name="MOL",
            topology_cif=tmp_path / "topology.cif",
            ligand_indices_path=tmp_path / "ligand_indices.json",
            log=lambda message: None,
        )


def _load_exporter():
    import importlib.util
    import sys

    path = REPO_ROOT / "abfe_scripts" / "export_exp025_g1_reference_payload.py"
    spec = importlib.util.spec_from_file_location("_exporter_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _fake_checkpoint(tmp_path, config):
    torch = pytest.importorskip("torch")

    path = tmp_path / "fake.pt"
    torch.save(
        {
            "experiment_id": "EXP-020",
            "held_out_run": 1,
            "seed": 0,
            "state_dict": {},
            "config": dict(config.__dict__),
        },
        path,
    )
    return path


def test_exporter_accepts_a_derived_config_only_with_the_opt_in(tmp_path):
    """换配体时 checkpoint 的 n_ligand_atoms 必然与出厂那份不同，导出不能因此拒绝。"""

    from local_residual.softlift import derived_r1_config, primary_r1_config

    exporter = _load_exporter()
    frozen = primary_r1_config(_FAKE_SHA)
    derived = derived_r1_config(
        _FAKE_SHA,
        type_vocabulary=frozen.type_vocabulary,
        n_ligand_atoms=15,  # 4W53 的配体
        max_environment_atoms=320,
        max_edges=2048,
        max_neighbors_per_ligand=80,
    )
    checkpoint = _fake_checkpoint(tmp_path, derived)
    strict = exporter.ExportInputs(checkpoint=checkpoint, checkpoint_sha256=None)
    with pytest.raises(exporter.ExporterError, match="allow-derived-r1-config"):
        exporter._load_checkpoint(strict)
    relaxed = exporter.ExportInputs(
        checkpoint=checkpoint, checkpoint_sha256=None, allow_derived_r1_config=True
    )
    _payload, config, _sha = exporter._load_checkpoint(relaxed)
    assert config.n_ligand_atoms == 15


def test_exporter_still_refuses_an_architecture_change_even_with_the_opt_in(tmp_path):
    """开关放开的是"体系相关"的那几项，不是架构。"""

    import dataclasses

    from local_residual.softlift import primary_r1_config

    exporter = _load_exporter()
    mutated = dataclasses.replace(primary_r1_config(_FAKE_SHA), n_radial_basis=32)
    inputs = exporter.ExportInputs(
        checkpoint=_fake_checkpoint(tmp_path, mutated),
        checkpoint_sha256=None,
        allow_derived_r1_config=True,
    )
    with pytest.raises(exporter.ExporterError, match="架构"):
        exporter._load_checkpoint(inputs)


def test_exporter_can_take_the_system_as_the_authoritative_element_source():
    """mmCIF 对某些离子不写 type_symbol，mdtraj 会静默给 atomic_number=0。"""

    exporter = _load_exporter()
    assert "system" in {field.name for field in __import__("dataclasses").fields(exporter.ExportInputs)}
    source = __import__("inspect").getsource(exporter._canonical_batch)
    assert "topology_atomic_numbers" in source


def test_probe_ladder_defaults_to_a_window_not_the_whole_path():
    """整条梯子横跨采样 = 鬼影构型被重加权到耦合态，r^-12 爆炸（2026-09-11 实测）。"""

    from local_residual import autofit

    assert autofit.DEFAULT_PROBE_LAMBDA_MAX == 1.0
    assert autofit.DEFAULT_PROBE_LAMBDA_MIN >= 0.5, "区间下端伸进鬼影区就会退化"
    span = autofit.DEFAULT_PROBE_LAMBDA_MAX - autofit.DEFAULT_PROBE_LAMBDA_MIN
    assert span <= 0.6, "探针区间必须是窗口形状的，不能接近整条梯子"
    # 54 帧实测 held-out 改善只有 +0.000；别再把采样量砍回去。
    assert autofit.DEFAULT_PROBE_STATES * autofit.DEFAULT_FRAMES_PER_STATE >= 400
    assert autofit.DEFAULT_FRAMES_PER_STATE % 3 == 0, "三个 LORO 分区要能均分"


def test_reweighting_sanity_gate_catches_a_whole_path_probe(tmp_path):
    """这道门要在 ledger 阶段就说清楚，而不是死在 loss 的 sum-to-one 上。"""

    from local_residual.autofit import AutofitError, _check_reweighting_sanity

    def _ledger(name, worst):
        path = tmp_path / name
        weights = np.zeros((10, 4), dtype=np.float64)
        weights[0, 0] = worst
        np.savez(path, log_importance_unnormalized=weights)
        return {"ledger_path": str(path)}

    # 单帧退化（10% 里的 1 帧）在预算内
    _check_reweighting_sanity(
        [_ledger(f"ok{i}.npz", 1.0e3) for i in range(3)], log=lambda message: None
    )
    with pytest.raises(AutofitError, match="不是窗口形状"):
        _check_reweighting_sanity(
            [_ledger(f"bad{i}.npz", -1.0e13) for i in range(3)],
            log=lambda message: None,
        )


def test_unsupported_frames_are_skipped_and_renumbered_contiguously():
    """跳帧之后帧号必须重新连续，否则下游 validate 会拒绝整份数据集。"""

    import inspect

    from local_residual import softlift_dataset

    source = inspect.getsource(softlift_dataset.build_dataset_v1)
    assert "skip_unsupported_frames" in source
    assert "frame_index - skipped_unsupported.get(run_id, 0)" in source
