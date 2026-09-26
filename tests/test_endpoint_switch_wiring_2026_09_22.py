"""独立端点开关的两端必须读同一个 kwarg（2026-09-22）。

以前：`_run_stage2_once` 逐个具名转发参数时漏了 `stage2_independent_endpoint`，
于是 `run_full_pipeline(stage2_independent_endpoint=True)` 打不开这条路径；
而 `runabfe` 的 `--analyze` 回退路径**无条件**排掉末窗并要 endpoint artifact，
默认 run 根本不产那个文件 ⟹ 默认 run 复算必抛错。两处是同一个开关的两端。
"""

import inspect
import json

import numpy as np
import openmm
import pytest
from openmm import unit

import abfe_pipeline
import ibs_engine as ie
import runabfe
from production_artifact_fixtures import identify_window


pytestmark = pytest.mark.cpu_only


def test_stage2_call_forwards_the_endpoint_switch():
    src = inspect.getsource(abfe_pipeline)
    i = src.index("def _run_stage2_once(")
    call = src[i:src.index("def ", i + 10)]
    assert 'stage2_independent_endpoint=kwargs.get(' in call, (
        "_run_stage2_once 必须把 stage2_independent_endpoint 透传给 "
        "_run_dual_lambda_stage，否则 6615 那个开关永远收不到 True"
    )


def test_analyze_path_reads_the_same_switch():
    src = inspect.getsource(runabfe._analyze_dual_leg_artifacts)
    assert 'kwargs.get("stage2_independent_endpoint", False)' in src, (
        "--analyze 必须按 run_config 里的同一个开关决定是否排末窗"
    )
    assert 'excluded = {len(ranges)-1} if kind == "vdw" else set()' not in src, (
        "不得无条件排掉 vdW 末窗：默认 run 没有独立端点段，排掉就等于少算一段 λ"
    )
    # 开关打开时仍必须 fail-closed：缺 artifact 不许退回纯 IBS 末窗。
    assert "回退分析拒绝用纯 IBS 窗口" in src


@pytest.fixture
def default_leg(tmp_path):
    """一棵**默认协议**的 dual-leg 产物树：两个 vanishing 窗口都有数据，没有端点段。

    每个窗口的合成数据自由能差恰好 0.5 kJ/mol ⟹ 完整两窗 = 1.0。
    少算末窗就会得到 0.5，所以这个数字同时钉住「末窗有没有被排掉」。
    """
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    system = openmm.System()
    system.addParticle(12.)
    identity = abfe_pipeline.constraint_identity_fingerprint(system, [0])
    paths = {"decharging": ([1., 0.], [[0, 2]]), "vanishing": ([1., .5, 0.], [[0, 2], [1, 3]])}
    protocols = {}
    for name, (lambdas, ranges) in paths.items():
        payload = dict(abfe_pipeline._stage_analysis_protocol_versions(name), stage_name=name,
                       temperature_K=300., constraint_identity=identity,
                       decharge_method="pme" if name == "decharging" else "n/a",
                       sampling_repair_policy="non_mutating_v1", boresch_params=None,
                       # 默认 run：kwargs 里**没有** stage2_independent_endpoint
                       run_config={"n_steps_per_window": 1000, "kwargs": {}},
                       final_gate_thresholds={})
        protocols[name] = abfe_pipeline._protocol_fingerprint(payload)
        (checkpoints / f"preopt_dual_{name}.json").write_text(
            json.dumps(dict(lambdas_var=lambdas, window_ranges=ranges)))

    dec = dict(stage="decharging", total_delta_G=2., total_error=.1,
               analysis_status="ANALYSIS_COMPLETE", analysis_incomplete_reasons=[],
               precision_status="UNMEASURED", precision_evidence={},
               min_overlap=.8, min_overlap_threshold=.03, diagnostics={})
    abfe_pipeline.ABFEPipeline._populate_stage_diagnostics(dec)
    (checkpoints / "stage1_decharging.json").write_text(json.dumps(
        abfe_pipeline.ABFEPipeline._build_stage_cache_payload(
            "decharging", dec, 2, protocols["decharging"], *paths["decharging"])))

    context = dict(stage_protocol_keys=protocols, constraint_identity=identity,
                   charge_treatment="neutral", charge_transfer_reservoir_correction=None,
                   co_alchemical_ion_runtime_identity=None,
                   ligand_conformer_diagnostics={"status": "fixture-evidence"})
    (checkpoints / "analysis_context.json").write_text(
        json.dumps(abfe_pipeline._protocol_fingerprint(context)))

    stage = tmp_path / "vanishing"
    stage.mkdir()
    rng = np.random.default_rng(83)
    for index, lambdas_vdw in ((0, [1., .5]), (1, [.5, 0.])):
        common = rng.normal(scale=.01, size=400)
        np.save(stage / f"dual_window_{index}_vdw_energies.npy", np.array([common, common + .5]))
        np.save(stage / f"dual_window_{index}_vdw_bias.npy", common)
        np.save(stage / f"dual_window_{index}_vdw_base.npy", common * 100)
        identify_window(stage, checkpoints, index, [0., 0.], lambdas_vdw,
                        stage_key=protocols["vanishing"], f_k=[0., .5])
    assert not (checkpoints / abfe_pipeline.ENDPOINT_ANALYSIS_FILENAME).exists()
    return tmp_path


def test_default_run_can_be_reanalyzed_with_every_window(default_leg):
    """默认 run（没开独立端点段）必须能复算，且末窗必须在里面。

    修复前：无条件排末窗 + 无条件要 endpoint artifact ⟹
    RuntimeError「回退分析拒绝用纯 IBS 窗口产出 vanishing 主值」。
    """
    result = runabfe._analyze_dual_leg_artifacts(str(default_leg), 300 * unit.kelvin)
    vanishing = result["stage_results"]["vanishing"]
    assert vanishing["total_delta_G"] == pytest.approx(1.0, abs=1e-6), (
        "两窗各 0.5；拿到 0.5 说明末窗又被排掉了"
    )
