"""outer-λ 残差的接线契约（2026-10-17 改）。

以前 manifest 覆盖不到配体时，主线会在基线预平衡之后用 `pre_equilibration.dcd` 自动闭式重训
（EXP-033 §5 P1）。那批帧和目标态没有重叠，权重塌到 1 帧，B 是噪声，所以这条路删了：
覆盖不到 ⟹ 入口就拒绝，提示用 tools/refit_outer_residual_from_stage2.py 从一次不开 outer 的
run 的 stage2 帧重训（docs/design/PLAN_outer_center_selection_2026-10-17.md）。
"""
import inspect

import pytest

import runabfe

pytestmark = pytest.mark.cpu_only


def test_missing_or_unreadable_manifest_counts_as_not_covered(tmp_path):
    assert (
        runabfe._frozen_residual_manifest_covers_ligand(
            tmp_path / "nope.json", object(), [0, 1, 2], object()
        )
        is False
    )
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert (
        runabfe._frozen_residual_manifest_covers_ligand(broken, object(), [0], object())
        is False
    )


def test_uncovered_ligand_is_refused_before_any_sampling():
    """覆盖不到 ⟹ 抛错，而且在预平衡（resolve_boresch_restraint）和采样之前。"""
    source = inspect.getsource(runabfe.main)
    check = source.index("if not _frozen_residual_manifest_covers_ligand(")
    refusal = source.index("tools/refit_outer_residual_from_stage2.py", check)
    assert check < refusal < source.index("resolve_boresch_restraint(config, pipeline)")
    assert refusal < source.index("pipeline.run_full_pipeline(")


def test_pre_equilibration_refit_path_is_gone():
    source = inspect.getsource(runabfe)
    assert "_refit_outer_lambda_residual_for_this_ligand" not in source
    assert "_residual_refit_pending" not in source
    assert "closed_form_refit" not in source


def test_runtime_is_bound_at_construction_through_the_shared_entry():
    """绑定走 pipeline 的统一入口，不许在 runabfe 里重抄赋值（会绕过 no_residual_twin /
    plugin identity 两道 fail-closed）。"""
    import abfe_pipeline

    source = inspect.getsource(runabfe.main)
    assert "pipeline.residual_sampling_enabled =" not in source
    attach = inspect.getsource(abfe_pipeline.ABFEPipeline.attach_residual_sampling_runtime)
    assert "no_residual_twin" in attach and "plugin/model identity" in attach
    ctor = inspect.getsource(abfe_pipeline.ABFEPipeline.__init__)
    assert "self.attach_residual_sampling_runtime(" in ctor


def test_only_the_complex_leg_reads_the_resource_manifest():
    """只有复合物腿建 outer 运行时；溶剂腿不开 outer（2026-10-19 用户定）。"""
    source = inspect.getsource(runabfe.main)
    assert source.count("resource_manifest=_residual_resource_manifest") == 1
    assert 'resource_manifest=getattr(config, "outer_lambda_resource_manifest"' not in source
    assert 'leg_name="solvent",\n            platform_name=config.platform,\n            # 见上面' not in source
    assert "outer_lambda_runtime_solv = None" in source
    assert "outer_lambda_runtime_solv = build_outer_lambda_local_residual_runtime(" not in source


def test_switch_on_without_a_bound_runtime_fails_instead_of_running_baseline():
    """开关开着却没绑上复合物腿运行时 ⟹ **报错**，不许静默跑 baseline；断言在采样之前。"""
    source = inspect.getsource(runabfe.main)
    assert "if config.outer_lambda_local_residual_ibs and outer_lambda_runtime is None:" in source
    assert source.index("and outer_lambda_runtime is None:") < source.index(
        "complex_results = pipeline.run_full_pipeline(")
    assert "and outer_lambda_runtime_solv is None:" not in source


def test_only_modes_are_rejected_up_front_rather_than_silently_dropping_residual():
    """`--only-charging` / `--only-attachment` 与残差互斥，且在入口就拒绝。"""
    source = inspect.getsource(runabfe.main)
    rejection = source.index("不能用于只跑 charging/attachment 的入口")
    first_sampling = source.index("complex_results = pipeline.run_full_pipeline(")
    assert rejection < first_sampling
