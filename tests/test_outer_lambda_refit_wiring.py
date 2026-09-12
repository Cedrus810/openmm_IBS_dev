"""换配体时主线自己重训 outer-λ 残差（EXP-033 §5 P1）的接线契约。

这条线上会静默坏掉的是**时序**：训练帧源 `pre_equilibration.dcd` 是基线预平衡的
产物，而残差运行时原来在 pipeline 构造期就要建好。挂错位置不会报错——只会找不到
轨迹，或者更糟：在预平衡之前拿旧轨迹训。
"""
import inspect

import pytest

import runabfe

pytestmark = pytest.mark.cpu_only


def test_missing_or_unreadable_manifest_defers_instead_of_raising(tmp_path):
    """读不动冻结 manifest 不是错误，是「这个配体要先重训」的信号。"""
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


def test_refit_runs_after_pre_equilibration_and_before_sampling():
    """重训必须落在 `resolve_boresch_restraint`（内含基线预平衡）之后、采样之前。"""
    source = inspect.getsource(runabfe.main)
    equilibration = source.index("resolve_boresch_restraint(config, pipeline)")
    refit = source.index("_refit_outer_lambda_residual_for_this_ligand(")
    sampling = source.index("pipeline.run_full_pipeline(")
    assert equilibration < refit < sampling, (equilibration, refit, sampling)


def test_frozen_path_is_untouched_when_the_manifest_covers_the_ligand():
    """覆盖得到时仍然走构造期那条原路 —— 冻结产物的行为一个字节都不变。"""
    source = inspect.getsource(runabfe.main)
    assert "if config.outer_lambda_local_residual_ibs and not _residual_refit_pending:" in source
    # 延迟绑定只在 pending 时发生
    deferred = source.index("if _residual_refit_pending:")
    assert source.index("attach_residual_sampling_runtime(") > deferred


def test_deferred_binding_goes_through_the_shared_fail_closed_entry():
    """延迟绑定必须用 pipeline 的统一入口，不许在 runabfe 里重抄一遍赋值。

    重抄就会绕过 `no_residual_twin` 与 plugin identity 那两道 fail-closed。
    """
    import abfe_pipeline

    source = inspect.getsource(runabfe.main)
    assert "pipeline.attach_residual_sampling_runtime(" in source
    assert "pipeline.residual_sampling_enabled =" not in source
    attach = inspect.getsource(abfe_pipeline.ABFEPipeline.attach_residual_sampling_runtime)
    assert "no_residual_twin" in attach and "plugin/model identity" in attach
    # 构造期与延迟绑定共用同一条实现
    ctor = inspect.getsource(abfe_pipeline.ABFEPipeline.__init__)
    assert "self.attach_residual_sampling_runtime(" in ctor


def test_training_trajectory_is_the_fixed_pre_equilibration_name():
    """帧源写死固定文件名，且不设帧数下限/抽稀。"""
    source = inspect.getsource(runabfe._refit_outer_lambda_residual_for_this_ligand)
    assert 'os.path.join(output_dir, "pre_equilibration.dcd")' in source
    assert "max_frames" not in source
    assert "min_frames" not in source


def test_both_vdw_legs_read_the_same_resource_manifest():
    """两段 vdW（复合物腿 + 溶剂腿）必须读同一个 manifest 变量。

    溶剂腿原来读的是 `config.outer_lambda_resource_manifest`，也就是**冻结**那份。
    重训产出的 manifest 到不了它 ⟹ 新配体会在溶剂腿撞身份闸门，而且是在复合物腿
    整段跑完之后才炸。权重绑配体不绑体系（docs/RETRAIN_LOCAL_RESIDUAL.md），
    所以一次重训两条腿共用，不该重训第二次。
    """
    source = inspect.getsource(runabfe.main)
    assert source.count("resource_manifest=_residual_resource_manifest") == 3
    assert 'resource_manifest=getattr(config, "outer_lambda_resource_manifest"' not in source
    # 重训只发生一次：只有一个调用点
    assert source.count("_refit_outer_lambda_residual_for_this_ligand(") == 1
    # 溶剂腿仍然挂在复合物腿之后，所以重训结果一定已经写进那个变量
    refit = source.index("_refit_outer_lambda_residual_for_this_ligand(")
    solvent = source.index('leg_name="solvent"')
    assert refit < solvent


def test_switch_on_without_a_bound_runtime_fails_instead_of_running_baseline():
    """开关开着却没绑上运行时 ⟹ **报错**，不许静默跑 baseline。

    fail-closed 的位置从「资源缺失」挪到了「训完了仍然没有」：缺冻结 manifest 是正常
    初始状态（输入时谁都没有这份蒸馏权重），而"开着开关跑完一整轮、报告一切正常、
    残差一次没生效"才是必须拦的那种失败。
    """
    source = inspect.getsource(runabfe.main)
    for runtime in ("outer_lambda_runtime", "outer_lambda_runtime_solv"):
        assert (
            f"if config.outer_lambda_local_residual_ibs and {runtime} is None:" in source
        ), runtime
    # 两道断言都必须在各自那条腿开始采样**之前**
    cplx_guard = source.index("and outer_lambda_runtime is None:")
    cplx_run = source.index("complex_results = pipeline.run_full_pipeline(")
    solv_guard = source.index("and outer_lambda_runtime_solv is None:")
    solv_run = source.index("solv_results = pipeline_solv.run_full_pipeline(")
    assert cplx_guard < cplx_run
    assert solv_guard < solv_run


def test_only_modes_are_rejected_up_front_rather_than_silently_dropping_residual():
    """`--only-charging` / `--only-attachment` 与残差互斥，且在入口就拒绝。"""
    source = inspect.getsource(runabfe.main)
    rejection = source.index("不能用于只跑 charging/attachment 的入口")
    first_sampling = source.index("complex_results = pipeline.run_full_pipeline(")
    assert rejection < first_sampling
