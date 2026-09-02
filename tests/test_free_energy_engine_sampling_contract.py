"""P2′ 共用采样契约（RBFE 计划 §4.1、REMD 计划 §4.2/§4.3）。

这层的存在意义是让 ABFE 的去耦 System 和 RBFE 的 hybrid System 走同一条采样路径，
**且不需要 OpenMM 8.6**。测试全部用假 sampler，不建任何 Context。
"""

from __future__ import annotations

import pytest

import free_energy_engine as fee

LEGACY = fee.resolve_remd_backend("legacy")


def _states(n=3, key="lambda_vdw"):
    return [fee.ThermodynamicStateSpec(i, {key: 1.0 - i / (n - 1)}) for i in range(n)]


def _request(**kw):
    n = kw.pop("n_states", 3)
    states = kw.pop("states", _states(n))
    defaults = dict(
        system=object(),
        topology=object(),
        states=states,
        initial_positions=[object()] * len(states),
        initial_box_vectors=[object()] * len(states),
        total_md_steps=9000,
        exchange_interval=3000,
        save_interval=1000,
        output_dir="/tmp/out",
        stage_name="vanishing",
        caller_protocol_fingerprint="fp1",
    )
    defaults.update(kw)
    return fee.SamplingRequest(**defaults)


class _FakeSampler:
    exchange_diagnostics = {"accepted_edges": 7}

    def __init__(self, n_files=3, checkpoint_kind="none"):
        self._n = n_files
        self.checkpoint_kind = checkpoint_kind
        self.calls = []

    def run(self, n_steps, exchange_interval, save_interval, stage_name):
        self.calls.append((n_steps, exchange_interval, save_interval, stage_name))
        return [f"{stage_name}_rep{i}.dcd" for i in range(self._n)]


# ---------------------------------------------------------------------------
# StepPlan：§4.2「不增跑、漏跑或额外交换」
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "total,interval,iters,tail",
    [(9000, 3000, 3, 0), (10000, 3000, 3, 1000), (100, 1000, 0, 100), (0, 1000, 0, 0)],
)
def test_step_plan_accounts_for_every_md_step(total, interval, iters, tail):
    plan = fee.resolve_step_plan(total, interval, 500)
    assert (plan.full_iterations, plan.tail_md_steps) == (iters, tail)
    assert plan.accounted_md_steps == total  # 逐步对账


def test_step_plan_does_not_round():
    """不整除时如实产出尾段，**不静默四舍五入**（§4.2）。"""
    plan = fee.resolve_step_plan(10_000, 3_000, 1_000)
    assert plan.has_tail and plan.tail_md_steps == 1000
    # 四舍五入到 4 个 iteration 会多跑 2000 步；到 3 个会少跑 1000 步。两者都不许。
    assert plan.full_iterations * plan.exchange_interval != 10_000


@pytest.mark.parametrize("bad", [(-1, 1000, 1000), (1000, 0, 1000), (1000, 1000, 0)])
def test_step_plan_rejects_invalid_units(bad):
    with pytest.raises(fee.SamplingContractError):
        fee.resolve_step_plan(*bad)


# ---------------------------------------------------------------------------
# SamplingRequest 校验
# ---------------------------------------------------------------------------


def test_valid_request_passes():
    _request().validate()


def test_at_least_two_states():
    """继承本项目既有的「至少两个状态」输入校验（§6 验证清单第 5 条）。"""
    with pytest.raises(fee.SamplingContractError, match="2 个热力学状态"):
        _request(states=[fee.ThermodynamicStateSpec(0, {"lambda_vdw": 1.0})]).validate()


def test_state_indices_must_be_contiguous_and_ordered():
    """顺序即身份——rep{i} 对应 states[i]，乱序会让轨迹和状态对不上。"""
    states = [fee.ThermodynamicStateSpec(i, {"x": 0.0}) for i in (0, 2, 1)]
    with pytest.raises(fee.SamplingContractError, match="连续升序"):
        _request(states=states).validate()


def test_all_states_must_share_the_same_parameter_keys():
    """🔑 §4：官方状态是 list[dict]，要求每个 dict 键集合相同。

    键集合不同时官方 API **不会替你报错**——某些副本会缺参数，静默跑出错的结果。
    """
    states = [
        fee.ThermodynamicStateSpec(0, {"lambda_vdw": 1.0, "lambda_coul": 1.0}),
        fee.ThermodynamicStateSpec(1, {"lambda_vdw": 0.5}),
    ]
    with pytest.raises(fee.SamplingContractError, match="相同的参数键集合"):
        _request(states=states).validate()
    # 而且要指出**是哪个键**只出现在部分状态里
    try:
        _request(states=states).validate()
    except fee.SamplingContractError as exc:
        assert "lambda_coul" in str(exc)


def test_empty_parameter_set_rejected():
    states = [fee.ThermodynamicStateSpec(i, {}) for i in range(2)]
    with pytest.raises(fee.SamplingContractError, match="参数表为空"):
        _request(states=states).validate()


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_state_parameters_rejected(bad):
    states = [
        fee.ThermodynamicStateSpec(0, {"lambda_vdw": 1.0}),
        fee.ThermodynamicStateSpec(1, {"lambda_vdw": bad}),
    ]
    with pytest.raises(fee.SamplingContractError, match="非有限值"):
        _request(states=states).validate()


def test_boolean_parameters_rejected():
    """bool 是 int 的子类，不显式排掉就会被当成 0/1 悄悄接受。"""
    states = [
        fee.ThermodynamicStateSpec(0, {"lambda_vdw": 1.0}),
        fee.ThermodynamicStateSpec(1, {"lambda_vdw": True}),
    ]
    with pytest.raises(fee.SamplingContractError, match="不是实数"):
        _request(states=states).validate()


def test_every_replica_needs_its_own_initial_configuration():
    """§4.1：为各副本创建独立的初始位置／速度 State，
    **不能把官方默认克隆初始构型当作项目预热已完成**。"""
    with pytest.raises(fee.SamplingContractError, match="initial_positions"):
        _request(initial_positions=[object()]).validate()


def test_initial_velocities_length_checked_when_given():
    with pytest.raises(fee.SamplingContractError, match="initial_velocities"):
        _request(initial_velocities=[object()] * 2).validate()


@pytest.mark.parametrize(
    "kw", [{"temperature_kelvin": 0.0}, {"pressure_bar": -1.0}, {"timestep_fs": 0.0},
           {"total_md_steps": 0}]
)
def test_invalid_ensemble_settings_rejected(kw):
    with pytest.raises(fee.SamplingContractError):
        _request(**kw).validate()


@pytest.mark.parametrize(
    "kw", [{"output_dir": ""}, {"stage_name": ""}, {"caller_protocol_fingerprint": ""}]
)
def test_identity_fields_are_mandatory(kw):
    """没有指纹的采样产物事后认不出是哪个 Hamiltonian 跑的。"""
    with pytest.raises(fee.SamplingContractError):
        _request(**kw).validate()


# ---------------------------------------------------------------------------
# run_sampling
# ---------------------------------------------------------------------------


def test_run_sampling_passes_the_resolved_step_plan_to_the_sampler():
    sampler = _FakeSampler()
    fee.run_sampling(_request(), sampler, LEGACY)
    assert sampler.calls == [(9000, 3000, 1000, "vanishing")]


def test_artifacts_are_per_state_not_per_replica():
    """🔑 REMD 计划 §2 点名：`{stage}_rep{i}.dcd` 的内容按**热力学状态**分流，
    新后端不得改成物理 replica 轨迹。"""
    art = fee.run_sampling(_request(), _FakeSampler(), LEGACY)
    assert art.n_states == 3
    assert art.to_provenance()["trajectory_files_are_per_state_not_per_replica"] is True


def test_trajectory_count_must_match_state_count():
    with pytest.raises(fee.SamplingContractError, match="轨迹文件数"):
        fee.run_sampling(_request(), _FakeSampler(n_files=2), LEGACY)


def test_sampler_returning_none_is_rejected():
    class _Silent:
        def run(self, **kw):
            return None

    with pytest.raises(fee.SamplingContractError, match="返回 None"):
        fee.run_sampling(_request(), _Silent(), LEGACY)


def test_object_without_run_is_rejected():
    with pytest.raises(fee.SamplingContractError, match="必须提供 run"):
        fee.run_sampling(_request(), object(), LEGACY)


def test_official_backend_path_is_not_silently_faked():
    """官方适配器没写，就必须抛 NotImplementedError，不能返回一个看着正常的产物。"""
    official = fee.BackendResolution(
        requested="openmm",
        resolved="openmm",
        reason="手工构造",
        version_info=fee.OpenMMVersionInfo(raw="8.6.0"),
        capability=fee.RemdCapabilityReport(available=True),
        adapter_implemented=True,
        exchange_scheme=fee.EXCHANGE_SCHEME_OFFICIAL,
    )
    with pytest.raises(NotImplementedError, match="P1"):
        fee.run_sampling(_request(), _FakeSampler(), official)


def test_checkpoint_kind_is_explicit():
    """§4.3：官方 reporter 的 checkpoint 是 serialized State，
    **不能视为包含完整 RNG 的二进制 Context checkpoint**——所以类型必须显式记录。"""
    art = fee.run_sampling(
        _request(), _FakeSampler(checkpoint_kind="coordinate_snapshot"), LEGACY
    )
    assert art.checkpoint_kind == "coordinate_snapshot"
    assert art.to_provenance()["checkpoint_kind"] == "coordinate_snapshot"


def test_unknown_checkpoint_kind_rejected():
    with pytest.raises(fee.SamplingContractError, match="checkpoint_kind"):
        fee.run_sampling(_request(), _FakeSampler(checkpoint_kind="magic"), LEGACY)


def test_artifacts_provenance_carries_backend_and_steps():
    prov = fee.run_sampling(_request(), _FakeSampler(), LEGACY).to_provenance()
    assert prov["remd_backend_resolved"] == "legacy"
    assert prov["remd_exchange_scheme"] == fee.EXCHANGE_SCHEME_LEGACY
    assert prov["total_md_steps"] == 9000
    assert prov["tail_md_steps"] == 0
    assert prov["caller_protocol_fingerprint"] == "fp1"


def test_diagnostics_are_passed_through_untouched():
    """engine 不解释诊断内容——它不知道 ABFE 还是 RBFE 的诊断长什么样。"""
    art = fee.run_sampling(_request(), _FakeSampler(), LEGACY)
    assert art.diagnostics == {"accepted_edges": 7}


def test_engine_never_imports_ibs_engine():
    """依赖方向硬约束：shared engine 不得反向 import 化学层。"""
    from pathlib import Path

    source = Path(fee.__file__).read_text(encoding="utf-8")
    # 只看真正的 import 语句行——注释/文档串里提到这些模块名是**应该**的
    # （那正是在说明依赖方向），不能拿散文当违规。
    import_lines = [
        line.strip()
        for line in source.splitlines()
        if line.lstrip().startswith(("import ", "from "))
        and not line.lstrip().startswith("#")
    ]
    for banned in ("ibs_engine", "abfe_pipeline", "abfe_core", "rbfe_core", "rbfe_pipeline"):
        offenders = [ln for ln in import_lines if banned in ln]
        assert not offenders, f"free_energy_engine 不得 import {banned}：{offenders}"
