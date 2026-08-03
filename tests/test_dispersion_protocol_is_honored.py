"""B6 接线：声明的 `dispersion_protocol` 必须真的改变 LRC 行为，而不只是进 provenance。

这条测试存在的理由：B6 最初只做了**校验层**——`resolve_dispersion_protocol()` 会
接受 `ff_native_isotropic_lrc`，但没有任何代码消费它，`build_ibs_dual_system` 照旧
按 `potential_type` 决定要不要算 `ibs_wrapper.lj_tail_lrc_coeff_kj_mol`。
结果是一次膜运行**通过校验、写进 provenance、然后照旧把均匀体相密度 LRC 加到炼金
ligand–environment 项上**——正是 memtodolist §1.3 明令禁止的那件事，而且完全静默。

这与 B2 里刻意设的 `CHARGE_TRANSFER_HAMILTONIAN_IMPLEMENTED = False` 是同一个纪律：
**声明的协议与实际跑的哈密顿量必须一致，做不到就 fail closed，不许静默不一致。**

守的三件事：
  1. 谓词 `ibs_lj_tail_lrc_is_applicable` 同时看 `potential_type` 与
     `dispersion_protocol`，且 legacy/None 时行为与改动前逐位一致（§7.7）。
  2. 生产者与报告者**共用**这个谓词（它 docstring 一直强调的纪律），
     所以 `final_results.json` 不会声称"已应用"而实际没有。
  3. 整条链路把 `dispersion_protocol` 传到底：
     `ABFEPipeline` → `IBSWindowManagerDualLambda` → `build_ibs_dual_system`。

全部 CPU 可跑：只做谓词求值与签名/接线检查，不建 System、不跑采样。
"""

import inspect
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

pytest.importorskip("openmm")
pytest.importorskip("pymbar")

import abfe_core as core
import ibs_engine as ie

ROOT = Path(__file__).absolute().parents[1]


# ---------------------------------------------------------------------------
# 1. 谓词语义
# ---------------------------------------------------------------------------


def test_legacy_and_none_keep_the_pre_change_behaviour():
    """§7.7：可溶体系生产路径逐位不变。"""
    assert ie.ibs_lj_tail_lrc_is_applicable("softcore") is True
    assert ie.ibs_lj_tail_lrc_is_applicable("softcore", None) is True
    assert (
        ie.ibs_lj_tail_lrc_is_applicable(
            "softcore", core.DISPERSION_PROTOCOL_LEGACY_UNIFORM_LRC
        )
        is True
    )
    assert ie.ibs_lj_tail_lrc_inapplicable_reason("softcore", None) == ""


def test_dexp_veto_is_unchanged_and_independent_of_dispersion_protocol():
    """原有的 DEXP 否决不能被新维度冲掉。"""
    assert ie.ibs_lj_tail_lrc_is_applicable("dexp") is False
    assert (
        ie.ibs_lj_tail_lrc_is_applicable(
            "dexp", core.DISPERSION_PROTOCOL_LEGACY_UNIFORM_LRC
        )
        is False
    )
    assert "dexp" in ie.ibs_lj_tail_lrc_inapplicable_reason("dexp")


def test_membrane_native_protocol_disables_the_uniform_density_lrc():
    """§1.3：膜体系必须关闭炼金 ligand–environment 的均匀密度 `lrc_coeff/V`。"""
    assert (
        ie.ibs_lj_tail_lrc_is_applicable(
            "softcore", core.DISPERSION_PROTOCOL_FF_NATIVE_ISOTROPIC_LRC
        )
        is False
    )


def test_reason_uses_the_exact_string_the_plan_specifies():
    """§1.3 指定 metadata 写 `disabled_by_membrane_forcefield_protocol`，
    并且明确要求"**不能写成遗漏**"——理由必须是主动关闭，不是缺失。
    """
    reason = ie.ibs_lj_tail_lrc_inapplicable_reason(
        "softcore", core.DISPERSION_PROTOCOL_FF_NATIVE_ISOTROPIC_LRC
    )
    assert reason.startswith("disabled_by_membrane_forcefield_protocol")
    # 理由里要说清"被关掉的只是炼金 ligand–environment 那一项"，
    # 否则会被误读成"膜体系一律禁用长程色散修正"（§1.3 修正框明确否掉了这个说法）。
    assert "ligand–environment" in reason
    assert "环境–环境" in reason


def test_all_non_legacy_protocols_disable_the_alchemical_lrc():
    """不只 amber 那一条：任何非 legacy 路线都不得沿用均匀密度假设。"""
    for protocol in core.DISPERSION_PROTOCOLS:
        if protocol == core.DISPERSION_PROTOCOL_LEGACY_UNIFORM_LRC:
            continue
        assert ie.ibs_lj_tail_lrc_is_applicable("softcore", protocol) is False, protocol


# ---------------------------------------------------------------------------
# 2. 生产者与报告者共用同一谓词
# ---------------------------------------------------------------------------


def test_producer_and_reporter_call_the_same_predicate_with_both_arguments():
    """谓词 docstring 的纪律：不许在报告侧另写一套判据。

    生产者 = `ibs_engine.build_ibs_dual_system`；
    报告者 = `abfe_pipeline.ABFEPipeline.compute_final_results`。
    两处都必须把 `dispersion_protocol` 传进去，否则一处认为"已应用"另一处认为
    "没应用"，`final_results.json` 就会说谎。
    """
    engine_src = (ROOT / "ibs_engine.py").read_text(encoding="utf-8")
    pipeline_src = (ROOT / "abfe_pipeline.py").read_text(encoding="utf-8")

    assert (
        "ibs_lj_tail_lrc_is_applicable(potential_type, dispersion_protocol)"
        in engine_src
    ), "生产者没有把 dispersion_protocol 传进谓词"
    # 报告者用 getattr 兜底：`compute_final_results` 会被 `ABFEPipeline.__new__`
    # 出来的裸实例调用（tests/test_lrc_reporting_honesty.py 刻意绕过 __init__），
    # 缺属性时按 None = legacy 处理，与改动前同义。
    assert (
        'getattr(self, "dispersion_protocol", None),' in pipeline_src
    ), "报告者没有把 dispersion_protocol 传进谓词"
    reporter_call = pipeline_src.split("_lj_lrc_truth_source = (")[0]
    assert "ibs_lj_tail_lrc_is_applicable(" in reporter_call

    # 报告里必须能看出是"膜协议主动关闭"还是"DEXP 未验证"。
    # 同样走 getattr 兜底（裸实例调用），所以断言的是落盘键 + getattr 形式。
    assert (
        '"dispersion_protocol": getattr(self, "dispersion_protocol", None),'
        in pipeline_src
    )
    assert (
        '"environment_type": getattr(self, "environment_type", None),' in pipeline_src
    )


# ---------------------------------------------------------------------------
# 3. 链路贯通
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target",
    [
        "ibs_engine.build_ibs_dual_system",
        "ibs_engine.IBSWindowManagerDualLambda.__init__",
        "abfe_pipeline.ABFEPipeline.__init__",
    ],
)
def test_dispersion_protocol_is_a_parameter_all_the_way_down(target):
    module_name, _, attr_path = target.partition(".")
    module = {"ibs_engine": ie}.get(module_name)
    if module is None:
        import abfe_pipeline

        module = abfe_pipeline
    obj = module
    for part in attr_path.split("."):
        obj = getattr(obj, part)
    params = inspect.signature(obj).parameters
    assert "dispersion_protocol" in params, f"{target} 缺 dispersion_protocol 形参"
    # 默认必须是 None —— 不传时行为与改动前一致。
    assert params["dispersion_protocol"].default is None


def test_window_manager_forwards_it_to_the_builder():
    """`_build_window_system` 必须把它转下去，否则窗口系统仍按旧口径构建。"""
    src = inspect.getsource(ie.IBSWindowManagerDualLambda._build_window_system)
    assert "dispersion_protocol=self.dispersion_protocol" in src


def test_pipeline_resolves_through_the_same_validator_not_a_second_one():
    """`ABFEPipeline` 必须走 `resolve_dispersion_protocol`，这样 membrane 的
    fail-closed（未声明 / legacy）在 pipeline 层也自动生效，不需要第二套判据。
    """
    pipeline_src = (ROOT / "abfe_pipeline.py").read_text(encoding="utf-8")
    assert "self.dispersion_protocol_info = resolve_dispersion_protocol(" in pipeline_src
    assert (
        'self.dispersion_protocol = self.dispersion_protocol_info["dispersion_protocol"]'
        in pipeline_src
    )


# ---------------------------------------------------------------------------
# 4. resume gate（§6.4）
# ---------------------------------------------------------------------------


def test_non_legacy_dispersion_protocol_enters_the_stage_resume_gate():
    """§6.4：新 LJ 协议必须进 energy cache / resume gate。

    换了色散路线后 u_kn 口径变了，旧的 "completed" stage 缓存不能再被复用；
    但 legacy 时必须**不写这个键**，否则现有生产 stage 缓存会全部失效（§7.7）。
    """
    pipeline_src = (ROOT / "abfe_pipeline.py").read_text(encoding="utf-8")
    block = pipeline_src.split('"kind": "dual_lambda_stage"')[1].split(
        "def _preopt_protocol_key"
    )[0]
    assert '_dispersion != DISPERSION_PROTOCOL_LEGACY_UNIFORM_LRC' in block
    assert 'payload["dispersion_protocol"] = str(_dispersion)' in block
    assert "return _protocol_fingerprint(payload)" in block
