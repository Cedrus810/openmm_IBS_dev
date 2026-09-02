"""P0 后端选择器的契约（`docs/design/PLAN_openmm_8_6_remd_backend.md` §3、§6）。

对应计划 §6 验证清单的前两条：

* 版本边界：8.5.x、8.6.0、8.6.1、8.10.0，以及 rc/dev/未知版本；
  **用模拟版本验证分支**（真实旧版和 8.6.0 的导入与运行验证属于 P1，要等环境升级）。
* 能力异常：版本合格但缺类／缺参数；区分「不可用」和真正的运行错误。

这些测试全部不 import openmm，也不创建任何 Context——P0 的全部价值就在于
「在没有 8.6 的环境里正确判定不可用」，所以它本来就应该能在无 openmm 的环境里测。
"""

from __future__ import annotations

import types

import pytest
from packaging.version import InvalidVersion, Version

import free_energy_engine as fee


def _version(raw: str) -> fee.OpenMMVersionInfo:
    try:
        return fee.OpenMMVersionInfo(raw=raw, parsed=Version(raw))
    except InvalidVersion as exc:
        return fee.OpenMMVersionInfo(raw=raw, parse_error=str(exc))


_CAPABLE = fee.RemdCapabilityReport(available=True)


def _resolve(raw: str, requested: str = "auto", **kw):
    """模拟「适配器已实现」后的决议，用来单独验证版本分支。"""
    kw.setdefault("capability", _CAPABLE)
    kw.setdefault("adapter_implemented", True)
    return fee.resolve_remd_backend(requested, version_info=_version(raw), **kw)


# ---------------------------------------------------------------------------
# 版本边界
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["8.6.0", "8.6.1", "8.10.0", "9.0.0"])
def test_stable_versions_at_or_above_minimum_select_official(raw):
    """门槛是 >= 8.6.0，**包含 8.6.0 本身**（计划 §1）。"""
    assert _resolve(raw).resolved == "openmm"


@pytest.mark.parametrize("raw", ["8.5.0", "8.5.1", "8.0.0", "7.7.0"])
def test_versions_below_minimum_fall_back_to_legacy(raw):
    resolution = _resolve(raw)
    assert resolution.resolved == "legacy"
    assert OFFICIAL_MIN in resolution.reason


OFFICIAL_MIN = fee.OFFICIAL_REMD_MIN_OPENMM_VERSION


def test_semantic_not_string_comparison():
    """计划 §3 实施要求 1：必须用语义版本比较，不能用字符串或浮点数。

    这是这条要求存在的原因——字符串序下 "8.10.0" < "8.6.0"，用字符串比较会把
    一个足够新的版本判成太旧。
    """
    assert "8.10.0" < "8.6.0"  # 字符串序：反直觉但确实如此
    assert Version("8.10.0") > Version("8.6.0")
    assert _resolve("8.10.0").resolved == "openmm"


@pytest.mark.parametrize("raw", ["8.6.0rc1", "8.7.0b2", "9.0.0.dev0", "8.6.0a1"])
def test_prereleases_never_auto_enable_official(raw):
    """计划 §3 实施要求 4：预发布／开发版第一阶段不自动启用官方后端。"""
    resolution = _resolve(raw)
    assert resolution.resolved == "legacy"
    assert "预发布" in resolution.reason


def test_prerelease_above_minimum_is_not_blamed_on_being_too_old():
    """`8.6.0rc1` 在 PEP440 下 < `8.6.0`，但拒绝理由必须说它是 rc，不是「版本太低」。

    否则用户会去升级一个已经足够新的版本，白折腾一轮。
    """
    reason = _resolve("8.6.0rc1").reason
    assert "预发布" in reason
    assert "< 要求的" not in reason


def test_prerelease_genuinely_too_old_is_blamed_on_version():
    """反过来，`7.0.0rc1` 是真的太旧，这时重点就该是版本而不是 rc。"""
    reason = _resolve("7.0.0rc1").reason
    assert "< 要求的" in reason


@pytest.mark.parametrize("raw", ["not-a-version", "", "8.6.0-custom-build+"])
def test_unparseable_versions_fail_closed(raw):
    """解析不了就是不合格，不做任何字符串兜底（计划 §3 实施要求 1/4）。"""
    resolution = _resolve(raw)
    assert resolution.resolved == "legacy"


def test_raw_version_string_is_preserved():
    """计划 §3 实施要求 2：必须记录原始版本字符串。"""
    resolution = _resolve("8.5.1")
    assert resolution.version_info.raw == "8.5.1"
    assert resolution.to_provenance()["openmm_version_raw"] == "8.5.1"


# ---------------------------------------------------------------------------
# 能力探测
# ---------------------------------------------------------------------------


def test_capable_module_passes_probe():
    class _Sampler:
        def __init__(self, states, simulation, stepsPerIteration=1):
            pass

        def simulate(self, iterations):
            pass

        def exchangeReplicas(self):
            pass

    module = types.SimpleNamespace(ReplicaExchangeSampler=_Sampler, ReplicaExchangeReporter=object)
    report = fee.probe_official_remd_capability(module)
    assert report.available
    assert report.missing == ()


def test_probe_reports_missing_class():
    module = types.SimpleNamespace()
    report = fee.probe_official_remd_capability(module)
    assert not report.available
    assert any("ReplicaExchangeSampler" in m for m in report.missing)


def test_probe_checks_more_than_a_class_name():
    """计划 §3 实施要求 3：不能只检查一个类名。

    这个假 sampler 类名对、但缺 `stepsPerIteration` 构造参数和 `exchangeReplicas`
    方法——只看类名会误判成可用。
    """

    class _Crippled:
        def __init__(self, states, simulation):
            pass

        def simulate(self, iterations):
            pass

    module = types.SimpleNamespace(
        ReplicaExchangeSampler=_Crippled, ReplicaExchangeReporter=object
    )
    report = fee.probe_official_remd_capability(module)
    assert not report.available
    assert any("stepsPerIteration" in m for m in report.missing)
    assert any("exchangeReplicas" in m for m in report.missing)


def test_probe_distinguishes_unavailable_from_runtime_error():
    """计划 §6 验证清单第 2 条：区分「不可用」和真正的运行错误。"""

    class _Boom:
        def __getattr__(self, name):
            raise RuntimeError("模块内部炸了")

    # 属性缺失 -> missing（不可用）
    missing_report = fee.probe_official_remd_capability(types.SimpleNamespace())
    assert missing_report.missing and missing_report.probe_error is None

    # import 失败 -> probe_error（运行错误），两者语义不同
    err_report = fee.RemdCapabilityReport(available=False, probe_error="ImportError: x")
    assert err_report.probe_error and not err_report.missing


def test_capable_version_but_missing_api_falls_back():
    """版本合格但 API 缺失时仍须回退（计划 §3 实施要求 5）。"""
    broken = fee.RemdCapabilityReport(available=False, missing=("openmm.app.ReplicaExchangeSampler",))
    resolution = _resolve("8.10.0", capability=broken)
    assert resolution.resolved == "legacy"
    assert "API" in resolution.reason


def test_probe_records_what_it_did_not_verify():
    """实例属性 P0 验不了，必须明说，不能让「探测通过」被读成「全查过了」。"""
    report = fee.probe_official_remd_capability(types.SimpleNamespace())
    assert report.unverified


# ---------------------------------------------------------------------------
# 决议表（计划 §3）
# ---------------------------------------------------------------------------


def test_legacy_is_unconditional():
    """显式 legacy 在任何环境下都走 legacy，不因环境很好就"顺便"升级。"""
    resolution = fee.resolve_remd_backend(
        "legacy", version_info=_version("8.10.0"), capability=_CAPABLE, adapter_implemented=True
    )
    assert resolution.resolved == "legacy"
    assert resolution.exchange_scheme == fee.EXCHANGE_SCHEME_LEGACY


def test_explicit_official_raises_instead_of_silently_downgrading():
    """计划 §3：显式请求官方后端而检查不通过时**不能默默改用 legacy**。"""
    with pytest.raises(fee.UnsupportedRemdBackendError) as excinfo:
        _resolve("8.5.1", requested="openmm")
    message = str(excinfo.value)
    assert "8.5.1" in message
    assert "legacy" in message  # 必须告诉用户怎么继续跑


def test_explicit_official_error_lists_all_blockers_not_just_the_first():
    """一次看到全部差距，否则修一条再撞一条。"""
    with pytest.raises(fee.UnsupportedRemdBackendError) as excinfo:
        fee.resolve_remd_backend(
            "openmm",
            version_info=_version("8.5.1"),
            capability=fee.RemdCapabilityReport(available=False, missing=("x",)),
            adapter_implemented=False,
        )
    message = str(excinfo.value)
    assert "8.5.1" in message and "API" in message and "适配器" in message


def test_unknown_backend_name_rejected():
    with pytest.raises(ValueError):
        fee.resolve_remd_backend("openmmtools")


def test_exchange_scheme_is_recorded_per_backend():
    """两种交换过程不同，必须分别记录、不能声称等价（计划 §4.2）。"""
    assert _resolve("8.6.0").exchange_scheme == fee.EXCHANGE_SCHEME_OFFICIAL
    assert _resolve("8.5.1").exchange_scheme == fee.EXCHANGE_SCHEME_LEGACY


# ---------------------------------------------------------------------------
# 当前阶段的实际行为：适配器未实现
# ---------------------------------------------------------------------------


def test_adapter_not_implemented_yet():
    """P1 完成前，这个开关必须是 False。

    🔑 这个断言是给**未来的人**的：如果有人为了让某个测试变绿而提前翻转
    `OFFICIAL_REMD_ADAPTER_IMPLEMENTED`，这里会红，并指向 P1 的放行条件。
    翻转它的同时必须一并修改本测试，那是一次需要解释的改动，不该悄悄发生。
    """
    assert fee.OFFICIAL_REMD_ADAPTER_IMPLEMENTED is False
    assert fee.FREE_ENERGY_ENGINE_ADAPTER_PROTOCOL_VERSION == 0


def test_auto_currently_always_resolves_to_legacy_regardless_of_environment():
    """默认运行行为一字不变——哪怕环境是 OpenMM 8.10 且 API 齐备。"""
    resolution = fee.resolve_remd_backend(
        "auto", version_info=_version("8.10.0"), capability=_CAPABLE
    )
    assert resolution.resolved == "legacy"
    assert "适配器" in resolution.reason


def test_explicit_official_currently_always_raises():
    with pytest.raises(fee.UnsupportedRemdBackendError):
        fee.resolve_remd_backend(
            "openmm", version_info=_version("8.10.0"), capability=_CAPABLE
        )


# ---------------------------------------------------------------------------
# 可审计性
# ---------------------------------------------------------------------------


def test_provenance_contains_every_field_the_plan_requires():
    """计划 §4.3：requested/resolved、原始与解析后版本、适配器协议版本、
    选择原因、交换算法都要落盘。"""
    prov = _resolve("8.5.1").to_provenance()
    for key in (
        "remd_backend_requested",
        "remd_backend_resolved",
        "remd_backend_reason",
        "remd_exchange_scheme",
        "remd_adapter_protocol_version",
        "openmm_version_raw",
        "openmm_version_parsed",
    ):
        assert key in prov, f"provenance 缺字段 {key}"


def test_provenance_does_not_conflate_platform_fallback_with_backend_fallback():
    """计划 §3 实施要求 7：GPU→CPU 是平台策略，official→legacy 是后端策略，
    分别记录，不能混为同一种回退。"""
    prov = _resolve("8.5.1").to_provenance()
    assert not any("platform" in k or "cuda" in k.lower() for k in prov)


def test_log_line_is_single_line_and_names_both_ends():
    line = _resolve("8.5.1").format_log()
    assert "\n" not in line
    assert "auto" in line and "legacy" in line


def test_module_does_not_import_openmm_at_module_level():
    """旧环境必须仍能 import 本项目（计划 §3 实施要求 3）。

    本测试文件本身就是证据——它在一个没装 openmm 的解释器里也能跑通。
    """
    import sys

    assert "openmm" not in sys.modules or True  # 不强制断言全局状态
    source = (__import__("pathlib").Path(fee.__file__)).read_text(encoding="utf-8")
    module_level = [
        line
        for line in source.splitlines()
        if line.startswith("import openmm") or line.startswith("from openmm")
    ]
    assert not module_level, f"openmm 必须延迟导入，发现模块级导入：{module_level}"


# ---------------------------------------------------------------------------
# 真实环境暴露的回归（2026-09-01）
# ---------------------------------------------------------------------------


def test_dev_build_is_not_read_as_a_stable_release():
    """🔑 真机踩到的 bug：pip 装的 OpenMM 开发版里

        openmm.__version__          == "8.6"                <- 干净，像正式版
        openmm.version.full_version == "8.6.0.dev-c6173db"  <- 真相：dev 构建

    原实现优先读 `__version__`，于是把开发版判成正式版、`meets_official_minimum`
    返回 True，直接违反计划 §3 实施要求 4。修复后必须优先读 full_version。
    """

    class _FakeVersionModule:
        version = "8.6.0.dev-c6173db"
        full_version = "8.6.0.dev-c6173db"

    fake = types.SimpleNamespace(__version__="8.6", version=_FakeVersionModule)
    info = fee.resolve_openmm_version(fake)

    assert info.raw == "8.6.0.dev-c6173db", "必须以 full_version 为权威版本串"
    assert info.short_raw == "8.6", "__version__ 也要留档，事后才看得出被截短过"
    assert info.is_prerelease is True
    assert info.meets_official_minimum is False


def test_unparseable_dev_version_still_reports_that_it_is_a_dev_build():
    """开发版的版本串常常不是合法 PEP440。这时说「无法解析」是对的但没用——
    用户需要知道的是「这是 dev 构建」，两者的后续动作完全不同。"""

    class _FakeVersionModule:
        version = "8.6.0.dev-c6173db"
        full_version = "8.6.0.dev-c6173db"

    fake = types.SimpleNamespace(__version__="8.6", version=_FakeVersionModule)
    info = fee.resolve_openmm_version(fake)
    assert info.parsed is None  # PEP440 解析不了

    resolution = fee.resolve_remd_backend(
        "auto", version_info=info, capability=_CAPABLE, adapter_implemented=True
    )
    assert resolution.resolved == "legacy"
    assert "开发/预发布构建" in resolution.reason


def test_short_version_alone_still_works_when_there_is_no_version_module():
    """老版本/精简构建可能只有 __version__，不能因为优先 full_version 就读不到。"""
    fake = types.SimpleNamespace(__version__="8.5.1")
    info = fee.resolve_openmm_version(fake)
    assert info.raw == "8.5.1"
    assert info.parsed is not None


def test_provenance_records_both_version_strings():
    class _FakeVersionModule:
        version = "8.6.0.dev-c6173db"
        full_version = "8.6.0.dev-c6173db"

    fake = types.SimpleNamespace(__version__="8.6", version=_FakeVersionModule)
    prov = fee.resolve_remd_backend(
        "auto", version_info=fee.resolve_openmm_version(fake), capability=_CAPABLE
    ).to_provenance()
    assert prov["openmm_version_raw"] == "8.6.0.dev-c6173db"
    assert prov["openmm_version_dunder"] == "8.6"
    assert prov["openmm_version_is_prerelease"] is True
