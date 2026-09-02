#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ABFE / RBFE 共用的自由能采样引擎。

设计依据：`docs/design/PLAN_openmm_8_6_remd_backend.md` §4.0 与
`docs/design/PLAN_rbfe_interface_and_implementation.md` §4。两份计划都把「采样后端
选择」和「通用采样契约」放在同一个大模块里，**不另拆 `remd_backends.py`**，避免
ABFE 和 RBFE 各自建立一套平行的版本判断与交换引擎。

## 当前实现进度

本文件目前实现两块：

* **P0：后端选择器**——版本解析 -> 能力探测 -> 后端决议(BackendResolution) -> 决议日志。
  已接进 `runabfe.py`（`--remd-backend`），在建立任何 Context 之前解析一次。
* **P2′：ABFE / RBFE 共用的采样契约**——`SamplingRequest` / `StepPlan` /
  `SamplingArtifacts` / `run_sampling()`。**尚未接进生产**：`abfe_pipeline.py` 的三个
  `REMDManager` 构造点目前仍直接调 `remd.run(...)`。

**还没有实现官方后端的任何采样。** P1（官方 `ReplicaExchangeSampler` 适配器）没开始，
见文件末尾的 `# === 未实现区 ===`。因此本模块当前对生产运行**没有任何行为影响**：
`OFFICIAL_REMD_ADAPTER_IMPLEMENTED = False` 让 `auto` 恒定解析为 legacy、显式
`openmm` 恒定报错（而不是悄悄降级）。

## 依赖方向（不得违反）

    runabfe.py -> abfe_pipeline.py -> free_energy_engine.py
    runrbfe.py -> rbfe_pipeline.py -> free_energy_engine.py

本模块**不反向 import** `abfe_pipeline` / `abfe_core` / `ibs_engine` / `rbfe_*`，
也不 import `openmm` 顶层——`openmm` 一律延迟导入，好让没装 openmm 或装了旧版
openmm 的环境仍然能 import 本项目并跑离线测试。

## 为什么把「能不能用」拆成三个独立的判据

计划 §3 的实施要求 1/3/5 明确要求版本、API、协议三者分别判定，不能互相顶替：

* **版本合格**（`>= 8.6.0`）只是必要条件。高版本发生 API 不兼容时，不得仅因为
  版本号大就强行调用。
* **API 可用**要逐个检查适配器真正会用到的类／方法／参数，不能只检查一个类名。
* **协议已适配**是本项目自己的状态：即使环境完全合格，只要本项目还没写出并验证
  过适配器（当前就是这个状态），也必须走 legacy。
"""

from __future__ import annotations

import inspect
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: `--remd-backend` 的合法取值。`auto` 是目标默认值（计划 §3）。
REMD_BACKEND_CHOICES: tuple[str, ...] = ("auto", "legacy", "openmm")
REMD_BACKEND_DEFAULT = "auto"

#: 官方 `ReplicaExchangeSampler` 的最低 OpenMM 版本。**包含 8.6.0 本身**，
#: 判据是 `>=` 而不是 `>`——8.6.0 的发布说明确认该版本已新增官方采样器。
OFFICIAL_REMD_MIN_OPENMM_VERSION = "8.6.0"

#: 本项目 legacy REMD 的交换算法身份：每轮按 (0,1),(1,2),... 顺序扫描所有相邻边。
EXCHANGE_SCHEME_LEGACY = "legacy_sequential_neighbors"
#: 官方采样器的交换算法身份：每轮随机选副本对，尝试 K**2 次。
#: **官方没有直接配置相邻交换的开关**，所以这两种交换过程不同，必须分别资格验证
#: （计划 §4.2）。目标平衡分布可以一致，但不能声称等价。
EXCHANGE_SCHEME_OFFICIAL = "openmm_random_pairs"

#: 适配器协议版本。进采样指纹（P2 的工作），使换后端／换适配器实现不会误复用缓存。
#: P0 阶段还没有适配器，先占位为 0 表示「尚无适配器」。
FREE_ENERGY_ENGINE_ADAPTER_PROTOCOL_VERSION = 0

#: 🔑 **P1 翻转这个开关。** 官方后端适配器（状态映射、DCD 桥接、reporter、seed
#: 映射）一行都还没写，所以哪怕环境是 OpenMM 8.6 也不能用。把它设成 True 之前
#: 必须先通过计划 §5 P1 的放行条件（小体系能量／力一致；状态分流、步数和帧数验证
#: 通过）——**不要为了让某个测试变绿而提前翻转它。**
OFFICIAL_REMD_ADAPTER_IMPLEMENTED = False

#: 适配器真正会用到的官方 API。能力探测逐项检查这张表，不是只看一个类名
#: （计划 §3 实施要求 3）。
_REQUIRED_OFFICIAL_ATTRS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # (openmm.app 下的类名, 该类必须具备的方法)
    ("ReplicaExchangeSampler", ("simulate", "exchangeReplicas")),
    ("ReplicaExchangeReporter", ()),
)

#: 🔑 [2026-08-31] 真实环境实测：pip 装的 OpenMM 开发版里
#:     openmm.__version__          == "8.6"                  <- 干净，看着像正式版
#:     openmm.version.full_version == "8.6.0.dev-c6173db"    <- 真相：dev 构建
#: 只读 `__version__` 会把开发版当成正式版放行，直接违反计划 §3 实施要求 4。
#: 而 full_version 又不是合法 PEP440（`.dev` 后无数字、`-` 不是合法 local 分隔符），
#: packaging 解析不了。所以这里保留一条**文本判据**：解析失败时仍然要能说出
#: 「这是开发版」，而不是笼统地说「无法解析」——两者的用户动作完全不同。
_DEV_OR_PRERELEASE_RE = re.compile(r"(?i)(?:^|[-_.+])(dev|rc|alpha|beta|a\d|b\d)")

#: `ReplicaExchangeSampler.__init__` 必须接受的关键字参数。
#: `stepsPerIteration` 是本项目把「交换间隔」映射过去的唯一入口，缺了它整个
#: 步数语义就对不上（计划 §4.2）。
_REQUIRED_SAMPLER_INIT_PARAMS: tuple[str, ...] = ("stepsPerIteration",)


class UnsupportedRemdBackendError(RuntimeError):
    """显式请求 `--remd-backend openmm` 但检查未通过。

    计划 §3 要求：显式请求官方后端而任一检查不通过时，**必须在创建生产 Context／
    写轨迹之前明确报错，不能默默改用 legacy**。静默降级会让用户以为自己跑的是
    官方后端，产物身份从此对不上。
    """


# ---------------------------------------------------------------------------
# 版本解析
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpenMMVersionInfo:
    """一次性解析出的 OpenMM 版本身份。

    `raw` 保留原始版本字符串（计划 §3 实施要求 2：必须记录原始字符串，不能只留
    解析后的结果）。`parsed` 解析失败时为 None，此时 `parse_error` 说明原因。
    """

    #: 权威版本串：优先 `openmm.version.full_version`（它带 dev/rc 后缀）。
    raw: Optional[str]
    parsed: Optional[Any] = None
    parse_error: Optional[str] = None
    import_error: Optional[str] = None
    #: `openmm.__version__` 的原样值。开发版里它会被**截短**成看着像正式版的
    #: "8.6"，两个都记下来才能在事后看出这次跑的到底是什么构建。
    short_raw: Optional[str] = None

    @property
    def importable(self) -> bool:
        return self.import_error is None

    @property
    def is_prerelease(self) -> bool:
        """预发布或开发版（rc / a / b / .devN）。

        解析成功时用 packaging 判定；**解析失败时退回文本判据**——开发版的版本串
        恰恰经常是不合法 PEP440 的，如果这时直接返回 False，就等于对最需要拦住的
        那一类构建失去判断力。
        """
        v = self.parsed
        if v is not None:
            return bool(
                getattr(v, "is_prerelease", False) or getattr(v, "is_devrelease", False)
            )
        return bool(self.raw and _DEV_OR_PRERELEASE_RE.search(self.raw))

    @property
    def meets_official_minimum(self) -> bool:
        """版本 `>= 8.6.0`。解析失败一律判为不合格（fail-closed）。"""
        if self.parsed is None:
            return False
        from packaging.version import Version

        return self.parsed >= Version(OFFICIAL_REMD_MIN_OPENMM_VERSION)

    @property
    def base_meets_official_minimum(self) -> bool:
        """**忽略预发布后缀**后的版本是否 >= 8.6.0。

        用来区分两种不同的拒绝理由：`8.6.0rc1` 在 PEP440 下确实 < `8.6.0`，
        但对用户说"版本太低"是误导——真正的原因是它是预发布版。而 `7.0.0rc1`
        则是真的太低，说预发布版反而没抓住重点。
        """
        if self.parsed is None:
            return False
        from packaging.version import Version

        return Version(self.parsed.base_version) >= Version(OFFICIAL_REMD_MIN_OPENMM_VERSION)

    def describe(self) -> str:
        if self.import_error is not None:
            return f"openmm 无法导入({self.import_error})"
        if self.raw is None:
            return "未探测"
        suffix = (
            f"（__version__ 报 {self.short_raw!r}）"
            if self.short_raw and self.short_raw != self.raw
            else ""
        )
        if self.parsed is None:
            return f"{self.raw!r}(无法解析: {self.parse_error}){suffix}"
        return f"{self.raw}{suffix}"


def resolve_openmm_version(module: Any = None) -> OpenMMVersionInfo:
    """解析**实际导入的** OpenMM 版本。

    计划 §3 实施要求 2：必须集中解析实际导入的版本，**不能仅凭另一环境的包管理器
    信息选择接口**——conda 的 metadata 和真正被 import 的那个 openmm 完全可能不是
    同一个。

    `module` 仅供测试注入假的模块对象；生产调用不传，走延迟导入。
    """
    if module is None:
        try:
            import openmm as module  # noqa: PLC0415 — 延迟导入是本模块的硬要求
        except Exception as exc:  # pragma: no cover - 取决于环境
            return OpenMMVersionInfo(raw=None, import_error=f"{type(exc).__name__}: {exc}")

    # 🔑 顺序**必须**是 full_version 优先。见 _DEV_OR_PRERELEASE_RE 上方的实测记录：
    # 开发版里 `__version__` 会被截短成 "8.6"，只读它等于把开发版当正式版。
    short_raw = getattr(module, "__version__", None)
    version_mod = getattr(module, "version", None)
    raw = None
    if version_mod is not None:
        raw = getattr(version_mod, "full_version", None) or getattr(
            version_mod, "version", None
        )
    if raw is None:
        raw = short_raw
    if raw is None:
        return OpenMMVersionInfo(
            raw=None,
            short_raw=None,
            parse_error="模块既无 version.full_version/version 也无 __version__",
        )

    raw = str(raw)
    short_raw = str(short_raw) if short_raw is not None else None
    try:
        from packaging.version import InvalidVersion, Version
    except Exception as exc:  # pragma: no cover - packaging 是显式声明的依赖
        return OpenMMVersionInfo(
            raw=raw, short_raw=short_raw, parse_error=f"packaging 不可用: {exc}"
        )

    try:
        return OpenMMVersionInfo(raw=raw, short_raw=short_raw, parsed=Version(raw))
    except InvalidVersion as exc:
        # 🔑 不做字符串比较或 float(raw) 兜底。计划 §3 实施要求 1 明确禁止——
        # "8.10.0" > "8.6.0" 在字符串序下为假，float("8.6.0") 直接抛异常。
        # 解析不了就是解析不了，fail-closed 走 legacy。
        return OpenMMVersionInfo(raw=raw, short_raw=short_raw, parse_error=str(exc))


# ---------------------------------------------------------------------------
# 能力探测
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RemdCapabilityReport:
    """官方 REMD API 的逐项探测结果。"""

    available: bool
    missing: tuple[str, ...] = ()
    probe_error: Optional[str] = None
    #: 探测**没有**覆盖的东西。实例属性（如 `replicaStateIndex`）只有构造出真实
    #: sampler 才能验证，P0 不构造任何 Context，所以只能留给 P1 适配器验证。
    #: 明确记下来，免得有人把「探测通过」误读成「API 全部验证过」。
    unverified: tuple[str, ...] = ()

    def describe(self) -> str:
        if self.probe_error is not None:
            return f"探测失败: {self.probe_error}"
        if self.available:
            return "官方 REMD API 齐备"
        return "缺少: " + ", ".join(self.missing)


def probe_official_remd_capability(module: Any = None) -> RemdCapabilityReport:
    """逐项检查官方 REMD 适配器需要的类、方法和构造参数。

    计划 §3 实施要求 3：**不能只检查一个类名**。§3 实施要求 5：高版本若发生 API
    不兼容，不得仅因为版本号大就强行调用——所以这个探测独立于版本判定，两边都要过。

    区分「不可用」和「真正的运行错误」（计划 §6 验证清单第 2 条）：import 本身炸了
    记进 `probe_error`，属性缺失记进 `missing`，两者语义不同。
    """
    if module is None:
        try:
            import openmm.app as module  # noqa: PLC0415 — 延迟导入
        except Exception as exc:
            return RemdCapabilityReport(
                available=False, probe_error=f"{type(exc).__name__}: {exc}"
            )

    missing: list[str] = []
    for cls_name, required_methods in _REQUIRED_OFFICIAL_ATTRS:
        cls = getattr(module, cls_name, None)
        if cls is None:
            missing.append(f"openmm.app.{cls_name}")
            continue
        for method in required_methods:
            if not callable(getattr(cls, method, None)):
                missing.append(f"openmm.app.{cls_name}.{method}()")

    sampler = getattr(module, "ReplicaExchangeSampler", None)
    if sampler is not None:
        try:
            params = inspect.signature(sampler.__init__).parameters
        except (TypeError, ValueError) as exc:
            missing.append(f"openmm.app.ReplicaExchangeSampler.__init__ 签名不可读({exc})")
        else:
            accepts_kwargs = any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            if not accepts_kwargs:
                for name in _REQUIRED_SAMPLER_INIT_PARAMS:
                    if name not in params:
                        missing.append(f"openmm.app.ReplicaExchangeSampler.__init__({name}=)")

    return RemdCapabilityReport(
        available=not missing,
        missing=tuple(missing),
        unverified=(
            "ReplicaExchangeSampler.reporters(实例属性)",
            "ReplicaExchangeSampler.replicaStateIndex(实例属性)",
        ),
    )


# ---------------------------------------------------------------------------
# 后端决议
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendResolution:
    """一次后端选择的完整可审计记录（计划 §4.1 的 `BackendResolution` 契约）。

    计划 §4.3 要求落盘：requested/resolved backend、原始与解析后 OpenMM 版本、
    适配器协议版本、选择原因、交换算法。这些字段在 P2 进采样指纹。
    """

    requested: str
    resolved: str
    reason: str
    version_info: OpenMMVersionInfo
    capability: RemdCapabilityReport
    adapter_implemented: bool
    exchange_scheme: str
    adapter_protocol_version: int = FREE_ENERGY_ENGINE_ADAPTER_PROTOCOL_VERSION

    @property
    def is_official(self) -> bool:
        return self.resolved == "openmm"

    def to_provenance(self) -> dict:
        """落盘用的扁平字典。

        🔑 平台回退（GPU→CPU）与后端回退（official→legacy）是**两种不同的回退**，
        必须分别记录、不能混为一谈（计划 §3 实施要求 7）。本字典只描述后端，
        平台身份由既有的 platform provenance 负责。
        """
        return {
            "remd_backend_requested": self.requested,
            "remd_backend_resolved": self.resolved,
            "remd_backend_reason": self.reason,
            "remd_exchange_scheme": self.exchange_scheme,
            "remd_adapter_protocol_version": int(self.adapter_protocol_version),
            "remd_adapter_implemented": bool(self.adapter_implemented),
            "openmm_version_raw": self.version_info.raw,
            "openmm_version_dunder": self.version_info.short_raw,
            "openmm_version_is_prerelease": bool(self.version_info.is_prerelease),
            "openmm_version_parsed": (
                str(self.version_info.parsed) if self.version_info.parsed is not None else None
            ),
            "openmm_version_parse_error": self.version_info.parse_error,
            "openmm_import_error": self.version_info.import_error,
            "official_remd_capability_available": bool(self.capability.available),
            "official_remd_capability_missing": list(self.capability.missing),
            "official_remd_capability_probe_error": self.capability.probe_error,
        }

    def format_log(self) -> str:
        """给运行日志的一行决议摘要（计划 P0 交付物「选择日志」）。"""
        arrow = f"{self.requested} -> {self.resolved}"
        return (
            f"[REMD backend] {arrow}（{self.reason}）"
            f" | openmm={self.version_info.describe()}"
            f" | exchange={self.exchange_scheme}"
        )


def _official_blockers(
    version_info: OpenMMVersionInfo,
    capability: RemdCapabilityReport,
    adapter_implemented: bool,
) -> list[str]:
    """列出所有阻止使用官方后端的原因。

    **返回全部原因而不是第一条**：显式 `openmm` 报错时用户需要一次看到全部差距，
    否则修一条再撞一条。
    """
    blockers: list[str] = []

    if not version_info.importable:
        blockers.append(f"openmm 无法导入：{version_info.import_error}")
        return blockers  # 连导入都失败，后面的判据没有意义

    if version_info.parsed is None:
        if version_info.is_prerelease:
            # 最常见的一种：开发版的版本串本来就不是合法 PEP440。这时说"无法解析"
            # 是对的但没用——用户需要知道的是"这是 dev 构建，第一阶段不放行"。
            blockers.append(
                f"OpenMM {version_info.raw} 是开发/预发布构建"
                f"（版本串不符合 PEP440：{version_info.parse_error}）；"
                f"第一阶段只支持正式版 >= {OFFICIAL_REMD_MIN_OPENMM_VERSION}"
            )
        else:
            blockers.append(
                f"OpenMM 版本 {version_info.raw!r} 无法按语义版本解析"
                f"（{version_info.parse_error}）"
            )
    elif version_info.is_prerelease and version_info.base_meets_official_minimum:
        # 🔑 计划 §3 实施要求 4：预发布／开发版第一阶段不自动启用官方后端。
        # 先判这条：8.6.0rc1 在 PEP440 下 < 8.6.0，若先报"版本太低"会误导用户去
        # 升级——它已经够新了，问题是它是 rc。
        blockers.append(
            f"OpenMM {version_info.raw} 是预发布/开发版；"
            f"第一阶段只支持正式版 >= {OFFICIAL_REMD_MIN_OPENMM_VERSION}"
        )
    elif not version_info.meets_official_minimum:
        suffix = "（且为预发布/开发版）" if version_info.is_prerelease else ""
        blockers.append(
            f"OpenMM {version_info.raw} < 要求的 {OFFICIAL_REMD_MIN_OPENMM_VERSION}{suffix}"
        )

    if not capability.available:
        blockers.append(f"官方 REMD API 检查未通过（{capability.describe()}）")

    if not adapter_implemented:
        blockers.append(
            "本项目的官方 REMD 适配器尚未实现并通过资格验证"
            "（PLAN_openmm_8_6_remd_backend.md P1 未完成）"
        )

    return blockers


def resolve_remd_backend(
    requested: Optional[str] = None,
    *,
    version_info: Optional[OpenMMVersionInfo] = None,
    capability: Optional[RemdCapabilityReport] = None,
    adapter_implemented: Optional[bool] = None,
) -> BackendResolution:
    """把请求的后端解析成实际使用的后端。

    计划 §3 的决议表：

    ==========  ==========================================  ==========================
    请求        条件                                        行为
    ==========  ==========================================  ==========================
    ``legacy``  任何环境                                    始终 legacy（便于复现/回退）
    ``auto``    版本+API+协议全过                            官方后端
    ``auto``    任一不过                                    legacy，并记录**具体原因**
    ``openmm``  全过                                        官方后端
    ``openmm``  任一不过                                    抛 UnsupportedRemdBackendError
    ==========  ==========================================  ==========================

    🔑 后端只在阶段开始前解析一次（计划 §3 实施要求 6）。采样开始后遇到 NaN、
    配置错误、能量错误或 I/O 错误必须停止并保留诊断，**禁止捕获所有异常后从头
    换引擎继续**——那会让产物身份变成两个后端的混合物。

    关键字参数仅供测试注入；生产调用只传 `requested`。
    """
    requested = (requested or REMD_BACKEND_DEFAULT).strip().lower()
    if requested not in REMD_BACKEND_CHOICES:
        raise ValueError(
            f"未知的 remd_backend={requested!r}；合法取值：{', '.join(REMD_BACKEND_CHOICES)}"
        )

    if adapter_implemented is None:
        adapter_implemented = OFFICIAL_REMD_ADAPTER_IMPLEMENTED

    # legacy 是无条件的：不探测环境，也不因为环境很好就"顺便"升级。
    # 它存在的意义就是"我要一条不受环境影响、可复现的老路"。
    if requested == "legacy":
        return BackendResolution(
            requested=requested,
            resolved="legacy",
            reason="显式请求 legacy",
            version_info=version_info or OpenMMVersionInfo(raw=None),
            capability=capability or RemdCapabilityReport(available=False),
            adapter_implemented=bool(adapter_implemented),
            exchange_scheme=EXCHANGE_SCHEME_LEGACY,
        )

    if version_info is None:
        version_info = resolve_openmm_version()
    if capability is None:
        capability = (
            probe_official_remd_capability()
            if version_info.importable
            else RemdCapabilityReport(available=False, probe_error="openmm 未导入，跳过 API 探测")
        )

    blockers = _official_blockers(version_info, capability, bool(adapter_implemented))

    if not blockers:
        return BackendResolution(
            requested=requested,
            resolved="openmm",
            reason="版本、API 与适配器协议检查全部通过",
            version_info=version_info,
            capability=capability,
            adapter_implemented=True,
            exchange_scheme=EXCHANGE_SCHEME_OFFICIAL,
        )

    joined = "；".join(blockers)

    if requested == "openmm":
        # 🔑 计划 §3：显式请求官方后端时**不静默降级**。这里抛错，调用方必须在
        # 创建生产 Context / 写轨迹之前调用本函数，让错误发生在烧 GPU 之前。
        raise UnsupportedRemdBackendError(
            f"显式请求 --remd-backend openmm，但检查未通过：{joined}。"
            f"（支持范围：正式版 OpenMM >= {OFFICIAL_REMD_MIN_OPENMM_VERSION} "
            f"且本项目适配器已通过资格验证。要继续跑请显式指定 --remd-backend legacy。）"
        )

    return BackendResolution(
        requested=requested,
        resolved="legacy",
        reason=f"auto 回退 legacy：{joined}",
        version_info=version_info,
        capability=capability,
        adapter_implemented=bool(adapter_implemented),
        exchange_scheme=EXCHANGE_SCHEME_LEGACY,
    )


def resolve_and_log_remd_backend(
    requested: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
    **kwargs: Any,
) -> BackendResolution:
    """`resolve_remd_backend` + 一行决议日志。阶段开始前调用一次。"""
    resolution = resolve_remd_backend(requested, **kwargs)
    if log is not None:
        log(resolution.format_log())
    else:
        print(resolution.format_log())
    return resolution


# ---------------------------------------------------------------------------
# P2′：ABFE / RBFE 共用的采样契约
# ---------------------------------------------------------------------------
#
# 这一段**不需要 OpenMM 8.6**。它把「采样」从「谁构建了这个 System」里剥出来，
# 让 ABFE 的去耦 System 和 RBFE 的 hybrid System 走同一条采样路径。
#
# 依赖方向的硬约束（两份计划都写死）：本模块**不得** import ibs_engine。因此
# legacy 引擎实例由 pipeline 构造好之后传进来，engine 只定义契约、校验契约、
# 并把结果规整成 SamplingArtifacts。engine 永远不知道 ligand A/B，也永远不自己
# 决定物理默认值——`RBFE 计划 §4.1`：未提供某项能力时必须显式报告不支持，
# 不由引擎猜测。


class SamplingContractError(ValueError):
    """采样请求不满足契约。在创建任何 Context 之前抛出。"""


@dataclass(frozen=True)
class StepPlan:
    """把「MD 步数 / 交换间隔 / 保存间隔」翻译成明确的执行计划。

    计划 §4.2：**单位明确区分 MD steps、交换间隔和 sampler iterations；准确处理
    `n_steps % exchange_interval` 尾段，不增跑、漏跑或额外交换。**

    这是最容易出错、也最容易被"差不多"掩盖的地方：官方采样器按 iteration 推进
    （每 iteration 跑 `stepsPerIteration` 步），而本项目按总 MD 步数配置。二者
    不整除时，四舍五入会悄悄多跑或少跑几千步——落盘的 `n_steps` 却仍然写着原值。
    """

    total_md_steps: int
    exchange_interval: int
    save_interval: int
    #: 完整 iteration 数（每个 iteration = exchange_interval 步 MD + 一次交换尝试）
    full_iterations: int
    #: 尾段剩余 MD 步数。非 0 表示最后有一段**不跟随交换**的推进。
    tail_md_steps: int

    @property
    def has_tail(self) -> bool:
        return self.tail_md_steps > 0

    @property
    def accounted_md_steps(self) -> int:
        """执行计划实际覆盖的 MD 步数。必须恒等于 total_md_steps。"""
        return self.full_iterations * self.exchange_interval + self.tail_md_steps


def resolve_step_plan(
    total_md_steps: int, exchange_interval: int, save_interval: int
) -> StepPlan:
    """解析步数计划，并校验它**逐步对账**。

    不做任何四舍五入：不整除时如实产出尾段，由后端决定是拒绝还是显式处理
    （§4.2：无法保持当前输出语义时，在开跑前拒绝显式官方模式或让 auto 选 legacy，
    **不静默四舍五入**）。
    """
    total_md_steps = int(total_md_steps)
    exchange_interval = int(exchange_interval)
    save_interval = int(save_interval)

    if total_md_steps < 0:
        raise SamplingContractError(f"total_md_steps 不能为负：{total_md_steps}")
    if exchange_interval < 1:
        raise SamplingContractError(f"exchange_interval 至少为 1：{exchange_interval}")
    if save_interval < 1:
        raise SamplingContractError(f"save_interval 至少为 1：{save_interval}")

    plan = StepPlan(
        total_md_steps=total_md_steps,
        exchange_interval=exchange_interval,
        save_interval=save_interval,
        full_iterations=total_md_steps // exchange_interval,
        tail_md_steps=total_md_steps % exchange_interval,
    )
    # 自我对账。这条断言不是装饰——整个 §4.2 的要求就是"不增跑、漏跑"。
    if plan.accounted_md_steps != total_md_steps:
        raise SamplingContractError(
            f"步数计划对不上账：{plan.accounted_md_steps} != {total_md_steps}"
        )
    return plan


@dataclass(frozen=True)
class ThermodynamicStateSpec:
    """一个热力学状态：一组全局参数值。

    刻意用 `dict[str, float]` 而不是任何 openmm 类型——engine 不认识
    `lambda_coul` 还是 `lambda_hybrid`，那是 ABFE / RBFE 各自化学层的事。
    """

    index: int
    parameters: dict


@dataclass(frozen=True)
class SamplingRequest:
    """一次采样请求（RBFE 计划 §4.1 的 `SamplingRequest` 契约）。

    `system` / `topology` / `initial_positions` 等一律是**不透明对象**：engine 只
    转交，不解释。这样 ABFE 的去耦 System 和 RBFE 的 hybrid System 可以走同一条路。

    🔑 engine **不构建 System**。计划 §4.1 明确：「引擎保证的是『按指定 Hamiltonian
    和状态表采样』，化学正确性由 core 验证，完整流程身份由 pipeline 验证。」
    """

    #: 已构建好的 System / Topology（由 pipeline 传入，engine 不碰内容）
    system: Any
    topology: Any
    #: 有序状态表。顺序即身份——落盘的 rep{i} 对应 states[i]。
    states: Sequence[ThermodynamicStateSpec]
    #: 各副本的初始构型／速度／盒矢量。长度必须等于状态数。
    initial_positions: Sequence[Any]
    initial_box_vectors: Sequence[Any]
    initial_velocities: Optional[Sequence[Any]] = None
    #: 积分器与系综
    temperature_kelvin: float = 298.15
    pressure_bar: Optional[float] = None
    timestep_fs: float = 2.0
    #: 步数
    total_md_steps: int = 0
    exchange_interval: int = 1000
    save_interval: int = 1000
    #: seed 计划。engine 不生成 seed，只转交并登记（§4.2 的 seed ledger 可审计映射）
    seed_plan: dict = field(default_factory=dict)
    #: 平台预算
    platform_name: str = "CUDA"
    max_resident_contexts: Optional[int] = None
    #: 输出目标与调用方协议指纹
    output_dir: str = ""
    stage_name: str = ""
    caller_protocol_fingerprint: str = ""

    @property
    def n_states(self) -> int:
        return len(self.states)

    def step_plan(self) -> StepPlan:
        return resolve_step_plan(
            self.total_md_steps, self.exchange_interval, self.save_interval
        )

    def validate(self) -> None:
        """契约校验。**在创建任何 Context 之前**调用，全部 fail-closed。"""
        if self.n_states < 2:
            # 继承本项目既有的"至少两个状态"输入校验（计划 §6 验证清单第 5 条）
            raise SamplingContractError(
                f"至少需要 2 个热力学状态：收到 {self.n_states}"
            )

        indices = [s.index for s in self.states]
        if indices != list(range(self.n_states)):
            raise SamplingContractError(
                f"状态 index 必须是 0..{self.n_states - 1} 的连续升序：收到 {indices}"
            )

        # 🔑 计划 §4：官方状态是 list[dict]，要求「每个 dict 的键集合相同」。
        # 键集合不同意味着某些副本会缺参数或多参数，而官方 API 不会替你报错。
        key_sets = [frozenset(s.parameters) for s in self.states]
        if len(set(key_sets)) != 1:
            missing = sorted(set().union(*key_sets) - set().intersection(*key_sets))
            raise SamplingContractError(
                f"所有状态必须有相同的参数键集合；这些键只出现在部分状态里：{missing}"
            )
        if not key_sets[0]:
            raise SamplingContractError("状态参数表为空——没有任何可调的全局参数")

        for state in self.states:
            for key, value in state.parameters.items():
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise SamplingContractError(
                        f"状态 {state.index} 的参数 {key!r} 不是实数：{value!r}"
                    )
                if not math.isfinite(float(value)):
                    raise SamplingContractError(
                        f"状态 {state.index} 的参数 {key!r} 非有限值：{value!r}"
                    )

        for label, seq in (
            ("initial_positions", self.initial_positions),
            ("initial_box_vectors", self.initial_box_vectors),
        ):
            if len(seq) != self.n_states:
                raise SamplingContractError(
                    f"{label} 长度 {len(seq)} 与状态数 {self.n_states} 不一致——"
                    "每个副本都必须有自己的初始构型，不能靠后端克隆一份了事"
                )
        if self.initial_velocities is not None and len(self.initial_velocities) != self.n_states:
            raise SamplingContractError(
                f"initial_velocities 长度 {len(self.initial_velocities)} 与状态数不一致"
            )

        if not (self.temperature_kelvin > 0):
            raise SamplingContractError(
                f"temperature_kelvin 必须为正：{self.temperature_kelvin}"
            )
        if self.pressure_bar is not None and not (self.pressure_bar > 0):
            raise SamplingContractError(f"pressure_bar 若给出必须为正：{self.pressure_bar}")
        if not (self.timestep_fs > 0):
            raise SamplingContractError(f"timestep_fs 必须为正：{self.timestep_fs}")

        if self.total_md_steps < 1:
            raise SamplingContractError(
                f"total_md_steps 必须为正：{self.total_md_steps}"
            )
        self.step_plan()  # 触发步数校验与对账

        if not self.output_dir:
            raise SamplingContractError("output_dir 未指定")
        if not self.stage_name:
            raise SamplingContractError("stage_name 未指定——它决定落盘文件名")
        if not self.caller_protocol_fingerprint:
            raise SamplingContractError(
                "caller_protocol_fingerprint 未指定——采样产物必须能追溯到调用方协议，"
                "否则换了 Hamiltonian 也认不出来"
            )


@dataclass(frozen=True)
class SamplingArtifacts:
    """一次采样的产物（RBFE 计划 §4.1 的 `SamplingArtifacts` 契约）。

    🔑 `trajectory_files[i]` 是**热力学状态 i** 的轨迹，不是物理 replica i 的轨迹。
    REMD 计划 §2 特别点名了这一条：现有 `{stage}_rep{i}.dcd` 的命名看着像 replica，
    实际内容按状态分流，**新后端不得改成物理 replica 轨迹**。
    """

    trajectory_files: tuple
    n_states: int
    step_plan: StepPlan
    #: 实际使用的后端决议
    backend: BackendResolution
    #: 交换/混合诊断。engine 不解释内容，原样转交。
    diagnostics: dict = field(default_factory=dict)
    #: checkpoint 类型：'none' / 'coordinate_snapshot' / 'full_context'
    #: 计划 §4.3 明确：官方 reporter 的 checkpoint 是 serialized State，
    #: **不能视为包含完整 RNG 的二进制 Context checkpoint**。所以类型必须显式记录。
    checkpoint_kind: str = "none"
    caller_protocol_fingerprint: str = ""

    def __post_init__(self) -> None:
        if len(self.trajectory_files) != self.n_states:
            raise SamplingContractError(
                f"轨迹文件数 {len(self.trajectory_files)} 与状态数 {self.n_states} 不一致——"
                "每个热力学状态必须恰好有一个轨迹"
            )
        if self.checkpoint_kind not in ("none", "coordinate_snapshot", "full_context"):
            raise SamplingContractError(f"未知 checkpoint_kind：{self.checkpoint_kind!r}")

    def to_provenance(self) -> dict:
        payload = {
            "n_states": int(self.n_states),
            "total_md_steps": int(self.step_plan.total_md_steps),
            "exchange_interval": int(self.step_plan.exchange_interval),
            "save_interval": int(self.step_plan.save_interval),
            "full_iterations": int(self.step_plan.full_iterations),
            "tail_md_steps": int(self.step_plan.tail_md_steps),
            "checkpoint_kind": self.checkpoint_kind,
            "caller_protocol_fingerprint": self.caller_protocol_fingerprint,
            "trajectory_files_are_per_state_not_per_replica": True,
        }
        payload.update(self.backend.to_provenance())
        return payload


def run_sampling(
    request: SamplingRequest,
    sampler: Any,
    resolution: BackendResolution,
) -> SamplingArtifacts:
    """按契约执行一次采样，并把结果规整成 `SamplingArtifacts`。

    `sampler` 是**已经由调用方构造好**的采样引擎（legacy 路径就是
    `ibs_engine.REMDManager` 的实例）。engine 不构造它、不 import 它的模块——
    这是依赖方向的硬约束。对 sampler 的唯一要求是计划 §4.0 定下的语义：

        run(n_steps, exchange_interval, save_interval, stage_name) -> traj_files

    官方后端路径（P1）尚未实现，这里显式拒绝，不走到一半才发现。
    """
    request.validate()

    if resolution.is_official:
        raise NotImplementedError(
            "官方 REMD 后端适配器尚未实现（PLAN_openmm_8_6_remd_backend.md P1）。"
            "resolve_remd_backend 正常情况下不会解析出 openmm，"
            "走到这里说明有人手工构造了 BackendResolution。"
        )

    if not callable(getattr(sampler, "run", None)):
        raise SamplingContractError(
            f"sampler 必须提供 run(n_steps, exchange_interval, save_interval, stage_name)："
            f"收到 {type(sampler).__name__}"
        )

    plan = request.step_plan()
    traj_files = sampler.run(
        n_steps=plan.total_md_steps,
        exchange_interval=plan.exchange_interval,
        save_interval=plan.save_interval,
        stage_name=request.stage_name,
    )
    if traj_files is None:
        raise SamplingContractError("sampler.run() 返回 None——采样失败必须抛异常，不能返回空")

    return SamplingArtifacts(
        trajectory_files=tuple(traj_files),
        n_states=request.n_states,
        step_plan=plan,
        backend=resolution,
        diagnostics=dict(getattr(sampler, "exchange_diagnostics", {}) or {}),
        checkpoint_kind=str(getattr(sampler, "checkpoint_kind", "none")),
        caller_protocol_fingerprint=request.caller_protocol_fingerprint,
    )


# === 未实现区 =============================================================
#
# 计划里属于本模块、但**尚未实现**的部分。不要在这里放"先返回成功"的占位实现——
# RBFE 计划 §4.1 明确禁止「不提供尚未实现但返回成功的占位 sampler」。
#
#   P1（等 OpenMM 升级到 >= 8.6）：
#       - 官方 ReplicaExchangeSampler 适配器：状态映射、DCD 桥接、自定义 reporter
#       - 官方 RNG 与本项目 seed ledger 的可审计映射
#       - 翻转 OFFICIAL_REMD_ADAPTER_IMPLEMENTED，把
#         FREE_ENERGY_ENGINE_ADAPTER_PROTOCOL_VERSION 提到 1
#
#   P2′ 剩余（不需要 8.6）：
#       - 上面的契约**尚未接进生产**：abfe_pipeline.py 的三个 REMDManager 构造点
#         （:4569 / :5500 / :11551）目前仍然直接调用 remd.run(...)，没有经过
#         SamplingRequest / run_sampling。接线时必须逐点验证产物逐字节不变。
#       - 采样指纹（P2）：把 BackendResolution + StepPlan 纳入 resume 身份。
#
# ==========================================================================
