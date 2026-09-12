"""preopt 缓存的两层划分：采样层 vs 派生路径层。

## 为什么要拆

原来 preopt 缓存是一整块：`stage2_final_n_states` / `free_energy_densify_points` /
`window_min|max_states` 这类**只影响布点分窗**的参数一改，整份缓存失配，连那份
要跑几十分钟 GPU 的 pilot 测量一起作废重跑。而且更糟——2026-09-10 之前这 5 个键
**根本不在指纹里**，改了它们再 resume，`protocol_match` 仍然成立、整段 Stage 2 会用
**旧 λ 路径**重采，落盘的 protocol_key 记的却是新配置。

拆分之后：

* **第 1 层 采样**：Hamiltonian、采样步数、差分步长、**遍历顺序**、加密探针协议。
  变了 ⟹ 必须重跑 pilot（GPU）。
* **第 2 层 派生路径**：final states、densify、window min/max。
  变了 ⟹ 从同一份 pilot 测量**离线重算**（`redistribute_vanishing_lambda_subdomains`
  是纯函数），不重烧 GPU。

## 两条硬约束（本文件一并钉住）

* **不靠协议版本号失效**，靠第 1 层里的 `pilot_traversal` 语义字段。
* **不放任何自产产物的 sha256**。
"""
from __future__ import annotations

import numpy as np
import pytest

import abfe_pipeline as ap
from abfe_preoptimizer import (
    VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS,
    recompute_vanishing_path_from_cached_pilot,
    redistribute_vanishing_lambda_subdomains,
)

pytestmark = pytest.mark.cpu_only


def _pilot_diagnostics(n_points=9):
    """一份形状真实的 pilot 测量：λ 从 1 降到 0，度规在 λ→1 一侧更陡。"""
    lambdas = np.linspace(1.0, 0.0, n_points)
    metric = 20.0 + 160.0 * lambdas ** 2          # λ≈1 处更难，与实测形状一致
    gradients = -(40.0 + 110.0 * lambdas)         # dU/dλ 单调、全负
    return {
        "pilot_lambdas": [float(x) for x in lambdas],
        "metric_g": [float(x) for x in metric],
        "pilot_points": [
            {
                "mean_dU_dlambda_kJ_mol": float(g),
                "std_dU_dlambda_kJ_mol": 1.0,
                "n_derivative_samples": 600,
                "is_refinement_point": False,
            }
            for g in gradients
        ],
    }


def test_legacy_cache_reads_as_legacy_traversal_and_unknown_derived():
    """旧缓存缺这些键，必须被补成 legacy 值，而不是"缺字段所以全不等"。"""
    legacy = {"kind": "dual_lambda_preopt", "potential_type": "softcore"}
    sampling, derived = ap._split_preopt_protocol_key(legacy)

    assert sampling["pilot_traversal"] == ap._LEGACY_PILOT_TRAVERSAL
    assert sampling["potential_type"] == "softcore"
    assert set(derived) == set(ap._PREOPT_DERIVED_PATH_KEYS)
    assert all(value is None for value in derived.values())


def test_legacy_cache_fails_the_sampling_layer_because_traversal_really_changed():
    """旧缓存在第 1 层失配 —— 这是**对的**，不是误伤。

    2026-09-10 起加密点从相邻高 λ 端点恢复状态后续接采样，旧缓存是从 λ=0 的
    完全解耦构型跳回高 λ 测的。两者测到的分布不同，不能互相冒充。
    """
    legacy_sampling, _ = ap._split_preopt_protocol_key(
        {"kind": "dual_lambda_preopt"}
    )
    fresh_sampling, _ = ap._split_preopt_protocol_key(
        {"kind": "dual_lambda_preopt", "pilot_traversal": ap.PILOT_TRAVERSAL_SEMANTICS}
    )
    assert legacy_sampling != fresh_sampling
    assert fresh_sampling["pilot_traversal"] == ap.PILOT_TRAVERSAL_SEMANTICS


def test_derived_only_change_keeps_the_sampling_layer_intact():
    """只改派生键时，第 1 层必须仍然相等 —— 否则离线重算的机会就没了。"""
    base = {
        "kind": "dual_lambda_preopt",
        "pilot_traversal": ap.PILOT_TRAVERSAL_SEMANTICS,
        "pilot_n_steps_per_state": 15000,
        "stage2_final_n_states": 16,
        "stage2_free_energy_densify_points": 2,
        "stage2_window_min_states": 4,
        "stage2_window_max_states": 5,
        "stage2_refine_extra_points_per_segment": 4,
    }
    changed = dict(base, stage2_final_n_states=12)

    s_base, d_base = ap._split_preopt_protocol_key(base)
    s_changed, d_changed = ap._split_preopt_protocol_key(changed)

    assert s_base == s_changed, "改 final_state_count 不该动到采样层"
    assert d_base != d_changed, "派生层必须察觉到它变了"


def test_no_self_produced_sha256_in_the_derived_layer():
    """派生层只放语义键。自产产物的 sha256 进缓存身份是本仓反复复发的真 bug。"""
    for name in ap._PREOPT_DERIVED_PATH_KEYS:
        assert "sha" not in name.lower(), name
        assert "hash" not in name.lower(), name


def test_offline_recompute_matches_the_pure_redistribution():
    """离线重算必须与主路径**同一个纯函数**给出同一条 λ —— 不是另写一套。"""
    diagnostics = _pilot_diagnostics()
    params = dict(
        n_states=9,
        final_state_count=16,
        min_states_per_window=4,
        max_states_per_window=5,
        free_energy_densify_points=2,
    )
    recomputed = recompute_vanishing_path_from_cached_pilot(diagnostics, **params)

    expected_lambdas, _cum, _edges, expected_ranges, _alloc = (
        redistribute_vanishing_lambda_subdomains(
            [float(x) for x in diagnostics["pilot_lambdas"]],
            np.asarray(diagnostics["metric_g"], dtype=float),
            params["n_states"],
            first_ensemble_target_intervals=VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS,
            final_state_count=params["final_state_count"],
            min_states_per_window=params["min_states_per_window"],
            max_states_per_window=params["max_states_per_window"],
            free_energy_densify_points=params["free_energy_densify_points"],
            pilot_mean_dU_dlambda=np.asarray(
                [p["mean_dU_dlambda_kJ_mol"] for p in diagnostics["pilot_points"]],
                dtype=float,
            ),
        )
    )
    expected = np.clip(np.asarray(expected_lambdas, dtype=float).ravel(), 0.0, 1.0)
    expected[0], expected[-1] = 1.0, 0.0

    np.testing.assert_array_equal(recomputed["lambdas_vdw"], expected)
    assert list(recomputed["window_ranges"]) == list(expected_ranges)
    # 端点必须精确，不是"约等于"——下游把它们当身份用。
    assert recomputed["lambdas_vdw"][0] == 1.0
    assert recomputed["lambdas_vdw"][-1] == 0.0


def test_changing_final_state_count_actually_changes_the_recomputed_path():
    """否则这个"重算"只是把旧结果原样抄回去，等于没修。"""
    diagnostics = _pilot_diagnostics()
    common = dict(
        n_states=9,
        min_states_per_window=4,
        max_states_per_window=5,
        free_energy_densify_points=0,
    )
    a = recompute_vanishing_path_from_cached_pilot(
        diagnostics, final_state_count=16, **common
    )
    b = recompute_vanishing_path_from_cached_pilot(
        diagnostics, final_state_count=12, **common
    )
    assert len(a["lambdas_vdw"]) == 16
    assert len(b["lambdas_vdw"]) == 12


@pytest.mark.parametrize("missing", ["pilot_lambdas", "metric_g", "pilot_points"])
def test_recompute_fails_closed_when_the_cached_pilot_is_incomplete(missing):
    """缺 pilot 测量就抛，**不猜** —— 猜出来的 λ 会静默改变生产态。"""
    diagnostics = _pilot_diagnostics()
    diagnostics.pop(missing)
    with pytest.raises(ValueError, match=missing):
        recompute_vanishing_path_from_cached_pilot(
            diagnostics,
            n_states=9,
            final_state_count=16,
            min_states_per_window=4,
            max_states_per_window=5,
            free_energy_densify_points=0,
        )


def test_recompute_rejects_length_mismatch_between_lambdas_and_metric():
    diagnostics = _pilot_diagnostics()
    diagnostics["metric_g"] = diagnostics["metric_g"][:-1]
    with pytest.raises(ValueError, match="长度不一致"):
        recompute_vanishing_path_from_cached_pilot(
            diagnostics,
            n_states=9,
            final_state_count=16,
            min_states_per_window=4,
            max_states_per_window=5,
            free_energy_densify_points=0,
        )


# ---------------------------------------------------------------------------
# 旧缓存的采样语义：只在真的用过加密点时才判为不同
# ---------------------------------------------------------------------------


def _cache(points, *, recorded_traversal=None):
    protocol_key = {"payload": {"kind": "dual_lambda_preopt"}}
    if recorded_traversal is not None:
        protocol_key["payload"]["pilot_traversal"] = recorded_traversal
    return {
        "protocol_key": protocol_key,
        "path_diagnostics": {"pilot_points": points},
    }


def test_legacy_cache_without_refinement_points_counts_as_current_semantics():
    """加密从未触发 ⟹ 新旧语义测到的是同一批点 ⟹ 不该逼着重烧 GPU。

    遍历语义只作用在加密点上（`_refine_pilot_grid_in_steep_segments` 改的是
    "加密点从哪个构型起步"），主网格的测法一个字没变。
    """
    cached = _cache([{"is_refinement_point": False} for _ in range(9)])
    assert ap._cached_pilot_traversal(cached) == ap.PILOT_TRAVERSAL_SEMANTICS


def test_legacy_cache_with_refinement_points_is_rejected():
    cached = _cache(
        [{"is_refinement_point": False}] * 8 + [{"is_refinement_point": True}]
    )
    assert ap._cached_pilot_traversal(cached) == ap._LEGACY_PILOT_TRAVERSAL


@pytest.mark.parametrize("points", [None, [], "not-a-list"])
def test_unprovable_legacy_cache_fails_closed(points):
    """读不到 pilot_points 就无法证明"没用过加密" ⟹ 一律按 legacy 处理。"""
    cached = _cache(points)
    assert ap._cached_pilot_traversal(cached) == ap._LEGACY_PILOT_TRAVERSAL


def test_recorded_traversal_always_wins_over_the_inference():
    """一旦缓存自己记了标签，就以它为准，不再去猜。"""
    cached = _cache(
        [{"is_refinement_point": False}], recorded_traversal="some_future_scheme_v9"
    )
    assert ap._cached_pilot_traversal(cached) == "some_future_scheme_v9"


# ---------------------------------------------------------------------------
# 生产真实形状：`_protocol_fingerprint()` 外壳，不是裸 payload
# ---------------------------------------------------------------------------
#
# 上面所有用例喂的都是**裸 payload**，而两个生产调用点（`_preopt_protocol_key()`）
# 传的全是 `_protocol_fingerprint()` 的外壳 `{schema_version, sha256, payload}`。
# 2026-09-10 由此发现：整套两层拆分在生产里**从来没生效过** ——
#
#   * 5 个派生键在外壳顶层一个都取不到 ⟹ derived 恒 5 个 None ⟹ _derived_match 恒 True；
#   * sampling 里混进了整个 sha256（派生键一变它就变）、pilot_traversal 又被
#     setdefault 补成 legacy ⟹ _sampling_match 对任何新缓存恒 False。
#
# 两头朝相反方向失效，「第 1 层匹配、只有第 2 层变了」这个条件**永远不成立**。
# 上面那批测试全绿，因为它们喂的形状生产从不产生。
#
# 下面这组钉住生产形状。别把它们改成裸 payload。


def _production_shaped(**overrides):
    """跟 `_preopt_protocol_key()` 一样，返回 `_protocol_fingerprint()` 的外壳。"""
    payload = {
        "kind": "dual_lambda_preopt",
        "pilot_traversal": ap.PILOT_TRAVERSAL_SEMANTICS,
        "pilot_n_steps_per_state": 15000,
        "temperature_K": 300.0,
        "stage2_final_n_states": 16,
        "stage2_free_energy_densify_points": 2,
        "stage2_window_min_states": 4,
        "stage2_window_max_states": 5,
        "stage2_refine_extra_points_per_segment": 4,
    }
    payload.update(overrides)
    return ap._protocol_fingerprint(payload)


def test_split_accepts_the_protocol_fingerprint_envelope():
    """外壳进去，派生层必须拿到**真实的值**，而不是 5 个 None。"""
    sampling, derived = ap._split_preopt_protocol_key(_production_shaped())

    assert derived["stage2_final_n_states"] == 16, (
        "派生层取到了 None —— 说明没有 unwrap ['payload']，两层拆分是死代码"
    )
    assert all(value is not None for value in derived.values())
    assert sampling["pilot_traversal"] == ap.PILOT_TRAVERSAL_SEMANTICS, (
        "pilot_traversal 被 setdefault 成 legacy —— 采样层会对任何新缓存恒不匹配"
    )


def test_envelope_sha256_never_leaks_into_the_sampling_layer():
    """外壳的 sha256 覆盖**整个** payload，派生键一动它就变。

    它要是留在采样层，「只改派生键」就会同时打翻采样层，离线重算的机会没了。
    """
    sampling, _ = ap._split_preopt_protocol_key(_production_shaped())
    assert "sha256" not in sampling
    assert "schema_version" not in sampling
    assert "payload" not in sampling


def test_derived_only_change_is_detectable_through_the_envelope():
    """这条就是整套拆分存在的理由：采样层相等、派生层不等 ⟹ 可以离线重算。"""
    s_base, d_base = ap._split_preopt_protocol_key(_production_shaped())
    s_new, d_new = ap._split_preopt_protocol_key(
        _production_shaped(stage2_final_n_states=12)
    )

    assert s_base == s_new, "只改 final_state_count 却打翻了采样层 ⟹ 白烧一次 pilot"
    assert d_base != d_new, "派生层没察觉 ⟹ 会拿旧 λ 路径重采、落盘却记新配置"


def test_sampling_change_through_the_envelope_still_forces_a_repilot():
    """反方向也要成立，否则这层拆分等于把采样层的保护也拆没了。"""
    s_base, _ = ap._split_preopt_protocol_key(_production_shaped())
    s_hot, _ = ap._split_preopt_protocol_key(_production_shaped(temperature_K=310.0))
    assert s_base != s_hot, "改温度必须重烧 pilot"
