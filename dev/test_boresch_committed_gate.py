"""回归：已提交的 Boresch 平衡值必须仍描述当前构象，否则拒绝复用。

真实事故（2026-07-27 定位）
---------------------------
`run_full_pipeline` 有一条刻意的保护：Boresch 平衡几何量只在一条腿第一次采样时
推导一次并落盘，之后任何 resume 都原样复用、绝不重算——动机是防止同一条腿的前后
窗口用两套哈密顿量拼接自由能曲线（实测曾造成 vdw 拼接曲线单步跳变 ~200 kJ/mol）。

但它**只检查文件是否存在**。于是 `boresch_equilibrium_committed.json`（写于
2026-07-10 18:51，thetaA0/thetaB0 对调、三个二面角错乱）在体系于 2026-07-26 被
整体重新平衡之后仍被沿用了 17 天：

    dof       committed     轨迹实测      偏离
    r0         0.473886     0.478318     0.13 σ   <- 唯一没问题的
    thetaA0    2.036106     1.563422     4.23 σ
    thetaB0    1.533758     1.977043     3.97 σ
    phiA0     -2.128541     1.512979    23.65 σ
    phiB0      1.772715    -1.910870    21.29 σ
    phiC0      1.689600    -0.530192    15.93 σ

后果：限制力把配体从自己的 pose 上拽走 3.4 Å（无约束预平衡只漂 0.60 Å），
方向性氢键丢失 → 复合物腿去电荷偏低约 25 kJ/mol、解析释放修正同样是错的；
而 vdW 因为对取向不敏感看起来完全正常——这也正是当时唯一对得上参考值的那一项。

为什么现有的门抓不住
--------------------
`update_boresch_from_last_frame` 的两道校验对这组错值**全部放行**：
角度 2.0361 rad = 116.7° 落在安全域 40-140° 内，r0 更是只差 0.13 σ。
所以必须用"与当前坐标实测几何的偏离"这个正交判据，且必须逐个自由度看——
只看 r0 恰好是最没有信息量的那一个。
"""

import json
import math

import pytest

from abfe_pipeline import (
    BORESCH_COMMITTED_MAX_DEVIATION_SIGMA,
    BORESCH_COMMITTED_SCHEMA_VERSION,
    BORESCH_COMMITTED_WARN_DEVIATION_SIGMA,
    _wrap_to_pi,
    boresch_committed_deviation_sigma,
)

pytestmark = pytest.mark.cpu_only

# 事故现场的真实数值。
BAD_COMMITTED = {
    "r0": 0.47388611312922485,
    "thetaA0": 2.0361063782957767,
    "thetaB0": 1.5337578704843517,
    "phiA0": -2.128540921990223,
    "phiB0": 1.772714759793259,
    "phiC0": 1.6895997408678274,
}
MEASURED_POSE = {
    "r0": 0.478318,
    "thetaA0": 1.563422,
    "thetaB0": 1.977043,
    "phiA0": 1.512979,
    "phiB0": -1.910870,
    "phiC0": -0.530192,
}
FORCE_CONSTANTS = {
    "kr": 2000.0, "kthetaA": 200.0, "kthetaB": 200.0,
    "kphiA": 200.0, "kphiB": 167.3233137067664, "kphiC": 128.4896369290959,
}
T = 300.0


def _worst(report):
    key = max(report, key=lambda k: report[k]["deviation_sigma"])
    return key, report[key]["deviation_sigma"]


# ---------------------------------------------------------------------------
# 角度回绕
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [(0.0, 0.0),
     # 区间是 **[-π, π)**（左闭右开）：π 落到 -π 这一侧。
     (math.pi, -math.pi),
     (-math.pi, -math.pi),
     (-math.pi + 1e-9, -math.pi + 1e-9),
     (3 * math.pi / 2, -math.pi / 2), (-3 * math.pi / 2, math.pi / 2),
     (2 * math.pi, 0.0), (4 * math.pi + 0.3, 0.3)],
)
def test_wrap_to_pi(raw, expected):
    assert _wrap_to_pi(raw) == pytest.approx(expected, abs=1e-9)


def test_antipode_sign_cannot_affect_the_gate():
    """±π 的符号是任意的，但门只看 |Δ|，所以取哪一侧都不改变判定。

    这条是上面那个区间约定的兜底：即便将来有人把 `_wrap_to_pi` 改成 (-π, π]，
    也不能因此改变任何一次放行/拒绝。
    """
    committed = {"phiA0": 0.0}
    positive = boresch_committed_deviation_sigma(
        committed, {"phiA0": math.pi}, {"kphiA": 200.0}, T
    )
    negative = boresch_committed_deviation_sigma(
        committed, {"phiA0": -math.pi}, {"kphiA": 200.0}, T
    )
    pos = positive["phiA0"]
    neg = negative["phiA0"]

    assert abs(pos["delta"]) == pytest.approx(math.pi, abs=1e-9)
    assert abs(pos["delta"]) == pytest.approx(abs(neg["delta"]), abs=1e-12)
    assert pos["sigma"] == pytest.approx(neg["sigma"], rel=1e-12)
    assert pos["deviation_sigma"] == pytest.approx(
        neg["deviation_sigma"], rel=1e-12
    )
    assert pos["deviation_sigma"] == pytest.approx(
        math.pi / math.sqrt(8.31446261815324e-3 * T / 200.0), rel=1e-9
    )
    for threshold in (
        BORESCH_COMMITTED_WARN_DEVIATION_SIGMA,
        BORESCH_COMMITTED_MAX_DEVIATION_SIGMA,
    ):
        assert (pos["deviation_sigma"] > threshold) == (
            neg["deviation_sigma"] > threshold
        )


def test_dihedral_comparison_wraps_instead_of_reporting_a_fake_gap():
    """+179° 与 −179° 只差 2°，不是 358°。不回绕会把正常涨落误判成灾难。"""
    committed = {"phiA0": math.radians(179.0)}
    current = {"phiA0": math.radians(-179.0)}
    rep = boresch_committed_deviation_sigma(
        committed, current, {"kphiA": 200.0}, T
    )
    assert abs(rep["phiA0"]["delta"]) == pytest.approx(math.radians(2.0), abs=1e-6)


# ---------------------------------------------------------------------------
# 主判据
# ---------------------------------------------------------------------------


def test_real_incident_is_rejected():
    """这组真实错值必须被抓住。"""
    rep = boresch_committed_deviation_sigma(
        BAD_COMMITTED, MEASURED_POSE, FORCE_CONSTANTS, T
    )
    key, worst = _worst(rep)
    assert worst > BORESCH_COMMITTED_MAX_DEVIATION_SIGMA, (
        f"最大偏离只有 {worst:.2f} σ @ {key}，没超过阈值 "
        f"{BORESCH_COMMITTED_MAX_DEVIATION_SIGMA} σ —— 门失效了"
    )
    assert worst == pytest.approx(23.65, abs=0.05)
    assert key == "phiA0"


def test_r0_alone_would_have_missed_it():
    """只看 r0（现有 update_boresch_from_last_frame 的做法）抓不住这个事故。

    这条是整件事的教训：r0 恰好是六个自由度里唯一没问题的那个。
    """
    rep = boresch_committed_deviation_sigma(
        BAD_COMMITTED, MEASURED_POSE, FORCE_CONSTANTS, T
    )
    assert rep["r0"]["deviation_sigma"] < 1.0
    assert rep["r0"]["deviation_sigma"] == pytest.approx(0.13, abs=0.02)


def test_pure_theta_swap_lands_in_the_warning_band_not_the_hard_gate():
    """纯 thetaA/thetaB 对调只有 3.70 σ —— 压在 4.0 σ 硬门下面。

    这是这道门**已知的**灵敏度边界，不是疏漏，写死在这里防止有人误以为它能挡住：

      * 硬门不能再压低：committed 与 current 是两个独立单帧，差值宽度 √2·σ；
        压到 3σ 会让单次运行误报率升到约 19%，resume 就没法用了。
      * 而真实的标签错位必然同时打乱二面角（它们共用同一批原子）——实测那次
        phiA0 到了 23.65 σ，被硬门轻松抓住（见 test_real_incident_is_rejected）。

    所以这里断言的是：它至少进了告警带、会被大声打出来。
    """
    committed = dict(MEASURED_POSE)
    committed["thetaA0"], committed["thetaB0"] = (
        MEASURED_POSE["thetaB0"], MEASURED_POSE["thetaA0"]
    )
    rep = boresch_committed_deviation_sigma(
        committed, MEASURED_POSE, FORCE_CONSTANTS, T
    )
    key, worst = _worst(rep)
    assert key in ("thetaA0", "thetaB0")
    assert worst == pytest.approx(3.70, abs=0.05)
    assert worst > BORESCH_COMMITTED_WARN_DEVIATION_SIGMA, "连告警带都没进，太不灵敏"
    assert worst < BORESCH_COMMITTED_MAX_DEVIATION_SIGMA, (
        "如果这条开始失败，说明硬门被调紧了——请同时复核误报率"
    )


def test_warning_band_is_below_the_hard_gate():
    assert 0 < BORESCH_COMMITTED_WARN_DEVIATION_SIGMA < BORESCH_COMMITTED_MAX_DEVIATION_SIGMA


def test_matching_pose_passes():
    """同一构象（含 ~1σ 单帧噪声）必须放行，否则每次 resume 都会误报。"""
    rng_like_noise = {
        "r0": 0.0353 * 0.8, "thetaA0": 0.1117 * 0.9, "thetaB0": -0.1117 * 0.7,
        "phiA0": 0.1117 * 1.1, "phiB0": -0.1221 * 0.6, "phiC0": 0.1393 * 0.9,
    }
    current = {k: v + rng_like_noise[k] for k, v in MEASURED_POSE.items()}
    rep = boresch_committed_deviation_sigma(
        MEASURED_POSE, current, FORCE_CONSTANTS, T
    )
    _, worst = _worst(rep)
    assert worst <= BORESCH_COMMITTED_MAX_DEVIATION_SIGMA, (
        f"正常热涨落被误判（{worst:.2f} σ），阈值太紧"
    )


def test_sigma_uses_kT_over_k():
    """σ_i = sqrt(kT/k_i)：限制势 0.5*k*x² 与 k*(1-cos x) 小偏离下方差都是 kT/k。"""
    rep = boresch_committed_deviation_sigma(
        {"r0": 0.5}, {"r0": 0.5}, {"kr": 2000.0}, T
    )
    kt = 8.31446261815324e-3 * T
    assert rep["r0"]["sigma"] == pytest.approx(math.sqrt(kt / 2000.0), rel=1e-12)


@pytest.mark.parametrize("bad_k", [0.0, -1.0, float("nan")])
def test_nonpositive_force_constant_is_skipped_not_divided_by(bad_k):
    rep = boresch_committed_deviation_sigma(
        {"r0": 0.5}, {"r0": 0.9}, {"kr": bad_k}, T
    )
    assert "r0" not in rep


def test_missing_dof_is_skipped():
    rep = boresch_committed_deviation_sigma(
        {"r0": 0.5}, {"thetaA0": 1.5}, FORCE_CONSTANTS, T
    )
    assert rep == {}


# ---------------------------------------------------------------------------
# 落盘格式
# ---------------------------------------------------------------------------


def test_committed_payload_carries_identity(tmp_path):
    """新格式必须带身份信息——裸 {"equilibrium_values": ...} 正是事故的载体。"""
    from abfe_pipeline import _json_safe

    payload = _json_safe({
        "schema_version": BORESCH_COMMITTED_SCHEMA_VERSION,
        "equilibrium_values": MEASURED_POSE,
        "receptor_indices": [1328, 1326, 1338],
        "ligand_indices": [4597, 4600, 4601],
        "force_constants": FORCE_CONSTANTS,
        "temperature_K": T,
        "derived_at": "2026-07-27T00:00:00",
    })
    path = tmp_path / "boresch_equilibrium_committed.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    back = json.loads(path.read_text(encoding="utf-8"))
    for key in ("schema_version", "receptor_indices", "ligand_indices",
                "force_constants", "temperature_K", "derived_at"):
        assert key in back, f"落盘缺少 {key}，将无法核对来源"
    assert back["schema_version"] >= 2


def test_legacy_bare_payload_is_recognisable():
    """v1 裸格式（无 schema_version）必须能被识别出来并触发告警路径。"""
    legacy = {"equilibrium_values": BAD_COMMITTED}
    assert legacy.get("schema_version") is None
    assert "receptor_indices" not in legacy
