"""湿空腔种子搜索的落盘 + 种子身份（0831issue.md P1 / HANDOFF D3）。

旧实现两个毛病：
  (a) 两个 seed 默认值是写死常量、唯一调用方从不传参 ⇒ 不同 repeat_seed 产生逐字节
      相同的 50 万步随机流，湿臂样本跨 repeat 相关、重复间离散度被系统性低估。
  (b) 该搜索在 manifest 校验**之前**无条件执行且结论不落盘 ⇒ 每次 resume 重跑 50 万步。
落盘 + key 带种子同时治两面：命中就不跑（治 b），换种子必然不命中（治 a）。
"""

import numpy as np
import pytest

pytest.importorskip("openmm")

from openmm import unit  # noqa: E402

from ibs_engine import (  # noqa: E402
    WET_CAVITY_SEED_CACHE_PROTOCOL_VERSION,
    build_wet_cavity_seed_key,
    load_wet_cavity_seed_cache,
    save_wet_cavity_seed_cache,
)


def _key(**over):
    base = dict(
        stage_type="vdw", common_system_xml="<System/>", cv_xml_decoupled="<CV/>",
        temperature_K=300.0, platform_name="CUDA", equilibration_steps=500_000,
        check_interval=5_000, cavity_probe_radius_nm=0.24, cavity_wet_min_waters=1,
        integrator_seed=111, velocity_seed=222,
        seed_identity={"repeat_seed": 20260905, "leg": "complex"},
    )
    base.update(over)
    return build_wet_cavity_seed_key(**base)


def _negative_result():
    """"没找到湿盆" —— 这也是一个有效的、必须能被复用的结论。"""
    return {
        "reached_wet": False,
        "positions": np.zeros((5, 3)) * unit.nanometer,
        "box_vectors": np.eye(3) * unit.nanometer,
        "cavity_water_counts": [0] * 100,
        "equilibration_steps": 500_000,
        "dynamics_hamiltonian": "U_common_plus_single_cv_int_no_wca",
    }


def test_empty_dir_is_a_miss(tmp_path):
    assert load_wet_cavity_seed_cache(str(tmp_path), _key()) is None


def test_negative_conclusion_is_persisted_and_reusable(tmp_path):
    """只存"找到了"会让"没找到"每次都重搜 50 万步。"""
    save_wet_cavity_seed_cache(str(tmp_path), _key(), _negative_result())
    hit = load_wet_cavity_seed_cache(str(tmp_path), _key())
    assert hit is not None
    assert hit["reached_wet"] is False
    assert len(hit["cavity_water_counts"]) == 100
    assert hit["equilibration_steps"] == 500_000
    assert hit["cache"]["hit"] is True
    assert hit["cache"]["protocol_version"] == WET_CAVITY_SEED_CACHE_PROTOCOL_VERSION


@pytest.mark.parametrize("over", [
    {"integrator_seed": 999},
    {"velocity_seed": 999},
    {"seed_identity": {"repeat_seed": 20260906, "leg": "complex"}},
])
def test_a_different_seed_never_inherits_the_conclusion(tmp_path, over):
    """★核心：换 repeat_seed 必须重新搜索。

    否则一次随机搜索的失败会被固化成体系的物理属性——把"这颗种子没找到湿盆"
    冒充成"这个体系没有湿盆"。
    """
    save_wet_cavity_seed_cache(str(tmp_path), _key(), _negative_result())
    assert load_wet_cavity_seed_cache(str(tmp_path), _key(**over)) is None


@pytest.mark.parametrize("over", [
    {"common_system_xml": "<System2/>"},
    {"cv_xml_decoupled": "<CV2/>"},
    {"temperature_K": 310.0},
    {"equilibration_steps": 1_000_000},
    {"check_interval": 1_000},
    {"cavity_probe_radius_nm": 0.30},
    {"cavity_wet_min_waters": 2},
    {"platform_name": "CPU"},
])
def test_any_dependency_change_invalidates_the_cache(tmp_path, over):
    save_wet_cavity_seed_cache(str(tmp_path), _key(), _negative_result())
    assert load_wet_cavity_seed_cache(str(tmp_path), _key(**over)) is None


def test_positive_conclusion_round_trips_the_configuration(tmp_path):
    """找到湿盆时，那个构型必须能完整取回——否则缓存命中反而丢了起点。"""
    res = _negative_result()
    res["reached_wet"] = True
    res["positions"] = np.arange(15, dtype=float).reshape(5, 3) * unit.nanometer
    res["box_vectors"] = (np.eye(3) * 3.5) * unit.nanometer
    save_wet_cavity_seed_cache(str(tmp_path), _key(), res)

    hit = load_wet_cavity_seed_cache(str(tmp_path), _key())
    assert hit["reached_wet"] is True
    np.testing.assert_allclose(
        hit["positions"].value_in_unit(unit.nanometer),
        np.arange(15).reshape(5, 3),
    )
    np.testing.assert_allclose(
        hit["box_vectors"].value_in_unit(unit.nanometer), np.eye(3) * 3.5
    )


def test_corrupt_payload_fails_closed(tmp_path):
    """写入中途被杀留下截断 npz 时必须判 miss 重搜，不能崩也不能用坏数据。"""
    save_wet_cavity_seed_cache(str(tmp_path), _key(), _negative_result())
    (tmp_path / "wet_seed_state.npz").write_bytes(b"garbage")
    assert load_wet_cavity_seed_cache(str(tmp_path), _key()) is None


def test_key_is_written_after_the_payload(tmp_path):
    """先写 npz、后写 key：否则可能命中一个还没写完的 payload。"""
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "ibs_engine.py"
    body = source.read_text(encoding="utf-8")
    i = body.index("def save_wet_cavity_seed_cache(")
    j = body.index("def prepare_wet_cavity_seed(")
    fn = body[i:j]
    # [0831issue P2] payload 的写法已从 `np.savez(path, ...)` 直写改成
    # `_atomic_save_npz`（temp + os.replace），避免写入中途被 kill 留下截断 npz
    # 让后续每次 resume 都在 np.load 崩。顺序契约（先 payload 后 key）不变，
    # 而且现在更强：key 出现时 payload 不仅存在、还必然是完整的。
    assert "np.savez(" not in fn, "payload 必须走原子写 _atomic_save_npz，不要直写 np.savez"
    assert fn.index("_atomic_save_npz(") < fn.index("wet_seed_key.json"), (
        "必须先写 npz 再写 key"
    )
