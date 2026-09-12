"""回归：Boresch JSON 格式化必须能处理 NumPy 锚点索引。"""

import json

import numpy as np
import pytest

pytest.importorskip("openmm")

from abfe_core import NumpyEncoder, UnitFormatter  # noqa: E402



pytestmark = pytest.mark.cpu_only

def test_format_boresch_json_casts_numpy_anchor_indices():
    params = {
        "boresch_anchors": {
            "receptor_indices": [np.int64(1), np.int64(2), np.int64(3)],
            "ligand_indices": [np.int64(4), np.int64(5), np.int64(6)],
            "equilibrium_values": {
                "r0": 0.4,
                "thetaA0": 2.0,
                "thetaB0": 1.8,
                "phiA0": 0.1,
                "phiB0": -0.2,
                "phiC0": 0.3,
            },
            "force_constants": {
                "kr": 2000.0,
                "kthetaA": 200.0,
                "kthetaB": 200.0,
                "kphiA": 100.0,
                "kphiB": 100.0,
                "kphiC": 100.0,
            },
        },
        "diagnostics": {"source": "regression"},
    }

    formatted = UnitFormatter.format_boresch_json(params)

    # This is intentionally the plain stdlib encoder: formatter output should
    # be directly serializable without relying on a NumPy-aware JSON encoder.
    json.dumps(formatted)
    assert formatted["boresch_anchors"]["receptor_indices"] == [1, 2, 3]
    assert formatted["boresch_anchors"]["ligand_indices"] == [4, 5, 6]


# ---------------------------------------------------------------------------
# 2026-08-31 P1 回归：format_boresch_json 必须幂等
#
# 旧实现只用裸键 `eq.get("r0", 0)` 读取，而自身输出带单位后缀，于是
# format(format(x)) 把 6 个平衡值 + 6 个力常数全部清零（锚点索引却保留，
# 文件结构看起来仍合法）。唯一生产调用方 apply_boresch_correction 正是
# "格式化后写回同一个 boresch_params.json"，所以 complex 腿走 autoload
# 时会静默销毁生产 Boresch 工件。
# ---------------------------------------------------------------------------

def _valid_boresch_params():
    return {
        "equilibrium_values": {
            "r0": 0.55, "thetaA0": 1.2, "thetaB0": 1.5,
            "phiA0": 0.3, "phiB0": -0.7, "phiC0": 2.1,
        },
        "force_constants": {
            "kr": 2000.0, "kthetaA": 200.0, "kthetaB": 200.0,
            "kphiA": 100.0, "kphiB": 100.0, "kphiC": 100.0,
        },
        "receptor_indices": [1, 2, 3],
        "ligand_indices": [4, 5, 6],
    }


def test_format_boresch_json_is_idempotent():
    once = UnitFormatter.format_boresch_json(_valid_boresch_params())
    twice = UnitFormatter.format_boresch_json(once)
    thrice = UnitFormatter.format_boresch_json(twice)
    assert once == twice == thrice
    # 具体钉住那两个会被清零的量，别只比字典相等
    assert twice["boresch_anchors"]["force_constants"]["kr_kJ_mol_nm2"] == 2000.0
    assert twice["boresch_anchors"]["equilibrium_values"]["r0_nm"] == 0.55


def test_format_boresch_json_never_zero_fills_a_missing_key():
    """缺键必须 fail-closed：0 会通过 3+3 锚点检查却让限制力变成空操作。"""
    params = _valid_boresch_params()
    del params["force_constants"]["kphiC"]
    with pytest.raises(ValueError, match="kphiC"):
        UnitFormatter.format_boresch_json(params)


def test_format_boresch_json_does_not_read_deg_as_rad():
    """只认本函数自己写出的后缀；`_deg` 需要换算，不能被当弧度读进来。"""
    params = _valid_boresch_params()
    params["equilibrium_values"] = {"thetaA0_deg": 89.5}
    with pytest.raises(ValueError):
        UnitFormatter.format_boresch_json(params)


# ---------------------------------------------------------------------------
# NumpyEncoder：abfe_core 的这一份是**全局唯一**的那份
# ---------------------------------------------------------------------------


def test_numpy_encoder_keeps_integers_as_integers():
    """`np.integer` 必须序列化成 JSON 整数，不能变成浮点。

    2026-09-09 回归：`abfe_core.py` 里曾有三份**函数内嵌套**的 `NumpyEncoder`
    （`OrbBoreschEstimator.estimate_from_trajectory` /
    `.estimate_multiple_anchors_from_trajectory` / `OnlineConvergenceMonitor.
    export_convergence_data`），它们把 `(np.integer, np.floating)` 一律映射成
    `float`，于是原子索引、帧数、窗口号这类整数会落盘成 `5.0`；其中一份还漏了
    `np.bool_`。它们**遮蔽**了模块级那份正确的实现。已删除三份嵌套定义。
    """
    payload = {
        "index": np.int64(5),
        "count": np.int32(7),
        "value": np.float64(1.5),
        "flag": np.bool_(True),
        "array": np.array([1, 2, 3], dtype=np.int64),
    }
    loaded = json.loads(json.dumps(payload, cls=NumpyEncoder))

    assert loaded["index"] == 5 and isinstance(loaded["index"], int)
    assert loaded["count"] == 7 and isinstance(loaded["count"], int)
    assert isinstance(loaded["value"], float) and loaded["value"] == 1.5
    assert loaded["flag"] is True
    assert loaded["array"] == [1, 2, 3]


def test_abfe_core_defines_exactly_one_numpy_encoder_at_module_level():
    """不允许再出现嵌套的 `NumpyEncoder` —— 嵌套定义会静默遮蔽模块级那份。

    模块级那份是共享实现：`runabfe.py` 与 `ibs_engine.py` 都从 `abfe_core` 导入它，
    `tools/validation/*` 用 `core.NumpyEncoder`。嵌套副本一旦漂移就只在局部产物里出错，
    而且不会有任何报错。
    """
    import ast
    import pathlib

    source = (pathlib.Path(__file__).resolve().parents[1] / "abfe_core.py").read_text(
        encoding="utf-8"
    )
    found = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ClassDef) and node.name == "NumpyEncoder"
    ]
    assert len(found) == 1, f"abfe_core.py 里有 {len(found)} 份 NumpyEncoder，应当只有 1 份"
    assert found[0].col_offset == 0, (
        f"abfe_core.py:{found[0].lineno} 的 NumpyEncoder 是嵌套定义（缩进 "
        f"{found[0].col_offset}），会遮蔽模块级那份"
    )
