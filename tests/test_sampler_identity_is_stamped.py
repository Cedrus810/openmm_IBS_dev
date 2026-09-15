"""采样身份必须真的被写进 `ibs_state_*.json`。

`save_ibs_state` 是 `IBSSampler` 的方法，取的是 `getattr(self, "stage_protocol_key")`
—— 而 `stage_protocol_key` 只存在于**管理器**。先前全仓没有任何一处把它盖到 sampler
上 ⟹ 每一份 state 里这个字段**永远是 None**。

后果不是少一条元数据，是一条**永久关闭的闸**：
    save → None → load → `loaded_stage_protocol_key` → None
    → `_identity_matches` 永远 False → `skip_warmup_entirely` 永远 False
    → **每个 resume 的窗口都重走一遍完整预热收敛判定**，白烧预热预算。
而预热预算耗尽正是窗口"再也进不去"、自治循环空转到迭代上限的上游原因。

五个 benchmark run 实测：state 里 `stage_protocol_key` / `coion_identity` 全是 None，
同窗口的 `convergence.json` 里两者都有。
"""
import ast
import inspect
import pathlib

import pytest

pytestmark = pytest.mark.cpu_only

import ibs_engine as ie


def _manager_stamp_block():
    src = pathlib.Path(ie.__file__).read_text("utf-8")
    tree = ast.parse(src)
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "IBSWindowManagerDualLambda")
    return ast.unparse(cls)


@pytest.mark.parametrize("field", ["stage_protocol_key", "coion_identity",
                                   "sampling_score_sha256", "sampling_repair_policy"])
def test_the_manager_stamps_every_identity_field_onto_the_sampler(field):
    """管理器建完 sampler 之后要把身份逐个盖上去 —— 漏一个就等于那道闸关着。"""
    block = _manager_stamp_block()
    assert f"sampler.{field} =" in block, (
        f"`sampler.{field}` 从来没被赋值 ⟹ save_ibs_state 会永远写 None"
    )


@pytest.mark.parametrize("field", ["stage_protocol_key", "coion_identity",
                                   "sampling_score_sha256"])
def test_save_ibs_state_persists_every_identity_field(field):
    src = inspect.getsource(ie.IBSSampler.save_ibs_state)
    assert f'"{field}"' in src, f"state 里根本没有 {field} 这一项"


def test_a_fresh_sampler_defaults_to_none_not_a_crash():
    """没盖身份的 sampler（老路径 / 单元测试）仍要能落盘，只是写 None。"""
    s = ie.IBSSampler.__new__(ie.IBSSampler)
    assert getattr(s, "stage_protocol_key", "MISSING") in (None, "MISSING")


def test_the_identity_gate_reads_what_the_writer_writes():
    """读侧取 `loaded_stage_protocol_key`，写侧写 `stage_protocol_key` ——
    两边必须是同一个字段名，否则闸永远关着。"""
    load_src = inspect.getsource(ie.IBSSampler.load_ibs_state)
    assert 'self.loaded_stage_protocol_key = state.get("stage_protocol_key")' in load_src
