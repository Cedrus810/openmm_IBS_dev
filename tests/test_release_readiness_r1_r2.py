"""RELEASE_READINESS_2026-08-31.md 的 R1 / R2 回归。

R1：配置里的 `repeat_seed` 与 IBS 主入口实际传参不一致 —— 用户在 JSON 里写了 seed、
    配置快照记着它，而 IBS 管线拿到的是 `None`（旧 `main()` 只读环境变量），
    traditional 路径却读 `config.get("repeat_seed")`。现在集中解析一次，
    优先级 `--seed > ABFE_RANDOM_SEED > config repeat_seed`，并回写 config.data。

R2：`_PipelineStateLock._pid_is_alive()` 把 `except PermissionError` 写在
    `except OSError` **之后**，而前者是后者的子类 —— 那个 `return True` 分支不可达，
    于是"权限不足、判不了"被当成"进程已退出"，stale-lock 清理可能删掉仍在使用的锁。
    连带修掉建锁/写 PID 之间的竞态，以及共享文件系统上跨节点 PID 不可比的问题。
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import socket
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only


REPO = Path(__file__).resolve().parents[1]
RUNABFE_PATH = REPO / "runabfe.py"
PIPELINE_PATH = REPO / "abfe_pipeline.py"


# ---------------------------------------------------------------------------
# R1 —— 单一 resolved seed
# ---------------------------------------------------------------------------


def _compile_runabfe_top_level(*names: str):
    """按名字从 runabfe.py 里抽出顶层函数单独编译。

    刻意不 import runabfe：它在模块导入期就 import OpenMM/pymbar，而这几个被测函数
    是纯参数解析、不需要任何 MD 依赖。
    """
    tree = ast.parse(RUNABFE_PATH.read_text(encoding="utf-8"), filename=str(RUNABFE_PATH))
    wanted = set(names)
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {n.name for n in nodes} == wanted, (
        f"runabfe.py 里找不到 {wanted - {n.name for n in nodes}}"
    )
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "os": os,
        "argparse": argparse,
        "Any": object,
        "Optional": object,
        "Tuple": tuple,
        "RunConfig": object,
    }
    exec(compile(module, str(RUNABFE_PATH), "exec"), namespace)
    return [namespace[name] for name in names]


class _Config:
    """最小 RunConfig 替身：`_resolve_repeat_seed` 只用到 `.get()`。"""

    def __init__(self, data=None):
        self.data = dict(data or {})

    def get(self, key, default=None):
        return self.data.get(key, default)


@pytest.fixture
def resolve_repeat_seed():
    _coerce, _resolve = _compile_runabfe_top_level(
        "_coerce_positive_seed", "_resolve_repeat_seed"
    )
    return _resolve


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("ABFE_RANDOM_SEED", raising=False)


def _args(seed=None):
    return argparse.Namespace(seed=seed)


def test_no_seed_anywhere_keeps_historical_behaviour(resolve_repeat_seed):
    """★三处都不给时必须返回 None —— 不许为旧 resume 偷补新 seed。"""
    assert resolve_repeat_seed(_args(), _Config()) == (None, None)
    assert resolve_repeat_seed(_args(), _Config({"repeat_seed": None})) == (None, None)
    assert resolve_repeat_seed(_args(), _Config({"repeat_seed": ""})) == (None, None)


def test_config_repeat_seed_requires_explicit_confirmation(resolve_repeat_seed):
    """★[2026-09-01] config 里的 seed **不再静默采用**，要求显式确认。

    R1 要求"建立唯一 resolved seed"，但同一段还有一句同等重要的：
    「保留未显式设置 seed 的历史行为；不要为旧 resume 偷补新 seed 或**更改随机流**。」
    把 config 的 repeat_seed 接进管线是对的（以前 IBS 管线根本收不到它），但对一个
    过去靠 env 或不带 seed 跑出来的 output 目录，直接采用等于在用户没表态时改掉它的
    随机流 —— 而随机流决定"独立重复"的身份。
    """
    with pytest.raises(SystemExit) as exc:
        resolve_repeat_seed(_args(), _Config({"repeat_seed": 12345}))
    msg = str(exc.value)
    # 报错必须给出可执行的两条出路，而不只是拒绝
    assert "12345" in msg
    assert "--seed" in msg and "ABFE_RANDOM_SEED" in msg
    assert "删掉" in msg


def test_explicit_intent_never_triggers_the_confirmation(resolve_repeat_seed, monkeypatch):
    """CLI / env 都是显式意图，即使 config 也写了 seed 也不得触发确认。"""
    assert resolve_repeat_seed(
        _args(seed=111), _Config({"repeat_seed": 12345})
    ) == (111, "cli:--seed")
    monkeypatch.setenv("ABFE_RANDOM_SEED", "222")
    assert resolve_repeat_seed(
        _args(), _Config({"repeat_seed": 12345})
    ) == (222, "env:ABFE_RANDOM_SEED")


def test_env_overrides_config(resolve_repeat_seed, monkeypatch):
    monkeypatch.setenv("ABFE_RANDOM_SEED", "777")
    seed, source = resolve_repeat_seed(_args(), _Config({"repeat_seed": 12345}))
    assert (seed, source) == (777, "env:ABFE_RANDOM_SEED")


def test_cli_overrides_env_and_config(resolve_repeat_seed, monkeypatch):
    monkeypatch.setenv("ABFE_RANDOM_SEED", "777")
    seed, source = resolve_repeat_seed(_args(seed=99), _Config({"repeat_seed": 12345}))
    assert (seed, source) == (99, "cli:--seed")


@pytest.mark.parametrize("bad", [0, -3, "abc", 1.5])
def test_illegal_seed_fails_at_startup(resolve_repeat_seed, bad):
    """非正整数必须在启动期 SystemExit，不能带着坏 seed 去烧 GPU。"""
    with pytest.raises(SystemExit):
        resolve_repeat_seed(_args(), _Config({"repeat_seed": bad}))


@pytest.mark.parametrize("bad", ["abc", "-1", "0"])
def test_illegal_env_seed_fails_at_startup(resolve_repeat_seed, monkeypatch, bad):
    monkeypatch.setenv("ABFE_RANDOM_SEED", bad)
    with pytest.raises(SystemExit):
        resolve_repeat_seed(_args(), _Config())


def test_main_writes_the_resolved_seed_back_into_the_config_snapshot():
    """两种模式必须从同一个值出发：traditional 读 config、IBS 读局部变量。

    源码契约测试 —— 真正执行 `main()` 需要完整 MD 输入。要钉的是"解析结果回写
    config.data"这一步存在，因为它正是让两条路径同源的机制。
    """
    body = RUNABFE_PATH.read_text(encoding="utf-8")
    assert "_repeat_seed, _repeat_seed_source = _resolve_repeat_seed(args, config)" in body
    assert 'config.data["repeat_seed"] = _repeat_seed' in body
    assert 'config.data["repeat_seed_source"] = _repeat_seed_source' in body
    # traditional 侧仍然读 config —— 回写之后它拿到的就是 resolved 值。
    assert 'repeat_seed=config.get("repeat_seed")' in body
    # 旧的"只读环境变量"实现必须彻底消失。
    assert '_repeat_seed_raw = os.environ.get("ABFE_RANDOM_SEED")' not in body


def test_seed_cli_flag_exists():
    body = RUNABFE_PATH.read_text(encoding="utf-8")
    assert 'parser.add_argument("--seed"' in body


# ---------------------------------------------------------------------------
# R2 —— 状态锁
# ---------------------------------------------------------------------------


def _compile_lock_class():
    """把 `_PipelineStateLock` 单独编译出来（abfe_pipeline 导入期需要 OpenMM）。"""
    tree = ast.parse(PIPELINE_PATH.read_text(encoding="utf-8"), filename=str(PIPELINE_PATH))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "_PipelineStateLock"
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "os": os,
        "time": time,
        "json": json,
        "socket": socket,
        "Optional": object,
        "Tuple": tuple,
    }
    exec(compile(module, str(PIPELINE_PATH), "exec"), namespace)
    return namespace["_PipelineStateLock"]


@pytest.fixture
def Lock():
    return _compile_lock_class()


def test_permission_error_means_alive(Lock, monkeypatch):
    """★R2 的核心：判不了归属时必须算"活着"，不能据此删锁。"""
    def _kill(_pid, _sig):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "kill", _kill)
    assert Lock._pid_is_alive(4242) is True


def test_process_lookup_error_is_the_only_stale_signal(Lock, monkeypatch):
    def _kill(_pid, _sig):
        raise ProcessLookupError(3, "No such process")

    monkeypatch.setattr(os, "kill", _kill)
    assert Lock._pid_is_alive(4242) is False


def test_other_oserror_is_treated_as_alive(Lock, monkeypatch):
    """EINVAL 之类说明我们判不了，保守当活着。"""
    def _kill(_pid, _sig):
        raise OSError(22, "Invalid argument")

    monkeypatch.setattr(os, "kill", _kill)
    assert Lock._pid_is_alive(4242) is True


def test_nonpositive_pid_is_not_alive(Lock):
    assert Lock._pid_is_alive(0) is False
    assert Lock._pid_is_alive(-1) is False


def test_acquire_writes_a_json_payload_with_hostname(Lock, tmp_path):
    path = tmp_path / "pipeline_state.lock"
    with Lock(str(path)):
        payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["pid"] == os.getpid()
    assert payload["hostname"] == socket.gethostname()
    # 正常释放必须删掉锁文件。
    assert not path.exists()


def test_legacy_bare_pid_payload_is_still_parsed(Lock, tmp_path):
    path = tmp_path / "pipeline_state.lock"
    path.write_text(str(os.getpid()), encoding="utf-8")
    lock = Lock(str(path))
    pid, hostname, present = lock._read_lock_owner()
    assert (pid, hostname, present) == (os.getpid(), None, True)


def test_a_live_lock_is_never_broken(Lock, tmp_path):
    """自己的 PID 一定活着 → 锁必须留着，__enter__ 应超时而不是抢锁。"""
    path = tmp_path / "pipeline_state.lock"
    path.write_text(
        json.dumps({"pid": os.getpid(), "hostname": socket.gethostname()}),
        encoding="utf-8",
    )
    with pytest.raises(TimeoutError):
        with Lock(str(path), timeout_s=0.2, poll_s=0.01):
            pass
    assert path.exists(), "活进程持有的锁被删了"


def test_a_foreign_hostname_lock_is_never_broken(Lock, tmp_path, monkeypatch):
    """共享文件系统：别的节点的 PID 在本机毫无意义，绝不据此删锁。

    刻意让 os.kill 报"进程不存在"——本机确实没有这个 PID，但它属于另一台机器。
    """
    def _kill(_pid, _sig):
        raise ProcessLookupError(3, "No such process")

    monkeypatch.setattr(os, "kill", _kill)
    path = tmp_path / "pipeline_state.lock"
    path.write_text(
        json.dumps({"pid": 999999, "hostname": "some-other-node"}), encoding="utf-8"
    )
    with pytest.raises(TimeoutError):
        with Lock(str(path), timeout_s=0.2, poll_s=0.01):
            pass
    assert path.exists(), "另一节点持有的锁被删了"


def test_a_dead_local_lock_is_broken(Lock, tmp_path, monkeypatch):
    """确认已退出、且是本机的锁 —— 这才是唯一该清理的情形。"""
    def _kill(_pid, _sig):
        raise ProcessLookupError(3, "No such process")

    monkeypatch.setattr(os, "kill", _kill)
    path = tmp_path / "pipeline_state.lock"
    path.write_text(
        json.dumps({"pid": 999999, "hostname": socket.gethostname()}), encoding="utf-8"
    )
    with Lock(str(path), timeout_s=1.0, poll_s=0.01):
        payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["pid"] == os.getpid()


def test_fresh_empty_payload_is_not_treated_as_stale(Lock, tmp_path):
    """★建锁与写 PID 之间的竞态：刚创建的空锁文件不能被别人当残留删掉。"""
    path = tmp_path / "pipeline_state.lock"
    path.write_bytes(b"")            # A 刚 O_EXCL 建好、还没写 payload
    with pytest.raises(TimeoutError):
        with Lock(str(path), timeout_s=0.2, poll_s=0.01):
            pass
    assert path.exists(), "竞态窗口里的锁被误删 → 双持锁"


def test_an_old_empty_payload_is_still_cleanable(Lock, tmp_path):
    """被 kill -9 留下的空壳锁仍要能清理，否则目录永久锁死。"""
    path = tmp_path / "pipeline_state.lock"
    path.write_bytes(b"")
    old = time.time() - (Lock._EMPTY_PAYLOAD_GRACE_S + 60.0)
    os.utime(path, (old, old))
    with Lock(str(path), timeout_s=1.0, poll_s=0.01):
        payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["pid"] == os.getpid()


def test_unreachable_permission_error_branch_is_gone():
    """源码契约：`except PermissionError` 不能再排在 `except OSError` 之后。"""
    body = PIPELINE_PATH.read_text(encoding="utf-8")
    i = body.index("def _pid_is_alive(")
    j = body.index("def _own_hostname(", i)
    fn = body[i:j]
    assert "except ProcessLookupError:" in fn
    assert fn.index("except PermissionError:") < fn.index("except OSError:"), (
        "PermissionError 是 OSError 的子类，必须排在前面，否则那个分支不可达"
    )
