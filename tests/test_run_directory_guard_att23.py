"""ATT-23 剩余三项（GitHub issue #142）：输出目录独占锁 / SIGTERM / 磁盘预检。

三项都为同一类事故服务：**作业以一种事后无从分辨的方式死掉或互相破坏**。

* 两个 pipeline 写同一个 `--output`：DCD 交叉 append、checkpoint 互踩、
  `pipeline_state.json` 后写的赢，全程不报错 —— 最后得到一份混了两次运行、
  看着却完全正常的结果。
* 默认 SIGTERM 直接终止进程，`finally` 一个不跑：锁不释放、Context 不销毁、
  日志里连"我被信号杀了"都没有。2026-09-09 那次作业就是这样消失的
  （两个 log 停在同一行、无 traceback）。
* 盘满是在跑了几小时、DCDReporter 写到一半时才发现的，那时轨迹已截断，
  而截断的 DCD 还会被下次 resume 当成"已存在"。

⚠️ **SIGKILL（含 OOM killer）捕不到** —— 内核直接回收，任何语言都装不上处理器。
本文件不假装能防它；那条路要靠内存预算本身（`_host_memory_mib()` 打点、
`compute_u_kn` 的 worker 内存上限）。
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from abfe_pipeline import (
    TerminationRequested,
    ensure_free_disk_for_stage,
    estimate_stage_trajectory_bytes,
    graceful_termination,
)

pytestmark = pytest.mark.cpu_only

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 2. SIGTERM / SIGINT 有序退出
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_signal_becomes_an_exception_so_finally_chains_run(sig):
    ran = []
    with pytest.raises(TerminationRequested):
        with graceful_termination(log=lambda _m: None):
            try:
                os.kill(os.getpid(), sig)
            finally:
                ran.append("cleanup")
    assert ran == ["cleanup"], "既有的 finally 清理链必须照常执行"


def test_previous_handlers_are_restored_afterwards():
    before = signal.getsignal(signal.SIGTERM)
    with graceful_termination(log=lambda _m: None):
        assert signal.getsignal(signal.SIGTERM) is not before
    assert signal.getsignal(signal.SIGTERM) is before


def test_termination_message_does_not_overclaim_checkpointing():
    """必须写明它**没有**新增 stage 中途 checkpoint —— 否则会被当成"随便 kill"。"""
    with pytest.raises(TerminationRequested) as excinfo:
        with graceful_termination(log=lambda _m: None):
            os.kill(os.getpid(), signal.SIGTERM)
    assert "没有" in str(excinfo.value) and "checkpoint" in str(excinfo.value)


def test_sigterm_becomes_terminationrequested_in_a_real_process(tmp_path):
    """端到端：真进程里 `guard_run_directory` 装上的处理器把 SIGTERM 变成异常。

    在**子进程**里跑而不是本进程：`signal.signal` 是进程全局的，本进程装了处理器
    会污染同一次 pytest 里后面的测试。
    2026-09-17 起这条不再断言锁文件 —— `RunDirectoryLock` 已删（见 archive/）。
    """
    script = textwrap.dedent(
        f"""
        import os, sys, signal
        sys.path.insert(0, {str(REPO)!r})
        from abfe_pipeline import guard_run_directory, TerminationRequested
        guard_run_directory({str(tmp_path)!r}, log=lambda m: None)
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except TerminationRequested:
            sys.exit(3)
        sys.exit(0)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    assert result.returncode == 3, (
        f"子进程未按 TerminationRequested 退出：rc={result.returncode}\\n"
        f"{result.stderr[-2000:]}"
    )


# ---------------------------------------------------------------------------
# 3. 磁盘预检
# ---------------------------------------------------------------------------


def test_disk_precheck_passes_when_there_is_room(tmp_path, capsys):
    ensure_free_disk_for_stage(str(tmp_path), 1024, "tiny stage")
    assert "磁盘预检" in capsys.readouterr().out


def test_disk_precheck_fails_closed_before_the_stage_starts(tmp_path):
    with pytest.raises(RuntimeError, match="磁盘空间不足"):
        ensure_free_disk_for_stage(str(tmp_path), 10 ** 18, "huge stage")


def test_disk_precheck_demands_a_two_times_margin(tmp_path):
    """余量是给 checkpoint、能量日志和同盘上别的作业留的，不是凑数。"""
    import shutil

    free = shutil.disk_usage(str(tmp_path)).free
    # 刚好等于剩余空间的一半 → 需要 2× 正好等于 free，应当通过。
    ensure_free_disk_for_stage(str(tmp_path), free // 2 - 1, "just fits")
    # 略多于一半 → 2× 超过 free，必须拒绝。
    with pytest.raises(RuntimeError, match="磁盘空间不足"):
        ensure_free_disk_for_stage(str(tmp_path), free // 2 + free // 10, "just over")


def test_trajectory_size_estimate_scales_with_every_factor():
    base = estimate_stage_trajectory_bytes(
        n_states=8, n_steps=250000, save_interval=5000, n_atoms=30710
    )
    assert base > 0
    for kwargs in (
        dict(n_states=16, n_steps=250000, save_interval=5000, n_atoms=30710),
        dict(n_states=8, n_steps=500000, save_interval=5000, n_atoms=30710),
        dict(n_states=8, n_steps=250000, save_interval=2500, n_atoms=30710),
        dict(n_states=8, n_steps=250000, save_interval=5000, n_atoms=61420),
    ):
        assert estimate_stage_trajectory_bytes(**kwargs) == pytest.approx(
            base * 2, rel=0.02
        ), f"翻倍任一因子应当让估算翻倍: {kwargs}"


def test_disk_precheck_does_not_block_the_run_when_usage_is_unreadable(tmp_path):
    """读不到用量时只告警不拦路 —— 预检本身不该成为新的失败源。"""
    ensure_free_disk_for_stage(
        str(tmp_path / "does-not-exist"), 1024, "unreadable", log=lambda _m: None
    )
