"""`RunDirectoryLock` —— 2026-09-17 从 `abfe_pipeline.py` 移出（`docs/TODO_P2.md` 的 `REL-09`）。

**不被任何东西 import，也不该被 import。** 留着只为「删了什么、为什么」有出处。

## 为什么删

1. **生产侧零调用点。** 它曾挂在 `runabfe.py::main` → `guard_run_directory()`，
   但那句已被删掉；`guard_run_directory` 现在只装 SIGTERM/SIGINT 处理器。
   删除当天全仓 `grep RunDirectoryLock` 只剩类定义本身与它自己的测试。
2. **删它的理由（`guard_run_directory` docstring 原文）**：进程被 SIGKILL 时锁文件残留，
   而 stale 判定在共享盘上判不出来（PID 跨节点无意义、hostname 不同一律当活着），
   于是 `--resume` 永远进不来 —— **锁挡住的是续跑本身**。
3. **它还带一个机制缺陷**（2026-09-17 复核确认，下面那句 `timeout_s=0.0` 是入口）：

       父类 `_PipelineStateLock.__enter__` 的获取循环是
           deadline = time.time() + self.timeout_s
           while True:
               try:  os.open(..., O_CREAT|O_EXCL);  return self
               except FileExistsError:
                   self._break_stale_lock_if_needed()   # ← 可能刚清掉残留锁
                   if time.time() >= deadline: raise TimeoutError(...)   # ← timeout_s=0 ⟹ 恒真
                   time.sleep(self.poll_s)              # ← 永远走不到

   `timeout_s=0.0` ⟹ 第一次 `FileExistsError` 之后必抛 `TimeoutError`，**永不重试
   `os.open`**。于是 `_break_stale_lock_if_needed()` 即便成功清掉了空壳锁
   （空 payload + mtime 老于 `_EMPTY_PAYLOAD_GRACE_S=30s` 就会清），**这一跑还是死**，
   清理的成果留给下一次启动 —— 与实测「拦一次 → 几十秒后重启就过」吻合。

   ⚠️ 坑不在「不等待」（那是 docstring 明写的设计：两个作业写同一目录是配置错误，
   排队没意义），在于**「不等待」和「不重试」被写成了同一件事**，
   导致 stale 清理路径对它实际失效。

   ⟹ 真要重新接线，**先加一次 break 后的重试**（一个 `broke_once` 标志即可），
   别直接把类贴回去。

   📌 父类 `_PipelineStateLock`（`timeout_s=10.0`）**不受影响**，仍在 `pipeline_state.json`
   上正常使用：它有真实 deadline，break 之后照常 sleep + 重试。

## 同时删掉的测试

`tests/test_run_directory_guard_att23.py` 的第 1 节（5 条锁测试）。同文件的
SIGTERM 与磁盘预检两节**保留** —— 那两项仍在生产里跑（`REL-09` 结案依据）。
端到端那条改名为 `test_sigterm_becomes_terminationrequested_in_a_real_process`，
去掉了锁文件断言。
"""

# ruff: noqa


# ==========================================================================
# abfe_pipeline.py 行 1715-1779
# ==========================================================================

# class RunDirectoryLock(_PipelineStateLock):
#     """一次作业对 `--output` 目录的**独占锁**（ATT-23 / issue #142）。
#
#     与父类 `_PipelineStateLock` 的区别只有三点，其余（跨主机不删别人的锁、
#     只在**确认** PID 不存在时才判 stale、空 payload 宽限期）**全部复用**：
#
#     1. **不等待**：`timeout_s=0`。两个作业写同一个目录是配置错误，不是竞态，
#        排队等 10 秒没有意义 —— 立刻失败并说清另一个持有者是谁。
#     2. **payload 更全**：额外记 `started_at` 与 `command`，好让错误信息能直接
#        告诉你"另一个是什么时候、用什么命令起的"。
#     3. **活得久**：父类那把锁只活几十毫秒（读改写 `pipeline_state.json`），
#        这把要横跨整次运行。
#
#     ## 为什么需要它
#
#     两个 pipeline 同时写一个 run 目录，产物会互相覆盖：DCD 交叉 append、
#     checkpoint 互相踩、`pipeline_state.json` 后写的赢。而且**没有任何一环会报错**
#     —— 最后得到一份看起来正常、实际混了两次运行的结果。
#     """
#
#     LOCK_BASENAME = ".abfe_run.lock"
#
#     def __init__(self, output_dir: str):
#         super().__init__(
#             os.path.join(output_dir, self.LOCK_BASENAME),
#             timeout_s=0.0,
#             poll_s=0.0,
#         )
#         self.output_dir = output_dir
#
#     def _lock_payload(self) -> str:
#         return json.dumps(
#             {
#                 "pid": int(os.getpid()),
#                 "hostname": self._own_hostname(),
#                 "started_at": datetime.now().isoformat(timespec="seconds"),
#                 "command": " ".join(sys.argv[:8]),
#             },
#             ensure_ascii=False,
#         )
#
#     def _describe_owner(self) -> str:
#         try:
#             with open(self.path, "r", encoding="utf-8") as handle:
#                 payload = json.load(handle)
#         except Exception:
#             return "（锁文件存在但读不出内容）"
#         return (
#             f"pid={payload.get('pid')} @ {payload.get('hostname')}，"
#             f"起于 {payload.get('started_at')}，命令: {payload.get('command')}"
#         )
#
#     def __enter__(self):
#         os.makedirs(self.output_dir, exist_ok=True)
#         try:
#             return super().__enter__()
#         except TimeoutError:
#             raise RuntimeError(
#                 f"输出目录已被另一次运行独占：{self.output_dir}\n"
#                 f"    持有者：{self._describe_owner()}\n"
#                 "  两个 pipeline 写同一个目录会互相覆盖产物（DCD 交叉 append、"
#                 "checkpoint 互踩、pipeline_state.json 后写的赢），\n"
#                 "  而且全程不会报错 —— 最后拿到一份看着正常、实际混了两次运行的结果。\n"
#                 f"  换一个 --output，或确认对方确实已经结束后删掉 {self.path}。"
#             ) from None


# ==========================================================================
# tests/test_run_directory_guard_att23.py 第 1 节（5 条锁测试）
# ==========================================================================

# # ---------------------------------------------------------------------------
# # 1. 输出目录独占锁
# # ---------------------------------------------------------------------------
#
#
# def test_second_pipeline_cannot_take_the_same_output_directory(tmp_path):
#     with RunDirectoryLock(str(tmp_path)):
#         with pytest.raises(RuntimeError, match="已被另一次运行独占"):
#             with RunDirectoryLock(str(tmp_path)):
#                 pass
#
#
# def test_lock_error_names_the_current_holder(tmp_path):
#     """错误信息必须能直接回答"另一个是谁、什么时候起的"，否则只能去 ps 里猜。"""
#     with RunDirectoryLock(str(tmp_path)):
#         with pytest.raises(RuntimeError) as excinfo:
#             with RunDirectoryLock(str(tmp_path)):
#                 pass
#     message = str(excinfo.value)
#     assert f"pid={os.getpid()}" in message
#     assert "起于" in message and "命令:" in message
#
#
# def test_lock_is_released_on_exit_and_can_be_retaken(tmp_path):
#     lock_file = tmp_path / RunDirectoryLock.LOCK_BASENAME
#     with RunDirectoryLock(str(tmp_path)):
#         assert lock_file.exists()
#     assert not lock_file.exists()
#     with RunDirectoryLock(str(tmp_path)):
#         pass
#
#
# def test_lock_does_not_wait(tmp_path):
#     """两个作业写同一目录是配置错误、不是竞态，排队等没有意义。"""
#     lock = RunDirectoryLock(str(tmp_path))
#     assert lock.timeout_s == 0.0
#
#
# def test_lock_never_breaks_another_hosts_lock(tmp_path):
#     """共享文件系统上别的节点的 PID 在本机毫无意义 —— 继承自 _PipelineStateLock。"""
#     lock_file = tmp_path / RunDirectoryLock.LOCK_BASENAME
#     lock_file.write_text(
#         json.dumps({"pid": 999999, "hostname": "some-other-node"}), encoding="utf-8"
#     )
#     with pytest.raises(RuntimeError, match="已被另一次运行独占"):
#         with RunDirectoryLock(str(tmp_path)):
#             pass
#     assert lock_file.exists(), "绝不能删掉别的节点的锁"
#
#
