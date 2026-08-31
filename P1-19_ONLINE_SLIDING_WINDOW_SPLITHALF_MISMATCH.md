# P1-19 附属发现：在线 TMBAR 滑动窗口下，split-half 诊断的"window 0"不是物理窗口

> 状态：`FIXED`——2026-08-24 会话内定位，2026-08-25 会话修复并通过离线测试
> （`tests/test_format_results_human_leg_label.py` + 扩展后的
> `tests/test_stage_diagnostics_persistence.py::test_populate_stage_diagnostics_carries_every_gate`，
> 用户在真实 `openmm_dev` 环境跑过，全 pass）。
> 关联：memtodolist.md Definition of Done / Phase D1「关闭 P1-19/P1-19b 的跨运行不确定度问题」。
> 触发场景：真实 production 日志（IBS dual_lambda 在线偏置更新阶段）。

## 修复摘要（2026-08-25）

用户带来一份更完整的现场分析，一并确认/处理了 5 条问题，其中 4 条是真实代码 bug（第 3 条是
真实但不算 bug 的统计观察，未改代码）：

1. **本文件的根因**（split-half 把在线滑动 minibatch 列表位置当物理窗口）——
   `ibs_engine.py::solve_stage_integrated` 新增 `skip_split_half_diagnostics` 参数；
   `IBSSampler._solve_tmbar_and_recenter()` 的调用点传 `skip_split_half_diagnostics=True`，
   跳过这条在该场景下没有信息量的诊断。整段生产/最终分析的另外两个调用点
   （`abfe_pipeline.py` 里走真实物理窗口的那条）不受影响，继续正常跑 split-half。
2. **TMBAR 在线更新日志自相矛盾**（`alpha=0.000` 但 `delta_f` 非零；
   `tmbar_self_consistent=True` 与 `method=...untrusted...` 同时出现）——
   `IBSSampler.update_weights()` 里的打印语句改为：`effective_damping` 缺失时打印
   `alpha=n/a(fallback)` 而不是假装 0.000；新增 `trusted=` 字段直接显示
   `tmbar_candidate_trusted`；`tmbar_self_consistent` 改名打印成
   `applied_step_within_selfconsistency_limit`，避免读成"TMBAR 被判定为可信"。
   只改诊断字符串/注释，未改任何判据或算法。
3. **最终收敛属于贴线通过**——纯观察，无代码改动。
4. **生产 split-half 结果没进最终报告**——`abfe_pipeline.py::_populate_stage_diagnostics`
   补上 `split_half_diagnostics`/`split_half_max_window_z`/`split_half_max_z_threshold`/
   `split_half_gate_failed`/`sigma_inflation_from_split_half`/`sigma_inflation_applied`
   六个字段的搬运，使其经 `diagnostics` → `_build_stage_cache_payload` →
   `final_results.json` 的 `stage_diagnostics.stageN` 全程落盘。此前这些字段在
   `solve_stage_integrated` 里算出来就被丢弃。
5. **命名问题**——
   - `ibs_engine.py`：`free_energy_history_kT` 改名为 `free_energy_history_kJ_mol`
     （原字段单位标注错误，`sampler.f_history` 存的其实是物理 F_k，单位 kJ/mol；
     全仓搜索确认没有代码读取旧键名，改名零风险）。
   - `abfe_pipeline.py::compute_final_results` 新增写入 `final["system_type"]`
     （"complex"/"solvent"）；`abfe_core.py::UnitFormatter.format_results_human`
     据此选人类可读标题，修复 solvent 腿被打印成"✅ 复合物总自由能 ΔG_complex"的问题。
     缺 `system_type` 字段的旧产物按 "complex" 处理，行为不变。

新增/扩展的回归测试：`tests/test_format_results_human_leg_label.py`（新文件）、
`tests/test_stage_diagnostics_persistence.py`（扩展 `REQUIRED_DIAGNOSTIC_KEYS` 与其测试）。

## 现象（用户直接观察到的）

在线偏置更新循环里，同一条 split-half 漂移警告**原样重复了 10 次**：

```
⚠️ [split-half] stage 前后半程不一致：window 0 漂移 -1.014 kJ/mol = 3.88×2σ（σ_win=0.131）。
   报出的不确定度低估了实际抽样波动。
ℹ️ [P1-19] stage 若按 σ≥|漂移|/2 定下界：1/1 个窗口的 σ 被抬高，总 σ 0.1308 → 0.5070 kJ/mol (×3.88)。默认未采用。
```

同一时刻紧邻的 `[IBS TMBAR 自洽权重更新 v9]` 那行，`p`/`delta_f`/`raw_residual`/`total_frames` 每次都在变
（`total_frames=40 → 80 → ...`），证明 TMBAR 求解本身确实在吃新数据；只有 split-half 那条 window 0
的数字纹丝不动，直到第 11 次才会变。

## 根因（已定位，未改代码）

`ibs_engine.py` 里在线更新走的是 `IBSSampler._solve_tmbar_and_recenter()`（约 6521 行），它调用
`solve_stage_integrated(self.tmbar_history, self.kt, ...)`。`self.tmbar_history` **不是**越滚越大的
完整历史，而是一个显式限容的滑动窗口：

```python
_TMBAR_SLIDING_WINDOW = 10
if len(self.tmbar_history) > _TMBAR_SLIDING_WINDOW:
    n_drop = len(self.tmbar_history) - _TMBAR_SLIDING_WINDOW
    self.tmbar_history = self.tmbar_history[-_TMBAR_SLIDING_WINDOW:]
    ...
```

`solve_stage_integrated`/`split_half_drift_diagnostics` 里说的"window 0"，在这条调用路径下**不是某个
真实的物理 λ 窗口**，而是这个滑动列表里的**第 0 个位置**——当前还留在窗口里、最早的那一条 minibatch
entry。只要列表还没塞满 10 条（即前 10 次在线更新），位置 0 指向的就是同一个、从来没变过的 minibatch，
split-half 拿它算出来的漂移/σ 自然逐字节相同；第 11 次更新之后，最老的一条才会被挤出去，这条警告的数字
才会真的换一批。

**这不是"读了旧缓存"的实现 bug，是把一个为"整段生产/最终分析里真实物理窗口"设计的诊断，直接套用在
在线滑动窗口学习循环里**——这里的"window 0"跟诊断原本要表达的"某个 λ 窗口前后半程漂不漂"完全不是
同一个概念，报出来的内容对这个场景没有实际信息量，只是噪音，而且噪音频率正好等于滑动窗口容量（10）。

## 尚未回答的问题（留给以后处理这条时先想清楚）

1. 这条诊断在**在线学习循环**里到底要不要跑——如果跑，"window 0"该怎么重新定义才有意义（比如按
   dominant λ 状态而不是按滑动窗口的列表位置分半）？
2. 如果决定继续跑但只是想去掉这种重复刷屏，是该"同一条结论只报一次"，还是干脆在
   `_solve_tmbar_and_recenter` 这条调用路径上跳过 split-half 诊断、只在最终/整段分析
   （`solve_stage_integrated` 的另外两个调用点，约 14423/6104 行附近）里跑？
3. 这与 memtodolist.md D1「关闭 P1-19/P1-19b」是不是同一件事、还是要分开单独记一条——本文件先只记
   现象和根因，不做这个判断。

## 涉及代码位置（供下次接手直接跳转）

- `ibs_engine.py:6483` 附近：`self.tmbar_history` 的滑动窗口截断逻辑（`_TMBAR_SLIDING_WINDOW = 10`）。
- `ibs_engine.py:6521`：`IBSSampler._solve_tmbar_and_recenter()`，在线更新的调用点。
- `ibs_engine.py:14540`：`solve_stage_integrated()` 定义，split-half 诊断和 `[P1-19]` σ 下界打印都在这里。
- `ibs_engine.py:14423` 附近：`split_half_drift_diagnostics()` 内部递归调用 `solve_stage_integrated`
  算前/后半程各自的解。
