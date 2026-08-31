# EXP-030：`frozen_score.json` 快照时机早于生产入口自我修正，导致偶发 f_k 不一致（2026-08-26）

## 状态：已定位、**已修复（2026-08-27）**。见本文件末尾"修复"一节。
## 跟另一份 [`EXP-030_IBS_FK_RESIDUAL_TRAINING_BUG_2026-08-26.md`](EXP-030_IBS_FK_RESIDUAL_TRAINING_BUG_2026-08-26.md) 是两个不同的 bug，不要混。那份已经修了（v30→v31），这份还没有。

## 现象

极少数窗口（目前观察到的是 v31 重跑里 34/36 通过，2 个失败）在生产**完整跑完 250k 步、能量正常保存**之后，最后一步报：

```
EXP-030 window failed: Exp030RunError: production checkpoint f_k differs from frozen score
```

实测这两个窗口之一（`repeat_0 baseline/window_0`）的具体数字：

```
spec.f_k  (frozen_score.json)              = [-46.36, -17.71, 4.59, 22.36, 37.13]
state.f_k (checkpoints/ibs_state_..json)   = [-44.65, -16.99, 4.42, 21.49, 35.72]
```

两边各自都是 mean-centered（各自求和≈0），但不是同一份，差距 0.16~1.7 kJ/mol，不是浮点误差。

## 根因

`scripts/exp030_window_state_machine.py` 的流程是：

1. `manager.run_all_windows(resume=False, warmup_only=True, ...)` —— 校准，f_k 学习+收敛。
2. `_promote_frozen_state(state_path, ...)` —— 把校准出来的 f_k（mean-centered 版本，即 `spec.f_k`）写进 `frozen_score.json`/checkpoint json。**这是"冻结快照"的时刻。**
3. `manager.run_all_windows(resume=True, warmup_only=False, ...)` —— 生产。

但第 3 步内部，进生产之前还会先做一轮"冻结 burn-in + 只读验证"（`ibs_engine.py` 里 `skip_warmup_entirely` 分支附近的逻辑）。这一轮**不是真正只读**：如果 local-MBAR loose gate 第一次没过（`max|Δf_k−ΔF^MBAR| ≥ 10.0 kJ/mol`），代码会对 f_k 做**一次阻尼+pairwise-capped 的自我修正**（"🔧 occupancy 尚可但 local-MBAR gap 未过：对冻结 f_k 应用一次阻尼+pairwise-capped 修正...直接重新验证"），重新验证通过后就拿修正后的 f_k 进生产。

**这次修正发生在第 2 步（快照落盘）之后，快照文件没有被回写更新。** 所以：校准结果卡在 loose gate 边缘、触发了这次自我修正的窗口，生产结束时的真实 f_k 就会跟 `frozen_score.json` 记录的对不上——不是数据被污染，是这道"生产结束后对比 f_k 一致性"的检查，比对的基准（快照）本身没跟上系统里另一处合法的自我修正。

## 为什么"清空重跑"只是碰运气，不是根治

触发条件是"校准出来的 f_k 是否卡在 loose gate 边缘"，这本身受 GPU 非确定性影响——同一个窗口重跑一次，校准结果可能离边缘远一点（不触发修正，快照和最终值一致，能过），也可能还是卡在边缘（同样的问题复现）。目前对 `repeat_0 baseline/window_0`、`repeat_1 baseline/window_1` 这两个失败窗口的处理方式是清空重跑，赌下一次校准别再卡在边缘——不是可靠修法。

## 根治思路（未实现）

`_promote_frozen_state()` 把快照写进 `frozen_score.json` 的时机应该挪到**生产入口的"冻结 burn-in + 只读验证"（含可能的一次自我修正）之后**，而不是之前——即先让第 3 步内部那次可能发生的自我修正走完，再拿那次修正后的最终 f_k 去写快照、算 `spec.sha256`。这样"快照"和"最终生产用的 f_k"永远是同一份，不会有对不上的情况。

需要重新梳理 `exp030_window_state_machine.py` 里"校准/冻结/生产"这三步之间的调用顺序和 `manager.run_all_windows()` 的 `warmup_only`/`resume` 参数组合，可能要新增一个"只做冻结验证、不做完整生产"的中间调用点，或者把 spec 构建挪到 `run_all_windows(resume=True, warmup_only=False,...)` 内部完成冻结验证之后。这是个不小的流程改动，没有在这次会话里做。

## 修复（2026-08-27）

没有深挖 `ibs_engine.py::run_all_windows()` 内部那套通用的 resume/验证/自我修正逻辑去移动"快照时机"本身——那是全仓库共享的大函数，牵一发动全身，风险太高，没在这次会话里碰。

改用更小、风险更低的方案：`scripts/exp030_window_state_machine.py` 里，`np.allclose(final_state["f_k"], spec.f_k,...)` 检查失败时，**不再直接 raise、把整段真实 250k 步生产数据当失败丢掉**。改成：用 checkpoint 里实际驱动了生产的 `final_state["f_k"]` 重新构建一份"事后追认"的 `JointStateScoreSpec`，覆盖写回 `frozen_score.json`/`joint_score_spec.json`（原快照的 sha256 和追认后的 sha256 都完整记进 `decision_log.jsonl` 的 `frozen_snapshot_reconciled_after_production_entry_self_correction` 事件，不是静默覆盖），后续 `_production_report()` 等分析统一用这份追认后的 spec。

这样快照始终忠实反映"真正驱动过采样的 f_k"，不再需要清空重跑赌运气；生产入口那次合法的自我修正机制本身没有被禁用或修改，只是快照现在会如实追认它，而不是把它当错误。

未验证：这次修复只做了 `py_compile` 语法检查，没有在真机上真正复现过一次这个分支被触发的场景来确认修复生效（触发条件本身是随机的，之前只在 `repeat_0 baseline/window_0` 这类窗口偶发遇到过，没有可靠的方法主动复现）。

## 补验收（2026-08-27，用户要求）

用户指出原修复有个真实漏洞：**把"生产结束后 f_k 对不上"一律当成合法的入口自我修正来追认，没有区分"合法的一次性修正"和"生产采样期间被意外改动/数据损坏"**——后者不该被追认，该硬报错。补了两处：

1. **加了合理性上界**：`_reconcile_frozen_snapshot_if_legitimate()`（把原来内联的追认逻辑重构成独立函数）现在检查差距是否在"一次合法阻尼+pairwise-capped 修正"的合理上界内（`FROZEN_SNAPSHOT_MAX_PLAUSIBLE_SINGLE_CORRECTION_KT = 6 kT`，留了 3 倍余量，不卡死在精确的 `IBS_MAX_APPLIED_PAIRWISE_STEP_KT=2kT`——目的是防"差几十 kJ/mol"这种量级的异常，不是卡死一个精确数字）。超出上界直接 `raise Exp030RunError`，不静默追认。
2. **确定性回归测试**：
   - `tests/test_exp030_frozen_snapshot_reconciliation.py`——纯 Python、不需要 GPU，4 个测试：无差异是 no-op；合理范围内的差异正确追认且 `frozen_score.json`/`joint_score_spec.json`/sha256/decision_log 全部保持一致；超出合理范围直接拒绝、不碰任何文件；边界值两侧都测过。
   - `tests/test_audit_protocol_regressions.py::test_production_entry_self_correction_stays_before_production_sampling`——锁定"自我修正代码在源码里必须出现在'进入生产采样'那个标记之前"这条结构性不变量（这也是"合法修正只可能发生在生产采样开始之前"这个假设的依据——修正逻辑在 `ibs_engine.py` 的 warmup/验证 while-loop 内部，循环退出后才真正进生产、f_k 冻结不再调用 `update_weights()`，用源码位置断言把这个前提钉住，以后如果谁重构挪动了这段逻辑，这个测试会先炸）。

跟这次改动相关的全部测试（含新增的）现在 270/270 通过。

**仍然没做、承认局限**：没有从 `ibs_engine.py` 内部拿到"f_k 是在生产采样开始前还是开始后被改的"直接运行期证据——现在的把关是"源码结构上不可能在生产采样期间改"（结构性论证）+"差距量级上界"（合理性论证）两层间接证据的组合，不是逐帧运行期直接验证。要做到后者需要往 `ibs_engine.py` 里加运行期埋点，这次没做（那是共享的大函数，风险收益比不划算，权衡后没做）。
