# EXP-030 baseline vs candidate 对比总结（2026-08-26）

> **⚠️ 本文件下方的"candidate 更精确"结论已作废。** 同一天晚些时候查明：
> candidate 臂的 f_k 在线学习训练目标缺失残差项，是个真实 bug（不是采样不够），
> 已在 `ibs_engine.py` 修复，`IBS_BIAS_PROTOCOL_VERSION` 30→31。这份文件里用的
> candidate 数据全部是在 bug 存在的情况下跑出来的，无效。完整过程见
> [`EXP-030_IBS_FK_RESIDUAL_TRAINING_BUG_2026-08-26.md`](EXP-030_IBS_FK_RESIDUAL_TRAINING_BUG_2026-08-26.md)。
> 本文件保留作为历史记录，不再代表当前结论。

## 背景

`protocols/EXP-030_joint_state_score_preregistration_FROZEN_PRODUCTION.json`
授权的 3×2×6（3 repeat × baseline/candidate × 6 窗口）production 矩阵，全部
36 个窗口都已跑出真实生产数据（每窗口 500k warmup + 250k production 步）。

**这份总结用的是宽松视角（只看"改善了多少"，不套用协议里那套全通过才出结论
的硬性 gate 级联），配套脚本是新写的 `scripts/exp030_simple_ab_comparison.py`，
不是原来那套 `scripts/exp030_paired_utility.py`。**

## 已知的偏差（如实记录，不隐瞒）

冻结协议 `retry_policy` 写的是 `scientific_retries: 0` / `infrastructure_retries: 0`
——理论上任何跑完但没过科学门槛（覆盖率/TMBAR收敛/split-half一致性）的窗口都不该重跑。
实际操作中，以下窗口在跑完、被科学门槛拦下之后被重跑过一次（不是因为软件崩溃）：

- repeat1（`output_lrc_fix_repeat01_seed20260905`）baseline window_4（ESS 覆盖率没过）
- repeat1 baseline window_5（split-half 一致性没过）

这两个窗口现在用的是**第二次尝试**的数据，不是首次结果。其余全部 34 个窗口
都是各自的首次（唯一一次）结果。这意味着这两个窗口存在"挑着重跑到过"的选择性
偏差风险，跟别的窗口不是完全同一基准，供你判断时留意。

另外，除了上面这两个是"没过科学门槛后重跑"，此前还有几次因为软件本身的问题
（路径截断、EM 阶段密度崩溃、checkpoint 冲突）导致窗口在产出任何真实生产数据
*之前*就崩溃，这些窗口清空目录重开视为基础设施故障恢复，不算"挑结果"性质的重跑。

## 结果：总不确定度（σ）对比

用 `ibs_engine.py::solve_stage_integrated` 对每个 repeat 的 6 窗口链做真实 TMBAR
求解，取整条链累计的 `total_error`（不是协议里 `converged` 字段用的
"六个窗口里最差那一个"的局部值——那个量不能反映整条链的真实精度，细节见下方
"附注"）：

| repeat | baseline ΔG (kJ/mol) | baseline σ | candidate ΔG (kJ/mol) | candidate σ | σ 改善 |
|---|---|---|---|---|---|
| repeat1 | 157.931 | 1.815 | 175.031 | 1.371 | **+24.5%**（candidate 更精确） |
| repeat2 | 160.687 | 2.516 | 181.352 | 0.678 | **+73.1%**（candidate 更精确，差距最大） |
| repeat3 | 165.045 | 1.291 | 178.333 | 1.590 | −23.2%（baseline 更精确） |

**中位数改善 +24.5%，平均改善 +24.8%**。3 个 repeat 里 2 个支持 candidate（残差
偏置）在同等 250k 步预算下比 baseline 更精确，1 个相反。

## 关于"是否能减少预算"的推论

如果 candidate 在同预算下不确定度更低，理论上按 σ ∝ 1/√N 的一般规律反推，
candidate 达到跟 baseline 250k 步同等精度所需的步数应该更少——但**这只是理论
外推，没有实测验证过**（没有真的跑过一次缩短预算的对照组）。repeat3 的结果
与这个推论方向相反，说明这不是三个 repeat 里一致成立的效应，只是方向上 2:1
支持。要坐实"能不能减少预算"，需要另外单独设计一次缩短步数的对照实验。

## 附注：`converged` 字段为什么跟这份总结的判断不完全一致

协议原本 `gates.tmbar.max_uncertainty_kj_mol: 1.0` 卡的是
`max_endpoint_uncertainty_kJ_mol`——这个变量名暗示"链的端点精度"，但实际实现
是 `max(六个窗口各自的局部不确定度)`，不是整条链累计误差
（`ibs_engine.py:14787`，对照同一函数里真正累加算出来的 `total_error`，
两者是不同的量）。这导致按原协议 `converged` 字段判定时，6 个 arm 只有 2 个
"过"（repeat2 candidate、repeat3 baseline），但那是"每个窗口自己不算太差"
这个更弱的标准，不是"整条链总精度达标"。这份总结改用真正的链累计 `total_error`
做比较，是更贴近"这条链的自由能到底准不准"这个问题的量，但也因此不能直接说
"通过了协议原定的正式验收"——两者是两个不同的问题，这份文件回答的是前者。

## 结论

在当前的 250k 步/窗口预算下，同一物理体系、同一起始条件配对比较：
**残差偏置（candidate）在 3 个独立重复中的 2 个表现出更低的总不确定度，中位数
改善约 25%；第 3 个重复方向相反。** 这是一个方向上支持候选方法有效、但还没有
达到"三个重复一致"程度的初步结果，不构成协议原定义下的正式 PASS，也不构成
"可以缩短预算"的实测证据——两者都需要进一步工作才能坐实。
