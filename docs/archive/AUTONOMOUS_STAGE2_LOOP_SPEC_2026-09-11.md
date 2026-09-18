# Stage-2 自治闭环规格（核心设计指标）

> ## 🗂 2026-09-18 归档 —— **指标仍然有效，过期的是本文的 §1 和 §4**
>
> **§0 那句核心指标没有被取代**，它已抄进
> [STAGE2_CONTROLLER_DESIGN_2026-09-12.md](STAGE2_CONTROLLER_DESIGN_2026-09-12.md)
> 页首，本文只是原件。
>
> | 本文的哪一段 | 状态 |
> |---|---|
> | §0 核心设计指标 | **仍然是唯一验收指标** |
> | §2 三个真终态 / 路由信号不得退出顶层循环 | **仍然有效**，现行实现见控制器设计文档 §2 |
> | §1 「当前 run 必须自动走完的路径」 | **已过期** —— 那个 `rep1` / `win3` 早已不存在 |
> | §4 「当前差距：控制器是只读的、零调用点、核心指标 = 0」 | **已过期** —— 执行器 09-11/12 就接通了 |
>
> ⚠️ 指标**达成与否**不在本文里定：控制器 09-12 首次独立走完六窗，但 09-18 的
> benchmark 批次仍有 7 次崩溃（见
> [STAGE2_BENCHMARK_CRASH_TRIAGE_2026-09-18.md](STAGE2_BENCHMARK_CRASH_TRIAGE_2026-09-18.md)）
> ⟹ **不得声称「Stage-2 自治闭环可用」**。

2026-09-11 老板定案。**这是唯一的验收指标**，其它一切都是它的下位项。

## 0. 核心设计指标

> **一次启动，无人干预；遇到采样/收敛问题，自己读证据、诊断原因、选择动作、执行、复验，直到完整结果。**

验收方式：**resume 当前 `rep1` 后，人不碰它**，它自己处理 win3，跑过 win4、win5，
并产出完整 Stage-2。

## 1. 当前 run 必须自动走完的路径

```text
win3 支撑不足
→ 判断累计 f_k 偏差
→ 生成候选
→ held-out 可测则验收
→ 不可测则自动开 PROBE_REANCHOR_EPOCH
→ 新 f_k + burn-in + 分块采样
→ 支撑改善则继续
→ 不改善则自动 tail-repartition / 插 λ
→ win3 通过
→ 自动跑 win4、win5
→ 全路径分析
→ 输出结果
```

## 2. 路由信号 ≠ 终态

以下**全部**只是路由信号，**不得退出顶层循环**：

```text
LOCAL_VALIDATION_CAP
INSUFFICIENT_DATA
CUMULATIVE_FK_MISALIGNMENT
SKIPPED_WINDOW
```

顶层**只允许三个真正终态**：

```text
DONE
GLOBAL_BUDGET_EXHAUSTED
INVALID_PHYSICAL_INPUT / NO_FEASIBLE_ACTION
```

## 3. 唯一在做的事

把 `decide → execute → reread` 接通。

- 不要再增加研究项。
- 不要再让用户选择动作。
- 不要再把影子判断当完成。

## 4. 当前差距（2026-09-11 实测）

`Stage2RepairController` **是只读的**（`test_controller_never_writes_anything` 还专门钉了这条），
生产代码里没有任何调用方。`decide()` 产出的动作**从未被执行过**。

⟹ 核心指标当前 = 0。缺的是执行器与顶层循环，不是更多诊断。
