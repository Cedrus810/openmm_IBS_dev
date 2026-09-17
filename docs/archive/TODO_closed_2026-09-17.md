# 已关闭条目存档 —— 2026-09-17

[文档导航](../README.md) · [P1 在推的](../TODO.md) · [P2 该做不挡路](../TODO_P2.md) · [P3 现在不做](../TODO_P3.md)

> **本文不是待办。** 2026-09-17 当天关闭的 6 条条目整段从 `TODO.md` / `TODO_P2.md`
> 移进来，**一字未改**（每条原文里的日期、判据、实测数字都保留）。
> 按 [docs/README.md](../README.md) 的规矩：已关闭的条目不留在待办文件里。
>
> ⚠️ 与 [TODO_closed_2026-09-16.md](TODO_closed_2026-09-16.md) 那次不同 ——
> 那一份是**一次对账**（13 条缺陷早已不在代码里、只是没人回来打勾）；
> 这一份是**当天真做掉的**，每条都有同日的改动与验证。
>
> 📌 「某条缺陷修没修」的权威仍然是**源码**，不是这里的 `- [x]`。

| 条目 | 原在 | 一句话 |
|---|---|---|
| `REWIND-01` [P1] | `TODO.md` §1 | `IMMUTABLE_REWINDOW` 曾是 13 个声明动作里唯一一个 `decide()` 发不出的；接在四条死线各自的兜底**之前**，一个父窗只切一层 |
| `TEST-01` [P2] | `TODO_P2.md` §1 | 全量 198 个测试文件都进了 `cpu_only` / `needs_gpu` 之一；判据用**收集数对账**，不是 grep 文件 |
| `LR-03` [P2] | `TODO_P2.md` §4 | 整片 EXP-010/011 的 CLI 表面已删，移进 `archive/` |
| `LR-04` [P2] | `TODO_P2.md` §4 | 加了 `--only-ligand-in-water` 调试入口 |
| `REL-08` [P2] | `TODO_P2.md` §5 | 基础环境本来就存在，欠的只是口径 |
| `REL-09` [P2] | `TODO_P2.md` §5 | 三项里两项早已落地，第三项（输出目录独占锁）是**已做出的判定**，不是缺口 |

> ### 留下的规矩
>
> 1. **死动作要接在各自兜底之前，不是在链尾加一条分支** —— 否则又变成「按书写顺序决定语义」（`REWIND-01`）。
> 2. **一个父窗只切一层**：不设深度上限就是新的无限循环；这条限制**不许放宽** —— 触发它的判定本身噪声 34×，会误触发在没问题的窗口上。
> 3. **测试选择口径的判据是收集数，不是 grep 文件**：同一个文件里可以既有带标记的测试、又有漏网的（`TEST-01` 就是这么发现的）。
> 4. **「真机证据」也会指向反面**：`REL-09` 原文引的那 4 次目录锁报错，咬的是**续跑**、来自后来被删掉的那把锁 ⟹ 它支持「删得对」，不是「该补回来」。

---

## **REWIND-01 [P1] ✅ 2026-09-17 已修** —— `IMMUTABLE_REWINDOW` 曾是 13 个声明动作里唯一一个 `decide()` 发不出的 ⟹ 一类盘面本来有对症动作却直接停机（2026-09-16 AST 实测）

  **修法**（`abfe_preoptimizer.py`）：
  ① 子窗划分 `vanishing_rescue_ranges()` 从 `abfe_pipeline.ABFEPipeline._build_vanishing_rescue_ranges`
  搬到 **模块级**（控制器 import 不动 abfe_pipeline：循环依赖 + 它 import openmm），
  pipeline 那个静态方法转成一行转发；成本表里那份 `2 if (b-a) > 2 else 1` 的复算随之删掉
  ⟹ 切法全仓一份。
  ② 新增 `Stage2RepairController.rewindow_feasible()` / `rewindow_children()`：
  可行 = **切得出两个各 ≥2 态的重叠子窗**（K ≥ 3）**且** 该父窗不在 `parents_done`
  （**一个父窗只切一层** —— 不设深度上限就是新的无限循环）。
  ③ 发出点接在 **四条死线各自的兜底之前**（1e 预热预算耗尽 / 5a-2 held-out REJECT /
  边际增益停滞 / 5b 偏斜类自检），不是在链尾加第 33 条分支 —— 否则又是「按书写顺序决定语义」。
  不可行时原来的 `NO_FEASIBLE_ACTION` 照旧，只是理由里多带一句有界重窗为什么不可行。

  **正向耦合**：同日分窗目标改成 `state_count`（min-max K）之后 23 个蛋白 rep 全变成
  `[5,5,5,5,5]` ⟹ **每个窗口 K=5 ⟹ 对每一个窗口都可行**（老布局 K=4 的窗只能切 3+2）。
  已钉成 `test_bounded_rewindow_is_feasible_for_every_window_of_a_state_count_layout`。

  ⚠️ **别期待它改善 ΔG。** 死线开了 = 循环能继续走，不等于答案更准；而且触发它的判定
  本身噪声 34×（[AUDIT_GATES_AND_CRITERIA_2026-09-17.md](AUDIT_GATES_AND_CRITERIA_2026-09-17.md)）⟹ 它会被误触发在其实
  没问题的窗口上。**「一层」那条限制因此不许放宽。**

  **验证**：3 条 strict-xfail 标记已删（全绿）；`tools/audit/static_controller_2026-09-17.py`
  死动作 1 → 0；全量 `2653 passed / 0 failed`（此前 2649，+1 新测试 +3 转正）。

  <details><summary>原始条目（存档）</summary>


  **实测**：对 `Stage2RepairController._decide_once` 做 AST 扫描（`plan()` 的首参 +
  `act=` 赋值，含条件表达式两支），13 个 `ACTIONS` 里 12 个发得出，
  **`IMMUTABLE_REWINDOW` 一处发出点都没有**。

  **它不是"还没实现"—— 是缺唯一入口**，周边全是齐的：
  - 成本表：`abfe_preoptimizer.py` 约 **4632** 有 `elif action == "IMMUTABLE_REWINDOW"`
    的逐子窗首块计价（审计 #44，还特地修过"硬编码 2 个子窗"的错）；
  - 执行器：`abfe_pipeline.py` 约 **12162** 的 `elif act == "IMMUTABLE_REWINDOW"`
    → `_immutable_rewindow_step()`（约 12867）完整实现；
  - 子窗调度：`sampling_units` / `rw:<id>:<i>` / `stage2_rewindow_ledger.json` 全在，
    并有 12 条测试覆盖（`tests/test_rewindow_sampling_units.py` 等）。

  **后果**：λ 表顶到溢出槽上限（末窗 K = 2·max−1）且失败落在**中间窗**时，
  拆末窗不对症、插 λ 不可行 ⟹ 本该退到**有界重窗**，现在直接 `NO_FEASIBLE_ACTION`
  停机。这正是 `BM-04` / `DATA-02` 那类死局的形状。

  **为什么一直没人发现**：本仓有"死出口"守卫（`test_no_dead_exits_left_in_the_declaration`）
  但**没有对应的"死动作"守卫**。2026-09-16 已补上
  （`tests/test_stage2_controller_vocabulary.py::test_no_dead_actions_left_in_the_declaration`）。

  **判据**：把发出点接回 `_decide_once` 的可行性链尾部（拆窗与插 λ 都不可行时的兜底），
  下列 3 条 `strict-xfail` 变红即修好，届时删标记：
  `test_no_dead_actions_left_in_the_declaration`、
  `test_stage2_repair_controller.py::test_fixed_lambda_table_middle_window_falls_back_to_bounded_rewindow`、
  `test_audit_2026_09_14_controller_budget.py::test_an_unknown_production_remainder_does_not_reject_a_bounded_rewindow`。

  </details>

---

## **TEST-01 ✅ 2026-09-17 关闭 —— 全量 198 个测试文件都进了 `cpu_only` / `needs_gpu` 之一。**
  改法：22 个无标记文件加 `pytestmark = pytest.mark.cpu_only`（8 个顺带补 `import pytest`），
  外加 `test_exp019_softlift_loro.py` 里那个**漏网的单个测试** —— 该文件已有
  `@pytest.mark.needs_gpu` 挂在另一个测试上，所以按文件 grep 看是"有标记的"，
  实际第一个测试两档都不属于。📌 **这就是为什么判据要用收集数对账，不能用 grep 文件。**

  **判据实测（`CUDA_VISIBLE_DEVICES=""`）**：
  `-m "not cpu_only and not needs_gpu"` → **no tests collected (2666 deselected)**；
  `cpu_only` + `needs_gpu` 并集 = 全量 **2666**。
  22 个新标文件在**无 GPU 可见**下 **235 passed**（不是"本机跑过了"——卡被藏掉了）。

  ⚠️ 判成 `cpu_only` 的依据是它们全都只读源码 / AST / monkeypatch：
  `test_lr06_…` 读 `.cpp` 文本、`test_nvcc_discovery_…` 全程 monkeypatch 环境变量，
  两个名字带 CUDA 的都不建 Context。

---

## **LR-03 ✅ 2026-09-17 关闭 —— 整片 EXP-010/011 的 CLI 表面已删，移进 `archive/`。**
  📌 **原条目的前提要修正**：不是「实现在发布清理时被移出的 `archive/` 里」，而是
  **`archive/` 从来没进过本仓库** —— 目录不存在、`git log --all -- archive/` 为空。
  13 个转发壳是 2026-08-31 `169514e`（主线迁移）引入的，1718 行真实现留在旧工地
  `Atenolol-rank11/archive/outer_lambda_exp010_exp011_legacy.py`。
  ⟹ 这些子命令**从迁移那天起就是 `ImportError`，一次都没跑起来过**；
  「要么补实现」在本仓库等于把旧库搬回来重新纳管，而「legacy 读 `lambda_shield` → `TypeError`」
  那半条**永远够不着**（`ImportError` 先炸）。
  ⟹ 而且不止 `sample-hard-window-scratch` 一个入口：**9 个子命令**
  （`exp011-coverage` / `exp011-fit-pmf` / `exp011-umbrella-sample` / `exp011-reweight-umbrella` /
  `exp010-label` / `exp010-fit` / `exp010-prepare-selection` / `freeze-slow-variable` /
  `sample-hard-window-scratch`）全部撞同一堵墙。
  **已删**：9 个子命令的 argparse 定义与处理块、13 个壳、`__all__` 里 12 个名字、
  `tests/test_outer_lambda_cli.py` 的 `--help` 清单对应行。原文逐字存进
  `archive/outer_lambda_exp010_exp011_removed_2026-09-17.py`（**不被任何东西 import**）。
  ⚠️ `screen-slow-variables` / `compare-slow-variable-screens` / `wp0-select` **没删** —— 不碰这些壳。
  验证：`outer_lambda_neural_basis.py` 9472 → 8109 行；`--help` 27 → 18 个子命令；
  删前逐个数过 13 个壳的外部引用 = **0**。

---

## **LR-04 ✅ 2026-09-17 关闭 —— 加了 `--only-ligand-in-water` 调试入口。**
  📌 **名字是维护者定的**：它跑的是「配体在水中」那条腿，产出 **ΔG_solvent（去耦自由能）**，
  **不是结合能** —— 帮助文案与收尾日志都按这个口径写死，防止有人把这个数当 ΔG_bind 引用。
  定位是**调试**不是生产：跳过 Boresch 估算、带限制力 rebalance 和整条复合物腿，
  只跑配体在水中那条腿，**不产出 ΔG_bind**、不覆盖主 `final_results.json`。
  三道 fail-closed：① 与 `--only-complex-charging` / `--only-boresch-attachment` 互斥；
  ② 与 `--outer-lambda-local-residual-ibs` 互斥（残差运行时是在复合物腿那段构造、
  再按溶剂腿拓扑重建的，跳过复合物腿就拿不到它 —— 抛错，不静默降级成 baseline 溶剂腿）；
  ③ 溶剂腿跑完**直接 return**，不让它走到第 8 节。
  📌 **③ 是刻意的**：比"补个 `None` 往下传"安全 —— 后者每加一个下游分支，
  就多一次把半份结果写成完整结果的机会（本仓库 `validate_final_leg_result` 那批
  fail-closed 就是为这个加的）。
  验证：`tests/test_only_ligand_in_water_lr04.py` 5 passed（控制流契约：开关/默认值/两条互斥/
  复合物腿整段在 `else` 里/早退存在）+ `test_no_undefined_globals` + CLI `--help`。
  ⚠️ **真跑一条腿要 GPU，没跑过** —— 钉住的只是"不跑也能错"的那部分。

---

## **REL-08 ✅ 2026-09-17 关闭 —— 基础环境本来就存在，欠的只是口径。**
  📌 **不用写第三个 yml**：`environment-ci.yml`（43 行手写）就是 CPU 基础环境
  —— python + openmm + pymbar-core + numpy/scipy/mdtraj/rdkit/pdbfixer，
  **不含** CUDA 12.9 工具链 / torch / openmm-torch / MACE / JAX（那些全在
  `environment.yml` 那份 333 行 `conda env export` 全钉死的生产环境里）。
  从 333 行的 export dump 里手术式删 CUDA 行删不干净，所以改的是**文档定位**：
  [GETTING_STARTED.md](../GETTING_STARTED.md) 新增「两个环境文件，装哪个」对照表。
  ⚠️ **`openmm=8.5.2` 的钉死是判定，不是疏漏** —— 8.6 尚不成熟、有已知严重 bug，
  本项目**暂**不使用；**是"暂"不是"永不"**，8.6 的官方 REMD 正是本项目想要的，
  **未来版本可能支持**，缺的是 8.6 本身稳定下来。它与 `free_energy_engine.OFFICIAL_REMD_MIN_OPENMM_VERSION = "8.6.0"`
  **不矛盾**：后者是「官方 REMD 采样器要 8.6 才有」的资格判据，前者是「今天不装 8.6」。
  📌 **本条曾被我误判成 bug 并动手改成下限 `>=8.5.2`（已全部还原）** ——
  理由现在写在 `environment-ci.yml` 与 `GETTING_STARTED.md` 里，别再重开一遍。

---

## **REL-09 ✅ 2026-09-17 关闭 —— 三项里两项早已落地，第三项是**已做出的判定**，不是缺口。**
  逐项回源码核实（本条原文的三项「欠覆盖」全部不成立）：

  | 子项 | 实际 |
  |---|---|
  | SIGTERM/SIGINT | ✅ `abfe_pipeline.install_termination_handlers`（`guard_run_directory` 调用）。`TerminationRequested` 基类是 `BaseException`，正是为了不被全仓 ~220 处 `except Exception` 吃掉。**2026-09-17 真机触发过**：`cyclod_ligand1_outer/rep1` 02:29 的有序退出 |
  | 磁盘预检 | ✅ `abfe_pipeline.ensure_free_disk_for_stage`：stage 开跑前按 `estimate_stage_trajectory_bytes` 估产出，要求 `free ≥ 2×`，不够 **fail-closed 拒绝开跑**（ATT-23 / issue #142）。原文「只有 doctor 里有 disk_usage」是它落地**之前**的状态 |
  | 输出目录独占锁 | ⚖️ **已判定不要**。`guard_run_directory` 的 docstring 写明理由：进程被 SIGKILL 时锁文件残留，而 stale 判定在共享盘上判不出来（PID 跨节点无意义、hostname 不同一律当活着）⟹ **锁挡住的是 `--resume` 本身** |

  ⚠️ **本条原来引的那条「真机证据」是反的**：`输出目录已被另一次运行独占（目录锁）` 那 4 次全在
  `runs/cyclod_ligand1/rep1/launch.log` 的 **2026-09-10** 段，traceback 指向
  `runabfe.py:6267 guard_run_directory` → `abfe_pipeline.py:1855`，即**后来被删掉的那把锁**，
  而且它咬的就是**续跑**。⟹ 那 4 次支持的是「删得对」，不是「该补回来」。**别拿它重开这条。**

  📌 **`RunDirectoryLock` 已于同日删除**（维护者确认「没用」）：生产侧零调用点，
  类 + 5 条锁测试进 `archive/run_directory_lock_removed_2026-09-17.py`。
  同文件的 SIGTERM 与磁盘预检两节**保留**（那两项仍在生产里跑，是本条结案依据）。
  ⚠️ 归档头部记了一条**复核确认的机制缺陷**，将来若有人想把锁接回来必须先看：
  `timeout_s=0.0` ⟹ 父类 `__enter__` 的 `deadline` 就是当下 ⟹ 第一次 `FileExistsError`
  后必抛 `TimeoutError`，**`os.open` 永不重试**。于是 `_break_stale_lock_if_needed()`
  即便成功清掉残留锁，**这一跑照样死**，清理成果留给下一次启动。
  ⟹ 「不等待」（设计如此）与「不重试」（缺陷）被写成了同一件事。
  📌 父类 `_PipelineStateLock`（`timeout_s=10.0`）**不受影响**，`pipeline_state.json` 照常用。

---
