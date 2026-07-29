# 2026-07-28 完成项与复核归档

本文件由 `docs/TODO.md` 移出已完成、已关闭或已证伪的记录。主 TODO 只保留仍需行动的事项。

## ATT-27 / E-03：已证伪路线清理

- [x] **ATT-27 / E-03：已被证伪路线的可执行死代码已清理（2026-07-27）。** GitHub [#35](https://github.com/Cedrus810/openmm_IBS_dev/issues/35)。共移除 **1143 行**，全部逐字归档在 `docs/archive/`：
  1. `_run_stage_with_overlap_autorepair` 无条件 `return` 之后的 ensemble 变异循环 **872 行**（原估约 900；含 833 行 `while True` 与末尾一条 `raise`），该函数 992 → 110 行 → [archive/removed_overlap_autorepair_mutation_loop.md](removed_overlap_autorepair_mutation_loop.md)
  2. `_refine_lambda_path_with_medium_probe` **83 行** —— `enable_lambda_refine` 的实现半边，全仓零调用 → [archive/removed_refine_lambda_path_with_medium_probe.md](removed_refine_lambda_path_with_medium_probe.md)
  3. `_retired_overlapping_vdw_schedule_design` **188 行** —— docstring 自认 "Retired v9 scratch designer; retained only for failure-record archaeology" → [archive/removed_retired_overlapping_vdw_schedule_design.md](removed_retired_overlapping_vdw_schedule_design.md)

  `enable_lambda_refine` 的**守卫保留**（防静默重新启用），但从函数体第 ~795 行**提到 `run_full_pipeline` 入口**：原来配置手滑设 true，要跑完预平衡 + Boresch 锚定 + Stage 1 才炸。`abfe_pipeline.py` **8501 → 7370 行**。

  回归 `test_att27_dead_code_removed.py`：三个函数名不得复现、短路 `return` 必须仍是函数体最后一条语句、函数体顶层不得再出现 `While`、守卫必须距 `run_full_pipeline` 入口 <60 行、三份归档必须存在且为非可执行 `.md`。

  **刻意未删**：同批系统扫描发现 42 个未被任何 `.py` 引用的函数共 **1085 行**，其中 `OnlineConvergenceMonitor`(230 行) 与 `ChunkedMBARAnalyzer`(98 行) 两个类整体无人调用。但它们属于「写完未接入流水线」——接错了只是不工作，**不会产出错误的自由能**；而本条清理的是「已被证伪的错误路线」（adjacent fixed-H overlap 当 IBS 收敛仲裁，曾烧掉约一周 GPU 无产出），复活代价远大于保留收益，两者风险性质不同。DEXP/MACE 相关（`run_torsion_scan` 等）按缓期决定不动；Boresch 估计器周边刚经 ATT-11/P0-10 改动，不叠加。剩余 42 项清单可用仓内扫描复现。

## V-06：endpoint-sigma 诊断归档

- [x] **V-06：endpoint-sigma 诊断证据已归档（2026-07-27）。** GitHub [#74](https://github.com/Cedrus810/openmm_IBS_dev/issues/74)。三份原始 JSON 已从 `/tmp/sigma_diag/` 迁入 [status/evidence_2026-07-27/](../status/evidence_2026-07-27/)（含 README 解读）：
  - `endpoint_sigma_diagnosis.json`：A1 复现 `145.90847168207642 / 1.384443322336141` 逐位一致，**内含原运行从未落盘的逐段诊断**（P1-15 所致）；A2 重复测量 `primary_z ≈ 0.89 < 2`；history_scan 显示 base 21 处不连续但 decorr 序列 0 跳变。
  - `charging_linear_response.json`：梯形 TI 与 MBAR 吻合到 1 kJ（复合物 50.30/49.51、溶剂 63.70/62.72）→ **估计器没问题**；⟨ΔU⟩@λ=1 水 180.92 vs 口袋 144.04 kJ/mol。
  - `pose_drift.json`：无约束预平衡末帧 RMSD 0.60 Å，带 Boresch 的 rebalance 3.42 Å → **P0-10 的判决性证据**。

  **注意**：这些证据描述的是已作废的那一轮采样，**关于估计器的结论仍有效**（只取决于分析路径），**关于物理量的数值不可引用**。README 里另记录了一处方法论修正：脚本中的两点线性响应判据不适用于本例（两端方差差 9 倍，违反等方差前提，虚高 20–26 kJ/mol），正确口径是梯形 TI。

## ATT-11：Boresch 锚点回归修复

- [x] **ATT-11 引入的生产回归已修（逐侧决定键来源）。** 18:21 重跑在 Boresch 锚点估算处崩溃：
  `🔗 化学连通候选组合数: 0` → `RuntimeError: 没有符合条件的6原子组合`。

  **不是 PBC/盒子问题**（用户最初的假设）：日志走到了「候选组合数」，说明接触对**已经找到**
  （接触对为空会先抛 `未找到锚点-配体接触对`），配体与受体在空间上仍相邻；且
  `image_molecules` + `center_coordinates` 只做整分子平移，不改变任何分子内相对坐标。

  **根因是我的判断失误。** `estimate_from_trajectory` 用的拓扑是 `topology.cif`
  （`_resolve_mdtraj_topology_input` 优先 mmCIF），而 OpenMM 写出的 mmCIF
  `_chem_comp_bond = 0` —— 不含任何残基内成键。mdtraj 只能靠标准残基模板推键：

  | | 原子数 | 存在 2 深链的 |
  |---|---|---|
  | 受体锚点（`protein and name CA CB C N O`） | 1404 | **1404 = 100%** |
  | 配体重原子（`resname MOL and not element H`） | 19 | **2 = 11%** |

  我加的覆盖度判据是 `any(adjacency.get(i) for i in idxs)`——「该侧只要有任意一个原子有键
  就放行」，被那 2 个原子放行，于是走了拓扑路径；那 2 个原子恰好不在接触对里，配体侧最内层
  枚举从不执行 → 0 组合。**我写那段时在注释里就标注过 `any()` 太弱**，当时的理由是
  「真实标签错位必然同时打乱二面角」——那对 P0-10 成立，对「拓扑本身缺键」不成立。

  **修法不是回退旧行为**，因为存在一个真实的不对称：
  - **配体侧** haystack 是全部重原子，`0.22 nm` 的最近邻**确实就是化学键**
    （小分子键长 0.13–0.16 nm，次近邻 ≥ 0.24 nm）→ 几何回退在这一侧可靠；
  - **受体侧** haystack 被按原子名预筛成 CA/CB/C/N/O，**非键**的残基间 C–N（≈0.133 nm）、
    CA…N（≈0.146 nm）也落在阈值内 → 必须用真实键。

  即 ATT-11 的收益全在受体侧，配体侧本来就不太需要它。因此改为**逐侧决策**：

  - 新增 `_count_two_deep_chain_starts()`：判据从「有没有键」换成「有多少原子能起出
    `a→b→c` 且 b、c 都还在该子集内」——这才是 `_generate_anchor_combos` 真正消费的性质。
  - 新增 `_resolve_side_adjacency()`：该侧覆盖度 ≥ `bond_topology_min_coverage`(0.5)
    用拓扑真实键，否则退几何并打印计数。0.5 把 100% 与 11% 分得很开，不是贴着数据挑的边界。
  - `bond_source` 拆成 `bond_source_receptor` / `bond_source_ligand` + `bond_coverage`
    落盘，原 `bond_source` 保留为两侧汇总以兼容既有读取方。
  - `allow_geometric_bond_fallback=False` 时报错点明**是哪一侧**、覆盖度多少，
    并提示 mmCIF 不含 `_chem_comp_bond` 这个常见成因。

  **实测修复效果**（真实 `pre_equilibration.dcd` + `topology.cif`）：

  ```
  ✓ [Boresch 锚点] 受体锚点侧使用拓扑真实键（2 深链起点 1404/1404 = 100%）
  ⚠️ [Boresch 锚点] 配体侧拓扑成键覆盖度不足（2 深链起点 2/19 = 11% < 50%），该侧退回几何阈值 0.22 nm
  🔗 化学连通候选组合数: 416
  🏆 最优锚点: 受体 [1328, 1326, 1324] | 配体 [4595, 4596, 4597]
  ```

  受体锚点由 `[1328,1326,1338]` 变为 `[1328,1326,1324]` 是**预期的**：旧的 `1338`
  正是几何阈值把非键近接判成键才连出来的，这就是 ATT-11 要修的东西。

  回归并入 `test_boresch_anchor_and_pbc_fixes.py`（复现事故的稀疏拓扑、链必须整条在子集内、
  逐侧独立性、fail-closed 报错点明缺键侧、`_resolve_side_adjacency` 必须被调用两次）。
  **382 passed。**

## REMD GPU 默认路径与 parallel-stages 清理

- [x] **REMD 的 `max_resident_contexts` 默认值让 GPU 路径不可达；完整链路又没接通该参数。**
  完整 dual_lambda 的 decharging"卡住"23 分钟，12 个 `decharging_rep*.dcd` 全 0 字节，
  `pipeline.log` 最后一行是阶段表头、**无 traceback**。终端里（不在日志里）有一行：

  ```
  ⚠️ REMD replica 数超过 GPU 常驻 Context 上限 (12>1)；为避免单 GPU OOM，在创建任何 GPU Context 前回退 CPU。
  ```

  **两个独立缺陷叠加：**

  1. **默认值。** `max_resident_contexts` 对 CUDA/OPENCL 默认 **1**，而交换实现天生要求
     每个 replica 同时常驻（`context_residency_mode = "all_resident"`）→ `n_replicas > 1`
     恒成立 → **任何** REMD 都静默回退 CPU。这是 ATT-03 的产物，其条目自己写着
     「这是安全优先的实现，尚未实现 GPU 分批加速」——意图对，默认值过度保守到把功能关死。
  2. **参数只接通了一条路径。** `--charging-max-resident-contexts` 只被
     `--only-complex-charging` 消费；完整链路两条腿都没透传，所以永远拿默认值。
     这也解释了 15:04 那轮 `--only-complex-charging` 为什么能留在 CUDA
     （manifest：`platform_name: CUDA`、`platform_fallback_reason: None`）。

  **代价极不对称**：CPU 回退慢约两个数量级（GPU ~24 分钟跑完 500 轮交换；CPU 上 23 分钟
  连第一个 DCD 帧都写不出来），而且决定**只 print 到终端**，归档日志查不到——完全像卡死。
  而真正的 GPU OOM 本来就有优雅处理：`_build_replicas` 的 `except` 分支会
  `_clear_replica_contexts()` 后判定 `_is_gpu_context_failure` 再回退 CPU 重建。
  **OOM 是响亮且立即的，慢 100 倍是静默的。**

  修复：
  - [x] CUDA/OPENCL 默认改为 `n_replicas`（不预防性回退），把判断交给真实的构建期 OOM 处理。
        显存小的机器仍可显式传小值强制回退。实测 12 个 73536 原子 PME Context 在 11 GB
        RTX 2080 Ti 上装得下。
  - [x] 回退告警改用 `logger.warning`（不再只 print），并写明「慢约两个数量级」的代价与
        提高上限的办法；同时设 `platform_fallback_reason`，避免 `exchange_diagnostics`
        谎报 `CUDA + None`。`__init__` 末尾那句无条件 `platform_fallback_reason = None`
        会抹掉刚写入的原因，已改为仅在未设置时初始化。
  - [x] 参数全链路透传：`runabfe` 两条腿的 `run_full_pipeline`、`abfe_pipeline` 三个
        `_run_dual_lambda_stage` 调用点、以及 `_run_stage_worker_process`
        （spawn worker，`--parallel-stages` 目前被无条件禁用、路径不可达，但一并接上，
        重新启用时不该再踩回同一个坑）+ `_common` 字典。
  - [x] 回归 `test_remd_gpu_context_budget.py`：默认分支不得再出现常量 1、构建期 OOM 回退
        必须仍在、显式小上限仍须回退、必须 `logger.warning`、必须设
        `platform_fallback_reason` 且不被无条件重置、**所有** `_run_dual_lambda_stage`
        与 `run_full_pipeline` 调用点都必须透传该参数。
        写这条测试时它立刻抓到我漏掉的第四个调用点（spawn worker）。**389 passed。**

- [x] **`--parallel-stages`（stage 级多 GPU）整体归档移除（2026-07-27，用户决定）。**
  共移除 **264 行**，逐字归档于
  [archive/removed_parallel_stages.md](removed_parallel_stages.md)：
  并行分支 149 行 + `_run_stage_worker_process` 87 行 + 随之孤立的 `_save_state_to_dir` 28 行。

  它确实是多 GPU 机制（`stage1_platform = f"CUDA:{env_stage1}"` / `stage2_platform = ...`，
  把去电荷与去VDW分派到两张卡），但**早已被无条件禁用**——`run_full_pipeline` 读到该参数后
  直接 `_parallel_stages = False`（理由：跨进程结构化 warmup 失败反馈尚未序列化），
  于是 `if _parallel_stages and ...:` 恒假、整块不可达。**删除的是死代码，没有丢失在工作的能力。**

  用户决定：「暂时用不上多 gpu」「多 gpu 会导致各种调度问题，所以直接归档」。
  若将来真要多卡，更合理的方向是**在 REMD 内部按 replica 分卡**（单卡上 12 个 Context
  本来就是时分复用），那与 stage 级并行是两码事，也不需要跨进程异常序列化。

  现在传 `parallel_stages=True` 会在 `run_full_pipeline` **入口** fail closed 并指向归档
  （与 `enable_lambda_refine` 同一模式）；`runabfe` 的 `--parallel-stages` CLI 参数**保留**，
  这样带着旧命令行/旧配置的人会拿到明确解释，而不是 argparse 的 "unrecognized arguments"。

  **两点顺带记录**：
  - `ibs_engine.py:13544` 仍有一处 `mp.get_context("spawn")`（`TraditionalMBARAnalyzer`
    的 u_kn 分块并行），所以 **ATT-04（消除 import 期 CUDA 初始化）的修复仍然是有用的**，
    不因本次删除而失效。
  - 「REMD 默认永远回退 CPU」那个修复**只靠改默认值就成立**：`n_replicas` 本来就从
    `lambdas_coul/vdw` 算得，`max_resident_contexts` 默认取 `self.n_replicas` 后条件恒假。
    我为它做的 4+ 处 `charging_max_resident_contexts` 透传对修复卡死**毫无贡献**，
    现在唯一用途是给小显存机器一个显式强制 CPU 回退的逃生口；其中一处还透传进了本次
    被删的 worker 路径——为不可达代码做了 plumbing。

  `abfe_pipeline.py` 今日累计 **8501 → 7127 行**（净减 1374 行）。**389 passed。**

## R-01：Boresch 非谐性误差结论

- [x] **R-01：Boresch 非谐性释放修正误差已量化，结论是可忽略，本项关闭（2026-07-27）。** GitHub [#37](https://github.com/Cedrus810/openmm_IBS_dev/issues/37)。

  不需要数值积分或新实验——有闭式解。施加的角度/二面角势是 `k(1−cosΔ)`，其精确配分函数是 von Mises 积分 `2π·e^{−x}·I₀(x)`（`x = βk`）；解析释放公式假定的纯谐波给 `√(2π/x)`。两者之比有渐近展开

  ```
  √(2πx)·e^{−x}·I₀(x) = 1 + 1/(8x) + 9/(128x²) + 225/(3072x³) + …
  ```

  于是每个角自由度的 ΔG 误差 = `−kT·ln(该比值)`。代入本体系（kT = 2.4943 kJ/mol）：

  | 情形 | k (kJ/mol/rad²) | x = βk | 5 个角自由度合计 |
  |---|---|---|---|
  | 本次实际（最硬 200） | 200.0 | 80.2 | −0.0196 kJ/mol |
  | 本次实际（最软 kphiC） | 128.5 | 51.5 | −0.0306 kJ/mol |
  | 裁剪下限（最坏情形） | 10.0 | 4.0 | −0.4494 kJ/mol |

  **误差界：在 `force_constant_clip_ranges` 允许的整个区间 k ∈ [10, 200] 内，|ΔΔG| < 0.107 kcal/mol；本次实际参数下仅 0.005 kcal/mol。**

  力常数估计本身用 `kBT/var`（同为谐波关系）的偏差是同阶的 `O(1/(8x))`，经 `−(RT/2)·Σln K` 传播后：实际参数 0.002 kcal/mol、裁剪下限 0.046 kcal/mol。两项都远小于任何在意的量。

  **对照量级**：本轮与 `result.txt` 的实测偏差是 restraint 2.31 kcal/mol、charging 4.84 kcal/mol，分别是非谐性误差的 **429 倍 / 899 倍**；而真正的根因 P0-10（Boresch 平衡值陈旧）造成的是 ~2 kcal/mol 量级。

  **同时更正原条目的一处未经证实的说法**：原文提到「偏差 1–3 kJ/mol」并担心它被写成硬阈值。按上表，在生产力常数下真实偏差是 **0.02–0.03 kJ/mol**，比那个说法小约两个数量级。因此不需要为非谐性设任何 hard gate；现有 harmonicity 诊断只报警不阻断是恰当的。（原条目另提的「涨落 >15°」是**分布形状**问题，与非谐性误差是两件事，仍由 `assess_boresch_harmonicity` 的 skew/kurtosis 诊断覆盖。）

## 原 TODO 第 432 行后 bug 清单复核

本节是对旧文末 #1–#27 的逐项复核记录；可行动项已并入上面的 P1/P2/ATT 主清单，
不再保留两套互相矛盾的评级。核实口径包括：读取完整控制流、追踪仓内调用方和产物
写入方、检查异常后的 fail-open/fail-closed 行为，以及区分“生产路径”“零调用遗留
函数”“单纯代码风格”。**结论：新增 P0 为 0；新增 P1 为 1；其余真实问题均为 P2
或已并入既有工程任务。**

### 仍需行动（已去重）

| 原编号 | 复核后评级 | 结论与归属 |
|---|---|---|
| #24 | **P1** | 真实的物理协议 fail-open；环境变量默认关闭，故不是 P0。已登记为 **P1-20**。 |
| #18、#19 | **P2（中）** | checkpoint/DCD 校验确实不能证明内容完整；真实加载失败虽会被捕获，但追加轨迹等决定发生得过早。并入 **ATT-23**。 |
| #14、#15 | **P2（中）** | `which` 与个人 `/home/...` 回退不便携；显式路径和 `GMXDATA` 可绕过，原“Windows 必失败/严重”评级过高。登记为 **P2-16**。 |
| #2、#3、#11 | **P2（中，休眠代码）** | 二次 Å→nm、按长短猜矩阵方向、不可达角度分支均真实存在；但 `scan_boresch_1d_pes()` 与 `aggregate_all_energies()` 全仓只有定义、没有调用，不能描述成当前生产结果风险。并入 **ATT-26**。 |
| #16 | **P2（中）** | 5 个 `NumpyEncoder` 的确不一致；风险是诊断/缓存 JSON 类型漂移或序列化失败，没有证据表明已改写自由能。并入 **ATT-28**。 |
| #1、#17 | **P2（低/防御性）** | `(3,3)` 猜测式转置值得删除，但合法 Boresch 输入至少要容纳 3+3 锚点；`optimize_alpha()` 的 `assert` 应改成 `ValueError`，但该方法当前零调用。并入 **ATT-24**。 |
| #10、#12、#13 | **P2（低）** | 3 处裸 `except:` 会吞 `KeyboardInterrupt/SystemExit`，是异常处理缺陷，不是已证实的数值 bug。并入 **ATT-22**；其中 checkpoint 的实质校验问题另由 ATT-23 覆盖。 |
| #21、#22、#25、#26 | **P2（低/清理）** | 分别归入 ATT-26（重复导入/局部 import）、ATT-28（日志入口）、ATT-23（Context/Integrator 回收）。未发现当前数值错误。 |
| #27 | **本次已完成** | 旧行号与重复评级已由本次按函数名重写取代。后续定位优先写符号名，行号只作辅助。 |

### 已证伪或不构成当前 bug

| 原编号 | 复核结论 | 代码证据 |
|---|---|---|
| #4 | **证伪** | `_wrap_ligand_to_box()` 只有在 `pos.shape[1] != 3` 时才检查 `pos.shape[0] == 3`；合法 `(3,3)` 不会转置。旧清单把嵌套条件读漏了。 |
| #5 | **证伪** | PyMBAR 4 明确保留 `pymbar.other_estimators.bar()`，且返回含 `Delta_f/dDelta_f` 的字典，当前代码正按该接口使用。见 [PyMBAR 4.0.1 官方文档](https://pymbar.readthedocs.io/en/4.0.1/other_estimators.html)；仓库 provenance 也记录 PyMBAR 4.0.3。 |
| #6 | **不构成 bug** | `top = results[0]` 只是局部变量改名不佳，赋值后仅用于紧随其后的两行打印，没有再把它当 topology 使用。 |
| #7 | **证伪** | `fc.get("kr", 0)` 缺键时得到 0，立即触发 `[50,5000]` 的 `ValueError`；不会“检查通过后在 `fc["kr"]` 才 KeyError”。 |
| #8 | **证伪** | `run_post_analysis()` 的解析 Boresch release 只需要 `equilibrium_values/force_constants`，不需要 3+3 锚点；参数存在但解析计算失败时当前代码会 fail closed。 |
| #9 | **不构成当前 bug** | `f_history` 恢复时确有重复引用，但全仓后续只 append、索引读取和序列化，没有对旧元素就地修改；“未来可能修改”不能当成当前缺陷。可在重构时改为显式 copy。 |
| #20 | **证伪** | `_boresch_core_signature()` 的比较用途是检测协议不一致；缺失锚点会形成空列表并与完整签名不等，从而拒绝拼腿。冻结 Stage 2 参数入口此前还会经过 strict 清洗。没有“残缺与完整参数产生相同签名”的证据。 |
| #23 | **证伪原描述** | `_value_in_inverse_nanometer()` 依次尝试两种 OpenMM Quantity 兼容转换；都失败后 `float(value)` 仍会抛错并向上传播，不会把对象 id 静默当成数值。宽泛捕获可收窄，但归为代码质量而非单位静默错算。 |

### 明确不单列为 bug 的项目

- `top` 变量遮蔽、`f_history` 重复引用、模块级日志 `print` 覆盖、函数内
  `import gc`、Context/Integrator 删除风格不一致，均没有当前错误结果的证据；分别
  作为可读性、未来维护或资源回收子项并入现有 ATT，不再制造独立高优先编号。
- `aggregate_all_energies()` 的旧说明称其被 `run_post_analysis()` 调用，实际全仓搜索只有
  定义，没有调用；即使算法本身应修，也不能再称“最容易触发”。
- 本轮只重整和评级 TODO，没有修改实现。文档核验为静态控制流/调用关系核验；当前桌面
  Python 环境未安装 PyMBAR，因此 #5 的动态导入没有在该解释器复现，结论依据项目记录
  的 4.0.3 环境与 PyMBAR 4 官方 API 文档。

### 执行顺序

1. **P1-20**：先删 Stage 2 指纹旁路并补 fail-closed 回归。
2. **ATT-23**：再修 checkpoint/DCD 真实完整性验证和追加轨迹时序。
3. **P2-16**：统一 GROMACS 路径发现，补 Windows/POSIX 测试。
4. **ATT-28**：统一 JSON encoder；其余休眠代码与裸 `except:` 随 ATT-22/24/26 清理。
