# 变更记录（缩略）

[项目入口](../README.md) · [文档导航](README.md) · [当前科学状态](STATUS.md) · [历史材料索引](HISTORY_LOG.md)

> **本文是全程缩略时间线：一行一条，只记"改了什么、哪天、破不破缓存"。**
>
> 详细版本说明**不在这里**，在三个地方，本文只指路：
>
> | 要查 | 去哪 |
> |---|---|
> | 某个协议版本号为什么跳（逐版全文） | 源码常量上方的注释块（`ibs_engine.py` / `abfe_preoptimizer.py`） |
> | 某份开发期材料写过什么 | [HISTORY_LOG.md](HISTORY_LOG.md)（148 份逐份登记） |
> | 某个数字能不能引用 | [STATUS.md](STATUS.md)（唯一的科学结论声明处） |
>
> **⚠️ 本文不是科学结论。** 条目写的是"代码/协议变了"，不是"结果对了"。

## 工作区与体系

本项目在三个体系、四个工作区上推进过。看日期先看体系：

| 体系 / 工作区 | 时间 | 现状 |
|---|---|---|
| **Atenolol-rank11**（β 阻滞剂，第一个体系） | 2026-06 ~ 08-31 | 结果全部 `INVALIDATED`；8.1G 原始材料保留，不再是真源 |
| **4W53**（T4 lysozyme L99A + toluene，**当前主线**） | 2026-08-27 ~ | 当前唯一有效体系 |
| **ABFE_IBS**（本仓库，工程区分支） | 2026-08-31 ~ | 只留生产代码 + 回归测试 + 使用文档 |
| **ABFE_IBS_CUDA**（EXP-031 沙箱） | 2026-09 | GPU 优化实验，正文与证据不在主线 |

---

## 2026-09

| 日期 | 变更 | 破缓存 |
|---|---|---|
| 09-12 | **EXP-033 P1 闭式重训接入主线**：配体指纹对不上冻结 manifest 时不再 fail，改为预平衡跑完后自动闭式重训 `B_φ`，两条 vdW 腿共用这份权重（`local_residual/refit.py` + `runabfe.py`）。取代旧四步重训链的 ①②。**离线测试全绿，真机一次没跑过** —— [EXP-033_P1_LANDED_2026-09-12.md](EXP-033_P1_LANDED_2026-09-12.md) | 否 |
| 09-12 | **EXP-033 P3（跨配体通用权重）划掉**（用户拍板）：P3 的前提是「逐配体重训很贵」，P1 把重训压成一次闭式求解后前提不成立。连带**不必改 CUDA kernel**（省掉 `KNOWN_PLUGIN_SOURCE_SHA256` 两处同步 + G0–G4 重验 + 成本门重跑）。EXP-033 只剩 P2 开着 | — |
| 09-12 | **`resources/outer_lambda_local_residual/` 改为随首发**（用户拍板，翻转 08-31 的「residual sampling 不随首发」）：当时不发是因为「只能跑 Atenolol、永远没法换体系」，P1 闭式重训之后这条前提没了 | — |
| 09-12 | **文档补上「CUDA 插件必须自己编译」**：仓库不发任何 `.so`（`.gitignore` 排除 `plugins/*/build*`），而 loader 默认路径正是 `build/`，新 clone 一开开关就炸。写进 GETTING_STARTED《CUDA 插件》+ TROUBLESHOOTING。支持的 SM 范围由 mamba 环境的 `cuda-version=12.9` 决定（PTX sm_75~120），插件本身不锁 SM | — |
| 09-12 | 诊断脚本里两个个人路径 argparse default 改 `required=True`，加 `tests/test_no_personal_paths_as_defaults.py` 全仓 AST 扫描防复发 | 否 |
| 09-12 | **新增 clone 可导入门** `tests/test_fresh_clone_imports.py`：AST 扫全部已跟踪 `.py`，模块级 import 未跟踪模块 → 硬红（clone 当场 `ModuleNotFoundError`），惰性 import → 软门登记。**这道门就是发布清单本身。** 实测捞出 `step_guard.py` 未跟踪导致四个入口模块在 fake clone 里全死 | 否 |
| 09-11 | **Stage-2 分窗与多采样段重构**：一次会话查出并修掉八个真 bug（预热失败与生产质量门混淆、f_k 加帧前从不重标定、第二段采样被 bridge rescue 悄悄丢掉、分窗判据用等弧长而非 ∫g dλ 等）。全量 2083 passed | 否 |
| 09-11 | **Stage-2 自治闭环接通**：`abfe_pipeline._run_stage2_autonomous()` 的 `decide → execute → reread`，默认开。剩余 4 件活见 [STAGE2_AUTONOMOUS_LOOP_STATUS](STAGE2_AUTONOMOUS_LOOP_STATUS_2026-09-11.md) | 否 |
| 09-11 | 三轴耦合（分窗口/分 λ/分采样量）现状快照 + 路径最小修补计划稿（**计划，未实现**） | — |
| 09-10 | **溶剂腿 5.5σ 残差结案**：100% 是参照臂的盒错了（一直跑建系盒 43.950 nm³，比 1 bar 平衡密度大 3.15%）。换盒重跑残差 −4.318(5.5σ) → **+0.592(0.72σ)**。09-09 写的「密度 20% + 单混合重加权 80%」两半都作废 | 否 |
| 09-10 | R1 残差模型**绑配体、不绑体系**这一事实落成文档与重训链（换蛋白/换膜/换腿都不用重训） | 否 |
| 09-09 | **PBC-01**：链数 > 26 时 `topology.cif` 往返造假键、把水撕开。归组改为只信 System，载入时删假键 | 否 |
| 09-09 | **全仓审计**：9 个只读代理覆盖 77,588 行，62 条候选 → 52 条已修、7 条有意不修（登记为 `AUDIT-01`~`07`）。**52 处改动无一上过 GPU** | 否 |
| 09-09 | `PYMBAR_DISABLE_JAX=1`：pymbar 的 JAX 后端慢 17 倍、胖 7 倍且不归还宿主内存 | 否 |
| 09-09 | **EXP-031 路线 A+B 并入主线**（融合内核不并）。端到端 1.09×，不是单步的 1.22× | — |
| 09-05 | `IBS_BIAS_PROTOCOL_VERSION` **32 → 33**：摘掉 K 个恒零的 `cv_k_rest` 占位 CV。数学上精确恒等（能量逐比特相同），但 system XML 变 ⟹ `system_xml_sha256` 变 | **是** |
| 09-03 | `THERMODYNAMIC_PATH_PROTOCOL_VERSION` **21 → 22**：λ 布点增加可选的自由能定向加密。默认关时与 v21 逐字节相同 | 否 |
| 09-03 | **RBFE 线 R1a/R1b/R2**：原子映射、hybrid builder、独立窗口 + MBAR 分析落地。A→A 自边 ΔG = 2e-8 | — |
| 09-03 | `warmup_profile` 四档开关**整套删除**：`no_ghost_ramp` 被判定热力学上错误，预热只剩默认一条路 | 否 |
| 09-02 | **λ-WCA 防护壳退役**（`WCA_SHIELD_RETIRED=True`，`WCA_ACCOUNTING_VERSION` → 3）。这是 4W53 的 ΔG_bind 从 **+12.75 变成 −21.36 kJ/mol** 的原因 | **是** |
| 09-02 | 撤销 `docs/status/`——它本身就是维护规则第 1 条不许有的「平行当前状态文档」 | — |
| 09-01 | `ESS_GATE_PROTOCOL_VERSION` **4 → 5**：raw 权重退化量改在**去相关之前**算。旧写法拿权重挑样本、再用挑出的样本量同一批权重的退化，是循环论证；实测判定被直接翻转（min 绝对 ESS 3.58 → 47.01，门槛 20） | 否（不进指纹） |
| 09-01 | checkpoint **跨 platform 迁移**：CPU 写的 checkpoint 此前载不进 CUDA、被当损坏，白烧 5M 步 | 否 |
| 09-01 | `0831issue` 第九轮审查的 13 条 P1 + 43 条 P2 收口 | — |

## 2026-08

| 日期 | 变更 | 破缓存 |
|---|---|---|
| 08-31 | **发布整理**：移出 200+ 文件，`docs/` 与 `curated_project/` 两套文档合并成一套，开发期材料压缩成 [HISTORY_LOG](HISTORY_LOG.md)。本仓库自此是**工程区分支** | — |
| 08-31 | **主线代码库迁移**到 `ABFE_IBS/`（新 git 库、大版本起点）。Atenolol-rank11 保留但不再是唯一真源 | — |
| 08-31 | `IBS_BIAS_PROTOCOL_VERSION` **31 → 32**：逐帧 `e_offset` 从 `energy_buffer` 泄漏进 `tmbar_history` 的 `u_kn`，等价于人为注入共模因子，把 raw 单参考 ESS 压到个位数 | **是** |
| 08-29 | `LIGAND_COM_RESTRAINT_PROTOCOL_VERSION` **1 → 2**：**移除** Group 5 配体 COM 约束。非周期绝对锚点在 CUDA 上与被折叠的 centroid 成像规则不一致，形成永久激活的错误外力；CPU 与静态测试都检不出 | **是** |
| 08-28 | **Stage-2 根因定案**：ΔG_bind 错 +32 kJ/mol 不是循环/Boresch/λ 布点/估计器的问题，是每窗口单轨迹重加权抓不到空腔重组熵项。**所有既有收敛门对这个失效模式失明** | — |
| 08-27 | 主线体系切到 **4W53**（T4 lysozyme L99A + toluene） | — |
| 08-26 | `IBS_BIAS_PROTOCOL_VERSION` **30 → 31**：f_k 在线学习喂的是纯物理能量、不含残差项，而实际驱动采样的偏置力含残差 —— 拿不含残差的信号去学一个含残差的分布 | **是** |
| 08-25 | `IBS_BIAS_PROTOCOL_VERSION` **29 → 30**：新增 `s_residual`，EM 阶段对 residual 臂临时关闭残差耦合（全强度残差力下的无温控最小化会让局部环境原子数雪崩） | **是** |
| 08-24 | **EXP-029 / EXP-030 接线**；P1-19 在线 split-half 的"window 0"是滑动列表位置不是物理窗口 | — |
| 08-13 ~ 14 | **EXP-025 ~ 028**：Local Many-Body Residual CUDA 插件全链（G0 ABI → G1 Reference → G2 brute-force → G3 local CSR → G4 成本门）。EXP-025 成本门**未过**（1.1234 vs ≤1.07）判 STOP；EXP-026 Patch A1/A2 优化后 O4 **过**（1.0414）；EXP-028 找到真凶：每步 `addArg()` 而非 bind-once，参数向量无界增长，30k 步内每步成本翻倍 | — |
| 08-04 ~ 09 | **膜体系 / charge-transfer 线** B3（复合物腿 charge transfer）、B4（溶剂腿 co-ion）、B5（离线测试全绿 1161 passed）、C1（水盒离子简化）、C2（密度 + NVT + co-ion 选点，五轮修复）、C3（端点验证 150 帧全过） | — |
| 08-06 | `VDW_NONBONDED_PROTOCOL_VERSION` 引入（MEM-00h 双边归一化） | — |
| 08-03 | MEM-17 帧数门删除：重复帧是真的，但根因在 DCD append，不该往质量门加帧数对账 | — |

## 2026-07

| 日期 | 变更 | 破缓存 |
|---|---|---|
| 07-29 | **Boresch 二面角符号反号事故**修复（BOR-01） | **是** |
| 07-29 | **DEXP 合并进 `abfe_core`**：它只是替代 LJ 的解析形式，只吃 alpha/beta；退役的拟合代码隔离进 `dexp_退役.py` | — |
| 07-27 | **移除四块不可达代码**：ensemble 变异自动修复循环、`--parallel-stages`、medium-probe λ 精修、重叠 vdW 调度设计。逐字存档在 `archive/removed_*.md`，由 `test_att27_dead_code_removed.py` 钉住防回归 | — |
| 07-20 ~ 26 | `THERMODYNAMIC_PATH_PROTOCOL_VERSION` **v8 ~ v21** 的长链演化：v8/v9/v10/v11 四版**全部撤回**；v12 改由热力学长度布点；v13~v16 逐次修 window 0 分组；v17 全 5 窗真机跑通；v18 纯 Fisher 等分**又失败**（解耦尾塌缩）；v19 退回二次调度；v20 插两个实测中点 → 23 态 / 6 系综；v21 改为弧长与几何进度的**混合**等分 | **是** |
| 07-19 ~ 26 | `IBS_BIAS_PROTOCOL_VERSION` **v13 ~ v29** 的长链演化：v15 收敛门换成论文的 LSE 自洽方程；v16 更新式与判据对齐；v17/v18 修时间平均估计器的重要性校正与配分函数；v19 换成 TMBAR；**v21/v22 的 f_k 符号翻转本身是错的、v27 已回退（永远不要再加那个负号）**；v24~v26 修预热预算与 best-effort 放行；v29 整体换成局部滑窗 MBAR loose gate | **是** |
| 07-16 ~ 18 | **IBS resume 一批修复**：mode-fallback、checkpoint 续跑、ladder 预算与终止态、累计步数重复计数、跨进程 ladder 状态丢失 | — |
| 07-10 | DEXP / MACE surrogate 工作线的第一份交接记录（本项目文档记录的起点） | — |

## 2026-06 及更早

`output/dexp_experiment_OLD/`（2026-06-05）是现存最早的产物：DEXP 核与 LJ / MACE 的对比实验，
当时还在 Atenolol-rank11 上、还没有 IBS。更早的材料没有留下可核对的记录。

---

## 协议版本现值

以下由 [`tests/test_doc_staleness_contract.py`](../tests/test_doc_staleness_contract.py) 对着源码常量钉住。
**这张表只在 [STATUS.md](STATUS.md) 和这里各存一份，别再往第三处复制。**

| 协议 | 常量 | 值 | 最近一次跳变 |
|---|---|---|---|
| IBS 偏置 | `ibs_engine.IBS_BIAS_PROTOCOL_VERSION` | **33** | 2026-09-05 |
| 热力学路径 | `abfe_preoptimizer.THERMODYNAMIC_PATH_PROTOCOL_VERSION` | **22** | 2026-09-03 |
| ESS 门 | `ibs_engine.ESS_GATE_PROTOCOL_VERSION` | **5** | 2026-09-01 |
| WCA 记账 | `ibs_engine.WCA_ACCOUNTING_VERSION` | **3** | 2026-09-02 |
| LJ 长程修正 | `ibs_engine.TRADITIONAL_LJ_LRC_PROTOCOL_VERSION` | **3** | — |
| 配体 COM 约束 | `ibs_engine.LIGAND_COM_RESTRAINT_PROTOCOL_VERSION` | **2** | 2026-08-29 |

「破缓存」= 兼容集合收窄成新版本自己，既有 `dual_window_*` / `ibs_state_*` /
`convergence.json` 全部失配、必须重跑，**不要试图复用**。
它**不作废**已经登记在 [STATUS.md](STATUS.md) 的数字——那些是已完成运行的记录，
失配影响的只是 resume 与缓存复用。

## 往这里加条目的规则

1. 一行一条，写"改了什么"，不写"为什么这么改"——理由写在源码注释或专题文档里，这里只指路；
2. 协议版本跳变必须标破不破缓存；
3. **计划、实现、测试通过、科学验证是四种状态**，条目里别混成"成功"；
4. 数字要带单位和符号约定，或者不写数字；
5. 不原地重写旧条目来"修正历史"——写新条目、指明它取代了哪条。
