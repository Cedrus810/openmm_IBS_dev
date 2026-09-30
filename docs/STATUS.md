# 当前科学状态与结果登记

[项目入口](../README.md) · [文档导航](README.md)

> **本页整理截至 2026-09-28**（09-28 只新增「跨软件对照」一节，其余各节内容仍截至 09-18；协议版本一栏最后一次逐条核对是 09-12，六个值与 09-09 相同；
> 09-13 至 09-18 的改动**没有动任何协议版本号**）。
> ⚠️ 09-15 至 09-18 破过缓存的是**配置**不是协议：`stage2_window_partition`
> （`metric_integral`→`state_count`）与 `stage2_window_max_states`（8→5）
> 都在 `_PREOPT_DERIVED_PATH_KEYS` 里，作废窗口缓存、不作废 pilot。
> **本次更新只改「控制器真机验证状态」一节**（那节的标题原来写着"真机零验证"，
> 已经不成立了），结果登记表与物理结论**一行未动**。
> 逐日变更看 [CHANGELOG.md](CHANGELOG.md)。
>
> **2026-09-28 补**：新增「跨软件对照：IBS vs OpenFE vs 实验」一节（只登记，不改其它节）。
>
> 这是本仓库**唯一**声明"当前科学结论"的文档。三份 README 只对外讲用法，
> 不再复制这里的任何表格——2026-09-05 之前同样的三张表在 `README.md`、
> `README_cn.md`、`README_en.md`、`docs/README.md` 各存了一份，
> 热力学路径版本已经在其中三份里烂成了旧值。别再复制。

## 符号约定

```text
Delta G_bind = Delta G_solvent - Delta G_complex + Delta G_APBS
```

## 结果登记

| 数值 | 体系 / 运行 | 状态 | 能否作为最终结论引用 |
|---|---|---|---|
| **`−21.36 ± 0.93 kJ/mol`**（`−5.11 ± 0.22 kcal/mol`） | 4W53，`output_v3_seed20260908`，2026-09-02 | **注册标签待维护者指定** | **否：单 seed（`20260908`），本仓库内无第二个独立重复** |
| `541.94 ± 1.30 kJ/mol`（复合物腿 total，`cyclod_ligand2/rep1`） | abfe-benchmark，2026-09-12 | **已作废（2026-09-14 判定）** | **否：vanishing 项是「只解了末窗一个窗口」的部分和**（`covered_lambda_indices=[17…24]`），被当成完整 ΔG 写进 stage 缓存并被 `final_results` 采信。同体系完整路径是 `+44.09`，该 run 自己的全路径中间结果 `+43.00` —— 差的 90 kJ/mol 全部来自这次冒充。两道新门（子集不得自称裁决 / 覆盖度硬门）已落地 |
| `−23.1622 ± 2.5139 kJ/mol`（`output_lrc_fix`） | Atenolol-rank11 | **已作废（2026-08-24 判定）** | 否 |
| `+40.8362 ± 1.3178 kJ/mol`（旧 `output`） | Atenolol-rank11 | `INVALIDATED` | 否：旧符号约定与当前相反，且有诊断问题 |
| `+16.00 ± 2.20 kJ/mol`（2026-07-27） | Atenolol-rank11 | `INVALIDATED` | 否：陈旧且错误的 Boresch 平衡几何 |

4W53 那一行的对照：实验值 **−23.10 kJ/mol**（`−5.52 ± 0.04 kcal/mol`），
差 **0.41 kcal/mol、1.83σ 内**；质量门同时转健康（溶剂腿 stage2 raw ESS
2.93 → 173.33，top1% 0.828 → 0.047）。逐项证据见
[archive/BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md](archive/BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md)。

文件名包含 `final` 不代表结果可以引用。Atenolol 那三行的原始 artifact 和机器可读
登记表（`RESULT_REGISTRY.csv`）在 `Atenolol-rank11` 工作区，不在本分支；
4W53 那一行的证据在本 `docs/` 里。

## 跨软件对照：IBS vs OpenFE vs 实验（2026-09-28，abfe-benchmark）

> 数据由 abfe-benchmark 会话汇总，原表 `abfe-benchmark/openmm_IBS/crosscheck_openfe_20260928.csv`（UTF-8 BOM）。
> **本节是登记，不是可引用的最终结论**：OpenFE 在 brd4_ligand1 / brd4_ligand2 / cyclod_ligand1 上 n=2，cmet 两个配体 n=1，IBS 各行来自不同批次。数据截至 09-28 21:31（19:41 后补了 brd4_ligand1、cyclod_ligand1 各一个 OpenFE rep2）。

口径：两边同力场、同输入（amber99sbildn + TIP3P + openff-2.0.0、同一份 AM1-BCC 电荷、同结构、同晶体水）。
实验值按「Kd 与 T 无关」换算到 IBS 的 300 K / OpenFE 的 298.15 K（下表列 300 K 值）；
⚠️ brd4 的实验值是 ITC ΔG，可能是 288 K 测的，原文献没核。单位 kcal/mol。

| 体系 | 实验 | OpenFE（均值 ± SD，n；未注明 = n=1） | IBS 均值 ± SD（批次，n） | OFE−实验 | IBS−实验 | IBS−OFE |
|---|---|---|---|---|---|---|
| brd4_ligand1 | −9.99 | −14.65 ± 1.42（n=2：−13.64 / −15.65） | −14.87 ± 0.75（09-27 lcr，3） | −4.72 | −4.88 | −0.22 |
| brd4_ligand2 | −7.70 | −13.10 ± 0.11（n=2） | −13.29 ± 1.16（09-27 lcr，3） | −5.44 | −5.58 | −0.19 |
| cmet_ligand1 | −12.36 | −21.83 | −21.52 ± 5.92（09-25 fixanchor，3） | −9.55 | −9.17 | +0.31 |
| cmet_ligand2 | −7.39 | −15.97 | −15.69 ± 0.41（09-23，2） | −8.63 | −8.30 | +0.28 |
| cyclod_ligand1 | −2.95 | −4.91 ± 0.11（n=2：−4.98 / −4.83） | −3.76 ± 1.22（09-27 lcr，3） | −1.98 | −0.81 | +1.15 |
| **平均** | | | | **−6.06** | **−5.75** | **+0.26** |

- cmet_ligand2 加上 09-24 resume 的 rep3（−17.82）⟹ −16.40 ± 1.26。
- ΔΔG(L2−L1)：brd4 实验 +2.30 / IBS lcr +1.58 / OpenFE +1.54（rep1 单值时是 +0.54）；cmet 实验 +4.97 / OpenFE +5.86 / IBS +5.83。
- **读法**：5 个体系上 IBS 与 OpenFE 平均只差 +0.26 kcal，两者相对实验**同样偏负**（约 −5.8 到 −6.1 kcal）。
  OpenFE 自身的 repeat 间差：brd4_ligand1 2.0、brd4_ligand2 0.15、cyclod_ligand1 0.15 kcal ⟹ brd4_ligand1 上 IBS−OFE 的 −1.2 已落回 OpenFE 自己的散布之内。
  ⟹ 偏负（见下文控制器一节的 ⚠️、`BOX-01`）的主要部分**两套软件共有**，更可能来自共同的力场 / 输入 / 口径，
  而不是 IBS 特有的采样问题。⚠️ 这只是观察，不是归因：OpenFE cmet 两个配体仍是 n=1（cmet_ligand1 rep2、cmet_ligand2 rep3 在跑）、只有 5 个体系、cmet_ligand1 的 IBS rep 间 SD 5.9，
  实验温度换算（尤其 brd4）未核。
- 同一体系各 IBS 批次（decharging8 / 09-19 / 09-23 / 09-23 repro / 09-24 resume / 09-25 fixanchor / 09-26–27 lcr）
  的均值散布见原表；repro 批在 cmet 两个配体上比同期 09-23 批负 6–11 kcal。

**耗时与采样量（粗略，未控 GPU 型号与是否共享）**：

| 体系 | IBS 全流程 | OpenFE 全流程 | 比 |
|---|---|---|---|
| brd4_ligand1 | 2:46–2:51（3090，lcr） | 13:56（4080S） | ≈5.0× |
| brd4_ligand2 | 2:40–2:47（3090，lcr） | 13:29（4080S） | ≈5.0× |
| cmet_ligand1 | 2:18–2:22（5090，fixanchor） | 16:06（5090） | ≈7.0×（**唯一同型号 GPU 的一对**） |
| cmet_ligand2 | 4:10–4:18（3090，09-23） | 20:41（5080） | ≈4.9× |
| cyclod_ligand1 | 1:29–1:31（5080，lcr） | 17:24（3090） | ≈11.6× |

- OpenFE 总采样约 490 ns（4 fs + HMR，约 1.2×10⁸ 步）：复合物 30 replica × 11 ns、溶剂 14 × 11 ns、预平衡约 6.6 ns。
- IBS（只数了 brd4_ligand1 lcr rep1）：stage 2 复合物 9000 帧 ≈ 4.5M 步、溶剂 ≈ 1.25M 步、平衡 2M 步（早停，请求 5M）；
  stage 1 / attach / pilot / warmup 没数进去 ⟹ 总计估 15–20M 步，约为 OpenFE 的 1/6–1/8，与耗时比同量级。
- 09-27 lcr 批：HEAD `e576a54`（工作树未提交改动没记录，`code_sha256` 为 null）；`run_provenance` 里 `stage2_n_states=22` 只给 pilot 用，
  实际跑的是 26 态等分 vdW 格点（步长 0.04）、窗口 `[[0,6],[5,11],[10,16],[15,21],[20,26]]`。

**OpenFE 侧现状（openfe 1.12.1.dev32，production，298.15 K）**：上表是各体系已跑完的 repeat（brd4_ligand2 = rep2 + rep3，rep3 09-28 19:38 出；其余单 repeat，报的不确定度 0.0 是因为只有一个 repeat；brd4_ligand2 rep1 launch.log 空、无结果）；数据截至 09-28 19:41；
其余 repeat 在跑或待续跑（cmet_ligand1 rep2 09-28 14:25 被误杀，待用 `--resume` 续）。GROMACS FEP-suite（300 K）还没出 ΔG。

## decharging 湮灭配体分子内库仑 —— **2026-09-14 已修复（v5 已落地）**

> 影响：该日之前的所有 ΔG 作废。症状 / 根因 / 为什么一直没暴露 / 修法 / 连带作废
> 的完整原文在 [archive/PME_DECHARGE_V5_CHANGE_RECORD_2026-09-14.md](archive/PME_DECHARGE_V5_CHANGE_RECORD_2026-09-14.md)。

## Stage-2 自治控制器：真机已跑过三批，**仍未有一次以 `DONE` 收口**（2026-09-18）

> 🔴 **[2026-09-18 更新]** 下面那节标题原来是「代码已收口，**真机零验证**（2026-09-14）」——
> **「零验证」已经不成立**：09-16 / 09-17 / 09-18 三批 benchmark 都上了真机。
> 但**验收口径仍未达成**：本批 20 完成 / 9 真崩（另有 4 条是 09-11 与 09-16 的历史归档目录
> 被扫进来的误报）/ 1 在跑，`publishable_as_accepted_result` **0/20**，
> `precision_status` 全 `UNMEASURED`（后者是设计内的：单次 run 无法自证精度）。
>
> **所以下面那三条禁令一条都没解除**（不得声称闭环可用 / 不得删旧修复路径 /
> 控制器产出的 ΔG 不是可引用结果）。
>
> 真机暴露的缺口逐条在
> [STAGE2_CONTROLLER_WAVE_2026-09-17.md](archive/STAGE2_CONTROLLER_WAVE_2026-09-17.md)（11 条，10 条已修）
> 与 [STAGE2_BENCHMARK_CRASH_TRIAGE_2026-09-18.md](archive/STAGE2_BENCHMARK_CRASH_TRIAGE_2026-09-18.md)
> （`S2-N`/`S2-O`，09-18 已修，**尚未上机复验**）。
> ⚠️ ΔG 系统性偏负**重新是开放问题**（09-23：旧口径「已定论的采样故障」作废 —— 采样器已修、复现路线是固定采样）。
> 头号线索是生产盒 bug（复合物腿在输入盒 +5% 上采样），待 `BOX-01` 重跑判定；仍不是控制器缺陷，别混着查。

> 📌 2026-09-14 当天那份修复快照（两轮静态复核 + 五次真机修复、65 条审计对账、
> 2383 passed 的离线验证）已移进 [HISTORY_LOG.md](HISTORY_LOG.md) 第六节 —— 那是当天的快照，不是当前状态。

## 仍开放（不阻塞，但必须随数字一起说）

- ~~**溶剂腿 stage2** 与真值差 −4.318 kJ/mol（5.5σ）~~ **→ 2026-09-10 已关闭。**
  那 5.5σ **100% 是参照臂的盒错了**：参照一直跑在**建系盒** 43.950 nm³，
  而它比 1 bar 平衡密度大 **3.15%**。配对重跑（同 λ 表、同 seed，只换盒）
  −6.594 → **−11.490 ± 0.342**，残差变 **+0.592（0.72σ）**。
  ⚠️ 09-09 写的「密度 20% + 单混合重加权 80%」**两半都作废**——
  单混合效应同盒重测后为零。
  ⚠️ 比较前仍**必须先对齐 LRC**，否则得 −1.49（1.9σ）的假象。
  证据与复现见 [STAGE2_SOLVENT_LEG_ERROR_BUDGET.md](archive/STAGE2_SOLVENT_LEG_ERROR_BUDGET.md)。
- 独立重复、随机种子账本、时间相关不确定度**仍未闭合**。

## 协议身份

直接读自源码常量。**这张表由 `tests/test_doc_staleness_contract.py` 对着源码钉住**，
改了常量而没改这里会红。

| 协议 | 常量 | 值 |
|---|---|---|
| IBS 偏置 | `ibs_engine.IBS_BIAS_PROTOCOL_VERSION` | 33 |
| 热力学路径 | `abfe_preoptimizer.THERMODYNAMIC_PATH_PROTOCOL_VERSION` | 22 |
| LJ 长程修正 | `ibs_engine.TRADITIONAL_LJ_LRC_PROTOCOL_VERSION` | 3 |
| WCA 记账 | `ibs_engine.WCA_ACCOUNTING_VERSION` | 3 |
| ESS 门 | `ibs_engine.ESS_GATE_PROTOCOL_VERSION` | 5 |
| 配体 COM 约束 | `ibs_engine.LIGAND_COM_RESTRAINT_PROTOCOL_VERSION` | 2 |

`ibs_engine.WCA_SHIELD_RETIRED = True`（λ-WCA 防护壳已退役——这就是上面 4W53
那个数字从 +12.75 变成 −21.36 的原因）。

> **2026-09-09：`IBS_BIAS_PROTOCOL_VERSION` 32 → 33**（EXP-031 路线 A+B 并入主线，
> 详见 [EXP-031_GPU_OPTIMIZATION_2026-09-09.md](archive/EXP-031_GPU_OPTIMIZATION_2026-09-09.md)；
> 融合内核本轮不接）。兼容集合已收窄成 `frozenset((33,))`。
>
> ⚠️ `system_xml_sha256` 随之改变 ⟹ **既有 `dual_window_*` / `ibs_state_*` /
> `convergence.json` 全部失配，必须重跑，不要试图复用。**
> 这不作废上表里已经登记的数字——那些是已完成运行的记录；失配影响的是 resume/缓存复用。

## 状态词

- `IMPLEMENTED`：代码存在；
- `VALIDATED`：在明确输入、环境和验收门下通过；
- `CANDIDATE`：可继续验证但不可宣称最终；
- `FAILED` / `INVALIDATED`：保留证据但不得作为当前科学结论；
- `PLAN` / `DESIGN`：计划不是执行结果，也不是生产授权。

DEXP、MACE、ORB、outer-lambda 神经基势、膜体系、charge-transfer、RBFE 都有代码或
计划，但都不在生产主线上——按上面的状态词判断，别按"仓库里有"判断。

## 证据保全

以下目录不得原地改写、移动或删除（原文在 `Atenolol-rank11`）：

```text
output/  output_lrc_fix/  output_lrc_fixonly-complex-charging/
validation/  solvent_box_scan/
memtest/output_membrane_100ns/  memtest/output_membrane_5ns/
```

新算法/新协议写进新的输出目录。旧结果、失败路线和无效结论继续保留，用于复核协议
演化和 bug 根因。

## 新数字进这张表的条件

来源 artifact、单位、符号约定、协议身份、有效性、是否可引用——六项齐了才登记。
不通过改写旧记录来"修正历史"，用替代关系保留。
