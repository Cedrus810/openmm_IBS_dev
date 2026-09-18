# Stage-2 控制器：2026-09-18 benchmark 批次崩溃分诊

> **状态：只分诊、未动手。** 等 `abfe-benchmark` 当前这批跑完再决定改不改。
> 数据由同僚会话 `abfe-benchmark-c2` 提供（09-17 23:00 – 09-18 03:00 的 7 次崩溃），
> 代码侧由本会话逐条核到 `文件:行号` 与真机台账。
>
> **⚠️ 这批跑的是 09-17 整波修复之后的代码**：工作树 `abfe_pipeline.py` / `abfe_preoptimizer.py`
> 的 mtime 是 09-17 19:51 / 20:15，早于首崩（23:00）；commit `9a18ded` 是 09-18 00:27。
> 也就是说 [STAGE2_CONTROLLER_WAVE_2026-09-17.md](STAGE2_CONTROLLER_WAVE_2026-09-17.md)
> 那 10 条修复**没有**挡住下面这两条链。

---

## 0. 三句话

1. **根因是块账两侧不对齐**：`_BLOCK_CHARGING_ACTIONS` 声明 4 个动作消耗生产块，
   而 `plan()` 里那道块数硬上限闸**只拦 `RUN_PRODUCTION`** ⟹ 重标定/探针/临时生产
   照样扣配额、从不过闸。真机实测一个窗口因此吃到 5 块（上限 4）。
2. **止血口是退出前那次全路径 ANALYZE**：它在有 stale 证据时被整个跳过 ⟹ 结果停在
   `ANALYSIS_INCOMPLETE` ⟹ `_assert_stage_result_sane` 抛 RuntimeError ⟹ **整跑零产出**，
   连诊断都没有；而诊断缺失又让 `S2-A` 那道刹车永远不带电 ⟹ 下次 `--resume` 逐字复现。
3. **`cmet_ligand2` 三个 rep 全崩不是控制器 bug**，是已知的慢模态死局，别往控制器里查
   （见 §4）。

---

## 1. 崩溃分布（同僚提供）

7 次全部抛在 `abfe_pipeline.py:8407`（`_assert_stage_result_sane`，
`analysis_status='ANALYSIS_INCOMPLETE'`），调用链
`runabfe.py:8065 main → abfe_pipeline.py:17689 run_full_pipeline`。

| 类 | 终态 | 次数 | run |
|---|---|---|---|
| A | `routing_signal=LOCAL_VALIDATION_CAP`（15/15 批，Δf−ΔF 从未求出） | 2 | `brd4_ligand1/rep1`(w5)、`cmet_ligand1/rep2`(w3) |
| B | 求解器提前退出 `window_overlap_broken` | 3 | `brd4_ligand1/rep3`、`cmet_ligand2/rep2`、`jnk1_ligand1/rep2` |
| C | 路径缺窗（6 个有效窗口只解出 4 个，跳过 `[4,5]`） | 2 | `cmet_ligand2/rep1`、`cmet_ligand2/rep3` |

按体系：`cmet_ligand2` 3/3 全崩零产出，`cmet_ligand1` 1，`brd4_ligand1` 2，`jnk1_ligand1` 1；
`brd4_ligand2` 与 cyclod 三个体系一次没崩。

同批跑完的 11 个结果：`results_untrusted=false` 4 个，
`publishable_as_accepted_result` **0** 个，`precision_status` 全 `UNMEASURED`。

---

## 2. 缺口 N：块账**写侧 4 个动作、读侧 1 道闸**

### 位置

```
abfe_preoptimizer.py:3526   _BLOCK_CHARGING_ACTIONS = ("RUN_PRODUCTION",
                                "PROBE_REANCHOR_EPOCH", "RECALIBRATE_FK",
                                "PROVISIONAL_PRODUCTION")     ← 写侧：这 4 个都扣账
abfe_preoptimizer.py:5533   if action == "RUN_PRODUCTION":    ← 读侧：只有这 1 个过闸
                                _adm = {w: _frames_admission(w, ...)}
```

`_frames_admission`（`6205`）是块数硬上限 + 边际增益刹车 + 射程闸三道的唯一入口。
另外三个动作**记账但不受账约束** —— 它们花掉的正是那个窗口稍后要用来重采的配额。

### 真机证据：`cmet_ligand2/rep3` 的 w4 逐轮

（读自 `runs/cmet_ligand2/rep3/checkpoints/stage2_autonomous_history.json`）

| 轮 | 动作 | 目标 | path_version | w4 `production_steps` / `segment` |
|---|---|---|---|---|
| 1–2 | RUN_PRODUCTION | [1] | 1 | 250000 `vanishing` |
| 3 | **INSERT_LAMBDA** | [1] | 1→2 | 250000 `vanishing` |
| 4–9 | RUN_PRODUCTION | [1][2][2][3][3][4] | 2 | `None` / `None`（证据被作废，占位记录） |
| 10 | RUN_PRODUCTION | [4] | 2 | 250000 `vanishing` |
| 11 | PROBE_CANDIDATE_FK | [4] | 2 | 500000 `vanishing` |
| 12 | **RECALIBRATE_FK** | [4] | 2 | 500000 `vanishing` |
| 13 | **RECALIBRATE_FK** | [4] | 2 | 250000 **`vanishing_2`** ← 500k 步作废 |
| 14 | **RECALIBRATE_FK** | [4] | 2 | 250000 **`vanishing_3`** ← 又作废一次 |
| 15 | INSERT_LAMBDA | [4] | 2→3 | 250000 `vanishing_3` |
| 16 | NO_ACTION | [4] | 3 | 「已经批过 **5** 块帧（上限 4）」 |

w4 的 5 块 = 2 块 `RUN_PRODUCTION`（轮 9/10）+ 3 块 `RECALIBRATE_FK`（轮 12/13/14）。
**「超上限一块」不是旧代码留下的历史欠账，是这一跑当场吃出来的** ——
那三块重标定从没被上限看过一眼。

轮 12–14 正是 `abfe_preoptimizer.py:2325` 那段 `S2-H` 注释里逐字记载的形状
（「新段 = 上一段累计步数作废」，真机 `cmet_ligand1/rep1` 750k→250k）。
`S2-H` 的次数闸（`STRUCTURAL_ACTION_MAX_PER_WINDOW = 3`）确实在**第 4 次**拦住了，
但前三次按定义免费，而三次已经够把块配额吃光 ——
**次数闸与块账是两本互不知情的账。**

### 候选修法（未实施）

`plan()` 里 `if action == "RUN_PRODUCTION":` → `if action in self._BLOCK_CHARGING_ACTIONS:`。

一行，而且形状正确：把「写侧记账」与「读侧拦截」合成同一张表，
正是本仓反复复发的那类缺陷（同一不变量两份实现）的标准修法。

**判据**：`cmet_ligand2/rep3` 那份台账离线重放时，w4 的第 5 块（轮 14 的
`RECALIBRATE_FK`）必须被改写成终态而不是被批准。

**⚠️ 未验证的风险**：会让重标定/探针在配额满时改走终态，需跑一遍全部控制器测试，
确认没有既有盘面依赖「这几个动作不被块闸拦」。

---

## 3. 缺口 O：退出前 ANALYZE 被跳过 ⟹ 零产出 + 自锁

### 位置

`abfe_pipeline.py:12597-12618`。判据是控制器视图的
`missing_windows` 与 `stale_layout_evidence` 任一非空 ⟹ 跳过收尾 ANALYZE。

### 真机证据：四个 run 的 `outcome.final_analyze_skipped`

```
cmet_ligand2/rep1   ['stale_layout_window_5']
cmet_ligand2/rep3   ['stale_layout_window_4', 'stale_layout_window_5']
brd4_ligand1/rep1   ['stale_layout_window_4', 'stale_layout_window_5']
cmet_ligand1/rep2   ['stale_layout_window_4', 'stale_layout_window_5']
```

四个 run 的终态全是 `NO_FEASIBLE_ACTION` / `exit_emitted_by=controller`，
理由全是同一句「已经批过 N 块帧（上限 4）」，逐块判据量全是
`[('vanishing', None), ('vanishing', None), ...]`。

### 完整链条

1. 控制器发 `INSERT_LAMBDA`（rep1 三次、rep3 两次、`cmet_ligand1/rep2` 三次），
   按契约**只提交布局、不采样**（`abfe_pipeline.py:12108` 那段长注释）；
2. 下游窗口证据因此 stale，重采要花块配额；
3. 配额已被 §2 那三个不受闸的动作吃掉 ⟹ `NO_FEASIBLE_ACTION`；
4. 收尾 ANALYZE 因 `stale_layout_evidence` 非空被跳过；
5. stage 结果停在 `ANALYSIS_INCOMPLETE` ⟹ `8407` RuntimeError ⟹ **整跑零产出**。

第 4 步是**自锁**：跳过 ANALYZE ⟹ `solver_n_frames_decorrelated` 永远算不出来
⟹ 就是台账里那串 `None` ⟹ `S2-A` 的边际增益刹车永远不带电
⟹ 下次 `--resume` 逐字复现同一条链。

`12584` 那段跳过的理由（「跑它就是重新采样，绕过控制器刚做的停止决定」）**本身成立**，
它没算到的是代价：不是"少一份诊断"，是**整跑炸掉**。

### 候选修法（未实施，两条择一）

- **收窄跳过判据**：stale 但**盘上有帧、只是 λ 对不上**时，合并求解是「重解」不是「重采」，
  不该一律跳。要区分得开「缺产物」与「产物在但身份不符」。
- **或者别让 `8407` 炸**：跳过收尾 ANALYZE 时落一份 `results_untrusted=true` 的结果。
  同批 11 个跑完的结果里已有 4 个 trusted / 0 个 publishable，
  说明「不可信但有产出」这条路已经能表达。

**判据**：上面四个 run 以 `--resume` 重跑时，必须产出可读的 stage 结果
（哪怕 `results_untrusted=true`），而不是 RuntimeError 退出。

---

## 4. 已排除，别重新论证

- **`cmet_ligand2` 三个 rep 全崩不是控制器 bug。** 那是已记录的物理死局：
  λ 0.26–0.195 段 g=82 的慢模态，真机 A/B 已把加帧 / 缩窗 / 插 λ / 重标定四条全部否掉，
  杠杆是 τ 不是这四个。控制器在那儿如实停下是**正确行为**。
- **`[f_k 探针] 计算失败（不影响本轮 rescue）` 是症状不是原因。**
  出处 `abfe_pipeline.py:17231`，它以 `only_windows=None` 载全路径 ⟹ 碰到 stale 窗口必抛
  `ValueError`，被 `except` 接住、不影响流程（rescue 轮数在自治模式下恒为 0，
  那句日志的措辞是历史遗留）。
  自治循环那个调用点 09-16 修过同一条（`abfe_pipeline.py:13687`「局部动作不许吃全路径」），
  **这个报告性调用点没跟上** ⟹ 在任何布局演化过的 run 上它恒失败，
  `stage2_fk_recalibration_probe.json` 因此不更新。属 P3，随手可补。
- **`no_local_tmbAR_results` 的怪大小写是源码原样**（`ibs_engine.py:23288`），
  `abfe_pipeline.py:2932` 有注释专门交代，不是拼写 bug，别去"修"它。

---

## 5. 批次跑完之后的补充（2026-09-18，同僚会话 `abfe-benchmark-a6`）

最终计数：20 完成 / **9 真崩** / 1 在跑（`jnk1_ligand2/rep3`）。

⚠️ 先前写的"13 崩"是错的：13 个崩溃 run 里有 **4 个是历史归档目录**被扫进来的误报。
同僚上报的 3 个"新签名"核完是 **1 真 2 假**，随后同僚自己又补出另外 2 具（p38 那两条）。

### 5.0 崩溃计数的口径：先按 mtime 切到本批

本批首崩 09-17 23:00。13 个崩溃 run 的 `launch.log` mtime：

| run | mtime | |
|---|---|---|
| `brd4_ligand1/rep1` | 09-17 23:59 | 本批 |
| `brd4_ligand1/rep3` | 09-18 01:46 | 本批 |
| `cmet_ligand1/rep2` | 09-18 02:18 | 本批 |
| `cmet_ligand1/rep3` | 09-18 03:11 | 本批 |
| `cmet_ligand2/rep2` | 09-18 03:27 | 本批 |
| `cmet_ligand2/rep3` | 09-18 03:57 | 本批 |
| `cyclod_ligand1_outer/rep1` | 09-18 03:43 | 本批 |
| `cyclod_ligand3/rep3` | 09-18 04:05 | 本批 |
| `jnk1_ligand1/rep2` | 09-18 02:47 | 本批 |
| `cyclod_ligand2/rep1_evidence` | **09-11 14:23** | ❌ 尸体 |
| `cyclod_ligand2/rep1_evidence_run2` | **09-11 20:00** | ❌ 尸体 |
| `p38_ligand1/rep1` | **09-16 06:29** | ❌ 尸体 |
| `p38_ligand2/rep3` | **09-16 04:09** | ❌ 尸体 |

⟹ **本批真实崩溃 9 次。** 另外：先前写的「新进 benchmark 的三个体系
`p38_ligand1` / `p38_ligand2` / `cyclod_ligand1_outer` 都用已知签名」那句**作废** ——
p38 两个体系**这批根本没跑**，它们那两条 traceback 是 09-16 的，
连本批代码版本（09-17 起才算数）都不是。

**扫 `launch.log` 取崩溃签名之前必须先按 mtime 切到本批。** 这次的误报率是 4/13。

### 5.1 ❌ 其中两条是 09-11 的尸体

`cyclod_ligand2/rep1_evidence`（`IBSIncompleteStageCoverageError`）与
`cyclod_ligand2/rep1_evidence_run2`（`IBSWarmupConvergenceError`，"0 次权重更新"）
**都不是本批产物**：

```
rep1_evidence/launch.log        mtime 09-11 14:23
rep1_evidence_run2/launch.log   mtime 09-11 20:00
rep1/launch.log                 mtime 09-18 01:17   ← 这批的
```

两个目录自 09-11 起一个字节没动，比自治控制器首次跑通（09-12）还早。
它们的 `run_provenance.json` 也自证：

- **`config.output`**（注意是 `config` 里那个键，**不是顶层** `output` —— 顶层没有这个键，
  读顶层会得到 `None`；一行验：
  `python3 -c "import json;print(json.load(open('run_provenance.json'))['config']['output'])"`）
  写的是 **`.../cyclod_ligand2/rep1`** —— 这解释了同僚问的
  「来源路径指向 `rep1/vanishing_2`，但 run 目录是 `rep1_evidence`」：
  进程当时就是以 `--output .../rep1` 跑的，目录是**事后被改名/归档**成 `_evidence` 的；
- 配置是旧的（`stage2_window_max_states=8`、无 `stage2_first_window_max_states`），
  与 09-11 吻合，也解释了报错里那个 `窗口 0 λ范围=[0,7]`（K=8，当前配置下不可能出现）；
- 目录里**没有** `vanishing/`、没有 `final_*`，只到 pre-equilibration + rebalance。

⟹ **别为这两条改任何代码。** 教训是操作性的：扫 `launch.log` 取崩溃签名时必须先按
mtime 过滤到本批，否则历史归档目录会伪装成新缺陷。

### 5.2 ✅ 第三条是真的，而且是 `S2-N`/`S2-O` 的第三个实例

`cyclod_ligand3/rep3`（`launch.log` mtime 09-18 04:05），
`routing_signal=WARMUP_F_K_NOT_CONVERGED`，仍走 `8407`。

逐轮台账（w1 = `production_steps` / `min_n_eff_over_g`）：

| 轮 | 动作 | 目标 | pv | w1 |
|---|---|---|---|---|
| 2 | INSERT_LAMBDA | [0] | 1→2 | — |
| 7 | INSERT_LAMBDA | [0] | 2→3 | — |
| 11 | **RECALIBRATE_FK** | [1] | 3 | 250000 / **1.255** |
| 14 | **RECALIBRATE_FK** | [1] | 3 | 500000 / **1.575** |
| 15 | **RECALIBRATE_FK** | [1] | 3 | 250000 / **0.378** ← 新段，500k 作废 |
| 16 | NO_ACTION | [1] | 3 | 250000 / 0.948 |

**`S2-H` 的效果闸首次在真机触发，而且判对了**：
`末点 0.378 / 基准中位数 1.42 = 0.27 < 0.9` ⟹ 拒发第 4 次。
这是设计内的正确行为 —— 它确实拦住了一个正在把窗口推得更差的动作。

但两条已登记的缺口原样复现：

- **`S2-N`**：轮 11/14/15 那三块 `RECALIBRATE_FK` 同样没过块闸，
  同样在轮 15 把累计 500k 作废回 250k（与 `cmet_ligand2/rep3` 的 w4 逐字同形）；
- **`S2-O`**：`final_analyze_skipped = ['stale_layout_window_2','stale_layout_window_3','stale_layout_window_4']`
  ⟹ 零产出 ⟹ `8407` 抛错。崩溃消息说的是 **w3 预热未收敛**，
  而 w3 在整条 16 轮台账里 `production_steps` 全程是 `None` —— 它**一次都没被采过**：
  两次 `INSERT_LAMBDA`（都打在 w0）作废了下游，而重采所需的动作全被 w1 的三次重标定占掉了。

⟹ **控制器现在有两种把自己停死的方式**（块配额吃满 / 效果闸拒发），
**止血口是同一个 `S2-O`**。这把 `S2-O` 的优先级顶到 `S2-N` 之上：
无论控制器因为什么停下，都不该零产出。

### 5.3 其余 8 次崩溃全是已上报签名

`window_overlap_broken`、`LOCAL_VALIDATION_CAP`、路径缺窗三类，没有新形状。
（原先按 10 次算，扣掉 p38 那两具尸体之后是 8 次。）

一条值得记的变体：`cmet_ligand2/rep2` 这次的 `LOCAL_VALIDATION_CAP` 是
**0/15 批、0 frames**（之前几次都是 15/15 批、600 frames）。
「一批都没跑成」与「跑满 15 批仍无结论」是两种盘面，
分支 3a 的 `PROVISIONAL_PRODUCTION` 只对后者是对症的 —— 前者是否同样适用**尚未核**。

### 5.4 对照实验：不在本文范围

20 个结果：**按臂 n=10 时 MUE 6.72 / MSE −5.92**，
`cyclod_ligand1_outer` 并回母体系后 n=9 时 MUE 7.28 / MSE −6.38 kcal/mol，
区间 [−20.03, +2.23]。
（先前写的 MUE 7.33 / MSE −6.44 **作废** —— 那是把 outer 臂当"缺实验值"剔掉算的。）
**ΔG 系统性偏负已有定论（采样故障，控制器修不了），本文不复述也不重新论证。**
这里只登记一条与控制器有关的：`results_untrusted=false` 9/20、
`publishable_as_accepted_result` **0/20**、`precision_status` 全 `UNMEASURED` ——
最后一项是设计内的（单次 run 无法自证精度，见 `EXIT_SPECS` 的
`ANALYSIS_COMPLETE_PRECISION_UNMEASURED`），不是缺陷。

`cyclod_ligand1_outer` 的口径已由同僚核实并改正：两份 config `diff` 只有两处差异
（`output`、`outer_lambda_local_residual_ibs: false→true`），配体同为 LIG ⟹
它是 `cyclod_ligand1` 的 **outer-λ 臂**，配对的实验值就是后者那个 −2.93，不该单列。

⚠️⚠️ **但这三个 rep 之间不构成 A/B。** config 自带注释（2026-09-16）写明：冻结 manifest
（`resources/outer_lambda_local_residual/manifest.json`）只覆盖 Atenolol，本配体 27 原子
指纹对不上 ⟹ 走 **EXP-033 P1 的自动闭式重训**，三个 rep **各自拟合各自的权重**
（`sampling_score_sha256` 逐 rep 不同）。单臂生产跑可以，但「同一把尺子」的 A/B
要按 [RETRAIN_LOCAL_RESIDUAL.md](../RETRAIN_LOCAL_RESIDUAL.md) 手工冻结一份权重给两臂共用。
⟹ **别拿 outer −4.14 vs baseline −5.36 这个差去论证 outer 臂的效果。**

### 5.5 两条修复已落地（2026-09-18，**尚未上机复验**）

| | 改动 | 回归钉子 |
|---|---|---|
| `S2-N` | `plan()` 的块数准入闸条件 `if action == "RUN_PRODUCTION"` → `if action in self._BLOCK_CHARGING_ACTIONS`；`_frames_admission` 新增 `hard_cap_only`，非补帧动作**只过第一道**（块数硬上限） | `tests/test_block_gate_covers_all_charging_actions_2026_09_18.py` |
| `S2-O` | 跳过收尾 ANALYZE 时落一份 `controller_comparison_manifest.json`（`comparison_manifest()` 是纯聚合、零采样），路径记进 `outcome.final_diagnostic_manifest` | 同上文件最后一条 |

**为什么非补帧动作只过第一道**：第二道（边际增益刹车）与第三道（射程闸）问的都是
「同分布加帧还有没有用」，而 `RECALIBRATE_FK` 恰恰是「加帧没用」时的**对症动作** ——
拿"加帧没用"去否决它，方向完全反了。块数硬上限是**资源账**，与动作是什么无关，
四个动作一视同仁。

**`S2-O` 的范围要说清楚，别记成比实际大**：

- ✅ 修的是「停下时零产出」—— 现在无论以什么终态停，盘上都有一份逐窗验收量 /
  块账 / 动作序列；
- ❌ **没有**解决 `S2-A`：`solver_n_frames_decorrelated` 只存在于 stage 求解结果里，
  而这条路径按定义跑不了求解。那条仍开着（`AUDIT-S2-02`，用户已押后）。
- ❌ **没有**让 `_assert_stage_result_sane` 不抛 —— 路径确实不完整，
  那道 fail-closed 是对的（缺窗口的和不是 ΔG），**不许在这里放宽**。
  先前 `S2-O` 的判据写成「必须产出可读的 stage 结果而不是 RuntimeError 退出」，
  那个判据**是错的**，已改正：RuntimeError 是正确行为，错的只是没有诊断。

全量测试 **2736 passed / 0 failed**；两条改动都做过还原变异验证
（把改动退回旧行为，确认对应测试真的变红）。

#### 代码版本时序：验判据前先按这个切

benchmark 的 run 是**按脚本绝对路径**直接拉 `runabfe.py` 的
（`run.zsh` 里 `ABFE_IBS=/home/ruigengji/ABFE_IBS/ABFE_IBS`，
没有已安装包副本、没有第二棵树 —— 同僚核过 39/39 个 run 的 provenance）
⟹ **工作树一改，下一个起的 run 立刻吃到**，没有中间副本挡着。

本次两条修复的落地时刻（工作树 mtime）：

    abfe_pipeline.py      09-18 08:30:05   ← S2-O
    abfe_preoptimizer.py  09-18 08:36:04   ← S2-N

⟹ **分界线取 `08:36:04`**（两个文件都就位）。

⚠️ **三件事三个工具，别混**（2026-09-18 我在这上面说错过一次，同僚纠正）：

| 要核 | 用什么 | 为什么不能用别的 |
|---|---|---|
| 改动**内容** | `git diff`（新增文件看 `git status --porcelain` 的 `??`） | **未提交的修改 git 完全看得见** —— 跟踪文件的改动进 `git diff`，新文件进 `git status`。mtime 只说"文件被写过"，说不出改了什么 |
| 改动**落地时刻** | 工作树 **mtime** | git 给不出（没提交就没有时间戳），而 run 吃到哪版代码只取决于它启动时树上是什么 |
| 版本**基线** | `git log` / `HEAD` | 未提交的改动不进历史，所以它只回答"基线是哪个 commit" |

（本次：`git status` 应显示 `M abfe_pipeline.py` / `M abfe_preoptimizer.py`、
`?? tests/test_block_gate_covers_all_charging_actions_2026_09_18.py`，`HEAD` 为 `9a18ded`。
⚠️ 仓库根是 `ABFE_IBS/ABFE_IBS`，**上一级不是仓库** —— 在上一级跑 git 会得到
"不是 Git 仓库"，那不代表没有版本控制。）

验本文 §5.5 那两条真机判据时：

- **只拿起始时间晚于 08:36:04 的 run 验**；
- 本批已完成的 run **全部是 08:30 之前跑的**，它们的 `checkpoints/` 下
  **没有** `controller_comparison_manifest.json` 是**预期**，不是修复没生效；
- `jnk1_ligand2/rep3` 起于 08:0x、跨过了改动时刻，但进程在启动时就把代码载进内存了
  ⟹ 它跑的是**改前**的代码，同样不能用来验。

⚠️ **一条操作教训**：还原变异验证是在**这棵活树**上做的（08:35:30–08:36:04 约 34 秒内，
`abfe_preoptimizer.py` 短暂带着故意注入的变异）。事后核对
`runs/*/run_provenance.json` **没有任何一个晚于 08:00** ⟹ 那段时间没有新 run 起来，
**没有造成污染**。但这是运气不是设计：调度器随时可能起新作业，
而它读的就是这棵树。**下次变异验证要在副本上做**，别在 GPU 调度器正在读的树上做。

---

## 6. 原始材料

- 日志：`/home/ruigengji/abfe-benchmark/openmm_IBS/runs/<system>/<rep>/launch.log`、`pipeline.log`
- 台账：同目录 `checkpoints/stage2_autonomous_history.json`
  （`outcome` 段含 `exit` / `exit_emitted_by` / `final_analyze_skipped`；
  `iterations[].snapshot` 是逐轮逐窗盘面快照）
- benchmark 跑的就是主线代码：`run.zsh` 里 `ABFE_IBS=/home/ruigengji/ABFE_IBS/ABFE_IBS`，
  benchmark 目录下**没有**代码副本。
