# 计划：prescribed ABFE path 的证据驱动最小修补

**2026-09-11 · 计划稿** · 事实依据：`STAGE2_THREE_AXES_COUPLING_2026-09-11.md`

## 0. 定位

**不是** autonomous path discovery，**是** autonomous path repair：

```
prescribed path  →  detect local defect  →  insert minimal repair  →  return to prescribed path
```

骨架固定，**不动**：

```
Attach → Decharge → IBS vdW path → Release
```

修补只存在于 `H_k` 与 `H_{k+1}` **之间**；两端本身不变。目标：

```
H_repair = H_IBS + δH ,   min ‖δH‖   s.t.  O > O_min , ESS > E_min , R_mix > R_min
```

不做 `argmin_γ ∫ds` 的全局最优路径搜索。

---

## 1. 三类修补

| | 名称 | 不变 | 变 |
|---|---|---|---|
| **I** | discretization repair | `H(λ)` 族 | `{λ_k}` |
| **II** | sampling repair | `H(λ)` 族 **和** `{λ_k}` | 轨迹长度 / replica / burn-in / **窗口划分**（拆窗仅末窗，§3ter） / `f_k` |
| **III** | Hamiltonian bridge repair | 端点 `H_i , H_{i+1}` | 插入临时局部桥 `H_i → H̃_1 → … → H̃_m → H_{i+1}` |

### 关键分类结论：窗口划分是 Type II

窗口划分既不改 `H(λ)` 也不改 `{λ_k}` —— 它只改"哪些态共享一个 bias"。

**⟹ 这解释了三轴文档 C2 那条债的性质**：覆盖不足是 Type II 缺陷，而插 λ 历史上
是按 Type I（边级）语义设计的。⚠️ **但别按 C2 早先那版理解** —— 那版说"两个语义
相反的诊断共用同一个动作"，描述的是 09-11 当天一次**被老板否掉并已回退**的改动。
老板的裁定语义（别再接回去）：

> **ESS / 帧数不够是采样问题，结构改动治不了。**
> 插 λ 同样是结构改动（下游 λ 内容左移 ⟹ 已采产物失配），**不是缓和版本**。

现状：`IBSValidationBudgetIndeterminateError`（"没测出来"）**不触发任何布局动作**；
唯一的布局动作触发源是 `IBSWarmupConvergenceError`（"f_k 压不平"）。而在 model B
下插点已经是**窗口级**工具（缩跨度），所以那条 type error 在行为上已被消解，
只剩命名/文档债。

### Type III 的定位

Type III 解决的不是"`λ_i` 与 `λ_{i+1}` 太远"，而是**两个各自合理的 Hamiltonian
族/estimator 在 junction 上没有合法直接连接**。`H̃` 只在这一小段存在，不是新的
ABFE path family，是 **compatibility bridge**。候选实例：`PME → shadow → IBS`。

---

## 2. 现状盘点：每类已经有多少

| | 已有实现 | 状态 | 缺口 |
|---|---|---|---|
| **I** | `insert_lambda_in_failed_ibs_window`（`abfe_preoptimizer.py:1648`） | **位置判据已对**：最长热力学长度边的 pilot 插值中点，支持 `arclength` / `metric_integral`。**数量判据错**：`n_needed = max(1, (2*lo_n−1) − K_failed)` —— 由分窗可行性反推 | 换成 `n ≈ ⌈L_edge / L_target⌉`。**一行公式** |
| **II** | `immutable_bridge_rescue`（`abfe_pipeline.py:13430`）= 不改原窗口、另开采样段、**分析时替换**；多采样段分析；`f_k_recalibration`；`production_rescue_targets`；冻结验证阶梯 | **机制齐全**，是"开新 Epoch / 旧段保留"的现成实例 | 触发判据散在三处；身份不足（无 `EnsembleSpec`）；缩窗在 `K ≤ 6` 时不可行（见下方更正） |
| **III** | **无** | `immutable_bridge_rescue` 名字里有 bridge 但它是**分析级替换**，不是 Hamiltonian 桥 | 全新。最接近的既有腿是 B3 charge-transfer |

**结论：Type I 是一行修补，Type II 是接线收口，Type III 才是新东西。**
整件事确实可以从修修补补开始。

### ⚠️ 更正：插点的记账方式（2026-09-11 讨论定案）

插一个 λ 之后，窗口范围**跟不跟着走**是两套完全不同的记账，物理效果相反：

| | 失败窗口 win0 | 态数 | λ 跨度 | 整窗总 ΔF |
|---|---|---|---|---|
| 原始 | `{λ0,λ1,λ2,λ3}` | 4 | λ0→λ3 | ΔF(λ0→λ3) |
| **model A**（范围 +1；09-11 的代码现状） | `{λ0,λ1,λnew,λ2,λ3}` | **5** | λ0→λ3 **不变** | **一模一样** |
| **model B**（范围不动、溢出流向末窗） | `{λ0,λ1,λnew,λ2}` | 4 | **λ0→λ2 缩了** | **变小** |

`ΔF(λ0→λ3)` 是物理量，与中间放几个点无关 ⟹ **model A 插点后 bias 要压平的总落差一分没少，对"窗口太宽"无效**（不是"更糟"，是"没用"）。只有 model B 真的缩小了窗口要跨的自由能落差。

**⟹ 定案：走 model B。**

```
前缀（插入点之前的窗口）严格不动   ← b_j ≤ at 的窗口 λ 逐位不变，全部复用
尾段范围不动、λ 内容左移一格       ← 顺序跑时这些窗口还没采，零成本
末窗吸收溢出、豁免 max_states      ← 新约束，当前代码没有
```

resume 安全，机制本来就为此而建：`lambda_path_versions.append_version` 同时记 λ 表
与 `window_ranges`，`resolve_path()` 取回当前版本，`changed_windows()` 给可复用清单，
插点事件带确定性 ID 防重复插。**每一次 λ/窗口变动都已落盘，resume 不会算错。**

⟹ `abfe_preoptimizer.py:1756-1765` 那次 09-11 改动（换成 model A）解决的是"重采多少
窗口"的**成本**问题，不是正确性问题，代价是弄丢了"插 λ 能缩窗跨度"这个**能力**。
它抱怨的"window 0 失败时前缀是空的、尾段就是整条路径"——在 model B 下**这正是想要的**：
win0 跨度要缩，整条下游就该跟着加密一格。

### 可行性规则（`feasible()` 要算的就是这三条）

```
插 λ（model B）：溢出由末窗吸收（末窗豁免 hi），但 **K_tail 不得超过 2·hi−1**
拆窗：          仅末窗，且 K_last ∈ [2·lo−1, 2·hi−1]
终点：          末窗拆完后两个孩子都受 hi 约束 ⟹ 溢出槽消失
                ⟹ 下一次插点无处可去 ⟹ **退出进程**，由人工改输入文件的 λ 总数
```

⚠️ **溢出槽不是无底洞（2026-09-11 发现，已 fail-closed）。**
"末窗豁免 hi"容易被读成"末窗可以无限长"，但**能一分为二**这件事本身有区间：
两子窗共享一个边界态 ⟹ `p+q−1=K`，两侧都要落在 `[lo,hi]` ⟹ 可拆的 K 只有
`2·lo−1 .. 2·hi−1`。4/5 配置下是 **7..9，只有三个值宽**：

```
K_tail = 5  还不够拆        K_tail = 8  可拆
K_tail = 6  还不够拆        K_tail = 9  可拆（上限）
K_tail = 7  可拆            K_tail = 10 ★ 永远拆不开 —— 死胡同
```

末窗一旦被插过 `2·hi−1`，就既不能再吸收（迟早压不住）又不能拆，而拆它恰恰是
唯一的收尾动作。当前默认 `max_insertions=3` 只是**巧合**把 K 压在 5+3=8 ≤ 9
（`rounds_done` 是跨 resume 从版本链累计的），那是巧合不是守卫 —— 谁把
`max_insertions` 调大就会踩进去。`insert_lambda_in_failed_ibs_window` 与
`feasible_repair_actions` 两处都已 fail-closed，处置二选一：
**(a)** 趁 K 还在可拆区间内先拆末窗（右孩子重新成为溢出槽）；
**(b)** 判 `HALT_LAMBDA_BUDGET_INSUFFICIENT`。

⚠️ **两个配置都要照顾，别按任何一个写死。**

| 来源 | lo/hi | 可拆区间 `[2lo−1, 2hi−1]` | 溢出预算（末窗从 K 长到上限） |
|---|---|---|---|
| 本仓库 `abfe_config.json` | **4/5** | 7..9 | 末窗 5 态 ⟹ 只有 4 个 |
| **实验用的** `~/abfe-benchmark/openmm_IBS/configs/*.json`（六个体系全部） | **4/8** | 7..15 | 末窗 6 态 ⟹ 9 个 |

两者都配 `max_path_insertions = 3`，所以**当前两个配置下溢出预算都不是约束**
（4/5 用 3 ≤ 4，4/8 用 3 ≤ 9）—— 死胡同守卫在当前实验里不可达。**但那是数字
碰巧，不是规则保证**，见下。

### ⚠️ 边级 n 与溢出预算之间没有任何约束（2026-09-11，同僚 abfe-ibs-27 发现）

**边级判据能产出的 `n`，和溢出槽能吸收的 `n`，没有任何东西保证前者 ≤ 后者。**
`n = Σ max(0, ⌈L_edge/L_target⌉ − 1)` 只由边长决定；溢出预算是
`2·hi−1 − K_tail`，只由布局决定。4/5 下溢出预算只有 4 个态，**一条 5 倍平均长的
边就能要到 6 个**，必然撞守卫。

**处置定案：撞上就 fail-closed 报错，不把 `n` 封顶到预算。** 理由三条：

1. 封顶等于**静默产出一个不满足触发它的那份证据的布局**。没人记下"这次修补是
   部分的"，控制器下一轮看到同一个缺陷再修一次，纯烧轮数。
2. 溢出预算是**真的有限**的：末窗拆过之后两个孩子都受 `hi` 约束（老板定的，
   §3ter.4），所以 4/5 下终身插点预算就是 4 个。"证据要的比布局装得下的多"
   **就是**「输入 λ 总数从一开始就不够」这个结论本身。
3. 封顶会把老板明确要求浮出来的那个条件藏起来（原话：这种情况该**杀掉进程、
   人工改输入文件的 λ 数目**，控制器不得自行加总态数重跑 preopt）。

⚠️ 一条边长到平均的 5 倍，本身说明**初始布点**就不对（度规布点不该产生这种边）。
那是布局期的问题，不是运行时修补能补的 —— 守卫的报错要把这层意思说出来。

---

## 3. 修补的判别（研究核心）

λ 插点不新、adaptive sampling 不新。**新的是怎么区分三类 failure。**

| 证据（多项联合，不是单量越界） | scope | 修补类型 |
|---|---|---|
| 平稳、支撑健康、`σ(N) ~ N^{-1/2}` 收缩 | window | **II — 延长当前 Epoch** |
| `N↑` 但 overlap 不改善 | window | **不是采样问题** → 继续判别 |
| 某边**双向 overlap** 异常 或 `L_{i,i+1} ≫ L_target` | edge | **I — 只在该边插 λ** |
| 各边都健康，但整窗混合 / round-trip 塌陷 | window | **II — 先重标定 `f_k`**；仍压不住则**补 λ 缩跨度**（§3ter.2，model B 下补 λ 才是对症的）。拆窗只留给末窗溢出收尾 |
| `Δf_k − ΔF_MBAR` **带精度地**明确不符 | window | **II — 只 recalibrate `f_k`**，新 Epoch 重新 burn-in |
| replica / wet-dry basin / 慢 CV 不一致 | path | **II — 加 replica / 增强采样** *[需新测量，见 §6]* |
| 两个 estimator / H 族在 junction 上无合法直接连接 | junction | **III — 最小桥** |
| box / Hamiltonian / LRC / 身份不一致 | — | `HALT_INVALID_INPUT`，**不是修补** |
| **现有证据分不清** | — | **跑最小诊断动作**；预算不够才 `HALT_NO_ATTRIBUTION` |

### 判别要用的量（v1 曾把两个 g 混了）

| 字段 | 定义 | 出处 | 回答 |
|---|---|---|---|
| `fisher_metric_g` | `β²Var[∂U/∂λ]` | `abfe_preoptimizer.py:971` | **λ 间距**够不够 → Type I |
| `tau_int` | 时间自相关 | `ibs_engine.py:19037-19044` | **采样时间**够不够 → Type II |
| `friction metric` *[待引入]* | 动力学代价 | Optimal Alchemistry | 区分"热力学距离长"与"动力学慢" |

混用 = 用时间预算回答布点问题，会稳定选错类型。

### `|ΔF|` 不能单独触发 Type I

ΔF 大仍可有良好双向 overlap；ΔF 小也可能因势垒零交换。
（注：现有代码**没有**犯这个错 —— `free_energy_densify_points` 的 ΔF 贪心只在
**布局期**跑一次，不是运行时修补触发器。）

---

## 3bis. 跨窗累加：一个还没被用起来的证据源

**IBS 的样本可以跨窗累加。** 每个窗口是一个已知的采样分布（冻结 `f_k` 的混合），
MBAR 本来就处理多个采样分布 —— 所以"某个 λ 态到底有没有采够"这个问题，不该只问
拥有它的那**一个**窗口。这不是新方法，是现在没用起来的既有性质。

### 3bis.1 实测：判据在窗口内问、答案在窗口外

`cyclod_ligand2/rep1` vanishing，每个窗口对自己每个 λ 态的重加权支撑
（rawESS / 帧数，`g` = `tau_int`）：

```
win0  λ=1.0000 ESS 436/500 g= 1.1     win1  λ=0.6079 ESS 332/500 g= 1.6
      λ=0.9201 ESS 401/500 g= 1.0           λ=0.5622 ESS 436/500 g= 1.0
      λ=0.8444 ESS 228/500 g=23.1           λ=0.5207 ESS 344/500 g= 3.8
      λ=0.7762 ESS 111/500 g=77.5           λ=0.4847 ESS 163/500 g= 8.2
      λ=0.7145 ESS  58/500 g=90.3
      λ=0.6583 ESS  35/500 g=88.0
      λ=0.6079 ESS  16/500 g=85.6   ← 同一个 λ，win1 那边好 20 倍
```

win0 因为"去相关后只剩 6 帧"被**整窗丢出 ΔG 求和**。但真正缺支撑的只有它的后三个
内部态；它的**端点** λ=0.6079 被 win1 采得很好（332 vs 16）。用单窗判据问，答案是
"整个 win0 不可用"；用累加判据问，答案是"win0 的 λ=0.7145/0.6583 缺支撑，端点没问题"。
**后者才是可执行的诊断。**

### 3bis.2 这个不对称是系统性的，不是巧合

每个 join 上，**结束**于该 λ 的窗口都比**起始**于它的那个差（rawESS）：

| join λ | 上游窗口 | 下游窗口 |
|---|---|---|
| 0.6079 | win0 **16** | win1 **332** |
| 0.4847 | win1 **163** | win2 **270** |
| 0.3852 | win2 **205** | win3 **838** |
| 0.3032 | win3 **91** | win4 **138** |

⟹ 一个窗口的**端点支撑不该由它自己单独裁决**。

### 3bis.3 递推单元 = 滑动相邻对，不是累积前缀

相邻窗口**只共享一个 λ**（win0 的 λ 集合与 win2 完全不相交），所以
`win0+win1+win2` 相比两次 pairwise join 不多任何信息：

```
win0 算完      → 查 win0 自己各 λ 的支撑剖面
win1 算完      → 查 join λ=0.6079（两侧独立估计）
win2 算完      → 查 join λ=0.4847        ← 是 (win1,win2)，不是累积
```

⚠️ 这**不是**被否决的那个设计（`[[ibs-overlap-arbiter-false-premise]]`：把
**相邻 λ 态之间的 fixed-H overlap** 当 IBS 正确性仲裁）。这里比的是两个独立系综对
**同一个 λ 态**的重要性支撑，是同一个量的两份独立估计，不是相邻态之间的重叠判据。

### 3bis.4 现在做得到多少

| | 能做 | 卡在哪 |
|---|---|---|
| join λ（相邻共享的那一个） | **现在就能做**，零 GPU，`energies.npy` 里两边都有 | — |
| 窗口**内部**的 λ | 做不到 | 邻窗只存自己那几个态的能量；vanishing 段**不存 DCD**，回头补算也不行 |
| 全局 MBAR（真正的"拼到一起"） | 做不到 | 需要采样时就存**跨窗交叉能量**（每帧在邻窗 λ 上的 reduced potential）。这正是 `[[ibs-overlap-arbiter-false-premise]]` 里记的那条缺口："arbitrary-target evaluability（need saved coords/energy basis — 当前分析器只读离散 `u_kn`）" |

### 3bis.5 新增判别证据

| 证据 | 用现有落盘数据算得出 | 回答 |
|---|---|---|
| **per-state support profile**：窗口内逐 λ 的 rawESS/`tau_int` 剖面 | ✅ | **单调衰减到远端** = 窗口太宽（II 缩窗）；**整体偏低** = 采样不够（II 延长）。比"整窗一个 g" 分得开得多 |
| **join 两侧支撑比** | ✅（仅 join λ） | 端点弱是不是真问题；上游窗口的 ΔF 端点值可否采信 |
| 内部 λ 的跨窗支撑 | ❌ 需协议改动 | — |

---

## 3ter. 拆窗的准入条件：只有最后一个窗口

**结论不变，理由换了。** 旧理由是"任何结构改动作废下游全部产物"——那描述的是
09-11 之前的行为，已不成立（见 §2 更正）。**真正的理由是：溢出只堆在末窗，所以
只有末窗会长到需要拆。**

### 3ter.1 为什么只有末窗会需要拆

model B 下每次插 λ：前缀不动、尾段 λ 内容左移一格、**多出来的那一个态落到末窗**。
于是只有末窗的 K 单调增长；其余窗口的态数永远是布局期定的那个值。

```
插点 ×1：  win0 跨度缩、win1..win2 态数不变、win3 → K+1
插点 ×2：  win3 → K+2
...
```

⟹ 拆窗不是一个"可以对任意窗口施加的动作"，而是**末窗溢出累积到压不住时的收尾**。
非末窗永远到不了 `K ≥ 2·lo−1`，所以"禁止非末窗拆窗"是**构造上自动成立**的，
写成断言只是防回归。

### ⚠️ 更正：下游作废的代价是**真实存在**的，只是被接受了

本节早先一版写过"旧理由（任何结构改动作废下游全部产物）已不成立"——**那句是错的。**
model B 下**插入点之后的窗口 λ 内容会左移一格**，缓存键是 `(window_index, λ 值)`
（`_resume_cached_window_gate_status` 的 `np.allclose`、reuse_map 的
`_window_lambda_key`），所以下游产物确实失配、确实要重采。**插 λ 与拆窗在代价上
是同一类结构改动，插 λ 不是"缓和版本"。**

这个代价是**被明确接受**的，理由只有两条，别扩大：

1. 窗口按 0,1,2… 顺序跑，`win_i` 失败时 `i+1..N` **还没采** ⟹ 首次跑零成本。
2. resume 安全：`lambda_path_versions` 把 λ 表与 `window_ranges` 一起落盘，
   `resolve_path()` 取回当前版本、`changed_windows()` 给可复用清单，插点事件带
   确定性 ID 防重复插。

在 **rescue / 多段采样**（下游已经有数据）场景下这个代价是实打实的重采。
09-11 那次"只拆失败窗口、其它一个都不动"的改动就是为了堵这个洞，而它换来的是
弄丢了"插 λ 能缩窗跨度"的能力 —— 这就是为什么最终定案是 model B + 接受代价，
而不是 model A。

### 3ter.2 补 λ 与拆窗的分工（⚠️ 结论已按 model B 修正）

**旧版本在这里判"补 λ 救不了占据倾斜、拆窗才对症"——那是 model A 口径下的结论**
（同跨度、加密一倍、`K` 翻倍 ⟹ `(K−1)·ε` 一阶不变），在 model B 下不成立。

model B 下：

| 动作 | 对整窗总 ΔF | 对占据倾斜 |
|---|---|---|
| 补 λ（态数不变、跨度缩小） | **变小** | **变小 —— 对症** |
| 拆窗（K 减半） | 每个子窗的落差减半 | 变小 |

⟹ **补 λ 是首选**：它不增加窗口数、不动前缀、顺序跑时下游零成本。
拆窗保留给末窗溢出的收尾。

### 3ter.3 实测反例：win0 看着像该拆，其实是该重标定

`cyclod_ligand2/rep1` win0（`K=7`，λ 1.0→0.6079）：

```
frozen f_k (减均值)  [-45.33 -23.93  -6.74   6.45  16.57  24.00  28.98]
生产帧自洽 f_k       [-59.33 -32.70 -10.60   6.98  21.17  32.71  41.77]
逐态差               [-14.00  -8.77  -3.86   0.54   4.59   8.70  12.79]

相邻 max|Δf_k − ΔF| = 5.24 kJ/mol   ← 阈值 10.0，**合法通过**
整窗累计倾斜        = 26.8 kJ/mol
实测占据 [0.681 0.167 0.060 0.033 0.024 0.019 0.015]   目标 0.143
```

一个**合法通过门**的 `f_k`，被窗口宽度放大成 4.8 倍超 / 9.5 倍欠。但它是
**可重标定**的：生产帧自洽解就在上面。按 §3ter 的规则 win0 **不进拆窗候选**。

管线本来就要这么做（日志：`窗口 [1,2,3,4] 的 f_k 位移超过阈值`），**唯独漏了 win0**
—— 因为它去相关后只剩 6 帧、不满足重标定的 `min_frames=10`。这条链
（门放行 → 占据塌 → 帧数不足 → 跳过重标定 → 不进 rescue → 段 2 缺窗口崩）
才是真正要修的东西，见 P0-2a/2b。
### 3ter.4 动作归属表（model B）

| 症状 | 动作 | 结构改动 | 位置限制 |
|---|---|---|---|
| `f_k` 倾斜（占据不平，但可重标定） | 重标定 `f_k` | 无 | 任意窗口 |
| 支撑整体薄、平稳收缩 | 加帧（延长当前 Epoch） | 无 | 任意窗口 |
| 某边 `L_edge ≫ L_target` | 补 λ | 前缀不动、尾段左移 | 任意窗口 |
| 窗口太宽（重标定后仍压不住） | **补 λ**（缩跨度） | 同上 | 任意窗口 |
| **末窗** K 累积到重标定后仍压不住 | 拆末窗 | 末窗一分为二 | **仅末窗** |
| 末窗已拆过、溢出槽消失 | **`HALT_LAMBDA_BUDGET_INSUFFICIENT`：退出进程**，报「输入 λ 数目不够」 | — | — |

最后一行是这套修补的**天然终点**，也是**唯一**需要显式停的地方：末窗要长到那个地步，
意味着输入的 λ 总数从一开始就不够。

⚠️ **控制器不得自行加总态数重跑 preopt。** 总态数是**输入**，属于 prescribed path 的
定义；自动改它等于偷偷换掉被修补的那条路径，本文 §0 的前提就没了。正确行为是
**杀掉进程、报清楚缺多少**，由人改输入文件的 λ 数目后重跑。

## 4. 升级规则 I → II → III

- **升级必须显式，且"归因不出来"不是升级理由。** 分不清就跑最小诊断，不许升级。
- I 与 II 之间**不排序**：按判别结果直接选类型，不做"先便宜后贵"的阶梯
  （证据表明 `f_k` 明显不符时先加帧是浪费；只缩窗可行时不必先插 λ）。
- **升级到 III 的必要条件 = 待定研究项**：必须能证明 I 和 II 都无法修复该 junction，
  而不是"I/II 试过没成"。

---

## 5. 借什么、不借什么

| 来源 | 借 | 不借 |
|---|---|---|
| **ALS** | Type I 的插点器，直接调用 | — |
| **pylambdaopt / thermodynamic length** | `n ≈ ⌈L_{i,i+1}/L_target⌉`，`L = ∫√g dλ` | — |
| **Optimal Alchemistry** | friction metric 作 **local** cost estimator | 全局 `argmin_γ ∫ds` 找最优路径 |
| **A3FE** | "repair 还是 just sample longer"的判据 | 其 λ selection 策略（它的 window = 独立模拟，加 λ 无 invalidation；IBS 有） |
| **HTBAC / EnTK** | 决策与执行分离的结构 | 框架本体（本仓库串行单卡，`n_workers` 从未实现） |
| **VI / Optimal Transport** | **只**回答"怎么造最小 bridge"（Type III） | "整条 ABFE 最优 H path 是什么" |

⚠️ 外部文献未在本机核实，按转述记录。A3FE 报告其自适应时间分配**在同等成本下未降低
不确定度** —— **不要拿"同等成本下不确定度更低"当验收指标。**

---

## 6. 修补动作的共同要求

不论哪一类，动作只有两种：**延长当前 Epoch** / **开新 Epoch**。

### 6.1 Epoch 身份 = 完整 `EnsembleSpec`

```
λ 状态定义（完整 Hamiltonian）/ topology / 力场 / restraint / soft-core
温度 / NVT-NPT / box 或参照体积
integrator / 时间步长 / barostat
WCA / LRC / sampling-energy vs target-energy 口径
canonical_f_k / protocol_version / code_version
```

**有实测代价**：4W53 溶剂腿残余 −4.32 kJ/mol（5.5σ）的真凶是参照臂**盒体积**不一致
（42.747 vs 43.950 nm³），不是 `f_k`、不是重加权。见 `STAGE2_SOLVENT_LEG_ERROR_BUDGET.md`。

⚠️ **只有用户输入才配做身份。** 自产产物 sha256 进身份是本仓库**已复发 4 次**的 bug。
存语义量（体积数值、温度、口径枚举），不存文件哈希。

### 6.2 `canonical_f_k`：现存 bug

`f_k` 整体加常数不改采样分布，逐位比较数学上就错。
`frozen_candidate_fingerprint`（`ibs_engine.py`）直接 `hash(",".join(f"{x:.6f}"))`，
未规范化 —— `f_k` 与 `f_k+c` 算成两个候选。修：`f_k ← f_k − mean(f_k)` 再进指纹。

### 6.3 LEARN 不是一个 Epoch

在线更新 `f_k` 时采样分布随时间变，整段不是单一平稳系综。二选一写死：
**(a)** 每冻结一版 `f_k` = 一个 Epoch；**(b)** 时变偏置估计器 + **完整**偏置历史
（本仓库走这条：TMBAR，`TMBAR_HISTORY_MAX_ENTRIES=200`）——
选 (b) 则"偏置历史完整"是该 Epoch 可用于分析的**前提**，历史截断即不可用。

### 6.4 verdict 词汇表（三值不够）

`VALID_PASS` / `INSUFFICIENT_DATA` / `ESTIMATOR_UNIDENTIFIABLE` /
`STATISTICALLY_REJECTED` / `NONSTATIONARY` / `REPLICA_DISAGREEMENT` /
`HAMILTONIAN_MISMATCH` / `NUMERICAL_FAILURE`

**为什么**：只有 FAIL/UNMEASURED 时，"FAIL 但有预算 → 再测一次"退化成**重试到碰巧通过**。
有统计功效的否决必须**立刻换 Epoch**；还需确认的就不该叫 FAIL。

### 6.5 停止标准三维

```
execution_status = COMPLETE | HALTED
evidence_status  = CONVERGED | INCONCLUSIVE | REJECTED | INVALID
trust_level      = STATISTICAL_ONLY | REPLICA_VALIDATED | OVERRIDDEN_UNTRUSTED
```

`allow_untrusted_stage_results` **只改 `trust_level`**，不得把 `evidence_status`
改写成 CONVERGED。`CONVERGED` = "所有**已定义**的判据通过"，不等于答案正确
（依据：`STAGE2_ROOT_CAUSE_2026-08-28.md` §2「所有收敛门对该失效模式失明」，
该节被其自身超越声明标为**仍然有效**；但**不要再引用那份文档的 +32 kJ/mol 幅度归因**，
98% 已定罪 λ-WCA 壳、残余已 100% 归盒体积）。

---

## 7. 修修补补计划

### P0 — 一行级，零 GPU，不依赖任何设计决定

1. ✅ **已改** `canonical_f_k` 减均值后再进指纹（§6.2，现存 bug）——
   `ibs_engine.frozen_candidate_fingerprint`。
2. ~~`fisher_metric_g` / `tau_int` 拆成两个字段~~ —— **核实后判定代码里并不存在这个问题**：
   `metric_g` 只出现在 pilot/preopt 侧（Fisher），`statistical_inefficiency` 只出现在
   去相关侧，两者从未混用。混的是本文档早先版本的口径，已在 §3 改正。**不做。**
2a. ~~**`converged` 里补上 `path_is_complete`**~~ —— **核实后判定误报，不做。**
   `converged`（`ibs_engine.py:20384`）第一项是 `len(local_results) == len(valid_windows)`，
   而 `valid_windows`（`:19535`）在跳过判断**之前**就定下、含被跳过的窗口，唯一的
   `local_results.append`（`:20071`）在跳过点 `continue`（`:19693`）**之后**
   ⟹ 缺窗口时 `local_results` 少一条 ⟹ 第一项即 False ⟹ **`converged=False`**。
   同一个不变量已被守住，`path_is_complete` 只是没进表达式，再加一条是重复。
   ⚠️ 连带：原文"2a+2b 缺一不可，只修 2b 则 rescue 第一行 `if converged: break` 就退出"
   也随之不成立 —— **2b 单独修就够了。**
2b. ✅ **已改** 被跳过的窗口进 `_stage_quality_failure_details`（现存 bug）。
   它只遍历 `window_overlap_diagnostics`，而因"去相关后帧数不足"被跳过的窗口在
   生成诊断记录**之前**就 `continue` 了 ⟹ 不在 rescue 候选名单里 ⟹
   **"II — 延长当前 Epoch"这个动作对最需要它的窗口永远不可达**。
   实例：win0 只跑了 250k 步（win3 500k、win4 1M），被跳过后再也没被加过帧。
   ✅ **已核实为真 bug**：`window_overlap_records.append` 在 `ibs_engine.py:20091`，
   跳过点 `continue` 在 `:19693`，确实进不了 `window_overlap_diagnostics`。
   （2a 是误报，见上 ⟹ **2b 单独修即可**，不需要成对。）
2c. ✅ **已改** per-state support profile 落盘（§3bis.5）：窗口内逐 λ 的 rawESS / `tau_int`
   剖面。零 GPU，现有 `energies/bias` 就能算，是区分"窗口太宽"与"采样不够"的
   最便宜证据。现在只落一个整窗 `g`，把单调衰减和整体偏低混成一个数。

### P1 — 小改，零 GPU

3. ✅ **已改** 插点改回 model B 记账（§2 更正）：`shifted`（范围 +1）换成"前缀不动、
   尾段 λ 内容左移、溢出落末窗"；同时把 `n_needed` 改成由 `⌈L_edge / L_target⌉`
   决定，删掉 `max(1, (2*lo_n−1) − K_failed)` —— **拆窗门槛不再参与插点数量**。
   连带删掉无条件的 `best_cut` 就地拆窗：拆窗从此只在末窗、只由 §3ter.4 最后两行触发。
3a. ✅ **已改** 末窗豁免 `max_states_per_window`：`validate_single_shared_boundary_ranges`
   与分窗器都要放开末窗上限（当前代码没有这个概念）。这是 model B 的溢出槽。
4. ✅ **已加** `abfe_preoptimizer.feasible_repair_actions()` 纯计算，按 §2「可行性规则」三条：
   补 λ 总是可行（溢出由末窗吸收）；拆窗仅末窗且需 `K_last ≥ 2·lo−1`；
   溢出槽消失则唯一出口是 `HALT_LAMBDA_BUDGET_INSUFFICIENT`。
   ⚠️ 按 `abfe_config.json` 的 `4/5` 算，别按某次 run 的 `4/8` 写死。
5. ✅ **已加**（随 P2-8 一起，它才是消费者）verdict 词汇表 ——
   `Stage2RepairController._VERDICT` 把 `f_k_evidence_status` 映射成
   `VALID_PASS` / `INSUFFICIENT_DATA` / `STATISTICALLY_REJECTED`。
   钉住的语义：**`INSUFFICIENT_DATA` 永不当 `STATISTICALLY_REJECTED` 用**
   （前者加预算不换轴，后者立刻换 Epoch），否则退化成"再测一次直到碰巧通过"。
6. ✅ **已加**（同上）停止标准三维 `execution_status` / `evidence_status` /
   `trust_level`；`allow_untrusted_stage_results` **只改 trust_level**。
6a. ✅ **构造上已成立**（插点函数不再拆窗，后置断言钉住"非末窗态数不变"）。注意 model B 下这是**构造上自动成立**的
   （溢出只堆末窗，非末窗永远到不了 `K ≥ 2·lo−1`），所以它只是**防回归**，
   不是修补逻辑的一部分。**补 λ 不受位置限制** —— 任意窗口都可以，前缀照样不动。

3b. ✅ **已改（顺带）** `partition_criterion` 贯穿"定 n / 选边 / 定位"三处。
   拆窗一去掉，旧实现里这个参数就变成完全无效的了（它原来只影响拆窗切点）——
   而"缺 metric_g 就 fail-closed"那条守卫存在的全部意义就是防补救悄悄换判据。

### P2 — 接线收口，零 GPU

7. ✅ **已加** 只读盘的状态视图 —— `abfe_preoptimizer.Stage2RepairController.read()`。
   ⚠️ 一并修掉一个**我自己写出来的 bug**：段的输出目录与 checkpoint 是两套命名
   （`vanishing_2` ↔ `checkpoints/segment_2`），按 stage_name 拼顶层 checkpoints
   会让段 2 读到段 1 的 `ibs_state`。λ 路径版本链与 stage 结果**永远在顶层**
   （路径是全局的），只有 `ibs_state` 分段。
8. ✅ **已加** `decide()` 纯函数 + `render()`，**影子运行、未接线**。
   11 个决策场景有测试（`tests/test_stage2_repair_controller.py`）。
   只读入口：`python abfe_preoptimizer.py <run_dir> [stage_name] [--json]`。
   ⏸ **还没做的是"对账"**：拿历史 run 逐条比对 `decide()` 的判断与当时真实发生的
   动作。这是接线（9）的前置条件，必须先做。
9. ⏸ **未做** 对账一致后换接线 —— 纯删代码：`run_all_windows` 尾部裁决、
   `abfe_pipeline:9919` 异常分诊、`:13027` rescue 循环
9a. ✅ **已加** join λ 两侧支撑，自动 ——
   `ibs_engine.join_lambda_two_sided_support()` 在**每个窗口产物全部持久化之后**
   与前一个窗口比一次，落成 `dual_join_{up}_{down}_{type}_support.json`；
   `Stage2RepairController` 读它并渲染。**只报告、不设门、失败不中断采样。**
   实测（cyclod_ligand2/rep1，join λ=0.5407）：上游 win0 末态 rawESS=13.21/500、
   top1%=0.545（一帧扛掉 54.5% 权重）；下游 win1 首态 rawESS=240.56/500、
   top1%=0.033 ⟹ 18.2×，`downstream_better`。
   两个坑（都实测过）：`energies.npy` 是 (K,N) 且 **Fortran order**、
   `sampling_states.npy` 是 (N,K) C order，**转置方向相反**；两者差一个逐态常数
   （LRC，win0 末态 mean=2.3633 kJ/mol、std=2.8e-16），而 ESS/g 对它**不变**
   （实测差 1e-14）—— 所以用哪个都行，**但不许混用**，落盘记 `gauge`。

9c. ✅ **第一刀已做**（老板拍板："每个窗口跑完就判一次，这不是更早判定吗"）——
   `ibs_engine.window_self_support_check()` 挂在窗口产物持久化之后（join 旁边、在它之前），
   算该窗口**自己**的去相关帧数，落成 `dual_window_{i}_{type}_self_support.json`；
   控制器读它，`decide()` 把不够的窗口列为补采目标（分支 5b = 分支 6 的提前版）。
   **与事后判定逐字同量**：`energies.npy`（分析器读的就是它作 `u_kj_raw`）+
   `production_segments` + 门槛 10（= `solve_stage_integrated` 的
   `min_frames_per_window`）。verdict 用 `INSUFFICIENT_DATA`（≠ FAIL）。
   **只判不动手**：不改 `production_step_overrides`、不碰 `converged`/rescue 判据
   （有静态测试钉着）。
   实测浪费（launch.log 的真实顺序）::

       [窗口 4] ... 生产采样 (250000 步)
       [OK] 所有窗口采样完成              ← 5×250k = 125 万步全部烧完
         [WARN] 窗口 4 去相关后 7 帧 < 10，跳过
         [WARN] 窗口 3 ... 5 帧，跳过
         [WARN] 窗口 2 ... 5 帧，跳过
         rescue round 1/2: 仅追加窗口 [0,1,2,3,4] → 全部 500000

9c-2. ⏸ **第二刀未做：增量前缀 MBAR**（老板："win0+win1 解完了之后，win0+win1 的解就可以
   直接加 win2"）。接在共享节点上（各窗自己 local MBAR、在共享 λ 处拼），不是合并
   重解（那需要跨窗交叉能量，没落盘）。数学上与最后一次性算等价，**价值全在时机**。
   机制已有：`solve_stage_integrated` 接受 `excluded_local_windows`，把还没跑的窗口
   排除掉就是前缀求解。
   ⚠️ **落点不是控制器**：前缀求解是**计算**，而控制器按契约**只读不算**
   （否则"拿历史 run 离线重放决策"就不成立，有 mtime 测试钉着）。所以它应该挂在
   `ibs_engine` 的窗口完成钩子上（join 旁边），由控制器读结果。

9d. ⏸ **未做** **失败归属给刚加进来的那个**（老板："如果加不了，当然是重新继续给
   后面那个人加采样量"）。顺序跑 + 每落一窗重算前缀 ⟹ "加不进去"的那一刻出问题的
   必然是刚加进来的这个，它的 checkpoint 还热。
   ⚠️ 必须分清两种失败，**只有第一种是"加采样"**：
   (a) win_i 自己支撑不够（去相关帧数不足 / local MBAR 解不出）→ 给 win_i 加采样；
   (b) win_i 自己好好的但 **join 两侧对不上** → 两份独立估计都够用却不一致 ⟹
       系统性问题（f_k 口径 / 能量口径 / 上游端点估计不可信），加采样是白加。
   上一跑的数据支持"通常不是 (b)"：每个 join 都是上游弱下游强，而下游正是刚跑完
   那个 ⟹ 共享节点实际被新窗口兜住。**这是 `decide()` 的决策，别硬编码。**（上游窗口末态 vs 下游窗口首态），作为
   "这个窗口的端点值可否采信"的独立证据。零 GPU，两边 `energies.npy` 都有。
9h. ✅ **已做（探针）** **偏斜判出来就更早重标定，别先加帧**
   （老板："判出偏斜就该更早触发重标定，少算几轮"）。
   `_recalibrate_f_k_and_resample_segment(probe_only=True)` 在**第一次 solve 之后、
   rescue 循环之前**跑一次（只算不采），落成 `stage2_fk_recalibration_probe.json`；
   控制器读它，`decide()` 的分支 **5a** 把"重标定"排在"加帧"（5b/6）**之前**。

   **这不是新设计，是已写明的规则没有执行者**：§3 判别表早写了「`Δf_k − ΔF_MBAR`
   带精度地明确不符 ⟹ **只 recalibrate `f_k`**」；§4 明写「I 与 II 之间**不排序**
   ……不做『先便宜后贵』的阶梯（证据表明 `f_k` 明显不符时先加帧是浪费）」。
   而现在的实现恰恰是 §4 禁止的那个阶梯 —— 配置项名字本身就说明结构：
   `stage2_recalibrate_f_k_on_rescue`，重标定挂在 rescue **之后**。

   **判据不新发明**：用既有的 `stage2_f_k_recalibration_min_shift`（0.5 kJ/mol）比
   `max_adjacent_shift_kJ_mol`。**位移是直接证据，偏斜（top1%）只是症状**，
   而且位移 gauge 无关（两边都减过均值）。⚠️ 这一刀必须用 `sampling_states`
   （逐 λ 态常数会改变 logsumexp 的形状，实测对账 sd：sampling_states=0.0000、
   energies=0.735/0.373/0.182/0.094）—— 与 P2-9c 自检**相反**（那边为对齐分析器
   判跳过而用 `energies`；那里 g/ESS 对常数不变，所以无所谓）。

   **只判不动手**：探针分支里只写本地 `diagnostics`、只调 `self._log`，
   AST 验过没有 `run_once`、不改 `production_step_overrides`、不碰放行判据；
   失败也不阻断 rescue。改执行顺序是 `decide()` 接线（P2-9）的事。

   实测代价（本轮 launch.log:1545 第一次 solve 就量出来了）::

       win0  top1%_raw_weight=0.545  N_decorrelated=40  ESS_ratio=0.037
             ⟹ N_decorrelated 已过重标定的 min_frames=10 门槛，那一刻就有资格重标定

       win0  250k → 1M（4× 采样）后：
             N_decorrelated  40 → 182    变好      ESS_ratio 0.037 → 0.0264 **变差**
             absolute_ESS  1.50 → 4.80   变好      top1%     0.545 → 0.604  **变差**

   首轮 5×250k=1.25M、最终 production 3.00M ⟹ 两轮 rescue 加出 **1.75M 步**，
   其中 win0 那 750k 是**已知无效**的（没换 f_k 之前偏斜不可能靠加帧改善）。
   上一跑手算的 win0 冻结 f_k vs 生产帧自洽 f_k 差 **14 kJ/mol**（相邻偏差 5.24，
   合法通过了 10 的 loose gate，但累计倾斜 26.8）—— 远超 0.5，说明**这个判别在
   第一次 solve 的数据上就能给出明确答案，不需要任何额外采样**。

9i. ✅ **已做** **`N_eff,k / g_k` 作为一等验收量 + 边际 N_eff 断崖早判**。

   **方法论定位（老板拍的，我核过成立）**：能不能恢复 `p_k`，问的是
   `w_kn ∝ e^{-βU_k(x_n)} / q_IBS(x_n)` 有没有足够 support，**不是 IBS mixture
   内部的 component occupancy 漂不漂亮**。判例：win1 段1 `occupancy_collapsed=True`
   但支撑健康 ⟹ occupancy 是**假阳性**。
   ⟹ **occupancy 是 f_k 的训练目标，不是结果的验收判据。**
   而 `N_eff,k` 单独也不够（把帧当独立的），真正的量是 **`N_eff,k / g_k`**：
   win2 的 853/50 ≈ 17、win0 段1 的 27.5/126 ≈ **0.2**。

   **早判用它的导数**：累计 N_eff 一定亚线性（"乒乓球效应"：f_k 从旧 f_k 下采到的
   样本里解出来 ⟹ 带旧偏置倾向 ⟹ 冻结后轨迹沿这个倾向走 ⟹ 样本堆到偏置偏爱
   的地方而非目标态需要支撑的地方）。真正的信号是**每块的边际增量**，实测断崖 30 倍::

       段2 win0  60.4 91.6 89.0 37.8 79.3 70.2 73.3 23.1 **2.3** 3.0   脱轨 0.90
       段1 win0   5.6  5.4  2.7  3.4  3.6  2.2  1.8  0.0    1.6 1.1   脱轨 0.80
       段1 win1  22.7 18.7 30.7 34.6 14.8  9.4 10.8 10.3    7.3 6.1   **未脱轨**
       段2 win1   3.3  3.6  3.2  0.4  1.9  1.9  0.3  2.3    0.3 0.4   脱轨 0.40

   判据「本块边际 < 前半程中位数的 1/4」在这四组上**逐条复现**同僚的手算结果。
   落在 `ibs_engine.window_self_support_check`（跟 P2-9c 同一处，g 那一半本来就有）。

   **早判比固定节奏更准**：脱轨点跨 **0.40~1.80 ns（4.5 倍）** —— 固定 1 ns 对
   段2 win1 太晚（0.4 ns 就该换，后面 1.6 ns 里 80% 是废的）、对段2 win0 偏早。
   ⟹ **谁先到用谁**：固定节奏当上限兜底，边际断崖当提前触发。

9j. ⚠️ **位移触发器作废（我先前记错了，在此更正）**。
   PLAN 早先版本按同僚建议写过「位移才是直接证据，偏斜只是症状」—— **反了**。
   两个独立实测反例：
   · **win1**：段1 支撑健康（N_eff=165/1000、top1%=0.123），只因位移超 0.5 被拖去
     重标定 ⟹ 段2 同帧同步数变成 N_eff=17.6、top1%=0.662。**纯亏**，而且管线采纳
     段2 之后进最终结果的是更差那份。（第三个独立证据：段1 win1 的边际序列**一直
     没脱轨**，说明它的 f_k 到最后都还有效 ⟹ 更不该被重标定。）
   · **win2**：段1 是全场最健康的（N_eff=853/2000、top1%=0.032、逐态 rawESS
     [1051,1370,1230,853] 全平）；位移超阈值把它拖进段2，新 f_k 要过 200 帧 warmup
     门、g=67 凑不出去相关样本，烧光最后 180k（0.36 ns）**一次都没测出来** → 崩。
   根因是两个错叠加：**(a) 误报的触发器**（0.5 对 50~80 kJ/mol 的 f_k 量程等于谁都超）
   **+ (b) 优化错目标的动作**（`recalibrate_f_k_from_production` 的目标函数是**占据平坦**，
   而占据平坦不是 `p_k` 可恢复性的判据 —— 对 win1 它**用支撑换占据**）。
   ⟹ 探针判据已换成「脱轨早判 ∨ 固定节奏兜底」，位移降级为**仅报告**。

9k. ⏸ **未做，等老板明确点头** —— **重标定之后跳过 200 帧 warmup 门**。
   理由（同僚的，记录备查）：新 f_k 是从 2000 帧**生产**数据解出来的，却要用
   5 批 × 40 = 200 帧 **warmup** 样本去"验证"它 —— 验的证据比产生它的证据弱一个
   数量级，而这正是 win2 的死因。成本账：重解 ~0 步、burn-in 5000 步（0.01 ns），
   **warmup 门是唯一贵的那项**；去掉它，每 1 ns 重锚的开销只剩 1%。
   数据不丢：多采样段分析本来就在（每段带自己的 f_k，MBAR 天生处理多个采样分布）。
   ⚠️ **同僚已跟老板确认过：他只确认了分析，没批这条行为改动。在批准之前不许写。**
   ⚠️ 并且：**固定节奏重锚（9i）单独上线可能比现状更糟** —— 每次重锚都要付这道门，
   而那正是 win2 的死因。所以 9i 目前只做到**探针（只判不动手）**，生产触发器未改。

9l. ✅ **已做（老板接线顺序第 1 步 A/H）** —— **`N_eff/g` 成为验收真源**。

   四种量的分工写死在 `window_self_support_check`：

   | 量 | 职责 |
   |---|---|
   | `n_decorr` | 求解器**是否有资格尝试** |
   | `N_eff,k / g_k` | 目标态能否被恢复 —— **主验收量** |
   | `top1% weight` | 重尾 / 单帧支配的**否决警报**（阈值复用既有 `TARGET_SUPPORT_MAX_TOP1PCT_WEIGHT=0.35`，不新发明） |
   | occupancy、coverage/mixture ESS | **只诊断 f_k 训练**，不决定 ΔG 是否有效 |

   协议安全下限（老板定）：`<1` HARD_INSUFFICIENT / `1–10` INSUFFICIENT_DATA /
   `>=10` ANALYSIS_ELIGIBLE。
   ⚠️ **`>=10` 只是"可以进入分析"，不是 `VALID_PASS`** —— 最终 PASS 还要
   endpoint CI、block 稳定性、全路径完整性。
   ⚠️ **低支撑永远不叫 FAIL**，它是「尚不可测」。只有支撑充分之后结果明确违反
   统计门才是 FAIL。（此前这里用 `n_decorr >= 10` 判 `VALID_PASS`，正是把
   "有资格尝试"当成了"能被恢复"。）

9m. ✅ **已做（第 2 步 I）** —— **缺窗 / 救援耗尽落成 `INSUFFICIENT_DATA` / `HALT_BUDGET`**。
   · 缺窗：reason 明写「缺窗口的和不是 ΔG，禁止当结果使用（不是「精度差一点」，
     是**另一个量**）」。
   · 救援跑过仍有窗口被踢出 ⟹ rescue 只在"converged 或轮数用完"时退出，所以
     「有 rescue 目标 **且** 仍有 skipped_windows」= 轮数用完仍不够 ⟹
     `HALT_BUDGET` + `evidence_status=INSUFFICIENT_DATA`，**禁止静默缺窗后端出部分和**。
   · `_evidence_status` 对这两种情况返回 `INSUFFICIENT_DATA` 而不是含糊的
     `INCONCLUSIVE`（后者容易被读成"结果差一点"）。

9n. ⚠️ **固定节奏重锚作废（老板纠正机制解释）**。
   原解释「冻结 f_k 越采越偏（乒乓球效应）」被老板更正为：**新样本暴露了先前没
   见到的重尾或 basin，累计 ESS 因而下降；不是固定 f_k 自己随时间漂移。**
   ⟹ **不要机械地每 1 ns 重锚，否则容易自适应过拟合。** 我此前只做到探针层、
   未接生产触发器，所以无需回退。新触发规格见 9o。

9o. ⏸ **未做（第 3 步 B/D/E/G）** —— 支撑恶化触发 + **候选反事实验收**。
   触发干预（任一）：`min(N_eff/g) < 10` ∨ **连续两个** block 的边际 N_eff <
   前半程中位数的 1/4 ∨ top1% 灾难性集中。
   然后**离线生成候选 f_k，只有候选满足全部条件才真正重标定**：
   在**未参与拟合的 held-out blocks** 上提高最差态的 `N_eff/g`、不显著伤害原本
   健康的状态、Hamiltonian/λ/sampling-energy 口径一致、**优化目标是 target
   support 而不是 occupancy 平坦**。`f_k` 位移**只保留为报告量**。
   （触发判据的"连续两块"已实现在自检里；**候选反事实验收未做**。）

   ⚠️ **「连续两块」与「单块」在实测数据上分歧很大，需要老板确认**::

       窗口        阈值    单块(同僚手算)   连续两块(老板规格)   差异
       段2 win0   19.82        0.90             0.90        一致
       段1 win0    0.90        0.80           **未脱轨**     分歧
       段1 win1    5.67       未脱轨            未脱轨        一致
       段2 win1    0.80        0.40           **0.90**       分歧

   **段2 win1 最要紧**：它的边际是 [3.3,3.6,3.2,**0.4**,1.9,1.9,**0.3**,2.3,**0.3**,**0.4**]
   —— 低于阈值的块是**震荡分布**而不是干净断崖，所以连续两块要到 block 8 才成对。
   而同僚正是用这个窗口论证"早判比固定节奏准"（0.4 ns 就该换、后面 80% 是废的）。
   **老板的规格在这个案例上会晚到 0.90。** 两种口径都已落盘（`derail_block_index`
   = 连续两块 = 触发用；`derail_block_index_single_block_REPORT_ONLY` = 单块 = 仅报告），
   等老板选。

9p. ⏸ **未做，按接线顺序排在后面**：
   · **第 4 步 C**：每窗结束立即决定（旧 verdict 已在 9l 修好，可以接了）
   · **第 5 步 F**：拆双层预算 `global_consumed_budget`（永远继承）/
     `epoch_validation_reservation`（每候选单独预留）；**决定重标定之前必须先确认
     还能给新 Epoch 分配最低验证额度，没钱验证就不启动**，直接 `HALT_BUDGET` +
     `INSUFFICIENT_DATA`（win2 就是"启动之后才发现没预算验"死的）
   · **第 6 步 J**：rescue plan / 目标 / 已执行块 / 证据版本**原子落盘**
   · 补采改 **`+250k` 加法 + 每块复判**（乘法会让检查间隔越来越粗；`×2` 只能作为
     **显式的大批量升级**，不做默认控制律）；**已脱轨就禁止继续加帧**，转重标定/布局诊断
   · **直放开关不转正**：改成 `REANCHORED → burn-in → PROVISIONAL_PRODUCTION`
     的条件协议，第一生产 block 立即用 `N_eff/g + top1% + stationarity` 复核

9b. ⏸ **未做** 判别表里把"端点弱"与"内部弱"分开：端点弱可由邻窗兜底，内部弱不能。

### P0 外的两条（实测中发现，已修，原计划里没有）

x1. ✅ **溢出槽上界守卫**。末窗豁免 `max_states_per_window` 曾被读成"可以无限长"，
   但可拆区间只有 `[2lo−1, 2hi−1]`；插过上界之后末窗**永远拆不开**，而拆它是唯一的
   收尾动作 ⟹ 死胡同且无人报错。`insert_lambda_in_failed_ibs_window` 与
   `feasible_repair_actions` 两处已 fail-closed。详见 §2「可行性规则」。

x2. ✅ **warmup 账本跨 checkpoint 子命名空间继承**（`inherit_warmup_ledger_across_segments`）。
   实测漏洞：多采样段有自己的 `checkpoints/segment_N/`，段内 `ibs_state` 从零开始 ⟹
   **每开一个段白送一整份完整额度**（window 4 跨段实烧 540k，段 2 的账本却报"剩 490k"）。
   与 `migrate_warmup_budget_ledger`「绝不默认已耗 0 步再赠送一整份额度」及
   `frozen_candidate_fingerprint`「总消耗永远不清零」两处约定直接矛盾。
   只继承三个消耗桶，**不**继承 f_k / 冻结候选 / 候选的验证进度（新段是新 Epoch）。
   锚点用 `path_current.json`，因此同时覆盖 rescue 的两层命名空间
   `checkpoints/vanishing_rescue/{plan_id}/`。
   ⚠️ **副作用是硬停更容易**：以前那个"剩 490k"是假的；烧得最多的窗口现在预算最紧
   （win4 只剩 80k，冻结验证阶梯首档 50k 刚够）。逃生口是**显式升档**，不是关掉继承。

### P3 — 需要决定才能做

9e. **warmup 剖面 → 生产预算的映射**（老板："哪个窗弱，就可以提前增加采样了"）。
   ✅ 剖面已提成**一等证据**并落盘 + 在影子里按三个口径**排序**报出
   （`Stage2RepairController._warmup_weakness_ranking`）。
   ⏸ **映射未定，不许拍阈值** —— 五个点不足以定"coverage_ess < X 就给 N× 预算"。
   已实现的是"只排序、不打分、不加权合成"（合成等于凭空发明权重）。

   上一跑（rep1_evidence）的支持性证据：两个 `occupancy_collapsed=True` 的窗口
   正是 coverage_ess 最低的两个；**win3 拿到 2× 预算活了、win0 拿默认 1× 死了**
   （去相关后剩 6 帧）。而 win3/win4 的 2×/4× 是上一轮 rescue **事后**补的，
   不是按 warmup 信号**提前**给的 —— 所以 win0 从来没进过那个名单。

   ⚠️ **本轮 run 的实测排序与上一跑不同，这是个可检验的预测**（2026-09-11 实测）::

       win     g  n_used  minESS  max_adj
        0   6.85     30    4.49    2.387
        1   4.12     49   15.14    1.099
        2  52.84     12    4.34    1.439
        3  62.26     10    1.41    3.143      ← 三个口径都最弱
        4   8.04     25   11.88    1.264
       弱→强（三口径名次和，不加权）: [3, 2, 4, 0, 1]

   也就是说本轮"该提前加预算"的是 **win3 / win2，不是 win0**。若跑完是 win0 出问题
   而 win3 无事，那么"warmup 信号预测生产结局"这条假设就被削弱 —— **先看结果，
   别先定公式。**

9f. **"去相关后帧数不足"的增量口径**（老板更正：`剩 6 帧 → 整窗丢掉` ✗、
   `剩 6 帧 → +250K` ✓）。语义是"这个窗口还需要 +250k 步"，**永远不是终态结论**；
   `solve_stage_integrated` 里 `min_frames_per_window` 那句 `continue` 只能是
   "本次求解暂不纳入"，不能是"这个窗口出局"。（P0-2b 让它进 rescue 名单，正是这条。）
   ⏸ **要定的是口径**：老板写的是 **+250k（加法，一个 `n_steps_per_window` 基础单位）**，
   而现在 rescue 走 `production_rescue_growth=2.0` 的**乘法**（250k→500k→1M）。
   乘法给得更多、方向一致，所以本轮不改；但两者不是同一个口径。**别自己改默认值。**

9g. **rescue 轮数用完之后怎么办**（真缺口）。`stage2_production_rescue_rounds` 默认 2
   ⟹ 一个窗口最多被扩两次；两次之后仍不够，现在**还是以"被跳过"收场** —— 又回到
   老板划 ✗ 的那个结局，只是晚两轮。正解是落到 verdict 词汇表的 `INSUFFICIENT_DATA`
   （**≠ FAIL**：样本不够是"还没测够"，不是"测出来不合格"），而不是静默丢窗。

9c. **「窗口跨度要缩到多少才够」的判据**（新增，2026-09-11 接线时暴露）。
   触发路径演化的永远是**窗口级**证据，而布局按热力学长度等分 ⟹ 窗内没有任何边
   超过 `L_target` ⟹ 插点函数的**边级**默认判据算出 `n=0` 并明着报错。
   那个报错本身是对的（边级证据确实不支持插点），但这条路径需要的是窗口级的量。
   **当前接线：调用方传 `n_insert=1`** —— model B 下每插一个点，失败窗口的 λ
   跨度缩掉原来的一条边；每轮缩一条、由外层 `max_insertions` 限轮数。
   这是**不引入未验证阈值的最小选择**，不是最终判据。定下真判据之前别改成公式。

10. `min_states_per_window < 4`？两态 BAR 在 overlap 足够时成立，但**不要全局改成 2**
    —— K 应由分窗器按 overlap/mixing 决定。
    ⚠️ **model B 定案后这条不再阻塞任何东西**：末窗无上限、靠溢出就能到 `K ≥ 2·lo−1`，
    而"窗口太宽"的对症动作是补 λ 不是拆窗。保留为可选优化。
11. `EnsembleSpec` 覆盖到哪一项为止（§6.1 全列还是分批）
10a. **跨窗交叉能量**是否落盘（每帧在邻窗 λ 上的 reduced potential）。不落就永远
   只能做 join 那一个 λ，"全局 MBAR 拼起来"无从谈起；`vanishing` 段连 DCD 都不存，
   事后补算这条路也是堵的。这是 `[[ibs-overlap-arbiter-false-premise]]` 里
   "arbitrary-target evaluability" 那条缺口的具体化。

### P4 — 研究项

12. 三类 failure 的判别器（§3）—— **novelty 主体**
13. I/II → III 的升级必要条件（§4）
14. 最小 bridge 构造：不改既定 endpoints / semantics（§1 Type III）
15. `REPLICA_DISAGREEMENT` / `NONSTATIONARY` 完整判据 —— **本仓库无窗口级 reseed**，
    一次独立重复 = `--reset` + 四个 seed 环境变量 + 全新 `--output`、**重跑整条流水线**。
    代价先评估
16. 全局预算分配（局部预算降级为安全上限）—— 注意 §5 的 A3FE 负结果

**P0+P1 全部零 GPU、互不依赖，可以立刻开始。**
**P0-2a/2b 是当前唯一在真机上持续造成损失的一对**（win0 被整窗丢掉、且永远拿不到
补采），优先级应高于其余 P0。
P3-10 只卡住 `K ≤ 6` 的窗口，不再是唯一的结构性决定（见 §2 更正）。
