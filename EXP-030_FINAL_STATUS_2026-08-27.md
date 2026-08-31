# EXP-030：v31 修复后的完整状态（2026-08-27）

> **当前总体结论（2026-08-28，§8.7）：** 排除window_5后，相同五窗的三次ESS/时间增益为+9.01%、+81.79%、+9.44%，均为正，repeat2提升很大。原完整六窗增益仍为−7.83%、+66.66%、+1.50%，未达到预设可复现增益标准。五窗敏感性结果与原六窗结果并列报告，不互相替代。


> **上一阶段结论（仅 repeat2，现由 §8 更新）** 本次新数据为 repeat2（目录 repeat_1），不是三个 repeat 全部重新运行。入口记录缺口已在本批 12 窗中补齐；candidate 的报告 σ 降低 27.7%，按已识别完整尝试成本计算的协议 ESS/小时提高 66.7%，但共同目标 ΔG 的 z=8.83>2，且既有 coverage 门槛未全过，本轮验收未通过。详细结果见 [重跑分析报告](output/outer_lambda_exp030/rerun_analysis_20260828T061239Z/SUMMARY.md)。
>
> **历史归因更正：** 下文关于“已证明慢弛豫”“方法特异性系统偏置”“更精确”“必须继续某项实验”的旧推断，不作为本次已证实结论或自动执行的待办。低 ESS、组内接近、A/B gate 失败各自都不能单独证明 ΔG 差异的根因。旧表保留供追溯，新旧数据不混用。

> **历史补充（2026-08-27）**：window 1/2/3 的 250k→500k→1M 扩展采样结果见 **§5h**。已测试预算内，臂间 ΔG 差距未稳定收窄，部分组合的 raw ESS 持续为个位数；这支持优先排查重加权与混合瓶颈，但尚不能证明“存在与采样时长无关的覆盖率上限”。§5h 对机制与后续行动的限定优先于下文历史推测。

## 前置阅读

本文档基于两个已记录的发现：
- [`EXP-030_IBS_FK_RESIDUAL_TRAINING_BUG_2026-08-26.md`](EXP-030_IBS_FK_RESIDUAL_TRAINING_BUG_2026-08-26.md) —— f_k 训练信号缺残差项的 bug，**已修复**（`IBS_BIAS_PROTOCOL_VERSION` 30→31）。
- [`EXP-030_FROZEN_SNAPSHOT_TIMING_BUG_2026-08-26.md`](EXP-030_FROZEN_SNAPSHOT_TIMING_BUG_2026-08-26.md) —— `frozen_score.json` 快照时机早于生产入口自我修正；原记录时尚未修复。2026-08-28 当前状态：相关记录链路已修复，本次 repeat2 入口与结束记录核对见 §7。

`EXP-030_AB_COMPARISON_SUMMARY_2026-08-26.md` 里 bug 修复前的"+24.5% 改善"结论已经作废，不要再引用。

## 1. 完成状态

v31 下全部 36 个窗口清空重跑（baseline 也重跑了，虽然理论物理过程不变，但版本号检查不分 arm）。

**35/36 有效**。唯一没过的：`repeat1（output_lrc_fix_repeat01_seed20260905）baseline window_5`，两次独立清空重跑，ESS 覆盖率门槛都没过（第一次 27.9、第二次 23.0，门槛要求 ≥50）。

## 2. window_5 覆盖率没过不是"这个窗口天生难"

细查发现：这个窗口**校准阶段其实完美**——冻结前最后一次 local-MBAR loose gate 验证，4 个态占据 `[0.251, 0.250, 0.2495, 0.2493]` 近乎完美均匀，各态 ESS ratio 都 ≥0.99，`max_adjacent_delta=0.109 kJ/mol`（门槛 10.0，轻松通过，且是一次性通过，没有触发 §"未修复"文档里那个自我修正机制）。

但那次验证只用了 91 帧（短窗口）。完整生产是 500 帧（250k 步）——占据在这段更长的时间尺度上塌缩到 `ess_ratio=0.046`。**这是短验证窗口没抓到、但长生产窗口抓到的慢弛豫现象，不是校准没做好，也不是这个窗口的 λ 位置天生难采**——其他两个 repeat 的 baseline window_5 都正常通过。跟本轮更早发现的"candidate 占据偏斜""baseline/candidate ΔG 分歧"是同一类"250k 步不足以让某些窗口跨过慢自由度"的表现，不是孤立的 bug。

## 3. baseline vs candidate 精度对比（同预算 250k 步，剔除 window_5 数据不全的 repeat1 baseline 单独标注）

| repeat | baseline σ (kJ/mol) | candidate σ (kJ/mol) | gate_converged (baseline/candidate) |
|---|---|---|---|
| repeat1 | 数据不全（window_5 缺）| 0.939 | 未知 / True |
| repeat2 | 2.334 | 1.590 | False / False |
| repeat3 | 1.452 | 0.892 | True / True |

repeat2 两臂 ΔG：baseline 164.963，candidate 174.292，差 **9.3 kJ/mol**（bug 修复前是 20.7 kJ/mol，方向对了、缩小了，但还没缩到跟报出的 σ 匹配的量级——修复解释了大部分、不是全部的分歧，可能还有别的因素或者单纯没采够）。

## 4. 精度随步数变化（截断分析，不花新 GPU 时间；`scripts/exp030_convergence_vs_steps.py`）

在 100k/150k/200k/250k 四个截断点，**12 组对比里 11 组 candidate 更精确**，唯一例外是 repeat1 在 200k 步这一点（baseline 1.701 vs candidate 2.141）。

**留意**：repeat1 baseline 这一整列数字都包含了那个覆盖率没过的 window_5（截断分析脚本本身不做覆盖率门槛检查，只要求数组存在），所以 repeat1 baseline 这一列的可信度比 repeat2/repeat3 baseline 低，结论时要打折扣看。

## 5. 诚实结论

- f_k 训练目标 bug 修复后，candidate 在"同预算下总不确定度更低"这件事上的信号**比修复前更一致**（11/12 而不是任意挑出来的几个点），方向上支持残差偏置有效。
- 但 baseline/candidate 的绝对 ΔG 仍然不完全一致（repeat2 差 9.3 kJ/mol，超出两边报出的 σ），说明这套 250k 步的预算本身还没能让两条臂在所有 repeat 上都收敛到同一个真值——这不是"谁更准"能简单回答的问题，需要用户之前提出的交叉裁决实验（baseline 从 candidate 终态启动、candidate 从 baseline 终态启动）才能真正区分"两个亚稳盆地"还是"方法特异性偏置"。
- `repeat1 baseline window_5` 的覆盖率问题两次重跑都没解决，不建议继续重跑碰运气——这是真实的慢弛豫现象，重跑解决不了，需要更长预算或者专门诊断。

## 5b. σ vs 步数（先按 repeat 拆开，再汇总）

| repeat | arm | 100k | 150k | 200k | 250k |
|---|---|---|---|---|---|
| repeat1 | baseline | 3.233 | 1.901 | 1.701 | 2.095 | ← 含 window_5 数据不全，仅供参考 |
| repeat1 | candidate | 1.230 | 1.366 | 2.141 | 0.939 | |
| repeat2 | baseline | 1.607 | 1.860 | 1.745 | 2.334 | |
| repeat2 | candidate | 1.353 | 1.147 | 0.855 | 1.590 | |
| repeat3 | baseline | 1.635 | 1.416 | 1.398 | 1.452 | |
| repeat3 | candidate | 1.198 | 0.882 | 0.903 | 0.892 | |

**汇总**（baseline 去掉数据不全的 repeat1，n=2 取平均；candidate n=3 取中位数）：

| | 100k | 150k | 200k | 250k |
|---|---|---|---|---|
| baseline (n=2, mean) | 1.62 | 1.64 | 1.57 | 1.89 |
| candidate (n=3, median) | 1.23 | 1.15 | 0.90 | 0.94 |

candidate 在全部 4 个截断点都更精确，且差距随步数增加而扩大（100k 差 0.4，250k 差 0.95）；baseline 250k 步反而比 200k 步更差（1.57→1.89），candidate 持续下降。

## 5c. 最终 ΔG 及组内一致性（协议正式门槛过滤后的有效数据）

| repeat | baseline ΔG (kJ/mol) | candidate ΔG (kJ/mol) |
|---|---|---|
| repeat1 | 缺（覆盖率未过）| 175.24 |
| repeat2 | 164.96 | 174.29 |
| repeat3 | 165.93 | 176.12 |

**关键发现**：baseline 仅有的两个有效 repeat 之间只差 0.97 kJ/mol；candidate 三个 repeat 之间只差 1.82 kJ/mol——**各自组内都高度一致、可重复**。但两条臂之间稳定相差 **~9.8~10.3 kJ/mol**，三次独立重复方向和量级一致。这个模式不太像纯随机的滞后/卡壳（卡壳的话组内应该更乱、repeat 之间应该更分散）——更像"两个方法各自稳定地收敛到了不同的值"，即存在**方法特异性的系统性偏差**，具体是哪种机制（candidate 采样了不同的构象子空间 vs 仍有未发现的系统性偏差）还没有定论，需要用户设计的交叉裁决实验（见下）来区分。

## 5d. 交叉裁决实验（repeat2 window_0，用户设计）

方法：`scripts/exp030_setup_cross_seed.py` 构造两个"假源目录"（跟真实源目录逐字节一致，只把 window_0 的起始 checkpoint 换成对方臂的真实终态），`scripts/exp030_window_state_machine_cross_seed.py`（monkeypatch 跳过冻结 checkpoint 哈希检查，不改协议脚本本身）实际真机跑了两个方向：baseline 从 candidate 终态启动、candidate 从 baseline 终态启动。四组数据（原始 baseline、原始 candidate、交叉 baseline、交叉 candidate）全部落在 `output/outer_lambda_exp030/cross_seed/`，跟真实生产矩阵完全隔离。

| 条件 | f_k 跨度（末态−首态，kJ/mol） | 占据分布（5 个态） |
|---|---|---|
| 原始 baseline（自己的起点） | 85.2 | **严重偏斜**（最后一态占 50.3%）|
| 原始 candidate（自己的起点） | 71.4 | 接近均匀（18~22%）|
| 交叉 baseline（从 candidate 终态启动） | 74.6 | 接近均匀（18~25%）|
| 交叉 candidate（从 baseline 终态启动） | 69.4 | 接近均匀（17~29%）|

**结果不完全落在用户预先列出的 5 种裁决结果任何一个里**，是个变体：四组里三组（原始 candidate、交叉 baseline、交叉 candidate）高度接近彼此（69~75 kJ/mol，占据都偏平），只有"原始 baseline"这一组是异类（85.2 kJ/mol，占据严重偏斜）。

**解读**（推测，未经进一步验证）：不是"两个对称的亚稳盆地"，更像是——baseline 自己默认的那个共享起点本身卡在一个局部陷阱里，只有 baseline 自己（没有残差辅助）困在那出不来；candidate 靠残差偏置能逃出这个陷阱；一旦从"已经逃出来"的构型开始（不管接下来用哪套算法：baseline 也好、candidate 也好），表现就趋同。换句话说：这次实验里 candidate 显得更精确，可能不是因为它本身的采样/重加权方法更好，而是它帮着从这个特定起点的陷阱里逃了出来——baseline 单独跑、没有这个帮助，就卡住了。

**局限**：只测了 repeat2 window_0 一个窗口一次，没有重复验证，不能排除这本身也是随机的（比如这次交叉 baseline 运气好躲过了陷阱）。要坐实这个解读，需要在更多窗口/repeat 上重复这个交叉裁决，以及/或者延长这次交叉 baseline 的采样看它是否会晚一点又掉回偏斜态。

**更新（同日）：补测了 repeat2 剩余 5 个窗口，window_0 的解读被推翻，不是主流模式**——

| window | 原始baseline | 交叉baseline | 原始candidate | 交叉candidate | 判读 |
|---|---|---|---|---|---|
| 0 | 85.2（异常偏斜） | 74.6 | 71.4 | 69.4 | 跟着起点走（baseline 自己默认起点是陷阱） |
| 1 | 30.8 | 30.2 | 27.8 | 27.0 | 四组接近，收敛一致 |
| 2 | 25.4 | 23.8 | 27.5 | 26.4 | **跟着算法走**（baseline 配对 vs candidate 配对，两簇分开）|
| 3 | 14.9 | 15.2 | 21.9 | 18.7 | **跟着算法走**（baseline 紧密~15，candidate 都更高~19-22）|
| 4 | 12.0 | 12.2 | 18.6（偏斜42.5%）| 14.7（不同形状） | 混合/不对称——这次反而是 candidate 更不稳定，跟 window_0 角色相反 |
| 5 | 7.5 | 7.3 | 1.5 | 1.0 | **跟着算法走**（最干净，两边都紧密、分得很开）|

（表内数值为该窗口自己的 f_k 跨度 = 末态 f_k − 首态 f_k，kJ/mol，越大代表这个窗口内部自由能变化越大）

六个窗口里 4 个（window 2/3/5 明确、window 1 收敛一致）显示"跟着算法走"——candidate 和 baseline 之间的差异是**方法本身带来的系统性偏置**，不是随机踩中不同的局部陷阱。只有 window_0 是"起点陷阱"模式；window_4 反而是 candidate 比 baseline 更不稳定，角色跟 window_0 相反。

**上面那段"起点陷阱"的解读作废，不能代表整体。** 真正站得住的结论是：baseline/candidate 之间稳定的系统性差距，大概率是残差偏置方法本身造成的**方法特异性定向偏置**，不是"谁运气好躲开了陷阱"——这对应用户最初裁决框架里的"结果跟着算法走"这一支，不是"两个亚稳盆地"那一支。这个系统性偏置本身是不是"更接近真值"，这次实验没法回答，需要更独立的方式（比如完全不同的采样方法做第三方裁决）才能判断。

## 5e. 更正：上面用 f_k 跨度做裁决观测量是错的，重新用物理目标态 ΔG 分析（同日）

**用户指出的关键问题**：f_k 是采样测度的归一化常数，baseline 和 candidate 按 v31 修复后的设计**本来就该**归一化不同的 Hamiltonian（baseline 是 \(U_k\)，candidate 是 \(U_k+A_kB_\phi\)），f_k 不同是设计如此，不是物理观测量，**不能拿来做交叉裁决**。真正该看的是每个窗口从物理目标态（排除残差）重加权算出来的 ΔG。§5d 那张 f_k 跨度表格的解读因此不成立，重新用正确的量分析。

**方法**：`scripts/exp030_cross_seed_target_dg.py`，对每个窗口单独调用一次 `solve_stage_integrated`（只含这一个窗口），取 `covariance_chain_segments[0]`（真实重加权出来的这个窗口自己的 ΔG）和 `window_overlap_diagnostics[0]`（目标态重加权的真实 overlap，不是采样态 mixture responsibility）。先验证过方法本身没错：六个窗口"原始baseline"target ΔG 加总 = 164.96，跟之前完整链算出来的 164.963 完全对上；"原始candidate"加总 = 174.30，也对上 174.292。

**逐窗口 target ΔG（kJ/mol）**：

| window | 原始baseline | 交叉baseline | 原始candidate | 交叉candidate | 判读 |
|---|---|---|---|---|---|
| 0 | 80.76 | 77.23 | 81.11 | 80.34 | **收敛**，四组接近 |
| 1 | 27.22 | 28.38 | 32.88 | 30.22 | 有点分散，candidate 侧偏高 |
| 2 | 27.32 | 20.78 | 28.54 | 27.66 | 分散，交叉baseline 是异类 |
| 3 | 9.75 | 14.51 | 11.51 | 18.02 | 分散，两个交叉都比对应原始高 |
| 4 | 12.92 | 11.81 | 12.39 | 12.50 | **收敛**，四组接近 |
| 5 | 6.99 | 7.49 | 7.87 | 7.93 | **收敛**，四组接近 |

跟 §5d 那张 f_k 跨度表比，**乱得多**：不是每个窗口都干净地"跟算法走"或"跟起点走"，window 0/4/5 四组基本收敛，window 2/3 比较分散没有清晰模式。

**但六个窗口加总（这才是决定最终结论的量）**：

| | 加总 target ΔG (kJ/mol) |
|---|---|
| 原始 baseline | 164.96 |
| 交叉 baseline（从 candidate 终态启动） | 160.20 |
| 原始 candidate | 174.30 |
| 交叉 candidate（从 baseline 终态启动） | 176.67 |

baseline 家族（160.20、164.96）彼此接近（差 4.8），candidate 家族（174.30、176.67）彼此接近（差 2.4），但两个家族之间还是差 **10~15 kJ/mol**，交叉种子没能抹掉这个差距。**换成正确的观测量之后，链总量级别的"跟着算法走"结论仍然成立**——但单窗口层面产生这个差距的机制比 f_k 跨度那张表暗示的更复杂、更不均匀，不是每个窗口都在贡献同一方向的系统性偏置。

**假精度排查**：window_0 的目标态重加权 overlap（`min_ess_ratio`，来自 `window_overlap_diagnostics`，衡量的是"采样分布 reweight 到物理目标态"的真实重叠，不是采样态 mixture 内部占据）——baseline 两组是 0.37~0.42，candidate 两组只有 0.15。candidate 在这个窗口上向物理目标态重加权的真实重叠明显更差，是一个需要单独盯着的隐患；但这个窗口恰好是 baseline/candidate ΔG 差距最小的一个（80.8 vs 81.1），说明"overlap 差→结果偏"这个因果链在这里没有直接体现，不能简单归因。其余窗口的 overlap 没有类似的系统性 baseline/candidate 差异。

**结论（更正版）**：候选方法造成的 ~10~15 kJ/mol 系统性差距，在换成正确的物理目标观测量之后依然存在、依然不随起点变化——支持"方法特异性偏置"而不是"两个亚稳盆地"，但产生机制目前只能定位到"链总量层面"，还不能定位到具体是哪个/哪几个窗口在贡献，也不能排除是重加权算法本身的某个环节（不是 f_k，是 estimator/reweighting algebra 别处）造成的。这个问题仍然悬而未决。

## 5g. 更正：之前引用的 ESS 是错的量；分两套目标重新查重加权质量（同日）

**用户指出的关键纠错**：§5d/§5e 里提到"window_0 candidate target_ess_ratio=0.15 vs baseline 0.37~0.42"，那个 `min_ess_ratio` 来自 `ibs_engine.py::_ibs_reweighting_quality_diagnostics`（`ibs_engine.py:14609`），是**用不含残差的物理目标能量配 f_k、再去掉逐帧共同因子**算出来的一个代理量，**不是真实的重加权 overlap**。真正字面意义上的"采样分布重加权到物理目标态"的 overlap，是同一个函数返回值里的 `raw_min_ess_ratio`（来自 `mbar.compute_effective_sample_number()` 在增广矩阵上的直接计算，用的是实测 `bias_kj`）——之前一直没用这个量，是我的引用错误。

**同时**，理论上"只固定全局 λ=0/1 两个端点、允许中间路径改变"这个想法数学上成立（两个窗口共享中间态时，中间态自由能在拼接时相消，只要求接口两边是同一个 Hamiltonian）——检查过 repeat2 六个窗口的共享边界（λ、A_k、模型哈希、残差偏移），没发现端点错误。但当前 EXP-030 选的是另一种同样合法的方案（`semantics=sampling_only`，`target_policy=baseline_physical_target_excludes_residual`）：用残差辅助采样，再重加权回不含残差的物理态——这个方案的正确性不取决于窗口首尾对不对，取决于重加权本身的 overlap 好不好。

**方法**：`scripts/exp030_dual_target_reweighting_audit.py`，同一批已有轨迹，分别构造两套目标态能量：
- **物理态**（现在协议实际估计的量）：\(U_k=\text{base}+\text{energies}[k]\)（`energies.npy`，含 LRC）
- **含残差的中间态**（candidate 实际采样的混合分布内部真正用的量）：\(U^*_k=U_k+A_k\cdot(B_\phi-B_{\text{offset}})\)（额外用 `residual_basis.npy` 和 `frozen_score.json` 里的 `A_k`，`B_offset` 协议里恒为 0）

两者都用跟 `exp030_minimal_reference_estimator.py` 相同的单参考重要性采样闭式解重新算 ΔG，并且各自独立算 Kish ESS（\(1/\sum\bar w_n^2\)，标准重要性采样有效样本数，不依赖 `ibs_engine.py` 的任何中间量）。baseline 的 A_k 恒为 0，两套目标对 baseline 完全等价，只对 candidate 有意义。

**发现两类不同性质的问题，之前混在一起没有分开**：

1. **window_0 型**：candidate 原生采样本身没问题，问题出在"往物理态方向重加权"这一步。window_0 最后一态（A_k=0.51，非零）：重加权到物理态 ESS 只有 **3.15**（500 帧里几乎被极少数帧支配），重加权到含残差中间态 ESS 是 **43.24**——差 14 倍。这是具体的"false precision"实锤：candidate 真实采样的是含残差的混合分布，往不含残差的物理态拉回来时权重塌缩。
2. **window_3/4 型**：不管往哪个目标重加权都差。window_3 原始 candidate：物理态 ESS[0]=1.15、ESS[末态]=1.03；含残差中间态 ESS[0]=29.36、ESS[末态]=2.15——两个目标都差，不是重加权方向的问题，是这两个窗口 candidate 自己的原生采样量本身就不够（500 帧里真实有效样本经常只有个位数）。

**结论**：之前笼统的"repeat2 差 9.3、repeat3 差 6.0 kJ/mol"这个链总量差距，背后至少混着两种不同机制——不是单一原因。window_0 这类"重加权塌缩"型可能是有限样本 importance bias 的直接证据（支持"差距会随更长采样收窄"这一支）；window_3/4 这类"原生采样都不够"型说明有些窗口的候选臂在当前预算下从根子上没采够，跟重加权算法无关。这个发现直接支持"先延长 window 1/2/3 采样，看差距是否收窄"这条已经在跑的诊断——而且现在有了更细的预期：如果差距收窄主要发生在 window_0 型窗口、window_3/4 型窗口即使延长采样 ESS 依然低，那就分别指向两种不同的后续动作（前者：采样够了就行；后者：这几个窗口本身候选臂的校准/覆盖机制需要重新看）。

## 5f. Estimator algebra 审计：极简独立参考估计器（同日）

**动机**：§5e 那个 ~10~15 kJ/mol 的差距，会不会是 `ibs_engine.py::solve_stage_integrated` 里那套复杂的 MBAR 自洽求解 + 去相关子采样 + 全局偏移数值稳定化 + 共模因子修正等环节里，某处 estimator/reweighting algebra 本身有 bug，而不是真实的物理/有限样本效应？

**方法**：`scripts/exp030_minimal_reference_estimator.py`，**完全不调用 `solve_stage_integrated`**，只用重要性采样恒等式直接闭式求解，只吃最原始的 `energies.npy`（物理目标态能量，已排除残差）和 `bias.npy`（实测的偏置能量，直接用测量值，不自己重新推导 A_k·B_φ 的符号约定，规避"猜错符号"的风险）：

$$F_{k2}-F_{k1}=-kT\log\Big[\textstyle\sum_n e^{\beta(\mathrm{bias}_n-\mathrm{energies}_{k2,n})}\Big]+kT\log\Big[\textstyle\sum_n e^{\beta(\mathrm{bias}_n-\mathrm{energies}_{k1,n})}\Big]$$

这是单一参考系综下的标准指数平均（Zwanzig 微扰）估计量的闭式解——只有一个真实被采样的分布时，MBAR 自洽方程退化成这个闭式解，不需要迭代；base_n 在同一窗口内对所有态相同，做差时直接抵消，连 base.npy 都不用读。用全部原始帧（不做去相关子采样——那只影响不确定度估计，不影响点估计）。

**结果：极简估计器跟 `solve_stage_integrated` 高度吻合，逐窗口差距 0.01~3.5 kJ/mol，没有暴走式分叉**：

| | solve_stage_integrated 链总量 | 极简估计器链总量 | 差 |
|---|---|---|---|
| 原始 baseline | 164.96 | 159.76 | 5.2 |
| 原始 candidate | 174.30 | 169.05 | 5.25 |
| 交叉 baseline | 160.20 | 157.13 | 3.07 |
| 交叉 candidate | 176.67 | 175.18 | 1.49 |

baseline 家族（159.76、157.13）vs candidate 家族（169.05、175.18）——**极简估计器算出来的差距依然是 ~12~14 kJ/mol，量级不变。**

**同帧核验（更强的版本）**：上面那版用的是全部原始帧 vs `solve_stage_integrated` 内部去相关子采样后的帧，帧集合不同，点估计本身就可能不同，不是纯粹的"估计器算法"对比。加了 `--same-frames-as-solver`，直接调用 `ibs_engine._decorrelate_by_worst_target_state`（`solve_stage_integrated` 内部用的同一个函数）拿到完全相同的去相关帧集合后重算：**逐窗口结果跟 `solve_stage_integrated` 精确一致到小数点后 4 位**（例：原始baseline 加总 164.9631 vs 164.963，原始candidate 174.2918 vs 174.292，交叉baseline 160.2071 vs 160.207，交叉candidate 176.6660 vs 176.666）。

**裁决**：同帧对比零分歧，"TMBAR/reweighting wiring 有实现 bug"这条嫌疑排除——两套完全独立的计算路径，给相同的输入，给出数值精确一致的答案。

**补充一层正交检查（同日）**：上面验证的是"每个窗口自己算得对不对"，没验证"窗口之间有没有接错、漏段或重复计入"。用 repeat2 baseline 真实算出来的链（跟 arm 无关，拼接逻辑是同一段代码，查一份即可）核对 `covariance_chain_segments`：

```
window_0: join=0  end=4   window_1: join=4  end=7   window_2: join=7  end=11
window_3: join=11 end=15  window_4: join=15 end=19  window_5: join=19 end=22
```

每个窗口的 `join_lambda_index` 精确等于上一个窗口的 `end_lambda_index`（衔接正确）；`coverage_diagnostics.covered_lambda_indices` = [0..22] 连续、无缺口、无重复，`n_covered_lambda_indices=23`，跟手算的"6 个窗口共 28 个态、5 个相邻共享边界、28−5=23 个独立 λ 点"精确吻合；六段 ΔG 加总 164.9631，跟 `total_delta_G` 一致。**结论：拼接逻辑没有接错、漏段或重复计入。**baseline/candidate 之间这个持续的系统性差距是**真实存在**的，不是算法实现的产物。剩下能解释这个差距的只有两种、且这次审计没法再进一步区分：(a) 有限样本下 importance overlap 不够产生的真实有限样本偏差，(b) 残差偏置确实系统性改变了采样探索到的构象区域（真实物理效应）。要分开这两个，需要一个完全独立的第三方物理裁决方法（比如 REMD），或者显著加长采样看差距是否随样本量收窄。

## 5h. 扩展采样结果：未见稳定收窄，raw ESS 持续偏低（2026-08-27）

### 实验范围与证据来源

本轮为独立诊断，使用 [扩展采样协议](protocols/EXP-030_extended_sampling_diagnostic_windows123_1M.json)，输出位于 `output/outer_lambda_exp030/extended_sampling/`，不属于原冻结的 3×2×6、250k 步正式生产矩阵。范围为 window 1/2/3、repeat2/repeat3、baseline/candidate，共 12 组；目录 `repeat_1` 对应 repeat2，`repeat_2` 对应 repeat3。

[分析脚本](scripts/exp030_extended_sampling_analysis.py) 对每条轨迹取 250k、500k、1M 步对应的累积前缀，再分别调用 solve_stage_integrated。这些检查点不是三次独立重复；本轮的 250k 前缀也不能默认等同于旧 production/ 中那次独立校准与生产的 250k 结果。

**数值来源**：以下表格记录用户在 2026-08-27 汇报的分析结果，保留到小数点后两位；本次文档更新未重新运行 MBAR，不将这些摘录冒充完整的 36 行复算报告。已独立只读核对全部 12 组 energies.npy 文件头和 production_report.json：均为 2000 帧；协议记录间隔为 500 步，因此足以提供 500/1000/2000 帧的三个前缀，排除了本轮因原始帧不足而被脚本截短、仍标成 1M 的情况。这里核对的是原始帧数，不是去相关后的独立样本数。

### 观测一：物理目标 ΔG 差距没有稳定收窄

定义差距为 candidate − baseline，单位 kJ/mol。

| window | repeat | 250k | 500k | 1M | 本次观察 |
|---|---|---:|---:|---:|---|
| 1 | repeat2 | +1.89 | +3.67 | +2.62 | 非单调；1M 的绝对差距仍大于 250k |
| 2 | repeat2 | +8.11 | +8.55 | +9.21 | 这三个检查点上持续增大 |
| 3 | repeat2 | −1.61 | +2.50 | −1.36 | 符号反转，未形成稳定方向 |

这些是局部窗口结果，不能据此报告新的完整六窗链总 ΔG。当前摘录未附各检查点的 σ 或臂间差值置信区间，不能仅凭上述增减宣称统计显著趋势。它们没有显示预期的稳定收窄，但有限样本估计也不要求沿每条轨迹逐点单调收窄。

### 观测二：部分组合的 raw ESS 仍由极少数样本贡献

以下数值对应分析脚本输出的 raw_abs_ess，读取自 `raw_min_absolute_ess`。

| window | repeat | arm | 250k | 500k | 1M | 本次观察 |
|---|---|---|---:|---:|---:|---|
| 2 | repeat3 | candidate | 1.17 | 1.50 | 1.83 | 名义步数增加 4 倍，该指标仅增至约 1.56 倍，仍小于 2 |
| 2 | repeat2 | baseline | 1.06 | 5.17 | 1.26 | 非单调；1M 时重新降到接近 1 |

**指标口径**：raw 指保留实测采样偏置的完整重要性权重，而不是“使用全部原始帧”。这里是在各前缀内部重新去相关之后，计算各物理目标态的单参考 Kish ESS，再取跨态最小值；它不同于去掉共同因子的 min_ess_ratio，也不等于独立构象数或对全部未访问区域覆盖程度的证明。

两臂都出现低 raw ESS，因此不能仅凭这些结果把问题归结为 candidate 的残差。该现象说明，在这些已观察样本中，至少一个目标态的重加权贡献高度集中；它是质量警报，不能单独证明 ΔG 的偏差大小、方向或所报 σ 必然错误。

### 可以下的结论与不能下的结论

**当前结论**：在已测试的 250k–1M 步预算内，所列窗口没有表现出稳定的臂间 ΔG 收窄；部分组合的完整目标态重加权 ESS 持续为个位数，有效信息增长不稳定。当前方案尚未显示进入可可靠外推采样收益的稳定区间，应优先排查采样分布与目标分布失配、极端权重、慢混合/非平稳性，以及去相关后样本数量的变化。

**尚不能写成**：“已经排除有限采样”“重叠与时长无关”“重加权存在绝对 ESS 的硬上限”或“加长采样永远无效”。理由是：

- Kish ESS 为 `(sum w)^2 / sum(w^2)`。在平稳、足够独立且权重二阶矩有限的条件下，ESS 与样本数之比才会趋近稳定值；低 overlap 通常意味着这个信息率很低，并不自动意味着绝对 ESS 有固定上限。
- 新出现的少数大权重帧可以使经验 ESS 下降，哪怕原始帧数增加；从 250k 到 1M 也不保证已经进入渐近区间。
- 本脚本对每个前缀重新估计相关性并选取子样本，去相关后的帧集合不一定嵌套，独立样本数未必增加 4 倍；跨态最小 ESS 对应的瓶颈态也可能变化。
- 低 overlap、稀有事件与慢构象混合会共同造成有限预算下的偏差，不是与“有限样本效应”互斥的第三类解释。正确估计同一物理目标时，“探索到不同构象区域”也不意味着存在两个不同的真实 ΔG。

因此，更准确的机制表述是：**结果与当前预算下持续的重加权/混合瓶颈相容，使优化采样分布与目标路径的优先级上升；瓶颈是否由固定分布失配、未充分混合、少量极端权重或它们的组合主导，仍未裁决。** 这替代 §5g 中“采样够了就行”“原生采样不足已被坐实”等过强推断，也不沿用 §5f 末尾“只剩两种解释”的排他分类。

Kish ESS 的定义与用途见 [PyMBAR 文档](https://pymbar.readthedocs.io/en/master/mbar.html#pymbar.MBAR.compute_effective_sample_number)；重要性采样所需样本量可能远超直观预期，见 [Chatterjee 与 Diaconis，The sample size required in importance sampling](https://arxiv.org/abs/1511.01437)。这些参考说明诊断的适用边界，不替本轮数据提供未计算的显著性或收敛证明。

### 下一步与实验边界

1. 先补全所有窗口、两臂、两个 repeat 的机器可读前缀报告，包含实际原始帧数、去相关帧数/统计非效率、逐态 raw ESS、最小 ESS 对应的 λ、最大归一化权重/前 1% 权重占比、ΔG 与 σ；同时报告完整权重和共同因子已去除的代理指标，不能混称。
2. 检查同一冻结采样势下的非重叠时间块、极端权重帧及慢自由度；非重叠块仍需考虑时间相关性。对可疑窗口分别重算物理目标与含残差中间目标，比较两种目标的权重质量，并保留 LRC。已有双目标审计无需原样重复，扩展到这批长轨迹即可。
3. 据此选择有限、预先定义的干预实验，例如改善桥接态/采样势、残差强度或混合方式。当前不把“再盲目延长相同步数”作为默认下一步，也不预先保证某个干预必然修复差距。
4. 本轮仅为诊断，不替换旧正式矩阵的失败项，不给出新的全链 PASS、candidate 更准或正式 utility 获益结论；不通过修改门槛追认成功。

## 6. 历史待办（2026-08-27；不代表当前任务）

- `EXP-030_FROZEN_SNAPSHOT_TIMING_BUG` 的根治（目前只是清空重跑绕过）。
- `repeat1 baseline window_5` 的覆盖率问题：要么接受当前 250k 步预算下的局限，要么专门给这一个窗口加长预算重跑一次看是否能跨过慢弛豫。
- 正式走一遍 `scripts/exp030_paired_utility.py`（协议规定的、带全部硬性门槛的官方分析）目前会在 repeat1 baseline 覆盖率这一步直接 STOP，拿不到正式 PASS/NO_GAIN 结论——除非先解决 window_5 或者修改协议本身。

## 7. 2026-08-28：本次重跑的当前结论与 ΔG 差异解释

### 7.1 重跑结果：已经完成验收，不是回到 step0

本次新数据是 repeat2（零起编号目录 `production/repeat_1`），两臂共 12 窗，每窗 250,000 production steps、500 保存帧。repeat1/3 为 2026-08-26 历史数据；没有把它们冒称为本次修复后重跑。

| 指标 | baseline | candidate |
|---|---:|---:|
| 物理目标 ΔG / kJ mol⁻¹ | 156.855229 | 174.411484 |
| 报告 σ / kJ mol⁻¹ | 1.611078 | 1.164597 |
| stage solver converged | False | True |
| strict joint-score coverage | False | False |
| 入口 marker、最终 f_k、frozen spec 相符 | 6/6 | 6/6 |
| 协议 mixture effective samples 总和 | 539.673545 | 972.401007 |
| ITT 秒（含本次已识别失败尝试） | 3928.678911 | 4247.485486 |
| 协议 ESS / 小时 | 494.523683 | 824.168473 |

candidate 报告 σ 减少 **27.7132%**；协议 ESS/小时增加 **66.6591%**。这里的 ESS 是去相关 responsibility ESS 的窗口调和平均后求和，不是物理目标重加权 ESS，也不是真实 ΔG 误差的倒数。

A/B 差 **17.556255 kJ/mol**，合并报告 σ 为 **1.987928 kJ/mol**，按原预注册公式 z=**8.831435**，超过原门槛 2；另有 baseline window_0/window_2、candidate window_4 的 joint-score absolute ESS 未达 50，baseline stage 最大窗口 σ=1.201334 未达 ≤1.0。这些是原有门槛，没有新增或修改。

**本轮结论：生产记录链路修复得到实测核对，指标有改善，方法有效性验收未通过。这个判定本身不回答为什么 ΔG 不同；原因必须另以公式与数据解释，不能用 gate 失败循环论证。**

本次分析使用真实 `solve_stage_integrated` 及同帧参考公式，未运行新模拟，未修改生产代码或冻结协议。六臂 reference 与 solver 总 ΔG 差均 <1e-10 kJ/mol。此项仅核对给定数组上的求解代数，不代表已证明上游采样、能量/力实现或统计收敛全部正确。

产物：[完整报告](output/outer_lambda_exp030/rerun_analysis_20260828T061239Z/SUMMARY.md)、[机器结果](output/outer_lambda_exp030/rerun_analysis_20260828T061239Z/analysis.json)、[失败成本补充](output/outer_lambda_exp030/rerun_analysis_20260828T061239Z/cost_audit.json)。已有 production/final_decision.json 未被本次分析覆盖，不可拿其旧内容冒充新结果。

### 7.2 两层公式：你的全局端点理解没有错

记窗口为 `w`、窗口内目标态为 `k`。令 `B(x)` 为共用 base 能量，`E_k(x)` 为物理相互作用能（代码中的 softcore + LRC），`b_a(x)` 为 arm `a` 实测的采样 bias（Group 1 + Group 4）。当前目标为 `H_k=B+E_k`，采样势为 `H_sample,a=B+b_a`。

**窗口内公式**（实际代码对选中的帧集合 I 求和）：

```text
w[a,k,n] = exp((bias[a,n] - energies[a,k,n]) / kT)
ΔG_hat[a,w] = -kT * log( sum(n∈I) w[a,last,n] / sum(n∈I) w[a,first,n] )
```

对应代码：[`collect_energies`](ibs_engine.py#L6658) 读取实测 bias；[`target_energies`](ibs_engine.py#L6682) 为 softcore+LRC，不含 residual；[`solve_stage_integrated`](ibs_engine.py#L14627) 构造 sampled row=`base+bias`、target rows=`base+energies`；[`minimal_window_span_dg`](scripts/exp030_minimal_reference_estimator.py#L57) 给出同一个闭式比值。

**窗口间公式**：

```text
ΔG_hat[total] = Σ_w ΔG_hat[w]
```

本次相邻窗口仅共享一个边界态，实际 [chain 代码](ibs_engine.py#L14955) 用每窗末态减衔接态再求和。常数 gauge offset 在窗内相减时消掉。对本批相同帧的 reference 求和与 solver 总值 <1e-10，排除了“这批数组在最后链求和时凭空多出 17.556”的解释。

**你说的“只有整条链 λ=0/1 固定，内部态允许变”在数学上成立。** 若使用含残差路径 `H*_k=H_k+R_k`，只要全局两端 `R=0`，且每个共享接口两侧是同一个 Hamiltonian，则

```text
Σ_w [F*(right_w) - F*(left_w)] = F(λ=0) - F(λ=1)
```

中间态相消；不要求每个窗口端点的残差都为零。当前 `sampling_only` 采用的是“所有目标态均为物理态，残差只用于采样”的合法策略。它与你允许保留中间残差的路径不同，但最终全局端点 ΔG 仍应相同；不能把不同中间路径说成必然有不同的真实总 ΔG。

**这批实际 frozen score 中，candidate 的采样残差系数并没有在每窗末端归零：**

| 相邻窗口接口 | 左窗末端 A | 右窗首端 A |
|---|---:|---:|
| window_0 → window_1 | 0.514165984 | 0.514165984 |
| window_1 → window_2 | 0.895221249 | 0.895221249 |
| window_2 → window_3 | 0.981788299 | 0.981788299 |
| window_3 → window_4 | 0.741965054 | 0.741965054 |
| window_4 → window_5 | 0.334225759 | 0.334225759 |

window_0 首端 A=0，window_5 末端 A=0；六窗模型 identity 相同、residual offset 均为 0。不能把“分析目标不含残差”混称为“采样时每个窗口端点被错误清零”。

### 7.3 为什么理论同一个 ΔG，实际却不同

若实际轨迹来自冻结后的目标采样分布 `q_a(x)=exp[-β(B+b_a)]/Z_sample,a`，有：

```text
E[q_a][ exp(β(b_a - E_k)) ] = Z_k / Z_sample,a
E[q_a][w_last] / E[q_a][w_first] = Z_last / Z_first
```

所以在这个积分恒等式里，baseline/candidate 的采样归一化常数相消，结果相同。**代码算的是有限帧的两个指数权重和之比，不是已知的精确积分。** 残差改变采样分布、实际访问的帧以及权重；不同有限轨迹的比值没有逐次相等的保证。对比值再取 log 也不保证有限样本无偏。

这并不等于“已经确定只需多跑”或“已经排除上游 bug”。恒等式以采样测度、能量/bias 记录正确为前提；相同数组上的两个求解器一致，不能反过来证明这些前提。本次能直接检查的是差异出现在哪层、哪些窗口及对帧的敏感性，见下。

### 7.4 本次 17.556 kJ/mol 从哪里相加出来

| window | baseline ΔG | candidate ΔG | candidate − baseline |
|---|---:|---:|---:|
| 0 | 82.009596 | 83.589675 | +1.580079 |
| 1 | 28.602616 | 33.462715 | +4.860100 |
| 2 | 18.551820 | 28.728342 | +10.176522 |
| 3 | 8.382076 | 10.165651 | +1.783575 |
| 4 | 12.071932 | 11.521612 | −0.550320 |
| 5 | 7.237189 | 6.943489 | −0.293700 |
| 总和 | 156.855229 | 174.411484 | **+17.556255** |

window_1 和 window_2 合计 **+15.036622 kJ/mol，占净差约 85.6%**。因此这批差异已经定位到窗口内的权重比估计，不能再泛称“链总量有差，尚不知道来自哪里”。

### 7.5 真实数组的敏感性证据（只读诊断，不替换正式结果）

本节仅用本次 repeat2 的真实 window_2 数组。raw 为全部 500 帧；selected 为真实 `_decorrelate_by_worst_target_state` 返回的同一子集，与正式 solver 相符。没有把 joint-score ESS 的另一套去相关序列拿来代替。

| window_2 指标 | baseline | candidate |
|---|---:|---:|
| raw 帧数 | 500 | 500 |
| solver selected 帧数 | 167 | 32 |
| solver g | 3.004406 | 15.795877 |
| raw ΔG / kJ mol⁻¹ | 19.215323 | 26.418544 |
| selected ΔG / kJ mol⁻¹ | 18.551820 | 28.728342 |
| selected − raw | −0.663502 | +2.309798 |
| selected 首端 / 末端 Kish ESS | 1.367451 / 1.197233 | 3.556062 / 4.225652 |
| selected 最大权重原始帧编号（从 0 开始） | 114 / 114 | 0 / 0 |
| 该帧占首端 / 末端权重 | 85.4136% / 91.3331% | 48.1666% / 34.5443% |
| 同时删去这一帧后的窗口 ΔG | 19.850312 | 28.146316 |
| 同删一帧造成的 ΔG 变化 | **+1.298492** | **−0.582026** |
| 原 solver 报告的窗口 σ | 0.185193 | 0.483128 |

**这里检查的是分子和分母同时删除同一帧后的比值变化。** 若令首、末端归一化权重分别为 `p_first,n`、`p_last,n`，有精确恒等式：

```text
ΔG_without_n − ΔG_all = -kT log[(1-p_last,n)/(1-p_first,n)]
```

因此，不是看到两个端点由同一帧支配就断言 ΔG 有问题；若两端归一化权重完全一样，这个影响会相消。本批 baseline 的两端占比并不一样，实际同删后变化 **1.298492**，约为报告 σ 的 **7.0 倍**。这是敏感性与报告误差尺度的比较，不是新增显著性检验，也不是用删帧结果当真值。

本窗 A/B 差距在 raw500 下为 **7.203221**，在 solver selected 下为 **10.176522**：选帧使这批数据上的差距增加了 **2.973300 kJ/mol**，但原始数据本身已有差距，所以不能把全部问题归咎于去相关选帧。相关采样的点估计并非因为使用全部帧就自动错误；同样，也不能直接取消去相关并把 500 个相关帧当独立样本来缩小 σ。选帧会改变有限样本点估计，不只是改变误差条。

对 raw500 的成对删帧核对也成立：baseline 删除帧 114，窗口 ΔG 从 19.215323 到 20.919243（+1.703921）；candidate 删除帧 352，从 26.418544 到 27.382934（+0.964390）。这说明观察到的敏感性并非仅因 solver 实现某个相减/offset 操作而凭空产生。

可复核数据与实际索引见 [window2_weight_audit.json](output/outer_lambda_exp030/rerun_analysis_20260828T061239Z/window2_weight_audit.json)。这些证据定位了有限轨迹权重比的不稳定，不能单独证明“无限采样也无法修复的 overlap 上限”，也不能单独排除上游能量、力或采样测度实现的问题。

### 7.6 为什么 ESS/时间变好，仍没有证明 ΔG 算得更好

[`exp030_analysis.py::joint_score_diagnostics`](exp030_analysis.py#L34) 读取含残差 sampling-state energies，计算：

```text
p_k(x) = softmax_k( -(sampling_state_energy_k(x)-f_k)/kT )
```

再用这些 responsibility 序列的去相关 ESS 形成效率指标。ΔG 用的却是 §7.2 的 `exp((bias-E_k)/kT)`。前者归一化掉的逐帧共模因子及残差/物理 target 差异，不能一般地从后者的两项“求和之比”中消去。只有两个端点的权重序列成比例等特殊情形，才会完全抵消。

所以本次 **+66.7% 证明的是协议指定的采样态 ESS/时间算术改善，不是 ΔG 真误差降低，也不是物理目标有效信息提高 66.7%**。当前协议的共同目标 gate 正是未让这项辅助改善直接变成 PASS。这里应纠正的是把不同量混称为“更准”的解释，不能靠忽略或修改 gate 解决。

### 7.7 已证实与未证实的边界

- **已证实：**全局两端才为零、内部采样残差系数连续；本批不同 ΔG 已出现在窗口内；主要来自 window_1/2；权重比与选帧存在可量化敏感性；协议效率量不等于物理目标估计质量。
- **未证实：**某一个新的上游 force/energy/bias bug 是 17.556 的唯一根因；某种具体构象机制已被坐实；存在与时间无关的绝对 ESS 上限；某一臂就是真值；延长步数一定修复或永远无效。
- **本轮不做：**新建小体系、改变物理 target、删极端帧追认成功、调整门槛、自动追加 GPU 模拟。上面的删帧仅为同一数据上的数学敏感性诊断，不是修复方案。

**直接回答“为什么 ΔG 不同”：本次差距来自两臂各自有限轨迹上不稳定的窗口内物理态权重比（主要 window_1/2），再被窗口求和累积；不能归因于已被检查排除的“每窗端点清零”或“链求和平白加项”。这定位了发生差异的计算环节与可测数值机制，但尚未唯一裁决造成这种轨迹/权重差异的上游根因。**

### 7.8 补充核对：不能只拆 bias；完整 log-weight 的逐项分解（2026-08-28）

本节针对后续审计中“baseline 已坐实、candidate 尚无法解释”的说法，读取同一批 window_2 数组，补上此前没有展示的物理目标项。未修改轨迹、原有选帧、生产代码或门槛。

#### 定义与证据边界

从记录的 sampling states `S_k`、冻结 `f_k` 定义满强度重建势：

```text
b_model = -kT log Σ_k exp[-(S_k-f_k)/kT]
p_k = exp[-(S_k-f_k)/kT] / Σ_j exp[-(S_j-f_j)/kT]
R_k = A_k · (residual_basis - offset)
L_k = E_k - (S_k-R_k)
D = measured_bias - b_model
```

对已有数组，逐帧恒等式为：

```text
kT log(weight_k) = measured_bias - E_k
                = kT log(p_k) - f_k + R_k - L_k + D
```

这里 `L_k` 按当前 energy ledger 定义为 LRC；`D` 先严格称为“实测 bias 减去满强度重建 IBS 势的差额”。只有实际 bias_scale=1、s_residual=1、CV/rest/力组约定都满足时，才能把 D 完全标为 Group4/WCA。查阅的 ibs_state JSON 没有逐帧保存这两个 scale；代码设置及正常路径支持该前提，但不能说已从每帧记录直接验证。不能先称“无条件精确 WCA”，末尾又承认前提未验证。

重建恒等式与直接 `bias-E` 的最大偏差为 baseline 1.1e-14、candidate 2.7e-14 kJ/mol。因为 D 本来就定义为差额，这验证的是分解的数值一致性，**不是对实际 Group4 力或上游采样正确性的独立证明**。各分解项也并不独立，不能把下面的数值归因当作改变某一势能后动力学结果的因果实验。

#### 完整差值：以各臂全部 500 帧的均值为参照

下表每项是该帧与本臂 raw500 均值之差，单位 kJ/mol；冻结的 f_k 为逐态常数，所以在比较帧间差异时消掉。“首端/末端”均为 window_2 内端点。

| arm / 原始帧 / 端点 | responsibility项 kT·log p | 残差项 R_k | −LRC项 | 差额项 D | 总 Δ(bias−E) |
|---|---:|---:|---:|---:|---:|
| baseline / 114 / 首端 | −2.869210 | 0 | ≈0 | +23.690530 | +20.821321 |
| baseline / 114 / 末端 | +1.124946 | 0 | ≈0 | +23.690530 | +24.815477 |
| candidate / 0 / 首端 | +0.011151 | +2.790345 | ≈0 | +7.156439 | +9.957935 |
| candidate / 0 / 末端 | +0.100674 | +3.060169 | ≈0 | +7.156439 | +10.317282 |

- **baseline：**完成目标能量一项的核对后，差额 D 的确是本次帧114高权重的主导数值贡献。首端的其余项反而部分抵消它。其 WCA 标签仍以以上重建前提为条件。
- **candidate：**不是“每项都非全窗最大，因此无法解释”。D 与残差重加权项共同抬高帧0的 `bias-E` 约 9.96–10.32 kJ/mol，也就是将该帧端点 log-weight 相对本臂 raw500 平均 log-weight 抬高约 3.99–4.14。无需某个单项达到全窗最大，组合后的指数权重也可以很高。
- 本节说的是 candidate **solver 选集中帧0** 的高权重。完整 raw500 中的最大权重帧是 352；不能把两个帧集合混为一谈。
- LRC 的帧间贡献在这批已读取数组上为数值零；这比“双方执行同一代码，所以 LRC 数值必然相同”的推断更具体，但不泛化到未检查的数据。

#### 对前一份回复的明确纠正

1. 只展示 `bias=b_model+D`，没有展示目标 E 的变化，不能直接排除目标能量对 `bias-E` 的贡献；本节已补算。
2. “某项不是最大”“残差在正常范围”不构成该项不能参与高权重形成的证据。candidate 的组合贡献已由上表算出。
3. 静态代码路径审阅“未发现时序/隔离问题”，不能扩大成“所有记录环节 bug 已排除”。相同数组长度也不能单独证明任意来源的数据逐帧对应。
4. 缺逐帧坐标确实限制“哪个几何构型产生该势能值”的归因，但不阻止完成已有能量数组的上述代数分解。数值来源与构型原因是不同层次，不能混在一起让任务停下来。

**当前进展：已进一步定位两条 arm 被检查帧的高权重数值来源；没有新证明整条 17.556 kJ/mol 差距的唯一动力学/代码根因，也不以此宣布方法成功。**

逐帧输出（同时保留 raw500 和 solver selected 两种参照）：[window2_full_logweight_decomposition.json](output/outer_lambda_exp030/rerun_analysis_20260828T061239Z/window2_full_logweight_decomposition.json)。复算脚本位于同目录，只读已有数组，不运行模拟。

### 7.9 对用户所问“约20 kJ/mol从哪里来”的全链数值对账（2026-08-28）

**本节直接针对本次重跑的全链差额 17.556255 kJ/mol，不再用“单帧 WCA 势能很大”“ESS 很低”代替整条链的差额来源。旧数据中约20 kJ/mol的具体数值不与本次新数据混用。**

#### 7.9.1 固定全部原始记录和 solver 选帧，逐项对账

保持每条 arm 各窗实际 solver 选帧集合 I 不变，令 E 为物理能量，b 为实测 bias，R=A·(residual_basis-offset)。定义：

```text
physical_window = -kT log[ Σ exp((b-E_last)/kT) / Σ exp((b-E_first)/kT) ]
auxiliary_window = -kT log[ Σ exp((b-E_last-R_last)/kT) / Σ exp((b-E_first-R_first)/kT) ]
c[w,k] = -kT [ logΣ exp((b-E_k)/kT) - logΣ exp((b-E_k-R_k)/kT) ]
physical_window - auxiliary_window = c[w,last] - c[w,first]
```

auxiliary 是为解释当前估计而构造的同帧辅助量，target 为 E+R，仍保留 LRC；不是另一次模拟，不是 f_k 跨度，也不替换正式 physical 结果。它作为同一全局物理端点差的另一条路径解释，需要共享接口的完整 residual Hamiltonian 相同；算术对账本身不需要该额外前提。

| 全链量 / kJ mol⁻¹ | baseline | candidate |
|---|---:|---:|
| 实际 physical ΔG | 156.855229 | 174.411484 |
| 同帧含残差目标辅助量 | 156.855229 | 159.621689 |
| physical − 辅助量 | 0 | **14.789795** |

因此这笔差额精确拆成：

```text
17.556255
= [159.621689 − 156.855229] + [174.411484 − 159.621689]
= 2.766460 + 14.789795 kJ/mol
```

**本次差额中 14.789795 来自 candidate 在每窗将含残差目标转换为物理目标的估计变化之和，另外 2.766460 已存在于上述辅助量与 baseline 的差中。** 这不是说代码硬加了 14.789795，也不是说删去该校正就修复了算法。这里没有证明辅助量是真值或已通过正式验收。

#### 7.9.2 14.789795 对应哪些共享接口

全局两端 R=0，所以 c 的全局端点值严格为零。有限样本下，左右两个窗口对同一接口的 c 估计不必一致，六窗校正之和可精确改写为五接口校正差之和：

| 接口 | λ_vdw | 左窗末端 c | 右窗首端 c | 左−右 |
|---|---:|---:|---:|---:|
| window_0 → window_1 | 0.745490 | +9.177133 | +8.464438 | +0.712695 |
| window_1 → window_2 | 0.604925 | +14.444237 | +13.597112 | +0.847125 |
| **window_2 → window_3** | **0.456912** | **+14.545031** | **−6.793091** | **+21.338122** |
| window_3 → window_4 | 0.330396 | −16.239271 | −7.104184 | −9.135086 |
| window_4 → window_5 | 0.196214 | −7.295167 | −8.322107 | +1.026940 |
| 合计 | | | | **+14.789795** |

这就是本次“约20”的最主要具体项：**同一个 λ=0.4569123818、A=0.9817882985 的共享接口，在 window_2/3 两侧的物理化校正估计相差 21.338122 kJ/mol**。其他接口部分抵消后剩14.789795，再加辅助量中的2.766460，得到实际17.556255。

这与 §7.4 的“window_1/2 净ΔG差贡献85.6%”不冲突：§7.4 按各窗最终 A/B ΔG 分账，本节按 candidate 的共享接口转换校正分账。不能把这两套分项混加。

#### 7.9.3 为什么该接口的校正一正一负：实际选帧权重

| candidate，共享 λ=0.4569123818 | window_2 末端 | window_3 首端 |
|---|---:|---:|
| solver 帧数 | 32 | 434 |
| 该态 R_k 在选帧中的范围 / kJ mol⁻¹ | −22.236 ～ −12.417 | −21.098 ～ +24.405 |
| 物理态最大权重帧编号 | 0 | **14** |
| 该帧物理态权重 | 34.5443% | **95.2756%** |
| 该帧 R_k / kJ mol⁻¹ | −13.610766 | **+24.404867** |
| 物理态 Kish ESS | 4.225652 | **1.100782** |
| 含残差目标 Kish ESS | 6.951595 | 14.985610 |
| c = F_physical − F_with_residual | +14.545031 | −6.793091 |

window_2 这批选帧的 R 全为负；window_3 则有正 R 的帧14，在去除残差、回到物理目标的权重中占95.3%。**两边转换同一接口目标时，有限帧的权重由不同的 R 区域支配，算出的校正因此一正一负。** 这比只检查 window_2 baseline帧114/candidate帧0更直接解释了全链差额的大项。

恒等式也可写成：

```text
c[w,k] = kT log( Σ_n normalized_physical_weight[w,k,n] * exp(-R[w,k,n]/kT) )
```

它显示 c 由整个残差指数权重平均决定；不是把最大帧的 R 值直接当成 c。window_3 的 +24.405 不能直接等同于 −6.793 的绝对值，不能省略指数平均步骤。

补充核对 raw500：同接口左右 c 为 +10.681744 和 −6.657014，差 **17.338758**；因此这一不一致在未去相关的完整保存帧中也存在，不是选帧才从零制造。raw 全链 gap=17.703667，physical转换校正和=13.879236；不能拿raw与selected跨口径混算。

#### 7.9.4 结论边界与文件

- **已经对上整笔数值账：**17.556255 = 2.766460 + 14.789795；后者五个接口贡献和已逐一核对，最大单项21.338122来自window_2/3共享态。
- **不是额外硬加的20 kJ常数，不是把单帧30.76 kJ采样势冒充全链ΔG差。**
- 该对账只使用实测 bias、物理 E、记录的 residual basis、A及实际选帧，不依赖把 `bias−重建IBS` 无条件叫作 WCA。因此它不受 §7.8 的 WCA归因前提影响。
- 不以辅助量与baseline更接近来追认成功，不把中间目标转换的有限样本差直接宣称新的代码bug。
- 跨窗代码核对：残差 factory 不接收局部λ、mean-λ、WCA或窗口参考几何（scripts/exp030_window_state_machine.py:437、scripts/exp027_common.py:239、ibs_engine.py:9961）。窗口参数改变A_k和独立Group4采样势，没有发现显式按窗口改写residual basis的分支。周期盒属于完整微观态；不同轨迹box/坐标不同本身不等于函数不同。未对两个实际Context做同一完整微观态的交叉求值，保留运行时一致性的验证边界。
- 全部 partial-weight 对账、接口校正及帧权重记录见 [chain_gap_accounting.json](output/outer_lambda_exp030/rerun_analysis_20260828T061239Z/chain_gap_accounting.json)。没有更改原始数据、正式目标协议、门槛或生产代码，没有新增模拟。


### 7.10 代码交付：完整双路径分析入口（2026-08-28）

**分工：Codex 完成代码修改、回归测试及已有数据的 CPU 验证；用户决定何时重算。没有启动新的 MD/GPU 模拟。**

#### 修改范围

- `exp030_analysis.py`：新增 E*=E+A·(residual_basis-offset) 目标重建、全链契约检查、真实重要性权重 Kish ESS、共享接口差额恒等式。
- `scripts/exp030_dual_target_reweighting_audit.py --chain`：完整读取同一 repeat 的 baseline/candidate 各六窗；检查冻结/最终 score、production_entry_f_k、最终 f_k、数组 SHA-256、温度、预算、来源和共享态。缺窗/坏哈希/缺 marker 明确失败，不跳窗拼结果。
- **只要求全局 vdw=1/0 两端 A=0；中间窗口首末态允许 A≠0。** 共享态要求 A、offset、完整 phi identity 一致；每窗 f_k 允许独立 gauge。
- 目标从已含 LRC 的 energies.npy 构造，不重加 LRC，不把 sampling_states 当作目标，不改变实测 bias/base。两目标各自把完整 raw arrays 交给原 solver，内部去相关一次。逐段及总量与闭式 reference 对照至 1e-8 kJ/mol，并检查 solver 实际六窗/全局态覆盖及 join/end 索引。
- 旧的无 --chain 调用保留；新 JSON 必须写在 production 输入树外，以独占创建防止覆盖既有文件。不改变采样动力学、冻结物理目标协议、门槛或正式 decision。

补齐的是可重复执行的完整残差路径分析入口。**没有把“物理路径与残差路径估计不同”直接认定为新的生产代码 bug，也没有修改冻结协议来追认 PASS。**

#### 三种帧口径

| JSON 字段 | 用途 | 帧口径 |
|---|---|---|
| solver_results / target_specific_selections | 各目标 ΔG、条件 MBAR σ、真实 Kish ESS | 每个目标按自己的最差目标态序列去相关 |
| same_physical_frames | 精确拆解原物理结果差额 | 两目标固定使用原物理选帧；不附新路径 σ |
| all_raw_frames | 全体保存帧的算术对照 | 不去相关；不能把相关帧当独立样本 |

solver 的 f_k-based mixture proxy 与 converged 保留为 diagnostic，不能替代新目标的真实重要性权重诊断。退出 0 仅表示完整计算和输入/代数校验成功；顶层 official_paired_decision=NOT_EVALUATED_DIAGNOSTIC_TARGET_POLICY 明确表示未执行正式方法验收。跨 Context 同一完整微观态的 residual basis 交叉求值仍未验证，报告明确 runtime_basis_identity_verified=false。

#### 已执行验证

在 yayoigw2 的原 openmm_dev 环境禁用 GPU 后：

1. **56 passed**：analysis、joint_score、protocol、dual_target_reweighting_audit、frozen_snapshot_reconciliation 五个相关测试文件。新增测试只用数组和临时 ledger，不构建分子体系或运行模拟。
2. 最终代码完整读取真实 production/repeat_1（人类编号 repeat2）的 12 个窗口；四组 arm/target 均覆盖六窗、23 个全局态，逐段及总量闭式对照通过。
3. 原物理结果完整复现；baseline A=0，因此两路径 ΔG、σ、选帧完全相同。

| arm | target | ΔG / kJ mol⁻¹ | 条件 MBAR σ / kJ mol⁻¹ |
|---|---|---:|---:|
| baseline | physical | 156.855229 | 1.611078 |
| baseline | residual_path | 156.855229 | 1.611078 |
| candidate | physical | 174.411484 | 1.164597 |
| candidate | residual_path，自身去相关 | 153.918416 | 2.329913 |

candidate 的 **153.918416** 使用新目标自己的选帧，不能与 §7.9 固定原物理选帧的 **159.621689** 混为一数。新目标各窗选帧数为 [205,63,210,143,166,132]，原物理选帧数为 [77,87,32,434,137,129]。同帧对账仍精确成立：

```text
17.55625534796755 = 2.7664603909880725 + 14.789794956979488 kJ/mol
```

这验证了新入口的计算、取帧和差额核算；**不以新 ΔG 更接近 baseline 就宣布方法有效，也不把条件 σ 当真实误差保证。**

最终真实验证产物：[repeat_1.json](output/outer_lambda_exp030/dual_target_code_validation_final_we38_8w4/repeat_1.json)、[console.log](output/outer_lambda_exp030/dual_target_code_validation_final_we38_8w4/console.log)。JSON 含源码及输入哈希、逐目标选帧、ESS 和接口账目。

#### 用户直接重算命令

在 yayoigw2 项目根目录、原 openmm_dev 环境执行：

```bash
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu OPENMM_CPU_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -B scripts/exp030_dual_target_reweighting_audit.py --chain --repeat-index 1
```

命令**只重算已有 repeat2 数据，不重采样、不启动 GPU、不删除旧结果**。自动新建 output/outer_lambda_exp030/dual_target_analysis_<UTC时间>/repeat_1.json 并打印路径。

无需为了本次分析改动重跑 scripts/exp030_run_node2_repeat2.sh。若用户以后另行重跑生产模拟，同一新命令可读取其输出。旧 repeat1/3 若缺 production_entry_f_k，会明确拒绝；不伪造历史 marker。


### 7.11 用户原方案：ESS 随 steps 的变化（2026-08-28 当前 repeat2 复算）

沿用既有 100k/150k/200k/250k 轨迹前缀方案，不另开模拟。原 convergence_vs_steps.py 主要输出 σ/ΔG，本次补齐原采样态 ESS 指标随 steps 的曲线。只使用此次完整重跑的 repeat2（目录 repeat_1），不混旧 repeat1/3 或另一批 1M 扩展轨迹。

ESS 定义与上一条 ESS/小时结果完全一致：每个状态的 responsibility 序列在当前前缀内重新估计自相关，以 ceil(g) 去相关后算 Kish ESS；每窗取各状态 ESS 的调和平均，最后将六窗相加。没有沿用全长 g，也没有把 raw frame 数当 ESS。

横轴是每窗的生产 MD steps，不是秒数；12窗 energy_query_diagnostics.failures 均为0、每窗500帧。生产先积分后查询，每500步一帧，因此四个前缀为200/300/400/500帧，无初始step0帧偏移。

| 每窗生产 steps | baseline 采样态 ESS | candidate 采样态 ESS | candidate/baseline | 同 steps 改善 |
|---:|---:|---:|---:|---:|
| 100000 | 545.223 | 512.317 | 0.939646 | -6.04% |
| 150000 | 582.577 | 703.925 | 1.208294 | +20.83% |
| 200000 | 582.026 | 930.949 | 1.599497 | +59.95% |
| 250000 | 539.674 | 972.401 | 1.801832 | +80.18% |

**本次观察：100k 时 candidate 略低；150k、200k、250k 时分别高20.83%、59.95%、80.18%。这支持当前 repeat2 在后段相同步数下有更高的协议定义采样态 ESS，并非所有前缀都领先。**

candidate 150k 的 ESS=703.925 已高于 baseline 250k 的539.674。这是已观测前缀比较，可以写“本次150k已超过baseline在250k的该项ESS”，不能据此声称所有重复均节省40%算力、精确交叉点为150k、或者同等ΔG精度所需时间减少40%。

250k 的采样态 ESS 比值1.801832，结合已记录 ITT 耗时比1.081149，得到单位时间效率比1.666591，即+66.6591%。两个结果来自同一批数据，分别回答同steps和单位实际时间，不矛盾。没有按总耗时线性缩放来伪造100k/150k/200k的实测秒数。

累计前缀不是四次独立重复。新增帧会改变经验自相关和responsibility分布，因此估计ESS可下降；不能强制曲线单调，也不能把baseline曲线平坦直接当成某个结构机制的证明。采样态ESS和最终ΔG重加权ESS不同：JSON另存physical/residual_path各状态的原始及目标去相关后字面Kish ESS，不混入此主表。

产物：
- [总曲线](output/outer_lambda_exp030/ess_vs_steps_current_repeat2_final_k016aq9y/ess_vs_steps.png)
- [逐窗口曲线](output/outer_lambda_exp030/ess_vs_steps_current_repeat2_final_k016aq9y/ess_vs_steps_by_window.png)
- [CSV](output/outer_lambda_exp030/ess_vs_steps_current_repeat2_final_k016aq9y/ess_vs_steps.csv)
- [全部数据、输入哈希、逐目标ESS](output/outer_lambda_exp030/ess_vs_steps_current_repeat2_final_k016aq9y/ess_vs_steps.json)
- [复现脚本](output/outer_lambda_exp030/ess_vs_steps_current_repeat2_final_k016aq9y/reproduce.py)


## 8. 三个 repeat 完成后的配对结论（2026-08-28）

**直接结论：本轮没有达到用户原定的可复现采样效率提升门槛。** 三个repeat的36窗都已读入，原效率指标已完整计算；但文件完整不等于全部通过生产覆盖检查，详见§8.5。没有新增模拟，也没有改判据。

### 8.1 按原定义计算ESS/ITT时间

| repeat | baseline ESS/小时 | candidate ESS/小时 | 相同steps的ESS增幅 | 实际耗时增加 | 时间效率改善 |
|---|---:|---:|---:|---:|---:|
| 1 | 765.742 | 705.771 | +2.17% | +10.85% | -7.83% |
| 2 | 494.524 | 824.168 | +80.18% | +8.11% | +66.66% |
| 3 | 723.039 | 733.856 | +14.06% | +12.38% | +1.50% |

原预设要求：至少2/3个重复效率为正，并且中位增益≥10%。本轮有2个为正，但中位数仅1.496020%，第二个条件未达到。reduce_paired_decision返回 EXP030_STOP_JOINT_SCORE_NO_REPRODUCIBLE_ITT_GAIN；这里是效率阈值算术结果，没有覆盖正式production/decision文件。

repeat1的ESS仅增2.17%、耗时增10.85%，净效率下降；repeat3的ESS增14.06%、耗时增12.38%，收益大部分抵消。repeat2单次+66.66%的改善确实存在，但不能代表另外两个。

### 8.2 ESS–steps不再只有repeat2

| repeat | 100k C/B | 150k C/B | 200k C/B | 250k C/B |
|---|---:|---:|---:|---:|
| 1 | 0.9246 | 0.9181 | 1.0146 | 1.0217 |
| 2 | 0.9396 | 1.2083 | 1.5995 | 1.8018 |
| 3 | 0.9705 | 0.9381 | 1.2975 | 1.1406 |

repeat1、repeat3的baseline ESS也会随步数增加，不呈现repeat2那条近乎停滞的曲线。此前“这一次缓解采样停滞”的描述只适用于repeat2的指标，不能推广为传统方法总卡住、候选方法不会卡住。累计前缀不是独立重复。

### 8.3 ΔG对照（各目标自己去相关）

C−B差额，kJ/mol：

| repeat | 原physical路径 | 保留中间残差路径 |
|---|---:|---:|
| 1 | +19.801004 | +6.010708 |
| 2 | +17.556255 | -2.936813 |
| 3 | +15.187019 | +1.326702 |

残差中间路径缩小了差距，但repeat1仍相差6.01 kJ/mol。新路径名义z为2.1103、1.0368、0.7893；σ来自条件MBAR，不作真实误差保证，不把新路径混入旧physical协议追认PASS。固定原physical选帧的接口对账另存报告，不能与此处各自去相关结果混算。

### 8.4 完整性、费用与范围

- 36窗均有报告、入口marker与500帧，零查询失败。冻结score/f_k链路、配对起点和输入账本检查通过；所有12组arm/target的全链solver与闭式reference核对通过。
- 相关回归71 passed。源文件跨repeat字节哈希历史差异按用户要求与既有生产规则不判失败，不再追查。
- 成本独立复核一致：36成功attempt+1失败attempt，repeat2 baseline失败158.501949579秒计入；24个已归档旧cohort attempt不计入。
- repeat2旧执行manifest不足以证明一次连续的counterbalanced执行顺序，相关文件时间记录的局限已写入完整报告；不改变本次ITT算术。
- 本轮结论是“在当前体系、预算与指标下未达到预定可复现增益”，不是证明方法在所有条件下永远无效。

完整结果：[SUMMARY.md](output/outer_lambda_exp030/three_repeat_final_analysis_lw5i7uud/SUMMARY.md)；[JSON](output/outer_lambda_exp030/three_repeat_final_analysis_lw5i7uud/analysis.json)；[曲线](output/outer_lambda_exp030/three_repeat_final_analysis_lw5i7uud/ess_vs_steps_three_repeats.png)；[成本审计](output/outer_lambda_exp030/three_repeat_final_analysis_lw5i7uud/cost_audit.json)。


### 8.5 repeat1 专项核查：历史缺窗与本次覆盖未过不是一件事

旧记录中的 repeat1 baseline window_5 两次覆盖未过（ESS 27.9/23.0）及结果缺失，属于历史批次。2026-08-28 完成的新 repeat1 已有 baseline/candidate 各6窗、每窗250000步和500帧；12窗均正常完成、零查询失败、预热验证通过，无 best-effort 放行，事件记录没有失败或重试。

当前未通过原生产 joint-score coverage 的是 **candidate window_5**：

| 原有指标 | baseline window_5 | candidate window_5 | 要求 |
|---|---:|---:|---:|
| 窗内采样态调和平均 ESS | 153.5294 | 20.2701 | ≥50 |
| ESS / 原始帧数 | 0.30706 | 0.04054 | ≥0.05 |
| 最少去相关帧数 | 125 | 14 | ≥20 |

candidate 的四态平均占据约为0.2584/0.2399/0.2456/0.2561，接近平坦，但其 responsibility 序列估计自相关因子分别为25.44/4.71/25.49/36.78；原分析按 ceil(g) 取帧，得到20/100/20/14帧。平坦占据不代表时间上独立。这解释本次 coverage 数字怎样产生，不等于已证明造成相关性的动力学机制或新的代码 bug。

完整运行、冻结数据链路核验和 solver/reference 对账通过，与生产覆盖不足可以同时成立。**当前 repeat1 应标注为“执行完整，生产覆盖未达标”；不能仅凭 −7.83% 宣称它是一个所有检查均通过的负效果试验。** −7.83% 的 ESS/ITT 计算仍是该次实际表现，coverage 失败也不能成为删去该次、只挑正增益重复的理由。

三个重复的中位数+1.50%是已计算的效率算术，未达到原+10%要求；不将其冒称完整有效性审核通过后的正式方法裁决。此轮准确结论是“尚未取得通过原验收的可复现效率增益证据”，不是证明方法永远无效。可将换体系作为独立实验，沿用事先固定的对照、预算、重复及指标，保留本体系所有结果；本次不修改体系、不启动模拟。

[逐窗证据与原始日志校验](output/outer_lambda_exp030/three_repeat_final_analysis_lw5i7uud/repeat1_focused_audit.json)

### 8.6 排除 window_5 的配对敏感性分析（用户指定）

repeat1 两臂同时排除 window_5，并扣除两边该窗口的ESS及对应成本，保留相同的window_0–4：baseline ESS=728.195、3478.303秒、753.673 ESS/小时；candidate ESS=880.559、3858.552秒、821.554 ESS/小时。

**repeat1 单位时间效率从六窗的−7.83%变为五窗的+9.01%；同steps ESS提高20.92%，耗时增加10.93%。剩余五窗两臂均通过原 joint-score coverage。** 这表明 window_5 明显拖低原repeat1整体指标，其余五窗合计存在正收益。

补充将三次重复都统一保留相同五窗（不混五窗与六窗口径）：时间效率增益依次为+9.01%、+81.79%、+9.44%，中位数+9.44%。repeat2其他窗口的原coverage未过状态仍保留。

以上是事后子集敏感性分析，不替代原六窗正式验收；未删除数据、未重跑，也未将缺少终端窗口的结果当作完整ABFE ΔG。

[详细报告、ESS–steps及复现文件](output/outer_lambda_exp030/three_repeat_final_analysis_lw5i7uud/EXCLUDE_WINDOW5_SENSITIVITY.md)


### 8.7 总体总结（2026-08-28）

#### 核心结论

**排除 window_5、只比较双方相同的五个窗口后，三个 repeat 均观察到采样效率正收益：相同步数下的采样态 ESS 提高约21%、104%、23%；计入实际成本后，单位时间效率提高约9%、82%、9%。repeat2提升很大，repeat1/3提升较温和。**

因此，不能把本轮结果概括成“方法没有收益”；同样，不能把repeat2的大幅收益概括成所有重复、完整六窗都大幅改善。当前证据支持的是：所分析五窗的采样态信息产出有一致正向表现，收益幅度随重复而变化，window_5会明显影响整体表现。

#### 1. 修复与核验已经完成的部分

此前已定位的 f_k 训练残差缺项，以及生产入口marker初始化、正确时点持久化、生产结束Context/f_k回写和load恢复链路已修复。当前36窗均有500帧、每窗250000生产steps，零能量查询失败；冻结记录和能量账本核验通过，12组arm/target全链solver与独立闭式reference对账通过。相关回归此前为71 passed。

这些证据支持已检查的实现链路和数值对账正确，不等于保证没有其他bug，也不保证任意有限轨迹都具有充分覆盖。

#### 2. 五窗对照：明确观察到的收益

两臂同时排除window_5的ESS和对应窗口成本，只统计window_0–4；三个repeat都使用相同窗口集合。

| repeat | 同steps ESS增幅 | 对应成本增加 | ESS/时间增幅 |
|---|---:|---:|---:|
| 1 | +20.92% | +10.93% | +9.01% |
| 2 | +103.81% | +12.11% | +81.79% |
| 3 | +23.24% | +12.61% | +9.44% |

三次五窗时间效率均为正，中位数为+9.44%。这里的ESS是原先定义的采样态responsibility ESS：逐状态去相关、每窗调和平均、窗口间相加；不是最终ΔG估计器ESS，也不是ΔG准确度。

当前repeat1唯一未过原生产joint-score coverage的窗口是candidate window_5：ESS=20.27、最少去相关帧数14；baseline同窗ESS=153.53。repeat1两边扣除该窗后，时间效率从−7.83%变为+9.01%，剩余五窗两臂覆盖均通过。repeat3剩余五窗也通过；repeat2其他窗口原有覆盖未过状态仍保留。

#### 3. 完整六窗的结果也保留

| repeat | 完整六窗ESS/时间增幅 |
|---|---:|
| 1 | -7.83% |
| 2 | +66.66% |
| 3 | +1.50% |

原六窗中位增益+1.50%，未达到事先要求的+10%，并且部分生产覆盖检查未过；不能追认为完整协议PASS。五窗分析是用户指定的事后敏感性分析，说明收益分布，不能替代完整六窗验收。所有原始数据和失败记录均保留。

#### 4. ΔG问题与采样效率分开表述

完整六窗、每个目标自行去相关的A/B差额如下（kJ/mol）：

| repeat | 原physical目标C−B | 保留中间残差目标C−B |
|---|---:|---:|
| 1 | +19.801 | +6.011 |
| 2 | +17.556 | -2.937 |
| 3 | +15.187 | +1.327 |

保留中间残差路径后，三次差距均缩小；但这不是已证明哪个估计值等于真值，也不能单凭接近baseline宣称准确。排除window_5后本次只比较采样态ESS与成本，没有将缺少终端窗口的结果当作完整ABFE ΔG。

#### 可直接引用的总体表述

> 在当前体系的配对实验中，统一排除window_5的敏感性分析显示，候选方法在三个重复的其余五窗中均提高了采样态ESS/实际时间，增幅分别为9.01%、81.79%和9.44%；相同步数的ESS增幅分别为20.92%、103.81%和23.24%。这为该窗口子集中的采样效率收益提供了正向证据，其中一个重复收益很大，另外两个约为9%。完整六窗的收益尚不稳定，原协议验收未通过；因此目前不能宣称完整ABFE流程已实现稳定的大幅加速或更高自由能准确度。

本次仅整理既有数据和计算结果，没有启动模拟、改预算、改门槛或删除结果。

[完整六窗分析](output/outer_lambda_exp030/three_repeat_final_analysis_lw5i7uud/SUMMARY.md) · [五窗敏感性分析](output/outer_lambda_exp030/three_repeat_final_analysis_lw5i7uud/EXCLUDE_WINDOW5_SENSITIVITY.md) · [数值与复现脚本](output/outer_lambda_exp030/three_repeat_final_analysis_lw5i7uud/exclude_window5_sensitivity.json)
