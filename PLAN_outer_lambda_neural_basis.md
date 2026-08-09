# 外层 λ 神经基势研发计划

## 1. 文档定位

本文定义“外层 λ 神经基势”（Outer-λ Neural Basis Hamiltonian）的研究与验证计划。
它是一份方法开发路线图。文中原则与验收门是规范性要求；实际完成状态以第 2.3 节和
实验日志为准，不能仅凭本文中的设计描述宣称 production 已实现。

核心原则：

- 固定参数的 MM/DEXP 或既有 softcore 路径负责定义物理端点。
- 神经势只改变非物理中间态，不替代端点 Hamiltonian。
- MACE/神经基势本身不接收连续 λ。
- λ 只由 Hamiltonian 外层的解析包络和插值系数控制。
- IBS 必须对包含神经项的完整目标中间 Hamiltonian 进行采样和自由能估计。
- production 前冻结全部神经模型和外层参数，production 中禁止在线训练。

---

## 2. 当前项目基线与前置决策

### 2.1 当前事实

当前已有生产产物使用：

- `mode = ibs`
- `decoupling = dual_lambda`
- `potential = softcore`
- `dexp_params = null`

代码虽然支持 DEXP，但当前结果并不是 DEXP 结果。现有 MACE/OpenMM-ML 功能主要用于
Boresch 锚点或口袋力估计，尚未进入 ABFE production Hamiltonian。

### 2.2 必须先做出的路线选择

在开发神经路径前，只选择下面一条基线，不得同时改变基础势和神经路径：

#### 路线 A：先在 softcore/IBS 基线上验证方法

优点：

- 复用当前已经跑通的生产主链。
- 更容易把差异归因于神经路径。
- 适合优先验证端点、能量账本、力稳定性和统计效率。

#### 路线 B：以 DEXP 为论文主线

前置条件：

- 先获得无神经项的 DEXP complex/solvent ABFE 基线。
- 验证 DEXP 端点、循环闭合、长程尾项处理和独立重复。
- 确认 DEXP 本身的误差和稳定性后，再加入神经路径。

推荐顺序：先用路线 A 完成方法学可证伪验证；若研究主题必须是 DEXP，再把通过验证的
神经路径框架迁移到路线 B。

### 2.3 截至 2026-08-03 的冻结状态

当前采用路线 A，且保持 production 隔离：

| 项目 | 已冻结事实 |
|---|---|
| 基础路径 | `softcore + dual_lambda + IBS` |
| 困难窗口 | complex vanishing Stage 2 window 0，states `[0,5)` |
| primary slow variable | ligand periodic torsion `[4591,4592,4593,4585]` |
| secondary candidate | 相邻 torsion `[4593,4585,4594,4595]` |
| 诊断变量 | `VAL251 chi1` 与 ligand hydration coordination |
| 端点包络 | \(w(\lambda)=\sin^2(\pi\lambda)\) |
| 已通过完整 MACE 幅度 | `coefficient=0.09`；EXP-007 NVT qualification 通过 |
| 完整 MACE 生产路线 | 停止；PythonForce/CUDA MTS 在 EXP-009 后端资格失败 |
| 当前路线 | EXP-012：CV-free 通用局部残差路径势；frozen-MACE latent 是主要候选表示，XED 仅作可选消融 |
| EXP-011 冻结状态 | `FORMAL_RUN1_OVERLAP_FAILED`；AUG-001 后 mutual overlap `0.02353 < 0.03` 且 22 个去相关样本 `< 25`，不再补采、不拟合 PMF |
| EXP-012 状态 | `SEALED`；preregistration 已 reseal（DEC-039）。`LocalResidualStudent` D1（held-out gap variance 改善，DEC-040）、D2（27/27 坐标/autograd 检查通过，DEC-040）已关闭；System 身份门已关闭（DEC-041）。D3 已于 2026-08-07（DEC-043）判定 `CLOSED`：cell-list 等价性单元测试通过；`.pow(2)`/`.square()` 假设已用真实数据（`student_d3_1_2_report_v3.json`）证伪（对残差无可测量影响），真正根因未查明、裁定不再追查；`reference_eager_vs_deployable_eager` 残差 `1.69e-7` 裁定为 `PASSED_OPERATIONAL_NUMERICAL_EQUIVALENCE`（比已接受的 CPU64→CPU32/CPU32↔CUDA32 误差还小 2 个数量级，`1e-8` 绝对阈值未进入 sealed preregistration）；sub-item 4 的 `~95%` overhead 按 §16 降级为非阻塞工程目标，如实保留。D4（短 NVT 动力学资格）已于同日（DEC-044）首次执行即通过：student TorchForce 真实注入 `hard_window0` win_sys、3 个独立种子重复实际积分，全程有限值、student 力/温度均在 sanity 门内。EXP-012 D0-D4 全部关闭，详见 EXPERIMENT_LOG DEC-043/DEC-044；WP-5A pilot 已由 DEC-048/050 判定 `NOT_PROMOTED`，当前执行方向以 §2.5 DEC-056 为准 |
| EXP-010 教师局部选择 | ligand 41 + protein atoms 216；事后审计显示 26 个涉及残基全部是不完整截断，仅作失败证据 |
| cheap-CV 候选 | EXP-010 的 1D order 2/4/6、2D order 2/3/4 全部不晋级；当前无 production 候选 |
| production 接入 | 尚未进行；生产模块明确不导入独立研发模块 |

WP-0、WP-1、WP-2、WP-3 通用部署骨架和 EXP-007 已完成。EXP-010 的正式 GPU
教师数据集保留 290/300 帧并通过支持域门，但六个预注册 Fourier 候选的
leave-one-run-out 能量 RMSE 均未优于 intercept-only，广义力验证也失败。因此
EXP-010 记为 `FAILED`，不选择最终模型，WP-5 三臂 IBS 不启动。
该失败不能归因于 torsion bias 本身：EXP-010 的固定 environment 由单帧
0.5 nm 逐原子半径选择构建，存在普遍的残基和共价边界截断；同时它要求
低维 torsion 势逐帧重现高维瞬时 interaction energy。两者都会破坏跨 run 可迁移性。

对“冻结”的限定：

- primary torsion 已冻结为 EXP-010/EXP-011 的候选慢坐标，但仍无 production approval；
- `coefficient=0.09` 是完整 MACE EXP-007 的合格幅度，不因蒸馏自动成为 cheap-CV
  的 production approval；
- 所有 slow-variable manifest 均明确 `production_approval=false`；
- EXP-010 的 cheap-CV 未通过能量/广义力验证，因此不能进入 OpenMM 力学资格或
  WP-5 三臂实验。

### 2.4 2026-08-03 路线收敛：CV-free 通用局部残差路径势

EXP-012 的研究对象不再是 XED 特征本身，而是一个不依赖人工慢变量的通用路径残差：

\[
\boxed{
H_k^*(\mathbf R)=H_k^{\rm MM}(\mathbf R)+A_kB_\theta^{\rm local}(\mathbf R),
\qquad A_0=A_K=0
}
\]

完整 MM/softcore/PME 始终负责基础势、长程作用和物理端点；\(B_\theta^{\rm local}\)
只学习对相邻态 reduced-energy gap 方差贡献最大的局部坐标相关残差。模型不接收连续 λ，
不要求事先指定 torsion、hydration 或其它慢 CV，也不计算 fragment total energy。

第一候选表示是 frozen-MACE 中间层原子描述符：

\[
\mathbf z_i^{\rm MACE}(\mathbf R)
=\operatorname{FrozenMACEEncoder}(\mathbf R)_i,
\]

\[
B_\theta^{\rm local}(\mathbf R)
=B_{\max}\tanh\!\left[
\frac{\sum_{i\in L}g_i^{\rm contact}
\operatorname{MLP}_\theta(\mathbf z_i^{\rm MACE})-b_0}{B_{\max}}
\right].
\]

这里 MACE latent 是学习到的局部几何/化学表示，不能宣称为真实电子密度、孤对电子或
Pauli/exchange 能量。它也不会自动解决局部图边界：必须冻结 ligand/environment 原子身份、
cutoff、message-passing 图闭包、PBC、元素/电荷支持域和 ligand-only readout。两层、6 Å
cutoff 的节点支持域必须定义为图上的两跳闭包
`S0=L, S1=N_6Å(S0), S2=N_6Å(S1)`；12 Å 仅是最远几何上界，禁止用 ligand-centered
12 Å 球替代两跳闭包。
不再复用 `E(complex)-E(ligand)-E(environment)` subtraction。

必须区分两种执行路线：

- **L1：在线 frozen-MACE encoder + MLP readout。** encoder 保留在坐标计算图内，
  每步通过 autograd 产生守恒力；表示最强，但仍承担 MACE forward/backward 成本。
- **L2：MACE latent 离线 teacher + 轻量可导 student。** 用 latent/gap 诊断指导一个
  小型局部等变 encoder；production 不运行完整 MACE，优先争取 ESS/GPU-hour。

  **现状（DEC-030，2026-08-04，见 EXPERIMENT_LOG）**：当前路线正式定性为 L2。
  `original_6a`/`derived_5a` 是离线 teacher，`LocalResidualStudent`（尚不存在代码）
  是唯一在线模型。L1 定义本身不删除，但降级为非当前下一步——原因见下方对
  line 190 要求的回应，不是绕开它。

`get_descriptors()` 式 NumPy 输出只能用于离线诊断。若在 production 中把 descriptor 当作
预计算常数，就无法对新坐标产生正确的 \(-\nabla_{\mathbf R}B\)。在线路线必须直接包装
Torch MACE forward，并保留 ligand、environment 和图构造所涉及坐标的梯度。

EXP-012 的表示消融固定为：

1. A：typed atom-centered RBF/contact baseline；
2. B：轻量等变 ligand-environment cross encoder；
3. C：frozen-MACE latent + 小型 invariant scalar readout；
4. D：XED-inspired field，仅作为可选物理启发消融，不作为主路线。

训练目标不再拟合 PMF，而是最小化相邻态双向 gap variance。若
\(\Delta u^0_{ak}=\beta(H_{k+1}^{\rm MM}-H_k^{\rm MM})\)，定义：

\[
\delta_{ak}(\theta)=\Delta u^0_{ak}
+\beta(A_{k+1}-A_k)B_\theta^{\rm local}(\mathbf R_a),
\]

\[
\mathcal L_{\rm gap}
=\sum_k\frac12\left[
\operatorname{Var}_{p_k}(\delta_k)
+\operatorname{Var}_{p_{k+1}}(\delta_k)
\right]
+\lambda_E\langle B_\theta^2\rangle
+\lambda_F\langle\|\nabla B_\theta\|^2\rangle.
\]

第一轮冻结全局 \(A_k\)，只训练表示和 readout；禁止同时学习 \(A_k\) 造成尺度简并。
数据按整条独立 run 切分，ESS、BAR mutual overlap 和 ESS/GPU-hour 只在 held-out 数据及
独立生产重复上判决。模型、中心化常数、图协议和全部超参数在 production 前冻结并哈希。

“通用”分成两个不同层级：

- **通用训练管线：** 架构、超参数、训练预算和验收门不变，但允许每个体系用自己的
  pilot ledger 重新训练权重；
- **通用冻结模型：** 同一权重跨体系不微调直接使用；这是更强的后续目标，不与第一层混写。

跨体系主验收首先使用至少两个额外 ABFE benchmark/困难窗口，不人工指定 CV、不改变
超参数。路径项端点归零，因此主正确性门是与 converged-MM 自由能在联合不确定度内一致，
而不是要求神经路径修复 MM 对实验的系统误差。实验误差 `<1.0 kcal/mol` 只作为次级报告；
TYK2/CDK2 通常属于 RBFE benchmark，必须在 RBFE Hamiltonian/ledger 接口另行资格通过后
再进入迁移验证。

第一版硬边界：

- 不使用 fragment-total subtraction 或未经处理的 pretrained MACE 总能量；
- 不把 frozen latent 解释为电子密度或真实 Pauli 势；
- 不用离线 NumPy descriptor 冒充可导 production Force；
- 不因 MLP 很小就假定在线 MACE 足够便宜，必须实际比较 L1/L2 的 ESS/GPU-hour；
- \(A_0=A_K=0\) 指全局物理 alchemical 端点，跨窗口共享 λ 的模型和系数必须一致；
- XED 若保留，只能作为 Arm D 特征消融，不向 PME 增加第二套静电。

DEC-030 对上面第四条的回应：本 session 已实测 teacher 侧单帧离线 latent 提取
成本（`derived_5a` CUDA 19.8s、`original_6a` CPU 172.7s，仅一次 forward+backward），
每 MD 步预算需要 O(ms) 级，两者差 3–4 个数量级。这不是"因为 MLP 很小就假定便宜"，
而是 teacher 自己最便宜的离线成本就已经比每步预算慢几个数量级，与 student 本身
的成本无关。因此 L1 在当前证据下降级为非当前下一步，不是未经比较就排除；完整
逐 step ESS/GPU-hour 对照仍保留在 §WP-4C.3 待执行列表中，只是不再是优先级最高的
下一步。

**2026-08-04 状态更新（DEC-031→035，详见 EXPERIMENT_LOG）**：DEC-030(a) 多帧支持域审计发现
frame0-only 固定 manifest 不足以覆盖 1500 帧；DEC-031 尝试的"跨帧闭包并集"修复本身被
DEC-032 撤销——离线 teacher 不需要固定图，正确做法是逐帧独立精确两跳闭包（graph
membership 固定在 CPU float64 决定一次，MACE forward 在 CUDA float32 执行，Option C）；
DEC-033 用这套策略完成了 `derived_5a` 的三条 run 完整 latent cache（各 500 帧，
`[41,1024]` float32，latent-only 不拼 ledger）。DEC-034/035 把这份 latent cache 与 MM
ledger 的 `adjacent_gap_reduced`/`log_importance_unnormalized` 拼接，在**冻结** `A_k`
（本节 sin² 包络，值直接取自 `protocols/EXP-012_preregistration.json` 的
`target.global_schedule.A_k`，不重新拟合，避免 line 174 warns 的尺度简并）的前提下只
训练线性/ridge readout，并以 leave-one-run-out 方式在真正 held-out 的那条 run 上比较
相邻态 gap variance 相对 `B=0`（无残差项）基线的变化：**三个 fold 全部改善**
（39.9%/29.1%/64.8%，均值 44.6%），比"至少一个 fold 有增益"的最低门槛更强。

**DEC-030(c) 冻结（2026-08-04）**：以上结果记为最终登记结果，不因"2/3 fold 选中了 ridge
网格最小候选值 `1e-3`"而重跑更宽的网格来替换它——那样做是看到边界结果后才决定加宽网格，
是事后调整（post-hoc），可能改善数值探针但不能改变已经很强的 go/no-go 结论。之后如果要
跑更宽的 ridge 网格（如 `1e-6/1e-5/1e-4`），必须显式标注为 `sensitivity-only`，不得替换
已登记的 DEC-034/035 结果。

**DEC-030(d0) `LocalResidualStudent` 设计契约（2026-08-04，编码开始前必须先冻结）**：
(c) 通过只说明"frozen-MACE 局部残差表示对这个系统有用"，不等于已经知道如何把它变成一个
可以在 OpenMM 每步内运行的在线可导模型。写学生代码之前必须先冻结以下设计，而不是直接
从一个 MACE 类张量-equivariant 网络开始：

1. **在线动态环境表示（最优先未决问题）**。Teacher 用的是逐帧动态水分子身份和逐帧图
   （DEC-032 Option C），student 必须给出：瞬态水分子身份如何处理、ligand–environment
   近邻发现如何进行、triclinic PBC 如何处理、cutoff 归属如何判定、如何在 TorchScript 里
   表达一个逐步都可能变化的动态近邻结构、以及如何避免每步扫描一个巨大的固定环境。
   这必须先于网络结构选择解决——网络设计依赖于这个答案，不是反过来。
2. **最小架构**：第一个候选不是完整张量-equivariant MACE 式 student，而是最便宜的
   标量、旋转不变局部模型：typed atom embedding + 平滑 ligand–environment
   radial/contact 特征 + 至多 1–2 个轻量 interaction block + ligand-only 不变
   pooling + 有界标量 `B_student`。旋转不变的标量能量对坐标求导本身就给出正确的
   等变力，不需要为了"等变"而引入不变量以外的 irreps 机器；只有这个最简单候选
   失败，才升级到更复杂的表示。
3. **Teacher-target 协议**：每个 outer fold 内——在两条 run 上拟合 ridge teacher、
   在同样两条 run 上训练 student、只在第三条 run 上评估——与 (c) 完全同构，可直接
   比较。Student loss 应同时包含直接 gap 优化项和蒸馏项；teacher 不能被当成无条件
   的 ground truth。
4. **必需的对照实验**：同一 student 架构训练两种版本——(a) direct-gap student（无
   teacher target，只优化 gap variance）；(b) distilled student（相同架构 + teacher
   loss）。否则任何增益都无法归因于 MACE teacher 本身，而可能只是"任意同等容量的
   可训练局部模型直接优化 gap 也能做到"。
5. **计算与部署预算**：编码前必须冻结——最大参数量、最大 neighbor/edge 数（可参考
   DEC-033/034 已实测的 teacher 精确闭包规模，1500 帧范围约 940–1066 节点/
   37834–45656 边，作为同系统近邻规模的经验参考，不直接决定 student cutoff）、
   目标每 MD 步毫秒数、GPU 显存上限、允许的 cutoff、训练 seed 与 epoch 数、早停
   规则、held-out 改善判据。
6. **分阶段条件式实现**（每阶段失败即停，不得跳阶段）：
   - D1 离线 student 拟合 → held-out gap variance 与 teacher fidelity（用真实
     每帧坐标计算 student 自己的特征，不是只读 teacher 的 cached pooled_latent——
     student 的意义就是不在推理时跑 MACE）；
   - D2 坐标/autograd 资格 → 有限差分、cutoff 平滑性、力尾部行为；
   - D3 部署资格 → TorchScript、OpenMM Reference、CUDA 一致性、耗时；
   - D4 动力学资格 → 短 NVT、稳定性，再做独立重复。
   只有 D1 离线 student 仍保留有意义的 held-out 改善，才允许启动 D3/D4 的
   OpenMM/NVT 工作。

本轮只冻结上述契约，不写任何 student 代码；下一步是把第 1 条（在线动态环境表示）
单独作为一次设计讨论解决，其余各项在该讨论之后才能细化为可执行任务。

**2026-08-05 状态更新（DEC-038，详见 EXPERIMENT_LOG）**：第 1 条已解决，且用真实数据验证而非
只靠代码阅读——`scripts/smoke_exp012_student_environment_funnel.py` 在 `openmm_dev` 环境对
run1/frame0 与 teacher 已知最坏图帧 run3/frame343 两条真实帧实测：不追踪瞬态水分子持久身份，
每步用 `local_residual.geometry.ligand_environment_cross_edges`（独立于 teacher 代码路径的
minimum-image cutoff 实现）动态重算 ligand–environment cutoff funnel，与
`local_residual/teacher_graph.py` 的已审计 canonical membership 在两帧上逐对（含周期 unit
shift）完全一致；能量权重改用 `local_residual.geometry.quintic_c2_cutoff` 的 quintic C2
平滑包络，边界扫描证实离散候选成员翻转（在 5.0 Å 处精确单次翻转）不会造成能量跳变，且存在
真实的非 0/1 中间权重值。离散候选成员与能量平滑性由此明确为两个独立层次。本决策只证明设计
可实现，不代表 student 有统计增益或生产资格；下一步是 (d0-5) 计算/部署预算冻结。

**(d0-5) 进度（2026-08-05，详见 IMPLEMENTATION_PLAN §7 WP-4C.3 d0-5，已写为 DEC-039）**：
图规模已用 1500 帧真实几何审计冻结（S1 原子 ≤256/320、边 1536/2048、单原子 neighbor
64/80）；funnel 的 CUDA float32 一致性已验证（与 CPU float64 完全一致）；训练 epoch/seed/
早停预算已冻结——改用训练 run 内部末尾 20% 连续时间块做早停验证（不复用被隔离的第三条
run 兼职早停+最终评估，避免乐观偏差），`max_epoch=500`、`early_stop_patience=30`、
`seeds_per_variant_per_fold=3`。**继续深挖后**，`win_sys_xml_sha256_matches_manifest=false`
的根因已定位：`output_lrc_fix/box_vectors.npy` 只在初次建系统缓存时写一次，真正建窗口 0
用的盒子取自内存里的 `pipeline.box_vectors`，会在预平衡 NPT 弛豫和 Boresch rebalance 后
被重新赋值但从不写回磁盘；窗口 0 生产 System 无 barostat，故盒子建窗口后即冻结，可直接
从 `openmm.chk` 读回真实盒子。诊断脚本已按此修复（两阶段构造，schema 升 v2），但修复后的
重新测量尚未执行，ms/step 数字仍暂不计入 DEC-039 的已冻结生产基线。Arm A/B/D 已正式退役为
`not_pursued`（非数值失败——从未实现；预注册偏离已显式记录，`C_vs_A`/`C_vs_B` 增量比较
从未执行，结论收窄为"MACE latent 信号可泛化、值得蒸馏"而非"Arm C 优于 A/B"），
`protocols/EXP-012_preregistration.json` 已 reseal 为 `sealed`（待跑
`scripts/reseal_exp012_preregistration.py` 落实真实 payload_sha256）。

**2026-08-05 D2 最终结果更新（DEC-040）**：以
`output/outer_lambda_exp012/student_d2_report_v4.json` 为最终报告，direct-gap student 的
3 个 leave-one-run-out fold × 3 seeds 共 9 个 checkpoint，分别在三条真实 run 的 frame 0
上执行，共 27 组坐标/autograd 资格检查；`all_checkpoints_passed=true`，状态为
`COMPLETED_D2_CHECKS`，report SHA-256
`329a98331400f22fe13b76e00f435f4c3a83431441f33bc35af502540d56f08b`。27/27 组有限差分、
cutoff 平滑性和 force-tail 检查全部通过：最大有限差分绝对误差
`2.4711e-7`（门 `1e-4`），最大相对误差 `1.8242e-5`（门 `1e-2`），所有非参与原子力均
严格为零；cutoff 粗/细扫描的能量跳变缩放比为 `22.6405–24.9856`（连续行为期望 25），
且每组被探测 pair 均只发生一次 membership flip；0.3 Å 合成近接触的能量和力 27/27
有限。由此 D2 正式通过；本报告明确未使用 TorchForce、未执行 NVT，也未以 held-out
run 选择 checkpoint，因此不授予 D3、D4、WP-5 或 production 资格。

**D3-0 provenance gate 已关闭（DEC-041，同日）**：`win_sys_xml_sha256_matches_manifest=false`
经冻结协议（同进程两次重建确定性一致 → 换真实 `resolve_dispersion_protocol`/
`resolve_membrane_protocol` 后哈希仍不匹配 → 10 项独立字段核对 + Force canonical
fingerprint 全部一致）判定为 `CLOSED_STEP3_OPERATIONAL_SEMANTIC_PASS`，report SHA-256
`2dc557092ce327c8af3eb2d137c489817a0267377604b077e4469b4d54ba32a8`。结论按冻结措辞
原样记录：记录里没有可检测出的语义差异，历史 byte-level 不一致视为非阻塞，不猜测
具体机制。`no_student_window0_baseline_v2.json` 的 median/P95 `1.3959/1.3968` ms/step
现在可采信为生产基线。System 身份门解除，D3 中依赖真实生产 System 的部分可以开始；
协议只跑一次，不再开第二轮调查。
---

### 2.5 当前执行裁决（DEC-056，2026-08-09）

上面的基线表按其标题日期保留为历史快照；当前主线以本节为准：EXP-012 已完成且
`NOT_PROMOTED`（DEC-048/050），EXP-016 已完成但结论为不晋级，后续没有动作。EXP-013
方案③的 013-B 已按预注册主门判定 `FAILED`（DEC-055/056）：N=8/16/32 的 `z>3`
系统性偏移不能被 `<1.3 K` 的物理量级补充报告推翻，也不得进入 013-C 或事后放宽阈值。

当前只执行方案①的独立资格入口：整个 fused Group-1 作为慢组，沿用 State API 初始化，
禁止跨 integrator `loadCheckpoint()`；先做 `N=1/2/4/8` 低成本物理预检，未通过不得触碰
N=16。Smoke 已通过但不授予 N=16；Qualification 使用 `400/2000` ticks 和 block-aware
SEM 后，N=2/4/8 系统偏移 gate 未通过（DEC-058），因此 `N16_NOT_AUTHORIZED`；该单
种子结果不足以判定方案①普遍存在物理系统偏差，登记
`PHYSICAL_SYSTEMATIC_BIAS_INCONCLUSIVE`。当前进入方案②的 N=1 ESS
信号检查；方案② N=1 无信号后已按 DEC-059 转 EXP-014；EXP-014 当前冻结 screen
未通过（DEC-060），不进入 OpenMM 资格化。ORB 暂只做 charge/spin
contract audit；若父体系总电荷与闭壳层 multiplicity 的合约不能冻结，OMol arm 保持
`EXPLORATORY_ONLY`，不执行 ORB-001 1500-frame probe。

---

## 3. 数学定义

基础路径记为：

\[
H_\lambda^0(\mathbf R)
\]

修改后的目标中间 Hamiltonian 为：

\[
\widetilde H_\lambda(\mathbf R)
=
H_\lambda^0(\mathbf R)
+
B_\lambda(\mathbf R)
\]

外层 λ 神经基势定义为：

\[
B_\lambda(\mathbf R)
=
w(\lambda)
\sum_{m=1}^{M}
c_m(\lambda)\,
\overline U_m(\mathbf R)
\]

其中：

- \(U_m(\mathbf R)\)：第 \(m\) 个冻结的神经基势，不显式依赖 λ。
- \(\overline U_m=U_m-b_m\)：经过能量基准平移后的神经基势。
- \(b_m\)：固定常数，只用于改善数值尺度，不改变神经力。
- \(c_m(\lambda)\)：有界、平滑的外层组合系数。
- \(w(\lambda)\)：端点归零包络。

严格约束：

\[
w(0)=w(1)=0
\]

推荐同时满足：

\[
w'(0)=w'(1)=0
\]

只要 \(c_m(\lambda)\) 在端点有限，就有：

\[
\widetilde H_0=H_0^0,\qquad
\widetilde H_1=H_1^0
\]

因此修改后的路径与基础路径具有相同的精确端点自由能差。

---

## 4. 方法层级与术语约定

必须区分两类不同对象。

### 4.1 神经路径项

\[
B_\lambda(\mathbf R)
\]

它属于每个 λ 状态的目标 Hamiltonian，必须进入：

- 各态目标能量；
- IBS log-sum-exp 中的各态能量；
- 所有跨态能量评估；
- TMBAR/MBAR 的目标 reduced potential；
- 端点和循环闭合验证。

它不能在统计分析中被当作临时采样偏置消除。

### 4.2 IBS 采样偏置

当前 IBS flattening bias 和纯采样 WCA 项负责提高采样效率。它们不属于目标
Hamiltonian，必须继续按现有协议记录并重加权。

禁止出现以下账本错误：

- 将神经路径能量写入 `bias_history` 后整体消除。
- MD 使用神经路径，但 TMBAR 仍使用原始基础路径能量。
- 交换或 IBS 占据使用神经项，离线跨态能量却遗漏神经项。
- 对已经定义为目标中间路径的神经项再做一次“回到旧路径”的全程重加权。

---

## 5. 神经基势的科学定义

### 5.1 不建议直接使用的对象

不应把未经处理的全体系 pretrained MACE 总能量直接作为 \(U_m\)，原因包括：

- 可能重复计算 MM 的键合、内部和环境能量。
- 总能量尺度可能远大于期望的几个 \(k_BT\)。
- 可能把与 alchemical 重组无关的自由度强行加入路径。
- 模型的元素、离子、蛋白和水覆盖范围可能不满足体系要求。
- 即使端点数学正确，也可能产生非物理中间深井。

### 5.2 推荐的神经基势对象

神经基势应优先描述以下一种或几种重组：

- ligand torsion 或结合姿态变化；
- pocket side-chain rotamer；
- cavity hydration 和局部氢键网络；
- 离子占位与交换；
- ligand–environment 多体接触拓扑；
- DEXP/两体中央势无法表达的正交残差力。

神经基势应满足：

- 输出单个旋转和平移不变的标量能量。
- 力来自能量对坐标的负梯度。
- 能量和力具有显式幅度限制或正则。
- 局部区域和支持域有明确记录。
- 对超出训练支持域的构象能够报警或安全衰减。

### 5.3 局部区域策略

按风险递增顺序评估：

1. 配体 + 固定核心口袋原子；
2. 配体 + 完整关键残基；
3. 配体 + 关键残基 + 固定选择的口袋水/离子；
4. 固定总原子集合、动态邻接边的局部环境；
5. 能覆盖水交换的更大区域或专门的局部能量 readout。

第一版不应同时解决动态水交换、离子交换和全蛋白多体重组。

### 5.4 MACE 教师势与生产 bias 的角色分离

外层 λ 方法并不要求生产阶段永久使用完整 MACE。应区分：

- **教师/诊断势**：完整 MACE 局部分解，用于离线标注、识别瓶颈、验证能量和力；
- **直接生产势候选**：经过支持域、力学和成本门后，以多时间步方式参与 MD；
- **廉价生产 bias**：把教师势与慢变量的关系蒸馏为
  \(V_\phi(s(\mathbf R))\)，其中 \(s\) 是少量平滑慢变量。

推荐的最终生产形式为：

\[
\widetilde H_\lambda(\mathbf R)
=
H_\lambda^0(\mathbf R)
+
w(\lambda)c(\lambda)V_\phi(s(\mathbf R))
\]

其中 \(V_\phi\) 可采用 1D/2D spline、tabulated function、低阶多项式或小型
MLP。完整 MACE 负责回答“应当沿哪个慢自由度推、势能形状是什么”，廉价模型负责
每步生产动力学。

不能因为外层系数较小，就把完整原子级 MACE 自动视为慢力。局部 MACE 仍包含氢振动、
短程接触、水取向和侧链局部运动等高频分量。若未经验证地每隔几十步更新一次并冻结旧
力，会改变实际采样 Hamiltonian；端点归零不能修复这种积分误差。

---

## 6. 外层 λ 控制器

### 6.1 最小包络

首个数学基线：

\[
w(\lambda)=\sin^2(\pi\lambda)
\]

优点：

- 两端能量严格为零。
- 两端一阶 λ 导数为零。
- 形式简单，容易测试。

局限：

- 关于 \(\lambda=0.5\) 对称。
- 在接近 0 和 1 的区域迅速变弱。
- 如果主要瓶颈位于当前路径的 \(0.10\rightarrow0\) 尾端，可能作用不足。

### 6.2 推荐的非对称扩展

可使用：

\[
w(\lambda)
=
\lambda^2(1-\lambda)^2g(\lambda)
\]

其中 \(g(\lambda)>0\) 为有界平滑函数，用于移动峰值位置或增加某个困难 λ 区域的权重。

约束：

- 不允许 \(g\) 或 \(c_m\) 在端点发散。
- 包络最大值和积分强度应有显式上限。
- 每次改变函数族都必须更新路径协议版本和缓存指纹。

### 6.3 多基势系数

建议从以下低维函数开始：

- 低阶 Bernstein 多项式；
- 少结点 B-spline；
- 手工定义的局部平滑基函数。

推荐约束：

\[
|c_m(\lambda)|\le c_{\max}
\]

并对下式正则：

\[
\int_0^1
\left|
\frac{d^2c_m}{d\lambda^2}
\right|^2d\lambda
\]

第一版基势数量：

\[
M=1
\]

完整候选版本：

\[
M=2\sim4
\]

除非消融实验明确证明不足，否则不继续增加基势数量。

---

## 7. 与当前 IBS 架构的概念映射

当前 Stage 2 被拆成每个约 4–5 态的小型 IBS ensemble。方案应利用“共享基势”结构：

1. 在同一坐标 \(\mathbf R\) 上计算 \(M\) 个基势能量。
2. 对窗口内每个目标状态 \(k\)，计算：

   \[
   B_k(\mathbf R)
   =
   w(\lambda_k)
   \sum_m c_m(\lambda_k)\overline U_m(\mathbf R)
   \]

3. 将 \(B_k\) 加入该状态的目标 interaction energy。
4. IBS log-sum-exp 使用完整的修改后各态能量。
5. production 的 `energy_history` 保存包含神经路径项的目标能量。
6. `bias_history` 只保存 IBS/WCA 采样偏置。

性能设计原则：

- 神经推理次数应随 \(M\) 增长，而不是随 \(K\times M\) 增长。
- 不应为每个 λ 复制同一个神经基势计算。
- 必须记录每个基势的推理时间和总 MD 性能损失。
- 必须检查 OpenMM CustomCVForce 的 CV 数量上限。
- 若直接使用完整 MACE，应通过独立 force group 和 MTS/r-RESPA 调度，而不是在普通
  积分器外手工冻结旧力。
- MTS 间隔必须同时记录“步数”和物理时间；同一个 \(N\) 在 0.5 fs 与 2 fs 内步长下
  不是同一实验。
- 直接 MACE 路线与蒸馏后的 cheap-CV 路线都必须以 ESS/GPU-hour 判决；不能只比较
  每步墙钟或表面交换率。

---

## 8. 分阶段研发路线

## 阶段 0：冻结基线与问题定位

当前状态：已完成 WP-0。困难窗口和 primary torsion 已由三个独立 scratch run
冻结；这不是 production 采样收益证明。

目标：确认神经路径要解决的具体瓶颈。

工作：

- 选择 softcore 或 DEXP 单一基础路径。
- 冻结 λ schedule、窗口划分、Boresch 参数和 IBS 协议。
- 从基础路径识别最困难的 1–2 个 IBS ensemble。
- 记录相邻 Δu、importance ESS、round trip、torsion、hydration 和异常结构率。
- 判断简单 λ 重排是否已经能解决问题。

通过条件：

- 基础路径结果可复现。
- 至少存在一个明确、可量化且不是简单增加采样即可解决的瓶颈。

停止条件：

- 简单 λ 重排已经达到同等或更好效果。
- 现有结果不足以区分采样不足和路径设计问题。

## 阶段 1：单基势接口与端点验证

当前状态：独立模块中的数学、mock 账本、TorchForce/OpenMM 部署骨架和真实 MACE
EXP-007 qualification 已完成；生产模块仍保持隔离。

目标：只验证 Hamiltonian 结构，不宣称统计效率提升。

形式：

\[
\widetilde H_\lambda
=
H_\lambda^0+w(\lambda)\overline U_1
\]

验证：

- λ=0、1 的总能量逐构象等于基础路径。
- λ=0、1 的坐标力逐原子等于基础路径。
- 中间 λ 的力与有限差分能量梯度一致。
- NVE 或短 NVT 不出现能量漂移、NaN 或异常大力。
- IBS 各态能量、采样偏置和离线目标能量账本闭合。
- 开启神经项后的端点自由能与基础路径在误差内一致。

注意：使用通用 pretrained MACE 只能作为接口测试，不得据此评价方法有效性。

## 阶段 2：单个任务化重组基势

当前状态：EXP-010、EXP-011 均已失败并冻结；EXP-012 已登记为新的可证伪路线。
完整 MACE 直接 MTS、atom-cut fragment teacher 和 torsion-PMF 三条路线均不再推进。
当前没有模型可以进入本阶段的生产对照；target-state ledger 与 backend 审计已补齐，下一门是
冻结并 seal local-residual A/B/C/D 表示、(A_k)、训练预算和数值门，再执行通用表示消融和
gap-variance 诊断。

2026-08-02 已实现 `exp011-umbrella-sample` 和 `exp011-reweight-umbrella`。前者在
完整 window-0 MM expanded-mixture System 上施加最短周期角差 torsion restraint，
逐帧记录角度与 umbrella energy；后者按 source window 去相关，用 MBAR 生成显式
`log_target_weight` 并检查 overlap 图连通性。纯数值、周期边界和相关 CLI 回归已通过。

完整体系 smoke 已晋级：Reference 跳过最小化的诊断运行在第一步出现 NaN，证明不得
省略原协议最小化；在可用 GPU 环境执行的 200 次最小化、单中心一帧正式 smoke 报告
`ok=true`，angle `-173.5426°`、umbrella energy `0.01656 kJ/mol`、temperature
`278.03 K`，checkpoint 与 DCD 完整。下一步先做同一中心短时稳定性 pilot，再扩展到少量
相邻中心检查 overlap；不得用普通 `sample-hard-window-scratch` 代替。

同一中心 10 ps 稳定性 pilot 已通过：10/10 帧有限，temperature `298.12–302.11 K`，
angle `-173.12°` 至 `-158.97°`，最大 umbrella energy `2.79 kJ/mol`。该短 pilot 仅用于
执行资格，不当作正式 PMF 数据。下一步运行相邻 `-157.5°` 中心，并以两窗 MBAR 检查
局部 overlap 后再决定是否扩展中心。

`-157.5°` pilot 与两窗 MBAR 已通过，邻窗 overlap 为 `0.3584`，明显高于 `0.03` 门；
两窗各保留 10 个样本。该结论只验证局部邻窗连通，不验证全周期覆盖或正式 PMF。
下一步增加第三个 `-142.5°` 中心并检查三窗 overlap，仍不直接批量启动 24 centers。

第三窗和三窗 MBAR 已通过；两个邻接界面的 overlap 分别为 `0.3105` 和
`0.1864/0.3728`，均高于 `0.03`。第三窗去相关后只余 5/10 帧，因此 10 ps 仍只作
pilot。下一步不继续顺着同一势阱取点，而是在历史空白区 `75°–165°` 的中部运行
`112.5°` 哨兵窗；通过后再冻结正式 24-center 的采样长度和 replicate 协议。

`112.5°` 哨兵窗已通过可达性检查：angle `93.22°–112.21°`，最后一帧到达
`112.21°`，10/10 帧有限。由于短轨迹分布偏向中心低侧，下一步增加高侧邻窗 `127.5°`
并检查 `112.5°/127.5°` overlap；该界面通过后再冻结正式批量协议。

空白区界面正反向 overlap 为 `0.0759/0.0949`，通过 `0.03` 门，但短 pilot 去相关后只余
5 与 4 帧，因此不进入 PMF。正式采样计划已在任何正式结果产生前冻结：24 个 15° 中心 ×
3 replicates，每窗 50 ps burn-in + 100 ps sampling，每 1 ps 一帧；三个 replicate 使用
三条不同历史困难窗口轨迹末帧和独立 seed。机器协议为
`protocols/EXP-011_umbrella_sampling_plan.json`，断点续跑入口为
`scripts/run_exp011_umbrella_grid.py`。下一步只跑 formal_run1 的单窗正式 smoke，验收后再
放开该 replicate 的剩余 23 窗。

formal_run1 的 `-172.5°` 正式单窗已通过：100/100 帧有限，temperature
`297.74–302.74 K`，周期中心偏差 `-8.82°` 至 `+21.92°`，初态轨迹哈希与断点续跑校验
一致。现在只放开 formal_run1 剩余 23 窗；该 run 的完整环形 overlap 验收前不启动
formal_run2/3。

formal_run1 已完成 24/24 窗和 2400 个有限帧，但严格环形 MBAR 未通过。审计发现旧实现只
检查有向 overlap 图可达，现已修正为 mutual overlap `min(O_ij,O_ji)` 并逐个检查周期邻接
接口，schema 升至 v2。唯一失败接口是 `112.5°↔127.5°`，值 `0.0110 < 0.03`；
`127.5°` 仅有 7 个去相关样本。因此不启动 formal_run2/3，也不查看 PMF。冻结的
EXP-011-AUG-001 只允许从该窗末帧补采 500 ps，合并后重新执行同一严格门。

EXP-011-AUG-001 已完成但未通过：补采轨迹 500/500 帧有限，角度覆盖
`77.50°–169.78°`；补采去相关后仅 15 帧，与原窗合计 22 帧，低于 25。严格 v2 MBAR
中唯一失败接口仍为 `112.5°↔127.5°`，mutual overlap 从 `0.0110` 改善到 `0.02353`，
但仍低于 `0.03`，故 `qualified_for_pmf_input=false`。状态冻结为
`FORMAL_RUN1_OVERLAP_FAILED` 并停止：不再补采，不启动 formal_run2/3，不拟合 PMF，
不进入 NVT、WP-5 或 production；所有现有数据保留供未来重新预注册。

目标：证明一个不使用人工慢 CV 的局部残差势能以可承受成本降低相邻态 energy-gap 方差，
并判定 frozen-MACE latent、轻量等变 encoder 或更简单接触基线中哪一种值得进入 production。

工作：

- 使用已经补齐并通过 CPU/CUDA 审计的三条逐帧五态 target-state ledger；
- 按整条 run 切分，禁止随机拆帧造成轨迹泄漏；
- 比较 A（RBF/contact）、B（轻量等变 cross encoder）、C（frozen-MACE latent）和
  D（可选 XED-inspired field）；
- C 路线只读取 ligand node latent 并使用 invariant scalar readout，不输出环境总能量；
- 分别评估在线 frozen encoder（L1）与 latent-teacher 蒸馏 student（L2）；
- 使用双向 gap variance 与能量/力安全正则训练，ESS/BAR overlap 为 held-out 硬门；
- 冻结模型、中心化常数、幅度、图选择、cutoff、单位和协议哈希后再进入 TorchForce；
- 比较基础路径、仅 λ 重排、解析接触基线和最优单局部残差基势。
通过条件：

- 相同或可比计算成本下，至少一个主指标显著改善。
- 改善不能只表现为表面交换率上升。
- 独立构象转换、ESS 或自相关时间同步改善。
- 没有显著增加异常结构或大力事件。

## 阶段 3：2–4 个神经基势

启动条件：单个基势无法同时覆盖不同 λ 区域。

工作：

- 构建 2–4 个互补基势或离散专家。
- 只优化外层 \(c_m(\lambda)\) 和整体幅度。
- 使用平滑、有界、低维的 λ 系数。
- 做基势数量、包络和局部区域消融。

训练目标：

- 降低相邻态 \(\Delta u\) 方差；
- 提高 overlap/importance ESS；
- 降低正反向迟滞；
- 缩短目标重组变量的自相关时间。

正则项：

- 神经能量幅度；
- 最大附加力；
- 力 RMS；
- λ 曲率；
- 外层系数曲率；
- 支持域外惩罚。

## 阶段 4：完整 IBS 联合验证

目标：证明修改后路径在现有 IBS/TMBAR 统计账本中严格自洽。

必须验证：

- 所有目标状态都包含神经路径项。
- IBS log-sum-exp 使用相同目标状态定义。
- 离线 TMBAR 与在线能量定义一致。
- 跨窗口公共 λ 状态的 Hamiltonian 完全一致。
- complex 和 solvent 两条腿分别保持自己的端点。
- 路径协议、模型哈希和外层系数进入缓存指纹。
- resume 不允许加载不同神经模型或系数生成的旧轨迹。

## 阶段 5：生产比较

至少比较：

1. 原始基础路径；
2. 只重新布置 λ；
3. 单个神经基势；
4. 2–4 个神经基势；
5. 如有必要，再比较连续 λ-conditioned GNN。

每种方案至少进行 3 个独立重复；最终论文级结论建议 5 个独立重复。

---

## 9. 训练目标与数据闭环

### 9.1 不作为主要目标

- 单帧总结合能 RMSE；
- \(E_{\mathrm{MACE}}-E_{\mathrm{DEXP}}\) 的全局拟合误差；
- 仅训练集上的能量相关系数；
- 仅交换接受率。

### 9.2 主要目标

可组合使用：

\[
\mathcal L
=
\alpha\,\mathrm{Var}(\Delta u)
+
\beta\,\mathcal L_{\mathrm{overlap}}
+
\gamma\,\mathcal L_{\mathrm{hysteresis}}
+
\mathcal L_{\mathrm{stability}}
\]

其中稳定性项至少包含：

- 能量幅度正则；
- 力范数正则；
- 力异常分位数正则；
- λ 平滑正则；
- 构象支持域正则。

### 9.3 数据循环

1. 基础路径短采样；
2. 识别低重叠或迟滞窗口；
3. 收集局部构象与力/统计标签；
4. 训练神经基势；
5. 冻结模型进行短验证；
6. 只在发现新支持域时补充一轮数据；
7. 最终冻结模型与外层参数；
8. 重新平衡后开始 production。

production 轨迹不得用于继续在线更新当前 production Hamiltonian。

---

## 10. 验证矩阵

| 类别 | 指标 | 最低要求 |
|---|---|---|
| 端点 | λ=0、1 能量差 | 与基础路径在数值容差内为零 |
| 端点 | λ=0、1 力差 | 与基础路径在数值容差内为零 |
| 力学 | 有限差分力检查 | 通过 |
| 稳定性 | NaN/异常大力/异常结构 | 不高于基线容许范围 |
| 热力学 | 基础路径与神经路径端点 ΔG | 统计误差内一致 |
| 局部重叠 | Δu、BAR/MBAR overlap | 优于基础路径 |
| IBS 效率 | importance ESS | 优于基础路径 |
| 全局遍历 | round trip/状态自相关 | 不劣于基线并有明确改善 |
| 构象遍历 | torsion/hydration/rotamer 转换 | 独立转换次数增加 |
| 迟滞 | 正反向分布差异 | 降低 |
| 成本 | ESS/GPU-hour | 优于基础路径 |
| 重复性 | 独立重复方差 | 不高于基础路径 |

---

## 11. 失败判据

出现以下任一情况，应停止升级复杂度：

- 神经项只提高表面交换率，但不提高独立构象遍历或 ESS。
- 需要过大能量或过大力才能看到改善。
- 中间态出现基础路径中没有的非物理深井。
- 收益可以被简单 λ 重排完全替代。
- 神经推理成本抵消了 ESS 增益。
- 只有把完整 MACE 的外层力更新间隔提高到不稳定或有积分偏差的范围，才能获得可接受
  性能。
- 蒸馏后的 cheap-CV bias 不能重现教师势与目标慢变量之间的稳定关系。
- 多基势相互高度相关，增加基势数量没有可测收益。
- 不同独立重复给出不一致的自由能或重组分布。
- 端点、公共 λ 状态或离线能量账本无法严格闭合。

“该体系不需要神经路径”是允许且有价值的研究结论。

EXP-010 已触发“当前教师–cheap-CV 协议不能跨 run 稳定重现”的失败判据。
该结果不能区分 atom-cut 教师边界伪影、未显式 CV 与 torsion 表达误差，因此不能据此
宣称 torsion bias 无效。EXP-011 必须以完整 MM 条件平均力/PMF 为新目标量；不得把
提高 Fourier order、改变正则或随机拆帧当作 EXP-010 的事后重试。

---

## 12. 配置、缓存与可追溯性要求

从独立研发阶段开始，每次运行至少记录：

- 基础势类型及参数；
- DEXP/softcore 路径协议版本；
- λ schedule 和窗口划分；
- 神经模型文件哈希；
- 模型类型、元素集合、cutoff、精度和设备；
- 局部原子选择规则；
- 基势能量基准 \(b_m\)；
- 包络函数及参数；
- \(c_m(\lambda)\) 的完整定义；
- 能量和力限制参数；
- 训练数据版本与训练随机种子；
- IBS 协议版本；
- OpenMM、OpenMM-Torch、OpenMM-ML、MACE 和 PyTorch 版本。

任一项变化都应使旧 production 缓存失效，除非有经过验证的显式兼容规则。

---

## 13. 推荐的最小可证伪实验

首个实验只做以下范围：

- 使用当前 softcore/IBS 基线。
- 只选择 complex leg 中一个最困难的 Stage 2 ensemble。
- 只使用一个任务化神经基势。
- 局部区域先限制为配体和固定关键口袋原子。
- 使用有界、端点归零的外层包络。
- 保持 λ schedule、IBS 参数和采样预算不变。
- 与“无神经项”和“只重排 λ”两个基线比较。

实验成功的必要条件：

1. 端点能量和力严格回归；
2. 修改后路径与基础路径给出一致的端点 ΔG；
3. 目标慢自由度的独立转换增加；
4. importance ESS 或 ESS/GPU-hour 提升；
5. 没有异常结构率和大力事件上升。

如果该实验失败，不进入多基势阶段。

---

## 14. 最终推荐路线

推荐主线：

\[
\text{基础路径冻结}
\rightarrow
\text{单基势端点验证}
\rightarrow
\text{完整 MACE 教师诊断}
\rightarrow
\text{cheap-CV 蒸馏}
\rightarrow
\text{单个重组问题验证}
\rightarrow
\text{2--4 个共享神经基势}
\rightarrow
\text{完整 IBS/TMBAR 联合验证}
\]

只有当低秩神经基势无法表达所需的 λ 依赖势能面变化时，才考虑连续
λ-conditioned GNN。

当前项目不再把“更换 CUDA 节点后重跑相同 PythonForce MTS 后端”列为推进步骤。
只有出现新的、经单独资格验证的原生 OpenMM/TorchForce MACE 后端时，才可作为新实验
重新打开直接 MTS 分支；该变化不得覆盖 EXP-009 的失败结论。

方法的成功标准不是“使用了 MACE”，而是：

> 在端点和统计账本严格不变的前提下，以更低的 GPU 成本获得更多有效独立样本，
> 并可重复地降低明确的 alchemical 重组瓶颈。



**能分析 λ 路径，但不能指望普通 MACE 从单个构象里“读出真实 λ”。**

因为 (\lambda) 是 Hamiltonian 的外部标签，不是原子坐标。对完全相同的构象 (\mathbf R)：

[
\mathbf z_{\rm MACE}(\mathbf R;\lambda_1)
=========================================

\mathbf z_{\rm MACE}(\mathbf R;\lambda_2)
]

只要 (\lambda) 没有显式输入网络，MACE 就无法区分它们。现成 MACE 可以输出逐原子 descriptor；multihead 则是共享主体加离散 readout，不等于连续输入 (\lambda)。([MACE Documentation][1])

但是，它可以抓住：

[
\boxed{p(\mathbf R\mid\lambda)}
]

也就是**不同 λ 产生了什么不同的构象分布**。这完全可以用于分析 λ 路径。

---

## 一、把 MACE 当作 λ 路径的“显微镜”

冻结一个 MACE encoder，对每个 λ 窗口的每帧构象提取局部描述符：

[
\mathbf z_{kn}
==============

\operatorname{MACEDescriptor}(\mathbf R_{kn}),
]

其中：

* (k)：λ 状态；
* (n)：该状态中的采样帧；
* (\mathbf z)：配体—口袋局部的 MACE descriptor。

然后分析：

[
p(\mathbf z\mid\lambda_k).
]

你的计划本来就规定神经基势不直接接收连续 λ，而由 Hamiltonian 外层的 (w(\lambda)) 和 (c_m(\lambda)) 控制；因此这里的 λ 只作为数据标签，而不是 MACE 输入。

### 最直接的统计量

每个 λ 的 latent 中心：

[
\boldsymbol\mu_k
================

\left\langle
\mathbf z
\right\rangle_{\lambda_k}.
]

相邻状态的结构距离：

[
D_k
===

\left|
\boldsymbol\mu_{k+1}-\boldsymbol\mu_k
\right|.
]

还可以考虑协方差后的 Mahalanobis 距离：

[
D_k^{\rm Mah}
=============

\sqrt{
(\boldsymbol\mu_{k+1}-\boldsymbol\mu_k)^\mathrm T
\Sigma^{-1}
(\boldsymbol\mu_{k+1}-\boldsymbol\mu_k)
}.
]

如果某个区间出现：

[
D_k\gg D_{k-1},D_{k+1},
]

就表示该 λ 区域发生了突然的局部重组，例如：

* cavity hydration 转换；
* rotamer 翻转；
* ligand contact topology 改变；
* 配体姿态或 torsion 转换；
* DEXP 接触突然消失。

这就是“抓到 λ 路径在哪里拐弯”。

---

## 二、可以训练一个 λ-probe，但它只是分析器

冻结 MACE，只在 descriptor 后面接一个很小的分类器或回归器：

[
\widehat\lambda
===============

q_\phi(\mathbf z).
]

MACE 不训练，只有 probe 训练。

### 离散分类

[
q_\phi(k\mid\mathbf z),
\qquad
k=0,\ldots,K.
]

看 confusion matrix：

* (\lambda_k) 和 (\lambda_{k+1}) 经常混淆：两态结构重叠较好；
* 相邻 λ 几乎可以完美区分：两态构象分布分离；
* 某个 λ 被识别成完全独立类别：可能存在特殊中间态或非物理深井。

### 连续回归

[
\mathcal L_\lambda
==================

\left|
q_\phi(\mathbf z)-\lambda
\right|^2.
]

如果预测结果随 λ 平滑变化，说明存在连续的结构响应。如果预测在某一区域跳跃，例如：

[
\widehat\lambda(\mathbf z)
\simeq
\begin{cases}
0.25,&\lambda<0.45,\
0.75,&\lambda>0.45,
\end{cases}
]

那么实际路径可能不是连续重组，而是在 (\lambda\approx0.45) 附近发生两态转换。

但解释必须反过来：

> λ 很容易被预测，不一定是好事。

如果连相邻状态都能被轻易分类，通常说明：

[
p(\mathbf z\mid\lambda_k)
\cap
p(\mathbf z\mid\lambda_{k+1})
]

很小，可能正是路径 overlap 不足。

---

## 三、不要随机拆 frame，要按轨迹拆训练集

这一点非常重要。

同一条 MD 轨迹中相邻帧高度相关。如果随机将 frame 分给 train/test，probe 可能只是记住轨迹，而不是学到 λ 路径。

应该按：

* 独立重复；
* 独立 trajectory；
* 或连续时间 block

进行切分。

例如：

* replica 1、2 用来训练；
* replica 3 完全作为测试。

否则 λ 分类准确率会虚高。

---

## 四、必须区分“显式 λ 效应”和“λ 导致的构象效应”

只收集每个 λ 自己的平衡轨迹：

[
\mathbf R\sim p(\mathbf R\mid\lambda)
]

时，probe 学到的是：

[
\lambda
\longrightarrow
\text{构象分布变化}
\longrightarrow
\mathbf z.
]

这可以分析路径，但无法判断：

* 是 λ 本身改变了能量；
* 还是该 λ 只采到了不同构象；
* 还是采样没收敛。

所以最好使用**公共构象池的跨态能量评价**。

从所有 λ 收集公共构象：

[
\mathcal X={\mathbf R_a}_{a=1}^{N}.
]

对每个构象计算所有 λ 状态的能量：

[
u_{ak}
======

\beta H_{\lambda_k}(\mathbf R_a).
]

得到矩阵：

[
\mathbf U=
\begin{pmatrix}
u_{11}&u_{12}&\cdots&u_{1K}\
u_{21}&u_{22}&\cdots&u_{2K}\
\vdots&\vdots&\ddots&\vdots\
u_{N1}&u_{N2}&\cdots&u_{NK}
\end{pmatrix}.
]

这样就能同时看到：

1. 同一个构象随 λ 的显式能量变化；
2. 各 λ 实际采到了哪些构象；
3. 哪些构象在相邻 λ 之间造成大的 (\Delta u)。

你的 IBS 设计本身也要求所有跨态能量、log-sum-exp 和 MBAR reduced potentials 都包含完整的目标 Hamiltonian，因此这种交叉评价正好与当前账本兼容。

---

## 五、真正判断路径瓶颈：把 latent 分析和 (\Delta u) 联合起来

相邻状态定义：

[
\Delta u_k(\mathbf R)
=====================

\beta
\left[
H_{\lambda_{k+1}}(\mathbf R)
----------------------------

H_{\lambda_k}(\mathbf R)
\right].
]

然后训练一个分析模型：

[
\widehat{\Delta u}_k
====================

f_k(\mathbf z).
]

这不是把它用于生产势，而是问：

> MACE descriptor 中的哪些局部结构，能够解释相邻 λ 能量跳变？

可以进一步算 descriptor channel 与 (\Delta u_k) 的关系：

[
I(\mathbf z;\Delta u_k),
]

或者用 attribution 找出主要原子和局部环境。

那么你可以得到类似：

* (\lambda=0.4\to0.3)：主要由 cavity water 数量控制；
* (\lambda=0.3\to0.2)：主要由 ligand torsion 控制；
* (\lambda=0.2\to0.1)：主要由短程 penetration contact 控制。

这时 MACE 就不只是说“这里 overlap 差”，还可以说**差在哪里**。

你的计划中已经把相邻 (\Delta u)、importance ESS、hydration、torsion 和迟滞列为路径定位指标；MACE descriptor 可以成为连接这些指标的统一表示。

---

## 六、最有价值的分析：检验你的外层低秩 λ 基势假设

你的方案是假设神经路径项可以写成：

[
B_\lambda(\mathbf R)
====================

w(\lambda)
\sum_{m=1}^{M}
c_m(\lambda)\overline U_m(\mathbf R).
]



这个假设本质上是：

> 构象依赖和 λ 依赖可以近似低秩分离。

可以直接分析它是否成立。

假设先为每个 λ 构造一个理想的局部修正或路径诊断量：

[
Y_{ak}
======

Y(\mathbf R_a,\lambda_k).
]

例如 (Y) 可以是：

* 相邻态困难能量；
* 某种局部 residual energy；
* 投影到重组坐标上的 residual force；
* 为降低 (\Delta u) 方差拟合出的窗口 correction。

对矩阵 (Y) 做 SVD：

[
Y
=

USV^\mathrm T.
]

也就是：

[
Y_{ak}
\approx
\sum_{m=1}^{M}
U_m(\mathbf R_a)c_m(\lambda_k).
]

这和你的目标表达式完全对应。

### 解释奇异值

如果：

[
\frac{\sum_{m=1}^{4}s_m^2}
{\sum_m s_m^2}
\approx 0.9\text{--}0.99,
]

那么说明用 2–4 个共享神经基势描述 λ 路径很有希望。

如果需要十几个甚至更多分量，说明：

* λ 路径依赖过于复杂；
* 固定神经基势加外层系数不够；
* 才有理由 fork 架构做真正的 continuous λ-conditioned GNN。

这正好可以在写代码前验证你文档里 (M=1)，随后 (M=2\sim4) 的假设。

---

## 七、还可以画出 λ 路径的“速度”和“曲率”

设 MACE latent 的平均路径为：

[
\boldsymbol\mu(\lambda)
=======================

\mathbb E_{\lambda}[\mathbf z].
]

路径速度：

[
v(\lambda)
==========

\left|
\frac{d\boldsymbol\mu}{d\lambda}
\right|.
]

路径曲率近似：

[
\kappa(\lambda)
===============

\left|
\frac{d^2\boldsymbol\mu}{d\lambda^2}
\right|.
]

离散窗口下：

[
v_k
\approx
\frac{
|\boldsymbol\mu_{k+1}-\boldsymbol\mu_{k-1}|
}{
\lambda_{k+1}-\lambda_{k-1}
},
]

[
\kappa_k
\approx
\left|
\frac{
\boldsymbol\mu_{k+1}-2\boldsymbol\mu_k+\boldsymbol\mu_{k-1}
}{
(\Delta\lambda)^2
}
\right|.
]

* (v_k) 峰值：结构变化最快的位置；
* (\kappa_k) 峰值：路径突然转向的位置；
* 大 (v_k) 同时伴随低 ESS / 大 (\Delta u)：真正的 alchemical bottleneck；
* 大 latent 变化但 (\Delta u) 很小：结构变化存在，但未必影响自由能估计；
* 大 (\Delta u) 但 latent 几乎不动：问题可能是显式 softcore/电荷缩放，而不是局部重组。

---

## 八、所以最合理的第一版不是训练“λ-MACE”

而是做一个 **Frozen-MACE λ-path analyzer**：

[
\boxed{
\mathbf R
\overset{\text{frozen MACE}}{\longrightarrow}
\mathbf z
}
]

然后外部使用：

[
(\mathbf z,\lambda,u_k,\Delta u_k,w_{\rm IBS})
]

进行分析。

最小流程：

1. 从每个 λ 和每个独立重复抽取等量 frame；
2. 用同一个冻结 MACE 提取 ligand–pocket descriptors；
3. 使用 IBS 无偏权重计算各 λ 的 descriptor 分布；
4. 训练 trajectory-blocked λ probe；
5. 计算相邻 λ classifier AUC、latent 距离和路径曲率；
6. 与 (\Delta u)、BAR overlap、importance ESS 联合；
7. 对窗口 correction 矩阵做 SVD；
8. 决定 (M=1)、(M=2\sim4)，还是必须做真正的 λ-conditioned 网络。

因此答案是：

[
\boxed{
\text{MACE 不能直接知道 λ，}
\quad
\text{但可以从 }p(\mathbf R\mid\lambda)
\text{ 中抓住 λ 路径引起的重组。}
}
]

更准确地说，它能做的是**λ 路径表征、瓶颈定位和低秩可分性分析**，而不是从单帧坐标唯一反演 λ。

[1]: https://mace-docs.readthedocs.io/en/latest/guide/descriptors.html?utm_source=chatgpt.com "MACE descriptors — mace 0.3.13 documentation"
