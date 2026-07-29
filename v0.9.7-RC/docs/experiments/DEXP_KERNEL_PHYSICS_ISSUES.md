# DEXP 核参数选择与物理问题记录

> 独立文件，只记录 DEXP softcore 替身势（LJ→DEXP 重设计）这一条线的方法、结果和未解决的物理问题。
> 不合并进 [`../status/AUDIT_STATUS.md`](../status/AUDIT_STATUS.md)。相关但更早的探索见
> [`../handoffs/POSE_SCAN_HANDOFF.md`](../handoffs/POSE_SCAN_HANDOFF.md)（已被本文件
> 里的 anchor-relative 扰动云方案取代，不再往那个方向调参数）。

**当前状态（2026-07-13）：见 §10（最新，Atenolol 单体系试点已结案）。** 一句话版本：这条线
最终定的框架是——MACE 是参考势能面，DEXP 和 LJ 都只是描述它的解析语言，DEXP 的任务是比
LJ 更好地投影 MACE 光滑的 even/径向骨架，不要求复现多体/角度/anchor-specific 细节(odd)。
`--kernel-projection-benchmark`(v2，含逐幅度/逐anchor/robust统计量/switch敏感性/条件均值/
距离分层/小幅子集七项复核，§10.4)在 Atenolol 上实测：**DEXP vs LJ 全面胜出**(所有复核角度
一致，含 odd——`odd 对应任何径向核都碰不到的部分`这句 §10.2 原话已过时修正)；
**(14,5) vs (12,6)：even 上 anchor 层面 19:1 稳固胜者，odd 上 10:10 精确打平、
无统一胜者**。这只是单体系试点，已结案；§10.5 有配套六联核心图
(`plot_kernel_projection_benchmark.py`)；§11 把这套方法论冻结成跨体系协议(含 H1/H2/H3
三个待检验假说 + 体系选择的化学多样性轴，含"配体+纯水无蛋白"廉价对照端点建议)，
供下一步 8-15 体系 `--mace-kernel-benchmark`(尚未实现，体系尚未选定)使用。§9 的双初始态
MD/VAL-SER动力学问题仍然是真实的科学问题，但不再是"要不要修DEXP"的判据，优先级已下调。
§6(contact-type修正)/§7(production equivalence)/§8(切换动力学)均已完成，结论保留在
各自章节，不再是下一步方向。**§10.6（2026-07-13，Phase 3 已完全结案）**：
`--mace-residual-force-benchmark`(force/torque/Hessian投影，跟§10.3/10.4完全不同的
方法论)独立复核，全部结论都有 95% bootstrap CI(逐anchor配对重采样)支持，不跨零——
DEXP 在残差范数/真正 cross-model held-out/完整3x3 Hessian 上都显著优于 LJ，
(14,5) 在 Hessian 曲率上显著优于 (12,6)(K2-K1 Frobenius/特征值CI均不跨零)，
(12,6)/(14,5) 在 force held-out/cosine 上打平(CI跨零)。
**§10.7（2026-07-13，Phase 2 已完全结案）**：`--mace-env-convergence`（这条线唯一需要
新增MACE计算的Phase，5 anchor x 4半径 x 2裁剪方式=520次`_compute_orb_decomposition`）
验证了以上全部结论不依赖当前0.50nm逐原子裁剪的MACE团簇协议——0.90nm相对0.50nm的
target漂移(mean 0.425 kJ/mol)远小于已知odd/even残差量级(~3-8kJ/mol)；odd符号翻转
(11/60)集中在梯度本身接近噪声量级的方向；**kernel排序(K2(14,5)最优)在全部8个环境
定义下100%稳定**。首跑暴露一个显存管理bug(MACE Context缓存只在整个anchor循环结束才
清一次，导致同一anchor内最多8份显存累积不释放，多anchor必OOM)，已修复为每个
(anchor,radius,mode)条件用完立刻清。至此 Phase 1(生产等价性)+Phase 2(环境收敛)+
Phase 3(force/torque/Hessian)全部结案，Atenolol单体系的"DEXP像不像MACE"这条验证链
完整闭环。**但用户2026-07-13指出这不等于"该用哪个核"已经有答案**——现有答案表：谁更
贴近MACE局部能量/力/曲率=(14,5)；谁更接近original LJ平衡构象分布=不知道(§8.4/8.5的
15条replica都未平衡)；谁更接近真实体系=缺实验/独立参考；r0是否需要移动=**已结案，不需要**
(见§9.1旁线实测：even在s_r=1.0处有尖锐最小值，odd全程无显著改善，LOAO 20/20折全票选
s_r=1.0)。alpha、beta、r0三个analytic kernel参数都测过，没有一个能动odd残差——这把
下一步收窄到只剩**主线**：分阶段V/S/B多初态平衡MD(3 condition x 3初态 x 2 replica x
首轮5ns=90ns，见§9.1详细方案，尚未开始，需要GPU排期+补一块"从既有轨迹某帧重新起跑"
的小代码)。8-15体系`--mace-kernel-benchmark`(§11)以及完整端到端ABFE
自由能计算，都往后放，不是当前最优先方向。**alpha/beta全网格扫描(含加宽复测，§9.1
末尾)确认MACE-fidelity这一侧(14,5)/q≈19山谷已经充分验证、不必再调**；但"(12,6)更
接近original、对物理结构更友好"这一侧的唯一依据是§4未平衡的replica MD，不是同等
分量的验证结论——这不是两个都验证过的优解二选一，而是一个已验证结论 vs 一个待验证
说法，V/S/B主线MD正是用来验证后者的。

---

## 1. 背景：为什么要重做 DEXP

原实现：`Orbv3SurrogateFitter.fit_parameters` 对全体 ligand-environment 原子对拟合一套全局
`(alpha_vdw,beta_vdw,r0_vdw,A_fit,B_fit)`，回归目标是结合态平衡轨迹逐帧的绝对能量差
`E_MACE_int - E_gauss_coul`。诊断出两个独立问题：

1. **学习信号问题**：结合态配体几乎不动，逐帧能量涨落主要是噪声，不是形状信息
   （chi2≈9.74/dof=8，跟纯噪声无法区分）。
2. **模型结构问题**：全 ligand-environment 共用一套全局 DEXP 形状，丢掉了原始 LJ 组合律
   本来就有的 pair-specific σ_ij/ε_ij 信息——比 LJ 本身还退化。

---

## 2. 第(a)步：pair-specific LJ-matched 解析基线（已完成，无争议）

### 2.1 做法

不再自由拟合 `A_fit/B_fit/r0_vdw`，改成对每个 pair 解析给出：

```
eps_ij   = sqrt(eps_i * eps_j)              # 标准 Lorentz-Berthelot
sigma_ij = 0.5 * (sigma_i + sigma_j)
r0_ij    = 2^(1/6) * sigma_ij               # 与 LJ 极小点位置完全一致

U_ij(r) = eps_ij * [ (beta/(a-b))*exp(-a*x) - (a/(a-b))*exp(-b*x) ],  x = r/r0_ij - 1
```

对任意 `alpha>beta>0`，在 `r=r0_ij` 处都精确满足 `U(r0)=-eps_ij`（井深）、`U'(r0)=0`（井位，非
假设——是从这两个条件解出系数的必然结果）、`U''(r0)=eps_ij*alpha*beta/r0_ij^2`（曲率）；
`r->0` 时能量/力都有限，无 LJ 的 `r^-12` 奇点。

实现位置：`abfe_core.py::DEXPSurrogatePotential`（表达式生成）+
`SurrogateSystemBuilder.build_surrogate_system`（把去耦前的原始 sigma/epsilon 喂给
`CustomNonbondedForce` 的 per-particle 参数，而不是全局 `addParticle([])`）。

### 2.2 验证

- 解析：二原子 `CustomNonbondedForce` 单测，`U(r0)=-eps` 精确、`U''(r0)` 精确匹配
  `alpha*beta*eps/r0^2`、`r->0` 处能量有限。
- 真实系统：在完整 Atenolol 体系（73536 原子）上建 surrogate system，力/能量有限，
  0.4ps Langevin @300K 无爆炸。

这一步**没有争议**，无论最终 `alpha,beta` 取什么值都应该保留。

---

## 3. 第(b)/(c)步（合并执行）：anchor-relative 局部扰动云 + leave-one-anchor-out 拟合

### 3.1 为什么合并、为什么不再用 pose-scan

`POSE_SCAN_HANDOFF.md` 记录的随机刚体扰动+短程弛豫方案，本质问题是"想把配体推到新的
min-distance 区间"——但结合口袋附近以外根本没有物理意义，而且扰动后完整 minimize 会把
人为构造的局部坡度重新抹回势阱底部。

新方案（`dexp_experiment.py --perturb-scan` / `--perturb-fit`）：从平衡轨迹尾段取 anchor 帧，
只对配体做 ±0.005~0.04nm 平移（沿主惯性轴+随机方向）、±0.5~3° 转动（绕主惯性轴），环境原子
不动，**扰动后不做任何 relax/minimize**。比较 anchor-relative 的 `ΔE_target`（真实 MACE）
与 `ΔU_DEXP`（(a)阶段解析基线）。

### 3.2 --perturb-scan 结果（20 anchor × 74 扰动/anchor = 1480 条记录）

| 指标 | 值 |
|---|---|
| residual RMSE (`ΔE_target - ΔU_DEXP`，核=12,6) | 7.884 kJ/mol |
| residual bias | 2.649 kJ/mol |
| corr(ΔU_DEXP, ΔE_target) | **0.944** |

按扰动类型：

| pert_type | n | residual RMSE | residual bias |
|---|---|---|---|
| rotation | 360 | 4.418 | 1.313 |
| translation | 1120 | 8.710 | 3.078 |

按幅度（残差随幅度**单调增长**，不是噪声）：

| 类型 | 幅度 | n | bias | RMSE |
|---|---|---|---|---|
| rotation | 0.5° | 120 | 0.089 | 0.975 |
| rotation | 1.5° | 120 | 0.780 | 3.059 |
| rotation | 3.0° | 120 | 3.069 | 6.946 |
| translation | 0.005nm | 280 | 0.149 | 1.426 |
| translation | 0.01nm | 280 | 0.588 | 2.911 |
| translation | 0.02nm | 280 | 2.327 | 6.273 |
| translation | 0.04nm | 280 | 9.248 | 15.925 |

**解读**：相关性 0.944 说明不需要学习的解析基线本身已经解释了大部分局部曲率。残差随幅度
增长是**任意有限阶 Taylor 匹配的固有特征**（基线只精确匹配到 `r0` 处的值+一阶导数=0，
没有约束更高阶项），不是基线"坏了"。

### 3.3 --perturb-fit 结果：重新挑选 alpha/beta

方法：按扰动档（3 个 rotation 幅度 + 4 个 translation 幅度 = 7 档）**等权**而非按记录等权
（否则 translation 每档 280 条会不成比例地主导，且 280≠120 纯粹是采样设计造成的，不代表
物理重要性）；`translation@0.04nm` 额外降权 0.5×（它是最偏离"局部"定义的扰动，适合当稳定性
检验，不该独占形状参数）；20 折 **leave-one-anchor-out** 交叉验证（同一 anchor 的 74 条记录
高度相关，不是独立样本，按 anchor 做 CV 才诚实）。

**LOAO 结果**：

| | 值 |
|---|---|
| 19/20 折选中 | 精确 `(14, 5)` |
| 1/20 折选中 | `(13, 6)` |
| alpha 均值±std（跨折） | 13.95 ± 0.22 |
| beta 均值±std（跨折） | 5.05 ± 0.22 |
| 留出集加权 RMSE：拟合 | 5.051 kJ/mol |
| 留出集加权 RMSE：默认(12,6) | 5.823 kJ/mol |
| 改善幅度 | **约 13%** |
| 全数据最优点 | `(14, 5)`，加权RMSE=5.215 kJ/mol |

**2D score surface 的盆地是对角谷（关键发现）**：`L(alpha,beta) <= L_min + 0.261` 的盆地里
8 个网格点：

```
(14, 5)    5.215     (13, 6)    5.229     (12, 7)    5.259     (11, 8)    5.281
(10, 9)    5.292     (15, 4)    5.307     (11, 7.5)  5.346     (12, 6.5)  5.371
```

每个点的 `alpha+beta` 都落在 18.5~19 之间，而 `alpha/beta` 从 1.11 到 3.75（3倍以上跨度）。
对盆地点云做 PCA（不预设退化方向），最紧约束方向是 `(-0.696)*alpha + (-0.718)*beta ≈ -13.285`，
换算即 **`alpha+beta ≈ 19.1`**——数据真正约束住的是这个和，不是 alpha、beta 各自的值，也不是
两者的比值（最初猜的 `(beta, rho=alpha/beta)` 重参数化猜错了方向，PCA 才是对的做法）。

**结论**：`(14,5)` 是这条对角脊上、19/20 折独立选中、恰好是整数的最优代表，**不是两个被独立
精确识别的物理常数**。脊上邻近的 `(13,6)` 几乎同样好。

### 3.4 奇偶分解：alpha/beta 调整到底修正了什么

把每对 `(+delta,-delta)` 拆成：

```
E_odd(delta)  = (E(+delta) - E(-delta)) / 2     # 主要反映局部梯度/平衡位置是否对齐
E_even(delta) = (E(+delta) + E(-delta)) / 2     # 主要反映井宽/曲率/更高偶数阶项
```

| | 默认核 (12,6) | 重拟合核 (14,5) |
|---|---|---|
| translation odd RMSE | 6.611 | 6.733（几乎不变） |
| translation even RMSE | 5.670 | **3.142**（↓45%） |
| rotation odd RMSE | 3.732 | 3.552（几乎不变） |
| rotation even RMSE | 2.365 | **1.390**（↓41%） |

按幅度看 even RMSE 的改善是**均匀的比例改善**（不是只在某个尺度起作用）：

| translation 幅度 | even RMSE 默认 | even RMSE (14,5) |
|---|---|---|
| 0.005nm | 0.176 | 0.108 |
| 0.01nm | 0.693 | 0.414 |
| 0.02nm | 2.737 | 1.589 |
| 0.04nm | 10.982 | 6.065 |

**关键物理解读**：

- `alpha,beta` 调整**几乎只修正 even(曲率)分量**，对 odd(梯度/平衡位置)分量**没有效果**——
  这是必然的，因为两个核共享同一个 `r0_ij`（由 LJ 几何解析给出，不受 alpha/beta 影响），
  且 `U'(r0)=0` 对任意 alpha>beta>0 都精确成立，odd 分量结构上就不可能被 alpha/beta 改变。
- Taylor 展开 `R(delta)=g*delta+0.5*h*delta^2+...`：`|even/odd| ~ |h*delta/(2g)| ∝ |delta|`。
  也就是说 **delta→0 时是 odd(线性/梯度)项主导局部误差预算，不是 even(二次/曲率)项**——
  不能因为小 delta 处 odd 残差的*绝对值*比大 delta 处小，就认为它"在近平衡尺度不重要"；
  真正该比较的是 `g` 与 `h` 这两个系数本身，而当前数据里 `g` 基本不随 alpha/beta 改变。
- 但这不能直接推出"需要加 angular 项"：这里的 odd residual 衡量的是孤立的
  `E_MACE,int - E_gauss_coul` 与 pairwise DEXP 和之间的力差，不是配体在完整体系里实际
  感受到的净力偏差——配体分子内/成键项、蛋白对邻近残基的约束、Gaussian-Coulomb 力本身
  都可能部分抵消这个局部力错配。是否会导致平衡 pose 偏移/接触占有率变化，需要完整体系
  的经验检验——见第 4 节。

---

## 4. 完整体系 replica 对比：LJ / DEXP(12,6) / DEXP(14,5)

### 4.1 方法

3 个 condition（`original`=原始 LJ+PME 力场不做任何修改；`dexp_12_6`；`dexp_14_5`），每个
condition 独立跑 5 个 replica × 2ns（复用已有的 `run_stability_ensemble`：minimize+分段升
vdW/Coulomb 的 softstart+production，同一套流程）。分析时把口袋（非配体）原子先做 Kabsch
叠合去掉系统整体平移转动，再看配体在该参考系里相对 anchor 的真实位移（不是配体对自身
第0帧的内部 RMSD）。Δ⟨q⟩ 显著性判据：`|delta| > 2×(SEM_cond^2+SEM_ref^2)^0.5`（`ref=original`）。

工具：`dexp_experiment.py --replica-run`（每个 condition 单独一次调用，可分别提交到计算
节点）+ `--replica-analyze`（只读分析）。

### 4.2 聚合结果（5 replica 的 mean±SEM）

| 指标 | original | dexp_12_6 | dexp_14_5 |
|---|---|---|---|
| 配体重原子 RMSD (Å) | 0.947±0.030 | 1.044±0.098 | 1.006±0.048 |
| 主 pose cluster 占比 | 0.998±0.001 | 0.930±0.070 | 0.920±0.080 |
| 平移轴0 (nm) | 0.0201±0.0117 | 0.0163±0.0027 | 0.0185±0.0013 |
| 平移轴1 (nm) | -0.0145±0.0028 | -0.0019±0.0061 | 0.0047±0.0018 |
| 平移轴2 (nm) | 0.0023±0.0071 | -0.0035±0.0043 | -0.0001±0.0020 |
| 旋转角 (度) | 9.386±0.511 | 8.464±0.996 | 7.919±0.514 |
| 最近 lig-env 距离 (nm) | 0.1728±0.0002 | 0.1759±0.0010 | 0.1803±0.0007 |
| 过近接触占比 | 0 | 0 | 0.0010±0.0010 |
| 能量 NaN/Inf | 无 | 无 | 无 |

### 4.3 Δ⟨q⟩ 显著性检验（相对 original）

| condition | 显著超过 2×SEM 的量 |
|---|---|
| dexp_12_6 | 仅 `min_dist_mean_nm`（+0.0031nm） |
| dexp_14_5 | `min_dist_mean_nm`（+0.0076nm，比12,6更大）、`translation_axis1`（+0.0192nm）、`rotation_deg`（-1.467°） |

**`dexp_14_5` 在更多、幅度更大的量上偏离 `original`，尽管它在第3.4节的孤立局部曲率诊断里
拟合得更好。**

### 4.4 关键发现：配体酰胺基团的氢键伙伴发生切换

| H-bond (donor-H...acceptor) | original occ | dexp_12_6 occ | dexp_14_5 occ |
|---|---|---|---|
| 羟基 O-H...受体4084（主氢键） | 0.929±0.010 | 0.816±0.019 | 0.691±0.104 |
| 酰胺 N-H...受体2134（原主要伙伴） | 0.791±0.021 | 0.251±0.088 | 0.227±0.031 |
| 环境供体1321...羟基O(4590) | 0.743±0.027 | 0.288±0.022 | 0.344±0.045 |
| 酰胺 N-H...受体2759（原几乎不占据） | 0.089±0.089 | 0.389±0.099 | 0.483±0.066 |

**解读**：这不只是"配体离原位稍微远一点"，而是配体酰胺基团在 `original` 中稳定占据的
氢键伙伴（79%占有率）在两个 DEXP condition 里都塌陷到 ~23-25%，同时一个在 `original`
里几乎不出现的伙伴（8.9%）变成主要占据对象（39-48%）——两个 DEXP condition 都出现了
**相同方向**的重排（伙伴切换），只是幅度略有不同（`dexp_14_5` 切换得更彻底）。羟基氢键
保持主导但持续减弱（93%→82%→69%）。

按第3.4节的奇偶分解框架解释：这正是"odd residual = 局部力方向/平衡位置不匹配"的具体
体现，而且两个 DEXP 核的 odd 分量本来就（几乎）相同（因为共享同一个 `r0_ij`，未被
alpha/beta 触碰），所以两个 condition 都出现方向一致的氢键重排，并不意外。

---

## 5. 意见与建议（Claude 的观点）

**倾向性建议：生产默认值改回 `(alpha,beta)=(12,6)`，不用 `(14,5)`——但用户已表态先都保留，
本节只是我的分析，供参考，不是最终决定。**

理由：

1. `(12,6)` 与 `(14,5)` 共享同一个 `r0_ij`（LJ 几何解析给出）和几乎相同的曲率
   （`alpha*beta`=72 vs 70，仅差~3%），第3.4节已确认 odd(平衡位置/梯度)分量在两者之间
   基本不变——`alpha,beta` 调整能改善的只是 even(曲率)分量，而第4节的氢键重排现象
   由 odd 分量主导，两个核都没有解决它。也就是说，`(14,5)` 相对 `(12,6)` 的唯一优势
   （更好的局部曲率拟合）**跟氢键重排这个更重要的问题完全不相关**。
2. `(14,5)` 的优势本身是在一条对角脊上、非唯一确定的（PCA 确认，α+β≈19 才是真正被
   约束住的量），13% 的留出集 RMSE 改善是真实但不大的效果。
3. 而在更接近生产场景的完整体系 replica 对比里，`(14,5)` 反而在更多指标上（平移、
   转动、最近距离）偏离 `original`，氢键占有率也偏离得比 `(12,6)` 更多——也就是说，
   "对孤立局部微扰拟合得更好"并没有换来"对完整体系动力学更保真"，甚至可能相反。
4. 更本质的问题是：我们并不确定"更贴近 MACE 局部曲率"就等于"更物理正确"——MACE
   本身是在截断的 ligand+environment 团簇上算的，而 `original` 的 LJ 参数也经过了
   自己的参数化/验证流程。在没有独立证据（比如晶体结构、实验结合模式）判断
   哪一个氢键伙伴（受体2134 vs 2759）才是"真实"结合模式之前，选择跟已建立力场
   偏离更小的核，是更保守、风险更低的选择。
5. 第(a)步（pair-specific 解析基线、去掉 LJ 奇点）是纯技术性的、无条件的改进，
   跟 alpha/beta 选哪个无关，应该保留。真正值得继续追的问题不是"要不要用(14,5)"，
   而是"氢键重排这个 odd-residual 驱动的效应本身"——这需要 angular/coordination
   相关的修正，不是继续在 alpha/beta 这两个数字上调。

**如果后续要在两者间做取舍，建议的判据**：看哪个核在更大规模的 replica（比如
现在 5×2ns 基础上再加长/加多）里，氢键占有率和最近距离更稳定地贴近 `original`；
或者如果有外部证据（晶体结构/突变实验）支持受体2134或2759哪个才是真实氢键伙伴，
直接用这个做判据，而不是继续比较 alpha/beta 的局部拟合优劣。

---

## 6. 下一步方案：contact-type odd/even 修正——可行性与意义分析（最小版本已实现，见 §6.5；跑出结果前仍是"待验证"状态）

用户提出的方案：在(a)阶段解析基线之上，按化学角色分类的 contact-type `t` 叠加一个小修正：

```
x_ij = r_ij/r0,ij - 1
U_ij^0 = eps_ij * [ (5/9)*exp(-14*x_ij) - (14/9)*exp(-5*x_ij) ]     # 仍是当前(14,5)基线

ΔU_ij^(t) = lambda_vdw * eps_ij * S(r_ij) * [ a_t * psi_o(x_ij) + b_t * psi_e(x_ij) ]
psi_o(x) = x   * exp(-gamma_o * x^2)      # odd：修正局部力/有效平衡位置，不动 r0 处的值
psi_e(x) = x^2 * exp(-gamma_e * x^2)      # even：修正曲率/高阶形状，不动 r0 处的值和一阶导
```

`gamma_o,gamma_e` 全局共享，每个 contact-type 只有 `a_t,b_t` 两个自由振幅——比"每种 pair 各自
一套 alpha_t,beta_t,r0_t"干净得多，不会重新引入第3.3节那种 alpha+beta 对角谷退化。

验证路径：把已有的 `--learned-rbf-diagnostic`（pair-type RBF ridge + holdout，见
`fit_learned_pair_rbf_diagnostic`/`_build_pair_rbf_matrix`）改造后，做 M0(纯(14,5)基线) vs
M1(+contact-type even修正) vs M2(+contact-type odd+even修正) 三层对比，按 anchor 分组做
grouped LOAO，看 M2 能否在未见 anchor 上显著降低 odd residual、且各折 `a_t,b_t` 符号/量级
稳定。

### 6.1 数学性质核对（正确）

- `psi_o(0)=0, psi_o'(0)=1≠0`：不改变 r0 处的能量值，但贡献非零一阶导数——等价于让该
  接触类型的"有效平衡位置"相对 `r0,ij`(纯 LJ 几何值) 发生小偏移，这正是第3.4节发现的、
  alpha/beta 调整完全碰不到的 odd residual 需要的修正形式。
- `psi_e(0)=0, psi_e'(0)=0`：不改变值也不改变一阶导，只修正二阶及更高阶形状——对应 even
  residual，全局 (14,5) 已经明显改善了这部分，但不同 contact-type 可能仍有各自的残留。
- **两者都是有界函数**（`psi_o` 在 `x=±1/sqrt(2*gamma_o)` 处取极值后衰减到0，`psi_e` 同理），
  这是比当前 DEXP 核心指数项更安全的性质：即使外推到训练扰动云从未覆盖过的位移(比如
  lambda-decoupling 路径上的极端构型)，修正项也不会像纯指数/多项式修正那样发散——不需要
  像 `Orbv3SurrogateFitter` 里那样另外加 clamp 防止 NaN。

### 6.2 可行性

**统计功效是目前最大的现实约束。** 当前 `--perturb-scan` 只有 20 个 anchor，grouped LOAO
只有 20 折。如果一次性摊开用户建议的 8-10 个化学角色类别（每类 2 个系数 = 16-20 个自由
参数），有效独立样本量(20 个 anchor)和参数量几乎同一数量级，即使加 ridge 正则，M2 在
每折上选出的 `a_t,b_t` 也很可能因为"某几个 anchor 恰好包含/不包含某类接触"而大幅波动，
很难达到"跨折稳定"这个晋升生产的判据。而且 Atenolol 口袋里很多类别(cation-anion、
halogen-acceptor 等)可能压根没有或极少出现——`--learned-rbf-diagnostic` 已有的
`min_group_pairs`/`max_type_groups` 过滤机制会自动把这些类别丢掉，但丢掉之后能真正参与
拟合的类别可能只剩 3-5 个（大致对应：配体羟基/酰胺的 donor-acceptor、芳环相关接触、
疏水侧链接触），而不是完整的 8-10 类。

**建议**：不要一开始就摊开全部类别。优先聚焦在第4.4节已经用真实 replica MD **证实**
存在问题的类别——配体两个供体基团(酰胺 N-H、羟基 O-H)相关的 donor-acceptor 接触——
先只用这一类做 M1/M2 对比，把其余接触先合并成一个粗粒度 fallback/other 组当对照。
如果这个最小版本能在 LOAO 下稳定复现"odd residual 显著下降、系数符号一致"，再考虑
要不要拓展到更多类别；如果需要更多类别，同时应该把 `--perturb-anchors` 从 20 提高到
50-100，换取更多 LOAO 折数和统计功效，而不是在 20 个 anchor 上硬拟合一个大模型。

**contact-type 来源不需要从头造。** 已确认 `Atenolol-rank1.itp` 里配体的 GAFF 原子类型
已经区分了 `os`(醚氧)/`o`(羰基氧)/`oh`(羟基氧)/`n3`(仲胺氮)/`n`(酰胺氮)/`ho`(羟基氢，
epsilon=0)——这正是用户要求的"不按元素硬分"所需的化学角色信息，不需要手写化学规则
分类器。蛋白/水一侧的标准 AMBER atom type 预期也有类似粒度(如 `OW/HW` 水、`OG/OG1`
丝氨酸苏氨酸羟基、`OD1/OD2` 天冬氨酸羧酸、`NZ` 赖氨酸胺等)。下一步需要：从
`.top`/`.itp` 解析出每个原子的 atom type 字符串，建一张**力场级别、非 pose 专属**的
静态映射表(GAFF/AMBER type -> ~3-5 个初版化学角色)，这张表可以跨体系复用，满足用户
"不能用具体残基编号/具体配体原子编号"的可迁移性要求。

**`S(r_ij)` 与 `gamma_o,gamma_e` 需要定下具体取值**，目前只有函数形式没有数值。建议：
`gamma_o,gamma_e` 先固定（不当自由参数拟合），取值使包络在 `--perturb-scan` 实际探索的
`|x_ij|` 范围内(平移0.005-0.04nm、转动0.5-3°对应的 x 范围，约 0.01-0.15 量级)不过早衰减，
这样 M1/M2 仍是关于 `a_t,b_t` 的线性模型，可以像现有 RBF 诊断一样直接岭回归求闭式解，
不需要再引入一次非线性网格搜索。`S(r_ij)` 建议直接复用 (a) 阶段 DEXP 核本身的
switch/cutoff(0.5~0.7nm)，而不是另外定义一个——数值上需要确认在这组 gamma 下修正项
在生产 cutoff 处是否已经充分衰减到可忽略，否则要么加大 gamma、要么显式让 `S(r)` 在
cutoff 处强制归零(仿照 abfe_core.py 里 Gaussian-Coulomb 的 shifted-force 处理方式)。

**生产环境下 OpenMM 怎么实现 contact-type 依赖的修正**：不能像现在 DEXP 核那样直接把
`alpha,beta` 写死进表达式字符串，因为 `a_t,b_t` 依赖两端原子各自的化学角色。可行做法是
给每个原子一个整数"角色码"当 per-particle 参数，用 `openmm.Discrete2DFunction`（或需要
连续插值时用 `Continuous2DFunction`）建两张按(角色1,角色2)查表的 `a`/`b` 系数表，在
`CustomNonbondedForce` 表达式里通过表格函数引用——这是 OpenMM 里处理"pair-type 相关
参数"的标准机制，不是要发明新东西。

### 6.3 意义：能验证到什么程度，验证不到什么程度

M0/M1/M2 的 grouped LOAO 只能证明："contact-type 修正能不能在**孤立的
`E_MACE,int - E_gauss_coul` vs pairwise DEXP 和**这个诊断量上，比全局(14,5)基线解释更多、
且跨 anchor 稳定"——这是必要条件，不是充分条件。真正回答"这样做是否修好了第4.4节发现
的氢键伙伴切换"，只能靠把验证通过的 M2 真正接入生产 force，重跑一遍第4节同样的
3(或4)-condition replica MD 对比，看酰胺氢键占有率是否真的从~23-25%回升、朝 `original`
的~79%靠拢。**LOAO 验证过关只说明这个残差具有可迁移的结构，不等于问题已经解决**——
这一步之后仍然需要经验验证，不能跳过。

### 6.4 结论

值得做，性价比高（现有 RBF 诊断代码、perturb-scan 数据、GAFF 原子类型都已经就位，
边际成本主要是重写 feature builder + 重新分类 + 加一层 grouped LOAO 对比，不是从零
搭一套新基础设施）。但要分阶段、聚焦：

1. 先只做"donor-acceptor vs fallback"两类的最小版本，而不是一次性摊开 8-10 类。
2. 如果最小版本在 LOAO 下有稳定信号，考虑把 `--perturb-anchors` 提到 50-100 再拓展类别。
3. `gamma_o,gamma_e` 先固定、只拟合 `a_t,b_t`，保持线性可闭式求解。
4. 通过 LOAO 只是"值得接入生产"的必要条件；接入生产后必须重跑第4节的 replica MD
   对比作为最终验证，不能只看 isolated residual 的改善就直接判定成功。

### 6.5 最小版本实现（`--contact-type-fit`，2026-07-12，尚未运行验证）

按上面 6.4 的分阶段建议，只做了 donor_acceptor vs fallback 两类，落地在
`dexp_experiment.py::run_contact_type_fit`：

- **目标改成 `R = ΔE_MACE - ΔU_DEXP(14,5)`**：注意 `perturb_scan_diagnostics.csv` 里的
  `delta_u_dexp_kjmol` 列实际对应 `_dexp_baseline_pairwise_sum` 的函数默认值 `(12,6)`，
  不是 `(14,5)`——新代码用 `_build_perturbation_distance_tensors`（从 `run_perturbation_fit`
  重构出的共享几何张量构建函数）重新算了一遍 `(14,5)` 的完整 pairwise 基线，不能直接复用
  CSV 里那一列。
- **donor_acceptor 判据**：不用 GAFF/AMBER atom type 字符串表，改成更底层、同样通用的
  "heavy atom 元素(N/O) + 是否键连 H"——键连关系复用 `run_replica_analysis` 里已验证过的
  `system.xml`(HarmonicBondForce+Constraints) 判据（`topology.cif` 的 bonds 对配体 H 不可信，
  这个坑之前已经踩过一次，见该函数内注释）。(i,j) 中一方是 donor(N/O 且带 H)、另一方是
  acceptor(N/O，不要求带H) 即算 donor_acceptor，其余归 fallback。
- **`lig_idx`/`env_idx` 不在 `--perturb-scan` 的 npz 里**：只存了 sigma/eps，没存拓扑原子
  编号，所以 `--contact-type-fit` 需要重新加载一次轨迹、用同一套
  `--ligand`/`--fit-env-radius`/`--fit-env-max-atoms`/`--perturb-anchors`/`--fit-last-ns`
  重新做一遍选择（确定性过程，不涉及随机数，只要参数跟生成该 `--perturb-scan` 时一致就能
  精确复现同一组原子），并加了一个长度校验，参数对不上会直接报错而不是静默用错原子。
- **M0/M1/M2 + grouped LOAO**：复用 `run_perturbation_fit` 同款的按(pert_type,magnitude)
  等权 + `translation@0.04nm` 降权方案，按 anchor 做 20 折 LOAO；M1/M2 用无截距、按列 RMS
  定标(不减均值)的岭回归（`_ridge_fit_no_intercept`），报告 OOF 加权 RMSE、按
  translation/rotation 的奇偶分解、以及 M1/M2 系数跨折 mean/std/符号稳定性。
- **仍是离线诊断**，还没有：(a) 实际跑一次看数值结果是否符合"M2 显著降低 odd residual 且
  系数稳定"这个晋升判据；(b) 接入生产 `CustomNonbondedForce`（`Discrete2DFunction` 查表）；
  (c) 重跑第4节的 replica MD 对比验证氢键占有率是否回升。这三步都还没做，不要假设
  这个最小版本已经解决了问题。

用法：

```
python dexp_experiment.py --contact-type-fit \
    --traj <同一次 --perturb-scan 用的 --traj> \
    --traj-top <同上> --ligand <同上> \
    --fit-env-radius <同上> --fit-env-max-atoms <同上> \
    --perturb-anchors <同上> --fit-last-ns <同上> \
    --system-xml output/system_native.xml
```

### 6.6 实测结果（2026-07-12，ridge_lambda=10.0 单点）：不通过，且 ridge 扫描/角度诊断已排入计划

用户实际跑了一次 `--contact-type-fit`（20 anchor，`--fit-env-radius 0.50 --fit-env-max-atoms 0
--perturb-anchors 20 --fit-last-ns 5.0`，跟生成 `--perturb-scan` 时的默认值一致）。

**首先做了正确性交叉验证**：M0 的奇偶分解——translation odd=6.733/even=3.142，rotation
odd=3.552/even=1.390——跟 §3.4 用 `--perturb-fit` 独立算出的 `(14,5)` 数字**逐位精确一致**。
这是两条完全独立的代码路径（`run_perturbation_fit::_predict_delta_u` vs
`run_contact_type_fit` 里重新算的 (14,5) 完整 pairwise 基线）算出同一个数，交叉验证了新代码
没有算错。

**但 M2 没有通过晋升判据**：

| | M0(基线) | M1(+even) | M2(+odd+even) |
|---|---|---|---|
| OOF 加权RMSE (kJ/mol) | 5.215 | 5.216 | 5.199（仅↓0.3%） |
| odd RMSE [translation] | 6.733 | 6.732 | 6.691（仅↓0.6%） |
| odd RMSE [rotation] | 3.552 | 3.557 | 3.576（**↑0.7%，变差**） |

`donor_acceptor_a_odd`(专门设计来修正 odd residual 的那个系数) 均值+1.215、std=0.168、
20折**100%符号稳定**——看起来"稳"，但对应的实际预测改善几乎为零，且 rotation 上还变差了。
`donor_acceptor_b_even` 只有 60% 符号稳定(std比|mean|还大)，基本是噪声。
`donor_acceptor_contact_active_fraction=1.0`——不是样本覆盖不足的问题(§6.2 担心的"某些
类别可能根本不出现"在这里不成立)。

**结论**：单点 ridge_lambda=10.0 下，donor_acceptor/fallback 两类的**径向** contact-type
修正不能兑现"显著降低 odd residual"这个晋升判据——系数方向稳定不等于系数有用。

**下一步（用户已确认，正在做，尚未跑）**：

1. **ridge_lambda 稳健性扫描**（`--contact-type-fit --contact-type-ridge-lambda-grid
   0.01,0.1,1,10`）：确认上面 ~1% 量级的改善不是 ridge_lambda=10 选太强的假阴性，也不是
   选太弱的假阳性。若扫描后 M2 相对 M0 的最佳改善仍 <=~1-2%（含改善为负的情况），
   代码会打印一条正式关闭 radial contact-type 修正这条路线的提示。
2. **对跨折 out-of-fold 残差做角度诊断**（`--contact-type-angular-diagnostic`，
   `run_contact_type_angular_diagnostic`，只诊断不拟合）：在 `residual_target - oof_pred_m2`
   （M2 用留出该 anchor 训练出的系数、在该 anchor 上给出的预测，不是训练残差）上，对配体
   每个 donor site (heavy atom + 具体 H) 检查：
   - D-H-A 夹角、Δ夹角(anchor->perturbed)；
   - 与最近 acceptor 的距离、Δ距离，以及 Δ距离×Δ夹角交互项；
   - 最近 acceptor 身份是否从 anchor 切换到 perturbed；
   - anchor 态的 acceptor 配位数(0.45nm 内)。
   按用户要求，**每个 anchor 内部单独算相关系数**(20个r值)，再看这些r的符号是否跨anchor
   一致(跟 ridge 系数稳定性同一套判据：sign_stability_frac>=0.85 且 |r|>0.3 的 anchor
   占比>=0.5 才算"稳定信号")，而不是把 20×74 条记录混着算一个池化相关性——避免
   Simpson's paradox 式的伪相关。配位数是 anchor 内部常数，改成跨 20 个 anchor 的
   (配位数, 该anchor残差RMSE) 相关。
3. **决策路径**（用户已定，代码里的 `decision_note` 字段同步了这个逻辑）：
   - 若角度诊断找到稳定信号 → 才最小化地对信号最强的那个 donor site 建立
     `M3 = M2 + k_DA*S(r)*f(theta)`，重新 grouped LOAO 跟 M0/M2 比较，M3 若不能带来
     "明显且跨 fold 稳定"的改善，也不再继续复杂化。
   - 若角度关系同样是 anchor-dependent(不稳定)→ 停止在这条 pairwise/three-body 势函数
     修正的路线上继续加复杂度，转向检查 Gaussian-width/charge-penetration(§6 "还有一个
     可能的接触特异性来源"一节)，或承认剩余项就是无法用任何全局函数形式捕捉的
     多体环境误差(呼应 §3.4 最后一条解读：odd residual 可能部分被配体分子内/成键项、
     蛋白邻近残基约束、Gaussian-Coulomb 力本身抵消，不一定需要"修")。

**实测（2026-07-12）**：ridge_lambda∈{0.01,0.1,1,10} 全网格扫描，M2 相对 M0 最好情形
(λ=1.0)也只改善 1.1%，且 rotation 的 odd RMSE 在**每一个** λ 上都比 M0 更差；角度诊断
(`--contact-type-angular-diagnostic`)里 `any_stable_angular_signal_found=False`——4 个配体
donor site × 4 个变量(Δ夹角/Δ距离/交互项/acceptor切换) × 2 个残差目标，20 个 anchor 里
符号稳定性最高只有 70%(接近随机)。**radial 和 angular 两条路都已确认无稳定信号，正式关闭**。
用户选择按 §6 "还有一个可能的接触特异性来源"一节，继续检查 Gaussian 宽度/电荷穿透，
而不是直接承认多体环境误差——见 §6.7。

### 6.7 Gaussian 宽度/电荷穿透诊断（`--gaussian-width-diagnostic`，已实现，尚未运行）

`run_gaussian_width_diagnostic` 检查 M0 残差 R 是否与"统一 `sigma_elec=0.10nm` 电荷穿透
误差"的代理量存在稳定关联，只诊断、不拟合 role-specific sigma_elec：

- 从 system.xml 的 NonbondedForce 读 lig/env 部分电荷；原样复刻
  `build_mm_le_contexts_from_system_xml` 里 "gauss_coul" 参考项的确切定义(sigma_elec 硬编码
  0.10nm；由 `--fit-mm-ref-cutoff`(默认 0.0→NoCutoff，不做 PBC wrap，跟 MACE 真空团簇边界
  条件一致)/`--fit-mm-ref-switch` 决定是否加 CutoffPeriodic+switching)——这跟 §6.5/§6.6 用
  的 DEXP vdW 0.70nm cutoff 是两套完全独立的距离/开关规则，不能混用。
- `Δpenetration = Δ(bare点电荷Coulomb) - Δ(gauss_coul当前sigma_elec)`：这是"用 Gaussian
  而不是点电荷"引入的修正量，纯粹由坐标+电荷决定，跟 MACE/vdW 无关，是电荷穿透误差的
  直接几何代理。分别在全部 lig-env pair 和只在 donor_acceptor(§6.5 定义) pair 上算。
- 跟 §6.6 完全一致的方法论(每个 anchor 内部单独算 Pearson r，再看 20 个 r 的符号是否
  跨 anchor 稳定)，检验 R(M0残差) 和 M2 跨折 out-of-fold 残差是否与 Δgauss/Δbare/
  Δpenetration(各 2 个变体)存在稳定关联。
- **如果这里也 `any_stable_signal_found=false`**：radial/angular/electrostatic 三条路都
  没有在这 20 个 anchor 上找到稳定信号，§6 这整条 contact-type 修正探索到此为止，剩余
  odd residual 应记录为"无法用任何已尝试的全局函数形式捕捉的多体环境误差"（呼应 §3.4
  最后一条解读），生产默认核维持 (14,5)（或参考 §5 改回 (12,6)，两者都跟这条 odd-residual
  探索的结论无关）。
- **如果为 true**：说明确实存在跟电荷/donor-acceptor 相关的稳定信号，下一步才是考虑
  role-specific sigma_elec，且必须用独立的静电专属验证(不能跟 DEXP vdW 残差一锅拟合，
  否则分不清 vdW shape/Gaussian charge penetration/angular 三种误差来源——用户明确要求)。

**实测（2026-07-12）**：`any_stable_signal_found=False`。6 个变量(Δgauss/Δbare/Δpenetration ×
全部pair/donor_acceptor限定) × 2 个残差目标(M0/M2 out-of-fold)，没有一个跨过
sign_stability>=85%的门槛，最高只有 70%(all-pairs 的 Δgauss/Δbare)。**尤其值得注意**：
`donor_acceptor` 限定的三个变量(charge-penetration假说本该最明显的地方)反而信号最弱
(mean_r 仅 -0.02~-0.06，符号稳定性 50-60%，接近随机)——这跟"电荷穿透在极性/氢键接触上
更严重"的物理预期方向相反，是不利于"统一sigma_elec电荷穿透"这个假说的证据，不只是
"没达到门槛"那么模糊。all-pairs 那组稍高但仍未达标的信号(mean_r 0.18~0.25, 70%稳定性)，
更像是"总静电环境变化"和残差之间某种泛泛的、非特异性关联，不是电荷穿透的定向证据。

### 6.8 §6 结论：三条修正路线均已排除，正式结案

Radial contact-type(§6.5)、angular H-bond 几何(§6.6)、Gaussian 宽度/电荷穿透(§6.7)——三条
独立诊断，各自用同一套"跨 anchor 符号稳定性"判据检验，**全部未在 20 个 anchor 上找到
可迁移信号**。按 §6.2/§6.4/§6.6/§6.7 一路定下的判据，不再继续在这条"contact-type/pairwise
修正 DEXP odd residual"的路线上加复杂度（不建 M1/M2 之外的更多 radial 类别，不建 M3
angular 项，不建 role-specific sigma_elec）。

**结论**：第3.4节发现的 odd residual（局部力/有效平衡位置不匹配，alpha/beta 结构性
无法触碰的那部分）应记录为**无法用目前尝试过的任何全局函数形式（pairwise 径向形状、
三体 donor-H-acceptor 角度、统一 Gaussian 电荷穿透）捕捉的残留误差**——很可能主要是
孤立 `E_MACE_int - E_gauss_coul` 二体分解本身的产物，在完整体系里可能被配体分子内/
成键项、蛋白邻近残基约束、Gaussian-Coulomb 力本身部分抵消（§3.4 最后一条解读），
不代表完整体系里存在同等大小的净力误差。第4.4节观察到的氢键伙伴切换现象仍然真实存在
(§4)，但这条 contact-type 修正路线未能解释它，也不再是本文件下一步会继续深挖的方向。

生产默认核维持现状——`(alpha,beta)=(14,5)`（第(a)步 pair-specific LJ-matched 解析基线
不变），是否改回 `(12,6)` 仍是 §5 的独立判断，不受本节结论影响。

---

## 7. Phase 1：production-equivalence audit（用户 2026-07-12 提出的 4 阶段方案，`--production-equivalence-audit`，已实现，尚未运行）

§6 结案后用户提出一个 4 阶段方案，重新审视 §3-§6 全部结论的地基是否可靠。最高优先级的
Phase 1（`run_production_equivalence_audit`）不需要重跑 MACE，只需要 OpenMM，核对：

1. **原子身份/顺序**：重新选取的 `lig_idx`/`env_idx`，其 sigma/epsilon/charge（直接读
   `system_native.xml`）是否跟 `--perturb-scan` 缓存在 npz 里的 `sigma_lig`/`eps_lig`/
   `sigma_env`/`eps_env` **逐元素精确相等**（`np.array_equal`，不是近似）。
2. **(14,5) DEXP 基线的关键差异点**：§3.3 起一直用的 NumPy pairwise 求和
   (`_dexp_baseline_pairwise_sum`/`_build_perturbation_distance_tensors`)只做硬 cutoff
   mask(`r<=0.70nm`)，**没有** switching function；原代码注释的理由是"switch 只在
   0.5~0.7nm 边缘生效，对 r<0.3nm 排斥墙区域影响可忽略"——这是一个从未验证过的假设。
   真正的生产力（`abfe_core.DEXPSurrogatePotential`，本审计直接调用它的
   `build_expression()`，不是重新推导表达式）在 0.50→0.70nm 用标准 quintic switching。
   审计对比四个量：现有基线(无switch) vs 生产(有switch)、加switch后的NumPy vs 生产、
   加switch前后的NumPy在各自"同款有/无switch"设定下是否跟对应的OpenMM结果一致(隔离
   switch缺失是否是唯一差异来源)。
3. **§6.7 用的 NoCutoff Gaussian-Coulomb 参考**的 NumPy 重实现 vs 同一个
   `build_mm_le_contexts_from_system_xml` 生成的 `gauss_coul` OpenMM Context——注意这
   **不是**"是否等于生产"检验：生产的静电项用 shifted-force+cutoff=0.70nm，`--perturb-scan`
   算 `delta_e_target` 时用的参考项刻意用 NoCutoff+无shifted-force（近似 MACE 真空团簇
   边界条件），两者本来就该不同，这里只检验"我的 NumPy 重实现有没有算对"。
4. **有限差分力**：NumPy(加switch)energy函数中心差分 vs 生产 OpenMM `getForces()`解析力，
   对最近接触的配体原子取相对误差。

**判据（用户指定）**：能量差 `< 1e-5 kJ/mol`（`--audit-energy-tol-kjmol`），有限差分力
相对误差 `< 1e-4`（`--audit-force-rel-tol`）。只有通过后，才能把 §3-§6 的 residual 称为
"物理模型误差"而不是"离线复现本来就跟生产不一致"的假象——尤其是第 2 点，如果
`existing_baseline_matches_production=false` 而 `switch_corrected_numpy_matches_production=true`，
说明 §3.4 起报告的所有 odd/even RMSE 数字都需要重新用"加了switch的基线"算一遍才算数，
§6 的三条结案(radial/angular/electrostatic 均无信号)需要重新确认结论是否变化。

**实测（2026-07-12）**：Phase 1 通过。

- 原子身份/顺序：`sigma_lig/eps_lig/sigma_env/eps_env` 与 npz 缓存**逐元素精确相等**。
- 有限差分力 vs 生产解析力：相对误差 9e-7~1e-5，远低于 1e-4 阈值。
- **关键判据**：加 switch 后，residual 分析真正用到的 `ΔU_DEXP(pert-anchor)` 这个差分量，
  全部 1480 条记录里 `max|Δ|=0.385 kJ/mol`（只在最大、且早就因"最偏离局部定义"被降权的
  0.04nm 平移档才达到这个值；核心幅度 0.005-0.02nm/0.5-1.5° 只有 0.02-0.24 kJ/mol）——
  比 residual RMSE 量级(3-8 kJ/mol)小 1-2 个数量级。加 switch 后重算的奇偶分解
  (translation odd=6.750/even=3.144, rotation odd=3.550/even=1.393)跟现有数字
  (6.733/3.142, 3.552/1.390)几乎完全一致(差异<1%)。**结论：§3.4 起报告的全部odd/even
  数字、以及§6三条"无信号"结案，都不受缺失switching function影响，不需要重算。**
- 绝对能量层面 `existing_baseline_matches_production=False`（~8-9kJ/mol差异，正是switch
  缺失的预期效应，符合假设）、`switch_corrected_matches_production=False`/
  `gauss_numpy_reimplementation_correct=False`(严格来说未过1e-5kJ/mol阈值，实测差异
  ~1e-4kJ/mol)——这三个"未通过"都是在绝对能量(数千个pair求和)尺度上判定，浮点求和顺序/
  erf库实现细节层面的噪声，比~kJ/mol的物理尺度小4-5个数量级，不是公式错误，且这个尺度
  本来就不是residual分析实际依赖的量——上面"关键判据"那条才是回答"是否影响结论"的
  正确判据，且明确通过。

后续 3 个 Phase（Phase 1 通过后无需调整已有结论，可按原计划或视优先级顺序推进）：
2. 5-anchor 环境半径收敛（0.50/0.60/0.70/0.90nm，只用最小幅度扰动，看 ΔE_MACE/residual
   gradient/odd residual 方向是否收敛，判断 residual 是否是 MACE 局部环境截断的产物）——
   **已完成（2026-07-13），`--mace-env-convergence`，结果见 §10.7：target漂移远小于残差
   量级，kernel排序((14,5)最优)在全部8个环境定义下100%稳定，Phase 2结案**。这是这条线里
   唯一需要新增 MACE 计算的 Phase（成本量级：5 anchor x 13 geometry x 4 半径 x 2 裁剪方式
   = 520 个 `_compute_orb_decomposition` 调用，约 1560 次实际 MACE 前向计算）；
3. 把 residual/target/kernel 都转成每个 anchor 的局部力/力矩投影（`g≈(R(+δ)-R(-δ))/(2δ)`，
   用跨幅度线性+三次拟合取 δ→0 极限），检验同一 anchor 内不同幅度估计的局部线性一致性，以及
   3 个平移/转动主轴重建的局部向量是否满足 cosine similarity/随机方向 held-out 检验——直接
   判断 K0/K1/K2 谁在方向上更好地投影 MACE 局部响应，不止是能量 RMSE——**已完成（2026-07-13），
   `--mace-residual-force-benchmark`，结果见 §10.6：跟§10.3/§10.4能量RMSE结论完全一致且
   互相独立印证，DEXP在残差范数/cross-model held-out/完整Hessian上显著优于LJ，
   (14,5)在Hessian曲率上明显优于(12,6)**；
4. 回到 original/(12,6)/(14,5) 三个 condition 已有的 replica 可观测量（pose 稳定性、
   RMSD、氢键占有率、min-distance/RDF/PMF、异常短接触）——如果 (14,5) 在这些真实指标上
   已经稳定且明显优于 (12,6)，即使局部 residual RMSE 仍有 ~5 kJ/mol，也没必要继续增加
   势函数复杂度（注：§4.2/4.3 已有数据显示 (14,5) 目前在多个指标上比 (12,6) 更偏离
   original，这一条件目前看来尚未满足，需要更大规模 replica 才能进一步确认）。

---

## 8. 关键修正：VAL/SER"伙伴切换"的真实机制 + 现有轨迹是否已平衡（用户 2026-07-12 提出）

### 8.1 受体身份核实 + 机制重新解读

用户核实了第4.4节表格里的受体原子身份：

| 原子编号 | 身份 |
|---|---|
| 2134 | VAL136 主链羰基 O |
| 2759 | SER177 侧链 OG |
| 4084 | ASN254 OD1 |
| 1321 | ASP85 HD2 |

更关键的发现：**H4607 和 H4608 是配体同一个酰胺 N(4587) 上的两个不同 H**，轨迹统计显示
H4607 主要跟 VAL136-O 相互作用、H4608 主要跟 SER177-OG 相互作用，且在 `original` replica 4
里两个氢键能同时出现。这意味着第4.4节说的"VAL→SER 伙伴切换"不是"同一个 H 在两个受体间
简单切换"，而是**配体酰胺 NH2 在(固定的)VAL136 主链羰基与(需要合适 rotamer 才能靠近的)
SER177 可转动侧链之间形成竞争性、可能双叉的氢键网络**。

这个重新解读直接说明了 §6.6 角度诊断为什么找不到信号：那个诊断固定环境、只对配体做刚体
扰动，结构上完全看不见 SER177 侧链 rotamer 转动和环境弛豫这个自由度——如果真实机制需要
环境弛豫，任何"只扰配体、冻结环境"的方案都不可能捕捉到它，不管径向/角度修正的函数形式
多复杂。这不代表 §6 的三条诊断做错了，而是说明它们问的问题（"pairwise/three-body 修正能
否解释这个孤立能量残差"）从一开始就问的不是产生氢键切换现象的那个自由度。

### 8.2 MACE 环境团簇的原子级截断（已确认，尚未验证是否有实际影响）

`abfe_core.py::_select_env_indices_from_mdtraj_frame`（line 95）文档明确写着："这里裁的是
原子，不是整盒水，也不是全环境残基"——纯半径近邻筛选，不补全残基。`_compute_orb_decomposition`
（line 4440 附近）用 `e_cplx - e_lig - e_env` 三个独立 MACE 团簇能量相减，这三个团簇都是
按这个原子级裁剪出来的（无残基补全、无断键补氢）。complex 与 environment 相减确实会抵消
一部分边界能量，但靠近配体的边界原子（例如恰好卡在 0.50nm 半径边缘、被切掉相邻原子的羰基/
侧链片段、被截断的水分子）的 MACE 响应仍可能依赖被删掉的邻域，且这个截断方式在不同 anchor
帧之间可能表现不同——这是完全合理的 anchor-dependent residual 来源候选。**这一条目前只是
确认代码行为符合用户描述，尚未验证是否真的对 residual 有可测量的影响**（会需要新的 MACE
计算，比较"原子级裁剪"vs"残基补全裁剪"的环境团簇能量差，成本不低，暂不作为下一步，
优先级低于 §8.3）。

### 8.3 下一步一：现有轨迹的 V/S/B/N 切换动力学分析（`--hbond-switching-dynamics`，已实现，尚未运行）

不需要新 MD、不需要 MACE，只读分析已经跑完的 `--replica-run` 轨迹（`run_hbond_switching_dynamics`）。
逐帧定义四态：V(只有VAL136-O氢键)、S(只有SER177-OG氢键)、B(双叉)、N(都没有)——V/S 判据用
H4607/H4608 任一满足距离(`--replica-hbond-dist-nm`)+角度(`--replica-hbond-angle-deg`)判据即可，
不假设哪个H对应哪个受体。每个 replica 输出：

- V/S/B/N occupancy；转移次数与转移矩阵；首次 V→S 转移帧；
- 各状态驻留时间(run-length)统计；前半段 vs 后半段 occupancy；
- indicator(是否S键合/是否V键合)自相关积分时间 + 有效样本数(`n_eff`)；
- 是否"离开初始态后再也没有回归"（配合总状态切换数一起看，判断是否只是单向漂移）。

第4.4节已有的异质性（`original`5个replica都保持VAL、只有replica4出现SER；`DEXP(12,6)`4个
replica有SER约0.46-0.56占比、1个保持VAL；`DEXP(14,5)`5个replica都有SER、占比0.23-0.60）
本身就已经暗示这些 occupancy 很可能不是平衡概率——如果每条2ns轨迹只经历0-1次真实切换，
不同replica间的occupancy差异可能只反映"从哪个初态出发、多久之后随机跳了一次"，不能直接
解读为DEXP核参数改变了平衡氢键偏好。这一步的结果决定：

- 如果确认是短轨迹初态依赖的产物（大部分replica只切换0-1次，或`n_eff`远小于帧数）：
  §5 的 (14,5) vs (12,6) 判断需要先加长/加多 replica，或者专门从 V/S/B 多个不同初始态
  分别启动做统计，而不能直接用现有 occupancy 数字下结论——现有第4节的对比结论需要标注
  "有效性存疑，待重新验证"。
- 如果确认转换本身发生多次、双向可逆、轨迹前后半段occupancy相近：说明现有2ns轨迹已经
  采样到准平衡分布，occupancy数字本身可信，(14,5)vs(12,6)的比较站得住脚，可以按原计划
  推进第4阶段(更大规模replica验证)。

用法（`--traj`/`--traj-top`/`--system-xml`/`--replica-conditions`/`--stability-replicas`
需要跟生成这批 `--replica-run` 轨迹时一致）：

```
python dexp_experiment.py --hbond-switching-dynamics \
    --traj output/pre_equilibration.dcd --traj-top output/topology.cif \
    --system-xml output/system_native.xml \
    --replica-conditions original,dexp_12_6,dexp_14_5 --stability-replicas 5
```

### 8.4 实测结果（2026-07-12）：不是单次跳变，但存在真实的未平衡漂移

`any_replica_with_at_most_one_state_change=False`——15 个 replica 没有一个是"只切换0-1次/
离开初始态后再没回归"的退化情形，每条 2ns 轨迹都有 47-140 次真实的状态切换，最坏情形
(单次不可逆跳变让occupancy纯粹是初态假象)没有发生。

但更细致的检验(n_eff、前后半段occupancy对比)揭示了一个不同的问题：

- **n_eff 在多个 replica 里大幅低于名义帧数**：最严重的是 `dexp_14_5` replica1，
  `n_eff=6.5/200`——尽管有109次名义状态切换，S指示量的自相关积分时间意味着只有约6个
  真正独立的观测。`original` replica3(21.2)、`dexp_14_5` replica3(45.5)/replica0(100.2)
  也有明显折损。
- **前后半段 occupancy 在这些低n_eff的replica里出现实质性漂移**：最突出的是
  `dexp_14_5` replica1，S+B占据从前半段的0.46掉到后半段的**0.00**——这条replica在2ns
  内做了一次慢弛豫，不是在采样一个平稳分布，报告的全轨迹occupancy很大程度上取决于
  恰好看到了哪一半。

**两层结论都要说**：

1. **定性方向大概率是真的**：`original` 5个replica里4个S occupancy恰好是0.00(尽管
   各自有47-66次真实的V<->N切换)，`dexp_12_6`4/5个replica打开了S，`dexp_14_5`5/5个都
   打开了S——这个跨replica一致的定性格局建立在每条轨迹几十到上百次真实转换之上，不是
   单次随机噪声，值得信任。
2. **§4.2/4.3 报告的精确 occupancy 数值目前不该被当成收敛量**：低n_eff+前后半段漂移
   说明现有 mean±SEM(按replica数算)低估了真实不确定度，且部分replica(尤其是撑起
   DEXP核occupancy估计的那几条)还没有到达平稳分布。§5 的 (14,5) vs (12,6) 量化比较
   (哪个核占据率更接近original多少个百分点)现在还不能拿来做精细判断，方向性结论
   （两个DEXP核都让配体从original的VAL偏好转向SER，(14,5)转得更彻底）目前看更站得住脚。

下一步建议：针对 n_eff 最低/前后半段漂移最明显的几条 replica(`dexp_14_5` r1、
`original` r3、`dexp_14_5` r3)专门延长(而不是所有15条replica都等比例延长)，看它们是否
在更长时间内重新回到、或者稳定偏离前半段的分布——这比不加区分地把所有replica都跑
更长更有信息量，但需要新的 MD，尚未做决定。

### 8.5 committed-state 升级版实测（`--hbond-committed-state-dynamics`，2026-07-12）：`any_replica_not_equilibrated=True`（全部15个replica）

用户指出 §8.4 的"几十次状态切换"大量是阈值附近抖动（驻留时间1-2帧），不能证明轨迹已平衡；
用连续coordination(quintic平滑)+Schmitt trigger(进入/离开阈值不同)+最小驻留(4帧)去抖动
重做后，结果比 §8.4 更明确：

- **`original`**：4/5 replica 从未真正commit过SER(纯V occupancy 0.88-0.98)，被
  `single_basin_only`+condition级`majority_single_basin`标记；唯一访问过两个basin的
  replica3，committed V<->S穿越各只有1次，drift=0.32，n_eff(val/ser)=10.6/7.1——都不够。
- **`dexp_12_6`**：2/5 replica单basin锁定(其中一个反而锁定在**纯S**，occ S=0.51)，其余3个
  visited_both_basins但穿越次数只有1-2次；condition内不同初始态replica后半段occupancy
  最大差达 **0.72**——同一condition的replica，因为起点不同，2ns内完全没有相互靠拢。
- **`dexp_14_5`**：5/5 replica都访问了两个basin，但committed V->S/S->V穿越次数经常为0——
  细究发现这不是矛盾：3/5 replica的**纯V occupancy恰好是0.00**，VAL commitment只要发生
  就总是跟SER commitment同时出现(即只以B形式出现，从不单独以V形式出现)，所以"进入纯V
  态"这个事件本身就没发生过，无法被(只追踪纯V/S互相进出的)穿越计数器捕捉到——这是该
  计数器的一个已知盲区(应该补一个独立追踪committed_val/committed_ser各自开关次数的指标)，
  但不影响其余四条判据(drift/单basin/n_eff/跨初始态发散)全部触发的结论。

**15/15 个replica全部被标记`not_equilibrated=True`**——不是"部分replica存疑"，是全部。

**定性/定量两层结论进一步收紧**：
- 定性方向更清楚了：`original`几乎锁死在纯V；`dexp_12_6`介于中间且组间严重不收敛；
  `dexp_14_5`让"纯V"这个态几乎消失(3/5 replica occ V=0.00)，被S或B取代。这个格局在
  三个独立condition、15个replica上一致，值得相信。
- 但§4.2/4.3 报告的精确occupancy数值现在可以确认**不是平衡量，对任何一个condition都不是**——
  不只是"DEXP核的那几条replica不够"，`original`自己的4/5个replica也只是"恰好没有被推离
  V"，不代表V就是original下的真实平衡态占据率。§5 的(14,5) vs (12,6)量化取舍在拿到
  真正收敛的数据前，两个方向都不成立。

---

## 9. 整体结论（2026-07-12，综合 §1-§8）

**站得住的结论**：

1. 第(a)步 pair-specific LJ-matched 解析基线本身是干净的、production-verified 的改进，
   跟 alpha/beta 选哪个无关，应该保留。
2. alpha/beta 形状调整在数学上只能修正局部能量失配的偶(曲率)分量，永远碰不到奇(梯度/
   平衡位置)分量——两个核共享同一个解析 r0_ij、且 U'(r0)=0 对任意 alpha>beta>0 恒成立，
   这是结构性事实，不是拟合不够好的问题（§3.4）。
3. 奇残差**无法**用更聪明的 pairwise/three-body/静电宽度修正解释——radial contact-type、
   angular H-bond 几何、Gaussian 电荷穿透三条独立诊断(grouped LOAO + 按anchor符号稳定性
   判据)全部返回无信号，且已确认不是"离线复现跟生产不一致"的假象(Phase 1 通过，见§7)。
   **这扇门已经关闭：不要再在"给DEXP核加更复杂的修正项"这条路上投入。**
4. 触发这一切的"配体酰胺 VAL<->SER 氢键伙伴切换"现象是真实的、跟力场核参数选择相关的
   定性效应，不是噪声——三个 condition、15个replica 一致地显示 original 偏向VAL、DEXP核
   打开SER、(14,5)最彻底(§8.5)。
5. 但**目前没有任何一个 condition 的 occupancy 数值是可信的平衡量**——committed-state
   分析确认全部15个replica都未通过平衡性判据(单basin锁定/前后半段漂移/n_eff过低/不同
   初始态不收敛，§8.5)。第4.2/4.3节报的具体百分比、以及§5"哪个核更接近original"的量化
   判断，在拿到真正收敛的数据前都不成立。

**核心转变**：这条调查线最初问的是"DEXP(14,5)的势函数形状是不是有缺陷、能不能修"，
现在的答案是——势函数形状本身大概率不是（或已经无法再靠更复杂的修正函数形式解决）
问题所在；真正悬而未决的是一个采样问题，不是势函数设计问题：**我们还不知道配体在
这个口袋里、在每一种力场变体下，真实的平衡氢键行为到底是什么**。

**因此**：
- 不要现在就在 (14,5) 和 (12,6) 之间做取舍——两个方向目前都缺乏收敛数据支持。
- 下一步最高价值的投入是 §8.5 末尾指向的双初始态(VAL-dominant / SER-bifurcated-dominant)
  MD 设计，先跑 5ns 检查两种初始态的后半段分布是否汇合，而不是继续分析现有数据或
  继续打磨 DEXP 核的函数形式。
- §8.2 记录的 MACE 环境原子级截断问题仍然是一个未验证但合理的独立关切，优先级低于
  上面的双初始态实验，可以并行准备。

### 9.1 分阶段 V/S/B 多初态平衡 MD 方案（用户 2026-07-13 提出，尚未开始）

Phase 1-3(生产等价性/环境收敛/force-torque-Hessian，见 §7/§10.6/§10.7)全部结案后，
用户重新梳理了"该用哪个核"这个问题的现有证据表：

| 问题 | 当前答案 |
|---|---|
| 谁更贴近 MACE 局部能量/力/曲率？ | (14,5) |
| 谁更接近 original LJ 的平衡构象分布？ | 不知道，现有 15 条 replica 都未平衡(§8.5) |
| 谁更接近真实体系？ | 缺少实验或独立参考 |
| r0 是否需要移动？ | 尚无跨-anchor/跨体系稳定证据 |

关键点：(12,6) 和 (14,5) 共用同一个 r0，但两个核都产生了同方向的 VAL/SER 重排——说明
这个现象主要不是 alpha/beta 能解决的，继续精扫 (14,5) 小数点填不上这个采样缺口。

**MD 方案（分阶段，不一次盲目承诺长时间）**：

- 3 conditions(original/DEXP(12,6)/DEXP(14,5)) x 3 initial states(V/S/B) x 2 independent
  replicas x 首轮 5ns = 90ns。每个 condition 用完全相同的 V/S/B 初态构造，仅随机速度不同。
  每 1ns 做一次中期分析（复用 §8.5 `--hbond-committed-state-dynamics` 的 occupancy/
  committed转移次数/n_eff/前后半段漂移诊断，不需要新代码）。
- 通过条件（同一 condition 内）：①从V/S/B出发的后半段occupancy汇合；②committed
  V/S/B/N 的block average稳定；③有足够有效样本量(不是阈值附近抖动)；④多次真正的
  basin往返；⑤不同初态之间的occupancy差异落入统计区间。
- 5ns 后的分支：已汇合→停止不浪费算力；接近汇合→延长到10ns；仍锁在不同basin→不要
  继续暴力延长，改用针对q_VAL/q_SER、SER177 rotamer的增强采样。
- **最终裁决树**：(12,6)与(14,5)平衡分布统计相同→选(14,5)(已有显著MACE
  Hessian/even优势)；(12,6)明显更接近original而(14,5)显著改变氢键网络→若目标是保持
  原力场端点则选(12,6)；两种DEXP都同样偏离original→说明问题不是alpha/beta，不能靠
  (14,5)精扫解决；有实验结构支持SER/B状态→original LJ也不一定是真值，不能因为(12,6)
  更像original就自动选它(避免"更像旧力场=更对"的循环论证)。
- **实现缺口（2026-07-13 已补上，尚未运行）**：新增两步流水线，都是零新增MD/MACE的
  纯读取+一次全新MD：
  1. `dexp_experiment.py::run_vsb_frame_scan`（`--vsb-frame-scan`，
     `--vsb-source-labels`默认扫`replica_original,replica_dexp_12_6,replica_dexp_14_5`
     全部15条已有replica，`--vsb-source-max-replicas`默认10，`--vsb-replicas-per-state`
     默认2）——只读，不需要新MD/MACE。新增`_classify_vsbn_frames`(跟
     `run_hbond_switching_dynamics`完全同款的V/S/B/N判据，抽成独立函数)逐帧分类现有
     轨迹，为V/S/B三态各挑2个起始帧：按该帧所在连续同态run长度降序(离状态切换边界越
     远越好，不是刚好卡在一次抖动上)，且强制不同候选来自不同source replica(增加起始
     构型多样性)。产出`vsb_frame_manifest.json`(每个候选帧的坐标/box直接存进JSON，
     不只存路径+帧号，避免下一步还要重新解析大DCD)+`vsb_frame_scan_all_candidates.csv`
     (完整审计轨迹)+对应的`.pdb`(人眼核查)。
  2. `dexp_experiment.py::run_vsb_staged_replica`（`--vsb-staged-run`，复用已有的
     `--replica-condition`）——读取上面的manifest，对给定的单个condition
     (original/dexp_12_6/dexp_14_5)，把V/S/B三态的起始帧当独立初始构型，各跑一次
     全新的`run_stability_simulation`(同一套minimize+softstart分段升vdW/Coulomb+
     production流程，直接复用`--replica-run`已经在用的机制，只是起始positions/box
     换成V/S/B帧、velocity依旧是全新Maxwell-Boltzmann抽样)。DEXP条件下system对每个
     起始帧单独重建(避免猜测`SurrogateSystemBuilder.build_surrogate_system`的
     reference_positions是否对起始构型敏感)。输出到
     `output_dir/vsb_staged/{condition}/{state}/rep{i}/`，每1ns的中期分析直接对
     增长中的traj.dcd跑`--hbond-committed-state-dynamics`风格分析即可，不需要额外代码。
  实际提交：先跑一次`--vsb-frame-scan`(建manifest，只需一次)，再分别提交3个
  `--vsb-staged-run --replica-condition {original|dexp_12_6|dexp_14_5}`
  (`--stability-replicas`此处不生效，起始构型数量由manifest里的2个V/S/B帧决定，
  不受这个flag控制)——对应§9.1原方案"3 condition x 3初态 x 2 replica"里的3个可独立
  提交作业，跟原来`--replica-run --replica-condition=X`的提交习惯一致。**注意默认
  `--platform CPU`**(跟`--replica-run`共用同一个CLI默认值)，跑真MD务必显式加
  `--platform CUDA`，否则GPU级任务会静默退化成CPU跑，5ns量级可能撞上walltime。

  **断点续跑（2026-07-13 补充，sub-run粒度）**：重新提交同一条命令，若某个
  (state,rep)的`summary.json`已存在(代表那次`run_stability_simulation`已完整跑完)，
  会直接读现成结果跳过，不重新跑——被kill后重新提交只补跑没做完的sub-run。单次sub-run
  内部(例如5ns跑到一半被杀)目前仍没有OpenMM checkpoint级别的续跑，会从step 0整个重来；
  这是`run_stability_simulation`(`--replica-run`也在用)本身的限制，评估后认为GPU上
  5ns量级一般不会长到撞上常见walltime，暂不需要补。

**旁线实测（2026-07-13）：`--r0-scale-diagnostic` 结案，r0 不需要移动**。固定(14,5)，
扫描`s_r∈{0.96,...,1.04}`(步长0.01)，20-anchor全量`--perturb-scan`缓存：

- **even 在 s_r=1.0 处有清晰、尖锐的最小值**(2.82 kJ/mol)，往任一方向偏移都单调、且
  多数幅度下显著变差(s_r=1.04→9.88，s_r=0.96→8.33)——符合预期，(a)步的LJ-matching
  本来就是把r0精确对齐LJ阱位置。
- **odd 在整个扫描范围内没有一个 s_r 显著改善**(全部`odd改善显著=False`)；s_r=0.99附近
  odd值最低(6.07 vs baseline 6.11)但bootstrap CI跨零(-0.35,0.16)，是噪声，不是信号；
  s_r=1.01及扫描两端反而显著更差。
- **LOAO 20/20 折一致选中 s_r=1.0**——不是多数，是全票。
- 满足全部四条判据的s_r：**无**，`promotable_to_md_condition=[]`。

**结论**：跟§6.8(contact-type/角度/Gaussian宽度修正均无信号)是同一模式的第三次确认——
alpha、beta、现在加上r0，三个analytic kernel参数全部测过，没有一个能动odd残差；
且(12,6)/(14,5)共用同一r0仍产生同方向VAL/SER重排，进一步印证这不是核参数问题。
**这把唯一剩下的、有实际产出可能性的方向收窄到了§9.1主线(V/S/B多初态MD)，不再有
"继续调kernel"这条备选路径**——r0维持LJ-matched(s_r=1.0)不变，不再是下一步方向。

**旁线设计（供跨体系复用参考）**：`--r0-scale-diagnostic`(已实现，见 §12 索引)——
固定(14,5)，扫描`r0_ij(new)=s_r*r0_ij(LJ)`，`s_r∈[0.96,1.04]`。只有同时满足
odd显著改善+even不退化+anchor-balanced LOAO稳定+bootstrap排除s_r=1，才把该s_r当作
候选，加入MD而不是直接替换生产默认值——Atenolol上这四条无一满足。

**"让MACE老师最后签字"：`--alpha-beta-scale-diagnostic`（已实现，尚未运行，见§12索引）**。
r0确认不需要移动之后，用户提出的最后一步不是继续找新的(alpha,beta)小数对，而是**确认
(14,5)位于一个稳定宽阔的最优盆地**：固定r0_scale=1.0/s_epsilon=1.0，全网格扫
alpha∈[12,16]步0.25、beta∈[4,7]步0.25(约束alpha>beta)，用
`p=alpha*beta`(曲率)/`q=alpha+beta`(高阶形状)重新参数化检验§3.3"对角谷"假说在更细
网格下是否仍成立，anchor-balanced LOAO(odd不设为必须判据)，只对grid最优/(14,5)/(12,6)
这3个候选做bootstrap CI(不对全网格做，省算力)。晋升为需要force/Hessian深度复核的候选
需同时满足：even上bootstrap显著优于(14,5)、改善不止是<5%的数值精修、LOAO多数折一致——
否则保留(14,5)。

**实测（2026-07-13）**：grid最优(12.5,6.5)只比(14,5)在even上改善3.1%，bootstrap CI跨零，
不显著——**判定：保留(14,5)，谷底宽阔，(14,5)确认位于稳定最优盆地**。但过程中发现一件
更值得记录的事：**q=alpha+beta=19(过(14,5))和q=18(过(12,6))根本不是同一条山谷**——
0.25网格下(12,6)自己所在的q=18线，其网格内最优点even RMSE(5.069)已经是q=19线最优点
(2.730)的近2倍，且bootstrap确认(12,6)vs(14,5)的even差异显著(2.20\[1.61,2.78\])。这
比"alpha+beta≈19是一条宽平山谷"的旧表述(§3.3)更精确：山谷相当窄，紧贴q=19，q=18已经
明显偏离。用户随后要求专门沿q=18/19两条线做更细的扫描并画图直接可视化这个差距——
`--alpha-beta-ridge-scan`+`plot_alpha_beta_ridge_scan.py`（已实现，尚未运行，见§12索引），
默认步长0.05(比0.25网格细5倍，"贵一点无所谓")，产出两面板PNG，每个面板一个q值，
even/odd RMSE双曲线+命名核位置标记，两面板共享y轴范围方便直接目视比较两条脊线的
绝对高度差。

**加宽网格复测（2026-07-13，α∈[9,19]/β∈[2,10]步长0.1，8115组合，"算力够"版）**：
grid最优仍是(12.5,6.5)(even=2.730)，(12,6)even=5.069跟窄网格完全一样——**确认(12,6)
在q=18这条线上的劣势不是原来12-16窗口的边界截断假象，即使把窗口大幅放宽，(12,6)仍是
q=18线上能找到的最优点，且仍然远差于q=19**。5%盆地(n=194)PCA给出`alpha+beta≈19.004`，
`q标准差=0.134`，最紧/最松方差比=0.0013(谷极窄极长)。**一个值得记录但不改变结论的
方法论细节**：LOAO在这个更细的网格下多数折(19/20)选中q=18.9而不是q=19.0——这不是
新矛盾，是LOAO用"19个训练anchor各自per-anchor RMSE的平均"打分，跟"grid最优"用的
"全部残差池化后算一个RMSE"是不同的统计量，两者没有义务给出完全一致的argmin；且
q=18.9仍落在q=19.004±0.134这同一个盆地内(1个标准差以内)，LOAO本身15%的低一致率
(散布在十几个q≈18.9附近的近邻点上)反而是"精确的alpha/beta分割由数据本身不可辨识，
只有q这个和被约束住"这个结论的又一个佐证，不是反例。

**现状小结（用户2026-07-13概括）：MACE-fidelity 和"物理结构友好度"看起来像两个
"超级优解"，但证据质量并不对等**——MACE-fidelity这一侧（(14,5)/q≈19山谷）已经被
alpha、beta、r0三个自由度独立扫描过，每一次都是同一个答案，没有再靠核参数调整挖掘
新信息的空间了；但"(12,6)更接近original、对物理结构更友好"这一侧的唯一证据来自§4的
replica MD比较，而§8.4/8.5已经证明那批replica全部未平衡——**这个证据本身还没有在
可信的数据上被验证过，不是跟MACE-fidelity同等分量的、已经证实的对立结论**。所以严格说
不是"两个都验证过的优解，必须二选一"，而是"一个已充分验证的结论 vs 一个尚未验证、
可能站得住也可能不站得住的说法"。这正是为什么下一步仍然是§9.1主线(V/S/B分阶段MD)，
而不是继续在核参数空间里找答案——只有等V/S/B平衡数据出来，才能知道"(12,6)更接近
original"这件事本身是不是真的。

### 9.2 §9.1 主线实测结果（2026-07-14）：original 大致汇合，两种 DEXP 都未汇合且彼此几乎同一模式

`--vsb-frame-scan`→3×`--vsb-staged-run --replica-condition {original|dexp_12_6|dexp_14_5}`
(3 condition × V/S/B × 2 replica × 5ns，全部18个sub-run跑完) 后，补上了流水线里缺的第三步
`--vsb-staged-analyze`(只读，复用`_classify_vsbn_frames`同款V/S/B/N判据，见 §12)。核心产物
是"固定condition，从V/S/B三个不同初态出发，后半段(2.5ns之后)occupancy是否收敛"：

| condition | 起始V→后半段occ(V/S/B/N) | 起始S→后半段occ | 起始B→后半段occ | 跨起始态spread(V/S/B/N) |
|---|---|---|---|---|
| original | 0.56/0.10/0.15/0.20 | 0.88/0.00/0.00/0.12 | 0.58/0.06/0.24/0.11 | 0.32/0.10/0.24/0.09 |
| dexp_12_6 | 0.17/0.15/0.05/0.63 | 0.09/0.50/0.13/0.28 | 0.10/0.47/0.11/0.32 | 0.08/0.34/0.09/0.35 |
| dexp_14_5 | 0.14/0.14/0.04/0.68 | 0.14/0.38/0.20/0.27 | 0.11/0.40/0.20/0.28 | 0.03/0.26/0.16/0.41 |

（逐frame的`n_state_changes`普遍很高，100-338/500帧，`single_irreversible_transition_from_intended_state`
几乎全部False——所以这不是"跳一次就再也不回头"的单次不可逆事件，轨迹内部一直在V/S/B/N间
高频flicker；但即便有大量flicker，**净occupancy仍然强烈依赖起始态**，说明5ns还不足以让
系统"忘记"从哪个初态出发。）

**对照 §9.1 的通过条件（①后半段occupancy汇合）**：
- **original**：三个起始态方向一致——V始终是主导态(56-88%)，S始终≈0，只是V的具体比例有
  量化差异。判定为"接近汇合"（qualitatively汇合，quantitatively仍有drift），落在§9.1
  "5ns后分支"里的"接近汇合→可以延长到10ns确认"这一档，不是彻底未汇合。
- **dexp_12_6 / dexp_14_5**：两者都完全没有汇合，且失败的方式几乎一模一样——从V起始，
  两个DEXP核都drift到以N(不成氢键)为主导(63%/68%)，V本身反而萎缩到14-18%且再也没有
  显著恢复；从S或B起始，两者都收敛到S为主导(38-51%)。也就是说：**两个DEXP核在这个
  实验里彼此的行为几乎不可区分**——都表现出"一旦离开V就回不去，容易被推向N或S"，
  与alpha/beta从(12,6)换到(14,5)无关。落在§9.1"5ns后分支"里的"仍锁在不同basin"这一档。

**对最终裁决树的影响**：
- 直接印证 §9.1 结尾"现状小结"里已经预告的方向——`"(12,6)更接近original"`这个说法
  建立在§8.4/8.5确认为未平衡的旧replica上，本来就不该信；现在有了刻意构造V/S/B多初态
  的新数据，结论不是"(12,6)更接近original"，而是**两个DEXP核都没有表现出接近original
  的平衡行为，且彼此无法区分**——不支持"选(12,6)因为它更保守/更像原力场"这个理由，
  因为(12,6)自己也没有在这套实验里表现得更接近original。
- 同时也不是"两个DEXP都同样偏离original，问题不是alpha/beta"这条裁决分支的简单确认，
  而是比它更具体：偏离的具体现象(V不可逆地让位给N/S)在(12,6)和(14,5)之间高度一致，
  跟 §9.1 已经从r0/alpha/beta三个核参数自由度独立测过的"这个odd/VAL-SER重排现象不是
  核参数问题"结论完全吻合的第四次独立确认。
- 按 §9.1 的分支规则："仍锁在不同basin→不要继续暴力延长，改用针对q_VAL/q_SER、
  SER177 rotamer的增强采样"——两个DEXP条件已经落入这一档，暴力延到10/20ns大概率
  还是锁在各自起始basin，下一步如果还想在DEXP侧继续深挖，方向是umbrella
  sampling/metadynamics沿一个显式的VAL-vs-SER配位数或SER177侧链rotamer二面角，
  不是再跑更长的plain MD。`original`可以考虑单独延长到10ns确认"接近汇合"是否收紧，
  因为它离汇合门槛比两个DEXP近得多，性价比更高。

**尚未做但如果要更严谨会做的补充**：本分析用的是`--hbond-switching-dynamics`同款的
逐帧距离+角度硬阈值判据，没有套 §8.5 `--hbond-committed-state-dynamics` 的 Schmitt
trigger 去抖动；上面"100-338次/500帧"的状态切换数很可能相当一部分是阈值附近的快速
抖动而非真正的basin穿越，如果要更干净地报告"多少次真正的basin往返"，应该对这批
vsb_staged轨迹补跑一次committed-state风格的重新分类（复用`_classify_vsbn_frames`
+committed-state去抖动逻辑，两个函数目前还没有被写成同一条流水线）。但这不影响
上面关于跨起始态occupancy不收敛的结论——那是净结果统计量，与是否去抖动无关。

### 9.3 现状总结（2026-07-14）：Atenolol 单体系核参数调优已经穷尽，下一步收窄到换体系

综合 §3（alpha/beta）、§6（contact-type/角度/Gaussian宽度修正）、§9.1（r0-scale、
alpha-beta-scale/ridge）、§9.2（V/S/B多初态MD）——**四条相互独立的探测路径**（核形状
参数、非核参数化的修正函数、平衡采样验证）**全部收敛到同一个结论：VAL→SER/N这个
odd残差驱动的氢键重排，不是DEXP核的参数选择或函数形式问题**，且不存在能同时解决它
又不牺牲MACE-fidelity的额外自由度。同时 §9.2 拿掉了(12,6)"更接近original、更保守"
这个此前唯一支持它的论据（未平衡数据上的旧结论）——两个DEXP核在真正受控的V/S/B多
初态实验里表现几乎不可区分，都未汇合，都是同一种失败模式。**这意味着在 Atenolol
这一个体系上，继续调DEXP核参数或者继续摸V/S/B/氢键这条线，已经没有更多能改变
(12,6) vs (14,5)取舍的新信息可挖**——这条调查线本身已经结案，不是暂停。

**因此，剩下有实际信息增量的方向只有两个，且优先级不对等**：

1. **（首要）换体系：§11 冻结协议的多体系 `--mace-kernel-benchmark`**。目前"DEXP全面
   优于LJ + (14,5)位于稳定宽阔最优盆地 + even/Hessian上(14,5)显著更优"这一整套结论
   只在 Atenolol 单一配体-口袋体系上验证过（§10.3/10.4/10.6/10.7），是否泛化到
   §11.3 列的正交化学多样性轴(氢键密度/芳香堆积/卤键/口袋极性-埋藏程度)完全未知。
   这是唯一还没被问过的问题，价值明显高于继续在Atenolol上打磨。
2. **（次要、可选、非阻塞）VAL/SER增强采样**：umbrella sampling/metadynamics沿显式
   VAL-vs-SER配位数或SER177侧链rotamer二面角，回答"DEXP作为一类势为什么系统性地
   偏离VAL偏好"这个物理机制问题——但按§9.1/9.2的裁决树，这不会改变(12,6)vs(14,5)
   的取舍(两者已确认表现一致)，只是满足对机制本身的理解，不是决策阻塞项。

**默认动作**：如果现在必须拍板生产用哪个核，(14,5)是唯一还有独立证据支持的选项
(MACE-fidelity的even/Hessian优势，§10.6/10.7/9.1)；(12,6)已经没有剩余论据。此前
§9.1"因为两条方向都缺乏收敛数据支持，不要现在做取舍"这句暂缓判断，在§9.2数据出来后
不再成立——不是因为(12,6)被证明更差，而是因为它唯一的支持论据(更接近original的
平衡分布)已被同一批数据证伪。真正被实际推迟的自由能生产计算(§9.1开头提到的"首轮
5ns=90ns...自由能计算都往后放")可以考虑以(14,5)为默认重新排上日程，除非要先做上面
的换体系验证。

---

## 10. 重新定框架：MACE 是参考曲面，DEXP/LJ 是描述它的两种解析语言（2026-07-12/13）

§9 写完之后，用户进一步指出 §9/§7 的"original↔surrogate endpoint 是否需要精确恢复"这个
问题问错了方向——如果目标从来就是构建一个新的替代势（不是恢复原始力场 endpoint），
那么 original LJ/PME 只是比较基准，不是热力学目标，不需要强行 morph 回去。真正要厘清的
是 MACE 在整个方法论里到底是什么角色。

### 10.1 当前势的准确名字

生产 Hamiltonian 里 MACE **不参与任何一步 MD**（`SurrogateSystemBuilder.build_surrogate_system`
从未添加 MLPotential/MACE force，唯一的 `MLPotential` 用法在 `scan_boresch_1d_pes`，是无关的
真空 torsion 扫描工具，已核实）。MACE 只在离线阶段给 `E_MACE,int = E_MACE(L+E)-E_MACE(L)-E_MACE(E)`
（`abfe_core.py:4440`附近），用来给 DEXP 的两个自由参数 alpha_vdw/beta_vdw 挑值（§3）。
`r0_ij`/`eps_ij` 来自原始 LJ 组合律，Gaussian 宽度固定不拟合，contact/angular 修正都没学进去
（§6 已确认学不进去）。**因此不应该叫"MACE force field"或"MACE-distilled potential"**，
准确名字是：

> **MACE-informed DEXP analytic projection**（MACE 引导的 DEXP 解析投影），
> 或强调结构时叫 **MACE-regularized DEXP kernel**（MACE 正则化 DEXP 核）。

### 10.2 DEXP 的任务被重新、更合理地限定

MACE 提供的是一张复杂、带多体细节和局部"毛刺"的参考势能面：

```
E_MACE = E_smooth_radial + E_structured_manybody + E_unresolved
```

DEXP 的任务不是复现整张曲面，只是要比 LJ 更好地投影 `E_smooth_radial`（光滑、有界、
径向对称的部分）——不要求捕捉多体/角度/anchor-specific 细节（`E_structured_manybody`），
也不要求消灭 `E_unresolved`（真正的高频噪声）。这样一来，§3.4 报告的"DEXP(14,5) even
RMSE 降 41-45%、odd 几乎不变"不再是部分失败，而是这个更朴素目标下的预期结果。

**〔2026-07-13 更正，见 §10.3/§10.4〕** 写本节时的假说曾以为 odd 是"任何各向同性
pairwise 径向核都结构性碰不到"的部分——**这个表述不准确，已被 §10.3/§10.4 的跨核族
实测推翻**：DEXP（相对 LJ）在 odd 上同样明显更优（33-58%，逐 anchor/逐幅度稳健），
不是只赢在 even。真正结构性碰不到 odd 的，仅限于 **DEXP 内部** 的 alpha/beta 调整
（§3.4——因为两个核共享同一个解析 `r0_ij` 且 `U'(r0)=0` 对任意 `alpha>beta>0` 恒成立，
这是 alpha/beta 这两个自由度本身的数学结构决定的，不是"pairwise 径向核"这整个函数
类别的结构性限制）。换成不同的核族（LJ→DEXP 的指数型 vs 幂律型）确实改变了 odd——
说明 odd 残差里至少有一部分是"局部形状/曲率误差在梯度上的投影"，换更优的径向形状
就能吃掉一部分，只是 DEXP 内部再怎么调 alpha/beta 也吃不到剩下那部分。

按这个框架，§9 提出的双初始态 MD、MACE↔DEXP thermodynamic morph、original-endpoint
correction、进一步的 contact residual 势函数扩展——都不再是当前最优先的方向（双初始态
MD 揭示的 VAL/SER 动力学本身仍然是真实、有意思的科学问题，但不再是"要不要修 DEXP"的
判据）。

### 10.3 最小试点实测（`--kernel-projection-benchmark`，2026-07-13）：DEXP 全面优于 LJ，超出预期

零新增 MACE 计算，只复用已有的 Atenolol `--perturb-scan` 缓存数据，把原始 pair-specific
LJ(K0) 也走一遍跟 DEXP(12,6)(K1)/DEXP(14,5)(K2) 完全相同的 anchor-relative 奇偶分解：

| | translation odd | translation even | rotation odd | rotation even |
|---|---|---|---|---|
| K0 LJ | 15.624 | 21.401 | 5.602 | 6.104 |
| K1 DEXP(12,6) | 6.611 (**-57.7%**) | 5.670 (**-73.5%**) | 3.732 (**-33.4%**) | 2.365 (**-61.3%**) |
| K2 DEXP(14,5) | 6.733 (**-56.9%**) | 3.142 (**-85.3%**) | 3.552 (**-36.6%**) | 1.390 (**-77.2%**) |

**结果比假说预期的更好，但需要修正表述**：假说原本预期 LJ 和 DEXP 在 odd 上应该同样差
（因为 odd 被认为是任何 pairwise 径向核都结构性碰不到的部分）。实测显示 DEXP 在 odd 上
也比 LJ 好 33-58%，不只是 even。**这不跟 §3.4 矛盾，而是两个不同层次的比较**：

- §3.4 比较的是 DEXP 内部的 alpha/beta（DEXP(12,6) vs DEXP(14,5)）——两者共享同一个
  LJ-matched `r0_ij` 和完全相同的解析形式(双指数)，且 `U'(r0)=0` 对任意 alpha>beta>0 恒成立，
  这是 odd 在**DEXP 内部**对 alpha/beta 不敏感的原因。
- 本节比较的是**跨函数族**（LJ 的 r⁻¹²/r⁻⁶ 幂律 vs DEXP 的双指数），即使两者在 r0
  处曲率相同（`alpha*beta=72` 精确匹配 LJ 在 r0 处的二阶导），离开 r0 之后的**形状**
  完全不同——而小幅刚体扰动恰好探测的是 r0 附近、离开 r0 的局部响应。实测说明 DEXP 的
  双指数形状在这个区域对 MACE 的局部匹配(不管是曲率/even 还是梯度响应/odd)都明显优于
  LJ 的幂律形状。

**正确表述**：换用 DEXP 这个函数族(相对 LJ)在 odd 和 even 上都是全面的改进；DEXP **内部**
调 alpha/beta 只能进一步压缩 even，压不动 odd——§6 反复确认无法用 contact-type/角度/
静电修正消灭的那部分 odd 残差(~6.6-6.7 kJ/mol)，是"已经换成更优函数族之后剩下的"，
不是"DEXP 完全没有改善 odd"。这让"DEXP 是 MACE 平滑径向骨架的更优解析投影"这个假说
在 Atenolol 这一个体系上得到了比预期更强的支持。

**仍然只是单体系试点结果**，不能代表跨体系普遍性。真正的普遍性验证需要按 §"下一步：
建立通用投影 benchmark"的设计，选 8-15 个化学多样体系跑 `--mace-kernel-benchmark`
（尚未实现），且要补上力/曲率/平滑性(r->0能量有限、短接触稳定性)对比，不能只看
odd/even energy RMSE 一个维度。

### 10.4 细粒度复核（v2，2026-07-13）：结论在每一项检验下都保持一致，Atenolol 单体系试点结案

用户要求把 v1 的聚合结果拆到每个扰动幅度、每个 anchor、robust统计量、switch敏感性、
MACE条件均值剖面、按距离分层、纯小幅扰动子集，逐一核实是否会改变结论。全部通过：

- **每个 (pert_type,magnitude) 档 (7档)**：DEXP 在 odd 和 even 上于每一个幅度都优于 LJ，
  含最小的 rotation:0.5° 和 translation:0.005nm——排除了"LJ 只在 0.04nm/3°最大扰动下
  爆墙"这个替代解释。
- **按 anchor 的整体 RMSE 胜负(20个anchor)**：LJ 0 胜、DEXP(12,6) 3 胜、DEXP(14,5) 17 胜——
  LJ 一次都没赢过。
- **K1(12,6) vs K2(14,5) 按 anchor 分开统计 odd/even 胜负**（本节最关键的证据）：
  **odd 上恰好 10:10 打平**——不是"大致接近"，是精确对半，20个独立anchor给出的证据
  直接证实"odd 没有统一胜者"；**even 上 19:1**，(14,5) 近乎一边倒地赢——"14,5 更擅长
  even"在anchor层面同样成立，不只是池化均值的假象。
- **加权/median/trimmed(10%) RMSE**：排序不变(LJ 16.1 / DEXP12,6 6.0 / DEXP14,5 5.2，
  trimmed后 LJ 仍有 8.6 vs DEXP 3.8-4.3)——不是被少数极端帧主导。
- **switch 敏感性**：三个核加/不加 switch 差异都在 0.01-0.02 kJ/mol，可忽略——把 Phase 1
  (§7)对 DEXP 的结论扩展到了 LJ。
- **MACE 条件均值剖面在合并SEM内的档数(共7档)**：LJ 1/7、DEXP(12,6) 2/7、**DEXP(14,5) 6/7**——
  这是本次复核里最有说服力的单项指标，直接衡量"分箱条件均值"而不是被噪声主导的原始RMSE，
  DEXP(14,5) 几乎全部命中，LJ 几乎全部不命中。
- **按 min_dist_anchor_nm 分三分位(近/中/远)**：DEXP 在三层里都全面优于 LJ——优势不是
  由少数极近接触的 anchor 撑起来的。
- **只用最小幅度子集(translation<=0.01nm, rotation<=1.5°)**：DEXP 仍明显优于 LJ
  (odd 2.2 vs 3.2，even 0.36-0.58 vs 1.26 kJ/mol)——直接排除"LJ 优势判断依赖大幅扰动"。

**结案表述**：DEXP vs LJ 在 Atenolol 这一个体系上，经过幅度/anchor/robust统计量/switch/
条件均值/距离分层/小幅子集七个独立角度的复核，结论保持一致且更加确定。(14,5) vs (12,6)：
even 上有稳固、anchor层面可验证的胜者(14,5)；odd 上确认没有统一胜者(10:10)。Atenolol
单体系试点到此结案，不再需要在这一个体系上继续深挖；下一步如果要验证普遍性，是独立的
8-15 体系 `--mace-kernel-benchmark`。

### 10.5 核心结论图（`plot_kernel_projection_benchmark.py`，2026-07-13）

零新增计算，纯读取 `kernel_projection_benchmark_summary.json` 画图，六联图：

- **Panel A/B**：按 7 档 (pert_type,magnitude) 分开的 odd/even RMSE，三核并列——对应
  §10.4 第一条复核。
- **Panel C**：overall 的 weighted/trimmed10%/median 三种 robust 统计量，三核并列——
  对应 §10.4 第四条复核。
- **Panel D**：逐 anchor 的 overall RMSE（log 轴，按 DEXP(14,5) 排序），标题里带
  LJ/DEXP(12,6)/DEXP(14,5) 的胜场数——对应 §10.4"按 anchor 整体胜负 0:3:17"。
- **Panel E/F**：逐 anchor 的 odd/even RMSE，只比较 K1 vs K2 的哑铃图（按差值排序）——
  对应 §10.4 最关键的一条复核，odd 10:10 打平、even 19:1。

输出：`output/dexp_experiment/kernel_projection_benchmark_core_figure.png`。这张图的
六个 panel 定义就是 §11 冻结协议里"标准报告"那一步的具体内容——新体系跑完
`--kernel-projection-benchmark` 后只需把脚本里的 `BASE` 指向新体系的 output 目录重跑，
不需要重新设计图。

### 10.6 Phase 3 实测（`--mace-residual-force-benchmark`，2026-07-13）：force/torque/Hessian 独立方法论，与 §10.3/§10.4 的能量RMSE结论互相印证

用完全独立于 §10.3/§10.4 的方法论（把 ±δ 能量差转成局部力/力矩投影，而不是比较 odd/even
energy RMSE）复核同一个问题，结论方向一致，且补出了新信息：

- **cosine similarity 不能区分三个核**：force_cosine(K0/K1/K2=0.754/0.779/0.779)、
  torque_cosine(0.648/0.622/0.632，LJ反而最高)——但 95% bootstrap CI(over 20 anchor)
  全部跨零(如 force_cosine K1-K0=0.024\[-0.043,0.092\])，说明这些差异**不显著**，
  方向相似度这个指标本身对三个核的区分力不足。
- **residual范数(方向+幅值联合)显著区分DEXP vs LJ，但区分不了(12,6)vs(14,5)**：
  force_vec_residual_norm K1-K0=-188.1\[-320,-68.6\]、K2-K0=-190.8\[-301.7,-82.3\]
  （两者都显著，DEXP比LJ更贴近MACE局部力向量），K2-K1=-2.7\[-27,22.5\]（不显著）；
  torque同款模式。
- **magnitude_ratio揭示了机制**：LJ系统性高估局部力幅值(mean/median=1.38/1.26)，
  DEXP(12,6)系统性低估(0.74/0.71)，DEXP(14,5)最接近1(0.87/0.82，三者里校准最好)——
  直接解释了为什么LJ在§10.3/§10.4的能量RMSE上输得最惨：其r⁻¹²排斥墙让局部力幅值
  比MACE真实响应大30-40%。
- **self-consistency(某个量自己是否是自洽线性向量场) K1最好(rmse=2.0)，LJ最差(24.1)，
  MACE target自己居中(7.6)**——但这只说明K1作为2体解析核天然最"线性"，不代表它最像
  MACE(MACE本身因为多体/各向异性有内禀非线性，rmse=7.6> K1的2.0是预期之中的)。
- **cross-model held-out(真正回答"谁更贴近MACE"的检验)**：kernel从3主轴重建的向量
  投影到MACE random-direction实测方向，vs MACE自己在该方向独立拟合的实测梯度——
  rmse: K0=417.8、K1=289.0、K2=295.6，corr: K0=0.672、K1=0.714、K2=0.709。
  bootstrap CI确认K1-K0=-119.8\[-197.8,-50.9\]、K2-K0=-114.1\[-183.9,-54.6\]都显著，
  K2-K1=5.8\[-5.4,17.4\]不显著——**DEXP(不论12,6还是14,5)在真正held-out的方向上
  显著更贴近MACE，比LJ的RMSE低约30%；(12,6)与(14,5)打平**，跟§10.3/§10.4靠完全
  不同方法论(energy odd/even RMSE)得到的"DEXP全面赢LJ，(12,6)vs(14,5)在odd上
  10:10打平"结论互相独立印证。
- **完整3x3平移Hessian(3主轴对角+随机方向解非对角，而不是简单的3维"曲率向量")**：
  mean Frobenius残差 K0=56598、K1=24823、**K2=16687**；mean特征值RMSE
  K0=31185、K1=12917、**K2=7189**——K2(14,5)在两个独立度量上都明显小于K1(12,6)，
  跟§10.3/§10.4"even上(14,5)19:1稳固胜过(12,6)"完全一致，且是用一个方法论上更严格
  的全Hessian比较(不是对角cosine)独立确认的。**已补 bootstrap CI(n_boot=10000，
  逐anchor配对，K0/K1/K2 共用同一组重采样anchor下标，与cosine/残差范数用同一套机制)**：
  按anchor胜场 Frobenius 0:1:19、特征值RMSE 0:1:19(K2全面碾压)；三组配对差值CI全部
  不跨零——K1-K0 Frobenius=-31775\[-39772,-24048\]、特征值=-18268\[-23315,-13572\]；
  K2-K0 Frobenius=-39911\[-47432,-32516\]、特征值=-23996\[-28715,-19341\]；
  **K2-K1 Frobenius=-8136\[-10272,-6088\]、特征值=-5728\[-6986,-4448\]，两者都显著为负**——
  (14,5) 在完整Hessian曲率上显著优于(12,6)，不是随机噪声，三个kernel间两两差异全部
  统计显著(LJ最差、K1次之、K2最好)。

**Phase 3 结论（已用bootstrap CI完全封口）**：跟§10.3/§10.4的能量RMSE结论完全一致且
互相独立印证——DEXP(不论12,6/14,5)在方向+幅值联合(残差范数)、真正held-out的力预测、
以及完整Hessian曲率上都显著优于LJ(全部bootstrap CI不跨零)；(12,6) vs (14,5) 在force
held-out/cosine上打平(CI跨零)，但在Hessian曲率(完整3x3，K2-K1的CI显著为负)上(14,5)
显著优于(12,6)，与"even上(14,5)赢"的既有结论完全吻合、且首次给出了严格的显著性检验。
单独看cosine similarity 会产生误导性的"LJ也不差甚至torque更好"的假象，因为cosine对幅值
误差不敏感；residual范数/cross-model held-out/Hessian 才是有区分力的指标。

### 10.7 Phase 2 实测（`--mace-env-convergence`，2026-07-13）：结论不依赖环境截断协议，Phase 2 结案

首跑就暴露了一个显存管理bug（同一 anchor 内 env_idx 随 radius/mode 变化 8 次，之前只在整个
anchor循环结束才清一次MACE Context缓存，导致同一anchor内最多8份、且逐步变大的显存同时
累积——用户当场指出，已修复为每个(anchor,radius,mode)条件用完立刻清）。修复后完整跑通
5 anchor x 4 半径 x 2 裁剪方式 = 520 次 `_compute_orb_decomposition`，全程无 OOM，最大条件
(anchor4, 0.90nm, residue_complete) 环境原子数 1485（同半径下 residue_complete 比 atomwise
多约 1.5-2x 原子，符合预期：atomwise 0.50→0.90nm 是 218→952，residue_complete 是
429→1485）。三项诊断结果：

- **target_convergence**：60个方向里 46 个(77%)相邻半径差值在缩小；0.90nm 相对当前生产
  0.50nm 的漂移 mean=0.425 kJ/mol、median=0.223、max=2.21 kJ/mol——比这批数据自己的
  odd RMSE(~1.0-1.6 kJ/mol)还小，更是远小于 §3.4/§10.4 那个 20-anchor/7-幅度 benchmark
  的 odd/even residual RMSE 量级(~3-8 kJ/mol)。**环境半径/裁剪方式的改变，不足以解释
  已有的odd/even残差结构**。
- **odd_sign_stability**：60个方向里 49 个(82%)符号稳定，11 个(18%)翻转。逐一检查这11个
  翻转方向的 `odd_by_radius` 具体数值后发现：其中约 8 个翻转发生在 odd 梯度本身就很小
  (|odd|<0.3-0.6 kJ/mol，例如 anchor1/atomwise/rotation/axis0 的四个值是
  -0.076/0.058/-0.064/0.032，纯粹是噪声围绕零点抖动，符号翻转没有物理意义)；只有 2-3 个
  (如 anchor0/atomwise/translation/axis0：-1.264→0.274→0.826→0.929；
  anchor1/atomwise/rotation/axis1：0.789→-0.681→-1.254→-1.136)是中等幅度(~0.7-1.3kJ/mol)
  且具体表现为"0.50nm 单独跟其余三个半径符号不一致"——这类值得留意，但幅度仍小于已知
  residual RMSE 量级，不改变整体结论。
- **kernel_ranking_stability**：**8个条件(4半径x2裁剪方式)里 K2_DEXP_14_5 全部胜出，
  `ranking_stable=true`，无一例外**——(14,5)是最优核这个结论跟环境半径/裁剪方式选择
  完全无关。

**Phase 2 结案**：§1-§10.6 建立在 0.50nm/逐原子裁剪 MACE 团簇协议之上的全部结论，都不是
环境截断的伪影——drift远小于残差尺度、kernel排序在8个环境定义下100%稳定、odd符号翻转
集中在噪声量级方向。Atenolol 这一条线的方法学验证到此完整（Phase 1 生产等价性 + Phase 2
环境收敛 + Phase 3 force/torque/Hessian），下一步是独立的 8-15 体系 `--mace-kernel-benchmark`
(§11)，不是继续在 Atenolol 单体系上打磨。

---

## 11. 冻结协议：面向未来多体系 `--mace-kernel-benchmark` 的统一方法论（2026-07-13）

Atenolol 单体系试点（§10.3-§10.5）用的方法论——扰动云生成方式、七项复核角度、核心图
六个 panel——在这里冻结成对任何新体系都应该原样套用的固定协议，避免每加一个新体系都
重新发明一遍验证方式（那样不同体系之间的数字会因为方法论漂移而不可比）。

### 11.1 每个新体系的固定流水线（零新增设计决策，只换体系）

实现层提供 `--debug-all` 一键入口，按依赖顺序执行本节协议及其正确性/深度复核：
`--perturb-scan` → `--production-equivalence-audit` → `--alpha-beta-scale-diagnostic` →
`--kernel-projection-benchmark` → `--mace-residual-force-benchmark` → 标准六联图 →
`--mace-env-convergence`。每一步状态、耗时和产物路径持续写入
`debug_all_summary.json`；任一步发生程序异常立即停止，科学判据未通过则保留结果并继续，供最终
人工裁决。该入口会重新生成 perturb-scan 缓存，环境收敛因成本最高固定放在最后。

1. `--perturb-scan`：固定用跟 Atenolol 试点相同的量级设定（±0.005~0.04nm 平移、
   ±0.5~3° 转动、7 档幅度、20 个 anchor 起步）——这些是刚体扰动的绝对物理量，不需要
   按口袋尺寸重新标定；env 半径/最大原子数等口袋筛选参数按各自体系的实际接触密度
   微调即可，但扰动本身的档位设计保持不变，否则跨体系的 odd/even 数字不可比。
2. `--kernel-projection-benchmark`（v2）：K0(原始 pair-specific LJ) / K1(DEXP 12,6) /
   K2(DEXP 当前默认，若后续体系发现更优 alpha/beta 则用该体系自己 LOAO 选出的值)
   三核对比，跑§10.4 那七项复核（逐幅度/逐anchor/robust统计量/switch敏感性/条件均值/
   距离分层/小幅子集），缺一不可——单看池化 RMSE 不能排除"被少数极端帧主导"等假象。
3. `plot_kernel_projection_benchmark.py`：跑一遍生成本体系的六联图（§10.5），作为
   每个体系的标准报告产物，人工判读前先看这张图。
4. 记录该体系的：DEXP vs LJ 的 odd/even 改善百分比（池化+按anchor胜负）、该体系自己
   LOAO 选出的最优 (alpha,beta)（若跟默认(14,5)不同，记录差多少、2D score surface
   是否也是对角谷退化）、K1 vs K2（或该体系最优核 vs (12,6)）在 odd/even 上各自的
   anchor 胜负比。

### 11.2 汇总后要检验的三个假说（按用户 2026-07-13 提出的框架）

拿到 N(=8-15) 个体系的上述记录后，只问三件事，不再重新设计新的统计量：

- **H1（DEXP family 是否普遍优于 LJ）**：在多数体系（建议门槛：>=75%，即 N=8 时至少
  6 个体系）里，DEXP（不管具体 alpha/beta）在 odd 和 even 上池化 RMSE 都优于 LJ，且
  按 anchor 的胜负比也明显偏向 DEXP（不是靠一两个体系的极端优势拉平均）。
- **H2（(14,5) 的 even 优势是否可迁移）**：(14,5) 相对 (12,6)（或体系自己 LOAO 选出的
  alpha/beta）的 even RMSE 优势，是否在多数体系里同样以类似量级、类似 anchor 胜负比
  出现，而不是 Atenolol 特有的巧合。odd 上是否在多数体系里同样保持"无统一胜者"。
- **H3（备选：DEXP family 普遍赢，但最优 alpha/beta 因体系而异）**：如果 H2 不成立
  ——即不同体系各自 LOAO 选出的最优 (alpha,beta) 明显不同（不落在同一条 alpha+beta≈常数
  的脊上，或脊的位置本身随体系漂移）——但 H1 仍然成立，就应该采纳这个更弱的结论：
  生产上不该有一个全局固定的 (alpha,beta)，而是每个体系/每类化学环境各自用同一套
  perturb-scan+LOAO 流水线现场选一次，DEXP 这个函数族本身才是可迁移的，具体数值不是。

**判定路径**：先看 H1 是否成立（这是最基础的必要条件，不成立就说明 Atenolol 的结果
是特例，DEXP 投影框架本身需要重新评估，不只是调 alpha/beta）；H1 成立后再看 H2 —— 
成立则维持单一全局默认核（生产更简单）；不成立则转向 H3（生产上接受"per-system 现场
选核"这个更麻烦但更诚实的方案）。

### 11.3 体系选择原则（不预设具体体系，留待用户按可用受体/配体库挑选）

冻结协议不预设具体测试哪些体系（那是数据可得性和研究优先级问题，不是方法论问题），
但选择时应该覆盖以下正交的化学多样性轴，而不是只堆数量：

- 配体净电荷（中性/阳离子如 Atenolol 本身/阴离子）；
- 配体极性基团密度（富含 H-bond donor/acceptor vs 以疏水侧链为主）；
- 芳香/π 堆积特征（口袋是否有显著芳香残基堆积接触）；
- 卤素（是否涉及卤键这类 LJ/DEXP 都不擅长的各向异性接触，作为"预期两者都表现一般"
  的压力测试，而不是预期 DEXP 会赢的场景）；
- 口袋极性/埋藏程度（深埋疏水口袋 vs 溶剂暴露的极性口袋）。

覆盖到 3-5 个正交轴、每轴至少 2 个体系，比堆到 15 个但轴上高度重复的体系更有信息量。

**额外建议：纳入 1-2 个"配体+纯水，无蛋白"体系作为廉价对照端点。** 这条协议本质上
只关心"DEXP 能否比 LJ 更好地投影任意 ligand-environment pair 的 MACE 光滑径向骨架"，
跟环境是不是蛋白口袋无关——纯水环境是这条化学多样性谱系上最简单、最均质的一端
（几乎全是 O/H donor-acceptor 接触），制备成本远低于蛋白复合物（不需要受体
准备/对接姿势/口袋筛选），且能把"跨体系是否泛化"和"跨蛋白口袋堆积方式是否泛化"
这两件事分开检验：如果 DEXP 在纯水这个最简单环境里都赢不了 LJ，那就是比任何蛋白
体系结果都更早、更便宜的止损信号，值得作为整批 8-15 体系里最先跑的一两个。但不能
用它替代蛋白-配体体系——真正决定生产用途（ABFE lambda-decoupling）的是口袋内的
各向异性接触/侧链堆积效应，纯水端点只覆盖化学多样性这一个轴，不覆盖"是否存在蛋白
挤压/构象约束下的多体效应"这另一个独立关切（呼应 §8.1 VAL/SER 竞争氢键网络需要
环境弛豫自由度这个发现——纯水环境里没有蛋白侧链 rotamer 这个自由度，测不到这类效应）。

---

## 12. 代码位置索引

- `abfe_core.py::DEXPSurrogatePotential` — pair-specific 解析核（当前默认 `alpha_vdw=14,beta_vdw=5`）
- `abfe_core.py::SurrogateSystemBuilder.build_surrogate_system` — 把 sigma/epsilon 喂给 DEXP CustomNonbondedForce
- `dexp_experiment.py::run_perturbation_scan`（`--perturb-scan`）— anchor-relative 局部扰动云生成
- `dexp_experiment.py::run_perturbation_fit`（`--perturb-fit`）— 分档等权 + LOAO + 2D score surface + PCA 对角谷检测 + 奇偶分解
- `dexp_experiment.py::_build_perturbation_distance_tensors` — 从 `run_perturbation_fit` 重构出的共享几何张量构建（距离/r0_ij/eps_ij/cutoff mask），供 `run_perturbation_fit`/`_contact_type_build_context` 共用
- `dexp_experiment.py::_bonded_hydrogens` — 从 system.xml 键连表里查一个重原子键连的所有 H 原子索引
- `dexp_experiment.py::_contact_type_build_context` — `run_contact_type_fit`/`run_contact_type_angular_diagnostic` 共用的几何/donor-acceptor角色/psi_o-psi_e特征构建（§6 最小实现的核心，只算一次）
- `dexp_experiment.py::_contact_type_ridge_loao` — 给定一个 ridge_lambda，跑一遍 M0/M1/M2 grouped LOAO，供单点拟合和 ridge_lambda 扫描共用
- `dexp_experiment.py::run_contact_type_fit`（`--contact-type-fit` / `--contact-type-ridge-lambda-grid`）— §6 donor_acceptor/fallback 最小版本 M0/M1/M2 对比，支持 ridge 稳健性扫描（见 §6.5/§6.6，2026-07-12 单点 ridge=10 实测未通过晋升判据）
- `dexp_experiment.py::run_contact_type_angular_diagnostic`（`--contact-type-angular-diagnostic`）— §6.6：对跨折 out-of-fold 残差做 D-H-A 角度/最近acceptor切换/配位数诊断，只诊断不拟合 angular force（2026-07-12 实测 any_stable_angular_signal_found=False）
- `dexp_experiment.py::_per_anchor_pearson_stability` — 从角度诊断重构出的共享按anchor相关性稳定性判据，供角度诊断/Gaussian宽度诊断共用
- `dexp_experiment.py::run_gaussian_width_diagnostic`（`--gaussian-width-diagnostic`）— §6.7：检验 M0/M2 残差是否与统一sigma_elec电荷穿透代理量存在稳定关联，只诊断不拟合（2026-07-12 实测 any_stable_signal_found=False，§6 正式结案见 §6.8）
- `dexp_experiment.py::run_production_equivalence_audit`（`--production-equivalence-audit`）— §7 Phase 1：核对 (14,5) DEXP 基线/§6.7 Gaussian 重实现与生产 OpenMM 是否一致，含原子身份核对+有限差分力+全量delta级核对（2026-07-12 实测通过，§3-§6 结论不受缺失switching function影响）
- `dexp_experiment.py::run_debug_all`（`--debug-all`）— §11.1 新体系一键诊断总控：重建 perturb-scan 后依次运行生产等价性、alpha/beta盆地、三核投影、force/torque/Hessian、标准六联图和环境收敛，持续落盘 `debug_all_summary.json`，异常即停、科学负结果不误判为程序失败。
- `dexp_experiment.py::run_kernel_projection_benchmark`（`--kernel-projection-benchmark`）— §10.3/§10.4：K0(原始LJ)/K1(DEXP12,6)/K2(DEXP14,5) 对 MACE odd/even 投影能力对比，含逐幅度/逐anchor胜负/robust统计量/switch敏感性/MACE条件均值剖面/距离分层/小幅子集七项复核，零新增MACE计算，`--mace-kernel-benchmark`(8-15体系版，尚未实现)的单体系试点（2026-07-13 实测：DEXP vs LJ全面胜出；14,5 vs 12,6在even上19:1稳固胜出、odd上10:10精确打平，Atenolol单体系试点结案）
- `dexp_experiment.py::run_mace_residual_force_benchmark`（`--mace-residual-force-benchmark`）— §11/§10.6 Phase 3（2026-07-13，**完全结案，含bootstrap CI**）：把 `--perturb-scan` 的 ±δ 能量差按 (anchor,pert_type,axis_kind,axis_index) 分组，用跨幅度线性+三次最小二乘拟合 e_odd(δ)=gδ+c3δ³ 取 δ→0 局部梯度 g（不是单点 e_odd/δ），force/torque projection=-g（旋转幅度先转弧度），e_even(δ)=hδ²/2+c4δ⁴ 给出曲率 h（拟合循环覆盖 target+K0/K1/K2+residual_*，见 `fit_qty_names`）；3 个平移/转动主轴的 lab-frame 单位向量从 `perturbed_lig_positions-anchor_lig_positions` 的刚体位移精确反解（不重新做特征值分解，避免退化/符号歧义）。含：residual_g=target_g-kernel_g 的线性性自检；force/torque cosine + 残差范数 + 幅值比(按anchor胜负)；**self_consistency_check**(某量自己是否是自洽线性向量场，K1(12,6)最线性但不代表最贴近MACE)与**cross_model_force_held_out_vs_mace**(kernel从3主轴重建投影到MACE random-direction实测方向，真正回答"谁更贴近MACE"，只覆盖force——`--perturb-scan`不生成随机旋转方向)分开报告；`curvature_profile_translation/rotation`(Hessian对角投影，非"三维曲率向量")+ 用3主轴对角+随机方向解出的完整对称3x3平移Hessian，比较Frobenius残差/特征值RMSE；kernel间两两(K1-K0/K2-K0/K2-K1)95% bootstrap CI(逐anchor配对重采样，K0/K1/K2共用同一组anchor下标，n_boot=4000覆盖cosine/残差范数/cross-model rmse，n_boot=10000覆盖Hessian两项)。字段命名严格区分 `local_target/kernel/residual_*_projection`(局部相互作用导数) 与真实体系净力。零新增MACE计算，只用已有 `--perturb-scan` 缓存。**实测结论（§10.6）**：cosine相似度不能区分三核(CI跨零)；残差范数/cross-model held-out/Hessian 三项独立指标下 DEXP 显著优于 LJ(CI全部不跨零)，(12,6)vs(14,5)在force held-out/cosine上打平、但在Hessian曲率上(14,5)显著更优(K2-K1 CI不跨零)——与§10.3/§10.4能量RMSE的"even上(14,5)赢、odd打平"结论完全独立印证。
- `dexp_experiment.py::_select_env_indices_residue_complete` — Phase 2 新增：半径近邻筛选出的原子扩展成完整残基/完整水分子(有一个原子落在半径内就整个残基全部纳入)，不补氢、不设 max_env_atoms 上限，跟现有 `abfe_core.py::_select_env_indices_from_mdtraj_frame`(逐原子裁剪，同样不设上限)对照使用，惯用法沿用 `abfe_core.py::OrbBoreschEstimator._build_pocket_context` 里已验证过的"遍历 top.residues 按 res.index 是否命中筛选"模式。
- `dexp_experiment.py::run_mace_env_convergence`（`--mace-env-convergence`，`--env-convergence-anchors`默认5，`--env-convergence-radii`默认"0.50,0.60,0.70,0.90"）— Phase 2（2026-07-13 **完全结案**，结果见 §10.7，**这条线唯一需要新增MACE计算的Phase**）：固定5个anchor、只用 `--perturb-trans-nm`/`--perturb-rot-deg` 里最小的一档幅度(3平移主轴+3转动主轴各±1，加anchor本身=每anchor 13个geometry)，在4个环境半径 x 2种裁剪方式(atomwise=现有生产协议 / residue_complete=新增的完整残基-水分子裁剪)下重新调 `_compute_orb_decomposition` 算MACE(约520次调用、约1560次实际MACE前向，量级与一次`--perturb-scan`相当)。三项诊断：①`target_convergence`——delta_e_target(定义与§3起一致：E_MACE_int-E_gauss_coul)是否随半径收敛(相邻半径差值是否变小)+0.90nm相对当前生产0.50nm的总漂移，跟已知odd/even残差量级(~3-8kJ/mol)对比；②`odd_sign_stability`——最小幅度下单点(ΔE(+)-ΔE(-))/2的符号是否随半径/裁剪方式翻转，直接检验§8.2"residual是否是MACE环境截断产物"这个假说；③`kernel_ranking_stability`——K0(LJ)/K1(DEXP12,6)/K2(DEXP14,5)的奇偶RMSE排序(5-anchor单幅度screening，样本量远小于§10.3/10.4的20-anchor版本，只判断排序是否稳健，不重新出具威信数字)是否随环境定义改变。输出 `env_convergence_diagnostics.csv`(长表，每行一个anchor×radius×mode×perturbation) + `env_convergence_summary.json`。**显存管理（用户2026-07-13当场指出的bug，已修复）**：env_idx在同一anchor内随radius/mode反复变化(8种)，`Orbv3DEXPFittingPipeline`的Context缓存键含env_idx，若只在整个anchor循环结束后清一次缓存(最初实现的bug)，会导致同一anchor内最多8份、且逐步变大(0.90nm+residue_complete最大)的显存同时累积不释放，多anchor跑下来必然OOM；修复为**每个(anchor,radius,mode)条件用完立刻清**：`pipeline._clear_orb_context_cache()`+显式`del gauss_ctx/mm_contexts/_e_target`+`gc.collect()`+可用时`torch.cuda.empty_cache()`，峰值显存只对应当前一个条件。
- `dexp_experiment.py::run_r0_scale_diagnostic`（`--r0-scale-diagnostic`，`--r0-scale-grid`默认"0.96,...,1.04"，`--r0-scale-alpha`默认14.0，`--r0-scale-beta`默认5.0，`--r0-scale-n-boot`默认4000）— 2026-07-13 用户提出的"旁线"（**已实测，结案，见§9.1：r0不需要移动**，零新增MACE计算，只用`--perturb-scan`缓存）：动机是(12,6)/(14,5)共用同一个LJ-matched r0_ij却产生了同方向的VAL/SER重排(§4.4/§8)，说明该现象主要不是alpha/beta能解决的，值得单独检验挪动r0本身。固定alpha/beta(默认当前生产核14,5)，把`r0_ij(new)=s_r*r0_ij(LJ)`（DEXP的`x=r/r0-1`归一化保证`U(r0)=-eps,U'(r0)=0`对任意r0都成立，这是良定义的单参数族），扫描s_r网格，对每个s_r重新算odd/even RMSE。晋升为第4个MD condition需同时满足用户指定的四条判据：①odd相对s_r=1.0基线bootstrap CI显著改善(逐anchor配对重采样，CI完全<0)；②even没有显著退化(只查RMSE，不是完整Hessian，真正候选建议之后单独跑`--mace-residual-force-benchmark`风格复核)；③anchor-balanced LOAO(留一个anchor验证，其余anchor选出的最优s_r)多数(>=50%)跨折一致；④以上均基于bootstrap而非点估计。输出`r0_scale_diagnostic_summary.json`，含`promotable_to_md_condition`列出通过全部四条的s_r。
- `dexp_experiment.py::run_alpha_beta_scale_diagnostic`（`--alpha-beta-scale-diagnostic`，`--ab-alpha-min/max/step`默认12.0/16.0/0.25，`--ab-beta-min/max/step`默认4.0/7.0/0.25，`--ab-scale-n-boot`默认4000）— 2026-07-13 用户提出的"让MACE老师最后签字"（**已实测**，零新增MACE计算，只用`--perturb-scan`缓存，r0_scale=1.0/s_epsilon=1.0全程固定，前提是§9.1旁线已确认r0不需要移动）：目的是**确认(14,5)位于稳定宽阔的最优盆地**，不是找新的小数对如(13.87,5.12)。两阶段设计避免对整个网格做昂贵操作：阶段1(全网格，零bootstrap，便宜)算每个(alpha,beta)组合的池化+按anchor odd/even RMSE，做anchor-balanced LOAO(按even RMSE选最优，odd不设为必须判据，因为§3.4已证实alpha/beta结构上只控制曲率/even)，并用`p=alpha*beta`(控制r0附近曲率)/`q=alpha+beta`(控制离开r0后高阶形状)重新参数化、比较沿这两个方向min-even-RMSE的标准差，检验§3.3"对角谷/alpha+beta≈19"假说在更细网格+更完整判据下是否仍成立；阶段2(只挑3个候选：grid最优/(14,5)/(12,6)，不是全网格)做两两bootstrap CI。晋升判据严格按用户原话：谷底宽阔且(14,5)在grid最优CI内→保留(14,5)；改善<5%→视为数值精修不改默认值；只有even上bootstrap CI显著优于(14,5)且LOAO多数折一致，才建议把新组合加成`--mace-residual-force-benchmark`的第4个命名核，跑force held-out+完整Hessian复核再最终拍板(本函数不重复那套机制)。输出`alpha_beta_scale_diagnostic_summary.json`，含`verdict.worth_force_hessian_deep_dive`布尔值和`recommendation`文字判定。**实测结果（2026-07-13）**：grid最优=(12.5,6.5)，even=2.730 vs (14,5)的2.818，只改善3.1%(bootstrap CI跨零，不显著)→`recommendation`="保留(14,5)"；LOAO多数折(55%)选中(13.0,6.0)，同样落在q=alpha+beta=19这条线上；**关键发现：q=19(过(14,5))与q=18(过(12,6))不是同一条山谷**——0.25网格下q=18的最优点(即(12,6)本身)even=5.069，是q=19最优点(2.730)的近2倍，bootstrap确认(12,6)vs(14,5)的even差异显著(2.20[1.61,2.78])。见 §9.1 追加段落与`--alpha-beta-ridge-scan`(下一条)。
- `dexp_experiment.py::run_alpha_beta_ridge_scan`（`--alpha-beta-ridge-scan`，`--ridge-q-values`默认"18,19"，`--ridge-beta-min`默认1.0，`--ridge-step`默认0.05）+ `plot_alpha_beta_ridge_scan.py` — 2026-07-13 用户追加要求（已实现，尚未运行，零新增MACE计算）：上面`--alpha-beta-scale-diagnostic`的0.25网格已经显示q=18和q=19不等价，本函数沿这两条(可配置)定值q直线做细得多的扫描(默认步长0.05，"贵一点无所谓")，逐beta(alpha=q-beta)算池化odd/even RMSE，输出长表CSV`alpha_beta_ridge_scan_by_point.csv`+`alpha_beta_ridge_scan_summary.json`(含每条脊线的网格内最优点、落在该线上的命名核(12,6)/(14,5)的具体指标)。`plot_alpha_beta_ridge_scan.py`零新增计算，纯读取生成两面板PNG(`alpha_beta_ridge_scan_core_figure.png`)，每个面板对应一个q值，左轴even RMSE/右轴odd RMSE双曲线+命名点(红色)/脊线最优点(灰色虚线)标记，两面板共享y轴范围以便直接目视比较两条脊线的绝对高度差。
- **完整2D热图（用户2026-07-13追加要求"落盘+画真正的二维热图，样本多一点，算力够"）**：`run_alpha_beta_scale_diagnostic`新增落盘`alpha_beta_scale_diagnostic_landscape.csv`(每个扫过的(alpha,beta)一行：odd/even池化RMSE，此前只嵌在JSON里)；新增`plot_alpha_beta_heatmap.py`(已实现，尚未运行)读取该CSV画真正的2D网格热图(不是1D脊线剖面)，两面板(even RMSE用log色标因为动态范围跨一个数量级以上、odd RMSE线性色标)，(14,5)/(12,6)白色菱形标出，q=18/19两条对角参考线叠加。想要更宽范围/更密采样(回答"q=18网格边界截断"那个问题)，用更宽的CLI参数重新跑`--alpha-beta-scale-diagnostic`再画图，例如`--ab-alpha-min 9 --ab-alpha-max 19 --ab-alpha-step 0.1 --ab-beta-min 2 --ab-beta-max 10 --ab-beta-step 0.1`（约数千个组合，仍是零新增MACE的纯NumPy计算，只是比默认221点跑得久一些）。
- `dexp_experiment.py::run_replica_condition`（`--replica-run`）— 单个 condition 的 N replica 跑法
- `dexp_experiment.py::run_replica_analysis`（`--replica-analyze`）— RMSD/聚类/接触/氢键/contact-feature协方差/平移转动/Δ<q>显著性检验
- `dexp_experiment.py::run_hbond_switching_dynamics`（`--hbond-switching-dynamics`）— §8.3：只读分析已跑完的 replica 轨迹，V/S/B/N 四态切换动力学(occupancy/转移矩阵/驻留时间/自相关有效样本数)，判断§4.4的occupancy是否是平衡概率（尚未运行）
- `dexp_experiment.py::run_vsb_staged_analysis`（`--vsb-staged-analyze`，2026-07-14）— §9.2：只读分析`--vsb-staged-run`产出的`vsb_staged/{condition}/{state}/rep{i}/traj.dcd`，同款V/S/B/N判据，核心输出是固定condition下V/S/B三个起始态后半段occupancy是否收敛（**已实测，见§9.2**：original接近汇合，dexp_12_6/dexp_14_5均未汇合且彼此几乎同一失败模式）
- `plot_dexp_vs_lj_vs_mace.py` — LJ/DEXP(12,6)/DEXP(14,5) 势能曲线 + 真实 MACE 局部扰动散点对比图（含 translation/rotation 局部精细图）
- `plot_kernel_projection_benchmark.py`（§10.5）— 六联图：分幅度odd/even RMSE + overall robust统计量 + 逐anchor overall RMSE(K0/K1/K2) + 逐anchor odd/even RMSE(K1 vs K2哑铃图)，零新增计算，纯读 `kernel_projection_benchmark_summary.json`；§11 冻结协议里每个新体系的标准报告产物
- `output/dexp_experiment/perturb_scan_summary.json` / `perturb_scan_diagnostics.csv` / `perturb_scan_geometry.npz`
- `output/dexp_experiment/perturb_fit_summary.json`
- `output/dexp_experiment/contact_type_fit_summary.json`（`--contact-type-fit` 输出，已生成一次单点 ridge=10 结果，含 ridge_lambda_sweep 结构等扫描重跑覆盖）
- `output/dexp_experiment/contact_type_angular_diagnostic_summary.json`（`--contact-type-angular-diagnostic` 输出，已生成，any_stable_angular_signal_found=False）
- `output/dexp_experiment/gaussian_width_diagnostic_summary.json`（`--gaussian-width-diagnostic` 输出，已生成，any_stable_signal_found=False）
- `output/dexp_experiment/production_equivalence_audit_summary.json`（`--production-equivalence-audit` 输出，已生成，Phase 1 通过）
- `output/dexp_experiment/kernel_projection_benchmark_summary.json`（`--kernel-projection-benchmark` 输出，已生成）
- `output/dexp_experiment/kernel_projection_benchmark_core_figure.png`（`plot_kernel_projection_benchmark.py` 输出，§10.5 六联图，尚未生成——脚本已写好，待用户运行）
- `output/dexp_experiment/replica_compare_summary.json` / `replica_compare_per_replica.json`
- `output/dexp_experiment/hbond_switching_dynamics_summary.json`（`--hbond-switching-dynamics` 输出，尚未生成）
