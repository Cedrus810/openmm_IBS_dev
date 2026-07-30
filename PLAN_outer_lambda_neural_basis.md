# 外层 λ 神经基势研发计划

## 1. 文档定位

本文定义“外层 λ 神经基势”（Outer-λ Neural Basis Hamiltonian）的研究与验证计划。
它是一份方法开发路线图，不代表相关模型、OpenMM 接口或 IBS 生产集成已经实现。

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

---

## 8. 分阶段研发路线

## 阶段 0：冻结基线与问题定位

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

目标：证明神经项能改善一个明确瓶颈。

工作：

- 只选择一种目标，例如 ligand torsion 或 cavity hydration。
- 构建与该目标对应的局部训练数据。
- 冻结训练后的 \(U_1\)。
- 扫描少量外层幅度，不在线更新模型。
- 比较基础路径、仅 λ 重排、单基势路径。

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
- 多基势相互高度相关，增加基势数量没有可测收益。
- 不同独立重复给出不一致的自由能或重组分布。
- 端点、公共 λ 状态或离线能量账本无法严格闭合。

“该体系不需要神经路径”是允许且有价值的研究结论。

---

## 12. 配置、缓存与可追溯性要求

未来若进入实现，每次运行至少记录：

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
\text{单个重组问题验证}
\rightarrow
\text{2--4 个共享神经基势}
\rightarrow
\text{完整 IBS/TMBAR 联合验证}
\]

只有当低秩神经基势无法表达所需的 λ 依赖势能面变化时，才考虑连续
λ-conditioned GNN。

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
