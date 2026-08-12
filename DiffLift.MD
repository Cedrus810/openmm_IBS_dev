# EXP-020 SoftLift：面向 Outer-λ 守恒路径势的可实现设计

> 文档版本：`0.3-active`  
> 状态：**EXP-020 已 sealed；D1/D2/reference-export preflight qualified；full D3 未通过；N0 full-system CustomGB 已停止；未授权 D4、production MD 或重新开启 WP-5**  
> 当前实验编号：`EXP-020`；`EXP-019` 保留为 sealed parent identity；不得修改 sealed EXP-012/019  
> 论文来源：`19021_Differentiable_Lifting_f.pdf`，ICLR 2026  
> 继承基线：EXP-012 的数据、Outer-λ 账本与验证工具；EXP-014 只作为冻结失败基线

> 阅读说明：下方“原始概念稿”保留为设计推导记录；文件末尾的“Part II：权威实现规范”是当前唯一实现依据。两者冲突时以 Part II 为准。

## Part I：原始概念稿（保留作历史背景，不作为实现契约）
<details>
<summary>展开历史概念稿</summary>


可以。下面我把它整理成一份**可以直接转成方法设计/新实验预注册的技术方案**。我会严格区分三层：

1. **DiffLift 论文原本做了什么；**
2. **为什么它对应你现在 EXP-012/014 的失败点；**
3. **怎样把它改造成适用于 Outer-λ Hamiltonian、能产生守恒力、并且有希望真正降低 online cost 的实现。**

核心结论先放前面：

[
\boxed{
\text{DiffLift 不应该原样塞进 MD。}
}
]

真正值得拿过来的是：

[
\boxed{
\text{先学习“哪些局部原子应该共同构成一个 interaction motif”，}
\quad
\text{再学习这个 motif 对 }B(R)\text{ 的贡献。}
}
]

这和 EXP-014 的区别是根本性的。EXP-014 学的是

[
\sum_{ij} f_{ij}(r_{ij}),
]

而新的方法应该学

[
\rho!\left(
\sum_j w_{ij}\phi(x_{ij})
\right),
]

也就是**先形成一个局部集合/软超边，再对整个集合做非线性 readout**。

而且我认为应该把实现拆成两级：

[
\boxed{
\text{Torch reference SoftLift}
\rightarrow
\text{native OpenMM density-lift}
}
]

前者回答“这个理论能不能保住 MACE signal”，后者回答“能不能真正解决 ESS/GPU-hour”。

---

# 1. 当前问题到底是什么

你的 Outer-λ 总体定义本身其实已经不需要改：

[
H_k^*(R)
========

H_k^{MM}(R)+A_kB_\theta(R),
\qquad
A_0=A_K=0.
]

MACE/神经基势不接收连续 λ，λ 只在 Hamiltonian 外层控制；MACE 现在属于离线 teacher，production 希望只留下一个廉价、可导的 student。

目前的数据给出了很清楚的三级漏损：

[
\text{MACE-OMOL latent}
\longrightarrow
\text{LocalResidualStudent}
\longrightarrow
\text{native radial compression}.
]

teacher-side cached latent readout 是 **3/3 folds 全改善，平均约 44.6% gap-variance improvement**；真正的 `LocalResidualStudent` direct-gap 只剩平均 **13.9348%**，而且 fold-level 只有 **2/3**。

student 不是完全没用：DEC-048 里三组 paired-reseed pilot 的 `mixture_ess_proxy` 都提高了，约 +10.6%、+10.7%、+27.8%。问题是计算成本变成 baseline 的约 **1.81–1.89×**，所以 ESS/GPU-hour 三组全跌。

然后 EXP-014 试图进一步压缩成：

[
\text{typed ligand/environment pair}
+
\text{quintic-C2 cutoff}
+
\text{radial RBF},
]

结果 8/16/32 radial bases 全部彻底失败，held-out (R^2) 从 (-11.46) 一路掉到极大的负值。

所以现在不是：

> “MACE 没信号。”

而是：

[
\boxed{
\text{rich representation 有信号，cheap pair representation 压不下来。}
}
]

这就是 DiffLift 能插进去的位置。

---

# 2. DiffLift 原论文的理论核心

论文的出发点是：传统图模型通常先人为规定 topology，然后模型只能在这个 topology 上学习。

DiffLift 把 topology 本身也变成 learned object。

论文的通用流程是：

[
G=(V,E,x)
]

先经过 GNN：

[
z_v = {\rm GNN}(G)_v,
]

得到 node embedding。

然后从节点中构造 candidate higher-order cells：

[
C\subseteq V,
]

对一个 candidate cell 做 permutation-invariant aggregation：

[
z_C
===

\bigoplus_{v\in C}z_v,
]

最后学习：

[
p_C=\phi(z_C)
]

表示这个 higher-order cell 是否应该存在。论文原始实现中再采样：

[
y_C\sim{\rm Bernoulli}(p_C),
]

并用 straight-through estimator 传梯度。

它的 hypergraph 版本更具体：

对于 node (v)，先在 embedding space 找 (k_v) 个邻居：

[
C(v)=\mathrm{kNN}(z_v;k_v),
]

而且 (k_v) 本身也是从 embedding 预测出来的随机变量。然后用一个 order-invariant set function：

[
\Psi({z_u:u\in C(v)})
]

决定这个 hyperedge 是否接受。接受后的 hyperedge feature 则用整个集合的 aggregate，而不是把 pair 独立看待。

真正值得我们偷的是：

[
\boxed{
\text{relation structure 本身是 task-dependent 的。}
}
]

不是它的 cell-complex/cycle 那部分。

---

# 3. 为什么 EXP-014 恰好属于“static lifting”

EXP-014 本质上固定了：

[
E=
{(i,j):
i\in L,,
j\in E,,
r_{ij}<r_c}.
]

然后每条 edge 独立贡献：

[
B_{\rm pair}(R)
===============

\sum_{i,j}
f_{Z_iZ_j}(r_{ij}).
]

这等于提前宣告：

> ligand–environment 的所有有效物理关系都可以分解成 pairwise radial functions。

假设某个 ligand O 周围同时存在两个 environment atoms (j,k)：

[
B_i=f(x_j)+f(x_k).
]

那么：

[
\frac{\partial^2B_i}
{\partial x_j\partial x_k}=0.
]

也就是说，**neighbor (j) 的存在永远不会改变 neighbor (k) 的意义。**

但如果改成：

[
B_i=
\rho\left[
\phi(x_j)+\phi(x_k)
\right],
]

则：

[
\frac{\partial^2B_i}
{\partial x_j\partial x_k}
==========================

\rho'',
\phi'(x_j)\phi'(x_k),
]

一般不再为零。

这就是最小的 many-body interaction。

所以 EXP-014 里把 RBF 从 8 加到 32，实际上只是让：

[
f(r)
]

变得更灵活。

但它始终没有得到：

[
f(r_j,r_k,\theta_{jk},\text{coordination},\ldots).
]

因此 EXP-014 的失败非常可能不是：

[
\text{radial resolution 不够},
]

而是：

[
\boxed{
\text{pair-additive functional family 根本不够。}
}
]

---

# 4. 对你的系统，DiffLift 应该怎样改造

这里开始已经不是论文原文，而是**针对 Outer-λ 的方法改造**。

我暂时叫它：

> **ligand-anchored soft lifting**

不是新论文名字，只是方便描述。

首先保留你已经验证好的动态环境 funnel：

[
\mathcal E(R)
=============

{
(i,j):
i\in L,,
j\in E,,
r_{ij}^{MIC}<r_c
}.
]

你已经有：

`ligand_environment_cross_edges`

以及 triclinic MIC、动态水、quintic-C2 cutoff 的实际测试；候选 edge 的离散 membership 在 cutoff 翻转，但能量权重在边界平滑归零。

所以**不要改 candidate discovery。**

变化从 candidate edge 生成以后开始。

---

# 5. 第一层：cheap pair representation

对每条 ligand-environment pair：

[
(i,j)\in\mathcal E
]

构造廉价 feature：

[
x_{ij}
======

[
e_i,
e_j,
R(r_{ij}),
q_i,q_j,\ldots
].
]

其中可以包括：

[
e_i,e_j
]

这种小型 atom-type embedding，

[
R(r)
]

是 radial basis，

固定 MM partial charge 可以作为 cheap scalar feature，但这是候选设计，不是 DiffLift 论文要求。

重点是：

[
x_{ij}
]

还**不是 energy contribution**。

这和 EXP-014 开始分叉了。

---

# 6. 第二层：把 DiffLift 的 accept/reject 改成连续 soft gate

原论文是：

[
y_C\sim{\rm Bernoulli}(p_C).
]

这个绝对不能原样进入 MD。

如果 topology 在两个非常接近的构型间发生：

[
y_C:0\rightarrow1,
]

那么：

[
B(R)
]

可能发生跳变。

更严重的是，如果每次 evaluation 还带 Bernoulli 随机采样，那么同一个 (R) 甚至不再对应唯一势能。

这与：

[
F=-\nabla_RB(R)
]

要求的确定、连续势能不兼容。

所以 MD 版本必须定义：

[
a_{ijm}(R)
==========

c_{\rm C2}(r_{ij})
,\sigma[s_m(x_{ij})],
]

其中：

[
0\le a_{ijm}\le1.
]

这里：

* (m) 是 motif/hyperedge channel；
* (s_m) 是极小的 scoring network；
* (c_{\rm C2}) 就是现有 smooth cutoff。

于是原论文的：

[
\text{accept / reject}
]

变成：

[
\boxed{
\text{continuous membership strength}.
}
]

---

# 7. 一个 ligand atom 可以形成多个 learned interaction motifs

例如对 ligand atom (i)，设有 (M) 个 motif channels。

那么：

[
C_{im}
======

{j:\ a_{ijm}>0}
]

只是概念上的 soft hyperedge；代码里根本不需要真的创建 hypergraph object。

每个 channel 做：

[
u_{im}
======

\sum_j
a_{ijm}\phi_m(x_{ij}).
]

同时记录有效 coordination：

[
n_{im}^{\rm eff}
================

\sum_j a_{ijm}.
]

所以一个 channel 可以自己学成：

* donor-sensitive；
* charged-contact-sensitive；
* hydrophobic/local packing；
* water competition；
* 其他 task-related local motif。

注意：**我们不需要人为给这些 channel 命名。**

它只是 latent relation。

---

# 8. 这是 DiffLift 最关键的改造：set-level readout

然后不是直接把 (u_{im}) 相加成 energy，而是：

[
h_{im}
======

\rho_m
\left(
u_{im},
n_{im}^{\rm eff}
\right).
]

于是对于一个 ligand atom：

[
h_i
===

[h_{i1},\ldots,h_{iM}].
]

最后：

[
b_i=\psi(h_i,e_i),
]

整个 ligand：

[
S(R)=\sum_{i\in L}b_i.
]

然后保持你原来的 bounded scalar：

[
\boxed{
B_\theta(R)
===========

B_{\max}
\tanh
\left[
\frac{S(R)-b_0}{B_{\max}}
\right].
}
]

所以整个 Hamiltonian 完全没变：

[
\boxed{
H_k^*(R)
========

H_k^{MM}(R)
+
A_kB_\theta(R)
}
]

以及：

[
A_0=A_K=0.
]

因此 physical endpoints、outer λ semantics、IBS/TMBAR ledger 的总体定义都不需要因为 DiffLift 改变。你的原计划明确要求神经项只是中间态 path term，并且必须进入完整 target Hamiltonian。

---

# 9. 为什么这个模型是 EXP-014 的严格扩展

这一点非常适合做方法学论文。

如果令：

[
M=1,
]

gate 不学习：

[
a_{ij}=c_{\rm C2}(r_{ij}),
]

并且令：

[
\rho(u)=u
]

是线性的，那么：

[
B(R)
====

\sum_{ij}
f_{Z_iZ_j}(r_{ij}).
]

也就是说，它自动退化回 EXP-014。

因此：

[
\boxed{
\text{EXP-014 是 SoftLift 的一个受限特例。}
}
]

这让实验设计非常干净。

你不是：

> “失败后换了一个完全不同的神经网络。”

而是在逐步解除一个明确的函数形式限制：

[
\text{pair additive}
\rightarrow
\text{set nonlinear}
\rightarrow
\text{learned topology}.
]

---

# 10. 第三层：如果仅 set aggregation 还不够，再加入廉价 angular information

只使用：

[
r_{ij}
]

仍然有一个问题。

两个局部环境可能拥有相同：

* neighbor types；
* neighbor distances；

但是方向完全不同。

因此可以加入**无需 environment-environment message passing 的局部几何 moments**。

例如不要用容易在 (r\to0) 处产生单位向量奇点的 (\hat r)，而直接使用 displacement：

[
\mathbf d_{ij}
==============

\mathbf r_j-\mathbf r_i
]

的 MIC 版本。

定义一阶 moment：

[
\mathbf M^{(1)}_{im}
====================

\sum_j
a_{ijm}f_m(r_{ij})\mathbf d_{ij}.
]

它是 equivariant vector。

二阶：

[
\mathbf M^{(2)}_{im}
====================

\sum_j
a_{ijm}f_m(r_{ij})
\mathbf d_{ij}\mathbf d_{ij}^{T}.
]

然后只把 rotation invariants 喂给 scalar network：

[
I_1=
|\mathbf M^{(1)}|^2,
]

[
I_2=
{\rm Tr}(\mathbf M^{(2)}),
]

[
I_3=
{\rm Tr}
[(\mathbf M^{(2)})^2].
]

于是：

[
h_{im}
======

\rho_m
(
u_{im},
n_{im}^{eff},
I_1,I_2,I_3
).
]

这已经能够识别不少：

[
\text{coordination + orientation + local packing}
]

差异，却完全不需要再跑一次 MACE-style environment-environment message passing。

---

# 11. 为什么它仍然给出严格守恒力

只要最终：

[
B_\theta=B_\theta(R)
]

是一个确定的、连续可导 scalar：

[
F_a^{\rm lift}
==============

-\frac{\partial B_\theta}{\partial R_a}.
]

由于 input 用的是：

* MIC displacement；
* distance；
* rotation invariant moments；
* permutation-invariant sums；

最终 energy 对：

* 平移不变；
* 旋转不变；
* environment atom 排序不敏感。

而由 scalar energy 自动微分产生的 force 会按旋转正确变换。

这实际上符合你原先 LocalResidualStudent 的设计哲学：第一候选就是廉价 rotation-invariant scalar energy，不需要为了得到 equivariant force 而直接上完整 irreps machinery。

---

# 12. 为什么不能用论文原来的 hard kNN/top-k

原版 DiffLift hypergraph 在 embedding space 做 kNN，并学习 (k_v)。

这个设计对 MD 不理想。

原因一是 continuity：

如果两个 neighbor 的 rank 交换：

[
j:k\leftrightarrow k+1,
]

hard top-k membership 可以瞬间改变。

原因二是成本。

论文自己给出的 hypergraph DiffLift complexity 包含：

[
O(N^2D)
]

的 embedding pair distances 和：

[
O(N^2\log k_{\max})
]

的 neighbor selection。

对你的 50k 原子系统显然不应该这么干。

所以 adaptation 必须是：

[
\boxed{
\text{physical cell list 生成 candidate}
\rightarrow
\text{learned soft lifting 只发生在 candidate 内}.
}
]

计算规模因此由：

[
N_{\rm system}^2
]

变成近似：

[
O(E_{\rm ligand-env}Md),
]

其中 (E_{\rm ligand-env}) 就是已经有预算限制的动态 cross edges。

你现有 student budget 已冻结到 S1 原子约 `≤256/320`、edge `1536/2048`、每原子 neighbor `64/80`，这正好是合理的 candidate pool 范围。

---

# 13. 第一版 reference implementation 应该非常简单

我会做一个很小的 PyTorch reference，而不是 TNN。

概念代码：

```python
class LigandSoftLiftBasis(nn.Module):
    def forward(self, xyz, box):

        # Existing audited dynamic funnel
        lig_idx, env_idx, disp, dist = ligand_environment_cross_edges(
            xyz, box, cutoff=self.cutoff
        )

        cutoff = quintic_c2_cutoff(
            dist,
            inner=self.inner,
            outer=self.cutoff,
        )

        # Cheap pair representation
        zi = self.atom_embedding(self.atom_type[lig_idx])
        zj = self.atom_embedding(self.atom_type[env_idx])
        rbf = self.radial_basis(dist)

        pair = torch.cat([zi, zj, rbf], dim=-1)

        # Soft lifting
        gate = cutoff[:, None] * torch.sigmoid(
            self.gate_mlp(pair)
        )   # [E, M]

        pair_feat = self.pair_mlp(pair)  # [E, M*d]
        pair_feat = pair_feat.view(E, M, d)

        msg = gate[..., None] * pair_feat

        # Set/hyperedge aggregation onto ligand anchors
        agg = scatter_sum(msg, lig_idx)
        n_eff = scatter_sum(gate, lig_idx)

        # optional geometric moments here

        h = self.hyperedge_readout(agg, n_eff)
        b_i = self.ligand_readout(h)

        S = b_i.sum()

        B = self.Bmax * torch.tanh(
            (S - self.b0) / self.Bmax
        )
        return B
```

实际实现里我会尽量不用 PyG/PyTorch Scatter 那些额外 runtime machinery，优先：

`index_add_`

或者现有 TorchScript 可编译的 indexed reduction。

因为这里的模型目的本来就是：

[
\text{cheap}.
]

---

# 14. 训练目标不需要重新发明

你原来的 Outer-λ gap loss 可以原封不动保留。

若：

[
\Delta u^0_{ak}
===============

\beta
(H^{MM}_{k+1}-H^{MM}_k),
]

那么：

[
\delta_{ak}(\theta)
===================

\Delta u^0_{ak}
+
\beta
(A_{k+1}-A_k)
B_\theta(R_a).
]

主 loss：

[
\mathcal L_{\rm gap}
====================

\sum_k
\frac12
\left[
{\rm Var}*{p_k}(\delta_k)
+
{\rm Var}*{p_{k+1}}(\delta_k)
\right].
]

再加已有的能量/force regularization：

[
\mathcal L
==========

\mathcal L_{\rm gap}
+
\lambda_E\langle B^2\rangle
+
\lambda_F
\langle|\nabla_RB|^2\rangle.
]

这和现有 PLAN 的定义一致。

SoftLift 只是在换：

[
B_\theta(R)
]

内部的 parameterization。

---

# 15. Teacher 应该怎么使用

你原来的设计契约这里其实非常正确：

同一个 student architecture 必须比较：

### Direct-gap

[
\mathcal L=
\mathcal L_{\rm gap}
+\cdots
]

完全不看 MACE teacher。

### Distilled

相同 architecture，加：

[
\mathcal L_{\rm teacher}
========================

\left[
B_\theta(R)
-----------

B_{\rm teacher}(R)
\right]^2.
]

于是：

[
\mathcal L=
\mathcal L_{\rm gap}
+
\lambda_T\mathcal L_{\rm teacher}
+
\cdots
]

否则如果 SoftLift 成功，你不知道成功来自：

> MACE teacher 的知识

还是：

> SoftLift architecture 本身直接优化 gap 就已经够了。

这一点已经是现有计划明确要求的对照。

---

# 16. 更有信息量的 teacher distillation：只蒸馏 task-relevant latent

你还有：

[
z_i^{MACE}
\in\mathbb R^{1024}
]

的 `[frame,41,1024]` cache。

没必要要求 cheap model 重建整个 1024 维 latent。

可以在每个 outer fold 的两条 training runs 上找：

[
P:
\mathbb R^{1024}
\rightarrow
\mathbb R^d,
]

其中 (d) 很小，例如只保留真正对 gap predictive 的 latent subspace：

[
q_i^{teacher}
=============

Pz_i^{MACE}.
]

然后 SoftLift 学：

[
h_i^{lift}
\approx
q_i^{teacher}.
]

loss：

[
\mathcal L_{\rm latent}
=======================

\sum_i
|
Qh_i^{lift}
-----------

Pz_i^{MACE}
|^2.
]

这个我建议作为**representation diagnostic**，而不是第一版 primary。

因为它能回答一个非常漂亮的问题：

> EXP-014 压缩失败，到底是因为 student 无法重建 MACE 的 relational latent，还是 latent 本身和最终 scalar gap target 没办法便宜化？

---

# 17. 最重要的实验 ladder

真正好的实验不是直接：

> “SoftLift vs old student”。

而应该做嵌套消融：

| 模型 | 新增能力                             | 回答的问题                                |
| -- | -------------------------------- | ------------------------------------ |
| R0 | EXP-014 pair radial              | 已知失败基线                               |
| R1 | (\rho(\sum_j\phi(x_{ij})))       | 单纯解除 pair-additivity 是否足够            |
| R2 | R1 + learned soft gate           | task-dependent topology 是否进一步有用      |
| R3 | R2 + local geometric moments     | 缺失 angular/local-shape 是否是剩余瓶颈       |
| R4 | R3 + teacher latent distillation | MACE relational representation 能保留多少 |

最关键的是 **R1**。

因为我甚至怀疑你不需要完整 DiffLift。

如果：

[
\boxed{
R1 \gg R0
}
]

就说明真正杀死 EXP-014 的东西只是：

[
\boxed{
\text{pairwise linear additivity}.
}
]

那么事情会突然简单很多。

---

# 18. 甚至可以做到 native OpenMM：这是我认为最值得测试的实现

这里是这整个方案最实际的部分。

你现在最大的 deployment 问题不是只有 FLOPs，而是 TorchForce/network evaluation 本身就很贵：现有记录是 baseline/network-only/full 约 `1.4765/2.7042/3.5678 ms/step`。

所以如果 SoftLift 最后仍然只能跑 TorchForce，**它可能解决 representation，却仍然解决不了 ESS/GPU-hour。**

因此我认为 production target 应该优先考虑一个更窄的：

[
\boxed{
\text{native density-lift}
}
]

形式。

定义对 ligand atom (i)：

[
q_i(R)
======

\sum_{j\in E}
c(r_{ij})
g_{t_i,t_j}(r_{ij}),
]

然后：

[
b_i(R)=\rho_{t_i}(q_i).
]

最后：

[
S(R)=\sum_{i\in L}b_i(R).
]

注意：

[
q_i
]

仍然只是 pairwise sum。

但是：

[
\rho(q_i)
]

是非线性的。

所以已经产生：

[
\frac{\partial^2b_i}
{\partial r_{ij}\partial r_{ik}}
\neq0.
]

也就是说，它已经是最小 many-body model。

---

# 19. 为什么 `CustomGBForce` 恰好适合这个形式

OpenMM 当前官方文档里，`CustomGBForce` 可以定义一个**per-particle computed value**：对每个 particle 与其他 particles 做 pair computation，然后求和；之后可以在这个 computed value 上做 single-particle nonlinear energy。它还支持 cutoff/PBC 和自定义 algebraic/tabulated functions。([OpenMM 文档][1])

这恰好就是：

[
q_i
===

\sum_jg(r_{ij})
]

然后：

[
E_i=\rho(q_i).
]

所以可以给所有粒子固定参数：

[
l_i=
\begin{cases}
1&i\in ligand\
0&otherwise
\end{cases}
]

以及：

[
e_i=1-l_i
]

或更严格的 environment mask。

第一阶段 pair computation：

[
q_i
===

\sum_j
l_i e_j
c(r_{ij})
g(t_i,t_j,r_{ij}).
]

第二阶段 single-particle energy：

[
E_i=
l_i\rho(q_i).
]

这样 environment particle 自己不会得到 ligand readout，但它仍然通过：

[
q_i(R)
]

获得正确反作用力。

---

# 20. 全局 `tanh` 也可以继续保留

`CustomGBForce` 内部先产生：

[
S(R)=\sum_iE_i.
]

然后用 `CustomCVForce` 把这个 Force 的总 potential energy 当一个 scalar CV：

[
S.
]

OpenMM 官方文档明确说明 `CustomCVForce` 的 collective variable 可以由任何 Force object 定义，其值就是该 Force 的 potential energy；外层再用任意 algebraic expression (f(S)) 定义最终 energy。([OpenMM 文档][2])

所以可以直接：

[
B(R)
====

B_{\max}
\tanh
\left(
\frac{S-b_0}{B_{\max}}
\right).
]

再在外层：

[
E_k=
A_kB(R).
]

`CustomCVForce` 也支持 global parameters，因此现有 (A_k) 可以继续作为外层 global parameter 处理。([OpenMM 文档][3])

所以理论上可以做到：

```text
CustomGBForce
    ↓
per-ligand environment density q_i
    ↓
nonlinear ligand contribution rho(q_i)
    ↓
sum S
    ↓
CustomCVForce
    ↓
Bmax*tanh((S-b0)/Bmax)
    ↓
A_k * B(R)
```

而完全不经过 TorchForce。

---

# 21. Native 版本有什么限制

这里不能吹过头。

当前 OpenMM 平台对 `CustomGBForce` 有一个重要限制：第一个 computed value 可以是 pairwise computation，但后续 computed values 必须是 single-particle computation。([OpenMM 文档][1])

所以一个 `CustomGBForce` 很自然地对应：

[
\boxed{\text{一个 pairwise density channel}}
]

而不是一个任意宽的 vector hyperedge embedding。

如果需要 (M) 个独立 density channels：

[
q_i^{(1)},q_i^{(2)},\ldots,q_i^{(M)},
]

一个务实方案是使用少量独立 inner forces，每个产生一个 channel，然后把它们当作多个 CV/energy components。

但 channel 多了以后会增加 neighbor evaluation 成本，因此第一版必须非常小。

我会先尝试：

[
M=1
]

甚至不要急着 (M=4,8)。

---

# 22. 这意味着一个非常现实的开发顺序

**第一阶段完全离线，不跑新 MD：**

用现有 `3×500` 帧重新测试 representation family。

先证明：

[
R1:
\quad
B_i
===

\rho
\left[
\sum_jf(r_{ij})
\right]
]

能不能从 EXP-014 的灾难性结果中恢复 signal。

如果 R1 都不行，再测试 soft gate：

[
R2:
\quad
B_i=
\rho
\left[
\sum_j
a_{ij}f(r_{ij})
\right].
]

再不行，才上 geometry moments。

这样开发成本很低。

---

# 23. 第二阶段才做 native OpenMM prototype

如果例如：

[
R1/R2
]

已经在 held-out runs 上保住大部分 gap signal，那么就把学出来的：

[
g_{type}(r)
]

和：

[
\rho(q)
]

蒸馏成：

* spline；
* tabulated function；
* 低阶解析函数。

然后放进：

`CustomGBForce + CustomCVForce`

而不是先写 TorchForce。

OpenMM custom forces 当前支持 algebraic expressions 和 tabulated functions。([OpenMM 文档][4])

这实际上把 DiffLift 变成：

[
\boxed{
\text{offline learned topology/aggregation}
\rightarrow
\text{native analytic Hamiltonian}.
}
]

这比“再造一个小 neural force”更符合你现在的真实瓶颈。

---

# 24. 训练/验证协议应该保持现有 LORO，不要重新随机切 frame

现有设计已经确定：

[
\text{train on 2 continuous runs}
\rightarrow
\text{test on 1 whole held-out run}.
]

Student teacher-target 也要求 outer-fold 内 teacher 只能在两条 training runs 上拟合，再到第三条评价。

因此新 SoftLift screen 继续：

```text
fold 1: run2+run3 → run1
fold 2: run1+run3 → run2
fold 3: run1+run2 → run3
```

而不是 frame shuffle。

这样才能与：

* MACE cached readout；
* LocalResidualStudent；
* EXP-014

真正横向比较。

---

# 25. 新 experiment 的 primary metric 应该重新定义，不能机械继承 EXP-014

这一点也很重要。

EXP-014 当时定义的是**compression screen**，所以要求：

[
R^2\ge0.90
]

并保留：

[
\ge80%
]

的 student gap-variance improvement。

如果 SoftLift 的目标改成：

[
\text{直接优化 gap}
]

那么它没有义务复现旧 student scalar。

因此：

[
R^2(B_{\rm SoftLift},B_{\rm oldstudent})
]

应该降级成 diagnostic。

真正 primary 应重新预注册为：

[
\boxed{
\text{held-out gap-variance improvement}
}
]

以及最终：

[
\boxed{
\text{ESS/GPU-hour}.
}
]

当然在任何 online promotion 之前仍需要 D2/D3/D4 那一套：

* finite difference；
* cutoff smoothness；
* force tail；
* CPU/CUDA；
* short NVT；
* independent repeats。

现有 TODO 也明确要求新的 student 只有 ESS/GPU-hour 真正改善才能重新进入 WP-5A。

---

# 26. 它解决什么，不解决什么

这部分必须分得非常清楚。

它**可能解决**：

[
\boxed{\text{representation compression bottleneck}}
]

即：

[
\text{MACE signal}
\rightarrow
\text{cheap Hamiltonian}
]

这一箭头。

特别是：

[
\text{pair radial 太弱}
]

而：

[
\text{LocalResidualStudent 太贵}
]

之间缺少的那一层。

但是它**不会自动解决**：

[
\boxed{\text{到底有没有真实的 physical mixing bottleneck 值得 bias}}
]

EXP-016 没有 physical state/replica history，因此只得到 surrogate-only temporal audit；不能据此声称 learned slow information。

EXP-017 又没有定位出一个明确 adjacent-λ bottleneck。

EXP-018 现在只是 stationarity confirmation，并且明确没有授权 student、lambda insertion 或 P1/P2。

所以新的 DiffLift/SoftLift work：

[
\boxed{
\text{可以重新研究 compression}
}
]

但是：

[
\boxed{
\text{不能因为 compression 成功就自动重开 WP-5。}
}
]

---

# 27. 最终我会把整个方法写成这个数学形式

这是我认为最干净的版本：

[
\mathcal E(R)
=============

{(i,j):
i\in L,,
j\in E,,
r_{ij}^{MIC}<r_c}
]

[
x_{ij}
======

\Phi_{\rm pair}
(Z_i,Z_j,r_{ij},\ldots)
]

[
a_{ijm}
=======

c_{\rm C2}(r_{ij})
\sigma
\left[
s_m(x_{ij})
\right]
]

[
u_{im}
======

\sum_{j:(i,j)\in\mathcal E}
a_{ijm}\phi_m(x_{ij})
]

[
n_{im}
======

\sum_j a_{ijm}
]

可选：

[
G_{im}
======

{
|\mathbf M_{im}^{(1)}|^2,
{\rm Tr}\mathbf M_{im}^{(2)},
{\rm Tr}[(\mathbf M_{im}^{(2)})^2]
}
]

然后：

[
h_{im}
======

\rho_m(u_{im},n_{im},G_{im})
]

[
S_\theta(R)
===========

\sum_{i\in L}
\psi_\theta
(
h_{i1},...,h_{iM}
)
]

[
\boxed{
B_\theta(R)
===========

B_{\max}
\tanh
\left[
\frac{S_\theta(R)-b_0}{B_{\max}}
\right]
}
]

最后：

[
\boxed{
H_k^*(R)
========

H_k^{MM}(R)
+
A_kB_\theta(R)
}
]

[
\boxed{
F_a
===

-\nabla_{R_a}H_k^*(R)
}
]

以及：

[
A_0=A_K=0.
]

这就是**把 DiffLift 的 learned higher-order relation 思想改造成可用于你的 Outer-λ conservative Hamiltonian 的完整形式**。

---

## 我认为最值得先跑的版本

不是完整 R3，也不是重新上一个 fancy TNN。

而是极简的：

[
\boxed{
q_i
===

\sum_j
f_{type(i),type(j)}(r_{ij})
}
]

[
\boxed{
B(R)
====

B_{\max}
\tanh
\left[
\frac{
\sum_i\rho_{type(i)}(q_i)-b_0
}{B_{\max}}
\right].
}
]

因为它只比 EXP-014 多一个关键操作：

[
\boxed{
\text{先按 ligand atom 聚合，再非线性 readout。}
}
]

但这个操作已经把：

[
\sum_{ij}f(r_{ij})
]

从 pair-additive potential 变成真正的 local many-body set potential。

而且它**刚好是最容易直接做成 native `CustomGBForce + CustomCVForce` 的版本**，因此同时正面攻击你现在两个失败：

[
\boxed{
\text{EXP-014：表达能力太弱}
}
]

和

[
\boxed{
\text{LocalResidualStudent：online cost 太高}.
}
]

如果这个极简版本就把 held-out gap signal 从 EXP-014 的崩溃状态救回来，那反而是最漂亮的结果：你根本不需要真正复制完整 DiffLift/TNN，论文真正给你的启发只是——**此前压缩错的不是 basis 数量，而是 relation topology 和 aggregation order。**

[1]: https://docs.openmm.org/development/api-c%2B%2B/generated/CustomGBForce.html?utm_source=chatgpt.com "CustomGBForce — OpenMM C++ API @OPENMM_MAJOR_VERSION@.@OPENMM_MINOR_VERSION@ documentation"
[2]: https://docs.openmm.org/latest/userguide/theory/03_custom_forces.html?utm_source=chatgpt.com "20. Custom Forces — OpenMM User Guide 8.5 documentation"
[3]: https://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomCVForce.html?utm_source=chatgpt.com "CustomCVForce — OpenMM Python API 8.5.0.dev-b55e608 documentation"
[4]: https://docs.openmm.org/development/userguide/application/06_creating_ffs.html?utm_source=chatgpt.com "7. Creating Force Fields — OpenMM User Guide 8.6 documentation"
</details>

# Part II：权威实现规范（当前唯一实现依据）

```text
document_status: ACTIVE_IMPLEMENTATION_SPEC
current_experiment: EXP-020
parent_experiment: EXP-019 (sealed baseline-reproducibility identity)
protocol: protocols/EXP-020_preregistration.json
protocol_payload_sha256: bbc7dcf97bb635d9a97d890d0f7358eebdc4c4084146aed4829aafd5a06517e9
status_as_of: 2026-08-12
```

当前门状态必须精确记录为：

```text
D0 dataset/support/provenance                 QUALIFIED
D1 R1 direct-gap LORO                         QUALIFIED
D2 R1 offline reference force/geometry        QUALIFIED
D3 reference/export preflight                 QUALIFIED
D3 full (native parity + native cost)          NOT_QUALIFIED / LOCAL_COST_FAILED
native OpenMM N0                              STOP_FULL_SYSTEM_CUSTOMGB
local-candidate N1/N2 backend                 PARITY_QUALIFIED / COST_FAILED
D4 / online / production promotion            BLOCKED
```

`D3 reference/export preflight=QUALIFIED` 绝不等于 `full_d3_qualification=true`。CPU64 real-frame、triclinic、no-contact、energy/force parity 和 TorchScript save/load 已通过；当前 CPU32 report 的对应 cases 也已通过，CUDA32 在该次 preflight 执行环境不可用。tiny synthetic GB parity 只证明语义；真实 CUDA N0 cost-floor 为 `1.6964626x > 1.10x`，因此 `STOP_FULL_SYSTEM_CUSTOMGB`。后续 N1/N2 已完成真实 CUDA parity，但分别以 `6.0717x` 和 `61.2922x` 失败 cost gate，不能进入 full D3 或 production。

## A. 结论、范围与论文边界

本项目不应把论文中的 DiffLift 原样放进 MD。原论文学习离散高阶拓扑，再把接受后的结构交给 TNN；它不定义势能、守恒力、PBC 邻居表或 OpenMM 部署。

本设计只借用一个思想：**关系结构应由任务决定，而不是预先固定为全局 pair-additive sum**。针对当前体系，将它改造成确定、连续、局部、可求导的 SoftLift：

1. 物理 cutoff/cell list 产生 ligand-environment 候选 pair；
2. 以 ligand atom 为 anchor 聚合局部环境；
3. 对聚合后的集合做非线性 readout；
4. 输出一个无 λ 输入的、无量纲 reduced scalar `B(R)`；
5. Outer-λ controller 只在外层施加 `A_k`，且只施加一次；
6. 只有离线、力学、部署和短动力学门全部通过后，才允许讨论 online promotion。

第一实现目标不是完整 DiffLift，而是 R1 单通道 density lift：

$$
q_i(R)=\sum_{j\in E}c_{\mathrm{C2}}(r_{ij})g_{t_i,t_j}(r_{ij}),
$$

$$
b_i(R)=\rho_{t_i}(q_i)-\rho_{t_i}(0),
\qquad
B_\theta(R)=B_{\max}\tanh\!\left(\frac{\sum_{i\in L}b_i}{B_{\max}}\right).
$$

`ρ(0)` 的显式相减固定 additive gauge，并保证无 contact 时 `B=0`。R1 比 EXP-014 多出的关键能力是：**先按 ligand atom 聚合，再做非线性 readout**。

### A.1 论文核验结果

论文第 4 页 §3.1 的真实流程是：输入 attributed graph `G=(V,E,x)`、目标域 `T` 和 `D_max`；用任意 GNN 得到 `z_v`；在目标域层级约束下生成当前维数的 candidate cells；以 permutation-invariant aggregation 得到 `z_C`；用 `φ(z_C)` 作为接受概率并采样 Bernoulli；若未到 `D_max` 则逐维继续；最后把接受后的 relational object 交给 off-the-shelf TNN。

第 5 页 Eq. (3)-(5) 的 hypergraph 版本还有这些精确条件：

- 每个 node 至多提出一个 hyperedge；
- `k_v` 来自 `MLP(z_v)` 参数化的 Categorical distribution；
- candidate 是 embedding-space Euclidean `k_v`-NN；
- 接受概率依赖整个候选集合的 order-invariant `Ψ({z_u})`；
- accepted feature 是对原始 `x_u` 的 scaled mean，不是本文的能量 readout。

第 5-6 页 Eq. (6)-(7) 的 cell-complex 版本先学习 edges，再由 cycle basis 产生 `D>=2` candidates；实验只到 `D_max=2`，使用 Paton/NetworkX basic cycle basis。第 17 页的复杂度为：

$$
O(N^2D+N^2\log k_{\max}+Nk_{\max}D)
$$

（hypergraph），以及

$$
O(N^2D+(E+Nk_{\max})^2+C\ell D)
$$

（cell complex）。第 9 页明确把 cycle-basis 成本和大图扩展列为限制。第 19 页 deterministic threshold 版本仍是离散 topology，不等价于 continuous membership。

### A.2 论文机制与 EXP-020 SoftLift 的关系

| 原论文机制 | MD 中的问题 | EXP-020 处理 |
|---|---|---|
| embedding-space 全局 kNN | rank 交换不连续；全体系 `N²` 不可接受 | 只用物理 cutoff/cell list 产生候选 |
| Categorical `k_v` | 随机且离散 | 不学习 hard cardinality |
| Bernoulli + STE | 同一 `R` 可能没有唯一势能；membership 跳变 | 确定连续权重；R2 gate 必须依赖集合 context |
| accepted topology → TNN | 不是标量势能 | 新增 rotation-invariant scalar energy readout |
| cycle/cell/simplicial lifting | 当前目标无关且昂贵 | 明确不实现 |
| 图分类结果 | 不能外推到 MD/守恒力/效率 | 只作为启发，不作为可行性证据 |

准确名称应为 **DiffLift-inspired ligand-anchored SoftLift**，不是“DiffLift 的 MD 实现”。

### A.3 非目标

EXP-020 不训练 MACE，不让模型接收 λ，不学习 `A_k`，不改变 MM endpoints/PME/softcore/IBS/TMBAR，不覆盖 EXP-012/014/019 输出，不自动重开 WP-5，也不在第一阶段实现 environment-environment message passing、完整 irreps 或 cell complex。任何新 MD 都需要新的 sealed pre-registration 和明确授权。

## B. 冻结证据与可证伪问题

### B.1 已登记事实

- MACE cached latent readout：3/3 held-out folds 改善，平均 gap-variance improvement `44.58%`；这是 teacher-side representation 结果。
- `LocalResidualStudent` direct-gap：平均 held-out improvement `13.9348%`，fold-level `2/3`；不能与上面的 44.58% 混称。
- DEC-048 paired-reseed exploratory pilot：`mixture_ess_proxy` 三次提高约 `10.6%`、`10.7%`、`27.8%`，但不是 production repeats。
- 当前 TorchForce 路线把单步成本提高到约 `1.81-1.89x` baseline，ESS/GPU-hour 三次都下降。
- DEC-050 profile：baseline/network-only/full 约 `1.4765/2.7042/3.5678 ms/step`；即使动态图成本变成零，旧网络仍远超 `1.10x baseline` 预算。
- EXP-014 使用 7 types、49 个有向 type pairs、4/5 Å C2 cutoff 和 `n_rbf=8/16/32`。平均 held-out `R²` 约为 `-11.455`、`-2.80e6`、`-2.09e12`；全部失败，OpenMM qualification 从未开始。
- EXP-020 R1 seed-fixed direct-gap D1：3/3 held-out folds、9/9 checkpoints 改善，fold-median improvement 平均 `55.552405%`，`3031` 个 trainable parameters；checkpoint/report/protocol/dataset hashes 已闭合。
- EXP-020 R1 D2：finite difference、C2 cutoff、triclinic PBC、rotation/translation invariance、no-contact exact zero 与 nonparticipant zero-force 均通过，`full_d2_qualification=true`。
- EXP-020 R1 D3 reference/export preflight：CPU64 reference/deploy/TorchScript energy-force parity、real frame、synthetic triclinic、no-contact 与 serialization round trip 均通过；当前 CPU32 report 的对应 cases 也通过，`reference_export_qualification=true`，但 `full_d3_qualification=false`。tiny synthetic GB parity 通过；真实 CUDA N0 cost-floor 为 `1.6964626x`，正式 `STOP_FULL_SYSTEM_CUSTOMGB`。N1/N2 真实 CUDA parity 均通过，但 N1 cost 为 `6.0717x`、N2 cost 为 `61.2922x`，均超过 `1.10x`，因此 local-candidate backend 仍未获得 cost qualification。
- 真实体系为 `73,536` atoms，其中 ligand 固定 `41` atoms。D0 的 1500 帧共有 `1,920,801` 条 active edge records；单帧最大 `1464` edges、`255` unique environment atoms。
- 1500 帧中进入 5 Å 支持域的 environment atom-ID 并集为 `796`；每条 run 的并集为 `508-532`；三条 run 都出现的稳定核心为 `353`，只在一条 run 出现的瞬态原子为 `376`。这支持局部缓存/skin neighbor list，但不支持把单帧或三条 run 的 atom-ID 并集永久冻结为 production 候选。

### B.2 EXP-020 要回答的问题

1. EXP-014 是因 pair-additive family 太弱，还是整个 cheap-local family 都不可行？
2. ligand-anchored nonlinear density readout 是否恢复 held-out gap signal？
3. set-conditioned membership 是否带来超出参数量的增益？
4. 是否还缺少 angular/local-shape information？
5. 恢复的 signal 能否部署成比旧 TorchForce 显著便宜的守恒路径势？

截至当前状态，问题 1-2 已得到正面离线证据；问题 3-4 尚未由 R1 回答。问题 5 对 full-system CustomGB 的答案已经是否定的：最轻量 N0 即为 `1.6965x baseline`；对 ligand-local alternative backend 仍未回答。任何离线 signal 都尚未证明 ESS/GPU-hour 改善。

## C. Hamiltonian、单位和一次性缩放

### C.1 唯一单位约定

模型输出 `B_θ(R)` 是无量纲 reduced basis。OpenMM 中实际 basis energy 是：

$$
U_B(R)=k_BT\,B_\theta(R)\quad [\mathrm{kJ/mol}].
$$

global state `k` 的目标势能和 reduced potential 分别是：

$$
U_k^*(R)=U_k^{MM}(R)+A_kU_B(R),
\qquad
u_k^*(R)=u_k^{MM}(R)+A_kB_\theta(R).
$$

相邻 reduced gap 必须写成：

$$
\delta_{a,k}=\Delta u^{MM}_{a,k}+\Delta A_k B_\theta(R_a),
\qquad
\Delta A_k=A_{k+1}-A_k.
$$

这里**不再乘 β**。`adjacent_gap_reduced` 已由 ledger 完成 β 转换，`bidirectional_gap_variance_loss` 也故意没有 `beta` 参数。

路径力为：

$$
F_{a,k}^*=F_{a,k}^{MM}-k_BT A_k\nabla_{R_a}B_\theta(R).
$$

reference/training geometry 使用 Å；OpenMM positions 使用 nm，deploy layer 只做一次 `nm -> Å`。OpenMM force 使用 `kJ mol^-1 nm^-1`。

### C.2 λ、endpoint 与 artifact 语义

`A_k=sin²(π λ_vdw,k)` 取自 sealed global 23-state schedule。只有 global states `0` 和 `22` 是 physical endpoints 且 `A=0`。EXP-012 window-0 ledger slice 只有 `[0,1,2,3,4]`，右边界不是 physical endpoint。

所有 TorchScript/native basis artifacts 必须返回 `U_B=kT*B_θ(R)`，并固定：artifact 内 `a_k=1.0`；不含 state-specific `A_k`；global `tanh` 只执行一次；默认 `energy_offset_reduced=0`；OuterLambdaController/IBS wrapper 在外层乘 `A_k`，不得重复乘。

### C.3 additive gauge 与 no-contact

gap variance 对 frame-independent constant 不敏感，但该常数会改变中间态 target energy/IBS weights。第一版禁止自由 `b0`，强制：

$$
b_i=\rho_{t_i}(q_i)-\rho_{t_i}(0),
\qquad
B(R_{\mathrm{no-contact}})=0.
$$

R2/R3 同样要减去网络在零输入处的输出；不得依赖训练“恰好学出零”。

## D. 数据、ragged edges 与 geometry 契约

### D.1 输入身份与现有 dataset

EXP-020 protocol 必须引用而不能修改 EXP-012 的 topology、system、box、stage/window/IBS state、三条 trajectory、三份 ledger、training dataset 及 report hashes。

现有 dataset 的事实：`F=1500`，三条连续 runs 各 500 帧；`N_L=41`；type vocabulary `[1,6,7,8,11,16,17]`；packed edges 总数 `1,920,801`；每帧 `1116-1464` edges；`adjacent_gap_reduced [1500,4]`；`log_importance_unnormalized [1500,5]`。当前 dataset 只有 distance，没有 displacement，不能直接训练 R3。

### D.2 EXP-020 dataset-v1

新增：

```text
output/outer_lambda_exp019_softlift/dataset/softlift_dataset_v1.npz
output/outer_lambda_exp019_softlift/dataset/softlift_dataset_v1_report.json
```

必须包含：

| 字段 | shape / dtype | 语义 |
|---|---|---|
| `partition_index` | `[F] int64` | run partition |
| `frame_index_within_run` | `[F] int64` | 原轨迹帧号 |
| `box_vectors_angstrom` | `[F,3,3] float64` | row-vector convention |
| `edge_offsets` | `[F+1] int64` | CSR frame pointers |
| `edge_ligand_topology` | `[E] int64` | global topology index |
| `edge_environment_topology` | `[E] int64` | global topology index |
| `edge_ligand_local` | `[E] int64` | `0..N_L-1` |
| `edge_distance_angstrom` | `[E] float64` | strict `< outer_cutoff` |
| `edge_displacement_angstrom` | `[E,3] float64` | environment-ligand MIC displacement |
| `edge_unit_shift` | `[E,3] int64` | periodic image shift |
| `edge_pair_type` | `[E,2] int64` | ligand/environment type index |
| ledger arrays | EXP-012 shapes | reduced units，不重复 β |

builder 必须验证 frame/ledger 一一对齐、distance 与 displacement norm 一致、type 与 topology 一致、源 hash 匹配和 canonical ordering。

### D.3 global-to-local reindex 与 packed batch

`ligand_environment_cross_edges` 返回 global topology indices。聚合前必须通过 `reindex_ligand_environment_edges` 或等价 `searchsorted` 转为 ligand-local index。禁止直接对 global ligand index 做 `scatter_sum`。

每帧 canonical key 固定为：

```text
(edge_ligand_local, edge_environment_topology, shift_x, shift_y, shift_z)
```

重复 key 直接失败。packed batch receiver 为：

```text
receiver = frame_in_batch * N_L + edge_ligand_local
```

所有 `index_add_` 显式分配 `[batch_size*N_L,...]`，再 reshape；padding 不进入 loss/aggregation。

### D.4 真实 geometry API 与候选语义

```python
edges = ligand_environment_cross_edges(
    positions_angstrom,
    box_angstrom,
    ligand_indices,
    environment_indices,
    outer_cutoff=5.0,
)
envelope = quintic_c2_cutoff(
    edges["distance"],
    inner_cutoff=4.0,
    outer_cutoff=5.0,
)
```

候选固定为 environment -> ligand 有向 bipartite edges；无 ligand-ligand、environment-environment、reverse/self edges；membership 是 `distance < 5 Å`；C2 在 `r<=4 Å` 为 1、`r>=5 Å` 为 0。edge membership 本身离散，连续性来自 C2 权重及其前两阶导数归零；不得声称 edge list 本身可导。

primary features 不使用 partial charge。charge 只能作为预注册 sensitivity variant，并记录来源和单位。

### D.5 PBC、安全域与 overflow

D0 必须记录 box determinant/condition number/face heights，并要求 `2*outer_cutoff < min(face_height)`。第一 native prototype 只授权 fixed-box NVT。half-box tie 不进入训练/qualification；出现即 fail closed。`r < r_min_support` 或重合原子也直接失败，禁止依赖 `norm(0)` 的未定义梯度。

旧 Torch deploy 的 linked-cell path 只支持 diagonal box 且每轴至少 3 bins；triclinic 会退回 `O(N_L*N_env)` brute force。EXP-020 不能把它写成通用高效 neighbor list。当前 R1 D3 reference/export 路径仍是显式全环境 brute force，只获得 CPU64 correctness preflight，不获得 online cost 资格。需另做 scriptable triclinic cell/Verlet list，或把 Torch online 标为 `NOT_IMPLEMENTED`。

当前 deploy 在形成 cutoff membership 前，对所有 Cartesian ligand-environment pairs 做 half-box tie assertion；float32 可能把 cutoff 外、物理上无贡献的 fractional component 舍入为精确 `0.5`，导致全批 fail closed。旧 CPU32/CUDA32 执行结果在定位具体 case/pair、证明 box face-height 与 cutoff 安全关系前曾保持 `UNSUPPORTED`；当前 CPU32 active-tie diagnostics 已为零且 report cases 通过，CUDA32 仍需在目标 GPU 上单独验证。不得简单删除 assertion 或任意改变 tolerance。

继承 audited ceilings：unique environment atoms `target<=256/hard<=320`；directed edges `target<=1536/hard<=2048`；每 ligand atom neighbors `target<=64/hard<=80`。现有 geometry 不自动执行这些门，EXP-020 dataset/reference/deploy 都必须显式检查。超过 ceiling 时中止并报告；禁止 top-k、静默截断、丢 edge 或改成 `B=0`。
## E. 模型 ladder 与张量契约

### E.1 R0：冻结 EXP-014 replay

R0 不重新调参，只重放并校验现有报告/hash：

$$
B_{R0}(R)=\sum_{(i,j)}\sum_p w_{t_i,t_j,p}\phi_p(r_{ij})c_{\mathrm{C2}}(r_{ij}).
$$

`n_rbf=[8,16,32]`、ridge grid、fold target、type vocabulary 和 cutoff 必须与 `run_exp014_native_compression_audit.py` 完全一致。R0 只是 regression baseline，不参与新 model choice。

### E.2 R1：单通道 NativeDensityLift（首要实现）

固定 primary config：

```text
T=7, P=16, M=1, hidden_rho=16
inner_cutoff=4 Å, outer_cutoff=5 Å
Bmax=10 reduced
ligand_pool=sum（N_L 固定 41）
```

radial basis 与旧 student/EXP-014 同 family：

$$
R_p(r)=\exp\left[-\frac12\left(\frac{r-\mu_p}{\sigma}\right)^2\right],
$$

`μ_p` 均匀覆盖 `[0,5 Å]`，`σ=5/(P-1) Å`。typed pair density：

$$
g_{ij}=c_{\mathrm{C2}}(r_{ij})\sum_{p=1}^PW_{t_i,t_j,p}R_p(r_{ij}),
\qquad
q_i=\sum_jg_{ij}.
$$

每个 ligand type 使用 `1 -> 16 -> 16 -> 1` SiLU MLP `ρ_t`：

$$
b_i=\rho_{t_i}(q_i)-\rho_{t_i}(0),
\quad S=\sum_i b_i,
\quad B=B_{\max}\tanh(S/B_{\max}).
$$

R1 强制：empty/no-contact 时 `B=0`；environment permutation/translation/rotation invariance；ligand identity 固定；parameter count 写入 report 且 `<10,000`。`P=8/32` 只作 sensitivity，不改变 primary 判决。

R0 只有在 `ρ` 线性并去掉 global saturation 等限制下才是 R1 的函数形式极限；由于 R0 replay 没有同一个 bounded head，不能再称“EXP-014 是当前实现的严格特例”。

### E.3 R2：set-conditioned SoftLift

仅依赖 pair feature 的 gate

$$
a_{ij}=c(r_{ij})\sigma(s(x_{ij}))
$$

可被吸收到 `g'(x)=σ(s(x))φ(x)`，不构成可辨识的 learned topology。它只能作为负对照。

R2 主模型先构造 anchor context：

$$
c_i=\frac{\sum_jc(r_{ij})\eta(x_{ij})}{1+\sum_jc(r_{ij})},
\qquad c_i\in\mathbb R^{d_c}.
$$

再定义 context-conditioned membership：

$$
a_{ijm}=c(r_{ij})\sigma[s_m(x_{ij},c_i)],
$$

$$
n_{im}=\sum_ja_{ijm},
\qquad
u_{im}=\frac{\sum_ja_{ijm}\phi_m(x_{ij})}{1+n_{im}}.
$$

固定第一版 `M=4, d=d_c=8`：

```text
pair_context [E,8]
gate         [E,4]
pair_value   [E,4,8]
u            [F,41,4,8]
n_eff        [F,41,4]
```

readout 使用 `(u,log1p(n_eff))` 并减去零输入 readout。R2 必须跑两个 capacity controls：相同参数量但 context 置零；相同模型但每帧打乱 anchor context。只有主模型相对两者都有 held-out incremental gain，才能把增益解释为 set-conditioned relation。

### E.4 R3：归一化局部几何 moments

定义无量纲 displacement `d_bar=d/r_c`：

$$
M^{(1)}_{im}=\frac{\sum_ja_{ijm}f_m(r_{ij})\bar d_{ij}}{1+n_{im}},
$$

$$
M^{(2)}_{im}=\frac{\sum_ja_{ijm}f_m(r_{ij})\bar d_{ij}\bar d_{ij}^{T}}{1+n_{im}}.
$$

只把无量纲 invariants 输入 readout：

$$
I_1=\|M^{(1)}\|^2,
\quad I_2=\mathrm{Tr}(M^{(2)}),
\quad I_3=\mathrm{Tr}[(M^{(2)})^2].
$$

shape：

```text
M1 [F,41,4,3]
M2 [F,41,4,3,3]
I  [F,41,4,3]
```

R3 依赖 dataset-v1 的 displacement/shift。它不是 Native R1 candidate；除非另有证明，不得声称 CustomGBForce 能表示 R3。

### E.5 R4：teacher latent diagnostic

R4 不重建完整 1024-d latent。每个 outer fold 只用两条 training runs：training-only 标准化 `[frame,41,1024]`；training-only PCA/SVD 得到 `d_T=8`；以最大绝对 loading 为正固定每个 component 符号；保存 mean/scale/projection hash；held-out run 只 transform。

$$
L_{latent}=\frac{1}{FN_L}\sum_{a,i}\|Qh_{ai}-Pz^{teacher}_{ai}\|^2.
$$

latent target 在 training split 标准化为每维方差 1，固定 `λ_T=1`。R4 只回答 representation preservation；teacher reconstruction 不能替代 held-out gap primary。

### E.6 能力边界

R1-R3 没有 environment-environment message passing。即使加 distillation，推理仍只看 ligand-anchored local candidates；无法恢复 MACE relational signal 是允许的失败。若 R3 仍失败，停止当前 revision，另立实验评估严格预算的 environment-environment block；不得在本实验中无限增加 channels。

MACE 的 `5.5 Å` 是每层 radial graph cutoff，不应写成“5.5 Å 外绝对失明”。多层 message passing 会把有效 receptive field 扩展到多跳范围；典型两层模型可依赖约两跳局部环境。R1 只直接聚合 5 Å ligand-environment pairs，表达域比 teacher 更窄。当前 D1 成功证明这种压缩在 frozen held-out surrogate 上有效，但不证明所有 online 构型都保持 teacher 信息。受体-配体局部结构稳定支持缓存 neighbor candidates；它不授权冻结坐标、冻结能量或忽略可交换 solvent。

## F. Reference API 与模块拆分

### F.1 文件责任

```text
local_residual/softlift.py
    SoftLiftConfig / PackedSoftLiftBatch / R1-R3 reference models
local_residual/softlift_dataset.py
    dataset-v1 build/validate/canonical packing
local_residual/softlift_deploy.py
    scriptable Torch module; nm->Å; reduced->kJ/mol
local_residual/softlift_openmm.py
    native R1 prototype only
scripts/run_exp019_softlift_d0.py
scripts/train_exp019_softlift_loro.py
scripts/check_exp019_softlift_d2.py
scripts/check_exp019_softlift_d3.py
scripts/check_exp019_softlift_d4.py
```

现有 `geometry.py` 的 Python validation、dict return 和动态索引不能直接放入 `torch.jit.script` forward。`softlift_deploy.py` 必须提供独立 scriptable path，并由 D3 证明与 reference 等价。

### F.2 配置对象

```python
@dataclass(frozen=True)
class SoftLiftConfig:
    schema_version: str
    rung: Literal["R1", "R2", "R3"]
    type_vocabulary: tuple[int, ...]
    n_ligand_atoms: int
    n_radial_basis: int
    n_channels: int
    pair_dim: int
    context_dim: int
    inner_cutoff_angstrom: float
    outer_cutoff_angstrom: float
    b_max_reduced: float
    max_environment_atoms: int
    max_edges: int
    max_neighbors_per_ligand: int
    no_contact_output: Literal["exact_zero"]
    protocol_sha256: str
```

构造时检查全部值；forward 只接受已验证、已 packed tensors。

### F.3 Packed R1/R2 forward

```python
def forward(batch: PackedSoftLiftBatch) -> Tensor:
    radial = radial_basis(batch.edge_distance_angstrom)          # [E,P]
    envelope = quintic_c2(batch.edge_distance_angstrom)          # [E]
    receiver = batch.edge_frame * N_L + batch.edge_ligand_local  # [E]

    if rung == "R1":
        pair_w = W[batch.edge_ligand_type,
                   batch.edge_environment_type]                  # [E,P]
        edge_q = envelope * (pair_w * radial).sum(-1)            # [E]
        q = zeros(F_b*N_L).index_add_(0,receiver,edge_q)
        q = q.view(F_b,N_L)
        b_i = typed_rho(q) - typed_rho(zeros_like(q))
    else:
        pair = pair_encoder(types,radial)                         # [E,d]
        c_num = zeros(F_b*N_L,d_c).index_add_(
            0,receiver,envelope[:,None]*eta(pair))
        c_den = zeros(F_b*N_L).index_add_(0,receiver,envelope)
        context = c_num/(1.0+c_den[:,None])
        gate = envelope[:,None]*sigmoid(
            gate_mlp(pair,context[receiver]))                    # [E,M]
        value = pair_value(pair).view(E,M,d)
        u_num = zeros(F_b*N_L,M,d).index_add_(
            0,receiver,gate[...,None]*value)
        n_eff = zeros(F_b*N_L,M).index_add_(0,receiver,gate)
        u = u_num/(1.0+n_eff[...,None])
        b_i = centered_readout(u,n_eff,optional_moments)

    S = b_i.view(F_b,N_L).sum(dim=1)
    return Bmax*tanh(S/Bmax)                                     # [F_b]
```

实现不得使用未定义 `scatter_sum`、未约束 `view(E,M,d)` 或 global ligand index。

## G. 训练协议

### G.1 split、seed 与 leakage

严格 whole-run LORO：

```text
fold_test_run1: train run2, validation run3, test run1
fold_test_run2: train run3, validation run1, test run2
fold_test_run3: train run1, validation run2, test run3
```

训练若同时使用两条非-test runs，则每条 run trailing 20% contiguous block 用于 early stopping，前 80% 用于 updates；test run 不参与 normalization、PCA、teacher projection、early stopping 或 model choice。

每个 rung/variant/fold 固定 seeds `[0,1,2]`，Adam `lr=1e-3`，最多 500 epochs，patience 30。修改即产生新 protocol hash。

### G.2 精确 loss 与 frozen-importance 边界

```python
bidirectional_gap_variance_loss(
    adjacent_gap_reduced,             # [F,K]
    basis_reduced,                    # [F]
    delta_A,                          # [K]
    log_importance_unnormalized,       # [F,K+1]
    partition_index=partition_index,
    energy_regularization_coefficient=1e-4,
    force_regularization_coefficient=0.0,
)
```

内部必须保持：每 partition、每 target-state column 单独 logsumexp normalization；不 clipping；corrected gap 为 `gap+delta_A*B`；每条 edge 取左右 target distributions 的 weighted population variance 平均，再对 partition-edge arithmetic mean。

D1 的 importance 是 frozen baseline importance，因此 loss 应称 **frozen-importance surrogate**，不是 θ-dependent target weights 的自洽重算。D1 通过后，在 D2/D3 对 held-out frames 以完整 `u_k^*` 重算 target weights；若结论方向翻转，不晋级。

`direct_gap` 使用 `L_gap+1e-4*mean(B²)`。`distilled` 用相同 architecture/seed/split/budget 再加 teacher loss。R1-R3 先报告 direct-gap；R4 才加 latent loss。old-student scalar `R²` 和 latent reconstruction 都只作 diagnostic。

force penalty 在 D1 固定为 0，因为 cached dataset 没有 live coordinate autograd graph。若 D1 通过，可在 D2 另立 training-only live-frame fine-tune；系数必须在 D2 开始前冻结。

### G.3 D1 判决

每 fold 以 3 seeds 的 median held-out improvement 为 fold value，并报告全部 seeds。一个 rung 的 direct-gap qualification 要求：

- 至少 `2/3` folds median improvement > 0；目标 `3/3`；
- 三折 mean relative improvement > 0；
- worst-fold degradation 不超过 `10%`；
- 每个合格 fold 至少 `2/3` seeds 改善；
- 无 nonfinite，全部 support/budget/hash gates 关闭；
- R2/R3 相对前一 rung 有正的 paired-fold median incremental gain。

R0-R3 作为一次 pre-registered nested ablation 全部离线运行，避免看结果后临时改变复杂度。R4 只有在 R1-R3 至少一个 rung 通过 D1 后运行。
## H. 验证阶梯与停止规则

### H.1 D0：身份与支持域

产物必须包含 dataset/report/protocol hashes、environment versions、dtype、edge statistics、box safety、min-distance 和 canonical edge hash。

任一源 hash 不符、frame/ledger 错位、shift/distance 不一致、budget overflow、box 不满足安全域或 nonfinite，D0 立即失败，不训练。

### H.2 D1：离线 held-out signal

执行 R0 replay 和 R1-R3 nested ablation。primary 是 held-out gap-variance improvement；old-student reconstruction `R²` 只解释 compression。

EXP-020 R1 seed-fixed direct-gap 已正式通过 D1：3/3 folds、9/9 seeds 改善，平均 fold-median improvement `55.552405%`。该结论只适用于 frozen-importance held-out surrogate；不等价于 online ESS/GPU-hour promotion。

R1-R3 全失败时，当前 revision 停止：不做 native、不做 D4、不增加 channels。当前 R1 已通过，因此只授权继续 D2/D3 检查。

### H.3 D2：能量、力和不变量

EXP-020 sealed D2 gate 已由 `output/outer_lambda_exp020_softlift_seedfixed001/d2/r1_d2_report.json` 正式通过，`full_d2_qualification=true`。已完成的四组检查是：

1. 一个 seed-fixed R1 checkpoint、一个真实 frame、一个 ligand atom 的三个坐标方向，autograd vs central finite difference；
2. C2 cutoff 函数在 inner/outer boundary 的值、一阶和二阶导检查；
3. synthetic triclinic MIC displacement；
4. synthetic translation/rotation energy-force covariance、empty neighborhood exact zero、nonparticipant zero-force。

预注册数值门及结果均通过：

```text
finite-difference abs error     4.5321e-10 <= 1e-5 reduced
finite-difference relative      2.6301e-9  <= 1e-3
nonparticipant force            0           <= 1e-10 reduced/Å
invariance max                  2.6021e-18 <= 1e-10 reduced
```

以下是进入 D4 前建议追加的 **hardening suite**，不是对现有 D2 qualification 的追溯性改门：多 checkpoint/多 real frames；cutoff `5 Å ± ε` 的完整模型 energy-force crossing；高倾斜安全 box；具体 half-box pair diagnostics；`r -> r_min_support`；environment permutation；total-force sanity；R3 moments（仅在执行 R3 时）。任何补充测试发现真实 correctness bug 都必须产生修复 revision 和新 artifact identity，不能覆盖现有报告。

### H.4 D3：reference/export preflight、native parity 与成本

D3 分成三个独立贡献：

```text
reference_export_qualification
native_parity_qualification
native_cost_qualification
```

只有三者都为真时，汇总报告才能写 `full_d3_qualification=true`。

**已完成的 reference/export preflight。** `output/outer_lambda_exp020_softlift_seedfixed001/d3/r1_reference_export_report.json` 已验证 checkpoint/D1/D2/dataset/protocol/script/TorchScript hashes；CPU64 覆盖 real frame、synthetic triclinic、no-contact、reference vs deploy eager、deploy eager vs TorchScript、energy/force、repeated call 和 save/load/rerun。该 preflight 使用：

```text
energy abs <= 1e-4 kJ/mol
force max abs <= 1e-3 kJ/mol/nm
CPU64 required
CPU32/CUDA32 diagnostic until separately qualified
```

CPU64 全部通过，`reference_export_qualification=true`。当前 CPU32 real-frame、synthetic triclinic、no-contact cases 均通过，active half-box tie count 为零；CUDA32 在该次执行环境不可用，仍未被宣传为支持。

**GB 语义 parity 与真实 parity 的边界。** tiny synthetic Reference prototype 已通过 energy/force/XML parity，只证明 `q_i -> ρ(q_i)` 可由 GB 表达；它不是 `73,536`-atom real-system parity，也没有资格单独贡献 `full_d3_qualification`。任何替代 backend 仍必须检查 Torch reference vs candidate energy/force、目标 CUDA platform、PBC、serialization、force groups、IBS wiring、endpoint `A=0` 与 `A_k`/beta/tanh 各一次。

**N0 cost 已失败。** 真实 CUDA cost-floor：baseline `0.3160764 ms/step`，minimal CustomGB probe `0.5362118 ms/step`，ratio `1.6964626 > 1.10`。正式结论 `STOP_FULL_SYSTEM_CUSTOMGB`。这是无 R1 MLP/完整 mask/nonlinear readout 的下界，因此停止 full-system GB，不再做完整 real parity。

**N1/N2 真实结果已完成。** 两条路线都以新的 local route identity 通过真实 frame energy/force parity 和 serialization；N1 为 per-anchor local CV，N2 为 triclinic ligand-anchored Verlet + TorchScript/OpenMM-Torch。两者均未通过 cost gate，因此只保留为 parity-qualified prototypes，不得写入 `full_d3_qualification=true`。

```text
report:
  output/outer_lambda_exp020_softlift_seedfixed001/d3/r1_n1_n2_local_report.json
report_self_hash:
  2856de64b213a283fc2cb21a3c02fe9471af5a2bc4c329663f1cb3240ce7a701
report_file_sha256:
  8e7094e64d7204ea05a75def01212327316bb22108a905d6e508eba50ad5be96
platform                         CUDA
candidate pool                   2,575 edges; max candidate count/anchor 82
N1 parity                        QUALIFIED
N1 energy abs error              1.8970e-6 kJ/mol
N1 force max error               5.6007e-5 kJ/mol/nm
N1 cost ratio                    6.0717110
N2 parity                        QUALIFIED
N2 OpenMM-Torch energy error     5.1156e-5 kJ/mol
N2 OpenMM-Torch force max error  8.8064e-4 kJ/mol/nm
N2 cost ratio                    61.2921598
pre-registered maximum           1.10
full_d3_qualification            false
```

N2 的 float32 TorchScript artifact SHA-256 为 `1a05d2f6ecc5a3ac482fe3d652ea0164ed675b11a51c8b44b1a3b9d67776dbcf`；float32 只用于 OpenMM-Torch CUDA boundary，CPU64 reference/script 仍保持 double。N1/N2 的 parity 通过不抵消 cost failure；当前 local backend 仍不能进入 production。

当前精确状态：`reference_export_qualification=true`；tiny GB semantic parity passed；N0 full-system cost failed and stopped；N1/N2 real-frame parity passed but both cost failed；所以 `full_d3_qualification=false`。

### H.5 D4：短动力学

只有 D2+D3 全通过才执行 3 组独立短 NVT：baseline/candidate 使用 paired velocity draw；全部 energy/force/temperature finite；无 support/edge overflow；checkpoint/resume hashes 完全一致；disabled candidate 回到 baseline-identical Hamiltonian；safety thresholds 在看 candidate NVT 前用 baseline+D2 分布冻结。

D4 只证明短时稳定，不证明 production benefit。

### H.6 Online promotion

任何 online pilot 都需要新决策。primary 是：

$$
\text{mixture ESS proxy per GPU-hour},
$$

而不是 raw ESS、reconstruction 或 wall-clock 单项。至少 3 independent repeats，沿用 paired-reseed、ledger closure 和 ΔG consistency gates。任何 SoftLift offline success 都不会自动推翻 DEC-050/060，也不会自动重开 WP-5。

## I. Native OpenMM：为什么考虑 GB，以及为什么全体系 GB 已停止

### I.1 需要的不是 GB 溶剂物理，而是两阶段计算语义

R1 的不可约计算结构是：

$$
q_i=\sum_{j\in E}g_{ij}(r_{ij}),
\qquad
B=B_{max}\tanh\left(\frac{\sum_{i\in L}[\rho_{t_i}(q_i)-\rho_{t_i}(0)]}{B_{max}}\right).
$$

关键不是 pair energy 本身，而是**先对每个 ligand anchor 聚合 pair density，再对聚合值做 nonlinear readout**。

`CustomNonbondedForce` 适合 pair-additive `E=\sum_{ij}f_{ij}(r_{ij})`，并支持 `addInteractionGroup(ligand, environment)`；但它不能在一次 force 中得到 41 个独立的 `q_i` 后分别计算 `ρ(q_i)`。把 `ρ(sum)` 移到 pair 内会退化回已失败的 R0 family。

`CustomGBForce` 被考虑，是因为第一项 `ParticlePair` computed value 能为每个 particle 形成 `q_i=sum_j g_ij`，后续 `SingleParticle` stage 能计算 `ρ(q_i)`；`CustomCVForce` 再承载 global `tanh`。这里 `GB` 只是 staged per-particle reduction DSL，**不是**启用 Generalized Born implicit solvent，也不改变 explicit-solvent Hamiltonian。

```text
CustomGBForce
  q_i = sum_j ligand_i * environment_j * g(type_i,type_j,r_ij)
  raw = kT * sum_i ligand_i * (rho_type_i(q_i)-rho_type_i(0))
CustomCVForce (basis only)
  U_B = kT * Bmax * tanh((raw/kT)/Bmax)
Existing OuterLambda controller / IBS wrapper
  U_path,k = A_k * U_B
```

computed value 需要 `ParticlePairNoExclusions`；artifact 固定 `a_k=1`，不含 beta 和 state-specific `A_k`。

### I.2 `N²` 数字与 cutoff 后真实成本必须分开

```text
N_total                         73,536
N_ligand                        41
N_environment                   73,495
all-system ordered pairs        5,407,469,760
ligand x environment Cartesian  3,013,295
D0 max active 5 Å edges         1,464
```

`NoCutoff` 的 `5.407e9` pair 量级可直接停止。N0 实际使用 `CutoffPeriodic(5 Å)`，所以运行时不等于完整 `N²`。但 `CustomGBForce` 没有 `addInteractionGroup()`；mask 只把无关贡献置零，不证明底层 neighbor kernel 预先跳过全体系无关 cutoff pairs。因此不能把组合数比值 `≈1794` 当作运行时倍率，必须以真实 CUDA timing 判决。

### I.3 N0 CUDA cost-floor：已完成并正式停止

```text
report:
  output/outer_lambda_exp020_softlift_seedfixed001/d3/r1_native_n0_cost_floor_report.json
report_sha256:
  edbba79edadfa13e03b8f4900608a32ee07db5276090efd19f4d5a910036a277
platform                         CUDA
baseline median                  0.3160764 ms/step
minimal CustomGB probe median    0.5362118 ms/step
candidate / baseline             1.6964626
pre-registered maximum           1.10
passed                           false
conclusion                       STOP_FULL_SYSTEM_CUSTOMGB
```

测试使用真实 `73,536`-atom System、`CutoffPeriodic(5 Å)`、`ParticlePairNoExclusions`、3 repeats，每次 5 warm-up + 20 timed steps。probe 没有 R1 MLP、完整 ligand/type mask、typed RBF tables 或 nonlinear readout，是 full-system masked route 的成本下界。真实 N0 R1 不可能从该下界的 `1.6965x` 降到 `<=1.10x`。

> **全体系 `CustomGBForce + CustomCVForce` 路线停止。不得继续投入完整 table/parity，也不得用 tiny-system parity 复活该 route。**

已有 tiny synthetic Reference prototype 的 energy/force/XML parity 通过，只证明 GB 语义可表达 R1；它不代表真实体系 native parity，不能覆盖已失败的 CUDA cost-floor。

### I.4 稳定局部环境导出的正确优化：缓存候选，不冻结物理

D0 的 1500 帧 active environment atom-ID 并集 `796`；每 run `508-532`；三 run 共同核心 `353`；只在一条 run 出现 `376`。观测并集对应：

$$
41\times796=32,636
$$

pairs/step，较全部 ligand-environment Cartesian pairs 少约 `92x`，但较单帧最大 active edges 多约 `22x`。这支持 ligand-anchored local candidates；`376` 个 run-specific atoms 同时证明 solvent/sidechain identity 会交换，观测并集不能永久封闭 production 支持域。

推荐：固定 receptor topology shell；solvent 使用 `r_list=5 Å+skin` 的 triclinic cell/Verlet list；位移超过 `skin/2` 或 box 改变时重建；能量仍仅由 `r<5 Å` 的 C2 support 决定；记录 candidate IDs、skin、重建次数、max pairs 和 overflow。

“局部环境基本不变”只支持低频重建，不支持冻结坐标、冻结距离、永久忽略新进入的水。

### I.5 剩余 backend 决策表

| backend | 为什么考虑 | 主要风险 | 当前状态 |
|---|---|---|---|
| N0 `CustomGBForce + CustomCVForce` | `q_i -> ρ(q_i)` 语义自然 | CUDA cost floor `1.6965x` | `STOP_FULL_SYSTEM_CUSTOMGB` |
| N1 41 个 per-anchor pair CV + outer `CustomCVForce` | interaction groups/local superset 可保住 R1 非线性 | per-anchor CV overhead；dynamic solvent IDs；CUDA cost `6.0717x` | `PARITY_QUALIFIED / COST_FAILED` |
| N2 TorchScript/OpenMM-Torch + ligand cell/Verlet list | 最容易表达动态 local funnel | TorchForce bridge overhead；CUDA cost `61.2922x` | `PARITY_QUALIFIED / COST_FAILED` |
| N3 专用 OpenMM C++/CUDA plugin | 一次完成 local pairs/density/readout | ABI/CUDA/维护成本 | `OUT_OF_SCOPE_UNLESS_AUTHORIZED` |
| R0 pair-additive `CustomNonbondedForce` | interaction group 简单 | 科学表达已失败；不能冒充 R1 | `REJECTED_AS_PRIMARY` |

下一步不能再做 full-system GB，也不能把当前 N1/N2 直接提升为 production。若要继续 D3，必须提出新的优化 route identity，并重新通过真实 parity、serialization 和 `<=1.10` cost gate；当前 N1/N2 artifact 的 parity 结果不能借用为 cost qualification。

官方依据：

- [OpenMM Custom Forces guide](https://docs.openmm.org/latest/userguide/theory/03_custom_forces.html)
- [CustomGBForce API](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomGBForce.html)
- [CustomNonbondedForce API](https://docs.openmm.org/development/api-python/generated/openmm.openmm.CustomNonbondedForce.html)
- [CustomCVForce API](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomCVForce.html)

## J. 实验矩阵、产物与 change control

### J.1 决策矩阵

| rung | 新能力 | primary 输出 | deployment eligibility | 失败动作 |
|---|---|---|---|---|
| R0 | 冻结 pair-additive baseline | replay/hash | 无 | 继续 R1；不调 R0 |
| R1 | per-anchor density + nonlinear readout | LORO gap improvement | local-candidate backend only；N0 已停止 | 新 backend 必须重新做 parity/cost |
| R2 | set-conditioned membership | 相对 R1/controls 增益 | 默认无 | 无增益则 learned-relation 解释失败 |
| R3 | normalized angular moments | 相对 R2 增益 + invariance | 默认无 | 停止当前 revision，不加复杂度 |
| R4 | train-only latent distillation | distilled vs direct non-inferiority | 无 | 仅 representation diagnostic |

### J.2 路径与命名

```text
protocols/EXP-020_preregistration.json
output/outer_lambda_exp020_softlift/
  dataset/
output/outer_lambda_exp020_softlift_seedfixed001/
  r1_density/
  d2/
  d3/
    r1_reference_export_report.json
    r1_native_prototype_tiny_report.json
    r1_native_n0_cost_floor_report.json
  local_backend_<new_route>/
  d4/  # blocked until full D3
```

checkpoint：`<rung>__<variant>__fold_test_runN__seedS.pt`。

每个 report 至少记录 schema/experiment/rung/variant/fold/seed、protocol hash、dataset/topology/system/trajectory/ledger hashes、config/parameter count/state_dict hash、Python/Torch/OpenMM/CUDA/platform/precision、units/cutoffs/types/edge-order hash、loss/held-out metrics/support gates、explicit verdict 和 stop reason。

### J.3 不可变与失败策略

- old EXP-012/014/019 artifacts 永不覆盖；output 已存在时拒绝写；
- seal 后修改阈值、shape、seed、cutoff、loss、table domain 都产生新 revision/hash；
- resume 时任一 model/protocol/table/system hash 不符都拒绝；
- nonfinite、support violation、edge overflow、unsupported box、table guard-band hit 都 fail closed；
- production 不允许“该帧自动退回 baseline”；只能在新 simulation 开始前显式选择不同 artifact identity 的 baseline mode。

## K. 当前完成度与下一步顺序

1. `[DONE]` seal EXP-020；完成 D0 identity/support/dataset。
2. `[DONE]` R1 seed-fixed D1：3/3 folds、9/9 checkpoints、平均 `55.552405%`。
3. `[DONE]` R1 D2 offline reference qualification。
4. `[DONE]` D3 CPU64 reference/export preflight；`full_d3=false`。
5. `[DONE]` tiny synthetic GB semantic parity；不贡献真实体系 cost qualification。
6. `[STOP]` N0 full-system CustomGB CUDA cost-floor `1.6964626x > 1.10x`。
7. `[DONE]` N1 per-anchor local CV 真实 CUDA parity/serialization；cost ratio `6.0717x`，`COST_FAILED`。
8. `[DONE]` N2 triclinic ligand-Verlet + TorchScript/OpenMM-Torch 真实 CUDA parity/serialization；cost ratio `61.2922x`，`COST_FAILED`。
9. `[STOP]` 当前 N1/N2 不得进入 full D3 qualification 或 production；不得继续 N0。
10. `[NEXT DECISION]` 若要继续，必须另提经过授权的新优化 backend；任何新 route 先做真实 CUDA cost-floor，再做完整 parity，两者都过才可写 `full_d3=true`。
11. `[BLOCKED]` D4、online 和 production promotion；即使 D4 将来通过，也仍需独立决策和 ESS/GPU-hour repeats。

## L. 必需测试

```text
tests/test_exp019_softlift_dataset.py
tests/test_exp019_softlift_reference.py
tests/test_exp019_softlift_many_body.py
tests/test_exp019_softlift_loro.py
tests/test_exp019_softlift_geometry.py
tests/test_exp019_softlift_deploy.py
tests/test_exp019_softlift_native_openmm.py
tests/test_exp020_softlift_d2.py
scripts/check_exp020_softlift_d3_reference.py
scripts/benchmark_exp020_r1_n0_customgb.py
```

最低覆盖：R1 两 neighbor mixed second derivative 非零；`ρ` 线性时退化为 per-anchor pair sum；empty edge exact zero；global->local reindex；packed vs per-frame；edge permutation；canonical hash；C2 value/一阶/二阶导；triclinic MIC/unit shift/half-box tie/r≈0；R2 pair-only gate redundancy control；R3 rotation invariants；no β/A duplication；global endpoints A=0；eager/TorchScript/CPU/CUDA/native parity；XML/context update；GB exclusion mode；disabled baseline identity；hash mismatch 拒绝 resume。

## M. 最终验收定义

```text
论文边界准确
  -> D0 数据/单位/几何闭合
  -> D1 whole-run held-out signal
  -> D2 守恒力与不变量
  -> D3 部署等价与真实成本
  -> D4 短动力学稳定
  -> independent online repeats 的 ESS/GPU-hour 提升
```

任何箭头失败，只能报告到该层，不得外推下一层。

R1 已完成第一项检验：EXP-014 的 global pair additivity 是主要表达瓶颈之一；per-anchor nonlinear density 在 held-out surrogate 上恢复了强 signal。部署检验目前得到边界性结论：full-system CustomGB 被真实 CUDA timing 否证；N1/N2 均证明了真实 parity/serialization 语义，但分别以 `6.0717x` 和 `61.2922x` 失败 cost gate，因此仍没有可部署 backend。

与 Part I 概念稿不同，当前规范已经不再把 `CustomGBForce` 当作默认 native 终点。已知事实是 N0 `1.6965x baseline` 并停止；后续任何 N1/N2 方案都必须以新的 route identity、真实 parity 和真实 ms/step 重新证明，不能借用 N0 tiny parity 或 D1/D2 成功。
