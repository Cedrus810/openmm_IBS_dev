# Pocket-Water Thermodynamic Patch：整体改造方案

> **核心原则：保留热力学闭环，替换昂贵的采样器。**
>
> 本项目不是放弃原来的“猴子补丁”，而是把它从  
> **GCNCMC → P(N) → 固定-N ABFE → \(k_BT\ln P(N)\) 修正**  
> 改造成  
> **更高效、满足 detailed balance 的离散水占据采样器 → P(N) → 固定-N ABFE → \(k_BT\ln P(N)\) 修正**。
>
> 换句话说：
>
> \[
> \boxed{\text{改 sampler，不改 thermodynamic cycle}}
> \]

---

## 0. 为什么现在要改

原方案的统计力学结构没有问题，真正卡住的是 **GCMC/GCNCMC 的采样成本**。

现有实测中，JNK1 ligand1、77k atoms、RTX 2080 Ti 下：

- 最优 switching 大约为 **0.95–1.9 ps**
- 约 **37–49 s / accepted move**
- 加长 switching 并不能持续提高接受率
- 如果继续按原设计铺完整 B 阶梯，成本会迅速放大到数百甚至上千 GPU·h

因此当前主要矛盾不是：

> “\(k_BT\ln P(N)\) 的热力学修正是否成立？”

而是：

> **“怎样更便宜地得到可信的 \(P(N)\)？”**

2026 JCTC 的 SwapMC 提供了一个重要启发：对于 cavity water，**固定总粒子数的 bulk ↔ cavity relocation** 可以绕过传统 insertion/deletion 的高势垒，并且非常适合 GPU 并行。

但本项目不直接改成 SwapMC，也不把水交换直接塞进 ABFE 主循环。我们保留原来“水热力学采样”和“ABFE 主体”解耦的设计，只吸收 SwapMC 的 **proposal / relocation / batch candidate** 思路。

---

# 1. 不变的部分：热力学闭环

## 1.1 核心公式保持不变

选定同一个固定口袋水数 \(N^*\)：

\[
\Delta G^\circ_{\mathrm{bind}}
=
\Delta G_{\mathrm{bind}}(N^*)
+
\Delta G_{\mathrm{hyd}}^{\mathrm{holo}}(N^*)
-
\Delta G_{\mathrm{hyd}}^{\mathrm{apo}}(N^*)
\]

其中

\[
\Delta G_{\mathrm{hyd}}^X(N^*)
=
k_BT \ln P_X(N^*)
\]

所以最终仍然是：

\[
\boxed{
\Delta G^\circ_{\mathrm{bind}}
=
\Delta G_{\mathrm{bind}}(N^*)
+
k_BT\ln P_{\mathrm{holo}}(N^*)
-
k_BT\ln P_{\mathrm{apo}}(N^*)
}
\]

这里不需要改变任何 ABFE estimator。

## 1.2 为什么这个公式本身不是“经验补偿”

对状态 \(X\)：

\[
Z_X = \sum_N Z_X(N)
\]

并定义

\[
P_X(N)=\frac{Z_X(N)}{Z_X}
\]

则：

\[
F_X(N)-F_X
=
-k_BT\ln Z_X(N)+k_BT\ln Z_X
=
-k_BT\ln P_X(N)
\]

因此

\[
F_X
=
F_X(N)+k_BT\ln P_X(N)
\]

这只是条件自由能与完整自由能之间的恒等关系。

所以本项目的“猴子”不在公式，而在：

1. 是否真正采样到了正确的 \(P(N)\)
2. 固定 \(N\) 的 ABFE 是否在该条件子空间内充分平衡
3. \(N\) 以外是否还有慢变量：
   - 水网络拓扑 \(\eta\)
   - 侧链 rotamer
   - loop 构象
   - 与 hydration state 强耦合的局部构象

因此我们要改的是 **采样器与诊断层**，不是热力学层。

---

# 2. 原方案哪些部分保留

## 2.1 cavity region 定义

保留现有：

- ligand-independent protein-side definition
- 小球并集（union of spheres）
- 球心由 pocket lining / buried ligand-reference geometry 确定
- 整体由 protein CA Kabsch 跟随
- union volume 恒定
- apo / holo / rep 间使用同一定义

这是整个方案非常重要的一部分。

区域必须同时满足：

1. 插入/relocation 位置定义一致
2. \(N\) 的统计定义一致
3. 不随着 ligand identity 任意变化
4. 不随着 trajectory deformation 改变体积

## 2.2 fixed-N ABFE

继续使用常规 ABFE：

- 水不参与 alchemical \(\lambda\)
- 生产 ABFE 不挂 GCMC
- 不修改主线 NVT/NPT estimator
- `f_k`
- overlap / ESS
- window cache
- split-half
- residual / convergence gate

全部保持原定义。

本项目继续坚持：

> **水修正只在 setup / endpoint thermodynamics 层发生，不侵入整条 alchemical path。**

## 2.3 \(P(N)\) 与 hydration network 分析

继续输出：

- \(P(N)\)
- \(\langle N\rangle\)
- \(\mathrm{Var}(N)\)
- occupancy vector
- hydration-site clustering
- network population
- network × state count matrix
- hysteresis / up-down scan

这些既用于热力学修正，也用于诊断“\(N\) 是否真的抓到了主要慢变量”。

---

# 3. 要删除 / 降级的部分

## 3.1 不再把长 switching GCNCMC 当主力 sampler

目前 NCMC insertion/deletion 的主要问题是：

- 每一次 proposal 都要沿 switching path 做大量积分
- cavity 插入在拥挤区域接受率低
- 单次 accepted move 成本过高
- 为了扫 \(P(N)\) 还需要大量离散状态

所以 GCNCMC 从：

> **production sampler**

降级成：

> **reference / validation sampler**

它仍有价值，因为它是非常明确的 grand-canonical reference，可用于小体系或少量状态交叉验证。

## 3.2 B 阶梯不再默认是主路线

原方案：

\[
B_1,B_2,\ldots,B_K
\rightarrow \mathrm{MBAR/GCI}
\rightarrow P(N)
\]

改造后不再假设必须用 20–30 个 B states。

新的目标是：

> **直接高效探索 N-space，使相邻 \(N\) 之间有足够跃迁；B 只作为 bias / expanded-ensemble 参数，而不是必须手工铺满的物理路径。**

## 3.3 截断体系不作为第一优化方向

原方案里“截断口袋周围体系”可能获得约 10× 加速，但代价是改变 Hamiltonian。

改造后首先尝试：

- 更聪明的 proposal
- batch candidate
- relocation
- configurational bias
- expanded-N
- delayed acceptance

只有这些仍不够时，才重新评估 truncated environment。

---

# 4. 新 sampler 的目标

新的采样器必须满足四个条件：

1. **目标分布明确**
2. **满足 detailed balance 或 global balance**
3. **能直接得到 \(P(N)\) 或可无偏重权得到 \(P(N)\)**
4. **不要求长 NCMC switching trajectory**

我们不要求所有 move 都直接改变 \(N\)。

可以把 sampler 分成两类 move：

\[
\boxed{
\text{intra-}N \text{ mixing}
+
\text{inter-}N \text{ transitions}
}
\]

也就是：

- **固定 \(N\)**：快速重排水网络
- **改变 \(N\)**：较少但严格地跨 occupancy state

---

# 5. 新算法总体结构

建议做成混合 Markov chain：

```text
                ┌─────────────────────────┐
                │      current state      │
                │  coordinates, N, η      │
                └────────────┬────────────┘
                             │
                  choose move type
              ┌──────────────┴──────────────┐
              │                             │
       relocation / swap              N-changing move
         fixed total N                  N → N±1
              │                             │
      cavity ↔ bulk water            expanded ensemble /
       configurational bias          CBMC insertion/deletion
              │                             │
      fast network mixing            occupancy diffusion
              └──────────────┬──────────────┘
                             │
                   unbiased/reweighted
                         P(N), P(η|N)
                             │
                             ▼
                   choose N* / networks
                             │
                             ▼
                      fixed-N ABFE
                             │
                             ▼
                 + kBT ln P corrections
```

核心思想：

> **让便宜的 relocation move 负责“把水网络搅活”，让较少的 N-changing move 负责探索 occupancy number。**

---

# 6. Move A：固定总水数的 cavity ↔ bulk relocation

这是从 SwapMC 借来的最核心思想。

## 6.1 proposal

从两个区域中搬一颗水：

\[
R_{\mathrm{in}}
\leftrightarrow
R_{\mathrm{out}}
\]

总水数不变。

例如 inward move：

\[
x_i \in R_{\mathrm{out}}
\rightarrow
y_j \in R_{\mathrm{in}}
\]

outward move 反之。

这类 move 本身不改变全系统粒子数，因此：

- 不需要创建/删除粒子
- 不需要 ghost particle topology trick
- 不需要 NCMC coupling path
- 可一次性产生大量 candidate

但口袋内水数 \(N_{\mathrm{in}}\) 会变化：

\[
N_{\mathrm{in}}\rightarrow N_{\mathrm{in}}\pm 1
\]

注意这里“固定-N”指 **全系统总水数固定**，不是 pocket occupancy 固定。

这正是我们需要的。

---

# 7. Move A 改造成 Multiple-Try / Configurational-Bias MC

不要直接照搬“随机一个位置就试一次”的 proposal。

对 inward move：

1. 从 bulk 选一个 source water
2. 在 cavity 内生成 \(K\) 个 trial configurations
3. 每个 trial 包含 position + orientation
4. 计算便宜的局部 trial energy：

\[
u_j^{(0)}
\]

定义 Rosenbluth weight：

\[
w_j = \exp[-\beta u_j^{(0)}]
\]

以及：

\[
W_{\mathrm{in}} = \sum_{j=1}^{K} w_j
\]

从中选择：

\[
P(j)=\frac{w_j}{W_{\mathrm{in}}}
\]

然后构造 reverse trial set，并形成严格的 multiple-try acceptance ratio。

## 7.1 为什么要 Multiple-Try

埋藏口袋最大的问题不是“没有合适位置”，而是：

> **随机撞到合适位置的概率太低。**

一次 proposal 用 \(K\) 个 trial，相当于把：

\[
\text{rare lucky insertion}
\]

变成：

\[
\text{biased but exactly corrected candidate search}
\]

只要 proposal probability 全部进入接受率，偏置本身没有问题。

---

# 8. Move B：显式 N-changing expanded ensemble

如果只做 cavity ↔ bulk relocation：

- 全盒总水数固定
- pocket \(N\) 可以变化
- 对大多数 binding-pocket hydration 问题已经足够

但为了继续保留原 GCMC thermodynamic interpretation，并允许更通用的边界条件，可以保留第二类 move：

\[
N_{\mathrm{total}} \rightarrow N_{\mathrm{total}}\pm1
\]

实现方式不再优先用长 NCMC，而优先尝试：

## 8.1 CBMC insertion/deletion

插入时生成 \(K\) 个 trial：

\[
W_{\rm ins}
=
\sum_{j=1}^{K}
e^{-\beta u_j}
\]

删除时构造 reverse Rosenbluth weight。

接受率中包含：

- volume
- particle number
- activity / chemical potential
- Rosenbluth ratio

这样可以直接采样 grand-canonical distribution。

## 8.2 Expanded-N / simulated tempering in occupancy space

也可以把目标写成：

\[
\pi(x,N)
\propto
 e^{-\beta U(x,N)+g(N)}
\]

其中 \(g(N)\) 是 bias。

目标不是把 \(g(N)\) 当物理量，而是让：

\[
P_{\mathrm{biased}}(N)
\]

尽量平。

最后：

\[
P_{\mathrm{phys}}(N)
\propto
P_{\mathrm{biased}}(N)e^{-g(N)}
\]

因此可以避免人为设计很密的 B 阶梯。

---

# 9. 推荐优先级：先做 relocation，再决定是否保留 grand-canonical N±1

## Phase A

只实现：

\[
\boxed{\text{bulk}\leftrightarrow\text{cavity relocation}}
\]

理由：

- 最简单
- 总粒子数不变
- 没有 chemical potential calibration
- 不需要 ghost insertion
- proposal 可以批量并行
- 直接测试能否恢复正确的 pocket \(P(N)\)

如果它已经能让 cavity occupancy 在合理范围内充分扩散，并且 \(N_{\mathrm{in}}(t)\) 有大量来回 crossing，那么传统 GCMC 可能只保留为 reference。

## Phase B

如果 relocation 的可达 occupancy range 受到 finite bulk reservoir 或其他约束，再增加：

\[
N_{\mathrm{total}}\rightarrow N_{\mathrm{total}}\pm1
\]

的 grand-canonical move。

---

# 10. 长程静电：不要照搬 RF 近似

SwapMC 的一个可改进点是：candidate interaction energy 用 reaction field 近似 PME。

我们自己的 sampler 最好明确区分：

\[
U_0
\]

和

\[
U
\]

其中：

- \(U_0\)：便宜的 surrogate / local energy，用于候选筛选
- \(U\)：生产 Hamiltonian，例如 PME

---

# 11. Delayed Acceptance：cheap proposal + exact Hamiltonian correction

可以用两级 Metropolis。

第一层：

\[
\alpha_1
=
\min\left[
1,
\frac{\pi_0(y)q(y\rightarrow x)}
     {\pi_0(x)q(x\rightarrow y)}
\right]
\]

其中：

\[
\pi_0(x)\propto e^{-\beta U_0(x)}
\]

通过后再计算真实 PME energy。

第二层：

\[
\alpha_2
=
\min\left[
1,
\frac{\pi(y)/\pi_0(y)}
     {\pi(x)/\pi_0(x)}
\right]
\]

即：

\[
\alpha_2
=
\min\left[
1,
e^{-\beta[(U(y)-U_0(y))-(U(x)-U_0(x))]}
\right]
\]

最终 stationary distribution 是真实：

\[
\pi(x)\propto e^{-\beta U(x)}
\]

这样：

- trial search 用便宜局域势
- 只有少量候选进入 exact PME
- 不牺牲 production Hamiltonian consistency

---

# 12. GPU 实现路线

## 12.1 Prototype：纯 OpenMM Python

第一版不要写 CUDA。

目标不是性能，而是验证：

1. detailed balance
2. occupancy distribution
3. move acceptance
4. reversibility
5. \(P(N)\) 与 GCMC reference 一致

建议：

- \(K=8,16,32,64\)
- cavity union uniform sampler
- random quaternion orientation
- 每若干 MD steps 尝试一次 move
- exact OpenMM energy 直接算

## 12.2 第二版：batch local-energy kernel

验证数学后再做 GPU。

一次生成：

\[
K_{\mathrm{pos}}
\times
K_{\mathrm{ori}}
\]

例如：

\[
64 \times 24 = 1536
\]

个 trial。

不一定需要 SwapMC 的 18,000 个。

先测 acceptance / effective sample / wall time 的 Pareto frontier。

## 12.3 local energy 内容

候选阶段优先计算：

- water–protein LJ
- water–water LJ
- real-space electrostatics
- neighbor-list local interactions

可忽略或近似：

- reciprocal PME

然后由 delayed acceptance 精确修正。

---

# 13. cavity union sampler 必须保留

现有 union-of-spheres 是很合适的 proposal domain。

需要补两个数学/工程工具。

## 13.1 union 内严格均匀位置采样

不能简单：

> 随机选一个 sphere → sphere 内均匀点

因为重叠区域会被重复计权。

需要实现真正 uniform-on-union sampler，例如：

### 方法 A：bounding proposal + membership rejection

如果 union 很小，这是最稳妥的 reference 实现。

### 方法 B：sphere-weighted proposal + overlap multiplicity correction

如果点落在 \(m(x)\) 个 sphere 中：

\[
q(x)\propto m(x)
\]

必须用：

\[
1/m(x)
\]

修正，或者用 rejection 消除 multiplicity bias。

第一版优先使用容易证明正确的方法。

---

# 14. source-water selection 也要写进 proposal probability

不要默认 source water uniform。

可以考虑：

### Uniform source

\[
p_i = \frac1{N_{\mathrm{src}}}
\]

最简单。

### Energy-biased source

例如：

\[
p_i
\propto
 e^{+\beta \gamma u_i}
\]

优先搬走不利水。

但是此时 forward / reverse：

\[
q(x\rightarrow y)
\]

必须包含 source-selection probability。

如果目标是先把方法做对，建议：

> **第一版 source water uniform，第二版再加 energy-biased selection。**

---

# 15. velocity 处理

MC relocation 后不能随便把 velocity 清零而不说明。

建议两种选择：

## 15.1 重新从 Maxwell-Boltzmann 抽样

对移动水的平动/原子速度重新采样：

\[
v \sim \mathcal N(0,k_BT/m)
\]

如果 MC 只针对 configurational ensemble，这样最干净。

## 15.2 MC move 后立即做短 thermostat relaxation

也可接受，但必须确认不影响 configurational stationary distribution。

第一版建议重新采样 velocity。

---

# 16. 新的 \(P(N)\) 获取方式

原来：

```text
B ladder
   ↓
GCI / MBAR
   ↓
P(N)
```

新方案优先：

```text
single expanded simulation
   ↓
N trajectory
   ↓
bias reweighting
   ↓
P(N)
```

或者：

```text
fixed-total-water relocation simulation
   ↓
direct histogram of cavity N
   ↓
P(N)
```

如果 relocation simulation 本身就在目标 canonical/NPT ensemble 下：

\[
P(N)
=
\frac{\text{time in occupancy }N}
     {\text{total time}}
\]

无需 MBAR。

---

# 17. GCMC 保留成 reference oracle

原来的 GCMC/GCNCMC 工作不要扔。

它现在变成非常有价值的 benchmark。

对小测试系统：

\[
P_{\mathrm{new}}(N)
\]

应与：

\[
P_{\mathrm{GCNCMC}}(N)
\]

在统计误差内一致。

比较：

- \(\langle N\rangle\)
- \(\mathrm{Var}(N)\)
- \(P(N)\)
- hydration-site occupancy
- network population

---

# 18. 最重要的 correctness tests

## Test 1：toy cavity

简单 LJ / electrostatic cavity。

应有可解析或高精度 reference。

检查：

\[
\frac{P(i\rightarrow j)}
     {P(j\rightarrow i)}
\]

是否符合目标 Boltzmann ratio。

## Test 2：bulk water box + artificial region

区域内没有特殊势。

则 cavity occupancy 应接近 binomial / canonical finite-reservoir prediction。

这是 relocation sampler 最干净的测试。

## Test 3：canonical vs GCMC marginal

大 bulk reservoir 极限下：

canonical fixed-total-water cavity marginal

应逼近 grand-canonical cavity occupancy distribution。

## Test 4：forward/reverse proposal symmetry

记录：

- forward source probability
- reverse source probability
- forward Rosenbluth weight
- reverse Rosenbluth weight
- volume factors

单独验证 log acceptance ratio。

## Test 5：surrogate + exact correction

比较：

### exact-only MC

和

### cheap + delayed acceptance

应得到同样的：

\[
P(N)
\]

否则 correction 有 bug。

---

# 19. 对原 W0–W4 工作流的重构

原来：

| Step | 原内容 |
|---|---|
| W-1 | GCMC 实现选型 |
| W0 | region + B_eq |
| W1 | switching 接受率 |
| W2 | B ladder |
| W3 | GCI / MBAR / network |
| W4 | fixed-N ABFE + correction |

改成：

| Step | 新内容 |
|---|---|
| **W-1** | 保留现有 GCNCMC 作为 reference |
| **W0** | 保留 union region；补严格 uniform-union sampler |
| **W1** | 实现 exact relocation MC prototype |
| **W2** | 加 multiple-try / CBMC proposal |
| **W3** | 加 cheap local energy + exact PME delayed acceptance |
| **W4** | 用 JNK1/P38 测 \(P(N)\)、network mixing、cost |
| **W5** | fixed-N ABFE + 原 \(k_BT\ln P\) correction |
| **W6** | 与原 GCNCMC patch 比较跨-rep collapse |

---

# 20. 旧的 B_eq / μ_ex 怎么处理

它们不删。

保留用途：

1. GCNCMC reference
2. sanity check
3. 将来如果 expanded grand-canonical N±1 move 回归，可直接复用
4. 与 grand-canonical occupancy 做 cross-check

但如果最终主 sampler 是 fixed-total-water relocation：

- 不需要 \(\mu_{\rm ex}\)
- 不需要 B_eq
- 不需要完整 B ladder

---

# 21. network 问题仍然必须管

即使 \(P(N)\) 完全正确，也不能默认：

\[
N
\]

已经描述全部慢自由度。

因此每个 \(N\) 内继续分析：

\[
P(\eta\mid N)
\]

其中 \(\eta\) 是 hydration-network label。

需要看：

- 同一个 \(N\) 下是否有多个长期不互通 network
- relocation 是否加速 network interconversion
- network occupancy 是否在独立 replica 间一致
- fixed-N ABFE 是否对 initial network 敏感

如果：

\[
P(\eta|N)
\]

仍然多峰并且 mixing 差，则下一步应是：

> **固定 N 条件下继续增强 network mixing**

而不是改热力学公式。

---

# 22. 最终验收指标

仍然保留原来的核心验收逻辑：

> **跨 replica 的 ΔG 散布是否塌缩。**

尤其是 JNK1 ligand1：

\[
-2.38
\quad vs \quad
-10.19
\]

这种约 7.8 kcal/mol 的差异。

不能只看：

- 单条 run internal error
- MBAR error bar
- split-half
- 与实验“碰巧更接近”

真正要看的是：

\[
\sigma_{\mathrm{across\ rep}}
\]

是否显著下降。

---

# 23. 新增 sampler-specific 验收指标

除了最终 ABFE，还必须单独验 sampler。

## 23.1 Occupancy mixing

- round trips in N-space
- integrated autocorrelation time of \(N\)
- transition count matrix
- first-passage time
- state residence times

## 23.2 Network mixing

- transition matrix between hydration-network labels
- replica consistency
- hysteresis
- state visitation overlap

## 23.3 Efficiency

定义：

\[
\mathrm{ESS}_N/\mathrm{GPU-hour}
\]

以及：

\[
\text{independent occupancy samples}/\text{GPU-hour}
\]

不要再只看：

\[
\text{acceptance rate}
\]

因为高 acceptance 不代表高 mixing。

---

# 24. 和 SwapMC 的关系

本项目可以明确写成：

### SwapMC 的启发

- cavity ↔ bulk relocation
- fixed total particle number
- batch candidate generation
- GPU parallel candidate energy evaluation

### 本项目保留的差异

1. 不把方法直接绑进整条 ABFE production
2. 仍采用独立 hydration thermodynamics + fixed-N ABFE correction
3. 把 \(P(N)\) 作为显式热力学接口
4. 用 multiple-try / CBMC proposal
5. 目标是 production-Hamiltonian-consistent acceptance
6. 考虑 exact PME delayed-acceptance correction
7. hydration-network degeneracy 仍作为独立慢变量诊断

---

# 25. 论文/方法学叙事可以怎么写

不再说：

> “我们发明一个新的 GCMC。”

更合适的是：

> **A thermodynamically closed hydration-occupancy correction framework with accelerated discrete water-state sampling.**

核心贡献分成两层。

## Layer 1：thermodynamic decomposition

\[
\Delta G_{\mathrm{bind}}
=
\Delta G_{\mathrm{bind}}(N)
+
k_BT\ln P_{\mathrm{holo}}(N)
-
k_BT\ln P_{\mathrm{apo}}(N)
\]

把慢 hydration occupancy 与 alchemical path 解耦。

## Layer 2：accelerated occupancy sampler

用：

- cavity↔bulk relocation
- configurational bias
- multiple-try MC
- optional expanded ensemble
- exact long-range correction

更高效地得到 \(P(N)\)。

---

# 26. 最推荐的最小实现

先别把项目做大。

第一版只做：

```text
[1] 现有 rigid union cavity
[2] bulk ↔ cavity relocation
[3] uniform source-water selection
[4] K-trial CBMC insertion geometry
[5] exact OpenMM energy
[6] strict forward/reverse Rosenbluth weights
[7] direct P(N) histogram
[8] 与现有 GCNCMC reference 比较
```

不要一上来做：

- custom CUDA
- reaction-field surrogate
- delayed acceptance
- adaptive bias
- RL
- learned proposal
- truncated system

先证明：

\[
\boxed{
P_{\mathrm{relocation}}(N)
=
P_{\mathrm{reference}}(N)
}
\]

然后再优化。

---

# 27. 最终路线图

```text
现有方案
GCNCMC → B ladder → MBAR/GCI → P(N)
                     │
                     ▼
              fixed-N ABFE
                     │
                     ▼
               kBT ln P(N)


             ↓ 改 sampler，不改 cycle ↓


新方案
CBMC / Multiple-Try Water Relocation
         + optional expanded-N
         + exact Hamiltonian correction
                     │
                     ▼
                  P(N)
                     │
                     ▼
              fixed-N ABFE
                     │
                     ▼
               kBT ln P(N)
```

---

# 28. 一句话结论

\[
\boxed{
\text{保留“热力学猴子补丁”，重写“采样猴子”。}
}
\]

原来的自由能闭环不动。

真正要替换的是：

\[
\text{昂贵的 GCNCMC insertion/deletion}
\]

变成：

\[
\text{严格 detailed-balance 的 relocation / CBMC / multiple-try occupancy sampler}
\]

并最终仍然用：

\[
P(N)
\rightarrow
k_BT\ln P(N)
\rightarrow
\text{fixed-}N\ \mathrm{ABFE}
\]

完成 thermodynamic closure。

---

## 参考依据

- Ding Y., Wang X., Zou R., Qu T., Zheng H. *Improving Binding Free Energy Predictions with Swap Monte Carlo for Water Sampling*. **J. Chem. Theory Comput. 2026, 22, 7955–7965.**
- 当前项目文档：`PLAN_pocket_water_gcmc_patch.md`，尤其是原始 thermodynamic cycle、union cavity、W0/W1 实测成本和跨-rep 验收逻辑。
