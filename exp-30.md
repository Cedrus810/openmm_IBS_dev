# EXP-030：统一状态条件化 IBS score——数学理论与操作方法

最后更新：2026-08-14  
状态：`DESIGN_DRAFT_NOT_STARTED`  
与既有实验的关系：本文件把 EXP-029 的候选体系重新表述为一个完整的、结构化的
state-conditioned score；在得到显式授权前，不取代 `exp027_result.md` 中已经登记的
EXP-029，也不授权修改模型、重调 residual 或启动新的 GPU production。

---

## 0. 核心结论

对每个 IBS 局部窗口 \(w\) 和其中的状态 \(k\)，定义

\[
\boxed{
C_{w,k}(R;\phi,\mathbf f_w)
=A_{w,k}B_\phi(R)-f_{w,k}
}
\]

以及等价的 log-weight correction

\[
\boxed{
g_{w,k}(R;\phi,\mathbf f_w)
=-C_{w,k}(R)
=f_{w,k}-A_{w,k}B_\phi(R).
}
\]

状态的有效 sampling energy 为

\[
\boxed{
u^*_{w,k}(R)
=u^0_{w,k}(R)+C_{w,k}(R)
=u^0_{w,k}(R)-g_{w,k}(R),
}
\]

对离散状态 \(k\) 边缘化后得到 IBS integrated/marginal potential

\[
\boxed{
u_{w,\mathrm{mix}}(R)
=-\operatorname{LSE}_k
\left[g_{w,k}(R)-u^0_{w,k}(R)\right]
=-\log\sum_k
\exp\left[f_{w,k}-u^0_{w,k}(R)-A_{w,k}B_\phi(R)\right].
}
\]

这里 \(f_{w,k}\) 和 \(A_{w,k}B_\phi(R)\) 在最终 state log-weight 层属于同一个
结构化对象：前者是 state intercept，后者是共享的 configuration-dependent feature。
但统一表达式不意味着删除二者在训练、校准、provenance 和 reweighting ledger 中的区别。

本实验的直接问题不是“能否把 \(f_k\) 塞进神经网络”，而是：

> 在保持 \(f_k\) 为 \(R\)-independent additive state offset 的前提下，能否把
> residual 与其条件校准后的 \(f_k\) 作为一个完整 score parameter set 冻结、审计和比较，
> 并在计入全部校准与生产成本后提高 mixture ESS/GPU-hour？

---

## 1. 研究对象与边界

### 1.1 当前可执行对象：条件完整的 joint parameter set

当前候选 residual 模型 \(B_{\phi_*}\)、\(A_{w,k}\)、包络、cutoff、`Bmax`、原子选择、
权重和插件身份均来自已经冻结的上游结果。EXP-030 当前允许拟合的只有与该固定
Hamiltonian 匹配的 \(f_{w,k}\)：

\[
\Theta_b
=\{\phi=0,\mathbf f_b^*\},
\qquad
\Theta_c
=\{\phi=\phi_*,\mathbf f_c^*(\phi_*)\}.
\]

因此，当前阶段可以称为：

- 完整 joint parameter set 的构造与 A/B 比较；
- frozen-φ conditional calibration；
- structured state-conditioned log-weight model。

当前阶段**不能**称为同时优化 \((\phi,\mathbf f)\)，因为 \(\phi_*\) 不在本实验中更新。

### 1.2 明确不做

- 不在 production 中更新 \(\phi\) 或 \(f_{w,k}\)；
- 不把 \(f_{w,k}\) 变成 \(R\)-dependent 网络输出；
- 不把 \(f_{w,k}\) 放进 `tanh` 或其他 configuration-dependent 非线性内部；
- 不重新训练/挑选 checkpoint，不重调 \(A_{w,k}\)、`Bmax`、cutoff、skin 或容量；
- 不把 EXP-027/029 中 sampling-only residual 偷换成 physical target correction；
- 不删除 `target_energy`、`bias_history`、`base_energy` 三本账；
- 不用 U4 旧数据重新拟合后冒充独立 confirmation；
- 不因为某个 repeat 表现差而换 seed、换 warm start、延长预算或改变门槛。

### 1.3 未来扩展：真正的 φ/f bilevel optimization

若以后明确授权真正的联合学习，合理形式不是 production 中同时在线更新两类参数，而是

\[
\mathbf f^*(\phi)
=\operatorname{CalibrateIBS}(\phi),
\]

\[
\phi^*
=\arg\max_\phi
\mathcal U_{\rm held\mbox{-}out}
\left(\phi,\mathbf f^*(\phi)\right),
\]

其中 φ 是 outer variable，φ 固定时的 \(f^*(\phi)\) 是 inner calibration result，
utility 必须在未参与训练/校准选择的独立轨迹上评价。该扩展需要新的预注册、训练数据隔离、
计算预算与多重比较控制，不属于当前 EXP-030 的执行范围。

---

## 2. 数学理论

### 2.1 reduced-unit 与物理能量写法

上面的 \(u\)、\(f\)、\(B\) 若用小写，统一表示无量纲 reduced quantities：

\[
u=\beta U,\qquad
\widetilde f=\beta f,\qquad
b_\phi=\beta U_{B,\phi},
\qquad \beta=(k_BT)^{-1}.
\]

当前 OpenMM 实现使用 kJ/mol 物理能量。其对应式为

\[
X_{w,k}(R)
=U^0_{w,k}(R)
+A_{w,k}\left[U_{B,\phi}(R)-U_{\rm offset}\right]
-f_{w,k},
\]

\[
\boxed{
V_{w,\mathrm{mix}}(R)
=-k_BT\log\sum_k\exp[-\beta X_{w,k}(R)].
}
\]

任何协议、报告和代码不得在同一公式中混用 dimensionless \(f_k\) 与 kJ/mol 的
\(U_B\)。本仓库当前 `IBSBiasForce` 的 `f_k` 和 residual basis 均为 kJ/mol。

### 2.2 joint distribution 与 marginal potential

若离散状态先验为 π（默认均匀），联合未归一化分布为

\[
\widetilde p_w(R,k)
=\pi_{w,k}
\exp\left[-u^0_{w,k}(R)+g_{w,k}(R)\right].
\]

真正的 joint reduced potential 是

\[
u_{w,\mathrm{joint}}(R,k)
=u^0_{w,k}(R)-g_{w,k}(R)-\log\pi_{w,k}.
\]

对 \(k\) 求和得到坐标边缘分布：

\[
p_w(R)\propto
\sum_k\widetilde p_w(R,k)
=\exp[-u_{w,\mathrm{mix}}(R)].
\]

因此 `u_mix`/`u_IBS` 才是只依赖 \(R\) 的 integrated potential；不应把它称为严格的
joint potential。均匀先验中的 \(1/K\) 只贡献全局常数，但若以后使用非均匀 state prior，
π 必须显式进入 score、manifest 和分析。

### 2.3 条件状态占据

给定构型 \(R\)，state responsibility/occupancy probability 为

\[
\boxed{
p_w(k\mid R)
=\operatorname{softmax}_k
\left[
\log\pi_{w,k}
+f_{w,k}
-u^0_{w,k}(R)
-A_{w,k}B_\phi(R)
\right].
}
\]

在均匀先验下省略 π。IBS 校准 \(f_{w,k}\) 的目标是让实际 mixture 中的各 state
占据达到预定平衡/覆盖条件。ingredient Hamiltonian 或 residual 改变后，条件最优的
\(f_{w,k}\) 一般也会改变。

定义加入 residual 后各 ingredient state 的配分函数

\[
Z^{(\phi)}_{w,k}
=\int dR\,\exp\left[-u^0_{w,k}(R)-A_{w,k}B_\phi(R)\right].
\]

则该 state 的全局边缘占据满足

\[
P_w(k)
\propto
\pi_{w,k}\exp(f_{w,k})Z^{(\phi)}_{w,k}.
\]

若目标是均匀占据且 \(\pi\) 均匀，理想 offset 为

\[
\boxed{
f^*_{w,k}(\phi)
=-\log Z^{(\phi)}_{w,k}+c_w,
}
\]

其中 \(c_w\) 是窗口 gauge。由此可直接看出：一旦 \(B_\phi\) 或 \(A_{w,k}\) 改变，
\(Z^{(\phi)}_{w,k}\) 也改变，旧 baseline \(f_k\) 一般不再是 candidate 的平衡解。

这里的“应重新校准”是有限预算效率和冻结验证要求，不是渐进正确性的必要条件：只要实际
sampling bias 被完整记录并正确 reweight，任意固定且有限的 \(f_{w,k}\) 原则上都不改变
最终 target observable；不匹配的 \(f_{w,k}\) 会降低 overlap、mixing 和有限预算 ESS。

### 2.4 力

令

\[
p_{w,k}(R)=p_w(k\mid R),
\]

则

\[
\boxed{
\nabla_Ru_{w,\mathrm{mix}}(R)
=\sum_kp_{w,k}(R)
\left[
\nabla_Ru^0_{w,k}(R)
+A_{w,k}\nabla_RB_\phi(R)
\right].
}
\]

所以 additive \(f_{w,k}\)：

- 对固定 state \(k\) 没有直接坐标梯度；
- 不改变该 state 内 residual 的解析形状；
- 但相对 \(f_{w,k}\) 会改变 softmax responsibility，因而间接改变 integrated force；
- 所有 \(f_{w,k}\) 同时加同一常数不改变 responsibility 或力。

因此正确表述不是“\(f_k\) 完全不影响力”，而是“\(f_k\) 是 additive state offset，
没有直接 per-state force；其作用通过 mixture responsibility 实现”。

### 2.5 为什么不能把 f 放进 tanh

允许的结构是

\[
C_{w,k}(R)
=A_{w,k}B_{\max}
\tanh\left(\frac{S_\phi(R)}{B_{\max}}\right)
-f_{w,k}.
\]

禁止的结构例如

\[
\widehat B_{w,k}(R)
=B_{\max}\tanh
\left(
\frac{A_{w,k}S_\phi(R)-f_{w,k}}{B_{\max}}
\right).
\]

此时

\[
\nabla_R\widehat B_{w,k}
=A_{w,k}\operatorname{sech}^2
\left(
\frac{A_{w,k}S_\phi-f_{w,k}}{B_{\max}}
\right)
\nabla_RS_\phi,
\]

\(f_{w,k}\) 会改变 residual force 的饱和区间与幅度，失去 additive free-energy offset
的语义；普通 \(f\mapsto f+c\) gauge 也被破坏。这将是新的 Hamiltonian family，而不是
IBS \(f_k\) 的统一记号。

### 2.6 gauge 与可辨识性

#### Gauge A：IBS 全局常数

\[
f_{w,k}\mapsto f_{w,k}+c_w
\]

会使 \(u_{w,\mathrm{mix}}\) 只改变常数 \(-c_w\)，不改变归一化概率和力。每个窗口必须固定
一个 gauge，当前建议沿用 mean-centered convention：

\[
\boxed{
\sum_{k=1}^{K_w} f_{w,k}=0.
}
\]

`f_0=0` 也可行，但同一实验内不得混用；保存、resume、比较和 hash 前都必须规范化。

#### Gauge B：B/f 分解常数

若 \(B_\phi\) 允许任意常数模，则

\[
B_\phi(R)\mapsto B_\phi(R)+b,
\qquad
f_{w,k}\mapsto f_{w,k}+A_{w,k}b
\]

使所有 \(g_{w,k}\) 完全不变。因此还必须冻结 residual 的能量原点，例如：

- 固定 `U_offset`；
- 保持 ρ readout 的零输入锚定；
- 禁止在 EXP-030 中重新中心化 \(B_\phi\)；
- 将 offset、锚定规则和模型 hash 写进 score manifest。

若未来连 \(A_{w,k}\) 也学习，还会出现 \(A\leftrightarrow B\) 的尺度/符号退化，必须另加
normalization/regularization；当前 \(A_{w,k}\) 冻结，因此不处理该扩展。

### 2.7 rank-1 结构边界

虽然记作 state-conditioned score，当前模型并不是任意的 \(g_k(R)\) 网络，而是

\[
g_{w,k}(R)=f_{w,k}-A_{w,k}B_\phi(R),
\]

即一个 state intercept 加一个跨 state 共享的 rank-1 configuration feature。不得把它
宣传成一般的 state-conditioned neural energy model。该低秩共享结构正是只计算一次
\(B_\phi(R)\)、再由 \(A_{w,k}\) 分配到各 state 的计算优势来源。

---

## 3. 与 U4、EXP-029 和 neural-path 文档的关系

### 3.1 U4 的正确解释

U4 使用的是

\[
\Theta_{\rm U4}
=\{\phi=\phi_*,\mathbf f=\mathbf f_b^*\},
\]

即 candidate configuration feature 已启用，但 state intercept 仍继承 baseline。它是
`BASELINE_FK_TRANSFER / NO_RECALIBRATION` stress test，不是完整 candidate score 的
production utility test。

这种错配不会自动制造渐进偏差，但会让高 \(A_{w,k}\) 窗口的 mixture occupancy 与有限预算
mixing 不稳定。U4 因而不能判定完整 Θc 是否有效。

### 3.2 EXP-029 的正确解释

EXP-029 要比较：

\[
\Theta_b=\{0,\mathbf f_b^*\}
\quad\text{vs}\quad
\Theta_c=\{\phi_*,\mathbf f_c^*(\phi_*)\},
\]

其中两臂分别在自己的真实 sampling Hamiltonian 下校准、验证和冻结 \(f\)。EXP-030 不改变
这项科学问题，只把它正式写成“两个完整 score parameter set 的 ITT utility A/B”。

### 3.3 sampling-only residual 与 physical target neural path

仓库中存在两种容易混淆的语义：

1. neural-path 设计中，\(A_k\overline U_\phi\) 可以是 target intermediate Hamiltonian 的一部分；
2. EXP-027/029 当前 local residual 明确是 sampling-only bias，最终分析 reweight 回
   baseline-only target。

本 EXP-030 当前继承第 2 种语义。因此应写

\[
\boxed{
C^{\rm samp}_{w,k}(R)
=A_{w,k}B_\phi(R)-f_{w,k},
}
\]

而不能把它无条件称为 physical state-energy correction。若未来要让 residual 进入 target，
必须另立实验身份、target ledger、端点协议和 ΔG 验收，不能通过改名悄悄完成。

---

## 4. 操作方法

### 4.1 实验臂

#### Baseline arm

- `residual_enabled=false`；
- 原 production softcore、restraint、WCA 和 λ schedule；
- 针对 baseline Hamiltonian 独立校准并冻结每个窗口的 \(\mathbf f_{b,w}\)；
- score：

  \[
  g^{(b)}_{w,k}(R)=f^{(b)}_{w,k}.
  \]

#### Candidate arm

- 加载冻结的 `LocalManyBodyResidualForce`、模型、\(A_{w,k}\)、`Bmax`、cutoff/skin、
  capacity 和 `U_offset`；
- 针对 residual-enabled sampling Hamiltonian 独立校准并冻结 \(\mathbf f_{c,w}\)；
- score：

  \[
  g^{(c)}_{w,k}(R)
  =f^{(c)}_{w,k}-A_{w,k}B_{\phi_*}(R).
  \]

两臂的 physical target 定义必须完全相同；candidate residual 只进入 sampling bias。

### 4.2 配对与独立性

- 至少 3 个独立 paired repeats；
- 每个 repeat 两臂从相同配对初态、速度分布和 seed family 出发，但轨迹独立演化；
- AB/BA 实际执行顺序预先固定并真实驱动运行，不能只写标签；
- 六个 Stage-2 窗口全部运行；
- 每个窗口的 \(f_{w,k}\) 独立校准，不把相同 state 数误当作相同物理 λ 身份；
- 不跨 arm 复用冻结 \(f\)；baseline \(f\) 至多作为预先固定的 candidate warm start，
  且 warm-start policy 必须在首个 scientific run 前冻结；
- calibration、freeze validation 和 production 的随机流必须可追溯。

### 4.3 Phase 0：冻结 protocol 与 manifest

正式运行前生成只读 manifest，至少包括：

- 实验/score protocol version；
- topology、System XML、positions、box、checkpoint；
- window/state/λ mapping 与 state prior π；
- temperature、β 与所有能量单位；
- baseline potential、LRC、WCA、restraint 定义；
- residual source、binary/source hash、model hash、\(A_{w,k}\)、`Bmax`、offset、cutoff、skin；
- \(B_\phi\) 零点规范；
- \(f\) gauge convention；
- calibration/update/freeze-validation/rescue protocol version；
- production 步数、query cadence、checkpoint cadence；
- TMBAR、ESS、overlap、uncertainty 和 ΔG 门槛；
- AB/BA 顺序、repeat seeds、允许的 infra retry 规则；
- sampling-only/target policy 明文枚举。

任何身份不匹配均 fail closed。

### 4.4 Phase 1：代数与 wiring smoke

在 tiny、非科学预算上完成以下检查：

1. 对同一组坐标独立计算

   \[
   X_{w,k}=U^0_{w,k}+A_{w,k}(U_B-U_{\rm offset})-f_{w,k};
   \]

2. 用稳定 NumPy `logsumexp` 计算 reference \(V_{\rm mix}\)；
3. 与 OpenMM Group-1 能量和 state probabilities 对照；
4. 对照

   \[
   u^0+C=u^0-g;
   \]

   防止符号翻转；
5. 验证 \(f\mapsto f+c\) 只改变势能常数、不改变 force/probability；
6. 验证 \(A_{w,k}=0\) 的 state 不含 residual term；
7. candidate on/off 的 target ledger 完全相同；
8. `bias_history` 包含实际 Group 1 + sampling-only WCA；
9. checkpoint/restart 后 score spec、\(f\)、gauge、模型身份和能量连续；
10. tiny calibration 能进入 learning → freeze → validation → production 状态机，且
    production 中不再调用 `update_weights()`。

数值阈值必须按 dtype 和平台预先冻结；CPU float64、CUDA float32 不得共用不合理的绝对门。
smoke 只验证接线和状态机，不产生 utility 结论。

### 4.5 Phase 2：每臂独立校准 f

对每个 `repeat × arm × window`：

1. 构建该 arm 的真实 sampling Hamiltonian；
2. 初始化 \(f_{w,k}\)：
   - baseline 按原 production policy；
   - candidate 使用预注册的 cold start 或 baseline warm start；运行中不得切换；
3. 运行 warmup；
4. 只用与实际 Group-1 sampling CV 同口径的能量更新 \(f\)，不得把只存在于 target ledger
   的 LRC 或其他项混入；
5. 所有 calibration batch、实际 \(f\) vector、occupancy、LSE residual、失败尝试和
   rescue 进入持久历史；
6. 达到 active IBS protocol 的候选收敛门后，mean-center \(f\) 并冻结；
7. 进入独立 freeze burn-in 与 frozen validation；
8. validation 失败按预注册 rescue 规则处理，不能事后手改 \(f\) 或门槛；
9. 只有 frozen Hamiltonian 验证通过才允许进入 production。

冻结验证至少报告：

- 每个 state 的 raw/mean occupancy；
- \(r_{w,k}=\log(K_w\langle p_{w,k}\rangle)\) 或 active protocol 的等价自洽量；
- max absolute residual；
- coverage ESS；
- 原始帧数和去相关帧数；
- 最终 \(f_{w,k}\)、gauge、state identity；
- freeze/retry/rescue 次数和 GPU 时间。

EXP-030 不重新定义生产 IBS 的收敛数值门，直接使用 manifest 冻结时的 active production
protocol；这样避免为了让 candidate 通过而另设较松门槛。

### 4.6 Phase 3：冻结 production

进入 production 后：

- \(\phi_*\)、\(A_{w,k}\)、offset、\(f_{w,k}\) 全部只读；
- 禁止 `update_weights()`、在线训练、在线 recenter 或临时 rescue；
- 若冻结身份与 Context 参数不一致，立即停止；
- 按相同 cadence 收集两臂的 target、bias、base 能量；
- 完整保存实际 `bias_history`，不能仅凭理论公式重建；
- 保存每帧 target state energies、base energy、box、state mapping 和必要的 residual 诊断；
- 任何非有限分量触发同步 hard gate，不能用零替代；
- burn-in、校准、失败尝试、freeze validation、rescue、production、ledger query 与 checkpoint
  时间全部计入 ITT GPU-hour。

建议逻辑伪代码：

```python
for repeat in preregistered_repeats:
    for arm in actual_ab_ba_order[repeat]:
        for window in all_stage2_windows:
            system = build_arm_specific_sampling_system(arm, window)
            score_spec = build_joint_score_spec(
                phi=frozen_phi_or_zero(arm),
                A=frozen_A(window),
                f=preregistered_initial_f(arm, window),
                gauge="mean_centered",
                sampling_only=True,
            )

            calibration = calibrate_f_with_real_sampling_hamiltonian(
                system, score_spec, active_ibs_protocol
            )
            frozen_score = freeze_and_validate(calibration)

            assert frozen_score.phi_is_read_only
            assert frozen_score.f_is_read_only
            assert target_policy_is_baseline_only

            production = run_production_and_collect_three_ledgers(
                system, frozen_score
            )

analyze_all_windows_with_tmbar()
compare_paired_itt_utility()
```

### 4.7 Phase 4：TMBAR 与三本账

每个 production frame 必须同步保存：

```text
target_state_energies
    = physical softcore state energies
    + LRC_if_applicable
    + physical restraints/other target terms
    # EXP-030 当前不含 residual

sampling_bias_energy
    = actual IBS integrated Group-1 energy
    + sampling-only WCA
    # 已经隐含当帧冻结 f 与 candidate residual

base_energy
    = lambda-independent physical base groups
```

TMBAR 使用真实记录的 sampling distribution 与 target energies 撤销 sampling bias。统一
\(C^{\rm samp}_{w,k}\) 只是 sampling score 的表达方式，不能把 `bias_history` 合并进 target
或省略。

分析必须：

- 调用完整 Stage-2 integrated TMBAR，而不是把六个窗口当独立结论后任意相加；
- 检查所有 window/state coverage；
- 使用 active production 门检查 convergence、overlap、absolute ESS、去相关样本数和
  endpoint uncertainty；
- 比较两臂共享 physical target 下的 ΔG 与联合不确定度；
- 单独报告每个窗口、每个 state 的 occupancy、ESS 和 decorrelation，不能只报告全局最小值；
- 保留失败 repeat，不做 outlier 删除或 best-window 选择。

### 4.8 Phase 5：成本与 utility

Primary 指标使用 intention-to-treat：

\[
\boxed{
\eta_{\rm ITT}^{(a,r)}
=\frac{N_{\rm eff}^{\rm mixture}(a,r)}
{C_{\rm warmup}+C_{\rm calib}+C_{\rm freeze}
+C_{\rm rescue}+C_{\rm prod}+C_{\rm ledger}}.
}
\]

配对 log utility difference：

\[
D_r
=\log\eta_{\rm ITT}^{(c,r)}
-\log\eta_{\rm ITT}^{(b,r)}.
\]

同时分解报告：

\[
r_r=\frac{N_{\rm eff}^{(c,r)}}{N_{\rm eff}^{(b,r)}},
\qquad
s_r=\frac{C_{\rm ITT}^{(c,r)}}{C_{\rm ITT}^{(b,r)}},
\qquad
e^{D_r}=\frac{r_r}{s_r}.
\]

必须把 improvement 来自哪里拆开：

- occupancy flattening；
- decorrelated frame yield；
- per-state ESS ratio；
- residual runtime overhead；
- candidate 专属 calibration/rescue 成本。

不能把历史 baseline calibration 当成候选的免费对照优势；本次 A/B 两臂都按同一 ITT 边界
重新计费。若另报“baseline 历史沉没成本”视角，只能作为 secondary amortization analysis。

---

## 5. 预注册验收门

### 5.1 Algebra/wiring gate

- `u0 + C == u0 - g` 在 dtype-aware tolerance 内；
- OpenMM 与独立 reference log-sum-exp/softmax 一致；
- gauge shift 不改变 probability/force；
- endpoint/zero-A state residual 精确归零；
- checkpoint/restart 后 score 身份与数值连续；
- target/bias/base ledger finite 且闭合；
- candidate residual 未进入 physical target ledger。

任一失败：

```text
EXP030_STOP_SCORE_ALGEBRA_OR_WIRING_INVALID
```

### 5.2 Calibration/freeze gate

- baseline/candidate 都使用同一 active IBS 收敛、冻结验证和 rescue 规则；
- 每个 arm/window 的 \(f\) 都在自己的真实 Hamiltonian 下得到；
- gauge、state/λ identity 与 protocol hash 完整；
- production 开始前 frozen validation 通过；
- production 中 \(f\) 与 φ 均无更新。

失败状态：

```text
EXP030_STOP_BASELINE_F_CALIBRATION_FAILED
EXP030_STOP_CANDIDATE_F_CALIBRATION_FAILED
EXP030_STOP_FROZEN_SCORE_VALIDATION_FAILED
EXP030_INVALID_SCORE_CHANGED_DURING_PRODUCTION
```

### 5.3 Scientific correctness gate

- 完整 Stage-2 TMBAR `converged=true`；
- coverage、overlap、absolute ESS、去相关样本和 endpoint uncertainty 达到 production 门；
- baseline/candidate 共享 target 的 complete-stage ΔG 满足 \(z_{\Delta G}\le2.0\)；
- 无 candidate 特有的温度、结构、constraint、force-tail、support 或 finite 异常；
- 不改变 physical endpoints/base path。

失败状态：

```text
EXP030_STOP_TMBAR_OR_COVERAGE_INSUFFICIENT
EXP030_STOP_SHARED_TARGET_DELTA_G_INCONSISTENT
EXP030_STOP_CANDIDATE_HEALTH_INVALID
```

### 5.4 Utility promotion gate

沿用 EXP-029 当前预定标准：

- 至少 2/3 独立 paired repeats 满足 \(D_r>0\)；
- median \(e^{D_r}-1\ge0.10\)；
- improvement 计入全部 ITT GPU-hour；
- 收益不能来自减少步数、query cadence、遗漏失败尝试或不同停止规则。

通过状态：

```text
EXP030_ATENOLOL_STAGE2_JOINT_SCORE_UTILITY_PASS
```

未达到可重复收益：

```text
EXP030_STOP_JOINT_SCORE_NO_REPRODUCIBLE_ITT_GAIN
```

PASS 只证明当前 Atenolol Stage-2、当前冻结 residual 和当前 IBS protocol 下的 utility；不外推
到 solvent leg、完整 ABFE cycle、其他 ligand、跨体系泛化或真正的 φ/f joint training。

---

## 6. 必需诊断与报告

每个 `repeat × arm × window` 至少报告：

- Θ 的完整 manifest 与 hash；
- φ/model/source/plugin identity；
- \(A_{w,k}\)、offset、\(f_{w,k}\)、gauge 与 state mapping；
- warm start 类型和来源；
- calibration batch 数、步数、\(f\) trajectory、LSE residual；
- freeze burn-in/validation 次数与实际结果；
- rescue 次数、失败原因与成本；
- raw frames、decorrelated frames、统计非效率 \(g_{\rm stat}\)；
- per-state occupancy、switching、ESS ratio、absolute ESS；
- residual energy/force 均值、标准差、分位数和安全门；
- target/bias/base finite/closure；
- production 与 ITT GPU-hour；
- TMBAR convergence、overlap、endpoint uncertainty、ΔG 和 \(z_{\Delta G}\)；
- 实际 AB/BA 顺序、seed、checkpoint/restart provenance。

特别禁止只报告 `min_absolute_ess`：需要把 occupancy flattening 与 decorrelated-frame yield
分开，否则无法判断收益来自 state weighting 还是动力学去相关变化。

---

## 7. 建议产物结构

```text
protocols/
  EXP-030_joint_state_score_preregistration.json

output/outer_lambda_exp030_joint_state_score/
  manifest.json
  decision_log.jsonl
  smoke/
    algebra_reference_report.json
    wiring_restart_report.json
  repeat_0/
    baseline/
      window_*/
        joint_score_spec.json
        f_calibration_history.jsonl
        frozen_score.json
        production_report.json
    candidate/
      window_*/
        joint_score_spec.json
        f_calibration_history.jsonl
        frozen_score.json
        production_report.json
  repeat_1/
  repeat_2/
  stage2_tmbar_report.json
  paired_itt_utility_report.json
  final_decision.json
```

`joint_score_spec.json` 至少应包含：

```json
{
  "protocol": "exp030-joint-state-score-v1",
  "semantics": "sampling_only",
  "units": "kJ_per_mol",
  "gauge": "mean_centered_f_per_window",
  "state_prior": "uniform",
  "formula": "g_k=f_k-A_k*(B_phi-U_offset)",
  "phi_identity": {},
  "A_k": [],
  "f_k": [],
  "state_lambda_identity": [],
  "target_policy": "baseline_physical_target_excludes_residual",
  "frozen": true
}
```

---

## 8. 实施顺序

1. 只读审计 `run_all_windows`、当前 IBS calibration/freeze/rescue 状态机和 resume 身份；
2. 新建 EXP-030 preregistration schema 与 `JointStateScoreSpec` 数据对象；
3. 不改物理实现，先用现有 `_state_expr` 做代数/reference smoke；
4. 增加 score/gauge/provenance 与 target-policy fail-closed 检查；
5. 用 tiny budget 跑两臂完整状态机 smoke；
6. smoke 通过并封存报告后，才允许完整 6-window × 2-arm × 3-repeat run；
7. production 全部完成后统一 TMBAR 与 ITT 分析；
8. 按预注册 failure code 封存结论，不事后改实验身份。

---

## 9. 最终方法表述

推荐论文/计划中的一句话定义：

> We use a structured state-conditioned integrated-ensemble model whose
> per-state log-weight correction is
> \(g_{w,k}(R)=f_{w,k}-A_{w,k}B_\phi(R)\). The learned residual supplies a shared
> configuration-dependent feature, while independently calibrated state
> intercepts balance the mixture generated by each arm. Both blocks are frozen
> before production, and the complete sampling bias is removed by TMBAR when
> estimating the common physical target.

中文：

> 本方法是一个结构化的状态条件化 integrated-ensemble 模型。每个状态的 log-weight
> correction 为 \(g_{w,k}(R)=f_{w,k}-A_{w,k}B_\phi(R)\)：冻结 residual 提供跨状态共享的
> 构型特征，每个实验臂在自己的真实 sampling Hamiltonian 下独立校准 state intercept；
> 两部分在 production 前共同冻结，最终用 TMBAR 完整撤销 sampling bias，估计两臂共享的
> physical target。

这比“neural potential + IBS 权重”更统一，但仍准确保留三条不可删除的边界：

1. \(f_k\) 必须保持 \(R\)-independent additive offset；
2. calibration 与 production 必须分阶段，production 中参数冻结；
3. sampling score 的统一不能抹掉 target/bias/base ledger 与 TMBAR reweighting。