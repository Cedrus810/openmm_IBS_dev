# Vanishing 阶段 / 窗口 0 交接笔记（写于 2026-07-19，2026-07-20 更新，下一个 agent/会话先看这个）

> 这是一份"别再兜圈子"笔记。用户从 **2026-07-10** 就在跟这个问题较劲（`output_lrc_fix/`
> 目录最早文件时间戳可查证），中途反复卡住。这份文档的目的是让下一个接手的人（不管是不是
> 我）不用重新摸索一遍已经摸索过的东西，也不要重复已经被证明是错的猜测。

## 一句话现状

Vanishing（vdW 去耦）阶段窗口 0（λ_vdw=1.0 全耦合端点附近）反复采样失败。**2026-07-20
新发现**：不只是"学不动"（自举困境），在线学习从 v19（TMBAR）起一直存在一个真实的
**符号错误**——`_solve_tmbar_and_recenter` 把 `solve_stage_integrated` 的原始输出
未反号直接用作 f_k，导致每次更新都在强化本来就过量代表的态。v21 修复了在线 TMBAR
调用点；紧接着用平衡条件 `f_k - U_k ≈ const` 独立核对，确认 v20 的 pilot-TI seed
是**同一个 bug 的第二个调用点**。v22 将 pilot-TI seed 同样反号并恢复热启动注入；
真实运行证明它成功把体系推出 state0，却又被“整值覆盖”的在线更新锁死到 state6。
v23 保留 TI 热启动，但将学习改为全态同步、有界的占据负反馈；TMBAR 只作累计覆盖
质量门，不再把绝对向量写入 Context。**尚未经过 GPU 验证**，见下方 v23 一节。

## 时间线（已核实，不用再重新翻）

- **2026-07-10**：`output_lrc_fix/` 开始产出（当前工作的起点）。
- **2026-07-13 ~ 07-16**：见 [`../status/AUDIT_STATUS.md`](../status/AUDIT_STATUS.md) 对应日期条目，多轮物理/协议修复
  （WCA 记账、LJ 长程修正、Boresch Context 等）。
- **2026-07-16/17**：`IBSSampler` resume/冻结验证状态机连续 5 处真实 bug 修复（mode
  回退、主窗口 checkpoint 续算、ladder 预算累计语义、累计步数重复计数、跨进程 ladder
  rung 状态丢失）。这些都是真 bug，修得对，见记忆
  `project_ibs_resume_validation_fix.md`。**修复结尾原话记着**：这次 window 1 落在
  λ_vdw≈0.974–0.988（接近全耦合端），"如果 window 1 再失败，这就是下一个要看的问题"
  ——**这条当时就已经点名了今天这个症状，但之后没人回来专门处理它**，因为接下来又冒出
  别的真 bug 先占用了注意力。
- **2026-07-17（同一天，另一件事）**：发现一个**致命设计错误**——用"相邻态 fixed-H
  overlap"当 IBS 收敛仲裁标准，触发自动拆窗/插 λ/`recalibrate_f_k` 环，**烧了大约一周
  GPU 没出任何 ΔG**。改用 `non_mutating_v1`（不再自动变异 ensemble）。见记忆
  `ibs_overlap_arbiter_false_premise.md`。
  **重要遗留缺口**：当时明确记录"窗口 0/1/3/4/7/8/9 在旧策略下其实收敛过，应该做一次
  rescue audit 去验证/保留这些旧结果，不用整窗重采"——**这个 rescue audit 从未真正执行
  过**（记忆原文：`Deferred (NOT done)`）。后续每次协议升级都是从零冷启动重新采样，没人
  回头认真核对过旧的收敛证据。这是这一路"改来改去總是不 work"感觉的一个真实成因：不是
  哪次修复错了，是该做的验证一直没做。
- **2026-07-18**：窗口 0 端点奇异性诊断——边 0↔1 reduced-ΔU std ≈ 32.7 kT，灾难性低
  overlap。审计确认 vanishing 全程没有已提交结果，需要全新 ensemble。见记忆
  `project_vanishing_window0_endpoint_singularity.md`。
- **2026-07-19（今天）**：
  1. 发现在线 IBS 学习用的朴素 TA 滚动平均是真·数值 bug（测出 state0↔state5 落差
     300+ kJ/mol，独立 TI 交叉验证只有 ~16.7 kJ/mol）。改用已有的 TMBAR
     （`solve_stage_integrated`）替换，`IBS_BIAS_PROTOCOL_VERSION` 18→19。**这部分已经
     GPU 验证**：真实运行测出的能量差量级（~7-24 kJ/mol）确实回到了跟 TI 一致的范围，
     不再是 300+ 那个错误量级。
  2. 但同一次真实运行暴露了 07-17 就已经点名过的那个问题：窗口 0（这次是 6 态，
     λ_vdw 1.0→0.963）`IBSWarmupConvergenceError`，占据几乎全卡在 state 0
     （`mean_p≈0.994`），`min_absolute_ess≈1.0`。这是真实的采样/overlap 问题，TMBAR
     只是诚实地报告了"学不出来"，不是它自己的 bug。
  3. 用户否决了"做结构/pose 诊断"（判断为徒劳），要求直接找一个能跑通的配置。
  4. 排查过程中犯过两次没有先验证就推荐的错误（教训见下），最后用真实缓存数据验证确认：
     **改窗口分组粒度**（不改密度算法本身）能把窗口 0 从 6 态收窄到 4 态。已经改进代码。
  5. **4 态版本 GPU 实跑：仍失败**，占据 99.4%→96.9%，`min_absolute_ess` 还是 ~1.0。按
     论文 Sec 2.4 继续细分，新增 `first_ensemble_target_intervals`（只切窗口 0，其余
     窗口不动，避免丢掉 IBS 效率优势），用户拍板定在 3 态（跟 07-17 真实收敛过的窗口 0
     态数一致）。`THERMODYNAMIC_PATH_PROTOCOL_VERSION` 13→14。
- **2026-07-20**：
  1. **3 态版本 GPU 实跑：仍失败**，占据 98.2%，`min_absolute_ess` 还是 ~1.0。
  2. **用户用 IBS 偏置公式（`V_bias(x) = -kT ln Σ_k exp[-β(U_k(x)-f_k)]`）直接指出
     根因，且被 6/4/3 三态三次真实 GPU 结果证实**：分组只改"哪些态共用一个 bias"，
     从来没改过 state0↔state1 之间真正的 ΔU/λ 间距——三次用的都是同一条 18 态 λ 数组
     的子集，间距一直是 ~0.006-0.007，占据比例几乎不随态数变化正是因为这条边本身
     从没被动过。**分组这条路到此彻底排除，不再尝试任何分组/态数调整。**
  3. 找到真正根因：pilot 探针只在 λ=1.0 和 0.9412 两个粗网格点之间做线性插值，这一段
     单独占了当前总热力学长度的 51.5%——不管态数怎么分组，这一段内部的密度分布从来
     没有真实数据支撑。新增 `_refine_pilot_grid_in_steep_segments`：粗探针跑完后，
     若单个区间占比超阈值（默认 20%），就在那个区间内部真实插入额外探针点重新测
     `metric_g`。`THERMODYNAMIC_PATH_PROTOCOL_VERSION` 14→15。**这是第一次真正改动
     密度算法本身，不是调分组/态数。尚未 GPU 验证。**

## 现在的状态（已实现，等 GPU 验证）

`abfe_preoptimizer.py`，`THERMODYNAMIC_PATH_PROTOCOL_VERSION` 现在是 **15**：

- v13/v14 的分组改动（`VANISHING_TARGET_INTERVALS_PER_ENSEMBLE=3`、
  `VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS=2`，窗口 0 现在是 3 态）**保留在代码里，
  但已经用 3 次真实 GPU 结果确认这条路走不通**——不要再调这两个常数。
- **v15（新）：`DualLambdaPreOptimizer._refine_pilot_grid_in_steep_segments`。**
  `optimize_stage2_vanishing` 跑完粗网格 pilot（18 点）后调用它：计算每个粗网格区间
  对总热力学长度的贡献占比，如果最差区间超过阈值（默认 20%），就在那个区间内部真实
  插入额外探针点（默认 4 个，均匀插在区间内部）重新测 `metric_g`，合并回 pilot 数组，
  最多重复 2 轮。**这是第一次真正改变密度算法拿到的原始数据，不是调分组参数。**

**为什么前两轮分组改动一定不够，这次不一样**：用真实缓存数据验证过——λ=1.0→0.9412
这一段目前占当前总热力学长度的 **51.5%**，但只有 2 个粗网格点定义它，`redistribute_
lambda_by_thermodynamic_length` 在这段内部只能线性插值，不管态数怎么分组都没用，因为
分组从不改变这段内部的密度分布，只改变"切到哪一格算窗口 0 的边界"。v15 直接往这段里面
真实插探针，让密度算法第一次拿到这段内部的真实数据。

**跑法**（用户自己在 GPU 节点上跑，不要 agent 代跑）：
```bash
python runabfe.py --config abfe_config.json --ligand MOL --resume
```
重点看：(1) 是否打印 `🔎 [pilot 加密]` 日志（确认触发了）；(2) 新的
`preopt_dual_vanishing.json` 里 `pilot_lambdas`/`metric_g` 是否变长且在 λ=1.0 附近
变密；(3) 窗口 0 的 state0↔state1 间距是否明显小于之前三次的 ~0.006-0.007；(4) 窗口
0 warmup 是否终于通过，或者 `mean_p`/`min_absolute_ess` 至少比三次失败明显改善。

- **2026-07-20（同一天，续）**：
  1. v15 pilot 加密版 GPU 实跑：**仍失败**（占据 98.2%，`min_absolute_ess≈1.0`）。但
     测出真实 pilot 数据显示：真正的难点是 λ≈0.96-0.97 附近一个尖锐的非单调 metric_g
     峰值（是端点 λ=1.0 本身的 ~50 倍），而且**这个峰值根本不在窗口 0 范围内**
     （窗口 0 只到 0.9848）——窗口 0 自己另有原因失败。
  2. 用户点破根本问题：**"探针"从设计上就是粗糙、近似的，不该被要求精确**——
     之前几轮一直在"把探针调得更准"，是在跟工具的本质较劲。
  3. 找到免费的真实数据：warmup 失败时 `IBSSampler.save_ibs_state` 会保留窗口 0 的
     `tmbar_history`（~1000 帧真实采样，不是探针）。用 `analyze_window0_real_tmbar_
     data.py` 对这份数据重新跑一次真 MBAR，测出窗口 0 内部 3 个真实态之间的
     **真实** Δf：state0↔1 = -25.3 kJ/mol（~10.2 kT），state1↔2 = -15.7 kJ/mol
     （~6.3 kT）——两条边隐含的 dF/dλ 几乎相等（~2694 vs ~2702 kJ/mol/单位λ），
     说明这一段本身**接近线性、不病态**，纯粹是"步子迈太大"，不是遇到了解不开的
     结构问题。
  4. **修复（`repair_stage2_window0_real_delta_f.py`）**：用真实 Δf 把窗口 0 从
     3 态扩到 7 态（6 步，每步 ~6.8 kJ/mol ~2.7 kT），不用重跑 pilot（复用已经存在
     的真实采样数据）。跟同一天早些时候写的 `repair_stage2_window0_regroup.py`
     用同一套"只改 lambdas_var/window_ranges/path_protocol_version，不碰
     protocol_key/provenance"的最小改动模式，配合 `ABFE_DEBUG_SKIP_STAGE2_
     FINGERPRINT=1` 跳过完整指纹校验（不重新触发整个 pilot）。
     `VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS` 2→6，`THERMODYNAMIC_PATH_
     PROTOCOL_VERSION` 15→16。
  5. **v16（真实 Δf 7 态）GPU 实跑：也失败了**（`mean_p`=[0.968,0.010,0.003,
     0.001,0.001,0.002,0.014]，`min_absolute_ess≈1.0`）。占据分布非单调（两端比
     中间高），跟分组阶段一样卡在 state0。**六种完全不同、每种都有真实数据支撑的
     λ-schedule 方案（6/4/3 态分组、pilot 加密、真实 Δf 均匀切分）全部失败**——
     结论是问题根本不在"λ 摆在哪"。
  6. **用户用 IBS 偏置公式指出真正的根因**：`V_bias=-kT ln Σ_k exp[-β(U_k(x)-f_k)]`
     里的 f_k 每次全新学习都从 OpenMM 力常量默认值 0.0 冷启动
     （`ibs_engine.py:2473`），SGD/TMBAR 只能从每批 ~20 帧的统计证据里"自举"出
     需要多大的偏置——如果构型跨越本来就难采，早期批次几乎看不到欠采样态的证据，
     学不出更大偏置，也就永远看不到更多证据。这是自举困境，换哪套 λ 都救不了。
  7. **修复（v20，`IBS_BIAS_PROTOCOL_VERSION` 19→20，尚未 GPU 验证）**：
     - 新增 `abfe_preoptimizer.estimate_f_k_from_pilot_ti`：对 Stage 2 pilot 早就
       测过的真实平均梯度 `mean_dU_dlambda_kJ_mol`（不是一直在用的方差代理
       `metric_g`）做热力学积分（TI）得到 F(λ)，插值到窗口自己的 λ 上，
       mean-centered 后作为该窗口第一次 learning 的 f_k 起点——这份数据 pilot
       阶段就有，不用等失败一次才能用，对任何窗口的第一次尝试都生效，不是只给
       窗口 0 的一次性补丁。`IBSWindowManagerDualLambda` 新增
       `pilot_lambdas`/`pilot_mean_dU_dlambda`（默认 `None`，向后兼容），
       `_run_dual_lambda_stage`/`run_full_pipeline` 里的调用点把 Stage 2
       `path_diagnostics` 里的真实 pilot 数据接过去，`run_all_windows` 只在
       `not is_resumed_ibs`（真全新学习，没有可复用的 resume 缓存）时注入，
       不碰 resume 语义。缺数据/数据非有限/长度不足时返回 `None`，静默回退到
       今天的 f_k=0.0 冷启动，不会报错也不会编造种子。
     - 同时把候选门 `candidate_min_absolute_ess` 从 2.5 降到 1.0：真实失败诊断
       显示候选门四项判据里 `min_overlap`/去相关样本数/端点不确定度三项早就
       过关，只有 `min_absolute_ess≈1.0`（有效样本数的数学下限，只要还有非零
       权重样本就不可能更低，这个区间里根本不是质量信号）卡在 2.5 这个门槛——
       2.5 在这个场景下意外地不可满足，调到数学下限不影响其余三项判据，也完全
       不碰独立、未变的 `final_*`/冻结验证严格门槛。

- **2026-07-20（同一天，再续）**：
  1. **v20（pilot TI 热启动）GPU 实跑：占据反而恶化**（96.8%→99.1%，state1
     `min_absolute_ess` 降到 8.6e-6）。同一批日志里还带着一个额外变量：因为
     `IBS_BIAS_PROTOCOL_VERSION` 18→19→20 的连续两次升级意外让 `_stage2_preopt_key`
     组合指纹也跟着变了（该指纹把 `ibs_bias_protocol_version` 也编码进去了），
     v16 精心用真实 Δf 摆好的 7 态 λ 被静默丢弃、重新触发了一次全新 pilot——这是
     agent 自己的操作失误，已经向用户承认，用真实读取 `preopt_dual_vanishing.json`
     的 `provenance`/时间戳确认过（n_states=18，全新生成），不是猜测。
  2. **用户直接问："符号是不是反了？"**——具体怀疑 `estimate_f_k_from_pilot_ti`
     里 pilot_lambdas 降序排列导致积分 Δλ 反号。**逐行核对代码后确认这个具体机制
     不存在**：函数一开始就 `order = np.argsort(pilot_lambdas)`，无论输入是升序还是
     降序都会被强制转成升序再做梯形积分，Δλ 恒正，不受输入顺序影响。同样核对了
     pilot 探针的有限差分代码（`_sample_scalar_metric`/`_finite_difference_
     derivative_1d`），差分对象就是 `self.param_vdw`——跟 `pilot_lambdas`
     完全同一个变量，没有隐藏的参数方向不一致。
  3. **但符号确实反了，只是反在别的地方，且证据更硬**：直接读 `IBSBiasForce`
     的能量表达式源码（`ibs_engine.py:2397` 起）逐步代数展开，验证了
     `exp(-β·V_bias(x)) = Σ_k exp(β·f_k)·exp(-β·U_k(x))`——跟论文公式完全对应，
     `n_k=exp(β·f_k)`，给某态更高的 f_k 直接、机械地推高它在混合分布里的权重
     （这一步不依赖任何物理假设，纯粹是把 CustomCVForce 表达式代数展开）。
     结合 `update_weights` 里实际生效的 `logits = beta*(f_old - u_mk)`
     （约 3364-3367 行），二者严格一致。用当前失败运行的真实数字直接验证：
     `_solve_tmbar_and_recenter`（服务于 v19 起的在线 TMBAR 更新）和新增的
     `estimate_f_k_from_pilot_ti`（v20 热启动种子）**都**把已经占主导的 state0
     判给全组最高（最不负）的 f_k——把这个值原样喂给 `context.setParameter`，
     等价于让每次更新都在火上浇油、进一步推高本来就过量代表的态，而不是压低它。
     这精确解释了 v20 为什么会让占据从 96.8% 恶化到 99.1%：热启动种子把这个已经
     存在的符号错误提前、放大注入到了学习的第一步。
  4. **根因定位在 `_solve_tmbar_and_recenter`（`ibs_engine.py`），不在
     `estimate_f_k_from_pilot_ti` 本身。** `solve_stage_integrated` 返回的
     `f_k` 来自 `df_matrix[0, 1:]`——pymbar 原始 `f_i=-ln(Z_i)` 相对"采样态"
     （row 0）的约定，对越贴近当前采样分布（也就是当前过量代表的态）给出
     越高的值，跟 `update_weights` 需要的方向正好相反。这个符号错误从
     **v19 引入 TMBAR 起就存在**——也就是说，不只是窗口 0，整条在线学习路径
     覆盖过的所有窗口，这一整天都在被这个符号问题拖后腿，这也是"换哪套 λ
     schedule 都没用"的一个新的、独立的解释（不排斥"自举困境"本身也是真的，
     两者会叠加，但符号错误更根本——正确的方向从一开始就在被主动推远）。
  5. **修复（`IBS_BIAS_PROTOCOL_VERSION` 20→21，尚未 GPU 验证）**：
     - `_solve_tmbar_and_recenter`（`ibs_engine.py`）现在对
       `solve_stage_integrated` 的原始 `f_k` 先取负号，再 mean-center。
     - **同时暂时禁用 v20 的 pilot TI 热启动注入**（`run_all_windows` 里对
       `estimate_f_k_from_pilot_ti` 的调用已注释掉，`warm_start_seed` 强制
       `None`，f_k 退回 v20 之前的 0.0 冷启动）——原因：`pilot_mean_dU_
       dlambda_kJ_mol` 是每个 λ 独立短程弛豫后测的系综平均梯度，跟
       `update_weights` 实际用的 `u_mk`（在当前占主导构型上跨态求值）是概念
       不同的量，其符号是否同样"反了"、还是本来就该是这个方向（比如端点附近
       真实存在一个独立弛豫才能跨过的能垒，这本身可能是有意义的物理发现）
       **没有独立验证过**。这次先只验证 TMBAR 符号修复这一个变量，避免跟一个
       未经验证的第二个改动叠加、又做出一次混杂了多个变量的测试。
     - 之前 v16 用真实 Δf 摆好的 7 态 λ 这次**没有被破坏**（`protocol_key`
       没变、这次全程没有再碰 `IBS_BIAS_PROTOCOL_VERSION` 以外的指纹字段），
       理论上不需要 `ABFE_DEBUG_SKIP_STAGE2_FINGERPRINT=1`，但仍建议带上
       作为保险（万一判断有误，这个环境变量本身无副作用，只是跳过一次
       本来就该通过的指纹检查）。

- **2026-07-20（同一天，又续；v22，是 v21 同一修复的第二个调用点）**：
  1. 用户用混合分布的平衡条件 `f_k - U_k ≈ const` 独立核对了 pilot TI seed：
     在 state0（λ=1、`U_k` 最负）已经占主导时，它必须拿到全组最低的 f_k；否则
     `n_k=exp(βf_k)` 会继续抬高它自己的权重。这与 v21 对在线 TMBAR 的修复方向相同。
  2. 用真实 `output_lrc_fix/checkpoints/preopt_dual_vanishing.json` 的 pilot 数据和
     同目录 `ibs_state_vdw_window_0.json` 记录的 v20 窗口 λ，v20 原始 seed
     `[+48.2, +35.3, +20.0, +4.3, -15.2, -35.9, -56.7]` 反号后得到
     `[-48.2, -35.3, -20.0, -4.3, +15.2, +35.9, +56.7]`（随 state0→state6
     单调增加），正是上述平衡条件要求的方向；原始 pilot-TI 与 v21 修复前的原始
     TMBAR 都呈现“占主导态拿最高 f_k”的同一种错误模式。
  3. 修复（`IBS_BIAS_PROTOCOL_VERSION` 21→22）：
     `estimate_f_k_from_pilot_ti` 现在先对原始 TI 插值结果取负，再 mean-center，
     明确返回可直接注入的 bias-parameter convention；`run_all_windows` 恢复 v20 的
     pilot-TI warm-start 调用及注入。原有长度校验和 injected/rejected/no-data 三条日志
     保持不变。v21 的 `_solve_tmbar_and_recenter` 修复不动，λ 布局、候选门、冻结验证和
     `non_mutating_v1` 策略也全部不动。

- **2026-07-20（v23；方向有效，但“绝对向量覆盖”方法错误）**：
  1. v22 真实 GPU 运行的 seed 为 `[-40.364, -29.811, -16.062, -1.951,
     +13.178, +29.229, +45.781]`（当前 7 态 λ 与旧 v20 快照不同，所以不是旧的
     `[-48.2, ..., +56.7]`；符号和单调方向一致）。它确实把原先约 99% 的 state0
     主导推出去了，说明“用 TI 给非零起点”这条思路有效。
  2. 但 50 次更新后占据完整翻到 state6：`mean_p=[~0, ~0, ~0, ~0,
     3.5e-7, 0.00469, 0.99531]`，落盘 f_k 发散到
     `[-1197.6, -53.1, +156.4, +237.2, +272.8, +288.1, +296.1] kJ/mol`；
     TMBAR `min_absolute_ess≈1`、`converged=False`，最后一步仍漂移约 5.8 kJ/mol。
     这不是“差一点收敛”，而是从一个端点塌缩翻成另一个端点塌缩。
  3. 根因：v19-v22 的 `update_weights` 无论 TMBAR 是否收敛，都会把一次 TMBAR
     绝对向量整组覆盖到 Context。取正号或反号都不是权重平均；mean-center 只消除
     任意规范零点，完全不会缩小 f_k 跨度。
  4. v23 修复：pilot-TI 继续提供非零起点；每个真实 batch 对全部态同步执行
     `delta_f_k=-eta*kT*log(K*<p_k>)`，log 残差限制在 ±2，每轮整个向量的最大
     `|delta_f_k|` 再限制为 2 kT，并使用 `eta_penalty/(1+t/100)` 衰减。强态必降、
     弱态必升；TMBAR 历史仍完整累计，但只用于候选覆盖质量门和诊断，不再把任何
     正号/反号绝对解注入。冻结验证失败的累计 p 也走同一条受限反馈，并将
     `eta_penalty` 减半；该 penalty 已进入 v23 resume 状态。
  5. **v23 首次 GPU 运行：权重学习已经成功，但被浮点边界误拒绝。** 50 次更新后
     `mean_p=[0.1608, 0.1338, 0.1458, 0.1372, 0.1291, 0.1366, 0.1568]`，
     LSE `max_abs_log_residual=0.118<0.25`、coverage ESS `6.958/7`；强态在整个
     更新序列中按设计轮换，没有再次端点塌缩。未进入冻结验证的唯一原因是 TMBAR
     两个 inclusive 门被 roundoff 卡住：`0.04999999999999431 >= 0.05` 和
     `0.9999999999998863 >= 1.0` 在机器比较中均为 False；其余两门
     `10>=5`、`3.381<=5` 已通过。修复只把距阈值 `1e-12` 内的数视为数学相等，
     `0.049999`/`0.999999` 仍会拒绝，物理门槛没有放宽，因此不升协议版本并保留
     已学好的 v23 state。resume 的 `min_bias_updates` 也改为计入持久化的
     `f_history`，仍要求 3 次全新的连续候选通过后才冻结验证。

**跑法**：
```bash
ABFE_DEBUG_SKIP_STAGE2_FINGERPRINT=1 python runabfe.py --config abfe_config.json --ligand MOL --resume
```
重点看：(1) 第一条 `🌱 [pilot TI 热启动]` 仍应是当前 λ 对应的约
`[-40.4, -29.8, -16.1, -2.0, +13.2, +29.2, +45.8]`；(2) 每轮
`weight_update.method=bounded_log_occupancy_v1`，且 `max_abs_delta_f_kJ_mol`
不超过约 2 kT（300 K 时约 5 kJ/mol）；(3) 若 state6 过强，它的 f_k 必须逐轮下降，
其余弱态同时上升，不能再出现跨度突然跳到上千 kJ/mol；(4) `mean_p` 应从端点饱和
逐渐向中间铺开，随后 TMBAR ESS/覆盖门才有资格通过并进入冻结验证。

## 已经验证过是错的，不要再试（省得浪费 GPU 时间）

1. **只调大 `stage2_n_states`（比如 18→26）不够。** 用真实缓存的 pilot metric_g 算过：
   第一段 pilot 区间（λ 1.0→0.9412）单独就占了全程热力学长度的 **46.7-51.5%**（两次
   pilot 重跑的读数），要把窗口 0 收窄到历史上跑通过的量级，总态数得堆到 **~64**，而且
   pilot 探针本身只有 8-18 个粗网格点，堆再多态数也解不出这段真实需要的分辨率。
2. **给密度估计加一个类似老版本（`v0.9.3-a5` tag）的上限（`max_ratio=0.15` + log1p 压缩）
   没用，反而更差。** 实算过：加了上限后窗口 0 会变宽到 λ 1.0→0.934（现在是 0.963），
   不是变窄。**不要重新引入这个上限**，`redistribute_lambda_by_thermodynamic_length`
   函数自己的注释里也解释了为什么之前特意去掉了 log1p/clip（会压缩真实高方差区域）。
3. **调整窗口分组/态数（6→4→3 态，三次独立真实 GPU 结果）完全没用，占据比例几乎不变
   （99.4%→96.9%→98.2%）。** 用户用 IBS 偏置公式直接证明了为什么：分组只改"哪些态
   共用一个 bias"，从不改变任何一个 λ 值本身，而三次用的都是同一条 18 态数组的子集，
   state0↔state1 间距全程没变过。**不要再调 `VANISHING_TARGET_INTERVALS_PER_ENSEMBLE`/
   `VANISHING_MIN_INTERVALS_PER_ENSEMBLE`/`VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS`
   这三个常数，这条路已经被 3 次真实数据关闭。**
4. **pilot 加密（v15）和真实 Δf 均匀切分（v16）也都真实跑过、都失败了。** 不管
   λ-schedule 怎么摆（探针方差、探针加密、真实 MBAR 测出的 Δf 均匀切分），窗口 0
   占据始终卡在 state0 96-99%、`min_absolute_ess≈1.0`。**不要再往"λ 该摆在哪"这个
   方向猜——六次真实 GPU 结果已经把这整条路排除了。**
5. **v20（pilot TI 热启动，未反号版本）真实跑过、占据反而更差（96.8%→99.1%）。**
   根因是与 v21 在线 TMBAR 相同的符号错误，不是“热启动这个思路本身不对”。这个
   第二调用点已由 v22 按平衡条件独立核对并反号后重新启用；后续不要恢复 v20 的原始
   正号约定，也不要把 v20 的失败误记成 pilot-TI warm-start 本身已被否定。

## 给下一个 agent/会话的规则（不是客套话，是教训）

1. **推荐任何 λ 路径/窗口设计的改动之前，先用已经缓存的真实数据（
   `output/checkpoints/preopt_dual_vanishing.json` 的 `pilot_lambdas`/`metric_g`，或者
   任何存在的旧 `output_lrc_fix`/其它 tag 的产出）离线算一遍，确认数字真的支持这个方向，
   再开口推荐。** 这次会话里，"调大 n_states"和"加密度上限"两次都是先讲了道理才发现算
   出来是错的——道理讲得通不代表数字对得上，必须真的算。
2. **动手改之前，先查有没有旧的、已经跑通过的证据**（老 tag、`output_lrc_fix`、
   `AUDIT_STATUS.md`）——07-17 那次"rescue audit 从未执行"就是没做这一步的直接后果，
   导致后面所有协议升级都从零重来，而不是先核对"以前到底怎么跑通的"。
3. 不要把"看起来更严谨/更符合论文"当成"实际更好"的证据（thermodynamic-length 方法从
   数学上确实更贴近论文 Sec 2.3，但对这个具体配体的端点，缺一个正则化上限就是在实践中
   更差）。
4. 如果 v15（pilot 加密）GPU 跑完还是失败，**不要凭感觉再猜一个新方案**——回来把这份
   文档更新（新的失败证据 + 已经排除的选项），再决定下一步。分组/态数这条路已经彻底
   排除；下一步大概率是 (a) 认真评估复活/改造 `refine_stage_lambda_path_from_data`
   （实测 |Δf|，2026-07-17 曾用它真正跑通过这个端点——见 `repair_stage2_window_
   partition.py` 里的说明，它测的是真实自由能差而不是短探针方差代理，可能比"pilot
   加密"更根本），或 (b) 结构/pose 诊断。
