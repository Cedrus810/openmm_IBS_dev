# EXP-017 计划：Overlap-first 的非神经 λ 路径与条件性解析 CV

> **文档状态：** `DRAFT / NOT_SEALED / NO_EXECUTION_AUTHORIZED`  
> **日期：** 2026-08-10  
> **实验编号：** `EXP-017`（当前仓库未发现占用）  
> **建议输出根目录：** `output/outer_lambda_exp017_overlap_first/`  
> **后续机器可读协议：** `protocols/EXP-017_preregistration.json`（尚未创建）

本文件只定义研究问题、阶段顺序、指标、晋级门、停止条件和工程边界。它不是运行授权，
也不允许仅凭本文启动 MD、修改 production 配置或覆盖既有缓存。正式执行前必须：

1. 将全部待冻结字段写入机器可读 preregistration；
2. 记录输入、代码、环境和 baseline artifact 的 SHA-256；
3. 在 `EXPERIMENT_LOG_outer_lambda_neural_basis.md` 登记 `PLANNED`；
4. 为每个实际命令使用新的空输出目录；
5. 明确计算节点、GPU、OpenMM/PyTorch/PyMBAR 版本和 wall-clock 计量口径。

---

## 1. 背景与立项边界

EXP-012 至 EXP-016 已经给出以下冻结事实：

- frozen-MACE latent 和 ORB layer-2 latent 都含有离线相邻态 gap 信息；
- `LocalResidualStudent` 只保留部分离线信号；
- 当前每 MD step 的 TorchForce 路线成本过高，正式 `ESS/GPU-hour` 没有改善；
- EXP-013 的三种在线/MTS 方案均未晋级；
- EXP-014 的 typed-pair radial compression screen 未通过；
- EXP-016 只有 energy-weighted surrogate，没有 physical alchemical-state/replica history，
  因而没有形成 physical learned slow-information 证据；
- ORB 的 matched-CUDA 在线成本门失败，最终角色为 `OFFLINE_TEACHER_ONLY`。

因此 EXP-017 **不是**继续优化当前 MACE student，也不是重新搜索 MTS 间隔。它先回答更窄、
更便宜而且可证伪的问题：当前困难窗口的效率瓶颈能否由 λ 路径本身解释和修复；若不能，
是否存在一个无需在线神经 encoder 的低维静态解析坐标可以提供额外收益。

EXP-017 是独立新实验，不自动重开 EXP-013、EXP-014、WP-5 或 production 接入。

---

## 2. 唯一研究问题

在冻结基础 Hamiltonian、物理端点和 IBS/TMBAR 记账语义的前提下：

> 能否先用现有 TMBAR/ledger 与真实 fixed-λ 双向 overlap probe 定位困难边，再通过一份
> 冻结的 λ-only 路径候选；必要时再通过一个 1D、最多 2D 的静态解析
> \(q(\mathbf R)\) 与 \(V_{\mathrm{bias}}(q)\)，在不调用任何在线 MACE、ORB 或 student
> 的情况下，提高相对 baseline 的 mixture-coverage ESS/GPU-hour，同时保持端点、完整
> target/bias/base ledger、TMBAR 自由能和结构健康一致？

EXP-017 **不回答**：

- 是否发现了 physical learned slow variable；
- student force 是否可以低频冻结；
- MACE/ORB 是否可以进入每步 OpenMM；
- DEXP 是否优于 softcore；
- post-hoc reweighting 是否能补救没有采到的构象。

上述问题若仍需研究，必须另立实验。

---

## 3. 冻结 baseline

除非机器可读协议明确列出并重新哈希，以下内容不得改变：

| 项目 | 冻结值/规则 |
|---|---|
| 方法 | `mode=ibs` |
| decoupling | `dual_lambda` |
| 基础势 | `softcore` |
| DEXP | `dexp_params=null` |
| 困难对象 | complex / vanishing / Stage 2 / window 0 |
| 当前局部状态 | schedule indices `[0,5)`；不能把局部 state 0/1 误写成物理 λ 端点 |
| 温度 | `300 K` |
| λ 协议来源 | 当前 v21 artifact 与 `path_protocol_version=21`；以实际 artifact/hash 为准，不手抄旧 v20 λ |
| IBS/TMBAR | 沿用当前生产版本、`ESS_GATE_PROTOCOL_VERSION=3` 及完整 target/bias/base 账本 |
| neural path | 关闭；MACE/ORB/student 只允许离线 attribution，不进入 MD force |
| production 模块 | 第一阶段不修改 `runabfe.py`、`abfe_core.py`、`abfe_pipeline.py`、`ibs_engine.py` 的默认行为 |

当前 v21 schedule、window ranges、checkpoint、`f_k`、LRC、topology、System XML 和 box 均须
从实际 artifact 读取并登记 hash。EXP-012 preregistration 中封存的旧 schedule fingerprint
只作历史输入证据；若它与当前 v21 production artifact 不同，禁止静默混用。

---

## 4. 术语与指标口径

### 4.1 production/TMBAR overlap

IBS 每个窗口只有一个真实采样的 mixture row；物理 λ rows 的 `n_k=0`。因此常规
`pymbar.compute_overlap()` 在这种“一个采样分布 + 多个零样本目标态”场景会退化，不能作为
production/TMBAR 的 overlap 矩阵。

EXP-017 对 production/TMBAR 使用仓库当前正式口径：

- `min_overlap`：去掉共模因子后的 per-window mixture-coverage ESS ratio；
- `min_decorrelated_samples`：最差窗口的去相关样本数；
- `max_endpoint_uncertainty_kJ_mol`：窗口端点自由能差的最大不确定度；
- `min_absolute_ess`：继续报告，但因其与 `min_overlap × N_decorrelated` 同构，不作为第二个
  独立硬门；
- `raw_min_overlap`、`raw_min_absolute_ess` 和 common-mode log-weight spread：仅作防护壳/LRC
  重加权税诊断。

### 4.2 fixed-λ 双向 overlap

只有 `ibs_engine.probe_bidirectional_overlap()` 产生的两条独立 fixed-H NVT 轨迹可以使用
普通两态 MBAR overlap matrix。该 probe：

- 分别从相邻的两个固定 Hamiltonian 采样；
- 不含 IBS bias，也不含 WCA sampling shell；
- 每帧在两个 target Hamiltonian 下重新评价能量；
- 报告 `min_bidirectional_overlap=min(O_01,O_10)`、BAR/MBAR Δf 与不确定度；
- 默认通过阈值为 `0.03`。

fixed-λ probe 只用于判断 path/λ-grid 的相邻边是否需要插点；它的 Δf 不能用来校准 production
IBS bias 的 `f_k`，因为两者采样 Hamiltonian 不同。

### 4.3 唯一主性能指标

对每个独立 repeat \(r\)，定义：

\[
\eta_r = \frac{\min_{w,k}N_{\mathrm{eff},w,k}^{\mathrm{mixture}}}
{\mathrm{GPU\ hour}_r},
\qquad
D_r = \log \eta_r^{\mathrm{candidate}}-\log \eta_r^{\mathrm{baseline}}.
\]

其中分子必须来自当前 `ESS_GATE_PROTOCOL_VERSION=3` 的 mixture-coverage ESS，不能换成
literal BAR overlap、raw single-reference ESS 或 EXP-016 surrogate switch count。

运行时间必须包括该 arm 的 warmup、正式采样和实验协议要求的在线能量/账本评价；不得只计
integrator 内核或排除对 candidate 不利的固定开销。baseline 与 candidate 必须在同一型号 GPU、
同一 OpenMM platform/precision、相同采样步数和输出频率下计量。

### 4.4 correctness 硬门

主性能指标只有在以下全部通过时才有资格参与比较：

- TMBAR `converged=true`；
- `min_overlap`、`min_decorrelated_samples` 和 endpoint uncertainty 达到当前正式门；
- target/bias/base ledger 闭合且全部 finite；
- 所有预期窗口和 λ indices 完整覆盖；
- baseline/candidate 的 ΔG 在联合统计误差内一致；
- 温度、能量、力、约束和结构异常门通过；
- 物理端点与基础路径未改变。

---

## 5. 总体阶段与固定分叉

```text
P0-A 现有数据/ledger 可行性与 null 审计（不新增 MD）
        ↓ 通过
P0-B 只对被定位的相邻边做 fixed-λ 双向 overlap probe
        ↓ 有实测失败边或合格的不对称瓶颈
P1   生成且只冻结一个 λ-only candidate，做正式 baseline 对照
        ├─ 达到 production promotion 门 → EXP-017 以 λ-only 结论结束，不做 P2
        ├─ 健康但收益不足，且 P0 有解析 q 的独立增量证据 → P2 preregistration addendum
        └─ correctness/数据门失败 → EXP-017 STOP
P2   条件性 1D/2D static analytic q + tabulated/spline bias
        ├─ 全部门通过 → 只登记为新的 qualification candidate
        └─ 任一门失败 → EXP-017 STOP
```

不得并行搜索多个 schedule、多个模型或多个 bias 幅度，然后事后挑最好结果。

---

## 6. P0-A：现有数据与 null 审计

### 6.1 目的

在不运行新 MD、不训练新 neural model 的情况下回答：

1. 当前数据能否重算正式 TMBAR mixture-coverage 指标；
2. 瓶颈是特定窗口/λ edge，还是整个窗口被慢构象或未收敛 `f_k` 一致拖低；
3. 已有物理描述符是否在 λ-only 诊断之外保留跨 run、连续时间块一致的增量信息；
4. 是否有资格启动 fixed-λ probe 或 analytic-q addendum。

### 6.2 冻结输入

- `hard_window0_run1/2/3` 三条连续轨迹，各 500 帧，实际 `Δt_save=1 ps`；
- 对齐的 MM ledger、sample report、frozen `f_k`、LRC 与 schedule artifact；
- EXP-016 data manifest 和 temporal audit；
- frozen-MACE、ORB layer-2、student cache，只作离线 attribution；
- 已登记物理描述符：primary/secondary torsion、`VAL251 chi1`、ligand–protein contact；
- hydration 只有在从原轨迹按预先冻结定义重建逐帧序列、保存 provenance/hash 后才可使用，
  不能把 run-level summary 广播成逐帧数据。

### 6.3 必做检查

- [ ] trajectory、ledger、latent、frame index、run id、时间间隔和原子 mapping 完全对齐；
- [ ] 所有输入存在 SHA-256，单位和 λ state mapping 一致；
- [ ] 使用实际 frozen `f_k` 重算 per-state mixture ESS 与 `min_overlap`；
- [ ] 同时报告 raw ESS/common-mode spread，但不把 raw 值当主门；
- [ ] 按完整 run 做 leave-one-run-out；禁止随机 frame split；
- [ ] attribution/association 使用连续 block；默认沿用 EXP-016 的 128-frame circular block
  bootstrap、2000 replicates；
- [ ] direct-gap student、adjacent-gap 和 energy-weighted argmin label 标为 target-derived；
- [ ] 运行 run-label/null/permutation 对照，拒绝把 run identity 当坐标；
- [ ] 对窗口级低 ESS 使用 `plan_vdw_overlap_repair_targets` 的 split-first 分类，不能直接从
  “某态 ESS 最差”推断相邻 λ edge 太宽；
- [ ] 检查现有数据是否足以重建完整 TMBAR 结果；缺字段时 fail closed，不以零填充。

### 6.4 P0-A 输出分类

只允许以下结论：

- `P0A_LEDGER_ELIGIBLE_EDGE_LOCALIZED`：账本合格，且有明确相邻边需要 fixed-H probe；
- `P0A_LEDGER_ELIGIBLE_WINDOW_LEVEL_ONLY`：账本合格，但只能定位到整个窗口；先按 split-first
  规则处理，不得盲插 λ；
- `P0A_ANALYTIC_Q_INCREMENTAL_SIGNAL`：在 λ-only/null/run controls 之后仍有 1D/2D 静态几何
  的 held-out block-level 增量信息；只授权编写 P2 addendum，不授权 MD；
- `P0A_SURROGATE_ONLY`：只能复现 target-derived/surrogate 信号，不晋级；
- `P0A_INELIGIBLE`：provenance、ledger 或支持域不完整，EXP-017 立即停止。

P0-A 不授予任何 production、slow-information 或 bias 资格。

---

## 7. P0-B：fixed-λ 双向 overlap probe

### 7.1 启动条件

只有 P0-A 或 split-first 后的正式诊断明确列出需要测量的相邻边，才允许运行。

### 7.2 冻结协议

第一版复用 `ibs_engine.probe_bidirectional_overlap()`：

| 参数 | 值 |
|---|---:|
| ensemble | 两个独立 fixed-H NVT |
| burn-in | `5000 steps` |
| sampling | `20000 steps` |
| interval | `500 steps` |
| integrator | 与 production 一致的 `LangevinMiddleIntegrator`，2 fs |
| overlap threshold | `0.03` |
| Hamiltonian | `U_common + single cv_int`，无 IBS bias、无 WCA shell |

默认函数内部 seed 只够做 edge-screening，不构成独立 production repeat。若需要把 probe 本身用于
统计选择，执行前必须通过 addendum 增加显式 `seed_base`、预登记至少 3 个独立 seeds，并为每个
seed 保存两态轨迹、`u_kn`、`n_k`、IAT、decorrelated count 和 overlap matrix。

### 7.3 唯一允许的 schedule 生成规则

- 只有 `min_bidirectional_overlap < 0.03` 的实测失败边，或仓库规则认定的
  passed-but-asymmetric bottleneck，才允许插点；
- 使用 `insert_lambda_from_overlap_failure()`；只插入该边的算术中点；
- EXP-017 最多生成 **一个** λ-only candidate，最多新增 **一个** λ state；
- 插点后所有 thermodynamic-length cache、preoptimization cache 和 window indices 全部失效；
- 新路径必须重新生成 window ranges、协议版本和 fingerprint；
- 禁止手工移动到“看起来更好”的 λ，禁止看 P1 结果后再做第二轮插点。

如果没有任何 measured failure/asymmetry，结论必须是
`P0B_NO_LAMBDA_INSERTION_JUSTIFIED`；不得为了继续实验而制造 schedule candidate。

---

## 8. P1：λ-only 正式对照

### 8.1 两臂

1. `baseline_v21`：当前冻结 production v21 path/window protocol；
2. `candidate_lambda_only`：P0-B 唯一生成的候选。

两臂不得加入 neural/analytic bias，不得改变基础势、IBS update 规则、温度、步长、采样长度、
输出频率或 estimator。

### 8.2 独立重复

- 至少 3 个真正独立 repeats；论文级结论建议 5 个；
- repeat 必须有独立初态/独立平衡历史和预登记 seed；
- repeat 内可 baseline/candidate 配对以降低硬件噪声；
- 从同一 checkpoint 只重抽速度的两臂不是两个独立 repeats；
- 运行顺序应交替或随机化并记录，避免 GPU 热状态/节点负载系统偏差。

### 8.3 primary 与 secondary endpoints

**唯一 primary：** 每个 repeat 的 \(D_r=\log\eta_{candidate}-\log\eta_{baseline}\)。

**correctness/co-safety：**

- TMBAR convergence 三硬门；
- fixed-λ probe 的 bidirectional overlap（只对被修改的 edge）；
- ΔG 与联合误差；
- endpoint、ledger、完整窗口/λ coverage；
- temperature/energy/constraint/structure health。

**secondary/exploratory：**

- per-window/per-state mixture ESS ratio；
- `min_absolute_ess`、raw ESS、common-mode spread；
- gap variance、IAT、`N_eff`；
- torsion/contact/chi1 转换；
- occupancy 与 surrogate switch count；
- wall time 分解、显存和 I/O。

secondary 指标不得替代 primary；没有 discrete physical state history 时不得报告 round trip 或
physical state crossing。

### 8.4 P1 promotion 门

只有同时满足以下全部条件才登记 `EXP017_LAMBDA_ONLY_PROMOTED`：

1. 所有 repeats 的 correctness/co-safety 硬门均通过；
2. 至少 `2/3` 独立 repeats 的 `D_r > 0`；
3. median \(\exp(D_r)-1\) 至少为 `10%`；
4. 没有 candidate 特有的结构异常、温度偏移或 estimator 不收敛；
5. 收益不能仅来自减少采样步数、降低输出频率或遗漏失败窗口。

通过后 EXP-017 在此结束；λ-only candidate 仍只是 qualification candidate，不能自动进入完整
production 或跨体系结论。

### 8.5 P1 停止/分叉

- correctness 任一失败：`EXP017_STOP_LAMBDA_ONLY_INCORRECT`，整个 EXP-017 停止；
- 健康但主性能不足，且 P0-A 没有 analytic-q 增量证据：
  `EXP017_STOP_NO_CHEAP_INCREMENTAL_SIGNAL`；
- 健康但主性能不足，且 P0-A 已有 analytic-q 增量证据：只允许起草并 seal P2 addendum；
- 不得根据 P1 结果再插第二个 λ 点或修改候选 schedule。

---

## 9. P2：条件性 static analytic q

### 9.1 启动条件

P2 默认阻塞。只有 P1 健康但未达到 material performance gate，且 P0-A 已登记
`P0A_ANALYTIC_Q_INCREMENTAL_SIGNAL`，才能编写 P2 addendum。addendum 未 seal 前禁止训练、
实现 OpenMM force 或运行 MD。

### 9.2 addendum 必须冻结的内容

- 唯一 q 定义：第一版 1D，只有预注册门证明不足时才允许 2D；
- 原子 stable ids、PBC/unwrap、单位、cutoff 与平滑函数；
- target schema：block/trajectory-level overlap/coverage 目标，不复用 EXP-010 的“低维 torsion
  逐帧拟合高维瞬时 interaction energy”协议；
- `V_bias(q)` 的唯一函数族、knots/order、正则、幅度上限和中心化；
- 外层包络和缩放；
- training/validation/test 的连续 run/block 划分；
- finite-difference、PBC、cutoff、近接触、动态水和非参与原子零力门；
- CPU/Reference/CUDA 性能预算；
- checkpoint/config/hash 和不得事后扩网格的规则。

### 9.3 允许的 Hamiltonian

\[
B_\lambda(\mathbf R)=\sin^2(\pi\lambda)\,V_{\mathrm{bias}}(q(\mathbf R)).
\]

- q 必须是当前坐标的静态函数，不得依赖历史；
- 每步只计算 cheap analytic q 和解析/tabulated bias；
- MACE、ORB 和 student 只能离线提供 attribution，不能进入 force path；
- 该项属于 target Hamiltonian，必须进入 target energy 和跨态 reduced potential；
- 不能把它整体塞入需要消除的 IBS `bias_history`；
- λ=0/1 的能量和力必须严格回到基础路径。

### 9.4 P2 阶段门

1. `P2-C0 offline compression`：LORO/连续 block、null/run-label controls、support/OOD；
2. `P2-C1 force qualification`：解析/FD 梯度、PBC、cutoff、端点、finite force；
3. `P2-C2 backend qualification`：CPU/Reference/CUDA 一致性与 matched-path 成本；
4. `P2-C3 short NVT`：至少 3 seeds 的绝对健康门；
5. `P2-C4 formal comparison`：沿用 P1 的独立重复、primary 和 correctness 门。

任一子阶段失败立即停止，不跳过失败门，不通过扩大模型/网格救援同一结果。

### 9.5 P2 promotion 门

沿用 P1 的全部 correctness 与 primary 门，并额外要求：

- analytic q 的 matched-path 增量成本不抵消 ESS；
- held-out support 内无异常梯度/饱和；
- 至少 `2/3` 独立 repeats 的正式 ESS/GPU-hour 优于 baseline；
- median material improvement 至少 `10%`；
- ΔG 与 baseline 在联合误差内一致；
- 若扩大到 complex/solvent，两条腿与 cycle closure 全部通过后才允许讨论 ABFE production。

通过只能登记为 `EXP017_ANALYTIC_Q_QUALIFICATION_CANDIDATE`，不能声称 physical learned slow
information。

---

## 10. 全局禁止事项

- 不重调 EXP-012/013 的 `c1`；
- 不切换或重训 LocalResidualStudent checkpoint；
- 不恢复 real-time TorchForce、MTS、rRESPA 或 blockwise adaptive Hamiltonian；
- 不把 ORB/MACE latent 直接作为在线坐标；
- 不用 EXP-016 energy-weighted switch 冒充 physical crossing；
- 不把 literal BAR overlap 与 IBS production mixture-coverage ESS 混称；
- 不使用旧 v20 λ、旧 cache 或不同 schedule fingerprint 的 `f_k`；
- 不把 post-hoc ranking/reweighting 当作采样收益；
- 不同时改变 λ schedule、基础势、IBS update、analytic q 和 estimator；
- 不用随机 frame split，不把同一 checkpoint 的 paired reseed 当独立 N；
- 不在失败后扩大 candidate 集合、换 seed、换目标或重写主指标；
- 不覆盖任何既有实验目录或有效报告。

---

## 11. 工程实现边界

### 11.1 优先复用

- `ibs_engine._ibs_reweighting_quality_diagnostics`：mixture-coverage ESS；
- `GlobalMBARAnalyzer.solve_stage_integrated`：唯一 Stage-2 TMBAR estimator；
- `abfe_preoptimizer.plan_vdw_overlap_repair_targets`：window-level split-first 分类；
- `ibs_engine.probe_bidirectional_overlap`：真实 fixed-H 双向 overlap；
- `abfe_preoptimizer.insert_lambda_from_overlap_failure`：measured failure 后的单点插入；
- 当前 ledger、resume/cache fingerprint 和 stage coverage 检查。

### 11.2 P0/P1 允许新增

- `scripts/audit_exp017_overlap_first.py`：只读 ledger/provenance/null audit；
- `scripts/run_exp017_fixed_lambda_probe.py`：受控调用既有 fixed-H probe；
- `scripts/run_exp017_lambda_only_pilot.py`：两臂、独立 repeats、统一计时和报告；
- `protocols/EXP-017_preregistration.json`；
- 对应单元测试和 JSON schema。

### 11.3 当前禁止修改

- production 默认配置；
- 既有 production cache；
- `outer_lambda_neural_basis.py` 的 MACE/student 路径；
- EXP-012/013/014/016 的有效报告和封存 artifact。

P2 的 analytic-q 接口要等 addendum seal 后另列文件，不在本计划中预先授权实现。

---

## 12. 输出与 provenance

建议目录：

```text
output/outer_lambda_exp017_overlap_first/
  p0a_audit/
    EXP-017_P0A_manifest.json
    EXP-017_P0A_audit.json
    EXP-017_P0A_summary.md
  p0b_fixed_overlap/
    <edge>/<seed>/...
    EXP-017_P0B_overlap_report.json
  p1_lambda_only/
    baseline_v21/<repeat>/...
    candidate_lambda_only/<repeat>/...
    EXP-017_P1_report.json
  p2_analytic_q/                 # 仅 sealed addendum 后允许创建
    ...
```

每份 machine-readable report 至少包含：

- experiment/phase/status/schema version；
- 实际命令和全部参数；
- 输入及输出 SHA-256；
- code commit；若无 Git 仓库，显式登记 `git_commit=null` 和关键文件 hash；
- Python/OpenMM/PyMBAR/PyTorch/CUDA/driver/GPU；
- schedule/path/window/System/topology/checkpoint/`f_k`/LRC hashes；
- seed、初态来源、独立性说明；
- wall-clock 计量范围；
- primary/correctness/secondary 指标；
- failed gates、异常和最终 decision；
- 不允许静默忽略 NaN、丢帧、未收敛 estimator 或缺失窗口。

所有脚本必须 refuse to overwrite non-empty output directory，并使用原子写入或临时文件后 rename。

---

## 13. 状态机与最终结论词汇

允许的阶段状态：

- `PLANNED`
- `RUNNING`
- `PASSED`
- `FAILED`
- `INCONCLUSIVE`
- `STOPPED`
- `SUPERSEDED`

允许的 EXP-017 决策：

- `P0A_INELIGIBLE_STOP`
- `P0B_NO_LAMBDA_INSERTION_JUSTIFIED`
- `EXP017_LAMBDA_ONLY_PROMOTED`
- `EXP017_STOP_LAMBDA_ONLY_INCORRECT`
- `EXP017_STOP_NO_CHEAP_INCREMENTAL_SIGNAL`
- `EXP017_P2_ADDENDUM_ALLOWED`
- `EXP017_ANALYTIC_Q_QUALIFICATION_CANDIDATE`
- `EXP017_STOP_ANALYTIC_Q_FAILED`

禁止使用：

- `FOUND_PHYSICAL_SLOW_VARIABLE`
- `PRODUCTION_READY`
- `MACE_STUDENT_FIXED`
- `ORB_ONLINE_APPROVED`

除非另有相应独立实验完整证明。

---

## 14. Seal 前待完成清单

- [ ] 将 EXP-017 登记为唯一未占用 registry identity；
- [ ] 冻结当前 v21 baseline artifact 和 hash；
- [ ] 核对 EXP-012 preregistration schedule 与当前 production v21 artifact 的差异；
- [ ] 冻结 P0 输入文件、hash、run independence 和缺失字段；
- [ ] 实现并测试 P0-A 只读 audit；
- [ ] 冻结 fixed-λ probe 的边选择和默认参数；
- [ ] 决定 probe 是否只作 screening；若参与统计选择，先增加显式 seed contract；
- [ ] 冻结 P1 独立初态、seeds、步数、输出频率、硬件和计时口径；
- [ ] 冻结 primary material-improvement 门 `10%`；
- [ ] 冻结 ΔG 联合误差与结构健康门的具体计算；
- [ ] 生成 `protocols/EXP-017_preregistration.json` 并通过 schema/test；
- [ ] 在实验日志登记 `PLANNED`，记录 plan/protocol SHA-256；
- [ ] 未完成上述项目以前，保持 `NO_EXECUTION_AUTHORIZED`。

---

## 15. 推荐执行顺序

```text
文档/协议 seal
  → P0-A 只读 audit
  → split-first 分类
  → 必要 edge 的 P0-B fixed-λ probe
  → 最多一个 λ-only candidate
  → P1 三个独立 repeats
  → 若 λ-only 达标：停止并保留最简单方案
  → 若健康但不足且确有 cheap-q 增量信号：另 seal P2 addendum
  → P2 C0→C1→C2→C3→C4
  → 任一失败：回滚 baseline，终止 EXP-017
```

当前下一动作不是运行模拟，而是将本 draft 转成机器可读 preregistration，并完成 P0-A 的
只读数据/指标可行性审计设计。
