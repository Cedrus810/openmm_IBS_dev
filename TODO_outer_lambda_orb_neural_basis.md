# TODO：ORB-v3 浅层神经基势分支实验

> 建立日期：2026-08-09  
> 实验身份：ORB-v3 作为 frozen representation / shallow scalar basis 的独立分支  
> 当前状态：`ORB-001 PRIMARY_STATISTICAL_GATE_PASSED / ORB-003 CUDA_MATCHED_COST_FAILED / ORB-004_005_STOPPED`；父体系 charge/spin conditioning contract、actual-edge equivalence、初始化/性能 gate、1500-frame cache 和 EXP-012 layer-2 LOO probe 已完成；ORB-003 已完成 production-checkpoint-matched CUDA cost probe，但在线增量远超预算，分支封存为 `OFFLINE/TEACHER_ONLY`
> 关联方案：`PLAN_outer_lambda_neural_basis.md`、`IMPLEMENTATION_PLAN_outer_lambda_neural_basis.md`、`EXPERIMENT_LOG_outer_lambda_neural_basis.md`

## 0. 这份 TODO 要回答的唯一问题

验证 ORB-v3 是否能提供一种比当前 MACE teacher 更便宜、但仍然足以降低相邻 λ 态 MM gap variance 的局部几何表示。

本分支的首要假设是：

```text
coordinates
    -> frozen ORB-v3 shallow node representation
    -> ligand-only pooling / tiny scalar readout
    -> B_ORB(R)
    -> optional -∇R B_ORB(R)
```

第一阶段只做离线 representation probe，不接 OpenMM、不接 production、不训练一个新的大型 student。只有离线信号和真实成本都过门，才允许继续做可导标量基势和在线部署资格。

### 0.1 重新核实后的总判断（2026-08-09）

重新核实后的结论是：**离线 ORB latent probe 值得做，方法学成立；但当前 TODO 对模型选择、局部图规模、统计晋级门和在线成本偏乐观，完整 ORB 在线路线目前应视为低概率候选。**

当前路线结论冻结为：

- `ORB offline representation probe`：`GO`。
- `ORB-L2 shallow scalar basis`：`CONDITIONAL GO`；layer 2 是 primary，其他层只作 exploratory。
- `完整 ORB 每步在线`：当前 `NO-GO` 先验；除非真实 matched-path 测量推翻成本先验，否则不进入在线路线。
- `直接用 ORB 替换 MACE teacher`：尚不能宣布；训练域、OMol/OMat 分离和完整在线成本仍未通过。

30–50% improvement 只能作为“达到 MACE 级信号”的强度标签，不能作为正式晋级硬门。正式统计门复用 EXP-012 的冻结协议：三折均改善为完整门，至少 2/3 folds 改善为 hard floor，平均 improvement 必须大于 0，且最差 fold 的恶化不得超过 10%。

### 0.2 已核实的模型与成本事实

| 事项 | 重新核实后的事实 | 对本 TODO 的约束 |
|---|---|---|
| 主模型 | `orb-v3-conservative-omol` 更接近当前 `MACE-OMOL-0` 对照；`orb-v3-conservative-inf-omat` 主要由小型周期晶体训练 | OMol 为预注册 primary；OMat 只能作为独立 OOD 对照，不混合择优 |
| 网络深度与维度 | ORB-v3 有 5 个 message-passing blocks；node latent 默认 256 维；1024 是内部 MLP hidden width | latent cache、pooling 和 ridge 输入登记为 256 维，不能按 1024 设计 |
| `inf` 含义 | 官方实现的 `inf` 仍使用 `max_num_neighbors=120`；只是对训练分布近似 unlimited | 必须审计真实帧的 6 Å 邻居数和 120-cap 命中率，不能把 `inf` 写成无 cap |
| 局部图规模 | 现有 MACE 5 Å 两跳图已有约 940–1066 nodes；ORB 使用 6 Å，L2 不应预设为 200–300 atoms，L5 可能迅速扩张 | 先做逐层 L-hop 闭包、node/edge、最大邻居和 cap 命中率审计，再做成本预测 |
| 速度口径 | 官方 step/s 是 1000-atom、H200 且不含 graph construction 的指标；direct force head 的速度也不能迁移给 scalar readout | 只能说明相对趋势；正式成本必须测 graph construction、forward、scalar backward 和 bridge |
| 对称性 | ORB 不是结构上严格 roto-equivariant；conservative 能量的 equigrad 训练不自动赋予自定义 shallow readout 对称性 | rotation、translation、净力、净扭矩和 cutoff smoothness 必须单独审计；必要时注册 rotation augmentation/equigrad readout sensitivity |

### 0.3 核实来源

- [ORB-v3 官方论文](https://arxiv.org/abs/2504.06231)：架构、模型命名、速度和连续性讨论。
- [官方 `pretrained.py`](https://raw.githubusercontent.com/orbital-materials/orb-models/main/orb_models/forcefield/pretrained.py)：模型默认 adapter、5 层配置和 120-neighbor cap。
- [官方 `gns.py`](https://raw.githubusercontent.com/orbital-materials/orb-models/main/orb_models/common/models/gns.py)：message-passing stack 与 256 维 node latent。
- [官方 `nn_util.py`](https://raw.githubusercontent.com/orbital-materials/orb-models/main/orb_models/common/models/nn_util.py)：`ChargeSpinConditioner` 从 `system_features` 取 charge/spin，并广播到 node/edge conditioning。
- [官方 `MODELS.md`](https://raw.githubusercontent.com/orbital-materials/orb-models/main/MODELS.md) 与 [ORB 官方仓库](https://github.com/orbital-materials/orb-models)：训练域、OMol/OMat 区别及官方部署路径。
- EXP-012 冻结统计协议：[protocols/EXP-012_preregistration.json](protocols/EXP-012_preregistration.json)；实际 ridge/LOO 实现：[scripts/fit_exp012_local_residual_linear_readout.py](scripts/fit_exp012_local_residual_linear_readout.py)。

### 0.4 当前实际预检（2026-08-09）

以下结果是 ORB-000 已完成的输入/conditioning 预检，以及 ORB-001 允许前的 representation sensitivity 证据：

- `orb-v3-conservative-omol` 已从本地缓存加载；在官方 charge/spin conditioner 生效的显式 layer-2 prefix 下，frame0 输出为 `(41, 256)`，全部 finite。父体系 contract 冻结为 `Q=0, M=1`；这不是截断 L-hop 子图的局部电子自旋。
- 现有 OpenMM `NonbondedForce` 审计显示全体系电荷约 `2.0e-8 e`、41 原子 fragment 电荷约 `2.0e-8 e`，在 `1e-6 e` 容差内与父体系中性 contract 一致。没有要求 XML 保存 multiplicity；闭壳层 singlet 是显式登记的父体系建模假设。详见 [parent conditioning contract audit](output/outer_lambda_orb/orb_parent_conditioning_contract_audit_v1.json)。
- run1/run2/run3 的 frame `0/250/499` 均完成 CPU float64 的 6 Å L-hop 审计；9 个样本均无 120-cap 命中。L2 node 数为 `1508–1564`、edge 数为 `108992–114580`；L5 node 数为 `12818–13277`、edge 数为 `1019844–1064154`。这已否定 L2 约 `200–300` atoms 的成本先验，但尚不是完整 1500-frame 分布。
- `M=1` primary 对 `M=3` sensitivity 已在 9 个样本上完成：pooled latent 相对 L2 差异均值约 `0.0995`，ligand-node cosine 均值约 `0.9952`，per-dimension std 相对 L2 变化约 `0.0240`。M=3 只作 sensitivity，不参与 multiplicity 选择；`M=0` 未测试。
- 修正后的报告：[layer-2 primary smoke](output/outer_lambda_orb/orb_latent_smoke_run1_frame0_parent_contract_v2.json)、[spin sensitivity report](output/outer_lambda_orb/orb_spin_conditioning_sensitivity_layer2_9frames_v2.json)。此前 v1 sensitivity 因显式 prefix 漏传 conditioner 而 invalidated，不作科学证据。
- actual-edge equivalence gate 已在 run1/run2/run3 的 frame `0/250/499` 通过：9/9 帧的 canonical `(topology_sender, topology_receiver, unit_shift)` edge set、edge count、每节点 neighbor-count hash 和 120-cap 状态均一致。详见 [ORB-001a report](output/outer_lambda_orb/orb001a_edge_equivalence_9frames.json)。
- ORB-001b 初始化/性能 gate 已通过：loader wall time 约 `16.39 s`（其中 `import_pretrained` 约 `15.74 s`、official loader call 约 `0.33 s`）；1 帧 cold extract 约 `1.82 s`，10 帧 warm 平均 `1.17 s/frame`；scalar coordinate backward end-to-end 约 `2.65 s`，梯度 finite。详见 [ORB-001b benchmark](output/outer_lambda_orb/orb001b_initialization_benchmark_1cold_10warm.json)。
- 1500-frame layer-2 primary cache 已完成：三条 run 各 `500` 帧、latent shape `(500,41,256)`、float32；完整 cache 的 node 范围为 `1460–1643`，edge 范围为 `104036–123010`，最大邻居数为 `119`，cap-hit 为 `0`。三份 NPZ SHA-256 均与 report 一致，且 cache/ledger 已完成 fail-closed join。
- EXP-012-compatible layer-2 primary LOO probe 已通过：3/3 folds 改善，平均 relative improvement `0.3968221`，最差 fold improvement `0.2805260`，baseline/fitted gap-variance 分别为 `0.448644→0.322787`、`0.269171→0.154311`、`0.393317→0.203257`。详见 [layer-2 probe report](output/outer_lambda_orb/orb_layer2_exp012_probe_report.json) 与 [join report](output/outer_lambda_orb/orb_layer2_exp012_join_report.json)。
- 因此 `ORB-001 PRIMARY_STATISTICAL_GATE_PASSED`；layer 2 的离线 representation signal 已进入成本评估，但不授予 scalar basis 或在线 ORB 资格。L5 仍为 `EXPLORATORY / NOT_PRIMARY`，不生成完整 cache。
- 随后在 GPU 节点完成了 production-checkpoint-matched CUDA 复核：L2 closure 为 `1569` nodes/`113804` edges、最大 outgoing neighbor `107`、无 120-cap 命中；official graph median `30.835 ms`，layer-2 forward median `36.640 ms`，measurement-only scalar coordinate backward median `80.490 ms`。TorchScript wrapper 与 offline adapter 的 scalar absolute difference 为 `7.59e-7`。详见 [ORB-003 CUDA matched cost probe](output/outer_lambda_orb/orb003_cost_probe_cuda_node.json)，report SHA-256 为 `10ac708502f5a3fdf160db7d1e8c55a9494052e3053989f5f5bfce7abea335be`。
- CUDA matched production step 的 baseline 为 `1.273 ms/step`，加入 temporary scalar 后为 `78.896 ms/step`，增量中位数 `77.622 ms/step`；这远超冻结的 `0.1–0.2 ms/step` 增量预算。CUDA checkpoint restore、OpenMM platform 和 TorchScript bridge 均已完成，但成本门失败。因此最终决策为 `ORB-003 COST_GATE_FAILED / OFFLINE_TEACHER_ONLY`，不进入 ORB-004/005、在线 TorchForce、MTS 或 OpenMM production wiring。

### 0.5 ORB-001 implementation gates（2026-08-09）

- [x] `ORB-001a`：官方 adapter 显式冻结为 `knn_alchemi`、`graph_construction_dtype=float64`、`output_dtype=float32`、`wrap=True`、`half_supercell=True`；逐帧比较 closure/ORB 实际输入的 canonical topology edge set、count、neighbor-count hash 和 cap 状态。
- [x] `ORB-001b`：分阶段记录 `resolve_model_path`、`import_pretrained`、official loader、freeze/eval、cold frame、warm frames、scalar backward 和 RSS；没有发现逐帧重复加载模型。
- [x] provenance 已登记 `orb_models=0.6.2`、PyTorch `2.12.0`、CPU、`float32-high`、`compile=False`、checkpoint path/size/SHA-256、edge backend、PBC wrapping 和 half-supercell。
- [x] 正式 1500-frame cache：仅 layer 2；每帧继续 edge-set/cap fail-fast；三条 run 各 500 帧完成并完成 EXP-012 join/LOO ridge。
- [x] EXP-012 layer-2 primary 统计门：3/3 folds 改善，平均 improvement `39.682%`，最差 fold `28.053%`，正式门通过；30–50% 仍仅作为强度标签，不是硬阈值。

## 1. 已冻结的项目边界

### 1.1 继承现有基线，不混入其它变量

- [ ] 保持当前方法基线：`mode=ibs`、`decoupling=dual_lambda`、`potential=softcore`、`dexp_params=null`。
- [ ] 保持困难窗口：complex vanishing Stage 2 window 0，states `[0,5)`。
- [ ] 保持已有三条独立 run 及其 1500 帧数据划分；不随机重排 frame。
- [ ] 保持现有 MM ledger、`adjacent_gap_reduced`、`log_importance_unnormalized`、`A_k` 和 `fold` 定义。
- [ ] 保持 leave-one-run-out（LOO）协议：两条 run 拟合，第三条完整 run held-out 评估。
- [ ] 保持 MACE probe 的 ridge readout 协议、中心化方式、正则化搜索范围和评价脚本；如需改变，必须另立 sensitivity 实验，不能替换主结果。
- [ ] 将当前对照明确登记为 `MACE-OMOL-0`；ORB 的 OMol/OMat arm 必须分开记录，不能把不同训练域的结果放进同一主选择流程。
- [x] 预注册 `orb-v3-conservative-omol` layer 2 为 primary；layer 1/3/5 只作 exploratory。不得看完 held-out 结果后再从四层中择优；如确需择层，必须在每个 outer fold 内做 nested layer selection。
- [x] 冻结父体系 conditioning contract：`orb-parent-system-charge-spin-v1`、scope `parent_full_system`、`Q=0`、`M=1`；这是 closed-shell singlet conditioning inherited from the parent system，不是截断 L-hop fragment 的电子 multiplicity。
- [x] ORB probe 晋级门完全复用 EXP-012：三折均改善为完整门，至少 2/3 folds 改善为 hard floor，平均 improvement > 0，最差 fold 恶化不超过 10%。
- [ ] 不同时改变 λ schedule、IBS 权重算法、MM 基础势、窗口定义或采样轨迹。
- [ ] ORB 分支继续放在独立研发命名空间，不修改 `runabfe.py`、`abfe_core.py`、`abfe_pipeline.py`、`ibs_engine.py`。

### 1.2 不允许的捷径

- [ ] 不把 `orb-v3-direct-20` 作为第一候选；20-neighbor cap 可能引入 PES discontinuity，与标量 Hamiltonian correction 的平滑性要求冲突。
- [ ] 不优先调用 ORB 的 direct force head；本分支真正需要的是 ORB node representation 加自定义 scalar readout，再通过 autograd 得到守恒力。
- [ ] 不把 direct backbone 的官方 direct-force step/s 直接迁移到 scalar readout；自定义标量必须保留对坐标的 backward，速度优势需重新测量。
- [ ] 不把官方 1000-atom/H200、且不含 graph construction 的 benchmark 换算成当前 `1.38 ms/step` 在线预算。
- [ ] 不把离线 NumPy latent 当作新坐标上的 production 常数。
- [ ] 不把 ORB latent 解释为电子密度、Pauli/exchange energy 或真实电子结构量。
- [ ] 不把 ORB 的 foundation-model 总能量直接作为 fragment energy、residual energy 或 IBS bias。
- [ ] 在 ORB-0 完成前不写 OpenMM bridge；在 ORB-1 成本门未通过前不做长程在线 MD。
- [ ] 不因为 ORB 比 MACE 快就自动宣布 production candidate；必须同时通过 representation、对称性/力学和成本门。

## 2. 分支实验编号与输出约定

建议使用独立编号，结果写入现有实验日志的新增 ORB 小节；不要覆盖 EXP-010～EXP-013 的历史条目。

| 编号 | 目的 | 主要输出 | 晋级条件 |
|---|---|---|---|
| `ORB-000` | 分支冻结、环境和数据审计 | manifest、版本、hash、数据审计报告 | 可复现且输入与 MACE probe 同构 |
| `ORB-001` | 预注册 layer 2 的 ORB 离线 latent probe；L1/L3/L5 仅作 exploratory | latent cache、LOO ridge report | layer 2 通过 EXP-012 统计门；其他层不得用于事后择优 |
| `ORB-002` | 失败分析和 sensitivity | null/permute、layer、cutoff、dtype 对照 | 只有在 ORB-001 信号可解释时继续 |
| `ORB-003` | 统计门通过后的真实局部图成本 probe | L-hop 图规模、cap 审计、matched-path timing、VRAM、图构造报告 | 只有真实 scalar backward 成本落入在线预算才允许继续 |
| `ORB-004` | frozen ORB + tiny scalar readout | checkpoint、训练报告、gap report | gap、能量幅度、泛化和正则门通过 |
| `ORB-005` | 标量能量/力/对称性资格 | invariance、FD、force sanity report | 所有 D2 硬门通过 |
| `ORB-006` | TorchScript/OpenMM 最小部署 | CPU/CUDA/reference smoke、成本报告 | 仅当 ORB-003 和 ORB-005 都通过 |
| `ORB-007` | 短程动力学及可选低频调度 | NVT/MTS 资格报告 | 仅当 ORB-006 的实时预算可行 |
| `ORB-008` | 与 MACE、baseline 的最终比较 | comparison table、go/no-go 记录 | 形成可审计结论，不自动进入 production |

建议输出目录：

```text
output/outer_lambda_orb/
├── orb_manifest.json
├── orb_environment_report.json
├── orb_dataset_audit.json
├── orb_latent_cache_layer1.npz
├── orb_latent_cache_layer2.npz
├── orb_latent_cache_layer3.npz
├── orb_latent_cache_layer5.npz
├── orb_probe_report.json
├── orb_cost_probe_report.json
├── orb_scalar_basis_report.json
├── orb_symmetry_force_report.json
└── orb_decision.md
```

## 3. ORB-000：先冻结输入、环境和可追溯性

### 3.1 建立实验 manifest

- [ ] 新建 `output/outer_lambda_orb/orb_manifest.json`。
- [ ] 登记实验编号、创建时间、git commit、工作区 dirty 状态和相关文件 hash。
- [ ] 登记 ORB 模型完整名称、checkpoint 来源、模型文件 hash、模型版本和 license/使用限制。
- [ ] 登记设备、CUDA、PyTorch、Python、ORB package、OpenMM、TorchScript/runtime 版本。
- [ ] 登记 dtype、device、batch size、是否使用 autocast、是否启用 compile；第一轮默认关闭会改变数值/图结构的优化。
- [ ] 登记 ORB cutoff、neighbor policy、PBC/cell 处理方式和模型默认输入规范。
- [ ] 登记每个 layer 的 node representation 名称、shape、维度和截取位置。
- [ ] 登记所有 CLI 参数和随机种子；冻结后不得只改命令行而不更新 manifest。

### 3.2 核对模型可用性

- [x] 将 `orb-v3-conservative-omol` 预注册为 primary，确认权重可加载、元素覆盖当前体系、支持当前 PBC/cell 输入，并冻结父体系 `Q=0, M=1` conditioning contract。
- [ ] 将 `orb-v3-conservative-inf-omat` 作为单独 OOD 对照；OMat 结果不得与 OMol arm 混合择优，也不能替代 OMol primary 结论。
- [ ] 将 direct ORB（包括 `orb-v3-direct-inf-omat`）限制为条件性 representation/symmetry 对照；不得把 direct-force head 的官方速度当作 scalar readout 的成本先验。
- [ ] 记录是否能访问 final per-atom node representation，而不是只拿到总能量或 direct force head 输出。
- [ ] 确认是否能取得 layer 1、2、3、5 的中间表示；官方默认 forward 只返回最终 `node_features`，离线可用 hook 做定位，但正式 adapter 应显式执行前 `L` 个 `gnn_stacks`，不能依赖整网 forward 后再猜中间层。
- [ ] 记录 ORB-v3 的结构事实：5 个 message-passing blocks、node latent 256 维、内部 MLP hidden width 1024；1024 不得误登记为 latent dimension。
- [ ] 对一帧固定输入重复运行至少 3 次，核对 latent、energy、node count、edge count 是否确定性一致。
- [ ] 明确 ORB 输入所需的元素、坐标单位、cell、周期边界、原子序号和可选电荷字段；形成输入转换测试。
- [ ] 记录模型是否依赖动态 neighbor ranking；官方 `inf` 仍是 `max_num_neighbors=120`，必须与 `20-neighbor` 结果分开，并单独记录 120-cap 命中率。

### 3.3 审计数据和原子映射

- [ ] 找到与 MACE probe 同一批次的三条 run、每条 run 的 500 帧和对应 MM ledger。
- [ ] 核对每一帧坐标、box、周期边界、原子数和原子顺序与 ledger 对齐。
- [ ] 冻结 ligand atom indices、protein/environment atom indices、元素标签和 ligand-only pooling mask。
- [ ] 核对 ORB 的 node output 是否包含全部体系原子；若 ORB 过滤/重排原子，建立显式 index map，不能靠位置猜测。
- [ ] 记录每帧 ORB 实际 node 数、edge 数、ligand node 数和失败帧数。
- [x] 对每条 run 的少量预检帧计算 6 Å 的精确 L1/L2/L3/L5 闭包规模、最大邻居数和 120-cap 命中率；禁止沿用“L2 约 200–300 atoms”的未经测量先验。
- [x] 在正式 cache 前完成实际 ORB edge-set equivalence：closure 图与 official adapter 图分开记录 device/dtype/backend，并逐帧冻结 `edge_set_sha256` 与 cap 状态。
- [ ] 将现有 MACE 5 Å 两跳图约 940–1066 nodes 作为规模参照；ORB 6 Å 的逐层 node/edge 分布必须实测，不能用官方 1000-atom benchmark 代替。
- [ ] 对缺失 cell、坏坐标、NaN、异常 box、元素不支持和 node-map 不一致建立 fail-fast 检查。
- [ ] 验证逐帧 ORB 图构造不会改变 MM ledger 的 frame 顺序；任何丢帧必须同时从 latent cache 和 ledger manifest 中显式登记。

## 4. ORB-001：离线 representation probe（第一枪）

### 4.1 四个层级的预注册角色

- [ ] **Primary：**提取并冻结 ORB layer 2 的 ligand node latent；ORB-001 的主 endpoint 只看 layer 2。
- [ ] **Exploratory：**layer 1/3 可作为解释性对照；L5/final 当前标记为 `EXPLORATORY / NOT_PRIMARY`，不生成完整 1500-frame cache，不得根据 held-out 结果事后替换 layer 2 primary。
- [ ] 不在第一轮加入更多 layer、更多 cutoff 或多个 foundation model，避免把试验变成无边界搜索。
- [ ] 每个层级单独保存 latent cache，附带 frame/run/lambda/atom-map metadata，并明确 `latent_dim=256`；不得把内部 MLP hidden width 1024 写入 cache shape。
- [ ] 缓存只保存 representation 和必要索引，不把 ORB 总能量误写成训练 target。
- [ ] 缓存生成后计算 sha256，并在 probe report 中记录 cache hash。
- [ ] 若未来要让 layer 成为可选择的超参数，必须在每个 outer fold 的训练 run 内做 nested layer selection；不得使用 outer held-out run 选择 layer。

### 4.2 Pooling 与 readout 先保持最小化

- [ ] 先使用与 MACE probe 同构的 ligand-only pooling；不把蛋白/水/离子节点直接平均进 readout。
- [ ] 固定当前 41 个 ligand 原子均值池化作为 primary pooling；sum、随机 pooling 或其它 pooling 只能作为显式 sensitivity。
- [ ] 记录 pooling 前后的 shape：`[frame, ligand_atom, latent_dim]` → `[frame, pooled_dim]`。
- [ ] 先测试简单 pooling：mean、sum 或已有 MACE probe 使用的同类 pooling；不同 pooling 作为显式 sensitivity，不得偷偷替换主协议。
- [ ] 对 latent 做训练集内拟合的中心化/标准化；禁止使用 held-out run 的统计量泄漏到训练阶段。
- [ ] 不在 ORB-001 学习复杂 MLP；主结果使用 frozen 256-d latent + 训练集标准化 + 无截距 linear/ridge readout，确保与 MACE 44.6% 结果可比。

### 4.3 复现 MACE 的 LOO ridge/gap protocol

- [ ] 对每个 layer 使用完全相同的三折 LOO：run-1 out、run-2 out、run-3 out。
- [ ] 训练 target 使用已有相邻态 MM reduced-energy gap ledger；不新增 PMF target，不改成总能量拟合。
- [ ] 固定全局 `A_k` 和既有 `sin²` outer schedule；不在 probe 内重新拟合 `A_k`。
- [ ] 在两条训练 run 上拟合 ridge readout，在第三条 run 上计算 held-out gap variance。
- [ ] 报告每折 `B=0` baseline、ORB improvement、训练/held-out RMSE、预测方差和读出系数范数。
- [ ] 完全复用 MACE/EXP-012 的训练集标准化、无截距 ridge、inner two-way CV、importance-weighted gap loss、ridge 网格和选择规则；如果要扩大网格，单独标记 `sensitivity-only`。
- [ ] 记录每个 layer 的训练耗时、latent 提取耗时、峰值显存和有效帧数。
- [ ] 生成统一表格：`ORB-L2 primary`、`ORB-L1/L3/L5 exploratory`、MACE-OMOL-0 latent、intercept-only baseline。

### 4.4 ORB-001 的最小验收输出

- [ ] 三折结果全部有独立数值，不允许只报告均值。
- [ ] 明确 improvement 的定义与 MACE 44.6% 完全一致。
- [ ] 对 layer 2 标记正式 `signal / weak / failed / invalid`；对 L1/L3/L5 标记 exploratory 结果，不把 exploratory 最优结果写成晋级结论。
- [ ] 画出每折 held-out gap variance 和 improvement 的比较图。
- [ ] 画出 improvement 与 latent 提取成本/维度/实际 node-edge 规模的 Pareto 图。
- [ ] 记录是否存在单独某一条 run 主导结论的情况。
- [ ] 记录 layer 越深是否单调增益；不预设“layer 5 一定最好”。
- [ ] 对 primary layer 2 单独给出 EXP-012 晋级门判定：3/3 folds 改善为完整门；2/3 为 hard floor；平均 improvement > 0；最差 fold 恶化不超过 10%。

## 5. ORB-002：离线 probe 的稳健性和失败分析

只有 ORB-001 出现可重复 held-out 信号后才执行本节；如果 ORB-001 完全无信号，不用为解释失败而无限扩展消融。

### 5.1 必做的低成本对照

- [ ] 运行 frame/run label permutation 或 target permutation null，确认 ridge 增益不是数据排列 artifact。
- [ ] 比较 ligand-only pooling 与错误/随机 pooling，确认信号依赖正确 atom map。
- [ ] 统计每层 latent 的有效秩、近零方差维度、异常大值和跨 run 分布漂移。
- [ ] 检查是否只有少数 ligand atom 的 node latent 在驱动结果；报告 atom ablation 或 per-atom coefficient norm。
- [ ] 检查不同 run 的 node/edge 数分布，确认 held-out 失败不是图规模域外问题。
- [ ] 检查每条 run 的最大 6 Å 邻居数和 120-cap 命中率；若存在 cap 命中，单独报告潜在 neighbor-ranking discontinuity，不得把 `inf` 当作连续图协议。
- [ ] 对主候选层做一次 dtype sensitivity（例如模型默认 dtype 与统一 float32）；不得用低精度结果替换主结果。

### 5.2 需要时才做的对照

- [ ] 若 OMol primary 无法定义 local charge/spin，登记为 `EXPLORATORY_ONLY`；OMat 结果只能作为独立 OOD 对照。
- [ ] 若 OMat 结果强但成本或训练域不适用，不得据此替代 OMol primary；分别登记 representation、OOD 和 symmetry 风险。
- [ ] 若 `inf` 结果强但 120-cap 命中率高或成本过高，再评估更浅层/更小局部图，而不是立即切换 `direct-20`。
- [ ] 若 layer 1/2 已达到 MACE 级别，停止继续寻找更深层的微小提升，优先进入成本 probe。
- [ ] 若只有 final layer 有信号，记录“ORB shallow route 未成立”的结论，不把 full ORB 直接视为 production candidate。

### 5.3 ORB-0/1 的停止门

- [ ] 若所有 layer 在三个 held-out fold 上均无稳定正向信号：登记 `ORB_REPRESENTATION_FAILED`，停止本分支的 OpenMM 工作。
- [ ] 若只在训练集或单一 fold 有增益、但 held-out 无重复支持：登记 `NO_GENERALIZATION`，停止扩大模型规模。
- [x] primary layer 2 已通过 EXP-012 统计门，标记为 `REPRESENTATION_PROMISING` 并允许进入 ORB-003；L1/L3/L5 未参与 primary 选择，30–50% 仍只是“达到 MACE 级信号”的 exploratory 强度标签。
- [ ] 若结果落在中间区域，先完成预注册的 null、fold 和成本报告，再决定是否继续；不得只凭均值主观晋级。

## 6. ORB-003：真实局部环境成本 probe

目标不是引用 ORB 官方 1000-atom benchmark，而是在当前 Atenolol 困难窗口的真实局部环境和真实 OpenMM/torch 调用路径下测量。

### 6.1 先冻结测量对象

- [x] 只有 layer 2 primary 通过 ORB-001 的正式统计门后，才启动成本 probe；L1/L3/L5 的 exploratory 结果未触发在线成本路线。
- [x] 已冻结并审计局部图范围：真实 ligand/environment atom 数、L2 node/edge、PBC、最大邻居数和 120-cap 状态。
- [x] 已记录测量对象为“全父体系坐标输入 + 精确 L2 local closure + ORB shallow prefix + ligand-node pooling”，没有把全图 ORB、局部裁剪和固定图成本混报。
- [x] 已记录设备差异：现有生产 baseline 为 CUDA 约 `1.396 ms/step`，本次节点无 CUDA，CPU 结果仅作 diagnostic。
- [x] 已固定 batch=1、真实坐标 shape、float64 graph construction、float32 model batch、真实 coordinate backward 路径。

### 6.2 四项成本必须拆开

- [x] **graph construction**：动态 L2 closure 与 official `knn_alchemi` adapter graph 已分开计时。
- [x] **ORB forward**：冻结 layer 2 prefix 到 selected node representation 的前向成本已测。
- [x] **scalar readout + backward**：measurement-only scalar 对全父体系坐标的 autograd backward 已测。
- [x] **bridge/synchronization diagnostic**：TorchScript + 真实 production System 上的 CPU TorchForce group evaluation 已测；checkpoint-matched CUDA bridge 未测且明确未通过资格化。

### 6.3 三种调用口径

- [x] forward-only：测 ORB encoder 与 layer tap，不包含 backward。
- [x] forward + scalar-readout backward：冻结 ORB，使用 measurement-only scalar，保留坐标 autograd。
- [x] matched-path deployment：真实 CUDA production checkpoint、CUDA OpenMM System 和 temporary TorchScript scalar bridge 均已完成；该路径因成本门失败，不具备在线资格。

### 6.4 成本门和决策

- [x] 已记录同一 CUDA production path 的 baseline `1.273 ms/step`、temporary scalar `78.896 ms/step` 和增量 `77.622 ms/step`；不使用官方 1000-atom/H200 step/s 替代成本门。
- [x] 已记录 CUDA matched bridge 的显存样本 `7042/8801/16303 MiB`，以及真实 production checkpoint、OpenMM CUDA platform 和 ORB CUDA compute device。
- [x] TorchScript 与 CUDA TorchForce group 均可调用，checkpoint-matched CUDA backend 已完成；但真实增量远超预算，不能把此项写成在线部署通过。
- [x] 当前登记 `OFFLINE/TEACHER_ONLY`（在现有环境与证据范围内），不进入 OpenMM 在线路线。
- [x] matched CUDA path 已完成，但成本增量 `77.622 ms/step` 不满足 `0.1–0.2 ms/step` 门，因此 ORB-004/005 停止。
- [x] 不通过增加 pooling、direct force head 或未测 compile hack 规避 ORB-003；任何未来 CUDA 复核都必须另立报告并保持 layer 2/模型冻结。

## 7. ORB-004：冻结 ORB + tiny scalar basis

本阶段只在 ORB-001 的 layer 2 primary 通过统计门、local charge/spin 合约已冻结且 ORB-003 的真实 scalar backward 成本有希望时进行。

### 7.1 标量基势定义

实现最小形式：

\[
B_{\mathrm{ORB}}(\mathbf R)
=B_{\max}\tanh\left[
\frac{W\,\mathrm{Pool}\{z_i^{\mathrm{ORB},L}(\mathbf R)\}+b}{B_{\max}}
\right].
\]

- [ ] ORB encoder 完全冻结；第一轮只训练 `W,b` 或一个同等规模的 tiny readout。
- [ ] 不给 ORB encoder 输入连续 λ；λ 只由现有外层 schedule 和 `A_k` 控制。
- [ ] 使用 ligand-only pooling；contact gate 如加入，必须固定定义并进入 manifest。
- [ ] 保持端点约束 `A_0=A_K=0`，不改变物理端点 Hamiltonian。
- [ ] 神经项进入 target energy/IBS 判别式的正式账本，不进入需要消除的 `bias_history`。
- [ ] 每个坐标只计算一次 ORB basis；同一窗口内全部 λ 状态共享该 basis。
- [ ] 不做 fragment total-energy subtraction，不对 ligand/environment 分别调用 ORB 再相减。

### 7.2 训练目标和对照

- [ ] 复用现有双向相邻态 gap-variance loss；第一轮固定 `A_k`，不同时学习 `A_k`。
- [ ] 保留 `lambda_E` 和 `lambda_F` 等正则项，但先冻结超参数；调整必须是预注册 sensitivity。
- [ ] 至少保留 `B=0` baseline、linear/ridge ORB probe、tiny readout ORB basis 三个对照。
- [ ] 如需要判断“ORB backbone 是否比任意小模型更有用”，加入同参数量 typed radial/contact baseline；不把 MACE teacher loss 默认加入主结果。
- [ ] 如果引入 MACE-distilled readout，必须单独标记为 secondary arm，不能替代 direct-gap 主路线。
- [ ] 训练/验证按完整 run 切分；所有中心化常数、图协议和超参数在 held-out 前冻结。
- [ ] 如需补偿自定义 readout 的旋转风险，可预注册 rotation augmentation/equigrad readout loss 作为 sensitivity；不得在看到对称性失败后临时加入并替换主结果。

### 7.3 训练验收

- [ ] held-out gap variance 相对 `B=0` 的改善在至少两个 fold 可重复。
- [ ] 不允许只在训练 run 上改善；报告 train/held-out gap、能量幅度和读出系数范数。
- [ ] `B_ORB` 的数值范围、饱和比例、每帧梯度范数和 force tail 必须有限且可解释。
- [ ] 检查 readout 是否把单一异常 atom、单一 frame 或 node-count artifact 当成 basis。
- [ ] 保存冻结 checkpoint、config、训练数据 hash、模型 hash 和推理 adapter 版本。

## 8. ORB-005：D2 标量能量、力和对称性资格

ORB-v3 并非像 MACE 那样结构上严格 roto-equivariant；即使 conservative 版本通过 equigrad 训练增强原模型能量的旋转性质，自定义浅层 node readout 也不会自动继承该性质，必须在当前模型和当前局部图上实测。

### 8.1 坐标导数正确性

- [ ] 用 autograd 计算 `F_B=-∇_R B_ORB`，确认能量和力使用同一坐标计算图。
- [ ] 在多组真实帧和人工扰动帧上做 coordinate finite-difference 对照。
- [ ] 分别检查 ligand、environment、边界原子的力；不能只检查 ligand force。
- [ ] 检查 cutoff 附近的能量/力连续性，特别是动态 neighbor 加入/移除时。
- [ ] 检查 PBC、triclinic box、最小镜像和 box 参数变化对能量/力的影响。
- [ ] 检查 NaN、Inf、极端梯度、force tail、readout tanh 饱和和异常构象。
- [ ] 在进入 OpenMM 前完成 primary layer 2 scalar readout 的 rotation/translation/净力/净扭矩审计；不得把 conservative backbone 的能量资格替代 readout 资格。

### 8.2 全局变换不变量

- [ ] 对同一帧施加多个随机 global rotation，比较 `B_ORB(R)`；同时比较 rotated force 与旋转后原 force 的关系。
- [ ] 对同一帧施加多个 global translation，比较能量、力和数值误差。
- [ ] 检查 `Σ_i F_i ≈ 0`，并记录绝对/相对误差门。
- [ ] 检查 `Σ_i r_i × F_i ≈ 0`；PBC 下明确使用的坐标原点和 unwrap 约定。
- [ ] 若上述任一门失败，先判定 scalar basis/图协议不合格，不进入 OpenMM。
- [ ] 将 conservative-inf 与 direct-inf 的 invariance 结果分开报告；不能用一个版本的结果替另一个版本背书。
- [ ] 若旋转门失败，可评估预注册的 rotation augmentation/equigrad readout loss；该路线必须作为 sensitivity 单独报告，不能事后替换 primary readout。

### 8.3 与现有 D2 证据对齐

- [ ] 复用 EXP-012 D2 的有限差分、cutoff、力尾部、坐标/autograd 证据格式。
- [ ] 复用现有 cell-list/动态图等价性测试思路；任何优化后重新跑等价性测试。
- [ ] 记录 ORB 的 node/edge 动态变化是否导致 TorchScript 图行为与 eager 行为不同。

## 9. ORB-006：最小可导部署（条件性）

只有 ORB-003 和 ORB-005 均通过时才启动。

### 9.1 TorchScript/adapter

- [ ] 将“坐标 → ORB shallow node representation → tiny scalar readout → scalar energy”封装成可导模块。
- [ ] 禁止在 production path 使用离线缓存 latent、Python callback 或 detached NumPy array。
- [ ] 明确 ORB 动态 neighbor 图在 TorchScript 中的表达；若只能依赖不支持的 Python/compile 路径，登记部署阻塞。
- [ ] 记录官方当前主要部署路径是 `torch.compile`、ASE、TorchSim 和 NVALCHEMI；当前没有现成的 ORB shallow-latent TorchForce 路线，OpenMM/TorchScript 只能作为本项目最后的条件性适配工作。
- [ ] 正式 adapter 必须显式执行前 `L` 个 `gnn_stacks`；离线 hook 仅用于定位和验证，不作为 production 实现。
- [ ] 生成 CPU eager、CPU scripted、CUDA eager、CUDA scripted 四种输出对照。
- [ ] 对固定真实帧比较 energy/force，记录绝对误差、相对误差和最大异常帧。
- [ ] 保存 scripted artifact hash，并将 runtime/library 版本写入 report。

### 9.2 OpenMM 最小 smoke

- [ ] 先使用单帧/少步 Reference 平台，验证 TorchForce 能量和力有限。
- [ ] 再使用 CUDA 少步 smoke；出现 backend handle、动态 shape、OOM 或 force-group 错误时立即停止并登记。
- [ ] 验证神经 force group 与经典 IBS group 的能量账本和 group mask。
- [ ] 验证 checkpoint/resume、box/parameters、λ 状态和 ledger 语义不变。
- [ ] 保持研发模块隔离，不把 ORB 配置写入 production 默认路径。

## 10. ORB-007：短程动力学与可选低频调度

### 10.1 每步在线先于 MTS

- [ ] 用普通 `LangevinMiddleIntegrator` 做短 NVT：有限值、温度、能量漂移、结构合理性、student/ORB force sanity。
- [ ] 采用与 EXP-012 D4 相同的绝对健康门，避免只做相对比较而漏掉共同崩溃。
- [ ] 以 matched-path 方式测真实 `ms/step`；不能只报告 ORB encoder 的 standalone forward。
- [ ] 若每步在线成本已明显超过预算，不启动 MTS 试图掩盖模型/部署成本问题。

### 10.2 只有每步路线有明确成本依据才考虑 MTS

- [ ] 若需要 MTS，先定义 slow object 是 scalar basis、exact residual 还是整个 fused group；不能把非线性 IBS log-sum-exp 随意拆成线性 force group。
- [ ] 若使用 exact residual split，先做 `V_0+ΔV_ORB ≡ V_*` 的真实 energy/force 等价性检查。
- [ ] MTS 初始化禁止把 `LangevinMiddleIntegrator` 的 binary checkpoint 直接灌入 `MTSLangevinIntegrator`；使用公开 State API 转移 positions、velocities、box 和 global parameters。
- [ ] 为每个 N 独立设置绝对温度/能量健康门，再做 N=1 相对比较。
- [ ] 比较 `N=1,8,16,32` 的能量分布、force tail、温度、构象分布、IBS 判别式相关量和积分误差代理。
- [ ] 如果出现随 N 的系统性温度/能量/构象偏移，按预注册门停止，不用“偏移量看起来不大”替代判据。

## 11. ORB-008：最终比较和 go/no-go

### 11.1 必须放在同一张表中的对象

| 对象 | 表示 | held-out gap improvement | forward 成本 | forward+backward 成本 | peak VRAM | 是否可导 | 结论 |
|---|---|---:|---:|---:|---:|---|---|
| MACE-OMOL-0 latent | frozen MACE-OMOL-0 | 44.6% avg（已登记） | 已知很高 | 已知很高 | 已知很高 | 可作为 teacher | baseline teacher |
| ORB-L2 OMol | ORB layer 2 | 待测 | 待测 | 待测 | 待测 | 待测 | primary shallow candidate |
| ORB-L1/L3/L5 OMol | ORB exploratory layers | 待测 | 待测 | 待测 | 待测 | 待测 | exploratory；不得事后择优 |
| ORB-L2 OMat | OMat layer 2 | 待测 | 待测 | 待测 | 待测 | 待测 | 独立 OOD 对照 |
| typed radial baseline | 手工轻量基线 | 待测 | 待测 | 待测 | 待测 | 待测 | 控容量对照 |

### 11.2 结论分类必须明确

- [ ] `STOP_NO_SIGNAL`：ORB latent 无 held-out 信号。
- [ ] `TEACHER_ONLY`：ORB latent 有信号，但真实在线成本不可行；保留为离线 teacher/后处理表示。
- [ ] `PROMISING_OFFLINE_BASIS`：信号和标量力学通过，但还未证明 OpenMM 实时成本。
- [ ] `ONLINE_CANDIDATE`：representation、成本、D2、TorchScript、短 NVT 全部通过。
- [ ] `MTS_CANDIDATE`：仅在每步在线不可行、且低频调度的物理门全部通过时使用。
- [ ] `PRODUCTION_ELIGIBLE`：只有通过独立小窗口 IBS、独立重复、自由能闭合和完整 provenance 后才允许使用；本 TODO 本身不授予此状态。

### 11.3 不可用单一指标替代结论

- [ ] 不能仅凭 ORB 官方 benchmark 宣称当前项目可行。
- [ ] 不能仅凭 held-out gap improvement 宣称 force 或 OpenMM 可用。
- [ ] 不能仅凭 ms/step 下降宣称统计效率提升；需要 ESS、GPU-hour 和自由能不确定度。
- [ ] 不能仅凭单次 NVT 稳定宣称 IBS 生产有效。
- [ ] 不能把 ORB teacher 成功等同于当前 `LocalResidualStudent` 成功；这是一条新的 shallow encoder + tiny readout 路线。

### 11.4 当前 go/no-go 结论

- [x] `ORB offline representation probe`：`GO`。
- [ ] `ORB-L2 shallow scalar basis`：`CONDITIONAL GO`；必须先通过 layer 2 的 EXP-012 统计门和真实成本门。
- [ ] `完整 ORB 每步在线`：当前 `NO-GO` 先验；只有真实 matched-path 测量推翻成本先验后才可改判。
- [ ] `直接用 ORB 替换 MACE teacher`：`NOT ESTABLISHED`；训练域、OMol/OMat 分离和 local charge/spin 合约均需先闭合。

## 12. 需要新增或修改的工程文件

### 12.1 允许新增的独立文件

- [x] `local_residual/orb_latent.py`：ORB 模型加载、显式 prefix layer tap、node-map 和 representation adapter；尚未形成 1500-frame primary cache。
- [x] `local_residual/orb_graph.py`：6 Å 精确 L-hop 闭包、node/edge 和 120-cap 审计。
- [x] `local_residual/orb_probe.py`：EXP-012-compatible LOO ridge/gap probe 和统计晋级门；待 layer-2 cache join 后运行。
- [ ] `local_residual/orb_scalar_basis.py`：冻结 ORB + tiny scalar readout。
- [x] `scripts/run_orb_probe.py`：layer-2 primary 的 prepared-NPZ probe 入口；缓存构建/ledger join 仍待完成。
- [x] `scripts/audit_orb_graph.py`：单帧真实 6 Å L-hop/cap 审计入口。
- [x] `scripts/smoke_orb_latent_frame.py`：单帧显式 layer-2 ORB representation smoke；支持 frozen parent contract 与 exploratory 标记。
- [x] `scripts/audit_orb_charge_spin_contract.py`：现有 topology/OpenMM system 的 parent charge 证据审计，并登记显式 singlet assumption。
- [x] `scripts/compare_orb_spin_conditioning.py`：9 帧 `M=1` primary / `M=3` sensitivity；不用于 multiplicity 选择。
- [x] `scripts/build_orb_latent_cache.py`：layer-2 full-cache builder；逐帧 120-cap fail-fast，输出可接 EXP-012 join 的 latent cache。
- [x] `local_residual/orb_latent.py`：显式 prefix 复用官方 charge/spin conditioner；禁止 `M=0` null sentinel。
- [ ] `scripts/profile_orb_local_cost.py`：真实局部图 matched-path profiling。
- [ ] `scripts/check_orb_symmetry_force.py`：rotation/translation/FD/force sanity。
- [ ] `scripts/check_orb_torchscript_openmm.py`：条件性部署 smoke。
- [ ] `tests/test_orb_latent_adapter.py`：输入、map、layer shape、determinism。
- [ ] `tests/test_orb_scalar_basis.py`：scalar output、gradient、端点和有限差分。
- [ ] `tests/test_orb_symmetry_force.py`：全局变换和净力/净扭矩门。

### 12.2 明确不应直接修改的文件

- [ ] `runabfe.py`。
- [ ] `abfe_core.py`。
- [ ] `abfe_pipeline.py`。
- [ ] `ibs_engine.py`。
- [ ] production 默认配置和已有 production cache。

如果后续确实需要接入上述文件，必须先完成 ORB-008，并创建独立的接入计划和回滚方案。

## 13. 实验日志模板要求

每个 ORB 实验完成后，在 `EXPERIMENT_LOG_outer_lambda_neural_basis.md` 新增对应条目，至少填写：

- [ ] 实验编号、日期、唯一研究问题。
- [ ] 预注册假设与停止门。
- [ ] 实际执行命令；不能只写脚本名。
- [ ] 输入轨迹、ledger、model checkpoint、代码 commit 和所有 cache hash。
- [ ] ORB model/version、layer、cutoff、neighbor policy、PBC 和 dtype。
- [ ] 有效帧数、失败帧数、node/edge 分布。
- [ ] 每折训练/held-out gap variance、improvement、RMSE 和不确定性。
- [ ] 成本分解、peak VRAM、设备和重复次数。
- [ ] 对称性、finite difference、力学和部署结果。
- [ ] 观测事实与解释分开写，不能把解释写成事实。
- [ ] `PASSED`、`FAILED`、`STOPPED`、`INVALIDATED` 或 `CONDITIONAL` 结论。
- [ ] 下一步、阻塞项和是否允许进入下一 ORB 编号。

## 14. 推荐执行顺序（实际开工清单）

### 第一轮：当天可完成的离线判断

- [ ] 完成 ORB-000 环境/权重/输入 manifest。
- [x] 冻结 `orb-v3-conservative-omol` primary、父体系 `Q=0,M=1` conditioning contract 和 `layer 2` primary；将 OMat 登记为独立 OOD 对照。
- [ ] 先对每条 run 的少量帧审计 6 Å 的 L1/L2/L3/L5 闭包规模、最大邻居数和 120-cap 命中率。
- [ ] 验证 layer 1/2/3/5 node representation 可取得；正式 adapter 使用显式 shallow prefix forward，不能只依赖最终输出或离线 hook。
- [x] 用同一 1500 帧、三 run、同一 MM ledger 生成 layer-2 primary latent cache；cache latent dimension 固定登记为 256；L5 full cache 不追求。`ORB-001a/001b` 已通过，cache 与 EXP-012 LOO probe 已完成。
- [ ] 运行完全同构的三折 LOO ridge/gap probe，primary endpoint 只看 layer 2。
- [ ] 生成与 MACE-OMOL-0 44.6% baseline 同表的 primary/exploratory comparison。
- [ ] 若 layer 2 无 held-out 信号，登记失败并停止；若通过 EXP-012 统计门，才允许进入成本 probe；不得根据 L1/L3/L5 的 held-out 最优结果替换 primary。

### 第二轮：真实成本判断

- [ ] 对最浅有效层测 graph construction、forward、backward、bridge、VRAM。
- [ ] 对 layer 2 primary 的真实局部环境规模测 graph construction、forward、scalar backward、bridge、VRAM；同时报告 L-hop node/edge 和 cap 命中率。
- [ ] 用真实局部环境规模，不用 1000-atom/H200 官方数字代替。
- [ ] 根据成本门决定 `TEACHER_ONLY` 或进入 scalar basis。

### 第三轮：可导标量基势

- [ ] 实现 frozen ORB + tiny scalar readout。
- [ ] 通过 held-out gap、幅度、梯度和有限差分检查。
- [ ] 在进入 OpenMM 前完成 rotation、translation、净力、净扭矩和 cutoff smoothness 资格；必要时只按预注册 sensitivity 评估 rotation augmentation/equigrad readout loss。

### 第四轮：部署和动力学

- [ ] 完成 TorchScript/eager/Reference/CUDA 等价性。
- [ ] 完成普通 Langevin 短 NVT。
- [ ] 只有成本确实合理时，才研究 MTS/rRESPA；优先保守、低复杂度的在线路径。

### 第五轮：科学闭环

- [ ] 与 MACE teacher、typed baseline、无神经项 baseline 做同窗口比较。
- [ ] 评估 ESS、BAR overlap、free-energy uncertainty、GPU-hour 和结构健康。
- [ ] 若通过，再单独安排三重复和跨体系验证；不直接进入完整 production。

## 15. 最终预期

本分支最有价值的成功形态不是“ORB 完整替代 MACE”，而是证明：

```text
ORB OMol shallow layer 2
    = 在预注册 LOO 统计门下保留有用的 gap 信息
    + 在真实 6 Å 局部图上具有可审计的 node/edge/cap 规模
    + 在 matched-path scalar backward 成本可接受时
      才有资格构造成平滑 scalar Hamiltonian correction
```

即使 ORB 无法满足每步在线预算，只要 layer 2 primary 稳定通过冻结的 held-out 统计门，它仍可作为离线 teacher、后处理 reweighting 表示或后续小型 student 的设计依据。反之，如果 primary probe、charge/spin 合约或 cap/图规模审计失败，应在对应 ORB-000/001/003 门停止，不继续投入 OpenMM 桥接和 production 工程。
