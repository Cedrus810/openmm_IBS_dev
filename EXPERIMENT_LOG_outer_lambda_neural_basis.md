# 外层 λ 神经基势实验日志

> **文档角色：实验事实与决策记录。** 方法原则见
> [`PLAN_outer_lambda_neural_basis.md`](PLAN_outer_lambda_neural_basis.md)，实施任务见
> [`IMPLEMENTATION_PLAN_outer_lambda_neural_basis.md`](IMPLEMENTATION_PLAN_outer_lambda_neural_basis.md)。
> 本文件只记录已经实际执行的实验，不把计划或推测写成已完成结果。

## 1. 记录规则

实验使用唯一编号：

```text
EXP-000
EXP-001
EXP-002
...
```

状态只允许：

- `PLANNED`
- `RUNNING`
- `PASSED`
- `FAILED`
- `INCONCLUSIVE`
- `SUPERSEDED`

每次实验必须：

- 保存完整配置和实际命令；
- 保存代码、模型和输入哈希；
- 使用独立输出目录；
- 区分观测事实与解释；
- 给出继续、回滚或停止的明确决定。

禁止：

- 覆盖旧实验目录；
- 只记录成功实验；
- 把接口测试或 warmup 当成 production 证据；
- 隐藏 NaN、异常力、结构异常或性能倒退；
- 没有独立重复就声称“稳定提升”。

## 2. 总体状态

> **最新状态（2026-08-09，supersedes the older summary rows below）：** EXP-012 D0–D4、wiring smoke 和 WP-5A pilot 已关闭；DEC-050 已关闭当前 real-time TorchForce 路线。EXP-013 按冻结顺序 ③→①→② 全部未形成可晋级路线：方案③ 013-B 未通过，方案① Qualification gate 未通过且 N=16 未授权，方案② N=1 ESS 信号为负（DEC-059）。因此不运行方案② MTS、不重调 `c1`、不进入 013-C；随后按 DEC-059 启动独立 EXP-014 native-compression screen，结果未通过其离线筛选门（DEC-060），不进入 OpenMM 资格化或 production promotion。EXP-016 已完成离线 temporal audit：3 条连续轨迹、1500 帧、`Δt_save=1 ps`，trajectory/ledger/latent 对齐通过，但没有 physical alchemical state/replica history；energy-weighted surrogate 结果只作 exploratory，未晋级 learned slow information。当前没有 online/MTS route 获得 production promotion。
> **口径补充：** DEC-034/035 中的 3/3 fold 全部改善属于 teacher-side cached-latent readout；D1 `LocalResidualStudent` direct-gap 的正式结果是平均 held-out gap-variance 改善 `13.9348%`、fold-level `2/3`，且 `direct_gap_all_folds_improved=false`。两者不能混称。
> **DEC-048 口径补充：** 三次 `mixture_ess_proxy` 提升来自 paired-reseed exploratory 两臂 pilot；它们不是三组独立平衡的 production repeats。三次 ESS/GPU-hour 均下降，DEC-050 关闭的范围是每个 MD step 调用 TorchForce 的实时部署方式。
> **当前执行边界（2026-08-09）：** EXP-013 三种在线/MTS 方案与 EXP-014 compression screen 均不晋级；不重调 `c1`、不重选 checkpoint、不继续搜索 MTS 间隔、不直接重开 WP-5。相关分支到此冻结，后续不再安排新的在线/MTS promotion 实验。

| 项目 | 当前值 |
|---|---|
| 总体阶段 | WP-0 完成；EXP-009/010/011 均已失败并冻结；EXP-012 为 CV-free 通用局部残差路线，C1 合成图 MACE graph/latent 合约已通过；Arm A/B/D 已退役为 `not_pursued`（DEC-039，§11A.12），`protocols/EXP-012_preregistration.json` 已 reseal 为 `sealed`；`LocalResidualStudent` direct-gap 变体 D1（held-out gap variance 改善）与 D2（坐标/autograd 资格，27/27 通过）均已完成（DEC-040）；System 身份门已关闭（DEC-041，`CLOSED_STEP3_OPERATIONAL_SEMANTIC_PASS`）。D3 进行中（DEC-042，2026-08-06）：sub-item 2（TorchForce/OpenMM Reference 注入）与 sub-item 3（端点归零）已通过；sub-item 4（生产耗时）确认 all-pairs 近邻发现是 258% 开销的主因，已替换为 linked-cell list，开销降至 95%，仍超过 (d0-5) 冻结的 ≤50% 淘汰门，未关闭；sub-item 1（deployment 一致性）残差误差从 1.4e-5 降到 1.7e-7 但仍未通过严格同精度门，一个可能成因已修复未经验证。D3 未关闭，D4 未开始 |
| 当前基础势 | `softcore` 原型；生产模块尚未接入神经路径 |
| 当前目标窗口 | complex vanishing window 0，Stage 2 states `[0,5)` |
| 历史慢变量 | primary ligand torsion `[4591,4592,4593,4585]` 仅保留为 EXP-010/011 诊断证据；EXP-012 不预设单一 torsion CV |
| 当前训练目标 | 最小化完整 MM 相邻态双向 target-state gap variance；不预设慢 CV，frozen-MACE latent 为主要候选表示 |
| 当前路线定性 | L2 的离线 teacher/student 证据保留；当前 real-time TorchForce 已由 DEC-050 关闭，EXP-016 仅完成 surrogate-only temporal audit；下一步不得直接进入 online/MTS promotion |
| 当前生产候选 | 无；`original_6a`（6 Å，DEC-024）CPU C1 通过，CUDA float32 对照三次尝试均在同一算子 OOM（`BLOCKED_ON_VRAM`，非碎片化，见 DEC-026/11A.11），6 Å 的 gradient checkpointing 尚未实现；`derived_5a`（5 Å，DEC-027）frame0-only 固定 manifest 的 CPU/CUDA C1（DEC-028/029）已被 DEC-032 撤销固定图策略取代，不再作为 teacher 图构造方式；模型训练、Force 和 NVT 均未进行。DEC-031：跨 1500 帧闭包并集重建 manifest 使候选池膨胀到 4874 原子/4915 节点（比 `original_6a` 已 OOM 的 2135 节点图更大），该方向已被 DEC-032 撤销。DEC-032：teacher 是离线工具、不进 OpenMM，正式改为逐帧独立精确两跳闭包（无固定图、无整残基收口）+ CPU float64 决定 membership/CUDA float32 执行的 Option C 分离设计。DEC-033：Option C 已在真实数据上验证并完成 `derived_5a` 的 per-frame teacher latent cache 生成（三条 run，各 500 帧，`ligand_latent [500,41,1024]` float32，latent-only 未拼 ledger）。DEC-034/035：join+线性/ridge readout 工具已实现并在真实 cache 上执行，三个 leave-one-run-out fold 全部相对 `B=0` 基线改善（39.9%/29.1%/64.8%，均值 44.6%），held-out 资格已通过（全部 fold 而非仅至少一个）。DEC-036：该结果已正式冻结登记，不因 ridge 网格边界现象事后加宽网格替换。DEC-037：`LocalResidualStudent` 编码前的设计契约已冻结（PLAN 文档同名章节），"在线动态环境表示"是编码前必须最先单独解决的问题，其余各项（最小架构、teacher-target 协议、必需对照、计算/部署预算、D1-D4 分阶段实现）随之冻结为框架；本轮不写任何 student 代码。DEC-038：设计契约第 1 项"在线动态环境表示"已用 `local_residual.geometry.ligand_environment_cross_edges`/`quintic_c2_cutoff` 的真实两帧对照关闭。DEC-039：(d0-5) 计算/部署预算除 ms/step 生产基线外全部冻结（模型/图规模数字未变、CUDA funnel 一致性关闭、训练 epoch/seed/早停改为训练 run 内时间块切分设计）；Arm A/B/D 正式退役为 `not_pursued`；preregistration reseal 为 `sealed`；ms/step 基线的 `win_sys_xml_sha256_matches_manifest=false` 根因（`box_vectors.npy` 陈旧）已定位并修复诊断脚本，但修复后的重新测量尚待执行 |
| 神经路径协议版本 | 独立模块 v1 已实现；production 未接入且明确保持隔离 |
| 已完成实验数 | 历史计数未重算；新增 `EXP-016` 已完成，`EXP-012` 的 D0–D4/WP-5A 证据链已关闭 |
| 已通过实验数 | 历史计数未重算；`EXP-016` 为 `INCONCLUSIVE / SURROGATE_ONLY`，不计入 physical slow-information 通过数 |
| 当前是否允许 production | 否 |
| 最近更新日期 | 2026-08-09 |

## 3. 决策日志

| 日期 | 编号 | 决策 | 依据 | 影响 |
|---|---|---|---|---|
| 2026-07-29 | DEC-000 | 建立科学总纲、实施计划、实验日志三层文档 | 分离原则、任务和证据 | 后续实验统一按本文件记录 |
| 2026-07-30 | DEC-001 | WP-1/2 和 WP-3 部署骨架先保持在独立模块，不接入生产模块 | 独立测试覆盖端点、账本、TorchForce/OpenMM；`test_production_modules_do_not_import_standalone_neural_module` 明确保证隔离 | production 未导入不是缺陷；成熟后才合并 |
| 2026-07-30 | DEC-002 | 将三批真实 MACE 运行登记为 WP-4 前置可调用性验证 | 1/10/50 帧均完成，能量和力有限、重复一致、路径端点严格归零 | 可以继续修正局部坐标/选择协议和设计每步 OpenMM Force；不能据此宣称 WP-4 qualification 或 WP-5 完成 |
| 2026-07-30 | DEC-003 | 结束短 smoke 循环，将 100-step 修正版运行作为资格阈值标定；下一次直接执行冻结的 1000-step NVT qualification | 每步 CUDA Force、PBC 成像、path-only 力和纯积分性能均已有可用观测 | 旧 smoke 命令仅保留作连通性诊断；正式推进只看 qualification 硬门 |
| 2026-07-30 | DEC-004 | 不事后放宽 250 kJ/mol/nm 路径力硬门；将外层系数从 0.10 降到 0.09 后执行一次正式重试 | EXP-006 仅路径力失败：258.949 kJ/mol/nm，超阈值约 3.6%；其余五项通过 | EXP-006 保持 FAILED；EXP-007 只改变一个预注册参数 |
| 2026-07-30 | DEC-005 | 冻结系数 0.09 的 MACE 路径作为 WP-5 单基势候选 | EXP-007 六个资格硬门全部通过 | 停止 WP-4 幅度和力学阈值调整；后续评价性能与科学收益 |
| 2026-07-30 | DEC-006 | 完整 MACE 不默认作为每个生产积分步的最终 bias；在 EXP-007 通过基础上单独比较 MTS 与 cheap-CV 蒸馏 | EXP-005/006/007 的实时后端约 0.10 s/step，频率成本仍高；完整局部 MACE 还包含高频原子力，不能未经验证每几十步冻结旧力 | MTS 必须使用独立 force group 并以 \(N=1\) 为参考；若性能门不通过，MACE 转为教师模型 |
| 2026-07-30 | DEC-007 | 将 complex vanishing window 0 和 atenolol `C4-N2-C9-C10` 扭转角冻结为第一组困难窗口/慢变量 | window 0 的最终 Stage 2 最小 ESS ratio 最低、绝对 ESS 最低且统计低效最高；预平衡轨迹中该扭转角观察到 trans/gauche-minus 交换 | EXP-009 三臂必须记录该扭转角；后续 WP-5 首先在该窗口检验转换与 ESS |
| 2026-07-30 | DEC-008 | EXP-009 固定使用 OpenMM BAOAB-rRESPA、`coefficient=0.09`、内步长 0.5 fs、\(N=1/2/4\)，正式脚本每臂 50 ps | 同一坐标、温度和种子可隔离调度差异；200 个诊断样本/臂用于首轮分布门 | \(N=4\) 的最低性能门预注册为 1.0 ns/day；不得依据结果修改本次阈值 |
| 2026-07-30 | DEC-009 | 完整 MACE 的 PythonForce CUDA MTS 路线不晋级，启动 EXP-010 | EXP-007 的普通 LangevinMiddleIntegrator 可运行；EXP-009 的 \(N=1\) 已在 CUDA MTS force-group 内核触发 `CUDA_ERROR_INVALID_HANDLE`。环境中的 openmmml 1.6 和独立实时桥接均使用 PythonForce | 不再重跑同一后端、不调 coefficient；完整 MACE 保留为离线教师，生产候选改为周期 torsion cheap-CV |
| 2026-07-30 | DEC-010 | 撤销 DEC-007 中“首个慢变量已冻结”的部分，只保留困难 window 0 已冻结 | 旧 window 能量账本没有坐标；预平衡 torsion 转换不能证明它造成困难 window 的低 ESS | 新增只读困难窗口 scratch 采样和自动 CV 排序；production CV 必须用稳定原子身份并经困难窗口轨迹验证 |
| 2026-07-30 | DEC-011 | 不凭单条困难窗口轨迹冻结 `VAL251 chi1` 或 hydration；要求至少三条独立种子重复 | run1 中 `VAL251 chi1` 只有 1 次 core 转换、\(N_\mathrm{eff}=3.91\)；hydration 的 \(g=82.19\)、\(N_\mathrm{eff}=6.08\)，两者不确定性都很大 | 新增按 `stable_id` 对齐的重复判定；两条轨迹只能比较，三条且转换可重复才可能晋级 |
| 2026-07-30 | DEC-012 | 冻结 ligand torsion `[4591,4592,4593,4585]` 为 EXP-010 的 primary slow variable；不授予 production approval | 三个独立种子的 core 转换为 14/4/10，重复排名 1；相邻 ligand torsion 和 `VAL251 chi1` 分别为候选 2/3 | WP-0 完成，EXP-010 可开始教师标注；secondary/hydration 保留作二维模型和诊断敏感性分析 |
| 2026-07-30 | DEC-013 | EXP-010 教师改用固定 protein-only environment，移除旧选择中的全部固定水；不放宽支持域门 | 旧 255 环境含 39 个水原子、涉及 14 个水残基且有不完整水；scratch frame 的跨度达 7.63 nm。移除后保留 216 个蛋白原子，CPU 一帧跨度 2.342 nm、Rg 0.728 nm 并通过 | 教师选择成为新的带哈希协议；原 EXP-007 含水选择仍保留为历史证据，不冒充适用于 scratch 数据 |
| 2026-07-30 | DEC-014 | 教师标注前按冻结支持域排除 frame，最大允许排除率 5%；不得调大 2.5 nm / 0.85 nm 门 | 三条轨迹每 5 帧预检共 300 帧，290 帧通过；10 帧跨度 2.50–2.60 nm，Rg 均小于 0.747 nm | MACE 不评价外推帧；数据集完整记录排除身份和比例，超过 5% 整批不得拟合 |
| 2026-07-31 | DEC-015 | EXP-010 记为 `FAILED`，不选择最终 Fourier 模型，不执行 cheap-CV NVT 或 WP-5 | 六个预注册候选的 leave-one-run-out 能量 RMSE 均未优于 intercept-only `21.5109 kJ/mol`；最好的 1D order 2 为 `22.1737 kJ/mol` 且广义力 \(R^2=-13.5934\)，2D 候选明显病态 | 不事后增加 Fourier order、不以训练集或 conditional-bin 表现晋级；任何新 CV、目标量、正则或模型族必须另立实验 |
| 2026-07-31 | DEC-016 | 修正 EXP-010 失败归因：它否定当前 atom-cut protein MACE 教师与逐帧总 interaction-energy 教学协议，不否定 torsion bias 本身 | 216 个 protein atoms 由单帧 0.5 nm 逐原子选择得到；实际涉及 26 个残基且完整残基数为 0。primary torsion 对教师 Cartesian force 的平均投影解释比仅 `0.00393`，三条 run 还有显著能量偏移 | 登记 EXP-011：首选从完整 MM Hamiltonian 学习条件平均力/PMF；若以后再用 MACE，必须另立实验验证完整残基/封端和环境半径收敛 |
| 2026-08-02 | DEC-017 | 冻结 EXP-011 目标加权周期 PMF 协议；历史三 run 仅用于覆盖诊断，不用空 bin 产模 | 7/24 pooled bins 为空；三 run 周期有效样本数为 141.41/6.62/3.76；最低 pairwise Bhattacharyya overlap 为 0.328 | 状态为 `PREREGISTERED / SAMPLING_REQUIRED`；下一步受限/增强采样，WP-5 保持未开始 |
| 2026-08-03 | DEC-018 | 冻结 EXP-011 失败并登记 EXP-012：完整 MM 基线 + XED-inspired 局部场 + ligand-only 短程路径残差 | AUG-001 后 mutual overlap 0.02353 < 0.03 且 22 个去相关样本 < 25；EXP-010 fragment-total MACE 边界不闭合 | 不再补采/拟合 EXP-011；EXP-012 先做 atom-centered / +XED / +overlap 三层消融，未证明增量前不训练完整 MACE、不进入 WP-5 |
| 2026-08-03 | DEC-019 | 将 EXP-012 从 XED 专项升级为 CV-free 通用局部残差路径势；frozen-MACE latent 作为主要候选，XED 降为可选 Arm D | MACE 可提供中间层 node descriptors，但在线守恒力必须保留 encoder 对坐标的 autograd；单 CV PMF 已在 EXP-011 失败，gap variance 可直接针对 overlap 瓶颈 | 保留 `exp012_xed` 草案哈希与 schema 证据，sealed 前迁移为 local-residual A/B/C/D 协议；新增 L1 在线 MACE 与 L2 latent-teacher student 对照，WP-5 分体系内和跨 ABFE 通用性验收 |
| 2026-08-03 | DEC-020 | 将真实 OFF24 合成三原子 graph/latent/autograd smoke 登记为 `C1_SYNTHETIC_PASSED`，但不视为 Atenolol 科学资格 | 严格 latent slice 为 `node_feats[:,512:640]`，shape `[1,128]`；ligand/environment gradient norm 为 `1.42825/7.93055`，MACE parameter grad count 为 `0`；联合回归 `107 passed, 2 skipped`。图/PBC 契约：`local_residual/mace_graph.py` 固定 `edge_cutoff_angstrom=6.0`（`smoke_exp012_mace_latent.py` 对此值硬校验，非 6.0 直接拒绝）、`interaction_layers=2`；12 Å 是两跳可能达到的几何上界，不定义径向候选集合（该语义由 DEC-024 明确修正）。triclinic PBC 用最小像约定检查。C0 架构核查报告已对真实（非合成）`MACE-OFF24_medium.model` 生成并通过，存于 `output/outer_lambda_exp012/mace_contract.json`（`status=PASSED_READ_ONLY_ARCHITECTURE_INSPECTION`，`policy` 全 `False`），可直接作为 `run1/frame0` smoke 的 `--c0-report` 输入 | 下一步限定为 Atenolol ligand/environment 身份冻结、两份 provisional 文件审计、`run1/frame0` CPU smoke 和 CUDA 同帧对照；在此之前禁止训练 |
| 2026-08-03 | DEC-021 | 补齐生成 provisional 环境/mapping 文件缺失的工具（发现脚本+CLI），但不在本机对真实 Atenolol 数据执行；不単方面提议或冻结 preregistration 剩余 9 个 unresolved 字段的具体数值 | 此前只有校验器（`local_residual/environment.py`、`atom_mapping.py`）而没有生成器；EXP-010 的失败根因之一是单帧 0.5 nm 逐原子选择切碎残基，新工具按整条残基取舍以避免重犯 | 新增 `scripts/discover_exp012_environment_config.py`（完整残基环境发现+config 组装，标记 chain-terminal 供后续 capping 决策）与 `scripts/build_exp012_atom_mapping.py`（对已 sealed 环境 manifest 生成 canonical atom mapping）；仅用合成 mdtraj topology 做过 py_compile 和人工代码走查，未在 openmm_dev 环境实际执行（本机默认 Python 无 numpy/mdtraj）；真实 Atenolol provisional 文件生成、preregistration 9 个 unresolved 字段仍需用户在计算节点上执行并决定 |
| 2026-08-03 | DEC-022 | 在真实 Atenolol 数据上生成并冻结两份 provisional 文件；`pocket_cutoff_nm=0.5` 与三条 scratch run 末帧并集为本轮采用值，不代表已对 cutoff/参考帧做过消融 | `openmm_dev` 环境 10/10 单元测试通过（一次修正：`load_atom_mapping` 需要文件路径而非已解析 dict，测试自身的 bug，非脚本 bug）；对 `output_lrc_fix/topology.cif` + `hard_window0_run1/2/3` 末帧执行发现，得到 21 个完整残基、339 个环境原子、0 个 chain-terminal，加 41 个 ligand 原子共 380 节点；元素集合 H/C/N/O/S 与 mapping config 的 `supported_atomic_numbers=[1,6,7,8,16]` 完全吻合 | `provisional_mace_environment_manifest.json`（sha `ffa52ebc...254f05`）与 `provisional_mace_atom_mapping.json`（sha `2c74e5e0...86597a`，`manifest_canonical` ordering）已生成，IMPLEMENTATION_PLAN §13 对应条目已勾选；下一门是 `run1/frame0` 真实 CPU MACE latent/autograd smoke（`scripts/smoke_exp012_mace_latent.py`），其 `--c0-report` 输入尚未核实，在此之前仍不得训练任何模型 |
| 2026-08-03 | DEC-023（已撤销） | DEC-022 的 provisional 文件仍作废，但“应改成 ligand-centered 12 Å 球”的修复结论错误 | 当时把 `edge_cutoff × interaction_layers = 12 Å` 的几何上界误当成了必须完整纳入的径向支持球；这会额外装入没有两跳路径连接 ligand 的水/蛋白节点和大量无关 environment–environment edges | 由 DEC-024 取代；禁止执行原建议的 1.5 nm 径向发现命令。旧实现与错误结论仅作为失败证据存档 |
| 2026-08-03 | DEC-024 | EXP-012 两层支持域冻结为 6 Å 邻接图的精确两跳闭包，而非 12 Å ligand-centered 球 | 真正依赖为 `S0=L`、`S1=N_6Å(S0)`、`S2=N_6Å(S1)`；8 Å 原子若没有中间 6 Å 邻居不会影响 ligand latent，而距 ligand 超过 6 Å 的原子若通过 S1 连通则必须纳入。12 Å 仅是最远几何上界 | `local_residual.mace_graph.topology_n_hop_closure` 成为 discovery/smoke 共用定义；每参考帧按 PBC 严格 `<6 Å` 做两层 frontier 扩展，多帧取并集，环境按完整残基收口。CLI 改为 `--edge-cutoff-angstrom 6.0 --interaction-layers 2`；旧 provisional 不覆盖，节点重生成并通过 CPU smoke 后再冻结新哈希 |
| 2026-08-03 | DEC-025 | 冻结 OMOL 两层 latent 的执行层为 zero-based `product_layer_index=1`，并登记 `run1/frame0` CPU smoke 为 `C1_REAL_FRAME_CPU_PASSED` | OMOL contract 有 3 个 product layers；若取最终 layer 2，就需要三跳且会计算不需要的第三层。修正后在 layer 1 hook 处 early-stop，只取该层开头的 `1024x0e`。frame0 精确闭包为 1538 原子（hop 0/1/2=`41/343/1154`），整残基收口后固定图为 2135 节点、155624 有向边；float32 latent `[41,1024]` 有限，ligand/environment 梯度 norm `32.2197/22.2791`，参数梯度数 0，重复差 0 | C1 CPU 可导性通过；报告 SHA `ce8fd06c...58db5`。仍不得宣称性能或科学资格；下一门只做同图 CUDA float32 对照，之后才决定 readout/训练。此前一次 96 GB RAM OOM 无完成报告，归因边界仅限于旧实现同时存在全配对构边、完整三层 float64 OMOL 和两份可导 forward；未单独测量各分量，不能把 OOM 精确归给某一项 |
| 2026-08-03 | DEC-026 | 修复 `MaceLatentBasisAdapter` 中不带 index 的 `--device cuda` 与实际张量具体设备（如 `cuda:0`）比较不相等的代码缺陷；同时排除显存碎片化作为 CUDA 对照 OOM 的原因，暂不采纳更大显存设备或 gradient checkpointing 中的任一个 | `torch.device("cuda")`（index=None）与创建后的张量 `.device`（具体 index）按 PyTorch 语义不相等，导致本门要求的确切命令形式必然先于 forward 被拒绝，已在 `mace_latent.py:223-225` 归一化修复；随后三次 CUDA 尝试（15.47/10.57/23.58 GiB 三张卡）在加与不加 `empty_cache`/`expandable_segments` 缓解措施下，OOM 处的已占用显存均为 14.57–14.59 GiB 且卡在同一算子，数字未因缓解措施变化 | EXP-012 CUDA float32 对照仍为 `BLOCKED_ON_VRAM`，未生成 CUDA 报告，不计入 `C1` 通过；下一步在“换更大显存设备”与“对已执行的两层 interaction/product block 做 gradient checkpointing”之间由用户选择，任一选项都不擅自执行；两跳支持域（DEC-024）和模型选择均不因显存不足而放宽 |
| 2026-08-03 | DEC-027 | 新增 `derived_5a` 为与 `original_6a`（DEC-024，6 Å）并列的第二个显式预注册 EXP-012 表示候选臂：`--edge-cutoff-angstrom` 只接受 `{6.0, 5.0}` 两个值，5.0 对应两层 `geometric_upper_bound_angstrom>=10.0`；两臂各自独立生成 manifest/mapping 与 C1 smoke，互不覆盖，最终按 held-out gap variance、梯度稳定性与显存成本判决，不预设胜负 | 依据是用户在本项目之外的研究经验（“4.5 Å 外信号噪声明显增加”）叠加当前 VRAM 约束，明确记为外部先验，不是 EXP-012 内测得的证据；`MaceLatentBasisAdapter.forward` 对 ligand latent 的读出没有环境原子级别的可分解结构（只有 `ligand_latent = node_feats[ligand_mask, slice]`），因此“latent 后按距离衰减”不能等价于收窄编码器输入图，已放弃该方案；`MaceGraphConfig` 本身已支持任意有限正 cutoff（`geometric_upper_bound_angstrom >= edge_cutoff_angstrom*interaction_layers` 是唯一约束），真正的功能阻塞只在 smoke 脚本的硬编码 `!=6.0` 校验 | 修改 `scripts/smoke_exp012_mace_latent.py`：用 `ENCODER_VARIANTS={6.0:"original_6a",5.0:"derived_5a"}` 替换硬编码校验，报告新增 `encoder_variant`、`model_r_max_angstrom`（取自 C0 报告的 `expected.r_max_angstrom=6.0`，两臂相同，因为模型本身未变）、`graph_cutoff_angstrom`、`original_encoder_numerically_preserved` 四个字段；新增 `--output` 已存在即拒绝写入的防覆盖门；`local_residual/mace_graph.py` 中两处写死 “6-Angstrom”/“6 Angstrom” 的注释与报错信息已参数化为实际配置的 cutoff。尚未执行：按 5 Å 重新生成 provisional environment/atom-mapping 与 CPU C1 smoke；6 Å 分支的 gradient checkpointing 尚未实现 |
| 2026-08-04 | DEC-028 | 登记 `derived_5a` 的 `run1/frame0` CPU float32 smoke 为 `C1_REAL_FRAME_CPU_PASSED`（第二个通过 C1 的候选臂）；两臂 C1 均已通过，进入各自下一门（5 Å 做 CUDA 对照，6 Å 等 checkpointing） | 同一 topology/ligand indices/base system/box/trajectory/frame0/C0 报告，仅 `--edge-cutoff-angstrom 5.0 --geometric-upper-bound-angstrom 10.0`；两跳精确闭包从 1538 原子（6 Å）降到 974 原子（5 Å，hop 0/1/2=`41/219/714`），整残基收口后节点数 2135→1444，边数 155624→60048（降至 38.6%，好于此前 42% 边数下降的估计）；float32 latent `[41,1024]` 有限，ligand/environment 梯度 norm `32.9454/19.9347`，repeat 最大差 `0`，参数梯度数 `0`，CPU 耗时 172.7s→34.5s | 报告 SHA `d28be435...9c7a0d`；`encoder_variant=derived_5a`、`model_r_max_angstrom=6.0`（模型未变）、`graph_cutoff_angstrom=5.0`、`original_encoder_numerically_preserved=false` 均按 DEC-027 schema 写入。仍不得宣称性能/科学资格；下一步是同图 CUDA float32 对照（预期显存需求随边数近似线性下降，但未实测，不得假设一定能进 24 GiB）。`scripts/discover_exp012_environment_config.py` 顺手修了一个真实 bug：`--report-output`/`--config-output` 写入前未 `mkdir -p` 父目录，已在 §该脚本 修复，与 5 Å/6 Å 的科学决策无关 |
| 2026-08-04 | DEC-029 | 登记 `derived_5a` 的同帧 CUDA float32 对照为 `C1_REAL_FRAME_CUDA_PASSED`；`derived_5a` 是首个同时通过 CPU 和 CUDA C1 的候选臂，`original_6a` 仍 `BLOCKED_ON_VRAM` | 同一 frame0/manifest/mapping/C0 报告/product-layer-1，仅 `--device cuda`；无 OOM，整次运行 23.87s（单帧 19.76s）。CPU↔CUDA 相对差：latent norm `7.53e-7`、ligand 梯度 norm `1.16e-7`、environment 梯度 norm `-3.83e-7`，均是 float32 舍入量级，非真实分歧；CUDA 内 repeat 差 `1.64e-7`（CPU 内 repeat 差为精确 `0`，非 bug，是 GPU kernel 的正常非确定性量级）；参数梯度数 `0`，两类坐标梯度均有限非零 | 报告 SHA `ba7d053d...deb37bc8d59`；`derived_5a` 的 C1（CPU+CUDA）现已完整通过，`original_6a` 因 VRAM 仍卡在 CUDA 半门。仍不得宣称性能/科学资格、不得训练。下一步：(a) 6 Å 分支等 `inspect.getsource(ScaleShiftMACE.forward)` 以设计 checkpointing；(b) 两臂真正的判决（held-out gap variance、梯度稳定性、显存成本）要等 Arm A/B/C/D 表示消融和训练本身，不能仅凭 C1 通过就选 `derived_5a` |
| 2026-08-04 | DEC-030 | 正式定性当前路线为 §11A.4 的 **L2**（离线 frozen-MACE teacher → 在线轻量可导 student），明确三个实体的边界，L1 从"当前下一步"降级（不删除）；修订下一步顺序为：多帧支持域审计（report-only）→ `derived_5a` 离线 latent cache → cached-latent 线性/ridge readout 的 held-out gap-variance 验证 → 有增益才蒸馏 `LocalResidualStudent` → student 在线力学与性能资格 | 三个实体：`original_6a`（离线参考 teacher，CPU 可跑，CUDA 24 GiB OOM，DEC-026，不进 MD 每步）；`derived_5a`（当前主要离线 teacher，CPU+CUDA C1 均通过，DEC-028/029，产出 latent/标量残差目标/必要时坐标梯度，同样不进 MD 每步）；`LocalResidualStudent`（唯一计划中的在线模型，尚不存在代码——三个并行只读探索确认 `local_residual/`、`scripts/`、`tests/` 内均无该模型/Arm A/B 的任何实现，仅 `IMPLEMENTATION_PLAN` 里标注"计划 | 尚不存在"）。teacher 的 CUDA smoke 是离线计算资格，不是 MD 部署资格，teacher 不需要过 OpenMM/NVT/ns-day 门。`PLAN_outer_lambda_neural_basis.md:190` 明确要求"不因 MLP 很小就假定在线 MACE 足够便宜，必须实际比较 L1/L2 的 ESS/GPU-hour"——本决策不是绕开这条要求，而是用本 session 已实测的 teacher 侧单帧数字（`derived_5a` CUDA 19.8s、`original_6a` CPU 172.7s 每帧，仅做一次 latent 提取）说明：真正的每 MD 步预算需要 O(ms)级，而 teacher 自己最便宜的离线单帧成本已是 O(10s)级，L1（同一 frozen encoder 直接进在线路径）在当前证据下已不可行，不依赖对 student 成本的任何假设。探索还确认：`local_residual/loss.py::bidirectional_gap_variance_loss` 已实现 §11A.2 的 \(\mathcal L_{\rm gap}\)（MBAR 式重要性加权双向 gap 方差+能量/力正则），三条真实 run 的五态逐帧 ledger（`adjacent_gap_reduced`、`log_importance_unnormalized`）已存在，但没有任何代码把 MACE latent 和 ledger 接起来，也没有多帧 latent cache——这正是本决策新排的下一步顺序里第 2/3 步要填的空 | 本决策不删除 §11A.4 的 L1 定义，只标注降级依据；不构成 production 批准；`original_6a` 的 gradient checkpointing 是独立、并行的另一条线（等 `inspect.getsource(ScaleShiftMACE.forward)`），与本决策的教师/学生分工无关 |
| 2026-08-04 | DEC-031 | 完成 DEC-030(a) 多帧支持域审计（`scripts/audit_exp012_multiframe_support_domain.py`，report-only，不设硬门），针对三条真实 run（`hard_window0_run1/2/3`，共 1500 帧）跑通：1499/1500 帧违规，`derived_5a` 的 frame0-only manifest 不足以支撑 (b) 的离线多帧 latent cache；修复方向定为改用这 1500 帧闭包的并集重建 manifest（精确覆盖，而非代表帧外推） | report_sha256 `a74ea2352263ea9b25e324c9d0930a0199b0fb826d453ab2715b54cd82cf9b69`，`num_workers=32`。三条 run 违规帧数分别为 499/500/500，单帧最坏遗漏原子数 122/134/141（占 1444 个固定原子的 8.5–9.8%）。遗漏原子中约 86–88% 是水（`HOH`），这是正常物理（水在两跳壳层内持续扩散进出），不是缺陷；但约 24 个非水残基 （ALA50/63/131/168、ARG167、ASN166、ASP246、CYS154/160/227、ILE233、 LEU57/142/182/229/256、PHE77、PRO230、TRP4/130、TYR155、VAL129/245）与一个钠离子 在三条独立 run 中都被遗漏——同一批真实残基而非逐 run 随机噪声，说明它们结构上正好卡在 5 Å/两跳边界。由于该 teacher 按 DEC-030 定性为离线、只需要嵌入这 1500 个已采集帧（不需要 泛化到未来帧），可以用这 1500 帧闭包的并集做到精确覆盖，而不是像 frame0-only 那样赌一个 代表帧 | 为此扩展了 `scripts/discover_exp012_environment_config.py`：新增 `--frame-stride-all N`（对每个 `--trajectory` 取第 0,N,2N,... 帧为参考帧，与 `--frame-index` 互斥，报告新增 `reference_frame_mode`/`reference_frame_provenance`/ `reference_frame_count` 字段）；`discover_complete_residue_environment()` 新增 `num_workers` 参数，逐参考帧闭包按 `ProcessPoolExecutor` 并行、跨进程按元素级 minimum 归约（顺序无关，新增测试 `test_parallel_and_serial_reference_frame_reduction_agree` 锁定并行/串行结果一致），默认仅在 `frame_count>=8` 时才自动并行。审计脚本本身也从串行 改为按 `ProcessPoolExecutor` 并行（`--num-workers` 默认取 `os.sched_getaffinity` 核数而非整机 `cpu_count`，避免在 SLURM/cgroup 限额节点上超订； 两处均不使用 GPU，因为本审计和本发现都不跑 MACE 模型）。顺手修了两个动态加载脚本的 测试文件（`test_exp012_environment_discovery.py`、 `test_exp012_multiframe_support_domain_audit.py`）里遗漏的 `sys.modules[spec.name]=module`——没有这行，动态加载模块里定义的函数无法被跨进程 pickle 找到，`num_workers>1` 会在测试环境下崩溃。尚未执行：用 `--frame-stride-all` 重建 `derived_5a` 的 config/manifest/mapping，重新过一遍 CPU/CUDA C1 smoke（新 manifest SHA 必然与 DEC-028/029 记录的不同，需要重新生成两份 smoke report，成本各自 ~35s/~24s，不是要作废的大计算），再对新 manifest 重跑本审计确认违规数降到 0（或给出 有记录的残留量）后才能进入 DEC-030(b) 的 latent cache |
| 2026-08-04 | DEC-032 | 撤销 DEC-031 的"跨 1500 帧闭包并集固定 environment manifest"策略，改为逐帧独立精确两跳闭包（无固定图、无整残基收口）作为离线 teacher 的正式图构造策略 | 为此撤销（不删除，作废使用）：`derived_5a` 的 discovery config/report 已实际用 `--frame-stride-all 1` 重建，结果候选池从 1444 膨胀到 4874 个环境原子（1136 个残基，chain-terminal 3 个），固定图节点数变为 4915——比 `original_6a` 已在 CUDA 上 OOM 的 2135 节点图还大一倍以上，若继续这条路会重新引入 DEC-027 选 5 Å 就是为了避开的显存问题。正确认识：teacher 是离线工具，从不进 OpenMM/MD，没有任何理由为一张跨帧共用的固定图付出代价；正确做法是逐帧独立构造精确两跳闭包 S_a=S0∪S1∪S2（ligand、5 Å 邻居、邻居的 5 Å 邻居），这样零支持域遗漏是构造性保证（不是审计结果），不需要把 1500 帧见过的残基同时塞进每张图，也不会为了控制固定图大小而排除环境交换最强、恰恰最有训练价值的困难帧。 新增 `local_residual/teacher_graph.py::build_teacher_graph_for_frame`：不接受也不需要 environment manifest/atom mapping，直接对当前帧的 `topology_n_hop_closure` 结果按 topology index 升序取节点（无整残基收口），复用 `mace_graph.py` 的 `_build_cutoff_edges_chunked`/`_face_heights`/`_floating_tensor`/`_torch` 构边；ligand 41 个原子在排序后相对顺序帧帧不变（拓扑索引固定），因此缓存的 `[41,1024]` latent 可以跨帧直接比较或喂线性 probe。整残基收口被移除的依据是 EXP-010 的 fragment energy subtraction 需要完整残基，而当前 ligand latent 读出从不计算 fragment energy——收口是否改变数值是可验证问题，不是可以直接假定的前提，因此新增 `scripts/smoke_exp012_teacher_graph_equivalence.py` 在 frame0 上比较 974-node 精确闭包（graph A）与 1444-node 整残基收口图（graph B，复用 DEC-028/029 已冻结的 manifest/mapping），比较完整 ligand latent 张量（不仅 norm）、scalar probe 和 ligand 坐标梯度，只报告差异不设门（`status=COMPARISON_ONLY_NOT_A_GATE`）。另新增 `scripts/audit_exp012_per_frame_teacher_graph_geometry.py`：对三条 run 1500 帧只做几何构图（不跑 MACE），报告每帧 node/edge count、hop 0/1/2 计数、water/ion/other 环境原子组成与 max/mean/P95/P99 汇总，显式给出 `overall_max_edge_count_frame`/`overall_max_node_count_frame`，替代"假设 frame0 是最坏情况"。再新增 `scripts/smoke_exp012_teacher_graph_latent.py`（`smoke_exp012_mace_latent.py` 的对应版本，改用 `build_teacher_graph_for_frame`），供在 geometry 扫描选出的最大图帧上跑 CPU/CUDA C1。新增测试 `tests/test_exp012_teacher_graph.py`：闭包即节点集（无收口）、ligand 相对顺序在环境原子数变化时不变、不支持元素 fail-closed、坐标自动求导可用。三个新脚本均未在真实数据上执行——需要 openmm_dev 环境和真实 MACE 模型/GPU。 | DEC-031 的 4874-atom union discovery 报告（`output/outer_lambda_exp012/two_hop_allframes_derived_5a/`）保留存档但标记 `FIXED_UNION_POLICY_REJECTED_FOR_OFFLINE_TEACHER`，不晋升为任何运行时 manifest，也不再重建 config/manifest/mapping 或对其执行 C1 smoke。DEC-030(a) 的审计脚本（`audit_exp012_multiframe_support_domain.py`）以及 `discover_exp012_environment_config.py` 的`--frame-stride-all`/并行扩展保留在库中，作为"发现某条路线不可行"过程的正当代码产出，不因方向撤销而删除。下一步顺序：① 跑 equivalence smoke 决定是否正式删除整残基收口；② 跑 1500 帧 geometry-only 扫描拿到真实 node/edge count 分布与最大图所在帧；③ 对该帧跑 CPU 再跑 CUDA C1；④ 通过后才进入 DEC-030(b) 的逐帧 latent cache 生成（不使用固定 manifest）。仍不得训练。 |
| 2026-08-04 | DEC-033 | DEC-032 Option C 验证通过并落地：per-frame teacher latent cache 三条 run 全部生成完成，latent-only，未与 ledger 拼接 | 实现：`local_residual/mace_latent.py` 的 `positions.requires_grad` 检查改为仅在 `require_coordinate_grad=True` 时强制（新增两个测试锁定放宽后与放宽前行为）；`local_residual/teacher_graph.py` 新增 `compute_canonical_graph_membership`（CPU float64 专用，强制校验，输出 `graph_membership_sha256`）与 `build_teacher_graph_from_membership`（在目标 device/dtype 上只做原子选择与 shifts 重算，不重新判定 membership），原 `build_teacher_graph_for_frame` 不变（C1 已用它通过，未回填改动）；四个新测试锁定：split 路径与原 combined 函数在同设备同精度下结果一致、membership hash 在精度变化后原样保留、target_positions 原子数不足时 fail-closed。`scripts/audit_exp012_per_frame_teacher_graph_geometry.py` 改为直接调用 `compute_canonical_graph_membership`，去掉不再有意义的 `--dtype`（membership 永远 CPU float64），报告 schema 升到 v2，逐帧新增 `graph_membership_sha256`，顶层新增 `graph_membership_device`/`graph_membership_dtype`。`scripts/build_exp012_teacher_latent_cache.py` 改为：CPU float64 决定 membership → `build_teacher_graph_from_membership` 在 CUDA float32 上组装 → `torch.no_grad()` 单次前向（不再需要 `requires_grad=True`）；硬门新增逐帧 membership hash 比对（不仅 node/edge count），拒绝加载非 v2 schema 或非 CPU/float64 membership 的 geometry report；checkpoint 与最终 npz 都新增 `graph_membership_sha256` 字段，resume 时对已存在 checkpoint 也重新核对 hash（不仅count），报告 schema 升到 v2 并新增 `model_execution_device`/`model_execution_dtype`。 | 三次真实执行（openmm_dev，CUDA float32，`hard_window0_run1/2/3`，各 500 帧）全部通过硬门，无一帧 membership hash 或 node/edge count 不一致：`latent_cache_hard_window0_run1.npz` report_sha256 `bfd1a8ef9df26b111b593ea734f1a8cf76c6658a087cd653d2bcdb3ab3bbb639`、run2 `50138e76b239e39944f2c9e55305b77f66f6c447cb44b20921ae02d15f0b72a8`、run3 `077c8737ef7d9674a9cf7a818ce5b9734ade0b407b3c943b4cf6b2e233476fcc`；三份 npz 的 `npz_sha256` 均与实际文件哈希核对一致，文件大小各 ~86.16 MB（共 ~258 MB，符合预估）。每份 `ligand_latent` 形状 `[500,41,1024]` float32；node/edge count 范围 run1 `[940,1052]`/`[37834,44444]`、run2 `[952,1050]`/`[38428,44152]`、run3 `[958,1066]`/`[39182,45656]`——run3 的上限恰好等于此前 C1 通过的那一帧（frame343，1066 节点/45656 边），与几何扫描结果完全吻合。三次运行实际耗时 150.8s/153.7s/154.8s（共 ~7.7 分钟），远低于早先基于双前向 smoke 外推的 ~4.3 小时估计（bulk 每帧只做一次 no-grad 前向，且实际图比 C1 smoke 用的最坏帧更小）。`ledger_joined=False` 确认未与 target ledger 拼接，representation 与 thermodynamic target身份保持分离。下一步是 DEC-030(c)：cached-latent 线性/ridge readout 的 held-out（leave-one-run-out）gap-variance 验证，需要先做 ledger join（按 run_id/frame_index/trajectory SHA/frame_count/preregistration SHA fail-closed 对齐），仍不得训练任何模型。 |
| 2026-08-04 | DEC-034 | DEC-030(c) 工具链实现完成（join + 线性/ridge readout + leave-one-run-out held-out gap-variance 验证），尚未在真实 latent cache 上执行 | 新增 `scripts/join_exp012_teacher_latent_cache_with_ledger.py`：latent cache（DEC-033）本身是纯 representation，不含任何 ledger 字段，这里把它和 `output/outer_lambda_exp012/mm_ledger_cuda/<run_id>/` 的 `adjacent_gap_reduced [500,4]`/`log_importance_unnormalized [500,5]` 逐帧拼接。`delta_A`（每条相邻边的包络增量）不重新拟合、不猜测，直接从 `protocols/EXP-012_preregistration.json` 的 `target.global_schedule.A_k`（`A_definition=sin_squared_pi_lambda_vdw`，在 `global_state_ids=[0,1,2,3,4]` 处切片得 `A_k_window=[0.0, 0.0566144229, 0.1952759894, 0.3737891695, 0.5568140579]`，`delta_A=[0.0566144229, 0.1386615665, 0.1785131802, 0.1830248884]`）读取，并独立用 `sin^2(pi*lambda_vdw)` 公式重新核对，对齐 PLAN 文档 line 174 "第一轮冻结全局 A_k，禁止同时学习 A_k 造成尺度简并" 的明确要求。fail-closed 对齐排查中发现一个真实但良性的差异：ledger report 自身的 `preregistration_payload_sha256`（`037342f1...`）与当前 preregistration 的 payload_sha256（`aba95cd2...`，与 cache report 的 `preregistration_sha256` 完全一致）不相等——核实后确认这只是 ledger 生成之后文档发生了无关编辑（ledger 自身记录的 `f_k_kj_mol`/`lambdas_vdw` 与当前 preregistration 的 target 部分逐位精确一致，三条 run 之间也互相一致），因此改为直接比对这两个物理相关字段而非要求整份文档哈希相等；同时要求 cache 与 ledger 的 `frame_index` 数组都逐位精确等于 `0..499` 才允许按位置拼接。新增 `scripts/fit_exp012_local_residual_linear_readout.py`：线性 readout `basis_reduced = w^T·standardize(pooled_latent)`，无 intercept——`Var(X+c)=Var(X)` 对任何逐帧常数 c 成立，intercept 进入 `corrected_gap` 只是逐边常数，对 `gap_variance_loss` 贡献恒为零，因此不是遗漏而是该目标函数下无需学习的参数；ridge 系数由只用两条训练 run 的内层 2-way CV 选择（不接触 held-out run，避免泄漏），再在两条训练 run 合并数据上用选中系数重新拟合，最后在真正 held-out 的第三条 run 上比较 `B=0` 基线与拟合值的 `gap_variance_loss`；`A_k`/`delta_A` 全程冻结不参与联合拟合；`delta_A` 固定后目标函数是 `w` 的凸二次型（仿射readout 嵌在方差目标里，加一个凸的 L2 ridge 项），因此用 `torch.optim.LBFGS`（strong Wolfe line search）保证收敛到唯一全局最优，不存在局部极小或学习率调参问题；直接复用已测试的 `local_residual/loss.py::bidirectional_gap_variance_loss`，不重新推导平行的闭式解，避免引入与被审计代码不一致的第二套数学实现。新增测试 `tests/test_exp012_local_residual_linear_readout.py`：构造一个 gap 恰好是 pooled_latent 线性函数、三条合成 run 共享同一真实关系的数据集，验证三个 leave-one-run-out fold 全部相对 `B=0` 基线改善 `>90%`，且输出 report 的 policy 明确标注 `a_k_learned=false`/`mace_encoder_trained=false`/`local_residual_student_trained=false`（这一步只训练线性 readout，不训练 MACE、不学习 A_k、不是 (d) 的 `LocalResidualStudent`）。 | 两个脚本均未在真实 `derived_5a` latent cache 上执行（需要 openmm_dev 环境；只需 numpy/torch 做 CPU 线性回归，不需要 MACE/GPU）。示例命令： `python scripts/join_exp012_teacher_latent_cache_with_ledger.py --latent-cache-dir output/outer_lambda_exp012/teacher_latent_cache --output output/outer_lambda_exp012/teacher_latent_ledger_join.npz` 然后 `python scripts/fit_exp012_local_residual_linear_readout.py --joined output/outer_lambda_exp012/teacher_latent_ledger_join.npz --output output/outer_lambda_exp012/local_residual_linear_readout_report.json`。只有三个 leave-one-run-out fold 中至少一个（理想情况是全部）显示 held-out gap variance 相对 `B=0` 基线下降，才能进入 DEC-030(d) 蒸馏 `LocalResidualStudent`；否则按 PLAN 文档既定标准记为该表示在当前 Arm C 下无法通过 held-out 资格，需要先做 Arm A/B/D 消融或重新考虑表示，而不是直接尝试蒸馏一个没有验证过增益的表示。仍不得训练 MACE 本身或学习 A_k。 |
| 2026-08-04 | DEC-035 | DEC-030(c) 已在真实 `derived_5a` latent cache 上执行：三个 leave-one-run-out fold 全部相对 `B=0` 基线改善（不只是最低门槛的"至少一个"），线性 readout 通过 held-out gap-variance 资格，可以进入 DEC-030(d) 蒸馏 `LocalResidualStudent` | join 报告 report_sha256 `8dfc47e3352534f8b67826ee570f6830de2b618cf935ff9354e74f0082c016ce`，npz_sha256 `1cf4e2cc4409c652005af322ea559ea0bd4c19e11a0cd6b75626526776f079cf`，总计 1500 帧三个 partition；`A_k_window=[0.0, 0.0566144229, 0.1952759894, 0.3737891695, 0.5568140579]`，`delta_A=[0.0566144229, 0.1386615665, 0.1785131802, 0.1830248884]`（直接取自 preregistration 并独立用 `sin^2(pi*lambda_vdw)` 核对一致，未重新拟合）。readout 报告 report_sha256 `d77a8e132780270363abb4a33572912e518c102ef8c1f4ed38d36df92c7b05c3`：三个 fold 结果为 held-out run1（选中 ridge=0.1）baseline 0.4486→fitted 0.2698，改善 39.9%；run2（ridge=0.001）baseline 0.2692→fitted 0.1909，改善 29.1%；run3（ridge=0.001）baseline 0.3933→fitted 0.1385，改善 64.8%；`all_folds_improved_over_baseline=true`，平均相对改善 44.6%。三个 fold 改善幅度彼此不同（不是可疑的一致值），且 2/3 fold 选中了 ridge 网格里最小的候选值 `1e-3`——这说明内层 CV 更偏好接近零正则化，网格下界可能限制了结果，值得后续把网格往下扩展（如 `1e-6/1e-5/1e-4`）复核，但不改变"held-out 确有改善"这个定性结论 | DEC-030(c) 判据（"只有 held-out 上有增益才进入 (d)"）已满足，且是最强形式（全部 fold 而非至少一个）。下一步是 DEC-030(d)：蒸馏 `LocalResidualStudent`（尚不存在代码），规模远大于本步骤——需要设计并实现一个轻量在线可导 encoder、训练循环、力守恒/TorchScript/OpenMM Reference 一致性检查、CPU/CUDA 对照、短 NVT 稳定性，这些都是 PLAN 文档 §WP-4C.4 已列出但尚未开始的资格门；在实现前需要与用户对齐范围和顺序，不能因为线性readout 通过就默认下一步的具体设计。仍不得训练 MACE 本身、不得学习 A_k、也不得在没有对齐范围前直接开始写 `LocalResidualStudent` 代码。 |
| 2026-08-04 | DEC-036 | DEC-030(c) 正式冻结登记：DEC-034/035 的三-fold 结果作为最终结果，不因 ridge 网格边界现象重跑更宽网格来替换它 | 2/3 fold（run2、run3）在内层 CV 里选中了 ridge 候选网格 `[1e-3,1e-2,1e-1,1.0,10.0,100.0,1000.0]` 里最小的 `1e-3`，说明内层 CV 可能更偏好比网格下界更小的正则化；但看到这个边界现象之后才决定把网格往下扩展去"验证/改善"，属于事后调整（post-hoc）——可能改变改善幅度的具体数值，不可能把"三个 fold 是否都改善"这个已经很确定的定性结论反转（三个 fold 分别 39.9%/29.1%/64.8%，幅度彼此不同，不是可疑的一致值，是真实信号的典型特征而非过拟合/泄漏的痕迹）。因此按照"先冻结判据再看结果，不看到结果再调判据"的原则，DEC-034/035 的数字直接作为 DEC-030(c) 的最终登记结果 | DEC-030(c) 判据（"只有 held-out 上有增益才进入 (d)"）已满足且是最强形式（全部 3 个 fold 而非至少 1 个）。允许后续单独跑一次更宽 ridge 网格（如 `1e-6/1e-5/1e-4`）作为补充分析，但该次运行的报告必须显式标注 `sensitivity-only`，不得覆盖或替换本次登记的 join report_sha256 `8dfc47e3352534f8b67826ee570f6830de2b618cf935ff9354e74f0082c016ce` 与 readout report_sha256 `d77a8e132780270363abb4a33572912e518c102ef8c1f4ed38d36df92c7b05c3`。下一步进入 DEC-030(d0) |
| 2026-08-04 | DEC-037 | 冻结 DEC-030(d0)：`LocalResidualStudent` 编码前的设计契约，明确"在线动态环境表示"是编码前必须最先单独解决的问题，其余各项（最小架构、teacher-target 协议、必需对照、计算/部署预算、分阶段实现）随之冻结为框架但细节待该问题解决后再定 | (c) 通过只证明"frozen-MACE 局部残差表示对这个系统有用"，不等于已经知道如何把它变成一个能在 OpenMM 每步内运行的在线可导模型——teacher 用的是逐帧动态水分子身份和逐帧图（DEC-032 Option C），student 必须先回答：瞬态水分子身份如何处理、ligand–environment 近邻发现如何进行、triclinic PBC 如何处理、cutoff 归属如何判定、如何在 TorchScript 里表达一个逐步都可能变化的动态近邻结构、如何避免每步扫描一个巨大固定环境；这必须先于网络结构选择解决，不能先选网络再回头凑答案。冻结的其余四项：① 最小架构第一候选是最便宜的标量旋转不变局部模型（typed embedding + 平滑 ligand–environment radial/contact 特征 + 至多 1–2 个轻量 interaction block + ligand-only 不变 pooling + 有界标量 `B_student`），不是完整张量-equivariant MACE 式 student——旋转不变标量能量对坐标求导本身就给出正确等变力，不需要为等变而引入 irreps 机器，只有最简单候选失败才升级；② teacher-target 协议与 (c) 同构的 leave-one-run-out（两条 run 拟合/训练，第三条 run 评估），student loss 同时含直接 gap 优化项和蒸馏项，teacher 不能被当成无条件 ground truth；③ 必需对照：同一架构必须同时训练 direct-gap student（无 teacher target）与 distilled student（同架构+teacher loss），否则任何增益无法归因于 MACE teacher 本身；④ 计算/部署预算须在编码前冻结：最大参数量、最大 neighbor/edge 数（经验参考：DEC-033/034 实测 teacher 精确闭包在这 1500 帧范围约 940–1066 节点/37834–45656 边，不直接决定 student cutoff）、目标每 MD 步毫秒数、GPU 显存上限、允许的 cutoff、训练 seed 与 epoch 数、早停规则、held-out 改善判据 | 分阶段条件式实现顺序冻结为 D1（离线 student 拟合，held-out gap variance 与 teacher fidelity，用真实逐帧坐标算 student 自己的特征，不是只读 teacher 的 cached `pooled_latent`）→ D2（坐标/autograd 资格：有限差分、cutoff 平滑性、力尾部）→ D3（部署资格：TorchScript、OpenMM Reference、CUDA 一致性、耗时）→ D4（动力学资格：短 NVT、稳定性、独立重复），每阶段失败即停，不得跳阶段；只有 D1 仍保留有意义的 held-out 改善才允许启动 D3/D4 的 OpenMM/NVT 工作。本轮不写任何 student 代码；下一步是把"在线动态环境表示"单独作为一次设计讨论解决，其余各项在该讨论之后才细化为可执行任务 |
| 2026-08-05 | DEC-038 | 冻结 DEC-030(d0) 第 1 项"在线动态环境表示"：`LocalResidualStudent` 不追踪瞬态水分子持久身份、不使用固定 manifest，每步用 `local_residual.geometry.ligand_environment_cross_edges`（一条独立于 teacher 代码路径的 minimum-image cutoff 实现）逐步动态重算 ligand–environment cutoff funnel；能量权重用 `local_residual.geometry.quintic_c2_cutoff`（quintic C2 平滑包络）而非硬 `g_i∈{0,1}` 门控——离散候选成员与能量平滑性是两个独立层次 | 新增 `scripts/smoke_exp012_student_environment_funnel.py`，在 `openmm_dev` 环境对两条真实帧实际执行（不是代码阅读推断）：run1/frame0 与 teacher 已知最坏图帧 run3/frame343。两帧 `all_checks_passed=true`，12 项布尔检查全部通过，`hop1_teacher_only`/`hop1_funnel_only`/`shift_mismatches` 均为空集：run1/frame0 的 funnel 与 teacher 的 ligand→hop1 边集合（219 个环境原子，funnel 1206 条边 vs teacher 974 节点/39858 边闭包的对应子集）逐对（含周期 unit shift）完全一致，10 个离 5 Å 边界最近的真实原子（最小 gap 0.00702 Å）teacher/funnel 判定全部一致；run3/frame343 同样在 244 个环境原子（teacher 1066 节点/45656 边）上完全一致。边界平滑性：对每帧最靠近边界的真实原子做 ±0.5 Å 合成扫描（0.05 Å 步长），`discrete_included` 在 5.0 Å 处精确单次翻转，`quintic_c2_cutoff`（inner=4.0 Å, outer=5.0 Å）在边界处连续变化、`≥outer_cutoff` 处严格为 `0.0`、且存在真实的非 0/1 中间值（如 run1 中 4.55 Å 处 weight=0.40687），证实离散膜员翻转不会造成能量跳变。report_sha256：run1/frame0 `101e3364f0cebd91694b43bc3b93e239ace49f54a8b5cefbaa960cade411bc7a`，run3/frame343 `107416dad1bf09c6c371adece36a969143be21610af0ac8254b6f22ea05a7ae4`。另确认 `torch`/`openmm`/`openmmtorch`/`NNPOps` 在 `omm_torch_126` 环境可正常导入（该环境缺 mdtraj，不能跑真实帧，仅作为未来可能后端的补充证据，不参与本决策 go/no-go） | DEC-030(d0) 第 1 项 (d0-1) 标记完成，`IMPLEMENTATION_PLAN` 对应复选框勾选。供 (d0-2)–(d0-5)/(d1) 引用的设计事实：原子身份固定为 topology index，邻域成员逐步动态重算，无固定 manifest；triclinic PBC 沿用 `local_residual/geometry.py` 与 `local_residual/mace_graph.py` 共享的行向量最小像约定（`cartesian = fractional @ box`）；此 funnel 只服务于 DEC-037 冻结的轻量 student（typed embedding + 平滑 ligand-environment radial/contact + ≤2 interaction block + ligand-only pooling），不要求复现 teacher 完整两跳闭包的环境–环境边（S2 未验证也不需要）。本决策只证明设计可实现，不代表 student 有统计增益或生产资格，不得据此训练。下一步是 §d0-5 计算/部署预算冻结（最大参数量、最大 neighbor/edge 数、每步毫秒目标、显存上限、cutoff、seed/epoch、早停规则、held-out 判据），完成后才进入 (d1) 离线 student 拟合 |
| 2026-08-05 | DEC-039 | 冻结 (d0-5) 计算/部署预算中除 ms/step 生产基线以外的全部数字；正式退役 Arm A/B/D 为 `not_pursued`（§11A.12）；`protocols/EXP-012_preregistration.json` reseal 为 `sealed`；训练早停验证集设计改为训练 run 内部时间块切分，不复用被隔离的第三条 run 兼职早停+最终评估。**ms/step 生产基线本条暂标记为部分完成**：v1 报告（`062c63b2...`）已确认 `win_sys_xml_sha256_matches_manifest=false` 的根因是 `output_lrc_fix/box_vectors.npy` 陈旧（只写一次，从未反映 `pre_equilibrate`/`_rebalance_with_boresch` 后的真实盒子），`scripts/benchmark_exp012_no_student_window0_baseline.py` 已修复为两阶段构造（先用陈旧盒子建一次性 probe Context 去 `loadCheckpoint`，读回真实盒子，再用它重建真正计时用的 System），但修复后的重新测量尚未执行，v1 的 1.3961/1.3988 ms/step 数字仍不计入本条 DEC 的生产基线冻结 | 模型规模（≤50k/100k 参数）与图规模（S1≤256/320、边≤1536/2048、单原子 neighbor≤64/80）：1500 帧真实几何审计，`report_sha256 782e58242233d3b2153e719dda7685d08f0e65e61f1dc35f4d6d33c114cf416f`，未改动。CPU float64 vs CUDA float32 funnel 一致性：`report_sha256 9671470e03e12029ae503f1a6f7b5fd31e193d51fc4fd5ede385fa93cfcf934b`，`all_edge_sets_identical=true`。Arm A/B/D 退役：见 §11A.12，判定 `not_pursued`（非 `FAILED`——从未实现，无法说"跑输了"），显式记录预注册偏离（`decision.arm_C_increment_comparisons=["C_vs_A","C_vs_B"]` 从未执行，实际只做了 `C vs B=0` 对照）。训练预算：outer split 沿用 `hard_window0_run1/2/3` 三折 leave-one-run-out（DEC-034/035/037）；早停验证集改为每条训练 run 末尾 20%（连续时间块，非随机拆帧）划为早停验证子集，被隔离的第三条 run 只用于最终评估，不参与模型选择或早停——避免同一 run 兼职验证+测试的乐观偏差；`max_epoch=500`、`early_stop_patience=30`、`seeds_per_variant_per_fold=3`（direct-gap 与 distilled 各自独立满足）、监控量沿用 `bidirectional_gap_variance_loss`。ms/step 根因：`box_vectors.npy` 仅在 `runabfe.py:880`/`1117` 写一次；真正建窗口 0 用的盒子取自 `pipeline.box_vectors`，会在 `abfe_pipeline.py:1915`/`6294`/`6308`（预平衡 NPT 弛豫后）与 `abfe_pipeline.py:2195→runabfe.py:4420`（Boresch rebalance 后）被内存内重新赋值但从不写回磁盘；`output_lrc_fix/` 真实 mtime（`box_vectors.npy` 01:18 → `rebalance.chk` 08:27 → `manifest.json` 09:08）佐证两者相差近 8 小时。窗口 0 生产 System 无 `MonteCarloBarostat`（benchmark 报告自带的 `force_groups` 已确认），故盒子一旦建窗口即冻结，可直接从已加载的 `openmm.chk` 读回真实盒子，不需要重放具体走了哪条 pipeline 分支 | (d0-5) 复选框标记为**部分完成**：模型/图规模、CUDA funnel 一致性、Arm 退役、训练预算/早停设计均已冻结；ms/step 生产基线待用户按修复后的脚本重新measure、确认 `win_sys_xml_sha256_matches_manifest=true` 后，作为一次追加 DEC（或 DEC-039 的直接编辑）补入，在此之前 §7 WP-4C.3 (d0-5) 复选框不得整项打勾，(d1) 离线 student 拟合可以在数据集/模型代码层面开始编写，但其 D3 部署阶段的性能验收仍需要这份真实基线 |
| 2026-08-05 | DEC-040 | 接受 `student_d2_report_v4.json` 为 EXP-012 direct-gap student 的 D2 最终资格报告，D2 判定 `PASSED` | 3 个 leave-one-run-out folds × 3 seeds，共 9 个 checkpoint；每个 checkpoint 在 `hard_window0_run1/2/3` 的 frame 0 上检查，共 27 组。`all_checkpoints_passed=true`、状态 `COMPLETED_D2_CHECKS`；有限差分最大绝对/相对误差 `2.4711e-7`/`1.8242e-5`，低于 `1e-4`/`1e-2` 门；27/27 非参与原子零力；cutoff 跳变缩放比 `22.6405–24.9856`（连续期望 25）且被探测 pair 每组恰好翻转一次；27/27 的 0.3 Å 合成近接触能量和力有限。report SHA-256 `329a98331400f22fe13b76e00f435f4c3a83431441f33bc35af502540d56f08b` | 关闭 D2。**D3 不能直接开始**——用户已明确裁定：`win_sys_xml_sha256_matches_manifest=false` 这道 System 身份门与 D2 无关、不阻塞 D2，但必须在 D3 的 OpenMM 在线增量比较之前关闭，否则 student 与 no-student 基线可能不是同一个 System。`no_student_window0_baseline_v2.json`（report_sha256 `6969d3d0b5316fca6f605beb23416a125433d54f41a825f6df484b4f97e651aa`，median/P95 `1.3959/1.3968` ms/step）继续保留为 **provisional 性能参考**，不算最终通过。D2 之后的下一步是对两份 System XML 做结构化逐项 diff（masses、constraints、Force 类型/顺序、force groups、表达式、global/per-particle 参数、barostat、virtual sites），判定是序列化顺序/浮点文本噪声还是真实语义差异——前者改用 canonical System fingerprint 并登记替换理由，后者从 checkpoint/manifest 对应的真实生产构造快照重建后重新测 ms/step；只有这道门关闭，D3 才能正式开始。范围严格限于 direct-gap checkpoints 的离线坐标/autograd 检查；distilled checkpoints 已排除，未用 held-out run 做 checkpoint selection，未使用 TorchForce、未执行 NVT，因此 D3、D4、WP-5 和 production 仍未通过 |
| 2026-08-05 | DEC-041 | 冻结 D3-0 provenance gate（协议在执行前已固定，见上一条备注与用户确认）：`win_sys_xml_sha256_matches_manifest=false` 判定为 `CLOSED_STEP3_OPERATIONAL_SEMANTIC_PASS`——记录里没有可检测出的语义差异；历史 byte-level 不一致视为非阻塞，关闭。System 身份门解除，D3 中依赖真实生产 System 的部分（OpenMM 注入、student/no-student 正式耗时比较）现在可以开始 | `scripts/verify_exp012_win_sys_provenance_d3_0.py`，report SHA-256 `2dc557092ce327c8af3eb2d137c489817a0267377604b077e4469b4d54ba32a8`。Step 1（同进程两次独立重建，使用真实 `resolve_dispersion_protocol`/`resolve_membrane_protocol` 而非硬编码字符串）：两次重建字节级完全一致（确定性确认）。Step 2：即便换成真实解析函数（结果与此前硬编码值一致：`dispersion_protocol=legacy_uniform_density_lrc`、`environment_type=soluble`，均 `was_defaulted=true`），重建哈希仍与 `manifest.json` 记录值不一致——说明此前怀疑的"硬编码解析函数"并不是根因。Step 3（10 项独立字段核对，未重新实现 `build_ibs_dual_system` 本身作为"第二真值"）：`masses`/`constraints`/virtual sites 数量与 `system_native.xml` 一致；`lambdas`/温度为 `manifest.json` 字面值；`potential_type`/Boresch/softcore 参数为 `stage2_vanishing.json` 字面值；IBS prefix 为 `ibs_state` 字面值；box vectors 取自已加载 checkpoint（`loadCheckpoint` 无报错，本身是较强的结构兼容证据）；10/10 项全部通过，Force 数量/类型/group/表达式的 canonical fingerprint（10 个 Force：group 0 的 `CMMotionRemover`/`HarmonicAngleForce`/`HarmonicBondForce`/`NonbondedForce`/`PeriodicTorsionForce`，group 1 的 `CustomCVForce`，group 2 的 `CustomBondForce`/`CustomNonbondedForce`，group 3 的 `CustomCompoundBondForce`，group 4 的 `CustomNonbondedForce`）未见异常 | 结论按冻结措辞原样记录，不猜测具体是 attribute 顺序、Force 顺序还是浮点文本格式差异——这个具体机制从未被独立证实，不写入结论。`no_student_window0_baseline_v2.json`（report SHA-256 `6969d3d0b5316fca6f605beb23416a125433d54f41a825f6df484b4f97e651aa`，median/P95 `1.3959/1.3968` ms/step）现在可以采信为无 student 生产基线（不再是 provisional）。协议只跑一次：不因这次关闭就回头再挖"到底是什么格式差异"，也不再开第二轮 provenance 调查。D3 中不依赖真实生产 System 的部分（TorchScript 导出、纯 Torch 能量/力一致性、CPU/CUDA 对照）从一开始就未被此门阻塞，可以并行推进 |
| 2026-08-06 | DEC-042 | D3 进行中，不全部通过：sub-item 2（TorchForce/OpenMM Reference 注入）与 sub-item 3（端点归零）通过；sub-item 4（生产耗时）确认 all-pairs 近邻发现是 258% 开销主因，替换为 linked-cell list 后降到 95%，仍超过 (d0-5) 冻结的 ≤50% 淘汰门，**未关闭**；sub-item 1（deployment 一致性）残差从 1.4e-5 降到 1.7e-7，仍未通过严格同精度门，未关闭。D4 未开始，本轮工作在此暂停 | **sub-item 2/3**（`scripts/check_exp012_student_torchforce_openmm_d3.py`，report_sha256 `d71fe52e5b65696e29fb9777d91da612dd87fa3a75f212d4f66e5b0908c8bf88`）：`all_passed=true`，`torchforce_consistency`/`endpoint_zeroing` 误差均为 `0.000e+00`。**sub-item 1**（`scripts/check_exp012_student_deployment_d3.py`）：第一次真实执行（report_sha256 `f847b497d281d...`）用单一绝对容差把 CPU64-vs-CPU64、CPU64-vs-CPU32、CPU32-vs-CUDA32 混在一起判定，被指出是错误方法论；改为三类容差（同精度 correctness=1e-8、精度包络 precision_envelope=5e-4、设备一致性 device_consistency=1e-4），并定位并修复测试脚本自身的一个 precision-lineage bug（mdtraj 原生 float32 存储的坐标先在 numpy 里乘 10 再转 float64，与 deployable 模块先转 float64 再乘 10 的顺序不一致）：残差从 `1.415e-05` 降到 `1.691e-07`（约 83 倍）。审计两条独立前向实现逐行对照后，发现 `local_residual/student_deploy.py::_radial_basis` 用 `.pow(2)` 而 `local_residual/student.py::_radial_basis` 用 `.square()`——数学等价但是不同 ATen kernel，已改为 `.square()` 消除这个具体差异来源，但**此修复尚未经真实数据重新验证**（用户尚未重跑 sub-item 1）。**sub-item 4**：`scripts/profile_exp012_student_torchforce_overhead_d3.py`（新增，四段独立计时：graph construction / model forward-backward / TorchForce 同步 / OpenMM 步耗时）确认替换前 all-pairs 近邻发现（O(n_ligand×n_system)，41×~73,500 全对距离）单独耗时 7.68ms/call，是网络本身数学运算（0.28ms/call）的约 27 倍，主导 258% 开销（`no_student_median=1.3914`、`with_student_median=4.9851`，report_sha256 `c2b4026cf9dc511491c033ad3027c2a57a89975a6c0e210ad65a0c366ed8fe0f`）。据此在 `local_residual/student_deploy.py` 新增周期性 linked-cell list（`_cell_list_candidates`，box 对角且每轴 ≥3 bins 时启用；否则退化到原 all-pairs `_brute_force_candidates`，保证正确性不依赖盒子形状假设，只有速度依赖），两条路径共用同一个 `distance < outer_cutoff` 判定与同一个 `_minimum_image_displacement`。替换后重跑：开销从 258% 降到 **95%**（`no_student_median=1.3884`、`with_student_median=2.7077`，report_sha256 `02ecc15244eee8d0d0c68c3de8dcee15fa78c9710ac7e08cb93def9181d6abb1`），仍高于硬门 ≤50%（目标 ≤15%）。再次跑 profiling 脚本定位新瓶颈时，发现 profiling 脚本自身的 `_graph_construction` 是替换前逻辑的手写镜像副本，未随 `student_deploy.py` 一起更新，导致测的是已经废弃的旧代码而非真正在用的新 cell list 路径（stage_a 数字与替换前完全相同的 `7.68377` 就是证据）——已修复为直接调用 `deployable_device` 对象的真实方法（`_cell_list_candidates`/`_brute_force_candidates`），**修复后的 profiling 脚本尚未重新执行**。此外新增 `tests/test_exp012_student_deploy_cell_list.py`（纯 CPU、合成几何，验证 cell list 与 brute force 找到完全相同的候选边集/距离/前向+反向能量力/TorchScript 可导出性），**尚未运行** | D3 未关闭，不得进入 D4。下一步（未执行，按顺序列出，供下一轮继续）：① 跑 `tests/test_exp012_student_deploy_cell_list.py` 确认 cell list 与 brute force 等价；② 用 `.square()` 修复后重跑 sub-item 1，确认 CPU64-vs-CPU64 残差是否归零；③ 用修好的 profiling 脚本重新定位当前 95% 开销里的新瓶颈（cell list 构造本身 vs 网络前向反向 vs TorchForce 同步），再决定是否需要近一步优化（如先用 ligand 包围盒粗筛再排序，减少每步对全部 ~73,500 个环境原子排序/`searchsorted` 的开销）；④ 从头重跑完整 D3（sub-item 1/2/3/4）三份脚本，确认全部通过且开销 ≤50%（目标 ≤15%）后才能登记 D3 关闭、开始 D4 |
| 2026-08-07 | DEC-043 | 关闭 D3 全部剩余 correctness 项：cell-list 等价性单元测试通过；`.pow(2)`→`.square()` 假设被实测证伪（对残差无可测量影响）；`reference_eager_vs_deployable_eager` 残差 `1.69e-7` 裁定为 `PASSED_OPERATIONAL_NUMERICAL_EQUIVALENCE`（不是数值归零，理由见依据列）。D3 状态改为 `CLOSED`，D4 可以开始 | (1) `tests/test_exp012_student_deploy_cell_list.py`：首次运行 `14 passed, 1 failed`——`test_cell_list_matches_brute_force_full_forward_energy_and_force` 用"扰动 box 逼 `forward()` 走 fallback 分支"的写法比较两次 `forward()`，能量误差 `1.420e-07 > 1e-8`；诊断为测试方法混淆变量（扰动 box 的非对角元使 `_minimum_image_displacement` 对所有候选边引入 `O(扰动)` 的真实几何偏移，与 cell list 算法本身无关），不是 `_cell_list_candidates` 的 bug——因为隔离该性质的低层测试（`test_cell_list_matches_brute_force_edge_set_and_distances`，12 组参数化）本来就已经以 `<1e-10` 通过。修复：把 `local_residual/student_deploy.py::_DeployableStudent.forward()` 里 embedding→消息传递→readout→能量那段下游流水线抽成新方法 `_energy_from_edges(positions, edge_ligand_topology, edge_environment_topology, edge_distance)`，`forward()` 只负责选路径后调用它；测试改为在同一个 box、同一份 positions 上分别调用 `_cell_list_candidates`/`_brute_force_candidates`，把各自的 edges 喂给同一个 `_energy_from_edges` 比较，不再依赖 box 扰动强制分支选择。重跑后用户确认全部通过。(2) 核对发现 `student_d3_1_2_report_v3.json`（report_sha256 `c8d940818111f0150c754690e2f519904166587effdfbe60f7383c696b8ca148`，生成于 2026-08-06 20:46，比 v2 晚 23 分钟）已经用 `.square()` 修复后的代码真实跑过（用 `zipfile` 打开其导出的 `student_torchscript_d3_1_2_v3.pt`，确认内嵌序列化源码里 `_radial_basis` 用的是 `torch.square()`，v2 是 `torch.pow(x, 2)`），但此前从未写回 `EXPERIMENT_LOG`/计划文档，是孤立产物。v3 的 `reference_eager_vs_deployable_eager` 误差为 `1.6908728595e-07`，v2 是 `1.6908728728e-07`——两者在第 10 位有效数字才出现差异，`.pow(2)` vs `.square()` 假设因此被判定为**证伪**：在当前 torch 版本/CPU 后端上这两个算子对本输入没有可测量的数值差异，从来不是残差的真正来源，真正根因未被查明也不再追查（用户裁定不值得继续考古）。裁定 `PASSED_OPERATIONAL_NUMERICAL_EQUIVALENCE` 的数值依据（均取自 v3 report）：`reference_eager` vs `deployable_eager` 绝对误差 `1.69e-7`（相对误差 `~4.7e-8`，基于 `basis_reduced≈-3.587`）、力最大绝对误差 `9.45e-8`；`deployable_eager` vs `deployable_scripted_cpu_float64` 严格 `0.0`；`deployable_scripted_cpu_float64` vs `_cpu_float32` 能量误差 `1.32e-5`；`_cpu_float32` vs `_cuda_float32` 能量误差 `7.65e-7`、力误差 `1.24e-6`；sub-item 2/3（TorchForce/OpenMM 注入、端点归零）此前已严格 `0.000e+00`（DEC-042）。即 reference-vs-deployable 残差比已被接受的 CPU64→CPU32 精度包络小约 2 个数量级、比已被接受的 CPU32↔CUDA32 设备误差还小，不会成为实际 float32 生产部署里可辨识的额外误差来源 | `correctness_cpu64_vs_cpu64=1e-8` 这个绝对阈值是 `scripts/check_exp012_student_deployment_d3.py` 自定的工程参数，不在 `protocols/EXP-012_preregistration.json` 的 sealed 条款里，也没有独立的物理尺度依据（类比 §16 已经对 `≤50%`/`≤15%` overhead 数字做过的同类澄清：工程目标不等于 sealed correctness gate）；比它实际部署会经历的 float32/CUDA 误差还严两个数量级，继续拿它卡住项目不代表更严谨的物理正确性，只是在用一个未经验证尺度的阈值。sub-item 4 的 `~95%` overhead 仍如实保留、不删除，按 §16 规则不再是 D4 前置门，最终判据是 production 独立重复里的 ESS/GPU-hour。D3 四个 sub-item 现状：1=`PASSED_OPERATIONAL_NUMERICAL_EQUIVALENCE`（本条新增）、2=`PASSED`（DEC-042）、3=`PASSED`（DEC-042）、4-correctness=`PASSED`（cell-list 等价单元测试，本条新增）、4-performance=非阻塞工程目标（§16，`~95%` overhead 如实保留）。D3 整体状态：`CLOSED`。下一步进入 D4（短 NVT 动力学资格），此前从未执行 |
| 2026-08-07 | DEC-044 | D4（短 NVT 动力学资格）首次执行即通过：`all_passed=true`，3 个独立种子重复、全程有限值、student 力/温度均在 sanity 门内。EXP-012 至此 D0-D4 全部关闭，下一步是 WP-5A（Atenolol 困难窗口体系内三重复资格） | 新增 `scripts/check_exp012_student_d4_short_nvt.py`：复用 DEC-039/041 checkpoint-derived-box 构造与 DEC-037 D3 sub-item 4 的“student TorchForce 挂到真实 `hard_window0` win_sys 独立 force group”注入方式，首次让 student 实际参与积分（此前 D0-D3 全部只做单帧静态评估）。方法：3 个独立重复 = 同一个真实生产 checkpoint、3 个不同 `LangevinMiddleIntegrator` 种子（`setRandomNumberSeed()` 刻意放在 `loadCheckpoint()` **之后**调用，防止 checkpoint 可能内嵌的 RNG 状态把种子选择悄悄覆盖、导致三次“独立重复”其实是同一条轨迹）；每个重复成对跑 no-student/with-student（同一种子），隔离“加 Force 的影响”与“噪声实现的影响”。500 步 warmup（丢弃）+ 2000 步监控（每 100 步记录一次），共 3 repeats × 2 configs × 2500 步。report_sha256 `f06ad7b03ce85ab4ee443fab20e259124e3ea2bc7e41777b0a586f9648783554`：`student_max_force_norm_observed_kj_mol_nm=39.448`（3 个重复范围 `10.0–39.4`，均远低于 `500` 的 sanity 阈值）；`student_energy_kj_mol` 范围 `-7.52~+0.31`，与 `a_k=0.5` 时理论上界 `±a_k·kT·b_max_reduced≈±12.47 kJ/mol` 一致；两种配置温度全程 `298.3–302.5K`（目标 `300K`，sanity 门 `150–600K`）；全系统最大力 `~4300–6300 kJ/mol/nm`——这是显式溶剂里正常的键伸缩背景力量级，与 student 无关，safety 阈值只套在 student 自己隔离出来的力贡献上，不会被这个正常背景值污染；`all_finite=true`，全程无 OpenMM 异常 | `--max-safe-force-norm-kj-mol-nm=500`/`--temperature-sanity-factor=2.0` 是脚本自定的工程 sanity 门（取自 PLAN 文档 §6 示例配置默认值），不是 sealed 数值门——D4 从未被预注册过（只有 D1-D3 sub-item 在 `protocols/EXP-012_preregistration.json` 里有 sealed 定义），这次实测结果远低于阈值本身就说明门的松紧不是判定的关键。D4 关闭，EXP-012 D0/D1/D2/D3/D4 全部通过；仍未做的是（a）真正的 per-window/per-state `A_k` 生产 wiring（`a_k=0.5` 只是这次和 D3 一样的冻结 smoke 常数）、（b）WP-5A 的 ESS/GPU-hour/mutual overlap 独立重复资格——D4 只回答了“加这个 Force 会不会让积分炸掉”，不回答“加了它采样有没有变好”。下一步是 WP-5A |
| 2026-08-07 | DEC-045 | 正式登记 D3/D4 收尾证据（模型 SHA、TorchScript SHA、运行环境）并冻结 EXP-012 唯一 production 候选为 `hard_window0_run1__direct_gap__seed0.pt`；不根据下游 production 结果重选 seed，暂不重新训练 final model | **候选身份**：`output/outer_lambda_exp012/student_checkpoints/hard_window0_run1__direct_gap__seed0.pt`（checkpoint SHA-256 `61abcd1f0d0ff809914003de522f05db66f9dc4b341391bfa0b7f1cb99e6f2e3`，79783 字节，`variant=direct_gap`，`held_out_run_id=hard_window0_run1`，`seed=0`）——D2（DEC-040，report SHA `329a98331400f22fe13b76e00f435f4c3a83431441f33bc35af502540d56f08b`）、D3 sub-item 1/2/3（DEC-042/043，`student_d3_1_2_report_v3.json` SHA `c8d940818111f0150c754690e2f519904166587effdfbe60f7383c696b8ca148`）、D4（DEC-044，`student_d4_short_nvt_report.json` SHA `f06ad7b03ce85ab4ee443fab20e259124e3ea2bc7e41777b0a586f9648783554`）**全部用的同一个 checkpoint**，此前从未换过候选，因此这里是登记既成事实，不是新选择。**D4 运行环境**：`torch 2.12.0`（CUDA 12.9）、`openmm 8.5.2.dev-36a30cb`、GPU `NVIDIA GeForce RTX 2080 Ti`（driver `580.173.02`，11264 MiB），conda 环境 `openmm_dev`，platform `CUDA`。**D4 TorchScript 产物**：`output/outer_lambda_exp012/student_torchscript_d4.pt`，SHA-256 `e576a99de109df9df77507b5ddd42aae76e720c41ab5353625676fa42584c143`——**明确标注不可复用于生产 wiring**：这份导出用 `a_k=0.5` 把标量系数直接烤进了模块输出（`build_deployable_student_module(..., a_k=0.5)`），是 D3/D4 单点/短程 smoke 的约定；真正的多态 IBS wiring 需要 `OuterLambdaController.state_coefficients`/`coefficient_matrix` 按每个态的 `A_k=w(λ_k)·c_m` 在 `OuterLambdaIBSBiasForce` 的 CV 表达式里逐态相乘，所以喂给它的 basis Force 必须只输出未缩放的原始模型能量（`a_k=1.0` 或等价 passthrough 约定），否则会把系数重复乘两次——WP-5A 的 IBS/TMBAR 接线 smoke 需要重新导出一份 `a_k=1.0` 的 TorchScript，不得直接沿用这份 D4 产物 | 候选冻结依据：D2/D3/D4 从建立到通过全程只使用这一个 checkpoint，没有做过任何跨 checkpoint 的比较挑选，所以"冻结"是显式记录当前唯一候选、承诺后续 WP-5A 不因为 production 结果不理想就切换到 `run2`/`run3`/`seed1`/`seed2` 的其他 8 个已训练 checkpoint（会构成事后挑选，污染 WP-5A 的统计意义），也不重新训练——重新训练需要重新走 D1→D4 整条资格链，与"先做 5A 看这条候选到底有没有用"的当前目标冲突。EXP-012 D0-D4 全部关闭 |
| 2026-08-07 | DEC-046 | 修正 `scripts/check_exp012_ibs_tmbar_wiring_smoke.py` 首次运行暴露的两个方法论问题后重跑：把 CUDA mixed-precision Group-1 能量与 float64 numpy 独立重算之间的比较从统一的 `1e-6` 绝对门拆成两档（同状态复读用严格 `1e-8` 绝对门；跨精度边界的 Group-1 log-sum-exp 交叉验证改用相对容差 `1e-4`）；把 `win_sys_xml_sha256` 与 `manifest.json` 的原始字节比较从 `all_passed` 里移除，改为核对 DEC-041 已 sealed 的 provenance report verdict | 首次运行（未记录 report_sha256，用户诊断后未采信为正式结果）：`ledger 20/20 闭合`；`target composition 最大误差 1.42e-14`（同一 Context 状态复读两次，接线组合逻辑严格正确）；`student contribution 非零`；`endpoint A_0=0` 精确成立；全部有限。表面失败的两项都是检查脚本自己的方法论问题，不是接线本身的问题：（1）Group-1 CUDA mixed-precision 能量（约 `-118~-151 kJ/mol`）与独立 numpy float64 重算比较，最大绝对误差 `1.657e-3 kJ/mol`、最大相对误差 `~1.29e-5`，量级完全符合 mixed precision，但脚本原来拿统一的 `1e-6` 绝对门卡这两个跨精度的量，必然不过；（2）`win_sys_xml_sha256_matches_manifest=False` 被脚本当时直接写进 `all_passed`，但 DEC-041 已经把这个原始字节比较问题正式关闭为 `CLOSED_STEP3_OPERATIONAL_SEMANTIC_PASS`（10 项独立结构字段+Force canonical fingerprint 全部一致，历史 byte-level 不一致视为非阻塞），继续拿它卡 `all_passed` 是在用一个已经关闭的问题重新判负。修正：新增 `--target-composition-tolerance-kj-mol`（默认 `1e-8`，严格档，两侧读的是同一 Context 状态）与 `--group1-relative-tolerance`（默认 `1e-4`，相对档，跨 CUDA-mixed/float64 精度边界）+`--group1-absolute-floor-kj-mol`（默认 `1e-3`，量级很小时的兜底）取代原来统一的 `--cross-check-tolerance-kj-mol`；新增 `--provenance-report`（默认指向 `output/outer_lambda_exp012/d3_0_provenance_gate_report.json`），读取其 `verdict` 字段，只有落在 `{CLOSED_STEP3_OPERATIONAL_SEMANTIC_PASS}` 才算通过，原始字节哈希比较改为只记录不参与 `all_passed`。报告新增 `platform.precision` 字段（从 `_build_platform_properties` 直接读出，CUDA 恒为 `"mixed"`），如实记录本次比较跨越的精度边界。核查发现仓库里唯一的 `simtk.openmm` 引用（`tests/test_lrc_interaction_group_compat.py`）本身已经是"优先 `openmm`、只有 `ImportError` 才退回 `simtk`"的防御性写法，在 `openmm 8.5.2` 下永远不会触发那个分支，不是产生用户看到的那条 deprecation warning的来源——warning 应该来自某个依赖内部，不在本仓库代码路径上，不需要改 | 修正后的脚本尚未重新执行；重跑通过后视为 WP-5A step 1（IBS/TMBAR 接线 smoke）正式关闭，随即启动 WP-5A step 3 的 baseline/student 三组独立重复 |
| 2026-08-07 | DEC-047 | 关闭 WP-5A step 1（IBS/TMBAR 接线 smoke）：`scripts/check_exp012_ibs_tmbar_wiring_smoke.py` 修正版重跑，`all_passed=true`。student 真正进入了驱动采样的 K 态判别式，接线力学正确性确认；下一步是 step 3 的 baseline/student 各至少 3 组配对独立重复 | report_sha256 `7d8f7ab3d4f98c950be589bbf7020ac1d94c2a1698761b6cdad2c631c97b9e06`：`all_ledger_closed=true`、`all_finite=true`、`platform.precision=mixed`；`group1_cross_check_passed=true`（CUDA-mixed vs 独立 numpy float64 log-sum-exp 重算，`max_abs_err=4.178e-04`、`max_rel_err=3.218e-06`，量级符合 mixed precision，用相对容差 `1e-4` 判定，不再用统一绝对门）；`target_composition_passed=true`（同 Context 状态复读两次重算 target=original+LRC+neural_path，`max_err=2.842e-14`，严格 `1e-8` 绝对门内，证明接线组合逻辑本身完全正确）；`endpoint_A0=0.000e+00`（window 0 的 k=0 全局端点严格归零）；`provenance_verdict='CLOSED_STEP3_OPERATIONAL_SEMANTIC_PASS'`（`accepted=true`，采信 DEC-041 已 sealed 的判决，不重新拿原始 XML 字节哈希卡门）；`raw_byte_hash_match=false` 如实记录但不参与 `all_passed`（DEC-046 的修正生效）。20/20 帧全部通过 | WP-5A step 1 正式 `CLOSED`。下一步（step 3）：冻结 production 候选——当前 direct-gap checkpoint（SHA `61abcd1f0d0ff809914003de522f05db66f9dc4b341391bfa0b7f1cb99e6f2e3`）、wiring smoke 用的 `a_k=1.0` TorchScript、`c1=0.5`、`A_k` 系数矩阵、cutoff 和运行配置全部不再因为下游结果调整；运行 baseline MM 与 MM+student 两臂，各至少 3 个配对独立 seed，相同步数/初态生成规则，记录真实 GPU 时间；比较 mutual overlap、importance/absolute ESS、round trip/自相关、稳定性、ΔG 一致性、ESS/GPU-hour（计入约 1.95× 单步成本）；只有 ESS/GPU-hour 改善才进入 WP-5B |
| 2026-08-07 | DEC-048 | WP-5A step 3 pilot（baseline MM vs MM+student，3 组配对重抽独立重复）执行完成，`pilot_promotion_verdict=false`——不是"student 无效"，而是"student 确实改善了原始 mixture ESS，但当前部署成本没有被补偿"；不进入 WP-5B，暂停在此等待下一步决策 | `scripts/run_exp012_wp5a_pilot_baseline_vs_student.py`，report_sha256 `979934bf3f6a905b2725d120acecd0895cda6b793e1a8f19ea002be6eac1c391`。三次配对重抽（`velocity_draw_matches=true`，速度注入修复后确认逐位一致）逐重复原始数据：`mixture_ess_proxy`（min-over-states，不是字面 pymbar overlap）baseline→student 分别为 `47.07→52.10`（repeat0，+10.7%）、`47.13→52.11`（repeat1，+10.6%）、`37.95→48.50`（repeat2，+27.8%）——**3/3 全部提升**，说明 student 项对 IBS 混合覆盖度是有真实正向信号的，不是噪声。但同时 `gpu_hours` 从 baseline 到 student 分别涨了 `1.83×`/`1.89×`/`1.81×`（与 DEC-042/043 已测的 student TorchForce ~95% 额外开销/约 1.95× 单步成本量级一致），ESS 增益（10-28%）远小于成本涨幅（~85-90%），故 `mixture_ess_proxy_per_gpu_hour` 三个重复全部下降（`1736.69→1052.47`、`1791.85→1047.79`、`1378.45→972.22`），`median_improvement=-684.2177`，`n_repeats_improved=0/3`（需要 ≥2/3）。ΔG：repeat0/1 两臂 `converged=false`（100 帧 pilot 量级下 `solve_stage_integrated` 默认去相关门槛未达标，大概率是样本量不足而非物理不一致）；repeat2 两臂 `converged=true`，`delta_g_z=2.268`（门槛 2.0，刚超）、原始差值 `2.67 kJ/mol`（≈0.64 kcal/mol）。`dominant_component_switch_count` student 普遍更低（75→70、87→60、88→68），`endpoint_proxy_traversals` 方向不一致（10→13、16→8、29→8），`consistently_worse_mixing=false`（未一致变差）。`all_ledger_closed=true`、`all_finite=true`。四条晋级判据：①≥2/3 改善——不满足（0/3）；②中位数改善为正——不满足（-684.22）；③proxy 不一致恶化——满足（未一致恶化，但已不影响整体判定）；④ledger/稳定性/ΔG 一致性——ΔG 一致性不满足（`all_delta_g_consistent=false`）。综合 `pilot_promotion_verdict=false` | 按计划 §16 已预先写明的规则："若约 1.95× 单步成本没有被足够的 ESS 增益补偿，才判该路线性能失败"——本次是这条规则第一次有真实数据可以对号，判定为性能失败，不是科学假设失败：student 修正本身确实改善了混合覆盖（3/3 一致），问题在当前 `student_deploy.py` 的推理开销（cell-list 版本仍 ~95% overhead，DEC-042/043 已如实记录、未强制优化到 ≤50%）太贵，没能把 ESS 增益换算成更划算的采样效率。按"每个子阶段失败即停，不得跳阶段推进"的项目规则，不进入 WP-5B；也不在没有新增明确决策的情况下就直接换 `c1` 系数重跑（DEC-045/047 冻结候选时已明确"不得看结果后调 coefficient"）。下一步需要用户决策：(a) 就此关闭 WP-5A 单基势路线，记录"科学信号真实存在但当前部署成本压制了收益"这一结论；(b) 授权一次新的、显式记录的决策去优化 student_deploy.py 的推理开销（把 ~95% 降到能被 10-28% ESS 增益覆盖的水平，需要远低于当前水平）；(c) 授权一次新的、显式记录的决策去尝试更大的 `c1`（如果更强耦合能进一步提升 ESS 而不显著增加崩溃/不稳定风险，改善幅度可能追上成本），但必须先说明这不是"看结果调参"而是新的一轮独立验证。ΔG 的两个未收敛 repeat 值得单独补测更长 pilot（更多帧）以排除样本量不足的解释，但这不改变 ESS/GPU-hour 已经给出的性能判决 |
| 2026-08-07 | DEC-049 | 正式授权一次**仅限部署实现**的性能救援（不是重新训练、不是改架构、不是调 `c1`/`A_k`/物理路径）：目标把 student 每步总耗时从当前 `1.81×~1.89×ратio baseline`（DEC-048 实测）降到约 `1.10×baseline`；成功标准同时要求物理输出等价（能量/力与救援前逐位或近逐位一致，沿用 D3 cell-list 等价性验证方法）。**未达到 `1.10×` 目标即正式关闭当前单基势在线（real-time TorchForce-during-MD）路线**，不再无限期尝试 | 触发依据：DEC-048 显示 student 项对 `mixture_ess_proxy` 有真实正向信号（3/3 重复提升 10.7%/10.6%/27.8%），但当前 `student_deploy.py` 推理开销（cell-list 版本仍 ~95%，DEC-042/043 记录）把这个增益完全吃掉，导致 ESS/GPU-hour 3/3 下降。`1.10×` 这个数字选取依据：三个重复的 ESS 增益下限是 10.6%，`1.10×` 成本涨幅正好卡在增益下限附近、留一点安全边际，确保救援成功后 ESS/GPU-hour 有真实、非边际的净改善，不是刚好卡着零改善 | 下一步（未执行，按顺序）：① 重跑 DEC-042 已修复但从未真正执行过的 `scripts/profile_exp012_student_torchforce_overhead_d3.py`，定位当前 ~95% 开销的真实来源（cell-list 构造本身 vs 网络前向反向 vs TorchForce/OpenMM 同步），不得在没有这份数据的情况下直接猜测优化点；② 根据定位结果实施针对性优化（可能方向包括进一步减少候选边发现开销、batch/精度调整、TorchScript 编译优化等，具体待①的数据确定）；③ 优化后重跑 cell-list 等价性单元测试（`tests/test_exp012_student_deploy_cell_list.py`）与真实数据 deployment 一致性检查确认物理等价性未被破坏；④ 重跑 D3 sub-item 4 耗时脚本确认达到 `1.10×` 目标；⑤ 通过后重跑本次 WP-5A pilot（`scripts/run_exp012_wp5a_pilot_baseline_vs_student.py`）确认 ESS/GPU-hour 净改善；未通过 ④ 则登记单基势在线路线正式关闭 |
| 2026-08-07 | DEC-050 | DEC-049 终局实验判决 `TARGET_UNREACHABLE_CLOSE_ONLINE_PATH`——按预先冻结的规则，正式关闭当前 EXP-012 单基势**在线**部署路线（real-time TorchForce-during-MD）。不是边际未达标，是结构性不可达：即使假设动态建图成本能优化到零，网络自身前向+反向的固有成本也远超预算 | `scripts/measure_exp012_student_matched_path_lower_bound.py`，report_sha256 `98496b12fa9ca61f74415d732478c4011f1361cecf268d3c761e61036104d9e1`（CUDA `Precision=mixed`；换了 GPU 设备，绝对数字与此前 profiling session 不完全一致，但判决只依赖本次同一 Context 内的相对差，不受影响）。4 个变体同一 win_sys/同一调用路径/同一计时方法测得（median ms/step，5 repeat）：`baseline=1.4765`、`zero_output=1.6683`（纯 TorchForce/OpenMM 桥接调用成本 `+0.1918`）、`network_only=2.7042`（固定边集+真实网络 forward/backward，`+1.2277`）、`full=3.5678`（当前真实部署，动态 cell-list 建图+网络，`+2.0913`）。目标 `1.10×baseline`→预算 `0.14765ms`；`network_only_delta=1.2277ms` 超预算 `8.3×`。分解全部开销（`2.0913ms`）：桥接 `0.1918ms`（9.2%）、网络自身 forward+backward `1.0359ms`（49.5%，`network_only-zero_output`）、动态建图 `0.8636ms`（41.3%，`full-network_only`）——**即便把建图成本优化到零**，剩下的 `baseline+桥接+网络=2.7042ms`（即 `network_only` 本身）仍是 `1.83×baseline`，离 `1.10×` 差距巨大，不是"再优化一点点就够"的量级 | DEC-049 规则执行：`network_only_delta(1.2277) > budget(0.14765)` → `dec049_verdict=TARGET_UNREACHABLE_CLOSE_ONLINE_PATH`。按此**正式关闭**当前单基势在线部署路线：`student_deploy.py`/`OuterLambdaIBSBiasForce`/`IBSSamplerNeuralPathAdapter` 这条"每个 MD 步都调用一次 TorchForce"的接线方式，在不改模型权重/架构的前提下，无法把 ESS/GPU-hour 做到正收益（DEC-048 已经measure出 ESS 增益 10-28%，本条确认部署成本结构性地补不上）。**关闭范围明确限定**：只关闭"在线/real-time-during-dynamics"这一种部署模式；不代表关闭"离线/post-hoc reweighting"式应用（即只在分析阶段而非每个 MD 步调用网络的替代思路）——那是完全不同的研究方向，需要独立设计，本决策不隐含批准或否定它。WP-4C/WP-5A 的 D0-D4+wiring smoke+pilot 全部结果（模型本身有真实统计信号、接线力学正确、checkpoint/resume/ledger 闭合）依然作为有效工程/科学证据保留，不因这次关闭而被推翻或删除——关闭的是"当前这个部署实现能不能在这个项目里产生净收益"，不是"student 模型本身是不是垂圾"。不进入 WP-5B/WP-6/WP-7/WP-8（全部阻塞于此）。DEC-045/047/048/049/050 五条决策合起来构成 EXP-012 单基势在线路线从冻结候选到最终关闭的完整证据链，任何后续会话若想重开这条路线，必须先解释这条证据链哪里不成立，而不是直接重跑 |
| 2026-08-07 | DEC-051 | 用户提出 EXP-013（低频在线 student，MTS/rRESPA）作为 DEC-050 关闭在线路线后的下一步，核实技术细节后写入 `IMPLEMENTATION_PLAN` §WP-4D。**规划决策，尚未执行任何代码或实验** | 核实结论：①EXP-013 不是重开 EXP-009——EXP-009 失败的是 openmmml 1.6 完整 MACE + `openmm.PythonForce` 后端在 `MTSLangevinIntegrator` force-group 内核 `N=1` 触发 `CUDA_ERROR_INVALID_HANDLE`（历史代码 `outer_lambda_neural_basis.py:2779-2789`，预注册转向 `start_exp010_cheap_cv_due_to_backend`）；当前 `LocalResidualStudent` 走 openmm-torch `TorchForce` 后端，是完全独立通道，从未在 MTS 下测试过。②理想成本下界算术核对无误：`Δt/t0=1.4165`（用 DEC-050 report `98496b12...` 的 `t0=1.4765`/`Δt=2.0913`），`t(N)≈t0+Δt/N` 给出 N=4/8/12/16/24/32 → 1.354×/1.177×/1.118×/1.089×/1.059×/1.044×，与用户表格逐位一致。③**发现并登记一个必须先解决的架构约束**：查到 EXP-009 的历史 MTS 实现（`outer_lambda_neural_basis.py:2761-2796`）确认 OpenMM 的 `MTSLangevinIntegrator` 在 force-GROUP 粒度工作，`groups=[(慢group,1),(快group,N)]`，组间能量**线性加和**（EXP-009 precedent：全部经典力塞进 group 0，MACE 独占 group 31，线性相加）。但 DEC-048 实测出正向信号的设计（`OuterLambdaIBSBiasForce`，wiring smoke 起）是把 student basis 值融合进 IBS log-sum-exp 判别式内部——一个**非线性**函数，不能拆成"经典部分 log-sum-exp + 神经部分 log-sum-exp"再线性相加。因此把 student 放进独立慢 force group（MTS 唯一支持的方式）在结构上做不到"只让 student 变慢、判别式其余部分保持融合"，只有两个选项：(1) 整个 Group 1（含经典 softcore 项）一起变慢（经济上更有利但物理改动更大，未验证）；(2) student 拆成独立线性加和的额外 force group（更接近 MTS 标准用法，但是一个跟 DEC-048 验证过的融合设计不同的新设计，其 ESS 表现从未单独测过）。已把这个约束写进 013-A 的强制前置步骤：必须先选定方案 1/2 并登记理由，若选方案 2 还必须先做一次简单 N=1 ESS 对照，确认新设计本身有没有 DEC-048 那种正向信号，再谈降频——否则会重复"把部署问题和模型/设计问题混在一起"的错误 | 已写入 `IMPLEMENTATION_PLAN_outer_lambda_neural_basis.md`：新增 `## WP-4D / EXP-013` 完整小节（背景、算术表、架构约束、013-A/B/C 三步、go/no-go、EXP-014 原生压缩 contingency、EXP-015 post-hoc 排序理由）；更新 §1.1 状态表新增 WP-4D 行、WP-5 行标注"条件性阻塞于 EXP-013"；更新 §14 完成定义，下一执行点改为 WP-4D/EXP-013（不再是"无下一执行点"）。**尚未执行 013-A 或任何后续代码**，等待进一步指示 |
| 2026-08-07 | DEC-052 | 用户改进 DEC-051 的 EXP-013 设计：不在方案①/②之间硬选，加入更干净的方案③（exact residual split，`ΔV_θ=V_*-V_0`，MTS 分组为 `V_0(快)+ΔV_θ(慢)`），并把决策顺序固定为 ③→①→②。013-A 的首要任务改为数值等价性 + 单次调用成本测量，不是先测 ESS | 核实要点：①`V_0+ΔV_θ≡V_*` 是构造性代数恒等式（`ΔV_θ:=V_*-V_0` 定义即保证），`N=1` 时 Hamiltonian 与 DEC-048 融合设计严格相同，不需要为③重新证明 ESS 信号，只需验证 `ΔV_θ` 这个 `CustomCVForce` 实现是否正确（E/F 数值等价，D3 方法学，不是新物理有效性检查）。②确认 `MTSLangevinIntegrator` 的语义：`dt_outer` 是最外层步长，`N`（即 EXP-009 代码里的 `mts_ratio`）是一个 outer step 内快 group 被计算的次数，慢 group 只算 1 次（`outer_lambda_neural_basis.py:2796`：`outer_timestep_fs = mts_ratio * inner_dt`）——与用户描述一致。③方案③唯一真正的风险是计算重复：`CustomCVForce` 各自有独立 inner Context，`ΔV_θ` 的慢 Force 必须在自己内部重新算一份经典 `cv_k_int`/`cv_k_rest`，不能免费复用快 group 已算出的值，这部分开销是真实成本，必须先测。用 DEC-050 的 `t0=1.4765ms`、目标 `1.10×` 反推出 `ΔV_θ` 单次调用的预算上限：`N=8→1.181ms`、`N=16→2.362ms`、`N=32→4.725ms`；已知 `full-baseline` 增量 `2.0913ms`（DEC-050），`ΔV_θ` 因为多算一份经典 CV 预期成本 `≳2.09ms`——`N=16` 非常紧，`N=32` 有明显余量但 outer interval 更长、动力学资格门槛更严，构成一组干净的 go/no-go | 已改写 `IMPLEMENTATION_PLAN` §WP-4D：重写"关键架构约束"段落为③→①→②三候选（附方案③的完整数学定义、方案②"已经是新 Hamiltonian"的构象依赖系数论证）；013-A 改为"exact-residual 数值等价性 + 单次调用成本"（附预算表），不再是"选①还是②"的二选一框架；013-B/C 未改动（仍适用于③③存活或退到①之后的候选）。**仍未执行任何 EXP-013 代码**，下一步是实现 `ΔV_θ` 的 `CustomCVForce` 构造与 013-A 的两项测量 |
| 2026-08-07 | DEC-053 | EXP-013 013-A（方案③ exact residual split）双检查全部通过：`ΔV_θ` 数值等价性成立，N=16/N=32 成本可行，N=8 不可行。方案③在工程/正确性层面站住，进入 013-B 物理/动力学资格 | `scripts/check_exp013_residual_split_equivalence_and_cost.py`，report_sha256 `bc9eb24dcb5d54297664028b2207156efff83b6693d8569e2f3c76d0bcc45519`。等价性：`equivalence_passed=true`（相对容差 `1e-3` 内，`V_0+ΔV_θ` 与 `V_*` 在同一真实生产帧上能量/力一致）。成本：`baseline_median=1.3822ms`（注：与 DEC-050 的 `1.4765ms` 不同，同类设备/session 间正常波动，不影响相对判决）；`ΔV_θ` 单次调用真实增量 `delta_v_delta=1.8822ms`（含冗余经典 CV 重算+student TorchForce+双重 log-sum-exp，用 DEC-050 同款 matched-path 方法测得，不是假设值）。按 `1.10×` 目标反推预算：`N=8→1.1058ms`（不可行，`1.8822>1.1058`）、`N=16→2.2115ms`（可行，余量约 15%，比预注册时担心的"非常紧"稍宽松）、`N=32→4.4230ms`（可行，余量更大） | 方案③通过 013-A，不需要转方案①。下一步 013-B：短程验证 `N=1,8,16,32`（重点 `N=16/32`，`N=8` 已知成本不可行仅作物理诊断参考），比较能量分布、力范数/尾部、温度、结构合理性、相对 `N=1` 参考的能量/构象分布偏移，如可行加 shadow-work/积分误差代理。013-B 首次需要真实构造并跑 `MTSLangevinIntegrator`（此前全程未测试过这个积分器与当前 TorchForce+CustomCVForce 嵌套的组合，需要留意 EXP-009 式 backend 报错风险，出现即当场停止不重试）|
| 2026-08-07 | DEC-054 | 013-B 首次执行表面通过（`all_passed=true`, report_sha256 `99d74b17...`），但复核发现 N=1 参考态本身已从 300 K 崩溃到 ~0.003 K，四臂共病、相对门失效；6 步消除法定位真根因为**跨 integrator 类型迁移二进制 checkpoint**（`LangevinMiddleIntegrator` 写、`MTSLangevinIntegrator` 读）。旧报告判定 `INVALIDATED_BY_INITIALIZATION_BUG`；013-B 脚本已修复为 state-transfer 初始化 + 新增 DEC-054 绝对健康门；013-A 不受影响，仍 `PASSED`。013-B 待用修复后脚本重新执行 | 复核起因：`comparisons_vs_n1` 里 `temperature_k` 的 z-score 呈单调上升（N=8/16/32: 0.10/0.41/1.68），但原始快照显示 N=1 自己的 `temperature_k` 均值只有 `8.56e-4 K`（不是 300 K），`total_energy_drift_first_vs_second_half_kj_mol` 四个 N 均约 `-14,250 kJ/mol`，方向、量级一致——判定为共享的、N 无关的病灶，而非 MTS stride 引入的差异，现有相对门（只比较 N vs N=1）结构性看不见这类共病。用 `scripts/debug_exp013_mts_thermostat_diagnostic.py`（CHECK A-D）+`scripts/debug_exp013_mts_force_group_consistency.py`（CHECK 静态力核对）逐一排除：①force-group coverage（程序文本里 `f0..f5` 全部出现，`mts_groups` 构造参数与 `win_sys` 实际 force group 完全一致）——排除；②逐 group 查询力求和 vs 不限 group 总力（`getState(groups={g})` 加总 vs `getState()`，max diff `2.276e-4 kJ/mol/nm`，相对误差 `4.19e-8`，浮点噪声量级）——排除；③热浴系数（`a=0.996008,b=0.089264,kT=2.494339`，与 γ=2/ps、dt=2fs、T=300K 理论值精确吻合，直接在真正跑崩的 CHECK B/C integrator 对象上验证，不是另外的探测对象）——排除；④逐步计算程序结构（N=1 21 步、N=32 548 步，均为教科书式 BAOAB+RATTLE，O-step 每个真实内步只出现一次，无嵌套重复）——排除；⑤`ΔV_θ`/`CustomCVForce`/TorchForce 残差项（CHECK C 去掉它、只用原始 production System，照样在 64 fs 内崩掉 99.85% 动能，与含残差项的 CHECK B 几乎逐位相同）——排除；⑥**checkpoint 跨 integrator 迁移**（CHECK E：只在源 `LangevinMiddleIntegrator` Context 里 `loadCheckpoint()`，用公开 `State` API 取出 positions/velocities/box/parameters，显式 `setPositions/setVelocities/setPeriodicBoxVectors/setParameter` 灌入全新 `MTSLangevinIntegrator` N=1 Context，全程不对 MTS Context 调用 `loadCheckpoint()`）——**确诊**：同一 System、同一 MTS integrator、同一初始物理态，仅换初始化方式，`t=0` KE `187933.3958 kJ/mol`（与之前完全一致）之后全程维持 `296.5–302.4 K`（`t=6.4ps` 时 `299.49 K`），对照组（`loadCheckpoint()` 直接灌 MTS Context）同一时刻是 `0.003 K`。定性依据：无噪声纯摩擦衰减理论预测 `t=0.064ps` 时应剩余 `KE₀×e^{-4γt}≈145,000 kJ/mol`（`γ=2/ps`），实测崩溃版本只剩 `282 kJ/mol`，差 500 倍以上——比"丢失热浴 noise"猛烈得多，与"checkpoint 携带的 integrator/Context 内部状态与新 integrator 不兼容、首次 stepping 后触发灾难性 velocity-state 投影"的机制定性吻合，且 `t=0`（loadCheckpoint 后未跑步）positions/velocities 本身是对的（`T=302.4K`），问题在开始 stepping 之后才暴露 | ①`scripts/check_exp013_013b_mts_dynamics_qualification.py` 已修复：新增 `probe_state`（`getPositions/getVelocities/getParameters`）一次性从源 `LangevinMiddleIntegrator` Context 提取可迁移状态；`_make_mts_simulation` 内的 `simulation.loadCheckpoint(str(checkpoint_path))` 已删除，改为 `setPositions/setVelocities/setPeriodicBoxVectors`+逐参数 `setParameter`（`_system_has_global_parameter` 守卫），`lambda_boresch_scale`/`lambda_shield` 保留显式 manifest 覆盖作为兜底；全脚本（Phase 1 smoke 与 Phase 2 全部 N）唯一的 `loadCheckpoint()` 调用点现在只在 probe 阶段、只用 `LangevinMiddleIntegrator`。②新增 DEC-054 绝对健康门（独立于 N-vs-N=1 相对比较）：每个 N 的 `mean_temperature_k`、`warmup_end_temperature_k` 必须落在 `[270,330]K`（CLI 可调 `--min/max-mean-temperature-k`），`relative_energy_drift`（`|total_energy_drift|/mean_kinetic`）必须 `≤0.10`（CLI 可调 `--max-relative-energy-drift`）；`all_passed` 现在要求 `all_absolute_health_passed` 且 `relative_comparison_meaningful`（= N=1 自己的绝对健康门通过）且相对门都通过，缺一不可；report `schema_version` 升至 `v2`。③旧报告 `output/outer_lambda_exp012/exp013_013b_mts_dynamics_qualification_report.json`（sha `99d74b17...`）不删除、不覆盖，旁边新增 `.INVALIDATED.md` 说明文件记录失效原因和完整证据链，供后续 session 直接查阅不必重新调查。④013-A（sha `bc9eb24...`）未受影响，仍为 `PASSED`——它从未构造 `MTSLangevinIntegrator`，不经过这条 checkpoint 迁移路径 | 下一步：用修复后的脚本重新执行 013-B（`--warmup-macro-ticks 100 --monitored-macro-ticks 500`，仍是 6.4ps warmup + 32ps monitored，DEC-053 已定的默认值不变）。只有新报告 `all_passed=true`（意味着 N=1 绝对健康门先过，再看相对门）才能真正宣布 `EXP-013 013-B PASSED`，随后进入 013-C（WP-5A-mini 三重复）；013-A 的结论、成本预算、N=16/32 可行性判断全部保留，不需要重跑 |
| 2026-08-07 | DEC-055 | 用修复后脚本重新执行 013-B，report_sha256 `64a963626ef36893d440823bd9845ca7c6123cda1b76159e175d9a893810caf3`。绝对健康门**四臂全部通过**（DEC-054 修复确认生效），但预注册的相对系统性偏移门在 N=8/16/32 全部失败，z-score 远超 `3.0` 阈值。**这是真实物理结果，不是初始化 artifact；go/no-go 判断本身留待用户下次会话决定，本条只如实记录数据，不做裁决** | 绝对健康：`mean_temperature_k` 四个 N 依次 `298.4976/299.1596/299.5057/299.7847 K`，`warmup_end_temperature_k` 依次 `299.8595/299.3476/299.6013/300.3273 K`，均落在 `[270,330]K` 门内；`relative_energy_drift` 依次 `0.00198/0.00199/0.00081/0.00193`，远低于 `0.10` 门；`all_absolute_health_passed=true`，`relative_comparison_meaningful=true`——N=1 参考态这次是真实健康的 ~300K 轨迹，不是共病假阳性。相对门：`e_v0_plus_dv_kj_mol`（驱动 IBS 判别式的量）N=8/16/32 相对 N=1（`-141.0419±0.3703`）分别为 `-135.7811±0.3256`（z=10.67）、`-138.4829±0.3758`（z=4.85）、`-134.9807±0.3635`（z=11.68）——三者均显著偏离，但**不是随 N 单调**（N=16 反而比 N=8/32 更接近参考）；`temperature_k` 相对 N=1（`298.4976±0.0488`）分别为 `299.1596±0.0490`（z=9.57）、`299.5057±0.0489`（z=14.59）、`299.7847±0.0505`（z=18.32）——**随 N 单调升高**（+0.66/+1.01/+1.29 K），是干净的 dose-response，量级虽小（<1.3K）但统计上极显著，物理上与"`ΔV_θ` 依赖构象相关的态权重 \(p_k(\mathbf R)\)、这些权重随快原子运动快速涨落，降频评价它会引入 MTS 式共振/离散化升温"这一此前已预见的风险（IMPLEMENTATION_PLAN §013-A 讨论"`ΔV_θ` 自身是否足够 slow"）定性吻合，不是随机噪声或新的 bug 迹象 | **决策悬而未决，用户明确表示"明天再说"，本条不代替用户做 go/no-go 判断**。留给下次会话的具体选项：(a) 按 013-B 预注册门（`z_threshold=3.0`，"没有可分辨的系统性偏移"）直接判定方案③在 013-B 未通过，按 §WP-4D 固定的决策顺序（③→①→②，失败即下一候选、不回头重试）转向方案①（整个 Group 1 一起变慢）——但需先专门检查 IMPLEMENTATION_PLAN 已预警的风险：`inner_dt=2fs` 下 N=16/32 对应经典项 outer step 达 32fs/64fs，经典软核项在这个尺度上可能比 `ΔV_θ` 更早出问题；(b) 重新审视 013-B 判据本身（比如仿照 DEC-054 的做法，把纯统计显著性 z-score 和物理量级的绝对偏移阈值分开评估，因为这次温度偏移的绝对量级——1.3K——相当小，尽管 z-score 很大）；(c) 其它用户认为合适的路线。三个选项均未执行，等待下次会话明确指示 |

| 2026-08-09 | DEC-057 | EXP-016 temporal audit completed; physical crossing claim `UNAVAILABLE`; MM-ledger energy-weighted surrogate is exploratory only | Valid report `output/outer_lambda_exp016_loro/EXP-016_temporal_audit.json` (SHA-256 `d1c5d4de6a14b985acf6e2cafd42dab5a345cd8e12ea86ec45bf053aa43674c`); manifest input hash recorded in report as `db5cbf8e30b57353f9324cb2ea5653d11f05904d6ddddf888b778acf1a4667e7`; 3×500 continuous frames, `Δt_save=1 ps`, trajectory/ledger/latent alignment passed, held-out student checkpoints matched per run, circular block bootstrap 128 ps/2000 reps; initial run-1-checkpoint reuse report invalidated | No physical state/replica history, so no physical crossing or `τ_information`; target-derived student/gap diagnostics are not independent surrogate-event evidence; stop online TorchForce/MTS promotion |

## 4. 基础路径登记

### BASE-000：当前已知生产基线

| 字段 | 值 |
|---|---|
| 状态 | 历史 production 产物；已用独立 scratch 重建生成同窗口筛选轨迹 |
| mode | `ibs` |
| decoupling | `dual_lambda` |
| potential | `softcore` |
| DEXP params | `null` |
| 主要输出目录 | `output_lrc_fix` |
| 可否直接作为完整神经对照 | 否；scratch 轨迹可用于 CV/教师研发，但不是历史 production checkpoint 的严格续算 |

待补：

- [ ] 代码哈希；
- [ ] 完整运行命令；
- [x] Stage 2 λ 和困难窗口；
- [ ] 每窗口生产步数；
- [x] 每窗口 ESS；
- [ ] GPU 和软件版本；
- [ ] ns/day；
- [x] 目标慢自由度定义和三种子重复；
- [ ] 异常结构率；
- [ ] 独立重复。

## 5. 实验索引

| 实验编号 | 日期 | 工作包 | 简述 | 状态 | 结论 |
|---|---|---|---|---|---|
| EXP-000 | 2026-07-30 | WP-0 | 冻结基础路径、困难窗口并筛选首个慢变量 | PASSED | window 0 与 primary ligand torsion `[4591,4592,4593,4585]` 已由三种子证据冻结 |
| EXP-001 | 2026-07-30 | WP-1 | 外层 λ 控制器、端点归零和协议哈希测试 | PASSED | 独立协议 v1 的数学契约已实现 |
| EXP-002 | 2026-07-30 | WP-2 | 解析 mock 基势、力和 IBS target/bias/base 账本测试 | PASSED | 端点、同步 finite gate 和账本分离已闭合 |
| EXP-003 | 2026-07-30 | WP-3 | TorchForce + CustomCVForce 独立部署测试 | PASSED | 通用部署骨架已跑通；不代表真实模型 production Force 已完成 |
| EXP-004 | 2026-07-30 | WP-4 前置 | 真实 MACE 局部基势 1/10/50 帧离线评价 | PASSED | 真实模型可调用性验证成功；不是 WP-4 qualification |
| EXP-005 | 2026-07-30 | WP-4 标定 | 每步 MACE PythonForce 与复制 System 的 1/10/100-step NVT | PASSED | 每步真实 Force、PBC 和性能口径跑通；用于冻结 EXP-006，不是 qualification |
| EXP-006 | 2026-07-30 | WP-4 | 系数 0.10 的 1000-step MACE NVT qualification | FAILED | 5/6 门通过；路径最大力 258.949 超过 250 |
| EXP-007 | 2026-07-30 | WP-4 | 系数 0.09 的 1000-step MACE NVT qualification | PASSED | 六个硬门全部通过；随后仍需通过调度或蒸馏性能门 |
| EXP-008 | 待定 | WP-5 | 单困难窗口三组 IBS 对照 | PLANNED | 等待 EXP-012 feature、守恒力、NVT 与性能资格通过 |
| EXP-009 | 2026-07-30 | WP-3A/WP-4 | 冻结系数 0.09 的 MTS 调度资格 | FAILED | \(N=1\) 即出现 PythonForce/CUDA MTS 后端错误；没有可用于物理分布比较的 arm |
| EXP-010 | 2026-07-31 | WP-4A | MACE 教师到目标慢变量 cheap-CV bias 的蒸馏 | FAILED | GPU 教师数据集通过支持域门，但六个预注册 Fourier 候选均未通过跨 run 能量/广义力验证 |
| EXP-011 | 2026-08-02 | WP-4B | 完整 MM Hamiltonian 条件平均力/PMF 到周期 torsion bias | FAILED | AUG-001 后 mutual overlap 与去相关样本数仍未过冻结门；`FORMAL_RUN1_OVERLAP_FAILED`，不拟合 PMF |
| EXP-012 | 2026-08-03 | WP-4C | CV-free 通用局部残差路径势：A/B/C/D 表示 + 双向 gap-variance loss | PLANNED / C1_REAL_FRAME_CPU_PASSED | ledger/backend audit、OFF24 合成图合约及 Atenolol frame0 OMOL CPU latent/autograd smoke 已通过；下一门是同图 CUDA float32 对照，仍未训练 |
| EXP-016 | 2026-08-09 | P0 temporal audit | 三条连续 scratch trajectory 的 attribution/autocorrelation 与 surrogate-event LORO 审计 | INCONCLUSIVE / SURROGATE_ONLY | 1500 帧、`Δt_save=1 ps`、trajectory/ledger/latent 对齐通过；无 physical state/replica history，energy-weighted surrogate 有 114 个首 horizon adjacent switches；student/gap 结果 target-derived，不能作独立 event 证据；不晋级 learned slow information，不启动 online/MTS promotion |
| EXP-017 | 2026-08-10 | P0-A overlap-first | 冻结 v21 path、window-0 三 run ledger/trajectory/latent 的只读可行性审计；fixed-λ probe、lambda-only candidate 和 analytic-q 均未授权 | PLANNED | preregistration `protocols/EXP-017_preregistration.json` 已 seal（SHA-256 `db06d506d404c9a5e2cb9c61b221569071df62a760b81d0cc811d6ea8ea55a9f`）；plan SHA-256 `f045cd46d56d089011040fbb8b995b623421e6088a1c3e964fe60f0b91d651a1`；只允许 P0-A read-only audit，执行前保持无 MD/production authorization |
| EXP-017 | 2026-08-10 | P0-B gate | P0-A TMBAR/ESS 通过默认门，但仅 window-level；无低 ESS window、无 split-first edge、无 fixed-H asymmetric bottleneck 证据。window 5 split-half drift 为 `4.46×2σ`，登记为 temporal stationarity warning | STOPPED | `P0B_NO_LAMBDA_INSERTION_JUSTIFIED`；未运行 fixed-λ probe、未插入 λ、未进入 P1 |
| EXP-017 | 2026-08-10 | terminal registry | P0-B 终局登记：无可定位的相邻 λ edge；window 5 漂移仅保留为 stationarity/uncertainty warning | INCONCLUSIVE / STOPPED | `P0B_NO_LAMBDA_INSERTION_JUSTIFIED`; `terminal_reason=NO_LOCALIZED_LAMBDA_EDGE`; `P1_authorized=false`; `P2_authorized=false`; 正式总误差 `1.6461 kJ/mol`，未预注册 split-half floor 的敏感性值 `2.4766 kJ/mol` 仅作报告，不替换正式误差门 |
| EXP-018 | 2026-08-11 | terminal registry / stationarity confirmation | provenance v2 已通过；3 条独立 500-frame repeat 完成。离线分析仅 1/3 同时满足负漂移与 `z>=2`，方向性 window-5 漂移不构成可重复证据；repeat variance ratio=`16.7599`，支持 MBAR covariance uncertainty 被低估的预注册诊断，但不等于 MBAR 算法错误 | INCONCLUSIVE / CLOSED | final registry `output/outer_lambda_exp018_stationarity_confirmation_v2/EXP-018_final_registry.json` 自哈希 `cfa14520e2d7ec5375f5dbed6b7c9120947e658f743dde57ec6033be6ac891ef`；analysis report 自哈希 `904d00b6b1272f3a227900642e9856a47cc4ef5ce32bf9b30bdbae32161cb2f3`；summary 自哈希 `0438b69a6e121c220710a734c81c24ce6540bcea5a528fc38e84a87be9e62c67`；sampling aggregate SHA-256 `781f09bf12c3fc3fd886c015245a78cf1e45ebfda25531ce95ee760c77c7ba2f`；`directional_drift_reproduced=false`; `uncertainty_underestimation_signal=true`; `repeat_sigma_floor=0.375125 kJ/mol`; `additional_sampling_authorized=false`; 不拼接三条轨迹、不调 λ、不重开 neural/analytic-q；EXP-017 formal error 保持不变 |
| EXP-019 | 2026-08-11 | baseline reproducibility / uncertainty calibration | 先完成零成本 attribution：state 0 weighted-interaction variance 占 state-level descriptive variance `77.47%`；三条 repeat 的 state-0 importance ESS=`4.04/1.91/2.52` 帧、top-1% 权重质量=`0.665/0.845/0.809`；固定十个 block 无一达到 `0.5` dominant share，未发现单一 block 主导证据。随后冻结 complex/solvent 的 softcore + dual_lambda + IBS + v21 独立重复设计（每腿至少 3、论文目标 5），当前不启动 MD | ATTRIBUTION_COMPLETED / BASELINE_MD_NOT_STARTED | plan SHA-256 `7c1a28588c380097aad5d6876664997f6dc900fc9e3357dbcff333fdc5a71710`；preregistration SHA-256 `0cd608e6eff1114bf1b08873935ef1b65bc778ff34c4008cb56728ca052375f1`；analyzer SHA-256 `797252452eabe2d18a440d07e18ee092ec515c42fe35ca11ebec31d49a66f256`；report 自哈希 `295891e0d99754e0fbfdda5f8d71eddad3bc89ae0183d4d5b76aff1c746ac96c`；summary 自哈希 `7d8c6b0a25565f24f2bc39619175b7243ea6501a2a7eb75c795fd071fd79275c`；EXP-018 aggregate SHA-256 `781f09bf12c3fc3fd886c015245a78cf1e45ebfda25531ce95ee760c77c7ba2f`；`complex_baseline_md=false`; `solvent_baseline_md=false`; 不追加 EXP-018 seed、不改 EXP-018 verdict、不改 EXP-017 formal error |
| EXP-019 | 2026-08-11 | baseline sampling authorized / seed-wired paired MD | 主 preregistration 保持不变；seed qualification、artifact manifest 和 launcher/backend hashes 保持冻结。sampling addendum 已显式 reseal 为 `BASELINE_SAMPLING_AUTHORIZED` / `AUTHORIZED_NOT_STARTED`；complex 与 solvent baseline MD 均授权，lambda insertion/neural/analytic-q/DEXP 仍关闭，production data 不可变。重新 `validate-only` 已通过：5 个 seed 不变、输出目录为空、`md_started=false` | AUTHORIZED / VALIDATION_PASSED / MD_AUTHORIZED_NOT_STARTED | addendum SHA-256 `ba3ced6e95b9c191f0fd72ab2a29774ef7ac42f5c3cfe1ab512b4c96397c8b75`；manifest 自哈希 `80d69289a65ff1a29c505a9b0fc8c217e29730f0367d9d2f6a3f0ee77d55a38d`；qualification report 自哈希 `63b183f5c9070c9616812384d186639f9b8d8fd6d215f42c09cb65d3f918c888`；`registered_seed_count=5`; `complex_baseline_md=true`; `solvent_baseline_md=true`; `lambda_insertion=false`; `neural=false`; `analytic_q=false`; `dexp=false`; `production_data_mutated=false`; `md_started=false`; `main_preregistration_changed=false` |
| EXP-019 | 2026-08-11 | authorized paired MD attempt / invalid environment | 按 sealed launcher 启动 `20260901`；OpenMM 在 complex 预平衡 Context 初始化阶段报告 `CUDA_ERROR_NO_DEVICE (100)`，因此未进入任何 production window，也未生成有效 complex/solvent endpoint。launcher fail-closed 停止，非空输出目录和 launcher log 保留为 invalid attempt，禁止删除、复用或部分晋级；当前无可分析 baseline repeat。 | INVALID_ATTEMPT / BLOCKED_ON_CUDA_DEVICE | attempt log `output/outer_lambda_exp019_baseline_sampling_v1/launcher_logs/baseline_repeat_01_seed_20260901.log`；不构成 MD 完成、不改变 authorization 语义。最终 analyzer 已独立冻结：`scripts/analyze_exp019_baseline_repeats.py` SHA-256 `d44541f30d93155141344788a560c443da60e98eb413b3a12c02be957a3ee9f4`；freeze `protocols/EXP-019_analysis_freeze.json` SHA-256 `f08a92b0b907ee08725e89699692e1d5d2b9e1404af15481bf990feb1c718afa`；analyzer 对当前 partial root fail-closed，未读取数值 payload |
| EXP-019 | 2026-08-11 | v2 infrastructure execution package reseal | 保留 `_v1` sampling root、v1 addendum、v1 launcher 和 invalid attempt 原样；不重做 seed qualification。v2 改用独立 sampling root `output/outer_lambda_exp019_baseline_sampling_v2`，launcher 在创建输出目录前执行真实最小 OpenMM CUDA Context preflight；每条 repeat 原子写入自哈希 report，五条完成后原子写入自哈希 aggregate。analysis algorithm unchanged，新的 v2 freeze 绑定 v2 addendum/root。当前主机不执行 preflight/MD，v2 root 保持未创建。 | PACKAGE_SEALED / PENDING_RETRY | v2 launcher `scripts/run_exp019_baseline_sampling_v2.py` SHA-256 `59a3dd182e779103b87db9ad82ac03edd5f6e5ce9e6f5af571943830ad0bc811`; manifest self-hash `894c679e0d913b6bc3673963377cca2f8b87692a94ce295fe694df462dfcf052`; addendum self-hash `714f44a382e3627674b09b7da12ab355e4aaa752182c997a0a2e35f553c1c930`; analysis freeze self-hash `6065f1868169df43a2bb8c16f3073d17a4374999d1dc72afa3400bef6cf3a0f4`; `seed_qualification_rerun_required=false`; `scientific_result=NONE`; `EXP-019_status=PENDING_RETRY` |
| EXP-019 | 2026-08-11 | v2 code-wiring execution attempt | v2 在真实 CUDA 预平衡已运行 500 万步、Boresch 再平衡和 attachment 后，于 seeded vanishing preoptimization 首次触发 `NameError: system_type is not defined`；同时审计发现首次 Boresch 提交仍会用末帧覆盖 ensemble mean。v2 目录必须原样保留，未完成任何注册 repeat，不能复用其预平衡或 attachment 数据。 | INVALID_ATTEMPT / CODE_WIRING_NAMEERROR | `attempt=EXP-019-v2`; `completed_repeat_count=0`; `scientific_result=NONE`; 修复范围为 `preoptimization/vanishing/dual_lambda/integrator` seed callsite 与 Boresch no-last-frame-reanchor；seed qualification 必须重跑，v3 使用新 sampling root |
| EXP-019 | 2026-08-11 | v3 code-wiring qualification / execution package reseal | 新增 v3 launcher 与独立 root `output/outer_lambda_exp019_baseline_sampling_v3`；v3 seed qualification 实际进入受控 vanishing dual-λ preoptimization callsite，捕获 integrator seed 与 ledger 一致；Boresch regression 确认 committed equilibrium 与 ensemble mean 完全一致，末帧仅作 diagnostic。v2 root 保留且禁止复用；当前仅完成 validate-only，未启动 CUDA preflight 或 MD。 | PACKAGE_SEALED / VALIDATION_PASSED / MD_NOT_STARTED | v3 launcher SHA-256 `d936807127fab6b927a5c838d65feba7584c8553ed766c7c4af5cec3c2af10cc`; backend `abfe_pipeline.py` SHA-256 `e417013a9f596dc9a135cfa5b01b345c1dc34bb168696ae9c7dbb54542864d6b`; manifest self-hash `dcf247b7558d3bc84ecddc83d2995fc872d3cd42c20f5ec4cbb34d6e461fecdf`; addendum self-hash `ab53e993f28a51711a5eb0b3cc1fd0f556d5283bb4db591201c4c73a1ec5dda2`; analysis freeze self-hash `88b5a41d72bcbbf3a614d2fa45c9d03381c15904c10bece7ea62f1fb9be067d7`; seed qualification self-hash `1af7ea395a2058b9d9e01e1df37c73bd7a17324905c0227ebbfc7615b53bfafd`; `complex_baseline_md=true`; `solvent_baseline_md=true`; `lambda_insertion=false`; `neural=false`; `analytic_q=false`; `dexp=false`; `production_data_mutated=false`; `md_started=false` |

---

## 5A. EXP-000：困难窗口与首个慢变量冻结

当前状态：`PASSED`。困难 window 与 EXP-010 primary slow variable 均已冻结；该结论
不授予 production approval。

输入为 `output_lrc_fix/final_results.json`、`pre_equilibration.dcd`、
`rebalance_traj.dcd` 和 `topology.cif`。窗口排序规则在读取结果前固定为：

1. 最小化最终 Stage 2 `min_ess_ratio`；
2. 并列时选择更低的绝对 ESS；
3. 再并列时选择更高的 statistical inefficiency。

实际选择：

| 指标 | 结果 |
|---|---:|
| leg / stage | complex vanishing / Stage 2 |
| window index / state range | 0 / `[0,5)` |
| λvdW | `1.0, 0.923529, 0.854304, 0.790614, 0.731876` |
| minimum ESS ratio | 0.391266 |
| absolute ESS | 37.5616 |
| statistical inefficiency | 5.20498 |
| endpoint-difference uncertainty | 0.924927 kJ/mol |

预平衡阶段候选之一为 atenolol `C4-N2-C9-C10` 周期扭转角：

| 轨迹 | 帧数 | circular mean ± std | basin occupancy | core transitions |
|---|---:|---:|---|---:|
| pre-equilibration | 500 | -169.682° ± 37.800° | trans 0.858；gauche-minus 0.142；gauche-plus 0 | 7 |
| rebalance | 25 | -63.976° ± 7.742° | gauche-minus 1.0 | 0 |

basin 定义为 trans `|phi| >= 120°`、gauche-minus
`-120° <= phi < 0°`、gauche-plus `0° <= phi < 120°`。转换计数另用
trans `|phi| >= 150°`、gauche-minus `[-90,-30]°`、gauche-plus
`[30,90]°` 的 core hysteresis，避免边界抖动被重复计数。

证据：

| 项目 | 路径 | SHA-256 |
|---|---|---|
| WP-0 报告 | `output/outer_lambda_wp0/wp0_selection.json` | `80c6ebafde9394169673a2a2ee8604614270be4356233c9c83eafefa421e4b3e` |
| final results | `output_lrc_fix/final_results.json` | `eac4c9cbf4656df24e92323aa50b94cb32d58b62eb6a4828f0b20026b261b34d` |
| topology | `output_lrc_fix/topology.cif` | `6602f537d13179fc8294bcbaea1c7247fa9148b7d372a6411bc9f705db744ccf` |

限制：现有完成的 IBS window 能量产物没有坐标轨迹，因此不能用预平衡候选替代
困难 window 的慢变量判定。进一步盘点发现 window 0 有原生 Context checkpoint，
但当前代码重建出的 window System SHA
`be34fd38...` 与历史 manifest 的 `f27d2648...` 不同，不能冒充严格续算。

因此新增独立 scratch 路线：只读历史 λ、冻结 f_k、Boresch 和软核参数，从
rebalance 末帧重新 burn-in，在新目录生成坐标；不写回 production checkpoint、
能量或 convergence。CPU 1-step 连通性已完成，`production_data_mutated=false`。

对预平衡轨迹每 10 帧抽样的功能性筛选共发现 8 个 ligand 非环 rotatable torsion
和 27 个 0.6 nm 口袋残基 chi1。该结果只用于验证筛选代码。

正式困难窗口 scratch run1 使用 50 ps burn-in、500 ps sampling、1 ps/帧，共
500 帧；CUDA 实测 116.965 ns/day，且 `production_data_mutated=false`。周期候选
第一为 `VAL251 chi1`：\(g=127.91\)、\(N_\mathrm{eff}=3.91\)、圆周标准差
41.63°，但仅观察到 1 次 core 转换。原先示例 torsion
`[4586,4584,4591,4592]` 排第 19，因此已经排除为首选。

同一 run1 轨迹补算的 ligand 第一水合壳层平滑 coordination 为
均值 2.993、标准差 0.666、\(g=82.19\)、\(N_\mathrm{eff}=6.08\)；它可能有关，
但尚未定义可重复的 wet/dry 状态，不能和周期 torsion 用不同量纲分数直接混排。

| 项目 | 路径 | SHA-256 |
|---|---|---|
| run1 原始筛选 | `output/outer_lambda_slow_variable_screen/hard_window0_run1/candidate_screen.json` | `af27ee07e2056b5500962a72c5ddccf8e29df8a7acd618f5f0d03491d9d68fa8` |
| run1 含 hydration 的 v2 筛选 | `output/outer_lambda_slow_variable_screen/hard_window0_run1/candidate_screen_v2.json` | `5ed88f64f19655e55243cddf0e12d693c99264475df414934e4f103cce62450b` |

run1 当时的结论为 `candidate_ranking_only`；预注册要求至少完成三个独立随机种子，
并按稳定原子身份对齐后，才允许冻结 EXP-010 输入。

三种子重复已经完成。预注册重复门得到三个合格周期候选：

1. ligand torsion `[4591,4592,4593,4585]`，三条轨迹 core 转换
   `14/4/10`，\(g=3.54/75.54/132.89\)；
2. 相邻 ligand torsion `[4593,4585,4594,4595]`，转换 `3/14/15`；
3. `VAL251 chi1` `[4020,4022,4024,4026]`，转换 `1/0/2`。

第一、第二 torsion 的离散盆 NMI 在三个种子中为 0.22、0.27、0.58，说明有关联但
不完全冗余。EXP-010 首先使用排名第一的 torsion；第二 torsion 作为二维模型候选，
`VAL251` 只作诊断。hydration 在三条轨迹中的 \(g\) 为
82.19/126.58/34.61，持续保留，但在定义 wet/dry 状态前不进入 production CV。

| 项目 | 路径 | SHA-256 |
|---|---|---|
| 三种子比较 v2 | `output/outer_lambda_slow_variable_screen/replicate_comparison_v2.json` | `fc339493a672f36ce9b5c4917208ac39ac71bb17364c081b5095ff0320cc5a85` |
| EXP-010 慢变量 manifest | `output/outer_lambda_slow_variable_screen/slow_variable_manifest.json` | `b14819c50149cadd4db3b641ba514d72126cfda049dd96351ab80aaf82d3455d` |

manifest 内部协议 SHA-256 为
`f09ecf3786a7c3f65db19181733d7afc2f9489d49a050012173d2ac9bd541805`，
状态是 `frozen_for_exp010_teacher_distillation`，且明确
`production_approval=false`。

### 5A.1 EXP-010 教师数据准备

独立模块现已实现：

- `exp010-prepare-selection`：删除交换水并冻结 protein-only 选择；
- `exp010-label`：按三条独立 run 生成 MACE 能量、primary/secondary torsion 和
  primary 广义力标签；
- `exp010-fit`：拟合 1D/2D 周期 Fourier 守恒势，并按整条 run 留一验证；
- 解析 `CustomTorsionForce` / `CustomCompoundBondForce` 导出、XML round-trip
  和重叠 torsion 测试。

教师能量 offset 使用所有通过支持域帧的冻结数据集均值，只消除常数分量，不调整
`coefficient=0.09`。周期 Fourier 的支持区间是完整
\([-\pi,\pi]\)，不存在非周期边界外推。

CPU 一帧 protein-only 连通性结果：

| 项目 | 值 |
|---|---|
| 教师选择 | ligand 41 + protein environment 216 |
| MACE raw interaction energy | -175.978129 kJ/mol |
| primary torsion | -149.470817° |
| primary generalized force | -8.295638 kJ/mol/rad |
| 教师最大原子力 | 714.669 kJ/mol/nm |
| 支持域 | max pair 2.341981 nm；Rg 0.727548 nm；通过 |
| safety / support violations | 0 / 0 |

| 项目 | 路径 | SHA-256 |
|---|---|---|
| protein-only 选择 smoke | `output/outer_lambda_exp010/smoke_cpu_protein_only/protein_only_selection_meta.json` | `76d25abe03ac47071b3a03bbde4988d83a9f0bcf0142e472f592b49f71393ae5` |
| CPU 一帧教师数据 | `output/outer_lambda_exp010/smoke_cpu_protein_only/teacher_dataset.json` | `4055fcf8142c82f747acda73f8e08c8b3edcdfd5cc3e5b922fcd0f0474855660` |
| 正式节点脚本 | `run_outer_lambda_exp010_teacher_and_fit.sh` | `1d6ffb3129cbd431d8e63241864faa7a40b0c839ddb6824e4cea9cdde8e5f3db` |

正式节点矩阵使用每 run 100 帧，共 300 个 source frames；支持域实际保留
290 帧。拟合候选固定为 1D order 2/4/6 和 2D order 2/3/4，结果返回前未增删
候选。正式结果和失败判定见第 11 节。

---

## 6. EXP-004：真实 MACE 局部基势离线可调用性

### 6.1 基本信息与研究问题

| 字段 | 值 |
|---|---|
| 状态 | PASSED |
| 日期 | 2026-07-30 |
| 对应工作包 | WP-4 前置 |
| 模型 | `mace-off24-medium` |
| 模型 SHA-256 | `e5ccf5837f685899811a68754e7c994393bfd1a81720393b03c643b46c70bc69` |
| 局部选择 | ligand 41 原子 + 固定 environment 255 原子 |
| 分解定义 | `E(complex)-E(ligand)-E(environment)`，力按同一线性组合映射回全坐标 |
| 外层系数 | `0.1` |
| λ schedule | `0,0.25,0.5,0.75,1` |
| energy offset | `0.0 kJ/mol`（尚未标定） |

唯一研究问题：

> 现有真实 MACE 模型能否通过 `ExistingOrbMaceBasisAdapter` 进入独立外层 λ
> 离线评价链路，并给出有限、可重复、端点严格归零的能量和力？

本实验不回答：

- MACE 路径能否改善 IBS、ESS 或 ΔG；
- 当前局部选择和未成像坐标是否已经满足正式支持域；
- 三 context 评价后端能否直接用于每个 MD step；
- 当前系数、offset 或安全阈值是否为 production 值。

### 6.2 输入与代码身份

| 项目 | 路径 | SHA-256 |
|---|---|---|
| 独立模块 | `outer_lambda_neural_basis.py` | `64f6b13a20ab01efa254403f16df3c48f5f9321a88b77f4ddf973c54a83e5a07` |
| topology | `output/topology.cif` | `27f78ec5fe761a27807a76e262a3d5efc7b5faff58d2b6a5a40ee8085768dde1` |
| trajectory | `output/pre_equilibration.dcd` | `1a28f2e076e110af861248345d854a19163f10e293e85b53bec937d79c5fc9f8` |
| 固定选择 meta | `output/dexp_experiment/fit_label_cache_meta.json` | `8f0d8af51f44804f50b7e9e665aa7b0ff1e761df1a9d3b0e88dc1882754c5d23` |
| MACE 模型 | `~/.cache/mace/MACE-OFF24_medium.model` | `e5ccf5837f685899811a68754e7c994393bfd1a81720393b03c643b46c70bc69` |

运行入口：

```bash
CUDA_VISIBLE_DEVICES=0 FRAME_SPEC=last \
  RUN_DIR="$PWD/output/outer_lambda_existing_model/mace_smoke" \
  bash run_outer_lambda_existing_model.sh

CUDA_VISIBLE_DEVICES=0 FRAME_SPEC=400:500:10 \
  RUN_DIR="$PWD/output/outer_lambda_existing_model/mace_10frames" \
  bash run_outer_lambda_existing_model.sh

CUDA_VISIBLE_DEVICES=0 FRAME_SPEC=tail:50 \
  RUN_DIR="$PWD/output/outer_lambda_existing_model/mace_tail50" \
  bash run_outer_lambda_existing_model.sh
```

### 6.3 三批运行结果

| 批次 | frame | 数量 | 总评价时间 s | s/frame | Basis 能量均值 ± SD kJ/mol | 最大 Basis 力 kJ/mol/nm | 端点能量/力 |
|---|---|---:|---:|---:|---:|---:|---|
| smoke | 499 | 1 | 38.741 | 38.741（含初始化） | -301.155 ± 0.000 | 1448.773 | 精确 0 / 精确 0 |
| 10-frame | 400:500:10 | 10 | 46.903 | 4.690 | -278.159 ± 25.831 | 2457.730 | 全部精确 0 / 全部精确 0 |
| tail-50 | 450–499 | 50 | 211.529 | 4.231 | -291.813 ± 26.579 | 2457.730 | 全部精确 0 / 全部精确 0 |

50 帧详细范围：

- Basis energy：`[-344.708, -236.595] kJ/mol`；
- Basis 最大原子力：`[874.800, 2457.730] kJ/mol/nm`；
- 50 帧均无非有限能量或力；
- 第 499 帧在独立 smoke 与 tail-50 中的能量差约
  `7.3e-11 kJ/mol`，最大力完全一致；
- 系数 `0.1` 下观测到的最大路径附加力为
  `245.773 kJ/mol/nm`。

证据目录：

- `output/outer_lambda_existing_model/mace_smoke/`
- `output/outer_lambda_existing_model/mace_10frames/`
- `output/outer_lambda_existing_model/mace_tail50/`

### 6.4 观测限制与解释

直接观测：

- 真实 MACE 局部分解能量和守恒力已成功进入外层 λ 离线评价层；
- 三批结果有限、同帧重复一致，外层 λ=0/1 能量和附加力严格归零；
- 持久 Context 稳态成本约 `4.2–4.7 s/frame`，明显不适合作为每步 MD
  后端；
- 配置未声明 `support_domain`，所以报告中的零 support violation 不能解释为
  正式支持域通过；
- 原始未成像固定选择在 frame 499 的最大两原子距离为 `43.982 nm`，
  明显是 PBC/坐标成像问题，不是合理局部几何尺度；
- `energy_offset=0`，尚未执行路径中心化和幅度标定。

结论边界：

> EXP-004 只证明“真实 MACE 基势评价层可调用、有限、可重复且端点契约正确”。
> 它不构成短 NVT、IBS、自由能、WP-5 或 production 证据。

### 6.5 决策与后续行动

总体状态：`PASSED`（WP-4 前置可调用性门）。

| 后续行动 | 完成条件 | 状态 |
|---|---|---|
| 修复局部选择的 PBC 成像和固定水/环境身份协议 | 局部几何连续，跨帧身份固定，支持 triclinic PBC | COMPLETED；EXP-010 改为 protein-only 教师 |
| 用代表性轨迹定义 support domain | 阈值预注册，违规统计有实际含义 | COMPLETED；2.5 nm / 0.85 nm 门保持不变 |
| 标定 energy offset 和外层幅度 | 能量中心、力分位数和安全门预注册 | COMPLETED for EXP-007；cheap-CV 使用数据集均值 offset |
| 将成熟基势变成每步可执行 OpenMM Force | 单一/共享模型后端，不依赖三 context 每帧 probe | DIRECT MACE STOPPED；解析 cheap-CV Force 机制已实现，但 EXP-010 无合格模型可资格 |
| 接入 IBS 现场采样和 cross-state ledger | 仅在独立 Force/NVT 门通过后执行 | PLANNED |
| WP-5 三臂实验 | baseline / λ relayout / neural path 全部完成 | PLANNED |

---

## 7. EXP-005：每步 MACE Force NVT 标定

当前状态：`PASSED`（WP-4 qualification 前置标定，不是 qualification）。

已完成：

| 批次 | CUDA 步数 | λ | 总墙钟 s | 纯积分 s/step | path energy kJ/mol | path-only 最大力 kJ/mol/nm | 最大闭合误差 kJ/mol | 结果 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `step1` | 1 | 0.5 | 8.274 | 旧口径未分离 | -35.327 | 旧口径未分离 | 0.00236 | PASSED |
| `step10` | 10 | 0.5 | 8.989 | 旧口径未分离 | -35.330 至 -33.633 | 旧口径未分离 | 0.01290 | PASSED |
| `step100_v2` | 100 | 0.5 | 21.329 | 0.10218 | 均值 -34.293 | 186.372 | 0.00508 | PASSED |

证据：

- `output/outer_lambda_mace_nvt/step1/nvt_smoke.json`
- `output/outer_lambda_mace_nvt/step10/nvt_smoke.json`
- `output/outer_lambda_mace_nvt/step100_v2/nvt_smoke.json`

已确认：

- MACE 路径 Force 已在真实 OpenMM Context 中随积分器每步执行；
- 无 NaN、无积分器异常；
- base + path 与 total 能量在数值精度内闭合；
- 调用方 base System 未被修改；
- λ=0/1 的端点 fast path 在运行模型前返回严格零。
- 修正版 100-step 报告中 PBC minimum-image 支持几何稳定：
  最大 pair distance `2.154–2.176 nm`，最小 pair distance
  `0.095719 nm`，回转半径约 `0.731 nm`；
- 100 步内 support violation 为 0，但该轮配置没有预先声明 support domain，
  因此它只能用于冻结下一轮阈值。

本轮边界：

- 100 步不足以作为正式稳定性 qualification；
- support domain 和 `energy_offset` 是看到本轮数据后才标定，不能追认本轮通过；
- 没有 IBS、ESS 或 ΔG 证据。

决定：结束 smoke 扩展，进入 EXP-006。

## 8. EXP-006：冻结协议的 MACE NVT qualification

当前状态：`FAILED`。

预注册配置：

| 项目 | 冻结值 |
|---|---:|
| NVT steps | 1000 |
| report interval | 25 |
| timestep | 0.5 fs |
| λ / coefficient | 0.5 / 0.1 |
| energy offset | -343.0 kJ/mol |
| support min pair | 0.07 nm |
| support max pair | 2.5 nm |
| support max radius of gyration | 0.85 nm |

硬门：

| 检查 | 阈值 |
|---|---:|
| 完整运行 | `passed=true` 且 steps ≥ 1000 |
| support domain | 已配置且违规数为 0 |
| path-only 最大原子力 | ≤ 250 kJ/mol/nm |
| 最大能量闭合误差 | ≤ 0.1 kJ/mol |
| 纯积分耗时 | ≤ 0.2 s/step |

实际命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
RUN_DIR="$PWD/output/outer_lambda_mace_qualification/run1" \
bash run_outer_lambda_mace_qualification.sh
```

输出：

- `output/outer_lambda_mace_qualification/run1/prepare_report.json`
- `output/outer_lambda_mace_qualification/run1/nvt_qualification.json`

实际结果：

| 检查 | 观测 | 结论 |
|---|---:|---|
| 完整运行 | 1000 steps，`passed=true` | PASS |
| support domain | 已配置，0 次违规 | PASS |
| path-only 最大原子力 | 258.949 kJ/mol/nm | **FAIL** |
| 最大能量闭合误差 | 0.01135 kJ/mol | PASS |
| 纯积分耗时 | 0.09905 s/step | PASS |
| 总墙钟 | 127.127 s | 记录项 |

路径力 40 个报告点的均值、P95 和最大值分别为
`201.803 / 255.883 / 258.949 kJ/mol/nm`；共有 6 个报告点超过 250。
路径能量范围为 `[-3.499, 2.480] kJ/mol`。支持域实际范围：

- minimum pair distance：`0.0957191–0.0957198 nm`；
- maximum pair distance：`2.1535–2.3269 nm`；
- radius of gyration：`0.7302–0.7355 nm`。

身份记录：

| 项目 | SHA-256 |
|---|---|
| 独立模块 | `362d59e2572eb26942201f6c180ca695aad5a09b6ca66434a416f0270a09ae39` |
| qualification 脚本（EXP-006 实际版本） | `a46a0beb3eda5b62e5e0d6df5bb57355ff2b91601f8dc78d63cec5192a006a6f` |
| 生成配置 | `7de6a5d26046ef86d53dfbded600e8603084c078995959a681362fda9c6054a4` |
| qualification 报告 | `5acf16a1fe5eea16ef3aaf081317c300017edd641d3bb3d404e71a5f707ae9ea` |
| system XML | `e2eb7b94fceec5b4cdf552972fc40fa633d49b8e4385acfc62a357e4bfc01717` |

结论：严格保持 `FAILED`，不修改原报告、不提高验收阈值。失败是幅度门，
不是部署、稳定性、支持域或性能失败。

## 9. EXP-007：系数 0.09 的 MACE NVT qualification

当前状态：`PASSED`。

相对 EXP-006 唯一协议变化：

```text
coefficient: 0.10 -> 0.09
```

支持域、offset、初始 frame、1000 steps、报告间隔、积分参数和全部资格硬门保持
不变。在 EXP-006 的相同坐标上，纯线性缩放会将最大路径力从 258.949 降至约
233.054 kJ/mol/nm；新轨迹仍必须独立通过实际硬门，不能用该推算代替运行。

预注册命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
COEFFICIENT=0.09 \
RUN_DIR="$PWD/output/outer_lambda_mace_qualification/run2_coeff009" \
bash run_outer_lambda_mace_qualification.sh
```

实际结果：

| 检查 | 观测 | 结论 |
|---|---:|---|
| 完整运行 | 1000 steps，`passed=true` | PASS |
| support domain | 已配置，0 次违规 | PASS |
| path-only 最大原子力 | 246.775 kJ/mol/nm | PASS |
| 最大能量闭合误差 | 0.01490 kJ/mol | PASS |
| 纯积分耗时 | 0.09771 s/step | PASS |
| 总墙钟 | 125.490 s | 记录项 |

路径力均值/P95/最大值为 `182.632 / 230.522 / 246.775 kJ/mol/nm`；
路径能量范围 `[-3.120, 2.466] kJ/mol`。support-domain 实际范围全部位于
预注册边界内。

| 项目 | SHA-256 |
|---|---|
| 独立模块 | `362d59e2572eb26942201f6c180ca695aad5a09b6ca66434a416f0270a09ae39` |
| qualification 脚本 | `91e59b7549e95d0c446a2abf5e99146eadd2c6a63c52a8b416f4d1b5e92a8d32` |
| 生成配置 | `8ab035799c190910477f4f28a10b9b02eec8e0f733d590031ea7d9a9c474babd` |
| qualification 报告 | `ce27671f6af95ef8f0a0b0d8095a8503e79fa0ea9c56ff81a84aa4a73fa72d8d` |
| prepare 报告 | `de9aed74c41643c1b4cd9fdb8a821ce6d52936789ea1a4ccb889beea7274423b` |

结论：WP-4 完整 MACE 的 NVT qualification 通过。冻结
`coefficient=0.09`、offset、支持域和模型身份；随后 EXP-009 证明当前直接 MTS
后端不晋级，因此实际路线转入 EXP-010，而不是直接开始 WP-5。不再调整 EXP-007
阈值或重复相同 NVT qualification。

---

## 10. EXP-009：完整 MACE 的多时间步性能资格

当前状态：`FAILED`。EXP-007 已通过并冻结 `coefficient=0.09`；本实验没有改变幅度。

唯一研究问题：

> 在不产生可分辨积分偏差的前提下，完整 MACE 路径 Force 能否通过 MTS 降低到足以
> 进入单窗口 IBS 的成本？

第一阶段预注册矩阵：

| coefficient | MTS ratio \(N\) | 当前 0.5 fs 内步长下的 MACE 间隔 |
|---:|---:|---:|
| 0.09 | 1 | 0.5 fs |
| 0.09 | 2 | 1 fs |
| 0.09 | 4 | 2 fs |

降低 coefficient 会减小路径力，但不会降低一次 MACE forward 的成本。因此系数不属于
本性能矩阵的扫描轴；任何新系数都必须另建幅度资格实验。

执行规则：

- \(N=1\) 是每个系数自己的参考，不得省略；
- 使用 OpenMM force group 与 MTS/r-RESPA；不得在 Python 循环中冻结旧力；
- 三臂使用同一初始坐标、300 K、随机种子 `20260730` 和 50 ps 物理时长；
- 正式入口每 0.25 ps 记录一次诊断，共 200 个样本/臂；
- 只有 \(N\le4\) 全部通过，才允许单独探索 \(N=8\)；
- 本阶段必须同时报告稳定性、分布偏差和 ns/day；ESS/GPU-hour 留到有 IBS
  target-energy 账本的 WP-5，不能由普通 NVT 伪造；
- 仅“没有崩溃”不构成通过。

预注册硬门：

| 检查 | 门限 |
|---|---:|
| 完整运行 / finite | 三臂全部完成，无非有限诊断 |
| support domain | 已配置且违规数为 0 |
| path-only 最大原子力 | `<= 250 kJ/mol/nm` |
| 最大能量闭合误差 | `<= 0.1 kJ/mol` |
| 温度均值相对 N=1 | 绝对差 `<= 5 K` |
| 总势能均值相对 N=1 | 标准化差 `<= 0.25` |
| 扭转角分布相对 N=1 | 24-bin Jensen-Shannon divergence `<= 0.05` |
| ligand Kabsch RMSD 均值相对 N=1 | 绝对差 `<= 0.05 nm` |
| N=4 实测性能 | `>= 1.0 ns/day` |

以上分布门是本次 EXP-009 的首轮工程资格判据，不等同于长时间平衡分布证明。
报告必须同时保存每个采样点，便于发现均值门掩盖的漂移。

实现入口：

```bash
CUDA_VISIBLE_DEVICES=0 \
  RUN_DIR="$PWD/output/outer_lambda_mace_mts/exp009_run2_input_fix" \
  bash run_outer_lambda_mace_mts_qualification.sh
```

脚本固定 `coefficient=0.09`，默认
`N_INNER_STEPS=100000`、`REPORT_INTERVAL_INNER_STEPS=500`。开发连通性短跑必须使用
不同输出目录，并且不得登记为 EXP-009 正式结果。

首次设置故障记录（未进入 MTS 比较，不改变本实验 `PLANNED` 状态）：

- `output/outer_lambda_mace_mts/exp009` 错误使用了
  `output_lrc_fix/pre_equilibration.dcd`，而 offset、固定原子选择和 WP-4
  qualification 来自 `output/` 协议；
- 两条轨迹 SHA-256 不同；该混用帧的离线 MACE 结果为 basis
  `-186.622 kJ/mol`、centered basis `156.378 kJ/mol`、λ=0.5 路径能
  `14.074 kJ/mol`；
- 更早的支持域预检应当拒绝该帧：最大 pair distance `7.473 nm`、
  Rg `1.927 nm`，超过冻结的 `2.5 nm / 0.85 nm`；
- 节点运行随后在 \(N=1\) 触发 `max_abs_path_energy=20 kJ/mol`，因此没有
  产生可用于 MTS 判定的 arm report；
- 修正只拆分输入协议，不改变 coefficient、offset、MTS ratio 或任何资格阈值：
  WP-0 继续读取 `output_lrc_fix/`，MTS 改为严格复用 EXP-007 的
  `output/pre_equilibration.dcd` 与 `output/topology.cif`；
- 独立模块已增加积分前支持域 fail-closed，脚本已增加运行目录防覆盖门。

第二次运行使用修正后的 EXP-007 输入，但 \(N=1\) 在
`MTSLangevinIntegrator` 的 CUDA force-group 执行中返回
`CUDA_ERROR_INVALID_HANDLE (400)`。这发生在任何 \(N=2/4\) 比较之前，故没有
资格声称存在或不存在积分分布偏差。

部署判定：

- OpenMM `8.5.1`；
- openmmml `1.6`，其 MACE 后端为 `openmm.PythonForce`；
- openmmtorch `1.5` 已安装，但当前实时 MACE 分解并非 TorchForce；
- torch `2.10.0`，mace-torch `0.3.16`；
- 普通 LangevinMiddleIntegrator 的 EXP-007 通过，但不能外推到 CUDA
  CustomIntegrator/MTS force-group。

总体结论：`FAILED`（直接完整 MACE MTS 的后端资格失败，不是 MACE 势能物理失败）。
按预注册决策 `start_exp010_cheap_cv_due_to_backend`，禁止通过重试、降低系数或放宽
阈值掩盖部署不兼容。

若只有 \(N\ge8\) 或更大间隔才能获得可接受速度，而该间隔不能通过分布/积分门，
直接 MACE 生产路线判为不晋级，启动 EXP-010。

预注册决策：

- \(N=4\) 物理门和性能门都通过：单独测试 \(N=8\)，再决定是否进入 WP-5；
- \(N=4\) 物理门通过但低于 1.0 ns/day：启动 EXP-010；
- \(N=2\) 或 \(N=4\) 出现积分/分布偏差：完整 MACE 只保留为教师，启动 EXP-010。

## 11. EXP-010：MACE 教师到 cheap-CV bias 的蒸馏

当前状态：`FAILED`。正式 GPU 教师标注成功完成并通过数据支持域门，但六个预注册
Fourier 候选均未通过跨 run 能量和广义力验证，因此没有最终 cheap-CV 模型。

目标不是重新训练连续 λ-conditioned MACE，而是：

\[
U_{\rm MACE}^{\rm local}(\mathbf R)
\longrightarrow
V_\phi(s(\mathbf R))
\]

primary \(s_1\) 已冻结为 `[4591,4592,4593,4585]`，secondary \(s_2\) 为
`[4593,4585,4594,4595]`。当前实现使用完整周期 Fourier：

- 1D：order 2、4、6；
- 2D：order 2、3、4；
- 每个候选都按整条 run 留一验证；
- 解析导出为 `CustomTorsionForce` / `CustomCompoundBondForce`。

生产时 cheap-CV bias 每步计算；完整 MACE 只用于训练/验证标签，不在每个 MD step
运行。实验必须记录教师模型哈希、训练轨迹拆分、慢变量定义、支持区间、外推衰减、
能量/平均力验证和 ESS/GPU-hour。

### 11.1 已冻结输入协议

| 项目 | 值 |
|---|---|
| 教师模型 | `mace-off24-medium` |
| 教师能量 | `E(complex)-E(ligand)-E(protein environment)` |
| ligand / environment | 41 / 216 atoms |
| 水策略 | 移除旧环境中全部 39 个固定水原子；不做动态水选择 |
| source trajectories | 三条独立 window-0 scratch trajectories |
| frame 选择 | 每条每 5 帧取样，100 frames/run，300 source frames |
| 支持域 | min pair 0.07 nm；max pair 2.5 nm；Rg 0.85 nm |
| 支持域违规策略 | MACE 前排除并登记；排除率硬门 5% |
| 实际预检 | 290/300 通过；10 帧因 max pair 2.50–2.60 nm 排除 |
| energy offset | 合格教师帧 raw energy 的冻结均值 |
| 外层 coefficient | 0.09；本实验不再调幅度 |
| 数据拆分 | leave-one-run-out；禁止随机 frame split |

### 11.2 已完成 smoke

protein-only CPU 一帧得到 raw interaction energy `-175.978129 kJ/mol`、
primary generalized force `-8.295638 kJ/mol/rad`、最大原子力
`714.669 kJ/mol/nm`。支持域 max pair `2.341981 nm`、Rg `0.727548 nm`，
support/safety violation 均为 0。

截至 2026-07-31，独立模块 SHA-256 为
`956a4401710c812df7125cf32431b362a75b1a5ec173203b805fbd8a0f0f11f3`；
相关独立测试最近一次为 `80 passed, 1 skipped`，skip 原因为本执行节点无 CUDA
device。

### 11.3 实际运行命令

```bash
CUDA_VISIBLE_DEVICES=0 \
RUN_DIR="$PWD/output/outer_lambda_exp010/run1" \
bash run_outer_lambda_exp010_teacher_and_fit.sh
```

该命令已完成，输出目录未覆盖既有实验。

### 11.4 教师数据集结果

| 检查 | 结果 | 判定 |
|---|---:|---|
| source / 合格帧 | 300 / 290 | PASS |
| 支持域排除率 | 3.333%（门限 5%） | PASS |
| 合格帧 support / safety violation | 0 / 0 | PASS |
| CUDA 标注时间 | 1564.997 s | 记录项 |
| dataset mean offset | -168.957841 kJ/mol | 已冻结 |
| centered energy 标准差 / 范围 | 21.510895 / `[-75.0555, 53.3613]` kJ/mol | 记录项 |
| primary generalized force 范围 | `[-20.7254, 5.81678]` kJ/mol/rad | 记录项 |
| teacher 最大原子力范围 | `[370.327, 1746.150]` kJ/mol/nm | 记录项 |

教师数据集 `ok=true`、`qualified_for_fit=true`。这只说明标签可用于拟合，不代表任何
cheap-CV 候选通过。

### 11.5 六候选 leave-one-run-out 结果

intercept-only 能量基线 RMSE 为 `21.510895 kJ/mol`。预注册候选结果如下：

| 候选 | 参数数 | 能量 RMSE kJ/mol | 能量 \(R^2\) | 广义力 RMSE kJ/mol/rad | 广义力 \(R^2\) | 判定 |
|---|---:|---:|---:|---:|---:|---|
| 1D order 2 | 5 | 22.1737 | -0.0626 | 19.8500 | -13.5934 | FAIL |
| 1D order 4 | 9 | 22.3263 | -0.0772 | 28.9483 | -30.0371 | FAIL |
| 1D order 6 | 13 | 22.8989 | -0.1332 | 73.2137 | -197.526 | FAIL |
| 2D order 2 | 25 | 362.122 | -282.395 | 335.942 | -4178.87 | FAIL |
| 2D order 3 | 49 | 1042.67 | -2348.50 | 904.546 | -30302.6 | FAIL |
| 2D order 4 | 81 | 5000.44 | -54037.0 | 3860.46 | -551963 | FAIL |

1D order 2 是六者中最小的留一能量 RMSE，但仍劣于 intercept-only，且广义力方向/
幅度不稳定。2D 候选随阶数增加出现严重病态拟合。部分全数据训练或 conditional-bin
指标较好，不能替代整条 run 留一验证，也不能据此选择模型。

| 证据 | 路径 | SHA-256 |
|---|---|---|
| 正式教师数据集 | `output/outer_lambda_exp010/run1/teacher_dataset.json` | `6cdffc3984302e80f0b342b69c5d7d0f666b7a461e221deb4e2a21a91ebfed35` |
| 1D order 2 | `output/outer_lambda_exp010/run1/fit_1d_order2.json` | `eebaf9fbc1bf2e628d5810f119c1d0da64bfc9fcc32df653ee4462fb541daada` |
| 1D order 4 | `output/outer_lambda_exp010/run1/fit_1d_order4.json` | `2fc3b98479752bd41b4c349dd47b083e5ad21fed71d4cdc1b69deb4cdeb473c0` |
| 1D order 6 | `output/outer_lambda_exp010/run1/fit_1d_order6.json` | `72783aebc84d0c0cba3e7720a500e3dae4bf926aff9bd576278ba13e2527ff81` |
| 2D order 2 | `output/outer_lambda_exp010/run1/fit_2d_order2.json` | `af2a122bfb544e86229623b9448646014f66582f00bb3e03de454ffeebca23ba` |
| 2D order 3 | `output/outer_lambda_exp010/run1/fit_2d_order3.json` | `0501749f2fa9d8884e77f6698836d228cffbc37175b7b4219e2525caa23a9cbc` |
| 2D order 4 | `output/outer_lambda_exp010/run1/fit_2d_order4.json` | `3dcaba470e07ec1eb48156d7efc73e6fd086fbbcbe76e45740b858f93849b57c` |

### 11.6 决定

EXP-010 严格记为 `FAILED`。不冻结六个候选中的任何一个，不执行 cheap-CV NVT
qualification，也不启动 EXP-008/WP-5。不得在看到结果后增加 Fourier order、改变
ridge、改用随机 frame split 或用训练集指标追认通过。

本实验不能将失败归因于 torsion bias 或 Fourier 表达本身。事后选择完整性
审计发现，216 个 protein atoms 是由单个参考帧的 0.5 nm 原子半径选择得到，
共涉及 26 个残基，但完整选中的残基数为 0。因此 MACE 看到的是存在断裂共价
边界和不完整局部邻域的人工簇。对这一冻结原子集，
\(E(L+P_{\rm sel})-E(L)-E(P_{\rm sel})\) 代数上有定义；但它不等于完整蛋白环境下的
物理 interaction energy，常数 offset 也无法修复这一边界依赖。

此外，当前的逐帧目标要求低维 \(V(\phi_1,\phi_2)\) 重现高维瞬时 MACE interaction
energy 和力，理论上只有当教师能量几乎只依赖这两个 CV 时才可能逐帧闭合。
实际 primary torsion 对教师 Cartesian force 的平均投影解释比仅 `0.00393`；三条 run
的 centered-energy 均值为 `+4.67/-12.44/+7.72 kJ/mol`。这些观测同时兼容
教师边界伪影和未显式慢变量，不能单独用来排除 torsion bias。

### 11.7 EXP-011 推进边界

EXP-011 将作为新实验，不是 EXP-010 重试。唯一研究问题是：

> 完整困难窗口 MM Hamiltonian 下 primary torsion 的条件平均力或 PMF，能否在
> 整条 run 留一验证中生成稳定、周期且可积分的 cheap bias？

主路线要求：

- 教师使用完整 MM Hamiltonian，不做局部 protein atom selection；
- 目标是沿 CV 的条件平均力/PMF，不是逐帧完整相互作用能；
- 先检查三条 run 的周期 bin 覆盖和重叠，不足时先增加受限/增强采样；
- 候选形式、平滑度、幅度和跨 run 门限必须在看正式结果前冻结；
- 只有先通过条件平均力/PMF 跨 run 验证，才进入 OpenMM NVT 力学资格。

2026-08-02 已冻结机器可读协议
`protocols/EXP-011_preregistration.json`（内部 SHA-256
`d18a38706b5bcd5aa4c7d713e1c34aa9fd19512398b2582c793b2a097096c596`），并实现
`exp011-coverage` 与 `exp011-fit-pmf`。验收说明见
`docs/experiments/EXP-011_PREREGISTRATION.md`。

重新审计后，EXP-011 专用 manifest 已冻结为
`output/outer_lambda_exp011/slow_variable_manifest.json`，状态为
`frozen_for_exp011_complete_mm_pmf`，primary CV 为 `[4591,4592,4593,4585]`，内部
manifest 哈希有效且 `production_approval=false`。

正式覆盖基线 `output/outer_lambda_exp011/coverage_report_v2.json` 为
`qualified_for_pmf=false`：pooled 空 bin 为 11、17–22；run 2/3 的周期有效样本数仅
6.62/3.76；run1/run2 overlap 为 0.328。决定为
`collect_restrained_or_enhanced_sampling`。因此没有生成 PMF model，也没有启动 NVT、
WP-5 或 production。

2026-08-02 已实现 `exp011-umbrella-sample` 与 `exp011-reweight-umbrella`。周期 restraint
使用最短周期角差；采样报告逐帧保存 angle/umbrella energy；重加权按 source window
去相关，用 MBAR 导出显式 `log_target_weight` 并检查 overlap 图。相关测试为
`58 passed`。

完整体系单中心 CUDA smoke 已通过。失败的 Reference 诊断目录
`output/outer_lambda_exp011/cpu_smoke_center_m172p5` 仍只含 DCD header，不得复用；正式结果
位于 `output/outer_lambda_exp011/cuda_smoke_center_m172p5/report.json`。该运行使用
`center=-172.5°`、`k=100 kJ mol^-1 rad^-2`、200 次最小化和 1 个积分/采样步；报告
`ok=true`、`platform=CUDA`、angle `-173.5426°`、umbrella energy
`0.01656 kJ/mol`、temperature `278.03 K`，且 checkpoint 与 DCD 均已写出。
因此前处理与周期 restraint 的 GPU 执行门已通过。下一动作是同一中心的短时稳定性 pilot；
它通过前仍不批量启动 24 centers、PMF、WP-5 或 production。

同一中心短时稳定性 pilot 随后通过，结果位于
`output/outer_lambda_exp011/pilot_run1_center_m172p5/report.json`：10/10 个诊断帧均为
有限值，temperature 为 `298.12–302.11 K`，angle 为 `-173.12°` 至 `-158.97°`，最大
umbrella energy 为 `2.79 kJ/mol`，10 ps 采样段性能为 `52.78 ns/day`。该结果仅批准相邻
中心 overlap pilot，不作为平衡性证明或正式 PMF 数据。下一中心冻结为 `-157.5°`，保持
相同 `k=100 kJ mol^-1 rad^-2` 和短 pilot 长度。

`-157.5°` 相邻中心 pilot 也通过：10/10 帧有限，temperature `298.12–301.15 K`，
angle `-168.47°` 至 `-151.35°`。两窗 MBAR 报告
`output/outer_lambda_exp011/pilot_run1_two_window_mbar.json` 为 `ok=true`，邻窗 overlap
`0.3584`，高于冻结门 `0.03`；两窗各保留 10 个去相关样本。这里的
`qualified_for_pmf_input=true` 仅表示这两个局部窗口的 overlap 图连通，不能解释为全周期
覆盖、平衡性或正式 PMF 验收通过。下一步只增加第三个相邻中心 `-142.5°`。

第三个 `-142.5°` pilot 通过：10/10 帧有限，temperature `297.07–301.49 K`，angle
`-149.75°` 至 `-131.84°`，最大 umbrella energy `1.73 kJ/mol`。三窗 MBAR 报告
`output/outer_lambda_exp011/pilot_run1_three_window_mbar.json` 为 `ok=true`；相邻 overlap
为 `0.3105` 以及 `0.1864/0.3728`，均高于 `0.03`。第三窗估计 statistical
inefficiency `1.943`，只保留 5/10 个去相关样本，进一步说明 10 ps pilot 不能当正式 PMF
数据。局部 15° spacing 与 `k=100` 已获执行资格；下一测试点改为历史空白区
`75°–165°` 中部的 `112.5°` 哨兵中心，用于检查远离初态时的可达性和数值稳定性。

`112.5°` 哨兵窗通过可达性与数值稳定性门：10/10 帧有限，temperature
`298.66–302.28 K`，angle `93.22°–112.21°`，最后一帧 `112.21°`，最大 umbrella
energy `5.66 kJ/mol`。它证明远端空白区可由当前最小化/burn-in 流程到达，但分布明显偏向
中心低侧，不能单独证明高侧邻接连通。下一步运行 `127.5°` 高侧邻窗并只对
`112.5°/127.5°` 做 MBAR overlap 检查。

空白区两窗 MBAR 通过：`112.5°→127.5°` overlap `0.0759`，反向 `0.0949`，均高于
`0.03`，但去相关后分别只有 5/10 与 4/10 帧。因此 pilot 在此停止，不拼接为 PMF。
正式采样计划已冻结为 `protocols/EXP-011_umbrella_sampling_plan.json`，内部 SHA-256
`1cd78aba12f15b52a52f27b5f6c8980544843887ebc2cf84d1b5bc12660c6912`：24 个 15°
bin 中心、`k=100`、每窗 50 ps burn-in + 100 ps sampling、每 1 ps 报告一帧、3 个
replicate，分别从历史 run1/run2/run3 的不同困难窗口轨迹末帧开始。每窗记录初始轨迹路径
与文件哈希。`scripts/run_exp011_umbrella_grid.py` 提供 fail-closed 断点续跑，默认只运行一个
pending window；formal_run1 dry-run 已通过，相关 CLI/controller 回归为 `21 passed`。
当前只批准 formal_run1 单窗正式 smoke，尚未启动 24×3 批量采样。

formal_run1 首个正式窗 `-172.5°` 已通过：100/100 帧有限，temperature
`297.74–302.74 K`，相对中心的周期角偏差 `-8.82°` 至 `+21.92°`，最大 umbrella
energy `7.32 kJ/mol`；初态轨迹 SHA-256 为
`47d2fca0d4189ec7eb5d5e6743406162494cc1aaf25ed4a6aeae6d0c75df3b11`，与 runner
记录一致。断点续跑 dry-run 正确识别 `completed=1`、`pending=23`。因此批准 formal_run1
继续剩余 23 窗；formal_run2/3 尚未批准启动，PMF/WP-5 仍未开始。

formal_run1 随后完成 24/24 窗，共 2400 个有限原始帧。审计发现旧 reweighter 以有向可达
图判 `qualified_for_pmf_input`，比冻结的逐邻窗 mutual-overlap 口径宽松；实现已改为
`min(O_ij,O_ji)` 并要求每个观测邻接接口通过，报告 schema 升至 v2，回归测试
`58 passed`。旧宽松报告保留为 `formal_run1_mbar_legacy_directed.json`。v2 严格报告
`formal_run1_mbar.json` 判定 `qualified_for_pmf_input=false`：唯一失败接口为
`112.5°↔127.5°`，mutual overlap `0.0110 < 0.03`；`127.5°` 的 100 帧因
`g=15.789` 只保留 7 帧。未查看或拟合正式 PMF，formal_run2/3 继续暂停。

针对该预注册门触发的补采已另行冻结为
`protocols/EXP-011_augmentation_001_p127p5.json`，内部 SHA-256
`4107c6719c68657f7a50ae730e6fdc8a212117456d31c31e8cde96dac82632ed`：仅补
`127.5°`，从原正式窗轨迹末帧开始，50 ps burn-in + 500 ps sampling，seed
`2026091200`。合并后必须同时满足该状态至少 25 个去相关样本、失败接口 mutual overlap
至少 `0.03` 及全部 24 个周期邻接接口通过；否则继续停止。

EXP-011-AUG-001 已执行并完成，但未通过冻结验收。补采报告
`formal_run1_supplement/center_p127p5_v1/report.json` 含 500/500 个有限帧，所有参数、seed
与初态轨迹哈希均匹配；angle 覆盖 `77.50°–169.78°`，temperature
`296.85–303.88 K`。补采自身 `g=33.120`，仅保留 15/500 个去相关样本；与原
`127.5°` 窗的 7 个样本合并后为 22，低于冻结门 25。严格 v2 合并报告
`output/outer_lambda_exp011/formal_umbrella_v1/formal_run1_mbar_post_aug001.json`
（文件 SHA-256 `1522c49af5cd5730506196967f9a05d9202ae4fd43dab6c3492a773bfc97771d`）
仍为 `qualified_for_pmf_input=false`：图整体 mutual-connected，但唯一失败接口
`112.5°→127.5°` 的 mutual overlap 仅从 `0.0110` 提升至 `0.02353`，仍低于 `0.03`；
反向值为 `0.10696`。目标加权数据文件 SHA-256 为
`c51a9cf1142b809bb6ecb35250d2b89dd8a6e31f43dc76a1d3d970a84aa0f9d0`。

最终状态记为 `EXP-011-AUG-001 COMPLETED_NOT_ACCEPTED / FORMAL_RUN1_OVERLAP_FAILED`。
按用户要求在此停止：不执行第二次补采，不启动 formal_run2/3，不拟合或查看正式 PMF，
不进入 NVT、WP-5 或 production。现有数据与失败报告全部保留用于后续重新预注册。

若另行恢复 MACE 教师，必须登记为独立实验，并至少要求：完整残基与 backbone
buffer；所有共价截断边界封端或由明确局部 readout 排除；对多个环境半径执行能量/配体力
收敛检查。该 MACE 支线不与 EXP-011 的主问题混合。

---

## 11A. EXP-012：CV-free 通用局部残差路径势

当前状态：`PLANNED / PREREG_DRAFT`。本节登记的是当前预注册方向，不是完成结果。
DEC-019 supersede DEC-018 的 XED 主路线，但不删除 DEC-018 或既有草案哈希。

### 11A.1 唯一研究问题

> 在保留完整 MM/softcore/PME 基线和物理端点、不人工指定慢 CV 的前提下，一个局部
> 坐标残差势能否降低困难 alchemical 窗口的相邻态双向 energy-gap 方差，并在计算成本
> 计入后提高 mutual overlap 和 ESS/GPU-hour？

本实验不回答 MACE latent 是否是真实电子密度，也不把残差解释为物理 Pauli/exchange 势。

### 11A.2 Hamiltonian 与训练目标

\[
H_k^*=H_k^{\rm MM}+A_kB_\theta^{\rm local},\qquad A_0=A_K=0.
\]

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

第一轮固定全局 \(A_k\)，禁止与网络联合调节；使用完整 MM target-state ledger、MBAR/target
权重和 whole-run folds。ESS、BAR mutual overlap 与 ESS/GPU-hour 只在 held-out 数据和
独立运行上判决。

### 11A.3 表示消融

| Arm | 输入 | 目的 |
|---|---|---|
| A | typed atom-centered RBF/contact | 最低复杂度基线 |
| B | 轻量等变 ligand-environment cross encoder | 判断通用角向/多体表示是否足够 |
| C | frozen-MACE node latent + invariant MLP | 判断 pretrained latent 是否有增量 |
| D | XED-inspired field（可选） | 仅测量额外物理启发信息，不定义方法身份 |

Arm C 新建 `MaceLatentBasisAdapter`，不得读取最终 interaction energy，不执行
`E(complex)-E(ligand)-E(environment)`。必须冻结 MACE layer、channel 类型、ligand-node
pooling、环境图 cutoff/receptive-field buffer、PBC 和支持域。

### 11A.4 在线与蒸馏路线

- **L1：在线 frozen MACE。** encoder、latent readout 和 \(B\) 位于同一 Torch autograd
  图中；ligand/environment 力均来自 \(-\nabla B\)。离线 NumPy descriptor 不得冒充
  production 可导特征。
- **L2：latent teacher + 轻量 student。** MACE 只离线提供表示诊断或蒸馏目标，production
  运行轻量等变 student。若 L1 的 MACE forward/backward 成本抵消 ESS，优先 L2。

两路线必须使用相同 frame folds、gap loss 和判决门，完整报告 OpenMM step 成本。

### 11A.5 实施和资格顺序

1. 补齐三条 scratch run 逐帧五态 target ledger，并核对坐标、状态顺序和 energy hash；
2. 将现有 `exp012_xed` draft 迁移/别名为通用 `local_residual` schema，保留旧哈希；
3. 冻结 A/B/C/D、MACE layer、图边界、readout、(A_k)、训练预算和数值门后 seal；
4. 完成 whole-run holdout 与至少 3 个训练 seed；
5. 对 ligand/environment 坐标分别做 autograd/finite-difference force check；
6. TorchScript/OpenMM Reference、XML round-trip、端点和 CPU/CUDA 一致；
7. 单困难窗口短 NVT 与至少 3 个独立 production 重复。

### 11A.6 WP-5 通用性验收

先在 Atenolol 困难窗口比较基础路径、仅 λ 重排、Arm A 和最优局部残差。通过后，将同一
架构、cutoff、loss 权重、训练预算、外层 envelope、安全门和验收门应用到至少两个额外
ABFE benchmark 体系；不人工指定 CV，但允许按统一 pilot-ledger 流程逐体系重训权重。

每个额外体系至少 3 个独立重复，并要求：

- 全局端点严格等于 MM；
- \(|\Delta G_{\rm residual}-\Delta G_{\rm converged\,MM}|
  <\max(0.5\ {\rm kcal/mol},2\sigma_{\rm combined})\)；
- mutual overlap 与 ESS/GPU-hour 通过 sealed 改善门；
- 不出现体系专属 CV、超参数或验收阈值调整。

实验误差 `<1.0 kcal/mol` 只作次级报告。TYK2/CDK2 属于 RBFE 使用场景，必须先完成 RBFE
Hamiltonian、mapping 和 ledger 接口资格，不能直接作为当前 ABFE EXP-012 的首轮硬门。
“同协议逐体系重训”与“同一冻结权重零样本迁移”必须分开报告。

### 11A.7 晋级与停止

任一情况停止：所有表示均无 held-out gap/overlap 增益；Arm C 对 A/B 无增量；在线 MACE
成本抵消 ESS；student 不保留收益；图边界/环境半径不收敛；非守恒力、端点漂移、CUDA
不一致或短 NVT 不稳定；需要未预注册地修改超参数或门限。

### 11A.8 已完成工程预检与当前下一步

2026-08-03 已将关闭的 EXP-010/011 实现迁入
`archive/outer_lambda_exp010_exp011_legacy.py`，主模块只保留 lazy 兼容入口；新增
`exp012_xed/schema.py` 和 `protocols/EXP-012_preregistration.json`。草案已登记三条
500-frame scratch run、真实 SHA-256、五态 λ/f_k 和三个 whole-run folds，相关 CPU 回归
为 58 passed。

随后已经完成三条 `hard_window0_run1/2/3` 的统一 CUDA ledger，每条 500 帧。三条重建
System SHA 均逐位等于 scratch SHA `be34fd38...8144`，arrays SHA 分别为
`fd2985dd...071fa`、`44c813e1...99ad7`、`4c0e35b8...6e689`；Group 1 IBS LSE 最大闭合误差
均为 `2.84e-14 kJ/mol`，force-group 最大闭合误差不超过 `3.77e-4 kJ/mol`。
run1 的 500-frame CPU/CUDA 审计通过：相邻 gap 最大差 `6.43e-6` reduced，未归一
importance log-weight 最大差 `1.29e-5`。机器报告为
`output/outer_lambda_exp012/mm_ledger_audit.json`，状态 `PASSED`。

draft 已升级为 `exp012-local-residual-prereg-v2`，主导入命名空间为 `local_residual`；旧
`exp012_xed` 保留作 DEC-018 兼容/哈希证据。这些结果只证明输入 ledger 与方法协议骨架，
不证明 feature、模型或 Force。当前唯一允许的下一步是冻结 A/B/C/D 表示、MACE layer、
图边界、readout、全局 (A_k)、训练预算和数值门并 seal；在此之前不得训练正式模型、
跑 production NVT、启动 WP-5 或修改 production 模块。

### 11A.9 DEC-021：provisional 环境/mapping 生成工具（尚未在真实数据执行）

此前 `local_residual/environment.py`、`local_residual/atom_mapping.py` 只有校验器，没有
生成器；真实 Atenolol 的 `provisional_mace_environment_manifest.json` 和
`provisional_mace_atom_mapping.json` 因此连输入草稿都不存在。新增两个脚本：

- `scripts/discover_exp012_environment_config.py`：给定 topology、一个或多个参考帧
  （每条 `--trajectory` 一帧，默认取最后一帧）、ligand 索引和口袋 cutoff，发现完整
  残基环境候选并组装 `exp012-environment-config-v1` config。核心函数
  `discover_complete_residue_environment` 对每个候选残基始终返回整条残基的全部原子，
  绝不像 EXP-010 的教师选择那样只取半径内的原子子集；同时对每个候选标记
  `is_chain_terminal`，供后续判断是否需要 capping/receptive-field buffer。支持多帧
  取并集（任一帧命中即入选），比单帧更保守，避免因单帧巧合漏掉真正邻近的残基。
- `scripts/build_exp012_atom_mapping.py`：对已 sealed 的环境 manifest 生成
  canonical topology/local-graph/MACE-node 三重映射，是
  `local_residual.atom_mapping.build_atom_mapping` 缺失的 CLI 入口。

新增单元测试 `tests/test_exp012_environment_discovery.py`（合成 mdtraj topology，专门
验证"部分重叠残基必须整条返回"这一 EXP-010 修正点、水/配体排除、chain-terminal 标记、
多帧并集、以及组装结果能通过 `build_environment_manifest` 校验）和
`tests/test_exp012_atom_mapping_cli.py`（CLI 子进程测试，覆盖成功路径与 manifest SHA
不匹配的 fail-closed 路径）。本机默认 Python 环境没有 numpy/mdtraj（`local_residual`
包 `__init__` 会经由 `ledger_audit` 传递依赖 numpy），因此这批代码在本机只做过
`py_compile` 和人工审阅。

用户已在 `openmm_dev` conda 环境实际验证：两个测试文件 10/10 passed（修正一处测试自身
的 bug，`load_atom_mapping` 需要文件路径而非已解析的 dict；脚本本身无需改动）。随后对
真实 Atenolol 数据执行发现流程（`output_lrc_fix/topology.cif`、`hard_window0_run1/2/3`
末帧、`pocket_cutoff_nm=0.5`），得到 21 个完整残基、339 个环境原子、0 个 chain-terminal，
加 41 个 ligand 原子共 380 节点，元素集合 H/C/N/O/S。两份 provisional 文件已生成并冻结：
`output/outer_lambda_exp012/provisional_mace_environment_manifest.json`（sha
`ffa52ebc38508fac929e3989252ec518f873ea3404431154744d1dd94b254f05`）与
`output/outer_lambda_exp012/provisional_mace_atom_mapping.json`（sha
`2c74e5e087c472ef07a3d964a3cdc515652a52aa58eb8c22d287fa623f86597a`，`manifest_canonical`
ordering）。见 DEC-022。IMPLEMENTATION_PLAN §13 对应条目已勾选。

这两个脚本只是生成/组装工具，不做任何科学决策；`pocket_cutoff_nm=0.5` 和三条 run 末帧
并集是本轮**采用值**，不是消融后的最优值——尚未比较过其它 cutoff 或参考帧选择。
Arm A/B/C/D 精确 schema、MACE layer/图边界/readout、训练预算与随机种子、数值判决门这
9 个 preregistration `unresolved` 字段仍然没有值——它们是有后果的设计选择（错误的
cutoff 或图边界会浪费 GPU-hour，过早 seal 等于让预注册失去意义），仍需要用户决定后
再写入 preregistration 并重新哈希、seal。

上述 DEC-022 径向选择及两份 SHA 随后已由 DEC-023 作废；DEC-023 提出的 1.5 nm 径向球
修复又由 DEC-024 撤销。它们保留在本节仅用于记录失败链，不代表当前执行协议。

### 11A.10 DEC-024/025：精确两跳闭包与 Atenolol CPU 真帧通过

当前协议不再用 ligand-centered 12 Å 球。discovery 与 runtime validator 共用
`local_residual.mace_graph.topology_n_hop_closure`，对 6 Å PBC 邻接图逐层扩展：

```text
S0 = ligand
S1 = N_6A(S0)
S2 = N_6A(S1)
S  = S0 union S1 union S2
```

`run1/frame0` 的原子级闭包为 1538 个原子，其中 hop 0/1/2 分别为 `41/343/1154`；触及的
环境按完整残基收口后，固定图包含 41 个 ligand 原子和 2094 个 environment 原子，共
2135 节点、155624 条 6 Å 有向边。12 Å 仅记录为两层最远几何上界。

OMOL C0 报告显示模型有 3 个 product layers。两层候选因此固定为 zero-based
`product_layer_index=1`，在第二个 product hook 后 early-stop，不执行第三层；从该层
`1024x0e+1024x1o+1024x2e` 输出中只取开头 `1024x0e` 标量块。构边改为
`chunked_no_n_by_n_allocation`，每批最多 100000 对；repeat reference 不保留 autograd graph，
正式 smoke 只保留一份可导 forward，并使用 float32。

第一次旧实现运行导致 96 GB RAM OOM，未生成完成报告。旧实现同时存在全配对构边、完整
三层 float64 OMOL 和两份可导 forward，因此只登记为组合工程失败，不能在没有分项测量时
断言某一个分量单独耗尽全部内存。

修正后的 CPU smoke 已完成：

| 项目 | 结果 |
|---|---:|
| 状态 | `COMPLETED_AUTOGRAD_SMOKE_ONLY` |
| 模型 | `MACE-omol-0-extra-large-1024.model` |
| 模型 SHA-256 | `9b64b4fd5153ca578c694abc57806d8111050de6ff652e695c9b525bc4d36469` |
| dtype / device | `torch.float32` / CPU |
| latent shape | `[41,1024]` |
| latent min / max / norm | `-0.837964 / 0.864419 / 16.308716` |
| ligand gradient norm | `32.219711` |
| environment gradient norm | `22.279121` |
| repeat max absolute difference | `0.0` |
| MACE parameter gradient count | `0` |
| 总耗时 | `172.713 s` |

身份与报告：

| 产物 | canonical/report SHA | 文件 SHA-256 |
|---|---|---|
| environment manifest | `0e399a9ec03c0c8c397bacdbf1c53032be8fb65fd3248c3747a577e88e223935` | `9c9cd4a062f6b6787ea4a86b1be556ffb23ee6e0a92277182bbe5254f764b95b` |
| atom mapping | `d84a65f9165bc89980e7564850eb48b7e8d42326f1edd707edccdbdc3e88d00f` | `355b48e16037157d44df57c9f28afd350c4c2b9e3e9bd50aca46f2f2e21a824c` |
| OMOL C0 report | `83ee5ab57057fb2559c8593804680e551e9dd72bd1d50d617bd238f18b8d5f1a` | `ca0352fcc5255a546cdf613b97d519a7763f37b7f3edb22f005f45f8049482c5` |
| CPU smoke | `ce8fd06cad0e4bdb8b65589c8d5eece5e7718c8174932d9d465817348ed58db5` | `2a27c24fae50ea7d09d3328cd00d8a43c8c326e4d1a1026da0822bae2e4bd8fe` |

报告路径为
`output/outer_lambda_exp012/two_hop_frame0/cpu_omol_latent_smoke.json`。policy 明确记录
`training_executed=false`、`full_dataset_scanned=false`、`scientific_qualification=false`、
`fragment_subtraction_used=false` 和 `latent_detached=false`。因此结论只到
`C1_REAL_FRAME_CPU_PASSED`；下一门是相同 frame、manifest、mapping、product layer 与 dtype
的 CUDA 对照，不允许据此开始训练或宣称 production 可用。

### 11A.11 CUDA 对照尝试与显存瓶颈（尚未通过）

当前状态：`BLOCKED_ON_VRAM`。DEC-025 的 CUDA float32 对照门尚未通过；没有生成
`cuda_omol_latent_smoke.json`，不构成失败判定，只是被显存容量卡住。

运行前发现并修复了两处代码问题（均在本轮对照准备中发现，不是 CPU smoke 遗留缺陷）：

1. `local_residual/mace_latent.py` 的 `--model` 身份校验对 `~` 路径不做展开比较修正——
   实为调用命令用了 `~/.cache/mace/...` 而 C0 报告冻结的是绝对路径
   `/home/ruigengji/.cache/mace/...`，属于命令用法问题，已改用绝对路径，非代码缺陷；
2. `MaceLatentBasisAdapter.forward`（`mace_latent.py:222-225`）用
   `torch.device(self.device_name)` 构造 `expected_device`；当 `--device cuda`
   不带显式 index 时，该对象的 `index=None`，而实际张量在 CUDA 上创建后
   `.device` 总是解析出具体 index（如 `cuda:0`），两者按 PyTorch 语义不相等，
   导致每次不带 index 的 `--device cuda`（即本门要求的形式）都会在真正跑到
   forward 之前被 `"graph dtype/device differs from the explicit adapter contract"`
   拒绝。修正为：当 `expected_device.type=="cuda"` 且 `index is None` 时，归一化为
   `torch.device("cuda", torch.cuda.current_device())` 后再比较（真实代码缺陷，已修复）。

代码修正后连续三次 CUDA 尝试，均在 product-layer-1 张量积内部的同一个
`cat`/`reshape` 操作处 OOM：

| 尝试 | GPU / 总容量 | 已修正的分配器改动 | 触发分配 | 已占用 | 结果 |
|---|---|---|---:|---:|---|
| 1 | 15.47 GiB（GPU 0，另有进程 96% 占用） | 无 | 9.50 GiB | 14.59 GiB | OOM |
| 2 | 10.57 GiB | 无 | 2.97 GiB | 7.89 GiB | OOM（更早的更小算子，未到达尝试 1/3 的算子） |
| 3 | 23.58 GiB（空闲 RTX 3090，物理 24576 MiB） | `torch.cuda.empty_cache()`（no-grad 参考 forward 后）+ `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 9.50 GiB | 14.57 GiB | OOM |

尝试 1 与尝试 3 的已占用显存几乎相同（14.59 对 14.57 GiB），且尝试 3 已经加上
`empty_cache()` 与 `expandable_segments` 缓解措施后数字仍未变化。据此排除
"显存碎片化"作为原因；这是该图（2135 节点、155624 有向边，其中两跳精确闭包
1538 个原子、另 597 个原子来自整残基收口）在 `MACE-omol-0-extra-large-1024`
product-layer-index=1、float32 下的真实显存需求，且需求 `>= 14.57+9.50≈24.07 GiB`，
超过尝试 3 这张 23.58 GiB 卡约 0.5 GiB；这只是遇到的第一个大算子，不能排除
之后还有更大的峰值。

未采纳/未执行的选项（等待用户决定，不擅自选择）：

- 使用更大显存的设备（32/48/80 GiB 级别）；
- 对已执行的两层 interaction/product block 做 gradient checkpointing，用重算换显存；
- 缩小两跳图（原子数）或更换更小的 OMOL 模型——均会改动已冻结的 DEC-024 支持域定义
  或模型选择，不在本次对照的授权范围内。

| 项目 | 路径/说明 |
|---|---|
| 代码修正 1 | `local_residual/mace_latent.py:223-225`（cuda 设备等价性归一化） |
| 代码修正 2 | `scripts/smoke_exp012_mace_latent.py:191-193`（no-grad 参考 forward 后 `torch.cuda.synchronize`+`empty_cache`） |
| CUDA 报告 | 未生成；三次尝试均以 `RuntimeError: CUDA out of memory` 终止，未落盘 JSON |

### 11A.12 Arm A/B/D 退役（`not_pursued`，DEC-039）

| Arm | 输入 | 实施状态 | 判定 |
|---|---|---|---|
| A | typed atom-centered RBF/contact | 从未编写任何代码 | `not_pursued` |
| B | 轻量等变 ligand-environment cross encoder | 从未编写任何代码 | `not_pursued` |
| C | frozen-MACE node latent + invariant MLP | 已实现并在真实数据上通过（DEC-034/035/036） | 保留、进入 D1 |
| D | XED-inspired field（可选） | 从未编写任何代码 | `not_pursued` |

判定用词说明：这里用 `not_pursued`，不用 EXP-010 的 `FAIL`。EXP-010 六个 cheap-CV
候选是真的跑了完整 leave-one-run-out 数值评估、数值上全部跑输；Arm A/B/D 从未被
构建或执行，无法说"跑输了"——两者是不同性质的判定，不能套同一个判定词。

| 证据 | 说明 |
|---|---|
| Arm C 教师侧 readout report_sha256 | `d77a8e132780270363abb4a33572912e518c102ef8c1f4ed38d36df92c7b05c3`（DEC-034/035/036 登记结果：3 折全部改善，均值 44.6%） |
| Arm C join report_sha256 | `8dfc47e3352534f8b67826ee570f6830de2b618cf935ff9354e74f0082c016ce` |
| 预注册文件 | `protocols/EXP-012_preregistration.json`，本条 DEC 生效后 `freeze.status="sealed"` |

**决定**：Arm A/B/D 从未实现过任何代码，正式标记为 `not_pursued`（不是 `FAILED`——
它们没有被跑过，无法说"跑输了"）。明确记录预注册偏离：
`protocols/EXP-012_preregistration.json` 的
`decision.arm_C_increment_comparisons=["C_vs_A","C_vs_B"]` 从未执行；实际只做了
"C vs B=0"（无残差项基线）对照，这个 `B=0` 基线不是 Arm B。D1 计划中的
direct-gap vs distilled 对照只验证"student 是否保留了 teacher 信号"，不能证明
Arm C 优于任何其它架构或简单特征基线。因此当前结论收窄为：MACE latent 表示存在
可泛化的 gap-variance 下降信号、值得蒸馏为在线 student；不得声称 Arm C 相对
Arm A/B 有验证过的表示优势。若未来需要"Arm C 表示优势"这一论文级结论，须作为
新的、独立预注册的对照实验补做，不阻塞当前 D1 蒸馏工作。

这个退役不能归因于"Arm A/B 被验证过更弱"——它们根本没有被验证过；退役的唯一
理由是 Arm C 已经在教师侧拿到足够强的 go 信号（3/3 fold 改善，均值 44.6%，超过
DEC-030(c) 最初"至少一折改善"的最低门槛），且继续扩大表示消融的边际信息价值
低于把这份资源投入 D1 蒸馏本身；这是资源优先级判断，不是科学证伪判断。

---

## 11B. EXP-016：transition-segment attribution / temporal audit

### 11B.1 输入与可行性结论

本实验不新增 MD。审计 `hard_window0_run1/2/3` 三条连续 scratch trajectory，各 `500` 帧，总计 `1500` 帧；由 `sample_report` 读取的实际保存间隔为 `1 ps`。trajectory、MM ledger 和 teacher latent 的 frame identity、长度与输入哈希均通过 manifest 校验。

数据中没有离散 alchemical state、replica exchange history 或可独立标注的 physical basin crossing。五态 MM target-energy ledger 只允许定义 energy-weighted surrogate：对每帧取 `argmin(target_interaction_kj_mol - f_k)`，以相邻 label `0↔1` 改变作为 surrogate event；首个 horizon 共 `114` 个相邻改变。它不是 physical crossing、dominant-component switch 或 replica round trip。

### 11B.2 时间审计与 LORO

候选包含 teacher latent PC1/norm、primary/secondary torsion、VAL251 chi1、student direct-gap scalar 和 adjacent state-gap。按连续 run 做 leave-one-run-out，horizon 固定为 `1/5/10/25 ps`，不做随机 frame split。held-out run 使用对应的 seed-0 direct-gap checkpoint：run1 模型只评估 run1，run2/3 同理，训练只用另外两条 run。早先把 run1 checkpoint 复用到全部 run 的报告已标记 `INVALIDATED`，不作为证据。

每条 run 的 IAT/有效样本数与 raw frame count 同时报告；attribution 使用 `128` 帧（`128 ps`）circular contiguous block bootstrap、`2000` 次重复。prediction AUC/AUPRC/Brier 仅是 surrogate exploratory point estimates；direct-gap student 和 adjacent-gap 是 target-derived diagnostics，不能作为独立 event-prediction 证据。hydration 没有逐帧 cache，force(signal) 也没有缓存，因此两者不重建、不静默推断。

### 11B.3 裁决

`EXP-016` 状态为 `INCONCLUSIVE / SURROGATE_ONLY`。本次未发现足以定义 physical learned slow information 的证据，不定义 physical `τ_information`，不启动 cheap Hamiltonian、online TorchForce 或 MTS promotion。后续唯一允许的方向是补充独立 physical/overlap event history，或另行预注册一个 cheap offline route；不能把本次 surrogate 预测结果写成 physical crossing 预测。

证据：`output/outer_lambda_exp016_loro/EXP-016_data_manifest.json`（manifest input hash `db5cbf8e30b57353f9324cb2ea5653d11f05904d6ddddf888b778acf1a4667e7`）、`output/outer_lambda_exp016_loro/EXP-016_temporal_audit.json`（SHA-256 `d1c5d4de6a14b985acf6e2cafd42dab5a345cd8e12ea86ec45bf053aa43674c`）、`output/outer_lambda_exp016_loro/EXP-016_summary.md`。

## 12. 单次实验模板

复制本节，并将标题改为实际实验编号。

## EXP-XXX：实验标题

### 12.1 基本信息

| 字段 | 值 |
|---|---|
| 状态 | PLANNED |
| 开始时间 | |
| 结束时间 | |
| 执行者 | |
| 对应工作包 | |
| 输出目录 | |
| 主机/GPU | |

### 12.2 唯一研究问题

> 待填写。

本实验不回答：

- 待填写；
- 待填写。

### 12.3 预注册假设

成功假设：

> 待填写。

失败假设：

> 待填写。

### 12.4 预注册验收门

| 验收项 | 阈值/条件 | 硬门 |
|---|---|---|
| 端点能量 | | 是 |
| 端点力 | | 是 |
| 非有限能量/力 | 0 次 | 是 |
| 最大附加力 | | 是 |
| ESS/GPU-hour | | 是 |
| 目标慢变量转换 | | 待定 |
| 端点 ΔG 一致性 | | 是 |

验收门必须在运行前填写，不能在看到结果后修改。

### 12.5 输入、模型和版本

| 项目 | 路径/版本 | SHA-256 或标识 |
|---|---|---|
| 代码 | | |
| 配置 | | |
| system XML | | |
| topology | | |
| 初始坐标 | | |
| 神经模型 0 | | |
| 神经模型 1 | | |
| 原子选择 | | |
| λ schedule | | |
| 外层系数 | | |

软件环境：

| 软件 | 版本 |
|---|---|
| Python | |
| OpenMM | |
| OpenMM-Torch | |
| OpenMM-ML | |
| MACE | |
| PyTorch/LibTorch | |
| CUDA driver/runtime | |

### 12.6 实际命令

```bash
# 必须填写实际执行命令。
```

### 12.7 Hamiltonian 定义

基础路径：

\[
H_\lambda^0 =
\]

神经路径：

\[
B_\lambda =
\]

包络：

\[
w(\lambda) =
\]

系数摘要：

| λ | \(A_{\lambda,0}\) | \(A_{\lambda,1}\) | 备注 |
|---:|---:|---:|---|
| | | | |

### 12.8 运行完整性

| 检查 | 结果 | 证据路径 |
|---|---|---|
| 输入哈希匹配 | | |
| 模型哈希匹配 | | |
| checkpoint 恢复 | | |
| production `f_k` 锁定 | | |
| target/bias/base 帧数一致 | | |
| 非有限帧 | | |
| 能量查询失败 | | |
| 轨迹完整 | | |

### 12.9 端点与力学

| 指标 | λ=0 | λ=1 | 阈值 | 结论 |
|---|---:|---:|---:|---|
| 最大能量差 kJ/mol | | | | |
| RMS 力差 kJ/mol/nm | | | | |
| 最大原子力差 kJ/mol/nm | | | | |
| 有限差分相对误差 | | | | |

### 12.10 神经能量与力

| 指标 | Basis 0 | Basis 1 | 总路径项 |
|---|---:|---:|---:|
| 能量均值 kJ/mol | | | |
| 能量标准差 | | | |
| 能量 P95 | | | |
| 最大绝对能量 | | | |
| 力 RMS kJ/mol/nm | | | |
| 力 P95 | | | |
| 最大力 | | | |
| 支持域违规次数 | | | |

### 12.11 采样与自由能

| 指标 | 基础路径 | 仅 λ 重排 | 神经路径 | 单位/备注 |
|---|---:|---:|---:|---|
| ΔG | | | | kJ/mol |
| 报告误差 | | | | kJ/mol |
| importance ESS ratio | | | | |
| absolute ESS | | | | |
| 去相关样本数 | | | | |
| round trip | | | | |
| 慢变量转换次数 | | | | |
| 自相关时间 | | | | |
| ns/day | | | | |
| GPU-hour | | | | |
| ESS/GPU-hour | | | | |

### 12.12 构象与稳定性

| 检查 | 基础路径 | 神经路径 | 结论 |
|---|---:|---:|---|
| 异常键长 | | | |
| 原子重叠 | | | |
| 配体逃逸 | | | |
| 口袋异常塌缩 | | | |
| 水/离子异常占位 | | | |
| NaN/积分器失败 | | | |

### 12.13 性能分解

| 项目 | 时间/显存 |
|---|---:|
| Context 创建时间 | |
| 基础 MD step 时间 | |
| 神经基势推理时间 | |
| probe energy 时间 | |
| 总 step 时间 | |
| 主 Context 显存 | |
| probe Context 显存 | |

### 12.14 观测事实

- 待填写；
- 待填写；
- 待填写。

这里只写直接观测，不写原因推测。

### 12.15 解释

- 待填写；
- 待填写。

必须指出哪些解释仍是推断。

### 12.16 验收结论

| 验收项 | PASS/FAIL | 依据 |
|---|---|---|
| 端点 | | |
| 力学 | | |
| 能量账本 | | |
| 稳定性 | | |
| 采样改善 | | |
| ESS/GPU-hour | | |
| 自由能一致性 | | |

总体状态：

```text
PASSED / FAILED / INCONCLUSIVE
```

### 12.17 决策

- [ ] 进入下一工作包；
- [ ] 保持复杂度，补充独立重复；
- [ ] 调整外层系数后重试；
- [ ] 重新训练同一目标基势；
- [ ] 回滚到基础路径；
- [ ] 停止该方向；
- [ ] 其它：待填写。

决策理由：

> 待填写。

### 12.18 后续行动

| 行动 | 负责人 | 截止条件 | 状态 |
|---|---|---|---|
| | | | |

---

## 13. 跨实验汇总

至少三个可比独立重复完成后填写：

| 方案 | 重复数 | ΔG 均值 | 重复间 SD | ESS/GPU-hour | 慢变量转换 | 异常率 |
|---|---:|---:|---:|---:|---:|---:|
| 基础路径 | | | | | | |
| 仅 λ 重排 | | | | | | |
| 单神经基势 | | | | | | |
| 多神经基势 | | | | | | |

## 14. Production 准入

- [ ] 禁用神经路径时旧行为回归。
- [ ] 端点能量和力严格一致。
- [ ] target/bias/base 账本闭合。
- [ ] 公共 λ 状态跨窗口一致。
- [ ] 模型和系数进入缓存指纹。
- [ ] GPU 稳定性测试通过。
- [ ] 至少 3 个独立重复。
- [ ] ESS/GPU-hour 优于基础路径。
- [ ] 收益不能被仅 λ 重排替代。
- [ ] complex/solvent 端点循环一致。

最终决定：

```text
NOT_READY
```

决定日期：

```text
2026-08-03
```

依据：

> 独立端点和账本契约已通过，但 EXP-010 fragment teacher、EXP-011 torsion-PMF 均已失败；
> EXP-012 已升级为 CV-free 通用局部残差路线；五态 ledger、输入哈希、whole-run folds 和
> backend audit 已完成，但 preregistration 仍为 draft，尚缺 sealed A/B/C/D 协议、表示消融、
> 守恒力、NVT 和跨体系性能结果。因此没有候选可进入
> WP-5 或 production。

---

## ORB-000：ORB-v3 shallow representation preflight

**日期：** 2026-08-09  
**状态：** `ORB-000 CLOSED / ORB-001 PRIMARY_STATISTICAL_GATE_PASSED / ORB-003_NOT_STARTED`  
**研究问题：** 在不调用 ORB 总能量头的前提下，真实 Atenolol 困难窗口的 6 Å 局部图能否稳定产生预注册的 layer-2、256 维 ligand latent？

### 观测事实

- 本地 `orb-v3-conservative-omol` 权重已加载；checkpoint SHA-256 为 `c284e99c45df928ae28443fb27223188cc2c33cced593488a4d28595e75cb6e8`。模型暴露 5 个 GNS blocks，显式 layer-2 prefix 的 ligand latent 为 `(41, 256)`，全部 finite。
- 现有 `system_native.xml` 的全体系电荷为 `2.0000014e-08 e`，41 原子 fragment 的电荷为 `1.999999996e-08 e`；在 `1e-6 e` 容差内与父体系中性 contract 一致。冻结的 contract 是 `orb-parent-system-charge-spin-v1`：scope `parent_full_system`、`Q=0`、`M=1`，解释为父体系闭壳层 singlet conditioning，不是截断 L-hop fragment 的电子 multiplicity。OpenMM XML 缺少 multiplicity 字段不再构成 blocker。
- run1/run2/run3 的 frame `0/250/499` 共 9 个样本完成 CPU float64、6 Å、120-neighbor 的精确 L-hop 审计。L2 node 数为 `1508–1564`、edge 数为 `108992–114580`；L5 node 数为 `12818–13277`、edge 数为 `1019844–1064154`；9 个样本均无 cap 命中。
- run1/frame0 的 conditioner-corrected primary layer-2 latent smoke 输出 `(41,256)` 且 finite；raw float32 latent SHA-256 为 `37f3d3801e8ad48dd5a6f9201babdbeca7854cbeb8e3d57c9df321916b9e1e8eeb`，报告为 `orb_latent_smoke_run1_frame0_parent_contract_v2.json`。
- `M=1` primary 对 `M=3` sensitivity 在 9 帧完成：pooled latent 相对 L2 差异均值 `0.0995357`，ligand-node cosine 均值 `0.9952286`，per-dimension std 相对 L2 变化 `0.0240154`。`M=3` 仅 sensitivity，不用于选择 multiplicity；`M=0` 未测试。旧 v1 零差异结果因 prefix 漏传 conditioner，已明确 invalidated。
- ORB-001a 已通过：9/9 个预检帧的 canonical `(topology_sender, topology_receiver, unit_shift)` edge set、edge count、per-node neighbor-count SHA-256 和 120-cap 状态全部一致。正式 adapter 冻结为 `knn_alchemi`、CPU float64 graph construction、float32 output、`wrap=True`、`half_supercell=True`；报告为 `orb001a_edge_equivalence_9frames.json`。
- ORB-001b 已通过：`orb_models=0.6.2`、PyTorch `2.12.0`、CPU、`float32-high`、`compile=False`；checkpoint size `103417970` bytes，SHA-256 为 `c284e99c45df928ae28443fb27223188cc2c33cced593488a4d28595e75cb6e8`。loader wall time `16.3887 s`，1 帧 cold extraction `1.8244 s`，10 帧 warm 平均 `1.1740 s/frame`，scalar backward end-to-end `2.6492 s`，gradient shape `[1538,3]` 且 finite。报告为 `orb001b_initialization_benchmark_1cold_10warm.json`。
- 图规模已经直接否定 L2 约 `200–300` atoms 的成本先验；完整 1500-frame layer-2 cache 的 node 范围为 `1460–1643`、edge 范围为 `104036–123010`，最大邻居数为 `119`，cap-hit 为 `0`。

### 执行入口与输入

```text
scripts/audit_orb_graph.py
scripts/smoke_orb_latent_frame.py
scripts/audit_orb_charge_spin_contract.py
scripts/compare_orb_spin_conditioning.py
scripts/build_orb_latent_cache.py
scripts/check_orb_edge_equivalence.py
scripts/benchmark_orb_initialization.py
scripts/join_exp012_teacher_latent_cache_with_ledger.py
scripts/run_orb_probe.py
```

代表性输出：

- `output/outer_lambda_orb/orb_graph_audit_run1_frame0.json`
- `output/outer_lambda_orb/orb_latent_smoke_run1_frame0_parent_contract_v2.json`
- `output/outer_lambda_orb/orb_latent_smoke_run1_frame0_parent_contract_v2.npz`
- `output/outer_lambda_orb/orb_parent_conditioning_contract_audit_v1.json`
- `output/outer_lambda_orb/orb_spin_conditioning_sensitivity_layer2_9frames_v2.json`
- `output/outer_lambda_orb/orb_spin_conditioning_sensitivity_layer2_9frames_v2.npz`
- `output/outer_lambda_orb/graph_audit_hard_window0_run{1,2,3}_frame{0,250,499}.json`

输入为 `output_lrc_fix/topology.cif`、三条既有 `hard_window_screening.dcd` 和同一组 41 个 ligand topology indices；ORB 模型来自本地 `cached_path` cache。adapter 与图审计均独立于 production ABFE 模块。

### 解释与限制

`total_charge=0` 与现有电荷账本一致到审计容差内；`spin_multiplicity=1` 是显式登记的父体系闭壳层建模假设，不伪装成 XML 测量值。conditioner 已在显式 prefix 中按 ORB 官方路径生成并广播到 node/edge。1500-frame layer-2 cache 和 EXP-012 primary probe 已完成；ORB-003 的 matched-path 成本、scalar basis、对称性/力学和 OpenMM 工作仍未执行。

### 决定

`ORB offline representation probe = GO` 保持；`ORB-001 PRIMARY_STATISTICAL_GATE_PASSED / ORB-003_NOT_STARTED`。layer-2 完整 cache、cache/ledger join 和 EXP-012-compatible primary LOO probe 已完成。3/3 folds 改善，平均 relative improvement `0.3968221`，最差 fold `0.2805260`；三折 baseline→fitted gap-variance 为 `0.448644→0.322787`、`0.269171→0.154311`、`0.393317→0.203257`。报告为 `orb_layer2_exp012_probe_report.json`，join report 为 `orb_layer2_exp012_join_report.json`。该结果只授予 `REPRESENTATION_PROMISING`，不授予 scalar basis、ORB-003 或 OpenMM 资格；L5 仍为 `EXPLORATORY / NOT_PRIMARY`。

## ORB-001：layer-2 primary cache 与 EXP-012 probe

**日期：** 2026-08-09  
**状态：** `PASSED / REPRESENTATION_PROMISING / ORB-003_PENDING`  
**预注册 primary：** `orb-v3-conservative-omol`, layer 2, parent-system `Q=0, M=1`  
**输入：** 三条既有 `hard_window0_run{1,2,3}`，各 500 帧；同一 EXP-012 MM ledger；不重排 frame，不训练 ORB。

### 实际执行与完整性

- `scripts/build_orb_latent_cache.py` 完成三条 run，各 `500` 帧；每帧均执行 CPU float64 closure、官方 `knn_alchemi` edge equivalence、120-cap fail-fast 和 layer-2 prefix extraction。
- 每条 cache 的 `ligand_latent` shape 为 `(500,41,256)`、dtype `float32`，`pooled_latent` shape 为 `(500,256)`，frame index 为 `0..499`。
- 三条 cache 的 node 范围为 `1460–1643`，edge 范围为 `104036–123010`，最大 outgoing neighbor 为 `119`，cap-hit frame count 为 `0`；每个 NPZ SHA-256 均与 cache report 一致。
- cache report SHA-256：run1 `bf74693e315166bbc2e6abbcf715f1bddc90eb7cad55d8631adf3075f65a1e0d`；run2 `7cea3437295934c23e5ff3d1269ee47ae04d5a5cf3650f3a211f20338641818b`；run3 `8188439fb143c996e03ea7b2e7a2e661c25b080d375223a7d7d769baf389a0b7`。
- cache 与 ledger 的 fail-closed join report SHA-256 为 `04f66a7ffc77b44ef9e1319dd185ce8c0990999157d3fe41b24d2dc9f242bb04`。

### 预注册统计门

EXP-012-compatible layer-2 primary probe report：`output/outer_lambda_orb/orb_layer2_exp012_probe_report.json`。

- held-out partition 0：`0.448644 → 0.322787`，relative improvement `0.280526`，ridge `0.01`；
- held-out partition 1：`0.269171 → 0.154311`，relative improvement `0.426717`，ridge `0.001`；
- held-out partition 2：`0.393317 → 0.203257`，relative improvement `0.483223`，ridge `0.001`。

统计门为 `passed=true`：3/3 folds 改善，hard floor 2/3 通过，平均 relative improvement `0.396822`，最差 fold improvement `0.280526`，无 fold 恶化。该数字用于 ORB-001 layer-2 representation qualification；30–50% 不是硬晋级阈值。

### 决定与停止边界

layer-2 primary 标记为 `REPRESENTATION_PROMISING`，允许进入 ORB-003 的真实成本/力学路线。`ORB-004` scalar basis、rotation/torque/cutoff audit、TorchScript/OpenMM 和 NVT 均未启动；完整 ORB 每步在线仍为 `NO-GO` 先验，需真实 matched-path 成本与力学门重新证明。

## DEC-056：`DESIGN_3_FAILED_013B`，转方案①

**日期：** 2026-08-09  
**状态：** `DECIDED / ACTIVE_PRECHECK`  
**对应实验：** EXP-013 方案③，013-B  
**证据：** DEC-055 report_sha256 `64a963626ef36893d440823bd9845ca7c6123cda1b76159e175d9a893810caf3`

### 决策

按 013-B 预注册主门，方案③判定 `DESIGN_3_FAILED_013B`。N=8/16/32 相对 N=1 的
温度和/或 fused IBS 判别式能量出现 `z>3` 的可分辨系统性偏移；温度偏移同时呈随 N
单调升高的 dose-response（约 `+0.66/+1.01/+1.29 K`）。`<1.3 K` 只作为物理量级
补充报告，不能覆盖预注册判据，也不修改 `z_threshold=3.0`。

因此：

- 方案③**不得进入 013-C**，不做其三重复，也不重开或重试方案③；
- 按冻结顺序转方案①：整个 fused Group-1 作为慢组，沿用 DEC-054 修复后的 State API
  初始化，禁止跨 integrator `loadCheckpoint()`；
- 先跑 `N=1/2/4/8` 低成本物理预检；只有无系统性偏移才允许另行考虑 N=16；
- 方案① Qualification gate 未通过后才进入方案②；方案②是新 Hamiltonian，必须先做 N=1 ESS 信号检查；
- 方案②若再失败，转 EXP-014；EXP-016 已完成且不晋级，无后续动作；ORB 只继续
  charge/spin contract audit，不跑 ORB-001 1500-frame probe。

### 已实现入口与运行审计

- 新增 `scripts/check_exp013_design1_mts_precheck.py` 和
  `run_exp013_design1_mts_precheck.sh`；入口固定只构造 `N=1/2/4/8`，报告协议显式
  标记 `N=16/32 未运行`。`smoke` 使用 `16/32` ticks（`0.256/0.512 ps`），只作
  backend/健康诊断；`qualification` 固定 `400/2000` ticks（`6.4/32 ps`），以每
  `50` ticks（`0.8 ps`）连续 block 的 block-mean SEM 判定系统偏移，只有 qualification
  才能设置 `eligible_for_n16_followup=true`。
- 该脚本唯一的 `loadCheckpoint()` 位于同类 `LangevinMiddleIntegrator` source Context；
  MTS Context 只使用 `setPositions/setVelocities/setPeriodicBoxVectors/setParameter`。
- 先前在无 CUDA 节点的尝试因 `CUDA_ERROR_NO_DEVICE` 停止，CPU fallback 因 production
  checkpoint 绑定 CUDA（`loadCheckpoint: Checkpoint was created with a different Platform:
  CUDA`）停止；这些尝试不作为资格证据。随后在兼容 CUDA checkpoint 的节点完成 Smoke：
  `output/outer_lambda_exp013_design1_smoke/report.json`，report_sha256
  `93f5cfbe6e4239c690bb2154524e9daa7d0a922c3b2d6f5f67242911536f0e7e`，
  `COMPLETED_DESIGN1_SMOKE`，CUDA `Precision=mixed`。`N=1/2/4/8` 全部有限且绝对健康，
  State API/Group-1 slow contract 通过；短程普通 SEM 未见偏移，但 `eligible_for_n16_followup`
  仍为 `false`，因为 Smoke 永不授予 N=16 资格。下一步只运行独立 Qualification。

### Provenance 限制

当前工作树未发现 Git 仓库（`git rev-parse --is-inside-work-tree` 在 `/home` 边界停止），
本次只提供文件状态和运行尝试审计，没有 commit/dirty provenance。

## DEC-058：`DESIGN_1_QUALIFICATION_GATE_NOT_MET`，N16 不授权，转方案② N=1 ESS 检查

**日期：** 2026-08-09  
**状态：** `DECIDED / N16_NOT_AUTHORIZED / BRANCH_TO_DESIGN_2`  
**对应实验：** EXP-013 方案① Qualification  
**证据：** `output/outer_lambda_exp013_design1_qualification/report.json`，report_sha256
`2d96b39e4f6571e131cc16fb98ee4a5b645f35b66455d8d53dfbd442ea3d6d9a`

### 结果与裁决

Qualification 确实使用了预注册的 `400` warmup ticks + `2000` monitored ticks，即
`6.4 ps + 32 ps`；固定采样间隔为 `0.016 ps`，连续 `50` ticks（`0.8 ps`）做
block-mean SEM。CUDA `Precision=mixed`，四臂绝对健康门全部通过，State API 初始化和
整个 fused Group-1 slow contract 通过，且没有运行 N=16/32。

但按预注册的单轨迹 Qualification gate，系统偏移子门未通过：

- temperature：N=2/4/8 的 block-aware `z=5.61/5.79/6.83`；
- fused Group-1 energy：N=2/4 的 `z=1.62/1.60`，N=8 的 `z=5.62`；
- `systematic_shift_detected_by_n={2: true, 4: true, 8: true}`，
  `eligible_for_n16_followup=false`。

因此登记三层结论：

- `DESIGN_1_QUALIFICATION_GATE_NOT_MET`；
- `N16_NOT_AUTHORIZED`；
- `PHYSICAL_SYSTEMATIC_BIAS_INCONCLUSIVE`。

这次 Qualification 每个 N 只有一个随机种子，监测时长为 `32 ps`。block-aware SEM
处理了时间自相关，但没有估计跨种子重复间变异；即使种子数相同，不同 MTS 调度在轨迹
分叉后也不构成真正的配对重复。因而，长轨迹上的 `z>3` 足以触发预注册的保守停止规则，
但不等于证明方案①普遍存在物理系统偏差或必然产生错误系综。尤其 N=2/4 的 fused
energy 未越门，只有 N=8 的 fused energy 越门；N=2/4 的主要证据是温度统计显著，不能
单独外推为整个设计的普遍物理失败。

程序性裁决仍然明确：不运行 N=16，不进入 013-C，不事后放宽 block/SEM 或 z 阈值；按
冻结顺序转方案②。方案②是新 Hamiltonian，下一步只做 N=1 ESS 信号检查；若没有信号，
转 EXP-014。方案① Smoke 的通过结果只保留为 backend 和短程健康证据，不改变本裁决。

若以后确实要回答“是否存在可重复物理偏差”，应另立不替换主结果的 confirmatory
sensitivity：至少 3 个独立 seed，以跨 seed 的 `N−N1` 差值和预先冻结的物理等价容差
为判据，而不是继续把单轨迹 `z>3` 当作充分的科学结论。本敏感性分析目前未启动。

## DEC-059：`DESIGN_2_N1_ESS_SIGNAL_FAILED`，EXP-013 不晋级

**日期：** 2026-08-09  
**状态：** `DECIDED / EXP013_NO_PROMOTION / BRANCH_TO_EXP014_CONTINGENCY`  
**对应实验：** EXP-013 方案② independent additive student，N=1 ESS signal check  
**证据：** [`output/outer_lambda_exp013_design2_n1_ess/report.json`](output/outer_lambda_exp013_design2_n1_ess/report.json)，report_sha256 `8727e69c32b24b5de9e0f4e0355582453d3b0dc566424af12a6e6355c39fdd4c`

### 结果

方案②使用一个与 classical IBS Group 1 分离的线性 additive student Force；同一
production checkpoint-derived State、同一 Langevin seed、相同 `10,000` 步 burn-in
和 `50,000` 步监测（每 `500` 步取一帧），共 `100` 帧/arm。student 以
`c1=0.5` 加入 sampled-row `bias_history`，不进入目标态 `u_kn`；这明确是新 sampling
Hamiltonian，不是 DEC-048 fused 设计的等价改写。

| 指标 | baseline classical IBS | independent additive student | 结论 |
|---|---:|---:|---|
| `mixture_ess_proxy` | 47.827779 | 38.798639 | `-9.029140`，相对 `-18.88%` |
| `mixture_ess_proxy_per_gpu_hour` | 932.2718 | 217.9007 | `-714.3711`，仅作辅助报告 |
| 平均温度 | 300.518 K | 300.423 K | 两臂都健康 |
| 最大 additive energy | 0 | 7.133 kJ/mol | 在 sanity 范围内 |
| 最大 additive force norm | 0 | 35.707 kJ/mol/nm | 在 sanity 范围内 |
| ledger / finite / temperature / safety | 通过 | 通过 | 不是数值崩溃导致 |

### 裁决

`n1_signal_passed=false`。失败的含义是：在冻结的 checkpoint、`c1=0.5` 和当前
independent additive 采样口径下，没有观察到 DEC-048 fused student 的正向
`mixture_ess_proxy` 信号。由于所有绝对健康门、有限值门和账本门均通过，不能把这次
失败解释为 CUDA、TorchForce、温度或力失稳。

同时，不能把它外推成“LocalResidualStudent 没有任何信号”：DEC-048 证明的是另一种
fused Hamiltonian 下的三次 paired-reseed exploratory proxy 改善；本次结果只否定当前
方案②的 N=1 screening signal。`mixture_ess_proxy` 本身也不是字面
`pymbar.compute_overlap()`，单次配对检查更不是独立重复 promotion 证据。

按 DEC-056 的固定顺序：

- 不运行方案②的 MTS/N>1 qualification；
- 不因结果重调 `c1`、切换 checkpoint 或重新挑 seed；
- EXP-013 的在线/MTS 路线不晋级，WP-5 不重新打开；
- EXP-014 native OpenMM compression 可以作为下一项独立 contingency，但不能把它写成
  EXP-013 已证明的延续，也不能使用此前 out-of-order、已标记 invalidated 的 EXP-014
  报告作为本次证据；
- 在启动 EXP-014 前，先保留本 DEC-059 作为 EXP-013 全顺序裁决，并重新为 EXP-014
  规定唯一有效输出和资格门。

### 综合结论

当前证据支持的最强、最窄结论是：训练得到的 LocalResidualStudent 在离线 held-out
gap-variance 与 DEC-048 fused N=1 exploratory ESS proxy 中表现出候选信号，但现有
real-time TorchForce 路线成本过高；方案③的 MTS residual split 未通过动力学主门，
方案① whole fused Group-1 的 qualification 未通过保守单种子相对门，方案② independent
additive student 又未通过 N=1 ESS signal。因而尚未证明存在可安全低频化、可净提高有效采样
效率的 production learned slow force；不得使用“找到了慢神经势”或“低频更新保持正确
采样”的表述。

## DEC-060：`EXP014_NATIVE_COMPRESSION_SCREEN_NOT_PASSED`，停止当前压缩 contingency

**日期：** 2026-08-09  
**状态：** `DECIDED / EXP014_SCREEN_NOT_PASSED / STOP`  
**对应实验：** EXP-014 native typed-pair radial compression offline screen  
**证据：** [`output/outer_lambda_exp014_native_compression_audit_after_exp013/EXP-014_native_compression_audit.json`](output/outer_lambda_exp014_native_compression_audit_after_exp013/EXP-014_native_compression_audit.json)，文件 SHA-256 `c19c8c3927cd1ecf0477657dc3028020cf7cf209722d64bccb569f6107f66198`

### 结果与裁决

EXP-014 使用冻结的 `3×500` 帧数据，leave-one-run-out 训练/验证，typed
ligand/environment pair + quintic-C2 cutoff + radial RBF，测试 `8/16/32` 个径向基。
这是离线压缩筛选，不运行 MD，也没有进行 OpenMM energy/force qualification。

三个基数均未满足预先冻结的共同门（所有三折 `R²≥0.90` 且 retained student
gap-variance improvement `≥0.80`）：

- `n_radial=8`：mean held-out `R²=-11.46`，mean retention `-24.35`；
- `n_radial=16`：mean held-out `R²=-2.80×10⁶`，mean retention `-6.18×10⁶`；
- `n_radial=32`：mean held-out `R²=-2.09×10¹²`，mean retention `-4.62×10¹²`。

因此登记 `EXP014_NATIVE_COMPRESSION_SCREEN_NOT_PASSED`，不进入 OpenMM force
qualification，不提升为 production route。该结论只关闭本次冻结的 typed-pair/radial
compression screen；不等于证明 LocalResidualStudent 的所有离线表示或其它新压缩形式
都没有信号。此前同目录中标记为 `INVALIDATED_OUT_OF_ORDER` 的报告不作为本裁决证据；
本报告是按 DEC-059 分支在独立新目录完成的有效 screen。

### 当前执行边界（2026-08-09）

EXP-013 的方案③、方案①、方案②均未晋级，EXP-014 native-compression offline screen 也未通过；因此当前在线/MTS/压缩分支最终停止。不得重调 `c1`、重选或重训 checkpoint、继续搜索 MTS 间隔，也不得直接重开 WP-5。既有报告和哈希证据保留为研究结论；任何新路线若未来需要讨论，必须另立范围与决策，不能作为本分支的自动后续。

## DEC-061：`ORB003_DEVICE_MISMATCH_NOT_ELIGIBLE`，ORB 保留为离线/教师路线

**日期：** 2026-08-09  
**状态：** `DECIDED / ORB003_CPU_DIAGNOSTIC_COMPLETED / ONLINE_NOT_ELIGIBLE / ORB004_005_STOPPED`  
**对应实验：** ORB-003 frozen `orb-v3-conservative-omol`, layer 2 cost probe  
**证据：** [`orb003_cost_probe_cpu_bridge_diag.json`](output/outer_lambda_orb/orb003_cost_probe_cpu_bridge_diag.json)，report_sha256 `0dc3a989dc1b07f75bb171bdbfa621d64eec618c5440adfba462c1b19e7acff5`

### 结果

ORB-003 只使用已冻结的 layer 2、父体系 `Q=0/M=1` contract、`knn_alchemi`、CPU float64 graph construction、float32 model batch、`compile=False`。真实 EXP-012 frame 的 L2 closure 为 `1538` nodes、`111622` edges，最大 outgoing neighbor `110`，无 120-cap 命中。分阶段 CPU diagnostic 为：

- official graph construction median `182.82 ms`；
- layer-2 prefix forward median `614.89 ms`；
- measurement-only scalar coordinate backward median `1611.05 ms`；
- TorchScript +真实 production `System` 的 CPU TorchForce group evaluation `2047.83 ms`（单次 diagnostic sample）。

临时 scalar 仅为成本探针，不是 ORB-004 readout。它与已核验 offline adapter 的 scalar absolute difference 为 `0`，TorchScript 可生成；这只证明调用链一致，不证明在线模型资格。

### 设备限制与裁决

本机 `torch.cuda.is_available()=false`、CUDA device count 为 `0`。真实 production `openmm.chk` 是 CUDA checkpoint，CPU 恢复明确失败：`Checkpoint was created with a different Platform: CUDA`。因此 TorchForce bridge 使用登记轨迹 frame 手动初始化，仅标为 `COMPLETED_DIAGNOSTIC_NOT_CHECKPOINT_MATCHED`；没有生成伪造的 CUDA matched-path 增量，也没有把 CPU 数字当作 CUDA 资格结果。现有 CUDA baseline 约 `1.396 ms/step`，本次没有同平台 ORB increment。

因此 ORB-003 状态为 `DEVICE_MISMATCH_NOT_ELIGIBLE`；在当前证据范围内登记 `OFFLINE/TEACHER_ONLY`，停止 ORB-004/005、在线 TorchForce、MTS 和 OpenMM production wiring。若未来获得同平台 CUDA，只能另开独立成本复核，模型、layer、contract 和本次统计结果不得事后重选。

## DEC-062：`ORB003_CUDA_MATCHED_COST_FAILED`，确认 ORB 为离线/教师路线

**日期：** 2026-08-10  
**状态：** `DECIDED / ORB003_CUDA_MATCHED_COMPLETED / COST_GATE_FAILED / ONLINE_NOT_ELIGIBLE / ORB004_005_STOPPED`  
**对应实验：** ORB-003 frozen `orb-v3-conservative-omol`, layer 2 CUDA matched-path cost probe  
**证据：** [`orb003_cost_probe_cuda_node.json`](output/outer_lambda_orb/orb003_cost_probe_cuda_node.json)，report_sha256 `10ac708502f5a3fdf160db7d1e8c55a9494052e3053989f5f5bfce7abea335be`

### 结果

本次使用真实 production CUDA checkpoint 恢复 OpenMM `System`，`platform_resolved=CUDA`，ORB compute device 为 `cuda:0`，没有 fallback frame。冻结 contract 仍为 `orb-v3-conservative-omol`、layer 2、父体系 `Q=0/M=1`、`compile=False`。L2 closure 为 `1569` nodes、`113804` edges，最大 outgoing neighbor `107`，无 120-cap 命中。

- official ORB graph construction median：`30.835 ms`；
- layer-2 prefix forward median：`36.640 ms`；
- measurement-only scalar coordinate backward median：`80.490 ms`；
- matched production baseline：`1.273 ms/step`；
- matched production + temporary scalar：`78.896 ms/step`；
- incremental delta：`77.622 ms/step`，冻结预算为 `0.1–0.2 ms/step`。

CUDA TorchForce bridge 状态为 `COMPLETED_CHECKPOINT_MATCHED`；显存样本为 `7042/8801/16303 MiB`。wrapper 与 offline adapter scalar absolute difference 为 `7.59e-7`，满足调用一致性检查。

### 裁决

这次复核关闭了此前的 `DEVICE_MISMATCH_NOT_ELIGIBLE` 限制，但没有改变路线结论：真实 matched CUDA 增量约为预算上限的 `388` 倍，故成本门明确失败。最终登记为 `OFFLINE_TEACHER_ONLY`；`ORB-004/005`、在线 TorchForce、MTS 和 OpenMM production wiring 均停止，不再重选模型、layer、checkpoint 或继续优化在线路径。
