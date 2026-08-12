# PLAN-EXP-019：baseline reproducibility / uncertainty calibration

状态：`SEALED / ZERO_COST_ATTRIBUTION / BASELINE_MD_NOT_STARTED`  
日期：2026-08-11

## 1. 科学问题与边界

EXP-019 的目标是先判断现有 baseline 的重复间波动来自哪些已记录的 state、energy/base/bias、occupancy 或构象代理区间，再用真正独立的 complex/solvent repeats 校准 ΔG 不确定度。EXP-018 保持封存为 `INCONCLUSIVE`；本实验不重新定义 EXP-018 verdict，不追加 EXP-018 seed，不回填修改 EXP-017 formal error。

本实验不进入 DEXP、lambda insertion、neural、analytic-q 或其他新势能比较。若 baseline 的 repeat variance 仍明显大于 TMBAR covariance uncertainty，路线先停在独立采样/误差模型层面。

## 2. 第一阶段：EXP-018 现有数据的零成本 attribution

输入仅为已完成的 `output/outer_lambda_exp018_stationarity_confirmation_v2/` 三条 ledger。分析器在读取 `ledger_arrays.npz` 数值前核验自身 hash、EXP-018 final registry、sampling aggregate、sample-report 和 ledger hash。

冻结的分析契约：

- 必须使用 seeds `20260811`、`20260812`、`20260813` 的全部三条 repeat；任一缺失即 fail-closed。
- 使用全部 500 帧；不做 post-hoc frame/block/state 筛选，不拼接三条轨迹。
- 逐 repeat、逐四态记录 `target_interaction`、`target_reduced_potential`、softcore、LRC、importance log-weight、归一化 weight ESS 和 top-1% weight mass。top-1% 只作集中度诊断，永不用于剔除帧或重算 EXP-018。
- 将 `base_energy`、IBS bias、WCA bias、sampling bias、total context 和直接 endpoint gap 的 repeat-mean variance 并列报告；这些是 ledger component diagnostics，不宣称为 ΔG 的因果分解。
- 每条 repeat 固定切成 10 个连续 block（当前 500 帧即每块 50 帧），报告每块的 energy/base/bias、endpoint-gap 和 state-weighted interaction；以预先固定的 `max absolute block deviation share >= 0.5` 标记“dominant block candidate”，但不删除该 block。
- 对 state-level weighted interaction 的跨-repeat variance 报 descriptive share，并明确这不是物理 state 或构象机制的因果证明。
- 报告 ledger closure、finite-value、frame-order、state 数和 input hash；任何失败都不做部分选择。

交付物：独立 attribution JSON、Markdown summary 和各自 self-hash，均引用 EXP-018 sampling aggregate SHA。报告必须明确 `exp018_verdict_modified=false`、`production_data_mutated=false`、`posthoc_frame_filtering=false`。

## 3. 第二阶段：真正独立的 baseline repeats

### 3.1 冻结 baseline

complex 与 solvent 两条腿均冻结当前实际 artifact，不手抄旧 λ 或静默替换路径：

| 维度 | 冻结规则 |
|---|---|
| method | `mode=ibs` |
| decoupling | `dual_lambda` |
| base potential | `softcore` |
| DEXP | `null` |
| path | `path_protocol_version=21`，实际 complex/solvent schedule 以封存 hash 为准 |
| IBS/TMBAR | 当前 production ledger 语义与 IBS bias protocol `29` |
| temperature | `300 K` |
| online additions | neural、student、analytic-q、lambda insertion 全部关闭 |
| legs | complex 与 solvent 均必须完成各自完整 charging/vanishing cycle |

当前 anchors：complex 使用 `system_native.xml`、`topology.cif`、`box_vectors.npy`、complex 两个 dual preopt/stage artifacts；solvent 使用 `system_solvent.xml`、`topology_solvent.cif`、`box_vectors_solvent.npy`、solvent 两个 dual preopt/stage artifacts。两条腿所有 window manifest、checkpoint、IBS state、endpoint ledger 和 cycle artifact 在 MD 授权前必须逐项生成并冻结 hash manifest。

### 3.2 独立性与重复数

- complex、solvent 各至少 3 个独立 repeat；论文目标为 5 个。
- 每个 repeat 独立平衡、独立初态、独立 integrator/random seed；不得从同一 production checkpoint 分叉。
- 不允许复用其他 repeat 的坐标、速度、trajectory、checkpoint、scratch 或 ledger；重复间只能共享上述只读 baseline definition。
- MD launcher 必须拒绝非空输出目录、共享 checkpoint、输入 hash 漂移、state/lambda 顺序漂移和未经冻结的 seed。
- 本计划和 attribution 分析完成前不授权 complex/solvent MD；本次立项不启动 GPU 采样。

### 3.3 每条 repeat 的交付物

每条 complex/solvent repeat 必须独立记录完整 ledger、trajectory、checkpoint、sample report、endpoint diagnostics、cycle diagnostics、环境与代码 hash。必须计算：

- `Delta_G_complex`、`Delta_G_solvent`、`Delta_G_bind = Delta_G_complex - Delta_G_solvent`；
- 每条腿及 binding 的 TMBAR covariance uncertainty；
- full/first-half/second-half ΔG、drift 与固定的 drift diagnostic；
- repeat-to-repeat SD、SEM、variance ratio，以及 repeat-aware/hierarchical uncertainty candidate；
- endpoint closure、ledger component closure、cycle closure、state coverage、finite-value 和完整性 gate。

三条或五条 repeat 不得简单拼成一个长轨迹作为 primary estimate；每个 repeat 必须保持独立身份。

## 4. 预注册成功门

成功门必须同时满足：

1. complex 与 solvent 所有 repeat 的完整 ledger、endpoint、cycle 和 provenance gate 通过；
2. 两条腿均无预注册 split-half 定向漂移；repeat 间 full ΔG 不显示同方向系统漂移；
3. 两条腿的 repeat variance 与 formal uncertainty 一致，或被预注册的 repeat-aware/hierarchical uncertainty 正式覆盖；不得只报告单条 TMBAR sigma；
4. complex、solvent 与 derived binding 的方向和 uncertainty 结论一致，cycle closure 通过；
5. 任一腿 repeat variance 仍显著大于 TMBAR uncertainty，或任一 integrity gate 失败：停止进入新势能/DEXP/neural 比较，先增加独立采样或另立误差模型实验。

EXP-019 的成功只表示 baseline 的 ΔG 与 uncertainty 达到预注册可信度；不自动授权任何创新路线。

## 5. 终局约束

EXP-018 final registry、sampling aggregate、三条 trajectory、checkpoint、ledger 和分析文件均为只读证据。EXP-019 attribution 不改变其 `INCONCLUSIVE` 结论。EXP-017 formal error 保持不变；任何正式 uncertainty-model 修订必须基于新的、独立预注册实验并另行登记。
