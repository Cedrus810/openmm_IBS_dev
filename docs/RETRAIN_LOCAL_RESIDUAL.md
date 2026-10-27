# 给新配体重训 Local-Residual R1 权重

更新日期：2026-09-10（2026-10-17 加了下面这一节，**生产重训以它为准**；2026-10-27 补插件与 manifest 说明）

## 2026-10-17 起：从一次不开 outer 的 run 的 stage2 帧重训

仓库**不随包发任何冻结权重**（`resources/` 为空）。开 `--outer-lambda-local-residual-ibs`
就必须同时给 `--outer-lambda-resource-manifest`，权重只能按下面三步自己重训出来。

预平衡帧自动重训（下文 "P1 已落地" 那条）**已删除**：那批帧是 λ_coul=1、λ_vdw=1，目标态是
λ_coul=0，重要性权重塌到 1 帧（实测 5 个 run 的 Kish ESS 全是 1.0），拟出来的 B 是噪声。
现在开 outer 而 manifest 覆盖不到配体，`runabfe` 在入口就拒绝。流程：

```bash
# 1. 不开 outer 跑（需要 outer 的体系本来就是先跑出了问题的那次）。stage2 旁车默认带水
#    （ABFE_STAGE2_COORDS_WATER_NM=0.8；10-17 以前的 run 没存水，不能用）
# 2. 从一个或多个这样的 run（同一配体，可混 rep、可混不同盒子）重训
python tools/refit_outer_residual_from_stage2.py --runs <run1> [<run2> ...] --output <dir> [--stride N]
# 3. 带冻结权重重跑
python runabfe.py ... --outer-lambda-local-residual-ibs --outer-lambda-resource-manifest <dir>/manifest.json
```

中心仍只用配体原子（离线比较过加 Y176 / 口袋残基当中心，在不带水的特征上没有额外收益）。
容量取 配体原子数 × 80（每原子邻居上限），配体最多 102 个原子（插件候选表 8192）。
`<dir>/stage2_refit_report.json` 里有每个 run 每个窗口目标态的 Kish ESS、留出 loss 比、tanh 改形比例。

只有复合物腿用 outer，溶剂腿不开（2026-10-19 定）。

**插件**：默认加载 `plugins/LocalManyBodyResidual/build/`，2026-10-27 起指向
`build_20261018/`（按当前源码编的）。`.so` 里不含 device code，CUDA kernel 由 OpenMM
运行时编译，不绑显卡架构；绑定的是 `environment.yml` 钉的 `openmm=8.5.2`。
要用别的编译目录：`export ABFE_LMBR_PLUGIN_DIR=<和 platforms/ 同级的编译目录>`（源码 sha 门照样过）。

**manifest 的可选 `deployment` 节**（2026-10-21）：`scale`（B 的缩放 c）、
`rereference_window_coefficients`（窗口内 A_k → A_k − ⟨A_k⟩）、
`window_basis_offsets_kj_mol`（按窗口 B → B − m_w，键 = 窗口首末 λ_vdW）。
`tools/refit_outer_residual_from_stage2.py` **不生成这一节**（不写 ⟹ 与 10-21 之前行为相同）；
要用需在生成的 `manifest.json` 里手动添加，读取与校验见
`local_residual/openmm_plugin.py` 的 `_parse_deployment`。

以下是 09-10 的旧说明，手工 A/B 链仍然可用。

`outer_lambda_local_residual_ibs` 用的那个残差模型（R1）**绑配体，不绑体系**。

| 换什么 | 要不要重训 |
|---|---|
| 换蛋白 / 换膜 / 换水盒 / 复合物腿↔溶剂腿 | **不用**，同一份权重照用 |
| 换配体（原子序数序列或内部键图不同） | **必须**，manifest 会 fail-closed 挡住 |
| 环境里出现 `type_vocabulary` 外的元素（默认 H/C/N/O/Na/S/Cl） | **必须**，且要先扩词表 |

判据是 `local_residual/openmm_plugin.ligand_chemical_identity()` 算出来的指纹
（配体局部原子序数序列 + 内部键图的 canonical-JSON SHA-256），跟蛋白无关。

模型本体很小：4–5 Å 壳层、quintic C2 包络、5 Å 外严格为零，每个配体原子类型一个
`Linear(1,16)→SiLU→Linear(16,16)→SiLU→Linear(16,1)`，Atenolol 那份一共 3048 个
float64（24 KB）。

> **⚠️ 2026-09-10 更正：这条链上没有 MACE。** 本文最初把 EXP-010 的 MACE 离线教师
> 标注写成了第 ① 步，是错的（由 abfe-ibs-4e 的离线核查指出）。出厂那份 R1 是
> `direct_gap` 变体，训练信号**纯 MM**：`local_residual/loss.py` 的
> `bidirectional_gap_variance_loss` 只吃 `adjacent_gap_reduced`（来自 MM ledger），
> `local_residual/softlift_dataset.py:467-471` 只读 ledger 的
> `adjacent_gap_reduced` / `log_importance_unnormalized` + 对应轨迹，
> `abfe_scripts/run_exp019_softlift_d0.py` 的参数里没有任何 teacher/model 入口。
> EXP-010 那套 `exp010-*` 教师子命令是**另一条谱系**，不在 R1 重训链上。

> 2026-09-12 曾有过「开 outer 时在预平衡之后自动闭式重训」的路径，**2026-10-17 已删除**（预平衡帧与目标态不重叠，权重塌到 1 帧）。现行流程见本文第一节；做 A/B 时两臂必须共用同一份冻结权重。

## 链条

```
① 数据集      abfe_scripts/run_exp019_softlift_d0.py            → softlift_dataset_v1.npz
              输入：生产每窗口的 MM ledger + 对应轨迹（无 MACE、无 GPU 标注）
② 训练        abfe_scripts/train_exp019_softlift_loro.py --rung R1 → *.pt
③ 导出        abfe_scripts/export_exp025_g1_reference_payload.py  → payload.json + weights.bin
④ 部署 manifest abfe_scripts/write_local_residual_resource_manifest.py → manifest.json
```

代码位置：训练本体在 `local_residual/`（`softlift*`、`student`、`loss`、
`environment`、`mace_graph`、`atom_mapping`、`geometry`）。**生产入口不会 import 到它们**——`local_residual/__init__.py`
故意保持空，`import runabfe` 只会拉进 `openmm_plugin` 和 `em_no_residual`
（`tests/test_import_time_side_effects.py` 守着这条）。

## ①② 数据集与训练

```bash
python abfe_scripts/run_exp019_softlift_d0.py       # → dataset/softlift_dataset_v1.npz
python abfe_scripts/train_exp019_softlift_loro.py \
    --dataset <上面那份 npz> --rung R1 --seeds 0 1 2 \
    --output-root output/<新体系>_softlift
```

产出 `r1_density/r1__direct_gap__fold_test_run{N}__seed{S}.pt`。LORO：按 run 分
折，每折留一条 run 做验证。

## ③ 导出成 PyTorch-free payload

不传参数 = EXP-020 Atenolol 那条冻结路径（产物逐字节不变）。新配体传自己的路径，
没有事先约定 sha 的输入传 `none` 表示"只记录不比对"：

```bash
python abfe_scripts/export_exp025_g1_reference_payload.py \
    --checkpoint <你的 .pt>            --checkpoint-sha256 none \
    --dataset    <你的 .npz>           --dataset-sha256 none \
    --ligand-indices <ligand_indices.json> --ligand-indices-sha256 none \
    --topology   <topology.cif>        --topology-sha256 none \
    --trajectory <任一同拓扑轨迹>       --trajectory-sha256 none \
    --expected-active-edges none --expected-atom-count none \
    --output-dir <输出目录>
```

它自带一道 round-trip 门：完全只用导出的 JSON+二进制重建第二个模型，在那一帧上
必须与 checkpoint 加载出来的模型**逐比特相同**，否则 fail-closed。

## ④ 生成部署 manifest

```bash
python abfe_scripts/write_local_residual_resource_manifest.py \
    --payload  <输出目录>/r1_model_payload_v1.json \
    --weights  <输出目录>/r1_model_weights_f64.bin \
    --topology output/topology.cif \
    --ligand-indices output/ligand_indices.json \
    --system   output/system_native.xml \
    --ligand-name <配体名> \
    --output   resources/outer_lambda_local_residual/manifest.json
```

`--system` 建议给：mmCIF 对非标准残基不保留键，缺 System 时内部键图只能从拓扑图
推，可能少键。写完会立刻用 loader 自己的 `_load_resource_manifest()` 回读校验，
不通过就删掉半成品。

payload/weights/manifest 三个文件必须在同一个目录（manifest 里是相对路径）。

## 验收

```bash
python plugins/LocalManyBodyResidual/cuda_smoke_test.py   # 需要 GPU
python -m pytest tests/test_outer_lambda_local_residual_runtime.py -q
```

然后才能开 `--outer-lambda-local-residual-ibs`（默认 false）。

## 两处会绊人的地方

- **插件源码 sha 是一道硬门**。`resources/.../manifest.json` 的
  `plugin.source_sha256` 必须等于 `local_residual/openmm_plugin.py` 里的
  `KNOWN_PLUGIN_SOURCE_SHA256`，而后者是
  `plugins/LocalManyBodyResidual/platforms/cuda/src/CudaLocalManyBodyResidualKernels.cpp`
  的 sha。改过那个 .cpp（**哪怕只加注释**）就要两处一起换。生成器已经自动带上
  当前值。
- **EXP-020 那份 canonical fixture 的轨迹已经没了**（`hard_window0_run1/
  scratch_sample/` 是 scratch，早被清）。所以③那道 round-trip 门在复现 Atenolol
  时要换一条同拓扑轨迹并传 `--expected-active-edges none`——权重本身与帧无关，
  产出的 `r1_model_weights_f64.bin` 仍然逐字节等于出厂那份
  （`c4492f9d…`，2026-09-10 实测）。
