# 给新配体重训 Local-Residual R1 权重

更新日期：2026-09-10

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

> **P1 已落地（2026-09-12）：** 这条四步链的 ①② 现在有了自动替代 —— 开
> `--outer-lambda-local-residual-ibs` 且配体不在冻结 manifest 覆盖范围内时，主线会在
> 基线预平衡之后自己做一次闭式重训（帧源 `pre_equilibration.dcd`），③④ 不变。
> 见 [EXP-033_P1_LANDED_2026-09-12.md](EXP-033_P1_LANDED_2026-09-12.md)。
> **做 A/B 仍然走本文这条手工链**（两臂必须共用同一份冻结权重）。
> 原始方案登记在
> [EXP-033_LOCAL_RESIDUAL_REFIT_2026-09-10.md](EXP-033_LOCAL_RESIDUAL_REFIT_2026-09-10.md)。

## 链条

```
① 数据集      abfe_scripts/run_exp019_softlift_d0.py            → softlift_dataset_v1.npz
              输入：生产每窗口的 MM ledger + 对应轨迹（无 MACE、无 GPU 标注）
② 训练        abfe_scripts/train_exp019_softlift_loro.py --rung R1 → *.pt
③ 导出        abfe_scripts/export_exp025_g1_reference_payload.py  → payload.json + weights.bin
④ 部署 manifest abfe_scripts/write_local_residual_resource_manifest.py → manifest.json
```

代码位置：训练本体在 `local_residual/`（`softlift*`、`student`、`loss`、
`environment`、`mace_graph`、`atom_mapping`、`geometry`），2026-09-10 从
Atenolol-rank11 搬入。**生产入口不会 import 到它们**——`local_residual/__init__.py`
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
