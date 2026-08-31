# 输出、结果解读与续跑

> 本文说明输出目录、Boresch 来源、结果口径、缓存/续跑以及 GPU 并行。它描述的是
> 工作流行为，不代表当前某一轮结果已经通过科学验证。当前结论请查阅
> [状态与验证文档](HISTORY_LOG.md)。

[返回项目首页](../README.md) · [文档导航](README.md)

## Boresch 来源

`--boresch-source` 可选值：

- `simple`：纯几何涨落估算，不依赖 ML。当前推荐和当前结果使用这一项。
- `fluctuation`：与 `simple` 类似的几何涨落路线。
- `traditional`：读取外部 Boresch 锚点文件，需要 `--boresch-anchors`。
- `orb_ml`：读取 ORB/ML 预测文件，需要 `--boresch-orb`。
- `orb_simple`：使用 ORB/MACE 口袋力投影单候选估算，需要 ML 依赖和模型许可。
- `auto`：使用 ORB/MACE 多候选枚举估算，需要 ML 依赖和模型许可。

内部估算路线会在 `output/boresch_*.json` 中保存参数和诊断。当前结果中的 Boresch 诊断要重点看：

- `diagnostics.boresch.analytical_release_reliable`
- `boresch_correction_diagnostics.diagnostics.boresch_harmonicity_check`
- `force_constant_clipped`
- `diagnostics.warnings`

## 输出结构

常见输出：

```text
output/run_provenance.json              配置、命令行、hash、软件版本
output/final_binding_results.json       最终 Delta G_bind 汇总
output/final_results.json               复合物腿结果
output/solvent_leg/final_results.json   溶剂腿结果
output/pipeline.log                     复合物腿日志
output/solvent_leg/pipeline.log         溶剂腿日志
output/system_native.xml                复合物 OpenMM System 缓存
output/system_solvent.xml               溶剂腿 OpenMM System 缓存
output/topology.cif                     复合物拓扑缓存
output/topology_solvent.cif             溶剂腿拓扑缓存
output/boresch_simple.json              当前 Boresch 参数缓存
output/checkpoints/                     复合物腿 checkpoint 与阶段状态
output/solvent_leg/checkpoints/         溶剂腿 checkpoint 与阶段状态
output/decharging/                      decharging 阶段轨迹/能量
output/vanishing/                       vanishing 阶段窗口能量
output/vanishing/production_fixed_h_overlap.json  production ESS 修复使用的逐边 fixed-H 诊断
output/vanishing/sampling_repair_decisions.json   偏置重校准/延长采样等自动修复决策
output/checkpoints/probes/              可续采的 fixed-H path/bias-calibration 轨迹库
```

窗口级文件通常包括：

```text
dual_window_*_energies.npy
dual_window_*_bias.npy
dual_window_*_base.npy
dual_window_*_convergence.json
decharging_pme_u_kn.npy
decharging_pme_u_kn.meta.json
```

这些文件是 `--analyze-only` 和 `refine-lambda-path` 的核心输入，清理前请确认不再需要后处理。

## 结果解读

最终结合自由能按以下形式汇总：

```text
Delta G_bind = Delta G_solvent - Delta G_complex + Delta G_APBS
```

其中：

- `Delta G_complex`/`Delta G_solvent` 都定义为各自腿 λ:1→0（coupled→decoupled）的去耦自由能，即数值越大代表越难去耦、相互作用越强。
- `Delta G_complex` 已包含复合物腿 Boresch 解析释放修正。
- 溶剂腿不使用 Boresch，因此 `boresch_correction_kJ_mol = 0`。
- 对真实结合的配体，`Delta G_complex > Delta G_solvent`（口袋里去耦更难），因此 `Delta G_bind` 应为负值（有利结合）。
- `Delta G_APBS` 默认为 0；只有显式传入 `--apbs-correction-kj-mol` 才会应用。
- 默认 ACE/`dual_lambda` 的 vanishing 腿会自动加入 switching-aware、softcore-aware 的解析 LJ 长程尾项（同时包含 `r^-6` 与 `r^-12`）；传统 Beutler REMD 在固定盒 NVT 轨迹上离线加入同一修正。传统路径若检测到明显 NPT 体积波动会硬停止，因为事后追加 `1/V` 不能修复未按该哈密尔顿量采样的体积分布。

判断一次结果是否能进入下一步讨论时，至少检查：

- `output/final_binding_results.json` 是否存在且 timestamp 符合本次运行。
- `provenance.hashes.code_sha256` 是否符合你要归档的代码版本。
- `lj_long_range_dispersion_correction.status` 是否为 `implemented_analytic_mean_field`，并核对相应 LRC 协议版本/指纹；旧的 `not_implemented` 输出不得与新协议结果混用。
- `stage_diagnostics.stage2.min_overlap_proxy` 是否过低。
- `stage_diagnostics.*.uncertainty_note` 是否仍提示缺少完整协方差/自相关修正。
- Boresch harmonicity check 是否 `ok = true` 且 `harmonic_assumption_ok = true`。
- `force_constant_clipped` 是否有大量 `true`。
- 是否做过独立重复运行；当前结果记录为 `independent_repeats.performed = false`。

## 缓存与续跑

`--resume` 会尝试复用：

- `system_native.xml` / `system_solvent.xml`
- `ligand_indices*.json`
- `topology*.cif`
- `pre_equilibration.dcd`
- `checkpoints/pre_equil.chk`
- `boresch_*.json`
- `preopt_dual_*.json`
- 阶段采样结果和窗口能量文件

以下情况建议使用新 `--output` 目录或 `--reset`：

- 改了 `gro/top/ligand`。
- 改了配体参数或 `gmx_path`。
- 改了 `decoupling`、`potential` 或 DEXP 参数。
- 改了 lambda 状态数、窗口密度或采样预算。
- 改了 Boresch 来源、锚点或候选选择。
- 想把旧 thermodynamic_cycle/provenance 文本完全刷新。

## 并行与 GPU

`--parallel-stages` 会尝试并行运行 decharging 和 vanishing。CUDA 下如果两阶段共用同一 GPU，代码可能回退为串行以避免上下文冲突。

Linux shell 示例：

```bash
IBS_STAGE1_CUDA_DEVICE=0 IBS_STAGE2_CUDA_DEVICE=1 \
python runabfe.py --config abfe_config.json --ligand MOL --resume --parallel-stages
```

Windows PowerShell 示例：

```powershell
$env:IBS_STAGE1_CUDA_DEVICE = "0"
$env:IBS_STAGE2_CUDA_DEVICE = "1"
python runabfe.py --config abfe_config.json --ligand MOL --resume --parallel-stages
```
