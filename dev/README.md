# Atenolol-rank11 ABFE/IBS workflow

[中文](README_cn.md) | [English](README_en.md)

本仓库是一个面向 Atenolol-rank11 体系的 ABFE（Absolute Binding Free Energy）计算工作目录。代码以 OpenMM 为核心，从 GROMACS `gro/top` 输入构建 OpenMM 原生缓存，运行复合物腿和溶剂腿，并用 IBS/MBAR/TMBAR 风格的窗口采样结果汇总结合自由能。

当前推荐路线是：

```text
mode = ibs
decoupling = dual_lambda
potential = softcore
boresch_source = simple
```

`traditional` REMD、DEXP、ORB/MACE Boresch 估算和 APBS 外部修正仍保留在代码中，但不是本目录当前结果的主路线。

## 当前状态

⚠️ **`output/final_binding_results.json` 里现有的数值是用旧的 `Delta G_bind = Delta
G_complex - Delta G_solvent` 符号约定算出来的，与当前代码（`runabfe.py` 里
`delta_g_bind_uncorrected = dg_solvent - dg_complex`）不一致，需要重新跑一遍才能
得到跟当前公式匹配的号（详见下方“结果解读”和 `docs/status/AUDIT_STATUS.md`）。** 下面这组
数字按当前公式重新算过（ΔG_complex/ΔG_solvent 两个原始值不变，只是改了合并符号），
仅供参考，不代表已经重新采样：

```text
Delta G_complex = 192.8876 kJ/mol
Delta G_solvent = 152.0514 kJ/mol
Boresch correction = -36.5108 kJ/mol
APBS correction = 0.0000 kJ/mol
Delta G_bind = Delta G_solvent - Delta G_complex = -40.8362 kJ/mol = -9.7601 kcal/mol
reported error = 1.3178 kJ/mol
```

（负值表示有利结合：口袋里去耦花的自由能 ΔG_complex 比溶液里去耦花的 ΔG_solvent 更
大，说明配体在口袋里的相互作用更强，这正是它愿意结合的原因；磁盘上残留的正号结果
物理上是反的，不要拿它做任何结论。）

这不是一个可以直接当作最终发表数值的“全修正闭环”结果。当前最重要的物理边界是：

- 默认 ACE/`dual_lambda` 的 VDW/vanishing 腿已启用解析 LJ 长程尾项修正（`traditional_lj_lrc_protocol_version=2`）：对每个 λ_vdw 数值积分 switching-aware、softcore-aware 的真实径向尾项，同时包含吸引 `r^-6` 与排斥 `r^-12`；不启用会把组合表达式中的 Coulomb 尾项一并错误积分、并可能令 CUDA 崩溃的 OpenMM `CustomNonbondedForce` 内建 LRC。`single_lambda`/REMD 的 Beutler 路径使用同一公式作离线修正；协议版本低于 2 的旧输出不得与当前结果混用。
- APBS 修正只作为最终外部项 `Delta G_APBS` 加到 `Delta G_bind`（当前 `apbs_correction_kJ_mol = 0.0`，未启用），不能替代 LJ tail correction。
- 旧输出中的 `thermodynamic_cycle.md` 和 provenance 里仍可能包含历史 PME self-correction 描述；`output/final_binding_results.json` 里 `provenance.thermodynamic_cycle` 目前就是这种未刷新的旧文本快照。请以 `docs/status/AUDIT_STATUS.md` 和当前 diagnostics 为准。当前结论是：手动 `+C*lambda^2` PME 自能“修正”已撤销，不应作为生产修正项使用。
- 当前 Boresch 谐振性校验通过（`diagnostics.boresch.boresch_harmonicity_check.harmonic_assumption_ok = true`），但 6 个力常数里有 3 个（`kr`、`kthetaA`、`kphiA`）被裁剪到保守范围（`force_constant_clipped`），需要在结果解释中保留。
- Stage 2 采用 `Local-TMBAR-Stitched`，误差已传播窗口 offset 方差（复合物腿 `offset_error_contribution ≈ 0.52 kJ/mol`，溶剂腿 `≈ 0.82 kJ/mol`），但尚未包含完整全局 MBAR 协方差、自相关时间和有效样本数修正。当前实现会把低 overlap/ESS 精确定位到失败窗口、相邻状态和 λ；先只续采失败窗口，仍不足时再用已有 λ 状态建立独立、不可变的重叠 rescue ensembles，原 ensemble 与生产 `f_k` 均不原地修改。
- 尚未做独立重复运行：`diagnostics.independent_repeats.performed = false`。

更详细的方法学缺陷、工程审计遗留项和修复状态见 `docs/status/AUDIT_STATUS.md`；文档导航见 `docs/README.md`。

截至 2026-07-22 的实现快照：默认生产主链使用
`IBS_BIAS_PROTOCOL_VERSION=27`（兼容读取 v27/v28 缓存）、`THERMODYNAMIC_PATH_PROTOCOL_VERSION=20`、
`TRADITIONAL_LJ_LRC_PROTOCOL_VERSION=2` 和 `WCA_ACCOUNTING_VERSION=2`。v12 已加入
按状态共享、带协议指纹和 OpenMM checkpoint 的 fixed-H 探针轨迹库。当前 IBS 协议已
修正 pilot-TI/TMBAR 的 `f_k` 符号；预热完成一次完整的固定权重验证后即冻结候选 `f_k`，
验证残差作为效率诊断，不再误当作生产准入的无限重学习条件。生产从独立的第 0 步开始，
不继承预热/验证帧，且 `f_k` 受运行时只读锁保护，生产阶段不得更新。Stage 2 质量门失败时，
默认按 250k→500k→1M 的累计预算只补失败窗口；若仍不足，则创建独立 rescue ensemble。
完整协议、边界与待验证项见 `docs/status/IBS_PRODUCTION_PROTOCOL_2026-07-22.md`。
当前没有剩余的、已确认会阻断默认生产主链的 P0/P1；未修代码行动以 `docs/TODO.md`
为准，代码已修但尚待真实 GPU/完整依赖验证的项目以 `docs/status/VALIDATION_MATRIX.md` 为准。

## 主要文件

```text
runabfe.py              命令行入口；配置合并、缓存构建、两条腿调度、最终汇总
abfe_pipeline.py        单腿 ABFE 流程；预平衡、预优化、采样、单腿结果写出
abfe_preoptimizer.py    lambda 路径预优化；dual-lambda 与 2D 路径辅助逻辑
ibs_engine.py           IBS 采样、窗口能量、MBAR/TMBAR 后处理和 checkpoint
abfe_core.py            softcore/DEXP 势、Boresch 限制、解析修正和工具函数
apbs_correction.py      可选 APBS 外部修正辅助脚本；不处理 LJ tail correction
abfe_config.json        当前目录示例/兼容配置
environment.yml         环境示例
docs/README.md          文档导航和维护规则
docs/TODO.md            唯一的当前行动清单
docs/status/            审计历史、结案状态和运行验证矩阵
docs/handoffs/          专题排障/实验交接快照
todo2.txt               指向 docs/TODO.md 的兼容入口
```

本目录还包含 Atenolol-rank11 输入与中间文件，例如 `solv_ions.gro`、`topol.top`、`Atenolol-rank1.*`、`output/` 等。

## 依赖

建议使用 conda/mamba 环境。核心依赖包括：

- Python 3.10+
- OpenMM
- NumPy
- SciPy
- MDTraj
- PyMBAR

可选依赖：

- CUDA 或 OpenCL，用于 GPU 运行。
- GROMACS force-field include 目录，用于首次从 `.top` 构建系统。
- OpenMM-ML、torch、MACE/ORB 相关依赖，仅在使用 `--boresch-source auto`、`orb_simple`、`orb_ml` 或相关 ML 功能时需要。

当前 `output/run_provenance.json` 记录的运行环境为 Python 3.12.13、OpenMM 8.5.1、NumPy 2.4.3、PyMBAR 4.0.3、MDTraj 1.10.3。

## 输入要求

首次构建缓存通常需要：

- `--gro`：GROMACS 坐标文件。当前配置使用 `solv_ions.gro`。
- `--top`：GROMACS 拓扑文件。当前配置使用 `topol.top`。
- `--ligand`：配体残基名。当前配置使用 `MOL`。
- `--gmx-path`：GROMACS force-field include 目录。也可通过 `GMXDATA` 或本机 `gmx` 自动探测。
- `--ligand-xml`：可选；构建溶剂腿时可显式指定配体 XML/FFXML。若未提供，代码会尝试从 GROMACS 拓扑抽取并生成 `output/ligand_only.xml`。

命令行参数优先级高于配置文件。

## 快速运行

使用当前配置续跑已有计算：

```bash
python runabfe.py --config abfe_config.json --ligand MOL --resume
```

首次运行或重新生成缓存：

```bash
python runabfe.py \
  --config abfe_config.json \
  --gro solv_ions.gro \
  --top topol.top \
  --ligand MOL \
  --gmx-path /path/to/gromacs/share/gromacs/top \
  --output ./output \
  --boresch \
  --boresch-source simple
```

忽略缓存重新开始：

```bash
python runabfe.py --config abfe_config.json --ligand MOL --reset
```

只对已有窗口能量和 checkpoint 做后处理：

```bash
python runabfe.py --config abfe_config.json --ligand MOL --analyze-only
```

CPU 小预算 smoke test 示例：

```bash
python runabfe.py \
  --config abfe_config.json \
  --platform CPU \
  --n-steps-per-window 1000 \
  --n-states-per-stage 4 \
  --output ./smoke_output \
  --reset
```

## 配置示例

当前 `abfe_config.json` 是兼容 JSON 配置。关键字段如下：

```json
{
  "preset": "production",
  "platform": "CUDA",
  "output": "./output",
  "temperature": 300.0,
  "solvent_ionic_strength_molar": 0.15,

  "gro": "solv_ions.gro",
  "top": "topol.top",
  "ligand": "MOL",
  "gmx_path": "/path/to/gromacs/share/gromacs/top",

  "decoupling": "dual_lambda",
  "potential": "softcore",

  "n_steps_per_window": 250000,
  "steps_per_update": 500,
  "stage1_n_states": 12,
  "stage2_n_states": 17,

  "boresch": true,
  "boresch_source": "simple",
  "boresch_batch": 0,
  "boresch_select": 1,

  "skip_rebalance": false,
  "rebalance_steps": 50000,

  "enable_early_stop": false,
  "enable_gradual_warmup": true,
  "warmup_steps": 500000,
  "min_bias_updates": 12,
  "max_bias_updates": 50,
  "required_consecutive_bias_updates": 3,
  "max_bias_warmup_steps": 500000,

  "pilot_finite_difference_delta": 0.01,

  "enable_lambda_refine": false,

  "resume": false,
  "reset": false
}
```

注意：仓库里的 `gmx_path` 是机器相关路径，换机器运行前必须检查。当前配置刻意保留 `enable_lambda_refine=false`，以免覆盖已有的 Stage 2 尾部局部修复；不要在不了解 `abfe_config.json` 中 `_comment_lambda_refine` 背景的情况下直接打开。

## 命令入口

主命令：

```bash
python runabfe.py [options]
```

重要参数：

- `--mode {ibs,traditional}`：采样引擎，默认 `ibs`。
- `--decoupling {dual_lambda,single_lambda,2d_diagonal,2d_geodesic}`：解耦路径，默认 `dual_lambda`。
- `--potential {softcore,dexp}`：势模型，默认 `softcore`。
- `--decharge-method {pme,shadow_ibs}`：仅影响 `--decoupling dual_lambda` 的 Stage 1 (去电荷)，默认 `pme`（原有行为不变）。`shadow_ibs` 是实验性的 Shadow-Coulomb IBS 路径，尚未经独立物理验证，只支持电中性配体，且暂不支持 `--parallel-stages`；生产结果请保留默认 `pme`。
- `--preset {test,production,high_accuracy}`：采样预设。
- `--stage1-n-states`：decharging 阶段 lambda 状态数，优先级高于 `--n-states-per-stage`。
- `--stage2-n-states`：vanishing 阶段 lambda 状态数，优先级高于 `--n-states-per-stage`。
- `--resume`：复用缓存、checkpoint、窗口能量和预优化路径。
- `--reset`：忽略缓存重新开始。
- `--parallel-stages`：尝试并行执行 decharging 与 vanishing 阶段。
- `--n-workers`：离线能量重算/后处理 worker 数。
- `--apbs-correction-kj-mol`：把外部 APBS 修正作为最终项加到 `Delta G_bind`。
- `--apbs-correction-note`：记录 APBS 修正来源。

内置子命令：

```bash
python runabfe.py self-test
```

运行轻量测试和物理约定检查。注意当前 self-test 里仍有历史 PME self-correction sign 检查；若其与最新物理结论冲突，应优先修代码/测试，而不是把它当作生产修正依据。

```bash
python runabfe.py prepare \
  --gro solv_ions.gro \
  --top topol.top \
  --ligand MOL \
  --gmx-path /path/to/gromacs/share/gromacs/top \
  --output-dir ./prep_output \
  --save-boresch boresch.json
```

生成预处理文件，例如 Boresch 参数或 DEXP 参数。

```bash
python runabfe.py refine-lambda-path \
  --stage-dir output/vanishing \
  --preopt-file output/checkpoints/preopt_dual_vanishing.json \
  --stage-type vdw \
  --max-window-span-kj 35.0 \
  --overlap 2
```

根据已有窗口能量重新分布 lambda 状态和窗口边界。它会覆盖写回 `--preopt-file`，使用前建议保留备份。

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

## 常见问题

### `ModuleNotFoundError: No module named 'openmm'`

当前 Python 环境没有 OpenMM，或没有激活正确环境：

```bash
python -c "import openmm; print(openmm.__version__)"
```

### GROMACS include 文件找不到

检查 `--gmx-path` 是否指向包含 `.ff` 文件夹的目录，例如：

```text
/path/to/gromacs/share/gromacs/top
```

也可设置 `GMXDATA`，让代码自动尝试 `$GMXDATA/top`。

### 找不到配体残基

确认 `--ligand MOL` 与 `.gro/.top` 中的残基名一致。当前目录使用 `MOL`。

### 溶剂腿构建失败

常见原因是配体 XML/FFXML 不完整或 GROMACS 拓扑抽取失败。可尝试显式提供 `--ligand-xml`，或检查 `output/ligand_only.xml` 是否生成。

### Boresch 自动估计失败

当前推荐先用不依赖 ML 的：

```bash
--boresch --boresch-source simple
```

若锚点不稳定，检查 `output/boresch_simple.json`、`pre_equilibration.dcd` 和 Boresch harmonicity diagnostics。

### `--analyze-only` 缺少能量文件

至少需要保留阶段 checkpoint 或窗口级 `.npy` 能量文件。不要随意删除：

```text
output/checkpoints/stage1_decharging.json
output/checkpoints/stage2_vanishing.json
output/decharging/decharging_pme_u_kn.npy
output/vanishing/dual_window_*_energies.npy
```

### 结果里的 `thermodynamic_cycle` 和缺陷清单冲突

这是历史 provenance 文本缓存造成的已知问题。当前 README 和 `docs/status/AUDIT_STATUS.md` 的结论优先（`PHYSICS_DEFECTS.md` 已被 `docs/status/AUDIT_STATUS.md` 取代，文件已不存在）：APBS 不替代 LJ tail correction，手动 PME self `+C*lambda^2` 不作为生产修正项，`Delta G_bind = Delta G_solvent - Delta G_complex + Delta G_APBS`（不是 `Delta G_complex - Delta G_solvent`）。如果手头的 `output/final_binding_results.json`/`thermodynamic_cycle.md` 是在这几处文档修正之前生成的，其中的 `delta_G_bind_kJ_mol` 符号可能是反的，`thermodynamic_cycle` 字段文本也会是旧版本——重新跑一遍复合物腿+溶剂腿的最终汇总（不需要重新采样，只要 `complex_results`/`solv_results` 能从缓存加载）即可刷新成当前约定。

## 维护建议

修改代码后先做语法检查：

```bash
python -c "import ast, pathlib; files=['runabfe.py','abfe_pipeline.py','abfe_preoptimizer.py','ibs_engine.py','abfe_core.py']; [ast.parse(pathlib.Path(f).read_text(encoding='utf-8'), filename=f) for f in files]; print('syntax ok')"
```

然后运行：

```bash
python runabfe.py self-test
```

如果 self-test 与 `docs/status/AUDIT_STATUS.md` 的最新物理结论不一致，应更新测试和热力学循环文档，避免旧假设继续进入新 provenance。完整测试还需要 OpenMM、PyMBAR 和 pytest；缺少运行依赖时，语法检查通过不等价于端到端验证通过。

推荐下一步优先事项：

1. 在目标环境运行完整 `python -m pytest -q`，重点覆盖 fixed-H bank、native checkpoint、LRC 和 v12 冻结验证状态机。
2. 在真实 GPU 上复验 v12 的 `calibrated_pending_validation` 续验和 fixed-H `lambda_shield` 同步修复。
3. 对当前 Atenolol-rank11 配置做至少一次独立重复运行。
4. 根据 stage diagnostics 判断是否需要进一步加密 vanishing 阶段窗口或增加采样；其余源码级 P2 以 `docs/TODO.md` 为准。
