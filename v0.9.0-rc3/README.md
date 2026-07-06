# ABFE/IBS Pipeline

这是一个基于 OpenMM 的 ABFE（Absolute Binding Free Energy）流水线，用于从 GROMACS 复合物体系出发，自动完成复合物腿、溶剂腿、Boresch 限制修正、窗口采样和最终结合自由能汇总。

默认推荐路线是 `ibs + dual_lambda + softcore`。传统 REMD 路线仍保留用于对照或兼容旧流程。

## 核心功能

- 从 GROMACS `gro/top` 构建 OpenMM 原生系统缓存。
- 自动构建 ligand-in-water 溶剂腿缓存。
- 支持复合物腿 Boresch 限制力与解析修正。
- 支持 `dual_lambda`、`single_lambda`、`2d_diagonal`、`2d_geodesic` 解耦方案。
- 支持 ACES softcore 与 DEXP 势模型。
- 支持 IBS 重叠窗口采样、预优化 lambda 路径、MBAR/TMBAR 分析。
- 支持 checkpoint、阶段缓存、`--resume` 续跑和 `--analyze-only` 后处理。

## 代码结构

主要逻辑集中在 5 个文件：

- `runabfe.py`: 命令行入口。负责参数解析、配置合并、系统缓存、Boresch 参数、复合物腿/溶剂腿调度和最终结果汇总。
- `abfe_pipeline.py`: 单腿 ABFE 流程。负责预平衡、安全松弛、路径预优化、采样、Boresch 修正和单腿结果写出。
- `abfe_preoptimizer.py`: lambda 路径预优化。负责方差探测、自适应状态分布和双 lambda 阶段路径生成。
- `ibs_engine.py`: IBS 采样与分析引擎。负责窗口划分、窗口模拟、能量矩阵保存、checkpoint 和 MBAR/TMBAR 求解。
- `abfe_core.py`: 物理核心库。提供 softcore/DEXP 势、Boresch 限制力、候选锚点估计、解析修正、单位格式化和系统工具。

## 依赖

运行环境需要能导入以下核心包：

- Python 3.10+
- OpenMM
- NumPy
- SciPy
- MDTraj
- PyMBAR

可选依赖取决于功能：

- CUDA/OpenCL 平台用于 GPU 运行。
- GROMACS force field include 目录用于从 `.top` 构建系统。
- OpenMM-ML、MACE/Orb 相关依赖仅在使用对应 ML 估计或拟合功能时需要。

当前仓库不负责安装依赖；请使用你自己的 conda/mamba/pip 环境准备运行时。

## 输入文件

首次运行通常需要：

- `--gro`: GROMACS 坐标文件，例如复合物水盒 `complex.gro`。
- `--top`: GROMACS 拓扑文件，例如 `topol.top`。
- `--ligand`: 配体残基名，例如 `MOL`。
- `--gmx-path`: GROMACS force field include 目录，视拓扑内容而定。
- `--ligand-xml`: 可选，构建溶剂腿时使用的配体 FFXML/XML。

也可以把这些字段写入配置文件，再用 `--config` 读取。

## 快速开始

首次运行：

```bash
python runabfe.py \
  --gro complex.gro \
  --top topol.top \
  --ligand MOL \
  --gmx-path /path/to/gromacs/share/gromacs/top \
  --output ./output \
  --preset production \
  --boresch \
  --boresch-source auto
```

使用配置文件：

```bash
python runabfe.py --config abfe_config.json --output ./output --ligand MOL
```

续跑：

```bash
python runabfe.py --config abfe_config.json --output ./output --ligand MOL --resume
```

强制忽略缓存重新开始：

```bash
python runabfe.py --config abfe_config.json --output ./output --ligand MOL --reset
```

只分析已有结果：

```bash
python runabfe.py --config abfe_config.json --output ./output --ligand MOL --analyze-only
```

传统 REMD 模式：

```bash
python runabfe.py --mode traditional --config abfe_config.json --output ./output --ligand MOL
```

## 配置文件

`--config` 支持 JSON/YAML 风格配置加载。命令行显式传入的参数优先级高于配置文件。

常用配置示例：

```json
{
  "preset": "production",
  "platform": "CUDA",
  "output": "./output",
  "temperature": 300.0,

  "gro": "complex.gro",
  "top": "topol.top",
  "ligand": "MOL",
  "gmx_path": "/path/to/gromacs/share/gromacs/top",

  "decoupling": "dual_lambda",
  "potential": "softcore",

  "n_steps_per_window": 250000,
  "steps_per_update": 500,
  "stage1_n_states": 16,
  "stage2_n_states": 16,

  "boresch": true,
  "boresch_source": "auto",
  "boresch_batch": 5,
  "boresch_select": 1,

  "skip_rebalance": false,
  "rebalance_steps": 50000,

  "enable_gradual_warmup": true,
  "warmup_steps": 500000,

  "resume": false,
  "reset": false
}
```

注意：仓库里的 `abfe_config.json` 只是示例/兼容配置。实际运行前请确认 `gro`、`top`、`ligand`、`gmx_path`、`output` 都符合你的体系和机器环境。

## 主要参数

基础输入：

- `--gro`: GROMACS 坐标文件，首次构建系统时需要。
- `--top`: GROMACS 拓扑文件，首次构建系统时需要。
- `--ligand`: 配体残基名。
- `--ligand-xml`: 配体 XML/FFXML，用于溶剂腿构建。
- `--gmx-path`: GROMACS force field include 目录。
- `--output`: 输出目录，同时也是缓存目录。
- `--config`: 配置文件。

运行控制：

- `--resume`: 从 checkpoint 和缓存续跑。
- `--reset`: 忽略已有缓存，强制重新开始。
- `--analyze-only`: 只分析已有能量文件和 checkpoint。
- `--parallel-stages`: 并行执行去电荷和去 vdW 阶段。CUDA 下建议通过环境变量给两阶段指定不同 GPU。

采样路线：

- `--mode`: `ibs` 或 `traditional`。
- `--decoupling`: `dual_lambda`、`single_lambda`、`2d_diagonal`、`2d_geodesic`。
- `--potential`: `softcore` 或 `dexp`。
- `--dexp-params`: DEXP 参数 JSON。

Boresch：

- `--boresch` / `--no-boresch`: 开启或关闭复合物腿 Boresch 限制。
- `--boresch-source`: `auto`、`simple`、`fluctuation`、`traditional`、`orb_ml`。
- `--boresch-anchors`: 外部传统 Boresch 锚点文件。
- `--boresch-orb`: Orb/ML 预测的 Boresch 文件。
- `--boresch-batch`: 自动估计候选数量。
- `--boresch-select`: 选择第 N 个候选。
- `--skip-rebalance`: 跳过带 Boresch 限制的再平衡。
- `--rebalance-steps`: Boresch 再平衡步数。

采样预算：

- `--preset`: 预设强度。
- `--n-steps-per-window`: 每个窗口生产步数。
- `--steps-per-update`: 采样更新间隔。
- `--n-states-per-stage`: 同时设置两个阶段的 lambda 状态数。
- `--temperature`: 温度，单位 K。
- `--platform`: `CUDA`、`OpenCL` 或 `CPU`。
- `--enable-early-stop`: 启用提前停止。
- `--enable-gradual-warmup` / `--disable-warmup`: 控制渐进预热。
- `--warmup-steps`: 预热步数。

## 预处理子命令

`prepare` 子命令用于单独生成预处理文件，例如 Boresch 参数或 DEXP 拟合结果：

```bash
python runabfe.py prepare \
  --gro complex.gro \
  --top topol.top \
  --ligand MOL \
  --gmx-path /path/to/gromacs/share/gromacs/top \
  --output-dir ./prep_output \
  --save-boresch boresch.json
```

DEXP 拟合示例：

```bash
python runabfe.py prepare \
  --gro complex.gro \
  --top topol.top \
  --ligand MOL \
  --output-dir ./prep_output \
  --fit-dexp \
  --save-dexp dexp_params.json
```

## 输出目录

默认输出根目录是 `./output`，可用 `--output` 修改。

常见文件：

- `output/run_provenance.json`: 本次运行输入、配置和体系摘要。
- `output/pipeline.log`: 复合物腿 pipeline 日志。
- `output/final_results.json`: 复合物腿单腿结果。
- `output/final_binding_results.json`: 最终结合自由能结果。
- `output/final_results_postprocess.json`: `--analyze-only` 生成的后处理结果。
- `output/thermodynamic_cycle.md`: 热力学循环说明。
- `output/boresch_*.json`: Boresch 参数缓存。
- `output/boresch_params.json`: pipeline 使用/格式化后的 Boresch 参数。
- `output/checkpoints/`: 复合物腿 checkpoint 和阶段状态。
- `output/solvent_leg/`: 溶剂腿输出目录。
- `output/solvent_leg/final_results.json`: 溶剂腿单腿结果。
- `output/solvent_leg/checkpoints/`: 溶剂腿 checkpoint。
- `output/traditional_complex/`、`output/traditional_solvent/`: traditional 模式输出。

窗口级能量文件通常位于阶段目录中，例如：

- `dual_window_*_energies.npy`
- `dual_window_*_bias.npy`
- `dual_window_*_base.npy`
- `dual_window_*_convergence.json`

这些文件是 MBAR/TMBAR 后处理的核心输入，清理前请确认你不再需要 `--analyze-only`。

## 计算流程

默认 `ibs` 模式会执行：

1. 从缓存加载系统；若缓存不存在，则从 `gro/top` 构建并写入 OpenMM 原生缓存。
2. 自动检查/构建 ligand-in-water 溶剂腿缓存。
3. 解析或估计复合物腿 Boresch 参数。
4. 可选执行带 Boresch 限制力的再平衡。
5. 运行复合物腿：预平衡、安全松弛、路径预优化、窗口采样、MBAR/TMBAR 分析。
6. 运行溶剂腿：使用同一解耦方案，但强制不使用 Boresch 限制。
7. 汇总：

```text
Delta G_bind = Delta G_complex - Delta G_solvent + optional external correction
```

Boresch 解析修正已经包含在复合物腿结果中；溶剂腿不加 Boresch。

## 双 lambda 默认路线

`dual_lambda` 将去耦合分成两段：

- Stage 1: decharging，通常固定 vdW，关闭电荷。
- Stage 2: vanishing，电荷关闭后继续关闭 vdW。

这种拆分通常比单一路径更稳定，也便于分别优化两个困难区域。`abfe_preoptimizer.py` 会依据探测采样中的能量波动调整 lambda 点分布。

## 续跑与缓存

推荐把每个体系放在独立输出目录中。`--resume` 会尝试复用：

- 系统原生缓存。
- 预平衡 checkpoint 和轨迹。
- Boresch 参数缓存。
- 预优化 lambda 路径缓存。
- 阶段采样结果和 MBAR 分析结果。

当你修改以下内容时，建议使用 `--reset` 或更换 `--output`：

- 输入结构或拓扑。
- 配体残基名。
- 关键采样状态数。
- 解耦方案。
- 势模型或 DEXP 参数。
- Boresch 来源或锚点选择。

## 并行阶段

`--parallel-stages` 会尝试并行运行 decharging 和 vanishing。CUDA 下如果两阶段共用同一 GPU，代码会回退为串行以避免上下文冲突。

可用环境变量指定两个不同 GPU：

```bash
IBS_STAGE1_CUDA_DEVICE=0 IBS_STAGE2_CUDA_DEVICE=1 \
python runabfe.py --config config.json --parallel-stages
```

Windows PowerShell 示例：

```powershell
$env:IBS_STAGE1_CUDA_DEVICE = "0"
$env:IBS_STAGE2_CUDA_DEVICE = "1"
python runabfe.py --config config.json --parallel-stages
```

## 常见问题

### `ModuleNotFoundError: No module named 'openmm'`

当前 Python 环境没有安装 OpenMM，或没有激活正确 conda 环境。先确认：

```bash
python -c "import openmm; print(openmm.__version__)"
```

### 首次运行找不到溶剂腿缓存

默认主流程会尝试自动构建溶剂腿。若失败，通常是以下原因：

- `--ligand` 残基名和拓扑中的残基名不一致。
- `--ligand-xml` 缺失，导致单独配体水盒无法参数化。
- `--gmx-path` 不正确，GROMACS include 文件找不到。

### Boresch 自动估计失败

可尝试：

- 改用 `--boresch-source simple`。
- 增加 `--boresch-batch` 并用 `--boresch-select` 选择候选。
- 提供外部锚点文件并使用 `--boresch-source traditional --boresch-anchors file.json`。
- 临时用 `--no-boresch` 做无约束调试运行。

### `--analyze-only` 缺少能量文件

后处理需要窗口级 `.npy` 能量文件或阶段 checkpoint。若你清理过中间文件，请保留：

- `checkpoints/stage1_decharging.json`
- `checkpoints/stage2_vanishing.json`
- 或完整的 `dual_window_*_energies.npy` / `_bias.npy` / `_base.npy`

### GPU 不可用或平台选择错误

确认 `--platform` 与 OpenMM 安装匹配。可以先用 CPU 做小规模 smoke test：

```bash
python runabfe.py --config config.json --platform CPU --n-steps-per-window 1000 --n-states-per-stage 4
```

## 维护建议

修改核心代码后至少做一次语法检查：

```bash
python -c "import ast, pathlib; files=['runabfe.py','abfe_pipeline.py','abfe_preoptimizer.py','ibs_engine.py','abfe_core.py']; [ast.parse(pathlib.Path(f).read_text(encoding='utf-8'), filename=f) for f in files]; print('syntax ok')"
```

如果运行环境完整，也建议执行：

```bash
python runabfe.py self-test
```

再用很小的采样预算跑一次 smoke test，确认系统构建、Boresch、复合物腿和溶剂腿都能进入预期阶段。
