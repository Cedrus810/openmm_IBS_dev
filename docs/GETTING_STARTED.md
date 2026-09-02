# 安装、输入与运行

> 本教程从原根目录 README 拆出。示例命令使用当前参考体系的
> `abfe_config.json` 和配体残基名 `MOL`；迁移到其他体系时必须替换这些值，并先阅读
> [迁移到新体系](MIGRATING_TO_A_NEW_SYSTEM.md)。

[返回项目首页](../README.md) · [文档导航](README.md)

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

版本以仓库里的环境文件为准（[`environment.yml`](../environment.yml) /
[`environment-ci.yml`](../environment-ci.yml)）：`python=3.12`，**`pymbar-core=4.2.0`（钉死）**。

> ⚠️ **`pymbar-core` 的版本是钉死的，不是"建议"。** 理由见
> [PYMBAR_UNCERTAINTY_PROTOCOL.md](PYMBAR_UNCERTAINTY_PROTOCOL.md)：报告出去的 ABFE
> 不确定度不能因为换了个 environment 就静默改变。换版本要走那份文档里的完整流程。
>
> 本节早期版本写的是「`output/run_provenance.json` 记录 …… PyMBAR 4.0.3」——
> 那个版本号**与当前 pin 冲突**，且本工程区分支的 `output/` 是空目录、没有
> `run_provenance.json`（它是跑起来才生成的）。已按环境文件更正。

实际跑的时候以运行产物 `<output>/run_provenance.json` 里记录的 `pymbar.__version__`
为准 —— 那才是那次运行真正导入的版本。

## 输入要求

首次构建缓存通常需要：

- `--gro`：GROMACS 坐标文件。当前配置使用 `solv_ions.gro`。
- `--top`：GROMACS 拓扑文件。当前配置使用 `topol.top`。
- `--ligand`：配体残基名。当前配置使用 `MOL`。
- `--gmx-path`：GROMACS 力场目录。**写安装前缀（如 `/opt/gromacs`）或写全到
  `share/gromacs/top` 都可以** —— 给前缀时会自动往下找 `share/gromacs/top`、`top`
  子目录（`runabfe.py:557` `find_gmx_include_dir` 第 1 步）。也可通过 `GMXLIB`
  （值本身就是力场目录）、`GMXDATA`（力场在其 `top/` 下）或 PATH 上的 `gmx` 自动探测。
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
