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
- **C++ 编译器 + CUDA toolkit**，仅在使用 `--outer-lambda-local-residual-ibs` 时需要 ——
  那个开关依赖一个**必须自己编译**的 OpenMM native plugin，见下节。
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

## CUDA 插件：**必须自己编译，仓库里不发编译产物**

> 只在用 `--outer-lambda-local-residual-ibs` 时才需要。**默认生产路径不加载插件**
> （开关默认 `false`），不用这个功能可以整节跳过。

`LocalManyBodyResidual` 是一个 OpenMM native plugin。仓库里**带了一份预编译
`.so`**，对着 **OpenMM 8.5.2** 编的 —— 而
[`environment.yml`](../environment.yml) / [`environment-ci.yml`](../environment-ci.yml)
钉的正是 `openmm=8.5.2`。**按环境文件建环境的话，直接能用，不用编。**

要自己编的只有两种情况：

1. 你没走环境文件，装的是**别的 OpenMM 版本**（`.so` 吃的是 OpenMM 私有平台头，
   跨版本不保证可用）；
2. 你改了插件源码。

重编是秒级的，出现加载期符号错误时第一个动作也是它。

### 怎么编

```bash
conda activate <装有 OpenMM 的环境>        # 脚本对 CONDA_PREFIX fail-closed，不给默认值

# CUDA 头文件在哪就指哪。按 environment.yml 建的环境自带 CUDA toolkit，用这行：
export CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux"
# 用系统 CUDA 的话是 /opt/cuda（脚本默认值）或 /usr/local/cuda。

bash plugins/LocalManyBodyResidual/g0_build.sh
```

> ⚠️ `g0_build.sh` 的 `CUDA_HOME` 默认值是 `/opt/cuda`，**但按 `environment.yml`
> 建出来的环境里 CUDA 头文件不在那儿**，而在 `$CONDA_PREFIX/targets/x86_64-linux/include`
> （环境里钉了 `cuda-nvcc=12.9.86`）。不设这个变量、机器上又没有 `/opt/cuda`，
> 就会在第 3 步报找不到 `cuda_runtime.h`。

产出三个 `.so` 到 `plugins/LocalManyBodyResidual/build/`：公有 API 层、Reference
平台层、CUDA 平台层。脚本是手写 `g++`，不是 CMake，编译时间是秒级。

### 为什么**换环境就必须重编**

`.so` 是对着**你这一个 OpenMM 安装**编出来的，而且绑得很死：

- 编译时吃 `$CONDA_PREFIX/include` 下的 OpenMM **公有 API + Reference 平台私有头
  + CUDA/Common 平台私有头**（私有头没有 ABI 稳定性承诺）；
- 链接 `-lOpenMM -lOpenMMCUDA`，`-rpath` 写死 `$CONDA_PREFIX/lib`。

⟹ 换 OpenMM 版本、换 conda 环境、甚至同版本换一次 build，都要重新跑一遍
`g0_build.sh`。**开发与验证用的是 OpenMM 8.5.2（`git_revision 36a30cb`）。**

### 显卡型号不用操心

插件只编译**宿主端 C++**，设备端 kernel 是 OpenMM 运行时用 NVRTC 现编的
（`CudaContext::createModule()`），源码里没有 `.cu`、没有 `-arch`、没有预编译
PTX/fatbin。所以**插件本身不锁定任何 SM 版本**。

支持哪些卡**由 mamba/conda 环境决定** —— [`environment.yml`](../environment.yml)
里钉的是 `cuda-version=12.9`（`cuda-nvrtc=12.9.86`），覆盖 **PTX sm_75 ~ sm_120**
（Turing 到 Blackwell），对绝大多数在用的卡够了。按环境文件建环境就自动拿到这个范围；
要支持更老的卡就在环境里换 CUDA 版本，不用改插件代码。

### 编完自检

```bash
python plugins/LocalManyBodyResidual/cuda_smoke_test.py   # 需要 GPU
```

还有一道**源码身份门**：loader 会核对
`platforms/cuda/src/CudaLocalManyBodyResidualKernels.cpp` 的 sha256 是否等于
`KNOWN_PLUGIN_SOURCE_SHA256`，对不上直接拒绝启用开关。
改了 kernel 源码就要同步更新那个常量（两处，见
[RETRAIN_LOCAL_RESIDUAL.md](RETRAIN_LOCAL_RESIDUAL.md)）。

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

### 只读诊断（不启动任何模拟）

跑之前先用这两个命令确认环境和配置，比直接开一次生产运行再等它报错便宜得多。
两个命令都不建 OpenMM Context、不积分、不写运行产物，有错误时退出码为 1。

```bash
python runabfe.py doctor                       # 环境体检
python runabfe.py doctor --output ./output     # 顺便查该目录所在盘的余量
python runabfe.py doctor --json                # 机器可读
```

`doctor` 报告：解释器版本、必需/可选依赖的版本（pymbar 不是契约钉住的
4.2.0 会告警）、OpenMM 真实可用平台、`nvidia-smi` 看到的 GPU 与显存、
GROMACS（`gmx` / `GMXLIB` / `GMXDATA` / 解析到的 include 目录）、
输出目录所在盘的可用空间、原生插件 build 目录里已构建的 `.so`。

```bash
python runabfe.py validate-config --config abfe_config.json
python runabfe.py validate-config --config abfe_config.json --json
```

`validate-config` 报告：

- **未知键与拼写**——例如 `temprature` 会被指出并建议 `temperature`。
  这类键运行期会被原样保留进配置快照、但**没有任何代码读它**，所以有效
  `temperature` 仍是默认值 300。只有命令行选项名（如 `preset`）会被单独说明
  "只能从命令行给"，不当拼写错误报。
- **类型与取值**——约束直接从 `runabfe.py` 的真 parser 上读，不维护第二份参数表。
- **输入路径**——`gro`/`top`/`ligand_xml` 等是否真的存在（换机器前必查）。
- **GROMACS include 目录**——`gmx_path` 实际会被解析成哪个目录。
- **生效参数及其来源**——每个关键参数是来自配置文件、预设，还是 argparse 默认值。
  注意这一节是「仅配置文件 + 预设」的解释结果，不含命令行覆盖。

```bash
python runabfe.py config-template > my_config.json      # 常用键 + 当前默认值 + 人工说明
python runabfe.py config-template --all --out my.json   # 连不常用/实验开关一起列出
python runabfe.py config-template --preset high_accuracy
```

`config-template` 打印一份可以直接改的配置模板。键顺序和那些人工写的
`_comment_*` 说明取自仓库的 `abfe_config.json`；**取值**是重新解析出来的
（预设 > 该文件的生产值 > argparse 默认），所以模板里不会带上任何机器本地路径。
`gro`/`top`/`ligand`/`gmx_path` 一律留空，需要自己填。生成后：

```bash
python runabfe.py validate-config --config my_config.json
```

模板本身是零错误通过自查的，所以报出来的错都是你改出来的。

模板**不会**告诉你的：Stage 2 的实际窗口划分。那是运行期 Fisher 探针跑出来的
结果（`stage2_n_states` 是探针网格点数，不是最终生产窗口数），配置里没有这个数字。

⚠️ 运行期**不会**因为未知键拒绝启动，这是有意的：启动期硬拒绝未知键曾经直接
炸掉 resume（见 `runabfe.py` `RunConfig.__init__` 里 2026-08-24 的注释）。
拼写检查只在 `validate-config` 里生效。

### 其他子命令

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
