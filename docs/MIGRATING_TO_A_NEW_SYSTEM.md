# 迁移到新的蛋白–配体体系

项目名称是 **ABFE-IBS workflow**。任何单个体系的名字都不应出现在通用模块名、
测试名或新体系的输出名称中。

> 📅 **2026-09-02 更新**：本文 2026-07-29 成稿时，参考体系是 Atenolol（工作区
> `Atenolol-rank11`）。**现在的主线体系是 4W53（T4 lysozyme L99A + toluene）**，
> 而 Atenolol 那条线的 `output_lrc_fix` 已于 2026-08-24 判定作废
> （见 [../README_cn.md](../README_cn.md)《当前科学状态》）。
> 下面的**迁移原则和步骤本身与体系无关、仍然适用**，只有提到具体体系名的地方
> 已按现状更正。

## 迁移原则

1. 复用代码和测试，不复用旧体系的 checkpoint、轨迹、缓存 XML/CIF 或结果 JSON。
2. 为新体系使用独立目录和独立输出名，例如 `systems/<system-id>/` 与
   `runs/<system-id>/<run-id>/`。**这条布局至今没有被强制执行**：4W53 实际用的是
   `/home/ruigengji/ABFE_IBS/4W53/output_v3_seed<seed>/` 这种「体系目录 + 带 seed
   的输出目录」写法。两种都能跑；关键是**新体系必须有自己的目录和自己的输出名**，
   不要往别的体系目录里写。
   （本条原文的理由是「以免打断 `output_lrc_fix` 的收尾」—— 那条线已于
   2026-08-24 作废，理由不再成立，但结论没变。）
3. 配体残基名不是固定的 `MOL`。必须从新体系的 `.gro/.top` 核对，并通过配置或
   `--ligand` 显式设置。
4. `gmx_path`、平台、离子强度、温度和采样预算都要针对目标机器/体系复核。
5. 新体系第一次运行必须使用全新的输出目录；禁止对旧体系目录执行 `--resume`。

## 推荐步骤

### 1. 先验证通用代码

在仓库根目录运行：

```bash
./tests/run_offline_tests.sh
```

### 2. 准备新体系输入

创建独立目录并放入相互一致的 GROMACS 输入：

```text
systems/<system-id>/
├── config.json
└── inputs/
    ├── system.gro
    ├── topol.top
    └── *.itp
```

`topol.top` 中的相对 include 必须能在新目录结构下解析。不要从旧体系只复制
`topol.top` 而漏掉对应的 `.itp`/位置约束文件。

### 3. 创建体系配置

从当前 `abfe_config.json` 复制参数结构，但至少替换：

```json
{
  "output": "./runs/<system-id>/<run-id>",
  "gro": "./systems/<system-id>/inputs/system.gro",
  "top": "./systems/<system-id>/inputs/topol.top",
  "ligand": "<真实配体残基名>",
  "resume": false,
  "reset": false
}
```

当前配置中的 `_comment_lambda_refine` 和局部 repair 背景属于旧运行历史，不能直接
当作新体系的科学依据。

### 4. 做输入与路径预检

```bash
python runabfe.py   --config systems/<system-id>/config.json   --ligand <真实配体残基名>   --output runs/<system-id>/<run-id>
```

首次提交前检查日志中的实际 `.gro/.top` 路径、配体原子数、盒尺寸、离子数、
Boresch 锚点和平台。发现程序读取了旧目录时立即停止。

### 5. 保留体系级结论

每个体系单独记录：

- 输入文件来源和校验值；
- 配体残基名与电荷；
- 配置快照和代码版本；
- 运行命令、环境与硬件；
- 最终结果及其适用限制。

不要把某个体系的实测阈值或修复说明写成通用默认值；若确实要进入通用代码，必须
增加跨体系测试和明确的物理依据。

## 目录名：这件事已经做完了

> 🗄️ **本节所描述的迁移已于 2026-08-31 完成，保留为历史记录。**

本节原文说「当前磁盘目录仍叫 `Atenolol-rank11` …… 完成封版后再改成通用名称
（例如 `ABFE-IBS`）」。**这件事已经发生了，但走的是另一条路**：没有重命名旧目录，
而是新建了一个干净的工程区分支。

现状：

- **主线库 = `/home/ruigengji/ABFE_IBS/ABFE_IBS`**（新 git 库，起点 commit
  `eaf1c7e`，2026-08-31 建立）—— 只含生产代码、生产回归测试、诊断工具、文档；
- `Atenolol-rank11` **仍然存在且保留**，但不再是主线：它是开发期工地，
  存放 `output*`、验证轨迹、实验脚本、失败记录和逐条决策历史。
  本仓库对它的引用一律带 `（在 Atenolol-rank11，不在本仓）` 标记，
  逐份登记在 [HISTORY_LOG.md](HISTORY_LOG.md)。

⟹ **不需要再重命名任何东西。** 目录布局约定见
[../PROJECT_LAYOUT.md](../PROJECT_LAYOUT.md)。
