# 迁移到新的蛋白–配体体系

项目名称是 **ABFE-IBS workflow**。`Atenolol-rank11` 只是当前工作区中的参考体系，
不应出现在通用模块名、测试名或新体系的输出名称中。

## 迁移原则

1. 复用代码和测试，不复用旧体系的 checkpoint、轨迹、缓存 XML/CIF 或结果 JSON。
2. 为新体系使用独立目录和独立输出名，例如 `systems/<system-id>/` 与
   `runs/<system-id>/<run-id>/`。当前活跃工作区尚未强制迁移到该布局，以免打断
   `output_lrc_fix` 的收尾。
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

## 当前目录名

当前磁盘目录仍叫 `Atenolol-rank11`，因为它同时是活跃 Codex 工作区和
`output_lrc_fix` 的路径。完成封版、关闭相关进程并确认无脚本依赖绝对路径后，再在
工作区之外把目录改成通用名称（例如 `ABFE-IBS`）。不要在活跃运行期间重命名。
