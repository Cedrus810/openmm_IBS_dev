# 运行脚本

生产 Python 入口仍是仓库根目录的 `runabfe.py`。

`examples/atenolol-rank11/` 保存当前参考体系的历史 PBS 与专项实验脚本。这些脚本
含机器、环境、队列、体系目录和输出目录的硬编码，只能作为示例阅读；迁移到新体系
时必须复制后逐项修改，不能直接提交。

通用脚本不得写死某个配体/体系名称。若新增可复用入口，应接受配置路径、体系 ID 和
输出目录参数，并把体系实例留在调用方。

## EXP-011 formal umbrella launcher

`run_exp011_umbrella_grid.py` 校验冻结 sampling-plan 哈希，并以 fail-closed 方式断点续跑
一个正式 replicate。默认只运行一个尚未完成的中心；只有单窗正式 smoke 验收后才使用
`--max-windows 24`。发现非空但无合格报告的目录，或已有报告参数不一致时会停止。

dry-run 示例：

```bash
python scripts/run_exp011_umbrella_grid.py --replicate formal_run1 --dry-run
```
