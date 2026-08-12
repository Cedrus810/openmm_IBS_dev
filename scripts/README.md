# 运行与实验脚本

[项目入口](../README.md) · [运行教程](../docs/GETTING_STARTED.md) · [测试入口](../tests/README.md)

生产 Python 入口是仓库根目录的 `runabfe.py`。本目录保存 PBS、实验、验证和批处理启动脚本；
脚本是否存在不代表对应路线已经获得 production 授权。

## 使用前检查

- 阅读脚本头部的体系、实验 ID、队列、环境和输出目录要求。
- 搜索硬编码的绝对路径、GPU 设备、账户、walltime 和 ligand 名称。
- 新体系使用新输出目录，不复用 Atenolol checkpoint。
- 先 dry-run 或小预算 smoke，再提交完整任务。
- 对 sealed 实验核对计划 hash、seed、replicate 和验收门。

## 脚本类别

- 通用运行/验证脚本：应接受配置、体系 ID 和输出路径参数；
- EXP 脚本：只在对应预注册或计划范围内使用；
- PBS/HPC 脚本：通常绑定特定集群，需要本机化；
- 诊断或恢复脚本：确认它是只读、追加写入还是会修改 checkpoint；
- `examples/atenolol-rank11/`：参考体系历史脚本，只供阅读和改造。

## EXP-011 formal umbrella launcher

`run_exp011_umbrella_grid.py` 校验冻结 sampling-plan hash，并 fail-closed 地续跑一个正式
replicate。先运行 dry-run：

```bash
python scripts/run_exp011_umbrella_grid.py --replicate formal_run1 --dry-run
```

只有单窗 smoke 通过后才扩大窗口数。目录非空但缺合格报告、参数与既有报告不一致或 hash
不匹配时，脚本应停止而不是猜测性续跑。

## 新增脚本规范

- 不把可复用逻辑复制进 shell；核心逻辑进入 Python 模块并接受显式参数。
- 输出日志记录命令、配置、环境、seed、输入 hash 和目标目录。
- 默认 fail closed；危险覆盖需要显式开关。
- 在脚本或配套文档中说明是否可恢复、是否幂等、会修改哪些文件。

