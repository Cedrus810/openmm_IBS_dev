# 维护与修改代码

[返回项目首页](../README.md) · [目录约定](../PROJECT_LAYOUT.md) ·
[测试说明](../tests/README.md)

## 维护建议

修改代码后先做语法检查：

```bash
python -c "import ast, pathlib; files=['runabfe.py','abfe_pipeline.py','abfe_preoptimizer.py','ibs_engine.py','abfe_core.py']; [ast.parse(pathlib.Path(f).read_text(encoding='utf-8'), filename=f) for f in files]; print('syntax ok')"
```

然后运行：

```bash
python runabfe.py self-test
```

如果 self-test 与 `status/AUDIT_STATUS.md` 的最新物理结论不一致，应更新测试和热力学循环文档，避免旧假设继续进入新 provenance。完整测试还需要 OpenMM、PyMBAR 和 pytest；缺少运行依赖时，语法检查通过不等价于端到端验证通过。

推荐下一步优先事项：

1. 在目标环境运行完整 `python -m pytest -q`，重点覆盖 fixed-H bank、native checkpoint、LRC 和 v12 冻结验证状态机。
2. 在真实 GPU 上复验 v12 的 `calibrated_pending_validation` 续验和 fixed-H `lambda_shield` 同步修复。
3. 对目标体系的最终配置做至少一次独立重复运行。
4. 根据 stage diagnostics 判断是否需要进一步加密 vanishing 阶段窗口或增加采样；其余源码级 P2 以 `TODO.md` 为准。


## 最低验证

```bash
./tests/run_offline_tests.sh
```

新增测试只能放在 `tests/`；一次性诊断、修复和绘图脚本分别放入
`tools/diagnostics/`、`tools/repairs/` 和 `tools/plots/`。历史补丁与源码副本只放
`archive/`，不得被生产代码导入。
