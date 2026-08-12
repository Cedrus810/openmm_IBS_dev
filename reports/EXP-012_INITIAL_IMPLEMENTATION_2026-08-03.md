# EXP-012 初始工程切片与结论（2026-08-03）

## 完成范围

- 将已关闭的 EXP-010/EXP-011/WP0-support 历史实现迁至
  `archive/outer_lambda_exp010_exp011_legacy.py`。
- `outer_lambda_neural_basis.py` 保留同名 lazy compatibility wrappers，旧 CLI、公开导入和
  历史复现入口不变；主文件由 10,668 行降至 9,136 行。
- 新建独立 `exp012_xed` 包；production 的 `runabfe.py`、`abfe_core.py`、
  `abfe_pipeline.py`、`ibs_engine.py` 均未导入或修改。
- 新建 EXP-012 preregistration 草案，登记真实 topology/System/box/protocol/run 哈希、
  五态 λ/f_k 和三个 train/validation/test whole-run folds。
- schema 对路径逃逸、SHA 格式、run 泄漏、局部窗口误当物理端点、reduced-potential
  单位、A/B/C 继承关系、未 sealed 执行及 sealed payload 篡改执行 fail-closed 检查。

## 验证

```text
./tests/run_offline_tests.sh -q \
  tests/test_neural_basis_ibs_accounting.py \
  tests/test_exp012_schema.py

58 passed in 5.42s
```

## 当前结论

1. 旧 EXP-010/011 物理实现已经与活跃主线隔离，同时没有切断历史复现接口。
2. EXP-012 的输入身份和防轨迹泄漏切分现在可机器校验，但 preregistration 仍是 draft，
   不能执行正式 A/B/C diagnostic。
3. 三条独立 scratch DCD 各有 500 帧，几何输入可用；它们没有逐帧五态 target energy /
   reduced-potential ledger，也没有对应 target/MBAR weights。这是当前第一数据阻塞。
4. 下一步应先重建并核对 scratch window System，对 1,500 帧重标完整五态 MM ledger；
   不能使用未对齐的历史 production energy matrix，也不能使用 EXP-010 MACE teacher 或
   EXP-011 torsion PMF 代替。
5. 本切片没有产生 XED feature、模型、Force、NVT 或 production approval，也没有提供
   XED 有效性的科学证据。
