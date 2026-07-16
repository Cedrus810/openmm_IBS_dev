# 运行验证监控表

更新日期：2026-07-16

本文件只跟踪“代码已经修复，但尚未获得完整运行证据”的项目。尚未实现的代码工作放在
`todolist.md`；问题背景、修复理由和最终审计结论放在 `AUDIT_STATUS.md`。

## 状态定义

| 状态 | 含义 |
|---|---|
| `待运行` | 修复与测试代码已准备，但尚未在目标环境执行 |
| `运行中` | 已启动验证，尚未得到完整结论 |
| `通过` | 验收条件全部满足，证据已归档 |
| `失败` | 已得到可复现的失败证据，需要重新进入 `todolist.md` |
| `阻塞` | 缺少环境、硬件、输入或外部依赖，暂时无法运行 |

## 当前监控矩阵

| ID | GitHub Issue | 验证对象 | 代码状态 | 已有静态/回归覆盖 | 目标环境 | 当前状态 | 最后更新 |
|---|---|---|---|---|---|---|---|
| `VAL-GPU-001` | [`#2`](https://github.com/Cedrus810/openmm_IBS_dev/issues/2) | IBS v12 `calibrated_pending_validation` 状态机与精确 log-sum-exp | 已修复 | 状态持久化、终态和累计预算测试已补 | OpenMM + CUDA | `待运行` / Project `In review` | 2026-07-16 |
| `VAL-GPU-002` | [`#3`](https://github.com/Cedrus810/openmm_IBS_dev/issues/3) | fixed-H bias 校准探针的 `lambda_shield` 同步与 NaN 修复 | 已修复 | 构建/最小化/步长爬升路径已有代码检查 | OpenMM + CUDA | `待运行` / Project `In review` | 2026-07-16 |
| `VAL-TEST-001` | [`#4`](https://github.com/Cedrus810/openmm_IBS_dev/issues/4) | 13 项 P2 修复与主窗口 native resume 的完整测试套件 | 已修复 | `ast.parse` / `py_compile` 已通过，针对性测试已补 | `openmm_dev` + OpenMM + PyMBAR + pytest | `待运行` / Project `In review` | 2026-07-16 |

## 验收条件与证据

### VAL-GPU-001 — v12 冻结校准续验

验收条件：

- 复跑曾出现 `p=[0.8485, 0.1502, 0.00126]` 的 VDW 窗口。
- MBAR 校准后按 50k→150k→300k **累计目标预算**续验，每档只运行尚未完成的差值。
- resume 保持冻结 `f_k`，不进入 SGD learning，不误触发拆窗或插 λ。
- 主窗口 native checkpoint 能恢复坐标、速度、盒子和积分器 RNG 状态；指纹漂移时拒绝复用。
- 最后一档仍失败时进入 `calibrated_validation_failed`，不再自动无限重试。
- 精确 log-sum-exp 偏置在此前正常窗口上无收敛回归。

建议证据：运行命令、作业 ID、代码 SHA256、对应的 `ibs_state_*.json`、
`dual_window_*_convergence.json`、主窗口 checkpoint manifest、窗口日志和最终判断。

### VAL-GPU-002 — fixed-H `lambda_shield` NaN 修复

验收条件：

- 复跑原窗口 `(5,9)`，bias 校准探针在真实 `lambda_shield` 和本态 CV 下完成局部最小化。
- burn-in 第一阶段不再出现 `Particle coordinate is NaN`。
- 边 `[5,6]`/`[6,7]` 若仍得到约 0.0037/0.0076 的真实低 overlap，应进入正常插 λ 流程；
  不能把物理低 overlap 与已修复的数值 NaN 混为一谈。
- 用 `nvidia-smi` 同时记录 fixed-H bank 的 Context/显存峰值，确认 evaluator 释放后没有
  无界累积旧 dynamics Context。

建议证据：窗口日志、`production_fixed_h_overlap.json`、probe manifest、逐边 overlap、
异常栈（若失败）和 `nvidia-smi` 采样记录。

### VAL-TEST-001 — 完整依赖测试

目标命令：

```bash
python -m pytest -q
```

除完整测试套件外，至少确认：

- fixed-H bank 短 segment 不进入 MBAR，缺失 `volume.npy` 会判坏并重采。
- evaluator Context 释放后显存/Context 数量符合预期。
- 主窗口 native checkpoint roundtrip、内容指纹拒绝和累计验证步数正确。
- Shadow-Bridge/Shadow-IBS 任一子腿低重叠时 fail closed。
- `finalize_descending_lambda_path` 在退化输入下仍满足端点、单调性和最小间距。
- APBS 网格体积与各向异性盒保护测试通过。
- 2D geodesic 分段积分会比旧端点近似更保守地避开高方差脊。

建议证据：环境导出、完整 pytest 输出、失败用例日志、GPU/平台信息和代码 SHA256。

## 更新规则

1. 尚未实现或需要继续改代码：加入 `todolist.md`，监控行标为 `失败` 并链接对应待办。
2. 代码修复完成但尚未实测：从 `todolist.md` 移到本表，状态设为 `待运行`。
3. 开始作业：状态改为 `运行中`，补充命令、作业 ID、输出目录和开始时间。
4. 验证完成：记录证据路径；通过则把结论追加到 `AUDIT_STATUS.md`，失败则重新进入
   `todolist.md`。
5. 不用“代码已修复”代替“运行验证通过”，也不因缺少运行环境把项目标成通过。
