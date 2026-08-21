# 带电膜 charge-transfer 工程基础合并到主线计划

目标仓库：`D:\ABFE_IBS\Atenolol-rank11`  
执行环境：Windows + WSL（文件级项目整合）  
状态：已合并到远端 dev；完整 pytest 待可用 OpenMM runtime 复核  
范围：合并 B3–B5、C1–C3、MEM-00h 与 Stage2 handoff；不执行 C4/C5，不宣称 production-qualified。

## 0. 2026-08-13 实际执行说明

当前目录没有可用的 Git HEAD/历史，因此本次“合并”按用户确认采用**项目文件级整合**，
没有执行 git init、commit、merge 或伪造历史。下文第 3/8/9 节的 Git 命令仅保留为未来
拿到完整仓库元数据后的可选发布步骤，不属于本次验收。

已完成：核心 qualification 状态、pipeline/run provenance、正式 binding result、
analyze-only postprocess 结果出口、配置/清单说明、fail-closed 契约测试均已接线；
C4/C5 仍未执行，production_qualified 仍为 false。未重新计算 SHA/hash，未重跑 MD，
C3 v1/v2 历史证据未覆盖。

本机完成的验证：相关 Python 文件 py_compile 全过，abfe_config.json 可解析，
post-analysis qualification 的独立 AST 契约检查通过。完整 pytest 未在当前会话执行：
D 盘既有 openmm_dev 环境缺少可用 C++ runtime 链接，Windows Python 又没有
OpenMM/pytest；没有擅自安装或升级依赖。

远端发布记录：GitHub 仓库 Cedrus810/openmm_IBS_dev，以 dev 为 base；PR #88 已合并，
merge commit 为 110f4325bbe94774553c8bf158388b4fd195dfb2。用户指定的主线范围是
runabfe.py、abfe_core.py、abfe_pipeline.py、ibs_engine.py、abfe_preoptimizer.py；
其中 abfe_preoptimizer.py 与 dev 已实质相同，因此 merge commit 的真实 diff 为其余四个。
当前 D 盘工作区的五个文件已逐文本核对与 PR 分支一致。

## 1. 定位与边界

将 co-alchemical charge-transfer 基础能力以 **experimental、fail-closed** 形式合入主线，并保持中性配体默认路径不变。

纳入：charge-treatment、reserved neutral dummy、charging Hamiltonian、co-ion identity/restraint/cache/resume/provenance、Stage1→Stage2 baked handoff、MEM-00h、C1–C3 validation/tests。

不纳入：C4/C5、长 GPU MD、目标体系生产自由能、production-grade 宣称、新 Hamiltonian 设计。

## 2. 执行纪律

1. 先读 `AGENTS.md`、`memtodolist.md`。
2. 保留用户未提交改动；禁止 reset/checkout 覆盖。
3. 保留 `validation/c3_real_endpoints_v1/` 与 v2 中间失败证据。
4. 不重新计算 SHA/hash；沿用已有 manifest/report 字段。
5. 只跑离线测试和必要 CPU smoke，不重跑 GPU/MD。
6. 不提交 DCD、NPZ、checkpoint 等大型产物。

## 3. 建立变更清单

```bash
git status --short
git branch --show-current
git branch -a
git log --oneline --decorate -n 20
git diff --stat
git diff --name-only
```

确认主线名称后：

```bash
git diff <main-branch>...HEAD --stat
git diff <main-branch>...HEAD --name-status
```

分类检查：`abfe_core.py`/`ibs_engine.py`；`abfe_pipeline.py`/`runabfe.py`；配置与文档；`tools/validation/`；`tests/`；小型审计 JSON/Markdown。无关改动不混入、不删除。

## 4. 合并前收口

### 4.1 文档一致性

- 修正 `abfe_config.json` 中“B3 尚未实现”的旧注释。
- 修正 `memtodolist.md` 中“C2 进行中”的旧流程图。
- Definition of Done 中 MEM-00h 与 closed 状态同步。
- 统一：B5、C1–C3、MEM-00h closed；C4 unstarted。
- 保留历史失败记录。

### 4.2 Qualification 硬边界

统一 capability/provenance/final report 必须表达等价状态：

```json
{
  "feature_status": "experimental",
  "production_qualified": false,
  "c4_passed": false,
  "c5_passed": false
}
```

中性路径和缓存身份不变。C4/C5 前不得写 true，experimental 结果不得进入正式 `DeltaG_bind`。添加测试证明不能伪造或绕过该门。

### 4.3 能力标志

核对：`CHARGE_TRANSFER_HAMILTONIAN_IMPLEMENTED`、`CHARGE_TRANSFER_SOLVENT_LEG_IMPLEMENTED`、`solvent_leg_builder_implemented`、`closes_thermodynamic_cycle`、Stage2 capability、`CHARGE_TRANSFER_VANISHING_HANDOFF_PROTOCOL_VERSION`。

链路存在不等于 production-qualified；真实 solvent leg 若不能运行，对应标志必须保持 false。

### 4.4 Fail-closed

必须拒绝：charged ligand + neutral；charge-transfer + APBS/Rocklin；membrane + co-annihilation；缺 dummy/spec；co-ion identity/charge/handoff/cache 不一致；charging System 二次配置；charged-membrane production 使用 shadow_ibs；C4/C5 未过却 production-qualified。

## 5. 重点审查

1. `resolve_charge_treatment()` 默认推导、双计数和能力字段。
2. complex/solvent dummy 各自构建及身份冻结。
3. charging：ligand `q→0`、co-ion `0→q`、总电荷恒定。
4. baking：只删除目标参数，保留无关 globals/offsets 和完整 `NonbondedForce`；duplicate/cross-force fail closed。
5. handoff 只在正确带电 Stage2 路径触发。
6. handoff version 进入 Stage2 cache/resume gate。
7. MEM-00h normalization 仅用于 C3 evaluation clone。
8. restraint 不重复注入。
9. final report 不提升 experimental 结果。

## 6. 测试

使用仓库既有环境，不在线升级依赖：

```bash
python --version
python -c "import openmm; print(openmm.__version__)"
python -m pytest -q tests/test_bake_global_parameter.py
python -m pytest -q tests/test_charge_transfer_real_endpoints.py
python -m pytest -q tests/test_generate_c3_summary_reports.py
python -m pytest -q tests/test_coalchemical_ion_identity.py
python -m pytest -q tests/test_resume_reuse_contracts.py
python -m pytest -q tests/test_stage_diagnostics_persistence.py
bash tests/run_offline_tests.sh
```

文件名变化时先执行：

```bash
rg --files tests | rg 'charge|coalchemical|resume|stage|c3|bake'
```

要求 0 failed；skip 只能是既有 opt-in。测试总数可变化，不为匹配旧的 `1213 passed` 删除测试。

## 7. C3 证据复核

只跑纯汇总，不重跑矩阵：

```bash
python tools/validation/generate_c3_summary_reports.py --help
```

按 CLI 核验现有 v2 数据：A/B 100/100、C/D 50/50、`n_failed=0`；summary/mem00h report complete/PASS；C2 raw switch 如实为 0.995 nm；evaluation clone cutoff 1.0 nm、switching false。

## 8. 提交拆分

保留已有合理历史；否则建议：

1. `feat: add experimental charge-transfer Hamiltonian and co-ion identity`
2. `feat: add charge-transfer Stage2 baked handoff and cache identity`
3. `test: add endpoint and fail-closed validation contracts`
4. `docs: record C3 closure and experimental qualification boundary`

提交前：

```bash
git diff --check
git diff --cached --stat
git diff --cached
```

禁止盲目 `git add .`/`git add -A`，按文件列表添加。

## 9. 合并策略

- 完整功能分支：测试后 PR/merge，主线再跑定向+完整回归。
- 改动混在主线脏工作树：建 merge-prep 分支，分组提交后 PR。
- 多提交 cherry-pick：core → pipeline/cache → tests → docs；冲突人工解决，不整边覆盖。

## 10. 主线验收

- 中性 Atenolol 旧路径通过。
- charge-transfer 契约通过。
- C3/MEM-00h PASS。
- Stage2 handoff/cache/resume 通过。
- 文档一致。
- 默认配置不自动启用 charged-membrane experimental。
- charged-membrane 报 `production_qualified=false`。
- 无新增大型轨迹/缓存。
- `git diff --check` 通过。
- 剩余未提交文件归属清楚。

报告模板：

```text
目标主线：<branch>
合并方式：merge / PR / cherry-pick
合并提交：<commit list>
定向测试：<passed/failed>
完整回归：<passed/skipped/failed>
C3 汇总：PASS/FAIL
production qualification：false（C4/C5 pending）
未合并文件：<list or none>
下一步：C4 real charged complex/solvent smoke
```

## 11. 停止条件

出现旧路径回归、任一测试失败、能力标志矛盾、双计数可绕过、qualification 可伪造、resume 接受错误 handoff protocol、必须覆盖未知用户改动、C3/MEM-00h 无法 fail-closed 复核时，停止并报告最小复现；不扩大到 C4，不重跑长 MD。

## 12. 合并后的准确状态

```text
B5、C1、C2、C3、MEM-00h：closed
Charged membrane charge-transfer engineering foundation：merged, experimental
C4：unstarted
C5：blocked by C4
Production qualification：false
```

合并成功不等于生产资格通过；下一任务单独执行 C4。
