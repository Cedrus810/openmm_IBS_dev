# 维护与修改代码

[返回项目首页](../README.md) · [目录约定](../PROJECT_LAYOUT.md) ·
[测试说明](../tests/README.md) · [文档导航](README.md)

更新日期：**2026-09-12**

## 改完代码先跑什么

三档，从快到慢。前两档不需要 OpenMM。

**1. 语法（秒级）**

```bash
python -m py_compile abfe_core.py abfe_pipeline.py abfe_preoptimizer.py ibs_engine.py runabfe.py
```

**2. 离线全量（CPU，唯一的"最低验证"标准）**

```bash
./tests/run_offline_tests.sh                                  # 全部
./tests/run_offline_tests.sh tests/test_core_physics_numerics.py   # 单文件
```

**3. 需要 OpenMM 的自检**

```bash
python runabfe.py self-test
```

`self-test`（`runabfe.py:3275` `run_self_tests`）会在缺依赖时逐项 `SKIP` 而不是
报错——**看到 PASS 之前先确认没有一片 SKIP**，否则它什么也没验证。

**4. 发布前：clone 下来 import 得动吗（秒级，CPU）**

```bash
pytest tests/test_fresh_clone_imports.py
```

**这道门就是发布清单本身。** 它防的不是逻辑错，是**漏把新文件加进 git**——
新模块在工作树里，本机跑得好好的，clone 下来当场 `ModuleNotFoundError`。
分两层，因为后果不同：

| 层 | 是什么 | 行为 |
|---|---|---|
| 硬门 | 已跟踪文件**模块级** import 未跟踪模块 | 红。clone 当场死 |
| 软门 | 已跟踪文件**函数内惰性** import 未跟踪模块 | 只登记；出现**新的**未登记项才红 |

软门的登记表就在那个测试里，每项写了为什么（哪些必须随发、哪些是默认关闭的
可选功能）。加了新的惰性依赖，要么把它随发布带上，要么进登记表并写清理由。

> 实测过一次：`step_guard.py` 未跟踪，而 `ibs_engine.py:34` 是模块级
> `from step_guard import guarded_step` ⟹ `ibs_engine` / `abfe_pipeline` /
> `abfe_preoptimizer` / `runabfe` 四个入口在 fake clone 里全部 `ModuleNotFoundError`；
> 只补这一个文件，四个立刻全绿。

> ⚠️ 语法检查通过 ≠ 端到端通过。GPU 相关行为（checkpoint 跨 platform 迁移、
> CUDA 插件、REMD 显存）在 CPU 上一条都验不到。

CI 跑的是哪些门见 [`.github/workflows/cpu-ci.yml`](../.github/workflows/cpu-ci.yml)：
`py_compile` + `ruff check` + 对 CI 自维护文件的 `black --check`，再加离线测试。

## 改完代码该更新哪份文档

按"结论的出处"决定，不要新开平行的"当前状态"文档（[docs/README.md](README.md) 维护规则第 1 条）：

| 改了什么 | 更新哪里 |
|---|---|
| **任何代码或协议改动** | [CHANGELOG.md](CHANGELOG.md) 加一行（规则见该文件末尾） |
| 协议版本号、fail-closed 判据、物理口径 | 改动点的代码注释 + [TODO.md](TODO.md) 的《未关闭的代码缺陷》 |
| 设计合同 / 提案的实施状态 | [design/README.md](design/README.md) 的状态表（**必须同步复核日期**） |
| 发布阻塞项 | [RELEASE_READINESS_2026-08-31.md](RELEASE_READINESS_2026-08-31.md) |
| 稳定用法 | [GETTING_STARTED.md](GETTING_STARTED.md) / [OUTPUTS_AND_RESUME.md](OUTPUTS_AND_RESUME.md) |
| 新的科学结论、数值 | [STATUS.md](STATUS.md)：必须附来源、单位、符号、协议身份、有效性、是否可引用 |

[STATUS.md](STATUS.md) 是本仓库**唯一**声明当前科学结论的文档——新数字、协议版本
和有效性判据都只改那一份，三份 README 只留指向它的一行，别把表复制回去。

它的整理日期戳由 `tools/diagnostics/check_doc_staleness.py` 盯着，协议版本表由
同一份契约测试 `tests/test_doc_staleness_contract.py` 对着源码常量钉住。
**改了任何 `*_PROTOCOL_VERSION` 常量，同一次改动里要把 STATUS.md 的表改掉。**

## 版本控制

本仓库是一个正常可用的 git 库（`master` 分支，起点 commit `eaf1c7e`
"Initial cleaned ABFE IBS version"，2026-08-31 迁移建立）。

> ⚠️ 已知环境问题：仓库在 NFS 上，文件属主 uid 与当前用户不一致，
> **提交可能因权限失败**。这不是仓库损坏。

2026-08-31 之前的改动**没有 git 历史可查**——那段时间线只能靠文件 mtime 和
`Atenolol-rank11` 工作区里的 DEC 记录追溯（登记在
[HISTORY_LOG.md](HISTORY_LOG.md)）。

## 放文件的规矩

完整规则见 [PROJECT_LAYOUT.md](../PROJECT_LAYOUT.md)《维护规则》。最常踩的三条：

1. 新的自动化测试**只**放 `tests/`。
2. 一次性诊断 → `tools/diagnostics/`；验证 → `tools/validation/`。
   **事故结案 = 它的诊断脚本也结案**：结论写进 `docs/`，脚本移到
   `Atenolol-rank11/archive/from_mainline_<日期>/`，别留在主线烂掉。
3. **新建的模块当场 `git add`。** 不是洁癖——`tests/test_fresh_clone_imports.py`
   的硬门会红，而且漏了它 clone 下来直接 import 不动（见上面第 4 档）。
4. **旧源码副本、`*_bak`、`*_pre_patch` 一律不进本分支**，留在 `Atenolol-rank11`。
   `docs/archive/` 只放文档，且其中 `removed_*.md` 是防回归凭证
   （`tests/test_att27_dead_code_removed.py` 断言它们存在），**不能删**。
