# 维护与修改代码

[返回项目首页](../README.md) · [目录约定](../PROJECT_LAYOUT.md) ·
[测试说明](../tests/README.md) · [文档导航](README.md)

更新日期：**2026-09-02**

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

> ⚠️ 语法检查通过 ≠ 端到端通过。GPU 相关行为（checkpoint 跨 platform 迁移、
> CUDA 插件、REMD 显存）在 CPU 上一条都验不到。

CI 跑的是哪些门见 [`.github/workflows/cpu-ci.yml`](../.github/workflows/cpu-ci.yml)：
`py_compile` + `ruff check` + 对 CI 自维护文件的 `black --check`，再加离线测试。

## 改完代码该更新哪份文档

按"结论的出处"决定，不要新开平行的"当前状态"文档（[docs/README.md](README.md) 维护规则第 1 条）：

| 改了什么 | 更新哪里 |
|---|---|
| 协议版本号、fail-closed 判据、物理口径 | 改动点的代码注释 + [TODO.md](TODO.md) 的《未关闭的代码缺陷》 |
| 设计合同 / 提案的实施状态 | [design/README.md](design/README.md) 的状态表（**必须同步复核日期**） |
| 发布阻塞项 | [RELEASE_READINESS_2026-08-31.md](RELEASE_READINESS_2026-08-31.md) |
| 稳定用法 | [GETTING_STARTED.md](GETTING_STARTED.md) / [OUTPUTS_AND_RESUME.md](OUTPUTS_AND_RESUME.md) |
| 新的科学结论、数值 | 见 docs/README.md 维护规则第 4 条：必须附来源、单位、符号、协议身份、有效性、是否可引用 |

`README.md`/`README_cn.md`/`README_en.md` 的科学状态日期戳由
`tools/diagnostics/check_doc_staleness.py` 盯着，契约测试是
`tests/test_doc_staleness_contract.py`。

> ⚠️ 那份契约测试里的 `test_snapshot_docs_are_not_stale` 带
> `xfail(strict=True)`。**刷新三份 README 的日期戳时必须同时摘掉这个标记**，
> 否则它会 XPASS 而 `strict=True` 把 XPASS 当失败报出来。这是刻意设计的握手：
> 逼"文档已经刷新"被显式确认一次。

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
2. 一次性诊断 → `tools/diagnostics/`；可复用修复 → `tools/repairs/`；
   画图 → `tools/plots/`；验证 → `tools/validation/`。
3. **旧源码副本、`*_bak`、`*_pre_patch` 一律不进本分支**，留在 `Atenolol-rank11`。
   `docs/archive/` 只放文档，且其中 `removed_*.md` 是防回归凭证
   （`tests/test_att27_dead_code_removed.py` 断言它们存在），**不能删**。
