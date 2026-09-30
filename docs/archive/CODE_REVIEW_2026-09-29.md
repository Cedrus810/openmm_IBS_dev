# 核心代码复查报告(2026-09-29)

> 复查人:ZCode(会话审查)。范围:ABFE 主线核心 + 支撑小模块 + 新增 pocket_water,
> 合计约 11 万行。结论登记在本文,未改动任何生产代码。
> pytest 结果取自远程 CI(权威口径)与本地 `.pytest_cache`(参考),本地未跑测试。

## 0. 方法与覆盖度声明

计划是 15 个子代理分段全文精读全部核心代码,但代理基础设施连续故障
(后台并发上限 1 + captcha 超时连杀 3 个),实际降级为:
**小模块全文精读 + 大文件定向精读高危数值段 + ruff 静态扫描 + 数学推导复核**。

| 文件 | 覆盖 |
|---|---|
| `tools/pocket_water/`(全部)、`stage2_rounds.py`、`stage2_rounds_evidence.py`、`multi_segment_analysis.py`、`lambda_path_versions.py`、`step_guard.py`、`tests/test_pocket_water_relocation_2026_09_28.py` | **全文逐行** |
| `ibs_engine.py`(28,058 行) | 定向精读 ~1.5k 行:Boresch TI、local MBAR 求解、ESS/top1%、去相关、endpoint 拼接、几何检查 |
| `abfe_pipeline.py`(20,462 行) | 定向精读:CPU 回退、`_assert_stage_result_sane`、endpoint 拼接调用点 |
| `runabfe.py` | 未提交 diff(52 行,pocket-water 预检门)+ 调用点上下文;**主体未逐行读** |
| `abfe_core.py` / `abfe_preoptimizer.py` / `outer_lambda_neural_basis.py` / `free_energy_engine.py` / `abfe_diagnostics.py` / `apbs_correction.py` / RBFE 三件 | **本轮未读**(待代理恢复可补,约 8 个串行代理 ~2 小时) |
| 全部核心文件 | ruff(E9/F821/F823/B006/B015/E722/F601 等,`--isolated`) |

`refactor/replay/` 下的影子副本一律未读(非生产)。

---

## 1. pytest 结果

### 1.1 CI(权威口径,2026-09-26 最近一次)

```
2 failed, 2840 passed, 18 skipped, 6 xfailed, 161 warnings  (87.6 s)
```

近 8 次推送 CI 全红,原因都是下面两条 failed + 一组 xfail:

1. **`test_doc_staleness_contract::test_snapshot_docs_are_not_stale`**
   `docs/STATUS.md` 声明日期 2026-09-18,落后 git 活跃日期 9 天(阈值 3 天)。
   **值得修**(改一行声明日期)。

2. **`test_nvcc_discovery_and_compile_failure_2026_09_15.py::test_a_compile_error_raises_instead_of_silently_using_cpu`**
   取了完整 traceback,根因**是测试自身的环境假设,不是生产 bug**:
   CI 的 OpenMM 没编 CUDA platform → `getPlatformByName("CUDA")` 在
   `abfe_pipeline.py:2946` 先抛 `OpenMMException` → 被归为"资源类"走 CPU 回退 →
   回退路径 `abfe_pipeline.py:2989` 的 `openmm.Context` 恰好被 monkeypatch 成抛
   `_Boom` → 逃逸出 `pytest.raises(RuntimeError)`。
   修法 2 行:像同文件 resource 测试那样把 `getPlatformByName` 也 stub 掉。
   **值得修**(CI 一直红的两个原因之一)。

3. **6 个 xfail(已知源码缺口,本次已核实根因链)**
   `combine_ibs_and_independent_endpoint`(ibs_engine.py:14549)返回 dict **没有顶层
   `analysis_status`**(只有 `ibs_segment` 里嵌着);`abfe_pipeline.py:7239` 用它
   **整体替换** stage_result,`runabfe.py:5452` 的缓存路径同样;下游
   `_assert_stage_result_sane`(abfe_pipeline.py:9205-9218)对缺 `analysis_status`
   fail-closed ⟹ **`--analyze-only` 恢复 vanishing 腿这条路今天是断的**。
   注释(ibs_engine.py:14691)明说是 2026-09-15 改造时刻意押后的。
   **值得修(发布前)**——但需要先做"两段 analysis_status 如何合成"的设计决定,
   不是一行 passthrough;修好后 strict-xfail 会变红,届时删标记。

4. 18 个 skip 全是环境类(torch 不进 CI、R1 资源已入 archive 等),符合预期;
   无"importorskip 静默跳过一批却显示 all passed"的复发。

### 1.2 本地 lastfailed(参考,来源存疑)

`.pytest_cache/v/cache/lastfailed`,mtime **2026-09-28 15:36**:
51 条失败 / 31 个文件;3 个 exp012 文件整体收集失败(典型缺依赖的样子)。
环境不明(本机 `python3` 无 openmm),只作参考,**CI 口径为准**。

---

## 2. 发现清单(每条带"是否值得修"判断)

### 2.1 高

| # | 位置 | 类别 | 问题 | 值得修吗 |
|---|---|---|---|---|
| H1 | ibs_engine.py:14549 + abfe_pipeline.py:7239 + runabfe.py:5452 | bug(API) | 即 §1.1 第 3 条:顶层 `analysis_status` 缺失 ⟹ `--analyze-only` 恢复 vanishing 腿必炸 | **值得**——发布前必修;需先定两段状态合成规则 |
| H2 | stage2_rounds_evidence.py:131 | bug(契约) | `except (MultiSegmentSolveError, FloatingPointError, ValueError)` 会把 `MultiSegmentInputError`(ValueError 子类)也吞掉:**输入账本损坏被静默转成 nan → 判"单段依赖" → 走补段**,而 `multi_segment_analysis` 的 docstring 明确"输入错误绝不允许降级"。full solve(106 行)不包 try,只有 LOO 路径静默 | **值得**——一行改动:把 `MultiSegmentInputError` 排除出捕获列表(或单独落盘原因) |

### 2.2 中

| # | 位置 | 类别 | 问题 | 值得修吗 |
|---|---|---|---|---|
| M1 | ibs_engine.py:21531 vs 21540 | bug(溯源) | dict 字面量键 `"n_eff_frame_set"` 定义两次,21540 的常量 `"segments_kept_by_decorrelation"` 无条件覆盖 21531 的条件表达式 ⟹ 无 segment 分析时帧集身份**错标**(实际用的是 `all_frames`)。ruff F601 抓到,全仓唯一一条 | **值得**——删一行;这是"这批数字是哪套帧集算的"的溯源字段,错了误导事后审计 |
| M2 | stage2_rounds.py:296-301 vs 373-377 | API 不一致 | `classify_window` 用 `σ_jk > σ_tol`(1.5)判 PROVISIONAL;`plan_next_round` 用 `σ_jk > max(2σ_MBAR, σ_tol)` 决定补段。σ_jk ∈ (σ_tol, 2σ_MBAR] 的窗口被标 PROVISIONAL 却**没有任何动作**,到 MAX_ROUNDS 也永远是 PROVISIONAL | 可选——可能有意(标签严、补段松),但值得一句注释写明,否则读者会当成漏补 |
| M3 | tools/pocket_water/relocation.py:64-72,101-102 | 数值 | V_R/V_S 用 MC 积分(2M 样本)估计,~1-2% 相对误差**直接进入接受判据** ⟹ 平稳分布有 ~0.02-0.04 kT 级系统偏移;且 V_R、V_S 是两次独立采样,误差不相关 | 可选——原型够用;若要产 ΔG 就改解析 inclusion-exclusion(球数少)或两次用同一组随机点 |
| M4 | abfe_pipeline.py:2987-2990 | bug(鲁棒性) | CPU 回退路径的 `openmm.Context(...)` 不在 try 内,失败裸抛、无上下文;也是 CI nvcc 测试失败的共因 | 可选——顺手包一层 `guarded_context` 即可 |

### 2.3 低 / 可选

| # | 位置 | 类别 | 问题 | 值得修吗 |
|---|---|---|---|---|
| L1 | relocation.py:75-85 | 假设 | `uniform_in_shell` 命中率趋零时 while 不终止(壳体积≈0 或 bounding box 远大于壳) | 可选,加最大迭代数 |
| L2 | relocation.py:261 | 数值 | `np.log(rng.random(n))` 在 `random()` 返回 0.0 时得 -inf → 无条件接受;概率 ~1e-16 | 不值得 |
| L3 | run_sampler.py:187 | 假设 | `final_state.npz` 存 float32(输入是 float64);restart 键名 `pos` 与 `--start` 一致 | 不值得 |
| L4 | runabfe.py:295-322(新 preflight) | 假设 | 旧运行目录缺 `ligand_indices_solvent.json` 时裸 `FileNotFoundError`,无"缓存太老请重建"提示 | 可选,一行友好评级 |
| L5 | ibs_engine.py:24962 `_check_boresch_geometry_safe` | 测试缺口 | 只查 θA/θB 的 sin 下限,二面角 φ 的奇异性(共线)无对应预检 | 可选 |
| L6 | step_guard.py docstring 自述 | 死代码(已知) | `guarded_minimize/set_parameter/deserialize/platform/context` 等**已实现未接线**;11 处裸 `.step()`(outer_lambda 3 处、free_energy_engine 2 处)未接 | 可选——发布前把 outer_lambda 生产路径 3 处接上收益最大(NaN 时有诊断现场) |
| L7 | stage2_rounds.py:267 `first_round_specs` | 假设 | seed_id 每窗从 0 重计(0/1/2 跨窗复用);per-window 证据层不受影响,但若别处假设 seed_id 全局唯一会踩 | 可选,docstring 写明作用域 |
| L8 | 测试缺口 | pocket_water 只有 U0/stage-1 数学测试;**stage-2 延迟接受、run_sampler 的 apo ghost-ligand 路径、kabsch 跟踪**无测试 | 值得补 stage-2 一个小模拟测试 |

---

## 3. 验证过、确认干净的部分

- **pocket_water 的 MC 数学**:完整推导了 stage-1 接受式——ΔU0 的 u 项在 π 比值与
  提议概率比之间严格相消,`(V_R/V_S)·Z_S(x)/Z_R(y)·exp(Δg)` 与代码逐项一致;
  out 方向同样;Z_R(y)/Z_S(y) 的增量更新(cutoff 内外 phi 项)正确;
  stage-2 `min(1, exp(-β[(ΔU)-(ΔU0)]))` 正确。测试的"正向/反向 log ratio 相消 +
  独立重算"是高质量的对账设计。
- **multi_segment_analysis.py**:三条硬约束(sampling_states 重建、塌缩时机、
  输入不降级)都有 enforce;S+K MBAR 的 kT↔kJ 换算、零计数段剔除、规范锚定全对。
- **lambda_path_versions.py**:孤儿版本采纳、指针只前进、event_id 幂等
  (note/detail 分离 + 旧形状兼容)都有测试钉住。
- **step_guard.py**:`finite_state_check` 三项同查(能量/坐标/力)、回退只在热化路径、
  生产路径只换错误信息不改动力学,接入状态表如实。
- **Boresch TI**(`_attachment_ti` / `reweighted_fd_ti`):符号约定、非均匀 λ 网格
  中心差分、kT 换算正确,历史上踩过的符号坑有注释+数值核对。
- **`_solve_single_window_local_mbar` / `_decorrelate_by_worst_target_state` /
  `_ess_from_log_weights` / `_top1pct_weight_by_empirical_cdf`**:
  fail-closed 模式彻底(形状/有限性/`sampled_row!=0` 拒绝/列稳定化/valid mask),
  ESS_GATE v5 的循环论证修复有实测依据;top1% 等权重→0.01 校验点成立。
- **runabfe.py 未提交 diff**(2026-09-28 HMR 质量修复 + 采样前预检门):
  方向正确,质量取自 System 而非元素表是正解;门的位置(缓存建成后、采样前)对,
  第 8 节旧门保留防采样途中改 System。
- **ruff 全核心文件扫描**:除 M1 外**零**正确性问题
  (无未定义名、无可变默认参、无裸 except-pass)。
- **测试套件健康度**:226 文件 / ~2468 个测试(`def test_` 计数,parametrize 展开后
  CI 口径 2,866);全部核心模块(含 pocket_water 4 个测试文件引用)都有测试触及,
  无零覆盖模块;pytest.ini / run_offline_tests.sh / CI workflow 三方一致
  (`-m "not needs_gpu"` 在两处都对齐)。

---

## 4. 遗留(未覆盖,可下次补)

`abfe_core` / `abfe_preoptimizer` / `runabfe` 主体 / `outer_lambda_neural_basis` /
`free_energy_engine` / `abfe_diagnostics` / `apbs_correction` / RBFE 三件
还没逐行读——因为代理管线故障。需要时按同样格式补完
(约 8 个串行代理,~2 小时)。

## 5. 建议的最小修复包(按性价比排序)

1. `docs/STATUS.md` 声明日期刷新(CI 红 → 消一条)。
2. nvcc 测试 stub 掉 `getPlatformByName`(CI 红 → 消一条,CI 转绿)。
3. H2:`stage2_rounds_evidence.py:131` 捕获列表剔除 `MultiSegmentInputError`。
4. M1:删 ibs_engine.py:21540 的重复键。
5. H1:开一条设计决定("合并段的 analysis_status 如何由两段硬不变量合成"),
   修好后删 6 个 strict-xfail 标记。
