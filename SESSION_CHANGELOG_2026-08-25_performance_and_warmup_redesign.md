# 会话总结：性能优化 + 窗口预热状态机重构（2026-08-25）

本文档记录这次会话里对代码仓库做的**全部实际改动**（已落盘、已过 `ast.parse`/`json.load` 静态检查），以及**已设计但未落地**的部分。本会话全程没有 GPU/OpenMM 访问，所有改动只做到静态语法验证，**真机验证是下一步的必做项**，见文末"待验证清单"。

同一时间，另一个 Claude 会话（`atenolol-rank11-c6`）在同一台机器上并发修改 `ibs_engine.py`（`{prefix}_s_residual` 残差力冷启动 bug），两边做过多轮显式协调，细节见文末"并发协调记录"。

---

## 一、已落地的性能优化（P0/P1，行为保持不变，纯优化）

### P0-1：生产控制面 GPU 同步瀑布（`ibs_engine.py`）

背景：生产主循环每 500 步一次的控制面迭代里，`collect_energies()` 与 guard（灾难检测）各自独立发起 `getState()` 查询，K 个 probe 态逐个单独查询，且完全没有计时手段区分耗时去向。

- 新增模块级 `_timed(bucket, key)` contextmanager（`ibs_engine.py:468`），默认常开。
- 生产主循环 + 余数补齐块 + warmup/learning 循环都接入了 `integration_s`/`guard_s`/`cv_probe_s`/`ledger_io_s`/`weight_update_s` 计时，写进 `production_conv_path`（`convergence.json`）的新增诊断字段 `"loop_timing_s"`；warmup 循环结束打印一行汇总。
- `collect_energies()`（`ibs_engine.py:6464`）、`_collect_interaction_energies()`（`6335`）、`_lj_tail_correction_kj_mol()`（`6366`）新增可选关键字参数 `reuse_positions`/`reuse_box_vectors`（默认 `None`，不传时行为逐位不变）：
  - guard 在 `do_force_check=True` 的 update 上已经做过一次 `getState(getPositions=True, ...)`，现在把这份数据传给 `collect_energies()`，省掉一次完全重复的 GPU 同步。
  - `_lj_tail_correction_kj_mol()` 原来无条件独立查一次 box vectors，现在复用 `_collect_interaction_energies()` 顺带返回的那一份——这条对**所有**调用点（生产/余数/warmup/冷启动）无条件生效，不需要 guard 命中。
- 新增共享方法 `_production_disaster_rollback()`（`ibs_engine.py:9771`），把生产主循环和余数补齐块里原来重复的"回退坐标→重设速度→局部最小化→步长减半→截断 history"代码去重，choreography 逐字节保留。
- **修复一个已确认的重复落盘 bug**：生产主循环每 100 update 的周期性 checkpoint 块里，`_enqueue_window_snapshot()` 和紧跟着的 5 个 `_atomic_save_npy()` 调用写的是**完全相同的文件路径**（`production_energies_path` 等变量本来就等于 `dual_window_*.npy`），逐字节重复写了两遍；余数补齐块也有同构的重复。已删除冗余写入，磁盘最终内容不变，只是少写一半。

### P0-2：preoptimizer 冷启动计时（`abfe_preoptimizer.py`）

`_sample_scalar_metric()`（`2494` 行）内部给"积分"和"有限差分能量读取"分别加了 `_timed()` 计时（复用 `ibs_engine._timed`），每个 pilot λ 点打印一行耗时拆分。**不改** `sample_interval`/`n_steps_per_state` 等默认值（会影响 λ 路径优化结果）。

### P1-1：离线 MBAR worker pool（`ibs_engine.py`）

- `_build_platform_properties()`（`2527`）新增可选 `cpu_threads` 参数，仅在 CPU 平台生效（`props["Threads"]`），默认 `None` 不影响其余全部调用点（CUDA/OPENCL 主 Context 等）。
- 新增 `_mbar_worker_init()`（`2350`）+ 模块级 `_MBAR_WORKER_CTX`（`2347`）：`multiprocessing.Pool(initializer=...)` 让每个 worker 进程只反序列化一次 System、建一次 Context，后续同一 worker 拉到的每个 chunk 复用它，不再像 `_compute_u_kn_chunk()`（`2366`）原来那样每个 chunk 都重新反序列化+重建。
- worker 数量算出后按 `physical_cores // n_workers` 给每个 CPU Context 分配线程预算，避免进程数×线程数过度并行。
- 串行路径（`n_workers==1`、多进程失败后的单进程回退）行为完全不变——`_MBAR_WORKER_CTX` 在主进程里恒为 `None`，自动走原来的独立建 Context 分支。

### P1-2：checkpoint/日志 I/O

- 上面 P0-1 已经描述的重复落盘 bug 修复也算这条。
- `abfe_pipeline.py` 的 `_StdoutTeeToFile`（`378`）原来每写一行日志就 `open()...write()...close()` 一次，现在常驻打开一个文件句柄、每行 `flush()`，`atexit` 注册收尾关闭。

### P1-3：`abfe_pipeline.py`/`runabfe.py`/`abfe_config.json` 死配置清理

- `_run_dual_lambda_stage()`（`3278`）签名里删除了从未被消费的 `n_workers`/`parallel`/`device_indices` 三个参数。
- `run_full_pipeline()` 新增运行时 warning：`n_workers` 非 `None` 时打印忽略提示（不再无声吃掉）。
- CLI `--n-workers` 帮助文本、`abfe_config.json` 里的 `n_workers` 字段都加了"当前无效"的说明。
- `parallel_stages` 的拒绝逻辑保持不变（本来就是正确行为）。

---

## 二、窗口预热状态机重构 — 已落地部分（Stage 0 + Stage 1a）

完整设计（含未落地的 Stage 1b-6）见计划文档 `/home/kasuga/.claude/plans/p0-sorted-fiddle.md`（本仓库之外，Claude Code 会话专用目录）。这里只记录**这次会话真正改了什么**。

### Stage 0：pilot 种子信任度诊断（新能力，尚未接入任何决策）

- 新增纯函数 `pilot_ti_seed_trust_diagnostics()`（`abfe_preoptimizer.py:1483`）：判断一个 pilot TI 种子（`estimate_f_k_from_pilot_ti()`，`1387`）对某个具体窗口**精度**够不够（SEM、粗略 trapezoidal 误差传播、是否需要外推），不判断**准确性**（pilot 探针系统是否真的代表这个窗口的真实环境）。纯 Python/numpy，不需要 OpenMM，可离线单测。
- `abfe_preoptimizer.py` 的 `_sample_scalar_metric()` 早就算出 `std_dU_dlambda_kJ_mol`/`n_derivative_samples`，只是没被提取——现在从 pilot 网格一路补到 `IBSWindowManagerDualLambda.__init__`（新增两个可选参数 `pilot_std_dU_dlambda`/`pilot_n_dU_dlambda_samples`，默认 `None`），途经 `abfe_pipeline.py` 的两处 pilot 数据提取点（cache 命中 + 新鲜生成）和两处 `_run_dual_lambda_stage(...)` 调用点（"vanishing"/"vanishing_rescue"）。
- 新增 8 个单元测试，`tests/test_core_physics_numerics.py`（紧跟在原有 `estimate_f_k_from_pilot_ti` 测试块之后）：可信案例、7 种拒绝案例（数据缺失/精度字段缺失/形状不对/含非有限值/零样本/需要外推/SEM超阈值/传播不确定度超阈值）、乱序不变性。

**这一步本身不改变任何运行时行为**——新函数没有被任何调用点消费，`IBSWindowManagerDualLambda` 的两个新参数默认 `None`。

### Stage 1a：`max_bias_warmup_steps` → `max_bias_learning_steps` 改名

纯改名，语义/默认值不变——这个参数控制的一直是"learning 阶段的步数预算"，跟真正的"爬坡"完全是两回事，旧名字容易误导。

- `run_all_windows()` 形参改名（`ibs_engine.py:9849`），连同函数体内所有内部引用（docstring、budget 算式、诊断字段 `max_bias_learning_steps_safety_cap`、注释）。
- **外部契约一律不改**：`abfe_config.json`/`runabfe.py` 的 `config.get("max_bias_warmup_steps", ...)`、`exp029_protocol.py` 的协议 schema 字段名、`_run_dual_lambda_stage()` 自己的 `**kwargs` 查找键名，全部保持 `"max_bias_warmup_steps"` 不变——只有真正绑定到 `run_all_windows()` 形参的那几处调用点跟着改。

#### 期间出的一次真实回归，已修复

第一轮改名时只 grep 了仓库顶层 `*.py`，漏掉了 `scripts/` 子目录下两个**直接调用** `run_all_windows()` 的脚本：

- `scripts/exp029_window_state_machine.py:316`：`max_bias_warmup_steps=int(budget["max_bias_warmup_steps"])` → 改成 `max_bias_learning_steps=...`（`budget` 字典键名不变）。
- `scripts/exp030_window_state_machine.py:349`：同样的 `common_run` 字典（被 `**common_run` 直接展开进两处 `run_all_windows(...)` 调用）里的键改名。

这个疏漏导致并发会话跑 EXP-030 手动验证矩阵时 `candidate/window_5` 直接 `TypeError: got an unexpected keyword argument 'max_bias_warmup_steps'`。已用全仓库递归 `grep -r --include="*.py"` 重新核对修复，确认全仓库 6 处真实 `run_all_windows(` 调用（`abfe_pipeline.py` 两处、`exp029`/`exp030` 脚本三处、`tests/test_non_mutating_policy.py` 一处）全部一致。`tools/diagnostics/diagnose_solvent_box_scan.py` 里的同名字段核对过是走 `run_full_pipeline(**kwargs)` 的转发，不需要改。

---

## 三、已设计但未落地的部分（全部记在计划文档里，需要真机验证才能落地）

以下内容**只有设计、没有代码改动**，故意留在计划文档而不是这里实现，原因见下一节"并发协调记录"：

- **A. 合并测试步进 + 时间步长爬坡**：新方法 `_dt_and_stability_ramp()`，5 个 dt 档位保留、按 `dt_ramp_steps`（默认 8000）比例分配，消除 0.5fs/5000步 的逐字节重复，查询频率从固定 50 步一次改成按档位步数缩放。
- **B. 偏置爬坡瘦身 + 三级健康检查**：`bias_ramp_steps` 替代写死的 8000 步，健康检查从"仅能量有限"扩成三条（能量+力有限、力峰值倍数、相邻能量跳变），三级降级（重试→退回老爬坡→硬报错），建议先 log-only 上线再切生效。
- **C. pilot-first 三分支状态机**：`resumed_frozen_f_k`/`pilot_first_frozen_f_k`/`else学习` 三分支，pilot 种子必须同时满足精度诊断（Stage 0 的 `pilot_ti_seed_trust_diagnostics`）**和**独立自举 TI 交叉验证一致，且 `residual_enabled=True` 的窗口首次上线一律排除，默认 `pilot_first_enabled=False`。
- **E. learning 阶段冻结判据重构**（2026-08-25 用户追加设计）：修复两个已核实的真实 bug——`f_stability_threshold_kJ_mol=0.05` 是从未参与判断的死代码；现有判据只看"应用后（已被 hard cap 裁剪）的步长"，不看原始残差 `residual_severity` 也不看 `hard_pairwise_cap_applied`，导致限幅/阻尼把步长压小时会被误判为"已收敛"。新判据要求三者同时满足：应用步长 ≤ 0.5kT（原 1.0kT）、原始残差 ≤ `IBS_UPDATE_ADAPTIVE_RESIDUAL_LOW`（复用仓库已有边界，不新造数字）、本轮未被硬 cap 触发；`min_bias_updates` 从 12 降到 4；`max_bias_updates=50`/frozen local-MBAR 验证完全不动。核心原则：**学习阶段允许激进早停，验证阶段绝不放松**。
- **D. 参数改名/新增汇总**：`dt_ramp_steps`/`bias_ramp_steps`/`pilot_first_enabled`/`pilot_seed_max_sem_kJ_mol` 等一批新参数的完整清单。

预期效果（未改变最终验收标准，只改变达到验收所需的步数）：普通冷启动首次冻结 ~120k→~40k 步；pilot 可信时直接验证，省掉整段 ~120k 步 learning；固定前置爬坡 33,900→约 5k-10k 步。

**独立风险复核的硬性结论**（已写入计划文档，任何后续实现都必须遵守）：
1. "pilot 存在"不足以跳过学习——必须叠加独立自举 TI 交叉验证。
2. `residual_enabled=True` 的窗口和近解离端点窗口首次上线必须排除在 pilot-first 之外。
3. 不能复用 `calibrated_pending_validation`/`calibrated_validation_failed` 这两个 `bias_status` 枚举值（它们代表"已经过 fixed-H overlap 探针证明"，pilot-first 候选没有这个背书）——需要新增专门的枚举值。
4. 语义变化必须 bump `IBS_WARMUP_UPDATE_PROTOCOL_VERSION`（当前 9），防止旧协议下的 in-flight 学习缓存被新逻辑误读。

---

## 四、并发协调记录

另一个 Claude 会话本周同时在改 `ibs_engine.py`（`{prefix}_s_residual` 全局参数，修复 EM 阶段残差力冷启动导致的密度雪崩 bug），维护的不变量是"任何进入真实动力学步进的路径，在步进前都必须摸到一次 `s_residual=1.0` 重设"。双方逐段核对确认互不重叠（对方改动集中在 `run_all_windows` 里"最小化"前后的设置点，约 10240-10262/10610-10625 行；本次会话的改动在这之后的 warmup/学习循环、生产主循环）。

**截至本文档写作时，对方的修复尚未稳定收尾**：6 窗口×2 臂验证矩阵显示 `s_residual` fix 让 EM 阶段崩溃概率降低、撞线更晚，但 `window_0`/`window_4` 仍会崩溃，对方判断"部分改善不是根治"，可能需要给 EM 阶段单独建一个不挂残差力 Force 的临时 System，方案还在讨论。**上面第三节列出的 A/B/C/E 四块设计，本次会话故意只写文档、不落地代码**，双方约定等对方主动确认"可以动了"再实施，避免在同一段还在变化的状态机代码上产生冲突。

---

## 五、待验证清单（下一步，需要真机 `openmm_dev` 环境/GPU 节点）

本仓库没有 git，所以"改前/改后对比"不能靠 `git diff`/`git stash`——下面每一项都设计成**在当前代码里就能自洽验证**（两条路径仍同时存在，或者用确定性/一致性检查代替直接 diff），不依赖旧版本文件。

### 1. Stage 0 单元测试（纯 CPU，几秒钟）

```bash
conda activate openmm_dev
cd /home/ruigengji/ABFE_IBS/Atenolol-rank11
pytest tests/test_core_physics_numerics.py -k "pilot_trust or pilot_ti_seed" -v
```
预期：新增的 8 个 `pilot_trust_*` 测试 + 原有的 `pilot_ti_seed_*` 测试全部 pass。本会话只手工逐步核对过算法（这个环境没有 numpy/pytest），第一次真正执行。

### 2. Stage 1a 改名回归

```bash
# 确认没有遗漏的直接绑定点（应该只剩 exp029_protocol.py 的 schema 字段名和
# abfe_config.json 的 key——这两个是有意保留的外部契约，不是漏改）
grep -rn "max_bias_warmup_steps" --include="*.py" . \
  | grep -v 'budget\[\|config\.get(\|kwargs\.get(\|"max_bias_warmup_steps"\|注释\|Stage 1a\|窗口预热\|向后兼容\|外部契约\|协议 schema'

python3 -c "import abfe_pipeline, ibs_engine, runabfe, abfe_preoptimizer" && echo IMPORT_OK
pytest tests/test_non_mutating_policy.py -v
```

### 3. P0-1：`collect_energies()` 复用 guard 状态的数值等价（两条路径都在当前代码里，不需要旧版本）

在一个真实窗口的某一帧上，同时用两种方式调用 `collect_energies()` 并 diff：
```python
# 老路径（不传 reuse_*，走独立 getState）
e_old = sampler.collect_energies()
# 新路径：先手动做一次 guard 会做的 getState，再喂给 collect_energies
state = sim.context.getState(getPositions=True)
e_new = sampler.collect_energies(
    reuse_positions=state.getPositions(asNumpy=True),
    reuse_box_vectors=state.getPeriodicBoxVectors(),
)
assert np.allclose(e_old, e_new, atol=1e-4)  # kJ/mol
```
连续两次调用之间**不能有 `sim.step()`**（否则物理状态已经变了，diff 没有意义）。同时人为触发一次灾难分支（比如临时把 `IBS_FORCE_CHECK_INTERVAL` 设成 1、或者手动注入一个非有限能量）确认回滚行为（坐标回退/reheat/步长减半/history 截断）跟改动前描述的逐字节一致。

### 4. P1-1：MBAR worker pool 数值等价（同样两条路径都在当前代码里）

对同一批已有的轨迹帧（比如 `memtest/output_membrane_100ns/vanishing/` 或任意一次真实产出的 `dual_window_*_energies.npy` 对应的原始轨迹），分别用 `n_workers=1`（老式串行、每 chunk 独立建 Context）和强制 `n_workers>1`（新的 pool-with-initializer 路径）重算一次 `u_kn`，两者应逐位一致（或仅有可忽略的浮点误差）。这是纯调度重构，没有改任何能量计算路径，理论上必须完全一致。

### 5. `warmup_only=True` 冒烟 + 落盘内容核对

```bash
# 挑一个已知能跑通的窗口，跑 warmup_only，确认不崩溃
python3 scripts/exp030_window_state_machine.py --mode smoke --arm baseline --window-index 0 ...  # 按实际参数调整
```
检查产出的 `dual_window_*_convergence.json` 里：
- 出现新字段 `"loop_timing_s"`，且 `integration_s`/`guard_s`/`cv_probe_s`/`ledger_io_s` 都是正数。
- 如果手头还留着这次改动之前跑过的同一个窗口的 `dual_window_*_energies.npy`/`_bias.npy`/`_base.npy`（比如 `memtest/` 或 `output_lrc_fix*/` 下的旧产出），用 `sha256sum` 对比新跑一次（同 seed、同配置）产出的文件——P0-1/P1-2 的改动理论上不改变任何采样轨迹或落盘内容，只是少查/少写，`sha256` 应该完全相同（前提是两次跑的随机种子、配置、代码里跟采样物理相关的部分都没变）。

### 6. A/B/C/E（爬坡瘦身 + pilot-first + 冻结判据重构）

全部依赖并发会话的 `s_residual`/EM 崩溃问题先真正收尾——**这四块目前完全没有代码，只有计划文档里的设计**，不在这次的"待验证"范围内，等对方确认"可以动了"、这边把代码写出来之后才需要验证。

---

## 六、后续会话（同日）：Candidate-first, Validate-or-Learn 状态机重写（已落地，取代第三节 C/部分 E）

**重要更正**：第四节记的"对方 `s_residual` 修复尚未合入 `ibs_engine.py`、只在 EXP-030 脚本里"这条记忆已过期。用户确认：`ibs_engine.py` 里 EM 阶段 `s_residual=0`（约 10345 行）和真实动力学前恢复 `s_residual=1.0`（约 10709 行）**已经是** v30 修复的一部分，是当前生产代码，不是外挂脚本。因此第三节末尾"等对方确认可以动了"这条阻塞条件，对**状态机本身**（不是对 EM/ramp 物理代码）已经解除。

### 背景与用户新设计

用户在新一次对话里给出了一版更彻底的重设计（"Candidate-first, Validate-or-Learn"），把预热/学习/验证收敛成 5 个概念状态——ACTIVATE / SEED / VALIDATE / LEARN / PRODUCTION / FAILED，核心变化：

- **VALIDATE（不再是批数计数器）才是唯一生产入口证明**：候选 f_k 一得到就冻结，直接拿真实窗口 Hamiltonian 的 local-MBAR loose gate（阈值不变，仍是既有的 10 kJ/mol）验证，不再强制先跑满 `min_bias_updates=12` 批全历史 TMBAR 学习。
- **LEARN 只管占据、不再解全历史 TMBAR**：删掉 `update_weights()` 里的 trusted/untrusted/self_consistent/legacy_quality 选择器，改成只用已有的 `_bounded_log_occupancy_update()`（bounded log-occupancy 反馈），raw residual 一旦 ≤ `IBS_UPDATE_ADAPTIVE_RESIDUAL_LOW` 立即冻结去验证。
- **取消 best-effort 默认放行**：预算耗尽不再默认接受未验证的 f_k 进生产。
- **不新增 pending-validation 的 resume 状态**：验证中途被杀就整段重做，不建 resume-mid-validation 状态机。

经用户逐条确认（AskUserQuestion 四问），这版设计**全面取代**第三节 C 部分（pilot-first 三分支 + 强制 bootstrap 交叉验证）和 E 部分的具体数值方案，但**明确保留**第三节结论 3/4（不复用 `calibrated_pending_validation`/`calibrated_validation_failed`、必须 bump `IBS_WARMUP_UPDATE_PROTOCOL_VERSION`）——只是新枚举值定为更简单的 `"failed"`（而不是新造一个 `pilot_seeded_*` 系列名字）。第三节 A/B（爬坡合并、偏置爬坡三级健康检查）**本次仍不动**，用户明确要求"第一版暂时不动现有 ramp，避免把性能改动和验收改动混在一起"。

### 已落地的具体改动（`ibs_engine.py`，均已过 `py_compile` + 真机 `openmm_dev` 环境 pytest 验证）

- **LEARN**（`mode=="learning"`）：不再调用 `sampler.update_weights()`；直接调用 `sampler.evaluate_frozen_batch_probability()` 取占据、`sampler._bounded_log_occupancy_update()` 更新 f_k，冻结判据从"应用步长稳定+连续通过+满 12 批"改成单一条件 `residual_severity <= IBS_UPDATE_ADAPTIVE_RESIDUAL_LOW`。`update_weights()`/`_solve_tmbar_and_recenter()`/`IBS_TMBAR_TRUST_*` 常量**保留但不再从循环里调用**（留作离线对照/未来复用，不删除）。
- **VALIDATE**：
  - 新增读诊断早退：攒满 5 批（约 200 帧）局部 MBAR 门数据前，用跟 `_diagnose_local_mbar_situation` 完全同源的 `_softmax_occupancy_per_state()`（新增模块级函数，两处共用，避免出现两份可能漂移的占据公式）判断占据是否已现塌陷迹象，是则提前退回 LEARN——纯路由决策，不参与 `gate_ok`。
  - 门失败分支从"无条件退回 learning"改成三路由：占据塌陷→LEARN；占据尚可但 gap 未过且本轮没用过重试机会→用这次 local-MBAR 结果做一次阻尼(`_damped_tmbar_absolute_update`)+pairwise-capped(新提取的 `IBSSampler._apply_pairwise_cap` 静态方法，从 `update_weights()` 原来内联的硬 cap 代码块提取，行为逐字节保持)修正，直接回 `freeze_burn_in` 重新验证（跳过 LEARN，每个冻结周期只给一次）；第二次连续 gap 失败或 MBAR 不可解→LEARN。`gate_ok` 本身的定义完全不变。
- **best-effort**：`allow_best_effort_warmup` 默认值 `True`→`False`（`ibs_engine.py:10000`）；回退代码路径本身保留，只有显式传 `True` 的调用方（smoke/debug）才会走到。生产两处 `run_all_windows(...)` 调用点（`abfe_pipeline.py`）本来就没显式传这个参数，默认翻转直接生效。
- **`bias_status`**：往后只写 `"unconverged"`/`"converged"`/`"failed"` 三值；`"calibrated_pending_validation"`/`"calibrated_validation_failed"` 只保留在 `resumed_calibration_pending` 这个**只对老缓存生效**的 legacy 分支里（本会话按用户要求完全不碰 `ibs_engine.py:10586-10689` 这段 resume 路由分支的结构），新代码永不再写这两个旧值。
- **唯一在 10586-10689 区间内做的改动**：终态硬停止检查从 `sampler.bias_status == "calibrated_validation_failed"` 扩成 `sampler.bias_status in ("calibrated_validation_failed", "failed")`——否则新协议下真正终态失败的窗口 resume 时会被误判成普通未收敛热启动。
- **`IBS_WARMUP_UPDATE_PROTOCOL_VERSION`**：9 → 10（`ibs_engine.py:5966`），语义变化足以让旧协议下 in-flight 的 learning 缓存失配重来。`IBS_BIAS_PROTOCOL_VERSION`（物理/Context 层，仍是 30）不受影响。
- **新增纯附加持久化字段**（`save_ibs_state`/`load_ibs_state`）：`seed_source`（"resume"/"pilot"/"bootstrap"/"learned"，纯元数据，不参与任何分支判断）、`validation_attempts`（只数真正跑完一次 local-MBAR 门评估的次数）、`last_failure_reason`。`learning_updates` 只是保存时 `t`（`len(f_history)`）的只读别名，故意不做成独立读回的字段，避免出现两个可能漂移的"学习次数"。缺失时全部安全回退默认值，不新增任何 fail-closed 规则。

### 期间发现并修复的一个真实 bug

`load_ibs_state()` 原来的 `bias_status` 恢复逻辑只认 `"calibrated_pending_validation"`/`"calibrated_validation_failed"` 两个旧值，其余（含新的 `"failed"`）一律落进 `else` 分支被静默改写成 `"converged"`/`"unconverged"`。这意味着如果不修，`ibs_engine.py:10586` 那个刚加的 `"failed"` 识别永远碰不到——一份已经终态失败的窗口 resume 时会被读成 `"unconverged"`，当成普通未收敛重新学习。已在 `load_ibs_state()` 里补一个 `elif cached_status == "failed":` 分支原样保留该值。是写单元测试（"failed" 状态 resume 后仍是 "failed"）时发现的，不是真机复现。

### 测试结果（真机 `openmm_dev` conda 环境，非 GPU，纯 CPU pytest）

- 更新了 `tests/test_audit_protocol_regressions.py` 里 4 条对旧 LEARN 机制做字符串断言的既有测试（原来测 `if f_updated is None:`/`min_bias_updates`/`consecutive_pass_count`/协议版本"9"等旧字面量，现改为断言新机制字面量+`assertNotIn` 旧字面量）。
- `tests/test_warmup_overlap_protocol.py` 新增 4 个测试：`"failed"` 状态 resume 后原样保留、新增元数据字段的保存/恢复往返、老缓存缺字段时的安全默认值、`_apply_pairwise_cap` 的封顶/保方向/不过度封顶的数值契约。
- 四个直接相关文件合计 182/182 通过。全仓库 `pytest tests/`（跳过两个跟本次改动无关、环境路径问题导致收集失败的既有文件 `test_exp017_overlap_first.py`/`test_exp029_harness.py`）：1311 passed / 12 failed / 3 skipped / 4 xfailed——12 个失败逐条核对过，全部是跟本次改动完全无关的既有失败（`_production_history_lengths` 缺属性、`charge_treatment_qualification_payload` 措辞、`outer_lambda_neural_basis` 字符串门、pymbar 4.0.3 vs 4.2.0 环境版本不符、`IBSBiasForce.__init__` 已有 `residual_basis_force` 等 EXP-025/026/027 遗留参数），改动前后这 12 条失败数量/内容完全一致。

### 未做 / 待做

- 第三节 A/B（合并测试步进+dt 爬坡、偏置爬坡三级健康检查）本次仍未落地，按用户指示第一版不动。
- 真机 GPU `warmup_only=True` 冒烟对比（已知好窗口 + 已知边缘窗口，对比新旧路径最终是否都落在同一个 `bias_status="converged"`/同一个 local-MBAR gate 结论）——本会话无 GPU 访问，是下一步的必做验证，思路同第五节"5. `warmup_only=True` 冒烟"那条。
- 详细的分函数/分行实施计划见 Claude Code 计划文档 `/home/kasuga/.claude/plans/candidate-first-validate-or-learn-conte-rustling-lagoon.md`（仓库之外，会话专用目录，格式同上面提过的 `p0-sorted-fiddle.md`）。

---

## 七、后续会话（2026-08-26）：λ 路径规划 pilot shadow early-stop 插桩（Phase A，已落地）

### 背景

用户提出 Stage2 vanishing 的 λ 路径规划（`abfe_preoptimizer.py` pilot 探针）"有点 high cost"。排查过程中先纠正了一个数字：第一节 P0-2 加计时埋点时用的口径是函数默认值/旧文档里的 `n_steps_per_state`，**当前生产配置实际是 `pilot_n_steps_per_state=30000`**（`abfe_config.json`），不是 10000——用 repeat01/repeat02 两次真实完整跑完的 `preopt_dual_vanishing.json` 落盘 `protocol_key.payload.pilot_n_steps_per_state` 交叉核实过。据此重算单腿最坏情况步数：基础 17 点 17×(500+30000)=518,500，加密最多 2 轮×4 点 8×(500+30000)=244,000，初始平衡 5,000，**单腿 worst case ≈ 767,500 步，双腿 ≈ 1.5M 步**，全部发生在生产窗口开始之前、串行、不可重叠。

`30000` 这个值不是随手定的：`abfe_pipeline.py` 里 2026-07-19 的原地注释记录了一次真实 GPU 回归——10000 步的短 pilot 在 λ≈1 端点系统性低估稀有/发作性事件主导的 `beta²Var[dU/dlambda]`，导致 window_0 反复 `IBSWarmupConvergenceError`。`_refine_pilot_grid_in_steep_segments()`（另一次独立的真实回归修复，THERMODYNAMIC_PATH_PROTOCOL_VERSION=15）的加密点用的也是同一个 30000，且加密点存在的理由本来就是"父区间空间信息不足"，最不适合缩短采样时长。任何缩短 pilot 预算的方案都必须先证明不会重新踩这两个坑，不能只凭静态论证。

**已排除的方向**（讨论中证伪或用户明确划走，跟本节改动无关）：跨 repeat 共享 λ 规划缓存（调查发现缓存指纹 `_preopt_protocol_key` 混进了逐 repeat 都会重算的 Boresch 限制平衡值，用户明确表示这次要解决的是 pipeline 代码本身、不是 repeat 复用问题）；Stage1 decharging 12→11 态 / Stage2 6→5 窗口（用户自己标注"需要先 A/B，不能只改常数"）；直接把 30000 全局砍短（被 07-19 历史证伪）。

### 最终设计（用户逐轮修正后敲定）

风险点（λ≈1 已知危险区、当前最长热力学区间两端、**所有加密点**——继承父区间风险属性、高峰度/突发异常值/时间漂移、metric 置信区间宽、metric 变化仍明显移动 production λ）保持 ≥30000 硬地板，**30000 只是当前真实生产历史验证过的最低预算，不是收敛证明、不是上限**；普通基础点允许分块采样+提前停，但必须同时满足稳定性判据 + 一个额外的压力测试（把当前 metric 按历史最坏比例向上膨胀后重算一次 production λ 布点，位移仍低于容差）——因为峰度/置信区间这类基于单条短轨迹的判据本质上无法可靠区分"真收敛"和"还没等到稀有事件所以看起来稳定"。

落地策略分两阶段：**Phase A（本次，零行为改变，纯插桩）**先在真实 30000 步预算不变的前提下，每隔一定步数额外记一次"假想现在停下会怎样"的诊断，不实际改变任何采样长度；**Phase B（下次会话，需要真机数据）**用这些 shadow 数据验证"风险点是否曾被误判可以提前停"（必须零次），验证通过后才实现真正的提前停。

### 已落地的具体改动（均已过 AST 检查 + 真机 `openmm_dev` 环境 import + pytest 验证）

- **`abfe_preoptimizer.py`** 新增 4 个纯 Python/numpy 函数（不依赖 OpenMM Context，`class DualLambdaPreOptimizer` 之前）：
  - `_pilot_segment_lengths()`：从 `_refine_pilot_grid_in_steep_segments()` 内联代码抽出的共享实现，数值行为不变，顺手去重复。
  - `pilot_block_running_diagnostics()`：给定某点截至目前的 dU/dλ 样本，算 mean/std/SEM/metric_g/超额峰度/基于 MAD 的稳健 z 分数（抓突发异常值，不像普通 z 分数那样被该值自己拉高的标准差稀释）。样本 <2 时全字段安全退化成 NaN，不抛异常。
  - `classify_pilot_point_risk_zone()`：事后打标签——加密点、λ≥0.875（覆盖 `human_vanishing_initial_lambdas` 17 点网格里 λ=1.0 起最前两段）、当前最长热力学区间两端 → `"risk"`，其余 `"easy"`；数组长度不一致时整体退化成全 `"risk"`。
  - `pilot_early_stop_pressure_test()`：把某点的部分/膨胀后估计代入已有的 `redistribute_vanishing_lambda_subdomains()`，跟用完整数据的基线比较 production λ 位移；`worst_case_inflation_ratio` 默认值 3.0 是**占位符**，需要 Phase B 真机数据回填。永不抛异常，输入非法时返回 `{"valid": False, ...}`。
  - `_sample_scalar_metric()`/`optimize_stage2_vanishing()`/`_refine_pilot_grid_in_steep_segments()` 各加一个默认 `None` 的可选参数（`shadow_checkpoint_steps`/`shadow_checkpoint_interval`）：不传时对应的 `pending_checkpoints` 恒为空列表，循环体不执行、不产生 `shadow_trace`/`risk_zone_tags` 字段，真实采样长度和现有返回值形状逐字节不变；显式传入正整数时才会在每跑够这么多步就多记一次 shadow 快照，不改变任何一次真实采样的步数或判断分支。加密点在 `point_diag["is_refinement_point"]` 上无条件标 `True`，基础点标 `False`。
- **`abfe_pipeline.py`**：`_run_dual_lambda_optimization()` 新增 `pilot_shadow_checkpoint_interval` 参数（默认 `None`），"vanishing" 分支从 `kwargs.get("pilot_shadow_checkpoint_interval")` 读取并透传给 `optimize_stage2_vanishing()`。
- **`abfe_config.json`**：新增 `"pilot_shadow_checkpoint_interval": null`（默认关闭，带 `_comment_` 说明字段）。

### 测试结果（真机 `openmm_dev` conda 环境，纯 CPU）

- `tests/test_core_physics_numerics.py` 新增 10 个测试：用 **repeat01 真实完整跑完的 pilot 数据**（`output_lrc_fix_repeat01_seed20260905/checkpoints/preopt_dual_vanishing.json` 落盘的 17 点 `pilot_lambdas`/`metric_g`）做 fixture，不是手造合成数据——避免踩 `redistribute_vanishing_lambda_subdomains()` 内部单调性等隐藏不变量的坑，同时断言的是这组函数在真实体系上到底做了什么。覆盖：λ≈1 三点/最长区间（index10↔11，用独立 numpy 表达式验证过是这个例子里的最大值）被正确标 risk、加密点无条件标 risk 即使既不近端点也不在最长区间上、数组形状不匹配退化成全 risk、压力测试对"自我替换"给零位移、对"把最大贡献点砍到 10%"给出真实位移且判定不通过、坏输入一律 fail-closed。
- 全文件 83/83 通过；`tests/test_warmup_overlap_protocol.py`（既有的、直接调用 `optimize_stage2_vanishing()` 的测试）56/56 通过，零回归。
- 顺带发现两个**跟本次改动无关的既有问题**，未修：`tests/run_offline_tests.sh` 硬编码的 mamba hook 路径 `/home/ruigengji/mambaforge/etc/profile.d/mamba.sh` 跟环境已迁移到 `miniforge3`（见本文件"P1-19"相关会话记录）不一致，官方入口脚本本身跑不起来，本次全程改用 `conda activate openmm_dev` 直接跑绕过；`scripts/` 缺 `__init__.py` 导致 `tests/test_exp017_overlap_first.py`/`tests/test_exp029_harness.py` 两个既有文件收集失败（跟本次改动无关，改动前后一致）。

### 未做 / 待做

- **Phase A 真机 smoke**：还没在真实 GPU 上跑过一次 Stage2 vanishing pilot，确认打开 shadow 插桩前后产出的 λ 路径逐字节相同（证明真的零行为改变）——需要真实体系拓扑 + GPU 节点，本会话无 GPU 访问。
- **Phase B**（离线分析脚本 + 用真实 shadow 数据验证风险点是否曾被误判提前停 + 回填 `worst_case_inflation_ratio`）：按设计要等 Phase A 真机数据出来才开始，不在本次范围内。
- 详细讨论过程与设计取舍记在 Claude Code 计划文档 `/home/kasuga/.claude/plans/swirling-strolling-octopus.md`（仓库之外，会话专用目录）。
