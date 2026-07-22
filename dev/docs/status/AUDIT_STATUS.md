# ABFE 审计状态总表（历史记录；当前覆盖更新至 2026-07-21）

> **当前覆盖说明**：源码采用 `IBS_BIAS_PROTOCOL_VERSION=26`、
> `THERMODYNAMIC_PATH_PROTOCOL_VERSION=18`、`TRADITIONAL_LJ_LRC_PROTOCOL_VERSION=2`
> 和 `WCA_ACCOUNTING_VERSION=2`。本文件正文保留 2026-07-14～20 的审计演进，较早章节中
> 的“当前”“未修复”和旧协议版本均只代表当时快照。现行行动清单以
> [`../TODO.md`](../TODO.md) 为准，代码完成但待运行的项目以
> [`VALIDATION_MATRIX.md`](VALIDATION_MATRIX.md) 为准。

本文件合并并取代以下三份旧审计文档的当前有效内容：

- `RELEASE_AUDIT_REMAINING.md`
- `PHYSICS_DEFECTS.md`
- `RE_AUDIT_2026-07-10.md`

审计范围：`abfe_core.py` / `abfe_pipeline.py` / `ibs_engine.py` / `runabfe.py` / `abfe_preoptimizer.py` / `apbs_correction.py`。`dexp_experiment.py` 属于独立实验模块，不作为当前主链审计对象。

运行上下文：中性配体 Atenolol（总电荷约 0）、`decoupling=dual_lambda`、`potential=softcore`、`boresch_source=simple`、IBS/TMBAR、CUDA、production 预设、T=300 K。当前结果中 APBS correction 默认为 0，未作为主链采样的一部分启用。

文档采用三层维护结构：

- `AUDIT_STATUS.md`：问题背景、修复依据、历史演进和最终审计结论。
- [`../TODO.md`](../TODO.md)：尚未实现、仍需修改代码或采取工程行动的事项。
- [`VALIDATION_MATRIX.md`](VALIDATION_MATRIX.md)：代码已修但仍等待完整测试、GPU 实测或证据归档的项目。

GitHub 跟踪入口：

- 仓库：[`Cedrus810/openmm_IBS_dev`](https://github.com/Cedrus810/openmm_IBS_dev)
- 看板：[`Project #1`](https://github.com/users/Cedrus810/projects/1)
- 当前未修代码：Issue [`#1`](https://github.com/Cedrus810/openmm_IBS_dev/issues/1)，Project `Ready`。
- 当前运行验证：Issues [`#2`](https://github.com/Cedrus810/openmm_IBS_dev/issues/2)–[`#4`](https://github.com/Cedrus810/openmm_IBS_dev/issues/4)，Project `In review`。
- 2026-07-16 已完成审计档案：Issues `#7`–`#22`，均已关闭并保留在 Project `Done`；
  `#5`/`#6` 是导入重试产生的重复项，已标记 `duplicate` 并关闭，未加入 Project。

## 2026-07-16 当前代码复审（IBS v10-v13 / P2 批量修复 / native resume / production checkpoint）

本节是当前源码状态的最新审计记录。下方 2026-07-14/15 章节保留问题发现与修复的
历史过程；其中仍写成“未修复”、旧协议版本或旧优先级的内容，若与本节冲突，以本节、
当前源码和 `todolist.md` 为准。

### IBS_BIAS_PROTOCOL_VERSION 10 — 已修复：路径 overlap 探针被错误用于校准生产偏置

v8/v9 曾直接使用路径/λ 密度探针的 `delta_f_kJ_mol` 校准生产 `f_k`。该探针采样
`U_common + CV_k`（不含生产窗口的 Group 4 WCA），并把仅用于最终 MBAR 的 LRC 加进
能量；生产偏置实际对应 `U_common + WCA_window(lambda_shield) + CV_k`，且偏置 CV 本身
不含 LRC，因此这是确定的 ensemble/能量口径错配。

v10 将两种物理问题拆成独立实现：

- `probe_bidirectional_overlap` 继续只回答路径连通性/λ 密度问题：不含 WCA，目标能量含 LRC。
- `probe_bidirectional_overlap_for_bias_calibration` 使用与生产一致的
  `U_common + WCA_window + CV_k` 动力学，只计算纯 bias-CV 自由能差，不含 LRC，并输出
  独立字段 `delta_f_bias_kJ_mol`。
- bias 校准新增每侧至少 20 个去相关样本和 `Delta F_bias` 不确定度不大于
  1.0 kJ/mol 的双重门；不达标时延长采样，最终仍不达标则拒绝覆盖 `f_k`。
- 校准验证失败诊断改用本次验证真正观测到的 `last_validation_batch_p`，不再打印
  校准前 SGD 的陈旧 EMA。

状态：**已修复。** 两类探针的结果不能再交叉使用。

### IBS_BIAS_PROTOCOL_VERSION 11 — 已修复：fixed-H 探针按边重复采样且不能可靠续采

旧实现对 K 态窗口的 K-1 条边分别新建两套 Simulation；内部态会重复 burn-in，精度
重试也会丢弃已经得到的样本。v11 改为按状态共享的
`probe_adjacent_path_overlap_bank` / `probe_adjacent_bias_calibration_bank`：K 态只采 K 份
轨迹，每个态同时服务左右两条边，各边的双态 MBAR 仍保持独立。

轨迹库位于
`checkpoints/probes/<stage_type>/window_<idx>/<path_probe|bias_calibration_probe>/`，
manifest 指纹覆盖系统/CV XML、λ、`lambda_shield`、温度、平台和协议版本。当前
`FIXED_H_PROBE_CACHE_PROTOCOL_VERSION=2` 使用 OpenMM native Context checkpoint 保存
积分器 RNG 状态；NPZ 仅作不兼容时的降级入口，降级后必须新开 segment 并重新 burn-in。
各 segment 独立做自相关去除，短于最小帧数的 segment 只进入
`short_segments_diagnostic_only`，不再按 `g=1` 混进 MBAR。

同轮加固还包括：`volume.npy` 缺失/非有限/非正时整态记录判坏并重采；每态结束后解除
evaluator 对 dynamics Context 的持有，降低 GPU Context 峰值；fixed-H 状态在设置真实
`lambda_shield` 和本态 CV 后再次最小化并逐级升到生产步长，修复首步坐标 NaN 路径。

状态：**代码级已修复；真实 GPU 显存峰值和原 `(5,9)` NaN 案例仍需复验。**

### IBS_BIAS_PROTOCOL_VERSION 12 — 已修复：近似偏置力、冻结校准状态混淆和无效续验

真实 GPU 日志显示，fixed-H overlap 与 bias 校准均通过后，某窗口仍长期停在
`p=[0.8485, 0.1502, 0.00126]`。审查确认并修复三类问题：

1. `IBSBiasForce` 删除 `80*tanh(logit/80)` 近似，改用精确、数值稳定的全局 max-shift
   log-sum-exp；采样偏置力、`update_weights()` 和冻结验证现在使用同一 softmax 数学模型。
2. `IBSSampler` 新增 `bias_status` 与 `frozen_f_k_pending`，明确区分
   `unconverged`、`calibrated_pending_validation`、`converged` 和终态
   `calibrated_validation_failed`。已由 fixed-H+bias 校准证明过的 `f_k` 在续验失败时
   不再回到 learning/SGD，也不会触发拆窗或插 λ。
3. 冻结验证使用 50k→150k→300k 的**累计目标预算**；落盘
   `frozen_validation_cumulative_steps`，每轮只运行“新目标减去已完成累计值”，避免把
   三档误跑成 50k+150k+300k。

v12 上线后的同日补丁修正了续验分支单批失败仍无条件切回 `mode="learning"` 的状态机
错误；续验现在始终保持冻结 `f_k`，用独立诊断计数器记录验证重试。最后一档仍失败时
写入 `calibrated_validation_failed` 并要求人工检查，不再无限自动重试。

随后补齐此前注释声称存在、实际没有接线的主窗口原生续算：
`MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION=1` 以系统 XML、λ、`lambda_shield`、平台、温度、
IBS 协议等内容指纹保护 `checkpoints/main_window/<stage_type>/window_<idx>/` 下的 OpenMM
native checkpoint。每个冻结验证 batch 后覆盖保存坐标、速度、盒子和积分器 RNG 状态；
指纹匹配时 resume 直接从 validating 继续，跳过重新最小化、dt 测试、Boresch 爬坡和
重复 freeze burn-in；缺失或不兼容时才回退完整重建，但仍不回退 SGD。窗口收敛后清理
checkpoint，终态失败时保留供人工排查。

状态：**代码级已修复，并新增 manifest/roundtrip/累计步数/终态状态回归测试；仍需真实
GPU 复验 v12 状态机和精确 log-sum-exp 无回归。**

### 2026-07-16 P2 批量修复 — 13 项已从 `todolist.md` 关闭

本轮按当前源码逐项确认以下修复已经落地：

1. fixed-H 短 segment 仅作诊断，不再进入 MBAR 去相关索引。
2. fixed-H bank 每态结束后释放 evaluator 持有的旧 dynamics Context。
3. `volume.npy` 缺失/损坏不再补零，直接判该态缓存损坏并重采。
4. λ 插入、窗口 split/canonicalize 或编号变化时清空 `pending_step_overrides`，避免把延长步数发给别的窗口。
5. fixed-H 逐边结果和 `sampling_repair_decisions.json` 改用 `_atomic_write_json`。
6. `--analyze-only` 严格校验 stage checkpoint 的 stage、`total_delta_G`、`total_error` 类型/有限性；窗口文件按数值编号并要求连续，回退求解未收敛时 fail closed。
7. 找到 Boresch 参数但解析修正计算失败时直接报错，不再把整项默认为 0。
8. 实验性 Shadow-Bridge/Shadow-IBS 两条子腿及组合结果均传播并硬门控 `converged`/`min_overlap`。
9. Boresch 最终 sanitation 与 estimator/解析函数统一：`kr=[100,2000]`，角/二面角力常数 `[10,1000]`，取消来源相关的二次软化。
10. Stage 1 与单 λ 路径共享 `finalize_descending_lambda_path`，统一有限性、端点、最小间距、去重和线性回退不变量。
11. `ChunkedMBARAnalyzer` 将 pymbar 的 kT 转为 kJ/mol；`OnlineConvergenceMonitor` 走兼容层；自动锚点 RMSF 改为相对平均结构。
12. APBS 网格体积改用 `(nx-1)(ny-1)(nz-1)` 个体素并与 `--box` 体积交叉校验；使用立方晶格常数的有限尺寸项在明显各向异性盒上 fail closed。
13. 2D geodesic 长跳代价沿路径分段、双线性插值并作梯形积分，不再只用两个端点跨过高方差脊。

此外，`todolist.md` 16:08 版本仍列出的“冻结验证延长重试未保存主 dynamics Context”已在
16:29 后的当前源码中由 `MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION=1` 完成，因此同步从
待办移除。

### IBS_BIAS_PROTOCOL_VERSION 13 — 已修复：fixed-H 探针 Context 与生产 Hamiltonian 的
Boresch 限制不一致

用户复核代码发现：fixed-H overlap 探针（`probe_bidirectional_overlap`）、bias 校准探针
（`probe_bidirectional_overlap_for_bias_calibration`）和探针轨迹库共享的
`_build_fixed_state_simulation` 新建 Context 后从未设置 `lambda_boresch_scale`——该参数
的 System 级默认值是 `0.0`（见 `LambdaDependentBoreschForce.__init__`，`fixed_lam=None`
时才注册为可变全局参数），而主窗口生产/冻结验证早已把它爬坡到 `1.0` 并全程保持
（`ibs_engine.py` 的 Boresch 安全爬坡）。三份探针据此评估、证明“f_k 物理正确”的其实是
关掉 Boresch 限制的系统，跟它们要验证的生产 Hamiltonian 不是同一个——这与用户此前报告
的“冻结 f_k 连续 29 批验证严重偏斜”现象一致（`state 0` 29 批全部未过门槛）。

修复：

1. 三处探针 Context 构建代码在 `setPositions` 之后、任何最小化/动力学之前，都补上
   `if _system_has_global_parameter(fixed_system, "lambda_boresch_scale"):
   simulation.context.setParameter("lambda_boresch_scale", 1.0)`，与已有的
   `lambda_shield` 处理模式一致。对没有 Boresch 限制（`fixed_lam=1.0` 常量路径，无全局
   参数）的系统这段判断自然是 no-op，不影响那条路径。
2. 升版本号强制作废所有在这个 bug 修复之前落盘、可能带着偏置的缓存/状态：
   `IBS_BIAS_PROTOCOL_VERSION` 12→13（`save_ibs_state`/`load_ibs_state` 门控，作废
   `frozen_f_k_pending`/`calibrated_pending_validation` 缓存）、
   `FIXED_H_PROBE_CACHE_PROTOCOL_VERSION` 2→3（作废探针轨迹库的原始采样数据）、
   `MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION` 1→2（作废主窗口 native checkpoint——它的
   manifest 只哈希 `win_sys_xml`/λ/`lambda_shield`/温度/平台，不包含代码指纹，Boresch
   Context 修复不改变其中任何一项，靠内容指纹本身感知不到这次修复）。三个版本号的
   不匹配都是 fail closed、优雅回退到重新计算，不抛异常。

状态：**代码级已修复，`ibs_engine.py`/`abfe_pipeline.py` `py_compile` 通过。** 生产验证
正在进行：修复后重跑观察到窗口 0 的冻结验证在新 Hamiltonian 下正常经历一次 learning
失败→学习率减半→收敛的完整周期并通过独立验证（不再是此前 29 批全部卡在门槛以下的
模式），但尚未跑完整个 stage 得到最终 ΔG 结论。

### 已修复：`_run_stage_with_overlap_autorepair` 的两处自动修复循环缺陷（production ESS 修复）

复审 vanishing 阶段一次真实、长期停滞的自动修复循环（持续拆窗/插 λ 数小时不收敛）
发现两个独立但会相互放大的控制流 bug：

1. **每轮只插一条边。** production ESS 分支即使同一轮里多个窗口各自有失败边，也只处理
   `to_probe` 里“最差窗口”的一条边，其余窗口的失败边留到下一轮——跟 `to_split`
   （已经批量处理、按 `-start` 逆序避免邻窗重排冲突）不对称。改为收集本轮所有
   `still_failing` 窗口各自的最差失败边，按全局边索引去重、从大到小依次插入，每次插入
   后用 `insert_lambda_from_overlap_failure` 内部同款的 4 分支位移规则同步更新尚未处理
   窗口的 `(start,end)`，一轮内一次性修完当前已知的所有密度缺口。同时修了一个连带的
   潜在 bug：`to_split` 阶段的邻窗重排可能已经移动了某个 `to_probe` 窗口的位置，插边前
   先按 λ 内容重新定位，定位不到时该窗口本轮跳过、留给下一轮，而不是从
   `insert_lambda_from_overlap_failure` 内部直接 `raise`。
2. **`already_good` 窗口只要本轮 `path_will_change` 就整轮推迟修复，且这种推迟没有独立
   预算。** 因为几乎每一轮 18-20 态的路径都会在别处冒出新的失败边，`already_good`
   窗口（fixed-H 已通过、真正需要 f_k 重新校准/重采样的窗口）被反复推迟、永远轮不到，
   同时白白消耗跟拆窗/插 λ 共享的 `repair_round` 预算，最终在从未真正修复这些窗口的
   情况下触发 `max_repair_rounds` 硬停止。改为区分“路径本轮不变”（原逻辑立即处理）与
   “路径本轮会变”（推迟到本轮拆窗/插 λ/`_invalidate_stage_window_files` 完成之后，
   在同一轮内按重映射后的新窗口范围应用修复，不再推迟到下一轮）两条路。新增
   `_window_lambda_key`/`_remap_window_by_lambda_content`（从 `_invalidate_stage_window_files`
   内部逻辑提出的共享辅助）和 `_apply_already_good_repairs`（诊断/落盘/打印的共享实现，
   两条路复用同一份逻辑）。`max_overlap_repair_rounds` 默认值同时从 5 提到 8 作为额外
   安全余量（不是替代上面两处真正的修复）。

状态：**代码级已修复，`abfe_pipeline.py` `py_compile` 通过。** 尚未获得真实 GPU 运行下
“批量插边”和“`already_good` 同轮修复”两条新路径被实际触发并正确工作的直接证据。

### 已修复：`reseed_resample` 生产采样修复实际是整窗丢弃重采，不是真延长

`_diagnose_and_repair_all_pass_low_ess_window` 的 `reseed_resample` 决策（fixed-H 全通过
且生产冻结 f_k 与 fixed-H BAR/MBAR ΔF 在噪声阈值内一致，判定偏置本身没问题）此前调用
`_invalidate_single_window_production` 无条件删除该窗口的
`energies`/`bias`/`base.npy`/`convergence.json`，下一轮从 stage 起始坐标完整重建 Context、
重新最小化/dt 测试/Boresch 爬坡/冻结重验证、从零步开始生产——`n_steps_per_window` 覆盖值
虽然确实按 2×（封顶 4×）增长，但增长后的目标步数是“重新采这么多步”，不是“接着已有的
继续采”，250k 延长到 500k 实际是丢弃 250k、独立重采 500k。

修复：

1. `ibs_engine.py` 新增生产窗口 checkpoint（`PRODUCTION_WINDOW_CHECKPOINT_PROTOCOL_VERSION=1`，
   manifest 含系统 XML 哈希、λ 值、`lambda_shield`、Boresch scale、冻结 f_k 哈希、协议
   版本），布局/落盘方式照抄已有的主窗口 checkpoint 模式。`run_all_windows` 进入生产
   采样前检测：λ/窗口/系统/冻结 f_k 完全一致的 checkpoint + 仍在的 `energies`/`bias`/
   `base.npy`/`convergence.json`（未被作废）+ `convergence.json` 里的
   `cumulative_production_steps` 小于本次目标时，从上次结束的坐标/速度/积分器 RNG 状态
   续算，只跑差值步数，把新样本追加到已读回的能量/偏置/基准能量历史，而不是从零开始。
   每 100 个 update 覆盖式落盘一次 checkpoint+当前数组+累计步数（应对 HPC 作业被抢占/
   撞墙时限杀掉），窗口正常结束时再落盘一次最终版本。顺带修复一个原子性缺口：
   `convergence.json` 最终落盘之前是普通 `open()+json.dump`，跟同一路径下
   `energies`/`bias`/`base.npy` 用的 `_atomic_save_npy` 原子性不一致，改为
   `_atomic_write_json`。
2. `_invalidate_single_window_production` 新增 `keep_production_data` 参数：
   `recalibrate_f_k`（f_k 真的被覆盖）保持默认 `False`，旧 f_k 下采的数据必须丢弃；
   `reseed_resample`（f_k 确认没问题）改为 `True`，保留 production 文件和 checkpoint，
   交给上面的续算检测真正接着跑。

状态：**代码级已修复，`ibs_engine.py`/`abfe_pipeline.py` `py_compile` 通过。** 尚未获得
真实 GPU 运行下续算路径被触发、`cumulative_production_steps` 正确累加、以及跨多次
`reseed_resample` 循环产出与真正不间断长轨迹等价结果的直接证据。

### 已修复：λ 路径预优化缓存被无关代码修复连带作废

`_stage_protocol_key()` 的 `code_sha256` 字段哈希 `abfe_pipeline.py`/`abfe_core.py`/
`ibs_engine.py`/`abfe_preoptimizer.py` 四个文件整体，Stage 1/Stage 2 的 λ 路径预优化
（`optimize_stage1_decharging`/`optimize_stage2_vanishing`，跑一次数小时的
thermodynamic-length 逐点扫描）此前复用同一份宽指纹判定缓存有效性——导致修复上面这些
跟预优化完全无关的 bug（窗口修复循环、production checkpoint 续采等）都会连带让这份
昂贵缓存判定失效、被迫整段重算。

修复：新增范围更窄的 `_preopt_code_hash()`（只哈希 `abfe_preoptimizer.py`+`abfe_core.py`）
和 `_preopt_protocol_key()`（去掉 `wca_accounting_version`/`ibs_bias_protocol_version`/
`final_gate_thresholds`，这些字段跟预优化物理内容无关），Stage 1/Stage 2 预优化缓存的
读写全部切到这份窄指纹。另外补了一次性 schema 迁移兼容：
`_preopt_cache_matches_ignoring_code_hash` 逐字段核对旧宽指纹缓存与当前窄指纹的物理相关
字段是否一致，一致则判定为纯 schema 迁移、原地重盖 `protocol_key`，不触发重新优化——
避免窄指纹上线前已经算好的缓存被当成“协议不一致”白白重算一次。另新增
`ABFE_DEBUG_FREEZE_CODE_HASH=1` 调试逃生舱：显式 opt-in 时把 `_code_hash()`/
`_preopt_code_hash()` 冻结为固定常量，方便调试收敛逻辑时反复改代码不连带让 stage/探针/
预优化缓存失效；系统/坐标/协议版本常量等其它字段仍正常参与比较，且启动时打印醒目警告，
提醒在正式出结果前必须取消设置。

状态：**代码级已修复，`abfe_pipeline.py` `py_compile` 通过。** 已在真实运行中观察到
迁移兼容路径正确触发（日志出现"🩹 ... 判定为 schema 迁移，不重新优化"而非旧的
"协议指纹不一致...重新运行"）。

### 运维记录：`ibs_engine.py` 一次性截断事故（非代码 bug）

本轮修复期间，`ibs_engine.py` 曾被另一个并发编辑进程截断在 `run_shadow_bridge_leg()`
的 docstring 中间，`TraditionalMBARAnalyzer` 整个类（含 `compute_u_kn`/`solve`）从磁盘
消失，`python -m py_compile` 报语法错误。恢复方式：`__pycache__/ibs_engine.cpython-314.pyc`
在截断前的某次导入时已完整编译成功（Python 只在源码干净编译时才写 `.pyc`），用
`marshal.load` 读出该字节码，结合同目录下更早的 `.pre_warmup_overlap_patch` 备份重建
源码，再用 `dis.dis` 反汇编逐条比对 `co_names`/字符串常量校验重建结果与原字节码完全
一致（包括发现并按字节码精确修正了 `compute_u_kn` 里一段 LJ 长程尾项公式在两份材料之间
真实存在的差异，不是简单的行数对齐）。恢复过程中并发进程自行完成了同一份修复，最终
只用字节码交叉校验确认磁盘上的文件已经完整、正确，未实际写回重建内容。记录在此仅为
说明这段时间内文件曾处于不一致状态，不代表源码本身有新增缺陷。

### 当前审计结论与剩余验证

- **默认生产路径没有剩余的已确认 P0/P1；上述 13 项 P2、主窗口 native resume，以及当晚
  新增的 fixed-H Boresch Context 修复、production ESS 修复循环批量插边/`already_good`
  饥饿修复、`reseed_resample` 真续算、预优化缓存范围收窄均已代码级修复。**
- **当前磁盘上此前用旧 Boresch=0 探针校准/验证过的 `frozen_f_k_pending`、探针轨迹库和
  主窗口 checkpoint 已被 IBS_BIAS_PROTOCOL_VERSION=13/FIXED_H_PROBE_CACHE_PROTOCOL_VERSION=3/
  MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION=2 的版本门控作废，重跑会从每个窗口的冻结验证
  重新开始，这是预期行为，不是回归。**
- 当前磁盘上的旧 `output/final_binding_results.json` 仍来自旧符号/旧协议，不能因为源码修好就视为新结果；必须按 README 说明刷新汇总并完成独立重复运行。
- 唯一保留的性能加固项是 production ESS 自动修复第二调用点仍使用旧 per-edge 探针；这不改变当前数值定义。
- 本地环境缺少 OpenMM/PyMBAR，本轮只完成静态检查（`py_compile`）；相关回归测试已经补入
  源码，但尚未在目标运行环境完整执行 pytest。fixed-H Boresch Context 修复已在真实运行
  中观察到窗口冻结验证正常收敛（不再是此前 29 批全部卡在门槛以下的模式），但尚未跑完
  整个 stage 得到最终 ΔG 结论；具体运行状态、验收条件和证据要求统一跟踪在
  `VALIDATION_MATRIX.md`。

## 2026-07-14 当前代码复审（warmup / fixed-H overlap / resume / LRC）

本节记录 2026-07-14 对当前工作区代码的重新审核结论；下方较早日期的条目保留为历史演进记录。若旧条目中的版本号或“仍未修复”描述与本节冲突，以本节和当前源码为准。

### 2026-07-14 同日复审第七轮新增 — 已修复：K=4 窗口被错误拆成 K=2+K=3，且拆分后未 canonicalize

第七轮在真实运行日志里发现两个叠加的窗口规划 bug（不是 OpenMM/LRC 崩溃，也不是第五轮的求解器问题）：

1. **拆分阈值边界条件错误**：`split_window_from_warmup_failure`/`plan_vdw_overlap_repair_targets` 都用 `min_states_before_split=4` 判断"能不能拆"，子窗口大小下限只查 `<2`。但两个孩子各自需要 >=3 态（2 态 IBS 窗口本来就统计脆弱，`run_all_windows` 自己的注释也这么说）、共享 1 态，父窗口因此至少要 `3+3-1=5` 态才够拆。真实案例：`[2,6)`（K=4）被拆成 `[2,4)`（K=2）+ `[3,6)`（K=3），产出了一个明知脆弱的两态窗口。`ibs_engine.py::run_all_windows` 里触发 fixed-H overlap 探针的边界同样用的是 `K<=3`，让 K=4 窗口从未被送去做探针，直接走了拆窗分支。
2. **warmup 拆窗后没有 canonicalize**：`abfe_pipeline.py` 的 warmup 失败修复分支（`except IBSWarmupConvergenceError` 分支）拆完窗口后直接落盘 `new_ranges`，没有像 production ESS 分支那样调用 `canonicalize_window_ranges()`。真实案例：`[2,6)` 拆成 `[2,4)`+`[3,6)` 后，未拆的旧邻窗 `[3,9)` 原样保留，`[3,6)` 完全被 `[3,9)` 包含——这次拆分产出的一个孩子对覆盖范围零贡献，纯属白跑。

窗口 `[2,6)` 的 warmup 失败本身（`ema p=[0.0218,0.4151,0.3433,0.2198]` 未通过冻结验证）只说明 IBS 偏置没通过验证，并不能证明是 λ 密度问题——K=4 本就不该盲拆，应该直接走 fixed-H 双向 overlap 探针（通过→求解器/弛豫问题，硬停止；失败→只在实测失败边插 λ）。

修复：

- `split_window_from_warmup_failure`/`plan_vdw_overlap_repair_targets` 的 `min_states_before_split` 默认值 4→5；子窗口大小下限 `<2`→`<3`；`abfe_pipeline.py` 两处调用点（warmup 失败分支的 `n_states>=4` 判断、production ESS 分支传给 `plan_vdw_overlap_repair_targets` 的显式 `min_states_before_split=4`）同步改为 5。
- `ibs_engine.py::run_all_windows` 里触发 fixed-H 双向 overlap 探针的边界从 `K<=3` 改为 `K<=4`，跟上面的 5 态下限保持一致；诊断字段 `feedback_action` 的判定边界同步从 `K>=4` 改为 `K>=5`。
- `abfe_pipeline.py` 的 warmup 失败修复分支落盘前新增 `new_ranges = canonicalize_window_ranges(new_ranges, len(new_lambdas))`，跟 production ESS 分支保持一致。
- `THERMODYNAMIC_PATH_PROTOCOL_VERSION` 6→7，`IBS_BIAS_PROTOCOL_VERSION` 8→9（窗口拆分阈值和 fixed-H 探针触发边界都变了，旧协议缓存可能包含用旧边界条件产出的脆弱窗口，不安全复用）。新增回归测试 `test_warmup_split_rejects_four_state_window_instead_of_bisecting_into_k2_plus_k3`（验证 K=4 直接拒绝而不是盲拆）。

状态：**已修复。**

### 2026-07-14 同日复审第六轮新增 — 已修复：split 只替换失败窗口，未与未拆邻窗重新协调重叠

第五轮以为根因在 f_k 求解器，但第六轮复审把它进一步定位到了窗口边界规划本身，是两层设计叠加：

1. **初始铺窗本就要求较大重叠（设计如此，非 bug）**：`pilot_overlap_thermodynamic_length=1.5`、优化后单边长约 0.957 时，规划器需要共享两条边（三个 λ 态）才能满足累计重叠预算，因此 `partition_windows_by_thermodynamic_length` 天然产出 `(0,6),(3,9),(6,12),...` 这种相邻窗口共享 3 态的方案。
2. **`split_window_from_warmup_failure` 只替换失败窗口，不重新协调邻窗**：`(0,6)` 拆成 `(0,3),(2,6)` 后，右孩子 `(2,6)` 的 `end` 跟原父窗口完全相同，未拆的旧邻窗 `(3,9)` 因此原样保留——绝对重叠态数没变（还是 `{3,4,5}` 3 态），但相对比例从"6 态窗口共享 3 态"恶化成"4 态孩子共享 3 态"（75%），几乎失去自己的独立采样区间。`canonicalize_window_ranges()` 故意不处理这种"部分重叠但不互相包含"的情况（那是它应该保留的合法场景），所以这个问题不会被现有归约逻辑捕获。真实缓存 `preopt_dual_vanishing.json` 的 provenance（`source=warmup_window_split_only`, `failed_global_state_range=[0,6]`, `child_ranges=[[0,3],[2,6]]`）证实了这正是触发路径。

修复（`abfe_preoptimizer.py::split_window_from_warmup_failure`，新增 `min_states_per_window_floor=2` 参数）：拆分后，只回看失败窗口在原列表中紧邻的**下一个**邻窗——若它跟新右孩子当前共享的状态数超过 1，就把它的 `start` 收缩到 `right_child_end - 1`（只共享 1 个态），前提是收缩后不低于最小态数下限（否则保留原样，不强行压缩）。不触碰失败窗口左侧的邻窗（左孩子的 `start` 跟原失败窗口相同，不受影响），也不级联到被调整邻窗右侧的下一个窗口（该邻窗的重叠对象由它自己不变的 `end`决定）。用真实的 5 窗口案例手动核算：`(0,6)` 拆分后 `(2,6)`-`(3,9)` 的重叠从 3 态收窄到 1 态（`(5,9)`），其余未涉及的邻窗对（`(5,9)`-`(6,12)` 等）重叠维持原设计的 3 态不变，覆盖范围完整。

同时修了一个由此暴露的批量场景问题：production ESS 修复分支里 `to_split` 循环若包含两个相邻的失败窗口，先处理低 `start` 的窗口可能被后处理的高 `start` 窗口的"邻窗收缩"逻辑改写掉、导致后者用来查找自身的原始 `(s,e)` 已经不在 `new_ranges` 里而报错。修复：该循环改为按 `start` 降序处理（`abfe_pipeline.py`），确保任何可能收缩某窗口"邻窗槽位"的处理都晚于该窗口自身被处理的时刻。

状态：**已修复。** 本轮未改动初始 `pilot_overlap_thermodynamic_length=1.5` 的全局默认值——按复审结论，那是有意的设计选择，不是需要修的 bug；只修复了拆分之后失去协调的边界维护。

### 2026-07-14 同日复审第五轮新增 — IBS_BIAS_PROTOCOL_VERSION 7→8：f_k 求解器本身不稳定

第四轮的 v7 状态机在真实运行中被证实工作正常：一个 6 态窗口冻结验证失败后正确拆成 `(0,3)`/`(2,6)`；`(2,6)` 这个最小 3 态窗口连续 4 次"learning→冻结验证→失败→重新学习"，50 次权重更新后仍塌缩到 `[0.1211, 0.00165, 0.8772]`——但同一组 λ 的 fixed-H 双向 overlap 全部通过（edge 0→1 min overlap=0.3618，edge 1→2 min overlap=0.0727，阈值 0.03）。结论明确：λ 网格没断、窗口已经不能再拆，失败来自 `f_k` 学习算法本身，v7 在这里正确硬停止，没有继续乱拆窗口或插 λ。

根因确认（均已直接读代码验证）：

1. **`update_weights()` 的梯度也读了跨 Hamiltonian 的 EMA**：`log_grad = np.log(self.ema_mean_p + ...) - np.log(target_p)` 用的是 `gamma=0.9` 的滞后 EMA，而不是这次在当前 `f_old` 下真实采出的 `mean_p_batch`——梯度因此系统性落后于当前真正需要的修正量。
2. **学习率衰减太慢**：`eta_sgd = 1/(1+t/100)`，第 50 次更新时仍有约 0.667；配合 `log_grad` 被裁剪到 `[-2,2]`，单次更新仍可移动约 `0.667*2.494*2≈3.3 kJ/mol`——不是接近收敛时的小修正，而是持续大幅度来回推。加上每批只有 10 个样本、EMA 严重滞后，容易出现"推过头→验证失败→反向推过头"的振荡，日志里 3/4 次冻结失败正是这个模式。
3. **fixed-H overlap probe 的 PyMBAR 求解结果被浪费**：probe 已经用真实两侧轨迹解出完整 `u_kn` 并跑过 PyMBAR，`compute_free_energy_differences()` 本可以直接给出相邻态 `ΔF`，但代码只取了 overlap matrix，`all_passed=True` 时只是硬停止，没有利用这个已经算出来的、比 SGD 可靠得多的自由能差。

修复（`ibs_engine.py`）：

- `update_weights()` 的 `log_grad` 改用当次真实 `mean_p_batch`，不再读 `self.ema_mean_p`（EMA 保留、仅供诊断趋势观察）。
- 学习率衰减时标从 100 缩短到 30（第 50 次更新时 eta 从 0.667 降到约 0.158）；新增 `IBSSampler.eta_penalty`（初值 1.0），每次冻结验证失败恢复 learning 时调用 `apply_learning_rate_penalty()` 减半（下限 0.05），让重新学习一次比一次保守。
- `update_weights()`/`evaluate_frozen_batch_probability()` 的 `min_buffer_size` 默认从 10 提到 20（连带两处调用方的 `len(energy_buffer)>=10` 检查同步改为 `>=20`），降低单批估计噪声；生产循环里两处纯粹"清空缓冲区避免无谓增长"的 `>=10` 判断（跟 `update_weights()`/验证无关）未改动。
- `_compute_bidirectional_overlap_from_u_kn()` 新增：在已经构建好的 `mbar` 上调用 `_compute_free_energy_result_compatible`/`_extract_free_energy_arrays` 顺带算出 `delta_f_reduced_kT`（`Delta_f[0,1]`，约化单位）；`probe_bidirectional_overlap()` 用自己的 `kt` 换算出 `delta_f_kJ_mol` 一并返回，不新增任何 BAR/MBAR 实现，只是把已经解出的结果暴露出来。
- `run_all_windows`：当最小窗口的 fixed-H 双向 overlap **全部通过**时（`all_passed=True`），不再直接判定失败——用各边 `delta_f_kJ_mol` 累加构造 `f_0=0, f_1=ΔF_01, f_2=ΔF_01+ΔF_12, ...`（去均值）直接写入 Context 校准 `f_k`，然后只给一次"冻结 burn-in + 只读独立验证"的机会（复用 v7 已有的 burn-in/validating 逻辑，不再回退 learning）；通过则 `bias_converged=True` 进生产；仍不通过则视为真实的构象弛豫过慢或偏置表达式问题，交给人工检查，不再自动重试。新状态：`frozen_validation_converged_after_mbar_calibration`；诊断新增 `mbar_calibration`（`delta_f_edges_kJ_mol`/`calibrated_f_k_kJ_mol`/`converged`/`steps_used`）。
- 顺带修复一处日志矛盾：`_create_softcore_force` 打印"CustomNonbondedForce LRC 已禁用"时容易被误读成"完全没有 LJ tail 补偿"，跟后面 `build_ibs_dual_system` 打印的"解析长程色散尾项已启用"看起来矛盾。改写为明确说明：禁用的只是 OpenMM 原生机制（LJ+Coulomb 拼在同一表达式导致原生修正对 Coulomb 发散、实测崩 CUDA），默认 softcore/ACE 路径随后会用手写解析尾项补偿（`potential_type='dexp'` 除外，该路径显式跳过）。不影响本轮收敛失败的根因，只是让日志不再自相矛盾。

`IBS_BIAS_PROTOCOL_VERSION` 7→8（求解器梯度公式、学习率日程、batch size、诊断字段形状均改变，旧协议缓存不安全复用）。`test_warmup_overlap_protocol.py`/`test_audit_protocol_regressions.py` 的版本号字面量同步更新为 v8。

尚未做的：本轮新增逻辑（`update_weights()` 梯度修复、eta_penalty 衰减、MBAR 校准分支）同样没有可重复运行的回归测试（跟第四轮遗留的同一项缺口一样，需要 OpenMM Context mock）；MBAR 校准只在最小窗口（`K<=3`、`stage_type=="vdw"`）路径接入，不影响大窗口的拆分判据。

### 2026-07-15 新增 — 已修复：MBAR 校准冻结验证与 SGD 预热共用同一个已耗尽的步数计数器

第七轮修复（K=4 直接走 fixed-H overlap、不再盲拆）之后，在真实运行里继续观察 K=4 窗口时发现：即便 fixed-H 双向 overlap 全部通过、代码用 BAR/MBAR ΔF 正确校准了 `f_k`，只要 SGD `learning/freeze_burn_in/validating` 三阶段恰好把 `max_bias_warmup_steps`（默认 500000 步）安全帽全部烧光才退出主循环，紧随其后的 MBAR 校准冻结验证就必然被判失败——不是校准本身有问题，是预算记账错误。

根因：`run_all_windows`（`ibs_engine.py`）里两个独立目的的 while 循环共用同一个计数器 `steps_at_full_bias` 和同一个上限 `max_bias_warmup_steps`：主循环 `while steps_at_full_bias < max_bias_warmup_steps`（原约 3587 行）跑 SGD 三阶段；fixed-H overlap 全通过后的 MBAR 校准验证循环 `while steps_at_full_bias < max_bias_warmup_steps`（原约 3819 行）本应独立跑一次"5000 步 burn-in + 连续 batch 验证"，但它检查的是同一个、可能已经被 SGD 阶段耗尽到上限的计数器。SGD 一旦用满预算才退出，校准验证循环的条件从一开始就是假，循环体一次都不会执行，`calibration_converged` 恒为 `False`，`mbar_calibration.converged` 必然报失败，跟校准出的 `f_k` 质量完全无关。

修复（`ibs_engine.py::run_all_windows`）：

- 新增函数参数 `mbar_calibration_reserved_steps: int = 50000`，从总预算 `max_bias_warmup_steps` 里为 MBAR 校准验证单独预留一块步数。
- SGD 三阶段的主循环上限从 `max_bias_warmup_steps` 改为 `sgd_step_budget = max(max_bias_warmup_steps - mbar_calibration_reserved_steps, frozen_burn_in_steps + check_chunk)`（默认即 450000 步），保证无论 SGD 是否用满自己的份额，都会给 MBAR 校准剩下至少 `mbar_calibration_reserved_steps` 步。
- MBAR 校准验证循环不再检查共享的 `steps_at_full_bias < max_bias_warmup_steps`，改为检查独立计数器 `calibration_steps_used < mbar_calibration_reserved_steps`，与 SGD 阶段是否已经耗尽预算完全解耦。
- 诊断新增 `sgd_step_budget`、`mbar_calibration_reserved_steps`（顶层）和 `mbar_calibration.steps_reserved`（校准子字段），便于事后核对某次失败到底是"校准验证真的没通过"还是"预算分配问题"。
- 所有现有调用点（`abfe_pipeline.py`、`runabfe.py`）均未显式传入这个新参数，因此默认值自动生效，无需改动调用链；不影响 `resumed_frozen_f_k`/`bias_converged` 缓存语义，未触发 `IBS_BIAS_PROTOCOL_VERSION` 版本号变更。

**注意**：本次修复之前，任何在 SGD 阶段把 `max_bias_warmup_steps` 用满、随后进入 fixed-H overlap 全通过并尝试 MBAR 校准、但 `mbar_calibration.converged=False` 的 `dual_window_*_warmup_failure.json`，其失败结论都不可信——校准验证循环在旧代码下当时必然是零次执行,应视为"未真正验证"而非"验证未通过"，需要用修复后的代码重跑才能得出可信结论。

状态：**已修复。**

### 2026-07-15 新增 — 已修复：production ESS 修复分支里 fixed-H 探针全通过会一票否决同批次真正失败的窗口，且探针结果只存在内存里

真实运行案例：Stage 2 (vanishing) 一轮修复里 production ESS 低于阈值的窗口是 `[2,6)`、`[5,9)`、`[14,18)`（均为 K=4，按 `min_states_before_split=5` 都不能拆，全部进入 `to_probe`）。fixed-H 双向 overlap 全部通过的只有 `[2,6)`、`[14,18)`；`[5,9)` 至少有一条边未通过。旧代码逻辑：

```python
already_good = [se for se in to_probe if probe_results[se] and all(p.get("passed") for p in probe_results[se])]
if already_good:
    raise RuntimeError(...)  # 整体硬停止
```

只要 `to_probe` 里任何一个窗口 fixed-H 全通过，就把整批 `to_probe`（包括 `[5,9)` 这种真正 fixed-H 失败、理应插 λ 的窗口）一起硬停止——`[5,9)` 的真实缺口从未被处理，也从未插过待重测的 λ。停止本身没有破坏任何已有 production 文件（异常发生在事后诊断阶段，不在采样/清理阶段），但这个"一票否决"混淆了两件不同的事：

1. `[2,6)`/`[14,18)` fixed-H 全通过，只说明 λ 边已经达到 fixed-H overlap 门槛（0.03）——跟 production ESS 门槛（0.05）本来就不是同一个阈值，"通过"只表示自动插点缺少实测证据支持，不能等价于"production ESS 低肯定不是 lambda 问题"；真正原因需要比较该窗口的生产冻结 `f_k` 与 fixed-H BAR/MBAR 累计出的 `f_k`，或核实采样时长/构象弛豫。
2. `[5,9)` 是另一类问题——fixed-H 边确实测出重叠不足，本该照常插 λ 修复，却被前者的"通过"连带一起硬停止。

另外两个附带问题：探针结果（overlap matrix、min overlap、去相关样本数 N、BAR/MBAR ΔF 及其不确定度）此前只存在 `probe_results` 局部字典里，`raise` 一旦触发就随异常一起丢失，磁盘上只留下一行模糊报错，无法事后判断是"阈值边界（如 0.031 险过）"还是"样本数不足"还是"production ESS 计算本身有问题"；且旧代码是把 `to_probe` 全部探针跑完才检查 `already_good`/报错，探针本身很贵（每条边独立 burn-in 5000 步 + 采样 20000 步）。

修复（`abfe_pipeline.py::_run_stage_with_overlap_autorepair`，探针汇总分支，原约 3329 行）：

- 新增 `_persist_fixed_h_overlap_probe_results()`：探针跑完、分类之前立即把每条边的完整结果（`window_range`/`global_edge`/`overlap_matrix`/`min_bidirectional_overlap`/`threshold`/`passed`/`n_k_decorrelated`/`delta_f_kJ_mol`/`delta_f_uncertainty_kJ_mol` 等）追加写入 `{output_dir}/{stage_name}/production_fixed_h_overlap.json`（按修复轮次 `round` 累加，不覆盖历史轮次），保证任何后续分支的 `raise` 都不会再丢失这次昂贵探针的结果；每条边同时打印一行日志，不用打开 JSON 就能看到数值。
- 删除 `already_good` 触发的全局硬停止。改为逐窗口分类：`already_good`（fixed-H 全通过）只记录日志（说明 λ 边已达最低连通标准、production ESS 低的原因需人工比较生产 `f_k` 与 fixed-H 校准 `f_k` 或检查采样时长)，不插 λ、不拆窗；`still_failing`（fixed-H 确有失败边）保留在 `to_probe` 里，按原有逻辑正常插入待重测 λ——同一轮内两类窗口互不阻塞，`[5,9)` 这种真正失败的窗口不再被 `[2,6)`/`[14,18)` 的"通过"连带搁置。
- 只有当本轮 `to_split` 和 `still_failing` 都为空（即除了 `already_good` 之外没有任何可自动处理的窗口）才停止并抛错，报错信息附带 `production_fixed_h_overlap.json` 路径和已打印的逐边数值，供人工判断下一步（MBAR `f_k` 重新冻结验证 vs 延长/换种子采样）。
- 不再有"先把全部探针跑完才检查、找到第一个 all_passed 就整体作废"的早停诉求——旧设计的浪费本质是"结果被扔掉"，不是"跑得太全"；改成逐窗口分类后，每个 `to_probe` 窗口的探针结果都会被实际使用（要么记录诊断，要么驱动插 λ），不存在可以提前丢弃的分支，因此不需要额外的提前终止逻辑。
- 现有 production 采样文件不受影响：这个分支本身发生在事后诊断阶段，`already_good` 窗口既不拆窗也不插 λ，`_invalidate_stage_window_files` 不会触碰它们的能量文件。

状态：**已修复。**

### 2026-07-15 同日复审第二轮 — 上一条修复的三处遗留缺口：状态白名单、真正逐边落盘+缓存复用、all-pass 但 production ESS 低仍无自动处理

上一条修复把"一票否决"改成了逐窗口分类，方向是对的，但复审又发现三处仍未闭环的问题：

**1. 已修复 — MBAR 校准后的收敛状态没进白名单，会被误判成"未确认收敛"而硬停止。** `_run_stage_with_overlap_autorepair` 里核实每个失败窗口 warmup 是否真的收敛时（原约 3367 行），只接受字面量 `status == "frozen_validation_converged"`；但上一节和更早的 `IBS_BIAS_PROTOCOL_VERSION=8` 修复里，fixed-H overlap 全通过后用 BAR/MBAR 校准 `f_k` 再验证通过的窗口，落盘状态是另一个字符串 `frozen_validation_converged_after_mbar_calibration`（`ibs_engine.py` 的 `bias_warmup_diag["status"]`）。以后任何被 MBAR 校准修好的窗口，都会在这个白名单检查处被误判成"没有确认收敛"，直接 `raise` 硬停止，即使它的 production ESS 完全正常也一样会被拦下。修复：改成集合判断，`valid_bias_warmup_statuses = {"frozen_validation_converged", "frozen_validation_converged_after_mbar_calibration"}`，`unvalidated` 用 `status not in valid_bias_warmup_statuses` 过滤。

**2. 已修复 — 探针结果不是真正逐边落盘，且 resume 会把已经测过的探针全部重跑。** 上一条修复的 `_persist_fixed_h_overlap_probe_results()` 先用完整字典推导 `probe_results = {se: probe_window_overlap_fn(...) for se in to_probe}` 把 `to_probe` 里所有窗口的所有边都探完，才整体落盘一次——如果第 9 条边崩溃，前 8 条边（每条边都是独立 5000 步 burn-in + 20000 步采样的真实 GPU 结果）随异常一起丢失，跟"逐边落盘"的初衷不符。另外落盘的结果只按修复轮次累加记录，没有任何内容指纹，跨进程 resume 或下一轮重新触发这条分支时会把所有探针重新算一遍，不管上次是不是已经measure 过完全相同的窗口/λ内容。
修复（`abfe_pipeline.py`）：
  - `_probe_vdw_window_fixed_overlap()` 新增可选参数 `on_edge_done`，每算完一条边立即回调；`_probe_stage2_window_overlap` 闭包同步转发。
  - 新增 `_fixed_h_probe_fingerprint()`：用 `protocol_key` + 该窗口实际 `lambda_vdw` 值（不是会被插点/拆窗改变的 `(start,end)` 索引）+ 探针阈值算一个内容指纹；`_persist_fixed_h_probe_edge()` 按这个指纹把结果写进 `{output_dir}/{stage_name}/production_fixed_h_overlap.json` 的 `windows[fingerprint]` 条目，每条边完成后立即调用一次（不等窗口全部边跑完），记录 `complete`/`rounds_seen`。
  - 主循环里对 `to_probe` 逐窗口检查：先算指纹、查 `_load_fixed_h_probe_cache()`，若已有 `complete=True` 的缓存条目直接复用（打印一行"复用已缓存"日志，不重新采样）；否则才调用 `probe_window_overlap_fn(..., on_edge_done=...)`，每条边算完当场落盘。
  - 删除了原来"整窗算完才落盘一次"的 `_persist_fixed_h_overlap_probe_results()`，改为上面这套指纹缓存 + 逐边持久化的组合。

**3. 已修复 — fixed-H 全通过但 production ESS 低的窗口此前只被记录、从不真正处理。** 上一条修复只是把这类窗口"不插 λ、不拆窗、打一行日志"，如果下一轮它的 production ESS 仍然低、又没有其它真正失败的边或大窗口可处理，会直接撞上"本轮没有可自动处理的窗口"硬停止——相当于遇到 all-pass 但低 ESS 的窗口，流水线除了停止什么也做不了，而最终自由能仍然来自这批低 ESS 的 production 数据，不能假装它已经收敛。
修复（`abfe_pipeline.py::_diagnose_and_repair_all_pass_low_ess_window` + `_invalidate_single_window_production`，均由主循环在 `already_good` 分支里调用）：
  - 用该窗口 fixed-H 探针相邻边的 `delta_f_kJ_mol` 累加、去均值，构造一份独立的 BAR/MBAR 校准 `f_k`（跟 `ibs_engine.py` warmup 阶段自己的 MBAR 校准分支用的是同一套构造）。
  - 从该窗口的 `ibs_state_{stage_type}_window_{idx}.json` 里读出生产阶段真正冻结、驱动了已完成 production 采样的 `f_k`（只有该缓存本身 `bias_converged=True` 时才可信；读不到或态数不匹配则整体判定 `skipped_missing_production_f_k`，不猜测、留给人工检查）。
  - 两者按**相邻边增量**逐边比较（见下方第三轮修复，已从"整段 f_k 绝对值差"改成这个更准确的版本）→ 判定 warmup 学到的偏置本身不准（`recalibrate_f_k`）：用校准出的 `f_k` 覆盖该窗口的 `ibs_state` 缓存（`bias_converged=True`、`n_states`/`prefix`/`ibs_bias_protocol_version`/`lambdas_coul`/`lambdas_vdw` 原样保留，保证 `load_ibs_state()` 的校验能通过），只清空这一个窗口自己的 production 产物（`_invalidate_single_window_production`，不经过整段路径的 λ 内容重新映射，不触碰任何其它窗口），下一轮该窗口会跳过 learning、直接从 fixed f_k 做一次冻结 burn-in + 只读验证 + 生产重采样。差异不大（`reseed_resample`，见下方第三轮修复的改名）→ 判定偏置本身没问题，只清空该窗口的 production 产物强制重采；`_run_dual_lambda_stage` 构造窗口 `Simulation` 时从未显式传 `setVelocitiesToTemperature` 的随机种子，OpenMM 会用系统熵重新随机初始化速度，因此单纯重采就已经是一次独立的"换种子"采样，不需要额外的每窗口步数覆盖机制。
  - 决策和支撑数值（校准 `f_k`、生产 `f_k`、逐边差异、阈值）写入 `{output_dir}/{stage_name}/sampling_repair_decisions.json`（`_persist_sampling_repair_actions`），按轮次累加。
  - 只有当 `to_split`、`still_failing`（真正 fixed-H 失败的窗口）都为空，且 `already_good` 窗口里没有一个真正触发了 `recalibrate_f_k`/`reseed_resample`（即全部落在 `skipped_*`）时，才停止并抛错；只要至少有一个窗口的低 ESS 触发了真实修复动作，就放行本轮继续（该窗口的产物已被清空，下一轮 `run_once()` 会在 `resume=True` 时只重采它，其余窗口复用现有产物）。

另有一处配套调整：`_run_stage_with_overlap_autorepair` 里"失败窗口分散在多个互不相邻区域就整体硬停止"的旧熔断器（原紧跟在状态白名单检查之后）降级为 `self._log()` 警告，不再在逐窗口 fixed-H 分类之前全局拦截——区域是否连续跟"是不是全局协议问题"并非等价，多处分散的失败完全可能是多处独立的局部 λ gap 或多处独立的 f_k/采样问题，都应该、也已经能被下面逐窗口的分类正确处理。

状态：**已修复。**

### 2026-07-15 同日复审第三轮 — 上一轮 sampling repair 的三处收紧：与 split/insert 抢跑的时序 bug、改名 reseed_resample、f_k 比较改成逐边噪声感知阈值

第二轮加的 all-pass sampling repair 本身方向没错，但复审又发现三处需要收紧的问题：

**1. 已修复 — sampling repair 和 split/insert 在同一轮内会互相破坏对方的文件。** `_diagnose_and_repair_all_pass_low_ess_window()` 在判定需要修复时，会直接删掉该窗口的 `dual_window_{idx}_{stage_type}_convergence.json`（`_invalidate_single_window_production`），但同一轮如果 `to_split`/`still_failing` 也非空，紧跟着还会执行 `_invalidate_stage_window_files()`——它判断"某个旧窗口是否可以按 λ 内容重用/重命名到新编号"依赖 `_old_window_accounting_ok(old_idx)`，而这个函数是靠**读取** `convergence.json` 来核对 `WCA_ACCOUNTING_VERSION`/`IBS_BIAS_PROTOCOL_VERSION` 是否匹配的。sampling repair 刚删掉的那个文件，会让这次检查因为文件不存在而直接判"不可信"，导致这个窗口被排除出 `reuse_map`；`_invalidate_stage_window_files()` 的第二阶段清理会把所有"未被认领"的旧编号文件（包括 sampling repair 刚写入的、带校准 `f_k` 的新 `ibs_state` 文件）一并删除——等于自己的两处清理逻辑互相踩踏，白做的 f_k 校准或"仅清空 production 产物"操作会被立刻抹掉。
修复：`already_good` 分支里新增判断 `path_will_change = bool(to_split or still_failing)`。若本轮确实还有需要拆分或插 λ 的窗口，直接跳过这一轮的 sampling repair（只打印一行日志说明"暂缓"），先让路径变更和 `_invalidate_stage_window_files()` 的重映射完整跑完；下一轮这些窗口会被重新诊断为低 ESS，重新计算 fixed-H 探针指纹——由于窗口 λ 内容通常未变，会直接命中已缓存的探针结果（见第二轮的指纹缓存机制），不需要重新采样，然后才安全地执行 sampling repair。只有当本轮 `to_split`/`still_failing` 都为空（λ 路径确定不变）时才在同一轮内直接执行 sampling repair。
另外，即使在"安全执行"的情况下，如果这一轮除了 sampling repair 什么都没做（`to_split`/`to_probe` 全空），主循环也不再往下走到 `_invalidate_stage_window_files()`/重写 `preopt_file` 那一段——λ 路径根本没变，没有东西需要它去"协调"，跑了反而会把 sampling repair 刚清空/改写的文件当"未被认领"再删一遍。这种情况下直接 `continue` 到下一轮重试，只依赖 `_invalidate_single_window_production()` 已经做过的、精确到单个窗口的清理。

**2. 已修复 — `extend_resample` 改名为 `reseed_resample`，避免误导。** 这个分支实际做的事是：删除旧的 production 产物、用相同的步数预算重新采一次——它不会累积样本，也不会真的"延长"采样时长，只是靠 OpenMM 默认不传随机种子（每次新建 `Simulation` 都会用系统熵重新初始化速度）拿到一条独立的新轨迹。原名字 `extend_resample` 容易让人以为它会增加步数或叠加样本，实际没有；改名为 `reseed_resample` 更准确地描述"丢弃旧样本、换一批独立样本"这件事本身。文档字符串里也补充说明：目前没有任何逐窗口步数覆盖机制，如果将来想要真正"延长"（增加该窗口的步数预算）而不只是"换种子"，需要另外给 `run_all_windows`/`IBSWindowManagerDualLambda` 加一个按窗口覆盖步数的参数，这条尚未实现。
所有相关字符串字面量（决策值、日志、`sampling_repair_decisions.json` 里的字段值）同步改名。

**3. 已修复 — f_k 比较阈值从"整段绝对值固定 1.0 kJ/mol"改成"逐边增量 + 探针误差感知阈值"。** 之前的判据是 `max(|f_calibrated - production_f_k|) > 1.0 kJ/mol`（两者都先去均值），固定阈值 1.0 kJ/mol 对真实探针噪声来说太紧：本仓库实测的 fixed-H BAR/MBAR ΔF 不确定度经常在每条边 2-3 kJ/mol 量级，几乎每次比较都会被判定为"差异明显"从而触发 `recalibrate_f_k`，即使生产 `f_k` 其实是对的、只是探针本身噪声大。另外比较"整段绝对 f_k"（哪怕去了均值）不如直接比较"相邻边的增量"直观和规范无关。
修复（`_diagnose_and_repair_all_pass_low_ess_window`）：改成逐边比较 `abs(diff(production_f_k)[i] - probe_delta_f_edges[i])`（`np.diff` 天然对整段可加常数不敏感，不再需要先去均值才能比较），每条边有自己的阈值 `max(f_k_edge_mismatch_floor_kJ_mol=1.0, f_k_edge_mismatch_sigma_multiplier=2.0 × probe_delta_f_sigmas[i])`（后者直接取自 `probe_bidirectional_overlap()` 已经算出的 `delta_f_uncertainty_kJ_mol`）——只要**任意一条边**超出它自己的阈值就判定为 `recalibrate_f_k`，否则 `reseed_resample`。返回结果新增 `production_edge_diffs_kJ_mol`/`probe_delta_f_edges_kJ_mol`/`probe_delta_f_sigmas_kJ_mol`/`edge_abs_diffs_kJ_mol`/`edge_mismatch_thresholds_kJ_mol`/`mismatched_edges`/`max_abs_edge_diff_kJ_mol` 等逐边诊断字段，日志里也会点名具体是哪几条边（按局部索引）触发了判定，而不只是一个笼统的"差异明显"。

状态：**已修复。**

### 2026-07-15 同日复审第四轮 — reseed_resample 真正延长采样、探针缓存支持断点续算、探针不确定度 fail closed

第三轮把 `reseed_resample` 改了名字并说清楚它"只是换一批同样长度的独立样本，不会真正延长"，但复审指出这只是承认问题存在，没有解决问题；另外还发现探针缓存和不确定度校验各有一处需要收紧的地方：

**1. 已修复 — `reseed_resample` 现在会真正延长该窗口的生产步数，而不是原地打转。** 之前删掉旧产物后，下一轮仍按 `n_steps_per_window` 默认步数重采——对偶然坏轨迹（比如恰好陷进一个不典型构象）有效，但对真正的慢弛豫/采样本身太短没有增加任何总信息量，只会消耗一轮 `max_repair_rounds` 额度。
修复：
  - `ibs_engine.py::run_all_windows` 新增 `production_step_overrides: Optional[Dict[int, int]]` 参数（`{window_idx: 该窗口实际生产步数}`），生产采样段的 `n_updates`/`remaining_steps` 改用 `effective_n_steps_per_window`（有覆盖用覆盖值，否则退回默认 `n_steps_per_window`），并把 `n_steps_per_window_default`/`n_steps_per_window_effective` 一并写入该窗口的 `convergence.json`，便于事后核对某个窗口到底用了多少步。
  - 这个参数逐层透传：`abfe_pipeline.py::_run_dual_lambda_stage` 新增同名参数并转发给 `manager.run_all_windows(...)`；`_run_stage2_once` 闭包新增 `_production_step_overrides` 形参并转发；`_run_stage1_once`（decharging，没有 sampling-repair 分支）接受但忽略同一位置的参数，只是为了让 `run_once(...)` 在两条阶段路径下能用同一个调用签名。
  - `_run_stage_with_overlap_autorepair` 新增 `n_steps_per_window` 参数（仅作为"该窗口默认步数基准"使用）和跨轮次持久的 `pending_step_overrides: Dict[window_idx, int]`；每次某个窗口的 `reseed_resample` 触发，就把它的步数覆盖值在当前基础上乘以 `resample_step_growth_factor=2.0`，封顶 `max_resample_step_multiplier=4.0`（避免一个持续不收敛的窗口无界烧 GPU；封顶后仍不收敛就交给 `max_repair_rounds` 熔断或人工检查），写入 `pending_step_overrides[window_idx]`；下一轮 `run_once(...)` 会带上这份覆盖表。`window_idx` 的有效性依赖"设置覆盖值的这一轮"和"消费它的下一轮"之间 λ 路径没变——这正是第三轮已经加的 `path_will_change` 门控保证的前提（reseed_resample 只在纯 sampling-repair 轮触发，不会和 split/insert 同轮发生，因此不会有 window_idx 因插 λ/拆窗而错位的风险）。
  - 未提供 `n_steps_per_window` 基准时（理论上不应发生，仅作防御性兜底），退化为原来的"按默认步数重采一次"并在日志里明确说明"无法计算延长后的步数"，不会静默假装延长了。

**2. 已修复 — fixed-H 探针的部分缓存现在可以真正断点续算，而不是从第一条边重来。** 之前 `_run_stage_with_overlap_autorepair` 的探针缓存查找只在 `complete=True` 时才复用；`complete=False` 的部分结果（比如上次跑到某个窗口第 5/9 条边时进程崩溃，前 5 条边其实已经落盘）会被整体忽略，重新从第 0 条边开始算——不影响最终数值正确性（每条边的探针本身是独立的，用同一份预先算好的 `relaxed_positions`/`relaxed_box` 起跑，不依赖跑的顺序或此前是否跑过其它边），纯粹是浪费已经花掉的 GPU 时间。
修复：`_probe_vdw_window_fixed_overlap()` 新增 `resume_pairs: Optional[List[Dict]]` 参数，传入时用它做 `pairs` 列表的前缀（并按 `n_edges` 防御性截断，防止损坏的缓存条目边数对不上），循环从 `len(resume_pairs)` 开始，跳过已经算过的边，只补算剩余边；`_probe_stage2_window_overlap` 闭包同步新增并转发这个参数。`_run_stage_with_overlap_autorepair` 的探针分支里，`complete=False` 的缓存条目现在会把已有的 `pairs` 作为 `resume_pairs` 传入并打印一行"从已缓存的 N 条边续算"日志，而不是直接丢弃重算。

**3. 已修复 — fixed-H 探针不再只检查 ΔF 本身，也检查它的不确定度是否有限且非负。** `ibs_engine.py::_compute_bidirectional_overlap_from_u_kn`（原约 2842 行）此前只对 `delta_f_reduced`（ΔF 本身）做了 `np.isfinite` 校验，从未检查 `delta_f_uncertainty_reduced`（ΔF 的不确定度）。这个不确定度会被下游 `_diagnose_and_repair_all_pass_low_ess_window` 直接当噪声尺度用在判据 `max(floor, sigma_multiplier × sigma)` 里——如果 `sigma` 是 NaN，这个阈值也会变成 NaN；Python 的 `>` 比较对 NaN 恒为 `False`，会让"任意差异"都被误判成"在噪声阈值内"，把本该判定为 `recalibrate_f_k` 的窗口错误地放行成 `reseed_resample`。负的不确定度同样没有物理意义，说明 PyMBAR 求解或上游数据有问题，也不应该被静默传播。
修复：`_compute_bidirectional_overlap_from_u_kn` 新增校验，`delta_f_uncertainty_reduced` 必须 `np.isfinite` 且 `>= 0.0`，否则直接 `raise RuntimeError`，fail closed，不再把无效不确定度包进返回字典。`abfe_pipeline.py::_diagnose_and_repair_all_pass_low_ess_window` 里也加了第二道防线（防止将来出现绕开上面那处校验的其它 probe 实现路径）：构造 `delta_f_sigmas` 数组后立即检查 `np.isfinite`/`>=0`，任一条边不满足就返回新的 `decision="skipped_invalid_probe_uncertainty"`（不猜测、不参与 `recalibrate_f_k`/`reseed_resample` 判定，只记录 note 供人工检查）。

状态：**已修复。**

### 2026-07-15 同日复审第五轮 — 已修复：双重预平衡会覆盖刚完成的 Boresch 再平衡坐标

`runabfe.py` 主流程的实际执行顺序是：`resolve_boresch_restraint()`（可能内部调用 `pipeline.pre_equilibrate()` 生成无约束轨迹供 Boresch 估算）→（若启用 Boresch 且未 `--skip-rebalance`）`pipeline._rebalance_with_boresch()`（带 Boresch 限制力的再平衡，结果写回 `pipeline.positions`/`box_vectors`）→ `pipeline.run_full_pipeline(..., run_equilibration=not equilibrium_is_done(output_dir) or config.reset, ...)`。第三步的 `run_equilibration` 只要为真，`run_full_pipeline` 内部会再调用一次 `self.pre_equilibrate()`——这是**无约束**的，会直接覆盖第二步刚产出的、带 Boresch 限制力平衡过的坐标。

两种真实会触发这个覆盖的场景（已用代码逐行核实，不是假设的边界情况）：

1. **`--reset`**：`resolve_boresch_restraint()` 里 Boresch 缓存判断和"是否需要预平衡"判断都显式 `and not config.reset`，reset 时一定重新估算+重新预平衡；随后 `run_equilibration = not equilibrium_is_done(output_dir) or config.reset` 里的 `or config.reset` 恒使其为真，不管磁盘上轨迹是否有效。确定存在重复。
2. **外部 Boresch 来源（`--boresch-source traditional/orb_ml`）+ 全新运行**：`resolve_boresch_restraint()` 对这两种来源直接读取外部参数文件返回，**从不**调用 `pre_equilibrate()`，也就从不写 `pre_equilibration.dcd`；随后 `_rebalance_with_boresch()` 仍然正常执行（从未经预平衡的原始坐标开始做限制力再平衡）；到第三步时 `equilibrium_is_done(output_dir)` 因为文件不存在而恒为 `False`，`run_equilibration` 因此恒为真——不需要 `--reset` 也必然触发：先做完 Boresch 再平衡，紧接着又跑一次无约束预平衡，把刚做完的限制力平衡坐标覆盖掉。

`simple`/`auto`/`fluctuation` 来源的普通全新运行（非 reset）通常不重复：`resolve_boresch_restraint()` 在这条路径下确实会调用 `pre_equilibrate()` 写出轨迹，使得第三步的 `equilibrium_is_done()` 变为真、`run_equilibration` 变为假，符合预期不重跑；但这依赖"第三步的判断恰好读到第一步刚写的文件"这个隐式假设，一旦文件条件有任何出入（例如 `.dcd` 大小阈值边界、checkpoint 缺失）就可能失效，属于脆弱的隐式正确，而不是设计上的保证。

修复思路：把"预平衡是否已经在本进程做过"收敛成显式状态，而不是继续用磁盘文件是否存在这种间接、对场景 1/2 都不可靠的信号：

- `ABFEPipeline.__init__` 新增两个仅进程内有效的标记：`self._pre_equilibration_done_this_process`、`self._boresch_rebalance_done_this_process`（均初始为 `False`）。
- `pre_equilibrate()` 成功完成后置位第一个；`_rebalance_with_boresch()` 的两个 return 分支（真正跑完的分支、以及 resume 时"状态已完成、跳过"的分支）都置位第二个——两种情况下 `self.positions` 都已经是可信的、带 Boresch 限制力平衡过的坐标。
- `run_full_pipeline()` 内部原来的 `if run_equilibration:` 改为 `if self._boresch_rebalance_done_this_process: 跳过（记日志） elif self._pre_equilibration_done_this_process: 跳过（记日志） elif run_equilibration: ...原有逻辑... else: ...原有 else...`——这两个进程内标记比调用方传入的 `run_equilibration`（一个基于磁盘/`config.reset` 推算出来、不知道本进程内部已经做过什么的值）更可信，因此在这里短路，而不是要求调用方把 `run_equilibration` 算对。`--reset`/`config.reset` 本身没有被简化或删除——reset 时该做的预平衡/Boresch 重估算仍然照常执行，只是执行完之后不会被 `run_full_pipeline` 自己的内部逻辑重复一遍。

另外按建议给 equilibration 缓存加了 system/config 指纹，堵上"同一个 `--output` 目录换了 gro/top/ligand/温度、没加 `--reset`，被静默当成已完成复用旧轨迹"的口子：

- 新增 `_pre_equilibration_fingerprint(system, ligand_indices, temperature)`（`abfe_pipeline.py`）：对 system XML 做哈希（复用已有的 `_system_xml_hash`）+ 配体原子索引 + 温度。刻意不含目标步数——`run_full_pipeline` 内部调用 `pre_equilibrate(resume=resume)` 时从不显式传 `n_steps`（永远用方法默认值，跟 `config.n_equil_steps` 无关），把步数纳入指纹会在合法配置下产生误报；且同一 system/ligand/温度、只是步数不同的历史预平衡，仍是物理上有效的起点，不属于这个指纹要拦截的"配置不匹配"。
- `pre_equilibrate()` 成功后把这个指纹写入 `{output_dir}/pre_equilibration_fingerprint.json`。
- `runabfe.py::equilibrium_is_done()` 新增可选参数 `expected_fingerprint`：不传时保持原来纯文件存在性判断（向后兼容）；传入时还要求指纹文件存在且与当前重新计算的指纹一致，否则视为未完成、强制重新预平衡。三处调用点（`resolve_boresch_restraint` 内、复合物腿 `run_full_pipeline` 调用、溶剂腿 `run_full_pipeline` 调用）均已改为传入用当前 `pipeline.system`/`pipeline.ligand_indices`/`pipeline.temperature` 现算的指纹。

状态：**已修复。**

### 2026-07-15 同日复审第六轮 — 已修复：`enable_early_stop` 曾是纯空转开关，现已接入真正的在线判据（默认仍关闭，阈值待离线回放校准）

`enable_early_stop` 此前在调用链上一路被接受、一路被转发（`run_full_pipeline` → `_run_stage1_once`/`_run_stage2_once` 闭包 → `_run_dual_lambda_stage`），但 `_run_dual_lambda_stage` 里 `manager.run_all_windows(...)` 那次调用从未把它带上，`ibs_engine.py::run_all_windows()` 本身也从未定义这个参数——打开这个开关在实际运行里没有任何效果，纯粹是个空壳。

修复思路采用保守在线判据，而不是简单接上一个"跑到某个比例就停"的粗糙开关：

- 新增 `EARLY_STOP_PROTOCOL_VERSION = 1`（`ibs_engine.py`，紧邻 `WCA_ACCOUNTING_VERSION`/`IBS_BIAS_PROTOCOL_VERSION` 定义）。**默认仍是关闭的（`enable_early_stop=False`）**，且判据里的阈值默认值目前只是工程上合理的起点，尚未用已有的完整轨迹做离线回放（"在第 N 步用当时的样本停下来，比较跟完整跑满的结果差多少"）验证过，在离线回放通过、确定合适默认值之前，不建议在生产配置里显式打开这个开关。
- 新增独立函数 `_solve_single_window_local_mbar()`：把 `GlobalMBARAnalyzer.solve_stage_integrated()` 内部单窗口的完整局部 MBAR 构造（自相关去相关子采样 → 全局能量偏移 → 采样态+物理态增广矩阵 → kT 约化 → 逐列数值稳定化 → MBAR 求解 → `compute_effective_sample_number()` 有效样本重叠诊断）原样复刻成一个独立实现，而不是重构抽取共享辅助函数——`solve_stage_integrated` 已经过多轮审计验证，为了 DRY 而重构它、冒着回归已验证行为的风险划不来；两个实现今天算法相同，但服务的问题不同（阶段收尾汇总 vs. 生产中途"现在能不能停"），允许各自独立演化。
- `run_all_windows()` 新增 `enable_early_stop`（默认 `False`）及八个 `early_stop_*` 阈值参数：`min_steps`（最小生产步数门槛，达到之前不检查）、`check_interval_steps`（多久做一次 local MBAR）、`required_consecutive_passes`（默认 3，需要连续多少个独立 block 全通过才停）、`min_ess_ratio`/`min_absolute_ess`/`min_decorrelated_samples`（三种 ESS 度量）、`max_delta_g_drift_kJ_mol`（相邻两次检查的局部 ΔG 漂移上限）、`max_uncertainty_kJ_mol`（局部 ΔG 端点合并不确定度上限）。生产循环里每满一个 check interval，用该窗口至今累积的样本跑一次 `_solve_single_window_local_mbar`，五项同时检查；任一项不通过（含第一次检查——没有"上一次"可比，漂移判据直接判不通过，因此至少要连续 `required_consecutive_passes+1` 次检查才可能真正停止）就把连续通过计数清零；连续达到要求次数才真正跳出生产循环。`n_steps_per_window`（含 `production_step_overrides` 覆盖值）始终是硬上限，early stop 只会提前结束，不会超过它；触发时跳过"余数补齐"（那是为凑满整步数设计的，跟提前停止的意图矛盾）。
- `convergence.json` 新增字段：`actual_production_steps`（真实跑了多少步；未提前停止时等于目标步数，含余数补齐）、`early_stop_enabled`、`early_stop_triggered`、`early_stop_stop_reason`、`early_stop_protocol_version`、`early_stop_check_history`（每次检查的 ESS/局部 ΔG/不确定度/漂移和每项判据是否通过）、`early_stop_config`（本次调用实际使用的八个阈值，供事后核对"当时到底是按什么标准停的"）。
- resume 时的窗口级缓存校验（原来只查 λ 值/`WCA_ACCOUNTING_VERSION`/`IBS_BIAS_PROTOCOL_VERSION` 是否匹配）新增第四项 `early_stop_ok` 检查：如果缓存的 `early_stop_triggered=True`（这份能量是提前停止产出的短样本），只有当前调用**同时**满足"确实启用了 early stop"、"`early_stop_protocol_version` 与当前一致"、"当前目标步数没有高于缓存产出时记录的目标步数（`n_steps_per_window_effective`）"这三条，才允许复用；任何一条不满足都视为无效缓存，强制重新采样该窗口——不会出现"当初 early stop 提前停在 8 万步，后来关掉 early stop 或把预算提到 40 万步，却因为文件存在就被当成已经采够了"的情况。未触发过 early stop 的旧缓存不受这条约束影响。
- `abfe_pipeline.py::_run_dual_lambda_stage` 里 `manager.run_all_windows(...)` 调用新增转发 `enable_early_stop` 和上述八个阈值（阈值通过 `kwargs.get(...)` 读取，同样的保守默认值），这是唯一被这次修复触碰的调用点——`_run_stage1_once`/`_run_stage2_once`（串行路径）和 `_run_stage_worker_process`（`--parallel-stages` 子进程路径）都统一走 `_run_dual_lambda_stage`，因此两条路径都被这一处修复覆盖，不需要分别改。`_refine_lambda_path_with_medium_probe`/`_run_shadow_ibs_decharging_leg` 里另外两处 `manager.run_all_windows(...)` 调用（λ 路径中等探针、实验性 shadow-Coulomb 路径）未改动，不在本次修复范围内。

状态：**已修复（功能已真正接入且默认关闭；阈值默认值仍待离线轨迹回放校准，校准通过前不建议在生产中打开）。**

### 2026-07-15 同日复审第七轮 — 四个 release 阻塞级修复：外部 Boresch 跳过基线预平衡、rebalance resume 没加载再平衡坐标且无指纹、early-stop ΔG 不确定度算法错误、early-stop 缓存复用检查不完整

前几轮的"双重预平衡""early-stop"修复本身方向没错，但复审发现新引入或遗留了四处会实际影响数值/流程正确性的问题，逐一修复：

**1. 已修复 — 外部 Boresch 来源（`traditional`/`orb_ml`）的全新运行会完全跳过基线预平衡。** 第五轮的双重预平衡修复本身是对的（`_boresch_rebalance_done_this_process`/`_pre_equilibration_done_this_process` 短路 `run_full_pipeline` 内部的预平衡块），但没有处理另一半问题：`resolve_boresch_restraint()` 对 `traditional`/`orb_ml` 直接读外部参数文件后立即 `return`，从来不会触发 `pipeline.pre_equilibrate()`；紧接着 `_rebalance_with_boresch()` 仍然正常执行 50,000 步 Boresch 限制力再平衡——真实流程因此变成"原始（未平衡）坐标 → 50k Boresch rebalance → IBS"，而不是"基线预平衡一次 → Boresch 参数生成/加载 → 带 Boresch rebalance"。
修复：把基线预平衡的触发逻辑（`equilibrium_is_done()`/指纹缓存判断 + 需要时调用 `pipeline.pre_equilibrate()`）从"只有 auto/simple/fluctuation 才会走到"的位置，挪到 `resolve_boresch_restraint()` 函数最开头、`source` 分支判断之前，无条件先执行一次（复用原有的 resume/指纹缓存逻辑，不会重复真正跑预平衡）。这样 `traditional`/`orb_ml` 也会先有一次基线预平衡，再进入外部参数读取分支返回；`runabfe.py` 里 `pipeline._rebalance_with_boresch()`（约 2108 行附近）之后再执行时，`self.positions` 已经是正确的平衡态坐标。

**2. 已修复 — `_rebalance_with_boresch()` 检测到"已完成"时直接返回错误坐标，且再平衡缓存没有指纹校验。** 第五轮修复双重预平衡时，把这里的早退分支改成了"检测到 completed 状态就直接 `return self.positions/self.box_vectors`"，并在注释里错误断言"此时 self.positions 就是已经带 Boresch 限制力平衡过的坐标"——这个断言是错的：`self.positions` 在这个时间点是 pipeline 构造时加载的坐标（通常来自基线预平衡轨迹的最后一帧），根本不是 `rebalance.chk`/`rebalance_traj.dcd` 里真正做完 Boresch 限制力再平衡的那一帧。等于把"还没做完 Boresch 限制力再平衡"的坐标当成"已经做完"的结果直接返回给下游 IBS 采样。
修复：删除这个提前 `return`，只保留一条提示性日志；让执行自然流向下面已有的续跑逻辑——`resume_enabled=True` + `simulation.loadCheckpoint(chk_path)` + `steps_remaining=max(0, n_steps-currentStep)`，如果确实已经做完，`steps_remaining<=0` 时不会再步进，但仍会从刚加载的 checkpoint 里正确提取出真正完成再平衡的坐标/盒子——这条路径本来就是对的，本轮之前的问题只是不必要地在它前面加了一段用错坐标的"快捷方式"。
另外新增 `_rebalance_fingerprint(system, boresch_params)`：对 system XML 哈希 + Boresch 锚点原子索引 + 平衡值 + 力常数做指纹。`rebalance_state.json` 之前只记录 `status`/`n_steps`，没有任何内容校验——`rebalance.chk` 是绑定到具体 System（含注入的 `LambdaDependentBoreschForce` 参数）的二进制状态，如果 Boresch 锚点、平衡值或力常数变了（重新估算过、或本方法内部的动态 r0 校正给出了不同值），加载旧 checkpoint 不会报错，只会从一个从未在当前限制力下平衡过的状态继续。现在 `rebalance_state.json` 写入时带上这个指纹；resume 时先比对指纹，不一致就把 `rebalance_cache_trusted` 置为 `False`——不仅不再触发提前返回的提示日志，连接下来 `append_traj`/`loadCheckpoint` 这两处真正消费旧文件的地方也一并跳过，强制从当前坐标做一次完整的全新再平衡。

**3. 已修复 — early-stop 在线判据里局部 ΔG 端点不确定度的合并方式在统计上是错的。** `run_all_windows` 生产循环里，局部 ΔG 的不确定度之前算的是 `sqrt(df_endpoint_first**2 + df_endpoint_last**2)`——把窗口内第一个和最后一个物理态各自相对采样态的边际不确定度当成两个独立量，用平方和开方合并。但这两个边际估计来自**同一次** MBAR 拟合、同一批样本，彼此存在协方差，独立量假设不成立，合并结果可能系统性偏大也可能偏小，取决于协方差符号——一个方向的错误会让判据过早通过（提前停止用了不够格的样本），另一个方向的错误会让判据实际上永远无法满足（局部 ΔG 波动被系统性高估，early stop 名存实亡）。
修复：`_solve_single_window_local_mbar()` 新增返回字段 `endpoint_diff_uncertainty_kJ_mol`，直接读增广 MBAR 矩阵里第一个物理态和最后一个物理态之间的**成对**不确定度 `ddf_matrix[1, len(win_lams)] * kt`（物理态在增广矩阵里占第 1..K 行，第 0 行是采样分布），而不是合并两个边际值。`run_all_windows` 的在线检查改用这个字段。

**4. 已修复 — early-stop 缓存的 resume 校验仍不完整，两类窗口可能被错误复用。** 第六轮加的 resume 校验（`enable_early_stop`/`early_stop_protocol_version`/目标步数）本身方向对，但漏了两种情况：(a) 只比对了协议版本，没有比对缓存记录的 `early_stop_config`（那八个具体阈值）——协议版本只在判据逻辑本身变了才会变，单纯把某个阈值调紧（例如 `max_uncertainty_kJ_mol` 从 5.0 收紧到 1.0）完全不影响协议版本号，但在旧的、更松阈值下提前停止通过的窗口，不能保证在新阈值下也一定能通过，之前的校验会直接放行复用；(b) 完全没有触发过 early stop 的"完整"缓存，没有检查其记录的目标步数是否满足当前调用的目标——把预算从 250k 提到 500k（不涉及 early stop 开关本身），旧的、确实跑满了 250k 的缓存仍会被当成满足新的 500k 目标而复用。
修复：新增模块级 `_early_stop_configs_match(cached_cfg, current_cfg)`（浮点用 `np.isclose` 容忍序列化误差，其余精确比较）。resume 校验逻辑重排为两层：第一层对**所有**缓存（不论是否触发过 early stop）都要求 `cached_conv["n_steps_per_window_effective"] >= 当前有效目标步数`，不满足直接判定缓存无效；只有通过这一层，才进入第二层——仅当 `early_stop_triggered=True` 时才需要额外核对 `enable_early_stop`/`early_stop_protocol_version`/`early_stop_config` 三项，任一不满足同样判定无效。

另外补上一处第五轮遗漏的字段：**`_pre_equilibration_fingerprint()` 之前没有包含压力（barostat pressure）**，只有 system XML 哈希 + 配体索引 + 温度——换了 `--pressure`（不同目标密度的 NPT 系综）但复用同一个 `--output` 目录、不加 `--reset`，会被静默当成"配置未变"复用旧预平衡轨迹。已在指纹里加入 `pressure_bar`（转换为 bar 并四舍五入），`pre_equilibrate()` 写指纹和 `runabfe.py` 三处调用点（`resolve_boresch_restraint` 内、复合物腿/溶剂腿 `run_full_pipeline` 调用）均已同步传入 `pipeline.pressure`/`pipeline_solv.pressure`。

状态：**已修复。**

**仍未完成 —— 窗口初始化冗余（第一块性能问题）未处理。** 上一轮提出的"每个 IBS 窗口第一个窗口完整 bootstrap、后续重叠窗口从相邻已完成窗口继承 positions/box + fixed-H 几何/力检查通过即跳过 16 档 Boresch ramp + 缩短最小化/dt 测试为短验证、任何检查失败自动回退完整流程"的两级启动协议，本轮**完全没有实现**：每个新窗口仍然执行 20,000 次最小化、10,900 步超细测试阶梯、16 档 Boresch ramp、15,000 步 timestep ramp、bias ramp 等完整固定开销（`ibs_engine.py` 约 3350/3391/3464 行附近），没有任何 previous-window 坐标传递或跳过逻辑。这是本次复审列出的三项里唯一完全未动工的一项，规模也明显大于另外两项（涉及改动每窗口 bootstrap 的核心物理逻辑 + 新增自动回退判据），需要单独作为后续工作展开，不应被当前这批 P1 修复的完成状态掩盖。

### 2026-07-15 同日复审第八轮 — 已修复：LJ 长程尾项修正未补 1.0-1.2nm switching 区间被削弱的能量（当前最重要的物理 release blocker）

`_create_softcore_force`/`BeutlerSoftcoreBuilder.build` 构造的软核 `CustomNonbondedForce` 都在 1.0-1.2nm 启用 `setUseSwitchingFunction`——真实模拟的能量在这段区间被 OpenMM 的五次多项式 `S(r)` 从满强度逐渐削到 0，1.2nm 之外再硬截断。但手算的解析 LRC（`ibs_engine.py` 原约 1677 行）只补了 `1.2nm → ∞` 的标准 `r^-6` 尾项，把 `1.0-1.2nm` 里被 switching 削掉的那部分能量完全当作不存在处理；同时它把 λ<1 时的 softcore 分母 `D(r) = alpha_lj*(1-λ)^m_lj + r^6` 当成纯 `r^6` 处理，忽略了 softcore 修改对尾区间积分的影响；而且只补了吸引项 `r^-6`，没有排斥项 `r^-12` 的尾贡献。三者叠加意味着这个手算修正跟实际采样的哈密顿量之间，本身就存在一个从未被积分覆盖的能量差——这是本项目当前最重要的未闭合物理缺口。另外，传统 Beutler REMD 路径（`ibs_engine.py` 原约 6564 行）用的是同一套过时逻辑（标量 `prefactor * lambda_vdw^power / V`），没有一并修。

修复：按 OpenMM 官方 `CustomNonbondedForce` LRC 文档给出的完整积分形式重新实现——

```
E_corr = (2π/V) ∫_{r_switch}^{r_cutoff} E(r)[1-S(r)] r² dr + (2π/V) ∫_{r_cutoff}^∞ E(r) r² dr
```

- 新增 `_lj_tail_correction_moments_kj_nm6_nm12()`（取代原 `_lj_tail_correction_S_kj_nm6`，只算吸引矩 S6）：同时算出吸引矩 `S6 = Σ eps_ij·sigma_ij^6` 和排斥矩 `S12 = Σ eps_ij·sigma_ij^12`（配体-环境原子对，混合规则同软核表达式）。
- 新增 `_lj_switching_function_value()`：OpenMM 标准五次多项式 `S(r)=1-10x³+15x⁴-6x⁵`。
- 新增 `_lj_softcore_tail_radial_integrals(lambda_vdw, alpha_lj, m_lj, ...)`：对每个 λ_vdw 数值积分（`scipy.integrate.quad`，只在系统/窗口构建时算一次，不进每帧热路径）真实的 `I6`/`I12`——`D(r)=alpha_lj*(1-λ)^m_lj+r^6` 在 switching 区间和 cutoff 之外两段分别积分再相加，同时覆盖 switching 削弱和 softcore 修改。之所以能把 `S6`/`S12`（跟具体原子对相关的几何量）和 `I6`/`I12`（跟 λ 相关的径向积分）分开相乘，是因为 ACE 软核表达式和 Beutler 软核表达式的 `D(r)` 都不含任何原子对相关的参数（`alpha_lj`/`m_lj` 是软核力的全局标量，不是逐对参数）——这是这套软核势的一个精确性质，不是近似简化。
- 新增 `_lj_tail_lrc_coefficients_kj_mol(lambdas_vdw, S6, S12, alpha_lj, m_lj, n_lj, ...)`：逐 λ 组装 `coeff[k] = 16π · λ_vdw[k]^n_lj · (S12·I12(λ_vdw[k]) - S6·I6(λ_vdw[k]))`；`λ_vdw=0` 时直接跳过积分、系数恒为精确的 `0.0`（不是"数值上接近 0"）。每帧实际修正 = `coeff[k] / V(t)`。这套系数在 `λ=1`（`(1-1)^m_lj=0`，`D(r)` 退化成纯 `r^6`）时精确退化为标准 12-6 LJ 的 switching-aware 尾项修正，可独立验证；在旧的"只补 cutoff 外 r^-6"极限下（`I12→0`、`I6` 只算 `cutoff→∞`），`16π·(-S6·I6)` 精确约化为原有已用真实 GPU 数据校验过的 `-(16π/3)·S6/rc³` 公式，说明新公式是旧公式的严格推广而不是另起炉灶。
- **三处能量入口现在共用同一个 `ibs_wrapper.lj_tail_lrc_coeff_kj_mol` 系数数组**：`IBSSampler._lj_tail_correction_kj_mol()`（生产采样）、`probe_bidirectional_overlap()` 的 fixed-H overlap 探针，以及 `TraditionalMBARAnalyzer.compute_u_kn()` 里传统 Beutler 路径的离线重算（用 `BeutlerSoftcoreBuilder.build()` 在本项目所有调用点实际使用的默认参数 `alpha_lj=0.5, power_lj=1` 构造，跟真正采样用的软核力保持一致）——不再是三份可能互相漂移的独立实现。原来的 `lj_tail_prefactor_kj_mol`/`lj_tail_lambda_vdw_pow`（标量+幂律，无法表达非线性的 switching+softcore 积分）和传统路径的 `lj_tail_prefactor_kj_nm3_mol`/`lj_tail_lambda_vdw_power`/`_traditional_lj_tail_energy_kj_mol()` 均已删除。
- `TRADITIONAL_LJ_LRC_PROTOCOL_VERSION` 1→2（尽管名字里写着 traditional，这个常量从一开始就同时覆盖 ACE/dual_lambda 和传统路径的 LRC 公式版本，未改名以保持向后兼容的 import）。dual_lambda 每窗口 `convergence.json` **新增** `lj_tail_lrc_protocol_version` 字段（此前这条路径完全没有任何 LRC 版本门控字段，v1 时代的窗口能量会被无条件复用）；resume 逐窗口校验（`lrc_version_match`）和 `abfe_pipeline.py::_old_window_accounting_ok`（λ 路径变更后的窗口内容复用判断）均已接入这个字段，版本不匹配一律 fail closed、强制重新采样，不会让 v1（无 switching 修正）的旧能量继续被当作 v2 结果使用。

新增/改写测试（`test_audit_protocol_regressions.py`）：
- `SwitchingAwareLJTailLRCTests.test_lambda_zero_gives_exact_zero_lrc`：λ=0 时系数恒为精确 `0.0`。
- `test_lambda_one_matches_independent_plain_lj_switching_integral`：λ=1 时与一份完全独立内联实现（不复用项目任何辅助函数）的标准 12-6 LJ switching 数值积分结果一致（`rel_tol=1e-6`）。
- `test_switching_aware_result_differs_from_old_cutoff_only_r6`：新公式结果与旧 v1 的 `-(16π/3)·S6/rc³` 公式在 `rel_tol=1e-3` 下确认不同。
- `test_softcore_denominator_changes_integrals_away_from_lambda_one`：λ<1 时软核位移确实让 `I6`/`I12` 相对 λ=1 变小。
- `SourceContractTests.test_lrc_consumers_share_one_coefficient_array`：源码层面核实三个消费点读的是同一个 `lj_tail_lrc_coeff_kj_mol` 属性/同一个构造函数（不是各自维护一份公式）。
- `SourceContractTests.test_resume_rejects_mismatched_lrc_protocol_version`：核实 resume 校验和窗口复用判断都接了 `lj_tail_lrc_protocol_version` 门控。
- 原来依赖已删除函数 `_traditional_lj_tail_energy_kj_mol` 的两个旧测试已随之改写/移除。

状态：**已修复。**

### 2026-07-15 同日复审第九轮 — 已修复：fixed-H 探针用错 ensemble/能量定义去校准生产 f_k（两处独立发生的同一类 bug）

fixed-H 探针原本只有一种实现（`probe_bidirectional_overlap`）：动力学是 `U_common + CV_k`（不含 Group 4 WCA 防护壳），能量含 LRC（跟 `IBSSampler.collect_energies()` 里喂 MBAR 的 `target_energies`口径一致）。这跟生产实际需要 `f_k` 复现的自由能——`U_common + WCA_window(lambda_shield) + CV_k` 动力学、纯 softcore CV 能量（不含 LRC，LRC 只在离线喂 MBAR 时才加）——是两个不同的物理量：WCA 只在同帧同窗口的态间能量*差*里抵消，不会从被采样的构象系综里消失；LRC 从不进偏置力本身。用前者的 `delta_f_kJ_mol` 去校准/覆盖后者定义的 `f_k`，是把两个不等价的自由能混为一谈。

发现时机：warmup 阶段的 MBAR 校准分支（`ibs_engine.py::run_all_windows`）直接拿 `probe_bidirectional_overlap` 的 `delta_f_kJ_mol` 累积成 `f_calibrated` 覆盖 `f_k`，且用 `overlap>0.03` 当作"ΔF 已经够精确"的证据（真实案例 `n_k_decorrelated=[39, 9]`、`overlap=0.105`：最低连通性达标，但去相关样本数远不足以支撑把 ΔF 当校准目标）；校准失败时打印的 `ema_mean_p[k]` 还是校准前 SGD 阶段的旧值，诊断信息不可靠。

修复（warmup 侧）：
- 新增 `_serialize_ibs_common_plus_wca_system()`：只剥离 Group 1（CV 偏置），保留 Group 4 WCA，断言剥离结果确有 Group 4 力；`build_ibs_dual_system()` 新增 `ibs_wrapper._common_plus_wca_system_xml`，与原有只保留 path-overlap 用的 `_common_system_xml`（剥离 Group 1+4）并列,不混用。
- 新增 `probe_bidirectional_overlap_for_bias_calibration()`：动力学用 `_common_plus_wca_system_xml` + 生产同一个 `lambda_shield`（`mean(lambdas_vdw_in_window)`），能量只用 `IBSSampler.evaluate_interaction_energies()` 的纯 softcore CV（不加 LRC）；`min_decorrelated_samples`/`max_delta_f_uncertainty_kJ_mol` 任一不达标就返回 `None`（不是"凑合返回"），调用方须延长采样重试，绝不接受不够精确的估计。返回字段刻意命名为 `delta_f_bias_kJ_mol`/`delta_f_bias_uncertainty_kJ_mol`，跟 `probe_bidirectional_overlap` 的 `delta_f_kJ_mol` 区分开，杜绝以后再被静默混用。
- `run_all_windows` 里原来直接复用 path-overlap 探针结果做校准的代码，改为在 path-overlap 全通过后单独跑一轮 `probe_bidirectional_overlap_for_bias_calibration`（每条边最多 3 次延长采样重试，采样步数逐次翻倍），只用它的 `delta_f_bias_kJ_mol` 校准 `f_k`；任一边最终仍不达标就放弃本次校准（不覆盖 `f_k`），交给人工检查。
- 修复校准失败分支打印 stale `ema_mean_p` 的 bug：现在会先用 `last_validation_batch_p` 刷新 `ema_mean_p_values`/`min_ema_p`/`max_ema_p`/`coverage_ess` 再打印。
- `IBS_BIAS_PROTOCOL_VERSION` 8→10（9 是这次改动前的中间值）。

复审时顺着同一处 `probe_bidirectional_overlap` 的另一个调用点（`abfe_pipeline.py::_probe_vdw_window_fixed_overlap`，供 production-ESS 自动修复用）核查，发现 `_diagnose_and_repair_all_pass_low_ess_window()` 存在完全相同的 bug：fixed-H 全通过、production ESS 仍低的窗口，用同一个 path-overlap 探针的 `delta_f_kJ_mol` 跟生产冻结的 `f_k` 逐边比较，判定不匹配（`decision="recalibrate_f_k"`）时直接把累积出来的 `f_calibrated` 写回 `ibs_state_*.json` 的 `f_k`——跟 warmup 侧是同一个"错 ensemble + 错能量定义"问题，只是发生在 production 修复路径而不是 warmup。这是本轮独立发现、之前未被审计覆盖的实例，同一次修复：

- `_probe_vdw_window_fixed_overlap()`：path-overlap 探针每条边通过后，额外用 `ibs_wrap._common_plus_wca_system_xml` + `mean(lv_win)` 跑一次 `probe_bidirectional_overlap_for_bias_calibration()`（同样最多 3 次延长采样重试），把 `delta_f_bias_kJ_mol`/`delta_f_bias_uncertainty_kJ_mol`/`bias_calibration_sufficient` 附加进每条边的 `pair` dict；path-overlap 本身未通过的边不跑校准子探针（`bias_calibration_sufficient=None`，跟"跑了但不够精确"的 `False` 区分）。
- `_diagnose_and_repair_all_pass_low_ess_window()`：改读 `delta_f_bias_kJ_mol`/`delta_f_bias_uncertainty_kJ_mol`（不再是 `delta_f_kJ_mol`/`delta_f_uncertainty_kJ_mol`）；新增前置校验，只要有一条边的 `bias_calibration_sufficient` 不是 `True` 就整体拒绝比较/覆盖（`decision="skipped_insufficient_bias_calibration"`），因为 recalibrate/reseed 两个分支都需要对每条边的独立 ΔF 有信心。返回 dict 里的字段相应改名为 `probe_delta_f_bias_edges_kJ_mol`/`probe_delta_f_bias_sigmas_kJ_mol`，避免与旧字段混淆。
- 缓存失效：`_fixed_h_probe_fingerprint` 的 payload 里已经包含 `protocol_key`（内含 `code_sha256` 和 `ibs_bias_protocol_version`），本次改动本身就改变了源码字节和 `IBS_BIAS_PROTOCOL_VERSION`，因此旧的 `production_fixed_h_overlap.json` 缓存（不含 `bias_calibration_sufficient` 等新字段）会自动因指纹不匹配被判定为 cache miss、重新计算，不需要额外的版本号改动。

状态：**已修复。**

### 已确认修复

- `IBS_BIAS_PROTOCOL_VERSION` 已升级至 8，`THERMODYNAMIC_PATH_PROTOCOL_VERSION` 已升级至 6（当前最新值；均见各自定义处的完整版本历史注释）；旧偏置/路径语义的缓存会因版本变化失效。路径协议 v3 新增了 production ESS 失败时的“先拆窗、最小窗再 fixed-H probe、实测失败才插 λ”语义；v4 给 `partition_windows_by_thermodynamic_length` 补上了 `max_states_per_window`（默认 6）硬上限；v5 修复了批量拆窗产生的冗余嵌套窗口；v6 修复了 split 后未拆邻窗重叠比例失衡问题（见下方“第六轮”条目）。IBS 偏置协议 v7→v8 是本节最重要的一次修复（见下方“P0：假收敛”“f_k 求解器不稳定”条目）。`test_warmup_overlap_protocol.py`/`test_audit_protocol_regressions.py` 里的版本号字面量已同步更新为当前值。
- **2026-07-14 同日复审再新增 — 已修复：批量拆窗产生冗余嵌套窗口**：`_run_stage_with_overlap_autorepair` 的 production ESS `to_split` 分支（`abfe_pipeline.py` 约 3250 行）此前对每个失败的父窗口独立调用 `split_window_from_warmup_failure`，没有做任何跨窗口去重/归并。IBS 窗口本身按设计相互重叠，独立拆分多个重叠父窗口会让某个子窗口整个落在相邻父窗口的子窗口内部——真实案例：18 态、5 个重叠 6 态父窗口 `(0,6),(3,9),(6,12),(9,15),(12,18)` 全部触发拆分后，产出 10 个窗口，其中 `(3,6)⊂(2,6)`、`(6,9)⊂(5,9)`、`(9,12)⊂(8,12)`、`(12,15)⊂(11,15)` 四个严格冗余；正确的最小连通链应是 `(0,3),(2,6),(5,9),(8,12),(11,15),(14,18)`（6 个窗口，相邻恰好共享一个态）。覆盖范围本身没错（冗余窗口不会漏态），但会重复采样浪费 GPU，且下一轮可能继续拆分冗余窗口导致窗口数量失控增长。修复：新增 `canonicalize_window_ranges()`（`abfe_preoptimizer.py`）——去重、剔除被其他窗口严格包含的窗口，再校验完整覆盖、相邻窗口至少共享一个态、以及归约后确实不再有嵌套窗口；批量拆分循环结束后立即调用一次，落盘前再调用一次兜底（覆盖 split+insert 组合的情况）。已用真实案例的 10 窗口输入手动核算，归约结果与上述 6 窗口最小链完全一致；新增 `test_canonicalize_window_ranges_removes_nested_windows_from_batch_split`/`test_canonicalize_window_ranges_rejects_incomplete_coverage` 两个回归测试（`test_warmup_overlap_protocol.py`）。

- **2026-07-14 同日复审第三轮新增 — 已修复：P0 假收敛（IBS_BIAS_PROTOCOL_VERSION 6→7）**：v6 的“连续 N 次通过”判据本身是假的。真实落盘数据实测（5 个窗口）：`update_weights()` 在同一次调用里先用 `f_old` 算出 `mean_p_batch`/更新 `ema_mean_p`，再立刻算出 `f_new` 并写回 `context.setParameter`（`ibs_engine.py` 原 `update_weights()`）——真正冻结进生产的那个 `f_k`（即最后一次调用产出的 `f_new`），从未被任何一次“通过”检验过；所谓“连续 3 次通过”检验的是 3 个不同、都已经被丢弃的旧 `f_k`，不是同一个冻结 Hamiltonian 下的 3 次独立验证。另外两处加重了污染：(a) `self.gamma=0.9` 的 EMA 保留 90% 历史，反应严重滞后；(b) `bias_scale` 从 0.5/0.7 爬到 1.0 时从未重置 `sampler.ema_mean_p`，导致 full-bias 判据混进了 0.5/0.7 这两个不同 Hamiltonian 下算出的旧统计。真实案例：5 个窗口的最后一次真实 batch 概率没有一个满足当前逐态门槛（如 `[0.738, 0.262, 2.2e-7]` vs 目标 `1/3`），却全部被写成 `converged`。
  修复为两阶段状态机（`ibs_engine.py::run_all_windows` 的 bias_scale=1.0 判据段，`IBSSampler` 新增只读方法 `evaluate_frozen_batch_probability()`）：**learning**（沿用旧 `update_weights()` + 连续通过判据，达到后只是“候选收敛”，不宣布 `converged`）→ **freeze_burn_in**（冻结 `f_k` 快照，此后到验证结束前绝不再调用 `update_weights()`，丢弃 5000 步 burn-in 重新平衡到冻结 Hamiltonian 下的分布，重置 EMA）→ **validating**（用 `evaluate_frozen_batch_probability()`——只读，从不写 `context.setParameter`——在完全固定的 `f_k` 下连续采若干新 batch，用同一套 `min(p_k)>0.5/K` 且 `coverage_ESS>0.8K` 门槛验证；连续通过才真正宣布 `bias_converged=True`；任何一次验证失败立刻恢复 `learning`，而不是死抠同一个已经证明不稳的 `f_k` 反复重测）。诊断新增 `validation_pass_count`/`learning_to_validation_cycles`/`frozen_burn_in_steps`/`frozen_f_k_at_last_freeze`/`final_mode` 字段；`status` 字段由 `converged` 改为 `frozen_validation_converged`，避免与旧协议的假收敛状态混淆。

- **同轮新增 — 已修复：P0 收敛后“生产前卸压”把刚验证过的分布毁掉**：即使上面的假收敛问题不存在，旧代码在宣布 `bias_converged=True`（或续传时判定已收敛）后，**无条件**执行：`bias_scale=0`→`minimizeEnergy`→5000 步无偏置动力学→`bias_scale` 重新爬坡 `0.25→1.0`→直接进入生产采样（`ibs_engine.py` 原“生产前卸压”块，紧跟在收敛判定之后，不在 `if/else` 分支内，对续传的已收敛窗口同样无条件执行）。这段“卸压”会实际改变构象分布，但重新爬坡后从未对这个新构型重新验证覆盖度/ESS，等于把刚证明稳定的分布替换成一个从未验证过的新分布，再直接送进生产。
  修复：整段“卸压→最小化→无偏置运行→重新爬坡”已删除。冻结验证（见上条）已经是在完全固定的 production Hamiltonian（`bias_scale=1.0`、`f_k` 不再变化）下完成的，验证通过后不再改变 `bias_scale`、不再最小化、不再重新爬坡；只做一次非破坏性的安全力检查（`max|F|` 超过 7000 kJ/(mol·nm) 直接 `raise`，不再尝试用“应急深度最小化+局部松弛+重新爬坡”去掩盖它），然后原样进入生产采样。

- **同轮新增 — 已修复：P0 production ESS 低时未区分“全局协议问题”与“局部 λ 密度问题”**：真实案例里全部 5 个窗口同时报告 production ESS 低于阈值，根因就是上面两个假收敛/卸压 bug（几乎每个窗口的偏置在进入生产时都不是真正冻结/验证过的），但 `_run_stage_with_overlap_autorepair` 此前把每个低 ESS 窗口无差别地丢给 `to_split`/`to_probe`，相当于把一个全局采样协议问题误判成局部 λ 密度不足，继续拆窗/插点只会在坏协议上反复浪费 GPU。~~初版修复只按失败窗口占比 `> 50%` 硬停止~~ **2026-07-14 第四轮复审指出该初版过于粗糙**：既没有真正读取每个失败窗口自己的 `bias_warmup.status`（只是假设"进了 production 就等于 converged 属实"），也没考虑总窗口数很少时一个局部坏边可能同时污染两个重叠窗口、让占比假阳性超过 50%。改为两步更精确的判据（`abfe_pipeline.py` 新增 `_load_window_bias_warmup_status`/`_merge_overlapping_ranges_into_components`）：(1) 先逐一核实每个失败窗口的 `convergence.json` 里 `bias_warmup.status` 是否确实是 `frozen_validation_converged`，不确认就直接硬停止；(2) 把失败窗口按全局 λ 区间重叠关系分组——IBS 相邻窗口按设计一定重叠，同一段连续失败的窗口会被分进同一个 connected component；只有失败窗口分散在多个互不相邻区域时才判定为全局协议问题硬停止，单一 component（哪怕包含全部窗口）仍按局部问题继续走拆窗/probe。

- **2026-07-14 第四轮复审新增 — 已修复：P0 冻结验证判据仍读 EMA，不是真正独立的 batch**：`IBSSampler.evaluate_frozen_batch_probability()` 已经正确返回当次真实、未经平滑的 `mean_p_batch`，但 `run_all_windows` 的 validating 分支调用后把返回值丢了，转而读 `sampler.ema_mean_p`（`gamma=0.9` 的滞后 EMA）来判断 pass/fail——相当于把 v7 本来要修的"用旧统计量冒充新证据"的 bug，原样复刻进了刚写好的验证阶段。数值示例：第一批均匀 `[1/3,1/3,1/3]`，第二批塌缩到 `[1,0,0]`，`EMA=0.9*[1/3,1/3,1/3]+0.1*[1,0,0]=[0.40,0.30,0.30]` 对 K=3 仍能轻松通过 `min(p)>0.1667` 和 coverage 门，第三批继续塌缩也可能通过，于是再次假收敛。修复：判据直接用 `evaluate_frozen_batch_probability()` 的返回值（`None` 时 `continue`，不计入任何判定），不再读 EMA；EMA 只保留在 `IBSSampler` 内部供其他诊断使用。诊断新增 `validation_batch_probabilities`（记录每一个真实 validation batch 的原始概率数组，供事后审计三次独立验证的真实证据，而不是只看一个被平滑过的数字）。
- **同轮新增 — 已修复：P1 更新次数上限会在恰好触发候选收敛时把 freeze/validate 整个跳过**：循环条件原为 `bias_update_count < max_bias_updates and steps_at_full_bias < max_bias_warmup_steps`；如果恰好在第 `max_bias_updates` 次更新形成候选收敛、`mode` 切到 `freeze_burn_in`，下一轮循环会因为 `count < max_bias_updates` 为假直接整体退出，burn-in/validation 一步都不会执行。修复：改为 `while steps_at_full_bias < max_bias_warmup_steps: if mode=="learning" and bias_update_count>=max_bias_updates: break; ...`——更新次数上限只约束 `learning`，`freeze_burn_in`/`validating` 不消耗、也不受这个额度限制，只受总步数安全帽 `max_bias_warmup_steps` 约束。
- **同轮新增 — 已修复：P1 resume 直接跳过冻结 burn-in 与验证**：`bias_converged=True` 的缓存此前会让代码整段跳过预热块，只做一次力检查就进 production——缓存只能证明这个 `f_k` 曾经在旧构型下有效，不能证明本次新建 Context 的当前构型（重新最小化/测试步进/Boresch 爬坡后的构型，并非原来的平衡构型，且代码本就没有保存/恢复旧窗口的最终 positions）已经在这个固定偏置下平衡过。修复：resume 且已收敛时不再整段跳过，改为跳过 `learning`（不重新调整已恢复的 `f_k`），直接从 `freeze_burn_in` 开始对这份恢复的 `f_k` 重新做一次冻结 burn-in + 只读验证；验证通过才放行进生产，验证失败则和全新窗口一样自动回退到 `learning`。诊断新增 `resumed_from_cache` 字段。
- full-bias warmup 只在真正发生一次新的 `update_weights()` 后评估收敛，未更新的检查轮次不再重复累计“连续通过”。
- `|Δf_k| < 0.05 kJ/mol` 已从硬性收敛门移除，只保留为诊断量。当前硬门为 `min(p_k) > 0.5/K` 且 participation coverage ESS `1/sum(p_k^2) > 0.8K`，并要求连续多次真实更新通过——**注意**：v7 之前这一条描述的就是最终收敛判据本身；v7 起这只是 learning 阶段的“候选收敛”信号，真正的 `bias_converged=True` 还需要冻结 `f_k` 后在只读的 `evaluate_frozen_batch_probability()` 下再次连续通过，见上方 P0 假收敛条目。
- 大窗口 warmup coverage 失败时不再使用 evolving-bias 混合分布下的 `std(beta·Δu)` 冒充固定 λ 热力学度量，也不再做算术中点插值和伪造 `L/2 + L/2` 边长；当前只拆分窗口，两个子窗口共享一个已有 λ 态，不新增 λ。
- 三态及以下的最小窗口仍失败时，才运行真正的 fixed-H 双向重叠探针：动力学 Hamiltonian 为 `U_common + cv_k_int`，不含 IBS `CustomCVForce` 和 WCA sampling bias；分别在相邻固定态采样，通过已有能量 probe 组装两态 `u_kn`，使用 PyMBAR `compute_overlap()`，门控 `min(O_01, O_10) >= 0.03`。
- `U_common` 不是从原始完整相互作用 `system_template` 直接克隆（那会双算 ligand–environment）；当前从已组装 VDW-IBS 窗口克隆并严格移除顶层 Group 1/4 后导出，结构方向正确。
- 只有 fixed-H 双向 overlap 实测失败的相邻边才允许插入一个待重新测量的 λ；不再从 warmup `Δu` 曲线直接驱动 λ 向端点连续二分聚集。
- 拆窗/插点纯索引逻辑复测通过；主 Python 文件通过语法编译。当前本地 Windows Python 缺少 OpenMM/PyMBAR/pytest，GPU fixed-H/MBAR 动态回归仍须在计算环境执行。

### 2026-07-14 同日复审第三轮新增 — 仍未修复的四项（本轮未处理，按原样记录）

以下四项在提出 P0 假收敛问题的同一轮复审中一并指出，均未在本轮处理：

1. **P1：pilot 仍是单轮粗网格**。`g(λ)` 在端点附近可能急降（如 `31121 → 17383 → 513`），但只在原始线性网格上测一次，插出新 λ 后从不重新测量（`abfe_preoptimizer.py` 约 1465 行 `redistribute_lambda_by_thermodynamic_length` 的调用方）。应做迭代 pilot：重新测已优化 λ 上的 `g(λ)`，直到实测热力学边长大致均匀，而不是信任第一轮线性网格的插值结果。
2. **P1：production 收敛门只看 ESS ratio**。当前只要求 `N_eff/N >= 0.05`（`ibs_engine.py` 约 4279 行），样本量很大时可能因 ratio 偏低而误拆；去相关后样本量很少时又可能 ratio 通过但绝对 `N_eff` 只有个位数。应同时检查绝对 `N_eff`、ratio 与自由能不确定度三者，而不是单一 ratio 阈值。
3. **P2：IBS 偏置的解析形式与更新公式不完全一致**。OpenMM 侧 `CustomCVForce` 对 logit 用 `80*tanh(x/80)` 做平滑饱和（`ibs_engine.py` 约 2192 行），但 `update_weights()`/`evaluate_frozen_batch_probability()` 用的是未饱和的原始 logit。能量跨度较大时，两边代表的混合分布不完全相同——采样用的是饱和版本，权重学习/冻结验证评估的是未饱和版本。
4. **P2：`canonicalize_window_ranges()` 不感知窗口状态**。v5 的归约逻辑只按坐标包含关系去重，不知道哪个候选窗口已经跑过/通过、哪个还需要重跑；混合成功窗口与失败窗口做批量拆分归约时，理论上可能删掉一个已经通过的小窗口、保留一个仍需重采的大窗口。长期应改成带采样状态的全局 connected-component planner，而不是纯几何去重。

第四轮复审重申了上述 1-3 条（原样未变，本轮仍未处理），并新增一项本轮同样未处理：

5. **P2：v7 两阶段冻结验证状态机没有任何回归测试**。`learning -> freeze_burn_in -> validating` 的模式切换、`resumed_frozen_f_k` 直接从 `freeze_burn_in` 起步、验证失败回退 `learning` 并清空 EMA 等分支，目前都只有本轮人工核对（读代码 + 手工过一遍状态转移），没有可重复运行的测试锁定。由于 `IBSSampler`/`run_all_windows` 深度依赖真实 OpenMM Context，写这类测试需要先做一个轻量 mock/fixture（例如假 `context.getParameter`/`setParameter` 加一个可编程的假 `energy_buffer` 生成器），比之前纯 Python 数组逻辑的测试（如 `canonicalize_window_ranges`）工作量更大，本轮未做。

### 仍未闭环 — P1：preopt resume 信任条件不一致

位置：`abfe_pipeline.py` 的 Stage 1/Stage 2 preopt cache loader（当前约 3598-3602、3705-3712 行）。

当前有两个相反方向的问题：

1. **同态数缓存过度信任**：Stage 1 在 `len(cached_lambdas) == stage1_states` 时、Stage 2 在路径版本匹配且 `len(cached_lambdas) == stage2_states` 时，会绕过 `cached_protocol == current_protocol` 检查。于是状态数相同但 system、参数、代码或采样协议不同的旧 preopt 路径仍可能被复用，跟“完整协议指纹 fail closed”的目标不一致。
2. ~~**fixed-H/production 插点缓存过度拒绝**：态数发生变化时，loader 只认可 `provenance.source == "auto_repair_by_overlap"`；但新 warmup 修复会写出 `fixed_hamiltonian_bidirectional_overlap`，新 production ESS 修复会写出 `production_overlap_repair_split_then_probe`。因此 fixed-H 实测失败后插入的新态数可以在当前进程继续使用，却会在跨进程 resume 时被当成非验证缓存丢弃，重新退回初始路径。纯拆窗不改变态数，通常会碰巧通过“同态数”分支，但这不能覆盖随后真正插点的情况。~~ **2026-07-14 修复**：Stage 2 preopt loader 的 `is_verified_auto_repair` 已改为检查 `cached_source in {"auto_repair_by_overlap", "fixed_hamiltonian_bidirectional_overlap", "production_overlap_repair_split_then_probe"}`（`abfe_pipeline.py` 约 3966-3972 行），态数变化的 fixed-H/production 插点缓存现在能跨进程正确恢复。纯拆窗不改变 λ 数量，走的是既有“同态数”分支，不受影响。

建议：任何 preopt 缓存都必须先满足 `cached_protocol == current_protocol`；态数变化时，再要求 `provenance.source` 属于明确的受信白名单（已实现，见上）。**第 1 点（同态数缓存绕过协议指纹检查）仍未修复**——需要新增“同态数但协议不匹配必须拒绝”和“fixed-H/production 插点后跨进程恢复成功”两类回归测试。

状态：**部分修复。第 2 点（provenance 白名单）已闭环；第 1 点（同态数缓存过度信任、绕过协议指纹检查）仍未修复，仍影响跨配置缓存隔离。**

### 2026-07-14 同日复审新增 — P1：旧全局 `min_overlap` 熔断器会提前截断 production split/probe 状态机

位置：`abfe_pipeline.py::_run_stage_with_overlap_autorepair`，当前约 `3166-3179` 行；新的 VDW production repair 分支从约 `3194` 行才开始。

当前 production ESS 修复的主体方向已经改对：

- `plan_vdw_overlap_repair_targets()` 不再按单态 ESS 噪声直接挑边插点；低 ESS 且态数 `K>=4` 的窗口只进入 `to_split`。
- 拆窗通过 `split_window_from_warmup_failure()` 完成，不改变 λ 数量，两个子窗口只共享一个已有 λ 态。
- `K<4` 的最小窗口才进入真实 fixed-H 双向 overlap probe；probe 使用 `U_common + 单个 cv_int` 的 NVT 固定 Hamiltonian，并由 PyMBAR 的 `min(O_ij,O_ji) >= 0.03` 门控。
- 只有 probe 实测失败的边才调用 `insert_lambda_from_overlap_failure()`；每轮至多插一个待重新测量的 λ，旧热力学边长被作废，不再伪造 `L/2 + L/2`。

但是旧熔断器仍在新的 split/probe 分类之前执行：只要本轮 stage-wide `min_overlap <= previous_min_overlap`，代码就直接抛错。这与新状态机冲突，原因有两类：

1. 第一次 repair 可能只是把 `K=6/7` 大窗口拆成 `K=3/4` 子窗口。拆窗的目的本来就是让下一轮继续拆 `K=4` 或让 `K=2/3` 进入 fixed-H probe，并不保证第一次拆窗后全局最差 ESS 立刻严格上升；若打平或因噪声变差，当前代码会在 probe 之前终止。
2. 多个窗口同时失败时，即使 fixed-H probe 证实并修复了其中一个真实缺口，stage-wide 全局最小值仍可能由另一个尚未处理的窗口控制，因此“全局最小值未上升”不能证明刚才的局部修复无效。

建议：VDW 且 `probe_window_overlap_fn is not None` 的新分支不要使用这个旧的 stage-wide 熔断器；依靠 `max_repair_rounds`、路径确实发生变化的检查以及 fixed-H probe 的 fail-closed 判据控制循环。旧熔断器最多只保留给没有 fixed-H probe 的 legacy/non-VDW 分支。如果仍需要熔断，应按“被修复的具体窗口/具体边”比较，而不是比较跨不同窗口划分的全局最小值。

**2026-07-14 修复**：`min_overlap <= previous_min_overlap` 熔断判定已改为 `if probe_window_overlap_fn is None and (...)`，只在没有 fixed-H probe 的 legacy/coul 分支生效；VDW 的新 split/probe 分支不再受这个跨窗口全局比较的影响，只由 `max_repair_rounds` 和 probe 自身的 fail-closed 判据（`already_good`/`missing` 立即 raise）控制循环。

状态：**已修复。**

### 2026-07-14 同日复审新增 — 已修复：热力学距离切窗缺少状态数硬上限

位置：`abfe_preoptimizer.py::partition_windows_by_thermodynamic_length`（当前约 92 行）、调用方 `optimize_stage2_vanishing`（约 1405 行）、`abfe_config.json` 的 `pilot_max_window_thermodynamic_length`/`pilot_overlap_thermodynamic_length`（约 33-34 行）。

问题：改成按累积热力学距离切窗之后，切窗循环只检查 `cumulative[end+1]-cumulative[start] <= max_window_length`，从未检查过窗口跨了多少个态。真实一次运行里 pilot 给出总长度 `14.488`、17 条边均摊约 `0.852`，`max_window_thermodynamic_length=6.0` 时 `6.0/0.852≈7` 条边（8 态）都不会超预算；配合 `overlap_thermodynamic_length=1.5`（约 2 条边）算出步进 5 态，实测切出 `(0,8),(5,13),(10,18)`——三个 8 态窗口喂给同一个 IBS bias。这些窗口后续大概率会因 warmup coverage 失败被 `split_window_from_warmup_failure()` 拆开，等于先烧一轮采样去证明窗口太大，纯属浪费。

修复：`partition_windows_by_thermodynamic_length` 新增 `max_states_per_window`（默认 6，对齐之前固定 `pts_per_window=6` 的约定），在增长循环里作为跟距离预算并列的独立停止条件（`end+1 <= state_cap_end`），不是切完之后再截断；并显式断言 `max_states_per_window >= min_states_per_window`，避免这两个上下界互相矛盾。`optimize_stage2_vanishing`/`_run_dual_lambda_optimization` 新增同名参数并写入 `path_diagnostics`/日志；`abfe_config.json` 新增 `pilot_max_states_per_window: 6`，经 `runabfe.py` 两处调用点透传。用与真实运行相同的数字（17×0.852、max=6.0、overlap=1.5）手动核算：新逻辑切出 `(0,6),(3,9),(7,13),(11,17),(15,18)`（5 个窗口，均为 6 态，末尾余 3 态），不再出现 8 态窗口。

状态：**已修复。**

### 仍未闭环 — P1：解析 LJ LRC 未包含 switching 区间

位置：`ibs_engine.py::_create_softcore_force`、`abfe_core.py::BeutlerSoftcoreBuilder.build` 及两条路径的解析 LRC 计算。

当前 ACE/IBS 和传统 Beutler 软核力均启用了 `1.0–1.2 nm` switching，但解析 LRC 只补了 cutoff `1.2 nm → ∞` 的渐近 `r^-6` 尾项，没有补 switching function 在 `1.0–1.2 nm` 内削掉的吸引能。现有 `test_lrc_interaction_group_compat.py` 的 Q1/Q2/Q3 都显式 `setUseSwitchingFunction(False)`，因此不能验证生产 Hamiltonian 的这一部分。

用纯 `r^-6` 渐近项和 OpenMM 标准五次 switching 多项式做数量级核对：`1.0–1.2 nm` 被 switching 削掉的积分约为 `1.2 nm → ∞` cutoff-tail 积分的 30.8%。真实 softcore 分母会改变具体比例，但足以说明该项不能在没有验证的情况下视为零。当前修正方向正确（避免混合 Coulomb 表达式的原生 LRC 发散），数值上仍属于不完整的 tail/switch correction。

建议二选一：

- 将 LJ 和 Coulomb 拆成独立 CustomNonbondedForce，只对 LJ-only 力使用经过 GPU 验证的原生 LRC；或
- 对每个目标态按真实 softcore 表达式和实际 switching function 数值积分 `1-S(r)` 区间，再与 cutoff 外尾项一起加入，并增加 switching-on 的合成体系基准测试。

**2026-07-15 已修复**（见上方"同日复审第八轮"）：采用了第二个方案——`_lj_tail_lrc_coefficients_kj_mol()` 现在对每个 λ_vdw 数值积分真实的 `(1-S(r))` switching 区间 + cutoff 外尾项，且用真实的 softcore 分母 `D(r)=alpha_lj*(1-λ)^m_lj+r^6`（不是纯 `r^6`），同时补上了排斥项 `r^-12` 的尾贡献（之前只有吸引项 `r^-6`）；dual_lambda 和 traditional/Beutler 两条路径共用同一套构造。`test_lrc_interaction_group_compat.py` 的 Q1/Q2/Q3（显式 `setUseSwitchingFunction(False)`）仍只验证不带 switching 的原生 LRC 兼容性问题，跟这里的手算解析修正是两回事，不因这次修复而过时；新的数值正确性测试在 `test_audit_protocol_regressions.py::SwitchingAwareLJTailLRCTests`。

状态：**已修复。**

### 仍需加固 — P2：fixed-H 探针 GPU Context 峰值

位置：`ibs_engine.py::probe_bidirectional_overlap`（当前约 2810-2812 行）。

当前在主 IBS `Simulation/Context` 仍存活时，同时创建并保留 `fixed_i`、`fixed_j` 两个完整体系 Context，之后又创建一个能量 probe Context。大型显式溶剂体系可能在最需要诊断的失败路径上触发 GPU OOM；异常虽会被记录并阻止盲目插点，但会让 fixed-H 判据本身无法完成。

建议顺序运行 fixed_i/fixed_j：采完一侧帧后显式释放其 `Simulation/Context`，再创建另一侧；能量 probe 也可在动力学 Context 释放后统一评估保存的帧，以降低峰值显存。

状态：**不改变成功运行时的数值定义，但属于实际生产鲁棒性缺口。**

### 测试与文档同步问题

- ~~`test_audit_protocol_regressions.py` 仍硬编码查找 `IBS_BIAS_PROTOCOL_VERSION = 5`，当前源码已是 6；该断言必须同步更新，否则完整测试环境会出现假失败。~~ **2026-07-14 已同步**：断言改为 `IBS_BIAS_PROTOCOL_VERSION = 6`。
- ~~`test_warmup_overlap_protocol.py::test_protocol_versions_reject_old_semantics` 仍断言 `THERMODYNAMIC_PATH_PROTOCOL_VERSION == 2`，而当前源码已升级为 3；安装依赖后该测试必然失败。~~ **2026-07-14 已同步**：断言改为 `== 4`（路径协议同日又因 `max_states_per_window` 硬上限升到 v4，见「已确认修复」）。
- 当前测试没有覆盖 production ESS repair 的关键状态转换：应新增“大窗口低 ESS 只拆窗且 λ 数量不变”“拆窗后即使全局 min_overlap 未严格改善仍可继续到最小窗 probe”“最小窗 fixed-H 全通过时拒绝插点”“实测失败时只插一个 λ”“v3/v4 新 provenance 跨进程 resume 可恢复”五类回归测试；仍是缺口，本轮只同步了版本号字面量，未新增测试用例。
- `test_warmup_overlap_protocol.py` 的 `U_common + cv_i` 测试是合成单粒子/force-group 测试，能锁定“移除 Group 1/4 后只加一个 CV”的结构语义，但尚未覆盖真实 `build_ibs_dual_system` 的 softcore exclusions、LRC 和直接能量一致性；建议补一份小型周期体系集成测试。
- 本文下方历史段落仍包含 `IBS_BIAS_PROTOCOL_VERSION=2/3/5` 及“连续通过尚未修复”等旧时点描述；这些记录不删除，但当前状态以本节的 version 6 结论为准。

### 2026-07-14 总结

warmup 的原始根因——把 SGD `Δf` 步幅当收敛、用 evolving-bias `Δu` 伪装热力学度量、算术二分并伪造子边长——已经实质修正。production ESS repair 的新主体也已改成拆窗 → 最小窗 fixed-H 双向 overlap → 实测失败才插点，Hamiltonian 记账方向成立；但旧 stage-wide `min_overlap` 熔断器仍可能在下一轮分类/probe 前提前终止，因此这条 production 状态机尚未完全闭环。发布前至少应修复该熔断器与 preopt resume 的新 provenance 白名单，并补齐 switching-aware LRC；fixed-H Context 峰值和过期测试属于随后应完成的加固项。

**LJ tail/LRC 更新**：VDW/vanishing 腿缺 LJ 长程色散 tail correction 这条（原「最高优先级」第 3 条）已针对默认 ACE/`dual_lambda` 路径修复，详见该条目全文；`single_lambda`/REMD（`BeutlerSoftcoreBuilder`）路径仍缺同等修正，见「建议修复优先级」。修复前先用独立脚本 `test_lrc_interaction_group_compat.py` 在真实 GPU 节点上实测了 OpenMM 原生 `setUseLongRangeCorrection`，确认它配合 `addInteractionGroup` 的组合表达式（LJ+Coulomb 拼在同一个 `CustomNonbondedForce` 里）一旦遇到真实非零电荷会让 CUDA 后端直接崩溃（发散积分），因此改为手算解析尾项，未使用 OpenMM 内建机制。解析公式本身已在 GPU 合成体系上核对；生产接线仍缺一份在**当前 IBS 冻结协议**下完成的端到端结果。

**同日更新（APBS correction helper）**：`apbs_correction.py` 已从 v2 的"PB 连续介质溶剂化能三体差分"整体重写为 Rocklin/Wu-Biggin 膜体系有限尺寸静电修正（RIP 方法），跟 LJ tail/LRC 仍是完全独立的两个物理项，详见「APBS correction helper 当前状态」。

**2026-07-13 更新（WCA 记账 + IBS 偏置冻结协议）**：审计发现 VDW/vanishing 阶段的 λ-WCA 防护壳（Group 4）被当成"λ 无关"物理量算进 `base_energy`，但其 `lambda_shield` 实际按每个采样窗口的平均 λ_vdw 设定，导致重叠 λ 态在不同窗口里携带不同的 WCA 取值却被当同一物理态拼接——是真正的 Hamiltonian 不一致，不只是效率问题。WCA 记账已修复（Group4 定义为纯采样偏置，`e_base={0,2,3,5}`、`e_bias={1,4}`，并以 `WCA_ACCOUNTING_VERSION=2` 门控缓存）；同时修复了 LRC 误入 `f_k` 训练目标、base 读取失败静默归零。

随后 `output_lrc_fix/` 的真实运行暴露出另一个 P0：多个窗口的 `bias_warmup` 已明确达到步数上限但未收敛，代码仍放行进入生产并在生产期继续更新 `f_k`；MBAR 却把这些随时间变化的偏置当成一个固定 sampled state。低 ESS 因而被误判为 λ 密度不足，出现 `18→19→20→21→22→23` 以及新一轮 `18→22→25` 的反复加点，`min_overlap` 总体反而下降。当前代码已引入 `IBS_BIAS_PROTOCOL_VERSION=2`：未收敛预热直接报错、生产期冻结 `f_k`、旧协议窗口/IBS state/stage/final 缓存 fail closed、加密后 overlap 不改善立即熔断。**但严格收敛判据仍有一个未修 P0：连续通过次数会在没有发生新 `update_weights()` 时重复使用同一批 `ema_mean_p/f_k` 递增，可能产生假收敛；见「建议修复优先级」。** 因此旧 `output_lrc_fix/` Stage2 结果只作为故障证据，不是当前协议的数值验证结果。

---

## 当前结论

在当前主路线下，热力学循环符号和核心物理组装已经核实：

```text
Delta G_bind = Delta G_solvent - Delta G_complex + Delta G_APBS
```

其中 `Delta G_complex` 已包含复合物腿 decoupling、约束/Jacobian 项和 Boresch 标准态释放项；`Delta G_solvent` 使用溶剂腿 total 结果；`Delta G_APBS` 只有显式传入 `--apbs-correction-kj-mol` 时才加入。

没有发现新的“当前主链会静默把 Delta G_bind 符号算反或重复加减 Boresch”的 bug。剩余最高风险集中在四类：

1. 自相关/MBAR overlap 指标已修复，但 IBS 偏置的“连续通过”实现仍可能重复计算同一批统计量，当前还不能把 `bias_converged=True` 无条件当作可靠结论。
2. ~~VDW/vanishing 腿缺少 LJ long-range dispersion/tail correction。~~ ~~默认 ACE/`dual_lambda` 路径已修复（手算解析尾项，未走 OpenMM 内建 LRC），`single_lambda`/REMD 路径仍缺，checkpoint/resume 版本门控也还没加。~~ **2026-07-15 已修复**：两条路径现在都用同一套 switching+softcore-aware 解析尾项（见「最高优先级」第 3 条最新状态和"同日复审第八轮"），`lj_tail_lrc_protocol_version` 门控已接入 dual_lambda 每窗口 `convergence.json` 的 resume 校验。
3. 自动加密 λ 路径仍不能跨进程 resume 保留；预优化缓存也没有完整协议指纹。
4. 若启用当前不常用路径，存在若干潜在静默错误或旧文件误读风险。

---

## 最高优先级：影响当前结果可信度

### 1. ✅ 已修复 — 缺少自相关子采样，误差棒系统性偏小

位置：`ibs_engine.py` 中 `TraditionalMBARAnalyzer.solve` 和 `GlobalMBARAnalyzer.solve_stage_integrated`；共享 helper `abfe_core.py::subsample_series_by_autocorrelation`。

原问题：之前把每一帧原始 MD 轨迹当作独立样本喂给 MBAR。`n_k` / `N_eff` 会把相关帧当作不相关帧计数，导致报告的 `error` / `ddf` 偏小。已有 statistical inefficiency 计算只作为诊断，不真正用于下采样，而且计算对象也不是每个状态的时间序列。

修复内容：

- 新增 `subsample_series_by_autocorrelation(series)`（`abfe_core.py`）：用 `pymbar.timeseries.statistical_inefficiency` 估计单一状态能量序列的统计非效率 `g`，再用 `subsample_correlated_data` 取出近似独立的帧索引；样本过少（<20）或 pymbar 不可用时原样返回全部帧（`g=1.0`），不强行子采样。
- `TraditionalMBARAnalyzer.solve()`：按 `n_k` 分块，对每个态自己的能量时间序列做去相关子采样后再建 MBAR（新增 `decorrelate: bool = True` 参数，默认开启）；原来对"跨状态 pooled 列做差"的无意义 `statistical_inefficiency` 诊断已删除，替换为每态真实的 `g` 值（`diagnostics.decorrelation`）。
- `GlobalMBARAnalyzer.solve_stage_integrated()`：对每个窗口的采样分布能量序列（`base+bias`）做同样的去相关子采样，再建局部 MBAR。
- 已用合成 AR(1) 相关序列验证：子采样后误差棒变大（更保守、更真实），帧数按 `g` 比例下降；iid 序列 `g≈1`（几乎不子采样）。

状态：已修复。

### 2. ✅ 已修复 — `GlobalMBARAnalyzer` 的收敛/重叠指标不是真正的 MBAR overlap

位置：`ibs_engine.py` 中 `GlobalMBARAnalyzer.solve_stage_integrated`、`TraditionalMBARAnalyzer.solve`；门控点在 `abfe_pipeline.py::ABFEPipeline._assert_stage_result_sane`。

原问题：

- `converged = len(local_results) == len(valid_windows)` 只表示每个窗口都解出了结果，不代表统计收敛。
- `min_overlap_proxy = 1/(1+max|Delta f|)` 是自由能间距的单调函数，不是 MBAR overlap matrix。
- `abfe_pipeline.py` 当前没有真正读取并门控 `converged`。

修复内容：

- `GlobalMBARAnalyzer.solve_stage_integrated`（单一采样分布 + 多个零样本目标态的场景，标准 overlap matrix 会退化）：改用 `mbar.compute_effective_sample_number()` 算出每个目标 λ 态的重加权有效样本比例（ESS ratio = neff / n_sampled），窗口 `min_overlap` = 该窗口最差的 ESS ratio；已用合成数据验证该指标随真实重叠单调变化（近距离目标 λ → ESS ratio≈0.96；极远目标 λ → ESS ratio≈0.006）。`converged` 现在要求所有窗口都解出 **且** 全局最小 ESS ratio ≥ 阈值 0.05。旧的 Δf 间距量保留在新字段 `lambda_spacing_max_step_kJ_mol`，不再冒充"重叠度"。
- `TraditionalMBARAnalyzer.solve`（REMD 场景，所有态都真实有样本，标准 overlap matrix 有效）：改用 `mbar.compute_overlap()["matrix"]` 的相邻态（`|i-j|=1`）最小值作为 `min_overlap`，阈值 0.03（与 `abfe_core.py` 在线监控已有的相邻窗口重叠阈值保持一致）。
- `abfe_pipeline.py::_assert_stage_result_sane` 新增硬性检查：结果带 `converged`/`min_overlap` 字段时，`converged is False` 或 `min_overlap < min_overlap_threshold` 直接 `raise RuntimeError`，拒绝把重叠不足的阶段标记为 completed；不带这些字段的旧路径结果不受影响（向后兼容）。已用单测验证三种情况（正常通过/低重叠正确拒绝/无字段的旧结果正常通过）。

状态：已修复。

### 3. VDW/vanishing 腿缺少 LJ 长程色散 tail 修正

位置：原始问题涉及四处 `setUseLongRangeCorrection`/`setUseDispersionCorrection(False)`（`ibs_engine.py` 的配体-环境软核势 `_create_softcore_force`、`ibs_engine.py` 的 IBS CV 探针软核势——经代码核实这其实和前一项是**同一个力对象**，不是两处独立缺口、`ibs_engine.py` 的 shadow-Coulomb 探针（纯 Coulomb，无 LJ 项，跟本条无关）、`abfe_core.py` 的配体内部 LJ+Coulomb 自定义力）。

问题：softcore VDW 交互组力不自动包含原始 `NonbondedForce` 的 LJ dispersion correction。截断外 LJ 吸引尾部对 `Delta G_vdw` 的贡献被丢弃，复合物盒子和纯水盒子环境不同，不能假设在 `Delta G_solvent - Delta G_complex` 中完全抵消。典型量级约 0.1-0.5 kcal/mol，随配体尺寸、极化率、cutoff 和环境密度变化。

**为什么不能直接翻 OpenMM 的开关**：用独立脚本 `test_lrc_interaction_group_compat.py`（不依赖项目代码，可独立在任意 GPU 节点跑）在真实 CUDA 节点上实测三个问题：

- Q1（`addInteractionGroup` + `setUseLongRangeCorrection(True)` 对纯 LJ 是否work）：PASS。电荷=0 时，该组合正确把修正限制在配体-环境 cross term 上，λ_vdw=1.0 和 0.5 两个点截断误差都消除了 ~88%。
- Q2（REMD/`BeutlerSoftcoreBuilder` 用 `context.setParameter` 换 λ，LRC 是否会卡在建 Context 时的旧值）：PASS，`setParameter` 后 LRC 正确重新计算。
- Q3（真实非零电荷下是否安全）：**FAIL，灾难性失败**。`_create_softcore_force`/`BeutlerSoftcoreBuilder` 的表达式把 LJ 和 Coulomb 拼在同一个 `CustomNonbondedForce` 里；OpenMM 的解析长程修正对 LJ 的 `r^-6` 尾项收敛，但对同一表达式里 Coulomb 的 `1/r` 尾项在数学上发散，实测直接让 CUDA 后端崩溃（"terminate called recursively" → core dump）。因此**不能**简单翻转这个开关。

修复内容（仅 ACE/`dual_lambda` 默认路径，即 `_create_softcore_force`）：

- 完全不用 OpenMM 内建 LRC 机制，改为 Python 侧手算标准的均匀密度假设下的解析尾项（只修吸引性 `r^-6` 项，跟 OpenMM 自己的 `setUseDispersionCorrection` 同一套近似/惯例，不涉及 Coulomb，天然规避了 Q3 的发散问题）。
- `ibs_engine.py::build_ibs_dual_system` 建窗口系统时一次性算出几何常数 `S = Σ_{i∈ligand} Σ_{j∈environment} sqrt(ε_iε_j)·(0.5(σ_i+σ_j))^6`（新增 `_lj_tail_correction_S_kj_nm6`），得到前置系数 `prefactor = -(16π/3)·S/rc^3` 和逐态数组 `λ_vdw^n_lj`，挂在 `ibs_wrapper.lj_tail_prefactor_kj_mol` / `lj_tail_lambda_vdw_pow` 上。
- `IBSSampler` 新增 `_lj_tail_correction_kj_mol()`：每帧读当前盒子体积 `V(t)`，算 `prefactor·λ_vdw^n_lj/V(t)`，在 `collect_energies()` 里加进 `interaction_energies`（`get_raw_interaction_energies()`/诊断路径保持不加、维持"raw"语义不变）。
- 公式系数已用 Q1 的实测数据反向核对：第一版系数误写成 `8π/3`，跟 λ=1.0 那组数据一比对差了整 2 倍，改成正确的 `16π/3`（配体-环境是跨物种对，不需要像同种粒子对那样再除以 2）后，λ=1.0 预测 -2.1852 kJ/mol vs 实测 -2.1845，λ=0.5 预测 -0.5463 vs 实测 -0.5387，均一致（<1.5% 残差，符合软核分母在 cutoff 附近偏离纯 `r^6` 的预期误差量级）。
- `potential_type="dexp"` 显式跳过并打印警告——DEXP 替身势的尾项公式没有验证过，不能直接套用同一公式。
- `abfe_pipeline.py` 里原先硬编码 `"status": "not_implemented"` 的 `lj_long_range_dispersion_correction` 字段已同步改为 `"implemented_analytic_mean_field"`（只有一处写出点，复合物腿/溶剂腿共用，已一并覆盖）。
- `abfe_core.py::THERMODYNAMIC_CYCLE_DOC`（`abfe_core.py:827` 附近）原文写着"任何 LJ tail/LRC 项必须由一个经过验证的额外 cycle term 处理"，跟现在"直接烤进 `IBSSampler.collect_energies()` 采样哈密顿量本身，不是事后加的独立 cycle term"的实现方式已经不符——已同步改写该段文案，写清楚当前实现方式、发散崩溃的原因、以及 `single_lambda` 路径仍缺修正。这段文本会被写进每次运行的 `thermodynamic_cycle.md` 和 `final_results.json`/`final_binding_results.json` 的 `thermodynamic_cycle` 字段，**所有在本次修复之前跑出来的 `output/` 目录（包括 `./output` 这份修复前基线）里嵌入的都是修复前的旧文案，不代表当前源码状态**；只有本次修复之后重新跑（如 `./output_lrc_fix`）才会带上新文案。

**尚未覆盖 / 仍是缺口**：

- `abfe_core.py::BeutlerSoftcoreBuilder`（`--decoupling single_lambda`/REMD 路径，`ibs_engine.py:505/4171/5116` 三处调用）**完全没有等效修正**——现在变成"`dual_lambda` 有 LJ LRC、`single_lambda` 没有"，两种解耦方案结果不可比,必须补齐才能消掉这个新的不对称。
- `single_lambda`/REMD 仍缺修正；默认 `dual_lambda` 的旧窗口/阶段/最终结果现在会因缺少或不匹配的 `WCA_ACCOUNTING_VERSION` / `IBS_BIAS_PROTOCOL_VERSION` 被拒绝复用，已不再是“旧数据静默混入”的原始状态。当前仍缺的是**完整、独立的 Hamiltonian/analysis 指纹**：若未来只改变 LRC 公式而没有同步递增某个现有版本，缓存仍可能误命中；这个一般性问题归入下方“完整协议 SHA256”条目。
- `abfe_core.py:3670` 的配体内部 LJ 力（`ll_force.setUseLongRangeCorrection(False)`）经代码核实：表达式不含 λ，是全程满强度的分子内项，小分子原子间距基本都在 1.2nm cutoff 内、没有截断对，即使有残余也是每个 λ 态相同的常数、在 MBAR 差分中抵消——判定为大概率非实际数值风险，本次未改动。

影响：当前主链（`dual_lambda`）已获得修正；旧 `output_lrc_fix/` Stage2 是在 IBS 偏置冻结协议修复前产生，不能作为最终数值验证。需要在修复“独立连续通过计数”后，用 `IBS_BIAS_PROTOCOL_VERSION` 再次升级所对应的新协议重跑 Stage2；`single_lambda` 仍是未修复状态。

**2026-07-15 更新**（详见"同日复审第八轮"）：上面这条记录的是当时（1.2nm 之外 cutoff-only、仅吸引项 `r^-6`、`single_lambda` 无修正）的状态，现已整体推进：
- 公式本身升级为 switching+softcore-aware（补上了 `1.0-1.2nm` 被 switching 削弱的能量、`r^-12` 排斥项、真实 softcore 分母积分），不再是当时"部分修复"描述的那个近似版本，`TRADITIONAL_LJ_LRC_PROTOCOL_VERSION` 1→2。
- "尚未覆盖"里的 `single_lambda`/REMD 缺口已补齐：`TraditionalMBARAnalyzer.compute_u_kn` 现在用同一套 `_lj_tail_lrc_coefficients_kj_mol`（`BeutlerSoftcoreBuilder` 实际使用的默认 `alpha_lj=0.5, power_lj=1`）计算逐 λ 系数，不再是"`dual_lambda` 有、`single_lambda` 没有"的不对称状态。
- dual_lambda 每窗口 `convergence.json` 新增 `lj_tail_lrc_protocol_version` 字段并接入 resume/窗口复用校验，堵上了当时提到的"缺完整 Hamiltonian 指纹、LRC 公式变了缓存可能误命中"这个具体缺口（更一般的完整协议 SHA256 仍是未闭环项，见下方对应条目）。
- `abfe_core.py:3670` 的配体内部 LJ 力判定（分子内项、不含 λ、cutoff 内无截断对）未受本次改动影响，判断依据不变。

状态：**已修复**（公式本身；仍不包含下方"完整协议 SHA256"条目指出的通用指纹缺口，那是一个更大范围的、不限于 LRC 的遗留问题）。

重要边界：`apbs_correction.py`（无论是旧版 PB 溶剂化差分还是新版 Rocklin 有限尺寸修正）都是静电/连续介质外部项，不能替代、也不影响本条 LJ tail correction 的状态。

### 4. VDW 窗口拼接 offset 权重不一致

位置：`ibs_engine.py` 中 VDW stage integrated/global MBAR 拼接逻辑（`ibs_engine.py:3604` 附近）。

问题（历史）：窗口间 offset 曾使用重叠点上的非加权平均，但每个 lambda 的合并使用逆方差加权。当重叠点不确定度差异较大时，offset 会相对最终合并曲线偏移，影响 `f_curve[-1] - f_curve[0]`。

状态：已修复。offset 计算已改为对重叠 lambda 的逆方差加权平均（`inv_var = 1 / max(offset_vars, 1e-12)`），并显式传播 `offset_var` 累加进每个窗口的 `var_loc`，与下方逐 lambda 合并使用同一套逆方差加权逻辑一致。

### 5. ✅ 已修复（部分）— WCA shield（Group 4）被当成 λ 无关物理量，相邻窗口 Hamiltonian 不一致

位置：`ibs_engine.py` 的 λ-WCA 防护壳力构造（Group 4，约 `ibs_engine.py:1495-1516`）、每窗口设置 `lambda_shield`（`run_all_windows`，约 `ibs_engine.py:2623`）、`IBSSampler.collect_energies()`（约 `ibs_engine.py:2249`）。

原问题：Group 4 的表达式含 `lambda_shield*(1-lambda_shield)` 项，每个采样窗口只设一次 `lambda_shield = mean(该窗口 λ_vdw)`，整窗采样期间不再变化。分析时 `e_base = getState(groups={0,2,3,4,5})`，注释写"严格 λ 无关"，但 Group4 的真实取值按窗口均值算，并不随窗口内被 reweight 的目标态变化。后果：两个相邻窗口在重叠的那个 λ 态上，`base_energy` 里的 WCA 贡献是用各自窗口的均值 λ 算出来的，不是该态本身该有的 λ；`solve_stage_integrated` 拼接重叠态时会把这个系统差异吸收进 offset，伪装成自由能差的一部分。量级上，`_estimate_wca_shield_parameters` 给出 `eps_wca∈[1.0,2.5]` kJ/mol、`rc∈[0.18,0.32]` nm（短程截断），`λ(1-λ)` 在 λ=0/1 端点为零、λ=0.5 附近最大——按此推算中段窗口边界处的系统偏差量级可能到低个位数 kJ/mol，但没有对已产数据做过实测拆解验证（已有 `*_base.npy` 只存了合并后的总量，无法反推 WCA 单独贡献）。

修复内容：

- Group 4 明确定义为**纯采样期偏置**（帮助窗口内动力学不塌缩，跟 Group 1 的 IBS flattening bias 同类角色），不再计入任何目标态的物理能量。
- `e_base` 力组从 `{0,2,3,4,5}` 改为 `{0,2,3,5}`；`e_bias` 从 `{1}` 改为 `{1,4}`——Group1+Group4 一起完整落盘、完整 reweight 掉。
- 新增 `WCA_ACCOUNTING_VERSION`（`ibs_engine.py`，当前值 2）常量，写入每个窗口的 `convergence.json`；`run_all_windows` 的 resume 判断、`abfe_pipeline.py::_invalidate_stage_window_files` 的窗口复用匹配、`_stage_protocol_key`/顶层 `_top_level_protocol_key` 均已接入该版本号，缺失或不匹配一律 fail closed、拒绝复用（含 stage1/2 "completed" 缓存和顶层 `final_results.json` 的早退路径——此前这两处对"无协议指纹"的旧缓存是"信任并跳过"，已改为"视为缓存失效"）。
- 顺带修复：LRC（`_lj_tail_correction_kj_mol`）此前跟 softcore 探针能量合并成同一个数组，同时喂给 `update_weights()`（训练 f_k）和 MBAR——但真正驱动 Group1 偏置力的 `cv_k_int` 只有 softcore、不含 LRC，导致 f_k 平坦化学习的目标偏离真实施加的偏置力（降低采样效率，不影响 MBAR 正确性，因为 sampled row 直接读取真实 `e_bias`）。现拆分为 `bias_cv_energies`（纯 softcore，喂 f_k 训练）与 `target_energies`（softcore+LRC，喂 MBAR）。
- 顺带修复：base 能量读取失败此前回退成假 `0.0` 并"标记"，但从未真正校验、照常写入 `base_energy_history`（见下方「已验证正确」条目的更正）。现改为 `NaN`，append 前要求 base/bias/交互能量全部 finite，否则整帧跳过；连续失败达到阈值（5 帧）直接 raise，不再无限静默跳帧。
- 窗口产物复用逻辑（`abfe_pipeline.py::_invalidate_stage_window_files`）已验证：λ 自动加密后，只有真正内容变化的窗口会被清理重采，其余窗口按 λ 内容比对直接复用（含跨编号重命名）。曾用真实日志中的一次 18→22 加密逐一核对插点和复用映射，确认当时 0 复用是因为每个旧窗口的 λ 内容都发生改变，而不是映射失效；但触发那次加密的低 ESS 后来已确认受未收敛 IBS 偏置污染，不能再把它解释成“物理 λ 密度不足”的证据。
- 新增 `IBS_BIAS_PROTOCOL_VERSION=2`，并接入窗口 `convergence.json`、窗口 resume、窗口跨编号复用、IBS state、stage cache 与顶层 final cache。旧协议下“预热未收敛仍放行 + 生产期继续更新 `f_k`”产出的数据一律不能作为当前协议缓存复用。
- 偏置预热现在逐态检查 `ema_mean_p` 是否落在 `(0.5/K, 2.0/K)`，同时要求最近 `f_k` 更新幅度低于阈值并连续通过；达到步数上限仍未通过时直接 raise，不再进入生产。生产入口另有 `sampler.bias_converged` 防御性断言。
- 生产采样阶段已删除 `update_weights()`：`f_k` 在生产期固定，`base+bias` 才对应一个可由当前 augmented MBAR 表示的固定 sampled distribution。
- λ 自动加密新增“未改善熔断器”：若下一轮 `min_overlap <= previous_min_overlap`，立即报错停止，不再机械耗尽五轮。真实运行里观察到的 `0.007973→0.003479` 现在会在第一次变差后停止。

尚未覆盖 / 仍是缺口：

- 顶层 `_top_level_protocol_key` 与 `_stage_protocol_key` 目前只覆盖少量离散字段（顶层含 `decoupling_scheme`/`potential_type`/`has_boresch`/`decharge_method`/`wca_accounting_version`/`ibs_bias_protocol_version`；stage 不含 decoupling scheme），不含实际 λ 数组/窗口边界、态数、采样步数、温度、softcore 参数、Boresch anchors/力常数/平衡值、system/Hamiltonian 指纹——这些少量字段不变时，换 λ 路径或态数仍可能直接复用旧结果。建议改成对完整规范化 JSON 做 SHA256。
- `_run_dual_lambda_optimization`（`abfe_pipeline.py`，约行 1573）里预优化器抛异常会被整体捕获，静默降级为线性 λ 路径当"优化成功"缓存，外层"拒绝静默回退线性路径"的报错永远等不到异常触发——需要让异常传播，或显式返回 `optimization_failed=True` 并禁止缓存。
- `frame_finite`（`ibs_engine.py::collect_energies`）只检查 `np.isnan`，未检查 `np.isinf`，+Inf/-Inf 帧仍可能进入 `energy_buffer`/`energy_history`。
- `diagnose_softcore_cv_values`（`ibs_engine.py:1317`）诊断打印仍是旧 Group 定义（`base={0,2,3}`, `bias={1}`），不影响生产数据但会误导排查。
- **P0：严格预热的“连续通过”计数仍会重复使用同一批统计量。** 主循环每 500 步检查一次，但 `update_weights()` 只有在 `energy_buffer` 累积 10 帧后才真正更新 `ema_mean_p/f_history`；当前 `consecutive_pass_count` 在没有新更新的中间检查点也会递增，同一批结果可能被算成连续三次通过。必须只在 `update_weights()` 本次返回新 `f_k` 且 `f_history` 至少有两个独立更新时评估/递增；修复后将 `IBS_BIAS_PROTOCOL_VERSION` 升至 3。
- **P0：自动加密出的态数不能跨 resume 保留。** Stage2 preopt loader 仍要求 `len(cached_lambdas) == 初始 stage2_states`；例如合法修复成 22 态后重启，仍会因 `22 != 18` 丢弃路径并从 18 态重来。应给 preopt payload 写入完整协议指纹；匹配且 `provenance.source == "auto_repair_by_overlap"` 时，以缓存长度接管 `stage2_states`。
- `_is_overlap_failure()` 仍把任意 `converged is False` 判成可插点修复。偏置预热失败现在会提前 raise，因此原来的主要误循环已阻断；但局部 MBAR 求解失败/窗口缺失等其他失败仍可能被错分。应只在 `min_overlap` 明确低于阈值且 `window_overlap_diagnostics` 完整时返回 True。
- 尚无回归测试锁定以上任何一条（base+bias 精确重构、Group4 纯采样偏置、LRC 只进 MBAR target、版本门控生效、独立连续通过计数、生产期 `f_k` 冻结、Inf 帧拒绝、优化失败不被静默缓存）。

状态：**部分修复**——WCA/LRC 记账和“预热失败不放行、生产期冻结 `f_k`”的主体语义已改对；独立连续通过计数、自动修复路径 resume、完整缓存指纹、静默线性降级、Inf 检测和回归测试仍是缺口。旧 `output_lrc_fix/` Stage2 不属于当前冻结协议的有效验证数据。

---

## 物理/建模状态

### PME self correction：`+C*lambda^2` 已撤销，仅保留诊断

位置：`ibs_engine.py` 的 PME decharging offline `u_kn` 分支和 `pme_self_correction_*` helper。

结论：此前“需要手动加回 `+C*lambda^2`”的判断是错误的。OpenMM 的 `NonbondedForce.addParticleParameterOffset` 会在每个 lambda 状态下用已缩放电荷重新计算完整 PME 能量，包含 Ewald self-energy。该 self-energy 是该 lambda 态哈密顿量的一部分，不是缺失伪项。手动加回 `+C*lambda^2` 会反向抵消真实存在的能量项。

当前状态：

- `apply_pme_self_correction` 在生产路径中保持 `False`。
- `pme_offset_charge_square_sum()` 仅用于诊断记录 `charge_square_sum_e2`。
- 旧输出若包含历史 PME self-correction 文字或旧数值，必须以当前代码和本文件为准。

仍需：用当前代码做一次干净重跑，确认 decharging 腿数值和旧缓存完全脱钩。

### Boresch 力常数估计与谐振假设

已修复内容：

- `GeometricRestraintEstimator` 会保留 raw force constants、clip ranges、clip flags、分布诊断和 warnings。
- `assess_boresch_harmonicity` 已接入内部 Boresch 来源（`auto` / `orb_simple` / `simple` / `fluctuation`），对 r / thetaA / thetaB / phiA / phiB / phiC 做分布诊断并写入 provenance。
- `final_binding_results.json` 使用真实计算出的 `analytical_release_assumption_checked` 和 `analytical_release_reliable`，不再只写静态说明。

局限：

- 外部锚点文件来源（`traditional` / `orb_ml`）没有绑定预平衡轨迹，不自动获得这项诊断。
- 当前诊断是涨落分布统计判据，不等价于沿 PES 的数值积分或能量扫描。

状态：主链已修，仍建议检查输出 JSON 中的 Boresch diagnostics。

### Softcore / WCA 参数

已修复内容：

- `_normalize_softcore_params` 不再把 softcore alpha 硬覆盖为固定 `alpha_lj=0.7` / `alpha_coul=0.5`。
- 默认使用 `ACESoftcorePotential.optimize_alpha()` 的自适应值。
- 显式传入 softcore 参数时尊重用户值。
- WCA shield 参数改为基于配体-环境 LJ sigma 的有界估计，并记录来源。

状态：已修复。

### 2D geodesic lambda 路径规划噪声

问题本质：短采样估计度量张量会有噪声，影响 lambda 点分布效率，但不直接进入最终 Delta G。

已修复内容：

- `compute_2d_metric_grid()` 可返回采样诊断。
- `optimize_2d_geodesic_path()` 打印有效网格比例、失败/unsafe 点数、每点导数样本数。

状态：已修复为可审计的效率启发式；最终仍需结合 overlap diagnostics 判断。

---

## APBS correction helper 当前状态

`apbs_correction.py` 已从 v2（PB 连续介质溶剂化能三体差分）**整体重写**为 Rocklin 膜体系有限尺寸静电修正，方法与 v2 完全不同，不是同一个工具的小版本迭代。

新方法：Rocklin/RIP 有限尺寸电荷修正（Wu & Biggin, JCTC 2022, DOI 10.1021/acs.jctc.1c01251，基于公开 RocklinC 参考实现）。

```text
Delta G_corr = Delta G_NET+USV + Delta G_RIP + Delta G_EMP + Delta G_DSC
```

对每个代表性 snapshot，APBS 生成三张势能网格（`protein_RIP_het` / `ligand_RIP_het` / `ligand_RIP_hom`），积分后代入上述解析+数值混合公式；支持 `--diel-map-x/y/z` + `--lipid-charge-map` 描述脂双层的介电/固定电荷分布（膜蛋白场景），二者必须同时提供或同时不提供。

**适用范围（硬性限制,`prepare` 会直接报错拒绝跑）**：只对**电荷发生改变**的炼金术微扰适用（`--ligand-net-charge` 非零），且要求该微扰用的是 neutralizing-plasma 电荷处理方式（`--charge-treatment neutralizing-plasma`），不能用于 co-alchemical-ion/charge-transfer 路线或中性配体。**当前项目上下文里 Atenolol 被记录为中性配体（净电荷≈0，见文件顶部"运行上下文"），也就是说按现有设置直接调用这个工具会被 `prepare` 拒绝——除非配体的质子化状态被改为带电态（Atenolol 生理 pH 下实际是仲胺阳离子，pKa≈9.6），这个工具目前对本项目当前跑法尚无实际适用场景，需要先确认是否要切换配体电荷态。**

已核实的实现细节：

- ✅ 已修复 — `_write_apbs_input` 里 `ligand_RIP_hom` 那个 `elec` block 曾用 `mol 2`（`ligand_in_protein.pqr`，配体带电+受体原子仍以零电荷几何形式存在）而不是本该用的 `mol 3`（`ligand_only.pqr`，真正孤立的配体）。当前代码（`apbs_correction.py:379`）已确认为 `"ligand_RIP_hom", 3`，直接用 `mol 3`，不再依赖"两个硬编码数字碰巧相等"的隐式等价性。
- ✅ 已修复 — `pdie=1.0`/无 `ion` 行（κ=0）曾是没有解释的硬编码。当前代码已把它们提升为显式命名的模块级协议常量 `ROCKLIN_SOLUTE_DIELECTRIC = 1.0`（`apbs_correction.py:53`）、`ROCKLIN_APBS_ION_STRENGTH_M = 0.0`（`apbs_correction.py:54`），`_write_elec_block` 处配了协议说明注释（`apbs_correction.py:315-316`："Rocklin/RIP APBS grids use an unscreened reference with solute dielectric fixed at 1.0. Keep this as protocol state, not a CLI knob."），且 `ROCKLIN_APBS_ION_STRENGTH_M` 已写入 manifest 的 `settings.apbs_mobile_ion_strength_M`（`apbs_correction.py:505-506`）留痕，不会被误"修复"成可调 CLI 参数。
- 仍未核实 — `NET`/`RIP`/`EMP`/`bq_*` 背景电荷自能项的具体系数是照抄论文/RocklinC 参考实现，本次审计只核对了单位量纲自洽（通过），没有对照论文原文或已知答案的测试用例核实系数本身是否抄对——建议接入生产前找一个有已知答案的测试案例跑一遍核对数值。

仍然不变：

- 无论新旧版本，APBS 都是静电/连续介质外部项，跟 LJ tail/LRC correction（见上一条）完全独立，互不替代。
- 当前主链默认 APBS correction 为 0；只有显式使用 `--apbs-correction-kj-mol` 才进入最终 Delta G。

---

## 当前不触发但启用相关路径前应修

### 1. ✅ 已修复 — decharging 分支边界情形可能掉到截断 Coulomb

位置：`ibs_engine.py::compute_u_kn`（`REMDManager._build_replicas` 的同名判定逻辑暂未改动，风险敞口不变但当前主链不走该类）。

原问题：`is_pme_coulomb_leg` 仅靠 `np.allclose(lambdas_vdw_arr, 1.0)` 静默判定 PME Coulomb leg，未来若误传非 decharging 的 λ 表会静默走错分支、给出错误结果，且没有硬性报错保护。

修复内容：在 `compute_u_kn` 进入 `is_pme_coulomb_leg` 分支前新增显式断言——若 `lambdas_vdw_arr` 不严格满足 `np.allclose(..., 1.0, atol=1e-6)`，直接 `raise RuntimeError` 并说明可能的误配置原因，而不是继续静默执行。

状态：已修复。`REMDManager.__init__` 里同名的 `is_pme_coulomb_leg` 判定（在线 REMD 路径）仍是纯 `np.allclose` 推断，未加同等断言，留作后续可选加固项。

### 2. ✅ 已修复 — `run_full_abfe_loop` 是 unused 但组装不一致

位置：`abfe_pipeline.py::run_full_abfe_loop`。

原问题：复合物侧正确取 `total_delta_G_complex_kJ_mol`（取负号），但溶剂侧优先取旧口径 `decoupling_delta_G_kJ_mol`，与 `total_delta_G_complex_kJ_mol`（已含约束/Boresch 修正）不一致；主链当前不调用它，但一旦接线会给出偏差结果。

修复内容：调整键优先级为先取 `total_delta_G_complex_kJ_mol`，再回退 `decoupling_delta_G_kJ_mol`/`total_delta_G`，与 `runabfe.py` 主流程及复合物侧口径保持一致。

状态：已修复（函数本身仍未被主链调用，属于孤立但已自洽的工具函数）。

### 3. ✅ 已修复 — parallel stages worker 崩溃可能读到旧 JSON

位置：`abfe_pipeline.py` 的 `parallel_results/` 写入/读取逻辑。

原问题：`parallel_results/` 目录用 `exist_ok=True` 创建且不清空；worker 子进程崩溃/被杀而未写出新 `stage1.json`/`stage2.json` 时，父进程会静默读到上一次运行遗留的结果，误判为本轮成功。

修复内容：在派生 worker 前，若 `stage1.json`/`stage2.json` 已存在则先删除，确保 worker 未正常写出时后续 `open()` 直接抛 `FileNotFoundError`，而不是读到陈旧数据。

状态：已修复（未额外引入 run fingerprint / 时间戳校验，删除已足以消除"静默读旧结果"的风险；如需更强审计可再加 run id）。

### 4. ✅ 已修复（但还有第三层更保守的 clip，见下）— 角度力常数 clip 范围和解析函数接受范围不一致

位置：`GeometricRestraintEstimator`（`abfe_core.py`，clip 上界 1000）与 `calculate_boresch_analytical_correction`（原上界 500）。

原问题：估计器可把角度力常数 clip 到 1000 kJ/mol/rad²，但解析修正函数对 > 500 会硬报错，导致估计器给出的合法值（500~1000 区间）在下游直接调用时崩溃。

修复内容：`calculate_boresch_analytical_correction` 的角度力常数校验上界从 500 统一改为 1000，与 `GeometricRestraintEstimator` 的 clip 范围完全一致（这一层的不一致已消除）。

**补充核实**：`runabfe.py:1291-1301` 在 Boresch 谐振性校验之后还有一次独立的"裁剪力常数到安全范围"后处理（`post_clip_ranges`），把 `kthetaA/B`、`kphiA/B/C` 又裁到 `[10.0, 200.0]`——比上面统一到的 1000 更保守，且这一步在估计器和解析修正函数之后执行，实际生效的角度/二面角力常数上界是 200，不是 1000。这不是新 bug（`kr` 的范围 `[100, 2000]` 也是独立设定，看起来是有意为之的额外保守层，不是遗漏），但上面"完全一致"的表述不完整——三层 clip（估计器 1000 / 解析函数 1000 / `runabfe.py` 最终 sanitization 200）里只有前两层互相对齐，第三层更紧，需要知道这一点才能正确解读最终力常数为什么会小于 1000。

状态：估计器与解析函数之间已修复一致；`runabfe.py` 的最终 sanitization 上界（200）是否也该统一到 1000，还是保持更保守的独立设定，需要产品决策，本次未改动。

### 5. ✅ 已修复 — ORB 相关类缺少 `HAS_ORB` 前置检查

位置：`OrbBoreschEstimator.__init__`（`abfe_core.py`）；`Orbv3DEXPFittingPipeline.__init__` 已有该检查，未受影响。

原问题：未安装 `torch` / `openmmml` 时，`OrbBoreschEstimator.__init__` 直接引用 `torch.cuda.is_available()`，会抛出裸 `NameError` 而非清楚的依赖缺失说明。

修复内容：构造函数开头新增 `if not HAS_ORB: raise ImportError(...)`，与 `Orbv3DEXPFittingPipeline` 的既有模式一致。

状态：已修复。

### 6. ✅ 已修复 — DEXP 拟合逐帧异常被归入 outlier 但不记录原因

位置：`Orbv3DEXPFittingPipeline` 逐帧标注循环（`abfe_core.py`）。

原问题：`except Exception as e` 只累计 `stats["skip_outlier"]` 并 `continue`，异常内容完全丢失，真实代码 bug（而非物理离群帧）可能被长期伪装成"outlier"。

修复内容：新增 `stats["skip_outlier_reasons"]` 列表记录每次异常的帧号 + 类型 + 消息；前 5 条异常立即打印，超出部分在采样诊断汇总时报告剩余条数，不再静默吞掉。

状态：已修复。

---

## 轻微报告/下游误用风险

### 1. ✅ 已修复 — `--analyze-only` 组装口径可能与正式输出不一致

问题：dual_lambda `--analyze-only` 路径此前恒用 `_analyze_dual_leg` 从原始窗口能量文件重新估算 `decoupling_delta_G_kJ_mol`，不含正式 pipeline 烘焙的 PME 自能/约束修正等项，也从未应用 `--apbs-correction-kj-mol`（即便该 CLI flag 本身存在）。

修复内容：

- 若复合物腿与溶剂腿的 `final_results.json` 均存在，优先复用其中权威的 `total_delta_G_complex_kJ_mol`（与主流程/正式组装口径一致），并相应把 `dg_boresch_term` 清零以避免对已烘焙的 Boresch 修正二次扣减；只有在缺少正式结果文件时才回退到 `_analyze_dual_leg` 重新估算，并在日志中明确提示这是粗略核查值。
- 补上此前完全缺失的 `--apbs-correction-kj-mol` / `--apbs-correction-note` 应用，输出中新增 `delta_G_bind_uncorrected_kJ_mol` / `apbs_correction_kJ_mol` 字段，与主流程组装口径对齐。

状态：已修复。仍非与主流程完全共享同一份组装函数（存在少量重复逻辑），但字段口径和修正项已对齐。

### 2. ✅ 已修复 — 最终 JSON 中 `boresch_correction_kJ_mol` 容易被下游二重减法

问题：`Delta G_complex`/`total_delta_G_complex_kJ_mol` 在部分路径下已经把 Boresch release 烘焙在内，但 JSON 里仍只并列写出 `boresch_correction_kJ_mol`，未标注它是否已被计入，容易让下游脚本误以为是独立可加项而二次扣减。

修复内容：在 `abfe_pipeline.py`（`run_full_pipeline` 的 `final_results.json`、`TraditionalABFEPipeline.run_full` 的 `final_results.json`）和 `runabfe.py`（主流程 `final_binding_results.json`、traditional 模式 `final_binding_results_traditional.json`、`--analyze-only` 的 `final_results_postprocess.json`，共 5 处写出点）新增显式布尔字段 `boresch_correction_already_included_in_complex_delta_G`（或 `..._in_total_delta_G`），并按各路径实际组装逻辑正确标注 `true`/`false`，附加说明性 note。修复过程中还发现并同步修正了一个由此暴露的潜在双重扣减风险：`--analyze-only` 复用权威 `final_results.json` 时必须把独立计算的 `dg_boresch_term` 清零（见上一条），否则会把同一个 Boresch 修正减两次。

状态：已修复。

---

## 已验证正确 / 已修复

核心物理与数值：

- 热力学循环符号：`Delta G_bind = Delta G_solvent - Delta G_complex + Delta G_APBS`。
- Boresch 解析修正公式与单位：Boresch 2003 形式、`V0=1.6605 nm^3`、`sin(theta)` 奇点保护、角/二面角力常数单位自洽。
- PME self correction `+C*lambda^2` 已停用，仅保留 inert diagnostics。
- `assess_boresch_harmonicity` 的 6 个坐标定义与 `LambdaDependentBoreschForce` / estimator 一致，二面角诊断前 unwrap。
- 采样哈密顿量和评估哈密顿量在当前主链中匹配：VDW 腿 IBS/ACE softcore/GlobalMBAR，Coul 腿 REMD/PME/TraditionalMBAR。
- ~~`collect_energies` 中 `base_energies` 失败归 0 不会偏置 MBAR Delta G，因为它作为每列常数相消，只影响数值 conditioning 和日志可解释性。~~ **2026-07-13 更正**：这条结论本身（归 0 在"每列常数"意义下无偏）没错，但下面"已不再裸吞异常导致静默归零"这条描述在本次审计开始时被发现并不成立——实际代码里 `e_base=0.0` 的静默回退分支仍然存在且从未被真正校验/跳过。现已按「最高优先级」第 5 条改为 NaN + 整帧跳过 + 连续失败 raise，不再依赖"常数相消所以无偏"这个论证。

工程修复已验证：

- ~~`IBSSampler.collect_energies()` 不再裸吞异常导致 `e_base` 静默归零。~~ **2026-07-13 更正**：这条在本次审计前并不成立，`e_base=0.0` 的静默回退当时仍在代码里且未被真正校验。现已修复（见「最高优先级」第 5 条：改为 NaN + 整帧跳过 + 连续失败 raise）。
- PBC 分子完整性修复已重新启用。
- Boresch 平衡值最后一帧刷新失败时记录 `is_fallback` / `equilibrium_update_error`。
- `debug_mode` 默认改为 `False`，生产采样诊断打印已门控。
- resume 会逐窗口检查能量文件形状并跳过已完成窗口。
- 溶剂腿已传递 `n_workers` / `parallel_stages`。
- softcore alpha 默认自适应，WCA 参数基于 sigma。
- 主系统总能量/力的 NaN/Inf 灾难检测无条件执行；但 `IBSSampler.collect_energies()` 的 `frame_finite` 对两个逐态数组仍只查 NaN、不查 Inf，不能把前者误写成后者也已覆盖。
- torsion exclusions 覆盖 CustomTorsionForce / RBTorsionForce。
- checkpoint 缺文件时抛错而非静默返回 0。
- 局部重复 `NumpyEncoder` 已删，统一从 `abfe_core` 导入。
- 预优化器端点和插值后 NaN 检测已补。
- `apbs_correction.py` 已重写为 Rocklin/RIP 膜有限尺寸修正（原 v2 PB 溶剂化差分方法已被取代），尚未用真实 APBS binary/真实带电配体跑通（当前 Atenolol 上下文为中性，见「APBS correction helper 当前状态」）。
- ACE/`dual_lambda` 路径的 LJ 长程色散尾项修正已实现（手算解析公式，非 OpenMM 内建 LRC），公式系数已用 `test_lrc_interaction_group_compat.py` 在真实 GPU 上的实测数据核对通过（λ=1.0/0.5 两点，误差 <1.5%）；`single_lambda`/REMD 路径仍缺同等修正。
- `compute_u_kn` 的 PME decharging 分支新增 `lambda_vdw == 1.0` 硬性断言。
- `run_full_abfe_loop` 溶剂侧键优先级已改为 `total_delta_G_complex_kJ_mol` 优先。
- `parallel_results/` 的 `stage1.json`/`stage2.json` 在派生 worker 前会先清空，避免读到旧运行结果。
- `calculate_boresch_analytical_correction` 角度力常数上界统一为 1000 kJ/mol/rad²，与 `GeometricRestraintEstimator` clip 范围一致。
- `OrbBoreschEstimator.__init__` 新增 `HAS_ORB` 前置检查。
- DEXP 逐帧异常改为记录原因（帧号+类型+消息）而非静默归入 outlier。
- `--analyze-only`（dual_lambda）优先复用正式 `final_results.json` 的 `total_delta_G_complex_kJ_mol`，并补上此前完全缺失的 APBS 修正应用。
- 5 处最终 JSON 写出点新增 `boresch_correction_already_included_in_*` 显式标记，避免下游对 Boresch 修正二次扣减。
- WCA 记账口径已由 `WCA_ACCOUNTING_VERSION=2` 在窗口、阶段和最终结果缓存层门控。
- IBS 偏置协议已升级为 `IBS_BIAS_PROTOCOL_VERSION=2`：未收敛 warmup 不再进入生产，生产期不再更新 `f_k`，旧协议缓存被拒绝复用；但“连续通过”仍有重复计算同一批统计量的缺口，因此 v2 尚不能视为最终可发布协议。
- λ 自动加密在 `min_overlap` 未改善或变差时会立即熔断，不再无条件跑满五轮。

---

## 建议修复优先级

1. ~~P0：实现自相关子采样，修正误差棒。~~ 已修复（见「最高优先级」第 1 条）。
2. ~~P0：使用真实 overlap matrix，替换假的 `converged` / `min_overlap_proxy`，并让 pipeline 门控。~~ 已修复（见「最高优先级」第 2 条）。
3. ~~P1：修正 VDW 窗口拼接 offset 权重或改为真正 global MBAR。~~ 已修复（offset 改为逆方差加权并传播 offset 方差，见「物理/建模状态」第 4 条）。
4. ~~P1：给 decharging 分支加 `lambda_vdw == 1.0` 断言。~~ 已修复（`compute_u_kn`；断言原用 `np.allclose(..., atol=1e-6)` 未设 `rtol=0`，`lambda_vdw=0.99999` 仍会因默认 `rtol=1e-5` 而通过，现已改为 `rtol=0.0, atol=1e-6` 使其真正严格；`REMDManager.__init__` 同名判定仍待加固）。
5. ~~P1：处理 LJ tail/LRC correction，或明确作为外部非 APBS 项处理。~~ 默认 ACE/`dual_lambda` 路径已修复（手算解析尾项，见「最高优先级」第 3 条）。
6. ~~P2：修正 unused `run_full_abfe_loop` 或删除。~~ 已修复（键优先级已与主流程对齐）。
7. ~~P2：修复 parallel stages 旧 JSON 误读风险。~~ 已修复（写入前清空 stage1.json/stage2.json）。
8. ~~P2：统一角度力常数上界。~~ 已修复（统一为 1000 kJ/mol/rad²）。
9. ~~P3：澄清 final JSON 的 Boresch 字段，修正 analyze-only 口径。~~ 已修复（新增 `boresch_correction_already_included_in_*` 标记；analyze-only 优先复用正式 `final_results.json` 并补上 APBS 修正）。
10. ~~P3：补 ORB 依赖前置检查和 DEXP 异常记录。~~ 已修复（`OrbBoreschEstimator` 加 `HAS_ORB` 检查；DEXP 逐帧异常记录原因）。
11. ~~P1：给 `BeutlerSoftcoreBuilder`（`--decoupling single_lambda`/REMD 路径）补上跟 ACE 路径同等的 LJ 长程色散尾项修正，消除两条解耦路径的不对称。~~ **2026-07-15 已修复**：`TraditionalMBARAnalyzer.compute_u_kn` 现在用同一套 `_lj_tail_lrc_coefficients_kj_mol`（switching+softcore-aware）计算逐 λ 系数，两条路径不再不对称，见"同日复审第八轮"。
12. ~~P1：阻止 LJ/WCA/IBS 旧口径窗口、stage 与 final 缓存静默复用。~~ 当前已由 `WCA_ACCOUNTING_VERSION=2` + `IBS_BIAS_PROTOCOL_VERSION=2` 在窗口/IBS state/stage/final 多层 fail closed；更一般的完整 Hamiltonian 指纹仍见第 14、19 条。
13. P2：`README.md`/`README_cn.md`/`README_en.md` 里若干处仍写着 LJ tail/LRC "未实现"，需要同步改为当前实际状态（部分修复，见「最高优先级」第 3 条）。
14. P0：给顶层 `_top_level_protocol_key`/阶段级 `_stage_protocol_key` 补齐完整 Hamiltonian/λ/窗口/采样参数/温度/softcore/Boresch 字段，改成对规范化 JSON 做 SHA256，而不是几个零散字段的相等比较。见「最高优先级」第 5 条。
15. P0：`_run_dual_lambda_optimization` 预优化失败不能静默降级缓存为线性路径——要么让异常传播，要么显式标记 `optimization_failed=True` 并禁止写缓存。见「最高优先级」第 5 条。
16. P1：`ibs_engine.py::collect_energies` 的 `frame_finite` 检查补齐 `np.isinf`（目前只查 `np.isnan`），避免 +Inf/-Inf 帧进入 MBAR。见「最高优先级」第 5 条。
17. P1：Stage2 λ 放置改为真正的热力学长度 `g(λ)=β²Var[∂U/∂λ]`、`s(λ)=∫√g dλ` 等距放点，再按实测相邻态 overlap/ESS 划窗口，替换现有 `optimize_stage2_vanishing`（`abfe_preoptimizer.py:990`）里 `Var[U_softcore]+log1p+15%截断` 的代理指标与固定 `pts_per_window=6/overlap=2` 划窗；`compute_2d_metric_grid()` 已有类似热力学度量思路，但目前只接给 `optimize_2d_geodesic_path`，未接入默认 `dual_lambda` 路径。
18. P1：`abfe_config.json` 的 `enable_lambda_refine=false` 是刻意关闭的（避免覆盖此前用 `repair_stage2_tail_gap_local.py` 对 stage2 窗口 12-18 做的手动局部修复）——重新设计 λ/窗口逻辑（上一条）前必须先决定如何处理这个手动补丁状态，不能只是翻开关。
19. P0：Stage2 预优化缓存（`preopt_dual_vanishing.json`/`preopt_dual_decharging.json`）目前只按初始 `n_states` 判断是否复用，没有方法版本号/production Hamiltonian 签名/softcore 参数签名；自动 overlap 修复把 18 态加密到 20/22 态后，进程一重启就会因态数不等于初始 18 而丢弃修复路径。应写入完整协议指纹；匹配且 provenance 为 `auto_repair_by_overlap` 时用缓存长度更新 stage state 数。
20. P2：`diagnose_softcore_cv_values`（`ibs_engine.py:1317`）诊断打印同步为生产口径 `base={0,2,3,5}`/`bias={1,4}`。
21. P2：补齐 WCA/LRC/IBS 记账相关回归测试（base+bias 精确重构、Group4 纯采样偏置、LRC 只进 MBAR target、缓存版本门控、独立连续通过计数、生产期不调用 `update_weights()`、Inf 帧拒绝、优化失败不被静默缓存），见「最高优先级」第 5 条。
22. P0：修复 IBS 严格预热的连续通过计数。只有本轮真实执行 `update_weights()` 并产生新 `f_k` 时才允许检查并递增 `consecutive_pass_count`；`f_history < 2` 时不得把 `f_delta_ok` 默认为 True。修复后将 `IBS_BIAS_PROTOCOL_VERSION` 从 2 升到 3，确保任何可能由假收敛放行的 v2 数据失效。
23. P1：收紧 `_is_overlap_failure()`：删除“任意 `converged=False` 都可插点”的分支，只在 `min_overlap < threshold` 且逐窗口 ESS 明细完整时进入 λ 加密。偏置 warmup、MBAR solver、缺文件/NaN 应各自保留独立失败类型。
24. P1：严格预热目前在 full-bias 段每 500 步采一个能量点、每 10 点才更新一次权重，40,000 步最多只有约 8 次新统计更新。修复第 22 条后需要用真实窗口评估这是否足够；若大量窗口安全地硬失败，应优先提高收集/更新频率或 warmup 上限，而不是放宽收敛判据或恢复自动插点。

---

## 可选清理项

| 位置 | 问题 |
|---|---|
| `abfe_core.py` | 局部变量 `top` 与 mdtraj topology 别名撞名，当前无害 |
| `abfe_core.py` | `scan_boresch_1d_pes` 中保留不可达分支 |
| `abfe_core.py` | 章节编号注释重复 |
| `ibs_engine.py` | 文件中部重复 import |
| `abfe_pipeline.py` | `generate_overlapping_windows` 重复导入 |
| `abfe_pipeline.py` | `_setup_boresch_params` 是未调用桩函数 |
| `runabfe.py` | 若干 import 未使用 |
| `runabfe.py` | GROMACS 路径硬编码兜底不便携，但已排在显式参数和环境变量之后 |

---

## 仍需动态验证

- CPU 极小步数端到端 smoke test：系统构建、Boresch、复合物腿、溶剂腿、最终 assembly。
- 目标 GPU 上完整 ABFE 实跑：确认修正后的 overlap、误差棒、Delta G 数值。
- `--parallel-stages` + `--resume` 在真实多 GPU 环境下验证。
- **LJ tail/WCA/IBS 冻结协议的联合端到端验证（尚未获得有效最终结果）**：`test_lrc_interaction_group_compat.py` 已验证解析 LRC 公式本身；旧 `output_lrc_fix/` 的真实运行也成功暴露并记录了 WCA/IBS 问题，但其 Stage2 生产数据是在“warmup 未收敛仍放行、生产期持续更新 `f_k`”的旧协议下产生，不能用于最终 ΔG 数值比较。下一次有效验证必须先修复第 22 条的独立连续通过计数并升级 `IBS_BIAS_PROTOCOL_VERSION`，然后让版本门控自动拒绝旧 Stage2 窗口。验证目标：(1) 每个窗口 warmup 的逐态概率与 `f_k` 稳定性由**独立权重更新**连续通过；(2) 生产期 `f_k` 完全不变；(3) `min_overlap≥0.05`，若不足则只有局部、可解释的 ESS gap 才触发一次加密；(4) 相同 λ 在不同窗口划分下不再依赖 window center；(5) 再比较修复前后 `total_delta_G_complex_kJ_mol`。
- **旧多轮加点日志的结论已更正**：`18→19→20→21→22→23` 与 `18→22→25` 不是“Stage2 初始 λ 密度不足已被自动修复”的证据；它们对应多个 `bias_warmup=hit_step_cap_unconverged` 窗口，`min_overlap` 还出现 `0.007973→0.003479` 的恶化，是错误失败分类与非平稳偏置采样的故障证据。当前熔断器可阻止继续恶化，但不能替代 warmup 收敛。
- 若启用 APBS：`apbs_correction.py` 已重写为 Rocklin/RIP 方法，需要用真实带电配体（当前 Atenolol 上下文为中性，工具会拒绝跑）+ 真实膜 dielectric/lipid-charge maps + 实际 APBS binary 跑通新版输入，并把 correction 的热力学含义写进 provenance；同时建议找一个已知答案的测试案例核对 NET/RIP/EMP/DSC 具体系数是否抄对。
