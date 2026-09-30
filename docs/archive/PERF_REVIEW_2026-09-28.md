# 性能静态审查（2026-09-28）

> **一次性审核的原始记录，不是待办清单。已归档（2026-09-28）**——26 条当天已逐条对照源码核过，
> 核实后的版本在 [TODO.md](../TODO.md)：默认路径上成立的 → §2.6 `BORESCH-AUTO-01`、`PERF-0928` [P2]；
> 非默认路径 / 驳回 → §3.7 `PERF-0928-COND` [P3]。
> ⚠️ **下面是审查代理的原文，未按核实结果修改**：半数以上「在不在默认路径」判错或优先级偏高
> （例：A3 实际只在配置没写 `boresch_source` 时触发；A6 频率说错；B4、C7 是死代码）。以 TODO 为准，别照本文修。

审查输入：工作区 `/home/ruigengji/ABFE_IBS/ABFE_IBS`，源码 HEAD `e576a54`（与 2026-09-26 报告同一提交，其后工作树只有文档改动）。仅只读静态审查；没有修改项目代码，没有运行 pytest、GPU 采样或 profiler。所有行号对应当前工作区源码；性能结论均为源码结构计数，未含实测耗时排名。

与 [ABFE_IBS_code_review_2026-09-26.md](CODE_REVIEW_2026-09-26.md) 的关系：该报告的 PERF-01～05、Stage1/离线分析三节、QUALITY-01～07 不在本文重复。本文全部为**新增**发现，共 26 项（默认主线路径 14、条件分支 5、离线与辅助工具 7）；叠加 9-26 已记录的 PERF-01～05，全库静态可见的性能问题约 31 项。

计数基准同 9-26：CUDA，production，n_steps_per_window=250000，steps_per_update=500，enable_early_stop=false，outer_lambda_local_residual_ibs=false；Stage2 约 30 窗/腿，窗口 K≈4–5。

---

## 1. Release 距离评估

按 README 的 clone-and-run 发布模型，主线（中性、非共价 dual-lambda ABFE）**不含口袋水新特性**时，剩余事项：

1. **两个已知 P1 正确性**（9-26 评审，均未修）：LCR 账本未绑定阶段协议（abfe_pipeline.py:14694）、final/stage 缓存可绕过生产盒校验（abfe_pipeline.py:17102）。
2. **本轮新发现一个 P1 行为问题**（见 A3）：默认 `--boresch auto` 链在 2026-09-24 切到 `openfe_fixed` 后，对无缓存的新体系很可能枚举数分钟后直接 RuntimeError。静态推断，建议下批生产前拿一个无缓存配体实测确认。
3. **性能整改**：9-26 已记录 PERF-01～05；本轮新增 26 项，其中默认路径 P1×2、P2×6。性能不挡科学正确性，但 controller 重复求解和窗口初始化开销直接吃生产吞吐。
4. **口袋水 v2 采样器**：[PLAN_pocket_water_swap_sampler_2026-09-27.md](../design/PLAN_pocket_water_swap_sampler_2026-09-27.md) S0/S1 完成，S2–S7 全部未开工，S5（改生产哈密顿量）需用户拍板。**release 范围含口袋水时这是最大的一块，数周级；不含则主线收尾是一个中等批次。**

---

## 2. A 组：默认主线路径

### A1. [P1] Controller 每轮全量重读窗口 + 全量 MBAR 重解；退出路径"双解"后丢弃前者

位置：[abfe_pipeline.py:13640-13647](../../abfe_pipeline.py)（退出双解）、[:14243-14349](../../abfe_pipeline.py)（`_solve_merged_segments_if_any` 重 glob/重读/重解，:14258/:14317/:14327）、[:14039-14059](../../abfe_pipeline.py)（每轮 `_persist_inprogress_stage_result`）、[:12113-12118](../../abfe_pipeline.py)（max_iterations=40）。

每轮 ANALYZE/merge 都从头 `load_ibs_window_outputs_from_dir` 重读所有窗口 npy/json 再重新求解，进程内无缓存；未变化的窗口也被重读、重 hstack（:15506）、重解。退出路径更实在：`run_once(...)` 完整解一遍后，`_solve_merged_segments_if_any` 有合并段时 `_final = _merged` 直接把刚算完的全路径解扔掉。每轮解是分钟级 CPU × 最多 40 轮 → 分析期可烧数小时纯重复计算。

修法：按 (window_idx, 文件 mtime/size) 缓存窗口输出；暖启动 f_k；无合并段或合并必成时跳过第二次解。注意退出全路径解的注释写明它承担"逐窗支撑/σ/overlap 落盘"职责，重组时保留该诊断产物。

### A2. [P1] 每窗口无条件序列化整份 System 存 `_common_system_xml`，默认路径零消费

位置：[ibs_engine.py:5845](../../ibs_engine.py)（构建处）、[:5214-5260](../../ibs_engine.py)（序列化器）、消费点 [:18146-18160](../../ibs_engine.py) 与 [:18217-18232](../../ibs_engine.py) 均被 `legacy_repair` 门控（[:8861-8872](../../ibs_engine.py)：仅已废弃的 `legacy_mutating` 返回 True）。

默认 `non_mutating_v1` 下每个 VDW 窗口都做一次 System 克隆 + 全量序列化成几十～上百 MB 字符串，且驻留到窗口结束（:20424），没有任何默认消费者。修法：`should_run_legacy_repair(repair_policy)` 为 False 时直接跳过，或惰性构建。

### A3. [P1，行为+性能] 默认 Boresch auto 链的 kr 门会拒绝全部候选

位置：[runabfe.py:3826-3833](../../runabfe.py)（复合物腿默认启用 Boresch，无锚点预设时 `boresch_source="auto"`）、[:4499-4502](../../runabfe.py)（auto 调用不传 `max_kr`）、[abfe_core.py:10365](../../abfe_core.py)（默认 max_kr=2000）、[:10015-10022](../../abfe_core.py)（`openfe_fixed` 下 `_apply_hybrid_filter` 对所有候选返回固定 kr=4184）、[:10589](../../abfe_core.py)（`if ks["kr"] > max_kr: continue`）、[:10636-10640](../../abfe_core.py)（finalize 同门槛）、[runabfe.py:4505](../../runabfe.py)（空结果 → `RuntimeError("自动 Boresch 估算失败，未找到合格候选")`）。

2026-09-24 起力常数默认 `openfe_fixed`（abfe_core.py:7968，`OPENFE_FIXED_KR_KJ_PER_MOL_NM2 = 4184.0`，:7960），该固定 kr 恒大于 max_kr=2000：每个候选在校验门被拒 → `validated`/`fallback_pool` 全空 → 无缓存的新体系在付完整个枚举+MACE 力评估成本后直接 RuntimeError。`estimate_multiple_anchors_from_trajectory` 开头（abfe_core.py:10357-10430）没有 openfe 模式旁路，已核实。

已有 Boresch 缓存（runabfe.py:4470 起的缓存命中分支）或显式锚点预设的体系不走此路——这解释了 09-24 后未炸。**静态推断，未运行验证**；修复方向：force_constant_mode 为 openfe_fixed 时跳过/钳制 kr 门（力常数是设置项，不该用拟合上限筛），或把 max_kr 从 config 传入。

### A4. [P2] 每窗口第二份全 System 序列化 + sha256 做 checkpoint manifest

位置：[ibs_engine.py:15683-15704](../../ibs_engine.py)（`win_sys_xml = openmm.XmlSerializer.serialize(win_sys)`，注释自认"几十 MB 的瞬时峰值"）、[:11747](../../ibs_engine.py)（sha256）、[:11885](../../ibs_engine.py)（production manifest 复用）。

在 `attempt_checkpoint_restore` 判断（:15705-15716）之前就做；fresh run 上 manifest 到 teardown 才被消费。修法：仅在 resume 或首次 manifest 写出时序列化，字符串与哈希一起缓存。

### A5. [P2] 每窗口 CV 构建浪费：模板力用后即弃 + K+1 次逐粒子单位换算

位置：[ibs_engine.py:5706-5717](../../ibs_engine.py)（`cv_template` 完整构建 30–70k addParticle+排除表后只读 5 个元数据字段即丢弃）、[:4772-4776](../../ibs_engine.py)（逐粒子三次 `.value_in_unit()`，×每态）、[:5777-5820](../../ibs_engine.py)（K 态循环逐态重建 + 每态重建排除元组 set 验证）、[:6342](../../ibs_engine.py)（shadow-coul 同款）。

每窗口 K+1 次力构建 × 30–70k 粒子 × ~30 窗 = 百万级 SWIG 单位换算调用。修法：粒子参数预转 float 三元组复用；模板元数据从首个真实 CV 力取；排除验证改长度检查。

### A6. [P2] warmup 续采验证期每 250 步重写全量 ibs_state JSON

位置：[ibs_engine.py:17167-17178](../../ibs_engine.py)（每 batch：OpenMM checkpoint + manifest + `save_ibs_state`；batch=`IBS_WARMUP_FRAME_STRIDE_STEPS=250`，:8068）、[:10238-10378](../../ibs_engine.py)（`save_ibs_state` 全量 json.dump f_history+tmbar_history）、[:1286-1301](../../ibs_engine.py)（每写 2×fsync）、legacy 同款 [:18306-18312](../../ibs_engine.py)。

OpenMM checkpoint 是抢占恢复所需该写的；ibs_state JSON 是 O(累计历史) 的重复序列化，一个 rung 200–1200 个批次。修法：仅在冻结/状态迁移时写 ibs_state，或按 N 个 batch 节流。

### A7. [P2] 每次 resume 把全部 replica DCD 完整读到 EOF 校验，指纹失配路径读两遍

位置：[abfe_pipeline.py:2138-2290](../../abfe_pipeline.py)（`_is_traj_valid` 全量 `_consume_to_eof`）、[:2718-2725](../../abfe_pipeline.py)（`_all_remd_trajs_valid` 每 replica 一次）、调用点 :6450、:6538、:7793、:7823、:20260；[runabfe.py:1091-1116](../../runabfe.py)（预平衡 DCD 同款）。

代码注释自认"decharging 的 resume 路径整套走两遍"（:2182-2185）。12 replica × 数 GB IO，共享文件系统上分钟级；`runabfe.py:1011-1053` 对 checkpoint 已有 (path,size,mtime_ns) 记忆化，DCD 没有；新的 NSET 头检查（:2270-2289）已覆盖截断场景。修法：头 NSET+文件大小算术+末记录读取，并记忆化。

### A8. [P2] traditional 模式每次调用重新解析 GRO/TOP 并重建 System

位置：[runabfe.py:5969-5974](../../runabfe.py)（无条件 `build_system_from_gromacs`），对照 IBS 主线 native cache（:7650、:7763）与该函数 docstring（:3213"仅用于首次创建缓存"）。

`--mode traditional` 的每次运行（含纯 resume/analyze）都付几十秒解析+createSystem 和多 GB 瞬时内存。修法：复杂腿走 `save_native_system`/`load_native_system` 同套缓存身份检查。

### A9. [P2] f_k recalibration 探针每轮重读+重解所有窗口，无记忆化

位置：[abfe_pipeline.py:15147-15175](../../abfe_pipeline.py)（每窗 `np.load` sampling_states :15162、`recalibrate_f_k_from_production` :15168），由 controller 各轮驱动（:12796-12819、:12878-12898）。

每轮对所有窗口重跑 MBAR+统计低效，包括输入文件与上一轮逐字节相同的窗口。修法：按 (segment dir, window_idx, energies mtime/size) 缓存 f_k/shift。

### A10. [P2] Boresch 枚举内部的逐候选/逐帧浪费（默认 openfe 模式也白算）

位置：[abfe_core.py:10241](../../abfe_core.py)（每候选把整条轨迹拷 float64，数百 MB/候选，只用 6 列）、[:10249-10257](../../abfe_core.py)（逐帧 Python `boresch_dihedral_rad` ×3）、[:9977-10007](../../abfe_core.py)（`_compute_geom_gradients` 每帧重建 mdtraj Topology/Trajectory + 24 次数组拷贝，由 `_quick_validate` :9906 每候选调 50 帧）、[:10069-10083](../../abfe_core.py)（MACE 口袋力对固定 `pocket_sel` 逐候选重算，力只依赖坐标）、[:10161-10162](../../abfe_core.py)（`_finalize_candidate` 每候选全轨迹 superpose，`pocket_sel` 恒定）。

在 `openfe_fixed` 模式下 `q_data/Fq_data` 算完全程不被读（`_apply_hybrid_filter` :10015-10022 短路返回常量）——配合 A3 等于整段白算。整簇集中在 `OrbBoreschEstimator`：一次 `mdtraj.compute_dihedrals` 向量化、Topo 缓存到类级、力逐帧算一次缓存、superpose 只做 frame 0。

### A11. [P3] 每 update 对 K 个 f_k 做 getParameter 轮询防篡改

位置：[ibs_engine.py:19583-19599](../../ibs_engine.py)（生产循环每 update）；同款 :17205-17211、:17294-17300、:19792-19798。K 次 SWIG 调用+字符串查找每 update（每 500 步），不进 loop_timing 账。该值只能经 `setParameter` 变，按快照（每 100 updates）校验即可。

### A12. [P3] 每 100 updates 对刚写盘的 npy 重读+sha256

位置：[ibs_engine.py:19736-19740](../../ibs_engine.py)、[:19753-19763](../../ibs_engine.py)。`_enqueue_window_snapshot`（:19699）刚写完的能量文件被 `_window_data_metadata` 再读回再哈希。`_array_content_fingerprint`（:21836）已支持内存数组直接哈希。

### A13. [P3] 坐标旁车每次 flush 全量 np.stack 重写（O(n²) 累计）

位置：[ibs_engine.py:1157](../../ibs_engine.py)（全系统位置 Quantity 再单位换算后切片）、[:1164-1177](../../ibs_engine.py)（`_save_coords_sidecar` 每次 flush 对整个累计 list stack 覆盖写，每 100 updates，:15027-15030）。9-26 PERF-05 的邻接细节，此处补 O(n²) 与逐帧全系统单位换算两点。修法：float32 直接切片免单位换算，分块存储只写增量。

### A14. [P3] 离线 u_kn chunk 内盒体积落在 k 循环内，K 倍冗余

位置：[ibs_engine.py:3256-3260](../../ibs_engine.py)。体积只依赖帧，提到 k 循环外即每帧一次 3×3 行列式。lj_tail_lrc_coeff 存在时触发（非 PME 腿）。

---

## 3. B 组：条件分支（默认关闭/罕见路径）

### B1. [P3] LCR residual 臂每帧多一次 getCollectiveVariableValues

位置：[ibs_engine.py:9129](../../ibs_engine.py)（live CV 一次返回全部 K 个）、[:9421-9423](../../ibs_engine.py)（`get_residual_basis_energy` 内 :6837 再查一次，而残差值就在首次返回数组的 `_residual_basis_cv_index` 位）。`residual_enabled` 时每帧生效；与 PERF-03 不同调用点。修法：collect_energies 从首次数组取残差值。

### B2. [P3] endpoint bank 每次加载全量文件重算 SHA-256

位置：[abfe_pipeline.py:2472-2506](../../abfe_pipeline.py)（读侧 :2503-2505 对所有 npz 重哈希）。`stage2_independent_endpoint=True` 时每次 resume/analyze 生效，bank 可达数十至数百 MB/文件。修法：写侧哈希一次，读侧按 (size, mtime_ns) 便宜校验。

### B3. [P3] `--only-complex-charging` 对同一 u_kn 二次全帧 MBAR

位置：[runabfe.py:6444-6505](../../runabfe.py)（`_charging_mbar_crosschecks` 重 np.load :6451、raw 全帧 MBAR :6455、相邻 BAR 链 :6463-6474），生产解之后立刻在同进程跑（:6908）。修法：持久化 crosscheck 结果按 u_kn 哈希跳过，或从已解 MBAR 对象取重叠数据。

### B4. [P3] fixed-H 探针每条边整文件 fsync+dirsync 重写

位置：[abfe_pipeline.py:10173-10179](../../abfe_pipeline.py)（每条边 `_atomic_write_json(probe_file, ...)`），`_atomic_write_json` 的 2×fsync 在 [ibs_engine.py:1275-1298](../../ibs_engine.py)。共享文件系统上 10–100+ ms/次 × 边 × 窗 × 轮。热缓存可批量 fsync，完整性标记保留全持久化。

### B5. [P3] legacy anchor 模式逐列逐帧 Python unwrap

位置：[abfe_core.py:13747-13755](../../abfe_core.py)（`periodic_unwrap` 双层 Python 循环）、[:11579-11588](../../abfe_core.py)（`assess_boresch_harmonicity._unwrap` 同款、无条件跑）。仅 legacy anchor 模式（`anchor_mode != "openfe"`）代价显著；`np.unwrap(axis=0)` 一行替代。

---

## 4. C 组：离线与辅助工具

### C1. [P1] pocket_water_s1_probe.py：逐水全量 NonbondedForce 参数重传 GPU

位置：[tools/diagnostics/pocket_water_s1_probe.py:426-439](../../tools/diagnostics/pocket_water_s1_probe.py)（`water_bu()` 对每个水两次 `setParticleParameters`+`updateParametersInContext`）、调用点 :455（最多 600 个 bulk 水）与 :461（每个口袋水），逐帧。

`updateParametersInContext` 每次重传全 ~10⁵ 粒子 charge/sigma/epsilon → 每帧数百次全设备参数传输。修法：`NonbondedForce.addParticleParameterOffset` + 每水一个全局参数开关，`setParameter` 切换零重传。该脚本是口袋水线 S1 探针工具，S2 采样器开工前值得先改。

### C2. [P1] apbs_correction.py：~17M 值 DX 网格逐 token 正则解析

位置：[apbs_correction.py:564-586](../../apbs_correction.py)（`_read_dx_values` 整文件 `read_text` ~250 MB + `re.finditer` 逐 float 入 Python list ≈ 0.5+ GB）、:595（`statistics.fmean` 再遍历），每网格分钟级，collect 每快照 ×3 网格（:624-632）。修法：切数据段后 `np.fromstring(..., sep=' ', count=expected)`。

### C3. [P2] apbs_correction.py：独立快照串行跑 APBS 子进程

位置：[apbs_correction.py:544-560](../../apbs_correction.py)（逐个 `subprocess.run`）。快照间完全独立，257³ 网格单个数分钟。修法：ThreadPoolExecutor + `--jobs`。

### C4. [P2] local_residual/refit.py：岭网格对 ridge 无关的二次型重复计算

位置：[local_residual/refit.py:293-313](../../local_residual/refit.py)（13 个 ridge 候选 × fold，每个重算 `_quadratic_form` :217-247，而 M,b 只依赖 train 子集）、:306（test 侧同）。修法：每 fold 预计算 (M, b, c) 复用整条岭网格。

### C5. [P2] local_residual/refit.py：O(frames × states) 串行全 PME 单点能

位置：[local_residual/refit.py:507-516](../../local_residual/refit.py)（逐帧 `setPeriodicBoxVectors`+`setPositions`+2×`setParameter`+能量，×~9 态），全量 `mdtraj.load`（:570）。LCR auto-refit 主线接线（runabfe.py:7220-7237）。修法：worker 池并行；`md.iterload` 流式。

### C6. [P3] local_residual/openmm_plugin.py：每窗口重建逐字节相同的力 XML

位置：[local_residual/openmm_plugin.py:445-455](../../local_residual/openmm_plugin.py)（`force_factory` 每窗口 repr 全部 float 再反序列化，:336-387），由 ibs_engine.py:14964 每窗口调。修法：缓存 XML 字符串或反序列化模板逐窗克隆。

### C7. [P3] local_residual/mace_graph.py：逐原子 `.item()` 造成逐原子 GPU 同步

位置：[local_residual/mace_graph.py:335-345](../../local_residual/mace_graph.py)（3 万原子的 CUDA 张量逐原子 `.item()`），仅 EXP-012 MACE teacher/smoke 路径。修法：`torch.nonzero` 一次取完；`environment.py:125-135` 的 manifest 校验（含 SHA-256 重哈希）提到 per-graph 路径外。

---

## 5. 已查过、明确不算问题的区域

- **REMD Stage1**（ibs_engine.py:25307-26403）：交换代价是物理必需（每相邻对每轮 4 次全能量）；replica Context 平衡→生产有意常驻；步进分块无每步状态查询。
- **MBAR/去相关数学**：`collapse_segments`（multi_segment_analysis.py:110-202）为必需交叉矩阵，无 O(frames²)；LCR jackknife 每窗 1+n 次全 MBAR 是设计规定且账本缓存（abfe_pipeline.py:14871-14874）；`subsample_series_by_autocorrelation` 走 pymbar fast=True 向量化（abfe_core.py:7786-7847）；`ChunkedMBARAnalyzer`、`_build_mbar_compatible` 无逐帧循环。
- **abfe_preoptimizer**：pilot 循环受 MD 积分主导且快照跨轮复用；窗口划分是正经 DP（:1583+）；2D 度量网格有 geodesic_path.json+协议指纹缓存（abfe_pipeline.py:19790-19865）。
- **协议键哈希**：`_system_xml_hash`/`_topology_hash` 已恒返 None（abfe_pipeline.py:720-800），不再序列化 System。
- **analyze-only**（runabfe.py:5287-5458）：仅在 stage checkpoint 缺失时才重解，其余读缓存。
- **`_system_xml_hash` stdout tee**：持久句柄，无逐行开关。
- **w34_water_tracking.py**：无实质性能问题（ghost_counts 广播、每 ps 一次，被 MD 步支配三个量级）。
- **内存增长**：sampler 历史按窗有界；tmbar_history ≤15 封顶；probe bank/端点 Context 逐态释放。唯一无界驻留是 A2 的 XML 字符串。
- **排除的旧候选**：PYMBAR_DISABLE_JAX 已设；9-26 PERF-01～05 不重复；串行 u_kn fallback 每 chunk 建 Context（ibs_engine.py:27552/3171）9-26 已记录，不另计。

---

## 6. 建议动作顺序与验证边界

1. **先实测再改**：`loop_timing_s`（[ibs_engine.py:20171](../../ibs_engine.py)，integration_s/guard_s/cv_probe_s/ledger_io_s）已有入口，先确认 A1（分析期 CPU）、A2/A4（窗口边界停顿）的实际占比；该计时不覆盖 Context 构建、warmup 全部与分析期。
2. **A3 优先实测确认**：一个无缓存配体跑一次 `--boresch` 即可证实/证伪——这不是性能问题，是默认链可用性问题；确认后修复应与 9-26 的两个 P1 同批处理。
3. **实施顺序建议**（改动成本×可验证性）：A2、A13、A4 → A1（含缓存与暖启动，保留诊断产物）→ A6、A7、A8 → A5、A10；PERF-03 类深度改动仍按 9-26 顺序。C 组工具按对应工作线开工前顺手改（C1 在口袋水 S2 前）。
4. **边界**：本文为静态审查，所有性能项均为源码结构计数（频率×单次成本的量级推断），没有实测墙钟排名；优化前后对照须同体系同协议，且不得以减少 Python 调用数代替物理账本一致性校验。
