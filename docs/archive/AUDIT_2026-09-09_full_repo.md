> **一次性审计的原始记录（2026-09-09），不是待办清单。**

> **已归档（2026-09-12）。** 那轮审计已结案：62 条候选 → 52 条已修、
> 7 条「判定不改」登记在 [TODO.md](../TODO.md)《AUDIT-2026-09-09》。
> 本文是**原始记录**，不是待办；那 7 条的理由别重新论证。
> ⚠️ 52 处改动**无一上过 GPU**，需要真机复验的三处见本文 §4。

---
>
> ⚠️ 本文件记录的 52 处改动**没有一处上过 GPU**，见 §4。
>
> 62 条候选里 **52 条已修**（§1），**0 条代码工作待办**（§2），
> 7 条**有意不修**（§3）已登记进 [`TODO.md`](../TODO.md)《未关闭的代码缺陷》
> `AUDIT-01`~`AUDIT-07`。

---

# 全仓审计 2026-09-09

> 触发：OpenFF benchmark 一批 16 个 run 大面积失败 + 用户看不到 stdout。
> 方法：9 个并行只读审计代理按行段切分覆盖 **77,588 行**（全部 Python 主文件，无重叠无空档），
> 外加本会话自己的定位。共 62 条候选，去重后 45 条判定值得处理。

**本文件是清单，不是结论稿。** 每条都带 `file:function` + 具体失效场景 + 置信度。
行号会随后续改动漂移，**以函数名为准**。

---

## 0. 两个现场

### 0.1 十个 run 在 Stage 2 窗口 0 偏置预热瞬间 NaN

```
[pilot TI 热启动] 窗口 0 f_k 初始值（非冷启动 0.0）: [-36.052, -18.026, 0.0, 18.026, 36.052] kJ/mol
[偏置预热] 开始... (从零缓慢加载偏置力)
→ openmm.OpenMMException: Particle coordinate is NaN.
```

根因见 §1.1。种子本身**没有错**：72 kJ/mol 的跨度等于 `subdomain_free_energy_kJ_mol[0]`，
完美等距是因为 `free_energy_densified_v22` 本来就按等 ΔF 布点、而种子又从同一条插值 TI
曲线上读回来（同义反复）。`pilot_ti_seed_trust_diagnostics()` 拿真实 preopt 数据实跑是
`trustworthy=True`（max_sem 0.85 对门槛 2.0、propagated 0.047 对 5.0），**接上去也拦不住**——
它是精度门，不是 NaN 闸门。

### 0.2 `std::bad_alloc`，93 GB 空闲、`ulimit -v unlimited`、单进程

```
开始离线能量重算 | 400 帧 × 8 态 | workers=16 | chunk_size=25 | cpu_threads_per_worker=2
terminate called after throwing an instance of 'std::bad_alloc'
```

排除过的：崩点本身（拿该 run 缓存离线重放 `_prepare_pme_coulomb_leg_system` 8 次，峰值
708 MiB，源 System 未被改动）、体系本身（30710 粒子 / 5 个力 / 7.4 MB，往返正常）、并发
（当时只有这一个进程）、ulimit。

**⚠️ 根因不是 §1.3。** 真凶是 **pymbar 的 JAX 后端啃宿主内存且不归还**
（λ 表/帧数一变形状就变 → 每次解重新 XLA 编译、缓存永不命中；编译产物 + BFC 池从不释放）。
实测 K=12 × N=9600：JAX 6.30 s / 887 MiB vs numpy 0.37 s / 120 MiB，f_k 差 8.9e-16。
宿主堆被解算器啃光之后，由**下游随便哪次大分配**替它抛错——`deserializeSystem` 只是恰好
排在后面的那一个。修法是 `abfe_core.py` 里 `os.environ.setdefault("PYMBAR_DISABLE_JAX", "1")`
（必须在任何 `import pymbar` 之前），`tests/test_import_time_side_effects.py` 钉住顺序。
**这条不是本次审计定位的，是另一路查出来的。**

§1.3 那批内存改动仍然成立、也确实降了峰值，但它们**不是**这次 `bad_alloc` 的成因——
本文件早前把 §0.2 指向 §1.3 是错的归因，此处更正。

**后续（同一步骤，JAX 关掉之后）**：同一行日志之后改为**静默挂死**，无任何 traceback。
机制是 `multiprocessing.Pool` **检测不到 worker 死亡**——子进程被 SIGKILL ⇒
`imap_unordered` 永不返回；`initializer` 在子进程里抛异常 ⇒ Pool 无限重生失败的 worker。
两种都不给父进程任何输出。已改成带时限取结果（`ABFE_UKN_CHUNK_TIMEOUT_S`），
超时抛错并回退单进程，见 §1.9。

### 0.3 stdout 消失

`pipeline.log` 里那 6 分钟一行不少，`launch.log` 里一行没有。见 §1.2。

---

## 1. 已修（52 处）

### 1.1 NaN 根因 — `ibs_engine.run_all_windows` 偏置爬坡

- **`ibs_engine.py` `IBSWindowManagerDualLambda.run_all_windows`** — `ramp_stages` 原来停在
  `0.7`，随后直接把 `bias_scale` 设成 `1.0`：**最后也是最大的一次增量既没有弛豫档、也没有
  分段回退保护**。跳完之后第一段动力学是裸的 `guarded_step`（偏置学习循环）——因为
  `f_k_warm_started=True` 时中间那段自举 TI 采样整块跳过。配上 ±36 kJ/mol（≈29 kT）的种子，
  Group-1 的 log-sum-exp 塌到单态、施加的偏置力就是那个态的完整 softcore 力。
  → 补上 `(1.0, 2000)` 档，整条爬坡每档都受 `step_with_chunk_rollback` 保护。**置信度 中-高**
- **同一 reset 块** — 只清了 `energy_buffer`/`energy_history`/`bias_history`/`base_energy_history`
  四条，漏了 `sampling_state_energy_history` / `residual_basis_history`。这两条与
  `energy_history` 受 `collect_energies()` 里**同一个** frame_finite 门同步 append，而
  `_append_tmbar_batch_from_buffer` 只校验尾部长度（尾部检查测不出前面已经错位）。
  → 一并清。**置信度 高（latent）**

### 1.2 stdout 消失 — `abfe_pipeline._StdoutTeeToFile.write`

文件那份每行 `_append_line()` → `flush()`，转发给真 stdout 的 `self._original.write(s)`
**从头到尾不 flush**。stdout 重定向后是 8 KB 块缓冲 ⇒ `pipeline.log` 逐行可见、
`launch.log` 整块滞留、进程被杀就丢掉崩因那一段；而 logging 走 stderr 不缓冲，
表现成"print 的输出全没了"。
→ 换行即 flush。已验证：重定向下 5 行 print，未退出未 close 时可见 5/5。**置信度 高**

> `flush()` 本来就转发 `self._original.flush()`，所以带 `flush=True` 的半行进度不受影响；
> 全仓 `end=""` 且无 `flush=True` 的 print 只有 1 处，覆盖完整。

### 1.3 宿主内存

| 位置 | 缺陷 | 处理 |
|---|---|---|
| `ibs_engine.TraditionalMBARAnalyzer.compute_u_kn` | `n_workers = min(cpu_count, n_chunks)` —— **只看 CPU 数，完全不看内存**。Pool 用 `spawn`：每个子进程重新 import 整条 ibs_engine（连带 openmm/mdtraj/torch/pymbar-JAX）+ 反序列化整份 System + 建全体系 Context，单 worker 常驻成本≈父进程 RSS | 按 `MemAvailable` 兜底，新增 `_available_host_mib()`；下调时打 warning |
| 同上 | 所有 chunk task **提前全物化**，每个带 `xyz_all[a:b].copy()` ⇒ 全部坐标的又一整份 float64；`pool_tasks` 再浅拷一遍 | 改成生成器，父进程任何时刻只压一个 chunk |
| 同上 | `traj_list`（每文件一份 float32）+ `traj`（join 的第二份）在 `xyz_all` 建好后仍活到函数结束，而函数后半段正是 spawn N 个 worker 的地方 | `del traj_list, traj` + `gc.collect()` |
| `abfe_pipeline._is_traj_valid` | 一次裸 `read()` 把**整条 DCD** 拉进内存，只为证明能读到 EOF。`_all_remd_trajs_valid` 每 replica 一次、decharging resume 整套走两遍，且紧挨 Stage 1 建 replica | 分块读（200 帧/块），语义不变。实测 73 MB DCD 峰值 +71 MiB；截断文件仍判 False。含对无 `n_frames` 参数的 reader 的回退 |
| `abfe_core.image_molecules_by_system` 回卷后复查 | 一条语句开五个全尺寸数组（两个 fancy-index 拷贝、float32 差、float64 双宽第四份、距离矩阵）。键数≈原子数（含刚性水约束），最大调用方是 `md.join` 了所有窗口的 `compute_u_kn` | 逐帧算，峰值降到 O(键数)，超限即 break |
| `ibs_engine` 两个 `_collect_frames`（`probe_bidirectional_overlap` / `..._for_bias_calibration`） | `state.getPositions()` 无 `asNumpy` ⇒ 每帧一个装 N 个 `Vec3` 的 list（73k 原子≈10 MB/帧、数百万小对象），且**两个系综的全部帧**先囤完再统一求能量；校准探针 sample_steps 有 20000→40000→80000 重试阶梯 | `asNumpy=True` |
| 同上两处的 `finally` | `simulations.clear(); gc.collect()` 是**空转**：`sim_i`/`sim_j`/`frames_by_ensemble` 仍绑在栈帧上，`evaluator.context` 还是 `sim_i.context` 的第三个引用。本文件其它持有 Context 的地方都是 `del sim; gc.collect()`，bank 变体还额外 `evaluator.context = None` | 预置 None + finally 里逐个断引用 |
| `runabfe.resolve_boresch_restraint` | 每个估算器分支都已 `md.load` 过一次，末尾又无条件重载；CPython 先建新对象再重绑 ⇒ 峰值两份全轨迹 | 先 `del traj, estimator` 再重载（估算器可能就地改过 traj，不能直接复用） |
| `runabfe._checkpoint_probe_simulation` | 把探针 `Simulation` `setattr` 挂在 pipeline 上当缓存 ⇒ 一整个 30710 原子 System + Context 从预平衡判定挂到整条腿跑完；失效方向坏（不报错，只让后面"显存不够"变成假线索） | 删缓存，用完即走 |
| `outer_lambda_neural_basis` `screen-slow-variables` | `loaded.xyz.tolist()` ×2，而两个被调方进门第一句都是 `np.asarray(...)`。`.tolist()` 造 N×n_atoms×3 个 Python float（约 float32 数组的 15 倍），且与随后的 float64 数组同时活着 | 直接传 ndarray |

### 1.4 静默错数

- **`abfe_core.subsample_series_by_autocorrelation`** — 两条 fail-open：序列含非有限值、以及
  `except Exception` 吞掉 pymbar 真实失败，都返回"全部索引 + g=1.0"。docstring 自己写明后果是
  **n_k 虚高、误差棒系统性偏小 2–10 倍**，而在输出上与"序列真的不相关"完全无法区分。
  在生产路径上（`ibs_engine` 三处 MBAR 组装都调它）。→ 两条都改抛；"太短/无涨落/没装 pymbar"
  仍按原样放行。**置信度 高**
- **`ibs_engine._slice_window_frames`** — `dict(window)` 把 residual 臂的 `(K,N)`
  `sampling_state_energies` 整份抄过来却不切，半窗带全长数组去对 `(K,N/2)` 的 u_kn ⇒
  `solve_stage_integrated` 抛形状错 ⇒ 被 `split_half_drift_diagnostics` 外层裸 `except Exception`
  吞成 `available=False`。**residual 臂的 split-half 漂移诊断永久是暗的**，而那是文件里唯一对
  帧时序敏感的门。→ 一起切；形状本就对不上时返回 None 交上游发现。**置信度 高**
- **`abfe_pipeline.pre_equilibrate`** — `assert_starting_state_is_sane` 挂在
  `elif self.seed_ledger is not None` 后面 ⇒ **任何带 seed ledger 的运行整段跳过起点体检**，
  而 EXP-019/EXP-029 的 independent repeat 全都带 ledger。这道门要抓的正是"缓存 System 与当前
  输入不一致"（实测 PE=4.1e13、max|F|=3.7e9），跳过它同一个损坏体系就会在几千步后变成一条没有
  上下文的 NaN。→ 速度播种与起点体检拆成两个独立 `if`。**置信度 高**
- **`abfe_pipeline.pre_equilibrate`（膜）** — 无条件
  `setVelocitiesToTemperature(..., MEMBRANE_EQUILIBRATION_VELOCITY_SEED)` 覆盖掉 25 行前刚按
  repeat 派生的速度种子 ⇒ 每个 repeat 从逐位相同的速度出发、repeat 之间不独立，而 ledger 记的是
  一个从未被消费的 seed（provenance 说谎）。→ 加 `and self.seed_ledger is None`。**置信度 高**
- **`abfe_pipeline._load_pipeline_state`** — 读失败静默返回 `{}`，而 `_update_stage_status` 的写法是
  "读出来 → 塞进本阶段 → 整份写回" ⇒ 一次 NFS 抖动 / 半截文件就把**此前所有阶段的完成记录**抹掉，
  含 `equilibration.status == "completed"`（09-09 之后判断预平衡是否跑完的唯一权威标记），
  代价是静默重跑 5M 步。→ 文件存在但读不出来时抛；"文件不存在"仍返回 `{}`。**置信度 中**
- **`runabfe`：`enable_convergence_stop` 三处没透传** — `resolve_boresch_restraint` /
  traditional 两条基线都直接调 `pre_equilibrate()` 而不传该参数，走默认 `True`，
  `enable_equilibration_convergence_stop: false` **完全无效**。`run_full_pipeline` 那条是转发的，
  但真正跑满基线预平衡的是前者（到 run_full_pipeline 时 `equilibrium_is_done` 已为 True、整段跳过）。
  → 三处都透传。**置信度 高**
- **`runabfe.RunConfig.get`** — `getattr(self.args, key, default)`：argparse 的 dest 总是存在、
  没给参数时值是 `None`，所以调用方写的 `default` 永远轮不上。实例：
  `config.get("n_equil_steps", 5_000_000)`（7 处）在既无 CLI 也无配置键时返回 `None`，
  随后 `pre_equilibrate(n_steps=None)` 在步数运算上 TypeError。
  → 只有"argparse 侧压根没给"才回落 default；**配置文件里显式 null 仍如实返回 None**。**置信度 高**
- **`abfe_core.system_molecule_grouping`** — 连通性只认 `HarmonicBondForce` + constraints。
  原 docstring 论证"归组取窄、删键取宽"，**方向是反的**：多一条边只是两块一起平移（无害），
  少一条边把分子拆开、被 `image_molecules` 分别平移撕碎；而回卷后的复查只遍历
  `sorted_bonds`（同一份邻接表长出来的）⇒ 对漏边**结构性失明**。
  `rbfe_core` 的 hybrid builder 把 core–core 键全放在 `CustomBondForce`（`interp_bond`）里。
  → 加宽到含 `CustomBondForce`，与 `prune_topology_bonds_unsupported_by_system` 统一。
  旁证：OpenMM 自己的 `findMolecules()` 通过 `CustomBondForceImpl::getBondedParticles()`
  也是看得见的，原判据比 OpenMM 本身还窄。**置信度 高**
- **`outer_lambda_neural_basis` `screen-slow-variables`** — 同一个错误前提：`bond_pairs` 只从
  `HarmonicBondForce` 建，而 `discover_ligand_rotatable_torsions` 把它当拓扑键图的**替代品**
  （只有 `is None` 才回落），空列表被当成"没有键"照单全收。两种真实体系会踩：配体分子内成键项在
  `CustomBondForce` 里 ⇒ 报告写 `ligand_rotatable_torsion_count: 0` 且不报错；`constraints=HBonds`
  ⇒ 重原子度数算错 ⇒ `outer_priority` 选出不同外侧原子 ⇒ 与拓扑路径不同的 torsion 和
  `stable_id`，一路带进 freeze-slow-variable / EXP-011 manifest。→ 加宽到 CustomBondForce + constraints。**置信度 高**
- **`outer_lambda_neural_basis.IBSSamplerNeuralPathAdapter.collect_energies`** — catch-all 只重抛
  `"hard gate"`，而它所替换的 `ibs_engine.IBSSampler.collect_energies` 重抛 `"hard gate"` **和**
  `"LJ 长程尾项"`。⇒ LRC 的 fail-closed 被降级成静默丢帧；`NeuralPathFrameError`（冻结安全包络的
  判定结果）同样被吞，最终系综只由"神经 basis 恰好没炸的那些帧"构成且无记录。
  → 两类都重抛。**置信度 高**
- **`abfe_core.image_molecules_by_system`** — 把 System 的原子序号直接拿去索引 `traj.topology`，
  少一个才 IndexError，**多一个是静默错位**。整个函数立论是"System 是唯一权威"，却从不核对两边
  是同一个体系。→ 加原子数对账。**置信度 高**
- **`runabfe.resolve_boresch_restraint`（Boresch 末帧）** — 回卷跑在 `superpose` + `center_coordinates`
  **之后**。`superpose` 只转坐标、不转 `unitcell_vectors`，转完坐标系与盒子已错开，再拿这套盒矢量
  做 minimum-image 就是在错的参照系里搬分子，r/θ/φ 全变。且回卷的 fail-closed 复查被外层
  `except Exception → is_fallback` 降级成一条 warning（另外两处调用都是 fail-closed，只有这里不是），
  撕开的构型照样喂给 `assess_boresch_harmonicity`。→ 回卷挪到 superpose 之前并移出 try，
  补 `log=log.warning` 与 abfe_pipeline 处签名一致。**置信度 高**

### 1.5 崩溃

- **`ibs_engine.compute_shadow_bridge_u_kn`** — `top=topology if not isinstance(topology, str) else topology`
  是两个分支完全一样的死条件，实际总把 OpenMM `app.Topology` 递给 mdtraj，`_parse_topology` 落到
  catch-all 抛 `TypeError`。两个同胞函数都正确地先 `from_openmm`。⇒ shadow-bridge 腿在付完整个
  REMD GPU 代价**之后**才死在分析步。→ 修正并把转换提到循环外。**置信度 高**
- **`abfe_pipeline`（independent endpoint）** — `solved_wet`/`solved_dry` 在 25 行前被显式置 None
  （干/湿双起点诊断已退役），紧接着 diagnostics 里 `solved_wet.get(...)` ⇒ 必然 AttributeError。
  默认关闭，所以是"一开就炸"：先跑完全部端点采样，在 return 最后一刻整段崩掉。→ 直接写 None。**置信度 高**
- **`abfe_pipeline._diagnose_and_repair_all_pass_low_ess_window`** — 构造完 `payload` **漏了 return**，
  成功分支返回 None ⇒ 调用方 `tuple(action["window_range"])` TypeError。当前不可达（非变异修复策略
  提前 return，`tests/test_non_mutating_policy.py` 钉着），恢复变异修复循环时会立刻踩到。→ 补 return。**置信度 高**
- **`abfe_pipeline.apply_torsion_corrections`** — `.get("format", ...)` 写在 `isinstance(..., list)`
  判断**之前**，docstring 明文支持的 list 形式会先炸 `AttributeError`。→ 先分派类型。**置信度 高（latent）**
- **`rbfe_core.build_hybrid_system`** — 只搬 `CMMotionRemover`，而 `_assert_supported_forces` 明确把
  `MonteCarloBarostat` 列为支持的输入 ⇒ NPT 输入静默产出 NVT 杂合 System。
  `verify_hybrid_endpoints` 查不出来（barostat 对势能贡献恒为 0）。→ 一并搬。**置信度 高**

### 1.6 工具

- **`tools/validation/compare_charge_transfer_endpoints.load_case_raw_inputs`** — 缺 fixture 时抛裸
  `FileNotFoundError`，看不出是"少了数据"还是"路径写错"。所有走这个 loader 的工具（含
  `generate_c3_summary_reports.py`、`diagnose_coion_parameteroffset_mixed_precision.py`）**任何调用
  包括 `--help`** 都是这个下场，因为 `c2_lipid_slab_v11`（94 MB）不随本仓分发。
  而 `tests/test_generate_c3_summary_reports.py` fixture 缺失就整份 skip ⇒ **测试绿着、工具是死的**。
  → 明确报错 + `generate_c3_summary_reports.py` 支持 `ABFE_VALIDATION_FIXTURES` 覆盖 fixture 根目录。

### 1.7 新增诊断

- **`ibs_engine._host_memory_mib()` / `_available_host_mib()`** — 全仓在阶段边界打了 **8 处显存**，
  **宿主内存一处都没有**，所以宿主侧耗尽只能表现成一句无法归因的 `std::bad_alloc`。
  接在 REMD 建 replica 之前与失败瞬间两处，当前 RSS 与进程峰值都打（只有峰值单调，看不出某段
  跑完有没有还回去）。

---

### 1.8 第二批：10 条机械缺陷（2026-09-09 晚，用户点名"修第一部分 10 个"）

判定标准：不动物理、不作废任何已有缓存、能离线自证。

| 位置 | 缺陷 | 处理 |
|---|---|---|
| `abfe_pipeline._build_stage_cache_payload` + `compute_final_results` | `results_untrusted` / `stage_quality_failures` 有 **6 处生产者、0 处消费者**：既不进 stage 缓存白名单也不进 `final_results.json` ⇒ `--allow-untrusted-stage-results` 放行的结果与干净结果**无法区分**。同一白名单还丢了 `immutable_bridge_rescue` / `production_rescue_targets` | 四个字段进白名单；`final_results.json` 新增顶层 `results_untrusted` / `results_untrusted_stages` / `stage_quality_failures` 汇总 |
| `abfe_pipeline.run_full_pipeline`（快速最小化） | 释放写在 `try` 体内，`except` 只打 WARN 就继续 ⇒ 一次"快速最小化失败"就留下一个活 Context + 一份完整 System 副本，横跨 Stage 0 / 两次 pilot / Stage 1 建 replica | 挪进 `finally` |
| `ibs_engine.GlobalMBARAnalyzer.solve_stage_integrated` | `sampling_gauge_required = sampling_kj is not None` —— 用**被守护的数组自己**推导要不要守护它，恒真、永不触发；它要防的正是"residual 臂窗口拿不到 sampling_state_energies 却照样按物理规范过门" | 生产者显式写 `residual_sampling_arm` 布尔标记（落盘侧在 `_load_validated_joint_score_ledgers`），求解器只读这个标记。tmbar_history 那条路径行为逐位不变 |
| `ibs_engine.REMDManager.run` | N 个 `DCDReporter` 只在成功路径 `clear()`，而交换循环失败按设计一律重抛 ⇒ 收尾时机交给 GC、留半截 DCD，而 resume 只看文件在不在 | 整段包 `try/finally`（143 行纯重缩进，已 diff 校验除 try/finally/注释外无内容变化） |
| `ibs_engine._build_replicas`（CPU 回退） | `_clear_replica_contexts()` 只清三个 list，失败那轮的 `ctx`/`integ`/`replica_sys` 仍绑在**同一个 except 帧**上，而回退重建就在该帧内 ⇒ 最后一个 GPU Context 与 N 个新 CPU Context 同时存在 | 显式置 None + `gc.collect()` |
| `ibs_engine.generate_overlapping_windows` | 调用方给显式 `n_windows` 且 `max_start < n_windows-1` 时 `linspace` 取整产生**重复窗口**，覆盖性检查只看并集所以放行；重复窗口各自成为独立 `local_idx` 进协方差链，同一段 λ 的 ΔF 走两遍 | 去重 + warning。实测：`(8,6,2,6)` 由 6 个重复窗口收敛到 3 个；`(13,6,2)` 等既有用例逐位不变 |
| `runabfe`（argparse） | 默认 `allow_abbrev=True` 把 `--outp` 解析进 `args`，而配置合并层的 `_flag_present` 用**原始 argv 精确 token** 比对 ⇒ 该值被静默丢弃、改用 preset 旧值。`--outp ./run2 --temp 310` 会在 300 K 下写进 `./output` | `allow_abbrev=False`，交给 argparse 自己报错。实测缩写现在被拒、全称不受影响 |
| `ibs_engine.build_ibs_dual_system` | 壳退役后 `_common_plus_wca_system_xml` 与 `_common_system_xml` **逐字节相同**，却由第二次独立的完整 serialize→deserialize→serialize 往返产生并各留一份常驻（序列化对象是装配好的窗口 System：Group-1 CustomCVForce 里 K 个 softcore 力 + 整张排除表） | 退役分支直接复用同一个字符串；等价性**当场验证**（扫 `new_sys` 有无 Group 4），壳一旦复活就 fail closed |
| `local_residual/em_no_residual` | ① 孪生 System 在 window build 时就物化并 stash，"建了窗口但没走到配对 EM"（resume 跳过 / 异常 / SHA 不匹配的 return 分支）就一直挂到进程结束；② `CudaPrecision` 抓不到就不传 ⇒ 孪生落回 CUDA **默认单精度**做 EM 再把坐标拷回 mixed 的生产 Context，无任何日志；③ `_find_global_parameter_suffix` 找不到就静默什么都不做，注释说要关的 post-EM 残差窗口实际没关 | ① 改成惰性构造器，只在 EM 期间存在、`finally` 释放；② CUDA 平台下读不到精度直接 fail closed；③ 找不到时打 WARN 并说明去哪改 |
| `outer_lambda_neural_basis` | ① `_orb_ctx_cache` 无淘汰无上限（每条 entry 3 个带模型的 Context），且 key **不含 `number_array`** ⇒ 同索引不同元素组成会静默复用错的 Context；② `label-trajectory` 同时持有 `loaded_frames` / `frames_nm` / `_normalize_frame_collection` 三份全量副本 | ① key 补 `number_array` + FIFO 容量上限 `_ORB_DECOMPOSITION_CACHE_MAX_ENTRIES=4`；② 抽完就 `del loaded_frames`，三份减到两份 |


### 1.9 离线能量重算会静默挂死 — `ibs_engine.TraditionalMBARAnalyzer.compute_u_kn`

`multiprocessing.Pool` 检测不到 worker 死亡（`concurrent.futures.ProcessPoolExecutor`
会抛 `BrokenProcessPool`，`Pool` 不会）。worker 被 OOM killer 杀掉 ⇒ `imap_unordered`
永远不返回；`_mbar_worker_init`（要反序列化整份 System 再建 Context）在子进程里抛异常
⇒ Pool **无限重生**失败的 worker。两种都不给父进程任何输出，现场表现就是打完
`开始离线能量重算 | 400 帧 × 8 态 | workers=16` 之后一行都没有、也没有任何 traceback。

→ 改成带时限地 `.next(timeout=...)` 取结果：首个 chunk 与其后分别给独立预算
（worker 冷启动要重新 import 全套依赖 + 建 Context，本来就慢），超时抛出带
"已完成几个 chunk / 当前宿主内存" 的错误，由既有 `except` 转成单进程回退——
宁可慢也不静默挂死。时限可用 `ABFE_UKN_FIRST_CHUNK_TIMEOUT_S` /
`ABFE_UKN_CHUNK_TIMEOUT_S` 调。另外在提交后立刻打一行"N 个 worker 已提交，等待第一个
chunk"，让这段等待有据可查。**置信度 高（机制）**

### 1.10 第三批（2026-09-09 深夜）

| 位置 | 缺陷 | 处理 |
|---|---|---|
| `abfe_core.frames_per_chunk`（新增）+ `abfe_pipeline._is_traj_valid` + `abfe_core.image_molecules_by_system` | 我第一版把两处都做成了**过度保守**的固定小块（DCD 200 帧/块、回卷复查逐帧），而且分块算术在两处各搓了一套 | 抽出共用的 `frames_per_chunk(bytes_per_frame, ...)`（1 GiB 预算，按 16 GB 机器）。实测 30710 原子 → 2913 帧/块、75000 原子 → 1193 帧/块；回卷复查也改成按同一预算成块而不是逐帧 |
| `ibs_engine._lj_tail_correction_sigma_resolved_moments` | 同时物化 n_ligand × n_environment 的 `sigma_ij`/`eps_ij`/`sigma_key`/`inverse`/两个权重数组 | 改成逐 ligand 行累加。bin 集合从 **unique σ 的外和**推出（力场里 distinct σ 极少），不需要遍历全部 pair。**逐位等价**：`tests/test_lj_tail_sigma_moments_bit_identical.py` 冻结旧 dense 实现当 oracle、用 `.tobytes()` 比对，7/7 过。实测 41×73000：峰值 **+186.9 → +37.5 MiB**，耗时 0.225→0.333 s（建系时只算一次） |
| `outer_lambda_neural_basis`（MTS 循环） | `completed_inner += report_interval_inner_steps` 只校验 `mts_ratio` 整除、不校验 interval 整除总步数 ⇒ 超跑最多一个 interval；而 `ns_per_day` 按**请求的** `n_inner_steps` 算 ⇒ 被抬高，且它是 `minimum_n4_ns_per_day` 硬门的输入 | 末段截断（`min(report_interval, remaining)`）；速率一律按**实际完成**步数算，请求值与实际值都落盘（新增 `n_inner_steps_completed`） |
| `ibs_engine.build_shadow_coul_ibs_system` | 没有 `validate_wiring()`，而它和 dual builder 一样靠调用方逐个注册 `cv_k_int`/`cv_k_rest`；漏一个不报错，`CustomCVForce` 引用未注册符号会**静默给有限但错误的能量**（EXP-025 G4 Layer-1 实测差 2.27 kJ/mol） | 与 dual builder 对齐，在 Force 进 System 之前 fail closed。纯防御性，不改有效哈密顿量 |
| `tests/test_open_issue_fail_closed_contracts.py` | 5 处把 `_is_traj_valid` 编译进合成命名空间 `{"os": os}`，新加的 `frames_per_chunk` 不在里面 ⇒ NameError 被函数自己的外层 `except` 吞成"轨迹无效" | 补 `_IS_TRAJ_VALID_NAMESPACE`。**本仓反复踩的同一个坑**：合成命名空间替身缺符号 |

### 1.11 第四批（2026-09-10，用户方案 ①③⑤④② 逐条落地）

方案由用户给出，我逐条评过再改；其中 ③ 我提出并采纳了替代方案（原方案会打断 Stage 1）。

| 编号 | 位置 | 缺陷 → 处理 | 测试 |
|---|---|---|---|
| ① | `abfe_pipeline.run_full_pipeline` Stage 2 缓存接受块 | 缓存态数在**布局校验之前**提交；校验失败只回滚 λ 表和窗口、不回滚 `stage2_states` ⇒ 后续 fresh pilot 用缓存态数当探针网格密度（例如 23 而非请求的 17），而写回的 `protocol_key` 记的仍是请求值 ⇒ 下次 resume 把 23 点 pilot 当 17 点复用。→ 改为全部校验通过才提交 | `test_stage2_states_not_contaminated_by_rejected_cache.py`（3 条：提交点在校验之后、旧写法已消失、失败分支不碰该变量） |
| ③ | `abfe_preoptimizer` 探针 force group | 原生 NB（带 B3 的 PME ParameterOffset）与软核力同在 group 1，而差分只读 group 1。**原方案"留在 group 0"会打断 Stage 1** —— λ_coul 的依赖恰恰在它上面。改为给它独立的 `PREOPT_NATIVE_NONBONDED_FORCE_GROUP = 3`，由 `_metric_force_groups()` **按被差分的参数**选集合：差 `lam_coul` 读 `{1,3}`，差 `lam_vdw` 读 `{1}`。2D 度规路径（`_sample_metric_energy(..., axis=)`）同一条规则。总积分哈密顿量不变，只失效 co-ion 的 Stage 2 preopt 缓存 | `test_preopt_probe_coion_hamiltonian.py` 改为断言独立组且 `!= 1` |
| ⑤ | `outer_lambda_neural_basis._evaluate_decomposition` | 三个 MACE 区域各做一次 minimum image，cplx/lig 锚 `ligand_indices[0]`、env 锚 `environment_indices[0]` ⇒ `E_cplx − E_lig − E_env` 吸收伪胞内项。→ 对 ligand+environment **一次成像再切片**，三项共用同一套镜像。关键是**一致**不是"物理"：坐标逐位同源分解就精确成立 | `test_mace_decomposition_shared_minimum_image.py`（3 条，其中一条用跨周期边界构型**证明旧写法与新写法确实不同**、且差恰为整盒矢量） |
| ④ | `ibs_engine.build_shadow_coul_ibs_system` + Shadow-PME bridge | 两个 builder 照抄 dual 的"全额重建配体内部力"，但它们的背景只清粒子电荷、**保留配体 σ/ε 与 L–L exception** ⇒ L–L 普通 LJ ×2、L–L 1-4（LJ 与库仑）×2。→ Group 2 只补普通库仑（配体 ε 压 0、σ 留 0.1nm 防除零），**不再挂 `ll_14_force`**；1-4 完整留在背景 exception 里，与 `create_ligand_internal_force` 排除 1-2/1-3/1-4 正好互补 | `test_shadow_coul_no_double_counted_ligand_internals.py`（3 条：全体零电荷时 group 0+2 必须逐位等于原体系总能量与力——**修复前实测 11303 vs 5655，翻了一倍**；Group 2 无 CustomBondForce 且配体 ε=0；背景侧对偶断言防止反向改坏） |
| ② 上半 | `abfe_preoptimizer._refine_pilot_grid_in_steep_segments` | 加密点在主 pilot 跑完 λ=0 之后才开始，直接跳回 λ≈0.99 只给 500 步预平衡 ⇒ 在"配体突然长回已塌陷空腔"的非平衡构型上测 dU/dλ。→ 新增必填的 `pilot_states`：细化某段前恢复该段**高 λ 端点**的 (坐标/速度/盒子)，再向低 λ 顺序采；新插入点的状态存回列表供下一轮续接。缺快照时**在动 λ 之前** fail closed | `test_stage2_refinement_restart.py`（他人先行写好的 2 条，现已转绿） |

> ⚠️ **②只做了上半（采样语义），下半（缓存两层拆分）没做** —— 见 §2。

### 1.12 ② 的下半：preopt 缓存两层拆分（2026-09-10）

| 位置 | 内容 |
|---|---|
| `abfe_pipeline._PREOPT_DERIVED_PATH_KEYS` / `_split_preopt_protocol_key()` | 把 preopt 指纹拆成 **第 1 层 采样**（Hamiltonian、采样步数、差分步长、遍历顺序、加密探针协议）与 **第 2 层 派生路径**（`stage2_final_n_states` / `refine_extra_points_per_segment` / `window_min|max_states` / `free_energy_densify_points`）。这 5 个键**原来根本不在指纹里** —— 改了它们再 resume，`protocol_match` 仍成立、整段 Stage 2 用旧 λ 路径重采，而落盘的 key 记的是新配置 |
| `abfe_pipeline.PILOT_TRAVERSAL_SEMANTICS` | 第 1 层新增采样语义标签。**不靠协议版本号失效**（用户明令不升号），靠这个语义字段——它只失效"采样语义真的变了"的缓存 |
| `abfe_pipeline._cached_pilot_traversal()` | 旧缓存没这个键，但遍历语义**只作用在加密点上**。若那份 pilot 的 `pilot_points` 里一个 `is_refinement_point=True` 都没有（加密从未触发），新旧测的是同一批点 ⟹ 判为等价，不白烧 GPU。读不到 `pilot_points`（无法证明）一律按 legacy 处理，fail closed |
| `abfe_preoptimizer.recompute_vanishing_path_from_cached_pilot()` | 第 1 层匹配、只有第 2 层变了时，从缓存的 `pilot_lambdas`/`metric_g`/`pilot_points` **离线重算**布点分窗。走的是与主路径**同一个纯函数** `redistribute_vanishing_lambda_subdomains`，不是另写一套。缺 pilot 测量直接抛——不猜 |
| `_preopt_cache_matches_ignoring_code_hash` | 三条分支都摘掉这两层的键：这个比较器看不到缓存正文，判不了 traversal 等价性，那是调用点的活。它回到本职——"除 code hash 外物理/协议输入是否同一套" |

测试：`tests/test_preopt_cache_two_layer_split.py`（16 条），含"离线重算与纯函数逐项一致"、
"改 final_state_count 真的改变结果（否则等于没修）"、"缺 pilot 数据 fail closed"、
"派生层里不许出现 sha/hash"、以及 legacy 等价性的 4 种情形。

> **`.npz` 落盘不需要做。** 早前把它列成待办是我判断错了：加密总是在同一次
> `optimize_stage2_vanishing` 调用里紧跟主 pilot 之后运行，状态只需活在内存里；
> 跨进程 resume 复用的是**缓存结果**，不是这些快照。

## 2. 未修（0 条）

### 2.1 已登记

§3 的 7 条判定已登记进 [`TODO.md`](../TODO.md)《未关闭的代码缺陷》，编号
`AUDIT-01`~`AUDIT-07`，各带位置与"为什么不修"。页首那张表因此兑现了。

### 2.3 已出结论、不再挂在这里的条目

- `abfe_pipeline._ligand_conformer_diagnostics` —— **从内存 bug 降级**（见 §3）。
  mdtraj 1.11.1 的 DCD reader 给了 `atom_indices` 时只分配
  `F × ligand_atoms × 3` 加**一个单帧**的全体系 framebuffer，不会分配全轨迹数组。
  剩下的只是一条腿调三次的重复 I/O 和 concatenate 的小数组复制。
- `ibs_engine._lj_tail_correction_sigma_resolved_moments` —— 已修（§1.10）。
- MTS 步数溢出、MACE 三区域锚点、shadow 双计、`validate_wiring` —— 均已修
  （§1.10 / §1.11）。

## 3. 判定不改（附理由）

- **`abfe_core.minimum_image_displacement_nm` 的候选立方体增长**（`radius > 64` 的 guard 在尝试
  `N × 2.1e6 × 3 × 8` 字节之后才触发）—— 真实触发需要极端长宽比 + 大批量输入；
  已有 `docs/design` 的盒型识别提案覆盖这一片，不在本轮拆开改。
- **`abfe_core` 膜 leaflet 的 wrapped/unwrapped 混用**（`_protein_leaflet_cross_sections_nm2`、
  `assign_lipid_leaflets`、`verify_membrane_normal_axis`）—— 只影响膜路径，
  且 `membrane_observables_from_trajectory` 已按最大空隙弧修过一次；当前生产是可溶体系，
  留给膜线单独一轮。
- **`free_energy_engine.run_independent_windows` 保留全部帧**（73k × 1000 帧 × 8 态 ≈ 14 GB）——
  当前无生产调用者，是"接线即爆"而不是现在就爆。
- **`apbs_correction._read_dx_values`**（257³ 网格约 1 GB 峰值）—— APBS 修正当前不在主线路径上。
- **`abfe_pipeline._rebalance_fingerprint` 没有 System 身份绑定** —— 记下来但**不修**：
  修法只能是往身份里放自产产物的 sha256，而那是用户明确、反复否决过的做法
  （复发记录：`code_sha256` 2026-08-24、`system_xml_hash` + `positions_sha256` 2026-09-09）。
  这类 payload 本来就有显式协议版本号承担"算法变了"的信号。
- **`abfe_core.OnlineConvergenceMonitor` 的 K==1 / n_k 不一致** —— 全仓无 `abfe_core` 之外的调用者。
- **`TraditionalABFEPipeline.pre_equilibration_identity_fingerprint` 引用未赋值的
  `self.pressure`/`self.barostat_protocol`** —— 全仓无活的调用点。
- **`step_guard`** —— 该文件由另一会话拥有，本会话报了两条（`finite_state_check` 的 `why` 报错了量；
  每段成功后把步长弹回 entry_dt 导致每段重爬阶梯），已由对方修完并加测试；
  剩下一条（非 `OpenMMException` 逃逸会跳过 `setStepSize(entry_dt)` 恢复）转给对方。

---

## 4. 验证到什么程度

- 全部改动 `py_compile` 通过；CI 的 ruff 门（`ruff check abfe_core.py abfe_pipeline.py
  abfe_preoptimizer.py ibs_engine.py runabfe.py outer_lambda_neural_basis.py
  rbfe_core.py tests`）**All checks passed**。
  期间顺手修了同僚报的 3 个 F821：`_nan_diagnostics` 闭包引用 `win_sys`，而
  `run_all_windows` 尾部有 `del win_sys` ⟹ ruff 的延迟作用域模型误报。
  两处 `del win_sys` 都在 `run_all_windows` 里（别处没有），改成 `win_sys = None`。
- 离线套件：见本节末尾的命令。本轮新增 4 个测试文件、12 条断言。
- **逐项实测过的**：`_StdoutTeeToFile`（重定向下 5/5 可见）、`_is_traj_valid`
  （73 MB DCD 峰值 +71 MiB、截断仍判 False）、LJ 尾项改写（41×73000 逐位等价，
  峰值 +186.9 → +37.5 MiB）、shadow 双计（修复前 group 0+2 = 11303 vs 原体系
  5655，翻一倍）、MACE 成像（跨周期构型下新旧确实不同、差为整盒矢量）。
- **没有任何一条上过 GPU / 跑过真实 pipeline。** 尤其需要真机复验的三条：
  §1.1 的偏置爬坡补 1.0 档、§1.11 ③ 的 force group 重划（带电腿 Stage 1/Stage 2
  度规）、§1.11 ② 的加密点状态续接（改变了 pilot 的采样语义）。
- 6 条 skip 里 4 条是 fixture 不随仓库分发、1 条是本节点无 CUDA。
  **这类自我跳过正是本轮起因之一，见 §1.6。**

### 复跑命令

```bash
cd /home/ruigengji/ABFE_IBS/ABFE_IBS
PY=/home/ruigengji/miniforge3/envs/openmm_dev/bin/python

# 1) CI 的静态门（与 .github/workflows/cpu-ci.yml 的 static job 同一条）
$PY -m ruff check abfe_core.py abfe_pipeline.py abfe_preoptimizer.py \
    ibs_engine.py runabfe.py outer_lambda_neural_basis.py rbfe_core.py tests

# 2) 全量离线套件（约 2 分 15 秒）
CUDA_VISIBLE_DEVICES= $PY -m pytest tests/ -q

# 3) 只跑本轮新增/改动的那几条
CUDA_VISIBLE_DEVICES= $PY -m pytest -q \
    tests/test_stage2_states_not_contaminated_by_rejected_cache.py \
    tests/test_preopt_probe_coion_hamiltonian.py \
    tests/test_mace_decomposition_shared_minimum_image.py \
    tests/test_shadow_coul_no_double_counted_ligand_internals.py \
    tests/test_shadow_coul_ibs_builder.py \
    tests/test_stage2_refinement_restart.py \
    tests/test_warmup_overlap_protocol.py \
    tests/test_lj_tail_sigma_moments_bit_identical.py \
    tests/test_open_issue_fail_closed_contracts.py \
    tests/test_em_noresidual_patch_contract.py
```

> `CUDA_VISIBLE_DEVICES=` 是为了别去抢 GPU 节点上正在跑的作业；本地这些测试
> 全部只需 CPU/Reference 平台。
