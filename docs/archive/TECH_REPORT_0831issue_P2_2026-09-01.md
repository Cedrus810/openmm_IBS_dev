> 🗄️ **已归档（2026-09-02）。原文一字未改，不是待办。**
>
> 主题（0831issue 的 P2 backlog + RELEASE_READINESS 的 R1/R2）**已关闭**，
> 逐条处置结果已并入
> [RELEASE_READINESS_2026-08-31.md](../RELEASE_READINESS_2026-08-31.md)
> 的《第九轮审查 backlog》与《P2：本轮处置结果》两节。
>
> **但本文有两节是那边没有的，要查只能来这里**：
>
> - **§7 协议版本变化** —— `ESS_GATE_PROTOCOL_VERSION` 3→4 为什么**不作废缓存**
>   的完整论证（它只被报告、不进任何 resume 指纹），以及 baseline 臂逐位不变的
>   钉法。⚠️ 注意该常量**此后又被抬到 5**（`ibs_engine.py:17478`），本文写的是
>   当时的 4。
> - **§8 修改期间调整的既有测试（5 处）** —— 每一处「原锚点是什么、为什么改、
>   改后断言是更强还是更弱」。这是判断「那批改动有没有偷偷放松测试」的唯一记录。
>
> 另外 §9《没有验证的（重要）》记的是当时的边界，读时按日期折算。

---

# 技术报告：0831issue P2 backlog + RELEASE_READINESS R1/R2

**日期**：2026-09-01
**范围**：`runabfe.py`、`abfe_pipeline.py`、`abfe_preoptimizer.py`、`ibs_engine.py`、`abfe_core.py`
**环境**：`/home/ruigengji/miniforge3/envs/openmm_dev`
**测试**：`tests/` 全套 **1568 passed / 0 failed / 5 skipped / 3 xfailed / 1 xpassed**（162 s；改动前基线 1515 passed）

---

## 1. 这轮做了什么

两件事合成一件：

1. 把 `Atenolol-rank11/0831issue.md`（第九轮 7 路分片审查清单，位于**旧工地目录**、不在主线库里）的
   **剩余项整体并入** `docs/RELEASE_READINESS_2026-08-31.md`，作为发布验收的统一入口。
2. 把其中「能修的」修掉，包括 RELEASE_READINESS 自己列的 R1、R2 两条。

结果：**P1 13 条全部收口**，**P2 37 条中 30 已修 / 4 加标注 / 3 明确暂缓**，**R1 + R2 已修**。

---

## 2. 先纠正一份账目

原文档「总汇总」写 **0 P0 / 13 P1 / 43 P2**。逐条点数后：

| 项 | 汇总声称 | 实际正文 | 差异 |
|---|---|---|---|
| P1 | 13 | 13 | 一致 |
| P2 | 43 | **37** | `abfe_core` 分片 5 条**只有汇总行、正文完全缺失**；`abfe_pipeline` 分片汇总写 7 而正文只有 6 |

`abfe_core.py` 的分片状态栏至今仍是「审查中」，却已被计入汇总。**一份自己都没写完的清单不能当"已审查过"的证据** —— 这条本身就是发布缺口，记在 RELEASE_READINESS 的「仍缺的两件事」里。

---

## 3. P1：13 条的最终处置

| 批次 | 结果 |
|---|---|
| 前 8 条（08-31） | 5 已修（含 `IBS_BIAS_PROTOCOL_VERSION` 31→32）、2 误报、1 转交 4w53-21 |
| 后 5 条（09-01） | 3 已修（含 `ESS_GATE_PROTOCOL_VERSION` 3→4）、1 误报、1 加固（当前不可达） |

三条与发布直接相关的结论：

- **`_compute_geom_gradients` 键角解析梯度错了 130%~200%**（原报告写 +29%）。
  正确式 `∂θ/∂a = (cosθ·ba/|ba| − bc/|bc|) / (|ba|·sinθ)`；旧式把两个端点的分子互换、
  而 `1/|ba|` vs `1/|bc|` 的缩放各自留在原处，两处错误不互相抵消。
  有限差分复核：旧式相对误差 130%~200%，新式与 FD 一致到 ~1e-9。
  经 `--boresch-source auto` / `orb_simple` 到达，直接决定 `kthetaA`/`kthetaB`。
- **residual 臂的混合覆盖度门此前用错口径**：物理 `u_kn`（softcore+LRC，不含残差）
  配 sampling-gauge 的 `f_k`（自 v31 起在 `S_k = U_k^sc + s_residual·A_k·(B_φ−offset)` 上学）。
  两者差一个**逐态 rank-1 项**，在逐帧 softmax 里不抵消、也不是共模。
  **ΔG 不受影响**（row 0 用直接读出的 `e_bias`），但 **EXP-030 candidate 臂在修复前的收敛门读数不可引用**。
- **"传统 REMD resume 清空 DCD" 是误报**：`_steps_completed` 没有任何从盘恢复的入口，
  新进程里恒为 0；而调用方只在判定缓存不可用时才走到 `run()`，随后从第 0 步跑满 `n_steps`。
  **截断是正确的** —— 旧帧属于被主动作废的采样分布，保留反而会把两个 Hamiltonian 的帧混进一个文件。
  已用正则把"`_steps_completed` 无非归零赋值"钉进测试：将来有人加跨进程续算，那条测试会红。

---

## 4. R1 / R2

### R1：唯一 resolved seed

**问题**：`RunConfig` 保留配置文件里的 `repeat_seed`，但 `main()` 只从 `ABFE_RANDOM_SEED` 取值，
traditional 路径又读 `config.get("repeat_seed")` —— 用户在 JSON 里写了 seed、配置快照记着它，
**IBS 管线根本没收到**。

**处置**：新增 `_resolve_repeat_seed(args, config)` 作为唯一入口，
优先级 **`--seed` > `ABFE_RANDOM_SEED` > config `repeat_seed`**；解析结果由 `main()` 回写
`config.data["repeat_seed"]`，于是两种模式从同一个值出发，配置快照不可能再声明一个管线没收到的 seed。
另落 `repeat_seed_source` 并在启动日志打印 resolved 值与来源。新增 `--seed` CLI。

两个刻意的选择：

- **三者都不给 → 返回 `(None, None)`**，保留未显式设置 seed 时的历史随机流，不偷补新 seed。
- **非整数值拒绝而不是 `int()` 截断**。写测试时发现 `1.5` 会被静默变成 `1` —— seed 决定整个
  独立重复的随机流身份，"我写了 1.5、实际用了 1" 会让 provenance 里的 seed 与真正消费的不是同一个数。

### R2：状态锁

**问题**：`_pid_is_alive` 把 `except PermissionError` 写在 `except OSError` **之后**，
而前者是后者的子类 —— 那个 `return True` 分支**不可达**，"权限不足、判不了"被当成"进程已退出"，
stale-lock 清理可能删掉仍在使用的锁。

**处置**：三处一起收紧，方向统一为"**只在能证明锁已无主人时才删**"。

1. 按三类分开：`ProcessLookupError` → 确认不存在（唯一算 stale 的信号）；
   `PermissionError` → 进程存在、只是不属本用户 → 活着；其它 `OSError` → 判不了 → 活着。
2. **建锁与写 PID 之间的竞态**：`O_CREAT|O_EXCL` 成功后才写 PID，另一进程在这个窗口读到空文件会
   得出 `pid=-1` 并删掉 A 的锁 → 双持锁。现在空 payload 需老于 30 s 宽限期才判残留
   （正常竞态是微秒级，同时仍能清理 `kill -9` 留下的空壳）。
3. **共享文件系统**：payload 改为 JSON `{"pid", "hostname"}`（兼容旧的裸整数），
   hostname 与本机不一致时一律视为活着，**绝不删别的节点的锁**；写入后加一次 `fsync`。

正常路径（无竞争时的获取/释放）行为不变，只改变"判不了"时的默认。

---

## 5. P2：30 条已修，按"改了什么"分组

### A 组：会算错数的（7 条）

| 条目 | 问题 | 处置 |
|---|---|---|
| `abfe_preoptimizer` CDF 构造 | `xp[-1] = 1.0` 事后覆盖把 `c_{N-2}` 也覆盖掉，最后一个区间宽度从 `w[N-2]` 变成 `w[N-2]+w[N-1]`，λ[N-2] 权重双重计入 | 改为用前 N-1 个权重按**自身和**归一化，末端天然为 1.0（实测最后一段 0.65 → 0.53）。两处副本同改 |
| `_reduced_energies_for_record` | LJ 尾项系数用"记录列布局位置"索引，而系数按**物理 λ 态**编址 | 显式用物理态号取系数 + 越界 fail closed。当前生产者写恒等布局故不可达，但任何非恒等布局都会静默把尾项配错 λ（自由能错而不报错） |
| `pme_offset_charge_square_sum` | `Σq(λ)²` 只算 `λ²Σscale²`，隐含"base 电荷为 0"，无守护 | 就地断言。前提确实成立（三个 builder 都先清零再挂 offset），但一旦改成分段去电荷，两腿同错、循环里不抵消 |
| `_normalize_softcore_params` | `n_lj <= 0` 会让 λ=0 态的 LJ 尾项系数被静默置零 | 拒绝 |
| **同函数顺带发现的真 bug** | `power_lj=getattr(softcore_params, "power_lj", (2,2))` —— 而 `ACESoftcorePotential` 把它拆成 `m_lj`/`n_lj`，**没有 `power_lj` 属性**，getattr 永远落到默认 (2,2) | 读真实属性。生产恰好就是 (2,2) 故数值无变化，但这与该函数 provenance 里 "no longer silently overwritten by a fixed production default" 直接矛盾 |
| `_collect_shadow_cross_exclusions` | 跨组 1-4 静电在背景力（归零）与 shadow 力（排除）**两边都不算** | fail closed。Atenolol 非共价 → 当前 0 影响；换共价抑制剂体系不会静默少一项 |
| `_safe_boresch_ramp` | 灾难判据 `abs(总势能) > 1e5` 对 7 万原子盒恒真（−5e5~−1e6 量级），一旦重新接线会把**全部正常体系**判失败 | 改判 ΔE（相对爬坡起点） |
| `analyze_gradient_and_optimize_path` | 仍用 `Var(U_group1)` 而非 `beta²Var[dU/dλ]`，且 NaN 样本被替换成前值/0.0 后继续计入方差（压低方差 / 首帧坏时注入虚构 0.0） | 按 **PHY-08 同等处置** fail-closed（唯一调用者 `run_preoptimization` 自身零调用方，生产不可达）；NaN 改为丢弃 |

### B 组：口径 / 契约不一致（10 条）

| 条目 | 处置 |
|---|---|
| `run_full_abfe_loop` 溶剂腿拿不到 repeat-seed contract（`seed_ledger` 恒 None，两条腿不是同一 repeat 的独立腿） | 透传 `repeat_seed`/`leg_name`；`run_solvent_decoupling` 侧从 `pipeline_kwargs` pop 出来交构造函数（否则会以未知 kwarg 撞进 `run_full_pipeline`） |
| traditional 两条路径对同一批工件报两个不同的 ±（`err_boresch` 只加在一条上） | 按 `combine_binding_free_energy` docstring 的唯一约定（解析量不并入）统一为**不并入**；`boresch_correction_error_kJ_mol` 仍作独立字段落盘，analyze-only 侧补齐同名字段（显式 `None` + 原因，不用 0.0 假装误差为零） |
| `--only-complex-charging` / `--only-boresch-attachment` 不校验冻结腿采样温度，可拼出跨温度非法求和（kBT 差 ~3%） | 在 `_load_frozen_stage_result` 一处集中比对：不一致拒绝；字段缺失则大声告警但不阻断（免得堵掉合法旧工件） |
| `_strip_unit_suffix` 接受 `_deg` 却按弧度消费（释放项错约 57 倍） | 直接拒绝（与 P1 #5 在 `format_boresch_json` 的处置同源）。**不做 deg→rad 换算** —— 静默换算会让"以什么单位落盘"变成两套并存的约定 |
| `pilot_shadow_checkpoint_interval` 在 `abfe_config.json` 文档化为可用开关，`main()` 从不透传 | 透传（两条腿） |
| `--analyze-only` 恢复了 APBS 修正值却把 note 重置为空 | 与值同源恢复 |
| `CUDA:N` 写法跳过预平衡/再平衡的显式 Context 释放 | 用 base 平台名比较 |
| REMD 种子域 phase 硬编码 `"charging"`（该类同时服务 mixed/vanishing） | 改为可注入，**默认值不变** —— 改字符串就是改随机流 |
| `error_leg_kJ_mol` 与 `delta_G_leg_kJ_mol` 口径不配对 | **值不动**：`abfe_core` 的 schema 把 `("delta_G_total_kJ_mol", "error_leg_kJ_mol")` 声明为同一对，下游拿它配 `delta_G_total` 是对的。按 issue 原建议改成 `err_leg` 会把 `err_attach` 从 ± 里悄悄丢掉，是引入新 bug。改为补 `error_leg_excluding_attachment_kJ_mol` + `error_leg_kJ_mol_pairs_with` 两个名副其实的字段 |
| R1 / R2 | 见上一节 |

### C 组：落盘 / 日志 / 可审计性（13 条）

| 条目 | 处置 |
|---|---|
| 同进程第二条腿的日志 | 两个坑：(a) `logging` FileHandler 每次**追加**从不摘除 → 第 N 条腿的行写 N 遍、第一条腿的文件继续收后面的行；(b) stdout tee 用 `isinstance` 短路 → 第二条腿**根本不装 tee**，裸 print 全进第一条腿的文件、自己的 `pipeline.log` 几乎是空的。改为进程级单例「摘旧挂新 + tee retarget」（不套娃）。已实测两条腿各自干净分离、零重复 |
| 独立端点 walker 记录 / 湿种子缓存 | 改**原子写**（`_atomic_save_npz`，与同文件 fixed-H 探针库统一口径）；读取侧加损坏容错 —— 截断 npz 以前会让每次 resume 都崩且指不出该删哪个文件 |
| 2D 测地线寻径失败静默回退对角线 | 返回值与成功路径**完全无法区分**，次优路径被当成功结果写进 `geodesic_path.json` 并被后续 run 复用。现在寻径 provenance（`fallback` / `fallback_reason` / `magnitude_gate_dropped_edges`）一起落盘 + 告警 |
| 测地线 `\|g\| > 1e7` 量级闸门 | 弃边数计入诊断。**阈值未动** —— 动它会改已验证路径的数值 |
| `window_overlap_records` 孤儿窗口 | 两个 append 放进同一原子块，消除"进了落盘统计却没进协方差链" |
| split-half σ 膨胀 | 缺证据窗口以前静默 `floor=0.0`（看不出哪些窗口其实无实测）→ 逐窗口 `sigma_floor_unavailable` + 汇总名单/计数。并显式标注 `df_k` 与 `endpoint_error_after_offset` **仍是 MBAR-only 口径**：逐窗口 σ 下界无法无歧义映射到逐态 `df_k`（MBAR 协方差非对角），随手缩放等于编误差棒 |
| `top1pct_raw_weight` | N<100 时 `n_frames//100 == 0`、`max(1,...)` 抬到 1 → 该量退化成"**最大单帧**"，而阈值 0.35 是按 N≈330–430 校准的。落盘 `n_top_frames` + `degenerate_max_single_frame` 标注。**门的行为未改** —— 置 not-evaluable 同样 fail-closed，并不能让小样本窗口通过，真正的修法是门的重新设计（协议变更） |
| 在线 early-stop `step_at_check` | 加回 resume 前已完成步数（纯诊断标签） |
| `diagnose_force_breakdown` | 标注"近似重建"：丢 switching / λ 电荷 offset / 配体内部清零，Group-12 `max\|F\|` 可能被幻影项主导、误导爆炸源定位 |
| 三个 decharging builder 的 `frozen_ll_pairs` | 收集后**从不读取** —— P0-01 赖以成立的「既有 L–L exception 已冻结」这个前提此前**零守护**。加共享断言：逐对复核 `chargeProd` 与改写前一致、且没有任何 λ offset 指向它们 |
| attachment 腿 `n_samples` 的 `max(2,...)` 下限 | 会让实跑步数**超过设定**、且 split-half 每半只剩 1 帧 → 直接拒绝该参数组合（生产 250000/1000 不受影响） |
| 软核告警块每 λ 态重复打印（K=23 时刷 23 行） | 只打一次 |
| `generate_overlapping_windows` docstring | 示例与实测不符（银行家舍入，`(8,13)` 实为 `(7,13)`）→ 改正并写明原因 |
| 模块常量被捕获为函数默认值 | `VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS` 历史 2→6→4，默认参数在**定义时**求值一次 → 默认 `None`、函数体内读当前值，与校验方同源 |
| `ABFEPreOptimizer` auto 探测优先 `lam_coul` | 加 `target_phase` 构造参数，默认 `"auto"` 行为不变 |

---

## 6. 明确暂缓的 3 条（不是漏掉）

| 条目 | 为什么不在本轮做 |
|---|---|
| 窗口 checkpoint manifest / 缓存门不含 repeat-seed 身份 | 往这三处指纹加字段会**作废现有 GPU checkpoint**。生产确实启用 `seed_ledger`，所以条件插入也躲不开冷启动代价。属于需要明确批准的协议变更（先例：`IBS_BIAS_PROTOCOL_VERSION` 32 那次） |
| WCA 力与软核 CV 排除表不一致（WCA 用 `softcore_excl`、CV 用 `full_softcore_excl`；`build_shadow_bridge_system` 口径相反） | 改排除表会**改变能量**。原条目自评置信度低、"后果未验证"，且生产正在跑。需要先在最小体系上量化两种口径的能量差再决定 |
| `compute_boresch_attachment_u_kn` / `attachment_convergence_diagnostics` 零调用方 | 这是 #79 的 attachment 收敛诊断（round-trip 硬门、⟨U_B⟩ 单调性），**当前不在线上执行**。接线＝新增硬门（协议变更、要定阈值）；删除有 att27 先例但 #79 未被撤销。已在源码就地写明这个决定点，不擅自二选一 |

---

## 7. 协议版本变化

| 常量 | 变化 | 是否作废缓存 |
|---|---|---|
| `ESS_GATE_PROTOCOL_VERSION` | 3 → 4 | **不作废**。该常量只被报告，不进任何 resume 指纹（`abfe_pipeline` 只导入 `TARGET_SUPPORT_GATE_PROTOCOL_VERSION`） |

baseline 臂（`residual_enabled=False`）下 `S_k ≡ U_k^softcore`，混合覆盖度口径切换**逐位不变**（已用 `A_k=0` 的零容差比对钉住）。落盘新增 `ess_gate_mixture_gauge` 记录实际口径。

---

## 8. 修改期间调整的既有测试（4 处，均为实现锚点漂移，不是放松断言）

| 测试 | 原锚点 | 为什么改 |
|---|---|---|
| `test_charging_only_mode.py` | `"boresch_restraint = _load_frozen_stage2_boresch(output_dir)"` 整行字面量 | 该调用现在多行传 `expected_temperature_K`。要钉的结构（frozen 分支在通用 resolver 之前、由 `else:` 分开）保持不变，锚点改成函数名 |
| `test_audit_protocol_regressions.py` | grep `"xp[-1] = 1.0"` | 那行赋值已经不该存在。测试要守的不变量是"xp 严格以 1.0 结束"，改为**数值验证**该不变量 + 断言每个区间宽度等于归一化后的对应权重（没有双重计入）。断言比原来更强 |
| `test_wet_cavity_seed_cache.py` | grep `"np.savez("` | payload 改成原子写。顺序契约（先 payload 后 key）不变，而且更强：key 出现时 payload 必然完整。新增断言禁止回退到直写 `np.savez` |
| `test_pipeline_pre_equilibration_regressions.py` | 合成名前空间缺 `_split_platform_spec` | `pre_equilibrate` 现在用共享解析器判 base 平台名。注入**真实**函数（纯字符串、不碰 OpenMM），顺带保证生产代码与测试用同一份 spec 语法 |
| `test_core_physics_numerics.py` | `ess_gate_protocol_version == 3` | 精确等号是该测试刻意设计的"版本一动就回来核对"机制。已核对：语义断言全部仍成立，同步为 4 并补一条 `ess_gate_mixture_gauge == "physical_targets"` |

---

## 9. 验证

**新增回归测试 3 个文件、53 条**（全 CPU，不需要 GPU）：

- `tests/test_release_readiness_r1_r2.py`（25 条）—— seed 优先级四组合 + 五类非法输入；
  锁的 `PermissionError`/`ProcessLookupError`/其它 `OSError` 三分支、JSON payload + hostname、
  兼容旧裸 PID、活锁不删、外节点锁不删、死本机锁可清、竞态空 payload 不删、老空壳可清、
  以及"`except PermissionError` 必须排在 `except OSError` 之前"的源码契约
- `tests/test_0831issue_p2_batch.py`（28 条）—— 按 A/B/C 三组覆盖行为有变的条目；
  其中 `test_second_leg_gets_its_own_clean_log_file` 是真实执行两条腿日志安装、
  逐文件核对"各自只有自己的行、零重复"
- 前一轮的 `tests/test_ess_gate_sampling_gauge.py` / `test_pocket_projection_angle_gradients.py` /
  `test_remd_dcd_append_mode.py`（22 条）

**全量 `tests/`**：**1568 passed / 0 failed / 5 skipped / 3 xfailed / 1 xpassed**，162 s，一次干净运行（`-p no:cacheprovider`）。
改动前基线 1515 passed —— 净增 53 条全部是本轮新增回归，**没有任何既有测试被删除或跳过**。

### 没有验证的（重要）

**本轮全部是 CPU / 静态验证，没有任何 GPU 运行验证。** 最需要真机复验的三处：

1. **residual 臂混合覆盖度门换口径后**，EXP-030 candidate 臂的门读数
   （落盘的 `ess_gate_mixture_gauge` 应为 `sampling_states`）；
2. **三个 decharging builder 新增的 `frozen_ll_pairs` 断言** —— 若它在真实体系上触发，
   说明 P0-01 的前提本来就不成立，那是一个需要立刻处理的**发现**，不是这次改动的回归；
3. **两条腿同进程时的 `pipeline.log` 分离** —— 已用最小复现验证，未在真实两腿运行上确认。

wheel 安装验证与端到端 CLI 测试同样仍未做。**CPU 全套通过不解除这两条限制。**

---

## 10. 相关文档

- `docs/RELEASE_READINESS_2026-08-31.md` —— backlog 已并入，R1/R2 各自的「处置」小节
- `Atenolol-rank11/0831issue.md` —— P1 两个回填小节（逐条依据与数值复核），P2 正文位置
