# 未提交改动静态审核（下午批）2026-09-24

> **一次性审核的原始记录，不是待办清单。已归档（2026-09-24）**——10 条已逐条处理完，结果见下面「处理结果」；
> 没处理的 #5 #8 #10 去了 [TODO.md](../TODO.md) `REVIEW-0924 [P2]`。

> 与同日凌晨那份 `CODE_REVIEW_2026-09-24.md`（01:11）是两次独立审核，别合并去重当同一份。

- 范围：工作树相对 HEAD（5dd781a）的未提交 `.py` 改动，`git diff -w`（`ibs_engine.py` 主体是 CRLF→LF，真实改动约 4.6k 行）
- 审核：`/code-review` 后台代理，**没跑验证轮**；openfe 口径已对照本机 `/home/ruigengji/openfe`，无出入
- 快照：代理审的树约 14:20；主会话 14:37 抽核 #1 #2
- ⚠️ 用户反馈：**很多条多半已经被别的会话修掉了**，这份可能落后。动手前先对照当前代码逐条复核，别照单全收

**处理结果（2026-09-24 晚，逐条对照当时代码）**：#2 #7 已修（测试 `tests/test_split_half_mainpath_and_final_state_atomic_2026_09_24.py`）；#1 不修（orb 线基本不用）；#3 不触发（环境有 RDKit）；#4 有意为之（净电荷门保证 charge=0）；#6 是 09-20 有意的 fail-loud，只需修正 config 里的路径；#9 押后（膜线暂停）；#5 #8 #10 未处理。

状态说明：**已核实** = 14:37 在当时的树上读代码确认；**未核实** = 只有代理的说法（量级也是它估的）。

## 已核实

### 1. openfe_fixed 下 Orb 路径选不出任何锚点
**abfe_core.py:10572**（`_apply_hybrid_filter` 在 9996）

`openfe_fixed` 时 `_apply_hybrid_filter` 恒返回 kr=4184；快速验证 `if ks["kr"] > max_kr: continue`，`max_kr` 默认 2000，runabfe 也没改它 ⟹ 候选全丢。另外 kr 全相同，`kr_seen` 去重会把候选压成一个。

- 失败场景：`boresch_source=auto`/`orb_ml` + `boresch_force_constant_mode=openfe_fixed` ⟹ `validated=[]` ⟹ 「自动 Boresch 估算失败，未找到合格候选」
- 对照：`runabfe.py:4648` 已经为同一原因跳过最终裁剪（"kr 上限 2000 会裁掉 4184"），Orb 内部的快速验证没跟上
- 修法方向：固定档下跳过 kr 上限与 `kr_seen` 去重

### 2. `split_half_max_z` 没进主 stage-1/stage-2 求解（指纹里有，执行时没有）
**abfe_pipeline.py:7042**

`_run_dual_lambda_stage` 把自己的 `**kwargs` 当 `run_kwargs` 传给 `_solve_stage_with_config_gates`；这份 kwargs 是 `abfe_pipeline.py:17922` 起逐键点名转发的，里面没有 `split_half_max_z`。又因为 `run_kwargs is not None`，所以不会回退去读 `_last_run_config["kwargs"]` ⟹ `split_half_max_z=None`。

- 失败场景：config 设了 `split_half_max_z=2.0` ⟹ `run_full_pipeline` 把它写进 `_final_gate_thresholds`（进了 stage 指纹），runabfe 也透传了（8467/8733），但主求解的漂移门不触发；只有 merged/rewindow/repro-merged 那几条路径生效
- Shadow-IBS 子腿（约 6326，`run_kwargs=kwargs`）同形
- `final_*` 走的是同一条路，但 runabfe 本来就不传它们（一直用默认值），所以实际只影响 `split_half_max_z`
- 形状：就是 `_solve_stage_with_config_gates` docstring 里说的"第 4 次"那个接线洞
- 修法方向：`run_kwargs` 与 `_last_run_config["kwargs"]` 合并，而不是二选一

## 未核实

### 3. 没装 RDKit 时 openfe 锚点池报 NameError
**abfe_core.py:8067**：`from rdkit import Chem` 抛 ImportError 后 `local_of_global` 未赋值，except 吞掉后下一行用它 ⟹ NameError，而不是按 docstring 退回重原子池。

### 4. `DetermineBondOrders(charge=0)` 写死电中性
**abfe_core.py:8055**：带电配体（本仓支持，见 reserved co-ion）键级推断失败，静默落回「RMSF<cutoff 重原子」，芳环锚点永远用不上，只留一条 log。修法：charge 取拓扑电荷和的整数。

### 5. openfe「角度不平」判据两份实现、宽严不一
**abfe_core.py:10196**：Orb 类 `_openfe_angle_not_flat` 用均值±2σ 落在 (0,π) 内；模块级 `openfe_angle_not_flat` 用 OpenFE 的逐帧 ½Kθ² ≥ 10kT。代理估算 K=334.72/298K 下前者放行到距边界约 10°，后者要求约 22°。`_circular_std_rad` 也与模块级 `circular_std_rad` 重复。修法：Orb 方法改调模块级函数。
（凌晨那份 #3 说的是另一件事：fitted 档下 `k_angle=None` 导致守卫整个跳过。）

### 6. `find_gmx_include_dir` 显式路径不存在时直接抛错
**runabfe.py:694**：以前回退到 GMXLIB/GMXDATA。`tests/fixtures/memtest/abfe_config.json` 的 `gmx_path=/home/ruigengji/gmx26.0C` 在本机不存在；其他 config 的 `/home/ruigengji/gmx26.3` 在 kasuga02 上不存在 ⟹ 启动时就挂（包括 `resolve_boresch_restraint`）。

### 7. `pre_equil_final_state.npz` 不是原子写入
**abfe_pipeline.py:4708**：直接 `np.savez`；`adopt_pre_equilibrated_state` 用 `np.load` 读时也不捕获损坏。写到一半被杀 ⟹ 之后每次 `--resume` 都抛 BadZipFile/EOFError，而不是回退到 `pre_equil.chk`。修法：tmp+replace，与本批其它产物一致。

### 8. 坐标 sidecar 默认开，覆盖所有 stage，并且反复整份重写
**ibs_engine.py:1128**：默认每 5 帧写一次，写在通用 production 循环里，所以不只 stage 2（和名字不符）；`_enqueue_window_snapshot` 每 100 次更新把累积的坐标数组整份重新 stack 再写。代理估算：复合物腿约 5k 非水原子、625k 步/窗 ⟹ 每窗约 15 MB、重写约 12 次，乘上所有窗口和 3 轮复现 ⟹ 每个 rep GB 级 I/O。修法：默认关，或者改成只追加分块。

### 9. 膜 leaflet 按最大周期间隙切，水层薄时上下层会反
**abfe_core.py:5849**：例如 L=6 nm、头基在 z≈1 和 z≈5（疏水核间隙 4 nm > 水层 2 nm），会切在双层内部，上下层标签反过来。`validate_membrane_input` 总是传 box_vectors。膜线现在暂停，优先级低。

### 10. `_pose_metrics` 每个子块都重建 subset 拓扑
**abfe_pipeline.py:4448**：约每块 25 次 `md_top.subset(_subset)` 加新建 `md.Trajectory`，而 subset 拓扑是不变的，可以和 `_subset/_bb_sub/_lig_sub` 一起只建一次。纯性能问题，在预平衡热路径上。
