# 主线核心模块静态审核 2026-09-24

> **一次性审核的原始记录（2026-09-24 凌晨 01:11），不是待办清单。已归档（2026-09-24）。**
> ⚠️ **这份没有逐条对照代码复核过**——全是审核代理的说法（量级也是它估的），也没对照 OpenFE。
> 同日下午另有一份独立审核 [CODE_REVIEW_2026-09-24_pm.md](CODE_REVIEW_2026-09-24_pm.md)，已处理完；
> 两份别合并去重。与下午份重叠的：本文 #9（圆标准差两份实现）≈ 下午份 #5 后半；本文 #3 与下午份 #5 是同一个守卫的两件不同的事。
> 未复核条目的去向：[TODO.md](../TODO.md) `REVIEW-0924 [P2]`。动手前先对当前代码逐条核，别照单全收。

范围：`abfe_core.py` / `abfe_pipeline.py` / `ibs_engine.py` / `runabfe.py` / `abfe_preoptimizer.py` 相对 HEAD 的未提交改动（`ibs_engine.py` 名义 diff 55k+ 行，`git diff -w` 后实际 387 行，五个文件真实改动约 2555 行）。high 力度、只审不改。

## 高危：静默正确性问题

### 1. stage0 attachment 缓存 resume 读的是缓存里的 gate 标志，不是本次运行的
**abfe_pipeline.py:16798**（`stage0_protocol_key` 构建在 16766-16785，未纳入 `attachment_enforce_convergence_gates`）

`require_converged` 从缓存结果自带的 `convergence_gates_enforced` 取值，而不是当前运行传入的 `attachment_enforce_convergence_gates` kwarg；这个 kwarg 也没进 `stage0_protocol_key`，所以既算不进身份也算不进校验。

**失败场景**：先用 `attachment_enforce_convergence_gates=False`（debug/repro）跑一次，attachment 腿没收敛，`stage0_attachment.json` 落盘 `converged=false, convergence_gates_enforced=false`。之后对同一目录用默认（gates=True）生产模式 `resume=True` 重跑同一条命令：`stage0_protocol_key` 照样匹配（从没编码过 gate 设置），缓存的未收敛结果被直接加载，`require_converged` 又是按缓存里的（debug）标志算的——`validate_boresch_attachment_result` 就把一个从没收敛过的 ΔG/σ 静默放进了名义上的生产结果。

### 2. `OrbBoreschEstimator` 没接 config，`source=orb`/`orb_simple` 路径的 Boresch 力常数/锚点模式配置是死的
**runabfe.py:4480**（同样的调用在 4504 又发生一次）

`OrbBoreschEstimator(temperature=config.temperature)` 没传 config；它的 `__init__`（abfe_core.py:9446）没有 config 参数，`self.config` 永远来自 `DEFAULT_CONFIG`。所以 `boresch_force_constant_mode`/`boresch_anchor_mode` 这两个配置对 `source="orb"`/`"orb_simple"` 完全不起作用。

**关联**：diff 自己在 runabfe.py:4510-4513 的注释记录了姐妹类 `GeometricRestraintEstimator` 已经真实踩过这个坑一次（cyclod_ligand1/rep1：config 写的 `openfe_fixed`，日志还是拟合出来的 kphiB=10），并在 4514-4519 修了——但同样的修法没有搬到 `OrbBoreschEstimator` 的两个调用点，尽管这个类自己的 `openfe_anchor_metrics`/`_openfe_angle_not_flat`（abfe_core.py:577, 1021, 1082-1086）本来是能读 `self.config` 里这两个键的。

## 中危

### 3. openfe 平角/sinθ 奇点守卫只在 `k_angle is not None` 时跑，默认 fitted 模式下永远跳过
**abfe_core.py:10219**（`k_angle` 何时为 None 见 10464-10467）

`k_angle` 只在 `boresch_force_constant_mode == "openfe_fixed"` 时才非空；默认的 `"fitted"` 模式下恒为 `None`，`_openfe_angle_not_flat` 这个守卫因此被跳过。

**失败场景**：`boresch_anchor_mode=openfe`（一个独立开关）配合默认 `boresch_force_constant_mode="fitted"` 时，θ 分布贴近 [0,π] 边界的候选锚点三元组，选取阶段只会跑 ±150° 二面角和圆标准差检查，平角守卫形同虚设——这正是本仓库历史上记录过的那种失效模式（拟合出的角力常数被夹到低至 10 kJ/mol/rad²，σ_θ≈0.5 rad，导致 λ_vdw 中段鬼影配体撞进口袋），而 `openfe_anchor_metrics` 自己的 docstring 说这个检查就是为了在入口处排除它。

### 4. `--analyze-only` 读窗口输出的路径没接 BOX-01/02 盒体积 resume 门
**runabfe.py:5346** → `load_ibs_window_outputs_from_dir`（ibs_engine.py:573-586，无盒体积参数）

盒体积 resume 门（`current_box_volume_nm3`，接在 ibs_engine.py:7555/7792-7916 的实跑路径上）没有被 `_analyze_dual_leg_artifacts`/`--analyze-only` 调用的这个读取函数接受。

**失败场景**：目录里缓存的窗口能量是在偏离记录盒体积的密度下采的（本仓库历史上真实发生过：rep3 采样盒比 System XML 记录的小 2.86%，约 4.9% ΔG_solv 影响）。对同一目录做真实 resume 时新加的盒体积门会正确拒绝并重采；但 `--analyze-only` 走的是另一条读取路径，从未接收盒体积做比对，会静默复用这些密度错误的窗口并报出一个没有任何警告的 ΔG。

### 5. `GeometricRestraintEstimator` 的 `rmsf_cutoff_nm` 配置项没有对应属性，永远落回默认值 0.15 nm
**abfe_core.py:13444**（`__init__` 在 13118-13151 从未设置该属性）

`estimate_from_trajectory` 里 `getattr(self, "rmsf_cutoff_nm", 0.15)` 用的是 getattr 兜底，但构造函数从没写过这个属性，所以永远是 0.15。

**失败场景**：`boresch_anchor_mode=openfe`（diff 自己在 13128-13130 说这是生产实际在走的路径）下，用来过滤 openfe 芳环锚点候选池的 RMSF 阈值被死死钉在 0.15 nm；用户想为小/柔性配体调这个值不会有任何效果。

### 6. `debug_provisional_on_warmup_failure` 挂在通用 `stage2_frozen_layout` kwarg 上，不是专门的 repro 信号
**abfe_pipeline.py:6881**（`stage2_frozen_layout` 的通用用途见 3285-3287 文档）

`debug_provisional_on_warmup_failure = bool(kwargs.get("stage2_frozen_layout", False))`——直接借用了控制器会读的通用回退标志，而不是一个 `_freeze_stage2_path_for_repro` 专属的内部信号。

**失败场景**：如果调用方出于正当理由（比如想让控制器在真实生产跑里停止挪窗口）直接传 `stage2_frozen_layout=True`，一次没收敛的 warmup 会被静默降级成 `provisional_production`/`f_k_evidence_status=indeterminate`，而不是抛 `IBSFrozenCalibrationValidationError`，且没有任何信号提示这本来是 debug-only 行为。

### 7. 离子 LJ 参数按元素符号全局收集，没按具体离子种类限定
**runabfe.py:1822** `_overwrite_solvent_leg_terms_from_top`

按元素在复合物 `.top` 里全局收集 `(sigma, epsilon)`，发现同一元素超过一组不同参数就直接 `RuntimeError`，而不是按要拷贝的具体离子种类限定范围。

**失败场景**：如果复合物拓扑里同时存在两个元素符号相同但 LJ 参数不同的单原子残基（比如一个结合态/晶体学离子 + 一个体相同元素反离子），`params` 会收集到 >1 组不同 tuple，溶剂腿构建对一个本来合法的体系直接硬失败（`"复合物 top 里 {elem} 离子有 N 套 LJ 参数，无法确定用哪套"`），且没有办法消歧选用哪一套。

## 低危 / 可维护性

### 8. `SOFTCORE_POWER_LJ_OVERRIDE` 是裸模块级全局变量，跑完不清空
**abfe_core.py:7946**（写入见 abfe_pipeline.py:15987，读取见 abfe_core.py:8150 等）

由 `run_full_pipeline` 设置、被软核力构建代码读取，用完不重置，也不按 pipeline 实例做隔离。

**失败场景**：任何不重启解释器就连续跑多条 leg/多个 pipeline 的进程（批量driver、notebook，或者跑完一次 pipeline 后直接调用软核构建函数的代码）会静默读到上一次 `run_full_pipeline` 留下的 override 值，把看似独立的两次运行的软核哈密顿量混在一起。diff 自己在 abfe_pipeline.py:15982-15984 的注释承认了"必须在构建任何力之前设置"这个顺序隐患，但没处理跨运行泄漏。

### 9. 圆标准差公式在两个 Boresch 估计器类里各存一份，边界处理还不一样
**abfe_core.py:10144** `OrbBoreschEstimator._circular_std_rad`（`np.hypot` 版本）vs 模块级 `circular_std_rad`（abfe_core.py:7960，复数指数均值版本，同一份 diff 新增）

`GeometricRestraintEstimator` 用模块级函数，`OrbBoreschEstimator.openfe_anchor_metrics` 用自己的 staticmethod 拷贝；一个 `np.clip`，一个显式 `r<=1e-12` 分支。

**失败场景**：未来只改一份公式的边界处理（比如 r 接近 0 的情况），两个估计器类"同样借自 openfe"的锚点接受判据就会在同一条轨迹上悄悄分道扬镳。

### 10. `_legalize_tail_window` 没用上这次 diff 自己刚提炼出的 `tail_window_max_states()` 辅助函数
**abfe_pipeline.py:14338**（辅助函数定义在 3179-3191，另外 3 处调用点在 ~6670/12240/12848 已接入）

这次 diff 的 docstring 自己说这个公式之前已经有三份拷贝互相跑偏过，所以才统一成了 `tail_window_max_states()`；但第 4 处（这里）依然手写了同一套 clamp 公式（`tail_hi = hi if last_window_max_states is None else int(last_window_max_states); tail_hi = max(lo, min(tail_hi, 2*hi-2)) if hi>=2 else hi`）。目前数值上等价，不是 bug，但下次改这个 clamp（比如改 `2*hi-2`）会像之前三份一样漏掉这一处。

---

## 审核中排除的候选（核实后确认不是问题）

- `_overwrite_solvent_leg_terms_from_top` 无条件覆盖溶剂腿配体扭转参数 —— 这是 09-23 那次"`ligand_only.xml` 零扭转"bug 的既定修复，不是回归。
- `partitioner_default_state_bounds()` 把校验范围放宽成并集 —— diff 自己的 docstring 说明这是刻意修复此前过窄的门在拒绝合法分窗器输出；结构正确性另有函数单独检查。
