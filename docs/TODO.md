# 当前行动清单

更新：2026-08-02。完整 2026-07-27 审计长记录已移入 [archive/TODO-2026-07-27-full.md](archive/TODO-2026-07-27-full.md)。历史原文见 [archive/todolist-2026-07-20.md](archive/todolist-2026-07-20.md)，审计证据见 [status/AUDIT_STATUS.md](status/AUDIT_STATUS.md)，运行验证见 [status/VALIDATION_MATRIX.md](status/VALIDATION_MATRIX.md)。本轮移出的完成/关闭项见 [../archive/todo-2026-07-29.md](../archive/todo-2026-07-29.md)。

## 当前决策

- **Boresch 二面角符号反号根因已修（BOR-01，2026-07-29）。** `abfe_core.py` 四份手写
  二面角副本都返回 **−φ**，其中 `calc_boresch_from_last_frame` 会用镜像参考值覆盖
  正确的 `boresch_simple.json`。现统一走 `abfe_core.boresch_dihedral_rad()`
  （`abfe_core.py:1142`）。全量记录见
  [handoffs/BORESCH_DIHEDRAL_SIGN_HANDOFF.md](handoffs/BORESCH_DIHEDRAL_SIGN_HANDOFF.md)。
- **当前生产结果（2026-07-29 11:02，符号修复后全量）**：

  ```
  复合物腿 ΔG_cplx = 181.00 ± 1.76    溶剂腿 ΔG_solv = 157.84 ± 1.79
  Boresch attachment = 4.39 ± 0.08    解析修正 = −38.76    APBS = 0.00
  ΔG_bind = −23.16 ± 2.51 kJ/mol = −5.54 ± 0.60 kcal/mol
  参考 result.txt total = −6.279 ± 0.457 → 差 +0.74，0.98σ，1σ 内一致
  ```

  对比 07-06 符号 bug 期的 −9.76 kcal/mol，**2.7 kcal 的改善基本全部来自本次符号修复**
  （复合物腿 192.89 → 181.00）。⚠️ ±0.60 是乐观的，真实约 **±1.0 kcal/mol**。
  这组数取代此前所有 ΔG_bind 候选（−2.121 / −3.460 / −3.4797）与 attachment
  5.601 ± 0.223 —— 那些都是符号 bug 期的值，不得再引用。
- **口径纪律：`result.txt` 只有 total 可比，分项不可比。** 它是旧方法的参考值，本仓库
  实现的是 IBS，分项拆法本来就不同；restraint 与被限制腿的采样还会结构性抵消，把
  charging 的差和 restraint 的差当成两条独立线索是重复计数。**「与 result.txt 差
  2.8 kcal，逐项归因」这条旧结论已整体撤销**，P1-18 因此关闭（见该条）。
- **vdW（stage2）只能用 TMBAR，本轮一字未动。** 不得引入 BAR / TI / 全帧主值 /
  √g σ / bootstrap σ。07-28 曾有一批这样的扩张被整批撤回，见 P1-21 末尾与 P1-22。
  charging（stage1）主值为相邻 BAR（P1-21，2026-07-28）。

- **当前没有尚未修复、会阻挡生产全量重跑的 P0。**
- P0-10 的生产代码已修；旧复合物腿采样作废，07-29 那轮已是修复后的 fresh full rerun。
- P0-11 已结案（2026-07-28）：溶剂腿盒子缺陷已修，生产默认 `SOLVENT_PADDING_NM = 1.5`（盒边 4.257 nm）。
  ⚠️ 当时「三档盒子对 ΔG_bind 零影响」的结论口径已被 07-29 修正：pad1.5→pad2.4 的
  −7.15 kJ/mol **不是**有限尺寸效应，而是 vanishing 腿的跑间散布（同盒子两次独立跑
  差 2.34σ，比跨盒子 1.13σ 还大）。见 P1-19。
- **当前最高优先的物理问题是 vanishing 腿的误差估计偏小（P1-19），不是盒子尺寸。**
  它**不是代码 bug**（渐近协方差在正确地算 within-run 统计误差），而是不确定度量化
  加采样不足；同一带里真正的代码 bug 是 P1-23 的 σ 采纳 fail-open（**已于 2026-08-03 修**）。
- 新 DEXP 已按 `experiments/DEXP_KERNEL_PHYSICS_ISSUES.md` 冻结为
  `abfe_core.py` 内唯一的 pair-specific LJ-matched 解析核；配置契约只接受
  `alpha_vdw/beta_vdw`。旧 Orb 全局 fitter 已隔离到 `dexp_退役.py`，不再属于生产协议，
  旧参数 JSON fail closed。
- **DEXP core 合并已完成（DEXP-MERGE-01，2026-07-29）。** `dexp_NEW.py` 已删除；
  `cutoff/switch` 与 Gaussian-Coulomb 参数分别归入命名常量，三斜 minimum-image
  校验已进入 `SurrogateSystemBuilder`；生产 Orb 拟合 CLI/pipeline 入口已移除。
  `dexp_退役.py` 只显式 import 所需 core 符号，不使用 `import *`，生产四文件对它零引用。
  `tests/test_dexp_new_production.py` 已改测 core 契约，覆盖未知/旧字段、非有限
  alpha/beta、缺失参数文件、旧拟合 JSON、minimum-image、井深与曲率；定向测试
  **24 passed**。softcore 壳仍为 1.2/1.0 nm，落盘基线逐位保持
  181.00 / 157.84 / −5.535906 kcal/mol。
- **测试契约修正（TEST-GATE-01，2026-07-29）。**
  `tests/test_audit_protocol_regressions.py::
  test_final_convergence_gate_uses_orthogonal_evidence_not_duplicated_ess`
  原先错误要求 diagnostics-only 的 `min_occupancy_normalized` 进入最终
  `converged` 门，与 `ibs_engine.py` 已落盘的 `min_occupancy_is_gate=False` 协议矛盾。
  测试现改为断言 occupancy 不进入最终门，并钉住 threshold=None、退役原因与
  diagnostics-only 元数据；不修改任何生产收敛判据。
- 主 TODO 只保留仍需行动的事项；已修复项目、长表格、诊断过程与 2026-07-27 复审细节都归档。

## P0 / P1 当前待办

- [ ] **P1-19b（已知统计限制，生产验收前处理；**不**阻塞工程闭环与 B3/B4/B5 开发）：
  单次 σ 不能代表 attachment 腿的跨运行散布，且五次里有一次显著离群。**
  同体系同协议五次独立运行 `ΔG(A′→A)`：
  `5.7726 ± 0.0969` / `5.8623 ± 0.0971` / `6.0786 ± 0.0976` / `6.0880 ± 0.1062` /
  `7.5216 ± 0.1141`（kJ/mol）。

  | 口径 | 数值 | 相对单次 σ ≈ 0.10 |
  | --- | --- | --- |
  | 五次样本标准差 | **0.716** | ≈ 7× |
  | 五次极差 | 1.749 | (17× —— **极差不是标准差，这个比值没有校准意义**) |
  | 去掉 7.5216 后四次样本标准差 | **0.158** | ≈ 1.6× |

  ⚠️ **本条早前写成"σ 系统性低估约 17 倍"是错的**（拿极差除以 σ）。正确表述：
  * 它能说明**单次 σ 不能代表跨运行散布**；
  * 但**不能**严谨宣称"σ 系统性低估 17 倍"—— 去掉离群点后只有 **1.6×**，属正常范围；
  * 那个 `7.5216` 是**一次显著离群运行**，先当异常个案查，不要当成 σ 口径的证据。

  待办：查清 7.5216 那次是怎么来的（λ 路径不同？起点坐标不同？前后半程漂移超容差？），
  以及在真正做三重复 / benchmark / 生产验收时按 §13.4 的口径重新评估散布。
  **不要**用"多跑几次取平均"当作 σ 口径的修复。
  与 P1-19 的关系：`memtodolist.md` §17 规定 P1-19 阻止**进生产**，
  并没有规定它阻止 B3/B4/B5 的方法开发 —— 别让它把工程闭环拽住。
  详见 `memtodolist.md` §0.5.12。

- [x] **P0-REMD-CUDA：根因是 pymbar 4 的 JAX 后端预分配整卡 75% 显存 —— 已修
  （2026-08-04）。** 现象是 REMD 建 12 个 CUDA Context 失败 → 静默退 CPU →
  Stage 1 事实上跑不出来。

  **根因（2026-08-04 由阶段显存点定位）**：pymbar 4 的后端是 JAX，JAX 默认
  `XLA_PYTHON_CLIENT_PREALLOCATE=true` + `XLA_PYTHON_CLIENT_MEM_FRACTION=0.75`，
  **一碰 GPU 就预分配整卡的 75%**。attachment 腿末尾用 pymbar 解 BAR/MBAR，
  日志时序逐行对上：

  ```
  06:15:21 | WARNING | ******* JAX 64-bit mode is now on! *******
  06:15:21 | INFO    | Reached a solution to within tolerance with hybr
    ✅ ΔG(A′→A) = 6.0786 ± 0.0976 kJ/mol
    📊 [显存] Stage 0 attachment 结束: used=12197 free=3646 total=16303 MiB
    📊 [显存] Stage 1 建 replica 之前: used=12197 free=3646 total=16303 MiB
  ```

  **12197 / 16303 = 74.8%**，就是那个 0.75。于是 Stage 1 只剩 3646 MiB，
  12 个 Context 需要 12 × 317 = 3804 MiB —— 建满 11 个、第 12 个抛
  `No compatible CUDA device is available`。

  这一条同时解释了此前**所有**反直觉现象：
  * 离线探针 `memtest/probe_remd_context_capacity.py` 能建满 12 个 —— 它**不调 pymbar**，
    JAX 从未初始化；
  * 更大的可溶体系（73536 原子）当初 12 个成功 —— 与体系大小无关；
  * 换全新进程 resume 照样失败 —— 每个进程都会重新预分配；
  * "约 10 GB 对不上账" —— 那 10 GB 就是 JAX 的预分配，不是 Context 泄漏。

  **修法**：`abfe_core.py` 顶部（**在任何 `import pymbar` 之前**）
  `os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")`。
  只关预分配、**不**把 MBAR 挪到 CPU（JAX 仍在 GPU 上按需申请，数值路径不变，
  避免动已落盘的基线）。用 `setdefault` 所以外部可覆盖；想把显存全留给 OpenMM
  可在外部导出 `JAX_PLATFORMS=cpu`，但那会改变 MBAR 的执行设备，属于需单独验证的改动。
  **证据**：`tests/test_import_time_side_effects.py` 两条新测试
  （源码顺序：设置必须早于 `import pymbar`；干净子进程 import 后标志为 `false`）。
  全套 **979 passed / 2 skipped / 0 failed**。

  ✅ **真机已确认（2026-08-04，同一台机、同一体系，前后两次）**：

  | 时间点 | 修之前 | 修之后 |
  | --- | --- | --- |
  | 预平衡结束 | used=269 | used=269 |
  | **Stage 0 attachment 结束** | **used=12197** | **used=445** |
  | Stage 1 建 replica 之前 | used=12197, free=3646 → **退 CPU** | used=445, **free=15398** |

  Stage 1 decharging **06:34:12 → 06:53:26 跑完**，
  `decharging_exchange_diagnostics.json`：`n_replicas=12`、
  **`platform_name=CUDA`、`platform_fallback_reason=None`**、
  500 轮交换、平均接受率 **0.675**；`ΔG_decharging = 75.931 ± 0.504 kJ/mol`。
  （回退前那次的证据仍留在 `decharging/remd_platform_fallback.json`，06:15 那份。）

  顺带修掉的两个**真实但非元凶**的泄漏（各约 317 MiB，一个 Context 的量）：
  * `ibs_engine.run_boresch_attachment_leg` 整段原来**没有任何 teardown**
    （唯一的 `app.Simulation` 在 `ibs_engine.py:1494`，全函数无 `del`/`gc.collect()`）
    → 已加显式释放 + 释放前后显存打印；
  * `abfe_pipeline` 的 λ 预优化原来只 `del context, integrator, probe_sys`，
    而 `optimizer`（`DualLambdaPreOptimizer(probe_sys, context, ...)`）仍持有 `context`
    → 改为 `del optimizer, context, integrator, probe_sys` + `gc.collect()`。

  以下为定位过程的完整记录（保留，因为四个被否掉的假设都很容易被再猜一遍）：

  实测(`memtest/output_membrane_100ns`,45354 原子,12 副本 decharging)：
  ```
  23:48:20 [双λ] Stage 1: 去电荷 (λ_coul: 1→0, λ_vdw=1)
           ⚠️ REMD GPU Context 构建失败，已释放已创建的 replica contexts；
              回退 CPU 重建。原始错误: No compatible CUDA device is available
  23:48:46 decharging/decharging_rep{0..11}.dcd 全部创建
  00:05:03 12 个 DCD 仍然**全是 0 字节**（16 分钟零帧），`pipeline.log` 自 23:48:20 无新行
  ```
  这正是 `ibs_engine.py:13573` 那段注释描述的病征（"CPU 回退慢约两个数量级，
  23 分钟连第一帧 DCD 都写不出来，表现得像卡死"）—— 只是这次触发点不是
  预防性回退（`max_resident_contexts` 默认已等于 `n_replicas`），而是**真实构建期失败**。

  已排除的原因：
  * 不是别人占卡：`nvidia-smi` 显存 **10820/11264 MiB 空闲，无任何 compute 进程**；
  * 不是 exclusive compute mode：`compute_mode = Default`（多 Context 合法）；
  * 不像纯粹显存不足：`ibs_engine.py:13590` 记录同一张 11 GB RTX 2080 Ti 上
    **12 × 73536 原子** PME Context 装得下（`--only-complex-charging` 路径一直这么跑，
    manifest 记 `platform_name: CUDA`），本体系只有 45354 原子。

  ❌ **已否掉的假设 1：进程内 Context/显存泄漏。** 曾猜"Stage 1 之前同进程已在 CUDA 上
  跑过 100 ns 预平衡 + Stage 0 四个 λ 态 + 两次 λ 路径预优化，Context 没释放干净"。
  用户用**全新进程 resume**（Stage 0 与两条 λ 路径都已缓存，直接从 Stage 1 起跑）
  实测：**依旧回退 CPU**。所以与进程历史无关。

  ❌ **已否掉的假设 2：体系太大/PME 网格太大。** 逐项比过：

  | | 原子数 | 盒 | 12×CUDA Context |
  |---|---|---|---|
  | `output_lrc_fix`（可溶） | **73536** | ~735 nm³（≈9 nm 立方，PME ≈ 90³） | ✅ `platform_name: CUDA`, `platform_fallback_reason: None` |
  | `memtest/output_membrane_100ns`（膜） | **45354** | ~440 nm³（5.94×5.94×12.6 nm，PME ≈ 60×60×128） | ❌ 失败 |

  （原子数由 `system_native.xml` 的 `<Particle ` 计数除 2 得到——`<Particles>` 与
  `NonbondedForce` 各出现一次；两个数正好对上 `ibs_engine.py:13590` 注释里的 73536/45354。）
  膜体系**更小**却建不出 Context，所以既不是原子数也不是 PME 网格。
  `ibs_engine.py:13590` 那条"12 个 Context 装得下"的注释是**真的**，但它的证据只覆盖可溶体系。

  ❌ **已否掉的假设 3：OOM（显存不够）。** 新增探针
  `memtest/probe_remd_context_capacity.py`（走与生产相同的
  `_prepare_pme_coulomb_leg_system` 路径，逐个建 Context 并打印剩余显存）实测：
  * 裸建：**12/12 全部建成**，`used=3787 MiB`，**平均每 Context ≈ 315 MiB**；
  * `--replay-preoptimizer`（先复现生产在 Stage 1 前 17 秒做的那次 λ 路径预优化
    Context + `LocalEnergyMinimizer.minimize`，再建 12 个）：**同样 12/12**，
    `used=4054 MiB`、每个 ≈ 338 MiB。预优化只留 ~267 MiB 残留，挡不住任何东西。

  口算同一个数：每 Context 的显存**由与体系无关的固定开销主导**（CUDA context +
  编译好的 kernel ≈ 300–500 MiB）；45354 原子那部分只有几十 MiB
  （posq/velm/force ≈ 18 MiB、PME 网格 60×60×128 ≈ 10 MiB 量级）。
  12 × 338 MiB ≈ 4.1 GB，对 11 GB 的 2080 Ti 也宽裕。
  ⟹ **"减少 λ 状态数"绕的是一个还没被证明存在的 OOM**；能不能绕过尚未证实。

  ❌ **已否掉的假设 4：并行 stage / fork 导致子进程 CUDA 不可用。**
  `--parallel-stages` 已于 2026-07-27 整体移除（`abfe_pipeline.py:5958` 入口直接拒绝，
  归档 `docs/archive/removed_parallel_stages.md`），Stage 1 在主进程串行跑，没有 fork/spawn。

  相关历史 bug（都不是本条元凶，但同族，查的时候会撞上）：
  **ATT-03** = GPU 默认 `max_resident_contexts=1` ⟹ 建任何 Context **之前**就预防性退 CPU
  （2026-07-27 改默认为 `n_replicas` 修掉；本条报的是**构建期**失败，文案带
  "已释放已创建的 replica contexts"，不是那一条）。
  **ATT-04** = `abfe_core` 模块级 `get_optimal_device_settings()` 在 import 期用 torch
  抓 CUDA（import 期已修，`tests/test_import_time_side_effects.py` 守着）。

  ⏸️ **2026-08-04 用户决定暂时挂起**，先做 MEM-00c/B3。

  ✅ **已就地打点（2026-08-04，纯诊断、不改任何数值）**：`REMDManager._build_replicas`
  现在记录建 Context 前的显存、每个 Context 建成后的 used/free 与平均占用；
  失败时**在 `_clear_replica_contexts()` 之前**读显存（释放之后再读只剩一个
  "失败后已回收"的数，判不了当时够不够用），并给出 OOM / 非 OOM 判定 ——
  剩余显存还够再建一个 Context 就明确写"**不像 OOM**，减 λ 数只会掩盖它"。
  显存读数走 `nvidia-smi` 子进程而**不是** torch/pynvml：诊断绝不能自己去初始化
  CUDA（ATT-04 的教训）。
  证据**当场落盘** `<stage_dir>/remd_platform_fallback.json`
  （requested_platform / props / 已建成个数 / 原始异常 / 前后显存 / 判定）。
  ⚠️ 为什么必须当场落盘、不能靠日志：`ibs_engine` 的 `print` 早就路由到 `logger`
  （`ibs_engine.py:314` `print = _log_print`），但 **`logger` 没有 FileHandler**，
  而 `pipeline.log` 是 `ABFEPipeline._log` 另一条通路自己写的 —— 所以这条告警在归档
  日志里**一行都没有**（2026-08-03/04 因此只能靠 `nvidia-smi` 看到卡是空的才推断出
  它退了 CPU）。而 `platform_fallback_reason` 只在阶段**跑完**时才进
  `*_exchange_diagnostics.json`，一旦像这次那样在 CPU 上磨到被杀，那份文件永远不会
  出现，证据彻底没了。

  ✅ **打点跑出根因了（2026-08-04，`decharging/remd_platform_fallback.json`）**：

  ```json
  "vram_before_mib":     [12197, 3646, 16303],   // REMD 还没建任何 Context
  "vram_at_failure_mib": [15689,   154, 16303],
  "n_contexts_built_before_failure": 11          // 每个 ≈ 317 MiB
  ```

  **失败不是"12 个 Context 太多"**：12 × 317 = 3804 MiB，卡有 16303 MiB。
  是**开跑前就已被占掉 12197 MiB（全卡 75%）**，只剩 3646 MiB —— 差 **158 MiB**，
  所以正好建满 11 个死在第 12 个。

  用户**独占节点**，所以不是别的作业占的 —— 是**本进程自己漏的**。记账：
  Stage 1 之前在 CUDA 上跑过的是「预平衡 1 个 Context + Stage 0 attachment 4 个
  （4 个 λ 态）+ Stage 1/2 两次 λ 预优化各 1 个」≈ 6 × 317 ≈ **2 GB**，
  而实测 12.2 GB ≈ **38 个 Context 的量** ⟹ **约 10 GB 对不上账，漏点未定位**。

  ⚠️ 所以"减少 λ 状态数"只是**掩盖**：它能让这次跑起来（12→11 就够了），
  但那 10 GB 还在，窗口一多必然再撞上。**不要把减 λ 当成本条的修复。**

  已做的两件事：
  1. **判定改准**：自动判定现在会区分"REMD 自己吃满卡"与"开跑前已被占掉大块"
     （后者按占比 >25% 触发，明确要求先查占用者、并写明减 λ 只会永久掩盖它）；
  2. **补上一个确定的漏点**：`abfe_pipeline.py` 的 λ 预优化原来只有
     `del context, integrator, probe_sys` —— 那只解掉三个**局部名字**，而
     `optimizer`（`DualLambdaPreOptimizer(probe_sys, context, ...)`）仍持有 `context`，
     显存要等函数返回、optimizer 被回收才可能释放，有引用环还要等 gc 周期；
     而 Stage 1 在本函数返回后**17 秒**就开始建 replica。现改为
     `del optimizer, context, integrator, probe_sys` + `gc.collect()`，并打印释放前后显存。

  下一步（重跑一次即可定位那 10 GB）：
  1. ~~记显存~~ ✅；~~判定区分两类 OOM~~ ✅；~~λ 预优化 Context 彻底释放~~ ✅；
  2. **看新增的阶段显存点**：`_log_vram()` 已插在「预平衡结束（Context 已清理）」、
     「Stage 0 attachment 结束」、「Stage 1 建 replica 之前」三处，
     再加 λ 预优化的「释放前 → 释放后」。哪一段的 used 跳了几个 GB，漏点就在那一段。
     最可疑的是 **Stage 0 attachment 腿**（`BoreschAttachmentREMDManager` 建 4 个
     Context，且找不到成功路径上的 teardown）。
  3. **CPU 回退必须响亮**：回退后 29 分钟只跑完第 0 轮交换（500 轮 ≈ 10 天）、
     `pipeline.log` 一行不写，和卡死无法区分。至少要按轮打进度。

- [ ] **P1-22：vdW/stage2 的帧选择与 σ 口径（独立课题，不得顺手做进估计量层）。**

  两件已知的事，但都必须单独设计、单独验证，**不要**再在
  `GlobalMBARAnalyzer.solve_stage_integrated` 上顺手扩张：

  1. **点估计**：去相关选帧在有限样本下不稳。曾用（已撤回的）全帧模式量到
     complex 143.1162 / solvent 101.6877，相对去相关分别移动 −3.708 / −0.136 kJ/mol，
     且 complex 的位移集中在 win0 (−1.884) 与 win2 (−2.122)，而全帧值正好落在两个
     半程 142.638 / 143.235 之间。**留作历史观测，不是当前结论。**
  2. **σ**：全帧 MBAR 的渐近协方差是 **naive σ**（把相关帧当独立样本，低估约 √g）；
     换全帧还会让两道门变松——独立样本数门必须吃 N/g（否则"≥20"对 500 帧恒真）、
     端点 σ 必须乘 √g。零 GPU 的正解是**移动块 bootstrap**（每个 replicate 重新执行
     local-TMBAR 与窗口拼接、扫块长取最保守值），而不是块间 SEM（dof 只有 4，
     那个 SEM 本身就是噪声量）。stage1 上实测 bootstrap σ ≈ 渐近 σ 的 1.9 倍。

  **⛔ 无论怎么设计，都不得给 vdW 引入 BAR 或 TI**：BAR 前提是两个端点系综各自
  有样本（IBS 物理 λ 行 n_k=0）；TI 前提是势对 λ 线性（vdW softcore 非线性，且从未
  落盘 ∂U/∂λ）。

- [x] **P1-23：σ 采纳路径的 fail-open 已修（2026-08-03）。**

  `inflate_sigma_from_split_half=True` 原先只替换 `total_error` 与逐段
  `uncertainty_kJ_mol`，却**不更新 `max_endpoint_uncertainty_kJ_mol`、也不重判
  `converged`** —— 等于"σ 抬上去了而门还在读抬高前的小 σ"。σ 抬高的**全部意义**就是
  让门看见真实不确定度，门读旧值就是 fail-open：一个本该被端点 σ 门拦下的 stage 会带着
  `converged=True` 通过。
  现在采纳 σ 之后会：从抬高后的逐段 σ 重算 `max_endpoint_uncertainty_kJ_mol`
  （旧值留成 `..._mbar_only_kJ_mol`）、用 `_meets_maximum_with_roundoff` 重判端点门、
  超限时把 `converged` 由 True 改判 False 并落 `converged_revoked_by_sigma_inflation`。
  只重算**端点 σ 这一项**：overlap 与独立样本数与 σ 口径正交，值没变，不动。
  ⚠️ 该标志**仍默认关闭**，所以本次改动对现有落盘基线零影响
  （181.00 / 157.84 / −5.535906 不变）；它保证的是"一旦启用，门是真的"。
  证据：`tests/run_offline_tests.sh` 全绿（956 passed）。

- [ ] **P1-19：per-window σ 系统性低估 2–4 倍，五道门全都看不见（2026-07-28；2026-07-29 并入原 BOR-04 的跨跑证据，当前最高优先物理项）。**

  **定性：这不是代码 bug。** `segment_error` 取 pymbar 渐近协方差，代码在正确地做它
  声称的事——渐近协方差算的就是「单次运行内、样本 iid 且已收敛」前提下的统计误差。
  缺陷在于**这个数被当成别的东西用**（进 ΔG_bind 误差棒、进收敛门），而它低估真实
  跑间散布 2–3 倍。根子多半在采样侧（vanishing 腿系综仍在慢移），σ 偏小只是盖住了它。
  归类为**不确定度量化 + 采样不足**，不是崩溃级缺陷。
  ⚠️ 与此相邻的 **P1-23 才是真 bug**（σ 采纳路径 fail-open：抬高了 σ 而门还在读小 σ），
  **已于 2026-08-03 修**：采纳 σ 后会重算端点 σ 门并重判 `converged`。
  ⚠️ 但那只修了『门读的是新 σ』，**没有**回答『σ 该不该抬』——
  `inflate_sigma_from_split_half` 仍默认关闭，本条（σ 是否系统性低估、
  要不要默认启用那条下界）仍然未决，见下面的行动顺序。

  措辞统一：这是**渐近协方差在"看起来独立、系综仍在慢移"时的低估**，
  与 P1-21 那条「自相关子采样导致有限样本点估计不稳定」是两件不同的事；
  两者都**不构成「MBAR 本身有偏」**。σ 口径的修法见 P1-22（移动块 bootstrap）。

  新增 `ibs_engine.split_half_drift_diagnostics()`：把每个窗口的帧按时间切前后两半
  各解一遍，判据 `z = |后半−前半| / (2σ_win)`（两半各自 SE≈√2σ，其差 SE≈2σ）。
  ⚠️ 两个半程走**与主值相同的帧选择**（各自重新去相关）。2026-07-28 曾把它改成强制
  全帧并被撤回，所以下面那三张表**继续有效、不需要重算**（实测复核：complex stage2
  win1 仍是 5.25×2σ，总 σ 1.5913 → 3.3771、×2.12）。
  `solve_stage_integrated()` 每次解完自动挂 `split_half_diagnostics` 并落盘。
  **默认只诊断不阻断**；传 `split_half_max_z=2.0` 才否掉 `converged`。

  溶剂盒扫描三轮 18 个窗口，**5 个超 2σ**（σ 正确时期望 0.8 个，二项概率 ~0.1%）：

  | | win0 | win1 | win2 | win3 | win4 | win5 |
  |---|---|---|---|---|---|---|
  | 3.000 | 0.69 | **2.05** | 0.85 | 0.88 | 1.70 | 1.34 |
  | 4.257 | **2.29** | 1.40 | 1.48 | 1.02 | **4.34** | 0.40 |
  | 6.057 | 0.20 | **3.05** | 0.16 | 0.60 | **2.93** | 0.64 |

  **window 4 是铁证**：3.000 那轮它在所有现有指标上都是优等生
  （`absolute_ess` 348.6、`n_decorr` 357、g 1.40、每个 λ 的 `ess_ratio` ≥ 0.976、
  `min_occupancy` 0.944），`σ_win` 只有 0.236，实漂 0.80–1.50，z 到 4.34。
  **没有伴随任何 ESS/overlap 退化**——问题不在采样质量的代理量，在 σ 本身。

  根因位置：`GlobalMBARAnalyzer.solve_stage_integrated` 里
  `segment_error = float(local["dDelta_f"][join_idx, end_idx])`（相对 def 偏移 +441），
  直接取 pymbar 的**渐近协方差**。渐近协方差假定样本独立同分布且已收敛，
  而 window 4 恰是「看起来独立、系综仍在慢移」的情形。

  拟议修法（零 GPU 成本，用已算出的量）：`σ_win ← max(σ_MBAR, |漂移|/2)`。
  改后 win4 的 σ：0.236→0.402（3.000）、0.173→0.750（4.257）、0.196→0.575（6.057）。

  **影响**：stage2 总 σ 会从 1.10–1.47 变成 3–5 kJ/mol，ΔG_bind 误差棒从
  ±0.62 kcal/mol 变成 ±1.0–1.5，与 `result.txt` 那 4.16 kcal 的差距性质从
  「约 7σ」变成「约 3σ」。**stage1 不受影响**（三轮 z 全 < 2，已验证）。

  工具：`tools/diagnostics/diagnose_split_half_convergence.py --stage both`（纯离线，秒级）。

  **跨跑证据（2026-07-29 从原 BOR-04 并入，与上面的 split-half 是同一现象的两个视角）：**
  同 padding 1.5、同一个盒子（`box_edge_nm=4.257`、Na=7 Cl=7）两次独立跑，
  vanishing 差 **4.675 kJ/mol = 2.34σ**，**比跨盒子差异（1.13σ）还大**；
  decharging 反而干净（同盒子 0.24σ、跨盒子 1.14σ）。所以问题精确定位在 **vanishing 腿**。

  | 运行 | decharging | vanishing | 总计 |
  |---|---|---|---|
  | pad 1.5 scan（07-28） | 63.115 ± 1.104 | **101.639** ± 1.100 | 162.826 ± 1.559 |
  | pad 1.5 主跑（07-29 11:02） | 62.800 ± 0.671 | **96.964** ± 1.663 | 157.836 ± 1.793 |
  | pad 2.4 scan（07-28） | 64.249 ± 1.078 | **94.491** ± 1.431 | 156.812 ± 1.792 |

  **推论：pad1.5→pad2.4 那 −7.15 kJ/mol 不是有限尺寸效应，就是这个跑间散布。**
  这同时否掉了 P0-11 当时「三档盒子对 ΔG_bind 零影响」的口径。

  **行动顺序（不得跳步）：**

  1. **在固定 padding 1.5 下再跑 1–2 次重复**，把 vanishing 的真实跑间 σ 钉下来
     （现在只有 2 个样本，σ ≈ 3.31 是极粗估计）。这是判断后续任何盒子扫描结果是否
     显著的**唯一基准**。
  2. 重新评估上面那条 `σ_win ← max(σ_MBAR, |漂移|/2)` 下界是否该默认启用
     （目前 `默认未采用`）。
  3. **只有 1 做完之后**才决定要不要为盒子尺寸加档。**现在别再扫盒子**——
     `--padding 3.0` 单跑一档分不清散布与尺寸效应，纯属浪费 2–3 h。
     若最终要改生产默认，改 `runabfe.py:101` 的 `SOLVENT_PADDING_NM`（目前 1.5）；
     改后 `solvent_cache_manifest.json` 的 `identity.padding_nm` 不匹配会自动重建缓存，
     **无需手工删文件，更不要把扫描目录的 `final_results.json` 拷进
     `output_lrc_fix/solvent_leg/`**（那是改产物不改生成器）。

- [ ] **P2-17：`tools/repairs/repair_stage2_window0_real_delta_f.py` 的文档化流程已跑不通（2026-07-29 发现）。**

  该修复工具在三处（`:42`、`:72`、`:230`）指导用户用
  `ABFE_DEBUG_SKIP_STAGE2_FINGERPRINT=1 python runabfe.py ...` 续跑，其中 `:230` 是运行时
  直接打印给用户的下一步命令。但该环境变量自 P1-20 起会让 `runabfe.py` **直接 raise**，
  所以照着这个提示走必然失败。改法：删掉这三处提示，改为说明「指纹不匹配就重跑 pilot」，
  或给该工具补一条真正的离线迁移路径（逐字段核验后重写指纹）。纯文本/提示修改，不动物理逻辑。

## P2 工程 / 发布质量

- [ ] **ATT-19：核心物理单元测试覆盖仍不足。** GitHub [#59](https://github.com/Cedrus810/openmm_IBS_dev/issues/59)。已补一批数值/协议测试，但仍缺软核势端点、DEXP LJ matching、PBC/离子计数更完整覆盖、并行 worker 等最小矩阵。**新增测试一律只放 `tests/`**（2026-07-29 起全部自动化测试已归位该目录，入口 `./tests/run_offline_tests.sh`，单文件 `./tests/run_offline_tests.sh tests/<file>.py`）。

- [ ] **ATT-20：缺少公开 ABFE benchmark 端到端集成验证。** GitHub [#60](https://github.com/Cedrus810/openmm_IBS_dev/issues/60)。需要中性/带电配体、两腿循环闭合、实验对比的可复现脚本。

- [ ] **ATT-21：文档缺口。** GitHub [#61](https://github.com/Cedrus810/openmm_IBS_dev/issues/61)。2026-07-29 已完成仓库与文档整理：目录导航见根 `PROJECT_LAYOUT.md`，文档导航见 `docs/README.md`，教程拆为 `GETTING_STARTED` / `OUTPUTS_AND_RESUME` / `TROUBLESHOOTING` / `MIGRATING_TO_A_NEW_SYSTEM` / `MAINTAINING`，旧 README 状态段迁入 `status/README_STATUS_SNAPSHOT_2026-07-29.md`。仍缺 API 参考、独立热力学循环推导文档、打包元数据。

- [ ] **ATT-22：CI/CD、静态检查与格式化仍缺。** GitHub [#62](https://github.com/Cedrus810/openmm_IBS_dev/issues/62)。`tests/run_offline_tests.sh` 已修并跑出 367 passed（2026-07-29 起测试统一在 `tests/`，CI 配置应指向该入口）；仍需 GitHub Actions、ruff/flake8、mypy、black/isort，并隔离 GPU 作业。顺带清掉 `abfe_core.py`、`abfe_pipeline.py`、`abfe_preoptimizer.py` 中仍会捕获 `KeyboardInterrupt/SystemExit` 的 3 处裸 `except:`。

- [ ] **ATT-23：运行恢复与资源保护能力不足。** GitHub [#63](https://github.com/Cedrus810/openmm_IBS_dev/issues/63)。继续评估 GPU OOM 降级/Context 回收、长任务中断恢复、磁盘空间预检和运行时估计。新增明确缺口：`_is_checkpoint_valid()` 目前只检查文件 ≥512 B 且可 seek，`_is_traj_valid()` 只检查粗略大小、`CORD` 和首个记录长度；二者都不能证明 checkpoint 可加载或 DCD 帧完整。恢复流程应以真实 `loadCheckpoint`/DCD 解析为准，并避免在 checkpoint 加载成功前决定追加旧轨迹。

- [~] **ATT-24：输入验证不足；显式 config/torsion 静默降级已修，DEXP 暂缓。** GitHub [#64](https://github.com/Cedrus810/openmm_IBS_dev/issues/64)。仍需 broader ligand/TOP/Boresch/box-size 前置诊断与 DEXP 输入契约。低优先防御项：移除 `calc_boresch_from_last_frame()` 对 `(3,3)` 坐标的猜测式转置；把 `ACESoftcorePotential.optimize_alpha()` 的 `assert` 改为显式 `ValueError`（该方法当前全仓零调用）。

- [ ] **P2-16：GROMACS include 自动发现不具备 Windows 可移植性。**

  `runabfe.find_gmx_include_dir()` 调用 Unix 命令 `which gmx`，失败后又扫描两条
  特定用户的 `/home/ruigengji/...` 目录。Windows 上仍可通过显式 `--gmx-path`
  或 `GMXDATA` 正常运行，所以原清单的“严重、必失败”评级不成立；但 PATH 中已有
  `gmx.exe` 时无法自动发现，属于真实的 P2 跨平台缺陷。改用 `shutil.which("gmx")`，
  从可执行文件位置推导 share 目录，并删除个人目录回退；补 Windows/POSIX mock 测试。

- [ ] **ATT-25：协议版本矩阵缺少统一注册/迁移工具。** GitHub [#65](https://github.com/Cedrus810/openmm_IBS_dev/issues/65)。需要统一注册表、缓存指纹组合规则、迁移说明和兼容性测试。

- [ ] **ATT-26：`IBSWindowManagerDualLambda.run_all_windows` 过长且职责混杂。** GitHub [#66](https://github.com/Cedrus810/openmm_IBS_dev/issues/66)。实测约 3055 行；端到端回归稳定前暂缓大拆。清理时一并处理 3 个零调用遗留点：`scan_boresch_1d_pes()` 的二次 Å→nm 转换及不可达角度分支、`aggregate_all_energies()` 用矩阵长短猜 `(K,N)` 方向、重复导入同一个 `generate_overlapping_windows`。

- [ ] **ATT-28 / R-05：日志与通用工具实现分裂。** GitHub [#41](https://github.com/Cedrus810/openmm_IBS_dev/issues/41)。统一结构化日志入口、级别和文件/控制台策略；另评估 DEXP 多随机种子优化。`ibs_engine.py` 的模块级 `print = _log_print` 属于本项而非数值 bug；同时统一 5 个 `NumpyEncoder` 实现，保持 `np.integer → int`、`np.bool_ → bool`，避免局部版本把整数写成浮点或拒绝 numpy 布尔值。

## 当前运行验证

- [ ] **V-02：传统 `single_lambda`/REMD 小型固定盒回归。** GitHub [#32](https://github.com/Cedrus810/openmm_IBS_dev/issues/32)。确认每个 task 收到有限、长度等于态数的 v3 LRC 数组，且每帧修正为 `coeff[k]/V(t)`。

- [ ] **V-08：核对 `stage2_n_states = 17` 与实际落地 23 个唯一 λ 的语义（2026-07-29 登记，低优先）。**
  V-03 的窗口契约完全满足，但 `provenance.config.stage2_n_states` 是 **17**，而实际落地的是
  23 个唯一 λ（6 窗口 × 槽位 − 5 次边界复用）。两个数都对得上各自的定义时没问题，
  但配置名叫 `n_states` 却不等于实际态数，容易被下一个人误读成契约违规。
  只需确认语义并在契约文档里写明二者关系，不改数值。

## Boresch 二面角符号事故（2026-07-29）与后续

全量诊断、证据链、时间线见
[handoffs/BORESCH_DIHEDRAL_SIGN_HANDOFF.md](handoffs/BORESCH_DIHEDRAL_SIGN_HANDOFF.md)。
下面只留仍需行动的部分。

- [x] **BOR-02：`update_boresch_from_last_frame` 的校验门已看二面角（2026-08-03 已修）。**

  原先那两道强校验只看 θA/θB（安全域 40–140°）与 r0 漂移（≤ 0.25 nm），
  所以 2026-07-29 那次二面角**整体反号**畅通无阻地覆盖了正确的参考几何
  （ΔG(A′→A) 5.5 → 98.8 kJ/mol）。
  现在新增第三道门，**复用同文件已有的** `boresch_committed_deviation_sigma()`
  （`abfe_pipeline.py:242`）逐自由度折算 σ，二面角先过 `_wrap_to_pi`，
  所以反号（Δφ ≈ 2φ）立刻表现为巨大 σ；阈值沿用现成常量
  `BORESCH_COMMITTED_MAX_DEVIATION_SIGMA = 4.0` / `..._WARN_... = 2.5`。
  ⚠️ **超限行为是"告警 + 保留 `orig_eq`"，不是 raise**：与本函数已有两道门风格一致；
  `orig_eq` 来自 500 帧系综均值，本来就比单帧重锚可靠，退回它是**严格更优**；
  且 4σ 在 6 个自由度上误报率约 2.8%，硬门会以约 1/36 概率无故杀掉一次 9 小时生产跑。
  真正的守门人仍是 `tests/test_boresch_dihedral_convention.py`。
  ⚠️ 实现时踩过一个坑并已钉住：那个 helper 同时返回 `sigma`（分布**宽度**）与
  `deviation_sigma`（偏离**几个** σ），读错键这道门就形同虚设（`sigma` 永远不会超 4）。
  证据：`tests/test_boresch_committed_gate.py` 新增 3 条 —— 反号被拦且保留原值、
  正常热漂移照样更新、以及"必须读 `deviation_sigma` 不是 `sigma`"的源码契约。

- [→] **BOR-04：vanishing 腿的报出 σ 偏小 —— 已并入 P1-19，不在本节重复登记。**
  同 padding 两次独立跑差 2.34σ（比跨盒子的 1.13σ 还大）这条跨跑证据，与 P1-19 的
  split-half 是同一现象的两个视角、同一个拟议修法，故合并。**不是代码 bug，是不确定度
  量化 + 采样不足**；真 bug 在 P1-23（已于 2026-08-03 修）。
  行动顺序（先固定 padding 重复跑、别再扫盒子）见 P1-19。
- [ ] **BOR-05：让 mdtraj 拿到带配体键的拓扑（原「Boresch 拓扑后续」）。** 配体侧几何回退是当前唯一可用判据，
  但管线本来就有真实键：`GromacsTopFile` 的 `top.topology.bonds()` 含配体键
  （`generate_ligand_xml_from_top` 就靠它建 `bond_neighbors`）。可选路径是把 OpenMM 拓扑
  连 CONECT 写成 PDB 供 mdtraj 读，或给估计器传 `bond_overrides`。这会让配体侧也用上真实键、
  彻底摆脱 0.22 nm 近似。动的是拓扑缓存格式，属独立改动。

## 低优先 / 后续稳健性

- [ ] **P0-9（延期，不阻挡生产重跑）：补齐 `--analyze-only` 的 stage 完整性与 ESS 契约。**
  GitHub [#67](https://github.com/Cedrus810/openmm_IBS_dev/issues/67)。这是离线分析入口的工程完整性事项；
  以后再复用主 pipeline loader 补齐 manifest、expected windows、checkpoint/f_k、
  stage checkpoint 协议与覆盖验证。

- [ ] **跨进程 production resume/rebuild 边界应作为独立 trajectory segments。** GitHub [#75](https://github.com/Cedrus810/openmm_IBS_dev/issues/75)。本批数据中 base jumps 被 `u_kn - bias` 抵消而无害，但不应依赖运气。

- [x] **膜受体前置：恒压器目前是各向同性的（2026-07-28 记）→ 已由 B1 关闭（2026-07-30）。**
  原文：`abfe_pipeline.py:1341/1345` 用的是 `openmm.MonteCarloBarostat`，它把 x/y/z 按
  同一因子缩放。放到双层膜上会把面积和厚度绑死，面积每脂（APL）会跑掉。膜体系需要
  `MonteCarloMembraneBarostat`（xy 耦合、z 独立、表面张力一般取 0）或
  `MonteCarloAnisotropicBarostat`，且需要一个 `system_type` 分支去选。
  ✅ 已实现，见下面的 MEM-00i / Phase B1 条目。注意分支参数**不叫** `system_type`
  （那个名字已被腿身份占用），叫 `environment_type`。
  盒子读取侧没问题：`abfe_pipeline.py:1567` 用 `np.linalg.norm(box_nm, axis=1)`，
  各向异性/三斜盒都能正确取边长，没有假设立方。
  另注：膜受体只影响复合物腿；溶剂腿是配体在体相水里，与膜无关，
  仍走 P0-11 修好的独立水盒逻辑，不应继承复合物盒。

- [ ] **MEM-00：膜受体–配体 ABFE 专项清单已成文（2026-07-29）。**
  设计与验收清单在仓库根的 [`memtodolist.md`](../memtodolist.md)，本轮补入了对现有代码的
  `file:line` 现状核对（§0.5）、阈值默认值（§13）、风险与放弃判据（§14）、成本估算（§15）
  和排序依赖（§17）。上面那条各向同性恒压器即该清单的 MEM-00i。
  **三件待拍板已于同日全部裁决**：(1) 生产走 charge-transfer，co-annihilation 降级为
  `co_annihilation_experimental` 只作负对照、膜生产 fail closed、旧缓存全部作废，
  `GhostIonHandler` 标记退役；(2) 脂质力场按输入自动识别力场族，amber 为首选路径，
  charmm 分支因 OpenMM 无 force-switch 默认 fail closed；(3) 首个体系 SERT、配体 +1、
  结合位点非深埋，须排除结构性 Na⁺/Cl⁻ 进入 co-ion 候选。
  膜工作在 P1-19/P1-22 收口前不进生产（P1-23 已于 2026-08-03 修）。

- [x] **B3 + MEM-00d：PME charge-transfer charging Hamiltonian 已落地（2026-08-04）。**
  主线 §17.0 的第 ② 步。`memtodolist.md` Phase B3 与 MEM-00d 绑定完成（restraint 形式
  进身份指纹，所以必须一起改，否则旧形式会被带进新路线）。

  **哈密顿量**（`ibs_engine.configure_charge_transfer_decharging`，§2.1/§2.4）：
  OpenMM 的 offset 语义是 `q(λ) = q_base + λ·q_scale`，于是

  ```
  ligand i : base 0        scale  q_i      ⟹ q(λ) = λ·q_i
  co-ion j : base share_j  scale −share_j  ⟹ q(λ) = (1−λ)·share_j
  ```

  总电荷守恒因此是**一次代数证明而不是抽查**：`Σq(λ) = Σq_base + λ·Σq_scale`，
  所以 `Σscale = 0` 就覆盖了所有 λ（含中间态，§7.2 的要求）。λ 电荷映射的唯一实现在
  `abfe_core.co_alchemical_charge_offset_plan()`（纯数学、无 OpenMM 依赖），
  `ibs_engine` 只负责把它写进 `NonbondedForce` 并用
  `charging_charge_conservation_report()` **读回真实 Force** 核对 —— 生产者与校验者
  同源，不各写一套。

  **λ=1 端必须是中性 dummy，这不是可选项**：体系总电荷在 λ=1 必须等于物理体系的。
  配体 +1、普通离子按 §4.3 配平（合计 −1）时，
  `λ=1: +1 −1 +0 = 0`、`λ=0: 0 −1 +1 = 0`。所以 co-ion 是建系时**额外预留**的一个
  电荷为 0 的 ion-shaped 粒子；**拿一个已经带电的物理盐离子顶上会让 λ=1 端总电荷变成 −1**，
  已 fail closed（`co_alchemical_charge_offset_plan` 与 `verify_...` 两处都拦）。

  **身份来源按路线分开，这是有意的**：charge-transfer 走
  `_identify_reserved_neutral_co_ions()` —— 判据只有"离子残基名 + 电荷严格为 0"两条，
  **与坐标无关**，所以 MEM-00c 那个"坐标动一点选择就翻转"的失效模式在这条路线上
  结构上不存在；数量不等于 |q_L| 就 fail closed（多了就得靠坐标挑，风险原地复活）。
  co-annihilation 仍走坐标相关的 `_select_bulk_water_counterion`，必须冻结。

  **MEM-00d restraint 换形式**：`flat_bottom_anchor_relative`，
  `0.5·k·max(0, pointdistance(x1,y1,z1, x2+dx0,y2+dy0,z2+dz0) − r₀)²`，
  k = 100 kJ/mol/nm²、r₀ = 0.5 nm（§13.1），force group 6，逐 λ 完全相同。
  井心 = **锚点原子当前位置 + 冻结位移 d0**，所以它随体系一起被 barostat 缩放，
  §2.3/MEM-00d 那个"绝对笛卡尔参考点在膜半各向异性 NPT 下把离子拖向膜"的缺陷消失。
  锚点 = 配体重原子中离配体质心最近的那一个（两条腿同一条规则 ⟹ 可用体积相同 ⟹
  restraint 自由能在 ΔG_solv − ΔG_cplx 里对消，§2.3 末条；取"离质心最近"是因为
  配体转动时该原子位移最小）。
  ⚠️ **实测过的 API 事实**：`periodicdistance` **只存在于 `CustomExternalForce`**
  （`CustomCentroidBondForce` / `CustomCompoundBondForce` 都报 unknown function），
  而 CustomExternalForce 只能吃绝对参考点。可行的组合是
  `CustomCompoundBondForce` + `pointdistance` + `setUsesPeriodicBoundaryConditions(True)`：
  打开 PBC 后 bond 内粒子会被平移到与第一个粒子同一镜像，于是 `pointdistance`
  **就是** minimum-image 距离（实测：离子 z=0.2、锚点 z=9.4、盒 z=12 → 0.2 nm）。

  **§13.1 从"事后诊断"变成"构造时可证"**：
  `|d0| − r₀ − 软墙余量 − 配体外缘 ≥ 1.2 nm` 不成立即 fail closed。
  软墙余量取"走出平坦区 2 kT 对应的位移"（k=100 时 0.316 nm）——只算平坦区半径会
  高估约束力。charge-transfer 强制这条；co-annihilation（实验对照，其反离子是从既有盐里
  挑的物理离子、不是我们摆的）只记录诊断。

  **协议版本**：`CO_ALCHEMICAL_ION_IDENTITY_PROTOCOL_VERSION` 1 → **2**
  （restraint 字典的键整体变了，v1 的 spec 一律拒绝复用）。
  退役的绝对参考点保留为审计字段但**改了键名**
  （`reference_position_nm` → `selection_time_absolute_position_nm`），
  这样还在读旧键的消费者会 KeyError 而不是静默拿到已退役的参考点。

  **B4 的边界写清楚了**：`CHARGE_TRANSFER_HAMILTONIAN_IMPLEMENTED = True`、
  `CHARGE_TRANSFER_SOLVENT_LEG_IMPLEMENTED = False`。复合物腿现在能跑（正好用于
  §6.4 要求的 pilot / λ 阶梯重估），但溶剂腿里没有 reserved co-ion ⟹ 循环闭不上 ⟹
  **不得报出 ΔG_bind**。唯一的门在 `runabfe.build_and_cache_solvent_leg`（真正建溶剂盒
  的那一处），runabfe 在开跑前就 WARNING，provenance 落
  `closes_thermodynamic_cycle: false` / `must_not_report_delta_g_bind: true`。

  **对现有可溶生产路径零影响**：中性配体（当前 Atenolol，Σq = 0）不进任何 co-ion 分支，
  `configure_pme_ligand_charge_offsets` 的 ligand-only offset 路径一字未动，
  §13 阈值常量与预平衡/stage 指纹都没变 ⟹ 已落盘的 181.00 / 157.84 /
  −5.535906 kcal/mol 基线与现有缓存不失效（§7.7）。

  **已知缺口（B3 没做，留给 B4/B5）**：
  * 配置里 `--co-alchemical-ion` 那份**声明**（进 provenance 的 `coion_identity`）与
    代码**冻结**的那份 spec 之间还没有交叉核对。声明写错一个 atom_index 不会被拦，
    provenance 会同时存在两份不一致的身份记录。要么在唯一那个选择入口加
    `declared_atom_indices` 只读核对，要么让 provenance 只记冻结那份 —— 与 B5 一起定。
  * co-ion 的 §13.1 几何判据在**构造时**只强制了"离配体够远"这一条；
    "离蛋白重原子 ≥ 1.2 nm / 离膜中面 |z| ≥ 3.0 nm / 离最近磷 ≥ 1.0 nm"目前只有
    **逐帧诊断**（§9 膜质量门里已有那四个观测量），没有建系期的 fail closed。

  handoff（做了什么 / 证据 / **被否掉的 5 个方案及原因** / 下一位的禁区 / 验证命令）：
  [handoffs/CHARGE_TRANSFER_B3_HANDOFF.md](handoffs/CHARGE_TRANSFER_B3_HANDOFF.md)。

  证据：`tests/test_charge_transfer_hamiltonian.py`（28 条：逐 λ 电荷守恒、
  λ=1/0/0.37 与**独立手写参照体系**的能量+逐原子力对照（§13.2 容差 1e-5 / 1e-3）、
  配体内部静电逐 λ 恒定、co-ion mass/LJ 逐 λ 不变、静电走 PME 而非 cutoff ghost force、
  flat-bottom 平坦区/软墙/minimum-image/盒缩放不拖拽、restraint 逐 λ 同能、
  §13.1 余量 fail closed、带电物理离子不能当 co-ion、dummy 数量不对 fail closed、
  多价配体要 |q_L| 个单价 co-ion、两条路线的 spec 不可互换、溶剂腿仍 fail closed）；
  `tests/test_coalchemical_ion_identity.py`（20 条已按新形式更新）。
  ⚠️ **尚未在真机带电体系上跑过** —— 当前 Atenolol 净电荷 = 0，这条路径不被触发；
  §17.0 第 ⑤ 步（C1 小水盒）才是它的第一次真机验证。

- [x] **MEM-00c：共炼金反离子身份漂移 —— 已修（2026-08-04）：选一次 + 冻结 + 处处只读核对。**

  修法（B3 的前置条件，不是 B5 的缓存细节）：

  ```
  ibs_engine.select_co_alchemical_ion_once()      ← 唯一允许发生选择的入口
      ↓
  abfe_core.build_co_alchemical_ion_identity()    ← 落成带指纹的 spec
      ↓
  checkpoints/coalchemical_ion_spec.json          ← 跨进程钉住身份的唯一办法
      ↓
  abfe_core.verify_co_alchemical_ion_identity()   ← dynamics / replicas / u_kn / resume 只读消费
      ↓
  带电配体而没有 spec ⟹ fail closed（"自动重选"这条路已删）
  ```

  落地细节：
  * `_select_bulk_water_counterion` 在 `ibs_engine.py` 里现在**只有一个调用点**
    （就是那个唯一入口），并在 docstring 里写明禁止别处调用；测试按出现次数钉住。
  * spec 字段复用 §3.4 的 `CO_ALCHEMICAL_ION_REQUIRED_FIELDS`，没另造 schema。
    指纹 `CO_ALCHEMICAL_ION_IDENTITY_FINGERPRINT_FIELDS` 覆盖 B5 那张作废清单的全部项：
    particle index / 离子类型（residue_name+element）/ mass / sigma / epsilon /
    端点电荷 / restraint 定义 + 参考位置；外加 protocol_version、lambda_direction、
    charge_treatment。**诊断量（当时的排序距离、水配位数）刻意不进指纹**——
    它们每次读坐标都会变一点，进指纹会让每次 resume 都误判成身份漂移。
  * restraint 参考锚点也取自 spec 而非当前坐标（MEM-00d：锚点漂了，离子被拖去的
    地方就变了）。改成 flat-bottom 时 `form` 会变 ⟹ 指纹变 ⟹ 旧缓存自动作废，这是刻意的。
  * spec 记录 `charge_treatment`，**不可跨路线复用**：co-annihilation 的端点是
    q_phys→0，charge-transfer 是 0→q_L；混用等于"声明一种哈密顿量、实际跑另一种"。
    按 charge-transfer 造出来的 spec 已验证能过 B2 的 `_validate_co_alchemical_ion_spec`
    ——这条接缝现在就钉住了，不等 B3 才发现字段对不上。
  * 消费点：3 处 `REMDManager` + 3 处 `compute_u_kn`，全部走
    `ABFEPipeline.resolve_co_alchemical_ion_spec()` 这一个 resolve，不各自读盘/各自选。

  **证据**：`tests/test_coalchemical_ion_identity.py` 重写为契约测试（20 条，CPU-only）。
  核心那条 `test_frozen_identity_survives_minimization_scale_coordinate_change`
  先证明 0.05 nm 位移确实会翻转**新鲜选择**（3 → 2），再证明冻结身份在同样坐标下
  仍然返回 3，且真实注入路径落在被钉住的粒子上。
  全套 `./tests/run_offline_tests.sh`：**977 passed / 2 skipped（环境门控）/ 0 failed**，98 s。

  ⚠️ 当前生产体系 Atenolol 净电荷 = 0（`Atenolol-rank11.itp` 的 `[ atoms ]`
  Σq = +0.000000 e），这条路径整个不被触发，落盘基线
  181.00 / 157.84 / −5.535906 kcal/mol 不变。**也就是说本修复尚未在带电体系上真机验证**
  ——要端到端验它得用 §1.1 已定的 SERT + |q|=1 配体。

  ---

  以下是修复前的诊断记录（保留，供 B3 复核漂移机制）：

  `_select_bulk_water_counterion`（`ibs_engine.py:766`）按当前坐标当场排序挑离子，
  排序主键是"到最近溶质的 minimum-image 距离"这个连续量；
  `_prepare_pme_mixed_alchemical_system`（:1576）、`_build_replicas`（:13347）、
  `compute_u_kn`（:14763）三处**各自独立**调用这条路径（`allow_charged_ligand=True`
  在 `ibs_engine.py` 里正好 3 处）。
  **漂移入口是跨进程 resume，不是单进程**：同一进程三处拿的是同一个 `self.positions`，
  但首跑走 `pre_equilibrate()` → `self.positions = equil_data["positions"]`
  （`abfe_pipeline.py:5575`）**再叠一次 2000 步快速最小化**（:5594），
  而 resume 且 `skip_equil=True` 时直接 `self.positions = traj.xyz[-1]`
  读 `pre_equilibration.dcd` 末帧（:5561）、**不做那次最小化**。
  三个 `compute_u_kn` 调用点都传 `reference_positions=self.positions`
  （`abfe_pipeline.py:2410/2996/7295`）→ 首跑用 P₁ 跑动力学、resume 用 P₂ 重算 u_kn，
  选出的离子可能不是同一个粒子 → u_kn 与动力学 Hamiltonian 静默不一致。
  实测 **0.05 nm 位移即可翻转选择结果**（远小于 2000 步最小化的原子位移量级），
  且 restraint 参考锚点随身份一起漂。
  只在带净电配体时触发（`lig_net_charge == 0` 直接短路返回），当前 Atenolol 中性，
  故落盘基线 181.00 / 157.84 / −5.535906 kcal/mol 不受影响。
  **证据**：`tests/test_coalchemical_ion_identity.py`（8 条，CPU-only，
  含 0.05 nm 翻转的可复现证据、metadata 无持久化身份、整条链路无"只读不选"参数、
  首跑/resume 坐标来源不同的源码事实）。
  ~~**为什么不在本轮修**：钉身份属于 B3 的 charge-transfer 新实现……~~
  ⟵ 2026-08-04 推翻：身份冻结与"哪条哈密顿量消费它"是**正交**的，
  spec 里记一个 `charge_treatment` 就能同时服务 co-annihilation 与 charge-transfer。
  等 B3 才做等于让 B3 同时改哈密顿量和身份机制，出问题无法二分定位 ——
  所以冻结先落地，作为 B3 的前置。

- [x] **MEM-00i / Phase B1 已实现（2026-07-30）：`system_type=membrane` + 膜恒压器。**
  上面那条「膜受体前置：恒压器目前是各向同性的」由此关闭。
  `abfe_core.py` 新增 `resolve_environment_type` / `resolve_membrane_protocol` /
  `detect_barostats` / `ensure_barostat_for_protocol` /
  `barostat_fingerprint_payload` 与 `MEMBRANE_BAROSTAT_PROTOCOL_VERSION = 1`；
  `ABFEPipeline(environment_type=..., membrane=...)`；
  `runabfe.py` 加 `--system-type` / `--membrane-*` / `--confirm-soluble-with-lipids`
  与 provenance 落盘。**纯增量**：不声明 `system_type` 时预平衡 fingerprint 与改动前
  逐字节相同（`barostat_fingerprint_payload()` 对 legacy soluble 协议返回 None），
  已有生产 checkpoint 不失效。
  实施中发现两条清单里没有的事实：
  (1) `system_type` 这个名字**已被 `run_full_pipeline(system_type="complex"|"solvent")`
  占用为腿身份**（20+ 处），因此环境类型在代码里一律叫 `environment_type`，只在
  配置键/provenance 里叫 `system_type`，混用会报错；
  (2) 原判据 `isinstance(f, openmm.MonteCarloBarostat)` **漏检**——实测 OpenMM 三种
  barostat 都直接继承 `Force`、互不为子类（`openmm.py:16777/17012/17406`），
  输入已带膜 barostat 时旧代码会再叠一个各向同性的且不报错；现按类名检测并 fail closed。
  证据：`tests/test_membrane_barostat_protocol.py`（24 条）。
  详见 [`memtodolist.md`](../memtodolist.md) 的 MEM-00i / §3.1 / §3.2 / §7.4 / B1。

- [x] **测试配置位置错误已修（2026-07-30）：`pytest.ini` 从 `tests/` 移到仓库根。**
  正式入口 `tests/run_offline_tests.sh` 是 `cd` 到仓库根后**不带路径参数**跑 pytest，
  而 pytest 只从 cwd **向上**找配置文件，`tests/` 在 cwd 之下永远搜不到，根目录当时
  也没有任何其它可识别的配置文件 → 那份 ini 对全量 pre-flight **整体静默失效**：
  `markers` 未注册（每条测试报 `PytestUnknownMarkWarning: Unknown pytest.mark.cpu_only`）、
  `addopts = -ra` 未生效（"importorskip 静默跳过一批却显示 all passed" 这个失败模式复活）、
  `filterwarnings`/`testpaths`/`norecursedirs` 同样未生效。
  只有带路径参数的调用才恰好命中旧位置，所以同一个脚本两种用法行为不一致。
  已移到根目录并删除旧位置（两份并存会把 rootdir 定成 `tests/`，
  使 `testpaths = tests` 被解释成 `tests/tests`）；`tests/README.md` 与
  `run_offline_tests.sh` 均已加不要移回去的说明。

- [ ] **MEM-RUN：memtest 膜体系首跑进展（2026-07-30）。**
  CHARMM-GUI FF-Converter 产的 AMBER 膜体系（PROA 1 + POPC 90 + Na⁺ 25 + Cl⁻ 36 +
  TP3 9542 + 配体 1，45354 原子）。配置与命令见
  [`memtest/README_MEMTEST.md`](../memtest/README_MEMTEST.md)。

  **已跑通到预平衡**：10 ns NPT 完成（480 ns/day），末段 T 303.6–305.8 K、
  PE −508k~−510k kJ/mol 无漂移、体积 438.97–440.37 nm³、密度 1.034 g/mL。

  沿途修掉的 5 个真缺陷（全部记在
  [`memtodolist.md`](../memtodolist.md) §0.5.4–§0.5.8，每条都有 file:line 与实测证据）：

  1. **身份判定靠残基名**（§0.5.4）——脂质数 ×3、水/离子静默为 0、蛋白漏 85 原子。
     改为从 `.top` 的 `[ molecules ]` + `[ moleculetype ]` 取权威组成。
  2. **`[ pairs ]` funct 2 OpenMM 不支持**（§0.5.5）——带逐对等价校验的转换，
     并收敛为**唯一**拓扑加载入口（原先 6 处裸调，补一处漏一片）。
  3. **水模型靠文件名识别**（§0.5.6）——`TP3` 认不出；改为按参数指纹匹配，
     候选参数直接读 OpenMM 自带 XML。
  4. **mmCIF 拓扑缓存丢键**（§0.5.7）——`PDBxFile` 写入端不写任何键记录，
     导致 PBC 修复把脂质撕开、最小化后 PE 4.1e13、随后 NaN。**这是首跑 NaN 的根因。**
  5. **§9 质量门双模式**（§0.5.8）——`enforce`（默认）/ `advisory`；
     advisory 不隐藏（报告仍落盘、模式进 provenance），但**不是生产资格**。

  **2026-08-02 进展**：又跑了一轮 10 ns 预平衡（30.2 min → **实测 476 ns/day**），
  这次 DCD **带 unitcell**，"重跑拿带 unitcell 的 DCD"那一项由此完成。
  但它暴露并修掉了三处，另外撞上一个还没修的：

  1. **MEM-01（已修）**：§9 质量门连续两次崩在
     `UnboundLocalError: head_by_residue`（07-31 14:51 与 08-02 16:53 逐字相同），
     `membrane_quality_gate.json` 一直是 `{"evaluated": false}` ——
     **质量门从来没有在真实膜体系上评估过一次**。见下面 MEM-01 条目。
  2. **MEM-08（已修）**：§9 的时间轴**错 20 倍**。mdtraj 读 DCD 不传播真实步长，
     `traj.time` 是整数帧号 `[0…499]`，10 ns / 500 帧（20 ps/帧）被当成 **0.499 ns**。
     见下面 MEM-08 条目。
  3. **MEM-03（已实现）**：APL 蛋白横截面校正落地，§13.3 的绝对值门可以开了。
  4. **Stage 0 NaN（未修，已加体检）**：17:03 Boresch attachment 腿在
     `ibs_engine.py:1425` 出 `Particle coordinate is NaN`。
     ⚠️ 第一个实际跑的态是 **λ=1.0**（全强度限制力），不是 λ 列表里的第一个 0.0 ——
     `order = list(range(K-1, -1, -1))` 是从全强度端往下扫。
     本轮只加了起点体检（MEM-06）让下次失败带上下文，**没有宣布根因**。

  **配置决策（2026-08-02）**：
  - `n_equil_steps` 5e6 → **5e7（100 ns，≈5.0 h）**：§9 末段窗口是 20 ns，
    10 ns 轨迹在结构上永远过不了门；且 `MEMBRANE_MIN_EQUILIBRATION_NS = 100.0`。
    （那道 100 ns 配置期预检此前没挡住，因为本体系声明
    `upstream_equilibration_status = completed_length_unrecorded` → 预检不适用。）
  - `membrane_quality_gate` advisory → **enforce**（门在预平衡之后判，不会白烧那 5 h）。
  - `boresch_source` auto → **simple**（纯几何涨落估算，不加载 MACE/e3nn）。
  - 输出目录换成 `output_membrane_100ns`：旧目录的
    `checkpoints/boresch_equilibrium_committed.json` 是"resume 强制复用"的，
    里面装着 auto 估出的锚点，换估算器后必须重新生成。

  **2026-08-03 进展**：100 ns 跑完，§9 门在 `enforce` 下**通过**
  （7 项余量 2–6 倍，`equilibration_length_ns` 已按 MEM-11 降为诊断，τ = 18.467 ns、
  比值 5.42）。Stage 0 的 NaN 根因已定位并修复，见 MEM-15。
  **下一步**：修复后重跑，让 Stage 0 走到 stage1/stage2。仍未做的：
  - 序参量 / 疏水核异常水 / 脂质横向弛豫时间尺度这三个量的**量级**与文献对照
    （判定层与提取器已就绪且有测试，但数值口径未用真实体系验过）；
  - 校正后 APL 与 POPC 文献值 0.645 的实际差距（10 ns 那轮的数已被方法修订作废，
    见 MEM-03）。

  ⚠️ 按 §17，膜工作在 P1-19 / P1-22 收口前**不进生产**（P1-23 已修）。

- [x] **MEM-01：§9 膜质量门在分子路径下崩在 `head_by_residue`（2026-08-02 已修）。**
  §0.5.4 把叶片划分从"按残基"改成"按分子"时分成两条分支
  （`abfe_core.py` 的 `membrane_observables_from_trajectory`），但
  `leaflet_composition` 那段仍然只读**残基分支的局部变量** `head_by_residue`。
  memtest 有 `.top` 组成 → 走分子分支 → 该变量从未绑定 → 整个门崩掉。
  **这是"同一件事多个入口，补一个漏一片"的第四次**（前三次见 memtodolist §0.5.4–§0.5.6）。
  修法是**改生成器**：两条分支统一产出 `head_units: List[(单元标签, 头基原子 index)]`，
  `head_indices` 与 `leaflet_composition` 都只从它派生，分支专属变量整个消失。
  分子路径的标签取 moleculetype 名（`POPC`），不是构成残基名（PA/PC/OL）。
  **为什么漏出来**：`tests/test_membrane_observable_extractor.py` 原有 22 条测试
  **没有一条**传 `composition=`，分子分支零覆盖。已补 5 条分子路径测试
  （含"模块化残基命名下按残基必须报错、按分子才对"）。
  **证据**：该文件现 40+ 条全绿；并用
  `tools/diagnostics/evaluate_membrane_quality_gate.py` 对真实 08-02 DCD 复判，
  确认从 `UnboundLocalError` 变成"跨度 9.980 ns 覆盖不了末段窗口 20 ns"这条正确失败。

- [x] **MEM-08：§9 质量门的时间轴错 20 倍（2026-08-02 已修）。**
  mdtraj 读 DCD **不传播真实步长**：`DCDTrajectoryFile.read_as_traj` 给出的
  `traj.time` 是**整数帧号**。实测 memtest 那条 10 ns / 500 帧
  （10000 步 × 2 fs = 20 ps/帧）的轨迹，`traj.time` 就是 `[0, 1, 2, …, 499]`，
  于是时间轴被当成 **0.499 ns**。提取器原先只校验"存在且单调递增"，
  帧号完全满足 —— 这条守卫（docstring 明写"不允许用帧号冒充 ns"）对 DCD 是 fail-open。
  **两道门往相反方向坏**：末段 20 ns 窗口过严（要 400 ns 真实时间才够），
  而"预平衡 ≥ 一个脂质横向弛豫时间"过松（MSD 拟合的 D 被同一倍数放大）。
  修法：时间轴由 `reporter 保存间隔 × integrator 步长` 显式重建
  （`abfe_core.pre_equilibration_frame_interval_ps()`；两个常量
  `PRE_EQUILIBRATION_TRAJ_INTERVAL_STEPS` / `PRE_EQUILIBRATION_TIMESTEP_PS`
  由写轨迹的 reporter/integrator 与判门的一侧**共用**，值不变故逐位兼容），
  不传时遇到整数 dtype 的时间数组一律拒绝；
  实际用了哪条写进 `diagnostics.time_axis_source`。
  **证据**：`tests/test_membrane_observable_extractor.py` 的时间轴一节（6 条，
  含复现 memtest 那 20 倍的具体数字 + AST 契约禁止 DCDReporter 写字面量间隔）；
  离线复判真实 DCD 现报 **9.980 ns**（此前 0.499 ns）。

- [x] **MEM-03：APL 蛋白横截面校正 + §13.3 绝对值门开启（2026-08-02）。**
  含蛋白膜的 raw APL 把跨膜蛋白占掉的横向面积也摊给脂质（实测 0.807 vs
  POPC 纯脂文献 ≈ 0.645），所以此前干脆不设 `literature_apl_nm2`，
  代价是 §13.3 那道门整条缺席。现在新增观测量 `apl_protein_corrected_nm2`
  并让绝对值门比它，`criterion` 写明是否校正（缺校正序列时退回 raw 并标
  `..._uncorrected`，老报告仍可判）。
  ⚠️ **走过一次弯路**：第一版用"蛋白重原子外扩 0.17 nm 求并集"，实测**系统性高估**
  蛋白面积（沿周长多加一圈，约 1.7 nm²），校正后 APL = 0.564、比文献值低 12.6% ——
  门会因**方法偏差**而不过，迟早被调参调绿。已改为**无探针半径**的
  最近参考原子归属（Voronoi 式，APL@Voro 同思路），边界自动落在脂质与蛋白原子中间；
  栅格边长是唯一方法参数，2× 粗栅格的复算结果随报告落盘。
  `MEMBRANE_QUALITY_GATE_PROTOCOL_VERSION` 1 → **2**（v1/v2 报告不可直接比较：
  per-ns 斜率与 APL 绝对值口径都变了）。方法参数进 `acceptance_thresholds`。

- [x] **MEM-02：§9 质量门收敛为唯一实现 + 离线复判工具（2026-08-02）。**
  接线原先只存在于 `ABFEPipeline._evaluate_membrane_quality_gate_after_equilibration`，
  于是"验证质量门"唯一的办法是**重烧一遍预平衡**（10–100 ns，5 h 起）——
  08-02 那次就是又烧 30 min 才看到同一个 `UnboundLocalError`。
  现在提取 → 判定 → 落盘收敛到 `abfe_core.run_membrane_quality_gate()`，
  pipeline 与 `tools/diagnostics/evaluate_membrane_quality_gate.py` 共用它
  （§0.5.7 的教训：离线重建与生产路径不一致，白花过好几轮）。
  离线工具纯 CPU、不建 Context，只调生产函数
  （`runabfe.load_native_system(require_bonded_topology=True)` +
  `classify_system_composition`），默认不落盘以免覆盖那次运行的记录。
  顺带：advisory 下"判不了门"时**观测量也一起落盘** —— 那些数字是烧了 GPU 才有的，
  判不了门不等于没价值（例如"跨度不够"时 APL/膜厚的实测值仍是延长平衡的唯一依据）。

- [x] **交付教训（2026-08-03）：改动只要影响预平衡坐标，就会作废已完成的预平衡，
  必须在用户启动前明确警告。**
  MEM-15 改了 `repair_pbc_molecule_integrity`（把约束补成键）→ 修复后坐标变化 →
  `pre_equilibrate` 的 `requested_fingerprint` 与落盘那份不符 → `resume` 被关掉 →
  **已经跑完的 100 ns 从头重跑**，而且 `append=False` 已把原来 2.72 GB 的轨迹与
  `pre_equil.chk` 覆盖掉，**没有东西可还原**。实测：15:22 起跑，1h16m 后到 25.19 ns，
  需再约 3.8 h。
  ⚠️ 本来**不需要**重跑：撕水只发生在 100 ns **之后**那次 PBC 修复上（水扩散过边界后
  才会被逐原子回卷）；100 ns **开始**时用的是 CHARMM-GUI 的 `.gro`，水整分子在盒内，
  旧修复对它是空操作。所以那 5 h 是被指纹敏感性连带掉的纯浪费。
  **规矩**：以后任何改动只要碰到 `repair_pbc_molecule_integrity` /
  `center_system_rigidly` / `load_native_system` 的坐标路径，或碰到
  `_pre_equilibration_fingerprint` 的任何输入，都必须在交付命令时**显式写明**
  「这会作废已有预平衡，需重跑 X 小时」，让用户先决定；不能让他跑起来才发现。

- [x] **MEM-17：resume 会往预平衡 DCD 追加重复帧，污染 §9 时间轴（2026-08-03 已修）。**
  `pre_equil.chk` 每 100000 步（200 ps）一次、DCD 每 10000 步（20 ps）一帧，
  所以中断后 `DCDReporter(append=True)` 会把「checkpoint → 中断点」之间那**最多 9 帧**
  写第二遍（同一段模拟时间、两条不同随机路径）。而 MEM-08 之后 §9 的时间轴是按
  `帧号 × frame_interval_ps` 重建的（mdtraj 读 DCD 只给整数帧号），
  多 N 帧 ⟹ 时间轴凭空长 N×20 ps、末段 20 ns 窗口取的不是真正的末段，
  **且没有任何现象**能提示这件事。
  ~~修法：`membrane_quality_inputs` 新增 `expected_pre_equilibration_frames`，
  `run_membrane_quality_gate` 在 `md.load` 之后对账，不符即 fail closed。~~
  **⛔ 2026-08-03 当天回退：这道帧数对账已从 `abfe_core.run_membrane_quality_gate`
  与 `runabfe.py` 删除（用户决定），且不许再加回来。**
  * 现象是**真的**：那条 100 ns 实测 5001 帧。`pipeline.log` 记了两次 resume
    （断点后分别还剩 11,500,000 / 9,500,000 步 ⟹ 已完成 38,500,000 / 40,500,000），
    `pre_equilibration_monitor.csv` 10004 行（应 10000）且两处非单调
    （`38505000` 重复 1 行、从 `40515000` 回退到 `40505000` ⟹ 重复 3 行）。
    折成 DCD（每 10000 步一帧）：`(38500000,38505000]` 内无整帧 ⟹ +0；
    `(40500000,40515000]` 含 `40510000` ⟹ **+1**。合计 5001，重复帧在索引 4050/4051。
  * "第 0 帧也算所以 5001 对"**不成立**：DCD 头 `ISTART=10000, NSAVC=10000`，
    monitor 首行是 5000 而非 0 —— OpenMM reporter 从第 `interval` 步开始写，
    不写初始帧，所以整除不 +1 本来是对的。
  * 删它的理由：判据没算错，但它 fail closed 拦住的是**主线**，代价是重跑 8 h 预平衡。
    根因不在质量门，在 `abfe_pipeline.pre_equilibrate`（约 1726 行）：
    `DCDReporter(append=resume_from_chk)` 没有先把 DCD 截断到 checkpoint 对应的帧边界
    （`floor(chk_step / PRE_EQUILIBRATION_TRAJ_INTERVAL_STEPS)` 帧），monitor.csv 同理。
    **真正的 TODO 见 MEM-17b。**
  * 副作用（已知，暂不处理）：门不再短路 ⇒ §9 现在会真的 `md.load` 整条 2.72 GB
    轨迹并算完观测量，耗时与内存都不小。
  * 现有那条 DCD 若要救：删掉索引 4050 那一帧（保留 4051 起的续跑路径）即得连续 5000 帧，
    不必重跑、不影响 checkpoint 与预平衡指纹。
  证据：`tests/test_membrane_observable_extractor.py::test_frame_count_reconciliation_is_gone_and_must_not_come_back`。

- [ ] **MEM-17b：resume 时预平衡 DCD / monitor.csv 不截断到 checkpoint 边界（MEM-17 的真根因）。**
  `abfe_pipeline.pre_equilibrate` 里 `DCDReporter(append=resume_from_chk)` /
  `StateDataReporter(append=resume_from_chk)` 直接追加，于是「checkpoint → 中断点」
  之间已经写过的帧/行会被**再写一遍**（同一段模拟时间、两条不同随机路径）。
  修法：resume 前按 checkpoint 的 `currentStep` 把 DCD 截到
  `floor(chk_step / PRE_EQUILIBRATION_TRAJ_INTERVAL_STEPS)` 帧、
  monitor.csv 截到 `floor(chk_step / MEMBRANE_EQUILIBRATION_MONITOR_INTERVAL)` 行，再 append。
  ⚠️ 只改写盘，不碰坐标、不进预平衡指纹 ⟹ 不作废已有预平衡。

- [ ] **MEM-16：`equilibrium_is_done()` 的指纹恒不匹配（2026-08-03 发现，⚠️ 是个雷，不许顺手修）。**
  同一个 `_pre_equilibration_fingerprint()` 被喂了**不同阶段的坐标**：
  * `equilibrium_is_done()`（`runabfe.py:405` 的调用点）用 `pipeline.positions`，
    而实测那是 **GRO 初始值** —— 日志明写 `⚠️ 坐标回退到 GRO 初始值`，因为
    `load_native_system` 的 `expected_pre_equilibration_fingerprint` 参数**全仓无人传**，
    所以它从不读预平衡 DCD 末帧；
  * 落盘那份指纹是 `pre_equilibrate()` 写的，而它在函数**开头**先跑了
    `repair_pbc_molecule_integrity()`，指纹算的是**修复后**坐标。
  于是每次运行都报「指纹不匹配，将重新执行预平衡」→ 重新进 `pre_equilibrate` →
  从 checkpoint 恢复、**剩余 0 步** → 实测每次白花 **78 s**（15:00:09 → 15:01:27）。
  同时 **MEM-09 那道"真的跑完了吗"的检查被这个伪不匹配绕过**（真正拦住的是
  `pre_equilibrate` 内部的 `steps_remaining = 目标 − currentStep`）。

  ⚠️ **不要用"让两边指纹一致"来修**：当前唯一把 `self.positions` 从 GRO 初始值换成
  平衡态坐标的路径，就是 `pre_equilibrate()` 内部那次 checkpoint 恢复。一旦
  `equilibrium_is_done()` 返回 True，该函数被整段跳过，就可能**静默地用未平衡坐标**
  去跑 stage1/stage2 —— 那比现在多花 78 s 严重得多。
  正解是让复用路径**显式**恢复平衡坐标（给 `load_native_system` 传那个 fingerprint，
  让它走已实现但从未被启用的"读 DCD 末帧 + 同步盒矢量"分支，`runabfe.py:1225-1237`），
  然后才谈指纹一致。必须单独立项、单独验证，并核对 §7.7 的可溶基线不受影响。

- [x] **MEM-15：刚性水被 PBC 修复撕开 → Stage 0 的 NaN（2026-08-03 已修）。**
  100 ns 那轮在 attachment 腿第一个 λ 态出 `Particle coordinate is NaN`。
  **与 Boresch 无关**（λ=1 时 `E_Boresch = 0`、λ=0 也活 100 ps、锚点几何干净）。
  根因：`.top` 的 TIP3P 用 settles ⟹ O–H/H–H 只以**约束**存在
  （实测 `topology.bonds()` 里涉及水的键数 = **0**，约束 28626 个 = 9542×3，
  `HarmonicBondForce` 里 0 项），而 `repair_pbc_molecule_integrity` 用
  `mdtraj.image_molecules()` **按 topology 的键**归组分子 ⟹ 每个水原子被当成独立分子
  ⟹ 跨边界的 **243 个水**被逐原子回卷、O 与 H 落到不同镜像。后果：
  **729 个 PME 排除对跨盒（最远 13.76 nm，cutoff 1.0）** ⟹ 虚假 **−30.9 MJ/mol**；
  **约束求解器**要在相距 5.9–12.4 nm 的 O/H 间满足 0.0957 nm ⟹ 不收敛 ⟹ **<1 ps NaN**。
  ⚠️ **这个损坏对既有诊断全部隐形**：水没有键力项，所以键能（9525.72，与健康坐标逐位
  相同）、最大键长（0.19 nm）、角/二面角能量全部正常；PME 误差是平滑长程项，所以
  `max|F|` = 5292 正常 —— **连 MEM-06 的起点体检都通过**。也因此离线忠实重放
  （同起点/同种子/走完整条 λ 序列 2.4 ns）**不复现**：它用的 rebalance 末帧还没过这步
  修复，排除对最远只有 0.4331 nm。
  修法：(1) 修复前把 System 的**约束补成键**再交给 `image_molecules()`（实测补 28626 个）
  —— 用约束而非 `.top` 的 `[ molecules ]` 区间，因为约束在任何输入来源下都有；
  (2) `assert_starting_state_is_sane` 新增**镜像一致性**检查（排除对/约束对必须同镜像），
  这不是冗余：力检查构造性地看不见这类损坏；
  (3) 这条腿此前**一帧轨迹、一行监控都不写**，现在落 `attachment/` 下的
  `stage0_attachment_start.npz`（起点坐标，可离线确定性复现）、
  `stage0_attachment_inputs.json`（Force 清单/锚点镜像距离/λ/种子/步数）、
  `stage0_attachment_monitor.csv`（头 1000 步 50 fs 一行）。
  **验证**：坏坐标喂守卫 → 力检查打印正常的 5292 后**紧接着** raise
  「729 个 nonbonded_exceptions 对跨了周期镜像（最远 13.760 nm）」；约束补键后重修 →
  排除对 0.433 nm、约束对 0.151 nm、`PE = −508657` 回到健康值，31 MJ/mol 消失。
  ⚠️ **这是 §0.5.7 那个根因的第二次**（那次是 mmCIF 丢非标准残基键撕开脂质）。
  共同教训：**"按 topology 的键归组分子"这个前提本身必须验证。**
  **证据**：`tests/test_membrane_barostat_protocol.py::
  test_torn_rigid_water_is_caught_before_dynamics` /
  `test_pbc_repair_promotes_constraints_to_bonds_for_grouping`；
  详见 [`memtodolist.md`](../memtodolist.md) §0.5.10。

- [x] **MEM-10：`superpose` 原地修改坐标，污染 6 个 §9 观测量（2026-08-02 已修）。**
  100 ns 预平衡跑完后 `enforce` 门卡在 `equilibration_length_ns`
  （100.04 vs 阈值 139.362）。查下来**那个 139.362 本身是错的**：
  `abfe_core.py:3607` 的 `aligned = traj.superpose(traj, 0, atom_indices=protein_backbone)`
  —— mdtraj 的 `superpose()` **原地改 `traj.xyz` 并返回 self**，所以这一行之后所有读
  `traj.xyz` 的量都在用"对齐到蛋白骨架"的坐标，而 `midplane`/`upper_z`/`lower_z`
  是在 3502–3503 行、对齐**之前**算的。
  受污染：脂质横向弛豫 τ（**放大 12 倍**：原始坐标 11.57 ns → 对齐后 139.36 ns，
  与门里报的 139.362 逐位一致）、跨膜倾角（在对齐帧里测 → 漂移被压掉）、
  蛋白横截面 / 校正后 APL、疏水核内水、水层间隙、沿法向密度分布。
  **而那行 superpose 对它本来要服务的三个 RMSD 毫无作用**：
  `md.rmsd(..., atom_indices=X)` 内部会自己在 X 上重新最优拟合，实测
  pocket 0.069400（对齐前）vs 0.069400（对齐后）、ligand 0.050201 vs 0.050201 ——
  它是**纯有害**的一行。
  修法：主 `traj` 一个字节不动；只为三个 RMSD 建 backbone ∪ pocket ∪ ligand 的
  **子集副本**（`atom_slice`），在副本上对齐。旋转矩阵只由骨架决定，骨架 RMSD 数值等价。
  **证据**：`tests/test_membrane_observable_extractor.py::
  test_extractor_does_not_mutate_the_caller_trajectory`（直接钉根因）+
  `test_rigid_body_motion_of_the_protein_does_not_change_lipid_relaxation`。
  `MEMBRANE_QUALITY_GATE_PROTOCOL_VERSION` 2 → **3**。

- [x] **MEM-11：脂质横向弛豫时间尺度——修估计器 + 从硬门降为诊断（2026-08-02）。**
  估计器原先用**单一参考帧**（每个 lag 只有一个样本）+ 过原点最小二乘拟合**全部**
  lag（权重 ∝ lag²）。而实测 MSD ~ t^**0.80**（亚扩散），这样拟合必然偏：
  同一条 100 ns 轨迹只改"用到前多少 ns"，τ 就是
  30.1 → 38.0 → 24.1 → 13.2 → 10.8 → 11.6 ns（**非单调乱跳**）。
  改成**时间平均 MSD**（多时间原点）+ 声明 lag 窗口（5–30 ns，常量
  `LIPID_LATERAL_MSD_FIT_LAG_MIN/MAX_NS`）带截距线性拟合，同一条轨迹给
  D = **0.008664** nm²/ns → τ = **18.467 ns**，与 POPC 文献 D ≈ 0.008 给出的 20 ns 吻合。
  拟合出的 D、幂律指数 α、实际用的窗口全部进 `diagnostics.lipid_lateral_diffusion`。
  **判据降级**：`equilibration_length_ns` 从硬 `checks` 移出，改为
  `statistics.equilibration_vs_relaxation` 诊断。依据：§9 原文只要求「**记录**弛豫
  尺度、**用它论证**」，`MEMBRANE_EQUILIBRATION_MIN_RELAXATION_MULTIPLE = 1.0`
  这个倍数是本实现自加的（旧注释已承认「§13 未给此倍数」）；常规膜蛋白平衡的判据是
  APL/膜厚/序参量/RMSD 走平（那些仍是硬门，本体系余量 2–6 倍）；且 τ 是方法依赖量，
  当硬门会产生假阴性。与 occupancy 退役为 diagnostics-only 同一先例（TEST-GATE-01）。
  ⚠️ **降级不等于不记录**：τ 与比值照样落盘，"永不弛豫的膜"会以极大 τ、比值 < 1
  被如实报出（`test_a_membrane_that_never_relaxes_laterally_is_reported_not_silently_dropped`
  与 `test_equilibration_shorter_than_lipid_relaxation_is_recorded_not_gated` 钉住）。
  ⚠️ **不得为了让某次运行通过而把它塞回 `checks`**（代码注释与测试双重钉住）。
  顺带：合成 fixture 从"沿固定方向确定性位移"改成**真正的二维随机行走** ——
  旧构造只让单参考帧 MSD 等于 4Dt，本身不是扩散运动，只是恰好配合了旧估计器。

- [x] **MEM-12：APL 与纯脂文献值的 3% 门——含蛋白膜不启用，只落诊断（2026-08-02）。**
  **先认账**：上一轮我说"§13.3 的绝对值门可以开了"，但**没有**把
  `literature_apl_nm2` 写进 `memtest/membrane_input.json`，所以那道门一直没跑。
  现在实测（100 ns）校正后 APL = 0.5907 vs POPC 0.645，差 **8.42%** —— 真设了也不过。
  含蛋白膜差百分之几**不构成"体系有问题"的证据**：annular lipid 被跨膜蛋白减速重排、
  蛋白占本体系约 24% 横向面积（8.5–9.2 / 36.3 nm²）、90 脂小膜片还有有限尺寸效应。
  改法：新增**诊断专用**字段 `pure_lipid_reference_apl_nm2`（= 0.645），
  名字与 `literature_apl_nm2` **刻意不同**（后者才是"要判"的开关），
  判定层落 `statistics.apl_vs_pure_lipid_literature`（校正后 APL、参考值、偏差%、
  `is_gate: false`、不判的理由）。这道 3% 门应在**无蛋白 POPC slab** 上启用
  （memtodolist §8.2），那里它才有定义 —— 不是删掉，是放到能判的地方。

- [x] **MEM-13：口袋/配体 RMSD 测的是内部构象，不是 pose 漂移（2026-08-02 已修）。**
  `md.rmsd(aligned, aligned, 0, atom_indices=pocket)` 会在口袋/配体**自身**上再做一次
  最优拟合，所以测到的是内部构象变化，而 §9 要的是"配体 pose"。
  改成"对齐蛋白骨架后**不重拟合**的位移 RMSD"。实测差别（末段 20 ns）：
  不重拟合 0.0857 / 0.0833 nm，重拟合 0.0760 / **0.0493** nm（配体差 1.7 倍）。
  阈值 0.20 / 0.25 **不变**，修正后仍有 2.3× / 3.0× 余量 —— 这不是放宽。
  `protein_backbone_rmsd_nm` 保持 `md.rmsd`（骨架自身的拟合 RMSD 本来就该重拟合）。

- [x] **MEM-14：§9 门写在 `pre_equilibrate()` 内部，重跑即绕过（2026-08-02 已修）。**
  `_update_stage_status("equilibration","completed")` 与预平衡指纹写在
  `abfe_pipeline.py:1745-1763`，**在门之前**；门在 `pre_equilibrate()` 内部。于是
  **门失败 → 原样重跑 → `equilibrium_is_done()` 为真 → `pre_equilibrate()` 跳过 →
  门也一起被跳过 → 直接进 Stage 0**。`enforce` 的语义被控制流击穿，而它存在的
  全部意义就是"门没过不许继续烧 λ 窗口"。
  修法：新增幂等的 `ABFEPipeline.ensure_membrane_quality_gate_passed()`
  （内部仍调唯一实现 `abfe_core.run_membrane_quality_gate`，进程内缓存报告），
  接在**每个消费预平衡的入口**：`run_full_pipeline()` 与
  `runabfe._run_boresch_attachment_only()` / `_run_complex_charging_only()`
  （后两个用 `--only-*` 增量重跑，同样在消费那次预平衡，此前也是绕过路径）。
  `pre_equilibrate()` 里那次调用保留（刚产出就 fail fast）。可溶体系短路返回。
  **证据**：`tests/test_membrane_barostat_protocol.py::
  test_quality_gate_cannot_be_bypassed_by_rerunning`（源码契约覆盖三个入口）。

- [x] **MEM-09：被中断的预平衡会被当成"已完成"复用（2026-08-02 已修）。**
  `runabfe.equilibrium_is_done()`（`runabfe.py:405`）原先只查
  「轨迹存在（>10 KB）+ checkpoint 存在 + 指纹相符」，而
  `pre_equilibration_fingerprint.json` 是 `pre_equilibrate()` 在**第一步之前**
  就写下的（`abfe_pipeline.py:1488`，为的是让被中断的运行下次能认出自己的身份），
  它记的 `n_steps` 是**目标**步数、不是已完成步数。
  于是一次 100 ns 跑到 40 ns 被杀掉的运行，下一次调用会被判成"已完成"→
  `pre_equilibrate()` 整段跳过 → **连它内部的 §9 膜质量门一起跳过**，
  而 provenance 与指纹都写着 100 ns。这是"短平衡冒充长平衡"，`enforce` 拦不到。
  修法：追加**完成判定** —— `checkpoints/pipeline_state.json` 的
  `equilibration.status` 必须是 `completed`（它只在步进真正结束后才写），
  且 `total_steps` 不小于指纹文件记录的目标步数；缺该文件时保守视为未完成。
  三种不通过的情形都给出"加 `--resume` 可从 checkpoint 续跑"的提示。
  ⚠️ **对既有目录零影响**：实测 `output_lrc_fix/checkpoints/pipeline_state.json`
  与 `memtest/output_membrane/...` 都有 `equilibration.status=completed`
  且 `total_steps=5000000`，仍判为已完成（§7.7）。
  顺带把 `memtest/abfe_config.json` 的 `resume` 设为 `true`：100 ns 约 5 h，
  CheckpointReporter 每 100000 步（200 ps）一次，中断最多丢 200 ps；
  不设 resume 时 `pre_equilibrate` 用 `append=False` 重开 DCD，已跑的部分直接作废。
  **证据**：`tests/test_membrane_barostat_protocol.py::
  test_interrupted_pre_equilibration_is_not_mistaken_for_a_finished_one`（4 种情形）
  与 `test_pre_equilibration_checkpoint_interval_bounds_the_work_lost_on_resume`。

- [x] **MEM-06：Boresch attachment 腿加起点体检（2026-08-02）。**
  §0.5.7 给 `pre_equilibrate` 加的"最小化后 max|F| 门"原先是**内联**在那里的；
  现在抽成 `abfe_core.assert_starting_state_is_sane()`（**唯一一份**），
  attachment 腿在第一次 `simulation.step` 之前调同一个函数，
  并额外落盘：六个 Boresch 力常数、**起点实测六个几何量 vs 已提交平衡值**
  （走 BOR-01 之后唯一正确的 `calc_boresch_from_last_frame`，不写第五份二面角副本）、
  逐 force group 能量。报错按 g0 异常 / 只有 Boresch 力组异常分两条排查路径。
  ⚠️ 这段**只读**：不最小化、不改坐标/速度/参数，所以对既有可溶生产路径数值逐位无影响。
  日志新增一行明说"第一个实际跑的态是 λ=1.0"，避免再按"λ 列表第一个是 0.0"误判。

- [ ] **MEM-BONDS：mmCIF 拓扑缓存丢掉非标准残基的键（可溶路径同类隐患待评估）。**
  2026-07-30 在膜体系上实测确认：`app.PDBxFile.writeFile` **不写任何键记录**
  （写入端没有 `struct_conn` / `chem_comp_bond`），读取端只能靠
  `createStandardBonds()` 补**标准残基**的键。所以 `topology.cif` 往返之后，
  非标准残基（脂质 `POPC`=PA+PC+OL、配体 `MOL`、离子）的键**全部静默丢失**，
  而 `load_native_system` 原先只校验原子数、不校验键数。

  `pre_equilibrate` 之前的「PBC 分子完整性修复」靠 topology 的键判断分子边界，
  键丢了就把跨周期边界的分子撕开。膜体系实测：丢键时最小化后
  PE = **4.109e+13** kJ/mol、max|F| = **3.72e+09**（落在脂质尾链氢 `PA334/H8S`），
  几千步后 NaN；从 `.top` 重建（有键）则 PE = −648536、max|F| = 2501（水），正常。

  **膜体系已修**（`require_bonded_topology=True` 优先从 `.top` 重建 + 键数校验 +
  最小化后受力合理性门），详见 [`memtodolist.md`](../memtodolist.md) §0.5.7。

  **⚠️ 可溶路径未动，需要评估**：可溶复合物腿与溶剂腿的配体 `MOL`（41 原子）
  在 mmCIF 往返里同样丢键，所以现有生产跑一直在用「配体无键」的拓扑做 PBC 修复。
  配体是单个小分子且被刚性居中，很可能没被撕到——但这是**推测，未验证**。
  改成优先 `.top` 会移动预平衡起点、改变落盘基线
  （181.00 / 157.84 / −5.535906 kcal/mol），属 §7.7 / R7 管辖，必须单独立项：
  1. 先**只验证不改**：对当前可溶输入分别用 mmCIF 拓扑与 `.top` 拓扑跑一遍
     PBC 修复，比较配体与蛋白坐标是否逐位一致；
  2. 若一致 → 记为"无影响"，关闭本条；
  3. 若不一致 → 说明现有基线的预平衡起点本身受此影响，需要决定是否重跑基线。

- [ ] **锁定 `pymbar-core` 版本。** GitHub [#76](https://github.com/Cedrus810/openmm_IBS_dev/issues/76)。当前 uncertainty semantics 依赖未锁版本，建议 pin/bound 并记录 intended method。

## 长期研究项

- [ ] **R-02：用真实 bridge 数据重新标定 Shadow-Coulomb。** GitHub [#38](https://github.com/Cedrus810/openmm_IBS_dev/issues/38)。
- [ ] **R-03：评估 point-ESS lambda insertion heuristic。** GitHub [#39](https://github.com/Cedrus810/openmm_IBS_dev/issues/39)。
- [ ] **R-04：评估 ACE softcore floor、pilot metric 样本数、PBC 重居中。** GitHub [#40](https://github.com/Cedrus810/openmm_IBS_dev/issues/40)。


