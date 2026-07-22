# 当前行动清单

更新：2026-07-22。此文件合并并取代根目录旧 `todo2.txt` 审查报告和 `todolist.md`；历史原文保存在 [archive/todolist-2026-07-20.md](archive/todolist-2026-07-20.md)，审计证据见 [status/AUDIT_STATUS.md](status/AUDIT_STATUS.md)。

## P0/P1

以下为 2026-07-22 新增、**尚未执行验证**的 P0/P1 问题；在代码/运行证据复核前，不应标记为已修复。

### P0

- [ ] **P0-1：主 System 缓存未绑定当前体系输入。** `system_cache_exists()` 只检查 `system_native.xml` 与 `ligand_indices.json`，未绑定当前 `--gro`、`--top`、配体名、include 文件或力场参数；加载还优先采用旧 mmCIF/DCD。换体系后可能静默复用旧 System；若旧 mmCIF 损坏且新 TOP 原子数相同，还可能形成旧 System XML 与新 Topology 的混合对象。必须以完整输入/参数哈希做 fail-closed 缓存身份校验。

- [ ] **P0-2：溶剂腿缓存未绑定配体或力场。** `solvent_cache_manifest.json` 目前仅记录协议、盐浓度、neutralize 和 Na/Cl 数量，未记录配体拓扑/参数/FFXML 哈希、电荷、原子数或 complex-leg 来源。换配体或 Hamiltonian 后可能复用旧溶剂腿，导致两腿并非同一配体 Hamiltonian；manifest 必须加入这些身份字段并拒绝旧缓存。

- [ ] **P0-3：Boresch 再平衡以非 minimum-image 距离改写 `r0`。** `np.linalg.norm(H0 - L0)` 未做 PBC/minimum-image，可能把跨周期像的正常 0.5–0.8 nm 锚点距离改写为盒子尺度的 `r0`；`_wrap_ligand_to_box()` 整体平移体系，不能改变受体–配体相对位移。应以 box vectors 的 minimum-image 向量计算距离后才允许更新平衡值。

- [ ] **P0-4：IBS 所谓 PBC 修复会使两个锚点重叠。** raw H0–L0 距离大于 1.5 nm 时，`pos_chk[ligand] += H0 - L0` 会令 `L0'=H0`，不是周期盒向量 unwrap。它可产生零距离约束、非键排斥爆炸，且会把真实脱位的配体瞬移到受体原子上；必须删除该坐标移动并使用真正的 minimum-image/unwrap 策略。

- [ ] **P0-5：缺失 IBS bias/base 文件被静默补零。** checkpoint 恢复和最终分析在缺少 bias/base 时以 `np.zeros(number_of_frames)` 继续，但 MBAR 采样分布依赖 `E_base + V_bias`；resume skip 也未核验 bias/base 是否存在及与 energy 的 hash/长度匹配。必须 fail closed，禁止以零替代有偏采样记录。

- [ ] **P0-6：DEXP 拟合模型与生产模型身份不一致。** Orb fitter 拟合全局 `alpha_vdw`、`beta_vdw`、`r0_vdw`、`A_fit`、`B_fit`、`offset_c0` 模型；生产 `DEXPSurrogatePotential` 却使用 pair-specific `sigma_ij`/`epsilon_ij` 形式，`from_dict()` 不读取 `r0_vdw`、`A_fit`、`B_fit`。CLI 加载拟合 JSON 后会静默丢弃核心拟合参数；必须统一拟合与生产 Hamiltonian，或拒绝不兼容参数文件。

- [ ] **P0-7：DEXP 的 `offset_c0` 被保存但未进入能量。** fitter 计算并保存最优常数偏移，`DEXPSurrogatePotential` 也保存该字段，但 `build_expression()` 不使用它。该偏移随 λ 消失时会贡献自由能差，不能随意忽略，也不能按 pair 重复加入；需以每系统一次、λ-dependent constant 的正确形式进入能量或明确从拟合模型移除。

### P1

- [ ] **P1-8：IBS 中 DEXP 的 cutoff/switch 配置失效。** DEXP 配置为 `cutoff_distance=0.70`、`switch_width=0.20`，但 `_create_softcore_force()` 无条件使用 1.2 nm cutoff/1.0 nm switch，未区分 DEXP。生产势、成本和拟合范围不一致；必须传递并验证 DEXP 专用参数。

- [ ] **P1-9：共炼金反离子选择忽略 PBC 且误用全体系质心。** 评分用裸笛卡尔距离，传入 `box_vectors` 未参与；膜体系中蛋白/脂/配体总质心不能代表 bulk water，跨盒水配位也不会计数。另有 `round(lig_net_charge)` 后再判中性的次级 bug，会把异常 `+0.49e` 当作中性；应基于 minimum-image 的最近溶质/水配位距离并对非整数配体电荷 fail closed。

- [ ] **P1-10：全局 TMBAR 最终误差忽略协方差。** 局部 MBAR 已正确要求读取 `dDelta_f[i,j]`，最终却用 `sqrt(var(first_endpoint)+var(last_endpoint))`，单窗口已自相矛盾，多窗口还丢失 offset/端点协方差。最终不确定度和 uncertainty hard gate 必须从完整协方差或等价的端点差估计取得。

- [ ] **P1-11：预平衡 fingerprint 未绑定起始坐标、盒子和请求步数。** 当前仅含 System XML、ligand indices、温度和压力，并刻意排除 equilibration steps；换 docking pose、坐标、periodic box 或从短测试切换到长 production 预平衡都可能复用旧轨迹/checkpoint。已有 `_positions_hash()` 必须接入，并加入 box 与请求步数身份。

- [ ] **P1-12：溶剂盒尺寸把 padding 当成边长增量。** `box_size=max(max_r+1.5,3.5)` 作为完整 `boxSize` 使用；若目标是四周 1.5 nm padding，应约为 `2*(max_r+1.5)`。大配体可产生周期自相互作用或小于 cutoff 的盒；优先改为 `addSolvent(..., padding=1.5 nm)`。

## 附件审查待复核（2026-07-22 新增，未验证）

以下项来自用户提供的审查文本；仅记录为待办，**尚未确认当前源码是否仍存在、未运行任何验证、不得据此标记为已修复或改变生产结论。**

GitHub 跟踪（均为未验证审查发现）：

- P0：ATT-01 [#42](https://github.com/Cedrus810/openmm_IBS_dev/issues/42)、ATT-02 [#43](https://github.com/Cedrus810/openmm_IBS_dev/issues/43)、ATT-03 [#44](https://github.com/Cedrus810/openmm_IBS_dev/issues/44)、ATT-04 [#45](https://github.com/Cedrus810/openmm_IBS_dev/issues/45)、ATT-05 [#46](https://github.com/Cedrus810/openmm_IBS_dev/issues/46)、ATT-06/07 [#47](https://github.com/Cedrus810/openmm_IBS_dev/issues/47)。
- P1：ATT-08 [#48](https://github.com/Cedrus810/openmm_IBS_dev/issues/48)、ATT-09 [#49](https://github.com/Cedrus810/openmm_IBS_dev/issues/49)、ATT-10 [#50](https://github.com/Cedrus810/openmm_IBS_dev/issues/50)、ATT-11 [#51](https://github.com/Cedrus810/openmm_IBS_dev/issues/51)、ATT-12 [#52](https://github.com/Cedrus810/openmm_IBS_dev/issues/52)、ATT-13 [#53](https://github.com/Cedrus810/openmm_IBS_dev/issues/53)、ATT-14 [#54](https://github.com/Cedrus810/openmm_IBS_dev/issues/54)、ATT-15 [#55](https://github.com/Cedrus810/openmm_IBS_dev/issues/55)、ATT-16 [#56](https://github.com/Cedrus810/openmm_IBS_dev/issues/56)、ATT-17 [#57](https://github.com/Cedrus810/openmm_IBS_dev/issues/57)、ATT-18 [#58](https://github.com/Cedrus810/openmm_IBS_dev/issues/58)。
- P2：ATT-19 [#59](https://github.com/Cedrus810/openmm_IBS_dev/issues/59)、ATT-20 [#60](https://github.com/Cedrus810/openmm_IBS_dev/issues/60)、ATT-21 [#61](https://github.com/Cedrus810/openmm_IBS_dev/issues/61)、ATT-22 [#62](https://github.com/Cedrus810/openmm_IBS_dev/issues/62)、ATT-23 [#63](https://github.com/Cedrus810/openmm_IBS_dev/issues/63)、ATT-24 [#64](https://github.com/Cedrus810/openmm_IBS_dev/issues/64)、ATT-25 [#65](https://github.com/Cedrus810/openmm_IBS_dev/issues/65)、ATT-26 [#66](https://github.com/Cedrus810/openmm_IBS_dev/issues/66)；ATT-27 复用 [#35](https://github.com/Cedrus810/openmm_IBS_dev/issues/35)，ATT-28 复用 [#41](https://github.com/Cedrus810/openmm_IBS_dev/issues/41)。

### P0 候选

- [ ] **ATT-01：`scan_boresch_1d_pes` 的孤立 `self` 参数。** 附件称 `abfe_core.py` 顶层函数带无 class 上下文的 `self`，且从未被调用；若调用可能立即失败。决定删除、修正签名或移入 `OrbScanner` 前，先确认当前定义和调用图。

- [ ] **ATT-02：`redistribute_vanishing_lambda_subdomains` 默认常量疑有空格拼写。** 附件称 `VANISHING_TARGET_INTERVALS_PER_EN SEMBLE` 会被解析为两个 token 并导致 `SyntaxError`，使 `abfe_preoptimizer.py` 无法导入；需先核对当前源码是否仍含该文本。

- [ ] **ATT-03：单 GPU REMD 一次常驻全部 replica Context。** 附件称 `REMDManager._build_replicas()` 在单 GPU 为全部 12–24 replica 创建 Context；第 N+1 个 OOM 前的 Context 未释放。需要设计按需/分批 Context、可靠清理或多 GPU 分配，且不得只依赖创建时异常捕获。

- [ ] **ATT-04：并行 stage worker 的 spawn 循环导入/CUDA 顶层副作用。** `_run_stage_worker_process()` 中重新 import `abfe_pipeline`，可能触发 `ibs_engine`/`abfe_core` 顶层 CUDA 初始化。需审查 spawn 安全性，并移除导入期设备初始化或建立明确的子进程初始化边界。

- [ ] **ATT-05：IBS checkpoint/最终分析缺失 bias 或 base 时补零。** 附件称恢复与最终分析把缺文件替换为零数组，且 skip 条件未核验 bias/base 与 energy 的存在、hash、长度；这会把有偏分布当无偏分布。必须 fail closed，禁止补零。

- [ ] **ATT-06：DEXP 拟合模型与生产 Hamiltonian 不同。** 附件称 Orb fitter 的全局 `r0_vdw`、`A_fit`、`B_fit` 等参数未被生产 `DEXPSurrogatePotential.from_dict()` 读取，造成拟合参数静默丢失。需统一拟合/生产模型或拒绝不兼容拟合 JSON。

- [ ] **ATT-07：DEXP `offset_c0` 未进入能量。** 附件称 offset 被拟合、保存但未在 `build_expression()` 使用；若其随 λ 消失会贡献 ΔG。需要以每系统一次的 λ-dependent constant 正确纳入，或从拟合契约中显式移除。

### P1 候选

- [ ] **ATT-08：溶剂腿构建函数保留未使用的 `topology` 参数。** `build_and_cache_solvent_leg()` 的签名/调用方仍传入但实现不用；应删除参数或恢复其明确用途，避免未来静默语义分歧。

- [ ] **ATT-09：热力学循环符号存在四处独立实现。** 附件认为当前公式均正确，但 `main()`、`run_traditional_mode()`、`run_post_analysis()`、`run_full_abfe_loop()` 分散维护。应提取单一、带解析 toy-cycle 测试的绑定自由能组合函数。

- [ ] **ATT-10：`IBSSampler.collect_energies` 的间歇 base 能量失败可能不硬停。** 附件称失败帧记 NaN 后被跳过，连续失败计数被一次成功清零，间歇性丢帧可能逃过阈值。需定义失败率/总丢帧 hard gate，并保留审计记录。

- [ ] **ATT-11：`GeometricRestraintEstimator` 的 0.22 nm 键距离阈值。** 该阈值可能漏掉边界 S-S/配位键并误判近距离非键接触；需以元素/拓扑信息或可配置、经验证的判据替代单一几何阈值。

- [ ] **ATT-12：vanishing 路径/子域范围可能被固定窗口常量覆盖。** 附件称 `VANISHING_FIXED_WINDOW_RANGES` 使 `target_intervals_per_ensemble`、pilot λ 和 metric 参数失效，令“自适应”名存实亡。需先与当前 v18 λ schedule contract 核对，明确这是有意的不可变协议还是实现错误。

- [ ] **ATT-13：TMBAR history 的内存和 checkpoint 体积风险。** 尽管历史上限目前设计为 200 entries，需评估更大态数/buffer 时矩阵内存及 JSON 序列化开销，并设定大小、截断、恢复一致性和可观测性契约。

- [ ] **ATT-14：IBS 中 DEXP cutoff/switch 配置可能被硬编码覆盖。** 附件称 `_create_softcore_force()` 对 DEXP 也无条件写 1.2 nm cutoff/1.0 nm switch，而配置为 0.70/0.20。需验证参数实际传递并禁止配置/生产势不一致。

- [ ] **ATT-15：共炼金反离子选择未正确使用 PBC/bulk-water 判据。** 附件称选择使用裸笛卡尔距离和全体系质心，`box_vectors` 未参与；另有 `round(lig_net_charge)` 先判中性而把 +0.49e 当 0 的风险。应采用 minimum-image 的最近溶质/水配位指标，并对非整数配体电荷 fail closed。

- [ ] **ATT-16：全局 TMBAR 最终误差忽略协方差。** 附件称最终 `sqrt(var(first_endpoint)+var(last_endpoint))` 与局部 `dDelta_f[i,j]` 契约矛盾，多窗口还丢 offset/端点协方差。最终 uncertainty/hard gate 应使用完整协方差或等价端点差估计。

- [ ] **ATT-17：预平衡 fingerprint 未绑定坐标、盒子与请求步数。** 附件称仅包含 System、indices、温度、压力，导致换 docking pose/box/步数仍可复用旧轨迹。应接入 `_positions_hash()` 并纳入 box、预平衡预算与 checkpoint 身份。

- [ ] **ATT-18：溶剂盒尺寸公式可能低估大配体所需 padding。** 附件指出 `boxSize` 是完整边长，推荐直接使用 `addSolvent(..., padding=1.5 nm)` 或正确的 `2*(r_max + padding)`；需保证最小像距离/cutoff 要求。

### P2/发布质量候选

- [ ] **ATT-19：核心物理单元测试覆盖不足。** 附件列出软核势 λ=0/0.5/1、DEXP LJ-matching、IBS log-sum-exp 稳定性、窗口拼接连续性、PBC、离子计数、resume、并行 worker 等缺口；需先盘点现有测试，形成最小覆盖矩阵。

- [ ] **ATT-20：缺少已知 ABFE 基准的端到端集成验证。** 需为公开基准体系建立可复现脚本，覆盖中性/带电配体、两腿循环闭合与实验对比；在此之前不能以发布级精度宣称生产结果。

- [ ] **ATT-21：文档、安装/API/输入格式与独立热力学循环说明不足。** 附件称 README、用户手册、API、输入格式、部署安装及 cycle 推导文档缺失；需先核对现有 `docs/`，再补齐用户面向的最小文档集。

- [ ] **ATT-22：缺少 CI/CD、静态检查与格式化。** 建立 GitHub Actions（或等价 CI）、pytest、ruff/flake8、mypy、black/isort 等分层门槛，并避免把需要 GPU 的作业误放入普通 CI。

- [ ] **ATT-23：运行恢复与资源保护能力不足。** 评估 GPU OOM 的降级/Context 回收、长任务中断后的窗口重跑判定、磁盘空间预检和运行时估计；必须保持科学状态不可变与 fail-closed 原则。

- [ ] **ATT-24：输入验证不足。** 补充 ligand 残基名、TOP include、配体原子数/Boresch 可构建性、最小盒尺寸（至少 2×cutoff）等明确的前置报错和诊断。

- [ ] **ATT-25：协议版本矩阵缺少统一注册/迁移工具。** 当前多处独立 protocol version 需要统一注册表、缓存指纹组合规则、迁移说明和兼容性测试，避免单个版本更新遗漏缓存失效。

- [ ] **ATT-26：`run_all_windows` 过长且职责混杂。** 将约 1200 行流程逐步拆分为恢复/建系/最小化与 Boresch/预热状态机/生产采样/落盘与 checkpoint 等独立方法，并保持行为回归覆盖。

- [ ] **ATT-27：死代码与不可达逻辑清理。** 包括已标记 deprecated 的 overlap autorepair、withdrawn 协议注释、未调用扫描函数和立即 raise 的 `enable_lambda_refine`；与 E-03 协同，先归档再移除可执行死路径。

- [ ] **ATT-28：日志实现分裂。** 多个模块分别覆盖 `print` 或使用标准 `logging`，可能导致导入顺序相关行为；需统一结构化日志入口、级别和文件/控制台策略。

本轮从旧 `todo2.txt` 复核并完成：

- [x] A-01/A-02：传统 REMD LRC 生产者改为逐 λ 的 v2 switching+softcore-aware `r^-6/r^-12` 系数，并使用 worker 实际读取的 `lj_tail_lrc_coeff_kj_mol` 键。
- [x] A-03：PME context 查询失败后改用 cutoff/tolerance 闭式派生 alpha，不再读取自动 PME 下通常为零的静态参数。
- [x] A-04：`tmbar_history` 上限设为 200 个 minibatch，checkpoint 保存丢弃计数，resume 只恢复最新有界后缀。
- [x] A-09：`ensure_owned_system` 的早退路径先验证底层 OpenMM 对象仍可访问。
- [x] A-11：base 能量第一次失败时立即检查坐标和力；发现非有限值则停止 MD。
- [x] A-18：JSON checkpoint 在原子替换前执行 flush/fsync；POSIX 额外同步父目录。
- [x] λ-18：撤销 equal-|ΔF| 排点。Stage 2 由 Fisher 探针生成 17 个常规节点，再在人类指定的前两个区间插入 4 点，得到 `λ0..λ20`；五个闭区间窗口固定为 `[0,5] [5,9] [9,13] [13,17] [17,20]`，总采样槽位 25。
- [x] SOLV-ION：溶剂腿从隐式纯水/仅中和改为显式 0.15 M NaCl，并保留必要中和离子；新增 v2 manifest，旧的 `0 Na / 0 Cl` 缓存自动失效重建。

上述修改仍需按 [status/VALIDATION_MATRIX.md](status/VALIDATION_MATRIX.md) 的环境门槛补齐完整依赖/GPU 证据。

## 当前运行验证

- [ ] V-01：用全新 GPU 进程 resume vanishing 窗口 0，确认进入冻结验证/production，而不是再次 `IBSWarmupConvergenceError`。复现和验收细节见 [handoffs/VANISHING_WINDOW0_HANDOFF.md](handoffs/VANISHING_WINDOW0_HANDOFF.md) 与验证矩阵 `VAL-GPU-007`。
- [ ] V-02：运行传统 `single_lambda`/REMD 的小型固定盒回归，确认每个 task 收到有限、长度等于态数的 v2 LRC 数组，且 worker 的每帧修正等于 `coeff[k]/V(t)`；GitHub [#32](https://github.com/Cedrus810/openmm_IBS_dev/issues/32)。
- [ ] V-03：在 OpenMM/CUDA 环境重新生成 Stage 2 v18 pilot，确认输出恰为 21 个唯一 λ、`λ20=0`、窗口槽位 25，并完成至少一个 vanishing 窗口的真实启动验证；见验证矩阵 `VAL-TEST-004`，GitHub [#33](https://github.com/Cedrus810/openmm_IBS_dev/issues/33)。
- [ ] V-04：重建溶剂腿缓存并核对 `solvent_cache_manifest.json` 与 `topology_solvent.cif`：Na/Cl 均非零、目标浓度 0.15 M，随后确认旧溶剂腿 checkpoint 因 System 指纹变化被拒绝；GitHub [#34](https://github.com/Cedrus810/openmm_IBS_dev/issues/34)。

## P2 工程工作

- [ ] E-01：production ESS 自动修复的第二调用点迁移到 per-state trajectory bank；对应 GitHub `openmm_IBS_dev#1`。
- [ ] E-02：为 pilot 热力学长度探测设计安全的长探针重测机制；对应 GitHub `openmm_IBS_dev#29`。不得复活已废弃的 fixed-H adjacent-overlap 自动变异环。
- [ ] E-03：把 `_run_stage_with_overlap_autorepair` 中早退之后约 900 行不可达旧变异逻辑移出生产类；保留历史可读性时放入归档文档，不得保留可被误激活的可执行代码；GitHub [#35](https://github.com/Cedrus810/openmm_IBS_dev/issues/35)。
- [ ] E-04：为膜/各向异性盒实现并验证适用的有限尺寸静电修正；当前 `apbs_correction.py` 对纵横比大于 1.10 的盒子应继续 fail closed；GitHub [#36](https://github.com/Cedrus810/openmm_IBS_dev/issues/36)。
- [ ] **P2-13：LSE 容差 0.5 的新默认值未在普通 CLI 路径生效。** engine 默认已为 `lse_log_residual_tolerance=0.5`，但 pipeline 和 complex-leg CLI 仍显式传 0.25，导致 v25 的放宽从未成为实际默认。需统一上游默认、配置落盘和 CLI 帮助。
- [ ] **P2-14：DEXP 最终结果谎报 LJ LRC 已应用。** IBS 构建 DEXP 时明确不附加 LRC，但 `final_results.json` 无条件声称 `lj_long_range_dispersion_correction.applied=true`。必须把 DEXP 标为未应用/不适用，或仅在实际附加修正且有证据时记录为 applied。

## P2/P3 科学与稳健性评估

- [ ] R-01：量化 Boresch `k(1-cos(delta))` 势相对谐波解析释放公式的非谐性误差，并确定 hard gate；在有理论/数值依据前不直接改生产公式；GitHub [#37](https://github.com/Cedrus810/openmm_IBS_dev/issues/37)。
- [ ] R-02：用真实 bridge 数据重新标定 Shadow-Coulomb 的窗口数和 overlap 阈值；GitHub [#38](https://github.com/Cedrus810/openmm_IBS_dev/issues/38)。
- [ ] R-03：评估 `refine_stage_lambda_path_by_overlap` 的点 ESS 插点启发式；它目前不在 `non_mutating_v1` 生产路径上；GitHub [#39](https://github.com/Cedrus810/openmm_IBS_dev/issues/39)。
- [ ] R-04：评估 ACE softcore 分母 `1e-6` floor、pilot metric 最小样本数、PBC 重居中对 Boresch 几何的影响；GitHub [#40](https://github.com/Cedrus810/openmm_IBS_dev/issues/40)。
- [ ] R-05：统一 warning/stdout 日志，并评估 DEXP 多随机种子优化；均不改变当前数值定义；GitHub [#41](https://github.com/Cedrus810/openmm_IBS_dev/issues/41)。

## 复核关闭（不再作为待办）

- [x] A-05：`gauss_coul` 在当前 `dexp_experiment.py` 中按 fit mode 参与 `delta_gauss_replacement`，并非“构建后从未使用”；旧报告已过时。
- [x] A-07：裁剪后的 Boresch 力常数同时用于实际施力和解析修正；原始轨迹只是估计器输入，旧报告所称“施力与修正使用不同 k”不成立。裁剪仍会留下显式诊断。
- [x] A-12：各向异性 APBS 目前是明确不支持并 fail closed，不是静默错误；功能扩展保留为 E-04。
- [x] A-14/A-17/A-19：属于开发期缓存语义、数值敏感性或日志体验，不是已确认的生产数值缺陷；归入 R-04/R-05 或不再单列。
