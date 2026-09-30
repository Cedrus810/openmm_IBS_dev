# 当前源码逻辑与物理审查（2026-09-26）

> **一次性审核的原始记录，不是待办清单。已归档（2026-09-28）**——条目**尚未对照代码核过**，
> 待核事项登记在 [TODO.md](../TODO.md) `REVIEW-0926 [P2]`。审查是在 Windows 副本上做的（原文 `D:\` 路径，链接已改成仓库相对路径），
> 行号是 HEAD `e576a54` 当时的。「结果质量与采样效率改进补充（2026-09-27）」一节是改进建议、不是缺陷，留在这里作参考。

审查输入：runtime 已加载的 D:\ABFE_IBS\ABFE_IBS。仅只读审查当前源码、文档与测试；没有修改项目，没有运行 pytest、构建或 GPU 采样。PowerShell 初次启动被 Windows split writable roots 沙箱错误阻断，后续通过原生 Windows 路径的只读文件接口阅读。以下是源码与调用链证据，不是现有 benchmark 数值偏差的测量。

结论：仍有缓存、续跑和新 Stage-2 LCR 统计/步数问题。本轮未确认默认中性、非共价、可溶 dual_lambda ABFE 主线有新的可达物理公式错误。RBFE 辅助路径有可证实的物理缺口。

## 1. [P1] LCR 账本没有绑定当前阶段协议，旧段可能混入新协议

范围：显式 stage2_lcr_rounds=true，默认关闭。

位置：[账本加载](../../abfe_pipeline.py#L14694)、[跳过旧段](../../abfe_pipeline.py#L14828)、[复用旧证据](../../abfe_pipeline.py#L14868)、[合并加载](../../abfe_pipeline.py#L15458)。

账本只比较 protocol_version 与 lambdas_var/window_ranges。保持等分 λ 和分窗不变，修改温度、softcore/Boresch 等阶段身份后，上游 stage 结果会失配，但账本仍通过。SAMPLED/FAILED 直接跳过；已经落账的 evidence 不重算。最终 merged loader 没有传 expected_window_protocol 或当前盒体积。旧段可被全部复用，或者与新协议段混合。比对冻结 f_k 只证明产物与旧账本自洽，不能证明它们属于本次协议。

建议：账本绑定既有阶段语义协议键和已提交生产盒身份，变化时拒绝复用；合并加载执行同一套段级协议/盒校验。不需要重新引入整个源码或自产 System XML 文件的哈希。

## 2. [P1] 已完成 final/stage 缓存可以绕过当前生产盒校验

范围：普通 resume；尤其没有 Boresch 的溶剂腿。

位置：[stage 协议键](../../abfe_pipeline.py#L15655)、[顶层协议键](../../abfe_pipeline.py#L16872)、[final 提前返回](../../abfe_pipeline.py#L17102)、[现有窗口盒门](../../ibs_engine.py#L7862)。

stage/top-level 键没有实际已提交生产盒体积；final 协议键相同且 sanity 校验通过后直接 return。现有严格盒门在进入窗口采样/加载之后才运行，无法保护提前返回。具体触发：同一个GRO路径仅修改盒向量，主System manifest发现SHA变化并重建native cache，但不清理pipeline_state/stage/final。equilibrium_is_done对预平衡fingerprint不匹配只告警，仍按旧completed状态放行（runabfe.py:1182–1225）。无Boresch路径不会调用fail-closed的预平衡恢复入口；adopt_pre_equilibrated_state因新盒不匹配返回False后，run_full_pipeline在16680–16684使用新输入盒继续。旧sampling仍为completed，随后17102–17144提前返回旧final。其它配置、拓扑、λ身份保持不变时，旧密度结果由这条路径被返回。

建议：completed 聚合缓存也校验已经提交的生产盒语义身份，或在提前返回前执行与窗口一致的体积/provenance 校验。不要恢复此前被明确否决的整份自产文件哈希方案。

## 3. [P2] analyze-only 未接入盒体积校验

位置：[分析协议参数](../../runabfe.py#L5353)、[加载窗口](../../runabfe.py#L5376)、[None 自动放行](../../ibs_engine.py#L7868)。

分析回退路径构造 expected_window_protocol 时没有 current_box_volume_nm3。底层 gate 在当前体积 None 时把 box_volume_match 设为真。于是其它协议仍匹配、但实际采样盒与已提交预平衡生产盒不同的旧窗口不会被这道门识别，包括 BOX-01 修复前在输入盒上采样的历史产物。

这不意味着“对旧盒样本做数学重分析本身错误”。独立触发是输出目录被重用/重建后，旧窗口仍在V_A、新analysis context对应V_B，并且正式final缺失/未完成而进入raw-window fallback。只改final-gate等result-only参数不会让窗口sampling identity失配（ibs_engine.py:451–480），但分析路径没有补上当前盒门，仍可把V_A窗口归入当前结果。建议从持久化的已提交预平衡/生产盒 provenance 恢复期望体积；无法确定时明确标为无法验证或拒绝作为当前生产结果。

## 4. [P2] LCR 跳过边缘窗口后，jackknife σ 配给错误窗口

位置：[真实窗口号](../../ibs_engine.py#L23970)、[错误建表](../../abfe_pipeline.py#L14995)。

covariance_chain_segments 的 window_index 是成功 local_results 的位置，source_window_index 才是真实窗口号。finalize 按前者建表、按物理窗口号 w 查询。

可达例：23 态布局 [(0,5),(4,8),(7,12),(11,16),(15,20),(19,23)]，窗口0去相关后帧数不足被跳过，窗口1–5仍能彼此拼接。求解返回 ANALYSIS_INCOMPLETE 而不是 error，finalize 继续运行。chain 局部编号0..4对应真实窗口1..5，σ整体错位，最后窗口漏记。结果的 sigma_window 和 total_error jackknife 增量失真。此例本身是路径不完整结果，不应声称它已成为可信完整 ΔG。

建议：以 source_window_index 建表，旧产物才回退 window_index，并保留路径不完整状态。

## 5. [P2] LCR 决策中的非有限 MBAR σ 可能让补采条件失效

位置：[底层误差无有限性校验](../../multi_segment_analysis.py#L183)、[转成标量](../../stage2_rounds_evidence.py#L65)、[补采阈值](../../stage2_rounds.py#L373)。

collapse_segments 只检查输入、f_k 和 log_D 有限，没有检查 dDelta_f。sigma_mbar 为 NaN/Inf 时，max(2*sigma_mbar, sigma_tol) 会是 NaN/Inf；在 ESS、独立段数、top-1% 等其它条件通过时，即使 jackknife σ很大，也不会生成该补段计划。classify_window 仍能标 provisional，因此更准确的后果是缺少预期修复动作/过早停止，不是已经证明会写出可信的非有限误差最终结果。

建议：把非有限 MBAR σ 显式视为证据失败，或使用有限 jackknife σ 与 sigma_tol 的保守规则；不得让 NaN 进入比较阈值。

## 6. [P2] 固定 f_k burn-in 向上取整，超过配置步数

位置：[块数与整块执行](../../ibs_engine.py#L16972)。

执行 ceil(total/block) 个完整块，没有截断最后一块；max(1,...) 还使非正步数也执行一块。5000步分块时，7000→10000，1000→5000，0→5000。默认25000恰好整除，不触发。影响额外 MD 消耗与 steps_at_full_bias/burn-in 诊断，不据此推断 ΔG 偏差。

建议：最后一块执行 min(block,total-consumed)，明确拒绝负值，0按约定跳过。

## RBFE 辅助路线（不属于当前 ABFE 生产主线）

### A. 保留所有 dummy 成键项不能保证跨腿抵消

位置：[全部保留 dummy 键角](../../rbfe_core.py#L2471)、[保留二面角](../../rbfe_core.py#L2514)、[减力组的端点检验](../../rbfe_core.py#L3227)。

代码2350–2354把全强度保留推断为构型积分可分离。多个 dummy-core 角/二面角可让 dummy 的积分依赖核心构型，而核心在complex与solvent中的分布不同。减去dummy力组后点态能量等价只是力分组/端点接线验证，不能证明配分函数分解。此问题的条件与反例已有原论文：[Dummy Atoms in Alchemical Free Energy Calculations](https://pmc.ncbi.nlm.nih.gov/articles/PMC8280730/)。

主线程做了一个纯数值反例（不运行项目代码、不写工作区）：一个固定键长的dummy通过两个109.5°谐振角约束核心，k=200 kJ/mol/rad²、T=300K，球面中点积分180×360格。核心夹角90°/120°时dummy角向自由能为4.59547/3.64598 kJ/mol。积分依赖核心角，已足以反驳一般可分离性；0.94949 kJ/mol仅是该反例的差值，不是当前真实体系的ΔΔG偏差。

建议：明确限制并验证可分离的dummy连接形式，或者建立额外成键耦合项的切换/修正账目；端点测试不能代替跨环境closure test。

### B. 核心硬约束缺失检查不对称

位置：[仅合并A约束](../../rbfe_core.py#L2574)。只检查两边共同约束距离，以及B独有的非B-only约束；A独有core-core约束被无条件保留。支持的A/B输入若核心键和参数相同、A有硬约束而B没有，lambda=1仍在额外约束下采样。纯能量/力比较看不出系综约束差异。建议对核心约束集合做双向一致性检查。

### C. NoCutoff 输入的炼金 LJ 被改为截断

位置：[方法转换](../../rbfe_core.py#L2774)。PME/Ewald/CutoffPeriodic以外全部映射为CutoffNonPeriodic，包含未被拒绝的NoCutoff输入。环境native力保持NoCutoff，炼金LJ却截断，不能恢复原端点（某些平台还可能直接拒绝不一致方法）。建议保留NoCutoff或明确拒绝该输入。[OpenMM官方方法定义](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomNonbondedForce.html)。

## 已核实不应继续报成当前主线 bug 的候选

- Stage0 attachment缓存门已修：abfe_pipeline.py:17244–17264使用本次运行的attachment_enforce_convergence_gates。
- Beutler Coulomb的无条件floor：默认/传统full pipeline vanishing都固定lambda_coul=0，当前主线不触发；仅人为固定lambda_coul=1的直接辅助API边界。
- LRC排除对：标准非共价GROMACS配体没有ligand-environment exception，不能据此宣布主线错误；共价/手工System另当别论。
- 多NonbondedForce：主GROMACS构建入口有单NB断言，不能把引擎通用API的first-only处理冒充默认production bug。
- 已知科学边界（独立重复、时间相关不确定度、PME周期残差/LRC近似等）与新的可达实现错误应分开；本次静态审查没有关闭这些验证事项，也没有证明当前benchmark偏差的归因。


## 主线性能审查补充（2026-09-26）

范围：`runabfe → ABFEPipeline → Stage1 REMD / Stage2 IBS → 离线分析` 的当前默认调用链。审查输入仍为 runtime 已加载的原生 Windows 工作区，源码只读。本补充没有修改项目或运行 GPU benchmark。

计数基准来自 [abfe_config.json](../../abfe_config.json#L18)：CUDA，production，n_steps_per_window=250000，steps_per_update=500，enable_early_stop=false，outer_lambda_local_residual_ibs=false。一个新生产块正常有500次update；普通Stage2窗口为4–5态，首尾窗口可能不同。以下次数按无丢帧、无回退、无续采历史的生产块计算；不能把全部16个目标态当成每个窗口的K。

结论：主线仍有额外Context构建、同帧能量/CV重复求值、诊断坐标下载以及同步累计历史重写。它们分别可以解释窗口边界停顿和生产过程的周期性GPU等待，但没有主线实测计时，尚不能确定耗时排名或提速比例。GPU利用率下降不等于GPU设备或CUDA Context真的重启。

### PERF-01 [P2] 每个窗口提前创建默认生产路径用不到的 probe Context

位置：[构造时立即创建](../../ibs_engine.py#L9029)、[额外Context](../../ibs_engine.py#L9085)、[主线sampler入口](../../ibs_engine.py#L16020)、[live CV读取](../../ibs_engine.py#L9128)。

IBSSampler.__init__只要wrapper含_int_cv_force_xmls就调用_build_probe_context。该函数复制全部粒子信息、逐λ反序列化力，并在与主Context相同的平台/设备上新建Context。默认non_mutating_v1的warmup/production使用主Context里的live CV；正常采样不调用该sampler的evaluate_interaction_energies，所以这个额外probe闲置。默认非变异路径还明确跳过旧的fixed-H修复分支：[ibs_engine.py:18132](../../ibs_engine.py#L18132)。

影响：每个进入实际采样/续采的窗口增加一次Context初始化，并在窗口内占用GPU/宿主资源；频率是每窗口一次，不是每MD步一次。窗口正常结束已有显式清理，不能描述成无限泄漏。

建议：主生产sampler关闭eager probe，或首次任意坐标求值时再创建。保留fixed-H与历史帧evaluator所需的probe。不能仅删掉构造调用：[evaluate_interaction_energies](../../ibs_engine.py#L9091)在probe缺失时当前返回全零，必须同时明确按需初始化/报错契约，避免把性能修复变成物理错误。

### PERF-02 [P2] 强 guard 与能量账本重复查询同一状态

位置：[检查周期](../../ibs_engine.py#L19563)、[强guard全状态查询](../../ibs_engine.py#L19620)、[base查询](../../ibs_engine.py#L9327)、[bias查询](../../ibs_engine.py#L9393)。

默认每10个update完整查一次力，最后一个update也完整检查。500个update中有51个强guard。强分支查询总能量、全系统力和坐标，但没有填写guard_e_base/guard_e_bias，随后collect_energies仍分别查询base与bias。同一帧因此有“全状态 + base + bias”三次带能量的getState。弱分支已经复用base/bias，不应把已修复的弱分支重复查询再报一次。

建议：重组强guard的查询与结果传递，使灾难检查和账本共用分组能量，同时保留力/有限性检查及Group0–5的分组契约。不能简单删除安全检查。若用总能量减去某分组推导另一分组，须检查浮点抵消与账本精度；结构上的重复调用不能直接等同于可按相同比例减少墙钟时间。

### PERF-03 [P2] bias已评估全部CV，账本读取仍逐K再次求值

位置：[bias账本查询](../../ibs_engine.py#L9393)、[逐态CV入口](../../ibs_engine.py#L9414)、[getCollectiveVariableValues调用](../../ibs_engine.py#L9128)。

Group1的log-sum-exp偏置需要所有cv_k_int。查询bias会计算这些CV，随后live CV账本再次调用getCollectiveVariableValues。OpenMM官方实现先copyState，然后对每个inner CV分别执行getState(State::Energy,false,1<<i)；“一次Python调用返回全部值”不代表“一次GPU查询返回缓存向量”。见[CustomCVForceImpl.cpp](https://raw.githubusercontent.com/openmm/openmm/master/openmmapi/src/CustomCVForceImpl.cpp)和[CommonKernels.cpp](https://raw.githubusercontent.com/openmm/openmm/master/platforms/common/src/CommonKernels.cpp)。

普通K=4–5、500帧生产块，会额外执行约2000–2500次inner energy query；强guard还会额外评估一次全系统偏置。逐态u_kn本身是统计必需的，不能据此删掉λ态。也不能把当前未使用的独立probe Context当成这些live查询的执行者。

建议：让偏置与逐态账本共享同一份CV求值结果。此项涉及OpenMM求值接口/力实现，改动成本高于PERF-01；必须校验CV、bias、base以及重加权账本的一致性，不能只靠减少Python调用数判断正确。

### PERF-04 [P3] 诊断坐标频率与强guard错开，额外下载全系统坐标

位置：[默认每5帧记录](../../ibs_engine.py#L1128)、[缺少复用坐标时下载](../../ibs_engine.py#L1148)、[只保存非水原子](../../ibs_engine.py#L19342)。

坐标旁车只作诊断，不参与采样判据、缓存身份或物理能量账本。默认每5个能量帧记录一次，共100帧；强guard每10次update带回坐标，两者正常重合50次。因此其余约50次要额外getState(getPositions=True)，取回全系统坐标后才筛选非水原子。最终只存蛋白/配体/离子，并不意味着设备只传输了这些原子。

建议：对齐诊断坐标和强guard频率、优先复用已有坐标；无坐标诊断需求时可显式设ABFE_STAGE2_COORDS_EVERY=0。调整诊断频率不应改变Hamiltonian，但会降低轨迹观察分辨率。主线已经使用asNumpy=True，没有w34独立脚本中的大规模Vec3 Python对象转换。

### PERF-05 [P2] 主线程同步重写累计历史，形成周期性停顿

位置：[同步快照](../../ibs_engine.py#L14990)、[每100个update触发](../../ibs_engine.py#L19697)、[坐标全历史np.stack/np.savez](../../ibs_engine.py#L1164)、[最终再次保存](../../ibs_engine.py#L20023)。

_enqueue_window_snapshot虽然名字含enqueue，实际同步复制和保存全部累计能量历史；坐标旁车也stack全部历史并覆盖写出。之后同一线程保存OpenMM checkpoint、manifest、元数据与convergence JSON。JSON包含flush/fsync及原子replace，POSIX还同步父目录：[ibs_engine.py:1275](../../ibs_engine.py#L1275)。

新窗口500帧正常按100、200、300、400、500及最终500累计写出，共2000个“帧等价”，即最终账本有效数据量的4倍；坐标旁车同理为20+40+60+80+100+100=400帧对100帧。这里的4倍指重复序列化的数据量，不是耗时或总文件大小的精确倍数。带续采历史时重写量还会增大。每50000个MD步同步保存一次，是生产过程中周期性GPU空闲的明确候选。

建议：按块持久化、减少未变化的最终重复保存，或采用一致快照的后台写入。必须保留checkpoint恢复能力，以及checkpoint、帧数和能量账本之间的一致提交关系。不能用取消持久化保障换取表面GPU占用率。原代码已经删除同一次周期保存中对相同文件立即写第二遍的旧问题；本项指仍存在的累计历史重写。

### Stage1 / 离线分析：正常CPU阶段与资源浪费应分开

位置：[REMD到CPU分析衔接](../../abfe_pipeline.py#L6548)、[compute_u_kn使用CPU](../../abfe_pipeline.py#L6585)、[REMD结束时仅清理reporters](../../ibs_engine.py#L26336)。

1. Stage1之后compute_u_kn和MBAR在CPU上运行，GPU空闲是正常阶段切换，不能认定为CUDA反复重启。但REMD的replica Context在CPU分析期间仍存活；默认12个状态且未限制驻留数时，占用相应GPU/宿主资源。建议在确认后续只依赖DCD和协议元数据后，于离线分析前显式释放replica Context。
2. compute_u_kn一次读取各replica轨迹，再拼接完整轨迹：[ibs_engine.py:27188](../../ibs_engine.py#L27188)。全轨迹与replica Context、CPU evaluator同时存在，增加内存峰值。可以流式读取；目前没有主线实测内存数据，不能声称已经发生OOM或交换内存。
3. 串行分支按chunk反复创建CPU evaluator Context：[ibs_engine.py:27552](../../ibs_engine.py#L27552)、[创建位置](../../ibs_engine.py#L3171)、[销毁位置](../../ibs_engine.py#L3263)。仅n_workers=1、资源限制降为1或Pool失败回退时触发；正常多worker路径复用各worker Context。不能报成当前配置必然发生的默认浪费。

当前resume=false会按新运行重新采样和分析；这是配置语义，不能独立算作默认重复计算bug。本轮未发现标准一次调用内重复执行相同MBAR solve的证据。

### 已排除的性能候选与测量边界

- w34独立脚本里未使用的额外CV与Vec3对象转换，不应直接套用到主线。主线的零rest CV默认省略；IBS混合偏置中的K个目标CV是必需的；坐标已用asNumpy=True。
- PYMBAR_DISABLE_JAX已在abfe_core设置，不能将旧JAX GPU预分配问题报成当前尚未修复的事实。
- 默认warmup与production复用同一主Context；正常结束已有清理，未确认每个update创建/销毁主Context。
- GPU利用率锯齿可能来自GPU/CPU同步、磁盘保存、CPU离线分析和窗口初始化。只有利用率观察不能证明设备启停，也不能确定哪个环节占主导。
- 当前工作区主线output为空，本轮没有主线CUDA profiler、GPU遥测或wall-clock实测，所有调用次数都是源码结构计数。

已有生产计时入口：[loop_timing_s](../../ibs_engine.py#L20171)，包含integration_s、guard_s、cv_probe_s和ledger_io_s。后续应先据此比较实际占时，再做同一体系、同一协议的优化前后对照。该计时不覆盖Context构建、全部warmup、离线分析，以及计时块外的最终保存；cv_probe_s也包括坐标旁车采集，不等于独立probe Context耗时。

建议实施顺序：先处理PERF-01和PERF-04，再优化PERF-05并保留续采一致性，然后处理PERF-02；PERF-03需要更深入的CV结果复用实现与物理账本验证。这里是改动成本和可验证性的顺序，尚不是实测收益排名。


## 结果质量与采样效率改进补充（2026-09-27）

本节对应“还有什么可以优化、可以计算更优结果”的进一步审查。源码已按 runtime 当前版本重新读取：HEAD `e576a5400afa1966c7a3aee4145ef4e3e75b0e3c`。本节的证据与2026-09-26各节分开；没有在此逐项重新判定旧问题是否已修复。沿用原报告文件名，保留前文历史审查内容。

范围：当前production ABFE主线的误差估计、独立重复、冻结偏置后的平衡、λ路径、预算和数值精度。以下以改进机会或代码明确承认的科学限制为主，不能全部归为新发现的实现bug，也不能据此证明当前benchmark的ΔG已经偏离真实值。

计数与时长：当前配置每生产块250000步、每500步采能量；IBS生产积分器为2fs：[abfe_config.json](../../abfe_config.json#L18)、[积分器](../../ibs_engine.py#L15718)。因此每块是0.5ns。不同窗口的混合采样不能按总时长直接等同于每个固定λ态独立采样0.5ns，也不能与w34每态3ns直接比较。

### QUALITY-01 优先修正Stage1 adjacent-BAR的不确定度估计

状态：当前默认可达的已知科学限制，优先级高；不等于已证明BAR点估计错误。

证据：[主估计器策略](../../abfe_pipeline.py#L1537)、[明确的低估标记](../../ibs_engine.py#L27985)、[TI一致性门](../../abfe_pipeline.py#L9236)。

Stage1 decharging采用adjacent_bar主值，每态使用全部帧。代码将sigma_policy设为bar_asymptotic_edge_variance_sum，并明确写出sigma_suspect_underestimated=True：当前误差和没有修正时间自相关。FD-TI一致性门检查两个点估计的差异，不能替代时间相关误差分析。

改进：保留时间/replica索引，按连续时间块重采样，每个重采样样本重算整条BAR链。对交换耦合的replica应保留同步时间块结构，对共享中间态的相邻边也应保留其协方差；不要把相关帧当成IID逐帧bootstrap，也不要只独立重采样各条边再机械相加方差。

验证：比较不同block长度的经验不确定度、现有BAR渐近σ和跨独立重复离散度；检查block长度超过主要相关时间后结果是否稳定。该改进可能改变误差条而不改变中心ΔG，目标是让报告不确定度更可信。MBAR及相关自由能误差工具对相关样本的限制见[PyMBAR官方文档](https://pymbar.readthedocs.io/en/master/mbar.html)。

### QUALITY-02 实际运行并汇总独立重复，保留UNMEASURED的语义

状态：已有seed和统计工具，但当前IBS主线不会自动完成独立重复精度测量。

证据：[单次精度未测声明](../../abfe_pipeline.py#L8555)、[最终产物声明](../../runabfe.py#L9151)、[seed入口](../../runabfe.py#L4878)、[随机流管理](../../ibs_engine.py#L156)、[cross_repeat_precision](../../abfe_core.py#L2694)。

当前单次run的precision_status是UNMEASURED，既不是精度达标，也不是精度失败。MBAR/分块误差只描述已访问样本，无法识别重要构象一次都未被访问的情况。

改进：至少3个同物理协议、不同显式seed的完整binding重复；对存在慢构象的体系，应从相隔足够远的平衡构象或明确的不同构象起点启动。不同输出目录、同一checkpoint的续采或同一条轨迹的三个切片不自动构成三个独立重复。固定输入应指力场、温度、端点、修正和分析语义一致，不要求起始坐标完全相同。

验证：汇总每次完整两腿binding ΔG，报告跨重复样本SD及样本数，并与单次误差并列展示。不要把样本SD与同一误差来源的MBAR σ简单相加，也不要把重复一致当作力场/体系准备的系统误差已经消失。实践依据见[炼金自由能最佳实践指南](https://pmc.ncbi.nlm.nih.gov/articles/PMC8388617/)。

### QUALITY-03 冻结偏置后增加慢构象平衡与burn-in敏感性验证

状态：当前固定策略的科学改进机会；没有证明所有体系的现有10ps都不足。

证据：[固定5000步burn-in](../../ibs_engine.py#L16545)、[丢弃与验证入口](../../ibs_engine.py#L17116)、[冻结验证门](../../ibs_engine.py#L8163)、[split-half启用](../../abfe_pipeline.py#L7322)。

标准路径冻结f_k后固定丢弃5000步，即10ps，再进入基于能量的local-MBAR验证。这能检查Δf−ΔF，但没有证明配体二面角、口袋侧链或溶剂空腔等慢自由度达到平稳。最终已有split-half诊断和σ下界，但默认split_half_max_z=None时不构成硬阻断；样本不足也可能无法给出下界。不能把现有诊断描述成完全缺失。

改进：选少量与自由能相关的结构量，联合采样能量，做时间block分布/均值稳定性检查。对代表性窗口比较5k、25k、100k步冻结后丢弃长度及最终ΔG；条件成熟后使用固定下限加若干估计相关时间的自适应burn-in。新长度不是可跨体系直接照搬的普适阈值，短轨迹上相关时间本身也可能估不准。

验证：检查不同burn-in和不同构象起点是否给出相容ΔG，且前后时间块的结构分布不持续漂移。若需要口袋水占据量，应另行记录该诊断：当前默认坐标旁车只保留非水原子，不能直接恢复水占据。

跨腿构象诊断可帮助定位慢采样，但复合物和溶剂的不同物理系综本就可以有不同构象分布，不能一律把“跨腿分布不相同”当成错误，也不建议无条件开启strict_cross_leg_conformer。当前WARN策略与裁决注释见[abfe_core.py:9082](../../abfe_core.py#L9082)。

### QUALITY-04 用可靠pilot和边际信息收益优化λ路径与生产预算

状态：已有自适应机制的进一步改进；不能通过重启旧探针或直接增加态数代替。

Stage1证据：[强制回退线性路径](../../abfe_pipeline.py#L5993)、[旧预优化拒绝入口](../../abfe_preoptimizer.py#L12570)。当前默认12态线性去电荷，原因是旧cutoff探针不保真生产PME。可实现与生产PME Hamiltonian一致的beta² Var[dU/dlambda] pilot，再按可靠的度规调整λ密度；不能恢复旧Var(U_group1)方法。

Stage2证据：[单点度规估计](../../abfe_preoptimizer.py#L12555)、[窗口目标](../../abfe_preoptimizer.py#L1703)。当前9个pilot点到16个最终态，窗口通常4–5态，首尾另有限制。单点metric_g来自一次样本方差，没有度规不确定度控制。state_count主要控制最大K，不能直接当成已最优化全路径统计效率。

改进：对最影响路径的pilot点做block/repeated-pilot稳定性评估；仅在度规或布局不稳定时追加探针预算。用多体系实测比较state_count和metric_integral目标，不无证据提高总态数，也不把所有16个目标态当成单窗口K。

预算证据：[当前总预算配置](../../abfe_config.json#L28)、[未知总预算语义](../../abfe_preoptimizer.py#L3263)、[弱窗排序的边界](../../abfe_preoptimizer.py#L8535)。当前全局生产预算为null，每窗最多4块；已有弱窗排序，但不是按每GPU块的信息收益进行全局预算分配。

改进：在现有缺窗优先、上游重锚和可调度顺序约束内，对同层可补采窗口按有效样本增长和不确定度下降分配额外预算，同时设置明确stage总预算。边际收益估计也需回放校准，不能用单个噪声很大的σ变化决定全部调度。

现有enable_early_stop=false是合适的保守默认：阈值尚未用完整轨迹回放校准，见[ibs_engine.py:6957](../../ibs_engine.py#L6957)。校准通过后才考虑启用，并保留既有五项质量门和连续通过要求。enable_lambda_refine也不应直接打开来恢复legacy精修路径。

### QUALITY-05 通过数值精度对照验证约束容差与积分步长

状态：低成本、可验证的精度研究机会；不是已证实的当前ΔG偏差。

证据：[IBS生产积分器及约束容差](../../ibs_engine.py#L15718)。生产步长固定2fs，约束容差显式设为1e-3。[OpenMM官方LangevinMiddleIntegrator源码](https://raw.githubusercontent.com/openmm/openmm/master/openmmapi/src/LangevinMiddleIntegrator.cpp)默认设为1e-5；[API说明](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.LangevinMiddleIntegrator.html)中该容差是受约束距离的相对比例。

改进：将实际积分参数显式化并纳入可追踪的协议身份，在代表性体系做1e-3与1e-5容差、必要时2fs与1fs步长的对照。当前硬编码参数不能靠添加一个未接入积分器的config键改变。

验证：以相同物理时长而非相同步数比较约束残差、结构分布、相关时间、ΔG与墙钟成本，并使用足够样本/独立重复区分统计波动与数值偏差。更紧容差不自动保证更准确的binding ΔG；应先量化是否有可分辨收益。不要在此基础上未经验证就增大到4fs。

### QUALITY-06 使Boresch释放主值与实际cosine势更加一致

状态：条件性模型近似的改进机会；当前已报告模型误差，不能再次报成“零误差修正”。

证据：[实际角度/二面角势](../../abfe_core.py#L9316)、[解析公式](../../abfe_core.py#L9188)、[已计算cosine参考并返回模型误差](../../abfe_pipeline.py#L8035)。

实际限制势使用K(1−cosΔφ)，释放主值仍使用局部Gaussian公式。当前已运行六坐标cosine/Jacobian积分，记录cosine_release_reference_kJ_mol和两者差的analytical_release_model_error_kJ_mol，并通过error字段报告；这项已有保护必须保留。

改进：先检查实际产物中的两种释放值差异。在低力常数、拟合力常数或harmonicity不佳的条件下，若差异不可忽略，验证采用与实际势一致的cosine积分作为释放主值，或采用sampled release。高力常数下差异可能很小，不应一律宣称现有修正有显著误差。

验证：在参数扫描和可解析/数值参考体系上验证积分符号、单位、标准态及修正进入热力学循环的次数。切换主值后应明确处理原近似误差项，避免重复加入同一差异，也不能把确定性模型差简单解释为随机标准差。

### QUALITY-07 优化验收加入实际时间与有效信息产出

证据：[当前A/B代价字段](../../stage2_ab_report.py#L362)、[生产计时](../../ibs_engine.py#L20171)。

当前A/B报告比较total_production_steps、n_sampling_segments、n_autonomous_iterations，没有比较实际wall-clock。窗口K、探针构建、warmup、I/O及分析开销不同，少步数不等于更高效率。

改进：增加完整阶段墙钟时间、有效样本/秒、达到预设可信度/误差目标的总时间，并同时保留协议可比性、完整路径状态和跨重复证据。生产loop_timing_s只覆盖部分过程，不能直接当成整个ABFE任务耗时。

验证：在同硬件、相同物理协议、明确总预算条件下比较baseline/candidate；改动λ布局时只比较可比的路径级结果，不硬配不同窗口。效率指标以时间相关处理后的样本信息为依据，不能只用原始帧数或未经校正的权重ESS。

### 条件分支补充：带电APBS修正的采样不确定度

仅适用于显式rocklin_apbs_neutralizing_plasma及有限APBS snapshots，不适用于默认中性路线。

[apbs_correction.py:747](../../apbs_correction.py#L747)已有snapshot mean、SD、SEM；[汇总入口](../../runabfe.py#L8980)仅传修正标量，[总误差公式](../../abfe_core.py#L9072)没有APBS不确定度项。可以进一步传递基于足够独立snapshot估计的修正不确定度，并根据与两腿误差的相关性传播；不能无条件把SEM和SD互换，也不能默认所有项独立。

### 建议执行顺序与本轮验证边界

先做QUALITY-01、QUALITY-02、QUALITY-03，建立可信的误差和慢构象采样基准；并同步补QUALITY-07的完整时间指标。之后再校准QUALITY-04的pilot/预算策略，做QUALITY-05和QUALITY-06的数值与修正对照。

本轮为当前源码的只读审查：D:项目未写入，未运行项目测试、GPU采样或profiler。工作区主线output仍为空；没有本体系的ΔG敏感性、相关时间、跨重复SD或耗时收益实测。上述顺序是科学验证与实施优先级，尚不是已测得的收益排名。
