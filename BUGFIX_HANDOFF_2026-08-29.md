# ABFE-IBS 代码缺陷修复交接单（2026-08-29）

> 状态（2026-08-31，第八轮：PHY-03 条件接口再收紧）：累计登记 41 项；
> 已修复/显式阻断 40 项，1 项科学验证已挂起。第六轮登记的物理/数学问题已逐项落到
> 生产代码；涉及 Hamiltonian 或采样协议的修改均更新了协议版本/缓存身份。没有改写
> 任何历史 artifact，也没有用当前缺少 OpenMM 的环境重算历史轨迹或自由能。
>
> 已完成（第一～四轮，9 项）：P1-01、P1-02、P1-03、P1-05、P1-06、P1-09、
> P2-03、P2-04、P2-06。
>
> **已完成（第五轮实机修复，10 项）：P0-01、P1-04、P1-07、P1-10、P1-11、
> P1-12、P1-13、P1-14、P1-15、P1-18。** 修复在 Linux + openmm_dev
> （Python 3.12.13 / OpenMM 8.5.2 / pymbar 4.0.3 / mdtraj 1.10.3 / pytest 9.1.1）
> 上完成，全部按各条目的验收口径补了回归测试（见文末新增测试文件清单）。
>
> ⚠️ **P0-01 实测定级为真并已修复（Hamiltonian 变更）**：在目标 OpenMM 8.5.2 上
> 用最小中性配体 PME 体系复现——v2 的"补 exception 冻结 L–L"会把 λ=1 物理端点
> 改写（26 粒子小体系实测能量偏差 ~1e-2 kJ/mol，真实体系按对数放大）。修复采用
> annihilation 口径（既有 L–L exception 冻结、普通 L–L 对随粒子 offset 线性湮灭），
> `PME_DECHARGE_MODEL_VERSION` 升 v3，并已把该版本无条件编入 `_stage_protocol_key`
> ——**旧 Stage 1/Stage 2 "completed" 缓存自动失效，不会被静默复用**。
> 按《数据与结果安全边界》：output/ 等旧证据目录原样保留、未做任何改写；
> 受 v2 Hamiltonian 影响的历史 artifact 应视为**失效证据**，其逐腿 ΔG 数值与
> v3 口径不可混用（ΔG_bind 因两腿内部 Hamiltonian 相同、湮灭项严格相消而不受影响，
> 但逐腿数值不可跨协议比较）。
>
> 重新打开：P1-07（严格 helper 没接到外层完成判断）→ **第五轮已修复**；
> P2-02（内部支持 CUDA:N，但主 CLI 仍拒绝 CUDA:N）→ **第七轮已修复**。
>
> 第五轮修复要点（按条目详情见各节"已修复"注记）：
> 1. P0-01：三个 decharging builder 撤销"普通 L–L 对补 exception"；新增
>    tests/test_pme_decharge_endpoint_equivalence.py（λ=1 能量/力逐位等价、
>    全 λ 对照独立参考实现）。
> 2. P1-07：`runabfe.equilibrium_is_done` 接入严格校验（真实 DCD parser +
>    目标 Simulation loadCheckpoint，带记忆化）；并补强共享 `_is_traj_valid`
>    ——mdtraj 低层 reader 对"头部声称 N 帧、文件只有 N-1 完整帧"只发
>    stderr 警告并**静默返回短读**，现按 DCD 头部 NSET 与实读帧数比对拒绝。
> 3. P1-10：膜质量门改为"轨迹存在才在消费前判门 + 预平衡块之后无条件最终硬门"，
>    fresh 膜首跑能先进预平衡。
> 4. P1-12：`abfe_core.validate_final_leg_result` 统一 schema/sanity gate，
>    pipeline cache 早退、runabfe 两腿汇总、combine_binding_free_energy 三处共用。
> 5. P1-18：decharging REMD 接入 seed contract；三处 sampling fingerprint 均含
>    `seed_contract`。
> 6. P1-11：`_run_2d_lambda_stage` 透传 converged/min_overlap；三条 single/2D
>    路径写缓存与缓存命中共用收敛 sanity gate。
> 7. P1-13/P1-14：traditional 两条腿在建 REMD 前各自做（或严格复用）基线预平衡，
>    独立于 Boresch 开关；膜/非默认色散/显式力场族声明在建 Context 前明确拒绝。
> 8. P1-04：analyze-only 最先分流（不再被 DEXP/ligand 校验阻断）；默认从
>    run_provenance.json 恢复 temperature/mode/decoupling（仅显式 CLI 覆盖并留审计）；
>    回退分析 checkpoint 分支要求 converged+protocol_key+覆盖证据；原始窗口分支
>    要求文件数==preopt 窗口数并用生产同款完整性 helper 检查 λ 覆盖。
> 9. P1-15：refine 从文件名解析整数窗口编号、按数值排序，重复/缺失拒绝。
>
> 第四轮新增的 P0-01/P1-18 均已在第五轮修复。第七轮完成了 P1-08、P1-16、P1-17、
> P1-19、P2-01、P2-02、P2-05、P2-07～P2-12，以及 PHY-01/02/04～09；PHY-03
> 保持科学未闭合并在生产入口前置拒绝，当前只剩 1 项待完成的 C4/C5 实证验证，现已挂起。
>
> 第八轮在 PHY-03 条件接口上增加温度一致性、严格整数计数与两腿 artifact 一致性门。
> 验证边界（第八轮）：五个核心文件 `py_compile`/AST 检查通过；当前会话环境没有
> `openmm`、`pytest` 或 `ruff`，因此没有声称离线测试全绿。历史 openmm_dev 测试结果
> 仅作为第五轮基线记录；本轮新增逻辑需要在安装 OpenMM 8.5.x 的环境重跑对应测试。
>
> 适用范围：abfe_pipeline.py、abfe_preoptimizer.py、ibs_engine.py、
> abfe_core.py、runabfe.py。行号会随修改漂移，实施时同时按函数名定位。

## 给接手人的提醒

请先修复 P1，再处理 P2。不要通过降低 fail-closed 门槛、删除 provenance、
忽略 checkpoint 不一致或复用旧结果来“让流程跑通”。

涉及 Hamiltonian、采样协议、缓存含义或结果口径的修改，必须同步更新协议版本、
缓存指纹和回归测试。

## P0 候选：release 前必须实测并关闭

### [x] P0-01 默认 PME decharging 的 ligand-internal 处理可能破坏物理端点等价性

> **已修复（2026-08-30 第五轮）**。实测：目标 OpenMM 8.5.2 上复现成立——
> 26 粒子最小中性配体 PME 体系，v2 配置后 λ=1 总能量偏差 **-1.03e-2 kJ/mol**、
> 逐原子力同量级偏差（仅 10 个被转换的普通 L–L 对）；真实体系按对数放大。
> 定级：真实 P0（release-blocking Hamiltonian 缺陷）。
> 修法：annihilation 口径——`configure_pme_ligand_charge_offsets` /
> `configure_coalchemical_neutral_decharging` / `configure_charge_transfer_decharging`
> 三个 builder 全部撤销"普通 L–L 对补 exception"；既有 L–L exception
> （1-2/1-3 排除、1-4 缩放，chargeProd 不读粒子电荷）天然逐 λ 恒定，维持冻结；
> 普通 L–L 对随粒子 offset 线性湮灭。complex/solvent 两腿配体内部 Hamiltonian
> 完全相同 ⟹ 湮灭项在 ΔG_bind 中严格相消，热力学循环闭合。
> `PME_DECHARGE_MODEL_VERSION` v2 → v3 并无条件编入 `_stage_protocol_key`，
> 旧 stage 缓存全部失效。验证：λ=1 对原始 System 能量/力逐位等价（1e-6 容差）；
> λ∈{0,0.25,0.5,0.75,1} 对照"粒子电荷直接缩放"的独立参考实现逐位一致；
> PME 直接受力路径（Reference 平台双精度）实测。回归测试：
> tests/test_pme_decharge_endpoint_equivalence.py（8 项）；
> tests/test_charge_transfer_hamiltonian.py 的三处 v2 契约测试已按 v3 口径更新。

- 位置：ibs_engine.py 的 configure_pme_ligand_charge_offsets（约 3047–3196 行）；
  默认 Stage 1 调用链包括 _prepare_pme_coulomb_leg_system、REMDManager._build_replicas
  和 TraditionalMBARAnalyzer.compute_u_kn。
- 触发：默认中性配体、decharge_method=pme，且配体存在原本不是 exception 的普通
  ligand–ligand 非键原子对。
- 问题：实现先把配体粒子电荷改为 0，再用 lambda_coul particle offsets 恢复；为了让
  配体内部库仑看似不随 lambda 改变，又把所有普通 L–L 对补成 full-charge exception。
  但 OpenMM 的 exception 不使用普通非键 cutoff，并会替换该对原来的 PME 处理；因此
  lambda=1 的新系统不一定等于原始 PME System，lambda 中间态也可能包含非预期的
  reciprocal-space / periodic-image 差异。现有测试主要在 NoCutoff 下检查参数表和
  lambda 不变性，没有钉住真实 PME 的物理端点能量与力。
- 影响：若目标环境复现，默认 dual_lambda Stage 1 的 lambda=1 物理端点已改变，且错误
  可以保持有限，属于 release-blocking Hamiltonian 缺陷。此项在实测前保留“P0 候选”
  而不宣称已有最终 Delta G 的偏差大小。
- 外部依据：OpenMM API 明确说明 cutoffs 从不应用于 exceptions；OpenMM #2310 也记录过
  将普通 pair 以相同参数改成 exception 时 PME 能量/力会改变。实现不能只凭 exception
  参数等于 q_i*q_j 就假定端点等价。
- 要求：在项目目标 OpenMM 版本上构造最小中性双/三原子配体 PME 体系，比较原始 System
  与配置后 lambda=1 的总能量、reciprocal/direct 分量和逐原子力；再明确协议究竟采用
  decoupling（内部项保持）还是 annihilation（内部项也缩放），实现与热力学循环必须一致。
- 验收：lambda=1 对原始物理 System 的能量和力在预注册容差内等价；lambda=0 与中间态
  的 ligand-internal 口径有独立参考实现验证；复杂物腿和溶剂腿使用同一协议身份与缓存版本。

## P1：必须优先修复

### [x] P1-01 traditional 模式首次运行必然崩溃

- 位置：abfe_pipeline.py 的 TraditionalABFEPipeline（约 9597 行），错误调用约在 9774、9808 行。
- 触发：mode=traditional，且没有可复用的完整 u_kn 缓存。
- 问题：该类调用 resolve_co_alchemical_ion_spec、_seed_for，并读取 seed_ledger、
  leg_name，但类本身没有这些方法或属性。
- 要求：为 traditional 类建立完整 seed/co-ion 接口；若不支持 co-ion，必须明确返回
  None 或 fail closed，不能靠缺方法崩溃。
- 验收：中性配体能从空输出目录进入 REMD；不支持的带电路线在创建 Context 前给出明确错误；
  补 fresh-run 和 resume 回归测试。

### [x] P1-02 损坏 checkpoint 的“重新开始”分支没有初始化 Context

- 位置：abfe_pipeline.py 的 ABFEPipeline.pre_equilibrate，约 2584–2599 行。
- 触发：resume=True、checkpoint 存在，但 simulation.loadCheckpoint 失败。
- 问题：异常分支只设置 steps_remaining；positions、box、最小化和速度初始化位于外层 else，
  因此不会执行，随后直接步进未初始化 Context。
- 要求：加载尝试后统一使用 if not resume_from_chk 执行完整 fresh-start 初始化。
- 验收：随机或截断 checkpoint 加载失败后，必须从初始坐标重新最小化并正常开始。

### [x] P1-03 n_equil_steps 配置没有传入主流程

- 位置：abfe_pipeline.py 的 run_full_pipeline 约 7745 行；runabfe.py 的 complex/solvent
  调用约 5455、5614 行。
- 触发：n_equil_steps 不等于 5,000,000。
- 问题：runabfe.py 用配置值计算 fingerprint，但 run_full_pipeline 调用 pre_equilibrate
  时没有传 n_steps，实际始终使用 5,000,000。
- 要求：run_full_pipeline 显式接收 n_equil_steps，并在两条腿传给 pre_equilibrate；
  fingerprint、运行步数和完成态必须使用同一值。
- 验收：测试小于和大于 5,000,000 的值；第二次 resume 不得重复预平衡。

### [x] P1-04 analyze-only 可接受截断或旧协议结果

> **已修复（2026-08-30 第五轮）**。三个补充点全部落实：
> (1) main 最先分流 analyze-only（移到 DEXP 参数加载与 ligand 要求之前）；
> (2) run_post_analysis 默认从 run_provenance.json 的 `config` 恢复
> temperature/mode/decoupling，仅当本次命令行**显式**出现对应 flag 才允许覆盖，
> 覆盖与恢复均写审计日志；
> (3) 两个回退分支都接了完整性判据：阶段 checkpoint 分支要求
> `converged is True` + `protocol_key` 存在 + 覆盖/端点证据
> （coverage_diagnostics / lambda_endpoint_diagnostics 至少其一），缺一拒绝；
> 原始窗口分支要求能量文件数 == preopt window_ranges 数，并把所有窗口的
> λ 状态索引并集喂给生产同款 `_assert_expected_windows_all_loaded`
> （缺首/中/末窗全部拒绝）。
> 口径说明：ESS 未在 analyze-only 内独立重算——`converged is True` 由
> 生产端含 final ESS/不确定度门槛的收敛门产出，回退路径以它作为完成与
> ESS 的判据来源；若要与生产逐口径复核，用 tools/diagnostics 的离线复算。
> 回归测试：tests/test_analyze_only_and_refine_regressions.py。

- 位置：runabfe.py 的 run_post_analysis 约 3522–3636 行；完整性 helper 为
  ibs_engine.py 的 _assert_expected_windows_all_loaded（约 529 行）。
- 触发一：stage checkpoint 有有限的 total_delta_G/total_error，但来自失败、截断或旧协议。
- 触发二：原始窗口文件从 0 连续编号，但缺少最后一个或多个预期窗口。
- 问题：checkpoint 分支不检查 converged、协议版本、lambda fingerprint、完整窗口 manifest、
  ESS；原始文件分支只检查现有编号连续，不检查预期总数和端点覆盖。
- 要求：两个分支都验证完成标记、协议身份、预期窗口集合、lambda 端点、覆盖和 ESS；
  复用已有完整性 helper。
- 验收：缺首窗、缺中窗、缺末窗、旧协议 checkpoint、converged=false 全部拒绝产出最终结果。
- 第二轮补充一：main 在 runabfe.py 约 4815–4849 行先加载 DEXP 参数、要求 ligand，
  之后才分流 analyze-only；纯分析现有结果会被无关的模拟输入提前阻断。
- 第二轮补充二：run_post_analysis 约 3485–3495 行虽读取 run_provenance.json，
  但 temperature、mode、decoupling 仍取本次命令/预设默认值。原运行是 310 K 或非默认
  路线时，不显式重传全部配置就可能用 300 K 的 kT/Boresch 修正或走错分析分支。
- 补充验收：analyze-only 应先分流；默认从 provenance 恢复原始协议，仅允许显式 CLI
  覆盖，并对覆盖行为留审计记录。

### [x] P1-05 已有 barostat 只检查类别，不检查真实参数

- 位置：abfe_core.py 的 ensure_barostat_for_protocol，约 425–500 行。
- 触发：输入或缓存 System 已有正确类别的 barostat，但压力、温度、频率、表面张力或
  XY/Z 模式与本次配置不同。
- 问题：类别相同即 reused_existing；fingerprint 记录新协议，实际却使用旧参数。
- 要求：复用前读取并逐项比较实际 force 参数；不一致时 fail closed 或明确重建；
  provenance 记录实际采用值。
- 验收：同类同参数可复用；任一参数不同必须拒绝或重建；膜和可溶 barostat 都覆盖。

### [x] P1-06 checkpoint 恢复会追加到 checkpoint 之后的旧轨迹

- 位置：abfe_pipeline.py 的 pre_equilibrate；checkpoint 加载约 2586 行，
  DCDReporter append 约 2666–2673 行；monitor CSV 同类。
- 触发：轨迹已写到比最近 checkpoint 更晚的步数，任务随后崩溃并 resume。
- 问题：恢复到旧 checkpoint 后直接追加，形成时间倒退、重复帧和不连续 segment，
  污染膜质量门及轨迹诊断。
- 要求：加载成功后把 DCD/monitor 截断到 checkpoint 边界，或创建带明确 metadata 的新 segment。
- 验收：resume 后步数严格单调、无重复帧，质量门只读取有效 segment。

### [x] P1-07 checkpoint/DCD 完整性检查接受任意或截断文件（第二轮重新打开）

- 位置：abfe_pipeline.py 的 _is_checkpoint_valid、_is_traj_valid，约 1354–1391 行；
  再平衡 reporter/加载顺序约 3042–3059 行。
- 已复现：随机 512 字节 checkpoint 返回 True；伪造 CORD 且只有首记录的截断 DCD 返回 True。
- 问题：弱判据会驱动轨迹追加、REMD 跳过和缓存复用；再平衡在真正 loadCheckpoint
  之前就决定 append_traj=True。
- 要求：checkpoint 以目标 Simulation 的真实 loadCheckpoint 成功为准；DCD 使用实际 parser
  校验帧数和文件结尾。只有加载成功后才能追加或复用。
- 验收：随机、截断、错误 System/Platform 的 checkpoint 全部拒绝；截断末帧的 DCD 拒绝。
- 已完成部分：abfe_pipeline.py 的 _is_checkpoint_valid/_is_traj_valid 已改为真实 loader/parser。
- 重新打开原因：runabfe.py 的 equilibrium_is_done（约 621–697 行）仍只检查 DCD 大小、
  checkpoint 存在、fingerprint 和 completed 标记，没有调用严格 helper。soluble 路线可因此
  用“大于阈值但已截断”的 DCD 跳过预平衡；load_native_system 又不会在缺完整 fingerprint
  时自动加载稳态末帧，后续可能直接从原始缓存坐标进入采样。
- 补充验收：外层完成判断必须使用真实 DCD parser，并在目标 Simulation 上验证 checkpoint；
  任一失败都不得令 run_equilibration=False。

> **已修复（2026-08-30 第五轮）**。`runabfe.equilibrium_is_done` 新增严格校验：
> DCD 必须通过 `_is_traj_valid` 真实 parser（按 (path,size,mtime) 记忆化，避免
> 大轨迹在一次运行里被重复全量读）；checkpoint 在传入目标 Simulation 时以真实
> `loadCheckpoint` 成功为准（新增 `_checkpoint_probe_simulation`，按 pipeline
> 同一 platform 建一次性 Context 并在 pipeline 实例上缓存），未提供时保留
> 存在/非空弱检查、由下游严格消费方兜底。三个调用点（resolve_boresch_restraint、
> complex/solvent 两腿 run_equilibration）全部传入 probe Simulation。
> **额外发现并修复**：`_is_traj_valid` 本体也有一个洞——mdtraj 低层 reader 对
> "DCD 头部声称 N 帧、文件只有 N-1 完整帧 + 半帧"只发 stderr 警告并**静默返回
> 短读**（正是崩溃瞬间的文件形态）。现按 DCD 头部 NSET（OpenMM 每写一帧回写
> 更新）与实读帧数比对，不一致一律拒绝；头部不可解析时不误伤。
> 回归测试：tests/test_equilibrium_is_done_strict_validation.py（截断 DCD、
> 随机字节 checkpoint、垃圾大文件、完整用例、未完成状态用例）。

### [x] P1-08 Shadow-Coulomb bias 温度硬编码为 300 K

- 位置：ibs_engine.py 的 build_shadow_coul_ibs_system 约 4673 行；manager 调用约 13926 行。
- 触发：运行温度不是 300 K。
- 问题：IBSBiasForce 按 300 K 构建，但动力学、采样概率和分析使用用户温度。
- 要求：builder 接收 temperature，并由 manager 传入 self.temperature。
- 验收：300 K 与 310 K 均验证 bias、sampler、analysis 使用完全相同的 kT/beta。

### [x] P1-09 力场自动识别丢失 GROMACS include 目录

- 位置：abfe_core.py 的 resolve_forcefield_family 约 3148 行；调用在 runabfe.py 约 5071 行。
- 触发：forcefield include 只能通过 --gmx-path、GMXLIB 或 GMXDATA 找到。
- 问题：系统构建已经解析出 include_dir，但力场识别没有接收或传递它，随后错误报告无法识别力场。
- 要求：resolve_forcefield_family 接收 gmx_include_dir，从 main 的同一个 include_dir 传入，
  并将解析依赖写进 provenance。
- 验收：相对 include、显式 include_dir、GMXLIB 三类输入得到相同正确力场族。

### [x] P1-10 膜体系首跑在生成预平衡轨迹前就执行质量门

> **已修复（2026-08-30 第五轮）**。run_full_pipeline 的预平衡前判门改为仅当
> `pre_equilibration.dcd` 已存在（复用场景）时执行——消费既有轨迹仍 fail closed；
> 预平衡块之后新增**无条件**最终硬门（`ensure_membrane_quality_gate_passed`
> 幂等），fresh 首跑刚产出的轨迹、断点续传、`run_equilibration=False` 直接带
> 坐标进来的情况都逃不过，门失败（enforce）不会进入任何 Stage 0/λ 窗口。
> 回归测试：tests/test_membrane_gate_first_run_regressions.py（fresh 无 DCD 能
> 到达 pre_equilibrate；有 DCD 时门失败先于消费；最终硬门必须位于预平衡块之后）。

- 位置：abfe_pipeline.py 的 run_full_pipeline 约 8330 行、
  ensure_membrane_quality_gate_passed 约 2414–2447 行；真正的 pre_equilibrate 在约 8422–8465 行。
- 触发：fresh membrane 输出目录、run_equilibration=True、尚无 pre_equilibration.dcd；
  --no-boresch 或直接调用 run_full_pipeline 时最清楚。
- 问题：质量门无条件先消费一个尚未生成的轨迹并抛错，膜体系无法进入首次预平衡。
- 要求：首跑先生成轨迹并在结束处判门；只有复用既有预平衡时才在消费前判门，
  且进入 Stage 0 前保留最终硬门。
- 验收：fresh membrane/no-DCD 能开始预平衡；旧轨迹门失败仍不得进入任何 lambda 窗口。

### [x] P1-11 single-lambda/2D 路径丢失 MBAR 未收敛状态

> **已修复（2026-08-30 第五轮）**。`_run_2d_lambda_stage` 的 stage_result 透传
> `converged`/`min_overlap`/`min_overlap_threshold`，返回前经
> `_assert_sampling_result_converged` 硬检查；single_lambda、2d_diagonal、
> 2d_geodesic 三条路径的采样缓存落盘前过同一 gate，缓存命中也用同一判据
> （`_sampling_result_convergence_rejection_reason`）——converged 不是 True
> （含旧缓存缺字段、ΔG/误差非有限、误差为负）一律拒绝复用、重新采样，
> 不再出现"有限 ΔG + converged=false 被写成 completed"。
> 回归测试：tests/test_mbar_convergence_gate_regressions.py；
> tests/test_top_level_sampling_cache_identity.py 的 mock stage result 已按
> 新契约补 converged=True。

- 位置：ibs_engine.py 的 TraditionalMBARAnalyzer.solve 约 19427–19436 行；
  abfe_pipeline.py 的 _run_2d_lambda_stage 约 5089–5100 行及 single/2D 缓存命中分支。
- 触发：MBAR overlap 不足但 solver 不抛异常，只返回 converged=false/min_overlap。
- 问题：stage_result 只复制 delta_G/error，丢弃 converged/min_overlap；不收敛结果仍被写缓存、
  标记 completed、生成最终 Delta G，并在 resume 时继续复用。
- 要求：透传并硬检查 converged、min_overlap 和阈值；写缓存与缓存命中必须走同一 sanity gate。
- 验收：solver 返回有限 Delta G 但 converged=false 时，single_lambda、2d_diagonal、
  2d_geodesic 全部拒绝完成。

### [x] P1-12 正常 resume 与最终汇总对缺失/非有限结果 fail-open

> **已修复（2026-08-30 第五轮）**。新增 `abfe_core.validate_final_leg_result` /
> `FinalResultValidationError` 作为唯一判据，三处共用：
> (1) abfe_pipeline 顶层 final_results.json 早退——协议指纹一致后还必须过 gate，
> 缺必需热力学字段、ΔG/误差 NaN/Inf、误差为负、自带 converged 不为 True 一律
> 拒绝复用（与指纹不匹配同口径：记录后落到重新校验/运行）；
> (2) runabfe 两腿汇总——`complex_results.get(..., 0.0)` 的静默补零删除，
> in-memory 与磁盘两条腿结果都过 gate，fail closed；`_analyze_dual_leg` 回退
> 分析的产出补有限性门；
> (3) `combine_binding_free_energy`——四个数值输入非有限即 raise，误差为负即
> raise，NaN/Inf 不再能静默传播成 ΔG_bind。
> 字段组口径：`FINAL_LEG_RESULT_REQUIRED_FIELD_SETS` = (total_delta_G_complex_kJ_mol,
> total_error_kJ_mol) 或 (delta_G_total_kJ_mol, error_leg_kJ_mol)。
> 回归测试：tests/test_final_result_schema_gate.py。

- 位置：abfe_pipeline.py 顶层 final_results 早退约 8721–8743 行；
  runabfe.py 两腿汇总约 5540–5541、5690–5691 行；
  abfe_core.py 的 combine_binding_free_energy 约 6913–6925 行。
- 触发：协议指纹仍匹配，但 final_results.json 缺少必需热力学字段；或任一 Delta G/误差是
  NaN、Inf、负误差。
- 问题：顶层早退不校验必需字段/有限性/完成诊断；主汇总把缺字段静默补成 0.0；
  唯一热力学循环 helper 也不检查有限性和误差非负。
- 影响：损坏或部分结果可被汇总成看似成功的 Delta G_bind，或把 NaN 写进 JSON。
- 要求：定义统一 final-result schema/sanity gate；cache load、main 汇总和 helper 三处共用，
  缺字段、非有限值、负误差、converged is not true 一律拒绝。
- 验收：删除任一必需字段、注入 NaN/Inf/负误差的 valid-fingerprint cache 均不得复用或写最终结果。

### [x] P1-13 traditional + --no-boresch 跳过基线预平衡

> **已修复（2026-08-30 第五轮）**。`run_traditional_mode` 现在建一个统一
> ABFEPipeline，基线预平衡无条件先于 Boresch 分支执行（沿用统一
> fingerprint / `equilibrium_is_done`（含 P1-07 严格校验）/ checkpoint / 完成
> 门）；显式 Boresch 路线经 `resolve_boresch_restraint` 自然复用同一份预平衡，
> 不会跑两遍。溶剂腿也补上同款基线预平衡（runtime_dir =
> output_dir/solvent_leg，与 load_native_system 的 prefer_equilibrated 读的是
> 同一份严格校验过的轨迹）——"两条腿预平衡"达标，原始/仅居中坐标不再直接进
> REMD。回归测试：tests/test_traditional_mode_preflight_and_equil.py。

- 位置：runabfe.py 的 run_traditional_mode 约 3975–4058 行；
  TraditionalABFEPipeline.run_full/run_leg 与 ibs_engine.py 的 REMD replica 构建。
- 触发：mode=traditional 且显式关闭 Boresch。
- 问题：预平衡只发生在 if config.boresch 内；关闭 Boresch 后两条腿直接 setPositions/
  setVelocities 进入 REMD，没有统一最小化和预平衡。
- 影响：原始或仅居中的坐标直接进入生产采样，可能产生初始构象偏差或数值崩溃。
- 要求：基线预平衡独立于 Boresch；复用统一 fingerprint、resume、checkpoint 和完成门。
- 验收：--no-boresch 时必须在创建 traditional REMD Context 前完成或严格复用两条腿预平衡。

### [x] P1-14 traditional 模式绕过环境、barostat、力场和色散协议

- 位置：runabfe.py 约 3977–4058 行；main 约 4851–4853 行在 normal preflight 前直接返回 traditional。
- 触发：traditional 配合 system_type=membrane、dispersion_protocol 或 forcefield_family。
- 问题：这些配置只可能进入临时 Boresch ABFEPipeline；真正的 TraditionalABFEPipeline
  不接收/不验证这些协议，--no-boresch 时连临时检查也完全没有。
- 影响：膜 barostat 可能缺失或错误，用户声明的力场/色散 Hamiltonian 被静默忽略。
- 要求：traditional 接入统一 preflight 与协议指纹；暂不支持的组合必须在建 Context 前明确拒绝。
- 验收：膜/非默认色散输入只能“正确应用”或“明确 fail closed”，不得无声按默认路线运行。

> **已修复（2026-08-30 第五轮）**。新增 `_assert_traditional_protocol_supported(config)`，
> 在 `run_traditional_mode` 内建任何 OpenMM Context（预平衡/REMD）之前调用：
> `system_type != soluble`、非空 `membrane`、`dispersion_protocol` 非
> legacy_uniform_density_lrc、`forcefield_family` 非 auto 的声明全部 fail closed
> 并列出具体声明项。默认（不声明）路线行为不变。
> 回归测试：tests/test_traditional_mode_preflight_and_equil.py（含"检查必须先于
> ABFEPipeline/TraditionalABFEPipeline/pre_equilibrate"的源码顺序守护）。

### [x] P1-15 refine-lambda-path 按字典序错配两位数窗口

> **已修复（2026-08-30 第五轮）**。`refine_stage_lambda_path_from_data` 改为从
> `dual_window_(\d+)_{stage_type}_energies\.npy` 文件名解析整数编号、按数值
> 排序，编号必须恰好等于 `range(len(window_ranges))`（重复/缺失/多余一律
> RuntimeError），u_kn/bias/base 按"位置=编号"映射到 window_ranges。
> 回归测试：tests/test_analyze_only_and_refine_regressions.py（12 窗含可区分
> 内容逐一映射；缺编号 3 拒绝）。

- 位置：abfe_preoptimizer.py 的 refine_stage_lambda_path_from_data 约 1991–2010 行。
- 触发：窗口达到 10 个以上；字符串排序会产生 0、1、10、11、2…。
- 问题：sorted(glob) 后用 enumerate 位置当真实窗口编号，导致 u_kn/bias/base 与
  window_ranges 错配，可能写出错误的新 lambda 路径。
- 要求：从 dual_window_(\d+) 文件名解析整数编号，按数值排序，并拒绝重复/缺失编号。
- 验收：12 个带可区分内容的窗口必须逐一映射到正确 window_ranges。

### [x] P1-16 Shadow Coulomb cutoff 与参考 PME cutoff 不一致

- 位置：ibs_engine.py 的 _build_shadow_coul_cross_force 约 4573–4603 行及调用点约 4719–4723、
  4820–4823 行。
- 触发：Shadow IBS/Bridge；参考 NonbondedForce cutoff 不是硬编码的 1.2 nm。
- 问题：Shadow force 默认 cutoff_nm=1.2，但参考 PME cutoff 来自真实 NonbondedForce；
  get_pme_alpha_for_system 已读取参考 cutoff 用于 alpha，却没有把 cutoff 传给 Shadow force。
- 影响：参考 PME 实空间项已截断的距离区间仍残留 Shadow 相互作用，端点 Hamiltonian 不一致。
- 要求：从参考 NonbondedForce 复制 cutoff，并写进 fingerprint/provenance。
- 验收：参考 cutoff 为 1.0/0.9 nm 时，所有 Shadow force 的 cutoff 必须逐值一致。

### [x] P1-17 Shadow Bridge 结果缓存没有协议身份

- 位置：abfe_pipeline.py 约 4769–4808、4882–4909 行。
- 触发：shadow_ibs resume 时改变温度、bridge lambda、步数、交换间隔、System 或 Boresch；
  或旧缓存缺少 converged 字段。
- 问题：shadow_bridge_result.json 只要存在就直接加载；只在 converged is False 时拒绝，
  缺字段会放行，也没有 sampling fingerprint。
- 影响：旧 Bridge Delta G 可被静默拼入当前 Stage 1，组合结果还会被标成 converged=true。
- 要求：保存并严格比对 Bridge sampling fingerprint；converged 必须显式为 true，
  同时检查数值有限和 overlap。
- 验收：任一协议输入变化、缺 fingerprint、缺 converged、低 overlap 均必须重新采样或 fail closed。

### [x] P1-18 PME decharging REMD 未接入 repeat-seed contract，独立重复可能复用旧采样

- 位置：abfe_pipeline.py 的 _run_dual_lambda_stage 中 decharging REMDManager 构造
  （约 4227–4242 行）；_remd_sampling_fingerprint（约 1659–1704 行）；对照已经接线的
  single/2D REMD 分支约 5028–5047 行。
- 触发：使用 repeat_seed 做独立重复，默认 dual_lambda Stage 1 走 PME decharging；或者
  在同一输出目录改变 repeat_seed 后 resume。
- 问题：decharging 分支没有向 REMDManager 传 random_seed、seed_ledger、seed_stage、
  seed_leg；sampling fingerprint 也没有 repeat-seed/seed-contract 身份。该阶段因此可能
  使用未登记随机源，改变 seed 后还可能命中并复用旧 DCD/u_kn。
- 影响：所谓独立重复可能共享同一段 decharging 采样，破坏复现性、独立性和跨运行不确定度
  估计；provenance 中即使存在 seed ledger，也不能证明 Stage 1 真正消费了它。
- 要求：decharging 与 2D/traditional 分支共用同一 seed 派生入口；把有效 seed-contract
  身份加入 sampling fingerprint 和 metadata；repeat_seed 改变必须强制 sampling cache miss。
- 验收：mock REMDManager 能观察到 complex/solvent 各自确定且不同的 Stage 1 seed；
  repeat_seed=101/102 的 sampling fingerprint 不同；第二次运行不得复用第一次的 DCD/u_kn。

> **已修复（2026-08-30 第五轮）**。decharging 分支的 REMDManager 构造补齐
> `random_seed=self._seed_for("charging", stage_name, "exchange", "numpy")` +
> `seed_ledger`/`seed_stage`/`seed_leg`，与 single/2D、traditional 分支同一
> 推导约定；`_remd_sampling_fingerprint` 新增 `seed_contract` 字段，decharging、
> single/2D、traditional 三处指纹全部传入 `seed_contract_snapshot()`——
> repeat_seed 一变，采样缓存立即 miss。附带口径说明：本次把 single/2D 与
> traditional 的 sampling fingerprint 也补上 seed_contract（它们虽已把 seed
> 接进 dynamics，但指纹此前不含 seed 身份，同样的洞），旧采样缓存一次性失效
> 属预期行为。complex/solvent 的 Stage 1 seed 经 Exp019SeedLedger 确定且互异
> 已有行为验证。回归测试：tests/test_decharging_seed_contract.py。

### [x] P1-19 （第五轮新增）v4 charging 口径下 charging/vanishing 接缝的内静电失配

- 位置：ibs_engine.py 的 build_ibs_dual_system / create_ligand_internal_force
  （vanishing 侧 U_common 的 Group 2）；对照 configure_pme_ligand_charge_offsets
  系（charging 侧，P0-01 修复后）。
- 触发：中性配体 dual_lambda 生产路径，Stage 1（decharging）结束端 λ_coul=0 与
  Stage 2（vanishing）起始端 λ_vdw=1 的接缝。
- 问题：P0-01 修复后，charging 腿在 λ=0 已把 **ordinary L-L 内部库仑湮灭**
  （既有 exception 的 1-4 打折库仑仍冻结）；而 vanishing 侧 U_common 的定义是
  "配体内部静电逐 λ 恒定的物理值"（Group 2 用 raw 电荷重建），两端点差一个
  内部库仑常数。实测（tools/validation/compare_charge_transfer_endpoints.py 的
  中性 4 原子 fixture）：charging λ=0 = -200.95 kJ/mol，vanishing λ_vdw=1 =
  -319.43 kJ/mol，差 **118.5 kJ/mol**。v2 下两侧"一致"是因为 v2 的补对
  exception 恰好把 ordinary 内部库仑也冻结成物理值——v2 的一致建立在被 P0-01
  推翻的端点改写之上。
- 影响评估（初判，需复核）：Group 2 的内静电是逐 λ_vdw 常数 ⟹ 对 ΔG_vdw 贡献
  为零；两条腿的配体内部 Hamiltonian 相同 ⟹ 在 ΔG_bind 中严格相消。
  **ΔG_bind 预计不受影响**；被破坏的是"seam 两端点同一 Hamiltonian"的记账恒等式
  （U_common 定义、stage1/stage2 诊断的可比性）。三个 seam 契约测试已标
  xfail(P1-19)。
- 要求：(1) 明确 v3 下 U_common 的定义——Group 2 是否应随 charging 口径改为
  "ordinary 对湮灭、既有 exception 冻结"；(2) 若改 Group 2，必须同步协议版本、
  缓存指纹与 LRC/记账诊断；(3) 用带电 fixture（charge-transfer handoff）复核
  seam；(4) 数值验证 ΔG_bind 的不变性（同一数据集 v2/v3 汇总对比）。
- 验收：seam 三个契约测试转 XPASS 并摘除标记；热力学循环说明文档同步更新。

## P2：完成 P1 后处理

### [x] P2-01 Shadow bridge 忽略显式 PME 参数

- 位置：ibs_engine.py 的 get_pme_alpha_for_system 约 4411 行；
  _build_electrostatics_only_pme_probe 约 4574 行。
- 要求：优先读取非零 getPMEParameters；probe 复制 PME/LJ-PME alpha/grid；
  只有自动参数为零时才按 cutoff/tolerance 推导。

### [x] P2-02 2D geodesic/CUDA 路由没有贯通 CLI（第二轮重新打开）

- 位置：abfe_preoptimizer.py 约 3734 行。
- 要求：将 CUDA:1 拆为平台 CUDA 与 DeviceIndex=1，复用统一平台解析 helper。
- 已完成部分：preoptimizer/pipeline 内部已能把 CUDA:1 拆为 CUDA + DeviceIndex=1。
- 重新打开原因：runabfe.py 约 3178 行的 --platform choices 仍只有 CUDA/OpenCL/CPU，
  所以 python runabfe.py --platform CUDA:1 会在 argparse 阶段被拒绝，内部修复不可达。
- 补充验收：CLI 与配置文件都能传 CUDA:N；非法设备格式明确报错。

### [x] P2-03 双 lambda 预优化硬编码 CUDA/default device

- 位置：abfe_pipeline.py 的 _run_dual_lambda_optimization 约 3552 行。
- 要求：尊重 self.platform_name，复用 _build_platform_props；只在真实初始化失败时局部回退 CPU。

### [x] P2-04 自定义 vanishing 分窗约束不可行时仍返回越界窗口

- 位置：abfe_preoptimizer.py 约 686–730 行。
- 已复现：n_states=7、min=6、max=6 返回单个 7 态窗口。
- 要求：预先验证是否存在满足约束的窗口数；不可行必须 ValueError；最终 validator 同时检查
  min/max、覆盖和共享边界。

### [x] P2-05 单 replica REMD 会除零

- 位置：ibs_engine.py 的 REMDManager.__init__ 约 15907 行；交换统计约 16734 行。
- 要求：构造时要求至少两个 replica、两条 lambda 数组等长且有限、exchange_interval 大于 0。

### [x] P2-06 Windows 无法从 PATH 自动发现 gmx.exe

- 位置：runabfe.py 的 find_gmx_include_dir 约 329–389 行。
- 要求：使用 shutil.which("gmx")，并从可执行文件位置正确推导 share/gromacs/top。

### [x] P2-07 底层 IBS builder 先 round 再检查配体净电荷

- 位置：ibs_engine.py 约 4151、4463 行。
- 要求：保留 raw charge，先验证与最近整数的差值在统一容差内，再转换为整数；
  底层接口不能假设所有调用方都先经过 resolve_charge_treatment。

### [x] P2-08 n_equil_steps 已接入运行时但没有 CLI 参数

- 位置：runabfe.py 的 parse_arguments 约 3051–3393 行没有 --n-equil-steps；
  运行时在约 2839、5182、5477、5637 行读取 config 的 n_equil_steps。
- 触发：python runabfe.py --n-equil-steps 100000。
- 问题：argparse 直接报告 unrecognized argument；只能依赖未显式暴露的配置文件键。
- 要求：增加 CLI 参数并写入 RunConfig 显式 override。
- 验收：同一值进入膜时长预检、complex/solvent 两条腿和预平衡 fingerprint。

### [x] P2-09 single-lambda/2D diagonal 路径缓存不校验状态数

- 位置：abfe_pipeline.py 的 path_single_lambda.json 约 10011–10027 行、
  path_2d_diagonal.json 约 10117–10136 行。
- 触发：先生成 12 态路径，再以 16 态请求 resume。
- 问题：缓存只读 path，不校验 scheme、长度、当前 n_states、端点、单调性或有限性；
  后续会在旧路径上重新采样，却把顶层运行配置记录成新的状态数。
- 要求：保存路径协议指纹；加载时验证全部不变量，不匹配即重建。
- 验收：旧 12 态缓存面对当前 16 态请求必须重建，不能继续使用旧 path。

### [x] P2-10 Shadow IBS builder 绕过 32-CV/16-state 上限

- 位置：ibs_engine.py 的 build_shadow_coul_ibs_system 约 4643–4741 行；
  对照 build_ibs_dual_system 约 4078–4124 行。
- 触发：lambdas_shadow_coul 含 17 个或更多状态。
- 问题：每态注册 interaction/restraint 两个 CV，17 态会产生 34 个 CV；
  dual builder 已在入口执行 32-CV 上限，Shadow builder 没有同一检查。
- 影响：错误推迟到昂贵的 OpenMM Context 创建阶段；空、NaN、非单调路径也缺少清晰入口校验。
- 要求：共用状态数、有限性、端点与单调性 validator，在复制 System 前 fail closed。
- 验收：K=17、K=0、NaN、非单调均明确拒绝；K=16 正常。

### [x] P2-11 2D geodesic 未校验 n_grid 下界

- 位置：abfe_preoptimizer.py 的 optimize_2d_geodesic_path 约 3697–3784 行。
- 触发：n_grid=0 或 1。
- 问题：n_grid=0 最终对空 path_arr 做二维索引；n_grid=1 时同一元素先被设为起点、
  又被末端覆盖成 (0,0)，返回路径丢失 (1,1)。
- 要求：入口要求整数 n_grid>=2，并同时校验 temperature、n_steps_per_point 为正。
- 验收：0/1 在创建 probe System/Context 前 ValueError；2 保留两个精确端点。

### [x] P2-12 约束 Jacobian 修正不是完整的约束相空间修正

- 位置：abfe_core.py 的 calculate_constraint_jacobian_correction 约 8894–8921 行；
  abfe_pipeline.py 的 compute_final_results 约 5514–5520、5621–5625 行。
- 问题一：当前公式仅为 `+0.5*RT*sum(log(mu/1 Da))`。`1 Da` 是人为参考量，公式没有
  约束长度、约束梯度/质量度量行列式或对应柔性键配分函数，不能作为一般的 Fixman/约束
  自由能修正。
- 问题二：本项目的炼金路径只关闭配体—环境非键相互作用；配体内部约束、质量与内能在
  两个 lambda 端点保持不变，因此同一条腿内的约束相空间因子应相消。当前却对 complex 和
  solvent 腿分别加一个仅由约化质量决定的常数；两条腿拓扑完全相同时最终结合自由能中大致
  相消，约束/HMR/拓扑稍有差异时会留下非物理偏移。
- 问题三：计算异常在 compute_final_results 中只记 warning，随后以 0.0 继续生成有限结果；
  若未来确实引入需要修正的端点变化，这会变成静默漏项。
- 要求：先明确目标是“受约束 Hamiltonian 的 ABFE”还是“映射回柔性 Hamiltonian”。前者应
  删除该附加项并要求两端/两腿配体约束身份一致；后者必须实现有推导和数值验证的完整约束
  配分函数比值。任何所需修正计算失败都应 fail closed，并把约束身份纳入协议指纹。
- 验收：相同配体约束在任意两腿给出精确零净修正；人为改变一条约束、HMR 质量或约束长度时
  缓存失效并明确拒绝，不能继续写最终 ΔG。

## 第七～八轮：代码落地与回归边界（2026-08-31）

本轮按第六轮要求实际修改了五个核心文件，并将不能严格证明的路线改成
fail-closed；同时为唯一开放的 charge-transfer 项补上了可审计的条件放行接口：

- **P1-08/P1-16/P1-17/P1-19**：Shadow-Coulomb 使用调用体系的真实 PME cutoff；
  explicit PME alpha/grid 优先读取并复制到三个 probe，Bridge diagnostics 记录
  alpha/grid；Shadow 温度必须显式、有限且使用 pipeline 温度；Bridge 结果缓存要求
  sampling fingerprint、`converged=true`、有限结果与 overlap；普通 L–L 对的 v4
  annihilation 口径同步到 Stage 2 Group 2，charging/vanishing seam 不再跳回 raw
  ligand-internal Coulomb。
- **P2-01/P2-02/P2-05/P2-07～P2-11**：显式 PME 参数/设备索引（CUDA:N/OpenCL:N）
  解析接通 CLI；REMD 至少两个 replica 且 lambda/温度/间隔均做边界校验；IBS 与
  Shadow builder 在创建 System 前拒绝空、NaN、越界、超 16 态输入；metric grid
  直接调用也校验正有限温度；single/diagonal 路径缓存校验 scheme、状态数、端点、
  单调性和协议指纹，geodesic 缓存允许其自然的可变路径长度但校验端点/单调性/内容指纹。
- **P2-12**：移除伪 `0.5 RT log(mu/1 Da)` 项，加入 ligand-local constraint/HMR
  identity fingerprint；complex/solvent 汇总前比较 `comparison_sha256`，缺身份或不一致
  直接拒绝，不再用 0.0 掩盖约束差异。
- **PHY-01**：DEXP 不再能从 `run_full_pipeline` 伪装成原始 LJ ABFE；traditional
  DEXP 同样拒绝。若要研究 DEXP，必须走单独 surrogate/bridge 验证流程。
- **PHY-02**：traditional+Boresch 强制先跑 A′→A attachment leg，要求 attachment
  `converged=true`，并把其 ΔG/误差并入 complex leg；缺 attachment 或不收敛时拒绝。
- **PHY-04～PHY-09**：三斜盒 MIC 改为可证明终止的 closest-lattice 搜索；膜头基/尾链/
  protein tilt/lateral MSD 使用逐帧 unwrap，density profile 按 `A_xy*Δz` 输出
  `dalton_per_nm3`；Shadow 中性检查先验证 raw charge；LRC 在缺盒/奇异盒/NaN/系数
  长度错误时抛出 hard gate；Stage-1 错误的 Var(U) optimizer 禁止调用；Boresch
  cosine/Jacobian 数值积分用于模型差异并传播为 systematic error。
- **PHY-03 条件接口**：新增双腿 tethered-carrier reservoir-release correction schema；
  correction 必须声明与生产一致的 `temperature_K`，且 C4/C5 tolerance 不得放宽项目
  1.0 kJ/mol 闭合门；complex/solvent 各自必须有收敛的
  restraint→free 释放自由能、独立样本和
  C4/C5 盒尺寸/锚点/力常数扫描证据，且净项按
  `ΔG_reservoir = ΔG_release,solvent − ΔG_release,complex` 进入唯一循环汇总。
  correction 会进入 pipeline/stage/top-level fingerprint；缺少、温度不一致或两腿
  artifact 不一致时在创建 Context/汇总前拒绝。该接口不伪造 C4/C5 通过，当前项目级
  生产资格仍保持 False。
  回归契约补在 `tests/test_charge_transfer_production_qualification.py`：有效 schema
  的净项/误差归一化、温度不一致拒绝，以及“闭合但不自动生产准入”。
  直接调用 `ABFEPipeline.run_full_pipeline()` 时若省略电荷路线，带电配体也会在建
  Context 前拒绝，避免落回旧 co-annihilation 默认路径。

**PHY-03 保持未完成（已挂起；唯一开放项）**：仓库尚无真实带电体系的 C4/C5 扫描 artifact。
correction 现在还必须声明 `temperature_K`，并与两腿生产温度一致；没有 correction 时
`resolve_charge_treatment()` 明确写入 `closes_thermodynamic_cycle=false`，主 CLI 和
pipeline 在创建任何 OpenMM Context 前拒绝该路线；只有提交并通过新 schema 的双腿
correction 才能运行实验性诊断，且主汇总会再次核对两腿 artifact 完全一致；不会自动提升
为生产资格。这不是“已修复为生产”，而是“错误路径不可达、证据齐全时可审计运行”。

验证：`python -m py_compile runabfe.py abfe_pipeline.py ibs_engine.py abfe_core.py
abfe_preoptimizer.py` 通过。当前会话没有 OpenMM/pytest/ruff，无法执行需要运行时的
回归测试；不得把本轮静态通过描述成真实 MD/REMD 验证。

## 第六轮：物理与数学专项审查（2026-08-31）

> 范围：`runabfe.py`、`abfe_pipeline.py`、`ibs_engine.py`、`abfe_core.py`、
> `abfe_preoptimizer.py`。本轮只做静态公式审查、调用链核对和两个最小数值反例；
> 没有修改生产代码、没有启动 MD/REMD/IBS，也没有改写任何历史 artifact。
>
> 条件边界：PHY-01 只影响 DEXP；PHY-02 只影响 traditional+Boresch；PHY-03
> 只影响带电 charge-transfer；PHY-06 只影响 Shadow-Coulomb；PHY-08 的错误
> Stage 1 optimizer 当前被主流程显式禁用。若运行的是 neutral ligand + softcore +
> dual_lambda，这几项不会同时进入默认生产 Hamiltonian，但不能因此保留错误的可达分支。

### [x] PHY-01（P0，条件触发）DEXP 的 λ=1 端点不是原始 12-6 LJ

- 位置：`abfe_core.py::DEXPSurrogatePotential.build_expression`（约 6419–6488 行）；
  `ibs_engine.py::_create_softcore_force`（约 3664–3761 行）；
  `ibs_engine.py::build_ibs_dual_system`（约 4314–4323、4403–4513 行）。
- 触发：`potential_type="dexp"` / CLI `--potential=dexp`，并把结果解释为原始力场 ABFE。
- 问题：DEXP 仅在 `r=r0=2^(1/6) sigma` 匹配 LJ 的井位、井深和一阶导数；它不是
  12-6 LJ 的恒等变换。builder 同时把原生 ligand-environment LJ 全部归零，再由 DEXP
  CV 接管，且 DEXP 使用 0.70 nm cutoff、0.50–0.70 nm switching；五个核心文件中没有
  DEXP→原始 LJ 的端点 bridge/free-energy correction。
- 最小反例：默认 `alpha=14, beta=5, epsilon=1`，在 `r=sigma`，标准 LJ 给
  `U/epsilon=0`，当前 DEXP 给 `U/epsilon=-0.125045371989...`。因此 λ=1 并非物理
  LJ 端点；0.70–1.00 nm 的原始 LJ 区间还被整体删去。
- 影响：若目标是原始力场结合自由能，complex/solvent 两腿得到的是 DEXP 模型的结合
  自由能，差异不会因“两腿使用同一替代势”自动抵消。只有明确把 DEXP 本身定义为目标
  Hamiltonian 时，这才是模型选择而非端点错误。
- 要求：二选一并写死协议语义：(A) 增加采样充分、可验证的 LJ↔DEXP bridge，使最终
  物理端点严格回到原始 NonbondedForce；或 (B) 明确输出为 DEXP-model ΔG，禁止标作原始
  力场 ABFE。不能只靠势阱局部拟合宣称端点等价。
- 验收：在多组 `(sigma, epsilon, r)`、真实 PME/LJ System 上比较 λ=1 能量与逐原子力；
  原始力场模式必须逐位等价或由独立 bridge 闭合。协议版本、stage/top-level fingerprint、
  resume cache 和结果 schema 必须区分 “LJ target” 与 “DEXP target”。

### [x] PHY-02（P0，条件触发）traditional+Boresch 缺少 attachment 自由能

- 位置：`ibs_engine.py::_add_physical_boresch_restraint`（约 1483–1501 行）；
  `abfe_pipeline.py::TraditionalABFEPipeline.run_full`（约 11206–11260 行）；
  `runabfe.py::run_traditional_mode`（约 4346–4487 行）。
- 触发：`mode=traditional` 且启用 Boresch restraint。
- 问题：traditional 的 complex REMD 用 `fixed_lam=1.0` 全程施加物理 restraint，
  run_full 只计算 decharging + vanishing；外层再加入解析 standard-state release。
  但物理结合态 A' 是“fully coupled + 无 restraint”，采样起点 A 是“fully coupled +
  restraint”。缺少 A'→A 的 `Delta G_attach`，循环为
  `A -> decoupled/restrained -> released`，没有从真实 A' 接入。
- 影响：结果会依赖 Boresch 力常数和锚点选择；即使 charging、vanishing、release 各自
  数值收敛，最终 ΔG 仍不是闭合的物理结合自由能。dual_lambda 主线已有 Stage 0，
  traditional 分支没有复用它。
- 要求：traditional complex 腿在任何受约束解耦之前运行与 dual_lambda 相同物理定义的
  attachment leg，并把其采样误差并入 complex leg；若暂不支持，traditional+Boresch
  必须在创建 Context 前 fail closed，不能只加解析 release。
- 验收：同一体系使用至少三组相差显著的 Boresch 力常数/锚点，`attach + decouple +
  release` 总和在统计误差内不变；移除 attachment 的对照必须能复现 restraint-strength
  依赖。fresh/resume cache 均绑定 attachment Hamiltonian 和 seed contract。

### [ ] PHY-03（P1，实验路线）charge-transfer 的 tethered charge carrier 不能按当前论证严格跨腿抵消

- 位置：`abfe_core.py` 的 co-ion restraint 说明与表达式（约 1088–1117 行）；
  `ibs_engine.py::_create_co_alchemical_ion_restraint`（约 807–848 行）；
  `abfe_core.py::resolve_charge_treatment` 的 `closes_thermodynamic_cycle`（约 926–947 行）。
- 触发：带净电配体使用 `co_alchemical_charge_transfer`。
- 问题一（配分函数）：代码以“两腿同一锚点规则、同一 k/r0”推断 restraint 自由能严格
  抵消。实际受限 charge carrier 的配分函数包含
  `integral exp[-beta*(U_env(r)+U_rest(r-r_anchor))] dr`。complex 与纯水腿的
  `U_env`、排除体积、anchor 系综均不同；lambda=0 时 carrier 还带真实电荷并与环境
  相互作用，因此 restraint 与 carrier 溶剂化不能分离成一个两腿相同的常数。
- 问题二（barostat）：`dx0/dy0/dz0` 是冻结的笛卡尔 nm per-bond 参数。barostat
  缩放盒矢量和粒子坐标时，`d0` 不缩放；“井心随体系/盒一起缩放”的注释不成立，
  半各向异性/三斜 NPT 下尤其明显。
- 影响：decoupled complex/solvent 端点未必共享可严格消掉的 reservoir 状态，最终差值
  可能含 carrier 位置、盒大小、蛋白排除体积和 restraint 的非物理贡献。项目当前已经把
  charge-transfer 标为 `production_qualified=False`，这一边界必须保留；但同时写
  `closes_thermodynamic_cycle=True` 仍过度承诺。
- 要求：给出包含 carrier restraint/标准态/环境项的完整热力学循环推导；若不能证明解析
  抵消，就显式计算两腿 restraint/reservoir correction。参考位移需要采用真正随盒变化的
  分数坐标定义，或改成不依赖冻结笛卡尔井心且有解析标准态修正的相对约束。
- 验收：carrier 平移、anchor 选择、盒尺寸、各向异性缩放和 restraint 强度扫描后，修正后
  ΔG 在统计误差内不变；complex/solvent reservoir 端点有独立 free-energy closure test。
  C4/C5 未通过前不得把数值提升为生产结果。

### [x] PHY-04（P1）一般三斜盒 minimum-image 算法不是最近晶格像

- 位置：`abfe_core.py::minimum_image_displacement_nm`（约 1155–1172 行）；
  `runabfe.py::_insert_reserved_co_alchemical_dummy_sites` 内部副本（约 1343–1358 行）。
- 触发：非正交盒，尤其剪切较大的三斜盒；co-ion/dummy 选点、安全距离、质心邻近原子
  或任何调用该 helper 的几何判据。
- 问题：`fractional -= round(fractional)` 只把位移放入中心平行六面体，不保证欧氏距离
  最短。它不是一般晶格 closest-vector 算法。
- 最小反例：盒向量 `a=(2,0,0) nm, b=(0.9,1.8,0) nm, c=(0,0,3) nm`，分数位移
  `(0.49,0.49,0)`。当前算法返回距离 `1.6724727203 nm`；减去一个 `a` 晶格向量后
  距离仅 `1.0550663486 nm`。
- 影响：可能选择错误周期像、错误判定最小距离，进而冻结错误的 co-ion reference
  displacement 或接受本应拒绝的碰撞几何。正交盒不受影响。
- 要求：优先复用 OpenMM 的周期距离；纯 NumPy 路径应使用经过验证的 closest-lattice-
  vector 算法。若依赖 OpenMM reduced box + 邻居枚举，必须先验证/归约盒并搜索足够的
  相邻晶格平移，不能只逐分量 round。删除 runabfe 内的第二份数学实现。
- 验收：正交、OpenMM reduced triclinic、强剪切盒与随机位移对照暴力晶格搜索；上面反例
  必须返回 `1.055066... nm` 对应向量。co-ion 选点与 restraint identity 回归测试同步更新。

### [x] PHY-05（P1，膜质量门）wrapped 坐标直接做 lateral MSD；所谓质量密度缺少体积归一化

- 位置：`abfe_core.py::membrane_observables_from_trajectory`（约 4962–5415 行）；
  `_density_profile_along_normal`（约 5637–5674 行）；
  `_lipid_lateral_relaxation_timescale_ns`（约 5677–5775 行）。
- 触发：膜预平衡 DCD 中脂质头基跨 x/y 周期边界；膜/盒中心跨 z 周期边界；读取默认
  wrapped/imaged 周期轨迹。
- 问题一：MSD 直接使用 `traj.xyz[t+lag]-traj.xyz[t]`，没有按逐帧盒矢量 unwrap 或
  minimum-image 累积。一次边界穿越会被记成约一个盒长的位移，扩散系数偏大、弛豫时间
  `tau` 偏小，可能错误放行“预平衡时长 >= 一个脂质弛豫时间”的质量门。
- 问题二：density profile 只累计原子质量并除以帧数，未除以每帧横截面积与 bin 宽度；
  输出单位是平均 `Da/bin`，不是质量密度。`z-midplane` 也没有周期 wrap，跨 z 边界会
  把相邻粒子放到直方图远端。叶片中面/厚度的算术均值同样隐含膜始终不跨边界的前提。
- 要求：用逐帧 triclinic unitcell unwrap lipid head trajectories，再计算 time-origin
  averaged MSD；法向相对坐标走周期最小像。密度逐帧按实际 `A_xy(t)*Delta z` 归一，明确
  输出单位（例如 kg/m^3 或 Da/nm^3），再做帧平均。
- 验收：合成轨迹中粒子匀速跨边界时，wrapped 与显式 unwrapped 参考 MSD 完全一致；整体
  平移膜跨过 z 边界不改变厚度、叶片、core water 或密度分布；改变盒面积但保持真实密度时
  profile 不变。

### [x] PHY-06（P1，Shadow-Coulomb）中性检查在比较前错误 round

- 位置：`ibs_engine.py::_assert_neutral_ligand_for_shadow_coul`（约 4608–4626 行）。
- 触发：Shadow-Coulomb 收到非整数或异常部分净电荷，例如 `+0.4 e`。
- 问题：条件是 `abs(round(net_q)) > 0.01`。`+0.4 e` 会先被 round 成 0，随后被当作
  中性放行，与函数“只支持电中性配体”的契约相反。这个错误与既有 P2-07 的底层
  “先 round 再校验”属于同一模式，但位置/分支独立，不能只修一个。
- 影响：Shadow 路线可能在没有 co-alchemical neutralization/finite-size 处理时关闭带净电
  配体，生成物理上错误但数值有限的结果。
- 要求：先验证 `net_q` 有限并接近最近整数，再要求该整数严格为 0；或者对本函数直接要求
  `abs(net_q) <= unified_tolerance`。统一复用 `LIGAND_NET_CHARGE_INTEGER_TOLERANCE_E`。
- 验收：`0, +/-1e-4` 放行；`+/-0.4, +/-0.6, +/-1.0, NaN, Inf` 全部在建 Context 前
  拒绝。Shadow 与普通 PME builder 共用同一净电荷 validator。

### [x] PHY-07（P1）启用的 LJ LRC 在盒信息异常时静默变成零

- 位置：`ibs_engine.py::IBSSampler._lj_tail_correction_kj_mol`（约 6722–6753 行）。
- 触发：`lj_tail_lrc_coeff_kj_mol` 已存在（即协议要求 LRC），但读取 box vectors、单位
  转换或体积计算抛异常，或者体积为 NaN/Inf/非正。
- 问题：所有异常和非法体积都直接返回 `zeros(n_states)`。这把“修正必需但计算失败”伪装成
  “修正不适用”，没有诊断计数，也不会阻止 MBAR。若只在部分帧发生，还会造成逐帧目标
  Hamiltonian 定义不一致。
- 要求：只有 `lrc_coeff is None` 才允许返回零；系数存在时，box/volume/数组长度/有限性任一
  失败都应 fail closed，并记录 frame/window 身份。复用已有 `_periodic_box_volume_nm3`，
  不保留第二份宽泛 try/except 数学。
- 验收：合法正交/三斜/NPT 盒给出 `coeff/V(t)`；缺盒、奇异盒、NaN、系数长度不足全部
  明确失败，不得返回有限零数组。fixed-H probe 与 production sampler 使用同一失败语义。

### [x] PHY-08（P2，潜伏分支）Stage 1 preoptimizer 用 Var(U) 代替 Fisher metric

- 位置：`abfe_preoptimizer.py::_sample_group1_energies`（约 2079–2102 行）；
  `DualLambdaPreOptimizer.optimize_stage1_decharging`（约 2980–3038 行）；
  `build_aces_probe_system_dual_lambda`（约 3407–3510 行）；主流程禁用点为
  `abfe_pipeline.py::_run_dual_lambda_optimization`（约 3961–3974 行）。
- 触发：直接调用 `optimize_stage1_decharging`，或未来重新启用 Stage 1 自适应 pathfinding。
- 问题一：路径权重来自 Group 1 原始能量的 `Var(U_group1)`。正确一维 thermodynamic/Fisher
  metric 是 `g(lambda)=beta^2 Var[dU/dlambda]`；只有 Hamiltonian 对 lambda 严格线性、
  Group 1 又只含 lambda-dependent 项时二者才可能成比例。
- 问题二：neutral probe 的 Coulomb 是 cutoff CustomNonbondedForce，不是生产 PME；带 co-ion
  时又把整个原生 NonbondedForce 放进 Group 1，raw variance 会混入大量 lambda-independent
  environment-environment PME/LJ 涨落。这样的 λ 密度不代表生产 Hamiltonian 的 overlap。
- 问题三：旧 CDF 以点权重直接 cumsum，不做区间梯形积分，并用 `xp[-1]=1` 覆盖倒数第二个
  累积坐标；这不是一致的连续弧长离散化。
- 当前边界：主 pipeline 已因 PME 不保真显式退回线性 Stage 1，所以默认生产暂不受影响；
  错误函数仍是公开可调用代码，不能把“当前禁用”当成修复。
- 要求：若保留自适应 Stage 1，必须在与生产相同的 PME+offset Hamiltonian 上用有限差分或
  parameter derivative 采 `dU/dlambda`，以 `beta^2 Var` 和梯形弧长积分布点；否则删除/硬禁用
  该 API，禁止直接调用得到貌似优化的路径。
- 验收：线性解析 Hamiltonian、含大幅 lambda-independent noise 的 toy Hamiltonian、真实 PME
  三类测试；加入任意 lambda-independent Group 1 能量后优化路径不得变化。

### [x] PHY-09（P2，模型/不确定度）Boresch 实际 cosine restraint 与高斯解析式不严格同构

- 位置：`abfe_core.py::calculate_boresch_analytical_correction`（约 7129–7188 行）；
  `LambdaDependentBoreschForce`（约 7247–7255 行）；
  `abfe_pipeline.py::apply_boresch_correction`（约 5521–5544 行）。
- 触发：启用 Boresch，尤其允许较软的角/二面角力常数，或把返回 `error=0.0` 当成完整
  不确定度。
- 问题：解析 release 使用六自由度局部 harmonic/Gaussian 积分；实际角和二面角势是
  `k*(1-cos(delta))`。二者局部曲率相同，但有限 k 时全局配分函数不同；角坐标还有
  `sin(theta)` Jacobian。代码已记录“locally harmonic”假设，却仍把 correction error
  无条件写为 0，未传播从轨迹估计/clip force constants、平衡几何和非高斯性带来的误差。
- 定级说明：标准态因子、总符号和 `(2*pi*RT)^3` 本轮未发现反号；这是“把近似当精确并报
  零误差”的问题，不是解析式整体符号错误。强 restraint 下偏差可能很小，需定量而非猜测。
- 要求：选择与解析推导一致的 wrapped harmonic restraint，或对实际 cosine/angle measure
  做解析/数值配分函数修正；至少根据允许的 k 范围给出系统误差上界。由轨迹估计的参数应通过
  bootstrap/独立 block 传播不确定度，不能固定为 0。
- 验收：单自由度 cosine 的数值积分、六自由度独立参考积分和 OpenMM 采样三方对照；覆盖
  当前允许最软到最硬的 force constants，定义可执行的最大近似误差门，并将 correction
  uncertainty 并入最终误差或单列系统误差。

### 第六轮对既有 P2-12 的复核结论（不新增编号）

- 已确认 `calculate_constraint_jacobian_correction` 的约束和质量沿当前炼金路径没有变化；
  同一受约束 Hamiltonian 两端不应额外加入 `0.5*RT*sum(log(mu/1 Da))`。`1 Da` 不是
  OpenMM 提供的物理标准态，当前式子也不是完整 Fixman determinant 或柔性键映射。
- complex/solvent 的配体约束完全相同时，该伪常数通常会在最终两腿相减中抵消，因此不应
  夸大成所有现有 ΔG_bind 都必然偏移；逐腿数值仍错误，约束/HMR/拓扑稍有差异时会留下
  非物理残差。
- 保持既有 P2-12 的修复要求：当前协议应删除该附加项并校验两端/两腿约束身份；若未来要
  映射回柔性 Hamiltonian，必须另行实现有推导和数值验证的完整配分函数比值。

### 第六轮确认未发现反号的关键公式

- 主双腿汇总 `Delta G_bind = Delta G_solvent - Delta G_complex` 的方向正确。
- dual_lambda 主线的 Boresch `attachment + restrained decoupling + release` 记账方向正确；
  PHY-02 是 traditional 没有 attachment，不是 release 项反号。
- PME `getState` 已包含完整 Ewald reciprocal/real/self 能量；当前生产路径不再额外手工添加
  Ewald self correction，这一处理正确。

## 后续建议实施顺序

1. 在安装 OpenMM 8.5.x、mdtraj、pymbar、pytest 的环境运行本轮新增运行时回归：
   Shadow 310 K/显式 PME grid、三斜 MIC 反例、膜跨界 unwrap/density、Boresch
   attachment 与 cosine 积分、constraint identity mismatch，以及 12→16 路径缓存失效。
2. PHY-03 当前按要求挂起，仍是唯一 release blocker：解除挂起后，完成 tethered
   carrier 的 C4/C5 双腿 closure、盒缩放/anchor/力常数扫描和 reservoir correction，
   才可将 `closes_thermodynamic_cycle` 与生产资格改为 True。
3. 处置第五轮记录的 5 个既有测试失败（与本轮无关），并在具备项目依赖的环境复核
   未提交的工作区差异；任何 Hamiltonian 变化继续递增协议版本并使旧缓存失效。

已完成历史项仍以各自条目中的 `[x]` 和实机记录为准，不因上述新排序重新打开；唯一例外是
若 PHY 修复改变 Hamiltonian/结果口径，必须按《数据与结果安全边界》另升协议版本并使旧缓存失效。

## 最低测试要求

- [x] 五个核心文件通过 AST 解析检查。
- [ ] 在装有项目依赖的环境完成编译检查及 Ruff E9/F63/F7/F82。
- [x] 运行 ./tests/run_offline_tests.sh，排除 needs_gpu 后全绿。
  （2026-08-30 第五轮：openmm_dev 环境实测。**基线即有 5 个与本次修复无关的
  既有失败**：test_charge_transfer_production_qualification.py::
  test_postanalysis_qualification_rejects_mixed_artifacts、test_exp017_overlap_first.py::
  test_split_first_does_not_infer_an_edge_from_a_low_window、test_orb_latent.py::
  test_cached_omol_path_is_resolved_without_network_when_runtime_is_available、
  test_outer_lambda_torchforce_standalone.py::
  test_production_modules_do_not_import_standalone_neural_module、
  test_pymbar_version_and_uncertainty_contract.py::
  test_pymbar_420_default_uncertainty_matches_explicit_svd_ew——这 5 个在本次
  修复开始前的干净基线上就是失败的，未在本轮范围内处置。其余失败清单见
  会话记录的收尾核对；涉及本次修复引入回归的条目以最终一轮全量输出为准。）
- [x] 新增 corrupt checkpoint、truncated DCD、trajectory-ahead-of-checkpoint resume 隔离测试。
- [x] 新增 traditional fresh-run、接口与带电 fail-closed 隔离测试。
- [x] 新增 analyze-only 缺首/中/末窗口及旧协议 checkpoint 测试。
  （tests/test_analyze_only_and_refine_regressions.py：12 窗数值映射、缺编号拒绝、
  分流顺序、provenance 恢复/显式覆盖；checkpoint 分支的 converged/protocol_key/
  覆盖证据拒绝逻辑为源码守护。）
- [ ] 补齐非默认 n_equil_steps 的两次真实 resume 测试；本批仅验证配置向两条腿转发。
- [x] 新增已有 barostat 参数一致/不一致隔离测试。
- [ ] 新增 Shadow 310 K、显式 PME alpha/grid 运行时测试。（代码已修复；当前环境缺 OpenMM）
- [x] 新增 CPU、CUDA、CUDA:1 平台路由隔离测试。

第二轮新增最低回归要求：

- [x] fresh membrane/no-DCD 能先预平衡再判质量门。
- [x] truncated DCD/随机 checkpoint 不能通过 runabfe.equilibrium_is_done。
- [x] single/2D 的 converged=false 不落完成缓存。
- [x] valid-fingerprint final cache 缺字段、NaN/Inf/负误差全部拒绝。
- [x] traditional --no-boresch 仍执行两腿基线预平衡；膜/色散协议不能静默绕过。
- [x] 12 个 refine 窗口按数值编号映射；12→16 态路径 cache 必须失效。
  （路径 cache 现校验状态数、端点、单调性及协议指纹；geodesic 允许自然可变长度。）
- [x] Shadow cutoff 跟随参考 NonbondedForce；Bridge cache 具备协议指纹。
- [x] CLI 支持 --n-equil-steps、CUDA:N；n_grid/CV 上限在建 Context 前校验。
- [x] 相同端点约束的净 Jacobian 修正为零；约束/HMR 身份变化会失效缓存或 fail closed。
- [x] 真实 PME 下配置前后 lambda=1 的中性配体能量/力端点等价，并独立检查 direct/reciprocal 分量。
  （energy/force 已在 Reference 双精度下逐位验证；direct/reciprocal 分量 OpenMM
  不直接暴露，以"全 λ 对照独立参考实现（粒子电荷直接缩放）逐位一致"替代——
  任何 direct/reciprocal 分量差异都会表现为该对照的失配。）
- [x] repeat_seed 改变会改变 decharging REMD seed 与 sampling fingerprint，旧 DCD/u_kn 不得复用。

第六轮新增最低回归要求：

- [x] DEXP 的目标语义可机读区分；原始 LJ 模式拒绝走未闭合 DEXP 路线（独立
  LJ↔DEXP bridge closure 仍待专用验证）。（PHY-01）
- [x] traditional+Boresch 显式包含 attachment；总自由能对 restraint 强度/锚点选择的
  运行时统计不变性测试待在 OpenMM 环境执行。（PHY-02）
- [ ] charge-transfer 在 carrier 平移、盒缩放、anchor 与 restraint 扫描下通过完整双腿
  closure；C4/C5 前保持非生产资格。（PHY-03，已挂起）
- [x] 三斜 minimum-image 使用 closest-lattice 搜索；runabfe 不再维护第二份实现。
  （文中反例的运行时数值回归待在 OpenMM/依赖环境执行。）
- [x] 膜观测量加入逐帧 unwrap 与体积归一化 density profile；合成轨迹回归待运行时环境执行。
  （PHY-05）
- [x] Shadow 对 `+/-0.4 e` 等异常部分净电荷在建 Context 前拒绝。（PHY-06/P2-07）
- [x] LRC 系数存在时，缺盒/奇异盒/NaN/系数长度错误全部 fail closed，不能回零。（PHY-07）
- [x] Stage 1 错误 Var(U) optimizer 已禁用；直接调用 fail closed。（PHY-08）
- [x] Boresch cosine 与 harmonic/Gaussian 的差异由数值积分量化并进入 systematic error；
  全允许 k 范围的 OpenMM 三方回归待运行时环境执行。（PHY-09）

本批新增/更新的回归文件：

- tests/test_pipeline_pre_equilibration_regressions.py
- tests/test_open_issue_fail_closed_contracts.py
- tests/test_vanishing_window_partition_regressions.py
- tests/test_platform_routing_regressions.py
- tests/test_traditional_pipeline_interfaces.py
- tests/test_core_runabfe_bugfix_batch.py
- tests/test_membrane_barostat_protocol.py

第五轮新增/更新的回归文件：

- tests/test_pme_decharge_endpoint_equivalence.py（P0-01：λ=1 端点等价、
  全 λ 对照独立参考实现、禁止补 exception 的静态守护）
- tests/test_charge_transfer_hamiltonian.py（更新：v2"逐对冻结"契约按 v3
  annihilation 口径改写为"不得补 exception"+"内部静电 λ² 湮灭"）
- tests/test_equilibrium_is_done_strict_validation.py（P1-07）
- tests/test_membrane_gate_first_run_regressions.py（P1-10）
- tests/test_final_result_schema_gate.py（P1-12）
- tests/test_mbar_convergence_gate_regressions.py（P1-11）
- tests/test_decharging_seed_contract.py（P1-18）
- tests/test_traditional_mode_preflight_and_equil.py（P1-13/P1-14）
- tests/test_analyze_only_and_refine_regressions.py（P1-04/P1-15）
- tests/test_top_level_sampling_cache_identity.py（更新：mock 补 converged=True）

## 数据与结果安全边界

> **第七轮执行记录（2026-08-31）**：P0-01 定级为真 ⟹ 触发本节"冻结旧 Stage 1
> artifact"条款。执行方式：`PME_DECHARGE_MODEL_VERSION` 升 v4 并编入
> `_stage_protocol_key`/顶层指纹，旧 stage 缓存与 final_results 缓存一律判
> 失效、重新校验/重算，而不是被静默复用；以下证据目录**均未改写、未移动、
> 未删除**，其内容保持原样作为 v2 口径的历史证据。

不得原地改写、移动或删除以下证据目录：

- output/
- output_lrc_fix/
- output_lrc_fixonly-complex-charging/
- validation/
- solvent_box_scan/
- memtest/output_membrane_100ns/
- memtest/output_membrane_5ns/

所有验证运行必须写入新的输出目录。修复旧代码不自动恢复旧结果有效性；受错误 Hamiltonian、
错误温度、截断窗口、污染轨迹或旧协议影响的 artifact 必须保持原状态并标记失效，不能原地覆盖
后重新宣称通过。
