# 项目记忆导出（交接用）

导出时间：2026-08-07。来源：本地长期记忆整理，只保留 **ABFE/IBS 方法本身的技术记录**——
项目状态、已修的坑、未完成的部分。

⚠️ 这些是历史时点快照，不是实时状态——具体行号/数值请对照当前源码核实，不要直接当结论引用。

---

## 项目背景

`ABFE_IBS/Atenolol-rank11` 实现的是 **Integrated Boltzmann Sampling (IBS)**
（Lin/Xia/Zhang/Gao, JCTC 2026, doi 10.1021/acs.jctc.5c01240）的一个
"converge-then-freeze、可微分"变体，用于计算 Atenolol 的**绝对**结合自由能
（Boresch restraint + DEXP/MACE 替代势），固定 300K，仅 λ 坐标（无 tempering/β 积分维度）。
CBFE/ACES（另一个不同方法，York/Ganguly 那篇）是误发的参考资料，**不是**本仓库实现的方法。

## IBS 核心设计纠错（2026-07-17）

最致命的设计错误：曾经把"相邻态 BAR/MBAR overlap"当成 IBS 路径正确性的仲裁标准，
这是**错误前提**——IBS 是把冻结积分混合势重加权到每个目标态（不是相邻→相邻），一个陡峭的
ΔF 会被权重 `a_k=q_k·e^{βf_k}` 精确吸收。相邻 overlap 是 pairwise BAR（f_k 标定器）的性质，
不是 Boltzmann/IBS 恒等式的性质。已纠正的设计：
- 采样网格（冻结的 MD 混合）与目标网格（自由重加权的任意查询网格）分离。
- 唯一收敛仲裁 = **f_k 占据自洽性**（⟨r_k⟩→π_k）+ 冻结批次概率评估，不是相邻 overlap。
- 生产估计量 = 单参考重要性重加权 MBAR，不是 TMBAR（TMBAR 只用于合并时变 f_k）。
- 五个不变量：不可变系综身份指纹、显式 a_k 定义、任意目标可评估性、块自举不确定度、
  不可变分析产物。

## f_k 符号（已作废的教训，别重犯）

2026-07-20 v21/v22 曾经给 f_k 加了个负号"修复"，**这个负号本身是错的，v27 已撤销**。
正确恒等式：`exp(β·f_k)·Z_k = exp[β·(f_k-F_k)]`，f_k 应该跟踪**物理**自由能，不是它的负数。
**永远不要再往 `_solve_tmbar_and_recenter`/`estimate_f_k_from_pilot_ti` 加负号。**
当前 `IBS_BIAS_PROTOCOL_VERSION = 29`。

## TMBAR 在线学习替换 TA 估计器（2026-07-19）

vanishing window 0 曾报出约 300+ kJ/mol 的虚假 state0↔state5 差异（真实 TI 交叉验证只有
~16.7 kJ/mol）——根因是原来 naive 时间平均在线学习估计器的数值病态，不是物理碰撞。
已替换成论文 eq.15 的 TMBAR（复用已有 `GlobalMBARAnalyzer.solve_stage_integrated`）。
`IBS_BIAS_PROTOCOL_VERSION` 18→19。当时代码层已改完但**未在 GPU 上验证**，追踪为
`VAL-GPU-007`。

## IBS 收敛门几轮演化

- **局部滑动窗口 MBAR 松门**（2026-07-24）：整个 f_k 收敛门换成局部滑动窗口 MBAR，
  只比较**相邻差值** `max_k|Δf_{k,k+1}^frozen − ΔF_{k,k+1}^MBAR| < 10 kJ/mol (~4kT)`。
  移除了累积占据 LSE 门、连续通过计数、50k→150k→300k 冻结验证阶梯、best-effort 接受、
  四联 ESS/覆盖门。
- **ESS 门重设计**（2026-07-26 v2 → 2026-07-29 v3）：v2 修复了 WCA 防护壳测量误差
  （`converged` = 三项正交证据：min_overlap/min_decorrelated_samples/
  max_endpoint_uncertainty_kJ_mol）。**v3 把 occupancy 完全退役为诊断项，不再是门**——
  这条已经踩过两次坑（测试钉着 v2 成员表而代码已是 v3），**永远不要把 occupancy 加回
  `converged` 来让测试变绿**。

## Boresch restraint 上游根因（2026-07-29）

约 20 天的估计量/收敛门工作背后，一直有个更上游的缺陷：Boresch restraint 的平衡值本身
放错了位置——`boresch_simple.json` 里 `equilibrium_values` 与
`diagnostics.fluctuation_distribution` 本应来自同一数组，但符号被反了
（phiA0/phiB0/phiC0 取反）；50k 步 Boresch 重平衡把这个 484kJ/mol=194kT 的应变松弛掉后，
提交的又是**最后一帧**当新平衡值，导致 committed 值偏离真实无限制系综模式 5.29σ。
自检本身是循环论证（拿新提交值对比当前受限坐标）。修复后：ΔG_bind = **−5.54 ± 0.60 kcal/mol**
（参考值 −6.279±0.457，0.98σ 一致）。**在此之前的 −2.121/−3.460/−3.4797 等候选值和
attachment 5.601±0.223 都是符号 bug 期间的值，不得再引用。**

`result.txt`（旧方法参考值）**只有 total 可比，分项不可比**——IBS 的分项拆法本就与参考
方法不同。

## DEXP 合并进核心（2026-07-29）

`dexp_NEW.py` 已合并进 `abfe_core.py` 并删除，退役的 Orb 拟合逻辑搬进 `dexp_退役.py`。
**DEXP 只是替代 LJ 的解析函数形式，本身不带 cutoff、无学习部分**：
- 唯一可配置面是 `alpha_vdw`/`beta_vdw`（当前 (14,5)，且该值仍未最终定案——
  (12,6) vs (14,5) 都没能解决 VAL→SER/N H-bond 重排问题）。
- `cutoff_distance`/`switch_width` 归属软化力壳（`DEXP_VDW_CUTOFF_NM=0.70`/
  `DEXP_VDW_SWITCH_WIDTH_NM=0.20`），`sigma_elec` 归属 Gaussian-Coulomb 静电
  （`GAUSS_COUL_SIGMA_NM=0.10`，`GAUSS_COUL_CUTOFF_NM` 必须保持 0.70）。
- DEXP 目前是**休眠路径**，生产用 `potential=softcore`，与 −5.54 kcal/mol 基线无关。

## 膜/带电配体专项（MEM-系列，2026-08-04 起，主线：MEM-00c→B3→B4→B5→C1）

- **MEM-00c**（co-ion 身份冻结，已修）：`select_co_alchemical_ion_once` 唯一选择入口，
  spec 落盘冻结，6 个消费点核对。真正的漂移入口是**跨进程 resume**（首跑坐标=预平衡+2000步
  最小化，resume=直接读DCD末帧），0.05nm 位移即可翻转选择结果。⚠️ Atenolol 净电荷=0，
  这条路径在本体系测不出来。
- **B3**（PME charge-transfer 充电哈密顿量，已落地）：只有复合物腿。三个坑：
  `periodicdistance` 只存在于 `CustomExternalForce`（锚点相对距离要用
  `CustomCompoundBondForce`+`pointdistance`+PBC）；co-ion 必须是建系时预留的电荷=0
  ion-shaped 粒子（不能拿已带电的物理盐离子顶上）；PME 下不能用"置零其它电荷比总能"验证
  静电冻结（Ewald 自能项本身随 λ 变）。
- **B4**（溶剂腿 builder，已落地）：`_insert_reserved_coalchemical_ion_dummies()` 摘掉
  最远的水分子换成同号 ion-shaped dummy，电荷 createSystem 后清零。⚠️ 只在合成 topology
  单测过，Atenolol 净电荷=0，本仓库测不出真实带电路径。
- **B5**（cache/resume/provenance）：代码已实现，定向测试 181 passed，但**未勾选**——
  还差 `./tests/run_offline_tests.sh -q` 全套 0 failed 这道最终证据门。
- **C1**（带电小水盒验证，用 Na+/Cl-/Ca2+ 当"配体"）：Na/Cl 大小盒硬性验收**已通过**
  （单 seed pilot，`|ΔΔG|` 远低于阈值，MBAR converged）。Ca2+（额外验证，非硬性要求）
  **不收敛**——挖出并修了一个真实的 co-ion 摆放 bug（多个单价 co-ion 彼此挤在同一角落），
  但修复后诊断显示不是根因；真根因是"1个二价点电荷拆成2个单价dummy"这个 charge-splitting
  本身就有的真实自由能差异，量级太大，11点均匀λ给不出足够overlap，需要更密λ表，**不是bug**，
  **不影响C1**（硬性验收只要求 q=±1）。
- **MEM-00h**（统一 ligand-environment 与 environment-environment LJ cutoff，旁线独立 PR，
  不与 charge-transfer 混改）：**代码改动已存在**——基础 NonbondedForce/ACE-IBS softcore/
  传统 Beutler softcore cutoff 全部统一为 1.0nm、switching 关闭，LRC 积分边界 1.0nm→∞，
  DEXP 独立 0.70/0.20 不受影响，co-ion↔ligand 1.2nm 运行时门保持独立。协议版本隔离机制
  （`VDW_NONBONDED_PROTOCOL_VERSION`）也已确认只作废 Stage2/vdW 缓存，不误废 Stage1/
  Boresch/预平衡/C1 缓存。**但 5 个关闭条件一个都没跑测试验证**——三条协议路径测试、
  Stage2缓存识别新版本、Stage1缓存不被误废、全套离线测试无新增失败、C3真实端点λ=1能量/
  力匹配，全部待验证（后两项尤其需要真机/GPU）。
- **已知未解决**：attachment 跨运行散布问题（P1-19b，统计限制不阻塞开发）；
  P0-13（OpenMM System 可以有多个同类型力，`next()` 只取第一个曾把配体71个键角静默丢掉，
  已修，教训是"抽取参数后必须逐项对账数量而不只是数值"）；P0-REMD-CUDA-Context
  （pymbar4的JAX后端预分配显存75%导致REMD建不出Context，已修：在任何`import pymbar`
  之前设 `XLA_PYTHON_CLIENT_PREALLOCATE=false`）；窗口预热排序bug（EM在Boresch
  scale=1.0生效前跑导致深度解耦窗口NaN，已改代码但未在真实GPU验证）；
  RESUME-FP-01（坐标哈希被误用作4处resume协议指纹的强判据，已删除，window级GPU
  checkpoint不受影响）。

## 环境信息

跑 `dexp_experiment.py`/`abfe_core.py` 的正确 conda 环境：
`/home/ruigengji/mambaforge/envs/openmm_dev`（python 3.12, OpenMM 8.5.1, openmm-ml 1.6,
mace_torch 0.3.16, mdtraj 1.11.1）。用 `mamba activate openmm_dev`（不是 `conda activate`，
这台机器的 conda 本身是坏的）。⚠️ 2026-08-04 起有些工作实际在 `omm_torch_126` 环境跑——
两个环境具体谁有什么包，用之前建议直接确认，不要假设。GPU 是 NVIDIA RTX 2080 Ti。
`/home/ruigengji` 是 NFS 挂载，`import openmm` 首次可能要 60-300+ 秒，并发 GPU 任务重时
可能卡更久（`cat /proc/<pid>/wchan` 显示 `rpc_wait_bit_killable` 说明是 NFS 阻塞不是脚本卡死）。
**`./tests/run_offline_tests.sh` 运行时不要编辑生产 .py 文件**——本仓库很多契约测试用
`inspect.getsource`，运行中改文件会读到错误行号导致假失败。

---

*本文件整理自历史记录，用于交接。如有疑问以仓库内 `memtodolist.md`/`docs/TODO.md`/
相应 handoff 文档为准，本文件内容可能滞后于代码。*
