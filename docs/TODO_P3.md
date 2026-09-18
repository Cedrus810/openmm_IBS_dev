# 待办 · P3 —— 现在明确不做

[项目入口](../README.md) · [文档导航](README.md) · [当前科学状态](STATUS.md) · [变更记录](CHANGELOG.md) · [P1 在推的](TODO.md) · [P2 该做不挡路](TODO_P2.md)

> 2026-09-16 从 `TODO.md` 按优先级拆出。**本文一条都不会被打勾**（除非触发条件到了 —— 见 §5）。
>
> 来这里只有一个理由：**你正准备重新论证某条东西** —— 先确认它不在这里。
> 每条都写了「理由别重新论证」，重查一遍的代价本仓已经付过好几次。
>
> 📌 **本文故意不用 `- [ ]`**（§3 膜线除外，那是暂停中的将来工作）——
> checkbox 的语义是"待办、将来会打勾"，会让 `grep -c '^- \[ \]'` 得到虚高的待办数。

---

## 1. 已定位、判定不改（`AUDIT-01`~`07` [P3]）

> 完整记录见 [`archive/AUDIT_2026-09-09_full_repo.md`](archive/AUDIT_2026-09-09_full_repo.md)。
> 那轮审计共 62 条候选，**52 条已修**（含 NaN 根因、stdout 消失、离线重算挂死、
> 一批宿主内存放大、preopt 缓存两层拆分）。下面 7 条是**有意不修**的，
> 登记在这里是因为它们确实"已定位、未修" —— 下一个人有权知道，
> 也免得被当成新发现重查一遍。**理由都别重新论证。**

> 📌 **这 7 条故意不用 `- [ ]`** —— checkbox 的语义是"待办、将来会打勾"，而这些永远不会
> 被打勾。用 checkbox 写会让任何 `grep -c '^- \[ \]'` 得到一个虚高的待办数。

- **AUDIT-01 [P3]** `abfe_core.minimum_image_displacement_nm` —— 候选立方体随长宽比增长，
  `radius > 64` 的 guard 在尝试 `N × 2.1e6 × 3 × 8` 字节之后才触发。
  *不修*：真实触发需要极端长宽比 + 大批量输入；`docs/design` 的盒型识别提案覆盖这一片。
- **AUDIT-02 [P3]** `abfe_core` 膜 leaflet 的 wrapped/unwrapped 混用
  （`_protein_leaflet_cross_sections_nm2`、`assign_lipid_leaflets`、
  `verify_membrane_normal_axis`）。
  *不修*：只影响膜路径，且 `membrane_observables_from_trajectory` 已按最大空隙弧修过一次；
  当前生产是可溶体系，留给膜线单独一轮。
- **AUDIT-03 [P3]** `free_energy_engine.run_independent_windows` 保留全部帧
  （73k × 1000 帧 × 8 态 ≈ 14 GB）。
  *不修*：当前无生产调用者，是"接线即爆"而不是现在就爆。接线前必须先改。
- **AUDIT-04 [P3]** `apbs_correction._read_dx_values` —— 257³ 网格约 1 GB 峰值
  （原文本 + 切片副本 + Python float list 三份）。
  *不修*：APBS 修正当前不在主线路径上。改法是 `np.frombuffer`，可降到 ~136 MB。
- **AUDIT-05 [P3]** `abfe_core.OnlineConvergenceMonitor` 的 K==1 会抛、
  `n_k_array` 与 `u_kn` 列数不一致。
  *不修*：`abfe_core` 之外无调用者。
- **AUDIT-06 [P3]** `TraditionalABFEPipeline.pre_equilibration_identity_fingerprint`
  引用 `self.pressure` / `self.barostat_protocol`，该类 `__init__` 从未赋值。
  *不修*：全仓无活的调用点（runabfe 里那两个 baseline 都是 `ABFEPipeline` 实例）。
- **AUDIT-07 [P3]** `abfe_pipeline._rebalance_fingerprint` 没有 System 身份绑定
  ⟹ System 变了而 Boresch 锚点没变时，`rebalance.chk` 会被复用。
  *🛑 明确不修*：唯一修法是把自产产物的 sha256 放进缓存身份，而那是**用户否决过
  4 次**的做法（`code_sha256` 2026-08-24、`system_xml_hash`+`positions_sha256`
  2026-09-09、`preopt_cache_sha256` 2026-09-09）。这类 payload 本来就有显式协议
  版本号承担"算法变了"的信号。**不要再提这个方案。**

> ⚠️ 那 52 处**没有一处上过 GPU**。最需要真机复验的：偏置爬坡补 1.0 档、
> preopt 探针的 force group 重划、加密点的采样语义变更。
> 复跑命令见审计文档 §4。

---

---

## 2. benchmark 里已确认「不是 bug」的（`BM-A` [P3]）

> **`BM-A` 这些不是 bug，别再查**（理由别重新论证）：
> · **求解器缺窗 4 个**（`p38_ligand1/rep2` 物理 win5 去相关后 5 帧 **g=100.1**；
>   `cmet_ligand2/rep1` 跳 [3,4]，win3 g=82.6；`p38_ligand2/rep3` 跳 [4] g=60.0；
>   `cmet_ligand1/rep2` 跳 [0] g=75.4）—— **真采样不足**，门抓对了。要治就是加采样
>   或缩窗跨度，不是改门。
> · `cmet_ligand1/rep3` attachment 腿前后半程 5.1638→4.4046、漂移 −0.7592 > 容差 0.5373
>   —— 真·系综未收敛。
> · `cmet_ligand2/rep3`（09-09 旧 run）50/50 次权重更新才判不收敛，预算给足了。
> · `cyclod_ligand2/rep1_evidence`（09-11 旧 run）已被 `rep1_evidence_run2` 取代，过期。
> · `cyclod_ligand2/rep1_evidence_run2`（09-11 旧 run）"累计已耗 555000 / 上限 555000、
>   本次可用 0 步" → 0 次更新 → 报"未收敛" → `IBSWarmupConvergenceError`。
>   **当时是真 bug（预算耗尽被写成 f_k 错了），现已修**：当前代码走
>   `IBSValidationBudgetIndeterminateError` + `last_failure_reason="halt_budget_no_steps_available"`
>   （`ibs_engine.py` 约 18253–18277）。旧 run 的这条不用再追。
> · **`Particle coordinate is NaN` 11 次 —— 10 次已修、1 次是已知设计**
>   （2026-09-16 逐条回日志 + 源码 + git 核实）：
>   · **10 次是 2026-09-09 单批事故**（`brd4_ligand1/rep2,rep3`、`brd4_ligand2/rep3`、
>     `cmet_ligand1/rep1-3`、`cmet_ligand2/rep1,rep2`、`cyclod_ligand1/rep2,rep3`），
>     全部在 **09-09 14:28–16:49** 这一批启动，签名逐个相同：死在偏置爬坡的
>     **第一个非零档 `scale=0.2`**，裸 `OpenMMException` 从 `sim.step()` 直通，无守卫。
>     `tests/test_step_guard.py` 的文件头独立记着同一件事（「10 个 run 全部这样在
>     窗口 0 的偏置预热里炸穿」）。守卫 `step_with_chunk_rollback` 于
>     **2026-09-12 15:50（commit `1769e71`）** 落地 ⟹ **已修**，旧 traceback 只是留在
>     追加式 `launch.log` 里（`BM-B` 取证口径② 的又一个实例）。
>     ⚠️ `ibs_engine.py` 那段注释写的 `🔑 [2026-09-09]` 是**诊断日期，不是落地日期** ——
>     照它推断「09-11 之后还炸就是没修好」会得出错误结论，本轮就这么错过一次。
>   · **剩下 1 次**（`cyclod_ligand2/rep3`，09-14 14:56，**守卫之后**）报的是
>     `窗口 0 偏置学习：积分 250 步过程中出现非有限坐标/能量`（坐标已到 3.7e9）——
>     那段用的是 `guarded_step`，它按设计**只换错误信息、不改动力学**
>     （docstring 原文），与热化/爬坡的 `step_with_chunk_rollback`（回滚+减半+重做）
>     **不是同一套守卫**。`ibs_engine.py` 自己也写着「warmup/learning 控制面没有 guard、
>     也没有周期性 checkpoint」⟹ **是已知设计，不是遗漏。**
>   ⚠️ **不是 `LR-06` 的鬼影期**（本轮先这么猜过，已证伪）：学习循环跑在
>   `bias_scale = 1.0`（循环前有显式赋值）。`LR-06` 那条鬼影链路讲的是爬坡**之前**的阶段。
>   📌 **要改这段守卫的前提**：回滚只还原 Context 的坐标/速度，而学习循环每轮都在往
>   `sampler.energy_buffer` / `energy_history` / `sampling_state_energy_history` 追加。
>   **回滚坐标却不回滚这些历史 = 偏置学习读到与构型对不上的样本**，正是本仓反复栽的
>   「同一份状态两个副本不同步」。⟹ 要接回滚就得连采样器历史一起回滚，那是另一件事，
>   **不许当成"顺手补个守卫"**。
>   判据已钉死在 `tests/test_warmup_guard_asymmetry_2026_09_16.py`（4 passed，已做还原变异）：
>   热化/爬坡必须可救、学习循环不可救、学习前 `bias_scale==1.0`，
>   外加一条**复现**——同一次瞬态 NaN 在爬坡侧被重做救回、在学习侧直接抛且不重试。

---

## 3. 膜受体–配体路线 [P3]（整条线暂停，下面 62 个框都是 P3）

> **2026-09-12 状态：整条线暂停**，停在 C4。当前主线是**可溶体系 4W53**，
> 膜线不阻塞任何在推的工作。本节内容的时间戳是 **2026-08-11**，之后没有人动过——
> 重启这条线之前先对着源码核一遍，别直接照着勾。
> 带电配体那一半另见本文 [PHY-03](#4-phy-03-p3带电路线的条件性阻塞)。

> 2026-08-31 发布整理并入，原文件 `docs/status/memtodolist.md`（在 `Atenolol-rank11`，**不在本仓**）。
>
> ⚠️ 本仓库的 `docs/status/` 已于 2026-09-02 撤销（见 [README](README.md)《运行期发现往哪写》）。
> `memtodolist*.md` 在 `Atenolol-rank11`，**别在本仓里找**。


更新日期：2026-08-11（**C3 与 MEM-00h 已正式关闭（用户确认），进入 C4**。
C3-0~C3-4 全部跑过一轮；co-ion/ParameterOffset 归因诊断完成；C3 protocol v2
双层门重设计已实现；C2 的 C-seam switch 不一致已用"MEM-00h 双边归一化"
修复并在全部真实 GPU 数据上验证——A/B 100/100 + C/D 50/50，全部 150 帧一次
通过，C2 的 C-seam 力差回落到机器精度；`summary.json`/`mem00h_report.json`
两份 fail-closed 汇总产物已生成，均 `status=complete, passed=true`）  
状态：Phase B 工程实现基本完成；B5 已关闭。C1、C2、C3 已关闭；MEM-00h 已
关闭。当前进入 C4。

**已关闭事项的完整过程、失败证据和验收记录均已原文迁移到 `docs/status/memtodolist_archive.md`（在 `Atenolol-rank11`，**不在本仓**）。**


---

### 1. 当前做到哪里

已完成的工程能力不再逐项放在本清单中，完整证据见归档。当前状态摘要：

- B1：膜体系识别和 `MonteCarloMembraneBarostat` 已实现。
- B2：`charge_treatment` 配置和双计数 fail-closed 已实现。
- B3：PME co-alchemical charge-transfer Hamiltonian 已实现。
- B4：溶剂腿 reserved co-ion dummy builder 已实现。
- B5 已关闭（2026-08-09）：cache、resume、provenance 全套离线测试 0 failed，
  co-ion 隔离/缓存拒绝/resume 一致性逐项复核通过。
- 中性 Atenolol 膜体系 complex/solvent 双腿工程 smoke test 已跑通。
- C1 已关闭：Na/Cl 硬性验收通过；采用单 seed pilot，不补 seed；Ca 为已知统计限制且不阻塞。

当前主线：

```text
B5、C1、C2、C3、MEM-00h（已关闭）
    ↓
C4 带电膜双腿 smoke test（当前，尚未开始）
    ↓
C5 co-ion 位置/restraint 敏感性
    ↓
Phase D 生产资格
```

---

### 2. Phase C：当前验证

#### C4：带电膜 complex/solvent 双腿 smoke test

前置：B5、C1、C2、C3 全部通过。**2026-08-11：确认 C3 与 MEM-00h 正式关闭，
C4 已解锁。**

**C4 是接线 smoke test，不是生产自由能计算**——不追求收敛，不出最终
ΔG；全部产物必须标 `production_qualified=false`（第 6 步）。C2 的纯脂质
slab（无蛋白）不能代替这里的真实 receptor–ligand complex；C4 第一次真正
需要"膜 + 蛋白 + 带净电配体"这套完整组合。

用户指定的执行顺序（2026-08-11 登记，按顺序执行，不并行跳步）：

**受体/配体组合——阻塞第 1 步，待用户决定，本文档不擅自选择**（2026-08-11
现状普查，只读，未改任何文件）：

- **已有、可复用的**：`memtest/` 下有一个真实的 283 残基 GPCR 样受体
  （`Atenolol-rank1apo.pdb`/`Atenolol-rank1.pdb`，含 TM3 的 `DRY`、TM7 的
  `NPxxY` 保守基序，疑似热稳定化突变体，ICL3 可能被截短）已经嵌入真实
  POPC 膜（`memtest/step7_production.gro`：`PROA 1 / POPC 90 / Na+ 25 /
  Cl- 36 / TP3 9542 / Atenolol-rank11 1`，45354 原子），配上中性 Atenolol
  （`Atenolol-rank1.gjf` 的 QM 电荷计算用的是 `Charge=0`，即去质子化的
  仲胺；`memtest/Atenolol-rank11.itp` 41 个原子电荷加总 Σq≈0），
  `memtest/README_MEMTEST.md` 记录了这套中性体系已经跑通的完整
  complex/solvent 双腿工程 smoke test（膜恒压器、quality gate、诊断脚本
  全部现成）。`abfe_core.py`/`runabfe.py` 的 charge-transfer + 膜恒压器
  通用接线（`--only-complex-charging`、`--membrane-input-declaration`、
  co-ion dummy 插入）已经用这套中性体系验证过，从未在带电配体上跑过。
- **真正的冲突**：`docs/status/memtodolist_archive.md`（在 `Atenolol-rank11`，**不在本仓**）（2026-07-29）记录过一条决定——
  **"首个体系 = SERT（血清素转运体），配体默认净电荷 +1"**。但实际建出来
  并跑通的是上面这个 GPCR + 中性 Atenolol，跟当年那条决定不是同一个体系：
  SERT 从未真正建过膜体系（没有对应的 CHARMM-GUI 产物、没有嵌膜、没有跑过
  任何 smoke）。
- **配体电荷缺口，跟选哪个受体无关，两条路都要补**：仓库里没有任何带电
  （质子化、净 +1）的 Atenolol 参数——所有现成拓扑（根目录
  `Atenolol-rank1.itp`、`memtest/Atenolol-rank11.itp`）都是从
  `Charge=0` 的 QM 计算导出的中性形式。要走"配体带净电"这条路，不管配哪个
  受体，都需要重新做一次质子化仲胺的 QM 电荷推导（Gaussian）+ 重新生成
  GAFF 拓扑——不是挪文件就能解决的工作量。
- **受体身份记录缺口**：`memtest/membrane_input.json` 明确写着
  "未记录上游 PDB ID"、构象态"unspecified"——呼应 §A5"记录受体结构 ID、
  构象状态、突变、缺失残基和质子化态"这条从未打勾的要求；C4 定位是接线
  smoke（`production_qualified=false`），这个记录缺口是否必须先补齐、
  还是可以先如实标注"未知"往前走，也需要用户决定。

**用户 2026-08-11 明确表示：这个选择稍后告诉我，现在只要求把决策点和现状
写清楚——不要自己选受体/配体组合，也不要开始任何构建。**

1. **准备真实带电膜 complex，以及匹配的 solvent leg**
   - [ ] ligand 必须带净电荷（不是 C1/C2 用的中性探针或单原子简化）；
   - [ ] build 时显式插入 reserved neutral ion-shaped dummy；
   - [ ] 排除结构性离子、孔道离子、口袋/膜头基/疏水核中的候选（呼应
     §A5 已经列出但从未做过的排除清单）；
   - [ ] complex 与 solvent 两腿冻结**同一个** co-ion identity 和 restraint
     定义（不能两腿各自独立选一次）。
2. **零步静态预检**（不积分，只建 Context 查一次）
   - [ ] charging 全部 λ 态总电荷恒定；
   - [ ] `λ_coul=1`：ligand 满电、co-ion 中性；`λ_coul=0`：ligand 去电、
     co-ion fully charged；
   - [ ] Stage2 输入已经 baking 完成，System 里不存在活的 `lam_coul`
     GlobalParameter；
   - [ ] complex 用膜恒压器（`MonteCarloMembraneBarostat`），solvent 用
     各向同性恒压器；
   - [ ] handoff protocol/version 和 co-ion fingerprint 都已经进入
     cache identity。
3. **最短 GPU smoke**（不追求自由能收敛，只要能跑）
   - [ ] complex charging 能建 Context、积分、写 checkpoint；
   - [ ] complex Stage2 能接上 charging 端点（真正走一次 Stage2 handoff）；
   - [ ] solvent charging/Stage2 同样可运行；
   - [ ] 全程 energy/force finite；无 NaN、PME error、粒子逃逸或
     restraint runaway；
   - [ ] Stage2 全程 co-ion 保持 fully charged。
4. **相同命令立即 resume 第二次**
   - [ ] 命中相同 co-ion identity；
   - [ ] 已完成窗口被复用，不重跑；
   - [ ] 不重复插入 dummy/offset/restraint；
   - [ ] handoff/cache protocol 字段一致。
5. **复制一份 co-ion spec、故意篡改**（atom index / fingerprint /
   endpoint charge 任选一种）
   - [ ] 必须在建 Context **之前** fail closed；
   - [ ] 原始产物不能被这次篡改测试覆盖/污染。
6. **所有 C4 输出统一标注**
   ```json
   {"production_qualified": false}
   ```
   C4 只是接线 smoke，即使全部 PASS 也不能当生产结果用。

**当前最先要做的是第 1 步**：确定并预检真实带电膜 complex/solvent 输入。
§A5"目标膜输入"下的清单（受体结构 ID、构象状态、配体质子化态/形式电荷、
结构性离子排除等）到目前为止都还没做过，是这一步要补的作业，不是重复劳动。

#### C5：co-ion 位置与 restraint 敏感性

前置：C4 通过。

- [ ] 至少 3 个合法 bulk-water 位置和 1 个故意违规位置。
- [ ] restraint 基线：k=100、r0=0.5。
- [ ] 弱/宽：k=50、r0=0.7。
- [ ] 强/窄：k=200、r0=0.3。
- [ ] 每个合法组合跑 complex/solvent 两腿和至少 3 seeds。
- [ ] 检查 dummy 吸附、charged endpoint 水合、触壁比例和 restraint 能量。
- [ ] 净 `ΔΔG_bind` 同时满足 2σ 和 1 kcal/mol 门。
- [ ] 若两腿 restraint 自由能不抵消，给出显式修正或判定路线失败。

---

### 3. 膜输入与科学协议仍缺

#### A5：目标膜输入

- [ ] 准备并验证真正用于带电生产的已平衡膜输入。
- [ ] 记录受体结构 ID、构象状态、突变、缺失残基和质子化态。
- [ ] 记录配体质子化态、互变异构体、形式电荷和参数来源。
- [ ] 核对结构性 Na⁺/Cl⁻，从 co-ion 候选中显式排除。
- [ ] 排除蛋白孔道、结合口袋、膜头基层和疏水核中的 co-ion 候选。
- [ ] 核对蛋白插膜方向、配体 pose、结构水、辅因子和二硫键。
- [ ] 记录膜组成、上下叶组成、胆固醇比例、盐浓度和温度。

#### 热力学循环和 restraint 账目

- [ ] 写清 co-ion restraint 在 complex/solvent 两腿是否抵消。
- [ ] 若可用体积不同，推导并实现显式修正。
- [ ] charge-transfer 路线最终报告必须明确 `APBS/Rocklin = 0`。
- [ ] co-annihilation 只允许实验对照，禁止进入膜生产 preset。
- [ ] `shadow_ibs` 对带电配体明确 fail closed，或完整实现同一 co-ion 路线。

#### 膜生产协议

- [ ] 明确炼金生产阶段使用 NPT 还是 NVT。
- [ ] 若使用 NVT，记录固定盒矢量来自哪一帧。
- [ ] 明确时间步、约束和是否使用 HMR。
- [ ] 明确膜位置限制的分级释放方案。
- [ ] 记录结合位点是水相可及、界面、脂质暴露还是疏水深埋。
- [ ] 对脂质暴露/空腔填充做正反向或双初态迟滞验证。

---

### 4. 生产资格 Phase D

- [ ] D1：关闭 P1-19/P1-19b 的跨运行不确定度问题。
- [ ] D1：对 P1-22 的 Stage 2 帧选择和 σ 口径形成正式结论。
- [ ] D2：完成 Boresch 真实键拓扑和二面角更新门。
- [ ] D3：至少 3 个独立生产重复一致。
- [ ] D4：至少一个公开或可追溯膜受体 benchmark 通过。
- [ ] D5：完整 provenance、运行命令、环境、seed、输入 SHA256 和复现实验脚本。
- [ ] 膜质量门通过。
- [ ] overlap/ESS 和修正后的不确定度门通过。

---

### 5. Definition of Done

只有以下项目全部完成，才能声明支持生产级膜受体–配体 ABFE：

- [ ] C2–C5 全部通过（C1、C2、C3 已关闭并归档；C4 已解锁，C5 未开始）。
- [ ] co-ion 两腿显式存在、进入 PME、受控并进入全部缓存指纹。
- [ ] 全部 λ 总电荷恒定，且未重复应用 APBS/Rocklin。
- [ ] 膜恒压和平衡质量门通过。
- [ ] Boresch、co-ion restraint 和标准态修正闭环。
- [ ] 至少 3 个独立重复一致。
- [ ] 公开 benchmark 通过。
- [ ] 最终结果可审计、可恢复、可复现。

---

## 4. PHY-03 [P3]（带电路线的条件性阻塞）

**P1，实验路线。**charge-transfer 的 tethered charge carrier 不能按当前论证严格跨腿抵消

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

---

## 5. 发布工程门里押后的三条（原 `REL-05` / `REL-06` / `REL-10`）

> 三条的触发条件是**同一个**：Stage-2 重构落地 + 方法定稿。在那之前做都是白做
> （改了要再改一遍、出的图要过期）。触发之后整节挪回 [TODO_P2.md](TODO_P2.md) §4。

- [ ] **REL-05 [P3] 代码内的中文要切成英文**（**末期收尾项，现在不做**）。
  **2026-09-16 重测**（原 09-12 的数字已过期，四项全部涨了，列在右边对照）：

  | 类别 | 2026-09-16 | 2026-09-12 |
  |---|---|---|
  | 已跟踪的 `.py` / 总行数 | **271 个 / 176,467 行** | 223 个 / 157,528 行 |
  | 含中文的行（合计） | **37,360**，分布在 **237** 个文件 | 30,642 / 190 个 |
  | **同行含 `raise`/`logger`/`print` 的** | **1,673 行、46 个文件** | 1,563 行 / 54 个 |

  **用户直接看得见的是最后那 1,673 行**；多行拼接没算进去，真实数只多不少。
  ⚠️ 这组数**每次改动都在涨**（4 天 +19k 行），引用前先重测，别直接抄。

  **优先级不是一刀切**：真正非切不可的是那 1,563 行**用户可见的异常与日志文本**
  （fail-closed 报错是这套流水线的主要交互面，报错看不懂等于没有 fail-closed）；
  注释与 docstring 可以最后再动、甚至不动。
  ⚠️ **不要现在做**：异常文本在大量测试里被 `match=` 断言，批量改动会跟正在进行的
  `S2-D` 重构直接撞车。**等重构落地、方法定稿再排。**

- [ ] **REL-06 [P3] 主页挂一张流程图。** 三份 README 的 `![` 计数都是 **0**；
  `docs/current-pipeline.svg` 是现成的，但只有 `docs/README.md` 引它。
  ⚠️ **不要现在做**：Stage-2 控制器正在重构（`S2-D`），现在挂等于挂一张要过期的图。
  **等重构落地后重新出图再挂。**

- [ ] **REL-10 [P3] 软件版本号与变更说明** ——
  [RELEASE_READINESS:178](archive/RELEASE_READINESS_2026-08-31.md)。LICENSE（MIT + NOTICE）已有，
  `CITATION` 已有意押后（见上方"明确不做"），**只欠版本号本身**：冻一个版本、写清支持范围。
  ⚠️ 与 `REL-05`/`REL-06` 同理，**方法定稿前不急**；登记在这里是为了不再从 RELEASE_READINESS
  里被重新"发现"一遍。

> ### 🛑 两条**明确不做**，理由别重新论证
>
> | | 为什么不做 |
> |---|---|
> | **`CITATION.cff` / 引用信息** | 项目还在 dev，**方法本身没做完**。现在写引用就是承诺一个还不存在的东西。等方法定稿再写 |
> | **英文文档追平中文** | **中文是主文档，这是为开发方便的既定选择**，不是疏漏。`docs/` 下的教程以中文为准；`README_en.md` 覆盖完整流程即可，不做逐字对等 |
