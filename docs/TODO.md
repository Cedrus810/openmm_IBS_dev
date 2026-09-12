# 当前行动清单

> ## ⚠️ 时效警告（2026-08-31 发布整理核对）
>
> **下面《当前决策》到《长期研究项》各节的主表停在 2026-08-06，已陈旧约 25 天，
> 其中相当一部分条目很可能已经完成——但本仓库里没有能证明这一点的判据。**
>
> 期间发生过：EXP-025~EXP-030、`BUGFIX_HANDOFF_2026-08-29` 的 41 项修复、
> 2026-08-31 第九轮代码审查。这些的逐项记录都**不在本工程区分支**：
>
> | 材料 | 位置 | 内容 |
> |---|---|---|
> | `BUGFIX_HANDOFF_2026-08-29_resolved_issues.md` | `Atenolol-rank11/archive/patches/` | 41 项已解决的逐条记录 |
> | `0831issue.md` | `Atenolol-rank11/` | 第九轮审查新发现（2026-08-31，**尚未完成**，两个分片还在审查中） |
> | `github issue.md` | `Atenolol-rank11/` | GitHub issue 状态 |
>
> **不要按编号机械对账。** 两套文档的编号体系互相冲突且同名不同义——例如
> `P1-19` 在 08-29 那份里是"v4 charging 接缝内静电失配"（已修），在本文件里是
> "per-window σ 系统性低估 2–4 倍"（未完成）。照编号勾选会把未完成的物理问题
> 错标成已修。本文件用的是更早的 `ATT-xx` / `MEM-xx` / `P0-9~13` 一套。
>
> 下面那些 `- [ ]` **只表示"2026-08-06 当时未完成"**，不表示现在仍未完成。
> 要更新状态，需要人对着上面三份材料逐条判定，本次发布整理没有代做这件事。

## 本文件的范围

发布整理后，本文件只保留**有明确时间戳、且来源比 2026-08-06 更新**的两节：

- 《膜受体–配体路线》——更新于 2026-08-11（C4 已解锁、C5 未开始）；
- 《未关闭的代码缺陷》——更新于 2026-08-29 / 08-31，其中 **PHY-03 是唯一
  被明确记录为"仍挂起"的条目**。

2026-08-06 那份主表（1350 行、`ATT-xx`/`MEM-xx`/`P0-9~13` 编号）已整段归档到
[archive/TODO_2026-08-06_unreconciled.md](archive/TODO_2026-08-06_unreconciled.md)，
一字未改。它需要人逐条对账后才能重新变成待办，本次发布整理没有代做。

---

## 膜受体–配体路线

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

## 未关闭的代码缺陷

> 2026-08-31 发布整理并入，原文件 `docs/status/BUGFIX_HANDOFF_2026-08-29.md`（在 `Atenolol-rank11`，**不在本仓**）。


> 2026-08-31 状态：40 项已修复并已拆分为 GitHub issue；1 项科学验证（PHY-03）挂起。
> 已解决条目的完整交接内容已归档到 `Atenolol-rank11/archive/patches/BUGFIX_HANDOFF_2026-08-29_resolved_issues.md`（在 `Atenolol-rank11`，**不在本仓**）。

### 给接手人的提醒

请先修复 P1，再处理 P2。不要通过降低 fail-closed 门槛、删除 provenance、
忽略 checkpoint 不一致或复用旧结果来让流程跑通。

涉及 Hamiltonian、采样协议、缓存含义或结果口径的修改，必须同步更新协议版本、
缓存指纹和回归测试。

### 当前未完成项

#### AUDIT-2026-09-09：全仓审计的 7 条「判定不改」

> 完整记录见 [`AUDIT_2026-09-09_full_repo.md`](AUDIT_2026-09-09_full_repo.md)。
> 那轮审计共 62 条候选，**52 条已修**（含 NaN 根因、stdout 消失、离线重算挂死、
> 一批宿主内存放大、preopt 缓存两层拆分）。下面 7 条是**有意不修**的，
> 登记在这里是因为它们确实"已定位、未修" —— 下一个人有权知道，
> 也免得被当成新发现重查一遍。**理由都别重新论证。**

- [ ] **AUDIT-01** `abfe_core.minimum_image_displacement_nm` —— 候选立方体随长宽比增长，
  `radius > 64` 的 guard 在尝试 `N × 2.1e6 × 3 × 8` 字节之后才触发。
  *不修*：真实触发需要极端长宽比 + 大批量输入；`docs/design` 的盒型识别提案覆盖这一片。
- [ ] **AUDIT-02** `abfe_core` 膜 leaflet 的 wrapped/unwrapped 混用
  （`_protein_leaflet_cross_sections_nm2`、`assign_lipid_leaflets`、
  `verify_membrane_normal_axis`）。
  *不修*：只影响膜路径，且 `membrane_observables_from_trajectory` 已按最大空隙弧修过一次；
  当前生产是可溶体系，留给膜线单独一轮。
- [ ] **AUDIT-03** `free_energy_engine.run_independent_windows` 保留全部帧
  （73k × 1000 帧 × 8 态 ≈ 14 GB）。
  *不修*：当前无生产调用者，是"接线即爆"而不是现在就爆。接线前必须先改。
- [ ] **AUDIT-04** `apbs_correction._read_dx_values` —— 257³ 网格约 1 GB 峰值
  （原文本 + 切片副本 + Python float list 三份）。
  *不修*：APBS 修正当前不在主线路径上。改法是 `np.frombuffer`，可降到 ~136 MB。
- [ ] **AUDIT-05** `abfe_core.OnlineConvergenceMonitor` 的 K==1 会抛、
  `n_k_array` 与 `u_kn` 列数不一致。
  *不修*：`abfe_core` 之外无调用者。
- [ ] **AUDIT-06** `TraditionalABFEPipeline.pre_equilibration_identity_fingerprint`
  引用 `self.pressure` / `self.barostat_protocol`，该类 `__init__` 从未赋值。
  *不修*：全仓无活的调用点（runabfe 里那两个 baseline 都是 `ABFEPipeline` 实例）。
- [ ] **AUDIT-07** `abfe_pipeline._rebalance_fingerprint` 没有 System 身份绑定
  ⟹ System 变了而 Boresch 锚点没变时，`rebalance.chk` 会被复用。
  *🛑 明确不修*：唯一修法是把自产产物的 sha256 放进缓存身份，而那是**用户否决过
  4 次**的做法（`code_sha256` 2026-08-24、`system_xml_hash`+`positions_sha256`
  2026-09-09、`preopt_cache_sha256` 2026-09-09）。这类 payload 本来就有显式协议
  版本号承担"算法变了"的信号。**不要再提这个方案。**

> ⚠️ 那 52 处**没有一处上过 GPU**。最需要真机复验的：偏置爬坡补 1.0 档、
> preopt 探针的 force group 重划、加密点的采样语义变更。
> 复跑命令见审计文档 §4。


#### [x] MIGRATE-01（已关闭 2026-09-09）EXP-031 接入主线时整份拷贝文件会抹掉 09-09 的 MEM-15 重构

> 迁移已完成（详见下方原记录）。**本条剩下的价值是一条规矩、不是任务**：
> 下次沙箱并线**别整份拷贝文件，逐 hunk 分拣**。


> **已完成（2026-09-09）**：路线 A 与路线 B 都已接入主线，融合内核（S3）本轮不接。
> 详见 `docs/EXP-031_GPU_OPTIMIZATION_2026-09-09.md` 两节顶部的落地记录。
> `IBS_BIAS_PROTOCOL_VERSION` 已到 **33**，兼容集合收窄成 `frozenset((33,))`。
> 回归 `pytest tests -m cpu_only` → 1079 passed。
> ⚠️ `system_xml_sha256` 已变 ⟹ 既有 `dual_window_*` / `ibs_state_*` /
> `convergence.json` 全部失配，**必须重跑**，不要试图复用。
> 本条（MIGRATE-01）的价值转为**留给下一次沙箱并线**：别整份拷贝文件，
> 逐 hunk 分拣。

- **发现**：2026-09-09 逐文件对账沙箱 `ABFE_IBS_CUDA` 与主线时发现。
- **事实**：三个文件里**只有 `ibs_engine.py` 是沙箱领先**；`abfe_core.py` 与
  `abfe_pipeline.py` **主线在 2026-09-09 12:25 刚动过，比沙箱新**。

  | 文件 | 主线 mtime | 沙箱 mtime | 谁领先 |
  |---|---|---|---|
  | `ibs_engine.py` | 09-03 20:48 | 09-08 16:22 | 沙箱 |
  | `abfe_core.py` | **09-09 12:25** | 09-05 10:44 | **主线** |
  | `abfe_pipeline.py` | **09-09 12:25** | 09-04 15:17 | **主线** |
  | `runabfe.py` / `free_energy_engine.py` | — | — | 零差异 |

- **危险动作**：主线领先的内容是 MEM-15 分子归组重构 —— 新增
  `abfe_core.py:4645 system_molecule_grouping()` 与 `:4696 image_molecules_by_system()`
  （按 **System** 的键+约束求连通分子，不信 topology 的键），并把 `abfe_pipeline.py`
  里内联的「约束补成键」换成调这两个 helper。**这两个函数在沙箱里完全不存在**
  （全仓 grep 零命中）。⟹ 整份拷贝沙箱那两个文件会抹掉 135 + 56 行。
  而「把 EXP-031 合并到主线」最自然的执行方式恰好就是整份拷贝。
- **为什么是 P0**：这是**静默数据丢失**，不报错。抹掉的 MEM-15 修复防的是刚性水
  被逐原子回卷撕开 → 729 个 PME 排除对跨盒 → `Particle coordinate is NaN`（不到 1 ps）。
  而该损坏对既有诊断是隐形的：键能、最大键长、最小化后 max|F| 全部正常，
  只有查排除对距离才看得见。
- **处置**：**逐条重放，不是合并。** `abfe_pipeline.py` 里没有任何 EXP-031 内容
  （那条线从未改过它），它的 56 行差异 100% 属于主线，一行都不要往主线迁；
  反过来沙箱应去拉主线那份。
- **清单**：`ABFE_IBS_CUDA/experiments/EXP-031_ibs_bias_fusion/MIGRATION_CHECKLIST_2026-09-09.md`
  （逐条动作、行号、代价栏、不迁清单）；主线侧摘要见
  [EXP-031_GPU_OPTIMIZATION_2026-09-09.md](EXP-031_GPU_OPTIMIZATION_2026-09-09.md) §1。
- **连带**：`docs/EXP-031_GPU_OPTIMIZATION_2026-09-04.md` 的两条结论已推翻、头条
  `1.45×` 是水盒数（真体系 1.162×）。已在该文件顶部加取代提示，正文留档未改。

#### [x] PBC-01（已关闭 2026-09-09）`topology.cif` 往返会凭空造出假键

- **发现**：2026-09-09，brd4/ligand1 benchmark（`abfe-benchmark/openmm_IBS/runs/brd4_ligand1/rep1`）。
  attachment 腿起点体检报「2 个 nonbonded_exceptions 对跨了周期镜像（最远 7.356 nm）」，
  而输入 `.gro` 干净（六个 target 实测 0 个跨镜像水、最大分子内跨度 0.096–0.100 nm）。
- **根因**：`app.PDBxFile` 写入端链 id 按 `chr(ord('A') + chainIndex % 26)` **循环**
  （`pdbxfile.py:392/473`），读取端 `_struct_conn` 只按 `(seq_id, asym_id, atom_name)`
  解析（`pdbxfile.py:218`），**不含链序号**。链数一超过 26 就有歧义。
  brd4/ligand1 有 12549 条链（每个水一条），第一个残基 `ACE(A,1)` 的三条键被解析到
  某个同样落在 `(A,1)` 的水上：

  ```
  (0, 39543) ACE1.CH3 — HOH12582.H1
  (0, 39544) ACE1.CH3 — HOH12582.H2
  (1, 39542) ACE1.C   — HOH12582.O
  ```

  `.top` 重建 27175 键，mmCIF 往返 27178 键，多的正好这 3 条；真键一条不缺
  （读取端 `createStandardBonds()` 补齐了，所以 3 条是**净多出**）。
  溶剂腿只有 3 条链，不触发。
- **已修（2026-09-09，两步都做完了）**：

  1. `repair_pbc_molecule_integrity` 的分子归组改成只信 System
     （`abfe_core.image_molecules_by_system` / `system_molecule_grouping`），
     回卷后逐对复查、fail closed。零指纹变动。
  2. `_load_system_from_native_cache` 载入 mmCIF 拓扑后调
     `abfe_core.prune_topology_bonds_unsupported_by_system()`，把 System
     完全不认的键删掉（判据宽：`HarmonicBondForce` ∪ `CustomBondForce` ∪
     `constraints`；漏删无害、误删致命）。同时两处裸 `traj.image_molecules()`
     换成 `image_molecules_by_system()`。

  原来记在这里的两个残留消费者，第 2 步一并解决了：

  | 位置 | 影响 |
  |---|---|
  | `ibs_engine.py` `compute_u_kn` 里的 `traj.image_molecules(inplace=True)` | **载荷相关**：重算 u_kn 之前给轨迹回卷，用的是 topology 的键，而且**连约束都没补**（比修好前的生产路径还弱），外面还包着 `except → warning` 的 fail-open |
  | `runabfe.py` 末帧 Boresch 诊断处的 `traj.image_molecules(inplace=True)` | 诊断用，末帧不重锚，影响面小 |

  `runabfe.py` 构造 Boresch 锚点图时会把**非配体键原样拷贝**
  （`for a, b in pipeline.topology.bonds()`），假边原本因此进了受体侧的锚点搜索图；
  拓扑在载入时就删干净了，这里不用再改。
- **本条最初写的方案（从 `.top` 重建拓扑）没有采用，理由记在这里，别再退回去**：
  删假键比换拓扑源好三点 —— 不需要 `.top`（`--openmm-cache-only` 也能用）、
  不引入任何新边（换成 System 的全图会多出 12478 条水的 H–H 边，
  可能干扰按键距计数的逻辑，如 `LIGAND_INTERNAL_POLAR_MIN_BOND_SEPARATION`）、
  且删完的键集与 `.top` **逐条相同**（实测 27178 → 27175 == `.top` 的 27175）。
- **代价：比原估的小得多，且不需要协议版本号 bump。**
  `abfe_pipeline._topology_hash()`（`abfe_pipeline.py:599`）把 bond 列表 +
  chain.id + residue.index 一起哈希，出现在 6 处 `_protocol_fingerprint(...)`
  （`abfe_pipeline.py:5710, 9363, 9564, 10224, 10554, 12421`）。但删键是**条件触发**的：
  一条都不用删时原样返回**同一个 topology 对象** ⟹ mmCIF 干净的体系
  （链数 ≤ 26，例如所有溶剂腿）`topology_sha256` 逐位不变、resume 全保住；
  只有真的带假键的体系指纹才变，而它们的既有结果本来就不可信。
  **不要再叠一个全局协议号 bump** —— 那会把没受影响的体系一起作废，
  正是「同一件事两套机制」那个反复踩过的坑（见 `code_sha256` 那条教训）。
- **物理量影响：无。** topology 不进哈密顿量，能量/力逐比特不变；唯一真实的坐标变化
  就是把被撕开的分子拼回去（以及随之而来的 ~0.005 nm 整体质心平移）。
  例外是 `compute_u_kn`：撕开的分子原本会给出跨盒 PME 排除对的错能量，
  现在不会了 —— 只在原本就错的情形下变，且回卷失败从 fail-open 改成直接抛。
- **验证**：从 `.top` + `.gro` 原始输入端到端复现整条机制（不依赖任何 run 产物），
  往返后多出 3 条、删完与 `.top` 键集逐条相同、原子/残基/链/盒矢量全保持；
  干净拓扑返回同一对象且哈希不变。回归 `pytest tests -m cpu_only`：1071 passed。
  测试：`tests/test_membrane_barostat_protocol.py` 的
  `test_image_molecules_ignores_phantom_topology_bonds` /
  `test_prune_drops_only_bonds_the_system_does_not_back` /
  `test_pbc_repair_groups_molecules_by_system_not_topology`。


#### [x] XFAIL-01（已关闭 2026-09-09）P1-19 的 C_seam 已修好，xfail 标记已摘除

- 位置：`tests/test_charge_transfer_real_endpoints.py:453` 的
  `@pytest.mark.xfail(strict=False)`，挂在
  `test_vanishing_lambda_one_seam_matches_charging_lambda_zero` 上。
- reason 自述「实测 118.5 kJ/mol（中性 4 原子 fixture）…… **修复后此标记应转
  XPASS 并摘除**」。它**现在正是 XPASS**，但 `strict=False` ⟹ 套件不会提醒任何人摘。
- **实测（2026-09-02）**：`tools/diagnostics/probe_p119_charge_transfer_seam.py`

  ```
  abs_delta_e = 3.63206042e-04 kJ/mol      ← 不是 118.5，小 5.51 个数量级
  rel_delta_e = 1.807e-06                  (门 1e-05)
  max|ΔF|分量 = 2.526e-06 kJ/mol/nm        (门 1e-03)
  ```

- **剩下这 3.6e-4 不是 seam 残余**：同文件
  `test_bake_handoff_seam_matches_for_charged_ligand_with_realistic_geometry`
  上方的注释精确描述过它——紧凑几何（配体 4 原子挤在 <0.2 nm 内）自带一个
  「与几何基本无关的 ~0.0005 kJ/mol 绝对残差」，数值性的，不是 Hamiltonian
  构造错误。量级吻合。**没有这条排除性说明，下一个人会以为 3.6e-4 是 seam 残余。**
- **已排掉「fixture 绕过失效路径」**（这是「真修好」与「绕过去了」的唯一分界）：
  `LIGAND_CHARGES_NEUTRAL_E = (0.5, 0.3, -0.4, -0.4)` 逐原子非零，
  `LIGAND_ORDINARY_PAIRS = {(0, 3)}` 是真正的 ordinary L-L 对（未定义任何
  exception、走标准 combining rule，q_i·q_j = −0.2 e²）⟹ 内部库仑真实存在、
  **机制被触发**，但常数不见了。
- **谁修的、什么时候修的：未知。** 跨会话核对过时间线，只能**排除**：不是
  2026-09-02 那两个会话中的任何一个，也不是 λ-WCA 壳退役、也不是力组切分收敛
  （`IBS_E_BASE_FORCE_GROUPS`/`IBS_E_BIAS_FORCE_GROUPS`）的连带效果——那天第一次
  全套跑之前 seam 就已经 XPASS。⚠️ **"未知"就是未知**，不要把它写成「大概是某次
  改动的连带效果」——那种猜测会被后人当结论。
- **已处置（2026-09-09）**：标记已摘除，原 reason 里的实测数据与「3.6e-4 是紧凑几何数值底噪、不是 seam 残余」这条排除性说明改写成函数上方的注释保留。
  `pytest tests/test_charge_transfer_real_endpoints.py` → 39 passed。
  ⚠️ 仍待人工确认：P1-19 在 issue 追踪里的状态该不该一起关（本仓没有 issue 追踪器，需要在上游做）。

#### [x] XFAIL-02（已关闭 2026-09-09）两个 xfail 的 reason 是错的：它们红在 D，不在 C——根因已查明并修复

- 位置：`tests/test_charge_transfer_real_endpoints.py:633` 与 `:822`
  （`test_run_protocol_v2_matrix_cd_wiring_passes_on_charged_fixture`、
  `test_run_protocol_v2_matrix_cd_normalizes_c2_style_switch_before_c_seam`）。
- 两条的 reason 都写「同 …… 的 xfail 理由」，即都记在 P1-19 的 C_seam 失配上。
  **这是错的。**
- **实测（2026-09-02）**，复现方式
  `python -m pytest tests/test_charge_transfer_real_endpoints.py --runxfail -q`，
  两条的 `failed_frames` 完全一致：

  ```
  'failing': ['D:gate1_reference_identity,gate3_mixed_production_vs_reference']
  ```

  **前缀是 `D:`。整个输出里 `failing` 一次都没出现 `C:`** ⟹ C（seam）在这两条里
  也是通过的，红的是 **D 端点**（全解耦：λ_coul≡0 且 λ_vdw=0）。
- 讽刺的是 `:822` 那条测试名叫 `..._normalizes_c2_style_switch_before_c_seam`
  ——它本身是为 C seam 写的，却卡在 D。
- 为什么与 XFAIL-01 是**不同机制**：这两条的 fixture 是 `_case(1, n_dummies=1)`
  （净电荷 +1 + reserved dummy），走完整 `run_protocol_v2_matrix_cd`，即
  **co-alchemical charge-transfer** 路径（配体 +1 e → 0、co-ion 0 → +1 e、
  flat-bottom 位置限制 k=100 kJ/mol/nm²）。失败的是 co-ion 在 λ=0 时的
  reference identity 与 mixed(CPU)-vs-reference 一致性，跟「配体内部库仑常数」
  没有关系。
- ⚠️ **不是静默的生产 bug**：`charge_treatment=co_alchemical_charge_transfer`
  本就 `production_qualified=False`，PHY-03（P1，见本节下方）仍挂着。属于
  **已知未合格路径上的已知未合格行为**，只是被错标成了 P1-19。
  **别当 P0 处理。**
- **已处置（2026-09-09）：不是重写 reason，是直接修好了根因；两个标记已摘除。**

  上面那段「跟 co-ion 有关」的推断**是错的，别再沿用**。逐 gate 打开 report 后
  （`gate1`/`gate3` 都给出同一个数）：

  ```
  D  abs_delta_e = 22.057534 kJ/mol   rel = 5.108e-05 (门 1e-05)
     max|ΔF| = 1143.130726 kJ/mol/nm  全部落在原子 0 的 x 分量
     reference 侧该原子受力 = 恰好 0.0     production 侧 = -1143.13
     strict_zero_reference / strict_zero_mixed 都 PASS ⟹ 配体–环境项确实是零
     C 两个 gate 都 PASS，abs_delta_e = 5.2e-4（就是 XFAIL-01 那个紧凑几何底噪）
  ```

  手算 fixture 里唯一的 ordinary L–L 对 `(0,3)`（Lorentz-Berthelot，
  r=0.265149 nm、σ=0.34 nm、ε=0.36 kJ/mol）：

  ```
  U_LJ(0,3) = 22.057512 kJ/mol      |F| = 1143.129513 kJ/mol/nm      方向 +x
  ```

  六位有效数字吻合 ⟹ **差值 100% 就是普通 L–L 对的内部 LJ**。

- **根因**：`tools/validation/compare_charge_transfer_endpoints.py`
  的 `reference_vanishing_zero_system` 把配体粒子 epsilon 一律置零，
  连**普通（无 exception）L–L 对**的 LJ 一起杀掉了；生产侧 `U_common` 正确地
  逐 λ_vdw 保留它。是 XFAIL-01（库仑版）的 **LJ 版本**，但错在**参照构造侧，
  不在生产侧**。
- **为什么会出现**：v3 [P0-01] 停掉了 `reference_charging_endpoint_system` 的
  内部对补 exception——对**库仑**是正确的（普通 L–L 库仑必须随粒子电荷线性湮灭），
  但普通对的 **LJ** 从此失去了庇护。该函数的 docstring 当时还写着
  「配体内部 LJ 已经被冻结成显式 exception」，已经陈旧。
- **已修**：置零之前先把普通 L–L 对冻结成显式 exception，搬进 combining-rule 的
  σ/ε；chargeProd 取当前粒子电荷之积（λ=0 时为 0）⟹ 库仑逐比特不变，
  PME 的 exception 倒扣项同样正比于 q_i·q_j ⟹ 也是 0。
  只补普通对，raw 拓扑自带的 excluded/1-4 exception 一个不动 ⟹ 没有触碰 P0-01。
- **影响面**：该常数逐 λ_vdw 不变 ⟹ **ΔG_vdw / ΔG_bind 不受影响**；且只改验证工具，
  生产哈密顿量一行未动，协议版本号未动。
- **定级结论：既不属于 PHY-03，也不是生产 bug**，是验证参照的构造缺陷。
  PHY-03 仍然独立挂着，状态不变。
- **回归**：新增 `test_reference_vanishing_zero_keeps_ordinary_intra_ligand_lj`
  直接对着机制断言（普通对必须存在且 σ/ε 等于 combining rule，配体粒子 epsilon 仍须为 0）——
  原来的 D 门失败信息只说 `D:gate1/gate3`，看不出是 LJ，正是这次查了半天的原因。
- **为什么这条值得单列**：三条 xfail 共用一条错 reason，是**能自我掩盖的**——
  谁照那两条的 reason 去修 C seam，会去修一个已经修好的东西；而真问题
  （co-ion 在 D 端点）继续没人管；而且它不会在测试里报警，因为 XFAIL 也算"预期"。

#### [x] CACHE-01（P2，纯噪音）`--openmm-cache-only` 下无条件调用 `find_gmx_include_dir`

**已修（2026-09-02）。** `find_gmx_include_dir(config.gmx_path)` 挪进了非
cache-only 的 `else` 分支，cache-only 路径 `include_dir` 留 `None`
（`runabfe.py` 里搜 `[CACHE-01`）。cache-only 下不再打那条警告。

- 原症状：带 `--openmm-cache-only` 跑时照样打「找不到 GROMACS 力场 include 目录」
  警告，而这一路根本不需要 include 树。
- 修改前已查清：审计通过的缓存上 `include_dir` **一次都不会被解引用**（三条使用
  路径逐个核对过，见 [TROUBLESHOOTING.md](TROUBLESHOOTING.md) 同名小节）。
  ⟹ 只是噪音，不影响任何数值，挪动对 cache-only 路径行为中立。
- 挪动后复核过的那一条：`system_cache_exists(...)` 仍然在 `or` 的右侧，
  `openmm_cache_only=True` 时整个调用不执行（短路顺序未变）。
- 来源：2026-09-02 运行期记录（原文已归档到
  [archive/RUNTIME_ISSUES_2026-09-02.md](archive/RUNTIME_ISSUES_2026-09-02.md) BUG-1）。

#### [x] CFG-01（P2，配置）`abfe_config.json` 的 `gmx_path` 指向不存在的路径

- **2026-09-07 已修**：发布清理时清空为 `""`（机器本地键，见
  `abfe_diagnostics._MACHINE_LOCAL_KEYS`），留空回退到 `GMXLIB`/`GMXDATA`/`PATH`
  自动探测。下面是当时的记录。
- 原值 `/home/ruigengji/gmx26.0C`，**本机不存在**。
- 2026-09-01 那次实跑的 provenance 记的是
  `/home/ruigengji/gmx26.3/share/gromacs/top`，与 config 里的值不同 ⟹ 配置里的
  值从来没被那次运行用上（那次带了 `--openmm-cache-only`）。
- 与 CACHE-01 **是两件事**：CACHE-01 是「不该问」，这条是「问了但答案是错的」。
  非 cache-only 路径会真的用到它。
- 修法：写 GROMACS 的**安装前缀**，不要写 `share/gromacs/top`
  （解析逻辑见 `runabfe.py:557` `find_gmx_include_dir`，两种写法都能吃，但前缀是
  2026-08-31 之后的约定）。**改配置会动 provenance，需用户确认取哪个版本的 GROMACS。**

#### [ ] PHY-03（P1，实验路线）charge-transfer 的 tethered charge carrier 不能按当前论证严格跨腿抵消

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

### 验证边界

本轮五个核心文件已通过 py_compile/AST 检查；当前会话环境没有 OpenMM、pytest 或 ruff，因此未声称离线测试全绿。历史 openmm_dev 测试结果仅作为第五轮基线记录。
