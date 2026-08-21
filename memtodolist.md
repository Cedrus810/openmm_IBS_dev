# 膜受体–配体 ABFE 当前行动清单

更新日期：2026-08-11（**C3 与 MEM-00h 已正式关闭（用户确认），进入 C4**。
C3-0~C3-4 全部跑过一轮；co-ion/ParameterOffset 归因诊断完成；C3 protocol v2
双层门重设计已实现；C2 的 C-seam switch 不一致已用"MEM-00h 双边归一化"
修复并在全部真实 GPU 数据上验证——A/B 100/100 + C/D 50/50，全部 150 帧一次
通过，C2 的 C-seam 力差回落到机器精度；`summary.json`/`mem00h_report.json`
两份 fail-closed 汇总产物已生成，均 `status=complete, passed=true`）  
状态：Phase B 工程实现基本完成；B5 已关闭。C1、C2、C3 已关闭；MEM-00h 已
关闭。当前进入 C4。

**已关闭事项的完整过程、失败证据和验收记录均已原文迁移到 `memtodolist_archive.md`。**


---

## 1. 当前做到哪里

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

## 2. Phase C：当前验证

### C4：带电膜 complex/solvent 双腿 smoke test

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
- **真正的冲突**：`memtodolist_archive.md`（2026-07-29）记录过一条决定——
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

### C5：co-ion 位置与 restraint 敏感性

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

## 3. 膜输入与科学协议仍缺

### A5：目标膜输入

- [ ] 准备并验证真正用于带电生产的已平衡膜输入。
- [ ] 记录受体结构 ID、构象状态、突变、缺失残基和质子化态。
- [ ] 记录配体质子化态、互变异构体、形式电荷和参数来源。
- [ ] 核对结构性 Na⁺/Cl⁻，从 co-ion 候选中显式排除。
- [ ] 排除蛋白孔道、结合口袋、膜头基层和疏水核中的 co-ion 候选。
- [ ] 核对蛋白插膜方向、配体 pose、结构水、辅因子和二硫键。
- [ ] 记录膜组成、上下叶组成、胆固醇比例、盐浓度和温度。

### 热力学循环和 restraint 账目

- [ ] 写清 co-ion restraint 在 complex/solvent 两腿是否抵消。
- [ ] 若可用体积不同，推导并实现显式修正。
- [ ] charge-transfer 路线最终报告必须明确 `APBS/Rocklin = 0`。
- [ ] co-annihilation 只允许实验对照，禁止进入膜生产 preset。
- [ ] `shadow_ibs` 对带电配体明确 fail closed，或完整实现同一 co-ion 路线。

### 膜生产协议

- [ ] 明确炼金生产阶段使用 NPT 还是 NVT。
- [ ] 若使用 NVT，记录固定盒矢量来自哪一帧。
- [ ] 明确时间步、约束和是否使用 HMR。
- [ ] 明确膜位置限制的分级释放方案。
- [ ] 记录结合位点是水相可及、界面、脂质暴露还是疏水深埋。
- [ ] 对脂质暴露/空腔填充做正反向或双初态迟滞验证。

---

## 4. 生产资格 Phase D

- [ ] D1：关闭 P1-19/P1-19b 的跨运行不确定度问题。
- [ ] D1：对 P1-22 的 Stage 2 帧选择和 σ 口径形成正式结论。
- [ ] D2：完成 Boresch 真实键拓扑和二面角更新门。
- [ ] D3：至少 3 个独立生产重复一致。
- [ ] D4：至少一个公开或可追溯膜受体 benchmark 通过。
- [ ] D5：完整 provenance、运行命令、环境、seed、输入 SHA256 和复现实验脚本。
- [ ] 膜质量门通过。
- [ ] overlap/ESS 和修正后的不确定度门通过。

---

## 5. Definition of Done

只有以下项目全部完成，才能声明支持生产级膜受体–配体 ABFE：

- [ ] C2–C5 全部通过（C1、C2、C3 已关闭并归档；C4 已解锁，C5 未开始）。
- [ ] co-ion 两腿显式存在、进入 PME、受控并进入全部缓存指纹。
- [ ] 全部 λ 总电荷恒定，且未重复应用 APBS/Rocklin。
- [ ] 膜恒压和平衡质量门通过。
- [ ] Boresch、co-ion restraint 和标准态修正闭环。
- [ ] 至少 3 个独立重复一致。
- [ ] 公开 benchmark 通过。
- [ ] 最终结果可审计、可恢复、可复现。
