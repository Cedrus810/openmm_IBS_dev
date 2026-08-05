# 膜体系溶剂腿：配体键角被静默丢掉（P0-13）+ 检测层（P0-12a/b）+ 色散分层（B6-FIX）

日期：2026-08-04
状态：**三处已修，CPU 全套 1036 passed / 2 skipped / 0 failed；等一次重跑验证。**
前序：[`CHARGE_TRANSFER_B3_HANDOFF.md`](CHARGE_TRANSFER_B3_HANDOFF.md)（同日的 B3 + MEM-00d）
行动状态一律以 [`../TODO.md`](../TODO.md) 的 P0-13 / P0-12 为准。

---

## 1. 起点：第一次端到端跑通，但 ΔG_bind 是 +23 kcal/mol

`memtest/output_membrane_100ns` 08-04 08:28 走完整条主链（预平衡 100 ns → attachment →
去电荷 → 去 VDW → 溶剂腿 → 汇总），§17.0 第 ① 步的**工程**目标达到。但：

```
复合物腿 175.57 ± 1.50   ← 用户确认没问题
溶剂腿   272.93 ± 1.46   ← 有缺陷
ΔG_bind  = +97.36 kJ/mol = +23.27 kcal/mol   ← 不可用
```

配体是**中性 Atenolol**，与可溶生产体系是同一个分子（`memtest/topol.top` 的
moleculetype `Atenolol-rank11`，41 原子，Σq = 0.000000），所以两次运行的溶剂腿
**必须可比** —— 但不可比：

| 溶剂腿分项（同一配体、同为纯水盒） | 可溶生产 | 膜运行 | 差 |
| --- | --- | --- | --- |
| decharging | 62.80 | **191.05** | **+128.25** |
| vanishing | 96.96 | 83.83 | −13.14 |

---

## 2. 排除路径（每一步都离线可复现，别再走一遍）

顺序是有讲究的 —— 先证明**估计量**没问题，再看 **u_kn 的结构**，最后才去比**参数**。
我一开始只比了电荷/α/L-L 冻结，全都对得上，差点得出"这是物理"的错误结论。

| 假设 | 判定与证据 |
| --- | --- |
| 估计量坏了 | ❌ pymbar MBAR 与相邻 BAR 都**逐位复现**落盘值（191.05 / 62.80）。⚠️ 我手搓的 BAR 给出 2605/2755 kJ/mol 的荒谬值 —— **别用自己现搓的估计量下结论** |
| PME 自能项不对消（α 或 Σq² 两腿不同） | ❌ 两条腿配体逐原子电荷**逐位相同**，Σq² = 4.9909 e²，α = 2.6283/nm，ΔC = 0.00 |
| 配体参数不同（CHARMM-GUI 重新赋电荷） | ❌ 两次运行电荷几乎相同（−0.8965 vs −0.9001…），Σq² 只差 0.036 e² → 7.35 kJ/mol |
| 配体内部库仑被 λ 缩放（L-L 没冻住） | ❌ 两边都 820/820 对全冻结、0 个挂 λ offset |
| 溶剂盒污染 / 重复配体 / 几何坏 | ❌ 组成干净（1 MOL + 2115/2454 水 + 6/7 对 NaCl），键长 0.096–0.159 nm，最近水 0.26–0.31 nm |
| u_kn 结构异常 | ❌ 两条腿结构**相同**：每帧严格 λ 线性（比值 10.994 / 10.998，σ ≈ 1e-3），σ/mean 都是 16–18% |
| 实空间裸库仑和当探针 | ⚠️ **没用**：被离子穿越 cutoff 主导（±567 / ±958 kJ/mol），信噪比不够 |

结构正常 ⟹ 问题在**哈密顿量或系综**，不在估计量。于是去比**参数计数**（不只是电荷）：

```
溶剂腿 System 里配体的成键项      bond   angle   PeriodicTorsion
  膜运行                          19     0       0
  可溶                            19     71      0        ← 差在 angle
```

---

## 3. 根因 P0-13：`next()` 对同类型力做了单例假设

`runabfe.generate_ligand_xml_from_top` 原先：

```python
angle_force = next((f for f in extracted_system.getForces()
                    if isinstance(f, openmm.HarmonicAngleForce)), None)
```

而膜体系的 System 里有**两个** `HarmonicAngleForce`（实测）：

```
force[2]  31401 个角，配体 0 个    ← next() 抓到的是这个
force[4]     71 个角，配体 71 个
可溶体系只有 1 个角力（配体那 71 个混在里面）⟹ 这个 bug 侥幸一直没被踩到
```

⟹ `ligand_only.xml` 的 `<HarmonicAngleForce>` 段是**空的** ⟹ 溶剂腿配体**无键角** ⟹
分子是软的。因果链每一环都有实测：

```
无键角 ⟹ 预平衡 0.996 → 0.660 nm 塌缩，12 个 replica 再没恢复（σ = 0.005 nm）
      ⟹ 极性基团聚拢，配体–水静电耦合强 3 倍（⟨U⟩ −569 ± 90 vs −190 ± 34 kJ/mol）
      ⟹ 去电荷 62.80 → 191.05 kJ/mol
      ⟹ ΔG_bind = +23.27 kcal/mol
```

对照四个数（重原子最大内距，去电荷 replica 600 帧汇总）：

| | 复合物腿 | 溶剂腿 |
| --- | --- | --- |
| 膜运行 | 1.28 nm（口袋撑着） | **0.66 nm** |
| 可溶生产 | 1.39 nm | 1.10 nm（p5–p95 = 0.733–1.391，有分布）|

**修法两层**：
1. bond / angle / torsion 三类都遍历**所有**同类型力；`NonbondedForce` 多于一个时
   直接报错让人收口拓扑（不猜）。已 grep 全仓：其余地方都是
   `for force in getForces()` 循环或显式列表，只有这一处做单例假设。
2. **写完就对账**（关键）：写出的项数必须与源体系里配体的项数逐项相等；
   "多原子配体 0 个键角"直接 fail closed。事故本该在 0.1 秒内被拦住。

实测修复后同一份 `memtest/topol.top`：`<Angle>` **0 → 71**，
对账 `bond=41 angle=71 torsion=104` 通过。

---

## 4. 同批的另两件

* **B6-FIX**：色散判据原先是一个**没有环境维度**的全局布尔，于是"膜口袋局域密度不均匀"
  这个对复合物腿正确的理由被套到**纯水溶剂腿**上，把那条腿**合法**的 bulk-water
  尾项修正一起关了（解释 vanishing 的 −13.1）。改成
  **目标（`dispersion_protocol`）× 该腿环境 → 实现**，唯一实现
  `abfe_core.resolve_leg_dispersion_implementation`。膜复合物腿行为不变但如实记
  `target_met=false`（真正达成要 §1.3 路线 C，未实现）。
* **P0-12a/b（检测层）**：两条腿逐 λ 记 Rg / 重原子最大内距 / 内部极性接触
  （§3.0 末条要求但从未实现）+ 逐 replica 均值；**跨腿构象一致性门**接在
  `combine_binding_free_energy`（循环闭合唯一实现），两腿 [p5, p95] 不相交即不许汇总
  ΔG_bind；溶剂腿缓存身份加**起始构象**指纹，`SOLVENT_CACHE_PROTOCOL_VERSION` 4 → 5。

---

## 5. 被撤销的方案（别再提）

**P0-12c 采样层（双起点验证 / 加构象采样维度 / 构象限制 + 解析修正）整条撤销。**
我一度判断"溶剂腿构象采样没收敛，需要双起点验证"，甚至考虑给溶剂腿加构象采样维度——
**方向错了**：根因是哈密顿量缺项，参数修对之后没有理由再塌缩。
⚠️ 只有在"修完重跑**仍然**塌缩"时才回到这一条，届时先跑双起点判性质，
不要直接上构象限制（那会改热力学循环）。

另外两个**不是** bug、别去"修"的：
* 那轮日志里的 `[膜质量门 · advisory]` —— provenance 记的是
  `config.membrane_quality_gate = advisory` 且 `config file = None`，说明那次是**命令行**
  跑的 advisory，**不是**配置文件被忽略。`memtest/abfe_config.json` 里一直是 `enforce`。
* `ligand_only.xml` 不在 `resolve_ligand_ffxml` 的候选名单里 ⟹ 每次都用当前生成器重写，
  **不存在复用旧 angle-less XML 的陷阱**。

---

## 6. 重跑：怎么跑、会作废什么、看哪三个数

**保留**：那 100 ns 预平衡（5.0 h）—— `_pre_equilibration_fingerprint` 的输入是
system/ligand/温度/压力/坐标/盒/步数/barostat，**不含 code hash**，已确认。
**重做**：attachment + Stage 1/2 + 溶剂腿 ≈ **2.1 h** —— `_stage_protocol_key` 含
`code_sha256`，今天改了代码必然作废；溶剂盒也会重建（缓存版本 4 → 5），
并用修好的生成器重写 `ligand_only.xml`。

```bash
cd /home/ruigengji/ABFE_IBS/Atenolol-rank11/memtest
source /home/ruigengji/mambaforge/etc/profile.d/mamba.sh
mamba activate openmm_dev
python ../runabfe.py --config abfe_config.json
```

⚠️ **不要加 `--membrane-quality-gate advisory`**（上轮是这么跑的）。配置里已是
`enforce`，而门在这条 100 ns 轨迹上已经通过过。

跑完先看三个数（都在 `final_results.json`）：

1. `ligand_conformer_diagnostics.per_replica_mean_max_internal_heavy_distance_nm`
   —— 溶剂腿是否还锁在 0.66 附近（修好后应当回到 1.0–1.4 且有分布）；
2. 溶剂腿 `decharging` 是否回到 **~63 kJ/mol** 量级（可溶基线 62.80）；
3. `ligand_conformer_cross_leg.passed` —— 为 false 时**不会**汇总 ΔG_bind，
   那时才回到第 5 节那条（先双起点判性质）。

验证命令（CPU，改完代码必跑）：

```bash
./tests/run_offline_tests.sh -q          # 预期 1036 passed / 2 skipped / 0 failed
python -m pytest tests/test_ligand_xml_extraction.py tests/test_ligand_conformer_gate.py -v
```

⚠️ 跑全套时**不要同时改生产 `.py`**：本仓库大量源码/AST 契约测试走
`inspect.getsource`（经 `linecache` 读 import 时的行号），跑到一半重写文件会报假失败。

---

## 7. 下一位接手者的禁区

1. **不要把三类成键力的聚合改回 `next(...)`**，也不要把写出/源体系的**逐项对账**降级成
   warning。那两条是这次事故唯一能被早期发现的地方。
2. **不要放宽跨腿构象门的百分位区间**（[p5, p95] → [p0, p100]）或改成"均值差 < 某个 nm"。
   不重叠的正解是修根因或修采样。
3. **不要因为"膜复合物腿 target_met=false 看着难受"就给它开均匀密度 LRC** ——
   口袋里那个假设不成立，正解是 §1.3 路线 C。
4. **可溶生产基线不受本轮影响**（它只有一个角力，配体参数一直是全的）：
   181.00 / 157.84 / −5.535906 kcal/mol 仍然有效，别去重跑它来"验证"。
5. 主线仍是 §17.0 的 ③ **B4 溶剂腿 builder**（charge-transfer）。B4 会走同一个 XML
   抽取路径，P0-13 的对账与 P0-12a 的门都已经在它之前落地了。
