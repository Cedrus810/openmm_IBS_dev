# Boresch 二面角符号反号事故 —— 诊断、修复、遗留项

日期：2026-07-29
状态：**根因已修复并实测确认**；遗留三项未动（见最后一节），等溶剂盒扫描结果再决定。

---

## 1. 现象

02:09 的 attachment 腿抛错退出：

```
λ=1.0000  ⟨U_Boresch⟩=   25.312 ±  7.598  max=   47.424 kJ/mol
λ=0.3500  ⟨U_Boresch⟩=   60.124 ± 24.174  max=  167.173 kJ/mol
λ=0.1000  ⟨U_Boresch⟩=  187.959 ± 62.348  max=  487.579 kJ/mol
λ=0.0000  ⟨U_Boresch⟩=  776.958 ± 67.050  max= 1114.958 kJ/mol
BAR(主) 98.7551 ± 2.7096 | TI 107.0230 | MBAR(诊断) 98.9139
RuntimeError: attachment 腿 BAR(98.7551) 与 TI(107.0230) 差 8.2679 kJ/mol，容差 8.1288
```

**BAR/TI 门失败是症状，不是病。** 门的报错文案说「被少数稀有大能量帧支配（多半是
二面角反转）」，在这里是误导：不是少数帧，是**整个 λ=0 系综**都坐在 `k(1−cosΔ)`、
Δ≈π 的势壁顶上 —— `mean=777` 而 `std` 只有 67。

λ 阶梯（4 态）也没问题：07-28 用同一张表跑出 BAR/TI 差 0.12 kJ/mol。**不要去加密 λ**，
那是给一个健康的估计量打补丁。

---

## 2. 根因

`abfe_core.py` 里有**四份**手写二面角副本，都写成

```python
m1 = np.cross(n1, b2 / np.linalg.norm(b2))
return np.arctan2(np.dot(m1, n2), np.dot(n1, n2))
```

标量三重积 `(n1×b2̂)·n2 = det[n1, b2̂, n2] = −(n1×n2)·b2̂`，所以这四份返回的是 **−φ**。
距离和键角不受影响（`arccos` 无符号），**只有三个二面角整体反号** —— 即限制势参考
几何的**镜像**。

可纸笔复核的例子：`a=(0,1,0) b=(0,0,0) c=(1,0,0) d=(1,0,1)`
→ IUPAC `+π/2`，这个公式给 `−π/2`。

Boresch 参考值 `phiA0/phiB0/phiC0` 是喂给 `LambdaDependentBoreschForce` 表达式里
OpenMM 的 `dihedral(p1,p2,p3,p4)` 的，而 OpenMM 与 `mdtraj.compute_dihedrals` 用的
都是标准（IUPAC）约定。

### 证据链（全部来自落盘数据）

| | phiA0 | phiB0 | phiC0 | r0 | θA | θB |
|---|---|---|---|---|---|---|
| `boresch_simple.json`（mdtraj 算，500 帧均值） | **+1.6696** | **−1.8045** | **−0.6839** | 0.4560 | 1.4767 | 1.8857 |
| `checkpoints/boresch_equilibrium_committed.json`（02:02:29） | **−1.7168** | **+1.8136** | **+0.6163** | 0.4476 | 1.5102 | 1.9002 |
| σ = √(kT/k) | 0.107 | 0.106 | 0.139 | — | — | — |

三个二面角整体反号，**模长各差 0.047 / 0.009 / 0.068 rad，全部 < 0.5σ**；无符号的
r0/θA/θB 差 < 0.03。不是构象变了，是同一个 pose 的镜像。

### 时间线（`output_lrc_fix/pipeline.log`）

1. `02:02:01–02:02:28` 带 Boresch 的 rebalance，用 `boresch_simple.json` 的参考值
   （mdtraj 约定，**正确**）。
2. `02:02:29` `update_boresch_from_last_frame` → `calc_boresch_from_last_frame`
   （反号那份）重算并**覆盖**参考值 → 提交下去的三个 φ0 全反号。
3. `02:02:32` attachment 腿拿这组镜像参考值开跑。

**顺带证明了 OpenMM 的 `dihedral()` 就是 mdtraj/IUPAC 约定**：第 1 步是 OpenMM 自己的
`dihedral()` 在拉配体，它产生的 pose 用这个手写函数量出来恰好等于 −(参考值)。也就是说
在 IUPAC 约定下 pose = +(参考值)，限制力**完美地**把配体按在了参考几何上。所以是手写
函数错，不是 force 错。

### 量级核对

用 committed 参考值 + 实测 pose 直接算：

```
kphiA(1−cos 3.386) = 216.7×1.971 = 427
kphiB(1−cos 3.618) = 223.0×1.888 = 421
kphiC(1−cos 1.300) = 130.1×0.732 =  95   → Σ ≈ 943 kJ/mol
```

实测 λ=0 `⟨U_B⟩=777`、`max=1115 ≈ Σ2k_φ=1140`。对得上。

---

## 3. 为什么以前没被抓到

| 防护 | 为什么漏了 |
|---|---|
| `update_boresch_from_last_frame` 的两道强校验 | 只看 `θ∈[40°,140°]` 和 `|Δr0|<2.5 Å`，**从不看二面角**。本次 phiA0 偏约 27σ 照样放行 |
| `boresch_committed_deviation_sigma` | 会看二面角，但**只在 resume 分支跑**（本次 `Resume: False` 走 else），且它的 `current_eq` 也是同一个反号函数算的 —— 错的比错的，永远一致 |
| `test_boresch_attachment_leg.py::_dihedral_rad` | 是同一个错公式的第 5 份拷贝，跟生产代码「自洽」，零分辨力 |
| `runabfe.py:1770-1775` 的 v2 补丁 | 只让 `simple`/`fluctuation` 分支**不调用**这个函数（绕开），没改函数本身。于是 `abfe_pipeline.py:3338` 这条完全另一条路上的调用照样中毒 |

第 4 条是关键教训：**当时修的是「不再调用」而不是「符号改对」**，所以留下了两个活的
注入点（`runabfe.py:1777` 的 `auto`/`orb_simple` 分支，以及 `abfe_pipeline.py:3338`）。

---

## 4. 已做的改动

- **`abfe_core.py`**：新增模块级 `boresch_dihedral_rad()`（标准符号 + 完整事故
  docstring），把四份手写副本全部替换成它：

  | 原行号 | 所在函数 |
  |---|---|
  | 1962 | `OrbBoreschEstimator.estimate_from_trajectory` |
  | 2029 | `OrbBoreschEstimator._finalize_candidate` |
  | 2837 | `scan_boresch_1d_pes._calc_geom` |
  | 3375 | `calc_boresch_from_last_frame` ← 本次的凶手 |

  这样三个注入点一次修完：`abfe_pipeline.py:3338`、`runabfe.py:1777`、
  `abfe_pipeline.py:3263`（resume 校验的 `current_eq`）。

- **`test_boresch_attachment_leg.py`**：`_dihedral_rad = boresch_dihedral_rad`，
  不再自带公式。

- **`test_boresch_dihedral_convention.py`**（新，8 条 `cpu_only`，6 粒子，不跑动力学）。
  原则：**绝不自己写二面角公式**，只拿 OpenMM/mdtraj 当基准。
  1. 手算基准 → `+π/2`
  2. 三个 Boresch 四元组逐个与 **OpenMM 自己的 `dihedral()`** 比（把能量表达式直接
     写成那个角度，基准完全来自 OpenMM，对「两边同时错号」免疫）
  3. 与 **mdtraj** 比
  4. fixture 判别力自检：三个 φ 的 `|sin|` 必须 > 0.2
  5. **端到端**：参考值取自某构象 ⟹ 该构象上 `U_Boresch` ≈ 0
  6. 反向：只把三个 φ0 取反 ⟹ `U_Boresch` 必须暴涨到 `Σ2k_φ`

  fixture 用独立纯 stdlib 实现交叉验算过：

  ```
  hand case = +1.5707963267948966  (= π/2)
  r0=0.4848 nm  θA=63.4°  θB=109.0°
  phiA0=-1.8638  |sin|=0.957  mirror= 397.3 kJ/mol
  phiB0=+1.0192  |sin|=0.852  mirror= 323.5
  phiC0=+1.2000  |sin|=0.932  mirror= 226.0
  mirror total = 946.8 kJ/mol
  ```

  946.8 与按真实体系算出的 943 同量级。
  ⚠️ 第一版 L2 取 `(0.55,−0.38,0.33)` 是坏的 —— φC = −3.105 rad，离 ±π 只有 2°，
  镜像代价塌到 0.36 kJ/mol，那条断言等于空转。现在的 L2 是反解出来的（注释里有构造）。

- **`test_core_physics_numerics.py`**：`ess_gate_protocol_version` 钉子 2 → 3。
  与二面角无关：`ibs_engine.py:11505` 在 07-29 05:30 被 bump 到 3
  （`v3: occupancy 与 warmup 协议统一为诊断项`），而这个测试写于 07-27。
  v3 只取消了「最终 stage 在全部采样完成后用 occupancy 反向否决」
  （`ibs_engine.py:12395-12407`），warmup 的 production-entry 占据门
  （`ibs_engine.py:11361`）没变，本测试的语义断言与那次改动正交，故只改版本号。

---

## 5. 修复后的实测（08:09，同锚点同 λ 表同 250 帧/态）

| λ | 修复前 (02:09) | 修复后 (08:09) |
|---|---|---|
| 1.00 | 25.312 ± 7.598 | **3.268 ± 2.061** |
| 0.35 | 60.124 ± 24.174 | **4.525 ± 2.705** |
| 0.10 | 187.959 ± 62.348 | **5.663 ± 3.484** |
| 0.00 | 776.958 ± 67.050 | **7.237 ± 5.222** |
| BAR / TI | 98.7551 / 107.0230（差 **8.27**，门 8.13 ✗） | 4.3889 / 4.4509（差 **0.062**，门 1.0 ✓） |

`ΔG(A′→A) = 4.3889 ± 0.0779 kJ/mol = 1.0490 kcal/mol`，二面角反转告警消失。

`⟨U_B⟩` 现在全程 1.3–2.9 kT —— `DEFAULT_BORESCH_ATTACHMENT_LAMBDAS` 那段注释断言的
「配体在口袋里被蛋白按住、U_B 处处 1.6–5.6 kT」终于真的成立，4 态阶梯的前提恢复。

参考几何也自洽了（committed vs mdtraj 系综均值，**六个自由度全部同号且亚 σ**，
之前三个 φ 是约 27σ 反号）：

```
   coord simple(mdtraj)    committed      sigma dev/sigma
       r       0.4560       0.4476     0.0353     0.24
  thetaA       1.4767       1.5102     0.0940     0.36
  thetaB       1.8857       1.9002     0.0955     0.15
    phiA       1.6696       1.7168     0.1073     0.44
    phiB      -1.8045      -1.8136     0.1058     0.09
    phiC      -0.6839      -0.6163     0.1385     0.49
```

---

## 6. 复合物腿历史对比

| 时间 | attach | charge | vdw | decoupling | 解析释放 | ΔG_complex (kJ/mol) |
|---|---|---|---|---|---|---|
| 07-06 | — | — | — | 231.33 | −36.51 | 192.89 |
| 07-28 HREMD（已弃） | 38.60 ± 110 | 64.41\* | 146.82\* | 249.84 | — | — |
| 07-28 顺序窗口 | 5.74 | 64.41\* | 146.82\* | 216.98 | — | — |
| 07-28 final | 5.54 | 64.41\* | 146.82\* | 216.78 | −37.65 | 177.20 |
| 07-29 02:09 | **98.76** | — | — | — | — | 崩 |
| **07-29 10:18（全新）** | **4.39** | **74.47** | **142.83** | **221.69** | **−38.76** | **180.998 ± 1.762** |

`*` = 07-28 是 attachment-only，charge/vdw 是更早的缓存值。

**charge 64.41 → 74.47（+10.06）、vdw 146.82 → 142.83（−3.99）是预期的、不是新问题**：
那两个缓存值是在镜像参考几何下采的样，`boresch_committed_deviation_sigma` 的 docstring
早就写过这个失效模式（「限制力把配体从自己的 pose 上拽走 3.4 Å…复合物腿去电荷因此偏低」），
方向完全对得上。

⚠️ **07-28 的 216.78 / 177.20 应当作废** —— 它的 stage1/stage2 是错姿态下的采样。

---

## 7. 最终结果（11:02，溶剂腿已补跑）

```
复合物腿 ΔG_cplx   = 181.00 ± 1.76 kJ/mol
溶剂腿   ΔG_solv   = 157.84 ± 1.79 kJ/mol
Boresch attachment  =   4.39 ± 0.08
Boresch 解析修正    = -38.76
APBS               =   0.00
─────────────────────────────────────
ΔG_bind = -23.16 ± 2.51 kJ/mol = -5.54 ± 0.60 kcal/mol
参考 result.txt total   = -6.279 ± 0.457 kcal/mol
差 +0.74，合并 σ 0.754  → 0.98σ   ✅ 1σ 内一致
```

对比 07-06（符号 bug 期）：−9.76 kcal/mol，偏参考 −3.48。**这 2.7 kcal/mol 的改善
基本全部来自本次二面角符号修复**（复合物腿 192.89 → 180.998）。

⚠️ 但 ±0.60 是乐观的，真实值约 **±1.0 kcal/mol**，见 7.1 与第 8 节。

⚠️ `result.txt` 是**旧方法**的参考值，本仓库实现的是 IBS。**只有 total 可以拿来对照，
分项不可比** —— 见 7.2，那里记着一次因为把参考分项当真值而推出整套错误排查计划的教训。

### 7.1 ~~盒子尺寸依赖~~ → 实际是 vanishing 腿的跑间散布（**修正早先结论**）

**早先在本文档里写过「有限尺寸偏差主要在 LJ 腿上」，那个结论是错的，已作废。**

11:02 这次溶剂腿用的是**与 `solvent_box_scan/pad_1.5000` 完全同一个盒子**
（`box_edge_nm = 4.257`、`padding_nm = 1.5`、Na=7 Cl=7；`output_lrc_fix` 的
`solvent_cache_manifest.json` 与 `system_solvent.xml` 自 01:18 起未变，
`SOLVENT_PADDING_NM` 仍是 1.5）：

| 运行 | decharging | vanishing | 总计 |
|---|---|---|---|
| pad 1.5 scan（07-28） | 63.115 ± 1.104 | **101.639** ± 1.100 | 162.826 ± 1.559 |
| pad 1.5 主跑（07-29 11:02） | 62.800 ± 0.671 | **96.964** ± 1.663 | 157.836 ± 1.793 |
| pad 2.4 scan（07-28） | 64.249 ± 1.078 | **94.491** ± 1.431 | 156.812 ± 1.792 |

```
同盒子 decharging: Δ=+0.315  σ_diff=1.292  → 0.24σ    可复现
同盒子 vanishing : Δ=+4.675  σ_diff=1.994  → 2.34σ    不可复现
跨盒子 decharging: Δ=-1.449  σ_diff=1.269  → 1.14σ
跨盒子 vanishing : Δ=+2.473  σ_diff=2.194  → 1.13σ
```

**同盒子的 vanishing 散布（2.34σ）比跨盒子差异（1.13σ）还大**，所以 pad1.5→pad2.4
那 −7.15 kJ/mol 不是有限尺寸效应，而是 vanishing 腿的跑间散布。这与两条腿都报过的
split-half 告警一致：**vanishing 的报出误差棒系统性偏小**。

推论：**加跑 pad 3.0 单独一档解决不了问题**。要分离散布与尺寸效应，必须在固定 padding
下做重复跑；更值得先做的是修 vanishing 的采样/误差估计。

`decharging` 反过来是干净的（同盒子 0.24σ，跨盒子 1.14σ），所以问题定位在 vanishing。

### 7.2 与 `result.txt` 的对照口径：**只有 total 可比，分项不可比**

> ⚠️ 本节是对本文档早先版本的更正。早先这里写的是「溶剂腿去电荷比参考低 5.3 kJ/mol，
> 是已识别的最大单项系统偏差，值得单独查」，还据此在遗留项里排了排查计划。
> **那是错的，已作废**，理由有两条，都很基本：

**（一）`result.txt` 是旧方法的参考值，本仓库实现的是 IBS —— 另一个方法。**
两者的分项拆法不同（例如参考是 annihilation、我们是 decoupling），分项数值本来就
不该逐项对上。**把参考的每一项当成真值、再去追每一项的差**，是方法论上的错误，
不要重犯。可比的只有 total。

**（二）restraint 项与被限制那条腿的采样是同一件事的两面，按热力学循环必须抵消，
所以单独比 restraint 项没有意义。** 我们的力常数是涨落反推的、比参考硬得多
（kr raw 7355.9，kθ/kφ 130–282），restraint 项因此必然比参考大；这个差额本该被
复合物腿的 charging/annihilation 吃掉。用「对结合的贡献」口径（kcal/mol）实测：

| | 我们 | 参考 | 差 |
|---|---|---|---|
| A = charging | −2.789 | −1.680 | −1.109 |
| B = annihilation | −10.962 | −11.016 | +0.054 |
| A + B（含参考单列的 LRC −0.191） | −13.751 | −12.887 | **−0.864** |
| C = restraint（= −attach + 解析释放） | +8.215 | +6.608 | **+1.607** |
| total | −5.536 | −6.279 | **+0.743** |

C 偏正 1.607、A+B 偏负 0.864 —— **抵消约 74%**。所以 A、C 各自的偏差不是两个独立
错误，把它们当成两条待查线索是重复计数。

**而且 +0.743 小于我们自己的诚实误差棒（±1.0 kcal/mol，见第 8 节），
所以「抵消不完整」目前无法与采样噪声区分，不构成任何已确立的系统偏差。**
真正该先做的是把 vanishing 的误差估计修好（7.1 + 第 8 节），而不是去查分项。

### 7.3 参考的 annihilation 分项其实是可比的（更正）

> ⚠️ 早先这里写「annihilation 项不可比」，**不准确**。

**合并项完全可比，且吻合到 0.054 kcal/mol**（我们 −10.962 vs 参考 −11.016，
远在参考自己的 ±0.402 之内）。

看着不可比只是因为**参考的 per-leg 值等于我们的原始功减去一个共同偏移**：
反解 `annihilation-complex −9.630 = −(34.137 − X) → X = 24.507`、
`annihilation-lig −1.386 = +(23.175 − Y) → Y = 24.561`，X ≈ Y ≈ 24.5 kcal/mol。
那是分子内非键/自能项 —— 参考做 annihilation（连分子内一起去掉），我们做 decoupling
（保留分子内），该项两边环境相同，在合并时抵消。X 与 Y 相差 0.054 就是合并项的吻合度。

charging 没有这个偏移（复合物侧 17.954 vs 17.799，配体侧 16.274 vs 15.010），
但见 7.2（一）：分项对不上本身不说明谁错。

### 7.4 溶剂腿与 Boresch 无关（供后来者）

溶剂腿里没有蛋白、没有 Boresch（`boresch_correction_kJ_mol = 0.0`），本次符号修复
碰不到它，**不需要因符号修复而重跑溶剂腿**。

另外记一笔以免后来者误用：`output/solvent_leg`（07-06，152.05 ± 0.93）的 LRC 是
`status = not_implemented, applied = False, version = None`，与 v3 的复合物腿**不同协议
世代，不可配对**。拿它配会得到好看的 −6.92 kcal/mol，那是假吻合 —— `output_lrc_fix`
这个目录名就是为此存在的。可用的只有 LRC v3 那几个。

---

## 8. 误差棒口径

```
⚠️ [split-half] vanishing 前后半程不一致：window 5 漂移 -0.559 kJ/mol = 4.46×2σ（σ_win=0.063）
ℹ️ [P1-19] vanishing 若按 σ≥|漂移|/2 定下界：4/6 个窗口的 σ 被抬高，总 σ 1.6461 → 2.4766 kJ/mol (×1.50)
```

代回复合物腿总误差：

```
√(2.4766² + (1.7616² − 1.6461²)) = 2.55 kJ/mol = 0.61 kcal/mol
```

而不是落盘的 ±1.7616 / ±0.42。

**7.1 的重复跑给出了独立、更强的证据**：同一个盒子两次独立跑，vanishing 差
4.675 kJ/mol，即单次 σ ≈ 4.675/√2 = **3.31 kJ/mol**，而报出的是 1.10 / 1.66 ——
**低估约 2–3 倍**。据此重算：

| | 报出 | 诚实值 |
|---|---|---|
| 溶剂腿 | 1.793 | √(0.671² + 3.31²) = **3.38** |
| 复合物腿 | 1.762 | **2.55**（P1-19 下界口径，见上） |
| ΔG_bind | 2.514 kJ/mol = 0.60 kcal/mol | √(2.55² + 3.38²) = **4.23 kJ/mol = 1.01 kcal/mol** |

即 **ΔG_bind = −5.54 ± 1.0 kcal/mol**（vs 参考 −6.279 ± 0.457 → 0.67σ，仍一致）。

attachment 腿的 split-half 漂移 −0.4457 也只差容差（0.5000，撞在绝对下限上，
6σ 才 0.467）11% —— 250 帧/态偏紧。

**建议**：对外报数用 **±1.0 kcal/mol**，不要用落盘的 ±0.60。要真正收紧就得加长
vanishing 的采样并做重复跑确认，**不是改口径**。

---

## 9. 遗留项（本次刻意未动）

### 9.1 `update_boresch_from_last_frame` 的二面角门

`abfe_pipeline.py:3328` 的两道强校验仍然只看 θ 和 r0。建议复用现成的
`boresch_committed_deviation_sigma`（`abfe_pipeline.py:197`，同文件模块级，直接可调）
比较「新推导的 `new_eq`」与「它要覆盖的 `orig_eq`」，阈值沿用现成常量
`BORESCH_COMMITTED_MAX_DEVIATION_SIGMA = 4.0` / `..._WARN_... = 2.5`，二面角先过
`_wrap_to_pi`。

**超限行为应为「告警 + 保留 `orig_eq`」而不是 raise**，理由：

- 与同一函数已有两道门风格一致（`self._log(...)` + `return boresch_params`）。
- `orig_eq` 来自 `boresch_simple.json` 的 500 帧系综均值，本来就比单帧重锚可靠；
  退回它是严格更优，不是「放过一个错值」。
- 4σ 在 6 个自由度上误报率约 2.8%（见 `BORESCH_COMMITTED_WARN_DEVIATION_SIGMA`
  上方推导），硬门会以约 1/36 概率无故杀掉一次 9 小时的生产跑。真正的守门人是
  `test_boresch_dihedral_convention.py`。

测试就近扩进 `test_boresch_committed_gate.py`（已 import 该函数、同套阈值常量）。

### 9.2 `BORESCH_GEOMETRY_CONVENTION_VERSION` 2 → 3

`runabfe.py:1633`。v2 缓存的 `simple`/`fluctuation` 文件本身没问题（实测
`output_lrc_fix/boresch_simple.json` 的 `equilibrium_values` 与
`diagnostics.fluctuation_distribution.mean` 逐位相等），但它们生成时函数还是错的，
而 `auto`/`orb_simple` 分支（`runabfe.py:1777`）当时仍用反号值覆盖。升版把那两类旧
缓存一并刷掉；`runabfe.py:1697-1707` 已有版本不匹配就重新生成的逻辑，只改常量即可。

### 9.3 vanishing 腿的误差估计与复现性（**优先级高于盒子尺寸**）

7.1 已证明：同盒子两次独立跑，vanishing 差 4.675 kJ/mol（2.34σ），比跨盒子差异
（1.13σ）还大。所以**先别再扫盒子**——`--padding 3.0` 单跑一档分不清散布与尺寸效应，
纯属浪费 2–3 h。

按重要性排：

1. **在固定 padding 1.5 下再跑 1–2 次重复**，把 vanishing 的真实跑间 σ 钉下来
   （现在只有 2 个样本，σ ≈ 3.31 是极粗的估计）。这也是判断后续任何盒子扫描结果
   是否显著的唯一基准。
2. **查 vanishing 的报出 σ 为什么低估 2–3 倍。** 已有两条独立线索指向同一处：
   split-half（window 5 漂移 4.46×2σ）与跨跑散布。P1-19 那条
   「σ ≥ |漂移|/2 下界」目前是 `默认未采用`，值得重新评估是否该默认启用。
3. 只有在 1 做完之后，再决定要不要为盒子尺寸加档；此时才有能力判断 pad 2.4 与
   pad 1.5 的差异是否真实。若最终要改生产默认，改 `runabfe.py:101` 的
   `SOLVENT_PADDING_NM`（目前 1.5）；改后 `solvent_cache_manifest.json` 的
   `identity.padding_nm` 会不匹配并自动重建溶剂缓存，**无需手工删文件、更不要把扫描
   目录的 `final_results.json` 拷进 `output_lrc_fix/solvent_leg/`**（padding 与该目录
   manifest 不一致，属于改产物不改生成器）。

### 9.4 ~~溶剂腿去电荷比参考低 5.3 kJ/mol，值得优先排查~~ —— **已撤销**

本文档早先版本在这里排了一条排查计划（PME self-energy / 共炼金反离子 / 净电荷修正）。
**那条计划是从一个方法论错误里推出来的，已撤销，不要照着做。**

错在把 `result.txt`（旧方法的参考值）的分项当成真值。本仓库实现的是 IBS，分项拆法
本来就不同，分项对不上不构成偏差；而且 restraint 与被限制腿的采样结构性抵消，
把 charging 的差和 restraint 的差当成两条线索是重复计数。详见 7.2。

真正的遗留项只有 9.1 / 9.2（两道 Boresch 防护）和 9.3（vanishing 的误差估计）。
9.3 之所以成立，是因为它的证据**完全来自内部复现性**（同盒子两次独立跑差 2.34σ、
split-half 漂移 4.46×2σ），不依赖任何与参考的比较。

---

## 10. 验证命令

```bash
mamba activate openmm_dev && cd /home/ruigengji/ABFE_IBS/Atenolol-rank11

# 符号约定 + 全部 CPU 回归
python -m pytest test_boresch_dihedral_convention.py -v
python -m pytest -m cpu_only -q
```

（`-m cpu_only` 已跑过一轮：285 项里只有 `ess_gate_protocol_version` 那一条失败，
已按第 4 节修正。注意该断言短路在版本号那一行，它后面几条
（`min_absolute_ess_threshold is None`、`min_absolute_ess_gate_retired_reason` 存在、
`raw_*` 四项非 None、`ess_gate_metric` 标签）那轮**没有执行过**，需要这次重跑确认。）
