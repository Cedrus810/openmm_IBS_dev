# 带电配体 charge-transfer：被约束的 carrier 在水体系里要补什么（推导与方案）

日期：2026-10-11　状态：🟡 **第 0、1 步已做（只读诊断 + PB 后处理），代码未改**（§5 的改动清单还没动；第 2 步采样验证未跑）　范围：**只管水体系（soluble）**，膜体系等水体系跑通后另写

起因：fragment-opt `MCL-1_ligand-1`（−1 e）rep1 用 `--run-unclosed-charge-transfer-diagnostics` 第一次跑通全流程，
`final_binding_results.json` 里 `charge_transfer_reservoir_correction = null`、汇总时按 0 计。
用户 10-11 定：先把水体系这一项想清楚、做出来，水的跑通了再做膜。

## 0. 结论先行

1. **不需要"释放到标准态"这一项。** 两条腿都是固定盒 NVT，carrier 的约束在每个 λ 态都完全相同，
   起点和终点都在。所以约束的体积项在每条腿内部就消掉了，两条腿盒子大小不同也没关系（§3.2）。
2. 现有接口 `validate_charge_transfer_reservoir_correction` 要的是"去耦端从约束释放到自由"的那一项（G_free − G_restrained）。
   **只在一端释放反而会引入伪项** kT·ln(V_box/V_well)，两条腿的伪项之差在 rep1 上约 5.1 kJ/mol（§3.2）。
   这份 schema 要改。
3. 真正剩下的是 carrier 在**被约束的位置上**充电，和"无限远处的体相"充电之间的差（§3.3），分三部分：
   - (a) 常数电势偏移（RIP/DSC 型）：在总电荷守恒的电荷转移里**严格相消**；
   - (b) carrier 与蛋白（MCL-1 净 +5 e）在有限距离上的屏蔽静电作用，以及周期像在非均匀介质里的位置相关残余；
   - (c) carrier 在井内的位置分布：井内 PMF 起伏小于 kT 时不贡献，否则已经含在 (b) 的系综平均里。
4. **大小**：(b) 的单极项按 Debye–Hückel 粗估是亚 kJ/mol 量级，但 carrier 一旦贴到 Lys/Arg 上就不止这个数（§3.4）。
   这只是量级估计，不是结论；要先用 rep1 的轨迹看 carrier 离蛋白多近（§4 第 0 步）。
   **10-11 第 0 步已做**：
   - 井内分布和理想均匀井一致，所以 (c) 看不出来；
   - 没有 Lys/Arg 接触，但最近的蛋白原子就是 Lys/Arg 的氮，距离 0.5–2 nm，所以 (b) 要用第 1 步的 PB 算。

   **10-11 第 1 步已做**（APBS，盐匹配）：MCL-1 L1 rep1 的 ΔG_bind 修正 ≈ **−0.07 kJ/mol**（δ_cplx = +0.073 ± 0.017，δ_solv ≈ 0），
   相对 ±5.3 kJ/mol 的统计误差可以忽略。不加盐屏蔽的上界是 2.65 kJ/mol。详见 §4 第 1 步结果。
5. **不改约束形式**（固定笛卡尔位移 + flat-bottom，k=100，r0=0.5）。这样 rep1 以及之后的 rep 都能用，修正可以纯后处理加上（§6）。
   PHY-03 里的 barostat 问题在水体系不存在：注入函数遇到 barostat 直接拒绝（§2）。

## 1. 范围与和 PHY-03 的关系

`docs/TODO.md` PHY-03 写在膜线下面，C4/C5 是膜线的验收（`docs/archive/MEMBRANE_LINE_PLAN_2026-08-11.md`）。
磷原子距离、膜中面、头基层这些项对水体系都不适用。本稿只回答水体系的问题：

- 现在缺的到底是哪一项；
- 怎么算；
- 用什么证据说它算对了；
- 代码要改哪几处。

## 2. 代码现状（2026-10-11 逐处读过）

| 事实 | 位置 |
|---|---|
| carrier 约束表达式 `0.5*k_ion*max(0, pointdistance(x1,y1,z1, x2+dx0, y2+dy0, z2+dz0) - r0_ion)^2`，**不含任何 λ** | `abfe_core.py:1519` |
| 约束在自己的 force group 6，"逐 λ 完全相同"，不进 u_kn 差值 | `abfe_core.py:1528`；`ibs_engine.py:3547` |
| 体系里有 barostat 就拒绝注入，box_model = `fixed_cartesian_displacement_nvt_only` | `ibs_engine.py:3515` 起；`abfe_core.py:1526` |
| 锚点 = 配体重原子里离配体质心最近的那个，两腿同一规则 | `abfe_core.py:1506` 起 |
| 放置门槛：离配体 ≥1.6 nm（初始）/ ≥1.2 nm（运行时保证），离蛋白重原子 ≥1.2 nm（初始） | `abfe_core.py:2540` 起 |
| λ_coul=1：配体满电、carrier 中性（保留 Cl 的 LJ）；λ_coul=0：配体去电、carrier −1 e；stage 2 全程 carrier 保持带电 | `ibs_engine.configure_charge_transfer_decharging`；rep1 spec |
| 修正接口只收外部测好的两腿 JSON，自己不算；仓库里没有任何代码产出它 | `abfe_core.py:801` |
| 带电 charge-transfer 不分水和膜，一律标 `charged_membrane_charge_transfer_requires_c4_and_c5` | `abfe_core.py:1055` |
| `ibs_engine._create_co_alchemical_ion_restraint` 的 docstring 说"井心随 barostat 缩放"。这话不对，但因为有 NVT 限制，实际不会触发 | `ibs_engine.py:1352` |

rep1 实测（`runs_lcr_s1long15/MCL-1_ligand-1/rep1`，T = 300 K）：

| | 复合物腿 | 溶剂腿 |
|---|---|---|
| carrier 粒子 / 锚点 | 35294 / 2455 | 4457 / 11 |
| \|d0\| | 3.33 nm | 3.01 nm |
| 盒体积 | 359.84 nm³（边长约 7.1 nm） | 46.54 nm³（边长约 3.6 nm） |
| 盐 | 33 Na⁺ / 37 Cl⁻（另有 1 个 carrier），约 0.17 M | 5 Na⁺ / 4 Cl⁻（另有 1 个 carrier），按盐对数（4 对）约 0.15 M |
| 溶质净电荷 | 蛋白 +5、配体 −1 | 配体 −1 |

## 3. 推导

### 3.1 每条腿算的是什么

记 X 为 carrier，R 为它的约束（每个 λ 都在），E ∈ {c, s} 为环境。每条腿的两端：

- 起点 A_E：配体满电 L^q，carrier 中性 X⁰[R]；
- 终点 B_E：配体去耦 L^∅（复合物腿另带 Boresch），carrier 带电 X^q[R]。

腿内 ΔG_E = G(B_E) − G(A_E)。把 X 的贡献单独拿出来，写成 X 在该环境、该约束下的自由能 g_E^0、g_E^q（其余溶质已固定在对应端点）。合成 ΔG_bind 时 X 留下的项是

    Δ_X = (g_c^q − g_c^0) − (g_s^q − g_s^0) ≡ ΔG_X(c) − ΔG_X(s)

其中 ΔG_X(E) 是"被约束的 carrier 在环境 E 里从 0 充到 q"的自由能。循环严格闭合，等价于 ΔG_X(c) = ΔG_X(s)
（这里不计有限尺寸项；有限尺寸项和配体去电那一半一起处理，见 §3.3）。

### 3.2 约束的体积项在腿内相消

g_E 里和约束有关的部分是 −kT ln ∫ exp[−β(U_R + W_E^{0/q})] d³r_X，其中 W 是 X 在该处的 PMF。
井内 W 平坦时，它等于 −kT ln V_well + 常数。

- V_well 在两端相同，所以 ΔG_X(E) 里的 V_well 抵消。按 rep1 参数算，V_well = 4π[r0³/3 + ∫(r0+x)² e^{−βkx²/2}dx] ≈ 1.5 nm³；但它反正会抵消。
- **盒体积根本不出现。** 两条腿的盒子一个 360 nm³、一个 47 nm³，不需要任何换算。
- **反过来，如果只在去耦端把 X 从约束释放到整个盒子**（现有 schema 的定义），每条腿会多出 −kT ln(V_box/V_well)。
  两腿之差是 kT ln(V_c/V_s) = 2.494 × ln(7.73) ≈ **5.1 kJ/mol**，这一项完全是伪的。
  要释放就得两端都释放，那样又抵消回去了，等于什么也没做。

结论：不需要"释放 / 标准态"这一项。剩下的只有 W 的不平坦和环境差。

### 3.3 剩下的项：位点充电 vs. 无限远体相充电

目标是热力学极限下的结合自由能：多出来的那个离子离一切都无限远，所以它的充电自由能在两条腿完全相同。
模拟里它被拴在有限距离上，还处在周期盒里。差别按来源分三部分：

**(a) 常数电势偏移：严格相消。**
Ewald（P 求和）下，模拟里的静电势相对宏观参考有一个与位置无关的常数偏移：溶剂四极项（DSC 型）和溶质积分电势项（RIP 型）。
电荷转移过程中，配体处的电荷变化是 Δq_L = −q，carrier 处是 Δq_X = +q。常数偏移 c 的贡献是 (−q)c + (+q)c = 0，对任何 c 都精确成立。
这一条是本稿自己推的。

更强的一点（10-11 补，同样是本稿的推理，没有做盒尺寸实验验证）：
- Rocklin 的 NET/USV/RIP/DSC 各项，都是"盒子净电荷 × 某个积分量"的形式。看 `apbs_correction._rocklin_snapshot_correction`：RIP = (Q_B·I_B − Q_A·I_A)/V，NET ∝ Q_B² − Q_A²。
- 那套公式里 Q 是溶质净电荷，因为它针对的是"中和等离子体"、没有显式抗衡离子的模拟。
- 本路线两条腿都带显式盐和抗衡离子，**整盒在每个 λ 都是电中性的**：system XML 总电荷 0.00；逐 λ 电荷守恒由 `ibs_engine.configure_charge_transfer_decharging` 检查。
  均匀电势偏移作用在整盒电荷上，乘积就是 0，所以这几项整个不存在，不只是常数偏移那部分抵消。
- 如果硬套等离子体口径（Q = 溶质 +4），会得到 Q·(I_X − I_L)/V ≠ 0。这个数不适用于本模拟，本稿不加。
- 与 `apbs_correction.py` 文档串"co-alchemical 路线不加 Rocklin 修正"（引 Wu & Biggin 2022）一致。那篇文献的结论【本稿未复核】。

**(b) 有限距离上的环境作用 + 位置相关的周期残余。**
- carrier 在复合物腿里离蛋白只有一两纳米，会感受到蛋白 +5 e 和局部带电残基的屏蔽场；溶剂腿里配体已去电，没有对应项。
- 另外，复合物里配体位点在低介电的蛋白内，carrier 位点在水里，两处周期像反应场不一样（USV 型）。
  这类周期像作用要穿过显式盐溶液，被屏蔽掉了：0.15 M 时 Debye 长度约 0.8 nm，盒边长 7.1 / 3.6 nm。第 1 步按 R_cut = 3 nm（约 4 个 Debye 长度）把附近的周期像显式放进 PB，更远的忽略。

**(c) 井内位置分布。**
carrier 带电后可能被蛋白往井的一侧拉。只要井内 W^q 的起伏小于 kT，就不贡献；否则 (b) 需要在井内按系综平均，不能只取井心。

### 3.4 修正量的定义与量级

每条腿的修正 = 宏观参考 − 模拟周期盒，对象是这条腿里的电荷转移过程：

    δ_E = W^∞(L: q→0；X: 0→q 于无限远体相) − W^per(L: q→0；X: 0→q 于约束位点)
    ΔG_bind 加上 δ_s − δ_c

  符号依据：rep1 的 ΔG_bind = ΔG_solv − ΔG_cplx（−18.62 − 12.32 = −30.94，与结果文件的 −30.95 一致），两腿方向都是耦合→去耦。
  第 1 步的实际算法：carrier 充电那一段按"先去配体电荷、再给 carrier 充电"的路径拆，配体电荷置 0，
  ΔU_X = q_X·φ_env(r_X) + [E(carrier 在全部溶质空腔里) − E(carrier 单独在溶剂里)]，
  δ_E = −½(⟨ΔU_X⟩_{λ_coul=1} + ⟨ΔU_X⟩_{λ_coul=0})（线性响应，两端帧平均）。

在连续介质线性响应下，δ_E 只需要四种 Green 函数作用项：配体自项、carrier 自项、蛋白–配体交叉项、蛋白–carrier 交叉项，每项都算周期和非周期两种。
X 在无限远时的自项两腿相同，蛋白–X 项为零。

量级粗估，只用来判断值不值得认真算，**不是结论**：
- 把蛋白当半径 a ≈ 2 nm、电荷 +5 的球，carrier 是 −1，εr = 78，0.15 M 盐（κ⁻¹ ≈ 0.78 nm）。按 Debye–Hückel：
  - 离表面 1.2 nm（放置门槛）时约 0.2 kJ/mol；
  - 贴到表面（r = 2.4 nm）时约 0.6 kJ/mol。
- 单极项以外，carrier 和一个 Lys/Arg 形成接触离子对，在水里可达几 kJ/mol。
  墙在平坦区外 0.32 nm 处才 2 kT，离蛋白 1.2 nm 放下的 carrier，付 2–3 kT 就能贴上去。

所以第一件事是看 rep1 里 carrier 实际离蛋白多近。

## 4. 怎么算（按顺序，每步决定下一步做不做）

**第 0 步：只读诊断（CPU，分钟级，用 rep1 现有轨迹）。**
- 对复合物腿里 carrier 带电的态（stage 1 的 λ_coul=0，以及 stage 2 各窗），统计：
  - carrier 到蛋白重原子的最小距离分布；
  - 离 Lys NZ / Arg NH* 小于 0.4 nm 的帧比例；
  - 约束能量大于 0 的帧比例（撞墙）。
- 溶剂腿统计撞墙比例。
- 这一步本身不判对错，只决定第 1 步要算到多细：从来不贴蛋白，就只算单极项；经常贴，就必须做井内系综平均。

> **第 0 步结果（2026-10-11，MCL-1 L1 rep1）**
> - 脚本：`experiments/2026-10-11_carrier_step0/step0_carrier_geometry.py`；结果：同目录 `step0_carrier_geometry.json`。
> - 帧来源：stage 1 两端各 150 帧；stage 2 共 17 个窗/段，每个 100 帧（坐标旁车）。帧之间有相关，下面的数只能看趋势。
>
> | | 撞墙比例 | ⟨E_rest⟩/kT | 离约束井心距离 中位/p95（nm） | 离蛋白重原子 最小 / p05 / 中位（nm） |
> |---|---|---|---|---|
> | 理想均匀井（同样的 k、r0、300 K，数值积分） | 0.656 | 0.472 | 0.568 / 0.826 | — |
> | 复合物 λ_coul=1（carrier 中性） | 0.61 | 0.50 | 0.57 / 0.84 | 1.05 / 1.30 / 1.86 |
> | 复合物 λ_coul=0（carrier −1） | 0.69 | 0.56 | 0.57 / 0.88 | 0.79 / 1.18 / 1.78 |
> | 复合物 stage 2 各窗（carrier −1） | 0.51–0.82 | 0.33–1.05 | 0.51–0.67 / 0.78–0.92 | 0.53–1.52 / 0.95–1.64 / 1.49–2.17 |
> | 溶剂 λ_coul=1 / 0 | 0.66 / 0.73 | 0.56 / 0.44 | 0.57–0.59 / 0.80–0.84 | — |
>
> - **(c) 井内位置项：看不出来。** 带电、不带电、两条腿的井内统计都和理想均匀井一致。"撞墙"六成多是均匀分布的正常结果，不是被拉过去。
> - **(b) 环境项：要算。**
>   - carrier 从没和 Lys NZ / Arg NE/NH 形成 0.4 nm 以内的接触（所有态比例都是 0）。
>   - 但离它最近的蛋白原子**几乎总是 Lys/Arg 的氮**：表里各行的最小距离和中位距离，跟到 Lys/Arg 氮的距离基本相同。λ=1 时就是这样，所以是位点几何决定的，不是带电以后才被吸过去。
>   - 带电后最小距离从 1.05 降到 0.79 nm，p05 从 1.30 降到 1.18 nm。方向符合吸引，但 150 帧不足以下结论。
>   - 一个 −1 离子在 0.15 M 盐水里离几个带正电侧链 0.8–2 nm：单个 Lys 按 Debye–Hückel 约 1 kJ/mol 以内，几个加起来可能到 kJ/mol 量级。所以第 1 步的 PB 必须做，而且要按帧平均，不能只取井心。
> - 溶剂腿 carrier 离配体最近 1.79–1.96 nm（盒边长 3.6 nm），终点配体去电，没有对应的环境项。

**第 1 步：δ_c、δ_s 的连续介质估计（CPU，后处理，不重跑）。**
- 取复合物腿端点若干帧，含 carrier 在井内的分布。电荷取自 `system_native.xml` / `system_solvent.xml`，盒子取自 `box_vectors*.npy`。
- 用周期和非周期两种边界解线性 PB，按 §3.4 的四项组装，帧平均并给 SEM。
- 介电与半径口径沿用 `apbs_correction.py`。只有蛋白–carrier 交叉项用模拟的离子强度：模拟里有显式盐，这一项按 0 盐会高估。
- PB 工具（10-11 用户指定）：现成二进制 `/home/ruigengji/software/APBS-3.4.1.Linux/bin/apbs`（APBS 3.4.1，`--version` 实测能跑）。不往任何 conda 环境里装。

> **第 1 步结果（2026-10-11，MCL-1 L1 rep1）**
> - 脚本：`experiments/2026-10-11_carrier_step1/step1_carrier_pb.py`。结果：同目录 `complex/summary.json`、`solvent/summary.json`；每帧的 `apbs.in` 和 `apbs.log.gz` 在各帧子目录。
> - 设置：
>   - APBS 3.4.1，线性 PB，mg-auto，dime 129；粗网格覆盖全部原子，细网格 24 Å、以 carrier 为中心（间距 0.19 Å）。
>   - pdie 1，sdie 97（TIP3P，与 `apbs_correction.py` 一致）。半径取 Rmin/2 = σ·2^(1/6)/2。
>   - 连续介质盐按模拟里的盐对数：复合物 0.168 M，溶剂 0.150 M。
>   - 显式水和显式离子都换成连续介质。carrier 3 nm 以内的溶质整分子周期像显式放进 PB（每帧 1–4 份拷贝）。
> - 帧：stage 1 两端各取 30 帧（每 5 帧取 1 帧）。帧之间有相关，下面的 ± 是朴素 SEM，偏乐观。
>
> | 复合物腿 | E_int = q_X·φ_env(r_X)（盐匹配） | E_int（0 M，对照） | E_desolv |
> |---|---|---|---|
> | λ_coul=1（30 帧） | −0.082 ± 0.016 kJ/mol | −2.41 ± 0.13 | +0.002 |
> | λ_coul=0（30 帧） | −0.067 ± 0.031 kJ/mol | −2.89 ± 0.14 | +0.002 |
>
> - δ_cplx = **+0.073 ± 0.017 kJ/mol**。单帧范围 −0.44 到 +0.67 kJ/mol。两端几乎一样，线性响应自洽。
> - 溶剂腿：配体去电后没有环境电荷，E_int 恒为 0；E_desolv 在 0.001 kJ/mol 以下。所以 δ_solv ≈ 0。
> - **对 ΔG_bind 的修正 = δ_s − δ_c ≈ −0.07 kJ/mol**，远小于 rep1 自身的 ±5.31 kJ/mol。
> - 0 M 对照（假如显式盐完全没有屏蔽）给出 δ_cplx ≈ +2.65 kJ/mol。这只是没有屏蔽时的上界，不是本模拟的物理。
>   本模拟里有 0.17 M 显式盐；盐匹配 PB 里蛋白场在 carrier 处只剩 0 M 时的约 3%，对应约 2 个 Debye 长度的距离。
> - **没覆盖到的**：
>   - stage 2 期间 carrier–蛋白作用随配体消失的漂移没算。E_int 本身只有 0.07 kJ/mol 量级，漂移只会更小。
>   - 周期项按 §3.3(a) 的电中性论证取 0，没有做盒尺寸实验。
>   - 只有一个体系、一个 rep。其他带电体系、尤其是蛋白净电荷更大的体系，要逐个算，不能照搬这个数。

**第 2 步：采样验证（GPU，按全局规则先报计划）。**
- 换位点：同一个 MCL-1 L1，换 2 个合法 carrier 位点，一个靠近蛋白（约 1.2 nm）、一个尽量远。复合物腿重跑 stage 1/2，溶剂腿不变。
- 判据：δ_CT 修正前，ΔG_bind 随位点变化；修正后，变化落在合并统计误差（2σ）以内。
- 这是"修正量算对了"的直接证据，判据来自统计误差，不新造阈值。
- 再加一组 k/r0 扫描（k=50/r0=0.7、k=200/r0=0.3）验证 §3.2：修正前后都应该不变。
- 耗时用 rep1 的实际耗时估，报计划时给数。

**不另设"可忽略"门槛**：每个带电结果都把 δ_CT 及其误差加进去、写进结果文件，不按大小决定加不加。

## 5. 代码改动清单（第 1 步出数之后再动）

1. **修正接口 schema v2**（`abfe_core.py:801` 起）：
   - 每条腿改成 `delta_X_kJ_mol`、`error_kJ_mol`、`method`（PB / 采样）和证据引用；
   - 去掉"去耦端释放"的定义（§3.2：只释放一端会引入伪项）；
   - `CHARGE_TRANSFER_RESERVOIR_CORRECTION_PROTOCOL_VERSION` 升到 2，v1 的 payload 拒收。
2. **离线工具**：读运行目录 → 产出 v2 JSON → `runabfe` 现有的 post-analysis 汇总路径把它读进去。改的只是汇总，不碰任何采样缓存。
3. **资格判定按水和膜分开**（`abfe_core.py:1055`）：水体系的原因改成 `soluble_charge_transfer_requires_carrier_correction`，不再背 C4/C5 的膜线标签。
   修正到位并通过第 2 步之后，水体系怎么升级为 production_qualified 由用户定。
4. 结果文件里 `external_corrections.charge_transfer_reservoir.status` 缺修正时写的是 `not_applicable`，应改成 `omitted`。
5. 订正 `ibs_engine.py:1352` 的 docstring（"井心随 barostat 缩放"）。
6. `docs/TODO.md` PHY-03 拆成水和膜两条；水的那条指向本稿，并订正 barostat 一点。

## 6. 对已有数据和在跑数据的影响

- 约束形式不变 ⟹ rep1 以及按现在这套跑出的 rep 都可以用：δ_CT 是逐体系加上去的一项，第 1 步完全从运行目录后处理。
- 什么情况下要重跑：只有决定换约束形式，比如改分数坐标或相对约束。本稿不建议换：水体系是固定盒 NVT，换形式的理由（barostat）不成立。
- 第 2 步的换位点运行是**额外**的验证运行，不替代已有的 rep。

## 7. 要用户定的事

1. ~~第 1 步的 PB 用什么~~ → 10-11 已定：用 `/home/ruigengji/software/APBS-3.4.1.Linux/bin/apbs`。
2. 第 2 步选哪个体系、几个位点、几个 rep，以及 GPU 时间。
3. 修正通过后，水体系的带电结果能不能从 `must_not_report_delta_g_bind` 升为可报告。

## 8. 参考

- Rocklin, Mobley, Dill, Hünenberger, *J. Chem. Phys.* 2013, 139, 184103（有限尺寸静电修正框架）。
- Wu & Biggin, *J. Chem. Theory Comput.* 2022（APBS/RIP 口径；推荐带电体系用 co-alchemical ion + charge transfer）。两篇都按 `docs/METHODS.md` 的引用，本稿未复核原文。
- OpenFE 的对照：RBFE 带电变换用 "alchemical water"（`openfe/protocols/openmm_rfe/_rfe_utils/topologyhelpers.py:271`，`get_alchemical_waters`，离溶质 ≥0.8 nm 选水），这个 helper 里没有约束；OpenFE 的 SepTop 不支持净电荷变化（`openmm_septop/equil_septop_method.py:17`）。
