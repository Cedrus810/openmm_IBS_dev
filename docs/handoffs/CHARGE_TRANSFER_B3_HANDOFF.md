# B3：PME co-alchemical charge-transfer charging Hamiltonian（含 MEM-00d）

日期：2026-08-04
状态：**代码已落地，尚未在真机带电体系上跑过。**
⚠️ 全套计数当时是 1014 passed；**同日之后又落了 B6-FIX / P0-12a/b / P0-13**，
现在是 **1036 passed / 2 skipped / 0 failed**。那批工作见
[`MEMBRANE_SOLVENT_LEG_P013_HANDOFF.md`](MEMBRANE_SOLVENT_LEG_P013_HANDOFF.md)。
主线位置：`memtodolist.md` §17.0 的第 ② 步完成，下一步是 ③ **B4 溶剂腿 builder**。

⚠️ **本次改动不闭合热力学循环。** 复合物腿的 charging 哈密顿量有了，溶剂腿没有
reserved co-ion ⟹ 两腿的 co-ion 项不对消 ⟹ **不得报出任何带电配体的 ΔG_bind**。
唯一的门在 `runabfe.build_and_cache_solvent_leg`，provenance 里落
`closes_thermodynamic_cycle: false` / `must_not_report_delta_g_bind: true`。

---

## 1. 做了什么

| 层 | 新增/改动 | 作用 |
| --- | --- | --- |
| `abfe_core` | `co_alchemical_charge_offset_plan()` | λ 电荷映射的**唯一**实现（纯数学，无 OpenMM 依赖） |
| `abfe_core` | `co_alchemical_ion_restraint_spec()` / `co_alchemical_ion_anchor_atom_index()` / `validate_co_alchemical_ion_placement()` | MEM-00d restraint 描述、锚点规则、§13.1 几何余量判据 |
| `abfe_core` | `minimum_image_displacement_nm()` | minimum-image 数学收敛为一份（`ibs_engine` 那个改成薄包装） |
| `abfe_core` | `build_co_alchemical_ion_identity()` 改签名 | 不再接受调用方给的 restraint 形式；自己算锚点与冻结位移 |
| `abfe_core` | 身份协议 1 → **2** | restraint 键整体改变，v1 spec 一律拒绝复用 |
| `ibs_engine` | `configure_charge_transfer_decharging()` | charge-transfer 的 charging builder（**新增**，没动 co-annihilation 那个） |
| `ibs_engine` | `_create_co_alchemical_ion_restraint()` | flat-bottom + 锚点相对（替换 `_create_bulk_water_ion_restraint`） |
| `ibs_engine` | `charging_charge_conservation_report()` | 读回**真实 Force** 核对逐 λ 电荷账目（两条路线共用） |
| `ibs_engine` | `_identify_reserved_neutral_co_ions()` | charge-transfer 的身份来源，**与坐标无关** |
| `abfe_pipeline` | `ABFEPipeline(charge_treatment=…)` | 把已判死的路线传给唯一那个选择入口 |
| `runabfe` | 6 个 Pipeline 构造点 + 2 个溶剂腿入口 | 两腿传同一份 charge 协议（§6.1）；B4 门；开跑前 WARNING |

哈密顿量（§2.1，OpenMM offset 语义 `q(λ) = q_base + λ·q_scale`）：

```
ligand i : base 0        scale  q_i      ⟹ q(λ) = λ·q_i
co-ion j : base share_j  scale −share_j  ⟹ q(λ) = (1−λ)·share_j     share_j = sign(q_L)·1
```

总电荷守恒因此是**代数结论**而非抽查：`Σq(λ) = Σq_base + λ·Σq_scale`，
所以 `Σscale = 0` 一次性覆盖所有 λ（含中间态，§7.2）。

---

## 2. 证据位置

* `tests/test_charge_transfer_hamiltonian.py`（28 条）：
  * §7.2 逐 λ（11 点）总电荷恒定、`Σq_lig = λ·q_L`、`q_coion = (1−λ)·q_L`；
  * §13.2 **与独立手写参照体系**在 λ=1 / 0 / 0.37 的能量（rel ≤ 1e-5）与逐原子力
    （max|Δ| ≤ 1e-3 kJ/mol/nm）对照 —— λ=1 的参照就是物理体系；
  * 配体内部静电逐 λ 恒定（逐对断言 + NoCutoff 能量侧对照）；
  * §7.3 co-ion mass/LJ 逐 λ 不变、静电走 PME 而非 cutoff ghost force；
  * MEM-00d：平坦区内零力、软墙数值、各向异性盒 minimum-image、
    **坐标+盒同乘 1.05 后井里仍为 0**（旧形式会有 >1 kJ/mol 伪能量）、restraint 逐 λ 同能；
  * fail-closed：带电物理离子当 co-ion、dummy 数量不对、几何余量不够、
    两条路线 spec 互换、缺 spec、溶剂腿（B4）。
* `tests/test_coalchemical_ion_identity.py`（20 条，已按新形式更新）。
* `tests/test_charge_treatment_protocol.py`：那条"B3 未落地"的钩子已按设计改写。
* 全套：`./tests/run_offline_tests.sh -q` → 当时 **1014 passed**；同日后续工作后为 **1036 passed**。

---

## 3. 被否掉的方案（别再走一遍）

### 3.1 `periodicdistance` + `CustomCentroidBondForce` / `CustomCompoundBondForce`

**不存在。** OpenMM 8.5.1 实测直接报 `unknown function: periodicdistance`；
该函数**只在 `CustomExternalForce`** 里有，而 CustomExternalForce 只能吃**绝对**参考点
——正是 MEM-00d 要退役的形式。

可行的组合是 `CustomCompoundBondForce` + `pointdistance` +
`setUsesPeriodicBoundaryConditions(True)`：打开 PBC 后 bond 内的粒子会被平移到与第一个
粒子相同的周期镜像，于是 `pointdistance` **就是** minimum-image 距离。
实测：离子 z=0.2、锚点 z=9.4、盒 z=12 → 0.2 nm（不是 9.2 nm）。

### 3.2 用盒分数坐标 + 每步更新 global parameter 来实现"随盒缩放"

否掉。custom force 拿不到盒矢量，只能靠外部按步更新 `x0/y0/z0`。那样
（a）需要一个每 N 步跑的钩子，（b）`compute_u_kn` 重算时无法复现动力学当时的参数
⟹ u_kn 与动力学用上不同哈密顿量，正是 MEM-00c 那类静默不一致。
**restraint 必须是静态哈密顿量**，锚点相对是唯一能同时满足"随盒缩放"与"静态"的形式。

### 3.3 拿一个已有的物理盐离子当 co-ion

**物理上错的**，不是省事的近似。λ=1 端的总电荷必须等于物理体系的总电荷；普通离子已按
§4.3 把配体的形式电荷配平掉，再把其中一个盐离子的 λ=1 电荷置 0，总电荷就少一个单位。
所以 co-ion 必须是**建系时额外预留的、电荷为 0 的 ion-shaped 粒子**。
代码在 `co_alchemical_charge_offset_plan` 与 `verify_co_alchemical_ion_identity`
两处 fail closed，报错里直接写了该怎么建系。

### 3.4 让 co-ion 也走"按 bulk-water 判据挑一个"

否掉。charge-transfer 的身份来源改成"认出电荷为 0 的离子残基"——**与坐标无关**，
所以 MEM-00c 那个"坐标动 0.05 nm 选择就翻转"的失效模式在这条路线上结构上不存在。
预留数量不等于 |q_L| 就 fail closed：多留一个就又得靠坐标去挑，风险原地复活。

### 3.5 把 §13.1 的几何判据只当事后诊断

否掉。flat-bottom 的软墙很软（k=100 时走出平坦区 0.316 nm 才 2 kT），所以
"co-ion ↔ 配体全程 ≥ 1.2 nm"必须在**构造时**用
`|d0| − r₀ − 软墙余量 − 配体外缘 ≥ 1.2 nm` 证明，不成立就 fail closed。
只留事后诊断等于等着它被违反。

---

## 4. 下一位接手者的禁区

1. **不要把 restraint 换回绝对笛卡尔参考点**，也不要为了让某个体系通过而缩小 r₀
   或放宽 §13.1 的余量判据。摆不下就换更大的盒 / 换摆放位置（§4.2）。
2. **不要在溶剂腿里临时挑一个盐离子当 co-ion 来"先跑通"。** 那既不是 charge-transfer，
   也会让两腿的离子强度口径分叉（§4.3）。B4 要的是那个一次性产出
   `ligand + water + ordinary salt + reserved co-ion` 并分开登记三类离子的 builder。
3. **不要动 co-annihilation 那条路径的物理**（`configure_coalchemical_neutral_decharging`）。
   它按 MEM-00a-3 保留作负对照，用来验证盒长依赖与膜偏差。本轮只把它的 restraint 形式
   与电荷账目核对换成共用实现（§4.4 要求两腿同形式），没有改它的电荷映射。
4. **不要给中性配体造 co-ion。** `configure_charge_transfer_decharging` 在 q_L=0 时直接
   raise：那只会凭空加一个不必要的炼金粒子。当前生产体系（Atenolol，Σq=0）完全不进
   这条路，落盘基线 181.00 / 157.84 / −5.535906 kcal/mol 不受影响（§7.7）。
5. **B5 还没做**：`co_alchemical_ion_fingerprint` 目前只出现在
   `configure_pme_ligand_charge_offsets` 的返回 info 里，**还没有**进窗口 manifest 与
   `_stage_protocol_key`。等 B3/B4 的对象定义稳定后再做（§17.0 第 ④ 步）。
6. **两处已知缺口**（写出来免得被当成已做）：
   * 配置里那份 co-ion **声明**（provenance 的 `coion_identity`）与代码**冻结**的 spec
     之间没有交叉核对；声明写错 atom_index 不会被拦。与 B5 一起定口径。
   * §13.1 的几何判据在构造时只强制了"离配体够远"；离蛋白 / 离膜中面 / 离最近磷
     那三条目前只有逐帧诊断（§9 质量门里已有那四个观测量），没有建系期 fail closed。
7. **MEM-00e 仍未完成**：restraint 自由能在两腿抵消的说明还没写进
   `THERMODYNAMIC_CYCLE_DOC`。论证材料已经有了（两腿同一条锚点规则、同一个 k 与 r₀
   ⟹ 可用体积相同），但要等 B4 让溶剂腿的 co-ion 真正存在时一起落，
   那时才能同时给数值对照。

---

## 5. 真机验证怎么做（§16：必须是可直接执行的命令）

⚠️ 本轮**不需要**、也**不应该**用当前的中性 Atenolol 体系去验证这条路径——它净电荷为 0，
根本不会进 co-ion 分支。第一次真机验证是 §17.0 的第 ⑤ 步 **C1：小水盒 charge-transfer**，
需要先有一个带电配体 + 预留中性 dummy 的输入体系（那是 B4 的产物之一）。

在此之前唯一该跑的是 CPU 全套回归：

```bash
cd /home/ruigengji/ABFE_IBS/Atenolol-rank11
source /home/ruigengji/mambaforge/etc/profile.d/mamba.sh
mamba activate openmm_dev
./tests/run_offline_tests.sh -q                       # 全套，约 2 min
python -m pytest tests/test_charge_transfer_hamiltonian.py -v   # 只看 B3 这一组
```

预期：**`1036 passed, 2 skipped, 1 deselected`**（1014 是本文档写作时的值，同日后续
工作又加了 22 条；见 `MEMBRANE_SOLVENT_LEG_P013_HANDOFF.md`）。

⚠️ **跑这一条的时候不要同时改生产 `.py` 文件。** 本仓库有大量源码/AST 契约测试走
`inspect.getsource`，而它经 `linecache` 读的是 import 时的行号；跑到一半重写
`ibs_engine.py` 会让这些测试读到错位的源码段、报一个假失败
（2026-08-04 实测：`test_bias_calibration_bank_never_applies_lrc_regardless_of_wrapper`
就这样"失败"过一次，静态重跑即通过）。别去 debug 那个失败，先重跑。
