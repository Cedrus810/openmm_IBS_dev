# C2：无蛋白 lipid–water slab charge-transfer 验证

脚本：`tools/validation/validate_charge_transfer_lipid_slab.py`（`PROTOCOL_VERSION = 10`）
测试：`tests/test_c2_lipid_slab_validation.py`
拓扑来源：`charmm-gui-8600905442/gromacs/`（`openmm/` 目录的 `.parm7`/`.rst7` 不被任何代码路径读取，不要用）

每次版本号升级都是 Hamiltonian/电荷账目层面的硬 bug 修复，旧版本产物必须作废
重建，不要跳过版本检查继续用：

**v7 产物冻结只读；v8 pilot 失败结果也冻结只读。下面的 v7 第 4–7 步和 v8
执行段是历史记录；当前执行以本文末尾的“v10 当前执行段”为准。v10 使用
`validation/c2_lipid_slab_v10/` 和 `c2_generated_v10_*`，不覆盖/续跑 v9/v8/v7
trajectory 或 `u_kn`。旧 v8/v9 DCD 只允许用新 gate 做 legacy 重评。**

- v1→v2：总电荷不为零；restraint/Hamiltonian 被重复配置两次。
- v2→v3（2026-08-07）：`MonteCarloMembraneBarostat`（几何上已是 semi-isotropic）
  配合 hard 1.0 nm LJ 截断在这个 Lipid21 slab 上产生人工面内压缩——实测
  10 ns 内 APL 0.683→0.590 nm² 仍在降、膜厚涨到 4.13 nm。根因是各向同性解析
  色散尾项（`setUseDispersionCorrection`）只是总体积的函数，分不清 MC 试探
  移动缩放的是各向异性膜结构的 XY 还是 Z。修法：两处 `NonbondedForce`
  都加一个窄 potential-switch（`C2_LJ_SWITCH_DISTANCE_NM = 0.995` nm，
  outer cutoff 仍是 1.0 nm）——**范围只到 C2 自己的 System 构建**，不改
  `abfe_core.py`/`ibs_engine.py` 的全局 MEM-00h 常量（那会牵动已跑通的
  复合物/溶剂腿，需要独立决策）。**barostat 类没有换**，仍是
  `MonteCarloMembraneBarostat(XYIsotropic, ZFree)`，问题从来不是"缺
  semi-isotropic"，而是 MC + hard cutoff 这个组合。
  **之前跑过的 `base_thin`（v2 协议下的旧 System）必须整个重新跑，不能续跑
  `equilibrated.gro`**——Hamiltonian 本身变了，不是同一个协议下需要更长时间
  这种情况。
- v3→v4（2026-08-09，诊断 thick base 快速塌缩时发现，不改 Hamiltonian，只改
  诊断/建水/预平衡流程本身，但仍然作废旧产物——数字口径变了）：
  1. `base-quality-gate` 的 `density_profile_along_normal` 之前存的是每个
     bin 逐帧平均**原子计数**（少除了一个 bin 体积），现在除以
     `bin_volume_nm3`（末段窗口 XY 面积均值 × bin 厚度），单位是真正的
     `nm⁻³` 数密度，可以跟"约 33 nm⁻³ 体相水"这类文献值直接比较。
  2. `extend-water` 之前固定 0.25 nm 边界缓冲 + `np.arange` 步进截断，在
     `--extra-water-nm` 较小时实际铺出的密度只有声称的
     `BULK_WATER_NUMBER_DENSITY_PER_NM3=33.33 nm⁻³` 的六成左右（1024 个水、
     约 49 nm³ 新增体积，摊到整段体积上只有约 20.9 nm⁻³）。改了两轮：第一轮
     把缓冲区改成与目标格点间距成比例、格点数按四舍五入精确铺满**扣掉缓冲后
     的子体积**——但这样算出来的目标格点数仍然只对应子体积，摊到**完整**新增
     体积上密度依旧只有六成左右，2026-08-09 review 抓住了这一点；第二轮改成
     目标格点数按**完整**新增体积算，缓冲只决定这些格点摆在哪，不影响摆几个
     ——同一个真实 C2 盒子现在铺出约 1536 个水（原来 1024 个），按完整体积算
     密度约 30.8 nm⁻³（偏离目标约 7.6%，原来约 37%）。
     `extend_water_manifest.json` 里新增 `full_added_volume_nm3` /
     `target_water_count_full_volume` / `actual_water_count` /
     `achieved_density_full_volume_nm3` 四个字段，全部按完整体积算，如实记录
     实际达到的密度——不再看扣掉缓冲后的子体积。
  3. `equilibrate-base` 新增 `--n-steps-nvt`（默认 0，向后兼容）：>0 时先在
     固定盒（不加 barostat）下跑这么多步，让新增水层扩散松弛，再把 barostat
     加进 System 并 `Context.reinitialize(preserveState=True)` 切到 NPT——
     不需要重建 Context、位置/速度/盒矢量原样保留。用于诊断 thick base
     "最小化后直接开 NPT、新增水层还没来得及弛豫"这个可能的塌缩根因。
- v4→v5（2026-08-09，实测 GPU pilot 后立刻发现，**严重、影响默认值**）：
  4. `--n-steps-nvt=0`（默认值）那条分支只对 `system.addForce(barostat)`，
     没调用 `Context.reinitialize`——`simulation.context` 早就用不带
     barostat 的 `system` 建好了，Python 端加 Force 不会让已经建好的
     Context 知道，新加的 barostat 完全是摆设。实测复现：某次续跑用
     `--n-steps-nvt 0` 跑了 8 ns「NPT」，`base-quality-gate` 时间序列显示
     box_z/APL 从头到尾逐帧 **bit-for-bit 原样不变**——一步体积试探移动都
     没真的发生过，整段其实是伪装成 NPT 的 NVT。`--n-steps-nvt > 0` 那条
     分支本来就有 `reinitialize`，不受影响。修法：抽出
     `_add_barostat_and_activate(system, membrane_protocol, temperature_kelvin,
     pressure_bar, simulation)` 把"加 barostat"和"reinitialize"绑死在一个
     函数里，两个分支都改用它——结构上不再可能只做一半。新增回归测试
     `test_add_barostat_and_activate_actually_changes_volume` 直接钉住这个
     退化情形（高压+每步试探，60 步内体积必须有可测量变化）。
     **任何用 v4 脚本、`--n-steps-nvt` 留默认值 0 跑出来的 `equilibrate-base`
     产物都必须作废重跑**——不止是本轮诊断 pilot 的续跑段。
- v5→v6（2026-08-09，thin base 真正跑 `build` 首次触发——此前测试只测合成
  System，从没走过这条候选点筛选路径）：
  5. `insert_ions_into_gromacs_files` 挑候选点只按 `COION_LIGAND_MIN_IMAGE_INITIAL_NM`
     （1.6 nm，§13.1 更松的"initial"判据）筛配体↔co-ion 距离，但真正决定
     restraint 是否构造性安全的是 `validate_co_alchemical_ion_placement`
     的 **runtime** 判据（默认 restraint 参数下实际需要 d0 ≥ 约 2.02 nm，
     比 1.6 nm 高 26%）。实测触发：thin base `build --ion Na
     --position-variant 0` 选出 d0=1.968 nm（满足 1.6 nm 但不满足 2.02 nm），
     `cmd_build` 直接 `ValueError` 中止。修法：新增
     `_required_ligand_coion_min_image_nm(restraint_k, restraint_r0_nm)`，
     用 `cmd_build` 实际会用的同一对 restraint 参数反解出真正需要的最小
     距离（+0.05 nm 安全余量）并入候选点筛选门槛——**没有改
     `abfe_core.py` 里 §13.1 本身的任何常量**，只是把 C2 自己的候选点筛选
     门槛提高到跟下游一致。
- v6→v7（2026-08-09，第一批真实 4 格 GPU pilot 跑完、`slab-quality-gate`
  首次在真实探针 case 上执行才触发——**严重**）：
  6. `_find_bulk_water_candidates` 算候选水"离膜中面多远"直接算
     `abs(z - midplane_z_nm)`，**没做 z 轴周期折叠**。`.gro` 坐标是 OpenMM
     跑出来的原始坐标，长时间模拟下扩散穿过周期边界的原子不会被自动折回
     `[0, box_z)`——实测约 24% 的原子受影响。这类原子被算出**虚高**的
     "离中面距离"：`Na_thin_pos0` 选中的三个点报的都是 5.42-5.69 nm
     （几何上不可能，超过 `box_z/2≈4.17 nm`），真实 minimum-image 距离只有
     2.65-2.92 nm——**全部低于 3.0 nm 的 bulk-water 安全下限**，本该被
     过滤掉。"farthest-first"贪心算法偏好 `abs_dz` 最大的候选，于是系统性
     选中了这些实际离膜很近的候选。**实测后果**：4 个真实 GPU pilot 里，
     `slab-quality-gate` 在每一个 λ 窗口都测到探针 40-140 ps 内就逼近
     磷原子到 0.64-1.3 nm、水配位跌到 0——不是意外扩散，是初始点位本来就
     没那么深。修法：新增 `_minimum_image_z_delta_nm`，把 z 差值沿 z 轴
     单轴折进 `[-box_z/2, box_z/2)` 再取绝对值/判号（`side` 同样受益，
     之前也可能因为同一个未折叠 z 判反上/下叶）；`abfe_core.assign_lipid_leaflets`
     算的膜中面本身核对过不受影响（磷原子坐标全部在盒内，不像水一样大范围
     扩散穿界）。**已跑完的四格 `build`/`static-check`/GPU pilot 全部作废，
     必须用 v7 重新 `build`+`static-check`，4 个完整 pilot 也要重新提交
     GPU 重跑**——thin/thick base 本身不受影响（不涉及候选点选择），不用
     重跑。

---

## 第 0 步：CPU 验证（不需要 GPU，先跑这个）

```bash
cd /home/ruigengji/ABFE_IBS/Atenolol-rank11

python -m py_compile tools/validation/validate_charge_transfer_lipid_slab.py

source /home/ruigengji/mambaforge/etc/profile.d/mamba.sh
mamba activate openmm_dev

pytest -q \
  tests/test_c2_lipid_slab_validation.py \
  tests/test_charge_transfer_hamiltonian.py \
  tests/test_coalchemical_ion_identity.py \
  tests/test_dispersion_and_forcefield_protocol.py
```

**这一步没有全部通过之前，GPU 预算为 0。** `py_compile` 和上面这一行 `pytest`
（v7 改动后，含新增的第 12/13/14/15/16 组回归测试）都已经在 `openmm_dev`
里实测跑过、全部通过（111 项：`test_c2_lipid_slab_validation.py` 32 + 其余
三个文件 79）。

---

## 第 1 步：平衡 thin base slab（GPU）

**如果 `validation/c2_lipid_slab/base_thin_v3_extend1/equilibrated.gro` 已经存在
且 `passed=true`，跳过这一步，直接用它当第 2 步的输入**——不要重新烧 GPU。
它是 v3 协议下已经跑通的 thin base（`base_thin_v3` 跑了 10 ns，
`base_thin_v3_extend1` 又续跑了 5 ns，`base_quality_gate_final.json` 全部
checks 为 true）。v3→v4 的三处修复（density_profile 单位、extend-water 密度、
`--n-steps-nvt`）都不改变 thin base 自己的 System/Hamiltonian，所以这份 v3
产物对 v4 来说不是"随便凑合的起始坐标"——已经用 v4 脚本对同一份 DCD
重新跑过一遍 `base-quality-gate`（纯 CPU，不需要新 GPU 时间）验证过：
`validation/c2_lipid_slab/base_thin_v3_extend1/base_quality_gate_v4.json`，
`checks`/`passed` 与 v3 那份逐位一致，`density_profile_along_normal` 换成了
v4 修复后的真实 nm⁻³ 单位。**只有** `base_thin_v3_extend1` 不存在时，才需要
从头跑下面这一整套：

```bash
python tools/validation/validate_charge_transfer_lipid_slab.py equilibrate-base \
  --top charmm-gui-8600905442/gromacs/topol.top \
  --gro charmm-gui-8600905442/gromacs/step5_input.gro \
  --water-thickness-label thin \
  --n-steps 5000000 \
  --seed 2026 \
  --platform CUDA \
  --precision mixed \
  --output-dir validation/c2_lipid_slab/base_thin
```

`--platform CUDA` 建不出来会直接报错退出（不会静默换 CPU 跑 10 ns——那要跑
几十倍时间且没有代表性）。只有秒级自检才该加 `--allow-cpu-fallback`。

```bash
python tools/validation/validate_charge_transfer_lipid_slab.py base-quality-gate \
  --top charmm-gui-8600905442/gromacs/topol.top \
  --gro validation/c2_lipid_slab/base_thin/equilibrated.gro \
  --dcd validation/c2_lipid_slab/base_thin/equilibration.dcd \
  --frame-interval-ps 10 \
  --tail-fraction 0.2 \
  --literature-apl-nm2 0.6392 \
  --output validation/c2_lipid_slab/base_thin/base_quality_gate.json
```

`--frame-interval-ps 10` = `--report-interval-steps 5000`（默认值）× `--timestep-ps 0.002`；
改了 `equilibrate-base` 的这两个参数记得同步改这里，不要靠记的。

**不通过就停，不生成 probe case。** 不通过时**不要**无条件续跑 5–10 ns——
`base_quality_gate.json` 里的 `recommendation` 字段会给出条件判断，读的是
`apl_tail_drift_significance` + `apl_moving_toward_literature_target`。**权威
判据是分块法**（`_block_mean_drift_significance`，块宽
`DRIFT_BLOCK_WIDTH_NS=1.0 ns`，两侧各至少 `DRIFT_MIN_BLOCKS_PER_HALF=2` 块）：
先分块求均值压掉高频噪声，再比较末段窗口前半 vs 后半的**块间**方差，配
`DRIFT_SIGNIFICANCE_Z=2.0` 门槛判"是不是真在漂"——不是用 OLS 线性回归的斜率
标准误（`apl_tail_ols_slope_percent_per_ns` 字段仍然算出来落盘，但只是点估计
参考值，不判显著性；同一条轨迹换 OLS 拟合窗口给出过相反结论，APL 这类量在
几 ns 尺度上强自相关、不是白噪声，OLS 标准误的独立性假设不成立）：

- 分块法判定显著为正、且正在朝 `literature_apl_nm2` 靠近 → 可以续跑（另开一次
  `equilibrate-base`，`--gro` 指向这份 `equilibrated.gro`，约 5 ns，拼接两段
  DCD 后重判，`equilibrate-base` 不支持原地 resume）。
- 分块法判定显著为负（仍在收缩）→ **禁止续跑**，说明
  `C2_LJ_SWITCH_DISTANCE_NM=0.995` 这个窄 switch 窗口不够，下一步应该改测
  0.95→1.00 nm 更宽的 switch pilot，不是给同一协议更多时间。
- 分块法判定不显著（`not_significant`，只是压线没过阈值）→ 先不续跑；如果
  同时 `apl_within_3_percent_of_literature=False`（稳定在一个偏离文献值的
  平台），续跑大概率解决不了，需要单独排查协议/文献值本身，不是"还没平衡够"。
- 块数不够（`insufficient_data`）→ 如实报告"数据不够判"，不能悄悄当作
  "判了说没事"，先延长轨迹或调大 `--tail-fraction` 拿到足够块数再判。

---

## 第 2 步：thick base slab —— 先短诊断 pilot，不要直接烧 10–15 ns（GPU）

**2026-08-09 现状**：thick 输入最小化后直接跑长 NPT，实测第一个 10 ps 帧 APL
就从初始 ~0.6128 nm²（`4.95082²/40`）掉到 0.5844，之后约 3 ns 回升到 ~0.61，
最终又落到 ~0.58 左右——不是"再多跑几 ns 自然回到文献值"这种形状，需要先做
短诊断把"早期 barostat 冲击 / 建水本身有问题 / switch 真的依赖盒高"这三种可能
分开，而不是继续烧同一条长轨迹。依次做 Plan A → B → C，只有前一个失败才做
下一个。

**Plan A：固定盒 NVT 松弛 + 短 NPT**（保持现有 thick 输入不变）：

```bash
python tools/validation/validate_charge_transfer_lipid_slab.py extend-water \
  --top charmm-gui-8600905442/gromacs/topol.top \
  --gro validation/c2_lipid_slab/base_thin_v3_extend1/equilibrated.gro \
  --extra-water-nm 2.0 \
  --output-dir validation/c2_lipid_slab/thick_input
```

（源坐标用已经通过 v3 base-quality-gate、并且已经用 v4 脚本重新验证过的
`base_thin_v3_extend1/equilibrated.gro`——不要用 `base_thin/equilibrated.gro`，
那是更早的 v2 产物，见上方第 1 步的说明。两侧对称各加约 1.0 nm，不是只加在
盒顶——v2 已修。先核对 `thick_input/extend_water_manifest.json` 里的
`achieved_density_full_volume_nm3` 有没有明显偏离
`target_number_density_per_nm3=33.33`——v4 修了「声称密度 vs 实际密度」这条
不一致，且是按**完整**新增体积算的密度，不是扣掉 buffer 后的子体积（见上方
v3→v4 changelog §2）；典型偏离约 5–10%，`--extra-water-nm` 越薄偏离通常越大。
如果这里偏离远超这个量级，先跳到下面 Plan B，不要先跑 Plan A。）

```bash
python tools/validation/validate_charge_transfer_lipid_slab.py equilibrate-base \
  --top charmm-gui-8600905442/gromacs/c2_generated_thick_topol.top \
  --gro charmm-gui-8600905442/gromacs/c2_generated_thick_step5_input.gro \
  --water-thickness-label thick \
  --n-steps-nvt 250000 \
  --n-steps 1000000 \
  --seed 2026 \
  --platform CUDA \
  --precision mixed \
  --output-dir validation/c2_lipid_slab/base_thick_pilot

python tools/validation/validate_charge_transfer_lipid_slab.py base-quality-gate \
  --top charmm-gui-8600905442/gromacs/c2_generated_thick_topol.top \
  --gro validation/c2_lipid_slab/base_thick_pilot/equilibrated.gro \
  --dcd validation/c2_lipid_slab/base_thick_pilot/equilibration.dcd \
  --frame-interval-ps 10 \
  --tail-fraction 0.3 \
  --literature-apl-nm2 0.6392 \
  --output validation/c2_lipid_slab/base_thick_pilot/base_quality_gate.json
```

`--n-steps-nvt 250000`（`250000 × 0.002 ps = 500 ps = 0.5 ns`，固定盒，不加
barostat）+ 剩余 `1000000 - 250000 = 750000` 步（`750000 × 0.002 ps =
1500 ps = 1.5 ns`）NPT——**总共 `1000000 × 0.002 ps = 2000 ps = 2.0 ns`
诊断 pilot**（0.5 ns NVT + 1.5 ns NPT），不是 10 ns 生产平衡。
`equilibration_monitor.csv` 新增的 `phase` 列（`nvt`/`npt`）可以直接按阶段
切分看 APL/box_z 是"NVT 阶段就已经稳定、NPT 一开就崩"还是"NVT 阶段本身就在
变化"。pilot 只有 2 ns，`--tail-fraction` 调大到 0.3（约 0.6 ns 末段窗口）
避免分块法因为块数不够直接判 `insufficient_data`。也重点看
`density_profile_along_normal`（v4 起是真正的 `nm⁻³` 数密度，见上方
changelog §1）——水氧数密度在膜疏水核区域应接近 0、体相区域应落在
30 nm⁻³ 量级，藉此判断"是不是水层本身就没铺对/有空隙"。

- pilot 通过（APL 分块法不显著漂移、在文献值 3% 以内）→ 续跑到完整生产平衡：

  **⚠️ 如果你已经用 v4 脚本跑过这一段续跑（`--n-steps-nvt 0`），产物必须作废
  重跑**——v4→v5 修的那个 bug 正好在这条路径上：`--n-steps-nvt=0` 分支加了
  barostat 却没 `reinitialize`，实测确认过续跑出来的 `equilibration.dcd`
  box_z/APL 从头到尾 bit-for-bit 不变，整段其实是伪装成 NPT 的 NVT，不是真
  的平衡过。用 v5 脚本（已修）重新跑下面这条命令：

  ```bash
  python tools/validation/validate_charge_transfer_lipid_slab.py equilibrate-base \
    --top charmm-gui-8600905442/gromacs/c2_generated_thick_topol.top \
    --gro validation/c2_lipid_slab/base_thick_pilot/equilibrated.gro \
    --water-thickness-label thick \
    --n-steps-nvt 0 \
    --n-steps 4000000 \
    --seed 2026 \
    --platform CUDA \
    --precision mixed \
    --output-dir validation/c2_lipid_slab/base_thick
  ```

  （pilot 已经跑了 2 ns，`4000000 × 0.002 ps = 8000 ps = 8.0 ns`，两段合计
  `2 + 8 = 10 ns`，凑满完整生产平衡的时长——不是 `8500000` 步/17 ns，那是
  之前版本的算术错误。）

  （`--n-steps-nvt 0`：新增水层已经在 pilot 里弛豫过了，续跑不用再来一次固定盒
  阶段。`equilibrate-base` 不支持原地 resume，这是重新从 pilot 的
  `equilibrated.gro` 出发独立跑的一段，用 `base-quality-gate` 的多 `--dcd` +
  `--restart-discard-ps` 拼接两段重判——用法与第 1 步"漂移显著为正、继续跑"
  完全一样：）

  ```bash
  python tools/validation/validate_charge_transfer_lipid_slab.py base-quality-gate \
    --top charmm-gui-8600905442/gromacs/c2_generated_thick_topol.top \
    --gro validation/c2_lipid_slab/base_thick/equilibrated.gro \
    --dcd validation/c2_lipid_slab/base_thick_pilot/equilibration.dcd \
          validation/c2_lipid_slab/base_thick/equilibration.dcd \
    --frame-interval-ps 10 \
    --tail-fraction 0.2 \
    --literature-apl-nm2 0.6392 \
    --output validation/c2_lipid_slab/base_thick/base_quality_gate.json
  ```

- pilot 仍然快速塌缩（第一个 NPT 报告点就大幅偏离初始 APL 且之后没有回升
  迹象）→ Plan A 不够，进 Plan B。

**Plan B：重新构建密度明确、无空隙/重叠的 thick 水层**（Plan A 失败才做）：
若 Plan A 一开始核对到的 `achieved_density_full_volume_nm3` 明显偏离目标，或
密度剖面显示铺水区域有明显空隙，考虑加大 `--extra-water-nm`（更厚的新增水层，
取整/缓冲损耗占比更小）或另想办法把水铺得更密实，然后重复 Plan A 那一套
NVT→NPT pilot。

**Plan C：测试 switch 的盒高敏感性**（只有 Plan A/B 都失败才做）：
把 `C2_LJ_SWITCH_DISTANCE_NM` 从 `0.995` 改成 `0.95`（更宽的 0.95→1.00 nm
switch 窗口）——**这会改变 Hamiltonian**，必须独立走一次 `PROTOCOL_VERSION`
升级，不能不升版本号就用这个改动继续跑，也不能跳过 Plan A/B 直接来试这一条。

thin/thick 两个 base 都通过才继续。

---

## 第 3 步：构建 4 个单-seed pilot（纯 CPU）

固定只用 Na⁺，4 格 = 2 thickness × 2 position：

| Case | Base | position |
|---|---|---:|
| `Na_thin_pos0` | thin | 0（upper） |
| `Na_thin_pos1` | thin | 1（lower） |
| `Na_thick_pos0` | thick | 0（upper） |
| `Na_thick_pos1` | thick | 1（lower） |

```bash
python tools/validation/validate_charge_transfer_lipid_slab.py build \
  --top charmm-gui-8600905442/gromacs/topol.top \
  --equilibrated-gro validation/c2_lipid_slab/base_thin_v3_extend1/equilibrated.gro \
  --ion Na --water-thickness-label thin --position-variant 0 \
  --output-dir validation/c2_lipid_slab/Na_thin_pos0

python tools/validation/validate_charge_transfer_lipid_slab.py build \
  --top charmm-gui-8600905442/gromacs/topol.top \
  --equilibrated-gro validation/c2_lipid_slab/base_thin_v3_extend1/equilibrated.gro \
  --ion Na --water-thickness-label thin --position-variant 1 \
  --output-dir validation/c2_lipid_slab/Na_thin_pos1

python tools/validation/validate_charge_transfer_lipid_slab.py build \
  --top charmm-gui-8600905442/gromacs/c2_generated_thick_topol.top \
  --equilibrated-gro validation/c2_lipid_slab/base_thick/equilibrated.gro \
  --ion Na --water-thickness-label thick --position-variant 0 \
  --output-dir validation/c2_lipid_slab/Na_thick_pos0

python tools/validation/validate_charge_transfer_lipid_slab.py build \
  --top charmm-gui-8600905442/gromacs/c2_generated_thick_topol.top \
  --equilibrated-gro validation/c2_lipid_slab/base_thick/equilibrated.gro \
  --ion Na --water-thickness-label thick --position-variant 1 \
  --output-dir validation/c2_lipid_slab/Na_thick_pos1
```

`build` 会在 `charmm-gui-8600905442/gromacs/` 里新增
`c2_generated_Na_<label>_pos<N>.top`/`.gro`（原始 `topol.top`/`step5_input.gro`
一个字节不动），四格各自独立。

四格都 `static-check`：

```bash
for c in Na_thin_pos0 Na_thin_pos1 Na_thick_pos0 Na_thick_pos1; do
  python tools/validation/validate_charge_transfer_lipid_slab.py static-check \
    --output-dir validation/c2_lipid_slab/$c
done
```

每份 `static_check_report.json` 的 `passed` 必须是 `true`；脚本本身在检查不过时
会直接 `SystemExit`，所以命令跑失败就是没过，不需要另外解析文件。

**四格任一 `static-check` 失败 → 0 charging GPU 预算。**

---

**2026-08-09 进度**：第 1-3 步已经用 v7 重新跑完并通过——`base_thin_v3_extend1`/
`base_thick`（gate `passed=true`，未受 v7 影响，不用重跑）+ 四格 `build`/
`static-check`（`Na_thin_pos0`/`pos1`/`Na_thick_pos0`/`pos1`，`passed=true`，
最近 P31 距离 1.76-3.05 nm，明显好于 v6 时的候选点）。**v6 下跑过的第 4/5
步（wiring smoke + 4 个完整 GPU pilot）产物已作废**——那是 v6 候选点筛选
bug 选出的错误站点，`slab-quality-gate` 在每个 λ 窗口都测到探针异常逼近
磷原子，必须在 v7 重建的站点上重新跑一遍，不能复用旧产物。下面第 4/5 步
需要 GPU，要用户在计算节点提交。

## 第 4 步：wiring smoke（GPU，只对 `Na_thin_pos0`，不计入结果）

```bash
python tools/validation/validate_charge_transfer_lipid_slab.py dynamics \
  --output-dir validation/c2_lipid_slab/Na_thin_pos0 \
  --lambda-coul 1.0,0.0 \
  --n-steps-equil 2000 --n-steps-sample 5000 --save-interval-steps 500 \
  --seed 2026 --platform CUDA --precision mixed
```

只验证：CUDA Context 能建、restraint 没重复（脚本自己会 assert）、DCD/CSV 能写、
两端总电荷为 0（脚本自己会 assert）、力有限。**这次跑完不要跑 `ukn`**，产物之后会被
第 5 步的真实 pilot 覆盖。

---

## 第 5 步：4 个完整 pilot（GPU）

每格（11 个 λ，默认表）：

```bash
python tools/validation/validate_charge_transfer_lipid_slab.py dynamics \
  --output-dir validation/c2_lipid_slab/Na_thin_pos0 \
  --n-steps-equil 20000 --n-steps-sample 50000 --save-interval-steps 500 \
  --seed 2026 --platform CUDA --precision mixed
```

其余三格只换 `--output-dir`。每格 `11 × (20k+50k) = 770k 步 × 2 fs ≈ 1.54 ns`，
4 格共 `≈ 6.16 ns`。

---

## 第 6 步：重算、质量门、汇总（每格依次，CPU/GPU 混合，`ukn`/gate 都是 CPU）

```bash
for c in Na_thin_pos0 Na_thin_pos1 Na_thick_pos0 Na_thick_pos1; do
  python tools/validation/validate_charge_transfer_lipid_slab.py ukn \
    --output-dir validation/c2_lipid_slab/$c --platform CPU

  python tools/validation/validate_charge_transfer_lipid_slab.py slab-quality-gate \
    --output-dir validation/c2_lipid_slab/$c

  python tools/validation/validate_charge_transfer_lipid_slab.py report \
    --output-dir validation/c2_lipid_slab/$c
done
```

四份 `summary.json` 的 `status` 都必须是 `"complete"`、`passed` 都必须是 `true`。

---

## 第 7 步：四个敏感性比较（CPU）

```bash
python tools/validation/validate_charge_transfer_lipid_slab.py compare \
  --report-a validation/c2_lipid_slab/Na_thin_pos0/report.json \
  --report-b validation/c2_lipid_slab/Na_thin_pos1/report.json \
  --label position_thin_pos0_vs_pos1 \
  --output validation/c2_lipid_slab/compare_position_thin.json

python tools/validation/validate_charge_transfer_lipid_slab.py compare \
  --report-a validation/c2_lipid_slab/Na_thick_pos0/report.json \
  --report-b validation/c2_lipid_slab/Na_thick_pos1/report.json \
  --label position_thick_pos0_vs_pos1 \
  --output validation/c2_lipid_slab/compare_position_thick.json

python tools/validation/validate_charge_transfer_lipid_slab.py compare \
  --report-a validation/c2_lipid_slab/Na_thin_pos0/report.json \
  --report-b validation/c2_lipid_slab/Na_thick_pos0/report.json \
  --label thickness_pos0_thin_vs_thick \
  --output validation/c2_lipid_slab/compare_thickness_pos0.json

python tools/validation/validate_charge_transfer_lipid_slab.py compare \
  --report-a validation/c2_lipid_slab/Na_thin_pos1/report.json \
  --report-b validation/c2_lipid_slab/Na_thick_pos1/report.json \
  --label thickness_pos1_thin_vs_thick \
  --output validation/c2_lipid_slab/compare_thickness_pos1.json
```

每项 `passed` 都要求 `|ΔΔG| ≤ 2σ_combined` **且** `|ΔΔG| ≤ 1.0 kcal/mol`（两条同时
满足）。四个 compare 全过后才决定是否扩到 3 seeds，不自动追加计算。

---

## v8 当前执行段（2026-08-10）

v8 只改变验收定义并加入独立 λ-independent bulk-water restraint：默认
`kZ=50 kJ mol⁻¹ nm⁻²`、`rZ=0.5 nm`，目标来自每个平衡盒子的初始 pair-center，
每个积分小段按动态 P31 膜中面更新。Na co-ion 的 hydration hard gate 只对
`abs(q_coion)/abs(q_final) >= 0.9` 的 λ 生效，即 λ=0.1/0.0；要求平均水配位 ≥5
且 ≥5 的帧比例 ≥95%。其它 λ 仍记录配位但只作诊断。

以下 GPU 命令只在 CUDA 计算节点执行；本地无 GPU 时不要加
`--allow-cpu-fallback` 代替 pilot：

```bash
source /home/ruigengji/mambaforge/etc/profile.d/mamba.sh
mamba activate openmm_dev

python tools/validation/validate_charge_transfer_lipid_slab.py dynamics \
  --output-dir validation/c2_lipid_slab_v8/Na_thick_pos0 \
  --lambda-coul 1.0,0.5,0.2,0.1,0.0 \
  --n-steps-equil 20000 --n-steps-sample 50000 --save-interval-steps 500 \
  --seed 2026 --platform CUDA --precision mixed

python tools/validation/validate_charge_transfer_lipid_slab.py dynamics \
  --output-dir validation/c2_lipid_slab_v8/Na_thin_pos1 \
  --lambda-coul 1.0,0.5,0.2,0.1,0.0 \
  --n-steps-equil 20000 --n-steps-sample 50000 --save-interval-steps 500 \
  --seed 2026 --platform CUDA --precision mixed
```

GPU 返回后，在 CPU 节点执行代表性 pilot 质量门：

```bash
for c in Na_thick_pos0 Na_thin_pos1; do
  python tools/validation/validate_charge_transfer_lipid_slab.py slab-quality-gate \
    --output-dir validation/c2_lipid_slab_v8/$c
done
```

两个 pilot 都通过后，再对这两个 case 使用默认 11 λ 完整重跑；随后用同样的
v8 build/static-check 命令新增 `Na_thin_pos0`、`Na_thick_pos1`，四格全部用
11 λ 重跑。四格完成后才执行 `ukn`、`slab-quality-gate`、`report`；v8 的
`u_kn` 不能从 v7 复制。

参数敏感性仍按 `kZ=50 → 100 → 200` 逐级筛选；只有前一级明显不足才升档。

## v9 当前执行段（2026-08-10）

v9 保留 λ-independent bulk restraint 与 hydration gate 定义，但修正三点：

- 膜侧判定使用分数坐标和连续 unwrap；只有 ligand/co-ion 实际进入
  `|Δz| < 3.0 nm` 的膜核心才失败。`±Lz/2` 的符号跳变记为
  `PBC_BOUNDARY_CROSSING`，不算换侧。
- target 使用 `z_target(t)=z_midplane(t)+signed_target_fraction*Lz(t)`；pair-center
  先对 ligand/co-ion 做相对 minimum-image unwrap，再计算中心。势能仍为
  `0.5*kZ*max(0,|d|-rZ)^2` 的 pair-center 形式。
- thin v9 先用 `target offset=0.20 nm`、`rZ=0.30 nm`、`kZ=50`；build 前静态
  geometry envelope 已通过，设计门为 `|Δz|≥3.0 nm`、nearest P31 `≥1.1 nm`
  并计入相对 Z 波动与膜起伏裕量。正式 gate 仍是 nearest P31 `≥1.0 nm`。

CPU 已完成：

```text
validation/c2_lipid_slab_v8/Na_thick_pos0/slab_quality_gate.json  → PASS（重评）
validation/c2_lipid_slab_v8/Na_thin_pos1/slab_quality_gate.json   → FAIL（配位门未通过，结果冻结）
validation/c2_lipid_slab_v9/Na_thin_pos1/                       → build/static-check PASS
```

hydration gate 不降：λ=0.1/0.0 仍要求 mean coordination ≥5 且
`fraction(coordination≥5)≥0.95`。v9 只跑代表 thin pressure case 的 λ=`0.2,0.1,0.0`；
GPU 命令只在 CUDA 计算节点执行：

```bash
source /home/ruigengji/mambaforge/etc/profile.d/mamba.sh
mamba activate openmm_dev

python tools/validation/validate_charge_transfer_lipid_slab.py dynamics \
  --output-dir validation/c2_lipid_slab_v9/Na_thin_pos1 \
  --lambda-coul 0.2,0.1,0.0 \
  --n-steps-equil 20000 --n-steps-sample 50000 --save-interval-steps 500 \
  --seed 2026 --platform CUDA --precision mixed
```

GPU 返回后 CPU 质量门：

```bash
python tools/validation/validate_charge_transfer_lipid_slab.py slab-quality-gate \
  --output-dir validation/c2_lipid_slab_v9/Na_thin_pos1
```

重点核对 `timeseries.csv` 中逐帧 P31 distance、water coordination、pair-center
displacement 和 bulk-restraint energy；低配位帧若与 P31 接近同步，说明 restraint
几何是根因；若不同步且深处 bulk 仍约 88%，再单独审查采样/配位门，当前不放宽门槛。

## v10 当前执行段（2026-08-10）

v9 的失败证据已冻结。失败帧为 λ=0 frame 43：最近 `PA21/P31` atom 2872，ligand
P31=`0.884 nm`、co-ion P31=`2.076 nm`；pair-center displacement=`-0.350 nm`、
pair bulk energy=`2.627 kJ/mol`，说明 pair-center CV 不能保护 ligand reference。
低配位帧与该 P31 事件不同步，因此配位门不降。

v10 在 v9 pair-center restraint 之外增加独立 λ-independent ligand reference/COM
safety wall（force group 8），默认 `kZ=50`、`rZ=0.20 nm`；build 前按 ligand 重原子
相对 Z 包络和 P31 1.1 nm 设计裕量检查，正式 gate 仍为 1.0 nm。λ=0 预设至少
200 帧，并输出 20 帧分块 occupancy。

CPU 已完成：`validation/c2_lipid_slab_v10/Na_thin_pos1/` build/static-check PASS。
只在 CUDA 节点执行：

```bash
source /home/ruigengji/mambaforge/etc/profile.d/mamba.sh
mamba activate openmm_dev

python tools/validation/validate_charge_transfer_lipid_slab.py dynamics \
  --output-dir validation/c2_lipid_slab_v10/Na_thin_pos1 \
  --lambda-coul 0.2,0.1,0.0 \
  --n-steps-equil 20000 --n-steps-sample 50000 \
  --n-steps-sample-lambda0 100000 --save-interval-steps 500 \
  --seed 2026 --platform CUDA --precision mixed
```

GPU 返回后：

```bash
python tools/validation/validate_charge_transfer_lipid_slab.py slab-quality-gate \
  --output-dir validation/c2_lipid_slab_v10/Na_thin_pos1 \
  --output-name slab_quality_gate_hydration_v2.json
```

重点查看 `ligand_safety_restraint_energy_kJ_mol`、`ligand_safety_wall_hit`、
`ligand_nearest_phosphorus_nm` 和 λ=0 的 `block_occupancy_fraction_ge_5`。

### v10 hydration gate v2 离线结论

λ=0 第 6 个 20 帧块的低 occupancy 是短暂水交换：coordination=4 最长连续 5
帧（约 5 ps），没有 coordination≤3；同期 co-ion–P31 最小约 1.44 nm、
膜中面距离约 3.45 nm，未出现几何接近同步。第五近水氧距离虽有一次约 0.402 nm
的瞬时离壳，但没有连续严重事件。

因此 v10 Hamiltonian 不再修改。hydration gate v2 只对
`charge_fraction >= 0.9` 的 λ 硬判：平均配位数≥5、20 帧 block-bootstrap 的
95% 下限≥5、相对 C1 Na 水盒对应 λ 的 block-bootstrap 差值不能显著低于 bulk；
`fraction(coordination≥5)` 仅作诊断；严重脱水统一要求同一帧满足
`coordination≤3 AND r5≥0.4 nm` 且连续至少 2 帧。现有 v10 轨迹重评结果为 PASS，写入
`validation/c2_lipid_slab_v10/Na_thin_pos1/slab_quality_gate_hydration_v2.json`，
原 `slab_quality_gate.json` 失败证据保持不变。

v10 `Na_thick_pos0` 的 5 λ pilot 加 λ=0.1 补充段已通过同一 hydration gate v2；
`Na_thin_pos1` 也已通过。两个代表 case 现在才允许进入完整 11 λ，且完整采样使用
全新输出目录，不能覆盖 pilot 证据。

## v11 当前执行段（2026-08-10）

v10 full-11 的 `Na_thick_pos0` 已通过 slab-quality gate 和 report，CPU `u_kn`
收敛为 `1.1065 kJ/mol = 0.2645 kcal/mol`，该结果冻结为 v10 thick 证据。
`Na_thin_pos1` 在 λ=0 frame 145 发生真实 co-ion 几何越界（膜中面距离
`2.861 nm`、最近 P31 `0.794 nm`），对应的 v10 ΔG 作废，不得用于 compare。

v11 不修改 hydration gate、PBC 判定、pair-center 或 ligand safety wall；只新增
独立的 λ-independent co-ion member safety wall：

- force group `9`，PBC-aware `periodicdistance`，target 随动态 P31 膜中面和当前 Lz 更新；
- 默认 `kZ=100 kJ mol⁻¹ nm⁻²`、`rZ=0.20 nm`，静态设计要求膜中面距离至少
  `3.0+0.2 nm`，P31 设计距离至少 `1.0+0.2 nm`；
- `timeseries.csv` 必须记录 co-ion safety energy、wall-hit 和 target，gate 缺少这些
  诊断即 FAIL；build manifest 记录 member-level fingerprint scope。

v11 thin build/static-check 已通过：
`validation/c2_lipid_slab_v11/Na_thin_pos1/`。下一步只在 CUDA 节点跑
`λ=0.2,0.1,0.0`；该 pilot 通过后再跑同一 v11 thin full-11，最后才重建并重跑
v11 thick full-11。v10 thick 和 v11 thin 不混合作为同一 C2 验收集。

v11 thin pilot 已完成：containment、几何、P31、PBC、restraint 和 λ=0 hydration/C1
reference 全部通过；唯一未过是 λ=0.1 前 60 帧尚未充分平衡，block mean 为
`4.4, 4.2, 4.7, 5.7, 5.7`。新增 `dynamics --restart-dcd` 续接接口，从
`traj_state01_lam0.10.dcd` 最后一帧直接做 100–200 ps equilibration + 至少 200 帧
confirmation，不重跑 λ=0.2，也不重新修改 Hamiltonian。

该 confirmation 已 PASS：λ=0.1 bootstrap lower=`5.113 > 5`，补充段前后 block mean
为 `5.61/5.59`，无连续两帧 severe joint event；因此 v11 thin 代表性 3-λ pilot
阶段正式通过。下一步进入 v11 thin full-11；所有 λ（包括 λ=0.1）统一预平衡
`100000` steps（200 ps），不得恢复到原来的 20000 steps（40 ps）。

thin v11 full-11 已 PASS：11 个 λ 完整，所有几何/P31/PBC/restraint、电荷/能量/力
和 hydration gate 通过；λ=0.1 bootstrap lower=`5.69`，λ=0 mean=`5.835`；MBAR
收敛，最小 overlap=`0.087`，ΔG=`-0.154 ± 0.548 kcal/mol`。λ=1 的 C1 reference
comparison 仅是中性 dummy 诊断态，不作为 hydration 硬门。下一步为同一 v11 协议
重新 build/static-check thick_pos0，再跑 thick full-11；v10 thick 不得替代 v11。

v11 thick_pos0 的 build/static-check 已通过，force group 7/8/9、co-ion safety 静态
几何、电荷守恒和单点能量/力均通过；产物位于
`validation/c2_lipid_slab_v11/Na_thick_pos0/`。下一步只提交同样
`n_steps_equil=100000`（200 ps）的 thick full-11 CUDA。

两个代表 case 的 v11 full-11 已正式通过：thin_pos1
`-0.1537±0.5481 kcal/mol`、thick_pos0 `-0.7580±0.5397 kcal/mol`，绝对差
`0.6043 kcal/mol`，合并 `1σ≈0.769 kcal/mol`，同时满足 2σ 和 1 kcal/mol 门；
MBAR overlap 约 `0.087/0.104`。C2 进入四格完整验收，但尚未关闭。下一步为
`Na_thin_pos0` 与 `Na_thick_pos1` 完成同一 v11 build/full-11/gate/u_kn/report，
然后执行四个方向的 compare；四格全部通过后才扩展 3 seeds。

在新增位置 build 时，thin_pos0 的 farthest-first 初始三点组合被 v11 静态 P31
包络正确拦截（最坏 `0.960 nm`）；候选选择现已在同侧候选池内继续搜索满足
pair/member 最坏包络的组合，未放宽 gate、未改 Hamiltonian。该修复后 thin_pos0
和 thick_pos1 的 v11 build/static-check 均通过。

四格 v11 full-11 已全部 PASS（单 seed）：thin_pos0=`-0.7071±0.6439`、
thin_pos1=`-0.1537±0.5481`、thick_pos0=`-0.7580±0.5397`、
thick_pos1=`-0.1806±0.5687 kcal/mol`；四个 compare 方向的绝对差分别为
`0.5534/0.5775/0.0510/0.0269 kcal/mol`，全部通过 2σ 和 1 kcal/mol 门。
C2 四格单 seed 阶段完成，下一步才是按清单扩展 3 seeds；当前不宣称最终生产资格。

seed 扩展口径固定为“每格总共 3 个 seed”：保留当前 `seed=2026`，只新增
`seed=2027,2028`，共 8 次新 full-11，最终 12 个 case-seed。v11 Hamiltonian、
λ schedule、`n_steps_equil=100000`（200 ps）、采样长度和 gate 全部固定；单 seed
产物冻结不覆盖。每个新 seed 独立完成 gate、MBAR、report/summary；最终按 3 个
seed 统计 case 均值和跨-seed 不确定度，再重做四个 contrast。12 个结果全部通过
后 C2 才正式关闭并进入 C3。

### v11 三 seed 最终验收与 hydration gate v3（2026-08-10）

12 个 case-seed 已完成统一离线重评：12/12 的 MBAR/u_kn、slab-quality-gate 和
report/summary 均 PASS。原始 equality gate 的边缘 FAIL 证据不删除，保留在各目录的
`slab_quality_gate.json`、`report.json`、`summary.json`；v3 结果使用独立文件名。

hydration gate v3 保留 absolute mean、bootstrap lower bound、severe-dehydration 和
全部几何/restraint 硬门；C1 comparison 改为预先声明的 non-inferiority 规则：
sample-minus-reference bootstrap CI 下界必须 `≥−0.5` 个水分子。该规则统一应用于
全部 12 个结果，不对单个 seed 豁免。

`Na_thin_pos1_seed2027` 的原始 C1 差值 CI 为 `(-0.415,-0.195,-0.010)`，按 v3
规则通过；这不是删除 FAIL，而是保留 FAIL 证据后用正式修订规则重新判定。重评脚本为
`tools/validation/recheck_hydration_reference_noninferiority.py`，审计文件为
`validation/c2_lipid_slab_v11_hydration_noninferiority_v3_summary.json`。

三 seed case 均值 ± seed 间 SD（kcal/mol）：thin_pos0=`−0.158±0.579`、
thin_pos1=`−0.164±0.780`、thick_pos0=`−0.471±0.396`、
thick_pos1=`+0.094±0.240`。四个 cross-seed contrast 均满足 `<1 kcal/mol`
和按 combined seed SD 计算的 `<2σ`，因此 C2 正式关闭，下一步进入 C3。

---

## 执行纪律（原样照抄，不要跳）

```text
CPU 测试不过        → 0 GPU
thin base gate 不过 → 不建 thick/probe
thick base 2 ns 诊断 pilot 不过 → 不烧完整 10 ns（进 Plan B/C，见第 2 步）
thick base gate 不过 → 不跑 charging
四格 static 任一不过 → 0 charging GPU
wiring smoke 不过    → 不跑完整 pilot
四格 pilot 任一硬门失败 → 不补 seeds
```

---

## 算力预算

thin base 10 ns（`base_thin_v3_extend1` 已经跑过、已通过，本轮不需要再烧，
见第 1 步）+ thick base（2 ns 诊断 pilot + 通过后续跑 8 ns，`2+8=10 ns`，
凑满同样的生产平衡时长，不是额外多烧）+ 4 charging case × 1.54 ns
（6.16 ns）+ wiring smoke ≈0.028 ns ⟹ **thin base 完成后，本轮待提交的 GPU
动力学总计约 16.2 ns**（10 + 6.16 + 0.028，thick base 那 10 ns 是本轮唯一
还没跑的部分），不是四格各自重复 10 ns（两个 position 共享同一个对应的
base，不重复平衡）。thick base 拆成 pilot+续跑两段只是把同样的 10 ns 分两次
提交、中途插入质量门检查点，不改变这 10 ns 本身——如果 pilot 没过转向
Plan B/C，才会产生额外的诊断性 GPU 消耗。

---

## 已核对的事实（写代码时对着源码逐行确认过，不是猜的）

- `ibs_engine.configure_charge_transfer_decharging`（ibs_engine.py:2696）内部会调用
  `_inject_co_alchemical_ion_restraints`——外面不能再手动调一次。
- `ibs_engine.TraditionalMBARAnalyzer.compute_u_kn` 内部会调用
  `_prepare_pme_coulomb_leg_system → configure_pme_ligand_charge_offsets`——喂给它的
  `system_template` 必须是**从未配置过**的原始 System。
- `ibs_engine.charging_charge_conservation_report` 的 `base_sum_e`/
  `total_charge_by_lambda_e` 是对 `NonbondedForce.getNumParticles()` **全部**粒子求和
  （ibs_engine.py:2405-2409），不是只算配体+co-ion 子系统——所以"总电荷恒定"和
  "总电荷为零"两条断言可以直接读它的返回值，不需要另外写判据。
- `abfe_core.MEMBRANE_MIN_WATER_SLAB_NM = 2.0`（abfe_core.py:3518）是
  `membrane_observables_from_trajectory` 里"周期镜像接触"用的同一个阈值；
  `slab-quality-gate` 的 water-gap 检查复用的是这个常量，不是另起的数。
- CHARMM-GUI 自带的 `gromacs/step7_production.mdp`（同一份 slab 的原生协议）本身
  就用 `pcoupltype = semiisotropic`（`C-rescale`，`tau_p = 5.0`），独立印证
  "semi-isotropic 压力耦合"这个方向没错，出问题的是 MC + hard cutoff 这个组合，
  不是缺 semi-isotropic 支持。
- `charmm-gui-8600905442/openmm/` 目录的 `.parm7`/`.rst7` 在全仓库任何 `.py` 里都没有
  被读取过；生产主链只吃 `charmm-gui-8600905442/gromacs/` 的 `.top`/`.gro`。
- v4 的 NVT→NPT 分阶段技术（先 `system.addForce(barostat)` 再
  `Context.reinitialize(preserveState=True)`）用一个独立的最小合成 OpenMM
  System（CPU 平台、无关本仓库任何拓扑）单独验证过：`reinitialize` 前后
  positions/velocities/box/势能逐位一致，barostat 加入后续步确实开始改变
  体积——不是只读了文档就假设这个 API 行为，是实测确认过的，现在也钉进了
  `test_run_equilibration_segment_phase_column_and_step_continuity`（第 14
  组）。**但当时只测了 `--n-steps-nvt > 0` 这条路径**——`=0`（默认值）那条
  路径当时是直接调 `core.ensure_barostat_for_protocol` 不经过这个验证过的
  技巧，少了 `reinitialize`，真实 GPU 续跑把这个漏洞暴露了出来（见 v4→v5）。
  修完之后两条路径统一走 `_add_barostat_and_activate`，`
  test_add_barostat_and_activate_actually_changes_volume` 直接复现过"只加
  Force 不 reinitialize"确实会让体积 60 步内 bit-for-bit 不变，再验证修好
  之后体积真的会变——这次两条分支都有对应测试覆盖，不再是"验证了机制、
  但没验证两条调用路径都用上了这个机制"这种半吊子状态。
- `_pack_water_slab` 的 `extend-water` 密度修复用**真实**
  `base_thin_v3_extend1`（v3 已通过的 GPU 产物）平衡后的盒子边长
  `lx=ly≈4.9942 nm`（不是原始 CHARMM-GUI 未平衡的 `5.22685 nm`——平衡后
  APL 收缩过，真实新增体积比用原始盒子估的更小）单独算过两轮：第一轮修复
  按 usable_volume 算密度，1024 个水摊到**完整**新增体积（约 49.9 nm³）上
  仍只有约 20.9 nm⁻³；第二轮（目标格点数改按完整体积算）铺出约 1536 个水，
  按完整体积算密度约 30.8 nm⁻³，偏离目标 33.33 nm⁻³ 约 7.6%（而不是原来的
  ~37%）。
- `base_thin_v3_extend1`（v3 协议下已通过的 thin base，`checks` 全部
  `true`）在 v4 修复后用**同一份**已有 DCD 重新跑过一遍 `base-quality-gate`
  （纯 CPU，零新增 GPU 时间）：`validation/c2_lipid_slab/base_thin_v3_extend1/
  base_quality_gate_v4.json`，`checks`/`passed`/APL/膜厚数字与 v3 那份逐位
  一致，`density_profile_along_normal` 换成了 v4 修复后的真实 `nm⁻³` 单位
  ——不是把 v3 的验收结论直接照搬当 v4 结论，是重新独立验证过的。附带观察
  （不在本轮修复范围内，供下一次排查参考）：这条 thin base 的水氧数密度
  剖面峰值只有约 17 nm⁻³，明显低于体相水的 33 nm⁻³，说明"thin"这个水层
  可能从未真正达到过体相密度——如果后续 thick 侧的诊断 pilot 排除了
  early-barostat-shock/建水 bug 之后仍然异常，这条也值得回头看。
