#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C2：无蛋白 lipid–water slab charge-transfer 验证（memtodolist.md「C2」/
memtodolist_archive.md「C2. protein-free lipid slab 测试」）。

PROTOCOL_VERSION = 11（见下方常量）——每次升级都是 Hamiltonian 或电荷账目层面的
硬 bug 修复，旧版本产物**必须作废重建**，不允许跳过版本检查继续用：

  v1→v2 修的两条：
  1. 总电荷不为零：v1 只删 2 个水、插入探针配体 + reserved dummy，λ=1 端净电荷
     等于配体净电荷（±1 e），不是 0。生产 charge-transfer 假设的是"配体电荷被
     搬运到 co-ion，全盒总电荷逐 λ 不变"，但没人规定这个不变的值必须是 0——
     不过 PME 对**非零**总电荷靠隐式中和背景处理，与"配体在生产体系里净电荷
     被普通反离子配平"这个真实场景不一致，必须额外插入一个普通反离子把它配平
     成 0（§2 修复见下）。
  2. restraint 与 charging Hamiltonian 被重复配置两次（build/static-check 手动
     调用 `_inject_co_alchemical_ion_restraints`，`configure_pme_ligand_charge_offsets`
     内部又调用一次；`ukn` 读取已经配置过的 `system_prepared.xml` 又喂给
     `compute_u_kn`，它内部还会再配置一次）。

  v2→v3 修的一条（2026-08-07 实测发现，见 `C2_LJ_SWITCH_DISTANCE_NM` 上方注释）：
  3. `MonteCarloMembraneBarostat`（几何上已经是 semi-isotropic：XYIsotropic +
     ZFree）配合 hard 1.0 nm LJ 截断，在这个 Lipid21 slab 上产生人工面内压缩
     ——实测 10 ns 内 APL 从 0.683 压到 0.590 nm² 且尾段仍在下降、膜厚涨到
     4.13 nm。根因是各向同性解析色散尾项（`setUseDispersionCorrection`）只是
     总体积的函数，无法区分 MC 试探移动缩放的是各向异性膜结构的 XY 还是 Z。
     修法是在两处 `NonbondedForce` 上都加一个很窄（0.995→1.000 nm）的
     potential-switch，尽量不偏离 Amber Lipid21 原始 hard-cutoff 拟合条件，
     只削弱 MC 最敏感的截断处那一小段的能量阶跃。**范围只到 C2 自己的 System
     构建**，不改 `abfe_core.DISPERSION_PROTOCOL_FF_NATIVE_ISOTROPIC_LRC` /
     `ibs_engine.SOFTCORE_CUTOFF_NM` 这些全局 MEM-00h 常量——那会牵动已跑通的
     复合物/溶剂腿，需要独立决策，不在本轮顺手改。

  v3→v4 修的三条（2026-08-09，诊断 thick base 快速塌缩时发现，不改 Hamiltonian，
  只改诊断/建水/预平衡流程本身）：
  4. `base-quality-gate` 的 `density_profile_along_normal` 存的是每个 bin 逐帧
     平均**原子计数**（`counts / n_frames`），不是数密度——量纲上少除了一个 bin
     体积，不能跟"约 33 nm⁻³ 体相水"这类文献数密度直接比较。现在除以
     `bin_volume_nm3`（= 末段窗口平均 XY 面积 × bin 厚度），单位变成
     `nm⁻³`，并把 `bin_volume_nm3` 本身也写进 `density_profile_along_normal`
     供核对。
  5. `extend-water` 声称按 `BULK_WATER_NUMBER_DENSITY_PER_NM3 = 33.33 nm⁻³`
     铺水，实测只填出六成左右。改了两轮：第一轮把固定 0.25 nm 边界缓冲改成与
     目标密度的格点间距成比例（`0.5 × spacing`）、格点数按 `round(...)` 取整后
     精确铺满（不再用 `np.arange` 步进截断掉不足一格的部分）——但格点数仍是按
     **扣掉缓冲后的子体积**算的，摊到完整新增体积上密度依旧只有六成左右
     （2026-08-09 review 抓到）；第二轮改成格点数按**完整**新增体积算，缓冲只
     决定摆在哪、不影响摆几个。`extend_water_manifest.json` 里的
     `full_added_volume_nm3`/`target_water_count_full_volume`/
     `actual_water_count`/`achieved_density_full_volume_nm3` 四个字段全部按
     完整体积算，如实记录实际达到的密度。
  6. `equilibrate-base` 对 thick 输入（新增水层后重新平衡）最小化完直接开
     `MonteCarloMembraneBarostat` 跑 NPT，新加的那层水还没来得及扩散松弛就先
     承受 barostat 的体积试探移动——新增 `--n-steps-nvt`（默认 0，向后兼容
     v3 行为）：>0 时先在**固定盒**（不加 barostat）下跑这么多步，让新增水层
     弛豫，再把 barostat 加进 System 并 `Context.reinitialize(preserveState=True)`
     切到 NPT 跑剩余步数——不需要重新建 Context/丢失已有位置速度。
     `equilibration_monitor.csv` 新增一列 `phase`（`nvt`/`npt`），方便事后按
     阶段切分诊断。

  v4→v5 修的一条（2026-08-09，实测 GPU pilot 后立刻发现，**严重**——影响
  `--n-steps-nvt` 默认值 `0` 这条"声称向后兼容"的路径，不是只影响诊断用的
  非默认参数）：
  7. `cmd_equilibrate_base` 的 `--n-steps-nvt=0` 分支（"不做 NVT，立即加
     barostat"）只对 `system` 对象调用了 `ensure_barostat_for_protocol`（即
     `system.addForce(barostat)`），**没有调用** `simulation.context.reinitialize`
     ——但 `simulation.context` 在这之前早就用不带 barostat 的 `system` 建好了，
     Python 端往 `system` 里加 Force 不会让已经建好的 Context 知道，新加的
     barostat 完全是摆设。实测复现：某次续跑用 `--n-steps-nvt 0` 跑了 8 ns
     "NPT"，`base-quality-gate` 逐帧时间序列显示 box_z/APL 从续跑一开始到结束
     逐帧原样不变（bit-for-bit 相同）——一步体积试探移动都没真的发生过，
     整段其实是伪装成 NPT 的 NVT。`--n-steps-nvt > 0` 那个分支本来就有
     `reinitialize`调用，不受影响；只有 `=0`（默认值！）这条路径受影响。
     修法：`else` 分支现在也补上 `simulation.context.reinitialize(preserveState=True)`，
     两个分支行为对称。**任何用 v4 脚本、`--n-steps-nvt` 留默认值 0 跑出来的
     `equilibrate-base` 产物都必须作废重跑**——不是只有本轮诊断 pilot 的续跑段，
     是所有 v4 产物。

  v5→v6 修的一条（2026-08-09，thin base 真正跑 `build` 时首次触发，此前从未
  被实际执行过——测试只测 `_build_synthetic_charge_transfer_system` 直接搭的
  合成 System，不经过 `insert_ions_into_gromacs_files` 这条候选点筛选路径）：
  8. `insert_ions_into_gromacs_files` 筛选候选点时，配体↔co-ion（dummy）的
     最小 minimum-image 距离只按 `core.COION_LIGAND_MIN_IMAGE_INITIAL_NM`
     （1.6 nm，§13.1 更松的"initial"判据）挑，但真正决定 restraint 是否
     构造性安全的是`abfe_core.validate_co_alchemical_ion_placement`的
     **runtime** 判据：`d0 − flat_bottom_radius_nm − wall_margin_nm ≥
     COION_LIGAND_MIN_IMAGE_RUNTIME_NM`——用默认 restraint 参数
     （`r0=0.5 nm`, `k=100 kJ/mol/nm²` ⟹ `margin≈0.316 nm`）反解出来，
     实际需要 `d0 ≥ 1.2+0.5+0.316 = 2.016 nm`，比 1.6 nm 高出约 26%。
     实测触发：真实 thin base 上 `build --ion Na --position-variant 0`
     选出的 dummy 距配体 `d0=1.968 nm`（满足 1.6 nm 的松判据），但
     `validate_co_alchemical_ion_placement` 算出"可保证的最小配体距离"只有
     1.152 nm < 1.2 nm，`cmd_build` 直接 `ValueError` 中止。修法：新增
     `_required_ligand_coion_min_image_nm(restraint_k, restraint_r0_nm)`，
     用**跟 `select_co_alchemical_ion_once` 实际会用的同一对**
     `restraint_k`/`restraint_r0_nm`（`cmd_build` 的 `--restraint-k`/
     `--restraint-r0-nm`，`None` 时落到同一组默认常量）反解出真正需要的
     最小距离，加 `LIGAND_COION_MIN_IMAGE_SAFETY_BUFFER_NM=0.05 nm` 的显式
     余量后并入 `min_mutual_nm`——不是放宽 §13.1 本身的判据，是把候选点筛选
     的门槛提高到跟下游真正要求的一致（`abfe_core.py` 里的 §13.1 常量一个
     字节没动）。

  v6→v7 修的一条（2026-08-09，第一批真实 4 格 GPU pilot 跑完、`slab-quality-gate`
  首次在真实探针 case 上执行才触发——CPU 契约测试从没测过真实 `.gro` 坐标，
  合成测试系统的坐标从来就是干净摆好的，不会有这条 bug 依赖的"扩散跑出盒子"
  这种真实积累效应）：
  9. `_find_bulk_water_candidates` 算候选水"离膜中面多远"（`abs_dz_from_midplane_nm`，
     用来判断是否满足 `COION_MEMBRANE_MIDPLANE_MIN_ABS_Z_NM=3.0 nm` 的 bulk-water
     下限，也用来给"farthest-first"贪心排序打分）时，直接算
     `abs(z - midplane_z_nm)`——**没有做 z 轴周期折叠**。`.gro` 里的坐标是
     OpenMM/GROMACS 跑出来的原始坐标，长时间模拟下扩散穿过周期边界的原子
     不会被自动折回 `[0, box_z)`；实测 `base_thin_v3_extend1/equilibrated.gro`
     约 24%（21525 个原子里 5244 个）的原子 z 坐标落在 `[0, box_z=8.335nm)`
     之外。这类原子被非周期性差值系统性算出**虚高**的"离中面距离"——
     `Na_thin_pos0` 选中的三个点（普通反离子/配体/dummy）报的都是
     5.42-5.69 nm（几何上不可能：任何点到膜中面的真实周期最短距离都不可能
     超过半个盒高 `box_z/2≈4.17 nm`），真实 minimum-image 距离只有
     2.65-2.92 nm——**全部低于 3.0 nm 的 bulk-water 安全下限**，本该被
     `COION_MEMBRANE_MIDPLANE_MIN_ABS_Z_NM` 过滤掉。贪心算法偏好
     `abs_dz` 最大的候选，于是系统性地优先选中这些被误判"最深"、实际上
     离膜很近的候选。实测后果：`Na_thin_pos0` 的 4 个完整 GPU pilot 里，
     `slab-quality-gate` 在**每一个** λ 窗口都测到探针 40-140 ps 内就逼近
     磷原子到 0.64-1.3 nm（多个低于 1.0 nm 门槛），水配位跌到 0——不是
     "扩散意外跑近"，是初始点位本来就没那么深。`side`（upper/lower）判断
     用的也是同一个未折叠的 `z`，同样受影响（一个真实"下叶"候选可能因为
     unwrapped 坐标被误判成"upper"）。修法：新增
     `_minimum_image_z_delta_nm(z_nm, reference_z_nm, box_z_nm)`，把
     `z - midplane_z_nm` 沿 z 轴单轴折进 `[-box_z/2, box_z/2)` 再取绝对值/
     判号；`assign_lipid_leaflets`（`abfe_core.py`）算 midplane 本身不受
     影响（磷原子坐标经核对全部在 `[0, box_z)` 内，脂质不会像水一样大范围
     扩散穿过周期边界）——问题只在这条候选水筛选路径，范围仍然只到 C2
     自己的代码。**已跑完的四格 `build`/`static-check`/GPU pilot 全部作废，
     必须用 v7 重新 `build`+`static-check`，4 个完整 pilot 也要重新提交 GPU
     重跑**——这不是诊断/统计口径修复，是选点选错了地方。

  v7→v8 修的两条（2026-08-10）：
  10. v7 把中性 co-ion dummy 也要求保持 Na⁺ 的完整水合壳，导致 charge-transfer
      后段把物理上已经中性的粒子误判为失败。v8 只在
      `abs(q_coion)/abs(q_final) >= 0.9` 的 λ（Na probe 即 λ=0.1/0.0）启用水配位
      hard gate；其余 λ 的配位仍逐帧记录，但只作诊断。
  11. v7 只有 ligand–co-ion 相对 restraint，pair 整体仍可能在长轴方向向膜漂移。
      v8 在每个新 build 中加入独立的 λ-independent soft flat-bottom bulk-water
      restraint：两粒子各承担一半能量，合起来约束 PBC-aware ligand–co-ion pair
      center；墙心由每个平衡盒子的初始 pair-center 相对动态 P31 中面位置确定，
      每个积分小段更新动态目标。默认 `kZ=50 kJ mol⁻¹ nm⁻²`、`rZ=0.5 nm`，
      其平坦区内缘约为 |Δz|=3.0 nm。v8 产物必须使用全新输出目录，不能覆盖/续跑
      v7 trajectory 或 `u_kn`。
  v8→v9 修的三条（2026-08-10）：
     12. 膜侧验收改为连续分数坐标下的膜核心穿越判定；±Lz/2 的符号跳变只记为
         PBC_BOUNDARY_CROSSING，不再误报换侧。
     13. bulk target 改为 `z_midplane + signed_target_fraction*Lz`，并保留
         PBC-aware pair-center 与动态 target 更新。
     14. thin v9 使用 target 向水层偏移 0.20 nm、rZ=0.30 nm，并在 build 前执行
         含相对 Z 波动和膜起伏裕量的 P31 静态几何包络检查；hydration gate 不变。

拓扑来自 `charmm-gui-8600905442/gromacs/`（CHARMM-GUI Membrane Builder + FF-Converter
产出的 AMBER Lipid21 + TIP3P 纯 POPC slab，`input.config.dat` 声明
`ltype=Lipid21, wtype=TIP3P`）。**不是** `charmm-gui-8600905442/openmm/` 那份
`.parm7`/`.rst7`——全仓库生产主链（`abfe_core.load_gromacs_topology_for_openmm`）
只解析 GROMACS 文本格式，`openmm/` 目录下的 AMBER prmtop 目前没有任何代码路径读取它。

体系组成（`gromacs/topol.top`）：POPC×80（Lipid21 模块化 `PA`+`PC`+`OL` 三残基）、
K+×8、Na+×8、Cl-×16、TP3(TIP3P)×3591，盒 5.22685×5.22685×8.5 nm，法向 z。
`build` 会自己重新跑一遍力场族识别 + dispersion protocol 校验（不在 manifest 里
硬写 amber），已验证结果与 `memtest/`（已跑通的膜复合物腿）同一力场族 ⟹
`dispersion_protocol = ff_native_isotropic_lrc`，cutoff 1.0 nm（MEM-00h 已收敛值）。

## 电荷配平（v2 新增，§1）

`build` 现在删 3 个水、插入 3 个新粒子（顺序即 `[ molecules ]` 追加顺序，也是
`.gro` 里最后 3 个原子）：

    普通反离子（与探针配体异号）×1  →  index = N-3
    探针配体（净电荷 ±1）        ×1  →  index = N-2
    reserved co-ion dummy（同号，清零）×1  →  index = N-1

Na 探针：普通反离子 = Cl⁻；Cl 探针：普通反离子 = Na⁺。三者都从满足 §13.1/§4.4
bulk-water 判据、彼此 minimum-image 距离 ≥ `COION_COION_MIN_IMAGE_INITIAL_NM`
的候选水里贪心选出。

硬断言（`build` 与 `static-check` 都做）：

    λ=1: probe(+1) + dummy(0)  + 普通反离子(-1) = 0
    λ=0: probe(0)  + dummy(+1) + 普通反离子(-1) = 0
    所有 λ：|Q_total| ≤ 1e-6 e

用的是生产函数 `ibs_engine.charging_charge_conservation_report`——它的
`base_sum_e`/`total_charge_by_lambda_e` 本来就是对 `NonbondedForce` **全部**
粒子求和（不是只算配体+co-ion 子系统），所以"总电荷恒定"和"总电荷为零"这两条
断言直接从它的返回值读，不需要另写一套判据。

## 与 C1 的架构差异

C1（`validate_charge_transfer_waterbox.py`）是纯水盒：从零用
`openmm.app.Modeller.addSolvent(ForceField(...))` 搭建，配体/reserved co-ion dummy
靠 `runabfe._insert_reserved_coalchemical_ion_dummies()` 插入，参数由
`ForceField.createSystem()` 按残基名模板在建 System 时解析。

本文件是 GROMACS `.top`/`.gro` 路线（与膜复合物腿同架构）：System 参数来自
`GromacsTopFile.createSystem()`，直接解析 `.top` 自带的原子类型/非键参数，
不经过任何 `ForceField` 残基模板匹配。往这类 System 里加新粒子，必须让粒子在
`.top`/`.gro` 文本层面就存在——`_insert_reserved_coalchemical_ion_dummies()`
那种"建好 System 之后再往 Modeller 里插"的手法在这里不适用（新粒子不会有任何
非键参数）。复合物腿对 reserved dummy 的说明是"建系脚本预先带上"，即在这个
Python 管线之外、手工/用 GROMACS 工具准备好——查过仓库全部代码确认没有对应的
Python 写入实现（Atenolol 净电荷为 0，这条路径至今没有被真正跑过一次）。

所以本文件新增了 GROMACS 文本层面的插入实现（`insert_ions_into_gromacs_files()`），
只做"选点 + 编辑 `[ molecules ]` + 编辑 `.gro` 坐标"——身份识别、restraint、
charging 电荷映射一律调用已有生产函数（`ibs_engine.select_co_alchemical_ion_once`
/ `configure_pme_ligand_charge_offsets` / `charging_charge_conservation_report`），
不新造判据。bulk-water 候选点筛选复用已有常量
（`abfe_core.COION_MEMBRANE_MIDPLANE_MIN_ABS_Z_NM` /
`COION_NEAREST_PHOSPHORUS_MIN_NM` / `COION_COION_MIN_IMAGE_INITIAL_NM` /
`COION_LIGAND_MIN_IMAGE_INITIAL_NM`），不新造阈值。

## 生成的 `.top`/`.gro` 写在哪里

编辑后的拓扑/坐标写在**与原始文件同目录**（`charmm-gui-8600905442/gromacs/`），
文件名前缀 `c2_generated_`——因为 `topol.top` 用**相对路径**
`#include "toppar/..."`，OpenMM 的 `GromacsTopFile` 按"相对于被 include 文件
所在目录"解析这些路径；挪到别的目录会让 include 找不到文件。原始
`topol.top` / `step5_input.gro` 一个字节都不动，只新增文件。

## 子命令

    equilibrate-base    对纯 slab（无探针配体）跑项目自己的膜预平衡协议（GPU，需用户提交）
    base-quality-gate   纯 slab（无探针）末段质量门：APL/膜厚/叶片计数/密度剖面/漂移（CPU）
    extend-water        生成对称加厚的第二种水层厚度起始坐标（纯 CPU）
    build               选 bulk-water 点位、插入普通反离子+探针配体+reserved dummy（纯 CPU）
    static-check        逐 λ 电荷守恒(=0)+bulk-water 几何+restraint 唯一性自检（纯 CPU）
    dynamics            逐 λ 平衡+采样（需要 GPU/CUDA，本脚本不会自动执行——用户在计算节点提交）
    ukn                 从**原始** system.xml 唯一配置 Hamiltonian，MBAR 求 charging ΔG（CPU）
    slab-quality-gate   探针 case 专属质量门：全部 11 个 λ 的 DCD + timeseries.csv（CPU）
    report              汇总以上全部产物为 report.json/summary.json，缺一项即 passed=false
    compare             比较两份 report.json 的 charging ΔG，判 2σ/1kcal 门（厚度/位置敏感性通用）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import openmm
from openmm import app, unit, Vec3, XmlSerializer

import abfe_core as core
import ibs_engine as engine

# ============================================================================
# 常量
# ============================================================================

CHARMM_GUI_JOB_DIR = os.path.join(_REPO_ROOT, "charmm-gui-8600905442")
DEFAULT_GROMACS_DIR = os.path.join(CHARMM_GUI_JOB_DIR, "gromacs")
DEFAULT_TOP_FILE = os.path.join(DEFAULT_GROMACS_DIR, "topol.top")
DEFAULT_RAW_GRO_FILE = os.path.join(DEFAULT_GROMACS_DIR, "step5_input.gro")

GENERATED_FILE_PREFIX = "c2_generated_v11_"

# 与生产复合物/溶剂腿一致（MEM-00h 已收敛到的值）。
NONBONDED_CUTOFF_NM = 1.0
EWALD_ERROR_TOLERANCE = 0.0005

# v3：C2 专属修法，**不改** `abfe_core.DISPERSION_PROTOCOL_FF_NATIVE_ISOTROPIC_LRC`/
# `ibs_engine.SOFTCORE_CUTOFF_NM`（那是全局 MEM-00h 决定，牵动已跑通的复合物/溶剂腿，
# 本轮不动，范围只到 C2 自己的 System 构建）。
#
# 根因（2026-08-07 实测确认）：`MonteCarloMembraneBarostat`（XYIsotropic/ZFree，
# 几何上就是 semi-isotropic）是 Monte Carlo（只看能量，不看维里）barostat；
# `NonbondedForce.setUseDispersionCorrection(True)` 加的解析色散尾项只是**总体积**
# 的函数（假设均匀各向同性流体），不知道 MC 试探移动具体缩放的是 XY 还是 Z——
# 而膜是分层的各向异性结构（脂尾致密、水相稀疏），真实"cutoff 之外缺失的色散能"
# 在 XY 方向和 Z 方向并不对称。hard 1.0 nm 截断把这个不对称性完全丢给了那个
# 各向同性近似项去"猜"，猜错的方向正好是把 XY 往里压、Z 往外顶——实测
# APL 0.683→0.590 nm² 且尾段仍在降、膜厚涨到 4.13 nm，与这个机制的预期方向一致。
# 换 barostat 类（比如 `MonteCarloAnisotropicBarostat`）不解决问题，因为它仍是
# MC + 同一套 LJ 处理。修法是在 MC 最敏感的截断附近换成 potential-switch，
# 削弱对那个各向同性近似项的依赖；窗口特意选得很窄（只有 5 pm），
# 尽量不偏离 Amber Lipid21 原始 hard-cutoff 拟合条件——解析 LRC 仍然保留
# （只是"临近截断"的这一小段不再是纯粹的能量阶跃）。
C2_LJ_SWITCH_DISTANCE_NM = 0.995

# Lipid21 POPC @ 303.15 K 参考 APL（2026-08-07 由 reviewer 提供，替换此前占位的
# 0.645 nm²——那个数没有标注力场/温度来源，且 0.6392 与本轮 v3 switch 修法后
# 实测的 0.6204 nm² 只差 2.94%，压线落在 ±3% 门内；0.645 差 3.79%，压不过）。
# ⚠️ 目前只有口头数值，没有可追溯的文献/模拟条目引用——引用补上之前，
# `base_quality_gate.json` 里的 `literature_apl_source` 字段会如实写"unverified"。
DEFAULT_LIPID21_POPC_303K_LITERATURE_APL_NM2 = 0.6392
LITERATURE_APL_SOURCE_NOTE = (
    "0.6392 nm^2 provided 2026-08-07 during C2 review as the Lipid21 POPC @ 303.15 K "
    "reference (superseding an earlier 0.645 nm^2 placeholder with no recorded "
    "forcefield/temperature provenance); no traceable literature/simulation citation "
    "attached yet -- treat as unverified until one is added."
)

# APL/膜厚漂移显著性判据用的置信倍数（≈95% 单侧），配合 `np.polyfit(..., cov=True)`
# 给出的斜率标准误——不是"点估计压过 0.2%/ns 就是真漂移"，噪声也能压过。
DRIFT_SIGNIFICANCE_Z = 2.0

# 探针配体模板：只用净电荷 ±1 的单原子离子（C2 archive 明文只要求"固定同一个
# q=+1 或 q=-1 的 probe ligand"）。moleculetype/atom 名与
# `gromacs/toppar/Na+.itp` / `Cl-.itp` 逐位一致（resname=atom名="Na+"/"Cl-"）。
# `counter_moleculetype`/`counter_charge_e`：v2 新增的普通反离子模板，见模块 docstring §1。
ION_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "Na": {
        "moleculetype": "Na+", "charge_e": 1,
        "counter_moleculetype": "Cl-", "counter_charge_e": -1,
    },
    "Cl": {
        "moleculetype": "Cl-", "charge_e": -1,
        "counter_moleculetype": "Na+", "counter_charge_e": 1,
    },
}

PROTOCOL_VERSION = 11
LEGACY_PROTOCOL_VERSIONS = frozenset({8, 9, 10})

POSITION_VARIANT_LABELS = {0: "upper", 1: "lower"}

RESTRAINT_FORCE_CLASS_NAME = "CustomCompoundBondForce"

# v9：独立于 charge-transfer co-ion restraint 的 bulk-water corridor restraint。
# 它不使用 lambda_coul；只改变采样所处的受限系综。
BULK_RESTRAINT_FORCE_GROUP = 7
BULK_RESTRAINT_KZ_KJ_PER_MOL_NM2 = 50.0
BULK_RESTRAINT_RZ_NM = 0.5
BULK_RESTRAINT_CHARGE_FRACTION_HARD_GATE = 0.9
BULK_RESTRAINT_COORDINATION_MIN_WATER = 5
BULK_RESTRAINT_COORDINATION_FRAME_FRACTION = 0.95
BULK_RESTRAINT_MAX_ENERGY_KJ_MOL = 500.0
BULK_RESTRAINT_TARGET_OFFSET_DEFAULT_NM = 0.0
BULK_RESTRAINT_RELATIVE_Z_FLUCTUATION_LIMIT_DEFAULT_NM = 0.35
BULK_RESTRAINT_MEMBRANE_UNDULATION_MARGIN_DEFAULT_NM = 0.20
BULK_RESTRAINT_DESIGN_MIN_P31_NM = 1.10
BULK_RESTRAINT_FORM = "pair_center_dynamic_midplane_fractional_target_soft_flat_bottom"
BULK_RESTRAINT_EXPRESSION = (
    "0.25*k_z*max(0, abs(periodicdistance(x,y,z, x,y,target_z)) - r_z)^2"
)
LIGAND_BULK_RESTRAINT_FORCE_GROUP = 8
LIGAND_BULK_RESTRAINT_KZ_KJ_PER_MOL_NM2 = BULK_RESTRAINT_KZ_KJ_PER_MOL_NM2
LIGAND_BULK_RESTRAINT_RZ_NM = 0.20
LIGAND_BULK_RESTRAINT_FORM = "ligand_heavy_atom_dynamic_midplane_fractional_target_soft_flat_bottom"
LIGAND_BULK_RESTRAINT_EXPRESSION = (
    "0.5*k_lig*max(0, abs(periodicdistance(x,y,z, x,y,ligand_target_z)) - r_lig)^2"
)
LIGAND_BULK_RESTRAINT_DESIGN_ENVELOPE_MARGIN_NM = 0.20
COION_BULK_SAFETY_FORCE_GROUP = 9
COION_BULK_SAFETY_KZ_KJ_PER_MOL_NM2 = 100.0
COION_BULK_SAFETY_RZ_NM = 0.20
COION_BULK_SAFETY_FORM = "coion_member_dynamic_midplane_fractional_target_soft_flat_bottom"
COION_BULK_SAFETY_EXPRESSION = (
    "0.5*k_coion*max(0, abs(periodicdistance(x,y,z, x,y,coion_target_z)) - r_coion)^2"
)
COION_BULK_SAFETY_DESIGN_ENVELOPE_MARGIN_NM = 0.20
COORDINATION_LAMBDA0_MIN_FRAMES_V10 = 200
COORDINATION_BLOCK_SIZE_FRAMES_V10 = 20

# Hydration gate statistical definition v3.  This is an evaluator-only change:
# it does not alter the v11 Hamiltonian, build manifest, or trajectories.
# The C1 comparison is a physical non-inferiority check rather than an exact
# equality test: a small negative difference is acceptable when it remains
# within the pre-declared 0.5-water margin.
HYDRATION_GATE_STATISTICAL_VERSION = 3
HYDRATION_BOOTSTRAP_REPLICATES = 20000
HYDRATION_BOOTSTRAP_SEED = 20260810
HYDRATION_REFERENCE_NONINFERIORITY_MARGIN_WATER = 0.5
HYDRATION_SEVERE_COORDINATION_MAX = 3
HYDRATION_SEVERE_COORDINATION_MIN_CONSECUTIVE_FRAMES = 2
HYDRATION_SEVERE_R5_NM = core.COION_FIRST_SHELL_WATER_CUTOFF_NM + 0.08
HYDRATION_SEVERE_R5_MIN_CONSECUTIVE_FRAMES = 2
HYDRATION_STABILITY_MAX_HALF_MEAN_DELTA = 0.5
HYDRATION_REFERENCE_DEFAULT_PATH = os.path.join(
    _REPO_ROOT, "validation", "c1_waterbox", "Na_large", "dynamics", "timeseries.csv",
)


def _contiguous_true_runs(mask: np.ndarray) -> List[Tuple[int, int, int]]:
    """Return inclusive (start, end, length) runs of True values."""
    runs: List[Tuple[int, int, int]] = []
    start: Optional[int] = None
    for index, value in enumerate(np.asarray(mask, dtype=bool).tolist() + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1, index - start))
            start = None
    return runs


def _block_means(values: np.ndarray, block_size: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.asarray([], dtype=float)
    return np.asarray(
        [float(np.mean(block)) for block in np.array_split(
            values, max(1, int(math.ceil(values.size / block_size)))
        ) if block.size],
        dtype=float,
    )


def _bootstrap_block_mean_ci(
    block_means: np.ndarray, rng: np.random.Generator, n_replicates: int,
) -> Tuple[float, float, float]:
    """Percentile 95% CI for a mean, resampling time-block means."""
    block_means = np.asarray(block_means, dtype=float)
    if block_means.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    samples = rng.choice(
        block_means, size=(int(n_replicates), block_means.size), replace=True,
    ).mean(axis=1)
    return tuple(float(x) for x in np.percentile(samples, [2.5, 50.0, 97.5]))


def _bootstrap_block_mean_difference_ci(
    sample_blocks: np.ndarray, reference_blocks: np.ndarray,
    rng: np.random.Generator, n_replicates: int,
) -> Tuple[float, float, float]:
    """95% CI for sample mean minus reference mean using independent blocks."""
    sample_blocks = np.asarray(sample_blocks, dtype=float)
    reference_blocks = np.asarray(reference_blocks, dtype=float)
    if sample_blocks.size == 0 or reference_blocks.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    sample_means = rng.choice(
        sample_blocks, size=(int(n_replicates), sample_blocks.size), replace=True,
    ).mean(axis=1)
    reference_means = rng.choice(
        reference_blocks, size=(int(n_replicates), reference_blocks.size), replace=True,
    ).mean(axis=1)
    return tuple(float(x) for x in np.percentile(
        sample_means - reference_means, [2.5, 50.0, 97.5],
    ))


def _load_coordination_reference(path: str) -> Dict[float, np.ndarray]:
    """Load C1 coordination values grouped by matching lambda_coul."""
    import csv as csv_module

    by_lambda: Dict[float, List[float]] = {}
    if not path or not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv_module.DictReader(fh):
            if "lambda_coul" not in row or "coion_water_coordination" not in row:
                continue
            lam = round(float(row["lambda_coul"]), 6)
            by_lambda.setdefault(lam, []).append(float(row["coion_water_coordination"]))
    return {lam: np.asarray(values, dtype=float) for lam, values in by_lambda.items()}


def _water_oxygen_indices(topology) -> List[int]:
    water_names = {"HOH", "WAT", "TIP3", "TP3", "SOL"}
    indices: List[int] = []
    for atom in topology.atoms():
        element = getattr(atom, "element", None)
        if (
            atom.residue.name.upper() in water_names
            and element is not None
            and element.symbol.upper() == "O"
        ):
            indices.append(int(atom.index))
    return indices


def _trajectory_r5_values(traj, coion_index: int, water_oxygen_indices: Sequence[int]) -> np.ndarray:
    """Fifth-nearest water-O distance for every trajectory frame (nm)."""
    if len(water_oxygen_indices) < 5:
        return np.full(traj.n_frames, np.inf, dtype=float)
    values: List[float] = []
    for frame_index in range(traj.n_frames):
        box = np.diag(np.asarray(traj.unitcell_lengths[frame_index], dtype=float))
        distances = _minimum_image_distances_nm(
            traj.xyz[frame_index, list(water_oxygen_indices), :],
            traj.xyz[frame_index, coion_index, :], box,
        )
        values.append(float(np.partition(distances, 4)[4]))
    return np.asarray(values, dtype=float)



def _ion_template(ion: str) -> Dict[str, Any]:
    if ion not in ION_TEMPLATES:
        raise SystemExit(f"--ion 只接受 {sorted(ION_TEMPLATES)}，收到 {ion!r}")
    return ION_TEMPLATES[ion]


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_tree(paths: List[str]) -> Dict[str, str]:
    return {os.path.relpath(p, _REPO_ROOT): _sha256_file(p) for p in paths}


def _require_protocol_version(
    manifest: Dict[str, Any], manifest_path: str, *, allow_legacy: bool = False,
) -> None:
    version = manifest.get("protocol_version")
    accepted_versions = {PROTOCOL_VERSION}
    if allow_legacy:
        accepted_versions.update(LEGACY_PROTOCOL_VERSIONS)
    if version not in accepted_versions:
        raise SystemExit(
            f"{manifest_path} 的 protocol_version={version!r}，"
            f"与当前脚本的 PROTOCOL_VERSION={PROTOCOL_VERSION} 不符。\n"
            "    每次版本号升级都是会让结果失真/失准的硬 bug 修复（v1→v2：总电荷不为零、"
            "restraint/Hamiltonian 重复配置；v2→v3：MonteCarloMembraneBarostat + "
            "hard cutoff 的各向异性压缩伪影；v3→v4：density_profile 少除 bin 体积、"
            "extend-water 实际密度与声称的 33.33 nm⁻³ 不符、thick 输入缺 NVT 松弛阶段；"
            "v4→v5：--n-steps-nvt=0（默认值！）分支加了 barostat 却没有 "
            "Context.reinitialize，barostat 完全不生效，整段『NPT』其实是 NVT；"
            "v5→v6：insert_ions_into_gromacs_files 挑候选点只按更松的 "
            "COION_LIGAND_MIN_IMAGE_INITIAL_NM=1.6nm 判据，跟 "
            "validate_co_alchemical_ion_placement 真正要求的 runtime 判据"
            "（默认 restraint 参数下约 2.02nm）脱节；v6→v7："
            "_find_bulk_water_candidates 算候选水离膜中面距离时没做 z 轴周期"
            "折叠，未折叠坐标的候选被系统性算出虚高距离、被贪心算法优先选中"
            "——真实 GPU pilot 里探针几十 ps 内就逼近磷原子，见模块 docstring "
            "v6→v7 一条）；v7→v8：中性 co-ion 不再使用全程 hydration hard gate，"
            "并加入新的动态膜中面 bulk-water restraint，旧版本产物必须作废重建，"
            "不能跳过版本检查继续用；v8→v9：target 改为当前 Lz 的有符号分数，"
            "膜侧门改为 no_membrane_core_crossing，旧 v8 轨迹只允许质量门 legacy 重评；"
            "v10→v11：fully charged co-ion 暴露出 pair-center/ligand safety 无法控制的"
            "成员级相对移动，新增独立 PBC-aware co-ion safety wall，v10 产物不得混入 v11。"
        )


def _minimum_image_distance_nm(p1_nm: np.ndarray, p2_nm: np.ndarray, box_nm: np.ndarray) -> float:
    inv_box = np.linalg.inv(box_nm)
    delta = np.asarray(p1_nm) - np.asarray(p2_nm)
    frac = delta @ inv_box
    frac -= np.round(frac)
    return float(np.linalg.norm(frac @ box_nm))


def _minimum_image_distances_nm(points_nm: np.ndarray, ref_nm: np.ndarray, box_nm: np.ndarray) -> np.ndarray:
    """`points_nm` (N,3) 相对单点 `ref_nm` 的逐点 minimum-image 距离。"""
    inv_box = np.linalg.inv(box_nm)
    delta = np.asarray(points_nm, dtype=np.float64) - np.asarray(ref_nm, dtype=np.float64)
    frac = delta @ inv_box
    frac -= np.round(frac)
    mic = frac @ box_nm
    return np.linalg.norm(mic, axis=1)


def _box_volume_nm3(box_nm: np.ndarray) -> float:
    return float(abs(np.linalg.det(np.asarray(box_nm, dtype=np.float64))))


def _z_number_density_profile(
    z_values_nm: np.ndarray, n_frames: int, bins_nm: np.ndarray, bin_volume_nm3: float,
) -> List[float]:
    """给定某个原子组在若干帧上的全部 z 坐标（已展平成 1D），按 `bins_nm` 分箱、
    除以帧数、再除以 `bin_volume_nm3`，得到真正的数密度（nm⁻³）。

    v3→v4 §4 修的就是这一步：之前只除了帧数（`counts / n_frames`），没除
    `bin_volume_nm3`，量纲上是"每帧平均原子计数"，不是数密度，不能跟"约
    33 nm⁻³ 体相水"这类文献值直接比较。拆成独立函数是为了能脱离
    `cmd_base_quality_gate` 需要的完整 GROMACS 拓扑/DCD，单独用合成 z 坐标
    回归测试这一步除法。

    `z_values_nm.size == 0`（该组分在这份轨迹里不存在，例如某个 case 没有
    脂尾碳原子）时直接返回全 0，不尝试对空数组分箱。
    """
    n_bins = len(bins_nm) - 1
    if z_values_nm.size == 0:
        return [0.0] * n_bins
    counts, _ = np.histogram(z_values_nm, bins=bins_nm)
    counts_per_frame = counts / max(int(n_frames), 1)
    return (counts_per_frame / bin_volume_nm3).tolist()


def _get_platform(platform_name: str, allow_cpu_fallback: bool, precision: str):
    """按 `--allow-cpu-fallback` 决定 CUDA 不可用时是 fail closed 还是回退 CPU。

    默认（`allow_cpu_fallback=False`）：CUDA 建不出来直接抛错，**不**静默换成
    CPU 跑一条本该几纳秒的 GPU 生产轨迹——那样跑出来的东西慢到没有代表性，
    而且日志不显眼的话很容易被当成"跑完了"。只有显式传 `--allow-cpu-fallback`
    （用于秒级自检、不是生产/pilot 采样）才允许回退。
    """
    try:
        platform = openmm.Platform.getPlatformByName(platform_name)
        properties = {"Precision": precision} if platform_name == "CUDA" else {}
        return platform, properties, platform_name, None
    except Exception as exc:  # noqa: BLE001
        if not allow_cpu_fallback:
            raise RuntimeError(
                f"平台 {platform_name} 不可用（{exc}），且未传 --allow-cpu-fallback——"
                "拒绝静默回退 CPU 跑生产/pilot 级别的动力学。"
            ) from exc
        print(f"⚠️  平台 {platform_name} 不可用（{exc}），已显式允许回退 CPU")
        return openmm.Platform.getPlatformByName("CPU"), {}, "CPU", str(exc)


# ============================================================================
# GRO 文本读写（定宽格式：5/5/5/5/8/8/8，无速度列）
# ============================================================================


def _read_gro(gro_path: str) -> Tuple[str, int, List[str], str]:
    with open(gro_path, encoding="utf-8") as fh:
        lines = fh.readlines()
    title = lines[0]
    n_atoms = int(lines[1].strip())
    atom_lines = lines[2 : 2 + n_atoms]
    if len(atom_lines) != n_atoms:
        raise ValueError(
            f"{gro_path}: 声明 {n_atoms} 个原子，实际只有 {len(atom_lines)} 行原子记录"
        )
    box_line = lines[2 + n_atoms]
    return title, n_atoms, atom_lines, box_line


def _format_gro_atom_line(
    resnum: int, resname: str, atomname: str, atomnum: int, xyz_nm: np.ndarray
) -> str:
    return (
        f"{resnum % 100000:5d}{resname:<5.5s}{atomname:>5.5s}{atomnum % 100000:5d}"
        f"{float(xyz_nm[0]):8.3f}{float(xyz_nm[1]):8.3f}{float(xyz_nm[2]):8.3f}\n"
    )


def _write_gro(out_path: str, title: str, atom_lines: List[str], box_line: str) -> None:
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(title if title.endswith("\n") else title + "\n")
        fh.write(f"{len(atom_lines)}\n")
        fh.writelines(atom_lines)
        fh.write(box_line if box_line.endswith("\n") else box_line + "\n")


def _edit_top_molecules_block(
    top_path: str,
    output_path: str,
    water_moleculetype: str,
    water_delta: int,
    appended_blocks: Sequence[Tuple[str, int]] = (),
) -> None:
    """复制 `top_path`，把 `[ molecules ]` 里水的计数改 `water_delta`，
    并在块尾依次追加 `appended_blocks` 里的每个 `(moleculetype, count)`。

    `appended_blocks` 的顺序就是新分子在 `.top`/`.gro` 里出现的顺序（GROMACS
    原子顺序 = `[ molecules ]` 展开顺序），调用方必须让它与 `.gro` 里追加的
    原子行顺序逐一对应。
    """
    with open(top_path, encoding="utf-8") as fh:
        lines = fh.readlines()

    out: List[str] = []
    in_molecules = False
    water_found = False
    for line in lines:
        stripped = line.split(";", 1)[0].strip()
        lowered = stripped.lower()
        if lowered in ("[ molecules ]", "[molecules]"):
            in_molecules = True
            out.append(line)
            continue
        if in_molecules and stripped.startswith("[") and lowered not in (
            "[ molecules ]", "[molecules]",
        ):
            in_molecules = False
        if in_molecules and stripped and not stripped.startswith(";"):
            parts = stripped.split()
            if len(parts) >= 2 and parts[0] == water_moleculetype:
                count = int(parts[1])
                new_count = count + water_delta
                if new_count < 0:
                    raise ValueError(
                        f"{water_moleculetype} 计数改 {water_delta:+d} 后变成负数（原 {count}）"
                    )
                out.append(f"{water_moleculetype}\t{new_count}\n")
                water_found = True
                continue
        out.append(line)

    if not water_found:
        raise ValueError(f"在 {top_path} 的 [ molecules ] 里没找到 {water_moleculetype!r} 这一行")
    for moleculetype, count in appended_blocks:
        if count > 0:
            out.append(f"{moleculetype}\t{count}\n")

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.writelines(out)


def _write_equilibrated_gro(template_gro_file: str, positions, box_vectors, out_path: str) -> None:
    """用某份已有 `.gro` 当"文本模板"（复用其原子行的 resnum/resname/atomname/
    atomnum 字段），只替换坐标和盒矢量——保证输出仍是同一批原子、同一顺序。
    """
    title, n_atoms, atom_lines, _old_box_line = _read_gro(template_gro_file)
    pos_nm = np.asarray(positions.value_in_unit(unit.nanometer), dtype=np.float64)
    if pos_nm.shape[0] != n_atoms:
        raise ValueError(
            f"坐标原子数 {pos_nm.shape[0]} 与模板 {template_gro_file} 的 {n_atoms} 不符"
        )
    new_lines = []
    for i, line in enumerate(atom_lines):
        prefix = line[0:20]
        new_lines.append(f"{prefix}{pos_nm[i,0]:8.3f}{pos_nm[i,1]:8.3f}{pos_nm[i,2]:8.3f}\n")
    box_nm = np.asarray([v.value_in_unit(unit.nanometer) for v in box_vectors])
    box_line = _format_gro_box_line(box_nm)
    _write_gro(out_path, title, new_lines, box_line)


def _format_gro_box_line(box_nm: np.ndarray) -> str:
    return (
        f"{box_nm[0,0]:10.5f}{box_nm[1,1]:10.5f}{box_nm[2,2]:10.5f}"
        f"{box_nm[0,1]:10.5f}{box_nm[0,2]:10.5f}{box_nm[1,0]:10.5f}"
        f"{box_nm[1,2]:10.5f}{box_nm[2,0]:10.5f}{box_nm[2,1]:10.5f}\n"
    )


# ============================================================================
# 力场族/dispersion protocol 校验（build 与 equilibrate-base 都要独立重新跑一遍，
# 不允许任何一处把结论硬编码进 manifest——见执行清单 §1「其余执行安全修改」）
# ============================================================================


def _resolve_and_verify_dispersion_protocol(top_file: str, include_dir: Optional[str]) -> Dict[str, Any]:
    ff_family_report = core.detect_forcefield_family_from_top(top_file, include_dir)
    ff_family = ff_family_report.get("family")
    if not ff_family:
        raise RuntimeError(
            f"{top_file} 的力场族识别不出来：{ff_family_report}。"
            "fail closed——不回落 amber（MEM-00k/§1.1）。"
        )
    dispersion = core.resolve_dispersion_protocol(
        core.DISPERSION_PROTOCOL_FF_NATIVE_ISOTROPIC_LRC,
        environment_type=core.ENVIRONMENT_TYPE_MEMBRANE,
        forcefield_family=ff_family,
    )
    dispersion["forcefield_family_report"] = ff_family_report
    return dispersion


# ============================================================================
# bulk-water 候选点筛选（复用 abfe_core 既有的 §13.1 常量，不新造阈值）
# ============================================================================


def _lipid_phosphorus_indices(topology) -> List[int]:
    # 复用生产共享常量而不是硬编码 "P31"——这份 slab 恰好是 Lipid21（用的就是
    # P31），但换一个不同命名的脂质力场时应当自动认得，不是再改一处硬编码字符串。
    head_names = set(core.LIPID_HEAD_REFERENCE_ATOM_NAMES)
    indices = [
        atom.index for atom in topology.atoms()
        if str(atom.name).strip().upper() in head_names
    ]
    if not indices:
        raise RuntimeError(
            f"找不到任何头基参考原子（{sorted(head_names)}）——本 slab 用的是 "
            "Amber Lipid21 命名（`abfe_core.LIPID_HEAD_REFERENCE_ATOM_NAMES` 已含 'P31'），"
            "找不到说明拓扑不是预期的这份 CHARMM-GUI POPC slab。"
        )
    return indices


def _water_oxygen_indices_with_residue(topology) -> List[Tuple[int, Any]]:
    out = []
    for residue in topology.residues():
        if str(residue.name).strip().upper() not in core.WATER_MOLECULE_NAMES:
            continue
        atoms = list(residue.atoms())
        o_atom = next((a for a in atoms if str(a.name).strip().upper() == "O"), None)
        if o_atom is not None:
            out.append((o_atom.index, residue))
    return out


def _minimum_image_z_delta_nm(z_nm: float, reference_z_nm: float, box_z_nm: float) -> float:
    """`z_nm - reference_z_nm` 沿 z 轴单轴折进 `[-box_z/2, box_z/2)` 的
    minimum-image 差值（**不是**绝对值——正负号保留，供"side"判断使用）。

    v6→v7 修复：`.gro` 里的坐标是 OpenMM/GROMACS 跑出来的原始坐标，长时间
    模拟下扩散穿过周期边界的原子**不会**被自动折回 `[0, box_z)`——本仓库
    实测 `base_thin_v3_extend1/equilibrated.gro` 约 24% 的原子 z 坐标落在
    `[0, box_z)` 之外。直接算 `z - reference_z` 这种非周期性差值对这些原子
    完全失真：某候选水报的是"离中面 5.448 nm"（几何上不可能——已经超过
    `box_z/2≈4.17 nm`，任何点到膜中面的真实周期最短距离都不可能超过半个
    盒高），真实 minimum-image 距离只有 2.887 nm，低于 bulk-water 安全下限
    3.0 nm。`_pick_n_well_separated` 的"farthest-first"贪心排序又偏好
    `abs_dz` 最大的候选，于是系统性地优先选中这些被错误判定"最深"、实际上
    离膜很近的候选。
    """
    dz = float(z_nm) - float(reference_z_nm)
    return dz - float(box_z_nm) * round(dz / float(box_z_nm))


def _continuous_unwrap_fractional(values: Sequence[float]) -> np.ndarray:
    """Unwrap a scalar periodic coordinate expressed as a fraction of the box."""
    wrapped = np.mod(np.asarray(values, dtype=np.float64), 1.0)
    if wrapped.size == 0:
        return wrapped.copy()
    out = np.empty_like(wrapped)
    out[0] = wrapped[0]
    for i in range(1, wrapped.size):
        step = wrapped[i] - wrapped[i - 1]
        step -= round(float(step))
        out[i] = out[i - 1] + step
    return out


def _pbc_pair_center_z_nm(
    positions_nm: np.ndarray, ligand_index: int, coion_index: int, box_z_nm: float,
) -> float:
    """Return the pair-center z using the ligand as the PBC image anchor.

    The co-ion is first moved to the minimum-image z relative to the ligand.  This
    is deliberately not ``(z_ligand + z_coion)/2``: a pair straddling the z
    boundary must still have a center in the same local image as the ligand.
    """
    ligand_z = float(positions_nm[int(ligand_index), 2])
    coion_z = float(positions_nm[int(coion_index), 2])
    coion_delta = _minimum_image_z_delta_nm(coion_z, ligand_z, box_z_nm)
    return ligand_z + 0.5 * coion_delta


def _bulk_target_fraction_from_initial_geometry(
    positions_nm: np.ndarray,
    ligand_index: int,
    coion_index: int,
    midplane_z_nm: float,
    box_z_nm: float,
    target_offset_toward_water_nm: float,
) -> Tuple[float, float, float]:
    """Return target fraction, initial Δz, and offset target Δz.

    The offset is applied away from the membrane core.  The shifted signed
    displacement is intentionally allowed to exceed ±Lz/2; the OpenMM force
    evaluates the resulting pair-target distance with minimum-image PBC.
    """
    box_z = float(box_z_nm)
    offset = float(target_offset_toward_water_nm)
    if not math.isfinite(box_z) or box_z <= 0.0:
        raise ValueError(f"box_z_nm must be positive finite, got {box_z_nm!r}")
    if not math.isfinite(offset) or offset < 0.0:
        raise ValueError(f"target offset must be finite and non-negative, got {offset!r}")
    pair_center_z_nm = _pbc_pair_center_z_nm(
        positions_nm, ligand_index, coion_index, box_z,
    )
    initial_signed_delta = _minimum_image_z_delta_nm(
        pair_center_z_nm, float(midplane_z_nm), box_z,
    )
    if abs(initial_signed_delta) <= 1.0e-12:
        raise ValueError("pair-center target side is undefined because initial signed Δz is zero")
    shifted_signed_delta = initial_signed_delta + math.copysign(offset, initial_signed_delta)
    return (
        float(shifted_signed_delta / box_z),
        float(initial_signed_delta),
        float(shifted_signed_delta),
    )


def _static_bulk_geometry_design(
    positions_nm: np.ndarray,
    p31_indices: Sequence[int],
    ligand_index: int,
    coion_index: int,
    midplane_z_nm: float,
    box_z_nm: float,
    signed_target_fraction: float,
    r_z_nm: float,
    relative_z_fluctuation_limit_nm: float,
    membrane_undulation_margin_nm: float,
    box_nm: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Check the v9 pair-center well before dynamics.

    The P31 design distance uses the initial pair-member x/y coordinates and
    the full minimum-image 3-D distance while sweeping the allowed z envelope;
    the production gate still checks the actual full trajectory frame by frame.
    """
    box_z = float(box_z_nm)
    r_z = float(r_z_nm)
    relative_limit = float(relative_z_fluctuation_limit_nm)
    undulation_margin = float(membrane_undulation_margin_nm)
    if not all(math.isfinite(x) and x >= 0.0 for x in (r_z, relative_limit, undulation_margin)):
        raise ValueError("v10 bulk geometry margins/radius must be finite and non-negative")
    initial_pair_center = _pbc_pair_center_z_nm(positions_nm, ligand_index, coion_index, box_z)
    relative_z = _minimum_image_z_delta_nm(
        float(positions_nm[coion_index, 2]), float(positions_nm[ligand_index, 2]), box_z,
    )
    initial_half_span = 0.5 * abs(relative_z)
    member_envelope = initial_half_span + relative_limit
    target_delta = float(signed_target_fraction) * box_z
    midplane = float(midplane_z_nm)
    candidate_centers = (midplane + target_delta - r_z, midplane + target_delta + r_z)
    candidate_member_z = [
        center + member_offset
        for center in candidate_centers
        for member_offset in (-member_envelope, member_envelope)
    ]
    p31_z = np.asarray(positions_nm, dtype=np.float64)[np.asarray(p31_indices, dtype=int), 2]
    min_abs_midplane = min(
        abs(_minimum_image_z_delta_nm(z, midplane, box_z)) for z in candidate_member_z
    )
    box_matrix = np.diag([1.0, 1.0, box_z]) if box_nm is None else np.asarray(box_nm, dtype=float)
    inv_box = np.linalg.inv(box_matrix)
    min_p31_z = np.inf
    for member_index in (int(ligand_index), int(coion_index)):
        for z in candidate_member_z:
            member = np.asarray(positions_nm[member_index], dtype=float).copy()
            member[2] = z
            delta = np.asarray(positions_nm)[np.asarray(p31_indices, dtype=int)] - member
            fractional = delta @ inv_box
            fractional -= np.round(fractional)
            distances = np.linalg.norm(fractional @ box_matrix, axis=1)
            min_p31_z = min(min_p31_z, float(np.min(distances)))
    required_midplane = core.COION_MEMBRANE_MIDPLANE_MIN_ABS_Z_NM + undulation_margin
    required_p31 = BULK_RESTRAINT_DESIGN_MIN_P31_NM + undulation_margin
    passed = bool(min_abs_midplane >= required_midplane and min_p31_z >= required_p31)
    return {
        "passed": passed,
        "initial_pair_center_z_nm": float(initial_pair_center),
        "signed_target_fraction": float(signed_target_fraction),
        "target_signed_delta_z_nm": float(target_delta),
        "pair_center_flat_bottom_radius_nm": r_z,
        "initial_pair_half_span_z_nm": float(initial_half_span),
        "relative_z_fluctuation_limit_nm": relative_limit,
        "member_z_envelope_nm": float(member_envelope),
        "membrane_undulation_margin_nm": undulation_margin,
        "min_worst_case_abs_dz_from_midplane_nm": float(min_abs_midplane),
        "required_design_abs_dz_from_midplane_nm": float(required_midplane),
        "min_worst_case_nearest_p31_z_nm": float(min_p31_z),
        "p31_design_metric": "3d_minimum_image_with_initial_xy_and_z_envelope",
        "required_design_nearest_p31_nm": float(required_p31),
    }


def _static_ligand_geometry_design(
    positions_nm: np.ndarray,
    p31_indices: Sequence[int],
    ligand_indices: Sequence[int],
    midplane_z_nm: float,
    box_z_nm: float,
    signed_target_fraction: float,
    r_z_nm: float,
    ligand_envelope_margin_nm: float,
    box_nm: np.ndarray,
) -> Dict[str, Any]:
    """Check the worst heavy-atom ligand envelope around its reference atom."""
    ligand_indices = [int(i) for i in ligand_indices]
    if not ligand_indices:
        raise ValueError("ligand safety geometry requires at least one ligand atom")
    box_z = float(box_z_nm)
    box_matrix = np.asarray(box_nm, dtype=float)
    inv_box = np.linalg.inv(box_matrix)
    reference_index = ligand_indices[0]
    reference_z = float(positions_nm[reference_index, 2])
    offsets = np.asarray([
        _minimum_image_z_delta_nm(float(positions_nm[i, 2]), reference_z, box_z)
        for i in ligand_indices
    ])
    envelope = float(np.max(np.abs(offsets)) + float(ligand_envelope_margin_nm))
    target_z = float(midplane_z_nm) + float(signed_target_fraction) * box_z
    candidate_reference_z = (target_z - float(r_z_nm), target_z + float(r_z_nm))
    # Enumerate the reference well endpoints, apply each atom's frozen relative
    # offset, and add the declared conformation margin symmetrically.
    candidate_atoms = [
        z + offset + margin
        for z in candidate_reference_z
        for offset in offsets
        for margin in (-float(ligand_envelope_margin_nm), float(ligand_envelope_margin_nm))
    ]
    p31_xyz = np.asarray(positions_nm, dtype=float)[np.asarray(p31_indices, dtype=int)]
    min_abs_midplane = min(
        abs(_minimum_image_z_delta_nm(z, float(midplane_z_nm), box_z)) for z in candidate_atoms
    )
    min_p31 = np.inf
    for atom_index, offset in zip(ligand_indices, offsets):
        for ref_z in candidate_reference_z:
            for margin in (-float(ligand_envelope_margin_nm), float(ligand_envelope_margin_nm)):
                point = np.asarray(positions_nm[atom_index], dtype=float).copy()
                point[2] = ref_z + float(offset) + margin
                delta = p31_xyz - point
                fractional = delta @ inv_box
                fractional -= np.round(fractional)
                min_p31 = min(min_p31, float(np.min(np.linalg.norm(fractional @ box_matrix, axis=1))))
    required_midplane = core.COION_MEMBRANE_MIDPLANE_MIN_ABS_Z_NM + BULK_RESTRAINT_MEMBRANE_UNDULATION_MARGIN_DEFAULT_NM
    required_p31 = BULK_RESTRAINT_DESIGN_MIN_P31_NM + BULK_RESTRAINT_MEMBRANE_UNDULATION_MARGIN_DEFAULT_NM
    return {
        "passed": bool(min_abs_midplane >= required_midplane and min_p31 >= required_p31),
        "reference_index": reference_index,
        "r_z_nm": float(r_z_nm),
        "ligand_envelope_margin_nm": float(ligand_envelope_margin_nm),
        "max_initial_heavy_atom_relative_z_nm": float(np.max(np.abs(offsets))),
        "worst_case_ligand_envelope_nm": envelope,
        "min_worst_case_abs_dz_from_midplane_nm": float(min_abs_midplane),
        "required_design_abs_dz_from_midplane_nm": float(required_midplane),
        "min_worst_case_nearest_p31_nm": float(min_p31),
        "required_design_nearest_p31_nm": float(required_p31),
        "design_metric": "ligand_reference_plus_initial_heavy_atom_z_envelope_3d_minimum_image",
    }


def _bulk_restraint_target_z_nm(
    positions_nm: np.ndarray,
    ligand_index: int,
    coion_index: int,
    p31_indices: Sequence[int],
    box_z_nm: float,
    signed_target_fraction: float,
) -> Tuple[float, float, float]:
    """Compute the box-fractional dynamic bulk-restraint target."""
    midplane_z_nm = float(np.mean(np.asarray(positions_nm)[np.asarray(p31_indices), 2]))
    pair_center_z_nm = _pbc_pair_center_z_nm(
        positions_nm, ligand_index, coion_index, box_z_nm,
    )
    desired_target = midplane_z_nm + float(signed_target_fraction) * box_z_nm
    ligand_z = float(positions_nm[int(ligand_index), 2])
    desired_target += box_z_nm * round((ligand_z - desired_target) / box_z_nm)
    return desired_target, pair_center_z_nm, midplane_z_nm


def _create_bulk_restraint_force(
    ligand_index: int,
    coion_index: int,
    target_z_nm: float,
    k_z_kj_per_mol_nm2: float = BULK_RESTRAINT_KZ_KJ_PER_MOL_NM2,
    r_z_nm: float = BULK_RESTRAINT_RZ_NM,
) -> openmm.CustomExternalForce:
    """Create the v9 λ-independent PBC-aware bulk-water pair restraint.

    Each pair atom receives one quarter of ``k_z``.  When the pair translates
    together, the two contributions sum to the requested ``0.5*k_z`` harmonic
    wall on the pair center.  ``periodicdistance`` supplies the z minimum-image
    displacement; the x/y arguments are identical so only the membrane-normal
    displacement contributes.  The target is updated from the dynamic P31
    midplane during dynamics, while the force definition itself remains in the
    build System and therefore is also present in analysis Systems.
    """
    if int(ligand_index) == int(coion_index):
        raise ValueError("bulk-water pair restraint requires distinct ligand/co-ion indices")
    if not (math.isfinite(float(k_z_kj_per_mol_nm2)) and float(k_z_kj_per_mol_nm2) > 0.0):
        raise ValueError(f"bulk restraint kZ must be positive finite, got {k_z_kj_per_mol_nm2!r}")
    if not (math.isfinite(float(r_z_nm)) and float(r_z_nm) >= 0.0):
        raise ValueError(f"bulk restraint rZ must be non-negative finite, got {r_z_nm!r}")
    force = openmm.CustomExternalForce(BULK_RESTRAINT_EXPRESSION)
    force.addGlobalParameter("k_z", float(k_z_kj_per_mol_nm2))
    force.addGlobalParameter("r_z", float(r_z_nm))
    force.addGlobalParameter("target_z", float(target_z_nm))
    force.addParticle(int(ligand_index), [])
    force.addParticle(int(coion_index), [])
    force.setForceGroup(BULK_RESTRAINT_FORCE_GROUP)
    return force


def _bulk_restraint_forces(system: openmm.System) -> List[openmm.CustomExternalForce]:
    return [
        f for f in system.getForces()
        if isinstance(f, openmm.CustomExternalForce)
        and f.getForceGroup() == BULK_RESTRAINT_FORCE_GROUP
    ]


def _assert_single_bulk_restraint_force(system: openmm.System) -> openmm.CustomExternalForce:
    matches = _bulk_restraint_forces(system)
    if len(matches) != 1:
        raise SystemExit(
            f"v10 pair bulk-water restraint force 数量={len(matches)}，应为 1；"
            "不得漏加或重复加。"
        )
    return matches[0]


def _create_ligand_bulk_safety_force(
    ligand_reference_index: int,
    target_z_nm: float,
    k_z_kj_per_mol_nm2: float = LIGAND_BULK_RESTRAINT_KZ_KJ_PER_MOL_NM2,
    r_z_nm: float = LIGAND_BULK_RESTRAINT_RZ_NM,
) -> openmm.CustomExternalForce:
    """Create a λ-independent wall on the ligand reference/COM coordinate."""
    if not (math.isfinite(float(k_z_kj_per_mol_nm2)) and float(k_z_kj_per_mol_nm2) > 0.0):
        raise ValueError("ligand safety wall kZ must be positive finite")
    if not (math.isfinite(float(r_z_nm)) and float(r_z_nm) >= 0.0):
        raise ValueError("ligand safety wall rZ must be non-negative finite")
    force = openmm.CustomExternalForce(LIGAND_BULK_RESTRAINT_EXPRESSION)
    force.addGlobalParameter("k_lig", float(k_z_kj_per_mol_nm2))
    force.addGlobalParameter("r_lig", float(r_z_nm))
    force.addGlobalParameter("ligand_target_z", float(target_z_nm))
    force.addParticle(int(ligand_reference_index), [])
    force.setForceGroup(LIGAND_BULK_RESTRAINT_FORCE_GROUP)
    return force


def _ligand_bulk_safety_forces(system: openmm.System) -> List[openmm.CustomExternalForce]:
    return [
        f for f in system.getForces()
        if isinstance(f, openmm.CustomExternalForce)
        and f.getForceGroup() == LIGAND_BULK_RESTRAINT_FORCE_GROUP
    ]


def _assert_single_ligand_bulk_safety_force(system: openmm.System) -> openmm.CustomExternalForce:
    matches = _ligand_bulk_safety_forces(system)
    if len(matches) != 1:
        raise SystemExit(
            f"v10 ligand bulk safety force 数量={len(matches)}，应为 1；"
            "不得漏加或重复加。"
        )
    return matches[0]


def _create_coion_bulk_safety_force(
    coion_index: int,
    target_z_nm: float,
    k_z_kj_per_mol_nm2: float = COION_BULK_SAFETY_KZ_KJ_PER_MOL_NM2,
    r_z_nm: float = COION_BULK_SAFETY_RZ_NM,
) -> openmm.CustomExternalForce:
    """Create the v11 λ-independent member-level co-ion bulk safety wall.

    This is deliberately separate from the pair-center restraint: a charged co-ion
    can move relative to the ligand inside a pair-center well, so its own PBC-aware
    Z wall must independently keep it in the bulk-water corridor.
    """
    if not (math.isfinite(float(k_z_kj_per_mol_nm2)) and float(k_z_kj_per_mol_nm2) > 0.0):
        raise ValueError("co-ion safety wall kZ must be positive finite")
    if not (math.isfinite(float(r_z_nm)) and float(r_z_nm) >= 0.0):
        raise ValueError("co-ion safety wall rZ must be non-negative finite")
    force = openmm.CustomExternalForce(COION_BULK_SAFETY_EXPRESSION)
    force.addGlobalParameter("k_coion", float(k_z_kj_per_mol_nm2))
    force.addGlobalParameter("r_coion", float(r_z_nm))
    force.addGlobalParameter("coion_target_z", float(target_z_nm))
    force.addParticle(int(coion_index), [])
    force.setForceGroup(COION_BULK_SAFETY_FORCE_GROUP)
    return force


def _coion_bulk_safety_forces(system: openmm.System) -> List[openmm.CustomExternalForce]:
    return [
        f for f in system.getForces()
        if isinstance(f, openmm.CustomExternalForce)
        and f.getForceGroup() == COION_BULK_SAFETY_FORCE_GROUP
    ]


def _assert_single_coion_bulk_safety_force(system: openmm.System) -> openmm.CustomExternalForce:
    matches = _coion_bulk_safety_forces(system)
    if len(matches) != 1:
        raise SystemExit(
            f"v11 co-ion bulk safety force 数量={len(matches)}，应为 1；"
            "不得漏加或重复加。"
        )
    return matches[0]


def _update_bulk_restraint_target(
    context: openmm.Context,
    positions_nm: np.ndarray,
    box_nm: np.ndarray,
    p31_indices: Sequence[int],
    ligand_index: int,
    coion_index: int,
    signed_target_fraction: float,
) -> Dict[str, float]:
    """Update the λ-independent target from P31 midplane and current box height."""
    box_z_nm = float(np.asarray(box_nm, dtype=float)[2, 2])
    target_z_nm, pair_center_z_nm, midplane_z_nm = _bulk_restraint_target_z_nm(
        positions_nm, ligand_index, coion_index, p31_indices, box_z_nm,
        signed_target_fraction,
    )
    context.setParameter("target_z", float(target_z_nm))
    try:
        context.setParameter("ligand_target_z", float(target_z_nm))
    except openmm.OpenMMException:
        # Legacy v8/v9 systems have no separate ligand safety force.
        pass
    try:
        context.setParameter("coion_target_z", float(target_z_nm))
    except openmm.OpenMMException:
        # Legacy v8-v10 systems have no separate co-ion member safety force.
        pass
    return {
        "target_z_nm": float(target_z_nm),
        "pair_center_z_nm": float(pair_center_z_nm),
        "midplane_z_nm": float(midplane_z_nm),
        "pair_center_signed_delta_z_nm": float(
            _minimum_image_z_delta_nm(pair_center_z_nm, midplane_z_nm, box_z_nm)
        ),
        "pair_center_target_displacement_nm": float(
            _minimum_image_z_delta_nm(pair_center_z_nm, target_z_nm, box_z_nm)
        ),
    }


def _load_restart_frame_from_dcd(
    dcd_path: str, topology, frame_index: int = -1,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Load one saved DCD frame for a continuation/confirmation segment.

    DCD coordinates are used only as the new initial state; the Hamiltonian is
    still configured from the untouched build System and build coordinates.
    """
    import mdtraj as md

    if not os.path.exists(dcd_path):
        raise SystemExit(f"restart DCD 不存在: {dcd_path}")
    traj = md.load_dcd(dcd_path, top=md.Topology.from_openmm(topology))
    if traj.n_frames == 0:
        raise SystemExit(f"restart DCD 没有帧: {dcd_path}")
    selected = int(frame_index)
    if selected < 0:
        selected += int(traj.n_frames)
    if selected < 0 or selected >= traj.n_frames:
        raise SystemExit(
            f"restart frame={frame_index} 超出 {dcd_path} 的范围 [0,{traj.n_frames - 1}]"
        )
    if traj.unitcell_lengths is None:
        raise SystemExit(f"restart DCD 缺少 unitcell_lengths，不能安全续跑: {dcd_path}")
    box_lengths = np.asarray(traj.unitcell_lengths[selected], dtype=np.float64)
    if box_lengths.shape != (3,) or not np.all(np.isfinite(box_lengths)) or np.any(box_lengths <= 0.0):
        raise SystemExit(f"restart DCD 的盒长无效: {box_lengths}")
    return (
        np.asarray(traj.xyz[selected], dtype=np.float64),
        np.diag(box_lengths),
        selected,
        int(traj.n_frames),
    )


def _find_bulk_water_candidates(
    positions_nm: np.ndarray,
    box_nm: np.ndarray,
    midplane_z_nm: float,
    phosphorus_indices: List[int],
    water_oxygens: List[Tuple[int, Any]],
) -> List[Dict[str, Any]]:
    """v6→v7 修复：`abs_dz_from_midplane_nm`/`side` 现在用
    `_minimum_image_z_delta_nm` 折叠过的 z 差值算，不是原始坐标的非周期性
    差值——理由见该函数 docstring。这个假设（z 轴单轴折叠足够）依赖膜法向
    沿 z、盒子在 z 方向不被 x/y 的斜切分量污染，跟本文件其余部分
    （`_format_gro_box_line`/膜法向校验）对这份 CHARMM-GUI 直角盒的假设
    一致，不是新引入的近似。
    """
    p_positions = positions_nm[np.asarray(phosphorus_indices, dtype=int)]
    box_z_nm = float(box_nm[2, 2])
    candidates: List[Dict[str, Any]] = []
    for o_index, residue in water_oxygens:
        z = float(positions_nm[o_index][2])
        dz = _minimum_image_z_delta_nm(z, midplane_z_nm, box_z_nm)
        abs_dz = abs(dz)
        if abs_dz < core.COION_MEMBRANE_MIDPLANE_MIN_ABS_Z_NM:
            continue
        distances = _minimum_image_distances_nm(p_positions, positions_nm[o_index], box_nm)
        nearest_p_nm = float(np.min(distances))
        if nearest_p_nm < core.COION_NEAREST_PHOSPHORUS_MIN_NM:
            continue
        candidates.append(
            {
                "o_index": int(o_index), "residue": residue, "z_nm": z,
                "abs_dz_from_midplane_nm": abs_dz, "nearest_phosphorus_nm": nearest_p_nm,
                "side": "upper" if dz > 0 else "lower",
            }
        )
    return candidates


def _pick_n_well_separated(
    pool: List[Dict[str, Any]], n: int, positions_nm: np.ndarray, box_nm: np.ndarray, min_mutual_nm: float,
    validator: Optional[Callable[[List[Dict[str, Any]]], bool]] = None,
) -> List[Dict[str, Any]]:
    """贪心 farthest-first：先取离双层中面最深的候选，再依次追加与**已选中全部**
    候选 minimum-image 距离都 ≥ `min_mutual_nm` 的下一个，直到凑够 `n` 个。

    与 `runabfe._insert_reserved_coalchemical_ion_dummies` 同一个思路（§2.2 教训：
    只按"离配体最远"独立打分会让多个候选挤在同一个远角）。
    """
    if len(pool) < n:
        raise RuntimeError(
            f"这一侧（side pool）只有 {len(pool)} 个满足 bulk-water 判据的候选水分子，"
            f"不够选出 {n} 个点位。考虑放宽判据或换 --position-variant。"
        )
    ordered = sorted(pool, key=lambda c: -c["abs_dz_from_midplane_nm"])

    def compatible(cand: Dict[str, Any], chosen: List[Dict[str, Any]]) -> bool:
        return all(
            _minimum_image_distance_nm(positions_nm[cand["o_index"]], positions_nm[c["o_index"]], box_nm)
            >= min_mutual_nm
            for c in chosen
        )

    # Keep the historical farthest-first result when no additional geometric
    # validator is requested.
    if validator is None:
        chosen: List[Dict[str, Any]] = [ordered[0]]
        for cand in ordered[1:]:
            if len(chosen) == n:
                break
            if compatible(cand, chosen):
                chosen.append(cand)
        if len(chosen) < n:
            raise RuntimeError(
                f"在这一侧找不到 {n} 个彼此 minimum-image 距离 ≥ {min_mutual_nm:.3f} nm 的候选"
                f"（只凑到 {len(chosen)} 个）。"
            )
        return chosen

    # v11 placement selection: the farthest candidate can be close to P31 in
    # x/y even when its initial |Δz| is excellent.  Search combinations in
    # farthest-first order and accept only a triple whose worst-case restraint
    # envelope passes the same static design used after build.
    def search(start: int, chosen: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        if len(chosen) == n:
            return list(chosen) if validator(chosen) else None
        remaining = n - len(chosen)
        if len(ordered) - start < remaining:
            return None
        for index in range(start, len(ordered)):
            cand = ordered[index]
            if not compatible(cand, chosen):
                continue
            found = search(index + 1, chosen + [cand])
            if found is not None:
                return found
        return None

    selected = search(0, [])
    if selected is None:
        raise RuntimeError(
            f"在这一侧找不到 {n} 个满足 minimum-image 距离 ≥ {min_mutual_nm:.3f} nm"
            "且通过 v11 静态 bulk/member safety 包络的候选组合。"
        )
    return selected


# ============================================================================
# 子命令：equilibrate-base（纯 slab，无探针配体；GPU，用户提交）
# ============================================================================


def _build_base_system(
    top_file: str, gro_file: str, include_dir: Optional[str], add_barostat: bool = True,
    temperature_kelvin: float = 303.15,
):
    """`add_barostat=False`：建 System 但**不**加 barostat（NVT 松弛阶段用）——
    调用方之后可以对同一个 `system` 对象调用 `core.ensure_barostat_for_protocol`
    把 barostat 加进去，再 `Context.reinitialize(preserveState=True)` 切到 NPT，
    见 `cmd_equilibrate_base` 里的分阶段调用（v3→v4 §6）。
    """
    gmx_top = core.load_gromacs_topology_for_openmm(top_file, includeDir=include_dir)
    gro = app.GromacsGroFile(gro_file)
    box_vectors = gro.getPeriodicBoxVectors()
    gmx_top.topology.setPeriodicBoxVectors(box_vectors)

    system = gmx_top.createSystem(
        nonbondedMethod=app.PME, nonbondedCutoff=NONBONDED_CUTOFF_NM * unit.nanometer,
        constraints=app.HBonds, rigidWater=True, ewaldErrorTolerance=EWALD_ERROR_TOLERANCE,
    )
    dispersion = _resolve_and_verify_dispersion_protocol(top_file, include_dir)
    nb_force = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    nb_force.setUseDispersionCorrection(True)
    # v3 修法：narrow potential-switch，见 C2_LJ_SWITCH_DISTANCE_NM 上方注释。
    nb_force.setUseSwitchingFunction(True)
    nb_force.setSwitchingDistance(C2_LJ_SWITCH_DISTANCE_NM * unit.nanometer)

    membrane_protocol = core.resolve_membrane_protocol(
        core.ENVIRONMENT_TYPE_MEMBRANE, membrane_config=None, topology=gmx_top.topology
    )
    if add_barostat:
        core.ensure_barostat_for_protocol(
            system, membrane_protocol, temperature=temperature_kelvin, pressure=1.0
        )
    return system, gmx_top.topology, gro.positions, box_vectors, dispersion, membrane_protocol


def _add_barostat_and_activate(
    system, membrane_protocol: Dict[str, Any], temperature_kelvin: float,
    pressure_bar: float, simulation,
) -> Dict[str, Any]:
    """把 barostat 加进 `system`（`ensure_barostat_for_protocol`，Python 端
    `system.addForce(...)`），并立刻 `simulation.context.reinitialize(preserveState=True)`
    让**已经建好**的 Context 真正用上它。

    v4→v5 修的就是曾经有一条调用路径只做了前一半：`simulation.context` 早在
    这之前就用不带 barostat 的 `system` 建好了，只对 `system` 对象
    `addForce` 不会让已经建好的 Context 知道有新 Force——不 `reinitialize`
    的话新加的 barostat 完全是摆设，实测复现过"整段 NPT 其实一步体积移动都
    没发生过"（见模块 docstring v4→v5 一条）。拆成这一个函数、内部把两步
    绑死在一起，是为了让"加了 barostat 却忘记 reinitialize"这种写法在结构上
    不可能再发生——调用方不可能只调用其中一半。
    """
    action = core.ensure_barostat_for_protocol(
        system, membrane_protocol, temperature=temperature_kelvin, pressure=pressure_bar
    )
    simulation.context.reinitialize(preserveState=True)
    return action


def _run_equilibration_segment(
    simulation, dcd, csv_fh, n_steps_segment: int, report_interval_steps: int,
    timestep_ps: float, n_degrees_of_freedom: int, step_offset: int, phase_label: str,
) -> int:
    """跑 `n_steps_segment` 步、每 `report_interval_steps` 写一次 DCD/CSV 报告点。

    `step_offset` 是这一段开始前已经跑过的累计步数——NVT→NPT 两段共用同一份
    `step`/`time_ps` 计数，不会在切阶段时从 0 重开；`phase_label`
    （`"nvt"`/`"npt"`）写进 CSV 的 `phase` 列，供事后按阶段切分诊断
    （v3→v4 §6）。返回这一段结束时的累计总步数，供下一段续用。
    """
    if n_steps_segment % report_interval_steps != 0:
        raise SystemExit(
            f"{phase_label} 阶段步数 {n_steps_segment} 必须能被 "
            f"--report-interval-steps={report_interval_steps} 整除"
        )
    n_reports = n_steps_segment // report_interval_steps
    for i in range(n_reports):
        simulation.step(report_interval_steps)
        state = simulation.context.getState(getPositions=True, getEnergy=True)
        pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        if not np.isfinite(pe):
            raise RuntimeError(f"{phase_label} 阶段第 {i} 个报告点势能非有限——立即停止")
        box_now = state.getPeriodicBoxVectors()
        box_now_nm = np.asarray([v.value_in_unit(unit.nanometer) for v in box_now])
        vol = _box_volume_nm3(box_now_nm)
        step_now = step_offset + (i + 1) * report_interval_steps
        time_ps = step_now * timestep_ps
        temp_k = (
            2.0 * state.getKineticEnergy() / (n_degrees_of_freedom * unit.MOLAR_GAS_CONSTANT_R)
        ).value_in_unit(unit.kelvin)
        csv_fh.write(f"{step_now},{time_ps:.3f},{pe:.4f},{vol:.4f},{temp_k:.2f},{phase_label}\n")
        csv_fh.flush()
        dcd.writeModel(state.getPositions(), periodicBoxVectors=box_now)
    return step_offset + n_steps_segment


def cmd_equilibrate_base(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    top_file, gro_file = args.top, args.gro

    # v3→v4 §6：`--n-steps-nvt=0`（默认）与 v3 之前行为逐位一致——barostat
    # 立即加入，从第一步就是 NPT。>0 时先在固定盒下跑这么多步，让新增/全部
    # 水层扩散松弛，再加 barostat 切到 NPT 跑剩余步数（诊断 thick base 快速
    # 塌缩用，见模块 docstring）。
    n_steps_nvt = int(args.n_steps_nvt)
    n_steps_total = int(args.n_steps)
    if n_steps_nvt < 0:
        raise SystemExit("--n-steps-nvt 不能为负数")
    if n_steps_nvt >= n_steps_total:
        raise SystemExit(
            "--n-steps-nvt 必须小于 --n-steps（NVT 只是切到 NPT 之前的固定盒松弛阶段，"
            "不能占满整段预平衡）"
        )
    n_steps_npt = n_steps_total - n_steps_nvt

    print(
        f"⚗️  建纯 slab System（无探针配体，barostat 暂缓到"
        f"{'NVT 阶段结束后' if n_steps_nvt > 0 else '立即'}加入）：top={top_file}, gro={gro_file}"
    )
    system, topology, positions, box_vectors, dispersion, membrane_protocol = _build_base_system(
        top_file, gro_file, args.gmx_include_dir, add_barostat=False,
    )
    print(
        f"  ✅ System：{system.getNumParticles()} 原子；"
        f"dispersion_protocol={dispersion['dispersion_protocol']}；"
        f"barostat={membrane_protocol['barostat_class']}"
    )

    platform, properties, platform_name, fallback_reason = _get_platform(
        args.platform, args.allow_cpu_fallback, args.precision
    )
    integrator = openmm.LangevinMiddleIntegrator(
        args.temperature_kelvin * unit.kelvin, args.friction_per_ps / unit.picosecond,
        args.timestep_ps * unit.picosecond,
    )
    integrator.setRandomNumberSeed(int(args.seed))
    simulation = app.Simulation(topology, system, integrator, platform, properties)
    simulation.context.setPositions(positions)
    simulation.context.setPeriodicBoxVectors(*box_vectors)

    print(f"⚙️  最小化（maxIterations={args.n_steps_minimize}）...")
    simulation.minimizeEnergy(maxIterations=int(args.n_steps_minimize))
    state0 = simulation.context.getState(getEnergy=True)
    pe0 = state0.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    if not np.isfinite(pe0):
        raise RuntimeError(f"最小化后势能非有限: {pe0}")
    print(f"   最小化后 PE = {pe0:.3f} kJ/mol")

    simulation.context.setVelocitiesToTemperature(args.temperature_kelvin * unit.kelvin)

    dcd_path = os.path.join(output_dir, "equilibration.dcd")
    csv_path = os.path.join(output_dir, "equilibration_monitor.csv")
    topology.setPeriodicBoxVectors(box_vectors)

    # 温度换算必须扣掉约束自由度——`rigidWater=True` + `constraints=HBonds` 移除的
    # 自由度对这个体系（上万个刚性 TIP3P + 全部 X-H 键）不是可以忽略的量，用
    # `3*N` 当分母会系统性把报出来的温度读低，容易被误读成"没热起来/温控出问题"。
    # 加/不加 barostat 都不改变约束数，这里什么时候算都一样。
    n_degrees_of_freedom = 3 * system.getNumParticles() - system.getNumConstraints()
    barostat_added_after_step: Optional[int] = None
    with open(dcd_path, "wb") as dcd_fh, open(csv_path, "w") as csv_fh:
        dcd = app.DCDFile(
            dcd_fh, topology, dt=args.timestep_ps * unit.picosecond,
            interval=int(args.report_interval_steps),
        )
        csv_fh.write("step,time_ps,potential_kJ_mol,volume_nm3,temperature_K,phase\n")

        step_count = 0
        if n_steps_nvt > 0:
            print(f"🧊 NVT 阶段（固定盒，{n_steps_nvt} 步，barostat 尚未加入）：让水层扩散松弛...")
            step_count = _run_equilibration_segment(
                simulation, dcd, csv_fh, n_steps_nvt, int(args.report_interval_steps),
                args.timestep_ps, n_degrees_of_freedom, step_count, "nvt",
            )
            print(
                f"🌡️  NVT 阶段结束（累计 {step_count} 步），加入 "
                f"{membrane_protocol['barostat_class']} 并 Context.reinitialize(preserveState=True)..."
            )
            _add_barostat_and_activate(
                system, membrane_protocol, args.temperature_kelvin, 1.0, simulation
            )
            barostat_added_after_step = step_count
        else:
            print(
                f"（--n-steps-nvt=0，跳过固定盒 NVT 阶段，立即加入 "
                f"{membrane_protocol['barostat_class']}——与 v3 之前行为一致）"
            )
            _add_barostat_and_activate(
                system, membrane_protocol, args.temperature_kelvin, 1.0, simulation
            )
            barostat_added_after_step = 0

        print(f"💧 NPT 阶段（{n_steps_npt} 步）...")
        step_count = _run_equilibration_segment(
            simulation, dcd, csv_fh, n_steps_npt, int(args.report_interval_steps),
            args.timestep_ps, n_degrees_of_freedom, step_count, "npt",
        )

    final_state = simulation.context.getState(getPositions=True, getEnergy=True)
    equilibrated_gro_path = os.path.join(output_dir, "equilibrated.gro")
    _write_equilibrated_gro(
        gro_file, final_state.getPositions(), final_state.getPeriodicBoxVectors(), equilibrated_gro_path
    )

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "top_file": os.path.relpath(top_file, _REPO_ROOT),
        "gro_file": os.path.relpath(gro_file, _REPO_ROOT),
        "input_sha256": _sha256_tree([top_file, gro_file]),
        "water_thickness_label": args.water_thickness_label,
        "n_steps": n_steps_total,
        "n_steps_nvt": n_steps_nvt,
        "n_steps_npt": n_steps_npt,
        "barostat_added_after_step": barostat_added_after_step,
        "timestep_ps": args.timestep_ps,
        "report_interval_steps": int(args.report_interval_steps),
        "frame_interval_ps": float(args.report_interval_steps) * args.timestep_ps,
        "temperature_kelvin": args.temperature_kelvin,
        "platform_requested": args.platform,
        "platform_used": platform_name,
        "platform_fallback_reason": fallback_reason,
        "seed": int(args.seed),
        "nonbonded_cutoff_nm": NONBONDED_CUTOFF_NM,
        "dispersion_protocol": dispersion["dispersion_protocol"],
        "forcefield_family": dispersion["forcefield_family_report"]["family"],
        "equilibration_dcd": os.path.relpath(dcd_path, _REPO_ROOT),
        "equilibrated_gro": os.path.relpath(equilibrated_gro_path, _REPO_ROOT),
        "equilibrated_gro_sha256": _sha256_file(equilibrated_gro_path),
        "final_potential_kJ_mol": float(
            final_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        ),
    }
    with open(os.path.join(output_dir, "equilibrate_base_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"✅ equilibrate-base 完成：{equilibrated_gro_path}")
    print(
        "   ⚠️ 这只是跑完了指定步数；C2 要求的『预先平衡』还需要对 equilibration.dcd "
        "跑 base-quality-gate 做末段漂移判定，不能只看跑完没跑完。"
    )


# ============================================================================
# 子命令：base-quality-gate（纯 slab，无探针配体；纯 CPU，读 DCD）
# ============================================================================


def _p31_indices(topology) -> np.ndarray:
    return np.asarray(_lipid_phosphorus_indices(topology), dtype=int)


def _leaflet_split_by_frame0(p31_z_by_frame: np.ndarray) -> Tuple[np.ndarray, float]:
    """按第 0 帧的 P31 z 相对当帧均值分上下叶，叶片身份此后固定不逐帧重选。"""
    midplane0 = float(p31_z_by_frame[0].mean())
    is_upper = p31_z_by_frame[0] > midplane0
    return is_upper, midplane0


def _tail_drift_per_ns(
    times_ns: np.ndarray, series: np.ndarray, tail_fraction: float
) -> Tuple[float, float, np.ndarray]:
    """末段线性拟合斜率 + `np.polyfit(..., cov=True)` 给的 OLS 斜率标准误。

    ⚠️ **这个标准误对本体系不可信，只留作参考点估计，不再用来判显著性**
    （2026-08-07 实测钉死）：OLS 标准误假设残差独立，而 APL 这类量在几 ns 的
    时间尺度上强自相关（是一个振荡而不是白噪声）——同一条 14 ns 轨迹，
    换三个末段窗口分别给出"平"、"显著正漂"、"显著负漂"三个互相矛盾的结论，
    而逐 1 ns 分块看原始曲线，真相是 APL 在 0.606~0.633 nm² 之间振荡、没有
    持续单调趋势。真正判"是否还在漂"用 `_block_mean_drift_significance`
    （分块法，块间方差而不是块内回归残差估计不确定度）。
    """
    n = len(times_ns)
    tail_start = max(0, int(round(n * (1.0 - tail_fraction))))
    tail_times = times_ns[tail_start:]
    tail_series = series[tail_start:]
    if len(tail_times) < 3 or tail_times[-1] <= tail_times[0]:
        return 0.0, 0.0, tail_series
    (slope, _intercept), cov = np.polyfit(tail_times, tail_series, 1, cov=True)
    slope_stderr = float(np.sqrt(max(float(cov[0, 0]), 0.0)))
    return float(slope), slope_stderr, tail_series


# 分块法用的块宽（ns）。选它的依据是"块内先把逐帧高频噪声粗略平均掉"，不是
# 拟合出来的自相关时间——这条轨迹只有 14 ns、不到 2 个可见振荡周期，还不够
# 拟合自相关时间；1 ns 只是一个保守的起点，换体系/更长轨迹应该重新核对。
DRIFT_BLOCK_WIDTH_NS = 1.0
DRIFT_MIN_BLOCKS_PER_HALF = 2


def _block_mean_drift_significance(
    times_ns: np.ndarray, series: np.ndarray, tail_fraction: float,
    block_width_ns: float = DRIFT_BLOCK_WIDTH_NS,
) -> Dict[str, Any]:
    """Flyvbjerg–Petersen 风格分块法：先分块求均值压掉高频噪声，再用**块间**方差
    （不是块内线性回归的残差）比较前半段与后半段块均值——对自相关（振荡）数据
    比 OLS 回归标准误诚实。块数 < `2 * DRIFT_MIN_BLOCKS_PER_HALF` 时不敢声称
    显著，如实报告 `insufficient_data`（不是把"数据不够"悄悄归到"不显著"里）。
    """
    n = len(times_ns)
    tail_start = max(0, int(round(n * (1.0 - tail_fraction))))
    tail_times = times_ns[tail_start:]
    tail_series = np.asarray(series[tail_start:], dtype=np.float64)
    if tail_times.size < 2:
        return {
            "significance": "insufficient_data", "n_blocks": 0,
            "first_half_mean": None, "second_half_mean": None,
            "difference": 0.0, "combined_stderr": 0.0, "block_means": [],
        }
    dt_ns = float(np.median(np.diff(tail_times))) if tail_times.size > 1 else 1.0
    block_size = max(1, int(round(block_width_ns / dt_ns))) if dt_ns > 0 else tail_series.size
    n_blocks = tail_series.size // block_size
    if n_blocks < 2 * DRIFT_MIN_BLOCKS_PER_HALF:
        return {
            "significance": "insufficient_data", "n_blocks": int(n_blocks),
            "first_half_mean": None, "second_half_mean": None,
            "difference": 0.0, "combined_stderr": 0.0, "block_means": [],
        }
    block_means = np.array(
        [tail_series[i * block_size:(i + 1) * block_size].mean() for i in range(n_blocks)]
    )
    half = n_blocks // 2
    first_half, second_half = block_means[:half], block_means[half:]
    first_mean, second_mean = float(first_half.mean()), float(second_half.mean())
    stderr_first = float(first_half.std(ddof=1) / np.sqrt(first_half.size)) if first_half.size > 1 else 0.0
    stderr_second = float(second_half.std(ddof=1) / np.sqrt(second_half.size)) if second_half.size > 1 else 0.0
    combined_stderr = float(np.sqrt(stderr_first**2 + stderr_second**2))
    difference = second_mean - first_mean
    significance = _classify_drift_significance(difference, combined_stderr)
    return {
        "significance": significance, "n_blocks": int(n_blocks),
        "first_half_mean": first_mean, "second_half_mean": second_mean,
        "difference": difference, "combined_stderr": combined_stderr,
        "block_means": block_means.tolist(),
    }


def _classify_drift_significance(value: float, stderr: float) -> str:
    """`value`/`stderr` 同单位。z 分数用 `DRIFT_SIGNIFICANCE_Z` 门槛。"""
    if stderr <= 0.0:
        return "not_significant"  # 拟合窗口太短/退化，不敢声称显著
    z = value / stderr
    if abs(z) < DRIFT_SIGNIFICANCE_Z:
        return "not_significant"
    return "significantly_positive" if value > 0 else "significantly_negative"


def _apl_drift_recommendation(
    checks: Dict[str, bool], apl_drift_significance: str, apl_moving_toward_target: Optional[bool],
) -> str:
    """替换掉原先"未通过就无条件建议续跑 5–10 ns"的写法——那条建议不看漂移方向，
    真在收缩的时候续跑只是白烧 GPU（说明协议本身不够，不是时间不够）。

    v3 修复（2026-08-07）：原先签名只传一个全局 `passed: bool`，未通过时**无论
    具体是哪个 check 失败**都统一走"看漂移方向"这条逻辑分支——thick base 那次
    实测踩到：`apl_drift_within_gate=True`（分块法判定不显著）但
    `apl_within_3_percent_of_literature=False`（APL 稳定在 0.585，偏离文献值
    8.4%），落进了写给"漂移显著但方向不对"那个分支的兜底 `return`，打印出"APL
    分块前后半有显著差异"这种和实际 `not_significant` 矛盾的话。现在先看
    **哪些 check 真的失败了**，再决定给哪条建议，不再从漂移显著性反推全局结论。

    `apl_drift_significance` 来自 `_block_mean_drift_significance`（分块法，
    见其 docstring）；`insufficient_data` 是独立的第三态——"块数不够、不敢判"，
    不能悄悄并进 `not_significant`（那会把"没法判"读成"判了说没事"）。
    """
    if all(checks.values()):
        return "gate 通过，不需要续跑。"

    failed = [name for name, ok in checks.items() if not ok]
    drift_failed = "apl_drift_within_gate" in failed
    literature_failed = "apl_within_3_percent_of_literature" in failed

    if drift_failed:
        if apl_drift_significance == "insufficient_data":
            return (
                f"⚠️ 数据不够判 APL 漂移：末段窗口分不出足够的独立块（每块 "
                f"{DRIFT_BLOCK_WIDTH_NS} ns，两侧各至少 {DRIFT_MIN_BLOCKS_PER_HALF} 块），"
                "不足以判断这是振荡还是真漂移。先延长轨迹（或增大 --tail-fraction）"
                f"拿到足够块数再判，不要在数据不够时就下结论。（其余失败项：{failed}）"
            )
        if apl_drift_significance == "significantly_negative":
            return (
                "❌ APL 末段窗口前半→后半显著下降（分块法，非 OLS 单一线性拟合）。"
                "先确认这不是振荡的下降半周期（看 timeseries.csv 的逐 ns 曲线形状，"
                "尤其是不是恰好只覆盖了一个振荡周期的降段）；如果延长/换更大分块窗口后"
                "仍然是持续下降而不是振荡，才说明当前 "
                f"{C2_LJ_SWITCH_DISTANCE_NM}→1.000 nm 的窄 potential-switch 不够，"
                "下一步测更宽的 0.95→1.00 nm switch pilot。不要看到一次"
                f"'significantly_negative'就直接切换 switch 宽度。（其余失败项：{failed}）"
            )
        if apl_drift_significance == "significantly_positive":
            # `apl_moving_toward_target` 是 `Optional[bool]`：`None` 表示没传
            # `--literature-apl-nm2`、根本判不了方向，**不等于**"方向不对"
            # （`is False`）——`not None` 和 `not False` 都是 `True`，两者不能
            # 共用同一个 `not apl_moving_toward_target` 分支。
            if apl_moving_toward_target is None:
                return (
                    "⚠️ APL 分块前后半显著上升，但没有传 --literature-apl-nm2，"
                    f"判不了这是不是朝目标靠近。（其余失败项：{failed}）"
                )
            if apl_moving_toward_target:
                return (
                    "⚠️ 可以续跑：APL 分块前后半显著上升且正在朝文献值靠近。"
                    "equilibrate-base 不支持 resume，需另开一次跑（--gro 指向这份 "
                    f"equilibrated.gro）续接一段，跑完拼接 DCD 后重判。（其余失败项：{failed}）"
                )
            return (
                "⚠️ APL 分块前后半有显著差异，但方向不是朝文献值靠近——先核对 "
                f"--literature-apl-nm2 是否选对了力场/温度条件，不要机械续跑。（其余失败项：{failed}）"
            )

    if literature_failed and not drift_failed:
        return (
            "❌ APL 已经稳定（分块法未见显著漂移），但稳定在的值偏离文献目标超过 3%——"
            "这不是『还没平衡够』，续跑大概率解决不了。需要单独排查：是这个协议/水层"
            "厚度下本来就稳不到目标值，还是 --literature-apl-nm2 本身不适用于当前"
            f"条件。（其余失败项：{failed}）"
        )

    return f"❌ 未通过，失败项：{failed}（不属于以上已知分支，需要逐项看 checks 人工排查）。"


def cmd_base_quality_gate(args: argparse.Namespace) -> None:
    import mdtraj as md

    gmx_top = core.load_gromacs_topology_for_openmm(args.top, includeDir=args.gmx_include_dir)
    topology = gmx_top.topology
    p31 = _p31_indices(topology)
    n_lipids_total = p31.size

    dcd_paths = list(args.dcd)  # nargs="+"：支持"续跑后拼接两段 DCD 重判"
    md_top = md.Topology.from_openmm(topology)
    segments = [md.load_dcd(path, top=md_top) for path in dcd_paths]
    for path, seg in zip(dcd_paths, segments):
        if seg.unitcell_lengths is None:
            raise SystemExit(f"{path} 没有 unitcell 信息，无法算 APL/膜厚")

    # 每一段独立的 equilibrate-base 跑都是从上一段的 equilibrated.gro 重新出发、
    # 重新用 Maxwell-Boltzmann 初始化速度（`equilibrate-base` 不支持真正的
    # resume）——这在续跑开头会留一段重新弛豫的瞬态。2026-08-07 实测踩过一次：
    # 拼接后只用短末段窗口判，会把这段瞬态误判成"仍在显著收缩"，而排除瞬态后
    # 同一段轨迹其实是平的（不显著）。所以除第一段外，每一段的开头都要丢弃
    # `--restart-discard-ps`，再拼接——这不是为了让门变绿而选的窗口，是每次
    # 续跑都会重新出现的同一个方法学问题，理应普遍处理。
    frame_interval_ps = float(args.frame_interval_ps)
    discard_frames = int(round(float(args.restart_discard_ps) / frame_interval_ps))
    trimmed_segments = []
    discarded_report = []
    for i, (path, seg) in enumerate(zip(dcd_paths, segments)):
        if i == 0 or discard_frames <= 0:
            trimmed_segments.append(seg)
            discarded_report.append({"dcd": path, "discarded_frames": 0})
            continue
        if discard_frames >= seg.n_frames:
            raise SystemExit(
                f"{path} 只有 {seg.n_frames} 帧，不够丢弃 --restart-discard-ps="
                f"{args.restart_discard_ps} ps（={discard_frames} 帧）——这一段太短，"
                "先跑久一点或调小 --restart-discard-ps。"
            )
        trimmed_segments.append(seg[discard_frames:])
        discarded_report.append({"dcd": path, "discarded_frames": discard_frames})

    # 两段独立的跑首尾相接只是为了让"末段窗口"覆盖到拼接后的总时长，不代表这是
    # 一条严格连续的动力学轨迹；`checks`/`recommendation` 判的是"APL 稳没稳"，
    # 这个近似（配合上面丢弃重启瞬态）对这个目的是合理的。
    traj = md.join(trimmed_segments, check_topology=False) if len(trimmed_segments) > 1 else trimmed_segments[0]
    times_ns = np.arange(traj.n_frames, dtype=float) * frame_interval_ps / 1000.0

    p31_z = traj.xyz[:, p31, 2]
    is_upper, _midplane0 = _leaflet_split_by_frame0(p31_z)
    n_upper, n_lower = int(is_upper.sum()), int((~is_upper).sum())

    upper_mean_z = p31_z[:, is_upper].mean(axis=1)
    lower_mean_z = p31_z[:, ~is_upper].mean(axis=1)
    thickness_nm = upper_mean_z - lower_mean_z

    lengths = np.asarray(traj.unitcell_lengths, dtype=float)
    box_xy_area_nm2 = lengths[:, 0] * lengths[:, 1]
    box_z_nm = lengths[:, 2]
    apl_nm2 = box_xy_area_nm2 / (n_lipids_total / 2.0)

    # 逐帧时间序列必须落盘，不能只留窗口拟合的汇总数字——2026-08-07 实测：两个
    # tail 窗口对同一条轨迹给出相反的显著性结论（ns1-5 平、ns2.2-5 显著降），
    # 这正是过阻尼/欠阻尼式过冲-回落这类非单调弛豫会产生的现象，线性拟合窗口选
    # 哪一段就信哪一段的结论并不可靠。这里把 APL/膜厚/box_z 逐帧原样写出来，
    # 判断"是过冲回落到平台"还是"仍在单调收缩"要看形状，不能只看某个窗口的斜率。
    timeseries_path = os.path.splitext(args.output)[0] + "_timeseries.csv"
    with open(timeseries_path, "w", newline="") as ts_fh:
        ts_fh.write("time_ns,apl_nm2,bilayer_thickness_nm,box_z_nm\n")
        for t, a, th, z in zip(times_ns, apl_nm2, thickness_nm, box_z_nm):
            ts_fh.write(f"{t:.4f},{a:.6f},{th:.6f},{z:.6f}\n")

    tail_fraction = float(args.tail_fraction)
    apl_norm = float(apl_nm2.mean()) if apl_nm2.mean() else 1.0
    # OLS 斜率仍然算出来落盘（人读起来直观），但**不再**用它的标准误判显著性——
    # 见 `_tail_drift_per_ns` 上方注释，同一条轨迹换窗口给出过矛盾结论。
    apl_slope_nm2_per_ns, _apl_slope_stderr_nm2_per_ns, apl_tail = _tail_drift_per_ns(
        times_ns, apl_nm2, tail_fraction
    )
    apl_drift_pct_per_ns = 100.0 * apl_slope_nm2_per_ns / apl_norm

    # 权威判据：分块法比较末段窗口前半 vs 后半的块均值（见
    # `_block_mean_drift_significance`），对振荡型（自相关）数据比 OLS 回归
    # 标准误诚实。
    apl_block = _block_mean_drift_significance(times_ns, apl_nm2, tail_fraction)
    apl_drift_significance = apl_block["significance"]

    thickness_slope_nm_per_ns, _thickness_slope_stderr_nm_per_ns, thickness_tail = _tail_drift_per_ns(
        times_ns, thickness_nm, tail_fraction
    )
    thickness_block = _block_mean_drift_significance(times_ns, thickness_nm, tail_fraction)
    thickness_drift_significance = thickness_block["significance"]

    # 密度剖面（末段窗口，1 埃分箱）：水氧、磷、脂尾碳（脂质里排除头基/甘油的所有
    # 碳原子——粗略但足够诊断"疏水核是否被水侵入"）。
    tail_start = max(0, int(round(traj.n_frames * (1.0 - tail_fraction))))
    tail_slice = traj[tail_start:]
    water_o = np.asarray(
        [a.index for a in topology.atoms()
         if str(a.residue.name).strip().upper() in core.WATER_MOLECULE_NAMES
         and str(a.name).strip().upper() == "O"],
        dtype=int,
    )
    lipid_tail_carbons = np.asarray(
        [a.index for a in topology.atoms()
         if str(a.residue.name).strip().upper() in ("PA", "OL")
         and str(a.element.symbol).strip().upper() == "C"],
        dtype=int,
    )
    bins = np.linspace(0.0, float(lengths[:, 2].mean()), 101)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_width_nm = float(bins[1] - bins[0])
    # v3→v4 §4：`density_profile` 之前存的是每个 bin 逐帧平均**原子计数**
    # （`counts / n_frames`），量纲上少除了一个 bin 体积，不能跟"约 33 nm⁻³
    # 体相水"这类文献数密度直接比较。这里除以 `bin_volume_nm3`（= 末段窗口
    # 的 XY 面积均值 × bin 厚度——跟 `_density_profile` 只看 `tail_slice`
    # 一致，不是用整条轨迹的 XY 面积）。
    tail_lengths = np.asarray(lengths[tail_start:], dtype=float)
    tail_xy_area_nm2_mean = float((tail_lengths[:, 0] * tail_lengths[:, 1]).mean())
    bin_volume_nm3 = tail_xy_area_nm2_mean * bin_width_nm

    def _density_profile(indices: np.ndarray) -> List[float]:
        z = tail_slice.xyz[:, indices, 2].reshape(-1) if indices.size else np.asarray([])
        return _z_number_density_profile(z, tail_slice.n_frames, bins, bin_volume_nm3)

    density_profile = {
        "units": "number density, nm^-3",
        "bin_centers_nm": bin_centers.tolist(),
        "bin_width_nm": bin_width_nm,
        "bin_volume_nm3": bin_volume_nm3,
        "water_oxygen": _density_profile(water_o),
        "phosphorus": _density_profile(p31),
        "lipid_tail_carbon": _density_profile(lipid_tail_carbons),
    }

    # `is not None`，不是真值判断——`--literature-apl-nm2 0` 字面意思是"用 0 当
    # 参考值"（虽然没有实际意义，但那是用户的输入，不该被 falsy 的 0 悄悄吞成 None）。
    literature_apl_nm2 = float(args.literature_apl_nm2) if args.literature_apl_nm2 is not None else None
    apl_tail_mean = float(apl_tail.mean())

    # 移动方向也改用分块法的前后半差值判（不是 OLS 点估计的符号）——理由同上，
    # 振荡数据里 OLS 斜率符号本身就不可信，用它判方向一样会被同一个问题污染。
    apl_moving_toward_target: Optional[bool] = None
    if literature_apl_nm2 is not None and apl_block["difference"] != 0.0:
        gap = literature_apl_nm2 - apl_tail_mean  # >0 当前偏低，<0 当前偏高
        apl_moving_toward_target = bool(np.sign(gap) == np.sign(apl_block["difference"]))

    # bool(...) 显式转换：见 2026-08-07 那次 numpy.bool_ 导致 json.dumps 炸掉的
    # traceback——`cls=core.NumpyEncoder` 能处理，但漏了 `cls=` 的调用会炸，
    # 干脆在源头就转成原生类型，不依赖每个消费点都记得传 `cls=`。
    checks = {
        "apl_within_3_percent_of_literature": bool(
            literature_apl_nm2 is None
            or abs(apl_tail_mean - literature_apl_nm2) / literature_apl_nm2 <= 0.03
        ),
        "apl_drift_within_gate": bool(apl_drift_significance == "not_significant"),
        "thickness_drift_within_gate": bool(thickness_drift_significance == "not_significant"),
        "leaflets_have_40_lipids_each": bool(n_upper == 40 and n_lower == 40),
        "no_nan_or_inf": bool(np.all(np.isfinite(apl_nm2)) and np.all(np.isfinite(thickness_nm))),
    }
    passed = all(checks.values())
    recommendation = _apl_drift_recommendation(checks, apl_drift_significance, apl_moving_toward_target)
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "top": os.path.relpath(args.top, _REPO_ROOT),
        "gro": os.path.relpath(args.gro, _REPO_ROOT),
        "dcd": dcd_paths,
        "restart_discard_ps": float(args.restart_discard_ps),
        "restart_discard_report": discarded_report,
        "timeseries_csv": timeseries_path,
        "frame_interval_ps": frame_interval_ps,
        "tail_fraction": tail_fraction,
        "drift_block_width_ns": DRIFT_BLOCK_WIDTH_NS,
        "n_frames": traj.n_frames,
        "n_upper_lipids": n_upper,
        "n_lower_lipids": n_lower,
        "literature_apl_nm2": literature_apl_nm2,
        "literature_apl_source": LITERATURE_APL_SOURCE_NOTE if literature_apl_nm2 is not None else None,
        "apl_nm2_tail_mean": apl_tail_mean,
        "apl_deviation_percent": (
            100.0 * abs(apl_tail_mean - literature_apl_nm2) / literature_apl_nm2
            if literature_apl_nm2 else None
        ),
        "apl_tail_ols_slope_percent_per_ns": apl_drift_pct_per_ns,  # 点估计，仅供参考，不判显著性
        "apl_tail_drift_significance": apl_drift_significance,
        "apl_tail_block_first_half_mean_nm2": apl_block["first_half_mean"],
        "apl_tail_block_second_half_mean_nm2": apl_block["second_half_mean"],
        "apl_tail_block_difference_nm2": apl_block["difference"],
        "apl_tail_block_combined_stderr_nm2": apl_block["combined_stderr"],
        "apl_tail_block_n_blocks": apl_block["n_blocks"],
        "apl_moving_toward_literature_target": apl_moving_toward_target,
        "bilayer_thickness_nm_tail_mean": float(thickness_tail.mean()),
        "bilayer_thickness_tail_ols_slope_nm_per_ns": thickness_slope_nm_per_ns,  # 仅供参考
        "bilayer_thickness_tail_drift_significance": thickness_drift_significance,
        "bilayer_thickness_tail_block_first_half_mean_nm": thickness_block["first_half_mean"],
        "bilayer_thickness_tail_block_second_half_mean_nm": thickness_block["second_half_mean"],
        "bilayer_thickness_tail_block_difference_nm": thickness_block["difference"],
        "bilayer_thickness_tail_block_combined_stderr_nm": thickness_block["combined_stderr"],
        "box_xy_area_nm2_mean": float(box_xy_area_nm2.mean()),
        "box_z_nm_mean": float(box_z_nm.mean()),
        "density_profile_along_normal": density_profile,
        "recommendation": recommendation,
        "checks": checks,
        "passed": passed,
    }
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, cls=core.NumpyEncoder)
    print(f"{'✅' if passed else '❌'} base-quality-gate 完成: {args.output}")
    print(f"   逐帧 APL/膜厚/box_z 时间序列: {timeseries_path}")
    print(json.dumps({
        "checks": checks, "apl_nm2_tail_mean": apl_tail_mean,
        "apl_tail_ols_slope_percent_per_ns": apl_drift_pct_per_ns,
        "apl_tail_drift_significance": apl_drift_significance,
        "apl_tail_block_first_half_mean_nm2": apl_block["first_half_mean"],
        "apl_tail_block_second_half_mean_nm2": apl_block["second_half_mean"],
        "apl_tail_block_n_blocks": apl_block["n_blocks"],
        "apl_moving_toward_literature_target": apl_moving_toward_target,
        "n_upper": n_upper, "n_lower": n_lower,
    }, indent=2, cls=core.NumpyEncoder))
    if not passed:
        print(recommendation)


# ============================================================================
# 子命令：extend-water（纯 CPU：对称加厚，两侧各加约一半）
# ============================================================================


# ---------------------------------------------------------------------------
# 自建水层填充（不走 `Modeller.addSolvent`/`ForceField`）
#
# 第一版用 `Modeller(gmx_top.topology, ...).addSolvent(ForceField(...))`：实测
# 直接炸——`addSolvent` 内部要用给定的 `ForceField` XML（amber14-all.xml）对
# **整个**合并后拓扑的每个残基做模板匹配，而 (a) GROMACS `[settles]` 水在
# `Topology` 里没有键（只有约束，MEM-15 同一类问题）、(b) 更根本的是 Lipid21 的
# `PA`/`PC`/`OL` 残基本来就不在 amber14-all.xml 里——那些参数只存在于
# `toppar/POPC.itp`，任何 `ForceField` 都不可能认得。这不是漏传水的键能修好的，
# 是"这份 GROMACS-native 拓扑根本没法喂给 ForceField"这个更根本的架构问题
# （与本文件模块 docstring「与 C1 的架构差异」一节是同一件事，只是这次踩在
# `addSolvent` 而不是插 co-ion 上）。
#
# 所以这里改成完全不经过 ForceField：新增的两段空隙里自己按标准体相水数密度
# 铺一层立方格点水（O 落格点，H1/H2 按 TIP3P 规范键长/键角+随机取向摆放）。
# 这不追求摆出物理上已平衡的构型——那是后面 `equilibrate-base` 的工作；
# 这里只需要"合理密度、无恶性重叠"的起点。
# ---------------------------------------------------------------------------

TIP3P_OH_BOND_NM = 0.09572
TIP3P_HOH_ANGLE_DEG = 104.52
# ~1 g/mL 液态水的数密度，等价于 18.015 g/mol、0.997 g/cm³——GROMACS
# genbox/solvate、OpenMM addSolvent 等几乎所有建水盒工具默认都是这个量级，
# 不是本文件独有的假设。
BULK_WATER_NUMBER_DENSITY_PER_NM3 = 33.33
# v3→v4 前：新水层与（平移后）已有体系边界之间留的缓冲曾经是一个固定
# 0.25 nm 常量——避免新格点紧贴在原有原子上产生恶性重叠。但固定值在
# `--extra-water-nm` 较小（两侧各约 1 nm）时会吃掉可用深度的近一半，导致实际
# 铺出的数密度远低于 `BULK_WATER_NUMBER_DENSITY_PER_NM3`（见模块 docstring
# §5）。v4 起 `_pack_water_slab` 改成用 `0.5 * 目标格点间距` 当缓冲（与密度
# 挂钩，量纲一致），这个固定常量已删除，不留死代码。


def _tip3p_local_geometry_nm() -> np.ndarray:
    """规范 TIP3P 内部几何，O 在原点：返回 (3,3) 的 [O, H1, H2] 坐标。"""
    half_angle = math.radians(TIP3P_HOH_ANGLE_DEG) / 2.0
    h1 = np.array([TIP3P_OH_BOND_NM * math.sin(half_angle), 0.0, TIP3P_OH_BOND_NM * math.cos(half_angle)])
    h2 = np.array([-TIP3P_OH_BOND_NM * math.sin(half_angle), 0.0, TIP3P_OH_BOND_NM * math.cos(half_angle)])
    return np.array([[0.0, 0.0, 0.0], h1, h2])


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """从随机单位四元数得到的旋转矩阵，在 SO(3) 上均匀分布——只是为了不让新水
    分子摆成一个完美晶格取向，不追求别的物理意义。"""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _pack_water_slab(
    lx_nm: float, ly_nm: float, z_min_nm: float, z_max_nm: float,
    rng: np.random.Generator, density_per_nm3: float = BULK_WATER_NUMBER_DENSITY_PER_NM3,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """在 `[0,lx] x [0,ly] x [z_min,z_max]` 这段新增空隙里铺一层立方格点水，
    尽量贴近 `density_per_nm3` 这个目标数密度。

    v3→v4 修法，两轮：

      第一轮（原先固定 0.25 nm 缓冲 + `np.arange` 定步长步进截断，实测在
      `--extra-water-nm` 较小时把实际密度压到目标值六成左右，见模块
      docstring §5）：边界缓冲区改成与目标格点间距成比例（`0.5 * spacing`），
      每个方向的格点数按 `round(长度 / 目标间距)` 取整、用等距网格精确铺满
      （不再用 `np.arange` 步进截断丢弃末尾不足一格的部分）。

      第二轮（2026-08-09 review 抓到第一轮仍不够）：第一轮的格点数是按
      **扣掉 buffer 后的 `usable_depth`** 算的——但 buffer 区域本身也是新增
      水层的真实物理体积，不是"不存在"，不能因为暂时不在那里摆格点就把它从
      密度的分母里删掉。真实 C2 盒子上实测：`n_z` 按 `usable_depth` 算只给
      出 2 层，`n_x*n_y*n_z=1024`，看起来相对 `usable_volume` 的密度有
      30.7 nm⁻³（"接近目标"），但摊到**整段**新增体积（含 buffer，约
      49 nm³）上只有约 20.9 nm⁻³——NPT 阶段仍会被追认这个真实的欠水状态，
      被迫把盒子压缩到这 1024 个水真正能撑住的体积。现在 `n_x`/`n_y`/`n_z`
      改成按**完整** `depth`（不是 `usable_depth`）算目标格点数，`buffer_nm`
      只决定这些格点摆在 `[z_lo, z_hi]` 这个子区间的什么位置——不再影响摆
      多少个格点。后续 NVT/NPT 弛豫拿到的因此是一段数密度真正接近目标的
      水层，不是一段"usable 子体积达标、整体仍欠水"的水层。

    返回 `(molecules, diagnostics)`：`molecules` 是 `(n_water, 3, 3)` 的
    `[O, H1, H2]` 绝对坐标（nm）；`diagnostics` 同时记录扣 buffer 前
    （`full_added_volume_nm3`/`target_water_count_full_volume`/
    `achieved_density_full_volume_nm3`）和扣 buffer 后
    （`usable_volume_nm3`）两套数字，供调用方如实写进 manifest——不能只看
    其中一套就断言"密度达标"。
    """
    depth = z_max_nm - z_min_nm
    target_spacing_nm = float(density_per_nm3) ** (-1.0 / 3.0)
    buffer_nm = 0.5 * target_spacing_nm
    if depth <= 2 * buffer_nm:
        raise ValueError(
            f"新增水层厚度 {depth:.3f} nm 太薄，摆不下一层水+边界缓冲"
            f"（缓冲区={buffer_nm:.4f} nm，与目标密度的格点间距成比例）"
        )
    usable_depth = depth - 2 * buffer_nm
    z_lo = z_min_nm + buffer_nm

    n_x = max(1, int(round(lx_nm / target_spacing_nm)))
    n_y = max(1, int(round(ly_nm / target_spacing_nm)))
    # 用**完整** depth 算目标层数——buffer 只是给这些层的摆放范围让出边界间隙，
    # 不能反过来先扣掉 buffer 再算"这段变薄的体积该摆几层"，那样会把 buffer
    # 那部分体积对应的水凭空丢掉（这正是第一轮 v4 修复遗漏的地方）。
    n_z = max(1, int(round(depth / target_spacing_nm)))

    # 用 `n` 个等距格点精确切满整个可用长度（每份中心落一个格点）——不是
    # `np.arange(half_spacing, L, spacing)` 那种从头按固定步长走、走到尽头前
    # 不足一格就丢弃剩余长度的做法。x/y 方向铺满整个 lx/ly（周期性方向本来
    # 就没有 buffer 的必要）；z 方向的 `n_z` 个格点摊在 `[z_lo, z_hi]` 这段
    # 缩进了 buffer 的子区间里——层数由完整 depth 决定，只是摆放位置让出了
    # 边界。
    xs = (np.arange(n_x) + 0.5) * (lx_nm / n_x)
    ys = (np.arange(n_y) + 0.5) * (ly_nm / n_y)
    zs = z_lo + (np.arange(n_z) + 0.5) * (usable_depth / n_z)

    local = _tip3p_local_geometry_nm()
    molecules = []
    for x in xs:
        for y in ys:
            for z in zs:
                rot = _random_rotation_matrix(rng)
                molecules.append(np.array([x, y, z]) + local @ rot.T)

    n_water = n_x * n_y * n_z
    full_added_volume_nm3 = lx_nm * ly_nm * depth
    usable_volume_nm3 = lx_nm * ly_nm * usable_depth
    target_water_count_full_volume = int(round(float(density_per_nm3) * full_added_volume_nm3))
    achieved_density_full_volume_nm3 = (
        float(n_water) / full_added_volume_nm3 if full_added_volume_nm3 > 0 else 0.0
    )
    diagnostics = {
        "n_x": n_x, "n_y": n_y, "n_z": n_z,
        "buffer_nm": buffer_nm, "usable_depth_nm": usable_depth,
        "usable_volume_nm3": usable_volume_nm3,
        "full_added_volume_nm3": full_added_volume_nm3,
        "target_number_density_per_nm3": float(density_per_nm3),
        "target_water_count_full_volume": target_water_count_full_volume,
        "actual_water_count": int(n_water),
        "achieved_density_full_volume_nm3": achieved_density_full_volume_nm3,
    }
    return np.asarray(molecules, dtype=np.float64), diagnostics


def cmd_extend_water(args: argparse.Namespace) -> None:
    """§C2「至少两种水层厚度（保持 XY 面积和每叶脂质数相同，只改 Z/水数）」。

    做法：把已有全部原子沿 Z 平移 `extra_water_nm/2`（让膜在新盒子里重新居中），
    盒子 Z 方向加大 `extra_water_nm`，在两端新增的空隙里各自铺一层按体相密度
    排布的新水（`_pack_water_slab`，不经过 `ForceField`/`Modeller.addSolvent`，
    理由见上方大段注释）——脂质/原有水/离子的 XY 排布和数量完全不动，两侧各
    获得约 `extra_water_nm/2` 的新水层。
    输出是**未平衡**的起始坐标，必须再跑一遍 `equilibrate-base` 才能进 `build`。
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    top_file, gro_file = args.top, args.gro

    # 不再需要 load_gromacs_topology_for_openmm/ForceField/Modeller——新水完全
    # 靠 `_pack_water_slab` 自建（见上方大段注释），这里只需要 `.gro` 的盒矢量。
    gro = app.GromacsGroFile(gro_file)
    box_vecs_nm = np.asarray([v.value_in_unit(unit.nanometer) for v in gro.getPeriodicBoxVectors()])
    if not np.allclose(box_vecs_nm, np.diag(np.diag(box_vecs_nm))):
        raise SystemExit("extend-water 只支持长方体盒（膜法向 z 的标准约定）")
    lx, ly, lz = np.diag(box_vecs_nm)
    extra = float(args.extra_water_nm)
    new_lz = lz + extra
    half_shift = extra / 2.0

    shifted_positions_nm = np.asarray(gro.positions.value_in_unit(unit.nanometer), dtype=np.float64)
    shifted_positions_nm[:, 2] += half_shift

    rng = np.random.default_rng(int(args.seed))
    bottom_waters, bottom_diag = _pack_water_slab(lx, ly, 0.0, half_shift, rng)
    top_waters, top_diag = _pack_water_slab(lx, ly, lz + half_shift, new_lz, rng)
    n_new_z_lt_half = bottom_waters.shape[0]
    n_new_z_gt_half = top_waters.shape[0]
    n_new_water = n_new_z_lt_half + n_new_z_gt_half
    new_water_positions = np.concatenate([bottom_waters, top_waters], axis=0).reshape(-1, 3)
    new_box_nm = np.diag([float(lx), float(ly), float(new_lz)])

    # v4 二次修法：目标水数/达到密度必须按**完整**新增体积算（`full_added_volume_nm3`），
    # 不是扣掉 buffer 后的 `usable_volume_nm3`——buffer 区域也是真实要填水的物理
    # 体积，见 `_pack_water_slab` docstring"第二轮"。两段合并算一个总的实际密度，
    # 如实打印+落盘，跟目标值的偏差一眼可见。
    combined_full_volume_nm3 = bottom_diag["full_added_volume_nm3"] + top_diag["full_added_volume_nm3"]
    combined_target_water_count = (
        bottom_diag["target_water_count_full_volume"] + top_diag["target_water_count_full_volume"]
    )
    combined_achieved_density_full_volume_nm3 = (
        float(n_new_water) / combined_full_volume_nm3 if combined_full_volume_nm3 > 0 else 0.0
    )
    density_deviation = abs(
        combined_achieved_density_full_volume_nm3 - BULK_WATER_NUMBER_DENSITY_PER_NM3
    ) / BULK_WATER_NUMBER_DENSITY_PER_NM3
    print(
        f"💧 沿 Z 平移 {half_shift:.3f} nm 后铺了 {n_new_water} 个水分子"
        f"（顶部 {n_new_z_gt_half} / 底部 {n_new_z_lt_half}——应大致对称）；"
        f"完整新增体积 {combined_full_volume_nm3:.2f} nm³ 对应目标水数约 "
        f"{combined_target_water_count}（目标数密度 {BULK_WATER_NUMBER_DENSITY_PER_NM3:.2f} /nm³）；"
        f"实际铺了 {n_new_water} 个，按完整体积算密度 {combined_achieved_density_full_volume_nm3:.2f} /nm³ "
        f"({'✅ 偏差' if density_deviation <= 0.10 else '⚠️ 偏差'} {100*density_deviation:.1f}%"
        f"{'' if density_deviation <= 0.10 else '，--extra-water-nm 越薄，取整损耗占比通常越大'})"
    )

    new_top_path = os.path.join(DEFAULT_GROMACS_DIR, f"{GENERATED_FILE_PREFIX}thick_topol.top")
    _edit_top_molecules_block(
        top_file, new_top_path, water_moleculetype="TP3", water_delta=n_new_water,
    )

    # 写新的 .gro：原有原子坐标（已平移）+ 自建的新水。
    title, n_atoms_old, atom_lines, _old_box_line = _read_gro(gro_file)
    shifted_atom_lines = [
        f"{line[0:20]}{shifted_positions_nm[i,0]:8.3f}{shifted_positions_nm[i,1]:8.3f}{shifted_positions_nm[i,2]:8.3f}\n"
        for i, line in enumerate(atom_lines)
    ]
    last_resnum = int(atom_lines[-1][0:5])
    appended_lines = []
    for w in range(n_new_water):
        resnum = last_resnum + 1 + w
        base_atomnum = n_atoms_old + 3 * w
        appended_lines.append(_format_gro_atom_line(resnum, "TP3", "O", base_atomnum + 1, new_water_positions[3*w]))
        appended_lines.append(_format_gro_atom_line(resnum, "TP3", "H1", base_atomnum + 2, new_water_positions[3*w+1]))
        appended_lines.append(_format_gro_atom_line(resnum, "TP3", "H2", base_atomnum + 3, new_water_positions[3*w+2]))

    new_gro_path = os.path.join(DEFAULT_GROMACS_DIR, f"{GENERATED_FILE_PREFIX}thick_step5_input.gro")
    _write_gro(new_gro_path, title, shifted_atom_lines + appended_lines, _format_gro_box_line(new_box_nm))

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "source_top": os.path.relpath(top_file, _REPO_ROOT),
        "source_gro": os.path.relpath(gro_file, _REPO_ROOT),
        "source_sha256": _sha256_tree([top_file, gro_file]),
        "extra_water_nm": extra, "half_shift_nm": half_shift,
        "n_new_water_molecules": int(n_new_water),
        "n_new_water_top": n_new_z_gt_half, "n_new_water_bottom": n_new_z_lt_half,
        "target_number_density_per_nm3": BULK_WATER_NUMBER_DENSITY_PER_NM3,
        # 按完整新增体积算的四个数字（不是扣掉 buffer 的 usable_volume）——
        # 2026-08-09 review 明确要求的口径，见 `_pack_water_slab` docstring
        # "第二轮"：buffer 区域是真实物理体积，密度分母不能把它扣掉。
        "full_added_volume_nm3": combined_full_volume_nm3,
        "target_water_count_full_volume": int(combined_target_water_count),
        "actual_water_count": int(n_new_water),
        "achieved_density_full_volume_nm3": combined_achieved_density_full_volume_nm3,
        "achieved_density_deviation_from_target": density_deviation,
        "top_slab_diagnostics": top_diag,
        "bottom_slab_diagnostics": bottom_diag,
        "generated_top": os.path.relpath(new_top_path, _REPO_ROOT),
        "generated_gro": os.path.relpath(new_gro_path, _REPO_ROOT),
        "generated_sha256": _sha256_tree([new_top_path, new_gro_path]),
        "box_before_nm": box_vecs_nm.diagonal().tolist(),
        "box_after_nm": new_box_nm.diagonal().tolist(),
        "note": "输出是未平衡的起始坐标，必须再跑 equilibrate-base 才能作为 build 的 --equilibrated-gro 输入。",
    }
    with open(os.path.join(output_dir, "extend_water_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"✅ extend-water 完成：{new_top_path}, {new_gro_path}")
    print("   ⚠️ 下一步：对这份新 .gro 跑 equilibrate-base，再进 build。")


# ============================================================================
# 子命令：build
# ============================================================================


def _required_ligand_coion_min_image_nm(
    restraint_k: Optional[float], restraint_r0_nm: Optional[float],
) -> float:
    """能保证 `abfe_core.validate_co_alchemical_ion_placement`（§13.1 runtime
    判据）通过所需要的、配体↔co-ion 的最小 minimum-image 距离——不是
    `core.COION_LIGAND_MIN_IMAGE_INITIAL_NM`（1.6 nm）那个更松的"initial"
    判据（v5→v6 修复，见模块 docstring）。

    C2 的探针都是单原子离子，`ligand_extent_from_anchor_nm` 恒为 0，判据
    公式（`abfe_core.validate_co_alchemical_ion_placement`）简化成：

        guaranteed = d0 − flat_bottom_radius_nm − wall_margin_nm(restraint_k)
        要求 guaranteed ≥ COION_LIGAND_MIN_IMAGE_RUNTIME_NM

    反解出 `d0` 至少要多大。`restraint_k`/`restraint_r0_nm` 为 `None` 时落到
    `select_co_alchemical_ion_once` 实际会用的同一组默认值
    （`core.COION_FLAT_BOTTOM_K_KJ_PER_MOL_NM2`/`core.COION_FLAT_BOTTOM_RADIUS_NM`），
    保证这里算的门槛跟后面真正配置 restraint 用的参数一致，不是两套互相
    脱节的数字。
    """
    radius = float(core.COION_FLAT_BOTTOM_RADIUS_NM if restraint_r0_nm is None else restraint_r0_nm)
    margin = core.co_alchemical_ion_restraint_wall_margin_nm(restraint_k)
    return core.COION_LIGAND_MIN_IMAGE_RUNTIME_NM + radius + margin


# 上面反解出的是"恰好压线"的最小距离——浮点误差 + `_pick_n_well_separated`
# 只保证 `>=` 而不追求刚好卡线，紧贴边界选出来的点在 `select_co_alchemical_ion_once`
# 内部重新算一遍距离时可能因为舍入差之毫厘就翻到判据的反面。加一点显式余量，
# 不靠运气卡边界。
LIGAND_COION_MIN_IMAGE_SAFETY_BUFFER_NM = 0.05


def insert_ions_into_gromacs_files(
    top_file: str, gro_file: str, include_dir: Optional[str],
    ion: str, position_variant: int, case_tag: str,
    restraint_k: Optional[float] = None, restraint_r0_nm: Optional[float] = None,
    bulk_restraint_rz_nm: float = BULK_RESTRAINT_RZ_NM,
    bulk_target_water_offset_nm: float = BULK_RESTRAINT_TARGET_OFFSET_DEFAULT_NM,
    bulk_relative_z_fluctuation_limit_nm: float = BULK_RESTRAINT_RELATIVE_Z_FLUCTUATION_LIMIT_DEFAULT_NM,
    bulk_membrane_undulation_margin_nm: float = BULK_RESTRAINT_MEMBRANE_UNDULATION_MARGIN_DEFAULT_NM,
    ligand_safety_rz_nm: float = LIGAND_BULK_RESTRAINT_RZ_NM,
    ligand_envelope_margin_nm: float = LIGAND_BULK_RESTRAINT_DESIGN_ENVELOPE_MARGIN_NM,
    coion_safety_rz_nm: float = COION_BULK_SAFETY_RZ_NM,
    coion_envelope_margin_nm: float = COION_BULK_SAFETY_DESIGN_ENVELOPE_MARGIN_NM,
) -> Dict[str, Any]:
    """核心插入逻辑（v2：普通反离子 + 探针配体 + reserved co-ion dummy 三粒子，
    保证 λ=1/λ=0 全盒总电荷都严格为 0，见模块 docstring §1）。

    三者顺序（也是 `.gro`/`[ molecules ]` 里追加的顺序，新拓扑里
    `N-3/N-2/N-1` 三个 index）：普通反离子 → 探针配体 → reserved dummy。

    `restraint_k`/`restraint_r0_nm`：与 `cmd_build` 最终喂给
    `select_co_alchemical_ion_once` 的**同一对**参数（`None` 时都落到同一组
    默认常量）——用来算配体↔co-ion 需要多远才能保证通过 §13.1 runtime 判据
    （v5→v6 修复：候选点筛选之前只按 `COION_LIGAND_MIN_IMAGE_INITIAL_NM`
    的松判据选，跟 restraint 真正需要的距离脱节，见模块 docstring）。
    """
    template = _ion_template(ion)
    ion_moleculetype = template["moleculetype"]
    counter_moleculetype = template["counter_moleculetype"]

    gmx_top = core.load_gromacs_topology_for_openmm(top_file, includeDir=include_dir)
    gro = app.GromacsGroFile(gro_file)
    topology = gmx_top.topology
    box_vectors = gro.getPeriodicBoxVectors()
    topology.setPeriodicBoxVectors(box_vectors)
    positions_nm = np.asarray(gro.positions.value_in_unit(unit.nanometer), dtype=np.float64)
    box_nm = np.asarray([v.value_in_unit(unit.nanometer) for v in box_vectors])

    parsed = core.parse_gromacs_topology(top_file, include_dir)
    composition = core.classify_system_composition(parsed)
    lipid_molecules = composition["molecules_by_role"]["lipid"]
    if not lipid_molecules:
        raise RuntimeError("classify_system_composition 没有识别出任何脂质分子")

    axis_report = core.verify_membrane_normal_axis(
        topology, gro.positions, declared_axis="z", lipid_molecules=lipid_molecules
    )
    leaflets = core.assign_lipid_leaflets(
        topology, gro.positions, normal_axis="z", lipid_molecules=lipid_molecules
    )
    midplane_z_nm = float(leaflets["midplane_coordinate_nm"])

    phosphorus_indices = _lipid_phosphorus_indices(topology)
    water_oxygens = _water_oxygen_indices_with_residue(topology)
    candidates = _find_bulk_water_candidates(
        positions_nm, box_nm, midplane_z_nm, phosphorus_indices, water_oxygens
    )
    side = POSITION_VARIANT_LABELS.get(int(position_variant))
    if side is None:
        raise SystemExit(f"--position-variant 只接受 {sorted(POSITION_VARIANT_LABELS)}")
    side_pool = [c for c in candidates if c["side"] == side]
    required_ligand_coion_nm = (
        _required_ligand_coion_min_image_nm(restraint_k, restraint_r0_nm)
        + LIGAND_COION_MIN_IMAGE_SAFETY_BUFFER_NM
    )
    min_mutual_nm = max(
        core.COION_COION_MIN_IMAGE_INITIAL_NM,
        core.COION_LIGAND_MIN_IMAGE_INITIAL_NM,
        required_ligand_coion_nm,
    )

    def v11_static_placement_validator(selected: List[Dict[str, Any]]) -> bool:
        ligand_candidate, dummy_candidate = selected[1], selected[2]
        ligand_index = int(ligand_candidate["o_index"])
        coion_index = int(dummy_candidate["o_index"])
        try:
            signed_target_fraction, _, _ = _bulk_target_fraction_from_initial_geometry(
                positions_nm, ligand_index, coion_index, midplane_z_nm, float(box_nm[2, 2]),
                float(bulk_target_water_offset_nm),
            )
            pair_design = _static_bulk_geometry_design(
                positions_nm, phosphorus_indices, ligand_index, coion_index, midplane_z_nm,
                float(box_nm[2, 2]), signed_target_fraction, float(bulk_restraint_rz_nm),
                float(bulk_relative_z_fluctuation_limit_nm),
                float(bulk_membrane_undulation_margin_nm), box_nm=box_nm,
            )
            ligand_design = _static_ligand_geometry_design(
                positions_nm, phosphorus_indices, [ligand_index], midplane_z_nm,
                float(box_nm[2, 2]), signed_target_fraction, float(ligand_safety_rz_nm),
                float(ligand_envelope_margin_nm), box_nm,
            )
            coion_design = _static_ligand_geometry_design(
                positions_nm, phosphorus_indices, [coion_index], midplane_z_nm,
                float(box_nm[2, 2]), signed_target_fraction, float(coion_safety_rz_nm),
                float(coion_envelope_margin_nm), box_nm,
            )
            return bool(pair_design["passed"] and ligand_design["passed"] and coion_design["passed"])
        except (ValueError, np.linalg.LinAlgError, KeyError):
            return False

    counter_cand, ligand_cand, dummy_cand = _pick_n_well_separated(
        side_pool, 3, positions_nm, box_nm, min_mutual_nm,
        validator=v11_static_placement_validator,
    )
    print(
        f"📍 position-variant={position_variant} ({side})：候选池 {len(side_pool)}/{len(candidates)}；"
        f"普通反离子 |Δz|={counter_cand['abs_dz_from_midplane_nm']:.3f} nm；"
        f"探针配体 |Δz|={ligand_cand['abs_dz_from_midplane_nm']:.3f} nm；"
        f"dummy |Δz|={dummy_cand['abs_dz_from_midplane_nm']:.3f} nm"
    )

    # ---- 编辑 .gro：摘掉 3 个整份水残基（各 3 原子），末尾按顺序追加 3 个新离子原子 ----
    title, n_atoms_old, atom_lines, box_line = _read_gro(gro_file)
    remove_atom_indices: set = set()
    for cand in (counter_cand, ligand_cand, dummy_cand):
        for atom in cand["residue"].atoms():
            remove_atom_indices.add(int(atom.index))
    if len(remove_atom_indices) != 9:
        raise RuntimeError(
            f"预期摘除 3 个水分子共 9 个原子，实际 {len(remove_atom_indices)}"
            f"（{sorted(remove_atom_indices)}）"
        )
    kept_lines = [line for i, line in enumerate(atom_lines) if i not in remove_atom_indices]
    last_resnum = int(kept_lines[-1][0:5])
    new_lines = []
    for offset, (moleculetype, cand) in enumerate(
        [(counter_moleculetype, counter_cand), (ion_moleculetype, ligand_cand), (ion_moleculetype, dummy_cand)]
    ):
        new_lines.append(
            _format_gro_atom_line(
                last_resnum + 1 + offset, moleculetype, moleculetype,
                len(kept_lines) + 1 + offset, positions_nm[cand["o_index"]],
            )
        )
    new_gro_path = os.path.join(DEFAULT_GROMACS_DIR, f"{GENERATED_FILE_PREFIX}{case_tag}.gro")
    _write_gro(new_gro_path, title, kept_lines + new_lines, box_line)

    # ---- 编辑 .top：TP3 计数 -3，末尾依次追加 counter×1、probe×2（配体+dummy 同一
    # moleculetype，各占一个 `[ molecules ]` 展开位置——两条 count=1 的连续块，
    # 用一行 count=2 与两行各 count=1 在原子顺序上完全等价，这里写成两行以避免
    # "配体和 dummy 共享一行"造成阅读歧义）----
    new_top_path = os.path.join(DEFAULT_GROMACS_DIR, f"{GENERATED_FILE_PREFIX}{case_tag}.top")
    _edit_top_molecules_block(
        top_file, new_top_path, water_moleculetype="TP3", water_delta=-3,
        appended_blocks=[(counter_moleculetype, 1), (ion_moleculetype, 1), (ion_moleculetype, 1)],
    )

    new_n_atoms = n_atoms_old - 9 + 3
    counter_index = new_n_atoms - 3
    ligand_index = new_n_atoms - 2
    dummy_index = new_n_atoms - 1

    return {
        "new_top_path": new_top_path, "new_gro_path": new_gro_path,
        "counter_index": int(counter_index), "ligand_index": int(ligand_index), "dummy_index": int(dummy_index),
        "ion": ion, "ion_moleculetype": ion_moleculetype, "counter_moleculetype": counter_moleculetype,
        "position_variant": int(position_variant), "position_variant_side": side,
        "midplane_z_nm": midplane_z_nm, "axis_report": axis_report, "leaflets_before_insertion": leaflets,
        "counter_site": {k: v for k, v in counter_cand.items() if k != "residue"},
        "ligand_site": {k: v for k, v in ligand_cand.items() if k != "residue"},
        "dummy_site": {k: v for k, v in dummy_cand.items() if k != "residue"},
        "removed_water_residue_atom_indices": sorted(remove_atom_indices),
        "source_top": os.path.relpath(top_file, _REPO_ROOT), "source_gro": os.path.relpath(gro_file, _REPO_ROOT),
    }


def cmd_build(args: argparse.Namespace) -> None:
    template = _ion_template(args.ion)
    q_l = int(template["charge_e"])
    q_counter = int(template["counter_charge_e"])
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    existing_manifest = os.path.join(output_dir, "build_manifest.json")
    if os.path.exists(existing_manifest):
        with open(existing_manifest, encoding="utf-8") as fh:
            existing = json.load(fh)
        raise SystemExit(
            f"拒绝覆盖已有 build 目录 {output_dir}（protocol_version={existing.get('protocol_version')!r}）。"
            "v9 必须使用全新的输出目录，不能覆盖或续跑 v8/v7 证据。"
        )

    top_file = args.top
    gro_file = args.equilibrated_gro
    if not os.path.isfile(gro_file):
        raise SystemExit(
            f"--equilibrated-gro 不存在: {gro_file}\n"
            "    C2 要求『每个体系预先平衡』——先跑 equilibrate-base + base-quality-gate"
            "（必要时先 extend-water），把通过质量门的 equilibrated.gro 传进来。"
        )

    case_tag = f"{args.ion}_{args.water_thickness_label}_pos{args.position_variant}"
    print(f"⚗️  case={case_tag}：探针配体={args.ion}{'+' if q_l > 0 else '-'} (净电荷 {q_l:+d} e)")

    insertion = insert_ions_into_gromacs_files(
        top_file, gro_file, args.gmx_include_dir, args.ion, args.position_variant, case_tag,
        restraint_k=args.restraint_k, restraint_r0_nm=args.restraint_r0_nm,
        bulk_restraint_rz_nm=float(getattr(args, "bulk_restraint_rz_nm", BULK_RESTRAINT_RZ_NM)),
        bulk_target_water_offset_nm=float(getattr(
            args, "bulk_target_water_offset_nm", BULK_RESTRAINT_TARGET_OFFSET_DEFAULT_NM,
        )),
        bulk_relative_z_fluctuation_limit_nm=float(getattr(
            args, "bulk_relative_z_fluctuation_limit_nm",
            BULK_RESTRAINT_RELATIVE_Z_FLUCTUATION_LIMIT_DEFAULT_NM,
        )),
        bulk_membrane_undulation_margin_nm=float(getattr(
            args, "bulk_membrane_undulation_margin_nm",
            BULK_RESTRAINT_MEMBRANE_UNDULATION_MARGIN_DEFAULT_NM,
        )),
        ligand_safety_rz_nm=float(getattr(args, "ligand_bulk_safety_rz_nm", LIGAND_BULK_RESTRAINT_RZ_NM)),
        ligand_envelope_margin_nm=float(getattr(
            args, "ligand_envelope_margin_nm", LIGAND_BULK_RESTRAINT_DESIGN_ENVELOPE_MARGIN_NM,
        )),
        coion_safety_rz_nm=float(getattr(args, "coion_bulk_safety_rz_nm", COION_BULK_SAFETY_RZ_NM)),
        coion_envelope_margin_nm=float(getattr(
            args, "coion_envelope_margin_nm", COION_BULK_SAFETY_DESIGN_ENVELOPE_MARGIN_NM,
        )),
    )

    gmx_top2 = core.load_gromacs_topology_for_openmm(insertion["new_top_path"], includeDir=args.gmx_include_dir)
    gro2 = app.GromacsGroFile(insertion["new_gro_path"])
    box_vectors2 = gro2.getPeriodicBoxVectors()
    gmx_top2.topology.setPeriodicBoxVectors(box_vectors2)
    positions2_nm = np.asarray(gro2.positions.value_in_unit(unit.nanometer), dtype=np.float64)
    box2_nm = np.asarray([v.value_in_unit(unit.nanometer) for v in box_vectors2])

    system = gmx_top2.createSystem(
        nonbondedMethod=app.PME, nonbondedCutoff=NONBONDED_CUTOFF_NM * unit.nanometer,
        constraints=app.HBonds, rigidWater=True, ewaldErrorTolerance=EWALD_ERROR_TOLERANCE,
    )
    dispersion = _resolve_and_verify_dispersion_protocol(insertion["new_top_path"], args.gmx_include_dir)
    nb_force = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    nb_force.setUseDispersionCorrection(True)
    # v3 修法：narrow potential-switch，见 C2_LJ_SWITCH_DISTANCE_NM 上方注释。
    nb_force.setUseSwitchingFunction(True)
    nb_force.setSwitchingDistance(C2_LJ_SWITCH_DISTANCE_NM * unit.nanometer)

    # v10：以平衡后盒子的实际初始位置定义 pair-center/ligand safety 的 signed Z 目标分数；
    # 不把实验室坐标写死。动态膜中面与 Lz 会在 dynamics 每个积分小段更新。
    p31_indices = _p31_indices(gmx_top2.topology)
    bulk_restraint_kz = float(getattr(args, "bulk_restraint_kz", BULK_RESTRAINT_KZ_KJ_PER_MOL_NM2))
    bulk_restraint_rz = float(getattr(args, "bulk_restraint_rz_nm", BULK_RESTRAINT_RZ_NM))
    target_offset_nm = float(getattr(
        args, "bulk_target_water_offset_nm", BULK_RESTRAINT_TARGET_OFFSET_DEFAULT_NM,
    ))
    relative_z_limit_nm = float(getattr(
        args, "bulk_relative_z_fluctuation_limit_nm",
        BULK_RESTRAINT_RELATIVE_Z_FLUCTUATION_LIMIT_DEFAULT_NM,
    ))
    undulation_margin_nm = float(getattr(
        args, "bulk_membrane_undulation_margin_nm",
        BULK_RESTRAINT_MEMBRANE_UNDULATION_MARGIN_DEFAULT_NM,
    ))
    ligand_safety_kz = float(getattr(
        args, "ligand_bulk_safety_kz", LIGAND_BULK_RESTRAINT_KZ_KJ_PER_MOL_NM2,
    ))
    ligand_safety_rz = float(getattr(
        args, "ligand_bulk_safety_rz_nm", LIGAND_BULK_RESTRAINT_RZ_NM,
    ))
    ligand_envelope_margin_nm = float(getattr(
        args, "ligand_envelope_margin_nm", LIGAND_BULK_RESTRAINT_DESIGN_ENVELOPE_MARGIN_NM,
    ))
    coion_safety_kz = float(getattr(
        args, "coion_bulk_safety_kz", COION_BULK_SAFETY_KZ_KJ_PER_MOL_NM2,
    ))
    coion_safety_rz = float(getattr(
        args, "coion_bulk_safety_rz_nm", COION_BULK_SAFETY_RZ_NM,
    ))
    coion_envelope_margin_nm = float(getattr(
        args, "coion_envelope_margin_nm", COION_BULK_SAFETY_DESIGN_ENVELOPE_MARGIN_NM,
    ))
    signed_target_fraction, initial_signed_delta_z_nm, initial_target_signed_delta_z_nm = (
        _bulk_target_fraction_from_initial_geometry(
            positions2_nm, insertion["ligand_index"], insertion["dummy_index"],
            float(insertion["midplane_z_nm"]), float(box2_nm[2, 2]), target_offset_nm,
        )
    )
    initial_pair_center_z_nm = _pbc_pair_center_z_nm(
        positions2_nm, insertion["ligand_index"], insertion["dummy_index"], float(box2_nm[2, 2])
    )
    static_geometry_design = _static_bulk_geometry_design(
        positions2_nm, p31_indices, insertion["ligand_index"], insertion["dummy_index"],
        float(insertion["midplane_z_nm"]), float(box2_nm[2, 2]), signed_target_fraction,
        bulk_restraint_rz, relative_z_limit_nm, undulation_margin_nm, box_nm=box2_nm,
    )
    if not static_geometry_design["passed"]:
        raise SystemExit(
            "v10 pair-center bulk restraint 静态几何设计失败："
            f" worst |Δz|={static_geometry_design['min_worst_case_abs_dz_from_midplane_nm']:.3f} nm,"
            f" worst P31 z-distance={static_geometry_design['min_worst_case_nearest_p31_z_nm']:.3f} nm"
            f"（设计要求分别 ≥{static_geometry_design['required_design_abs_dz_from_midplane_nm']:.3f}/"
            f"{static_geometry_design['required_design_nearest_p31_nm']:.3f} nm）"
        )
    initial_target_z_nm = float(insertion["midplane_z_nm"]) + initial_target_signed_delta_z_nm
    initial_target_z_nm += float(box2_nm[2, 2]) * round(
        (float(positions2_nm[insertion["ligand_index"], 2]) - initial_target_z_nm) / float(box2_nm[2, 2])
    )
    bulk_force = _create_bulk_restraint_force(
        insertion["ligand_index"], insertion["dummy_index"], initial_target_z_nm,
        k_z_kj_per_mol_nm2=bulk_restraint_kz, r_z_nm=bulk_restraint_rz,
    )
    system.addForce(bulk_force)
    ligand_geometry_design = _static_ligand_geometry_design(
        positions2_nm, p31_indices, [insertion["ligand_index"]],
        float(insertion["midplane_z_nm"]), float(box2_nm[2, 2]), signed_target_fraction,
        ligand_safety_rz, ligand_envelope_margin_nm, box2_nm,
    )
    if not ligand_geometry_design["passed"]:
        raise SystemExit(
            "v10 ligand safety 静态几何设计失败："
            f" worst |Δz|={ligand_geometry_design['min_worst_case_abs_dz_from_midplane_nm']:.3f} nm,"
            f" worst P31={ligand_geometry_design['min_worst_case_nearest_p31_nm']:.3f} nm"
        )
    ligand_safety_force = _create_ligand_bulk_safety_force(
        insertion["ligand_index"], initial_target_z_nm,
        k_z_kj_per_mol_nm2=ligand_safety_kz, r_z_nm=ligand_safety_rz,
    )
    system.addForce(ligand_safety_force)
    coion_geometry_design = _static_ligand_geometry_design(
        positions2_nm, p31_indices, [insertion["dummy_index"]],
        float(insertion["midplane_z_nm"]), float(box2_nm[2, 2]), signed_target_fraction,
        coion_safety_rz, coion_envelope_margin_nm, box2_nm,
    )
    if not coion_geometry_design["passed"]:
        raise SystemExit(
            "v11 co-ion safety 静态几何设计失败："
            f" worst |Δz|={coion_geometry_design['min_worst_case_abs_dz_from_midplane_nm']:.3f} nm,"
            f" worst P31={coion_geometry_design['min_worst_case_nearest_p31_nm']:.3f} nm"
        )
    coion_safety_force = _create_coion_bulk_safety_force(
        insertion["dummy_index"], initial_target_z_nm,
        k_z_kj_per_mol_nm2=coion_safety_kz, r_z_nm=coion_safety_rz,
    )
    system.addForce(coion_safety_force)
    print(
        f"  ✅ v10 pair bulk restraint: kZ={bulk_restraint_kz:.1f}, rZ={bulk_restraint_rz:.3f} nm;"
        f" target fraction={signed_target_fraction:+.6f}, initial pair-center Δz={initial_signed_delta_z_nm:+.3f} nm"
    )
    print(
        f"  ✅ v11 ligand safety wall: kZ={ligand_safety_kz:.1f}, rZ={ligand_safety_rz:.3f} nm;"
        f" reference atom={insertion['ligand_index']}"
    )
    print(
        f"  ✅ v11 co-ion safety wall: kZ={coion_safety_kz:.1f}, rZ={coion_safety_rz:.3f} nm;"
        f" reference atom={insertion['dummy_index']}"
    )

    membrane_protocol = core.resolve_membrane_protocol(
        core.ENVIRONMENT_TYPE_MEMBRANE, membrane_config=None, topology=gmx_top2.topology
    )
    core.ensure_barostat_for_protocol(system, membrane_protocol, temperature=303.15, pressure=1.0)

    counter_index = insertion["counter_index"]
    ligand_indices = [insertion["ligand_index"]]
    dummy_index = insertion["dummy_index"]

    q_counter_read, sigma_c, eps_c = nb_force.getParticleParameters(counter_index)
    q_counter_e = q_counter_read.value_in_unit(unit.elementary_charge)
    if abs(q_counter_e - q_counter) > 1.0e-9:
        raise RuntimeError(f"普通反离子读回电荷 {q_counter_e:+.9f} e，与模板 {q_counter:+d} e 不符")

    q_ligand, sigma_l, eps_l = nb_force.getParticleParameters(ligand_indices[0])
    q_ligand_e = q_ligand.value_in_unit(unit.elementary_charge)
    if abs(q_ligand_e - q_l) > 1.0e-9:
        raise RuntimeError(f"探针配体读回电荷 {q_ligand_e:+.9f} e，与模板 {q_l:+d} e 不符")

    q_dummy, sigma_d, eps_d = nb_force.getParticleParameters(dummy_index)
    q_dummy_e_before = q_dummy.value_in_unit(unit.elementary_charge)
    if abs(q_dummy_e_before - q_l) > 1.0e-9:
        raise RuntimeError(f"reserved dummy 清零前电荷 {q_dummy_e_before:+.9f} e，与模板 {q_l:+d} e 不符")
    nb_force.setParticleParameters(dummy_index, 0.0 * unit.elementary_charge, sigma_d, eps_d)
    print(f"  ✅ reserved dummy (index={dummy_index}) 电荷已清零：{q_dummy_e_before:+.0f} → 0")

    # ---- v2 §1 硬断言：λ=1 端（此刻的原始 System，还没配置任何 λ offset）
    # 全盒总电荷必须严格为 0，不是"配体+co-ion+反离子=0"这个子集算术——直接对
    # NonbondedForce 全部粒子求和，逐一核对反离子确实在起作用。 ----
    total_charge_e = sum(
        nb_force.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
        for i in range(nb_force.getNumParticles())
    )
    if abs(total_charge_e) > core.TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:
        raise RuntimeError(
            f"build 完成后（λ=1 端）全盒总电荷 = {total_charge_e:+.9f} e，应严格为 0。"
            f"探针={q_l:+d}, dummy=0, 普通反离子={q_counter:+d} —— 请检查普通反离子"
            "是否真的插入、电荷是否被误改。"
        )
    print(f"  ✅ λ=1 端全盒总电荷 = {total_charge_e:+.2e} e（应为 0）")

    spec = engine.select_co_alchemical_ion_once(
        system, ligand_indices, gmx_top2.topology, gro2.positions, box_vectors2,
        charge_treatment=core.CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
        ion_restraint_k=args.restraint_k, flat_bottom_radius_nm=args.restraint_r0_nm,
    )
    if spec is None:
        raise RuntimeError("select_co_alchemical_ion_once 返回 None——不应该发生（配体带净电荷）")
    ion_indices = [int(i["atom_index"]) for i in spec["ions"]]
    if ion_indices != [dummy_index]:
        raise RuntimeError(
            f"select_co_alchemical_ion_once 选出的 co-ion index={ion_indices}，"
            f"与插入的 dummy index={dummy_index} 不一致——说明体系里还有别的零电荷"
            "离子残基粒子，_identify_reserved_neutral_co_ions 的『恰好一个』判据"
            "本该拦住这个，请检查（尤其是普通反离子有没有被误清零成 0）。"
        )

    ligand_coion_distance_nm = _minimum_image_distance_nm(
        positions2_nm[ligand_indices[0]], positions2_nm[dummy_index], box2_nm
    )
    print(f"  📏 配体↔co-ion minimum-image 距离: {ligand_coion_distance_nm:.3f} nm")

    sys_xml_path = os.path.join(output_dir, "system.xml")
    with open(sys_xml_path, "w") as fh:
        fh.write(XmlSerializer.serialize(core.ensure_owned_system(system)))
    top_cif_path = os.path.join(output_dir, "topology.cif")
    app.PDBxFile.writeFile(gmx_top2.topology, gro2.positions, top_cif_path)
    np.save(os.path.join(output_dir, "positions_nm.npy"), positions2_nm)
    np.save(os.path.join(output_dir, "box_vectors_nm.npy"), box2_nm)
    with open(os.path.join(output_dir, "ligand_indices.json"), "w") as fh:
        json.dump({"ligand_indices": ligand_indices}, fh)
    with open(os.path.join(output_dir, "coalchemical_ion_spec.json"), "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2, cls=core.NumpyEncoder)

    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "case": case_tag, "ion": args.ion, "ligand_net_charge_e": q_l,
        "counter_ion_index": counter_index, "counter_ion_moleculetype": insertion["counter_moleculetype"],
        "counter_ion_charge_e": q_counter,
        "water_thickness_label": args.water_thickness_label,
        "position_variant": int(args.position_variant), "position_variant_side": insertion["position_variant_side"],
        "n_atoms": system.getNumParticles(), "ligand_indices": ligand_indices, "co_ion_indices": ion_indices,
        "ligand_coion_min_image_distance_nm": ligand_coion_distance_nm,
        "total_charge_at_build_e": float(total_charge_e),
        "midplane_z_nm": insertion["midplane_z_nm"],
        "counter_site": insertion["counter_site"], "ligand_site": insertion["ligand_site"], "dummy_site": insertion["dummy_site"],
        "leaflets_before_insertion": insertion["leaflets_before_insertion"],
        "membrane_normal_axis_report": insertion["axis_report"],
        "nonbonded_cutoff_nm": NONBONDED_CUTOFF_NM,
        "dispersion_protocol": dispersion["dispersion_protocol"],
        "forcefield_family": dispersion["forcefield_family_report"]["family"],
        "barostat_class": membrane_protocol["barostat_class"],
        "charge_treatment": core.CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
        "coalchemical_ion_fingerprint": spec.get("fingerprint"),
        "bulk_water_restraint": {
            "enabled": True, "form": BULK_RESTRAINT_FORM,
            "expression": BULK_RESTRAINT_EXPRESSION,
            "force_group": BULK_RESTRAINT_FORCE_GROUP,
            "k_z_kJ_mol_nm2": bulk_restraint_kz, "r_z_nm": bulk_restraint_rz,
            "initial_pair_center_z_nm": initial_pair_center_z_nm,
            "initial_signed_delta_z_nm": initial_signed_delta_z_nm,
            "initial_target_signed_delta_z_nm": initial_target_signed_delta_z_nm,
            "signed_target_fraction": signed_target_fraction,
            "target_offset_toward_water_nm": target_offset_nm,
            "initial_target_z_nm": initial_target_z_nm,
            "relative_z_fluctuation_limit_nm": relative_z_limit_nm,
            "membrane_undulation_margin_nm": undulation_margin_nm,
            "static_geometry_design": static_geometry_design,
            "target_reference": "dynamic_P31_midplane_plus_signed_fraction_of_current_Lz",
        },
        "ligand_bulk_safety": {
            "enabled": True,
            "form": LIGAND_BULK_RESTRAINT_FORM,
            "expression": LIGAND_BULK_RESTRAINT_EXPRESSION,
            "force_group": LIGAND_BULK_RESTRAINT_FORCE_GROUP,
            "reference_index": int(insertion["ligand_index"]),
            "k_z_kJ_mol_nm2": ligand_safety_kz,
            "r_z_nm": ligand_safety_rz,
            "envelope_margin_nm": ligand_envelope_margin_nm,
            "static_geometry_design": ligand_geometry_design,
            "target_reference": "dynamic_P31_midplane_plus_same_signed_fraction_of_current_Lz",
        },
        "coion_bulk_safety": {
            "enabled": True,
            "form": COION_BULK_SAFETY_FORM,
            "expression": COION_BULK_SAFETY_EXPRESSION,
            "force_group": COION_BULK_SAFETY_FORCE_GROUP,
            "reference_index": int(insertion["dummy_index"]),
            "k_z_kJ_mol_nm2": coion_safety_kz,
            "r_z_nm": coion_safety_rz,
            "envelope_margin_nm": coion_envelope_margin_nm,
            "static_geometry_design": coion_geometry_design,
            "target_reference": "dynamic_P31_midplane_plus_same_signed_fraction_of_current_Lz",
            "fingerprint_scope": "member_level_coion_bulk_safety",
        },
        "source_top": insertion["source_top"], "source_gro": insertion["source_gro"],
        "generated_top": os.path.relpath(insertion["new_top_path"], _REPO_ROOT),
        "generated_gro": os.path.relpath(insertion["new_gro_path"], _REPO_ROOT),
        "generated_sha256": _sha256_tree([insertion["new_top_path"], insertion["new_gro_path"]]),
        "system_xml_sha256": _sha256_file(sys_xml_path),
    }
    with open(os.path.join(output_dir, "build_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, cls=core.NumpyEncoder)
    print(f"✅ build 完成: {output_dir}")


# ============================================================================
# 子命令：static-check（纯 CPU）
# ============================================================================


def _load_build_artifacts(output_dir: str, *, allow_legacy: bool = False):
    with open(os.path.join(output_dir, "build_manifest.json")) as fh:
        manifest = json.load(fh)
    _require_protocol_version(
        manifest, os.path.join(output_dir, "build_manifest.json"), allow_legacy=allow_legacy,
    )
    with open(os.path.join(output_dir, "system.xml")) as fh:
        system = XmlSerializer.deserialize(fh.read())
    topology = app.PDBxFile(os.path.join(output_dir, "topology.cif")).topology
    positions_nm = np.load(os.path.join(output_dir, "positions_nm.npy"))
    box_nm = np.load(os.path.join(output_dir, "box_vectors_nm.npy"))
    with open(os.path.join(output_dir, "ligand_indices.json")) as fh:
        ligand_indices = json.load(fh)["ligand_indices"]
    with open(os.path.join(output_dir, "coalchemical_ion_spec.json")) as fh:
        spec = json.load(fh)
    return system, topology, positions_nm, box_nm, ligand_indices, spec, manifest


def _assert_single_restraint_force(system: openmm.System, n_expected: int) -> None:
    matches = [
        f for f in system.getForces()
        if type(f).__name__ == RESTRAINT_FORCE_CLASS_NAME
        and f.getForceGroup() == core.CO_ALCHEMICAL_ION_RESTRAINT_FORCE_GROUP
    ]
    if len(matches) != n_expected:
        raise SystemExit(
            f"System 里 co-ion restraint force（{RESTRAINT_FORCE_CLASS_NAME}，"
            f"force group={core.CO_ALCHEMICAL_ION_RESTRAINT_FORCE_GROUP}）数量={len(matches)}，"
            f"应为 {n_expected}。v1 的 bug 是 static-check/dynamics 手动调用 "
            "`_inject_co_alchemical_ion_restraints` 之后，`configure_pme_ligand_charge_offsets` "
            "内部（经 `configure_charge_transfer_decharging`）又注入一次——现在只应调用"
            "一次配置函数，不要在外面重复注入。"
        )


def cmd_static_check(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    system, topology, positions_nm, box_nm, ligand_indices, spec, manifest = _load_build_artifacts(output_dir)
    q_l = int(manifest["ligand_net_charge_e"])
    box_vectors = tuple(Vec3(*row) for row in box_nm) * unit.nanometer

    _assert_single_restraint_force(system, n_expected=0)  # build 阶段的 system.xml 还不该带 restraint
    bulk_cfg = manifest.get("bulk_water_restraint", {})
    if bulk_cfg.get("enabled"):
        _assert_single_bulk_restraint_force(system)
    ligand_safety_cfg = manifest.get("ligand_bulk_safety", {})
    if ligand_safety_cfg.get("enabled"):
        _assert_single_ligand_bulk_safety_force(system)
    coion_safety_cfg = manifest.get("coion_bulk_safety", {})
    if coion_safety_cfg.get("enabled"):
        _assert_single_coion_bulk_safety_force(system)

    pinned = core.verify_co_alchemical_ion_identity(
        spec, system=system, topology=topology,
        charge_treatment=core.CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
        ligand_net_charge_e=q_l, context="validate_charge_transfer_lipid_slab.static-check",
    )
    print(f"✅ verify_co_alchemical_ion_identity 通过，co-ion indices={list(pinned)}")

    # v2 修复：只调用一次 configure_pme_ligand_charge_offsets（它内部会调用
    # configure_charge_transfer_decharging，后者会自己注入 restraint）——不再在
    # 外面手动调用 `_inject_co_alchemical_ion_restraints`。
    info = engine.configure_pme_ligand_charge_offsets(
        system, ligand_indices, lambda_name="lambda_coul", allow_charged_ligand=True,
        topology=topology, positions=positions_nm * unit.nanometer,
        box_vectors=box_vectors, co_alchemical_ion_spec=spec,
    )
    print(f"✅ configure_pme_ligand_charge_offsets: mode={info['mode']}, n_offsets={info['n_offsets']}")
    _assert_single_restraint_force(system, n_expected=len(spec["ions"]))
    print(f"✅ restraint force 数量核对通过：恰好 {len(spec['ions'])} 份")
    if bulk_cfg.get("enabled"):
        _assert_single_bulk_restraint_force(system)
        print(f"✅ v10 pair bulk-water restraint force 数量核对通过：恰好 1 份（group={BULK_RESTRAINT_FORCE_GROUP}）")
    if ligand_safety_cfg.get("enabled"):
        _assert_single_ligand_bulk_safety_force(system)
        print(f"✅ v10 ligand safety force 数量核对通过：恰好 1 份（group={LIGAND_BULK_RESTRAINT_FORCE_GROUP}）")
    if coion_safety_cfg.get("enabled"):
        _assert_single_coion_bulk_safety_force(system)
        print(f"✅ v11 co-ion safety force 数量核对通过：恰好 1 份（group={COION_BULK_SAFETY_FORCE_GROUP}）")

    nb_force = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    lambdas = [round(x, 2) for x in np.arange(1.0, -0.001, -0.1)]
    report = engine.charging_charge_conservation_report(
        nb_force, "lambda_coul", ligand_indices=ligand_indices,
        co_ion_indices=list(pinned), ligand_net_charge_e=q_l, lambdas=lambdas,
    )
    print(f"✅ 电荷守恒代数证明（Σq_scale = {report['scale_sum_e']:+.3e} e）")

    # v2 §1 硬断言：不是只查"恒定"，要查"恒为 0"——`report["base_sum_e"]` 是
    # NonbondedForce **全部**粒子的电荷和（见 `charging_charge_conservation_report`
    # 的实现，n = nb_force.getNumParticles()），配合 total_charge_is_lambda_independent
    # 就是"所有 λ 的全盒总电荷都是这个值"。
    if abs(report["base_sum_e"]) > core.TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:
        raise SystemExit(
            f"全盒总电荷 = {report['base_sum_e']:+.9f} e，不是 0（容差 "
            f"{core.TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:g} e）。"
            "普通反离子没有正确配平——检查 build 是否用的是 v2 三粒子插入逻辑。"
        )
    for lam_key, total in report["total_charge_by_lambda_e"].items():
        if abs(total) > core.TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:
            raise SystemExit(f"λ_coul={lam_key} 时全盒总电荷 = {total:+.9f} e，应为 0")
    print(f"✅ 全部 {len(lambdas)} 个 λ 的全盒总电荷都严格为 0")

    coion_index = int(next(iter(pinned)))
    p_indices = _lipid_phosphorus_indices(topology)
    p_positions = positions_nm[np.asarray(p_indices, dtype=int)]
    midplane_z_nm = float(manifest["midplane_z_nm"])
    bulk_geometry_design = None
    if bulk_cfg.get("enabled") and int(manifest.get("protocol_version", 0)) >= PROTOCOL_VERSION:
        bulk_geometry_design = _static_bulk_geometry_design(
            positions_nm, p_indices, int(ligand_indices[0]), coion_index,
            midplane_z_nm, float(box_nm[2, 2]), float(bulk_cfg["signed_target_fraction"]),
            float(bulk_cfg["r_z_nm"]), float(bulk_cfg["relative_z_fluctuation_limit_nm"]),
            float(bulk_cfg["membrane_undulation_margin_nm"]), box_nm=box_nm,
        )
        if not bulk_geometry_design["passed"]:
            raise SystemExit(f"v9 static bulk geometry design failed: {bulk_geometry_design}")
        print("✅ v9 pair-center well 的静态几何包络通过（含相对 Z 波动与膜起伏裕量）")
    ligand_geometry_design = None
    if ligand_safety_cfg.get("enabled") and int(manifest.get("protocol_version", 0)) >= PROTOCOL_VERSION:
        ligand_geometry_design = _static_ligand_geometry_design(
            positions_nm, p_indices, ligand_indices, midplane_z_nm, float(box_nm[2, 2]),
            float(bulk_cfg["signed_target_fraction"]), float(ligand_safety_cfg["r_z_nm"]),
            float(ligand_safety_cfg["envelope_margin_nm"]), box_nm,
        )
        if not ligand_geometry_design["passed"]:
            raise SystemExit(f"v10 static ligand safety geometry design failed: {ligand_geometry_design}")
        print("✅ v10 ligand heavy-atom envelope static geometry passed")
    coion_geometry_design = None
    if coion_safety_cfg.get("enabled") and int(manifest.get("protocol_version", 0)) >= PROTOCOL_VERSION:
        coion_geometry_design = _static_ligand_geometry_design(
            positions_nm, p_indices, [coion_index], midplane_z_nm, float(box_nm[2, 2]),
            float(bulk_cfg["signed_target_fraction"]), float(coion_safety_cfg["r_z_nm"]),
            float(coion_safety_cfg["envelope_margin_nm"]), box_nm,
        )
        if not coion_geometry_design["passed"]:
            raise SystemExit(f"v11 static co-ion safety geometry design failed: {coion_geometry_design}")
        print("✅ v11 co-ion member safety geometry passed")
    for label, idx in (("配体", ligand_indices[0]), ("co-ion", coion_index), ("普通反离子", manifest["counter_ion_index"])):
        abs_dz = abs(_minimum_image_z_delta_nm(positions_nm[idx][2], midplane_z_nm, float(box_nm[2, 2])))
        nearest_p = float(np.min(_minimum_image_distances_nm(p_positions, positions_nm[idx], box_nm)))
        print(f"   {label} (index={idx}): |Δz|={abs_dz:.3f} nm, 最近 P31={nearest_p:.3f} nm")
        if abs_dz < core.COION_MEMBRANE_MIDPLANE_MIN_ABS_Z_NM:
            raise SystemExit(f"{label} |Δz|={abs_dz:.3f} nm 低于 bulk-water 门槛")
        if nearest_p < core.COION_NEAREST_PHOSPHORUS_MIN_NM:
            raise SystemExit(f"{label} 最近 P31={nearest_p:.3f} nm 低于 bulk-water 门槛")
    ligand_coion_distance_nm = _minimum_image_distance_nm(
        positions_nm[ligand_indices[0]], positions_nm[coion_index], box_nm
    )
    if ligand_coion_distance_nm < core.COION_LIGAND_MIN_IMAGE_INITIAL_NM:
        raise SystemExit(
            f"配体↔co-ion 距离 {ligand_coion_distance_nm:.3f} nm 低于 §13.1 初始门槛 "
            f"{core.COION_LIGAND_MIN_IMAGE_INITIAL_NM} nm"
        )
    print(f"✅ bulk-water 几何自检通过（配体↔co-ion={ligand_coion_distance_nm:.3f} nm）")

    for ion in spec["ions"]:
        idx = int(ion["atom_index"])
        q1 = float(ion["charge_at_lambda1_e"])
        if abs(q1) > core.TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:
            raise SystemExit(f"co-ion {idx} 在 λ=1 端不是严格中性: {q1:+.9f} e")
    print(f"✅ {len(spec['ions'])} 个 co-ion 在 λ=1 端严格中性")

    for ion in spec["ions"]:
        fg = int(ion["restraint"]["force_group"])
        if fg != core.CO_ALCHEMICAL_ION_RESTRAINT_FORCE_GROUP:
            raise SystemExit(f"co-ion {ion['atom_index']} restraint force group={fg}")
    print(f"✅ 全部 co-ion restraint 声明的 force group = {core.CO_ALCHEMICAL_ION_RESTRAINT_FORCE_GROUP}")

    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    platform = openmm.Platform.getPlatformByName("CPU")
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions_nm * unit.nanometer)
    context.setPeriodicBoxVectors(*box_vectors)
    energies = {}
    for lam in (1.0, 0.5, 0.0):
        context.setParameter("lambda_coul", lam)
        state = context.getState(getEnergy=True, getForces=True)
        pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        max_force = float(np.max(np.linalg.norm(forces, axis=1)))
        if not np.isfinite(pe) or not np.isfinite(max_force):
            raise SystemExit(f"λ_coul={lam}: 能量或力出现非有限值 (PE={pe}, max|F|={max_force})")
        energies[lam] = {"potential_kj_mol": pe, "max_force_kj_mol_nm": max_force}
        print(f"   λ_coul={lam:.2f}: PE={pe:.3f} kJ/mol, max|F|={max_force:.3f} kJ/mol/nm")
    del context, integrator
    print("✅ CPU 单点能量/力自检：全部有限")

    result = {
        "case": manifest["case"], "identity_fingerprint": pinned,
        "charge_conservation_report": report, "endpoint_single_point_energies_cpu": energies,
        "ligand_coion_min_image_distance_nm": ligand_coion_distance_nm,
        "total_charge_is_zero_at_all_lambda": True,
        "passed": bool(report["total_charge_is_lambda_independent"]) and abs(report["base_sum_e"]) <= core.TOTAL_CHARGE_CONSERVATION_TOLERANCE_E,
        "bulk_geometry_design": bulk_geometry_design,
        "ligand_geometry_design": ligand_geometry_design,
        "coion_geometry_design": coion_geometry_design,
    }
    out_path = os.path.join(output_dir, "static_check_report.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, cls=core.NumpyEncoder)
    print(f"✅ static-check 完成，报告: {out_path}")


# ============================================================================
# 子命令：dynamics（GPU/CUDA，用户在计算节点提交）
# ============================================================================


def cmd_dynamics(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    build_artifacts_dir = getattr(args, "build_artifacts_dir", None) or output_dir
    system, topology, positions_nm, box_nm, ligand_indices, spec, manifest = _load_build_artifacts(build_artifacts_dir)
    restart_dcd = getattr(args, "restart_dcd", None)
    restart_frame = int(getattr(args, "restart_frame", -1))
    restart_info = None
    simulation_positions_nm = np.asarray(positions_nm, dtype=np.float64)
    simulation_box_nm = np.asarray(box_nm, dtype=np.float64)
    if restart_dcd:
        simulation_positions_nm, simulation_box_nm, selected_restart_frame, restart_n_frames = (
            _load_restart_frame_from_dcd(restart_dcd, topology, restart_frame)
        )
        restart_info = {
            "dcd": os.path.relpath(restart_dcd, _REPO_ROOT),
            "requested_frame": restart_frame,
            "selected_frame": selected_restart_frame,
            "n_frames": restart_n_frames,
        }
        print(
            f"🔁 从 restart DCD 续接：{restart_dcd} frame={selected_restart_frame}/"
            f"{restart_n_frames - 1}（不重跑前一 λ）"
        )
    q_l = int(manifest["ligand_net_charge_e"])
    ion_indices = [int(ion["atom_index"]) for ion in spec["ions"]]
    build_box_vectors = tuple(Vec3(*row) for row in box_nm) * unit.nanometer
    simulation_box_vectors = tuple(Vec3(*row) for row in simulation_box_nm) * unit.nanometer
    bulk_cfg = manifest.get("bulk_water_restraint", {})
    if not bulk_cfg.get("enabled"):
        raise SystemExit("v10 dynamics 要求 build_manifest.json 含 enabled=true 的 bulk_water_restraint")
    _assert_single_bulk_restraint_force(system)
    ligand_safety_cfg = manifest.get("ligand_bulk_safety", {})
    if not ligand_safety_cfg.get("enabled"):
        raise SystemExit("v11 dynamics 要求 build_manifest.json 含 enabled=true 的 ligand_bulk_safety")
    _assert_single_ligand_bulk_safety_force(system)
    coion_safety_cfg = manifest.get("coion_bulk_safety", {})
    if int(manifest.get("protocol_version", 0)) >= PROTOCOL_VERSION and not coion_safety_cfg.get("enabled"):
        raise SystemExit("v11 dynamics 要求 build_manifest.json 含 enabled=true 的 coion_bulk_safety")
    if coion_safety_cfg.get("enabled"):
        _assert_single_coion_bulk_safety_force(system)
    p31_indices = _p31_indices(topology)
    signed_target_fraction = float(bulk_cfg["signed_target_fraction"])
    bulk_kz = float(bulk_cfg["k_z_kJ_mol_nm2"])
    bulk_rz = float(bulk_cfg["r_z_nm"])
    ligand_safety_kz = float(ligand_safety_cfg["k_z_kJ_mol_nm2"])
    ligand_safety_rz = float(ligand_safety_cfg["r_z_nm"])
    coion_safety_kz = float(coion_safety_cfg.get("k_z_kJ_mol_nm2", COION_BULK_SAFETY_KZ_KJ_PER_MOL_NM2))
    coion_safety_rz = float(coion_safety_cfg.get("r_z_nm", COION_BULK_SAFETY_RZ_NM))

    lambdas_coul = [float(x) for x in args.lambda_coul.split(",")]
    if abs(lambdas_coul[0] - 1.0) > 1.0e-9 or abs(lambdas_coul[-1] - 0.0) > 1.0e-9:
        print("⚠️  λ 表两端不是 1.0/0.0——不阻断，但请确认这是故意的。")
    if int(args.n_steps_sample) % int(args.save_interval_steps) != 0:
        raise SystemExit(
            f"--n-steps-sample={args.n_steps_sample} 必须能被 "
            f"--save-interval-steps={args.save_interval_steps} 整除"
        )
    if int(args.n_steps_sample_lambda0) % int(args.save_interval_steps) != 0:
        raise SystemExit(
            f"--n-steps-sample-lambda0={args.n_steps_sample_lambda0} 必须能被 "
            f"--save-interval-steps={args.save_interval_steps} 整除"
        )

    _assert_single_restraint_force(system, n_expected=0)  # 原始 system.xml 不该带 restraint
    # v2 修复：只调用一次配置函数，它内部会注入 restraint，不再手动调用
    # `_inject_co_alchemical_ion_restraints`。
    info = engine.configure_pme_ligand_charge_offsets(
        system, ligand_indices, lambda_name="lambda_coul", allow_charged_ligand=True,
        topology=topology, positions=positions_nm * unit.nanometer,
        box_vectors=build_box_vectors, co_alchemical_ion_spec=spec,
    )
    _assert_single_restraint_force(system, n_expected=len(spec["ions"]))
    print(f"✅ prepared system: mode={info['mode']}, n_offsets={info['n_offsets']}, restraint 唯一")
    _assert_single_bulk_restraint_force(system)
    print(f"✅ v10 pair bulk-water restraint 唯一（kZ={bulk_kz:.1f}, rZ={bulk_rz:.3f} nm）")
    print(f"✅ v10 ligand safety wall 唯一（kZ={ligand_safety_kz:.1f}, rZ={ligand_safety_rz:.3f} nm）")
    if coion_safety_cfg.get("enabled"):
        print(f"✅ v11 co-ion safety wall 唯一（kZ={coion_safety_kz:.1f}, rZ={coion_safety_rz:.3f} nm）")

    os.makedirs(output_dir, exist_ok=True)
    prepared_xml_path = os.path.join(output_dir, "system_prepared.xml")
    with open(prepared_xml_path, "w") as fh:
        fh.write(XmlSerializer.serialize(core.ensure_owned_system(system)))
    print("   （system_prepared.xml 只作审计存档；ukn 不会读它，见模块 docstring §「与 C1 的架构差异」上方 v2 说明）")

    nb_force = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    charge_by_lambda = engine.charging_charge_conservation_report(
        nb_force, "lambda_coul", ligand_indices=ligand_indices,
        co_ion_indices=ion_indices, ligand_net_charge_e=q_l, lambdas=lambdas_coul,
    )
    for lam_key, total in charge_by_lambda["total_charge_by_lambda_e"].items():
        if abs(total) > core.TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:
            raise SystemExit(f"λ_coul={lam_key} 全盒总电荷 = {total:+.9f} e，应为 0——不开始跑 GPU")

    def _new_integrator() -> openmm.LangevinMiddleIntegrator:
        integ = openmm.LangevinMiddleIntegrator(
            args.temperature_kelvin * unit.kelvin, args.friction_per_ps / unit.picosecond,
            args.timestep_ps * unit.picosecond,
        )
        integ.setRandomNumberSeed(int(args.seed))
        return integ

    platform, properties, platform_name, fallback_reason = _get_platform(
        args.platform, args.allow_cpu_fallback, args.precision
    )
    try:
        simulation = app.Simulation(topology, system, _new_integrator(), platform, properties)
    except Exception as exc:  # noqa: BLE001
        if not args.allow_cpu_fallback:
            raise RuntimeError(
                f"平台 {args.platform} 在 Context 建立阶段不可用（{exc}），且未传 "
                "--allow-cpu-fallback——拒绝静默回退 CPU。"
            ) from exc
        print(f"⚠️  平台 {args.platform} 在 Context 建立阶段不可用（{exc}），已显式允许回退 CPU")
        platform = openmm.Platform.getPlatformByName("CPU")
        properties = {}
        platform_name = "CPU"
        fallback_reason = str(exc)
        simulation = app.Simulation(topology, system, _new_integrator(), platform, properties)
    simulation.context.setPositions(simulation_positions_nm * unit.nanometer)
    simulation.context.setPeriodicBoxVectors(*simulation_box_vectors)
    initial_bulk_diag = _update_bulk_restraint_target(
        simulation.context, simulation_positions_nm, simulation_box_nm, p31_indices,
        ligand_indices[0], ion_indices[0], signed_target_fraction,
    )

    if restart_dcd:
        simulation.context.setVelocitiesToTemperature(
            args.temperature_kelvin * unit.kelvin, int(args.seed) + 1,
        )
        print("⚙️  restart continuation：保留末帧坐标，跳过重新最小化；重新生成速度用于确认段")
    else:
        print(f"⚙️  最小化（maxIterations={args.n_steps_minimize}）...")
        simulation.minimizeEnergy(maxIterations=int(args.n_steps_minimize))
    state0 = simulation.context.getState(getEnergy=True)
    pe0 = state0.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    if not np.isfinite(pe0):
        raise RuntimeError(f"最小化后势能非有限: {pe0}")
    print(f"   最小化后 PE = {pe0:.3f} kJ/mol")

    def _step_with_dynamic_bulk_target(n_steps: int) -> None:
        """Advance in short chunks so the P31 midplane target is dynamic."""
        remaining = int(n_steps)
        interval = max(1, int(args.save_interval_steps))
        while remaining > 0:
            state = simulation.context.getState(getPositions=True)
            pos_now = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            box_now = np.asarray(
                [v.value_in_unit(unit.nanometer) for v in state.getPeriodicBoxVectors()],
                dtype=np.float64,
            )
            _update_bulk_restraint_target(
                simulation.context, pos_now, box_now, p31_indices,
                ligand_indices[0], ion_indices[0], signed_target_fraction,
            )
            n = min(remaining, interval)
            simulation.step(n)
            remaining -= n

    dynamics_dir = os.path.join(output_dir, "dynamics")
    os.makedirs(dynamics_dir, exist_ok=True)
    csv_path = os.path.join(dynamics_dir, "timeseries.csv")
    csv_fields = [
        "lambda_state_index", "lambda_coul", "step", "time_ps",
        "total_charge_e", "ligand_charge_e", "coion_charge_e",
        "potential_kJ_mol", "max_force_kJ_mol_nm",
        "ligand_coion_distance_nm", "coion_water_coordination",
        "restraint_energy_kJ_mol", "bulk_restraint_energy_kJ_mol",
        "ligand_safety_restraint_energy_kJ_mol", "ligand_safety_wall_hit",
        "coion_safety_restraint_energy_kJ_mol", "coion_safety_wall_hit", "coion_safety_target_z_nm",
        "bulk_restraint_target_z_nm", "bulk_pair_center_z_nm", "bulk_midplane_z_nm",
        "bulk_pair_center_signed_delta_z_nm", "bulk_pair_center_target_displacement_nm",
        "bulk_restraint_wall_hit", "ligand_abs_dz_from_midplane_nm", "coion_abs_dz_from_midplane_nm",
        "ligand_nearest_phosphorus_nm", "coion_nearest_phosphorus_nm",
        "box_x_nm", "box_y_nm", "box_z_nm", "box_volume_nm3",
    ]
    water_o = np.asarray(
        [
            atom.index for atom in topology.atoms()
            if atom.residue.name.strip().upper() in core.WATER_MOLECULE_NAMES
            and str(atom.name).strip().upper() == "O"
        ],
        dtype=int,
    )
    restraint_group = {core.CO_ALCHEMICAL_ION_RESTRAINT_FORCE_GROUP}
    bulk_restraint_group = {BULK_RESTRAINT_FORCE_GROUP}
    ligand_safety_group = {LIGAND_BULK_RESTRAINT_FORCE_GROUP}
    coion_safety_group = {COION_BULK_SAFETY_FORCE_GROUP}
    dcd_paths: List[str] = []
    wall_clock_by_state: List[float] = []
    topology.setPeriodicBoxVectors(simulation_box_vectors)

    with open(csv_path, "w") as csv_fh:
        csv_fh.write(",".join(csv_fields) + "\n")
        for state_idx, lam in enumerate(lambdas_coul):
            t_state_start = time.time()
            simulation.context.setParameter("lambda_coul", float(lam))
            key = f"{float(lam):.6g}"
            total_q = charge_by_lambda["total_charge_by_lambda_e"][key]
            lig_q = charge_by_lambda["ligand_charge_by_lambda_e"][key]
            coion_q = charge_by_lambda["co_ion_charge_by_lambda_e"][key]

            print(f"— λ_coul={lam:.2f}（态 {state_idx + 1}/{len(lambdas_coul)}）：平衡 {args.n_steps_equil} 步...")
            _step_with_dynamic_bulk_target(int(args.n_steps_equil))

            dcd_path = os.path.join(dynamics_dir, f"traj_state{state_idx:02d}_lam{lam:.2f}.dcd")
            dcd_paths.append(dcd_path)
            n_steps_sample_state = (
                int(args.n_steps_sample_lambda0) if abs(float(lam)) <= 1.0e-12
                else int(args.n_steps_sample)
            )
            n_chunks = n_steps_sample_state // int(args.save_interval_steps)
            with open(dcd_path, "wb") as dcd_fh:
                dcd = app.DCDFile(
                    dcd_fh, topology, dt=args.timestep_ps * unit.picosecond,
                    interval=int(args.save_interval_steps),
                )
                for chunk in range(n_chunks):
                    _step_with_dynamic_bulk_target(int(args.save_interval_steps))
                    state = simulation.context.getState(getPositions=True)
                    pos_quantity = state.getPositions(asNumpy=True)
                    pos_nm_frame = pos_quantity.value_in_unit(unit.nanometer)
                    box_frame_vecs = state.getPeriodicBoxVectors()
                    box_frame_nm = np.array([v.value_in_unit(unit.nanometer) for v in box_frame_vecs], dtype=np.float64)
                    bulk_diag = _update_bulk_restraint_target(
                        simulation.context, pos_nm_frame, box_frame_nm, p31_indices,
                        ligand_indices[0], ion_indices[0], signed_target_fraction,
                    )
                    state = simulation.context.getState(getPositions=True, getEnergy=True, getForces=True)
                    pos_quantity = state.getPositions(asNumpy=True)
                    pos_nm_frame = pos_quantity.value_in_unit(unit.nanometer)
                    pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
                    max_force = float(np.max(np.linalg.norm(forces, axis=1)))
                    if not np.isfinite(pe) or not np.isfinite(max_force):
                        raise RuntimeError(
                            f"λ_coul={lam}, 态内第 {chunk} 个保存点：能量/力出现非有限值 (PE={pe}, max|F|={max_force})"
                        )
                    restraint_state = simulation.context.getState(getEnergy=True, groups=restraint_group)
                    restraint_e = restraint_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                    bulk_state = simulation.context.getState(getEnergy=True, groups=bulk_restraint_group)
                    bulk_e = bulk_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                    if not np.isfinite(bulk_e):
                        raise RuntimeError(f"λ_coul={lam}: bulk restraint energy 非有限: {bulk_e}")
                    ligand_safety_state = simulation.context.getState(getEnergy=True, groups=ligand_safety_group)
                    ligand_safety_e = ligand_safety_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                    if not np.isfinite(ligand_safety_e):
                        raise RuntimeError(f"λ_coul={lam}: ligand safety restraint energy 非有限: {ligand_safety_e}")
                    coion_safety_e = 0.0
                    coion_safety_wall_hit = 0
                    if coion_safety_cfg.get("enabled"):
                        coion_safety_state = simulation.context.getState(getEnergy=True, groups=coion_safety_group)
                        coion_safety_e = coion_safety_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                        if not np.isfinite(coion_safety_e):
                            raise RuntimeError(f"λ_coul={lam}: co-ion safety restraint energy 非有限: {coion_safety_e}")
                        coion_safety_wall_hit = int(
                            abs(float(_minimum_image_z_delta_nm(
                                pos_nm_frame[ion_indices[0], 2],
                                bulk_diag["target_z_nm"], box_frame_nm[2, 2],
                            ))) > coion_safety_rz
                        )
                    coion_dist = min(
                        _minimum_image_distance_nm(pos_nm_frame[ligand_indices[0]], pos_nm_frame[i], box_frame_nm)
                        for i in ion_indices
                    )
                    # 向量化：`water_o` 在这份 slab 里有 ~3600 个氧原子，逐个原子
                    # 跑一次纯 Python 距离计算，乘上"每 λ 态几十~上百个保存点 ×
                    # 11 个 λ 态"，会在 GPU 采样的 `simulation.step()` 之间垒出
                    # 实打实的墙钟开销——`_minimum_image_distances_nm` 一次算完
                    # 全部候选原子的距离。
                    coord = int(np.sum(
                        _minimum_image_distances_nm(pos_nm_frame[water_o], pos_nm_frame[ion_indices[0]], box_frame_nm)
                        <= core.COION_FIRST_SHELL_WATER_CUTOFF_NM
                    ))
                    p31_positions = pos_nm_frame[np.asarray(p31_indices, dtype=int)]
                    ligand_nearest_p = float(np.min(_minimum_image_distances_nm(
                        p31_positions, pos_nm_frame[ligand_indices[0]], box_frame_nm
                    )))
                    coion_nearest_p = float(np.min(_minimum_image_distances_nm(
                        p31_positions, pos_nm_frame[ion_indices[0]], box_frame_nm
                    )))
                    wall_hit = int(
                        abs(float(bulk_diag["pair_center_target_displacement_nm"])) > bulk_rz
                    )
                    ligand_safety_wall_hit = int(
                        abs(float(_minimum_image_z_delta_nm(
                            pos_nm_frame[ligand_indices[0], 2],
                            bulk_diag["target_z_nm"], box_frame_nm[2, 2],
                        ))) > ligand_safety_rz
                    )
                    step_now = (
                        sum(
                            int(args.n_steps_equil) + (
                                int(args.n_steps_sample_lambda0) if abs(float(prev_lam)) <= 1.0e-12
                                else int(args.n_steps_sample)
                            )
                            for prev_lam in lambdas_coul[:state_idx]
                        )
                        + int(args.n_steps_equil) + (chunk + 1) * int(args.save_interval_steps)
                    )
                    time_ps = step_now * args.timestep_ps
                    dcd.writeModel(pos_quantity, periodicBoxVectors=box_frame_vecs)
                    row = [
                        state_idx, f"{lam:.6f}", step_now, f"{time_ps:.3f}",
                        f"{total_q:.9f}", f"{lig_q:.6f}", f"{coion_q:.6f}",
                        f"{pe:.4f}", f"{max_force:.4f}", f"{coion_dist:.4f}", coord, f"{restraint_e:.4f}",
                        f"{bulk_e:.4f}", f"{ligand_safety_e:.4f}", ligand_safety_wall_hit,
                        f"{coion_safety_e:.4f}", coion_safety_wall_hit, f"{bulk_diag['target_z_nm']:.4f}",
                        f"{bulk_diag['target_z_nm']:.4f}",
                        f"{bulk_diag['pair_center_z_nm']:.4f}", f"{bulk_diag['midplane_z_nm']:.4f}",
                        f"{bulk_diag['pair_center_signed_delta_z_nm']:.4f}",
                        f"{bulk_diag['pair_center_target_displacement_nm']:.4f}", wall_hit,
                        f"{abs(_minimum_image_z_delta_nm(pos_nm_frame[ligand_indices[0],2], bulk_diag['midplane_z_nm'], box_frame_nm[2,2])):.4f}",
                        f"{abs(_minimum_image_z_delta_nm(pos_nm_frame[ion_indices[0],2], bulk_diag['midplane_z_nm'], box_frame_nm[2,2])):.4f}",
                        f"{ligand_nearest_p:.4f}", f"{coion_nearest_p:.4f}",
                        f"{box_frame_nm[0,0]:.4f}", f"{box_frame_nm[1,1]:.4f}", f"{box_frame_nm[2,2]:.4f}",
                        f"{_box_volume_nm3(box_frame_nm):.4f}",
                    ]
                    csv_fh.write(",".join(str(v) for v in row) + "\n")
            csv_fh.flush()
            dt_state = time.time() - t_state_start
            wall_clock_by_state.append(dt_state)
            print(f"   ✅ 态 {state_idx + 1} 采样完成（{n_chunks} 帧），耗时 {dt_state / 60.0:.1f} min")

    lambda_schedule_type = (
        "pilot_5" if set(round(x, 6) for x in lambdas_coul) == {1.0, 0.5, 0.2, 0.1, 0.0}
        else "full_11" if len(lambdas_coul) == 11 else "custom"
    )
    n_steps_sample_by_state = [
        int(args.n_steps_sample_lambda0) if abs(float(lam)) <= 1.0e-12 else int(args.n_steps_sample)
        for lam in lambdas_coul
    ]
    expected_frames_by_state = [n // int(args.save_interval_steps) for n in n_steps_sample_by_state]
    dyn_manifest = {
        "protocol_version": PROTOCOL_VERSION, "case": manifest["case"], "lambdas_coul": lambdas_coul,
        "lambda_schedule_type": lambda_schedule_type,
        "platform_requested": args.platform, "platform_used": platform_name, "platform_fallback_reason": fallback_reason,
        "seed": int(args.seed), "temperature_kelvin": args.temperature_kelvin,
        "timestep_ps": args.timestep_ps, "friction_per_ps": args.friction_per_ps,
        "n_steps_minimize": int(args.n_steps_minimize), "n_steps_equil_per_state": int(args.n_steps_equil),
        "n_steps_sample_per_state": int(args.n_steps_sample),
        "n_steps_sample_lambda0": int(args.n_steps_sample_lambda0),
        "n_steps_sample_per_state_by_state": n_steps_sample_by_state,
        "save_interval_steps": int(args.save_interval_steps),
        "expected_frames_per_state": expected_frames_by_state[0] if len(set(expected_frames_by_state)) == 1 else None,
        "expected_frames_per_state_by_state": expected_frames_by_state,
        "post_minimization_potential_kJ_mol": float(pe0), "dcd_paths": dcd_paths, "timeseries_csv": csv_path,
        "bulk_water_restraint": {
            "enabled": True, "form": BULK_RESTRAINT_FORM,
            "force_group": BULK_RESTRAINT_FORCE_GROUP,
            "k_z_kJ_mol_nm2": bulk_kz, "r_z_nm": bulk_rz,
            "initial_signed_delta_z_nm": float(bulk_cfg.get("initial_signed_delta_z_nm", 0.0)),
            "signed_target_fraction": signed_target_fraction,
            "initial_target_signed_delta_z_nm": float(bulk_cfg.get("initial_target_signed_delta_z_nm", 0.0)),
            "target_offset_toward_water_nm": float(bulk_cfg.get("target_offset_toward_water_nm", 0.0)),
            "relative_z_fluctuation_limit_nm": float(bulk_cfg.get("relative_z_fluctuation_limit_nm", 0.0)),
            "membrane_undulation_margin_nm": float(bulk_cfg.get("membrane_undulation_margin_nm", 0.0)),
            "target_update_interval_steps": int(args.save_interval_steps),
            "dynamic_reference": "P31_midplane",
            "initial_target": initial_bulk_diag,
        },
        "ligand_bulk_safety": {
            "enabled": True,
            "form": LIGAND_BULK_RESTRAINT_FORM,
            "force_group": LIGAND_BULK_RESTRAINT_FORCE_GROUP,
            "reference_index": int(ligand_safety_cfg["reference_index"]),
            "k_z_kJ_mol_nm2": ligand_safety_kz,
            "r_z_nm": ligand_safety_rz,
            "target_update_interval_steps": int(args.save_interval_steps),
        },
        "coion_bulk_safety": {
            "enabled": bool(coion_safety_cfg.get("enabled")),
            "form": coion_safety_cfg.get("form", COION_BULK_SAFETY_FORM),
            "force_group": COION_BULK_SAFETY_FORCE_GROUP,
            "reference_index": int(coion_safety_cfg.get("reference_index", ion_indices[0])),
            "k_z_kJ_mol_nm2": coion_safety_kz,
            "r_z_nm": coion_safety_rz,
            "target_update_interval_steps": int(args.save_interval_steps),
            "dynamic_reference": "P31_midplane_same_signed_fraction_as_pair_target",
            "fingerprint_scope": "member_level_coion_bulk_safety",
        },
        "system_prepared_xml": prepared_xml_path, "system_prepared_xml_sha256": _sha256_file(prepared_xml_path),
        "wall_clock_seconds_by_state": wall_clock_by_state, "wall_clock_seconds_total": float(sum(wall_clock_by_state)),
        "restart_source": restart_info,
    }
    dyn_manifest_path = os.path.join(output_dir, "dynamics_manifest.json")
    with open(dyn_manifest_path, "w", encoding="utf-8") as fh:
        json.dump(dyn_manifest, fh, indent=2)
    print(f"✅ dynamics 完成，manifest: {dyn_manifest_path}")


# ============================================================================
# 子命令：ukn（CPU，v2 修复：从**原始** system.xml 唯一配置 Hamiltonian）
# ============================================================================


def cmd_ukn(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    build_artifacts_dir = getattr(args, "build_artifacts_dir", None) or output_dir
    raw_system, topology, _pos, _box, ligand_indices, spec, manifest = _load_build_artifacts(build_artifacts_dir)
    dyn_manifest_path = os.path.join(output_dir, "dynamics_manifest.json")
    if not os.path.exists(dyn_manifest_path):
        raise SystemExit(f"缺少 {dyn_manifest_path}——请先跑 dynamics 子命令")
    with open(dyn_manifest_path) as fh:
        dyn_manifest = json.load(fh)
    _require_protocol_version(dyn_manifest, dyn_manifest_path)

    # v2 修复：不读 dynamics 产出的 system_prepared.xml（那份已经配置过 charge
    # offset + restraint）。`compute_u_kn` 内部会自己调用
    # `_prepare_pme_coulomb_leg_system → configure_pme_ligand_charge_offsets`
    # 唯一配置一次；如果这里再喂一份已经配置过的 system，就是配置两次
    # （v1 的第二个 bug）。先只读校验身份没有漂，再把**原始** system 交给它。
    core.verify_co_alchemical_ion_identity(
        spec, system=raw_system, topology=topology,
        charge_treatment=core.CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
        ligand_net_charge_e=int(manifest["ligand_net_charge_e"]),
        context="validate_charge_transfer_lipid_slab.ukn",
    )
    _assert_single_restraint_force(raw_system, n_expected=0)
    if manifest.get("bulk_water_restraint", {}).get("enabled"):
        _assert_single_bulk_restraint_force(raw_system)
    if manifest.get("ligand_bulk_safety", {}).get("enabled"):
        _assert_single_ligand_bulk_safety_force(raw_system)
    if manifest.get("coion_bulk_safety", {}).get("enabled"):
        _assert_single_coion_bulk_safety_force(raw_system)

    lambdas_coul = dyn_manifest["lambdas_coul"]
    lambdas_vdw = [1.0] * len(lambdas_coul)  # 只验 charging；vdW 全程保持耦合，同 C1
    dcd_paths = dyn_manifest["dcd_paths"]
    if len(dcd_paths) != len(lambdas_coul):
        raise SystemExit(f"dcd_paths 数量 {len(dcd_paths)} 与 lambdas_coul 数量 {len(lambdas_coul)} 不一致")
    for path in dcd_paths:
        if not os.path.exists(path):
            raise SystemExit(f"轨迹文件不存在: {path}")

    analyzer = engine.TraditionalMBARAnalyzer(temperature=dyn_manifest["temperature_kelvin"])
    u_kn = analyzer.compute_u_kn(
        traj_files=dcd_paths, system_template=raw_system, ligand_indices=ligand_indices,
        lambdas_coul=lambdas_coul, lambdas_vdw=lambdas_vdw, platform_name=args.platform,
        topology=topology, co_alchemical_ion_spec=spec,
    )
    n_k = np.asarray(analyzer._last_n_k, dtype=int)
    if not np.all(np.isfinite(u_kn)):
        raise RuntimeError("u_kn 含 NaN/Inf")

    result = analyzer.solve(u_kn, decorrelate=True)
    dg_kj = float(result["delta_G"])
    err_kj = float(result["error"])
    dg_kcal, err_kcal = dg_kj / 4.184, err_kj / 4.184
    print(
        f"✅ charging ΔG(λ_coul: 1→0) = {dg_kj:.4f} ± {err_kj:.4f} kJ/mol "
        f"= {dg_kcal:.4f} ± {err_kcal:.4f} kcal/mol "
        f"(method={result.get('method')}, converged={result.get('converged')}, min_overlap={result.get('min_overlap')})"
    )

    u_kn_path = os.path.join(output_dir, "u_kn.npz")
    np.savez(
        u_kn_path, u_kn=u_kn, n_k=n_k,
        lambdas_coul=np.asarray(lambdas_coul, dtype=float), lambdas_vdw=np.asarray(lambdas_vdw, dtype=float),
        temperature_kelvin=dyn_manifest["temperature_kelvin"], beta=analyzer.beta,
        coion_fingerprint=spec.get("fingerprint", ""), system_sha256=manifest["system_xml_sha256"],
    )
    dg_result = {
        "protocol_version": PROTOCOL_VERSION, "case": manifest["case"],
        "delta_G_charging_kJ_mol": dg_kj, "uncertainty_kJ_mol": err_kj,
        "delta_G_charging_kcal_mol": dg_kcal, "uncertainty_kcal_mol": err_kcal,
        "method": result.get("method"), "converged": result.get("converged"),
        "min_overlap": result.get("min_overlap"), "n_states": result.get("n_states"),
        "n_frames": result.get("n_frames"), "u_kn_path": u_kn_path,
        "hamiltonian_source": "raw_build_system_xml_configured_once_inside_compute_u_kn",
    }
    with open(os.path.join(output_dir, "charging_delta_G.json"), "w", encoding="utf-8") as fh:
        json.dump(dg_result, fh, indent=2, cls=core.NumpyEncoder)
    print(f"✅ ukn 完成: {u_kn_path}")


# ============================================================================
# 子命令：slab-quality-gate（case 专属；全部 11 个 λ；纯 CPU）
# ============================================================================


def cmd_slab_quality_gate(args: argparse.Namespace) -> None:
    import mdtraj as md
    import csv as csv_module

    output_dir = args.output_dir
    _system, topology, _pos, _box, ligand_indices, spec, manifest = _load_build_artifacts(output_dir)
    dyn_manifest_path = os.path.join(output_dir, "dynamics_manifest.json")
    with open(dyn_manifest_path) as fh:
        dyn_manifest = json.load(fh)
    _require_protocol_version(dyn_manifest, dyn_manifest_path)

    coion_index = int(spec["ions"][0]["atom_index"])
    ligand_index = int(ligand_indices[0])
    lambdas_coul = dyn_manifest["lambdas_coul"]
    dcd_paths = dyn_manifest["dcd_paths"]
    expected_frames = int(dyn_manifest["expected_frames_per_state"])
    frame_interval_ps = dyn_manifest["save_interval_steps"] * dyn_manifest["timestep_ps"]

    checks: Dict[str, bool] = {}
    reasons: List[str] = []

    # ---- 1. 11 个 λ 齐全，每个 λ 的 DCD 都存在且帧数达标 ----
    checks["all_lambda_present"] = len(dcd_paths) == len(lambdas_coul) == 11
    if not checks["all_lambda_present"]:
        reasons.append(f"λ 数量不是 11：dcd_paths={len(dcd_paths)}, lambdas_coul={len(lambdas_coul)}")

    frame_counts_ok = True
    for path in dcd_paths:
        if not os.path.exists(path):
            frame_counts_ok = False
            reasons.append(f"缺少轨迹文件 {path}")
            continue
        traj = md.load_dcd(path, top=md.Topology.from_openmm(topology))
        if traj.n_frames != expected_frames:
            frame_counts_ok = False
            reasons.append(f"{path} 帧数={traj.n_frames}，应为 {expected_frames}")
    checks["frame_counts_match_expected"] = frame_counts_ok

    # ---- 2. timeseries.csv：总电荷=0、能量/力/restraint 有限、co-ion↔ligand 距离 ----
    csv_path = dyn_manifest["timeseries_csv"]
    if not os.path.exists(csv_path):
        raise SystemExit(f"缺少 {csv_path}")
    rows_by_state: Dict[int, List[Dict[str, str]]] = {}
    with open(csv_path, newline="") as fh:
        for row in csv_module.DictReader(fh):
            rows_by_state.setdefault(int(row["lambda_state_index"]), []).append(row)

    total_charge_ok, finite_ok, restraint_bounded_ok, distance_ok = True, True, True, True
    restraint_ceiling_kj_mol = 500.0
    for state_idx, rows in rows_by_state.items():
        for row in rows:
            if abs(float(row["total_charge_e"])) > core.TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:
                total_charge_ok = False
                reasons.append(f"state {state_idx}: total_charge_e={row['total_charge_e']} 不为 0")
            if not (np.isfinite(float(row["potential_kJ_mol"])) and np.isfinite(float(row["max_force_kJ_mol_nm"]))
                    and np.isfinite(float(row["restraint_energy_kJ_mol"]))):
                finite_ok = False
                reasons.append(f"state {state_idx}: 能量/力/restraint 非有限")
            if float(row["restraint_energy_kJ_mol"]) > restraint_ceiling_kj_mol:
                restraint_bounded_ok = False
                reasons.append(
                    f"state {state_idx}: restraint 能量 {row['restraint_energy_kJ_mol']} kJ/mol "
                    f"超过 {restraint_ceiling_kj_mol} kJ/mol 的 sanity ceiling（可能 runaway）"
                )
            if float(row["ligand_coion_distance_nm"]) < core.COION_LIGAND_MIN_IMAGE_RUNTIME_NM:
                distance_ok = False
                reasons.append(
                    f"state {state_idx}: ligand_coion_distance_nm={row['ligand_coion_distance_nm']} "
                    f"< {core.COION_LIGAND_MIN_IMAGE_RUNTIME_NM} nm"
                )
    checks["total_charge_zero_every_frame"] = total_charge_ok
    checks["energy_force_restraint_finite_every_frame"] = finite_ok
    checks["restraint_not_runaway"] = restraint_bounded_ok
    checks["ligand_coion_distance_ge_1p2nm"] = distance_ok

    # ---- 3. 逐 DCD 的几何观测量：co-ion 不换侧、|Δz|≥3.0nm、最近 P31≥1.0nm、
    #      水配位、周期镜像/water gap ----
    ion_element = spec["ions"][0].get("element", "").upper()
    min_water_coordination = core.COION_FIRST_SHELL_MIN_WATER_COUNT.get(ion_element)

    p31 = _p31_indices(topology)
    coion_signs: List[int] = []
    coion_min_abs_dz = np.inf
    coion_min_nearest_p = np.inf
    coion_min_water_coord = np.inf
    min_water_gap_nm = np.inf
    for path in dcd_paths:
        if not os.path.exists(path):
            continue
        traj = md.load_dcd(path, top=md.Topology.from_openmm(topology))
        times_ns = np.arange(traj.n_frames, dtype=float) * frame_interval_ps / 1000.0
        p31_z = traj.xyz[:, p31, 2]
        midplane_per_frame = p31_z.mean(axis=1)
        coion_z = traj.xyz[:, coion_index, 2]
        signs = np.sign(coion_z - midplane_per_frame)
        coion_signs.extend(signs.tolist())

        # `_coion_observables_from_trajectory` 只接受一个标量 midplane（对整条
        # DCD 取一次），而上面的换侧检查用的是**逐帧** `midplane_per_frame`——
        # 传标量进去会让"co-ion 是否曾经贴近中面"用一个和换侧检查不一致的参考系
        # 判，膜中面在这条 DCD 内真的偏移时会漏判瞬时违规。这里不用它返回的
        # `coion_abs_z_from_midplane_nm` 字段，自己按逐帧 midplane 算；
        # 最近磷原子距离/水配位与 midplane 无关，仍直接复用它的结果。
        coion_obs = core._coion_observables_from_trajectory(
            traj, coion_index, 2, float(midplane_per_frame.mean()), times_ns,
            np.asarray([ligand_index], dtype=int), composition=None,
        )
        abs_dz_vals = np.abs(coion_z - midplane_per_frame)
        nearest_p_vals = np.asarray(coion_obs["coion_nearest_phosphorus_distance_nm"]["values"])
        coord_vals = np.asarray(coion_obs["coion_first_shell_water_count"]["values"])
        coion_min_abs_dz = min(coion_min_abs_dz, float(np.min(abs_dz_vals)))
        coion_min_nearest_p = min(coion_min_nearest_p, float(np.min(nearest_p_vals)))
        coion_min_water_coord = min(coion_min_water_coord, float(np.min(coord_vals)))

        lengths = np.asarray(traj.unitcell_lengths, dtype=float)
        is_upper0 = p31_z[0] > midplane_per_frame[0]
        thickness_nm = p31_z[:, is_upper0].mean(axis=1) - p31_z[:, ~is_upper0].mean(axis=1)
        water_gap_nm = lengths[:, 2] - thickness_nm
        min_water_gap_nm = min(min_water_gap_nm, float(np.min(water_gap_nm)))

    unique_signs = set(int(s) for s in coion_signs if s != 0)
    checks["coion_never_flips_membrane_side"] = len(unique_signs) <= 1
    if not checks["coion_never_flips_membrane_side"]:
        reasons.append(f"co-ion 相对膜中面的符号出现了 {unique_signs}，应恒为一种")

    checks["coion_stays_above_midplane_threshold"] = coion_min_abs_dz >= core.COION_MEMBRANE_MIDPLANE_MIN_ABS_Z_NM
    if not checks["coion_stays_above_midplane_threshold"]:
        reasons.append(f"co-ion 最小 |Δz| = {coion_min_abs_dz:.3f} nm < {core.COION_MEMBRANE_MIDPLANE_MIN_ABS_Z_NM} nm")

    checks["coion_stays_away_from_phosphorus"] = coion_min_nearest_p >= core.COION_NEAREST_PHOSPHORUS_MIN_NM
    if not checks["coion_stays_away_from_phosphorus"]:
        reasons.append(f"co-ion 最近 P31 距离最小值 = {coion_min_nearest_p:.3f} nm < {core.COION_NEAREST_PHOSPHORUS_MIN_NM} nm")

    # fail closed，不是 fail open：`min_water_coordination is None`（认不出这个
    # 离子的判据）曾经被写成"跳过=通过"，那样任何未来把 `element` 字段改名/
    # 换了个陌生离子符号，这道真实的物理检查会悄悄变成永远通过且不报警。
    if min_water_coordination is None:
        checks["coion_water_coordination_sufficient"] = False
        reasons.append(
            f"co-ion 元素 {ion_element!r} 不在 COION_FIRST_SHELL_MIN_WATER_COUNT "
            f"判据表里（{sorted(core.COION_FIRST_SHELL_MIN_WATER_COUNT)}）——"
            "判不了水配位就不能算过，不是默认放行。"
        )
    else:
        checks["coion_water_coordination_sufficient"] = bool(coion_min_water_coord >= min_water_coordination)
        if not checks["coion_water_coordination_sufficient"]:
            reasons.append(f"co-ion 最小水配位 {coion_min_water_coord} < C1 判据 {min_water_coordination}（离子 {ion_element}）")

    # 周期镜像/water gap：水层厚度必须始终大于 `abfe_core.MEMBRANE_MIN_WATER_SLAB_NM`
    # ——这正是 `membrane_observables_from_trajectory` 里
    # `image_contact_frames = count(water_gap < MEMBRANE_MIN_WATER_SLAB_NM)` 用的
    # 同一个阈值（那段逻辑本身不依赖蛋白，只是被嵌在需要蛋白的大函数里调不到），
    # 复用它而不是另起一个（v1 那版"返回 min(Lx,Ly)"的假实现已删除）。
    water_gap_safety_margin_nm = core.MEMBRANE_MIN_WATER_SLAB_NM
    checks["no_membrane_periodic_image_contact"] = min_water_gap_nm >= water_gap_safety_margin_nm
    if not checks["no_membrane_periodic_image_contact"]:
        reasons.append(
            f"最小 water gap = {min_water_gap_nm:.3f} nm < 安全边距 {water_gap_safety_margin_nm} nm"
            "（膜的周期像可能通过过薄的水层互相看见）"
        )

    passed = all(checks.values())
    result = {
        "protocol_version": PROTOCOL_VERSION, "case": manifest["case"], "checks": checks,
        "failure_reasons": reasons, "passed": passed,
        "coion_min_abs_dz_from_midplane_nm": coion_min_abs_dz,
        "coion_min_nearest_phosphorus_nm": coion_min_nearest_p,
        "coion_min_water_coordination": coion_min_water_coord,
        "min_water_gap_nm": min_water_gap_nm,
    }
    out_path = os.path.join(output_dir, "slab_quality_gate.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, cls=core.NumpyEncoder)
    print(f"{'✅' if passed else '❌'} slab-quality-gate 完成: {out_path}")
    print(json.dumps({"checks": checks, "failure_reasons": reasons}, indent=2, ensure_ascii=False, cls=core.NumpyEncoder))


# v10 replacement for the v7/v8/v9 gate.  The older implementations remain above as
# historical references, but this definition is the command actually exported.
def cmd_slab_quality_gate(args: argparse.Namespace) -> None:
    import csv as csv_module
    import mdtraj as md

    output_dir = args.output_dir
    build_artifacts_dir = getattr(args, "build_artifacts_dir", None) or output_dir
    _system, topology, _pos, _box, ligand_indices, spec, manifest = _load_build_artifacts(
        build_artifacts_dir, allow_legacy=True,
    )
    with open(os.path.join(output_dir, "dynamics_manifest.json"), encoding="utf-8") as fh:
        dyn_manifest = json.load(fh)
    _require_protocol_version(
        dyn_manifest, os.path.join(output_dir, "dynamics_manifest.json"), allow_legacy=True,
    )

    ligand_index = int(ligand_indices[0])
    coion_index = int(spec["ions"][0]["atom_index"])
    lambdas = [float(x) for x in dyn_manifest["lambdas_coul"]]
    dcd_paths = dyn_manifest["dcd_paths"]
    expected_frames_by_state = dyn_manifest.get("expected_frames_per_state_by_state")
    if expected_frames_by_state is None:
        expected_frames_by_state = [int(dyn_manifest["expected_frames_per_state"])] * len(dcd_paths)
    frame_interval_ps = float(dyn_manifest["save_interval_steps"]) * float(dyn_manifest["timestep_ps"])
    bulk_enabled = bool(dyn_manifest.get("bulk_water_restraint", {}).get("enabled"))
    ligand_safety_enabled = bool(dyn_manifest.get("ligand_bulk_safety", {}).get("enabled"))
    coion_safety_enabled = bool(dyn_manifest.get("coion_bulk_safety", {}).get("enabled"))
    current_protocol = int(manifest.get("protocol_version", dyn_manifest.get("protocol_version", 0)))
    checks: Dict[str, bool] = {}
    reasons: List[str] = []

    actual_lambdas = {round(x, 6) for x in lambdas}
    if bulk_enabled:
        schedule_type = str(dyn_manifest.get("lambda_schedule_type", ""))
        if schedule_type == "pilot_5":
            required = {1.0, 0.5, 0.2, 0.1, 0.0}
            required_ok = actual_lambdas == required
            schedule_label = "pilot_5"
        elif schedule_type == "full_11":
            required = {round(x, 6) for x in np.arange(1.0, -0.001, -0.1)}
            required_ok = actual_lambdas == required
            schedule_label = "full_11"
        elif schedule_type == "custom" and int(dyn_manifest.get("protocol_version", 0)) >= 10:
            required = {0.2, 0.1, 0.0}
            required_ok = actual_lambdas == required
            schedule_label = "custom_v10_representative"
        else:
            required = {1.0, 0.5, 0.2, 0.1, 0.0}
            required_ok = actual_lambdas == required
            schedule_label = "legacy_pilot_5"
        checks["required_v10_lambda_set_present"] = required_ok
        checks["required_v11_lambda_set_present"] = required_ok if current_protocol >= PROTOCOL_VERSION else True
        checks["lambda_schedule_type_matches_manifest"] = (
            schedule_type in {"pilot_5", "full_11", "custom"}
            or int(dyn_manifest.get("protocol_version", 0)) < PROTOCOL_VERSION
        )
        if not required_ok:
            reasons.append(
                f"{schedule_label} required λ 不匹配：实际 {sorted(actual_lambdas)}，"
                f"应为 {sorted(required)}"
            )
        checks["required_v8_lambda_set_present"] = required_ok
        checks["required_v9_lambda_set_present"] = required_ok
    else:
        # Existing synthetic fixtures predate v8 and intentionally contain one λ.
        required_ok = len(dcd_paths) == len(lambdas) == 11
        checks["required_v10_lambda_set_present"] = required_ok
        checks["required_v11_lambda_set_present"] = True
        checks["required_v8_lambda_set_present"] = required_ok
        checks["required_v9_lambda_set_present"] = required_ok
    checks["all_lambda_present"] = len(dcd_paths) == len(lambdas)
    if not checks["all_lambda_present"]:
        reasons.append("dcd_paths 与 lambdas_coul 数量不一致")

    frame_counts_ok = True
    md_top = md.Topology.from_openmm(topology)
    trajectories: Dict[int, Any] = {}
    for state_idx, path in enumerate(dcd_paths):
        if not os.path.exists(path):
            frame_counts_ok = False
            reasons.append(f"缺少轨迹文件 {path}")
            continue
        traj = md.load_dcd(path, top=md_top)
        trajectories[state_idx] = traj
        expected = int(expected_frames_by_state[state_idx]) if state_idx < len(expected_frames_by_state) else -1
        if traj.n_frames != expected:
            frame_counts_ok = False
            reasons.append(f"{path} 帧数={traj.n_frames}，应为 {expected}")
    checks["frame_counts_match_expected"] = frame_counts_ok

    csv_path = dyn_manifest["timeseries_csv"]
    if not os.path.exists(csv_path):
        raise SystemExit(f"缺少 {csv_path}")
    rows_by_state: Dict[int, List[Dict[str, str]]] = {}
    with open(csv_path, newline="") as fh:
        for row in csv_module.DictReader(fh):
            rows_by_state.setdefault(int(row["lambda_state_index"]), []).append(row)

    # Optional supplemental segment: keep the original pilot manifest/λ set and
    # append a separately equilibrated confirmation trajectory to matching λ.
    # This preserves the frozen pilot evidence and avoids treating a one-state
    # confirmation directory as a complete pilot schedule.
    supplement_dir = getattr(args, "supplement_dir", None)
    supplement_info: Dict[str, Any] = {"enabled": False}
    supplement_coords_by_lambda: Dict[float, np.ndarray] = {}
    if supplement_dir:
        supplement_manifest_path = os.path.join(supplement_dir, "dynamics_manifest.json")
        if not os.path.exists(supplement_manifest_path):
            raise SystemExit(f"补充采样目录缺少 {supplement_manifest_path}")
        with open(supplement_manifest_path, encoding="utf-8") as fh:
            supplement_manifest = json.load(fh)
        _require_protocol_version(supplement_manifest, supplement_manifest_path, allow_legacy=True)
        supplement_lambdas = [float(x) for x in supplement_manifest["lambdas_coul"]]
        supplement_expected = supplement_manifest.get("expected_frames_per_state_by_state")
        if supplement_expected is None:
            supplement_expected = [int(supplement_manifest["expected_frames_per_state"])] * len(supplement_lambdas)
        supplement_csv_path = supplement_manifest["timeseries_csv"]
        supplement_rows: Dict[int, List[Dict[str, str]]] = {}
        with open(supplement_csv_path, newline="") as fh:
            for row in csv_module.DictReader(fh):
                supplement_rows.setdefault(int(row["lambda_state_index"]), []).append(row)
        supplemented_lambdas: List[float] = []
        for supplement_state, supplement_lam in enumerate(supplement_lambdas):
            matches = [
                base_state for base_state, base_lam in enumerate(lambdas)
                if abs(float(base_lam) - supplement_lam) <= 1.0e-6
            ]
            if len(matches) != 1:
                raise SystemExit(
                    f"补充 λ={supplement_lam:.6f} 在原 pilot 中没有唯一匹配：{matches}"
                )
            base_state = matches[0]
            supplement_dcd = supplement_manifest["dcd_paths"][supplement_state]
            if not os.path.exists(supplement_dcd):
                raise SystemExit(f"缺少补充轨迹 {supplement_dcd}")
            supplement_traj = md.load_dcd(supplement_dcd, top=md_top)
            if base_state not in trajectories:
                raise SystemExit(f"原 pilot 缺少要补充的 state {base_state}")
            trajectories[base_state] = md.join(
                [trajectories[base_state], supplement_traj], check_topology=False,
            )
            expected_frames_by_state[base_state] = (
                int(expected_frames_by_state[base_state]) + int(supplement_expected[supplement_state])
            )
            for row in supplement_rows.get(supplement_state, []):
                merged_row = dict(row)
                merged_row["lambda_state_index"] = str(base_state)
                rows_by_state.setdefault(base_state, []).append(merged_row)
            supplement_coords_by_lambda[round(supplement_lam, 6)] = np.asarray(
                [float(row["coion_water_coordination"]) for row in supplement_rows.get(supplement_state, [])],
                dtype=float,
            )
            supplemented_lambdas.append(supplement_lam)
        for base_state, traj in trajectories.items():
            expected = int(expected_frames_by_state[base_state])
            if traj.n_frames != expected:
                frame_counts_ok = False
                reasons.append(
                    f"合并补充采样后 state {base_state} 帧数={traj.n_frames}，应为 {expected}"
                )
        supplement_info = {
            "enabled": True,
            "directory": os.path.relpath(supplement_dir, _REPO_ROOT),
            "supplemented_lambdas": supplemented_lambdas,
        }
        checks["frame_counts_match_expected"] = frame_counts_ok

    total_charge_ok = True
    finite_ok = True
    restraint_ok = True
    distance_ok = True
    bulk_columns_ok = True
    bulk_energies: List[float] = []
    bulk_wall_hits: List[int] = []
    ligand_safety_energies: List[float] = []
    ligand_safety_wall_hits: List[int] = []
    coion_safety_energies: List[float] = []
    coion_safety_wall_hits: List[int] = []
    for state_idx, rows in rows_by_state.items():
        for row in rows:
            if abs(float(row["total_charge_e"])) > core.TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:
                total_charge_ok = False
                reasons.append(f"state {state_idx}: total charge 非零")
            values = [row["potential_kJ_mol"], row["max_force_kJ_mol_nm"], row["restraint_energy_kJ_mol"]]
            if bulk_enabled:
                required_columns = (
                    "bulk_restraint_energy_kJ_mol", "bulk_pair_center_z_nm", "bulk_midplane_z_nm",
                    "bulk_pair_center_target_displacement_nm", "bulk_restraint_wall_hit",
                    "ligand_abs_dz_from_midplane_nm", "coion_abs_dz_from_midplane_nm",
                    "ligand_nearest_phosphorus_nm", "coion_nearest_phosphorus_nm",
                )
                if ligand_safety_enabled:
                    required_columns += ("ligand_safety_restraint_energy_kJ_mol", "ligand_safety_wall_hit")
                if coion_safety_enabled:
                    required_columns += (
                        "coion_safety_restraint_energy_kJ_mol", "coion_safety_wall_hit",
                        "coion_safety_target_z_nm",
                    )
                if any(name not in row for name in required_columns):
                    bulk_columns_ok = False
                else:
                    values.append(row["bulk_restraint_energy_kJ_mol"])
                    bulk_energies.append(float(row["bulk_restraint_energy_kJ_mol"]))
                    bulk_wall_hits.append(int(float(row["bulk_restraint_wall_hit"])))
                    if ligand_safety_enabled:
                        values.append(row["ligand_safety_restraint_energy_kJ_mol"])
                        ligand_safety_energies.append(float(row["ligand_safety_restraint_energy_kJ_mol"]))
                        ligand_safety_wall_hits.append(int(float(row["ligand_safety_wall_hit"])))
                    if coion_safety_enabled:
                        values.append(row["coion_safety_restraint_energy_kJ_mol"])
                        coion_safety_energies.append(float(row["coion_safety_restraint_energy_kJ_mol"]))
                        coion_safety_wall_hits.append(int(float(row["coion_safety_wall_hit"])))
            if not all(np.isfinite(float(value)) for value in values):
                finite_ok = False
                reasons.append(f"state {state_idx}: energy/force/restraint 非有限")
            if float(row["restraint_energy_kJ_mol"]) > 500.0:
                restraint_ok = False
            if bulk_enabled and "bulk_restraint_energy_kJ_mol" in row and float(row["bulk_restraint_energy_kJ_mol"]) > BULK_RESTRAINT_MAX_ENERGY_KJ_MOL:
                restraint_ok = False
            if ligand_safety_enabled and "ligand_safety_restraint_energy_kJ_mol" in row and float(row["ligand_safety_restraint_energy_kJ_mol"]) > BULK_RESTRAINT_MAX_ENERGY_KJ_MOL:
                restraint_ok = False
            if coion_safety_enabled and "coion_safety_restraint_energy_kJ_mol" in row and float(row["coion_safety_restraint_energy_kJ_mol"]) > BULK_RESTRAINT_MAX_ENERGY_KJ_MOL:
                restraint_ok = False
            if float(row["ligand_coion_distance_nm"]) < core.COION_LIGAND_MIN_IMAGE_RUNTIME_NM:
                distance_ok = False
    checks["total_charge_zero_every_frame"] = total_charge_ok
    checks["energy_force_restraint_finite_every_frame"] = finite_ok
    checks["restraint_not_runaway"] = restraint_ok
    checks["ligand_coion_distance_ge_1p2nm"] = distance_ok
    checks["v9_bulk_restraint_diagnostics_present"] = (not bulk_enabled) or bulk_columns_ok
    checks["v8_bulk_restraint_diagnostics_present"] = checks["v9_bulk_restraint_diagnostics_present"]
    checks["v10_ligand_safety_diagnostics_present"] = (not ligand_safety_enabled) or bulk_columns_ok
    checks["v11_coion_safety_diagnostics_present"] = (
        (current_protocol < PROTOCOL_VERSION)
        or (coion_safety_enabled and bulk_columns_ok)
    )

    p31 = _p31_indices(topology)
    ligand_min_dz = coion_min_dz = np.inf
    ligand_min_p = coion_min_p = np.inf
    coion_min_coord = np.inf
    min_water_gap = np.inf
    pbc_boundary_crossings: List[Dict[str, Any]] = []
    membrane_core_crossings: List[Dict[str, Any]] = []
    legacy_ligand_signs: List[int] = []
    legacy_coion_signs: List[int] = []
    for state_idx, traj in trajectories.items():
        p31_z = traj.xyz[:, p31, 2]
        box_z = np.asarray(traj.unitcell_lengths[:, 2], dtype=float)
        midplane = p31_z.mean(axis=1)
        ligand_z = traj.xyz[:, ligand_index, 2]
        coion_z = traj.xyz[:, coion_index, 2]

        # Use fractional coordinates and continuous unwrapping for the
        # trajectory diagnostic.  The hard decision is the distance to the
        # membrane core, never the sign of a minimum-image delta at ±Lz/2.
        midplane_frac = _continuous_unwrap_fractional(midplane / box_z)
        ligand_frac = _continuous_unwrap_fractional(ligand_z / box_z)
        coion_frac = _continuous_unwrap_fractional(coion_z / box_z)
        pair_center_z = ligand_z + 0.5 * (
            coion_z - ligand_z - np.round((coion_z - ligand_z) / box_z) * box_z
        )
        pair_center_frac = _continuous_unwrap_fractional(pair_center_z / box_z)
        ligand_dz = (ligand_frac - midplane_frac)
        coion_dz = (coion_frac - midplane_frac)
        pair_dz = pair_center_frac - midplane_frac
        ligand_dz -= np.round(ligand_dz)  # fractional minimum image
        coion_dz -= np.round(coion_dz)
        pair_dz -= np.round(pair_dz)
        ligand_dz_nm = ligand_dz * box_z
        coion_dz_nm = coion_dz * box_z
        pair_dz_nm = pair_dz * box_z
        legacy_ligand_dz = ligand_z - midplane
        legacy_coion_dz = coion_z - midplane
        legacy_ligand_dz -= np.round(legacy_ligand_dz / box_z) * box_z
        legacy_coion_dz -= np.round(legacy_coion_dz / box_z) * box_z
        legacy_ligand_signs.extend(np.sign(legacy_ligand_dz).tolist())
        legacy_coion_signs.extend(np.sign(legacy_coion_dz).tolist())
        for label, dz_nm in (("ligand", ligand_dz_nm), ("co-ion", coion_dz_nm)):
            sign = np.sign(dz_nm)
            changes = np.flatnonzero(sign[1:] * sign[:-1] < 0)
            for frame_idx in changes.tolist():
                local_min = min(abs(float(dz_nm[frame_idx])), abs(float(dz_nm[frame_idx + 1])))
                event = {
                    "state_index": int(state_idx),
                    "member": label,
                    "frame_before": int(frame_idx),
                    "frame_after": int(frame_idx + 1),
                    "min_abs_dz_nm": local_min,
                    "classification": "PBC_BOUNDARY_CROSSING" if local_min >= core.COION_MEMBRANE_MIDPLANE_MIN_ABS_Z_NM else "MEMBRANE_CORE_CROSSING",
                }
                if event["classification"] == "PBC_BOUNDARY_CROSSING":
                    pbc_boundary_crossings.append(event)
                else:
                    membrane_core_crossings.append(event)
        ligand_min_dz = min(ligand_min_dz, float(np.min(np.abs(ligand_dz_nm))))
        coion_min_dz = min(coion_min_dz, float(np.min(np.abs(coion_dz_nm))))
        obs = core._coion_observables_from_trajectory(
            traj, coion_index, 2, float(midplane.mean()),
            np.arange(traj.n_frames, dtype=float) * frame_interval_ps / 1000.0,
            np.asarray([ligand_index], dtype=int), composition=None,
        )
        coion_min_coord = min(
            coion_min_coord,
            float(np.min(np.asarray(obs["coion_first_shell_water_count"]["values"]))),
        )
        for frame_idx in range(traj.n_frames):
            box = np.diag(np.asarray(traj.unitcell_lengths[frame_idx], dtype=float))
            p_xyz = traj.xyz[frame_idx, p31, :]
            ligand_min_p = min(ligand_min_p, float(np.min(_minimum_image_distances_nm(p_xyz, traj.xyz[frame_idx, ligand_index], box))))
            coion_min_p = min(coion_min_p, float(np.min(_minimum_image_distances_nm(p_xyz, traj.xyz[frame_idx, coion_index], box))))
        is_upper = p31_z[0] > midplane[0]
        thickness = p31_z[:, is_upper].mean(axis=1) - p31_z[:, ~is_upper].mean(axis=1)
        min_water_gap = min(min_water_gap, float(np.min(traj.unitcell_lengths[:, 2] - thickness)))

    no_membrane_core_crossing = not membrane_core_crossings
    checks["no_membrane_core_crossing"] = no_membrane_core_crossing
    # Compatibility aliases for existing reports/tests.  Real v9 bulk runs use
    # the core-crossing decision; pre-v8 synthetic fixtures retain the legacy
    # sign diagnostic so the old regression remains meaningful.
    if bulk_enabled:
        checks["ligand_never_flips_membrane_side"] = no_membrane_core_crossing
        checks["coion_never_flips_membrane_side"] = no_membrane_core_crossing
    else:
        checks["ligand_never_flips_membrane_side"] = len({int(s) for s in legacy_ligand_signs if s != 0}) <= 1
        checks["coion_never_flips_membrane_side"] = len({int(s) for s in legacy_coion_signs if s != 0}) <= 1
    checks["ligand_stays_above_midplane_threshold"] = ligand_min_dz >= core.COION_MEMBRANE_MIDPLANE_MIN_ABS_Z_NM
    checks["coion_stays_above_midplane_threshold"] = coion_min_dz >= core.COION_MEMBRANE_MIDPLANE_MIN_ABS_Z_NM
    checks["ligand_stays_away_from_phosphorus"] = ligand_min_p >= core.COION_NEAREST_PHOSPHORUS_MIN_NM
    checks["coion_stays_away_from_phosphorus"] = coion_min_p >= core.COION_NEAREST_PHOSPHORUS_MIN_NM
    if not no_membrane_core_crossing:
        reasons.append(f"pair 穿过膜核心：{len(membrane_core_crossings)} 个事件")
    if not checks["ligand_stays_above_midplane_threshold"] or not checks["coion_stays_above_midplane_threshold"]:
        reasons.append(f"pair 成员离膜中面不足 3.0 nm：ligand={ligand_min_dz:.3f}, co-ion={coion_min_dz:.3f}")
    if not checks["ligand_stays_away_from_phosphorus"] or not checks["coion_stays_away_from_phosphorus"]:
        reasons.append(f"pair 成员离 P31 不足 1.0 nm：ligand={ligand_min_p:.3f}, co-ion={coion_min_p:.3f}")

    ion_element = spec["ions"][0].get("element", "").upper()
    min_coord_gate = core.COION_FIRST_SHELL_MIN_WATER_COUNT.get(ion_element)
    q_final = abs(float(manifest.get("ligand_net_charge_e", 0.0)))
    coordination_by_lambda: Dict[str, Dict[str, Any]] = {}
    hydration_ok = min_coord_gate is not None
    v10_gate = int(dyn_manifest.get("protocol_version", 0)) >= PROTOCOL_VERSION
    water_oxygen_indices = _water_oxygen_indices(topology)
    r5_by_state = {
        state_idx: _trajectory_r5_values(traj, coion_index, water_oxygen_indices)
        for state_idx, traj in trajectories.items()
    }
    reference_path = getattr(args, "hydration_reference", HYDRATION_REFERENCE_DEFAULT_PATH)
    reference_by_lambda = _load_coordination_reference(reference_path)
    bootstrap_rng = np.random.default_rng(HYDRATION_BOOTSTRAP_SEED)
    for state_idx, rows in rows_by_state.items():
        if not rows:
            continue
        lam = float(rows[0]["lambda_coul"])
        q_coion = float(np.mean(np.abs([float(r["coion_charge_e"]) for r in rows])))
        charge_fraction = q_coion / q_final if q_final > 0.0 else float("nan")
        coords = np.asarray([float(r["coion_water_coordination"]) for r in rows], dtype=float)
        eligible = bool(np.isfinite(charge_fraction) and charge_fraction >= BULK_RESTRAINT_CHARGE_FRACTION_HARD_GATE)
        frame_fraction = float(np.mean(coords >= BULK_RESTRAINT_COORDINATION_MIN_WATER)) if coords.size else 0.0
        block_occupancy = [
            float(np.mean(block >= BULK_RESTRAINT_COORDINATION_MIN_WATER))
            for block in np.array_split(coords, max(1, int(math.ceil(coords.size / COORDINATION_BLOCK_SIZE_FRAMES_V10))))
            if block.size
        ]
        lambda0_sample_ok = not (v10_gate and abs(lam) <= 1.0e-12 and coords.size < COORDINATION_LAMBDA0_MIN_FRAMES_V10)
        sample_blocks = _block_means(coords, COORDINATION_BLOCK_SIZE_FRAMES_V10)
        sample_mean_ci = _bootstrap_block_mean_ci(
            sample_blocks, bootstrap_rng, HYDRATION_BOOTSTRAP_REPLICATES,
        )
        r5_values = r5_by_state.get(state_idx, np.asarray([], dtype=float))
        if r5_values.size != coords.size:
            r5_values = np.asarray([], dtype=float)
        severe_coordination_runs = _contiguous_true_runs(coords <= HYDRATION_SEVERE_COORDINATION_MAX)
        severe_coordination_frames = int(np.sum(coords <= HYDRATION_SEVERE_COORDINATION_MAX))
        severe_r5_runs = _contiguous_true_runs(r5_values >= HYDRATION_SEVERE_R5_NM)
        severe_joint_runs = _contiguous_true_runs(
            (coords <= HYDRATION_SEVERE_COORDINATION_MAX)
            & (r5_values >= HYDRATION_SEVERE_R5_NM)
        ) if r5_values.size == coords.size else []
        severe_joint_sustained_runs = [
            run for run in severe_joint_runs
            if run[2] >= HYDRATION_SEVERE_COORDINATION_MIN_CONSECUTIVE_FRAMES
        ]
        severe_dehydration_ok = bool(
            r5_values.size == coords.size
            and not severe_joint_sustained_runs
        )

        stability_ok = True
        stability_first_half_mean = None
        stability_second_half_mean = None
        stability_delta = None
        supplement_coords = supplement_coords_by_lambda.get(round(lam, 6))
        if eligible and supplement_dir:
            if supplement_coords is None:
                stability_ok = True  # no supplemental segment was requested for this λ
            else:
                supplement_blocks = _block_means(
                    supplement_coords, COORDINATION_BLOCK_SIZE_FRAMES_V10,
                )
                if supplement_blocks.size < 4:
                    stability_ok = False
                else:
                    midpoint = supplement_blocks.size // 2
                    stability_first_half_mean = float(np.mean(supplement_blocks[:midpoint]))
                    stability_second_half_mean = float(np.mean(supplement_blocks[midpoint:]))
                    stability_delta = stability_second_half_mean - stability_first_half_mean
                    stability_ok = abs(stability_delta) <= HYDRATION_STABILITY_MAX_HALF_MEAN_DELTA
        
        reference_coords = reference_by_lambda.get(round(lam, 6), np.asarray([], dtype=float))
        reference_blocks = _block_means(reference_coords, COORDINATION_BLOCK_SIZE_FRAMES_V10)
        reference_mean = float(np.mean(reference_coords)) if reference_coords.size else float("nan")
        reference_mean_ci = _bootstrap_block_mean_ci(
            reference_blocks, bootstrap_rng, HYDRATION_BOOTSTRAP_REPLICATES,
        )
        difference_ci = _bootstrap_block_mean_difference_ci(
            sample_blocks, reference_blocks, bootstrap_rng, HYDRATION_BOOTSTRAP_REPLICATES,
        )
        mean_ok = bool(coords.size and np.mean(coords) >= BULK_RESTRAINT_COORDINATION_MIN_WATER)
        uncertainty_ok = bool(
            np.isfinite(sample_mean_ci[0])
            and sample_mean_ci[0] >= BULK_RESTRAINT_COORDINATION_MIN_WATER
        )
        reference_comparison_ok = bool(
            np.isfinite(difference_ci[0])
            and difference_ci[0] >= -HYDRATION_REFERENCE_NONINFERIORITY_MARGIN_WATER
        )
        passed_state = (not eligible) or bool(
            lambda0_sample_ok
            and mean_ok
            and uncertainty_ok
            and reference_comparison_ok
            and severe_dehydration_ok
            and stability_ok
        )
        coordination_by_lambda[f"{lam:.6f}"] = {
            "lambda_coul": lam, "charge_fraction": charge_fraction,
            "hard_gate_eligible": eligible, "mean": float(np.mean(coords)),
            "min": float(np.min(coords)), "max": float(np.max(coords)),
            "n_frames": int(coords.size), "frame_fraction_ge_5": frame_fraction,
            "frame_fraction_ge_5_is_diagnostic_only": True,
            "block_size_frames": COORDINATION_BLOCK_SIZE_FRAMES_V10,
            "block_occupancy_fraction_ge_5": block_occupancy,
            "lambda0_min_frames": COORDINATION_LAMBDA0_MIN_FRAMES_V10 if v10_gate else None,
            "block_means": sample_blocks,
            "bootstrap_mean_ci95": sample_mean_ci,
            "mean_gate_passed": mean_ok,
            "bootstrap_lower_bound_ge_5_passed": uncertainty_ok,
            "r5_min_nm": float(np.min(r5_values)) if r5_values.size else None,
            "r5_mean_nm": float(np.mean(r5_values)) if r5_values.size else None,
            "r5_max_nm": float(np.max(r5_values)) if r5_values.size else None,
            "r5_severe_threshold_nm": HYDRATION_SEVERE_R5_NM,
            "r5_frames_at_or_above_severe_threshold": int(np.sum(r5_values >= HYDRATION_SEVERE_R5_NM)) if r5_values.size else None,
            "r5_severe_runs": severe_r5_runs,
            "severe_joint_runs": severe_joint_runs,
            "severe_joint_min_consecutive_frames": HYDRATION_SEVERE_COORDINATION_MIN_CONSECUTIVE_FRAMES,
            "severe_dehydration_rule": "coordination<=3 AND r5>=0.4nm for >=2 consecutive frames",
            "severe_coordination_max": HYDRATION_SEVERE_COORDINATION_MAX,
            "severe_coordination_frames": severe_coordination_frames,
            "severe_coordination_runs": severe_coordination_runs,
            "severe_coordination_min_consecutive_frames": HYDRATION_SEVERE_COORDINATION_MIN_CONSECUTIVE_FRAMES,
            "severe_r5_min_consecutive_frames": HYDRATION_SEVERE_R5_MIN_CONSECUTIVE_FRAMES,
            "severe_dehydration_gate_passed": severe_dehydration_ok,
            "supplement_stability_first_half_mean": stability_first_half_mean,
            "supplement_stability_second_half_mean": stability_second_half_mean,
            "supplement_stability_delta": stability_delta,
            "supplement_stability_max_half_mean_delta": HYDRATION_STABILITY_MAX_HALF_MEAN_DELTA,
            "supplement_stability_gate_passed": stability_ok,
            "reference_path": os.path.relpath(reference_path, _REPO_ROOT) if reference_path else None,
            "reference_lambda_coul": lam if reference_coords.size else None,
            "reference_n_frames": int(reference_coords.size),
            "reference_mean": reference_mean,
            "reference_block_means": reference_blocks,
            "reference_bootstrap_mean_ci95": reference_mean_ci,
            "sample_minus_reference_bootstrap_ci95": difference_ci,
            "reference_comparison_passed": reference_comparison_ok,
            "reference_comparison_rule": "non_inferiority_lower_ci_ge_minus_margin",
            "reference_noninferiority_margin_water": HYDRATION_REFERENCE_NONINFERIORITY_MARGIN_WATER,
            "passed": passed_state,
        }
        if eligible and not passed_state:
            hydration_ok = False
            failed_parts = []
            if not lambda0_sample_ok:
                failed_parts.append("λ=0 样本数不足")
            if not mean_ok:
                failed_parts.append(f"mean={np.mean(coords):.3f}<5")
            if not uncertainty_ok:
                failed_parts.append(f"bootstrap lower={sample_mean_ci[0]:.3f}<5")
            if not reference_comparison_ok:
                failed_parts.append(
                    "相对 C1 参考差值 CI="
                    f"{difference_ci}（下界 < -{HYDRATION_REFERENCE_NONINFERIORITY_MARGIN_WATER:.3f} 水分子）"
                )
            if not severe_dehydration_ok:
                failed_parts.append(
                    f"severe dehydration joint runs={severe_joint_sustained_runs}"
                )
            if not stability_ok:
                failed_parts.append(f"supplement 前后半 block mean 漂移={stability_delta}")
            reasons.append(f"λ={lam:.3f} hydration gate 失败：" + "; ".join(failed_parts))
    checks["coion_water_coordination_sufficient"] = hydration_ok
    checks["coion_hydration_gate_at_charge_fraction_ge_0p9"] = hydration_ok

    checks["no_membrane_periodic_image_contact"] = min_water_gap >= core.MEMBRANE_MIN_WATER_SLAB_NM
    bulk_wall_fraction = float(np.mean(bulk_wall_hits)) if bulk_wall_hits else None
    bulk_energy_mean = float(np.mean(bulk_energies)) if bulk_energies else None
    bulk_energy_max = float(np.max(bulk_energies)) if bulk_energies else None
    if not checks["no_membrane_periodic_image_contact"]:
        reasons.append(f"最小 water gap={min_water_gap:.3f} nm 不足 {core.MEMBRANE_MIN_WATER_SLAB_NM} nm")

    passed = all(checks.values())
    result = {
        "protocol_version": int(manifest.get("protocol_version", PROTOCOL_VERSION)),
        "gate_evaluator_protocol_version": PROTOCOL_VERSION,
        "case": manifest["case"], "checks": checks,
        "failure_reasons": reasons, "passed": passed,
        "hydration_supplement": supplement_info,
        "bulk_restraint_enabled": bulk_enabled,
        "bulk_restraint_wall_hit_fraction": bulk_wall_fraction,
        "bulk_restraint_energy_mean_kJ_mol": bulk_energy_mean,
        "bulk_restraint_energy_max_kJ_mol": bulk_energy_max,
        "ligand_safety_restraint_enabled": ligand_safety_enabled,
        "ligand_safety_restraint_wall_hit_fraction": float(np.mean(ligand_safety_wall_hits)) if ligand_safety_wall_hits else None,
        "ligand_safety_restraint_energy_mean_kJ_mol": float(np.mean(ligand_safety_energies)) if ligand_safety_energies else None,
        "ligand_safety_restraint_energy_max_kJ_mol": float(np.max(ligand_safety_energies)) if ligand_safety_energies else None,
        "coion_safety_restraint_enabled": coion_safety_enabled,
        "coion_safety_restraint_wall_hit_fraction": float(np.mean(coion_safety_wall_hits)) if coion_safety_wall_hits else None,
        "coion_safety_restraint_energy_mean_kJ_mol": float(np.mean(coion_safety_energies)) if coion_safety_energies else None,
        "coion_safety_restraint_energy_max_kJ_mol": float(np.max(coion_safety_energies)) if coion_safety_energies else None,
        "ligand_min_abs_dz_from_midplane_nm": ligand_min_dz,
        "coion_min_abs_dz_from_midplane_nm": coion_min_dz,
        "ligand_min_nearest_phosphorus_nm": ligand_min_p,
        "coion_min_nearest_phosphorus_nm": coion_min_p,
        "coion_min_water_coordination": coion_min_coord,
        "pbc_boundary_crossings": pbc_boundary_crossings,
        "membrane_core_crossings": membrane_core_crossings,
        "pbc_boundary_crossing_count": len(pbc_boundary_crossings),
        "membrane_core_crossing_count": len(membrane_core_crossings),
        "coion_coordination_by_lambda": coordination_by_lambda,
        "coordination_charge_fraction_hard_gate": BULK_RESTRAINT_CHARGE_FRACTION_HARD_GATE,
        "coordination_min_mean": BULK_RESTRAINT_COORDINATION_MIN_WATER,
        "coordination_min_frame_fraction": BULK_RESTRAINT_COORDINATION_FRAME_FRACTION,
        "coordination_frame_fraction_gate": "diagnostic_only",
        "hydration_gate_statistical_version": HYDRATION_GATE_STATISTICAL_VERSION,
        "hydration_bootstrap_replicates": HYDRATION_BOOTSTRAP_REPLICATES,
        "hydration_bootstrap_seed": HYDRATION_BOOTSTRAP_SEED,
        "hydration_block_size_frames": COORDINATION_BLOCK_SIZE_FRAMES_V10,
        "hydration_severe_coordination_max": HYDRATION_SEVERE_COORDINATION_MAX,
        "hydration_severe_coordination_min_consecutive_frames": HYDRATION_SEVERE_COORDINATION_MIN_CONSECUTIVE_FRAMES,
        "hydration_severe_r5_nm": HYDRATION_SEVERE_R5_NM,
        "hydration_severe_r5_min_consecutive_frames": HYDRATION_SEVERE_R5_MIN_CONSECUTIVE_FRAMES,
        "hydration_severe_rule": "coordination<=3 AND r5>=0.4nm for >=2 consecutive frames",
        "hydration_stability_max_half_mean_delta": HYDRATION_STABILITY_MAX_HALF_MEAN_DELTA,
        "hydration_reference_path": os.path.relpath(reference_path, _REPO_ROOT) if reference_path else None,
        "hydration_reference_comparison_rule": "non_inferiority_lower_ci_ge_minus_margin",
        "hydration_reference_noninferiority_margin_water": HYDRATION_REFERENCE_NONINFERIORITY_MARGIN_WATER,
        "min_water_gap_nm": min_water_gap,
    }
    output_name = getattr(args, "output_name", "slab_quality_gate.json")
    out_path = os.path.join(output_dir, output_name)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, cls=core.NumpyEncoder)
    print(f"{'✅' if passed else '❌'} v{current_protocol} slab-quality-gate 完成: {out_path}")
    print(json.dumps({"checks": checks, "failure_reasons": reasons, "coordination_by_lambda": coordination_by_lambda}, indent=2, ensure_ascii=False, cls=core.NumpyEncoder))


# ============================================================================
# 子命令：report（不硬编码通过；缺文件/缺 λ/缺帧 → passed=false, status=incomplete）
# ============================================================================


def cmd_report(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    build_artifacts_dir = getattr(args, "build_artifacts_dir", None) or output_dir

    def _load(name):
        path = os.path.join(output_dir, name)
        if not os.path.exists(path):
            return None, path
        with open(path) as fh:
            return json.load(fh), path

    build_manifest_path = os.path.join(build_artifacts_dir, "build_manifest.json")
    if not os.path.exists(build_manifest_path):
        raise SystemExit(f"缺少 {build_manifest_path}，report 无法生成")
    with open(build_manifest_path, encoding="utf-8") as fh:
        build_manifest = json.load(fh)
    static_report_path = os.path.join(build_artifacts_dir, "static_check_report.json")
    if os.path.exists(static_report_path):
        with open(static_report_path, encoding="utf-8") as fh:
            static_report = json.load(fh)
    else:
        static_report = None
    dyn_manifest, _ = _load("dynamics_manifest.json")
    dg_result, _ = _load("charging_delta_G.json")
    slab_gate, _ = _load(getattr(args, "slab_gate_file", "slab_quality_gate.json"))

    missing = [
        name for name, value in [
            ("static_check_report.json", static_report), ("dynamics_manifest.json", dyn_manifest),
            ("charging_delta_G.json", dg_result), ("slab_quality_gate.json", slab_gate),
        ] if value is None
    ]

    checks = {
        "static_check_present": static_report is not None,
        "static_check_passed": bool(static_report and static_report.get("passed")),
        "dynamics_present": dyn_manifest is not None,
        "dynamics_all_11_lambda_present": bool(
            dyn_manifest and len(dyn_manifest.get("dcd_paths", [])) == len(dyn_manifest.get("lambdas_coul", [])) == 11
        ),
        "ukn_present": dg_result is not None,
        "ukn_converged": bool(dg_result and dg_result.get("converged")),
        "slab_quality_gate_present": slab_gate is not None,
        "slab_quality_gate_passed": bool(slab_gate and slab_gate.get("passed")),
        "coion_geometry_initial_passed": bool(
            build_manifest["ligand_coion_min_image_distance_nm"] >= core.COION_LIGAND_MIN_IMAGE_INITIAL_NM
        ),
        "total_charge_zero_at_build_passed": bool(
            abs(build_manifest.get("total_charge_at_build_e", float("nan"))) <= core.TOTAL_CHARGE_CONSERVATION_TOLERANCE_E
        ),
    }
    passed = len(missing) == 0 and all(checks.values())
    status = "complete" if not missing else "incomplete"

    report = {
        "protocol_version": PROTOCOL_VERSION, "status": status, "missing_artifacts": missing,
        "case": build_manifest["case"], "ion": build_manifest["ion"],
        "ligand_net_charge_e": build_manifest["ligand_net_charge_e"],
        "water_thickness_label": build_manifest["water_thickness_label"],
        "position_variant": build_manifest["position_variant"],
        "build_manifest": build_manifest, "static_check_report": static_report,
        "dynamics_manifest": dyn_manifest, "charging_delta_G": dg_result, "slab_quality_gate": slab_gate,
        "checks": checks, "passed": passed,
        "failure_reasons": missing + [k for k, v in checks.items() if not v],
    }
    report_name = getattr(args, "report_name", "report.json")
    summary_name = getattr(args, "summary_name", "summary.json")
    with open(os.path.join(output_dir, report_name), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, cls=core.NumpyEncoder)

    summary = {
        "case": build_manifest["case"], "status": status, "passed": passed, **checks,
        "delta_G_charging_kJ_mol": dg_result["delta_G_charging_kJ_mol"] if dg_result else None,
        "uncertainty_kJ_mol": dg_result["uncertainty_kJ_mol"] if dg_result else None,
        "failure_reasons": report["failure_reasons"],
    }
    with open(os.path.join(output_dir, summary_name), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"{'✅' if passed else '❌'} report 完成（status={status}）: {output_dir}/{report_name}")
    print(json.dumps(summary, indent=2, ensure_ascii=False, cls=core.NumpyEncoder))


def cmd_compare(args: argparse.Namespace) -> None:
    with open(args.report_a) as fh:
        a = json.load(fh)
    with open(args.report_b) as fh:
        b = json.load(fh)
    if a.get("charging_delta_G") is None or b.get("charging_delta_G") is None:
        raise SystemExit("两份 report.json 都必须已经跑完 ukn")

    dg_a = float(a["charging_delta_G"]["delta_G_charging_kcal_mol"])
    err_a = float(a["charging_delta_G"]["uncertainty_kcal_mol"])
    dg_b = float(b["charging_delta_G"]["delta_G_charging_kcal_mol"])
    err_b = float(b["charging_delta_G"]["uncertainty_kcal_mol"])

    ddg = dg_b - dg_a
    combined_sigma = float(np.sqrt(err_a**2 + err_b**2))
    threshold_2sigma = 2.0 * combined_sigma
    threshold_abs_kcal_mol = 1.0
    passed = (abs(ddg) <= threshold_2sigma) and (abs(ddg) <= threshold_abs_kcal_mol)

    result = {
        "comparison_label": args.label, "case_a": a.get("case"), "case_b": b.get("case"),
        "delta_G_a_kcal_mol": dg_a, "uncertainty_a_kcal_mol": err_a,
        "delta_G_b_kcal_mol": dg_b, "uncertainty_b_kcal_mol": err_b,
        "delta_delta_G_kcal_mol": ddg, "combined_sigma_kcal_mol": combined_sigma,
        "threshold_2sigma_kcal_mol": threshold_2sigma, "threshold_abs_kcal_mol": threshold_abs_kcal_mol,
        "passed": passed,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False, cls=core.NumpyEncoder))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"✅ compare 结果已写入 {args.output}")


# ============================================================================
# CLI
# ============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    p_eq = sub.add_parser("equilibrate-base", help="纯 slab 预平衡（无探针配体，GPU）")
    p_eq.add_argument("--top", default=DEFAULT_TOP_FILE)
    p_eq.add_argument("--gmx-include-dir", default=None)
    p_eq.add_argument("--gro", default=DEFAULT_RAW_GRO_FILE)
    p_eq.add_argument("--water-thickness-label", choices=["thin", "thick"], required=True)
    p_eq.add_argument("--output-dir", required=True)
    p_eq.add_argument("--n-steps", type=int, default=5_000_000)
    p_eq.add_argument(
        "--n-steps-nvt", type=int, default=0,
        help=(
            "切到 NPT（加 barostat）之前，先在固定盒下跑这么多步（默认 0=不跑，"
            "立即 NPT，与 v3 之前行为一致）。诊断 thick base 快速塌缩用："
            "先给新增水层一段固定盒 NVT 扩散松弛时间，再开 barostat；"
            "必须严格小于 --n-steps，且这一段与剩余 NPT 段各自都要能被 "
            "--report-interval-steps 整除。"
        ),
    )
    p_eq.add_argument("--n-steps-minimize", type=int, default=5000)
    p_eq.add_argument("--report-interval-steps", type=int, default=5000)
    p_eq.add_argument("--timestep-ps", type=float, default=0.002)
    p_eq.add_argument("--temperature-kelvin", type=float, default=303.15)
    p_eq.add_argument("--friction-per-ps", type=float, default=1.0)
    p_eq.add_argument("--seed", type=int, default=2026)
    p_eq.add_argument("--platform", default="CUDA", choices=["CUDA", "CPU", "OpenCL", "Reference"])
    p_eq.add_argument("--precision", default="mixed")
    p_eq.add_argument("--allow-cpu-fallback", action="store_true", default=False)
    p_eq.set_defaults(func=cmd_equilibrate_base)

    p_bqg = sub.add_parser("base-quality-gate", help="纯 slab（无探针）质量门（CPU，读 DCD）")
    p_bqg.add_argument("--top", required=True)
    p_bqg.add_argument("--gmx-include-dir", default=None)
    p_bqg.add_argument("--gro", required=True)
    p_bqg.add_argument(
        "--dcd", required=True, nargs="+",
        help="一个或多个 DCD 路径；多个时按给定顺序在时间轴上拼接（用于续跑后重判，"
             "见 recommendation 字段）",
    )
    p_bqg.add_argument("--frame-interval-ps", type=float, required=True)
    p_bqg.add_argument(
        "--literature-apl-nm2", type=float, default=DEFAULT_LIPID21_POPC_303K_LITERATURE_APL_NM2,
    )
    p_bqg.add_argument("--tail-fraction", type=float, default=0.2)
    p_bqg.add_argument(
        "--restart-discard-ps", type=float, default=1000.0,
        help="每个非首段 DCD（即从上一段 equilibrated.gro 重新初始化速度续跑出来的"
             "那些）开头丢弃多少 ps 再拼接——排除重启后的速度弛豫瞬态，避免把瞬态"
             "误判成真实漂移（2026-08-07 实测：不排除时窄末段窗口会把瞬态判成"
             "'仍在显著收缩'，排除后同一段其实是平的）。默认 1000 ps 是这次实测"
             "够用的量，不是普适物理常数——换体系/换协议要自己核对。",
    )
    p_bqg.add_argument("--output", required=True)
    p_bqg.set_defaults(func=cmd_base_quality_gate)

    p_ext = sub.add_parser("extend-water", help="对称生成第二种水层厚度的起始坐标（纯 CPU）")
    p_ext.add_argument("--top", default=DEFAULT_TOP_FILE)
    p_ext.add_argument("--gmx-include-dir", default=None)
    p_ext.add_argument("--gro", required=True, help="已平衡的『薄』水层坐标（equilibrate-base 的输出）")
    p_ext.add_argument("--extra-water-nm", type=float, required=True, help="沿 Z 追加的总厚度（两侧对称各半），nm")
    p_ext.add_argument("--seed", type=int, default=2026, help="新水随机取向用的种子")
    p_ext.add_argument("--output-dir", required=True)
    p_ext.set_defaults(func=cmd_extend_water)

    p_build = sub.add_parser("build", help="插入普通反离子+探针配体+reserved co-ion dummy（纯 CPU）")
    p_build.add_argument("--top", default=DEFAULT_TOP_FILE)
    p_build.add_argument("--gmx-include-dir", default=None)
    p_build.add_argument("--equilibrated-gro", required=True)
    p_build.add_argument("--ion", required=True, choices=sorted(ION_TEMPLATES))
    p_build.add_argument("--water-thickness-label", choices=["thin", "thick"], required=True)
    p_build.add_argument("--position-variant", type=int, required=True, choices=sorted(POSITION_VARIANT_LABELS))
    p_build.add_argument("--restraint-k", type=float, default=None)
    p_build.add_argument("--restraint-r0-nm", type=float, default=None)
    p_build.add_argument("--bulk-restraint-kz", type=float, default=BULK_RESTRAINT_KZ_KJ_PER_MOL_NM2)
    p_build.add_argument("--bulk-restraint-rz-nm", type=float, default=BULK_RESTRAINT_RZ_NM)
    p_build.add_argument(
        "--bulk-target-water-offset-nm", type=float,
        default=BULK_RESTRAINT_TARGET_OFFSET_DEFAULT_NM,
        help="把 pair-center target 沿远离膜核心方向移入 bulk water 的距离（v9，nm）",
    )
    p_build.add_argument(
        "--bulk-relative-z-fluctuation-limit-nm", type=float,
        default=BULK_RESTRAINT_RELATIVE_Z_FLUCTUATION_LIMIT_DEFAULT_NM,
        help="v9 静态几何设计采用的 ligand/co-ion 相对 Z 波动上限（nm）",
    )
    p_build.add_argument(
        "--bulk-membrane-undulation-margin-nm", type=float,
        default=BULK_RESTRAINT_MEMBRANE_UNDULATION_MARGIN_DEFAULT_NM,
        help="v9 静态几何设计的膜起伏裕量（nm）",
    )
    p_build.add_argument("--ligand-bulk-safety-kz", type=float, default=LIGAND_BULK_RESTRAINT_KZ_KJ_PER_MOL_NM2)
    p_build.add_argument("--ligand-bulk-safety-rz-nm", type=float, default=LIGAND_BULK_RESTRAINT_RZ_NM)
    p_build.add_argument(
        "--ligand-envelope-margin-nm", type=float,
        default=LIGAND_BULK_RESTRAINT_DESIGN_ENVELOPE_MARGIN_NM,
        help="v10 ligand heavy-atom envelope margin（nm）",
    )
    p_build.add_argument(
        "--coion-bulk-safety-kz", type=float,
        default=COION_BULK_SAFETY_KZ_KJ_PER_MOL_NM2,
        help="v11 co-ion member safety wall kZ（kJ mol^-1 nm^-2）",
    )
    p_build.add_argument(
        "--coion-bulk-safety-rz-nm", type=float,
        default=COION_BULK_SAFETY_RZ_NM,
        help="v11 co-ion member safety wall flat-bottom radius（nm）",
    )
    p_build.add_argument(
        "--coion-envelope-margin-nm", type=float,
        default=COION_BULK_SAFETY_DESIGN_ENVELOPE_MARGIN_NM,
        help="v11 co-ion safety static geometry margin（nm）",
    )
    p_build.add_argument("--output-dir", required=True)
    p_build.set_defaults(func=cmd_build)

    p_check = sub.add_parser("static-check", help="逐 λ 电荷=0+bulk-water 几何+restraint 唯一性自检（纯 CPU）")
    p_check.add_argument("--output-dir", required=True)
    p_check.set_defaults(func=cmd_static_check)

    default_lambda_coul = "1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0"
    p_dyn = sub.add_parser("dynamics", help="逐 λ 平衡+采样，写 DCD/timeseries.csv（GPU，用户提交）")
    p_dyn.add_argument("--output-dir", required=True)
    p_dyn.add_argument(
        "--build-artifacts-dir", default=None,
        help="从另一目录读取 v10 build 产物；用于不覆盖既有 pilot 的补充采样",
    )
    p_dyn.add_argument(
        "--restart-dcd", default=None,
        help="从指定 DCD 的单帧坐标续接确认段；Hamiltonian 仍从 build System 唯一配置",
    )
    p_dyn.add_argument(
        "--restart-frame", type=int, default=-1,
        help="restart DCD 帧号，默认 -1（最后一帧）",
    )
    p_dyn.add_argument("--lambda-coul", default=default_lambda_coul)
    p_dyn.add_argument("--n-steps-minimize", type=int, default=5000)
    p_dyn.add_argument("--n-steps-equil", type=int, default=20000)
    p_dyn.add_argument("--n-steps-sample", type=int, default=50000)
    p_dyn.add_argument("--n-steps-sample-lambda0", type=int, default=100000)
    p_dyn.add_argument("--save-interval-steps", type=int, default=500)
    p_dyn.add_argument("--temperature-kelvin", type=float, default=303.15)
    p_dyn.add_argument("--friction-per-ps", type=float, default=1.0)
    p_dyn.add_argument("--timestep-ps", type=float, default=0.002)
    p_dyn.add_argument("--seed", type=int, default=2026)
    p_dyn.add_argument("--platform", default="CUDA", choices=["CUDA", "CPU", "OpenCL", "Reference"])
    p_dyn.add_argument("--precision", default="mixed")
    p_dyn.add_argument("--allow-cpu-fallback", action="store_true", default=False)
    p_dyn.set_defaults(func=cmd_dynamics)

    p_ukn = sub.add_parser("ukn", help="从原始 system.xml 唯一配置 Hamiltonian，MBAR 求 charging ΔG（CPU）")
    p_ukn.add_argument("--output-dir", required=True)
    p_ukn.add_argument(
        "--build-artifacts-dir", default=None,
        help="从另一目录读取 v10 build/system.xml；轨迹仍从 output-dir 读取",
    )
    p_ukn.add_argument("--platform", default="CPU")
    p_ukn.set_defaults(func=cmd_ukn)

    p_gate = sub.add_parser("slab-quality-gate", help="case 专属质量门：读 DCD/timeseries（CPU）")
    p_gate.add_argument("--output-dir", required=True)
    p_gate.add_argument(
        "--build-artifacts-dir", default=None,
        help="从另一目录读取 build 产物；用于补充采样目录的质量门",
    )
    p_gate.add_argument(
        "--supplement-dir", default=None,
        help="把该目录的 matching-λ 轨迹/CSV 追加到主 pilot 后再验收",
    )
    p_gate.add_argument(
        "--output-name", default="slab_quality_gate.json",
        help="质量门 JSON 文件名；可用独立名称保留旧版 gate 证据",
    )
    p_gate.add_argument(
        "--hydration-reference", default=HYDRATION_REFERENCE_DEFAULT_PATH,
        help="C1 Na 水盒 coordination timeseries.csv 参考路径",
    )
    p_gate.set_defaults(func=cmd_slab_quality_gate)

    p_report = sub.add_parser("report", help="汇总为 report.json/summary.json（缺项即 incomplete）")
    p_report.add_argument("--output-dir", required=True)
    p_report.add_argument(
        "--build-artifacts-dir", default=None,
        help="从另一目录读取 build_manifest/static_check_report",
    )
    p_report.add_argument(
        "--slab-gate-file", default="slab_quality_gate.json",
        help="output-dir 内质量门 JSON 文件名",
    )
    p_report.add_argument(
        "--report-name", default="report.json",
        help="报告输出文件名；可用独立名称保留既有 report.json 证据",
    )
    p_report.add_argument(
        "--summary-name", default="summary.json",
        help="summary 输出文件名；可用独立名称保留既有 summary.json 证据",
    )
    p_report.set_defaults(func=cmd_report)

    p_cmp = sub.add_parser("compare", help="比较两份 report.json 的 charging ΔG 敏感性")
    p_cmp.add_argument("--report-a", required=True)
    p_cmp.add_argument("--report-b", required=True)
    p_cmp.add_argument("--label", required=True)
    p_cmp.add_argument("--output", default=None)
    p_cmp.set_defaults(func=cmd_compare)

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
