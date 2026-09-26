# B3 逐边决策表 —— 44 条非合法边的三个出口

**写于 2026-09-19，配 `HOW_TO_CONTINUE.md` B1/B2/B3 与 `layer_check.py` 读。**
复现命令：仓库根 `python3 refactor/tools/layer_check.py`（纯 AST，不 import 被测代码）。

> **勘误（2026-09-20）**：本文全部基数（1336/97、707/629/44、253/629）是
> **infra 抽取前**的基线。抽取后实测（17 域口径）：总 **1357 次/107 边**，
> 合法 913/62、违反 130/23、同层 314/22，文档口径非合法 **444 次/45 边**。
> infra 抽取正是本文 §7 的第一刀（消掉 4 条边/253 次引用的模拟已落地），
> 逐边决策的**相对结构与出口**仍然有效，绝对数按上折算。

---

## 0. 实测口径（先对账，再决策）

### 0.1 707 / 629 / 44 全部复现，但「违反」的口径要澄清

| 桶 | 引用次数 | 边数 |
|---|---|---|
| 合法 = 严格 高层→低层 | **707** | 53 |
| 同层（层号相等） | **353** | 21 |
| 违反 = 低层→高层 | **276** | 23 |
| 合计 | 1336 | 97 |

**文档的 629 次 / 44 条边 = 同层 353/21 + 向上 276/23。** 即文档口径是
「合法 = 严格下降；**其余一切（含同层）都算违反**」。
本表建议把口径明确为三桶：**同层互引是分区内聚，不构成分层违反**；
真正要逐条拍板的是 276 次 / 23 条严格上升边。本文对 44 条全部给出口子，
同层的 21 条统一走「接受」，附个别附加动作。

### 0.2 B2 infra 名单核验（数字全部对上）

文档 253/629=40% 的口径 = **落在「同层+向上」非合法边上的引用**（不含合法边）：

| 名字 | 非合法边上 | 文档 | 定义位置（主线） | 域文件 |
|---|---|---|---|---|
| `log` | 104 | 104 ✓ | runabfe:145 `log = logging.getLogger("runabfe")` | pipeline_orchestration/runabfe_head |
| `logger` | 33 | 33 ✓ | abfe_core:186 / ibs_engine:100 / abfe_pipeline:242，三处都是 `logging.getLogger(__name__)` | runtime_env / mbar_analysis / pipeline_orchestration |
| `NumpyEncoder` | 29 | 29 ✓ | abfe_core:7881、abfe_pipeline:2971（两份 body 仅 isinstance 检查顺序与 docstring 不同，`default()` 输出等价） | system_build/units_validators、pipeline_orchestration/pipeline_head |
| `_sha256_file` | 28 | 28 ✓ | ibs_engine:368、runabfe:428、apbs_correction:104（三份摘要等价，可合）；**abfe_pipeline:1112 `_file_sha256` 是第四份，🚫 语义不同** | mbar_analysis、pipeline_orchestration、charging |
| `_atomic_write_json` | 21 | 21 ✓ | ibs_engine:1201、rbfe_pipeline:205、runabfe:3879（三份落盘字节不同，🚫 不合） | mbar_analysis、rbfe、pipeline_orchestration |
| `print_once` | 14 | 14 ✓ | abfe_core:7415、ibs_engine:337（代码体逐字同，✅ 可合，留 ibs_engine docstring） | membrane、mbar_analysis |
| `_timed` | 12 | 12 ✓ | ibs_engine:1004（纯计时 contextmanager） | mbar_analysis |
| `_atomic_save_npy` | 12 | 12 ✓ | ibs_engine:357（另有 `_atomic_save_npz` ibs_engine:1225，一并收） | mbar_analysis |
| **合计** | **253** | **253 ✓** | | |

只算严格上升边时这 8 个名合计 180 次（`_atomic_write_json`、`_atomic_save_npy`
的全部引用都在同层边 windows_sampling→mbar_analysis 上）——这就是为什么
文档口径必须是「同层+向上」才能得出 253。

### 0.3 决策 #1：抽 L0 `infra` 域（B2，第一号决策）

**出口：反转（全部 8 个名下沉 L0）。** 模拟（`layer_check.py` 把这 8 个名强制
改归虚拟 L0 域后重算）：

- **文档口径：非合法边 44 条/629 次 → 40 条/376 次，消掉 4 条边 / 253 次引用。**
  整条消失的边：membrane→mbar_analysis(3)、membrane→pipeline_orchestration(4)、
  orb_surrogate→pipeline_orchestration(2)、orb_surrogate→system_build(2)。
  「40% 自动消失」指的是**引用次数**（253/629），不是边数（4/44）——
  剩下的边只是被削薄，不消失。
- 严格口径：上升边 23 条/276 次 → **19 条 / 96 次**。

**infra 收编的落地方式（逐名，含 🚫 禁令）：**

| 名字 | 处理 |
|---|---|
| `log` | 下沉**原样一份**（`logging.getLogger("runabfe")`，通道名字符串逐字保留——改通道名会改日志输出归属）。使用方 104 次全是各 runabfe 系文件读模块全局 |
| `logger` | **不下沉。** 三处都是 `getLogger(__name__)`，通道名随模块走；包化时每个域文件恢复自己的模块头（本文件里那行本来就是它的），边是切分伪影，不是真耦合 |
| `NumpyEncoder` | 两份输出等价；下沉一份到 infra，删另一份属「合并」，按 3.3 是改行为刀，需单独 diff 证明（或保守：两份都原样入 infra 同域） |
| `_sha256_file` | ✅ 可合：ibs_engine:368 / runabfe:428 / apbs_correction:104 三份下沉合并。**🚫 abfe_pipeline:1112 的 `_file_sha256`（`except OSError: return None`）不许并入**——sha256 进过缓存身份路径，None 与抛错的差别会静默改缓存键；它**原样下沉、单独一份**，不合并不去重 |
| `_atomic_write_json` | **🚫 三份落盘字节不同（`sort_keys,ensure_ascii` / `cls=NumpyEncoder` / tmp 名不带 PID），本阶段不合。** windows_sampling→mbar_analysis 那条同层边上的 21 次，随 ibs_data_loading_head 的拆分走（见 §2 第 1 条）；rbfe、runabfe 两份留在原域。统一参数 = 改 JSON 字节 = 改缓存身份，单独立项先算存量代价 |
| `print_once`+`_ONCE_EMITTED_KEYS` | ✅ 可合（留 ibs_engine 那份 docstring） |
| `_timed` / `_atomic_save_npy` / `_atomic_save_npz` | 原样下沉 |

另注：WIRING 接线事实 3 的 `print = _log_print`（abfe_core:7409，membrane_gates 域）
在 deps.py/layer_check.py 里因 `print` 是 builtin 被排除，**不出现在任何边里**；
接线时按 WIRING 的指示从 membrane_gates 域引。

---

## 1. infra 抽取后剩余的 19 条严格上升边（96 次）——逐条决策

排序按 infra 抽取后的剩余引用数。每条出口三选一。

### 1.1 charging → pipeline_orchestration（13 次）
infra 后：`ABFEPipeline:4, RunConfig:2, _write_run_provenance:2, _has_valid_boresch_restraint:2, _pme_u_kn_meta_payload:1, _load_dexp_params_fail_closed:1, MAIN_SYSTEM_CACHE_PROTOCOL_VERSION:1`
全部来自 **charging/charging_only_rerun.py**（runabfe 6118-6822）：
它实例化 `ABFEPipeline`（abfe_pipeline:3226）、读 `RunConfig`（runabfe:3378，
argparse+预设合并的运行配置）、写 `_write_run_provenance`（runabfe:3740，落盘 provenance）。
**出口：拆分 charging。** charging_only_rerun 是 runabfe 的「只跑电荷段重跑」编排模式，
不是电荷物理——划归 pipeline_orchestration（L4），整条边消失（13 次随之变 L4 域内引用）。
charging 域剩下的 5 个文件是纯电荷协议物理。

### 1.2 system_build → pipeline_orchestration（11 次）
infra 后：`SOLVENT_CACHE_PROTOCOL_VERSION:3, DEFAULT_SOLVENT_IONIC_STRENGTH_MOLAR:2, _main_cache_identity:1, MAIN_SYSTEM_CACHE_PROTOCOL_VERSION:1, _build_platform_props:1, _is_traj_valid:1, _is_checkpoint_valid:1`
- 溶剂缓存身份块（runabfe:407-477：`MAIN_SYSTEM_CACHE_PROTOCOL_VERSION=3`、
  `SOLVENT_CACHE_PROTOCOL_VERSION=7`、`DEFAULT_SOLVENT_IONIC_STRENGTH_MOLAR`、
  `SOLVENT_PADDING_NM`、`_main_cache_identity`、`_gromacs_dependency_hashes`）：
  纯缓存身份定义，被 system_build/solvent_leg_cache、gmx_paths_status 消费。
- `_is_checkpoint_valid`（abfe_pipeline:2093）、`_is_traj_valid`（2134）、
  `_build_platform_props`（2871）：fail-closed 文件/平台校验纯函数，runabfe:110
  从 abfe_pipeline import 它们——本就是跨文件共享的低层校验。
**出口：反转。** 缓存身份块下沉 system_build/solvent_leg_cache；
三个校验函数下沉 system_build（gmx_paths_status 旁）。
⚠️ **缓存身份路径：纯搬运逐字不变**（协议版本号、sha256 进 key 的字段一字不动）。

### 1.3 lambda_path → stage2_controller（12 次）⚠️ 含待用户决策
infra 后：`Stage2RepairController:3, finalize_descending_lambda_path:2, _sample_group1_energies:2, classify_pilot_point_risk_zone:1, pilot_block_running_diagnostics:1, _normalize_variance_weights:1, _pilot_segment_lengths:1, canonicalize_window_ranges:1`
- 8 次是 S2-D 纯函数与诊断纯函数（abfe_preoptimizer 9386-12341：
  `finalize_descending_lambda_path` 10953、`_sample_group1_energies` 11555、
  `canonicalize_window_ranges` 10881、`_normalize_variance_weights` 10944、
  `classify_pilot_point_risk_zone` 12224、`pilot_block_running_diagnostics` 12172、
  `_pilot_segment_lengths` 12158），使用方是 lambda_path/abfe_preoptimizer_class、
  dual_lambda_preoptimizer、path_math——**这些函数本来就是 λ 路径数学**。
- 3 次是 `Stage2RepairController` 类本身（abfe_preoptimizer:3020，6386 行巨类），
  使用方 lambda_path/probe_systems_2d.py:771-786（replay/诊断入口）。
**出口：反转（主）。** 上述 8 个纯函数下沉 lambda_path（它们与 Stage2RepairController
无类依赖）。**剩余 3 次 Stage2RepairController 类引用：待用户决策**——
(a) 接受：stage2_controller 整域下调 L2（它的入边只有 lambda_path 与 pipeline_orchestration，
下调不产生新上升边）；(b) 拆分 lambda_path：probe_systems_2d 上移 L3。
两者都成立，涉及巨类邻居关系，留给用户拍板。

### 1.4 windows_sampling → pipeline_orchestration（6 次）
infra 后：`_has_valid_boresch_restraint:6`（windows_sampling/window_manager_dual_lambda）。
`_has_valid_boresch_restraint` 有两份（abfe_pipeline:1647、ibs_engine:24441，
misc/force_diagnostics 已持后者）：6 行纯 dict 检查（3+3 锚点齐全性），
无任何 openmm 依赖。**出口：反转。** 下沉 misc（L1），两份合一；
全部 6 处使用方（windows_sampling、lambda_path、charging、mbar_analysis、remd、
pipeline_orchestration 自身）都变合法/同层。

### 1.5 boresch → mbar_analysis（9 次）
infra 后：`TraditionalMBARAnalyzer:2, _prepare_pme_coulomb_leg_system:1, _restore_ligand_internal_nonbonded:1, _add_physical_boresch_restraint:1, pme_self_correction_prefactor_kj:1, pme_self_correction_energy_kj:1, _value_in_inverse_nanometer:1, Exp019SeedLedger:1`
使用方是 **attachment_leg.py 与 boresch_attachment_remd.py**——两条腿的**运行编排**
（跑腿、跑 REMD、腿内做 MBAR 分析），不是 Boresch 力/几何物理。
**出口：拆分 boresch**，按「物理 vs 腿编排」切：
- 留 L2：boresch_forces、analytic_corrections_fixups、boresch_geometry_estimators（物理）；
- 上移 L3：attachment_leg（ibs_engine 1994-3308）、boresch_attachment_remd（25825-26145）
  ——它们与 remd、mbar_analysis 同层互引，是采样编排；
- 上移 L4：runabfe_boresch_params（runabfe 4149-4529；用 `ABFEPipeline`、`RunConfig`、
  `log`×14，是 runabfe 参数管道）——其中 `_sanitize_boresch_params`（runabfe:4152）、
  `_sanitize_boresch_params_strict`（4210）、`_has_valid_boresch_anchors`（4201）
  是纯 dict 校验，**先下沉留 L2 boresch**（charging 同层要用，见 §2）。
PME 腿制备 5 个名字随 §1.6/§1.7 的下沉消掉。

### 1.6 charging → mbar_analysis（9 次）
infra 后：`_value_in_elementary_charge:4, _compute_ligand_net_charge:2, TraditionalMBARAnalyzer:1, _create_co_alchemical_ion_restraint:1, _value_in_inverse_nanometer:1`
- `_value_in_elementary_charge`（ibs_engine:1334）、`_value_in_inverse_nanometer`（1317）：
  OpenMM Quantity 归一化纯助手 → 下沉 misc（L1）。
- `_compute_ligand_net_charge`（1918）、`_create_co_alchemical_ion_restraint`（1273）：
  PME/电荷协议物理，被 coion_charge_transfer、reserved_coions_build 用——
  是切分时被扫进 ibs_data_loading_head 的 charging 内容 → **搬回 charging**。
- `TraditionalMBARAnalyzer:1` 来自 charging_only_rerun，随 §1.1 归 L4。
**出口：反转。** 全部消掉。

### 1.7 system_build → mbar_analysis（7 次）
infra 后：`SOLVENT_NONBONDED_CUTOFF_NM:3, _compute_ligand_net_charge:2, resolve_water_model_xml:1, solvent_box_edge_nm:1`
全部定义在 **mbar_analysis/solvation_closure.py**（abfe_core 11332-11747「修复 5：
溶剂化能闭环」）：水模型 itp→XML 表（11528）、溶剂盒边长计算（11600）、
nonbonded cutoff 常量（11336）——**这是溶剂体系构建，不是 MBAR 分析**。
**出口：反转（整文件搬移）。** solvation_closure 划归 system_build（L2）。
注意它 387-399 有一个**既存的**函数级 `from abfe_pipeline import ABFEPipeline`
（「委托给 ABFEPipeline」的钩子）——搬运不加新 lazy import，这个既存钩子在 C 阶段接线时单独处理。

### 1.8 remd → pipeline_orchestration（2 次）
infra 后：`_split_platform_spec:1, _has_valid_boresch_restraint:1`。
`_split_platform_spec`（abfe_pipeline:2810 / ibs_engine:3180，两份逐字同，✅ 可合）
随 infra 下沉；`_has_valid_boresch_restraint` 随 §1.4 下沉 misc。**出口：反转。** 消掉。

### 1.9 lambda_path → windows_sampling（4 次）
`IBSBiasForce:2, IBS_E_BASE_FORCE_GROUPS:1, IBS_E_BIAS_FORCE_GROUPS:1`
使用方 lambda_path/dual_lambda_system（建双 λ IBS 体系时就要装偏置力）与
charging/shadow_coulomb_decharge（§2 相关 1 次）。`IBSBiasForce`（ibs_engine:6425）
是 OpenMM Force 封装类（含 EXP-025 G4 native residual basis，docstring 明写
byte-identical 约束、有 XML-diff 回归测试）；两个 force-group 常量在 6864/6865。
**出口：反转。** `IBSBiasForce` + `IBS_E_*_FORCE_GROUPS` 纯搬运下沉到新 L2 域
（建议 `lambda_path/dual_lambda_system` 旁或独立 `ibs_force`），windows_sampling
保留采样器部分。⚠️ 类体**逐字不动**（EXP-025 G4 的 XML-diff 测试钉着）。
此刀口大、置信度中——见 §5 待用户决策。

### 1.10 boresch → remd（3 次）
`REMDManager:3`（ibs_engine:24726，REMD 编排器），使用方 attachment_leg×2 +
boresch_attachment_remd×1。**出口：拆分 boresch**（同 §1.5：两腿文件上移 L3），
L3→L3 同层，消掉。

### 1.11 mbar_analysis → pipeline_orchestration（1 次）
`_has_valid_boresch_restraint:1` → 随 §1.4 下沉 misc。**出口：反转。** 消掉。

### 1.12 orb_surrogate → membrane（2 次）
`_build_mace_potential:1`（abfe_core:7337）、`_build_openmmml_kwargs:1`（7307）：
openmm-ml/MACE 参数组织与本地模型路径映射（`_MACE_LOCAL_MODEL_PATHS` 7352），
纯函数、依赖 `_require_mlpotential()`（runtime_env 惰性探测）——
是 ML 势机器，切分时被扫进 membrane_gates 域文件（abfe_core 7272-7430 区段）。
**出口：反转。** 两个函数（连同 `_MACE_LOCAL_MODEL_PATHS`）下沉 orb_surrogate。

### 1.13 result_schema → boresch（2 次）
`evaluate_cross_leg_conformer_consistency:1`（abfe_core:8464）、
`assert_cross_leg_conformer_consistency:1`（8540）：P0-12a 跨腿构象一致性门——
纯 dict in / dict out / raise，无 openmm，被 result_schema_gates（abfe_core 8547-9147，
紧随其后）调用。**出口：反转。** 连同常量 `LIGAND_CONFORMER_OVERLAP_PERCENTILES`、
`LIGAND_CONFORMER_DIAGNOSTICS_VERSION` 下沉 result_schema（L1）——它们本来就是
门/契约逻辑。

### 1.14 misc → charging（1 次）
`unwrap_boresch_anchors_nm:1`（abfe_core:1661，**BOR-01：六锚点解缠唯一实现**，
docstring 明写平移量取整数格矢、没跨边界时返回坐标与输入逐位相同——**行为敏感**）。
同区还有 `minimum_image_displacement_nm`（1561，**minimum-image 数学唯一实现**），
被 membrane、boresch、charging 跨域用。**出口：反转。** 两个唯一实现下沉 misc
（或新 L1 `pbc_geom` 域），纯搬运逐字不变；misc→charging 消掉，
charging/membrane/boresch→misc 全部变合法。

### 1.15 lambda_path → mbar_analysis（5 次）
infra 后：`SOFTCORE_CUTOFF_NM:2, ibs_lj_tail_lrc_is_applicable:1, _value_in_elementary_charge:1, ibs_lj_tail_lrc_inapplicable_reason:1`
- `SOFTCORE_CUTOFF_NM`（ibs_engine:240）：softcore 势参数，注释明写只影响
  `potential_type="softcore"` → 下沉 orb_surrogate/softcore_fitters（L1）。
- `ibs_lj_tail_lrc_is_applicable`（915）/`_inapplicable_reason`（981）：LRC 色散适用性
  物理判定 → 下沉 lambda_path/forcefield_dispersion（同域消掉）。
- `_value_in_elementary_charge` 随 §1.6。
**出口：反转。** 消掉。

### 1.16 charging → windows_sampling（1 次）
`IBSBiasForce:1`（shadow_coulomb_decharge）→ 随 §1.9 下沉。**出口：反转。** 消掉。

### 1.17 lambda_path → pipeline_orchestration（2 次）
infra 后：`_has_valid_boresch_restraint:2`（dual_lambda_system）→ 随 §1.4。**出口：反转。** 消掉。

### 1.18 outer_lambda_neural → pipeline_orchestration（2 次）——假边
`_minimum_image_displacement:2`。outer_lambda_neural/protocol_head.py **已自带副本**
（outer_lambda_neural_basis:644，刻意不 import 主线——WIRING 接线事实 1）；
边之所以存在，是因为 abfe_pipeline:1017 也有一份同名薄包装（跨域重名让
deps.py 把每次 Load 记到每个非本域定义域头上）。HOW_TO_CONTINUE B2 已把
`_minimum_image_displacement` 组定性为**已撤回的误报，不要动**。
**出口：接受。** 包化后 protocol_head 持有自己的名字，边自然消失；不搬运、不重命名主线。

（§1.4/1.8/1.11/1.17 之外，membrane→pipeline_orchestration 4 次、
orb_surrogate→pipeline_orchestration 2 次、orb_surrogate→system_build 2 次、
membrane→mbar_analysis 3 次已由决策 #1 infra 整条消掉。）

---

## 2. 21 条同层边——默认「接受」，3 条附动作

同层互引不构成分层违反（§0.1 口径建议）。逐条过一遍，除前三条外均无动作：

| # | 边 | 次 | 决策 | 说明 |
|---|---|---|---|---|
| 1 | windows_sampling → mbar_analysis | 133 | **拆分 mbar_analysis** | 见 §2.1 |
| 2 | mbar_analysis → windows_sampling | 16 | 接受（可顺手反转） | `IBS_LOCAL_MBAR_GATE_*`（ibs_engine:7981-8047）、`_logsumexp_rows`/`_ess_from_log_weights`（20245/20255，纯统计小函数）、`frozen_candidate_fingerprint`（8156）——可下沉 L2 数学助手域，非必须 |
| 3 | remd → mbar_analysis | 14 | 接受（随 §2.1 大幅缩小） | infra 后剩 `Exp019SeedLedger:3` 等，随契约域下沉变合法 |
| 4 | membrane ↔ charging（37 / 2） | 39 | 接受 | membrane 用的 APL/跨截面/漂移门限常量（abfe_core:2631-2668）住在 charge_protocol_core——同源文件切成两域的产物，L2 内聚合法 |
| 5 | system_build ↔ membrane（28 / 9） | 37 | 接受 | MEMBRANE_UPSTREAM_* 契约常量（abfe_core:4592-4630）、`_topology_atoms`（5703）；同层 |
| 6 | charging ↔ system_build（19 / 15） | 34 | 接受 | `ensure_owned_system`/`create_ligand_internal_force`（abfe_core:11938-12569，io_system_utils）与电荷常量互引；同层 |
| 7 | membrane ↔ lambda_path（14 / 9 / 3） | 26 | 接受 | `WATER_MOLECULE_NAMES`/`DISPERSION_PROTOCOL*`（abfe_core:2868-3173）同层共享 |
| 8 | charging ↔ lambda_path（10 / 6） | 16 | 接受 | 电荷协议常量与窗口生成互引；同层 |
| 9 | boresch → system_build | 12 | 接受 | infra 后剩 7 次 gmx/体系校验函数；同层 |
| 10 | system_build ↔ lambda_path（9 / 8） | 17 | 接受 | `load_gromacs_topology_for_openmm`（3981）与 `ensure_owned_system` 互引；同层 |
| 11 | charging ↔ boresch（5 / 3） | 8 | 接受 | `_sanitize_boresch_params`（§1.5 下沉后仍同层） |
| 12 | boresch ↔ lambda_path（1 / 1） | 2 | 接受 | 单函数互引 |
| 13 | lambda_path ↔ system_build 见 #10；lambda_path→charging 见 #8 | | | |
| 14 | orb_surrogate → result_schema | 2 | 接受 | `boresch_dihedral_rad`（abfe_core:8880）纯几何契约，L1 内聚 |

（方向与次数以 `layer_check.py` 实测输出为准；上表覆盖全部 21 条同层边。）

### 2.1 拆分 mbar_analysis（本表最大的一个决策）

windows_sampling→mbar_analysis 的 133 次（infra 后 85 次）几乎全部指向
**ibs_data_loading_head**（ibs_engine 1-1993）：窗口数据协议版本
（`IBS_WINDOW_DATA_PROTOCOL_VERSION` 102、`IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION` 111）、
窗口三元组加载校验（`_load_validated_window_data_triplet` 418、
`load_ibs_window_outputs_from_dir` 573）、production segment 机器（1021-1096）、
`Exp019SeedLedger`（155）、`IBSControlPlaneError`（1119）、roundoff 比较助手（1570/1596）。
**这是「窗口数据契约层」，不是 MBAR 分析**——采样器（windows_sampling）、
REMD、腿编排都要读它。

**出口：拆分。** mbar_analysis 一分为二：
- 新 L2 域 `window_contract`（暂名）：ibs_data_loading_head 里的契约/加载/ledger/
  production-segment 部分（infra 名已先行抽走、单位与 PME 助手按 §1.6 迁走后）；
- mbar_analysis 保留：GlobalMBARAnalyzer（1350 行）、TraditionalMBARAnalyzer（897 行）、
  ess_target_support_gates、split_half_diagnostics、chunked_mbar、stage1_crosschecks、
  multi_segment_analysis。

效果：windows_sampling→window_contract 变 L3→L2 合法（133 次大头消掉）；
remd→window_contract 合法；mbar_analysis→window_contract 合法（分析器读窗口数据本就该向下）。
反向 mbar→windows 16 次保留同层（§2 #2）。
**具体刀口（哪些函数进契约域）需在动手前逐函数核对再用户新拍板——本文只定方向。**

---

## 3. 🚫 禁令继承（原样，来自 HOW_TO_CONTINUE B2）

1. **`abfe_pipeline:1112` 的 `_file_sha256`（`except OSError: return None`）不许与
   ibs_engine:368 / runabfe:428 / apbs_correction:104 三份合并**——语义不同，
   sha256 进过缓存身份路径，None 与抛错的差别会静默改缓存键。
   涉及它的决策只能写「**下沉原样的一份**」，不能写「合并去重」。
2. **三份 `_atomic_write_json`（ibs_engine:1201 / rbfe_pipeline:205 / runabfe:3879）
   落盘字节都不同**（`sort_keys=True,ensure_ascii=False` / `cls=NumpyEncoder` /
   tmp 名不带 PID 且失败不清理）——**本阶段不合**。统一任一份的参数都会改 JSON
   字节 ⟹ 下游 sha256 变 ⟹ 缓存身份变。涉及它们的决策只能写「下沉原样的一份」。
3. **`_minimum_image_displacement` 组不要动**（abfe_pipeline:1017、ibs_engine:1257、
   ibs_engine:12874 薄包装 + outer_lambda_neural_basis:644 独立实现是设计不是债）。
4. 所有标记「反转/下沉」的搬运都是**纯搬运**（3.3）：行内容逐字不变，
   `sort | md5sum` 前后相同；缓存身份相关（§1.2 的协议版本常量与 `_main_cache_identity`、
   §1.14 的 BOR-01 解缠、§1.9 的 IBSBiasForce）一字不改。

---

## 4. 落地后的账

| 阶段 | 非合法边（文档口径） | 其中严格上升 |
|---|---|---|
| 现状 | 44 条 / 629 次 | 23 条 / 276 次 |
| 决策 #1 infra 抽取后 | 40 条 / 376 次 | 19 条 / 96 次 |
| §1 全部反转/拆分落地后 | 同层 21 条保留（口径内不算违反） | **严格上升 0 条真耦合**；仅剩 §1.3 的 3 次 `Stage2RepairController` 待拍板（选 (a) 或 (b) 后也为 0） |

每把刀落地后重跑 `python3 refactor/tools/layer_check.py`：严格上升边数必须单调下降或持平。

---

## 5. 待用户决策清单

| # | 事项 | 选项 |
|---|---|---|
| 1 | **§1.3 剩余 3 次 `Stage2RepairController`**（probe_systems_2d 的 replay 入口） | (a) stage2_controller 下调 L2；(b) probe_systems_2d 上移 L3 |
| 2 | **§2.1 拆分 mbar_analysis 的刀口**（window_contract 新域收录哪些函数、命名） | 方向已定，函数清单动手前逐个核对拍板 |
| 3 | **§1.9 IBSBiasForce 下沉**（类体 400+ 行，EXP-025 G4 byte-identical 约束 + XML-diff 回归测试） | (a) 下沉新 L2 域；(b) 不动、接受 L2→L3 同层化（把 dual_lambda_system/shadow_coulomb 的建系钩子改走 windows_sampling 的工厂） |

**最难的 3 条**：上表 3 条。共同点是都碰「巨类/行为敏感实现的归属」，
而不是纯函数搬家。

---

## 6. 用户拍板（2026-09-19，对 §5 三条的结论）

拍板集口径同步修正：**23 条严格上升边 + 9 对同层互环 = 32 项 / 601 次**
（同层 21 条拆成互环 9 对/325 次 + 单向 3 条/28 次；单向不管）。
L2 里 lambda_path/charging/membrane/boresch/system_build 五域咬合的 8 对互环
**合并成一项整体决策**，不逐边过。

| # | 事项 | 拍板 | 理由（用户原话要点） |
|---|---|---|---|
| 1 | §1.3 剩余 3 次 `Stage2RepairController` | **(a)(b) 都不选**：probe_systems_2d:771-786 是 replay/诊断入口，把 controller 当**参数传进去**，不 import | 为 3 次引用挪整个域或拆 lambda_path 都太贵；参数注入结构上零成本 |
| 2 | §2.1 window_contract 刀口 | **方向对，函数清单押后**：infra 抽取落地后重跑 `layer_check.py`，对着残量定清单 | 133 次里约 38% 是 infra 名；拿 133 去切域会把 window_contract 切歪，抽完 infra 它是 ~82 不是 133 |
| 3 | §1.9 IBSBiasForce | **(b) 不动，押后** | 有 EXP-025 G4 byte-identical + XML-diff 回归钉着，风险收益倒挂；且不在 9 对同层互环里，不是路上的石头 |

---

## 7. L2 五域互环的整体决策（补 §6 留下的空洞）

§6 定了「L2 里 `lambda_path`/`charging`/`membrane`/`boresch`/`system_build`
五域咬合的 8 对互环合并成一项整体决策，不逐边过」，但没写那项决策是什么。
§2 把它们逐条标成「接受」——**那是错的**：同层不违反分层，但同层**互环**
照样是 `ImportError: cannot import name`。本节补上。

### 7.1 实测：抽取能消掉 84% 的流量，但消不掉拓扑

`layer_check.py` 口径，只统计这五域**内部**的互引：

| 场景 | 引用 | 有向边 | 互环 |
|---|---|---|---|
| 现状 | 188 | 17 | **8 对** |
| ① 抽 L0 `infra` 后 | 169 | 17 | **8 对** |
| ①+② 再抽 **L1 常量契约域** | 72 | 17 | **8 对** |
| ①+②+③ 再抽 **L1 拓扑/体系助手域** | **30** | 15 | **6 对** |

构成：常量 97 次（52%）、函数/类 72 次、infra 19 次。

**关键结论：常量和助手抽走 84% 的引用次数，互环只从 8 对降到 6 对。**
所以「抽一个域让 L2 自己散开」是行不通的——流量和拓扑是两件事，
这一条要写死，别再拿引用次数当进度指标。

② 的内容：`WATER_MOLECULE_NAMES`、`CHARGE_TREATMENT_*`、`MEMBRANE_UPSTREAM_*`、
`APL_*`、`LIPID_HEAD_*`、`DISPERSION_PROTOCOL_*`、`PROTEIN_CROSS_SECTION_GRID_NM`
等全部 L2 跨域常量。
③ 的内容：`_topology_atoms`、`ensure_owned_system`、`load_gromacs_topology_for_openmm`、
`create_ligand_internal_force`、`sync_all_exclusions`、`minimum_image_displacement_nm`、
`_normalize_ion_name`、`_box_vectors_to_nm_array` 等拓扑/体系操作助手。

### 7.2 残留 6 对互环：全是 1–3 次的单函数互引

```
boresch ⇄ charging
  → unwrap_boresch_anchors_nm×1, get_pme_alpha_for_system×1
  ← _sanitize_boresch_params×3, _sanitize_boresch_params_strict×1, _has_valid_boresch_anchors×1
boresch ⇄ lambda_path        → _periodic_box_volume_nm3×1   ← _build_platform_properties×1
charging ⇄ lambda_path       → ibs_dual_max_lambda_states×1
                             ← generate_overlapping_windows×1, configure_charge_transfer_decharging×1,
                               configure_coalchemical_neutral_decharging×1
charging ⇄ membrane          → resolve_environment_type×1   ← pre_equilibration_frame_interval_ps×1
charging ⇄ system_build      → build_system_from_gromacs×1
                             ← co_alchemical_ion_builder_identity_payload×2,
                               _insert_reserved_coalchemical_ion_dummies×1
lambda_path ⇄ membrane       → resolve_environment_type×1   ← normalize_protein_residue_name×1
```

### 7.3 决策：**5 对是「切歪了」，改归属即可；只有 1 对是真耦合**

查了这几个名字的当前归属，多数是机械切分把符号切到了错误的域：

| 符号 | 现归属 | 判定 | 处置 |
|---|---|---|---|
| `unwrap_boresch_anchors_nm` | `charging/charge_protocol_core` | **切歪**（Boresch 几何助手落进电荷域，是 `abfe_core` L715-2832 整段切分的产物） | 移入 `boresch` |
| `_build_platform_properties` | `boresch/attachment_leg` | **切歪**（平台属性助手，与 Boresch 无关） | 移入 L0 `infra` |
| `_periodic_box_volume_nm3` | `lambda_path/dual_lambda_system` | **切歪**（纯几何） | 随 ③ 入拓扑助手域 |
| `resolve_environment_type` | `membrane/membrane_protocol` | **切歪**（`charging` 和 `lambda_path` 都要用的通用环境判定，不专属膜） | 随 ② 入 L1 契约域 |
| `normalize_protein_residue_name` | `lambda_path/forcefield_dispersion` | **切歪**（残基命名规范化） | 随 ② 入 L1 契约域 |
| `get_pme_alpha_for_system` | `charging/shadow_coulomb_decharge` | 边界情况 | 随 ③ 或留原处，看 ③ 定稿 |

处置完之后 `boresch ⇄ charging`、`boresch ⇄ lambda_path`、`charging ⇄ membrane`、
`lambda_path ⇄ membrane` 四对直接消失；`charging ⇄ lambda_path` 只剩
`ibs_dual_max_lambda_states×1` 一个方向，也消失。

**真耦合只剩 1 对**：`charging ⇄ system_build`
——`charging` 要 `build_system_from_gromacs` 建系，`system_build` 要
`co_alchemical_ion_builder_identity_payload` 拿 co-ion 身份。这一对按
§6 对 `Stage2RepairController` 的同一处置：**参数注入，不 import**
（建系时把 co-ion 身份 payload 当参数传入，而不是让 `system_build` 反向引 `charging`）。

### 7.4 所以整体决策是

> **不是「抽一个域让 L2 散开」，而是「一次抽取（②③）+ 五处改归属 + 一处参数注入」。**
> 抽取消掉 84% 引用但消不掉环；环靠改归属消，因为其中 5/6 根本不是设计耦合，
> 是机械切分的产物。

⚠️ 落地顺序上有一条硬约束：**②③ 的抽取必须在改归属之前**，否则「移入 L1 契约域」
这个动作没有目的地。而 ②③ 本身是纯搬运（HOW_TO_CONTINUE 3.3），
`sort | md5sum` 可校，零行为风险。

⚠️ `minimum_image_displacement_nm` 进 ③ 时注意：主线三处薄包装 + 
`outer_lambda_neural_basis:644` 那份独立实现**不是重复**（HOW_TO_CONTINUE ❸ 撤回说明），
③ 只收那一份真实现，四个调用点原样不动。

---

## 7. L2 整体决策定稿（2026-09-19，用户实测）

对 lambda_path/charging/membrane/boresch/system_build 五域咬合簇做了抽取模拟：

| 场景 | 引用 | 有向边 | 互环 |
|---|---|---|---|
| 现状 | 188 | 17 | 8 对 |
| ① 抽 L0 infra | 169 | 17 | 8 对 |
| ② 再抽 L1 常量契约域 | 72 | 17 | 8 对 |
| ③ 再抽 L1 拓扑助手域 | 30 | 15 | **6 对** |

- 抽取消掉 **84% 的引用，但消不掉环**；常量占 52%（97 次）。
  **口径教训：引用次数（流量）和环（拓扑）是两件事，以后不许拿引用次数当进度指标。**
- 残留 6 对全是 1–3 次的单函数互引；查归属，**5 对是机械切分切歪的**：
  - `unwrap_boresch_anchors_nm` 落在 charging/ ← Boresch 几何助手，应归 boresch
  - `_build_platform_properties` 落在 boresch/ ← 平台助手，与 Boresch 无关
  - `resolve_environment_type` 落在 membrane/ ← charging 和 lambda_path 都要用，不专属膜
  - `normalize_protein_residue_name` 落在 lambda_path/
  - `_periodic_box_volume_nm3` 落在 lambda_path/ ← 纯几何
  改归属后四对直接消失，第五对只剩一个方向也消失。
- 真耦合只剩 **1 对：charging ⇄ system_build** —— 按 Stage2RepairController 同一处置：
  **参数注入，不 import**。

**L2 最终决策：一次抽取（② 常量契约域 + ③ 拓扑助手域）+ 五处改归属 + 一处参数注入。**
不是「抽一个域让 L2 散开」。

硬约束：**②③ 必须在改归属之前**（否则「移入 L1 契约域」没有目的地）；
②③ 是纯搬运，`sort | md5sum` 可校，零行为风险。

B 阶段（§0–§7）至此可以结稿。
