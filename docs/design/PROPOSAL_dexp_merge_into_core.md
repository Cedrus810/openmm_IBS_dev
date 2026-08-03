# 提案：把 `dexp_NEW.py` 合并进 `abfe_core.py`，并隔离退役的 Orb 拟合代码

日期：2026-07-29
状态：**提案，未实施**。不改默认势（生产保持 `potential = softcore`）。
关联：[`../experiments/DEXP_KERNEL_PHYSICS_ISSUES.md`](../experiments/DEXP_KERNEL_PHYSICS_ISSUES.md) §1 / §2 / §6.7 / §12

---

## 1. 结论

**能合，但不能照原样合。**

`dexp_NEW.py` 里的物理一行都不新。`DEXP_KERNEL_PHYSICS_ISSUES.md` §12 记的实现位置就是
`abfe_core.py::DEXPSurrogatePotential`（表达式生成）+
`SurrogateSystemBuilder.build_surrogate_system`（把去耦前的原始 sigma/epsilon 喂给
`CustomNonbondedForce` 的 per-particle 参数），§2.2 的结论是「这一步**没有争议**，无论最终
alpha,beta 取什么值都应该保留」。`dexp_NEW.py` 独有的只有两样：一个更严的校验器
`DEXPProductionConfig`，和一个生成 System XML + manifest 的 CLI。

而它的配置面是错的：`DEXPProductionConfig` 把 `cutoff_distance` / `switch_width`
（属于**软化力**）和 `sigma_elec`（属于 **Gaussian-Coulomb 静电**）当成 DEXP 自己的字段，
还开了 `--cutoff-nm` / `--switch-width-nm` / `--sigma-elec-nm` 三个 CLI 开关。
照原样合并等于把这个错误框架搬进 core。

**DEXP 只是替代 LJ 的一个解析函数形式，它自己的可配置面只有 `alpha_vdw` / `beta_vdw`。**

这些字段被塞进 `DEXPSurrogatePotential` 是历史包袱：退役 Orb fitter 的返回字典
（`abfe_core.py:925-950`）一次吐出 `alpha_vdw/beta_vdw/sigma_elec/switch_width/cutoff_distance`
五个键，加 `fitting_success/final_cost` 和一堆 `diagnostic_*`；而 `from_dict` 照单收下它
认识的五个、静默丢掉其余。

## 2. 目标

- core 里只留一个 DEXP 定义：解析核，只吃 `alpha_vdw`/`beta_vdw`。
- cutoff/switch 归软化；`sigma_elec` 归静电。
- 退役的 Orb 拟合代码标记退役，搬进独立文件，生产不得 import。
- `dexp_NEW.py` 消失。
- **当前基线逐位不变**：ΔG_bind = −5.54 ± 0.60 kcal/mol（复合物腿 181.00 / 溶剂腿 157.84）
  是 softcore 跑出来的，本次改动不得让它移动。

## 3. 当前事实（已逐条对源码核对）

| 事实 | 位置 |
|---|---|
| `dexp_NEW.py` 不在生产链上：全仓只有一处 import 它 | `tests/test_dexp_new_production.py:10`（`abfe_core.py:683` 只在报错文案里提到） |
| DEXP 是休眠路径：07-29 生产跑 `potential_type = softcore`、`dexp_params = None` | `output_lrc_fix/final_results.json`、`abfe_config.json` |
| 解析核表达式（真正的 DEXP） | `abfe_core.py:590` `DEXPSurrogatePotential` |
| `alpha>beta>0` 已在 core 校验 | `abfe_core.py:631-637` |
| `from_dict` 只挑 5 个已知键、**静默丢弃未知键**、不查 `isfinite`、不查 `switch<cutoff`、不查 `sigma_elec>0` | `abfe_core.py:672-693` |
| cutoff/switch 实际是**软化力**的壳：dexp 读参数得 0.70/0.20（switch 0.50），否则硬编码 1.2/1.0 | `ibs_engine.py:2386-2399` |
| `sigma_elec` 实际只喂 **Gaussian-Coulomb**：`gamma_eff = 1/(√2·sigma_elec)` | `abfe_core.py:2686-2689` |
| ⚠️ Gaussian-Coulomb 的 shifted-force cutoff **也去读 `surrogate_potential.cutoff_distance`** | `abfe_core.py:2693` |
| 文档明写这两套规则「完全独立，不能混用」 | `DEXP_KERNEL_PHYSICS_ISSUES.md:527` |
| 生产 DEXP 路径**没有** minimum-image 检查（`abfe_core.py:3820` 是溶剂盒的，`abfe_pipeline.py:463` 是 Boresch 位移的，都不是这个） | — |
| 退役拟合代码 4 个类 + 1 个模块级函数 | `abfe_core.py:695` `Orbv3SurrogateFitter`、`:2772` `OrbScanner`、`:3033` `Orbv3SurrogatePipeline`、`:4929` `Orbv3DEXPFittingPipeline`、`run_orbv3_dexp_fitting` |
| 研究 harness 从 core import 这些类 | `dexp_experiment.py:97-110` |
| 生产 CLI 仍挂着拟合入口 | `runabfe.py:1892-1893`、`:2469-2484`；`abfe_pipeline.py:1497` `fit_dexp_parameters` |
| **fail-open**：`--potential dexp` 但 `--dexp-params` 路径不存在 → 静默 `dexp_params=None` → `from_dict({})` → 全默认值开跑，不报错不警告 | `runabfe.py:3503-3510` + `abfe_pipeline.py:274` |

## 4. 改动

### 4.1 `abfe_core.py`：DEXP 契约收窄到 alpha/beta

- `DEXPSurrogatePotential.__init__` 只留 `alpha_vdw=14.0, beta_vdw=5.0`，删掉
  `sigma_elec/switch_width/cutoff_distance` 三个参数与属性。保留现有 `alpha>beta>0`
  校验，补 `math.isfinite`（现在 `inf > 5 > 0` 为真，会漏过）。
- `from_dict` 改为**先拒未知键**再构造（沿用 `dexp_NEW.DEXPProductionConfig.from_mapping`
  的写法：`unknown = sorted(set(raw) - allowed)` → raise）；保留现有 `LEGACY_FIT_KEYS`
  fail-closed 分支，并把 fitter 那批键（`fitting_success`/`final_cost`/`diagnostic_*`/
  `optimizer_*`）一并列入拒绝名单。
- 新增两组**模块级常量**，数值全部保持现状：

  ```python
  DEXP_VDW_CUTOFF_NM       = 0.70   # 软化力的壳，归 softcore
  DEXP_VDW_SWITCH_WIDTH_NM = 0.20
  GAUSS_COUL_SIGMA_NM      = 0.10   # 静电，归 Gaussian-Coulomb
  GAUSS_COUL_CUTOFF_NM     = 0.70   # 见第 5 节「必须留意」
  ```

- `SurrogateSystemBuilder`：`sigma_gauss_nm` 默认取 `GAUSS_COUL_SIGMA_NM`，不再
  `getattr(surrogate_potential, "sigma_elec", ...)`；`:2693` 的 `rc_nm` 改读
  `GAUSS_COUL_CUTOFF_NM`，不再读 `surrogate_potential.cutoff_distance`。
- 把 `dexp_NEW._validate_minimum_image`（三斜正确的 plane-spacing 判据
  `h_i = V/|a_j×a_k|`）搬进 core 作模块级函数，并在 `build_surrogate_system` 里调用。

### 4.2 `ibs_engine.py`：软化力读自己的常量

`:2388-2389` 改读 `DEXP_VDW_CUTOFF_NM` / `DEXP_VDW_SWITCH_WIDTH_NM`，不再
`resolved_params["cutoff_distance"]`；保留 `cutoff > switch > 0` 断言。`else` 分支的
`1.2, 1.0` 一并提成命名常量（`SOFTCORE_CUTOFF_NM` / `SOFTCORE_SWITCH_NM`），消掉裸魔数。

### 4.3 新文件 `dexp_退役.py`：隔离退役的 Orb 拟合

文件头写明：**已退役，与当前生产方法无关，生产代码不得 import 本文件**；附退役理由
（`DEXP_KERNEL_PHYSICS_ISSUES.md` §1 的两条：学习信号是噪声，chi2≈9.74/dof=8，与纯噪声
无法区分；全 ligand-environment 共用一套全局形状，丢掉了原始 LJ 组合律本来就有的
pair-specific σ_ij/ε_ij 信息，比 LJ 本身还退化）。

搬入 `Orbv3SurrogateFitter`、`OrbScanner`、`Orbv3SurrogatePipeline`、
`Orbv3DEXPFittingPipeline`、`run_orbv3_dexp_fitting`，从 `abfe_core.py` 删除。
`dexp_experiment.py:97-110` 的 import 改指 `dexp_退役`。

⚠️ 文件名含非 ASCII，`import dexp_退役` 在 Python 3 合法，但本机 NFS / 工具链未实测；
若有阻碍则退回 `dexp_retired.py`，在文件头中文标注「退役」。

### 4.4 生产入口：去掉拟合 + 堵 fail-open

- 删 `runabfe.py:1892-1893` 的 `--save-dexp` / `--fit-dexp` 与 `:2469-2484` 的分支；
  删 `abfe_pipeline.py:1497` 的 `fit_dexp_parameters`。
- `runabfe.py:3503-3510` 改 fail-closed：`potential == "dexp"` 时 `--dexp-params` 缺失或
  路径不存在 → `raise`，不再静默落到 `None` → 全默认值。与 ATT-24「显式 config 静默降级
  已修」同类，那批漏了这一处。

### 4.5 删 `dexp_NEW.py`，测试改指 core

`tests/test_dexp_new_production.py` 现有 6 条测试全部保留、改 import，其中三条调整断言对象：

| 测试 | 改成 |
|---|---|
| `test_new_contract_contains_only_pair_specific_controls` | 断言契约**只有** alpha/beta |
| `test_new_production_rejects_box_smaller_than_twice_cutoff` | 打 core 的 minimum-image 函数 |
| `test_ibs_dexp_uses_dexp_cutoff_and_switch_not_softcore_defaults` | 断言软化力读的是 `DEXP_VDW_*` 常量 |

新增一条：未知键必须 raise。

### 4.6 文档

`DEXP_KERNEL_PHYSICS_ISSUES.md` §12 代码位置索引更新；`docs/TODO.md`「当前决策」里
「拆出 `dexp_NEW.py` 生产入口」那句改写；`docs/status/README_STATUS_SNAPSHOT_2026-07-29.md`
若提到则同步。

## 5. 必须留意

**`GAUSS_COUL_CUTOFF_NM` 必须先取 0.70。** `abfe_core.py:2693` 现在让静电 shifted-force
cutoff 复用 vdW 的 `cutoff_distance`。按 `DEXP_KERNEL_PHYSICS_ISSUES.md:527`，二者本该是
两套完全独立的距离/开关规则；但**一旦给静电换一个不同的数值，Gaussian-Coulomb 的势能面
就变了**。本次只做归属拆分、不动物理：两个常量都先设 0.70，让落盘数值逐位不变。
静电 cutoff 该不该独立取值，单独立项验证，不在本次范围。

`--dexp-params` 的 JSON 契约会收窄（只接受 alpha/beta），旧的 `dexp_fitted_params.json`
（5 键 + 拟合诊断）将被 fail-closed 拒绝。这是**预期行为**；DEXP 是休眠路径，无生产缓存受影响。

## 6. 验证

1. `./tests/run_offline_tests.sh` 全绿（= `pytest -m "not needs_gpu"`；当前 22 个测试文件
   没有一个打 `needs_gpu`，所以等于跑全套）。
2. **基线不动证明（最关键）**：softcore 路径不得有任何数值变化。
   - 核对 `_create_softcore_force` 的非 dexp 分支仍取 1.2/1.0；
   - 用 `output_lrc_fix/` 现有 `u_kn` 跑 `--analyze-only` 或
     `tools/diagnostics/diagnose_estimator_matrix.py`，确认复合物腿 181.00 /
     溶剂腿 157.84 / ΔG_bind −5.54 逐位不变。
3. DEXP 路径最小烟测：`tests/smoke_test_dexp_baseline.py`，加二原子解析断言
   `U(r0) = −eps_ij`、`U''(r0) = alpha·beta·eps/r0²`（现有测试已覆盖，确认仍过）。
4. `python -c "import dexp_退役"` 可导入；且
   `grep -rn "dexp_退役" runabfe.py abfe_pipeline.py ibs_engine.py abfe_core.py`
   必须**零命中**（生产不得依赖退役文件）。
5. fail-closed 验证：`--potential dexp` 不给 `--dexp-params` → 必须报错退出；
   给一个含 `fitting_success` 的旧 JSON → 必须报错。
