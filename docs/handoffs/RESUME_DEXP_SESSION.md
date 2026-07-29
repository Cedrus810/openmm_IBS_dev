# DEXP / MACE surrogate 工作恢复笔记

> 用途：下次接着干时快速恢复上下文。记录到 2026-07-10 本轮结束（上一轮 2026-07-08）。
> 体系：`Atenolol-rank11` ABFE + MACE(mace-off24-medium) 局部相互作用的 DEXP surrogate 修正。
> 2026-07-10 本轮做了什么，见 §8（switch 修复重跑 + 10ns 诊断）和 §9（坐标/环境一致性 bug + r0 实验，本轮第二部分，比 §8 更重要）。§4/§5 已按本轮结果更新，其余章节（§0-§3, §6-§7）内容仍然有效，未改。
> **当前最重要的单条结论（写在最前面，别漏看）**：§8 的地板判据"未过地板"是被两个 bug 污染的假结论。修完后（§9.1/§9.2）1ns 上反转成"DEXP 通过地板"（形状 RMSE 5.88 vs MM 7.46）。10ns 版本的确认还没跑完，见 §9.3 待办。

---

## 0. 一句话核心哲学（本次确立，别再退回去）

**不要问 "DEXP 像不像 MM"；要问 "DEXP 跑出来的世界，MACE 认不认"。**
- MM 只是**地板/null baseline**，不是真理（MM 也漏色散等）。
- 逐帧 ΔE 的方差（σ≈25 kJ/mol ≫ kT=2.5）是**学不到的噪声**——学习型 RBF 上界证明了（train R²=0.51 → holdout R²=0.008）。
- 所以能判的只有**系综/平均层面**：⟨δ⟩(s) 均值剖面、形状。逐帧 parity / R² 不是验收标准。
- **重加权（exp(-δ/kT)）对本体系结构上就塌**（σ(δ)≫kT，ESS≈1）——无论 MM→MACE 还是 DEXP→MACE。重加权 PMF **拿不到**。
- **三者（MACE-local / Gaussian+DEXP / MM）不共享能量零点**：绝对偏移 C **不可跨势比较**；只能比零点无关的量（形状 / 相对同一物理参考态 / 力 / PMF 形状）。C 只通过"态与态之间的 ΔC（同一套分解内）"进 ΔG，绝对值永不进。

---

## 1. 评价框架（A–F，取代旧的逐帧 R² 记分卡）

| 指标 | 含义 | 怎么算 | 现状 |
|---|---|---|---|
| A | MD 稳定 | 生产轨迹不塌/不 NaN | 待正式重跑确认 |
| B | MACE relabel 合理 | DEXP 帧单点 MACE → ⟨δ⟩(s) within-SEM（零点无关） | harness 已就绪 |
| C | 1D PMF (min_dist) | DEXP-world 直方图 PMF；MACE PMF 不可得(重加权塌) | 只能出 DEXP-world + ⟨δ⟩(s) |
| D | 2D PMF (min_dist + contact) | 同上 2D | 1ns 采不满，需 AWH/OPES |
| E | switch artifact | RDF 在 cutoff/switch 附近有无假峰 | **已修(见 §3)，待重跑验证** |
| F | 短程墙 0.2–0.3nm | 生产轨迹 min-dist 分布 / 过近帧数 | 老轨迹 0% 过近，过 |

---

## 2. 本次改的代码（都在 `dexp_experiment.py`，除 §3）

按时间顺序：

1. **offset 口径修复**：`predict_dexp_delta_e` 评估时加回 `offset_c0`；拟合后在**训练集**估 `C=⟨target⟩-⟨pairsum⟩` 写回 offset_c0（无泄漏）。`build_expression` 不消费 offset_c0，故只影响诊断、不改 MD 力。
2. **PMF matching 剔稀疏箱**：新增 `--fit-pmf-min-bin-frames`(默认10)，稀疏箱整箱剔除；保留逐帧原始 `accepted_delta_e_perframe`。
3. **holdout 判据重做**：验证改用**逐帧真值**（不是箱均值，避免循环验证）；新增 `evaluate_holdout_free_energy`（判据 A：系综均值 + ⟨δ⟩(s) 均值剖面 within-SEM）。
4. **删掉整套 FEP 重加权**（结构不可用）：`build_1d_pmf_battle_rows`、`_fep_free_energy`、`_bin_reweighted_pmf`、surface 里所有 reweighted PMF 列/图。
5. **6 个结果 bug**：
   - A: 删 surface 的 holdout 重复调用（去掉矛盾的第二个 RMSE），surface 只跑 all_accepted。
   - B: 新增 `fit_health`(degraded/ok)+`fit_health_reasons`，拟合完成行 + fit-only 结尾打印。
   - C: 均值剖面主判据改 within-SEM（RMSE 降级为参考）。
   - D: `qm_mm_offset_kjmol` 正名 `target_center_kjmol`（旧名保留别名）。
   - E: 删掉重复/错位的 `[2/4]` 打印。
   - F: 顶部 `warnings.filterwarnings` 静音 pymbar timeseries。
6. **relabel + 同帧 1D PMF harness**（新，`--relabel-traj` 模式，读现有 dexp_fitted_params.json 后退出）：
   - 函数：`relabel_trajectory_local`（逐帧 MACE 单点 + 分量 + DEXP 预测 + min-dist）、`same_frame_pmf_compare`（零点无关：锚最远箱的 ⟨δ⟩(s) within-SEM + DEXP-world 直方图 PMF + reweight ESS 告警）、`_filter_too_close`、`run_relabel_pmf`。
   - 参数：`--relabel-traj --relabel-baseline-traj --relabel-max-frames(300) --relabel-pmf-bins(24) --relabel-pmf-min-bin-frames(8) --relabel-min-dist-floor(0.12)`。
   - 产物：`relabel_dexp_1d_pmf.csv/png`、`relabel_mm_baseline_1d_pmf.csv`、`relabel_pmf_summary.json`。

## 3. abfe_core.py 改动（switch 伪影修复）

`SurrogateSystemBuilder.build_surrogate_system` 里 **Gaussian-Coulomb (coul_force)** 原来用能量 `setUseSwitchingFunction(True)` 截断 `erf(γr)/r` 长程尾 → `-S'(r)U` 假力，RDF 在 ~0.63nm（switch 区间 [0.50,0.70] 内、贴 cutoff 0.70）堆出假峰（delta_g 峰 +0.31）。
**改为 shifted-force**：`U_sf=U(r)-U(rc)-(r-rc)U'(rc)`，势与力在 cutoff 处都归零，**关掉 switching**。已数值验证 U 在 rc 平滑归零。DEXP vdW 核在 0.70 处 ~1e-9，其 switch 无害，未动。**修复后尚未在新轨迹上确认 0.63 峰消失。**

---

## 4. 当前结果快照（2026-07-10 更新，见 §8 详情；最新数字是 §8.4 的 10ns 版本，下面这段 relabel 数字是先做的 1ns 版本，两者结论一致只是统计力度不同）

**拟合（--fit-only --reuse-fit-labels）**：
- fit_target_mode = `mace_surrogate_residual` = E_MACE_local − E_gauss_coul（中心 target_center≈−192）。
- **`fit_health = degraded`**：`global_optimizer_failed, A_fit_clamped_for_repulsion, raw_core_not_repulsive, r0_pinned_at_bound(0.3000)`。← 拟合本身不健康（顶界/短程核靠夹 A）。仍未修（见 §5.2，用户决定暂缓）。
- 学习 RBF 上界：train R²=0.51 / holdout R²=0.008 → 逐帧无可泛化信号（DEXP 反而略优于 RBF，因不会过拟合）。
- holdout 均值剖面 within-SEM 3/4 箱。
- `offset_c0` 现在是 fitter 内部**联合拟合的一等参数**（见 §8.2），本次数值 = −26.716（与旧的"未修剪全训练集朴素均值对齐"诊断值几乎一致，非巧合——见 §8.2 解释，但语义上已不同、且下游不再依赖未修剪集合）。

**relabel（✅ 已用 switch 修复后的新生产轨迹重跑，1ns×2，各100帧，min-dist 仍是窄缝：DEXP 0.185–0.226nm / MM 0.157–0.199nm）**：
- E：0.63nm 附近的假峰确认消失（`le_rdf_comparison.csv` 该区间 delta_g 平滑变化，无孤立尖峰，见 §8.1）。
- F：0% 过近（<0.12nm），短程墙守住。
- 形状（锚最远 2 箱加权均值，零点无关，新默认估计量见 §8.3）：**DEXP RMSE 14.04 vs MM 5.27，within-SEM 1/2 vs 2/2 → 新轨迹上 DEXP 仍未过地板**。
  - 用旧的单箱锚点复核（`--relabel-shape-anchor-bins 1`）：DEXP 15.73 vs MM 8.08——**两种估计量给出同一个定性结论**（DEXP 未过地板），估计量选择不影响结论，仅影响绝对数值噪声大小（见 §8.3）。
- 重加权 ESS：DEXP 2% / MM 1%（都塌，弃，符合预期）。

---

## 5. 待办 / 下次从这里接

1. ~~[最优先] 用修好的 abfe_core 从头重跑生产轨迹~~ **已完成（2026-07-10）**，见 §8.1。E 的 0.63 假峰确认消失。
2. **fit_health=degraded**：r0 顶 0.30 下界、短程核不排斥、全局优化失败。用户观点：DEXP 公式自带墙、不会塌，先别修墙，跑出来让 MACE 判。已跑（§8.1 的新轨迹），MACE 判据是"未过地板"（见上）。是否要回头修 fit_health 取决于下一步：如果准备做 §5.5 的 AWH/OPES 拉伸采样，全接触态缺距离杠杆这个根因会被更宽的采样直接缓解，值得先跑 AWH 再决定要不要专门修。
3. ~~**C（offset）转正**~~ **已完成（2026-07-10）**，见 §8.2：`Orbv3SurrogateFitter.fit_parameters`（`abfe_core.py`）现在直接返回联合拟合的 `offset_c0`（用护栏 clamp 后真正施加到 OpenMM 的 (a,b,r0,A,B) + 拟合时实际用的 trimmed/weighted 帧集合解析求出），不再由 `dexp_experiment.py` 用未修剪全训练集重新估一遍。旧估计保留为 `offset_c0_naive_full_train_mean_diagnostic` 交叉核对用。**下游 ΔG 仍然只应使用态间 ΔC，绝对 C 永不用**——这条规则本身不因本次改动而变化，只是现在 C 的来源更严谨。
4. ~~**shape 估计量选择**~~ **已定（2026-07-10）**，见 §8.3：默认改为"锚最远 `--relabel-shape-anchor-bins`（默认2）个箱的逆方差加权均值"，并把锚点自身 SEM 传播进每箱 within-SEM 判据（`combined_sem = sqrt(bin_sem²+anchor_sem²)`），不再假装锚点无噪声。`--relabel-shape-anchor-bins 1` 可复现旧的单箱行为。用新旧两种估计量在新轨迹上交叉验证：结论一致（DEXP 未过地板），估计量只影响噪声大小不影响定性判断。
5. **[现在最优先] 有偏采样拉宽 min-distance 覆盖**：已经用 10ns 无偏 MD 验证过（§8.4），min-dist 卡在 0.15–0.23nm 是真实的局部势阱，不是采样时长不够，加长无偏 MD 不会自己拓宽。环境里目前**没有装 PLUMED / openmm-plumed，也没有 gmx**（AWH 是 GROMACS 原生的，OPES 是 PLUMED 的功能，这套体系目前完全是 OpenMM-native）。两条可选路径：(a) 装 PLUMED + openmm-plumed 插件用 OPES_METAD，需要新装依赖、可能要匹配 CUDA/OpenMM 版本编译，有安装风险；(b) 原生 OpenMM 伞形采样——用 CustomBondForce/CustomCVForce 对 min-distance 加谐振偏置，跑一串窗口（复用现有 `lambda_windows` 式的多窗口跑法），再用项目里已经在用的 pymbar MBAR 拼出沿 min-distance 的 1D PMF，零新依赖。用户还没最终拍板选哪条路，下次接着做之前先确认。
6. **主动学习闭环**：fit → DEXP-MD → relabel OOD 帧 → 并进训练集 refit → 循环。仍待 §5.5 先完成。

---

## 6. 运行环境 & 命令

- **Python**：`/home/ruigengji/mambaforge/envs/openmm_dev/bin/python`（有 numpy/openmm/mdtraj，MACE 经 openmmml，torch cuda 可用）。
  - ⚠️ 原 job 环境 `omm_torch_124` 已不存在；`omm_torch_126` 缺 mdtraj。用 `openmm_dev`。
- **GPU**：RTX 2080 Ti，11GB。跑 MACE 用 `--device cuda`（CPU 慢很多）。
- **拟合（不跑 MD，复用 MACE 缓存）**：
  ```
  python dexp_experiment.py --fit-only --reuse-fit-labels --learned-rbf-diagnostic --device cuda
  ```
- **完整 pipeline（fit + surrogate MD + baseline + 后处理）**：`python dexp_experiment.py`（默认路径见 DEFAULT_PATHS：traj=output/pre_equilibration.dcd, top=output/topology.cif, system=output/system_native.xml, ligand=output/ligand_indices.json, gmx_top=topol.top）。
- **relabel + 同帧 PMF（本轮新增，读现有参数）**：
  ```
  python dexp_experiment.py \
    --relabel-traj output/dexp_experiment/dexp_surrogate/traj.dcd \
    --relabel-baseline-traj output/dexp_experiment/original_baseline/traj.dcd \
    --relabel-max-frames 300 --relabel-pmf-bins 16 --device cuda
  ```

## 7. 关键文件

- `dexp_experiment.py` — 拟合 + 诊断 + relabel harness。
- `abfe_core.py` — SurrogateSystemBuilder / Orbv3SurrogateFitter。
- `output/dexp_experiment/dexp_fitted_params.json` — 拟合参数（含 fit_health、offset_c0）。
- `output/dexp_experiment/relabel_pmf_summary.json` — relabel 结果（当前是 switch 修复后的新轨迹 + 新锚点估计量）。
- `output/dexp_experiment/{dexp_surrogate,original_baseline}/traj.dcd` — 生产 / baseline 轨迹（✅ 已是 switch 修复后重跑的新轨迹，2026-07-10）。
- `output/dexp_experiment/rerun_logs/` — 2026-07-10 第一轮（1ns）重跑/relabel 的完整 stdout 日志，含 `production_rerun_*.log`（全流程重跑）、`refit_with_item3fix.log`（拿到新 offset_c0 的 fit-only 重跑）、`relabel_fresh.log` / `relabel_anchor1.log` / `relabel_final_default.log`（不同锚点设置下的 relabel 交叉验证）。
- `run_dexp_longmd.zsh` — PBS 提交脚本，跑独立目录里的更长无偏 MD 诊断（用户自己在计算节点跑的），当前配置 `--sim-ns 10.0`，可以改 `--sim-ns`/`--output-dir` 复用来跑别的时长。
- `output/dexp_experiment_10ns_diag/` — §8.4 的 10ns 诊断跑的完整输出（独立于 `output/dexp_experiment` 这套 1ns 基线），含 `relabel_10ns.log` 的 relabel 日志和 `relabel_pmf_summary.json`。

---

## 8. 本轮（2026-07-10）改动详情

### 8.1 用修好的 abfe_core 重跑生产轨迹（§5 旧第 1 项，已完成）

跑法：`python dexp_experiment.py --reuse-fit-labels --learned-rbf-diagnostic --device cuda --platform CUDA`（默认 `--sim-ns 1.0`，覆盖了旧的 `dexp_surrogate/traj.dcd` 和 `original_baseline/traj.dcd`）。

结果：
- 全流程正常跑完，无 NaN/崩溃（判据 A 过）。
- `le_rdf_comparison.csv` 里 0.55–0.72nm 区间（switch 区间 [0.50,0.70] 附近、旧版假峰在 ~0.63nm）：`delta_g_r` 现在是 −0.17 到 +0.09 之间的平滑变化，没有孤立尖峰。旧版报告的 "+0.31 尖峰" 确认消失（判据 E 过）。
- fit 阶段结果（a,b,r0,A,B、fit_health=degraded、holdout within-SEM 3/4）与旧轨迹时基本一致（同一套拟合帧、同一个 fitter，预期如此）。

⚠️ 踩坑记录（给以后的自己）：**先启动长跑任务，再编辑它依赖的源文件，编辑不会影响已经在跑的进程**——Python 在进程启动时就把整个模块读进内存了，之后改文件对这个进程无效。本轮先跑了全流程重跑（用的是修 §8.2/§8.3 之前的代码），之后才做 §8.2/§8.3 的代码改动，所以又追加跑了一次 `--fit-only --reuse-fit-labels` 补上新的 `offset_c0` 语义（MD 力本身不受影响，因为 `build_expression` 从不消费 `offset_c0`，所以不需要重跑 MD，只需重跑 fit 阶段）。以后同一次会话里如果要"改代码 + 跑长任务"，先改完代码再启动进程。

### 8.2 offset_c0 转正为联合拟合一等参数（§5 旧第 3 项，已完成）

问题定位：`dexp_experiment.py` 里原来在 fit 完成后，用**未经离群点修剪、未加权**的完整训练集重新算一遍 `offset_c0 = mean(target) - mean(pairsum)`，但 `Orbv3SurrogateFitter.fit_parameters`（`abfe_core.py`）内部实际优化用的是 median/MAD 离群点裁剪 + 均匀帧权重后的子集，且返回的 `A_fit` 可能被短程排斥护栏 clamp 过（此系统 `fit_health` 里的 `A_fit_clamped_for_repulsion` 就是真实发生的情况）。两处用的"training set"和"最终会施加到 OpenMM 的参数"并不是同一件事，`offset_c0` 应该基于后者算才对。

修复：
- `abfe_core.py::Orbv3SurrogateFitter.fit_parameters` 现在在护栏 clamp 之后，用 clamp 后的最终 `(a,b,r0,A,B)`、以及拟合时实际用的 trimmed+weighted 帧集合，解析算出 `offset_c0 = <target> − <pairsum>`（对固定形状参数，这就是让残差平方和最小的唯一最优常数，数学上等价于把 c0 当作与其它 5 个参数联合拟合的第 6 个参数），作为一等字段返回：`offset_c0`、`offset_c0_source="joint_fit_trimmed_weighted_post_clamp"`、`offset_c0_pred_center_kjmol`、`offset_c0_target_center_kjmol`。
- `dexp_experiment.py` 不再覆盖这个值；旧的"未修剪全训练集朴素均值对齐"算法保留为诊断字段 `offset_c0_naive_full_train_mean_diagnostic`，仅用于交叉核对，且只在 fitter 版本太旧、没有提供 `offset_c0` 时才当 fallback 使用。
- 本系统上验证：新旧两个值几乎相等（−26.7165 vs −26.7165，8 位小数才有差异）——说明这次修剪/clamp 对这批数据影响很小，不是一次会显著改变数值的修复，但语义上现在是对的、且以后换数据集/换配体时不会再有这个隐性不一致。
- **仍然成立、没有变化的规则**：下游 ΔG 组装只能用**态间 ΔC（同一分解内）**，绝对 C 永远不进 ΔG。`dexp_experiment.py` 目前没有把 DEXP 接进主链 ΔG 组装（`AUDIT_STATUS.md` 明确把 `dexp_experiment.py` 排除在主链审计范围外），所以这条规则目前是"未来集成时必须遵守的约束"，不是"当前有个 bug 在违反它"。

### 8.3 relabel 形状剖面锚点估计量（§5 旧第 4 项，已定）

问题：`same_frame_pmf_compare` 原来锚"单个最远 min-dist 箱"做零点无关的形状比较，物理上唯一合法（去均值会把所有箱的信息混进零点，非物理），但把该箱的采样噪声整个传进了其它每一个箱的 `d_rel`，本系统数据量小时这个噪声不小。

修复：
- 新增 `--relabel-shape-anchor-bins`（默认 2）：锚点改为最远 N 个箱的**逆方差加权均值**（用更多帧稀释锚点自身噪声），并把锚点的 SEM（`shape_anchor_sem_kjmol`）传播进每个非锚点箱的 within-SEM 判据：`combined_sem = sqrt(bin_sem² + anchor_sem²)`，不再假装锚点零噪声。`--relabel-shape-anchor-bins 1` 精确复现旧的单箱行为。
- 在 §8.1 的新轨迹上交叉验证（100+100帧，`--relabel-pmf-bins 24`，实际落在窄缝区间后剩 4 个可信箱）：
  - 新默认（锚 2 箱）：DEXP RMSE=14.04 / within-SEM 1/2；MM RMSE=5.27 / within-SEM 2/2 → 未过地板。
  - 旧行为（锚 1 箱，`--relabel-shape-anchor-bins 1`）：DEXP RMSE=15.73 / within-SEM 2/3；MM RMSE=8.08 / within-SEM 2/3 → 未过地板。
  - **结论一致**：无论用哪种估计量，DEXP 在这条新轨迹上都没有过 MM 地板。估计量选择只改变了噪声大小（新估计量数值更小、更稳），没有改变定性判断，说明这一项不是导致"未过地板"结论的关键变量——真正的瓶颈是 §5 第 5 项（采样范围太窄）。
- 磁盘上当前的 `relabel_pmf_summary.json`/`relabel_dexp_1d_pmf.csv` 是用默认锚点（2箱）跑的最终版本。

### 8.4 10ns 无偏 MD 诊断结果（2026-07-10，用户自己在计算节点跑的）

用 `run_dexp_longmd.zsh`（PBS 脚本）在独立目录 `output/dexp_experiment_10ns_diag` 跑了 10ns×2 条腿（`--sim-ns 10.0`，其余同 §8.1 的重跑），验证"1ns 结果是不是纯粹因为跑太短/箱子太细看不清"这个问题。跑完后又在同一目录上做了 relabel（500 帧/条腿，`--relabel-pmf-bins 24`）。

**结果对比（1ns@100帧 vs 10ns@500帧，同一套 switch 修复后的代码）**：

| | 1ns (100帧) | 10ns (500帧) |
|---|---|---|
| DEXP min-dist 范围 | 0.185–0.226 nm | 0.173–0.227 nm |
| MM min-dist 范围 | 0.157–0.199 nm | 0.151–0.204 nm |
| 可信 bin 数（DEXP/MM） | 2 / 2 | 14 / 13 |
| 形状 RMSE（DEXP） | 14.04 kJ/mol | **4.60 kJ/mol** |
| 形状 RMSE（MM） | 5.27 kJ/mol | **2.72 kJ/mol** |
| within-SEM（DEXP） | 1/2 (50%) | 11/14 (79%) |
| within-SEM（MM） | 2/2 (100%) | 8/13 (62%) |
| 地板判据 | 未过 | 未过 |

**结论（两点都成立，不矛盾）**：

1. **1ns 的"未过地板"结论里，相当一部分是小样本噪声被放大**：可信 bin 数从 2 个涨到 14/13 个，DEXP/MM 的 RMSE 都跟着大幅下降（DEXP 14.04→4.60，MM 5.27→2.72）。这印证了用户的猜测——1ns+24箱确实统计力度太弱，14.04 vs 5.27 这两个数字本身不是很可信的定量差距。
2. **但定性结论没有变，min-dist 范围也没有变宽**：DEXP min-dist 依然卡在 0.17–0.23nm，MM 依然卡在 0.15–0.20nm，10 倍的无偏 MD 时长几乎没有让这条窄缝变宽（对比 §8.1 的 1ns 数字，两个范围几乎重合）。这证实了 §8.1 末尾的保留意见：这大概率是一个真实的局部势阱，无偏 MD 不会自己爬出来，需要真正加偏置力才能拓宽覆盖。而且 DEXP 的形状 RMSE（4.60）依然比 MM（2.72）大，只是差距比 1ns 时看起来的小、也更可信了。

**净效果**：更长的无偏 MD 把"DEXP 未过地板"这个结论从"基于 2 个噪声很大的 bin 的弱证据"变成了"基于 14 个统计上更扎实的 bin 的、差距更小但依然成立的结论"。它解决了统计力度问题，但没有解决采样范围问题——两者是独立的两件事。下一步要拓宽到 DEXP 真正的拟合窗口 [0.20,0.45]nm，仍然需要 §5 第 5 项的有偏采样（伞形采样 / AWH / OPES），不是靠继续加长无偏 MD。

### 8.5 净结论（⚠️ 已被 §9 部分推翻，见下）

三项代码修复（switch 伪影、offset_c0 一等化、锚点估计量）+ 一次 10ns 无偏 MD 诊断都做完了，结论稳定收敛到同一点：**DEXP 在当前采样下仍未过 MM 地板**（形状 RMSE 更大、within-SEM 命中率更低），但差距比最早 1ns 时看起来的小很多，而且现在是建立在统计上扎实得多的 14 个 bin 上（§8.4），不是代码 bug 或估计量选择造成的假象。真正的瓶颈是采样范围：min-dist 一直卡在 0.15–0.23nm 这条窄缝里，10 倍时长的无偏 MD 没能拓宽它，说明这是个真实的局部势阱，不是"跑久一点自己就出来了"能解决的。下一步真正该做的是 §5 第 5 项（伞形采样 / AWH / OPES），把 min-dist 从现在的窄缝拉到 DEXP 真正的拟合窗口 [0.20,0.45]nm 乃至更宽（0.6+nm），才能在一个有真实距离杠杆、更宽 CV 范围的采样上重新问"MACE 认不认 DEXP 的世界"这个问题；继续在这条窄缝里调整拟合细节、估计量或加长无偏 MD 时长，预期都不会再改变这个结论。

> **2026-07-10 晚间更新**：上面这个"结论稳定收敛"是错的——不是因为物理变了，是因为判据本身有两个 bug（见 §9.1/§9.2）。修完之后 1ns 上的地板判据从"未过"变成"通过"。§8 的 MD 重跑结论（switch 伪影确认消失、MD 稳定、10ns 没让 min-dist 变宽）仍然全部有效，只有"地板判据未过"这一条被推翻。

---

## 9. 用户 code review 揪出的两个真 bug + r0 实验（2026-07-10 晚间，比 §8 更重要）

### 9.1 坐标不一致：分箱/判据用的"距离"和 DEXP 公式实际依赖的"距离"不是一回事

用户直接指出（并给了具体行号）：

- DEXP 的拟合/预测只用 `dists[(dists>=fit_r_min)&(dists<=fit_r_max)]`（即 `[0.20,0.45]nm`）这个子集（`dexp_experiment.py` 原 1328/4271 行）。
- 但 fit 阶段 PMF matching 分箱、relabel 阶段"探索了多远"，一直在用**全原子（含H）最近距离** `dists.min()`（原 1405/1531/4288 行）——这是完全没做 `[fit_r_min,fit_r_max]` 过滤的量。

实测验证（10ns 诊断数据）：`fit_frame_diagnostics.csv` 里 `used_for_fit=1` 的 500 帧中 **497 帧（99.4%）全原子最近距离 < 0.20nm**；`fit_pmf_matching_profile.csv` 的可信箱中心全落在 0.158–0.191nm——也就是说，之前一直在用一个 DEXP 公式基本"看不见"的坐标去给 DEXP 打分、去分箱拟合目标。

**修复**（`dexp_experiment.py`）：
- fit 阶段：`accepted_min_dist`（喂给 PMF matching 和留出集自由能判据 A 的分箱坐标）改成用 `valid_dists.min()`（已经过 `[fit_r_min,fit_r_max]` 过滤的最近距离），不再用未过滤的 `row["min_le_distance_nm"]`。新增诊断列 `min_valid_le_distance_nm` 到 `fit_frame_diagnostics.csv`（保留旧的 `min_le_distance_nm` 列不删，只是不再拿它分箱）。
- relabel 阶段：`relabel_trajectory_local` 现在同时返回 `min_dist`（全原子，只用于"过近/碰撞"判据 F）和 `min_dist_valid`（限定在 `[fit_r_min,fit_r_max]`，`same_frame_pmf_compare` 的分箱坐标改用这个）。

### 9.2 环境原子集合不一致：DEXP/MM/fit 三边 MACE 局部分解用的不是同一套原子

用户指出：relabel 里 DEXP 轨迹和 MM 轨迹各自按"自己那条轨迹最后一帧"重选环境原子，跟 fit 阶段用的环境原子集合也不是同一套。10ns 数据实测：fit 用 env=255，DEXP relabel 用 env=219，MM relabel 用 env=242——三边都不一样。MACE 的局部能量分解只对固定的原子子集有意义，环境集合一变，"局部能量"这个量本身就不可比了，形状 RMSE 的高低会掺进"环境定义差异"这个跟物理无关的噪声。

**修复**：新增 `_load_fixed_env_indices(output_dir)`，从 fit 阶段留下的 `fit_label_cache_meta.json` 里读固定的 255 个环境原子索引；`run_relabel_pmf` 现在把这套固定索引通过 `relabel_trajectory_local(..., env_idx_override=...)` 传给 DEXP 和 MM 两次 relabel 调用，两边强制用同一套环境原子。找不到缓存文件时回退成旧行为（各自重选）并打印警告，不静默。

### 9.3 修复后的结果：1ns 地板判据反转，10ns 待确认

用同样的 1ns 生产轨迹（§8.1 的产物，switch 修复后），只是换了判据代码，重新 fit + relabel：

| | 1ns，修复前（§8 数字） | 1ns，修复后（§9） |
|---|---|---|
| DEXP 形状 RMSE | 14.04 | **5.88** |
| MM 形状 RMSE | 5.27 | **7.46** |
| 地板判据 | 未过 | **通过** |

DEXP 和 MM 的 RMSE 都变了（不只是谁大谁小换了），说明这不是噪声巧合，是判据本身之前系统性地对两条轨迹做了不对等的处理。

**⚠️ 待办：10ns 版本还没跑完**。已经准备好 `run_dexp_relabel_10ns.zsh`（PBS 脚本，在 `output/dexp_experiment_10ns_diag` 上先 `--fit-only --reuse-fit-labels` 重新算一遍（会用新坐标重建 `fit_pmf_matching_profile.csv`，样本本身不变，很快），再重新 relabel 500+500 帧）。我自己在共享节点上跑到一半被用户叫停（那次没跑完，没有产出脏数据，`output/dexp_experiment_10ns_diag/fit_frame_diagnostics.csv` 等 fit 产物是干净的、已经用新坐标算过），用户会自己去计算节点跑。**下次接着做的第一件事：跑 `qsub run_dexp_relabel_10ns.zsh`，确认 500 帧统计量下地板判据是不是也翻成"通过"**——1ns 只有 2-3 个可信 bin，这个反转需要 10ns 的统计力度才能真正确认，不能只信 1ns 这一次。

### 9.4 r0 实验：试了三版，全部说明"不要调 r0"，已恢复

背景：§9.1 修好坐标后，仍然剩下一个问题——DEXP 的 OpenMM 力在 MD 里对所有距离都算（没有在 `fit_r_min` 处截断），但拟合数据几乎全在 0.20-0.227nm 这一条窄缝里，`r0_vdw` 反复被顶在下界 0.30（`abfe_core.py::Orbv3SurrogateFitter` 里硬编码的边界 `(0.30,0.38)`）。一开始怀疑是这个边界设错了、该往下调以贴近实际采样距离。三次实验：

1. **只降 `--fit-r-min` 到 0.13，不动 r0 边界**：A/B 振幅塌到接近零（优化器发现"没有信号"），r0 还是顶在 0.30（边界跟 `fit_r_min` 完全脱节，降数据窗口没用），holdout 均值剖面 Pearson r 从 +0.34 变成 **-0.91**（剖面趋势反了）。比不改还差。
2. **`fit_r_min=0.13` 同时把 r0 边界放宽到 `[0.12,0.30]`**：r0 真的动了，落在 0.231（看起来合理），A/B 也变成"正常大小"的数（5.7/3.9，不再靠安全阀硬撑）。但 holdout RMSE 直接炸到 **9714 kJ/mol**（正常应该是二十几），`offset_c0=-12815`。根因：DEXP 不是只算最近一对原子，是把 `[fit_r_min,fit_r_max]` 内**所有**配体-环境原子对加总（实测每帧 666-906 对）。r0 一旦挪进真实接触密集的区间，会有几十上百对原子同时落在双指数陡峭拐点附近，单对稍微陡一点，乘上几百对直接爆炸。`fit_health` 的现有判据完全没抓到这个（还报"ok"），因为它只检查几个特定失败模式，没检查"预测量级是否离谱"——**这是 `fit_health` 诊断本身的一个待修 gap**，先记着，本轮没修。
3. **`fit_r_min` 保持默认 0.20，只把 r0 边界温和放宽到 `[0.25,0.38]`**：单对能量峰值降到 773 kJ/mol（远没有第2版夸张），但 holdout RMSE 仍然 **875 kJ/mol**，within-SEM 从 3/4 掉到 0/4。日志显示两次（含原始基线）差分进化全局搜索其实都失败（`global_optimizer_failed`），靠局部 least_squares 从初始猜测收尾；边界一变、初始猜测的相对位置跟着变，局部优化掉进了完全不同、明显更差的局部极小值。**这说明当前这套优化流程（DE 常年失败 + 单起点局部精修）本身鲁棒性不够，边界稍微一动结果就可能大幅变差**，跟"该不该调 r0"这个物理问题本身无关。

**关键认知转折（查文献确认）**：r0（文献里叫 r_m）在双指数 vdW 势里是"排斥项和吸引项刚好平衡"的那个参考距离，类似 LJ 势的 sigma/R_min，对常见重原子对通常就是 3-4 Å（0.30-0.38nm）——查到的是 Zhang 等人 *A double exponential potential for van der Waals interaction*（AIP Advances 2019，[PMC6555761](https://pmc.ncbi.nlm.nih.gov/articles/PMC6555761/)），α/β 典型值例子 (17.470, 4.099)。**这个量应该由原子种类决定、基本固定，不应该跟着"这次模拟配体恰好离多近"去调**。配体现在待的地方（0.15-0.23nm）比 r0 更近，是正常物理（排斥墙区），不是"r0 离数据太远"的信号。

**结论**：三次实验 + 文献核对，一致指向同一件事——**r0 边界 `[0.30,0.38]` 不用动，之前想下调是方向性错误**。已把 `abfe_core.py` 恢复到原始边界，`dexp_fitted_params.json` 重新用默认参数（`--fit-r-min 0.20`，恢复后的边界）拟合过，跟本轮验证"通过地板"时用的完全一致（RMSE=26.05，与 §9.3 的 1ns relabel 数字对应的拟合状态相同）。**下次不要再尝试调 r0 边界或 `fit_r_min` 来"扩大拟合窗口"**——真正该做的是 §9.5 讨论的"为什么可学习的信号这么少"，以及 §5 第 5 项的增强采样。

### 9.5 待讨论：为什么可学习的内容这么少（本轮结束时提出，下次接着想）

用户提出的问题，还没有定论，先把已知线索记下来：

1. **拟合数据本身来自 `pre_equilibration.dcd` 的最后 5ns，跟这次做的 1ns/10ns production 重跑无关**——production 重跑得再长，也不会让 fit 阶段的训练数据窗口变宽，因为 fit 从来不读 production 轨迹。fit 的窄窗口（0.20-0.227nm）是 pre-equilibration 阶段本身（很可能也是普通无偏 MD）留下的，跟 §8.4 的"10ns 无偏 MD 没让窄缝变宽"是同一个根因在两个不同轨迹上的重复验证。
2. **窄窗口对"形状拟合"是双重打击，不只是噪声大**：(a) 帧数少、每箱统计误差大（这个 §8.4 已经证明可以靠更多帧缓解）；(b) 更根本的是，x=r/r0-1 的动态范围太窄时，双指数曲线在这段范围内和一条直线几乎没区别——A、α（陡峭度）这些参数彼此高度简并/不可辨识，数据给不出"到底该多陡"这个信息，这不是加帧数能解决的，是**动态范围**不够，不是**样本量**不够。
3. **可能还有一个更深的天花板**：项目最早（§0）就验证过，逐帧 ΔE 的方差 σ≈25 kJ/mol ≫ kT=2.5，而且更灵活的学习型 RBF 上界都只有 holdout R²≈0.008——这说明哪怕换一个更灵活的函数形式，纯"配体-环境原子对距离"这一类两体坐标本身可能就解释不了大部分逐帧方差（真正的方差可能来自三体+的协同效应，比如水分子网络重排、极化——这些不是任何纯 pairwise 加和模型能表达的）。这不代表 DEXP 没用（PMF-mean-matching 这条路本来就是绕开这个天花板、只学系综均值剖面，不学逐帧噪声），但回答"为什么可学的东西这么少"时，这个结构性上限和"窗口太窄"是两个独立的原因，得分开看：窗口太窄可以靠增强采样解决；两体模型的解释力上限，增强采样解决不了，只能通过换模型形式（比如加个三体项）或者接受"只学均值剖面"这个更谦逊的目标。
