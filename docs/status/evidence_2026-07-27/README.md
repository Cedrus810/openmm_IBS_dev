# 2026-07-27 诊断证据（V-06 / P0-10）

这三份 JSON 是 2026-07-27 排查 `endpoint_σ` 可信度、并最终定位到 **P0-10**
（Boresch 已提交平衡值陈旧）时产出的原始输出。原先只存在 `/tmp/sigma_diag/`，
重启即失；现归档进项目，作为 **V-06** 的证据。

对应的结果记录与完整叙述见
[`../RESULT_2026-07-27_atenolol_rank11.md`](../RESULT_2026-07-27_atenolol_rank11.md)。

> ⚠️ 这些证据描述的是**已作废**的那一轮采样（复合物腿用了错的 Boresch 平衡值）。
> 其中**关于估计器的结论仍然有效**（TI≈MBAR、渐近极限成立），
> 因为那些只取决于分析路径、不取决于采样的物理正确性；
> 而**关于物理量的数值（ΔG、⟨U_elec⟩）不可引用**。

## 文件

### `endpoint_sigma_diagnosis.json`
`diagnose_endpoint_sigma.py` 的输出。三部分：

- `A1_reproduce` —— 离线复现最终解作为控制实验。`total_delta_G = 145.90847168207642`、
  `total_error = 1.384443322336141`，与当时 `checkpoints/stage2_vanishing.json` 逐位一致
  （`_reproduced = true`）。**这里面含那份原运行从未落盘的逐段诊断**
  （`covariance_chain_segments`、`window_overlap_diagnostics`、`coverage_diagnostics`）——
  原因是 P1-15：rescue 分支绕过 `_run_ibs_stage`，落盘 `diagnostics={}`。
- `A2_repeat_measurement` —— 对同一 λ 区间 11→15 的两个独立估计
  （老 window 3 的 1M 步 vs 两个 rescue ensemble 各 250k 步）。
  `primary_z ≈ 0.89`（全链口径）/ 1.42（段口径），**z < 2 → 误差棒与实测漂移自洽**。
- `history_scan` —— `base` 序列 21 处不连续（`bias` 0 处），但真正用于估计 `g` 的
  `(u_kn[k] − bias)/kT` **0 跳变**。结论：轨迹不连续没有污染自相关估计。
  另经 `grep -c "触发回退\|灾难检测触发" pipeline.log` = 0 确认那些不连续
  **不是 P1-13**（生产灾难回退从未触发），而是跨进程续跑边界。

### `charging_linear_response.json`
`diagnose_charging_linear_response.py` 的输出，逐 λ 态的 `⟨ΔU⟩`。

**注意一处方法论修正**：脚本里那个"两点线性响应"判据（`½(⟨ΔU⟩₀ + ⟨ΔU⟩₁)`）
**不适用**于本例——σ(ΔU) 在 λ=1 端是 38.09、λ=0 端是 12.65，方差差 9 倍，
严重违反对称两点估计要求的等方差高斯前提，于是它在两条腿上都虚高 20–26 kJ/mol。
正确口径是对 `per_state_mean_dU` 做**梯形 TI**（`∫⟨ΔU⟩dλ`），结果与 MBAR 吻合到 1 kJ：

| | TI | MBAR |
|---|---|---|
| 复合物 decharging | 50.30 | 49.51 |
| 溶剂 decharging | 63.70 | 62.72 |

**这就是"估计器没问题"的依据**；偏差在系综，不在分析。

关键的纯系综量（不经任何自由能机器）：⟨ΔU⟩@λ=1 在水里 **180.92** kJ/mol、
口袋里 **144.04** kJ/mol —— 配体在口袋里静电上比在水里差 36.9 kJ/mol。

### `pose_drift.json`
`diagnose_pose_drift.py` 的输出，配体重原子相对输入 `solv_ions.gro` 的 RMSD。

| 轨迹 | 口袋骨架对齐 RMSD | 质心位移 |
|---|---|---|
| `pre_equilibration.dcd` 末帧（**无约束** 5 ns） | **0.60 Å** | 0.17 Å |
| `rebalance_traj.dcd`（**带 Boresch**） | **3.42 Å** | 2.46 Å |

**这是 P0-10 的判决性证据**：不加约束时 pose 稳得很，一加 Boresch 反而被拽走
3.4 Å —— 限制力锚在错误几何上，主动把配体拉离自己的构象。
（也因此证伪了先前"配体在预平衡里漂走"的假设。）

## 复现命令

```bash
source /home/ruigengji/mambaforge/etc/profile.d/mamba.sh && mamba activate openmm_dev

python diagnose_endpoint_sigma.py --run-dir <output> --out-dir <evidence>
python diagnose_charging_linear_response.py \
    --complex-meta <output>/charging_rerun/*/decharging/decharging_pme_u_kn.meta.json \
    --solvent-meta <output>/solvent_leg/decharging/decharging_pme_u_kn.meta.json \
    --complex-mbar-dg 49.508507666967176 --solvent-mbar-dg 62.71721848624468
python diagnose_pose_drift.py --run-dir <output> --also-traj <output>/rebalance_traj.dcd
```

⚠️ 这些脚本读的 `output_lrc_fix/` 已于 2026-07-27 18:16 清空，
上述命令需要新一轮运行的输出才能重跑。
