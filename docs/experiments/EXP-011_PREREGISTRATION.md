# EXP-011 / WP-4B 预注册与验收协议

状态：`PREREGISTERED / SAMPLING_REQUIRED`（2026-08-02）。当前只完成实现、协议冻结和历史三 run 覆盖诊断；正式目标加权 PMF、独立 NVT、WP-5 和 production 均未开始。

机器可读协议为 `protocols/EXP-011_preregistration.json`，内部规范 JSON SHA-256 为
`d18a38706b5bcd5aa4c7d713e1c34aa9fd19512398b2582c793b2a097096c596`。命令会重新计算该哈希，协议被修改后将 fail-closed。

## 唯一研究问题

完整 complex vanishing window 0 MM expanded-mixture Hamiltonian 下，primary torsion
`[4591,4592,4593,4585]` 的周期 PMF，能否在整条 run 留一验证中生成稳定、可积且幅度受限的 cheap bias？

本实验不回答三臂 IBS、ESS/GPU-hour、端点 ΔG 或 production 收益。

## 数据契约

正式 PMF 输入的 `report_type` 必须为 `outer_lambda_exp011_target_samples`，并包含：

- `target_hamiltonian_id`；
- `temperature_kelvin`；
- `torsion_atom_indices`；
- 至少三条独立 run 的 `samples`；
- 每个 sample 的 `run_id`、`angle_degrees` 和显式 `log_target_weight`。

同一目标 Hamiltonian 的无偏样本也必须显式写 `log_target_weight: 0.0`。umbrella 或其他增强采样必须提供去偏后的目标 log weight。禁止把 EXP-010 atom-cut MACE 标签或没有目标态权重的任意混合轨迹当作 PMF 标签。

## 冻结工作流与硬门

1. 使用 24 个固定 15° 周期 bins 检查三 run 覆盖。
2. 每 run 至少 500 帧、周期有效样本数至少 25、占据至少 50% bins。
3. 每个 pooled bin 目标加权有效样本数至少 2；每 bin 至少由两条 run 各提供 3 个原始样本。
4. 每对 run 的 Bhattacharyya coefficient 至少 0.5；每 run 的三个 torsion basin 各至少 5 个目标加权有效样本。
5. 覆盖全部通过后，才比较冻结的 Fourier order 2/4/6，ridge 固定为 `1e-6`，不加 pseudocount。
6. 整条 run 留一的每 fold PMF RMSE 不超过 2.5 kJ/mol、相关系数不低于 0.8、barrier 误差不超过 3.0 kJ/mol。
7. 导出的 bias 固定为 `-0.5 * PMF`，peak-to-peak 不超过 12 kJ/mol；选择规则为合格候选中 holdout RMSE 最低，平手选低 order。
8. PMF 通过后仍只允许进入独立 cheap-CV NVT qualification；不得直接进入 WP-5。

禁止事后增加 Fourier order、给空 bin 加 pseudocount、随机拆帧或放宽门限。

## 可复现命令

历史三 run 覆盖诊断（复用已有精确 histogram 与周期统计低效）：

```bash
/home/ruigengji/mambaforge/envs/FEP/bin/python outer_lambda_neural_basis.py \
  exp011-coverage \
  --protocol protocols/EXP-011_preregistration.json \
  --manifest output/outer_lambda_slow_variable_screen/slow_variable_manifest.json \
  --topology output_lrc_fix/topology.cif \
  --screen-report output/outer_lambda_slow_variable_screen/hard_window0_run1/candidate_screen_v2.json \
  --screen-report output/outer_lambda_slow_variable_screen/hard_window0_run2/candidate_screen.json \
  --screen-report output/outer_lambda_slow_variable_screen/hard_window0_run3/candidate_screen.json \
  -o output/outer_lambda_exp011/coverage_report.json
```

未来目标加权数据通过覆盖门后：

```bash
/home/ruigengji/mambaforge/envs/FEP/bin/python outer_lambda_neural_basis.py \
  exp011-fit-pmf \
  --protocol protocols/EXP-011_preregistration.json \
  --dataset output/outer_lambda_exp011/target_samples.json \
  -o output/outer_lambda_exp011/pmf_fit_report.json
```

## 2026-08-02 覆盖结论

`output/outer_lambda_exp011/coverage_report.json` 的 `qualified_for_pmf=false`。失败门为：run 有效样本数、单 run 占据率、pooled bin 有效样本数、每 bin run 数、pairwise overlap 和各 basin 有效样本数。

- pooled 空 bin 为 11、17–22，共 7/24；
- 三 run 周期有效样本数为 141.41、6.62、3.76；
- run1 占据率仅 0.417；
- 最低 pairwise Bhattacharyya coefficient 为 0.328（run1/run2）；
- run1 未观察到 gauche-plus，run2/run3 的多个 basin 有效样本数远低于 5。

结论固定为 `collect_restrained_or_enhanced_sampling`。当前不存在可验收 PMF 或可导出的 production candidate。
