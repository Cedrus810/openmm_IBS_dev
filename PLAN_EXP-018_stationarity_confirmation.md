# PLAN-EXP-018：window 5 stationarity confirmation

状态：`SEALED / PROVENANCE_AUDIT_REQUIRED / MD_NOT_STARTED`  
日期：2026-08-10  
实验目的：只判断 EXP-017 的 window 5 split-half 漂移是否可复现，以及现有 MBAR 不确定度是否系统性低估重复间波动。

## 1. 与 EXP-017 的边界

EXP-017 已终止并登记为：

```text
registry_status: INCONCLUSIVE
phase_status: STOPPED
decision: P0B_NO_LAMBDA_INSERTION_JUSTIFIED
terminal_reason: NO_LOCALIZED_LAMBDA_EDGE
P1_authorized: false
P2_authorized: false
decision: RAW_HASH_MISMATCH_STRUCTURAL_EQUIVALENCE_PENDING
md_status: MD_NOT_STARTED
```

EXP-018 不重新打开 EXP-017，不改变 v21 ledger，不插入 λ，不运行 fixed-λ probe，不加入 analytic-q，不调用 student，也不产生 P1 candidate 或 P2 authorization。

## 2. 冻结对象

所有 repeat 使用同一份 v21 schedule、System、Hamiltonian、IBS 状态和 window 5 production checkpoint。禁止在 `output_lrc_fix` 中写入或覆盖任何文件；EXP-018 的所有输出必须位于新的目录：

```text
output/outer_lambda_exp018_stationarity_confirmation/
```

由于首次 launcher 尝试在 System provenance gate 处 fail-closed，实际采样输出改用新的目录：

```text
output/outer_lambda_exp018_stationarity_confirmation_v2/
```

原 provenance 失败报告不覆盖，登记为 `INVALIDATED / INVALIDATED_AUDIT_IMPLEMENTATION`；修正版审计使用：

```text
output/outer_lambda_exp018_provenance_audit_v2/EXP-018_window5_provenance_verdict.json
```

修正版独立记录 `cv_shape == (8,)`；CUDA mixed 跨 Context 比较使用 sealed 的 DEC-046 dtype-aware 相对/绝对容差。修正版 verdict 通过前，`md_sampling=false`。

冻结的关键输入：

| 对象 | 路径 | SHA-256 |
|---|---|---|
| preopt checkpoint | `output_lrc_fix/checkpoints/preopt_dual_vanishing.json` | `0b2e1feffd2bf7b305f2ec088ee64556722753933b166e8179b7a0d95fc53919` |
| stage2 checkpoint | `output_lrc_fix/checkpoints/stage2_vanishing.json` | `ac3f4b17cf4d5cc9db30a665a34acc9651e51587b6df321f64956505885b3da2` |
| System | `output_lrc_fix/system_native.xml` | `e2eb7b94fceec5b4cdf552972fc40fa633d49b8e4385acfc62a357e4bfc01717` |
| topology | `output_lrc_fix/topology.cif` | `6602f537d13179fc8294bcbaea1c7247fa9148b7d372a6411bc9f705db744ccf` |
| box | `output_lrc_fix/box_vectors.npy` | `bce72109cd2e57e1eebe12feaf4262cc1b9fdc1716d8ae341239809e0375bdb3` |
| window 5 manifest | `output_lrc_fix/checkpoints/production_window/vdw/window_5/manifest.json` | `fbe26d58cf481f64bc3f239bf41a91cfa1630f3b86203376d194a656ca6a4d14` |
| window 5 checkpoint | `output_lrc_fix/checkpoints/production_window/vdw/window_5/openmm.chk` | `c239922d59ff65e779290e911f0a7872f6660ef47f1aa27f9671d8c2591aefa0` |
| window 5 IBS state | `output_lrc_fix/checkpoints/ibs_state_vdw_window_5.json` | `c44ec3b647c9a29b9bfb6b93dd002f35915f2be273392197b698e43ecde20051` |

window 5 的 Hamiltonian 仍为 4 个 alchemical states，vdW λ 为 `[0.19738462625408026, 0.15557806428999377, 0.10014598833508535, 0.0]`，coul λ 全为 `0.0`；温度 `300 K`、步长 `0.002 ps`、friction `2.0 / ps`、IBS bias protocol `29`。

## 3. 采样设计

本节已写入 sealed preregistration，并由受控 launcher 校验输入 hash。

- 最少 2 条真正独立的 continuation；计划使用 3 条 repeat，seed 固定为 `20260811`、`20260812`、`20260813`。
- 每条 repeat 从同一个冻结的 window 5 `openmm.chk` 分叉到独立 scratch 目录；不得退回共享 trajectory，不得复用其他 repeat 的 checkpoint、速度或输出。
- 每条 repeat 使用独立 integrator/random seed，先 burn-in `25000` steps（`50 ps`），再 production `250000` steps（`500 ps`），每 `500` steps 保存一次（`1 ps`，目标 500 帧）。
- 仅允许原 v21 System/Hamiltonian/IBS；不允许改变 λ、温度、时间步长、摩擦、cutoff、约束或 bias protocol。
- 运行只产生 stationarity 数据和诊断；不回写 EXP-017 ledger，不改写正式 MBAR 结果。

这是一个 GPU 采样任务，但当前 MD 尚未授权。必须先完成 window-5 专属 provenance audit，并由 launcher 读取 sealed semantic-pass verdict；在此之前任何采样命令都会 fail-closed。lambda insertion、analytic-q、student、P1 和 P2 仍全部未授权。

## 4. 预注册统计口径

所有规则在看到 EXP-018 数值前固定。分析对象是 window 5 的原始 frame-major `u_kj`、bias 和对应的直接 `dDelta_f` series；不得用 surrogate、student 或 analytic signal 替代。

### 4.1 每条 repeat

对 full、first half、second half 分别重放与 EXP-017 相同的 local-TMBAR/直接 `dDelta_f` 计算，记录：

- `delta_G_full`、`delta_G_first_half`、`delta_G_second_half`；
- `drift = second_half - first_half`；
- MBAR covariance uncertainty `sigma_mbar`；
- `abs(drift) / (2 * sigma_mbar)`；
- per-state mixture overlap、ESS、去相关帧数和 endpoint uncertainty；
- 是否存在非有限值、输入 hash 漂移、缺帧或 state 顺序漂移。

split-half 的方向性规则在执行前冻结为“与 EXP-017 window 5 漂移同号”即负号；它只用于可复现性描述，不作为 λ edge 证据。

### 4.2 IAT 与 block-mean

- IAT 使用 EXP-017 audit 的同一 positive-sequence autocorrelation 规则：逐 state 对 `u - bias` 计算正相关项，遇到首个非正项停止，`g = max(1, 1 + 2*sum(rho_positive))`，取 window 5 各 state 的最大 `g`。
- `n_decorrelated = floor(n_frames / g)`；每个 repeat 和两个 half 都报告 `g`、`n_decorrelated` 和 lag-1 autocorrelation。
- block length 预先定义为 `B = max(1, ceil(g_window5))` 帧；按时间顺序使用不重叠 contiguous blocks，尾部不足一个 block 的帧丢弃并记录。
- 对 full/first/second half 的 block means 计算均值、block-mean SEM、block 数和 block-mean 95% percentile interval；block-mean 结果是 stationarity 诊断，不替换 EXP-017 的正式 MBAR error。

### 4.3 重复间方差

三个 repeat 全部通过输入和有限性检查后，计算：

- full `delta_G` 的样本方差 `s_repeat^2`；
- first-half、second-half 和 drift 的样本方差；
- `variance_ratio = s_repeat^2 / mean(sigma_mbar_i^2)`；
- `repeat_sigma_floor = max(mean(sigma_mbar_i), s_repeat)`，并另报其相对正式 MBAR error 的倍数。

`variance_ratio >= 2` 预注册为“重复间波动支持 MBAR 不确定度低估的信号”；`variance_ratio < 2` 不支持该强结论。这个判定只适用于 EXP-018 的描述性问题，不会回填或改写 EXP-017 正式误差门。

### 4.4 window 5 漂移是否复现

定义每条 repeat 的 `z_i = abs(drift_i) / (2*sigma_mbar_i)`。在三条 repeat 均有效时：

- `REPRODUCED_STATIONARITY_WARNING`：至少 2/3 条 repeat 的 drift 与 EXP-017 同号，且这两条的 `z_i >= 2`；
- `NOT_REPRODUCED`：至少 2/3 条 repeat 的 drift 与 EXP-017 不同号，或至少 2/3 条的 `z_i < 1`；
- 其余情况为 `INCONCLUSIVE`。

上述标签只回答“window 5 漂移是否可复现”。无论标签为何，均不得推导 lambda insertion、student 信号、物理慢模或 P1/P2 授权。

## 5. 失败关闭条件与交付物

任一 repeat 的 checkpoint/system/manifest/IBS hash 不匹配、state 数不为 4、lambda 顺序改变、生产帧不足 400、输入出现 NaN/Inf，或存在共享 scratch/output，即整项 EXP-018 标记 `INCONCLUSIVE / INVALID_INPUT_OR_RUN`，不做部分晋级。

预期交付物为每条 repeat 的只读审计 JSON、trajectory/checkpoint hash 清单、aggregate stationarity JSON 和一份 markdown summary。报告必须同时列出：

- 三条 repeat 的 full/half/drift、IAT、block-mean 和 repeat variance；
- 默认 MBAR error 与预注册 floor 的并列值；
- `REPRODUCED_STATIONARITY_WARNING` / `NOT_REPRODUCED` / `INCONCLUSIVE` 标签；
- `lambda_insertion_authorized=false`、`P1_authorized=false`、`P2_authorized=false`；
- `production_data_mutated=false` 和所有冻结输入 hash。

EXP-018 完成后仍需另行做科学判断；它不能自动改变 EXP-017 的终局登记。
