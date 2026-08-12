# ABFE-IBS 开发失败路线、负结果与证据底稿

> 本文件是 `SOFTWARE_PROGRESS_AND_TECHNICAL_DRAFT_2026-08-12.md` 的失败路线专章。  
> 它保留确认失败、结论不充分、主动终止和历史无效结果，供进度报告及论文 Negative results/Limitations 使用。  
> 证据截止：2026-08-12。

## 1. 证据口径

| 标签 | 使用条件 |
|---|---|
| `FAILED` | 已真实执行并未通过冻结门，只关闭该门所定义的路线 |
| `INCONCLUSIVE` | 数据不足以支持肯定或否定结论 |
| `NOT_PURSUED` | 主动不继续，不能写成实验失败 |
| `NOT_STARTED` | 仅有设计或草案，尚无实验结论 |
| `INVALIDATED` | 结果因实现、输入或协议错误而失去科学效力 |
| `HISTORICAL_FIXED` | 历史失败已修复，不应再报告为当前缺陷 |

## 2. 生产主线的重要失败

### 2.1 2026-07-27 Atenolol ABFE：`INVALIDATED`

旧结果曾记录 `Delta G_bind=+16.00 +/- 2.20 kJ/mol`。复核确认复合物腿沿用了陈旧且错误的 `boresch_equilibrium_committed.json`：两个角度对调、三个二面角错误，体系重新平衡后却仍仅凭“文件存在”继续复用。约束将配体拉离原 pose 约 3.42 Å；无约束预平衡仅漂移约 0.60 Å。方向性氢键丢失后，复合物去电荷与 Boresch 释放项均被污染。

结论：该数值只能作为 bug 审计基线，禁止科学引用。对应证据：

- `docs/status/RESULT_2026-07-27_atenolol_rank11.md`
- `docs/status/README_STATUS_SNAPSHOT_2026-07-29.md`

### 2.2 当前 `output_lrc_fix`：候选，不是最终结果

实际 artifact `output_lrc_fix/final_binding_results.json` 记录：

```text
Delta G_complex = 180.9981 kJ/mol
Delta G_solvent = 157.8358 kJ/mol
Delta G_bind    = -23.1622 +/- 2.5139 kJ/mol
                = -5.5359 +/- 0.6014 kcal/mol
```

协议身份为 IBS v29 / path v21 / LJ LRC v3 / WCA v2。两腿 Stage 2 都标记 converged、23 lambda nodes、无 dropped windows；但独立重复为 false，main production seed ledger 为空，Boresch 有 1 个 `kr` 从约 7355.9 裁剪至 2000。故只能写成“2026-07-29 archived candidate artifact”。旧 status snapshot 还出现过另一组两腿值重算的约 `-40.84 kJ/mol`，不得与当前 artifact 混用。

### 2.3 Stage 2 window 0：多次 λ/分组尝试失败

vanishing window 0 曾长期塌缩在单一态，`min_absolute_ess≈1`。依次真实尝试 6/4/3 态分组、pilot 加密和真实 Delta-f 均匀切分，端点占据仍约 96–99%。这些结果排除了“只改变 ensemble 态数”以及“只重新摆放 lambda”可解决该问题。

协议演化不能简化为一个符号 bug：v19–v20 曾认为 TMBAR/pilot-TI 符号错误，v21/v22 反号；v27 又通过完整平衡约定核对证明反号本身错误并撤销。当前 v29 通过候选可信门、阻尼、pairwise cap、局部 MBAR 冻结门和 fail-closed 状态机共同处理。

最重要的开发结论是：`f_k` 的符号、gauge、绝对解与增量、可信度和冻结时机必须一起定义，不能根据占据方向凭经验决定正负号，也不能把未收敛的绝对 TMBAR 解整组覆盖进 Context。

证据：`docs/handoffs/VANISHING_WINDOW0_HANDOFF.md`、`ibs_engine.py` 当前 v29 注释与常量。

### 2.4 原地自动修复：`REMOVED/NOT_PURSUED`

历史 production 自动拆窗、插 lambda、重校准并继续复用旧数据，会使已有 ensemble 与新 Hamiltonian 身份混杂，破坏缓存、checkpoint 和统计可审计性。该循环已移除，替换为：

- production 不改 lambda 和冻结 `f_k`；
- 只追加失败窗口采样；
- 必须改变 coverage 时建立独立 immutable rescue ensemble；
- 原文件不覆盖，离线拼接显式登记。

证据：`docs/archive/removed_overlap_autorepair_mutation_loop.md`、`docs/status/NON_MUTATING_V1_STATUS.md`。

### 2.5 EXP-017/018/019：不确定度门未闭合

- EXP-017：`INCONCLUSIVE/STOPPED`。账本/TMBAR 门通过，`min_overlap≈0.3913`、最小去相关样本 96，但没有 localized bad lambda edge。window 5 split-half drift 为 `-0.5587 kJ/mol`，约 `4.46×2sigma`，因此不授权插 lambda、P1 或 P2。
- EXP-018：`INCONCLUSIVE/CLOSED`。3 个 seed 的 drift z 约为 `1.134/2.568/1.381`，只有 1/3 重现负漂移；repeat variance ratio `16.7599`，显示单次 MBAR sigma 可能低估跨运行散布，但不足以确认统一漂移方向。
- EXP-019 v3：`FAILED_QUALIFICATION`。在正式 baseline repeats 前，Stage 2 endpoint uncertainty 为 `1.2481 > 1.0 kJ/mol`；completed baseline repeats=0。诊断 rescue 值 `159.3165 +/- 2.0618 kJ/mol` 不可晋级。下一步只授权 read-only Stage 2 rescue/coverage audit。

证据：对应 `output/outer_lambda_exp017*`、`output/outer_lambda_exp018*`、`output/outer_lambda_exp019*` 的 final registry/summary。

## 3. DEXP、MACE、MTS、ORB 与解析 CV 路线

### 3.1 DEXP：单体系解析表示有信号，生产收敛未证明

Atenolol 单体系的 MACE-reference kernel projection 中，DEXP 在能量、force/torque/Hessian 和环境截断复核下优于 LJ，这是有效正结果。但 15 个 V/S/B 多初态 replica 均未平衡或未汇合，`(12,6)` 和 `(14,5)` 调参都没有解决。Atenolol 单体系参数搜索因此停止；尚无 8–15 个体系外部 benchmark 和多体系 CLI。

可写：“DEXP 对局部参考曲面的解析表达具有单体系优势。”  
不可写：“DEXP 已改善 ABFE 收敛或已成为生产势。”

证据：`docs/experiments/DEXP_KERNEL_PHYSICS_ISSUES.md`。

### 3.2 EXP-006/007：力学资格的失败与修正

EXP-006 最大路径力 `258.949 kJ/mol/nm > 250`，其余门通过，故该次 qualification 失败。仅将系数 0.10 改为 0.09 后，EXP-007 六项门通过。这个通过只证明特定幅度下可稳定积分，不授予成本或 production 资格。

### 3.3 EXP-009：direct MACE-MTS `FAILED`

完整 MACE 经 PythonForce/openmmml 进入 CUDA MTS force group 后，N=1 即触发 `CUDA_ERROR_INVALID_HANDLE`。同后端不再重跑，MACE 转离线 teacher。失败范围是当前实时后端/MTS 组合，不是 MACE 表示本身。

### 3.4 EXP-010：cheap torsion CV 蒸馏 `FAILED`

六个预注册 Fourier 候选均未胜 intercept-only：

```text
intercept-only RMSE       21.5109 kJ/mol
best 1D order-2 RMSE     22.1737 kJ/mol
generalized-force R2    -13.5934
```

失败更可能限定在 atom-cut protein-only teacher 与逐帧总 interaction-energy 目标不闭合，不证明所有 torsion bias 都失败。没有事后加 Fourier order 或更改目标。

### 3.5 EXP-011：periodic PMF `FAILED/STOPPED`

严格 MBAR mutual overlap `0.02353 < 0.03`，去相关样本 `22 < 25`。因此不拟合最终 PMF、不继续补采，转向不预设单一慢 CV 的 EXP-012。

### 3.6 EXP-012：离线信号存在，实时成本效益 `FAILED`

LocalResidualStudent direct-gap held-out 平均改善 `13.9348%`，2/3 folds；teacher cached-latent readout 三个 LORO fold 均改善，平均约 44.6%；D2 coordinate/autograd 27/27 通过。

但在线 TorchForce 约为 baseline 的 `1.81–1.89×`，三组 ESS/GPU-hour 均下降；理想网络成本估计约 `1.83× > 1.10×` 预算门。因此当前每 MD step 实时 TorchForce 路线关闭。Arm A/B/D 为 `NOT_PURSUED`，不是失败。

### 3.7 EXP-013：三种方案均未晋级

- 方案③：N=8/16/32 的温度和/或 fused energy 出现 `z>3`，温度随 N 偏移约 `+0.66/+1.01/+1.29 K`，013-B 失败。
- 方案①：qualification 中 N=2/4/8 温度 z 为 `5.61/5.79/6.83`，N=8 fused energy z `5.62`，不授权 N=16。只有单 seed/32 ps，足以触发保守停止门，但不证明普遍物理 bias。
- 方案②：N=1 mixture ESS proxy `47.83 -> 38.80`，相对 `-18.88%`；ESS/GPU-hour 约 `932 -> 218`。健康、有限值、温度、力和账本门均通过，失败不是数值崩溃。

最终：不重调 `c1`、不重选 checkpoint、不继续搜索 MTS 间隔；尚未证明可安全低频化且净提高采样效率的 learned slow force。

### 3.8 EXP-014：native radial compression `FAILED`

`n_radial=8/16/32` 的 mean held-out R2 约为 `-11.46/-2.80e6/-2.09e12`，没有通过所有 folds `R2>=0.90` 且 retention `>=0.80` 的门，未进入 OpenMM force qualification。只关闭此冻结表示，不否定全部局部压缩形式。

### 3.9 EXP-016：`INCONCLUSIVE/SURROGATE_ONLY`

3 条轨迹共 1500 帧、1 ps 保存间隔，ledger/latent 对齐通过；但缺 physical alchemical state/replica history 和 physical crossing。不能定义物理 `tau_information`，不能授权 online/MTS promotion。

### 3.10 ORB：表示通过，部署成本 `FAILED`

ORB layer-2 3/3 LORO folds 改善，平均 relative improvement `39.68%`，得到 `REPRESENTATION_PROMISING`。同平台 CUDA matched-path：

```text
baseline                  1.273 ms/step
baseline + ORB scalar    78.896 ms/step
increment                77.622 ms/step
budget                    0.1-0.2 ms/step
```

增量约为预算上限的 388 倍。最终为 `OFFLINE_TEACHER_ONLY`，ORB-004/005、在线 TorchForce、MTS 和 production wiring 停止。

### 3.11 EXP-020/021：科学信号和部署成本分离

EXP-020 R1 离线 mean relative improvement `55.55%`，D2 力学资格通过；但 full-system CustomGB `1.6965×`、N1 `6.0717×`、N2 Torch `61.2922×`，full D3=false。

EXP-021 已完成 sealed preregistration 和真实 CUDA G1 D0-COST。G1 qualification median ratio 为 `1.107419 > 1.07`，bootstrap 95% upper 为 `1.114105 > 1.10`，因此结论为 `STOP_EXP021_NATIVE_DENSITY`。训练、G2 和 G4 均未授权。该结论关闭的是当前 sealed native grouped-density calculation graph 的成本资格，不否定 grouped-density 表示可能包含科学信号。

## 4. 扩展路线

### 4.1 膜体系

已有 100 ns NPT 质量门通过记录，但 Stage 0→Stage 1/2 完整 ABFE 未完成。旧 `+23.27 kcal/mol` 结果因 solvent-leg angles dropped 等问题无效。在 P1-19/P1-22 统计不确定度闭合前，不应报告膜体系 production ABFE。

### 4.2 带电配体 charge-transfer

C3 v1 mixed-force gate 曾失败；双侧 switch 归一化后 v2/MEM-00h 局部门 20/20 通过。但没有真实 charged ligand 的 Stage 1→2 完整循环，Atenolol neutral 分支也不触发该路径。状态应写“局部工程门通过，端到端未验证”。

### 4.3 pose/pull

扫描观察到接触依次刮过 ASP85、VAL136、TYR169，而非单一 sticky hydrogen bond。路线主动停止，属于 `NOT_PURSUED`，不是 production 功能。

### 4.4 REMD/CUDA 历史失败

早期 12-context CUDA 失败根因是 PyMBAR4/JAX 默认预分配约 75% GPU 内存。调整 import 前环境和 teardown 后，同机 12 CUDA contexts + 500 exchanges 已重测通过，accept 约 0.675。因此这是 `HISTORICAL_FIXED`，不应报告为当前缺陷；但 traditional fixed-box/LRC 回归 V-02 仍待运行。

## 5. 测试与验证边界

静态扫描发现 88 个 `test_*.py` 模块、1,022 个顶层测试函数。历史日志记录 2026-08-09 `1161 passed/3 skipped/1 deselected/0 failed`，2026-08-11 `1213 passed/0 failed`；更早 TODO 还有多组不同数字。本次没有在当前环境重跑 OpenMM/PyMBAR pytest，且 `VALIDATION_MATRIX.md` 与部分 TODO 互相冲突。因此这些只能称为 dated documented runs，不能称为本次验证。

当前必须补齐：

1. P1-19/P1-22：独立重复、去相关、moving-block/bootstrap 和跨运行方差；
2. v29/v21/LRC3/WCA2/ESS3 联合协议的完整 CPU + CUDA 复核；
3. immutable rescue、checkpoint 和跨进程 resume 的目标环境证据；
4. traditional fixed-box/LRC 小回归；
5. charged ligand、membrane 和 EXP-021 各自的端到端资格。

## 6. 进度报告可直接使用的负结果总结

本项目没有把未达门的增强采样模型接入生产流程。DEXP 在单体系局部参考曲面上出现正信号，但多初态动力学未汇合；MACE/ORB 表示在离线 held-out 数据上能够降低部分 gap variance，但实时部署分别受到后端、Hamiltonian 等价性和 1.8×至数百倍成本的限制；低频/MTS 学生势又未通过保守的动力学或 ESS 门。由此得到的阶段性结论是：离线预测性能不能替代在线采样正确性和 ESS/GPU-hour，任何 learned path force 都必须同时通过统计、力学、Hamiltonian 和成本门。

生产主线目前最重要的未完成项也不再是 lambda overlap，而是端点不确定度的跨时间、跨运行校准。EXP-017 没有找到局部坏边，EXP-018 只在 1/3 seed 中重现 drift，EXP-019 又在正式 baseline repeat 前未通过 endpoint-sigma 门。这要求下一阶段优先完成只读 coverage audit、独立重复和 block bootstrap，而不是继续增加模型复杂度。

## 7. 写作禁区

- 不把旧 `output/` 或 2026-07-27 数值写成早期有效预测；
- 不把 `output_lrc_fix` 的 `-23.16 kJ/mol` 写成最终发表结果；
- 不把 v21/v22 的历史反号处理写成当前仍成立的单一根因；
- 不把 EXP-007 力学通过写成 MACE production 通过；
- 不把 EXP-012/ORB 离线改善写成 ESS/GPU-hour 改善；
- 不把 EXP-013 单 seed 停止门写成普遍物理偏差证明；
- 不把 EXP-016 surrogate changes 写成 physical crossing；
- 不把 EXP-021 写成已经训练或已经失败；
- 不把 `NOT_PURSUED` 的 Arm A/B/D、parallel stages、pose/pull 写成实验失败；
- 不把测试代码数量写成测试通过数量。

