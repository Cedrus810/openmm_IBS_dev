# EXP-030：candidate 臂 f_k 在线学习训练目标缺失残差项的 bug（2026-08-26）

## 状态：已定位、已修复（`ibs_engine.py`，`IBS_BIAS_PROTOCOL_VERSION` 30→31）。
## 影响范围：所有在 v30 及更早版本下跑过的 `residual_enabled`（candidate）臂窗口的 f_k 校准结果全部作废，需要在 v31 下重新校准。baseline 臂不受影响。

---

## 1. 起因

`protocols/EXP-030_joint_state_score_preregistration_FROZEN_PRODUCTION.json` 授权
的 3×2×6 生产矩阵全部 36 个窗口跑完真实数据后，做 baseline vs candidate 配对
对比（见同日 `EXP-030_AB_COMPARISON_SUMMARY_2026-08-26.md`，**该文件的"candidate
更精确"结论现已作废，见该文件顶部更新说明**）时发现：同一个 repeat 里 baseline
和 candidate 报出的整条链 ΔG 相差 13~21 kJ/mol，而两边各自报的总不确定度只有
0.7~2.5 kJ/mol——差距是报出误差棒的几十倍，不可能是统计噪声。

用户在此基础上追问了三轮，逐步把范围收窄：
1. "共享目标 ΔG 严重不一致"——指出 baseline/candidate 理论上测的是同一个物理量。
2. 提出严谨的交叉裁决实验设计（baseline 从 candidate 终态启动、candidate 从
   baseline 终态启动，判断结果跟着初态走还是跟着算法走）。
3. "是不是 ibs 的 fk 做功呢"——把怀疑焦点指向 f_k 本身，这是最终定位到真正
   bug 的关键提示。

## 2. 排查过程（按时间顺序，每一步的结论）

1. **怀疑残差泄漏进物理目标能量** → 排除。`collect_energies()` 里
   `target_energies = softcore_energies + lrc_energies`，完全不依赖
   `residual_enabled`，与含残差的 `sampling_state_energies` 是两个分开存储、
   分开使用的量，没有混。
2. **怀疑 MBAR 重加权矩阵构造错误** → 排除。`u_sampled_eff = base + bias`
   （采样行，含偏置）vs `u_kj_shifted = base + u_int`（物理态行，不含偏置）是
   标准的"用偏置采样、拿物理态重加权回来"写法，理论上无偏。
3. **怀疑物理态相互作用能计算本身有残差混入**（比如 force group 号碰撞）→
   排除。`evaluate_interaction_energies` 用的是一个完全独立的 probe
   `openmm.Context`（`_build_probe_context`，只从序列化的每态 softcore CV
   force XML 重建，不含残差力），跟主采样 Context 是两个物理上分开的系统。
4. **占据分布诊断**（用已有 `production_report.json` 里的 `per_state.mean_occupancy`，
   零新增 GPU 计算）→ **找到真正线索**：baseline 三个 repeat 全部窗口占据接近
   均匀（各态 ~0.15~0.3），candidate 三个 repeat **系统性、方向一致地**严重偏斜
   （某些窗口单态占据高达 69~77%），三次独立重复的偏斜模式几乎相同——排除了
   "运气不好、没采够"的随机滞后解释，指向某个确定性机制在起作用。
5. **顺着"f_k 做功"这条线查到根因**（见下）。

## 3. 根因

`IBSSampler.collect_energies()`（`ibs_engine.py`）里，喂给 `self.energy_buffer`
（进而喂给 `update_weights()` / `_solve_tmbar_and_recenter()` 做 f_k 在线学习）
的量，此前一直是：

```python
bias_cv_energies = softcore_energies - self.e_offset   # 纯物理项
energies = bias_cv_energies
```

**完全不含残差项。** 但实际驱动采样动力学的 Group-1 偏置力（`_state_expr(k)`，
`IBS_BIAS_PROTOCOL_VERSION=30` 引入 `s_residual` 时改的那处）公式是：

```
X_k = U_k + s_residual · A_k · B_φ − f_k
```

**含残差项。** 也就是说：f_k 在线学习时看到的"当前混合分布长什么样"的训练信号，
和它实际被部署去拉平的那个分布，根本不是同一个函数。f_k 学的是"如何拉平不含
残差的分布"，却被拿去压着一个"含残差的分布"做采样——A_k 越大的态，训练信号
和实际部署目标之间的落差就越大，天然学不对。这个不对齐跟"占据分布系统性
偏斜、三次重复方向一致"完全吻合：同一套残差模型在三次独立重复里以同一种
方式让训练目标失配，不是随机效应。

代码里其实已经算出了正确的量，只是从未接入训练链路：

```python
sampling_state_energies = (
    softcore_energies
    + residual_coefficients * (residual_basis_energy - residual_offset)
)  # 含残差，这才是实际被采样的混合态能量
```

这个量此前只被存进 `self.sampling_state_energy_history`（供 TMBAR 分析用的
`sampling_states.npy` 落盘、供占据诊断），从未被路由进 `self.energy_buffer`。

`residual_state_coefficients`（即 A_k，来自 `get_sampling_state_coefficients()`）
是 full-strength 的原始系数，不随 `bias_scale`/`s_residual` 的爬坡缩放——这是
有意为之，跟物理项那边的既有处理方式一致（`bias_cv_energies` 此前也从不乘
`bias_scale`）：f_k 的训练目标应该始终对准最终满强度的混合分布，爬坡只是部署期
的数值稳定机制，不应该被训练目标追着走。

## 4. 修复

`ibs_engine.py::collect_energies()`：

```python
if self.n_states > 0 and np.isfinite(sampling_state_energies[0]):
    self.e_offset = sampling_state_energies[0]
bias_cv_energies = sampling_state_energies - self.e_offset
energies = bias_cv_energies
```

`baseline` 臂（`residual_enabled=False`）时 `sampling_state_energies` 就是
`softcore_energies` 的原样拷贝，这次改动对 baseline 逐字节不变（已用测试验证）。

`IBS_BIAS_PROTOCOL_VERSION`：30 → 31，`IBS_BIAS_CACHE_COMPATIBLE_PROTOCOL_VERSIONS`
收窄到只有 `{31}`（v30 及更早版本下 `residual_enabled` 窗口学出来的 f_k
是在错误训练目标下学的，不能被当成 v31 的有效热启动/冻结验证结果直接续用；
版本号检查本身不区分 arm，所以 baseline 窗口理论上物理过程不变，也会跟着多
一次冷启动，代价可以接受）。同步更新了两处硬编码锁死版本号的测试
（`tests/test_audit_protocol_regressions.py`、`tests/test_warmup_overlap_protocol.py`）。

## 5. 验证

- `py_compile ibs_engine.py` 通过。
- 全仓库 `pytest`：1333 个测试里 1321 通过；失败的 12 个逐一核对过（pymbar
  环境版本漂移 4.0.3 vs 协议要求的 4.2.0、以及一个在 EXP-025/026/028 就已经
  过时的"构造函数不该有 residual 参数"断言），**全部是跟这次改动无关的既有
  问题**，不是这次改动引入的新回归。
- 跟这次改动直接相关的测试全绿：`test_audit_protocol_regressions.py`、
  `test_warmup_overlap_protocol.py`、`test_core_physics_numerics.py`、
  `test_resume_reuse_contracts.py`、`test_non_mutating_policy.py`。
- **没有做过任何真机 GPU 验证**（没有在 v31 下重新跑过一个真实窗口去确认
  占据分布真的被拉平了）——这是下一步必须做的事，不能只凭代码审查和单元测试
  就认定问题已经解决。

## 6. 对已有 EXP-030 生产数据的影响

3×2×6 矩阵里全部 candidate 窗口（18 个）的 f_k 校准都是在这个 bug 存在的情况下
跑出来的，属于无效结果，需要在 v31 下重新跑。baseline 窗口（18 个）理论上物理
过程没变，重跑预期得到逐字节一致的结果，但按版本号检查的设计（不分 arm）也会
被强制冷启动。

`EXP-030_AB_COMPARISON_SUMMARY_2026-08-26.md` 里"3 个 repeat 里 2 个 candidate
更精确、中位数改善 +24.5%"这个结论，是建立在有 bug 的 candidate 数据上得出的，
**现在不能再当作有效结论使用**，该文件已加更新说明指向本文档。

## 7. 尚未做、留给下一步的事

- 真机验证：至少一个 candidate 窗口在 v31 下重新跑一次完整校准，确认占据分布
  是否真的接近均匀了（不能只信代码审查）。
- 全部 18 个 candidate 窗口（3 repeat × 6 window）需要重新校准+生产。
- 重新校准完成后，重新做一次 baseline vs candidate 对比，看 ΔG 是否收敛到
  接近的值（如果仍有分歧，再考虑用户提出的交叉裁决实验设计——baseline 从
  candidate 终态启动、candidate 从 baseline 终态启动，判断分歧是"两个亚稳
  盆地"还是"方法特异性偏置"）。
- 用户提出的占据分布/B_φ 采样分布诊断这次只用了已有数据里的 `mean_occupancy`
  做了初步验证；更细致的诊断（比如逐帧 B_φ 分布、逐 λ 态占据随时间演化）
  还没做。
