# pose-scan / pull-scan 交接笔记（临时，下次接着 debug 用）

> 短期笔记，写于本次会话中途（还没验证完就要下班）。跟长期的
> [`RESUME_DEXP_SESSION.md`](RESUME_DEXP_SESSION.md) 是分开的——
> 那边等这一块验证完/收敛了再补一节进去，现在先记流水账，够自己接着捡起来就行。

## 背景一句话

`--fit-only` 拟合数据只有 0.20-0.227nm 这一条窄缝（来自 pre_equilibration.dcd 尾部），双指数核参数
在这么窄的范围里统计上不可辨识（chi2=9.74, dof=8，跟纯噪声没区别）。想手动造一批跨越更宽距离范围
的构型喂给 fit，绕开等增强采样。

## 已经确认的东西（不用再验证）

1. **`--pull-scan`（沿单一方向硬拉配体质心）机制本身是对的，但目标坐标选错了**：
   - 配体质心能被干净地从 COM-COM 0.09nm 拉到 1.7nm（力学机制、锚点位置约束都工作正常）。
   - 但 DEXP 真正在意的"最近原子对距离"(`min_valid_le_distance`，限定在 `[fit_r_min,fit_r_max]=[0.20,0.45]` 内)**完全不为所动**，17-16 帧全程卡在 0.20-0.21nm。
   - 原因确认：不是某个氢键"粘住"（换了拉力方向对准最近接触残基 ASN254 也没用），是配体沿任何单一方向穿过密集蛋白环境时，**总会连续擦过不同的原子**（实测最近接触点从 ASP85→VAL136→TYR169 一路换人）。**结论：沿单一方向硬拉这条路线到此为止，不用再试了。**

2. **改用 `--pose-scan`（随机刚体扰动+短程弛豫+分箱筛选，不追求连续轨迹）**，用户明确要求的方案：
   - 已实现：随机旋转（均匀四元数）+ 随机平移（0 到 `--pose-scan-translate-max-nm` 之间）配体，`LocalEnergyMinimizer` 去除硬碰撞，再跑 `--pose-scan-relax-steps` 步弛豫，然后测 `min_valid_le_distance`/短接触数/heavy-atom 最近距离，按 `--pose-scan-reject-min-dist` 拒绝、按 `min_valid_le_distance` 分箱（`--pose-scan-bins` 个箱，每箱目标 `--pose-scan-per-bin` 帧）保留。
   - 额外加了"整体短接触惩罚"力（`CustomNonbondedForce`）：`U_bias = k_bias * sum_ij sigmoid((r_cut-r_ij)/w)`，配体-环境(含水)所有 pair 一起压，不是只顶开最近那一对——这是用户明确要求的、比单纯 soft-min 更抗"换人"的方案。
   - **踩过一个坑，已修**：`CustomNonbondedForce` 跟原生 `NonbondedForce` 必须有完全相同的 exclusions，不然报 `All Forces must have identical exclusions`。修法：把原生 `NonbondedForce` 的 `getExceptionParameters` 全部复制成 `addExclusion`（本项目其它地方新建 CustomNonbondedForce 时也是这个套路，参考 `abfe_core.py` 里同名模式）。

## 当前卡住的地方（下次从这里接）

跑了个小规模 smoke test（`--pose-scan-trials 6 --pose-scan-relax-steps 500 --pose-scan-per-bin 2 --pose-scan-bins 6`），机制上不再报错，但：

**6 次随机扰动（平移幅度最高到 0.6nm）之后，minimize+弛豫全部把配体"拉回"到几乎同一个窄范围**（`min_valid_le_distance` 落在 0.209-0.221nm，6 个里 5 个都在同一个 bin），完全没有体现出应有的多样性。

怀疑原因（按可能性排序，下次先试这几个）：

1. **`LocalEnergyMinimizer` 本身就在把扰动"擦掉"**——从一个大幅偏移但仍在同一个能量盆地捕获范围内的位置做纯能量最小化，很可能直接梯度下降回原生构象附近。可以试：减少 minimize 的 `maxIterations`（现在 200，可以试 50 或者干脆跳过，只留一个很短的"去除硬碰撞"步骤，比如只在检测到极端 clash 能量时才 minimize）。
2. **短接触惩罚力 `k_bias=10` 太弱，弛豫阶段配体自己"重新对接"回原位的驱动力（静电+范德华+氢键）完全压制得住这个惩罚**——可以试把 `--pose-scan-short-contact-k` 提到 50-200 量级看能不能真的顶住重新对接的趋势。
3. **弛豫步数 500 太短或太长都可能有问题**——太短可能还没来得及让短接触惩罚力起作用就截帧了；太长则可能让系统有充分时间弛豫回原生盆地。两个方向都值得各试一版对照。
4. **随机平移的方向本身可能大部分时候撞回同一个"路径"**——因为是纯随机方向+随机幅度，如果大多数方向阻力都很大（密集蛋白环境），只有少数方向能走出去，6 次尝试可能全部落在了"被弹回来"的方向。可以加大 `--pose-scan-trials`（比如 100-300）先看看是不是只是运气不好、多试几次自然会出现"逃出去"的方向；如果 300 次里还是全部弹回同一个窄 bin，就说明是 1/2/3 的问题而不是运气问题。

## 建议下次的调试顺序

1. 先加大 trials（比如 `--pose-scan-trials 100 --pose-scan-per-bin 5`），别的参数不变，看看多试几次是不是自然出现分散——如果出现了，说明只是之前 6 次样本量太小，直接跑大规模就行。
2. 如果 100 次还是全挤在一个 bin，把 `--pose-scan-short-contact-k` 从 10 提到 100 再试。
3. 如果还不行，把 minimize 的 `maxIterations` 降到 50（或者干脆先跳过 minimize，直接进弛豫，只在能量爆炸时才做 minimize 兜底）。
4. 都不行的话，回头考虑要不要放弃"先随机扰动再弛豫"这个思路，改成更直接地在弛豫过程里全程加一个逐渐增大的短接触惩罚（类似退火），而不是扰动后一次性弛豫。

## 快速复现命令

```bash
# 当前用来验证机制是否work的最小 smoke test（GPU，几分钟内跑完）
python dexp_experiment.py --pose-scan \
  --pose-scan-trials 6 --pose-scan-relax-steps 500 --pose-scan-per-bin 2 --pose-scan-bins 6 \
  --device cuda --platform CUDA

# 下次先试这个（加大样本量，其它不变）
python dexp_experiment.py --pose-scan \
  --pose-scan-trials 100 --pose-scan-relax-steps 500 --pose-scan-per-bin 5 --pose-scan-bins 6 \
  --device cuda --platform CUDA

# 结果看这两个文件
output/dexp_experiment/pose_scan/pose_scan_trials.csv   # 每次尝试的 min_valid_le_distance/短接触数/是否接受
output/dexp_experiment/pose_scan/pose_scan.dcd          # 被接受帧的轨迹
```

## 代码位置

- `dexp_experiment.py::run_pose_scan`（方案2+3合并实现，含随机扰动、短接触惩罚力、分箱筛选）
- `dexp_experiment.py::run_pull_scan`（方案1，已确认沿单一方向拉不出想要的坐标，保留代码但不建议继续往这个方向调参数）
- `dexp_experiment.py::_random_rotation_matrix`（随机旋转矩阵）
- CLI 参数：搜 `--pose-scan` 前缀（`--pose-scan-trials/-translate-max-nm/-relax-steps/-anchor-k/-reject-min-dist/-bin-max-nm/-bins/-per-bin/-short-contact-rcut/-short-contact-width/-short-contact-k`）

## 其它这次顺带记一下、还没处理的东西

- `abfe_core.py:767-769`：`Orbv3SurrogateFitter.fit_parameters` 返回的 `sigma_elec=0.1, switch_width=0.20, cutoff_distance=0.70` 是硬编码字面量，从来没被拟合过，混在真正拟合出来的参数（`alpha_vdw`/`r0_vdw`/`A_fit`等）中间一起写进 `dexp_fitted_params.json`，容易被误认为是拟合结果的一部分。已确认位置，还没决定怎么处理（要不要挪到诊断字段、加个 `_hardcoded` 后缀提醒之类的），下次一起看。
- `output/dexp_experiment/dexp_fitted_params.json` 当前是干净的验证过的状态（r0=0.30, 边界正常），本次 pose-scan/pull-scan 的探索没有覆盖它，可以放心用。
