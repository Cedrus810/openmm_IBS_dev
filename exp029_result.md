# EXP-029 结果汇总（会话中间记录，未合并主线）

最后更新：2026-08-24

本文件记录 2026-08-24 这次会话对 EXP-029（完整生产规模 A/B：两臂各自独立
`cold_start` 校准+冻结自己的 \(f_k\)，见 `EXP-029_WIRING_HANDOFF_2026-08-24.md`）
的 smoke 级排查结果。**这是中间记录，不是最终结论**——按用户要求，先记下来，
等 EXP-029 全部跑完、且 EXP-030 也跑完之后，两者一起合并进主线版本号，本文件
届时会被那次合并取代或整合。

---

## 1. 当前状态

```
EXP-029 harness 接线            = 已完成（同 2026-08-24 更早的 AUTHORIZED_WIRED_NOT_RUN）
EXP-029 完整生产 36-job matrix   = 仍未跑（本次只做了单窗口 smoke 级诊断）
本次会话新发现的问题             = [阶段1] 能量最小化会把 window_1 的局部密度
                                    拉穿残差力模型的支持域上限（见 §2）
候选修复（跳过 EM）              = 对 window_1/candidate/repeat_0 验证干净通过
                                    到生产，但仅此一个窗口、仅临时补丁脚本，
                                    未走正式协议参数、未验证其余窗口（见 §3）
是否已合并进生产流程/主线版本号   = 否——按用户指示暂不改代码，等整体跑完
                                    与 EXP-030 一起合并
```

---

## 2. 本次发现的问题：EM 把 window_1 局部密度拉穿模型支持域

### 2.1 症状

`candidate` 臂、`window_1`（λ_vdw = [0.7274, 0.6740, 0.6265, 0.5845]，
`lambda_shield` 均值 0.6531）在 `[阶段1] 开始能量最小化` 内部触发插件
fail-closed：

```
LocalManyBodyResidualForce (CUDA) fail-closed after force evaluation in
K1/K6a (computeQ) (code 5): unique environment atom count exceeded
max_environment_atoms
```

用真实入口（`scripts/exp029_window_state_machine.py`，未改动）连续复现 3 次，
具体撞线的迭代步数和峰值不完全一致（GPU 浮点归约顺序不完全确定，这是已知
特征，见 `[EXP-025 G2 CUDA brute-force 通过]` 相关记录）：

| 尝试 | epoch | active_edges | unique_environments |
|---|---|---|---|
| 1（本次会话最早那次真实 smoke） | 97 | 1638 | 644 |
| 2（原样重跑一次） | 77 | 1480 | 376 |
| 3（monkeypatch 探针跑的那次） | 63 | 1531 | 619 |

模型的 `max_environment_atoms = 320`（冻结于 `output/outer_lambda_exp025_local_manybody_cuda/g1_reference/r1_model_payload_v1.json`），起始构型（checkpoint 冻结的那份）本身只有 244 个
unique environment atoms、最近配体-环境距离 2.04 Å，远低于上限——问题不在
起始构型，在最小化过程本身。

### 2.2 排查过程（避免下次重复走弯路）

1. 一开始怀疑"膜体系比溶剂密"——查证后确认 `output_lrc_fix_repeat01_seed20260905`
   是复合物腿（`system_type: "complex"`），不是溶剂腿，此假设不成立。
2. 怀疑 λ-WCA 防护壳强弱——查证后发现撞线窗口的防护壳系数（0.906）反而比
   没出问题的 window_0（0.492）更强，此假设被推翻。
3. 用 `Exp027Repeat`（`scripts/exp027_common.py`）手搭探针复现，两次独立方式
   （分段最小化、连续单次调用+`MinimizationReporter`）都没能复现，密度全程
   平稳在 244~251。
4. 怀疑是探针里 `make_sim()` 多调了一次 `ibs_wrapper.update_parameters(...)`，
   把 source checkpoint 自己已经校准好的 \(f_k\) 提前加载了进去（生产真实
   路径在 EM 时 \(f_k\) 全部是构造默认值 0.0，`update_parameters` 从未在
   EM 之前被调用）——修正后重跑，仍未复现。此假设本身成立（生产 EM 时确实
   f_k=0），但不足以解释探针和真实入口之间的差异。
5. 放弃继续手搭探针，改为对真实入口 `exp029_window_state_machine.main()`
   本身的 `openmm.app.Simulation.minimizeEnergy` 打猴子补丁挂 `MinimizationReporter`
   （`scripts/diag_exp029_real_entrypoint_instrumented.py`），第一次尝试即
   复现，拿到了完整的密度/最近距离随 L-BFGS 迭代变化曲线（见 §2.3）。

### 2.3 根因

用户先提出、随后被曲线数据证实：**这段 λ_vdw 区间（0.58~0.73）的 softcore
vdW 排斥本来就被削弱了，配体和环境原子之间没有多少硬墙顶着，能量最小化
（纯梯度下降）会顺着"往里收更省能量"的方向把水拉近，拉近到一定程度后
局部塌陷式地把更多水拉进残差力的 4~5 Å 壳层**：

```
pre-min : unique_env=244, min_dist=2.042 Å
iter 0  : unique_env=243, min_dist=2.060 Å
iter 10 : unique_env=246, min_dist=2.055 Å
iter 15 : unique_env=246, min_dist=1.984 Å   ← 开始明显收紧
iter 20 : unique_env=252, min_dist=1.791 Å   ← 持续下探
epoch 63: unique_env=619 → fail-closed        ← 雪崩式塌陷
```

全程 \(f_k\) = 0（阶段5 校准还没开始），跟 candidate 该不该有自己的 \(f_k\)
校准（`exp027_result.md` §5.6 那个已知问题，EXP-030 的 joint_score 就是为了
解决那个）无关——这是两个独立的问题，时间点对不上：§5.6 那个问题发生在
阶段5 warmup 期间，这次崩溃发生在阶段1 EM 期间，f_k 校准还没开始。

残差力的 fail-closed 只是恰好在这个窗口装了检测器，把这个跟残差力本身无关、
本来就存在于基础势能面（base force + Boresch + softcore + WCA 防护壳）上的
塌陷过程当场抓了出来。baseline 臂大概率经历同样的收紧趋势，只是没人在那边
装密度检测器，不会报错，会悄悄产出一个局部过密但不报警的构型。

---

## 3. 候选修复：跳过 EM

### 3.1 已验证的部分

对 `window_1 / candidate / repeat_0`：把 `sim.minimizeEnergy()` 整段跳过（构型
原样从 checkpoint 进入 `[阶段2] 测试性步进`），全程干净跑完
`阶段2 → 渐进预热 → f_k 校准 → 冻结验证 → 生产`，产出完整、健康的
`window_complete.json`（`frozen_f_k = [-17.23, -4.33, 6.35, 15.21]` kJ/mol，
无 NaN/异常跳变，`authoritative_scientific_result: false` 如实标注为 smoke
级、非科学结论）。

跳过 EM 用的是一次性 monkeypatch 脚本
`scripts/diag_exp029_skip_em_test.py`，**不是生产代码的一部分**，只在这一次
诊断进程里把 `openmm.app.Simulation.minimizeEnergy` 替换成空操作。

### 3.2 六个窗口的完整排查结果（2026-08-24 补测，已完成）

用真实入口原样跑了 window_2/3/4/5（EM 照旧开着，candidate 臂，repeat_0），
外加 window_0 跳过 EM 的对照：

| window | lambda_shield | EM | 结果 |
|---|---|---|---|
| 0 | 0.8565 | 跳过 | ✅ 干净跑完，`frozen_f_k` 跟原来 EM 开着那次几乎一致（差 ~1 kJ/mol，10 帧 smoke 噪声范围内） |
| 1 | 0.6531 | 开着 | ❌ fail-closed（见 §2） |
| 1 | 0.6531 | 跳过 | ✅ 干净跑完（见 §3.1） |
| 2 | 0.5088 | 开着 | ✅ 干净跑完，`frozen_f_k` 正常无异常 |
| 3 | 0.3768 | 开着 | ✅ 干净跑完 |
| 4 | 0.2647 | 开着 | ✅ 干净跑完 |
| 5 | 0.1182 | 开着 | ✅ 干净跑完 |

**结论：6 个窗口里只有 window_1 一个会崩，不是这一段 λ 区间的系统性问题。**
一开始猜的"λ_shield≈0.5 这个防护壳系数峰值点扛不住"不成立——window_2 的
`lambda_shield=0.5088`，比 window_1 更贴近 0.5，反而完全没事。这更像是
window_1 这一份具体的冻结 checkpoint（`.../window_1/openmm.chk`）自身带着某种
特有的、跟这次 EXP-029 重新构建的系统（新的 λ 边界、新接入的残差力）搭配后
会释放的局部应力，不是"这一段 λ 普遍危险"。是不是真的仅限这一份 checkpoint、
换一个 repeat 的 window_1（不同 seed、大概率不同的冻结构型）会不会一样崩，
还没测——这是唯一还剩的、值得在合并主线前确认一下的点。

### 3.3 跨 repeat 补测结果（2026-08-24）

| repeat_index | seed | window_1（EM 开着，真实入口原样跑） |
|---|---|---|
| 0 | 20260905 | ❌ fail-closed（可复现，见 §2） |
| 1 | 20260906 | ✅ 干净跑完 |
| 2 | 20260907 | ⏳ 该 repeat 的 source（`output_lrc_fix_repeat03_seed20260907`）
   6 个 `production_window/vdw/window_*/openmm.chk` 当前全部缺失——该 repeat
   的源头生成还没跑完，不是数据损坏，等它跑完后需要补测这一格 |

**同一个 window_1、同一套 λ 梯度，repeat_0 崩、repeat_1 不崩**——证实了 §2.3
"是 repeat_0 这一份具体冻结 checkpoint 的特例，不是这段 λ 区间/这个协议的
系统性问题"这个结论。repeat_2 等它的 source 生成完之后需要补上这一测，把
三个 repeat 全部确认完。

### 3.4 还没验证的部分（下一步）

1. repeat_2（seed 20260907）的 window_1 待补测（见 §3.3，阻塞在它自己的
   source 还没生成完，不是这次排查能解决的）。
2. 还没有做成正式的、带版本号的协议参数——如果决定采用，需要在
   `ibs_engine.py` 的 `run_all_windows` 里加一个显式参数（比如
   `skip_energy_minimization`），默认关闭保持旧行为字节不变，只在
   EXP-029/030 的 cold_start 协议里显式打开，并按本项目惯例把这个算法变化
   焊进一个新的、递增的协议版本号（不是靠 code_sha256，见
   `[code_sha256级联失效反复出现的bug]`）；鉴于 §3.2 显示这是单窗口特例，
   也可以考虑不做成全局参数，而是针对这一个 window/repeat 组合单独处理
   （比如换一个 seed 重新生成这份 checkpoint，看塌陷是否随之消失）。
3. 还没检查 baseline 臂在同一窗口是否也有类似的收紧趋势（没有 fail-closed
   检测器，无法从报错直接看出来，需要额外测）。
4. 完整 36-job production matrix（3 repeats × 6 windows × 2 arms）尚未启动。

---

## 4. 本次新增的诊断脚本（一次性，不属于生产流程）

- `scripts/diag_exp029_window1_minimization_drift.py` —— 分段最小化探针（未复现，方法有缺陷，见 §2.2 第 3 点）
- `scripts/diag_exp029_window1_minimization_drift_v2.py` —— 连续单次最小化+reporter 探针（未复现，见 §2.2 第 3-4 点）
- `scripts/diag_exp029_real_entrypoint_instrumented.py` —— 对真实入口打 monkeypatch 挂密度 reporter（**成功复现**，产出 §2.3 曲线）
- `scripts/diag_exp029_skip_em_test.py` —— 对真实入口打 monkeypatch 把 EM 变空操作（**验证跳过 EM 对 window_1 干净可行**）

这四个都不改 `ibs_engine.py`/`exp027_common.py`/`exp029_window_state_machine.py`
本身，只在诊断进程内 monkeypatch，随时可删，不影响生产路径。

---

## 5. 合并计划

本文件是中间记录。按用户 2026-08-24 的明确指示：**先不改生产代码**，等
（a）EXP-029 §3.2 剩余验证做完、（b）EXP-030 也跑完之后，两者一起合并进
主线版本号。合并时需要同时处理：本文件 §3.2 的开放项、`exp027_result.md`
状态块里 EXP-029 那一行的更新、以及是否要把"跳过 EM"正式做成协议参数。
