# EXP-031 IBS 偏置力 GPU 优化 —— 指路(正文不在主线)

> **已归档（2026-09-12）。** 本文三条结论有两条已被推翻，取代它的是
> [EXP-031_GPU_OPTIMIZATION_2026-09-09.md](../EXP-031_GPU_OPTIMIZATION_2026-09-09.md)（仍是 live，被 `abfe_core.py` 与两个测试引用）。
> 保留原文只为追溯当时的判断依据，**不是待办**。

---

> ## ⚠️ 已被取代 —— 本文件的两条结论已推翻，头条数字不可引用
>
> 看 **[EXP-031_GPU_OPTIMIZATION_2026-09-09.md](../EXP-031_GPU_OPTIMIZATION_2026-09-09.md)**。
> 具体哪些话不能再引用，见那份的 §4。留档不改，仅加此提示。

2026-09-04。**这份只是指路。完整记录、全部脚本与证据都在沙箱仓库里,不在主线。**

沙箱(与主线同源的独立副本,主线在本实验期间一行未动):

```
/home/ruigengji/ABFE_IBS/ABFE_IBS_CUDA/experiments/EXP-031_ibs_bias_fusion/
├── PROPOSAL_mainline_integration.md      ← **先读这份**:总账、三条路线、代价栏、必做验收项、证伪清单
├── PLAN_EXP-031_ibs_bias_fusion.md       ← 立项计划书 + S0–S2 逐步执行记录(§0–§13)
├── PREREG_S2_G2_cuda_vs_reference.md     ← G2 的事先注册容差
├── benchmarks/  validation/              ← 可复跑的基准与验收脚本
└── results/
    ├── s2_g4_attempt2.json               ← 成本门判决
    ├── s2_exclusion_order_tax.log        ← 排除表顺序税
    └── independent_verification_by_abfe-ibs-5f/  ← 独立核查报告 + why_*.py 诊断脚本
```

新增的 CUDA 插件源码(**未接入生产,任何默认行为都没变**):
`/home/ruigengji/ABFE_IBS/ABFE_IBS_CUDA/plugins/IBSMixtureBias/`

## 三条结论(细节见沙箱)

1. **融合 CUDA kernel 判 STOP** —— 正确性全过(G0–G3),但比它要替换的 stock
   `CustomCVForce` **慢 16%**。原因:融合的真实上限本来只有 ~1.3×(不是 3×),
   而手写暴力扫描比 OpenMM 的共享邻居表多做约 509 倍距离计算。
   **G4 run 1 那个"1.897× 更快"是基线臂被污染的假象,不得引用。**
2. **意外收获,与 CUDA 无关:排除表顺序税。** `sync_all_exclusions`
   (`abfe_core.py:11098-11100`)以 Python `set` 迭代序灌排除表,使一个只覆盖 45×45 的
   配体内部力每步白烧 **1.1 ms(整步 30%)**。排序后 3.335 → 2.296 ms/step
   (**1.45×**),集合、条数、能量**逐比特不变**。修法是 `abfe_core.py:11099` 一行
   `sorted(missing)`。**主线尚未改。**
3. **摘掉恒零的 `cv_k_rest`** 可再拿 +0.377 ms/step(整步 1.16×),已在沙箱实现并三门
   验收 + 独立核查,但开关默认关闭。

## 动手前必须知道的三件

- 上面任一改动都会改 System XML(排序改的是排除表**书写顺序**)⟹
  **`system_xml_sha256` 变 ⟹ 现有窗口产物 / resume 全部失配**。
  不要因为"能量逐比特不变"就以为缓存能用。
- 全部数字是在"15 个水分子冒充 45 原子配体"的水盒上测的。
  **在真实配体拓扑上复验是提案的阻塞项,尚未完成。**
- 沙箱 `PROPOSAL_mainline_integration.md` 的 §6 有证伪清单(已被实测排除的假说),
  §7 有双方的错误更正记录 —— 动手前读一遍,能省掉重新论证。
