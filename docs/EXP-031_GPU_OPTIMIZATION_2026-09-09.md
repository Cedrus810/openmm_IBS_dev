# EXP-031 IBS 偏置力 GPU 优化 —— 主线接入现状（2026-09-09）

**取代 [EXP-031_GPU_OPTIMIZATION_2026-09-04.md](archive/EXP-031_GPU_OPTIMIZATION_2026-09-04.md)。
那份的三条结论有两条已被推翻，头条数字也不能引用 —— 见 §4。**

正文、脚本与全部证据仍在沙箱，不在主线：
`/home/ruigengji/ABFE_IBS/ABFE_IBS_CUDA/experiments/EXP-031_ibs_bias_fusion/`

先读这三份：

| 文件 | 内容 |
|---|---|
| `MIGRATION_CHECKLIST_2026-09-09.md` | **动手前必读**：逐条重放动作、行号、代价栏、不迁清单 |
| `STATUS_2026-09-08_sm120.md` | sm_120 全套闸门 + 第一次真生产跑的效能数（§8） |
| `PROPOSAL_mainline_integration.md` | 路线定义、证伪清单、双方的错误更正记录 |

---

## 1. 🚨 P0：不要整份拷贝任何文件

**主线的 `abfe_core.py` 与 `abfe_pipeline.py` 在 2026-09-09 12:25 刚动过，比沙箱新。**

| 文件 | 主线 mtime | 沙箱 mtime | 谁领先 |
|---|---|---|---|
| `ibs_engine.py` | 2026-09-03 20:48 | 2026-09-08 16:22 | **沙箱**（EXP-031 全在这里） |
| `abfe_core.py` | **2026-09-09 12:25** | 2026-09-05 10:44 | **主线** |
| `abfe_pipeline.py` | **2026-09-09 12:25** | 2026-09-04 15:17 | **主线** |
| `runabfe.py` | — | — | 零差异 |
| `free_energy_engine.py` | — | — | 零差异 |

主线领先的内容是 **MEM-15 分子归组重构**：主线新增
`abfe_core.py:4645 system_molecule_grouping()` 与 `:4696 image_molecules_by_system()`
（按 **System** 的键+约束求连通分子，不信 topology 的键），并把 `abfe_pipeline.py`
里原先内联的「约束补成键」换成调这两个 helper。
**这两个函数在沙箱里完全不存在**（全仓 grep 零命中）。

⟹ 把沙箱的 `abfe_core.py` 或 `abfe_pipeline.py` 整份拷过来，会抹掉主线今天的
135 + 56 行重构。而「把 EXP-031 合并到主线」这句话最自然的执行方式恰好就是这个。

**正确做法是逐条重放，不是合并。**
`abfe_pipeline.py` 里**没有任何 EXP-031 内容**（这条线从未改过它），
它那 56 行差异 100% 是主线自己的东西 —— 一行都不要往主线迁。

---

## 2. 可以接入的两项（前置已全部关闭）

### 路线 A：排除表排序 —— ✅ **已接入主线（2026-09-09）**

> 落点：`abfe_core.py` 的 `sync_all_exclusions()`（行号已因当日 PBC-01 的
> 三个新 helper 下移，按函数名找）。上方补了 20 行说明，含实测口径与"为什么
> 修在这里而不是各产地"。守它的测试：`tests/test_exclusion_order_is_ascending.py`
> （静态、不需 GPU）—— 集合与能量都不变，**没有任何数值测试能守住这一行**，
> 只能直接断言写入顺序，否则下一个人会当整洁癖删掉。
> 回归：`pytest tests -m cpu_only` → 1073 passed。

```diff
-        for p1, p2 in missing:
+        for p1, p2 in sorted(missing):
```

`sync_all_exclusions` 以 Python `set` 迭代序灌排除表；OpenMM 的
`CustomNonbondedForce` 对**排除表的写入顺序**敏感。集合、条数、能量**逐比特不变**。

- **真体系实测 1.162×**（真 Atenolol 膜体系 45354 原子，三次独立测量）
- ⚠️ 09-04 那份文档的 **1.45× 是水盒工作点的数，不可引用**
- 沙箱实现在 `abfe_core.py:11099`，上方 22 行说明注释建议一起搬，
  否则下一个人会以为 `sorted()` 是整洁癖而删掉

### 路线 B：摘掉恒零的 `cv_k_rest` + 协议 v33 —— ✅ **已接入主线（2026-09-09）**

> 由 `abfe-ibs-cuda-d5` 从 28 个 hunk 里分拣出 S1-only 补丁（取 13 弃 15），
> 主线侧应用后与其独立构建的目标文件逐字节相同。落位：
> `IBS_BIAS_OMIT_ZERO_REST_CVS = True`（:4886）、`IBS_BIAS_PROTOCOL_VERSION = 33`
> （:6943）、`IBS_BIAS_CACHE_COMPATIBLE_PROTOCOL_VERSIONS = frozenset((33,))`（:6949）。
> 零 S3 残留（`grep -c "IBS_BIAS_USE_FUSED_KERNEL\|fused_force"` = 0），
> 且未倒退当日的 PBC-01 改动（弃掉的 15 个 hunk 里有 2 个正是那两处）。
> 新增测试 `tests/test_shadow_coul_ibs_builder.py`、
> `tests/test_exclusion_order_is_ascending.py`（两者都补了 `pytestmark =
> pytest.mark.cpu_only`，否则进不了常规门）。
> 回归：`pytest tests -m cpu_only` → **1079 passed**（= 修改前 1071 + 新增 8）。

主线当前：`ibs_engine.py:6799` v32、`:6805` `frozenset((32,))`，
**且完全没有 `IBS_BIAS_OMIT_ZERO_REST_CVS` 这个开关**。

⚠️ **这不是「翻个默认值」** —— 整套机制在主线根本不存在。沙箱 `ibs_engine.py`
相对主线共 **27 个 hunk**，S1（摘零 CV）与 S3（融合内核）交织，须按 hunk 分拣。

这三行必须同时进，缺一不可（实测）：

```
IBS_BIAS_OMIT_ZERO_REST_CVS                 = True
IBS_BIAS_PROTOCOL_VERSION                   = 33
IBS_BIAS_CACHE_COMPATIBLE_PROTOCOL_VERSIONS = frozenset((33,))
```

只升版本号不收窄兼容集合 ⟹ v33 与自己判不兼容 ⟹ **27 条 resume/缓存契约测试变红**。
另有 `tests/test_audit_protocol_regressions.py`、`tests/test_warmup_overlap_protocol.py`
两处锁版本号的契约测试要一起改。

- **真体系实测 1.2206×**，Group-1 能量三路径逐比特相同（精确恒等变形）

### 必须一起走的测试（主线全无）

- `tests/test_exclusion_order_is_ascending.py` —— 排除表升序静态断言，不需 GPU
- `tests/test_shadow_coul_ibs_builder.py` —— 两种形态（legacy 2K / omit K）

主线 `tests/test_core_physics_numerics.py` 的参考实现也要改成形态感知
（沙箱做法见其 `:696 _omit_zero_rest_cvs()`），否则那 7 条会被误读成回归。

---

## 3. 融合 CUDA 内核：**暂不接入**

09-04 那份说它「判 STOP、慢 16%」—— **该判定已于 2026-09-05 推翻**：
根因是缺了 OpenMM 基线有的那层邻居表（从 `libOpenMMCUDA.so` 挖出 `buildNeighborList`
实锤），补上后 0.859× → 1.394×，φ 层从 `mixed` 降到 `real` 后真体系整步 2.163×。
G2/G3/变异测试在 2080 Ti / 3090 / **RTX 5080（sm_120）** 三张卡上各过一次。

但四条合并前验收仍未关闭：

- [ ] **完整 ABFE 跑通并出 ΔG** —— 2026-09-09 那次死在分析层
      （`window_overlap_broken`：窗口 3 去相关后仅 7 有效帧 < 10 ⟹ 窗口被跳过 ⟹
      共享边界节点断链 ⟹ fail-closed 拒绝拼接），溶剂腿未走到
- [ ] **融合 vs legacy 的运行级对照** —— 同 seed 同配置各跑完整 ABFE、ΔG 误差内一致。
      G2 只给逐帧 parity。这也是唯一能洗清「窗口 3 帧数不足是否与内核有关」的测试
- [ ] **插件随包安装** —— `.gitignore` 排除 `*.so` 与 `plugins/*/build/`，而
      `ibs_mixture_bias.py` 默认路径指向沙箱的 `experiments/.../build/out`。
      主线上开开关就是 `FileNotFoundError`
- [ ] **协议版本再升一次** —— 开关是运行时的 ⟹ 同一个 v33 描述两种 System

---

## 4. 09-04 那份文档哪些话不能再引用

| 09-04 的说法 | 现状 |
|---|---|
| 「融合 CUDA kernel 判 STOP，慢 16%」 | ❌ 2026-09-05 推翻，见 §3 |
| 「排序后 3.335 → 2.296 ms/step（**1.45×**）」 | ⚠️ 水盒工作点；**真体系是 1.162×** |
| 「在真实配体拓扑上复验是阻塞项，尚未完成」 | ✅ 已完成 |
| 「主线在本实验期间一行未动」 | ❌ 主线已于 2026-09-09 12:25 前进，见 §1 |
| 「摘掉 `cv_k_rest` 可再拿整步 1.16×」 | ✅ 修正为 **1.2206×**（真体系实测） |

---

## 5. 动手前必须知道的（任何路线都逃不掉）

1. **`system_xml_sha256` 必变 ⟹ 现有窗口产物 / resume 全部失配、需重跑。**
   路线 A 虽然能量逐比特不变，但排除表在 System XML 里的**书写顺序**变了。
   **别因为「能量不变」就以为缓存能用。**
2. 路线 B 另需 v32→v33，`dual_window_*` / `convergence.json` 一律失配。
3. **别往协议指纹里加 `code_sha256`** —— 已被否决两次：改任意一行都会让 resume 重跑 GPU。
4. **重取基线用当前主线**（09-09 12:25 那份），别引用任何旧快照。

## 6. ⚠️ 报数口径：`1.39×` 是每步、K=16；端到端只有 `1.09×`

A+B 合入后主线净收益 ≈ **1.39×**（1.162 × 1.2206，加性估算，**未跑四臂对照**）。
但这是**每一步 MD** 的比值，不是整条 ABFE 的加速。2026-09-09 第一次在真膜体系上
跑完整生产管线，复合物腿端到端只有 **≈1.09×**：

| 阶段 | 时长 | 占比 | 受加速 |
|---|---|---|---|
| 预平衡 2.5e6 步 | 16:10 | 21% | ❌ 无 Group-1 |
| Boresch 再平衡 / attachment / λ 布点 | ~10:30 | 14% | ❌ |
| **decharging REMD** | **20:44** | **27%** | ❌ 走 `NonbondedForce` ParameterOffset 保 PME，独立分支，**压根不建 Group-1** |
| **vanishing** | **30:08** | **39%** | ✅ |

外加一条：真实 λ 布局给出的是 6 个 **K=4–5** 的窗口，而上述倍数全在 **K=16** 上量的。
融合内核的成本对 K 近乎免疫（0.118 ms @ K=4–5 vs 0.154 ms @ K=16），
legacy 随 K 线性（每态约 0.1 ms）⟹ **K 越小差距越小**，K=5 上每步只有约 1.39×
（fused 端实测 0.4287 ms/step，离散度 0.19%；legacy 端为外推）。

**引用这些数字时必须带上口径。**
