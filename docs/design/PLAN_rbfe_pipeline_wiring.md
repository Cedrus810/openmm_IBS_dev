# RBFE 编排层接线计划（pipeline 还缺什么）

日期：2026-09-03  
状态：**盘点完成，未开工。**  
关联：[RBFE 接口与实现计划](PLAN_rbfe_interface_and_implementation.md) §4 / §7 / §8

> 本文只做一件事：把「RBFE 作为一条 pipeline 现在还缺什么」摆清楚。
> 下面每一条都在 2026-09-03 逐行核对过当前源码，带行号，不是转述。
>
> 背景：R0 / R1a / R1b / R2（一半）已经落地，`rbfe_core` 侧能力齐了；
> 但 `rbfe_pipeline.py` 那条线是 2026-09-01 写的，**还指着当时设想的接口**，
> 中间发生了漂移。本文按「现在能不能补」分类，不按「重要不重要」。

---

## 0. 一句话结论

**除了建系（`prepare_edge`）之外，所有缺口都不依赖那份还没有的配体 B。**
补完就是「缺一个建系函数就能端到端跑」的状态。

---

## 1. 最直接的断点：pipeline 与 core 的接口已经漂移

`rbfe_pipeline.run_leg`（`rbfe_pipeline.py:291`）现在长这样：

```python
prepared = rc.prepare_edge(spec)             # 未实现
leg = rc.build_hybrid_leg(prepared, phase)   # 未实现
artifacts = fee.run_sampling(request, leg.sampler, backend)
return rc.analyze_leg(leg, artifacts)        # 签名已经不是这个了
```

而 core 侧现在真正提供的是：

| 步骤 | 现在的入口 |
|---|---|
| 映射 | `rbfe_core.map_atoms`（`rbfe_core.py:1594`）+ `validate_mapping`（`:1747`） |
| 建 hybrid | `rbfe_core.build_hybrid_system`（`:2983`） |
| 采样 | `free_energy_engine.run_independent_windows`（`free_energy_engine.py:947`） |
| u_kn | `rbfe_core.compute_hybrid_u_kn`（`:3382`） |
| 分析 | `rbfe_core.analyze_leg(bundle, samples, *, phase, ...)`（`:3495`） |

**三个调用点全部对不上。** `analyze_leg` 的签名尤其：它现在吃
`(bundle, samples)` 加一串 keyword-only 参数，不是 `(leg, artifacts)`。

> 这次漂移已经造成过一次真实回归：`test_rbfe_core_r0.py` 里
> `test_unimplemented_stages_raise` 的参数化表还挂着 `analyze_leg`，
> 实现之后那条用例先抛 `TypeError` 而不是 `NotImplementedError`。
> 由同僚会话在跑全套时发现。**接口漂移不会自己暴露，只会在别处炸。**

**要做的**：重写 `run_leg`，改走上表的新接口。

---

## 2. 身份指纹有两个真洞（现在就能补，而且该马上补）

`edge_identity()`（`rbfe_pipeline.py:89`）有两个参数：

```python
atom_mapping_hash: Optional[str] = None,
hybrid_builder_version: Optional[str] = None,
```

核对结果：**全仓只有测试传过它们**（`tests/test_rbfe_pipeline_framework.py:82/133/134`），
生产代码一个调用点都没有 —— 实际运行中它们**永远是 `None`**。

后果：换了原子映射、换了 hybrid builder 版本，边身份**不变**，
于是 `assert_reusable`（`:249`）照样放行复用旧产物。
这正是计划 §7 明令禁止的「禁止跨 A/B 方向、**映射**、力场、后端或 λ 路径直接追加」。

两个值现在都是现成的：`AtomMapping.fingerprint()` 与 `RBFE_HYBRID_PROTOCOL_VERSION`。

> ⚠ 这个洞的形状值得单独记一笔：**「默认值 `None` 同时表示『没有这一项』和『没记录』」**。
> 前者合法（R0 阶段确实还没有映射），后者是事故。补的时候要让「已经有映射却没传」
> 变成**报错**，而不是继续沿用 `None`。

---

## 3. 计划 §7 那棵产物树，落了 2/6

§7 要求每条边每次重复独立保存：

```text
output_rbfe/A_to_B/repeat_01/
  edge_manifest.json          ✅ prepare_run_directory 写（:243）
  atom_mapping.json           ❌ 只有 `runrbfe.py map` 会写，pipeline 不写、不进 manifest
  endpoint_validation.json    ❌ verify_hybrid_endpoints 返回 dict，没有落盘点
  complex/                    ❌ hybrid System、状态表、样本、交叉能量、checkpoint、诊断，全无
  solvent/                    ❌
  rbfe_result.json            ✅ combine_legs 写（:360）
```

额外一处：`LEG_RESULT_NAME = "leg_result.json"`（`:61`）与
`RunLayout.leg_result_path()`（`:193`）都定义了，**但没有任何代码写它**。

逐项缺什么：

| 产物 | 现状 | 补起来要做什么 |
|---|---|---|
| `atom_mapping.json` | `AtomMapping.to_dict()` 已经产出全部内容 | pipeline 落盘 + 把指纹并进边身份（见 §2） |
| `endpoint_validation.json` | `verify_hybrid_endpoints` 已返回全部数字（含 `lj_lrc_gap_kJ_per_mol`） | 落盘；并决定「不过怎么办」——现在没有 fail-closed 接线 |
| hybrid System | `HybridSystemBundle.system` 在内存 | `XmlSerializer.serialize` 落盘（一行），进腿指纹 |
| 状态表 | `HybridLambdaSchedule.to_dict()` 有 | 落盘 |
| 样本 | `InMemoryWindowSamples`，**只在内存** | 需要落盘格式的决定（DCD？npz？）+ 读回路径 |
| 交叉能量 u_kn | `compute_hybrid_u_kn` 已经返回 §7 要求的**全部**元数据：维度、状态顺序、样本状态索引、是否约化、是否含 βpV | 只差写出去 |
| checkpoint | 完全没有 | 见 §4 |
| `leg_result.json` | 路径定义了没人写 | 落盘 `LegResult` + 诊断 |

---

## 4. 续跑只有「拒绝」，没有「继续」

`assert_reusable`（`:249`）做的是身份比对然后拒绝——**这部分是对的，不要改**。
但完全没有另一半：

- 没有 checkpoint（一个都没有）；
- 没有「已完成多少步」的记账；
- 没有「这条腿已经跑完了，跳过」的判断（`leg_result.json` 都没人写，自然也没人读）；
- 计划 §7 专门要求「**动态 checkpoint 与已完成采样缓存复用分开**」——
  两者都不存在，谈不上分开。

> 参考 ABFE 那边的教训：4W53 续跑反复失效的四个根因是**同一个形状**——
> **键里混进了非身份的东西**（代码 sha256、坐标数组、执行历史）。
> `edge_identity` 的注释已经把这条写进去了（刻意不放 code sha256、步数、平台），
> 补续跑时不要破坏它。

---

## 5. 另外五处

### 5.1 λ 表两处各说各话

`ProtocolSpec` 里是 `n_lambda_states: int` + `lambda_schedule_name: str`；
`rbfe_core` 里是 `HybridLambdaSchedule`（真表，带三条 λ）。
**没人把前者解析成后者，也没人校验二者一致。** 于是 `edge_identity` 里记的
`lambda_schedule_name` 可以跟实际建系用的 λ 表完全无关。

要做的：定一个「名字 → 表」的解析器，并在 `build_hybrid_system` 之前校验
`n_lambda_states == schedule.n_states`。

### 5.2 seed 派生规则没有

`free_energy_engine` **刻意不生成 seed**（`_resolve_window_seeds`，
`free_energy_engine.py` 内，拿不到 `integrator_seeds` 直接报错）。
所以派生规则必须由 pipeline 定义：`ProtocolSpec.seed` 只有一个标量，
而需要的是 `(repeat_index, phase, state_index) → seed`。现在这条链断着。

> ABFE 那边有 `Exp019SeedLedger` 可以参考其**形状**，但不要直接复用它的域
> （它的 `stage/leg/phase` 是 ABFE 的语义）。

### 5.3 质量门只有 overlap 一条

`analyze_leg` 现在把 `LegResult.quality_gate_passed` 直接取自 MBAR 的
`converged`（即 `min_overlap` 阈值）。计划 §7 要的是
「分析结合 overlap、**有效样本量、时间稳定性、交换混合和结合姿势诊断**；
不能只看交换接受率」。现在只有第一项。

### 5.4 独立重复没有驱动，两腿也没有

`aggregate_repeats`（`:408`）有，但没有「跑 N 个 repeat」的入口；
更基本的是**没有 `run_edge`**——`combine_legs`（`:329`）要两个 `LegResult`，
但没有任何代码生产它们（`run_leg` 是断的，见 §1）。

### 5.5 CLI 断三处，网络层完全没有入口

`runrbfe.py:449-451` 的 `prepare` / `run` / `analyze` 仍然是退出码 3。
`analyze_network`（`:575`）与 `absolute_from_anchor`（`:689`）实现完整、有测试，
但**全仓没有任何调用点**——CLI 也没有对应子命令。

---

## 6. 唯一真正被配体 B 卡住的：建系

`prepare_edge`（`rbfe_core.py:3618`）要从 `EdgeSpec` 的 `input_path` 建出
**两个 OpenMM System**，而 `_assert_environment_matches` 要求它们的环境
**逐位相同**（粒子数、质量、非键参数、成键项、约束）。

这不是「读个文件」，而是要一套确定性的建系 + 溶剂化流程：保证配体 A 和 B
放进**同一个盒子、同一批水和离子**。而 B 本身还需要重新给部分电荷与成键参数
——§5.1 不允许为了调第三方 builder 悄悄重参数化，所以这条参数化路线要单独验收。

**这条不解决，端到端跑不了；但它也不挡上面任何一条。**

---

## 7. 建议顺序

| 步 | 内容 | 依赖配体 B |
|---|---|---|
| P1 | 补 §2 的两个身份洞（映射指纹 + builder 版本进边身份，缺失变报错） | 否 |
| P2 | 重写 `run_leg` 接新接口；同时定 §5.1 的 λ 表解析与 §5.2 的 seed 派生 | 否 |
| P3 | 补齐 §3 的产物落盘 + §4 的「腿已完成则跳过」 | 否 |
| P4 | `run_edge` / 多重复驱动 / CLI 三个子命令 / 网络层入口（§5.4、§5.5） | 否 |
| P5 | 质量门补齐（§5.3） | 否 |
| P6 | `prepare_edge` 建系 | **是** |

P1 建议优先，因为它现在就是一个**能让人复用错产物**的活漏洞，而且改动最小。

---

## 8. 本文不涉及

- R2 剩下的一半（RBFE 的 REMD 路径）。障碍是
  `ibs_engine.REMDManager.__init__` 的签名写死了 ABFE 去耦语义
  （`lambdas_coul / lambdas_vdw / ligand_indices / boresch_params /
  co_alchemical_ion_spec`），RBFE 用不了，需要另写一个吃「通用状态表」的副本交换。
  见 [RBFE 计划](PLAN_rbfe_interface_and_implementation.md) 第 0 节的 R2 段。
- R3/R4 的科学验收（A→B 与 B→A 相容、三角闭合、生产 qualification）。
  那些要真实体系，前提同样是配体 B。
