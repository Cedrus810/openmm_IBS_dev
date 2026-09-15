# 已关闭的待办（2026-09-12 收口）

> 从 [TODO.md](../TODO.md) 整段移出的**两条已关闭条目**，**都不是待办**。
>
> 保留原文的理由是每条都留下一条仍然有效的规矩：
>
> | 条目 | 留下的规矩 |
> |---|---|
> | `BOR-01` | 同一个几何量，**"写进哈密顿量的那份"和"做校验的那份"必须共用一个实现** —— 校验那份自己解了缠、提交那份没解 ⟹ 分歧被校验函数自己掩盖掉，这种形状最难发现（跟 λ 身份那次"四份实现"是同一个病） |
> | `S2-D` | 收拢的判据不是"都塞进一个文件"，是**决策同源的进 `abfe_preoptimizer`、写盘的留 `abfe_pipeline`**；"执行器"的定义是**写盘**，由测试钉住、不靠文档约定 |

---

## [x] `BOR-01`（已关闭 2026-09-12）同一组六原子 Boresch 几何有两份实现，minimum-image 口径不一致

**原文**（TODO §7）：
  - `ibs_engine.py:22491` `_check_boresch_geometry_safe`：逐跳
    `_minimum_image_displacement_nm` 解缠后再算 r0/θ（`:22504`）。
  - `abfe_core.py:10743` `calc_boresch_from_last_frame`：裸
    `np.linalg.norm(a-b)`，**不接 box、不做 minimum image**（r0 在 `:10755`）。

  **写进限制势的是后者**（`new_eq`），做校验的是前者 ⟹ 锚点对一旦跨周期边界，
  提交的 r0 静默差一个盒矢量（~4–5 nm），而校验函数因为解了缠**看不出分歧**。

  四个调用点里只有 `runabfe.py:3798` 安全（前面刚跑过 `image_molecules_by_system`
  + `center_coordinates`）。`abfe_pipeline.py:7040` / `7191` 直接喂 `self.positions`
  —— 该对象按 `abfe_pipeline.py:2986` 的注释"会被 PBC 修复、居中、再平衡反复改写"，
  调用时刻没有成像保证；`ibs_engine.py:2388` 喂的是**运行中的 context state**，
  解耦配体在窗口里漂过边界正是最可能触发的场景。

  跟"同一不变量多份实现"是同一个形状（λ 身份那次是四份）。

  **改法**：给 `calc_boresch_from_last_frame` 接 box，走 `abfe_core.py:1524`
  的 `minimum_image_displacement_nm` —— **一处解缠**，不在 5 个调用点各加守卫。

  **判据**：
  1. 两个函数在同一份跨边界坐标上给出**逐位相同**的 r0/θA/θB；
  2. 新增回归：构造一份锚点跨边界的坐标，旧实现会差约一个盒矢量、新实现不差；
  3. 不跨边界的既有坐标上 r0/θ/φ **逐位不变**（否则会作废全部 Boresch 缓存）。

  来源：GitHub #40（R-04 的第三项）2026-09-12 的核查。

---


---

### BOR-01 关闭记录（2026-09-12）

**改法**（就是当初写的那条）：新增**唯一**解缠实现
`abfe_core.unwrap_boresch_anchors_nm(rec_coords, lig_coords, box_vectors)`，
解缠链与原校验实现一致（`H0→H1→H2`、`H0→L0→L1→L2`，逐跳）。

关键取舍：平移量取**整数格矢**
（`raw - ((raw-ref) - minimum_image(raw-ref))`），而不是 `ref + minimum_image(raw-ref)`。
不跨边界时格矢恰为 `0.0` ⟹ 返回坐标与输入**逐位相同**，既有 Boresch 平衡值
不会被一次浮点重排整体作废（判据 3）。

| 改动 | 位置 |
|---|---|
| 唯一解缠实现 + `box_vectors_to_nm_array` | `abfe_core.py`（`minimum_image_displacement_nm` 之后）|
| `calc_boresch_from_last_frame(..., box_vectors=None)` | `abfe_core.py` |
| `_check_boresch_geometry_safe` 改调同一份 | `ibs_engine.py` |
| `_box_vectors_to_nm_array` 变成 abfe_core 那份的别名 | `ibs_engine.py` |
| 5 个调用点全部传 box | `abfe_pipeline.py` ×3（`getattr` 兜 `__new__` stub）、`runabfe.py`、`ibs_engine.py` |

**判据逐条验收**（`tests/test_boresch_minimum_image_bor01.py`，5 passed）：

1. 两份实现在同一份跨边界坐标上给出一致的 r0/θA/θB —— `test_both_implementations_share_one_unwrap`
   （同时断言两边 `unwrap_boresch_anchors_nm` **是同一个对象**）；
2. 跨边界坐标上旧口径差约一个盒矢量（直接撞 `[3, 20] Å` 硬门）、解缠后不差 ——
   `test_unwrapped_geometry_matches_the_physical_conformer`；
3. 不跨边界的坐标上六个几何量**逐位不变** —— `test_contiguous_geometry_is_bitwise_unchanged`。

全套离线测试对比动手前**零新增失败**。


---

## [x] `S2-D`（已关闭 2026-09-12）Stage-2 控制器代码散在三个文件

**原文**（TODO §1《剩余缺口》，源：控制器 design §7 第 4 条）：

> **S2-D 代码仍散在三个文件** —— `abfe_preoptimizer`（`decide` / `read_aggregated`）、
> `abfe_pipeline`（执行器 + 若干 helper）、`ibs_engine`（两个纯函数）。
> 设计要求是**包在一起**，尚未收拢。

### S2-D 关闭记录

**结论不是"都塞进一个文件"**，是 **决策同源的进 `abfe_preoptimizer`、写盘的留
`abfe_pipeline`**。"执行器"的定义就是**写盘**，由 `test_controller_never_writes_anything`
钉住，不靠文档约定。

| 符号 | 位置 | 理由 |
|---|---|---|
| `relearn_epoch_required_steps` | → `abfe_preoptimizer` | 决策同源 |
| `segment_dirs_for_evidence` | → `abfe_preoptimizer` | 决策同源 |
| `lambdas_from_version_record` | → `abfe_preoptimizer` | 决策同源 |
| `existing_segment_names` | → `abfe_preoptimizer` | 决策同源 |
| `_legalize_tail_window` | 留 `abfe_pipeline` | 会 `append_version` / `record_tail_repartition_version` —— **写盘**。它用到的纯函数（插 λ / 尾段重分 / 版本记录）都在 preopt，但"合法化"这个**动作**是执行器的事 |
| `_solve_merged_segments_if_any` | 留 `abfe_pipeline` | 合并求解是**求解**不是决策；为满足归档规则给它加个回调参数，是为形式增加间接层 |
| `_latest_segment_dirs` | **已删除** | 不是"没人调所以清理"，是被 `segment_dirs_for_evidence` **取代**：真机实证「段号最大的段」可能根本没有目标窗口的帧（`vanishing_2` 是 win0-3 的部分段，拿它去重解 win4 的 f_k 既炸 loader 又逻辑不通），正确语义是「**有这个窗口数据的**最新段」。留着等于把一个已修的崩溃摆在下一个人手边 —— 原地留了墓碑注释 |
| `ibs_engine` 的三个纯函数/常量 | 留 `ibs_engine` | 本来就该在引擎侧 |
| `_run_stage2_autonomous` 主循环 | 留 `abfe_pipeline` | 它**是**执行器 |

**顺带修的真 bug**：`_relearn_epoch_required_steps` 是 `ABFEPipeline` 的类级
`@staticmethod`，却在 `_run_stage2_autonomous` 里被当**裸名字**调
（`HEAD~:abfe_pipeline.py:10841`）⟹ `RELEARN_FK_EPOCH` 分支一走到就 `NameError`。
搬成模块级后改成 `_pre.relearn_epoch_required_steps()`。
**那条分支此前从没真跑过。**

**归属由测试钉住**：`tests/test_stage2_controller_module_boundary.py`（6 条），
包括"那个 NameError 不许回来"。
逐符号现状同步在 [设计文档 §9.4](../STAGE2_CONTROLLER_DESIGN_2026-09-12.md)；
§9.5 是"以后再搬的话"的注意事项。

⚠️ **一处走过弯路，记下来免得重来**：本轮曾把 `_legalize_tail_window` 一并搬进
preopt（理由是它调的三个纯函数都在那边），又曾把 `_latest_segment_dirs` 当"死代码"
直接删。前者是错的 —— **它写盘**；后者删对了但**理由错了**（不是"没人调"，
是"语义被证伪、留着是个陷阱"）。两处都已更正，设计文档 §9.4 与 preopt
迁入处的注释块同步改过。
