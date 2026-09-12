# 已关闭的代码缺陷（2026-09-09 前后收口）

> **已归档（2026-09-12）。** 本文是从 [TODO.md](../TODO.md)《未关闭的代码缺陷》整段移出的
> **六条已关闭条目**的完整记录，一字未改。它们**不是待办**。
>
> 保留原文的理由是每条都留下了一条仍然有效的规矩：
>
> | 条目 | 留下的规矩 |
> |---|---|
> | `MIGRATE-01` | 沙箱并线**别整份拷贝文件，逐 hunk 分拣** |
> | `PBC-01` | 分子归组只信 System，载入时删 `topology.cif` 往返造出的假键 |
> | `XFAIL-01` / `XFAIL-02` | xfail 的 `reason` 写错会让人查错方向；摘标记前先确认它红在哪一档 |
> | `CACHE-01` | `--openmm-cache-only` 下不得调用 GROMACS 解析 |
> | `CFG-01` | `abfe_config.json` 的 `gmx_path` 是机器本地值，不是模板 |

---

#### [x] MIGRATE-01（已关闭 2026-09-09）EXP-031 接入主线时整份拷贝文件会抹掉 09-09 的 MEM-15 重构

> 迁移已完成（详见下方原记录）。**本条剩下的价值是一条规矩、不是任务**：
> 下次沙箱并线**别整份拷贝文件，逐 hunk 分拣**。


> **已完成（2026-09-09）**：路线 A 与路线 B 都已接入主线，融合内核（S3）本轮不接。
> 详见 `docs/EXP-031_GPU_OPTIMIZATION_2026-09-09.md` 两节顶部的落地记录。
> `IBS_BIAS_PROTOCOL_VERSION` 已到 **33**，兼容集合收窄成 `frozenset((33,))`。
> 回归 `pytest tests -m cpu_only` → 1079 passed。
> ⚠️ `system_xml_sha256` 已变 ⟹ 既有 `dual_window_*` / `ibs_state_*` /
> `convergence.json` 全部失配，**必须重跑**，不要试图复用。
> 本条（MIGRATE-01）的价值转为**留给下一次沙箱并线**：别整份拷贝文件，
> 逐 hunk 分拣。

- **发现**：2026-09-09 逐文件对账沙箱 `ABFE_IBS_CUDA` 与主线时发现。
- **事实**：三个文件里**只有 `ibs_engine.py` 是沙箱领先**；`abfe_core.py` 与
  `abfe_pipeline.py` **主线在 2026-09-09 12:25 刚动过，比沙箱新**。

  | 文件 | 主线 mtime | 沙箱 mtime | 谁领先 |
  |---|---|---|---|
  | `ibs_engine.py` | 09-03 20:48 | 09-08 16:22 | 沙箱 |
  | `abfe_core.py` | **09-09 12:25** | 09-05 10:44 | **主线** |
  | `abfe_pipeline.py` | **09-09 12:25** | 09-04 15:17 | **主线** |
  | `runabfe.py` / `free_energy_engine.py` | — | — | 零差异 |

- **危险动作**：主线领先的内容是 MEM-15 分子归组重构 —— 新增
  `abfe_core.py:4645 system_molecule_grouping()` 与 `:4696 image_molecules_by_system()`
  （按 **System** 的键+约束求连通分子，不信 topology 的键），并把 `abfe_pipeline.py`
  里内联的「约束补成键」换成调这两个 helper。**这两个函数在沙箱里完全不存在**
  （全仓 grep 零命中）。⟹ 整份拷贝沙箱那两个文件会抹掉 135 + 56 行。
  而「把 EXP-031 合并到主线」最自然的执行方式恰好就是整份拷贝。
- **为什么是 P0**：这是**静默数据丢失**，不报错。抹掉的 MEM-15 修复防的是刚性水
  被逐原子回卷撕开 → 729 个 PME 排除对跨盒 → `Particle coordinate is NaN`（不到 1 ps）。
  而该损坏对既有诊断是隐形的：键能、最大键长、最小化后 max|F| 全部正常，
  只有查排除对距离才看得见。
- **处置**：**逐条重放，不是合并。** `abfe_pipeline.py` 里没有任何 EXP-031 内容
  （那条线从未改过它），它的 56 行差异 100% 属于主线，一行都不要往主线迁；
  反过来沙箱应去拉主线那份。
- **清单**：`ABFE_IBS_CUDA/experiments/EXP-031_ibs_bias_fusion/MIGRATION_CHECKLIST_2026-09-09.md`
  （逐条动作、行号、代价栏、不迁清单）；主线侧摘要见
  [EXP-031_GPU_OPTIMIZATION_2026-09-09.md](../EXP-031_GPU_OPTIMIZATION_2026-09-09.md) §1。
- **连带**：`docs/EXP-031_GPU_OPTIMIZATION_2026-09-04.md` 的两条结论已推翻、头条
  `1.45×` 是水盒数（真体系 1.162×）。已在该文件顶部加取代提示，正文留档未改。

#### [x] PBC-01（已关闭 2026-09-09）`topology.cif` 往返会凭空造出假键

- **发现**：2026-09-09，brd4/ligand1 benchmark（`abfe-benchmark/openmm_IBS/runs/brd4_ligand1/rep1`）。
  attachment 腿起点体检报「2 个 nonbonded_exceptions 对跨了周期镜像（最远 7.356 nm）」，
  而输入 `.gro` 干净（六个 target 实测 0 个跨镜像水、最大分子内跨度 0.096–0.100 nm）。
- **根因**：`app.PDBxFile` 写入端链 id 按 `chr(ord('A') + chainIndex % 26)` **循环**
  （`pdbxfile.py:392/473`），读取端 `_struct_conn` 只按 `(seq_id, asym_id, atom_name)`
  解析（`pdbxfile.py:218`），**不含链序号**。链数一超过 26 就有歧义。
  brd4/ligand1 有 12549 条链（每个水一条），第一个残基 `ACE(A,1)` 的三条键被解析到
  某个同样落在 `(A,1)` 的水上：

  ```
  (0, 39543) ACE1.CH3 — HOH12582.H1
  (0, 39544) ACE1.CH3 — HOH12582.H2
  (1, 39542) ACE1.C   — HOH12582.O
  ```

  `.top` 重建 27175 键，mmCIF 往返 27178 键，多的正好这 3 条；真键一条不缺
  （读取端 `createStandardBonds()` 补齐了，所以 3 条是**净多出**）。
  溶剂腿只有 3 条链，不触发。
- **已修（2026-09-09，两步都做完了）**：

  1. `repair_pbc_molecule_integrity` 的分子归组改成只信 System
     （`abfe_core.image_molecules_by_system` / `system_molecule_grouping`），
     回卷后逐对复查、fail closed。零指纹变动。
  2. `_load_system_from_native_cache` 载入 mmCIF 拓扑后调
     `abfe_core.prune_topology_bonds_unsupported_by_system()`，把 System
     完全不认的键删掉（判据宽：`HarmonicBondForce` ∪ `CustomBondForce` ∪
     `constraints`；漏删无害、误删致命）。同时两处裸 `traj.image_molecules()`
     换成 `image_molecules_by_system()`。

  原来记在这里的两个残留消费者，第 2 步一并解决了：

  | 位置 | 影响 |
  |---|---|
  | `ibs_engine.py` `compute_u_kn` 里的 `traj.image_molecules(inplace=True)` | **载荷相关**：重算 u_kn 之前给轨迹回卷，用的是 topology 的键，而且**连约束都没补**（比修好前的生产路径还弱），外面还包着 `except → warning` 的 fail-open |
  | `runabfe.py` 末帧 Boresch 诊断处的 `traj.image_molecules(inplace=True)` | 诊断用，末帧不重锚，影响面小 |

  `runabfe.py` 构造 Boresch 锚点图时会把**非配体键原样拷贝**
  （`for a, b in pipeline.topology.bonds()`），假边原本因此进了受体侧的锚点搜索图；
  拓扑在载入时就删干净了，这里不用再改。
- **本条最初写的方案（从 `.top` 重建拓扑）没有采用，理由记在这里，别再退回去**：
  删假键比换拓扑源好三点 —— 不需要 `.top`（`--openmm-cache-only` 也能用）、
  不引入任何新边（换成 System 的全图会多出 12478 条水的 H–H 边，
  可能干扰按键距计数的逻辑，如 `LIGAND_INTERNAL_POLAR_MIN_BOND_SEPARATION`）、
  且删完的键集与 `.top` **逐条相同**（实测 27178 → 27175 == `.top` 的 27175）。
- **代价：比原估的小得多，且不需要协议版本号 bump。**
  `abfe_pipeline._topology_hash()`（`abfe_pipeline.py:599`）把 bond 列表 +
  chain.id + residue.index 一起哈希，出现在 6 处 `_protocol_fingerprint(...)`
  （`abfe_pipeline.py:5710, 9363, 9564, 10224, 10554, 12421`）。但删键是**条件触发**的：
  一条都不用删时原样返回**同一个 topology 对象** ⟹ mmCIF 干净的体系
  （链数 ≤ 26，例如所有溶剂腿）`topology_sha256` 逐位不变、resume 全保住；
  只有真的带假键的体系指纹才变，而它们的既有结果本来就不可信。
  **不要再叠一个全局协议号 bump** —— 那会把没受影响的体系一起作废，
  正是「同一件事两套机制」那个反复踩过的坑（见 `code_sha256` 那条教训）。
- **物理量影响：无。** topology 不进哈密顿量，能量/力逐比特不变；唯一真实的坐标变化
  就是把被撕开的分子拼回去（以及随之而来的 ~0.005 nm 整体质心平移）。
  例外是 `compute_u_kn`：撕开的分子原本会给出跨盒 PME 排除对的错能量，
  现在不会了 —— 只在原本就错的情形下变，且回卷失败从 fail-open 改成直接抛。
- **验证**：从 `.top` + `.gro` 原始输入端到端复现整条机制（不依赖任何 run 产物），
  往返后多出 3 条、删完与 `.top` 键集逐条相同、原子/残基/链/盒矢量全保持；
  干净拓扑返回同一对象且哈希不变。回归 `pytest tests -m cpu_only`：1071 passed。
  测试：`tests/test_membrane_barostat_protocol.py` 的
  `test_image_molecules_ignores_phantom_topology_bonds` /
  `test_prune_drops_only_bonds_the_system_does_not_back` /
  `test_pbc_repair_groups_molecules_by_system_not_topology`。


#### [x] XFAIL-01（已关闭 2026-09-09）P1-19 的 C_seam 已修好，xfail 标记已摘除

- 位置：`tests/test_charge_transfer_real_endpoints.py:453` 的
  `@pytest.mark.xfail(strict=False)`，挂在
  `test_vanishing_lambda_one_seam_matches_charging_lambda_zero` 上。
- reason 自述「实测 118.5 kJ/mol（中性 4 原子 fixture）…… **修复后此标记应转
  XPASS 并摘除**」。它**现在正是 XPASS**，但 `strict=False` ⟹ 套件不会提醒任何人摘。
- **实测（2026-09-02）**：`tools/diagnostics/probe_p119_charge_transfer_seam.py`

  ```
  abs_delta_e = 3.63206042e-04 kJ/mol      ← 不是 118.5，小 5.51 个数量级
  rel_delta_e = 1.807e-06                  (门 1e-05)
  max|ΔF|分量 = 2.526e-06 kJ/mol/nm        (门 1e-03)
  ```

- **剩下这 3.6e-4 不是 seam 残余**：同文件
  `test_bake_handoff_seam_matches_for_charged_ligand_with_realistic_geometry`
  上方的注释精确描述过它——紧凑几何（配体 4 原子挤在 <0.2 nm 内）自带一个
  「与几何基本无关的 ~0.0005 kJ/mol 绝对残差」，数值性的，不是 Hamiltonian
  构造错误。量级吻合。**没有这条排除性说明，下一个人会以为 3.6e-4 是 seam 残余。**
- **已排掉「fixture 绕过失效路径」**（这是「真修好」与「绕过去了」的唯一分界）：
  `LIGAND_CHARGES_NEUTRAL_E = (0.5, 0.3, -0.4, -0.4)` 逐原子非零，
  `LIGAND_ORDINARY_PAIRS = {(0, 3)}` 是真正的 ordinary L-L 对（未定义任何
  exception、走标准 combining rule，q_i·q_j = −0.2 e²）⟹ 内部库仑真实存在、
  **机制被触发**，但常数不见了。
- **谁修的、什么时候修的：未知。** 跨会话核对过时间线，只能**排除**：不是
  2026-09-02 那两个会话中的任何一个，也不是 λ-WCA 壳退役、也不是力组切分收敛
  （`IBS_E_BASE_FORCE_GROUPS`/`IBS_E_BIAS_FORCE_GROUPS`）的连带效果——那天第一次
  全套跑之前 seam 就已经 XPASS。⚠️ **"未知"就是未知**，不要把它写成「大概是某次
  改动的连带效果」——那种猜测会被后人当结论。
- **已处置（2026-09-09）**：标记已摘除，原 reason 里的实测数据与「3.6e-4 是紧凑几何数值底噪、不是 seam 残余」这条排除性说明改写成函数上方的注释保留。
  `pytest tests/test_charge_transfer_real_endpoints.py` → 39 passed。
  ⚠️ 仍待人工确认：P1-19 在 issue 追踪里的状态该不该一起关（本仓没有 issue 追踪器，需要在上游做）。

#### [x] XFAIL-02（已关闭 2026-09-09）两个 xfail 的 reason 是错的：它们红在 D，不在 C——根因已查明并修复

- 位置：`tests/test_charge_transfer_real_endpoints.py:633` 与 `:822`
  （`test_run_protocol_v2_matrix_cd_wiring_passes_on_charged_fixture`、
  `test_run_protocol_v2_matrix_cd_normalizes_c2_style_switch_before_c_seam`）。
- 两条的 reason 都写「同 …… 的 xfail 理由」，即都记在 P1-19 的 C_seam 失配上。
  **这是错的。**
- **实测（2026-09-02）**，复现方式
  `python -m pytest tests/test_charge_transfer_real_endpoints.py --runxfail -q`，
  两条的 `failed_frames` 完全一致：

  ```
  'failing': ['D:gate1_reference_identity,gate3_mixed_production_vs_reference']
  ```

  **前缀是 `D:`。整个输出里 `failing` 一次都没出现 `C:`** ⟹ C（seam）在这两条里
  也是通过的，红的是 **D 端点**（全解耦：λ_coul≡0 且 λ_vdw=0）。
- 讽刺的是 `:822` 那条测试名叫 `..._normalizes_c2_style_switch_before_c_seam`
  ——它本身是为 C seam 写的，却卡在 D。
- 为什么与 XFAIL-01 是**不同机制**：这两条的 fixture 是 `_case(1, n_dummies=1)`
  （净电荷 +1 + reserved dummy），走完整 `run_protocol_v2_matrix_cd`，即
  **co-alchemical charge-transfer** 路径（配体 +1 e → 0、co-ion 0 → +1 e、
  flat-bottom 位置限制 k=100 kJ/mol/nm²）。失败的是 co-ion 在 λ=0 时的
  reference identity 与 mixed(CPU)-vs-reference 一致性，跟「配体内部库仑常数」
  没有关系。
- ⚠️ **不是静默的生产 bug**：`charge_treatment=co_alchemical_charge_transfer`
  本就 `production_qualified=False`，PHY-03（P1，见本节下方）仍挂着。属于
  **已知未合格路径上的已知未合格行为**，只是被错标成了 P1-19。
  **别当 P0 处理。**
- **已处置（2026-09-09）：不是重写 reason，是直接修好了根因；两个标记已摘除。**

  上面那段「跟 co-ion 有关」的推断**是错的，别再沿用**。逐 gate 打开 report 后
  （`gate1`/`gate3` 都给出同一个数）：

  ```
  D  abs_delta_e = 22.057534 kJ/mol   rel = 5.108e-05 (门 1e-05)
     max|ΔF| = 1143.130726 kJ/mol/nm  全部落在原子 0 的 x 分量
     reference 侧该原子受力 = 恰好 0.0     production 侧 = -1143.13
     strict_zero_reference / strict_zero_mixed 都 PASS ⟹ 配体–环境项确实是零
     C 两个 gate 都 PASS，abs_delta_e = 5.2e-4（就是 XFAIL-01 那个紧凑几何底噪）
  ```

  手算 fixture 里唯一的 ordinary L–L 对 `(0,3)`（Lorentz-Berthelot，
  r=0.265149 nm、σ=0.34 nm、ε=0.36 kJ/mol）：

  ```
  U_LJ(0,3) = 22.057512 kJ/mol      |F| = 1143.129513 kJ/mol/nm      方向 +x
  ```

  六位有效数字吻合 ⟹ **差值 100% 就是普通 L–L 对的内部 LJ**。

- **根因**：`tools/validation/compare_charge_transfer_endpoints.py`
  的 `reference_vanishing_zero_system` 把配体粒子 epsilon 一律置零，
  连**普通（无 exception）L–L 对**的 LJ 一起杀掉了；生产侧 `U_common` 正确地
  逐 λ_vdw 保留它。是 XFAIL-01（库仑版）的 **LJ 版本**，但错在**参照构造侧，
  不在生产侧**。
- **为什么会出现**：v3 [P0-01] 停掉了 `reference_charging_endpoint_system` 的
  内部对补 exception——对**库仑**是正确的（普通 L–L 库仑必须随粒子电荷线性湮灭），
  但普通对的 **LJ** 从此失去了庇护。该函数的 docstring 当时还写着
  「配体内部 LJ 已经被冻结成显式 exception」，已经陈旧。
- **已修**：置零之前先把普通 L–L 对冻结成显式 exception，搬进 combining-rule 的
  σ/ε；chargeProd 取当前粒子电荷之积（λ=0 时为 0）⟹ 库仑逐比特不变，
  PME 的 exception 倒扣项同样正比于 q_i·q_j ⟹ 也是 0。
  只补普通对，raw 拓扑自带的 excluded/1-4 exception 一个不动 ⟹ 没有触碰 P0-01。
- **影响面**：该常数逐 λ_vdw 不变 ⟹ **ΔG_vdw / ΔG_bind 不受影响**；且只改验证工具，
  生产哈密顿量一行未动，协议版本号未动。
- **定级结论：既不属于 PHY-03，也不是生产 bug**，是验证参照的构造缺陷。
  PHY-03 仍然独立挂着，状态不变。
- **回归**：新增 `test_reference_vanishing_zero_keeps_ordinary_intra_ligand_lj`
  直接对着机制断言（普通对必须存在且 σ/ε 等于 combining rule，配体粒子 epsilon 仍须为 0）——
  原来的 D 门失败信息只说 `D:gate1/gate3`，看不出是 LJ，正是这次查了半天的原因。
- **为什么这条值得单列**：三条 xfail 共用一条错 reason，是**能自我掩盖的**——
  谁照那两条的 reason 去修 C seam，会去修一个已经修好的东西；而真问题
  （co-ion 在 D 端点）继续没人管；而且它不会在测试里报警，因为 XFAIL 也算"预期"。

#### [x] CACHE-01（P2，纯噪音）`--openmm-cache-only` 下无条件调用 `find_gmx_include_dir`

**已修（2026-09-02）。** `find_gmx_include_dir(config.gmx_path)` 挪进了非
cache-only 的 `else` 分支，cache-only 路径 `include_dir` 留 `None`
（`runabfe.py` 里搜 `[CACHE-01`）。cache-only 下不再打那条警告。

- 原症状：带 `--openmm-cache-only` 跑时照样打「找不到 GROMACS 力场 include 目录」
  警告，而这一路根本不需要 include 树。
- 修改前已查清：审计通过的缓存上 `include_dir` **一次都不会被解引用**（三条使用
  路径逐个核对过，见 [TROUBLESHOOTING.md](../TROUBLESHOOTING.md) 同名小节）。
  ⟹ 只是噪音，不影响任何数值，挪动对 cache-only 路径行为中立。
- 挪动后复核过的那一条：`system_cache_exists(...)` 仍然在 `or` 的右侧，
  `openmm_cache_only=True` 时整个调用不执行（短路顺序未变）。
- 来源：2026-09-02 运行期记录（原文已归档到
  [archive/RUNTIME_ISSUES_2026-09-02.md](RUNTIME_ISSUES_2026-09-02.md) BUG-1）。

#### [x] CFG-01（P2，配置）`abfe_config.json` 的 `gmx_path` 指向不存在的路径

- **2026-09-07 已修**：发布清理时清空为 `""`（机器本地键，见
  `abfe_diagnostics._MACHINE_LOCAL_KEYS`），留空回退到 `GMXLIB`/`GMXDATA`/`PATH`
  自动探测。下面是当时的记录。
- 原值 `/home/ruigengji/gmx26.0C`，**本机不存在**。
- 2026-09-01 那次实跑的 provenance 记的是
  `/home/ruigengji/gmx26.3/share/gromacs/top`，与 config 里的值不同 ⟹ 配置里的
  值从来没被那次运行用上（那次带了 `--openmm-cache-only`）。
- 与 CACHE-01 **是两件事**：CACHE-01 是「不该问」，这条是「问了但答案是错的」。
  非 cache-only 路径会真的用到它。
- 修法：写 GROMACS 的**安装前缀**，不要写 `share/gromacs/top`
  （解析逻辑见 `runabfe.py:557` `find_gmx_include_dir`，两种写法都能吃，但前缀是
  2026-08-31 之后的约定）。**改配置会动 provenance，需用户确认取哪个版本的 GROMACS。**

