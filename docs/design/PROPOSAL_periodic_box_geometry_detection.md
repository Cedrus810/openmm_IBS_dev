# 提案：非长方体周期盒的早期识别与处理

日期：2026-09-02
状态：**提案，待决定。未实施，未写任何代码。**
关联：`abfe_core.validate_membrane_input`（膜路径已有的盒型门）

> 本文回答两件事：**(1) 怎么可靠地区分不同盒型；(2) 怎么在建 System 之前就做到。**
> 下面所有代码位置都在 2026-09-02 逐条核对过当前源码，不是转述。
>
> **用户已定的范围**（2026-09-02）：本轮只写文档不动代码；实施时的重点是
> **识别 + 处理能力**，不做验收协议、不做 fail-closed 拒跑门。

---

## 1. 问题从哪来

膜体系的盒型被硬性约束成长方体（`abfe_core.py:5108-5113` 已 fail-closed）。
但**可溶体系的输入盒可以是任意形状**，而且非长方体是常规做法：

| 来源 | 命令 | 盒型 | 相对立方盒体积 |
|---|---|---|---|
| AMBER / tleap | `solvateoct` | 截角八面体 | ~77% |
| GROMACS | `editconf -bt dodecahedron` | 菱形十二面体 | ~71% |
| GROMACS | `editconf -bt octahedron` | 截角八面体 | ~77% |
| CHARMM-GUI 可溶 | 默认 | 立方 / 长方体 | 100% |

同样的最小镜像距离下省 23~29% 的水 ⟹ 做可溶 ABFE 的人**有动机**这么建盒。
当前这个库对它们**没有任何识别**：不报告、不分类、直接跑。

---

## 2. 核心难点：不能靠盒矩阵字面值分类

**同一个格子有无穷多种盒向量表示。** 具体地：

- `tleap solvateoct` 的截角八面体 → rst7 里是 `a=b=c`, `α=β=γ=109.471°`，
  经 OpenMM 转成向量后是一种矩阵；
- `gmx editconf -bt octahedron` 的截角八面体 → `.gro` 里直接是 9 个数的盒矩阵，
  **和上面那个矩阵长得完全不一样**；
- 两者是**同一个 BCC 格子**，物理上等价。

再加上 barostat 会把盒子压歪一点点（各向同性缩放保形状，但数值不会精确）
⟹ 任何「看非对角元是不是 0」「看角度是不是正好 109.471」的判据都是脆的。

所以分类必须走**表示无关的不变量**。

---

## 3. 提案的三层结构

### 3.0 先纠正一个常见预期：`.gro` 并不声明盒型

`.gro` 的最后一行给的是**盒向量**，不是盒型标签：

- **3 个数** → `v1x v2y v3z`，对角矩阵 ⟹ **这一种情况格式本身就定死了是长方体**。
- **9 个数** → `v1x v2y v3z v1y v1z v2x v2z v3x v3y`（注意：**不是行主序**）
  ⟹ 只知道「是三斜」，**是十二面体还是截角八面体、还是一般三斜，文件里没写**。

所以「用户提交了什么盒子」这个问题，对 `.gro` 来说只能**从向量反推**，
而反推必须走 §3.2 的归约（理由见 §2：同一个格子有无穷多种矩阵表示）。

全仓唯一**直接声明**盒型的格式是 AMBER `.parm7` 的 `%FLAG POINTERS` → `IFBOX`
（`1` 长方体 / `2` 截角八面体 / `3` 一般三斜）。它可以用来**交叉核对**我们自己
反推出来的结论——但数值以 `.rst7` 为准（prmtop 里的是建系时刻的，可能已过时）。

### 3.1 第一层：从各种格式里把盒子读出来

| 格式 | 盒子在哪 | 给的是什么 | 陷阱 |
|---|---|---|---|
| `.gro` | **最后一行**，3 个或 9 个浮点数 | **盒向量** | 9 个数时 GROMACS 的排列是 `v1x v2y v3z v1y v1z v2x v2z v3x v3y`——**不是行主序**。`app.GromacsGroFile` 已正确处理；自己解析必踩 |
| `.rst7` / `.inpcrd` | 最后一行 | **长度 + 角度** `a b c α β γ` | 截角八面体是 `α=β=γ=109.471°`。必须转向量，不能直接用 |
| `.prmtop` / `.parm7` | `%FLAG POINTERS` 的 `IFBOX` + `%FLAG BOX_DIMENSIONS` | **盒型枚举 + 长度** | `IFBOX = 1` 长方体 / `2` 截角八面体 / `3` 一般三斜。**这是唯一直接声明盒型的格式**——可以拿来交叉核对我们自己的分类结果，但数值以 rst7 为准（prmtop 里的是建系时的，可能已过时） |
| PDB | `CRYST1` 行 | **长度 + 角度** | 同 rst7；没有 `CRYST1` 就是没盒子 |
| mmCIF（本库原生缓存） | `_cell.*` | **长度 + 角度** | `runabfe.py:1159` 已在读，并与 `box_vectors.npy` 对账（`:1166`） |
| `system.xml`（本库原生缓存） | `<PeriodicBoxVectors>` | **盒向量，已是 reduced form** | 权威，但只在缓存命中时存在 |
| `box_vectors.npy`（本库） | `(3,3)` 数组 | **盒向量** | `runabfe.py:1577` 落盘、`:1121` 记 sha256 |
| DCD / XTC | 每帧 unitcell | 长度+角度（DCD） | mdtraj 给 `unitcell_vectors`；读不到时本库有四处静默退化成正交盒，见 §5.2 |

**长度+角度 → 向量的转换只有一个正确入口**：
`openmm.app.internal.unitcell.computePeriodicBoxVectors(a, b, c, alpha, beta, gamma)`。
它返回的就是 OpenMM reduced form。反向是 `computeLengthsAndAngles()`。
**绝不要自己写那套三角公式**——α/β/γ 到底哪个角对哪两条边、以及归约那一步，是
经典错误来源。

> ⚠️ 待在装了 openmm 的机器上确认：同模块的 `reducePeriodicBoxVectors()`
> （用于「已有向量但可能非归约」的输入，例如手工构造的 `.gro`）。
> 本会话的机器上没有 openmm 环境，这个函数名与签名**未实测**。
> `computePeriodicBoxVectors` 是确定存在的（`AmberInpcrdFile` / `PDBFile` 内部就用它）。

现状：**本库生产路径只读 `.gro`**（`runabfe.py:2350/2378`、`abfe_core.py:10467`、
`abfe_pipeline.py:12212`）。AMBER 有一个最小加载层
`abfe_core.load_amber_topology_for_openmm`（`:3417`），但它自己的 docstring 就写着
「尚未接入 CLI/生产流程」，且**完全不碰盒子**——`inpcrd.boxVectors` 拿到就拿到，
没有任何检查。所以 §3.1 这张表里除 `.gro` 和原生缓存以外的行，都是**新能力**。

### 3.2 第二层：格基归约到规范形

拿到盒矩阵 `B`（行向量约定，nm）之后：

1. **Minkowski 归约**（三维下等价于 Niggli 归约）得到最短基 `B*`。
   三维 Minkowski 归约有确定性算法，且 `abfe_core.minimum_image_displacement_nm`
   （`:1415`）里已经有「在格点邻域搜索 + 用最小奇异值下界证明收敛」的模式可以复用
   思路（**不是复用代码**——那个函数解的是最近向量问题，这里要的是最短基）。
2. 算 **Gram 矩阵** `G = B* B*ᵀ`，除以 `G[0,0]` 归一化。
3. 对 `G` 的三行做规范排序（按对角元升序）+ 符号规范化（把非对角元的符号按固定
   规则翻正）。

得到的六个数（三个归一化边长平方 + 三个夹角余弦）就是这个格子的
**表示无关指纹**。上面 §2 那两种截角八面体表示，归约后指纹相同。

### 3.3 第三层：按指纹分类

| 盒型 | 归约后归一化 Gram 的对角 | 三个夹角余弦 | 格子 |
|---|---|---|---|
| 立方 | `1, 1, 1` | `0, 0, 0` | SC |
| 长方体 | `1, β², γ²`（不全相等） | `0, 0, 0` | 正交 |
| 菱形十二面体 | `1, 1, 1` | 含 `±½` 的组合（`gmx -bt dodecahedron` 给 `α=β=60°, γ=90°` ⟹ `½, ½, 0`） | FCC |
| 截角八面体 | `1, 1, 1` | 归约前 `−⅓, −⅓, −⅓`（`α=β=γ=109.471°`）；归约后是等价的 `±½` 变体 | BCC |
| 一般三斜 | 其他 | 其他 | — |

**容差用相对量**（例如余弦上 `1e-4`），不是绝对的 `1e-6 nm`——理由见 §2（barostat）。

分类结果只是**给人看的标签**。任何下游判断都必须走下面这些**量**，不走标签字符串：

- `plane_spacings_nm = [V/|b×c|, V/|c×a|, V/|a×b|]` ← 这才是「盒尺度」
- `inscribed_radius_nm = 0.5 * min(plane_spacings)`
- `volume_nm3 = abs(det(B))`
- `is_openmm_reduced_form` + 具体违反了哪条

### 3.4 「处理」是什么

**处理 ≠ 改格子。处理 = 统一表示 + 报出正确的尺度。**

1. 任何格式的盒子 → 走 §3.1 的唯一入口拿到 **OpenMM reduced form**。
   这一步不改物理，只换表示。
2. 报出面间距 / 内切球半径，让下游有正确的尺度可用（§5 列的三个地方现在拿的
   是错的尺度）。

明确**不**做：不把斜盒转成正交盒（那要重新溶剂化，是建系工作）；
不自动降 cutoff 迁就小盒。

### 3.4b 处理流水线：一个漏斗，下游只认一种东西

「给他正常处理」的关键是**所有格式在最早的地方就收敛成同一个对象**，
之后整条生产链再也不需要知道用户提交的是什么格式、什么盒型。

```
用户提交的任意输入
  .gro (3 或 9 个数)  ┐
  .rst7 + .parm7      │
  PDB (CRYST1)        ├──►  read_box_from_input()      ← §3.1，只读盒子那几行
  mmCIF (_cell.*)     │        （长度+角度 一律走 computePeriodicBoxVectors）
  box_vectors.npy     │
  system.xml          ┘
                              ↓  (3,3) 盒向量，nm
                       normalize_periodic_box()         ← §3.2 + §3.4
                              ↓
              ┌─────────────────────────────────────────┐
              │  PeriodicBoxReport（唯一的下游契约）      │
              │   box_vectors_nm      OpenMM reduced form │
              │   shape               标签，只给人看       │
              │   plane_spacings_nm   ← 真正的「盒尺度」   │
              │   inscribed_radius_nm                    │
              │   volume_nm3          abs(det)           │
              │   is_openmm_reduced_form + 违反项明细     │
              │   source_format / source_declared_shape  │
              └─────────────────────────────────────────┘
                              ↓
   建 System / cutoff 判断 / Boresch r0 / 居中 / LRC / 诊断  ——全部只读这个对象
```

三条硬规矩：

1. **`box_vectors_nm` 是 OpenMM reduced form。** 非归约输入在漏斗里就被换成等价的
   归约表示（换表示，**不换格子**）。下游拿到的永远是 OpenMM 能直接吃的形式，
   于是「建 Context 时抛一句没上下文的异常」这个失败模式**从源头消失**。
2. **`shape` 只是标签。** 任何判断都走 `plane_spacings_nm` /
   `inscribed_radius_nm` / `is_openmm_reduced_form` 这些量。用标签当判据，就会在
   「被 barostat 压歪了一点点的十二面体」上分类失败然后走错分支。
3. **漏斗是只读的。** 不改坐标、不改 System、不改 `box_vectors.npy`。
   ⟹ `system_xml_sha256` / `box_vectors_sha256` / 各 resume 协议指纹**全部不变**，
   已完成的 GPU 窗口不会被迫重跑。报告进 provenance，但**不进任何协议指纹**
   （同 `ESS_GATE_PROTOCOL_VERSION` 的先例；也是 `code_sha256` 那个已经踩过两次
   的坑的教训）。

**接线点**：`runabfe.py` prepare 段，插在 `center_system_rigidly`（`:6172`）之后、
`resolve_membrane_protocol`（`:6184`）之前——此处的 `box_vectors` 已经是
`load_native_system`（`:6152`）从落盘缓存读回来的那一份，也就是生产真正会用的那份。
膜路径的 `validate_membrane_input`（`abfe_core.py:5108`）改成从同一份报告读
`is_orthogonal`，判据、报错文案、`box_is_rectangular` 返回键全部不变——
目的只是消掉第二份盒型判据。

### 3.5 「早期」是什么

**在建任何 System 之前，甚至在 `runabfe prepare` 之前。**

第一层（§3.1）刻意设计成**只读文件里盒子那几行**，不构建 OpenMM System、不解析
拓扑、不需要 `gmx_path`。于是可以给一个独立子命令：

```
runabfe.py inspect-box --gro complex.gro
runabfe.py inspect-box --rst7 complex.rst7 --parm7 complex.parm7
runabfe.py inspect-box --output <已有 run 目录>     # 读原生缓存
```

输出：盒型标签、盒向量（reduced form）、三个面间距、内切球半径、体积、
是否满足 `min(面间距) > 2*cutoff`（cutoff 从 System 实测；`inspect-box` 里若没有
System 就按 1.0 nm 报告并**注明这是假设值**）、Boresch `r0` 能不能塞进内切球。

这样用户建完系统的第一件事就能知道自己拿的是什么盒子——而不是等跑到一半、
或者等 OpenMM 在建 Context 时抛一句没有上下文的异常。

---

## 4. 仓库里已有什么（已核对，别重做）

好消息：三斜盒的**核心数学基本上是对的**。

| 东西 | 位置 | 状态 |
|---|---|---|
| minimum-image 位移 | `abfe_core.py:1415` | 🟢 **精确最近格点搜索**（不是分量 round），带奇异值下界证明。三斜正确，注释明写是「唯一实现」 |
| 盒体积 | `ibs_engine.py:4202`、`:7619`、`:9294` | 🟢 全部 `abs(det)` / `abs(a·(b×c))` |
| 面间距校验 | `abfe_core.py:6344` `_validate_minimum_image` | 🟢 **算法完全正确**（按 `V/|b×c|`，不是向量长度）。但全仓**只有 `abfe_core.py:9228` 一个调用点**（DEXP 路径） |
| 刚性居中 | `runabfe.py:2435` | 🟢 `0.5*Σ(box)` 求盒心 |
| 配体回盒 | `abfe_pipeline.py:4124` | 🟢 同上，只整体平移 |
| 恒压器 | `MonteCarloBarostat`（各向同性） | 🟢 三斜安全。`MonteCarloAnisotropicBarostat` 只出现在「检测已有 barostat」的名单里（`abfe_core.py:139`），**从不被创建** |
| 溶剂腿盒子 | `abfe_core.py:10405` | 🟢 自己算立方盒边并显式传 `boxSize=`，盒型不从输入进来 |
| 插件 G3 cell list | `CudaLocalManyBodyResidualKernels.cpp:1382-1395` | 🟢 **已按三斜面高**算网格，`<3` 格抛 `UNSUPPORTED_BOX`；`nCells=floor(h/r_list)` 保证 ±1 stencil 在三斜下仍覆盖完整 |
| 插件 CUDA minimum-image | 同文件 `:196-205` | 🟢 OpenMM 式 c→b→a 顺序归约（与 OpenMM 自己的非键 kernel 同算法） |
| 膜盒型门 | `abfe_core.py:5108-5113` | 🟢 已 fail-closed 拒绝截角八面体/十二面体 |

⟹ **要做的是「识别 + 统一表示」，不是重写几何。**

`_validate_minimum_image` 里那段面间距算法应该抽成内部 helper 给新代码共用，
**不要写第二份**。

---

## 5. 顺手能修的（都是纯离线、零 GPU）

### 5.1 三处拿错了「盒尺度」

| 位置 | 现在用的 | 应该用的 |
|---|---|---|
| `abfe_pipeline.py:3922-3923` | `0.4 * min(np.linalg.norm(box, axis=1))`（向量长度） | 面间距。十二面体的最短面间距明显小于最短向量长度 ⟹ 这个居中判据在斜盒上偏松 |
| `abfe_core.py:4599` → `:4614` | `np.diag(box)` → `0.5*min(lengths)`（仅 System 里找不到 cutoff 时的 fallback 上限） | 面间距 |
| `abfe_core.py:7799` | 只查 `r0 > 0` | 还要查 `r0 < inscribed_radius`。**生产 Boresch 力是开 PBC 的**（`ibs_engine.py:1802/1885/4986/5618/5732`、`abfe_pipeline.py:3996` 全 `use_pbc=True`；`abfe_core.py:7931` 据此调 `setUsesPeriodicBoundaryConditions`）⟹ `distance()` 走最小镜像 ⟹ `r0` 超过内切球半径时被约束的距离**物理上饱和**，而 `abfe_core.py:7790` 的解析参考积分照旧按无界谐振子算标准态修正 ⟹ **静默错值**。斜盒把内切球半径压得更小，更容易踩到 |

### 5.2 四处静默退化成正交盒

读不到 `traj.unitcell_vectors` 时回退 `np.diag(lengths)`，**丢掉所有夹角**：
`abfe_core.py:6134`（轨迹解缠）、`:6219`（脂质横向 MSD）、`:6290`（co-ion 逐帧距离）、
`:6451`（环境原子近邻选择）。

实践中 mdtraj 只要有 unitcell 就会给出 vectors，所以很少走到。但
`abfe_core.py:6122` 已经对 `unitcell_lengths` 缺失/非正 raise 了，vectors 缺失时
没有同等待遇——按本库规矩应改成 fail-closed，而不是猜一个正交盒。

### 5.3 两条零散项

- `abfe_pipeline.py:3272`：`volume_nm3 = float(np.linalg.det(box))` 没取 `abs`
  ⟹ 左手序盒向量给负体积 ⟹ `:3277` 的 `if volume_nm3 > 0` 把密度打成 `nan`。
  一个 `abs` 的事。
- `abfe_core.py:9110-9133` `GhostIonHandler._resolve_ghost_anchor` 按向量长度
  `np.mod` 摆锚点（正交假设）——**但它已退役**
  （`tests/test_charge_transfer_hamiltonian.py:509-513` 断言它不得成为
  charge-transfer 的实现）⟹ **不修，忽略**。

---

## 6. 一个已知的、本轮不碰的分歧

局部残差那条线上有**两套不同的 minimum-image 定义**：

| 实现 | 算法 | 正交盒 | 斜盒 |
|---|---|---|---|
| `abfe_core.py:1415`（主线唯一实现） | 精确最近格点搜索 | ✅ | ✅ 精确 |
| `local_residual/geometry.py:65-67` | Babai 分量 round | ✅ 逐位相同 | ⚠️ 近似 |
| `CudaLocalManyBodyResidualKernels.cpp:196-205` | c→b→a 顺序归约 | ✅ 逐位相同 | ⚠️ 近似，**与 Babai 也不一定一致** |
| `outer_lambda_neural_basis.py:1814/2850/5497/5768/5852/6001` | Babai 分量 round | ✅ | ⚠️ 同上 |

三者在正交盒下逐位相同，斜盒下可以互不相同。后果：EXP-025 G4 的
「A=B / C=D=E 三方等价」和 G1/G2/G3 parity 全是在正交盒上验的，
**结构上抓不到这个分歧**——不是那些验证做得差，是输入空间没覆盖斜盒。
`g1_math_core.h:141-143` 的注释自己就写着它是「independent re-implementation
-- not shared code with the Python side」，那份独立性在正交盒下没有代价。

> 🛑 这一条要动就会重开 EXP-025/026 的等价验证，需要 GPU。**必须单独立项，
> 不要和识别能力捆在一起做。**

---

## 7. 需要拍板的问题

1. **`inspect-box` 是独立子命令，还是 `prepare` 里顺便打印一段？**
   独立子命令能做到「建完系统立刻查」（真正的早期），但要多写一套参数解析。
2. **AMBER 那条路要不要一起接？** §3.1 里 rst7 / parm7 的读取是**新能力**——
   `load_amber_topology_for_openmm`（`abfe_core.py:3417`）目前没有任何生产调用点。
   如果 AMBER 输入还不在计划内，第一版可以只做 `.gro` + 原生缓存 + PDB `CRYST1`
   （PDB 几乎零成本，`app.PDBFile` 已经会读）。
3. **§5 那几条顺手修的要不要同批做？** 都是纯离线、不碰采样逻辑，但会动到
   `abfe_pipeline.py` 和 `abfe_core.py` 的七处。

---

## 8. 实施状态

⬜ 未开始。本文档是第 0 步。
