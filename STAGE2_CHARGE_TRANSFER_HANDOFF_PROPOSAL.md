# Stage2 charging→vanishing handoff：设计提案 + 已实现的工具函数

状态：**方向已获用户批准（2026-08-11）；核心工具函数
`abfe_core.bake_global_parameter_into_fixed_nonbonded_force` 已实现并有
12 项 CPU 契约测试全部通过（`tests/test_bake_global_parameter.py`）；
`build_ibs_dual_system`/`create_ligand_internal_force`/`abfe_pipeline.py`/
`runabfe.py` 仍未改动——真正的 charge-transfer Stage1→Stage2 生产接线
（把这个工具函数接进 `runabfe.py`/`abfe_pipeline.py` 的调用路径）还没有做**。

本文件经过一轮用户审阅修订：第一版对契约的两处描述（"多条 offset 先聚合"、
只用 `charge_at_lambda` 处理三个分量）在实现+测试阶段被证明需要修正，见 §1
末尾"实测纠正"。写这份文档前的每一条结论都先用真实 OpenMM Context 或诊断
脚本验证过，不是凭空写的。

## 0. 根因（已用诊断脚本钉死，不是"Group 2 用错电荷"）

C3-1 会话里第一次发现"vanishing λ_vdw=1 与 charging λ_coul=0 差
0.71 kJ/mol"时，最初把根因归到"Group 2（`create_ligand_internal_force`）
在配体电荷已被 charging 置零之后，重建配体内部 Coulomb 时用错了电荷来源"。
**这个归因是错的**，用诊断脚本逐 force-group 拆账证伪：

| 求值方式 | Group 0 | Group 2 | TOTAL(0,1,2,3,5) | 与 charging0 参照(40.154527)的差 |
|---|---|---|---|---|
| 不显式设 `lam_coul`（用默认值 1.0） | −1724.364098 | 1765.232902 | 40.868804 | **0.714277** |
| 显式设 `lam_coul=0.0` | −1725.077854 | 1765.232902 | 40.155048 | **0.000521** |

Group 2 的数值两次求值完全没变——`create_ligand_internal_force` 从
`NonbondedForce.getExceptionParameters()` 直接读物理 chargeProd，不看粒子
当前（可能已置零）的电荷，本来就是对的。**真正根因是 Group 0**：
`build_ibs_dual_system` 对输入 System 做 `XmlSerializer` 深拷贝
（`ibs_engine.py:3901`），charging 留在 System 上的 `lam_coul`
`GlobalParameter`（**默认值 1.0**）连同它的 offset 会被原样克隆过去；后续
求值只要忘了显式 `setParameter("lam_coul", 0.0)`，OpenMM 就用默认值 1.0，
把"配体 0 电荷、co-ion 满电"的 λ=0 端点悄悄翻成"配体满电、co-ion 中性"。
我自己在 C3-1 第一版 `compare_endpoint(...)` 调用里就忘了设——这正是要修的
那类"纪律型安全"问题。

## 1. 已实现的函数

```python
abfe_core.bake_global_parameter_into_fixed_nonbonded_force(
    system: openmm.System,
    parameter_name: str,
    lambda_value: float,
) -> openmm.System
```

职责单一、与 charge-transfer 无关（纯 OpenMM 机制工具）。实际行为（与代码
逐条对应，`abfe_core.py`，紧邻 `create_ligand_internal_force` 之前）：

1. 深拷贝输入 System。
2. 扫描**整个 System**（不只是 NonbondedForce）找出所有声明了
   `parameter_name` 这个 GlobalParameter 的 Force；若命中的不是恰好一个
   `NonbondedForce`（零个、多个，或者被其它类型的 Force 引用），fail
   closed——不猜语义，不假装已经处理干净。
3. 断言 `lambda_value` 是精确的 `0.0`/`1.0`（C3 的容差体系只在端点上有
   意义）。
4. 只对挂在 `parameter_name` 下的 `ParticleParameterOffset`/
   `ExceptionParameterOffset` 做处理，charge/sigma/epsilon 各自用同一个
   通用公式 `base + lambda_value * scale` 固化；挂在**其它** GlobalParameter
   下的 offset，以及其它 GlobalParameter 本身，原样保留、原样重新声明到
   新 Force 上。
5. 新建的 `NonbondedForce` 完整复制原 force 的非 particle/exception 配置
   （name、force group、reciprocal-space force group、nonbonded method、
   cutoff、switching 函数+距离、dispersion correction、reaction-field
   dielectric、Ewald tolerance、PME/LJPME 参数、exceptions 的 PBC 设置、
   include-direct-space）——用 `getattr` 逐个探测，当前 OpenMM 版本缺哪个
   属性就跳过并打印警告，不假装复制了。
6. System 里除目标 `NonbondedForce` 之外的所有其它 Force（co-ion
   flat-bottom restraint、barostat 等）原样保留，完全不碰。
7. 结构性核验：烘焙后再扫一次整个 System，确认 `parameter_name` 已经彻底
   不存在——不是"应该没有了"，是真的查了一遍。

**实测纠正（相对最初写的契约，两处改了）：**

- **不聚合多条 offset，改成检测到就 fail closed。** 最初的契约说
  "同一粒子/exception 上如果有多条挂在 `parameter_name` 下的 offset，先把
  scale 累加再烘焙一次"，理由是 OpenMM 文档里
  `parameter = base + Σ(global_i × scale_i)` 那个公式。**用真实 Context
  验证后发现这个理解是错的**：那个 Σ 说的是"同一个粒子上不同
  GlobalParameter 各自的 offset 相加"，不是"同一个 (parameter, particle)
  重复挂多条也相加"。实测：对同一个 (parameter, particle) 追加两条 offset
  （scale=0.3 和 0.2），`getNumParticleParameterOffsets()` 确实报出两条，
  但 Context 求值时只认**最后一条**（对应 0.2，不是 0.5）；exception 同理
  （0.01+0.02 两条追加，结果对应 0.02，不是 0.03）。这是没有文档、容易被
  误用的 OpenMM 行为。真实生产代码（`configure_charge_transfer_decharging`
  等）从不对同一个粒子重复调用 `addParticleParameterOffset`，所以选择不去
  复现"取最后一条"这个隐藏规则——遇到真的重复就直接报错，不猜语义。
- **不借用 `charge_at_lambda`。** 通用公式内联实现（`base + lambda_value *
  scale`，对 charge/sigma/epsilon 三个分量分别应用），不复用那个电荷专用
  命名的函数——语义上更贴切，也避免给一个通用工具引入一个专用函数的依赖。
- **SWIG 类型不稳定**（实现过程中发现，不是契约层面但值得记录）：
  `getParticleParameterOffset`/`getExceptionParameterOffset` 对"恰好是某个
  值"的 scale 字段，有时返回裸 `float`，有时返回 `Quantity`，类型不稳定。
  函数内部统一用 `unit.is_quantity()` 探测后转成裸 float 做运算，只在写回
  System 时才重新套上明确单位，不依赖返回值本身的类型。

CPU 契约测试（`tests/test_bake_global_parameter.py`，12 项全过）覆盖：
烘焙结果与显式设 Context 参数逐位相同；结构性删除 GlobalParameter（Context
再设这个参数名会报错）；不相关 GlobalParameter/offset 原样保留；重复
offset fail closed（particle 和 exception 各一条）；完整保留
NonbondedForce 配置（cutoff/switching/method/dispersion/reaction-field/
Ewald/force group/name/exceptions-PBC 逐项断言）；参数被非 NonbondedForce
引用时 fail closed；参数不存在时 fail closed；多个 NonbondedForce 共享同一
参数名时 fail closed；sigma/epsilon offset 正确烘焙（不只是 charge）；非
端点 λ 拒绝。

调用方式（charge-transfer 路线的 Stage1→Stage2，**尚未接入生产代码**）：

```python
charging0 = configure_pme_ligand_charge_offsets(raw_system_copy, ligand_indices,
                lambda_name="lam_coul", co_alchemical_ion_spec=spec, ...)
vanishing_input = bake_global_parameter_into_fixed_nonbonded_force(
    charging0, "lam_coul", 0.0)
new_sys, wrapper = build_ibs_dual_system(vanishing_input, ...)
```

`build_ibs_dual_system`/`create_ligand_internal_force` 完全不用改：喂给它们
的 `vanishing_input` 已经是"配体 0 电荷、co-ion 满电、配体内部 exception
带物理 chargeProd、没有任何活的 `lam_coul`"的静态 System。

## 2. 逐项对照用户要求的清单

- **ligand-environment 使用 `q_ligand=0`**：烘焙后配体粒子的静态电荷就是
  0（`base=0, scale=q_i` 在 λ=0 处固化为 0），`build_ibs_dual_system` 自己
  的 Step 3 会再把它和 LJ 一起清零用于 Group 0——两层清零叠加没有副作用。
- **co-ion 使用 full charge**：烘焙后 co-ion 静态电荷是 `base=share,
  scale=-share` 在 λ=0 处固化为 `share`（满电），`build_ibs_dual_system`
  把它当普通环境粒子对待，正确参与 Group 0 的 PME。
- **Group 2 保留原始 ligand internal 参数**：烘焙不改配体内部 exception 的
  chargeProd/sigma/epsilon（charging 冻结时写入的物理值，与 `lam_coul`
  无关），诊断已确认 Group 2 从这些 exception 原样读出，逐位不变。
- **exceptions/1–4、PME、LRC 的处理**：配体内部 1-2/1-3/1-4 由
  `create_ligand_internal_force` 已有机制消费，baking 不触碰；单端炼金
  exception（若存在）固化到 λ=0 值，行为与现状一致；PME 倒空间照常处理
  满电 co-ion；LRC 系数计算只吃 σ/ε，不受 baking 影响，D 端点"λ_vdw=0
  时 LRC 严格为 0"已用真实函数调用验证过，不受影响。
- **charging `λ=0` → vanishing `λ=1` seam 的严格等价关系**（2026-08-11
  用真实 Context 在 Reference 与 CUDA `Precision=double`+
  `DeterministicForces=true` 上重新验证，不是只接受"大数相减"的解释）：

  | 几何 | 平台 | 相对差 | 绝对差 (kJ/mol) | 结论 |
  |---|---|---|---|---|
  | 原诊断用的紧凑合成 fixture（原子间距 <0.2nm，非真实分子构象） | Reference | 1.298e-05 | 0.000521 | 略超 1e-5 门 |
  | 同上 | CUDA double+deterministic | 1.288e-05 | 0.000517 | 略超 1e-5 门 |
  | 改用真实键长/键角的伸展锌链几何（0.153nm 键长、~111.5° 键角、anti 二面角） | Reference | 1.206e-09 | 0.000521 | 门内 4 个量级 |
  | 同上 | CUDA double+deterministic | 1.196e-09 | 0.000517 | 门内 4 个量级 |

  **关键观察，如实记录、不过度解释**：绝对差在两种几何下几乎不变
  （≈0.00052 kJ/mol，Reference 与 CUDA-double 之间的微小差异在浮点噪声
  量级），说明它不是"配体原子挤在一起、LJ 排斥核把两侧都推到 ±1700
  kJ/mol 再相减"这种大数相减误差被几何放大——那个解释是最初诊断时的猜测，
  被这次重新验证证伪了：如果是大数相减，换成伸展几何后 Group 0/Group 2 的
  绝对量级会大幅下降，绝对差也应该跟着大幅下降，但它没有。真实情况是：
  存在一个**很小、与几何基本无关、Reference/CUDA-double 两个独立实现互相
  吻合**的绝对残差（~0.0005 kJ/mol），**具体来源尚未定位**——本次没有继续
  往下查是哪一项（怀疑候选：co-ion restraint 在两侧的取值方式细节、
  WCA/LRC 某个次级项、或某个未被两侧完全对称处理的 exception），只确认了
  它的量级足够小、且不随几何/精度变化。**对生产判定的实际影响**：真实分子
  的键长键角远比紧凑合成 fixture 正常，相对差门槛在真实几何上有 4 个量级
  的安全余量；不建议现在就去追这 0.0005 kJ/mol 残差的具体来源，但也不该
  假装它不存在——留作已知、量级极小、暂不阻塞的观察项，写入
  `memtodolist.md`。
- **protocol/version、fingerprint、cache invalidation 和旧结果兼容策略**：
  - `bake_global_parameter_into_fixed_nonbonded_force` 是纯函数式工具，
    不读写任何缓存/落盘文件，本身不需要协议版本。
  - 需要新增的是："charge-transfer 配体的 Stage 2 System 现在包含一次
    baking 步骤"这件事要不要进入某个已有协议版本——建议**新增专属常量**
    `CHARGE_TRANSFER_VANISHING_HANDOFF_PROTOCOL_VERSION = 1`（放在
    `abfe_core.py`，与 `CHARGE_TRANSFER_SOLVENT_LEG_IMPLEMENTED` 这类既有
    "能力标记"常量放在一起），只在 `charge_treatment ==
    co_alchemical_charge_transfer` 且启用了 vanishing 阶段时才写入相关
    manifest/cache key。
  - **旧结果兼容性是空话题**：C1/C2 从未跑过 charge-transfer 配体的
    vanishing 阶段（只有 11 个 λ_coul 态的 charging），这个组合在磁盘上
    没有任何既有产物——不存在误判旧缓存的风险。
  - `resolve_charge_treatment()` 里 `solvent_leg_builder_implemented`/
    `closes_thermodynamic_cycle` 这类既有布尔字段需要新增一个对应字段
    （例如 `vanishing_stage_implemented`），实现后从 `False`→`True`。
- **对 C4 的影响**：C4 准入清单里"Stage 2 中 co-ion 保持 fully charged"
  在当前代码状态下**无法验证**（没有一条真实路径能把带电配体送进
  Stage 2）；把这个函数**接入生产调用链**（本文档尚未做这一步）后，这条
  要求变得可构造、可验证。但工具函数本身不构成 C4 的完整准入：C4 还需要
  真实带电膜复合物、reserved dummy 建系、两腿 resume/cache 隔离都要在这条
  新的 vanishing 路径上重新走一遍。
- **独立 reference 验证办法**：
  1. 工具函数自身的 CPU 契约测试已完成（见 §1）。
  2. 仍未做：把 `production_vanishing_fixed_hamiltonian_systems` 的实际
     调用路径从"要求外部保证净中性的 raw system"换成"raw system → charging
     配置 → baking"，用带电 fixture 重新验证 seam 相对差和 D 的严格零门
     ——本次只用一个独立诊断脚本验证了原理（§0、上表），没有把它并入
     `tests/test_charge_transfer_real_endpoints.py` 的正式测试。
  3. 真实数据：C1/C2 没有真实的 charge-transfer + vanishing 轨迹，仍然
     只能在合成 fixture 上做契约测试；真实数据验证需要新跑一段真实的带电
     配体 vanishing 阶段 GPU 采样，不在本提案范围内。

## 3. 明确拒绝的替代方案

- **给 `build_ibs_dual_system` 加一个 `ligand_internal_charge_override`
  参数**：不需要——`create_ligand_internal_force` 已经从 exception 表
  正确读物理值，加这个参数只是给一个已经工作的机制包一层不必要的接口。
- **调用方每次求值都记得手动设 `lam_coul=0`，不做代码改动**：这正是导致
  0.714 kJ/mol 那次错误的做法——"记得设参数"是纪律，不是结构性保证。
- **在 `build_ibs_dual_system` 内部自动检测并清零陌生 GlobalParameter**：
  太隐式——不知道"清零"对某个陌生参数是否是正确语义，容易误伤该保留的
  参数。职责应该在调用方，不该塞进这个通用 builder。
- **同一 (parameter, target) 上多条 offset 时按 Σscale 聚合**：§1"实测
  纠正"已经说明——OpenMM 实际不按这个语义处理重复 offset，聚合是在复现一个
  没有文档、容易误用的隐藏行为，改成 fail closed 更安全。

## 4. 尚未做、留给下一步的事

- **把这个函数接入真正的生产调用链**（`runabfe.py`/`abfe_pipeline.py` 里
  charge-transfer 的 Stage1→Stage2 交接处）——本文档和已实现的函数只是
  这一步的前置工具，实际接线还没做。
- `CHARGE_TRANSFER_VANISHING_HANDOFF_PROTOCOL_VERSION` 具体挂在哪个
  manifest/cache-key 上、由谁在 resume 时核对——需要看 Stage 2 cache 落盘
  的具体位置，尚未深入。
- 把带电 fixture 的 seam/D 测试正式并入
  `tests/test_charge_transfer_real_endpoints.py`（目前只有独立诊断脚本）。
- 追查 §2 表格里那个 ~0.0005 kJ/mol 的小绝对残差具体来源（不阻塞，但也
  不该忘记）。
- 是否同时支持 co-annihilation 路线复用同一个 baking 工具——函数本身与
  charge_treatment 无关，天然适用；但 co-annihilation 是实验对照路线
  （MEM-00a-2 已降级），优先级留给用户判断。
