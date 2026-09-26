# ---------------------------------------------------------------------------
# 命令行解析
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """构造主 parser 并返回，**不** 解析 argv。

    [CLI-01, 2026-09-02] 从 `parse_arguments()` 里拆出来的，函数体一字未动。
    拆的目的：`validate-config` 诊断需要拿到 argparse 自己的 dest/type/choices/
    default，才能在不复制一份参数表的前提下校验配置文件——复制一份必然和这里
    各改一半。`parse_arguments()` 保留原签名和原行为（见其下方）。
    """
    parser = argparse.ArgumentParser(
        description="ABFE 计算流程控制器 (v3.0 重构版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # 🔑 [2026-09-09] 必须关掉参数缩写。
        #
        # argparse 默认 `allow_abbrev=True`，会把 `--outp` 解析成 `--output`、
        # `--temp` 解析成 `--temperature` 并写进 `args`。而配置合并那一层用的是
        # `_flag_present()`，它拿**原始 argv 的精确 token** 去比对 —— 缩写形式匹配
        # 不上，于是那个已经解析成功的值被**静默丢弃**，运行改用 preset/配置文件
        # 里的旧值。
        # 净效果：`runabfe.py --outp ./run2 --temp 310 ...` 会在 300 K 下写进
        # `./output`，读错缓存，且全程没有任何警告。
        # 关掉之后 argparse 自己会对未知/歧义参数报错 —— fail closed。
        allow_abbrev=False,
        epilog="""示例:
  # 首次运行（自动创建缓存并运行）
  python runabfe.py --gro complex.gro --top complex.top --ligand MOL \\
      --output ./output --boresch --boresch-source auto --preset production

  # 续跑（自动检测缓存）
  python runabfe.py --output ./output --ligand MOL --resume

  # 使用配置文件
  python runabfe.py --config params.json --output ./output --ligand MOL

只读诊断子命令（不启动任何模拟）:
  python runabfe.py doctor [--output ./output] [--json]
      环境体检：解释器、依赖版本、OpenMM 平台、GPU、GROMACS、磁盘余量、原生插件。
  python runabfe.py validate-config --config abfe_config.json [--json]
      配置检查：未知键/拼写、类型与取值、输入路径、生效参数及其来源。
  python runabfe.py config-template [--preset production] [--all] [--out my.json]
      打印一份带说明和当前默认值的配置模板；填好后用 validate-config 自查。
  python runabfe.py self-test
      轻量内置单元测试与物理约定检查。
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    prep_parser = subparsers.add_parser("prepare", help="生成预处理文件 (Boresch/DEXP等)")
    prep_parser.add_argument("--gro", required=True)
    prep_parser.add_argument("--top", required=True)
    prep_parser.add_argument("--ligand", required=True)
    prep_parser.add_argument("--ligand-xml", default=None, help="配体力场 XML/FFXML，用于溶剂腿构建")
    prep_parser.add_argument("--gmx-path", default=None)
    prep_parser.add_argument("--output-dir", default="./prep_output")
    prep_parser.add_argument("--save-boresch", default=None, help="保存 Boresch 文件")
    prep_parser.add_argument("--temperature", type=float, default=300.0)
    prep_parser.add_argument("--platform", default="CUDA")
    prep_parser.add_argument("--n-steps", type=int, default=5_000_000)
    subparsers.add_parser("self-test", help="运行轻量内置单元测试/物理约定检查")
    refine_parser = subparsers.add_parser(
        "refine-lambda-path",
        help="用某个 stage 已经采集到的窗口能量数据，按累积|Δf|重新设计λ分布与窗口边界（不是手写数字）",
    )
    refine_parser.add_argument("--stage-dir", required=True, help="该 stage 的窗口能量文件目录，如 output/vanishing")
    refine_parser.add_argument("--preopt-file", required=True, help="要读取并覆盖写回的 preopt_dual_*.json 路径")
    refine_parser.add_argument("--stage-type", default="vdw", choices=["vdw", "coul"], help="窗口文件名里的阶段类型后缀")
    refine_parser.add_argument("--temperature", type=float, default=300.0)
    refine_parser.add_argument("--n-states", type=int, default=None, help="重新分布后的总态数；默认保持原状态数不变")
    refine_parser.add_argument("--max-window-span-kj", type=float, default=35.0, help="单窗口允许的累积|Δf|预算上限 (kJ/mol)")
    refine_parser.add_argument("--overlap", type=int, default=2, help="相邻窗口之间共享的状态点数")
    # 基本输入
    parser.add_argument("--gro", default=None, help="GROMACS 结构文件 (首次运行时必需)")
    parser.add_argument("--top", default=None, help="GROMACS 拓扑文件 (首次运行时必需)")
    parser.add_argument("--ligand", default=None, help="配体残基名称 (如 MOL)")
    parser.add_argument("--ligand-xml", default=None, help="配体力场 XML/FFXML，用于溶剂腿构建")
    parser.add_argument("--gmx-path", default=None, help="GROMACS 力场 include 目录")
    parser.add_argument("--output", default="./output", help="输出目录 (同时也是缓存目录)")
    parser.add_argument("--config", default=None, help="JSON 配置文件，可覆盖预设")

    # 运行模式
    parser.add_argument("--resume", action="store_true", help="从 Checkpoint 恢复运行")
    parser.add_argument("--reset", action="store_true", help="忽略所有缓存，强制重新开始")
    parser.add_argument(
        "--openmm-cache-only",
        action="store_true",
        help="迁移服务器专用：验证并复用 output 中现有 OpenMM XML/CIF/index/box 缓存，"
             "不解析已不可用的 GROMACS include 树；--reset 仍会重置后续采样",
    )

    # 策略选择
    parser.add_argument("--mode", default="ibs", choices=["ibs", "traditional"],
                        help="采样引擎: ibs (默认) 或 traditional-REMD")
    parser.add_argument(
        "--decoupling",
        default="dual_lambda",
        choices=["dual_lambda", "single_lambda", "2d_diagonal", "2d_geodesic"],
    )
    parser.add_argument("--potential", default="softcore", choices=["softcore", "dexp"])
    parser.add_argument("--dexp-params", default=None, help="DEXP 参数文件 (JSON)")
    # 🔑 [PLAN_openmm_8_6_remd_backend.md P0] REMD 采样后端选择。
    # auto：环境与协议都合格时用官方 ReplicaExchangeSampler，否则回退 legacy 并记录原因。
    # legacy：始终用本项目自带的相邻扫描 REMD（复现/回退用）。
    # openmm：显式要求官方后端，任一检查不通过就在烧 GPU 之前报错，**不静默降级**。
    # 当前官方适配器尚未实现，所以 auto 恒定解析为 legacy、openmm 恒定报错。
    parser.add_argument(
        "--remd-backend",
        default=None,
        choices=list(free_energy_engine.REMD_BACKEND_CHOICES),
        help=(
            "REMD 采样后端：auto（默认，自动判定并可回退）/ legacy（始终用旧引擎）/ "
            "openmm（显式要求 OpenMM >= 8.6 官方采样器，不可用则报错）"
        ),
    )
    parser.add_argument(
        "--element-coverage-model",
        dest="element_coverage_model",
        default=None,
        help=(
            "启动期判定：体系的元素能不能被这个参考 ML 势表示。给 .model 路径或文件名"
            "（可配 $ABFE_MACE_MODEL_DIR），z-table 直接从模型文件读。不覆盖就 fail-closed "
            "并列全缺失元素。**不加载模型跑力、不训练任何东西**，只是一道覆盖判定。"
            "推荐 MACE-omol-0-extra-large-1024（83 种元素，含 Na；MACE-OFF24 没有 Na）。"
        ),
    )
    parser.add_argument(
        "--outer-lambda-resource-manifest",
        dest="outer_lambda_resource_manifest",
        default=None,
        help=(
            "冻结 R1 资源 manifest 的路径。**仓库不再随包发任何冻结权重**"
            "（2026-09-17 移出 resources/，理由见 archive/resources/…/WHY_RETIRED.md）："
            "不给这个参数就没有可加载的冻结模型，配体不匹配会走 EXP-033 P1 闭式重训。"
            "换配体重训出来的产物用这个显式指过去——只是指路，不放宽任何身份门："
            "配体指纹、原子数、payload/weights sha、插件源码 sha 全部照常校验。"
        ),
    )
    parser.add_argument(
        "--outer-lambda-local-residual-ibs",
        dest="outer_lambda_local_residual_ibs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            f"启用 {OUTER_LAMBDA_RESIDUAL_FEATURE_NAME}；默认关闭。"
            "启用后仅接入 dual_lambda 的 Stage-2 VDW，并自动采用残差-free twin EM。"
        ),
    )
    # 仅影响 --decoupling=dual_lambda 的 Stage 1 (去电荷)；Stage 2 (去VDW)/
    # single_lambda/2d_diagonal/2d_geodesic 不受影响，接口保持原样。
    #   - "pme"（默认，原有接口/行为不变）：IBS-CustomNonbondedForce 对
    #     λ_coul≠0 会硬性拒绝（CustomNonbondedForce 不支持 PME，真实电荷放进
    #     去会被截断成 cutoff 库仑，物理不可信），所以去电荷阶段走的是
    #     NonbondedForce ParameterOffset + 传统 REMD/MBAR，保留完整 PME 长程
    #     静电。这是本仓库当前推荐路线（见 README「当前推荐路线」）。
    #   - "shadow_ibs"（实验性，尚未经独立物理验证，生产结果不要用它）：改走
    #     Shadow-Coulomb IBS——用 erfc(alpha*r)/r 短程"影子"库仑 CV（Ewald 实
    #     空间形式，可以合法塞进 CustomNonbondedForce）做真正的 IBS 多态偏置
    #     采样，再加一段独立的 Bridge 腿（几个窗口的传统 REMD）把体系从"真实
    #     PME 满电荷"搭桥到"Shadow 满电荷"端点，两段 ΔG 相加得到完整去电荷
    #     自由能。只支持电中性配体（带净电配体会在构建系统时直接报错，不会
    #     静默算错），且暂不支持 --parallel-stages。详见
    #     ibs_engine.py::run_shadow_bridge_leg /
    #     ibs_engine.py::IBSWindowManagerShadowCoul。
    parser.add_argument(
        "--decharge-method",
        default="pme",
        choices=["pme", "shadow_ibs"],
        help="dual_lambda 去电荷阶段的实现方式：pme(默认，生产路线) 或 "
             "shadow_ibs(实验性 Shadow-Coulomb IBS，尚未经物理验证)",
    )

    # 🔑 [RELEASE_READINESS R1] 独立重复的唯一 seed 入口。优先级
    # --seed > ABFE_RANDOM_SEED > config 里的 repeat_seed；三者都不给则保留历史
    # 随机流（不注入新 seed）。解析见 `_resolve_repeat_seed`。
    parser.add_argument("--seed", type=int, default=None,
                        help="独立重复的 repeat seed（正整数）。优先级高于 "
                             "ABFE_RANDOM_SEED 与配置文件的 repeat_seed；"
                             "不给则沿用历史默认随机流")

    # Boresch 相关
    parser.add_argument("--boresch", action=argparse.BooleanOptionalAction, default=None,
                        help="是否启用 Boresch 限制力；复合物腿默认启用，可用 --no-boresch 显式关闭")
    parser.add_argument("--boresch-source", default=None,
                        choices=["traditional", "orb_ml", "simple", "orb_simple", "auto", "fluctuation"],
                        help="Boresch 来源：traditional/orb_ml=加载外部参数文件；"
                             "simple/fluctuation=纯几何涨落估算(不需要MACE)；"
                             "orb_simple=ORB/MACE口袋力投影单候选估算(需要MACE-OFF模型)；"
                             "auto=ORB/MACE多候选枚举估算(需要MACE-OFF模型)")
    parser.add_argument("--boresch-anchors", default=None, help="传统 Boresch 锚点文件")
    parser.add_argument("--boresch-orb", default=None, help="Orb ML 预测 Boresch 文件")
    parser.add_argument("--boresch-batch", type=int, default=0, help="批量估算候选数量")
    parser.add_argument("--boresch-select", type=int, default=1, help="选择第 N 个候选")
    parser.add_argument("--skip-rebalance", action="store_true", help="跳过带 Boresch 的再平衡")
    parser.add_argument("--rebalance-steps", type=int, default=50000, help="再平衡步数")

    # 采样控制
    parser.add_argument("--preset", default="production", choices=PRESET_CONFIGS.keys())
    parser.add_argument("--n-steps-per-window", type=int, default=None)
    parser.add_argument("--steps-per-update", type=int, default=None)
    parser.add_argument("--n-states-per-stage", type=int, default=None, help="每阶段 λ 状态数（同时设置两阶段）")
    parser.add_argument("--stage1-n-states", type=int, default=None, help="去电荷阶段 λ 状态数，优先级高于 --n-states-per-stage")
    parser.add_argument("--stage2-n-states", type=int, default=None, help="去VDW阶段 Fisher 探针网格点数（不是最终生产窗口数），优先级高于 --n-states-per-stage")
    parser.add_argument("--stage2-final-n-states", type=int, default=None, help="去VDW阶段最终生产窗口数，独立于 --stage2-n-states；默认 23（走手工验证过的窗口分组）。传别的值会用新的通用分组算法，2026-08-27 添加，尚未在真机验证过——第一次用时请检查 window 0 的 ESS")
    parser.add_argument("--stage2-refine-extra-points", type=int, default=None, help="去VDW阶段 Fisher 探针在陡峭段自动加密时每段插入的额外探针点数；默认 4")
    parser.add_argument("--stage2-window-min-states", type=int, default=None, help="去VDW阶段每个窗口最少多少态（仅在 --stage2-final-n-states 设成非 23 时生效）；默认 4")
    parser.add_argument("--stage2-window-max-states", type=int, default=None, help="去VDW阶段每个窗口最多多少态（仅在 --stage2-final-n-states 设成非 23 时生效）；默认 6")
    parser.add_argument("--stage2-production-budget-steps", type=int, default=None,
                        help="[审计 #31] Stage-2 生产步数总上限（整个 stage 的 GPU 预算）。不传 = 保持配置/预设值；默认 null = **上限未知**，控制器不拦（未知 ≠ 0）。给它一个具体数字等于打开一个此前从未生效过的终止条件，请只在明知要限额时设。")
    parser.add_argument("--stage2-max-production-blocks-per-window", type=int, default=None,
                        help="[审计 #31] 每个采样单元（子窗/物理窗）最多补几块生产帧，`_frames_admission()` 的块数硬上限；默认 4，与 abfe_preoptimizer 的兜底一致。")
    parser.add_argument("--max-path-insertions", type=int, default=None,
                        help="[审计 #31] λ 插点的**终身**预算（跨 resume 从版本链数出，不是每次调用重置）；默认 3，与 abfe_preoptimizer / abfe_pipeline 的兜底一致。")
    parser.add_argument("--stage2-free-energy-densify-points", type=int, default=None, help="[路径协议 v22] 去VDW阶段自由能定向加密点数 k：基础布点用 (最终态数-k) 态，再按 pilot 实测 <dU/dλ> 在 |ΔF| 最大的边上贪心插 k 个点。**总态数不变，采样成本不变**，只是把节点从平坦中段挪到自由能陡峭段（通常是 λ≈1）。默认 0 = 关闭，布点与 v21 逐字节相同。4W53 实测：16 态下 14+2 使最大边 ΔF 从 13.6 降到 7.7 kJ/mol，δ_max 0.94→1.08")
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument(
        "--n-equil-steps",
        type=int,
        default=None,
        help="预平衡步数；覆盖配置文件中的 n_equil_steps",
    )
    parser.add_argument(
        "--solvent-ionic-strength-molar",
        type=float,
        default=None,
        help="溶剂腿 NaCl 浓度 (M)，默认 0.15；另加必要反离子保持中性",
    )
    # Accept explicit OpenMM device specifications such as CUDA:1/OpenCL:0.
    # The base platform and DeviceIndex are validated by _build_platform_props.
    parser.add_argument("--platform", default="CUDA")
    parser.add_argument(
        "--allow-untrusted-stage-results",
        action="store_true",
        help=(
            "把 stage 质量门（物理目标支撑度 / raw ESS 等 6 条）从"
            "「中止流程」降级为「标记结果为 results_untrusted 并继续」。"
            "证据一条不少地写进结果与日志，但带此标记的 ΔG **不得**作为可发布"
            "结果——它的唯一用途是把流程推过复合物腿、跑到溶剂腿（溶剂腿是唯一"
            "有独立参考真值 ΔG_LJ 能判断对错的地方；口径见 "
            "docs/reference_data/README.md —— 与生产逐窗口可比的是 m2n2 的 "
            "−6.581±0.256 kJ/mol，且为 no-LRC，别直接相减）。"
            "默认关闭 = fail-closed。"
        ),
    )

    parser.add_argument(
        "--run-unclosed-charge-transfer-diagnostics",
        action="store_true",
        help=(
            "**诊断专用执行模式**：解除「charge-transfer 未通过 carrier "
            "reservoir/C4-C5 closure 验证 ⟹ 启动即拒」这一条，只为拿诊断数值。"
            "不改变任何被算出来的数；逐腿 final_results.json 与 "
            "final_binding_results.json 都会带 must_not_report_delta_g_bind=true / "
            "production_qualified=false / closes_thermodynamic_cycle=false ⟹ "
            "放行出来的 ΔG_bind **不得**发布或进任何汇总。默认关闭 = fail-closed。"
            "⚠️ **只认命令行**：写进配置文件不生效（每次运行必须显式指定，"
            "防止它沉进生产默认配置）。"
        ),
    )

    # ---- 膜受体体系（memtodolist §3.1/§3.2，B1）。默认 soluble，不传等于关闭 ----
    # ⚠️ 这里的 --system-type 是**环境类型**（soluble/membrane），决定用哪种
    # barostat。它与 run_full_pipeline(system_type="complex"|"solvent") 那个
    # 同名的**腿身份**参数是两个不相干的轴，不要混。
    parser.add_argument(
        "--system-type",
        default=None,
        choices=["soluble", "membrane"],
        help="体系环境类型；membrane 启用 MonteCarloMembraneBarostat。默认 soluble（不改变任何现有行为）",
    )
    parser.add_argument(
        "--membrane-normal-axis",
        default=None,
        choices=["x", "y", "z"],
        help="膜法向轴；OpenMM 的膜 barostat 只支持 z，其它值 fail closed",
    )
    parser.add_argument(
        "--membrane-surface-tension-bar-nm",
        type=float,
        default=None,
        help="膜表面张力 (bar·nm)，默认 0.0",
    )
    parser.add_argument(
        "--membrane-xy-mode",
        default=None,
        choices=["isotropic", "anisotropic"],
        help="XY 平面缩放模式，默认 isotropic（XY 等比例）",
    )
    parser.add_argument(
        "--membrane-z-mode",
        default=None,
        choices=["free", "fixed", "constant_volume"],
        help="Z 轴模式，默认 free（Z 独立变化）",
    )
    parser.add_argument(
        "--membrane-barostat-frequency",
        type=int,
        default=None,
        help="膜 barostat 体积移动频率，默认 25",
    )
    parser.add_argument(
        "--membrane-quality-gate",
        default=None,
        choices=["enforce", "advisory"],
        help=(
            "§9 膜质量门模式。enforce（默认）门未过即阻断；"
            "advisory 照样计算并落盘报告、失败大声 warning 但不阻断"
            "（用于先把管路跑通；**不是生产资格**）"
        ),
    )
    parser.add_argument(
        "--confirm-soluble-with-lipids",
        action="store_true",
        help="确认 system_type=soluble 但拓扑含大量脂质残基（留记录，不静默放行）",
    )

    # ---- 净电荷处理协议（memtodolist §1.2，B2）----
    # 不传时按配体净电荷推导：中性 → neutral（当前生产路径），带电 → charge-transfer。
    parser.add_argument(
        "--charge-treatment",
        default=None,
        choices=[
            "neutral",
            "co_alchemical_charge_transfer",
            "rocklin_apbs_neutralizing_plasma",
            "co_annihilation_experimental",
        ],
        help=(
            "净电荷处理路线。co-ion 路线与 Rocklin/APBS 是二选一，同时用即重复修正、"
            "会 fail closed。co_annihilation_experimental 仅供方法对照，膜体系禁用"
        ),
    )
    parser.add_argument(
        "--charge-transfer-reservoir-correction",
        default=None,
        help=(
            "charge-transfer 双腿 carrier reservoir-release correction JSON 路径；"
            "必须包含 temperature_K、两腿收敛释放自由能及 C4/C5 扫描证据"
        ),
    )
    parser.add_argument(
        "--co-alchemical-ion",
        default=None,
        help=(
            "[deprecated] 旧版全局 co-ion spec；不参与实际运行。"
            "B5 会为 complex/solvent 各自冻结并验证身份。"
        ),
    )
    parser.add_argument(
        "--apbs-evidence",
        default=None,
        help="rocklin_apbs_neutralizing_plasma 路线的 APBS 证据 JSON（manifest/result/介电图/脂质电荷图）",
    )

    # ---- LJ/色散路线与力场族（memtodolist §1.1 / §1.3，B6）----
    parser.add_argument(
        "--dispersion-protocol",
        default=None,
        choices=[
            "legacy_uniform_density_lrc",
            "ff_native_isotropic_lrc",
            "ff_native_force_switch_no_lrc",
            "lj_pme",
            "membrane_inhomogeneous",
        ],
        help=(
            "LJ/色散路线。soluble 不传 = legacy_uniform_density_lrc（行为不变）；"
            "membrane 不传即 fail closed。lj_pme / membrane_inhomogeneous 是路线 B/C，未实现"
        ),
    )
    parser.add_argument(
        "--forcefield-family",
        default=None,
        choices=["amber", "charmm"],
        help="显式覆盖从 .top 自动识别的力场族；覆盖会留记录，不静默",
    )
    parser.add_argument(
        "--membrane-input-declaration",
        default=None,
        help=(
            "膜输入声明 JSON（§3.3）：build_tool / build_parameters / "
            "final_equilibration_job / source_structure_id / conformational_state / "
            "membrane_composition / binding_site_solvent_exposure / "
            "leaflet_assignment_basis，可选 n_atoms / n_upper / n_lower / n_water / ion_counts。"
            "system_type=membrane 时必填"
        ),
    )
    parser.add_argument(
        "--force-switch-deviation-evidence",
        default=None,
        help=(
            "charmm 分支放行所需的定量偏差论证 JSON（APL / 膜厚 / 单点能对照）。"
            "OpenMM 没有 force-switch，无此证据一律 fail closed"
        ),
    )

    # 高级选项
    parser.add_argument("--torsion-params", default=None)
    parser.add_argument("--enable-early-stop", action="store_true")
    parser.add_argument("--enable-gradual-warmup", action="store_true")
    parser.add_argument("--disable-warmup", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=500000)
    parser.add_argument(
        "--n-workers", type=int, default=None,
        help="[WARN] 当前无效：从未被 run_full_pipeline/_run_dual_lambda_stage 消费，"
        "设置非默认值只会在运行时打印一条忽略警告，不会产生任何并行。",
    )
    parser.add_argument("--analyze-only", action="store_true", help="仅分析已有 .npy")
    parser.add_argument("--parallel-stages", action="store_true", help="并行执行去电荷和去VDW阶段")
    parser.add_argument(
        "--only-complex-charging",
        action="store_true",
        help=(
            "隔离重跑复合物腿 Stage 1 PME charging；跳过 Stage 2 路径优化、"
            "VDW 采样和溶剂腿，并用现有 Stage 2/solvent 结果生成候选汇总"
        ),
    )
    parser.add_argument(
        "--only-ligand-in-water",
        action="store_true",
        help=(
            "[LR-04] 只跑「配体在水中」那条腿：跳过 Boresch 估算、rebalance 和整条"
            "复合物腿。给的是配体在水里的去耦自由能 ΔG_solvent，"
            "**不是结合能** —— 没有复合物腿就没有 ΔG_bind，主 final_results.json 不被覆盖。"
            "调试用，不是生产入口"
        ),
    )
    parser.add_argument(
        "--only-boresch-attachment",
        action="store_true",
        help=(
            "[P1-17] 只跑 Boresch attachment 腿 ΔG(A′→A)（限制从关到开），"
            "复用冻结的 stage1/stage2/solvent 生成闭合循环的候选汇总。"
            "stage1/stage2 本就是受约束系综里测的，补这一项不改变它们"
        ),
    )
    parser.add_argument(
        "--attachment-rerun-dir",
        default=None,
        help="attachment 腿的隔离输出目录（默认 <output>/attachment_rerun/<时间戳>）",
    )
    parser.add_argument(
        "--attachment-n-steps-per-state",
        type=int,
        default=None,
        help="attachment 腿每个 λ 态的生产步数（默认 250000）",
    )
    parser.add_argument(
        "--attachment-n-seeds",
        type=int,
        default=None,
        help=(
            "attachment 腿跑几条独立轨迹（默认 1）。>1 时误差棒改用**跨 seed SEM**，"
            "而不是 MBAR 渐近协方差——后者在本体系已证明低估约 7 倍"
        ),
    )
    parser.add_argument(
        "--attachment-lambdas",
        default=None,
        help="逗号分隔的 λ 表，必须严格升序且从 0 到 1（默认 0.0,0.1,0.35,1.0）",
    )
    parser.add_argument(
        "--charging-rerun-dir",
        default=None,
        help=(
            "charging-only 隔离输出目录；默认在 OUTPUT/charging_rerun/ 下创建"
            "带时间戳的新目录，现有非空目录会被拒绝"
        ),
    )
    parser.add_argument(
        "--charging-max-resident-contexts",
        type=int,
        default=None,
        help=(
            "charging REMD 允许同时驻留的 Context 数；默认使用安全策略（CUDA "
            "会回退 CPU）。仅在显存足够时显式设为 Stage 1 状态数"
        ),
    )
    parser.add_argument("--n-lambda", type=int, default=12, help="传统 REMD 模式的 λ 状态数")
    parser.add_argument(
        "--apbs-correction-kj-mol",
        type=float,
        default=None,
        help="外部 APBS 长程/连续介质修正，单位 kJ/mol；最终 ΔG_bind 会加上该项",
    )
    parser.add_argument(
        "--apbs-correction-note",
        default=None,
        help="APBS 修正来源说明，例如网格、介电常数、输入文件或结果文件",
    )

    return parser


def parse_arguments():
    return build_parser().parse_args()


