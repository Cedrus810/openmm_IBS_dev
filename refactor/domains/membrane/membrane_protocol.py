

# ============================================================================
# 膜体系协议：system_type + 膜恒压器（docs/status/memtodolist.md §3.1 / §3.2；MEM-00i）
#
# 这一节只负责"声明与校验"，不做任何采样决策。设计约束逐条对应清单：
#   §3.1 不根据残基名自动猜 system_type，用户必须显式声明；脂质残基检测只做交叉
#        检查，不作唯一判据；membrane 但无脂质 → fail closed；soluble 但检测到大量
#        脂质 → 警告并要求确认。
#   §3.2 预平衡用 MonteCarloMembraneBarostat（XY 等比例、Z 独立、默认表面张力 0）；
#        溶剂腿继续用普通 MonteCarloBarostat；检测任意已有 barostat 禁止重复添加；
#        已有不兼容 barostat → fail closed；barostat 类型/压力/表面张力/XY-Z 模式/
#        频率进入预平衡 fingerprint，改任一项使旧 checkpoint 失效。
#   §7.7 膜功能默认关闭：不声明 system_type 时行为与改动前逐位一致——这就是
#        barostat_fingerprint_payload() 对 legacy soluble 协议返回 None 的原因。
# ============================================================================

MEMBRANE_BAROSTAT_PROTOCOL_VERSION = 1

ENVIRONMENT_TYPE_SOLUBLE = "soluble"
ENVIRONMENT_TYPE_MEMBRANE = "membrane"
ENVIRONMENT_TYPES = (ENVIRONMENT_TYPE_SOLUBLE, ENVIRONMENT_TYPE_MEMBRANE)

# OpenMM 的三种 barostat 都是彼此独立的 Force 子类——MonteCarloMembraneBarostat
# **不是** MonteCarloBarostat 的子类。所以原来 abfe_pipeline.py 里
# `isinstance(f, openmm.MonteCarloBarostat)` 这一个判断检测不到膜/各向异性
# barostat：输入 System 若已带膜 barostat，旧代码会在它之上再叠一个各向同性的，
# 两个 barostat 同时生效 → 集合定义错误且不会报错。这里改用类名集合检测。
BAROSTAT_FORCE_CLASS_NAMES = (
    "MonteCarloBarostat",
    "MonteCarloAnisotropicBarostat",
    "MonteCarloMembraneBarostat",
)

# 仅用于交叉检查（§3.1），**绝不**用来推断 system_type。
#
# 刻意拆成两套，因为两个方向的误判后果完全不对称：
#
#   - `membrane` 但找不到脂质 → fail closed。这里用**宽**集合（含 Amber Lipid21
#     的模块化短残基名），误认成脂质只会放过一个用户已经显式声明为膜的运行，
#     无害；漏认才会挡住合法的膜运行。
#   - `soluble` 却检测到大量脂质 → 拦下来要求确认。这里只用**窄**集合（无歧义的
#     全名），因为误判会挡住一个完全合法的可溶体系运行。`PC`/`PE`/`PS`/`OL`/`ST`
#     这类 2–3 字母 token 在别的力场/配体命名里撞车的概率不低，不能拿它挡人。
#
# 实测当前生产体系 `solv_ions.gro` 的残基名只有 SOL / 20 种氨基酸 / ASH / CL /
# NA / MOL，与两套集合都无交集——本节新增的检查对现有可溶路径零影响。
LIPID_RESIDUE_NAMES_UNAMBIGUOUS = frozenset(
    {
        # 磷脂全分子残基名（CHARMM36 / Slipids 风格）
        "POPC", "POPE", "POPS", "POPG", "POPA", "POPI",
        "DOPC", "DOPE", "DOPS", "DOPG", "DOPA",
        "DPPC", "DPPE", "DPPS", "DPPG",
        "DMPC", "DMPE", "DMPG",
        "DSPC", "DLPC", "DPPI", "SAPI", "PSM", "SSM",
        # 胆固醇
        "CHL1", "CHOL", "CLR",
    }
)

# Amber Lipid21/Lipid17 把一个脂质拆成"头基 + 甘油 + 两条尾链"多个残基。
LIPID_RESIDUE_NAMES_AMBER_MODULAR = frozenset(
    {
        "PC", "PE", "PS", "PGR", "PA", "PH-",
        "OL", "ST", "MY", "LAL", "AR", "DHA",
    }
)

KNOWN_LIPID_RESIDUE_NAMES = LIPID_RESIDUE_NAMES_UNAMBIGUOUS | LIPID_RESIDUE_NAMES_AMBER_MODULAR

# soluble 却检测到多少个**无歧义**脂质残基就算"大量"（§3.1 最后一条）。取 8：
# 单个脂质分子被误命名不至于触发，任何真实双层（最小 slab 也有几十个）必定触发。
SOLUBLE_LIPID_RESIDUE_WARN_THRESHOLD = 8

MEMBRANE_XY_MODES = ("isotropic", "anisotropic")
MEMBRANE_Z_MODES = ("free", "fixed", "constant_volume")
MEMBRANE_NORMAL_AXES = ("x", "y", "z")

DEFAULT_MEMBRANE_PROTOCOL: Dict[str, Any] = {
    "normal_axis": "z",
    "surface_tension_bar_nm": 0.0,
    "xy_mode": "isotropic",
    "z_mode": "free",
    "barostat_frequency": 25,
}

# 改动前唯一存在的 barostat 协议：各向同性、频率 25（abfe_pipeline.py:1382）。
# fingerprint 对它保持沉默，以保证 §7.7 的"不声明 system_type 时逐位一致"。
LEGACY_SOLUBLE_BAROSTAT_FREQUENCY = 25


def _openmm_membrane_xy_mode(xy_mode: str):
    mapping = {
        "isotropic": openmm.MonteCarloMembraneBarostat.XYIsotropic,
        "anisotropic": openmm.MonteCarloMembraneBarostat.XYAnisotropic,
    }
    return mapping[xy_mode]


def _openmm_membrane_z_mode(z_mode: str):
    mapping = {
        "free": openmm.MonteCarloMembraneBarostat.ZFree,
        "fixed": openmm.MonteCarloMembraneBarostat.ZFixed,
        "constant_volume": openmm.MonteCarloMembraneBarostat.ConstantVolume,
    }
    return mapping[z_mode]


# ⚠️ 命名撞车警告，读到这里的人请务必分清两个**互不相干**的轴：
#
#   1. 本节的 soluble / membrane —— 环境类型，决定用哪种 barostat。
#      配置与 provenance 里的键名按 memtodolist §3.1/§10 定为 `system_type`。
#   2. 仓库里早就存在的 `system_type="complex"` / `"solvent"`
#      （`ABFEPipeline.run_full_pipeline` 等 20+ 处）—— **腿身份**，
#      决定加不加 Boresch、走复合物还是溶剂盒。
#
# 两者同名但含义完全不同。为避免在代码里混淆，本节内部一律用
# `environment_type` 这个标识符，只在**序列化**（fingerprint / provenance /
# 配置键）时才叫 `system_type`。任何函数都不要用 `system_type` 当形参名。
_LEG_IDENTITY_VALUES = ("complex", "solvent")


def resolve_environment_type(value: Optional[str]) -> str:
    """把配置键 `system_type` 规范化为环境类型；未声明即 soluble（§7.7 默认关闭）。

    不接受 None 之外的任何"聪明"回落：拼错的值必须报错，而不是静默当 soluble 跑。
    """
    if value is None:
        return ENVIRONMENT_TYPE_SOLUBLE
    normalized = str(value).strip().lower()
    if normalized == "":
        return ENVIRONMENT_TYPE_SOLUBLE
    if normalized in _LEG_IDENTITY_VALUES:
        raise ValueError(
            f"环境类型收到 {value!r}，但这是**腿身份**的取值。"
            "本仓库里 `system_type` 被两个不同的轴共用："
            "`run_full_pipeline(system_type='complex'|'solvent')` 指腿身份，"
            f"而环境类型只接受 {list(ENVIRONMENT_TYPES)}。"
            "请检查是不是把腿身份传进了膜协议解析。"
        )
    if normalized not in ENVIRONMENT_TYPES:
        raise ValueError(
            f"system_type={value!r} 不是合法的环境类型；允许 {list(ENVIRONMENT_TYPES)}。"
            "不会静默回落到 soluble——膜体系被误判为可溶体系会用错恒压器集合。"
        )
    return normalized


def count_lipid_residues(
    topology,
    names: Optional[frozenset] = None,
) -> Dict[str, int]:
    """按残基名统计疑似脂质残基数量，仅用于交叉检查（§3.1）。

    `names` 默认用宽集合 `KNOWN_LIPID_RESIDUE_NAMES`；传
    `LIPID_RESIDUE_NAMES_UNAMBIGUOUS` 可只统计无歧义全名。
    """
    counts: Dict[str, int] = {}
    if topology is None:
        return counts
    allowed = KNOWN_LIPID_RESIDUE_NAMES if names is None else names
    for residue in topology.residues():
        name = str(residue.name).strip().upper()
        if name in allowed:
            counts[name] = counts.get(name, 0) + 1
    return counts


def resolve_membrane_protocol(
    environment_type: Optional[str],
    membrane_config: Optional[Dict[str, Any]] = None,
    topology=None,
    confirm_soluble_with_lipids: bool = False,
) -> Dict[str, Any]:
    """校验环境类型与 membrane.* 组合，返回可直接进 fingerprint/provenance 的协议。

    `environment_type` 来自配置键 `system_type`（soluble/membrane），
    **不是**腿身份 complex/solvent——见上方命名撞车警告。

    在创建任何 Context 之前调用（§6.1）。返回的 dict 是**唯一**的下游真相来源，
    调用方不应再自己读原始配置字段。
    """
    resolved_type = resolve_environment_type(environment_type)
    lipid_counts = count_lipid_residues(topology)
    n_lipid_residues = int(sum(lipid_counts.values()))
    # 拦人的方向只认无歧义全名，见 LIPID_RESIDUE_NAMES_UNAMBIGUOUS 处的说明。
    unambiguous_counts = count_lipid_residues(
        topology, names=LIPID_RESIDUE_NAMES_UNAMBIGUOUS
    )
    n_unambiguous = int(sum(unambiguous_counts.values()))

    if resolved_type == ENVIRONMENT_TYPE_SOLUBLE:
        if membrane_config:
            raise ValueError(
                "system_type=soluble 却提供了 membrane.* 配置："
                f"{sorted(membrane_config)}。这两者组合含义不明，拒绝猜测；"
                "要跑膜体系请显式声明 system_type=membrane。"
            )
        if topology is not None and n_unambiguous >= SOLUBLE_LIPID_RESIDUE_WARN_THRESHOLD:
            message = (
                f"system_type=soluble 但拓扑里检测到 {n_unambiguous} 个脂质残基 "
                f"{dict(sorted(unambiguous_counts.items()))}。各向同性恒压器会把膜面积与厚度"
                "绑死，APL 会跑掉（memtodolist §3.1 / MEM-00i）。"
            )
            if not confirm_soluble_with_lipids:
                raise ValueError(
                    message
                    + " 若确认这不是双层膜（例如只是几个游离脂质配体），"
                    "请显式传 confirm_soluble_with_lipids=True 留下记录。"
                )
            logger.warning("[WARN] %s 已由 confirm_soluble_with_lipids=True 显式确认。", message)
        return {
            "protocol_version": MEMBRANE_BAROSTAT_PROTOCOL_VERSION,
            "system_type": ENVIRONMENT_TYPE_SOLUBLE,
            "barostat_class": "MonteCarloBarostat",
            "barostat_frequency": LEGACY_SOLUBLE_BAROSTAT_FREQUENCY,
            "membrane": None,
            "lipid_residue_counts": dict(sorted(lipid_counts.items())),
            "lipid_residue_total": n_lipid_residues,
            "lipid_residue_total_unambiguous": n_unambiguous,
            "soluble_with_lipids_confirmed": bool(confirm_soluble_with_lipids),
        }

    # ---- system_type == membrane ----
    if topology is not None and n_lipid_residues == 0:
        raise ValueError(
            "system_type=membrane 但拓扑里找不到任何已知脂质残基名"
            f"（已知集合 {len(KNOWN_LIPID_RESIDUE_NAMES)} 项）。"
            "按 memtodolist §3.1 这里 fail closed：要么输入不是膜体系，"
            "要么脂质残基名不在已知集合里——后者请先把残基名加进 "
            "KNOWN_LIPID_RESIDUE_NAMES 并说明力场来源，不要绕过本检查。"
        )

    protocol = dict(DEFAULT_MEMBRANE_PROTOCOL)
    unknown = sorted(set(membrane_config or {}) - set(DEFAULT_MEMBRANE_PROTOCOL))
    if unknown:
        raise ValueError(
            f"membrane.* 出现未知字段 {unknown}；允许 "
            f"{sorted(DEFAULT_MEMBRANE_PROTOCOL)}。拒绝静默忽略拼错的协议字段。"
        )
    protocol.update(membrane_config or {})

    normal_axis = str(protocol["normal_axis"]).strip().lower()
    if normal_axis not in MEMBRANE_NORMAL_AXES:
        raise ValueError(
            f"membrane.normal_axis={protocol['normal_axis']!r} 非法；允许 {list(MEMBRANE_NORMAL_AXES)}。"
        )
    if normal_axis != "z":
        raise ValueError(
            f"membrane.normal_axis={normal_axis!r}：OpenMM 的 MonteCarloMembraneBarostat "
            "把膜法向硬编码为 z（它只区分 XY 平面与 Z 轴，没有换轴选项）。"
            "请在建系时把膜法向对齐 z，不要在这里换轴——那会让 XY 等比例缩放"
            "作用在错误的平面上，且不会报错。"
        )

    xy_mode = str(protocol["xy_mode"]).strip().lower()
    if xy_mode not in MEMBRANE_XY_MODES:
        raise ValueError(
            f"membrane.xy_mode={protocol['xy_mode']!r} 非法；允许 {list(MEMBRANE_XY_MODES)}。"
        )
    z_mode = str(protocol["z_mode"]).strip().lower()
    if z_mode not in MEMBRANE_Z_MODES:
        raise ValueError(
            f"membrane.z_mode={protocol['z_mode']!r} 非法；允许 {list(MEMBRANE_Z_MODES)}。"
        )

    try:
        surface_tension = float(protocol["surface_tension_bar_nm"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"membrane.surface_tension_bar_nm={protocol['surface_tension_bar_nm']!r} 不是数值"
        ) from exc
    if not math.isfinite(surface_tension):
        raise ValueError("membrane.surface_tension_bar_nm 必须有限")

    try:
        frequency = int(protocol["barostat_frequency"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"membrane.barostat_frequency={protocol['barostat_frequency']!r} 不是整数"
        ) from exc
    if frequency <= 0:
        raise ValueError("membrane.barostat_frequency 必须为正整数")

    return {
        "protocol_version": MEMBRANE_BAROSTAT_PROTOCOL_VERSION,
        "system_type": ENVIRONMENT_TYPE_MEMBRANE,
        "barostat_class": "MonteCarloMembraneBarostat",
        "barostat_frequency": frequency,
        "membrane": {
            "normal_axis": normal_axis,
            "surface_tension_bar_nm": surface_tension,
            "xy_mode": xy_mode,
            "z_mode": z_mode,
        },
        "lipid_residue_counts": dict(sorted(lipid_counts.items())),
        "lipid_residue_total": n_lipid_residues,
        "lipid_residue_total_unambiguous": n_unambiguous,
        "soluble_with_lipids_confirmed": False,
    }


def detect_barostats(system) -> List[Tuple[int, str]]:
    """列出 System 里所有 barostat 的 (force index, 类名)。

    覆盖各向同性/各向异性/膜三种（BAROSTAT_FORCE_CLASS_NAMES），按类名而非
    isinstance 判断——三者不共享基类，isinstance 单查一种必然漏检。
    """
    found: List[Tuple[int, str]] = []
    for index, force in enumerate(system.getForces()):
        class_name = type(force).__name__
        if class_name in BAROSTAT_FORCE_CLASS_NAMES:
            found.append((index, class_name))
    return found


def _barostat_actual_parameters(force, class_name: str) -> Dict[str, Any]:
    """Read the parameters that OpenMM will actually use for a barostat."""
    actual = {
        "pressure_bar": float(force.getDefaultPressure().value_in_unit(unit.bar)),
        "temperature_K": float(
            force.getDefaultTemperature().value_in_unit(unit.kelvin)
        ),
        "frequency": int(force.getFrequency()),
    }
    if class_name == "MonteCarloMembraneBarostat":
        actual.update(
            {
                "surface_tension_bar_nm": float(
                    force.getDefaultSurfaceTension().value_in_unit(
                        unit.bar * unit.nanometer
                    )
                ),
                "xy_mode": force.getXYMode(),
                "z_mode": force.getZMode(),
            }
        )
    return actual


def _barostat_parameters_match(
    actual: Dict[str, Any], expected: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    mismatches: List[str] = []
    for key in ("pressure_bar", "temperature_K"):
        if not math.isclose(
            float(actual[key]), float(expected[key]), rel_tol=1.0e-10, abs_tol=1.0e-8
        ):
            mismatches.append(f"{key}: actual={actual[key]!r}, expected={expected[key]!r}")
    if int(actual["frequency"]) != int(expected["frequency"]):
        mismatches.append(
            f"frequency: actual={actual['frequency']!r}, expected={expected['frequency']!r}"
        )
    for key in ("surface_tension_bar_nm",):
        if key in expected and not math.isclose(
            float(actual[key]), float(expected[key]), rel_tol=1.0e-10, abs_tol=1.0e-8
        ):
            mismatches.append(f"{key}: actual={actual[key]!r}, expected={expected[key]!r}")
    for key in ("xy_mode", "z_mode"):
        if key in expected and actual[key] != expected[key]:
            mismatches.append(f"{key}: actual={actual[key]!r}, expected={expected[key]!r}")
    return not mismatches, mismatches


def ensure_barostat_for_protocol(
    system,
    protocol: Dict[str, Any],
    temperature,
    pressure,
) -> Dict[str, Any]:
    """按协议保证 System 上恰好有一个正确类型的 barostat（§3.2）。

    三种结局，都不静默：
      - 已有正确类型的 barostat → 复用，不重复添加（`action="reused_existing"`）；
      - 没有 barostat → 按协议添加（`action="added"`）；
      - 已有不兼容 barostat，或有多个 → **fail closed**，绝不再叠一个。

    `temperature` / `pressure` 接受裸数值（分别按 K / bar 解释）或 openmm Quantity。
    """
    temperature_k = (
        temperature.value_in_unit(unit.kelvin)
        if hasattr(temperature, "value_in_unit")
        else float(temperature)
    )
    pressure_bar = (
        pressure.value_in_unit(unit.bar)
        if hasattr(pressure, "value_in_unit")
        else float(pressure)
    )

    expected_class = protocol["barostat_class"]
    expected_parameters: Dict[str, Any] = {
        "pressure_bar": pressure_bar,
        "temperature_K": temperature_k,
        "frequency": int(protocol["barostat_frequency"]),
    }
    if expected_class == "MonteCarloMembraneBarostat":
        membrane = protocol["membrane"]
        expected_parameters.update(
            {
                "surface_tension_bar_nm": float(membrane["surface_tension_bar_nm"]),
                "xy_mode": _openmm_membrane_xy_mode(membrane["xy_mode"]),
                "z_mode": _openmm_membrane_z_mode(membrane["z_mode"]),
            }
        )
    existing = detect_barostats(system)

    if len(existing) > 1:
        raise RuntimeError(
            f"输入 System 上已有 {len(existing)} 个 barostat "
            f"{[name for _, name in existing]}；多个 barostat 同时生效的集合定义不明，"
            "拒绝继续（memtodolist §3.2）。"
        )
    if existing:
        _, existing_class = existing[0]
        if existing_class != expected_class:
            raise RuntimeError(
                f"输入 System 已带 {existing_class}，但当前协议 "
                f"system_type={protocol['system_type']} 要求 {expected_class}。"
                "按 memtodolist §3.2 这里 fail closed，而不是再叠加一个——"
                "叠加会让两个 barostat 同时做体积移动，集合定义错误且不报错。"
            )
        force = system.getForce(int(existing[0][0]))
        actual_parameters = _barostat_actual_parameters(force, existing_class)
        matches, mismatches = _barostat_parameters_match(
            actual_parameters, expected_parameters
        )
        if not matches:
            raise RuntimeError(
                f"输入 System 已带同类 {existing_class}，但实际参数与当前协议不一致；"
                "拒绝把旧参数静默认证为新协议。差异：" + "; ".join(mismatches)
            )
        return {
            "action": "reused_existing",
            "barostat_class": existing_class,
            "force_index": int(existing[0][0]),
            "protocol": protocol,
            "actual_parameters": actual_parameters,
        }

    if expected_class == "MonteCarloBarostat":
        force = openmm.MonteCarloBarostat(
            pressure_bar * unit.bar,
            temperature_k * unit.kelvin,
            int(protocol["barostat_frequency"]),
        )
    elif expected_class == "MonteCarloMembraneBarostat":
        membrane = protocol["membrane"]
        force = openmm.MonteCarloMembraneBarostat(
            pressure_bar * unit.bar,
            float(membrane["surface_tension_bar_nm"]) * unit.bar * unit.nanometer,
            temperature_k * unit.kelvin,
            _openmm_membrane_xy_mode(membrane["xy_mode"]),
            _openmm_membrane_z_mode(membrane["z_mode"]),
            int(protocol["barostat_frequency"]),
        )
    else:  # pragma: no cover - resolve_membrane_protocol 只产出上面两种
        raise RuntimeError(f"未知 barostat_class={expected_class!r}")

    force_index = system.addForce(force)
    actual_parameters = _barostat_actual_parameters(force, expected_class)
    return {
        "action": "added",
        "barostat_class": expected_class,
        "force_index": int(force_index),
        "protocol": protocol,
        "actual_parameters": actual_parameters,
    }


def barostat_fingerprint_payload(protocol: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """预平衡 fingerprint 里的 barostat 身份；legacy soluble 协议返回 None。

    返回 None 的意义是"与改动前完全相同的协议"，调用方据此**不往 fingerprint
    payload 里加任何键**，从而保证 §7.7：不声明 system_type 的运行，其
    fingerprint 与本次改动前逐位一致，已有的生产预平衡 checkpoint 不会失效。

    反过来，只要 system_type 变成 membrane、或频率偏离 legacy 值，payload 就非
    None，旧 checkpoint 自动失效——这正是 §3.2 要求的"改任一项使旧 checkpoint 失效"。
    """
    if not protocol:
        return None
    if (
        protocol.get("system_type") == ENVIRONMENT_TYPE_SOLUBLE
        and protocol.get("barostat_class") == "MonteCarloBarostat"
        and int(protocol.get("barostat_frequency", LEGACY_SOLUBLE_BAROSTAT_FREQUENCY))
        == LEGACY_SOLUBLE_BAROSTAT_FREQUENCY
    ):
        return None
    payload = {
        "protocol_version": int(protocol["protocol_version"]),
        "system_type": protocol["system_type"],
        "barostat_class": protocol["barostat_class"],
        "barostat_frequency": int(protocol["barostat_frequency"]),
        "membrane": protocol.get("membrane"),
    }
    return payload
