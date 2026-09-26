# ---------------------------------------------------------------------------
# 工具函数：GROMACS 力场路径探测
# ---------------------------------------------------------------------------
def _gmx_include_dir_is_populated(path: str) -> bool:
    """这个目录看起来是不是**真的** GROMACS 力场 include 目录。

    只用来区分"安装前缀"和"力场目录"（见 `find_gmx_include_dir` 第 1 步）。
    判据取 GROMACS 自己 `share/gromacs/top` 下必然存在的东西：任意 `*.ff`
    力场目录，或 `residuetypes.dat`/`forcefield.itp` 这类顶层文件。刻意不做
    更强的校验——这里的作用是"别把前缀误当 include 目录"，不是替 GROMACS
    验证力场完整性。
    """
    try:
        if not os.path.isdir(path):
            return False
        for name in ("residuetypes.dat", "forcefield.itp", "atommass.dat"):
            if os.path.isfile(os.path.join(path, name)):
                return True
        for entry in os.listdir(path):
            if entry.endswith(".ff") and os.path.isdir(os.path.join(path, entry)):
                return True
    except OSError:
        return False
    return False


def find_gmx_include_dir(user_path: Optional[str] = None) -> Optional[str]:
    """智能查找 GROMACS 力场 include 目录。

    查找顺序（先到先得，全部失败才返回 None）：
    1. `user_path`——显式传入的 `--gmx-path`/`config.gmx_path`，用户指定优先于任何猜测。
    2. `$GMXLIB`——gmx 官方约定的力场搜索路径env var，值本身就是力场目录（不拼 "top"）。
    3. `$GMXDATA`——gmx 安装数据前缀，力场在其 `top/` 子目录下。
    4. `shutil.which("gmx")` 反推出的安装目录下的 `share/gromacs/top`。
    5. 几个通用发行版安装路径（`/usr/local/gromacs`、`/usr/share`、`/opt/gromacs`）。

    ## 关于曾经存在的两条个人路径

    2026-08-24 从第 5 步删掉了 `/home/ruigengji/gmx25.1/...` 和
    `/home/ruigengji/gmx26.1_avx2/...`。当时的顾虑是：沙盒机器不等于真正跑生产
    GPU 计算的节点，如果生产节点一直靠这两条硬编码路径落地，删掉会改变
    include_dir 解析结果，进而改变 GROMACS 解析出的力场文件和
    `system_xml_sha256`/`topology_sha256`——这些哈希参与协议指纹比较，
    跟 code_sha256 那次事故是同一类"不确定会不会影响真实节点结果就别冒险改"的风险。

    **这个顾虑仍然成立，但结论已经反过来：发布物里不能带任何人的家目录路径。**
    2026-08-31 发布整理确认代码中已无个人路径（此前 docstring 一度写成"已恢复、
    不再删除"，与代码不符，已在本次更正）。生产节点要指定力场目录，用
    `--gmx-path` / `config.gmx_path` / `GMXLIB` / `GMXDATA` 四条显式入口中的任意一条;
    第 1 步现在还会自动把安装前缀补成 `share/gromacs/top`，所以配置里写前缀也能跑。

    迁移提醒：如果某个生产节点此前确实是靠那两条个人路径隐式命中的，
    它现在会走到最后的 warning 分支并返回 None。这是**有意的 fail-loud**——
    好过继续依赖一条不在仓库控制范围内的机器本地路径。
    """
    if user_path and os.path.exists(user_path):
        # 🔑 [2026-08-31] `gmx_path` 允许写 GROMACS **安装前缀**，不必一路写到
        # `share/gromacs/top`。
        #
        # 这里原来是 `return user_path` 一行。于是 `abfe_config.json` 里那种
        # `"gmx_path": "/home/ruigengji/gmx26.0C"`（安装前缀，最自然的写法）会被
        # 原样当成力场 include 目录交给 `GromacsTopFile(includeDir=...)`，而力场
        # 文件其实在它的 `share/gromacs/top/` 下面 —— 结果是
        # `ValueError: Could not locate #include file`，用户只能把配置改成写全
        # `/home/ruigengji/gmx26.0C/share/gromacs/top` 才跑得动。第 3 步的
        # `$GMXDATA` 早就会去试 `top/` 子目录，只有显式 `--gmx-path` 这条没试。
        #
        # 非回归性：只有当 `user_path` 自己**不是**一个已填充的 include 目录、
        # 且其下能找到一个已填充的候选时才改用候选。凡是改动前能成功解析拓扑的
        # 输入（user_path 本身就是真 include 目录），这里返回值逐位不变；找不到
        # 候选时也原样返回 user_path。所以不可能改变任何已成功运行的
        # include_dir → 不会动 system_xml_sha256 / topology_sha256（见本 docstring
        # 里 2026-08-24 那段关于协议指纹的告警）。
        if _gmx_include_dir_is_populated(user_path):
            return user_path
        for _sub in ("share/gromacs/top", "top"):
            _candidate = os.path.join(user_path, _sub)
            if os.path.exists(_candidate) and _gmx_include_dir_is_populated(_candidate):
                log.info(
                    "find_gmx_include_dir: --gmx-path/%s 给的是安装前缀 %s，"
                    "已自动定位到力场 include 目录 %s",
                    "config.gmx_path", user_path, _candidate,
                )
                return _candidate
        return user_path

    gmxlib_path = os.environ.get("GMXLIB")
    if gmxlib_path and os.path.exists(gmxlib_path):
        return gmxlib_path

    gmxdata_path = os.environ.get("GMXDATA")
    if gmxdata_path and os.path.exists(gmxdata_path):
        top_path = os.path.join(gmxdata_path, "top")
        return top_path if os.path.exists(top_path) else gmxdata_path

    try:
        gmx_bin = shutil.which("gmx")
        if not gmx_bin:
            raise FileNotFoundError("gmx executable not found on PATH")
        gmx_root = os.path.abspath(os.path.join(os.path.dirname(gmx_bin), ".."))
        ff_path = os.path.join(gmx_root, "share/gromacs/top")
        if os.path.exists(ff_path):
            return ff_path
    except Exception:
        pass

    for path in [
        "/usr/local/gromacs/share/gromacs/top",
        "/usr/share/gromacs/top",
        "/opt/gromacs/share/gromacs/top",
    ]:
        if os.path.exists(path):
            return path

    log.warning(
        "find_gmx_include_dir: 找不到 GROMACS 力场 include 目录"
        "（--gmx-path 未给 / 未命中，GMXLIB、GMXDATA 未设置，PATH 里也没有 gmx）。"
        "如果拓扑里有需要它才能解析的 #include，请用 --gmx-path 显式指定，"
        "或设置 GMXLIB/GMXDATA 环境变量。"
    )
    return None


# ---------------------------------------------------------------------------
# 状态检测函数
# ---------------------------------------------------------------------------
def charge_transfer_admission_error(charge_protocol, *, diagnostics_mode=False):
    """charge-transfer 能不能开跑。返回 `None`（放行）或该抛的错误信息。

    🔑 [2026-09-18] carrier reservoir / C4-C5 closure 仍未验证 ⟹ **默认拒绝**：
    一个有限的两腿数会被当成闭合的 ABFE 结果。这道闸在建任何 Context 之前判死。

    `diagnostics_mode`（`--run-unclosed-charge-transfer-diagnostics`）只解除"不许开跑"
    这一条，**不改变任何被算出来的数**：`closes_thermodynamic_cycle` 仍是
    `False`，provenance 仍然带 `must_not_report_delta_g_bind=True`（那段逻辑
    只看 `closes_thermodynamic_cycle`，本开关碰不到它）⟹ 放行出来的 ΔG_bind
    依旧**不得发布**。它不是把 fail-closed 降级成放行，是把"启动即拒"换成
    "跑完但全程标记未验收"。
    """
    if (str(charge_protocol.get("charge_treatment"))
            != CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER):
        return None
    if charge_protocol.get("closes_thermodynamic_cycle") is True:
        return None
    if diagnostics_mode:
        return None
    return (
        "charge-transfer 当前未通过 carrier reservoir/C4-C5 closure 验证，"
        "禁止运行并报告 ΔG_bind。请完成双腿 restraint/盒缩放 closure 后再启用；"
        "本实现仅保留底层 Hamiltonian 作为实验诊断。"
        "（确知这一点、只要诊断数值 ⟹ 加 `--run-unclosed-charge-transfer-diagnostics`；"
        "产物仍带 must_not_report_delta_g_bind=True，不得发布。）"
    )


def _recompute_cached_builder_identity(
    *,
    xml_path: str,
    topology_path: str,
    ligand_indices_path: str,
    charge_treatment: Optional[str],
) -> Optional[Dict]:
    """Recompute the reserved-dummy identity from cache payloads.

    A manifest is metadata, not an authority.  For charge-transfer caches we
    deserialize the cached base System and Topology and derive the ligand net
    charge from the cached ligand indices, then call the single core helper.
    This makes a one-field edit to ``reserved_coion_builder_identity`` fail
    closed instead of being accepted merely because the field is a dictionary.
    """
    if str(charge_treatment) != CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER:
        return None
    try:
        with open(xml_path, "r", encoding="utf-8") as handle:
            cached_system = XmlSerializer.deserialize(handle.read())
        cached_cif = app.PDBxFile(topology_path)
        with open(ligand_indices_path, "r", encoding="utf-8") as handle:
            ligand_indices = [int(i) for i in json.load(handle)["ligand_indices"]]
        raw_q = _compute_ligand_net_charge(cached_system, ligand_indices)
        if not np.isfinite(float(raw_q)):
            raise ValueError(f"cached ligand net charge {raw_q!r} is not finite")
        q_l = int(round(float(raw_q)))
        if abs(float(raw_q) - q_l) > LIGAND_NET_CHARGE_INTEGER_TOLERANCE_E:
            raise ValueError(
                f"cached ligand net charge {raw_q:+.6f} e is not integral"
            )
        return co_alchemical_ion_builder_identity_payload(
            system=cached_system,
            topology=cached_cif.topology,
            charge_treatment=charge_treatment,
            ligand_net_charge_e=q_l,
        )
    except Exception as exc:
        log.warning("[WARN] reserved co-ion builder identity 重算失败，将拒绝复用缓存: %s", exc)
        return None


def system_cache_exists(
    output_dir: str,
    gro_file: Optional[str] = None,
    top_file: Optional[str] = None,
    ligand_resname: Optional[str] = None,
    gmx_include_dir: Optional[str] = None,
    charge_treatment: Optional[str] = None,
) -> bool:
    """Only accept a native cache bound to the complete current GROMACS input."""
    xml = os.path.join(output_dir, "system_native.xml")
    idx = os.path.join(output_dir, "ligand_indices.json")
    top = os.path.join(output_dir, "topology.cif")
    box = os.path.join(output_dir, "box_vectors.npy")
    manifest_path = os.path.join(output_dir, "system_cache_manifest.json")
    if not all(os.path.isfile(path) for path in (xml, idx, top, box, manifest_path)):
        return False
    try:
        expected = _main_cache_identity(
            gro_file, top_file, ligand_resname, gmx_include_dir
        )
        with open(manifest_path, encoding="utf-8") as handle:
            recorded = json.load(handle)
    except Exception as exc:
        log.warning("[WARN] 主 System 缓存身份无法校验 (%s)，将重建", exc)
        return False
    recorded_builder = recorded.get("reserved_coion_builder_identity")
    recorded_treatment = recorded.get("charge_treatment")
    if recorded_treatment is None and isinstance(recorded_builder, dict):
        recorded_treatment = recorded_builder.get("charge_treatment")
    # Old manifests did not carry a top-level route.  Infer it from the cached
    # base System; this preserves neutral-cache behavior while still refusing
    # to reuse an old charged cache that has no builder identity.
    if recorded_treatment is None:
        try:
            with open(xml, "r", encoding="utf-8") as handle:
                cached_system = XmlSerializer.deserialize(handle.read())
            with open(idx, "r", encoding="utf-8") as handle:
                cached_ligand_indices = [
                    int(i) for i in json.load(handle)["ligand_indices"]
                ]
            cached_q = _compute_ligand_net_charge(
                cached_system, cached_ligand_indices
            )
            recorded_treatment = (
                CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER
                if abs(float(cached_q)) > 1.0e-6
                else "neutral"
            )
        except Exception:
            # If route inference is impossible, retain the legacy comparison;
            # the later charged identity check will fail closed when requested.
            recorded_treatment = None
    route_matches = True
    if charge_treatment is not None and recorded_treatment is not None:
        route_matches = str(charge_treatment) == str(recorded_treatment)
    effective_treatment = charge_treatment or recorded_treatment
    matches = (
        recorded.get("protocol_version") == MAIN_SYSTEM_CACHE_PROTOCOL_VERSION
        and recorded.get("identity_sha256") == expected["identity_sha256"]
        and route_matches
    )
    if str(effective_treatment) == CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER:
        expected_builder = _recompute_cached_builder_identity(
            xml_path=xml,
            topology_path=top,
            ligand_indices_path=idx,
            charge_treatment=effective_treatment,
        )
        matches = matches and expected_builder is not None
        matches = matches and recorded_builder == expected_builder
    if not matches:
        log.warning("[WARN] 主 System 缓存与当前 GRO/TOP/include/配体/构建参数不匹配，将重建")
    return bool(matches)


def solvent_cache_exists(
    output_dir: str,
    ionic_strength_molar: float = DEFAULT_SOLVENT_IONIC_STRENGTH_MOLAR,
    expected_identity: Optional[Dict] = None,
    charge_treatment: Optional[str] = None,
    expected_builder_identity: Optional[Dict] = None,
) -> bool:
    """Only accept solvent caches built with the requested explicit salt.

    [B4，§4.5] 也校验 `charge_treatment` 是否与当前请求一致——同一个 output_dir
    如果换了净电荷路线（例如从 `neutral` 换成 `co_alchemical_charge_transfer`），
    旧盒子里没有 reserved co-ion，绝不能被静默复用。
    """
    xml = os.path.join(output_dir, "system_solvent.xml")
    idx = os.path.join(output_dir, "ligand_indices_solvent.json")
    top = os.path.join(output_dir, "topology_solvent.cif")
    manifest_path = os.path.join(output_dir, "solvent_cache_manifest.json")
    if not all(os.path.isfile(path) for path in (xml, idx, top, manifest_path)):
        return False
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except Exception as exc:
        log.warning("[WARN] 溶剂腿缓存 manifest 不可读 (%s)，将重建", exc)
        return False
    expected_strength = float(ionic_strength_molar)
    matches = (
        manifest.get("protocol_version") == SOLVENT_CACHE_PROTOCOL_VERSION
        and abs(float(manifest.get("ionic_strength_molar", -1.0)) - expected_strength)
        <= 1.0e-12
        and bool(manifest.get("neutralize"))
        and expected_identity is not None
        and manifest.get("identity_sha256") == expected_identity.get("identity_sha256")
        and manifest.get("charge_treatment") == charge_treatment
    )
    if str(charge_treatment) == CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER:
        if expected_builder_identity is None:
            expected_builder_identity = _recompute_cached_builder_identity(
                xml_path=xml,
                topology_path=top,
                ligand_indices_path=idx,
                charge_treatment=charge_treatment,
            )
            matches = matches and expected_builder_identity is not None
        else:
            # The caller-provided identity is authoritative for a freshly built
            # cache, but still require the persisted System/Topology to agree so
            # a hand-edited manifest cannot make a mutated cache resumable.
            recomputed = _recompute_cached_builder_identity(
                xml_path=xml,
                topology_path=top,
                ligand_indices_path=idx,
                charge_treatment=charge_treatment,
            )
            matches = matches and recomputed == expected_builder_identity
        matches = matches and manifest.get(
            "reserved_coion_builder_identity"
        ) == expected_builder_identity
    elif charge_treatment in (None, "neutral"):
        # Neutral legacy caches intentionally carry no co-ion identity.
        pass
    # `na_count`/`cl_count` include reserved alchemical dummies and are not
    # evidence that the requested physical salt was built.  New manifests
    # carry ordinary counts; charge-transfer caches missing either field are
    # rejected regardless of the requested strength rather than guessing from
    # the legacy totals.
    if str(charge_treatment) == CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER:
        ordinary_fields_present = (
            "ordinary_na_count" in manifest and "ordinary_cl_count" in manifest
        )
        matches = matches and ordinary_fields_present
        if expected_strength > 0.0 and ordinary_fields_present:
            matches = matches and int(manifest["ordinary_na_count"]) > 0
            matches = matches and int(manifest["ordinary_cl_count"]) > 0
    elif expected_strength > 0.0:
        if "ordinary_na_count" in manifest and "ordinary_cl_count" in manifest:
            matches = matches and int(manifest["ordinary_na_count"]) > 0
            matches = matches and int(manifest["ordinary_cl_count"]) > 0
        else:
            # Neutral legacy caches predate the split and remain compatible.
            matches = matches and int(manifest.get("na_count", 0)) > 0
            matches = matches and int(manifest.get("cl_count", 0)) > 0
    if not matches:
        log.warning(
            "[WARN] 已有溶剂腿缓存不是当前显式盐协议 "
            "(v%d, %.3f M NaCl)，将自动重建",
            SOLVENT_CACHE_PROTOCOL_VERSION,
            expected_strength,
        )
    return bool(matches)


# 严格校验结果按 (path, size, mtime_ns) 记忆化：equilibrium_is_done 在一次
# 运行里会被多处以同一份文件调用，而真实 DCD parser 要逐帧读到 EOF（膜体系
# 轨迹在 NFS 上是几十 GB 量级），没必要重复付这个 I/O。文件一旦变化
# （size/mtime），缓存自动失效。
_TRAJ_STRICT_VALIDITY_CACHE: Dict[Tuple[str, int, float], bool] = {}


def _checkpoint_probe_simulation(pipeline: "ABFEPipeline"):
    """为 checkpoint 校验建一个一次性目标 Simulation（P1-07）。

    OpenMM checkpoint 是 platform/version 相关的不透明二进制，只有"目标
    Simulation 真的 loadCheckpoint 成功"才算数。这里按 pipeline 将要使用的
    同一个 platform 建 Context——它能否加载，与下游 pre_equilibrate(resume)
    的行为一致：probe 加载失败的 checkpoint，真正的消费方同样加载不了。
    建不出来（platform 不可用等）时返回 None，调用方 fail closed。

    ## 为什么不缓存（2026-09-09）

    这里原本把建好的 probe `setattr` 到 pipeline 上当缓存，为的是同一个 pipeline
    被查两次（`runabfe` 里 3576 与 6931）时省一次构建。代价是：一整个 30710 原子的
    System + Context **从预平衡判定那一刻挂到整条腿跑完**，纯属白占。
    而它省下的只有几秒，在一条跑几小时的腿上没有意义。

    更坏的是失效方向：这块常驻占用不会报错，只会让后面 REMD 建 Context 时的
    "显存不够" 变成一条假线索（正是 `step_guard.guarded_context` 那段错误信息
    要提醒的"先看开跑前 used 是多少"）。

    现在用完即走：调用方拿到的 probe 在 `equilibrium_is_done()` 返回后就没有
    引用，随即回收。多出来的那次构建换掉一整条腿的常驻占用。
    """
    import openmm.app as _app

    try:
        # Match the consumer's System and Integrator, including the barostat
        # added by pre_equilibrate (checkpoint state depends on both).
        probe_system = XmlSerializer.clone(pipeline.system)
        ensure_barostat_for_protocol(
            probe_system, pipeline.barostat_protocol,
            temperature=pipeline.temperature, pressure=pipeline.pressure,
        )
        integrator = openmm.LangevinMiddleIntegrator(
            pipeline.temperature, 1.0 / unit.picosecond,
            PRE_EQUILIBRATION_TIMESTEP_PS * unit.picosecond,
        )
        platform_name, props = _build_platform_props(getattr(pipeline, "platform_name", None) or "CPU")
        platform = openmm.Platform.getPlatformByName(platform_name)
        probe = _app.Simulation(pipeline.topology, probe_system, integrator, platform, props)
    except Exception as exc:
        log.warning("[WARN] 无法为目标 platform 构建 checkpoint 校验 Simulation (%s)", exc)
        return None
    return probe


# 🔑 [2026-09-01] 本函数返回 False 只表示"不判定为已完成"，**不**表示"要从零重跑"。
# 下游 `pre_equilibrate(resume=True)` 仍会尝试 loadCheckpoint；命中时剩余步数可能是 0。
# 实测（4W53 resume_v3.log）：日志连报两次"将重新执行预平衡"，紧接着却是
# `♻️ 从 Checkpoint 恢复 | 已完成: 5000000 | 剩余: 0`，整段不到 7 分钟、一步没跑。
# 旧文案让人以为烧掉了 500 万步 GPU，是"消息说谎"——与本项目其它几处同类问题一样，
# 它不报错、不影响数值，只误导读日志的人去查一个不存在的性能问题。
_REENTER_NOTE = (
    "（注意：这里只是**不把它判定为已完成**；下游 pre_equilibrate(resume=True) 仍会"
    "尝试加载 checkpoint，命中则从断点续跑、剩余步数可能为 0，并不等于从零重跑）"
)


def equilibrium_is_done(
    output_dir: str,
    expected_fingerprint: Optional[str] = None,
    simulation=None,
) -> bool:
    """Only completed, identity-matching DCD/checkpoint pairs may be reused.

    A target Simulation is mandatory: missing/unavailable probes fail closed.
    DCDs must parse to EOF and checkpoints must load into the target Context.
    With expected_fingerprint, require matching metadata and a completed state
    with sufficient steps (or a documented convergence early stop).
    """
    traj = os.path.join(output_dir, "pre_equilibration.dcd")
    chk = os.path.join(output_dir, "checkpoints", "pre_equil.chk")
    # [P1-07] 弱判据（大小阈值/文件存在）只保留为快速短路：连弱判据都不过
    # 直接 False；过了弱判据还必须过下面的严格校验。
    if not (os.path.isfile(traj) and os.path.getsize(traj) > 10000 and os.path.isfile(chk)):
        return False
    try:
        traj_cache_key = (traj, os.path.getsize(traj), os.stat(traj).st_mtime_ns)
    except OSError:
        traj_cache_key = None
    if traj_cache_key is not None and traj_cache_key in _TRAJ_STRICT_VALIDITY_CACHE:
        traj_is_valid = _TRAJ_STRICT_VALIDITY_CACHE[traj_cache_key]
    else:
        traj_is_valid = _is_traj_valid(traj, min_frames=1)
        if traj_cache_key is not None:
            _TRAJ_STRICT_VALIDITY_CACHE[traj_cache_key] = traj_is_valid
    if not traj_is_valid:
        log.warning(
            "[WARN] %s 未通过真实 DCD parser 校验（截断/损坏/0 帧）。拒绝把它当成"
            "已完成的预平衡复用，将重新进入预平衡阶段" + _REENTER_NOTE + "。",
            traj,
        )
        return False
    if simulation is not None:
        if not _is_checkpoint_valid(chk, simulation=simulation):
            log.warning(
                "[WARN] %s 无法被目标 Simulation loadCheckpoint（损坏/不属于当前 "
                "System/Platform）。拒绝复用该预平衡，将重新进入预平衡阶段"
                + _REENTER_NOTE + "。",
                chk,
            )
            return False
    else:
        log.warning("[WARN] 缺少目标 Simulation，无法严格校验预平衡 checkpoint；拒绝判定完成")
        return False
    if expected_fingerprint is None:
        return True
    # ---- 指纹：只报告，不判定（2026-09-09）----
    #
    # 这三层以前都 `return False`。**它们从来没有保护过任何东西**：本函数返回
    # False 之后，下游 `pre_equilibrate(resume=True)` 加载的正是同一个被判定
    # 「不可信」的 checkpoint（见本文件上方 `_REENTER_NOTE` 记录的实测：连报两次
    # 「拒绝复用」两次，紧接着 `已完成: 5000000 | 剩余: 0`，一步没跑）。
    #   （此处刻意不引用那句旧文案原文 —— `test_equilibrium_rejection_does_not_claim_a_full_rerun`
    #    按本函数源码文本 grep 它，正确地禁止它出现在这里。）
    #
    #   * 配置真的换了 → 拦不住，下游照样把旧 checkpoint 读回来；
    #   * 配置没换、只是重建出的 System 序列化字节变了 → 误判，resume 被打断。
    #
    # 后者是实际发生的那一种：`system_xml_hash` 哈希的是当次在内存里重建的
    # System，任何只改字节不改物理的改动（例如排除表灌入顺序换成 `sorted()`，
    # 能量逐比特不变）都会让它翻脸。该字段已于同日从
    # `_pre_equilibration_fingerprint()` 的 payload 中移除。
    #
    # 判定"是否已完成"的权威只有下面那一个：`pipeline_state.json` 的
    # `equilibration.status == "completed"`（本函数自己的注释早就写着这句）。
    # 指纹保留为**信息性日志**，用来提示"你可能复用了别的配置的 --output 目录"。
    #
    # ⚠️ 本注释最初写成"不再阻断 resume"，那是错的 —— 当时还有下一层
    # `abfe_pipeline.pre_equilibrate()` 里的同名检查在 `resume = False`（丢 checkpoint、
    # 从零重跑）。由同仓另一会话复核指出，已于同日一并改为只告警。两侧现在一致。
    fp_file = os.path.join(output_dir, "pre_equilibration_fingerprint.json")
    if not os.path.isfile(fp_file):
        log.info(
            "[INFO] 缺少 pre_equilibration_fingerprint.json（可能来自更早的运行）；"
            "不影响判定，继续按 pipeline_state.json 的完成状态决定。"
        )
    else:
        try:
            with open(fp_file, encoding="utf-8") as f:
                recorded = json.load(f).get("fingerprint")
        except Exception as e:
            log.info("[INFO] 读取 pre_equilibration_fingerprint.json 失败 (%s)；不影响判定。", e)
        else:
            if recorded != expected_fingerprint:
                log.warning(
                    "[WARN] 预平衡指纹与当前 system/config 不同（换了 gro/top/ligand/温度/"
                    "目标步数，或复用了同一个 --output 目录）。**这不影响是否复用** —— "
                    "完成与否只看 pipeline_state.json 的 equilibration.status。"
                    "若你确实换了体系，请用 --reset 或换一个 --output。"
                )

    # ---- 真的跑完了吗（MEM-09，2026-08-02）----
    #
    # ⚠️ 到这里为止的判据全部是 **fail-open** 的：轨迹与 checkpoint 只要存在就算数，
    # 而 `pre_equilibration_fingerprint.json` 是 `pre_equilibrate()` 在**第一步之前**
    # 就写下的（`abfe_pipeline.py`，为的是让被中断的运行下次能认出自己的身份），
    # 它记的 `n_steps` 是**目标**步数，不是已完成步数。
    #
    # 于是一次被中断的运行（例如 100 ns 只跑到 40 ns）在下一次调用里会被判成
    # "已完成"，`pre_equilibrate()` 整段跳过 —— 连同它内部的 §9 膜质量门一起跳过，
    # 而 provenance 与 fingerprint 都写着 100 ns。这是"短平衡冒充长平衡"，
    # 且 `enforce` 模式根本没机会拦。
    #
    # 唯一可信的完成标记是 `pipeline_state.json` 里 equilibration 的
    # `status=completed` —— 它只在步进真正结束之后才写。
    state_file = os.path.join(output_dir, "checkpoints", "pipeline_state.json")
    if not os.path.isfile(state_file):
        log.warning(
            "[WARN] 预平衡轨迹/Checkpoint/指纹都在，但缺少 checkpoints/pipeline_state.json，"
            "无法确认预平衡**真的跑完了**（指纹是第一步之前就写的，只代表目标）。"
            "保守视为未完成。加 --resume 可从 Checkpoint 续跑，不会白烧已完成的部分。"
        )
        return False
    try:
        with open(state_file, encoding="utf-8") as f:
            equil_state = (json.load(f).get("stages") or {}).get("equilibration") or {}
    except Exception as e:
        log.warning("[WARN] 读取 pipeline_state.json 失败 (%s)，保守视为预平衡未完成", e)
        return False
    if equil_state.get("status") != "completed":
        log.warning(
            "[WARN] 上一次预平衡**没有跑完**（pipeline_state.json 里 equilibration.status=%r）。"
            "拒绝把它当成已完成的平衡复用 —— 那会让一段短平衡冒充目标时长，"
            "并连带跳过 §9 膜质量门。加 --resume 可从 Checkpoint 续跑。",
            equil_state.get("status"),
        )
        return False
    # 🔑 [2026-09-10] 读不到目标步数时**保守视为未完成**，不再当成 0。
    #
    # 原来这里 `except: recorded_target = 0`，而下面的门是
    # `if recorded_target and achieved < recorded_target:` —— 0 是 falsy，
    # 于是"fp 文件缺失/损坏"会把整条 MEM-09 门（短平衡冒充长平衡 + 连带跳过
    # §9 膜质量门）**完全短路掉**，一次只跑了 40 ns 的中断运行会被判成
    # "已完成 100 ns"。缺目标值恰恰是最该保守的情形，不是最该放行的。
    # 保守的代价只是 --resume 从 checkpoint 续跑，不会白烧已完成的部分。
    recorded_target = None
    try:
        with open(fp_file, encoding="utf-8") as handle:
            _recorded_n_steps = json.load(handle).get("n_steps")
        if _recorded_n_steps is not None:
            recorded_target = int(_recorded_n_steps)
    except Exception as exc:
        log.warning("[WARN] 读取预平衡目标步数失败 (%s)", exc)
    if recorded_target is None:
        log.warning(
            "[WARN] 读不到预平衡的**目标**步数（%s 缺失或损坏），无法判断上一次是"
            "跑满了还是中途被中断。保守视为未完成 —— 加 --resume 可从 Checkpoint "
            "续跑，不会白烧已完成的部分。",
            os.path.basename(fp_file),
        )
        return False
    achieved = int(equil_state.get("total_steps") or 0)
    # 🔑 [2026-09-03] 合法的**收敛早停**必须放行，否则会净亏。
    #
    # 这个门原本只比 `achieved < recorded_target`，防的是"短平衡冒充长平衡"
    # （见上面 §9 膜质量门被跳过的事故）。它一直成立，是因为
    # `_pre_equilibration_convergence_block_stop` 的早停在实测中**从未触发过**
    # （配体位姿漂移的周期镜像跳变 bug 把 stable_streak 永久清零）。镜像 bug
    # 修好之后早停真的会触发，于是一次**合法**早停（例如收敛判据认可地停在
    # 4,250,000 步，目标 5,000,000）在门看来跟一次被中断的运行完全一样，
    # 每次 --resume 都会重烧整段预平衡——早停省下的 75 万步，代价是之后每次
    # resume 白烧 425 万步。
    #
    # `pipeline_state.json` 里已经有区分两者所需的信息（`pre_equilibrate` 在
    # 两个写入点都落了 `convergence_stopped_early` / `convergence_stopped_at_step`），
    # 之前只是没人读。这里补上，门的原意不丢：**只有**收敛判据自己宣布的早停
    # 才放行，被中断的运行仍然照旧拒绝。
    #
    # ⚠️ 刻意**不**采用"早停时把指纹里的 n_steps 改写成实际达成步数"那种做法：
    # 那会让缓存键依赖执行历史而不是配置身份，正是
    # `seed_contract_identity()` 那条注释在防的形状。指纹保持只描述**目标**。
    if recorded_target and achieved < recorded_target:
        stopped_early = bool(equil_state.get("convergence_stopped_early"))
        stopped_at = equil_state.get("convergence_stopped_at_step")
        # 三重自证，任何一条不成立就按"被中断"处理：
        #   1. 收敛判据确实宣布了早停；
        #   2. 落盘的停止步与实际达成步数一致（防状态被拼接/改写）；
        #   3. 达成步数不低于收敛地板——早停在地板内根本不可能发生，
        #      低于地板说明这不是早停而是中断。
        legitimate_early_stop = (
            stopped_early
            and stopped_at is not None
            and int(stopped_at) == achieved
            and achieved >= int(PRE_EQUILIBRATION_CONVERGENCE_MIN_STEPS)
        )
        if legitimate_early_stop:
            log.info(
                "[OK] 预平衡在第 %d 步由 block 收敛判据提前停止（目标 %d 步，"
                "地板 %d 步）。这是合法完成，复用之，不重跑。",
                achieved, recorded_target,
                int(PRE_EQUILIBRATION_CONVERGENCE_MIN_STEPS),
            )
            return True
        log.warning(
            "[WARN] 已完成的预平衡只有 %d 步，而当前目标是 %d 步，且不是收敛判据"
            "认可的早停（convergence_stopped_early=%r, stopped_at=%r）。"
            "拒绝复用。注意这不是可续跑的情形——预平衡会从头重跑；"
            "若确认那段平衡本身可用，请检查 pipeline_state.json。",
            achieved, recorded_target, stopped_early, stopped_at,
        )
        return False
    return True


def boresch_params_ready(output_dir: str, source: str) -> bool:
    """检查 Boresch 参数文件是否已生成"""
    if source in ("auto", "simple", "orb_simple", "fluctuation"):
        return os.path.isfile(os.path.join(output_dir, f"boresch_{source}.json"))
    return False


def _cache_paths(output_dir: str, phase: str = "complex") -> Dict[str, str]:
    if phase == "solvent":
        return {
            "xml": os.path.join(output_dir, "system_solvent.xml"),
            "idx": os.path.join(output_dir, "ligand_indices_solvent.json"),
            "box": os.path.join(output_dir, "box_vectors_solvent.npy"),
            "top": os.path.join(output_dir, "topology_solvent.cif"),
            "runtime_dir": os.path.join(output_dir, "solvent_leg"),
        }
    return {
        "xml": os.path.join(output_dir, "system_native.xml"),
        "idx": os.path.join(output_dir, "ligand_indices.json"),
        "box": os.path.join(output_dir, "box_vectors.npy"),
        "top": os.path.join(output_dir, "topology.cif"),
        "runtime_dir": output_dir,
    }


def validate_openmm_cache_only(output_dir: str, phase: str = "complex") -> Dict:
    """Fail closed on a migrated native OpenMM cache without GROMACS inputs.

    This mode deliberately authenticates the serialized artifacts against the
    manifest that travelled with them.  It does not claim that the unavailable
    GROMACS dependency tree was revalidated on the new host; that limitation is
    recorded explicitly in the returned audit payload.
    """
    if phase not in {"complex", "solvent"}:
        raise ValueError(f"unsupported cache-only phase: {phase}")
    paths = _cache_paths(output_dir, phase=phase)
    manifest_path = os.path.join(
        output_dir,
        "system_cache_manifest.json" if phase == "complex" else "solvent_cache_manifest.json",
    )
    required = {
        "system_xml": paths["xml"],
        "topology": paths["top"],
        "ligand_indices": paths["idx"],
        "box_vectors": paths["box"],
        "manifest": manifest_path,
    }
    missing = [path for path in required.values() if not os.path.isfile(path)]
    if missing:
        raise RuntimeError(
            "--openmm-cache-only requires a complete migrated cache; missing: "
            f"{missing}"
        )
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest_identity = manifest.get("identity")
    manifest_identity_sha = manifest.get("identity_sha256")
    if isinstance(manifest_identity, dict) and isinstance(manifest_identity_sha, str):
        encoded_identity = json.dumps(
            manifest_identity, sort_keys=True, separators=(",", ":")
        ).encode()
        if hashlib.sha256(encoded_identity).hexdigest() != manifest_identity_sha:
            raise RuntimeError(
                f"--openmm-cache-only {phase} manifest identity is not self-consistent"
            )
    # 自产缓存文件（system_native.xml / topology.cif / ligand_indices.json /
    # box_vectors.npy）的 sha256 **不做身份判据**：pipeline 自己合法改写它们
    # （PBC-01 删假键、盒子更新、排除表书写顺序）就会让哈希变，缓存被判无效、
    # 白重跑几百万步。身份只认用户输入（identity_sha256 来自 gro/top/config）。
    # 内容正确性由下面的结构校验（原子数、shape、有限性）负责。

    with open(paths["xml"], encoding="utf-8") as handle:
        cached_system = XmlSerializer.deserialize(handle.read())
    cached_cif = app.PDBxFile(paths["top"])
    with open(paths["idx"], encoding="utf-8") as handle:
        ligand_indices = [int(value) for value in json.load(handle)["ligand_indices"]]
    positions_nm = np.asarray(
        cached_cif.positions.value_in_unit(unit.nanometer), dtype=np.float64
    )
    box_nm = np.asarray(np.load(paths["box"]), dtype=np.float64)
    n_particles = int(cached_system.getNumParticles())
    n_topology_atoms = int(cached_cif.topology.getNumAtoms())
    if (
        positions_nm.shape != (n_particles, 3)
        or n_topology_atoms != n_particles
        or not np.all(np.isfinite(positions_nm))
    ):
        raise RuntimeError(
            f"--openmm-cache-only {phase} System/topology/positions are not atomic"
        )
    if box_nm.shape != (3, 3) or not np.all(np.isfinite(box_nm)):
        raise RuntimeError(f"--openmm-cache-only {phase} box is not a finite 3x3 matrix")
    cif_box = cached_cif.topology.getPeriodicBoxVectors()
    if cif_box is not None:
        cif_box_nm = np.asarray(
            cif_box.value_in_unit(unit.nanometer), dtype=np.float64
        )
        if not np.allclose(cif_box_nm, box_nm, rtol=0.0, atol=1.0e-6):
            raise RuntimeError(
                f"--openmm-cache-only {phase} CIF cell and box_vectors.npy differ"
            )
    if (
        not ligand_indices
        or len(set(ligand_indices)) != len(ligand_indices)
        or min(ligand_indices) < 0
        or max(ligand_indices) >= n_particles
    ):
        raise RuntimeError(f"--openmm-cache-only {phase} ligand indices are invalid")
    audit = {
        "protocol": "openmm-migrated-cache-only-v1",
        "phase": phase,
        "output_dir": os.path.abspath(output_dir),
        "gromacs_dependency_tree_revalidated": False,
        "manifest_path": os.path.abspath(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "manifest_protocol_version": manifest.get("protocol_version"),
        "manifest_identity_sha256": manifest.get("identity_sha256"),
        "bonded_topology_reconstruction": "harmonic_bond_force_plus_constraints",
        "n_particles": n_particles,
        "n_ligand_atoms": len(ligand_indices),
    }
    encoded = json.dumps(audit, sort_keys=True, separators=(",", ":"))
    return {**audit, "audit_sha256": hashlib.sha256(encoded.encode()).hexdigest()}


def _get_residue_name_by_atom_index(topology: app.Topology, atom_index: int) -> str:
    atoms = list(topology.atoms())
    if atom_index < 0 or atom_index >= len(atoms):
        raise IndexError(f"原子索引越界: {atom_index}")
    return atoms[atom_index].residue.name


def resolve_ligand_ffxml(output_dir: str, ligand_resname: str, explicit_path: Optional[str] = None) -> Optional[str]:
    """定位配体力场 XML/FFXML，优先用户显式指定，其次尝试 output 下常见命名。"""
    candidates: List[Optional[str]] = [
        explicit_path,
        os.path.join(output_dir, "ligand.xml"),
        os.path.join(output_dir, "ligand.ffxml"),
        os.path.join(output_dir, f"{ligand_resname}.xml"),
        os.path.join(output_dir, f"{ligand_resname}.ffxml"),
        os.path.join(output_dir, "ffxml", "ligand.xml"),
        os.path.join(output_dir, "ffxml", "ligand.ffxml"),
        os.path.join(output_dir, "ffxml", f"{ligand_resname}.xml"),
        os.path.join(output_dir, "ffxml", f"{ligand_resname}.ffxml"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def _build_stable_ligand_atom_names(gmx_top: app.GromacsTopFile, ligand_resname: str) -> Dict[int, str]:
    """基于 .top 的原始原子顺序生成稳定且唯一的配体原子名映射。"""
    lig_atoms = [a for a in gmx_top.topology.atoms() if a.residue.name == ligand_resname]
    if not lig_atoms:
        raise ValueError(f"未在拓扑中找到配体残基: {ligand_resname}")

    raw_names = [atom.name.strip() or atom.element.symbol for atom in lig_atoms]
    if len(set(raw_names)) == len(raw_names):
        return {atom.index: raw for atom, raw in zip(lig_atoms, raw_names)}

    elem_counter: Dict[str, int] = {}
    stable_names: Dict[int, str] = {}
    for atom in lig_atoms:
        elem = atom.element.symbol if atom.element is not None else "X"
        elem_counter[elem] = elem_counter.get(elem, 0) + 1
        stable_names[atom.index] = f"{elem}{elem_counter[elem]}"
    return stable_names


def _resolve_mdtraj_topology_input(output_dir: str, gro_file: Optional[str], top_file: Optional[str]) -> str:
    """为 mdtraj.load 解析最稳妥的拓扑输入，优先缓存的 mmCIF。"""
    cif_path = os.path.join(output_dir, "topology.cif")
    if os.path.isfile(cif_path):
        return cif_path
    if gro_file and os.path.isfile(gro_file):
        return gro_file
    if top_file and os.path.isfile(top_file):
        return top_file
    raise FileNotFoundError(
        "无法为轨迹加载解析拓扑：未找到 topology.cif 缓存，也未提供有效的 --gro/--top。"
    )


RESERVED_COION_DERIVED_DIRNAME = "coion_reserved"
RESERVED_COION_REPORT_BASENAME = "coion_reserved.json"


def runtime_gromacs_top(output_dir, config_top):
    """与**运行时 System** 对得上的那份 `.top`。

    🔑🔑🔑 [2026-09-18 真机 `thrombin_ligand1/rep1`] **带电配体的复合物腿里，
    `config.top` 不再描述正在跑的体系。**

    净电荷 ≠ 0 时复合物腿会派生一份带 reserved co-ion dummy 的 `.gro`/`.top`
    （删掉 N 个水分子、在 `[molecules]` 末尾追加 N 个离子），随后
    `system` / `topology` / `ligand_indices` **全部**从派生件重建，原始输入只读。
    真机数字：原始 43761 原子 → 派生 43759（−3 水 +1 Na）。此后任何再拿
    `config.top` 去读拓扑的地方，原子数和配体索引都对不上 —— Boresch 锚点估算
    就是这么死的（`GROMACS 与轨迹拓扑原子数/配体索引不匹配`）。

    ⚠️ **不能直接改 `config.top`**：它是**缓存身份**的一部分
    （`system_cache_manifest.json` 的 `identity.gro/top.sha256` 记的就是用户给的
    那两份原始文件）。派生件是本次运行的产物，不是用户输入 —— 把自产产物写进
    身份是这个仓库反复复发的同一类 bug。所以身份留给原始件，**只有读拓扑的人**
    改读派生件。

    中性配体没有派生件 ⟹ 原样返回 `config_top`，行为逐字节不变。
    """
    report = os.path.join(output_dir, RESERVED_COION_DERIVED_DIRNAME,
                          RESERVED_COION_REPORT_BASENAME)
    if not os.path.isfile(report):
        return config_top
    try:
        with open(report, "r", encoding="utf-8") as fh:
            derived = json.load(fh).get("derived_top")
    except (OSError, ValueError, json.JSONDecodeError):
        derived = None
    if derived and os.path.isfile(derived):
        return derived
    # 报告在但派生件不在 = 产物被删/搬过。**不静默退回原始件**：那会拿一份
    # 对不上的拓扑继续跑，正是本次要修掉的失效模式。
    raise FileNotFoundError(
        f"本次运行派生过带 reserved co-ion 的 GROMACS 拓扑（见 {report}），"
        f"但 derived_top={derived!r} 不存在 ⟹ 无法取得与运行时 System 一致的拓扑。"
        "原始 --top 的原子数与之不同，拿它继续会静默算错，拒绝回退。"
    )


def _boresch_mdtraj_topology(pipeline, top_file=None, include_dir=None):
    """Build a trajectory topology with authoritative ligand bonds, without moving atoms."""
    import mdtraj as md

    target_atoms = list(pipeline.topology.atoms())
    ligand = set(int(i) for i in pipeline.ligand_indices)
    if not ligand or min(ligand) < 0 or max(ligand) >= len(target_atoms):
        raise ValueError("Boresch ligand indices 与拓扑不一致")
    if top_file:
        if not os.path.isfile(top_file):
            raise FileNotFoundError(f"Boresch GROMACS 拓扑不存在: {top_file}")
        source = load_gromacs_topology_for_openmm(
            top_file, includeDir=include_dir,
            periodicBoxVectors=pipeline.box_vectors,
        ).topology
        source_atoms = list(source.atoms())
        if len(source_atoms) > len(target_atoms) or max(ligand) >= len(source_atoms):
            raise ValueError("GROMACS 与轨迹拓扑原子数/配体索引不匹配")
        for a, b in zip(source_atoms, target_atoms):
            if (a.name, a.residue.name, a.element) != (b.name, b.residue.name, b.element):
                raise ValueError(f"GROMACS 与轨迹拓扑原子顺序不一致: index={a.index}")
        pairs = [(int(a.index), int(b.index)) for a, b in source.bonds()]
        source_name = "gromacs_topology"
    else:
        # Cache-only runs still have real force-field bonds/constraints in the
        # System. mmCIF's inferred ligand links are never authoritative.
        pairs = []
        for force in pipeline.system.getForces():
            if isinstance(force, openmm.HarmonicBondForce):
                pairs.extend(tuple(map(int, force.getBondParameters(i)[:2]))
                             for i in range(force.getNumBonds()))
        pairs.extend(tuple(map(int, pipeline.system.getConstraintParameters(i)[:2]))
                     for i in range(pipeline.system.getNumConstraints()))
        source_name = "system_bonds_and_constraints"
    real_ligand_pairs = {tuple(sorted((a, b))) for a, b in pairs if a in ligand and b in ligand}
    if not real_ligand_pairs:
        raise ValueError("Boresch 缺少真实配体键；拒绝距离猜键")
    # Build a copy and replace all ligand-internal bonds, including any mmCIF
    # guesses, with the force-field graph. Keep appended co-ions and atom order.
    fixed = app.Topology()
    copied = {}
    for chain in pipeline.topology.chains():
        new_chain = fixed.addChain(chain.id)
        for residue in chain.residues():
            new_residue = fixed.addResidue(residue.name, new_chain, residue.id, residue.insertionCode)
            for atom in residue.atoms():
                copied[atom.index] = fixed.addAtom(atom.name, atom.element, new_residue, atom.id)
    for a, b in pipeline.topology.bonds():
        if not (a.index in ligand and b.index in ligand):
            fixed.addBond(copied[a.index], copied[b.index])
    for a, b in sorted(real_ligand_pairs):
        fixed.addBond(copied[a], copied[b])
    result = md.Topology.from_openmm(fixed)
    log.info("Boresch 使用 %s 的 %d 条真实配体键", source_name, len(real_ligand_pairs))
    return result


def generate_ligand_xml_from_top(
    top_file: str,
    ligand_resname: str,
    output_dir: str,
    gmx_include_dir: Optional[str] = None,
) -> str:
    """从 GROMACS .top 中抽取配体参数，生成可被 OpenMM ForceField 加载的 ligand_only.xml。"""
    log.info("🔹 从 GROMACS 拓扑抽取配体参数: 生成 ligand_only.xml")
    # 走唯一入口（见 abfe_core.load_gromacs_topology_for_openmm）。
    top = load_gromacs_topology_for_openmm(top_file, includeDir=gmx_include_dir)
    lig_res = next((r for r in top.topology.residues() if r.name == ligand_resname), None)
    if lig_res is None:
        raise ValueError(f"未在拓扑中找到配体残基: {ligand_resname}")

    lig_idx = [a.index for a in lig_res.atoms()]
    stable_names = _build_stable_ligand_atom_names(top, ligand_resname)
    global_to_local = {g: l for l, g in enumerate(lig_idx)}
    atoms = list(top.topology.atoms())
    lig_set = set(lig_idx)
    bond_neighbors: Dict[int, set] = {idx: set() for idx in lig_idx}
    for bond in top.topology.bonds():
        i, j = bond.atom1.index, bond.atom2.index
        if i in lig_set and j in lig_set:
            bond_neighbors[i].add(j)
            bond_neighbors[j].add(i)

    # 不再依赖 GromacsTopFile 的私有字段；直接用 createSystem() 后的真实 OpenMM 力参数。
    extracted_system = top.createSystem(
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False,
    )
    # 🔑 [P0-13 2026-08-04] **必须聚合同类型的所有力，不能 next() 取第一个。**
    #
    # 实测事故：`memtest` 那个膜体系的 System 里有**两个** HarmonicAngleForce ——
    # force[2] 有 31401 个角但**配体一个都不在里面**，force[4] 才是配体那 71 个。
    # 旧代码 `next(...)` 抓到 force[2]，于是 `ligand_only.xml` 的
    # <HarmonicAngleForce> 段是**空的**，溶剂腿的配体因此**没有任何键角项** ⟹
    # 分子变软 ⟹ 预平衡里从 0.996 塌到 0.660 nm 再也回不来 ⟹ 配体–水静电耦合强
    # 3 倍 ⟹ 溶剂腿去电荷 62.80 → 191.05 kJ/mol ⟹ ΔG_bind 变成 +23 kcal/mol。
    # 可溶体系只有一个角力（配体那 71 个混在里面），所以侥幸一直没踩到。
    #
    # 同一个隐患对 bond/torsion 完全一样（那次只是恰好各只有一个力），所以三类
    # 一起改成聚合，并在函数末尾对账写出的项数（见下方 self-check）。
    def _forces_of(kind) -> List:
        return [f for f in extracted_system.getForces() if isinstance(f, kind)]

    nb_candidates = _forces_of(openmm.NonbondedForce)
    if len(nb_candidates) != 1:
        raise RuntimeError(
            f"GROMACS 拓扑构建出的 System 里有 {len(nb_candidates)} 个 NonbondedForce，"
            "无法确定配体的非键参数该从哪一个取。请先收口拓扑，不要让本函数猜。"
        )
    nb_force = nb_candidates[0]
    bond_forces = _forces_of(openmm.HarmonicBondForce)
    angle_forces = _forces_of(openmm.HarmonicAngleForce)
    torsion_forces = _forces_of(openmm.PeriodicTorsionForce)

    def classify_torsion(p1: int, p2: int, p3: int, p4: int) -> Tuple[str, Tuple[int, int, int, int]]:
        atoms4 = (p1, p2, p3, p4)
        if (
            p2 in bond_neighbors.get(p1, ())
            and p3 in bond_neighbors.get(p2, ())
            and p4 in bond_neighbors.get(p3, ())
        ):
            return "Proper", atoms4

        for center in atoms4:
            outer = [x for x in atoms4 if x != center]
            if all(x in bond_neighbors.get(center, ()) for x in outer):
                return "Improper", (center, outer[0], outer[1], outer[2])

        return "Proper", atoms4

    unique_names: Dict[int, str] = {int(i): stable_names[int(i)] for i in lig_idx}

    xml_lines = ["<ForceField>", "  <AtomTypes>"]
    for i in lig_idx:
        elem = atoms[i].element
        mass = elem.mass.value_in_unit(unit.dalton) if hasattr(elem, "mass") else 12.0
        xml_lines.append(f'    <Type name="T{i}" class="T{i}" element="{elem.symbol}" mass="{mass:.6f}"/>')
    xml_lines += ["  </AtomTypes>", "  <Residues>", f'    <Residue name="{ligand_resname}">']
    for i in lig_idx:
        charge = nb_force.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
        xml_lines.append(f'      <Atom name="{unique_names[i]}" type="T{i}" charge="{charge:.8f}"/>')

    for bond in top.topology.bonds():
        i, j = bond.atom1.index, bond.atom2.index
        if i in global_to_local and j in global_to_local:
            xml_lines.append(
                f'      <Bond from="{global_to_local[i]}" to="{global_to_local[j]}"/>'
            )
    xml_lines += ["    </Residue>", "  </Residues>"]

    n_written = {"bond": 0, "angle": 0, "torsion": 0}
    xml_lines.append("  <HarmonicBondForce>")
    for bond_force in bond_forces:
        for bidx in range(bond_force.getNumBonds()):
            p1, p2, length, k = bond_force.getBondParameters(bidx)
            p1 = int(p1)
            p2 = int(p2)
            if p1 in unique_names and p2 in unique_names:
                n_written["bond"] += 1
                xml_lines.append(
                    f'    <Bond class1="T{p1}" class2="T{p2}" '
                    f'length="{length.value_in_unit(unit.nanometer):.8f}" '
                    f'k="{k.value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2):.8f}"/>'
                )
    xml_lines.append("  </HarmonicBondForce>")

    xml_lines.append("  <HarmonicAngleForce>")
    for angle_force in angle_forces:
        for aidx in range(angle_force.getNumAngles()):
            p1, p2, p3, angle, k = angle_force.getAngleParameters(aidx)
            p1, p2, p3 = int(p1), int(p2), int(p3)
            if all(x in unique_names for x in (p1, p2, p3)):
                n_written["angle"] += 1
                xml_lines.append(
                    f'    <Angle class1="T{p1}" class2="T{p2}" class3="T{p3}" '
                    f'angle="{angle.value_in_unit(unit.radian):.8f}" '
                    f'k="{k.value_in_unit(unit.kilojoule_per_mole / unit.radian**2):.8f}"/>'
                )
    xml_lines.append("  </HarmonicAngleForce>")

    xml_lines.append("  <PeriodicTorsionForce>")
    for torsion_force in torsion_forces:
        for tidx in range(torsion_force.getNumTorsions()):
            p1, p2, p3, p4, periodicity, phase, k = torsion_force.getTorsionParameters(tidx)
            p1, p2, p3, p4 = int(p1), int(p2), int(p3), int(p4)
            if all(x in unique_names for x in (p1, p2, p3, p4)):
                n_written["torsion"] += 1
                torsion_tag, ordered = classify_torsion(p1, p2, p3, p4)
                t1, t2, t3, t4 = ordered
                xml_lines.append(
                    f'    <{torsion_tag} class1="T{t1}" class2="T{t2}" class3="T{t3}" class4="T{t4}" '
                    f'periodicity="{int(periodicity)}" '
                    f'phase="{phase.value_in_unit(unit.radian):.8f}" '
                    f'k="{k.value_in_unit(unit.kilojoule_per_mole):.8f}"/>'
                )
    xml_lines.append("  </PeriodicTorsionForce>")

    xml_lines.append('  <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">')
    for i in lig_idx:
        charge, sigma, epsilon = nb_force.getParticleParameters(i)
        xml_lines.append(
            f'    <Atom type="T{i}" '
            f'charge="{charge.value_in_unit(unit.elementary_charge):.8f}" '
            f'sigma="{sigma.value_in_unit(unit.nanometer):.8f}" '
            f'epsilon="{epsilon.value_in_unit(unit.kilojoule_per_mole):.8f}"/>'
        )
    xml_lines.append("  </NonbondedForce>")
    xml_lines.append("</ForceField>")

    # 🔑 [P0-13] 写完就对账：**源体系里配体有多少个成键项，XML 里就必须有多少个**。
    # 这一步就是那个"7 小时之后才发现"的事故本该在 0.1 秒内被拦住的地方 ——
    # 旧代码把 71 个键角悄悄丢成 0，而下游没有任何环节会注意到分子变软了。
    expected = {
        "bond": sum(
            1
            for f in bond_forces
            for i in range(f.getNumBonds())
            if all(int(x) in unique_names for x in f.getBondParameters(i)[:2])
        ),
        "angle": sum(
            1
            for f in angle_forces
            for i in range(f.getNumAngles())
            if all(int(x) in unique_names for x in f.getAngleParameters(i)[:3])
        ),
        "torsion": sum(
            1
            for f in torsion_forces
            for i in range(f.getNumTorsions())
            if all(int(x) in unique_names for x in f.getTorsionParameters(i)[:4])
        ),
    }
    if n_written != expected:
        raise RuntimeError(
            "配体 XML 抽取对账失败（P0-13）：写出的成键项数与源 System 不一致。\n"
            f"    写出 {n_written}，源体系 {expected}\n"
            "    这类静默丢项的后果是分子变软：2026-08-04 实测丢掉全部 71 个键角后，"
            "溶剂腿配体在预平衡里从 0.996 塌到 0.660 nm 且再不恢复，去电荷自由能从"
            "62.80 变成 191.05 kJ/mol，ΔG_bind 变成 +23 kcal/mol。\n"
            "    不要放宽本对账 —— 它是唯一能在 0.1 秒内发现这件事的地方。"
        )
    # 配体本来就该有键角：一个多原子分子写出 0 个角，只可能是抽取错了。
    if len(lig_idx) > 2 and expected["angle"] == 0:
        raise RuntimeError(
            f"配体 {ligand_resname} 有 {len(lig_idx)} 个原子，但源 System 里"
            "**一个键角项都没有** —— 拓扑或力场抽取有问题。"
            "带着这样的参数跑，分子是软的（P0-13）。"
        )
    log.info(
        "  [OK] [P0-13] 配体成键项对账通过: bond=%d angle=%d torsion=%d（源体系逐项相同）",
        n_written["bond"], n_written["angle"], n_written["torsion"],
    )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ligand_only.xml")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(xml_lines))
    log.info("[OK] 已生成配体 XML: %s", out_path)
    return out_path


