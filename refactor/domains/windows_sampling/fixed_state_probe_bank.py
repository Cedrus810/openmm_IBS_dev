# ============================================================================
# 2b. 按"态"共享的可续采 fixed-H 探针轨迹库 [IBS_BIAS_PROTOCOL_VERSION=11]
#
# 替代上面两个按"边"独立采样的探针在 run_all_windows 里的用法（这两个函数
# 本身原样保留，继续服务 abfe_pipeline.py::_probe_vdw_window_fixed_overlap
# ——见 IBS_BIAS_PROTOCOL_VERSION=11 changelog，那个调用点的起始构型是重新
# 最小化过的，跟这里 warmup 阶段活体 context 的当前构型不是同一份，不能共用
# checkpoint，本次只重构 run_all_windows 这一个调用点）。
#
# 与 abfe_pipeline.py 里已有的边级别落盘缓存
# （_fixed_h_probe_fingerprint/_persist_fixed_h_probe_edge/
# _load_fixed_h_probe_cache）不是同一层、也不是重复实现：那是"边结果缓存"
# （存的是某条边探针跑完之后的结果字典），这里是"态原始轨迹缓存"（存的是
# 某个态自己的能量/体积时间序列，供任意需要它的边现算 MBAR）。调用顺序永远
# 是先查边结果缓存，miss 了再落到这里补采，不会反过来。
# ============================================================================

# 🔑 随机种子按"全局态序号"（global_state_index = global_state_start + 窗口内
# 局部态序号）生成，不再按"边"用固定 offset（旧的 probe_bidirectional_overlap
# 系列用 0/1、41337/62143 等常量+0/1 offset）——因为现在一个态的轨迹要同时服务
# 它左右两条边，如果还按边分配种子，同一个态在两次不同的边循环里会被要求
# "既是这个种子又是那个种子"，无意义。四个独立 BASE（路径/校准各自的积分器
# 种子、速度种子）+ 质数步长 97（避免 97*k 在 k 较小时出现周期性碰撞）保证
# 不同态、不同探针类型之间的种子互不相撞，且远高于旧常量，不会跟旧代码路径
# （仍在用的两个按边探针）撞车。
PATH_PROBE_INTEGRATOR_SEED_BASE = 900_000
PATH_PROBE_VELOCITY_SEED_BASE = 901_000
BIAS_CALIB_INTEGRATOR_SEED_BASE = 902_000
BIAS_CALIB_VELOCITY_SEED_BASE = 903_000
_FIXED_H_PROBE_SEED_STRIDE = 97


def _build_fixed_state_simulation(
    topology,
    system_xml: str,
    cv_xml: str,
    require_group4: bool,
    platform_name: str,
    temperature_q,
    positions,
    box_vectors,
    integrator_seed: int,
    velocity_seed: int,
    velocities=None,
    lambda_shield: Optional[float] = None,
):
    """Build one fixed-Hamiltonian ``app.Simulation`` for a single state's own
    trajectory-bank entry -- shared by both ``probe_adjacent_path_overlap_bank``
    (``require_group4=False``, dynamics is ``U_common + CV_k``) and
    ``probe_adjacent_bias_calibration_bank`` (``require_group4=True``,
    dynamics is ``U_common + WCA_window(lambda_shield) + CV_k``). Factors out
    the two hard Group-4 assertions that used to be duplicated, inline, in
    ``probe_bidirectional_overlap``/``probe_bidirectional_overlap_for_bias_calibration``,
    so both directions (missing WCA when required, present WCA when
    forbidden) are independent, unit-testable failures instead of silently
    sampling the wrong ensemble.

    ``velocities``, if given, seeds the Simulation from an exact checkpointed
    velocity array (trajectory continuation -- no re-thermalization); when
    omitted, velocities are freshly drawn from the Maxwell-Boltzmann
    distribution at ``temperature_q`` using ``velocity_seed`` (a brand-new
    segment).
    """
    fixed_system = ensure_owned_system(XmlSerializer.deserialize(system_xml))
    fixed_cv = XmlSerializer.deserialize(cv_xml)
    fixed_cv.setForceGroup(1)
    fixed_system.addForce(fixed_cv)
    has_group4 = any(
        int(fixed_system.getForce(k).getForceGroup()) == 4
        for k in range(fixed_system.getNumForces())
    )
    # 🔑 [WCA_SHIELD_RETIRED，2026-09-02] 壳退役后，`require_group4=True` 这个请求
    # 本身已经不可满足（生产系统里没有 Group 4 可保留）。**不静默降级**：把
    # require_group4 收敛成"必须没有 Group 4"，并让两个分支给出各自的可读报错。
    # `require_group4=False` 那条（独立端点采样的硬断言，见约 10978-11020 要求 #1）
    # 语义完全不变 —— 它本来就要求没有壳，退役只是让它恒真。
    if WCA_SHIELD_RETIRED:
        # `require_group4=True` 在退役状态下是**不可满足**的请求（生产系统里没有
        # Group 4 可保留）。不静默降级成 False：那会让调用方以为自己拿到了含壳
        # 系综。调用方应显式传 `not WCA_SHIELD_RETIRED`。
        if require_group4:
            raise RuntimeError(
                "WCA_SHIELD_RETIRED=True 时不能请求 require_group4=True —— "
                "生产采样已不含 λ-WCA 防护壳（该壳占 stage2 溶剂腿误差 98%），"
                "不存在可保留的 Group 4。调用方请传 `not WCA_SHIELD_RETIRED`；"
                "要复活防护壳请翻该常量，不要绕过本断言。"
            )
        if has_group4:
            raise RuntimeError(
                "WCA_SHIELD_RETIRED=True 但 fixed-H 系统里仍存在 WCA group 4 —— "
                "生产采样已不含防护壳，拒绝在含壳系统上建 fixed-λ 动力学。"
            )
    else:
        if require_group4 and not has_group4:
            raise RuntimeError("bias 校准轨迹库的 fixed-H 系统缺少 Group 4 WCA 防护壳")
        if not require_group4 and has_group4:
            raise RuntimeError("path overlap 轨迹库的 dynamics system 中不应存在 WCA group 4")
    resolved_platform, props = _build_platform_properties(platform_name)
    platform = openmm.Platform.getPlatformByName(resolved_platform)
    integrator = LangevinMiddleIntegrator(
        temperature_q, 2.0 / unit.picosecond, 0.002 * unit.picosecond
    )
    integrator.setConstraintTolerance(1e-3)
    integrator.setRandomNumberSeed(int(integrator_seed))
    if hasattr(integrator, "setRemoveCMMotion"):
        integrator.setRemoveCMMotion(True)
    simulation = app.Simulation(topology, fixed_system, integrator, platform, props)
    if box_vectors is not None:
        simulation.context.setPeriodicBoxVectors(*box_vectors)
    simulation.context.setPositions(positions)
    if require_group4 and lambda_shield is not None and _system_has_global_parameter(fixed_system, "lambda_shield"):
        simulation.context.setParameter("lambda_shield", float(lambda_shield))
    # 🔑 [Hamiltonian mismatch fix] 同上：必须显式对齐到主窗口早已爬坡到的 1.0，
    # 否则 probe_adjacent_path_overlap_bank/probe_adjacent_bias_calibration_bank
    # 用来"证明 f_k 物理正确"的轨迹库实际跑在关闭 Boresch 限制的系统上。
    if _system_has_global_parameter(fixed_system, "lambda_boresch_scale"):
        simulation.context.setParameter("lambda_boresch_scale", 1.0)
    if velocities is not None:
        simulation.context.setVelocities(velocities)
    else:
        simulation.context.setVelocitiesToTemperature(temperature_q, int(velocity_seed))
    return simulation


def _fixed_h_probe_bank_dir(checkpoint_dir: str, stage_type: str, window_idx: int, probe_type: str) -> str:
    return os.path.join(checkpoint_dir, "probes", stage_type, f"window_{window_idx}", probe_type)


def _fixed_h_probe_bank_manifest_path(bank_dir: str) -> str:
    return os.path.join(bank_dir, "manifest.json")


def _fixed_h_probe_bank_state_pointer_path(bank_dir: str, k: int) -> str:
    """Path to the single small pointer file that says which *generation* of
    a state's 5-file artifact set is currently trustworthy -- see
    ``_fixed_h_probe_bank_state_paths`` and ``_persist_state_trajectory_record``.
    """
    return os.path.join(bank_dir, f"state_{k}_generation.json")


def _read_fixed_h_probe_bank_state_generation(bank_dir: str, k: int) -> Optional[int]:
    """Read one state's currently-committed generation number, or ``None``
    if the pointer file is missing/corrupt -- callers must treat that as "no
    persisted record at all" (this pointer is written only after every
    generation-tagged file below it has been durably written, so its absence
    means nothing for this state has ever been fully committed)."""
    pointer_path = _fixed_h_probe_bank_state_pointer_path(bank_dir, k)
    if not os.path.exists(pointer_path):
        return None
    try:
        with open(pointer_path, "r", encoding="utf-8") as f:
            return int(json.load(f)["generation"])
    except Exception:
        return None


def _fixed_h_probe_bank_state_paths(bank_dir: str, k: int, generation: int) -> Tuple[str, str, str, str, str]:
    """Paths for one state's generation-tagged artifact set.

    ``generation`` is the state's own ``sampled_steps`` value at the moment
    these were written -- a naturally monotonic version stamp, reused
    instead of introducing a separate counter. These 5 files are never
    overwritten in place: each persist writes a *new* generation's files,
    then atomically flips ``state_{k}_generation.json`` to point at it (see
    ``_persist_state_trajectory_record``). This is what makes the 5-file set
    atomic as a whole despite each file only being individually atomic on
    its own -- a crash between individual file writes just leaves the
    pointer referencing the last complete generation; the partially-written
    new generation's files are orphaned and ignored, never read, because
    nothing ever resolves a state's files except through the pointer.
    """
    tag = f"g{int(generation)}"
    return (
        os.path.join(bank_dir, f"state_{k}_{tag}_energies.npy"),
        os.path.join(bank_dir, f"state_{k}_{tag}_volume.npy"),
        os.path.join(bank_dir, f"state_{k}_{tag}_checkpoint.npz"),
        os.path.join(bank_dir, f"state_{k}_{tag}_meta.json"),
        os.path.join(bank_dir, f"state_{k}_{tag}_openmm.chk"),
    )


def _build_fixed_h_probe_bank_manifest(
    probe_type: str,
    stage_type: str,
    window_idx: int,
    K: int,
    global_state_start: int,
    common_system_xml: str,
    cv_xmls: List[str],
    lambda_shield: Optional[float],
    temperature_K: float,
    sample_interval: int,
    platform_name: str,
    coion_identity: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Content-based fingerprint for one window's fixed-H probe trajectory
    bank. Any change to the physical Hamiltonian (system/CV XML content,
    cache/bias protocol version, lambda_shield, temperature, sample_interval,
    platform) must invalidate every state's already-sampled frames in this
    window+probe_type directory -- see the resume trust rules in
    ``probe_adjacent_path_overlap_bank``/``probe_adjacent_bias_calibration_bank``.
    XML blobs are hashed (not stored verbatim) to keep manifest.json small.
    """
    manifest = {
        "fixed_h_probe_cache_protocol_version": FIXED_H_PROBE_CACHE_PROTOCOL_VERSION,
        "ibs_bias_protocol_version": IBS_BIAS_PROTOCOL_VERSION,
        "probe_type": str(probe_type),
        "stage_type": str(stage_type),
        "window_idx": int(window_idx),
        "K": int(K),
        "global_state_range": [int(global_state_start), int(global_state_start) + int(K)],
        "common_system_xml_sha256": hashlib.sha256(common_system_xml.encode("utf-8")).hexdigest(),
        "cv_xmls_sha256": [hashlib.sha256(x.encode("utf-8")).hexdigest() for x in cv_xmls],
        "lambda_shield": (float(lambda_shield) if lambda_shield is not None else None),
        "temperature_K": float(temperature_K),
        "step_size_ps": 0.002,
        "friction_per_ps": 2.0,
        "sample_interval": int(sample_interval),
        "platform_name": str(platform_name),
    }
    if coion_identity is not None:
        manifest["coion_identity"] = coion_identity
    if stage_type == "vdw":
        manifest["vdw_nonbonded_protocol_version"] = VDW_NONBONDED_PROTOCOL_VERSION
    return manifest


def _fixed_h_probe_bank_manifest_matches(bank_dir: str, expected_manifest: Dict[str, Any]) -> bool:
    manifest_path = _fixed_h_probe_bank_manifest_path(bank_dir)
    if not os.path.exists(manifest_path):
        return False
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except Exception:
        return False
    normalized_expected = _normalize_ibs_protocol_for_cache_compare(expected_manifest)
    normalized_loaded = _normalize_ibs_protocol_for_cache_compare(loaded)
    return normalized_loaded == normalized_expected


def _invalidate_fixed_h_probe_bank(bank_dir: str) -> None:
    """Discard an entire window+probe_type trajectory bank -- any manifest
    mismatch (protocol version, lambda content, lambda_shield, ...) makes
    every state's already-sampled frames untrustworthy at once, matching the
    existing whole-window convergence-cache philosophy elsewhere in this
    file: trust everything only if every fingerprinted field matches, else
    reject the whole thing and resample from scratch."""
    if os.path.isdir(bank_dir):
        shutil.rmtree(bank_dir, ignore_errors=True)


def _load_state_trajectory_record(
    bank_dir: str, k: int, K: int, sample_interval: int
) -> Optional[Dict[str, Any]]:
    """Load one state's persisted energies/volumes/segment-metadata for the
    generation named by its pointer file, if present and internally
    consistent -- ``None`` otherwise (never a partial/best-effort record):

    - the pointer (``_read_fixed_h_probe_bank_state_generation``) must exist;
    - the pointer's generation-tagged ``energies.npy``/``meta.json`` must
      both exist and parse;
    - ``energies.npy`` must be 2D with exactly ``K`` columns and all-finite;
    - ``volume.npy`` must exist (a missing file is treated as corruption,
      never as "no volume data"), must have the same frame count as
      ``energies.npy``, and must be strictly positive everywhere;
    - ``meta.json``'s own bookkeeping must be self-consistent:
      ``sum(segment.n_frames) == energies.npy`` frame count,
      ``sum(segment.sample_steps) == sampled_steps``, and every segment's
      ``sample_steps == n_frames * sample_interval`` individually (not just
      in aggregate, so a corrupted middle segment can't hide behind
      coincidentally-matching sums).

    Callers must treat any failure here as "start this state's segment 0
    from scratch", never as an empty-but-trustworthy record. Continuation
    state (the native OpenMM checkpoint and its NPZ fallback) is loaded
    separately, lazily, by ``_resume_or_start_state_simulation`` -- this
    function only ever reads the frame-history side of the record.
    """
    generation = _read_fixed_h_probe_bank_state_generation(bank_dir, k)
    if generation is None:
        return None
    energies_path, volume_path, _, meta_path, _ = _fixed_h_probe_bank_state_paths(bank_dir, k, generation)
    if not (os.path.exists(energies_path) and os.path.exists(meta_path)):
        return None
    try:
        u_cv_kj_mol = np.asarray(np.load(energies_path), dtype=np.float64)
        if u_cv_kj_mol.ndim != 2 or u_cv_kj_mol.shape[1] != int(K):
            raise ValueError(f"energies.npy 形状与窗口 K 不符: {u_cv_kj_mol.shape} vs K={K}")
        if not np.all(np.isfinite(u_cv_kj_mol)):
            raise ValueError("energies.npy 含非有限值")
        if not os.path.exists(volume_path):
            # 🔑 volume.npy 缺失不能补零继续：下游路径探针在 LRC 里算
            # coefficient / volume，零体积会先产生 Inf，续采时又可能把这段
            # 零体积和后续真实体积拼接成同一条 volume_nm3。缺失就是记录损坏，
            # 必须让外层 except 捕获、返回 None，强制该态从 segment 0 重采。
            raise ValueError("volume.npy 缺失，判定该态记录损坏")
        volume_nm3 = np.asarray(np.load(volume_path), dtype=np.float64)
        if volume_nm3.shape[0] != u_cv_kj_mol.shape[0]:
            raise ValueError("volume.npy 长度与 energies.npy 帧数不一致")
        if not np.all(np.isfinite(volume_nm3)) or not np.all(volume_nm3 > 0.0):
            raise ValueError("volume.npy 含非有限或非正值")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if int(meta.get("generation", generation)) != int(generation):
            raise ValueError("meta.json 记录的 generation 与 pointer 不一致")
        segments = list(meta["segments"])
        sampled_steps = int(meta["sampled_steps"])
        total_frames = sum(int(seg["n_frames"]) for seg in segments)
        if total_frames != u_cv_kj_mol.shape[0]:
            raise ValueError("segment n_frames 之和与 energies.npy 帧数不一致")
        total_sample_steps = sum(int(seg["sample_steps"]) for seg in segments)
        if total_sample_steps != sampled_steps:
            raise ValueError("segment sample_steps 之和与 sampled_steps 不一致")
        for seg in segments:
            if int(seg["sample_steps"]) != int(seg["n_frames"]) * int(sample_interval):
                raise ValueError("segment 的 sample_steps 与 n_frames*sample_interval 不匹配")
        record = {
            "u_cv_kj_mol": u_cv_kj_mol,
            "volume_nm3": volume_nm3,
            "segments": segments,
            "sampled_steps": sampled_steps,
        }
    except Exception:
        return None
    return record


def _load_state_checkpoint_npz(bank_dir: str, k: int) -> Optional[Dict[str, np.ndarray]]:
    """Load one state's NPZ positions/velocities/box fallback snapshot for
    its currently-committed generation, if present, intact, and stamped with
    a ``sampled_steps`` matching the pointer -- ``None`` otherwise. This is
    the *degraded* resume path (see ``_resume_or_start_state_simulation``):
    it does not capture the integrator's internal RNG state, so it is only
    ever used to seed a fresh, re-burned segment, never as a true
    continuation. Read fresh from disk every time it's needed (never cached
    in memory) since a new generation can be committed between tiers within
    a single bank call.
    """
    generation = _read_fixed_h_probe_bank_state_generation(bank_dir, k)
    if generation is None:
        return None
    _, _, checkpoint_path, _, _ = _fixed_h_probe_bank_state_paths(bank_dir, k, generation)
    if not os.path.exists(checkpoint_path):
        return None
    try:
        with np.load(checkpoint_path) as npz:
            if int(npz["sampled_steps"]) != int(generation):
                return None
            return {
                "positions_nm": np.asarray(npz["positions_nm"], dtype=np.float64),
                "velocities_nm_per_ps": np.asarray(npz["velocities_nm_per_ps"], dtype=np.float64),
                "box_nm": np.asarray(npz["box_nm"], dtype=np.float64),
            }
    except Exception:
        return None


def _atomic_save_openmm_checkpoint(simulation, filepath: str) -> None:
    """Atomically persist OpenMM's own binary Context checkpoint (positions,
    velocities, box *and* the integrator's internal state, including the
    stochastic-thermostat RNG stream) -- write to a temp path, then
    ``os.replace``. This, not the NPZ array dump, is the trajectory bank's
    real continuation mechanism: NPZ alone reconstructs a Context with
    matching positions/velocities but a freshly re-seeded integrator, which
    is a *different* (if still valid) Langevin trajectory, not a bit-level
    continuation of the one that was running -- see
    ``_resume_or_start_state_simulation``.
    """
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    tmp_file = filepath + ".tmp"
    simulation.saveCheckpoint(tmp_file)
    os.replace(tmp_file, filepath)


# 独立于 IBS_BIAS_PROTOCOL_VERSION：只管主窗口（run_all_windows，不是探针轨迹库）
# 这份 OpenMM 原生 checkpoint 文件本身的格式/指纹字段，不影响 ibs_state_*.json
# 里任何已有字段的含义——那部分完全不变，见 IBSSampler.save_ibs_state/load_ibs_state。
#   version 1: 首次引入。真实 GPU 生产日志发现：resumed_calibration_pending 续验
#     冻结校准 f_k 时，每一次阶梯升级重试（frozen_validation_step_overrides 给的
#     50k/150k/300k）都会对这个窗口重新最小化 + dt 测试步进 + Boresch 安全爬坡，
#     即使窗口的构象/速度/盒子在上一次尝试结束时早已充分平衡、冻结 f_k 完全没变
#     ——这是实现缺口，不是物理上必须的（IBS_BIAS_PROTOCOL_VERSION=12 changelog
#     (3) 条曾经声称已经用 OpenMM 原生 checkpoint 解决了这一点，但实际从未真正
#     实现在主窗口路径里，只实现在下面的探针轨迹库里；那条注释已经改正）。
#     现在把探针轨迹库已验证好的 native checkpoint 续算模式原样搬到主窗口。
#   version 2: 同 IBS_BIAS_PROTOCOL_VERSION=13/FIXED_H_PROBE_CACHE_PROTOCOL_
#     VERSION=3 的 lambda_boresch_scale 修复——这份 checkpoint 的 manifest
#     （_build_main_window_checkpoint_manifest）只哈希 win_sys_xml/lambdas/
#     lambda_shield/temperature/platform，Boresch Context 修复不改变其中任何
#     一项（win_sys_xml 序列化字节不变，Boresch 力默认值仍是 0.0，只是运行时
#     setParameter 覆盖），所以旧 checkpoint 会被误判为仍然可用而直接
#     loadCheckpoint 续算——但那份坐标/速度/积分器状态是在 Boresch=0（或未经
#     校准探针验证过 Boresch=1 时）的轨迹上产生的，不能再假设它是干净续算的
#     起点。升版本号强制这次 resume 全部回退到完整重建流程（最小化+dt测试+
#     Boresch 爬坡），不再信任修复之前的主窗口 checkpoint。
MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION = 2


def _main_window_checkpoint_paths(checkpoint_dir: str, stage_type: str, window_idx: int) -> Tuple[str, str]:
    """主窗口 checkpoint 的落盘路径（openmm.chk + manifest.json），布局风格
    照抄探针轨迹库：{checkpoint_dir}/main_window/{stage_type}/window_{idx}/。
    跟探针轨迹库不同，这里只保留最新一份（原地覆盖，不按代际累积）——主窗口
    resume 永远只需要"最后一次看到的状态"，没有多代际分别去相关的需求。
    """
    window_dir = os.path.join(checkpoint_dir, "main_window", str(stage_type), f"window_{int(window_idx)}")
    return (
        os.path.join(window_dir, "openmm.chk"),
        os.path.join(window_dir, "manifest.json"),
    )


def _build_main_window_checkpoint_manifest(
    stage_type: str,
    window_idx: int,
    K: int,
    win_sys_xml: str,
    lambdas_coul: Any,
    lambdas_vdw: Any,
    lambda_shield: Optional[float],
    temperature_K: float,
    platform_name: str,
    repair_policy: str = "non_mutating_v1",
    coion_identity: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """主窗口 checkpoint 的内容指纹——任何一项不匹配都必须整体拒绝这份
    checkpoint（λ 网格被自动加密/重新划分窗口、协议版本变化、平台不同等），
    直接沿用探针轨迹库 `_build_fixed_h_probe_bank_manifest` 的字段/哈希写法。
    """
    manifest = {
        "main_window_checkpoint_protocol_version": MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION,
        "ibs_bias_protocol_version": IBS_BIAS_PROTOCOL_VERSION,
        # 🔑 [non_mutating_v1] 采样修复策略进入 checkpoint 指纹：旧变异策略下的
        # 主窗口 checkpoint（可能带就地重校准过的 f_k）与非变异策略不匹配即拒绝。
        "sampling_repair_policy": str(repair_policy),
        "stage_type": str(stage_type),
        "window_idx": int(window_idx),
        "K": int(K),
        "win_sys_xml_sha256": hashlib.sha256(win_sys_xml.encode("utf-8")).hexdigest(),
        "lambdas_coul": [float(x) for x in lambdas_coul],
        "lambdas_vdw": [float(x) for x in lambdas_vdw],
        "lambda_shield": (float(lambda_shield) if lambda_shield is not None else None),
        "temperature_K": float(temperature_K),
        "step_size_ps": 0.002,
        "friction_per_ps": 2.0,
        "platform_name": str(platform_name),
    }
    if coion_identity is not None:
        manifest["coion_identity"] = coion_identity
    if stage_type == "vdw":
        manifest["vdw_nonbonded_protocol_version"] = VDW_NONBONDED_PROTOCOL_VERSION
    return manifest


def _main_window_checkpoint_is_usable(
    checkpoint_dir: str, stage_type: str, window_idx: int, expected_manifest: Dict[str, Any]
) -> bool:
    """checkpoint 文件与 manifest 文件都必须存在，且 manifest 内容完全匹配才
    返回 True——照抄 `_fixed_h_probe_bank_manifest_matches` 的比较写法（json
    归一化后按内容比较，不是按字符串逐字节比较）。任何缺失/损坏/不匹配都
    安全返回 False，让调用方回退到完整重建流程，绝不抛异常。
    """
    ckpt_path, manifest_path = _main_window_checkpoint_paths(checkpoint_dir, stage_type, window_idx)
    if not (os.path.exists(ckpt_path) and os.path.exists(manifest_path)):
        return False
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except Exception:
        return False
    normalized_expected = _normalize_ibs_protocol_for_cache_compare(expected_manifest)
    normalized_loaded = _normalize_ibs_protocol_for_cache_compare(loaded)
    return normalized_loaded == normalized_expected


def _peek_ibs_bias_status(ibs_state_file: str) -> Optional[str]:
    """只读一次 ibs_state_*.json、返回其中的 bias_status 字段，不做任何其它
    校验（n_states/协议版本/λ 内容是否匹配仍然只由 IBSSampler.load_ibs_state
    的完整校验负责）。这里只是一个"值不值得尝试 loadCheckpoint"的廉价预判，
    任何异常（文件不存在/JSON 损坏/字段缺失）都返回 None，绝不抛出。
    """
    try:
        with open(ibs_state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
        return state.get("bias_status")
    except Exception:
        return None


def _try_load_main_window_checkpoint(sim, checkpoint_path: str) -> bool:
    """尝试从主窗口 checkpoint 恢复这个 Context 的坐标/速度/盒子/积分器 RNG
    状态。任何失败（文件不存在、损坏、跟当前 System/Integrator/Platform 不
    兼容）都必须安全返回 False，绝不能让一个坏 checkpoint 搞崩整个窗口——
    调用方在 False 时应回退到完整的重建+最小化+爬坡流程。
    """
    try:
        sim.loadCheckpoint(checkpoint_path)
        return True
    except Exception:
        return False


# 🔑 [production checkpoint 续采] 独立于 MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION
# （只覆盖 warmup/冻结验证阶段）——那份 checkpoint 在 bias_converged 变 True、
# 真正进入生产采样的那一刻就被删除（见 run_all_windows 里那段清理代码），
# 从未在生产采样阶段被复用过。之前 production ESS 低触发 reseed_resample 时，
# _invalidate_single_window_production 会整窗删掉 energies/bias/base/
# convergence.json，下一轮从头重建 Context、从 stage 起始坐标重新最小化+
# dt测试+Boresch爬坡+冻结重验证+从零步production——"250k 延长到 500k"实际是
# "扔掉这 250k、重跑一条独立的 500k"，不是真正的续算。这里补上生产采样自己的
# native OpenMM checkpoint，让"λ/窗口/系统/冻结 f_k 都没变，只是 production
# ESS 还不够、需要更多样本"这一种（也是最常见的）情况能真正从上次结束的坐标/
# 速度/积分器 RNG 状态接着跑，而不是重新起跑一条独立轨迹。
PRODUCTION_WINDOW_CHECKPOINT_PROTOCOL_VERSION = 1


def _production_window_checkpoint_paths(
    checkpoint_dir: str, stage_type: str, window_idx: int
) -> Tuple[str, str]:
    """生产采样 checkpoint 的落盘路径——布局风格照抄
    `_main_window_checkpoint_paths`：{checkpoint_dir}/production_window/
    {stage_type}/window_{idx}/。同样只保留最新一份，原地覆盖，不按代际累积
    ——续算永远只需要"最后一次看到的状态"，生产阶段没有需要分段去相关的
    多代际需求（不同于探针轨迹库，见 _fixed_h_probe_bank_state_paths）。
    """
    window_dir = os.path.join(
        checkpoint_dir, "production_window", str(stage_type), f"window_{int(window_idx)}"
    )
    return (
        os.path.join(window_dir, "openmm.chk"),
        os.path.join(window_dir, "manifest.json"),
    )


def _build_production_window_checkpoint_manifest(
    stage_type: str,
    window_idx: int,
    K: int,
    win_sys_xml: str,
    lambdas_coul: Any,
    lambdas_vdw: Any,
    lambda_shield: Optional[float],
    lambda_boresch_scale: Optional[float],
    frozen_f_k: Any,
    temperature_K: float,
    platform_name: str,
    repair_policy: str = "non_mutating_v1",
    coion_identity: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """生产采样 checkpoint 的内容指纹——直接沿用
    `_build_main_window_checkpoint_manifest` 的字段/哈希写法（win_sys_xml/
    lambdas/lambda_shield/温度/平台/协议版本），但多两个 warmup checkpoint
    不需要的字段：

    - `lambda_boresch_scale`：warmup checkpoint 続采的是"还在冻结验证阶段"的
      窗口，那时 Boresch 早就爬坡到 1.0 且全程不变，不需要单独校验；但这里
      要独立确认，避免把这份指纹的正确性建立在"warmup 那份检查已经做过"这个
      隐含假设上。
    - `frozen_f_k_sha256`：production checkpoint 场景是"这个窗口已经用某个
      冻结 f_k 采过一批生产样本，现在还要不要接着采"——f_k 若变了（比如同一
      窗口后来被 recalibrate_f_k 覆盖），新旧样本就不再来自同一个采样分布，
      必须老实重采，不能续算；warmup checkpoint 没有这个概念，因为它续的是
      "验证 f_k 是否可信"这件事本身，f_k 在那个阶段还不该被信任为"已经在用
      它采生产样本"。
    """
    manifest = {
        "production_window_checkpoint_protocol_version": PRODUCTION_WINDOW_CHECKPOINT_PROTOCOL_VERSION,
        "ibs_bias_protocol_version": IBS_BIAS_PROTOCOL_VERSION,
        # 🔑 [non_mutating_v1] 采样修复策略进入生产 checkpoint 指纹。
        "sampling_repair_policy": str(repair_policy),
        "stage_type": str(stage_type),
        "window_idx": int(window_idx),
        "K": int(K),
        "win_sys_xml_sha256": hashlib.sha256(win_sys_xml.encode("utf-8")).hexdigest(),
        "lambdas_coul": [float(x) for x in lambdas_coul],
        "lambdas_vdw": [float(x) for x in lambdas_vdw],
        "lambda_shield": (float(lambda_shield) if lambda_shield is not None else None),
        "lambda_boresch_scale": (
            float(lambda_boresch_scale) if lambda_boresch_scale is not None else None
        ),
        "frozen_f_k_sha256": hashlib.sha256(
            json.dumps([round(float(x), 10) for x in frozen_f_k], sort_keys=False).encode("utf-8")
        ).hexdigest(),
        "temperature_K": float(temperature_K),
        "step_size_ps": 0.002,
        "friction_per_ps": 2.0,
        "platform_name": str(platform_name),
    }
    if coion_identity is not None:
        manifest["coion_identity"] = coion_identity
    if stage_type == "vdw":
        manifest["vdw_nonbonded_protocol_version"] = VDW_NONBONDED_PROTOCOL_VERSION
    return manifest


def _resolve_production_entry_marker(
    current_marker: Optional[List[float]],
    current_f_k: np.ndarray,
    resumed_production_checkpoint: bool,
) -> Tuple[str, Optional[List[float]]]:
    """决定 production_entry_f_k 在"即将真正开始生产采样"这一刻该怎么处理
    （checkpoint 恢复/续采判断、安全检查、f_k 与冻结 manifest 一致性断言全部
    已经通过之后调用）。纯函数，不碰 sampler/磁盘，方便独立测试；调用方
    （run_all_windows）据返回值决定是否真的写 sampler.production_entry_f_k
    和调用 save_ibs_state()。

    返回 (action, marker)：
      "keep"   —— 原样保留 sampler 当前的 production_entry_f_k，不写、不存。
      "set"    —— 调用方应把 production_entry_f_k 设为 marker 并 save_ibs_state()。
      "refuse" —— 调用方应拒绝继续生产（真实数据矛盾），marker 为 None。

    🔑 [续跑边界，2026-08-27] current_marker 为 None 不能一律当成"这是第一次
    进入生产"直接写 current_f_k：如果 resumed_production_checkpoint 为
    True——已经存在一份真正驱动过采样、且 manifest 已确认属于这份冻结 f_k
    的旧生产历史（energies/bias/base.npy + 匹配的 checkpoint）——那份历史
    最初进入生产时的 f_k 本来就是真的不知道（旧版本 ibs_engine.py 从没记过
    这个字段）；这时候写一个"现在"的值会把一段入口未知的旧历史悄悄升级成
    "生产入口标记验证通过"，正是这个字段本该防止的事——必须继续保持 None，
    让下游 reconcile 走 degraded 路径。只有 resumed_production_checkpoint 为
    False（这次生产确实是从 0 步重新开始，没有沿用任何旧生产样本）时，
    marker=None 才代表"这次的生产历史确实还没有入口标记"，可以第一次写入。

    同理，marker 非 None 但与 current_f_k 对不上：如果 resumed_production_
    checkpoint 为 True，manifest 已经确认这份 checkpoint 的生产历史就是
    current_f_k 驱动出来的，那这份不匹配的旧 marker 就是真正的数据不一致
    （标记与它本该对应的历史对不上），必须拒绝，不能用当前值静默覆盖掉一份
    可能记录着真实矛盾的标记；resumed_production_checkpoint 为 False 时，
    这份不匹配的 marker 只是来自另一份已作废的校准，可以安全覆盖。
    """
    current_arr = np.asarray(current_f_k, dtype=np.float64)
    if current_marker is None:
        if resumed_production_checkpoint:
            return ("keep", None)
        return ("set", [float(x) for x in current_arr])
    matches = (
        len(current_marker) == len(current_arr)
        and np.allclose(
            np.asarray(current_marker, dtype=np.float64), current_arr, rtol=0.0, atol=1.0e-12
        )
    )
    if matches:
        return ("keep", None)
    if resumed_production_checkpoint:
        return ("refuse", None)
    return ("set", [float(x) for x in current_arr])


def _production_window_checkpoint_is_usable(
    checkpoint_dir: str, stage_type: str, window_idx: int, expected_manifest: Dict[str, Any]
) -> bool:
    """checkpoint 文件与 manifest 文件都必须存在，且 manifest 内容完全匹配才
    返回 True——照抄 `_main_window_checkpoint_is_usable` 的比较写法。任何
    缺失/损坏/不匹配都安全返回 False，让调用方回退到今天的删除+完整重采
    流程，绝不抛异常。
    """
    ckpt_path, manifest_path = _production_window_checkpoint_paths(
        checkpoint_dir, stage_type, window_idx
    )
    if not (os.path.exists(ckpt_path) and os.path.exists(manifest_path)):
        return False
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except Exception:
        return False
    normalized_expected = _normalize_ibs_protocol_for_cache_compare(expected_manifest)
    normalized_loaded = _normalize_ibs_protocol_for_cache_compare(loaded)
    for manifest in (normalized_expected, normalized_loaded):
        if "stage_protocol_key" in manifest:
            manifest["stage_protocol_key"] = _stage_window_sampling_identity(manifest["stage_protocol_key"])
    return normalized_loaded == normalized_expected


def _invalidate_production_window_checkpoint(checkpoint_dir: str, stage_type: str, window_idx: int) -> None:
    """删除一个窗口的生产 checkpoint+manifest——production 数据本身因为
    f_k/λ 真的变了而被丢弃时（`recalibrate_f_k`，或路径变化的整窗重排），
    这份 checkpoint 也必须一起作废，否则会在物理上不再对应的新一轮生产
    采样里被误当成"续算起点"。任何缺失都安全忽略。
    """
    ckpt_path, manifest_path = _production_window_checkpoint_paths(
        checkpoint_dir, stage_type, window_idx
    )
    for _path in (ckpt_path, manifest_path):
        if os.path.exists(_path):
            try:
                os.remove(_path)
            except OSError:
                pass


def _persist_state_trajectory_record(bank_dir: str, k: int, record: Dict[str, Any], simulation) -> None:
    """Persist one state's energies/volumes/segment-metadata, a native
    OpenMM checkpoint (the real continuation mechanism) and an NPZ
    positions/velocities/box fallback snapshot as a new *generation*
    (tagged with this state's post-extension ``sampled_steps``), then
    atomically commit the pointer that names it current -- only after all 5
    generation-tagged files have been individually, durably written.

    This is what makes the 5-file set atomic as a *whole*, not just
    file-by-file: a crash at any point before the pointer write leaves it
    referencing the previous (complete, self-consistent) generation, so a
    resume never sees energies/meta claiming more sampled steps than the
    checkpoint can actually back up (the scenario this guards against: a
    crash between updating energies/meta and updating the checkpoint used to
    silently leave ``sampled_steps`` ahead of the checkpoint's real physical
    state, so the *next* resume would treat a discontinuous restart as a
    same-segment continuation). The previous generation's files are removed
    only after the new pointer is committed (best-effort; leftover orphans
    from an interrupted cleanup are harmless since nothing ever reads a
    generation the pointer doesn't name).
    """
    new_generation = int(record["sampled_steps"])
    old_generation = _read_fixed_h_probe_bank_state_generation(bank_dir, k)

    energies_path, volume_path, checkpoint_path, meta_path, openmm_checkpoint_path = (
        _fixed_h_probe_bank_state_paths(bank_dir, k, new_generation)
    )
    _atomic_save_npy(energies_path, record["u_cv_kj_mol"])
    _atomic_save_npy(volume_path, record["volume_nm3"])
    _atomic_write_json(meta_path, {
        "segments": record["segments"],
        "sampled_steps": new_generation,
        "generation": new_generation,
    })
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    positions_nm = _positions_to_nm_array(state.getPositions())
    velocities_nm_per_ps = np.asarray(
        state.getVelocities().value_in_unit(unit.nanometer / unit.picosecond), dtype=np.float64
    )
    box_nm = _box_vectors_to_nm_array(state.getPeriodicBoxVectors())
    _atomic_save_npz(
        checkpoint_path,
        positions_nm=positions_nm,
        velocities_nm_per_ps=velocities_nm_per_ps,
        box_nm=box_nm,
        sampled_steps=np.asarray(new_generation),
    )
    _atomic_save_openmm_checkpoint(simulation, openmm_checkpoint_path)

    # 🔑 提交点：只有以上 5 个 generation 文件全部落盘成功后，才原子翻转这个
    # 小指针文件——在此之前崩溃，指针仍指向上一个完整、自洽的 generation，
    # 半成品的新 generation 文件成为孤儿，永远不会被读到。
    _atomic_write_json(
        _fixed_h_probe_bank_state_pointer_path(bank_dir, k), {"generation": new_generation}
    )

    if old_generation is not None and old_generation != new_generation:
        for stale_path in _fixed_h_probe_bank_state_paths(bank_dir, k, old_generation):
            try:
                os.remove(stale_path)
            except OSError:
                pass


def _resume_or_start_state_simulation(
    k: int,
    bank_dir: str,
    has_prior_segments: bool,
    topology,
    system_xml: str,
    cv_xml: str,
    require_group4: bool,
    platform_name: str,
    temperature_q,
    positions,
    box_vectors,
    integrator_seed: int,
    velocity_seed: int,
    lambda_shield: Optional[float] = None,
):
    """Build one state's fixed-H ``Simulation`` for this round, resuming it
    via OpenMM's own binary Context checkpoint whenever possible.

    The native checkpoint (``_atomic_save_openmm_checkpoint``) is the only
    mechanism that also restores the integrator's internal RNG state, so
    only a successful ``loadCheckpoint`` here counts as a true, uninterrupted
    continuation (``needs_burn_in=False``, same segment). If it is missing,
    corrupted, or incompatible (different platform/OpenMM build -- native
    checkpoints are not portable) but the NPZ positions/velocities/box
    fallback is available, that seeds a *new* (re-burned) segment instead --
    reusing the last known configuration avoids re-equilibrating from a far
    configuration, but does not claim a continuation it cannot back up.
    Falls further back to the caller-supplied current window configuration
    if neither is available.

    Returns ``(simulation, needs_burn_in, reason)``. The caller is
    responsible for releasing ``simulation`` (``del`` + ``gc.collect()``)
    right after extending and persisting it -- this function never caches or
    retains a reference to it, by design: only one dynamics Context should be
    alive at a time (see IBS_BIAS_PROTOCOL_VERSION=11 changelog).
    """
    checkpoint_arrays = _load_state_checkpoint_npz(bank_dir, k)
    if checkpoint_arrays is not None:
        sim_positions = checkpoint_arrays["positions_nm"] * unit.nanometer
        sim_velocities = checkpoint_arrays["velocities_nm_per_ps"] * (unit.nanometer / unit.picosecond)
        sim_box = checkpoint_arrays["box_nm"] * unit.nanometer
    else:
        sim_positions = positions
        sim_velocities = None
        sim_box = box_vectors

    sim = _build_fixed_state_simulation(
        topology=topology,
        system_xml=system_xml,
        cv_xml=cv_xml,
        require_group4=require_group4,
        platform_name=platform_name,
        temperature_q=temperature_q,
        positions=sim_positions,
        box_vectors=sim_box,
        integrator_seed=integrator_seed,
        velocity_seed=velocity_seed,
        velocities=sim_velocities,
        lambda_shield=lambda_shield,
    )

    resumed_native = False
    generation = _read_fixed_h_probe_bank_state_generation(bank_dir, k)
    if has_prior_segments and generation is not None:
        _, _, _, _, openmm_checkpoint_path = _fixed_h_probe_bank_state_paths(bank_dir, k, generation)
        if os.path.exists(openmm_checkpoint_path):
            try:
                sim.loadCheckpoint(openmm_checkpoint_path)
                resumed_native = True
            except Exception:
                resumed_native = False

    if resumed_native:
        return sim, False, "resumed_from_native_checkpoint"
    if checkpoint_arrays is not None:
        return sim, True, "native_checkpoint_missing_or_incompatible_npz_fallback"
    return sim, True, ("checkpoint_missing_or_corrupted" if has_prior_segments else "fresh")


def _decorrelate_per_segment(
    diff_series: np.ndarray,
    segments: List[Dict[str, Any]],
    min_frames_for_subsampling: int = 20,
) -> Tuple[np.ndarray, List[float], List[Dict[str, Any]]]:
    """Decorrelate one state's difference-energy series one segment at a
    time, never across a restart boundary.

    A state's trajectory-bank record can span multiple segments (a brand-new
    burn-in, plus zero or more re-burned segments opened after a
    ``checkpoint.npz`` was lost -- see ``probe_adjacent_path_overlap_bank``'s
    resume rules). Concatenating frames from two independently-reburned
    segments before estimating the autocorrelation time would treat the
    discontinuous jump between them as real dynamics, corrupting the
    statistical inefficiency estimate -- so each segment's own frame slice is
    passed to ``subsample_series_by_autocorrelation`` independently, and only
    the resulting (already-decorrelated) index sets are concatenated.

    A segment shorter than ``min_frames_for_subsampling`` is excluded from
    the returned ``combined`` index set entirely -- estimating an
    autocorrelation time from too few points is noisier than not
    decorrelating at all, and a re-burn-in segment opened after checkpoint
    loss is exactly the case where those few points are also the most
    suspect (fresh burn-in, not yet equilibrated). Letting such a segment
    into MBAR at full weight (``g=1``, unweighted) would let it silently
    inflate ``n_k_decorrelated`` and blend with normally-decorrelated
    segments. Such segments are instead reported in the returned
    ``short_segments`` list (segment index + frame count) for
    diagnostic-only use by callers; if excluding them leaves a state with
    too few (or zero) decorrelated frames, the downstream MBAR call fails
    closed on its own sample-count check rather than silently proceeding.
    """
    if sum(int(seg["n_frames"]) for seg in segments) != diff_series.shape[0]:
        raise RuntimeError("segment 边界与能量序列长度不一致，拒绝跨段去相关")
    all_indices = []
    inefficiencies = []
    short_segments = []
    cursor = 0
    for seg_idx, seg in enumerate(segments):
        n = int(seg["n_frames"])
        if n < int(min_frames_for_subsampling):
            short_segments.append({"segment_index": seg_idx, "n_frames": n})
            cursor += n
            continue
        segment_slice = diff_series[cursor:cursor + n]
        local_indices, g = subsample_series_by_autocorrelation(
            segment_slice, min_frames_for_subsampling=min_frames_for_subsampling
        )
        all_indices.append(np.asarray(local_indices, dtype=int) + cursor)
        inefficiencies.append(float(g))
        cursor += n
    combined = np.concatenate(all_indices) if all_indices else np.arange(0, dtype=int)
    return combined, inefficiencies, short_segments


def _extend_state_trajectory(
    simulation,
    record: Dict[str, Any],
    evaluator,
    target_steps: int,
    sample_interval: int,
    burn_in_steps: int,
    needs_burn_in: bool,
    segment_reason: str,
    frame_observer=None,
) -> Dict[str, Any]:
    """Advance one state's fixed-H trajectory in-place to >= ``target_steps``
    sampled steps, mutating and returning ``record``.

    Never re-burns an already-open segment: ``needs_burn_in`` must be True
    only for a state's very first segment or a segment reopened after
    checkpoint loss (see ``probe_adjacent_*_bank``'s resume trust rules) --
    repeated calls against the same live ``simulation`` within an
    already-open segment must pass ``needs_burn_in=False``, which only steps
    ``target_steps - record["sampled_steps"]`` additional steps and appends
    to the existing (last) segment's own counters instead of opening a new
    one. Energies are stored raw (kJ/mol, no beta reduction, no LRC) against
    every state in the window -- reduction to reduced units and adding LRC
    (path bank only) happens later, in ``_analyze_adjacent_pair``, once it is
    known which two columns a given edge actually needs.

    ``frame_observer``, if given, is called as ``frame_observer(positions,
    box_vectors)`` once per stored frame and its return value is appended to
    ``record["observer_values"]`` -- used by the independent-endpoint
    production sampler (INDEPENDENT_ENDPOINT_PROTOCOL_VERSION) to record the
    per-frame cavity water count alongside the energies. It is deliberately a
    callback rather than a second sampling pass: the structural observable
    must come from *exactly* the frames that feed MBAR, not from a separately
    strided re-read of the trajectory.
    """
    additional_steps = int(target_steps) - int(record["sampled_steps"])
    if additional_steps <= 0:
        return record
    if sample_interval <= 0 or additional_steps < sample_interval:
        raise ValueError("fixed-H 轨迹库单次延长的步数不足一个 sample_interval")
    if needs_burn_in:
        guarded_step(simulation, int(burn_in_steps), "fixed-H 轨迹库 burn-in")
        record["segments"].append({
            "burn_in_steps": int(burn_in_steps),
            "sample_steps": 0,
            "n_frames": 0,
            "reason": str(segment_reason),
        })
    elif not record["segments"]:
        raise RuntimeError("延长一个尚无任何 segment 的轨迹时必须先烧 burn-in")

    n_new_samples = additional_steps // int(sample_interval)
    new_energies = []
    new_volumes = []
    new_observations = []
    for _ in range(n_new_samples):
        guarded_step(simulation, int(sample_interval), "fixed-H 轨迹库采样")
        frame_state = simulation.context.getState(getPositions=True)
        frame_positions = frame_state.getPositions()
        frame_box = frame_state.getPeriodicBoxVectors()
        e_kj = evaluator.evaluate_interaction_energies(frame_positions, frame_box)
        new_energies.append(np.asarray(e_kj, dtype=np.float64))
        box_nm = _box_vectors_to_nm_array(frame_box)
        volume_nm3 = abs(float(np.linalg.det(box_nm)))
        if not np.isfinite(volume_nm3) or volume_nm3 <= 0.0:
            raise RuntimeError("fixed-H 轨迹库帧的盒子体积无效")
        new_volumes.append(volume_nm3)
        if frame_observer is not None:
            new_observations.append(frame_observer(frame_positions, frame_box))
    new_energies = np.asarray(new_energies, dtype=np.float64)
    new_volumes = np.asarray(new_volumes, dtype=np.float64)
    if frame_observer is not None:
        record["observer_values"] = (
            list(record.get("observer_values") or []) + new_observations
        )
    if record["u_cv_kj_mol"] is None:
        record["u_cv_kj_mol"] = new_energies
        record["volume_nm3"] = new_volumes
    else:
        record["u_cv_kj_mol"] = np.concatenate([record["u_cv_kj_mol"], new_energies], axis=0)
        record["volume_nm3"] = np.concatenate([record["volume_nm3"], new_volumes], axis=0)
    added_steps = int(n_new_samples) * int(sample_interval)
    record["sampled_steps"] += added_steps
    record["segments"][-1]["sample_steps"] += added_steps
    record["segments"][-1]["n_frames"] += int(n_new_samples)
    return record


def _analyze_adjacent_pair(
    record_i: Dict[str, Any],
    record_j: Dict[str, Any],
    local_i: int,
    local_j: int,
    kt: float,
    lrc_coeff: Optional[np.ndarray] = None,
    threshold: float = 0.03,
    min_frames_for_subsampling: int = 20,
) -> Dict[str, Any]:
    """Reduce two states' shared trajectory-bank records to one adjacent-pair
    MBAR overlap/delta_f result, reusing
    ``_compute_bidirectional_overlap_from_u_kn`` unchanged.

    Each state's trajectory independently serves every edge it borders --
    state k's frames back both edge (k-1,k) and edge (k,k+1) -- but this
    function only ever reads the two energy columns ``[local_i, local_j]``
    needed for one specific edge and is otherwise self-contained, exactly
    like the original per-edge probes: sharing a trajectory across edges does
    not bias either edge's own MBAR estimate (each edge's u_kn/n_k pairing is
    unchanged); it only introduces a correlation between adjacent edges'
    uncertainty estimates if those were ever summed into a path-level
    uncertainty -- no code currently does that, but it is worth remembering
    if that ever changes.
    """
    beta = 1.0 / float(kt)
    reduced_by_ensemble = []
    inefficiencies = []
    short_segments_report = {}
    for label, record in (("state_i", record_i), ("state_j", record_j)):
        e_kj = record["u_cv_kj_mol"][:, [local_i, local_j]].astype(np.float64, copy=True)
        if lrc_coeff is not None:
            volume = np.asarray(record["volume_nm3"], dtype=np.float64)
            lrc_pair = np.asarray(lrc_coeff, dtype=np.float64)[[local_i, local_j]]
            e_kj = e_kj + lrc_pair[None, :] / volume[:, None]
        reduced = beta * e_kj  # (n_frames, 2): column 0 = local_i, column 1 = local_j
        diff = reduced[:, 1] - reduced[:, 0]
        decorrelated, g_list, short_segs = _decorrelate_per_segment(
            diff, record["segments"], min_frames_for_subsampling=min_frames_for_subsampling
        )
        reduced_by_ensemble.append(reduced[decorrelated, :].T)  # (2, n_decorrelated)
        inefficiencies.append(g_list)
        if short_segs:
            short_segments_report[label] = short_segs

    n_k = np.asarray([block.shape[1] for block in reduced_by_ensemble], dtype=int)
    u_kn = np.concatenate(reduced_by_ensemble, axis=1)
    result = _compute_bidirectional_overlap_from_u_kn(u_kn, n_k, threshold=threshold)
    result.update({
        "local_states": [int(local_i), int(local_j)],
        "n_k_decorrelated": n_k.tolist(),
        "statistical_inefficiency": inefficiencies,
        "short_segments_diagnostic_only": short_segments_report,
    })
    return result


def detect_passed_but_asymmetric_overlap_bottleneck(
    pairs: List[Dict[str, Any]],
    lambdas_window,
    max_weak_overlap: float = 0.12,
    min_overlap_ratio: float = 2.5,
    min_slope_ratio: float = 3.0,
    min_delta_f_gap_kj_mol: float = 5.0,
    uncertainty_sigma_multiplier: float = 2.0,
) -> Optional[Dict[str, Any]]:
    """识别 fixed-H 全通过、但局部热力学密度严重不均匀的瓶颈边。

    只有当同一条边同时是最低 overlap 和最大 |ΔF|/|Δλ| 时才可能命中，且差距必须
    显著超过 ΔF 的联合不确定度噪声地板；否则返回 None，绝不在指标含糊时插点。
    """
    if len(pairs) < 2 or not all(bool(p.get("passed")) for p in pairs):
        return None

    lambdas_window = np.asarray(lambdas_window, dtype=np.float64)
    records = []
    for pair in pairs:
        local_states = pair.get("local_states")
        if not local_states or len(local_states) != 2:
            return None

        i, j = map(int, local_states)
        delta_lambda = abs(float(lambdas_window[j] - lambdas_window[i]))
        overlap = float(pair.get("min_bidirectional_overlap", np.nan))
        delta_f = abs(float(pair.get("delta_f_kJ_mol", np.nan)))
        uncertainty = float(
            pair.get("delta_f_uncertainty_kJ_mol", np.nan)
        )

        if (
            delta_lambda <= 0.0
            or not np.isfinite(overlap)
            or not np.isfinite(delta_f)
        ):
            return None

        records.append({
            "pair": pair,
            "overlap": overlap,
            "delta_f": delta_f,
            "uncertainty": uncertainty,
            "slope": delta_f / delta_lambda,
        })

    weakest = min(records, key=lambda x: x["overlap"])
    steepest = max(records, key=lambda x: x["slope"])

    # 必须由同一条边同时表现为最低 overlap 和最大 |ΔF|/|Δλ|。
    if weakest is not steepest:
        return None

    references = [r for r in records if r is not weakest]
    reference = max(references, key=lambda x: x["overlap"])

    overlap_ratio = reference["overlap"] / max(weakest["overlap"], 1e-12)
    slope_ratio = weakest["slope"] / max(reference["slope"], 1e-12)
    delta_f_gap = weakest["delta_f"] - reference["delta_f"]

    significant_gap = float(min_delta_f_gap_kj_mol)
    if (
        np.isfinite(weakest["uncertainty"])
        and np.isfinite(reference["uncertainty"])
    ):
        significant_gap = max(
            significant_gap,
            uncertainty_sigma_multiplier * np.hypot(
                weakest["uncertainty"],
                reference["uncertainty"],
            ),
        )

    if not (
        weakest["overlap"] <= max_weak_overlap
        and overlap_ratio >= min_overlap_ratio
        and slope_ratio >= min_slope_ratio
        and delta_f_gap >= significant_gap
    ):
        return None

    return {
        "qualified": True,
        "pair": dict(weakest["pair"]),
        "global_edge": list(weakest["pair"]["global_edge"]),
        "weak_overlap": weakest["overlap"],
        "reference_overlap": reference["overlap"],
        "overlap_ratio": overlap_ratio,
        "slope_ratio": slope_ratio,
        "delta_f_gap_kJ_mol": delta_f_gap,
        "required_delta_f_gap_kJ_mol": significant_gap,
    }


def _bias_calibration_pair_is_sufficient(
    pair: Optional[Dict[str, Any]],
    min_decorrelated_samples: int = 20,
    max_delta_f_uncertainty_kJ_mol: float = 1.0,
) -> bool:
    """Explicit three-way AND gate for accepting one edge's bias-calibration
    result as precise enough to overwrite f_k: it must (1) pass the overlap
    threshold, (2) have >= ``min_decorrelated_samples`` on the smaller side,
    and (3) have delta_f_bias uncertainty <= ``max_delta_f_uncertainty_kJ_mol``.
    Any one failing condition rejects the pair -- this used to be enforced ad
    hoc inline (``probe_bidirectional_overlap_for_bias_calibration`` returning
    ``None``); factoring it out here makes each condition independently
    unit-testable.
    """
    if pair is None:
        return False
    if not pair.get("passed"):
        return False
    n_k = pair.get("n_k_decorrelated")
    if not n_k or int(np.min(n_k)) < int(min_decorrelated_samples):
        return False
    uncertainty = pair.get("delta_f_bias_uncertainty_kJ_mol")
    if uncertainty is None or not np.isfinite(uncertainty):
        return False
    if float(uncertainty) > float(max_delta_f_uncertainty_kJ_mol):
        return False
    return True


def probe_adjacent_path_overlap_bank(
    topology,
    common_system_xml: str,
    ibs_wrapper,
    K: int,
    positions,
    box_vectors,
    temperature,
    platform_name: str,
    checkpoint_dir: str,
    stage_type: str,
    window_idx: int,
    global_state_start: int,
    burn_in_steps: int = 5000,
    sample_targets: Tuple[int, ...] = (20_000,),
    sample_interval: int = 500,
    threshold: float = 0.03,
    coion_identity: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Per-state, resumable replacement for calling ``probe_bidirectional_overlap``
    once per edge.

    A K-state window has K-1 adjacent edges but only K states; the old
    per-edge probe built two brand-new fixed-H Simulations and burned two
    fresh burn-ins for *every* edge, so an interior state (shared by two
    edges) was independently equilibrated and sampled twice. This builds
    exactly one Simulation per state, burns its Langevin dynamics in once,
    and lets ``_analyze_adjacent_pair`` read whichever two energy columns a
    given edge needs from that one shared trajectory.

    ``sample_targets`` defaults to a *single* 20k-step tier, not a growing
    one -- unlike bias calibration, path/overlap failure means the lambda
    grid itself is too sparse (an edge's two states genuinely don't share
    enough phase space), which more sampling of the *same* two states cannot
    fix; growing the tier here would just burn extra GPU time re-confirming
    the same structural verdict (K=3's verified savings, 100,000 -> 75,000
    steps, assumes exactly this single-tier default). Callers may still pass
    a longer/growing ``sample_targets`` explicitly if they have a reason to;
    when more than one tier is given, only the two end states of any edge
    that still fails the overlap threshold are extended to the next tier --
    states whose surrounding edges already passed stop consuming GPU time.
    Trajectories, segment metadata, a native OpenMM checkpoint and an NPZ
    fallback snapshot are persisted after every extension (see
    ``_persist_state_trajectory_record``), keyed by a content fingerprint
    (``_build_fixed_h_probe_bank_manifest``) that invalidates the *entire*
    window+probe_type directory if the physical Hamiltonian changed, but
    otherwise resumes each state independently -- see
    ``_resume_or_start_state_simulation`` for the native-checkpoint /
    NPZ-fallback / current-configuration resume order.

    At most one state's dynamics ``Simulation``/Context is ever alive at a
    time (plus the one lightweight energy-evaluation probe Context shared
    across all states) -- each state is built, extended, checkpointed and
    released before the next one starts, even within a single
    ``sample_targets`` tier, so a K-state window never holds K simultaneous
    GPU Contexts.

    This bank is checked *before* falling back to any edge-result cache in
    ``abfe_pipeline.py`` (``_fixed_h_probe_fingerprint``/
    ``_load_fixed_h_probe_cache``) -- that cache stores *edge results*, keyed
    by window content, one layer above this one; this bank stores *raw
    per-state trajectories* one layer below. A miss in the edge-result cache
    falls through to sampling here, never the other way around; the two are
    not competing implementations of the same thing.
    """
    if K < 2:
        raise ValueError("path overlap 轨迹库至少需要两个态")
    cv_xmls = list(getattr(ibs_wrapper, "_int_cv_force_xmls", []))
    if len(cv_xmls) < K:
        raise RuntimeError("IBS wrapper 缺少 path overlap 轨迹库所需的 CV XML")
    if sample_interval <= 0:
        raise ValueError("path overlap 轨迹库采样间隔必须为正")
    temperature_q = (
        temperature if hasattr(temperature, "value_in_unit") else float(temperature) * unit.kelvin
    )
    kt = (unit.MOLAR_GAS_CONSTANT_R * temperature_q).value_in_unit(unit.kilojoule_per_mole)
    lrc_coeff = getattr(ibs_wrapper, "lj_tail_lrc_coeff_kj_mol", None)

    bank_dir = _fixed_h_probe_bank_dir(checkpoint_dir, stage_type, window_idx, "path_probe")
    expected_manifest = _build_fixed_h_probe_bank_manifest(
        probe_type="path_probe",
        stage_type=stage_type,
        window_idx=window_idx,
        K=K,
        global_state_start=global_state_start,
        common_system_xml=common_system_xml,
        cv_xmls=cv_xmls[:K],
        lambda_shield=None,
        temperature_K=temperature_q.value_in_unit(unit.kelvin),
        sample_interval=sample_interval,
        platform_name=platform_name,
        coion_identity=coion_identity,
    )
    # 🔑 先查指纹、指纹对了才续采：任何一项不匹配就整个目录判定不可信，
    # 必须在决定"续采 vs 重新分段"之前完成，不能只是隐含意图。
    if not _fixed_h_probe_bank_manifest_matches(bank_dir, expected_manifest):
        _invalidate_fixed_h_probe_bank(bank_dir)
        _atomic_write_json(_fixed_h_probe_bank_manifest_path(bank_dir), expected_manifest)

    records: Dict[int, Dict[str, Any]] = {}
    for k in range(K):
        record = _load_state_trajectory_record(bank_dir, k, K, sample_interval)
        if record is None:
            record = {"u_cv_kj_mol": None, "volume_nm3": None, "segments": [], "sampled_steps": 0}
        records[k] = record

    evaluator = None
    pairs: List[Optional[Dict[str, Any]]] = [None] * (K - 1)
    active = set(range(K))
    for target in sample_targets:
        states_to_extend = sorted(s for s in active if records[s]["sampled_steps"] < target)
        for k in states_to_extend:
            global_k = int(global_state_start) + int(k)
            sim, needs_burn_in_k, segment_reason = _resume_or_start_state_simulation(
                k=k,
                bank_dir=bank_dir,
                has_prior_segments=bool(records[k]["segments"]),
                topology=topology,
                system_xml=common_system_xml,
                cv_xml=cv_xmls[k],
                require_group4=False,
                platform_name=platform_name,
                temperature_q=temperature_q,
                positions=positions,
                box_vectors=box_vectors,
                integrator_seed=PATH_PROBE_INTEGRATOR_SEED_BASE + _FIXED_H_PROBE_SEED_STRIDE * global_k,
                velocity_seed=PATH_PROBE_VELOCITY_SEED_BASE + _FIXED_H_PROBE_SEED_STRIDE * global_k,
            )
            try:
                if evaluator is None:
                    evaluator = IBSSampler(
                        sim.context, K, temperature_q,
                        prefix=getattr(ibs_wrapper, "prefix", "abfe"),
                        ibs_wrapper=ibs_wrapper,
                    )
                    # 🔑 evaluator 在这条路径上只会被调用 evaluate_interaction_energies，
                    # 那个方法只读 self._probe_context/self._probe_groups（已经在
                    # 上面 __init__ 里用这台 sim 的 Context 建好了独立的探针
                    # Context），永远不再读 self.context。构造完就立刻丢掉这份
                    # 对本态动力学 Context 的引用，否则它会跟随 evaluator 活到
                    # 整个 bank 探针函数结束，导致下一态的 dynamics Context 建立
                    # 后，旧 dynamics Context、当前 dynamics Context、probe
                    # Context 同时占着显存。
                    evaluator.context = None
                _extend_state_trajectory(
                    sim,
                    records[k],
                    evaluator,
                    target_steps=target,
                    sample_interval=sample_interval,
                    burn_in_steps=burn_in_steps,
                    needs_burn_in=needs_burn_in_k,
                    segment_reason=segment_reason,
                )
                _persist_state_trajectory_record(bank_dir, k, records[k], sim)
            finally:
                # 🔑 每个态延长完、落盘完立刻释放它的动力学 Context -- 峰值只
                # 保持一个动力学 Context + 一个 evaluator 探针 Context，K 态
                # 窗口不会同时常驻 K 个 GPU Context。
                del sim
                gc.collect()

        all_passed = True
        for local_i in range(K - 1):
            pair = _analyze_adjacent_pair(
                records[local_i], records[local_i + 1], local_i, local_i + 1, kt,
                lrc_coeff=lrc_coeff, threshold=threshold,
            )
            pair["global_edge"] = [int(global_state_start) + local_i, int(global_state_start) + local_i + 1]
            pair["burn_in_steps"] = int(burn_in_steps)
            pair["sample_interval"] = int(sample_interval)
            pair["dynamics_hamiltonian"] = "U_common_plus_single_cv_int"
            pair["ensemble"] = "NVT"
            pair["delta_f_kJ_mol"] = pair["delta_f_reduced_kT"] * kt
            pair["delta_f_uncertainty_kJ_mol"] = pair["delta_f_uncertainty_reduced_kT"] * kt
            pairs[local_i] = pair
            if not pair["passed"]:
                all_passed = False
        if all_passed:
            break
        active = set()
        for local_i, pair in enumerate(pairs):
            if not pair["passed"]:
                active.add(local_i)
                active.add(local_i + 1)

    return {
        "pairs": pairs,
        "all_passed": bool(all(p["passed"] for p in pairs)),
        "sample_targets": [int(x) for x in sample_targets],
        "final_sampled_steps": {int(k): int(records[k]["sampled_steps"]) for k in range(K)},
    }


def probe_adjacent_bias_calibration_bank(
    topology,
    common_plus_wca_system_xml: str,
    ibs_wrapper,
    K: int,
    positions,
    box_vectors,
    temperature,
    platform_name: str,
    checkpoint_dir: str,
    stage_type: str,
    window_idx: int,
    global_state_start: int,
    lambda_shield: float,
    burn_in_steps: int = 5000,
    sample_targets: Tuple[int, ...] = (20_000, 40_000, 80_000),
    sample_interval: int = 500,
    min_decorrelated_samples: int = 20,
    max_delta_f_uncertainty_kJ_mol: float = 1.0,
    overlap_threshold: float = 0.03,
    coion_identity: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Per-state, resumable replacement for the old retry loop around
    ``probe_bidirectional_overlap_for_bias_calibration`` (rebuild two fresh
    Simulations, burn a fresh burn-in, and throw away every already-sampled
    frame whenever one edge's decorrelated-sample-count/uncertainty gate
    failed, up to 3 times with doubled sample_steps). Same per-state sharing
    and resumable-checkpoint design as ``probe_adjacent_path_overlap_bank``,
    with the same U_common + WCA_window(lambda_shield) + CV_k dynamics and
    LRC-free energies as the original single-edge bias-calibration probe --
    see that function's docstring for why both of those must never change.

    Growing through ``sample_targets`` tiers replaces the old
    retry-with-doubled-sample_steps loop: an edge only keeps extending its
    two end states while ``_bias_calibration_pair_is_sufficient`` is False,
    and never discards already-sampled frames to do so.
    """
    if K < 2:
        raise ValueError("bias 校准轨迹库至少需要两个态")
    cv_xmls = list(getattr(ibs_wrapper, "_int_cv_force_xmls", []))
    if len(cv_xmls) < K:
        raise RuntimeError("IBS wrapper 缺少 bias 校准轨迹库所需的 CV XML")
    if sample_interval <= 0:
        raise ValueError("bias 校准轨迹库采样间隔必须为正")
    temperature_q = (
        temperature if hasattr(temperature, "value_in_unit") else float(temperature) * unit.kelvin
    )
    kt = (unit.MOLAR_GAS_CONSTANT_R * temperature_q).value_in_unit(unit.kilojoule_per_mole)

    bank_dir = _fixed_h_probe_bank_dir(checkpoint_dir, stage_type, window_idx, "bias_calibration_probe")
    expected_manifest = _build_fixed_h_probe_bank_manifest(
        probe_type="bias_calibration_probe",
        stage_type=stage_type,
        window_idx=window_idx,
        K=K,
        global_state_start=global_state_start,
        common_system_xml=common_plus_wca_system_xml,
        cv_xmls=cv_xmls[:K],
        lambda_shield=lambda_shield,
        temperature_K=temperature_q.value_in_unit(unit.kelvin),
        sample_interval=sample_interval,
        platform_name=platform_name,
        coion_identity=coion_identity,
    )
    if not _fixed_h_probe_bank_manifest_matches(bank_dir, expected_manifest):
        _invalidate_fixed_h_probe_bank(bank_dir)
        _atomic_write_json(_fixed_h_probe_bank_manifest_path(bank_dir), expected_manifest)

    records: Dict[int, Dict[str, Any]] = {}
    for k in range(K):
        record = _load_state_trajectory_record(bank_dir, k, K, sample_interval)
        if record is None:
            record = {"u_cv_kj_mol": None, "volume_nm3": None, "segments": [], "sampled_steps": 0}
        records[k] = record

    evaluator = None
    pairs: List[Optional[Dict[str, Any]]] = [None] * (K - 1)
    active = set(range(K))
    for target in sample_targets:
        states_to_extend = sorted(s for s in active if records[s]["sampled_steps"] < target)
        for k in states_to_extend:
            global_k = int(global_state_start) + int(k)
            sim, needs_burn_in_k, segment_reason = _resume_or_start_state_simulation(
                k=k,
                bank_dir=bank_dir,
                has_prior_segments=bool(records[k]["segments"]),
                topology=topology,
                system_xml=common_plus_wca_system_xml,
                cv_xml=cv_xmls[k],
                # 🔑 [WCA_SHIELD_RETIRED] 壳退役后这里必须是 False：
                # `common_plus_wca_system_xml` 在退役状态下等于无壳的 U_common
                # （见 _serialize_ibs_common_plus_wca_system 的反转断言），
                # 再要求 group 4 就会 fail closed。写成常量表达式而不是硬 False，
                # 这样复活防护壳时本调用点自动跟着回到 True。
                require_group4=not WCA_SHIELD_RETIRED,
                platform_name=platform_name,
                temperature_q=temperature_q,
                positions=positions,
                box_vectors=box_vectors,
                integrator_seed=BIAS_CALIB_INTEGRATOR_SEED_BASE + _FIXED_H_PROBE_SEED_STRIDE * global_k,
                velocity_seed=BIAS_CALIB_VELOCITY_SEED_BASE + _FIXED_H_PROBE_SEED_STRIDE * global_k,
                lambda_shield=lambda_shield,
            )
            try:
                if evaluator is None:
                    evaluator = IBSSampler(
                        sim.context, K, temperature_q,
                        prefix=getattr(ibs_wrapper, "prefix", "abfe"),
                        ibs_wrapper=ibs_wrapper,
                    )
                    # 🔑 同上（见 probe_adjacent_path_overlap_bank）：evaluator
                    # 这条路径上只调用 evaluate_interaction_energies，只依赖
                    # __init__ 里已经建好的独立探针 Context，不再需要这份
                    # 动力学 Context 引用——留着它会让首个态的 dynamics Context
                    # 跟随 evaluator 存活到函数结束，和后续态的 dynamics
                    # Context、探针 Context 一起同时占用显存。
                    evaluator.context = None
                _extend_state_trajectory(
                    sim,
                    records[k],
                    evaluator,
                    target_steps=target,
                    sample_interval=sample_interval,
                    burn_in_steps=burn_in_steps,
                    needs_burn_in=needs_burn_in_k,
                    segment_reason=segment_reason,
                )
                _persist_state_trajectory_record(bank_dir, k, records[k], sim)
            finally:
                # 🔑 同上（见 probe_adjacent_path_overlap_bank）：每个态延长完、
                # 落盘完立刻释放它的动力学 Context，峰值只保持一个动力学
                # Context + 一个 evaluator 探针 Context。
                del sim
                gc.collect()

        all_sufficient = True
        for local_i in range(K - 1):
            pair = _analyze_adjacent_pair(
                records[local_i], records[local_i + 1], local_i, local_i + 1, kt,
                lrc_coeff=None, threshold=overlap_threshold,
            )
            pair["global_edge"] = [int(global_state_start) + local_i, int(global_state_start) + local_i + 1]
            pair["burn_in_steps"] = int(burn_in_steps)
            pair["sample_interval"] = int(sample_interval)
            pair["dynamics_hamiltonian"] = "U_common_plus_wca_window_plus_single_cv_int_no_lrc"
            pair["ensemble"] = "NVT"
            pair["lambda_shield"] = float(lambda_shield)
            pair["delta_f_bias_kJ_mol"] = pair["delta_f_reduced_kT"] * kt
            pair["delta_f_bias_uncertainty_kJ_mol"] = pair["delta_f_uncertainty_reduced_kT"] * kt
            pairs[local_i] = pair
            if not _bias_calibration_pair_is_sufficient(
                pair, min_decorrelated_samples, max_delta_f_uncertainty_kJ_mol
            ):
                all_sufficient = False
        if all_sufficient:
            break
        active = set()
        for local_i, pair in enumerate(pairs):
            if not _bias_calibration_pair_is_sufficient(
                pair, min_decorrelated_samples, max_delta_f_uncertainty_kJ_mol
            ):
                active.add(local_i)
                active.add(local_i + 1)

    return {
        "pairs": pairs,
        "all_sufficient": bool(all(
            _bias_calibration_pair_is_sufficient(p, min_decorrelated_samples, max_delta_f_uncertainty_kJ_mol)
            for p in pairs
        )),
        "lambda_shield": float(lambda_shield),
        "sample_targets": [int(x) for x in sample_targets],
        "final_sampled_steps": {int(k): int(records[k]["sampled_steps"]) for k in range(K)},
    }


