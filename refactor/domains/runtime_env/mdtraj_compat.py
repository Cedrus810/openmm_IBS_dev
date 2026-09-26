

# ============================================================================
# mdtraj 内置 VMD `dcdplugin` 的 C 层噪声
# ============================================================================
# 每打开一个 DCD，mdtraj 的 `dcd.cpython-*.so`（VMD dcdplugin.c）都会用 **C 的
# printf 直接写 fd 1** 刷两行：
#     dcdplugin) detected standard 32-bit DCD file of native endianness
#     dcdplugin) CHARMM format DCD file (also NAMD 2.1 and later)
# 这两行绕过 `sys.stdout`，所以 `contextlib.redirect_stdout`、`warnings` 和 logging
# 一律拦不住；`DCDTrajectoryFile` 是 Cython 扩展类型，也没法直接 monkeypatch 它的
# `__init__`（扩展类型不允许设属性）。唯一的拦截点在 fd 层。
#
# ⚠️ 关键：**不能**把 fd 1 一律重定向到 /dev/null。同一个 printf 通道里还会出来
#     dcdplugin) unrecognized DCD header:
#     dcdplugin) read_dcdheader: corruption or unrecognized file structure
# 那是 DCD 截断/损坏的真实诊断（本仓库的 DCD append 出过这类事故，见
# `_is_traj_valid` 的 fail-closed 说明），吞掉它等于把故障线索删了。所以这里做的是
# **逐行过滤**：只丢下面这张白名单里的信息性行，其余原样保留。
_C_STDOUT_DROP_PREFIXES: Tuple[str, ...] = (
    "dcdplugin) detected ",
    "dcdplugin) CHARMM format DCD file",
)


_LIBC_FFLUSH: Any = None


def _c_fflush_all() -> None:
    """强制 C 层 stdio 落盘（`fflush(NULL)` 刷所有输出流）。

    ⚠️ 少了这一步，`_filter_c_stdout` 会**丢输出**，而且是静默丢：fd 1 一旦被重定向
    到普通文件，glibc 就把 stdout 从行缓冲切成**全缓冲**，dcdplugin 的 printf 留在
    C 的缓冲区里而不是落到那个临时文件；恢复 fd 1 再去读临时文件就读到空的，连
    `dcdplugin) read_dcdheader: corruption or unrecognized file structure` 这种真实
    故障诊断都会一起消失（实测确认过）。

    拿不到 libc 时降级成 no-op：宁可留着噪声，也不能让日志装饰打断计算。
    """
    global _LIBC_FFLUSH
    if _LIBC_FFLUSH is None:
        try:
            import ctypes

            fflush = ctypes.CDLL(None).fflush
            fflush.argtypes = [ctypes.c_void_p]
            fflush.restype = ctypes.c_int
            _LIBC_FFLUSH = fflush
        except Exception:
            _LIBC_FFLUSH = False
    if _LIBC_FFLUSH:
        try:
            _LIBC_FFLUSH(None)
        except Exception:
            pass


@contextlib.contextmanager
def _filter_c_stdout(drop_prefixes: Tuple[str, ...]):
    """在 fd 层捕获 C 库直写 fd 1 的输出，丢掉 `drop_prefixes` 命中的行，其余重发。

    保留下来的行改用 `print()` 重发，于是它们**第一次**进了 `pipeline.log`——原来
    C 层输出完全绕过 `_StdoutTeeToFile`，只出现在终端里，归档日志查不到。

    用临时文件而不是管道做中转：管道有 64 KiB 容量上限，一个损坏的 DCD 可能刷出
    大量行并把 C 侧的写阻塞死。

    ⚠️ 作用域必须只包住第三方那一次调用。块内如果有 Python 侧 `print()`，它会被
    一起捕获再重发，从而在 `pipeline.log` 里出现两遍（tee 已经写过一次）。

    fd 1 拿不到（已被关闭等）时直接放行、不做任何拦截：日志装饰绝不能反过来打断计算。
    """
    try:
        sys.stdout.flush()
    except Exception:
        pass
    try:
        saved_fd = os.dup(1)
    except OSError:
        yield
        return

    captured = b""
    tmp = tempfile.TemporaryFile()
    try:
        os.dup2(tmp.fileno(), 1)
        try:
            yield
        finally:
            # 必须在恢复 fd 1 **之前**刷 C 缓冲，否则那些字节根本没进 tmp。
            _c_fflush_all()
            os.dup2(saved_fd, 1)
            try:
                tmp.seek(0)
                captured = tmp.read()
            except OSError:
                captured = b""
    finally:
        os.close(saved_fd)
        tmp.close()
        # 重发必须放在 `finally` 里。写在函数末尾会有一个致命的失效模式：块内抛异常
        # （损坏 DCD 让 ctor 抛 OSError 正是如此）时，异常从 `yield` 处向外传播，末尾
        # 的循环永远不执行——于是恰好在最需要看到诊断的那一刻把诊断丢干净。实测确认过。
        for line in captured.decode("utf-8", errors="replace").splitlines():
            if not line.startswith(drop_prefixes):
                print(line)


_MDTRAJ_DCD_QUIET_PATCHED = False


def _install_mdtraj_dcd_quiet_patch() -> bool:
    """给 mdtraj 的载入入口包一层 `_filter_c_stdout`。幂等，失败即放弃。

    打在 **mdtraj 模块命名空间**上而不是逐个改调用点：本仓库 5 个模块里约 25 处
    DCD 读取全都写成 `md.load(...)` 这种属性查找（已确认没有 `from mdtraj import
    load`），所以换掉模块属性就能一次覆盖全部，也覆盖以后新增的调用点。补丁是
    纯输出装饰，不改任何返回值或异常语义。
    """
    global _MDTRAJ_DCD_QUIET_PATCHED
    if _MDTRAJ_DCD_QUIET_PATCHED or not HAS_MDTRAJ:
        return False

    def _wrap_call(fn):
        @functools.wraps(fn)
        def _quiet(*args, **kwargs):
            with _filter_c_stdout(_C_STDOUT_DROP_PREFIXES):
                return fn(*args, **kwargs)
        _quiet._abfe_dcd_quiet = True
        return _quiet

    def _wrap_generator(fn):
        # `yield` 必须在 with **外面**：否则调用方消费 chunk 期间 fd 1 还被捕获着，
        # 会把调用方自己的输出吞进来再重发一遍。
        @functools.wraps(fn)
        def _quiet(*args, **kwargs):
            with _filter_c_stdout(_C_STDOUT_DROP_PREFIXES):
                iterator = fn(*args, **kwargs)
            while True:
                with _filter_c_stdout(_C_STDOUT_DROP_PREFIXES):
                    try:
                        chunk = next(iterator)
                    except StopIteration:
                        return
                yield chunk
        _quiet._abfe_dcd_quiet = True
        return _quiet

    patched_any = False
    for owner, attr, wrapper in (
        (mdtraj, "load", _wrap_call),
        (mdtraj, "load_dcd", _wrap_call),
        (mdtraj, "load_frame", _wrap_call),
        (mdtraj, "iterload", _wrap_generator),
        (getattr(mdtraj, "formats", None), "DCDTrajectoryFile", _wrap_call),
    ):
        if owner is None:
            continue
        original = getattr(owner, attr, None)
        if original is None or getattr(original, "_abfe_dcd_quiet", False):
            continue
        try:
            setattr(owner, attr, wrapper(original))
        except (AttributeError, TypeError):
            # 扩展模块可能拒绝改属性；静默跳过，日志噪声不值得让 import 失败。
            continue
        patched_any = True

    _MDTRAJ_DCD_QUIET_PATCHED = True
    return patched_any


_install_mdtraj_dcd_quiet_patch()


def _pymbar_version_tuple() -> Tuple[int, ...]:
    if not has_pymbar():
        return (0,)
    pymbar_module = _require_pymbar()
    version_str = str(
        getattr(pymbar_module, "__version__", getattr(pymbar_module, "version", "0"))
    )
    parts = []
    for token in version_str.replace("-", ".").split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits:
            parts.append(int(digits))
        else:
            break
    return tuple(parts) if parts else (0,)


def _build_mbar_compatible(u_kn, n_k, **kwargs):
    """
    兼容 PyMBAR 3.x/4.x 的 MBAR 构造器。
    若某些关键字参数在当前版本不可用，则按保守顺序回退。
    """
    if not has_pymbar():
        raise ImportError("需要 pymbar 包，请安装: pip install pymbar")

    base_kwargs = dict(kwargs)
    drop_order = [
        "solver_protocol",
        "initialize",
        "relative_tolerance",
        "solver_tolerance",
        "initial_f_k",
        "verbose",
    ]
    variants = [base_kwargs]
    seen = {tuple(sorted((k, repr(v)) for k, v in base_kwargs.items()))}
    current = dict(base_kwargs)
    for key in drop_order:
        if key in current:
            current = dict(current)
            current.pop(key, None)
            signature = tuple(sorted((k, repr(v)) for k, v in current.items()))
            if signature not in seen:
                variants.append(current)
                seen.add(signature)

    last_type_error = None
    for candidate in variants:
        try:
            return _require_pymbar().MBAR(u_kn, n_k, **candidate)
        except TypeError as exc:
            last_type_error = exc
            continue
    if last_type_error is not None:
        raise last_type_error
    return _require_pymbar().MBAR(u_kn, n_k)


def _extract_mbar_matrix(result, primary_name: str, fallback_names: Tuple[str, ...]) -> Optional[np.ndarray]:
    candidate_names = (primary_name,) + tuple(fallback_names)
    for name in candidate_names:
        if isinstance(result, dict) and name in result:
            return np.asarray(result[name], dtype=float)
        if hasattr(result, name):
            return np.asarray(getattr(result, name), dtype=float)
    return None


def _compute_free_energy_result_compatible(mbar, compute_uncertainty: bool = True):
    methods = []
    if hasattr(mbar, "compute_free_energy_differences"):
        methods.append(("compute_free_energy_differences", {"compute_uncertainty": compute_uncertainty}))
    if hasattr(mbar, "compute_free_energy"):
        methods.append(("compute_free_energy", {}))
    if not methods:
        raise AttributeError("当前 pymbar.MBAR 对象不包含可用的自由能计算方法")

    last_exc = None
    for method_name, kwargs in methods:
        try:
            return getattr(mbar, method_name)(**kwargs)
        except TypeError:
            try:
                return getattr(mbar, method_name)()
            except Exception as exc:
                last_exc = exc
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("MBAR 自由能计算失败")


def _extract_free_energy_arrays(result, require_uncertainty: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    delta_f = _extract_mbar_matrix(result, "Delta_f", ("delta_f", "free_energy"))
    if delta_f is None:
        raise KeyError("无法从 pymbar 结果中提取自由能矩阵")

    delta_df = _extract_mbar_matrix(result, "dDelta_f", ("d_delta_f", "error", "uncertainty"))
    if delta_df is None:
        if require_uncertainty:
            raise KeyError("无法从 pymbar 结果中提取不确定度矩阵")
        delta_df = np.full_like(delta_f, np.nan, dtype=float)
    return delta_f, delta_df


def _compute_overlap_matrix_compatible(mbar) -> np.ndarray:
    """兼容层：pymbar.MBAR.compute_overlap() 的返回形状。

    ibs_engine.py 里同类调用点（如 TraditionalMBARAnalyzer.solve()）已经
    对"返回 dict（含 'matrix' 键）还是直接返回 ndarray"做了兼容判断；这里
    补上同样的分支，并加一道有限性校验（不让 NaN/Inf 静默流入调用方的
    收敛/重叠判据），而不是像之前那样无条件假设一定是 dict、也从不校验数值。
    """
    overlap_res = mbar.compute_overlap()
    overlap_matrix = overlap_res["matrix"] if isinstance(overlap_res, dict) else overlap_res
    overlap_matrix = np.asarray(overlap_matrix, dtype=float)
    if not np.all(np.isfinite(overlap_matrix)):
        raise RuntimeError(f"PyMBAR compute_overlap 返回非有限矩阵: {overlap_matrix}")
    return overlap_matrix


def _compute_effective_sample_number_compatible(mbar) -> np.ndarray:
    """兼容层：同 _compute_overlap_matrix_compatible，对
    compute_effective_sample_number() 的返回值做统一的有限性校验。
    """
    neff = np.asarray(mbar.compute_effective_sample_number(), dtype=float)
    if not np.all(np.isfinite(neff)):
        raise RuntimeError(f"PyMBAR compute_effective_sample_number 返回非有限值: {neff}")
    return neff


def subsample_series_by_autocorrelation(
    series: np.ndarray,
    min_frames_for_subsampling: int = 20,
) -> Tuple[np.ndarray, float]:
    """
    对单一状态自身的（约化）能量时间序列估计统计非效率 g，返回近似去相关的帧索引。

    MBAR/BAR 假设逐帧样本互相独立；直接把每一帧原始 MD 轨迹都当独立样本喂给
    MBAR 会让 n_k/有效样本数虚高，报告的误差棒因此系统性偏小（常见 2-10 倍）。
    这里用该状态"自己产生"的能量时间序列估计关联时间的代理量 g，再用 pymbar
    的分块子采样取出近似独立的帧索引；调用方需要用同一组索引去裁剪该状态对应
    的所有相关数组，以保持帧与帧之间的对应关系。

    样本过少（< min_frames_for_subsampling）、序列本身没有涨落（比如恒定值）、
    或 pymbar 不可用时，原样返回全部索引、g=1.0（即不子采样）——对极短序列
    强行估计 g 噪声本身就很大，不如不做。

    ## 两条原来 fail-open、现在 fail-closed 的路径（2026-09-09）

    上面 docstring 自己写明了后果："n_k/有效样本数虚高，报告的误差棒因此系统性
    偏小（常见 2-10 倍）"。而原实现有两处会**静默**造成正是这个后果：

    1. 序列里出现非有限值 → 原来和"序列太短"走同一个 `return full_indices, 1.0`。
       那等于**既把 NaN 帧原样交给 MBAR，又顺手关掉去相关**。非有限的约化能量
       是上游 frame_finite 门应该已经拦住的东西，走到这里就是缺陷，不是"这段
       序列恰好不相关"。
    2. `except Exception` 把 pymbar 的真实失败也吞成同一个返回值。

    这两种情况在输出上与"序列真的不相关"**完全无法区分**，而它们出现在生产路径上
    （`ibs_engine` 的三处 MBAR 组装都调它）。现在都抛。
    "太短 / 无涨落 / 没装 pymbar" 仍然按原样放行——那三种是真的不该子采样。
    """
    series = np.asarray(series, dtype=np.float64)
    n = series.shape[0]
    full_indices = np.arange(n)
    if not has_pymbar() or n < min_frames_for_subsampling:
        return full_indices, 1.0
    if not np.all(np.isfinite(series)):
        n_bad = int(np.count_nonzero(~np.isfinite(series)))
        raise RuntimeError(
            f"去相关的输入序列里有 {n_bad}/{n} 个非有限值。原来这里会静默返回"
            "全部索引 + g=1.0，等于把 NaN 帧交给 MBAR 的同时关掉去相关，"
            "报告的误差棒会系统性偏小 2-10 倍且无从察觉。fail closed。"
        )
    if np.std(series) < 1e-12:
        return full_indices, 1.0
    try:
        from pymbar import timeseries
        g = float(timeseries.statistical_inefficiency(series, fast=True))
        if not np.isfinite(g) or g < 1.0:
            g = 1.0
        indices = np.asarray(timeseries.subsample_correlated_data(series, g=g), dtype=int)
        if indices.size < 2:
            return full_indices, g
        return indices, g
    except Exception as exc:
        raise RuntimeError(
            f"pymbar 去相关失败（{type(exc).__name__}: {exc}）。原来这里会静默退回"
            "全部索引 + g=1.0，与'序列真的不相关'在输出上完全无法区分，"
            "而后果是误差棒偏小 2-10 倍。fail closed。"
        ) from exc


def get_optimal_device_settings():
    if not has_orb():
        return "cpu", False
    torch = _require_torch()
    if not torch.cuda.is_available():
        return "cpu", False
    device = "cuda"
    major, minor = torch.cuda.get_device_capability()
    support_tf32 = False
    if major >= 8:
        support_tf32 = True
        torch.set_float32_matmul_precision("high")
    return device, support_tf32


# 🔑 [ATT-04] 这里原本是模块级的
#     GLOBAL_DEVICE, SUPPORTS_TF32 = get_optimal_device_settings()
# 也就是 **import 期** 就会调 `torch.cuda.is_available()` /
# `torch.cuda.get_device_capability()`（惰性建 CUDA context）并
# `torch.set_float32_matmul_precision("high")`（改全局 torch 状态）。
#
# 并行 stage worker 用 `mp.get_context("spawn")`（abfe_pipeline.py、ibs_engine.py），
# spawn 反序列化 target 时必然 import `abfe_pipeline → ibs_engine → abfe_core`，
# 于是每个子进程都在 import 期就抓一次 CUDA。更糟的是子进程的 GPU 归属只通过
# OpenMM 的 `props["DeviceIndex"]` 表达、从不设 `CUDA_VISIBLE_DEVICES`，所以双 GPU
# 并行时**两个**子进程都会在 device 0 上建 torch context，然后才各自去用被分配的
# OpenMM 设备。
#
# 注意：OpenMM 侧本来就没有 import 期副作用——abfe_core/ibs_engine/abfe_pipeline 里
# 所有 `Platform.getPlatformByName` / `Context(...)` 都在函数内。要消除的只有 torch
# 这一处。
#
# 改成惰性 memoized：真正需要 device 的只有下面两个 MACE/ML 入口的默认参数，
# 它们改成 `device=None` 后在函数体里解析。
_DEVICE_SETTINGS_CACHE = None


def _resolve_device_settings():
    """首次真正需要时才探测设备；结果缓存，语义与旧的模块级求值一致。"""
    global _DEVICE_SETTINGS_CACHE
    if _DEVICE_SETTINGS_CACHE is None:
        _DEVICE_SETTINGS_CACHE = get_optimal_device_settings()
    return _DEVICE_SETTINGS_CACHE


def get_global_device() -> str:
    return _resolve_device_settings()[0]


def supports_tf32() -> bool:
    return _resolve_device_settings()[1]
