def _infer_log_level_from_message(message: str) -> int:
    if any(token in message for token in ("[WARN]", "警告", "warning")):
        return logging.WARNING
    if any(token in message for token in ("[ERR]", "失败", "错误", "异常", "error")):
        return logging.ERROR
    return logging.INFO


def _log_print(*args, sep=" ", end="\n", file=None, flush=False):
    message = sep.join(str(arg) for arg in args)
    if end and end != "\n":
        message += end.rstrip("\n")
    if logger.handlers:
        logger.log(_infer_log_level_from_message(message), message)
    builtins.print(*args, sep=sep, end=end, file=file, flush=flush)


_ONCE_EMITTED_KEYS: set = set()


def print_once(key: str, *args, **kwargs) -> None:
    """同一份日志文件里，同一个 `key` 只打印一次。

    `ibs_engine.print_once` 的同源副本——`_log_print` /
    `_infer_log_level_from_message` 在本仓库就是这样逐模块复制的（避免 abfe_core 与
    ibs_engine 之间新增 import 耦合），这里沿用同一模式。

    去重键里带上当前 tee 的目标文件（`_StdoutTeeToFile._log_path`）：同进程跑第二条腿、
    日志 retarget 到新的 `pipeline.log` 时，这段说明会在新文件里重新出现一次。
    """
    scope = getattr(sys.stdout, "_log_path", None)
    dedup_key = (scope, key)
    if dedup_key in _ONCE_EMITTED_KEYS:
        return
    _ONCE_EMITTED_KEYS.add(dedup_key)
    print(*args, **kwargs)
def _infer_log_level_from_message(message: str) -> int:
    if any(token in message for token in ("[WARN]", "警告", "warning")):
        return logging.WARNING
    if any(token in message for token in ("[ERR]", "失败", "错误", "异常", "error")):
        return logging.ERROR
    return logging.INFO


def _log_print(*args, sep=" ", end="\n", file=None, flush=False):
    message = sep.join(str(arg) for arg in args)
    if end and end != "\n":
        message += end.rstrip("\n")
    if logger.handlers:
        logger.log(_infer_log_level_from_message(message), message)
    builtins.print(*args, sep=sep, end=end, file=file, flush=flush)


_ONCE_EMITTED_KEYS: set = set()


def print_once(key: str, *args, **kwargs) -> None:
    """同一份日志文件里，同一个 `key` 只打印一次。

    专门给"纯静态说理"类输出用：内容跟运行时状态无关，但打印点在热路径上。例如
    `_create_softcore_force` 每建一次软核力就调一次，4W53 的一次 run 里同一段 460 字
    的理由重复了 257 次、占了 `pipeline.log` 体积的 31.5%。

    去重键里带上当前 tee 的目标文件（`_StdoutTeeToFile._log_path`）：同进程跑第二条腿、
    日志 retarget 到新的 `pipeline.log` 时，这段说明会在新文件里重新出现一次，而不是
    因为"进程内已经打过"就让整条腿的日志都查不到它。tee 没装时 scope 为 None，退化成
    进程级去重。
    """
    scope = getattr(sys.stdout, "_log_path", None)
    dedup_key = (scope, key)
    if dedup_key in _ONCE_EMITTED_KEYS:
        return
    _ONCE_EMITTED_KEYS.add(dedup_key)
    print(*args, **kwargs)


def _atomic_save_npy(filepath: str, array: np.ndarray) -> None:
    """同步原子写 NPY：先写临时文件，再原子替换。"""
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    tmp_file = filepath + ".tmp"
    with open(tmp_file, "wb") as handle:
        np.save(handle, array)
    os.replace(tmp_file, filepath)


def _sha256_file(filepath: str) -> str:
    digest = hashlib.sha256()
    with open(filepath, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _timed(bucket: Dict[str, float], key: str):
    """把一段代码的墙钟耗时累加进 `bucket[key]`（性能分档计时用）。

    `time.perf_counter()` 单次开销约 50-100ns，控制面每 500 步一次的循环里
    加这一层可忽略不计，因此默认常开——目的是把"积分 vs guard vs CV/probe
    vs ledger I/O"的耗时占比变成可测量的数字，而不是靠猜测决定优化方向。
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        bucket[key] = bucket.get(key, 0.0) + (time.perf_counter() - t0)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _infer_log_level_from_message(message: str) -> int:
    if any(token in message for token in ("[WARN]", "警告", "warning")):
        return logging.WARNING
    if any(token in message for token in ("[ERR]", "失败", "错误", "异常", "error")):
        return logging.ERROR
    return logging.INFO


def _log_print(*args, sep=" ", end="\n", file=None, flush=False):
    message = sep.join(str(arg) for arg in args)
    if end and end != "\n":
        message += end.rstrip("\n")
    if logger.handlers:
        logger.log(_infer_log_level_from_message(message), message)
    builtins.print(*args, sep=sep, end=end, file=file, flush=flush)


def _file_sha256(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return None


log = logging.getLogger("runabfe")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


#============================================================================
# 辅助类：NumpyEncoder (JSON 序列化支持)
#============================================================================
class NumpyEncoder(json.JSONEncoder):
    """🔑 支持 numpy 数组/类型的 JSON 编码器"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class NumpyEncoder(json.JSONEncoder):
    """🔑 全局统一 JSON 序列化器（处理 numpy 类型/数组）"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)): return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_,)): return bool(obj)
        return super().default(obj)
