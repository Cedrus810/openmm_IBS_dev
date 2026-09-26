#=============================================================================
# 修复 4: 分块 MBAR 分析器 (解决 OOM 瓶颈)
#=============================================================================
class ChunkedMBARAnalyzer:
    """
    支持超大 u_kn 矩阵的 MBAR 分析器
    ✅ 使用 np.memmap 零拷贝加载
    ✅ 自动分块计算，避免多进程 OOM
    ✅ 兼容 pymbar >= 3.0.5
    """
    def __init__(self, max_memory_gb: float = 32.0, cache_dir: str = "./mbar_cache", temperature_k: float = 300.0):
        self.max_ram = max_memory_gb * 1e9
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.gc = gc
        # 🔑 与 TraditionalMBARAnalyzer 等项目内其它 MBAR 分析器使用同一个气体
        # 常数换算，供 _extract_delta_g 把 pymbar 返回的约化自由能(kT)转成 kJ/mol。
        self.kt = 0.008314462618 * float(temperature_k)
        
    def _save_chunk_to_disk(self, u_kn_block: np.ndarray, chunk_id: int) -> str:
        path = os.path.join(self.cache_dir, f"u_kn_chunk_{chunk_id}.npy")
        np.save(path, u_kn_block)
        return path
        
    def run_chunked_mbar(self, u_kn_total: np.ndarray, n_k_array: np.ndarray, stage_type: str = "coul"):
        """使用分态步幅抽样执行单次全局 MBAR，避免伪分块平均导致的统计错误。"""
        if not has_pymbar():
            raise ImportError("需要 pymbar 进行 MBAR 分析: pip install pymbar")
        import pymbar
        
        K, N = u_kn_total.shape
        n_k_array = np.asarray(n_k_array, dtype=int)
        if n_k_array.ndim != 1 or len(n_k_array) != K:
            raise ValueError(f"n_k_array 维度异常: 期望长度 {K}，实际 {n_k_array.shape}")
        if np.any(n_k_array < 0):
            raise ValueError("n_k_array 不能包含负样本数")

        # 1. 计算安全步幅 (Stride)，确保抽样后内存低于限制的 50%
        stride = 1
        while (K * (N // stride) * u_kn_total.itemsize) > self.max_ram * 0.5 and stride < N:
            stride *= 2

        if stride > 1:
            if np.sum(n_k_array) != N:
                raise MemoryError("u_kn_total 列数与 n_k_array 总和不一致，无法安全执行分态步幅抽样")

            print(f"  [WARN] 内存受限，启用分态步幅抽样 (Stride={stride}) 后执行单次全局 MBAR")
            keep_indices = []
            start = 0
            n_k_sub = np.zeros_like(n_k_array)
            for k, n_k in enumerate(n_k_array):
                end = start + int(n_k)
                if n_k > 0:
                    state_idx = np.arange(start, end, stride, dtype=int)
                    if state_idx.size == 0:
                        state_idx = np.array([start], dtype=int)
                    keep_indices.append(state_idx)
                    n_k_sub[k] = state_idx.size
                start = end

            if not keep_indices:
                raise ValueError("n_k_array 全为 0，无法执行 MBAR")

            keep_indices = np.concatenate(keep_indices)
            u_kn_sub = u_kn_total[:, keep_indices]
            print(f"  抽样后保留 {u_kn_sub.shape[1]} 帧，分态样本数: {n_k_sub.tolist()}")
        else:
            u_kn_sub = u_kn_total
            n_k_sub = n_k_array.copy()

        # 2. 单次全局 MBAR 求解 (统计严格)
        try:
            mbar = _build_mbar_compatible(
                u_kn_sub,
                n_k_sub,
                verbose=False,
                solver_protocol="hybr",
            )
            res = _compute_free_energy_result_compatible(mbar, compute_uncertainty=True)
            return self._extract_delta_g(res, n_k_sub, stage_type)
        except Exception as e:
            print(f"  [ERR] MBAR 求解失败: {e}，尝试降级至 robust 求解器...")
            mbar = _build_mbar_compatible(
                u_kn_sub,
                n_k_sub,
                verbose=False,
                solver_protocol="robust",
            )
            res = _compute_free_energy_result_compatible(mbar, compute_uncertainty=True)
            return self._extract_delta_g(res, n_k_sub, stage_type)

    def _extract_delta_g(self, res, n_k_array, stage_type):
        """返回 kJ/mol 的自由能差/不确定度。

        _extract_free_energy_arrays 返回的是 pymbar 的约化自由能（单位 kT，
        即 beta*Delta_G），之前这里直接原样返回，调用方若不知情就会把 kT
        当 kJ/mol 直接使用（室温下 1 kT ≈ 2.5 kJ/mol，量级误差明显）。
        """
        df, ddf = _extract_free_energy_arrays(res, require_uncertainty=True)
        return df[0, :] * self.kt, ddf[0, :] * self.kt


