#!/usr/bin/env bash
# CPU-only pre-flight：改完代码、在节点上烧 GPU 之前跑这一条。
#
# 只跑不需要 CUDA 的测试（-m "not needs_gpu"）。当前所有测试都是 CPU 可跑的，
# 所以这条命令等于跑全套；将来若加了真需要 GPU 的测试，给它打 needs_gpu 标记
# 就会被这里自动排除，不必再改这个脚本。
#
#   ./run_offline_tests.sh                     # 全部 CPU 测试
#   ./run_offline_tests.sh -x -q               # 追加任意 pytest 参数
#   ./run_offline_tests.sh test_core_physics_numerics.py
#
# 注意：这台机上首次 `import openmm` 要 60-100 s（/home/ruigengji 是 NFS，
# 有并发 ABFE 作业在写 checkpoint/轨迹时会更久，曾观察到卡在
# rpc_wait_bit_killable 20 分钟以上）。慢不等于挂——先用
#   cat /proc/<pid>/wchan
# 确认是不是 NFS 争用再判断。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAMBA_HOOK="/home/ruigengji/mambaforge/etc/profile.d/mamba.sh"
ENV_NAME="openmm_dev"

# openmm + openmm-ml + mace + mdtraj + pymbar 齐全的那个环境。用 `mamba activate`
# 而不是直接调 env 里的 python：raw-path 调用会跳过 env 的初始化（conda/mamba hook
# 管理的环境变量）。这个 mambaforge 里 `conda activate` 本身是坏的（base 环境的
# conda 缺 boltons/requests），只能用 mamba。
if [[ -f "${MAMBA_HOOK}" ]]; then
    # shellcheck disable=SC1090
    source "${MAMBA_HOOK}"
    mamba activate "${ENV_NAME}"
else
    echo "⚠️  找不到 ${MAMBA_HOOK}，沿用当前已激活的 Python 环境。" >&2
fi

cd "${REPO_ROOT}"

echo "── 环境 ──"
python -c 'import sys; print("python:", sys.version.split()[0], sys.executable)'
python - <<'PY'
for module in ("openmm", "pymbar", "numpy", "scipy"):
    try:
        loaded = __import__(module)
        print(f"{module}: {getattr(loaded, '__version__', '?')}")
    except ImportError as exc:
        print(f"{module}: 未安装 ({exc})")
PY

echo "── pytest（排除 needs_gpu）──"
exec python -m pytest -m "not needs_gpu" "$@"
