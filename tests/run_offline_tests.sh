#!/usr/bin/env bash
# CPU-only pre-flight：改完代码、在节点上烧 GPU 之前跑这一条。
#
# pytest 配置在**仓库根目录**的 pytest.ini（不在 tests/ 下）。本脚本 cd 到仓库根
# 之后不带路径参数跑 pytest，而 pytest 只从 cwd 向上找配置文件——配置放在 tests/ 里
# 会对这条命令整体静默失效（2026-07-30 修，详见 pytest.ini 顶部注释）。
#
# 只跑不需要 CUDA 的测试（-m "not needs_gpu"）。当前所有测试都是 CPU 可跑的，
# 所以这条命令等于跑全套；将来若加了真需要 GPU 的测试，给它打 needs_gpu 标记
# 就会被这里自动排除，不必再改这个脚本。
#
#   ./tests/run_offline_tests.sh                     # 全部 CPU 测试
#   ./tests/run_offline_tests.sh -x -q               # 追加任意 pytest 参数
#   ./tests/run_offline_tests.sh tests/test_core_physics_numerics.py
#
# 注意：这台机上首次 `import openmm` 要 60-100 s（/home/ruigengji 是 NFS，
# 有并发 ABFE 作业在写 checkpoint/轨迹时会更久，曾观察到卡在
# rpc_wait_bit_killable 20 分钟以上）。慢不等于挂——先用
#   cat /proc/<pid>/wchan
# 确认是不是 NFS 争用再判断。

# 🔑 注意这里**不能**带 `-u`（nounset）。原因不是 mamba hook 本身，而是这个 env
# 自己的 activate 脚本：
#   openmm_dev/etc/conda/activate.d/env_vars.sh:2
#       export CPATH=$CONDA_PREFIX/include:$CPATH
# `$CPATH` 在干净 shell 里没有定义，`-u` 下这一行直接
#   "CPATH: 未绑定的变量"
# 中止激活，脚本还没跑到 pytest 就退出了——表现成"测试入口失败"，
# 但其实一条测试都没跑。（`LIBRARY_PATH`/`LD_LIBRARY_PATH` 同理。）
#
# 所以：激活阶段不开 -u，激活完成后再开，让我们自己的代码仍受 nounset 保护。
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 🔑 [2026-08-31] env 已从 mambaforge 搬到 miniforge3。旧的写死路径
# /home/ruigengji/mambaforge/etc/profile.d/mamba.sh 文件还在，但那个安装里
# 已经没有 envs/ 目录，source 之后 MAMBA_EXE 为空 → `mamba` 不是命令 →
# `set -e` 让脚本在**一条测试都没跑**的情况下 exit 127。表现成"测试入口失败"，
# 而不是"测试失败"，两者非常容易看混。
# 现在按顺序探测候选安装，取第一个**真的含有目标 env** 的那个；
# 一个都没有时才退回"沿用当前已激活环境"。
MAMBA_HOOK=""
for _root in /home/ruigengji/miniforge3 /home/ruigengji/mambaforge; do
    if [[ -d "${_root}/envs/openmm_dev" && -f "${_root}/etc/profile.d/mamba.sh" ]]; then
        MAMBA_HOOK="${_root}/etc/profile.d/mamba.sh"
        break
    fi
done
ENV_NAME="openmm_dev"

# openmm + openmm-ml + mace + mdtraj + pymbar 齐全的那个环境。用 `mamba activate`
# 而不是直接调 env 里的 python：raw-path 调用会跳过 env 的初始化（conda/mamba hook
# 管理的环境变量）。这个 mambaforge 里 `conda activate` 本身是坏的（base 环境的
# conda 缺 boltons/requests），只能用 mamba。
if [[ -f "${MAMBA_HOOK}" ]]; then
    # shellcheck disable=SC1090
    source "${MAMBA_HOOK}"
    mamba activate "${ENV_NAME}"

    # 激活可能静默半成功（hook 在，env 名打错/env 损坏）。不核对的话下面会用
    # 系统 python 跑，import openmm 失败又被读成"测试挂了"。宁可在这里响亮地死。
    ACTIVE_PY="$(command -v python || true)"
    case "${ACTIVE_PY}" in
        *"/envs/${ENV_NAME}/bin/python") ;;
        *)
            echo "❌ mamba activate ${ENV_NAME} 之后 python 仍是 '${ACTIVE_PY:-<none>}'，" >&2
            echo "   不在 envs/${ENV_NAME} 里。拒绝用错误的解释器跑测试。" >&2
            exit 1
            ;;
    esac
else
    echo "⚠️  找不到 ${MAMBA_HOOK}，沿用当前已激活的 Python 环境。" >&2
fi

# 激活完成，从这里开始恢复 nounset。
set -u

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
