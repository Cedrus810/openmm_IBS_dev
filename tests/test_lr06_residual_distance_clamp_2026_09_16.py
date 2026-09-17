"""LR-06 方案 A：插件的 `0.1 Å` 硬门改成训练支撑域下界处的**夹取**。

## 为什么改

`bias_scale` 乘在 Group-1 **整个表达式**外面，而里面就是 `cv_k_int + cv_k_rest`
（配体↔环境软核）。所以 `bias_scale = 0` 不是「关掉偏置」，是**把配体关成鬼影** ——
水直接穿过配体，`r → 0` 是必然。三处会进入这个状态（EM、dt 爬坡、偏置爬坡），
后两处是有意的设计。而 `CustomCVForce` 会求值它的**每一个** CV（与系数是否为 0 无关），
于是插件照样在鬼影几何上跑 K1/K6a，撞上硬门。

「别在鬼影期求值残差」这条路走不通：一天之内找到**三个**求值点（积分器力组、
不带 groups 的 `getState`、`getCollectiveVariableValues`），堵一个冒一个。
⟹ 方案 A **消除失效模式**，而不是安排它不被观察到。

## 为什么下限是 1.5 而不是 0.1

`0.1` 的来历只是防 `1/r` 发散，不是关于模型的陈述。`1.5` 是**训练支撑域下界**：
实测 shipped R1 模型的训练帧（`pre_equilibration.dcd`，200 帧，λ=1，按每帧自己的盒子）
配体↔环境最近距离 **1.517 Å**（水氢）/ **1.524 Å**（带 LJ），中位 1.8 Å。

这条很关键，因为**径向基在小 r 处并不衰减**：16 个中心均匀铺在 `[0, 5] Å`、宽 0.333，
所以 0.1 Å 处基函数值仍有 ~0.96。而落在支撑域下界以下那 5 个中心
（0.333 / 0.667 / 1.0 / 1.333 Å）上的 pair weight 量级与训练充分的**完全一样**
（max|w| 0.30–0.37 vs 0.32–0.39）：**完全活跃，却从未被数据约束过**。
夹在 0.1 Å 等于继续求值它们。

## 真机 A/B（2026-09-16，RTX 2080 Ti，mixed precision）

同一批构型、改动前后两个独立编译的 `.so`、两个独立进程：

| 全局最小 lig–env 距离 | 新旧残差能量 |
|---|---|
| ≥ 1.5 Å | **逐位相同**（20.874434910728507 / 19.078136849613827 / 19.07729951664752） |
| < 1.5 Å | 变了（夹取生效） |
| 0.05 / 0.093 Å | 旧：`fail-closed`；新：有限值。**0.093 Å 正是真机崩点。** |

⚠️ 一致性**不是**按「移动的那个原子到锚点的距离」判的，而是按**全局最小**配体↔环境
距离判的 —— 把一个水原子推到距配体原子 0 为 1.8 Å 处时，它到**另一个**配体原子只有
1.146 Å，照样触发夹取。第一版对照表就是这么看岔过一次。
"""
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins/LocalManyBodyResidual"
CUDA_SRC = PLUGIN / "platforms/cuda/src/CudaLocalManyBodyResidualKernels.cpp"
REF_SRC = PLUGIN / "g1_math_core.h"
LAYOUT = PLUGIN / "r1_model_layout.h"


def test_the_floor_is_the_training_support_bound_not_the_old_singularity_guard():
    text = LAYOUT.read_text()
    m = re.search(r"#define\s+EXP025_R_FLOOR_ANGSTROM\s+([0-9.]+)", text)
    assert m, "EXP025_R_FLOOR_ANGSTROM 不见了"
    floor = float(m.group(1))
    assert floor == pytest.approx(1.5), (
        f"夹取下限变成了 {floor}。它必须是**训练支撑域下界**（shipped R1 实测 1.517/1.524 Å），"
        "不是 EXP025_MIN_DISTANCE_ANGSTROM 那个防 1/r 发散的 0.1。"
        "换模型重训后要一起改，见 r1_model_layout.h 里的 ponytail 注释。"
    )
    # 旧常量必须保留：错误码 2 被 exp026_control_plane_layout.h 的 static_assert 钉着 ABI
    assert "#define EXP025_MIN_DISTANCE_ANGSTROM" in text
    assert "#define EXP025_DEVICE_ERROR_MIN_DISTANCE 2" in text


def test_no_cuda_kernel_still_raises_the_min_distance_error():
    """4 个站点都不再报错。**这是 LR-06 的正题。**"""
    text = CUDA_SRC.read_text()
    raisers = [
        ln for ln in text.split("\n")
        if "exp026SetFirstError" in ln and "EXP025_DEVICE_ERROR_MIN_DISTANCE" in ln
    ]
    assert not raisers, (
        "又有内核在抛 MIN_DISTANCE 了 —— 鬼影期必然产生 <0.1 Å 的构型，"
        f"这会让 outer 臂在窗口 0 必死。命中：{raisers}"
    )


def test_all_four_sites_clamp_and_the_two_halves_do_different_things():
    """能量侧夹取、力侧跳过 —— 两半语义不同，不是同一句话抄四遍。"""
    text = CUDA_SRC.read_text()
    clamp = text.count(
        "if (rAngstrom < (real) EXP025_R_FLOOR_ANGSTROM) rAngstrom = (real) EXP025_R_FLOOR_ANGSTROM;"
    )
    skip = text.count("if (rAngstrom < (real) EXP025_R_FLOOR_ANGSTROM) continue;")
    assert clamp == 2, f"computeQ 侧夹取应有 2 处（K1/K6a），实得 {clamp}"
    assert skip == 2, f"force-scatter 侧跳过应有 2 处（K1/K6a），实得 {skip}"
    # 力侧不许写成夹取：那会让 1/r 在下限内被求值
    # 能量侧不许写成 continue：那会在下限处制造能量跳变、给出冲量


def test_reference_mirrors_cuda_because_g1_g2_assert_parity():
    """Reference 平台经头文件用 g1_math_core.h ⟹ 单边改就是崩溃源。"""
    text = REF_SRC.read_text()
    assert "#include \"r1_model_layout.h\"" in text, (
        "g1_math_core.h 必须复用同一个 EXP025_R_FLOOR_ANGSTROM，不许再抄一份数字"
    )
    assert "throw MathError(\"near-singular pair distance" not in text, (
        "Reference 侧还在抛 —— 它与 CUDA 是同一条规则的两份实现，G1/G2 钉的就是两者一致"
    )
    assert "bool belowFloor;" in text
    assert "if (e.belowFloor) continue;" in text, (
        "Reference 的力循环必须跳过被夹取的边，才与 CUDA force-scatter 一致"
    )


def test_the_plugin_source_identity_gate_was_resynced():
    """改了内核源码就必须同步两处 sha256，否则生产开关直接拒绝启用。"""
    import hashlib
    import json

    actual = hashlib.sha256(CUDA_SRC.read_bytes()).hexdigest()
    py = (ROOT / "local_residual/openmm_plugin.py").read_text()
    m = re.search(r'KNOWN_PLUGIN_SOURCE_SHA256 = \(\s*"([0-9a-f]{64})"', py)
    assert m and m.group(1) == actual, "local_residual/openmm_plugin.py 的 sha256 没跟上"
    # 2026-09-17：出厂那份冻结权重绑的是已收工体系，已移出 `resources/`（见
    # archive/.../WHY_RETIRED.md）。sha 对账仍然有意义——归档那份的插件 sha
    # 必须与源码一致，否则将来有人取回来用就是对着旧内核的权重。
    archived = ROOT / "archive/resources/outer_lambda_local_residual_atenolol_retired/manifest.json"
    if archived.is_file():
        manifest = json.loads(archived.read_text())
        assert manifest["plugin"]["source_sha256"] == actual, "归档 manifest 的 sha256 没跟上"
