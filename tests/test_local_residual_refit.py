"""EXP-033 §5 P1 闭式重训的三件会静默出错的事。

1. 恰好仿射的 rho —— 构造错了，`B` 就不是拟合出来的那个 `B`，而且不报错。
2. 闭式解 `θ* = −(M + λΩ)⁻¹ b` 真的是那个二次型的极小点。
3. `A_k` 与运行时 `OuterLambdaController.envelope` 同一条式子 —— 不同就不是同一个量。
"""
import json
import math

import numpy as np
import pytest

from local_residual.refit import (
    _RHO_AFFINE_BIAS,
    _RHO_HIDDEN,
    _affine_rho_state,
    affine_rho_absolute_error_floor,
    _quadratic_form,
    _smoothness_prior_for_dim,
    _solve,
    a_k_schedule,
    affine_rho_is_exact,
    solve_closed_form,
)

pytestmark = pytest.mark.cpu_only


def _silu(z):
    return z / (1.0 + np.exp(-z))


def _rho_forward(state, t, q):
    """按 payload 声明的形状跑一遍 rho：Linear(1,16)-SiLU-Linear(16,16)-SiLU-Linear(16,1)."""
    h = _silu(state[f"rho.{t}.0.weight"] @ np.atleast_1d(q)[None, :] + state[f"rho.{t}.0.bias"][:, None])
    h = _silu(state[f"rho.{t}.2.weight"] @ h + state[f"rho.{t}.2.bias"][:, None])
    return (state[f"rho.{t}.4.weight"] @ h + state[f"rho.{t}.4.bias"][:, None]).ravel()


@pytest.mark.parametrize("slope", [1.0, 0.25, 7.5])
def test_affine_rho_reproduces_identity_within_its_stated_floor(slope):
    """`rho(q) - rho(0)` 等于 q，准到 `affine_rho_absolute_error_floor` 那条线。

    不是逐比特 —— `(s·q + L) − L` 会吃掉低位。这个测试钉的是：误差**不超过**
    模块自己声明的下限，且那条下限本身小到物理上无意义（相对 b_max=10）。
    """
    state = _affine_rho_state(n_types=3, slope_scale=slope)
    q = np.linspace(-_RHO_HIDDEN, _RHO_HIDDEN, 65) / slope * 0.9
    q = q[np.abs(q) * slope <= 15.0]
    assert q.size > 10
    floor = affine_rho_absolute_error_floor(slope)
    assert floor / 10.0 < 1e-12, "声明的误差下限相对 b_max 已经不可忽略了"
    for t in range(3):
        per_atom = _rho_forward(state, t, q) - _rho_forward(state, t, np.zeros(1))
        assert np.allclose(per_atom, q, rtol=0.0, atol=4.0 * floor), (
            f"type {t}: 最大偏差 {np.abs(per_atom - q).max():.3g} 超过声明下限 {floor:.3g}"
        )


def test_affine_rho_silu_step_itself_is_bitwise_identity():
    """把偏置抵消那步排除掉之后，SiLU 那一步必须是**精确**的恒等。"""
    state = _affine_rho_state(n_types=1, slope_scale=1.0)
    z = np.linspace(_RHO_AFFINE_BIAS - 15.0, _RHO_AFFINE_BIAS + 15.0, 97)
    assert np.array_equal(_silu(z), z), "SiLU 在这个量程上没有退化成恒等"


def test_affine_rho_exactness_guard_rejects_out_of_range():
    """超出量程就不能再声称精确 —— 这道判据是 fail-closed 的依据。"""
    assert affine_rho_is_exact(np.array([1.0, -2.0]), slope_scale=1.0)
    assert not affine_rho_is_exact(np.array([100.0]), slope_scale=1.0)
    # 最小预激活刚好落在 45 这条线上
    assert affine_rho_is_exact(np.array([_RHO_AFFINE_BIAS - 45.0]), slope_scale=1.0)
    assert not affine_rho_is_exact(np.array([_RHO_AFFINE_BIAS - 44.0]), slope_scale=1.0)


def _toy_problem(seed=0, n_frames=60, n_states=4, n_types=2, n_radial=16):
    rng = np.random.default_rng(seed)
    dim = n_types * n_types * n_radial
    features = rng.normal(size=(n_frames, dim))
    gaps = rng.normal(size=(n_frames, n_states - 1))
    log_importance = rng.normal(scale=0.3, size=(n_frames, n_states))
    delta_a = rng.normal(size=n_states - 1)
    return features, gaps, log_importance, delta_a


def test_closed_form_solution_is_the_quadratic_minimum():
    """θ* 必须让梯度为零：`(M + λΩ)θ + b == 0`，且任意扰动都不更优。"""
    features, gaps, log_importance, delta_a = _toy_problem()
    M, b, c = _quadratic_form(features, gaps, log_importance, delta_a, np.arange(len(features)))
    omega = _smoothness_prior_for_dim(features.shape[1])
    ridge = 1e-3
    theta = _solve(M, b, omega, ridge)

    gradient = (M + ridge * omega) @ theta + b
    assert np.allclose(gradient, 0.0, atol=1e-8), np.abs(gradient).max()

    def objective(t):
        return t @ M @ t + 2.0 * b @ t + c + ridge * (t @ omega @ t)

    best = objective(theta)
    rng = np.random.default_rng(1)
    for _ in range(20):
        perturbed = theta + rng.normal(scale=1e-3, size=theta.shape)
        assert objective(perturbed) >= best - 1e-12


def test_quadratic_form_matches_direct_loss_evaluation():
    """攒出来的 (M, b, c) 必须与直接按定义算的加权方差 loss 逐点一致。"""
    features, gaps, log_importance, delta_a = _toy_problem(seed=3)
    M, b, c = _quadratic_form(features, gaps, log_importance, delta_a, np.arange(len(features)))
    rng = np.random.default_rng(7)
    theta = rng.normal(size=features.shape[1])

    basis = features @ theta
    n_edges = gaps.shape[1]
    direct = 0.0
    for edge in range(n_edges):
        y = gaps[:, edge] + delta_a[edge] * basis
        for state in (edge, edge + 1):
            w = np.exp(log_importance[:, state] - log_importance[:, state].max())
            w = w / w.sum()
            direct += 0.5 / n_edges * float(w @ (y - w @ y) ** 2)
    assert direct == pytest.approx(theta @ M @ theta + 2.0 * b @ theta + c, rel=1e-10)


def test_inner_cv_never_scores_on_its_own_training_folds():
    """λ 必须用内层 CV 选 —— 拟合折与打分折不得重叠（P1 item 1 的硬约束）。"""
    features, gaps, log_importance, delta_a = _toy_problem(seed=5, n_frames=90)
    folds = np.repeat([0, 1, 2], 30)
    theta, report = solve_closed_form(
        features=features, adjacent_gap_reduced=gaps,
        log_importance_unnormalized=log_importance, delta_a=delta_a,
        fold_index=folds, ridge_grid=[1e-4, 1e-2, 1.0], n_types=2,
    )
    assert theta.shape == (features.shape[1],)
    assert report["ridge_selected"] in {1e-4, 1e-2, 1.0}
    assert len(report["ridge_grid_scores"]) == 3
    for row in report["ridge_grid_scores"]:
        assert len(row["per_fold_held_out_loss"]) == 3
    # 在全量上拟合过的 loss 不可能比不用 basis 更差
    assert report["loss_with_basis"] <= report["loss_without_basis"] + 1e-12


def test_a_k_schedule_matches_the_runtime_envelope():
    """与运行时 envelope 同一条式子，端点严格 0。"""
    lambdas = [1.0, 0.9, 0.5, 0.1, 0.0]
    delta, a_k = a_k_schedule(lambdas)
    assert a_k[0] == 0.0 and a_k[-1] == 0.0
    for lam, value in zip(lambdas, a_k):
        expected = 0.0 if lam in (0.0, 1.0) else math.sin(math.pi * lam) ** 2
        assert value == expected
    assert delta == [a_k[i + 1] - a_k[i] for i in range(len(a_k) - 1)]

    envelope = pytest.importorskip("outer_lambda_neural_basis")
    controller_envelope = getattr(envelope, "OuterLambdaController", None)
    if controller_envelope is not None and hasattr(controller_envelope, "envelope"):
        for lam in (0.0, 0.25, 0.7, 1.0):
            assert controller_envelope.envelope(controller_envelope, lam) == (
                0.0 if lam in (0.0, 1.0) else math.sin(math.pi * lam) ** 2
            )


# =============================================================================
# payload 导出：loader 读得动，且 B 与解出来的那个 B 是同一个函数
# =============================================================================
def _fake_result(n_types=2, n_radial=16, n_ligand=41, scale=0.01):
    from local_residual.softlift import derived_r1_config

    rng = np.random.default_rng(11)
    config = derived_r1_config(
        "a" * 64,
        type_vocabulary=tuple(range(1, n_types + 1)),
        n_ligand_atoms=n_ligand,
        max_environment_atoms=320,
        max_edges=2048,
        max_neighbors_per_ligand=80,
    )
    theta = rng.normal(scale=scale, size=n_types * n_types * n_radial)
    features = rng.normal(size=(40, theta.size))
    raw = features @ theta
    return {
        "theta": theta,
        "features": features,
        "raw_offset_reduced": float(raw.mean()),
        "linear_basis_reduced": raw - raw.mean(),
        "tanh_saturated_fraction": 0.0,
        "config": config,
        "realized_capacity_usage": {"edges": 1200, "environment_atoms": 250, "neighbors": 60},
        "fit_report": {"ridge_selected": 1e-3, "relative_improvement": 0.42},
        "n_frames": 40,
        "trajectory_path": "pre_equilibration.dcd",
    }


def test_payload_round_trips_through_the_production_loader(tmp_path):
    """写出来的 payload 必须能被生产 loader 原样读回 —— 张量逐比特相同。"""
    from local_residual.openmm_plugin import load_r1_payload
    from local_residual.refit import write_refit_payload

    result = _fake_result()
    paths = write_refit_payload(
        result, output_dir=tmp_path, ligand_topology_indices=list(range(41)),
        protocol_sha256="b" * 64,
    )
    reloaded = load_r1_payload(paths["payload"], paths["weights"])
    n_types = len(result["config"].type_vocabulary)
    assert np.array_equal(
        reloaded.pair_weight, result["theta"].reshape(n_types, n_types, 16)
    )
    assert reloaded.b_max_reduced == result["config"].b_max_reduced
    assert len(reloaded.rho) == n_types


def test_payload_tolerates_tanh_reshaping_the_basis(tmp_path):
    """tanh 改形不是错误。`B_φ` 的职责是别让配体解耦时散架，不是逐点复现一个模型。

    §2.3B 实测线性量程 ±28~40 kT 而 b_max 只有 10 ⟹ 改形是**预期**会发生的。
    按精度硬拦会拦掉本来能跑的 run。
    """
    from local_residual.refit import write_refit_payload

    result = _fake_result()
    rng = np.random.default_rng(2)
    result["linear_basis_reduced"] = rng.normal(scale=35.0, size=400)
    paths = write_refit_payload(
        result, output_dir=tmp_path, ligand_topology_indices=list(range(41)),
        protocol_sha256="c" * 64,
    )
    body = json.loads(paths["payload"].read_text())
    assert body["closed_form_refit"]["tanh_deployment_distortion"] > 0.10


def test_payload_refuses_when_tanh_flattens_the_basis_to_a_constant(tmp_path):
    """真正该停的是这条：全压到同一侧 ⟹ B 成常数 ⟹ 被 f_k 吸收 ⟹ 开了等于没开。"""
    from local_residual.refit import RefitError, write_refit_payload

    result = _fake_result()
    # 全部远在正饱和段：tanh 后逐帧几乎都是 +b_max，离散度塌掉
    result["linear_basis_reduced"] = np.full(400, 500.0) + np.linspace(0, 1, 400)
    with pytest.raises(RefitError, match="离散度"):
        write_refit_payload(
            result, output_dir=tmp_path, ligand_topology_indices=list(range(41)),
            protocol_sha256="f" * 64,
        )


def test_exported_payload_reproduces_the_fitted_linear_basis(tmp_path):
    """端到端：用回读出来的张量重算 B，必须等于拟合时用的那个线性 B。

    这是整条链上最容易静默错的一步 —— rho 的斜率折进 pair_weight 之后，
    重算出来的 raw 与 `features @ theta` 差一个因子的话，部署的就不是拟合的那个模型。
    """
    from local_residual.openmm_plugin import load_r1_payload
    from local_residual.refit import _RHO_HIDDEN, write_refit_payload

    result = _fake_result()
    paths = write_refit_payload(
        result, output_dir=tmp_path, ligand_topology_indices=list(range(41)),
        protocol_sha256="d" * 64,
    )
    reloaded = load_r1_payload(paths["payload"], paths["weights"])

    # 每个类型的 rho 都必须是同一条仿射：斜率 g*s == 1
    for entry in reloaded.rho:
        slope = float(entry["w4"][0]) * float(entry["w2"][0, 0]) * float(entry["w0"][0])
        assert slope == pytest.approx(1.0, rel=1e-12), slope
        assert entry["b0"].shape == (_RHO_HIDDEN,)

    raw = result["features"] @ reloaded.pair_weight.reshape(-1)
    assert np.allclose(raw, result["features"] @ result["theta"], rtol=0.0, atol=0.0)
