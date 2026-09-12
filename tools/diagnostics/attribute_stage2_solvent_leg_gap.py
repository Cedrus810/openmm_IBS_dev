#!/usr/bin/env python
"""溶剂腿 stage2 生产结果 vs 独立参考真值：口径对齐 + 逐窗口归因。

零 GPU，只读产物。

两个必须先做、否则结论会反的口径对齐
------------------------------------
1. **LRC**：生产 stage2 含 LJ 解析尾项，`docs/reference_data/*.json` 的真值不含。
   本脚本**不重算**尾项，而是从产物里直接读：
   `energies.npy - sampling_states.npy.T` 是**逐 λ 态的严格常数**（实测 sd~1e-16），
   λ=0 处恰好为 0 —— 那就是该次运行真正施加的 `lrc_coeff[k]/V`。
   （重算需要猜 V；实测生产用的 V 比 `box_vectors_solvent.npy` 小 2.7%，
   4W53 那次差 +2.746 vs +2.823。）
   本脚本把尾项**减在生产侧**，只是为了让"真值"一栏永远是原始测量值——
   「加在真值侧」与之**数学等价**，差值相同；致命的是**根本不对齐**。
2. **λ 表**：真值的 λ 表**未必**是这次运行实跑的那把梯子（4W53 那次真值用的是
   `baseline_v3_pre_wca_retirement_20260901/solvent_leg/` 的 preopt，
   实跑是 `output_v3_seed20260908/solvent_leg/`，逐点差 0.001~0.007）。
   端点相同 ⟹ **总量仍可比**，但**逐窗口拆分必须把真值插值到生产实跑的 λ**，
   否则 win4 那种 dF/dλ≈-157 的段，λ 错 0.0069 就假造出 1.1 kJ/mol 的"残差"。

3. **盒体积（密度）**：生产 stage2 走 NVT，用的是**NPT 预平衡末帧冻结的盒**；
   而 `docs/reference_data/` 的真值脚本（和 `attribute_shell_vs_single_ensemble.py`）
   固定用 `output/box_vectors_solvent.npy` 的**建系盒**，并显式拒绝恒压器。
   4W53 那次：生产 **42.747 nm³** vs 真值 **43.950 nm³**，**生产密度高 2.8%**。
   本脚本会从 LRC 偏移反解生产实际用的 V 并与建系盒对比，不一致就告警。
   实测灵敏度（2026-09-10 配对重跑：同 λ 表、同 seed，只换盒）：
   `∂ΔA_LJ/∂V = 4.07 kJ/mol/nm³`（68 bar）
   ⟹ 那 1.20 nm³ 的差值值 **−4.90 kJ/mol**，即 4W53 残差的 **100%**。
   ⚠️ 09-09 这里写过 0.71 ± 0.11（12 bar）/ −0.86 / 20% —— 那个 100 帧的有限差分
   **错了 5.7 倍**，已作废。

⚠️ 前两个口径错误会互相掩护：LRC 与真实误差符号相反，4W53 那次把 5.5σ 伪装成 1.9σ。

用法：
    python tools/diagnostics/attribute_stage2_solvent_leg_gap.py <run_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

TRUTH = REPO / "docs/reference_data/stage2_vanishing_truth_toluene_2026-09-02.json"
KB_KJ = 8.314462618e-3


def _pchip(x_desc, y_desc):
    """真值曲线的单调插值；x 递减，内部转成递增。"""
    from scipy.interpolate import PchipInterpolator

    return PchipInterpolator(x_desc[::-1], y_desc[::-1])


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_dir", type=Path, help="运行输出目录（含 solvent_leg/）")
    ap.add_argument("--truth-key", default="m2n2",
                    help="真值臂。逐窗口比只能用与生产同软核指数的那条（默认 m2n2）。")
    args = ap.parse_args()

    solv = args.run_dir / "solvent_leg"
    vdir = solv / "vanishing"
    results = json.loads((solv / "final_results.json").read_text())
    s2 = results["stage_diagnostics"]["stage2"]
    segments = s2["covariance_chain_segments"]
    kT = KB_KJ * float(results["protocol_key"]["payload"]["stage2_protocol_key"]
                       ["payload"]["temperature_K"])

    # ---- 从产物直接读每个 λ 态的 LRC 偏移，并顺带自检估计器 ----------------
    lam_prod, lrc_prod = [], []
    est_delta = []
    for w, seg in enumerate(segments):
        E = np.load(vdir / f"dual_window_{w}_vdw_energies.npy")            # (K, N)
        S = np.load(vdir / f"dual_window_{w}_vdw_sampling_states.npy").T   # (K, N)
        bias = np.load(vdir / f"dual_window_{w}_vdw_bias.npy")
        conv = json.loads((vdir / f"dual_window_{w}_vdw_convergence.json").read_text())
        lv = np.asarray(conv["lambdas_vdw"], dtype=float)

        off = E - S
        spread = float(np.abs(off - off.mean(axis=1, keepdims=True)).max())
        if spread > 1e-6:
            raise SystemExit(
                f"win{w}: energies-sampling_states 不是逐态常数（最大偏离 {spread:.3g}）。"
                "本脚本读 LRC 的前提不成立，先查产物。"
            )
        off = off.mean(axis=1)

        # 独立重算 ΔF（单参考重加权），用来自检落盘值
        lg0 = -(E[0] - bias) / kT
        lg1 = -(E[-1] - bias) / kT
        dF = -kT * (np.log(np.exp(lg1 - lg1.max()).sum()) + lg1.max()
                    - np.log(np.exp(lg0 - lg0.max()).sum()) - lg0.max())
        est_delta.append(dF - seg["delta_G_kJ_mol"])

        take = slice(0, len(lv)) if w == 0 else slice(1, len(lv))
        lam_prod.extend(lv[take])
        lrc_prod.extend(off[take])

    lam_prod = np.asarray(lam_prod)
    lrc_prod = np.asarray(lrc_prod)

    truth = json.loads(TRUTH.read_text())[args.truth_key]
    lam_t = np.asarray(truth["lambda_vdw"], dtype=float)
    f_t = np.asarray(truth["per_state_Delta_f_kT"], dtype=float) * (
        KB_KJ * float(truth["temperature_K"]))
    interp = _pchip(lam_t, f_t)

    same_ladder = len(lam_t) == len(lam_prod) and np.allclose(lam_t, lam_prod, atol=1e-9)
    print(f"λ 表：真值 {len(lam_t)} 态 / 实跑 {len(lam_prod)} 态；"
          f"{'完全相同' if same_ladder else f'**不同**（最大差 {np.abs(lam_t-lam_prod).max():.6f}）→ 真值已插值到实跑 λ'}")
    print(f"估计器自检：独立重加权 − 落盘 ΔG，逐窗口最大 |差| = "
          f"{max(abs(x) for x in est_delta):.3f} kJ/mol")
    print(f"LRC（从 energies−sampling_states 直接读）：λ=1 {lrc_prod[0]:+.5f} → "
          f"λ=0 {lrc_prod[-1]:+.5f}   Δ全程 = {lrc_prod[-1]-lrc_prod[0]:+.5f} kJ/mol")

    # 反解生产实际用的盒体积：offset[k] == coeff[k]/V ⟹ 全部 k 必须给同一个 V。
    # 这既是 LRC 身份的判据，也用来发现"生产与参考不在同一密度"。
    try:
        import ibs_engine as _ie
        import openmm as _mm

        _sys = _mm.XmlSerializer.deserialize(
            (args.run_dir / "system_solvent.xml").read_text())
        _lig = json.loads((args.run_dir / "ligand_indices_solvent.json").read_text())
        _lig = set(_lig["ligand_indices"] if isinstance(_lig, dict) else _lig)
        _nb = next(f for f in _sys.getForces() if isinstance(f, _mm.NonbondedForce))
        _ap = [_nb.getParticleParameters(i) for i in range(_nb.getNumParticles())]
        _env = [i for i in range(_nb.getNumParticles()) if i not in _lig]
        _sg, _s6, _s12 = _ie._lj_tail_correction_sigma_resolved_moments(_ap, sorted(_lig), _env)
        _sc = json.loads((solv / "final_results.json").read_text())["protocol_key"]["payload"][
            "stage2_protocol_key"]["payload"]["aces_softcore_params"]
        _coeff = _ie._lj_tail_lrc_coefficients_kj_mol(
            lam_prod, _sg, _s6, _s12, float(_sc["alpha_lj"]),
            *(float(x) for x in _sc["power_lj"]),
            _ie.LJ_TAIL_LRC_R_SWITCH_NM, _ie.LJ_TAIL_LRC_R_CUTOFF_NM)
        _nz = lrc_prod != 0.0
        _V = _coeff[_nz] / lrc_prod[_nz]
        _box = np.load(args.run_dir / "box_vectors_solvent.npy")
        _Vbuild = float(abs(np.linalg.det(_box)))
        print(f"反解生产实际盒：V = {_V.mean():.5f} nm^3（{_nz.sum()} 个 λ 态一致到 "
              f"{_V.std()/_V.mean():.1e} 相对精度 ⟹ 确认该偏移就是 lrc_coeff/V）")
        if abs(_V.mean() - _Vbuild) / _Vbuild > 1e-4:
            _dV = _Vbuild - _V.mean()
            print(f"⚠️ **密度不一致**：建系盒（真值脚本固定用它，且拒绝恒压器）= "
                  f"{_Vbuild:.5f} nm^3，生产用 NPT 预平衡末帧盒 = {_V.mean():.5f} nm^3，"
                  f"ΔV = {_dV:+.4f} nm^3（生产密度高 {100*_dV/_V.mean():.2f}%）。")
            print(f"   按配对重跑实测 ∂ΔA_LJ/∂V = 4.07 kJ/mol/nm^3 折算，这一项值 "
                  f"{-_dV*4.07:+.2f} kJ/mol —— 4W53 那次这**就是残差的全部**"
                  f"（重测真值 −11.490，残差 +0.59 = 0.72σ）。别把它归给采样协议。")
    except Exception as _e:  # 诊断辅助，失败不影响主结论
        print(f"（盒体积反解跳过：{type(_e).__name__}: {_e}）")

    print(f"\n{'win':>3} {'实跑 λ 段':>17} {'生产':>9} {'ΔLRC':>7} {'生产−LRC':>10} "
          f"{'真值(插值)':>11} {'残差':>8} {'σ':>6}")
    tp = tt = 0.0
    for w, seg in enumerate(segments):
        a, b = seg["join_lambda_index"], seg["end_lambda_index"]
        d_lrc = lrc_prod[b] - lrc_prod[a]
        prod = seg["delta_G_kJ_mol"] - d_lrc
        tru = float(interp(lam_prod[b]) - interp(lam_prod[a]))
        tp += prod
        tt += tru
        print(f"{w:>3} {f'{lam_prod[a]:.4f}->{lam_prod[b]:.4f}':>17} "
              f"{seg['delta_G_kJ_mol']:>+9.3f} {d_lrc:>+7.3f} {prod:>+10.3f} "
              f"{tru:>+11.3f} {prod - tru:>+8.3f} {seg['uncertainty_kJ_mol']:>6.3f}")
    resid = tp - tt
    print(f"{'合计':>3} {'':>17} {sum(s['delta_G_kJ_mol'] for s in segments):>+9.3f} "
          f"{lrc_prod[-1]-lrc_prod[0]:>+7.3f} {tp:>+10.3f} {tt:>+11.3f} {resid:>+8.3f}")

    sig = (float(results["total_error_kJ_mol"]) ** 2
           + float(truth["error_kJ_mol"]) ** 2) ** 0.5
    naive = sum(s["delta_G_kJ_mol"] for s in segments) - tt
    print(f"\n口径对齐后残差 = {resid:+.3f} kJ/mol  ({abs(resid)/sig:.2f}σ, 合并 σ={sig:.3f})")
    print(f"未对齐（直接相减）= {naive:+.3f} kJ/mol  ({abs(naive)/sig:.2f}σ)  ← 别引用这个数")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
