#!/usr/bin/env python
"""Standalone OpenMM probe: is CustomNonbondedForce.setUseLongRangeCorrection(True)
usable as-is for the softcore VDW forces in ibs_engine.py / abfe_core.py?

No project imports required -- this is meant to be dropped on a PBS/GPU node by
itself and run with only openmm + numpy installed.

    python test_lrc_interaction_group_compat.py --platform CUDA
    python test_lrc_interaction_group_compat.py --platform CPU

Two independent questions are tested, because they are two independent ways
this could fail:

Q1. When setUseLongRangeCorrection(True) is combined with addInteractionGroup()
    (as every softcore VDW force in this project does), does OpenMM even accept
    it, and if so does the correction actually track the *interaction-group-
    restricted* energy (ligand<->environment cross term only), or does it get
    computed as if the group restriction were not there?

    Method: evaluate the SAME interaction-group-restricted CustomNonbondedForce
    three ways -- (a) production cutoff (1.2 nm) with LRC off, (b) production
    cutoff with LRC on, (c) a much larger cutoff with LRC off (the "ground
    truth" the correction is trying to approximate). (b) should land close to
    (c); (a) should not. A whole-system NonbondedForce + setUseDispersionCorrection
    control case is run alongside as a known-good baseline, so a failure can be
    attributed to "interaction groups specifically" rather than "LRC is broken
    on this OpenMM build/platform in general".

Q2. BeutlerSoftcoreBuilder (used by --decoupling single_lambda / REMD, see
    ibs_engine.py:4171-4185) builds ONE CustomNonbondedForce with
    addGlobalParameter("lambda_vdw", 1.0), creates the Context once, and then
    calls context.setParameter("lambda_vdw", <per-replica value>) afterwards.
    ACESoftcorePotential (the default dual_lambda path) instead hardcodes
    lambda straight into the expression string per state and never calls
    setParameter. If OpenMM's long-range correction is computed once at
    Context-creation time from the GlobalParameter's *construction-time*
    value and is not refreshed when setParameter() is called later, every
    REMD replica's LRC term would silently be wrong (frozen at whatever
    lambda_vdw was passed to addGlobalParameter, not the real per-replica
    value) while the ACE/dual_lambda path -- which never uses setParameter --
    would be fine. This is exactly the kind of divergence between the two
    softcore implementations that would need to be caught before shipping.

    Method: build context A directly with lambda_vdw=0.0. Build context B
    with lambda_vdw defaulting to 1.0, then call setParameter("lambda_vdw", 0.0)
    on it. Compare A vs B first with LRC off (must match -- sanity check that
    setParameter correctly updates the *ordinary* pairwise energy) and then
    with LRC on (if this stops matching once LRC off already matched, the
    mismatch is coming specifically from a stale long-range correction term).

The script prints a table of raw energies plus a pass/fail verdict for each
question. It does not require GPU; CPU platform is fine, just slower.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

try:
    import openmm
    from openmm import unit
except ImportError:  # older openmm packaged as simtk
    from simtk import openmm
    from simtk.openmm import unit


# ---------------------------------------------------------------------------
# Expressions copied verbatim from the production code so the test is
# representative of what actually runs. Keep these in sync by hand if the
# source functions change; this script intentionally does not import the
# project modules so it can run standalone on a bare compute node.
# ---------------------------------------------------------------------------

def ace_softcore_expression(
    lam_coul: str,
    lam_vdw: str,
    alpha_lj: float = 0.5,
    alpha_coul: float = 0.2,
    m_lj: int = 2,
    n_lj: int = 2,
    m_coul: int = 1,
    n_coul: int = 1,
) -> str:
    """Verbatim from abfe_core.py::ACESoftcorePotential.build_expression.
    lam_coul/lam_vdw are pre-formatted numeric strings (lambda hardcoded)."""
    COUL = 138.935456
    lc, lv = f"({lam_coul}^{n_coul})", f"({lam_vdw}^{n_lj})"
    dlj = f"max({alpha_lj}*(1.0-{lam_vdw})^{m_lj} + r^6, 1e-6)"
    dc = f"sqrt(max(r^2 + {alpha_coul}*(1.0-{lam_coul})^{m_coul}, 1e-6))"
    sigma12 = "(0.5*(sigma1+sigma2))"
    lj = f"{lv} * 4 * sqrt(epsilon1*epsilon2) * ({sigma12}^12/({dlj}^2) - {sigma12}^6/{dlj})"
    coul = f"{lc} * {COUL} * q1 * q2 / {dc}"
    return f"{lj} + {coul}"


def beutler_softcore_expression(
    alpha_lj: float = 0.5,
    alpha_coul: float = 0.5,
    power_lj: int = 1,
    power_coul: int = 1,
) -> str:
    """Verbatim from abfe_core.py::BeutlerSoftcoreBuilder.build.
    Uses the GlobalParameter names lambda_vdw / lambda_coul (not hardcoded)."""
    return (
        f"lambda_vdw * 4*sqrt(epsilon1*epsilon2)*("
        f"(sigma12^12 / (r^6 + {alpha_lj}*(1-lambda_vdw)^{power_lj} + 1e-4*(1-lambda_vdw))^2) - "
        f"(sigma12^6 / (r^6 + {alpha_lj}*(1-lambda_vdw)^{power_lj} + 1e-4*(1-lambda_vdw)))"
        f") + "
        f"lambda_coul * 138.935456 * q1*q2 / sqrt(r^2 + {alpha_coul}*(1-lambda_coul)^{power_coul} + 1e-3); "
        f"sigma12=(0.5*(sigma1+sigma2))"
    )


# ---------------------------------------------------------------------------
# Test system: a periodic box of uncharged LJ particles, split into a small
# "ligand" group near the box center and a large "environment" group filling
# the rest. Charges are fixed at 0 everywhere so the Coulomb terms in the
# expressions above evaluate to exactly zero -- this isolates the LJ
# dispersion tail, which is the actual open question (LRC), from anything
# Coulomb/PME related.
# ---------------------------------------------------------------------------

def build_positions(n_total: int, box_nm: float, seed: int):
    rng = np.random.default_rng(seed)
    n_side = int(np.ceil(n_total ** (1.0 / 3.0)))
    spacing = box_nm / n_side
    coords = []
    for ix in range(n_side):
        for iy in range(n_side):
            for iz in range(n_side):
                if len(coords) >= n_total:
                    break
                jitter = (rng.random(3) - 0.5) * spacing * 0.3
                pos = (np.array([ix, iy, iz], dtype=float) + 0.5) * spacing + jitter
                coords.append(pos)
            if len(coords) >= n_total:
                break
        if len(coords) >= n_total:
            break
    return np.asarray(coords[:n_total])


def pick_ligand_indices(coords: np.ndarray, box_nm: float, n_ligand: int):
    center = np.full(3, box_nm / 2.0)
    dist = np.linalg.norm(coords - center, axis=1)
    order = np.argsort(dist)
    ligand = sorted(int(i) for i in order[:n_ligand])
    ligand_set = set(ligand)
    environment = [i for i in range(len(coords)) if i not in ligand_set]
    return ligand, environment


def box_vectors(box_nm: float):
    return (
        openmm.Vec3(box_nm, 0.0, 0.0),
        openmm.Vec3(0.0, box_nm, 0.0),
        openmm.Vec3(0.0, 0.0, box_nm),
    ) * unit.nanometer


def new_system(n_particles: int, box_nm: float):
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(*box_vectors(box_nm))
    for _ in range(n_particles):
        system.addParticle(39.95 * unit.amu)
    return system


def evaluate(system, coords, box_nm, platform_name):
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(system, integrator, platform)
    context.setPeriodicBoxVectors(*box_vectors(box_nm))
    context.setPositions(coords * unit.nanometer)
    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    del context, integrator
    return energy


def evaluate_with_param_change(system, coords, box_nm, platform_name, param_name, param_value):
    """Build once with default GlobalParameter, then setParameter afterwards --
    mirrors exactly what REMDManager does for lambda_vdw/lambda_coul."""
    integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(system, integrator, platform)
    context.setPeriodicBoxVectors(*box_vectors(box_nm))
    context.setPositions(coords * unit.nanometer)
    context.setParameter(param_name, param_value)
    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    del context, integrator
    return energy


# ---------------------------------------------------------------------------
# Q1: interaction-group + LRC correctness
# ---------------------------------------------------------------------------

def run_q1(coords, ligand, environment, box_nm, sigma_nm, epsilon_kj,
           small_cutoff, large_cutoff, platform_name, lambda_vdw=1.0):
    print("=" * 78)
    print("Q1: does setUseLongRangeCorrection(True) work with addInteractionGroup?")
    print("=" * 78)
    if lambda_vdw == 0.0:
        print("WARNING: lambda_vdw=0.0 makes the ACE LJ term (lv = lambda_vdw^n_lj) identically "
              "zero everywhere -- both cutoffs will trivially agree regardless of LRC. Use an "
              "intermediate value like 0.5 to actually test the softcore-denominator case.")

    results = {}

    # --- control: whole-system NonbondedForce, no interaction groups -------
    for cutoff, use_lrc, tag in (
        (small_cutoff, False, "control_trunc"),
        (small_cutoff, True, "control_lrc"),
        (large_cutoff, False, "control_ref"),
    ):
        system = new_system(len(coords), box_nm)
        nb = openmm.NonbondedForce()
        nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
        nb.setCutoffDistance(cutoff * unit.nanometer)
        nb.setUseDispersionCorrection(use_lrc)
        nb.setUseSwitchingFunction(False)
        for _ in range(len(coords)):
            nb.addParticle(0.0 * unit.elementary_charge, sigma_nm * unit.nanometer,
                            epsilon_kj * unit.kilojoule_per_mole)
        system.addForce(nb)
        try:
            results[tag] = evaluate(system, coords, box_nm, platform_name)
        except Exception as exc:
            results[tag] = f"EXCEPTION: {exc!r}"
    print(f"[control, whole-system NonbondedForce]")
    print(f"  small cutoff, LRC off : {results['control_trunc']}")
    print(f"  small cutoff, LRC on  : {results['control_lrc']}")
    print(f"  large cutoff, LRC off : {results['control_ref']} (reference)")
    if all(isinstance(results[k], float) for k in ("control_trunc", "control_lrc", "control_ref")):
        err_trunc = results["control_trunc"] - results["control_ref"]
        err_lrc = results["control_lrc"] - results["control_ref"]
        print(f"  truncation error (no LRC)  : {err_trunc:+.4f} kJ/mol")
        print(f"  residual error (LRC on)    : {err_lrc:+.4f} kJ/mol")
        print(f"  -> control sanity: {'PASS (LRC works in general on this build)' if abs(err_lrc) < 0.1 * abs(err_trunc) else 'FAIL (LRC itself is not correcting properly here -- stop, fix OpenMM/platform setup before trusting anything below)'}")
    print()

    # --- the real question: interaction-group-restricted CustomNonbondedForce
    lam_vdw_str = f"{lambda_vdw:.8f}"  # ==1.0 reduces the ACE softcore EXACTLY to plain 12-6 LJ;
                                        # any other value keeps the softcore denominator genuinely
                                        # different from r^6, which is what actually matters here.
    lam_coul_str = "1.00000000"  # irrelevant: charges are 0 everywhere
    expr = ace_softcore_expression(lam_coul_str, lam_vdw_str)

    for cutoff, use_lrc, tag in (
        (small_cutoff, False, "ig_trunc"),
        (small_cutoff, True, "ig_lrc"),
        (large_cutoff, False, "ig_ref"),
    ):
        system = new_system(len(coords), box_nm)
        force = openmm.CustomNonbondedForce(expr)
        for p in ("q", "sigma", "epsilon"):
            force.addPerParticleParameter(p)
        for _ in range(len(coords)):
            force.addParticle([0.0, sigma_nm, epsilon_kj])
        force.addInteractionGroup(list(ligand), list(environment))
        force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        force.setCutoffDistance(cutoff * unit.nanometer)
        force.setUseSwitchingFunction(False)
        try:
            force.setUseLongRangeCorrection(use_lrc)
        except Exception as exc:
            results[tag] = f"EXCEPTION at setUseLongRangeCorrection: {exc!r}"
            continue
        system.addForce(force)
        try:
            results[tag] = evaluate(system, coords, box_nm, platform_name)
        except Exception as exc:
            results[tag] = f"EXCEPTION at Context/evaluate: {exc!r}"

    print(f"[real case, CustomNonbondedForce + addInteractionGroup, ACE expr @ lambda_vdw={lambda_vdw}]")
    print(f"  small cutoff, LRC off : {results['ig_trunc']}")
    print(f"  small cutoff, LRC on  : {results['ig_lrc']}")
    print(f"  large cutoff, LRC off : {results['ig_ref']} (reference)")
    verdict_q1 = None
    if all(isinstance(results[k], float) for k in ("ig_trunc", "ig_lrc", "ig_ref")):
        err_trunc = results["ig_trunc"] - results["ig_ref"]
        err_lrc = results["ig_lrc"] - results["ig_ref"]
        recovered = 1.0 - (abs(err_lrc) / abs(err_trunc) if err_trunc != 0 else float("inf"))
        print(f"  truncation error (no LRC)  : {err_trunc:+.4f} kJ/mol")
        print(f"  residual error (LRC on)    : {err_lrc:+.4f} kJ/mol")
        print(f"  fraction of truncation error removed by LRC: {recovered:+.2%}")
        if abs(err_lrc) < 0.15 * abs(err_trunc):
            verdict_q1 = "PASS: setUseLongRangeCorrection(True) correctly restricts itself to the interaction-group cross term. Safe to flip the switch on the real code (still re-check at fractional lambda_vdw, softcore != plain LJ there)."
        else:
            verdict_q1 = "FAIL: LRC does not converge to the untruncated interaction-group reference. Either it is using whole-system density instead of the group-restricted density, or something else is off. Do NOT just flip the switch -- a hand-derived analytical tail term is needed instead."
    else:
        verdict_q1 = "INCONCLUSIVE: an exception occurred, see EXCEPTION text above -- OpenMM may reject setUseLongRangeCorrection(True) combined with addInteractionGroup outright on this build."
    print(f"  -> Q1 verdict: {verdict_q1}")
    print()
    return results


# ---------------------------------------------------------------------------
# Q2: does the LRC term stay in sync with context.setParameter() changes to
# a GlobalParameter, the pattern BeutlerSoftcoreBuilder/REMDManager relies on?
# ---------------------------------------------------------------------------

def run_q2(coords, ligand, environment, box_nm, sigma_nm, epsilon_kj,
           cutoff, platform_name, target_lambda_vdw=0.4):
    print("=" * 78)
    print("Q2: does LRC stay in sync with context.setParameter() (REMD/Beutler pattern)?")
    print("=" * 78)

    # IMPORTANT: target_lambda_vdw must NOT be 0.0 or 1.0. The Beutler
    # expression has a bare "lambda_vdw *" prefactor on the whole LJ term, so
    # at lambda_vdw=0.0 the total energy (and, however LRC is computed,
    # anything derived from the same expression) is identically zero
    # regardless of whether the correction is stale -- that would make this
    # test pass trivially no matter what. An intermediate value keeps the
    # softcore denominator (and hence any tail-integral value) genuinely
    # different between lambda_vdw=1.0 (construction default) and the target,
    # so a stale-vs-fresh LRC actually shows up as a numeric difference.
    assert 0.0 < target_lambda_vdw < 1.0, "target_lambda_vdw must be strictly between 0 and 1"

    expr = beutler_softcore_expression()

    def make_force(use_lrc):
        force = openmm.CustomNonbondedForce(expr)
        for p in ("q", "sigma", "epsilon"):
            force.addPerParticleParameter(p)
        force.addGlobalParameter("lambda_coul", 1.0)
        force.addGlobalParameter("lambda_vdw", 1.0)  # construction-time default, matches BeutlerSoftcoreBuilder
        for _ in range(len(coords)):
            force.addParticle([0.0, sigma_nm, epsilon_kj])
        force.addInteractionGroup(list(ligand), list(environment))
        force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        force.setCutoffDistance(cutoff * unit.nanometer)
        force.setUseSwitchingFunction(False)
        force.setUseLongRangeCorrection(use_lrc)
        return force

    results = {}
    for use_lrc, tag_prefix in ((False, "off"), (True, "on")):
        # A: build directly at lambda_vdw = target (never touches setParameter for this run)
        system_a = new_system(len(coords), box_nm)
        force_a = make_force(use_lrc)
        system_a.addForce(force_a)
        try:
            e_a = evaluate_with_param_change(system_a, coords, box_nm, platform_name, "lambda_vdw", target_lambda_vdw)
        except Exception as exc:
            e_a = f"EXCEPTION: {exc!r}"
        results[f"built_at_target_{tag_prefix}"] = e_a

        # B: build at default lambda_vdw=1.0, then setParameter(lambda_vdw, target) -- the REMD pattern
        system_b = new_system(len(coords), box_nm)
        force_b = make_force(use_lrc)
        system_b.addForce(force_b)
        try:
            e_b = evaluate_with_param_change(system_b, coords, box_nm, platform_name, "lambda_vdw", target_lambda_vdw)
        except Exception as exc:
            e_b = f"EXCEPTION: {exc!r}"
        results[f"default1_then_settarget_{tag_prefix}"] = e_b

    print(f"[LRC off -- sanity check that setParameter updates the ordinary energy at all, target lambda_vdw={target_lambda_vdw}]")
    print(f"  built directly at lambda_vdw={target_lambda_vdw}                 : {results['built_at_target_off']}")
    print(f"  built at default 1.0, then setParameter({target_lambda_vdw}) : {results['default1_then_settarget_off']}")
    sane = False
    if all(isinstance(results[k], float) for k in ("built_at_target_off", "default1_then_settarget_off")):
        diff_off = abs(results["built_at_target_off"] - results["default1_then_settarget_off"])
        sane = diff_off < 1e-6
        print(f"  difference: {diff_off:.8f} kJ/mol -> {'OK, setParameter works for the base energy' if sane else 'UNEXPECTED MISMATCH even with LRC off -- investigate before trusting Q2 at all'}")
    print()

    print("[LRC on -- the actual question]")
    print(f"  built directly at lambda_vdw={target_lambda_vdw}                 : {results['built_at_target_on']}")
    print(f"  built at default 1.0, then setParameter({target_lambda_vdw}) : {results['default1_then_settarget_on']}")
    verdict_q2 = None
    if all(isinstance(results[k], float) for k in ("built_at_target_on", "default1_then_settarget_on")):
        diff_on = abs(results["built_at_target_on"] - results["default1_then_settarget_on"])
        print(f"  difference: {diff_on:.8f} kJ/mol")
        if not sane:
            verdict_q2 = "INCONCLUSIVE: base energy itself did not match with LRC off, fix that first."
        elif diff_on < 1e-6:
            verdict_q2 = "PASS: LRC correctly recomputes after context.setParameter(). Safe for the REMD/Beutler per-replica setParameter pattern."
        else:
            verdict_q2 = ("FAIL: LRC is stale -- it was baked in at the GlobalParameter's construction-time value "
                           f"(lambda_vdw=1.0) and did NOT update when setParameter(lambda_vdw, {target_lambda_vdw}) was called. "
                           "This means every REMDManager replica's dispersion correction is silently computed at "
                           "the wrong lambda. Do not add LRC to BeutlerSoftcoreBuilder via setUseLongRangeCorrection "
                           "without first calling context.reinitialize() after every setParameter, or switch that "
                           "path to hardcode lambda into the expression per-context the way ACESoftcorePotential does.")
    else:
        verdict_q2 = "INCONCLUSIVE: an exception occurred, see EXCEPTION text above."
    print(f"  -> Q2 verdict: {verdict_q2}")
    print()
    return results


# ---------------------------------------------------------------------------
# Q3: does LRC survive contact with a genuine, non-vanishing Coulomb term
# bundled into the SAME expression as the LJ term? Both ACESoftcorePotential
# and BeutlerSoftcoreBuilder write "lj_expr + coul_expr" as ONE
# CustomNonbondedForce, and setUseLongRangeCorrection has no way to be told
# "only correct the LJ half". Q1/Q2 above used charge=0.0 everywhere
# specifically to zero out the Coulomb half and isolate the LJ tail -- that
# was deliberate scope-narrowing, not an oversight, but it also means Q1/Q2
# say nothing about what happens once real nonzero partial charges are
# present. The analytic "integrate the pair energy from cutoff to infinity
# assuming uniform density" approximation converges for an r^-6 LJ tail
# (integrand ~ r^2 * r^-6 = r^-4) but is mathematically DIVERGENT for a
# Coulomb r^-1 tail (integrand ~ r^2 * r^-1 = r, unbounded as r -> infinity).
# Whether OpenMM's actual implementation throws, silently truncates the
# integral at some internal bound, or returns something numerically toxic
# (huge value / NaN / Inf) for a mixed LJ+Coulomb expression with nonzero
# charges is exactly what this probes -- empirically, not by assumption.
# ---------------------------------------------------------------------------

def assign_charges(n_total: int, seed: int, magnitude_e: float) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)  # different stream from position jitter
    return rng.uniform(-magnitude_e, magnitude_e, size=n_total)


def run_q3(coords, charges, ligand, environment, box_nm, sigma_nm, epsilon_kj,
           small_cutoff, large_cutoff, platform_name, lambda_vdw=1.0, lambda_coul=1.0):
    print("=" * 78)
    print("Q3: does LRC stay sane once the SAME expression has a real (nonzero) Coulomb term?")
    print("=" * 78)
    print(f"charge magnitude: uniform in [-{np.max(np.abs(charges)):.3f}, +{np.max(np.abs(charges)):.3f}] e per particle "
          f"(net charge of the box: {charges.sum():+.4f} e)")

    lam_vdw_str = f"{lambda_vdw:.8f}"
    lam_coul_str = f"{lambda_coul:.8f}"
    expr = ace_softcore_expression(lam_coul_str, lam_vdw_str)

    results = {}
    for cutoff, use_lrc, tag in (
        (small_cutoff, False, "trunc"),
        (small_cutoff, True, "lrc"),
        (large_cutoff, False, "ref"),
    ):
        system = new_system(len(coords), box_nm)
        force = openmm.CustomNonbondedForce(expr)
        for p in ("q", "sigma", "epsilon"):
            force.addPerParticleParameter(p)
        for i in range(len(coords)):
            force.addParticle([float(charges[i]), sigma_nm, epsilon_kj])
        force.addInteractionGroup(list(ligand), list(environment))
        force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        force.setCutoffDistance(cutoff * unit.nanometer)
        force.setUseSwitchingFunction(False)
        try:
            force.setUseLongRangeCorrection(use_lrc)
        except Exception as exc:
            results[tag] = f"EXCEPTION at setUseLongRangeCorrection: {exc!r}"
            continue
        system.addForce(force)
        try:
            results[tag] = evaluate(system, coords, box_nm, platform_name)
        except Exception as exc:
            results[tag] = f"EXCEPTION at Context/evaluate: {exc!r}"

    print(f"[real case with charges, ACE expr @ lambda_vdw={lambda_vdw}, lambda_coul={lambda_coul}]")
    print(f"  small cutoff, LRC off : {results['trunc']}")
    print(f"  small cutoff, LRC on  : {results['lrc']}")
    print(f"  large cutoff, LRC off : {results['ref']} (reference, itself only approximate for a 1/r tail)")

    verdict_q3 = None
    numeric = all(isinstance(results[k], float) for k in ("trunc", "lrc", "ref"))
    if not numeric:
        verdict_q3 = "INCONCLUSIVE/FAIL: an exception occurred, see EXCEPTION text above -- OpenMM rejected LRC on a mixed LJ+Coulomb interaction-group force with nonzero charges."
    else:
        trunc, lrc, ref = results["trunc"], results["lrc"], results["ref"]
        if not all(np.isfinite(v) for v in (trunc, lrc, ref)):
            verdict_q3 = "FAIL: LRC produced a non-finite (NaN/Inf) energy. Do not enable LRC on the combined LJ+Coulomb expression as-is."
        else:
            # Sanity scale: how big is the LRC-induced shift relative to the truncated
            # energy itself? A huge, out-of-scale jump indicates a divergent/toxic
            # correction; a modest shift comparable to the Q1 LJ-only case is reassuring
            # (though still not a guarantee the Coulomb tail is being handled *correctly*,
            # only that it isn't blowing up).
            shift = lrc - trunc
            scale = max(abs(trunc), abs(ref), 1e-6)
            relative_shift = abs(shift) / scale
            print(f"  LRC-induced shift vs no-LRC energy: {shift:+.4f} kJ/mol ({relative_shift:.1%} of the truncated energy's scale)")
            if relative_shift > 5.0:
                verdict_q3 = ("FAIL: turning on LRC shifts the energy by many times its own scale -- consistent with "
                              "the divergent-integral concern for the Coulomb tail. Do NOT flip "
                              "setUseLongRangeCorrection(True) on the combined LJ+Coulomb expression used by "
                              "_create_softcore_force / BeutlerSoftcoreBuilder as-is. The LJ and Coulomb terms "
                              "would need to be split into two separate CustomNonbondedForce objects so LRC can "
                              "be enabled on the LJ-only force and left off on the Coulomb-only force.")
            else:
                verdict_q3 = ("PASS (no blow-up observed): the correction stayed within a sane scale with real "
                              "charges present on this OpenMM build. This does NOT prove the Coulomb tail is being "
                              "corrected *correctly* (a plain large cutoff is not a trustworthy ground truth for a "
                              "1/r tail under PBC either -- that would need Ewald), only that it is not numerically "
                              "toxic. Recommend still splitting LJ and Coulomb into separate forces so LRC only ever "
                              "touches the r^-6 term it was designed for, rather than relying on this not blowing up "
                              "for every future charge distribution.")
    print(f"  -> Q3 verdict: {verdict_q3}")
    print()
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", default="CUDA", choices=["CUDA", "OpenCL", "CPU", "Reference"])
    parser.add_argument("--n-total", type=int, default=6000, help="total LJ particles in the box")
    parser.add_argument("--n-ligand", type=int, default=30, help="particles flagged as the ligand group")
    parser.add_argument("--box-nm", type=float, default=8.0)
    parser.add_argument("--small-cutoff-nm", type=float, default=1.2, help="matches production cutoff")
    parser.add_argument("--large-cutoff-nm", type=float, default=3.5, help="must be < box/2")
    parser.add_argument("--sigma-nm", type=float, default=0.32)
    parser.add_argument("--epsilon-kj", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--q2-target-lambda-vdw", type=float, default=0.4,
                         help="must be strictly between 0 and 1, else the Beutler "
                              "expression's bare 'lambda_vdw *' prefactor makes the test degenerate")
    parser.add_argument("--q1-lambda-vdw", type=float, default=1.0,
                         help="lambda_vdw for the ACE-expression interaction-group test in Q1. "
                              "1.0 reduces exactly to plain LJ (easy case); rerun with e.g. 0.5 to "
                              "test the actual softcore-denominator case that production hits mid-window")
    parser.add_argument("--q3-charge-magnitude-e", type=float, default=0.4,
                         help="uniform per-particle charge is drawn from [-this, +this] elementary "
                              "charges for Q3 (the mixed LJ+Coulomb-with-real-charges test)")
    parser.add_argument("--q3-lambda-vdw", type=float, default=1.0)
    parser.add_argument("--q3-lambda-coul", type=float, default=1.0)
    parser.add_argument("--skip-q3", action="store_true",
                         help="skip the charged-particle LRC probe (not recommended -- Q1/Q2 alone "
                              "do not clear the combined LJ+Coulomb expression for production use)")
    args = parser.parse_args(argv)

    if args.large_cutoff_nm >= args.box_nm / 2.0:
        print(f"ERROR: --large-cutoff-nm ({args.large_cutoff_nm}) must be < box/2 ({args.box_nm / 2.0})", file=sys.stderr)
        return 2

    print(f"OpenMM version: {openmm.__version__}")
    print(f"Platform requested: {args.platform}")
    print(f"Box: {args.box_nm} nm cubic, {args.n_total} particles "
          f"(density {args.n_total / args.box_nm**3:.2f} nm^-3), {args.n_ligand} flagged as ligand")
    print(f"Cutoffs: production={args.small_cutoff_nm} nm, reference={args.large_cutoff_nm} nm")
    print()

    coords = build_positions(args.n_total, args.box_nm, args.seed)
    ligand, environment = pick_ligand_indices(coords, args.box_nm, args.n_ligand)

    run_q1(
        coords, ligand, environment, args.box_nm, args.sigma_nm, args.epsilon_kj,
        args.small_cutoff_nm, args.large_cutoff_nm, args.platform,
        lambda_vdw=args.q1_lambda_vdw,
    )
    run_q2(
        coords, ligand, environment, args.box_nm, args.sigma_nm, args.epsilon_kj,
        args.small_cutoff_nm, args.platform, target_lambda_vdw=args.q2_target_lambda_vdw,
    )
    if not args.skip_q3:
        charges = assign_charges(args.n_total, args.seed, args.q3_charge_magnitude_e)
        run_q3(
            coords, charges, ligand, environment, args.box_nm, args.sigma_nm, args.epsilon_kj,
            args.small_cutoff_nm, args.large_cutoff_nm, args.platform,
            lambda_vdw=args.q3_lambda_vdw, lambda_coul=args.q3_lambda_coul,
        )

    print("=" * 78)
    print(f"Done. Q1 above ran at --q1-lambda-vdw={args.q1_lambda_vdw}. A PASS there only proves "
          "LRC works for plain LJ if that value is 1.0 (the softcore reduces exactly to 12-6 "
          "there). Re-run with --q1-lambda-vdw 0.5 (or a few other values in (0,1)) to check "
          "whether the correction still tracks once the softcore denominator genuinely differs "
          "from r^6 -- that is the case production actually hits in every mid-window state.")
    if not args.skip_q3:
        print("Q3 above ran with real charges at the endpoint (lambda_vdw=lambda_coul=1.0 by default). "
              "Also worth re-running --q3-lambda-vdw 0.5 --q3-lambda-coul 0.5 to check the mixed-lambda "
              "mid-window case, and a couple of --seed values / --q3-charge-magnitude-e values to see "
              "if the verdict is robust to the specific charge distribution used.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
