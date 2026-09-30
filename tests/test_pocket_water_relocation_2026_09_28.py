"""Pocket-water relocation sampler (tools/pocket_water): U0 engine and stage-1 acceptance.

1. U0 difference of moving one water == OpenMM direct-space energy difference
   (Reference platform, reciprocal space in its own force group).
2. Incremental Z_R(y) / Z_S(y) used by the acceptance == recomputed from scratch.
3. Stage-1 log ratio == -beta dU0 + ln q(y->x) - ln q(x->y) + dg, with every
   term recomputed independently, for in- and out-moves; forward and reverse
   log ratios cancel.
"""

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import app, unit  # noqa: E402

from tools.pocket_water.relocation import RelocationMC, logsumexp, min_dist  # noqa: E402
from tools.pocket_water.u0 import U0Engine  # noqa: E402

pytestmark = pytest.mark.cpu_only

BETA = 1.0 / (0.0083144626 * 300.0)


@pytest.fixture(scope="module")
def waterbox():
    ff = app.ForceField("tip3p.xml")
    mod = app.Modeller(app.Topology(), [])
    mod.addSolvent(ff, boxSize=openmm.Vec3(2.6, 2.6, 2.6) * unit.nanometer)
    system = ff.createSystem(mod.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * unit.nanometer,
                             rigidWater=True, ewaldErrorTolerance=5e-4)
    nb = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    for f in system.getForces():
        f.setForceGroup(2)
    nb.setForceGroup(0)
    nb.setReciprocalSpaceForceGroup(1)
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    pos = np.array(mod.positions.value_in_unit(unit.nanometer))
    box = np.array([2.6, 2.6, 2.6])
    ctx.setPositions(pos)
    waters = np.array([[a.index for a in r.atoms()] for r in mod.topology.residues()])
    eng = U0Engine.from_openmm(nb, ctx, waters)
    return ctx, eng, pos, box, waters


def _direct(ctx, pos):
    ctx.setPositions(pos)
    return ctx.getState(getEnergy=True, groups={0}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def _mc(eng, box, rng, g=None):
    centers = np.array([[1.3, 1.3, 1.3], [1.55, 1.3, 1.3]])
    mc = RelocationMC(eng, BETA, 0.35, 0.8, centers, lambda p: centers, lambda p: 0.0, rng, g=g,
                      volume_samples=400_000)
    return mc, centers


def test_u0_difference_matches_openmm_direct_space(waterbox):
    ctx, eng, pos, box, waters = waterbox
    rng = np.random.default_rng(1)
    eng.set_state(pos, box)
    e0 = _direct(ctx, pos)
    for w in rng.choice(len(waters), 5, replace=False):
        u_old = eng.water_energies([w])[0]
        new = pos.copy()
        new[waters[w]] = pos[waters[w]] + rng.normal(scale=0.05, size=3)
        eng.set_state(new, box)
        u_new = eng.water_energies([w])[0]
        eng.set_state(pos, box)
        d_ref = _direct(ctx, new) - e0
        assert abs((u_new - u_old) - d_ref) < 1e-3 * max(1.0, abs(d_ref)), (u_new - u_old, d_ref)


def _from_scratch(mc, eng, pos, box, s, sites):
    """Everything the acceptance needs, recomputed in state y without increments."""
    y = pos.copy()
    y[eng.water_atoms[s]] = sites
    eng.set_state(y, box)
    O = eng.water_atoms[:, 0]
    d = min_dist(y[O], mc.centers, box)
    R = np.where(d < mc.r_in)[0]
    S = np.where((d >= mc.r_in) & (d < mc.r_out))[0]
    uR, uS = eng.water_energies(R), eng.water_energies(S)
    u_s_y = eng.water_energies([s])[0]
    eng.set_state(pos, box)
    return R, S, logsumexp(BETA * uR), logsumexp(BETA * uS), u_s_y


@pytest.mark.parametrize("direction", ["in", "out"])
def test_stage1_ratio_matches_independent_terms(waterbox, direction):
    ctx, eng, pos, box, waters = waterbox
    rng = np.random.default_rng(7 if direction == "in" else 11)
    g = np.linspace(0.0, 3.0, 200) ** 1.3  # arbitrary non-flat bias
    mc, _ = _mc(eng, box, rng, g=g)
    mc.setup(pos, box)
    assert mc.N > 0 and len(mc.S) > 0
    is_in, src, sites, valid = mc.propose(400)
    want = is_in if direction == "in" else ~is_in
    idx = np.where(want & valid)[0][:6]
    log_r, dU0, u_y = mc.evaluate(is_in[idx], src[idx], sites[idx], clip=False)
    N = mc.N
    for k, t in enumerate(idx):
        s = int(src[t])
        R_y, S_y, lzR_y, lzS_y, u_s_y = _from_scratch(mc, eng, pos, box, s, sites[t])
        u_s_x = (mc.u_S[mc.S_index[s]] if is_in[t] else mc.u_R[mc.R_index[s]])
        assert abs(u_y[k] - u_s_y) < 1e-8 * max(1.0, abs(u_s_y))
        assert abs(dU0[k] - (u_s_y - u_s_x)) < 1e-8 * max(1.0, abs(dU0[k]))
        if is_in[t]:
            assert s in R_y and len(R_y) == N + 1
            lq_f = BETA * u_s_x - mc.logZ_S - np.log(mc.V_R)
            lq_r = BETA * u_s_y - lzR_y - np.log(mc.V_S)
            dg = g[N + 1] - g[N]
        else:
            assert s in S_y and len(R_y) == N - 1
            lq_f = BETA * u_s_x - mc.logZ_R - np.log(mc.V_S)
            lq_r = BETA * u_s_y - lzS_y - np.log(mc.V_R)
            dg = g[N - 1] - g[N]
        expect = -BETA * (u_s_y - u_s_x) + lq_r - lq_f + dg
        if np.isfinite(expect):
            assert abs(log_r[k] - expect) < 1e-6 * max(1.0, abs(expect)), (log_r[k], expect)

        # reverse move from y back to x cancels the forward ratio
        y = pos.copy()
        y[eng.water_atoms[s]] = sites[t]
        mc_r, _ = _mc(eng, box, np.random.default_rng(0), g=g)
        mc_r.V_R, mc_r.V_S = mc.V_R, mc.V_S  # same region volumes (MC-integrated once)
        mc_r.setup(y, box)
        back = pos[eng.water_atoms[s]][None]
        log_rr, _, _ = mc_r.evaluate(np.array([not is_in[t]]), np.array([s]), back, clip=False)
        if np.isfinite(log_r[k]) and np.isfinite(log_rr[0]):
            assert abs(log_r[k] + log_rr[0]) < 1e-6 * max(1.0, abs(log_r[k]))
        eng.set_state(pos, box)
