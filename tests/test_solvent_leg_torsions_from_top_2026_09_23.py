"""溶剂腿配体扭转 / 离子 LJ 必须与复合物 top 同源（2026-09-23）。

旧 ligand_only.xml 写无编号 `periodicity=`，ForceField 静默读成 0 项：benchmark 13/13
个体系溶剂腿配体没有扭转。离子则是 amber14 而非 top 的参数。
"""
import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import app, unit  # noqa: E402

runabfe = pytest.importorskip("runabfe")

TOP = """[ defaults ]
1 2 yes 0.5 0.8333
[ atomtypes ]
c3 6 12.01 0.0 A 0.3400 0.4577
hc 1 1.008 0.0 A 0.2600 0.0657
Na 11 22.99 0.0 A 0.33284 0.01159
Cl 17 35.45 0.0 A 0.44010 0.41840
[ moleculetype ]
MOL 3
[ atoms ]
1 c3 1 MOL C1 1 -0.1 12.01
2 c3 1 MOL C2 2 0.0 12.01
3 c3 1 MOL C3 3 0.0 12.01
4 c3 1 MOL C4 4 -0.1 12.01
5 hc 1 MOL H1 5 0.1 1.008
6 hc 1 MOL H2 6 0.1 1.008
[ bonds ]
1 2 1 0.153 250000
2 3 1 0.153 250000
3 4 1 0.153 250000
1 5 1 0.109 280000
4 6 1 0.109 280000
[ pairs ]
1 4 1
5 3 1
6 2 1
[ angles ]
1 2 3 1 110 400
2 3 4 1 110 400
5 1 2 1 110 400
3 4 6 1 110 400
[ dihedrals ]
1 2 3 4 9 0 1.0 1
1 2 3 4 9 180 2.0 2
1 2 3 4 9 0 3.0 3
5 1 2 3 9 0 0.6 3
2 3 4 6 9 0 0.6 3
[ moleculetype ]
NA 1
[ atoms ]
1 Na 1 NA NA 1 1.0 22.99
[ moleculetype ]
CL 1
[ atoms ]
1 Cl 1 CL CL 1 -1.0 35.45
[ system ]
t
[ molecules ]
MOL 1
NA 1
CL 1
"""


def _torsion_energy(system, positions):
    for i, f in enumerate(system.getForces()):
        f.setForceGroup(1 if isinstance(f, openmm.PeriodicTorsionForce) else 0)
    ctx = openmm.Context(system, openmm.VerletIntegrator(1.0),
                         openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(positions)
    if system.usesPeriodicBoundaryConditions():
        ctx.setPeriodicBoxVectors(*system.getDefaultPeriodicBoxVectors())
    return ctx.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole)


def test_solvent_leg_ligand_torsions_and_ions_come_from_top(tmp_path):
    top_path = tmp_path / "t.top"
    top_path.write_text(TOP)
    gmx = app.GromacsTopFile(str(top_path))
    lig_top = [a.index for a in gmx.topology.atoms() if a.residue.name == "MOL"]
    rng = np.random.default_rng(0)
    lig_xyz = np.cumsum(rng.normal(0, 0.1, (len(lig_top), 3)), axis=0) + 1.5
    all_xyz = np.vstack([lig_xyz, [[0.5, 0.5, 0.5], [2.5, 2.5, 2.5]]])

    xml = runabfe.generate_ligand_xml_from_top(str(top_path), "MOL", str(tmp_path))
    modeller = app.Modeller(gmx.topology, all_xyz * unit.nanometer)
    modeller.delete([a for a in modeller.topology.atoms() if a.index not in lig_top])
    for atom, g in zip(modeller.topology.atoms(),
                       lig_top):
        atom.name = runabfe._build_stable_ligand_atom_names(gmx, "MOL")[g]
    ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml", xml)
    modeller.addSolvent(ff, boxSize=openmm.Vec3(2.5, 2.5, 2.5) * unit.nanometer,
                        positiveIon="Na+", negativeIon="Cl-",
                        ionicStrength=1.0 * unit.molar)
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=1.0 * unit.nanometer,
                             constraints=app.HBonds, rigidWater=True)
    solv_lig = [a.index for a in modeller.topology.atoms() if a.residue.name == "MOL"]
    top_system = gmx.createSystem(nonbondedMethod=app.NoCutoff, constraints=None)

    out = runabfe._overwrite_solvent_leg_terms_from_top(
        system, modeller.topology, top_system, gmx.topology, dict(zip(lig_top, solv_lig)))
    assert out["ligand_torsions"] == 5
    assert out["ions"] > 0

    # 扭转能逐位等于 top 的（同一组配体坐标）
    pos = modeller.positions
    lig_pos = [pos[i] for i in solv_lig]
    ref_pos = list(lig_pos) + [openmm.Vec3(0.5, 0.5, 0.5) * unit.nanometer,
                               openmm.Vec3(2.5, 2.5, 2.5) * unit.nanometer]
    e_top = _torsion_energy(top_system, ref_pos)
    e_solv = _torsion_energy(system, pos)
    assert abs(e_top) > 1e-3
    assert e_solv == pytest.approx(e_top, abs=1e-8)

    nb = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    for res in modeller.topology.residues():
        atoms = list(res.atoms())
        if len(atoms) == 1:
            _, sig, eps = nb.getParticleParameters(atoms[0].index)
            want = {"Na": (0.33284, 0.01159), "Cl": (0.44010, 0.41840)}[atoms[0].element.symbol]
            assert sig.value_in_unit(unit.nanometer) == pytest.approx(want[0])
            assert eps.value_in_unit(unit.kilojoule_per_mole) == pytest.approx(want[1])
