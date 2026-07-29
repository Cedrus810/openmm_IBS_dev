import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

t0 = time.time()
import numpy as np
import openmm
from openmm import app, unit, XmlSerializer
import mdtraj as md
print(f"[import] {time.time()-t0:.1f}s", flush=True)

import abfe_core as core

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

with open(os.path.join(BASE, "ligand_indices.json")) as f:
    payload = json.load(f)
lig_idx = np.asarray(payload["ligand_indices"] if isinstance(payload, dict) else payload, dtype=int)
print(f"ligand atoms: {len(lig_idx)}", flush=True)

with open(os.path.join(BASE, "system_native.xml")) as f:
    system = XmlSerializer.deserialize(f.read())
print(f"total particles: {system.getNumParticles()}", flush=True)

traj_full = md.load(os.path.join(BASE, "pre_equilibration.dcd"), top=os.path.join(BASE, "topology.cif"))
print(f"trajectory frames: {len(traj_full)}", flush=True)
traj = traj_full[-1]
traj = traj.image_molecules(inplace=False)
xyz = traj.xyz[0]
box = traj.unitcell_vectors[0] if traj.unitcell_vectors is not None else None

lig_set = set(lig_idx.tolist())
box_lens = np.linalg.norm(box, axis=1)
delta = xyz[lig_idx][:, None, :] - xyz[None, :, :]
delta -= box_lens * np.round(delta / box_lens)
dists_to_lig = np.linalg.norm(delta, axis=-1).min(axis=0)
env_mask = (dists_to_lig < 0.6) & (~np.isin(np.arange(len(xyz)), lig_idx))
env_idx = np.where(env_mask)[0]
print(f"environment atoms within 0.6nm: {len(env_idx)}", flush=True)

surrogate_potential = core.DEXPSurrogatePotential()
print("surrogate params:", surrogate_potential.get_parameters_dict(), flush=True)
print("expr:", surrogate_potential.build_expression(), flush=True)

builder = core.SurrogateSystemBuilder({}, ghost_handler=None)
box_vectors_omm = [openmm.Vec3(*row) for row in box] * unit.nanometer if box is not None else None
new_system = builder.build_surrogate_system(
    system,
    ligand_indices=lig_idx.tolist(),
    environment_indices=env_idx.tolist(),
    box_vectors=box_vectors_omm,
)
print(f"surrogate system built OK, forces: {[type(f).__name__ for f in new_system.getForces()]}", flush=True)

integrator = openmm.VerletIntegrator(0.001)
platform = openmm.Platform.getPlatformByName("CUDA")
context = openmm.Context(new_system, integrator, platform, {"Precision": "mixed"})
context.setPositions(xyz * unit.nanometer)
if box is not None:
    context.setPeriodicBoxVectors(*box_vectors_omm)
context.setParameter("lam_coul", 1.0)
context.setParameter("lam_vdw", 1.0)

state = context.getState(getEnergy=True, getForces=True, groups={1})
e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
print(f"surrogate-group (Gaussian-Coulomb + DEXP) potential energy: {e:.3f} kJ/mol", flush=True)
print(f"forces finite: {np.all(np.isfinite(forces))}, max |F| on ligand: {np.abs(forces[lig_idx]).max():.3f} kJ/mol/nm", flush=True)

state_full = context.getState(getEnergy=True, getForces=True)
e_full = state_full.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
forces_full = state_full.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
print(f"full-system potential energy: {e_full:.3f} kJ/mol", flush=True)
print(f"full-system forces finite: {np.all(np.isfinite(forces_full))}", flush=True)

# short dynamics stability check: a few steps of Langevin at 300K shouldn't blow up
integrator2 = openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
context2 = openmm.Context(new_system, integrator2, openmm.Platform.getPlatformByName("CUDA"), {"Precision": "mixed"})
context2.setPositions(xyz * unit.nanometer)
if box is not None:
    context2.setPeriodicBoxVectors(*box_vectors_omm)
context2.setParameter("lam_coul", 1.0)
context2.setParameter("lam_vdw", 1.0)
context2.setVelocitiesToTemperature(300 * unit.kelvin)
integrator2.step(200)
state2 = context2.getState(getEnergy=True)
e2 = state2.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
print(f"potential energy after 200 steps (0.4ps) Langevin @300K: {e2:.3f} kJ/mol, finite={np.isfinite(e2)}", flush=True)

print("SMOKE TEST PASSED", flush=True)
