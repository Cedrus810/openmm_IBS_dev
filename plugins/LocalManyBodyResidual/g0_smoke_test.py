"""EXP-025 G0 build/ABI smoke test.

Loads the three plugin libraries built by g0_build.sh via
Platform.loadPluginLibrary() (no install into the OpenMM plugin directory,
no modification of the existing OpenMM install), builds a trivial System,
adds an empty LocalManyBodyResidualForce, and confirms:

  1. all three .so load without error;
  2. a Reference Context can be created and getState(energy=True) reports
     exactly 0.0;
  3. a CUDA Context can be created -- which internally forces the plugin's
     NVRTC JIT smoke kernel to compile+execute+verify inside initialize()
     -- and getState(energy=True) also reports exactly 0.0.

This does NOT test any R1 math. It only answers the G0 question. See
docs/experiments/PLAN_EXP-025_local_manybody_cuda.md section 10 (G0).
"""
import os
import sys
import traceback

import openmm
from openmm import unit

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")

PLUGINS = [
    os.path.join(BUILD, "libOpenMMLocalManyBodyResidual.so"),
    os.path.join(BUILD, "libOpenMMLocalManyBodyResidualReference.so"),
    os.path.join(BUILD, "libOpenMMLocalManyBodyResidualCUDA.so"),
]


def load_plugins():
    for p in PLUGINS:
        print(f"loadPluginLibrary: {p}")
        openmm.Platform.loadPluginLibrary(p)
    print("all plugin libraries loaded OK")


def build_trivial_system():
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(2, 0, 0), openmm.Vec3(0, 2, 0), openmm.Vec3(0, 0, 2)
    )
    for _ in range(4):
        system.addParticle(1.0)
    force = openmm.LocalManyBodyResidualForce()
    system.addForce(force)
    positions = [
        openmm.Vec3(0.0, 0.0, 0.0),
        openmm.Vec3(0.3, 0.0, 0.0),
        openmm.Vec3(0.0, 0.3, 0.0),
        openmm.Vec3(0.0, 0.0, 0.3),
    ] * unit.nanometer
    return system, positions


def run_on_platform(platform_name):
    system, positions = build_trivial_system()
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = state.getForces(asNumpy=True)
    max_abs_force = float(abs(forces).max()) if len(forces) else 0.0
    print(f"[{platform_name}] potential energy = {energy!r} kJ/mol, max|force| = {max_abs_force!r}")
    assert energy == 0.0, f"{platform_name}: expected exactly 0.0 kJ/mol, got {energy}"
    assert max_abs_force == 0.0, f"{platform_name}: expected exactly zero force, got max {max_abs_force}"
    return energy


def main():
    load_plugins()

    print("\n--- Reference platform ---")
    run_on_platform("Reference")

    print("\n--- CUDA platform ---")
    try:
        run_on_platform("CUDA")
    except Exception:
        print("CUDA platform G0 smoke test FAILED:")
        traceback.print_exc()
        sys.exit(1)

    print("\nG0 SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
