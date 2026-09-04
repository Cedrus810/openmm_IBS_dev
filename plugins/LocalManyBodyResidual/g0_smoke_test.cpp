/* ---------------------------------------------------------------------------- *
 * ABFE-IBS -- LocalManyBodyResidual plugin                                     *
 *                                                                              *
 * Copyright (c) 2026 Ruigeng Ji                                                *
 *                                                                              *
 * This plugin's directory layout, build scaffolding and API skeleton are       *
 * derived from the OpenMM example plugin, which is MIT-licensed.               *
 * Portions copyright (c) Stanford University and the Authors.                  *
 *                                                                              *
 * Distributed under the MIT License; see LICENSE at the repository root for    *
 * the full text.  This plugin is compiled against OpenMM headers and linked    *
 * at run time against a separately installed OpenMM, whose CUDA, HIP and       *
 * OpenCL platforms are covered by the LGPL -- see NOTICE.                      *
 * ---------------------------------------------------------------------------- */

// EXP-025 G0 build/ABI smoke test -- native C++ harness.
//
// The PLAN doc (section 17) explicitly says not to invest in Python
// packaging before G4, so this deliberately bypasses SWIG/Python bindings
// (which don't exist yet for LocalManyBodyResidualForce) and drives the
// OpenMM C++ API directly: build a tiny System, add a LocalManyBodyResidualForce,
// create a Context on Reference then CUDA, and check that energy/forces are
// exactly zero and nothing crashes. Creating the CUDA Context forces the
// plugin's NVRTC JIT smoke kernel (see CudaLocalManyBodyResidualKernels.cpp)
// to compile+execute+verify inside initialize() -- if that fails, an
// OpenMMException propagates here.
//
// Since G1-G, the Reference kernel fail-closed validates its parameters at
// initialize() time (see ReferenceLocalManyBodyResidualKernels.cpp), so a
// fully-default-constructed Force is no longer a valid "empty" test case --
// this now builds the smallest VALID Force instead: one ligand atom and one
// environment atom placed far enough apart that zero edges form, which still
// exercises the full real validation/initialize() path while trivially
// giving energy=0, force=0 (q=0 with no edges => per_ligand=rho(0)-rho(0)=0
// regardless of the (here all-zero, otherwise-arbitrary) MLP weights).
#include "openmm/LocalManyBodyResidualForce.h"
#include "OpenMM.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <algorithm>

using namespace OpenMM;
using namespace std;

static LocalManyBodyResidualForce* buildMinimalNoContactForce() {
    LocalManyBodyResidualForce* force = new LocalManyBodyResidualForce();
    force->setTemperatureKelvin(300.0);
    force->setLigandTopologyIds({0});
    force->setAtomTypeIndex({0, 0});
    force->setTypeVocabulary({6});
    force->setInnerCutoffAngstrom(4.0);
    force->setOuterCutoffAngstrom(5.0);
    force->setRadialCenters(vector<double>(16, 0.0));
    force->setRadialWidthAngstrom(1.0);
    force->setPairWeight(vector<double>(1 * 1 * 16, 0.0));
    force->setBMaxReduced(10.0);
    force->setCapacityCeilings(10, 10, 10);
    LocalManyBodyTypedMLP mlp;
    mlp.w0.assign(16, 0.0);
    mlp.b0.assign(16, 0.0);
    mlp.w2.assign(256, 0.0);
    mlp.b2.assign(16, 0.0);
    mlp.w4.assign(16, 0.0);
    mlp.b4 = 0.0;
    force->setTypedMLP(0, mlp);
    return force;
}

static bool runOnPlatform(const string& platformName) {
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(3, 0, 0), Vec3(0, 3, 0), Vec3(0, 0, 3));
    system.addParticle(1.0);  // atom 0: the one ligand atom
    system.addParticle(1.0);  // atom 1: a lone environment atom, far away
    LocalManyBodyResidualForce* force = buildMinimalNoContactForce();
    system.addForce(force);

    vector<Vec3> positions = {
        Vec3(0.0, 0.0, 0.0),
        Vec3(1.2, 0.0, 0.0),  // 12 Angstrom away (>> 5 Angstrom cutoff); NOT exactly
                               // half the 3nm box, which would hit the half-box MIC
                               // tie this plugin deliberately fails closed on.
    };

    VerletIntegrator integrator(0.001);
    Platform& platform = Platform::getPlatformByName(platformName);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces);

    double energy = state.getPotentialEnergy();
    double maxAbsForce = 0.0;
    for (const Vec3& f : state.getForces())
        maxAbsForce = max({maxAbsForce, fabs(f[0]), fabs(f[1]), fabs(f[2])});

    cout << "[" << platformName << "] potential energy = " << energy
         << " kJ/mol, max|force| = " << maxAbsForce << endl;

    if (energy != 0.0) {
        cout << "[" << platformName << "] FAIL: expected exactly 0.0 kJ/mol" << endl;
        return false;
    }
    if (maxAbsForce != 0.0) {
        cout << "[" << platformName << "] FAIL: expected exactly zero force" << endl;
        return false;
    }
    return true;
}

int main() {
    // Our C++ harness bypasses the Python `openmm` module entirely, so
    // nothing has registered OpenMM's own built-in CUDA platform yet
    // (normally openmm/__init__.py does this on import via
    // loadPluginsFromDirectory). Do that first, then load our three plugin
    // libraries exactly the way a real deployment would: as standalone .so
    // files via loadPluginLibrary(), no install into the OpenMM plugin
    // directory, no rebuild of OpenMM itself.
    // NOTE: Platform::getDefaultPluginsDirectory() returns OpenMM's
    // build-time-baked-in default ("/usr/local/openmm/lib/plugins"), which
    // is NOT where this conda-forge install actually keeps its bundled
    // platform plugins. Point at the real conda plugins dir explicitly
    // (OPENMM_CONDA_PLUGIN_DIR, set via -D at compile time in g0_build.sh).
    cout << "loading builtin platform plugins from: " << OPENMM_CONDA_PLUGIN_DIR << endl;
    for (const string& err : Platform::loadPluginsFromDirectory(OPENMM_CONDA_PLUGIN_DIR))
        cout << "  (builtin plugin load message) " << err << endl;
    Platform::loadPluginLibrary(PLUGIN_DIR "/libOpenMMLocalManyBodyResidual.so");
    Platform::loadPluginLibrary(PLUGIN_DIR "/libOpenMMLocalManyBodyResidualReference.so");
    Platform::loadPluginLibrary(PLUGIN_DIR "/libOpenMMLocalManyBodyResidualCUDA.so");
    cout << "all plugin libraries loaded OK" << endl;

    bool ok = true;
    cout << "\n--- Reference platform ---" << endl;
    try {
        ok = runOnPlatform("Reference") && ok;
    } catch (const exception& e) {
        cout << "Reference platform G0 smoke test threw: " << e.what() << endl;
        ok = false;
    }

    cout << "\n--- CUDA platform ---" << endl;
    try {
        ok = runOnPlatform("CUDA") && ok;
    } catch (const exception& e) {
        cout << "CUDA platform G0 smoke test threw: " << e.what() << endl;
        ok = false;
    }

    cout << "\nG0 SMOKE TEST: " << (ok ? "PASS" : "FAIL") << endl;
    return ok ? 0 : 1;
}
