#include "openmm/internal/LocalManyBodyResidualForceImpl.h"
#include "openmm/LocalManyBodyResidualKernels.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace OpenMM;
using namespace std;

LocalManyBodyResidualForceImpl::LocalManyBodyResidualForceImpl(const LocalManyBodyResidualForce& owner) : owner(owner) {
}

LocalManyBodyResidualForceImpl::~LocalManyBodyResidualForceImpl() {
}

void LocalManyBodyResidualForceImpl::initialize(ContextImpl& context) {
    kernel = context.getPlatform().createKernel(CalcLocalManyBodyResidualForceKernel::Name(), context);
    kernel.getAs<CalcLocalManyBodyResidualForceKernel>().initialize(context.getSystem(), owner);
}

double LocalManyBodyResidualForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups&(1<<owner.getForceGroup())) != 0)
        return kernel.getAs<CalcLocalManyBodyResidualForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

vector<string> LocalManyBodyResidualForceImpl::getKernelNames() {
    return {CalcLocalManyBodyResidualForceKernel::Name()};
}

void LocalManyBodyResidualForceImpl::updateParametersInContext(ContextImpl& context) {
    // In-place same-shape coefficient update (PLAN_EXP-025 section 4.2) is
    // not implemented yet. Fail loud rather than silently no-op: silently
    // ignoring a parameter change would leave a stale kernel running with
    // the OLD weights/cutoffs/kBT while the caller believes they were
    // updated. Create a new Context instead.
    throw OpenMMException(
        "LocalManyBodyResidualForce::updateParametersInContext is not yet implemented; "
        "create a new Context after changing parameters instead of updating in place.");
}
