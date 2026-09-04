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

#ifndef OPENMM_LOCALMANYBODYRESIDUAL_FORCE_PROXY_H_
#define OPENMM_LOCALMANYBODYRESIDUAL_FORCE_PROXY_H_

/*
 * EXP-025 serialization contract.
 *
 * `schema_version` is an explicit, versioned XML contract for
 * LocalManyBodyResidualForce, independent of the plugin binary/OpenMM ABI.
 * G0 ships schema_version=1 covering only the standard Force fields
 * (forceGroup, name) since no R1 parameters exist yet. G1 is expected to
 * ADD properties to this same schema_version=1 payload (ligand topology
 * ids, atom type index, cutoff/skin/temperature, RBF centers, typed MLP
 * weights) -- bump to schema_version=2 only if a genuinely incompatible
 * layout change is required, not for ordinary additive growth.
 */

#include "openmm/serialization/SerializationProxy.h"
#include "openmm/internal/windowsExportLocalManyBodyResidual.h"

namespace OpenMM {

class OPENMM_EXPORT_LMBR LocalManyBodyResidualForceProxy : public SerializationProxy {
public:
    LocalManyBodyResidualForceProxy();
    void serialize(const void* object, SerializationNode& node) const;
    void* deserialize(const SerializationNode& node) const;
};

} // namespace OpenMM

#endif /*OPENMM_LOCALMANYBODYRESIDUAL_FORCE_PROXY_H_*/
