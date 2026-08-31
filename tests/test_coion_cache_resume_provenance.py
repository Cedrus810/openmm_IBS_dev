"""B5 contract tests: cache identities are frozen, minimal, and auditable.

These tests intentionally use a tiny synthetic OpenMM system.  They do not run
MD; the production acceptance of a cache is the deterministic System/Topology
identity comparison implemented in ``runabfe`` and the runtime payload is the
same canonical spec fingerprint used by B3.
"""

import ast
import copy
import json
from pathlib import Path

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import NonbondedForce, Vec3, app, unit

import abfe_core as core


ROOT = Path(__file__).absolute().parents[1]
CT = core.CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER


def _synthetic_system():
    topology = app.Topology()
    topology.setPeriodicBoxVectors(
        (
            Vec3(4.0, 0.0, 0.0),
            Vec3(0.0, 4.0, 0.0),
            Vec3(0.0, 0.0, 4.0),
        )
        * unit.nanometer
    )
    chain = topology.addChain()
    ligand = topology.addResidue("LIG", chain)
    topology.addAtom("C1", app.element.carbon, ligand)
    coion = topology.addResidue("CL", chain)
    topology.addAtom("CL", app.element.chlorine, coion)

    system = openmm.System()
    system.addParticle(12.011 * unit.dalton)
    system.addParticle(35.45 * unit.dalton)
    nb = NonbondedForce()
    nb.addParticle(+1.0, 0.34, 0.4)
    nb.addParticle(0.0, 0.44, 0.2)
    system.addForce(nb)
    return system, topology


def _spec(system, topology):
    positions = [[2.0, 2.0, 2.0], [3.0, 2.0, 2.0]]
    box_vectors = np.asarray(
        [v.value_in_unit(unit.nanometer) for v in topology.getPeriodicBoxVectors()],
        dtype=np.float64,
    )
    return core.build_co_alchemical_ion_identity(
        system=system,
        topology=topology,
        ion_atom_indices=[1],
        ligand_indices=[0],
        positions_nm=positions,
        box_vectors=box_vectors,
        ligand_net_charge_e=1,
        charge_treatment=CT,
        enforce_placement_thresholds=False,
    )


def _contains_forbidden_identity_data(value):
    if isinstance(value, dict):
        return any(
            key in {"selection_provenance", "selection_time_absolute_position_nm"}
            or _contains_forbidden_identity_data(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_identity_data(child) for child in value)
    return False


def test_runtime_identity_is_minimal_and_ignores_selection_diagnostics():
    system, topology = _synthetic_system()
    spec = _spec(system, topology)
    payload = core.co_alchemical_ion_cache_identity_payload(
        spec,
        system=system,
        topology=topology,
        leg="complex",
        spec_relative_path="checkpoints/coalchemical_ion_spec.json",
        charge_treatment=CT,
    )
    assert payload["fingerprint"] == spec["fingerprint"]
    assert payload["leg"] == "complex"
    assert payload["ion_atom_indices"] == [1]
    assert not _contains_forbidden_identity_data(payload)

    changed_diagnostics = copy.deepcopy(spec)
    changed_diagnostics["selection_provenance"] = {"distance_nm": 999.0}
    changed_diagnostics["ions"][0]["selection_time_absolute_position_nm"] = [
        0.1,
        0.2,
        0.3,
    ]
    changed_payload = core.co_alchemical_ion_cache_identity_payload(
        changed_diagnostics,
        system=system,
        topology=topology,
        leg="complex",
        spec_relative_path="checkpoints/coalchemical_ion_spec.json",
        charge_treatment=CT,
    )
    assert changed_payload == payload


def test_builder_identity_is_recomputed_from_system_not_manifest():
    system, topology = _synthetic_system()
    first = core.co_alchemical_ion_builder_identity_payload(
        system=system,
        topology=topology,
        charge_treatment=CT,
        ligand_net_charge_e=1,
    )
    assert first["reserved_coion_count"] == 1
    assert "selection_provenance" not in first
    assert "restraint_protocol" in first

    nb = next(force for force in system.getForces() if isinstance(force, NonbondedForce))
    charge, sigma, epsilon = nb.getParticleParameters(1)
    nb.setParticleParameters(1, charge, sigma, 0.3 * unit.kilojoule_per_mole)
    second = core.co_alchemical_ion_builder_identity_payload(
        system=system,
        topology=topology,
        charge_treatment=CT,
        ligand_net_charge_e=1,
    )
    assert second != first
    assert second["ions"][0]["epsilon_kj_mol"] == pytest.approx(0.3)


def test_missing_charge_transfer_spec_fails_closed_with_leg_and_path():
    system, topology = _synthetic_system()
    with pytest.raises(ValueError, match="complex.*coalchemical_ion_spec.json"):
        core.co_alchemical_ion_cache_identity_payload(
            None,
            system=system,
            topology=topology,
            leg="complex",
            spec_relative_path="checkpoints/coalchemical_ion_spec.json",
            charge_treatment=CT,
        )


def test_neutral_identity_remains_none():
    system, topology = _synthetic_system()
    assert core.co_alchemical_ion_cache_identity_payload(
        None,
        system=system,
        topology=topology,
        leg="solvent",
        spec_relative_path="checkpoints/coalchemical_ion_spec.json",
        charge_treatment="neutral",
    ) is None


@pytest.mark.parametrize(
    "requested, recorded",
    [
        (CT, core.CHARGE_TREATMENT_CO_ANNIHILATION_EXPERIMENTAL),
        (core.CHARGE_TREATMENT_CO_ANNIHILATION_EXPERIMENTAL, CT),
    ],
)
def test_runtime_identity_rejects_mismatched_charge_route(requested, recorded):
    """The running route, not the spec's self-declaration, is authoritative."""
    system, topology = _synthetic_system()
    spec = _spec(system, topology)
    spec["charge_treatment"] = recorded
    spec["fingerprint"] = core.co_alchemical_ion_identity_fingerprint(spec)
    with pytest.raises(ValueError, match="charge_treatment.*不匹配"):
        core.co_alchemical_ion_cache_identity_payload(
            spec,
            system=system,
            topology=topology,
            leg="complex",
            spec_relative_path="checkpoints/coalchemical_ion_spec.json",
            charge_treatment=requested,
        )


def test_remd_sampling_metadata_requires_exact_fingerprint(tmp_path):
    """A complete DCD set is not enough when its sampling metadata changed."""
    import abfe_pipeline as pipeline

    expected = {"sha256": "sampling-a"}
    pipeline._write_remd_sampling_metadata(str(tmp_path), "decharging", expected)
    assert pipeline._remd_sampling_metadata_matches(
        str(tmp_path), "decharging", expected
    )
    assert not pipeline._remd_sampling_metadata_matches(
        str(tmp_path), "decharging", {"sha256": "sampling-b"}
    )
    (tmp_path / "decharging_sampling.meta.json").unlink()
    assert not pipeline._remd_sampling_metadata_matches(
        str(tmp_path), "decharging", expected
    )


def test_charge_transfer_final_gate_requires_both_legs_and_matching_protocol():
    import runabfe

    system, topology = _synthetic_system()
    complex_spec = _spec(system, topology)
    solvent_spec = copy.deepcopy(complex_spec)
    # Different frozen atom/fingerprint is allowed across legs; protocol is not.
    solvent_spec["ions"][0]["atom_index"] = 1
    solvent_spec["fingerprint"] = core.co_alchemical_ion_identity_fingerprint(
        solvent_spec
    )
    identity = {
        "schema_version": 1,
        "charge_treatment": CT,
        "fingerprint": complex_spec["fingerprint"],
    }
    solvent_identity = dict(identity, fingerprint=solvent_spec["fingerprint"])
    runabfe._assert_coion_legs_complete_and_compatible(
        {"complex": identity, "solvent": solvent_identity},
        complex_spec,
        solvent_spec,
        CT,
    )
    with pytest.raises(RuntimeError, match="两条腿都提供"):
        runabfe._assert_coion_legs_complete_and_compatible(
            {"complex": identity}, complex_spec, None, CT
        )
    incompatible = copy.deepcopy(solvent_spec)
    incompatible["lambda_direction"] = "wrong"
    with pytest.raises(RuntimeError, match="协议不一致"):
        runabfe._assert_coion_legs_complete_and_compatible(
            {"complex": identity, "solvent": solvent_identity},
            complex_spec,
            incompatible,
            CT,
        )


def test_charge_transfer_solvent_cache_requires_ordinary_salt_counts(tmp_path, monkeypatch):
    """Reserved dummy counts cannot satisfy the physical NaCl acceptance gate."""
    import runabfe

    for name in (
        "system_solvent.xml",
        "ligand_indices_solvent.json",
        "topology_solvent.cif",
    ):
        (tmp_path / name).write_text(name, encoding="utf-8")
    expected_identity = {"identity_sha256": "identity"}
    builder_identity = {"builder": "identity"}
    monkeypatch.setattr(runabfe, "_sha256_file", lambda path: "hash")
    monkeypatch.setattr(
        runabfe,
        "_recompute_cached_builder_identity",
        lambda **kwargs: builder_identity,
    )
    manifest = {
        "protocol_version": runabfe.SOLVENT_CACHE_PROTOCOL_VERSION,
        "ionic_strength_molar": 0.15,
        "neutralize": True,
        "identity_sha256": "identity",
        "system_xml_sha256": "hash",
        "ligand_indices_sha256": "hash",
        "topology_sha256": "hash",
        "charge_treatment": CT,
        "reserved_coion_builder_identity": builder_identity,
        "na_count": 1,
        "cl_count": 1,
        # No ordinary_* fields: this must fail closed even though total counts pass.
    }
    (tmp_path / "solvent_cache_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    assert not runabfe.solvent_cache_exists(
        str(tmp_path),
        expected_identity=expected_identity,
        charge_treatment=CT,
        expected_builder_identity=builder_identity,
    )
    manifest["ordinary_na_count"] = 1
    manifest["ordinary_cl_count"] = 1
    (tmp_path / "solvent_cache_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    assert runabfe.solvent_cache_exists(
        str(tmp_path),
        expected_identity=expected_identity,
        charge_treatment=CT,
        expected_builder_identity=builder_identity,
    )


def test_b5_source_contracts_are_present_and_selector_is_not_duplicated():
    pipeline_src = (ROOT / "abfe_pipeline.py").read_text(encoding="utf-8")
    runabfe_src = (ROOT / "runabfe.py").read_text(encoding="utf-8")
    assert pipeline_src.count("select_co_alchemical_ion_once(") == 1
    for marker in (
        "co_alchemical_ion_runtime_identity",
        "_stage_protocol_key",
        "_preopt_protocol_key",
        "coion_identity",
        "co_alchemical_ion_runtime_identity",
    ):
        assert marker in pipeline_src
    assert "_atomic_write_json_with_encoder" in runabfe_src
    assert "co_alchemical_ions" in runabfe_src

    # Keep this test AST-only: it remains useful in environments without OpenMM.
    ast.parse(pipeline_src)
    ast.parse(runabfe_src)
