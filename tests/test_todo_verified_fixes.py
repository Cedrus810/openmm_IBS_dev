"""Focused regressions for the verified, non-DEXP TODO fixes."""

import json

import numpy as np
import pytest
from openmm import NonbondedForce, System, Vec3, app, unit

import ibs_engine as ie
import runabfe
from abfe_pipeline import _pre_equilibration_fingerprint


def _write_window_triplet(tmp_path):
    energies_path = tmp_path / "energies.npy"
    bias_path = tmp_path / "bias.npy"
    base_path = tmp_path / "base.npy"
    np.save(energies_path, np.arange(12, dtype=float).reshape(3, 4))
    np.save(bias_path, np.linspace(0.0, 0.3, 4))
    np.save(base_path, np.linspace(1.0, 1.3, 4))
    metadata = ie._window_data_metadata(
        str(energies_path), str(bias_path), str(base_path)
    )
    convergence = {
        "window_data_protocol_version": ie.IBS_WINDOW_DATA_PROTOCOL_VERSION,
        "window_data": metadata,
    }
    return energies_path, bias_path, base_path, convergence


def test_window_analysis_triplet_requires_complete_matching_manifest(tmp_path):
    energies, bias, base, convergence = _write_window_triplet(tmp_path)
    loaded = ie._load_validated_window_data_triplet(
        str(energies), str(bias), str(base), convergence
    )
    assert [array.shape for array in loaded] == [(3, 4), (4,), (4,)]

    bias.unlink()
    with pytest.raises(FileNotFoundError, match="bias"):
        ie._load_validated_window_data_triplet(
            str(energies), str(bias), str(base), convergence
        )


def test_window_analysis_triplet_rejects_tampering_and_length_mismatch(tmp_path):
    energies, bias, base, convergence = _write_window_triplet(tmp_path)
    np.save(base, np.linspace(1.0, 1.4, 5))
    with pytest.raises(ValueError, match="hash"):
        ie._load_validated_window_data_triplet(
            str(energies), str(bias), str(base), convergence
        )

    # A newly generated manifest must also reject an internally inconsistent
    # triplet instead of allowing downstream min-length truncation.
    with pytest.raises(ValueError, match="形状、长度或有限性"):
        ie._window_data_metadata(str(energies), str(bias), str(base))


def test_online_local_mbar_rejects_length_mismatch_without_truncation():
    result = ie._solve_single_window_local_mbar(
        u_kj_raw=np.zeros((2, 10)),
        bias_kj=np.zeros(9),
        base_kj=np.zeros(10),
        win_lams=[0, 1],
        kt=2.5,
    )
    assert "帧数不一致" in result["error"]


def test_energy_query_failure_gates_cover_fraction_total_and_streak():
    sampler = object.__new__(ie.IBSSampler)

    sampler._energy_query_attempts = 100
    sampler._energy_query_failures = 1
    sampler._energy_query_consecutive_failures = 0
    sampler._energy_query_failure_reasons = {"probe": 1}
    sampler.assert_energy_query_quality()

    sampler._energy_query_failures = 2
    with pytest.raises(RuntimeError, match="hard gate"):
        sampler.assert_energy_query_quality()

    sampler._energy_query_attempts = 10
    sampler._energy_query_failures = 1
    with pytest.raises(RuntimeError, match="hard gate"):
        sampler.assert_energy_query_quality(final=True)

    sampler._energy_query_attempts = 50
    sampler._energy_query_failures = 5
    sampler._energy_query_consecutive_failures = (
        ie.ENERGY_QUERY_MAX_CONSECUTIVE_FAILURES
    )
    with pytest.raises(RuntimeError, match="hard gate"):
        sampler.assert_energy_query_quality()

    sampler._energy_query_attempts = 1000
    sampler._energy_query_failures = ie.ENERGY_QUERY_MAX_TOTAL_FAILURES
    sampler._energy_query_consecutive_failures = 0
    with pytest.raises(RuntimeError, match="hard gate"):
        sampler.assert_energy_query_quality()


def test_triclinic_minimum_image_wraps_in_fractional_coordinates():
    box = np.asarray(
        [[2.0, 0.0, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 2.0]],
        dtype=float,
    )
    displacement = np.asarray([2.25, 1.8, 0.0])
    wrapped = ie._minimum_image_displacement_nm(displacement, box)
    np.testing.assert_allclose(wrapped, [-0.25, -0.2, 0.0], atol=1.0e-12)


def test_counterion_selection_uses_nearest_solute_pbc_and_handles_multivalent():
    topology = app.Topology()
    chain = topology.addChain()
    ligand = topology.addResidue("LIG", chain)
    protein = topology.addResidue("ALA", chain)
    ions = [topology.addResidue("CL", chain) for _ in range(3)]
    topology.addAtom("C1", app.element.carbon, ligand)
    topology.addAtom("CA", app.element.carbon, protein)
    for residue in ions:
        topology.addAtom("CL", app.element.chlorine, residue)

    force = NonbondedForce()
    for charge in (2.0, 0.0, -1.0, -1.0, -1.0):
        force.addParticle(
            charge * unit.elementary_charge,
            0.3 * unit.nanometer,
            0.0 * unit.kilojoule_per_mole,
        )
    positions = np.asarray(
        [
            [0.1, 0.1, 0.1],  # ligand
            [1.0, 1.0, 1.0],  # protein
            [1.8, 0.1, 0.1],  # close to ligand through PBC
            [0.6, 0.6, 0.6],
            [1.5, 1.5, 1.5],
        ]
    ) * unit.nanometer
    box = np.eye(3) * 2.0

    selected, references, metadata = ie._select_bulk_water_counterion(
        force, [0], topology, positions, box
    )
    assert set(selected) == {3, 4}
    assert len(references) == 2
    assert metadata["required_count"] == 2

    charge, sigma, epsilon = force.getParticleParameters(0)
    force.setParticleParameters(
        0, 0.49 * unit.elementary_charge, sigma, epsilon
    )
    with pytest.raises(RuntimeError, match="不接近整数"):
        ie._select_bulk_water_counterion(force, [0], topology, positions, box)


def test_pre_equilibration_fingerprint_binds_pose_box_and_step_budget():
    system = System()
    system.addParticle(12.0)
    positions = np.asarray([[0.1, 0.2, 0.3]]) * unit.nanometer
    moved = np.asarray([[0.2, 0.2, 0.3]]) * unit.nanometer
    box = [
        Vec3(2.0, 0.0, 0.0),
        Vec3(0.0, 2.0, 0.0),
        Vec3(0.0, 0.0, 2.0),
    ] * unit.nanometer

    common = dict(
        system=system,
        ligand_indices=[0],
        temperature=300.0,
        pressure=1.0,
        box_vectors=box,
    )
    baseline = _pre_equilibration_fingerprint(
        positions=positions, requested_steps=1000, **common
    )
    assert baseline != _pre_equilibration_fingerprint(
        positions=moved, requested_steps=1000, **common
    )
    assert baseline != _pre_equilibration_fingerprint(
        positions=positions, requested_steps=2000, **common
    )
    changed_box = [
        Vec3(2.1, 0.0, 0.0),
        Vec3(0.0, 2.0, 0.0),
        Vec3(0.0, 0.0, 2.0),
    ] * unit.nanometer
    assert baseline != _pre_equilibration_fingerprint(
        positions=positions,
        requested_steps=1000,
        **{**common, "box_vectors": changed_box},
    )


def test_main_cache_rejects_tampered_topology_or_box(tmp_path, monkeypatch):
    files = {
        "system_native.xml": "system",
        "ligand_indices.json": json.dumps({"ligand_indices": [0]}),
        "topology.cif": "topology",
    }
    for name, content in files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    np.save(tmp_path / "box_vectors.npy", np.eye(3))

    identity = {"identity_sha256": "expected"}
    monkeypatch.setattr(runabfe, "_main_cache_identity", lambda *args: identity)
    manifest = {
        "protocol_version": runabfe.MAIN_SYSTEM_CACHE_PROTOCOL_VERSION,
        **identity,
        "system_xml_sha256": runabfe._sha256_file(
            str(tmp_path / "system_native.xml")
        ),
        "ligand_indices_sha256": runabfe._sha256_file(
            str(tmp_path / "ligand_indices.json")
        ),
        "topology_sha256": runabfe._sha256_file(str(tmp_path / "topology.cif")),
        "box_vectors_sha256": runabfe._sha256_file(
            str(tmp_path / "box_vectors.npy")
        ),
    }
    (tmp_path / "system_cache_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    assert runabfe.system_cache_exists(str(tmp_path))

    (tmp_path / "topology.cif").write_text("tampered", encoding="utf-8")
    assert not runabfe.system_cache_exists(str(tmp_path))


def test_remd_gpu_context_limit_falls_back_before_replica_build(
    tmp_path, monkeypatch
):
    observed_platforms = []

    def _record_build(self, _system_template):
        observed_platforms.append(self.platform_name)

    monkeypatch.setattr(ie.REMDManager, "_build_replicas", _record_build)
    manager = ie.REMDManager(
        system_template=None,
        topology=None,
        positions=None,
        box_vectors=None,
        ligand_indices=[],
        lambdas_coul=[1.0, 0.5, 0.0],
        lambdas_vdw=[1.0, 1.0, 1.0],
        platform_name="CUDA",
        output_dir=str(tmp_path),
        max_resident_contexts=1,
    )
    assert observed_platforms == ["CPU"]
    assert manager.context_residency_mode == "cpu_fallback_bounded_gpu_contexts"
    assert manager.max_resident_contexts == 1
