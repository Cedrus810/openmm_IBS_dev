"""EXP-012 complete-residue environment discovery: the direct fix for the
EXP-010 failure mode, where a single-frame per-atom radius cut sliced through
26 residues without including a single complete one.
"""

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

from local_residual.environment import build_environment_manifest


# Loaded by file path rather than ``from scripts.discover_exp012_environment_config
# import ...``: ``scripts/`` has no ``__init__.py``, so it is a PEP 420 namespace
# package, and depending on how a given pytest invocation builds sys.path for this
# rootdir, ``import scripts`` can bind to an empty/wrong namespace portion instead of
# this repository's ``scripts/`` directory.  Loading the module directly from its
# absolute path sidesteps that resolution entirely.
#
# ``sys.modules`` is populated *before* ``exec_module`` (the documented pattern
# for importing a source file directly) so that a function defined in this
# module -- e.g. the ProcessPoolExecutor worker used for multi-frame discovery
# -- can still be pickled by module+qualname and found by a child process.
# Without this, num_workers > 1 raises ModuleNotFoundError when unpickled.
_MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "discover_exp012_environment_config.py"
_SPEC = importlib.util.spec_from_file_location("discover_exp012_environment_config", _MODULE_PATH)
_discover_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _discover_module
_SPEC.loader.exec_module(_discover_module)

Exp012EnvironmentDiscoveryError = _discover_module.Exp012EnvironmentDiscoveryError
assemble_environment_config = _discover_module.assemble_environment_config
discover_complete_residue_environment = _discover_module.discover_complete_residue_environment


pytestmark = pytest.mark.cpu_only


def _protein_ligand_topology():
    import mdtraj as md

    topology = md.Topology()
    chain = topology.add_chain()

    ligand = topology.add_residue("LIG", chain, resSeq=1)
    topology.add_atom("C1", md.element.carbon, ligand)

    # Residue whose CA sits inside the cutoff but whose side-chain tip is far
    # outside it -- this is exactly the EXP-010 scenario that must still
    # return every atom of the residue, not just the close ones.
    near = topology.add_residue("ALA", chain, resSeq=2)
    topology.add_atom("CA", md.element.carbon, near)
    topology.add_atom("CB", md.element.carbon, near)

    far = topology.add_residue("GLY", chain, resSeq=3)
    topology.add_atom("CA", md.element.carbon, far)

    water = topology.add_residue("HOH", chain, resSeq=4)
    topology.add_atom("O", md.element.oxygen, water)
    topology.add_atom("H1", md.element.hydrogen, water)
    topology.add_atom("H2", md.element.hydrogen, water)

    return topology


def _index(topology, residue_name, atom_name):
    for residue in topology.residues:
        if residue.name == residue_name:
            for atom in residue.atoms:
                if atom.name == atom_name:
                    return atom.index
    raise AssertionError(f"atom not found: {residue_name}.{atom_name}")


def test_partial_overlap_residue_is_returned_whole_not_truncated():
    topology = _protein_ligand_topology()
    ligand_index = _index(topology, "LIG", "C1")
    near_ca = _index(topology, "ALA", "CA")
    near_cb = _index(topology, "ALA", "CB")

    frame = [[0.0, 0.0, 0.0] for _ in range(topology.n_atoms)]
    frame[ligand_index] = [0.0, 0.0, 0.0]
    frame[near_ca] = [0.3, 0.0, 0.0]  # within a 0.5 nm cutoff
    frame[near_cb] = [2.0, 0.0, 0.0]  # far outside the cutoff by itself
    far_ca = _index(topology, "GLY", "CA")
    frame[far_ca] = [5.0, 0.0, 0.0]

    candidates = discover_complete_residue_environment(
        topology,
        [frame],
        [ligand_index],
        edge_cutoff_angstrom=6.0,
        interaction_layers=2,
    )

    selected = next(candidate for candidate in candidates if "ALA" in candidate["stable_id"])
    assert selected["stable_id"].startswith("complete_residue:")
    assert selected["residue_atom_count"] == 2
    assert sorted(selected["atom_indices"]) == sorted([near_ca, near_cb])


def test_nearby_water_is_complete_candidate_but_ligand_residue_is_excluded():
    topology = _protein_ligand_topology()
    ligand_index = _index(topology, "LIG", "C1")
    near_ca = _index(topology, "ALA", "CA")

    frame = [[10.0, 10.0, 10.0] for _ in range(topology.n_atoms)]
    frame[ligand_index] = [0.0, 0.0, 0.0]
    frame[near_ca] = [0.1, 0.0, 0.0]
    water_o = _index(topology, "HOH", "O")
    frame[water_o] = [0.1, 0.1, 0.0]  # close enough to qualify if not excluded

    candidates = discover_complete_residue_environment(
        topology,
        [frame],
        [ligand_index],
        edge_cutoff_angstrom=6.0,
        interaction_layers=2,
    )

    stable_ids = {candidate["stable_id"] for candidate in candidates}
    assert any("HOH" in stable_id for stable_id in stable_ids)
    assert not any("LIG" in stable_id for stable_id in stable_ids)
    water_candidate = next(candidate for candidate in candidates if "HOH" in candidate["stable_id"])
    assert water_candidate["residue_atom_count"] == 3
    assert water_candidate["is_chain_terminal"] is False


def test_chain_terminal_residue_is_flagged():
    topology = _protein_ligand_topology()
    ligand_index = _index(topology, "LIG", "C1")
    far_ca = _index(topology, "GLY", "CA")

    frame = [[10.0, 10.0, 10.0] for _ in range(topology.n_atoms)]
    frame[ligand_index] = [0.0, 0.0, 0.0]
    frame[far_ca] = [0.1, 0.0, 0.0]

    candidates = discover_complete_residue_environment(
        topology,
        [frame],
        [ligand_index],
        edge_cutoff_angstrom=6.0,
        interaction_layers=2,
    )

    assert len(candidates) == 1
    assert candidates[0]["is_chain_terminal"] is True


def test_multi_frame_union_is_more_inclusive_than_any_single_frame():
    topology = _protein_ligand_topology()
    ligand_index = _index(topology, "LIG", "C1")
    near_ca = _index(topology, "ALA", "CA")
    far_ca = _index(topology, "GLY", "CA")

    frame_a = [[10.0, 10.0, 10.0] for _ in range(topology.n_atoms)]
    frame_a[ligand_index] = [0.0, 0.0, 0.0]
    frame_a[near_ca] = [0.1, 0.0, 0.0]

    frame_b = [[10.0, 10.0, 10.0] for _ in range(topology.n_atoms)]
    frame_b[ligand_index] = [0.0, 0.0, 0.0]
    frame_b[far_ca] = [0.1, 0.0, 0.0]

    only_a = discover_complete_residue_environment(
        topology,
        [frame_a],
        [ligand_index],
        edge_cutoff_angstrom=6.0,
        interaction_layers=2,
    )
    union = discover_complete_residue_environment(
        topology,
        [frame_a, frame_b],
        [ligand_index],
        edge_cutoff_angstrom=6.0,
        interaction_layers=2,
    )

    assert {c["stable_id"] for c in only_a} < {c["stable_id"] for c in union}


def test_parallel_and_serial_reference_frame_reduction_agree():
    # DEC-030(a) found real cutoff-graph neighbors reached by some but not all
    # of a trajectory's frames, motivating rebuilding derived_5a's manifest
    # from a union over many/all reference frames instead of just frame0. That
    # union is reduced by an elementwise per-atom minimum hop across frames --
    # order-independent, so forcing it through a process pool (num_workers=2)
    # must find exactly the same residues as the serial path (num_workers=1).
    topology = _protein_ligand_topology()
    ligand_index = _index(topology, "LIG", "C1")
    near_ca = _index(topology, "ALA", "CA")
    far_ca = _index(topology, "GLY", "CA")

    frame_a = [[10.0, 10.0, 10.0] for _ in range(topology.n_atoms)]
    frame_a[ligand_index] = [0.0, 0.0, 0.0]
    frame_a[near_ca] = [0.1, 0.0, 0.0]

    frame_b = [[10.0, 10.0, 10.0] for _ in range(topology.n_atoms)]
    frame_b[ligand_index] = [0.0, 0.0, 0.0]
    frame_b[far_ca] = [0.1, 0.0, 0.0]

    serial = discover_complete_residue_environment(
        topology,
        [frame_a, frame_b],
        [ligand_index],
        edge_cutoff_angstrom=6.0,
        interaction_layers=2,
        num_workers=1,
    )
    parallel = discover_complete_residue_environment(
        topology,
        [frame_a, frame_b],
        [ligand_index],
        edge_cutoff_angstrom=6.0,
        interaction_layers=2,
        num_workers=2,
    )

    assert serial == parallel


def test_two_hop_closure_is_not_a_twelve_angstrom_ligand_sphere():
    topology = _protein_ligand_topology()
    ligand_index = _index(topology, "LIG", "C1")
    near_ca = _index(topology, "ALA", "CA")
    near_cb = _index(topology, "ALA", "CB")
    far_ca = _index(topology, "GLY", "CA")
    water_o = _index(topology, "HOH", "O")
    water_h1 = _index(topology, "HOH", "H1")
    water_h2 = _index(topology, "HOH", "H2")

    frame = [[20.0, 20.0, 20.0] for _ in range(topology.n_atoms)]
    frame[ligand_index] = [0.0, 0.0, 0.0]
    # ALA is one hop from ligand; GLY is two hops via ALA.
    frame[near_ca] = [0.5, 0.0, 0.0]
    frame[near_cb] = [0.5, 0.1, 0.0]
    frame[far_ca] = [1.0, 0.0, 0.0]
    # This water is only 8 A from ligand, but has no 6 A path through S1.
    frame[water_o] = [-0.8, 0.0, 0.0]
    frame[water_h1] = [-0.8, 0.01, 0.0]
    frame[water_h2] = [-0.8, -0.01, 0.0]

    candidates = discover_complete_residue_environment(
        topology,
        [frame],
        [ligand_index],
        edge_cutoff_angstrom=6.0,
        interaction_layers=2,
    )

    by_residue = {candidate["atoms"][0]["residue_name"]: candidate for candidate in candidates}
    assert by_residue["ALA"]["minimum_graph_hop"] == 1
    assert by_residue["GLY"]["minimum_graph_hop"] == 2
    assert "HOH" not in by_residue


def test_non_positive_cutoff_fails_closed():
    topology = _protein_ligand_topology()
    ligand_index = _index(topology, "LIG", "C1")
    frame = [[0.0, 0.0, 0.0] for _ in range(topology.n_atoms)]
    with pytest.raises(Exp012EnvironmentDiscoveryError):
        discover_complete_residue_environment(
            topology,
            [frame],
            [ligand_index],
            edge_cutoff_angstrom=0.0,
            interaction_layers=2,
        )


def _write(path: Path, contents: bytes) -> tuple[str, str]:
    path.write_bytes(contents)
    return path.name, hashlib.sha256(contents).hexdigest()


def test_assembled_config_validates_through_the_manifest_builder(tmp_path):
    topology = _protein_ligand_topology()
    ligand_index = _index(topology, "LIG", "C1")
    near_ca = _index(topology, "ALA", "CA")
    near_cb = _index(topology, "ALA", "CB")

    frame = [[10.0, 10.0, 10.0] for _ in range(topology.n_atoms)]
    frame[ligand_index] = [0.0, 0.0, 0.0]
    frame[near_ca] = [0.2, 0.0, 0.0]
    frame[near_cb] = [0.3, 0.0, 0.0]

    candidates = discover_complete_residue_environment(
        topology,
        [frame],
        [ligand_index],
        edge_cutoff_angstrom=6.0,
        interaction_layers=2,
    )

    topology_name, topology_sha = _write(tmp_path / "topology.dat", b"topology")
    system_name, system_sha = _write(tmp_path / "system.dat", b"<System/>")
    box_name, box_sha = _write(tmp_path / "box.dat", b"1 0 0\n0 1 0\n0 0 1\n")

    config = assemble_environment_config(
        topology,
        [ligand_index],
        candidates,
        sources={
            "topology": (topology_name, topology_sha),
            "base_system": (system_name, system_sha),
            "box": (box_name, box_sha),
        },
    )

    manifest = build_environment_manifest(config, workspace_root=tmp_path)
    assert manifest["payload"]["ligand_indices"] == [ligand_index]
    assert set(manifest["payload"]["environment_candidate_indices"]) == {near_ca, near_cb}


def test_assembled_config_rejects_ligand_environment_overlap():
    topology = _protein_ligand_topology()
    ligand_index = _index(topology, "LIG", "C1")
    near_ca = _index(topology, "ALA", "CA")

    with pytest.raises(Exp012EnvironmentDiscoveryError):
        assemble_environment_config(
            topology,
            [ligand_index],
            [
                {
                    "atom_indices": [ligand_index, near_ca],
                }
            ],
            sources={
                "topology": ("t", "0" * 64),
                "base_system": ("s", "0" * 64),
                "box": ("b", "0" * 64),
            },
        )
