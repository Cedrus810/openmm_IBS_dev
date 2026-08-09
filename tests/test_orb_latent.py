from __future__ import annotations

import os

import pytest

from local_residual.orb_latent import (
    OrbLatentError,
    OrbModelSpec,
    OrbParentConditioningContract,
    resolve_model_path,
)


def test_orb_model_spec_freezes_compile_and_layer_contract():
    with pytest.raises(OrbLatentError, match="compile=False"):
        OrbModelSpec(compile=True).validate()
    with pytest.raises(OrbLatentError, match="primary_layer"):
        OrbModelSpec(primary_layer=6).validate()


def test_parent_conditioning_contract_is_not_a_local_fragment_spin_claim():
    primary = OrbParentConditioningContract()
    primary.validate()
    assert primary.conditioning_scope == "parent_full_system"
    assert primary.to_dict()["interpretation"].startswith("closed-shell singlet conditioning")
    with pytest.raises(OrbLatentError, match="null/missing-spin"):
        OrbParentConditioningContract(spin_multiplicity=0.0).validate()
    sensitivity = OrbParentConditioningContract(spin_multiplicity=3.0, role="sensitivity")
    sensitivity.validate()


def test_cached_omol_path_is_resolved_without_network_when_runtime_is_available():
    pytest.importorskip("orb_models")
    path = resolve_model_path("orb-v3-conservative-omol")
    assert path.is_file()
    assert path.stat().st_size > 1_000_000


@pytest.mark.skipif(
    os.environ.get("ORB_RUN_REAL_TESTS") != "1",
    reason="set ORB_RUN_REAL_TESTS=1 for the real OMol shallow-prefix smoke",
)
def test_real_omol_prefix_is_256d_and_coordinate_differentiable():
    os.environ.setdefault("WARP_CACHE_PATH", "/tmp/atenolol_orb_warp_cache")
    import torch

    from local_residual.orb_latent import OrbLatentAdapter, OrbModelSpec

    adapter = OrbLatentAdapter(
        OrbModelSpec(model_name="orb-v3-conservative-omol", primary_layer=2)
    )
    result = adapter.extract_frame(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.7, 1.0, 0.0], [0.7, -1.0, 0.0]],
        [[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]],
        atomic_numbers=[6, 8, 1, 1],
        ligand_indices=[0, 1],
        total_charge=0.0,
        spin_multiplicity=1.0,
        require_coordinate_grad=True,
    )
    gradient = torch.autograd.grad(
        result.ligand_latent.square().mean(),
        result.batch.node_features["positions"],
    )[0]
    assert tuple(result.ligand_latent.shape) == (2, 256)
    assert torch.isfinite(result.ligand_latent).all()
    assert torch.isfinite(gradient).all()
