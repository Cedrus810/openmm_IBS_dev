"""Focused regressions for the verified, non-DEXP TODO fixes."""

import json

import numpy as np
import pytest
from openmm import NonbondedForce, System, Vec3, app, unit

import ibs_engine as ie
import runabfe
from abfe_pipeline import _pre_equilibration_fingerprint



pytestmark = pytest.mark.cpu_only

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


def test_pre_equilibration_fingerprint_ignores_pose_and_box_but_binds_step_budget():
    """预平衡身份**不含**坐标/盒子，但仍含目标步数。

    ⚠️ 本测试 2026-09-09 从「换 pose / 换盒子必须让指纹变」**反转**而来。别改回去。

    旧契约把可变状态当身份，与本项目自己的行为直接冲突：PBC-01（同日）按设计
    修坐标（把被 mmCIF 假键撕开的分子拼回去，~0.005 nm 质心平移）⟹ 旧契约要求
    所有已有 run 目录的预平衡 checkpoint 因此作废、5M 步从零重跑。实测已发生
    （cyclod_ligand1/rep1）。

    换输入的绑定在 `runabfe.system_cache_exists()`（哈希的是用户的 gro/top/box
    文件本身，不是我们生成的中间产物）；"跑完没有"看 `pipeline_state.json`。
    """
    system = System()
    system.addParticle(12.0)
    positions = np.asarray([[0.1, 0.2, 0.3]]) * unit.nanometer
    moved = np.asarray([[0.2, 0.2, 0.3]]) * unit.nanometer
    box = [
        Vec3(2.0, 0.0, 0.0),
        Vec3(0.0, 2.0, 0.0),
        Vec3(0.0, 0.0, 2.0),
    ] * unit.nanometer
    changed_box = [
        Vec3(2.1, 0.0, 0.0),
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
    assert baseline == _pre_equilibration_fingerprint(
        positions=moved, requested_steps=1000, **common
    ), "坐标不得参与身份"
    assert baseline == _pre_equilibration_fingerprint(
        positions=positions,
        requested_steps=1000,
        **{**common, "box_vectors": changed_box},
    ), "盒矢量不得参与身份"

    # 仍然属于身份的：目标步数、温度、压力、配体索引。
    assert baseline != _pre_equilibration_fingerprint(
        positions=positions, requested_steps=2000, **common
    ), "目标步数必须抓到（短平衡不得冒充长平衡）"
    assert baseline != _pre_equilibration_fingerprint(
        positions=positions,
        requested_steps=1000,
        **{**common, "temperature": 310.0},
    ), "温度必须抓到"
    assert baseline != _pre_equilibration_fingerprint(
        positions=positions,
        requested_steps=1000,
        **{**common, "ligand_indices": []},
    ), "配体索引必须抓到"


def test_no_self_generated_hash_may_enter_a_cache_identity():
    """本仓库**自己生成的产物/代码**的 sha256 一律不得进缓存身份。

    这条契约 2026-08-24 的 code_sha256 复盘里就写着"应该做但没时间做"，
    结果同一个坑又踩了两次。断的是**产物**（那五个 helper 恒返回 None），
    比断某个具体字段名更耐重构 —— 字段可以改名，helper 的语义不会变。

    复发记录：
      * 2026-08-24  `code_sha256`      —— 改任一行代码 ⟹ resume 被迫重跑 GPU
      * 2026-09-09  `system_xml_hash`  —— 排除表灌入顺序换成 sorted() ⟹ 字节变
      * 2026-09-09  `positions_sha256` —— PBC-01 按设计修坐标 ⟹ 身份翻脸

    ⚠️ 用户输入文件的哈希（`runabfe._sha256_file` 绑 gro/top/box）**不在此列**，
    那是防"换了输入还复用缓存"，不会因为我们改代码而翻脸。
    """
    import abfe_pipeline as _P

    for name in (
        "_code_hash",
        "_system_xml_hash",
        "_topology_hash",
        "_positions_hash",
        "_box_vectors_hash",
    ):
        fn = getattr(_P, name)
        try:
            value = fn()
        except TypeError:
            value = fn(None)
        assert value is None, (
            f"abfe_pipeline.{name}() 又开始返回真实哈希了。它哈希的是我们自己生成的"
            "产物/代码，进了缓存身份就意味着：我们改一行代码、或做一个按设计修坐标的"
            "修复，用户已经烧掉的 GPU 时间全部作废。这个坑已经踩过三次，别再踩。"
        )


def _write_main_cache(tmp_path, monkeypatch, *, identity_sha256="expected"):
    """铺一份形状合法的主 System 缓存，返回 manifest 的可变副本。"""
    for name, content in {
        "system_native.xml": "system",
        "ligand_indices.json": json.dumps({"ligand_indices": [0]}),
        "topology.cif": "topology",
    }.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    np.save(tmp_path / "box_vectors.npy", np.eye(3))

    identity = {"identity_sha256": identity_sha256}
    monkeypatch.setattr(runabfe, "_main_cache_identity", lambda *args: identity)
    manifest = {
        "protocol_version": runabfe.MAIN_SYSTEM_CACHE_PROTOCOL_VERSION,
        **identity,
    }
    (tmp_path / "system_cache_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return manifest


def _rewrite_manifest(tmp_path, manifest):
    (tmp_path / "system_cache_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# 主 System 缓存的身份：**只有用户输入**
# ---------------------------------------------------------------------------
#
# ⚠️ 本组测试 2026-09-09 从「篡改 topology.cif / box_vectors.npy 必须让缓存失效」
# **反转**而来，2026-09-10 补齐。别改回去。
#
# 旧契约把我们自己生成的产物当身份。它与本项目自己的行为直接冲突：PBC-01 按设计
# 重写 topology.cif（删掉 mmCIF 往返造出来的假键）、按设计修坐标 ⟹ 旧契约要求
# 所有已有 run 的主缓存因此作废、从建系开始重来。实测已发生。
# 这是同一形状的第四次复发，规则收紧成：**只有用户输入才配做身份。**
#
# 现在的身份 = `identity_sha256`（用户的 gro/top/--ligand/gmx_include）
#            + `protocol_version` + charge route。
#
# ⚠️ **代价说清楚**：手改 topology.cif 之后缓存照收。这是**故意**的取舍——
# 那份文件是我们自己写出来的中间产物，不是用户给的输入；归组只信 System
# （`system_molecule_grouping`），载入时还会删假键。要防手改就得换成**语义**身份
# （原子数 + 元素序列），而不是回到比字节。


def test_main_cache_accepts_when_only_self_produced_artifacts_changed(
    tmp_path, monkeypatch
):
    """篡改 topology.cif **不**让缓存失效 —— 它不是身份的一部分。

    这条看起来像在放水，其实是在钉住"别再把自产产物塞回身份"。
    """
    _write_main_cache(tmp_path, monkeypatch)
    assert runabfe.system_cache_exists(str(tmp_path))

    (tmp_path / "topology.cif").write_text("tampered", encoding="utf-8")
    assert runabfe.system_cache_exists(str(tmp_path)), (
        "topology.cif 又进缓存身份了。PBC-01 按设计重写这个文件，"
        "把它当身份 = 每次修假键都让用户的建系缓存作废。已踩过四次。"
    )

    np.save(tmp_path / "box_vectors.npy", np.eye(3) * 2.0)
    assert runabfe.system_cache_exists(str(tmp_path)), (
        "box_vectors.npy 又进缓存身份了 —— 同上，它也是我们自己写出来的。"
    )


def test_main_cache_still_rejects_a_changed_user_input(tmp_path, monkeypatch):
    """反过来：用户真换了 gro/top/--ligand，必须拒。否则这层保护就是零。"""
    _write_main_cache(tmp_path, monkeypatch, identity_sha256="expected")
    assert runabfe.system_cache_exists(str(tmp_path))

    monkeypatch.setattr(
        runabfe, "_main_cache_identity", lambda *args: {"identity_sha256": "different"}
    )
    assert not runabfe.system_cache_exists(str(tmp_path))


def test_main_cache_rejects_a_stale_protocol_version(tmp_path, monkeypatch):
    manifest = _write_main_cache(tmp_path, monkeypatch)
    manifest["protocol_version"] = runabfe.MAIN_SYSTEM_CACHE_PROTOCOL_VERSION - 1
    _rewrite_manifest(tmp_path, manifest)
    assert not runabfe.system_cache_exists(str(tmp_path))


def test_main_cache_rejects_when_a_required_artifact_is_missing(
    tmp_path, monkeypatch
):
    """产物不进身份，但**存在性**仍然是硬条件 —— 少一个文件下游就装不起来。"""
    _write_main_cache(tmp_path, monkeypatch)
    assert runabfe.system_cache_exists(str(tmp_path))

    (tmp_path / "topology.cif").unlink()
    assert not runabfe.system_cache_exists(str(tmp_path))


def test_main_cache_rejects_an_unreadable_manifest(tmp_path, monkeypatch):
    _write_main_cache(tmp_path, monkeypatch)
    (tmp_path / "system_cache_manifest.json").write_text("{ not json", encoding="utf-8")
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
