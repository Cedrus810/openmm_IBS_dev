"""可溶蛋白口袋里的配体：LRC 照加，但**不许声称达成**力场参数化条件。

`resolve_dispersion_protocol` 那道 fail-closed 的判据是 `is_membrane`，
而它引用的论证是：

    「`lj_tail_lrc_coeff[k]/V(t)` 假设配体周围是**均匀体相密度**；
      配体埋在口袋里时这个假设直接不成立」

这段话对**可溶蛋白的口袋**同样成立 —— 复合物腿里配体的第一/第二壳层是蛋白，
而 `coeff/V` 用的是盒平均密度。判据卡的是**环境类型**，论证讲的是**配体待在哪**。

⚠️ **不改数值行为**。brd4_ligand1 两条腿离线复算（2026-09-15，零 GPU）：

    complex  N_env=39633  V=416.9  ρ=95.07  E_LRC=−13.2983 kJ/mol
    solvent  N_env= 5914  V= 62.3  ρ=94.87  E_LRC=−12.7079 kJ/mol
    两腿差 −0.5904 kJ/mol（÷56 个有 LJ 的配体原子 = −0.0105/原子）

6.7 倍体积差被 `coeff ∝ N_env` 抵消（ρ_env 只差 0.2%）⟹ 关掉它反而是一个更大的、
没验证过的改动。所以只改**记账**：`target_met` 不再无条件报 True。
"""
import inspect

import abfe_core as core
from abfe_core import (
    LIGAND_SURROUNDING_BULK,
    LIGAND_SURROUNDING_POCKET,
    resolve_leg_dispersion_implementation as resolve,
)
import pytest

pytestmark = pytest.mark.cpu_only

LEGACY = "legacy_uniform_density_lrc"


def test_a_pocket_leg_does_not_claim_the_target_is_met():
    r = resolve(LEGACY, "soluble", LIGAND_SURROUNDING_POCKET)
    assert r["target_met"] is False
    assert r["ligand_environment_is_uniform_bulk"] is False
    assert "ligand_is_in_a_pocket" in r["reason"]


def test_the_correction_is_still_applied():
    """**不改数值行为** —— 实测两腿只差 ~0.6 kJ/mol，关掉是更大的未验证改动。"""
    for sur in (None, LIGAND_SURROUNDING_BULK, LIGAND_SURROUNDING_POCKET):
        assert resolve(LEGACY, "soluble", sur)["alchemical_uniform_density_lrc"] is True


def test_a_bulk_leg_is_unchanged():
    """纯水溶剂腿里配体周围就是均匀体相 ⟹ 修正成立、target_met 仍为 True。"""
    r = resolve(LEGACY, "soluble", LIGAND_SURROUNDING_BULK)
    assert r["target_met"] is True
    assert r["reason"] == ""


def test_not_declaring_the_surrounding_keeps_the_old_behaviour():
    """老调用方（不传第三个参数）**逐字不变** —— 这条改动不许追溯影响既有产物。"""
    r = resolve(LEGACY, "soluble")
    assert r["target_met"] is True
    assert r["ligand_environment_is_uniform_bulk"] is True
    assert r["reason"] == ""


def test_membrane_plus_legacy_is_still_fail_closed():
    """膜那道硬闸**没被放松** —— 它在 `resolve_dispersion_protocol` 里，仍然 raise。"""
    src = inspect.getsource(core)
    assert "system_type=membrane 不得使用 legacy_uniform_density_lrc" in src


def test_the_coefficients_are_persisted_for_offline_audit():
    """系数先前只活在内存里，核对一次要从 XML 重建整套 σ/ε 重算。"""
    import ibs_engine as ie
    src = inspect.getsource(ie)
    assert '"lj_tail_lrc_coeff_kj_nm3_per_mol"' in src
    # 没有 V 那串系数换算不成能量（每帧修正 = coeff/V）
    assert '"lj_tail_lrc_box_volume_nm3"' in src
