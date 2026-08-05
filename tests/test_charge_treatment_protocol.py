"""B2：`charge_treatment` 与 §1.2 的 fail-closed 矩阵（memtodolist.md §1.2 / §2.2 / §7.1）。

覆盖 §7.1 的配置与 fail-closed 组合，外加 §1.2 那 5 条"必须 fail closed"：

  1. `co_alchemical_charge_transfer` 且 `apbs_correction_kJ_mol != 0`（重复修正）
  2. `neutral` 但检测到配体净电荷不为 0
  3. `co_alchemical_charge_transfer` 但缺 co-ion 身份、参数或 restraint
  4. `rocklin_apbs_neutralizing_plasma` 但缺 APBS 来源说明/结果文件
  5. 配体电荷变化与 co-ion 电荷变化之和不为 0

以及 MEM-00a-2：`co_annihilation_experimental` 在膜体系一律 fail closed，
输出必须带 `experimental_not_for_production: true`。

## B3 已落地、B4 还没有（2026-08-04 更新）

`CHARGE_TRANSFER_HAMILTONIAN_IMPLEMENTED = True`：charging Hamiltonian 实现在
`ibs_engine.configure_charge_transfer_decharging`（ligand q→0 / co-ion 0→q，
restraint 换成 MEM-00d 的 flat-bottom 锚点相对形式），所以本层不再对合法的 co-ion
规格抛 `NotImplementedError`。原先那条"提醒改写的钩子"已经按设计触发过了。

`CHARGE_TRANSFER_SOLVENT_LEG_IMPLEMENTED = False`：溶剂腿 builder（§4.1）是 B4。
这个状态**不**在本层 raise —— 那会连 §6.4 要求的 pilot / λ 阶梯重估都做不了；
它作为 `solvent_leg_builder_implemented` / `closes_thermodynamic_cycle` 落进解析结果
与 provenance，真正的门在 `runabfe.build_and_cache_solvent_leg`（唯一一处）。
⚠️ 一条声明 charge-transfer 的运行在 B4 落地前**不得报出 ΔG_bind**。

## 对现有生产路径的影响

当前生产体系 Atenolol 是**中性**配体，走 `neutral`，本节全部检查对它是恒真的
（`test_neutral_ligand_is_the_current_production_path`）。落盘基线
181.00 / 157.84 / −5.535906 kcal/mol 不受影响。

⚠️ 但**带电配体的行为确实变了**：改动前带净电配体会静默走
`configure_coalchemical_neutral_decharging`（co-annihilation）跑完；现在必须显式
声明 `co_annihilation_experimental` 才允许，且该路线的数值不得进入 ΔG_bind 汇总。
这是 memtodolist §1.2 与 MEM-00a-2/00a-4 明确要求的门（旧带电配体数据一律作废）。

全部 CPU 可跑：纯校验层，不建 System、不建 Context。
"""

import pytest

pytestmark = pytest.mark.cpu_only

pytest.importorskip("openmm")

import abfe_core as core


# ---------------------------------------------------------------------------
# 构造器：一份合法的 co-ion 规格（§3.4 字段齐全）
# ---------------------------------------------------------------------------


def _coion(transferred_charge=+1.0, atom_index=12345, **overrides):
    spec = {
        "atom_index": atom_index,
        "residue_index": 4321,
        "residue_name": "NA",
        "element": "Na",
        # §2.2：λ=1 是中性 dummy，λ=0 才带电。
        "charge_at_lambda1_e": 0.0,
        "charge_at_lambda0_e": float(transferred_charge),
        "sigma_nm": 0.2439,
        "epsilon_kj_mol": 0.3658,
        "mass_amu": 22.99,
        "restraint": {
            "type": "flat_bottom",
            "flat_radius_nm": 0.5,
            "k_kj_per_mol_nm2": 100.0,
            "reference_fractional": [0.25, 0.25, 0.75],
        },
    }
    spec.update(overrides)
    return spec


# ---------------------------------------------------------------------------
# §7.1 主表
# ---------------------------------------------------------------------------


def test_neutral_ligand_with_neutral_treatment_passes():
    """§7.1 第 1 条：中性配体 + neutral → 通过。"""
    payload = core.resolve_charge_treatment("neutral", ligand_net_charge_e=0.0)
    assert payload["charge_treatment"] == "neutral"
    assert payload["ligand_net_charge_e"] == 0
    assert payload["apbs_applicable"] is False
    assert payload["apbs_applied"] is False
    assert payload["experimental_not_for_production"] is False
    assert payload["total_charge_conserved_at_every_lambda"] is True
    assert payload["protocol_version"] == core.CHARGE_TRANSFER_PROTOCOL_VERSION


def test_neutral_ligand_is_the_current_production_path():
    """不声明 charge_treatment 且配体中性 → neutral，即当前 Atenolol 生产路径。"""
    payload = core.resolve_charge_treatment(None, ligand_net_charge_e=0.0)
    assert payload["charge_treatment"] == "neutral"
    assert payload["was_defaulted_from_net_charge"] is True


def test_charged_ligand_with_neutral_treatment_fails():
    """§7.1 第 2 条 / §1.2 fail-closed #2：带电配体 + neutral → 失败。"""
    with pytest.raises(ValueError, match="但配体净电荷为"):
        core.resolve_charge_treatment("neutral", ligand_net_charge_e=1.0)


def test_charged_ligand_with_coion_passes_validation_and_reports_the_closed_cycle():
    """§7.1 第 3 条：带电配体 + co-ion → 校验通过。

    ⚠️ 2026-08-04 起这条断言变了一次：**B3（charging Hamiltonian）落地**，不再抛
    `NotImplementedError`。原先"分两步断言"的设计意图（把"规格校验过了"与
    "哈密顿量还没有"分开，好让 B3 落地时看得出当初是哪一环在挡）在那一刻兑现——
    挡住的那一环是哈密顿量，当时它有了。

    2026-08-05 起再变一次：**B4（溶剂腿 builder）也落地**了
    （`runabfe._insert_reserved_coalchemical_ion_dummies` +
    `build_and_cache_solvent_leg` 不再对 charge-transfer fail closed，见
    `tests/test_charge_transfer_hamiltonian.py::
    test_solvent_leg_builder_inserts_reserved_dummy_for_charge_transfer`）。
    循环闭得上了，解析结果必须如实反映这一点。
    """
    spec = _coion(+1.0)
    # 规格本身合法：单独校验不抛错。
    core._validate_co_alchemical_ion_spec(spec, 1)

    payload = core.resolve_charge_treatment(
        "co_alchemical_charge_transfer",
        ligand_net_charge_e=1.0,
        co_alchemical_ion=spec,
    )
    assert payload["charge_treatment"] == "co_alchemical_charge_transfer"
    assert payload["charging_hamiltonian_implemented"] is True
    assert core.CHARGE_TRANSFER_HAMILTONIAN_IMPLEMENTED is True
    # B4 已落地 ⟹ 循环闭得上，这一条必须能从解析结果直接读出来。
    assert payload["solvent_leg_builder_implemented"] is True
    assert payload["closes_thermodynamic_cycle"] is True
    # APBS 仍然一律为 0（co-ion 路线与 Rocklin 二选一，禁止双计数）。
    assert payload["apbs_applicable"] is False
    assert payload["apbs_correction_kJ_mol"] == 0.0
    assert (
        payload["apbs_not_applicable_reason"]
        == "not_applicable_co_alchemical_charge_transfer"
    )


def test_coion_with_nonzero_apbs_fails():
    """§7.1 第 4 条 / §1.2 fail-closed #1：co-ion + 非零 APBS → 失败（重复修正）。

    必须在"缺 co-ion"之前就拦下——双计数是协议错误，
    不该被别的检查顺序掩盖（检查顺序有意义，本文件另有测试钉住）。
    """
    with pytest.raises(ValueError, match="重复修正"):
        core.resolve_charge_treatment(
            "co_alchemical_charge_transfer",
            ligand_net_charge_e=1.0,
            apbs_correction_kJ_mol=-12.5,
            co_alchemical_ion=_coion(+1.0),
        )


def test_neutral_treatment_with_nonzero_apbs_fails():
    """§0 第 3 条：中性配体既不需要 co-ion，也不需要 Rocklin 净电荷修正。"""
    with pytest.raises(ValueError, match="apbs_correction_kJ_mol"):
        core.resolve_charge_treatment(
            "neutral", ligand_net_charge_e=0.0, apbs_correction_kJ_mol=3.0
        )


def test_charge_transfer_without_coion_fails():
    """显式要求旧版全局 spec 时仍可使用 fail-closed 校验。"""
    with pytest.raises(ValueError, match="没有提供"):
        core.resolve_charge_treatment(
            "co_alchemical_charge_transfer", ligand_net_charge_e=1.0,
            require_co_alchemical_ion=True,
        )


def test_charge_transfer_preflight_does_not_require_global_coion_spec():
    """B5：全局前置解析只解析路线，spec 由两条 pipeline 各自冻结。"""
    payload = core.resolve_charge_treatment(
        "co_alchemical_charge_transfer", ligand_net_charge_e=1.0
    )
    assert payload["charge_treatment"] == "co_alchemical_charge_transfer"
    assert payload["co_alchemical_ion"] is None


def test_charge_transfer_with_coion_missing_restraint_fails():
    """restraint 缺失单独算一条——§2.3 要求可审计的 flat-bottom restraint。"""
    spec = _coion(+1.0)
    spec["restraint"] = None
    with pytest.raises(ValueError, match="缺少 restraint"):
        core.resolve_charge_treatment(
            "co_alchemical_charge_transfer",
            ligand_net_charge_e=1.0,
            co_alchemical_ion=spec,
        )


@pytest.mark.parametrize("dropped", core.CO_ALCHEMICAL_ION_REQUIRED_FIELDS)
def test_charge_transfer_with_incomplete_coion_spec_fails(dropped):
    """§3.4 的每一个身份字段都是必需的，少一个就 fail closed。"""
    spec = _coion(+1.0)
    del spec[dropped]
    with pytest.raises(ValueError, match="缺少"):
        core.resolve_charge_treatment(
            "co_alchemical_charge_transfer",
            ligand_net_charge_e=1.0,
            co_alchemical_ion=spec,
        )


def test_rocklin_without_apbs_evidence_fails():
    """§1.2 fail-closed #4：只填一个数值不算证据。"""
    with pytest.raises(ValueError, match="缺少 APBS 证据字段"):
        core.resolve_charge_treatment(
            "rocklin_apbs_neutralizing_plasma",
            ligand_net_charge_e=1.0,
            apbs_correction_kJ_mol=-42.0,
        )


def test_rocklin_with_full_evidence_passes_and_marks_apbs_applied():
    evidence = {field: f"/path/{field}" for field in core.APBS_REQUIRED_EVIDENCE_FIELDS}
    evidence["net_charge_e"] = 1
    payload = core.resolve_charge_treatment(
        "rocklin_apbs_neutralizing_plasma",
        ligand_net_charge_e=1.0,
        apbs_correction_kJ_mol=-42.0,
        apbs_evidence=evidence,
    )
    assert payload["apbs_applicable"] is True
    assert payload["apbs_applied"] is True
    # 这条路线允许总电荷随 λ 改变——这正是它需要 Rocklin 修正的原因。
    assert payload["total_charge_conserved_at_every_lambda"] is False


def test_rocklin_forbids_coion():
    """两条路线二选一：同时给 co-ion 就是重复修正。"""
    evidence = {field: f"/path/{field}" for field in core.APBS_REQUIRED_EVIDENCE_FIELDS}
    with pytest.raises(ValueError, match="禁止创建"):
        core.resolve_charge_treatment(
            "rocklin_apbs_neutralizing_plasma",
            ligand_net_charge_e=1.0,
            co_alchemical_ion=_coion(+1.0),
            apbs_evidence=evidence,
        )


# ---------------------------------------------------------------------------
# §1.2 fail-closed #5：总电荷守恒的算术
# ---------------------------------------------------------------------------


def test_coion_transferring_wrong_amount_fails_charge_conservation():
    """配体 +2 只配一个 +1 co-ion → 电荷变化之和不为 0。"""
    with pytest.raises(ValueError, match="不为 0"):
        core._validate_co_alchemical_ion_spec(_coion(+1.0), 2)


def test_divalent_ligand_needs_two_monovalent_coions():
    """§2.2：|q_L| > 1 必须用多个单价 co-ion 分担，两个 +1 才配平 +2。"""
    core._validate_co_alchemical_ion_spec(
        [_coion(+1.0, atom_index=1), _coion(+1.0, atom_index=2)], 2
    )


def test_single_multivalent_coion_is_rejected():
    """不许把两个单位电荷集中到一个非物理多价粒子上。"""
    with pytest.raises(ValueError, match="最多转移一个单位电荷"):
        core._validate_co_alchemical_ion_spec(_coion(+2.0), 2)


def test_duplicate_coion_atom_index_is_rejected():
    with pytest.raises(ValueError, match="重复 atom_index"):
        core._validate_co_alchemical_ion_spec(
            [_coion(+1.0, atom_index=7), _coion(+1.0, atom_index=7)], 2
        )


def test_coion_must_be_neutral_dummy_at_lambda1():
    """§2.2：λ=1 时 co-ion 是"中性但保留 LJ 的 ion-shaped dummy"。"""
    spec = _coion(+1.0)
    spec["charge_at_lambda1_e"] = 1.0
    spec["charge_at_lambda0_e"] = 2.0
    with pytest.raises(ValueError, match="λ=1 的电荷应为 0"):
        core._validate_co_alchemical_ion_spec(spec, 1)


def test_coion_must_have_same_sign_as_ligand_not_opposite():
    """charge-transfer 不是 co-annihilation：co-ion 与配体**同号**。

    给正电配体配一个 0 → −1 的 co-ion 是把两种方法搞混了，必须报错。
    """
    with pytest.raises(ValueError, match="必须是同号"):
        core._validate_co_alchemical_ion_spec(_coion(-1.0), 1)
    with pytest.raises(ValueError, match="必须是同号"):
        core._validate_co_alchemical_ion_spec(_coion(+1.0), -1)


def test_negative_ligand_with_anionic_coion_passes():
    core._validate_co_alchemical_ion_spec(_coion(-1.0), -1)


# ---------------------------------------------------------------------------
# MEM-00a-2：co-annihilation 降级为实验对照
# ---------------------------------------------------------------------------


def test_co_annihilation_is_allowed_in_water_box_but_marked_experimental():
    payload = core.resolve_charge_treatment(
        "co_annihilation_experimental",
        ligand_net_charge_e=1.0,
        environment_type="soluble",
    )
    assert payload["experimental_not_for_production"] is True
    assert payload["charge_treatment"] == "co_annihilation_experimental"
    # 它逐 λ 守恒总电荷，所以 APBS 同样不适用。
    assert payload["apbs_applicable"] is False
    assert payload["total_charge_conserved_at_every_lambda"] is True


def test_co_annihilation_fails_closed_in_membrane():
    """MEM-00a-2：`system_type=membrane` + co-annihilation → 一律 fail closed。"""
    with pytest.raises(ValueError, match="不允许出现在膜体系中"):
        core.resolve_charge_treatment(
            "co_annihilation_experimental",
            ligand_net_charge_e=1.0,
            environment_type="membrane",
        )


def test_co_annihilation_with_nonzero_apbs_fails():
    with pytest.raises(ValueError, match="重复修正"):
        core.resolve_charge_treatment(
            "co_annihilation_experimental",
            ligand_net_charge_e=1.0,
            apbs_correction_kJ_mol=1.0,
        )


# ---------------------------------------------------------------------------
# 输入卫生
# ---------------------------------------------------------------------------


def test_misspelled_charge_treatment_fails_instead_of_falling_back():
    with pytest.raises(ValueError, match="不是合法值"):
        core.resolve_charge_treatment("co_alchemical", ligand_net_charge_e=1.0)


def test_non_integer_ligand_net_charge_is_an_input_error():
    """§2.2：非整数净电荷必须先作为输入错误调查，不要塞给分数价 co-ion。"""
    with pytest.raises(ValueError, match="不接近整数"):
        core.resolve_charge_treatment(None, ligand_net_charge_e=0.47)


def test_tiny_numerical_residue_in_net_charge_is_tolerated_as_neutral():
    """真实体系的电荷求和总有 1e-7 量级残差，不能因此判成带电。"""
    payload = core.resolve_charge_treatment(None, ligand_net_charge_e=-3.2e-7)
    assert payload["charge_treatment"] == "neutral"
    assert payload["ligand_net_charge_e"] == 0


def test_charged_ligand_defaults_to_charge_transfer_not_to_the_legacy_path():
    """不声明时带电配体默认 charge-transfer，而不是沿用现存的 co-annihilation。

    §1.2 的生产默认值。默认解析成 charge-transfer；spec 在每条 pipeline
    构建后分别冻结，不从全局 CLI 输入借用。
    """
    payload = core.resolve_charge_treatment(None, ligand_net_charge_e=1.0)
    assert payload["charge_treatment"] == "co_alchemical_charge_transfer"


def test_apbs_correction_must_be_finite_number():
    with pytest.raises(ValueError, match="不是数值|必须有限"):
        core.resolve_charge_treatment(
            "neutral", ligand_net_charge_e=0.0, apbs_correction_kJ_mol="很多"
        )
