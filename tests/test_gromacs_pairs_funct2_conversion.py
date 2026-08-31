"""`[ pairs ]` funct 2 → funct 1：OpenMM 兼容性转换必须**可证等价**，否则 fail closed。

## 背景

OpenMM 的 `app.GromacsTopFile` 只接受 `[ pairs ]` funct 1
（`gromacstopfile.py::_processPair` 里 `if fields[2] != '1': raise`）。
CHARMM-GUI 的 AMBER FF-Converter 会对**部分**对写 funct 2：

    ai  aj  2  fudgeQQ  q1  q2  sigma14  epsilon14

实测 `memtest/toppar/POPC.itp` 有 **21 条**（共 356 条 pairs），其余 7 个文件一条没有。

## 为什么不能盲转

OpenMM 读的是 `fields[3:5]`。对 funct 1 那正好是 `sigma eps`；对 funct 2 会读成
`fudgeQQ q1` —— 所以它报错是**对的**，不是它太严。

funct 2 多出的三列只有在下面两条同时成立时才是冗余重述：

1. 逐对 `fudgeQQ` == `[ defaults ]` 的全局 fudgeQQ；
2. `q1`/`q2` == 该 moleculetype `[ atoms ]` 里的真实电荷。

因为 OpenMM 算 1-4 exception 的电荷用的是「粒子电荷 × 全局 fudgeQQ」
（`atom1params[0]*atom2params[0]*fudgeQQ`）。任一条不成立就意味着该对**真的**覆盖了
静电缩放或电荷，硬转会静默改变哈密顿量——所以那种情况必须 fail closed。

本文件既验证真实文件上的等价性，也验证两条不成立时确实拒绝转换。
"""

import filecmp
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

pytest.importorskip("openmm")

import abfe_core as core

ROOT = Path(__file__).absolute().parents[1]
MEMBRANE_TOP = ROOT / "tests/fixtures/memtest" / "topol.top"
POPC_ITP = ROOT / "tests/fixtures/memtest" / "toppar" / "POPC.itp"

# 实测基准
EXPECTED_FUNCT2_PAIRS = 21
EXPECTED_FILES_IN_TREE = 8


def _skip_if_missing(path: Path):
    if not path.is_file():
        pytest.skip(f"{path} 不在，跳过（该文件应随仓库提供）")


# ---------------------------------------------------------------------------
# 检测
# ---------------------------------------------------------------------------


def test_membrane_topology_is_detected_as_needing_conversion():
    _skip_if_missing(MEMBRANE_TOP)
    assert core.gromacs_topology_has_funct2_pairs(str(MEMBRANE_TOP)) is True


def test_existing_soluble_topology_does_not_need_conversion():
    """回归：现有可溶体系没有 funct-2 pairs，不该被这条路径碰到。"""
    soluble = ROOT / "tests/fixtures/topol.top"
    _skip_if_missing(soluble)
    assert core.gromacs_topology_has_funct2_pairs(str(soluble)) is False


def test_only_popc_carries_funct2_pairs_in_this_system():
    """实测：21 条全在 POPC.itp，其余文件一条没有。

    ⚠️ 这里必须用**非递归**的 `gromacs_file_has_funct2_pairs`。
    `gromacs_topology_has_funct2_pairs` 是整棵树语义——对顶层 `topol.top` 调用它会
    因为 include 了 `POPC.itp` 而返回 True。那是它该有的行为（runabfe 要问的是
    "这棵树需不需要转换"），但回答不了"哪个文件自己带"。
    """
    _skip_if_missing(MEMBRANE_TOP)
    files = core.gromacs_topology_files(str(MEMBRANE_TOP))
    assert len(files) == EXPECTED_FILES_IN_TREE
    carriers = [
        Path(f).name for f in files if core.gromacs_file_has_funct2_pairs(f)
    ]
    assert carriers == ["POPC.itp"]


def test_tree_scope_and_file_scope_predicates_differ_as_documented():
    """整棵树语义 vs 单文件语义：顶层 `.top` 自己不带，但树里有。"""
    _skip_if_missing(MEMBRANE_TOP)
    assert core.gromacs_file_has_funct2_pairs(str(MEMBRANE_TOP)) is False
    assert core.gromacs_topology_has_funct2_pairs(str(MEMBRANE_TOP)) is True
    assert core.gromacs_file_has_funct2_pairs(str(POPC_ITP)) is True


# ---------------------------------------------------------------------------
# 转换：等价、局部、不动原文件
# ---------------------------------------------------------------------------


def test_conversion_rewrites_exactly_the_funct2_lines_and_nothing_else(tmp_path):
    _skip_if_missing(MEMBRANE_TOP)
    result = core.convert_gromacs_pairs_funct2(str(MEMBRANE_TOP), str(tmp_path))

    assert result["n_pairs_converted"] == EXPECTED_FUNCT2_PAIRS
    assert result["n_files_copied"] == EXPECTED_FILES_IN_TREE
    assert [Path(f["source"]).name for f in result["patched_files"]] == ["POPC.itp"]
    assert result["global_fudge_qq"] == pytest.approx(0.833333, abs=1e-6)
    assert result["conversion_version"] == core.GROMACS_PAIRS_FUNCT2_CONVERSION_VERSION

    # 转换后不应再有 funct 2 —— 否则 OpenMM 还是会拒。
    assert core.gromacs_topology_has_funct2_pairs(result["converted_top_path"]) is False

    # 只改那 21 行，行数不变，其余逐行相同。
    original = POPC_ITP.read_text(encoding="utf-8").splitlines()
    converted = (tmp_path / "toppar" / "POPC.itp").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(original) == len(converted)
    changed = [i for i, (a, b) in enumerate(zip(original, converted)) if a != b]
    assert len(changed) == EXPECTED_FUNCT2_PAIRS


def test_untouched_files_are_byte_identical(tmp_path):
    """没有 funct-2 的文件必须逐字节拷贝，不做任何"顺手整理"。"""
    _skip_if_missing(MEMBRANE_TOP)
    core.convert_gromacs_pairs_funct2(str(MEMBRANE_TOP), str(tmp_path))
    for relative in (
        "topol.top",
        "Atenolol-rank11.itp",
        "toppar/PROA.itp",
        "toppar/TP3.itp",
        "toppar/forcefield.itp",
        "toppar/Na+.itp",
        "toppar/Cl-.itp",
    ):
        source = ROOT / "tests/fixtures/memtest" / relative
        if not source.is_file():
            continue
        assert filecmp.cmp(source, tmp_path / relative, shallow=False), relative


def test_inactive_preprocessor_blocks_are_preserved(tmp_path):
    """`#ifdef POSRES` / `#ifdef DIHRES` 必须原样保留——不展开、不丢弃。

    位置限制就藏在这些块里（`POPC.itp` 与 `PROA.itp` 各有），
    丢掉它们等于悄悄改变了"能不能做位置限制阶梯"这件事。
    """
    _skip_if_missing(MEMBRANE_TOP)
    core.convert_gromacs_pairs_funct2(str(MEMBRANE_TOP), str(tmp_path))
    converted = (tmp_path / "toppar" / "POPC.itp").read_text(encoding="utf-8")
    assert "#ifdef POSRES" in converted
    assert "#ifdef DIHRES" in converted
    assert converted.count("#endif") == 2


def test_converted_lines_keep_sigma_epsilon_in_the_columns_openmm_reads(tmp_path):
    """OpenMM 取 `fields[3:5]`，所以转换后第 4/5 列必须是 sigma14/epsilon14。"""
    _skip_if_missing(MEMBRANE_TOP)
    core.convert_gromacs_pairs_funct2(str(MEMBRANE_TOP), str(tmp_path))
    lines = [
        line
        for line in (tmp_path / "toppar" / "POPC.itp").read_text(
            encoding="utf-8"
        ).splitlines()
        if "[pairs funct 2 -> 1]" in line
    ]
    assert len(lines) == EXPECTED_FUNCT2_PAIRS
    for line in lines:
        fields = line.split(";", 1)[0].split()
        assert len(fields) == 5, fields
        assert fields[2] == "1"
        assert float(fields[3]) == pytest.approx(3.39966950842e-01)
        assert float(fields[4]) == pytest.approx(7.62882666667e-02)
    # 原始三列必须留在注释里，转换是可审计的。
    assert "fudgeQQ=0.833333" in lines[0]


# ---------------------------------------------------------------------------
# fail closed：两条等价条件任一不成立即拒绝
# ---------------------------------------------------------------------------


def _synthetic_topology(tmp_path, pair_line, fudge_qq="0.8333"):
    top = tmp_path / "topol.top"
    top.write_text(
        f"[ defaults ]\n1 2 yes 0.5 {fudge_qq}\n"
        "[ moleculetype ]\nFOO 3\n"
        "[ atoms ]\n"
        "1 CT 1 FOO C1 1 -0.100000 12.01\n"
        "2 CT 1 FOO C2 2 -0.200000 12.01\n"
        "[ pairs ]\n"
        f"{pair_line}\n"
        "[ molecules ]\nFOO 1\n",
        encoding="utf-8",
    )
    return top


def test_equivalent_synthetic_pair_converts(tmp_path):
    top = _synthetic_topology(tmp_path, "1 2 2 0.8333 -0.100000 -0.200000 0.34 0.076")
    out = tmp_path / "out"
    result = core.convert_gromacs_pairs_funct2(str(top), str(out))
    assert result["n_pairs_converted"] == 1


def test_pair_overriding_fudge_qq_fails_closed(tmp_path):
    """逐对 fudgeQQ ≠ 全局 → OpenMM 只会用全局值，硬转会静默改变静电缩放。"""
    top = _synthetic_topology(tmp_path, "1 2 2 0.5000 -0.100000 -0.200000 0.34 0.076")
    with pytest.raises(ValueError, match="覆盖了 fudgeQQ"):
        core.convert_gromacs_pairs_funct2(str(top), str(tmp_path / "out"))


def test_pair_overriding_atom_charge_fails_closed(tmp_path):
    """q1/q2 ≠ `[ atoms ]` 电荷 → OpenMM 用的是粒子电荷，硬转会静默改变哈密顿量。"""
    top = _synthetic_topology(tmp_path, "1 2 2 0.8333 -0.999000 -0.200000 0.34 0.076")
    with pytest.raises(ValueError, match="覆盖了原子 1 的电荷"):
        core.convert_gromacs_pairs_funct2(str(top), str(tmp_path / "out"))


def test_truncated_funct2_line_fails_closed(tmp_path):
    top = _synthetic_topology(tmp_path, "1 2 2 0.8333 -0.100000")
    with pytest.raises(ValueError, match="字段不足 8 个"):
        core.convert_gromacs_pairs_funct2(str(top), str(tmp_path / "out"))


def test_missing_global_fudge_qq_refuses_to_guess(tmp_path):
    """没有 `[ defaults ]` 的 fudgeQQ 就无从校验等价性 —— 拒绝盲转。"""
    top = tmp_path / "topol.top"
    top.write_text(
        "[ moleculetype ]\nFOO 3\n"
        "[ atoms ]\n1 CT 1 FOO C1 1 -0.1 12.01\n2 CT 1 FOO C2 2 -0.2 12.01\n"
        "[ pairs ]\n1 2 2 0.8333 -0.1 -0.2 0.34 0.076\n"
        "[ molecules ]\nFOO 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="找不到 `\\[ defaults \\]` 的 fudgeQQ"):
        core.convert_gromacs_pairs_funct2(str(top), str(tmp_path / "out"))


def test_original_input_is_never_modified(tmp_path):
    """原始输入一个字节都不能动 —— 缓存指纹与可追溯性都靠它。"""
    _skip_if_missing(MEMBRANE_TOP)
    before = POPC_ITP.read_bytes()
    core.convert_gromacs_pairs_funct2(str(MEMBRANE_TOP), str(tmp_path))
    assert POPC_ITP.read_bytes() == before


# ---------------------------------------------------------------------------
# 接线
# ---------------------------------------------------------------------------


PRODUCTION_FILES = (
    "abfe_core.py",
    "abfe_pipeline.py",
    "runabfe.py",
    "ibs_engine.py",
    "abfe_preoptimizer.py",
)


def test_no_production_file_calls_GromacsTopFile_directly():
    """全仓只允许**一个**拓扑加载入口。

    这是本轮最贵的教训：第一次修 funct-2 时只在 `build_system_from_gromacs` 一处
    接了转换，`build_and_cache_solvent_leg` 里另一个裸调 `app.GromacsTopFile` 的
    地方照样炸——**与 B1 当初只接了 1 个 `ABFEPipeline` 构造点是同一个毛病**：
    同一件事有多个入口，补一个漏一片。

    所以这条契约钉死：除 `abfe_core.load_gromacs_topology_for_openmm` 内部那一处，
    任何生产文件都不得裸调 `app.GromacsTopFile(`。新加加载点会在这里失败。
    """
    offenders = []
    for name in PRODUCTION_FILES:
        path = ROOT / name
        if not path.is_file():
            continue
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if "app.GromacsTopFile(" not in line:
                continue
            # 唯一豁免：入口函数自己那一行。
            if name == "abfe_core.py" and "return app.GromacsTopFile(resolved" in line:
                continue
            offenders.append(f"{name}:{lineno}: {line.strip()}")
    assert offenders == [], (
        "这些地方绕过了唯一入口，含 funct-2 pairs 的拓扑会在此处炸：\n"
        + "\n".join(offenders)
        + "\n改用 abfe_core.load_gromacs_topology_for_openmm()。"
    )


def test_every_known_loading_site_goes_through_the_single_entry():
    """反向断言：曾经出问题的那几个函数确实都在用唯一入口。"""
    runabfe_src = (ROOT / "runabfe.py").read_text(encoding="utf-8")
    core_src = (ROOT / "abfe_core.py").read_text(encoding="utf-8")
    pipeline_src = (ROOT / "abfe_pipeline.py").read_text(encoding="utf-8")

    for name, source, needle in (
        ("generate_ligand_xml_from_top", runabfe_src, "def generate_ligand_xml_from_top"),
        ("build_and_cache_solvent_leg", runabfe_src, "def build_and_cache_solvent_leg"),
        ("load_native_system", runabfe_src, "def load_native_system"),
        ("build_system_from_gromacs", runabfe_src, "def build_system_from_gromacs"),
        ("build_solvent_system", core_src, "def build_solvent_system"),
    ):
        body = source.split(needle)[1].split("\ndef ")[0]
        assert "load_gromacs_topology_for_openmm(" in body, f"{name} 没走唯一入口"
    assert "load_gromacs_topology_for_openmm(" in pipeline_src


def test_single_entry_is_idempotent_and_content_addressed(tmp_path):
    """同一份输入重复调用只转换一次；转换产物可复用。"""
    _skip_if_missing(MEMBRANE_TOP)
    first_path, first_info = core.openmm_compatible_gromacs_top(
        str(MEMBRANE_TOP), compat_dir=str(tmp_path)
    )
    assert first_info is not None
    assert first_info["n_pairs_converted"] == EXPECTED_FUNCT2_PAIRS

    second_path, second_info = core.openmm_compatible_gromacs_top(
        str(MEMBRANE_TOP), compat_dir=str(tmp_path)
    )
    assert second_path == first_path
    assert second_info["reused_existing_conversion"] is True


def test_reuse_requires_a_complete_and_matching_conversion(tmp_path):
    """复用转换产物前必须核对**整棵树**完整且与当前输入一致。

    早先这里只检查"顶层 top 存在 && 树里没有 funct-2"就直接复用——那是 fail-open：
    一次被中断的转换会留下半成品（部分文件已拷、部分没拷，或某个 `.itp` 被截断），
    而那两个条件仍可能成立。于是 OpenMM 读到一棵**不自洽**的拓扑、建出参数错乱的
    System，最小化后 PE 到 1e13、几千步后变成一个没有上下文的
    `Particle coordinate is NaN`（实测就踩了这个坑）。
    """
    _skip_if_missing(MEMBRANE_TOP)
    compat = tmp_path / "compat"

    _, first = core.openmm_compatible_gromacs_top(
        str(MEMBRANE_TOP), None, str(compat)
    )
    assert first["n_pairs_converted"] == EXPECTED_FUNCT2_PAIRS
    assert Path(first["manifest_path"]).is_file()

    # 完整且一致 → 复用
    _, second = core.openmm_compatible_gromacs_top(
        str(MEMBRANE_TOP), None, str(compat)
    )
    assert second["reused_existing_conversion"] is True

    # 产物被改过 → 必须重转
    tampered = compat / "toppar" / "forcefield.itp"
    tampered.write_text(
        tampered.read_text(encoding="utf-8") + "\n; tampered\n", encoding="utf-8"
    )
    _, third = core.openmm_compatible_gromacs_top(
        str(MEMBRANE_TOP), None, str(compat)
    )
    assert third.get("reused_existing_conversion") is None
    assert third["n_pairs_converted"] == EXPECTED_FUNCT2_PAIRS


def test_reuse_is_rejected_when_a_converted_file_is_missing(tmp_path):
    """半成品（中断的转换）不得被复用。"""
    _skip_if_missing(MEMBRANE_TOP)
    compat = tmp_path / "compat"
    core.openmm_compatible_gromacs_top(str(MEMBRANE_TOP), None, str(compat))
    (compat / "toppar" / "TP3.itp").unlink()
    _, again = core.openmm_compatible_gromacs_top(
        str(MEMBRANE_TOP), None, str(compat)
    )
    assert again.get("reused_existing_conversion") is None


def test_reuse_is_rejected_without_a_manifest(tmp_path):
    """没有 manifest 就无从核对 → 不复用（旧目录不会被盲信）。"""
    _skip_if_missing(MEMBRANE_TOP)
    compat = tmp_path / "compat"
    core.openmm_compatible_gromacs_top(str(MEMBRANE_TOP), None, str(compat))
    (compat / "conversion_manifest.json").unlink()
    _, again = core.openmm_compatible_gromacs_top(
        str(MEMBRANE_TOP), None, str(compat)
    )
    assert again.get("reused_existing_conversion") is None


def test_single_entry_is_a_no_op_when_no_funct2_pairs():
    """可溶体系不该被这条路径碰到：原路返回、不产生任何转换目录。"""
    soluble = ROOT / "tests/fixtures/topol.top"
    _skip_if_missing(soluble)
    path, info = core.openmm_compatible_gromacs_top(str(soluble))
    assert path == str(soluble)
    assert info is None


def test_conversion_version_enters_the_main_system_cache_identity():
    """改了转换逻辑必须让 System 缓存失效，否则是静默串协议。"""
    source = (ROOT / "runabfe.py").read_text(encoding="utf-8")
    assert '"gromacs_pairs_funct2_conversion_version": (' in source
    identity_block = source.split("def _main_cache_identity")[1].split("\ndef ")[0]
    assert "GROMACS_PAIRS_FUNCT2_CONVERSION_VERSION" in identity_block


def test_runabfe_uses_the_converted_topology_only_for_openmm():
    """交给 OpenMM 的是转换后的 top，但配体名/坐标与缓存指纹仍用原始输入。"""
    source = (ROOT / "runabfe.py").read_text(encoding="utf-8")
    # 主路径改走统一入口 `openmm_compatible_gromacs_top`（幂等 + 内容寻址），
    # 不再自己判断+调用 convert_*；这条断言随之更新。
    assert "_top_for_openmm, _pairs_conversion = openmm_compatible_gromacs_top(" in source
    assert "config.gro, _top_for_openmm, config.ligand," in source
    # 指纹算的是原始 config.top，不是转换产物。
    identity_call = source.split("def _main_cache_identity")[1].split("\ndef ")[0]
    assert "_top_for_openmm" not in identity_call
    # 转换结果要进 provenance，便于审计。
    assert 'provenance["gromacs_pairs_funct2_conversion"] = pairs_conversion' in source
