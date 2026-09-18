"""带电配体派生 co-ion 拓扑后，读拓扑的人必须读派生件。

真机 `thrombin_ligand1/rep1`（2026-09-18 17:49）：预平衡跑完 2,000,000 步之后
死在 Boresch 锚点估算 —— `GROMACS 与轨迹拓扑原子数/配体索引不匹配`。
根因是净电荷 +1 的复合物腿派生了带 reserved co-ion dummy 的 `.gro`/`.top`
（原始 43761 原子 → 派生 43759，删一个水加一个 Na），而 Boresch 仍拿
`config.top`（原始件）去读。

⚠️ 身份仍然是原始件：派生件是本次运行的**产物**，不配做缓存身份。
"""

import json
import os

import pytest

import runabfe


def _make_derivation(tmp_path, derived_name="complex.top", create_file=True):
    out = tmp_path / "run"
    derived_dir = out / runabfe.RESERVED_COION_DERIVED_DIRNAME
    derived_dir.mkdir(parents=True)
    derived_top = derived_dir / derived_name
    if create_file:
        derived_top.write_text("; derived with reserved co-ion\n", encoding="utf-8")
    (derived_dir / runabfe.RESERVED_COION_REPORT_BASENAME).write_text(
        json.dumps({"derived_top": str(derived_top), "count": 1}), encoding="utf-8")
    return str(out), str(derived_top)


def test_neutral_run_is_unchanged(tmp_path):
    # 没有派生件 ⟹ 原样返回，中性配体路径逐字节不变。
    out = tmp_path / "run"
    out.mkdir()
    assert runabfe.runtime_gromacs_top(str(out), "/orig/complex.top") == "/orig/complex.top"


def test_charged_run_reads_the_derived_top(tmp_path):
    out, derived = _make_derivation(tmp_path)
    assert runabfe.runtime_gromacs_top(out, "/orig/complex.top") == derived


def test_missing_derived_file_fails_closed(tmp_path):
    # 报告在、派生件没了 ⟹ **不**静默退回原始件（那正是要修掉的失效模式）。
    out, _ = _make_derivation(tmp_path, create_file=False)
    with pytest.raises(FileNotFoundError, match="拒绝回退"):
        runabfe.runtime_gromacs_top(out, "/orig/complex.top")


def test_both_boresch_call_sites_go_through_it():
    """两个 `_boresch_mdtraj_topology` 调用点都必须先过 `runtime_gromacs_top`。

    这条钉的是"有没有漏掉兄弟调用点"——派生拓扑的消费者不止一个，
    漏一个就是"预平衡白跑几小时才炸"。
    """
    import pathlib
    src = pathlib.Path(runabfe.__file__).read_text(encoding="utf-8").splitlines()
    calls = [(i, ln) for i, ln in enumerate(src)
             if "_boresch_mdtraj_topology(" in ln and not ln.lstrip().startswith("def ")]
    assert len(calls) >= 2, f"只找到 {len(calls)} 个调用点，判据失效了"
    for i, ln in calls:
        window = "\n".join(src[i:i + 4])
        assert "runtime_gromacs_top(" in window, (
            f"runabfe.py:{i + 1} 直接把 config/args 的 .top 交给 Boresch 拓扑构建，"
            f"派生 co-ion 拓扑的运行会原子数对不上：{ln.strip()}"
        )
