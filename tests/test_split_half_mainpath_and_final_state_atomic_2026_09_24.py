"""2026-09-24 下午审查 #2 / #7。

#2：`_solve_stage_with_config_gates` 原来「给了 run_kwargs 就只用它」；主 stage 求解
    传的是逐键点名转发的那份（没有 `split_half_max_z`）⟹ 阈值进了指纹、执行时是 None。
#7：`pre_equil_final_state.npz` 直接 `np.savez`、读时不捕获 ⟹ 写到一半被杀之后
    每次 resume 都抛 BadZipFile，而不是回退到 pre_equil.chk。
"""
import functools
import types

import numpy as np
import pytest

pytest.importorskip("openmm")

from openmm import unit

import abfe_pipeline as ap


def test_gate_kwargs_merge_last_run_config_under_run_kwargs(monkeypatch):
    seen = {}

    def _fake_solve(**kw):
        seen.update(kw)
        return {}

    monkeypatch.setattr(ap, "solve_stage_integrated", _fake_solve)
    fake = types.SimpleNamespace(_last_run_config={"kwargs": {
        "split_half_max_z": 2.0, "final_min_ess_ratio": 0.1,
    }})
    ap.ABFEPipeline._solve_stage_with_config_gates(
        fake, window_outputs=[], kt=2.5, stage_name="stage1",
        run_kwargs={"final_min_ess_ratio": 0.2}, populate_diagnostics=False,
    )
    assert seen["split_half_max_z"] == 2.0, (
        "run_kwargs 里没有的键必须回落到 _last_run_config —— 否则主路径的漂移门永远是 None"
    )
    assert seen["final_min_ess_ratio"] == 0.2, "run_kwargs 里有的键仍然优先"


def _fake_pipeline(tmp_path, identity="ID"):
    chk_dir = tmp_path / "checkpoints"
    chk_dir.mkdir(exist_ok=True)
    fake = types.SimpleNamespace(
        checkpoint_dir=str(chk_dir), output_dir=str(tmp_path),
        PRE_EQUILIBRATION_FINAL_STATE_FILENAME=ap.ABFEPipeline.PRE_EQUILIBRATION_FINAL_STATE_FILENAME,
        pre_equilibration_identity_fingerprint=lambda n: identity, _log=lambda *_: None,
    )
    fake._set_pre_equilibrated_state = functools.partial(
        ap.ABFEPipeline._set_pre_equilibrated_state, fake)
    return fake


def test_final_state_written_atomically_and_read_back(tmp_path):
    fake = _fake_pipeline(tmp_path)
    pos = np.arange(9, dtype=float).reshape(3, 3) * 0.1
    box = np.eye(3) * 2.0
    state = types.SimpleNamespace(
        getPositions=lambda asNumpy: pos * unit.nanometer,
        getPeriodicBoxVectors=lambda asNumpy: box * unit.nanometer,
    )
    ap.ABFEPipeline._commit_pre_equilibrated_final_state(fake, None, state, "ID", rescale=False)

    files = sorted(p.name for p in (tmp_path / "checkpoints").iterdir())
    assert files == [ap.ABFEPipeline.PRE_EQUILIBRATION_FINAL_STATE_FILENAME], (
        f"只该留下最终文件（没有 .tmp、没有 np.savez 自补的 .npz.npz）：{files}"
    )
    fake.box_vectors = None
    assert ap.ABFEPipeline.adopt_pre_equilibrated_state(fake) is True
    got = np.asarray([v.value_in_unit(unit.nanometer) for v in fake.box_vectors])
    assert np.allclose(got, box)


def test_truncated_final_state_falls_back_instead_of_raising(tmp_path):
    fake = _fake_pipeline(tmp_path)
    final = tmp_path / "checkpoints" / ap.ABFEPipeline.PRE_EQUILIBRATION_FINAL_STATE_FILENAME
    final.write_bytes(b"PK\x03\x04 half a zip")
    # 没有 pre_equil.chk ⟹ 回落路径也找不到 ⟹ False，交给调用方决定；关键是不抛。
    assert ap.ABFEPipeline.adopt_pre_equilibrated_state(fake) is False
