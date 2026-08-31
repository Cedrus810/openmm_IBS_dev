"""Small, fully identified production artifacts for offline contract tests."""
import json
from pathlib import Path
import numpy as np
import ibs_engine as ie


def segments(n, split=None):
    edges = [0, n] if split is None else [0, split, n]
    return [dict(start_frame=a, end_frame=b, n_frames=b-a,
                 session_id=f"test-session-{i}", reason="fresh" if i == 0 else "cross_process_resume")
            for i, (a, b) in enumerate(zip(edges, edges[1:]))]


def identify_window(output, checkpoints, index, lc, lv, *, kind="vdw", stage_key=None, f_k=None):
    """Attach the manifests/weights actually matching the existing arrays."""
    output, checkpoints = Path(output), Path(checkpoints)
    stem = output / f"dual_window_{index}_{kind}"
    energy_path, bias_path, base_path = [str(stem)+f"_{name}.npy" for name in ("energies", "bias", "base")]
    e = np.load(energy_path)
    weights = list(f_k if f_k is not None else np.zeros(len(lc)))
    conv = dict(window_data_protocol_version=ie.IBS_WINDOW_DATA_PROTOCOL_VERSION,
                window_data=ie._window_data_metadata(energy_path, bias_path, base_path),
                lambdas_coul=list(lc), lambdas_vdw=list(lv),
                wca_accounting_version=ie.WCA_ACCOUNTING_VERSION,
                ibs_bias_protocol_version=ie.IBS_BIAS_PROTOCOL_VERSION,
                ligand_com_restraint_protocol_version=ie.LIGAND_COM_RESTRAINT_PROTOCOL_VERSION,
                lj_tail_lrc_protocol_version=ie.TRADITIONAL_LJ_LRC_PROTOCOL_VERSION,
                vdw_nonbonded_protocol_version=ie.VDW_NONBONDED_PROTOCOL_VERSION,
                production_segment_protocol_version=ie.PRODUCTION_SEGMENT_PROTOCOL_VERSION,
                production_segments=segments(e.shape[1]), sampling_repair_policy="non_mutating_v1",
                lse_log_residual_tolerance=0.5, n_steps_per_window_effective=500000,
                stage_protocol_key=stage_key, early_stop_triggered=False)
    Path(str(stem)+"_convergence.json").write_text(json.dumps(conv))
    checkpoints.mkdir(parents=True, exist_ok=True)
    (checkpoints / f"ibs_state_{kind}_window_{index}.json").write_text(json.dumps(dict(f_k=weights, production_entry_f_k=weights)))
    manifest = ie._build_production_window_checkpoint_manifest(kind, index, len(lc), "test-system", lc, lv, None, None, weights, 300., "Reference")
    manifest["stage_protocol_key"] = stage_key
    _, manifest_path = ie._production_window_checkpoint_paths(str(checkpoints), kind, index)
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).write_text(json.dumps(manifest))
    return conv
