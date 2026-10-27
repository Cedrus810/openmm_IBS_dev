"""Refit the outer-λ local residual from finished stage-2 runs of one ligand and write a deployable
manifest (payload + weights + manifest.json in one directory).

    python tools/refit_outer_residual_from_stage2.py --runs RUN [RUN ...] --output DIR \
        [--stride N] [--temperature 300] [--ligand-name MOL]

RUN = a finished complex-leg output directory (the one with vanishing*/, checkpoints/, topology.cif,
ligand_indices.json, system_native.xml) whose stage-2 sidecars kept nearby water. Several runs of the
same ligand (e.g. reps, 1Na and 2Na boxes) can be pooled. Then rerun with
    --outer-lambda-local-residual-ibs --outer-lambda-resource-manifest DIR/manifest.json
Workflow: docs/RETRAIN_LOCAL_RESIDUAL.md; method: local_residual/stage2_refit.py.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_residual.refit import write_refit_payload  # noqa: E402
from local_residual.stage2_refit import stage2_closed_form_refit  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stride", type=int, default=1, help="use every N-th sidecar frame (default 1 = all)")
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--ligand-name", default="MOL")
    a = p.parse_args(argv)

    log = lambda m: print(m, flush=True)  # noqa: E731
    result = stage2_closed_form_refit(a.runs, temperature_kelvin=a.temperature, frame_stride=a.stride, log=log)
    os.makedirs(a.output, exist_ok=True)
    write_refit_payload(result, output_dir=a.output, ligand_topology_indices=result["ligand_ids"],
                        protocol_sha256=result["protocol_sha256"], log=log)
    ref = result["reference_run"]
    from abfe_scripts.write_local_residual_resource_manifest import main as write_manifest

    write_manifest([
        "--payload", os.path.join(a.output, "r1_model_payload_v1.json"),
        "--weights", os.path.join(a.output, "r1_model_weights_f64.bin"),
        "--topology", os.path.join(ref, "topology.cif"),
        "--ligand-indices", os.path.join(ref, "ligand_indices.json"),
        "--system", os.path.join(ref, "system_native.xml"),
        "--ligand-name", a.ligand_name,
        "--experiment-id", "STAGE2-REFIT-2026-10-17",
        "--output", os.path.join(a.output, "manifest.json"),
    ])
    with open(os.path.join(a.output, "stage2_refit_report.json"), "w") as fh:
        json.dump({"runs": a.runs, "stride": a.stride, "fit_report": result["fit_report"],
                   "realized_capacity_usage": result["realized_capacity_usage"],
                   "capacities": result["capacities"], "target_state_ess": result["target_state_ess"],
                   "tanh_saturated_fraction": result["tanh_saturated_fraction"]}, fh, indent=1)
    print(f"manifest: {os.path.join(a.output, 'manifest.json')}")


if __name__ == "__main__":
    main()
