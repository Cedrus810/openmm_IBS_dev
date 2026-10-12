#!/usr/bin/env python3
"""Export paired complete-run component covariance from frozen existing artifacts.

No trajectories are sampled/reanalysed and no run status is revised. Covariance
is descriptive (three completed adaptive-policy outcomes per system), not a
calibrated covariance for an equilibrium target or a within-run standard error.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

COMPONENTS = (
    ['complex_attachment', 'complex_stage1']
    + [f'complex_stage2_w{w}' for w in range(5)]
    + ['complex_release', 'solvent_stage1']
    + [f'solvent_stage2_w{w}' for w in range(5)]
    + ['external_apbs', 'external_reservoir']
)
SIGNS = [-1] * 8 + [1] * 8


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a: float, b: float, context: str, tol: float = 1e-6) -> float:
    delta = float(a) - float(b)
    if not math.isfinite(delta) or abs(delta) > tol:
        raise ValueError(f'{context}: assembly residual {delta} kJ/mol')
    return delta


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def assemble(snapshot: Path, output: Path) -> None:
    manifest = json.loads((snapshot / 'manifest.json').read_text())
    frozen_hashes = {record['path']: record['sha256']
                     for record in manifest['raw_aibs_sources']}
    systems = manifest['common_systems_min_three_repeats']
    sources: dict[str, str] = {}

    def read(path: Path) -> dict:
        digest = sha256(path)
        expected = frozen_hashes.get(str(path))
        if expected is not None and digest != expected:
            raise ValueError(f'Frozen input changed: {path}')
        sources[str(path)] = digest
        return json.loads(path.read_text())

    all_rows, covariance_rows, summaries = [], [], []
    for system in systems:
        finals = sorted(Path(record['path']) for record in manifest['raw_aibs_sources']
                        if record['path'].endswith('/final_binding_results.json')
                        and Path(record['path']).parent.parent.name == system)
        if len(finals) != 3:
            raise ValueError(f'{system}: expected frozen three-repeat set')
        vectors, binding, leg_values = [], [], []
        for final_path in finals:
            final = read(final_path)
            run = final_path.parent
            if final['provenance']['charge_treatment'] != 'neutral':
                raise ValueError(f'{run}: covariance export is scoped to the recorded neutral cycle')
            parts, stage1_methods = {}, []
            for leg, prefix in [('complex', ''), ('solvent', 'solvent_leg/')]:
                one = read(run / f'{prefix}checkpoints/stage1_decharging.json')
                two = read(run / f'{prefix}checkpoints/stage2_vanishing.json')
                chain = two['diagnostics']['covariance_chain_segments']
                if sorted(int(item['window_index']) for item in chain) != list(range(5)):
                    raise ValueError(f'{run}: incomplete/duplicate {leg} windows')
                parts[f'{leg}_stage1'] = float(one['total_delta_G'])
                for item in chain:
                    parts[f'{leg}_stage2_w{item["window_index"]}'] = float(item['delta_G_kJ_mol'])
                close(sum(parts[f'{leg}_stage2_w{w}'] for w in range(5)),
                      two['total_delta_G'], f'{run} {leg} stage2')
                stage1_methods.append(one['method'])
            if not final['boresch_attachment']['already_included_in_decoupling_delta_G']:
                raise ValueError(f'{run}: unsupported attachment convention')
            if not final['boresch_correction_already_included_in_complex_delta_G']:
                raise ValueError(f'{run}: unsupported release convention')
            parts['complex_attachment'] = float(final['boresch_attachment']['delta_G_kJ_mol'])
            parts['complex_release'] = float(final['boresch_correction_kJ_mol'])
            parts['external_apbs'] = float(final['apbs_correction_kJ_mol'])
            reservoir = final['thermodynamic_cycle_terms']['charge_transfer_reservoir_correction_kJ_mol']
            parts['external_reservoir'] = float(reservoir or 0.0)
            vector = [parts[name] for name in COMPONENTS]
            if not all(math.isfinite(value) for value in vector):
                raise ValueError(f'{run}: nonfinite component')
            complex_total, solvent_total = sum(vector[:8]), sum(vector[8:14])
            c_res = close(complex_total, final['complex_delta_G_kJ_mol'], f'{run} complex cycle')
            s_res = close(solvent_total, final['solvent_delta_G_kJ_mol'], f'{run} solvent cycle')
            reconstructed = sum(a * x for a, x in zip(SIGNS, vector))
            residual = close(reconstructed, final['delta_G_bind_kJ_mol'], f'{run} binding cycle')
            vectors.append(vector)
            binding.append(float(final['delta_G_bind_kJ_mol']))
            leg_values.append([complex_total, solvent_total])
            all_rows.append(dict(system=system, repeat=run.name,
                                 units='kJ/mol', **parts,
                                 complex_total=complex_total, solvent_total=solvent_total,
                                 binding_total=reconstructed, complex_residual=c_res,
                                 solvent_residual=s_res, binding_residual=residual,
                                 complex_stage1_method=stage1_methods[0],
                                 solvent_stage1_method=stage1_methods[1],
                                 results_untrusted=final['results_untrusted'],
                                 publishable_as_accepted_result=final['publishable_as_accepted_result']))
        n = len(vectors)
        center = [mean(row[i] for row in vectors) for i in range(len(COMPONENTS))]
        covariance = [[sum((row[i] - center[i]) * (row[j] - center[j]) for row in vectors) / (n - 1)
                       for j in range(len(COMPONENTS))] for i in range(len(COMPONENTS))]
        for i, first in enumerate(COMPONENTS):
            for j, second in enumerate(COMPONENTS):
                covariance_rows.append(dict(system=system, n_completed=n,
                                            component_i=first, component_j=second,
                                            covariance_kJ2_mol2=covariance[i][j],
                                            rank_upper_bound=n - 1))
        full_var = sum(SIGNS[i] * SIGNS[j] * covariance[i][j]
                       for i in range(len(COMPONENTS)) for j in range(len(COMPONENTS)))
        direct_var = stdev(binding) ** 2
        close(full_var, direct_var, f'{system} covariance contrast', tol=1e-8)
        c_mean = mean(row[0] for row in leg_values)
        s_mean = mean(row[1] for row in leg_values)
        leg_cov = sum((row[0]-c_mean)*(row[1]-s_mean) for row in leg_values)/(n-1)
        summaries.append(dict(system=system, n_completed=n, rank_upper_bound=n-1,
                              covariance_contrast_variance_kJ2_mol2=full_var,
                              component_diagonal_only_variance_kJ2_mol2=sum(covariance[i][i] for i in range(len(COMPONENTS))),
                              complex_solvent_covariance_kJ2_mol2=leg_cov,
                              binding_sd_kcal_mol=math.sqrt(max(full_var, 0.0))/4.184,
                              direct_binding_sd_kcal_mol=stdev(binding)/4.184,
                              repeat_mean_sem_kcal_mol=stdev(binding)/4.184/math.sqrt(n),
                              interpretation='descriptive completed adaptive-policy repeat covariance; not interval calibration'))
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'component_repeats.csv', all_rows)
    write_csv(output / 'component_covariance.csv', covariance_rows)
    write_csv(output / 'contrast_summary.csv', summaries)
    metadata = dict(parent_snapshot=str(snapshot.resolve()),
                    parent_manifest_sha256=sha256(snapshot/'manifest.json'),
                    systems=systems, component_order=COMPONENTS, binding_contrast=SIGNS,
                    covariance_units='(kJ/mol)^2', n_completed_per_system=3,
                    status_policy='all completed outcomes retained; original status unchanged',
                    scope='descriptive paired repeat covariance conditional on completed recorded preparation protocol; no bias correction or interval calibration',
                    newly_bound_inputs='Stage 1 checkpoints hashed at covariance extraction; final and Stage 2 inputs checked against parent snapshot hashes',
                    source_artifacts=[dict(path=path, sha256=digest) for path, digest in sorted(sources.items())],
                    script_sha256=sha256(Path(__file__)),
                    output_sha256={name:sha256(output/name) for name in ('component_repeats.csv','component_covariance.csv','contrast_summary.csv')})
    (output/'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
    (output/'README.md').write_text('''# Paired complete-run component covariance

Derived from the frozen three-repeat HSP90 set 15/17/18 without new simulations or trajectory reanalysis. Each run retains all five windows per leg, Stage 1, attachment, signed release and external corrections. Source final/Stage 2 hashes match the parent snapshot; additional Stage 1 checkpoints are hashed in this package.

`component_repeats.csv` records 9 paired vectors, archived Stage 1 estimator names, original status and cycle assembly residuals. `component_covariance.csv` is the per-system sample covariance with denominator n−1. `contrast_summary.csv` verifies aᵀCa against direct binding-total sample variance and exposes the difference from diagonal-only component aggregation. No component covariance is assumed zero by default.

All runs remain untrusted. With n=3 the covariance rank is at most 2; these estimates are unstable, conditional on completed outcomes and the recorded initialization protocol, and cannot calibrate intervals or estimate physical-target bias. The repeat mean SEM is descriptive under independent same-policy replication assumptions. No independent-confirmation or block-bootstrap protocol has been executed.

Reproduce into a new directory:

```sh
python tools/audit/assemble_fragment_opt_covariance.py --snapshot docs/manuscript_data/fragment_opt_20261005T113824 --output /tmp/aibs_component_covariance_check
```
''')
    print(json.dumps(dict(output=str(output), runs=len(all_rows), systems=len(summaries),
                          covariance_rows=len(covariance_rows), cycle_checks='passed')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assemble(args.snapshot, args.output)
