#!/usr/bin/env python3
"""Bind the 2026-10-11 handoff to manuscript statistics without new simulations.

Retains all eligible completed outcomes, including untrusted runs. Reads only
explicit final-result paths from the handoff; never enumerates the benchmark.
The output is a new immutable package, not a replacement for older snapshots.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def close(a, b, label, tol=0.00016):
    if not math.isfinite(float(a)) or abs(float(a) - float(b)) > tol:
        raise ValueError(f'{label}: {a} != {b}')


def metrics(rows, method, reference, cohort):
    x = [float(r[reference + '_kcal']) for r in rows]
    y = [float(r[method + '_mean_kcal']) for r in rows]
    d = [b-a for a, b in zip(x, y)]
    xm, ym = mean(x), mean(y)
    denominator = math.sqrt(sum((a-xm)**2 for a in x)*sum((b-ym)**2 for b in y))
    concordant = discordant = tied_x = tied_y = 0
    for i in range(len(x)):
        for j in range(i):
            dx, dy = x[i]-x[j], y[i]-y[j]
            if dx == dy == 0:
                continue
            if dx == 0:
                tied_x += 1
            elif dy == 0:
                tied_y += 1
            elif dx*dy > 0:
                concordant += 1
            else:
                discordant += 1
    tau_denom = math.sqrt((concordant+discordant+tied_x)*(concordant+discordant+tied_y))
    return dict(cohort=cohort, method=method, reference=reference,
                n_systems=len(rows), mse_kcal=mean(d), mue_kcal=mean(map(abs, d)),
                rmse_kcal=math.sqrt(mean(v*v for v in d)),
                rmse_offset_removed_kcal=math.sqrt(mean((v-mean(d))**2 for v in d)),
                pearson_r=sum((a-xm)*(b-ym) for a, b in zip(x, y))/denominator,
                kendall_tau=(concordant-discordant)/tau_denom,
                systems=';'.join(sorted(r['system'] for r in rows)))


def assemble(snapshot, benchmark, output):
    reps = read_csv(snapshot/'abfe_reps.csv')
    summary = read_csv(snapshot/'abfe_summary.csv')
    table = read_csv(snapshot/'abfe_table.csv')
    supplied_stats = read_csv(snapshot/'abfe_stats.csv')
    if len({r['system'] for r in table}) != len(table):
        raise ValueError('Duplicate wide-table system')
    keys = [(r['system'], r['method'], r['protocol'], r['rep']) for r in reps]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate replicate identity')
    groups = defaultdict(list)
    for r in reps:
        groups[(r['system'], r['method'], r['protocol'])].append(r)
        close(float(r['solvent_kcal'])-float(r['complex_kcal']), r['dG_kcal'],
              f"{r['source']} signed cycle", tol=0.021 if r['method']=='paper_gmx' else 0.00016)
        if r['method']=='ibs':
            close(sum(float(r[k]) for k in ('ibs_attach_kcal','ibs_stage1_kcal',
                                           'ibs_stage2_kcal','restraint_kcal')),
                  r['complex_kcal'], f"{r['source']} complex decomposition", tol=0.00026)
    for r in summary:
        values = [float(x['dG_kcal']) for x in groups[(r['system'],r['method'],r['protocol'])]]
        if len(values) != int(r['n']):
            raise ValueError('Summary replicate count mismatch')
        close(mean(values), r['mean_kcal'], 'summary mean')
        if len(values)>1:
            close(stdev(values), r['sd_kcal'], 'summary SD')
            close(stdev(values)/math.sqrt(len(values)), r['sem_kcal'], 'summary SEM')
    neutral = [r for r in table if r['charge_e']=='0']
    for r in table:
        for m in ('ibs','openfe'):
            if r['charge_e']!='0' and r[m+'_mean_kcal']:
                raise ValueError('Charged result leaked into wide table')
            p = 'runs_lcr_s1long15' if m=='ibs' else 'production'
            values = [float(x['dG_kcal']) for x in groups[(r['system'],m,p)]
                      if x['benchmark_eligible']=='1']
            if len(values)!=int(r[m+'_n']):
                raise ValueError('Wide-table replicate count mismatch')
            if values:
                close(mean(values),r[m+'_mean_kcal'],'wide-table mean')
            if r[m+'_status']=='complete' and len(values)!=3:
                raise ValueError('Complete row without three repeats')
    complete = {m:[r for r in neutral if r[m+'_status']=='complete'] for m in ('ibs','openfe')}
    common = [r for r in neutral if r['ibs_status']==r['openfe_status']=='complete']
    derived = []
    for m in ('ibs','openfe'):
        for ref in ('exp','paper_mean'):
            derived.append(metrics(complete[m],m,ref,'method_complete'))
            derived.append(metrics(common,m,ref,'common_complete'))
    derived.append(metrics(common,'paper','exp','common_complete'))
    # Check every supplied min_reps=3 metric, including its exact membership.
    pair_rows = [dict(r, openfe_reference_kcal=r['openfe_mean_kcal']) for r in common]
    pair_stat = metrics(pair_rows,'ibs','openfe_reference','common_complete')
    derived.append(pair_stat)
    for s in supplied_stats:
        if s['min_reps']!='3':
            continue
        m = s['comparison'].split(':')[0]
        ref = {'experiment':'exp','paper_gmx':'paper_mean','openfe:production':'openfe_reference'}[s['reference']]
        found = next(d for d in derived if d['method']==m and d['reference']==ref
                     and d['cohort']==('common_complete' if ref=='openfe_reference' else 'method_complete'))
        if found['n_systems']!=int(s['n_systems']) or found['systems']!=s['systems']:
            raise ValueError('Supplied statistics membership mismatch')
        for k in ('mse_kcal','mue_kcal','rmse_kcal','rmse_offset_removed_kcal','pearson_r','kendall_tau'):
            close(found[k],s[k],k)
    statuses = []
    source_hashes = []
    paper_rows = read_csv(benchmark/'free_energy_data/abfe.csv')
    paper_values = {}
    experiment_values = defaultdict(list)
    for r in paper_rows:
        target = 'CyclophilinD' if r['System']=='CycloD' else r['System']
        system = f"{target}_ligand-{r['ligand_ID']}"
        paper_values[(system,r['run'])] = float(r['calc_dG'])
        experiment_values[system].append(float(r['exp_dG']))
    for r in reps:
        if r['method']=='paper_gmx':
            close(paper_values[(r['system'],r['rep'])],r['dG_kcal'],'paper source/CSV value',tol=0.000051)
    for r in table:
        close(mean(experiment_values[r['system']]),r['exp_kcal'],'experiment source/CSV mean',tol=0.000051)
    openfe_prefixes = []
    for r in reps:
        if r['method']!='openfe':
            continue
        path = (benchmark/r['source']).resolve()
        if not path.is_relative_to(benchmark.resolve()):
            raise ValueError('Source outside benchmark')
        # Only the leading scalar estimate is needed; avoid reading the large
        # per-unit energy/serialization records on HDD-backed NFS.
        with path.open('rb') as handle:
            prefix = handle.read(1024)
        text = prefix.decode('utf-8')
        if not text.startswith('{"estimate":'):
            raise ValueError(f'{path}: unsupported leading estimate layout')
        estimate, _ = json.JSONDecoder().raw_decode(text[text.index(':')+1:].lstrip())
        if estimate['unit']!='kilocalorie_per_mole':
            raise ValueError('Unsupported OpenFE estimate units')
        close(estimate['magnitude'],r['dG_kcal'],'OpenFE source/CSV estimate',tol=0.000051)
        openfe_prefixes.append(dict(system=r['system'],rep=r['rep'],source=r['source'],
                                   estimate_kcal=estimate['magnitude'],prefix_bytes=len(prefix),
                                   prefix_sha256=hashlib.sha256(prefix).hexdigest(),
                                   source_bytes=path.stat().st_size))
    for r in reps:
        if r['method']!='ibs' or r['protocol']!='runs_lcr_s1long15':
            continue
        path = (benchmark/r['source']).resolve()
        if not path.is_relative_to(benchmark.resolve()):
            raise ValueError('Source outside benchmark')
        raw = path.read_bytes()
        d = json.loads(raw)
        close(d['delta_G_bind_kcal_mol'],r['dG_kcal'],'raw/CSV binding value',tol=0.000051)
        if int(bool(d['results_untrusted']))!=int(r['ibs_results_untrusted']):
            raise ValueError('Raw/CSV trust mismatch')
        if r['benchmark_eligible']=='1' and (d.get('must_not_report_delta_g_bind') or
                                           d['provenance']['charge_treatment']!='neutral'):
            raise ValueError('Unreportable/charged result entered neutral comparison')
        config = d['provenance']['config']
        if r['benchmark_eligible']=='1':
            for k, value in [('temperature',300),('n_steps_per_window',250000),
                             ('stage1_n_steps_per_window',750000),
                             ('stage1_remd_equilibration_steps',250000),
                             ('stage2_uniform_lambda_states',26),
                             ('stage2_window_ranges',[[0,6],[5,11],[10,16],[15,21],[20,26]])]:
                if config.get(k)!=value:
                    raise ValueError(f'{path}: protocol mismatch {k}')
        flags = d.get('stage_quality_failures') or []
        stages = sorted({f['stage'] for f in flags})
        statuses.append(dict(system=r['system'],rep=r['rep'],benchmark_eligible=r['benchmark_eligible'],
                             results_untrusted=d['results_untrusted'],
                             publishable_as_accepted_result=d['publishable_as_accepted_result'],
                             precision_status=d['precision_status'],
                             must_not_report_delta_g_bind=d.get('must_not_report_delta_g_bind',False),
                             quality_stages=';'.join(stages),
                             quality_flags_json=json.dumps(flags,ensure_ascii=False),source=r['source']))
        source_hashes.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()))
    eligible_statuses = [s for s in statuses if s['benchmark_eligible']=='1']
    coverage = []
    for target in sorted({r['target'] for r in table}):
        rs = [r for r in table if r['target']==target]
        ns = [r for r in rs if r['charge_e']=='0']
        coverage.append(dict(target=target,dataset_systems=len(rs),neutral_systems=len(ns),
                             charged_excluded=len(rs)-len(ns),
                             ibs_complete=sum(r['ibs_status']=='complete' for r in ns),
                             openfe_complete=sum(r['openfe_status']=='complete' for r in ns),
                             common_complete=sum(r['ibs_status']==r['openfe_status']=='complete' for r in ns)))
    splits = [r for r in reps if r['method']=='ibs' and r['protocol']=='runs_lcr_s1long15'
              and r['system'] in ('HSP90_ligand-20','HSP90_ligand-28')]
    split_stats = []
    for system in sorted({r['system'] for r in splits}):
        rs = [r for r in splits if r['system']==system]
        for field in ('dG_kcal','complex_kcal','solvent_kcal','ibs_attach_kcal',
                      'ibs_stage1_kcal','ibs_stage2_kcal','restraint_kcal'):
            values = [float(r[field]) for r in rs]
            split_stats.append(dict(system=system,component=field,n=len(values),
                                    mean_kcal=mean(values),sd_kcal=stdev(values),
                                    range_kcal=max(values)-min(values)))
    output.mkdir(parents=True,exist_ok=False)
    for name in ('README.md','abfe_reps.csv','abfe_summary.csv','abfe_table.csv','abfe_stats.csv'):
        shutil.copyfile(snapshot/name,output/('handoff_README.md' if name=='README.md' else name))
    write_csv(output/'derived_metrics.csv',derived)
    write_csv(output/'coverage.csv',coverage)
    write_csv(output/'raw_result_status.csv',statuses)
    write_csv(output/'openfe_estimate_prefixes.csv',openfe_prefixes)
    write_csv(output/'stage_split_reps.csv',splits)
    write_csv(output/'stage_split_stats.csv',split_stats)
    # Preserve the exact source descriptions used, without scanning the other checkout.
    doc_paths = ['openfe_ABFE/README.md','openmmIBS/README.md','openmmIBS/s2long_pilot/README.md',
                 'free_energy_data/README.md',
                 'experiments/2026-10-11_ibs_stage_split/stage_split.py',
                 'experiments/2026-10-11_ibs_stage_split/results_HSP90-20_28.txt',
                 'analysis/collect_abfe_results.py','free_energy_data/abfe.csv']
    source_dir = output/'source_records'
    source_dir.mkdir()
    for name in doc_paths:
        path = benchmark/name
        raw = path.read_bytes()
        (source_dir/name.replace('/','__')).write_bytes(raw)
        source_hashes.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()))
    raw_neutral = [r for r in reps if r['method']=='ibs' and r['protocol']=='runs_lcr_s1long15'
                   and r['benchmark_eligible']=='1']
    metadata = dict(assembled_at_utc=datetime.now(timezone.utc).isoformat(),
                    handoff_snapshot_jst='2026-10-11 06:39',input_snapshot=str(snapshot.resolve()),
                    input_hashes={p.name:sha(p) for p in snapshot.iterdir() if p.is_file()},
                    benchmark_source_root=str(benchmark.resolve()),
                    counts=dict(dataset=len(table),neutral=len(neutral),charged_excluded=len(table)-len(neutral),
                                ibs_complete=len(complete['ibs']),openfe_complete=len(complete['openfe']),
                                common_complete=len(common),neutral_ibs_runs=len(eligible_statuses),
                                neutral_ibs_untrusted=sum(s['results_untrusted'] for s in eligible_statuses),
                                neutral_ibs_accepted=sum(s['publishable_as_accepted_result'] for s in eligible_statuses),
                                neutral_ibs_dirty_source=sum('dirty' in r['code_version'] for r in raw_neutral)),
                    common_systems=[r['system'] for r in common],
                    method_complete_systems={m:[r['system'] for r in rs] for m,rs in complete.items()},
                    repeat_sd_medians={m:median(float(r[m+'_sd_kcal']) for r in rs)
                                       for m,rs in complete.items()},
                    scope='Descriptive existing-result assembly; no new MD, trajectory analysis, status changes, or acceptance.',
                    source_artifacts=source_hashes,script_sha256=sha(Path(__file__)))
    metadata['output_hashes']={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}
    (output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (output/'README.md').write_text('''# Expanded fragment-opt manuscript evidence, 2026-10-11

Frozen source cutoff: **2026-10-11 06:39 JST**. `handoff_README.md` is the
unaltered source description. The four `abfe_*.csv` files are exact copies of
`analysis/handoff_2026-10-11/`, not a recollection of still-running jobs.
The original 2026-10-05 snapshot and covariance package remain unchanged.

The dataset has 59 systems: 34 neutral and 25 charged exclusions. AIBS has
33 complete triples and one partial system; OpenFE has 24 complete triples,
six partial systems, and four neutral systems without results. The common
complete-repeat set has 23 systems (three CyclophilinD, 15 HSP90, five PWWP1).
All 100 eligible neutral AIBS outcomes are retained: 83 are untrusted, all
100 have UNMEASURED single-run precision, and zero are labeled accepted.
The source handoff's 84/101 count includes the excluded charged diagnostic run.

## Contents and interpretation

- `derived_metrics.csv`: unshifted equally weighted system-mean metrics on each
  method's own complete set and on the identical 23-system set, with membership.
  MSE means mean signed error. Offset-removed RMSE is descriptive residual
  scatter, not the reported unshifted RMSE or a corrected affinity prediction.
- `coverage.csv`: dataset/neutral/charged and complete/common counts by target.
- `raw_result_status.csv`: original acceptance, trust, precision, and detailed
  quality flags from all 101 s1long15 final JSONs, including the charged exclusion.
  Endpoint-uncertainty warning objects explicitly do not affect trust.
- `openfe_estimate_prefixes.csv`: checks all 83 OpenFE leading scalar estimates
  against the handoff; hashes cover only the first 1024 bytes, not whole large
  protocol results. Full OpenFE execution archives remain to be deposited.
- `stage_split_reps.csv`, `stage_split_stats.csv`: six paired HSP90-20/28 outputs.
  Stage 1 = decoupling total minus attachment minus Stage 2. Stage 2 is the sum
  of recorded window differences. The older printed S1resid includes attachment;
  it is not the Stage 1 column used here. No trajectories or MBAR were rerun.
- `source_records/`: exact benchmark protocol READMEs, original split script and
  printed results, collector, and published per-repeat/experimental source CSV.
- `manifest.json`: input/output hashes, exact system sets, full hashes of IBS
  final-result sources and source descriptions, counts, and exporter identity.
- `figures/`: Figures 3–4 as PNG, PDF, and SVG, with their own input/output/script
  hash manifest. Figure bars show repeat SD, not calibrated intervals.

The assembler checks duplicate identities, replicate counts, means/SD/SEM,
signed cycles, complex-leg decomposition, every supplied min_reps=3 metric and
membership, published/experimental source values, raw OpenFE binding scalars,
and raw IBS values, status, and the recorded neutral sampling schedule.
These are accounting/source checks, not physical-target or sampling acceptance.
Leg Hamiltonian/cycle correspondence, dirty source diffs (72/100 neutral IBS
launches), independent-basin coverage, and complete cost remain unresolved.
No charged binding value, historical protocol, partial repeat, or longer-Stage-2
pilot enters the main complete-repeat comparisons.

## Reproduce into a new directory

```sh
python3 tools/audit/assemble_fragment_opt_expanded.py \\
  --snapshot analysis/handoff_2026-10-11 \\
  --benchmark /home/ruigengji/fragment-opt-abfe-benchmark \\
  --output /tmp/aibs_expanded_reproduction
MPLCONFIGDIR=/tmp/aibs_matplotlib /home/ruigengji/miniforge3/envs/openfe_env/bin/python -I \\
  tools/audit/plot_fragment_opt_expanded.py --package /tmp/aibs_expanded_reproduction
```

Existing inputs only; no new simulations, job submissions, database writes,
trajectory analysis, or changes to acceptance/trust fields. The original run
files must remain available and agree with the frozen handoff for reassembly.
''')
    print(json.dumps(metadata['counts']))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--benchmark',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    assemble(args.snapshot,args.benchmark,args.output)
