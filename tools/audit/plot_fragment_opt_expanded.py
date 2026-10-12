#!/usr/bin/env python3
"""Render manuscript Figures 3–4 from the checked 2026-10-11 CSV package.

Requires matplotlib; does not read trajectories, original run directories, or
change scientific status. Error bars are between-repeat sample SDs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p):
    with p.open(newline='') as handle:
        return list(csv.DictReader(handle))


def plot(package, output=None):
    manifest = json.loads((package/'manifest.json').read_text())
    inputs = ['abfe_table.csv','stage_split_reps.csv','stage_split_stats.csv']
    for name in inputs:
        if sha(package/name)!=manifest['output_hashes'][name]:
            raise ValueError(f'Frozen plot input changed: {name}')
    table = load(package/'abfe_table.csv')
    common = [r for r in table if r['ibs_status']==r['openfe_status']=='complete']
    if [r['system'] for r in common]!=manifest['common_systems']:
        raise ValueError('Common-set identity mismatch')
    out = output or package/'figures'
    out.mkdir(exist_ok=False)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    colors = {'CyclophilinD':'#0072B2','HSP90':'#D55E00','PWWP1':'#009E73'}
    fig = plt.figure(figsize=(13,10))
    grid = fig.add_gridspec(2,2,height_ratios=[1.15,1])
    for j, (method,label) in enumerate([('openfe','OpenFE'),('ibs','AIBS')]):
        ax = fig.add_subplot(grid[0,j])
        for target,color in colors.items():
            rows = [r for r in common if r['target']==target]
            ax.errorbar([float(r['paper_mean_kcal']) for r in rows],
                        [float(r[method+'_mean_kcal']) for r in rows],
                        xerr=[float(r['paper_sd_kcal']) for r in rows],
                        yerr=[float(r[method+'_sd_kcal']) for r in rows],
                        fmt='o',ms=5,color=color,ecolor=color,alpha=0.8,
                        elinewidth=0.8,capsize=2,label=target)
        ax.plot([-23,0],[-23,0],color='#555555',lw=1,ls='--')
        ax.set(xlim=(-23,0),ylim=(-23,0),xlabel='Published GROMACS mean (kcal/mol)',
               ylabel=f'{label} mean (kcal/mol)',title=f'{chr(65+j)}  {label} vs GROMACS; same 23 systems')
        ax.set_aspect('equal',adjustable='box')
        ax.grid(alpha=0.15)
        if j==0:
            ax.legend(frameon=False,loc='upper left',fontsize=9)
    ax = fig.add_subplot(grid[1,:])
    for method,label,color,marker,shift in [('paper','GROMACS','#777777','D',-0.18),
                                           ('openfe','OpenFE','#0072B2','s',0),
                                           ('ibs','AIBS','#D55E00','o',0.18)]:
        ax.scatter([i+shift for i in range(len(common))],
                   [float(r[method+'_sd_kcal']) for r in common],
                   label=label,color=color,marker=marker,s=35)
    short = {'CyclophilinD':'CypD','HSP90':'HSP90','PWWP1':'PWWP1'}
    labels = [f"{short[r['target']]}-{r['ligand']}" for r in common]
    ax.set_xticks(range(len(common)),labels,rotation=60,ha='right',fontsize=9)
    ax.set(ylabel='Between-repeat sample SD (kcal/mol)',ylim=(0,6.2),
           title='C  Repeat dispersion on the identical 23-system set')
    ax.grid(axis='y',alpha=0.2)
    ax.legend(frameon=False,ncol=3)
    fig.subplots_adjust(left=0.08,right=0.98,bottom=0.18,top=0.92,hspace=0.40,wspace=0.28)
    fig.suptitle('Fragment-opt benchmark: reference agreement and repeat stability',fontsize=15)
    fig.text(0.08,0.04,'Means and SDs: AIBS/OpenFE n = 3; GROMACS n = 5 (CypD-39: n = 20).\n'
             'SD bars describe dispersion, not calibrated intervals. AIBS: no output labeled accepted.\n'
             'Frozen handoff: 2026-10-11 06:39 JST; all eligible completed outcomes retained.',fontsize=10)
    for ext in ('png','pdf','svg'):
        fig.savefig(out/f'figure3_reference_and_repeat_dispersion.{ext}',dpi=300)
    plt.close(fig)
    splits = load(package/'stage_split_reps.csv')
    stats = load(package/'stage_split_stats.csv')
    fig, axes = plt.subplots(1,2,figsize=(10,5),sharey=True)
    repeat_colors = ['#0072B2','#D55E00','#009E73']
    for j, ligand in enumerate(('20','28')):
        ax = axes[j]
        rows = sorted([r for r in splits if r['system']==f'HSP90_ligand-{ligand}'],key=lambda r:int(r['rep']))
        for i,r in enumerate(rows):
            ax.scatter([0+(i-1)*0.12,1+(i-1)*0.12],
                       [float(r['ibs_stage1_kcal']),float(r['ibs_stage2_kcal'])],
                       color=repeat_colors[i],s=65,label=f"Repeat {r['rep']}")
        ax.set_xticks([0,1],['Stage 1\ndecharging','Stage 2\nvanishing'])
        ax.set(xlim=(-0.4,1.4),ylim=(0,25),title=f'{chr(65+j)}  HSP90 ligand {ligand}; n = 3')
        for k,field in enumerate(('ibs_stage1_kcal','ibs_stage2_kcal')):
            st = next(s for s in stats if s['system']==f'HSP90_ligand-{ligand}' and s['component']==field)
            ax.text(k,23.2,f"Range: {float(st['range_kcal']):.2f}",ha='center',fontsize=10)
        ax.grid(axis='y',alpha=0.2)
    axes[0].set_ylabel('Complex-leg contribution (kcal/mol)')
    axes[1].legend(frameon=False,loc='lower left',fontsize=9)
    fig.subplots_adjust(left=0.09,right=0.98,bottom=0.28,top=0.84,wspace=0.18)
    fig.suptitle('Largest observed component variation occurs in Stage 2',fontsize=14)
    fig.text(0.09,0.05,'Colors retain paired repeat identity; Stage 1 excludes restraint attachment.\n'
             'All six binding outputs are untrusted. Component SDs are not additive.\n'
             'Frozen handoff: 2026-10-11 06:39 JST.',fontsize=10)
    for ext in ('png','pdf','svg'):
        fig.savefig(out/f'figure4_stage_split.{ext}',dpi=300)
    plt.close(fig)
    metadata = dict(script_sha256=sha(Path(__file__)),parent_manifest_sha256=sha(package/'manifest.json'),
                    input_hashes={name:sha(package/name) for name in inputs},
                    output_hashes={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()},
                    error_bars='between-repeat sample SD, not calibrated intervals',
                    matplotlib_version=matplotlib.__version__)
    (out/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Rendered Figures 3–4 in {out}')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--output',type=Path,help='New artwork directory; defaults to <package>/figures')
    args = parser.parse_args()
    plot(args.package,args.output)
