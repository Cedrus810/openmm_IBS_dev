#!/usr/bin/env python3
"""Create separate figure candidates and descriptive analyses for editorial review.

Reads frozen CSVs and nine explicitly identified historical controller ledgers.
Does not edit the adopted manuscript, perform MD, or relabel scientific status.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    with path.open('w', newline='') as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def build(package, controller, output):
    parent = json.loads((package/'manifest.json').read_text())
    sources = []
    for name in ('abfe_table.csv','abfe_reps.csv'):
        path = package/name
        if sha(path)!=parent['output_hashes'][name]:
            raise ValueError(f'Changed frozen input: {path}')
        sources.append(dict(path=str(path.resolve()),sha256=sha(path)))
    table, reps = load(package/'abfe_table.csv'), load(package/'abfe_reps.csv')
    common = [r for r in table if r['ibs_status']==r['openfe_status']=='complete']
    if [r['system'] for r in common]!=parent['common_systems']:
        raise ValueError('Common system membership changed')
    output.mkdir(parents=True,exist_ok=False)
    raw_dir = output/'controller_sources'
    raw_dir.mkdir()
    ledgers = {}
    windows = []
    for system in ('brd4_ligand1','brd4_ligand2','cyclod_ligand1'):
        for rep in (1,2,3):
            path = controller/system/f'rep{rep}/checkpoints/stage2_rounds_ledger.json'
            raw = path.read_bytes()
            (raw_dir/f'{system}_rep{rep}.json').write_bytes(raw)
            sources.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()))
            data = json.loads(raw)
            ledgers[(system,rep)] = data
            for rd in data['rounds']:
                for window in range(5):
                    label, reasons = rd['labels'][str(window)]
                    ev = rd['evidence'][str(window)]
                    windows.append(dict(system=system,rep=rep,round=rd['round'],window=window,
                                        raw_label=label,reasons=';'.join(reasons),
                                        delta_f_kJ=ev['delta_f_kJ'],
                                        sigma_mbar_kJ=ev['sigma_mbar_all_frames_kJ'],
                                        deletion_kJ=ev['sigma_jk_kJ'],min_ess=min(ev['state_ess']),
                                        top1pct_mass=ev['top1pct_weight'],n_segments=ev['n_segments']))
    # Stop-complete runs may omit later rounds; carry their last observation
    # forward for the aggregate display, preserving the last observed round.
    display = []
    for key,data in ledgers.items():
        for rd in (1,2,3):
            prior = max(x['round'] for x in data['rounds'] if x['round']<=rd)
            for w in range(5):
                row = next(x for x in windows if (x['system'],x['rep'],x['round'],x['window'])==(*key,prior,w))
                display.append(dict(row,display_round=rd,observed_round=prior))
    passes = [sum(r['raw_label']=='CONVERGED' for r in display if r['display_round']==rd) for rd in (1,2,3)]
    if passes!=[27,38,40]:
        raise ValueError(f'Historical count mismatch: {passes}')
    additions = [sum(len(x['specs']) for d in ledgers.values() for x in d['rounds'] if x['round']==rd) for rd in (2,3)]
    if additions!=[18,7]:
        raise ValueError(f'Historical addition mismatch: {additions}')
    write(output/'controller_window_observations.csv',windows)
    write(output/'controller_heatmap_rows.csv',display)
    targets = ('CyclophilinD','HSP90','PWWP1')
    stratified = []
    for target in targets:
        rows = [r for r in common if r['target']==target]
        for method in ('ibs','openfe','paper'):
            for reference in ('exp_kcal','paper_mean_kcal'):
                if method=='paper' and reference=='paper_mean_kcal':
                    continue
                delta = [float(r[method+'_mean_kcal'])-float(r[reference]) for r in rows]
                stratified.append(dict(target=target,method=method,reference=reference,
                                       n_systems=len(rows),mse_kcal=mean(delta),
                                       mue_kcal=mean(map(abs,delta)),rmse_kcal=math.sqrt(mean(x*x for x in delta)),
                                       systems=';'.join(r['system'] for r in rows)))
    write(output/'target_stratified_metrics.csv',stratified)
    components = ('ibs_attach_kcal','ibs_stage1_kcal','ibs_stage2_kcal','restraint_kcal','solvent_kcal')
    component_rows = []
    closure = []
    for row in table:
        if row['ibs_status']!='complete':
            continue
        rs = [r for r in reps if r['system']==row['system'] and r['method']=='ibs'
              and r['protocol']=='runs_lcr_s1long15' and r['benchmark_eligible']=='1']
        if len(rs)!=3:
            raise ValueError('Expected three paired outcomes')
        vectors = [[float(r[k]) for k in components] for r in rs]
        binding = [float(r['dG_kcal']) for r in rs]
        for v,g in zip(vectors,binding):
            if abs(-sum(v[:4])+v[4]-g)>0.00031:
                raise ValueError('Paired cycle mismatch')
        sd = {k:stdev(float(r[k]) for r in rs) for k in components}
        component_rows.append(dict(system=row['system'],target=row['target'],ligand=row['ligand'],n=3,
                                   **{k+'_sd':v for k,v in sd.items()},
                                   complex_sd=stdev(float(r['complex_kcal']) for r in rs),
                                   binding_sd=stdev(binding),largest_marginal_sd=max(sd,key=sd.get),
                                   untrusted=sum(r['ibs_results_untrusted']=='1' for r in rs)))
        signed = [[-x for x in v[:4]]+[v[4]] for v in vectors]
        center = [mean(v[k] for v in signed) for k in range(5)]
        covariance = [[sum((v[i]-center[i])*(v[j]-center[j]) for v in signed)/2
                       for j in range(5)] for i in range(5)]
        reconstructed = sum(sum(v) for v in covariance)
        direct = stdev(binding)**2
        if abs(reconstructed-direct)>0.003:
            raise ValueError('Rounded covariance contrast mismatch')
        closure.append(dict(system=row['system'],variance_from_paired_components=reconstructed,
                            direct_binding_variance=direct,rounding_residual=reconstructed-direct,
                            diagonal_only_variance=sum(covariance[i][i] for i in range(5))))
    write(output/'component_sd_by_system.csv',component_rows)
    write(output/'paired_cycle_variance_check.csv',closure)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    artwork = output/'figures'
    artwork.mkdir()

    def save(fig,name):
        for ext in ('png','pdf','svg'):
            fig.savefig(artwork/f'{name}.{ext}',dpi=220)
        plt.close(fig)

    # Figure 1: a recorded placement example + a method schematic. The local
    # time illustration is explicitly schematic, not a measured trajectory.
    fig,axs = plt.subplots(1,3,figsize=(14,4.8),gridspec_kw={'width_ratios':[1,1,1.15]})
    data = ledgers[('brd4_ligand1',3)]
    specs = [x for x in data['rounds'][0]['specs'] if x['window']==4]
    center = next(x for x in specs if x['tilt_sign']==0)['sampling_f_k_kJ']
    kt = 0.008314462618*300
    palette = {'C':'#0072B2','L':'#D55E00','R':'#009E73'}
    for spec in specs:
        name = {0:'C',1:'L',-1:'R'}[spec['tilt_sign']]
        delta = (np.array(spec['sampling_f_k_kJ'])-np.array(center))/kt
        axs[0].plot(range(6),delta,'o-',color=palette[name],label=name)
        axs[1].plot([0,1],[delta[0],delta[0]],lw=3,color=palette[name],label=name)
    axs[0].set(xlabel='Local alchemical state',ylabel=r'Offset perturbation $(b_s-b_C)/kT$',
               title='A  Initial placement: distinct generators',ylim=(-1.4,1.4))
    axs[0].legend(frameon=False,ncol=3)
    axs[1].set(xlabel='Local production time (schematic)',ylabel='First-state offset perturbation (kT)',
               title='B  Offsets stay fixed within each segment',ylim=(-1.4,1.4))
    axs[1].set_xticks([0,1],['Start','End'])
    axs[1].text(0.03,-1.32,'Fresh streams from shared starting coordinates',fontsize=8)
    ax = axs[2]
    ax.set_axis_off(); ax.set_title('C  Add a generator; retain earlier identities')

    def box(x,y,w,h,label,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02',fc=color,ec='#555555',lw=0.8))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9)

    def arrow(a,b):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color='#555555',lw=1))

    box(0.02,0.68,0.4,0.2,'Recorded C / L / R\ngenerators + their frames','#EAF2F8')
    box(0.58,0.68,0.39,0.2,'Fixed physical\ntarget energies','#EEEEEE')
    box(0.25,0.37,0.5,0.16,'Generator-aware MBAR\n+ support / sensitivity checks','#F5F5F5')
    arrow((0.22,0.66),(0.38,0.55));arrow((0.76,0.66),(0.61,0.55))
    box(0.12,0.04,0.4,0.18,'Flag + budget available:\nnew centered generator','#FFF1D5')
    box(0.66,0.04,0.3,0.18,'Pass / budget limit:\nreport + stop','#E8F3ED')
    arrow((0.40,0.35),(0.34,0.24));arrow((0.64,0.35),(0.77,0.24))
    arrow((0.12,0.12),(0.02,0.67))
    fig.subplots_adjust(left=0.055,right=0.985,bottom=0.22,top=0.88,wspace=0.35)
    fig.text(0.055,0.06,'A: recorded BRD4 ligand 1 / rep3 / w4 offsets; B–C: method schematic.\n'
             'L/C/R denote different sampling potentials. New offsets do not alter old frames or physical targets.',fontsize=9)
    save(fig,'candidate_figure1_generators')

    fig = plt.figure(figsize=(12,7.8));grid=fig.add_gridspec(2,2,width_ratios=[1.15,1])
    ax=fig.add_subplot(grid[:,0]);matrix=[];labels=[]
    for key in ledgers:
        matrix.append([int(next(r for r in display if (r['system'],r['rep'],r['display_round'],r['window'])==(*key,rd,w))['raw_label']=='CONVERGED') for rd in (1,2,3) for w in range(5)])
        short={'brd4_ligand1':'BRD4 L1','brd4_ligand2':'BRD4 L2','cyclod_ligand1':'Cyclod L1'}
        labels.append(f'{short[key[0]]} / r{key[1]}')
    ax.imshow(matrix,cmap=ListedColormap(['#E6A04B','#5EAA8D']),vmin=0,vmax=1,aspect='auto')
    ax.set_yticks(range(9),labels);ax.set_xticks(range(15),[str(i%5) for i in range(15)])
    ax.set(xlabel='Window index: 0–4 in each round')
    ax.set_title('A  Whole cohort: diagnostic labels',pad=48)
    for x in (4.5,9.5):ax.axvline(x,color='white',lw=3)
    for x,rd,count in zip((2,7,12),(1,2,3),passes):ax.text(x,-0.8,f'Round {rd}\n{count}/45 pass',ha='center',fontsize=10)
    ax.legend(handles=[Patch(color='#5EAA8D',label='Diagnostic pass'),Patch(color='#E6A04B',label='Flag retained')],
              loc='upper center',bbox_to_anchor=(0.5,-0.1),frameon=False,fontsize=9,ncol=2)
    for j,rep in enumerate((3,1)):
        ax=fig.add_subplot(grid[j,1])
        rows=[r for r in windows if r['system']=='brd4_ligand1' and r['rep']==rep and r['window']==4]
        ax.plot([r['round'] for r in rows],[r['deletion_kJ'] for r in rows],'o-',color='#D55E00',label='Deletion sensitivity')
        ax.plot([r['round'] for r in rows],[r['sigma_mbar_kJ'] for r in rows],'s--',color='#0072B2',label='Nominal MBAR scale')
        ax.axhline(1.5,color='#777777',ls=':',label='Sensitivity gate')
        ax.set(xticks=[1,2,3],xlim=(0.8,3.2),ylim=(0,6.7),ylabel='Diagnostic magnitude (kJ/mol)',
               title=f'{chr(66+j)}  BRD4 L1 / r{rep} / w4: '+('improves' if rep==3 else 'remains flagged'))
        ax.text(0.95,6.05,f"Min ESS: {rows[0]['min_ess']:.1f} → {rows[-1]['min_ess']:.1f}",fontsize=9)
        ax.grid(alpha=0.15)
        if j==0:ax.legend(frameon=False,loc='upper right',fontsize=8)
        if j==1:ax.set_xlabel('Completed round')
    fig.subplots_adjust(left=0.13,right=0.98,bottom=0.20,top=0.86,wspace=0.5,hspace=0.40)
    fig.text(0.06,0.035,'Historical nine-run controller cohort, 2026-09-27. Pass counts: 27 → 38 → 40 of 45.\n'
             'Panel B: no new w4 segment in round 3; its prior result is retained.\n'
             'Diagnostic gates are not equilibrium or accuracy tests; neither magnitude is a calibrated interval.',fontsize=9)
    save(fig,'candidate_figure2_controller')

    fig,axs=plt.subplots(1,3,figsize=(15,5.5),gridspec_kw={'width_ratios':[1,1,1]})
    method_colors={'ibs':'#D55E00','openfe':'#0072B2','paper':'#777777'}
    names={'ibs':'AIBS','openfe':'OpenFE','paper':'GROMACS'}
    for j,(ref,methods) in enumerate([('paper_mean_kcal',('ibs','openfe')),('exp_kcal',('ibs','openfe','paper'))]):
        ax=axs[j]
        for k,target in enumerate(targets):
            rows=[r for r in common if r['target']==target]
            for mi,m in enumerate(methods):
                x=k+(mi-(len(methods)-1)/2)*0.23
                d=[float(r[m+'_mean_kcal'])-float(r[ref]) for r in rows]
                ax.scatter([x+v for v in np.linspace(-0.06,0.06,len(d))],d,s=28,color=method_colors[m],alpha=0.8,
                           label=names[m] if k==0 else None)
                ax.plot([x-0.08,x+0.08],[mean(d)]*2,color=method_colors[m],lw=3)
        ax.axhline(0,color='#555555',lw=1,ls='--')
        ax.set_xticks(range(3),['CypD\nn = 3','HSP90\nn = 15','PWWP1\nn = 5'])
        ax.set(title=f'{chr(65+j)}  Signed deviations by target',ylabel='Calculation − '+('GROMACS' if j==0 else 'experiment')+' (kcal/mol)')
        ax.grid(axis='y',alpha=0.15);ax.legend(frameon=False,fontsize=9)
    target_colors={'CyclophilinD':'#0072B2','HSP90':'#D55E00','PWWP1':'#009E73'}
    ax=axs[2]
    for target in targets:
        rows=[r for r in common if r['target']==target]
        ax.scatter([float(r['openfe_sd_kcal']) for r in rows],[float(r['ibs_sd_kcal']) for r in rows],
                   color=target_colors[target],s=40,label=target)
    ax.plot([0,6],[0,6],'--',color='#555555');ax.set(xlim=(0,6),ylim=(0,6),
         xlabel='OpenFE repeat SD (kcal/mol)',ylabel='AIBS repeat SD (kcal/mol)',title='C  Paired repeat dispersion')
    ax.set_aspect('equal',adjustable='box');ax.grid(alpha=0.15);ax.legend(frameon=False,fontsize=9,loc='lower right')
    fig.subplots_adjust(left=0.065,right=0.985,top=0.87,bottom=0.23,wspace=0.35)
    fig.text(0.065,0.07,'Identical 23-system complete-repeat intersection. A–B: one point per system mean; short lines mark target-group means.\n'
             'C: sample SD of three repeats per method/system. Target groups are small and unequal; no causal accuracy ranking is implied.',fontsize=9)
    save(fig,'candidate_figure3_target_deviations')

    ordered=sorted(component_rows,key=lambda r:r['binding_sd'],reverse=True)
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12,10),gridspec_kw={'width_ratios':[1.25,1]})
    vals=[[r[k+'_sd'] for k in components] for r in ordered]
    im=ax.imshow(vals,aspect='auto',cmap='YlOrRd',vmin=0,vmax=max(max(v) for v in vals))
    short={'CyclophilinD':'CypD','HSP90':'HSP90','PWWP1':'PWWP1'}
    ax.set_yticks(range(len(ordered)),[f"{short[r['target']]}-{r['ligand']}" for r in ordered],fontsize=8)
    ax.set_xticks(range(5),['Attach','Stage 1','Stage 2','Release','Solvent'],rotation=35,ha='right')
    for i,r in enumerate(ordered):ax.plot(components.index(r['largest_marginal_sd']),i,'s',ms=8,mfc='none',mec='#222222',mew=1)
    ax.set_title('A  Component SD across all 33 complete AIBS systems')
    cbar=fig.colorbar(im,ax=ax,fraction=0.04,pad=0.025);cbar.set_label('Between-repeat sample SD (kcal/mol)')
    for target in targets:
        rows=[r for r in ordered if r['target']==target]
        bx.scatter([r['ibs_stage2_kcal_sd'] for r in rows],[r['binding_sd'] for r in rows],
                   color=target_colors[target],s=42,label=target)
    for r in ordered:
        if r['system'] in ('HSP90_ligand-20','HSP90_ligand-28'):
            bx.annotate('HSP90-'+r['ligand'],(r['ibs_stage2_kcal_sd'],r['binding_sd']),
                        xytext=(-60,10 if r['ligand']=='20' else -20),textcoords='offset points',fontsize=9,
                        arrowprops={'arrowstyle':'-','color':'#555555'})
    lim=max(max(r['binding_sd'],r['ibs_stage2_kcal_sd']) for r in ordered)+0.5
    bx.plot([0,lim],[0,lim],'--',color='#555555');bx.set(xlim=(0,lim),ylim=(0,lim),
            xlabel='Complex Stage 2 repeat SD (kcal/mol)',ylabel='Binding repeat SD (kcal/mol)',
            title='B  Stage 2 and binding dispersion')
    bx.set_aspect('equal',adjustable='box');bx.legend(frameon=False,loc='lower right',fontsize=9);bx.grid(alpha=0.15)
    fig.subplots_adjust(left=0.10,right=0.97,bottom=0.16,top=0.92,wspace=0.45)
    count=Counter(r['largest_marginal_sd'] for r in ordered)
    fig.text(0.10,0.045,f"Stage 2 has the largest component SD in {count['ibs_stage2_kcal']}/33 systems; outlined cells mark each row's maximum.\n"
             'Each SD uses the same three paired outcomes. Solvent denotes the entire solvent leg.\n'
             'Component covariance is retained in cycle checks; marginal SDs do not define additive variance fractions.',fontsize=9)
    save(fig,'candidate_figure4_component_dispersion')
    metadata=dict(parent_snapshot=str(package.resolve()),parent_manifest_sha256=sha(package/'manifest.json'),
                  common_systems=parent['common_systems'],historical_pass_counts=passes,historical_additions=additions,
                  largest_component_sd_counts=dict(count),
                  complex_sd_exceeds_solvent=sum(r['complex_sd']>r['solvent_kcal_sd'] for r in component_rows),
                  controller_source_policy='Historical recorded labels; last observation carried forward only after early stop.',
                  scope='Descriptive existing-data figure candidates. No adopted manuscript edits, simulation, trajectory analysis or status changes.',
                  source_artifacts=sources,script_sha256=sha(Path(__file__)),matplotlib_version=matplotlib.__version__,
                  output_hashes={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({k:metadata[k] for k in ('historical_pass_counts','historical_additions','largest_component_sd_counts','complex_sd_exceeds_solvent')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--controller',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.package,a.controller,a.output)
