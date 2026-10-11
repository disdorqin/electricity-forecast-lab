"""Auditable complete OR deadline-partial report; never promotes partial metrics to success."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,datetime,itertools
from pathlib import Path
import numpy as np
from ...contracts import assert_write_path
from ...artifacts import new_run,write_json,write_csv
from ...config import FAMILIES,SOURCE_SHA256
from .profile import ARMS,BOUNDS
from .metrics import pooled,monthly,promotion
from .pipeline import DIAG12,load_record

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def scalar_metrics(obj):
    return {k:v for k,v in (obj or {}).items() if isinstance(v,(float,int)) or v is None}
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--final',action='store_true');args=p.parse_args()
    run=assert_write_path(args.run);progress=read(run/'PROGRESS.json');contract=read(run/'FROZEN_RUN_CONTRACT.json')
    benchmark=read(run/'RUNTIME_BENCHMARK.json') if (run/'RUNTIME_BENCHMARK.json').exists() else None
    records={a:{} for a in ARMS}
    for arm,dates in progress['completed'].items():
        for date in dates:
            root=run/arm/date
            if arm=='FULL_V2' and not root.exists():root=Path(read(run/'REUSE_RECEIPT.json')['source'])
            records[arm][date]=load_record(root,date)
            rec=records[arm][date]
            target_da=rec['out'].get('target_da')
            if target_da is None:
                with np.load(assert_write_path(root/date/'EVALUATION_TRUTH.npz'),allow_pickle=False) as f:target_da=f['target_da']
            rt_error=rec['out']['rt_median'][0]-(target_da+rec['truth'][:,0])
            spread_error=rec['out']['spread_median'][0]-rec['truth'][:,0]
            assert np.allclose(rt_error,spread_error,rtol=1e-6,atol=1e-5),'RT/Spread full DA identity invalid'
            rec['actual_rt_mae']=float(abs(rt_error).mean())
    report=new_run(run/'reports');primary=progress['primary'];comparisons=[]
    metrics={};months={};daily=[]
    for arm,rs in records.items():
        days=[rs[d] for d in primary if d in rs];metrics[arm]=pooled(days);months[arm]=monthly(days)
        if metrics[arm]:metrics[arm]['rt_mae']=float(np.mean([r['actual_rt_mae'] for r in days]))
        for date,r in rs.items():daily.append(dict(arm=arm,target_day=date,**scalar_metrics(pooled([r]))))
        comparisons.append(dict(arm=arm,panel='PRIMARY',complete=bool(primary) and len(days)==len(primary),
            **scalar_metrics(metrics[arm])))
        diag=[rs[d] for d in DIAG12 if d in rs]
        comparisons.append(dict(arm=arm,panel='DIAG12',complete=len(diag)==12,**scalar_metrics(pooled(diag))))
    # Heterogeneous CSV schemas must retain explicit missing fields.
    def csv(path,rows):
        if not rows:return
        keys=sorted(set().union(*(r.keys() for r in rows)))
        write_csv(path,[{k:r.get(k) for k in keys} for r in rows])
    csv(report/'COMPONENT_COMPARISON.csv',comparisons);csv(report/'DAILY_METRICS.csv',daily)
    write_json(report/'MONTHLY_METRICS.json',months);write_json(report/'POOLED_METRICS.json',metrics)
    gate_rows,dep_rows,opt_rows,gradient_rows=[],[],[],[]
    gate_curves={};dep_curves={};underfit={};missing_initial=[]
    for arm,rs in records.items():
        for date,r in sorted(rs.items()):
            root=Path(r['root']);initial=read(root/'INITIAL_DIAGNOSTICS.json') if (root/'INITIAL_DIAGNOSTICS.json').exists() else {}
            final=read(root/'FINAL_DIAGNOSTICS.json') if (root/'FINAL_DIAGNOSTICS.json').exists() else {}
            if not final and arm=='FULL_V2':
                final={'spectral':read(root/date/'SPECTRAL_GATES.json'),
                       'dependence':{k:r['out'][k] for k in ('rho','phiL','phiH')}}
            if not initial and arm!='DIRECT96':missing_initial.append(dict(arm=arm,date=date))
            gates=final.get('spectral',{})
            for family,g in gates.items():
                curve=np.asarray(g['gate']).reshape(-1,96)[0];gate_curves[(arm,date,family)]=curve
                init=initial.get('spectral',{}).get(family)
                dist=float(np.sqrt(np.mean((curve-np.asarray(init['gate']).reshape(-1,96)[0])**2))) if init else None
                pairs={other:float(np.sqrt(np.mean((curve-np.asarray(h['gate']).reshape(-1,96)[0])**2))) for other,h in gates.items() if other!=family}
                gate_rows.append(dict(arm=arm,date=date,family=family,cutoff=float(np.asarray(g['cutoff']).mean()),
                    tau=float(g['tau']),delta=float(g['delta']),c0=float(g['c0']),rms_distance_initial=dist,
                    max_pairwise_rms=max(pairs.values()),mean_pairwise_rms=float(np.mean(list(pairs.values()))),
                    **{'distance_'+k:v for k,v in pairs.items()}))
            for name,v in final.get('dependence',{}).items():
                if name not in ('rho','phiL','phiH'):continue
                curve=np.asarray(v).reshape(-1,96)[0];dep_curves[(arm,date,name)]=curve
                init=initial.get('dependence',{}).get(name)
                dep_rows.append(dict(arm=arm,date=date,curve=name,min=float(curve.min()),max=float(curve.max()),
                    mean=float(curve.mean()),std=float(curve.std()),smoothness=float(np.mean(np.diff(curve)**2)),
                    rms_initial=float(np.sqrt(np.mean((curve-np.asarray(init).reshape(-1,96)[0])**2))) if init is not None else None,
                    rms_neutral=float(np.sqrt(np.mean(curve**2)))))
            training=r['training']
            for stage,floor,maximum in BOUNDS:
                selected=training['selected_epochs'][stage];history=[h for h in training['history'] if h['phase']=='selection' and h['stage']==stage]
                key='stage_a' if stage=='A' else 'crps'
                vals=[h['validation'][key] for h in history if key in h['validation']]
                improving=len(vals)>=4 and float(np.polyfit(np.arange(4),vals[-4:],1)[0])<-1e-5
                underfit.setdefault((arm,stage),[]).append(bool(selected==floor and improving))
                opt_rows.append(dict(arm=arm,date=date,stage=stage,selected=selected,floor=floor,maximum=maximum,
                    hit_min=selected==floor,hit_max=selected==maximum,selection_executed=len(history),
                    validation_tail_improving=improving,floor_and_improving=selected==floor and improving,
                    selected_score=vals[selected-1],last_score=vals[-1]))
            audit=training['gradient_audit']
            for group,norm in audit['norms']['final'].items():
                gradient_rows.append(dict(arm=arm,date=date,group=group,final_norm=norm,
                    intentionally_ablated=group in audit.get('intentionally_ablated_groups',[]),
                    final_low_cosine=audit['final_vs_component_cosine'].get(group+'/low'),
                    final_high_cosine=audit['final_vs_component_cosine'].get(group+'/high')))
    gate_summary={};dep_summary={};opt_summary={}
    for arm in ARMS:
        gate_summary[arm]={}
        for family in FAMILIES:
            rows=[r for r in gate_rows if r['arm']==arm and r['family']==family]
            if rows:gate_summary[arm][family]=dict(cutoff_mean=float(np.mean([r['cutoff'] for r in rows])),
                cutoff_across_day_std=float(np.std([r['cutoff'] for r in rows])),
                mean_distance_initial=float(np.mean([r['rms_distance_initial'] for r in rows if r['rms_distance_initial'] is not None]))
                    if any(r['rms_distance_initial'] is not None for r in rows) else None,
                mean_pairwise_distance=float(np.mean([r['mean_pairwise_rms'] for r in rows])))
        dep_summary[arm]={}
        for name in ('rho','phiL','phiH'):
            rows=[r for r in dep_rows if r['arm']==arm and r['curve']==name]
            if rows:
                curves=np.stack([dep_curves[(arm,r['date'],name)] for r in rows])
                dep_summary[arm][name]=dict(min=float(curves.min()),max=float(curves.max()),mean=float(curves.mean()),
                    std=float(curves.std()),daily_curve_variation=float(np.std(curves,axis=0).mean()),
                    smoothness=float(np.mean([r['smoothness'] for r in rows])),
                    mean_rms_initial=float(np.mean([r['rms_initial'] for r in rows if r['rms_initial'] is not None]))
                        if any(r['rms_initial'] is not None for r in rows) else None)
        opt_summary[arm]={stage:dict(min_hit_rate=float(np.mean([r['hit_min'] for r in opt_rows if r['arm']==arm and r['stage']==stage])),
            max_hit_rate=float(np.mean([r['hit_max'] for r in opt_rows if r['arm']==arm and r['stage']==stage])),
            floor_improving_rate=float(np.mean(underfit[(arm,stage)])),
            underfit_flag=float(np.mean(underfit[(arm,stage)]))>.5) for stage,_,_ in BOUNDS if (arm,stage) in underfit}
    gate_status='UNADJUDICATED_INCOMPLETE_PRIMARY'
    rows=[r for r in gate_rows if r['arm']=='FULL_V2'];known=[r for r in rows if r['rms_distance_initial'] is not None]
    if rows and len(records['FULL_V2'])>=len(primary)>0:
        inactive=bool(known) and max(r['max_pairwise_rms'] for r in rows)<.01 and max(r['rms_distance_initial'] for r in known)<.01
        inactive=inactive and all(v['cutoff_across_day_std']<.1 for v in gate_summary['FULL_V2'].values())
        gate_status='GATE_INACTIVE' if inactive else 'GATE_ACTIVE_OR_NONIDENTICAL'
    matched=sorted(set(records['FULL_V2'])&set(records['NO_DYNAMIC_DEP']))
    f=pooled([records['FULL_V2'][d] for d in matched]);n=pooled([records['NO_DYNAMIC_DEP'][d] for d in matched])
    dependence_status='UNADJUDICATED_NO_MATCHED_ABLATION'
    if matched:
        negligible=not promotion(f,n)
        ds=dep_summary['FULL_V2']
        near=all(v.get('mean_rms_initial') is not None and v['mean_rms_initial']<.01 for v in ds.values())
        dependence_status='DEPENDENCE_INACTIVE' if near and negligible and len(matched)>=12 else 'DEPENDENCE_EFFECT_OBSERVED_OR_PARTIAL'
    all_underfit=any(v['underfit_flag'] for v in opt_summary.get('FULL_V2',{}).values())
    informative=[];ablation_effects={}
    for arm in ARMS[2:]:
        matched=sorted(set(DIAG12)&set(records[arm])&set(records['FULL_V2']))
        if not matched:continue
        ref=pooled([records['FULL_V2'][d] for d in matched]);other=pooled([records[arm][d] for d in matched]);hits=promotion(ref,other)
        effects=dict(matched_days=len(matched),complete_diag12=len(matched)==12,threshold_hits=hits,
            balanced_delta=other['balanced_accuracy']-ref['balanced_accuracy'],direction_delta=other['raw_direction']-ref['raw_direction'],
            mae_relative=other['spread_mae']/ref['spread_mae']-1,crps_relative=other['crps']/ref['crps']-1,brier_delta=other['brier0']-ref['brier0'])
        ablation_effects[arm]=effects
        if len(matched)==12:
            informative.append((max(abs(effects['balanced_delta'])/.015,abs(effects['direction_delta'])/.015,
                abs(effects['mae_relative'])/.03,abs(effects['crps_relative'])/.03,abs(effects['brier_delta'])/.01),arm))
    structural_needed=[]
    if gate_status=='GATE_INACTIVE':structural_needed.append('FIXED_BUSINESS_GATES')
    if dependence_status=='DEPENDENCE_INACTIVE':structural_needed.append('NO_DYNAMIC_DEP')
    structural_pending=[a for a in structural_needed if any(d not in records[a] for d in primary)]
    complete=progress['status']=='COMPLETE' and not all_underfit and not structural_pending
    full,direct=metrics['FULL_V2'],metrics['DIRECT96'];diagnosis=[];token='MSR_V2_EXPERIMENT_BLOCKED'
    common=sorted(set(primary)&set(records['FULL_V2'])&set(records['DIRECT96']))
    full_matched=pooled([records['FULL_V2'][d] for d in common])
    direct_matched=pooled([records['DIRECT96'][d] for d in common])
    if full and direct:
        mae_ratio=full_matched['spread_mae']/direct_matched['spread_mae'];balanced=full_matched['balanced_accuracy']-direct_matched['balanced_accuracy'];direction=full_matched['raw_direction']-direct_matched['raw_direction']
        diagnosis.append(f'FULL/DIRECT MAE ratio={mae_ratio:.6f}; Balanced delta={balanced:.6f}; Direction delta={direction:.6f}')
        supported=[]
        for month in sorted(set(months['FULL_V2'])&set(months['DIRECT96'])):
            a,b=months['FULL_V2'][month],months['DIRECT96'][month]
            if (max(a['balanced_accuracy']-b['balanced_accuracy'],a['raw_direction']-b['raw_direction'])>=.02 and a['spread_mae']<=1.03*b['spread_mae']) or (a['spread_mae']<=.95*b['spread_mae'] and a['balanced_accuracy']>=b['balanced_accuracy']-.01):supported.append(month)
        promising=(max(balanced,direction)>=.02 and mae_ratio<=1.03) or (mae_ratio<=.95 and balanced>=-.01)
        if complete:
            token='MSR_V2_CORE_STRUCTURALLY_DOMINATED' if mae_ratio>1.05 and balanced<-.01 else (
                'MSR_V2_CORE_SCIENTIFICALLY_PROMISING' if promising and len(supported)>=2 else 'MSR_V2_CORE_MIXED_NEEDS_TARGETED_REDESIGN')
        diagnosis.append('Cross-month qualifying gains: '+str(supported))
        low=full.get('low',{}).get('correlation')
        if mae_ratio>=1 and low is not None and low<.3:diagnosis.append('Decomposition/Low representation is first suspect; evidence only, no redesign applied.')
        high=full.get('high',{}).get('correlation')
        if low is not None and high is not None and low>=.5 and high>=.5 and (balanced<0 or mae_ratio>=1):
            diagnosis.append('Both marginal components have moderate correlation (>=.5), but final business score is weaker: dependence/reconstruction/final objective is the first suspect, not component collapse.')
        if full['raw_direction']<full['all_negative_baseline']:
            diagnosis.append('FULL raw Direction is below the all-negative baseline; positive/negative recall and ranking need joint review. Not a business-direction breakthrough.')
        if (mae_ratio<1 and balanced<0) or (mae_ratio>1 and balanced>0):
            diagnosis.append('Numerical/sign tradeoff: retain paired Pareto evidence; no loss weights or architecture modified.')
    for arm,e in ablation_effects.items():
        if not e['complete_diag12']:
            diagnosis.append(arm+': partial matched diagnostic panel; no component adjudication or promotion from these partial scores.')
            continue
        dominates=e['mae_relative']<=0 and e['balanced_delta']>=0 and e['crps_relative']<=0
        if dominates and arm=='NO_DYNAMIC_DEP':diagnosis.append('Removing learned dependence is no worse on MAE/CRPS/Balanced in DIAG12; current dependence is implicated. Full-panel promotion remains mandatory if triggered.')
        if dominates and arm=='FIXED_BUSINESS_GATES':diagnosis.append('Fixed gates are no worse on MAE/CRPS/Balanced in DIAG12; learned routing is not earning complexity on this evidence.')
        if dominates and arm=='NO_LOW_TO_HIGH_CONTEXT':diagnosis.append('Removing Low->High context is no worse in DIAG12; conditional coupling may be redundant.')
        if arm=='NO_BRIER0' and e['mae_relative']<=-.03 and e['direction_delta']<=-.015:
            diagnosis.append('NO_BRIER0 improves MAE but materially harms Direction: Brier task alignment is justified; do not tune its weight in this run.')
    primary_ablation_effects={}
    for arm in ARMS[2:]:
        if primary and all(d in records[arm] for d in primary):
            reference=pooled([records['FULL_V2'][d] for d in primary])
            other=pooled([records[arm][d] for d in primary])
            primary_ablation_effects[arm]=dict(matched_days=len(primary),
                mae_relative=other['spread_mae']/reference['spread_mae']-1,
                crps_relative=other['crps']/reference['crps']-1,
                balanced_delta=other['balanced_accuracy']-reference['balanced_accuracy'],
                direction_delta=other['raw_direction']-reference['raw_direction'],
                brier_delta=other['brier0']-reference['brier0'])
    if 'NO_LOW_TO_HIGH_CONTEXT' in primary_ablation_effects:
        e=primary_ablation_effects['NO_LOW_TO_HIGH_CONTEXT']
        diagnosis.append('Promoted NO_LOW_TO_HIGH_CONTEXT primary-panel effect: '+str(e)+
            '; removal harms pooled MAE/Direction/Balanced if their deltas have those signs; Low conditioning is not shown redundant. DIAG12 magnitude need not persist on the full panel.')
    if 'NO_BRIER0' in ablation_effects and not ablation_effects['NO_BRIER0']['threshold_hits']:
        diagnosis.append('NO_BRIER0 is below every E4 threshold on complete DIAG12; no evidence here justifies changing its frozen coefficient.')
    gradient_summary={}
    for arm in ARMS:
        gradient_summary[arm]={}
        for group in sorted({r['group'] for r in gradient_rows if r['arm']==arm}):
            rows=[r for r in gradient_rows if r['arm']==arm and r['group']==group]
            norms=[r['final_norm'] for r in rows]
            item=dict(days=len(rows),final_norm_min=float(min(norms)),final_norm_median=float(np.median(norms)))
            for key in ('final_low_cosine','final_high_cosine'):
                values=[r[key] for r in rows if r.get(key) is not None]
                if values:item[key]=dict(mean=float(np.mean(values)),negative_fraction=float(np.mean(np.array(values)<0)))
            gradient_summary[arm][group]=item
    conflict=gradient_summary.get('FULL_V2',{}).get('high_head',{}).get('final_high_cosine',{})
    if conflict.get('negative_fraction',0)>.5:
        diagnosis.append('Persistent FULL_V2 final-vs-High marginal gradient conflict at high_head: '+str(conflict)+
            '; supports final-objective/component tradeoff diagnosis, not a zero-gradient or underfit collapse. No loss change applied.')
    if gate_status=='GATE_INACTIVE':diagnosis.append('Business gate curves remain near-identical/initial under stated diagnostic tolerances; next-round semantics review only, no WIND/SOLAR split applied.')
    if dependence_status=='DEPENDENCE_INACTIVE':diagnosis.append('Learned dependence stays near initialization and its removal is below E4 effect thresholds; simplification is a next-round hypothesis only.')
    if all_underfit:diagnosis.append('UNDERFIT floor+improving signal unresolved; component interpretation withheld.')
    if structural_pending:diagnosis.append('Structural-inactivity full-panel promotions still required: '+str(structural_pending))
    if not complete:diagnosis.append('Incomplete protocol or unresolved optimization gate; partial metrics are not a final scientific architecture judgment.')
    summary=dict(GPU_ENV='PASS',GPU_DEVICE=read(run/'GPU_QUALIFICATION.json')['cuda']['name'],BATCH_SIZE=contract['config']['batch_days'],
        MEDIAN_RUNTIME_PER_DAY=benchmark['median_seconds'] if benchmark else None,
        RUNTIME_BENCHMARK_SOURCE='immutable original RUNTIME_BENCHMARK.json',
        runtime_bookkeeping_discrepancy=progress['benchmark']!=benchmark['per_day'] if benchmark else False,
        PRIMARY_PANEL_DAYS=len(primary),FULL_V2=full,DIRECT96=direct,
        FULL_MINUS_DIRECT={k:full_matched[k]-direct_matched[k] for k in ('spread_mae','raw_direction','balanced_accuracy','crps','brier0')} if common else None,
        FULL_DIRECT_COMPARISON_DATES=common,
        LOW_CORR=full.get('low',{}).get('correlation') if full else None,HIGH_CORR=full.get('high',{}).get('correlation') if full else None,
        GATE_STATUS=gate_status,DEPENDENCE_STATUS=dependence_status,
        MOST_INFORMATIVE_ABLATION=max(informative)[1] if informative and not all_underfit else None,
        ARCHITECTURE_DIAGNOSIS=diagnosis,terminal_token=token,phase=progress['phase'],status=progress['status'],
        completed_days=progress['completed'],ablation_effects=ablation_effects,
        missing_initial_diagnostics=missing_initial,gate_summary=gate_summary,dependence_summary=dep_summary,
        structural_promotions_required=structural_needed,structural_promotions_pending=structural_pending,
        optimization_summary=opt_summary,gradient_summary=gradient_summary,primary_ablation_effects=primary_ablation_effects,source_sha256=SOURCE_SHA256,
        diagnostic_tolerances='Descriptive only: gate pair/init RMS<.01 and cutoff std<.1; dependence init RMS<.01 plus below E4 metric thresholds. Not a model-selection sweep.',
        report_root=str(report))
    csv(report/'GATE_DIAGNOSTICS.csv',gate_rows);write_json(report/'GATE_DIAGNOSTICS.json',dict(summary=gate_summary,status=gate_status,missing_initial=missing_initial))
    csv(report/'DEPENDENCE_DIAGNOSTICS.csv',dep_rows);write_json(report/'DEPENDENCE_DIAGNOSTICS.json',dict(summary=dep_summary,status=dependence_status))
    csv(report/'OPTIMIZATION_DIAGNOSTICS.csv',opt_rows);csv(report/'GRADIENT_DIAGNOSTICS.csv',gradient_rows)
    write_json(report/'OPTIMIZATION_DIAGNOSTICS.json',dict(summary=opt_summary,gradient_rows=gradient_rows,gradient_summary=gradient_summary,underfit=all_underfit))
    write_json(report/'ADJUDICATION.json',summary)
    text='# MSR-V2 GPU evidence (complete or deadline-partial)\n\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n\n'+token+'\n'
    (report/'ADJUDICATION.md').write_text(text,encoding='utf-8')
    if args.final:
        import shutil
        for name in ('COMPONENT_COMPARISON.csv','GATE_DIAGNOSTICS.csv','GATE_DIAGNOSTICS.json','DEPENDENCE_DIAGNOSTICS.csv','DEPENDENCE_DIAGNOSTICS.json',
                     'OPTIMIZATION_DIAGNOSTICS.csv','OPTIMIZATION_DIAGNOSTICS.json','ADJUDICATION.json','ADJUDICATION.md','DAILY_METRICS.csv','MONTHLY_METRICS.json','POOLED_METRICS.json','GRADIENT_DIAGNOSTICS.csv'):
            if (report/name).exists():shutil.copyfile(report/name,assert_write_path(run/name))
    print(json.dumps(dict(report_root=str(report),terminal_token=token,completed={k:len(v) for k,v in records.items()},summary_file=str(report/'ADJUDICATION.json'))),flush=True)

if __name__=='__main__':main()
