"""Read-only frozen-panel oracle/error/context diagnosis; never training or deployment."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,json,hashlib,datetime,sys
from pathlib import Path
import numpy as np
import torch
from ...artifacts import new_run,write_json,write_csv,to_numpy
from ...contracts import assert_write_path
from ...config import OUTPUT_ROOT,FAST24,Q_KNOTS
from ...audit import package_hashes,file_hash,frozen_check
from ...dataset import CurveStore,build_sample
from ...feature_registry import registry
from ...trainer import Preprocessor,seed_everything
from ...model import MSRSpreadV2
from ...quantile_head import quantile,low_summary
from ...t24 import operator
from ...spectral_router import dct_matrix
from .pipeline import load_record
from .profile import scientific_config
from .metrics import pooled,monthly

def read(path):
    return json.loads(assert_write_path(path).read_text(encoding='utf-8'))

def scenario_record(date,scenarios,truth):
    s=np.asarray(scenarios,dtype=np.float64)
    assert s.shape==(256,96) and np.isfinite(s).all()
    q=np.quantile(s,Q_KNOTS,axis=0).T
    return dict(date=date,truth=truth,out=dict(spread_scenarios=s[None],
        spread_quantiles=q[None],spread_median=q[None,:,4],p_positive=(s>0).mean(0)[None]))

def error_anatomy(pred,truth):
    e=np.asarray(pred,dtype=np.float64)-truth
    bias=float(e.mean());mse=float(np.mean(e**2));centered=float(np.mean((e-bias)**2))
    assert abs(mse-bias*bias-centered)<1e-6*max(mse,1)
    return dict(mae=float(abs(e).mean()),mse=mse,bias=bias,centered_mse=centered,
        correlation=float(np.corrcoef(pred,truth)[0,1]) if np.std(pred)>0 and np.std(truth)>0 else None)

def structure(curve):
    curve=np.asarray(curve,dtype=np.float64)
    coefficients=curve@dct_matrix().numpy().T
    energy=(coefficients[1:]**2).sum()
    return dict(roughness=float(np.mean(np.diff(curve,n=2)**2)),
        # Bin8 is the frozen T24 characteristic half-gain index, not a tuned cutoff.
        ac_energy_bin8_plus=float((coefficients[8:]**2).sum()/energy) if energy>1e-12 else 0.,
        zero_crossings=int(np.count_nonzero(np.diff(curve>0))))

def context_hook(mode,normalized_true):
    """High inputs only: no change to Low output/latent or saved ranks."""
    def hook(module,args):
        x,width,ramp,mask,*tail=args
        new=x.clone();summary=new[...,-7:]
        if mode=='ZERO_CONTEXT':
            summary.zero_();width=torch.zeros_like(width)
        elif mode=='ORACLE_LOCATION_CONTEXT':
            shift=normalized_true-summary[...,2]
            summary[...,:5]+=shift[...,None]
        elif mode=='ORACLE_DEGENERATE_CONTEXT':
            summary[...,:5]=normalized_true[...,None];summary[...,5:]=0
            width=torch.zeros_like(width)
        elif mode=='PRED_LOCATION_ZERO_WIDTH':
            summary[...,:5]=summary[...,2:3].clone();summary[...,5:]=0
            width=torch.zeros_like(width)
        else:raise ValueError(mode)
        return (new,width,ramp,mask,*tail)
    return hook

def reload_model(record,sample,batch):
    root=assert_write_path(Path(record['root'])/record['date'])
    manifest=read(root/'PREPROCESSOR.json')
    pre=Preprocessor();pre.median=np.array(manifest['median']);pre.scale=np.array(manifest['scale'])
    pre.stats=manifest['stats'];pre.fit_days=manifest['fit_days']
    pre.bypass_ids=[i for i,v in enumerate(registry().values()) if v['role']=='bypass']
    seed_everything(20261009)
    model=MSRSpreadV2(sample.names,scientific_config(batch),pre.stats).cuda().eval()
    state=torch.load(assert_write_path(root/'MODEL.pt'),map_location='cuda',weights_only=True)
    model.load_state_dict(state,strict=True)
    x=torch.as_tensor(pre.transform(sample.x[None]),device='cuda')
    return model,x

def flatten(arm,metric,panel):
    return dict(arm=arm,panel=panel,**{k:v for k,v in metric.items() if not isinstance(v,(list,dict))})

def selection_drift(record):
    training=record['training'];history=training['history'];selected=training['selected_epochs']
    a=next(h for h in history if h['phase']=='selection' and h['stage']=='A' and h['epoch']==selected['A'])
    c=next(h for h in history if h['phase']=='selection' and h['stage']=='C' and h['epoch']==selected['C'])
    return dict(date=record['date'],A_selected=selected['A'],C_selected=selected['C'],
        validation_low_A=a['validation']['low'],validation_low_C=c['validation']['low'],
        validation_high_A=a['validation']['high'],validation_high_C=c['validation']['high'],
        low_relative_change=c['validation']['low']/a['validation']['low']-1,
        high_relative_change=c['validation']['high']/a['validation']['high']-1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--reuse',required=True)
    p.add_argument('--baseline',required=True);args=p.parse_args()
    run=assert_write_path(args.run);reuse=assert_write_path(args.reuse)
    progress=read(run/'PROGRESS.json');assert progress['status']=='COMPLETE'
    assert tuple(progress['primary'])==FAST24
    assert torch.cuda.is_available(),'CUDA inference required, no environment changes'
    baseline=read(args.baseline);initial_code=package_hashes()
    assert all(initial_code.get(k)==v for k,v in baseline['v2_core'].items())
    output=new_run(OUTPUT_ROOT/'low_diagnosis_20261010');print('DIAGNOSIS_ROOT='+str(output),flush=True)
    preflight=frozen_check(args.baseline,production=True)
    write_json(output/'PREFLIGHT_ZERO_POLLUTION.json',preflight)
    assert all(not any(preflight[k].values()) for k in ('v1','tafm','production_source'))
    contract=Path(__file__).with_name('LOW_DIAGNOSIS_CONTRACT.md')
    write_json(output/'CONTRACT.json',dict(classification='ORACLE/PRIVILEGED mechanisms; original baseline only STRICT chronology under docs84 research assumptions',
        dates=FAST24,no_training=True,no_new_truth_panel=True,contract_sha256=file_hash(contract),
        original_run=str(run),source_progress_sha256=file_hash(run/'PROGRESS.json'),
        command=' '.join(sys.argv),code_hashes=initial_code,
        evaluation_semantics='Scenario Direction uses P(S>0)>.5; point comparison uses component knot medians, separately labelled',
        reproduction_tolerance='raw-unit atol=.002 rtol=1e-5; T24 identity max error1e-8',
        interventions='T24 smoothing uses frozen soft non-idempotent operator, no parameter tuning'))
    arms={};anatomy=[];structures=[];drift=[];reproductions=[];context_rows=[];point_records={};input_provenance=[]
    def add(name,date,s,truth):arms.setdefault(name,[]).append(scenario_record(date,s,truth))
    try:
        for date in FAST24:
            root=run/'FULL_V2'/date
            if not root.exists():root=reuse
            record=load_record(root,date);truth=record['truth'];saved=record['out']
            seal=read(root/'sealed'/date/'PREDICTION_SEAL.json')
            assert seal['prediction_sha256']==file_hash(assert_write_path(root/'sealed'/date/'PREDICTION_BEFORE_TRUTH.npz'))
            ls,hs=(saved[k][0].astype(np.float64) for k in ('low_scenarios','high_scenarios'))
            lm,hm=(saved[k][0,:,4].astype(np.float64) for k in ('low_q_raw','high_q_raw'))
            true_l,true_h,y=truth[:,1],truth[:,2],truth[:,0]
            assert np.max(abs(true_l+true_h-y))<1e-8
            arms.setdefault('BASELINE_SAVED',[]).append(record)
            add('ORACLE_TRUE_LOW_PRED_HIGH_FROZEN',date,true_l[None]+hs,truth)
            add('ORACLE_PRED_LOW_TRUE_HIGH',date,ls+true_h[None],truth)
            exact=np.broadcast_to(true_l+true_h,(256,96)).copy()
            # Declared numerical identity tolerance, never a direction threshold search.
            exact[:,abs(y)<1e-10]=0
            add('ORACLE_TRUE_LOW_TRUE_HIGH_IDENTITY',date,exact,truth)
            mean_shift=true_l.mean()-lm.mean();shape_shift=(true_l-true_l.mean())-(lm-lm.mean())
            add('ORACLE_LOW_LEVEL_ONLY',date,ls+mean_shift+hs,truth)
            add('ORACLE_LOW_SHAPE_ONLY',date,ls+shape_shift[None]+hs,truth)
            add('FIXED_T24_SMOOTH_PRED_LOW',date,ls@operator().T+hs,truth)
            for name,point in [('POINT_SAVED_SCENARIO_MEDIAN',saved['spread_median'][0]),
                ('POINT_SUM_COMPONENT_MEDIANS',lm+hm),('POINT_TRUE_LOW_PRED_HIGH',true_l+hm),
                ('POINT_PRED_LOW_TRUE_HIGH',lm+true_h)]:
                point_records.setdefault(name,[]).append(scenario_record(date,np.broadcast_to(point,(256,96)),truth))
            for name,pred,target in [('low',lm,true_l),('high',hm,true_h)]:
                anatomy.append(dict(date=date,component=name,**error_anatomy(pred,target)))
                structures.append(dict(date=date,component=name,kind='predicted_median',**structure(pred)))
                structures.append(dict(date=date,component=name,kind='true_component',**structure(target)))
            drift.append(selection_drift(record))
            store=CurveStore.canonical(date);sample=build_sample(store,date)
            input_provenance.append(dict(date=date,**sample.provenance,source=store.source_identity))
            model,x=reload_model(record,sample,read(run/'GPU_QUALIFICATION.json')['batch_size'])
            da=torch.as_tensor(sample.da[None],dtype=torch.float64,device='cuda')
            with torch.no_grad():original=model(x,da,scenarios=256)
            errors={k:float(np.max(abs(to_numpy(original[k])-saved[k]))) for k in
                ('low_q_raw','high_q_raw','low_scenarios','high_scenarios','spread_scenarios')}
            assert all(np.allclose(to_numpy(original[k]),saved[k],atol=.002,rtol=1e-5) for k in errors),errors
            reproductions.append(dict(date=date,**errors))
            _,ranks,_,_=model.dependence.ranks(original['dependence'],256)
            normalized_true=torch.as_tensor((true_l-float(model.component_center[0]))/float(model.component_scale[0]),dtype=x.dtype,device='cuda')[None]
            for mode in ('ZERO_CONTEXT','ORACLE_LOCATION_CONTEXT','ORACLE_DEGENERATE_CONTEXT','PRED_LOCATION_ZERO_WIDTH'):
                handle=model.high.register_forward_pre_hook(context_hook(mode,normalized_true))
                try:
                    with torch.no_grad():changed=model(x,da,scenarios=256)
                finally:handle.remove()
                with torch.no_grad():h,_=quantile(changed['high_q'],ranks)
                hf=to_numpy(h*model.component_scale[1]+model.component_center[1])[0]
                if mode.startswith('ORACLE'):
                    add(mode+'_TRUE_LOW_FIXED_RANKS',date,true_l[None]+hf,truth)
                    add(mode+'_TRUE_LOW_NATURAL_RANKS',date,true_l[None]+to_numpy(changed['high_scenarios'])[0],truth)
                else:add(mode+'_PRED_LOW_FIXED_RANKS',date,ls+hf,truth)
                row=dict(date=date,mode=mode,high_q_median_change_mae=float(abs(to_numpy(changed['high_q_raw'])[0,:,4]-hm).mean()),
                    high_fusion_mean=float(to_numpy(changed['fusion_high']).mean()),
                    high_mae=float(abs(to_numpy(changed['high_q_raw'])[0,:,4]-true_h).mean()),
                    original_high_mae=float(abs(hm-true_h).mean()))
                context_rows.append(row)
            del model,x,store;torch.cuda.empty_cache()
            print('DAY_DIAGNOSED='+date,flush=True)
        metrics={k:pooled(v) for k,v in arms.items()};months={k:monthly(v) for k,v in arms.items()}
        points={k:pooled(v) for k,v in point_records.items()}
        tables=[flatten(k,v,'FAST24') for k,v in metrics.items()]
        write_csv(output/'ORACLE_COMPARISON.csv',tables);write_json(output/'ORACLE_COMPARISON.json',metrics)
        write_json(output/'MONTHLY_ORACLE.json',months);write_json(output/'POINT_COMPARISON.json',points)
        write_csv(output/'ERROR_ANATOMY.csv',anatomy);write_csv(output/'STRUCTURE_DIAGNOSTICS.csv',structures)
        write_csv(output/'SELECTION_STAGE_DRIFT.csv',drift);write_csv(output/'CONTEXT_SENSITIVITY.csv',context_rows)
        write_json(output/'REPRODUCTION.json',reproductions);write_json(output/'INPUT_PROVENANCE.json',input_provenance)
        # Variance/covariance anatomy uses knot medians, not an assumed identity
        # between summed medians and the median of the summed distribution.
        covariance=[];slice_rows=[]
        for i,date in enumerate(FAST24):
            rec=arms['BASELINE_SAVED'][i];out=rec['out'];t=rec['truth']
            el=out['low_q_raw'][0,:,4]-t[:,1];eh=out['high_q_raw'][0,:,4]-t[:,2]
            covariance.append(dict(date=date,low_mse=float(np.mean(el**2)),high_mse=float(np.mean(eh**2)),
                cross_term=float(2*np.mean(el*eh)),sum_component_median_mse=float(np.mean((el+eh)**2))))
            for subset,mask in [('true_positive',t[:,0]>0),('true_nonpositive',t[:,0]<=0),
                ('abs_spread_le20',abs(t[:,0])<=20),('abs_spread_20to50',(abs(t[:,0])>20)&(abs(t[:,0])<=50)),
                ('abs_spread_50to100',(abs(t[:,0])>50)&(abs(t[:,0])<=100)),('abs_spread_gt100',abs(t[:,0])>100)]:
                for arm,records in arms.items():
                    v=records[i]['out'];n=int(mask.sum())
                    slice_rows.append(dict(date=date,arm=arm,subset=subset,n=n,
                        correct=int(np.count_nonzero(((v['p_positive'][0]>.5)==(t[:,0]>0))&mask)),
                        absolute_error_sum=float(abs(v['spread_median'][0]-t[:,0])[mask].sum())))
        write_csv(output/'ERROR_COVARIANCE.csv',covariance);write_csv(output/'DIRECTION_SLICES.csv',slice_rows)
        basic={}
        for side in ('low','high'):
            rr=[x for x in anatomy if x['component']==side];total=np.mean([x['mse'] for x in rr]);level=np.mean([x['bias']**2 for x in rr])
            basic[side]=dict(rmse=float(np.sqrt(total)),daily_level_bias_rmse=float(np.sqrt(level)),
                level_bias_share_of_mse=float(level/total),negative_daily_correlation_days=sum(x['correlation'] is not None and x['correlation']<0 for x in rr))
        stage_summary={side:dict(median_relative_change=float(np.median([x[side+'_relative_change'] for x in drift])),
            worsened_days=sum(x[side+'_relative_change']>0 for x in drift)) for side in ('low','high')}
        summary=dict(status='COMPLETE',classification='ORACLE/PRIVILEGED diagnostic only; consumed development panel',
            days=24,metrics=metrics,point_metrics=points,error_summary=basic,selection_validation_drift=stage_summary,
            saved_checkpoints_reproduced=24,core_unchanged=True,
            limitations=['Real Low + saved High retains High conditioned on wrong predicted Low.',
                'Oracle context holds original Low latent and original ranks in fixed-rank arm; no hypothetical perfectly trained High.',
                'Oracle replacement removes uncertainty, so CRPS/Brier gain is not achievable evidence.',
                '24 days, single seed, retrospective; selection-stage scores are not saved refit Stage-A OOS predictions.',
                'T24 is a soft smoother, not a hard projection; repeated smoothing would also alter true Low.'])
        write_json(output/'SUMMARY.json',summary)
        (assert_write_path(output/'REPORT.md')).write_text('# Low diagnosis\n\n'+json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
        print('LOW_DIAGNOSIS_COMPLETE='+str(output),flush=True)
    finally:
        current=package_hashes();assert all(current.get(k)==v for k,v in baseline['v2_core'].items())
        audit=frozen_check(args.baseline,production=True)
        audit['protected_source_zero_drift']=all(not any(audit[k].values()) for k in ('v1','tafm','production_source'))
        write_json(output/'ZERO_POLLUTION.json',audit)
        write_json(output/'SOURCE_HASHES.json',dict(initial=initial_code,current=current))
        assert audit['protected_source_zero_drift'],'protected source changed'

if __name__=='__main__':main()
