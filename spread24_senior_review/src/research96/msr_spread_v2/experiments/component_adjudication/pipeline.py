"""Ordered docs86 execution with optional user deadline and durable progress."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,json,time,shutil,datetime,gc
from pathlib import Path
import numpy as np
import torch
from ...config import OUTPUT_ROOT,FAST24,LAB_ROOT
from ...contracts import assert_write_path,ContractError
from ...audit import package_hashes,file_hash,frozen_check
from ...dataset import CurveStore,training_panel,build_sample,evaluation_label
from ...artifacts import new_run,write_json,to_numpy,save_day,run_manifest,write_csv
from ...evaluate import evaluate_day
from .profile import NAME,BOUNDS,scientific_config,ARMS
from .trainer import fit_day,synchronized_time
from .metrics import pooled,monthly,promotion

DIAG12=('2026-01-05','2026-01-25','2026-02-15','2026-03-05','2026-03-31','2026-04-01',
        '2026-04-25','2026-05-15','2026-06-15','2026-07-05','2026-07-25','2026-08-13')
BENCH=('2026-01-15','2026-04-01','2026-07-15')
EXTENSION=('2026-01-31','2026-02-28','2026-03-01','2026-04-30','2026-07-31','2026-08-14')

def load_record(root,date):
    root=assert_write_path(root)
    with np.load(root/date/'DISTRIBUTIONS.npz',allow_pickle=False) as f:
        saved={k:f[k] for k in f.files}
    truth_path=root/date/'EVALUATION_TRUTH.npz'
    if truth_path.exists():
        with np.load(truth_path,allow_pickle=False) as f: truth=f['truth']
    else:
        # Reused source has a documented prediction seal before its first label read.
        seal=json.loads((root/'sealed'/date/'PREDICTION_SEAL.json').read_text(encoding='utf-8'))
        assert seal['prediction_sha256']==file_hash(assert_write_path(root/'sealed'/date/'PREDICTION_BEFORE_TRUTH.npz'))
        truth=evaluation_label(date)
    return dict(date=date,out=saved,truth=truth,training=json.loads((root/date/'TRAINING.json').read_text(encoding='utf-8')),
                root=str(root))

def run_day(run,arm,date,batch):
    started=synchronized_time()
    root=assert_write_path(run/arm/date);root.mkdir(parents=True,exist_ok=False)
    t=time.perf_counter();store=CurveStore.canonical(date);source=time.perf_counter()-t
    t=time.perf_counter();samples,labels,rejected=training_panel(store,date,smoke=False);sample=build_sample(store,date)
    data=time.perf_counter()-t
    write_json(root/'DATA_AUDIT.json',dict(source=store.source_identity,rejected=rejected,eligible_days=len(samples),**sample.provenance))
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    t=synchronized_time();model,pre,training=fit_day(samples,labels,date,batch,arm);fit=synchronized_time()-t
    write_json(root/'TRAINING_COMPLETED.json',training)
    # Initialization comparison is sampled with the refit preprocessing and identical seed.
    from ...trainer import seed_everything
    from .arms import make_model
    rng_cpu=torch.get_rng_state();rng_cuda=torch.cuda.get_rng_state()
    seed_everything(20261009);initial=make_model(sample.names,scientific_config(batch,arm),pre.stats,arm).cuda().eval()
    x=torch.as_tensor(pre.transform(sample.x[None]),device='cuda')
    da=torch.as_tensor(sample.da[None],dtype=torch.float64,device='cuda')
    with torch.no_grad():
        init=to_numpy(initial(x,scenarios=32));model.eval();out=model(x,da,scenarios=256)
    torch.set_rng_state(rng_cpu);torch.cuda.set_rng_state(rng_cuda)
    saved=to_numpy(out);del initial,init['spread_scenarios']
    write_json(root/'INITIAL_DIAGNOSTICS.json',{k:init[k] for k in ('spectral','dependence') if k in init})
    sealed=assert_write_path(root/'sealed'/date);sealed.mkdir(parents=True,exist_ok=False)
    prediction=assert_write_path(sealed/'PREDICTION_BEFORE_TRUTH.npz')
    with prediction.open('xb') as f:np.savez_compressed(f,spread_scenarios=saved['spread_scenarios'],
        spread_median=saved['spread_median'],p_positive=saved['p_positive'],target_da=sample.da)
    write_json(sealed/'PREDICTION_SEAL.json',dict(prediction_sha256=file_hash(prediction),target_day=date,
        target_truth_opened=False,max_label_day=training['max_label_day']))
    truth=evaluation_label(date)
    if arm!='DIRECT96':
        metrics,slots=evaluate_day(saved,truth,sample.da)
        save_day(root,sample,out,model,pre,training,metrics,slots,training['timings'])
    else:
        dayroot=assert_write_path(root/date);dayroot.mkdir()
        with (dayroot/'DISTRIBUTIONS.npz').open('xb') as f:
            np.savez_compressed(f,**{k:saved[k] for k in ('spread_scenarios','spread_quantiles','spread_median','rt_median','rt_quantiles','p_positive','direct_q_raw','clamp_fraction')})
        write_json(dayroot/'TRAINING.json',training);write_json(dayroot/'PREPROCESSOR.json',pre.manifest())
        with (dayroot/'MODEL.pt').open('xb') as f:torch.save(model.state_dict(),f)
        write_json(dayroot/'METRICS.json',pooled([dict(out=saved,truth=truth)]))
    with (root/date/'EVALUATION_TRUTH.npz').open('xb') as f:np.savez_compressed(f,truth=truth,target_da=sample.da)
    record=dict(date=date,out=saved,truth=truth,training=training,root=str(root))
    if 'spectral' in saved:write_json(root/'FINAL_DIAGNOSTICS.json',dict(spectral=saved['spectral'],dependence=saved['dependence']))
    timings=dict(source_read=source,data=data,fit_total=fit,total_day=synchronized_time()-started,**training['timings'])
    write_json(root/'RUNTIME.json',dict(arm=arm,target_day=date,timings=timings,
        parameter_count=sum(p.numel() for p in model.parameters()),selected_epochs=training['selected_epochs'],
        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
    print(f'DAY_COMPLETE arm={arm} day={date} seconds={timings["total_day"]:.2f}',flush=True)
    del model,out,x,labels,samples,store;gc.collect();torch.cuda.empty_cache()
    return record,timings['total_day']

def main():
    p=argparse.ArgumentParser();p.add_argument('--qualification',required=True);p.add_argument('--baseline',required=True)
    p.add_argument('--review',required=True);p.add_argument('--reuse',required=True);p.add_argument('--deadline')
    p.add_argument('--resume')
    args=p.parse_args();deadline=datetime.datetime.fromisoformat(args.deadline).timestamp() if args.deadline else None
    q=json.loads(Path(args.qualification).read_text(encoding='utf-8'));assert q['status']=='PASS' and torch.cuda.is_available()
    batch=q['batch_size'];baseline=json.loads(Path(args.baseline).read_text(encoding='utf-8'));code=package_hashes()
    assert all(code.get(k)==v for k,v in baseline['v2_core'].items()),'core changed'
    review=json.loads(Path(args.review).read_text(encoding='utf-8'))
    assert review['status']=='PASS' and review['independent_review']=='PASS'
    assert review['code_hashes']=={k:v for k,v in baseline['v2_core'].items() if k.endswith('.py')}
    run=assert_write_path(args.resume) if args.resume else new_run(OUTPUT_ROOT/'v2_gpu_component_adjudication_20261009')
    continuation=new_run(run/'continuations') if args.resume else None
    print('RUN_ROOT='+str(run),flush=True)
    records={a:{} for a in ARMS};runtimes={a:[] for a in ARMS};primary=[];bench={};promoted={};phase='G2'
    frozen=dict(profile=NAME,config=scientific_config(batch).to_dict(),bounds=BOUNDS,
        benchmark=BENCH,diag12=DIAG12,fast24=FAST24,extension=EXTENSION,deadline=args.deadline,code_hashes=code,
        core_review_sha=file_hash(args.review),baseline_sha=file_hash(args.baseline),
        legal_source='canonical projections <= day; RT complete <=D-2, D-1 prefix56; truth opened only after seal',
        no_op_stage_B='DIRECT96 and NO_DYNAMIC_DEP lack learnable dependence; B remains explicit no-op, never repurposed')
    if not args.resume:write_json(run/'FROZEN_RUN_CONTRACT.json',frozen)
    else:
        original=json.loads((run/'FROZEN_RUN_CONTRACT.json').read_text(encoding='utf-8'))
        assert original['config']==frozen['config']
        module_names=('arms.py','trainer.py','profile.py','metrics.py')
        assert all(code['experiments/component_adjudication/'+n]==original['code_hashes']['experiments/component_adjudication/'+n] for n in module_names),'training/math changed on resume'
        prior=json.loads((run/'PROGRESS.json').read_text(encoding='utf-8'))
        write_json(continuation/'RESUME_RECEIPT.json',dict(previous_progress_sha=file_hash(run/'PROGRESS.json'),previous_error=prior.get('error'),
            reason='User authorized continuation beyond noon; repeatable versioned receipts, optional deadline and structural promotion orchestration only; no model/trainer/loss/profile changes',
            previous_deadline=prior.get('deadline'),current_deadline=args.deadline,
            old_pipeline_sha=original['code_hashes']['experiments/component_adjudication/pipeline.py'],new_pipeline_sha=code['experiments/component_adjudication/pipeline.py']))
    qualroot=Path(args.qualification).parent
    for name in ('ENV_BEFORE.json','ENV_BEFORE.txt','ENV_AFTER.json','ENV_AFTER.txt','GPU_QUALIFICATION.json','CONDA_BEFORE.json','CONDA_AFTER.json'):
        path=assert_write_path(qualroot/name)
        if path.exists() and not (run/name).exists():shutil.copyfile(path,assert_write_path(run/name))
    reuse=assert_write_path(args.reuse);one=json.loads((reuse/'ONE_DAY_RUNTIME.json').read_text(encoding='utf-8'))
    assert one['profile']==NAME and one['batch_size']==batch and one['target_day']==BENCH[0]
    z=json.loads((reuse/'ZERO_POLLUTION.json').read_text(encoding='utf-8'))
    assert not z['v2_core_drift'] and all(not any(z[k].values()) for k in ('v1','tafm','production_source'))
    records['FULL_V2'][BENCH[0]]=load_record(reuse,BENCH[0]);bench[BENCH[0]]=one['timings']['total_day']
    runtimes['FULL_V2'].append(bench[BENCH[0]])
    reuse_receipt=dict(source=str(reuse),source_receipt_sha=file_hash(reuse/'ONE_DAY_RUNTIME.json'),
        completed_training_reused=True,source_and_core_unchanged=True,external_production_log_metadata=z['production_metadata'],
        original_post_audit_block='concurrent external pull.log metadata only; not discarded or silently labelled zero drift')
    if not args.resume:write_json(run/'REUSE_RECEIPT.json',reuse_receipt)
    else:
        primary=prior['primary'];bench=prior['benchmark'];promoted=prior['promotions']
        for arm,dates in prior['completed'].items():
            for date in dates:
                path=run/arm/date
                if arm=='FULL_V2' and not path.exists():path=reuse
                records[arm][date]=load_record(path,date)
                if (path/'RUNTIME.json').exists():runtimes[arm].append(json.loads((path/'RUNTIME.json').read_text(encoding='utf-8'))['timings']['total_day'])
    def progress(status='RUNNING',error=None):
        obj=dict(status=status,phase=phase,completed={a:sorted(r) for a,r in records.items()},primary=primary,
            benchmark=bench,promotions=promoted,deadline=args.deadline,error=error,
            pooled={a:pooled(list(rs.values())) for a,rs in records.items()},monthly={a:monthly(list(rs.values())) for a,rs in records.items()})
        path=assert_write_path(run/'PROGRESS.json');tmp=assert_write_path(run/'PROGRESS.next.json')
        tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)
    def ensure(arm,date):
        if date in records[arm]:return records[arm][date]
        estimate=max(runtimes[arm] or runtimes['FULL_V2'] or [180])*1.15
        if deadline is not None and time.time()+estimate+30>=deadline:raise TimeoutError('User deadline: insufficient budget for another complete day')
        record,seconds=run_day(run,arm,date,batch);records[arm][date]=record;runtimes[arm].append(seconds);progress()
        return record
    try:
        progress()
        for date in BENCH[1:]:
            ensure('FULL_V2',date)
            if date not in bench:
                bench[date]=json.loads((run/'FULL_V2'/date/'RUNTIME.json').read_text(encoding='utf-8'))['timings']['total_day']
        median=float(np.median(list(bench.values())))
        primary=sorted((*FAST24,*EXTENSION)) if median<=75 else list(FAST24)
        if not (run/'RUNTIME_BENCHMARK.json').exists():
            write_json(run/'RUNTIME_BENCHMARK.json',dict(per_day=bench,median_seconds=median,primary_panel_days=len(primary),
                rule='median <=75 -> FULL30; otherwise FAST24',primary_dates=primary))
        phase='E1_FULL_V2';progress()
        for date in primary:ensure('FULL_V2',date)
        phase='E2_DIRECT96';progress()
        for date in primary:ensure('DIRECT96',date)
        phase='E3_DIAG12';progress()
        for arm in ARMS[2:]:
            for date in DIAG12:ensure(arm,date)
        phase='E4_PROMOTION';progress()
        # Apply the existing evidence report's structural-inactivity and underfit
        # gates before promotion, not just its numerical thresholds.
        import subprocess,sys
        generated=subprocess.check_output([sys.executable,'-B','-m',
            'src.research96.msr_spread_v2.experiments.component_adjudication.report',
            '--run',str(run)],text=True)
        report_path=Path(json.loads(generated.strip().splitlines()[-1])['summary_file'])
        diagnostic=json.loads(assert_write_path(report_path).read_text(encoding='utf-8'))
        if any(v['underfit_flag'] for v in diagnostic['optimization_summary']['FULL_V2'].values()):
            raise ContractError('Frozen FULL_V2 schedule underfit; component interpretation withheld, no unapproved schedule changes')
        structural=set(diagnostic['structural_promotions_required'])
        reference=pooled([records['FULL_V2'][d] for d in DIAG12])
        for arm in ARMS[2:]:
            hits=promotion(reference,pooled([records[arm][d] for d in DIAG12]))
            if arm in structural:hits.append('structural_inactivity')
            if hits:
                promoted[arm]=hits;progress()
                for date in primary:ensure(arm,date)
        phase='COMPLETE';progress('COMPLETE')
    except TimeoutError as e:progress('PARTIAL_USER_DEADLINE',str(e))
    except Exception as e:
        progress('BLOCKED',type(e).__name__+': '+str(e));raise
    finally:
        audit_root=continuation if args.resume else run
        write_json(audit_root/'SOURCE_HASHES.json',dict(initial=code,current=package_hashes(),source=baseline.get('source_identity'),
            command=' '.join(__import__('sys').argv)))
        isolation=frozen_check(args.baseline,production=True)
        isolation['boundary']='All training/artifact writes guarded inside V2 roots; no production runtime imports or elevated training'
        isolation['protected_source_zero_drift']=all(not any(isolation[k].values()) for k in ('v1','tafm','production_source'))
        write_json(audit_root/'ZERO_POLLUTION.json',isolation)
        print('PIPELINE_RESULT='+str(run),flush=True)

if __name__=='__main__':main()
