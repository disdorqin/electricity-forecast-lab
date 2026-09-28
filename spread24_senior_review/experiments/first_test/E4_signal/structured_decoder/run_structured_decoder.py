from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.TafM_改进源码.config import default_selector_path
from src.TafM_改进源码.dataset import SequenceStore, load_frozen_selector
from src.TafM_改进源码.target_adapter import source_to_model_target
from src.TafM_改进源码.train import stage_a_split_indices
from src.TafM_改进源码.metrics import canonical_metrics

Q2=ROOT/'experiments/first_test/E2_architecture/horizon_specialized_head/runs'
OUT=ROOT/'experiments/first_test/E4_signal/structured_decoder'
OUT.mkdir(parents=True,exist_ok=True)

def fit_transitions(y, hour_specific: bool, alpha=1.0):
    # y [N,24] bool
    init=np.array([(y[:,0]==s).sum()+alpha for s in (0,1)],float); init/=init.sum()
    if hour_specific:
        trans=np.empty((23,2,2),float)
        for h in range(23):
            c=np.full((2,2),alpha,float)
            for a,b in zip(y[:,h],y[:,h+1]): c[int(a),int(b)]+=1
            trans[h]=c/c.sum(axis=1,keepdims=True)
    else:
        c=np.full((2,2),alpha,float)
        for h in range(23):
            for a,b in zip(y[:,h],y[:,h+1]): c[int(a),int(b)]+=1
        t=c/c.sum(axis=1,keepdims=True)
        trans=np.repeat(t[None,:,:],23,axis=0)
    return init,trans

def viterbi(p, init, trans, beta=1.0):
    p=np.clip(np.asarray(p,float),1e-6,1-1e-6)
    emit=np.column_stack([1-p,p])
    dp=np.full((24,2),-np.inf); back=np.zeros((24,2),int)
    dp[0]=float(beta)*np.log(init)+np.log(emit[0])
    for h in range(1,24):
        for s in (0,1):
            cand=dp[h-1]+float(beta)*np.log(trans[h-1,:,s])
            j=int(np.argmax(cand)); back[h,s]=j
            dp[h,s]=cand[j]+np.log(emit[h,s])
    z=np.zeros(24,int); z[-1]=int(np.argmax(dp[-1]))
    for h in range(23,0,-1): z[h-1]=back[h,z[h]]
    return z

def summary(df):
    out=[]
    for arm in ['S0','S1','S2']:
        z=df[df.arm==arm]
        m=canonical_metrics(z.y,z.pred,np.ones(len(z)),direction_probability=z.p)
        out.append({'arm':arm,**{k:m[k] for k in ['raw_direction_accuracy','balanced_accuracy','positive_recall','nonpositive_recall','auc','brier']},
                    'ppf':float(z.pred.mean()),
                    'flips':int(sum(np.sum(g.pred.to_numpy()[1:]!=g.pred.to_numpy()[:-1]) for _,g in z.groupby('target_day')))})
    return pd.DataFrame(out)

def paired(df,a,b,B=10000):
    da=df[df.arm==a].groupby('target_day').correct.mean()
    db=df[df.arm==b].groupby('target_day').correct.mean()
    d=(da-db).to_numpy(float);rng=np.random.default_rng(20260924)
    boot=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(B)])
    return {'comparison':f'{a}-{b}','delta':d.mean(),'ci_low':np.quantile(boot,.025),'ci_high':np.quantile(boot,.975),
            'W':int((d>0).sum()),'T':int((d==0).sum()),'L':int((d<0).sum())}

def main():
    store=SequenceStore.load(); selector,_=load_frozen_selector(default_selector_path())
    rows=[]; audits=[]
    for rp in sorted(Q2.glob('E2E1-Q2-*/RUN_RECORD.json')):
        r=json.loads(rp.read_text(encoding='utf-8')); day=r['target_day']
        pred=pd.read_parquet(Path(r['run_dir'])/'predictions.parquet').sort_values('hour_business')
        p=pred.p_positive.to_numpy(float); y=(pred.y_true_model.to_numpy(float)>0).astype(int)
        eligible=np.asarray(store.eligibility(selector,current_target_day=day)['eligible_indices'],dtype=int)
        _,mon,_=stage_a_split_indices(eligible)
        ym=(source_to_model_target(np.asarray(store.y_source[mon],dtype=np.float32))>0).astype(int)
        i1,t1=fit_transitions(ym,False); i2,t2=fit_transitions(ym,True)
        z0=(p>=.5).astype(int); z1=viterbi(p,i1,t1); z2=viterbi(p,i2,t2)
        audits.append({'target_day':day,'monitor_days':len(mon),'truth_flips':int(np.sum(y[1:]!=y[:-1])),
                       'S0_flips':int(np.sum(z0[1:]!=z0[:-1])),'S1_flips':int(np.sum(z1[1:]!=z1[:-1])),
                       'S2_flips':int(np.sum(z2[1:]!=z2[:-1])),
                       'global_p00':t1[0,0,0],'global_p01':t1[0,0,1],'global_p10':t1[0,1,0],'global_p11':t1[0,1,1]})
        for arm,z in [('S0',z0),('S1',z1),('S2',z2)]:
            for h in range(24):
                rows.append({'arm':arm,'target_day':day,'hour':h+1,'y':int(y[h]),'p':float(p[h]),'pred':int(z[h]),'correct':int(z[h]==y[h])})
    df=pd.DataFrame(rows);df.to_csv(OUT/'slot_predictions.csv',index=False)
    pd.DataFrame(audits).to_csv(OUT/'transition_audit.csv',index=False)
    sm=summary(df);sm.to_csv(OUT/'model_summary.csv',index=False)
    ps=pd.DataFrame([paired(df,'S1','S0'),paired(df,'S2','S0'),paired(df,'S2','S1')]);ps.to_csv(OUT/'paired_summary.csv',index=False)
    # windows/hours
    windows={'W1':('2026-02-12','2026-02-18'),'W2':('2026-04-12','2026-04-18'),'W3':('2026-06-12','2026-06-18'),'W4':('2026-08-07','2026-08-13')}
    wr=[];hr=[]
    for arm in ['S0','S1','S2']:
      for w,(lo,hi) in windows.items():
        z=df[(df.arm==arm)&df.target_day.between(lo,hi)]
        m=canonical_metrics(z.y,z.pred,np.ones(len(z)),direction_probability=z.p)
        wr.append({'arm':arm,'window':w,**{k:m[k] for k in ['raw_direction_accuracy','balanced_accuracy','positive_recall','nonpositive_recall']}})
      for s,lo,hi in [('H1',1,8),('H2',9,16),('H3',17,24)]:
        z=df[(df.arm==arm)&df.hour.between(lo,hi)]
        m=canonical_metrics(z.y,z.pred,np.ones(len(z)),direction_probability=z.p)
        hr.append({'arm':arm,'segment':s,**{k:m[k] for k in ['raw_direction_accuracy','balanced_accuracy','positive_recall','nonpositive_recall']}})
    pd.DataFrame(wr).to_csv(OUT/'window_metrics.csv',index=False);pd.DataFrame(hr).to_csv(OUT/'hour_metrics.csv',index=False)
    print(sm.to_string(index=False));print('\n',ps.to_string(index=False));print('\nwindows\n',pd.DataFrame(wr).to_string(index=False))
if __name__=='__main__':main()
