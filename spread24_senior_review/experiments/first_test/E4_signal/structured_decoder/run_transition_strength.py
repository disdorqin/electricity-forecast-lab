from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.TafM_改进源码.config import default_selector_path
from src.TafM_改进源码.dataset import SequenceStore, load_frozen_selector
from src.TafM_改进源码.target_adapter import source_to_model_target
from src.TafM_改进源码.train import stage_a_split_indices
from src.TafM_改进源码.metrics import canonical_metrics
from experiments.first_test.E4_signal.structured_decoder.run_structured_decoder import fit_transitions, viterbi

Q2=ROOT/'experiments/first_test/E2_architecture/horizon_specialized_head/runs'
OUT=ROOT/'experiments/first_test/E4_signal/structured_decoder'
BETAS=[0.0,0.10,0.25,0.50,0.75,1.0]

store=SequenceStore.load(); selector,_=load_frozen_selector(default_selector_path())
rows=[]
for rp in sorted(Q2.glob('E2E1-Q2-*/RUN_RECORD.json')):
    r=json.loads(rp.read_text(encoding='utf-8')); day=r['target_day']
    pred=pd.read_parquet(Path(r['run_dir'])/'predictions.parquet').sort_values('hour_business')
    p=pred.p_positive.to_numpy(float); y=(pred.y_true_model.to_numpy(float)>0).astype(int)
    eligible=np.asarray(store.eligibility(selector,current_target_day=day)['eligible_indices'],dtype=int)
    _,mon,_=stage_a_split_indices(eligible)
    ym=(source_to_model_target(np.asarray(store.y_source[mon],dtype=np.float32))>0).astype(int)
    init,trans=fit_transitions(ym,True)
    for beta in BETAS:
        z=viterbi(p,init,trans,beta=beta)
        for h in range(24):
            rows.append({'beta':beta,'target_day':day,'hour':h+1,'y':int(y[h]),'p':float(p[h]),
                         'pred':int(z[h]),'correct':int(z[h]==y[h])})
df=pd.DataFrame(rows);df.to_csv(OUT/'transition_strength_slots.csv',index=False)
out=[]
for beta in BETAS:
    z=df[df.beta==beta]
    m=canonical_metrics(z.y,z.pred,np.ones(len(z)),direction_probability=z.p)
    days=z.groupby('target_day')
    one=int(days.pred.nunique().eq(1).sum())
    zr=int(sum(((g.y==1).any() and (g.loc[g.y==1,'pred']==1).mean()==0) for _,g in days))
    out.append({'beta':beta,'raw':m['raw_direction_accuracy'],'balanced':m['balanced_accuracy'],
                'pos_recall':m['positive_recall'],'neg_recall':m['nonpositive_recall'],
                'ppf':float(z.pred.mean()),'one_class_days':one,'pos_recall_zero_days':zr,
                'correct':int(z.correct.sum())})
res=pd.DataFrame(out);res.to_csv(OUT/'transition_strength_summary.csv',index=False)
anchor=res[res.beta==0].iloc[0]
res['safe']=(res.balanced>=anchor.balanced-.01)&(res.pos_recall>=anchor.pos_recall-.05)&(res.one_class_days<=anchor.one_class_days+2)&(res.pos_recall_zero_days<=anchor.pos_recall_zero_days+2)
print(res.to_string(index=False))
