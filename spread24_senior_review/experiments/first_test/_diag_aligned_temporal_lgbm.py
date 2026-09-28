import sys, numpy as np, pandas as pd
sys.path.insert(0,'src')
from TafM_改进源码.dataset import SequenceStore, load_frozen_selector
from TafM_改进源码.config import default_selector_path
from lightgbm import LGBMClassifier

store=SequenceStore.load()
sel,_=load_frozen_selector(default_selector_path())
names=list(store.feature_names)
core=['fcast_直调负荷','fcast_竞价空间','fcast_新能源总加','fcast_风电总加','fcast_光伏总加','residual_load_renew','bidding_space_ratio','renewable_share','net_ramp_pressure','ramp_tightness','ctx_spread_positive_rate14','spread_same_slot_28d_positive_rate']
core=[f for f in core if f in names]
core_idx=[names.index(f) for f in core]

def aligned_for_day(i):
    xh=np.asarray(store.x_hist[i],float).copy()
    xh[:,0]*=-1
    xf=np.asarray(store.x_future[i][:,core_idx],float)
    d=store.days[i]
    origin=pd.Timestamp(d)-pd.Timedelta(days=1)+pd.Timedelta(hours=14)
    ts=pd.date_range(origin-pd.Timedelta(hours=167),origin,freq='h')
    pos={t:j for j,t in enumerate(ts)}
    rows=[]
    recent=xh[-14:,0]
    recent_feats=[np.mean(recent),np.std(recent),np.median(recent),recent[-1],np.mean(recent>0),np.polyfit(np.arange(14),recent,1)[0]]
    for h in range(1,25):
        target=pd.Timestamp(d)+pd.Timedelta(hours=h)
        vals=[]
        for ld in [1,2,3,7]:
            j=pos.get(target-pd.Timedelta(days=ld))
            vals.append(xh[j,0] if j is not None else 0.0)
            if ld==1:
                vals.append(1.0 if j is not None else 0.0)
        same=[]
        for ld in range(1,8):
            j=pos.get(target-pd.Timedelta(days=ld))
            if j is not None:
                same.append(xh[j,0])
        vals += [np.mean(same),np.std(same),np.median(same),np.mean(np.asarray(same)>0),same[-1] if same else 0.0]
        for c in range(7):
            for ld in [1,2,7]:
                j=pos.get(target-pd.Timedelta(days=ld))
                if j is None and ld==1:
                    j=pos.get(target-pd.Timedelta(days=2))
                vals.append(xh[j,c] if j is not None else 0.0)
        vals += recent_feats
        vals += list(xf[h-1])
        vals += [np.sin(2*np.pi*h/24),np.cos(2*np.pi*h/24)]
        rows.append(vals)
    return np.asarray(rows,np.float32)

Xcache=[aligned_for_day(i) for i in range(len(store.days))]
ranges={'W1':('2026-02-12','2026-02-18'),'W2':('2026-04-12','2026-04-18'),'W3':('2026-06-12','2026-06-18'),'W4':('2026-08-07','2026-08-13')}
days=[]
for w,(a,b) in ranges.items():
    days += [(w,d) for d in pd.date_range(a,b).date]
idxmap={d:i for i,d in enumerate(store.days)}
out=[]
for w,d in days:
    ti=idxmap[d]
    elig=np.asarray(store.eligibility(sel,current_target_day=d)['eligible_indices'],int)
    Xtr=np.concatenate([Xcache[i] for i in elig],axis=0)
    ytr=np.concatenate([(-np.asarray(store.y_source[i],float)>0).astype(int) for i in elig])
    model=LGBMClassifier(n_estimators=250,learning_rate=.03,num_leaves=15,min_child_samples=80,colsample_bytree=.8,reg_lambda=1.0,verbosity=-1,random_state=20260924,n_jobs=-1)
    model.fit(Xtr,ytr)
    p=model.predict_proba(Xcache[ti])[:,1]
    pred=p>=.5
    y=(-np.asarray(store.y_source[ti],float)>0)
    out.append((w,d,y,pred,p))
truth=np.concatenate([x[2] for x in out]); pred=np.concatenate([x[3] for x in out])
raw=(truth==pred).mean(); pos=pred[truth].mean(); neg=(~pred[~truth]).mean(); bal=(pos+neg)/2
print('ALIGNED_LGBM Raw',round(raw,4),'Bal',round(bal,4),'+R',round(pos,4),'-R',round(neg,4),'ppf',round(pred.mean(),4), flush=True)
for w in ranges:
    rr=[x for x in out if x[0]==w]
    t=np.concatenate([x[2] for x in rr]); p=np.concatenate([x[3] for x in rr])
    print(w,round((t==p).mean(),4), flush=True)
for lo,hi in [(1,8),(9,16),(17,24)]:
    t=np.concatenate([x[2][lo-1:hi] for x in out]); p=np.concatenate([x[3][lo-1:hi] for x in out])
    print('H',lo,hi,round((t==p).mean(),4), flush=True)
    for w in ranges:
        rr=[x for x in out if x[0]==w]
        tw=np.concatenate([x[2][lo-1:hi] for x in rr]); pw=np.concatenate([x[3][lo-1:hi] for x in rr])
        print(' ',w,round((tw==pw).mean(),4),end='')
    print(flush=True)

# persist diagnostic predictions
recrows=[]
for w,d,y,pred,prob in out:
    for h in range(1,25): recrows.append({'window':w,'target_day':str(d),'hour_business':h,'y_true':int(y[h-1]),'aligned_pred':int(pred[h-1]),'aligned_p':float(prob[h-1])})
pd.DataFrame(recrows).to_csv('experiments/first_test/aligned_temporal_lgbm_predictions.csv',index=False)
