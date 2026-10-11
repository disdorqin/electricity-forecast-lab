"""Slot-pooled final/component metrics, not averages of daily recall/AUC."""
import numpy as np
from sklearn.metrics import roc_auc_score
from ...config import Q_KNOTS
from ...evaluate import crps_np, marginal_score_np

def pooled(records):
    if not records: return None
    y=np.concatenate([r['truth'][:,0] for r in records]); positive=y>0
    point=np.concatenate([r['out']['spread_median'][0] for r in records])
    p=np.concatenate([r['out']['p_positive'][0] for r in records]); pred=p>.5
    q=np.concatenate([r['out']['spread_quantiles'][0] for r in records]); error=point-y
    rp=float(pred[positive].mean()) if positive.any() else None
    rn=float((~pred[~positive]).mean()) if (~positive).any() else None
    result=dict(days=len(records),slots=len(y),spread_mae=float(abs(error).mean()),
        spread_rmse=float(np.sqrt((error**2).mean())),rt_mae=float(abs(error).mean()),
        raw_direction=float((pred==positive).mean()),balanced_accuracy=float(np.mean([v for v in (rp,rn) if v is not None])),
        positive_recall=rp,negative_recall=rn,auc=float(roc_auc_score(positive,p)) if positive.any() and (~positive).any() else None,
        brier0=float(((p-positive)**2).mean()),truth_positive_fraction=float(positive.mean()),
        predicted_positive_fraction=float(pred.mean()),mean_predicted_probability=float(p.mean()),
        prevalence_calibration_error=float(p.mean()-positive.mean()),all_negative_baseline=float((~positive).mean()),
        point_positive_fraction=float((point>0).mean()),
        crps=float(np.mean([crps_np(r['out']['spread_scenarios'][0],r['truth'][:,0]) for r in records])),
        coverage_q10_q90=float(((y>=q[:,2])&(y<=q[:,6])).mean()),width_q10_q90=float((q[:,6]-q[:,2]).mean()),
        width_q01_q99=float((q[:,-1]-q[:,0]).mean()),
        quantile_coverage={str(lv):float((y<=q[:,i]).mean()) for i,lv in enumerate(Q_KNOTS)})
    result['calibration_bins']=[]
    for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        mask=(p>=lo)&((p<=hi) if hi==1 else (p<hi))
        result['calibration_bins'].append(dict(lo=float(lo),hi=float(hi),count=int(mask.sum()),
            predicted=float(p[mask].mean()) if mask.any() else None,observed=float(positive[mask].mean()) if mask.any() else None))
    for name,index in (('low',1),('high',2)):
        if name+'_q_raw' not in records[0]['out']: continue
        knots=np.concatenate([r['out'][name+'_q_raw'][0] for r in records]);truth=np.concatenate([r['truth'][:,index] for r in records])
        median=knots[:,4]
        result[name]=dict(mae=float(abs(median-truth).mean()),crps_knots=marginal_score_np(truth,knots),
            correlation=float(np.corrcoef(median,truth)[0,1]) if np.std(median)>0 and np.std(truth)>0 else None,
            sign_accuracy=float(((median>0)==(truth>0)).mean()),
            predicted_zero_crossings=int(sum(np.count_nonzero(np.diff(r['out'][name+'_q_raw'][0,:,4]>0)) for r in records)),
            truth_zero_crossings=int(sum(np.count_nonzero(np.diff(r['truth'][:,index]>0)) for r in records)))
    return result

def monthly(records):
    return {month:pooled([r for r in records if r['date'][:7]==month]) for month in sorted({r['date'][:7] for r in records})}

def promotion(full,other):
    if full is None or other is None: return []
    hits=[]
    for k,threshold in (('balanced_accuracy',.015),('raw_direction',.015),('brier0',.01)):
        if abs(other[k]-full[k])>=threshold: hits.append(k)
    for k in ('spread_mae','crps'):
        if abs(other[k]/full[k]-1)>=.03: hits.append(k)
    return hits
