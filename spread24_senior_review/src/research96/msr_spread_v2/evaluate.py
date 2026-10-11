"""Engineering/retrospective metrics; smoke never establishes scientific gain."""
import numpy as np
from sklearn.metrics import roc_auc_score
from .config import Q_KNOTS

def crps_np(s,y):
    ordered = np.sort(s,axis=0)
    m = s.shape[0]
    weights = (2*np.arange(1,m+1)-m-1)[:,None]
    return float(np.mean(np.mean(np.abs(s-y),axis=0)-(weights*ordered).sum(0)/(m*m)))

def marginal_score_np(y,q):
    lv = np.array(Q_KNOTS)
    edges = np.r_[0,(lv[:-1]+lv[1:])/2,1]
    err = y[:,None]-q
    return float(np.mean(2*(np.maximum(lv*err,(lv-1)*err)*np.diff(edges)).sum(-1)))

def _correlation(a,b):
    return float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None

def evaluate_day(out,y,da):
    s = np.asarray(out["spread_scenarios"])[0]
    q = np.asarray(out["spread_quantiles"])[0]
    point,p = np.asarray(out["spread_median"])[0],np.asarray(out["p_positive"])[0]
    truth = y[:,0]
    positive = truth>0
    pred = p>.5
    pos_recall = float(pred[positive].mean()) if positive.any() else None
    neg_recall = float((~pred[~positive]).mean()) if (~positive).any() else None
    recalls = [v for v in (pos_recall,neg_recall) if v is not None]
    error = point-truth
    metrics = dict(spread_mae=float(np.abs(error).mean()),spread_rmse=float(np.sqrt(np.mean(error**2))),
        rt_mae=float(np.mean(np.abs(np.asarray(out["rt_median"])[0]-(da+truth)))),
        raw_direction=float(np.mean(pred==positive)),balanced_accuracy=float(np.mean(recalls)),
        positive_recall=pos_recall,negative_recall=neg_recall,
        auc=float(roc_auc_score(positive,p)) if positive.any() and (~positive).any() else None,
        brier0=float(np.mean((p-positive)**2)),predicted_positive_fraction=float(pred.mean()),
        truth_positive_fraction=float(positive.mean()),point_positive_fraction=float((point>0).mean()),
        all_negative_baseline=float((~positive).mean()),
        majority_baseline=float(max(positive.mean(),1-positive.mean())),
        crps=crps_np(s,truth),
        quantile_coverage={f"q{int(round(lv*100)):02d}":float(np.mean(truth<=q[:,i])) for i,lv in enumerate(Q_KNOTS)},
        width_q01_q99=float(np.mean(q[:,-1]-q[:,0])),
        width_q10_q90=float(np.mean(q[:,6]-q[:,2])),
        coverage_q10_q90=float(np.mean((truth>=q[:,2])&(truth<=q[:,6]))),
        clamp_fraction=float(np.asarray(out["clamp_fraction"])))
    for name,i in (("low",1),("high",2)):
        knots = np.asarray(out[name+"_q_raw"])[0]
        scenarios = np.asarray(out[name+"_scenarios"])[0]
        metrics[name] = dict(mae=float(np.mean(np.abs(knots[:,4]-y[:,i]))),
            crps_knots=marginal_score_np(y[:,i],knots),crps_scenarios=crps_np(scenarios,y[:,i]),
            correlation=_correlation(knots[:,4],y[:,i]))
    metrics["dependence"] = {k:dict(min=float(np.min(np.asarray(out["dependence"][k]))),
        max=float(np.max(np.asarray(out["dependence"][k]))),
        squared_diff=float(np.mean(np.diff(np.asarray(out["dependence"][k]),axis=-1)**2)))
        for k in ("rho","phiL","phiH")}
    edges = np.linspace(0,1,11)
    metrics["calibration_bins"] = []
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask = (p>=lo)&((p<=hi) if hi==1 else (p<hi))
        metrics["calibration_bins"].append(dict(lo=float(lo),hi=float(hi),count=int(mask.sum()),
            predicted=float(p[mask].mean()) if mask.any() else None,
            observed=float(positive[mask].mean()) if mask.any() else None))
    slots = [dict(slot=t+1,spread_abs_error=float(abs(error[t])),direction=int(pred[t]==positive[t]),
                  p_positive=float(p[t]),truth_positive=int(positive[t])) for t in range(96)]
    return metrics,slots
