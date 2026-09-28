from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.TafM_改进源码.config import load_v21_config, resolve_device
from src.TafM_改进源码.dataset import SequenceStore, load_frozen_selector
from src.TafM_改进源码.preprocessing import PreprocessorState, transform_future, transform_hist
from src.TafM_改进源码.train import _model_for, stage_a_split_indices
from src.TafM_改进源码.direction_postprocess import fit_logistic_stacker
from src.TafM_改进源码.target_adapter import source_to_model_target
from src.TafM_改进源码.metrics import canonical_metrics

E2 = ROOT / "experiments/first_test/E2_architecture/horizon_specialized_head/runs"
OUT = ROOT / "experiments/first_test/E4_signal/regime_postprocess"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "runs").mkdir(exist_ok=True)

WINDOWS = {
    "W1": ("2026-02-12", "2026-02-18"),
    "W2": ("2026-04-12", "2026-04-18"),
    "W3": ("2026-06-12", "2026-06-18"),
    "W4": ("2026-08-07", "2026-08-13"),
}
ARMS = ["P0", "P1", "P2"]


def predict_batched(model, xh, xf, device, batch=32):
    ps = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(xh), batch):
            out = model(torch.from_numpy(xh[i:i+batch]).to(device),
                        torch.from_numpy(xf[i:i+batch]).to(device))
            ps.append(out["p"].detach().cpu().numpy())
    return np.concatenate(ps, axis=0)


def q2_records():
    records = []
    for p in sorted(E2.glob("E2E1-Q2-*/RUN_RECORD.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("status") == "PASS":
            records.append(r)
    if len(records) != 28:
        raise RuntimeError(f"expected 28 Q2 PASS records, found {len(records)}")
    return records


def binary_summary(y, p):
    d = p >= .5
    mag = np.ones_like(y, dtype=float)
    m = canonical_metrics(y, d, mag, direction_probability=p)
    return {k:m.get(k) for k in ["raw_direction_accuracy","balanced_accuracy","positive_recall",
                                 "nonpositive_recall","auc","brier"]}


def aggregate(df):
    rows = []
    for arm in ARMS:
        z = df[df.arm == arm]
        m = binary_summary(z.y.to_numpy(), z.p.to_numpy())
        m.update({
            "arm": arm,
            "slots": len(z),
            "ppf": float((z.p.to_numpy() >= .5).mean()),
            "one_class_days": int(z.groupby("target_day").pred.nunique().eq(1).sum()),
            "positive_recall_zero_days": int(sum(
                ((g.y>0).any() and ((g.pred[g.y>0] == 1).mean() == 0))
                for _,g in z.groupby("target_day")
            )),
            "nonpositive_recall_zero_days": int(sum(
                ((g.y<=0).any() and ((g.pred[g.y<=0] == 0).mean() == 0))
                for _,g in z.groupby("target_day")
            )),
        })
        rows.append(m)
    return pd.DataFrame(rows)


def window_table(df):
    rows=[]
    for arm in ARMS:
        for w,(a,b) in WINDOWS.items():
            z=df[(df.arm==arm)&(df.target_day>=a)&(df.target_day<=b)]
            m=binary_summary(z.y.to_numpy(), z.p.to_numpy())
            rows.append({"arm":arm,"window":w,**m})
    return pd.DataFrame(rows)


def hour_table(df):
    rows=[]
    for arm in ARMS:
        for seg,lo,hi in [("H1",1,8),("H2",9,16),("H3",17,24)]:
            z=df[(df.arm==arm)&df.hour.between(lo,hi)]
            m=binary_summary(z.y.to_numpy(),z.p.to_numpy())
            rows.append({"arm":arm,"segment":seg,**m})
    return pd.DataFrame(rows)


def paired_days(df, a, b, seed=20260924, B=10000):
    da=df[df.arm==a].groupby("target_day").correct.mean()
    db=df[df.arm==b].groupby("target_day").correct.mean()
    days=sorted(set(da.index)&set(db.index))
    delta=np.array([da[d]-db[d] for d in days],float)
    rng=np.random.default_rng(seed)
    boot=np.array([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(B)])
    return {
        "comparison":f"{a}_minus_{b}",
        "delta_raw":float(delta.mean()),
        "ci_low":float(np.quantile(boot,.025)),
        "ci_high":float(np.quantile(boot,.975)),
        "wins":int((delta>0).sum()),
        "ties":int((delta==0).sum()),
        "losses":int((delta<0).sum()),
    }


def main():
    store=SequenceStore.load()
    cfg=load_v21_config().with_profile("default")
    device=resolve_device(cfg)
    selector_path=Path(cfg.config_path).parent / "outputs/tabm_v21/selector/selector_cutoff_2025-12-31/manifest.json"
    if not selector_path.exists():
        from src.TafM_改进源码.config import default_selector_path
        selector_path=default_selector_path()
    selector,_=load_frozen_selector(selector_path)
    day_to_idx={str(d):i for i,d in enumerate(store.days)}

    all_rows=[]
    audits=[]
    for rec in q2_records():
        day=str(rec["target_day"])
        run=Path(rec["run_dir"])
        manifest=json.loads((run/"manifest.json").read_text(encoding="utf-8"))
        state=PreprocessorState.load(run/"preprocessor_state.json")
        eligible=np.asarray(store.eligibility(selector,current_target_day=day)["eligible_indices"],dtype=np.int64)
        base_idx, mon_idx, _split_mode = stage_a_split_indices(eligible)
        expected_base=int(manifest["stage_a_base_train_days"])
        expected_mon=int(manifest["stage_a_monitor_days"])
        if len(base_idx)!=expected_base or len(mon_idx)!=expected_mon:
            raise RuntimeError(f"{day} split mismatch {len(base_idx)}/{len(mon_idx)} != {expected_base}/{expected_mon}")

        model=_model_for(store,selector,state,cfg,
            legacy_v20=False, architecture_mode="full_current", direction_fusion_alpha=.8,
            direction_tabular_mode="current", direction_horizon_gate_mode="current",
            strong_role_profile="all", numeric_encoding_mode="canonical",
            direction_readout_mode="segment_heads").to(device)
        ck=torch.load(run/"stage_a_best.pt",map_location=device,weights_only=False)
        model.load_state_dict(ck["model"])

        # monitor
        mon_xh=transform_hist(np.asarray(store.x_hist[mon_idx],dtype=np.float32),state)
        mon_xf=transform_future(np.asarray(store.x_future[mon_idx][:,:,state.feature_indices],dtype=np.float32),state)
        mon_y=source_to_model_target(np.asarray(store.y_source[mon_idx],dtype=np.float32))
        mon_p=predict_batched(model,mon_xh,mon_xf,device)

        # target exact reproduction
        ti=day_to_idx[day]
        txh=transform_hist(np.asarray(store.x_hist[ti:ti+1],dtype=np.float32),state)
        txf=transform_future(np.asarray(store.x_future[ti:ti+1][:,:,state.feature_indices],dtype=np.float32),state)
        base_p=predict_batched(model,txh,txf,device).reshape(-1)
        saved=pd.read_parquet(run/"predictions.parquet").sort_values("hour_business").reset_index(drop=True)
        delta=float(np.max(np.abs(base_p-saved.p_positive.to_numpy())))
        if delta>1e-9:
            raise RuntimeError(f"{day} Q2 reproduction failed: {delta}")
        y=saved.y_true_model.to_numpy(float)
        mag=saved.magnitude_hat.to_numpy(float)

        stack1=fit_logistic_stacker(mon_p,mon_xf,mon_y,state.feature_names,"segment_logit")
        stack2=fit_logistic_stacker(mon_p,mon_xf,mon_y,state.feature_names,"regime_logit")
        p1=stack1.predict_proba(base_p.reshape(1,24),txf,state.feature_names).reshape(-1)
        p2=stack2.predict_proba(base_p.reshape(1,24),txf,state.feature_names).reshape(-1)

        for arm,p in [("P0",base_p),("P1",p1),("P2",p2)]:
            pred=(p>=.5).astype(int)
            for h in range(24):
                all_rows.append({"arm":arm,"target_day":day,"hour":h+1,"y":y[h],
                                 "p":float(p[h]),"pred":int(pred[h]),"correct":int(pred[h]==(y[h]>0))})
        audits.append({
            "target_day":day,
            "q2_reproduction_max_abs_delta":delta,
            "monitor_days":len(mon_idx),
            "monitor_start":str(store.days[mon_idx[0]]),
            "monitor_end":str(store.days[mon_idx[-1]]),
            "P1_monitor_base_raw":stack1.monitor_base_raw,
            "P1_monitor_post_raw":stack1.monitor_post_raw,
            "P2_monitor_base_raw":stack2.monitor_base_raw,
            "P2_monitor_post_raw":stack2.monitor_post_raw,
            "P1_target_ppf":float((p1>=.5).mean()),
            "P2_target_ppf":float((p2>=.5).mean()),
            "P1_weights":json.dumps(stack1.audit(),ensure_ascii=False),
            "P2_weights":json.dumps(stack2.audit(),ensure_ascii=False),
        })
        print(day,"PASS","base",round(float(((base_p>=.5)==(y>0)).mean()),4),
              "P1",round(float(((p1>=.5)==(y>0)).mean()),4),
              "P2",round(float(((p2>=.5)==(y>0)).mean()),4))

    df=pd.DataFrame(all_rows)
    df.to_csv(OUT/"daily_slot_predictions.csv",index=False)
    pd.DataFrame(audits).to_csv(OUT/"postprocess_audit.csv",index=False)
    overall=aggregate(df); overall.to_csv(OUT/"model_summary.csv",index=False)
    wt=window_table(df); wt.to_csv(OUT/"window_metrics.csv",index=False)
    ht=hour_table(df); ht.to_csv(OUT/"hour_segment_metrics.csv",index=False)
    paired=pd.DataFrame([paired_days(df,"P1","P0"),paired_days(df,"P2","P0"),paired_days(df,"P2","P1")])
    paired.to_csv(OUT/"paired_summary.csv",index=False)

    # slot gains by segment
    gains=[]
    for arm in ["P1","P2"]:
        for seg,lo,hi in [("ALL",1,24),("H1",1,8),("H2",9,16),("H3",17,24)]:
            a=df[(df.arm==arm)&df.hour.between(lo,hi)].correct.sum()
            b=df[(df.arm=="P0")&df.hour.between(lo,hi)].correct.sum()
            gains.append({"arm":arm,"segment":seg,"slot_gain_vs_P0":int(a-b)})
    pd.DataFrame(gains).to_csv(OUT/"slot_gain_summary.csv",index=False)

    # safety + gate
    o=overall.set_index("arm")
    safety={}
    for arm in ARMS:
        safety[arm]=bool(
            (o.loc[arm,"balanced_accuracy"] >= o.loc["P0","balanced_accuracy"]-.01) and
            (o.loc[arm,"positive_recall"] >= o.loc["P0","positive_recall"]-.05) and
            (o.loc[arm,"one_class_days"] <= o.loc["P0","one_class_days"]+2) and
            (o.loc[arm,"positive_recall_zero_days"] <= o.loc["P0","positive_recall_zero_days"]+2)
        )
    p2=float(o.loc["P2","raw_direction_accuracy"]); p1=float(o.loc["P1","raw_direction_accuracy"]); p0=float(o.loc["P0","raw_direction_accuracy"])
    pair2=paired.set_index("comparison").loc["P2_minus_P0"]
    if safety["P2"] and p2>=.62:
        gate="PROMISING_POSTPROCESS_UNPROVEN" if pair2.ci_low<=0 else "REGIME_STACKING_CONFIRMED"
    elif safety["P2"] and pair2.ci_low>0 and p2>p0:
        gate="REGIME_STACKING_CONFIRMED"
    elif safety["P1"] and paired.set_index("comparison").loc["P1_minus_P0"].ci_low>0 and p1>p0:
        gate="SEGMENT_CALIBRATION_CONFIRMED"
    elif (p1<p0-.01 and p2<p0-.01) or (not safety["P1"] and not safety["P2"]):
        gate="POSTPROCESS_HARMFUL"
    elif max(p1,p2)>p0 and (safety["P1"] or safety["P2"]):
        gate="PROMISING_POSTPROCESS_UNPROVEN"
    else:
        gate="NO_POSTPROCESS_GAIN"

    summary={
        "status":"COMPLETE",
        "q2_reproduction_max_abs_delta":float(pd.DataFrame(audits).q2_reproduction_max_abs_delta.max()),
        "overall":overall.to_dict(orient="records"),
        "paired":paired.to_dict(orient="records"),
        "safety":safety,
        "POSTPROCESS_SIGNAL":gate,
    }
    (OUT/"E4_A_GATE.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("\nOVERALL\n",overall.to_string(index=False))
    print("\nPAIRED\n",paired.to_string(index=False))
    print("\nWINDOWS\n",wt.to_string(index=False))
    print("\nHOURS\n",ht.to_string(index=False))
    print("\nSAFETY",safety,"GATE",gate)


if __name__=="__main__":
    main()
