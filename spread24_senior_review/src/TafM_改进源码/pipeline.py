"""V2.1 bounded integration orchestration; no historical output writers are called."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import json
import time
from datetime import datetime,timezone

import numpy as np
import pandas as pd
import xgboost as xgb

from .config import V2Config,formal_output_dir
from .dataset import SequenceStore
from .evaluate import (evaluate_range,run_ablation,run_strict_lightgbm_baseline,
    summarize_regression_prediction_frame)
from .metrics import canonical_metrics
from .selector import load_selector_manifest
from .target_adapter import source_to_model_target
from .train import train_target_day
from .models.legacy_v20_ablation import run_legacy_v20_target_day


def run_gate_a(start="2026-06-01",end="2026-06-07",*,profile="smoke",config=None):
    """Paired V2.0 legacy vs V2.1 KPI target/head/loss/member alignment."""
    v21=evaluate_range(start,end,mode="A2",profile=profile,train_mode="stage_a",config=config)
    store=SequenceStore.load(); days=[pd.Timestamp(start).date()+timedelta(days=i) for i in range((pd.Timestamp(end)-pd.Timestamp(start)).days+1)]
    selector_path=None; legacy_frames=[]; legacy_dirs=[]
    for day in days:
        result=run_legacy_v20_target_day(day,mode="A2",profile=profile,config=config,store=store,selector_path=selector_path)
        legacy_frames.append(result["predictions"]); legacy_dirs.append(result["run_dir"])
    legacy=pd.concat(legacy_frames,ignore_index=True)
    from .metrics import canonical_metrics
    legacy_metrics=canonical_metrics(legacy.y_true_model,legacy.direction_hat.astype(bool),legacy.magnitude_hat,
        direction_probability=legacy.p_positive,month=legacy.target_day.map(lambda d:pd.Timestamp(d).month),hour=legacy.hour_business)
    rows=[{"model":"V2.0_legacy",**legacy_metrics},
          {"model":"V2.1_A2_stage_a",**v21["metrics"]}]
    # Persist paired report without overwriting the standalone V2.0 or prior V2.1 artifacts.
    out=formal_output_dir()/"ablations"/f"gate_a_v20_vs_v21_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(rows).to_csv(out/"comparison.csv",index=False)
    legacy.to_parquet(out/"v20_legacy_predictions.parquet",index=False)
    (out/"manifest.json").write_text(json.dumps({"range":[start,end],"profile":profile,
        "train_mode":"stage_a","v20_run_dirs":legacy_dirs,"v21_run_dir":v21["output_dir"],
        "same_selector":"V2.1 frozen cutoff 2025-12-31 for isolation","status":"COMPLETE"},indent=2)+"\n",encoding="utf-8")
    return {"output_dir":str(out),"comparison":pd.DataFrame(rows),"v21":v21,"legacy_predictions":legacy}


def run_ablate_core(start="2026-06-01",end="2026-06-07",*,modes=("A0","A1","A2"),profile="smoke",config=None):
    gate_a=run_gate_a(start,end,profile=profile,config=config)
    # Gate B isolates A0/A1/A2 adapter semantics; Stage B belongs only to Gate C.
    gate_b=run_ablation(start,end,modes=tuple(modes),profile=profile,train_mode="stage_a",config=config)
    return {"gate_a":gate_a,"gate_b":gate_b}


def run_train_strategy(start="2026-06-01",end="2026-06-07",*,modes=("stage_a","stage_ab","full_retrain"),profile="smoke",config=None):
    results={}
    for mode in modes:
        if mode not in {"stage_a","stage_ab","full_retrain"}: raise ValueError(f"invalid training strategy {mode}")
        results[mode]=evaluate_range(start,end,mode="A2",profile=profile,train_mode=mode,config=config)
    out=formal_output_dir()/"ablations"/f"train_strategy_comparison_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True,exist_ok=False)
    summary=pd.DataFrame([{"train_mode":m,**r["metrics"],"output_dir":r["output_dir"]} for m,r in results.items()])
    summary.to_csv(out/"comparison.csv",index=False)
    (out/"manifest.json").write_text(json.dumps({"range":[start,end],"modes":list(modes),"profile":profile,"stage_b_production_default":False,"status":"COMPLETE"},indent=2)+"\n",encoding="utf-8")
    return {"output_dir":str(out),"comparison":summary,"runs":results}


def run_direction_benchmark(start="2026-06-01",end="2026-06-07",*,profile="smoke",config=None):
    store=SequenceStore.load(); selector,selector_hash=load_selector_manifest(); _,cols,_=store.selector_indices(selector)
    days=[pd.Timestamp(start).date()+timedelta(days=i) for i in range((pd.Timestamp(end)-pd.Timestamp(start)).days+1)]
    tabm_frames=[]; xgb_frames=[]
    cfg=config or V2Config()
    for day in days:
        sample=np.flatnonzero(np.asarray([d==day for d in store.days]))
        if len(sample)!=1: raise ValueError(f"day absent: {day}")
        result=train_target_day(day,mode="A2",profile=profile,train_mode="stage_a",config=cfg,store=store)
        frame=result["predictions"][["target_day","hour_business","y_true_model","p_positive","direction_hat"]].copy()
        frame["model"]="TabM+Temporal-DIR"; tabm_frames.append(frame)
        eligible=store.eligibility(selector,current_target_day=day)["eligible_indices"]
        eligible=[i for i in eligible if store.days[i]>=day-timedelta(days=180)]
        if len(eligible)<30: raise ValueError("XGB direction benchmark insufficient legal history")
        xt=np.asarray(store.x_future[eligible][:,:,cols],dtype=np.float32).reshape(-1,len(cols))
        yt=source_to_model_target(np.asarray(store.y_source[eligible],dtype=np.float32)).reshape(-1)
        xq=np.asarray(store.x_future[sample][:,:,cols],dtype=np.float32).reshape(-1,len(cols))
        model=xgb.XGBClassifier(n_estimators=90,max_depth=4,learning_rate=.05,subsample=1.0,colsample_bytree=.8,
            reg_lambda=1,objective="binary:logistic",tree_method="hist",n_jobs=1,random_state=cfg.seed,eval_metric="logloss")
        model.fit(xt,(yt>0).astype(np.int8)); prob=model.predict_proba(xq)[:,1]
        xgb_frames.append(pd.DataFrame({"target_day":str(day),"hour_business":np.arange(1,25),
            "y_true_model":source_to_model_target(np.asarray(store.y_source[int(sample[0])],dtype=np.float32)),
            "p_positive":prob,"direction_hat":prob>=.5,"model":"XGBoost-DIR"}))
    lgb=run_strict_lightgbm_baseline(start,end)
    lgb["direction_hat"]=lgb.prediction_model>0; lgb["model"]="LightGBM-DIR"
    all_frames=[pd.concat(tabm_frames,ignore_index=True),pd.concat(xgb_frames,ignore_index=True),lgb]
    rows=[]
    for f in all_frames:
        if str(f.model.iloc[0]) == "LightGBM-DIR":
            m=summarize_regression_prediction_frame(f)
            rows.append({"model":str(f.model.iloc[0]),**{k:m[k] for k in ("raw_direction_accuracy","balanced_accuracy","positive_recall","nonpositive_recall")},
                         "auc":m["rank_auc"],"rank_auc":m["rank_auc"],"brier":"N/A"})
        else:
            m=canonical_metrics(f.y_true_model,f.direction_hat,np.zeros(len(f)),direction_probability=f.p_positive,
                month=f.target_day.map(lambda d:pd.Timestamp(d).month),hour=f.hour_business)
            rows.append({"model":str(f.model.iloc[0]),**{k:m[k] for k in ("raw_direction_accuracy","balanced_accuracy","positive_recall","nonpositive_recall","auc","brier")}})
    out=formal_output_dir()/"direction_benchmark"/f"benchmark_{start}_{end}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True,exist_ok=True)
    table=pd.DataFrame(rows); table.to_csv(out/"metrics.csv",index=False)
    for f in all_frames: f.to_parquet(out/(str(f.model.iloc[0]).replace("+","_")+"_predictions.parquet"),index=False)
    (out/"manifest.json").write_text(json.dumps({"range":[start,end],"selector_sha256":selector_hash,
        "models":["LightGBM-DIR","XGBoost-DIR","TabM+Temporal-DIR"],"label_cutoff":"D-2","claim":"benchmark only, not theoretical ceiling"},indent=2)+"\n",encoding="utf-8")
    return {"output_dir":str(out),"metrics":table}
