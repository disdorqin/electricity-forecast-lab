"""Descriptive reports only; these statistics never feed training or thresholds."""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import formal_output_dir
from .dataset import SequenceStore
from .target_adapter import source_to_model_target


def transition_report(end: str="2025-12-31",*,store:SequenceStore|None=None,output_dir:Path|None=None):
    store=store or SequenceStore.load(); cutoff=pd.Timestamp(end).date()
    selected=np.asarray([d<=cutoff for d in store.days])
    ys=source_to_model_target(np.asarray(store.y_source[selected],dtype=np.float64))
    days=store.days[selected]; directions=ys>0; abs_y=np.abs(ys)
    names=list(store.feature_names); x=np.asarray(store.x_future[selected],dtype=np.float32)
    pairs=[]; run_lengths=[]
    for day_i,day in enumerate(days):
        d=directions[day_i]
        for h in range(23): pairs.append({"day":day,"hour_pair":f"{h+1}-{h+2}","flip":bool(d[h]!=d[h+1]),"month":day.month})
        edges=np.diff(np.r_[0,np.flatnonzero(d[1:]!=d[:-1])+1,24])
        run_lengths.extend(int(n) for n in edges)
    pair_df=pd.DataFrame(pairs)
    q=np.quantile(abs_y,[0,.25,.5,.75,1]).tolist()
    magnitude_bins=np.digitize(abs_y,np.unique(q)[1:-1],right=True)
    pair_flat=np.abs(abs_y[:,1:]-abs_y[:,:-1])
    ramp_features=[n for n in ("fcast_直调负荷","fcast_风电总加","fcast_光伏总加","fcast_新能源总加") if n in names]
    ramp_values=[]
    for name in ramp_features:
        idx=names.index(name); values=x[:,:,idx]
        ramp_values.extend(np.abs(np.diff(values,axis=1)).reshape(-1).tolist())
    per_cell_ramp={}
    if ramp_features:
        for name in ramp_features:
            idx=names.index(name); ramps=np.abs(np.diff(x[:,:,idx],axis=1)); cutoff_ramp=np.nanquantile(ramps,.75)
            flip=np.diff(directions.astype(np.int8),axis=1)!=0
            per_cell_ramp[name]={"q75_threshold_from_report_window":float(cutoff_ramp),
                "low_ramp_flip_rate":float(flip[ramps<=cutoff_ramp].mean()) if np.any(ramps<=cutoff_ramp) else None,
                "high_ramp_flip_rate":float(flip[ramps>cutoff_ramp].mean()) if np.any(ramps>cutoff_ramp) else None,
                "source":"target-day forecast feature only; descriptive, not model input/threshold"}
    report={"schema":"spread24_direction_transition_report_v1","status":"DESCRIPTIVE_ONLY",
        "end_date":end,"target_days":len(days),"hour_pair_observations":len(pair_df),
        "overall_flip_rate":float(pair_df.flip.mean()),
        "hour_pair_flip_rate":{str(k):float(v) for k,v in pair_df.groupby("hour_pair").flip.mean().items()},
        "run_length_distribution":{str(k):int(v) for k,v in pd.Series(run_lengths).value_counts().sort_index().items()},
        "monthly_flip_rate":{str(k):float(v) for k,v in pair_df.groupby("month").flip.mean().items()},
        "abs_y_quantile_edges":q,
        "flip_by_abs_y_bin":{str(b):float((np.diff(directions.astype(np.int8),axis=1)!=0)[magnitude_bins[:,1:]==b].mean())
            for b in np.unique(magnitude_bins[:,1:])},
        "ramp_conditioned_flip":per_cell_ramp,"ramp_features":ramp_features,
        "smoothness_loss_added":False,"lockbox_touched":False}
    out=Path(output_dir) if output_dir else formal_output_dir()/"transition"/f"transition_through_{end}.json"
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return report,out
