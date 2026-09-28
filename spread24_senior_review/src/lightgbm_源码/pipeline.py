from pathlib import Path
import yaml, pandas as pd
from .data import load_frozen
from .features import feature_columns,matrix
from .model import fit_predict
from .leakage import training_last_day,audit
from .evaluate import summarize

def run(root, config_path=None, smoke=False):
    root=Path(root); cfg=yaml.safe_load((Path(config_path) if config_path else root/"config.yaml").read_text(encoding="utf-8"))
    df=load_frozen(root); cols=feature_columns(root,df); x=matrix(df,cols)
    start=pd.Timestamp(cfg["eval_start"]).date(); end=pd.Timestamp(cfg["eval_end"]).date()
    days=sorted(d for d in df.target_day.unique() if start<=d<=end)
    if smoke: days=days[:2]
    rows=[]
    for d in days:
        train_last=training_last_day(d); train_first=train_last-pd.Timedelta(days=cfg["training_days"]-1)
        for period in cfg["periods"]:
            te=(df.target_day==d)&(df.period==period); tr=(df.target_day>=train_first)&(df.target_day<=train_last)&(df.period==period)
            if not te.any() or tr.sum()<20: continue
            tx=x.loc[tr].fillna(x.loc[tr].median()).fillna(0); vx=x.loc[te].reindex(columns=tx.columns).fillna(tx.median()).fillna(0)
            pred=fit_predict(tx,df.loc[tr,"target_spread"],vx,cfg)
            q=df.loc[te,["target_day","period","hour_business","target_spread"]].copy(); q["predicted_spread"]=pred; q["month"]=q.target_day.map(lambda z:str(z)[:7]); q["training_last_day"]=str(train_last); rows.append(q)
    out=pd.concat(rows,ignore_index=True); od=root/"outputs"; od.mkdir(exist_ok=True)
    out.to_csv(od/"predictions.csv",index=False); summarize(out).to_csv(od/"metrics.csv",index=False)
    pd.DataFrame(audit(days,14,cols)).to_csv(od/"leakage_audit.csv",index=False)
    print(summarize(out).to_string(index=False)); return out


