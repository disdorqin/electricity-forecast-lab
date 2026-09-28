import json
from pathlib import Path
import pandas as pd

def feature_columns(root, df):
    p=Path(root)/"data"/"frozen_repro"/"feature_groups.json"
    groups=json.loads(p.read_text(encoding="utf-8"))
    cols=[]
    for names in groups.values():
        cols.extend([c for c in names if c in df.columns])
    return list(dict.fromkeys(cols))

def matrix(df, cols):
    x=df[cols].apply(pd.to_numeric, errors="coerce")
    return x.replace([float("inf"),-float("inf")], pd.NA)
