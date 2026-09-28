import numpy as np
import pandas as pd

def metrics(y, pred):
    y=np.asarray(y,float); pred=np.asarray(pred,float)
    yp=pred>=0; yt=y>=0
    pos=(yt.sum()); neg=(~yt).sum()
    pr=((yp & yt).sum()/pos) if pos else np.nan
    nr=((~yp & ~yt).sum()/neg) if neg else np.nan
    return {"raw":float((yp==yt).mean()),"positive_recall":float(pr),"negative_recall":float(nr),"balanced":float(np.nanmean([pr,nr])),"all_negative_baseline":float((~yt).mean()),"mae":float(np.mean(np.abs(y-pred))),"rmse":float(np.sqrt(np.mean((y-pred)**2))),"n":int(len(y))}

def summarize(predictions):
    rows=[]
    for scope,g in predictions.groupby("month"):
        m=metrics(g.target_spread,g.predicted_spread); m["scope"]=scope; rows.append(m)
    m=metrics(predictions.target_spread,predictions.predicted_spread); m["scope"]="overall"; rows.append(m)
    return pd.DataFrame(rows)
