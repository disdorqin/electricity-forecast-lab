"""Analyze frozen E2-C3 panel; target-day bootstrap is the paired unit."""
import json
from pathlib import Path
import numpy as np, pandas as pd
HERE=Path(__file__).resolve().parent; F80=HERE.parent/"fusion_coarse"/"runs"
W={"W1":("2026-02-12","2026-02-18"),"W2":("2026-04-12","2026-04-18"),"W3":("2026-06-12","2026-06-18"),"W4":("2026-08-07","2026-08-13")}
def rec(a,d,benchmark=False):
 p=F80/f"E2B1-F80-{d}/RUN_RECORD.json" if a=="R0" else (HERE/"runs"/"benchmark"/f"E2C3-{a}-{d}"/"RUN_RECORD.json" if benchmark else HERE/"runs"/f"E2C3-{a}-{d}"/"RUN_RECORD.json")
 return json.loads(p.read_text(encoding="utf-8"))
def frame(a,d,benchmark=False):
 r=rec(a,d,benchmark);folder=Path(r["run_dir"]);m=json.loads((folder/"manifest.json").read_text(encoding="utf-8"));q=pd.read_parquet(folder/"predictions.parquet")
 expected={"R0":"all","R1":"drop_mag","R2":"drop_dir"}[a]
 checks={"status":r.get("status")=="PASS","mode":m.get("mode")=="A2","profile":m.get("profile")=="default","device":str(m.get("device","")).startswith("cuda"),"amp":m.get("amp") is True,"seed":m.get("seed")==20260924,"objective":m.get("objective_mode")=="dir_only","checkpoint":m.get("checkpoint_policy")=="direction_first","gradient":m.get("gradient_policy")=="vanilla","architecture":m.get("architecture_mode")=="full_current","tabular_route":m.get("direction_tabular_mode","current")=="current","gate_route":m.get("strong_role_profile","all")==expected,"alpha":abs(float(m.get("direction_fusion_alpha",-1))-.8)<1e-12,"config":m.get("config_sha256")=="8c981156cecf6e114cf3d4eeae6ba418d62e361195a3d191d1b4b11766b5488c","selector":m.get("selector_sha256")=="ed348cfd9fd911bc675d7fd920485b1a748c159a3e0a395b3af02af01015092f","source":m.get("source_sha256")=="a1b86f956d9fb18a483d473cbc0334e1078097f6ef75274b2804488968a349ea","sequence":m.get("sequence_manifest_sha256")=="9144bbed33369a4bed5a8acc508a71f50836e55667e3cc4e768badfd35ce7a82"}
 if not all(checks.values()) or len(q)!=24 or q.hour_business.tolist()!=list(range(1,25)): raise RuntimeError(f"invalid/missing run {a} {d}: {checks}")
 q["arm"]=a;q["target_day"]=d;return q
def metric(q,scope):
 y=q.y_true_model.to_numpy()>0;p=q.direction_hat.to_numpy().astype(bool);prob=q.p_positive.to_numpy()
 from sklearn.metrics import roc_auc_score,brier_score_loss
 tp=(p&y).sum();tn=(~p&~y).sum();pos=y.sum();neg=(~y).sum()
 pr=tp/pos if pos else np.nan;nr=tn/neg if neg else np.nan
 groups=list(q.groupby("target_day")); zero_pos=sum(((z.y_true_model.to_numpy()>0).sum()>0) and (z.loc[z.y_true_model>0,"direction_hat"].astype(bool).sum()==0) for _,z in groups);zero_neg=sum(((z.y_true_model.to_numpy()<=0).sum()>0) and (z.loc[z.y_true_model<=0,"direction_hat"].astype(bool).sum()==len(z.loc[z.y_true_model<=0])) for _,z in groups)
 return {"arm":q.arm.iloc[0],"scope":scope,"n_slots":len(q),"n_days":q.target_day.nunique(),"raw":float((p==y).mean()),"balanced":float(np.nanmean([pr,nr])),"positive_recall":float(pr),"negative_recall":float(nr),"auc":float(roc_auc_score(y,prob)) if len(np.unique(y))==2 else np.nan,"brier":float(brier_score_loss(y,prob)),"predicted_positive_fraction":float(p.mean()),"true_positive_fraction":float(y.mean()),"one_class_days":int(q.groupby("target_day").direction_hat.nunique().eq(1).sum()),"positive_recall_zero_days":int(zero_pos),"negative_recall_zero_days":int(zero_neg)}
def main():
 days=[(w,str((pd.Timestamp(a)+pd.Timedelta(days=i)).date())) for w,(a,b) in W.items() for i in range(7)]
 allq=pd.concat([frame(a,d) for a in ("R0","R1","R2") for _,d in days],ignore_index=True)
 daily=[]
 for a in ("R0","R1","R2"):
  for w,d in days: daily.append({**metric(allq[(allq.arm==a)&(allq.target_day==d)],w),"target_day":d})
 day=pd.DataFrame(daily);day.to_csv(HERE/"daily_metrics.csv",index=False)
 over=pd.DataFrame([metric(allq[allq.arm==a],"overall") for a in ("R0","R1","R2")])
 win=pd.DataFrame([metric(allq[(allq.arm==a)&allq.target_day.isin(day.loc[day.scope==w,"target_day"])],w) for a in ("R0","R1","R2") for w in W])
 win.to_csv(HERE/"window_metrics.csv",index=False)
 hour=pd.DataFrame([metric(allq[(allq.arm==a)&allq.hour_business.between(lo,hi)],h) for a in ("R0","R1","R2") for h,lo,hi in (("H1",1,8),("H2",9,16),("H3",17,24))]);hour.to_csv(HERE/"hour_segment_metrics.csv",index=False)
 summary=over.copy();summary["min_window_raw"]=[win[win.arm==a].raw.min() for a in summary.arm];summary["window_raw_std"]=[win[win.arm==a].raw.std(ddof=0) for a in summary.arm];summary.to_csv(HERE/"model_summary.csv",index=False)
 rng=np.random.default_rng(20260924);paired=[]
 for a,b in (("R1","R0"),("R2","R0"),("R1","R2")):
  x=day[day.arm==a].set_index("target_day").sort_index();y=day[day.arm==b].set_index("target_day").sort_index();dr=x.raw.to_numpy()-y.raw.to_numpy();db=x.balanced.to_numpy()-y.balanced.to_numpy();ix=rng.integers(0,28,(10000,28));rc=np.quantile(dr[ix].mean(1),[.025,.975]);bc=np.quantile(db[ix].mean(1),[.025,.975]); signs=[np.sign(dr[np.array([z==w for z in x.scope])].mean()) for w in W]
  paired.append({"comparison":f"{a}-{b}","mean_delta_raw":dr.mean(),"raw_ci_low":rc[0],"raw_ci_high":rc[1],"raw_ci_excludes_zero":bool(rc[0]>0 or rc[1]<0),"raw_W_T_L":f"{(dr>0).sum()}/{(dr==0).sum()}/{(dr<0).sum()}","mean_delta_balanced":db.mean(),"balanced_ci_low":bc[0],"balanced_ci_high":bc[1],"balanced_ci_excludes_zero":bool(bc[0]>0 or bc[1]<0),"window_delta_raw_signs":json.dumps(signs)})
 pd.DataFrame(paired).to_csv(HERE/"paired_summary.csv",index=False)
 c=summary.set_index("arm"); safety={}
 for a in ("R0","R1","R2"):
  r=c.loc[a];z=day[day.arm==a];checks={"balanced":bool(r.balanced>=c.loc["R0"].balanced-.01),"positive_recall":bool(r.positive_recall>=c.loc["R0"].positive_recall-.05),"one_class_days":int(z.one_class_days.sum())<=int(day[day.arm=="R0"].one_class_days.sum())+2,"positive_recall_zero_days":int(z.positive_recall_zero_days.sum())<=int(day[day.arm=="R0"].positive_recall_zero_days.sum())+2};safety[a]={"admissible":all(checks.values()),"checks":checks}
 (HERE/"safety_admissibility.json").write_text(json.dumps(safety,indent=2),encoding="utf-8")
 p={x["comparison"]:x for x in paired}; g1,g2=c.loc["R1"],c.loc["R2"]
 def stable(key,sgn):
  reverse=key in {"R0-R1","R0-R2","R2-R1"}; base={"R0-R1":"R1-R0","R0-R2":"R2-R0","R2-R1":"R1-R2"}.get(key,key);x=p[base]; signs=np.sign(json.loads(x["window_delta_raw_signs"]))*(-1 if reverse else 1); delta=x["mean_delta_raw"]*(-1 if reverse else 1)
  ci_excludes=x["raw_ci_excludes_zero"]
  return ci_excludes and abs(delta)>=.02 and sum(signs==sgn)>=3
 if safety["R1"]["admissible"] and not stable("R1-R0",-1): label="DIR_ROLE_ALIGNED" if not p["R1-R0"]["raw_ci_excludes_zero"] or abs(p["R1-R0"]["mean_delta_raw"])<.02 else "NO_CLEAR_ROLE_EFFECT"
 elif safety["R2"]["admissible"] and abs(p["R2-R0"]["mean_delta_raw"])<.02 and (stable("R0-R1",1) or stable("R2-R1",1)): label="DIR_FEATURES_REDUNDANT"
 elif safety["R2"]["admissible"] and stable("R2-R0",1): label="MAG_FEATURES_HELP_DIRECTION"
 elif safety["R0"]["admissible"] and stable("R0-R1",1) and stable("R0-R2",1): label="BOTH_TASK_SPECIFIC_GROUPS_HELP"
 elif safety["R1"]["admissible"] and stable("R1-R0",1): label="ROLE_FILTER_PROMISING_BUT_UNPROVEN"
 else: label="NO_CLEAR_ROLE_EFFECT"
 (HERE/"E2_C3_GATE.md").write_text(f"# E2-C3 Gate\n\n**E2_C3_GATE ROLE_SIGNAL = `{label}`**\n\nUses only frozen W1-W4 panel, paired day-cluster bootstrap (10,000 draws, seed 20260924), cross-window evidence and R0 safety. Benchmark excluded. Stop for human review.\n",encoding="utf-8")
 # engineering day summary and figures
 bench=pd.concat([frame(a,"2026-02-13",benchmark=True) for a in ("R0","R1","R2")],ignore_index=True);pd.DataFrame([metric(bench[bench.arm==a],"benchmark_engineering_only") for a in ("R0","R1","R2")]).to_csv(HERE/"benchmark/benchmark_metrics.csv",index=False)
 import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
 ax=summary.set_index("arm")[["raw","balanced"]].plot.bar();ax.figure.tight_layout();ax.figure.savefig(HERE/"figures/overall_metrics.png",dpi=150);plt.close(ax.figure)
 ax=win.pivot(index="scope",columns="arm",values="raw").reindex(["W1","W2","W3","W4"]).plot.bar(ylim=(.4,.8));ax.figure.tight_layout();ax.figure.savefig(HERE/"figures/window_raw.png",dpi=150);plt.close(ax.figure)
 pp=pd.DataFrame(paired).iloc[:2];fig,ax=plt.subplots(figsize=(6,3));yy=np.arange(len(pp));ax.errorbar(pp.mean_delta_raw,yy,xerr=[pp.mean_delta_raw-pp.raw_ci_low,pp.raw_ci_high-pp.mean_delta_raw],fmt="o",capsize=4);ax.axvline(0,color="black",lw=1);ax.axvline(.02,color="gray",ls="--");ax.axvline(-.02,color="gray",ls="--");ax.set_yticks(yy,pp.comparison);ax.set_xlabel("Day-paired ΔRaw with 95% bootstrap CI");fig.tight_layout();fig.savefig(HERE/"figures/paired_raw_ci.png",dpi=150);plt.close(fig)
 print(json.dumps({"signal":label,"overall":summary.to_dict("records"),"paired":paired,"safety":safety},indent=2,default=float))
if __name__=="__main__":main()


