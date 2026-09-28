"""Summarize the fixed E2-C1 panel, paired bootstrap, guardrail, and preregistered gate."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent; REPO=HERE.parents[3]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(REPO/"src"))
from src.TafM_改进源码.metrics import canonical_metrics

F80=REPO/"experiments/first_test/E2_architecture/fusion_coarse/runs"
RUNS=HERE/"runs";SEED=20260924;N_BOOT=10000
WINDOWS={"W1":("2026-02-12","2026-02-18"),"W2":("2026-04-12","2026-04-18"),
         "W3":("2026-06-12","2026-06-18"),"W4":("2026-08-07","2026-08-13")}

def get_record(arm,day):
    path=(F80/f"E2B1-F80-{day}/RUN_RECORD.json") if arm=="C0" else (RUNS/f"E2C1-{arm}-{day}/RUN_RECORD.json")
    return json.loads(path.read_text(encoding="utf-8"))

def run_manifest(rec):
    path=Path(rec["run_dir"])/"manifest.json"
    raw=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    raw.update(rec.get("manifest",{}))
    return raw

def load_predictions(arm,day):
    rec=get_record(arm,day)
    manifest=run_manifest(rec)
    expected_route={"C0":"current","C1":"strong_only","C2":"weak_only"}[arm]
    checks={"status":rec.get("status")=="PASS","cuda":str(manifest.get("device","")).startswith("cuda"),
        "amp":manifest.get("amp") is True,"mode":manifest.get("mode")=="A2",
        "objective":manifest.get("objective_mode")=="dir_only","checkpoint":manifest.get("checkpoint_policy")=="direction_first",
        "gradient":manifest.get("gradient_policy")=="vanilla","train":manifest.get("train_mode")=="stage_a",
        "seed":int(manifest.get("seed",-1))==20260924,"architecture":manifest.get("architecture_mode")=="full_current",
        "alpha":abs(float(manifest.get("direction_fusion_alpha",-1))-.8)<1e-12,
        "profile":manifest.get("profile")=="default","route":manifest.get("direction_tabular_mode",expected_route)==expected_route,
        "config":manifest.get("config_sha256")=="8c981156cecf6e114cf3d4eeae6ba418d62e361195a3d191d1b4b11766b5488c",
        "selector":manifest.get("selector_sha256")=="ed348cfd9fd911bc675d7fd920485b1a748c159a3e0a395b3af02af01015092f",
        "source":manifest.get("source_sha256")=="a1b86f956d9fb18a483d473cbc0334e1078097f6ef75274b2804488968a349ea",
        "sequence":manifest.get("sequence_manifest_sha256")=="9144bbed33369a4bed5a8acc508a71f50836e55667e3cc4e768badfd35ce7a82"}
    if not all(checks.values()):
        raise RuntimeError(f"invalid protocol artifact {arm} {day}: {checks}")
    f=Path(rec["run_dir"])/"predictions.parquet"
    q=pd.read_parquet(f)
    if len(q)!=24 or q.hour_business.tolist()!=list(range(1,25)):
        raise RuntimeError(f"incomplete 24-slot prediction artifact: {f}")
    q["arm"]=arm;q["target_day"]=day
    return q,rec

def calc(q):
    return canonical_metrics(q.y_true_model.to_numpy(),q.direction_hat.to_numpy().astype(bool),q.magnitude_hat.to_numpy(),
        direction_probability=q.p_positive.to_numpy(),hour=q.hour_business.to_numpy())

def metrics_row(q,scope,arm):
    m=calc(q); pred=q.direction_hat.to_numpy().astype(bool)
    for key in ("direction_by_hour","magnitude_by_hour","magnitude_by_true_sign"):
        m.pop(key,None)
    m.update({"arm":arm,"scope":scope,"n_slots":len(q),"n_days":q.target_day.nunique(),
        "predicted_positive_fraction":float(pred.mean()),"true_positive_fraction":float((q.y_true_model.to_numpy()>0).mean())})
    return m

def collect():
    rows=[];run_rows=[];preds=[]
    all_days=[(w,str((pd.Timestamp(a)+pd.Timedelta(days=i)).date())) for w,(a,b) in WINDOWS.items() for i in range(7)]
    for arm in ("C0","C1","C2"):
        for window,day in all_days:
            q,rec=load_predictions(arm,day);preds.append(q)
            m=calc(q);p=q.direction_hat.to_numpy().astype(bool);y=q.y_true_model.to_numpy()>0
            rows.append({"arm":arm,"window":window,"target_day":day,"n_slots":len(q),
                "raw":m["raw_direction_accuracy"],"balanced":m["balanced_accuracy"],
                "positive_recall":m["positive_recall"],"negative_recall":m["nonpositive_recall"],
                "auc":m["auc"],"brier":m["brier"],"predicted_positive_fraction":float(p.mean()),
                "true_positive_fraction":float(y.mean()),"one_class_day":bool(p.all() or (~p).all()),
                "positive_recall_zero_day":bool(m["positive_recall"]==0),
                "negative_recall_zero_day":bool(m["nonpositive_recall"]==0)})
            manifest=run_manifest(rec)
            run_rows.append({"arm":arm,"window":window,"target_day":day,"run_id":rec.get("run_id"),
                "source_kind":"REUSED_F80" if arm=="C0" else "FRESH",
                "status":rec.get("status"),"device":manifest.get("device", "cuda" if arm=="C0" else None),
                "amp":manifest.get("amp", True if arm=="C0" else None),
                "run_wall_seconds":rec.get("wall_seconds"),
                "training_wall_seconds":manifest.get("wall_time_total_seconds"),
                "mean_epoch_seconds":manifest.get("training_epoch_seconds_mean"),
                "best_epoch":manifest.get("best_epoch"),"stop_epoch":manifest.get("stop_epoch"),
                "epochs_run":manifest.get("epochs_run"),"cuda_peak_memory_bytes":manifest.get("cuda_peak_memory_bytes"),
                "parameter_count_total":manifest.get("parameter_count_total"),
                "config_sha256":manifest.get("config_sha256"),"selector_sha256":manifest.get("selector_sha256"),
                "source_sha256":manifest.get("source_sha256"),"sequence_manifest_sha256":manifest.get("sequence_manifest_sha256")})
    return pd.DataFrame(rows),pd.DataFrame(run_rows),pd.concat(preds,ignore_index=True)

def paired(df,a,b):
    x=df[df.arm==a].set_index("target_day").sort_index();y=df[df.arm==b].set_index("target_day").sort_index()
    if not x.index.equals(y.index) or len(x)!=28: raise RuntimeError(f"paired panel mismatch {a}/{b}")
    d_raw=(x.raw-y.raw).to_numpy();d_bal=(x.balanced-y.balanced).to_numpy()
    rng=np.random.default_rng(SEED);draws=rng.integers(0,len(x),size=(N_BOOT,len(x)))
    raw_boot=d_raw[draws].mean(1);bal_boot=d_bal[draws].mean(1)
    raw_ci=np.quantile(raw_boot,[.025,.975]);bal_ci=np.quantile(bal_boot,[.025,.975])
    wins=int((d_raw>0).sum());ties=int((d_raw==0).sum());losses=int((d_raw<0).sum())
    sign_by_window={w:int(np.sign(d_raw[np.asarray([z==w for z in x.window])].mean())) for w in WINDOWS}
    return {"comparison":f"{a}-{b}","mean_delta_raw":float(d_raw.mean()),"raw_ci_low":float(raw_ci[0]),"raw_ci_high":float(raw_ci[1]),
        "raw_ci_excludes_zero":bool(raw_ci[0]>0 or raw_ci[1]<0),"raw_W_T_L":f"{wins}/{ties}/{losses}",
        "mean_delta_balanced":float(np.nanmean(d_bal)),"balanced_ci_low":float(bal_ci[0]),"balanced_ci_high":float(bal_ci[1]),
        "balanced_ci_excludes_zero":bool(bal_ci[0]>0 or bal_ci[1]<0),"window_delta_raw_signs":sign_by_window,
        "same_sign_windows":max(sum(v>0 for v in sign_by_window.values()),sum(v<0 for v in sign_by_window.values()))}

def gate_label(summary,paired_rows,safety):
    p={x["comparison"]:x for x in paired_rows};c0=summary.set_index("arm").loc["C0"]
    c1=summary.set_index("arm").loc["C1"];c2=summary.set_index("arm").loc["C2"]
    def stable(name,sign):
        x=p[name];return x["raw_ci_excludes_zero"] and x["same_sign_windows"]>=3 and (x["mean_delta_raw"]*sign)>=.02
    improve1=safety["C1"]["admissible"] and stable("C1-C0",1)
    improve2=safety["C2"]["admissible"] and stable("C2-C0",1)
    if improve1 or improve2:return "TABULAR_MIXING_PROBLEM"
    c0beats1=stable("C0-C1",1);c0beats2=stable("C0-C2",1)
    c2_weak=(c2.balanced_accuracy<=.52 or stable("C1-C2",1))
    competitive=(abs(p["C2-C0"]["mean_delta_raw"])<=.02 and abs(p["C2-C1"]["mean_delta_raw"])<=.02 and
        not(p["C2-C0"]["raw_ci_excludes_zero"] and p["C2-C0"]["mean_delta_raw"]<=-.02) and
        not(p["C2-C1"]["raw_ci_excludes_zero"] and p["C2-C1"]["mean_delta_raw"]<=-.02))
    if c0beats1 and c2_weak:return "WEAK_RESIDUAL"
    if competitive and c2.balanced_accuracy>.52:return "WEAK_UNEXPECTEDLY_STRONG"
    if safety["C1"]["admissible"] and c1.raw_direction_accuracy>=c0.raw_direction_accuracy-.02 and c2_weak and stable("C1-C2",1):return "STRONG_DOMINANT"
    if c0beats1 and c0beats2 and safety["C0"]["admissible"]:return "STRONG_WEAK_SYNERGY"
    near1=abs(c1.balanced_accuracy-.5)<=.02;near2=abs(c2.balanced_accuracy-.5)<=.02
    if near1 and near2 and not c0beats1 and not c0beats2:return "BOTH_INTERNAL_WEAK"
    return "INCONCLUSIVE"

def make_figures(summary,window_df,paired_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir=HERE/"figures";figdir.mkdir(exist_ok=True)
    arms=["C0","C1","C2"]
    ax=summary.set_index("arm").loc[arms,["raw_direction_accuracy","balanced_accuracy"]].plot.bar(figsize=(7,4),ylim=(.45,.65),rot=0)
    ax.set_ylabel("Accuracy");ax.set_title("E2-C1 overall direction metrics (28 DEV days)");ax.legend(["Raw","Balanced"])
    fig=ax.get_figure();fig.tight_layout();fig.savefig(figdir/"overall_metrics.png",dpi=160);plt.close(fig)
    w=window_df.pivot(index="scope",columns="arm",values="raw_direction_accuracy").reindex(["W1","W2","W3","W4"])
    ax=w[arms].plot.bar(figsize=(7,4),ylim=(.4,.8),rot=0);ax.set_ylabel("Raw");ax.set_title("Raw by fixed window");ax.legend(title="Arm")
    fig=ax.get_figure();fig.tight_layout();fig.savefig(figdir/"window_raw.png",dpi=160);plt.close(fig)
    p=paired_df[paired_df.comparison.isin(["C1-C0","C2-C0"])].copy();y=np.arange(len(p))
    fig,ax=plt.subplots(figsize=(7,3.5));ax.errorbar(p.mean_delta_raw,y,xerr=[p.mean_delta_raw-p.raw_ci_low,p.raw_ci_high-p.mean_delta_raw],fmt="o",capsize=4)
    ax.axvline(0,color="black",lw=1);ax.axvline(.02,color="gray",ls="--",lw=1);ax.axvline(-.02,color="gray",ls="--",lw=1)
    ax.set_yticks(y,p.comparison);ax.set_xlabel("Paired day-mean ΔRaw (95% day-bootstrap CI)");ax.set_title("Strong/Weak arms vs current F80")
    fig.tight_layout();fig.savefig(figdir/"paired_raw_ci.png",dpi=160);plt.close(fig)

def main():
    daily,runtime,preds=collect()
    bench_rows=[];bench_runtime=[]
    for arm in ("C0","C1","C2"):
        q,rec=load_predictions(arm,"2026-02-13");m=metrics_row(q,"benchmark_engineering_only",arm)
        bench_rows.append(m);manifest=run_manifest(rec)
        bench_runtime.append({"arm":arm,"window":"benchmark_engineering_only","target_day":"2026-02-13",
            "run_id":rec.get("run_id"),"source_kind":"REUSED_F80" if arm=="C0" else "FRESH",
            "status":rec.get("status"),"device":manifest.get("device"),"amp":manifest.get("amp"),
            "run_wall_seconds":rec.get("wall_seconds"),"training_wall_seconds":manifest.get("wall_time_total_seconds"),
            "mean_epoch_seconds":manifest.get("training_epoch_seconds_mean"),"best_epoch":manifest.get("best_epoch"),
            "stop_epoch":manifest.get("stop_epoch"),"epochs_run":manifest.get("epochs_run"),
            "cuda_peak_memory_bytes":manifest.get("cuda_peak_memory_bytes"),"parameter_count_total":manifest.get("parameter_count_total"),
            "config_sha256":manifest.get("config_sha256"),"selector_sha256":manifest.get("selector_sha256"),
            "source_sha256":manifest.get("source_sha256"),"sequence_manifest_sha256":manifest.get("sequence_manifest_sha256")})
    daily.to_csv(HERE/"daily_metrics.csv",index=False)
    pd.DataFrame(bench_rows).to_csv(HERE/"benchmark/benchmark_metrics.csv",index=False)
    runtime=pd.concat([runtime,pd.DataFrame(bench_runtime)],ignore_index=True)
    runtime.to_csv(HERE/"runtime.csv",index=False)
    overall=pd.DataFrame([metrics_row(preds[preds.arm==a],"overall",a) for a in ("C0","C1","C2")])
    counts=daily.groupby("arm").agg(one_class_days=("one_class_day","sum"),positive_recall_zero_days=("positive_recall_zero_day","sum"),negative_recall_zero_days=("negative_recall_zero_day","sum")).reset_index()
    summary=overall.merge(counts,on="arm")
    windows=[]
    for a in ("C0","C1","C2"):
        vals=[]
        for w in WINDOWS:
            q=preds[(preds.arm==a)&(preds.target_day.isin(daily.loc[daily.window==w,"target_day"]))]
            row=metrics_row(q,w,a);windows.append(row);vals.append(row["raw_direction_accuracy"])
        summary.loc[summary.arm==a,"min_window_raw"]=min(vals);summary.loc[summary.arm==a,"window_raw_std"]=float(np.std(vals,ddof=0))
    win_df=pd.DataFrame(windows);win_df.to_csv(HERE/"window_metrics.csv",index=False)
    segments=[]
    for a in ("C0","C1","C2"):
        for h,(lo,hi) in {"H1":(1,8),"H2":(9,16),"H3":(17,24)}.items():
            q=preds[(preds.arm==a)&preds.hour_business.between(lo,hi)]
            segments.append(metrics_row(q,h,a))
    pd.DataFrame(segments).to_csv(HERE/"hour_segment_metrics.csv",index=False)
    summary.to_csv(HERE/"model_summary.csv",index=False)
    comparisons=[]
    for a,b in (("C1","C0"),("C2","C0"),("C0","C1"),("C0","C2"),("C1","C2"),("C2","C1")):
        comparisons.append(paired(daily,a,b))
    pd.DataFrame([{k:v for k,v in x.items() if k!="window_delta_raw_signs"} for x in comparisons]).to_csv(HERE/"paired_summary.csv",index=False)
    make_figures(summary,win_df,pd.DataFrame([{k:v for k,v in x.items() if k!="window_delta_raw_signs"} for x in comparisons]))
    c0=summary.set_index("arm").loc["C0"]
    safety={}
    for a in ("C0","C1","C2"):
        r=summary.set_index("arm").loc[a]
        parts={"balanced":bool(r.balanced_accuracy>=c0.balanced_accuracy-.01),
            "positive_recall":bool(r.positive_recall>=c0.positive_recall-.05),
            "one_class_days":bool(r.one_class_days<=c0.one_class_days+2),
            "positive_recall_zero_days":bool(r.positive_recall_zero_days<=c0.positive_recall_zero_days+2)}
        safety[a]={"admissible":all(parts.values()),"checks":parts,
            "balanced_delta":float(r.balanced_accuracy-c0.balanced_accuracy),
            "positive_recall_delta":float(r.positive_recall-c0.positive_recall),
            "one_class_delta":int(r.one_class_days-c0.one_class_days),
            "positive_recall_zero_delta":int(r.positive_recall_zero_days-c0.positive_recall_zero_days)}
    (HERE/"safety_admissibility.json").write_text(json.dumps(safety,ensure_ascii=False,indent=2),encoding="utf-8")
    label=gate_label(summary,comparisons,safety)
    (HERE/"E2_C1_GATE.md").write_text("# E2-C1 Gate\n\n**E2_C1_GATE TABULAR_SIGNAL = `"+label+"`**\n\nDecision uses only the frozen W1-W4 DEV panel, the pre-registered 2 pp materiality rule, paired day-cluster bootstrap (10,000 draws, seed 20260924), cross-window signs, and the C0 safety anchor. See `paired_summary.csv`, `window_metrics.csv`, and `safety_admissibility.json`. Benchmark-day figures are engineering-only and excluded from this decision.\n\nNo model/selector/config changes are promoted by this screening result. Stop for human review.\n",encoding="utf-8")
    # concise machine-readable gate record used by the report
    report={"phase":"E2-C1","tabular_signal":label,"bootstrap_draws":N_BOOT,"bootstrap_seed":SEED,
        "safety":safety,"paired":comparisons,"overall":summary.to_dict(orient="records"),
        "windows":win_df.to_dict(orient="records"),"hour_segments":pd.DataFrame(segments).to_dict(orient="records")}
    (HERE/"analysis_summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,default=float),encoding="utf-8")
    print(json.dumps({"tabular_signal":label,"overall":summary[["arm","raw_direction_accuracy","balanced_accuracy","positive_recall","nonpositive_recall","auc","brier","predicted_positive_fraction","one_class_days","positive_recall_zero_days"]].to_dict(orient="records"),"paired":pd.DataFrame([{k:v for k,v in x.items() if k!="window_delta_raw_signs"} for x in comparisons]).to_dict(orient="records"),"safety":safety},ensure_ascii=False,indent=2,default=float))
    return 0

if __name__=="__main__":raise SystemExit(main())
