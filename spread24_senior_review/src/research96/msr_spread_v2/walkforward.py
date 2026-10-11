"""docs/82 daily expanding fresh fit; smoke explicitly truncates debug history."""
from dataclasses import replace
from pathlib import Path
import sys
import time
import numpy as np
import torch
from .config import Config,FAST24
from .contracts import allowed_day,assert_write_path,ContractError
from .dataset import CurveStore,build_sample,training_panel,evaluation_label
from .trainer import fit_day
from .evaluate import evaluate_day
from .artifacts import new_run,write_json,write_csv,save_day,run_manifest,to_numpy
from .audit import scientific_gate,code_audit,file_hash

def resolve_device(device):
    if device=="auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device=="cuda" and not torch.cuda.is_available():
        raise ContractError("CUDA unavailable; CPU is supported")
    return device

def run_walkforward(dates,mode="smoke",device="cpu",profile="scientific",
                    output_root=None,source=None,store=None,label_reader=None,cfg=None,
                    review_receipt=None,command=None):
    dates = [allowed_day(d).isoformat() for d in dates]
    if not dates or dates!=sorted(set(dates)):
        raise ContractError("unique chronological target dates required")
    if mode not in ("smoke","single-day","fast24-lowhigh","fast24-full","fast24-screen"):
        raise ContractError("unknown execution mode")
    if mode!="smoke":
        scientific_gate(review_receipt)
        if mode=="fast24-screen" and profile!="fast-screen":
            raise ContractError("fast24-screen requires profile=fast-screen")
        if mode in ("fast24-lowhigh","fast24-full") and profile!="scientific":
            raise ContractError("scientific FAST24 requires profile=scientific")
        if mode=="single-day" and profile=="no-refit-fast-dev":
            raise ContractError("no-refit-fast-dev is debugging-only")
    cfg = cfg or Config()
    if mode=="smoke":
        cfg = replace(cfg,epochs_a=1,epochs_b=1,epochs_c=1)
    device = resolve_device(device)
    run = new_run(output_root)
    started = time.perf_counter()
    start = time.perf_counter()
    if store is None:
        store = CurveStore.canonical(max(dates),**({"path":source} if source else {}))
    read_seconds = time.perf_counter()-start
    label_reader = label_reader or (lambda d:evaluation_label(d,**({"path":source} if source else {})))
    days,metric_rows = [],[]
    try:
        audit = code_audit()
        if audit["status"]!="PASS":
            raise ContractError(str(audit))
        write_json(run/"CODE_AUDIT.json",audit)
        for target in dates:
            timing = {}
            start = time.perf_counter()
            samples,y,rejected = training_panel(store,target,smoke=mode=="smoke")
            sample = build_sample(store,target) # no label read in eligibility/features
            timing["data"] = time.perf_counter()-start
            write_json(run/(target+"_DATA_AUDIT.json"),dict(rejected=rejected,eligible_days=len(samples),
                history_profile="SMOKE_TRUNCATED_46_DAYS" if mode=="smoke" else "DAILY_EXPANDING_FROM_2023_01_01",
                max_label_day=max(s.target_day for s in samples),**sample.provenance))
            model,pre,training = fit_day(samples,y,target,cfg,device,profile)
            timing.update(training["timings"])
            start = time.perf_counter()
            model.eval()
            x = torch.as_tensor(pre.transform(sample.x[None]),device=device)
            da = torch.as_tensor(sample.da[None],dtype=torch.float64,device=device)
            with torch.no_grad():
                out = model(x,da,scenarios=cfg.m_eval)
            timing["scenario"] = time.perf_counter()-start
            if not torch.equal(out["spread_scenarios"],out["low_scenarios"]+out["high_scenarios"]):
                raise RuntimeError("scenario identity failed")
            # Persist actual prediction/scenario arrays BEFORE the target evaluation reader.
            seal_root = assert_write_path(run/"sealed"/target)
            seal_root.mkdir(parents=True,exist_ok=False)
            prediction = assert_write_path(seal_root/"PREDICTION_BEFORE_TRUTH.npz")
            saved = to_numpy(out)
            with prediction.open("xb") as f:
                np.savez_compressed(f,spread_scenarios=saved["spread_scenarios"],
                    spread_median=saved["spread_median"],p_positive=saved["p_positive"],
                    target_da=sample.da)
            write_json(seal_root/"PREDICTION_SEAL.json",dict(prediction_sha256=file_hash(prediction),
                target_day=target,max_label_day=training["max_label_day"],target_truth_opened=False))
            start = time.perf_counter()
            truth = label_reader(target)
            metrics,slots = evaluate_day(saved,truth,sample.da)
            timing["evaluation"] = time.perf_counter()-start
            start = time.perf_counter()
            day_root = save_day(run,sample,out,model,pre,training,metrics,slots,timing)
            timing["artifacts"] = time.perf_counter()-start
            days.append(dict(target_day=target,max_label_day=training["max_label_day"],
                parameter_count=sum(p.numel() for p in model.parameters()),directory=str(day_root),
                timing=timing,gradient_reachability=training["gradient_audit"]["reachability"]))
            metric_rows.append(dict(target_day=target,month=target[:7],
                **{k:v for k,v in metrics.items() if isinstance(v,(int,float)) or v is None}))
            print(f"CPU_SMOKE_DAY={target} complete; labels<={training['max_label_day']}; params={days[-1]['parameter_count']}",flush=True)
        timings = dict(source_read=read_seconds,total=time.perf_counter()-started,days=days)
        write_csv(run/"DAILY_METRICS.csv",metric_rows)
        # Fixed target panel only: monthly diagnostics do not select model/config.
        monthly = []
        for month in sorted({r["month"] for r in metric_rows}):
            rows = [r for r in metric_rows if r["month"]==month]
            summary = dict(month=month,target_days=len(rows))
            for k in metric_rows[0]:
                if k not in ("target_day","month"):
                    vals = [r[k] for r in rows if r[k] is not None]
                    summary[k] = float(np.mean(vals)) if vals else None
            monthly.append(summary)
        write_csv(run/"MONTHLY_DIAGNOSTICS.csv",monthly)
        manifest = run_manifest(run,cfg,store,dates,mode,profile,days,timings,command or " ".join(sys.argv))
        write_json(run/"RUN_MANIFEST.json",manifest)
        return run,manifest
    except Exception as e:
        write_json(run/"FAILURE.json",dict(status="BLOCKED",error_type=type(e).__name__,error=str(e),
                   elapsed=time.perf_counter()-started,completed_days=days,smoke=mode=="smoke"))
        raise
