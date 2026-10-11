"""One frozen scientific GPU day, not a completed component adjudication panel."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from ...config import OUTPUT_ROOT, LAB_ROOT
from ...contracts import assert_write_path, ContractError
from ...audit import file_hash, package_hashes, frozen_check
from ...dataset import CurveStore, training_panel, build_sample, evaluation_label
from ...evaluate import evaluate_day
from ...artifacts import new_run, write_json, save_day, to_numpy, run_manifest
from .profile import NAME, BOUNDS, scientific_config
from .trainer import fit_day, synchronized_time

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--date", choices=("2026-01-15","2026-04-01","2026-07-15"), default="2026-01-15")
    p.add_argument("--qualification", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--review", required=True)
    args=p.parse_args()
    qualification=json.loads(Path(args.qualification).read_text(encoding="utf-8"))
    if qualification.get("status")!="PASS" or not torch.cuda.is_available():
        raise ContractError("G0 CUDA qualification missing")
    batch=qualification["batch_size"]
    cfg=scientific_config(batch)
    baseline=json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    review=json.loads(Path(args.review).read_text(encoding="utf-8"))
    code=package_hashes()
    drift={k:[v,code.get(k)] for k,v in baseline["v2_core"].items() if code.get(k)!=v}
    if drift or review.get("status")!="PASS" or review.get("independent_review")!="PASS":
        raise ContractError("independently reviewed core drift: "+str(drift))
    reviewed={k:v for k,v in baseline["v2_core"].items() if k.endswith(".py")}
    if review.get("code_hashes")!=reviewed:
        raise ContractError("independent review does not bind frozen core")
    run=new_run(OUTPUT_ROOT/"runtime_probe_20261010")
    print("RUN_ROOT="+str(run), flush=True)
    write_json(run/"CORE_REVIEW_BINDING.json", dict(core_drift=drift,
        reviewed_core_files=reviewed, original_review_sha256=file_hash(args.review),
        experiment_additions=sorted(set(code)-set(baseline["v2_core"])),
        note="Existing literal FULL core review preserved; new trainer tested separately."))
    write_json(run/"GPU_QUALIFICATION.json", qualification)
    write_json(run/"PROFILE.json",dict(name=NAME,config=cfg.to_dict(),bounds=BOUNDS,
        purpose="User-requested ONE-DAY runtime measurement; not full docs87 adjudication."))
    before=frozen_check(args.baseline,production=True)
    write_json(run/"ISOLATION_BEFORE.json",before)
    if not before["zero_drift"]:
        raise ContractError("protected trees changed since snapshot; inspect before training")
    started=synchronized_time()
    try:
        start=time.perf_counter()
        store=CurveStore.canonical(args.date)
        source_seconds=time.perf_counter()-start
        start=time.perf_counter()
        samples,labels,rejected=training_panel(store,args.date,smoke=False)
        sample=build_sample(store,args.date)
        data_seconds=time.perf_counter()-start
        write_json(run/"DATA_AUDIT.json",dict(eligible_days=len(samples),rejected=rejected,
            first_training_day=samples[0].target_day,max_label_day=samples[-1].target_day,
            source=store.source_identity,profile="DAILY_EXPANDING_FROM_2023_01_01",**sample.provenance))
        print(f"DATA_READY days={len(samples)} labels<={samples[-1].target_day} "
              f"source={source_seconds:.2f}s panel={data_seconds:.2f}s",flush=True)
        torch.cuda.reset_peak_memory_stats()
        fit_start=synchronized_time()
        model,pre,training=fit_day(samples,labels,args.date,batch)
        fit_seconds=synchronized_time()-fit_start
        write_json(run/"TRAINING_COMPLETED.json",training)
        start=synchronized_time()
        model.eval()
        x=torch.as_tensor(pre.transform(sample.x[None]),device="cuda")
        da=torch.as_tensor(sample.da[None],dtype=torch.float64,device="cuda")
        with torch.no_grad():
            out=model(x,da,scenarios=cfg.m_eval)
        prediction_seconds=synchronized_time()-start
        if not torch.equal(out["spread_scenarios"],out["low_scenarios"]+out["high_scenarios"]):
            raise RuntimeError("exact scenario reconstruction violated")
        seal=assert_write_path(run/"sealed"/args.date)
        seal.mkdir(parents=True,exist_ok=False)
        saved=to_numpy(out)
        prediction=assert_write_path(seal/"PREDICTION_BEFORE_TRUTH.npz")
        with prediction.open("xb") as f:
            np.savez_compressed(f,spread_scenarios=saved["spread_scenarios"],
                spread_median=saved["spread_median"],p_positive=saved["p_positive"],target_da=sample.da)
        write_json(seal/"PREDICTION_SEAL.json",dict(target_day=args.date,target_truth_opened=False,
            prediction_sha256=file_hash(prediction),max_label_day=training["max_label_day"]))
        start=time.perf_counter()
        truth=evaluation_label(args.date)
        metrics,slots=evaluate_day(saved,truth,sample.da)
        evaluation_seconds=time.perf_counter()-start
        timing=dict(source_read=source_seconds,data=data_seconds,fit_total=fit_seconds,
                    prediction=prediction_seconds,evaluation=evaluation_seconds,**training["timings"])
        start=time.perf_counter()
        root=save_day(run,sample,out,model,pre,training,metrics,slots,timing)
        timing["artifacts"]=time.perf_counter()-start
        timing["total_day"]=synchronized_time()-started
        receipt=dict(status="COMPLETE",target_day=args.date,profile=NAME,
            gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
            batch_size=batch,training_days=len(samples),selected_epochs=training["selected_epochs"],
            executed_epochs=[{k:a[k] for k in ("phase","stage","executed_epochs")} for a in training["stage_audits"]],
            timings=timing,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
            parameter_count=sum(p.numel() for p in model.parameters()),
            cold_data_preparation_included=True,setup_and_tests_included=False,
            one_day_only=True,three_day_median_not_measured=True,primary_panel_not_selected=True,
            scientific_adjudication_not_complete=True,run_root=str(run))
        write_json(run/"ONE_DAY_RUNTIME.json",receipt)
        manifest=run_manifest(run,cfg,store,[args.date],"scientific-runtime-probe",NAME,
            [dict(target_day=args.date,directory=str(root),max_label_day=training["max_label_day"])],timing," ".join(sys.argv))
        manifest["eligible_for_scientific_claim"]=False
        manifest["scope"]="one-day scientific-budget engineering runtime, no architecture judgment"
        write_json(run/"RUN_MANIFEST.json",manifest)
        after=frozen_check(args.baseline,production=True)
        current=package_hashes()
        after["v2_core_drift"]={k:[v,current.get(k)] for k,v in baseline["v2_core"].items()
                                 if current.get(k)!=v}
        write_json(run/"ZERO_POLLUTION.json",after)
        if not after["zero_drift"] or after["v2_core_drift"]:
            raise ContractError("post-run protected/core drift")
        print("ONE_DAY_RUNTIME="+json.dumps(receipt),flush=True)
    except Exception as e:
        write_json(run/"FAILURE.json",dict(status="BLOCKED",error_type=type(e).__name__,error=str(e),
                                          elapsed=time.perf_counter()-started))
        raise

if __name__=="__main__":
    main()
