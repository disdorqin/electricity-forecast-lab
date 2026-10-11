"""Guarded non-overwriting V2 run artifacts, model and prediction provenance."""
from datetime import datetime,timezone
from pathlib import Path
import csv
import hashlib
import importlib.metadata as metadata
import json
import subprocess
import sys
import uuid
import numpy as np
import torch
from .config import OUTPUT_ROOT,PACKAGE_ROOT,LAB_ROOT
from .contracts import assert_write_path,ContractError
from .feature_registry import registry
from .audit import file_hash,package_hashes

def jsonable(obj):
    if isinstance(obj,torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj,np.ndarray):
        return obj.tolist()
    if isinstance(obj,np.generic):
        return obj.item()
    if isinstance(obj,Path):
        return str(obj)
    raise TypeError(type(obj).__name__)

def write_json(path,obj):
    path = assert_write_path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,default=jsonable,allow_nan=False)

def write_csv(path,rows):
    path = assert_write_path(path)
    if not rows:
        raise ContractError("empty CSV artifact")
    with path.open("x",encoding="utf-8",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def new_run(root=None):
    base = assert_write_path(root or OUTPUT_ROOT/"runs")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"_"+uuid.uuid4().hex[:6]
    path = assert_write_path(base/stamp)
    path.mkdir(parents=True,exist_ok=False)
    return path

def to_numpy(obj):
    if isinstance(obj,torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj,dict):
        return {k:to_numpy(v) for k,v in obj.items()}
    return obj

def save_day(run_dir,sample,out,model,preprocessor,training,metrics,slots,timings):
    root = assert_write_path(run_dir/sample.target_day)
    root.mkdir(exist_ok=False)
    out = to_numpy(out)
    npz = assert_write_path(root/"DISTRIBUTIONS.npz")
    # Exclusive creation prevents a partial rerun overwriting saved scenarios.
    arrays = {k:out[k] for k in ("low_q_raw","high_q_raw","low_scenarios","high_scenarios",
        "spread_scenarios","spread_quantiles","spread_median","rt_quantiles","rt_median","p_positive",
        "fusion_low","fusion_high","clamp_fraction")}
    arrays["target_da"] = sample.da
    arrays.update({k:out["dependence"][k] for k in ("rho","phiL","phiH","coefficients")})
    with npz.open("xb") as f:
        np.savez_compressed(f,**arrays)
    rows = [dict(target_day=sample.target_day,slot=i+1,target_da=float(sample.da[i]),
                 spread_median=float(out["spread_median"][0,i]),rt_median=float(out["rt_median"][0,i]),
                 p_positive=float(out["p_positive"][0,i]),
                 **{f"spread_q{int(round(q*100)):02d}":float(out["spread_quantiles"][0,i,j])
                    for j,q in enumerate(__import__(__package__+".config",fromlist=["Q_KNOTS"]).Q_KNOTS)})
            for i in range(96)]
    write_csv(root/"PREDICTIONS.csv",rows)
    write_json(root/"SPECTRAL_GATES.json",out["spectral"])
    write_json(root/"FEATURE_REGISTRY.json",registry())
    write_json(root/"PREPROCESSOR.json",preprocessor.manifest())
    write_json(root/"TRAINING.json",training)
    write_json(root/"GRADIENT_AUDIT.json",training["gradient_audit"])
    write_json(root/"METRICS.json",metrics)
    write_csv(root/"SLOT_METRICS.csv",slots)
    write_json(root/"TIMINGS.json",timings)
    path = assert_write_path(root/"MODEL.pt")
    with path.open("xb") as f:
        torch.save(model.state_dict(),f)
    files = {p.name:file_hash(p) for p in root.iterdir() if p.is_file()}
    write_json(root/"DAY_MANIFEST.json",dict(target_day=sample.target_day,**sample.provenance,
        max_label_day_used=training["max_label_day"],training_days=training["training_days"],
        selected_epochs=training["selected_epochs"],profile=training["profile"],
        fresh_fit=True,previous_day_warm_start=False,artifacts=files))
    return root

def run_manifest(run_dir,cfg,store,dates,mode,profile,days,timings,command):
    code = package_hashes()
    packages = {}
    for name in ("torch","tabm","numpy","pandas","pyarrow","scikit-learn","pytest"):
        packages[name] = metadata.version(name)
    config = cfg.to_dict()
    return dict(status="COMPLETE",protocol="MSR_V2_CORE_DAILY_ASOF_V1",resolution="Native96",
        origin="D-1 14:00 Asia/Shanghai",label_maturity="D-2",mode=mode,profile=profile,
        smoke=mode=="smoke",retrospective_diagnostic=True,fresh_confirm_eligible=False,
        eligible_for_scientific_claim=False if (mode=="smoke" or profile=="fast-screen") else "requires independent review",
        source=store.source_identity,code_hashes=code,
        code_hash=hashlib.sha256(json.dumps(code,sort_keys=True).encode()).hexdigest(),
        config=config,config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
        seed=cfg.seed,python=sys.version,executable=sys.executable,packages=packages,
        git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=LAB_ROOT,text=True).strip(),
        target_dates=dates,days=days,timings=timings,command=command,
        research_assumptions=["full target DA96","full D-1 actual grid96","full target grid forecast96"],
        historical_vintage="LEGACY-UNVERIFIED / PROXY-EXPERIMENTAL",
        leakage_status="STRICT/PASS chronology under explicit docs84 assumptions, not historical live proof",
        production_write=False,hard_state=False,markov=False)
