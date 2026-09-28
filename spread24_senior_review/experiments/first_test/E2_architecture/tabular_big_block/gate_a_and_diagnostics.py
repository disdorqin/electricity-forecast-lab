"""Artifact-based exactness, gradient ownership, and selected-checkpoint component diagnostics."""
from __future__ import annotations
import json, math, sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent; REPO=HERE.parents[3]
sys.path.insert(0,str(REPO)); sys.path.insert(0,str(REPO/"src"))
from src.TafM_改进源码.config import default_selector_path,load_v21_config
from src.TafM_改进源码.dataset import SequenceStore,load_frozen_selector
from src.TafM_改进源码.preprocessing import fit_preprocessor
from src.TafM_改进源码.train import _model_for,_selected_arrays,_predict_one
from src.TafM_改进源码.losses import v21_loss
from src.TafM_改进源码.metrics import canonical_metrics

F80=REPO/"experiments/first_test/E2_architecture/fusion_coarse/runs"
RUNS=HERE/"runs"

def record(arm,day):
    if arm=="C0": p=F80/f"E2B1-F80-{day}/RUN_RECORD.json"
    else: p=RUNS/"benchmark"/f"E2C1-{arm}-{day}/RUN_RECORD.json"
    return json.loads(p.read_text(encoding="utf-8"))

def prepared(target,store,selector,selector_hash,cfg):
    eligibility=store.eligibility(selector,current_target_day=target)
    train=np.asarray(eligibility["eligible_indices"],dtype=np.int64)
    split=max(2,min(len(train)-1,int(math.floor(.8*len(train)))))
    state=fit_preprocessor(store,train[:split],selector,selector_sha256=selector_hash,
        n_bins=cfg.ple_bins,ple_embedding_dim=cfg.ple_embedding_dim,
        future_clip_abs=cfg.future_clip_abs,temporal_clip_abs=cfg.temporal_clip_abs,
        clip_after_robust_scale=cfg.clip_after_robust_scale,ple_enabled=cfg.ple_enabled)
    index=int(np.flatnonzero(np.asarray([d==target for d in store.days]))[0])
    xh,xf,ym,ys=_selected_arrays(store,[index],state)
    return state,index,xh,xf,ym,ys

def load_checkpoint(rec,store,selector,state,cfg,*,route="current",fixed=True,device="cuda"):
    model=_model_for(store,selector,state,cfg,architecture_mode="full_current",
        direction_fusion_alpha=.8 if fixed else None,direction_tabular_mode=route)
    payload=torch.load(Path(rec["run_dir"])/"stage_a_best.pt",map_location="cpu",weights_only=False)
    model.load_state_dict(payload["model"],strict=True); model.to(device); model.eval()
    return model,payload

def compare_frame(a,b):
    common=[c for c in a.columns if c in b.columns and pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])]
    result={"n_numeric_columns_compared":len(common),"max_abs_delta":0.0}
    if len(a)!=len(b): result["max_abs_delta"]=float("inf")
    elif common: result["max_abs_delta"]=float(np.max(np.abs(a[common].to_numpy(dtype=float)-b[common].to_numpy(dtype=float))))
    return result

def numeric_leaves(obj,prefix=""):
    out={}
    if isinstance(obj,dict):
        for k,v in obj.items(): out.update(numeric_leaves(v,f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj,(int,float)) and not isinstance(obj,bool) and math.isfinite(float(obj)):
        out[prefix]=float(obj)
    return out

def grads(loss,params):
    p=tuple(params); g=torch.autograd.grad(loss,p,allow_unused=True,retain_graph=True)
    live=[x for x in g if x is not None]
    return {"n_params":len(p),"n_with_grad":sum(bool(torch.count_nonzero(x)) for x in live),
        "norm_l2":float(torch.sqrt(sum(x.detach().float().square().sum() for x in live))) if live else 0.0,
        "abs_sum":float(sum(x.detach().abs().sum() for x in live)) if live else 0.0}

def ownership(model,xh,xf,ys):
    device=next(model.parameters()).device
    out=model(torch.from_numpy(xh).to(device),torch.from_numpy(xf).to(device)); loss=v21_loss(out,torch.from_numpy(ys).to(device))
    tab=model.tabular_encoder
    groups={"strong_tabm":tuple(tab.backbone.parameters()),"weak_mlp":tuple(tab.weak_mlp.parameters()),
        "horizon_gate":(tab.horizon_gate_logits,),"temporal":tuple(model.temporal_encoder.parameters())}
    report={"direction_tabular_mode":model.direction_tabular_mode,"direction_fusion_alpha":.8,
        "L_dir":{k:grads(loss["L_dir"],v) for k,v in groups.items()},
        "L_mag":{k:grads(loss["L_mag"],v) for k,v in groups.items()}}
    for key in ("strong_tabm","weak_mlp","horizon_gate"):
        report["L_dir"][key]["live"] = report["L_dir"][key]["n_with_grad"]>0
        report["L_mag"][key]["live"] = report["L_mag"][key]["n_with_grad"]>0
    report["L_dir"]["temporal"]["live"]=report["L_dir"]["temporal"]["n_with_grad"]>0
    return report,out

def main():
    outdir=HERE; (outdir/"benchmark").mkdir(exist_ok=True)
    cfg=replace(load_v21_config(),mode="A2"); store=SequenceStore.load()
    selector,selector_hash=load_frozen_selector(default_selector_path())
    target=pd.Timestamp("2026-02-13").date(); state,index,xh,xf,ym,ys=prepared(target,store,selector,selector_hash,cfg)
    c0=record("C0","2026-02-13"); archived=Path(c0["run_dir"])
    old=pd.read_parquet(archived/"predictions.parquet"); old_metrics=json.loads((archived/"metrics.json").read_text(encoding="utf-8"))
    exactness=[]; loaded={}
    for name,fixed in (("current_fixed_0.8",True),("default_no_flag",False)):
        model,_=load_checkpoint(c0,store,selector,state,cfg,route="current",fixed=fixed)
        prediction=_predict_one(model,xh,xf,"cuda")
        frame=pd.DataFrame({"target_day":target.isoformat(),"hour_business":np.arange(1,25),
            "y_true_model":ym.reshape(-1),"p_positive":prediction["p"].reshape(-1),
            "direction_hat":prediction["direction_hat"].reshape(-1).astype(np.int8),
            "magnitude_hat":prediction["magnitude_hat"].reshape(-1),
            "signed_kpi_hat":prediction["signed_kpi_hat"].reshape(-1),
            "alpha_dir":float(prediction["alpha_dir"]),"alpha_mag":float(prediction["alpha_mag"]),
            "direction_member_std":prediction["direction_member_std"].reshape(-1),
            "direction_vote_fraction":prediction["direction_vote_fraction"].reshape(-1),
            "direction_vote_entropy":prediction["direction_vote_entropy"].reshape(-1),
            "magnitude_member_std":prediction["magnitude_member_std"].reshape(-1),
            "magnitude_member_mean_scaled":prediction["a_scaled_members"].mean(axis=1).reshape(-1)})
        delta=compare_frame(old,frame)
        metrics=canonical_metrics(ym.reshape(-1),prediction["direction_hat"].reshape(-1),prediction["magnitude_hat"].reshape(-1),
            direction_probability=prediction["p"].reshape(-1),month=np.repeat(target.month,24),hour=np.arange(1,25))
        old_leaf=numeric_leaves(old_metrics);new_leaf=numeric_leaves(metrics)
        shared=sorted(set(old_leaf)&set(new_leaf))
        delta_m=max((abs(old_leaf[k]-new_leaf[k]) for k in shared),default=0.0)
        exactness.append({"check":name,"artifact_run_dir":str(archived),**delta,
            "n_metrics_compared":len(shared),"max_abs_delta_metrics":delta_m,
            "within_1e-9":delta["max_abs_delta"]<=1e-9 and delta_m<=1e-9})
        loaded[name]=model
    (outdir/"benchmark/gate_a_exactness.json").write_text(json.dumps(exactness,ensure_ascii=False,indent=2),encoding="utf-8")

    # Trained benchmark checkpoints prove real autograd ownership. Compare components separately
    # from Magnitude's always-canonical route.
    ownership_rows=[]; ownership_data={}
    for arm in ("C0","C1","C2"):
        rec=record(arm,"2026-02-13");route={"C0":"current","C1":"strong_only","C2":"weak_only"}[arm]
        model,payload=load_checkpoint(rec,store,selector,state,cfg,route=route,fixed=True)
        report,_=ownership(model,xh,xf,ys); report.update({"arm":arm,"target_day":str(target),"checkpoint_epoch":payload["epoch"]})
        ownership_data[arm]=report; ownership_rows.append(report)
    checks={
      "C1_Ldir_reaches_strong_tabm":ownership_data["C1"]["L_dir"]["strong_tabm"]["live"],
      "C1_Ldir_blocked_from_weak_mlp":not ownership_data["C1"]["L_dir"]["weak_mlp"]["live"],
      "C1_Ldir_blocked_from_horizon_gate":not ownership_data["C1"]["L_dir"]["horizon_gate"]["live"],
      "C2_Ldir_reaches_weak_mlp":ownership_data["C2"]["L_dir"]["weak_mlp"]["live"],
      "C2_Ldir_blocked_from_strong_tabm":not ownership_data["C2"]["L_dir"]["strong_tabm"]["live"],
      "C2_Ldir_blocked_from_horizon_gate":not ownership_data["C2"]["L_dir"]["horizon_gate"]["live"],
      "C1_C2_temporal_Ldir_live":all(ownership_data[a]["L_dir"]["temporal"]["live"] for a in ("C1","C2")),
      "magnitude_reaches_current_strong_weak_gate_all":all(all(ownership_data[a]["L_mag"][g]["live"] for g in ("strong_tabm","weak_mlp","horizon_gate")) for a in ("C0","C1","C2")),
      "fixed_alpha_is_0.8":all(abs(ownership_data[a]["direction_fusion_alpha"]-.8)<1e-12 for a in ownership_data),
    }
    for row in ownership_rows: row["checks"]={k:v for k,v in checks.items() if k.startswith(row["arm"])}
    (outdir/"gradient_ownership.json").write_text(json.dumps({"benchmark_day":str(target),"arms":ownership_data,"checks":checks,"pass":all(checks.values())},ensure_ascii=False,indent=2),encoding="utf-8")

    # C0 selected-checkpoint gate and component norms for every formal F80 day.
    gate_rows=[]; component_rows=[]
    f80_paths=sorted((F80).glob("E2B1-F80-*/RUN_RECORD.json"))
    formal=[]
    for path in f80_paths:
        rec0=json.loads(path.read_text(encoding="utf-8"))
        if rec0.get("status")=="PASS" and rec0.get("window") in {"W1","W2","W3","W4"}: formal.append(rec0)
    if len(formal)!=28: raise RuntimeError(f"expected 28 reusable C0 F80 runs, found {len(formal)}")
    for rec0 in formal:
        d=pd.Timestamp(rec0["target_day"]).date(); st,_,dxh,dxf,_,_=prepared(d,store,selector,selector_hash,cfg)
        model,payload=load_checkpoint(rec0,store,selector,st,cfg,route="current",fixed=True)
        with torch.no_grad():
            device=next(model.parameters()).device
            hs,hw,g,hc=model.tabular_encoder.forward_components(torch.from_numpy(dxf).to(device))
            gb=g.view(1,1,24,1); cs=gb*hs; cw=(1-gb)*hw.unsqueeze(1).expand(-1,cfg.k,-1,-1)
            ns=float(torch.linalg.vector_norm(cs)); nw=float(torch.linalg.vector_norm(cw))
            cosine=float(torch.nn.functional.cosine_similarity(cs.reshape(1,-1),cw.reshape(1,-1),dim=1).mean())
            normcur=float(torch.linalg.vector_norm(hc))
        gv=g.detach().cpu().numpy()
        for h,value in enumerate(gv,1): gate_rows.append({"arm":"C0","run_id":rec0["run_id"],"target_day":rec0["target_day"],"hour_business":h,"gate_init":.8,"gate_value":float(value),"delta_from_init":float(value-.8)})
        row={"arm":"C0","run_id":rec0["run_id"],"target_day":rec0["target_day"],"checkpoint_epoch":payload["epoch"],
            "gate_mean":float(gv.mean()),"gate_min":float(gv.min()),"gate_max":float(gv.max()),
            "gate_delta_mean":float((gv-.8).mean()),"gate_delta_min":float((gv-.8).min()),"gate_delta_max":float((gv-.8).max()),
            "strong_component_norm":ns,"weak_component_norm":nw,"effective_weak_norm_fraction":nw/(ns+nw+1e-12),
            "cosine_strong_weak":cosine,"h_current_norm":normcur}
        for label,sl in (("H1",slice(0,8)),("H2",slice(8,16)),("H3",slice(16,24))):
            row[f"{label}_gate_mean"]=float(gv[sl].mean());row[f"{label}_gate_delta_mean"]=float((gv[sl]-.8).mean())
            row[f"{label}_gate_min"]=float(gv[sl].min());row[f"{label}_gate_max"]=float(gv[sl].max())
        component_rows.append(row)
    pd.DataFrame(gate_rows).to_csv(outdir/"gate_values_by_run.csv",index=False)
    pd.DataFrame(component_rows).to_csv(outdir/"tabular_component_diagnostics.csv",index=False)
    roles=selector["feature_roles"]
    role_map={r["feature_name"]:r["role"] for r in roles}
    role_names={role:[name for name in state.feature_names if role_map.get(name)==role] for role in sorted(set(role_map.values()))}
    (outdir/"benchmark/selector_role_inventory.json").write_text(json.dumps({k:{"count":len(v),"features":v} for k,v in role_names.items()},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"gate_a":exactness,"gradient_checks":checks,"formal_c0_runs":len(formal),"gate_rows":len(gate_rows)},ensure_ascii=False,indent=2))
    return 0 if all(x["within_1e-9"] for x in exactness) and all(checks.values()) else 2

if __name__=="__main__": raise SystemExit(main())
