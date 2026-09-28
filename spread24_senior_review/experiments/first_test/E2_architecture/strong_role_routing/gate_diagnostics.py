"""Checkpoint gate trajectories, component geometry and real-checkpoint gradients."""
import json,math,sys
from dataclasses import replace
from pathlib import Path
import numpy as np,pandas as pd,torch
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3];sys.path[:0]=[str(REPO),str(REPO/"src")]
from src.TafM_改进源码.config import load_v21_config,default_selector_path
from src.TafM_改进源码.dataset import SequenceStore,load_frozen_selector
from src.TafM_改进源码.preprocessing import fit_preprocessor
from src.TafM_改进源码.train import _model_for,_selected_arrays,_predict_one
from src.TafM_改进源码.losses import v21_loss
from src.TafM_改进源码.metrics import canonical_metrics
F80=HERE.parent/"fusion_coarse"/"runs"; C1=HERE.parent/"tabular_big_block"/"runs"
W=[("W1", "2026-02-12","2026-02-18"),("W2","2026-04-12","2026-04-18"),("W3","2026-06-12","2026-06-18"),("W4","2026-08-07","2026-08-13")]
def rec(a,d,benchmark=False):
 p=F80/f"E2B1-F80-{d}/RUN_RECORD.json" if a=="R0" else HERE/"runs"/("benchmark" if benchmark else "")/f"E2C3-{a}-{d}/RUN_RECORD.json"
 return json.loads(p.read_text(encoding="utf-8"))
def setup(day,store,selector,selhash,cfg):
 target=pd.Timestamp(day).date(); elig=store.eligibility(selector,all_target_day=target);ids=np.asarray(elig["eligible_indices"],dtype=np.int64);split=max(2,min(len(ids)-1,int(math.floor(.8*len(ids)))))
 from src.TafM_改进源码.preprocessing import fit_preprocessor
 state=fit_preprocessor(store,ids[:split],selector,selector_sha256=selhash,n_bins=cfg.ple_bins,ple_embedding_dim=cfg.ple_embedding_dim,future_clip_abs=cfg.future_clip_abs,temporal_clip_abs=cfg.temporal_clip_abs,clip_after_robust_scale=cfg.clip_after_robust_scale,ple_enabled=cfg.ple_enabled)
 i=int(np.flatnonzero(np.asarray([x==target for x in store.days]))[0]);xh,xf,ym,ys=_selected_arrays(store,[i],state);return target,state,xh,xf,ym,ys
def grad(loss,pars):
 ps=tuple(pars);gs=torch.autograd.grad(loss,ps,allow_unused=True,retain_graph=True);live=[g for g in gs if g is not None];return {"n_params":len(ps),"n_with_grad":sum(bool(torch.count_nonzero(g)) for g in live),"norm_l2":float(torch.sqrt(sum(g.detach().float().square().sum() for g in live))) if live else 0.}
def numeric_leaves(o,prefix=""):
 out={}
 if isinstance(o,dict):
  for k,v in o.items():out.update(numeric_leaves(v,f"{prefix}.{k}" if prefix else str(k)))
 elif isinstance(o,(int,float)) and not isinstance(o,bool) and math.isfinite(float(o)):out[prefix]=float(o)
 return out
def main():
 cfg=replace(load_v21_config(),mode="A2");store=SequenceStore.load();selector,selhash=load_frozen_selector(default_selector_path());rows=[];ownership={};exactness=[]
 days=[(str((pd.Timestamp(a)+pd.Timedelta(days=i)).date()),False) for _,a,b in W for i in range(7)]+[("2026-02-13",True)]
 for arm in ("R0","R1","R2"):
  for day,is_benchmark in days:
   r=rec(arm,day,benchmark=is_benchmark);target,state,xh,xf,ym,ys=setup(day,store,selector,selhash,cfg)
   mode={"R0":"all","R1":"all","R2":"drop_mag"}[arm]
   model=_model_for(store,selector,state,cfg,architecture_mode="full_all",direction_fusion_alpha=.8,direction_strong_role_routing_mode=mode)
   checkpoint=torch.load(Path(r["run_dir"])/"stage_a_best.pt",map_location="cpu",weights_only=False);model.load_state_dict(checkpoint["model"],strict=True);model.cuda().eval()
   with torch.no_grad():
    xft=torch.from_numpy(xf).cuda();hs,hw,g,hc=model.tabular_encoder.forward_components(xft); gb=torch.sigmoid(model.direction_global_gate_logit) if model.direction_global_gate_logit is not None else None
    gd=gb.expand(24) if gb is not None else (torch.full((24,),.8,device="cuda") if arm=="R1" else g)
    w=hw.unsqueeze(1).expand_as(hs);st=gd.view(1,1,24,1)*hs;wk=(1-gd).view(1,1,24,1)*w
    ns=float(st.norm());nw=float(wk.norm());cos=float(torch.nn.functional.cosine_similarity(hs.flatten(),w.flatten(),dim=0))
   try: manifest=json.loads((Path(r["run_dir"])/"manifest.json").read_text(encoding="utf-8"))
   except Exception: manifest={}
   trajectory=[z.get("gate") for z in manifest.get("direction_global_gate_trajectories",[]) if z.get("gate") is not None]
   rows.append({"arm":arm,"day":day,"scope":"benchmark_engineering_only" if is_benchmark else "formal_DEV","mode":mode,"gate_mean":float(gd.mean()),"gate_min":float(gd.min()),"gate_max":float(gd.max()),"gate_values":";".join(f"{v:.8f}" for v in gd.detach().cpu().tolist()),"global_gate":float(gb) if gb is not None else np.nan,"global_gate_init":.8 if trajectory else np.nan,"global_gate_final":trajectory[-1] if trajectory else np.nan,"global_gate_trajectory_min":min([.8]+trajectory) if trajectory else np.nan,"global_gate_trajectory_max":max([.8]+trajectory) if trajectory else np.nan,"global_gate_delta":trajectory[-1]-.8 if trajectory else np.nan,"strong_gated_norm":ns,"weak_gated_norm":nw,"effective_weak_norm_fraction":nw/(ns+nw) if ns+nw else np.nan,"strong_weak_cosine":cos,"checkpoint_epoch":checkpoint.get("epoch")})
   if is_benchmark:
    if arm=="R0":
     old=pd.read_parquet(Path(r["run_dir"])/"predictions.parquet");old_metrics=json.loads((Path(r["run_dir"])/"metrics.json").read_text(encoding="utf-8"))
     for check,alpha in (("all_all",.8),("default_no_flag",None)):
      em=model if alpha==.8 else _model_for(store,selector,state,cfg,architecture_mode="full_all",direction_fusion_alpha=None)
      if em is not model: em.load_state_dict(checkpoint["model"],strict=True);em.cuda().eval()
      pred=_predict_one(em,xh,xf,"cuda");new=pd.DataFrame({"target_day":target.isoformat(),"hour_business":np.arange(1,25),"y_true_model":ym.reshape(-1),"p_positive":pred["p"].reshape(-1),"direction_hat":pred["direction_hat"].reshape(-1).astype(np.int8),"magnitude_hat":pred["magnitude_hat"].reshape(-1),"signed_kpi_hat":pred["signed_kpi_hat"].reshape(-1),"alpha_dir":float(pred["alpha_dir"]),"alpha_mag":float(pred["alpha_mag"]),"direction_member_std":pred["direction_member_std"].reshape(-1),"direction_vote_fraction":pred["direction_vote_fraction"].reshape(-1),"direction_vote_entropy":pred["direction_vote_entropy"].reshape(-1),"magnitude_member_std":pred["magnitude_member_std"].reshape(-1),"magnitude_member_mean_scaled":pred["a_scaled_members"].mean(axis=1).reshape(-1)})
      cols=[c for c in old.columns if c in new.columns and pd.api.types.is_numeric_dtype(old[c]) and pd.api.types.is_numeric_dtype(new[c])];delta=float(np.max(np.abs(old[cols].to_numpy(dtype=float)-new[cols].to_numpy(dtype=float))))
      calc=canonical_metrics(ym.reshape(-1),pred["direction_hat"].reshape(-1),pred["magnitude_hat"].reshape(-1),direction_probability=pred["p"].reshape(-1),month=np.repeat(target.month,24),hour=np.arange(1,25));a,b=numeric_leaves(old_metrics),numeric_leaves(calc);shared=set(a)&set(b);dm=max((abs(a[k]-b[k]) for k in shared),default=0.)
      exactness.append({"check":check,"n_numeric_columns_compared":len(cols),"max_abs_delta":delta,"n_metrics_compared":len(shared),"max_abs_delta_metrics":dm,"within_1e-9":delta<=1e-9 and dm<=1e-9})
    xt=torch.from_numpy(xh).cuda();yt=torch.from_numpy(ys).cuda();out=model(xt,xft);loss=v21_loss(out,yt);tab=model.tabular_encoder
    groups={"strong":tuple(tab.backbone.parameters()),"weak":tuple(tab.weak_mlp.parameters()),"canonical_strong_role_routing":(tab.strong_role_routing_logits,),"temporal":tuple(model.temporal_encoder.parameters())}
    if gb is not None:groups["global_scalar"]=(model.direction_global_gate_logit,)
    ownership[arm]={"L_dir":{k:grad(loss["L_dir"],v) for k,v in groups.items()},"L_mag":{k:grad(loss["L_mag"],v) for k,v in groups.items()},"direction_strong_role_routing_mode":mode}
 pd.DataFrame(rows).to_csv(HERE/"gate_diagnostics.csv",index=False)
 (HERE/"gradient_ownership.json").write_text(json.dumps(ownership,indent=2),encoding="utf-8")
 (HERE/"benchmark/gate_a_exactness.json").write_text(json.dumps(exactness,indent=2),encoding="utf-8")
 # Human-readable runtime aggregates from formal and engineering records.
 records=[]
 for arm in ("R0","R1","R2"):
  for day,is_benchmark in days:
   r=rec(arm,day,benchmark=is_benchmark);m=r.get("manifest",{});records.append({"arm":arm,"target_day":day,"scope":"benchmark_engineering_only" if is_benchmark else "formal_DEV","run_wall_seconds":r.get("wall_seconds"),"training_wall_seconds":m.get("wall_time_total_seconds"),"epoch_seconds":m.get("training_epoch_seconds_mean"),"best_epoch":m.get("best_epoch"),"stop_epoch":m.get("stop_epoch"),"cuda_peak_memory_bytes":m.get("cuda_peak_memory_bytes"),"parameter_count":m.get("parameter_count_total"),"status":r.get("status")})
 pd.DataFrame(records).to_csv(HERE/"runtime.csv",index=False)
if __name__=="__main__":main()

