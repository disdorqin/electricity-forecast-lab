"""Fresh per-day Stage A/B/C; chronological inner validation and one exact refit."""
import copy
import random
import time
import numpy as np
import torch
from .config import Config
from .contracts import assert_mature,ContractError
from .model import MSRSpreadV2
from .losses import objectives

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    # CPU baseline uses a bounded thread pool; 8 threads materially reduces full-history
    # daily refit wall time on the current 24-thread workstation while preserving
    # deterministic CPU reruns in the V2 integration suite.
    torch.set_num_threads(8)

def robust_scale(values,axis=None):
    q25,q75 = np.nanpercentile(values,[25,75],axis=axis)
    return np.maximum((q75-q25)/1.349,1e-4)

class Preprocessor:
    """Train-local numeric median/IQR with original availability masks preserved."""
    def fit(self,x,y,fit_days):
        if not np.isfinite(y).all():
            raise ContractError("incomplete training labels")
        self.median = np.nanmedian(x,axis=(0,1))
        if not np.isfinite(self.median).all():
            raise ContractError("feature has no train evidence for imputation; fail closed")
        self.scale = robust_scale(x,axis=(0,1))
        self.stats = dict(component_center=np.median(y[...,1:],axis=(0,1)).tolist(),
                          component_scale=robust_scale(y[...,1:],axis=(0,1)).tolist(),
                          spread_scale=float(robust_scale(y[...,0])))
        self.fit_days = list(fit_days)
        from .feature_registry import registry
        self.bypass_ids = [i for i,v in enumerate(registry().values()) if v["role"]=="bypass"]
        return self
    def transform(self,x):
        filled = np.where(np.isfinite(x),x,self.median)
        scaled = np.clip((filled-self.median)/self.scale,-10,10)
        scaled[...,self.bypass_ids] = x[...,self.bypass_ids]
        if not np.isfinite(scaled).all():
            raise ContractError("nonfinite preprocessing/mask")
        return scaled.astype(np.float32)
    def manifest(self):
        return dict(median=self.median.tolist(),scale=self.scale.tolist(),stats=self.stats,
                    fit_days=self.fit_days,clip=10,fit_scope="training only; inner val excluded for selection")

def set_stage(model,stage):
    if stage not in ("A","B","C"):
        raise ValueError("unknown training stage")
    for p in model.parameters():
        p.requires_grad_(stage!="B")
    if stage=="A":
        for p in model.dependence.parameters():
            p.requires_grad_(False)
    if stage=="B":
        for p in model.dependence.parameters():
            p.requires_grad_(True)
        model.eval() # Frozen branch dropout must not move marginals.
        model.dependence.train()
    else:
        model.train()

def gradient_audit(model,losses):
    """Norms by individual final/component loss; unused paths are explicit zeros."""
    groups = model.gradient_groups()
    params = [p for p in model.parameters() if p.requires_grad]
    positions = {id(p):i for i,p in enumerate(params)}
    vectors,norms = {},{}
    for term in ("crps","brier","point","final","low","high","dep_smooth","total"):
        grads = torch.autograd.grad(losses[term],params,retain_graph=True,allow_unused=True)
        vectors[term],norms[term] = {},{}
        for name,ps in groups.items():
            chunks = []
            for p in ps:
                i = positions.get(id(p))
                g = grads[i] if i is not None else None
                chunks.append(torch.zeros_like(p).flatten() if g is None else g.detach().flatten())
            v = torch.cat(chunks)
            vectors[term][name] = v
            norms[term][name] = float(v.double().norm())
    cosines = {}
    for name in groups:
        for term in ("low","high"):
            a,b = vectors["final"][name],vectors[term][name]
            denom = a.norm()*b.norm()
            cosines[name+"/"+term] = float(torch.dot(a,b)/denom) if denom>0 else None
    required = norms["final"]
    if not all(np.isfinite(v) and v>1e-12 for v in required.values()):
        raise RuntimeError("final gradient reachability blocked: "+str(required))
    return dict(norms=norms,final_vs_component_cosine=cosines,reachability="PASS",threshold=1e-12)

def _tensor(x,device):
    return torch.as_tensor(x,dtype=torch.float32,device=device)

def fit_day(samples,labels,target,cfg=None,device="cpu",profile="scientific"):
    cfg = cfg or Config()
    if profile not in ("scientific","no-refit-fast-dev","fast-screen"):
        raise ContractError("unknown training profile")
    days = [s.target_day for s in samples]
    assert_mature(days,target)
    if days != sorted(set(days)):
        raise ContractError("training days must be unique chronological")
    n = len(samples)
    if n<=cfg.validation_days:
        raise ContractError("insufficient days for 28-day inner validation plus fit")
    raw = np.stack([s.x for s in samples])
    names = samples[0].names
    if any(s.names!=names for s in samples):
        raise ContractError("mixed feature schemas")
    timings = {}
    history,stage_audits = [],[]
    fast_screen = profile=="fast-screen"
    base_n = n if fast_screen else n-cfg.validation_days
    selection_stats = Preprocessor().fit(raw[:base_n],labels[:base_n],days[:base_n])
    seed_everything(cfg.seed)
    start = time.perf_counter()
    model = MSRSpreadV2(names,cfg,selection_stats.stats).to(device)
    timings["model"] = time.perf_counter()-start
    x = _tensor(selection_stats.transform(raw),device); y = _tensor(labels,device)
    selected = {}
    def run_stage(m,stage,indices,epochs,validation=None,select=False):
        set_stage(m,stage)
        before = {name:p.detach().clone() for name,p in m.named_parameters()}
        lr = {"A":cfg.lr_a,"B":cfg.lr_b,"C":cfg.lr_c}[stage]
        optimizer = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                                     lr=lr,weight_decay=cfg.weight_decay)
        best,best_state,best_epoch,bad = float("inf"),None,0,0
        for epoch in range(1,epochs+1):
            set_stage(m,stage)
            epoch_loss = 0.
            for start_index in range(0,len(indices),cfg.batch_days):
                ids = indices[start_index:start_index+cfg.batch_days]
                optimizer.zero_grad(set_to_none=True)
                out = m(x[ids],components_only=stage=="A")
                losses = objectives(out,y[ids],m)
                objective = losses[{"A":"stage_a","B":"stage_b","C":"total"}[stage]]
                if not torch.isfinite(objective):
                    raise FloatingPointError("nonfinite stage objective")
                objective.backward()
                torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],cfg.grad_clip)
                optimizer.step()
                epoch_loss += float(objective.detach())*len(ids)
            metrics = {}
            if validation is not None:
                m.eval()
                totals = {}
                with torch.no_grad():
                    for vstart in range(0,len(validation),cfg.batch_days):
                        ids = validation[vstart:vstart+cfg.batch_days]
                        vo = m(x[ids],components_only=stage=="A",scenarios=cfg.m_train)
                        for key,value in objectives(vo,y[ids],m).items():
                            totals[key] = totals.get(key,0.)+float(value)*len(ids)
                metrics = {k:v/len(validation) for k,v in totals.items()}
                primary = metrics["stage_a" if stage=="A" else "crps"]
                # Stage C CRPS primary with non-worsening Brier/point guardrails.
                guard = stage!="C" or best_state is None or (
                    metrics["brier"]<=best_guard["brier"]+.01 and
                    metrics["point"]<=best_guard["point"]*1.05+1e-6)
                if primary<best and guard:
                    best,best_state,best_epoch,bad = primary,copy.deepcopy(m.state_dict()),epoch,0
                    best_guard = metrics
                else:
                    bad += 1
            history.append(dict(phase="selection" if select else "refit",stage=stage,epoch=epoch,
                                train_loss=epoch_loss/len(indices),validation=metrics))
            if select and bad>=cfg.patience:
                break
        if select:
            if best_state is None:
                raise RuntimeError("no finite validation checkpoint")
            m.load_state_dict(best_state)
        after = dict(m.named_parameters())
        changed = [k for k,v in before.items() if not torch.equal(v,after[k])]
        forbidden = [k for k in changed if (stage=="A" and k.startswith("dependence.")) or
                     (stage=="B" and not k.startswith("dependence."))]
        if forbidden:
            raise RuntimeError("freeze violation: "+str(forbidden))
        stage_audits.append(dict(phase="selection" if select else "refit",stage=stage,
                                changed=changed,freeze_pass=True))
        return best_epoch if select else epochs
    train_ids,val_ids = np.arange(base_n),np.arange(base_n,n)
    if fast_screen:
        # Fixed-budget DAILY_ASOF screening: use all mature rows, no validation tail,
        # no refit. It is for architecture triage only and never scientific claims.
        for stage,max_epochs in (("A",1),("B",1),("C",1)):
            start = time.perf_counter()
            selected[stage] = run_stage(model,stage,np.arange(n),max_epochs,None,False)
            timings["screen_"+stage] = time.perf_counter()-start
    else:
        for stage,max_epochs in (("A",cfg.epochs_a),("B",cfg.epochs_b),("C",cfg.epochs_c)):
            start = time.perf_counter()
            selected[stage] = run_stage(model,stage,train_ids,max_epochs,val_ids,True)
            timings["selection_"+stage] = time.perf_counter()-start
    final_preprocessor = selection_stats
    if profile=="scientific":
        final_preprocessor = Preprocessor().fit(raw,labels,days)
        seed_everything(cfg.seed)
        model = MSRSpreadV2(names,cfg,final_preprocessor.stats).to(device)
        x = _tensor(final_preprocessor.transform(raw),device)
        for stage in ("A","B","C"):
            start = time.perf_counter()
            run_stage(model,stage,np.arange(n),selected[stage])
            timings["refit_"+stage] = time.perf_counter()-start
    set_stage(model,"C"); model.eval()
    start = time.perf_counter()
    audit_ids = np.arange(min(n,cfg.batch_days))
    audit_out = model(x[audit_ids],scenarios=cfg.m_train)
    gradients = gradient_audit(model,objectives(audit_out,y[audit_ids],model))
    timings["gradient_audit"] = time.perf_counter()-start
    return model,final_preprocessor,dict(selected_epochs=selected,history=history,stage_audits=stage_audits,
        gradient_audit=gradients,timings=timings,max_label_day=max(days),training_days=days,
        inner_validation_days=[] if fast_screen else days[-cfg.validation_days:],profile=profile,
        fresh_fit=True,warm_start=False,selection_preprocessor=selection_stats.manifest())
