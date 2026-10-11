"""Experiment-only floor-aware trainer. Reuses unchanged FULL graph/loss/stages."""
import copy
import time
import numpy as np
import torch
from ...contracts import assert_mature, ContractError
from ...trainer import Preprocessor, seed_everything, set_stage, gradient_audit
from .profile import BOUNDS, NAME, FloorSelector, scientific_config
from .arms import make_model, stage_mode, arm_objectives, arm_gradient_audit

def synchronized_time():
    torch.cuda.synchronize()
    return time.perf_counter()

def fit_day(samples, labels, target, batch, arm="FULL_V2"):
    if not torch.cuda.is_available():
        raise ContractError("qualified CUDA required; no CPU fallback")
    cfg = scientific_config(batch,arm)
    days = [s.target_day for s in samples]
    assert_mature(days, target)
    if days != sorted(set(days)) or len(days) <= cfg.validation_days:
        raise ContractError("unique chronological history and validation required")
    names = samples[0].names
    if any(s.names != names for s in samples):
        raise ContractError("mixed schemas")
    start = synchronized_time()
    raw = np.stack([s.x for s in samples])
    n, base_n = len(samples), len(samples)-cfg.validation_days
    pre = Preprocessor().fit(raw[:base_n], labels[:base_n], days[:base_n])
    selection_manifest = pre.manifest()
    seed_everything(cfg.seed)
    model = make_model(names, cfg, pre.stats,arm).cuda()
    x = torch.as_tensor(pre.transform(raw), device="cuda")
    y = torch.as_tensor(labels, dtype=torch.float32, device="cuda")
    timings = {"selection_setup": synchronized_time()-start}
    history, audits, selected = [], [], {}

    def stage_fit(m, stage, floor, epochs, indices, validation=None):
        phase = "selection" if validation is not None else "refit"
        stage_mode(m, stage)
        before = {k:p.detach().clone() for k,p in m.named_parameters()}
        params = [p for p in m.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr={"A":cfg.lr_a,"B":cfg.lr_b,"C":cfg.lr_c}[stage],
                                     weight_decay=cfg.weight_decay) if params else None
        selector = FloorSelector(floor, cfg.patience)
        best_state, best_guard = None, None
        begun = synchronized_time()
        for epoch in range(1, epochs+1):
            stage_mode(m, stage)
            train_loss = 0.
            norm = torch.tensor(0.)
            for offset in range(0, len(indices) if params else 0, batch):
                ids = indices[offset:offset+batch]
                optimizer.zero_grad(set_to_none=True)
                out = m(x[ids], components_only=stage=="A")
                loss = arm_objectives(out, y[ids], m)[{"A":"stage_a","B":"stage_b","C":"total"}[stage]]
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training objective")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip, error_if_nonfinite=True)
                optimizer.step()
                train_loss += float(loss.detach())*len(ids)
            metrics = {}
            if validation is not None:
                m.eval()
                with torch.no_grad():
                    for offset in range(0, len(validation), batch):
                        ids = validation[offset:offset+batch]
                        out = m(x[ids], components_only=stage=="A", scenarios=cfg.m_train)
                        for key, value in arm_objectives(out, y[ids], m).items():
                            metrics[key] = metrics.get(key, 0.) + float(value)*len(ids)
                metrics = {k:v/len(validation) for k,v in metrics.items()}
                if not all(np.isfinite(v) for v in metrics.values()):
                    raise FloatingPointError("nonfinite validation objective")
                guard = stage!="C" or best_guard is None or (
                    metrics["brier"] <= best_guard["brier"]+.01 and
                    metrics["point"] <= best_guard["point"]*1.05+1e-6)
                primary = metrics["stage_a" if stage=="A" else "crps"]
                if selector.consider(epoch, primary, guard):
                    best_state, best_guard = copy.deepcopy(m.state_dict()), metrics
            record = dict(phase=phase, stage=stage, epoch=epoch, train_loss=train_loss/len(indices) if params else None,
                          validation=metrics, last_batch_preclip_grad_norm=float(norm),
                          elapsed=synchronized_time()-begun,no_op_stage=not bool(params))
            history.append(record)
            print(f"{arm} {phase} {stage} epoch={epoch}/{epochs} loss={record['train_loss']} "
                  f"best={selector.epoch} elapsed={record['elapsed']:.2f}s", flush=True)
            if validation is not None and selector.stop:
                break
        if validation is not None:
            if best_state is None or not floor <= selector.epoch <= epochs:
                raise RuntimeError("no eligible floor-respecting checkpoint")
            m.load_state_dict(best_state)
        after = dict(m.named_parameters())
        changed = [k for k,v in before.items() if not torch.equal(v, after[k])]
        forbidden = [k for k in changed if (stage=="A" and k.startswith("dependence.")) or
                     (stage=="B" and not k.startswith("dependence."))]
        if forbidden:
            raise RuntimeError("stage freeze violation: "+str(forbidden))
        audits.append(dict(phase=phase, stage=stage, changed=changed, freeze_pass=True,
                           executed_epochs=epoch, selected_epoch=selector.epoch if validation is not None else epochs,
                           no_op_stage=not bool(params),
                           no_op_reason="No dependence parameters exist in DIRECT96/NO_DYNAMIC_DEP; no surrogate B updates" if not params else None))
        return selector.epoch if validation is not None else epochs

    train_ids, val_ids = np.arange(base_n), np.arange(base_n, n)
    for stage, floor, maximum in BOUNDS:
        start = synchronized_time()
        selected[stage] = stage_fit(model, stage, floor, maximum, train_ids, val_ids)
        timings["selection_"+stage] = synchronized_time()-start
    # Discard selected model; a genuinely fresh exact refit and fresh train-local transforms.
    del model, x
    torch.cuda.empty_cache()
    start = synchronized_time()
    pre = Preprocessor().fit(raw, labels, days)
    seed_everything(cfg.seed)
    model = make_model(names, cfg, pre.stats,arm).cuda()
    x = torch.as_tensor(pre.transform(raw), device="cuda")
    timings["refit_setup"] = synchronized_time()-start
    for stage, floor, _ in BOUNDS:
        start = synchronized_time()
        stage_fit(model, stage, floor, selected[stage], np.arange(n))
        timings["refit_"+stage] = synchronized_time()-start
    stage_mode(model, "C")
    model.eval()
    start = synchronized_time()
    ids = np.arange(min(n, batch))
    out = model(x[ids], scenarios=cfg.m_train)
    gradients = arm_gradient_audit(model, arm_objectives(out, y[ids], model))
    timings["gradient_audit"] = synchronized_time()-start
    return model, pre, dict(profile=NAME, selected_epochs=selected, history=history, stage_audits=audits,
        timings=timings, gradient_audit=gradients, training_days=days, max_label_day=max(days),
        inner_validation_days=days[-cfg.validation_days:], selection_fit_days=days[:base_n],
        selection_preprocessor=selection_manifest,
        fresh_fit=True, warm_start=False, model_class=type(model).__name__, arm=arm,
        loss="unchanged core objectives" if arm=="FULL_V2" else "docs86 explicitly ablated objectives only")
