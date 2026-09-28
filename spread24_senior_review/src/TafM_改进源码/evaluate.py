"""Bounded date-wise evaluation helpers for V2 engineering/DEV runs."""
from __future__ import annotations

from datetime import date, timedelta, datetime, timezone
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
import time

from .config import V2Config, config_provenance, formal_output_dir
from .contracts import TEMPORAL_FEATURES
from .dataset import SequenceStore
from .metrics import canonical_metrics
from .selector import load_selector_manifest
from .target_adapter import source_to_model_target
from .train import train_target_day


def _dates(start: str, end: str) -> list[date]:
    a, b = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    if a > b:
        raise ValueError("start must not be after end")
    return [a + timedelta(days=i) for i in range((b-a).days+1)]


def _effective_profile_config(config: V2Config | None, profile: str, *, mode: str = "A2") -> V2Config:
    """Resolve profile overrides once so run artifacts describe actual execution."""
    from dataclasses import replace
    return replace((config or V2Config(mode=mode)).with_profile(profile), mode=mode)


def evaluate_range(start: str, end: str, *, mode: str = "A2", profile: str = "smoke",
                   train_mode: str="stage_a",config: V2Config | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    cfg = _effective_profile_config(config, profile, mode=mode)
    store = SequenceStore.load()
    selector, _ = load_selector_manifest()
    days = _dates(start, end)
    available = set(store.days)
    rows, result_rows = [], []
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(output_dir) if output_dir else formal_output_dir() / "ablations" / f"{start}_{end}_{mode}_{profile}_{stamp}"
    root.mkdir(parents=True, exist_ok=False)
    for day in days:
        if day not in available:
            raise ValueError(f"target day absent from sequence assets: {day}")
        result = train_target_day(day, mode=mode, profile=profile,train_mode=train_mode, config=cfg, store=store,
                                  selector_path=Path(selector["_path"]) if "_path" in selector else None)
        rows.append(result["predictions"])
        result_rows.append({"target_day": day.isoformat(), "run_dir": result["run_dir"], **result["metrics"]})
        result_rows[-1]["parameter_count"] = result["manifest"]["parameter_count"]
        result_rows[-1]["training_time_seconds"] = float(result["training_history"].epoch_seconds.sum())
    predictions = pd.concat(rows, ignore_index=True)
    predictions.to_parquet(root / "predictions.parquet", index=False)
    metrics = canonical_metrics(predictions.y_true_model.to_numpy(),
        predictions.direction_hat.to_numpy().astype(bool), predictions.magnitude_hat.to_numpy(),
        direction_probability=predictions.p_positive.to_numpy(),
        month=predictions.target_day.map(lambda d: pd.Timestamp(d).month).to_numpy(),
        hour=predictions.hour_business.to_numpy())
    pd.DataFrame(result_rows).to_parquet(root / "daily_metrics.parquet", index=False)
    (root / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    manifest = {"range": [start, end], "mode": mode, "profile": profile,"train_mode":train_mode,
                "same_seed": cfg.seed, "seed": cfg.seed, "selector_sha256": selector.get("selector_sha256"),
                **config_provenance(cfg),
                "target_days": [d.isoformat() for d in days], "run_dirs": [r["run_dir"] for r in result_rows],
                "status": "COMPLETE"}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return {"output_dir": str(root), "metrics": metrics, "predictions": predictions,
            "daily_metrics": pd.DataFrame(result_rows)}


def run_ablation(start: str, end: str, *, modes: tuple[str, ...] = ("A0", "A1", "A2"),
                 profile: str = "smoke",train_mode:str="stage_a", config: V2Config | None = None) -> dict[str, Any]:
    cfg = _effective_profile_config(config, profile)
    summaries = {}
    for mode in modes:
        summaries[mode] = evaluate_range(start, end, mode=mode, profile=profile,train_mode=train_mode, config=cfg)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = formal_output_dir() / "ablations" / f"comparison_{start}_{end}_{profile}_{stamp}"
    root.mkdir(parents=True, exist_ok=False)
    comparison_rows = []
    for mode,res in summaries.items():
        daily=res["daily_metrics"]
        comparison_rows.append({"model":mode,**res["metrics"],
            "parameter_count":int(daily.parameter_count.iloc[0]),
            "parameter_count_status":"measured",
            "training_time_seconds":float(daily.training_time_seconds.sum())})
    store = SequenceStore.load()
    selector, selector_hash = load_selector_manifest()
    direct_models = run_direct24_baselines(store, _dates(start,end), selector, config=cfg, profile=profile)
    for baseline_name, direct in direct_models.items():
        direct_records=[]
        for item in direct["rows"]:
            for hour,(truth,pred) in enumerate(zip(item["actual"],item["prediction"]),start=1):
                direct_records.append({"target_day":item["target_day"],"hour_business":hour,
                                       "y_true_model":float(truth),"prediction_model":float(pred)})
        direct_frame=pd.DataFrame(direct_records)
        direct_frame.to_parquet(root/f"{baseline_name}_predictions.parquet",index=False)
        for day_state in direct["preprocessing_audits"]:
            audit_dir=root/"baseline_preprocessing"/baseline_name/day_state["target_day"]
            audit_dir.mkdir(parents=True,exist_ok=True)
            (audit_dir/"preprocessor_state.json").write_text(json.dumps(day_state["preprocessor_state"],indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
            (audit_dir/"numerical_audit.json").write_text(json.dumps(day_state["numerical_audit"],indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        comparison_rows.append({"model":baseline_name,**summarize_regression_prediction_frame(direct_frame),
            "parameter_count":direct["parameter_count"],"parameter_count_status":"measured",
            "training_time_seconds":direct["training_time_seconds"]})
    lgb_frame=run_strict_lightgbm_baseline(start,end)
    lgb_frame.to_parquet(root/"strict_lightgbm_predictions.parquet",index=False)
    comparison_rows.append({"model":"strict_LightGBM",**summarize_prediction_frame(lgb_frame),
        "parameter_count":"N/A","parameter_count_status":"not exposed by legacy fit_predict API",
        "training_time_seconds":lgb_frame.attrs.get("training_time_seconds")})
    table = pd.DataFrame(comparison_rows)
    table.to_csv(root / "comparison.csv", index=False)
    (root/"manifest.json").write_text(__import__("json").dumps({
        "range":[start,end],"profile":profile,"train_mode":train_mode,"modes":list(modes),"seed":cfg.seed,
        "selector_sha256":selector_hash,"baselines":["FutureOnly-Direct24","SameInput-Direct24","strict LightGBM canonical in-memory reevaluation"],
        "config_path":cfg.config_path,"config_sha256":cfg.config_sha256,"resolved_config":cfg.resolved_config(),
        "legacy_lightgbm_outputs_written":False,"status":"COMPLETE"},indent=2)+"\n",encoding="utf-8")
    return {"output_dir": str(root), "comparison": table, "runs": summaries}


def run_direct24_baselines(store: SequenceStore, target_days: list[date], selector: dict[str, Any], *,
                           config: V2Config | None = None, profile: str = "smoke",
                           window_days: int = 180) -> dict[str, dict[str, Any]]:
    """Train FutureOnly and SameInput signed-24 models with shared preprocessing contract."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from .config import V2Config
    from .preprocessing import fit_preprocessor, transform_future, transform_hist
    from .selector import load_selector_manifest
    from .models.direct24_baseline import FutureOnlyDirect24, SameInputDirect24

    cfg = (config or V2Config()).with_profile(profile)
    _, selector_sha256 = load_selector_manifest()
    names, cols, _ = store.selector_indices(selector)
    model_types = {"FutureOnly-Direct24": FutureOnlyDirect24, "SameInput-Direct24": SameInputDirect24}
    outputs = {key: {"rows": [], "preprocessing_audits": [], "training_time_seconds": 0.0}
               for key in model_types}
    for day in target_days:
        train_idx = store.eligibility(selector, current_target_day=day)["eligible_indices"]
        train_idx = [i for i in train_idx if store.days[i] >= day - timedelta(days=window_days)]
        test_idx = [i for i, d in enumerate(store.days) if d == day]
        if len(train_idx) < 30 or len(test_idx) != 1:
            raise ValueError(f"insufficient legal D-2 baseline train/test rows for {day}")
        test_eligible = set(store.eligibility(selector, requested_days=[day])["eligible_indices"])
        if test_idx[0] not in test_eligible:
            raise ValueError(f"selected-feature quarantine excludes baseline target day {day}")
        state = fit_preprocessor(store, train_idx, selector, selector_sha256=selector_sha256,
            n_bins=cfg.ple_bins, ple_embedding_dim=cfg.ple_embedding_dim,
            future_clip_abs=cfg.future_clip_abs, temporal_clip_abs=cfg.temporal_clip_abs,
            clip_after_robust_scale=cfg.clip_after_robust_scale, ple_enabled=cfg.ple_enabled)
        xh_train = transform_hist(np.asarray(store.x_hist[train_idx], dtype=np.float32), state)
        xf_train = transform_future(np.asarray(store.x_future[train_idx][:, :, cols], dtype=np.float32), state)
        y_train = source_to_model_target(np.asarray(store.y_source[train_idx], dtype=np.float32)) / state.target_scale_c
        xh_test = transform_hist(np.asarray(store.x_hist[test_idx], dtype=np.float32), state)
        xf_test = transform_future(np.asarray(store.x_future[test_idx][:, :, cols], dtype=np.float32), state)
        actual = source_to_model_target(np.asarray(store.y_source[test_idx[0]], dtype=np.float64))
        numerical_audit = {
            "fit_scope": state.fit_scope, "future_clip_abs": state.future_clip_abs,
            "temporal_clip_abs": state.temporal_clip_abs, "clip_after_robust_scale": state.clip_after_robust_scale,
            "future": {"preclip_abs_p99": state.future_preclip_abs_p99, "preclip_abs_p999": state.future_preclip_abs_p999,
                "preclip_abs_max": state.future_preclip_abs_max, "postclip_abs_max": state.future_postclip_abs_max,
                "clip_fraction": state.future_clip_fraction_fit, "features": state.future_feature_audit,
                "robust_scale": dict(zip(state.feature_names, state.future_scale)), "median": dict(zip(state.feature_names, state.future_median))},
            "temporal": {"preclip_abs_p99": state.temporal_preclip_abs_p99, "preclip_abs_p999": state.temporal_preclip_abs_p999,
                "preclip_abs_max": state.temporal_preclip_abs_max, "postclip_abs_max": state.temporal_postclip_abs_max,
                "clip_fraction": state.temporal_clip_fraction_fit, "features": state.temporal_channel_audit,
                "robust_scale": dict(zip(TEMPORAL_FEATURES, state.temporal_scale)),
                "median": dict(zip(TEMPORAL_FEATURES, state.temporal_median))},
        }
        for baseline_name, model_type in model_types.items():
            started = time.perf_counter()
            torch.manual_seed(cfg.seed)
            model = model_type(len(cols), hidden=64)
            tensors = (torch.from_numpy(xf_train), torch.from_numpy(y_train.astype(np.float32))) if baseline_name.startswith("FutureOnly") else (
                torch.from_numpy(xh_train), torch.from_numpy(xf_train), torch.from_numpy(y_train.astype(np.float32)))
            loader = DataLoader(TensorDataset(*tensors), batch_size=cfg.batch_size, shuffle=True,
                generator=torch.Generator().manual_seed(cfg.seed), num_workers=0)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            model.train()
            for _ in range(cfg.max_epochs):
                for batch in loader:
                    if baseline_name.startswith("FutureOnly"):
                        xb_future, yb = batch
                        prediction = model(xb_future)
                    else:
                        xb_hist, xb_future, yb = batch
                        prediction = model(xb_hist, xb_future)
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.l1_loss(prediction, yb)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite {baseline_name} baseline loss")
                    loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise FloatingPointError(f"non-finite {baseline_name} baseline gradient")
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            model.eval()
            with torch.no_grad():
                pred_scaled = model(torch.from_numpy(xf_test)) if baseline_name.startswith("FutureOnly") else model(
                    torch.from_numpy(xh_test), torch.from_numpy(xf_test))
                prediction = (pred_scaled.numpy()[0] * state.target_scale_c).astype(float)
            if not np.isfinite(prediction).all():
                raise FloatingPointError(f"non-finite {baseline_name} prediction")
            outputs[baseline_name]["rows"].append({"target_day": day.isoformat(), "actual": actual, "prediction": prediction})
            outputs[baseline_name]["training_time_seconds"] += time.perf_counter() - started
            outputs[baseline_name]["preprocessing_audits"].append({"target_day": day.isoformat(),
                "preprocessor_state": state.to_dict(), "numerical_audit": numerical_audit})
            outputs[baseline_name]["parameter_count"] = sum(p.numel() for p in model.parameters())
            outputs[baseline_name]["method"] = f"{baseline_name}; selected X_future, {'X_hist + ' if baseline_name.startswith('SameInput') else ''}D-2; BASE_TRAIN-only shared preprocessing/clipping"
    return outputs


def run_strict_lightgbm_baseline(start: str, end: str, *, window_days: int = 180) -> pd.DataFrame:
    """In-memory strict re-evaluation of the unchanged historical LightGBM code.

    Does not invoke the legacy pipeline or write/replace its outputs. The label
    cutoff is D-2 and source predictions are converted DA-RT -> RT-DA for KPI.
    """
    timer=time.perf_counter()
    import yaml
    from ..lightgbm_源码.data import load_frozen
    from ..lightgbm_源码.features import feature_columns, matrix
    from ..lightgbm_源码.model import fit_predict
    from .contracts import project_root
    root = project_root()
    cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    frame = load_frozen(root)
    columns = feature_columns(root, frame)
    x = matrix(frame, columns)
    records = []
    for day in _dates(start, end):
        latest = day - timedelta(days=2)
        earliest = latest - timedelta(days=window_days-1)
        for period in cfg["periods"]:
            test_mask = (frame.target_day == day) & (frame.period == period)
            train_mask = (frame.target_day >= earliest) & (frame.target_day <= latest) & (frame.period == period)
            if not test_mask.any() or train_mask.sum() < 20:
                continue
            train_x = x.loc[train_mask].fillna(x.loc[train_mask].median()).fillna(0.0)
            test_x = x.loc[test_mask].reindex(columns=train_x.columns).fillna(train_x.median()).fillna(0.0)
            pred_da_rt = fit_predict(train_x, frame.loc[train_mask, "target_spread"], test_x, cfg)
            subset = frame.loc[test_mask, ["target_day", "hour_business", "target_spread"]].copy()
            subset["y_true_model"] = -subset.pop("target_spread").astype(float)
            subset["prediction_model"] = -np.asarray(pred_da_rt, dtype=float)
            subset["baseline"] = "strict_LightGBM"
            records.append(subset)
    if not records:
        raise ValueError("strict LightGBM returned no bounded DEV predictions")
    output=pd.concat(records, ignore_index=True)
    output.attrs["training_time_seconds"]=time.perf_counter()-timer
    return output


def summarize_prediction_frame(frame: pd.DataFrame) -> dict[str, Any]:
    actual=frame.y_true_model.to_numpy(dtype=float); pred=frame.prediction_model.to_numpy(dtype=float)
    return summarize_regression_prediction_frame(frame)


def summarize_regression_prediction_frame(frame: pd.DataFrame) -> dict[str, Any]:
    actual=frame.y_true_model.to_numpy(dtype=float); pred=frame.prediction_model.to_numpy(dtype=float)
    metrics=canonical_metrics(actual, pred > 0, np.abs(pred), allow_missing_probability=True,
        month=frame.target_day.map(lambda d: pd.Timestamp(d).month).to_numpy(),
        hour=frame.hour_business.to_numpy())
    from .metrics import _binary_auc
    metrics["rank_auc"] = _binary_auc(actual > 0, pred)
    metrics["auc"] = "N/A"
    metrics["auc_status"] = "NOT_A_PROBABILITY_METRIC"
    metrics["brier"] = "N/A"
    metrics["signed_mae"] = float(np.mean(np.abs(pred - actual)))
    return metrics
