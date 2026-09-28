"""Chronological three-task XGBoost + OOS TreeSHAP shadow selector."""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from .config import V2Config, default_selector_path, formal_output_dir
from .contracts import ContractError, TEMPORAL_FEATURES, project_root
from .dataset import SequenceStore
from .source_resolver import sha256_file


def within_hour_shadow(x:np.ndarray,hours:np.ndarray,rng:np.random.Generator)->np.ndarray:
    """Independent feature permutations only across days at a fixed hour."""
    x=np.asarray(x);hours=np.asarray(hours)
    if x.ndim!=2 or len(hours)!=len(x):raise ValueError("x/hours shape mismatch")
    shadow=np.empty_like(x)
    for h in np.unique(hours):
        idx=np.flatnonzero(hours==h)
        for j in range(x.shape[1]): shadow[idx,j]=x[rng.permutation(idx),j]
    return shadow


def chronological_folds(days: np.ndarray, cutoff: date | str) -> list[dict[str, str]]:
    cutoff_day = pd.Timestamp(cutoff).date()
    windows = [
        (date(2025, 7, 1), date(2025, 8, 31)),
        (date(2025, 9, 1), date(2025, 10, 31)),
        (date(2025, 11, 1), date(2025, 11, 30)),
        (date(2025, 12, 1), date(2025, 12, 31)),
    ]
    available = set(pd.Timestamp(d).date() for d in days)
    folds = []
    for start, stop in windows:
        stop = min(stop, cutoff_day)
        valid = [d for d in available if start <= d <= stop]
        if not valid:
            continue
        first = min(valid)
        latest_train = first - pd.Timedelta(days=2)
        train = [d for d in available if d <= latest_train]
        if len(train) < 30:
            raise ContractError(f"selector fold starting {first} has insufficient chronological training days")
        folds.append({"train_start": min(train).isoformat(), "train_end": max(train).isoformat(),
                      "valid_start": min(valid).isoformat(), "valid_end": max(valid).isoformat(),
                      "d2_gap_days": 1})
    if len(folds) != 4 or folds[-1]["valid_end"] != cutoff_day.isoformat():
        raise ContractError("fixed selector cutoff requires all four 2025 chronological OOS folds")
    return folds


def _positive_class_shap(values: Any, n_features: int) -> np.ndarray:
    if isinstance(values, list):
        values = values[-1]
    arr = np.asarray(values)
    if arr.ndim == 3:
        if arr.shape[-1] >= 2:
            arr = arr[..., 1]
        elif arr.shape[0] >= 2:
            arr = arr[1]
    if arr.ndim != 2 or arr.shape[1] != n_features:
        raise RuntimeError(f"unexpected official SHAP output shape {arr.shape}")
    return arr


def _oos_tree_shap(model, x: np.ndarray, *, seed: int, max_rows: int = 256) -> np.ndarray:
    if len(x) == 0:
        raise ValueError("OOS SHAP needs validation rows")
    rng = np.random.default_rng(seed)
    take = np.arange(len(x)) if len(x) <= max_rows else np.sort(rng.choice(len(x), max_rows, replace=False))
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x[take], check_additivity=False)
    return np.asarray(values)


def _fit_one_task(task: str, train_x: np.ndarray, valid_x: np.ndarray, train_y: np.ndarray, valid_y: np.ndarray,
                  feature_names: list[str], train_hours: np.ndarray, valid_hours: np.ndarray,
                  *, cfg: V2Config, fold_number: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed = cfg.seed + 997 * fold_number + {"XGB-DIR": 1, "XGB-MAG-ALL": 2}[task]
    rng = np.random.default_rng(seed)
    shadow_train = within_hour_shadow(train_x, train_hours,rng)
    shadow_valid = within_hour_shadow(valid_x, valid_hours,rng)
    x_train = np.concatenate([train_x, shadow_train], axis=1)
    x_valid = np.concatenate([valid_x, shadow_valid], axis=1)
    if task == "XGB-DIR":
        model = xgb.XGBClassifier(n_estimators=90, max_depth=4, learning_rate=0.05, subsample=1.0,
            colsample_bytree=0.8, reg_lambda=1.0, objective="binary:logistic", tree_method="hist",
            n_jobs=1, random_state=seed, eval_metric="logloss")
    else:
        model = xgb.XGBRegressor(n_estimators=90, max_depth=4, learning_rate=0.05, subsample=1.0,
            colsample_bytree=0.8, reg_lambda=1.0, objective="reg:squarederror", tree_method="hist",
            n_jobs=1, random_state=seed)
    model.fit(x_train, train_y)
    n = train_x.shape[1]
    shap_values = _positive_class_shap(_oos_tree_shap(model, x_valid, seed=seed, max_rows=256), 2 * n)
    importance = np.abs(shap_values).mean(axis=0)
    real, shadow = importance[:n], importance[n:]
    threshold = float(np.quantile(shadow, cfg.selector_q))
    hits = real > threshold
    rows = [{"task": task, "fold": fold_number, "feature_name": feature_names[j],
             "real_oos_tree_shap": float(real[j]), "shadow_threshold_q90": threshold,
             "shadow_importance": float(shadow[j]), "hit": bool(hits[j]),
             "oos_shap_rows": int(len(shap_values))} for j in range(n)]
    summary = {"task": task, "fold": fold_number, "train_rows": len(train_x),
               "validation_rows": len(valid_x), "oos_shap_rows": len(shap_values),
               "q": cfg.selector_q, "shadow_threshold": threshold,
               "hit_count": int(hits.sum()), "model": "xgboost_hist_cpu"}
    return rows, summary


def build_selector_manifest(store: SequenceStore | None = None, *, cutoff: date | str = "2025-12-31",
                            config: V2Config | None = None, output_dir: Path | None = None) -> tuple[dict[str, Any], Path, str]:
    cfg = config or V2Config()
    if pd.Timestamp(cutoff).date() != date(2025, 12, 31):
        raise ContractError("first deployable selector is fixed to 2025-12-31")
    store = store or SequenceStore.load()
    x_all, y_all, row_days, row_hours = store.candidate_selector_matrix(cutoff=cutoff)
    folds = chronological_folds(store.days, cutoff)
    feature_names = list(store.feature_names)
    registry = {r["feature_name"]: r for r in store.registry["candidate_features"]}
    rows: list[dict[str, Any]] = []
    fold_summaries = []
    for i, fold in enumerate(folds):
        train_day = pd.Timestamp(fold["train_end"]).date()
        valid_start, valid_end = pd.Timestamp(fold["valid_start"]).date(), pd.Timestamp(fold["valid_end"]).date()
        tr = np.asarray([d <= train_day for d in row_days])
        va = np.asarray([(valid_start <= d <= valid_end) for d in row_days])
        if not tr.any() or not va.any():
            raise ContractError(f"empty selector fold {i+1}")
        y_tr, y_va = y_all[tr], y_all[va]
        for task in ("XGB-DIR", "XGB-MAG-ALL"):
            if task == "XGB-DIR":
                train_mask, valid_mask, target_train, target_valid = np.ones(len(y_tr), bool), np.ones(len(y_va), bool), (y_tr > 0).astype(np.int8), (y_va > 0).astype(np.int8)
                if np.unique(target_train).size != 2:
                    raise ContractError(f"direction selector fold {i+1} has one class")
            else:
                train_mask, valid_mask = np.ones(len(y_tr),bool), np.ones(len(y_va),bool)
                target_train, target_valid = np.abs(y_tr), np.abs(y_va)
            if int(train_mask.sum()) < 30 or int(valid_mask.sum()) < 10:
                raise ContractError(f"{task} fold {i+1} has insufficient class-specific rows")
            target_scale = 1.0
            if task == "XGB-MAG-ALL":
                q25, q75 = np.quantile(y_tr, [0.25, 0.75])
                target_scale = max(float((q75 - q25) / 1.349), 1e-8)
                target_train = target_train / target_scale
                target_valid = target_valid / target_scale
            task_rows, summary = _fit_one_task(task, x_all[tr][train_mask], x_all[va][valid_mask],
                target_train, target_valid, feature_names, row_hours[tr][train_mask], row_hours[va][valid_mask],
                cfg=cfg, fold_number=i+1)
            summary["target_scale_from_training_fold"] = target_scale
            summary.update({"train_start": fold["train_start"], "train_end": fold["train_end"],
                            "valid_start": fold["valid_start"], "valid_end": fold["valid_end"]})
            rows.extend(task_rows); fold_summaries.append(summary)

    scores = pd.DataFrame(rows)
    hit_rates = scores.groupby(["task", "feature_name"], sort=False)["hit"].mean().unstack("task").fillna(0.0)
    selected = []
    feature_roles: list[dict[str, Any]] = []
    for idx, name in enumerate(feature_names):
        rates = {task: float(hit_rates.loc[name, task]) if name in hit_rates.index and task in hit_rates.columns else 0.0
                 for task in ("XGB-DIR", "XGB-MAG-ALL")}
        strong_tasks = [task for task, rate in rates.items() if rate >= cfg.strong_hit_rate]
        max_hit = max(rates.values(), default=0.0)
        core = bool(registry[name].get("business_core", False))
        temporal = name in TEMPORAL_FEATURES and name != "target_spread"
        if core:
            role = "Forced-Core"
        elif temporal:
            role = "Forced-Temporal"
        elif len(strong_tasks) > 1:
            role = "Strong-BOTH"
        elif strong_tasks:
            role = {"XGB-DIR": "Strong-DIR", "XGB-MAG-ALL": "Strong-MAG"}[strong_tasks[0]]
        elif max_hit >= cfg.weak_hit_rate:
            role = "Weak"
        else:
            role = "Noise"
        if role != "Noise":
            selected.append(name)
        feature_roles.append({"feature_name": name, "feature_index": idx, "role": role,
                              "task_hit_rates": rates, "strong_hit_tasks": strong_tasks,
                              "max_task_hit_rate": max_hit, "business_core": core,
                              "temporal_whitelist": temporal})
    if not selected:
        raise ContractError("selector selected no features")
    output_dir = Path(output_dir) if output_dir is not None else formal_output_dir() / "selector" / f"selector_cutoff_{pd.Timestamp(cutoff).date()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing frozen selector: {manifest_path}")
    selected_indices = [feature_names.index(name) for name in selected]
    manifest: dict[str, Any] = {
        "schema": "spread24_tabm_v2_selector_v1",
        "status": "FROZEN",
        "selector_cutoff": pd.Timestamp(cutoff).date().isoformat(),
        "selector_sha256": None,
        "seed": cfg.seed,
        "q": cfg.selector_q,
        "strong_hit_rate": cfg.strong_hit_rate,
        "weak_hit_rate": cfg.weak_hit_rate,
        "tasks": ["XGB-DIR", "XGB-MAG-ALL"],
        "folds": folds,
        "fold_model_summaries": fold_summaries,
        "selector_policy": "chronological OOS TreeSHAP shadow q90; forced business_core and temporal whitelist",
        "selected_features": selected,
        "selected_indices": selected_indices,
        "feature_roles": feature_roles,
        "feature_importance_asset": "feature_scores.parquet",
        "fold_scores_asset": "fold_scores.parquet",
        "candidate_feature_count": len(feature_names),
        "selected_feature_count": len(selected),
        "source_sha256": store.source_sha256,
        "sequence_manifest_sha256": store.sequence_sha256,
        "source_gate_status": store.source_gate["status"],
        "sequence_gate_status": store.sequence_gate["status"],
    }
    scores.to_parquet(output_dir / "feature_scores.parquet", index=False)
    pd.DataFrame(fold_summaries).to_parquet(output_dir / "fold_scores.parquet", index=False)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digest = sha256_file(manifest_path)
    (output_dir / "manifest.sha256").write_text(digest + "  manifest.json\n", encoding="ascii")
    manifest["selector_sha256"] = digest
    return manifest, manifest_path, digest


def load_selector_manifest(path: Path | None = None) -> tuple[dict[str, Any], str]:
    path = Path(path) if path is not None else default_selector_path()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    digest = sha256_file(path)
    sidecar = path.with_name("manifest.sha256")
    if not sidecar.exists() or sidecar.read_text(encoding="ascii").split()[0] != digest:
        raise ContractError("selector manifest sidecar SHA256 mismatch")
    if manifest.get("status") != "FROZEN":
        raise ContractError("selector manifest must have FROZEN status")
    manifest["selector_sha256"] = digest
    return manifest, digest
