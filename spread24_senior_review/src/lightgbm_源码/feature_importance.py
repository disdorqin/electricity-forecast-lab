"""FEATURE_IMPORTANCE_V1: observational importance analysis on strict rolling fits."""
from __future__ import annotations

import json
import hashlib
import platform
import sys
import subprocess
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from .data import load_frozen
from .evaluate import metrics, summarize
from .features import feature_columns, matrix
from .leakage import audit, training_last_day

EXPECTED = {"F0": 11, "F1": 12, "F2": 10, "F3": 11, "F4": 12,
            "F5": 82, "F6": 40, "F7": 24, "F8": 14, "F9": 28}
CORE = {"fcast_直调负荷", "fcast_竞价空间", "fcast_新能源总加", "fcast_风电总加",
        "fcast_光伏总加", "bidding_space_ratio", "renewable_minus_space",
        "residual_load_renew", "renewable_share", "ramp_wind"}


def _make_model(cfg):
    p = cfg["model"]
    return lgb.LGBMRegressor(objective=p["objective"], n_estimators=p["n_estimators"],
        learning_rate=p["learning_rate"], num_leaves=p["num_leaves"],
        min_child_samples=p["min_child_samples"], subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"], reg_lambda=p["reg_lambda"],
        random_state=cfg["seed"], verbosity=-1, n_jobs=-1)


def _norm(a):
    a = np.asarray(a, dtype=float)
    return a / a.sum() if a.sum() else np.zeros_like(a)


def run_feature_importance(root, config_path=None):
    root = Path(root)
    cfg = yaml.safe_load((Path(config_path) if config_path else root / "config.yaml").read_text(encoding="utf-8"))
    groups = json.loads((root / "data/frozen_repro/feature_groups.json").read_text(encoding="utf-8"))
    counts = {k: len(v) for k, v in groups.items()}
    if counts != EXPECTED or sum(counts.values()) != 244:
        raise AssertionError(f"FEATURE_COUNT mismatch: {counts}")
    df = load_frozen(root)
    cols = feature_columns(root, df)
    if len(cols) != 244 or set(cols) != {c for vs in groups.values() for c in vs}:
        raise AssertionError("frozen feature list does not resolve to exactly 244 columns")
    forbidden = [c for c in cols if "actual" in c.lower() or "target_day" in c.lower() or c in {"target_spread", "target_direction"}]
    if forbidden:
        raise AssertionError(f"forbidden features: {forbidden}")
    registry = json.loads((root / "data/frozen_repro/feature_registry.json").read_text(encoding="utf-8"))
    registry_by_feature = {z["feature"]: z for z in registry["features"]}
    allowed_availability = {"D-1 <=14:00", "D-2 or earlier", "known",
        "target D forecast known at D-1 14:00", "target D forecast known at origin",
        "target forecast + D-2 or earlier errors", "target forecast vs D-2 or earlier forecast history"}
    bad_availability = {f: registry_by_feature.get(f, {}).get("availability") for f in cols
        if f not in registry_by_feature or registry_by_feature[f].get("availability") not in allowed_availability}
    if bad_availability:
        raise AssertionError(f"feature registry availability audit failed: {bad_availability}")
    group_of = {c: g for g, names in groups.items() for c in names}
    x = matrix(df, cols)
    start, end = pd.Timestamp(cfg["eval_start"]).date(), pd.Timestamp(cfg["eval_end"]).date()
    days = sorted(d for d in df.target_day.unique() if start <= d <= end)
    outdir = root / "outputs/feature_importance_v1"
    outdir.mkdir(parents=True, exist_ok=True)
    native, shap_rows, predictions, pfi_rows, fitted = [], [], [], [], {}
    fit_count = 0
    for d in days:
        last = training_last_day(d)
        first = last - pd.Timedelta(days=cfg["training_days"] - 1)
        if last > d - pd.Timedelta(days=2):
            raise AssertionError(f"training cutoff violation for {d}")
        for period in cfg["periods"]:
            te = (df.target_day == d) & (df.period == period)
            tr = (df.target_day >= first) & (df.target_day <= last) & (df.period == period)
            if not te.any() or tr.sum() < 20:
                continue
            train_x = x.loc[tr]
            med = train_x.median()
            tx = train_x.fillna(med).fillna(0)
            vx = x.loc[te].reindex(columns=tx.columns).fillna(med).fillna(0)
            model = _make_model(cfg)
            model.fit(tx, df.loc[tr, "target_spread"])
            fitted[(str(d), period)] = (model, med)
            pred = model.predict(vx)
            fit_count += 1
            q = df.loc[te, ["target_day", "period", "hour_business", "target_spread"]].copy().reset_index(drop=True)
            q["predicted_spread"] = pred
            q["month"] = str(d)[:7]
            q["training_last_day"] = str(last)
            predictions.append(q)

            booster = model.booster_
            gain = booster.feature_importance(importance_type="gain")
            split = booster.feature_importance(importance_type="split")
            gs, ss = _norm(gain), _norm(split)
            gr = pd.Series(-gs).rank(method="min").astype(int).to_numpy()
            sr = pd.Series(-ss).rank(method="min").astype(int).to_numpy()
            for i, f in enumerate(cols):
                native.append({"target_day": str(d), "month": str(d)[:7], "period": period, "feature": f,
                    "group": group_of[f], "gain": gain[i], "gain_share": gs[i], "split": split[i],
                    "split_share": ss[i], "gain_rank": gr[i], "split_rank": sr[i]})

            # LightGBM pred_contrib returns per-row TreeSHAP plus the bias column.
            contrib = booster.predict(vx, pred_contrib=True)[:, :-1]
            for ri in range(len(vx)):
                for j, f in enumerate(cols):
                    shap_rows.append({"target_day": str(d), "month": str(d)[:7], "period": period,
                        "hour_business": int(q.loc[ri, "hour_business"]), "feature": f,
                        "shap_value": float(contrib[ri, j]), "abs_shap": float(abs(contrib[ri, j]))})

    if not predictions:
        raise RuntimeError("No rolling fits produced")
    pred = pd.concat(predictions, ignore_index=True)
    nat = pd.DataFrame(native)
    sh = pd.DataFrame(shap_rows)
    pred.to_csv(outdir / "predictions.csv", index=False)
    nat.to_csv(outdir / "native_importance_per_fit.csv", index=False)
    sh.to_csv(outdir / "oos_shap_per_row.csv", index=False)
    metrics_df = summarize(pred)
    metrics_df.to_csv(outdir / "metrics.csv", index=False)

    # Baseline is the repository's pre-existing pipeline output, never refit/tuned here.
    baseline_path = root / "outputs/predictions.csv"
    baseline_match = False
    baseline_detail = "baseline predictions.csv unavailable"
    if baseline_path.exists():
        base = pd.read_csv(baseline_path)
        keys = ["target_day", "period", "hour_business"]
        check = pred.copy(); check["target_day"] = check.target_day.astype(str)
        baseline_match = len(check) == len(base) and check[keys].astype(str).equals(base[keys].astype(str)) and np.allclose(check.predicted_spread, base.predicted_spread, rtol=1e-9, atol=1e-9)
        baseline_detail = f"rows={len(check)}/{len(base)}; predictions={'match' if baseline_match else 'differ'}"

    ns = nat.groupby("feature").agg(gain_share_mean=("gain_share", "mean"), gain_share_median=("gain_share", "median"), gain_share_std=("gain_share", "std"), split_share_mean=("split_share", "mean"), zero_gain_rate=("gain", lambda z: float((z == 0).mean())), top20_gain_rate=("gain_rank", lambda z: float((z <= 20).mean())), gain_rank_std=("gain_rank", "std"), gain_rank_iqr=("gain_rank", lambda z: float(z.quantile(.75)-z.quantile(.25))), nonzero_month_count=("month", lambda z: int(nat.loc[z.index].loc[nat.loc[z.index, "gain"] > 0, "month"].nunique()))).reset_index()
    sh["top20"] = sh.groupby(["target_day", "period"]).abs_shap.rank(method="min", ascending=False) <= 20
    shfit = sh.groupby(["target_day", "period", "feature"]).abs_shap.mean().reset_index()
    shfit["shap_rank"] = shfit.groupby(["target_day", "period"]).abs_shap.rank(method="min", ascending=False)
    shap_rank_stats = shfit.groupby("feature").shap_rank.agg(["std", lambda z: float(z.quantile(.75)-z.quantile(.25))]).rename(columns={"std":"shap_rank_std", "<lambda_0>":"shap_rank_iqr"}).reset_index()
    shs = sh.groupby("feature").agg(shap_mean_abs=("abs_shap", "mean"), median_abs_shap=("abs_shap", "median"), mean_signed_shap=("shap_value", "mean"), top20_shap_rate=("top20", "mean")).reset_index().merge(shap_rank_stats, on="feature")
    month_sh = sh.groupby(["month", "feature"]).abs_shap.mean().reset_index()
    month_std = month_sh.groupby("feature").abs_shap.std().rename("shap_month_std").reset_index()
    period_top = nat.assign(top20=nat.groupby(["target_day", "period"]).gain_share.rank(method="min", ascending=False).le(20)).groupby(["feature", "period"]).top20.any().groupby("feature").sum().rename("period_top20_count").reset_index()
    stability = ns.merge(shs, on="feature").merge(month_std, on="feature", how="left").merge(period_top, on="feature", how="left").fillna({"period_top20_count": 0})
    stability["overall_gain_rank"] = stability.gain_share_mean.rank(method="min", ascending=False).astype(int)
    stability["overall_shap_rank"] = stability.shap_mean_abs.rank(method="min", ascending=False).astype(int)
    monthly_ranks = nat.groupby(["month", "feature"]).gain_share.mean().groupby(level=0, group_keys=False).rank(ascending=False).rename("rank").reset_index()
    monthly_rank_stats = monthly_ranks.groupby("feature").rank.agg(["std", lambda z: float(z.quantile(.75)-z.quantile(.25))]).rename(columns={"std":"month_rank_std", "<lambda_0>":"month_rank_iqr"}).reset_index()
    period_ranks = nat.groupby(["period", "feature"]).gain_share.mean().groupby(level=0, group_keys=False).rank(ascending=False).rename("rank").reset_index()
    period_rank_stats = period_ranks.groupby("feature").rank.agg(["std", lambda z: float(z.quantile(.75)-z.quantile(.25))]).rename(columns={"std":"period_rank_std", "<lambda_0>":"period_rank_iqr"}).reset_index()
    stability = stability.merge(monthly_rank_stats, on="feature", how="left").merge(period_rank_stats, on="feature", how="left")
    stability = stability.assign(pfi_mae_delta_mean=np.nan, pfi_raw_delta_mean=np.nan)
    stability.to_csv(outdir / "feature_stability.csv", index=False)
    nat_summary = nat.groupby("feature").agg(gain_mean=("gain", "mean"), gain_share_mean=("gain_share", "mean"), split_mean=("split", "mean"), split_share_mean=("split_share", "mean"), gain_rank_mean=("gain_rank", "mean"), gain_rank_std=("gain_rank", "std")).reset_index()
    nat_summary.to_csv(outdir / "native_importance_summary.csv", index=False)
    shap_summary = sh.groupby("feature").agg(mean_abs_shap=("abs_shap", "mean"), median_abs_shap=("abs_shap", "median"), mean_signed_shap=("shap_value", "mean"), top20_shap_rate=("top20", "mean")).reset_index()
    shap_summary.to_csv(outdir / "oos_shap_summary.csv", index=False)
    # Required reporting slices.
    sh.groupby(["month", "period", "feature"]).agg(mean_abs_shap=("abs_shap", "mean"), median_abs_shap=("abs_shap", "median"), mean_signed_shap=("shap_value", "mean"), top20_shap_rate=("top20", "mean")).reset_index().to_csv(outdir / "oos_shap_by_month_period.csv", index=False)
    sh.groupby(["month", "feature"]).agg(mean_abs_shap=("abs_shap", "mean"), median_abs_shap=("abs_shap", "median"), mean_signed_shap=("shap_value", "mean"), top20_shap_rate=("top20", "mean")).reset_index().to_csv(outdir / "oos_shap_by_month.csv", index=False)
    sh.groupby(["period", "feature"]).agg(mean_abs_shap=("abs_shap", "mean"), median_abs_shap=("abs_shap", "median"), mean_signed_shap=("shap_value", "mean"), top20_shap_rate=("top20", "mean")).reset_index().to_csv(outdir / "oos_shap_by_period.csv", index=False)
    nat.groupby(["month", "period", "feature"]).agg(gain_share_mean=("gain_share", "mean"), split_share_mean=("split_share", "mean")).reset_index().to_csv(outdir / "native_by_month_period.csv", index=False)
    group_rows = []
    top_set = nat.loc[nat.gain_rank <= 20].groupby(["target_day", "period", "group"]).size().groupby("group").sum()
    for g, names in groups.items():
        group_share = nat[nat.group == g].groupby(["target_day", "period"]).gain_share.sum().mean()
        group_rows.append({"group": g, "group_gain_share": float(group_share),
            "group_shap_share": float(sh.loc[sh.feature.isin(names), "abs_shap"].sum() / sh.abs_shap.sum()),
            "group_top20_count": int(top_set.get(g, 0)),
            "group_feature_count": len(names), "importance_per_feature": float(group_share / len(names))})
    pd.DataFrame(group_rows).to_csv(outdir / "group_importance.csv", index=False)

    candidates = set(nat_summary.nlargest(40, "gain_share_mean").feature) | set(shap_summary.nlargest(40, "mean_abs_shap").feature)
    for p in cfg["periods"]:
        candidates |= set(nat[nat.period == p].groupby("feature").gain_share.mean().nlargest(20).index)
        candidates |= set(sh[(sh.period == p)].groupby("feature").abs_shap.mean().nlargest(20).index)
    # PFI is local to each OOS target-day/period block; 8-hour structure and day boundaries remain intact.
    rng = np.random.default_rng(cfg["seed"])
    for (d, period), q in pred.groupby(["target_day", "period"], sort=False):
        ix = (df.target_day == pd.Timestamp(d).date()) & (df.period == period)
        rows = df.loc[ix].reset_index(drop=True)
        base_x = x.loc[ix].copy(); med = x.loc[(df.target_day >= pd.Timestamp(q.training_last_day.iloc[0]).date()-pd.Timedelta(days=cfg["training_days"]-1)) & (df.target_day <= pd.Timestamp(q.training_last_day.iloc[0]).date()) & (df.period == period)].median()
        base_x = base_x.fillna(med).fillna(0)
        base_y = q.target_spread.to_numpy(); base_pred = q.predicted_spread.to_numpy(); bm = metrics(base_y, base_pred)
        m, _ = fitted[(str(d), period)]
        # Batch all perturbations for the block into one prediction call; avoids
        # hundreds of thousands of tiny LightGBM calls while preserving block boundaries.
        perturbed, tags = [], []
        for f in sorted(candidates):
            for rep in range(5):
                z=base_x.copy(); z[f]=rng.permutation(z[f].to_numpy()); perturbed.append(z.to_numpy()); tags.append((f,rep))
        batch=np.vstack(perturbed)
        batch_pred=m.booster_.predict(batch).reshape(len(tags),len(base_y))
        for (f,rep),yp in zip(tags,batch_pred):
            mm=metrics(base_y,yp)
            pfi_rows.append({"target_day":str(d),"month":str(d)[:7],"period":period,"feature":f,"repeat":rep,"delta_mae":mm["mae"]-bm["mae"],"delta_raw":mm["raw"]-bm["raw"],"delta_balanced":mm["balanced"]-bm["balanced"]})
    pfi = pd.DataFrame(pfi_rows)
    pfi.to_csv(outdir / "pfi_candidates.csv", index=False)
    pfi_summary = pfi.groupby("feature").agg(pfi_mae_delta_mean=("delta_mae","mean"),pfi_mae_delta_std=("delta_mae","std"),pfi_raw_delta_mean=("delta_raw","mean"),pfi_balanced_delta_mean=("delta_balanced","mean")).reset_index()
    pfi_summary.to_csv(outdir / "pfi_summary.csv", index=False)
    stability.drop(columns=[c for c in ["pfi_mae_delta_mean","pfi_raw_delta_mean"] if c in stability], errors="ignore").merge(pfi_summary,on="feature",how="left").to_csv(outdir/"feature_stability.csv",index=False)

    for name, frame, value in [("overall", nat_summary, "gain_share_mean"), ("shap", shap_summary, "mean_abs_shap")]:
        frame.nlargest(30,value).to_csv(outdir/f"top30_{name}.csv",index=False)
    nat.groupby(["month","feature"]).gain_share.mean().rename("gain_share").reset_index().sort_values(["month","gain_share"],ascending=[True,False]).groupby("month").head(30).to_csv(outdir/"top_features_by_month.csv",index=False)
    nat.groupby(["period","feature"]).gain_share.mean().rename("gain_share").reset_index().sort_values(["period","gain_share"],ascending=[True,False]).groupby("period").head(30).to_csv(outdir/"top_features_by_period.csv",index=False)
    nat_summary.nlargest(20,"gain_share_mean").to_csv(outdir/"top_features_overall.csv",index=False)

    st = pd.read_csv(outdir/"feature_stability.csv")
    pfs = pfi_summary.set_index("feature")
    month_gain = nat.groupby(["month","feature"]).gain_share.mean().rename("gain_share").reset_index()
    month_gain["rank"] = month_gain.groupby("month").gain_share.rank(method="min",ascending=False)
    period_gain = nat.groupby(["period","feature"]).gain_share.mean().rename("gain_share").reset_index()
    period_gain["rank"] = period_gain.groupby("period").gain_share.rank(method="min",ascending=False)
    period_top20_count = period_gain.loc[period_gain["rank"] <= 20].groupby("feature").period.nunique().to_dict()
    actions=[]
    for _,r in st.iterrows():
        f=r.feature; core=f in CORE
        stable=(r.overall_gain_rank<=30 or r.overall_shap_rank<=30 or (month_gain.loc[month_gain.feature==f,"rank"]<=30).sum()>=4 or period_top20_count.get(f,0)>=2 or (f in pfs.index and (pfs.loc[f,"pfi_mae_delta_mean"]>0 or pfs.loc[f,"pfi_raw_delta_mean"]>0)))
        bottom_shap=r.overall_shap_rank>183
        if core: action="KEEP_CORE"
        elif stable: action="KEEP_STABLE"
        elif r.zero_gain_rate>=.8 and bottom_shap and r.nonzero_month_count<=2 and period_top20_count.get(f,0)==0 and f in pfs.index and pfs.loc[f,"pfi_mae_delta_mean"]<=0 and pfs.loc[f,"pfi_raw_delta_mean"]<=0: action="DROP_CANDIDATE"
        else: action="REVIEW"
        actions.append({"feature":f,"group":group_of[f],"action":action,"business_core":core,"overall_gain_rank":r.overall_gain_rank,"overall_shap_rank":r.overall_shap_rank,"zero_gain_rate":r.zero_gain_rate,"nonzero_month_count":r.nonzero_month_count})
    pd.DataFrame(actions).to_csv(outdir/"feature_action_candidates.csv",index=False)

    audit(days,14,cols)
    leakage_ok=all(pd.Timestamp(row.training_last_day)<=pd.Timestamp(row.target_day)-pd.Timedelta(days=2) for row in pred.itertuples()) and not forbidden
    base_summary=summarize(base if False else pred)
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = None
    manifest={"phase":"FEATURE_IMPORTANCE_V1","status":"PASS" if baseline_match and leakage_ok else "FAIL","target":"DA-RT","forecast_origin":"D-1 14:00","training_days":180,"training_last_day_rule":"D-2 or earlier","d1_post14_realized_as_feature":False,"target_day_actual_as_feature":False,"target_day_DA_as_feature":False,"feature_counts":counts,"feature_count":244,"feature_source":"data/frozen_repro/feature_groups.json","feature_groups_sha256":sha(root/"data/frozen_repro/feature_groups.json"),"feature_registry_sha256":sha(root/"data/frozen_repro/feature_registry.json"),"slot_table_sha256":sha(root/"data/frozen_repro/slot_table.parquet"),"rolling_fits":fit_count,"baseline_match":baseline_match,"baseline_detail":baseline_detail,"leakage_status":"STRICT/PASS" if leakage_ok else "INVALID-LEAKAGE","selection_period":"2026-01-01..2026-08-14 development/selection only","confirmation_period":"not designated","final_holdout":"not designated; none opened","final_holdout_touched":False,"seed":cfg["seed"],"run_command":"python -c 'from pathlib import Path; from src.feature_importance import run_feature_importance; run_feature_importance(Path.cwd())'","git_commit":commit,"python":sys.version,"platform":platform.platform(),"pandas":pd.__version__,"numpy":np.__version__,"lightgbm":lgb.__version__,"model_config":cfg["model"],"data_rows":len(df),"prediction_rows":len(pred),"prediction_metrics":base_summary.to_dict(orient="records"),"notes":"Importance only; no feature removal or feature_groups.json edits. OOS TreeSHAP generated from per-fit test rows without labels; PFI block-local with 5 repeats."}
    (outdir/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    counts_action=pd.DataFrame(actions).action.value_counts().to_dict()
    (outdir/"README.md").write_text(f"# FEATURE_IMPORTANCE_V1\n\nStatus: {manifest['status']}\n\nThis is a strict rolling importance audit. The frozen F0-F9 feature list is unchanged; action labels are candidates only and do not remove features. Evaluation predictions and metrics are included for baseline comparison.\n\n- Rolling fits: {fit_count}\n- Feature count: 244\n- Baseline match: {baseline_match} ({baseline_detail})\n- Leakage audit: {manifest['leakage_status']}\n- Action counts: {counts_action}\n- PFI candidates: {len(candidates)}\n",encoding="utf-8")
    return manifest
