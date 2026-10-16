"""Self-contained I0/I1 checks for the relocatable production runtime."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

TZ = ZoneInfo("Asia/Shanghai")
DA = "日前电价"
RT = "实时电价"
CORE_FORECASTS = ("直调负荷预测值", "地方电厂总加预测值", "联络线受电负荷预测值", "风电总加预测值", "光伏总加预测值")
CORE_ACTUALS = ("直调负荷实际值", "地方电厂总加实际值", "联络线受电负荷实际值", "风电总加实际值", "光伏总加实际值")
ALL_ACTUALS = (*CORE_ACTUALS, "全网负荷实际值", "核电实际值", "自备机组总加实际值", "试验机组总加实际值", "抽蓄实际值")
FORECASTS = CORE_FORECASTS


class QualificationError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _time(value):
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise QualificationError("capture timestamp missing")
    if ts.tzinfo is None:
        ts = ts.tz_localize(TZ)
    return ts.tz_convert(TZ).to_pydatetime()


def _load_values(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, low_memory=False)
    else:
        raise QualificationError("capture values must be parquet or csv")
    if not {"market_date", "period_no"}.issubset(frame.columns):
        raise QualificationError("capture schema missing market_date/period_no")
    frame = frame.copy()
    frame["market_date"] = pd.to_datetime(frame.market_date, errors="coerce").dt.normalize()
    frame["period_no"] = pd.to_numeric(frame.period_no, errors="coerce")
    if frame[["market_date", "period_no"]].isna().any().any():
        raise QualificationError("invalid date/period")
    frame["period_no"] = frame.period_no.astype(int)
    if frame.duplicated(["market_date", "period_no"]).any():
        raise QualificationError("duplicate day/period capture row")
    return frame


def _finite(frame, column, periods=None):
    if column not in frame:
        return 0
    x = frame if periods is None else frame[frame.period_no.isin(list(periods))]
    return int(np.isfinite(pd.to_numeric(x[column], errors="coerce").to_numpy(float)).sum())


def qualify_i0(manifest_path: Path, values_path: Path, *, evidence_mode: str = "promotion") -> dict:
    manifest_path, values_path = Path(manifest_path).resolve(), Path(values_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_protocol = {
        "promotion": "spread96_exact_origin_capture_v1",
        "synthetic_test": "spread96_synthetic_fixture_v1",
        "consumed_replay": "spread96_consumed_reference_projection_v1",
    }.get(evidence_mode)
    if expected_protocol is None or manifest.get("protocol") != expected_protocol:
        raise QualificationError("capture protocol does not match explicit evidence mode")
    before = {"manifest": sha256_file(manifest_path), "values": sha256_file(values_path)}
    values_sha = before["values"]
    declared = Path(str(manifest.get("values_path", "")))
    if not declared.is_absolute():
        declared = manifest_path.parent / declared
    if declared.resolve() != values_path or manifest.get("values_sha256") != values_sha:
        raise QualificationError("capture values path/SHA mismatch")
    target = pd.Timestamp(manifest.get("target_day")).normalize()
    if pd.isna(target):
        raise QualificationError("target day missing")
    d1 = target - pd.Timedelta(days=1)
    d2 = target - pd.Timedelta(days=2)
    origin = datetime.combine(d1.date(), time(14, 0), tzinfo=TZ)
    snapshot_at = None
    if evidence_mode == "promotion":
        snapshot_at = _time(manifest.get("snapshot_established_at") or manifest.get("captured_at"))
        if snapshot_at > origin:
            raise QualificationError("capture established after D-1 14:00")
        if (manifest.get("consistent_snapshot") is not True or manifest.get("db_mutation_statements") is not False or
                manifest.get("target_truth_selected") is not False or manifest.get("data_access_statement_types") != ["SELECT"]):
            raise QualificationError("capture read-only/snapshot provenance failed")
    elif evidence_mode == "synthetic_test":
        if manifest.get("synthetic_fixture") is not True or manifest.get("target_truth_selected") is not False:
            raise QualificationError("synthetic fixture provenance must be explicit and truth-free")
    elif (manifest.get("consumed_reference") is not True or manifest.get("target_truth_selected") is not False or
          manifest.get("target_truth_columns_projected_out") is not True):
        raise QualificationError("consumed replay projection must explicitly exclude target truth")
    if str(manifest.get("decision_day"))[:10] != d1.date().isoformat():
        raise QualificationError("decision-day mismatch")
    frame = _load_values(values_path)
    prior = frame[frame.market_date.eq(d1)].sort_values("period_no")
    target_frame = frame[frame.market_date.eq(target)].sort_values("period_no")
    for label, part in (("D-1", prior), ("D", target_frame)):
        if part.period_no.astype(int).tolist() != list(range(1, 97)):
            raise QualificationError(f"{label} period grid must be exactly p1..p96")
    checks = {
        "d1_da96": _finite(prior, DA) == 96,
        "d1_actual96": all(_finite(prior, c) == 96 for c in CORE_ACTUALS),
        "d1_forecast96": all(_finite(prior, c) == 96 for c in CORE_FORECASTS),
        "d1_rt_p1_p56": _finite(prior, RT, range(1, 57)) == 56,
        "target_da96": _finite(target_frame, DA) == 96,
        "target_forecast96": all(_finite(target_frame, c) == 96 for c in CORE_FORECASTS),
    }
    if any(not v for v in checks.values()):
        raise QualificationError("I0 required fields incomplete: " + str([k for k, v in checks.items() if not v]))
    target_truth = [RT, *ALL_ACTUALS]
    if any(c in target_frame and target_frame[c].notna().any() for c in target_truth):
        raise QualificationError("target RT/actual truth is present")
    raw_before = None
    raw_path = None
    has_raw_path = bool(manifest.get("raw_path"))
    has_raw_sha = bool(manifest.get("raw_sha256"))
    if has_raw_path != has_raw_sha:
        raise QualificationError("raw evidence path/SHA must be declared together")
    if evidence_mode == "promotion" and not (has_raw_path and has_raw_sha):
        raise QualificationError("promotion requires immutable raw evidence")
    if has_raw_path and has_raw_sha:
        raw_declared = Path(str(manifest["raw_path"]))
        if not raw_declared.is_absolute():
            raw_declared = manifest_path.parent / raw_declared
        raw_path = raw_declared.resolve()
        if not raw_path.is_file() or sha256_file(raw_path) != manifest["raw_sha256"]:
            raise QualificationError("raw evidence SHA mismatch")
        raw_before = sha256_file(raw_path)
    if (sha256_file(manifest_path) != before["manifest"] or sha256_file(values_path) != before["values"] or
            (raw_path is not None and sha256_file(raw_path) != raw_before)):
        raise QualificationError("capture changed while I0 qualification was running")
    return {
        "status": {"promotion": "PASS", "synthetic_test": "SYNTHETIC_TEST_ONLY",
                   "consumed_replay": "CONSUMED_REFERENCE_ONLY"}[evidence_mode],
        "target_day": target.date().isoformat(), "decision_day": d1.date().isoformat(),
        "label_cutoff": d2.date().isoformat(), "forecast_origin": origin.isoformat(),
        "manifest_sha256": sha256_file(manifest_path), "values_sha256": values_sha,
        "raw_sha256": None if raw_before is None else raw_before, "checks": checks, "target_truth_opened": False,
        "snapshot_established_at": None if snapshot_at is None else snapshot_at.isoformat(),
    }


def require_i1(ledger_path: Path | None, i0: dict) -> dict:
    if ledger_path is None:
        raise QualificationError("promotion requires matching 30-consecutive-day I1 PASS ledger")
    p = Path(ledger_path).resolve()
    ledger = json.loads(p.read_text(encoding="utf-8"))
    if ledger.get("status") != "I1_PASS" or int(ledger.get("longest_consecutive_pass_streak", 0)) < 30:
        raise QualificationError("I1 ledger is not 30-day PASS")
    rows = [r for r in ledger.get("rows", []) if r.get("target_day") == i0["target_day"] and r.get("passed") is True]
    if not rows:
        raise QualificationError("I1 ledger has no PASS row for target")
    row = rows[-1]
    if row.get("manifest_sha256") != i0["manifest_sha256"] or row.get("values_sha256") != i0["values_sha256"]:
        raise QualificationError("I1 row hashes do not bind exact capture")
    return {"path": str(p), "sha256": sha256_file(p), "row": row}

