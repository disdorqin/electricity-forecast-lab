"""Production-qualified capture mapping onto frozen CurveStore field semantics."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..dataset import CurveStore
from ..feature_registry import GRID
from .history_store import ACTUAL, FORECAST
from .qualification import DA, RT, CORE_ACTUALS, CORE_FORECASTS, QualificationError, _load_values, qualify_i0, require_i1, sha256_file


@dataclass(frozen=True)
class Spread96InputBundle:
    target_day: str
    store: CurveStore
    receipt: dict


def _readonly(array):
    x = np.asarray(array, dtype=np.float64).copy()
    x.setflags(write=False)
    return x


def build_input_bundle(manifest_path: Path, values_path: Path, mature_history: CurveStore,
                       *, mode: str = "promotion", i1_ledger_path: Path | None = None,
                       bundle_manifest_sha256: str | None = None) -> Spread96InputBundle:
    if mode not in {"promotion", "synthetic_test", "consumed_replay"}:
        raise QualificationError("unknown bundle mode")
    if mode == "promotion" and i1_ledger_path is None:
        raise QualificationError("promotion requires matching 30-consecutive-day I1 PASS ledger")
    i0 = qualify_i0(manifest_path, values_path, evidence_mode=mode)
    i1 = require_i1(i1_ledger_path, i0) if mode == "promotion" else None
    target = pd.Timestamp(i0["target_day"]).date()
    d1 = target - timedelta(days=1)
    cutoff = target - timedelta(days=2)
    values = _load_values(Path(values_path))
    d1_frame = values[values.market_date.eq(pd.Timestamp(d1))].sort_values("period_no")
    target_frame = values[values.market_date.eq(pd.Timestamp(target))].sort_values("period_no")
    merged = {}
    for (day, field), raw in mature_history.values.items():
        date_key = pd.Timestamp(day).date()
        if date_key > cutoff:
            raise QualificationError("mature-history CurveStore contains a date after D-2")
        curve = np.asarray(raw, dtype=np.float64)
        if curve.shape != (96,):
            raise QualificationError("mature-history CurveStore curves must be [96]")
        merged[(date_key, field)] = _readonly(curve)

    def curve(frame, column, required=True, prefix=None):
        if column not in frame:
            if required:
                raise QualificationError(f"required capture field missing: {column}")
            return None
        x = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64).copy()
        if prefix is not None:
            if not np.isfinite(x[:prefix]).all():
                raise QualificationError(f"nonfinite D-1 p1..p{prefix}: {column}")
            x[prefix:] = np.nan
        elif required and not np.isfinite(x).all():
            raise QualificationError(f"nonfinite required capture curve: {column}")
        return x

    merged[(d1, "da")] = _readonly(curve(d1_frame, DA))
    merged[(d1, "rt")] = _readonly(curve(d1_frame, RT, prefix=56))
    merged[(target, "da")] = _readonly(curve(target_frame, DA))
    for key, (_, actual, forecast) in GRID.items():
        actual_col, forecast_col = ACTUAL[actual], FORECAST[forecast]
        ac = curve(d1_frame, actual_col, required=actual_col in CORE_ACTUALS)
        fc = curve(target_frame, forecast_col, required=forecast_col in CORE_FORECASTS)
        # D-1 forecasts are first-class input because frozen d1_error=actual-forecast.
        d1_fc = curve(d1_frame, forecast_col, required=forecast_col in CORE_FORECASTS)
        if ac is not None:
            merged[(d1, f"a_{key}")] = _readonly(ac)
        if fc is not None:
            merged[(target, f"f_{key}")] = _readonly(fc)
        if d1_fc is not None:
            merged[(d1, f"f_{key}")] = _readonly(d1_fc)
    psh_col = "抽蓄实际值"
    psh = curve(d1_frame, psh_col, required=False)
    if psh is not None:
        merged[(d1, "a_psh")] = _readonly(psh)

    if (target, "rt") in merged or any((target, f"a_{key}") in merged for key in GRID):
        raise QualificationError("target RT/actual may not enter model-facing CurveStore")
    if not np.isfinite(merged[(d1, "rt")][:56]).all() or np.isfinite(merged[(d1, "rt")][56:]).any():
        raise QualificationError("D-1 RT model context must be p1..p56 only")
    source_identity = {
        "kind": "PRODUCTION_SPREAD96_INPUT_BUNDLE_V1", "target_day": target.isoformat(),
        "manifest_sha256": i0["manifest_sha256"], "values_sha256": i0["values_sha256"],
        "raw_sha256": i0["raw_sha256"], "i1_ledger_sha256": None if i1 is None else i1["sha256"],
        "mature_history_sha256": mature_history.source_identity["sha256"],
        "mature_history_cutoff": cutoff.isoformat(), "bundle_manifest_sha256": bundle_manifest_sha256,
        "mode": mode,
    }
    map_digest = hashlib.sha256()
    for (day, key), arr in sorted(merged.items(), key=lambda x: (x[0][0], x[0][1])):
        map_digest.update(day.isoformat().encode() + b"\0" + key.encode() + b"\0")
        map_digest.update(np.asarray(arr, dtype="<f8").tobytes())
    manifest_obj = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    raw_path = None
    if i0["raw_sha256"] is not None:
        raw_declared = Path(str(manifest_obj.get("raw_path", "")))
        if not str(raw_declared):
            raise QualificationError("qualified raw evidence path disappeared during mapping")
        raw_path = raw_declared if raw_declared.is_absolute() else Path(manifest_path).resolve().parent / raw_declared
    if (sha256_file(Path(manifest_path)) != i0["manifest_sha256"] or
            sha256_file(Path(values_path)) != i0["values_sha256"] or
            (raw_path is not None and sha256_file(raw_path.resolve()) != i0["raw_sha256"])):
        raise QualificationError("input files changed during CurveStore mapping")
    receipt = {
        **source_identity, "mapping_sha256": map_digest.hexdigest(),
        "evidence_status": i0["status"],
        "production_eligible": mode == "promotion" and i1 is not None and bool(bundle_manifest_sha256),
        "target_truth_opened": False, "target_da_source": "C1 captured final DA96",
        "d1_actual_source": "C1 captured actual96", "d1_forecast_source": "C1 captured forecast96",
        "d1_rt_visible_prefix_end": 56, "max_label_day": cutoff.isoformat(),
        "i0_checks": i0["checks"], "i1": i1,
    }
    return Spread96InputBundle(target.isoformat(), CurveStore(merged, source_identity), receipt)

