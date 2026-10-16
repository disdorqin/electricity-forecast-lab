"""Immutable mature-history projection; never uses the research parquet SHA gate."""
from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ..dataset import CurveStore
from ..feature_registry import GRID
from .qualification import QualificationError, sha256_file

DA = "日前电价"
RT = "实时电价"
ACTUAL = {
    "直调负荷实际": "直调负荷实际值", "全网负荷实际": "全网负荷实际值",
    "风电实际": "风电总加实际值", "光伏实际": "光伏总加实际值",
    "地方电厂出力实际": "地方电厂总加实际值", "外电实际": "联络线受电负荷实际值",
    "核电实际": "核电实际值", "自备电厂实际": "自备机组总加实际值", "试验机组实际": "试验机组总加实际值",
}
FORECAST = {
    "直调负荷预测": "直调负荷预测值", "全网负荷预测": "全网负荷预测值",
    "风电预测": "风电总加预测值", "光伏预测": "光伏总加预测值",
    "地方电厂出力预测": "地方电厂总加预测值", "外电预测": "联络线受电负荷预测值",
    "核电预测": "核电总加预测值", "自备电厂预测": "自备机组总加预测值", "试验机组预测": "试验机组总加预测值",
}


def _periods(frame: pd.DataFrame) -> pd.Series:
    if "period_no" in frame:
        p = pd.to_numeric(frame.period_no, errors="coerce")
    elif "时段" in frame:
        def one(value):
            text = str(value).strip()
            if text == "24:00":
                return 96
            try:
                hour, minute = (int(x) for x in text.split(":", 1))
            except Exception as exc:
                raise QualificationError(f"invalid historical period {value!r}") from exc
            mins = 1440 if hour == 0 and minute == 0 else hour * 60 + minute
            if mins < 15 or mins > 1440 or mins % 15:
                raise QualificationError(f"invalid historical period {value!r}")
            return mins // 15
        p = frame["时段"].map(one)
    else:
        raise QualificationError("history schema requires period_no or 时段")
    if p.isna().any() or not p.between(1, 96).all():
        raise QualificationError("history period outside p1..p96")
    return p.astype(int)


def load_mature_history(path: Path, *, target_day: str) -> tuple[CurveStore, dict]:
    """Load content-hash-bound history and hard reject any row later than D-2."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    before_sha = sha256_file(path)
    cutoff = pd.Timestamp(target_day).normalize() - pd.Timedelta(days=2)
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, low_memory=False)
    else:
        raise QualificationError("mature history must be parquet or csv")
    if "market_date" not in frame:
        raise QualificationError("history schema missing market_date")
    frame = frame.copy()
    frame["market_date"] = pd.to_datetime(frame.market_date, errors="coerce").dt.normalize()
    if frame.market_date.isna().any():
        raise QualificationError("invalid mature-history date")
    frame["period_no"] = _periods(frame)
    if frame.duplicated(["market_date", "period_no"]).any():
        raise QualificationError("duplicate mature-history day/period")
    if (frame.market_date > cutoff).any():
        raise QualificationError("mature history has a row after D-2")
    columns = {"da": DA, "rt": RT}
    for key, (_, actual, forecast) in GRID.items():
        columns[f"a_{key}"] = ACTUAL[actual]
        columns[f"f_{key}"] = FORECAST[forecast]
    columns["a_psh"] = "抽蓄实际值"
    values = {}
    for day, group in frame.groupby("market_date", sort=True):
        group = group.sort_values("period_no")
        slots = group.period_no.to_numpy(dtype=int)
        if len(slots) != len(set(slots)):
            raise QualificationError("duplicate mature-history periods")
        for field, column in columns.items():
            curve = np.full(96, np.nan, dtype=np.float64)
            if column in group:
                curve[slots - 1] = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=np.float64)
                values[(day.date(), field)] = curve
    digest = sha256_file(path)
    if digest != before_sha:
        raise QualificationError("mature history changed while loading")
    identity = {
        "kind": "PRODUCTION_MATURE_HISTORY_V1", "path": str(path), "sha256": digest,
        "max_allowed_date": cutoff.date().isoformat(),
        "observed_max_date": frame.market_date.max().date().isoformat() if len(frame) else None,
        "row_count": len(frame), "source_identity_is_research_sha": False,
    }
    store = CurveStore(values, identity)
    return store, identity

