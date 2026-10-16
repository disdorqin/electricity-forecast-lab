from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.research96.msr_spread_v2.production.api import ProductionAPI
from src.research96.msr_spread_v2.production.bundle_identity import verify_bundle
from src.research96.msr_spread_v2.production.qualification import QualificationError, qualify_i0, sha256_file
from src.research96.msr_spread_v2.production.history_store import load_mature_history
from src.research96.msr_spread_v2.production.runtime import (
    _file_map, _require_complete_training_panel, _stage_identity, _write_json, _write_npz, run_target,
)
from src.research96.msr_spread_v2.contracts import ContractError


TARGET = date(2024, 6, 15)
DA, RT = "日前电价", "实时电价"
ACTUALS = ("直调负荷实际值", "地方电厂总加实际值", "联络线受电负荷实际值", "风电总加实际值", "光伏总加实际值")
EXTRA_ACTUALS = ("全网负荷实际值", "核电实际值", "自备机组总加实际值", "试验机组总加实际值", "抽蓄实际值")
FORECASTS = ("直调负荷预测值", "地方电厂总加预测值", "联络线受电负荷预测值", "风电总加预测值", "光伏总加预测值")
ACTUAL_MAP = {
    "直调负荷实际": "直调负荷实际值", "全网负荷实际": "全网负荷实际值", "风电实际": "风电总加实际值",
    "光伏实际": "光伏总加实际值", "地方电厂出力实际": "地方电厂总加实际值", "外电实际": "联络线受电负荷实际值",
    "核电实际": "核电实际值", "自备电厂实际": "自备机组总加实际值", "试验机组实际": "试验机组总加实际值",
}
FORECAST_MAP = {
    "直调负荷预测": "直调负荷预测值", "全网负荷预测": "全网负荷预测值", "风电预测": "风电总加预测值",
    "光伏预测": "光伏总加预测值", "地方电厂出力预测": "地方电厂总加预测值", "外电预测": "联络线受电负荷预测值",
    "核电预测": "核电总加预测值", "自备电厂预测": "自备机组总加预测值", "试验机组预测": "试验机组总加预测值",
}


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _capture(tmp_path, *, d1_forecast=True, target_da=True, truth=False, evidence_mode="synthetic_test"):
    d1 = TARGET - timedelta(days=1)
    rows = []
    for day in (d1, TARGET):
        for period in range(1, 97):
            row = {"market_date": day.isoformat(), "period_no": period}
            row[DA] = float(period + (1000 if day == TARGET else 0)) if day != TARGET or target_da else np.nan
            row[RT] = float(period + 20) if day == d1 and period <= 56 else np.nan
            for c in set(ACTUALS + EXTRA_ACTUALS + FORECASTS):
                row[c] = np.nan
            if day == d1:
                for c in ACTUALS:
                    row[c] = float(period + 30)
                if d1_forecast:
                    for c in FORECASTS:
                        row[c] = float(period + 40)
            if day == TARGET:
                for c in FORECASTS:
                    row[c] = float(period + 50)
                if truth:
                    row[RT] = float(period + 60)
                    for c in ACTUALS + EXTRA_ACTUALS:
                        row[c] = float(period + 70)
            rows.append(row)
    values = tmp_path / "values.csv"
    pd.DataFrame(rows).to_csv(values, index=False)
    raw = tmp_path / "raw.bin"
    raw.write_bytes(b"synthetic immutable evidence")
    origin = pd.Timestamp(d1).tz_localize("Asia/Shanghai") + pd.Timedelta(hours=14)
    manifest = {
        "protocol": {"synthetic_test": "spread96_synthetic_fixture_v1",
                     "consumed_replay": "spread96_consumed_reference_projection_v1",
                     "promotion": "spread96_exact_origin_capture_v1"}[evidence_mode],
        "snapshot_kind": evidence_mode,
        "snapshot_id": "synthetic", "target_day": TARGET.isoformat(), "decision_day": d1.isoformat(),
        "captured_at": origin.isoformat(), "snapshot_established_at": origin.isoformat(),
        "consistent_snapshot": True, "db_mutation_statements": False, "target_truth_selected": False,
        "data_access_statement_types": ["SELECT"], "values_path": str(values.resolve()), "values_sha256": _sha(values),
        "raw_path": str(raw.resolve()), "raw_sha256": _sha(raw),
        "synthetic_fixture": evidence_mode == "synthetic_test",
        "consumed_reference": evidence_mode == "consumed_replay",
        "target_truth_columns_projected_out": evidence_mode == "consumed_replay",
    }
    mp = tmp_path / "snapshot_manifest.json"
    mp.write_text(json.dumps(manifest), encoding="utf-8")
    return mp, values


def _history(tmp_path):
    rows = []
    for lag in range(2, 9):
        day = TARGET - timedelta(days=lag)
        for p in range(1, 97):
            row = {"market_date": day.isoformat(), "period_no": p, DA: float(p + lag), RT: float(p + lag + 20)}
            for raw, col in ACTUAL_MAP.items():
                row[col] = float(p + lag + 100)
            for raw, col in FORECAST_MAP.items():
                row[col] = float(p + lag + 200)
            rows.append(row)
    path = tmp_path / "history.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_synthetic_cpu_prepare_and_promotion_gate(tmp_path, monkeypatch):
    from src.research96.msr_spread_v2 import dataset
    monkeypatch.setattr(dataset, "SOURCE_PATH", tmp_path / "absent_research_source.parquet")
    monkeypatch.setattr(dataset, "SOURCE_SHA256", "deliberately-not-the-research-sha")
    monkeypatch.setattr(dataset.CurveStore, "canonical", lambda *a, **k: pytest.fail("research canonical path used"))
    root = Path(__file__).resolve().parents[1]
    assert verify_bundle(root)["file_count"] > 0
    api = ProductionAPI(root)
    m, v = _capture(tmp_path)
    history_path = _history(tmp_path)
    history_store, _ = load_mature_history(history_path, target_day=TARGET.isoformat())
    assert np.isfinite(history_store.values[(TARGET - timedelta(days=2), "f_nuclear")]).all()
    prepared = api.prepare_target(target_day=TARGET.isoformat(), manifest_path=m, values_path=v,
                                 history_path=history_path, mode="synthetic_test")
    assert prepared["sample"].x.shape[0] == 96
    assert prepared["production_eligible"] is False
    assert prepared["bundle"].receipt["max_label_day"] == "2024-06-13"
    assert prepared["bundle"].receipt["target_truth_opened"] is False
    assert prepared["bundle"].receipt["evidence_status"] == "SYNTHETIC_TEST_ONLY"
    with pytest.raises(QualificationError, match="I1"):
        api.prepare_target(target_day=TARGET.isoformat(), manifest_path=m, values_path=v,
                           history_path=_history(tmp_path), mode="promotion")


def test_consumed_replay_is_explicitly_non_live(tmp_path):
    api = ProductionAPI(Path(__file__).resolve().parents[1])
    m, v = _capture(tmp_path, evidence_mode="consumed_replay")
    prepared = api.prepare_target(target_day=TARGET.isoformat(), manifest_path=m, values_path=v,
                                 history_path=_history(tmp_path), mode="consumed_replay")
    assert prepared["bundle"].receipt["evidence_status"] == "CONSUMED_REFERENCE_ONLY"
    assert prepared["production_eligible"] is False


@pytest.mark.parametrize("kwargs,pattern", [
    ({"d1_forecast": False}, "I0 required fields incomplete"),
    ({"target_da": False}, "I0 required fields incomplete"),
    ({"truth": True}, "target RT/actual truth"),
])
def test_fail_closed_production_input_requirements(tmp_path, kwargs, pattern):
    api = ProductionAPI(Path(__file__).resolve().parents[1])
    m, v = _capture(tmp_path, **kwargs)
    with pytest.raises(QualificationError, match=pattern):
        api.prepare_target(target_day=TARGET.isoformat(), manifest_path=m, values_path=v,
                           history_path=_history(tmp_path), mode="synthetic_test")


def test_short_path_relocation_import_and_manifest_verification():
    source = Path(__file__).resolve().parents[1]
    temp_base = Path(tempfile.gettempdir())
    with tempfile.TemporaryDirectory(prefix="s96rb_", dir=temp_base) as td:
        short = Path(td) / "bundle"
        shutil.copytree(source, short)
        code = ("from pathlib import Path; from src.research96.msr_spread_v2.production.bundle_identity "
                "import verify_bundle; from src.research96.msr_spread_v2.production.api import ProductionAPI; "
                "r=Path.cwd(); print(verify_bundle(r)['aggregate_sha256']); ProductionAPI(r)")
        result = subprocess.run([sys.executable, "-c", code], cwd=short, capture_output=True, text=True, check=True)
        assert len(result.stdout.strip()) == 64


def test_complete_training_calendar_is_fail_closed():
    cutoff = TARGET - timedelta(days=2)
    start = date(2023, 1, 1)
    days = [(start + timedelta(days=i)).isoformat() for i in range((cutoff - start).days + 1)]
    samples = [SimpleNamespace(target_day=d) for d in days]
    assert _require_complete_training_panel(samples, [], TARGET.isoformat()) == days
    with pytest.raises(ContractError, match="every day"):
        _require_complete_training_panel(samples[:-1], [], TARGET.isoformat())
    with pytest.raises(ContractError, match="every day"):
        _require_complete_training_panel(samples, [{"day": days[10], "reason": "missing"}], TARGET.isoformat())


def test_promotion_requires_raw_evidence(tmp_path):
    m, v = _capture(tmp_path, evidence_mode="promotion")
    payload = json.loads(m.read_text(encoding="utf-8"))
    payload.pop("raw_path")
    payload.pop("raw_sha256")
    m.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(QualificationError, match="raw evidence"):
        qualify_i0(m, v, evidence_mode="promotion")


def test_existing_target_reuse_is_exact_and_tamper_blocks(tmp_path):
    root = Path(__file__).resolve().parents[1]
    api = ProductionAPI(root)
    m, v = _capture(tmp_path, evidence_mode="consumed_replay")
    history_path = _history(tmp_path)
    prepared = api.prepare_target(target_day=TARGET.isoformat(), manifest_path=m, values_path=v,
                                  history_path=history_path, mode="consumed_replay")
    bundle = prepared["bundle"]
    output_root = tmp_path / "out"
    final = output_root / TARGET.isoformat()
    final.mkdir(parents=True)

    def make_stage(stage, stem, identity):
        folder = final / stage
        folder.mkdir()
        _write_npz(folder / f"{stem}.npz", {"fixture": np.arange(3, dtype=np.float64)})
        seal = {"schema": "spread96-production-stage-seal-v1", "stage": stage,
                "target_day": TARGET.isoformat(), "target_truth_opened": False,
                "arrays_sha256": sha256_file(folder / f"{stem}.npz"), "identity": identity,
                "sealed_utc": "2026-10-16T00:00:00+00:00"}
        _write_json(folder / f"{stem}_SEAL.json", seal)
        return sha256_file(folder / f"{stem}_SEAL.json")

    full_sha = make_stage("FULL", "FULL", _stage_identity(api, bundle, TARGET.isoformat(), "FULL", {"fixture": True}))
    dd_sha = make_stage("DIRECTION", "DIRECTION", _stage_identity(
        api, bundle, TARGET.isoformat(), "DIRECTION", {"full_seal_sha256": full_sha}))
    make_stage("F", "F", _stage_identity(api, bundle, TARGET.isoformat(), "F", {
        "full_seal_sha256": full_sha, "direction_seal_sha256": dd_sha,
        "history_artifact_hashes": {}, "f_history_start": "2026-01-01"}))
    _write_json(final / "ARTIFACT_MANIFEST.json", {"files": _file_map(final), "truth_scoring_performed": False})

    reused = run_target(api, target_day=TARGET.isoformat(), manifest_path=m, values_path=v,
                        history_path=history_path, output_root=output_root,
                        mode="consumed_replay", f_history_start="2026-01-01")
    assert reused["status"] == "REUSED" and reused["new_full_fit"] is False and reused["reused"] is True

    full_npz = final / "FULL/FULL.npz"
    full_npz.write_bytes(full_npz.read_bytes() + b"x")
    with pytest.raises(ContractError, match="hash drift"):
        run_target(api, target_day=TARGET.isoformat(), manifest_path=m, values_path=v,
                   history_path=history_path, output_root=output_root,
                   mode="consumed_replay", f_history_start="2026-01-01")


def test_clean_bundle_has_no_experiment_runtime_imports_or_forbidden_assets():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))
    paths = set(manifest["files"])
    assert not any("candidate_asl" in p or "experiments/" in p or "outputs/" in p for p in paths)
    for rel in paths:
        if rel.endswith(".py"):
            source = (root / rel).read_text(encoding="utf-8")
            assert ("experiments." + "production_qualification") not in source
            assert ("from " + "experiments") not in source and ("import " + "experiments") not in source
            if rel.startswith("src/research96/msr_spread_v2/production/"):
                assert "SOURCE_PATH" not in source
                assert "SOURCE_SHA256" not in source
                assert "CurveStore.canonical" not in source
    assert "src/research96/msr_spread_v2/t24_operator.npy" in paths

