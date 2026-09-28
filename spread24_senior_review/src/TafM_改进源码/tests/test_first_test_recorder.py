import json
from pathlib import Path

from src.TafM_改进源码.config import V2Config
from src.TafM_改进源码.first_test_recorder import _tracker_update, _write_record_files


def test_default_profile_preserves_formal_budget_and_smoke_override():
    cfg = V2Config()
    default = cfg.with_profile("default")
    smoke = cfg.with_profile("smoke")
    assert (default.k, default.max_epochs, default.patience, default.batch_size) == (8, 120, 15, 64)
    assert (smoke.k, smoke.max_epochs, smoke.patience, smoke.batch_size) == (4, 2, 2, 32)


def test_first_test_recorder_writes_traceable_artifacts_and_tracker(tmp_path: Path):
    tracker = tmp_path / "01_EXPERIMENT_TRACKER.md"
    tracker.write_text(
        "| Run ID | Gate | Purpose | Target / Window | Variant | Seed | Status | Raw | Bal | +R | -R | Mag MAE | Best/Stop Epoch | Wall Time | Raw Run Dir | Decision / Notes |\n"
        "|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|---|---|\n"
        "| TEST | P0 | test | date | v | 1 | TODO | | | | | | | | | |\n",
        encoding="utf-8")
    record_dir = tmp_path / "E0_pilot" / "runs" / "TEST"
    record = {"run_id": "TEST", "status": "RUNNING", "purpose": "unit test",
              "raw_run_dir": None, "metrics": {}, "telemetry": {}, "warnings": []}
    _write_record_files(record_dir, record)
    assert {"RUN_RECORD.json", "RUN_SUMMARY.md", "RAW_RUN_PATH.txt"}.issubset(
        {p.name for p in record_dir.iterdir()})
    assert json.loads((record_dir / "RUN_RECORD.json").read_text(encoding="utf-8"))["status"] == "RUNNING"
    _tracker_update(tracker, "TEST", status="RUNNING", note="command=python ...; config=config.yaml")
    assert "| TEST | P0 | test | date | v | 1 | RUNNING |" in tracker.read_text(encoding="utf-8")


def test_cli_advertises_direction_experiment_default_profile():
    import subprocess
    import sys

    from src.TafM_改进源码.config import project_root
    result = subprocess.run([sys.executable, str(project_root() / "src" / "run_tabm_v21.py"),
        "direction-experiment", "--help"], cwd=project_root() / "src", text=True,
        capture_output=True, check=True)
    assert "smoke,default" in result.stdout
