"""Minimal FIRST_TEST run recorder; raw training artifacts remain in outputs/tabm_v21."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import pandas as pd

from .contracts import project_root
from .config import load_v21_config


def _json_value(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _environment_snapshot(python_executable: str) -> dict[str, Any]:
    snippet = ("import json,platform,sys,torch; "+
        "ok=torch.cuda.is_available(); "+
        "print(json.dumps({'python_version':platform.python_version(),'python_executable':sys.executable,"+
        "'conda_prefix':sys.prefix,'torch_version':torch.__version__,'cuda_available':ok,"+
        "'cuda_version':torch.version.cuda,'gpu_name':torch.cuda.get_device_name(0) if ok else None}))")
    result = subprocess.run([python_executable, "-c", snippet], cwd=project_root() / "src",
                            text=True, capture_output=True, check=False)
    if result.returncode:
        return {"status": "UNAVAILABLE", "stderr": result.stderr[-2000:],
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    info = json.loads(result.stdout)
    info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return info


def _write_record_files(record_dir: Path, record: dict[str, Any]) -> None:
    record_dir.mkdir(parents=True, exist_ok=True)
    (record_dir / "RUN_RECORD.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    raw_path = record.get("raw_run_dir") or "PENDING"
    (record_dir / "RAW_RUN_PATH.txt").write_text(str(raw_path) + "\n", encoding="utf-8")
    m = record.get("metrics") or {}
    t = record.get("telemetry") or {}
    lines = [
        f"# {record.get('run_id', '')}", "",
        f"RUN_ID: {record.get('run_id', '')}",
        f"PURPOSE: {record.get('purpose', '')}",
        f"CONFIG: {record.get('resolved_config', {})}",
        f"RAW_RUN_DIR: {raw_path}",
        f"STATUS: {record.get('status', '')}", "",
        "## RESULT", "",
        f"- Direction Raw: {m.get('raw_direction_accuracy', 'N/A')}",
        f"- Balanced: {m.get('balanced_accuracy', 'N/A')}",
        f"- +Recall: {m.get('positive_recall', 'N/A')}",
        f"- -Recall: {m.get('nonpositive_recall', 'N/A')}",
        f"- Magnitude MAE: {m.get('magnitude_mae', 'N/A')}",
        f"- Best / stop epoch: {t.get('best_epoch', 'N/A')} / {t.get('stop_epoch', 'N/A')}",
        f"- Wall time (s): {t.get('wall_time_total_seconds', 'N/A')}",
        f"- Parameters: {t.get('parameter_count_total', 'N/A')} total / {t.get('parameter_count_trainable', 'N/A')} trainable",
        "", "## OBSERVATION", "",
        "Single-target diagnostic only; this record is not a performance conclusion.",
        "", "## WARNING", "",
        ", ".join(record.get("warnings", [])) or "None recorded.",
        "", "## DECISION", "",
        str(record.get("decision", "DIAGNOSTIC_ONLY")),
        "", "## NEXT", "",
        str(record.get("next", "Continue only within the current FIRST_TEST gate.")), "",
    ]
    (record_dir / "RUN_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def _tracker_update(tracker_path: Path, run_id: str, *, status: str,
                    raw_run_dir: str | None = None, metrics: dict[str, Any] | None = None,
                    telemetry: dict[str, Any] | None = None, note: str = "") -> None:
    if not tracker_path.is_file():
        raise FileNotFoundError(f"FIRST_TEST tracker not found: {tracker_path}")
    metrics, telemetry = metrics or {}, telemetry or {}
    lines = tracker_path.read_text(encoding="utf-8").splitlines()
    matched = False
    for i, line in enumerate(lines):
        if not line.startswith(f"| {run_id} |"):
            continue
        cells = line.split("|")
        if len(cells) < 18:
            raise ValueError(f"unexpected FIRST_TEST tracker row shape for {run_id}")
        cells[7] = f" {status} "
        cells[8] = f" {metrics.get('raw_direction_accuracy', '')} "
        cells[9] = f" {metrics.get('balanced_accuracy', '')} "
        cells[10] = f" {metrics.get('positive_recall', '')} "
        cells[11] = f" {metrics.get('nonpositive_recall', '')} "
        cells[12] = f" {metrics.get('magnitude_mae', '')} "
        cells[13] = f" {telemetry.get('best_epoch', '')}/{telemetry.get('stop_epoch', '')} "
        cells[14] = f" {telemetry.get('wall_time_total_seconds', '')} "
        cells[15] = f" {raw_run_dir or ''} "
        if note:
            cells[16] = f" {note.replace('|', '/')} "
        lines[i] = "|".join(cells)
        matched = True
        break
    if not matched:
        raise KeyError(f"run id is missing from FIRST_TEST tracker: {run_id}")
    tracker_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_recorded_command(*, run_id: str, gate: str, target_day: str, purpose: str,
                         command: Sequence[str], config_path: Path,
                         first_test_root: Path | None = None) -> dict[str, Any]:
    """Update tracker before launch; preserve command/result and reference raw artifacts."""
    root = Path(first_test_root) if first_test_root else project_root() / "experiments" / "first_test"
    stage = "E0_pilot"
    record_dir = root / stage / "runs" / run_id
    tracker_path = root / "01_EXPERIMENT_TRACKER.md"
    command_text = subprocess.list2cmdline([str(x) for x in command])
    config_path = Path(config_path).resolve()
    environment = _environment_snapshot(str(command[0]))
    config = load_v21_config(config_path)
    if "--profile" in command:
        profile = str(command[list(command).index("--profile") + 1])
        config = config.with_profile(profile)
    if "--mode" in command:
        from dataclasses import replace
        config = replace(config, mode=str(command[list(command).index("--mode") + 1]))
    initial = {
        "run_id": run_id, "gate": gate, "date": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING", "purpose": purpose, "target_day_or_window": target_day,
        "command": command_text, "config_path": str(config_path),
        "config_sha256": config.config_sha256, "resolved_config": config.resolved_config(),
        "seed": config.seed, "raw_run_dir": None, "environment": environment,
        "decision": "DIAGNOSTIC_ONLY", "warnings": [],
    }
    _write_record_files(record_dir, initial)
    _tracker_update(tracker_path, run_id, status="RUNNING",
                    note=f"command={command_text}; config={config_path}")

    completed = subprocess.run(list(command), cwd=project_root() / "src", text=True,
                               capture_output=True, check=False)
    (record_dir / "COMMAND_STDOUT.txt").write_text(completed.stdout, encoding="utf-8")
    (record_dir / "COMMAND_STDERR.txt").write_text(completed.stderr, encoding="utf-8")
    result: dict[str, Any] = {}
    if completed.returncode == 0:
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError:
            completed = subprocess.CompletedProcess(command, 1, completed.stdout,
                "CLI output was not valid JSON; refusing to mark the run complete.")

    raw_dir = Path(result["run_dir"]) if completed.returncode == 0 and result.get("run_dir") else None
    manifest = _json_value(raw_dir / "manifest.json") if raw_dir else {}
    metrics = _json_value(raw_dir / "metrics.json") if raw_dir else {}
    telemetry_keys = ("wall_time_total_seconds", "training_epoch_seconds_total",
        "training_epoch_seconds_mean", "training_epoch_seconds_p50", "training_epoch_seconds_p95",
        "best_epoch", "stop_epoch", "epochs_run", "prediction_latency_ms", "parameter_count_total",
        "parameter_count_trainable", "checkpoint_size_bytes", "device", "cuda_peak_memory_bytes")
    telemetry = {key: manifest.get(key) for key in telemetry_keys}
    history_path = raw_dir / "training_history.parquet" if raw_dir else None
    history = pd.read_parquet(history_path).to_dict(orient="records") if history_path and history_path.is_file() else []
    warnings = sorted(set(metrics.get("diagnostic_warnings", [])) |
                      set(manifest.get("checkpoint_warnings", [])))
    status = "PASS" if completed.returncode == 0 and manifest.get("status") == "COMPLETE" else "FAIL"
    try:
        git_status = subprocess.run(["git", "status", "--short"], cwd=project_root(),
            text=True, capture_output=True, check=False).stdout
    except OSError:
        git_status = "UNAVAILABLE"
    record = {
        **initial, "status": status, "finished_at": datetime.now(timezone.utc).isoformat(),
        "return_code": completed.returncode, "failure_reason": completed.stderr if status != "PASS" else None,
        "raw_run_dir": str(raw_dir) if raw_dir else None,
        "raw_manifest_path": str(raw_dir / "manifest.json") if raw_dir else None,
        "raw_manifest_sha256": (hashlib.sha256((raw_dir / "manifest.json").read_bytes()).hexdigest()
                                 if raw_dir and (raw_dir / "manifest.json").is_file() else None),
        "config_sha256": manifest.get("config_sha256"),
        "resolved_config": manifest.get("resolved_config"),
        "selector_sha256": manifest.get("selector_sha256"),
        "source_sha256": manifest.get("source_sha256"),
        "sequence_manifest_sha256": manifest.get("sequence_manifest_sha256"),
        "seed": manifest.get("seed", initial["seed"]),
        "device": manifest.get("device"), "amp": manifest.get("amp"),
        "package_versions": manifest.get("package_versions"),
        "metrics": metrics, "telemetry": telemetry, "warnings": warnings,
        "training_history": history, "git_status_short": git_status,
        "decision": "DIAGNOSTIC_ONLY" if status == "PASS" else "INVALID",
    }
    _write_record_files(record_dir, record)
    _tracker_update(tracker_path, run_id, status=status,
        raw_run_dir=str(raw_dir) if raw_dir else None, metrics=metrics, telemetry=telemetry,
        note=("DIAGNOSTIC_ONLY; no ranking/promotion" if status == "PASS" else
              f"FAIL; {str(record['failure_reason']).splitlines()[-1] if record['failure_reason'] else 'no manifest'}"))
    return record
