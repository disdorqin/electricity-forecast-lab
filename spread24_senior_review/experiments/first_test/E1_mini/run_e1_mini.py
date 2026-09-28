"""E1-mini orchestration: M0 / M1 / M2 x 28 target days.

Experiment-only driver. It never imports or mutates the frozen V2.1 training path:
every run is a fresh `run_tabm_v21.py direction-experiment` subprocess, and raw
model artifacts stay in outputs/tabm_v21/experiments/direction_first/**.

Record layout mirrors the E0 pilot recorder (RUN_RECORD.json / RUN_SUMMARY.md /
RAW_RUN_PATH.txt) so E1 runs remain auditable in the same way as E0.

Usage:
    python experiments/first_test/E1_mini/run_e1_mini.py preflight
    python experiments/first_test/E1_mini/run_e1_mini.py run --block m0m1
    python experiments/first_test/E1_mini/run_e1_mini.py run --block m2
    python experiments/first_test/E1_mini/run_e1_mini.py run --block all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from TafM_改进源码.config import load_v21_config, default_selector_path  # noqa: E402
from TafM_改进源码.dataset import SequenceStore, load_frozen_selector  # noqa: E402

E1_ROOT = PROJECT_ROOT / "experiments" / "first_test" / "E1_mini"
RUNS_DIR = E1_ROOT / "runs"
CONFIG_PATH = SRC_ROOT / "config_tabm_v21.yaml"
CLI = PROJECT_ROOT / "run_tabm_v21.py"

PYTHON = sys.executable

WINDOWS: list[tuple[str, str, str]] = [
    ("W1", "2026-02-12", "2026-02-18"),
    ("W2", "2026-04-12", "2026-04-18"),
    ("W3", "2026-06-12", "2026-06-18"),
    ("W4", "2026-08-07", "2026-08-13"),
]

# Frozen E1-mini variant matrix. Only the listed axes differ between variants.
VARIANTS: dict[str, dict[str, str]] = {
    "M0": {"objective_mode": "joint_v21", "checkpoint_policy": "v21_guardrail", "gradient_policy": "vanilla"},
    "M1": {"objective_mode": "joint_v21", "checkpoint_policy": "direction_first", "gradient_policy": "vanilla"},
    "M2": {"objective_mode": "dir_only", "checkpoint_policy": "direction_first", "gradient_policy": "vanilla"},
}

BLOCKS = {"m0m1": ("M0", "M1"), "m2": ("M2",), "all": ("M0", "M1", "M2")}

TELEMETRY_KEYS = (
    "wall_time_total_seconds", "training_epoch_seconds_total", "training_epoch_seconds_mean",
    "training_epoch_seconds_p50", "training_epoch_seconds_p95", "best_epoch", "stop_epoch",
    "epochs_run", "prediction_latency_ms", "parameter_count_total", "parameter_count_trainable",
    "checkpoint_size_bytes", "device", "cuda_peak_memory_bytes",
)

# Fields lifted verbatim from the raw manifest for checkpoint/safety diagnostics.
MANIFEST_KEYS = (
    "objective_mode", "checkpoint_policy", "gradient_policy", "checkpoint_rule",
    "checkpoint_raw_tolerance", "checkpoint_raw_anchor", "checkpoint_selected_raw",
    "checkpoint_guardrail_status", "stage_a_best_epoch", "stage_a_base_train_days",
    "stage_a_monitor_days", "selected_feature_count", "numerical_stability_status",
    "checkpoint_reload", "prediction_hours", "source_gate", "sequence_gate",
)


def target_days() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for window, start, end in WINDOWS:
        for day in pd.date_range(start, end):
            out.append((window, str(day.date())))
    return out


def run_id_for(variant: str, day: str, attempt: int) -> str:
    base = f"E1-{variant}-{day}"
    return base if attempt == 1 else f"{base}__r{attempt}"


def command_for(variant: str, day: str) -> list[str]:
    spec = VARIANTS[variant]
    return [
        PYTHON, str(CLI), "direction-experiment",
        "--target-day", day,
        "--profile", "default",
        "--mode", "A2",
        "--objective-mode", spec["objective_mode"],
        "--checkpoint-policy", spec["checkpoint_policy"],
        "--gradient-policy", spec["gradient_policy"],
    ]


def _json_file(path: Path | None) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path and path.is_file() else {}


def _environment_snapshot() -> dict:
    snippet = ("import json,platform,sys,torch; ok=torch.cuda.is_available(); "
               "print(json.dumps({'python_version':platform.python_version(),'python_executable':sys.executable,"
               "'conda_prefix':sys.prefix,'torch_version':torch.__version__,'cuda_available':ok,"
               "'cuda_version':torch.version.cuda,'gpu_name':torch.cuda.get_device_name(0) if ok else None}))")
    result = subprocess.run([PYTHON, "-c", snippet], cwd=SRC_ROOT, text=True, capture_output=True, check=False)
    if result.returncode:
        return {"status": "UNAVAILABLE", "stderr": result.stderr[-2000:]}
    info = json.loads(result.stdout)
    info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    return info


def _write_record(record_dir: Path, record: dict) -> None:
    record_dir.mkdir(parents=True, exist_ok=True)
    (record_dir / "RUN_RECORD.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    raw_path = record.get("raw_run_dir") or "PENDING"
    (record_dir / "RAW_RUN_PATH.txt").write_text(str(raw_path) + "\n", encoding="utf-8")
    m = record.get("metrics") or {}
    t = record.get("telemetry") or {}
    c = record.get("checkpoint") or {}
    lines = [
        f"# {record.get('run_id', '')}", "",
        f"RUN_ID: {record.get('run_id', '')}",
        f"GATE: {record.get('gate', '')}",
        f"VARIANT: {record.get('variant', '')} (objective={VARIANTS[record.get('variant', 'M0')]['objective_mode']}, "
        f"checkpoint={VARIANTS[record.get('variant', 'M0')]['checkpoint_policy']}, gradient=vanilla)",
        f"WINDOW: {record.get('window', '')}",
        f"TARGET_DAY: {record.get('target_day', '')}",
        f"PURPOSE: {record.get('purpose', '')}",
        f"COMMAND: {record.get('command', '')}",
        f"RAW_RUN_DIR: {raw_path}",
        f"STATUS: {record.get('status', '')}", "",
        "## RESULT", "",
        f"- Direction Raw: {m.get('raw_direction_accuracy', 'N/A')}",
        f"- Balanced: {m.get('balanced_accuracy', 'N/A')}",
        f"- +Recall: {m.get('positive_recall', 'N/A')}",
        f"- -Recall: {m.get('nonpositive_recall', 'N/A')}",
        f"- AUC / Brier: {m.get('auc', 'N/A')} / {m.get('brier', 'N/A')}",
        f"- Magnitude MAE: {m.get('magnitude_mae', 'N/A')} (skill {m.get('magnitude_skill', 'N/A')})",
        f"- Predicted positive fraction: {m.get('predicted_positive_fraction', 'N/A')}",
        f"- Best / stop epoch: {t.get('best_epoch', 'N/A')} / {t.get('stop_epoch', 'N/A')}",
        f"- Wall time (s): {t.get('wall_time_total_seconds', 'N/A')}",
        f"- Parameters: {t.get('parameter_count_total', 'N/A')} total / {t.get('parameter_count_trainable', 'N/A')} trainable",
        "", "## SELECTED CHECKPOINT MONITOR", "",
        f"- monitor Raw: {c.get('monitor_raw', 'N/A')}",
        f"- monitor Balanced: {c.get('monitor_balanced', 'N/A')}",
        f"- monitor +Recall / -Recall: {c.get('monitor_positive_recall', 'N/A')} / {c.get('monitor_nonpositive_recall', 'N/A')}",
        f"- monitor L_dir: {c.get('monitor_L_dir', 'N/A')}",
        f"- monitor Magnitude MAE: {c.get('monitor_magnitude_mae', 'N/A')}",
        f"- raw anchor / selected Raw: {c.get('raw_anchor', 'N/A')} / {c.get('selected_raw', 'N/A')}",
        "", "## OBSERVATION", "",
        f"E1-mini {record.get('variant', '')} rolling-origin observation for {record.get('window', '')}; "
        "aggregated only through the frozen E1-mini analysis, not a per-day conclusion.",
        "", "## WARNING", "",
        ", ".join(record.get("warnings", [])) or "None recorded.",
        "", "## DECISION", "",
        str(record.get("decision", "E1_OBSERVATION")),
        "", "## NEXT", "",
        str(record.get("next", "Aggregate in experiments/first_test/E1_mini; no automatic branch execution.")), "",
    ]
    (record_dir / "RUN_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def _selected_checkpoint_row(raw_dir: Path | None, manifest: dict) -> dict:
    """Monitor metrics of the selected checkpoint epoch, plus target-day collapse flags."""
    best_epoch = manifest.get("stage_a_best_epoch")
    row: dict = {"best_epoch": best_epoch}
    history_path = raw_dir / "training_history.parquet" if raw_dir else None
    if history_path and history_path.is_file() and isinstance(best_epoch, int) and best_epoch >= 1:
        history = pd.read_parquet(history_path)
        selected = history[history["epoch"] == best_epoch]
        if len(selected) == 1:
            r = selected.iloc[0]
            row.update({
                "monitor_raw": float(r["monitor_raw_direction_accuracy"]),
                "monitor_balanced": float(r["monitor_balanced_accuracy"]),
                "monitor_positive_recall": float(r["monitor_positive_recall"]),
                "monitor_nonpositive_recall": float(r["monitor_nonpositive_recall"]),
                "monitor_L_dir": float(r["monitor_L_dir"]),
                "monitor_magnitude_mae": float(r["monitor_magnitude_mae"]),
                "monitor_predicted_positive_fraction": float(r["monitor_predicted_positive_fraction"]),
                "monitor_diagnostic_warnings": str(r.get("monitor_diagnostic_warnings") or ""),
            })
    row["raw_anchor"] = manifest.get("checkpoint_raw_anchor")
    row["selected_raw"] = manifest.get("checkpoint_selected_raw")
    row["guardrail_status"] = manifest.get("checkpoint_guardrail_status")
    return row


def execute_run(variant: str, window: str, day: str, *, attempt: int = 1, force: bool = False) -> dict:
    run_id = run_id_for(variant, day, attempt)
    record_dir = RUNS_DIR / run_id
    existing = _json_file(record_dir / "RUN_RECORD.json")
    if existing.get("status") == "PASS" and not force:
        print(f"[skip] {run_id} already PASS", flush=True)
        return existing

    command = command_for(variant, day)
    command_text = subprocess.list2cmdline(command)
    cfg = load_v21_config(CONFIG_PATH)
    started = datetime.now(timezone.utc)
    initial = {
        "run_id": run_id, "gate": "E1", "variant": variant, "window": window,
        "target_day": day, "date": started.isoformat(), "status": "RUNNING",
        "purpose": f"E1-mini {variant} rolling-origin observation",
        "command": command_text, "config_path": str(CONFIG_PATH),
        "config_sha256": cfg.config_sha256, "resolved_config": cfg.resolved_config(),
        "seed": cfg.seed, "raw_run_dir": None, "environment": _environment_snapshot(),
        "decision": "E1_OBSERVATION", "warnings": [],
    }
    _write_record(record_dir, initial)
    print(f"[run ] {run_id} :: {command_text}", flush=True)

    completed = subprocess.run(command, cwd=SRC_ROOT, text=True, capture_output=True, check=False)
    (record_dir / "COMMAND_STDOUT.txt").write_text(completed.stdout, encoding="utf-8")
    (record_dir / "COMMAND_STDERR.txt").write_text(completed.stderr, encoding="utf-8")

    result: dict = {}
    return_code = completed.returncode
    stderr = completed.stderr
    if return_code == 0:
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return_code = 1
            stderr = "CLI output was not valid JSON; refusing to mark the run complete."

    raw_dir = Path(result["run_dir"]) if return_code == 0 and result.get("run_dir") else None
    manifest = _json_file(raw_dir / "manifest.json") if raw_dir else {}
    metrics = _json_file(raw_dir / "metrics.json") if raw_dir else {}
    telemetry = {key: manifest.get(key) for key in TELEMETRY_KEYS}
    manifest_subset = {key: manifest.get(key) for key in MANIFEST_KEYS}
    warnings = sorted(set(metrics.get("diagnostic_warnings", [])) | set(manifest.get("checkpoint_warnings", [])))
    status = "PASS" if return_code == 0 and manifest.get("status") == "COMPLETE" else "FAIL"
    history_path = raw_dir / "training_history.parquet" if raw_dir else None
    history = (pd.read_parquet(history_path).to_dict(orient="records")
               if history_path and history_path.is_file() else [])
    try:
        git_status = subprocess.run(["git", "status", "--short"], cwd=PROJECT_ROOT,
                                    text=True, capture_output=True, check=False).stdout
    except OSError:
        git_status = "UNAVAILABLE"

    record = {
        **initial, "status": status, "finished_at": datetime.now(timezone.utc).isoformat(),
        "return_code": return_code, "failure_reason": stderr if status != "PASS" else None,
        "raw_run_dir": str(raw_dir) if raw_dir else None,
        "raw_manifest_path": str(raw_dir / "manifest.json") if raw_dir else None,
        "raw_manifest_sha256": (hashlib.sha256((raw_dir / "manifest.json").read_bytes()).hexdigest()
                                if raw_dir and (raw_dir / "manifest.json").is_file() else None),
        "config_sha256": manifest.get("config_sha256", cfg.config_sha256),
        "resolved_config": manifest.get("resolved_config"),
        "selector_sha256": manifest.get("selector_sha256"),
        "source_sha256": manifest.get("source_sha256"),
        "sequence_manifest_sha256": manifest.get("sequence_manifest_sha256"),
        "seed": manifest.get("seed", cfg.seed),
        "device": manifest.get("device"), "amp": manifest.get("amp"),
        "package_versions": manifest.get("package_versions"),
        "manifest_fields": manifest_subset,
        "checkpoint": _selected_checkpoint_row(raw_dir, manifest),
        "metrics": metrics, "telemetry": telemetry, "warnings": warnings,
        "training_history": history, "git_status_short": git_status,
        "decision": "E1_OBSERVATION" if status == "PASS" else "INVALID",
    }
    _write_record(record_dir, record)
    print(f"[done] {run_id} status={status} raw={telemetry.get('wall_time_total_seconds')}s "
          f"raw_acc={metrics.get('raw_direction_accuracy')}", flush=True)
    return record


def preflight() -> int:
    store = SequenceStore.load()
    selector, selector_hash = load_frozen_selector(default_selector_path())
    days = {pd.Timestamp(d).date() for d in store.days}
    cfg = load_v21_config(CONFIG_PATH)
    rows, failures = [], []
    for window, day in target_days():
        d = pd.Timestamp(day).date()
        present = d in days
        eligibility = store.eligibility(selector, current_target_day=d) if present else {}
        train_n = len(eligibility.get("eligible_indices", []))
        self_index = [i for i, dd in enumerate(store.days) if pd.Timestamp(dd).date() == d]
        self_quarantined = bool(self_index) and self_index[0] in set(eligibility.get("quarantine_excluded_indices", []))
        ok = present and train_n >= 40 and not self_quarantined
        rows.append({"window": window, "target_day": day, "present": present, "train_samples": train_n,
                     "self_quarantined": self_quarantined, "eligible": ok})
        if not ok:
            failures.append(day)
    table = pd.DataFrame(rows)
    table.to_csv(E1_ROOT / "preflight_target_days.csv", index=False)
    print(table.to_string(index=False))
    print(f"\nTARGET_DAYS={len(table)} ELIGIBLE={int(table['eligible'].sum())} FAILURES={failures}")
    print(f"CONFIG_SHA256={cfg.config_sha256} SELECTOR_SHA256={selector_hash} SEED={cfg.seed}")
    print("PREFLIGHT=" + ("PASS" if not failures else "FAIL"))
    return 0 if not failures else 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    r = sub.add_parser("run")
    r.add_argument("--block", choices=sorted(BLOCKS), default="all")
    r.add_argument("--attempt", type=int, default=1)
    r.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "preflight":
        return preflight()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    variants = BLOCKS[args.block]
    records = []
    for variant in variants:
        for window, day in target_days():
            records.append(execute_run(variant, window, day, attempt=args.attempt, force=args.force))
    passed = sum(1 for rec in records if rec.get("status") == "PASS")
    print(f"\nBLOCK={args.block} RUNS={len(records)} PASS={passed} FAIL={len(records) - passed}")
    return 0 if passed == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
