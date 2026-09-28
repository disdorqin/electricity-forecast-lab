"""E2-A Gate C driver — A1 (tabular_only) and A2 (temporal_only) over the fixed 28-day DEV panel.

A0 (full_current) is NOT re-run here: it reuses E1 T2 verbatim, whose bit-exact reproduction on
the benchmark day was already proven in Gate A.

Scope is read from E1-mini's preflight_target_days.csv so the two experiments cannot drift apart.
One architecture axis, no per-window structure, no early stopping on intermediate windows.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SRC = REPO / "src"
CLI = SRC / "run_tabm_v21.py"
PYTHON = sys.executable
SCOPE_CSV = REPO / "experiments/first_test/E1_mini/preflight_target_days.csv"

# Frozen E2-A training protocol (docs/11 §Gate A routing).
ARCHS = {"A1": "tabular_only", "A2": "temporal_only"}
FROZEN = ["--profile", "default", "--mode", "A2",
          "--objective-mode", "dir_only",
          "--checkpoint-policy", "direction_first",
          "--gradient-policy", "vanilla"]


def scope():
    with SCOPE_CSV.open(newline="", encoding="utf-8") as handle:
        return [(r["window"], r["target_day"]) for r in csv.DictReader(handle)]


def command_for(target_day, architecture_mode):
    return [PYTHON, str(CLI), "direction-experiment", "--target-day", target_day,
            *FROZEN, "--architecture-mode", architecture_mode]


def run_one(arch, architecture_mode, window, target_day, force=False):
    out_dir = HERE / "runs" / f"E2-{arch}-{target_day}"
    record_path = out_dir / "RUN_RECORD.json"
    if record_path.exists() and not force:
        prior = json.loads(record_path.read_text(encoding="utf-8"))
        if prior.get("status") == "PASS":
            return prior

    out_dir.mkdir(parents=True, exist_ok=True)
    command = command_for(target_day, architecture_mode)
    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    proc = subprocess.run(command, cwd=str(SRC), capture_output=True, text=True, encoding="utf-8", errors="replace")
    wall = time.perf_counter() - clock

    (out_dir / "COMMAND_STDOUT.txt").write_text(proc.stdout or "", encoding="utf-8")
    (out_dir / "COMMAND_STDERR.txt").write_text(proc.stderr or "", encoding="utf-8")

    record = {"run_id": f"E2-{arch}-{target_day}", "arch": arch, "architecture_mode": architecture_mode,
              "window": window, "target_day": target_day, "command": " ".join(command),
              "started_utc": started.isoformat(), "wall_seconds": round(wall, 3),
              "return_code": proc.returncode, "status": "FAIL"}
    if proc.returncode == 0:
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            record["error"] = f"stdout not JSON: {exc}"
        else:
            manifest, metrics = payload["manifest"], payload["metrics"]
            run_dir = Path(payload["run_dir"])
            record["run_dir"] = str(run_dir)
            (out_dir / "RAW_RUN_PATH.txt").write_text(str(run_dir), encoding="utf-8")
            record["metrics"] = metrics
            record["manifest"] = {
                "status": manifest.get("status"),
                "architecture_mode": manifest.get("architecture_mode"),
                "objective_mode": manifest.get("objective_mode"),
                "checkpoint_policy": manifest.get("checkpoint_policy"),
                "gradient_policy": manifest.get("gradient_policy"),
                "train_mode": manifest.get("train_mode"),
                "parameter_count_total": manifest.get("parameter_count_total"),
                "parameter_count_trainable": manifest.get("parameter_count_trainable"),
                "best_epoch": manifest.get("best_epoch"),
                "stop_epoch": manifest.get("stop_epoch"),
                "epochs_run": manifest.get("epochs_run"),
                "stage_a_best_epoch": manifest.get("stage_a_best_epoch"),
                "stage_b_epochs": manifest.get("stage_b_epochs"),
                "wall_time_total_seconds": manifest.get("wall_time_total_seconds"),
                "training_epoch_seconds_mean": manifest.get("training_epoch_seconds_mean"),
                "cuda_peak_memory_bytes": manifest.get("cuda_peak_memory_bytes"),
                "config_sha256": manifest.get("config_sha256"),
                "alpha_trajectories": manifest.get("alpha_trajectories"),
            }
            record["status"] = "PASS" if manifest.get("status") == "COMPLETE" else "FAIL"
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archs", default="A1,A2")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    days = scope()
    archs = [(a.strip(), ARCHS[a.strip()]) for a in args.archs.split(",") if a.strip()]

    results = []
    for arch, mode in archs:
        for window, target_day in days:
            record = run_one(arch, mode, window, target_day, force=args.force)
            results.append(record)
            metrics = record.get("metrics") or {}
            print(f"{arch} {target_day} {record['status']:4} "
                  f"raw={metrics.get('raw_direction_accuracy')} "
                  f"wall={record['wall_seconds']}s", flush=True)

    summary_path = HERE / "runs" / "GATE_C_RUNS.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    passed = sum(1 for r in results if r["status"] == "PASS")
    print(f"\nGATE C: {passed}/{len(results)} PASS -> {summary_path}")
    return 0 if passed == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
