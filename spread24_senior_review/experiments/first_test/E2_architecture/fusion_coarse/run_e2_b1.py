"""E2-B1 driver — fixed Direction fusion alpha over the frozen DEV panel.

Reused arms (not re-run here):
    F00  = E2-A A1 TABULAR_ONLY          (architecture_mode=tabular_only)
    F100 = E2-A A2 TEMPORAL_ONLY         (architecture_mode=temporal_only)
    FL08 = E2-A A0 FULL_CURRENT          (canonical learnable alpha, init 0.8)

Fresh arms: F20/F40/F60/F80 = direction_fusion_alpha 0.2/0.4/0.6/0.8, full 28 days each.
Gate A additionally trains the two fixed endpoints (alpha 0.0 / 1.0) on the benchmark day to
prove they reproduce the tabular_only / temporal_only benchmark runs bit-for-bit.

Scope is read from E1-mini's preflight_target_days.csv so the panels cannot drift.
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
REPO = HERE.parents[3]
SRC = REPO / "src"
CLI = SRC / "run_tabm_v21.py"
PYTHON = sys.executable
SCOPE_CSV = REPO / "experiments/first_test/E1_mini/preflight_target_days.csv"
BENCHMARK_DAY = "2026-02-13"

ALPHAS = {"F00": 0.0, "F20": 0.2, "F40": 0.4, "F60": 0.6, "F80": 0.8, "F100": 1.0}
FROZEN = ["--profile", "default", "--mode", "A2",
          "--objective-mode", "dir_only",
          "--checkpoint-policy", "direction_first",
          "--gradient-policy", "vanilla"]


def scope():
    with SCOPE_CSV.open(newline="", encoding="utf-8") as handle:
        return [(r["window"], r["target_day"]) for r in csv.DictReader(handle)]


def command_for(target_day, alpha):
    return [PYTHON, str(CLI), "direction-experiment", "--target-day", target_day,
            *FROZEN, "--direction-fusion-alpha", str(alpha)]


def run_one(arm, alpha, window, target_day, force=False):
    out_dir = HERE / "runs" / f"E2B1-{arm}-{target_day}"
    record_path = out_dir / "RUN_RECORD.json"
    if record_path.exists() and not force:
        prior = json.loads(record_path.read_text(encoding="utf-8"))
        if prior.get("status") == "PASS":
            return prior

    out_dir.mkdir(parents=True, exist_ok=True)
    command = command_for(target_day, alpha)
    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    proc = subprocess.run(command, cwd=str(SRC), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    wall = time.perf_counter() - clock
    (out_dir / "COMMAND_STDOUT.txt").write_text(proc.stdout or "", encoding="utf-8")
    (out_dir / "COMMAND_STDERR.txt").write_text(proc.stderr or "", encoding="utf-8")

    record = {"run_id": f"E2B1-{arm}-{target_day}", "arm": arm, "direction_fusion_alpha": alpha,
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
                key: manifest.get(key) for key in (
                    "status", "architecture_mode", "direction_fusion_alpha", "objective_mode",
                    "checkpoint_policy", "gradient_policy", "train_mode",
                    "parameter_count_total", "parameter_count_trainable", "best_epoch",
                    "stop_epoch", "epochs_run", "stage_a_best_epoch", "stage_b_epochs",
                    "wall_time_total_seconds", "training_epoch_seconds_mean",
                    "cuda_peak_memory_bytes", "config_sha256", "alpha_trajectories")}
            record["status"] = "PASS" if manifest.get("status") == "COMPLETE" else "FAIL"
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="F20,F40,F60,F80")
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.benchmark_only:
        # Gate A: fixed endpoints on the benchmark day, plus the four fresh intermediates.
        jobs = [("F00", BENCHMARK_DAY), ("F20", BENCHMARK_DAY), ("F40", BENCHMARK_DAY),
                ("F60", BENCHMARK_DAY), ("F80", BENCHMARK_DAY), ("F100", BENCHMARK_DAY)]
    else:
        days = scope()
        jobs = [(arm.strip(), day) for arm in args.arms.split(",") if arm.strip()
                for _, day in days]

    results = []
    for arm, target_day in jobs:
        window = next((w for w, d in scope() if d == target_day), "benchmark")
        record = run_one(arm, ALPHAS[arm], window, target_day, force=args.force)
        results.append(record)
        metrics = record.get("metrics") or {}
        print(f"{arm} alpha={ALPHAS[arm]} {target_day} {record['status']:4} "
              f"raw={metrics.get('raw_direction_accuracy')} wall={record['wall_seconds']}s", flush=True)

    name = "GATE_A_BENCHMARK_RUNS.json" if args.benchmark_only else "GATE_C_RUNS.json"
    (HERE / "runs" / name).write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    passed = sum(1 for r in results if r["status"] == "PASS")
    print(f"\n{passed}/{len(results)} PASS -> runs/{name}")
    return 0 if passed == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
