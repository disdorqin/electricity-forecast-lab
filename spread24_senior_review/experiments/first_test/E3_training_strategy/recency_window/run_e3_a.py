"""Run the preregistered E3-A recency-window arms (T0 reuse E2-E1 Q2; T1/T2/T3 fresh)."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SRC = REPO / "src"
CLI = SRC / "run_tabm_v21.py"
SCOPE = REPO / "experiments/first_test/E1_mini/preflight_target_days.csv"
F80_RUNS = REPO / "experiments/first_test/E2_architecture/fusion_coarse/runs"
WINDOWS = {"W1": ("2026-02-12", "2026-02-18"), "W2": ("2026-04-12", "2026-04-18"),
           "W3": ("2026-06-12", "2026-06-18"), "W4": ("2026-08-07", "2026-08-13")}
# (monitor_days, history_window_days)
SPLIT = {"T0": (None, None), "T1": (60, None), "T2": (60, 365), "T3": (60, 1095)}
FROZEN = ["--profile", "default", "--mode", "A2", "--objective-mode", "dir_only",
          "--checkpoint-policy", "direction_first", "--gradient-policy", "vanilla",
          "--architecture-mode", "full_current", "--direction-fusion-alpha", "0.8",
          "--strong-role-profile", "all", "--numeric-encoding-mode", "canonical",
          "--direction-readout-mode", "segment_heads"]


def experiment_python():
    record = json.loads((F80_RUNS / "E2B1-F80-2026-02-13/RUN_RECORD.json").read_text(encoding="utf-8"))
    candidate = record["command"].split()[0]
    if not Path(candidate).is_file():
        raise RuntimeError(f"frozen F80 interpreter is missing: {candidate}")
    probe = subprocess.run([candidate, "-c",
                            "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"],
                           capture_output=True, text=True)
    if probe.returncode:
        raise RuntimeError(f"E3-A requires the CUDA interpreter used for F80: {probe.stderr}")
    return candidate


def scope():
    with SCOPE.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    days = [(r["window"], r["target_day"]) for r in rows if r.get("eligible", "True").lower() == "true"]
    wanted = [(w, (date.fromisoformat(a) + timedelta(days=i)).isoformat())
              for w, (a, b) in WINDOWS.items() for i in range(7)]
    if sorted(days) != sorted(wanted) or len(wanted) != 28:
        raise RuntimeError(f"frozen formal panel mismatch: got {sorted(days)}")
    return wanted


def command_for(day, arm):
    cmd = [experiment_python(), str(CLI), "direction-experiment", "--target-day", day, *FROZEN]
    monitor, window = SPLIT[arm]
    if monitor is not None:
        cmd += ["--stage-a-monitor-days", str(monitor)]
    if window is not None:
        cmd += ["--stage-a-history-window-days", str(window)]
    return cmd


def run_one(arm, window, day, *, benchmark=False, force=False):
    bucket = (HERE / ("runs/benchmark" if benchmark else "runs"))
    out = bucket / f"E3A-{arm}-{day}"
    record_path = out / "RUN_RECORD.json"
    if record_path.exists() and not force:
        old = json.loads(record_path.read_text(encoding="utf-8"))
        if old.get("status") == "PASS":
            return old
    out.mkdir(parents=True, exist_ok=True)
    command = command_for(day, arm)
    started = datetime.now(timezone.utc)
    tic = time.perf_counter()
    proc = subprocess.run(command, cwd=SRC, capture_output=True, text=True, encoding="utf-8", errors="replace")
    wall = time.perf_counter() - tic
    (out / "COMMAND_STDOUT.txt").write_text(proc.stdout or "", encoding="utf-8")
    (out / "COMMAND_STDERR.txt").write_text(proc.stderr or "", encoding="utf-8")
    row = {"run_id": f"E3A-{arm}-{day}", "arm": arm, "monitor_days": SPLIT[arm][0],
           "history_window_days": SPLIT[arm][1], "window": window, "target_day": day,
           "command": " ".join(command), "started_utc": started.isoformat(),
           "wall_seconds": round(wall, 3), "return_code": proc.returncode, "status": "FAIL"}
    if proc.returncode == 0:
        try:
            payload = json.loads(proc.stdout)
            manifest, metrics = payload["manifest"], payload["metrics"]
            run_dir = Path(payload["run_dir"])
            row.update({"run_dir": str(run_dir), "metrics": metrics,
                "manifest": {k: manifest.get(k) for k in (
                    "status", "stage_a_split", "stage_a_split_spec", "direction_readout_mode",
                    "direction_readout_audit", "numeric_encoding_mode", "strong_role_profile",
                    "architecture_mode", "direction_fusion_alpha", "objective_mode", "checkpoint_policy",
                    "gradient_policy", "direction_tabular_mode", "direction_horizon_gate_mode",
                    "train_mode", "seed", "parameter_count", "best_epoch", "stop_epoch", "epochs_run",
                    "stage_a_base_train_days", "stage_a_monitor_days", "preprocessing_fit_day_start",
                    "preprocessing_fit_day_end", "wall_time_total_seconds", "training_epoch_seconds_mean",
                    "cuda_peak_memory_bytes", "config_sha256", "selector_sha256", "source_sha256",
                    "sequence_manifest_sha256", "numerical_stability_status", "device", "amp")}})
            row["status"] = ("PASS" if manifest.get("status") == "COMPLETE" and
                str(manifest.get("device", "")).startswith("cuda") and manifest.get("amp") is True else "FAIL")
            if row["status"] == "FAIL":
                row["error"] = "training did not satisfy required CUDA+AMP protocol"
        except Exception as exc:
            row["error"] = str(exc)
    record_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", default="T1,T2,T3")
    p.add_argument("--benchmark-only", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    all_days = scope()
    arms = [a for a in args.arms.split(",") if a in SPLIT]
    if args.benchmark_only:
        jobs = [(a, "benchmark", "2026-02-13") for a in arms]
    else:
        jobs = [(a, w, d) for a in arms for w, d in all_days]
    results = []
    for arm, window, day in jobs:
        r = run_one(arm, window, day, benchmark=args.benchmark_only, force=args.force)
        results.append(r)
        m = r.get("metrics") or {}
        print(f"{arm} {day} {r['status']} Raw={m.get('raw_direction_accuracy')} wall={r['wall_seconds']:.2f}s", flush=True)
    file = HERE / "runs" / ("GATE_A_BENCHMARK_RUNS.json" if args.benchmark_only else "FORMAL_RUNS.json")
    file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    passed = sum(r.get("status") == "PASS" for r in results)
    print(f"{passed}/{len(results)} fresh runs PASS -> {file}")
    return 0 if passed == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
