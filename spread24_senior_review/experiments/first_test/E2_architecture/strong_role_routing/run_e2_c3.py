"""Run the preregistered E2-C3 strong role routing arms without early stopping."""
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
SCOPE = REPO / "experiments/first_test/E1_mini/preflight_target_days.csv"
F80_RUNS = REPO / "experiments/first_test/E2_architecture/fusion_coarse/runs"
FROZEN = ["--profile", "default", "--mode", "A2", "--objective-mode", "dir_only",
          "--checkpoint-policy", "direction_first", "--gradient-policy", "vanilla",
          "--architecture-mode", "full_current", "--direction-fusion-alpha", "0.8"]


def experiment_python():
    # Use the exact interpreter that produced the frozen F80 control.
    record = json.loads((F80_RUNS / "E2B1-F80-2026-02-13/RUN_RECORD.json").read_text(encoding="utf-8"))
    candidate = record["command"].split()[0]
    if not Path(candidate).is_file():
        raise RuntimeError(f"frozen F80 interpreter is missing: {candidate}")
    probe = subprocess.run([candidate, "-c", "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0) )"],
                          capture_output=True, text=True)
    if probe.returncode:
        raise RuntimeError(f"E2-C1 requires the CUDA interpreter used for F80: {probe.stderr}")
    return candidate


def scope():
    with SCOPE.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    expected = [("W1", "2026-02-12", "2026-02-18"), ("W2", "2026-04-12", "2026-04-18"),
                ("W3", "2026-06-12", "2026-06-18"), ("W4", "2026-08-07", "2026-08-13")]
    days = [(r["window"], r["target_day"]) for r in rows if r.get("eligible", "True").lower() == "true"]
    actual = [(w, d) for w, a, b in expected for d in sorted(x for ww, x in days if ww == w)]
    wanted = [(w, str((__import__("datetime").date.fromisoformat(a) + __import__("datetime").timedelta(days=i))))
              for w, a, b in expected for i in range((__import__("datetime").date.fromisoformat(b)-__import__("datetime").date.fromisoformat(a)).days+1)]
    if actual != wanted or len(actual) != 28:
        raise RuntimeError(f"frozen formal panel mismatch: got {actual}")
    return days


def command_for(day, direction_mode):
    return [experiment_python(), str(CLI), "direction-experiment", "--target-day", day, *FROZEN,
            "--strong-role-profile", direction_mode]


def run_one(arm, mode, window, day, *, benchmark=False, force=False):
    bucket = (HERE / "runs" / "benchmark") if benchmark else (HERE / "runs")
    out = bucket / f"E2C3-{arm}-{day}"
    record_path = out / "RUN_RECORD.json"
    if record_path.exists() and not force:
        old = json.loads(record_path.read_text(encoding="utf-8"))
        if old.get("status") == "PASS":
            return old
    out.mkdir(parents=True, exist_ok=True)
    command = command_for(day, mode)
    started = datetime.now(timezone.utc)
    tic = time.perf_counter()
    proc = subprocess.run(command, cwd=SRC, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    wall = time.perf_counter() - tic
    (out / "COMMAND_STDOUT.txt").write_text(proc.stdout or "", encoding="utf-8")
    (out / "COMMAND_STDERR.txt").write_text(proc.stderr or "", encoding="utf-8")
    row = {"run_id": f"E2C3-{arm}-{day}", "arm": arm, "direction_role_profile": mode,
           "window": window, "target_day": day, "command": " ".join(command),
           "started_utc": started.isoformat(), "wall_seconds": round(wall, 3),
           "return_code": proc.returncode, "status": "FAIL"}
    if proc.returncode == 0:
        try:
            payload = json.loads(proc.stdout)
            manifest, metrics = payload["manifest"], payload["metrics"]
            run_dir = Path(payload["run_dir"])
            row.update({"run_dir": str(run_dir), "metrics": metrics,
                "manifest": {k: manifest.get(k) for k in (
                    "status", "direction_role_profile", "architecture_mode", "direction_fusion_alpha",
                    "objective_mode", "checkpoint_policy", "gradient_policy", "train_mode", "seed",
                    "parameter_count_total", "parameter_count_trainable", "best_epoch", "stop_epoch",
                    "epochs_run", "wall_time_total_seconds", "training_epoch_seconds_mean",
                    "cuda_peak_memory_bytes", "config_sha256", "selector_sha256", "source_sha256",
                    "sequence_manifest_sha256", "device", "amp")}})
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
    p.add_argument("--arms", default="R1,R2")
    p.add_argument("--benchmark-only", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    all_days = scope()
    jobs = [("R0", "all", "benchmark", "2026-02-13")] if args.benchmark_only else []
    if args.benchmark_only:
        jobs += [(a, {"R1":"drop_mag", "R2":"drop_dir"}[a], "benchmark", "2026-02-13")
                 for a in args.arms.split(",") if a in {"R1", "R2"}]
    else:
        jobs = [(a, {"R1":"drop_mag", "R2":"drop_dir"}[a], w, d)
                for a in args.arms.split(",") for w, d in all_days]
    results = []
    for arm, route, window, day in jobs:
        if arm == "R0":
            # C0 is a reuse control; do not train it here.
            continue
        r = run_one(arm, route, window, day, benchmark=args.benchmark_only, force=args.force)
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


