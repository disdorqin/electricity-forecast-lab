"""Run the 28 E4-A deep Q2 checkpoints under the three-way split (P1 SPLIT_CONTROL).

Only 28 fresh deep trainings are authorized; P2/P3 are derived from these identical checkpoints by
derive_e4_a.py. P0 reuses the E2-E1 Q2 runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SRC = REPO / "src"
CLI = SRC / "run_tabm_v21.py"
SCOPE = REPO / "experiments/first_test/E1_mini/preflight_target_days.csv"
F80_RUNS = REPO / "experiments/first_test/E2_architecture/fusion_coarse/runs"
WINDOWS = {"W1": ("2026-02-12", "2026-02-18"), "W2": ("2026-04-12", "2026-04-18"),
           "W3": ("2026-06-12", "2026-06-18"), "W4": ("2026-08-07", "2026-08-13")}
FROZEN = ["--profile", "default", "--mode", "A2", "--objective-mode", "dir_only",
          "--checkpoint-policy", "direction_first", "--gradient-policy", "vanilla",
          "--architecture-mode", "full_current", "--direction-fusion-alpha", "0.8",
          "--strong-role-profile", "all", "--numeric-encoding-mode", "canonical",
          "--direction-readout-mode", "segment_heads",
          "--direction-postprocess-mode", "none", "--e4-three-way-split"]


def experiment_python():
    record = json.loads((F80_RUNS / "E2B1-F80-2026-02-13/RUN_RECORD.json").read_text(encoding="utf-8"))
    candidate = record["command"].split()[0]
    if not Path(candidate).is_file():
        raise RuntimeError(f"frozen F80 interpreter is missing: {candidate}")
    probe = subprocess.run([candidate, "-c",
                            "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"],
                           capture_output=True, text=True)
    if probe.returncode:
        raise RuntimeError(f"E4-A requires the CUDA interpreter used for F80: {probe.stderr}")
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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_one(window, day, *, benchmark=False, force=False):
    bucket = (HERE / ("runs/benchmark" if benchmark else "runs"))
    out = bucket / f"E4A-P1-{day}"
    record_path = out / "RUN_RECORD.json"
    if record_path.exists() and not force:
        old = json.loads(record_path.read_text(encoding="utf-8"))
        if old.get("status") == "PASS":
            return old
    out.mkdir(parents=True, exist_ok=True)
    command = [experiment_python(), str(CLI), "direction-experiment", "--target-day", day, *FROZEN]
    started = datetime.now(timezone.utc)
    tic = time.perf_counter()
    proc = subprocess.run(command, cwd=SRC, capture_output=True, text=True, encoding="utf-8", errors="replace")
    wall = time.perf_counter() - tic
    (out / "COMMAND_STDOUT.txt").write_text(proc.stdout or "", encoding="utf-8")
    (out / "COMMAND_STDERR.txt").write_text(proc.stderr or "", encoding="utf-8")
    row = {"run_id": f"E4A-P1-{day}", "arm": "P1", "window": window, "target_day": day,
           "command": " ".join(command), "started_utc": started.isoformat(),
           "wall_seconds": round(wall, 3), "return_code": proc.returncode, "status": "FAIL"}
    if proc.returncode == 0:
        try:
            payload = json.loads(proc.stdout)
            manifest, metrics = payload["manifest"], payload["metrics"]
            run_dir = Path(payload["run_dir"])
            ckpt = run_dir / "stage_a_best.pt"
            row.update({"run_dir": str(run_dir), "metrics": metrics,
                "checkpoint_sha256": sha256_file(ckpt) if ckpt.exists() else None,
                "manifest": {k: manifest.get(k) for k in (
                    "status", "e4_three_way_split", "e4_three_way_split_spec", "e4_checkpoint_monitor_days",
                    "e4_calibrator_days", "stage_a_split", "stage_a_split_spec", "direction_readout_mode",
                    "direction_readout_audit", "direction_postprocess_mode", "numeric_encoding_mode",
                    "strong_role_profile", "architecture_mode", "direction_fusion_alpha", "objective_mode",
                    "checkpoint_policy", "gradient_policy", "direction_tabular_mode",
                    "direction_horizon_gate_mode", "train_mode", "seed", "parameter_count",
                    "best_epoch", "stop_epoch", "epochs_run", "stage_a_base_train_days",
                    "stage_a_monitor_days", "preprocessing_fit_day_start", "preprocessing_fit_day_end",
                    "wall_time_total_seconds", "training_epoch_seconds_mean", "cuda_peak_memory_bytes",
                    "config_sha256", "selector_sha256", "source_sha256", "sequence_manifest_sha256",
                    "numerical_stability_status", "device", "amp")}})
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
    p.add_argument("--benchmark-only", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    jobs = [("benchmark", "2026-02-13")] if args.benchmark_only else scope()
    results = []
    for window, day in jobs:
        r = run_one(window, day, benchmark=args.benchmark_only, force=args.force)
        results.append(r)
        m = r.get("metrics") or {}
        print(f"P1 {day} {r['status']} Raw={m.get('raw_direction_accuracy')} wall={r['wall_seconds']:.2f}s", flush=True)
    file = HERE / "runs" / ("GATE_A_BENCHMARK_RUNS.json" if args.benchmark_only else "FORMAL_RUNS.json")
    file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    passed = sum(r.get("status") == "PASS" for r in results)
    print(f"{passed}/{len(results)} fresh deep runs PASS -> {file}")
    return 0 if passed == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
