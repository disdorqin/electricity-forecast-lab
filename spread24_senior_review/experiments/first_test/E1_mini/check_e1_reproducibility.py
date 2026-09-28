"""Cross-check E1 runs against E0 pilot runs that share an identical configuration.

Any E0 target day whose (objective_mode, checkpoint_policy, gradient_policy) triple
also appears in E1 is re-run territory: same seed, same source/sequence/selector,
same profile. If the frozen training path is deterministic, the two runs must agree.

This is a provenance check only. It never feeds E1 aggregation; a mismatch here
means the training path is not reproducible and the E1 numbers must be re-examined.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
E0_RUNS = PROJECT_ROOT / "experiments" / "first_test" / "E0_pilot" / "runs"
E1_RUNS = PROJECT_ROOT / "experiments" / "first_test" / "E1_mini" / "runs"

COMPARE_KEYS = (
    "raw_direction_accuracy", "balanced_accuracy", "positive_recall", "nonpositive_recall",
    "auc", "brier", "magnitude_mae", "magnitude_skill", "predicted_positive_fraction",
)
TELEMETRY_KEYS = ("best_epoch", "stop_epoch", "epochs_run", "wall_time_total_seconds")


FLAG_NAMES = ("--objective-mode", "--checkpoint-policy", "--gradient-policy")


def _flag(command: str | None, name: str) -> str | None:
    """Read `--name value` out of a recorded command line (E0 records store flags only here)."""
    parts = (command or "").split()
    if name in parts and parts.index(name) + 1 < len(parts):
        return parts[parts.index(name) + 1]
    return None


def load_runs(root: Path) -> list[dict]:
    records = []
    for path in sorted(root.glob("*/RUN_RECORD.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("status") != "PASS":
            continue
        manifest = record.get("manifest_fields") or {}
        resolved = record.get("resolved_config") or {}
        command = record.get("command")
        record["_target_day"] = record.get("target_day") or record.get("target_day_or_window")
        record["_config_key"] = (
            manifest.get("objective_mode") or resolved.get("objective_mode") or _flag(command, "--objective-mode"),
            manifest.get("checkpoint_policy") or resolved.get("checkpoint_policy") or _flag(command, "--checkpoint-policy"),
            manifest.get("gradient_policy") or resolved.get("gradient_policy") or _flag(command, "--gradient-policy"),
        )
        # Gradient-diagnostics probes add flags that change the run; keep them out of the match.
        record["_comparable"] = not any(
            extra in (command or "") for extra in ("--gradient-diagnostics", "--profile smoke"))
        records.append(record)
    return records


def main() -> int:
    e0 = [r for r in load_runs(E0_RUNS) if r["_comparable"]]
    e1 = load_runs(E1_RUNS)
    index = {(r["_target_day"], r["_config_key"]): r for r in e1}

    rows, mismatches = [], []
    for rec in e0:
        key = (rec["_target_day"], rec["_config_key"])
        if key not in index:
            rows.append({"target_day": rec["_target_day"], "config": "/".join(str(x) for x in rec["_config_key"]),
                         "e0_run": rec["run_id"], "e1_run": None, "status": "NO_E1_COUNTERPART"})
            continue
        other = index[key]
        metrics_a, metrics_b = rec.get("metrics") or {}, other.get("metrics") or {}
        tele_a, tele_b = rec.get("telemetry") or {}, other.get("telemetry") or {}
        deltas = {}
        for name in COMPARE_KEYS:
            a, b = metrics_a.get(name), metrics_b.get(name)
            deltas[name] = None if a is None or b is None else abs(float(a) - float(b))
        worst = max((v for v in deltas.values() if v is not None), default=None)
        if worst is None:
            status = "NO_COMPARABLE_METRIC"
        elif worst == 0.0:
            status = "EXACT"
        elif worst <= 1e-9:
            status = "NUMERIC"
        else:
            status = "DRIFT"
        epoch_match = tele_a.get("best_epoch") == tele_b.get("best_epoch")
        row = {
            "target_day": rec["_target_day"], "config": "/".join(str(x) for x in rec["_config_key"]),
            "e0_run": rec["run_id"], "e1_run": other["run_id"], "status": status,
            "max_abs_metric_delta": worst, "best_epoch_e0": tele_a.get("best_epoch"),
            "best_epoch_e1": tele_b.get("best_epoch"), "best_epoch_match": epoch_match,
            "wall_e0": tele_a.get("wall_time_total_seconds"),
            "wall_e1": tele_b.get("wall_time_total_seconds"),
        }
        row.update({f"delta_{k}": v for k, v in deltas.items()})
        rows.append(row)
        if status not in {"EXACT", "NUMERIC"} or not epoch_match:
            mismatches.append(row)

    table = pd.DataFrame(rows)
    out = PROJECT_ROOT / "experiments" / "first_test" / "E1_mini" / "reproducibility_check.csv"
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    if len(table):
        print("\nstatus counts:\n" + table["status"].value_counts().to_string())
    print(f"\nOVERLAPPING_DAYS={int(table['e1_run'].notna().sum()) if len(table) else 0} "
          f"MISMATCHES={len(mismatches)}")
    print("REPRODUCIBILITY=" + ("PASS" if not mismatches else "REVIEW"))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
