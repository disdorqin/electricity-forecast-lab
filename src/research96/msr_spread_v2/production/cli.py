"""Explicit single-target prepare (CPU) and run (CUDA fresh-fit) commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import ProductionAPI


def main(argv=None):
    parser = argparse.ArgumentParser(description="Frozen Spread96 production/shadow runtime")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        p = sub.add_parser(name)
        p.add_argument("--bundle-root", type=Path, required=True)
        p.add_argument("--target-day", required=True)
        p.add_argument("--manifest", type=Path, required=True)
        p.add_argument("--values", type=Path, required=True)
        p.add_argument("--history", type=Path, required=True)
        p.add_argument("--mode", choices=("promotion", "consumed_replay", "synthetic_test"), default="promotion")
        p.add_argument("--i1-ledger", type=Path)
        if name == "run":
            p.add_argument("--output-root", type=Path, required=True)
            p.add_argument("--f-history-root", type=Path)
            p.add_argument("--f-history-start", default="2026-01-01")
    args = parser.parse_args(argv)
    api = ProductionAPI(args.bundle_root)
    if args.command == "prepare":
        result = api.prepare_target(target_day=args.target_day, manifest_path=args.manifest,
                                   values_path=args.values, history_path=args.history,
                                   mode=args.mode, i1_ledger_path=args.i1_ledger)
        output = {
            "target_day": args.target_day, "input_features_shape": list(result["sample"].x.shape),
            "feature_names": len(result["sample"].names), "label_cutoff": result["bundle"].receipt["max_label_day"],
            "production_eligible": result["production_eligible"], "target_truth_opened": False,
            "bundle_identity": result["bundle_identity"],
        }
    else:
        from .runtime import run_target
        output = run_target(api, target_day=args.target_day, manifest_path=args.manifest,
                            values_path=args.values, history_path=args.history,
                            output_root=args.output_root, mode=args.mode,
                            i1_ledger_path=args.i1_ledger, f_history_root=args.f_history_root,
                            f_history_start=args.f_history_start)
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
