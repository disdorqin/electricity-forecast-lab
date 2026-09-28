# E3-A — Recency-Aware Training Window (plan snapshot)

STATUS=AUTHORIZED_EXECUTED
DATE=2026-09-26
PARENT=E2-E1
EXPERIMENT_ROOT=experiments/first_test/E3_training_strategy/recency_window
PRIMARY_GOAL=CROSS_MONTH_DIRECTION_65_PLUS

## Root cause to test
Q2 SEGMENT_HEADS is the strongest architecture candidate (Raw=.6086, Balanced=.5424, AUC=.5911,
+13/672 slots, safety PASS). Canonical Stage A uses all eligible history then a chronological 80/20
BASE_TRAIN/MONITOR percentage split. Because the store now spans >4 years, the newest 20% is ~300+
days: for 2026-08-11 BASE=1339 / MONITOR=335 and BASE ends 2025-09-08, so almost 11 months of the
newest legal history only selects the checkpoint and never updates Stage-A weights or preprocessing.
Training curves show rapid old-distribution overfit (train BCE falls while monitor BCE worsens).

## Experiment-only split interface
`stage_a_monitor_days: int|None` and `stage_a_history_window_days: int|None`; defaults None/None
preserve the canonical 80/20 split bit-identically. When monitor_days=N: MONITOR = newest N eligible
days ending D-2; candidate_base = all eligible days before MONITOR; history_window_days=None keeps all
candidate_base, otherwise BASE = newest M days of candidate_base. Preprocessing fits BASE only. No
overlap, no random split.

## Arms
- T0 CURRENT_PERCENT20: reuse E2-E1 Q2 (canonical 80/20). Safety anchor.
- T1 EXPANDING_RECENT60: monitor_days=60, history_window_days=None.
- T2 ROLLING365_RECENT60: monitor_days=60, history_window_days=365.
- T3 ROLLING1095_RECENT60: monitor_days=60, history_window_days=1095.

Q2 architecture (segment_heads, fixed 1-8/9-16/17-24), canonical PLE+raw, all roles, current 24-h
gate, full_current, fixed fusion alpha=.8, k=8, dir_only, direction_first, vanilla, Stage A only.
No Stage B, no ensemble, no new features, no architecture change.

Existing Stage B is NOT usable unchanged for Q2/dir_only (its freeze audit assumes a shared
direction_head and it consumes all Stage-A monitor without a recent holdout); it was not run.

## Gate A (all PASS — see benchmark/GATE_A_EVIDENCE.md)
T0 reproduces the saved Q2 benchmark <=1e-9; T1/T2/T3 MONITOR is exactly the newest 60 eligible days
ending D-2; T1 BASE ends immediately before MONITOR; T2 BASE=365 and T3 BASE=1095 exactly; BASE/MONITOR
no overlap and no target/D-1; preprocessing fit range == BASE range; selector/source/config/sequence
frozen; Q2 segment-head route and k=8 preserved; baseline 94 + full 182 tests PASS.

## Frozen protocol
Benchmark 2026-02-13 (engineering only). Formal DEV W1 02-12..02-18, W2 04-12..04-18, W3 06-12..06-18,
W4 08-07..08-13. T0 reuse; T1/T2/T3 fresh 28 days; no intermediate stopping.

## Forbidden (not run)
Stage B, new features, architecture change, threshold/calibration tuning, PLE bins/dim, k/depth/width,
role filtering, gate/fusion sweep, Temporal/FFT, broader DEV, lockbox.
