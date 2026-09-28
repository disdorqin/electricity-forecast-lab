# E3-A — Recency-Aware Training Window Big-Block

STATUS=AUTHORIZED_NEXT
DATE=2026-09-26
PRIMARY_GOAL=CROSS_MONTH_DIRECTION_65_PLUS

## Root cause to test
Q2 SEGMENT_HEADS is the strongest architecture candidate so far: Raw=.6086, Balanced=.5424, AUC=.5911, +13/672 correct slots, safety PASS.

Canonical Stage A currently uses all eligible history then chronological 80% BASE_TRAIN / newest 20% MONITOR.
Because the store now spans >4 years, that newest 20% is ~300+ days rather than a small recent monitor.

Observed Q2 manifests:
2026-02-12: BASE=1195, MONITOR=299, BASE end=2025-04-17.
2026-04-15: BASE=1244, MONITOR=312, BASE end=2025-06-05.
2026-06-15: BASE=1293, MONITOR=324, BASE end=2025-07-24.
2026-08-11: BASE=1339, MONITOR=335, BASE end=2025-09-08.

For 2026-08-11, almost 11 months of newest legal history select the checkpoint but never update Stage-A weights or preprocessing.
Training curves show genuine rapid overfit: train BCE continues falling while monitor BCE worsens after roughly epoch 2-4.

## Fixed architecture
Use Q2 exactly: segment_heads, fixed H1=1-8/H2=9-16/H3=17-24, canonical PLE+raw, all roles, current 24-h gate, full_current, fixed fusion alpha=.8, k=8, dir_only, direction_first, vanilla, Stage A only.

## Experiment-only split interface
Add stage_a_monitor_days: int|None and stage_a_history_window_days: int|None.
Defaults None/None must preserve existing 80/20 behavior bit-identically.
When monitor_days=N: monitor = newest N eligible days ending D-2; candidate_base = all eligible days before monitor.
If history_window_days=None use all candidate_base; otherwise use newest M days from candidate_base.
Preprocessing fits BASE only. No overlap. No random split.

## Arms
T0 CURRENT_PERCENT20: reuse E2-E1 Q2.
T1 EXPANDING_RECENT60: monitor_days=60, history_window_days=None.
T2 ROLLING365_RECENT60: monitor_days=60, history_window_days=365.
T3 ROLLING1095_RECENT60: monitor_days=60, history_window_days=1095.

No Stage B and no ensemble in E3-A.

## Gate A
1 T0 reproduces saved Q2 benchmark <=1e-9.
2 T1/T2/T3 monitor is exactly newest60 eligible days ending D-2.
3 T1 base ends immediately before monitor.
4 T2 base count exactly365; T3 exactly1095 where available.
5 BASE/MONITOR no overlap and no target/D-1 label.
6 preprocessing fit range equals BASE range exactly.
7 selector/source/config/leakage frozen.
8 Q2 segment-head route and k=8 preserved.
9 canonical94 + all prior E2 + focused E3-A tests PASS.

Manifest must record eligible/base/monitor counts and date ranges plus preprocessing fit range.

## Formal panel
Same 28 days W1 Feb12-18, W2 Apr12-18, W3 Jun12-18, W4 Aug07-13 2026.
T0 reuse; T1/T2/T3 fresh all 28. No intermediate stopping.

## Training fixed
dir_only / direction_first / vanilla / segment_heads / full_current / current tabular / current horizon gate / all roles / canonical encoding / fixed alpha=.8 / Stage A / default / seed20260924 / k8 / max120 / patience15 / batch64 / frozen LR-WD / CUDA AMP.

## Metrics
Overall Raw/Balanced/+R/-R/AUC/Brier/ppf/collapse.
W1-W4 and H1-H3.
Paired T1/T2/T3 vs T0 plus fresh-arm comparisons with day-bootstrap 95% CI and W/T/L.
Slot gains overall and by H1/H2/H3.
Best/stop epoch, train-vs-monitor BCE at selected checkpoint, monitor Raw/AUC/Brier trajectories, runtime/GPU.

## Safety
Anchor T0. Balanced >= T0-.01; +Recall >= T0-.05; one-class <= T0+2; +R=0 days <= T0+2.

## E3_A_GATE RECENCY_SIGNAL
STALE_SPLIT_CONFIRMED / RECENT_YEAR_BEST / THREE_YEAR_BEST / EXPANDING_RECENT_BEST / LONG_HISTORY_NEEDED / PROMISING_RECENCY_UNPROVEN / NO_RECENCY_GAIN.

## Acceleration
If any fresh arm >=.62 Raw with safety: immediately 3-seed confirm T0 + winner.
If confirmed >=.62: freeze Q2 + training-window candidate, then corrected E3-B and broader DEV.
If winner >=.61 but <.62: 3-seed confirm, then E3-B.
If no recency gain: stop window sweeps and pivot to E4 legal feature/regime work.

## Existing Stage B must NOT be used unchanged
Current Stage B is not valid for Q2 experiment because its freeze audit assumes a shared direction_head, it uses the canonical joint train epoch even for a dir_only experiment, it consumes all Stage-A monitor for adaptation with no recent holdout, and it uses fixed 8 epochs without recent early stopping.

## STOP
After E3-A, stop for human review. Do not stage/commit.