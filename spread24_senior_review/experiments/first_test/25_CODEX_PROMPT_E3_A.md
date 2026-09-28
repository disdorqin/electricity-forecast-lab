# Codex Prompt — E3-A Recency-Aware Training Window

Project root:
D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_lab\spread24_senior_review

Only authorized task: execute experiments/first_test/24_E3_A_RECENCY_WINDOW_PLAN.md exactly.

Read first:
1 experiments/first_test/24_E3_A_RECENCY_WINDOW_PLAN.md
2 experiments/first_test/E2_architecture/horizon_specialized_head/E2_E1_summary.md
3 experiments/first_test/E2_architecture/horizon_specialized_head/E2_E1_GATE.md
4 docs/17_最终模型设计与编码规范.md
5 docs/01_业务数据与防泄漏合同.md

Frozen diagnosis:
Q2 segment_heads is current candidate at Raw .6086 with safety PASS.
Stage-A 80/20 percentage split now withholds roughly 300+ newest days from gradient training.
For 2026-08-11 BASE ends 2025-09-08 while 335 newer legal days are monitor-only.
Training curves show rapid old-distribution overfit.

Implement experiment-only stage_a_monitor_days and stage_a_history_window_days.
Default None/None must reproduce canonical 80/20 exactly.

Arms:
T0 current percent20 reuse Q2.
T1 monitor60 + expanding all earlier base.
T2 monitor60 + rolling365 base.
T3 monitor60 + rolling1095 base.

Keep Q2 architecture and every other training parameter fixed.
No Stage B. No new features. No architecture changes.

Gate A must prove exact date/count routing, no overlap/leakage, preprocessing BASE-only, hashes unchanged, Q2 route exact, tests green.

Run same 28-day W1-W4 panel.
Report overall/window/hour metrics, paired bootstrap, slot gains, exact split date ranges, best/stop epochs, train-monitor BCE gaps and trajectories, runtime.

Safety anchor T0.
E3_A_GATE exactly one of STALE_SPLIT_CONFIRMED / RECENT_YEAR_BEST / THREE_YEAR_BEST / EXPANDING_RECENT_BEST / LONG_HISTORY_NEEDED / PROMISING_RECENCY_UNPROVEN / NO_RECENCY_GAIN.

If any fresh arm >=.62 with safety, next suggestion must be 3-seed T0+winner; do not execute it.
If no recency gain, next suggestion must pivot to legal feature/regime work rather than more window tuning.

Do not run current Stage B unchanged; it is incompatible with Q2/dir_only as documented in the plan.
After E3-A STOP. Do not stage/commit.