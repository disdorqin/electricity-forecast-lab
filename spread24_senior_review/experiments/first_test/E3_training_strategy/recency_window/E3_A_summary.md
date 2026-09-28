# E3-A — Recency-Aware Training Window

**Screening-only; STOP after this experiment.** T0 reuses E2-E1 Q2 (canonical chronological 80/20); T1/T2/T3 are fresh CUDA+AMP runs (28 formal DEV days each). Only the Stage-A BASE/MONITOR split changes: the newest 60 eligible days are MONITOR and BASE is the newest M days strictly before it (M=None keeps all earlier days). Q2 architecture (segment_heads) and every other training parameter are fixed; preprocessing still fits BASE only. No Stage B, no new features, no architecture change.

## Gate A / frozen protocol
- T0 reproduces the saved E2-E1 Q2 benchmark checkpoint; max prediction delta 0.0e+00 (<=1e-9).
- T1/T2/T3 MONITOR is exactly the newest 60 eligible days ending D-2; T1 BASE ends immediately before MONITOR; T2 BASE is exactly 365 days and T3 exactly 1095; BASE/MONITOR never overlap and never contain target or D-1; `preprocessing_fit_day_start/end` equals the BASE range exactly.
- Source/config/selector/sequence hashes remain the frozen values; the Q2 segment-head route (3 heads) and k=8 are preserved.

- fresh arm records: 28/28 PASS each; all CUDA+AMP.

## Exact split (formal, last day 2026-08-13)
| Arm | split mode | monitor n | monitor start | monitor end | base n | base start | base end | eligible n | fit start | fit end |
|---|---|---|---|---|---|---|---|---|---|---|
| T1 | explicit_monitor_expanding | 60 | 2026-06-13 | 2026-08-11 | 1434–1616 | 2022-01-09 | 2026-06-12 | 1676 | 2022-01-09 | 2026-06-12 |
| T2 | explicit_monitor_rolling | 60 | 2026-06-13 | 2026-08-11 | 365–365 | 2025-06-13 | 2026-06-12 | 1676 | 2025-06-13 | 2026-06-12 |
| T3 | explicit_monitor_rolling | 60 | 2026-06-13 | 2026-08-11 | 1095–1095 | 2023-06-14 | 2026-06-12 | 1676 | 2023-06-14 | 2026-06-12 |

## Benchmark day — engineering only
| Arm | Raw | Balanced | +Recall | -Recall | AUC | Brier | ppf |
|---|---|---|---|---|---|---|---|
| T0 | 0.5833 | 0.4958 | 0.2857 | 0.7059 | 0.5378 | 0.2267 | 0.2917 |
| T1 | 0.6250 | 0.5672 | 0.4286 | 0.7059 | 0.6134 | 0.2233 | 0.3333 |
| T2 | 0.6250 | 0.6513 | 0.7143 | 0.5882 | 0.6555 | 0.2380 | 0.5000 |
| T3 | 0.3750 | 0.3487 | 0.2857 | 0.4118 | 0.4286 | 0.2501 | 0.5000 |

2026-02-13 is excluded from the gate ranking. T0 uses the frozen E2-E1 Q2 control.

## Overall formal DEV (28 days / 672 slots)
| Arm | Raw | Balanced | +Recall | -Recall | AUC | Brier | ppf | one-class days | +R=0 days | -R=0 days | min W Raw | W Raw std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | 0.6086 | 0.5424 | 0.3058 | 0.7791 | 0.5911 | 0.2297 | 0.2515 | 2 | 8 | 0 | 0.5595 | 0.0597 |
| T1 | 0.5685 | 0.5074 | 0.2893 | 0.7256 | 0.5313 | 0.2424 | 0.2798 | 1 | 9 | 0 | 0.5179 | 0.0479 |
| T2 | 0.5506 | 0.5070 | 0.3512 | 0.6628 | 0.4963 | 0.2617 | 0.3423 | 0 | 6 | 0 | 0.4286 | 0.0799 |
| T3 | 0.5357 | 0.4683 | 0.2273 | 0.7093 | 0.4695 | 0.2535 | 0.2679 | 1 | 7 | 1 | 0.4881 | 0.0498 |

## W1–W4
| Arm | Scope | Raw | Balanced | +Recall | -Recall | AUC | Brier |
|---|---|---|---|---|---|---|---|
| T0 | W1 | 0.7083 | 0.5516 | 0.2381 | 0.8651 | 0.6447 | 0.1923 |
| T0 | W2 | 0.5655 | 0.5673 | 0.4118 | 0.7229 | 0.6320 | 0.2417 |
| T0 | W3 | 0.5595 | 0.5056 | 0.3167 | 0.6944 | 0.4724 | 0.2514 |
| T0 | W4 | 0.6012 | 0.4936 | 0.1818 | 0.8053 | 0.5532 | 0.2332 |
| T1 | W1 | 0.6369 | 0.5040 | 0.2381 | 0.7698 | 0.5469 | 0.2122 |
| T1 | W2 | 0.5179 | 0.5194 | 0.3882 | 0.6506 | 0.5236 | 0.2725 |
| T1 | W3 | 0.5298 | 0.4713 | 0.2667 | 0.6759 | 0.4832 | 0.2504 |
| T1 | W4 | 0.5893 | 0.4894 | 0.2000 | 0.7788 | 0.5237 | 0.2346 |
| T2 | W1 | 0.6190 | 0.6429 | 0.6905 | 0.5952 | 0.6149 | 0.2279 |
| T2 | W2 | 0.5298 | 0.5320 | 0.3412 | 0.7229 | 0.5551 | 0.3018 |
| T2 | W3 | 0.4286 | 0.3778 | 0.2000 | 0.5556 | 0.3508 | 0.2795 |
| T2 | W4 | 0.6250 | 0.5346 | 0.2727 | 0.7965 | 0.5237 | 0.2376 |
| T3 | W1 | 0.6190 | 0.4683 | 0.1667 | 0.7698 | 0.5463 | 0.2227 |
| T3 | W2 | 0.4881 | 0.4903 | 0.3059 | 0.6747 | 0.5256 | 0.2769 |
| T3 | W3 | 0.5119 | 0.4241 | 0.1167 | 0.7315 | 0.3494 | 0.2621 |
| T3 | W4 | 0.5238 | 0.4594 | 0.2727 | 0.6460 | 0.4582 | 0.2522 |

## H1–H3
| Arm | Scope | Raw | Balanced | +Recall | -Recall | AUC | Brier |
|---|---|---|---|---|---|---|---|
| T0 | H1 | 0.5179 | 0.4813 | 0.3294 | 0.6331 | 0.4870 | 0.2546 |
| T0 | H2 | 0.6920 | 0.5578 | 0.2000 | 0.9156 | 0.6617 | 0.2026 |
| T0 | H3 | 0.6161 | 0.5708 | 0.3678 | 0.7737 | 0.6111 | 0.2318 |
| T1 | H1 | 0.5179 | 0.4881 | 0.3647 | 0.6115 | 0.4895 | 0.2567 |
| T1 | H2 | 0.6429 | 0.5104 | 0.1571 | 0.8636 | 0.5785 | 0.2171 |
| T1 | H3 | 0.5446 | 0.5040 | 0.3218 | 0.6861 | 0.5133 | 0.2535 |
| T2 | H1 | 0.4732 | 0.4727 | 0.4706 | 0.4748 | 0.4771 | 0.2747 |
| T2 | H2 | 0.6786 | 0.5286 | 0.1286 | 0.9286 | 0.4857 | 0.2380 |
| T2 | H3 | 0.5000 | 0.4843 | 0.4138 | 0.5547 | 0.4657 | 0.2723 |
| T3 | H1 | 0.4420 | 0.3995 | 0.2235 | 0.5755 | 0.4135 | 0.2687 |
| T3 | H2 | 0.6116 | 0.4955 | 0.1857 | 0.8052 | 0.4341 | 0.2428 |
| T3 | H3 | 0.5536 | 0.5008 | 0.2644 | 0.7372 | 0.4969 | 0.2490 |

## Paired day-cluster bootstrap (10,000 draws, seed 20260924)
| Pair | ΔRaw | Raw CI low | Raw CI high | Raw W/T/L | ΔBalanced | Balanced CI low | Balanced CI high |
|---|---|---|---|---|---|---|---|
| T1-T0 | -0.0402 | -0.0744 | -0.0074 | 9/4/15 | -0.0304 | -0.0763 | 0.0122 |
| T2-T0 | -0.0580 | -0.1101 | -0.0074 | 9/3/16 | 0.0057 | -0.0695 | 0.0800 |
| T3-T0 | -0.0729 | -0.1190 | -0.0253 | 5/7/16 | -0.0392 | -0.0854 | 0.0092 |
| T2-T1 | -0.0179 | -0.0640 | 0.0283 | 10/2/16 | 0.0360 | -0.0332 | 0.1081 |
| T3-T1 | -0.0327 | -0.0863 | 0.0179 | 10/3/15 | -0.0088 | -0.0608 | 0.0454 |
| T3-T2 | -0.0149 | -0.0774 | 0.0491 | 11/3/14 | -0.0449 | -0.1092 | 0.0212 |

T2/T3 vs T1 and T3 vs T2 are fresh-arm comparisons. No ranking is based on benchmark-day figures or pooled Raw alone.

## Slot gains vs T0
| Arm | Scope | correct arm | correct T0 | Δslots | slots | Δslots/day |
|---|---|---|---|---|---|---|
| T1 | overall | 382 | 409 | -27 | 672 | -0.9643 |
| T1 | H1 | 116 | 116 | 0 | 224 | 0.0000 |
| T1 | H2 | 144 | 155 | -11 | 224 | -0.3929 |
| T1 | H3 | 122 | 138 | -16 | 224 | -0.5714 |
| T2 | overall | 370 | 409 | -39 | 672 | -1.3929 |
| T2 | H1 | 106 | 116 | -10 | 224 | -0.3571 |
| T2 | H2 | 152 | 155 | -3 | 224 | -0.1071 |
| T2 | H3 | 112 | 138 | -26 | 224 | -0.9286 |
| T3 | overall | 360 | 409 | -49 | 672 | -1.7500 |
| T3 | H1 | 99 | 116 | -17 | 224 | -0.6071 |
| T3 | H2 | 137 | 155 | -18 | 224 | -0.6429 |
| T3 | H3 | 124 | 138 | -14 | 224 | -0.5000 |

## Epoch / BCE diagnostics (formal means)
| Arm | best ep | stop ep | epochs | train BCE@sel | monitor BCE@sel | BCE gap | mon RAW@sel |
|---|---|---|---|---|---|---|---|
| T0 | 2.4286 | 17.4286 | 17.4286 | 0.6668 | 0.6902 | 0.0234 | 0.5594 |
| T1 | 3.0714 | 18.0714 | 18.0714 | 0.6527 | 0.6865 | 0.0339 | 0.5837 |
| T2 | 5.1429 | 20.1429 | 20.1429 | 0.6225 | 0.7097 | 0.0872 | 0.5763 |
| T3 | 2.5357 | 17.5357 | 17.5357 | 0.6677 | 0.6844 | 0.0167 | 0.5848 |

`epoch_diagnostics.csv` holds the per-run selected-epoch train/monitor BCE, the BCE gap and the monitor Raw/AUC/Brier at the selected checkpoint plus monitor Raw min/final; the full per-epoch trajectories are in each run's `training_history.parquet`.

## Safety anchor: T0/Q2
| Arm | Admissible | Checks |
|---|---|---|
| T0 | True | {'balanced': True, 'positive_recall': True, 'one_class_days': True, 'positive_recall_zero_days': True} |
| T1 | False | {'balanced': False, 'positive_recall': True, 'one_class_days': True, 'positive_recall_zero_days': True} |
| T2 | False | {'balanced': False, 'positive_recall': True, 'one_class_days': True, 'positive_recall_zero_days': True} |
| T3 | False | {'balanced': False, 'positive_recall': False, 'one_class_days': True, 'positive_recall_zero_days': True} |

## Runtime (formal run means)
| Arm | External wall s | Train wall s | Epoch s | Best epoch | Stop epoch | CUDA peak MiB | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| T0 | 19.35 | 15.63 | 0.635 | 2.43 | 17.43 | 371.8 | 383,252 |
| T1 | 24.44 | 20.20 | 0.742 | 3.07 | 18.07 | 371.8 | 383,252 |
| T2 | 11.74 | 7.67 | 0.178 | 5.14 | 20.14 | 371.8 | 383,252 |
| T3 | 16.18 | 12.44 | 0.436 | 2.54 | 17.54 | 371.8 | 383,252 |

## Signal / decision
`NO_RECENCY_GAIN`

T1 Raw -4.02 pp (CI [-7.44,-0.74]pp), T2 Raw -5.80 pp (CI [-11.01,-0.74]pp), T3 Raw -7.29 pp (CI [-11.90,-2.53]pp). Best fresh arm = T1. Safety: T0=True, T1=False, T2=False, T3=False. Acceleration trigger (fresh arm >=.62 Raw with safety) NOT met; no promotion. 
62/63/65 is not claimed unless the actual 28-day metric reaches it with acceptable safety. Window micro-tuning stops here.

### Interpretation (recency hypothesis refuted)
- All three recency-restricted arms are **materially and stably worse** than T0 (paired Raw CIs exclude zero, all-negative window signs) and all fail the safety screen. The plan's hypothesis — that withholding the large recent MONITOR caused old-distribution overfit — is **not supported**: T0 (large recent monitor, BASE≈80% of eligible history) wins by +4.0 to +7.3 pp Raw.
- T1 has *more* BASE days than T0 (1616 vs ~1400) yet is 4.0 pp worse, so enlarging BASE alone does not help; the change shared by all fresh arms is the 60-day MONITOR. Because T1/T2/T3 all use monitor=60, E3-A cannot fully separate monitor-size from base-window effects, but the safest reading is that a 60-day MONITOR destabilises checkpoint selection (T2's BCE gap 0.087 vs T0's 0.023) and that restricting BASE to ≤1095 days loses useful regime coverage.
- NEXT: do NOT continue window tuning; pivot to legal feature/regime work (E4).

## Provenance and scope
Run commands, metrics and manifests are recorded under `runs/`. Run manifests bind source/config/selector/sequence hashes and record eligible/base/monitor counts and date ranges plus the preprocessing fit range (`split_summary.csv`). No Stage B, no new features, no architecture change, no threshold/calibration tuning, no PLE bins/dim, k/depth/width change, role filtering, gate/fusion sweep, Temporal/FFT change, broader DEV or lockbox was run.

Tests: canonical non-destructive baseline **94 passed**; full suite (canonical + all prior E2 + focused E3-A) **182 passed** (+8 focused). Fail-closed CLI combinations return exit 2. Non-blocking single-bin PLE and matplotlib warnings only.
