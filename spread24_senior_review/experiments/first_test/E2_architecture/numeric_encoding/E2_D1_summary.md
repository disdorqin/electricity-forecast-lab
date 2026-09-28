# E2-D1 — Numeric Encoding Big-Block Ablation

**Screening-only; STOP after this experiment.** N0 reuses R0/G0/C0/F80; N1/N2 are fresh CUDA+AMP runs (28 formal DEV days each). Only the numeric representation entering the Strong and Weak Tabular branches changes. Direction fusion alpha stays fixed at .8 and the current 24-h horizon gate stays active.

## Gate A / frozen protocol
- canonical (`--numeric-encoding-mode canonical`) and default/no-flag reproduced the F80 benchmark checkpoint; max prediction delta 0.0e+00 (<=1e-9).
- Source/config/selector/sequence hashes remain the frozen values in each run manifest; target and eligibility continue through the docs/01 origin=D-1 14:00, labels<=D-2 pipeline.
- Encoded dims: N0=9 (PLE_8D+raw), N1=1 (raw only), N2=8 (PLE only). All 211 Strong/Core + 11 Weak retained; frozen PLE bins/hashes unchanged; Temporal and k=8 remain active. Objective is dir_only, so E2-D1 makes no Magnitude-capability claim.
- fresh arm records: 28/28 PASS each; all CUDA+AMP.

## Benchmark day — engineering only
| Arm | Raw | Balanced | +Recall | -Recall | AUC | Brier | ppf |
|---|---|---|---|---|---|---|---|
| N0 | 0.5833 | 0.5378 | 0.4286 | 0.6471 | 0.5630 | 0.2364 | 0.3750 |
| N1 | 0.4167 | 0.3361 | 0.1429 | 0.5294 | 0.3529 | 0.2658 | 0.3750 |
| N2 | 0.5833 | 0.4958 | 0.2857 | 0.7059 | 0.4454 | 0.2508 | 0.2917 |

2026-02-13 is excluded from the gate ranking. N0 uses the frozen F80 control.

## Overall formal DEV (28 days / 672 slots)
| Arm | Raw | Balanced | +Recall | -Recall | AUC | Brier | ppf | one-class days | +R=0 days | -R=0 days | min W Raw | W Raw std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N0 | 0.5893 | 0.5372 | 0.3512 | 0.7233 | 0.5671 | 0.2351 | 0.3036 | 3 | 7 | 1 | 0.5298 | 0.0566 |
| N1 | 0.5833 | 0.5082 | 0.2397 | 0.7767 | 0.5180 | 0.2460 | 0.2292 | 2 | 9 | 0 | 0.4583 | 0.0870 |
| N2 | 0.5670 | 0.5243 | 0.3719 | 0.6767 | 0.5469 | 0.2458 | 0.3408 | 1 | 5 | 1 | 0.5298 | 0.0353 |

## W1–W4
| Arm | Window | Raw | Balanced | +Recall | -Recall | AUC | Brier |
|---|---|---|---|---|---|---|---|
| N0 | W1 | 0.6488 | 0.5040 | 0.2143 | 0.7937 | 0.5782 | 0.2074 |
| N0 | W2 | 0.5298 | 0.5303 | 0.4824 | 0.5783 | 0.5544 | 0.2500 |
| N0 | W3 | 0.5357 | 0.4870 | 0.3167 | 0.6574 | 0.4292 | 0.2562 |
| N0 | W4 | 0.6429 | 0.5525 | 0.2909 | 0.8142 | 0.5746 | 0.2271 |
| N1 | W1 | 0.6786 | 0.5238 | 0.2143 | 0.8333 | 0.5918 | 0.2002 |
| N1 | W2 | 0.4583 | 0.4613 | 0.2118 | 0.7108 | 0.4836 | 0.3002 |
| N1 | W3 | 0.5476 | 0.4926 | 0.3000 | 0.6852 | 0.4068 | 0.2642 |
| N1 | W4 | 0.6488 | 0.5430 | 0.2364 | 0.8496 | 0.5979 | 0.2193 |
| N2 | W1 | 0.6131 | 0.5040 | 0.2857 | 0.7222 | 0.5070 | 0.2405 |
| N2 | W2 | 0.5298 | 0.5309 | 0.4353 | 0.6265 | 0.6018 | 0.2442 |
| N2 | W3 | 0.5357 | 0.5204 | 0.4667 | 0.5741 | 0.5282 | 0.2589 |
| N2 | W4 | 0.5893 | 0.4987 | 0.2364 | 0.7611 | 0.5414 | 0.2394 |

## H1–H3
| Arm | Segment | Raw | Balanced | +Recall | -Recall | AUC | Brier |
|---|---|---|---|---|---|---|---|
| N0 | H1 | 0.5000 | 0.4714 | 0.3529 | 0.5899 | 0.4919 | 0.2534 |
| N0 | H2 | 0.6607 | 0.5545 | 0.2714 | 0.8377 | 0.5751 | 0.2196 |
| N0 | H3 | 0.6071 | 0.5719 | 0.4138 | 0.7299 | 0.6136 | 0.2324 |
| N1 | H1 | 0.5045 | 0.4636 | 0.2941 | 0.6331 | 0.4184 | 0.2769 |
| N1 | H2 | 0.6875 | 0.5584 | 0.2143 | 0.9026 | 0.6411 | 0.2059 |
| N1 | H3 | 0.5580 | 0.4940 | 0.2069 | 0.7810 | 0.4962 | 0.2551 |
| N2 | H1 | 0.5446 | 0.5371 | 0.5059 | 0.5683 | 0.5456 | 0.2520 |
| N2 | H2 | 0.5893 | 0.4909 | 0.2286 | 0.7532 | 0.4794 | 0.2440 |
| N2 | H3 | 0.5670 | 0.5285 | 0.3563 | 0.7007 | 0.5802 | 0.2412 |

## Paired day-cluster bootstrap (10,000 draws, seed 20260924)
| Pair | ΔRaw | Raw CI low | Raw CI high | Raw W/T/L | ΔBalanced | Balanced CI low | Balanced CI high |
|---|---|---|---|---|---|---|---|
| N1-N0 | -0.0060 | -0.0640 | 0.0432 | 14/5/9 | -0.0018 | -0.0677 | 0.0682 |
| N2-N0 | -0.0223 | -0.0551 | 0.0104 | 9/5/14 | -0.0244 | -0.0800 | 0.0272 |
| N1-N2 | 0.0164 | -0.0432 | 0.0714 | 17/0/11 | 0.0226 | -0.0467 | 0.0952 |

N1−N2 is descriptive. No ranking is based on benchmark-day figures or pooled Raw alone.

## Safety anchor: N0/F80
| Arm | Admissible | Checks |
|---|---|---|
| N0 | True | {'balanced': True, 'positive_recall': True, 'one_class_days': True, 'positive_recall_zero_days': True} |
| N1 | False | {'balanced': False, 'positive_recall': False, 'one_class_days': True, 'positive_recall_zero_days': True} |
| N2 | False | {'balanced': False, 'positive_recall': True, 'one_class_days': True, 'positive_recall_zero_days': True} |

## Encoding dimensions / parameter counts
| Arm | mode | encoded dim | Strong | Weak | TabM in-dim | Weak in-dim | params (benchmark) |
|---|---|---:|---:|---:|---:|---:|---:|
| N0 | `canonical` | 9 | 211 | 11 | 1899 | 99 | 382,724 |
| N1 | `raw_only` | 1 | 211 | 11 | 211 | 11 | 111,700 |
| N2 | `ple_only` | 8 | 211 | 11 | 1688 | 88 | 352,620 |

## Runtime (formal run means)
| Arm | External wall s | Train wall s | Epoch s | Best epoch | Stop epoch | CUDA peak MiB | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| N0 | 14.99 | 12.06 | 0.491 | 2.11 | 17.11 | 371.8 | 382,724 |
| N1 | 17.61 | 13.98 | 0.523 | 2.39 | 17.39 | 83.0 | 111,700 |
| N2 | 19.58 | 15.78 | 0.626 | 2.25 | 17.25 | 336.3 | 352,620 |

## Signal / decision
`NO_CLEAR_ENCODING_EFFECT`

N1 Raw is -0.60 pp vs N0 (paired CI [-6.40,+4.32]pp); N2 Raw is -2.23 pp vs N0 (paired CI [-5.51,+1.04]pp). Safety: N1 admissible=False, N2 admissible=False, N0 admissible=True. Signal = `NO_CLEAR_ENCODING_EFFECT`; do not promote or alter production architecture. Any candidate improvement still needs multi-seed confirmation before promotion.

## Provenance and scope
Run commands, metrics and manifests are recorded under `runs/`. Run manifests bind source/config/selector/sequence hashes; `runtime.csv` and `encoding_audit.csv` contain per-run telemetry and the numeric-encoding audit. No role filtering, selector rerun, PLE bins/dim sweep, k/depth/width change, gate tuning, Temporal/FFT, fusion/alpha sweep, threshold/calibration, checkpoint change, Stage B, broader DEV or lockbox was run.

Tests: canonical non-destructive baseline **94 passed**; full suite (canonical + all prior E2 + focused E2-D1) **165 passed** (+8 focused). Fail-closed CLI combinations return exit 2. Non-blocking single-bin PLE and matplotlib warnings only.
