# E2-E1 — Horizon-Specialized Direction Readout

**Screening-only; STOP after this experiment.** Q0 reuses R0/G0/C0/F80; Q1/Q2 are fresh CUDA+AMP runs (28 formal DEV days each). Only the Direction readout over the fixed 1-8 / 9-16 / 17-24 segments changes. Encoding (canonical PLE+raw), all 211 Strong/Core + 11 Weak, the current 24-h gate and fixed fusion alpha=.8 are unchanged; the Magnitude path is untouched.

## Gate A / frozen protocol
- shared (`--direction-readout-mode shared`) and default/no-flag reproduced the F80 benchmark checkpoint; max prediction delta 0.0e+00 (<=1e-9).
- Source/config/selector/sequence hashes remain the frozen values in each run manifest; target and eligibility continue through the docs/01 origin=D-1 14:00, labels<=D-2 pipeline.
- Q1 adds exactly 3 zero-init scalar logit biases over one shared head (params 382,724 -> 382,727). Q2 replaces the head with exactly 3 independent heads and keeps no shared Direction head (params 383,252); each hour is produced by exactly one head. k=8 and [B,k,24] are preserved; p>=.5 threshold logic and the Magnitude head are unchanged.
- fresh arm records: 28/28 PASS each; all CUDA+AMP.

## Benchmark day — engineering only
| Arm | Raw | Balanced | +Recall | -Recall | AUC | Brier | ppf |
|---|---|---|---|---|---|---|---|
| Q0 | 0.5833 | 0.5378 | 0.4286 | 0.6471 | 0.5630 | 0.2364 | 0.3750 |
| Q1 | 0.5833 | 0.5378 | 0.4286 | 0.6471 | 0.5630 | 0.2362 | 0.3750 |
| Q2 | 0.5833 | 0.4958 | 0.2857 | 0.7059 | 0.5378 | 0.2267 | 0.2917 |

2026-02-13 is excluded from the gate ranking. Q0 uses the frozen F80 control.

## Overall formal DEV (28 days / 672 slots)
| Arm | Raw | Balanced | +Recall | -Recall | AUC | Brier | ppf | one-class days | +R=0 days | -R=0 days | min W Raw | W Raw std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q0 | 0.5893 | 0.5372 | 0.3512 | 0.7233 | 0.5671 | 0.2351 | 0.3036 | 3 | 7 | 1 | 0.5298 | 0.0566 |
| Q1 | 0.5893 | 0.5363 | 0.3471 | 0.7256 | 0.5677 | 0.2350 | 0.3006 | 3 | 7 | 1 | 0.5238 | 0.0597 |
| Q2 | 0.6086 | 0.5424 | 0.3058 | 0.7791 | 0.5911 | 0.2297 | 0.2515 | 2 | 8 | 0 | 0.5595 | 0.0597 |

## W1–W4
| Arm | Scope | Raw | Balanced | +Recall | -Recall | AUC | Brier |
|---|---|---|---|---|---|---|---|
| Q0 | W1 | 0.6488 | 0.5040 | 0.2143 | 0.7937 | 0.5782 | 0.2074 |
| Q0 | W2 | 0.5298 | 0.5303 | 0.4824 | 0.5783 | 0.5544 | 0.2500 |
| Q0 | W3 | 0.5357 | 0.4870 | 0.3167 | 0.6574 | 0.4292 | 0.2562 |
| Q0 | W4 | 0.6429 | 0.5525 | 0.2909 | 0.8142 | 0.5746 | 0.2271 |
| Q1 | W1 | 0.6488 | 0.5040 | 0.2143 | 0.7937 | 0.5807 | 0.2072 |
| Q1 | W2 | 0.5238 | 0.5245 | 0.4706 | 0.5783 | 0.5549 | 0.2500 |
| Q1 | W3 | 0.5357 | 0.4870 | 0.3167 | 0.6574 | 0.4299 | 0.2563 |
| Q1 | W4 | 0.6488 | 0.5570 | 0.2909 | 0.8230 | 0.5763 | 0.2267 |
| Q2 | W1 | 0.7083 | 0.5516 | 0.2381 | 0.8651 | 0.6447 | 0.1923 |
| Q2 | W2 | 0.5655 | 0.5673 | 0.4118 | 0.7229 | 0.6320 | 0.2417 |
| Q2 | W3 | 0.5595 | 0.5056 | 0.3167 | 0.6944 | 0.4724 | 0.2514 |
| Q2 | W4 | 0.6012 | 0.4936 | 0.1818 | 0.8053 | 0.5532 | 0.2332 |

## H1–H3 (primary diagnostic)
| Arm | Scope | Raw | Balanced | +Recall | -Recall | AUC | Brier |
|---|---|---|---|---|---|---|---|
| Q0 | H1 | 0.5000 | 0.4714 | 0.3529 | 0.5899 | 0.4919 | 0.2534 |
| Q0 | H2 | 0.6607 | 0.5545 | 0.2714 | 0.8377 | 0.5751 | 0.2196 |
| Q0 | H3 | 0.6071 | 0.5719 | 0.4138 | 0.7299 | 0.6136 | 0.2324 |
| Q1 | H1 | 0.5000 | 0.4714 | 0.3529 | 0.5899 | 0.4914 | 0.2534 |
| Q1 | H2 | 0.6652 | 0.5578 | 0.2714 | 0.8442 | 0.5779 | 0.2192 |
| Q1 | H3 | 0.6027 | 0.5661 | 0.4023 | 0.7299 | 0.6139 | 0.2325 |
| Q2 | H1 | 0.5179 | 0.4813 | 0.3294 | 0.6331 | 0.4870 | 0.2546 |
| Q2 | H2 | 0.6920 | 0.5578 | 0.2000 | 0.9156 | 0.6617 | 0.2026 |
| Q2 | H3 | 0.6161 | 0.5708 | 0.3678 | 0.7737 | 0.6111 | 0.2318 |

## Paired day-cluster bootstrap (10,000 draws, seed 20260924)
| Pair | ΔRaw | Raw CI low | Raw CI high | Raw W/T/L | ΔBalanced | Balanced CI low | Balanced CI high |
|---|---|---|---|---|---|---|---|
| Q1-Q0 | 0.0000 | -0.0045 | 0.0045 | 1/26/1 | -0.0002 | -0.0033 | 0.0027 |
| Q2-Q0 | 0.0193 | -0.0193 | 0.0595 | 11/5/12 | 0.0092 | -0.0307 | 0.0479 |
| Q2-Q1 | 0.0193 | -0.0208 | 0.0625 | 11/4/13 | 0.0094 | -0.0312 | 0.0482 |

Q2−Q1 is included. No ranking is based on benchmark-day figures or pooled Raw alone.

## Slot gains vs Q0
| Arm | Scope | correct arm | correct Q0 | Δslots | slots | Δslots/day |
|---|---|---|---|---|---|---|
| Q1 | overall | 396 | 396 | 0 | 672 | 0.0000 |
| Q1 | H1 | 112 | 112 | 0 | 224 | 0.0000 |
| Q1 | H2 | 149 | 148 | 1 | 224 | 0.0357 |
| Q1 | H3 | 135 | 136 | -1 | 224 | -0.0357 |
| Q2 | overall | 409 | 396 | 13 | 672 | 0.4643 |
| Q2 | H1 | 116 | 112 | 4 | 224 | 0.1429 |
| Q2 | H2 | 155 | 148 | 7 | 224 | 0.2500 |
| Q2 | H3 | 138 | 136 | 2 | 224 | 0.0714 |

## Safety anchor: Q0/F80
| Arm | Admissible | Checks |
|---|---|---|
| Q0 | True | {'balanced': True, 'positive_recall': True, 'one_class_days': True, 'positive_recall_zero_days': True} |
| Q1 | True | {'balanced': True, 'positive_recall': True, 'one_class_days': True, 'positive_recall_zero_days': True} |
| Q2 | True | {'balanced': True, 'positive_recall': True, 'one_class_days': True, 'positive_recall_zero_days': True} |

## Readout diagnostics / parameter counts
| Arm | mode | shared head | n segment-bias | n segment-heads | params |
|---|---|---|---:|---:|---:|
| Q0 | `shared` | True | 0 | 0 | 382,724 |
| Q1 | `segment_bias` | True | 3 | 0 | 382,727 |
| Q2 | `segment_heads` | False | 0 | 3 | 383,252 |

Q2 per-head parameter/gradient L2 norms (selected-checkpoint probe, mean over 28 runs):

| head | param norm | grad norm |
|---|---:|---:|
| H1 | 1.798 | 0.0760 |
| H2 | 1.606 | 0.1039 |
| H3 | 1.712 | 0.0820 |

Q1 selected-run segment-bias means: H1 +0.0075, H2 +0.0055, H3 -0.0172; per-run trajectory in `readout_diagnostics.json`.

## Runtime (formal run means)
| Arm | External wall s | Train wall s | Epoch s | Best epoch | Stop epoch | CUDA peak MiB | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q0 | 14.99 | 12.06 | 0.491 | 2.11 | 17.11 | 371.8 | 382,724 |
| Q1 | 19.80 | 15.70 | 0.630 | 2.11 | 17.11 | 371.8 | 382,727 |
| Q2 | 19.35 | 15.63 | 0.635 | 2.43 | 17.43 | 371.8 | 383,252 |

## Signal / decision
`HETEROGENEITY_PROMISING`

Q1 Raw +0.00 pp (paired CI [-0.45,+0.45]pp), Q2 Raw +1.93 pp (paired CI [-1.93,+5.95]pp). Safety: Q0=True, Q1=True, Q2=True. 62/63/65 is not claimed unless the actual 28-day metric reaches it with acceptable safety.

## Provenance and scope
Run commands, metrics and manifests are recorded under `runs/`. Run manifests bind source/config/selector/sequence hashes; `runtime.csv` and `readout_diagnostics.csv/json` hold per-run telemetry and the readout audit. No 24-head design, new segment boundaries, month heads, oracle routing, threshold/calibration tuning, PLE bins/dim, k/depth/width, role filtering, gate/fusion sweep, Temporal/FFT change, Stage B, broader DEV or lockbox was run.

Tests: canonical non-destructive baseline **94 passed**; full suite (canonical + all prior E2 + focused E2-E1) **174 passed** (+9 focused). Fail-closed CLI combinations return exit 2. Non-blocking single-bin PLE and matplotlib warnings only.
