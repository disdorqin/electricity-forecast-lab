# E2-C2 — Horizon Gate Necessity & Simplification

**Screening-only; STOP after this experiment.** G0 reuses C0/F80; G1/G2 are fresh CUDA+AMP runs (28 formal DEV days each). Only Direction's horizon gate changes. Fixed Direction fusion alpha=.8; Magnitude retains canonical `H_current`.

## Gate A / frozen protocol
- Current fixed .8 and default/no-flag reproduced the G0/F80 benchmark checkpoint; max prediction delta 0.0 (<=1e-9).
- Source/config/selector/sequence hashes remain the frozen values in each run manifest; target and eligibility continue through the docs/01 origin=D-1 14:00, labels<=D-2 pipeline.
- G1 uses fixed .8 mix with no Direction gate parameter; G2 uses one scalar initialized .8. L_dir does not reach the canonical horizon gate in either; G2 scalar receives L_dir; Temporal remains live; L_mag reaches canonical gate in all arms. See `gradient_ownership.json`.
- k=8 preserved; fresh arm records: 28/28 PASS each; all CUDA+AMP.

## Benchmark day — engineering only
| Arm | Raw | Balanced | +Recall | -Recall | AUC | Brier | ppf |
|---|---|---|---|---|---|---|---|
| G0 | 0.5833 | 0.5378 | 0.4286 | 0.6471 | 0.5630 | 0.2364 | 0.3750 |
| G1 | 0.5833 | 0.5378 | 0.4286 | 0.6471 | 0.5630 | 0.2366 | 0.3750 |
| G2 | 0.5833 | 0.5378 | 0.4286 | 0.6471 | 0.5630 | 0.2367 | 0.3750 |

2026-02-13 is excluded from the formal panel and gate ranking.

## Overall formal DEV (28 days / 672 slots)
| Arm | Raw | Balanced | +Recall | -Recall | AUC | Brier | ppf | one-class days | +R=0 days | -R=0 days | min W Raw | W Raw std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G0 | 0.5893 | 0.5372 | 0.3512 | 0.7233 | 0.5671 | 0.2351 | 0.3036 | 3 | 7 | 1 | 0.5298 | 0.0566 |
| G1 | 0.5685 | 0.5201 | 0.3471 | 0.6930 | 0.5571 | 0.2374 | 0.3214 | 3 | 7 | 1 | 0.4881 | 0.0650 |
| G2 | 0.5804 | 0.5294 | 0.3471 | 0.7116 | 0.5663 | 0.2350 | 0.3095 | 3 | 7 | 1 | 0.5238 | 0.0523 |

## W1–W4
| Arm | Window | Raw | Balanced | +Recall | -Recall | AUC | Brier |
|---|---|---|---|---|---|---|---|
| G0 | W1 | 0.6488 | 0.5040 | 0.2143 | 0.7937 | 0.5782 | 0.2074 |
| G0 | W2 | 0.5298 | 0.5303 | 0.4824 | 0.5783 | 0.5544 | 0.2500 |
| G0 | W3 | 0.5357 | 0.4870 | 0.3167 | 0.6574 | 0.4292 | 0.2562 |
| G0 | W4 | 0.6429 | 0.5525 | 0.2909 | 0.8142 | 0.5746 | 0.2271 |
| G1 | W1 | 0.6488 | 0.5040 | 0.2143 | 0.7937 | 0.5784 | 0.2075 |
| G1 | W2 | 0.5238 | 0.5243 | 0.4824 | 0.5663 | 0.5546 | 0.2500 |
| G1 | W3 | 0.4881 | 0.4500 | 0.3167 | 0.5833 | 0.3972 | 0.2657 |
| G1 | W4 | 0.6131 | 0.5257 | 0.2727 | 0.7788 | 0.5792 | 0.2266 |
| G2 | W1 | 0.6488 | 0.5040 | 0.2143 | 0.7937 | 0.5782 | 0.2075 |
| G2 | W2 | 0.5238 | 0.5243 | 0.4824 | 0.5663 | 0.5546 | 0.2500 |
| G2 | W3 | 0.5357 | 0.4870 | 0.3167 | 0.6574 | 0.4293 | 0.2561 |
| G2 | W4 | 0.6131 | 0.5257 | 0.2727 | 0.7788 | 0.5791 | 0.2266 |

## H1–H3
| Arm | Segment | Raw | Balanced | +Recall | -Recall | AUC | Brier |
|---|---|---|---|---|---|---|---|
| G0 | H1 | 0.5000 | 0.4714 | 0.3529 | 0.5899 | 0.4919 | 0.2534 |
| G0 | H2 | 0.6607 | 0.5545 | 0.2714 | 0.8377 | 0.5751 | 0.2196 |
| G0 | H3 | 0.6071 | 0.5719 | 0.4138 | 0.7299 | 0.6136 | 0.2324 |
| G1 | H1 | 0.4777 | 0.4534 | 0.3529 | 0.5540 | 0.4808 | 0.2556 |
| G1 | H2 | 0.6339 | 0.5312 | 0.2571 | 0.8052 | 0.5690 | 0.2211 |
| G1 | H3 | 0.5938 | 0.5609 | 0.4138 | 0.7080 | 0.5994 | 0.2355 |
| G2 | H1 | 0.4911 | 0.4642 | 0.3529 | 0.5755 | 0.4933 | 0.2523 |
| G2 | H2 | 0.6429 | 0.5377 | 0.2571 | 0.8182 | 0.5708 | 0.2203 |
| G2 | H3 | 0.6071 | 0.5719 | 0.4138 | 0.7299 | 0.6152 | 0.2324 |

## Paired day-cluster bootstrap (10,000 draws, seed 20260924)
| Pair | ΔRaw | Raw CI low | Raw CI high | Raw W/T/L | ΔBalanced | Balanced CI low | Balanced CI high | Window Raw signs [W1..W4] |
|---|---|---|---|---|---|---|---|---|
| G1-G0 | -0.0208 | -0.0506 | 0.0000 | 0/25/3 | -0.0152 | -0.0361 | 0.0000 | [0.0, -1.0, -1.0, -1.0] |
| G2-G0 | -0.0089 | -0.0253 | 0.0000 | 0/26/2 | -0.0084 | -0.0241 | 0.0000 | [0.0, -1.0, 0.0, -1.0] |
| G1-G2 | -0.0119 | -0.0357 | 0.0000 | 0/27/1 | -0.0068 | -0.0204 | 0.0000 | [0.0, 0.0, -1.0, 0.0] |

G1−G2 is descriptive. No ranking is based on benchmark-day figures or pooled Raw alone.

## Safety anchor: G0/F80
| Arm | Admissible | Checks |
|---|---|---|
| G0 | True | {'balanced': True, 'positive_recall': True, 'one_class_days': True, 'positive_recall_zero_days': True} |
| G1 | False | {'balanced': False, 'positive_recall': True, 'one_class_days': True, 'positive_recall_zero_days': True} |
| G2 | True | {'balanced': True, 'positive_recall': True, 'one_class_days': True, 'positive_recall_zero_days': True} |

## Gate/component diagnostics
G0's selected-checkpoint 24-value gate is retained by day in `gate_diagnostics.csv`. G1's Direction gate is fixed at .8; G2's one scalar trajectory (init/final/min/max/delta per run) and selected scalar are recorded in the same table.

- G0: selected gate mean/range 0.80114 / 0.78797–0.80938; gated Strong norm 142.489, gated Weak norm 8.756, effective Weak fraction 6.5414%, Strong/Weak cosine 0.2068.
- G1: selected gate mean/range 0.80000 / 0.80000–0.80000; gated Strong norm 140.051, gated Weak norm 8.823, effective Weak fraction 6.7746%, Strong/Weak cosine 0.2066.
- G2: selected gate mean/range 0.80171 / 0.80004–0.80322; gated Strong norm 141.246, gated Weak norm 8.683, effective Weak fraction 6.6290%, Strong/Weak cosine 0.2106.
- G2 scalar across selected runs: mean 0.80171, range 0.80004–0.80322; trajectory initialized .8, final mean/range 0.81785 / 0.81386–0.82181, mean delta +0.01785, trajectory extrema 0.79997–0.82181.

Gradient L2 norms (real benchmark selected checkpoints):
| Arm | Loss | Strong/TabM | Weak MLP | canonical gate | Temporal | global scalar |
|---|---|---:|---:|---:|---:|---:|
| G0 | L_dir | 0.315952 | 0.024832 | 0.006212 | 0.797467 | 0.000000 |
| G0 | L_mag | 0.216593 | 0.022272 | 0.001221 | 0.021492 | 0.000000 |
| G1 | L_dir | 0.315769 | 0.024686 | 0.000000 | 0.797728 | 0.000000 |
| G1 | L_mag | 0.216630 | 0.022242 | 0.001227 | 0.021495 | 0.000000 |
| G2 | L_dir | 0.316330 | 0.024373 | 0.000000 | 0.797856 | 0.001813 |
| G2 | L_mag | 0.216301 | 0.022259 | 0.001224 | 0.021498 | 0.000000 |

## Runtime (formal run means)
| Arm | External wall s | Train wall s | Epoch s | Best epoch | Stop epoch | CUDA peak MiB | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | 14.99 | 12.06 | 0.491 | 2.11 | 17.11 | 371.8 | 382,724 |
| G1 | 18.21 | 15.15 | 0.622 | 2.07 | 17.07 | 375.4 | 382,724 |
| G2 | 20.84 | 17.56 | 0.729 | 2.00 | 17.00 | 375.4 | 382,725 |

## Signal / decision
`NO_CLEAR_GATE_EFFECT`

G1 Raw is -2.08 pp vs G0 and fails Balanced safety; G2 is within 2 pp Raw of G0 and is safety-admissible, but paired intervals include zero and the preregistered evidence does not establish the required superiority/equivalence pattern. Signal remains `NO_CLEAR_GATE_EFFECT`; do not promote or alter production architecture.
Descriptive reused endpoint G100 (E2-C1 Strong-only): Raw 0.5744, Balanced 0.5211; not retrained or included in this gate.

## Provenance and scope
Run commands, metrics and manifests are recorded under `runs/`. Run manifests bind source/config/selector/sequence hashes; `runtime.csv` contains per-run timing/device/epoch/memory telemetry. No threshold/calibration, alpha sweep, selector/model-width/depth/k change, Stage B, expanded DEV or lockbox was run.

Tests: canonical non-destructive suite plus previous E2 tests and focused E2-C2 routing tests: **156 passed**. Fail-closed CLI combinations were also checked. Non-blocking single-bin PLE and matplotlib deprecation warnings only.
