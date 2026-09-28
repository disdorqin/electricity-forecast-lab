# E2-C3 — Task-Aware Strong Role Routing

**Screening-only; STOP after this experiment.** R0 reuses C0/F80; R1/R2 are fresh CUDA+AMP runs (28 formal DEV days each). Only the Strong/Core role composition entering the Tabular encoder changes. Direction fusion alpha remains fixed at .8 and the current 24-h horizon gate remains active.

## Gate A / frozen protocol
- Current fixed .8 and default/no-flag reproduced the R0/F80 benchmark checkpoint; max prediction delta 0.0 (<=1e-9).
- Source/config/selector/sequence hashes remain the frozen values in each run manifest; target and eligibility continue through the docs/01 origin=D-1 14:00, labels<=D-2 pipeline.
- R1 removes exactly the 22 Strong-MAG features from the Strong/Core TabM input; R2 removes exactly the 32 Strong-DIR features. The 11 Weak features are unchanged in all arms. Temporal, the current 24-h horizon gate and k=8 remain active. Objective is dir_only, so E2-C3 makes no Magnitude-capability claim.
- k=8 preserved; fresh arm records: 28/28 PASS each; all CUDA+AMP.

## Benchmark day — engineering only
| Arm | Raw | Balanced | +Recall | -Recall | AUC | Brier | ppf |
|---|---|---|---|---|---|---|---|
| R0 | 0.5833 | 0.5378 | 0.4286 | 0.6471 | 0.5630 | 0.2364 | 0.3750 |
| R1 | 0.5417 | 0.4244 | 0.1429 | 0.7059 | 0.4286 | 0.2422 | 0.2500 |
| R2 | 0.5833 | 0.4958 | 0.2857 | 0.7059 | 0.4454 | 0.2844 | 0.2917 |

2026-02-13 is excluded from the formal panel and gate ranking.

## Overall formal DEV (28 days / 672 slots)
| Arm | Raw | Balanced | +Recall | -Recall | AUC | Brier | ppf | one-class days | +R=0 days | -R=0 days | min W Raw | W Raw std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | 0.5893 | 0.5372 | 0.3512 | 0.7233 | 0.5671 | 0.2351 | 0.3036 | 3 | 7 | 1 | 0.5298 | 0.0566 |
| R1 | 0.5759 | 0.5132 | 0.2893 | 0.7372 | 0.5583 | 0.2350 | 0.2723 | 2 | 11 | 1 | 0.5298 | 0.0516 |
| R2 | 0.5923 | 0.5305 | 0.3099 | 0.7512 | 0.5623 | 0.2359 | 0.2708 | 2 | 9 | 0 | 0.4940 | 0.0581 |

## W1–W4
| Arm | Window | Raw | Balanced | +Recall | -Recall | AUC | Brier |
|---|---|---|---|---|---|---|---|
| R0 | W1 | 0.6488 | 0.5040 | 0.2143 | 0.7937 | 0.5782 | 0.2074 |
| R0 | W2 | 0.5298 | 0.5303 | 0.4824 | 0.5783 | 0.5544 | 0.2500 |
| R0 | W3 | 0.5357 | 0.4870 | 0.3167 | 0.6574 | 0.4292 | 0.2562 |
| R0 | W4 | 0.6429 | 0.5525 | 0.2909 | 0.8142 | 0.5746 | 0.2271 |
| R1 | W1 | 0.6548 | 0.5000 | 0.1905 | 0.8095 | 0.5550 | 0.2115 |
| R1 | W2 | 0.5298 | 0.5309 | 0.4353 | 0.6265 | 0.6028 | 0.2413 |
| R1 | W3 | 0.5298 | 0.4713 | 0.2667 | 0.6759 | 0.4165 | 0.2583 |
| R1 | W4 | 0.5893 | 0.4800 | 0.1636 | 0.7965 | 0.5533 | 0.2290 |
| R2 | W1 | 0.6429 | 0.5000 | 0.2143 | 0.7857 | 0.5962 | 0.2057 |
| R2 | W2 | 0.4940 | 0.4955 | 0.3765 | 0.6145 | 0.5563 | 0.2518 |
| R2 | W3 | 0.6071 | 0.5463 | 0.3333 | 0.7593 | 0.4227 | 0.2542 |
| R2 | W4 | 0.6250 | 0.5299 | 0.2545 | 0.8053 | 0.5490 | 0.2318 |

## H1–H3
| Arm | Segment | Raw | Balanced | +Recall | -Recall | AUC | Brier |
|---|---|---|---|---|---|---|---|
| R0 | H1 | 0.5000 | 0.4714 | 0.3529 | 0.5899 | 0.4919 | 0.2534 |
| R0 | H2 | 0.6607 | 0.5545 | 0.2714 | 0.8377 | 0.5751 | 0.2196 |
| R0 | H3 | 0.6071 | 0.5719 | 0.4138 | 0.7299 | 0.6136 | 0.2324 |
| R1 | H1 | 0.4866 | 0.4584 | 0.3412 | 0.5755 | 0.4706 | 0.2557 |
| R1 | H2 | 0.6652 | 0.5188 | 0.1286 | 0.9091 | 0.5945 | 0.2134 |
| R1 | H3 | 0.5759 | 0.5379 | 0.3678 | 0.7080 | 0.5859 | 0.2359 |
| R2 | H1 | 0.5179 | 0.4858 | 0.3529 | 0.6187 | 0.4813 | 0.2531 |
| R2 | H2 | 0.6875 | 0.5740 | 0.2714 | 0.8766 | 0.6356 | 0.2063 |
| R2 | H3 | 0.5714 | 0.5217 | 0.2989 | 0.7445 | 0.5516 | 0.2483 |

## Paired day-cluster bootstrap (10,000 draws, seed 20260924)
| Pair | ΔRaw | Raw CI low | Raw CI high | Raw W/T/L | ΔBalanced | Balanced CI low | Balanced CI high | Window Raw signs [W1..W4] |
|---|---|---|---|---|---|---|---|---|
| R1-R0 | -0.0134 | -0.0417 | 0.0149 | 10/7/11 | -0.0382 | -0.0855 | 0.0028 | [1.0, -1.0, -1.0, -1.0] |
| R2-R0 | 0.0030 | -0.0417 | 0.0461 | 10/10/8 | 0.0066 | -0.0418 | 0.0538 | [-1.0, -1.0, 1.0, -1.0] |
| R1-R2 | -0.0164 | -0.0610 | 0.0283 | 11/6/11 | -0.0449 | -0.1105 | 0.0122 | [1.0, 1.0, -1.0, -1.0] |

R1−R2 is descriptive. No ranking is based on benchmark-day figures or pooled Raw alone.

## Safety anchor: R0/F80
| Arm | Admissible | Checks |
|---|---|---|
| R0 | True | {'balanced': True, 'positive_recall': True, 'one_class_days': True, 'positive_recall_zero_days': True} |
| R1 | False | {'balanced': False, 'positive_recall': False, 'one_class_days': True, 'positive_recall_zero_days': False} |
| R2 | True | {'balanced': True, 'positive_recall': True, 'one_class_days': True, 'positive_recall_zero_days': True} |

## Route diagnostics
- `role_inventory.csv/json` records the exact included/excluded frozen feature names for R0/R1/R2.
- All role profiles retain the current 24-h horizon gate, Temporal pathway and k=8 member axis; E2-C3 introduces no new gate parameter.
- `gradient_ownership.json` and focused tests verify finite outputs and the intended experiment-only routing.
- `gate_diagnostics.csv` is retained only as auxiliary provenance; NaN component fields are not interpreted because component norms were not the E2-C3 experimental axis.

## Role inventory / parameter counts
R0 includes 211 Strong/Core + 11 Weak; R1 includes 189 Strong/Core (147 BOTH + 32 DIR + 10 Core) + 11 Weak; R2 includes 179 Strong/Core (147 BOTH + 22 MAG + 10 Core) + 11 Weak. Exact frozen feature names are in `role_inventory.csv/json`.

## Runtime (formal run means)
| Arm | External wall s | Train wall s | Epoch s | Best epoch | Stop epoch | CUDA peak MiB | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 | 14.99 | 12.06 | 0.491 | 2.11 | 17.11 | 371.8 | 382,724 |
| R1 | 17.41 | 14.51 | 0.569 | 2.04 | 17.04 | 337.6 | 355,796 |
| R2 | 17.62 | 14.69 | 0.565 | 2.50 | 17.50 | 324.7 | 343,556 |

## Signal / decision
`NO_CLEAR_ROLE_EFFECT`

R1 Raw is -1.34 pp vs R0 and fails Balanced safety; R2 is within 2 pp Raw of R0 and is safety-admissible, but paired intervals include zero and the preregistered evidence does not establish the required superiority/equivalence pattern. Signal remains `NO_CLEAR_ROLE_EFFECT`; do not promote or alter production architecture.
Descriptive reused endpoint R100 (E2-C1 Strong-only): Raw 0.5744, Balanced 0.5211; not retrained or included in this gate.

## Provenance and scope
Run commands, metrics and manifests are recorded under `runs/`. Run manifests bind source/config/selector/sequence hashes; `runtime.csv` contains per-run timing/device/epoch/memory telemetry. No threshold/calibration, alpha sweep, selector/model-width/depth/k change, Stage B, expanded DEV or lockbox was run.

Tests: canonical non-destructive suite plus previous E2 tests and focused E2-C3 routing tests: **157 passed**. Fail-closed CLI combinations were also checked. Non-blocking single-bin PLE and matplotlib deprecation warnings only.
