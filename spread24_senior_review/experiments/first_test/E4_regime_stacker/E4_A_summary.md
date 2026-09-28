# E4-A — Regime-Conditioned Direction Probability Stacking

**Screening-only; STOP after this experiment.** Only 28 fresh deep Q2 checkpoints were trained (P1). P2/P3 are fitted on the E4-A CALIBRATOR only and applied to the same checkpoint's target base prediction, so all three arms share one checkpoint per day. P0 reuses E2-E1 Q2. The deep model, canonical encoding, 24-h gate, fixed fusion alpha=.8 and k=8 are unchanged; the Magnitude path is untouched. No Stage B.

## Gate A / three-way split
| day | BASE n | BASE start | BASE end | FULL_MON n | CKPT_MON n | CKPT_MON end | CAL n | CAL start | CAL end | fit start | fit end |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-02-12 (W1) | 1195 | 2022-01-09 | 2025-04-17 | 299 | 209 | 2025-11-12 | 90 | 2025-11-13 | 2026-02-10 | 2022-01-09 | 2025-04-17 |
| 2026-08-13 (W4) | 1340 | 2022-01-09 | 2025-09-09 | 336 | 246 | 2026-05-13 | 90 | 2026-05-14 | 2026-08-11 | 2022-01-09 | 2025-09-09 |

- Default Q2 without E4 flags reproduces the saved Q2 benchmark: max |Δ| = 0.0e+00 (<=1e-9).
- BASE equals the canonical BASE exactly; FULL_MONITOR equals the canonical monitor exactly; CALIBRATOR is the newest 90 FULL_MONITOR days ending D-2; CHECKPOINT_MONITOR is the rest (>=60), adjacent and non-overlapping; preprocessing fits BASE only.
- P1/P2/P3 share one checkpoint: `same_checkpoint_for_P1_P2_P3 = True`; P1 checkpoint reproduction max |Δ| = 0.0e+00.
- In-train postprocess path == offline derivation on 2026-02-13: {"weights_match": true, "post_predictions_match": true, "base_predictions_match": true}.
- P2 uses exactly 5 meta features and P3 exactly 14; the 9 regime features are all in the frozen selected list; no target-day truth enters any fit; threshold stays .5; source/config/selector/sequence frozen; segment_heads + k=8 preserved.

## Deep checkpoint identity / split audit
- 28/28 P1 deep runs PASS (CUDA+AMP). BASE 1195–1340 days; FULL_MONITOR 299–336; CHECKPOINT_MONITOR 209–246; CALIBRATOR 90–90 days. Per-day detail: `split_audit.csv`.
- `checkpoint_identity.csv` records the identical SHA256 used by P1/P2/P3 and the per-day reproduction delta.

## Benchmark day — engineering only
| Arm | Raw | Balanced | +Recall | -Recall | AUC | Brier | ppf |
|---|---|---|---|---|---|---|---|
| P0 | 0.5833 | 0.4958 | 0.2857 | 0.7059 | 0.5378 | 0.2267 | 0.2917 |
| P1 | 0.5833 | 0.4958 | 0.2857 | 0.7059 | 0.5378 | 0.2267 | 0.2917 |
| P2 | 0.6667 | 0.5126 | 0.1429 | 0.8824 | 0.4706 | 0.2306 | 0.1250 |
| P3 | 0.4583 | 0.5756 | 0.8571 | 0.2941 | 0.7059 | 0.2426 | 0.7500 |

2026-02-13 is excluded from the gate ranking.

## Overall formal DEV (28 days / 672 slots)
| Arm | Raw | Balanced | +Recall | -Recall | AUC | Brier | ppf | one-class days | +R=0 days | -R=0 days | min W Raw | W Raw std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P0 | 0.6086 | 0.5424 | 0.3058 | 0.7791 | 0.5911 | 0.2297 | 0.2515 | 2 | 8 | 0 | 0.5595 | 0.0597 |
| P1 | 0.5833 | 0.5281 | 0.3306 | 0.7256 | 0.5837 | 0.2328 | 0.2946 | 1 | 5 | 1 | 0.5179 | 0.0695 |
| P2 | 0.5908 | 0.4815 | 0.0909 | 0.8721 | 0.5531 | 0.2339 | 0.1146 | 11 | 18 | 0 | 0.4762 | 0.0772 |
| P3 | 0.5848 | 0.5274 | 0.3223 | 0.7326 | 0.5277 | 0.2438 | 0.2872 | 7 | 11 | 0 | 0.5476 | 0.0267 |

## W1–W4
| Arm | Scope | Raw | Balanced | +Recall | -Recall | AUC | Brier |
|---|---|---|---|---|---|---|---|
| P0 | W1 | 0.7083 | 0.5516 | 0.2381 | 0.8651 | 0.6447 | 0.1923 |
| P0 | W2 | 0.5655 | 0.5673 | 0.4118 | 0.7229 | 0.6320 | 0.2417 |
| P0 | W3 | 0.5595 | 0.5056 | 0.3167 | 0.6944 | 0.4724 | 0.2514 |
| P0 | W4 | 0.6012 | 0.4936 | 0.1818 | 0.8053 | 0.5532 | 0.2332 |
| P1 | W1 | 0.6964 | 0.5437 | 0.2381 | 0.8492 | 0.6213 | 0.1980 |
| P1 | W2 | 0.5357 | 0.5372 | 0.4118 | 0.6627 | 0.6018 | 0.2441 |
| P1 | W3 | 0.5179 | 0.4843 | 0.3667 | 0.6019 | 0.4562 | 0.2576 |
| P1 | W4 | 0.5833 | 0.4943 | 0.2364 | 0.7522 | 0.5778 | 0.2314 |
| P2 | W1 | 0.6488 | 0.4802 | 0.1429 | 0.8175 | 0.5805 | 0.2148 |
| P2 | W2 | 0.4762 | 0.4815 | 0.0353 | 0.9277 | 0.6296 | 0.2450 |
| P2 | W3 | 0.5655 | 0.4880 | 0.2167 | 0.7593 | 0.4110 | 0.2550 |
| P2 | W4 | 0.6726 | 0.5000 | 0.0000 | 1.0000 | 0.5601 | 0.2206 |
| P3 | W1 | 0.6071 | 0.6587 | 0.7619 | 0.5556 | 0.6640 | 0.2263 |
| P3 | W2 | 0.5476 | 0.5500 | 0.3529 | 0.7470 | 0.5614 | 0.2567 |
| P3 | W3 | 0.5714 | 0.4815 | 0.1667 | 0.7963 | 0.3677 | 0.2598 |
| P3 | W4 | 0.6131 | 0.4837 | 0.1091 | 0.8584 | 0.5167 | 0.2323 |

## H1–H3
| Arm | Scope | Raw | Balanced | +Recall | -Recall | AUC | Brier |
|---|---|---|---|---|---|---|---|
| P0 | H1 | 0.5179 | 0.4813 | 0.3294 | 0.6331 | 0.4870 | 0.2546 |
| P0 | H2 | 0.6920 | 0.5578 | 0.2000 | 0.9156 | 0.6617 | 0.2026 |
| P0 | H3 | 0.6161 | 0.5708 | 0.3678 | 0.7737 | 0.6111 | 0.2318 |
| P1 | H1 | 0.4554 | 0.4332 | 0.3412 | 0.5252 | 0.4186 | 0.2691 |
| P1 | H2 | 0.7054 | 0.5831 | 0.2571 | 0.9091 | 0.6983 | 0.1983 |
| P1 | H3 | 0.5893 | 0.5510 | 0.3793 | 0.7226 | 0.6191 | 0.2310 |
| P2 | H1 | 0.5848 | 0.4918 | 0.1059 | 0.8777 | 0.5001 | 0.2418 |
| P2 | H2 | 0.6652 | 0.4877 | 0.0143 | 0.9610 | 0.5004 | 0.2236 |
| P2 | H3 | 0.5223 | 0.4522 | 0.1379 | 0.7664 | 0.5700 | 0.2362 |
| P3 | H1 | 0.6473 | 0.6153 | 0.4824 | 0.7482 | 0.5617 | 0.2414 |
| P3 | H2 | 0.6116 | 0.4916 | 0.1714 | 0.8117 | 0.5117 | 0.2351 |
| P3 | H3 | 0.4955 | 0.4575 | 0.2874 | 0.6277 | 0.4905 | 0.2549 |

## Paired day-cluster bootstrap (10,000 draws, seed 20260924)
| Pair | ΔRaw | Raw CI low | Raw CI high | Raw W/T/L | ΔBalanced | Balanced CI low | Balanced CI high |
|---|---|---|---|---|---|---|---|
| P1-P0 | -0.0253 | -0.0506 | -0.0045 | 1/22/5 | -0.0137 | -0.0329 | 0.0016 |
| P2-P1 | 0.0074 | -0.0491 | 0.0610 | 13/4/11 | -0.0288 | -0.0860 | 0.0247 |
| P3-P1 | 0.0015 | -0.0655 | 0.0685 | 12/3/13 | 0.0368 | -0.0199 | 0.0962 |
| P2-P0 | -0.0179 | -0.0685 | 0.0312 | 11/4/13 | -0.0425 | -0.0937 | 0.0017 |
| P3-P0 | -0.0238 | -0.0863 | 0.0417 | 12/2/14 | 0.0231 | -0.0314 | 0.0782 |
| P3-P2 | -0.0060 | -0.0685 | 0.0580 | 13/3/12 | 0.0656 | -0.0022 | 0.1367 |

## Slot gains vs P0
| Arm | Scope | correct arm | correct P0 | Δslots | slots | Δslots/day |
|---|---|---|---|---|---|---|
| P0 | overall | 409 | 409 | 0 | 672 | 0.0000 |
| P0 | H1 | 116 | 116 | 0 | 224 | 0.0000 |
| P0 | H2 | 155 | 155 | 0 | 224 | 0.0000 |
| P0 | H3 | 138 | 138 | 0 | 224 | 0.0000 |
| P1 | overall | 392 | 409 | -17 | 672 | -0.6071 |
| P1 | H1 | 102 | 116 | -14 | 224 | -0.5000 |
| P1 | H2 | 158 | 155 | 3 | 224 | 0.1071 |
| P1 | H3 | 132 | 138 | -6 | 224 | -0.2143 |
| P2 | overall | 397 | 409 | -12 | 672 | -0.4286 |
| P2 | H1 | 131 | 116 | 15 | 224 | 0.5357 |
| P2 | H2 | 149 | 155 | -6 | 224 | -0.2143 |
| P2 | H3 | 117 | 138 | -21 | 224 | -0.7500 |
| P3 | overall | 393 | 409 | -16 | 672 | -0.5714 |
| P3 | H1 | 145 | 116 | 29 | 224 | 1.0357 |
| P3 | H2 | 137 | 155 | -18 | 224 | -0.6429 |
| P3 | H3 | 111 | 138 | -27 | 224 | -0.9643 |

## Calibrator diagnostics (CALIBRATOR only, 90 days / 2160 slots, mean over 28 runs)
| Arm | Scope | Raw | Balanced | +Recall | -Recall | AUC | Brier |
|---|---|---|---|---|---|---|---|
| P2 | calibrator_post | 0.5891 | 0.5240 | 0.1380 | 0.9099 | 0.5713 | 0.2380 |
| P2_base | calibrator_base | 0.5643 | 0.5291 | 0.3162 | 0.7420 | 0.5489 | 0.2463 |
| P3 | calibrator_post | 0.6109 | 0.5636 | 0.3027 | 0.8245 | 0.6205 | 0.2310 |
| P3_base | calibrator_base | 0.5643 | 0.5291 | 0.3162 | 0.7420 | 0.5489 | 0.2463 |

Per-day calibrator base/post metrics are in `calibrator_metrics.csv`. In-sample both stackers improve their own calibrator (mean calibrator Raw .5643 -> .5891 for P2 and -> .6109 for P3), but that in-sample gain does not transfer to the 28 formal target days.
## Stacker coefficients across 28 days (sign consistency = share with the majority sign)
| arm | coefficient | mean | std | min | max | sign_consistency |
|---|---|---|---|---|---|---|
| P2 | base_logit | 0.3134 | 0.3032 | -0.1725 | 1.0733 | 0.8929 |
| P2 | H2 | 0.0416 | 0.2580 | -0.5050 | 0.3975 | 0.6429 |
| P2 | H3 | 0.1044 | 0.1120 | -0.0678 | 0.3321 | 0.7857 |
| P2 | base_logit_x_H2 | 0.2335 | 0.3651 | -0.3964 | 0.9466 | 0.6786 |
| P2 | base_logit_x_H3 | 0.1531 | 0.1211 | -0.0614 | 0.4030 | 0.8929 |
| P3 | base_logit | -0.0766 | 0.3124 | -0.9426 | 0.4441 | 0.5000 |
| P3 | H2 | 0.0910 | 0.1033 | -0.0592 | 0.3379 | 0.8214 |
| P3 | H3 | 0.1861 | 0.1896 | -0.0815 | 0.7211 | 0.9286 |
| P3 | base_logit_x_H2 | 0.0948 | 0.3750 | -0.5988 | 0.6242 | 0.6071 |
| P3 | base_logit_x_H3 | 0.2225 | 0.1251 | -0.2424 | 0.4303 | 0.9643 |
| P3 | residual_load_renew | -0.3218 | 0.1154 | -0.5970 | -0.0851 | 1.0000 |
| P3 | renewable_share | -0.1265 | 0.1913 | -0.3635 | 0.2284 | 0.7500 |
| P3 | bidding_space_ratio | 0.3207 | 0.0843 | 0.1500 | 0.5237 | 1.0000 |
| P3 | net_ramp_pressure | 0.0208 | 0.0281 | -0.0253 | 0.0904 | 0.7143 |
| P3 | err_net_load_28d_std | 0.0924 | 0.2085 | -0.3216 | 0.3231 | 0.7500 |
| P3 | uncert_风电总加_width | -0.1240 | 0.0989 | -0.2698 | 0.0615 | 0.8571 |
| P3 | uncert_光伏总加_width | 0.0259 | 0.1321 | -0.1453 | 0.2704 | 0.5714 |
| P3 | ctx_spread_positive_rate14 | 0.2361 | 0.0968 | 0.0678 | 0.4113 | 1.0000 |
| P3 | spread_same_slot_28d_positive_rate | -0.0399 | 0.1129 | -0.1708 | 0.1832 | 0.7500 |

Stacker contract: {"loss": "unweighted BCE", "l2": 0.001, "max_iter": 100, "optimizer": "deterministic LBFGS", "base_logit_init": 1.0, "other_init": 0.0, "bias_init": 0.0}. `stacker_coefficients.csv` holds the per-day coefficients; `stacker_audit.json` holds the contract and identity audit.

## Safety anchor: P0/Q2
| Arm | Admissible | Checks |
|---|---|---|
| P0 | True | {'balanced': True, 'positive_recall': True, 'one_class_days': True, 'positive_recall_zero_days': True} |
| P1 | False | {'balanced': False, 'positive_recall': True, 'one_class_days': True, 'positive_recall_zero_days': True} |
| P2 | False | {'balanced': False, 'positive_recall': False, 'one_class_days': False, 'positive_recall_zero_days': False} |
| P3 | False | {'balanced': False, 'positive_recall': True, 'one_class_days': False, 'positive_recall_zero_days': False} |

## Runtime
| Arm | Deep wall s (mean) | Best epoch | Stop epoch | Params |
|---|---:|---:|---:|---:|
| P1 deep | 24.89 | 2.43 | 17.43 | 383,252 |

Postprocessing (P2/P3 fitting + target application) runs inside `derive_e4_a.py` and is negligible next to the deep run; the in-train equivalent is available via `--direction-postprocess-mode segment_logit|regime_logit`.


## Signal / decision
`SPLIT_COST_DOMINATES`

P1−P0 [-5.06,-0.45]pp, P2−P1 [-4.91,+6.10]pp, P3−P1 [-6.55,+6.85]pp, P2−P0 [-6.85,+3.12]pp, P3−P0 [-8.63,+4.17]pp, P3−P2 [-6.85,+5.80]pp. Best arm = P2. Safety: P0=True, P1=False, P2=False, P3=False. Acceleration trigger (P2/P3 >=.62 Raw with safety) NOT met; no promotion. 
62/63/65 is not claimed unless the actual 28-day metric reaches it with acceptable safety.

## Provenance and scope
Run commands, metrics and manifests are recorded under `runs/`; `benchmark/` holds the engineering-only day and the default-Q2 reproduction. No selector rerun, new raw source, new arbitrary feature, threshold/class-weight/L2/calibrator-days tuning, Stage B, architecture change, PLE/k/depth/width/gate/fusion change, 24-head model, oracle routing, broader DEV or lockbox was run.

Tests: canonical non-destructive baseline **94 passed**; full suite (canonical + prior E2/E3 + focused E4) **198 passed** (11 focused E4-A + 5 E4-B). Fail-closed CLI combinations return exit 2. Non-blocking single-bin PLE and matplotlib warnings only.
