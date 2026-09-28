# E4-A — Regime-Conditioned Direction Probability Stacking (plan snapshot)

STATUS=AUTHORIZED_EXECUTED
DATE=2026-09-26
PARENT=E3-A
EXPERIMENT_ROOT=experiments/first_test/E4_regime_stacker
ARCHITECTURE_CARRIER=E2-E1 Q2 SEGMENT_HEADS
PRIMARY_GOAL=CROSS_MONTH_DIRECTION_65_PLUS

## Frozen evidence
Q2 SEGMENT_HEADS: Raw .6086 / Balanced .5424 / AUC .5911 / safety PASS / +13-672 slots vs shared head.
E3-A: T0 .6086, T1 .5685, T2 .5506, T3 .5357 — all recency arms materially worse; RECENCY_SIGNAL=NO_RECENCY_GAIN.
Therefore: no more window sweeps, no architecture micro-ablation; keep Q2 as the base predictor; next test is a
low-capacity regime-aware probability correction.

## Hypothesis
Q2 has useful ranking information but its probability/decision mapping is not stable across market regimes.
A low-capacity, independently fitted, regime-conditioned logistic correction can convert that signal into
more accurate Direction decisions without changing the deep model.

## Anti-leakage / anti-double-dip design (three-way chronological split)
For each target day D:
1. canonical Stage-A split unchanged: eligible <= D-2; BASE = older 80%; FULL_MONITOR = newest 20%.
2. CALIBRATOR = newest 90 days of FULL_MONITOR (ends at D-2).
3. CHECKPOINT_MONITOR = the remaining older FULL_MONITOR days (>=60).
4. BASE unchanged; preprocessor fits BASE only; CHECKPOINT_MONITOR alone selects the Q2 checkpoint;
   the stacker is fitted on CALIBRATOR only. Target-day truth never influences anything.

Implemented as experiment-only `e4_three_way_split` (helper `e4a_three_way_split`,
`E4_CALIBRATOR_DAYS=90`, `E4_CHECKPOINT_MONITOR_MIN_DAYS=60`); defaults keep the canonical 80/20
bit-identical. `direction_postprocess_mode != none` now REQUIRES the three-way split, so the WIP path
that refitted the checkpoint-selection monitor (double dip) is impossible.

## Arms (same deep checkpoint per day)
- P0 ORIGINAL_Q2 — reuse E2-E1 Q2; external anchor.
- P1 SPLIT_CONTROL — fresh 28 deep Q2 runs under the three-way split, no postprocessing.
- P2 SEGMENT_LOGIT — fitted on CALIBRATOR only with 5 meta features: base_logit, H2, H3, base_logit×H2,
  base_logit×H3 (H1 reference).
- P3 REGIME_LOGIT — P2 features plus the 9 pre-registered already-selected legal features:
  residual_load_renew, renewable_share, bidding_space_ratio, net_ramp_pressure, err_net_load_28d_std,
  uncert_风电总加_width, uncert_光伏总加_width, ctx_spread_positive_rate14,
  spread_same_slot_28d_positive_rate (14 total).

## Stacker contract
sigmoid(Xw+b); unweighted BCE; L2=1e-3 (applied to all non-anchor coefficients so the base_logit anchor
is not shrunk); max_iter=100; deterministic LBFGS; base_logit init 1, all others 0, bias 0; threshold .5.
No class weighting, no threshold tuning, no L2 search, no feature search, no calibrator-days sweep.

## Efficiency
Exactly 28 fresh deep Q2 trainings (P1). P2/P3 are derived from the SAME checkpoint
(`derive_e4_a.py`); checkpoint SHA256 identity is recorded in `checkpoint_identity.csv`.
Engine: `run_e4_a.py` (deep), `derive_e4_a.py` (P2/P3 + audits), `gate_a_e4_a.py`, `analyze_e4_a.py`.

## Frozen protocol
Benchmark 2026-02-13 (engineering only). Formal DEV W1 02-12..02-18, W2 04-12..04-18, W3 06-12..06-18,
W4 08-07..08-13. P0 reuse; P1 fresh 28; P2/P3 derived. No intermediate stopping. No Stage B.

## Gate A (see benchmark/GATE_A_EVIDENCE.md)
Q2 default reproduces the saved benchmark <=1e-9; BASE equals canonical BASE; FULL_MONITOR equals
canonical monitor; CALIBRATOR = newest 90 ending D-2; CHECKPOINT_MONITOR = the rest with adjacency and no
overlap; preprocessing BASE-only; P1/P2/P3 share one checkpoint SHA256; the in-train postprocess path
reproduces the offline derivation exactly; P2 = 5 features and P3 = 14; the 9 regime features are in the
frozen selector; no target truth in any fit; threshold .5; frozen hashes; segment_heads + k=8 preserved;
canonical 94 + prior E2/E3 + focused E4 tests PASS; incompatible combinations fail closed.

## Forbidden (not run)
selector rerun, new raw source, arbitrary new feature, threshold tuning, class weighting, L2 search,
calibrator-days sweep, Stage B, architecture change, PLE/k/depth/width/gate/fusion changes, 24-head model,
target-day oracle routing, broader DEV/lockbox.
