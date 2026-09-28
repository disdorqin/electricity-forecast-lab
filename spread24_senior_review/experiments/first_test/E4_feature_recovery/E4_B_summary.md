# E4-B — Frozen Selector Reopening via Weak Feature Recovery

STATUS=AUTO_FINAL_REPORT  PARENT=E4-A  ANCHOR=E2-E1_Q2_SEGMENT_HEADS  DATE=2026-09-26

## 1. FILES_CHANGED

- src/TafM_改进源码/train.py  (+feature_recovery_profile param; +build_experiment_feature_manifest; in-memory experiment manifest; fail-closed)
- src/run_tabm_v21.py  (+--feature-recovery-profile CLI flag routed to train_target_day)
- src/TafM_改进源码/tests/test_e4_b_feature_recovery.py  (new focused contract tests)
- experiments/first_test/E4_feature_recovery/run_e4_b.py / analyze_e4_b.py / gate_a_e4_b.py / write_summary_e4_b.py / benchmark_e4_b.py / probe_e4_b.py / family_diag_e4_b.py

## 2. TESTS

- Focused E4-B contract tests: `src/TafM_改进源码/tests/test_e4_b_feature_recovery.py` (Gate A 1-13, 18-20).
- Gate A (20 items): `benchmark/gate_a_results.json` + `benchmark/gate_a_tests.json` + `benchmark/gate_a_failclosed.txt`.
- Code-level checks include identical eligible-index SETS for all 28 formal days across F0/F1/F2.

## 3. FEATURE_PROFILE_AUDIT

| arm | feature_recovery_profile | selected_feature_count | strong_count | weak_count | recovered_count | recovered_role |
| --- | --- | --- | --- | --- | --- | --- |
| F0 | selected222 | 222 | 211 | 11 | 0 |  |
| F1 | literature240 | 240 | 211 | 29 | 18 | Weak |
| F2 | all259 | 259 | 211 | 48 | 37 | Weak |


## 4. ELIGIBILITY_IDENTITY

- `store.eligibility` returns **identical eligible-index SETS** for selected222 / literature240 / all259 on **28/28** formal target days (Gate A item 13). Recovered features carry no quarantine on any in-window day, so widening the input set does not change the training/eval day set.

## 5. BENCHMARK (2026-02-13 engineering only)

| arm | feature_recovery_profile | raw | balanced | auc | brier | parameter_count | best_epoch | stop_epoch | selected_feature_count | strong_count | weak_count | recovered_count | wall_seconds | device | amp | experiment_feature_profile_sha256 | selector_sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F0 | selected222 (reuse Q2) | 0.5833 |  |  |  | 383252 | 2 | 17 | 222 | 211 | 11 | 0 | 17.239 | cuda | True |  | ed348cfd9fd911bc675d7fd920485b1a748c159a3e0a395b3af02af01015092f |
| F1 | literature240 | 0.5833 |  |  |  | 406436 | 3 | 18 | 240 | 211 | 29 | 18 | 24.059 | cuda | True | 1dc70d3a7b9c1c98136e6cc5b4303542d1594a2226c2ce92f92657332a972385 | ed348cfd9fd911bc675d7fd920485b1a748c159a3e0a395b3af02af01015092f |
| F2 | all259 | 0.6250 |  |  |  | 430908 | 5 | 20 | 259 | 211 | 48 | 37 | 33.696 | cuda | True | 32e9f2dd4800e3c2e199f9c03f20133d10f73f2f61a1f2526db3b96ad0f6576e | ed348cfd9fd911bc675d7fd920485b1a748c159a3e0a395b3af02af01015092f |

- F0 reuses the E2-E1 Q2 run (selected222 == frozen selector). F1/F2 are fresh deep Q2 trainings under the canonical frozen-split route with only the experiment feature-recovery profile changed. No ranking is derived from the benchmark.

## 6. F0 / F1 / F2 OVERALL

| arm | raw | balanced | positive_recall | negative_recall | auc | brier | min_window_raw | window_raw_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F0 | 0.6086 | 0.5424 | 0.3058 | 0.7791 | 0.5911 | 0.2297 | 0.5595 | 0.0597 |
| F1 | 0.5923 | 0.5269 | 0.2934 | 0.7605 | 0.5826 | 0.2344 | 0.5238 | 0.0697 |
| F2 | 0.6012 | 0.5366 | 0.3058 | 0.7674 | 0.5700 | 0.2347 | 0.5476 | 0.0541 |


## 7. W1-W4 WINDOW METRICS (Raw)

| scope | F0 | F1 | F2 |
| --- | --- | --- | --- |
| W1 | 0.7083 | 0.7083 | 0.6845 |
| W2 | 0.5655 | 0.5595 | 0.5595 |
| W3 | 0.5595 | 0.5238 | 0.5476 |
| W4 | 0.6012 | 0.5774 | 0.6131 |


## 8. H1-H3 HOUR-SEGMENT METRICS (Raw)

| scope | F0 | F1 | F2 |
| --- | --- | --- | --- |
| H1 | 0.5179 | 0.5045 | 0.4955 |
| H2 | 0.6920 | 0.6830 | 0.6875 |
| H3 | 0.6161 | 0.5893 | 0.6205 |


## 9. PAIRED COMPARISONS (day-cluster bootstrap 10k, seed 20260924)

| comparison | mean_delta_raw | raw_ci_low | raw_ci_high | raw_ci_excludes_zero | raw_W_T_L |
| --- | --- | --- | --- | --- | --- |
| F1-F0 | -0.0164 | -0.0446 | 0.0089 | False | 9/10/9 |
| F2-F0 | -0.0074 | -0.0327 | 0.0208 | False | 6/11/11 |
| F1-F2 | -0.0089 | -0.0402 | 0.0193 | False | 11/8/9 |


## 10. SLOT GAINS vs F0 (correct slots)

| arm | delta_correct_slots | delta_slots_per_day |
| --- | --- | --- |
| F0 | 0 | 0.00 |
| F1 | -11 | -0.39 |
| F2 | -5 | -0.18 |


## 11. SAFETY ADMISSIBILITY (anchor F0)

- F0: admissible=True checks={'balanced': True, 'positive_recall': True, 'one_class_days': True, 'positive_recall_zero_days': True}
- F1: admissible=False checks={'balanced': False, 'positive_recall': True, 'one_class_days': True, 'positive_recall_zero_days': True}
- F2: admissible=True checks={'balanced': True, 'positive_recall': True, 'one_class_days': True, 'positive_recall_zero_days': True}


## 12. FAMILY DIAGNOSTICS (F1 recovered literature features)

| group | n_features | literature_features | perm_mean_abs_dp | perm_flip_fraction | occ_mean_abs_dp | occ_flip_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| A_renewable_adjusted_error | 8 | fcast_renewable_adjusted_direct_load, err_renewable_adjusted_direct_28d_mean, err_renewable_adjusted_direct_28d_std, err_renewable_adjusted_direct_28d_q10, err_renewable_adjusted_direct_28d_q90, delta_renewable_adjusted_direct_28d_mean, delta_renewable_adjusted_direct_28d_std, delta_renewable_adjusted_direct_28d_q90 | 0.0017 | 0.0045 | 0.0018 | 0.0015 |
| B_ramp2 | 4 | ramp2_load, ramp2_solar, ramp2_renewable, ramp2_residual_load | 0.0034 | 0.0074 | 0.0025 | 0.0074 |
| C_regime_flags | 6 | regime_high_residual_load_renew, regime_low_residual_load_renew, regime_high_renewable_share, regime_low_renewable_share, regime_high_bidding_space_ratio, regime_low_bidding_space_ratio | 0.0002 | 0.0000 | 0.0002 | 0.0000 |

Permutation = shuffle the group's 24 slot values (per column); occlusion = zero the group columns. Higher |Δp_positive| / flip fraction ⇒ stronger leverage of the recovered feature group on F1's own target-day inputs.

## 13. COMPONENT / PARAMETER / RUNTIME

| arm | parameter_count | strong_count | weak_count | weak_mlp_input_dim | best_epoch |
| --- | --- | --- | --- | --- | --- |
| F0 | 383252 |  |  |  | 2.4 |
| F1 | 406436 | 211.0 | 29.0 | 29.0 | 2.5 |
| F2 | 430908 | 211.0 | 48.0 | 48.0 | 2.9 |


## 14. E4_B_GATE FEATURE_RECOVERY_SIGNAL


**E4_B_GATE FEATURE_RECOVERY_SIGNAL = `RECOVERY_HARMFUL`**

## 15. NEXT ACCELERATION RULE

Recovered features materially worsen Q2 or fail safety. Next = E5 recurrent-regime / similar-day sample selection or weighting.

## 16. UNEXECUTED LATER WORK

- E5 (if E4-B fails): recurrent-regime / similar-day sample selection or weighting using legal target-day fundamental state; NOT recency windows or model micro-tuning.

- If RECOVERY_PROMISING_UNPROVEN: confirmation must include 3 seeds + matched-size random recovery controls from the remaining Noise pool before promotion.

- If BROAD_SELECTOR_BOTTLENECK: next family attribution on the remaining 19 Noise features.

- No Stage B, no threshold tuning, no alpha/gate/PLE/k/depth/width changes, no new raw source, no per-day oracle feature profile.
