# FIRST_TEST Experiment Tracker

> 运行前先填 command/config；运行后补 raw_run_dir、metrics、耗时和结论。  
> INVALID/FAIL 不删除，修复后新建 Run ID。

## Latest status — 2026-09-26

- E2-C1 / E2-C2 / E2-C3 architecture screening completed. Latest gate: E2-C3 `NO_CLEAR_ROLE_EFFECT`; see `E2_architecture/strong_role_routing/E2_C3_summary.md`.
- E2-C3 R1 failed its safety gate; R2 was safety-admissible but did not establish a clear paired/cross-window effect. No feature, selector, or canonical V2.1 changes were promoted.
- STOP for human review. E2-C4 and any broader DEV/lockbox run are not authorized by this result.
- The E0/E1 table and gate below remain historical records of that separate direction-first experiment track; they do not describe the latest E2 architecture-screening status.

| Run ID | Gate | Purpose | Target / Window | Variant | Seed | Status | Raw | Bal | +R | -R | Mag MAE | Best/Stop Epoch | Wall Time | Raw Run Dir | Decision / Notes |
|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|---|---|
| P0-SMOKE | P0 | experiment routing + telemetry sanity | 2026-06-15 | JOINT / V2.1 guardrail / vanilla / smoke | 20260924 | PASS | 0.4166666666666667 | 0.23809523809523808 | 0.0 | 0.47619047619047616 | 53.16171348467469 | 2/2 | 7.370624600007432 | D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_lab\spread24_senior_review\src\TafM_改进源码\outputs\tabm_v21\experiments\direction_first\smoke\2026-06-15_joint_v21g_van_20260925T070311677052Z | DIAGNOSTIC_ONLY; no ranking/promotion |
| P0-GPU | P0 | CPU vs GPU timing sanity counterpart | 2026-06-15 | JOINT / V2.1 guardrail / vanilla / smoke / epf-2 CUDA | 20260924 | PASS | 0.375 | 0.21428571428571427 | 0.0 | 0.42857142857142855 | 54.72487265616655 | 2/2 | 6.104165800003102 | D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_lab\spread24_senior_review\src\TafM_改进源码\outputs\tabm_v21\experiments\direction_first\smoke\2026-06-15_joint_v21g_van_20260925T070410762629Z | DIAGNOSTIC_ONLY; no ranking/promotion |
| E0-D1 | E0 | formal convergence/runtime pilot | 2026-02-15 | JOINT / Direction-first / vanilla | 20260924 | PASS | 0.7083333333333334 | 0.5 | 0.0 | 1.0 | 44.034741462518774 | 2/17 | 11.403657300019404 | D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_lab\spread24_senior_review\src\TafM_改进源码\outputs\tabm_v21\experiments\direction_first\checkpoint_policy\2026-02-15_joint_dirfirst_van_20260925T070746181500Z | DIAGNOSTIC_ONLY; no ranking/promotion |
| E0-D2 | E0 | formal convergence/runtime pilot | 2026-04-15 | JOINT / Direction-first / vanilla | 20260924 | PASS | 0.5416666666666666 | 0.7105263157894737 | 0.42105263157894735 | 1.0 | 22.67681337396304 | 1/16 | 12.346242600004189 | D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_lab\spread24_senior_review\src\TafM_改进源码\outputs\tabm_v21\experiments\direction_first\checkpoint_policy\2026-04-15_joint_dirfirst_van_20260925T070815887089Z | DIAGNOSTIC_ONLY; no ranking/promotion |
| E0-D3 | E0 | formal convergence/runtime pilot | 2026-06-15 | JOINT / Direction-first / vanilla | 20260924 | PASS | 0.4166666666666667 | 0.23809523809523808 | 0.0 | 0.47619047619047616 | 49.807508697112404 | 4/19 | 13.99004779997631 | D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_lab\spread24_senior_review\src\TafM_改进源码\outputs\tabm_v21\experiments\direction_first\checkpoint_policy\2026-06-15_joint_dirfirst_van_20260925T070844194588Z | DIAGNOSTIC_ONLY; no ranking/promotion |
| E0-D4 | E0 | formal convergence/runtime pilot | 2026-08-10 | JOINT / Direction-first / vanilla | 20260924 | PASS | 0.4166666666666667 | 0.22727272727272727 | 0.45454545454545453 | 0.0 | 44.793136827647686 | 3/18 | 12.772393599996576 | D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_lab\spread24_senior_review\src\TafM_改进源码\outputs\tabm_v21\experiments\direction_first\checkpoint_policy\2026-08-10_joint_dirfirst_van_20260925T070914237989Z | DIAGNOSTIC_ONLY; no ranking/promotion |
| E0-G | E0-G | formal gradient conflict diagnosis | 2026-06-15 | JOINT / Direction-first / vanilla + diagnostics | 20260924 | PASS | 0.4166666666666667 | 0.23809523809523808 | 0.0 | 0.47619047619047616 | 49.807508697112404 | 4/19 | 14.353171199996723 | D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_lab\spread24_senior_review\src\TafM_改进源码\outputs\tabm_v21\experiments\direction_first\gradient_diagnostics\2026-06-15_joint_dirfirst_van_20260925T070946042430Z | DIAGNOSTIC_ONLY; no ranking/promotion |

## E0 Gate

- [x] P0 infrastructure PASS
- [x] 94-test non-destructive suite PASS
- [x] E0-D1 complete
- [x] E0-D2 complete
- [x] E0-D3 complete
- [x] E0-D4 complete
- [x] E0-G complete
- [x] E0_summary.csv written
- [x] E0_summary.md written
- [x] E0_GATE.md written
- [ ] No E1 run launched
- [ ] Full Jan-Aug DEV NOT RUN
- [ ] Lockbox NOT TOUCHED

## Future E1-mini matrix — DO NOT RUN BEFORE E0 REVIEW

| Variant | Objective | Checkpoint | Gradient | Target Days | Seed |
|---|---|---|---|---:|---:|
| M0 | JOINT | V2.1 guardrail | vanilla | 28 | 20260924 |
| M1 | JOINT | Direction-first | vanilla | 28 | 20260924 |
| M2 | DIR-only | Direction-first | vanilla | 28 | 20260924 |

