# E2-C1 — Tabular Strong/Weak Big-Block Attribution

**STATUS=COMPLETE — screening evidence only; stop for human review.**  
Formal panel: 28 fixed DEV target days × 24 hours per arm. C0 is the reused E2-B1 F80 checkpoint set; C1/C2 are fresh CUDA+AMP runs. All use the frozen protocol, with only Direction's tabular source changed for C1/C2. No target/config/selector/leakage/training-axis change was made.

## Routing, freeze, and Gate A

- Selector roles: Strong-BOTH 147, Strong-DIR 32, Strong-MAG 22, Forced-Core 10, Weak 11; Strong/Core total 211 and Weak 11. Full feature names/counts: `benchmark/selector_role_inventory.json`.
- `TabularEncoder.forward_components()` exposes `(H_strong, H_weak, gate, H_current)`; `forward()` returns the original gated `H_current`. Direction receives current/strong-only/broadcast-weak-only; Magnitude always receives `H_current`. No member/head axis was collapsed.
- C0 fixed `.8` inference reproduced the saved F80 checkpoint: 13 numeric prediction columns and 143 numeric metric leaves compared, max absolute delta **0.0**. Default/no routing flag on the same checkpoint also matched exactly. See `benchmark/gate_a_exactness.json`.
- Config, selector, source-slot, and sequence-manifest hashes stayed fixed across arms: `8c981156…66b5488c`, `ed348cfd…15092f`, `a1b86f95…8968a349ea`, `9144bbed…e768badfd35ce7a82`. All formal runs used A2/default/dir_only/direction_first/vanilla/Stage-A/fixed-alpha .8/seed 20260924/CUDA+AMP; source and selector hashes are repeated per run in `runtime.csv`.
- Focused tests prove current gated formula exactness, C1/C2 gradient ownership, finite output, and retained k=8 axis. Benchmark-checkpoint autograd evidence is in `gradient_ownership.json`.

## Benchmark day — engineering only

2026-02-13 was excluded from all ranking and inference. C0/C1/C2 each had Raw **0.5833**, Balanced **0.5378**, +Recall **0.4286**, −Recall **0.6471**, and predicted-positive fraction **0.3750**. AUC was 0.5630/0.5630/0.5882 and Brier 0.2364/0.2371/0.2413. See `benchmark/benchmark_metrics.csv`; runtime/epoch/VRAM data is in `runtime.csv`.

## Overall formal panel (672 slots, 28 days)

| Arm | Raw | Balanced | +Recall | −Recall | AUC | Brier | ppf | one-class days | +Recall=0 days | −Recall=0 days | min-window Raw | window Raw std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 CURRENT_TABULAR | .5893 | .5372 | .3512 | .7233 | .5671 | .2351 | .3036 | 3 | 7 | 1 | .5298 | .0566 |
| C1 STRONG_ONLY | .5744 | .5211 | .3306 | .7116 | .5479 | .2387 | .3036 | 3 | 8 | 1 | .4940 | .0666 |
| C2 WEAK_ONLY | .5298 | .5043 | .4132 | .5953 | .5212 | .2545 | .4077 | 0 | 3 | 1 | .4702 | .0544 |

True positive prevalence was .3601 for every arm. Use Balanced/recalls and collapse counters alongside Raw; do not read pooled Raw alone as stable.

## W1–W4 (pooled slots per fixed 7-day window)

| Arm/window | Raw | Balanced | +Recall | −Recall | AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|
| C0 W1 | .6488 | .5040 | .2143 | .7937 | .5782 | .2074 |
| C0 W2 | .5298 | .5303 | .4824 | .5783 | .5544 | .2500 |
| C0 W3 | .5357 | .4870 | .3167 | .6574 | .4292 | .2562 |
| C0 W4 | .6429 | .5525 | .2909 | .8142 | .5746 | .2271 |
| C1 W1 | .6488 | .5040 | .2143 | .7937 | .5809 | .2064 |
| C1 W2 | .5238 | .5246 | .4588 | .5904 | .5519 | .2510 |
| C1 W3 | .4940 | .4546 | .3167 | .5926 | .3914 | .2660 |
| C1 W4 | .6310 | .5297 | .2364 | .8230 | .5529 | .2315 |
| C2 W1 | .6071 | .5000 | .2857 | .7143 | .5291 | .2306 |
| C2 W2 | .4881 | .4889 | .4235 | .5542 | .4982 | .2579 |
| C2 W3 | .4702 | .4769 | .5000 | .4537 | .4546 | .2732 |
| C2 W4 | .5536 | .5142 | .4000 | .6283 | .5416 | .2564 |

## H1–H3 (pooled hours across 28 days)

| Arm/segment | Raw | Balanced | +Recall | −Recall | AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|
| C0 H1 | .5000 | .4714 | .3529 | .5899 | .4919 | .2534 |
| C0 H2 | .6607 | .5545 | .2714 | .8377 | .5751 | .2196 |
| C0 H3 | .6071 | .5719 | .4138 | .7299 | .6136 | .2324 |
| C1 H1 | .4866 | .4606 | .3529 | .5683 | .4750 | .2563 |
| C1 H2 | .6652 | .5578 | .2714 | .8442 | .5811 | .2177 |
| C1 H3 | .5714 | .5322 | .3563 | .7080 | .5592 | .2421 |
| C2 H1 | .4509 | .4547 | .4706 | .4388 | .4532 | .2726 |
| C2 H2 | .5982 | .5325 | .3571 | .7078 | .5324 | .2441 |
| C2 H3 | .5402 | .5150 | .4023 | .6277 | .5579 | .2469 |

## Paired day-cluster bootstrap (10,000 draws, seed 20260924)

| Pair | Mean ΔRaw | 95% CI | W/T/L Raw | Mean ΔBalanced | Balanced 95% CI | CI excludes zero? | Windows same sign |
|---|---:|---:|---:|---:|---:|---|---:|
| C1−C0 | −.0149 | [−.0402, .0015] | 2/21/5 | −.0105 | [−.0259, .0007] | No | 3/4 |
| C2−C0 | −.0595 | [−.1146, −.0089] | 9/4/15 | −.0226 | [−.0659, .0208] | Raw yes; Balanced no | 4/4 |

C2 is stably lower than C0 on Raw by the frozen paired rule. C1's −1.49 pp mean Raw difference is below the 2 pp materiality threshold and its CI includes zero. Full paired comparisons, including C0−C1/C2 and C1−C2, are in `paired_summary.csv` and window signs in `analysis_summary.json`.

## Safety anchor (C0/F80)

| Arm | Balanced Δ | +Recall Δ | one-class Δ | +Recall=0 Δ | Admissible |
|---|---:|---:|---:|---:|---|
| C0 | .0000 | .0000 | 0 | 0 | Yes (anchor) |
| C1 | −.0161 | −.0207 | 0 | +1 | **No** (Balanced fails the −.01 bound) |
| C2 | −.0330 | +.0620 | −3 | −4 | **No** (Balanced fails the −.01 bound) |

## Horizon gate and component diagnostics (reused C0 selected checkpoints)

- 28 F80 selected checkpoints × 24 horizon values are recorded in `gate_values_by_run.csv` (672 rows). Across those values: mean gate **.80114** (init .8), global min/max **.78797/.80938**, mean movement **+.00114**, movement range **−.01203..+.00938**.
- By segment, mean gate / mean movement / observed value range: H1 **.80306 / +.00306 / .79503–.80938**; H2 **.80027 / +.00027 / .78797–.80465**; H3 **.80008 / +.00008 / .79016–.80538**.
- Across deterministic target-day diagnostic batches: `||g·H_strong||` mean **142.49**; `||(1−g)·H_weak||` mean **8.76**; effective Weak norm fraction mean **6.54%** (range 2.42–12.28%); Strong vs broadcast-Weak cosine mean **.2064** (range −.0164–.3373). These are explanatory diagnostics, not routing/oracle inputs.
- Real benchmark-checkpoint gradient norms (L2; autograd):

| Arm | Loss | Strong/TabM | Weak MLP | horizon gate | Temporal |
|---|---|---:|---:|---:|---:|
| C0 | L_dir | .31595 | .02483 | .006212 | .79747 |
| C1 | L_dir | .35043 | 0 (blocked) | 0 (blocked) | .79858 |
| C2 | L_dir | 0 (blocked) | .10099 | 0 (blocked) | .85545 |
| C0 | L_mag | .21659 | .02227 | .001221 | .02149 |
| C1 | L_mag | .20384 | .01894 | .001367 | .02146 |
| C2 | L_mag | .11244 | .02399 | .000519 | .02340 |

Thus L_dir ownership matches C1/C2 intent, Temporal remains live at fixed alpha .8, and L_mag reaches canonical Strong+Weak+gate in all arms. Full per-parameter counts/norms and Boolean Gate A checks are in `gradient_ownership.json`.

## Runtime

Formal per-run means, 28 runs each; all fresh C1/C2 runs used CUDA+AMP. C0 runtime is reused F80 telemetry.

| Arm | external wall/run s | train wall/run s | epoch s | best epoch | stop epoch | CUDA peak MB | parameters |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 14.99 | 12.06 | .491 | 2.11 | 17.11 | 371.8 | 382,724 |
| C1 | 20.86 | 17.47 | .705 | 2.11 | 17.11 | 364.3 | 382,724 |
| C2 | 17.72 | 14.63 | .582 | 2.32 | 17.32 | 296.0 | 382,724 |

## E2-C1 gate

**E2_C1_GATE TABULAR_SIGNAL = `INCONCLUSIVE`**.

C2 is stably below C0 on paired Raw, but both C1 and C2 fail the frozen safety screen on Balanced. C1 is within 2 pp of C0 on pooled Raw, yet is not admissible and its paired CI includes zero; the pre-registered criteria therefore do not establish `STRONG_DOMINANT` or `WEAK_RESIDUAL`. The evidence is not a promotion or a production result.

## Provenance, tests, and scope

- C0=28 reused F80; C1=28 fresh; C2=28 fresh; benchmark=1 reused + 2 fresh engineering runs. Formal record: `runs/FORMAL_RUNS.json`; artifacts remain in their run directories under `src/TafM_改进源码/outputs/tabm_v21/...` and are linked by each record.
- Gate A exactness and gradient checks PASS. Canonical non-destructive suite plus E2-A/B1 and new focused tests: **154 passed**. One pytest attempt hit the managed Windows temp permission boundary; rerun with workspace-local `TEMP/TMP` passed. Non-blocking official PLE single-bin warnings are unchanged.
- Unexecuted: any post-E2-C1 study/promotion, alpha refinement, other selector-role split, PLE/TabM/Temporal changes, Stage B, full Jan–Aug DEV, and lockbox.
