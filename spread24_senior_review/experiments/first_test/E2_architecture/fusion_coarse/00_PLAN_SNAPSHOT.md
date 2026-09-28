# E2-B1 — Direction Fusion Coarse Map: plan snapshot

Frozen before any formal (non-benchmark) result was read. Written from
`experiments/first_test/12_E2_B1_FUSION_COARSE_PLAN.md` and `13_CODEX_PROMPT_E2_B1.md`.

## What is being varied, and what is not

One axis only: the Direction fusion coefficient

    H_dir = alpha_time * time_dir(H_time) + (1 - alpha_time) * tab_dir(H_tab)

`alpha_time` is pinned to a constant for the Direction fusion. Magnitude fusion
(`H_mag = alpha_mag * tab_mag + (1 - alpha_mag) * time_mag`), the parameter set, the TabM member
axis, the encoders and every training hyper-parameter are untouched. `alpha` is **not** read as a
literal contribution percentage — see §10 below; that is the point of the diagnostics.

Implementation surface (experiment-only, canonical defaults unchanged):

| file | change |
| --- | --- |
| `src/TafM_改进源码/models/task_adapters.py` | `direction_fusion_alpha` argument + validation + endpoint short-circuit |
| `src/TafM_改进源码/models/dual_branch_v21.py` | pass-through |
| `src/TafM_改进源码/train.py` | `direction_fusion_alpha` threaded to `_model_for` / `train_target_day`; run-dir suffix `_fa{NNN}`; manifest field |
| `src/run_tabm_v21.py` | `--direction-fusion-alpha FLOAT`, default `None` |

`direction_fusion_alpha=None` preserves the canonical learnable fusion exactly. A non-`None` value
is rejected unless `architecture_mode=full_current`, and rejected outright under `legacy_v20`.

At the endpoints the module **short-circuits** rather than multiplying by zero, so `alpha=0.0`
produces structurally the same graph as `architecture_mode=tabular_only` (the temporal branch is
absent from the Direction graph, not present with a zero gradient):

    alpha=0.0 -> h_dir = tab_dir(H_tab)
    alpha=1.0 -> h_dir = time_dir(H_time)
    0<a<1     -> h_dir = a*td + (1-a)*sd

## Arms

| arm | alpha_time | source |
| --- | --- | --- |
| F00 | 0.0 fixed | reused E2-A A1 (`tabular_only`) |
| F20 | 0.2 fixed | fresh, 28 days |
| F40 | 0.4 fixed | fresh, 28 days |
| F60 | 0.6 fixed | fresh, 28 days |
| F80 | 0.8 fixed | fresh, 28 days |
| F100 | 1.0 fixed | reused E2-A A2 (`temporal_only`) |
| FL08 | learnable, init 0.8 | reused E2-A A0 (E1-M2 records) — **SAFETY ANCHOR** |

F80 and FL08 are **different arms even though they coincide at initialisation**: the fixed arm's
coefficient cannot move, the learnable arm's can (and in fact does — see the diagnostics).

## Frozen protocol

`dir_only` objective / `direction_first` checkpoint policy / `vanilla` gradient policy /
default profile / mode A2 / Stage A only / Stage B OFF / seed 20260924 / k=8 / max 120 epochs /
patience 15 / batch 64 / frozen LR-WD / CUDA AMP.
Panel: W1 2026-02-12..02-18, W2 2026-04-12..04-18, W3 2026-06-12..06-18, W4 2026-08-07..08-13
(28 target days, read from `E1_mini/preflight_target_days.csv` so the panel cannot drift).
Fresh work: 4 arms × 28 days = **112 fresh formal runs**. Plus 6 benchmark-day runs for Gate A.

## Gates, defined before running

**Gate A (routing audit).** `alpha=0` must reproduce the E2-A `tabular_only` benchmark run and
`alpha=1` the `temporal_only` run — compared as artifacts (`predictions.parquet` SHA256, element-wise
max |delta| over every shared numeric column, every scalar in `metrics.json`), required `<= 1e-9`.
A fixed coefficient must receive no update (bit-constant across every epoch). Endpoint gradient
ownership must be correct and the coefficient must receive no Direction gradient. Finite / shape /
leakage PASS. Canonical non-destructive suite ≥ 94 PASS plus the new tests. Formal and default
behaviour unchanged.

**Gate B (benchmark day).** Day 2026-02-13, engineering only. Never used to rank or eliminate arms.

**Gate C (formal).** A1/A2-equivalent endpoints reused; F20/F40/F60/F80 each run all 28 days.
No stopping on intermediate window results.

## Safety anchor and admissibility (plan §9)

SAFETY_ADMISSIBLE, measured against FL08:

- Balanced ≥ FL08 − 0.01
- +Recall ≥ FL08 − 0.05
- one-class days ≤ FL08 + 2
- +Recall = 0 days ≤ FL08 + 2

Non-admissible arms are always reported, never hidden. Raw accuracy is the primary metric but is
never read without this filter.

## Effective fusion diagnostics (plan §8)

`alpha` alone is not interpretable. For every formal run, on a deterministic diagnostic batch:

    norm_tab  = || (1 - alpha) * tab_dir(H_tab) ||
    norm_time = || alpha * time_dir(H_time) ||
    effective_time_norm_fraction = norm_time / (norm_tab + norm_time + eps)
    cosine(tab_component, time_component)

Observation only — no gradients, no ranking, no routing decision taken from these numbers.

## FUSION_REGION decision rule (encoded before results)

`best` = highest-Raw arm among SAFETY_ADMISSIBLE arms, ties broken by Balanced then AUC.

1. no admissible arm → `NO_CLEAR_REGION`
2. `best` = FL08 → `CURRENT_FUSION_OK`
3. `best` = F00 → `ENDPOINT_TABULAR`
4. otherwise, with `alpha = alpha(best)`:
   - `alpha <= 0.4` → `TABULAR_HEAVY`
   - `0.4 < alpha < 0.7` → `BALANCED_MIX`
   - `alpha >= 0.7` → `TEMPORAL_HEAVY`
   - but if `best` is **not** separated (≥ MATERIAL = 0.02 Raw) from every admissible arm in a
     different region **and** its paired day-cluster CI vs FL08 includes zero, then
     `NO_CLEAR_REGION` — uncertainty prevents a conclusion.

## Forbidden in this round

PLE/FFT/hidden/depth/k changes, Strong/Weak ablation, concat/late fusion, attention/MMoE/router,
threshold/calibration, checkpoint tolerance, protected-gradient, Stage B, full DEV, lockbox,
oracle routing, day/month oracle stitching. No stage, no commit.
