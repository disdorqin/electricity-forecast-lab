# 86 — MSR-V2 GPU Core + Component Adjudication Protocol (2026-10-09)

STATUS: CURRENT_EXPERIMENT_AUTHORITY

This document does **not** replace docs/82 or docs/84. It freezes the first scientific experiment protocol after the V2 core implementation passed engineering audit.

- Training/data authority: `docs/82_msr_daily_asof_training_and_fast24_protocol_20261009.md`.
- Architecture authority: `docs/84_msr_v2_native96_unified_architecture_implementation_spec_20261009.md`.
- Implementation prompt/history: docs/85.
- Current V2 core package: `src/research96/msr_spread_v2/`.
- V1, frozen TafM and production remain read-only.

The purpose of this round is **not hyperparameter tuning**. It is to answer, in order:

1. Can the intended dozens-of-epochs V2 training run efficiently on GPU?
2. Does the complete V2 core beat a direct Native-96 probabilistic baseline?
3. Which structural components actually add value: learned spectral routing, Low->High conditioning, dynamic dependence, and the Direction-focused Brier term?
4. If the full model fails, where is the bottleneck: Low, High, routing, dependence, or final objective?

No late-September truth and no October lockbox may be opened.

---

## 1. Environment / GPU qualification gate

The workstation has an NVIDIA RTX 4060-class GPU and current `nvidia-smi` reports a usable driver. At the time of this protocol, `D:\computer_download\environment\conda\epf-2\python.exe` contains the scientific data stack but **does not currently contain torch or tabm**. Therefore the execution agent must not assume CUDA readiness.

### G0-A. Snapshot before environment work

Persist under:

`outputs/experiments/msr_spread_v2/gpu_qualification_20261009/`

at least:

- `nvidia-smi` output;
- `conda list -p D:\computer_download\environment\conda\epf-2`;
- Python version;
- imports/version receipts for numpy/pandas/pyarrow/sklearn/pywt/lightgbm;
- current torch/tabm availability.

### G0-B. CUDA torch setup is authorized

The user explicitly authorizes GPU use for this experiment. If `epf-2` lacks torch/tabm, the agent may install a CUDA-enabled PyTorch build compatible with the installed NVIDIA driver and install `tabm` into **epf-2 only**.

Rules:

- do not modify project source merely to make CUDA import;
- do not remove/downgrade the existing scientific stack unless a hard dependency conflict is proven;
- after installation re-run imports for numpy/pandas/pyarrow/sklearn/pywt/lightgbm/tabm/torch;
- require `torch.cuda.is_available()==True` before scientific GPU execution;
- persist exact versions, CUDA runtime, GPU name, and package changes;
- if safe CUDA setup cannot be achieved, stop with `MSR_V2_GPU_ENV_BLOCKED`; do not silently run the expensive scientific panel on CPU.

### G0-C. GPU batch qualification

Test batch sizes in this fixed order: `64 -> 32`.

Use the first value that:

- completes forward/backward Stage A/B/C smoke;
- has no OOM/nonfinite loss;
- leaves a reasonable VRAM margin;
- preserves 26/26 V2 tests.

Once selected, freeze the batch size for all arms in this protocol. Do not use batch size as a model-selection axis.

---

## 2. Scientific training profile — V2_GPU_SCIENTIFIC_V1

The 1/1/1 `fast-screen` profile is engineering-only and is **not** admissible for this round.

For every target day D:

`Train(D) = 2023-01-01 .. D-2`, fresh fit, no previous-day warm-start.

Freeze one GPU training profile for all compared arms:

- Stage A: max 40 epochs, minimum 15 before early stopping is allowed;
- Stage B: max 12 epochs, minimum 5;
- Stage C: max 30 epochs, minimum 10;
- patience: 7 after the minimum epoch floor;
- chronological inner validation: latest 28 mature days;
- after epoch selection, exact refit on all mature training days using the selected epoch counts;
- AdamW / learning rates remain docs/84 defaults unless code correctness requires a fix;
- gradient clip 1.0;
- deterministic seed 20261009;
- `M_TRAIN=32`, `M_EVAL=256` remain frozen;
- no loss-weight grid search.

This guarantees that the model receives **dozens of optimization epochs** while retaining chronological early stopping.

Persist per day:

- selected A/B/C epochs;
- selection/refit wall time;
- train/validation loss histories;
- gradient norms/cosines;
- peak VRAM where available.

---

## 3. Date panel and adaptive runtime rule

### 3.1 Mandatory FAST-24 panel

Use the already frozen docs/82 dates:

- Jan: 01-05, 01-15, 01-25
- Feb: 02-05, 02-15, 02-25
- Mar: 03-05, 03-15, 03-31
- Apr: 04-01, 04-15, 04-25
- May: 05-05, 05-15, 05-25
- Jun: 06-05, 06-15, 06-25
- Jul: 07-05, 07-15, 07-25
- Aug: 08-03, 08-08, 08-13

These dates are retrospective consumed development evidence, not a fresh test set.

### 3.2 Optional FULL-30 extension

If the median full scientific GPU wall time measured on `2026-01-15`, `2026-04-01`, and `2026-07-15` is <= 75 seconds/day, extend the primary panel to 30 target days by adding the following six pre-registered dates:

`2026-01-31, 2026-02-28, 2026-03-01, 2026-04-30, 2026-07-31, 2026-08-14`.

If median wall time is >75 seconds/day, stay at FAST-24. Do not change dates based on observed accuracy.

### 3.3 DIAG-12 attribution subset

Component ablations run first on this fixed 12-day subset:

`2026-01-05, 2026-01-25, 2026-02-15, 2026-03-05, 2026-03-31, 2026-04-01, 2026-04-25, 2026-05-15, 2026-06-15, 2026-07-05, 2026-07-25, 2026-08-13`.

The subset intentionally covers winter, spring transition, the March->April reversal, early summer and difficult July/August conditions.

---

## 4. Experiment order — do not begin with a combinatorial ablation sweep

### E1 — Core model qualification

Run `FULL_V2` first on the primary panel using `V2_GPU_SCIENTIFIC_V1`.

FULL_V2 is exactly docs/84:

- Native96 multivariate input;
- five business-family learnable complementary DCT gates;
- T24 Low/High supervision;
- Low TCN + TabM + monotone quantile function;
- High residual-history TCN + TabM + Low-distribution context;
- smooth dynamic dependence layer;
- exact scenario `S=L+H`;
- unified CRPS + Brier@0 + point + auxiliary Low/High objective;
- no hard State / no Markov / no mandatory U.

Before comparing components, establish that FULL_V2 trains stably across months.

### E2 — Direct Native96 baseline

Implement an **experiment-isolated** `DIRECT96` baseline; do not modify the default FULL_V2 forward path.

DIRECT96 must use the same legal Native96 data, preprocessing, target-day DA, D-1 proxy contract, TCN+TabM capacity scale, monotone quantile head, final CRPS/Brier/point objective, and training profile, but it predicts final Spread distribution directly without T24 Low/High factorization or dependence reconstruction.

Purpose:

`Does the Low+High factorization justify its complexity?`

This is the most important structural control.

Run DIRECT96 on the same primary panel as FULL_V2.

### E3 — Mandatory component attribution on DIAG-12

Use experiment-only wrappers/configuration. The frozen FULL_V2 default path must remain unchanged.

Arms:

1. `FULL_V2` — reference.
2. `NO_DYNAMIC_DEP` — keep both marginal branches, but use independent scenario ranks: `rho=0, phiL=0, phiH=0`. Tests the complete learned dependence layer.
3. `FIXED_BUSINESS_GATES` — freeze each family spectral gate at its initialized T24-like response and remove daily cutoff adaptation; Low/High remain complementary. Tests whether learned business-aware routing adds value over a fixed decomposition prior.
4. `NO_LOW_TO_HIGH_CONTEXT` — High keeps its own residual-history/fast inputs but receives no Low quantile-width/latent Low context. Tests slow->fast conditional coupling.
5. `NO_BRIER0` — identical architecture, set only the final zero-threshold Brier coefficient to zero. Tests whether the Direction-aware proper-score term adds useful signal or merely distorts numerical fit.

Do **not** add State, U, more backbones, new windows, more gates, or new loss weights in this round.

### E4 — Promote only informative ablations

For each DIAG-12 ablation compute deltas against FULL_V2.

Promote an ablation to the full primary panel only if at least one of the following holds:

- absolute Balanced Accuracy difference >= 1.5 percentage points;
- absolute raw Direction difference >= 1.5 pp;
- Spread MAE difference >= 3%;
- CRPS difference >= 3%;
- Brier difference >= 0.01;
- or the arm exposes a clear structural failure (collapse, no gradient, gate/dependence inactivity).

This rule is symmetric: an ablation can be promoted because removing a component helps or because removing it clearly hurts.

---

## 5. Required metrics and diagnostics

### 5.1 Final Spread / RT

Report pooled and per-month:

- Spread MAE / RMSE;
- RT MAE (must equal Spread MAE under full target DA contract within numerical tolerance);
- raw Direction accuracy;
- Balanced Accuracy;
- positive recall / negative recall;
- AUC;
- Brier@0;
- CRPS;
- predicted-positive fraction vs true-positive fraction;
- q10-q90 coverage and width;
- q01-q99 width;
- calibration bins.

### 5.2 Low / High

For each component:

- median MAE;
- marginal CRPS;
- Pearson correlation;
- monthly correlation/MAE;
- sign/crossing diagnostics where meaningful.

A poor final score must not be diagnosed without checking whether Low or High failed first.

### 5.3 Business gates

For each family and target day persist:

- learned cutoff `c_g(D)`;
- `tau_g`, `delta_g`;
- distance from initialized gate curve;
- pairwise gate-curve distances between PRICE / DEMAND / RENEWABLE / SUPPLY_SYSTEM / MARKET_TIGHTNESS;
- across-day std of the cutoff.

Flag `GATE_INACTIVE` if all families remain effectively identical and close to initialization throughout the panel. Do not automatically split WIND/SOLAR in this round; only recommend that redesign if evidence warrants it.

### 5.4 Dynamic dependence

Persist and summarize:

- rho min/max/mean/std;
- phiL min/max/mean/std;
- phiH min/max/mean/std;
- daily curve variation;
- smoothness penalty;
- comparison of FULL_V2 vs NO_DYNAMIC_DEP.

Flag `DEPENDENCE_INACTIVE` if learned curves stay near initialization/neutral and removing the layer has negligible metric effect.

### 5.5 Optimization / gradient health

Report:

- selected epochs A/B/C per day;
- how often selections hit min/max bounds;
- final gradient norm to each business gate, Low temporal/TabM/head, High temporal/TabM/head, dependence network;
- final-vs-component gradient cosine for Low/High groups;
- any persistent gradient conflict or near-zero path.

If most days select exactly the minimum epoch floors and validation is still improving, flag training schedule as underfit; do not interpret component ablations before resolving it.

---

## 6. Decision logic after results

Do not compress everything into one scalar. Use Direction/Balanced and MAE/CRPS jointly.

### Architecture clearly useful

FULL_V2 is a strong candidate if, versus DIRECT96, it has either:

- Balanced/Direction improvement >=2 pp with MAE no worse than 3%, or
- MAE improvement >=5% with Balanced no worse than 1 pp,

and the gain is not isolated to one month.

### Architecture clearly dominated

Treat FULL_V2 as structurally suspect if BOTH occur versus DIRECT96:

- MAE is worse by >5%, and
- Balanced Accuracy is worse by >1 pp,

or if Low/High/reconstruction collapses across multiple months.

### Mixed result

If one business objective improves and the other degrades, retain both models as Pareto points and use E3 diagnostics to localize the tradeoff. Do not tune loss weights immediately.

### Interpretation map

- `DIRECT96 >= FULL_V2` and Low correlation weak -> decomposition/Low representation is the first suspect.
- Low/High individually strong but final weak -> dependence/reconstruction/final objective is the first suspect.
- `NO_DYNAMIC_DEP >= FULL_V2` -> simplify/remove current dependence layer before inventing a more complex copula.
- `FIXED_BUSINESS_GATES >= FULL_V2` -> learned routing is not earning its complexity; retain fixed or redesign gate semantics.
- `NO_LOW_TO_HIGH_CONTEXT >= FULL_V2` -> remove redundant Low conditioning from High.
- `NO_BRIER0` improves MAE but harms Direction materially -> keep Brier as a justified task-alignment term; if it harms both, remove/reduce only in a separately authorized loss study.
- Gate families all identical -> consider finer business gates such as WIND/SOLAR or PRICE_LEVEL/SPREAD_CONTEXT in the next architecture round, not now.

---

## 7. Isolation / implementation rules

- No writes to production `electricity_forecast_model2.5`.
- No edits to V1 or frozen TafM.
- Do not rewrite FULL_V2 core math to implement an ablation.
- Prefer a new experiment package under `src/research96/msr_spread_v2/experiments/component_adjudication/` and experiment-specific configs/wrappers.
- If a tiny hook in V2 core is unavoidable, default behavior must remain bit-for-bit/semantically FULL_V2 and must be covered by a regression test; report every changed core file explicitly.
- Snapshot V2 core hashes before experiment implementation and verify no unintended drift afterward.
- Run all existing 26 V2 tests plus new experiment tests before science runs.
- Every prediction must be sealed before opening target truth.
- No fresh September / October access.

---

## 8. Required artifacts

Root:

`outputs/experiments/msr_spread_v2/v2_gpu_component_adjudication_20261009/`

Required:

- `ENV_BEFORE.txt/json`, `ENV_AFTER.txt/json`, `GPU_QUALIFICATION.json`;
- `RUNTIME_BENCHMARK.json`;
- `FULL_V2/`, `DIRECT96/`, and ablation run roots;
- daily and monthly metric tables;
- `COMPONENT_COMPARISON.csv`;
- `GATE_DIAGNOSTICS.csv/json`;
- `DEPENDENCE_DIAGNOSTICS.csv/json`;
- `OPTIMIZATION_DIAGNOSTICS.csv/json`;
- `ADJUDICATION.md` and machine-readable `ADJUDICATION.json`;
- source/code/config/environment hashes;
- zero-pollution receipt.

The final report must explicitly state one of:

- `MSR_V2_CORE_SCIENTIFICALLY_PROMISING`
- `MSR_V2_CORE_MIXED_NEEDS_TARGETED_REDESIGN`
- `MSR_V2_CORE_STRUCTURALLY_DOMINATED`
- `MSR_V2_EXPERIMENT_BLOCKED`

and must explain the conclusion using final + component + gate + dependence evidence rather than one headline metric.
