# 87 — Codex Prompt: MSR-V2 GPU Core + Component Adjudication

Use DevSpace.

Project root:

`D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_lab\spread24_senior_review`

You are the **MSR-V2 GPU Scientific Experiment Director**.

Do not redesign the architecture before obtaining evidence. Do not trust historical completion tokens. Use current disk state, current tests, current environment, and actual run artifacts as truth.

## Mandatory reading, in order

1. `AGENTS.md`
2. `docs/82_msr_daily_asof_training_and_fast24_protocol_20261009.md`
3. `docs/84_msr_v2_native96_unified_architecture_implementation_spec_20261009.md`
4. `docs/86_msr_v2_gpu_core_and_component_adjudication_protocol_20261009.md`
5. `src/research96/msr_spread_v2/README.md`
6. current V2 implementation, especially `config.py`, `dataset.py`, `spectral_router.py`, `low_branch.py`, `high_branch.py`, `dependence.py`, `losses.py`, `trainer.py`, `walkforward.py`, `evaluate.py`, `artifacts.py` and tests.

`docs/86` is the experiment authority. `docs/84` remains the architecture authority. `docs/82` remains the DAILY_ASOF training/data authority.

## Hard safety / isolation rules

- NO production writes to `electricity_forecast_model2.5`.
- NO edits to `src/research96/msr_spread_v1/`.
- NO edits to `src/TafM_改进源码/`.
- NO Git stage/commit/push.
- NO late-September truth and NO October lockbox access.
- Snapshot V2 core hashes before experiment work.
- Ablations must live under an experiment-specific package such as `src/research96/msr_spread_v2/experiments/component_adjudication/` whenever possible.
- Do not mutate FULL_V2 core behavior to make an ablation easier. If a tiny hook is unavoidable, default FULL behavior must remain unchanged and regression-tested.

## Phase G0 — GPU qualification FIRST

The machine has an NVIDIA RTX 4060-class GPU, but at protocol creation time:

`D:\computer_download\environment\conda\epf-2\python.exe`

had numpy/pandas/pyarrow/sklearn but **no torch and no tabm**.

The user explicitly authorizes GPU setup for this experiment.

1. Persist environment-before receipts and `nvidia-smi` under:
   `outputs/experiments/msr_spread_v2/gpu_qualification_20261009/`.
2. If necessary, install a CUDA-enabled PyTorch build compatible with the installed driver and install `tabm` into `epf-2` only.
3. Do not silently downgrade/remove the existing scientific stack. Recheck imports for numpy/pandas/pyarrow/sklearn/pywt/lightgbm/tabm/torch after setup.
4. Require:
   - `torch.cuda.is_available() == True`
   - GPU name recorded
   - torch version and CUDA runtime recorded
5. Test GPU batch sizes `64`, then `32`; freeze the first safe one for all scientific arms.
6. Run all existing V2 tests after environment qualification.
7. If safe GPU setup fails, output `MSR_V2_GPU_ENV_BLOCKED` and stop. Do not quietly switch the expensive protocol to CPU.

## Phase G1 — implement frozen GPU scientific profile

Implement only what docs/86 authorizes.

Scientific profile:

- Stage A max 40, min 15 epochs;
- Stage B max 12, min 5;
- Stage C max 30, min 10;
- patience 7 after the minimum floor;
- chronological 28-day inner validation;
- exact refit on all mature rows with selected epoch counts;
- DAILY_ASOF fresh fit, labels <= D-2;
- deterministic seed 20261009;
- M_TRAIN 32 / M_EVAL 256;
- same profile for every compared arm.

Do not convert this into a hyperparameter sweep.

Add tests proving minimum-epoch-floor semantics and that the scientific profile is identical across arms except for the explicitly ablated component.

## Phase G2 — runtime benchmark

Run FULL_V2 scientific GPU training on exactly:

- `2026-01-15`
- `2026-04-01`
- `2026-07-15`

Record complete per-stage selection/refit timing, selected epochs, peak VRAM, and total/day.

Decision:

- median <= 75 s/day -> use FULL-30 primary panel;
- median > 75 s/day -> use frozen FAST-24 primary panel.

FULL-30 = FAST-24 plus:

`2026-01-31, 2026-02-28, 2026-03-01, 2026-04-30, 2026-07-31, 2026-08-14`.

Do not choose dates based on model metrics.

## Phase E1 — FULL_V2 primary scientific run

Run current FULL_V2 on the selected primary panel.

Persist all docs/86 final/component/gate/dependence/optimization diagnostics.

Do not interpret the model using only Direction or only MAE.

## Phase E2 — DIRECT96 control

Implement an isolated `DIRECT96` experiment baseline with approximately comparable capacity and the exact same legal Native96 data, preprocessing, DA contract, D-1 price proxy, TCN+TabM scale, monotone quantile output, and final CRPS+Brier+point objective.

DIRECT96 predicts final Spread directly and does NOT use:

- T24 Low/High factorization;
- Low/High reconstruction;
- dynamic Low/High dependence.

Run DIRECT96 on the same primary panel.

This is mandatory. The main question is whether the Low+High architecture earns its complexity.

## Phase E3 — component attribution on DIAG-12

Fixed DIAG-12:

`2026-01-05, 2026-01-25, 2026-02-15, 2026-03-05, 2026-03-31, 2026-04-01, 2026-04-25, 2026-05-15, 2026-06-15, 2026-07-05, 2026-07-25, 2026-08-13`.

Run these arms with the exact same GPU scientific profile:

- FULL_V2
- NO_DYNAMIC_DEP: rho=0, phiL=0, phiH=0
- FIXED_BUSINESS_GATES: fixed initialized T24-like complementary family gates, no learned daily cutoff adaptation
- NO_LOW_TO_HIGH_CONTEXT: High gets no Low distribution/latent Low context
- NO_BRIER0: only the final Brier@0 coefficient is zero

Do not add State, Markov, U, extra backbones, rolling windows, new loss weights, or extra feature families.

## Phase E4 — conditional promotion

Promote a DIAG-12 ablation to the full primary panel only if its absolute effect versus FULL_V2 meets at least one docs/86 threshold:

- Balanced >= 1.5 pp
- Direction >= 1.5 pp
- MAE >= 3%
- CRPS >= 3%
- Brier >= 0.01
- or clear structural failure/inactivity

Promotion is symmetric: removal helping or hurting both count.

## Mandatory diagnostics

Final:

- Spread MAE/RMSE
- RT MAE
- Direction
- Balanced Accuracy
- positive/negative recall
- AUC
- Brier@0
- CRPS
- positive prevalence calibration
- quantile coverage/width

Components:

- Low MAE/CRPS/correlation by month
- High MAE/CRPS/correlation by month

Business gates:

- family cutoff/tau/delta
- distance from initialization
- pairwise family gate distance
- day-to-day variation
- explicit `GATE_INACTIVE` adjudication if appropriate

Dependence:

- rho / phiL / phiH ranges, means, stds, smoothness
- FULL vs NO_DYNAMIC_DEP effect
- explicit `DEPENDENCE_INACTIVE` adjudication if appropriate

Optimization:

- selected A/B/C epochs
- min/max-bound hit rates
- gradient norms
- final-vs-component gradient cosine
- any underfit or gradient-conflict signal

## Required outputs

Root:

`outputs/experiments/msr_spread_v2/v2_gpu_component_adjudication_20261009/`

Must contain:

- environment before/after receipts
- GPU qualification receipt
- runtime benchmark
- FULL_V2 and DIRECT96 primary runs
- DIAG-12 ablation runs
- promoted full-panel ablations if triggered
- `COMPONENT_COMPARISON.csv`
- `GATE_DIAGNOSTICS.csv/json`
- `DEPENDENCE_DIAGNOSTICS.csv/json`
- `OPTIMIZATION_DIAGNOSTICS.csv/json`
- `ADJUDICATION.md`
- `ADJUDICATION.json`
- zero-pollution/source-hash receipts

## Final scientific judgment

Do not simply say PASS/FAIL. Use the decision map in docs/86.

The final line must be exactly one of:

`MSR_V2_CORE_SCIENTIFICALLY_PROMISING`

`MSR_V2_CORE_MIXED_NEEDS_TARGETED_REDESIGN`

`MSR_V2_CORE_STRUCTURALLY_DOMINATED`

`MSR_V2_EXPERIMENT_BLOCKED`

Before that line, print a compact machine-readable summary including:

- GPU_ENV
- GPU_DEVICE
- BATCH_SIZE
- MEDIAN_RUNTIME_PER_DAY
- PRIMARY_PANEL_DAYS
- FULL_V2_MAE / DIRECTION / BALANCED / CRPS / BRIER
- DIRECT96_MAE / DIRECTION / BALANCED / CRPS / BRIER
- FULL_MINUS_DIRECT deltas
- LOW_CORR / HIGH_CORR
- GATE_STATUS
- DEPENDENCE_STATUS
- MOST_INFORMATIVE_ABLATION
- ARCHITECTURE_DIAGNOSIS

If FULL_V2 is poor, do not redesign the architecture during this run. Diagnose which module is implicated and stop with the evidence package so the next architecture decision is made from results rather than improvisation.
