# E0 Pilot Summary

**Scope:** FIRST_TEST P0 + E0 only. All four formal target-day runs are diagnostic single-day observations, not evidence of generalization or promotion. E0-G is a gradient-diagnostic run and excluded from runtime baseline. No E1 was run.

## Protocol and environment

- Target: RT - DA; origin D-1 14:00; supervised labels through D-2. Stage A only; Stage B OFF.
- Formal runs: A2, joint_v21, direction_first, vanilla, diagnostics OFF, profile=default, seed=20260924, k=8, max_epochs=120, patience=15, batch_size=64.
- Environment: epf-2; Python 3.11.14; torch 2.6.0+cu124; CUDA available; resolved device CUDA; AMP=true; NVIDIA GeForce RTX 4060 Laptop GPU.
- All E0 formal runs share config SHA `8c981156cecf6e114cf3d4eeae6ba418d62e361195a3d191d1b4b11766b5488c`, selector SHA `ed348cfd9fd911bc675d7fd920485b1a748c159a3e0a395b3af02af01015092f`, and source/sequence provenance recorded per run.
- Source and sequence gates passed; finite/numerical audit and checkpoint reload passed for all runs.

## Formal target-day results (diagnostic only)

| Run | Target | Raw | Balanced | +Recall | -Recall | AUC | Brier | Mag MAE | Skill | Best/Stop | Epochs | Wall (s) | Epoch mean / p50 / p95 (s) | Pred (ms) | GPU peak (MiB) | Warnings |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| E0-D1 | 2026-02-15 | 0.7083 | 0.5000 | 0.0000 | 1.0000 | 0.7143 | 0.1846 | 44.035 | 0.0786 | 2/17 | 17 | 11.40 | 0.466 / 0.440 / 0.538 | 13.52 | 368.0 | one_class_prediction_collapse;positive_recall_zero |
| E0-D2 | 2026-04-15 | 0.5417 | 0.7105 | 0.4211 | 1.0000 | 0.6842 | 0.2607 | 22.677 | 0.0519 | 1/16 | 16 | 12.35 | 0.549 / 0.536 / 0.645 | 8.88 | 368.0 | none |
| E0-D3 | 2026-06-15 | 0.4167 | 0.2381 | 0.0000 | 0.4762 | 0.1270 | 0.2776 | 49.808 | 0.0914 | 4/19 | 19 | 13.99 | 0.548 / 0.519 / 0.611 | 8.52 | 368.0 | positive_recall_zero |
| E0-D4 | 2026-08-10 | 0.4167 | 0.2273 | 0.4545 | 0.0000 | 0.0909 | 0.2919 | 44.793 | -0.6034 | 3/18 | 18 | 12.77 | 0.504 / 0.480 / 0.562 | 7.07 | 368.0 | nonpositive_recall_zero |

All reported target metrics above are single-day diagnostics; no ranking, threshold change, tuning, or model promotion is inferred.

## Runtime / convergence

- Formal run wall time range: 11.40–13.99s; average epoch mean range: 0.466–0.549s.
- Trainable/total parameters and checkpoints are stable across four target days; see E0_summary.csv for exact values.
- Best epochs were E0-D1=2, E0-D2=1, E0-D3=4, E0-D4=3. Stops were 17, 16, 19, 18; none approached max_epochs=120. Stopping followed configured patience, not a runtime failure.
- P0 CPU/GPU 2-epoch timing sanity (same target/config/seed; timing-only): CPU wall 7.37s vs CUDA 6.10s (wall speedup 1.207x); CPU mean epoch 1.694s vs CUDA 1.205s (epoch speedup 1.406x). This tiny check is not an efficacy comparison; single-call inference latency is not used for conclusions.

## E0-G gradient conflict (diagnostic only)

| Parameter group | N | Mean cosine | Median cosine | Conflict rate | Mean ||g_dir|| | Mean ||g_mag|| | Median norm ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| tabular_encoder | 38 | 0.027098 | 0.022884 | 0.3421 | 0.108172 | 0.388869 | 3.5687 |
| temporal_encoder | 38 | 0.040925 | 0.045042 | 0.3421 | 0.346525 | 0.056595 | 0.1511 |

- E0-G used 2 diagnostic batches per epoch and is excluded from runtime baseline. Aggregates are descriptive only; no gradient policy change follows.

## Warnings and artifacts

- One of four formal days (E0-D1) reported one-class prediction collapse. Additional zero-recall warnings occurred on E0-D3 (+Recall=0) and E0-D4 (-Recall=0). These are review flags, not post-hoc selection rules.
- All runs preserve raw_run_dir; per-run records include package/environment/config/source/sequence/selector hashes, command and raw manifest hash.
- Curves: `E0_curves/`; machine-readable metrics/runtime: `E0_summary.csv`, `E0_runtime.csv`.
- Full Jan-Aug DEV: NOT RUN. Lockbox: NOT TOUCHED. E1: NOT RUN.
