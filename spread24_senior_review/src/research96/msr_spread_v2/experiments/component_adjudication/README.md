# docs/86 GPU scientific runtime scaffold

Core graph, losses, preprocessing and stage freeze functions are imported unchanged.
Only experiment-local training scheduling adds A 15–40 / B 5–12 / C 10–30,
patience 7 after the floor, chronological last-28-day selection and fresh exact refit.
Pre-floor checkpoints cannot be selected for refit. No warm-start or 1/1/1 profile.

`benchmark.py` executes exactly one of the three registered benchmark dates on CUDA.
The initial user request was one-day timing; the subsequent request authorizes
ordered panel execution until **2026-10-10 12:00 Asia/Shanghai**. `pipeline.py`
finishes all three registered benchmarks before choosing FAST24/FULL30; preserves
durable per-day results and reports incomplete panels explicitly at the deadline.
Existing independently reviewed core hashes must match; additions are separately tested.
Predictions are sealed before target truth is opened. Protected trees remain read-only.
Source: immutable lab canonical; Native96; origin D-1 14:00; complete labels ≤D-2.
Full target DA / full D-1 grid actual research assumptions remain LEGACY-UNVERIFIED,
PROXY/EXPERIMENTAL, not production/live claims. No September/October targets accepted.

Profile configuration is identical across registered arms. `arms.py` provides the
four explicitly authorized wrappers plus a ~comparable single width80 TCN/TabM
DIRECT96 distribution. DIRECT96 excludes T24-derived historical anchor columns
but keeps all original primitive history/DA/proxy/grid information and the same
training-local preprocessing. Its output has no Low/High decomposition or learned
dependence. Stage A fits its direct quantile distribution; C fits the exact frozen
final objective. DIRECT96 / NO_DYNAMIC_DEP have no trainable B parameters: B is an
explicit no-op with the same floor/patience accounting, never relabelled head training.
CPU regression tests verify original FULL graph/loss exact equality, ablation
semantics, monotone direct head, 80–120% capacity, and floor/profile invariants.
