# Production runtime glue (Lab-only)

This nested package maps qualified exact-origin captures and mature-history
files onto the unchanged frozen `CurveStore`/`build_sample()` contract. It is
separate from the research CLI and does not depend on the research parquet SHA.

- `qualification.py`: independent I0/I1 checks; synthetic and consumed-replay
  evidence are explicitly non-live/non-promotable.
- `history_store.py`: content-SHA-bound projection limited to D-2.
- `input_bundle.py`: captured D-1 actual + forecast, RT p1..p56, captured target
  DA96/forecasts, no target truth or fallback.
- `api.py`: stable CPU-only `ProductionAPI.prepare_target()`.
- `runtime.py`: one-target fresh CUDA FULL fit, followed by frozen DDV1/F seals;
  F reads only prior sealed artifacts and retains calendar gaps.
- `cli.py`: `prepare` (CPU, no fit) and `run` (CUDA fresh fit) entry points.
- `bundle_identity.py`: verifies the clean relocatable bundle and code manifest.

No production2.5 writes, DB access, truth scoring, model tuning, or fallback
features are implemented. Promotion remains blocked until a real matching I1
PASS ledger and a separate promotion review exist.
