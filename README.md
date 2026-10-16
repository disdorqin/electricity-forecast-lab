# Spread96 production/shadow runtime bundle

This is a relocatable, Lab-generated runtime bundle for the frozen chain
`FULL_V2 -> DirectionDecisionV1 -> F_PHASE44_V1`. It is not an integration into
`electricity_forecast_model2.5`. Production promotion remains
`BLOCKED_INFORMATION_CONTRACT` until a real 30-consecutive-day I1 ledger passes.

## Contract

- Origin: D-1 14:00 Asia/Shanghai; training labels/features mature through D-2.
- Each target uses a new FULL_V2 model, optimizer, preprocessing fit and complete expanding history from 2023-01-01 through D-2. No warm start, rolling window or sampled history.
- Inputs are an immutable C1 exact-origin capture (manifest + values + raw evidence) and a mature-history data file. The history loader rejects any row later than D-2 and binds its content SHA.
- D-1 actual and forecast curves are both required for the frozen `actual - forecast` feature. D-1 RT is model-visible only for p1..p56. Target DA96 must be the captured final DA; target RT and actuals are absent.
- F consumes only pre-existing sealed historical FULL/DDV1/F evidence chronologically. Missing F history is passed to the frozen policy as a calendar gap and invokes its existing V1 fail-safe; this runtime never fits a missing prior day to fill F history.
- The `prepare` command is CPU-only and performs no fit. `run` invokes the existing frozen CUDA-only docs86 FULL trainer; then it seals DDV1 and F. No truth-scoring method or target truth reader is present.

## Install / verify

Use Python 3.11.16 and the pinned versions in `requirements_production_20261016.txt`.
Install the matching CUDA PyTorch wheel for the replay host, then run:

```powershell
python -m src.research96.msr_spread_v2.production.cli prepare `
  --bundle-root . --target-day YYYY-MM-DD --manifest CAPTURE/snapshot_manifest.json `
  --values CAPTURE/values.parquet --history MATURE/history.parquet --mode consumed_replay
```

The prepare command verifies all input hashes/schema, enforces the D-2 history
cutoff, and builds one frozen feature-ready sample without training. `promotion`
mode additionally reruns exact-origin I0 and requires `--i1-ledger` with a
matching capture row from a 30-day I1 PASS ledger. `consumed_replay` accepts only
an explicitly labelled consumed-reference projection with target truth columns
projected out; it is retrospective parity input and never live availability
evidence. `synthetic_test` is structural test mode only. Neither non-promotion
mode can become production eligible.

For an authorized CUDA replay, `run` takes the same inputs plus `--output-root`
and an optional `--f-history-root`.

```powershell
python -m src.research96.msr_spread_v2.production.cli run `
  --bundle-root . --target-day YYYY-MM-DD --manifest CAPTURE/snapshot_manifest.json `
  --values CAPTURE/values.parquet --history MATURE/history.parquet `
  --mode promotion --i1-ledger I1_LEDGER.json --output-root REPLAY_OUTPUT `
  --f-history-root REPLAY_OUTPUT --f-history-start 2026-01-01
```

For the frozen consumed 2026 YTD replay, F warm-up begins at 2026-01-01; for a
future shadow launch, set `--f-history-start` to the separately authorized
launch date. Missing calendar artifacts are not compressed or backfilled.

Before use, verify `BUNDLE_MANIFEST.json` and keep it with all outputs. Each
target has FULL, DIRECTION and F seals; target truth is never opened by this
runtime. This bundle is for the separately authorized consumed-data server
replay only; it does not authorize a LIVE deploy or production2.5 changes.
