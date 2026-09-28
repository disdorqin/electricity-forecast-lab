# E2-D1 Gate A — encoding audit evidence

All ten pre-registered Gate A items were checked before formal runs. Full machine-readable record:
`gate_a_results.json`, `gate_a_tests.json`, `gate_a_failclosed.txt`.

## 1 & 2. Canonical / default-no-flag reproduces the frozen F80 control (<=1e-9)

A fresh `--numeric-encoding-mode canonical` run and a fresh default (no-flag) run were both executed on
2026-02-13 and compared against the frozen control `E2B1-F80-2026-02-13`.

| comparison | max |Δ| over numeric prediction columns | item |
|---|---|---|
| N0 canonical vs F80 | 0.0 | 1 PASS |
| N0D no-flag vs F80 | 0.0 | 2 PASS |
| N0 vs N0D | 0.0 | 2 PASS |

Headline benchmark agreement: Raw `0.5833333333333334` for N0/N0D/N2, matching the pre-registered
F80 value; parameter count identical (`382,724`).

## 3–5. Encoded feature dimensions on the real frozen feature set

| arm | mode | encoded_feature_dim | Strong | Weak | TabM in-dim | Weak in-dim | params |
|---|---|---:|---:|---:|---:|---:|---:|
| N0 | canonical | **9** | 211 | 11 | 1899 | 99 | 382,724 |
| N1 | raw_only | **1** | 211 | 11 | 211 | 11 | 111,700 |
| N2 | ple_only | **8** | 211 | 11 | 1688 | 88 | 352,620 |

1899 = 211 × 9 exactly, as preregistered.

## 6. Frozen PLE bins / hashes unchanged

`preprocessor_state.json` `ple_bins` SHA256 is identical across the frozen F80 control and all three
E2-D1 arms (N0/N1/N2). `ple_bins` construction is untouched, so raw_only still fits the canonical
bins. `src/config_tabm_v21.yaml` SHA256 stays `8c981156…5488c` (the axis is a call argument, never a
YAML key), so `config_sha256` is byte-identical to E1/E2.

## 7. Strong/Core = 211, Weak = 11

Recorded in every arm's manifest `numeric_encoding_audit` (`strong_feature_count=211`,
`weak_feature_count=11`); all 211 Strong/Core + 11 Weak are retained in N0/N1/N2.

## 8. Current 24-h gate + Temporal + k=8 remain live

The selected checkpoint `stage_a_best.pt` of each arm contains `tabular_encoder.horizon_gate_logits`
and `temporal_encoder.*` state keys, and every manifest records `config.k=8`,
`direction_horizon_gate_mode=current`, `architecture_mode=full_current`, `direction_fusion_alpha=0.8`.

## 9. Finite outputs / leakage anchors unchanged

All prediction tables are finite; `numerical_stability_status=PASS` in every run. The frozen hashes
`source_sha256`, `selector_sha256`, `sequence_manifest_sha256` are unchanged in all arms:

| anchor | sha256 |
|---|---|
| config | `8c981156cecf6e114cf3d4eeae6ba418d62e361195a3d191d1b4b11766b5488c` |
| source (`data/frozen_repro/slot_table.parquet`) | `a1b86f956d9fb18a483d473cbc0334e1078097f6ef75274b2804488968a349ea` |
| selector | `ed348cfd9fd911bc675d7fd920485b1a748c159a3e0a395b3af02af01015092f` |
| sequence | `9144bbed33369a4bed5a8acc508a71f50836e55667e3cc4e768badfd35ce7a82` |

## 10. Test suites

| suite | result |
|---|---|
| canonical non-destructive baseline (excluding E2-A/B1/C1/D1 focused files) | **94 passed** |
| full suite (canonical + all prior E2 + focused E2-D1) | **165 passed** |

Command (doc 21 §13):
```
epf-2\python.exe -m pytest -q -p no:cacheprovider "src/TafM_改进源码/tests" \
  tests/test_data_contract.py tests/test_leakage.py --ignore=tests/test_smoke.py
```

## Fail-closed CLI probes

Non-canonical `numeric_encoding_mode` with an incompatible route/alpha/objective/train_mode (and the
invalid mode string) all exit `2` before any training. See `gate_a_failclosed.txt`.

## Source changes for this axis

Exactly one experiment-only axis was added, default `canonical` at every entry point.

| file | SHA256 (16) | change |
|---|---|---|
| `models/tabular_encoder.py` | `b4cf2a94a0458ccb` | `numeric_encoding_mode` family; `use_ple`/`use_raw`; `encoding_audit()` |
| `train.py` | `a34e054afbb4ce1a` | thread the argument; fail-closed validation; `is_experiment`; run-dir suffix; manifest fields |
| `run_tabm_v21.py` | `088a62e0a4f985c7` | `--numeric-encoding-mode` CLI flag |
| `tests/test_e2_d1_numeric_encoding.py` | `eb08406c60d6f8e9` | 8 new focused E2-D1 tests |

No incidental refactoring. `legacy_v20` continues to reject every non-default experiment axis.
