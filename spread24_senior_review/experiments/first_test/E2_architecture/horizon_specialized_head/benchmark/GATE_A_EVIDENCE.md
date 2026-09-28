# E2-E1 Gate A — readout routing audit evidence

All Gate A items were checked before formal runs. Machine-readable records: `gate_a_results.json`,
`gate_a_tests.json`, `gate_a_failclosed.txt`.

## 1 & 2. Shared / default-no-flag reproduces the frozen F80 control (<=1e-9)

A fresh `--direction-readout-mode shared` run and a fresh default (no-flag) run were both executed on
2026-02-13 and compared against the frozen control `E2B1-F80-2026-02-13`.

| comparison | max |Δ| over numeric prediction columns | item |
|---|---|---|
| Q0 shared vs F80 | 0.0 | 1 PASS |
| Q0D no-flag vs F80 | 0.0 | 2 PASS |

Benchmark agreement: Raw `0.5833333333333334` for Q0/Q0D/Q1/Q2. Q0 parameter count `382,724`
(identical to F80).

## 3 & 4. Q1 = one shared head + exactly 3 zero-init scalar biases

`direction_readout_audit` reports `n_segment_bias=3`, `n_segment_heads=0`, `shared_direction_head=True`.
The selected checkpoint contains exactly one `direction_hour_bias` tensor of shape `(3,)`
(structurally zero-initialised; unit test `test_segment_bias_has_exactly_three_zero_init_scalars_and_one_shared_head`).
Parameter count: `382,724 -> 382,727` (+3), i.e. only the three scalars are added.

## 5 & 6. Q2 = exactly 3 independent heads, no shared Direction head, exclusive slicing

`direction_readout_audit` reports `n_segment_heads=3`, `n_segment_bias=0`, `shared_direction_head=False`.
The selected checkpoint contains no `direction_head.*` keys and carries `direction_segment_heads.*`.
Parameter count: `382,724 -> 383,252` = `382,724 - 264 + 3×264` (head = 264 params), exactly one shared
head replaced by three.
Unit test `test_segment_heads_each_hour_is_produced_by_exactly_one_head` proves each hour is produced
by exactly one head via perturbation of a single head.

## 7. k = 8 and [B,k,24] preserved

Every manifest records `config.k=8`; every prediction table has 24 hourly rows with
`hour_business == 1..24`.

## 8. Threshold / readout semantics unchanged

`direction_hat == (p_positive >= 0.5)` holds exactly for Q0/Q1/Q2.

## 9. Magnitude path unchanged

`magnitude_head.*` state-dict key shapes are identical across Q0/Q1/Q2 and all `magnitude_hat` values
are finite.

## 10. Frozen source/config/selector/sequence

| anchor | sha256 |
|---|---|
| config | `8c981156cecf6e114cf3d4eeae6ba418d62e361195a3d191d1b4b11766b5488c` |
| source (`data/frozen_repro/slot_table.parquet`) | `a1b86f956d9fb18a483d473cbc0334e1078097f6ef75274b2804488968a349ea` |
| selector | `ed348cfd9fd911bc675d7fd920485b1a748c159a3e0a395b3af02af01015092f` |
| sequence | `9144bbed33369a4bed5a8acc508a71f50836e55667e3cc4e768badfd35ce7a82` |

## 11. Test suites

| suite | result |
|---|---|
| canonical non-destructive baseline (excluding E2-A/B1/C1/D1/E1 focused files) | **94 passed** |
| full suite (canonical + all prior E2 + focused E2-E1) | **174 passed** |

Command (doc 21 §13):
```
epf-2\python.exe -m pytest -q -p no:cacheprovider "src/TafM_改进源码/tests" \
  tests/test_data_contract.py tests/test_leakage.py --ignore=tests/test_smoke.py
```

## Fail-closed CLI probes

Non-shared `direction_readout_mode` with an incompatible objective/alpha/encoding/role route (and the
invalid mode string) all exit `2` before any training. See `gate_a_failclosed.txt`.

## Source changes for this axis

One experiment-only axis was added, default `shared` at every entry point.

| file | SHA256 (16) | change |
|---|---|---|
| `models/dual_branch_v21.py` | `d89bcb385018aa0a` | `direction_readout_mode`; `_direction_logits`; segment modules; `direction_readout_audit()` |
| `train.py` | `bd90853578f482d0` | thread argument; fail-closed validation; `is_experiment`; run-dir suffix; manifest fields; per-epoch segment-bias history |
| `run_tabm_v21.py` | `b895b1e66047388a` | `--direction-readout-mode` CLI flag |
| `tests/test_e2_e1_horizon_readout.py` | `3bd6d552c3b901ac` | 9 new focused E2-E1 tests |

No incidental refactoring. `legacy_v20` continues to reject every non-default experiment axis.
