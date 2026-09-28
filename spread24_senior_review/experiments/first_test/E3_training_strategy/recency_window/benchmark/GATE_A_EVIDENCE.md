# E3-A Gate A — split routing audit evidence

All Gate A items were checked before formal runs. Machine-readable records: `gate_a_results.json`,
`gate_a_tests.json`, `gate_a_failclosed.txt`.

## 1. T0 reproduces the saved E2-E1 Q2 benchmark (<=1e-9)

A fresh `--direction-readout-mode segment_heads` run with no split flags was executed on 2026-02-13 and
compared against the saved E2-E1 Q2 benchmark control.

| comparison | max |Δ| over numeric prediction columns | item |
|---|---|---|---|
| T0 vs E2-E1 Q2 | 0.0 | 1 PASS |

Raw `0.5833333333333334` for T0/Q1-style control; parameter count unchanged.

## 2–6. Exact date/count routing (benchmark runs, target 2026-02-13)

| arm | monitor n | monitor start | monitor end | base n | base start | base end | checks |
|---|---:|---|---|---:|---|---|---|
| T1 | 60 | 2025-12-14 | 2026-02-11 | 1435 | 2022-01-09 | 2025-12-13 | 2,3,4,5,6 PASS |
| T2 | 60 | 2025-12-14 | 2026-02-11 | 365 | 2024-12-14 | 2025-12-13 | 2,3,4,5,6 PASS |
| T3 | 60 | 2025-12-14 | 2026-02-11 | 1095 | 2022-12-15 | 2025-12-13 | 2,3,4,5,6 PASS |

- MONITOR is exactly the newest 60 eligible days and ends at D-2 (`2026-02-11`).
- BASE ends `2025-12-13`, immediately before MONITOR starts (`2025-12-14`) — adjacent, no overlap.
- T2 BASE count is exactly 365 and T3 exactly 1095; T1 keeps all earlier days (1435).
- no day appears in both BASE and MONITOR; neither contains target (D) or D-1.
- `preprocessing_fit_day_start/end` equals the BASE start/end exactly for every arm (`fit == BASE`).

## 7. Frozen source/config/selector/sequence

| anchor | sha256 |
|---|---|
| config | `8c981156cecf6e114cf3d4eeae6ba418d62e361195a3d191d1b4b11766b5488c` |
| source (`data/frozen_repro/slot_table.parquet`) | `a1b86f956d9fb18a483d473cbc0334e1078097f6ef75274b2804488968a349ea` |
| selector | `ed348cfd9fd911bc675d7fd920485b1a748c159a3e0a395b3af02af01015092f` |
| sequence | `9144bbed33369a4bed5a8acc508a71f50836e55667e3cc4e768badfd35ce7a82` |

## 8. Q2 route and k=8 preserved

Every arm records `direction_readout_mode=segment_heads` with `n_segment_heads=3` and `config.k=8`;
the `segment_bias`/`shared` routes are not used.

## 9. Test suites

| suite | result |
|---|---|
| canonical non-destructive baseline (excluding E2-A/B1/C1/D1/E1 and E3-A focused files) | **94 passed** |
| full suite (canonical + all prior E2 + focused E3-A) | **182 passed** |

Command (doc 21 §13):
```
epf-2\python.exe -m pytest -q -p no:cacheprovider "src/TafM_改进源码/tests" \
  tests/test_data_contract.py tests/test_leakage.py --ignore=tests/test_smoke.py
```

## Fail-closed CLI probes

`--stage-a-monitor-days 0`, `--stage-a-history-window-days` without `--stage-a-monitor-days`, an
explicit split without the E3-A route (segment_heads), and an explicit split with a non-.8 fusion alpha
all exit `2` before any training. See `gate_a_failclosed.txt`.

## Source changes for this axis

Two related experiment-only split arguments were added, defaults `None/None` at every entry point.

| file | SHA256 (16) | change |
|---|---|---|
| `train.py` | `b65eed12ae6fbf3a` | `stage_a_monitor_days`/`stage_a_history_window_days`; `stage_a_split_indices` helper; fail-closed validation; `is_experiment`; run-dir suffix; manifest split audit |
| `run_tabm_v21.py` | `67595b5007a03ca4` | `--stage-a-monitor-days` / `--stage-a-history-window-days` CLI flags |
| `tests/test_e3_a_recency_window.py` | `cac0979b309d1e79` | 8 new focused E3-A tests |

No incidental refactoring; `legacy_v20` continues to reject every non-default experiment axis. The
`segment_bias`/`segment_heads` run-dir suffixes were also shortened (`_rb`/`_rh`) because the previous
`_rheads` leaf pushed `training_history.parquet` to the Windows MAX_PATH limit on the longest E3-A
leaf; the manifest records the mode, so this only affects newly generated run-dir names.
