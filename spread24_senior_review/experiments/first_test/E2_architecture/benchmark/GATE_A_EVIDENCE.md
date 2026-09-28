# E2-A Gate A — routing audit evidence

Gate A had to prove six things before any Gate B/C training. All six are recorded here with the
command or artifact that establishes each one.

## 1. `architecture_mode=full_current` reproduces E1 T2 exactly on 2026-02-13

Required: exact or `<= 1e-9`. Achieved: **bit-identical**.

| quantity | old E1 T2 leaf | new A0 run |
|---|---|---|
| `predictions.parquet` SHA256 | `12da7233d4964a55bd4a3b7ce2e435a20270277e92541e5c7b9d30f39bfa48d7` | same |
| max \|delta\| over 13 numeric columns | — | **0.0** |
| `metrics.json` scalar keys compared | 147 | max \|delta\| **0.0** |

Headline agreement on the benchmark day: Raw `0.5833333333333334`, Balanced `0.5378151260504201`,
AUC `0.5630252100840336`, Brier `0.23634054518374412` — matching the values pre-registered in
`10_E2_ARCHITECTURE_BIG_BLOCK_PLAN.md`.

Run leaves:
- old (E1 T2): `.../2026-02-13_dir_dirfirst_van_20260925T095128837985Z`
- new (A0, `--architecture-mode full_current`): `.../2026-02-13_dir_dirfirst_van_20260925T111526536685Z`

The run-dir leaf keeps its exact historical shape because `full_current` maps to an empty
architecture suffix, so the canonical experiment path is unchanged.

## 2 & 3. Direction gradient ownership is severed exactly where required

Two independent proofs.

**(a) Unit contract tests** — `src/TafM_改进源码/tests/test_architecture_mode.py`, 17 tests, all PASS:

| proof | test |
|---|---|
| `tabular_only` h_dir is exactly `tab_dir(H_tab)` | `test_tabular_only_direction_reads_the_tabular_branch_only` |
| `temporal_only` h_dir is exactly `time_dir(H_time)` | `test_temporal_only_direction_reads_the_temporal_branch_only` |
| `full_current` h_dir is exactly the canonical alpha fusion | `test_full_current_direction_is_the_canonical_alpha_fusion` |
| temporal params get **no** Direction gradient | `test_tabular_only_direction_cannot_reach_temporal_parameters` |
| tabular params get **no** Direction gradient | `test_temporal_only_direction_cannot_reach_tabular_parameters` |
| Magnitude fusion identical in all three modes | `test_magnitude_fusion_is_identical_in_all_three_modes` |
| parameter count and state_dict shapes mode-invariant | `test_parameter_count_and_state_dict_shape_are_mode_invariant` |

**(b) Runtime probe on the trained checkpoints** — `verify_gradient_ownership.py` rebuilds each
benchmark-day model from its frozen config and loads `stage_a_best.pt`, then asks autograd on the
real target-day tensors. `n_with_grad / n_params`:

| model | `L_dir` → tabular_encoder | `L_dir` → temporal_encoder | `L_mag` → excluded encoder |
|---|---|---|---|
| A0 full_current | 12/12 | 16/16 | live |
| A1 tabular_only | 12/12 | **0/16** | live |
| A2 temporal_only | **0/12** | 16/16 | live |

All 7 checks PASS (`gradient_ownership_checks.json`). The magnitude column matters: it shows the
excluded encoder is still a live part of the graph, so the Direction disconnect is specific rather
than a dead-graph artefact.

## 3b. Reproducibility of the new modes

An independent second process re-ran A1 `tabular_only` on 2026-02-13 with the identical command.
`predictions.parquet` came back **byte-identical** (same SHA256), max |delta| `0.0` over all 13
numeric columns. Combined with the A0 bit-exact reproduction in §1, all three modes are
deterministic under the frozen seed `20260924`.

## 4. Finite / shape PASS

`test_all_modes_keep_finite_shapes_and_the_member_axis` runs all three modes through the real
`DualBranchV21` and asserts `H_tab (2,2,24,8)`, `H_time (2,24,8)`, `z_members (2,2,24)`,
`a_scaled_members (2,2,24)`, `direction_hat == (p >= 0.5)`, all outputs finite, and a finite
`L_total`. The TabM member axis `k` survives to both heads in every mode.

## 5. Anti-leakage boundary unchanged

| contract | expected | observed |
|---|---|---|
| `src/config_tabm_v21.yaml` SHA256 | `8c981156cecf6e114cf3d4eeae6ba418d62e361195a3d191d1b4b11766b5488c` | unchanged |
| frozen `data/frozen_repro/slot_table.parquet` SHA256 | `a1b86f956d9fb18a483d473cbc0334e1078097f6ef75274b2804488968a349ea` | unchanged |
| frozen selector SHA256 | `ed348cfd9fd911bc675d7fd920485b1a748c159a3e0a395b3af02af01015092f` | unchanged in all 4 new runs |
| selector / target adapter / data contract / source resolver | not edited | untouched |

The axis is a call argument (like `objective_mode`), never a YAML key, so `config_sha256` is
byte-identical to E1's and the frozen source/selector are provably the same artifacts.

## 6. Canonical non-destructive suite still 94 PASS

| suite | result |
|---|---|
| canonical command **excluding** the new file | **94 passed** |
| canonical command **including** the new Gate A file | **111 passed** (94 + 17) |

Command (doc 21 §13):
```
epf-2\python.exe -m pytest -q -p no:cacheprovider "src/TafM_改进源码/tests" \
  tests/test_data_contract.py tests/test_leakage.py --ignore=tests/test_smoke.py
```

## Source changes for this axis

Exactly one axis was added, with `full_current` as the default at every entry point.

| file | SHA256 (16) | change |
|---|---|---|
| `models/task_adapters.py` | `688daac25228467d` | `ARCHITECTURE_MODES`; `architecture_mode` selected in `MemberWiseTaskAdaptersV21.forward` |
| `models/dual_branch_v21.py` | `d97a39c0b5bef142` | pass-through + validation |
| `train.py` | `cf2b94b09fa62ae6` | thread the argument; manifest records `architecture_mode`; run-dir suffix |
| `run_tabm_v21.py` | `3cb15b8b586fb00b` | `--architecture-mode` CLI flag |
| `tests/test_architecture_mode.py` | `035cf013b59e0ea6` | 17 new Gate A tests |

No incidental refactoring. `legacy_v20` explicitly rejects any non-default `architecture_mode`.

## Known provenance gap (recorded, not fixed)

`stage_a_best.pt` does not carry `architecture_mode`; the trained mode is recoverable from the
run's `manifest.json` and the run-directory name. Fixing this was out of scope for a
no-incidental-refactoring round and is noted here so a future round can decide on it.
