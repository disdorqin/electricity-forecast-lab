# E2-C1 — Tabular Strong/Weak Big-Block Attribution: frozen snapshot

Frozen before C1/C2 formal outputs are read. Source plan: `14_E2_C1_TABULAR_BIG_BLOCK_PLAN.md`.

## Axis and arms

Only Direction's tabular input changes. All runs use `full_current`, fixed
`direction_fusion_alpha=0.8`, `dir_only`, `direction_first`, `vanilla`, Stage A,
default profile, A2, seed 20260924, k=8, max 120 epochs, patience 15, batch 64,
frozen LR/WD, CUDA+AMP. Magnitude always sees canonical gated `H_current`.

- C0 CURRENT_TABULAR: reuse E2-B1 F80 records/checkpoints for all 28 days.
- C1 STRONG_ONLY: Direction receives `H_strong`; Weak MLP and gate bypassed.
- C2 WEAK_ONLY: Direction receives broadcast `H_weak`; TabM and gate bypassed.

Formal panel is exactly the E1 preflight: W1 2026-02-12..18, W2 2026-04-12..18,
W3 2026-06-12..18, W4 2026-08-07..13. Benchmark 2026-02-13 is engineering-only.
All 28 target days per fresh arm will run; no interim result can stop the panel.

## Frozen interpretation operationalization

To make the plan's words “material”, “stable”, “weak”, and “competitive” auditable:

- Material Raw difference: at least 0.02 pooled-slot accuracy (2 pp).
- Stable paired direction: day-cluster bootstrap 95% CI for the paired mean Raw
  difference excludes zero, and at least 3 of 4 windows have the same sign.
- Safety is the plan's exact four-part C0-relative rule.
- “Near chance” Balanced means within 0.02 of 0.50.
- C2 is competitive with C1/C0 when its paired Raw difference to each is within
  0.02 and neither paired CI establishes a disadvantage of at least 0.02.
- Gate precedence: safety-admissible C1/C2 stable material improvement over C0
  => `TABULAR_MIXING_PROBLEM`; otherwise, stable C0 superiority over C1 with
  C2 weak alone => `WEAK_RESIDUAL`; otherwise competitive nontrivial C2 =>
  `WEAK_UNEXPECTEDLY_STRONG`; otherwise C1 meeting C0 within 2 pp with stable
  material C1-C2 advantage and C2 weak => `STRONG_DOMINANT`; otherwise stable
  material C0 superiority over both => `STRONG_WEAK_SYNERGY`; otherwise both
  isolated arms near chance and no stable C0 gain => `BOTH_INTERNAL_WEAK`;
  else `INCONCLUSIVE`.

All paired inference is day-cluster bootstrap (10,000 draws, seed 20260924),
with W/T/L day counts and cross-window metrics reported. Pooled Raw alone never
supports a stable attribution. This operationalization does not change the
experiment's frozen data, selector, training, or safety contract.

## Gate A and provenance

Gate A compares the reused F80 benchmark checkpoint with two freshly rebuilt
inference graphs: explicit `current` + fixed .8, and default/no direction route
flag with learnable A2 alpha loaded from the same checkpoint. Artifact predictions
and recursively flattened numeric metric leaves must match within 1e-9. Gradient
ownership is checked on the real benchmark-day tensors/checkpoints. F80 C0 is
reused without fresh training.

Expected frozen provenance: config `8c981156cecf6e114cf3d4eeae6ba418d62e361195a3d191d1b4b11766b5488c`,
selector `ed348cfd9fd911bc675d7fd920485b1a748c159a3e0a395b3af02af01015092f`,
source slot `a1b86f956d9fb18a483d473cbc0334e1078097f6ef75274b2804488968a349ea`.

No alpha sweep/refinement, other feature-role split, selector/feature/config change,
Stage B, thresholding/calibration, full DEV, or lockbox access. E2-C1 ends at its
own gate for human review; no subsequent stage is authorized here.
