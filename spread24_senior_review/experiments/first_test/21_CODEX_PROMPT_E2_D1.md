# Codex Prompt — E2-D1 Numeric Encoding Big-Block Ablation

Project root:
D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_lab\spread24_senior_review

Only authorized task:
Execute E2-D1 Numeric Encoding Big-Block Ablation.

Read first:
1. experiments/first_test/20_E2_D1_NUMERIC_ENCODING_PLAN.md
2. experiments/first_test/E2_architecture/strong_role_routing/E2_C3_summary.md
3. experiments/first_test/E2_architecture/strong_role_routing/E2_C3_GATE.md
4. experiments/first_test/E2_architecture/horizon_gate/E2_C2_summary.md
5. docs/01_业务数据与防泄漏合同.md
6. docs/17_最终模型设计与编码规范.md
7. docs/21_正式运行、验收与实验入口规范.md

Follow plan 20 exactly.

Frozen human decision:
- E2-C3 ROLE_SIGNAL=NO_CLEAR_ROLE_EFFECT.
- DROP_MAG is not promising; do not run matched-random-drop/3-seed E2-C4 because its trigger was not met.
- DROP_DIR has no stable benefit.
- Return to all 211 Strong/Core + 11 Weak.
- Keep current 24-h horizon gate.
- Keep fixed direction_fusion_alpha=.8 for architecture screening.
- Next question is numeric representation, not selector role routing.

Canonical encoding:
PLE enabled, bins=16, embedding_dim=8, plus raw skip.
Each scalar feature => 8-D PLE + 1 raw = 9 channels.

Arms:

N0 CURRENT_PLE_RAW:
reuse R0/G0/C0/F80.
encoding = concat(PLE_8D, raw).

N1 RAW_ONLY:
fresh 28.
encoding = raw only.
No PLE reaches Strong or Weak.

N2 PLE_ONLY:
fresh 28.
encoding = PLE_8D only.
No raw skip channel.

Implement experiment-only:
numeric_encoding_mode =
canonical / raw_only / ple_only
default=canonical.

Canonical must preserve existing config semantics exactly:
cfg.ple_enabled=True => PLE+raw;
cfg.ple_enabled=False => raw-only.

raw_only experiment forces raw-only.
ple_only forces PLE-only and requires cfg.ple_enabled=True.

Do not change preprocessing, robust scale, clipping or PLE bin construction.
Same encoding mode applies to Strong and Weak.

Fail closed:
non-canonical requires
legacy_v20=False,
objective_mode=dir_only,
architecture_mode=full_current,
direction_tabular_mode=current,
direction_horizon_gate_mode=current,
strong_role_profile=all,
direction_fusion_alpha=.8,
train_mode=stage_a.
ple_only also requires cfg.ple_enabled=True.

Gate A:
1 canonical profile reproduces R0/F80 benchmark <=1e-9.
2 default/no flag exact.
3 N0 encoded dim=9.
4 N1 encoded dim=1.
5 N2 encoded dim=8.
6 frozen PLE bins/hashes unchanged.
7 Strong/Core=211, Weak=11.
8 current gate/Temporal/k8 live.
9 finite and leakage unchanged.
10 canonical94 + all prior E2 tests + focused tests PASS.

Benchmark day=2026-02-13 engineering only.

Formal DEV:
W1 2026-02-12..02-18
W2 2026-04-12..04-18
W3 2026-06-12..06-18
W4 2026-08-07..08-13

N0 reuse.
N1 fresh28.
N2 fresh28.
No intermediate stopping.

Training fixed:
dir_only / direction_first / vanilla /
full_current / current tabular route / current horizon gate /
strong_role_profile=all / fixed fusion alpha=.8 /
Stage A / default / seed20260924 / k8 /
d_tab128 / n_blocks2 / dropout.05 /
max120 / patience15 / batch64 / frozen LR-WD / CUDA AMP.

Report:
overall Raw/Balanced/+R/-R/AUC/Brier/ppf/collapse.
W1-W4 + min/std.
H1-H3.
paired N1-N0, N2-N0, descriptive N1-N2.
day-cluster bootstrap 95% CI + W/T/L.
encoded dims, TabM input dims, params, runtime/GPU/best-stop.

Safety anchor=N0:
Balanced>=N0-.01
+Recall>=N0-.05
one-class<=N0+2
+R=0 days<=N0+2.

E2_D1_GATE ENCODING_SIGNAL exactly one:
RAW_SUFFICIENT
PLE_REQUIRED
RAW_SKIP_REQUIRED
PLE_RAW_SYNERGY
RAW_ONLY_BETTER
PLE_ONLY_BETTER
NO_CLEAR_ENCODING_EFFECT

Do not call a pooled-score change stable without paired/cross-window support.

Critical anti-overfit:
Do NOT tune ple_bins or ple_embedding_dim after seeing this result.
First establish the representation family.
Any apparent improved candidate requires later multi-seed confirmation before production promotion.

Forbidden:
role filtering, selector rerun, PLE bins/dim sweep, k/depth/width changes,
gate tuning, Temporal/FFT, fusion/alpha sweep, threshold/calibration,
checkpoint change, Stage B, broader DEV, lockbox.

Output:
experiments/first_test/E2_architecture/numeric_encoding/

At least:
00_PLAN_SNAPSHOT.md
benchmark/
runs/
daily_metrics.csv
model_summary.csv
window_metrics.csv
hour_segment_metrics.csv
paired_summary.csv
encoding_audit.json/csv
runtime.csv
E2_D1_summary.md
E2_D1_GATE.md
figures/

After E2-D1 STOP.

Final report only:
1 FILES_CHANGED
2 TESTS
3 ENCODING_ROUTING_AUDIT
4 BENCHMARK engineering
5 N0/N1/N2 overall
6 W1-W4
7 H1-H3
8 paired N1-N0
9 paired N2-N0
10 safety
11 encoding dimensions / parameter counts
12 runtime
13 E2_D1_GATE ENCODING_SIGNAL
14 NEXT suggestion only
15 unexecuted later work

Do not stage/commit.
