# FIRST_TEST Single Run Record Template

RUN_ID=
GATE=
DATE=
STATUS=TODO

## 1. Purpose

Claim tested:

Why this run exists:

## 2. Reproducibility

~~~text
command=
target_day_or_window=
objective_mode=
checkpoint_policy=
gradient_policy=
profile=
mode=
seed=
device=
config_path=
config_sha256=
selector_sha256=
raw_run_dir=
~~~

## 3. Engineering

~~~text
total_params=
trainable_params=
checkpoint_size_bytes=
epochs_run=
best_epoch=
stop_epoch=
wall_time_total_seconds=
epoch_seconds_mean=
epoch_seconds_p50=
epoch_seconds_p95=
prediction_latency_ms=
cuda_peak_memory_bytes=
~~~

## 4. Metrics

~~~text
Raw=
Balanced=
+Recall=
-Recall=
AUC=
Brier=
Magnitude_MAE=
Magnitude_RMSE=
Magnitude_tail_MAE=
Magnitude_skill=
~~~

## 5. Training behavior

- Raw trajectory:
- L_dir trajectory:
- L_mag trajectory:
- Magnitude MAE trajectory:
- class collapse warning:
- numerical warning:

## 6. Evidence / files

~~~text
manifest=
metrics=
predictions=
training_history=
gradient_conflict=
parameter_group_audit=
~~~

## 7. Observation

只写当前 run 能支持的事实：

## 8. Decision

~~~text
KEEP / REJECT / DIAGNOSTIC_ONLY / INVALID / NEED_REVIEW
~~~

Reason:

## 9. Next

只写当前 Gate 内的下一步。不得越 Gate。
