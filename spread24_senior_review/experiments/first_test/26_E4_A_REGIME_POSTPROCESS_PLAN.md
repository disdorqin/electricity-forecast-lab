# E4-A — Legal Regime-Conditioned Direction Postprocessor

STATUS=AUTHORIZED
DATE=2026-09-26
PARENT=E3-A
EXPERIMENT_ROOT=experiments/first_test/E4_signal/regime_postprocess
PRIMARY_GOAL=CROSS_MONTH_DIRECTION_65_PLUS

## Frozen evidence
- E3-A RECENCY_SIGNAL=NO_RECENCY_GAIN. T1/T2/T3 are materially worse than T0/Q2; do not tune windows further.
- Q2 SEGMENT_HEADS remains architecture anchor: Raw=.6086, Balanced=.5424, AUC=.5911.
- Q2 has severe class asymmetry (+Recall=.3058, -Recall=.7791).
- Error slicing on the 672 Q2 target slots shows materially different accuracy across legal physical regimes (residual load, renewable share, bidding-space ratio, ramps, forecast-error uncertainty).
- The frozen 259 candidates already contain many literature-motivated spread drivers; the problem may be how the final probability uses regime context, not absence of all such variables.

## Literature-motivated families already legal in Q2 inputs
Use only these existing selected legal features:
- residual_load_renew
- renewable_share
- bidding_space_ratio
- net_ramp_pressure
- err_net_load_28d_std
- uncert_风电总加_width
- uncert_光伏总加_width
- ctx_spread_positive_rate14
- spread_same_slot_28d_positive_rate

No target-day realized values and no new raw data.

## Key methodological warning
The Q2 Stage-A checkpoint was selected using the same legal historical monitor on which the postprocessor is fitted.
Therefore monitor improvement is NOT evidence of generalization and must not enter the Gate.
The Gate uses only the 28 unseen target days.

## Arms
P0 Q2_RAW: reuse E2-E1 Q2 exactly.
P1 SEGMENT_LOGIT:
- frozen Q2 probability logit
- fixed H1/H2/H3 indicators and logit interactions
- low-capacity logistic stacker fit on that target day's legal canonical Stage-A monitor
- threshold remains .5

P2 REGIME_LOGIT:
- P1 features
- plus the 9 legal physical/regime features above
- same logistic formulation / regularization / monitor-only fit
- threshold remains .5

No C sweep, threshold tuning, per-window feature search or oracle routing.

## Execution
Reuse each saved Q2 checkpoint and its exact preprocessor.
Do NOT retrain Q2.
For each of the same 28 DEV target days:
1. load saved Q2 checkpoint/preprocessor;
2. reconstruct canonical monitor indices exactly;
3. verify reloaded target base prediction == saved Q2 <=1e-9;
4. predict Q2 probabilities on legal monitor;
5. fit P1 and P2 independently on monitor labels;
6. predict target-day P1/P2 probabilities;
7. save base and postprocessed outputs/audit.

## Metrics
Overall Raw/Balanced/+R/-R/AUC/Brier/ppf/collapse.
W1-W4, H1-H3.
Paired P1-P0, P2-P0, P2-P1 day-cluster bootstrap 95% CI + W/T/L.
Slot gains overall and by H1/H2/H3.
Coefficient summaries only as diagnostics.

## Safety anchor
P0.
Admissible iff:
Balanced >= P0-.01
+Recall >= P0-.05
one-class <= P0+2
+R=0 days <= P0+2.

## Gate
POSTPROCESS_SIGNAL exactly one:
REGIME_STACKING_CONFIRMED
SEGMENT_CALIBRATION_CONFIRMED
PROMISING_POSTPROCESS_UNPROVEN
CALIBRATION_ONLY
POSTPROCESS_HARMFUL
NO_POSTPROCESS_GAIN

If P2 >=.62 and safety: immediate 3-seed / broader-day confirmation before any more architecture work.
If P2 fails materially: stop postprocessing and move to E4-B selector/feature-family recovery.
