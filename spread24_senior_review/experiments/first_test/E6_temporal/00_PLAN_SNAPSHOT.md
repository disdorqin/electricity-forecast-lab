# E6-A Temporal Diagnostic — Frozen Plan

STATUS=DIAGNOSTIC_ONLY
DATE=2026-09-26
PARENT=docs/26_E6_temporal_representation_reconstruction_plan.md
CANONICAL_MODEL=docs/17_最终模型设计与编码规范.md
DATA_CONTRACT=docs/01_业务数据与防泄漏合同.md
SCOPE=NO_MODEL_CHANGE

## Question and output

Compare four pre-defined sources of temporal information for RT-DA direction: (1) recent D-1 spread state, (2) latest legal same-business-hour spread lag, (3) same-hour weekly periodic sign, and (4) the existing current TemporalEncoder in its previously saved temporal-only E2-A runs. Report Raw, Balanced Accuracy, positive recall, non-positive/negative recall overall and separately for W1-W4. This is diagnostic evidence only; no threshold search, model selection, ensemble or architecture change.

## Forecast-origin visibility

For target day D, origin is D-1 14:00. `X_hist` covers D-8 15:00 through D-1 14:00. The last 14 spread observations (history indices 154..167) are the only recent-state input. Target labels are used only to score predictions; labels available to earlier model training must end by D-2. No D-1 h15..h24, target-day actuals, future spread, or target labels enter any baseline prediction.

## Frozen baseline definitions

1. **Recent spread state (3h / 6h / 14h):** sign of the arithmetic mean of the last n observed historical spreads, repeated across 24 target hours; positive iff mean > 0. The three fixed horizons are reported separately; no horizon is selected from results.
2. **Same-hour lag:** for business hour h, use the most recent available spread from the same business hour among lag-days 1..7. Index mapping is `177 + h - 24*d` for lag-day d; if lag-1 is outside [0,167] (h=15..24), use lag-2. Predict positive iff that observed lag > 0.
3. **Periodic:** (a) majority sign among the available same-business-hour spread observations at lag-days 1..7; unavailable indices are omitted; ties are non-positive; and (b) the existing selected `spread_same_slot_28d_positive_rate` feature, positive iff its historic rate > 0.5. Its frozen lineage is shift_days=2 (labels through D-2). The 168-hour `X_hist` cannot expose 14-day raw same-hour lags; no feature will be added to manufacture them.
4. **Current TemporalEncoder diagnostic:** reuse saved E2-A `architecture_mode=temporal_only`, `objective_mode=dir_only` predictions for the exact 28 frozen W1-W4 days. No retraining. All duplicates must have identical predictions; source, sequence, selector, and target-day identity must match. This is a temporal-only ablation of the current encoder, not an attribution of its causal contribution inside the full Q2 model.

All deterministic baselines use a fixed zero sign boundary (`value > 0`), not a tuned threshold. Also report all-negative as an orientation/reference baseline.

## Frozen evaluation panel and gates

- W1: 2026-02-12..18; W2: 2026-04-12..18; W3: 2026-06-12..18; W4: 2026-08-07..13.
- 28 eligible target days / 672 slots, audited against frozen selector quarantine eligibility.
- No fit on evaluation labels, no reweighting, feature creation/selection, threshold/calibration, output blend, or ensemble.
- `E6_A_GATE.md` must name which temporal source is visibly useful/weak by windows and whether evidence is clear enough for a later design. Only clear, cross-window evidence may motivate a separately reviewed E6-B design; otherwise stop.

## Provenance

Record frozen source/sequence/selector hashes, E2-A source run directories and manifest hashes, chosen duplicate rule, Python/package versions, seed/prediction resolution, run command, and exact target window. Outputs stay under `experiments/first_test/E6_temporal/`.
