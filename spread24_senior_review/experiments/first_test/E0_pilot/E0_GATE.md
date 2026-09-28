# E0 Gate

**Decision: GO (E0 completed; human review required before any next-stage execution).** GO means the bounded pilot satisfied its engineering/runtime gate sufficiently to submit for review. It is not a model efficacy claim and does not authorize automatic E1 execution.

| Criterion | Status | Evidence / note |
|---|---|---|
| Four formal target days completed | PASS | E0-D1 through E0-D4, all PASS on CUDA/epf-2 under identical formal configuration. |
| Leakage / contract / numerical integrity | PASS | Source/Sequence gates PASS; target RT-DA, D-1 14:00 origin, labels through D-2; numerical audit, finite training, and checkpoint reload PASS. |
| Convergence trajectory | PASS with review note | Best epochs 1, 2, 4, 3; stops 16, 17, 19, 18; no run reached max_epochs=120 or stopped abnormally early. |
| Runtime supports bounded rolling-origin feasibility | PASS | 11.40–13.99s wall per formal run; mean epoch 0.466–0.549s on RTX 4060 Laptop GPU (this does not estimate full rolling campaign overhead). |
| No majority one-class collapse | PASS with material warning | One of four days had explicit one-class prediction collapse; zero recall also occurred on D3 (+) and D4 (-). Requires human review. |
| No contract change required | PASS | No source, target adapter, selector, config, canonical model, or data contract changed for E0. |

## Required caution

Single-day metrics are diagnostic only. E0 demonstrates one explicit collapse day and further zero-class-recall warnings; do not interpret GO as effectiveness, do not tune from these four days, and do not promote V2.2.

**NEXT: stop for human review of E0. Do not run E1 until explicitly reviewed/authorized.**

FULL_JAN_AUG_DEV=NOT_RUN  
LOCKBOX=NOT_TOUCHED  
STAGE_B=NOT_RUN
