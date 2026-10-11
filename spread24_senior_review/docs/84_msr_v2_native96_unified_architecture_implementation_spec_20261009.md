# 84 — MSR-V2 Native-96 Unified Multi-Scale Distribution Architecture & Implementation Specification (2026-10-09)

STATUS=CURRENT_ARCHITECTURE_AND_IMPLEMENTATION_AUTHORITY
TRAINING_AUTHORITY=docs/82_msr_daily_asof_training_and_fast24_protocol_20261009.md
SUPERSEDES_ARCHITECTURE_DRAFT=docs/83_msr_core_v1_architecture_redesign_draft_20261009.md
DOES_NOT_SUPERSEDE_PRODUCTION_CONTRACT=production repository remains strictly read-only
CODE_ROOT=src/research96/msr_spread_v2
OUTPUT_ROOT=outputs/experiments/msr_spread_v2
TARGET=Spread S = RT - DA
RESOLUTION=96 points/day, 15-minute
FORECAST_ORIGIN=D-1 14:00
LABEL_MATURITY=D-2
CORE_STATE=STATE_FREE_K1
OPTIONAL_STATE_EXTENSION=SOFT_LATENT_MIXTURE_K3_ONLY
PHYSICAL_MEDIATOR_SYMBOL=U
LATENT_REGIME_SYMBOL=R

## 0. Purpose and non-negotiable design position

This file is the coding authority for the second-generation MSR research model. It is written so that a single capable coding agent can implement the full isolated V2 research branch without redesigning the method while coding.

The V2 model has one probabilistic story:

1. all legal Native-96 information is assembled at the true D-1 14:00 forecast origin;
2. a fixed T24 operator defines the target coordinates `S = L + H`;
3. business-semantic feature families are embedded at native 96-slot resolution;
4. each feature family has its own learnable complementary spectral gate, because price, demand, renewable, supply and tightness series have different frequency structures;
5. the Low branch predicts a full monotone quantile function for `L`;
6. the High branch predicts a full monotone quantile function for `H`, using residual history and Low-distribution context;
7. a lightweight learned smooth dynamic dependence layer models both same-slot Low/High dependence and temporal continuity across the 96 slots;
8. scenario reconstruction preserves the exact identity `S = L + H`;
9. one final distribution `F_S` yields both Spread point prediction and `P(S>0)`; target-day DA is then used as a known deterministic anchor to obtain RT.

The Core intentionally excludes post-hoc hard State, Markov smoothing and mandatory physical mediator U. They remain optional extensions only after the Core is scientifically supported.

### 0.1 Hard prohibitions

The implementation agent MUST NOT:

- modify `src/research96/msr_spread_v1/**`;
- modify `src/TafM_改进源码/**`;
- modify any file under the production repository `electricity_forecast_model2.5`;
- write to the production database;
- overwrite any historical V1 artifact;
- silently restore `Q_L -> hard NEG/NEU/POS -> Markov`;
- use a hard 1-32 / 33-64 / 65-96 dependence split;
- use `Q_S(q)=Q_L(q)+Q_H(q)` as the principal reconstruction rule;
- introduce Transformer/global self-attention, dual complete High experts, Direction/Magnitude heads, oracle U or target-day RT/Spread features;
- change T24 lambda/operator or re-search the decomposition cutoff;
- tune on 2026-09-16..09-30 or any October data.

All new code must live under `src/research96/msr_spread_v2/**`. All new outputs must live under `outputs/experiments/msr_spread_v2/**`. V1/TafM code may be read as implementation reference, but V2 must own its implementation and tests.

---

## 1. Business-time and data contract

### 1.1 Forecast semantics

For target day `D`:

- forecast origin = `D-1 14:00`;
- prediction target = all 96 quarter-hours of `D` in one issuance;
- target = `S_D,t = RT_D,t - DA_D,t`;
- latest complete supervised label day = `D-2`;
- zero policy = positive iff `S>0`, zero belongs to non-positive.

### 1.2 V2 research assumption: target-day DA is fully visible

For this V2 research branch, **the full target-day day-ahead price curve is explicitly allowed as an input**:

`DA_D[1:96]` is legal.

This is a V2 research assumption and intentionally supersedes the earlier conservative `target_day_dayahead_price = forbidden` rule **inside this isolated V2 branch only**. It does not rewrite historical V1 conclusions or production contracts.

Forbidden target-day inputs remain:

- `RT_D`;
- `Spread_D`;
- any target-day realized/actual physical field that is not genuinely available at D-1 14:00;
- any derived feature containing these truths.

Because target-day DA is deterministic/known in this research setup:

`RT_hat_D,t = DA_D,t + S_hat_D,t`,

and therefore point-error identity holds:

`abs(RT_hat-RT) = abs(S_hat-S)`.

So improving Spread MAE directly improves RT MAE under this assumption.

### 1.3 Decision-day price context: only RT is truncated at p56

The p56 boundary applies to the **price context**, not to grid-operation information.

At D-1 14:00:

- `DA_{D-1}[1:96]` is available;
- `RT_{D-1}[1:56]` is available;
- `RT_{D-1}[57:96]` is unavailable.

Construct a fixed-length D-1 price context without pretending the suffix is observed:

`S_proxy[D-1,t] = RT[D-1,t] - DA[D-1,t]` for `t<=56`;

`S_proxy[D-1,t] = S[D-2,t]` for `t>=57`.

Then enforce price arithmetic consistency:

`RT_proxy[D-1,t] = DA[D-1,t] + S_proxy[D-1,t]` for `t>=57`.

Provide provenance channels:

- `price_observed_mask[t]=1` for t<=56, else 0;
- `price_proxy_mask[t]=1-price_observed_mask[t]`;
- `price_boundary_distance[t]` or a compact boundary-position embedding is allowed;
- never label the proxy suffix as actual RT/Spread.

The D-1 proxy curve is context only. It is never decomposed into supervised `L_true/H_true` and never enters the label ledger as truth.

### 1.4 Grid-operation information

For grid/physical variables, V2 uses the stated business assumption:

- complete `D-1 actual` 96-point curves are available;
- complete `D forecast` 96-point curves are available.

Examples include load, direct load, wind, solar, renewable total, tieline/import, local generation, nuclear, storage/pumped storage, bidding space and other already-authorized native-96 fields.

Where both D-1 forecast and D-1 actual are legally available, causal recent forecast-error context is allowed:

`err_{D-1,t}=actual_{D-1,t}-forecast_{D-1,t}`.

No proxy is needed for D-1 grid actuals.

---

## 2. Native-96 tensor contract

V2 does **not** reduce the model input to 24 hourly points. The model operates directly on a target-aligned 96-slot tensor.

For each target day D construct:

`X96[D] in R^[96, C]`.

Each row is one target quarter t. Historical information is aligned into channels rather than flattening the model into a 672-token global sequence.

### 2.1 Historical price channels

For complete mature lags `ell=2..8` (seven complete days), include same-slot:

- `DA_{D-ell,t}`;
- `RT_{D-ell,t}`;
- `S_{D-ell,t}`;
- `L_{D-ell,t}` from frozen T24;
- `H_{D-ell,t}` from frozen T24.

For D-1 include:

- full `DA_{D-1,t}`;
- `RT_proxy_{D-1,t}`;
- `S_proxy_{D-1,t}`;
- observed/proxy masks;
- causal prefix summaries/ramp features using only truly observed RT/Spread values through p56.

### 2.2 Target-day known price channel

Include full:

- `DA_D,t`;
- DA ramp/curvature/profile summaries derived solely from the known DA curve.

Do not create target-day RT or Spread placeholders.

### 2.3 Grid channels

At minimum preserve separate primitive channels for:

- D-1 actual curves;
- D target forecast curves;
- D-1 actual-minus-forecast error where legal;
- historical same-slot lag summaries;
- ramp15/ramp30/ramp60/curvature where semantically meaningful.

Do not collapse all primitives into a single opaque engineered score before the model.

### 2.4 Calendar / positional channels

Use deterministic future-known features such as:

- slot 1..96;
- quarter-in-hour;
- hour-of-day;
- day-of-week;
- holiday/working-day if already legal;
- month / season encoding.

Calendar/mask channels bypass spectral decomposition and are projected into both branches directly.

---

## 3. Frozen target decomposition: T24 remains the label coordinate system

V2 retains the accepted natural-trend T24 operator exactly.

For each complete target-day Spread label S:

`L_true = P_T24 S`,

`H_true = S - L_true`.

Use the already frozen natural-trend operator and lambda from V1/T24 authority. Do not reselect lambda, period or boundary condition.

Required invariant:

`max_abs(S - (L_true + H_true)) <= 1e-10` in float64 tests, subject only to the existing frozen numerical tolerance if the copied operator has a documented tolerance.

T24 determines **what the branches must predict**. It does not dictate how every input variable must be spectrally split.

---

## 4. Business-aware spectral routing

A single universal gate for every input variable is forbidden. Price, demand, renewable, supply/system and tightness variables have different physical frequency structures.

### 4.1 Feature families

At minimum define these routing families in a registry:

1. `PRICE`
   - historical DA / RT / Spread;
   - D-1 price proxy context;
   - target-day DA;
   - price ramps / volatility / DA profile shape.

2. `DEMAND`
   - total/direct load;
   - residual load components;
   - load forecast errors / ramps.

3. `RENEWABLE`
   - wind;
   - solar;
   - renewable total/share;
   - renewable ramps and forecast errors.

4. `SUPPLY_SYSTEM`
   - local generation;
   - tieline/import;
   - nuclear;
   - pumped storage / other supply-side variables with verified semantics.

5. `MARKET_TIGHTNESS`
   - bidding space;
   - legal reserve/scarcity/tightness proxies;
   - residual-space / load-space pressure primitives.

6. `CALENDAR_MASK`
   - calendar, slot and provenance masks; bypass spectral split.

Each source field must belong to exactly one primary spectral family or the bypass family. The registry is explicit and testable.

### 4.2 Per-family embedding

For each spectral family g, project its native scalar channels into a compact family embedding:

`E_g in R^[B,96,d_g]`.

Do not concatenate all raw fields and then learn one unconstrained gate.

### 4.3 DCT-based complementary router

Along the 96-slot dimension apply an orthonormal DCT-II/IDCT-III pair:

`A_g = DCT(E_g)`.

For one-shot portability, do not add a fragile DCT dependency. Precompute the orthonormal `96 x 96` DCT-II matrix once, register it as a non-trainable torch buffer, and use matrix multiplication/einsum in the forward path. Its transpose/inverse gives the orthonormal reconstruction. The implementation must be differentiable with respect to `E_g` and gate parameters.

Each family has its own learnable low-frequency response:

`m_g,D(k) = sigmoid((c_g(D) - k) / tau_g)`.

Use:

`c_g(D) = c0_g + delta_g * tanh(w_g^T context_g(D))`.

Contract:

- `c0_g`, `tau_g`, `delta_g`, `w_g` are family-specific;
- each family is initialized around the frozen T24 characteristic spectral scale, but all families learn independently;
- `c_g(D)` is bounded to a conservative valid frequency-bin interval, recommended `[1,32]`;
- `tau_g` is positive and bounded, recommended `[0.5,8]` bins;
- `delta_g` is bounded so one day cannot move a gate across the full spectrum;
- the gate is monotone low-pass by construction;
- High receives the exact complement, never a separately learned unrelated mask.

Then:

`A_L,g = m_g,D * A_g`,

`A_H,g = (1-m_g,D) * A_g`,

`E_L,g = IDCT(A_L,g)`,

`E_H,g = IDCT(A_H,g)`.

Required numerical invariant:

`E_g ~= E_L,g + E_H,g`.

This router is the input-information routing mechanism. It does not redefine T24 labels.

### 4.4 Semantic anchor channels

To keep Low/High meaning explicit even while raw inputs are adaptively routed:

- Low branch receives explicit historical `L_{D-2..D-8}` channels;
- High branch receives explicit historical `H_{D-2..D-8}` channels;
- D-1 proxy price channels remain context with masks and are not called true H/L.

This prevents the learned spectral router from erasing the target-coordinate semantics.

---

## 5. Core encoders

V2 is a lightweight neural model because end-to-end final loss must reach the routing, Low, High and dependence parameters. LightGBM remains a diagnostic/baseline only and is not the V2 Core backend.

Model parameter budget should remain small relative to ~1.7k business days. Initial engineering target: <=3 million trainable parameters, preferably substantially below.

### 5.1 Low branch

Inputs:

- all `E_L,g` spectral-family low representations;
- explicit L-history anchor channels;
- bypass calendar/mask embeddings;
- slow/business-level primitives such as target DA level/profile, load level, renewable share, residual-load level, broad market tightness.

Temporal path:

- Native 96 input/output;
- shallow long-receptive-field TCN;
- recommended channels 48;
- kernel size 3;
- dilations `{1,2,4,8,16,32}` with at most one compact residual stack;
- pre-LayerNorm or GroupNorm;
- GELU;
- dropout 0.05.

Local tabular path:

- research-local TabM mini wrapper using the installed official `tabm` package;
- ensemble size initial `k=4`;
- compact block width around 64;
- no edits/import-side effects on `src/TafM_改进源码`.

Fusion:

`g_L,t = sigmoid(MLP([h_L_time,t, h_L_tab,t, low_uncertainty_context_t]))`;

`h_L,t = LayerNorm(g_L,t*h_L_time,t + (1-g_L,t)*h_L_tab,t)`.

Fusion is bounded convex mixing, not unrestricted additive concatenation followed by a large generic MLP.

### 5.2 High branch

High is formally a residual-history branch.

Inputs:

- all `E_H,g` spectral-family high representations;
- explicit H-history anchor channels;
- D-1 price proxy context + mask;
- D-1 full grid actual context;
- D full grid forecast context;
- fast ramps / curvature / forecast-error primitives;
- Low-distribution context from the current model.

The High branch does not receive `H_true_D` or any target truth.

Temporal path:

- Native 96 input/output;
- phase-aligned dilated TCN;
- recommended channels 64;
- kernel size 3;
- dilations `{1,2,4,8}`;
- max two compact residual stacks;
- GELU;
- pre-LayerNorm/GroupNorm;
- dropout 0.05.

Local TabM path:

- per-slot nonlinear interaction of wind/load/solar/residual-load/bidding-space/DA-level/ramp/error context;
- official TabM mini, research-local wrapper, k=4 initially.

Low-to-High conditioning:

Pass a compact differentiable Low-distribution summary, initially:

- q10, q25, q50, q75, q90;
- low interval widths q90-q10 and q75-q25;
- optional projected `h_L` context through a small bounded gate.

During Stage B dependence warm-up the Low context can be stop-gradient; during Stage C joint fine-tuning gradients are allowed through it.

Fusion:

`g_H,t = sigmoid(MLP([h_H_time,t, h_H_tab,t, low_width_t, ramp_norm_t, price_proxy_mask_t]))`;

`h_H,t = LayerNorm(g_H,t*h_H_time,t + (1-g_H,t)*h_H_tab,t)`.

---

## 6. Monotone quantile-function heads

Both Low and High output differentiable monotone quantile functions, not isolated independently sorted quantile models and not a forced Gaussian family.

### 6.1 Quantile knots

Initial knots:

`Q_KNOTS = {0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95,0.99}`.

The wider ladder addresses the earlier finite-ladder endpoint problem while remaining estimable on the available data.

### 6.2 Structural monotonicity

For each slot:

- predict median `m`;
- predict positive softplus increments below and above the median;
- construct lower knots by cumulative subtraction;
- construct upper knots by cumulative addition.

Quantile crossing must be impossible by construction. Do not sort independently predicted quantiles after the fact.

### 6.3 Continuous inverse CDF

Implement `quantile(u)` through differentiable piecewise-linear interpolation between quantile knots.

Scenario `u` values are clamped to `[0.01,0.99]` in Core V2. Record clamp fraction as a diagnostic. Tail extrapolation is a later single-axis extension, not part of first implementation.

Outputs from both heads must expose:

- knot tensor `[B,96,Kq]`;
- `quantile(u)` API;
- median;
- q10/q90 and q25/q75 widths.

---

## 7. Smooth Dynamic Dependence Layer

V2 does not use global dependence, 96 independent slot parameters, or hard block boundaries.

The dependence layer must be low-dimensional, smooth across the day, data-adaptive, and differentiable.

### 7.1 What it models

There are two distinct dependence questions:

1. same-slot Low/High dependence: when Low realization is high/low within its marginal distribution, does High tend to move with it or counteract it?
2. temporal continuity: how persistent are Low and High latent ranks from slot t-1 to t?

The layer therefore learns three smooth curves per target day:

- `rho_t`: same-time Low/High innovation dependence;
- `phiL_t`: Low temporal persistence;
- `phiH_t`: High temporal persistence / mean reversion.

### 7.2 Smooth basis parameterization

Do not output 96 unrelated values.

Use a low-order smooth basis over t=1..96. Recommended first implementation: the first `K_DEP=6` orthonormal DCT basis functions, precomputed and fixed.

A compact day-context network outputs coefficient vectors:

`a_rho, a_phiL, a_phiH in R^[6]`.

Then:

`rho_t  = 0.95 * tanh(B_t @ a_rho)`;

`phiL_t = 0.98 * sigmoid(B_t @ a_phiL)`;

`phiH_t = 0.95 * tanh(B_t @ a_phiH)`.

Interpretation/constraints:

- Low persistence is nonnegative by construction;
- High may exhibit either persistence or mean reversion;
- Low/High dependence may be positive or negative;
- all three vary smoothly with t because they live in a six-dimensional basis;
- no p32/p33 or other artificial boundary exists.

Day context for the coefficient network can include pooled summaries of:

- Low/High fused representations;
- target DA profile;
- target grid forecast profile;
- D-1 actual grid profile;
- D-1 price proxy mask summary.

The dependence network must never ingest target S/RT truth.

### 7.3 Differentiable Gaussian-copula-like latent process

Use deterministic base noise during training from a fixed Sobol/normal transform so the path is reproducible. Recommended implementation: `torch.quasirandom.SobolEngine` -> clamp uniforms away from {0,1} -> standard-normal transform with `sqrt(2)*erfinv(2u-1)`. Cache the base draws by `(scenario_count, seed, device, dtype)`; they are buffers/noise, never trainable parameters.

For each scenario m and slot t:

`epsL_m,t`, `epsH_m,t` are independent standard-normal base noises.

Initialize the latent Low state with a standard-normal sample. Then for t>1:

`zL_t = phiL_t*zL_{t-1} + sqrt(1-phiL_t^2)*epsL_t`.

Create correlated High innovation:

`epsH*_t = rho_t*epsL_t + sqrt(1-rho_t^2)*epsH_t`.

Then:

`zH_t = phiH_t*zH_{t-1} + sqrt(1-phiH_t^2)*epsH*_t`.

Map to ranks:

`u_t = Phi(zL_t)`;

`v_t = Phi(zH_t)`.

Then evaluate marginals:

`L_t = Q_L,t(u_t)`;

`H_t = Q_H,t(v_t)`.

This construction lets the final Spread loss update the dependence parameters and both marginal branches.

### 7.4 Scenario counts

Recommended initial values:

- `M_TRAIN=32` Sobol scenarios per day/batch;
- `M_EVAL=256` scenarios for metrics/artifacts.

The base Sobol draws must be seeded/frozen for deterministic reruns. Scenario count is an engineering setting, not a model-selection axis in the first round.

### 7.5 Optional regularization

Because the basis is already smooth, only a small guardrail is allowed initially:

`L_dep_smooth = mean(diff(rho)^2) + 0.5*mean(diff(phiL)^2) + 0.5*mean(diff(phiH)^2)`.

Use a small frozen coefficient, recommended `lambda_dep_smooth=0.01`. Do not use hard persistence penalties or direction smoothing.

---

## 8. Exact scenario reconstruction and outputs

For each scenario m:

`S_m,t = L_m,t + H_m,t`.

This identity is exact at scenario level.

The scenario set represents final `F_S`.

### 8.1 Training-time differentiable outputs

CRPS is computed directly from scenarios:

`CRPS = mean_m |S_m-y| - 0.5*mean_{m,n}|S_m-S_n|`.

For Direction, exact empirical `mean(1[S_m>0])` is not differentiable in scenario values. Therefore training uses a smooth zero-threshold event probability:

`P_pos_soft = mean_m sigmoid(S_m / tau0)`.

Use normalized Spread units with initial/frozen `tau0=0.10`.

For the point guardrail, use a differentiable soft median:

1. sort scenarios along m;
2. apply fixed smooth rank weights centered at q=0.5, recommended rank bandwidth 0.08;
3. weighted sum gives `S_soft50`.

Sorting is piecewise differentiable and the rank window distributes gradient across central scenarios.

### 8.2 Evaluation outputs

At evaluation/inference report exact empirical outputs:

- `Spread_median = median_m(S_m)`;
- q01/q05/q10/q25/q50/q75/q90/q95/q99 from scenarios;
- `P_pos = mean_m 1[S_m>0]`;
- `RT_median = DA_D + Spread_median`;
- RT quantiles = deterministic DA shift of Spread quantiles.

Also persist component outputs:

- Low quantile knots;
- High quantile knots;
- rho/phiL/phiH curves;
- spectral family gate curves/cutoffs;
- scenario clamp fraction.

---

## 9. Unified loss and gradient design

Hard Direction accuracy is an evaluation metric, not a training loss.

All continuous component targets are robustly standardized using training-only statistics fitted inside each target-day training set.

### 9.1 Component losses

Use quantile/pinball approximation to marginal CRPS at the fixed knot set:

`L_low = 2 * sum_q w_q * pinball(L_true, Q_L(q))`;

`L_high = 2 * sum_q w_q * pinball(H_true, Q_H(q))`.

Weights approximate integration over quantile levels; normalize them to sum to one. Do not manually overweight sign classes in Core V2.

### 9.2 Final loss

Use:

`L_final = L_CRPS_S + 0.30*L_Brier0 + 0.25*L_point`.

where:

- `L_CRPS_S` = scenario CRPS on standardized Spread;
- `L_Brier0 = mean((P_pos_soft - 1[S_true>0])^2)`;
- `L_point = SmoothL1(S_soft50, S_true)` on standardized Spread.

### 9.3 Total joint loss

Initial frozen V2 objective:

`L_total = L_final + 0.20*L_low + 0.20*L_high + 0.01*L_dep_smooth`.

These weights are architecture constants for the first V2 implementation. Do not grid-search them on FAST-24. Change them only after a dedicated gradient-scale/conflict audit authorizes a new protocol.

### 9.4 Why this objective is coherent

- CRPS trains the whole final distribution;
- Brier@0 gives extra proper-score emphasis to the exact business threshold `S>0`;
- SmoothL1 protects point/RT numerical usefulness;
- Low/High auxiliary proper scores prevent the two components from becoming arbitrary cancelling coordinates;
- the final loss has direct differentiable paths through reconstruction, dependence, both quantile functions, both branches and the business-aware spectral router.

### 9.5 Gradient audit is mandatory

During development/training persist gradient norms from each loss family to at least:

- PRICE spectral gate;
- DEMAND spectral gate;
- RENEWABLE spectral gate;
- Low temporal encoder;
- Low TabM encoder;
- Low quantile head;
- High temporal encoder;
- High TabM encoder;
- High quantile head;
- dependence coefficient network.

Also compute cosine between `grad(L_final)` and `grad(L_low/L_high)` for Low/High shared parameter groups where feasible.

A model is not accepted merely because total loss decreases. If final gradients are effectively zero at an upstream group, implementation is incomplete/blocked.

---

## 10. Training schedule inside each DAILY_ASOF_V1 fit

For each target day D use only rows with labels `<=D-2`.

No warm-start from D-1 model in the scientific baseline.

### Stage A — component pretraining

Train shared embeddings/router + Low + High marginal branches using:

`L_A = L_low + L_high`.

Dependence/reconstruction is disabled.

Recommended max 30 epochs, early stopping on an internal chronological validation tail derived solely from `T_D`.

### Stage B — dependence/reconstruction warm-up

Freeze branch encoders/quantile heads temporarily and train the dependence layer/reconstruction using `L_final + 0.01*L_dep_smooth` for a short warm-up, recommended max 8 epochs.

This prevents random dependence parameters from destabilizing already meaningful marginals.

### Stage C — joint fine-tune

Unfreeze all V2 parameters and optimize `L_total`.

Recommended max 20 epochs with small learning rate and early stopping on final CRPS primary, Brier/point as guardrails.

### 10.1 Inner validation and final refit

For a target day D:

1. reserve the most recent 28 mature days inside `T_D` as inner validation;
2. choose epoch counts/early stopping using only that internal validation;
3. refit once on all of `T_D` for the selected epoch counts if practical;
4. predict only D.

If exact refit materially increases runtime, the implementation may expose a clearly labeled `NO_REFIT_FAST_DEV` profile for code debugging only. Scientific FAST-24 comparison must use one frozen profile consistently and must never mix the two silently.

### 10.2 Initial optimizer profile

Recommended first implementation:

- AdamW;
- Stage A lr 1e-3;
- Stage B lr 5e-4;
- Stage C lr 2e-4;
- weight decay 1e-4;
- batch = complete days, initial 16 or 32 depending on memory;
- gradient clip global norm 1.0;
- deterministic seed 20261009;
- CPU is a supported baseline; CUDA is optional if available, not required.

Do not trigger an architecture sweep to tune these values in the first implementation.

---

## 11. State and U extension hooks — implement interfaces, not functionality

### 11.1 State/R hook

Core V2 is K=1 and has no State classifier.

Future regime extension is permitted only as:

`p_L(L|X)=sum_{r=1}^K gamma_r(X) p_{L,r}(L|X)`, normally K=3.

Rules:

- gamma is soft simplex context;
- no hard argmax State enters High;
- no post-hoc `Q_L -> State` pipeline;
- temporal continuity remains in the dynamic dependence layer unless a future protocol proves latent Markov transition adds value;
- Viterbi, if ever used, is diagnostic only.

V2 code should therefore define a minimal `RegimeContext` interface/default `K=1` identity implementation so K=3 can later be added without rewriting Low/High/reconstruction.

### 11.2 Physical mediator U hook

Core V2 does not use U. Define no trainable U branch now.

The High branch API may reserve an optional `u_context=None` argument. A future U module must prove incremental value before this argument is populated.

---

## 12. Code isolation and file-by-file implementation map

Create a new package only:

`src/research96/msr_spread_v2/`

Recommended files and responsibilities:

- `__init__.py` — package version/status only.
- `config.py` — all frozen constants, paths, feature-family registry, hyperparameters, FAST-24 dates.
- `contracts.py` — V2 data/leakage/business-time invariants, including full target DA allowance.
- `dataset.py` — target-aligned Native-96 tensor construction; no model logic.
- `price_context.py` — D-1 RT p56 proxy construction, arithmetic consistency and masks.
- `t24.py` — local frozen copy/adaptation of accepted T24 operator with provenance comment/hash; do not modify/import mutable V1 source at runtime.
- `feature_registry.py` — every input field assigned to PRICE/DEMAND/RENEWABLE/SUPPLY_SYSTEM/MARKET_TIGHTNESS/CALENDAR_MASK.
- `embeddings.py` — family-specific projections.
- `spectral_router.py` — orthonormal DCT/IDCT and business-aware complementary gates.
- `tabm_encoder.py` — local wrapper around installed official `tabm`; no edits to frozen TafM code.
- `temporal.py` — LowTCN and HighTCN.
- `quantile_head.py` — structural monotone quantile knots + differentiable interpolation.
- `low_branch.py` — Low temporal/tabular fusion and distribution head.
- `high_branch.py` — High residual-history temporal/tabular fusion, Low conditioning and distribution head.
- `dependence.py` — smooth basis curves + Sobol latent process.
- `reconstruct.py` — scenario S=L+H, CRPS helpers, soft/exact point and P-positive functions.
- `losses.py` — component/final/joint objectives and loss decomposition.
- `model.py` — one `MSRSpreadV2` forward graph; no training loop.
- `trainer.py` — Stage A/B/C training, inner validation, gradient audit.
- `walkforward.py` — docs/82 DAILY_ASOF_V1 / FAST-24 target-day loop.
- `evaluate.py` — point/direction/probability/calibration metrics.
- `artifacts.py` — non-overwriting manifests, hashes, prediction/scenario summaries.
- `audit.py` — read-only/source/code isolation and leakage audits.
- `run_experiment.py` — CLI only; no hidden model logic.
- `README.md` — architecture summary, commands, output schema, known limitations.
- `tests/` — V2-only tests.

### 12.1 Reuse policy

The coding agent must inspect V1 and frozen TafM implementations before coding, but reuse is controlled:

- pure logic may be copied/adapted into V2 with a provenance comment naming source file and frozen scientific contract;
- no V2 runtime import from `src/TafM_改进源码`;
- avoid V2 runtime imports from `msr_spread_v1` except immutable generic helpers only if absolutely necessary; preferred path is a local V2 implementation for scientific-critical logic;
- V1 files must hash identically before/after V2 implementation;
- frozen TafM files must hash identically before/after.

---

## 13. Required tests before any scientific run

At minimum implement and pass the following test families.

### T01 — write/isolation guard

- V2 writes only in V2 package/output roots;
- production root rejects writes;
- V1/TafM source hashes unchanged after tests/smoke.

### T02 — D-2 label maturity

For every target D, fit rows satisfy `label_day<=D-2`.

### T03 — target-day feature legality

- full `DA_D` present;
- `RT_D`, `Spread_D`, target-day forbidden actuals absent.

### T04 — D-1 price proxy

- t<=56 uses actual RT/Spread;
- t>=57 uses D-2 same-slot Spread proxy;
- `RT_proxy=DA_{D-1}+S_proxy` exactly;
- masks identify proxy suffix;
- no future D-1 RT truth is read when building eligibility/features.

### T05 — grid data contract

- D-1 grid actual full96 accepted;
- D target grid forecast full96 accepted;
- no target actual leakage.

### T06 — T24 invariants

- frozen operator identity/hash;
- exact `S=L+H` reconstruction;
- no re-selection.

### T07 — feature-family registry

- every spectral field maps to exactly one family;
- calendar/mask bypass family is explicit;
- unknown field fails closed.

### T08 — family spectral router

For each family:

- DCT/IDCT numerical inversion;
- gate in [0,1];
- low/high complementary;
- `E ~= E_L+E_H`;
- family parameter objects are distinct;
- cutoff/slope bounded;
- day-conditioned movement bounded;
- gate finite for all legal inputs.

### T09 — monotone quantile function

- knots strictly nondecreasing;
- interpolation finite;
- gradients reach all head parameters;
- u clamp is audited.

### T10 — High residual-history contract

- High receives true H only from complete mature history;
- D-1 proxy never labeled H_true;
- target H truth never enters forward features.

### T11 — dynamic dependence validity

- `rho in (-.95,.95)`;
- `phiL in (0,.98)`;
- `phiH in (-.95,.95)`;
- no hard block boundary;
- smooth-basis curves finite;
- marginal normal ranks remain approximately standard/uniform in simulation smoke;
- same seed gives same scenarios.

### T12 — scenario reconstruction

For every scenario:

`S_scenario == L_scenario + H_scenario` within numerical tolerance.

### T13 — final output consistency

- exact eval median/P-positive finite;
- `RT_hat=DA_D+Spread_hat` exactly;
- zero is non-positive.

### T14 — differentiable final losses

Final CRPS/Brier/point losses produce non-zero finite gradients to:

- at least one parameter in each spectral family gate;
- Low TCN/TabM/head;
- High TCN/TabM/head;
- dependence coefficient network.

### T15 — loss correctness

Compare CRPS/scenario and pinball helpers against small hand-computable examples.

### T16 — Stage A/B/C freeze semantics

- Stage A dependence params unchanged;
- Stage B branch params frozen;
- Stage C all intended params trainable.

### T17 — daily walk-forward determinism

Same seed/data/config reproduces identical selected target-day predictions within declared deterministic tolerance.

### T18 — CPU integration smoke

At least two legal target dates complete:

`dataset -> Stage A/B/C short smoke -> scenario reconstruction -> metrics -> artifacts`.

No scientific metric claim is allowed from this smoke.

---

## 14. Experiment runner and artifact contract

CLI should support at least:

- `--mode smoke --date YYYY-MM-DD`;
- `--mode fast24-lowhigh` or equivalent Core FAST-24 path;
- `--mode fast24-full` after implementation readiness;
- `--mode single-day --date ...`;
- `--device cpu|cuda|auto`;
- `--profile scientific|no-refit-fast-dev`.

Artifacts per run must include:

- `RUN_MANIFEST.json` with source/code/config hashes;
- target dates and max label day per target;
- feature-family registry snapshot;
- spectral gate parameters and daily cutoffs;
- Low/High quantile outputs;
- dependence rho/phi curves;
- final Spread/RT predictions and P-positive;
- metrics JSON/CSV/parquet;
- gradient audit;
- wall-clock split by data/model/stages/scenario/evaluation;
- explicit flags stating whether the run is smoke, retrospective diagnostic, or fresh-confirm eligible.

Never overwrite a prior run directory.

---

## 15. Initial evaluation metrics

Every V2 evaluation reports both value and direction. No single metric is allowed to hide a collapse in another family.

Point/value:

- Spread MAE;
- Spread RMSE;
- RT MAE (under known target DA this should numerically equal Spread MAE);
- monthly/by-slot diagnostics.

Direction/probability:

- Raw Direction accuracy;
- Balanced accuracy;
- positive recall;
- negative recall;
- AUC;
- Brier@0;
- predicted positive fraction vs truth.

Distribution:

- CRPS;
- quantile coverage for q01/q05/q10/q25/q50/q75/q90/q95/q99;
- interval width / under-dispersion diagnostics.

Component diagnostics:

- Low MAE/CRPS/correlation;
- High MAE/CRPS/correlation;
- gate cutoffs by family/day;
- rho/phi curve ranges and smoothness;
- scenario clamp fraction.

---

## 16. FAST-24 and training paradigm

Training cadence and dates remain governed by docs/82.

The coding agent must copy the fixed FAST-24 list into V2 config without changing dates from observed results.

V2 architecture work does not open late-September fresh confirm or October.

Initial V2 implementation can run smoke only. Scientific comparison starts only after this architecture implementation passes all engineering tests and an independent review confirms the design was implemented literally.

---

## 17. Implementation order for a one-shot coding agent

The coding agent should proceed in this exact order and should not stop after planning:

1. Read mandatory docs and inspect V1/TafM source read-only.
2. Snapshot hashes of `src/research96/msr_spread_v1/**` and `src/TafM_改进源码/**`.
3. Create `src/research96/msr_spread_v2/` only.
4. Implement contracts/config/feature registry/price context/T24 first.
5. Implement dataset tensorization and leakage tests before model code.
6. Implement business-aware spectral router + tests.
7. Implement Low/High TCN and local TabM wrappers.
8. Implement monotone quantile heads + tests.
9. Implement smooth dynamic dependence layer + simulation tests.
10. Implement reconstruction/losses + hand-check tests.
11. Implement integrated model.
12. Implement Stage A/B/C trainer and gradient audit.
13. Implement DAILY_ASOF/FAST-24 runner and artifacts.
14. Run V2 unit tests.
15. Run CPU two-date integration smoke.
16. Rehash V1/TafM and prove zero source drift.
17. Update only `src/research96/msr_spread_v2/README.md` and V2-local implementation notes if needed; do not rewrite historical V1 docs.
18. Report exact changed files, tests, runtime and blockers.

If a dependency/API mismatch prevents literal implementation, the agent must fail closed and report it; it must not silently swap in another architecture.

---

## 18. Acceptance token

The implementation may report:

`MSR_V2_CORE_IMPLEMENTATION_READY`

only if all of the following are true:

- V2 package exists under the isolated root;
- all T01-T18 applicable tests pass;
- CPU integration smoke completes;
- final gradients reach router + Low + High + dependence;
- V1/TafM/production sources have zero drift;
- no forbidden truth feature is used;
- target-day DA is the only newly allowed target-day price curve;
- D-1 RT suffix is proxy+mask, not future truth;
- exact scenario identity S=L+H holds;
- no post-hoc hard State/Markov was reintroduced.

Otherwise the only valid terminal status is:

`MSR_V2_CORE_IMPLEMENTATION_BLOCKED`

with concrete blockers.

---

## 19. Architecture summary

The authoritative V2 graph is:

```text
Legal Native-96 information at D-1 14:00
  |
  |-- 7 complete mature price histories: DA / RT / Spread + T24 L/H anchors
  |-- D-1 price context: full DA + RT p1-p56 + D-2-spread proxy suffix + mask
  |-- target-day full DA
  |-- D-1 full grid actual
  |-- target-day full grid forecast
  |-- calendar / provenance masks
  v
Target-aligned X96 [96,C]
  |
  +--> PRICE family embedding -------- family-specific learnable DCT gate --+--> Low price repr
  |                                                                          +--> High price repr
  +--> DEMAND family embedding ------ family-specific learnable DCT gate --+--> Low demand repr
  |                                                                          +--> High demand repr
  +--> RENEWABLE embedding ---------- family-specific learnable DCT gate --+--> Low renew repr
  |                                                                          +--> High renew repr
  +--> SUPPLY_SYSTEM embedding ------ family-specific learnable DCT gate --+--> Low supply repr
  |                                                                          +--> High supply repr
  +--> MARKET_TIGHTNESS embedding --- family-specific learnable DCT gate --+--> Low tight repr
  |                                                                          +--> High tight repr
  +--> CALENDAR/MASK bypass -----------------------------------------------> both branches

Low side:  routed low representations + L-history anchors
           -> Long-receptive TCN + local TabM
           -> bounded fusion
           -> monotone quantile function Q_L(u)
                                     |
                                     | differentiable distribution context
                                     v
High side: routed high representations + H-history anchors
           + D-1 proxy price context/mask + D-1 grid actual + D forecast
           -> Dilated TCN + local TabM
           -> bounded fusion
           -> monotone quantile function Q_H(v)

Low/High/day context
  -> smooth dynamic dependence network
  -> rho_t, phiL_t, phiH_t via 6-basis curves
  -> Sobol latent ranks (u_t, v_t)
  -> L_m = Q_L(u_m), H_m = Q_H(v_m)
  -> exact S_m = L_m + H_m
  -> final F_S
       |-- Spread median / quantiles
       |-- P(S>0)
       `-- RT = target DA + Spread

Training:
  Stage A: Low/High marginal proper scores
  Stage B: dependence + final reconstruction warm-up
  Stage C: joint CRPS_S + Brier@0 + point + Low/High auxiliary losses
```

This graph, not any earlier C1/V1 pipeline diagram, is the implementation authority for MSR-V2.
