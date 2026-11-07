# How to Approach This Problem From Scratch

A practical guide for multivariate time series forecasting with known future covariates,
grounded in what worked and what didn't across six experiments on this dataset.

---

## Index

- [The mindset](#the-mindset-read-the-data-before-reading-papers)
- [Phase 1 — Understand the problem setup (before any code)](#phase-1--understand-the-problem-setup-before-any-code)
- [Phase 2 — Study the data](#phase-2--study-the-data)
- [Phase 3 — Set up honest evaluation first](#phase-3--set-up-honest-evaluation-first)
- [Phase 4 — Start with the simplest model that could work](#phase-4--start-with-the-simplest-model-that-could-work)
- [Phase 5 — Feature engineering](#phase-5--feature-engineering)
- [Phase 6 — Model architecture decision](#phase-6--model-architecture-decision)
- [Phase 6.5 — Three classes of forecasting methods](#phase-65--three-classes-of-forecasting-methods)
- [Phase 6.6 — Coverage ledger after six experiments](#phase-66--coverage-ledger-after-six-experiments)
- [Phase 6.7 — Deep learning architecture roadmap](#phase-67--deep-learning-architecture-roadmap)
  - [6.7.1 The framing constraint — known future covariates](#671-the-framing-constraint--known-future-covariates)
  - [6.7.2 Tier 1 — Architectures designed for known future covariates](#672-tier-1--architectures-designed-for-known-future-covariates)
  - [6.7.3 Tier 2 — Strong general-purpose architectures (need adaptation)](#673-tier-2--strong-general-purpose-architectures-need-adaptation)
  - [6.7.4 Tier 3 — Foundation models (zero-shot or fine-tuned)](#674-tier-3--foundation-models-zero-shot-or-fine-tuned)
  - [6.7.5 Tier 4 — Diagnostic baselines (cheap, informative)](#675-tier-4--diagnostic-baselines-cheap-informative)
  - [6.7.6 Library landscape](#676-library-landscape)
  - [6.7.7 Recommended experiment order](#677-recommended-experiment-order)
- [Phase 7 — Align the objective to the metric](#phase-7--align-the-objective-to-the-metric)
- [Phase 8 — Residual analysis before hyperparameter tuning](#phase-8--residual-analysis-before-hyperparameter-tuning)
- [Phase 9 — Hyperparameter tuning](#phase-9--hyperparameter-tuning)
- [Phase 10 — Further improvements (advanced)](#phase-10--further-improvements-advanced)
- [Phase 11 — When to stop](#phase-11--when-to-stop)
- [Summary checklist](#summary-checklist)

---

## The mindset: read the data before reading papers

The single biggest mistake in this project was deciding "this is a time series → build an LSTM"
before actually understanding the problem. Three experiments failed because of that. The
questions below, answered honestly before any modeling, would have saved all of it.

---

## Phase 1 — Understand the problem setup (before any code)

**This is the most important phase. Do not skip it.**

### 1.1 What is available at inference time?

Open `validation_input.csv` and count its columns. Compare to `train.csv`. The difference
tells you everything.

- If `validation_input.csv` is missing the `target` column but has all the forecast features,
  the problem is a **per-hour regression on known covariates**, not a sequence forecast.
  Future covariates are already provided to you. An autoregressive rollout is solving a problem
  that doesn't exist.
- If `validation_input.csv` had only timestamps and nothing else, *then* you would need a
  sequence model that predicts forward from its own outputs.

**This dataset falls into the first category.** The answer changes everything downstream.

### 1.2 What is the metric?

- WAPE = total absolute error / total actual values. Structurally identical to MAE but
  weighted by the sum of actuals.
- Training with L2 (MSE) is misaligned: it chases big-error rows at the expense of
  small-value rows. Use MAE (L1) as your training objective. This alone was worth −1.08 WAPE.
- WAPE has a small-denominator problem: rows with low actual values inflate the metric even
  when the absolute error is tiny. Know this so you don't chase phantom improvements on
  low-value rows that are really just a math artifact.

### 1.3 What does "per series" mean?

There are 96 independent units. Each has its own target distribution. Per-series mean
target ranges from ~6 to ~14 in this dataset — a 2× spread. A single global model that
ignores this structure will systematically over- or under-predict for many units.

---

## Phase 2 — Study the data

Do this in a notebook before writing a single model line.

### 2.1 Feature-target correlations

Compute correlation (or r²) of each feature against the target, both pooled and within
each series. For this dataset:

| feature | r² |
|---|---|
| `queue_pressure_forecast` | 47% |
| `network_pressure_forecast` | 43% |
| `workload_intensity` | 30% |

Two features explained ~90% of the variance. A few features were near zero. If you had
seen this early, you would have known a linear regression on two features was already
competitive. Features with r² < 1% (`staffing_forecast`, `throughput_disruption_risk_forecast`,
`nominal_capacity`) can be dropped unless they improve out-of-sample performance.

Do this on data held out of training — not the training data itself. If the correlation
holds on data the model hasn't touched, it's real. If it collapses, your training set was
misleading.

### 2.2 NaN patterns

Identify the fraction of missing values in each feature and the gap length distribution.
In this dataset, ~4–5% of forecast feature values were missing, in gaps of 1–3 hours.
If you naively dropped all rows with any NaN, you'd lose ~40% of training data. The
correct imputation recipe is **`ffill().bfill()` within each series, sorted by
timestamp**.

`data-analysis-2.ipynb` verified this end-to-end:

- Gaps are uniformly short — max **4 hours** across all 96 × 10 = 960 (series, feature)
  pairs, mean ~1.05h, p99 ≤ 3h.
- Missingness is **column-independent** — pairwise NaN-mask correlations ≤ 0.004; 80.5%
  of any-missing rows are missing exactly one feature. The 36.9% "any-missing" rate
  decomposes cleanly as ~10 independent 4.5%-per-column Bernoulli events, not as block
  missingness.
- Missingness is **effectively MCAR in time** — max hour-of-day deviation 0.81 percentage
  points on a 36.9% base rate (~2σ noise on the per-hour binomial). No structural
  hour-of-day or day-of-week pattern.
- → `ffill` carries the last observed value across short gaps with near-zero error,
  exploiting strong local autocorrelation. `bfill` catches the rare leading-NaN edge
  case (first timestamp of a series).
- → No row-level missingness indicator (`n_missing_features`, `any_missing`) is
  needed, because both the block-missingness check (§3) and the temporal-clustering
  check (§4) came back clean.

The earlier `ffill().fillna(0)` recipe (used in exp-1 through exp-5) is also viable on
this data because leading-NaN rows are very rare. But it injects zeros into features
whose typical values are in the single-digits-to-tens range, which is wrong in
principle. Switching to `ffill().bfill()` in exp-5.1 moved leaderboard WAPE by +0.004
(within fold noise) — a correctness fix, not a feature win. Adopt it as the principled
default.

### 2.3 Target stationarity

Split the training period into thirds and compare means. For this dataset they were
8.93, 9.40, 9.09 — essentially flat. This means: whatever you train on the last 14 days
will generalize to the validation window. Distribution shift is not your problem.

### 2.4 Per-series spread

Compute the WAPE baseline for each individual series separately. If the best-case WAPE
is 8.5 and the worst is 28, that 19-point spread is a direct opportunity — one global
model is badly averaged. Adding per-series structure will help.

---

## Phase 3 — Set up honest evaluation first

This must come before modeling.

### 3.1 Time-respecting split

Never use random splits for time series. Your val set must come *after* your train set in
time. For this dataset: take the last 336 hours of each series as the test block, train
on everything before it. A random split lets the model see future windows during training
and gives fake local scores.

### 3.2 Rolling-origin cross-validation

A single held-out window can be lucky or unlucky. Use at least 3 folds:

- Fold 0: train on days 1–138, test days 139–152.
- Fold 1: train on days 1–152, test days 153–166.
- Fold 2: train on days 1–166, test days 167–180.

Report mean ± std. The std tells you whether an improvement is real or noise. In this
dataset a single-split score of 16.53 turned into CV 17.27 ± 1.22 — the spread across
folds was bigger than many of the improvements being chased.

### 3.3 Compute `series_mean` per fold

If you add a per-series mean feature, recompute it from each fold's training data only.
Don't compute it on the full dataset and then use it in all folds — that's a small data
leak (about 8% of the held-out targets sneak in through the mean). The effect is modest
here because the target is stationary, but the principle matters.

### 3.4 Establish baselines

Before training anything, score the simple baselines:

- `naive_last_value`: predict the last observed value.
- `lag24_repeat`: repeat the value from 24h ago.
- `seasonal_mean`: per-series, per (hour-of-day × day-of-week) average.

`seasonal_mean` was the best at WAPE ~33. Any model you build must beat this. It is not
a high bar.

---

## Phase 4 — Start with the simplest model that could work

Before any neural network, run:

```python
from sklearn.linear_model import Ridge
# Features: top 3-5 forecast columns
# Target: target
```

Evaluate on your CV. If linear regression beats the baseline, the features have real
signal and you can proceed. If it doesn't, feature engineering is the problem, not the
model architecture.

For this dataset:
- Linear regression on domain features → WAPE 21.
- Linear regression + `series_mean` → WAPE 18.
- LightGBM + all features + `series_mean` → WAPE 16.5.

The linear regression result should be your sanity check. If it fails, stop and fix your
features before touching a neural network.

---

## Phase 5 — Feature engineering

Given that this is a direct regression problem (future covariates provided), your feature
space is:

### 5.1 Core features (always include)

- All domain forecast features (after NaN imputation).
- Time encodings: hour-of-day, day-of-week as sin/cos, weekend flag. These capture
  periodic patterns that the domain features don't fully explain.
- `series_mean`: the per-series average target from the training fold. This is the
  cheapest per-series feature you can add and worth ~3 WAPE on its own.
- `series_id` as a categorical: lets the model learn per-series offsets without the
  information bottleneck of a single number. Worth another 0.6–1.0 WAPE in LightGBM.

### 5.2 Temporal context features (add after the core model works)

Since `validation_input.csv` provides the full domain trajectory across the forecast
horizon, you can compute both lags AND leads of the domain features — all are known at
inference time.

For the top 5 domain features, add:
- `feature_lag_1`, `feature_lag_24` — short-term and day-prior context.
- `feature_lead_1`, `feature_lead_24` — where things are heading.
- Rolling mean (6h, 24h) and rolling std (6h, 24h) for the top 3 features.

That's ~32 extra features, all leakage-free. Worth about −0.6 WAPE on top of the core model.

**Important implementation detail**: when computing lags for the val window, the lag at
the start of val must reach back into training data. Concatenate `train.csv` and
`validation_input.csv` before computing temporal features, then split them back after.

### 5.3 What not to include

- `nominal_capacity` — near-zero correlation with target, no improvement empirically.
- `trend` — the target has no trend across the 180-day window (chunk means: 8.93, 9.40, 9.09).
- Target lags — do not include these. They don't exist at inference time. If you include
  them during training, you're training on a problem that the deployment setup doesn't
  support.

---

## Phase 6 — Model architecture decision

### If future covariates are provided (this dataset)

The problem is **tabular regression**, not sequence forecasting.

**Best default: LightGBM (or XGBoost).**

Why GBMs over neural networks for this problem:
- Handles mixed-scale features natively (some at scale 10, some at 0.01).
- Handles 96 categorical series natively with `categorical_feature`.
- Handles NaNs natively without imputation.
- Moderate data per series (4,320 rows × 96 series) — neural networks would need
  careful normalization, embedding, regularization for marginal gain.
- Fast to iterate — a full 3-fold CV runs in minutes on CPU.

LightGBM starting config:
```python
params = dict(
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=50,
    reg_lambda=1.0,
    objective='regression_l1',   # MAE — match the metric
    categorical_feature=['sid_code'],
)
```

### If future covariates are NOT provided (true sequence forecasting)

Then you need a model that predicts forward from its own outputs or from temporal patterns
in the history. Options in rough order of ease-to-deploy:

| model | notes |
|---|---|
| N-BEATS / N-HiTS | Fast, interpretable, strong forecasting baseline. No external features. |
| PatchTST | Transformer treating time series as patches. Good for long sequences. |
| TiDE | MLP-based, efficient, handles covariates cleanly. |
| iTransformer | Attends across variate dimension instead of time. Good for multivariate. |
| LSTM / GRU | Fine as a baseline, but harder to train and usually beaten by the above. |

For any sequence model: use teacher forcing during training *only if* you will have real
targets at inference. If you won't (like here), train with scheduled sampling or direct
multi-step output to match the inference setup.

---

## Phase 6.5 — Three classes of forecasting methods

Multivariate time series forecasting methods fall into three broad classes. Knowing
where each method sits — and which classes this project has actually explored — clarifies
what to try next.

### 6.5.1 Classical forecasting methods

Statistical methods with well-understood theory. They typically model the target series
directly with parametric or semi-parametric structure (trend, seasonality, AR, MA terms)
and treat external covariates as optional extensions.

**Strengths:**
- Interpretable, fast, low hyperparameter sensitivity.
- Strong on stationary seasonal data with few covariates.
- Per-series fitting is natural — each unit gets its own model with no
  cross-contamination.

**Weaknesses:**
- Struggle to incorporate many exogenous features cleanly.
- Assume specific distributional / structural forms.
- Weak when the signal is in non-linear feature interactions.

**Examples:**
- **Naive baselines** — `naive_last_value`, `lag24_repeat`, `lag168_repeat`, `seasonal_mean`.
- **SARIMA / SARIMAX** — seasonal ARIMA with optional exogenous regressors.
- **ETS / Exponential Smoothing** — state-space models with trend and seasonality.
- **Theta method** — M3 competition winner; decomposes target into theta lines.
- **Prophet** (Facebook) — trend + seasonality + holidays; often beaten by simpler models.
- **TBATS** — trigonometric seasonality + Box-Cox + ARMA errors. Good for multiple seasonalities.

**On this dataset:**
- Naive baselines in `data-analysis.ipynb` §13: WAPE 33–48.
- `seasonal_mean` (per-series day-of-week × hour-of-day mean) → CV-WAPE **~33**, leaderboard 34.4.
- Best of the simple baselines; the bar every other model must clear.

### 6.5.2 Machine learning methods

Tabular learning over engineered features. Gradient-boosted trees, random forests,
linear models. These treat the problem as "map features at time `t` → target at time `t`" —
exactly the framing this dataset's inference setup demands (Phase 1).

**Strengths:**
- Handles mixed-scale features, NaNs, and categorical features natively.
- Captures non-linear interactions through tree splits.
- Fast to iterate; minimal preprocessing.
- CV-ready with reasonable hyperparameters out of the box.

**Weaknesses:**
- Doesn't model sequence structure (must be hand-engineered via lag/lead features).
- Limited if features are noisy or weakly informative.
- Ensemble diversity is hard to extract when models share features
  (see exp-5 post-mortem — the 4-model blend collapsed to ≈ MAE alone because all
  base models trained on the same 52 features).

**Examples:**
- **LightGBM** — leaf-wise growth, fastest of the three boosters, native categorical handling.
- **XGBoost** — level-wise growth, similar accuracy, slightly slower.
- **CatBoost** — ordered boosting + symmetric trees, best for high-cardinality categoricals.
- **Random Forest** — bagging baseline.
- **Linear models** (Ridge, Lasso, ElasticNet) — sanity check; competitive with the right features.

**On this dataset:**
- **Experiment 3.2** — linear regression + LightGBM with domain features + `series_mean`.
  Linear: WAPE 18; LightGBM: WAPE 16.5 (single split).
- **Experiment 4** — LightGBM + MAE objective + `sid_code` categorical + temporal context
  (lags/leads/rolling). CV-WAPE **14.80 ± 0.64**; leaderboard 15.48.
- **Experiment 5** — LightGBM with multi-seed averaging on top of exp-4. CV-WAPE
  **14.73 ± 0.62**, leaderboard **14.92**. Current best.

### 6.5.3 Deep learning methods

Neural network architectures designed for sequential, temporal, or tabular data.
Operate either on raw sequences (LSTM, Transformer) or on decomposed/patched views
(N-BEATS, PatchTST).

**Strengths:**
- Can learn complex temporal structure end-to-end.
- Effective with large data and many input series.
- Flexible architectures for different input shapes (covariates, static features,
  multiple series).

**Weaknesses:**
- Require careful normalization, embedding design, and regularization.
- Typically need lots of data per series.
- Slow to iterate.
- Brittle to data leakage and split bugs (see exp-1 random-split failure).
- Need explicit handling of future covariates vs. auto-regressive rollout — getting
  this wrong is fatal (see exp-1 through exp-3.1).

**Examples:**
- **LSTM / GRU** — sequence-to-sequence with autoregressive rollout. Old default; mostly outperformed by the below.
- **N-BEATS / N-HiTS** — pure feed-forward stacks with backcast/forecast residuals.
  Strong forecasting baseline; no external features by default.
- **TFT (Temporal Fusion Transformer)** — explicitly handles known future inputs +
  static covariates. Theoretically the best DL match for this dataset's shape.
- **PatchTST / iTransformer** — Transformers over time patches or variate dimension.
- **TiDE** — MLP-based, efficient, handles covariates cleanly. Often beats Transformers
  with less compute.
- **DeepAR** — probabilistic recurrent model with quantile outputs.
- **MLP with series embedding** — simplest neural option; treats the problem as tabular regression with a learned per-series embedding.

**On this dataset:**
- **Experiments 1, 2, 3.1** — LSTM with autoregressive rollout.
  - Exp 1: random train/val split (leakage). Leaderboard WAPE 37.16 — worse than seasonal baseline.
  - Exp 2: chronological split. Train loss dropped, val loss diverged from epoch 1.
  - Exp 3.1: added domain features. Same divergence shape. Never beat baseline.
  - Diagnosed in the experiment-3 pivot (`explanation/03-the-pivot.md`): the rollout
    architecture was structurally mismatched with the deployment task, because
    `validation_input.csv` provides all future covariates and target lags don't exist
    at inference. The LSTM was solving a problem that didn't exist.

---

## Phase 6.6 — Coverage ledger after six experiments

| Class | Tried? | Best WAPE (CV / LB) | Coverage |
|---|---|---|---|
| Classical | Yes — baselines only | seasonal_mean **33 / 34.4** | shallow |
| Machine learning — LightGBM | Yes — extensively | exp-5.1 **14.73 / 14.93** | deep (saturated) |
| Machine learning — XGBoost | Yes — exp-6 | XGB-MAE solo **14.80** (rejected for submission) | one experiment |
| Machine learning — within-class ensembles | Yes — exp-5 §7, exp-6 §7 & §8 | all rejected by Δ-WAPE rules | exhaustive |
| Deep learning | Only LSTM (exp-1–3.1, wrong framing) | never beat baseline | shallow + wrong |

ML is now well explored across two algorithm families. Classical remains at baselines.
DL has been tried only with autoregressive LSTMs that solved the wrong problem.

### What experiments 5, 5.1, and 6 conclusively established

1. **MAE objective + `sid_code` + temporal context is the LightGBM ceiling on this data.**
   Exp-5's random 8-config grid over `learning_rate`, `num_leaves`, `min_child_samples`,
   `n_estimators` found no improvement > 0.1 WAPE over the exp-4 config. The model is
   well-tuned.

2. **Bias correction makes things worse on this dataset.** Exp-5 §6 tested per-series
   additive corrections fitted via leave-one-fold-out on OOF residuals — result was a
   **+0.93 WAPE regression**. The "MAE finds the conditional median, target is
   right-skewed, so correct upward" theory is sound in principle, but empirically the
   per-series mean of OOF residuals overfits or worsens the median fit. Rejected.

3. **The imputation choice barely matters.** Switching from `ffill().fillna(0)` to
   `ffill().bfill()` within series (exp-5.1) moved leaderboard WAPE by **+0.004**, well
   within fold noise (±0.6). data-analysis-2.ipynb verified the underlying reason:
   gaps are too short (max 4h) and too rare for the choice to drive results. Adopt
   `ffill().bfill()` as the principled default; don't expect WAPE gains from it.

4. **Algorithm-diversity ensembles within tree-GBM families don't work on this data.**
   This is the headline finding from exp-6: OOF residuals of LightGBM-MAE and
   XGBoost-MAE — with identical features, identical CV, identical loss — correlate at
   **r = 0.9955**. By the Krogh–Vedelsby ambiguity decomposition, the upper bound on
   blend gain at that correlation is ~0.045 WAPE; the actual observed gain was +0.048
   (just below the 0.05 accept threshold). The blend is performing at its theoretical
   maximum; that maximum is just small. **Implication:** CatBoost, sklearn
   HistGradientBoosting, and other tree-GBMs are highly unlikely to break the 14.78 CV
   ceiling either — they share the same inductive bias.

5. **Fold 1 (test days 153–166) is universally hard.** Every model tested in exp-6 —
   LightGBM, XGBoost, Poisson-LGB, Tweedie-LGB, Ridge — scores ~0.8 WAPE worse on
   fold 1 than on folds 0 and 2. This is now the most concrete unresolved lead: what
   specifically about days 153–166 is hard, and is the excess error concentrated in a
   handful of series or spread across all 96?

### Where to go next

Two remaining avenues, in order of expected return:

**A. Switch to a fundamentally different model class (deep learning).** Exp-6
falsified the "stay in tree-GBM space and find diversity" hypothesis. The only way to
produce genuinely decorrelated residuals from this point is a different inductive bias:
MLP, attention, state-space, or a pretrained foundation model. This is also exactly
what the course **Deep Learning Architectures and Methods** is about. **See Phase 6.7
for the detailed roadmap.**

**B. Investigate fold-1 difficulty.** Independent of model choice. Add a per-(fold ×
series) WAPE table; if the excess error is concentrated in a few series, isolate them
for either dedicated models or for understanding what makes them hard. Phase 8 flagged
this in general; exp-6 confirmed it has the strongest remaining concrete signal.

---

## Phase 6.7 — Deep learning architecture roadmap

The class diagnosis is now clear: tree-GBMs are saturated on this feature set
(Phase 6.6 finding #4). The next class to explore is deep learning — but with two
non-obvious caveats specific to this dataset.

### 6.7.1 The framing constraint — known future covariates

Most published time-series forecasters assume the model has *only history* and must
autoregress forward. **That framing is wrong for this dataset.** `validation_input.csv`
provides the full domain-feature trajectory for the 336-hour forecast window. The
problem is per-hour regression where covariates at every prediction step are already
known.

This rules out (or requires non-trivial adaptation of) any model that does not accept
**known future covariates** as a first-class input. Empirically confirmed by the public
leaderboard:

| model | LB WAPE | covariate handling |
|---|---:|---|
| `chronos-2-native-train` (fine-tuned with covariates) | 14.93 | ✓ native |
| `chronos-t5-v2` (older Chronos, history-only) | 17.44 | ✗ history only |
| `chronos-t5` (older Chronos, history-only) | 19.76 | ✗ history only |
| `itransformer-1-hybrid` | 24.61 | ✗ partial |
| `itransformer-1` | 25.67 | ✗ adaptation needed |

Native covariate support is worth ~10 WAPE on this dataset. **Pick architectures
accordingly.**

### 6.7.2 Tier 1 — Architectures designed for known future covariates

**Start here.** All three are usable via `neuralforecast` (Nixtla) with a consistent
sklearn-style API.

#### TiDE — Time-series Dense Encoder (Google, 2023)

- **What:** Pure MLP architecture. Encodes (past + covariates) through dense MLPs;
  decodes through dense MLPs that include a **direct pathway from known future
  covariates to the forecast**. No attention, no recurrence.
- **Why first:** The "embarrassingly simple MLP that beats Transformers." Matches or
  beats TFT and PatchTST on long-horizon benchmarks at 5–10× the speed. Inductive bias
  is maximally different from tree-GBMs (smooth, fully-connected) — best chance of
  producing residuals decorrelated from LightGBM's.
- **Expected effort:** Low. ~5 hyperparameters. Trains in 5–15 min on CPU.
- **Strongest single recommendation for experiment-7.**

#### TFT — Temporal Fusion Transformer (Google, 2019)

- **What:** Encoder-decoder with explicit, *separate* input pathways for static
  metadata, past observed inputs, and known future inputs. Variable-selection networks
  learn which features matter per timestep; multi-head attention handles multi-horizon
  dependencies. Interpretable (attention weights and feature importances are inspectable).
- **Why second:** Purpose-built for this exact data shape — static categorical
  `series_id` + past observed + future covariates. The most "natural fit" architecture.
- **Caveat:** Has ~10 important hyperparameters (`hidden_size`,
  `attention_head_size`, `dropout`, `lstm_layers`, …). Tends to over-parameterize on
  small data — 410k rows × 96 series is on the small side for transformers. Careful
  regularization required.
- **Expected effort:** Medium. 1–2 days to a tuned config.

#### NHITS — Neural Hierarchical Interpolation (Nixtla, 2022)

- **What:** Multi-rate hierarchical MLP that processes the input at different temporal
  resolutions, like multi-scale signal processing. Supports both past and future
  exogenous variables.
- **Why third:** Computationally cheap, native covariate support, and the multi-rate
  decomposition can capture both short and long temporal patterns in one model. Often
  the strongest small-model baseline.
- **Expected effort:** Low. ~5 hyperparameters.

### 6.7.3 Tier 2 — Strong general-purpose architectures (need adaptation)

| Architecture | Year | Key idea | Why try it |
|---|---|---|---|
| **PatchTST** | 2023 | Patch the series → tokens → standard transformer attention. Channel-independent. | Strong on long horizons. Doesn't natively accept future covariates — needs explicit adaptation. |
| **iTransformer** | 2023 | Inverts standard transformer: each *variable* is a token, attention runs across variables (channel-dependent). | Could model cross-series structure (96 series → cross-variable attention). The leaderboard `itransformer-1` underperformed because covariates weren't used; concat-into-variable-tokens adaptation could compete. |
| **Mamba / S-Mamba / TimeMachine** | 2024–25 | Selective state-space models. Linear-time complexity over sequence length. | Most architecturally distinct from both trees and transformers. Best theoretical pick for residual decorrelation. Less battle-tested. |
| **NBEATSx** | 2022 | N-BEATS with exogenous variables. Pure MLP with backcast/forecast residual blocks. | Older but cleaner than full N-BEATS for covariate use. TiDE/NHITS generally dominate it now. |

### 6.7.4 Tier 3 — Foundation models (zero-shot or fine-tuned)

#### Chronos-2 (Amazon, October 2025) — recommended as experiment-8

- **What:** 120M-parameter **encoder-only** model, T5-based. Pretrained on billions of
  time points (real + large-scale synthetic). Native support for **univariate,
  multivariate, and covariate-informed forecasting** via in-context learning + group
  attention. Outputs quantile forecasts.
- **Two modes:**
  - **Zero-shot.** Pass historical target + 336h future covariates as in-context
    examples, get predictions. No training. Quick to try.
  - **Fine-tune.** The `chronos-2-native-train` entry on the leaderboard is fine-tuned
    and scored 14.93 — matching LightGBM-MAE exactly. So Chronos-2 + fine-tuning +
    covariates is *empirically* competitive on this dataset.
- **Library:** Hugging Face `amazon/chronos-2` + the `chronos-forecasting` repo.
- **Hardware:** GPU recommended (300+ series/s on a single A10G). Kaggle T4 sufficient.

#### TimesFM (Google, 2024)

- 200M-parameter decoder-only foundation model trained on 100B time points. Strong
  zero-shot.
- Covariate support less mature than Chronos-2 as of v2.

#### Moirai / Moirai-MoE (Salesforce, 2024–2025)

- Masked encoder foundation model. Mixture-of-experts variant released 2025.
- Less covariate-friendly than Chronos-2 by default.

#### Skip for this problem

- **Lag-Llama** — univariate only.
- **Time-LLM** — reprograms a frozen LLaMA-2 for time-series. Cool research, awkward
  for covariates, expensive at inference.

### 6.7.5 Tier 4 — Diagnostic baselines (cheap, informative)

#### Small MLP-MAE on the exp-5 feature set

The single highest-information cheap experiment. Build a 2–3 layer MLP (~200 units,
dropout, BatchNorm) on the existing 52 features, train with `nn.L1Loss()`. Compute
residual correlation against LGBM-MAE OOF.

- If r < 0.85 → MLP gives genuine diversity. Build the LGBM+MLP blend; expect real
  WAPE gain.
- If r > 0.95 → the dataset's signal is saturated on this feature set; no neural
  architecture is going to beat ~14.78 without different features. Time to look at
  feature engineering or accept the floor.

Either result is a clear signal. ~1 hour of work. Run this in parallel with
experiment-7.

#### Encoder-decoder LSTM with future covariates

Plain LSTM where the encoder eats history and the decoder eats future covariates
step-by-step. Less expressive than TFT for similar effort, but a useful third arm in a
cross-class ensemble.

### 6.7.6 Library landscape

| Library | Strengths | Best for |
|---|---|---|
| **NeuralForecast** (Nixtla) | Cleanest unified API; consistent train/predict; TFT, NHITS, TiDE, PatchTST, iTransformer all available | **Start here.** Tier 1 + Tier 2 experiments. |
| **Darts** | Broad model zoo incl. TFT, NHITS, TiDE, N-BEATS, RNN, classical | Quick comparative experiments. |
| **chronos-forecasting + Hugging Face `amazon/chronos-2`** | Direct Chronos-2 access | Tier 3 foundation models. |
| **pytorch-forecasting** | TFT, DeepAR, NHiTS | More configurable than NeuralForecast; more verbose. |
| **GluonTS** (Amazon) | Production-grade DeepAR/Transformer family | Heavier; less essential for this work. |

### 6.7.7 Recommended experiment order

Keeping the same discipline that worked through exp-1 to exp-6 (explicit decision
rules; residual-correlation diagnostic *before* blending):

1. **experiment-7 — TiDE.** Single model. Compute OOF residual correlation vs
   LGBM-MAE.
   - If r < 0.85 → build LGBM + TiDE blend in experiment-7.5.
   - If r > 0.95 → write a short README and move on (same discipline as exp-6).

2. **experiment-8 — Chronos-2.** Zero-shot first (no training cost). Fine-tune if
   zero-shot is competitive (CV-WAPE < 16). Native covariate support means you can
   pass the 11 domain forecasts as future covariates directly without API surgery.

3. **experiment-9 (conditional) — TFT.** Only if TiDE and Chronos-2 both stall. TFT
   is heavier than TiDE; the explicit variable-selection structure might extract
   per-series patterns trees miss.

4. **Diagnostic in parallel with experiment-7 — small MLP-MAE.** Cheapest possible
   test of the "non-tree class produces decorrelation" hypothesis. ~1 hour of work,
   either result useful.

5. **Cross-class ensemble (if any of 1–3 produces low r):** Blend LightGBM + (best DL
   model) with OOF-tuned weights. This is where the exp-5 ensemble would have worked
   if any of the arms had been from a different class.

Realistic targets, given the empirical floor at CV-WAPE ~14.78:

- **experiment-7 (TiDE solo or LGBM+TiDE):** 14.5–14.7
- **experiment-8 (Chronos-2 zero-shot):** 15–18 (probably worse than fine-tuned)
- **experiment-8 (Chronos-2 fine-tuned):** 14.7–14.9 (matches the leaderboard
  precedent set by `chronos-2-native-train`)
- **experiment-9 (TFT solo or blend):** 14.4–14.7 if it works at all
- **Cross-class ensemble:** 14.3–14.6 if any DL arm produces r < 0.85

---

## Phase 7 — Align the objective to the metric

| metric | use this loss |
|---|---|
| WAPE, MAE | `regression_l1` (MAE) |
| MSE, RMSE | `regression_l2` (MSE) |
| MAPE | custom weighted L1 or Tweedie |
| sMAPE | custom |

For this dataset, switching from L2 to MAE was worth −1.08 WAPE on its own — the largest
single improvement. L2 concentrates effort on high-value rows (large errors are squared),
which under-fits low-value rows and inflates WAPE where the denominator is small.

MAE also reduces variance across CV folds (std dropped from 1.22 to 0.47), making your
estimates more reliable.

---

## Phase 8 — Residual analysis before hyperparameter tuning

After your first working model, before touching hyperparameters, break down the
out-of-fold residuals. This is the highest-value diagnostic step.

```python
oof_df['abs_err'] = (oof_df['pred'] - oof_df['target']).abs()

# By hour of day
oof_df.groupby('hour').apply(lambda g: g['abs_err'].sum() / g['target'].sum())

# By series
oof_df.groupby('series_id').apply(lambda g: g['abs_err'].sum() / g['target'].sum())

# By feature quintile (e.g. queue_pressure)
oof_df['qp_q'] = pd.qcut(oof_df['queue_pressure_forecast'], 5, labels=False)
oof_df.groupby('qp_q').apply(lambda g: g['abs_err'].sum() / g['target'].sum())
```

**What to look for:**

- **Large per-series spread** (e.g. best 8.5 WAPE vs worst 28): add `series_id` as
  categorical. If still large, consider per-series secondary models.
- **Feature-quintile pattern** (high WAPE at low feature values): could be model
  under-fitting, or just WAPE arithmetic (small denominator). Check if absolute error is
  also higher there. If it's only WAPE that's inflated, don't chase it.
- **Hour-of-day pattern**: add hour encodings or lag features if missing.
- **Systematic bias** (mean residual ≠ 0): add a per-series bias correction post-hoc
  (fit a single constant per series on OOF residuals).

---

## Phase 9 — Hyperparameter tuning

Tune under cross-validation, not a single split. A single split is too noisy to detect
improvements smaller than ~1 WAPE.

**What to tune first (high leverage):**

1. `objective` — L2 vs MAE vs Poisson vs Tweedie. Run the grid. Takes minutes.
2. `series_id` on/off. Often the biggest structural lever.
3. `n_estimators` × `learning_rate` tradeoff (lower LR + more trees = better generalization).
4. `num_leaves` — controls model complexity.

**What to tune second (lower leverage):**

5. `subsample`, `colsample_bytree` — reduce overfitting.
6. `min_child_samples` — minimum samples per leaf.
7. `reg_lambda` — L2 regularization.

**What not to tune until everything else is right:**

- Feature set. Fix your features before tuning hyperparameters. A bad feature set with
  tuned hyperparameters still has a bad feature set.
- Ensemble weights. Tune the base models first.

**A practical grid:**

```python
# Run on your 3-fold CV. The whole grid takes ~20 minutes.
for objective in ['regression_l2', 'regression_l1', 'poisson', 'mape']:
    for use_series_id in [False, True]:
        cv_scores = []
        for fold in [0, 1, 2]:
            model = lgb.LGBMRegressor(objective=objective, ...)
            model.fit(X_train_fold, y_train_fold,
                      categorical_feature=['sid_code'] if use_series_id else 'auto')
            cv_scores.append(wape(y_val_fold, model.predict(X_val_fold)))
        print(f"{objective:20s} series_id={use_series_id}: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")
```

---

## Phase 10 — Further improvements (advanced)

Once the core model is solid and you've done residual analysis, these are the levers
roughly in order of expected gain vs effort:

### Temporal context features — low effort, ~0.6 WAPE
Already described in Phase 5. Lags and leads of top domain features. Do this before
any of the below.

### Bias correction — very low effort, ~0.1–0.3 WAPE
Fit a per-series additive constant on out-of-fold residuals:
```python
series_bias = oof_df.groupby('series_id')['residual'].mean()
final_preds -= preds['series_id'].map(series_bias)
```
MAE training produces the conditional median; for right-skewed targets this sits below
the mean. A bias correction patch fixes this without retraining.

### Ensemble — medium effort, ~0.2–0.4 WAPE
Blend 2–3 models:
- MAE-LightGBM (your best single model)
- Poisson-LightGBM (different inductive bias)
- A simple linear regression on the top features (adds diversity)

Simple average or a weighted blend fit on OOF predictions. Do this only after each
base model is individually well-tuned.

### Light hyperparameter tuning with Optuna — medium effort, ~0.1–0.3 WAPE
```python
import optuna

def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
    }
    return cross_val_wape(params)  # your 3-fold CV function

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

### Per-series secondary models — high effort, variable, overfit risk
For the worst-performing units (the "unit_073 cluster"), train a per-series adjustment
on top of the global model. Fitting 96 separate tiny models on residuals is the idea.
Risk: overfitting if the series are short. Only pursue if the worst-series WAPE is
significantly worse than median and you have enough data per series.

### Data augmentation — generally not useful for this problem
Standard time series augmentation (jittering, time warping, window slicing) is designed
for classification tasks with limited data. This problem has plenty of data and the issue
is model structure, not data scarcity. Not a worthwhile avenue unless you've exhausted
everything else.

---

## Phase 11 — When to stop

Plot your improvements in order:

| step | CV-WAPE | Δ |
|---|---|---|
| Seasonal baseline | 33 | — |
| Linear regression | 21 | −12 |
| LightGBM, L2 | 17.3 | −3.7 |
| + MAE + series_id | 15.4 | −1.9 |
| + temporal features (exp-4) | 14.80 | −0.6 |
| + multi-seed averaging (exp-5.1, exp-6 winner) | 14.73 | −0.07 |
| + tree-GBM ensemble attempts (exp-5 §7, exp-6 §7 & §8) | rejected | — |

The curve bends fast. When you're looking at −0.3 WAPE for significant extra complexity,
ask whether it's worth it. Some error is irreducible:

- WAPE arithmetic on low-target rows is not fixable by modeling.
- Per-series noise in the hard units may be genuine stochasticity, not signal.

Stop when the next expected improvement is smaller than your CV standard deviation.

### Empirical floor on this dataset (after six experiments)

| quantity | observed | source |
|---|---|---|
| CV-WAPE for tuned LightGBM-MAE pipeline | **14.78 ± 0.62** | exp-5.1 §4, exp-6 §4 |
| CV-WAPE with multi-seed averaging on top | **14.73** | exp-5.1 §8, exp-6 §9 |
| Public leaderboard WAPE for that composition | **14.93** | exp-5, exp-5.1, exp-6 |
| OOF residual correlation between LGBM-MAE and XGBoost-MAE | **r = 0.9955** | exp-6 §7 |
| Per-fold WAPE: fold 0 / fold 1 / fold 2 | **14.08 / 15.58 / 14.68** | exp-6 §4 |
| Fold-1 excess over folds 0 and 2 (every model tested) | **~0.8 WAPE** | exp-6 §4–§8 |

If you reproduce these numbers, you are at the tree-GBM ceiling for this feature set.
Three concrete avenues remain:

1. **Different model class** — Phase 6.7. TiDE / Chronos-2 / TFT are the candidates
   with the strongest theoretical case for producing decorrelated residuals.
2. **Per-series investigation of fold-1 difficulty** — concentrated where the excess
   error lives. May be 5–10 hard series rather than a global effect.
3. **Accept ~14.8 as close to the WAPE-irreducible noise floor** on this dataset
   given the current feature set, and stop. Defensible.

The exp-6 finding (r = 0.9955 between two different boosting libraries) is the
strongest single piece of evidence that the data's signal is largely captured by the
current pipeline. Future wins likely require either a fundamentally different
inductive bias (deep learning) or features the current 52-column set is missing.

---

## Summary checklist

### Tree-GBM phase (the ML class — saturated at CV-WAPE ~14.78 on this dataset)

```
[ ] Read validation_input.csv before building any model
[ ] Compute feature-target correlations on held-out data
[ ] Set up chronological train/val split
[ ] Set up 3-fold rolling-origin CV
[ ] Compute and score the seasonal_mean baseline (WAPE ~33 here)
[ ] Run linear regression on top features — must beat baseline before proceeding
[ ] Apply ffill().bfill() within-series imputation (data-analysis-2 recipe)
[ ] Build LightGBM with: domain features + time + series_mean + series_id (categorical)
[ ] Use MAE objective (not L2) — worth -1.08 WAPE on this dataset
[ ] Run residual breakdown by hour / series / feature quintile
[ ] Add temporal context features (lag/lead/rolling of top domain features) — -0.6 WAPE
[ ] Run objective × series_id grid on CV
[ ] Add multi-seed averaging (5 seeds) — typically -0.05 WAPE
[ ] If ensembling, compute residual correlation; accept only if r < 0.85
[ ] (Skip per exp-5/6 findings on this dataset:) bias correction, tree-GBM diversity blends
[ ] Stop when Δ < CV std OR when CV-WAPE ≈ 14.78 (tree-GBM empirical floor)
```

### Deep learning phase (the next class — see Phase 6.7)

```
[ ] Verify the framing: this dataset has KNOWN FUTURE COVARIATES
    → pick architectures that accept them natively (TiDE, TFT, NHITS, Chronos-2)
    → models that assume history-only (vanilla iTransformer, older Chronos) underperform
[ ] experiment-7: train TiDE via NeuralForecast on the same 52 features
[ ] Compute OOF residual correlation vs LGBM-MAE; gate the blend on r < 0.85
[ ] In parallel: train a small MLP-MAE (~1h work) — cheapest diagnostic of "any
    non-tree class produces decorrelation"
[ ] experiment-8: Chronos-2 zero-shot, then fine-tune if competitive
[ ] experiment-9 (conditional): TFT if TiDE and Chronos-2 both stall
[ ] If any DL arm produces r < 0.85 with LGBM, build the cross-class ensemble
[ ] Independently investigate fold-1 (test days 153-166): per-(fold × series) WAPE table
[ ] Stop when Δ < CV std OR when satisfied 14.7-ish is close to the dataset's floor
```
