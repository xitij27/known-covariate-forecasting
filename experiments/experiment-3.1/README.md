# Experiment 3 — Domain Features, Fixed Inference, and L1 Loss

Kaggle notebook: https://www.kaggle.com/code/xitij27/kp-experiment-3

> **Outcome:** The LSTM changes documented below did *not* fix the val divergence (full
> explanation in the **Outcome** section at the end of this file). A diagnostic added to the
> notebook (**Section 6b**) showed the domain features are genuinely strong, but that this is
> a **direct-regression problem on known future covariates**, not a sequence problem — a
> plain linear model scores WAPE 21 and LightGBM **16.53**, while the LSTM's val loss
> diverges. Winning approach → notebook **Section 12** and
> [`../experiment-4/README.md`](../experiment-4/README.md).

---

## Context: What We Knew Before Writing This Notebook

After two failed experiments, a dedicated data analysis was run on `train.csv`
(`data-analysis.ipynb`) to understand what actually drives the `target` before
making further model changes. The key findings that directly shaped experiment-3:

| Finding | Value |
|---|---|
| `queue_pressure_forecast` correlation with target | R² = **45.2%** |
| `network_pressure_forecast` correlation with target | R² = **42.5%** |
| `workload_intensity` correlation with target | R² = **28.7%** |
| `shock_risk` correlation with target | R² = **24.6%** |
| `nominal_capacity` correlation with target | r = −0.004 (zero) |
| `trend` correlation with target | r = +0.052 (negligible) |
| Missing value run length (domain features) | avg 1.1 h, max 3 h |
| Distribution shift (train tail → val period) | < 0.25σ on all features |
| Target stationarity over 180 days | r = +0.019 (stationary) |

The previous experiments used only 21 features: 7 time encodings + 8 lags + 6 rolling stats.
The **four strongest predictors were all absent from the feature set**.
No amount of regularisation or architecture tuning could fix that.

---

## Experiment History and Why Each One Failed

### Experiment 1 — Baseline LSTM (WAPE = 37.16 on leaderboard)

Appeared to work locally (WAPE = 27.25 on the held-out train tail, +13.2% over `seasonal_mean`).
Failed on the leaderboard (WAPE = 37.16, worse than `seasonal_mean` at 34.41).

**Root cause:** `torch.utils.data.random_split` on a pooled multi-series sliding-window
dataset creates temporal leakage. Windows from the future of a series land in `train_dataset`
while windows from its past land in `val_dataset`. The model trains on observations it should
never have seen, which inflates local validation metrics and hides generalisation failure.

### Experiment 2 — Chronological Split Applied (not submitted)

Fixed the temporal leakage. Val loss immediately diverged upward from epoch 1:

```
Epoch   1/100  train=0.21650  val=0.25432  lr=1.00e-03  ← best
Epoch   2/100  train=0.16957  val=0.29051
Epoch   6/100  train=0.06515  val=0.34151
```

Attempts to fix this with aggressive regularisation (dropout=0.3, LR=3e-4, hidden=128,
batch_size=64) made no difference:

```
Epoch   1/100  train=0.53290  val=0.56816  lr=3.00e-04  ← best
Epoch   6/100  train=0.41628  val=0.64497
```

**Root cause (confirmed by data analysis):** With the chronological split now honest,
the val loss correctly reflected that the model had no real signal to learn from.
It was trying to predict a target driven 45% by `queue_pressure_forecast` and 42% by
`network_pressure_forecast` using only lag features — which degrade rapidly over a
336-hour autoregressive rollout. No regularisation strategy fixes a missing-feature problem.

---

## Changes in Experiment 3

### Change 1 — Add 11 Domain Features to `FEATURE_COLS`

**Where:** Section 1 (Config) and Section 6 (Feature Engineering).

**What was changed:**

```python
# experiments 1-2 — 21 features
FEATURE_COLS = (
    ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_weekend"]
    + [f"lag_{l}" for l in LAGS]
    + [f"roll_mean_{w}" for w in WINDOWS]
    + [f"roll_std_{w}" for w in WINDOWS]
)

# experiment-3 — 32 features
DOMAIN_FEAT_COLS = [
    'workload_intensity',                  # R²=28.7%
    'demand_forecast',                     # R²=7.3%
    'upstream_quality_forecast',           # R²=5.5%
    'promotion_intensity',                 # R²=4.2%
    'shock_risk',                          # R²=24.6%
    'maintenance_known',                   # R²=6.1%
    'unit_reliability_forecast',           # R²=3.5%
    'queue_pressure_forecast',             # R²=45.2%
    'network_pressure_forecast',           # R²=42.5%
    'event_load_forecast',                 # R²=17.1%
    'service_irregularity_risk_forecast',  # R²=3.2%
]
FEATURE_COLS = TIME_COLS + DOMAIN_FEAT_COLS + LAG_COLS + ROLL_COLS
```

**Why:** The data analysis showed that `queue_pressure_forecast` and
`network_pressure_forecast` alone explain ~87% of target variance between them.
These two features are present in both `train.csv` and `val_input.csv` — they are
pre-computed for the entire forecast window and carry no leakage risk.

The per-series correlation analysis confirmed these signals are **universal across
all 96 series** (tight distribution, min r=+0.42, max r=+0.80 for queue_pressure).
The model does not need to learn series-specific relationships — it learns one
consistent mapping shared across all units.

Four features were **excluded** from `DOMAIN_FEAT_COLS`:

| Feature | Reason for exclusion |
|---|---|
| `nominal_capacity` | r = −0.004 vs. per-series target mean — zero predictive signal |
| `trend` | r = +0.052 — target is stationary (chunk means 8.93, 9.40, 9.09 over 180 days) |
| `staffing_forecast` | R² = 0.2% — noise |
| `throughput_disruption_risk_forecast` | R² = 0.1% — noise |

Including zero-signal features adds input dimensionality without benefit and can
harm gradient flow in the LSTM.

---

### Change 2 — NaN Imputation Before Lag Computation

**Where:** Section 6 (Feature Engineering), new `impute_domain_features()` function.

**What was changed:**

```python
# experiments 1-2 — no imputation, dropna handled everything
feat_df = add_time_features(train_df)
feat_df = add_lag_features(feat_df, lags=LAGS)
feat_df = add_rolling_features(feat_df, windows=WINDOWS)
feat_df = feat_df.dropna(subset=FEATURE_COLS)

# experiment-3 — impute domain features BEFORE computing lags
def impute_domain_features(df, domain_cols):
    df = df.copy().sort_values(["series_id", "timestamp"])
    for col in domain_cols:
        if col in df.columns:
            df[col] = (
                df.groupby("series_id")[col]
                .transform(lambda x: x.ffill().fillna(0))
            )
    return df

feat_df = add_time_features(train_df)
feat_df = impute_domain_features(feat_df, DOMAIN_FEAT_COLS)  # ← must come first
feat_df = add_lag_features(feat_df, lags=LAGS)
feat_df = add_rolling_features(feat_df, windows=WINDOWS)
feat_df = feat_df.dropna(subset=FEATURE_COLS)
```

**Why:** The data analysis revealed that 10 domain features each have ~4-5% missing
values. The NaN co-occurrence analysis showed these are **independently missing** across
columns (max off-diagonal correlation ≈ 0.012). With 11 domain columns each 4.5% missing
and NaNs independent, the probability of a row having at least one NaN across all 11 is:

```
P(at least one NaN) = 1 − (1 − 0.045)^11 ≈ 40%
```

Without imputation, `dropna(subset=FEATURE_COLS)` would silently discard ~40% of
training rows — eliminating almost half the data from a dataset that already has
only 4,320 hours per series.

The NaN run-length analysis showed gaps are **short and random** (average 1.1 hours,
maximum 3 hours, across 186 separate gaps per series). Forward-fill within each
series is the correct strategy: it propagates the most recent valid observation
across a gap that is almost always just 1 hour. Any leading NaNs at the start of
a series (before the first valid observation) are filled with 0.

**The imputation must happen before `add_lag_features`** because lag features are
computed from the `target` column — if domain columns had NaNs, the `dropna` in
lag computation would propagate and create additional row losses.

---

### Change 3 — Drop `nominal_capacity` and `trend`

**Where:** Section 1 (Config) — simply not included in `DOMAIN_FEAT_COLS`.

**Why:**

`nominal_capacity` is a fixed per-series constant (one value per series, never changes
within a series). The data analysis computed its Pearson correlation with per-series
target mean: r = −0.004, p = 0.971. It has zero predictive relationship with how
busy a unit is. Including it adds a constant column to every lookback window that
the LSTM must learn to ignore.

`trend` is a slowly changing linear ramp. The stationarity check split the training
data into three equal 60-day chunks and computed target statistics:
- Chunk 1 (days 0–60): mean = 8.928, std = 5.024
- Chunk 2 (days 60–120): mean = 9.402, std = 5.077
- Chunk 3 (days 120–180): mean = 9.094, std = 5.139

The linear trend correlation was r = +0.019 — essentially zero. The target has no
meaningful drift over the 180-day training period. The `trend` column is redundant
with the per-series mean normalisation already applied to `target_norm`.

---

### Change 4 — Fix Rolling Inference to Use `val_input_df` Domain Features

**Where:** Section 10 (Validation Inference), `rolling_forecast()` function.

**What was changed:**

```python
# experiments 1-2 — domain features for predicted rows set to 0 or not present at all
for lag in LAGS:
    new_rows[f"lag_{lag}"] = [...]
for w in WINDOWS:
    new_rows[f"roll_mean_{w}"] = ...

# experiment-3 — look up real domain features from val_input_df
@torch.no_grad()
def rolling_forecast(model, context_df, index_df, norm_stats, feature_cols,
                     future_domain_df=None,   # ← new parameter
                     lookback=LOOKBACK, horizon=HORIZON):

    # Pre-index by (series_id, timestamp) for O(1) lookup
    future_lookup = {}
    if future_domain_df is not None:
        for sid, grp in future_domain_df.groupby("series_id", sort=False):
            future_lookup[sid] = grp.set_index("timestamp")

    ...
    # When building context rows for a predicted 24-step block:
    if sid in future_lookup:
        fut = future_lookup[sid]
        for col in DOMAIN_FEAT_COLS:
            if col in fut.columns:
                new_rows[col] = [
                    float(fut.loc[ts, col]) if ts in fut.index else 0.0
                    for ts in step_times
                ]
```

**Why:** The model must make 14 consecutive 24-step rollouts to cover the 336-hour
forecast horizon. After the first rollout, the context window begins to overlap with
the forecast period. Rows in that overlap need all 32 feature values — including the
11 domain features.

In experiments 1-2 (which did not use domain features) this was not an issue. In
experiment-3, if domain features for context rows in the val period are set to 0
or approximated from predicted targets, the model sees garbage values for its two
strongest predictors (`queue_pressure_forecast` R²=45%, `network_pressure_forecast`
R²=42%) during the later rollout blocks.

`val_input_df` provides **pre-computed real values** for all domain features at every
forecast timestamp. The distribution shift analysis confirmed these values are
statistically indistinguishable from the training distribution (< 0.25σ shift on all
features). Using them eliminates error accumulation in the strongest predictors over
the full 14-step rollout.

The function is called differently for the two inference use-cases:

```python
# Competition predictions — use real domain features from val_input_df
predictions_df = rolling_forecast(
    model, train_df, val_index_df, norm_stats, FEATURE_COLS,
    future_domain_df=val_input_df,   # ← oracle features provided
)

# Local held-out evaluation — no oracle, domain features default to 0
eval_preds_df = rolling_forecast(
    model, train_eval_df, held_out_index, norm_stats, FEATURE_COLS,
    future_domain_df=None,           # ← not available for local eval
)
```

The local eval is therefore a **conservative lower bound** — the competition score
should be meaningfully better because the model receives real domain features.

---

### Change 5 — Chronological Train/Val Split (Carried Over from Experiment 2)

**Where:** Section 7 (Dataset & DataLoader).

No code change — this fix from experiment-2 is kept exactly as-is. For completeness:

```python
train_rows, val_rows = [], []
for sid, grp in feat_df.groupby("series_id", sort=False):
    grp = grp.sort_values("timestamp")
    train_rows.append(grp.iloc[:-EVAL_HORIZON])  # all but last 336 hours
    val_rows.append(grp.iloc[-EVAL_HORIZON:])     # last 336 hours

train_feat_df = pd.concat(train_rows, ignore_index=True)
val_feat_df   = pd.concat(val_rows,   ignore_index=True)

train_dataset = SlidingWindowDataset(train_feat_df, FEATURE_COLS)
val_dataset   = SlidingWindowDataset(val_feat_df,   FEATURE_COLS)
```

**Why:** Prevents temporal leakage (the root cause of experiment-1's inflated local score).
Every training window is strictly earlier in time than every validation window.

---

### Change 6 — Switch Loss from `HuberLoss` to `L1Loss`

**Where:** Section 9 (Training Loop).

**What was changed:**

```python
# experiments 1-2
criterion = nn.HuberLoss(delta=1.0)

# experiment-3
criterion = nn.L1Loss()
```

**Why:** The competition metric is WAPE:

```
WAPE = Σ|y_true - y_pred| / Σ|y_true| × 100
```

WAPE is a **sum of absolute errors** normalised by the sum of actuals. It is
structurally equivalent to a weighted L1 loss. `HuberLoss` behaves as L2 (squared
error) for small residuals and L1 for large ones — it is a compromise that does not
fully align with either metric. Training directly with `nn.L1Loss()` ensures every
gradient step pushes the model in the direction that reduces the evaluation metric.

This is a secondary change — the domain features are the primary fix. But with the
model now having access to strong signals, aligning the loss with the metric ensures
the final checkpoint is selected and optimised for the right objective.

---

## Complete Diff: What Changed Between Experiments

| Component | Experiment 1 | Experiment 2 | Experiment 3 |
|---|---|---|---|
| Train/val split | `random_split` (leakage) | Chronological | Chronological |
| Feature count | 21 | 21 | **32** |
| Domain features used | None | None | **11 added** |
| `nominal_capacity`, `trend` | Included | Included | **Dropped** |
| NaN imputation | None | None | **ffill + fillna(0)** |
| Imputation order | — | — | **Before lag computation** |
| Loss function | HuberLoss | HuberLoss | **L1Loss** |
| Inference domain features | N/A (not in model) | N/A | **Looked up from `val_input_df`** |
| Model architecture | LSTM h=256, 2L | LSTM h=256, 2L | LSTM h=256, 2L |
| LR / weight decay | 1e-3 / 1e-4 | 1e-3 / 1e-4 | 1e-3 / 1e-4 |
| Leaderboard WAPE | 37.158 (worse than baseline) | Not submitted | Pending |

---

## What to Expect

The val loss should now decrease alongside train loss for at least the first 30–60
epochs. Unlike experiment-2, the model has access to the features that explain the
majority of target variance — so the chronological val set is no longer starving the
model of signal.

The local held-out eval (without oracle domain features) will underestimate competition
performance. The leaderboard score benefits from real `queue_pressure_forecast` and
`network_pressure_forecast` values at every forecast step via `val_input_df`.

The target to beat is `seasonal_mean` at WAPE = **34.408848** on the public leaderboard.

> The prediction above was wrong. The Outcome section below records what actually happened.

---

## Outcome — the LSTM changes did NOT fix the divergence (and why)

With the 11 domain features added, the chronological val loss diverged from epoch 1 in
exactly the same shape as experiment-2:

```
Epoch   1/100  train=0.48494  val=0.54823  ← best
Epoch   4/100  train=0.32686  val=0.64686
Epoch   8/100  train=0.24578  val=0.64017
```

That identical pattern was the clue. If `queue_pressure_forecast` (R²=45%) really carried
that signal, adding it should have bent the val curve down. It didn't move at all — so either
the features were worthless out-of-sample, or the val loss was the wrong thing to watch. A
diagnostic was added (notebook **Section 6b**) to settle which.

**Finding 1 — the features are real.** Out-of-sample univariate r² on the held-out tail
reproduces the data-analysis claims almost exactly: `queue_pressure_forecast` 46.8% (claimed
45.2), `network_pressure_forecast` 42.5% (42.5), `workload_intensity` 29.7% (28.7). Not an
artifact.

**Finding 2 — the architecture was the problem, not the features.** A *trivial* linear
regression on the 11 domain features — no LSTM, no lookback, no rollout — scores **WAPE 21.0**
vs `seasonal_mean` 31.40. The LSTM cannot match a linear model here because:

- The training/val loss is **teacher-forced** (real lags every step), so it measures the very
  24h prediction the lags already solve — it overfits from epoch 1 regardless of features.
  Worse, the checkpoint is selected on this loss, so `best_checkpoint.pt` is the **epoch-1**
  model.
- The competition is a 336h autoregressive rollout, and `validation_input.csv` has **no
  `target` column** — so the target-lags the LSTM leans on don't exist at inference. The
  rollout manufactured pseudo-lags from its own predictions, compounding error.
- Inputs were never normalised: raw lags/rolling means (~9–11) numerically swamped the domain
  features (~0.01–2), so the model largely ignored its strongest signals.

**Conclusion: this is a direct-regression problem on known future covariates, not an
autoregressive sequence problem.** The winning approach — LightGBM on domain + time +
per-series mean, **held-out WAPE 16.53** — is implemented in notebook **Section 12** and
written up in [`../experiment-4/README.md`](../experiment-4/README.md).

