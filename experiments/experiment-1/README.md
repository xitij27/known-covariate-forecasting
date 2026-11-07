# Experiment 1 — Dataset Analysis & DataLoader Shuffling Notes  
https://www.kaggle.com/code/xitij27/kp-experiment-1/edit#Time-Series-Forecasting-%E2%80%94-Kaggle-Notebook  


## Dataset Overview

| Property | Value |
|---|---|
| Task | Multivariate hourly time series forecasting |
| Series | 96 (`unit_000` – `unit_095`) |
| Timesteps per series | 4,320 (hourly, 2023-01-01 → 2023-06-29) |
| Total training rows | 414,720 |
| Forecast horizon (rollout block) | 24 hours |
| Required prediction horizon (val/test) | 336 hours per series |
| Primary metric | WAPE (lower is better) |
| Target | Hourly operational load index per series |

### Features (25 columns in train.csv)

`series_id`, `timestamp`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `is_weekend`, `trend`, `workload_intensity`, `demand_forecast`, `staffing_forecast`, `upstream_quality_forecast`, `promotion_intensity`, `shock_risk`, `maintenance_known`, `unit_reliability_forecast`, `queue_pressure_forecast`, `network_pressure_forecast`, `event_load_forecast`, `service_irregularity_risk_forecast`, `throughput_disruption_risk_forecast`, `nominal_capacity`, `zone_sin`, `zone_cos`, `target`

---

## Baseline Results (evaluated on last 336 hours of training data per series)

| Baseline | MAE | MSE | RMSE | MAPE | sMAPE | WAPE |
|---|---|---|---|---|---|---|
| naive_last_value | 5.8214 | 57.0340 | 7.5521 | 55.65 | 65.73 | 54.50 |
| lag24_repeat | 4.4646 | 34.6364 | 5.8853 | 48.60 | 49.50 | 41.80 |
| lag168_repeat | 4.8584 | 40.4624 | 6.3610 | 54.07 | 54.43 | 45.49 |
| seasonal_mean | **3.3541** | **20.6407** | **4.5432** | **42.85** | **34.70** | **31.40** |

`seasonal_mean` (per-series day-of-week × hour-of-day average) is the strongest baseline and the reference to beat.

---

## DataLoader Shuffling — Findings & Recommendations

### Train loader — `shuffle=True` is OK

The `SlidingWindowDataset` (in `experiment.ipynb`) pre-extracts all sliding windows
and stores them as self-contained `(X, y)` pairs before any DataLoader is created:

```python
for start in range(0, n - lookback - horizon + 1):
    X_win = X_arr[start : start + lookback]
    y_win = y_arr[start + lookback : start + lookback + horizon]
    self.samples.append((X_win, y_win))
```

Each sample already encodes its full temporal context (lookback window + target horizon).
Shuffling only changes **the order batches are fed to the optimizer** — it does not
reorder timesteps inside any window. This is standard practice for sliding-window
time series training and beneficial for SGD convergence.

**Verdict: `shuffle=True` on the train loader is correct and safe.**

### Val / test loader — `shuffle=False` is correct, keep it

The template already uses `shuffle=False` for `val_loader`. Since evaluation metrics
are computed over the full set of windows regardless of order, shuffling would not
corrupt results — but keeping it `False` ensures reproducibility and makes loss curves
and per-batch debugging easier to interpret.

**Verdict: do not shuffle the val or test loader.**

---

## The Real Risk — Train/Val Split Causes Temporal Leakage

The current notebook splits the dataset using `torch.utils.data.random_split`:

```python
train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED)
)
```

Because all 96 series' windows are pooled into one flat list before splitting,
a window from the **future** of `unit_000` can land in `train_dataset` while a
window from the **past** of `unit_000` lands in `val_dataset`. The model trains
on future observations and is validated on past ones — this inflates in-notebook
validation metrics and gives a false picture of generalisation.

### Recommended fix — chronological split

Hold out the last N timesteps of every series for validation and train only on
the earlier portion, matching how the baseline evaluation code already works:

```python
EVAL_HORIZON = 336  # same as competition horizon

train_rows, val_rows = [], []
for sid, grp in feat_df.groupby("series_id", sort=False):
    grp = grp.sort_values("timestamp")
    train_rows.append(grp.iloc[:-EVAL_HORIZON])
    val_rows.append(grp.iloc[-EVAL_HORIZON:])

train_feat_df = pd.concat(train_rows, ignore_index=True)
val_feat_df   = pd.concat(val_rows,   ignore_index=True)

train_dataset = SlidingWindowDataset(train_feat_df, feature_cols, ...)
val_dataset   = SlidingWindowDataset(val_feat_df,   feature_cols, ...)
```

With this split, `train_loader` can still use `shuffle=True` safely — each window
within `train_feat_df` is strictly earlier in time than any window in `val_feat_df`.

---

## Summary

| Question | Answer |
|---|---|
| Shuffle train loader? | **Yes** — windows are pre-computed; shuffling order is safe and recommended |
| Shuffle val/test loader? | **No** — keep `shuffle=False` for reproducibility |
| Bigger concern than shuffling? | **Yes** — `random_split` creates temporal leakage; switch to a chronological split |
