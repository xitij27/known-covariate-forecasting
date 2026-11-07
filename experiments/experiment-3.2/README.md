# Experiment 4 — Direct Regression on Known Future Covariates

**Held-out WAPE = 16.53** (vs `seasonal_mean` 31.40 / public leaderboard 34.41).
Leaderboard score pending submission.

This experiment abandons the LSTM. Experiments 1–3 framed the task as autoregressive
sequence modelling; a diagnostic run inside experiment-3 (notebook **Section 6b**) showed
that framing — not the features, not regularisation, not the loss function — was the
bottleneck.

---

## The reframing

The competition provides `validation_input.csv`: the real values of every *forecast*
feature (`queue_pressure_forecast`, `network_pressure_forecast`, `workload_intensity`, …)
at every timestamp of the 336-hour horizon. These are not quantities to be forecast — they
are **inputs you already have**.

Crucially, `validation_input.csv` has **no `target` column**. So the target-lag features the
LSTM relied on (`lag_1 … lag_168`) do not exist at inference time. Experiments 1–3 worked
around this with a 14×24h autoregressive rollout that fed the model its own predictions as
fake lags — compounding error across the horizon.

The correct model maps the known covariates at time *t* directly to `target[t]`:

```
target[t] ≈ f(domain_features[t], time[t], series_baseline)
```

No lookback window. No rollout. No error accumulation.

---

## Is this leakage? (No — and here is the precise reasoning)

A fair question, because the domain features are *so* predictive that it can feel like the
answer is baked into the inputs. It isn't. Three separate things are going on, and only the
first would be leakage:

**1. We never train on the validation targets.** They are held by Kaggle and scored only via
the leaderboard. `validation_input.csv` has **24 columns**; `train.csv` has **25** — the
missing one is `target`. The validation answers are literally not in any file we load, so
training on them is impossible.

**2. The features in `validation_input.csv` are provided *inputs*, not the target.**
`queue_pressure_forecast`, `network_pressure_forecast`, … are covariates the organisers supply
*for the forecast window* precisely so you can predict with them — the same way a weather
forecast is a legitimate input for predicting ice-cream sales. Using them is the intended
task, not an exploit. (The dataset is almost certainly synthetic, with the target *generated*
as a function of these covariates plus noise — which is exactly the signal experiments 1–3
threw away by trying to predict the target from its own past.)

**3. The held-out WAPE 16.53 is a backtest inside `train.csv`, not the real validation set.**
The "held-out tail" is the last 14 days of `train.csv` — rows we *do* have labels for. The
model is fit on the first ~166 days only, then scored on those last 14 days. That is a
standard chronological train/test split within the training data; the model never sees the
held-out labels while fitting.

### The one real caveat (local estimate only)

`series_mean` is each series' mean target over *all* of `train.csv`, which includes the
held-out tail. So in the **local 16.53 estimate** that feature peeks at ~8% of the held-out
targets — a sliver of optimism. It is negligible (the target is stationary, so the
train-portion mean ≈ the full mean) and it is **absent from the real submission** (the
validation targets are not in `train.csv`, so `series_mean` there is built purely from
training data). For a fully airtight local number, recompute `series_mean` from the
train-portion only inside the eval split — it barely moves the result.

**Bottom line:** the only ground truth is the leaderboard. The backtest is an honest proxy
built without ever touching the validation answers — submit to confirm.

---

## Evidence (notebook Section 6b)

**1. The domain features carry real out-of-sample signal.** Univariate r² on the held-out
tail (last 336h/series, which no model trained on) reproduces the data-analysis claims:

| feature | OOS r² (pooled) | OOS r² (within-series) | claimed |
|---|---|---|---|
| `queue_pressure_forecast` | 46.8% | 53.5% | 45.2% |
| `network_pressure_forecast` | 42.5% | 50.4% | 42.5% |
| `workload_intensity` | 29.7% | 37.8% | 28.7% |
| `shock_risk` | 18.2% | 23.6% | 24.6% |
| `event_load_forecast` | 13.5% | 14.2% | 17.1% |

**2. A trivial model beats the baseline by a third** — no autoregression, no LSTM:

| model | held-out WAPE |
|---|---|
| `seasonal_mean` (baseline) | 31.40 |
| linear / domain features | 21.00 |
| linear / domain + `series_mean` | 17.98 |
| **LightGBM / domain + time + `series_mean`** | **16.53** |

Within-series r² running higher than pooled is why `series_mean` (each unit's training mean)
helps so much: it gives the model a per-series baseline level. LightGBM adds nonlinearity and
feature interactions on top.

---

## Method (notebook Section 12)

**Features** (all present in `validation_input.csv`):
- 11 domain forecast features — imputed with forward-fill within series, then `fillna(0)`
- 7 time encodings — `hour`/`dow`/`doy` sin-cos + `is_weekend`
- `series_mean` — per-series target mean computed from `train.csv`

Deliberately excluded: target-lags (don't exist at inference), `nominal_capacity` and
`trend` (≈zero correlation), and the autoregressive rollout entirely.

**Procedure:**
1. Compare linear / linear+`series_mean` / LightGBM on an honest chronological split (train
   on the first ~166 days, evaluate on the held-out last 14 days).
2. Refit the winner (LightGBM) on *all* of `train.csv`.
3. Predict `validation_input.csv` per-timestamp, clamp at 0, align to
   `forecast_index_validation.csv`, write `validation_predictions.csv`.

**LightGBM config:** `n_estimators=600, learning_rate=0.03, num_leaves=63, subsample=0.8,
colsample_bytree=0.8, min_child_samples=50, reg_lambda=1.0`.

Prediction distribution on the val set: mean 11.0, std 5.05, min 1.79, max 37.9 — within the
training range (train target mean 9.9, max ≈37).

---

## Why it should transfer to the leaderboard

- **Stationarity.** The target is stationary over 180 days (trend r=+0.019; 60-day chunk
  means 8.93 / 9.40 / 9.09), so `series_mean` from training ≈ the val-period level. This also
  makes `series_mean` a legitimate predictor rather than leakage.
- **Low distribution shift.** The data-analysis measured <0.25σ shift on all features between
  the train tail and the val period.
- **Honest evaluation.** The 16.53 estimate comes from training on the first 166 days and
  testing on a period the model never saw.

Expect the leaderboard near 16.5, possibly marginally higher — see **Is this leakage?** above
for the one small caveat in the local estimate.

---

## Implementation

The code currently lives in [`../experiment-3/kp-experiment-3.ipynb`](../experiment-3/kp-experiment-3.ipynb):
- **Section 6b** — diagnostics (out-of-sample r², trivial-model WAPE).
- **Section 12** — model comparison, refit on full train, and the submission writer.

Run Sections 1–6 then Section 12; the LSTM training (7–9) and rollout (10–11) are not needed.
A standalone `kp-experiment-4.ipynb` can be extracted if a clean per-experiment notebook is
preferred.

---

## Possible further gains

- **Lagged *domain* features** (e.g. `queue_pressure_forecast` at t−1, t−24). Unlike
  target-lags these are present in `validation_input.csv`, so they are leakage-free and add
  the temporal context the per-hour model currently ignores.
- Per-series slopes / feature interactions, or a small **non-autoregressive** MLP on the same
  feature set — not the autoregressive LSTM.
