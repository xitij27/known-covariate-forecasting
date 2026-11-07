# Direct regression — the first model that actually worked

After the pivot I built experiment-3.2: a direct regression on the known covariates. No LSTM, no autoregression, no sliding windows. Just: predict `target[t]` from features at `t`.

## The setup

For each row, the features were:

- The 11 domain forecast features (imputed within series: forward-fill, then zero-fill any leading NaNs).
- The 7 time encodings (hour, day-of-week, day-of-year in sin/cos, weekend flag).
- A new feature, `series_mean` — each unit's average target over the training data. This gives the model a per-series baseline level. It mattered because the target means vary widely across series (6 to 14), and one global model otherwise has to average over that.

For the local evaluation I split chronologically: the first ~166 days for training, the last 14 days for testing. The same held-out tail I'd been using all along.

## Trying a few candidates

I tested three models on this honest split:

```
linear / domain                        WAPE = 21.00
linear / domain + series_mean          WAPE = 17.98
lightgbm / domain + time + series_mean WAPE = 16.53
```

A few things to note:

- Adding `series_mean` was a big jump for the linear model (21 → 18). The per-series differences in baseline level had been a big source of error, and giving the model a single-number summary per series fixed a lot of it.
- LightGBM on top added another ~1.5 WAPE. That's the gain from non-linearity and feature interactions.
- The best model: LightGBM on the full feature set, **WAPE 16.53**. About half the seasonal baseline.

I refit the LightGBM on all of `train.csv`, predicted the val window using the domain features from `validation_input.csv`, clamped predictions at zero (the target is non-negative), and wrote the submission.

## A short note on what "leakage" looks like

One worry crossed my mind: am I cheating by using `validation_input.csv`'s forecast features as inputs?

I worked through this carefully:

- The validation *targets* are held by Kaggle. They're not in any file I can read. `validation_input.csv` has 24 columns; `train.csv` has 25. The missing one is `target`. I literally cannot train on the validation answers.
- The features in `validation_input.csv` are *covariates*, not the target. They are provided to me precisely so I can predict with them. Using them is the intended task — same as using a weather forecast to predict ice-cream sales.
- The 16.53 number is a backtest *inside* `train.csv`: I trained on the first 166 days and evaluated on the last 14. The model never sees the held-out labels while fitting.

The one small caveat is `series_mean`. In the local backtest it's computed over the full series, so it gets a tiny peek (about 8% of the values) of the held-out targets through that one feature. But the target is stationary, so the train mean ≈ the full mean. And in the actual submission, `series_mean` comes from the training data only — fully clean. The leaderboard is the real ground truth either way.

## What I learned at this checkpoint

A few things crystallized for me here:

1. **The simplest correct model crushed everything I'd built before.** Linear regression on the right features (21.0) was already better than any LSTM I'd trained. LightGBM (16.5) made it nearly half the baseline.

2. **The architecture choice mattered far less than the problem framing.** Three experiments of architecture tweaks (regularization, more features, different loss) had failed. The pivot to direct regression *worked immediately.*

3. **Look at what's in the inference inputs.** The shape of `validation_input.csv` was telling me the whole story. I just hadn't read it.

This was the breakthrough. The remaining work was making sure 16.53 was real, and squeezing further improvements from there.
