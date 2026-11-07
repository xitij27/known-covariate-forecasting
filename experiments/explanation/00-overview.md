# Overview — How I got from a broken LSTM to a working model

These notes are my own account of how I worked through this forecasting project from start to finish. The short version: I spent a lot of effort on the wrong approach before finally understanding what kind of problem I was actually looking at.

## The problem

I was given hourly measurements from 96 different "units" (machines, locations — the data doesn't say) over 180 days. I had to predict each unit's target value for the next 14 days. The score is WAPE, which is basically "total absolute error divided by total actuals" — a percentage-style metric. Lower is better.

The twist: alongside the target column, the data also includes a bunch of *forecast* columns — things like `queue_pressure_forecast`, `network_pressure_forecast`, `workload_intensity`. And critically, those same columns are provided in the validation input file too. So I had pre-computed forecasts of various conditions for every hour I needed to predict.

## The journey, in short

1. **Started with an LSTM** because "this is a time series, use a recurrent network." Worked locally, fell apart on the leaderboard. Turned out my train/val split was leaking.
2. **Fixed the split.** Now the training loss dropped while validation loss rose. Tried regularization. No effect.
3. **Added the forecast features.** Same shape as before — val loss diverged from epoch 1. That's when I started suspecting the LSTM itself was the problem.
4. **Ran a sanity check.** A plain linear regression on the same forecast features hit WAPE 21 — beating both my LSTM and the seasonal baseline (31). No neural network, no rollout, just a regression.
5. **Realized the framing was wrong.** This isn't a sequence forecasting problem. It's a per-hour regression where the future covariates are already provided. I dropped the LSTM.
6. **Built a direct regression model.** LightGBM on the forecast features + time + per-series mean. WAPE 16.53.
7. **Made the evaluation trustworthy.** Cross-validated properly. Honest score: 17.27.
8. **Looked at residuals to find leverage.** Per-series spread huge, low-target rows had inflated WAPE.
9. **Switched to MAE objective + `series_id` as a categorical feature.** Dropped to 15.39.
10. **Added temporal context from the covariates.** Dropped to 14.80.
11. **Stopped.** Diminishing returns. The remaining error was mostly WAPE arithmetic and per-series noise.

The headline numbers:

- Seasonal baseline: ~33 WAPE
- LSTM attempts: never beat the baseline
- Final model: **14.80 WAPE** (cross-validated)

## What this folder contains

- `01-understanding-the-problem.md` — what I learned about the data before modeling
- `02-the-lstm-attempts.md` — experiments 1, 2, 3 and why they all failed
- `03-the-pivot.md` — the diagnostic that broke me out of the LSTM mindset
- `04-direct-regression.md` — the first model that actually worked
- `05-trustworthy-and-improving.md` — cross-validation, residual analysis, and the final improvements
- `06-lessons.md` — what I'd do differently if I started over

Reading them in order tells the full story.
