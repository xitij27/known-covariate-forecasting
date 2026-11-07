# Making it trustworthy, and pushing further

I had a working model at WAPE 16.53. But the score came from a single held-out 14-day window. Was it real? Could I do better? Experiment 4 was about answering both questions, in that order.

## Step 1 — proper cross-validation

A single train/test split can be lucky. To get a trustworthy number I set up rolling-origin cross-validation: three expanding-window folds, each test block 336 hours.

- Fold 0: train on days 1–138, test on days 139–152.
- Fold 1: train on days 1–152, test on days 153–166.
- Fold 2: train on days 1–166, test on days 167–180.

For each fold I recomputed `series_mean` using only that fold's training data, which eliminated the small peek from the single-split estimate.

Result for the experiment-3.2 model (LightGBM, L2 objective):

| fold | seasonal | linear | lightgbm |
|---|---|---|---|
| 0 | 30.17 | 17.42 | 16.28 |
| 1 | 37.27 | 19.24 | 18.99 |
| 2 | 31.40 | 18.09 | 16.55 |
| **mean ± std** | 32.95 ± 3.10 | 18.25 ± 0.75 | **17.27 ± 1.22** |

The honest cross-validated score was **17.27 ± 1.22**. About 0.75 higher than the single-split 16.53 — so the single split had been slightly optimistic, but in the right ballpark. Fold 1 was the structurally hardest (the seasonal baseline also did worst there).

This was my new baseline. Every improvement going forward had to beat 17.27 ± 1.22 on the CV.

## Step 2 — where does the error live?

Before piling on changes, I broke down the out-of-fold residuals three ways. This was the most useful single diagnostic of the whole project.

**By hour of day.** Hours 0–2 (late night) had WAPE around 25, while hours 19–20 (evening) sat around 14. An 11-point spread.

**By series.** Best series: 8.55 WAPE. Worst: 27.99 (`unit_073`). A 19-point spread across units.

**By queue_pressure quintile.** Bottom quintile: 28 WAPE. Top quintile: 14. Monotonic decrease.

**Bias.** Mean of (predicted − actual) was −0.31. A slight under-prediction.

## What the residuals told me

Three patterns, three different stories:

1. **The hour and queue_pressure patterns are partly a WAPE math artifact.** WAPE is total absolute error divided by total actual. When actual values are small (late at night, low queue pressure), the denominator is small, and even small errors look big in percentage terms. The model wasn't necessarily failing on those rows — WAPE just hates them by construction.

2. **But there's also a real signal in the low-target rows.** L2 loss optimizes mean-squared error. It cares a lot about big errors on big-value rows, and not much about small-value rows. So the model was under-fitting low-y regions. Switching to MAE — which treats every row equally in absolute error — should help.

3. **The per-series spread (8.5 to 28) was the loudest signal.** It meant my one global model was averaging over very different per-series relationships. Some units behaved like the "average unit" the model fit. Some didn't.

These observations gave me a clear next step.

## Step 3 + 5 — fix the objective and add per-series structure

I tested a 4 × 2 grid:

- 4 objectives: L2 (the default), MAE, Poisson, Tweedie.
- 2 settings: with vs without `series_id` as a categorical feature in LightGBM.

That's 8 configurations × 3 CV folds = 24 model fits. Took a few minutes on Kaggle.

| objective | series_id=no | series_id=yes |
|---|---|---|
| L2 | 17.27 ± 1.22 | 16.64 ± 1.48 |
| **MAE** | 16.19 ± 0.47 | **15.39 ± 0.59** |
| Poisson | 17.21 ± 1.00 | 16.23 ± 1.09 |
| Tweedie | 17.17 ± 0.98 | 16.55 ± 1.01 |

The decomposition:

- **MAE alone was the biggest single lever** (−1.08 over L2). And the variance dropped enormously (std 0.47 vs L2's 1.22). MAE produces stable models because it's not pulled around by big outlier errors the way L2 is. And it matches WAPE structurally.
- **Poisson and Tweedie helped a bit** but not as much as MAE. So the lesson was "stop using squared error," not "the target is Poisson-distributed."
- **`series_id` as a categorical added another −0.6 to −1.0** on top of any objective. The 19-point per-series spread was real; letting LightGBM split on series identity captured a meaningful chunk of it.

Combined: MAE + `series_id` categorical, **CV-WAPE 15.39 ± 0.59**. Down −1.88 from baseline.

Sanity check: the L2/no-categorical cell reproduced 17.27 exactly, so I knew the grid setup was honest.

## Step 4 — temporal context from the covariates

Now I had a stable, well-objective model with per-series structure. What was missing? Probably temporal context. Even though the per-hour prediction depended mostly on the hour's covariates, the target at time *t* could also depend on how those covariates had been moving — was queue pressure building? About to spike?

The key realization: **`validation_input.csv` has the full domain trajectory across the 14-day horizon.** So I could use *both* lags AND leads of the domain features. Not target lags (those don't exist), but covariate lags and leads — all known at inference time.

I added:

- Lags at t−1, t−24 of the top 5 domain features (10 features).
- Leads at t+1, t+24 of the same 5 (10 features).
- Rolling mean and std over 6h and 24h windows for the top 3 (12 features).

That's 32 new features. NaN at the start/end of each series (where a lag can't reach back, or a lead can't reach forward) — I left these as NaN since LightGBM handles missing natively.

For the actual submission I had to be careful: when computing lags for the val window, the lag at the *start* of val should reach back into the last hours of training data. So before computing the temporal features, I concatenated `train.csv` and `validation_input.csv` and ran the temporal feature engineering on the combined frame. This way val rows got real lag context from training, and late-train rows could "see" leads into val.

Result on CV:

| fold | without temporal (§7) | with temporal (§8) | Δ |
|---|---|---|---|
| 0 | 14.78 | 14.08 | −0.70 |
| 1 | 16.18 | 15.64 | −0.54 |
| 2 | 15.20 | 14.68 | −0.52 |
| **mean** | **15.39** | **14.80** | **−0.58** |

All three folds improved. **CV-WAPE 14.80 ± 0.64.**

## Re-checking the residuals

After this round I re-ran the residual breakdown to see what had changed:

| pattern | before (§5) | after (§8) |
|---|---|---|
| overall OOF WAPE | 17.17 | 14.75 |
| hour-of-day min / max | 14 / 26 | 12 / 21 |
| per-series best / median / worst | 8.5 / 17 / 28 | 7.5 / 14.6 / 26 |
| QP quintile lowest → highest | 28 → 14 | 23 → 12 |
| mean bias | −0.31 | −0.48 |

Improvement was broad — every hour, every series, every quintile got better. But the patterns persisted:

- **The per-series spread is still big** (7.5 vs 26). The worst-5 units (`unit_073`, `unit_058`, etc.) remained hard. Either they're genuinely noisier, or their feature-to-target relationship diverges more from the global average than `series_mean` + `series_id` can capture.
- **The QP-quintile spread narrowed but didn't flatten.** Most of what's left is the WAPE arithmetic — low queue_pressure → low target → small denominator → inflated per-row WAPE. Modeling won't change that.
- **The bias actually got slightly worse.** MAE finds the conditional median; for a right-skewed target the median sits below the mean. Cheap to correct post-hoc if I wanted to.

## Why I stopped at 14.80

I added up where the remaining error was:

- About half is WAPE arithmetic on low-target rows.
- About half is per-series specific noise or dynamics that would need per-series secondary models to chase.

Realistic ceiling with another pass (ensembling, light hyperparameter tuning, bias correction): maybe 14.3–14.5. Worth ~0.3–0.5 WAPE, not transformative. Better to stop, bank the gain, and confirm the leaderboard transfer.

The cumulative arc on cross-validation:

- L2 baseline (= experiment-3.2 honestly evaluated): **17.27**
- + MAE + `series_id`: **15.39** (−1.89)
- + temporal context: **14.80** (−0.59)
- Total: **−2.47 WAPE**, about 14% relative.

And against `seasonal_mean` (~33 on CV, 34.4 on the public leaderboard), the final model is about 55% better.

Saved the submission. Wrote up.
