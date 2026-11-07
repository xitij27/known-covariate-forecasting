# Experiment 4 — Best Direct-Regression Model (CV-WAPE 14.80)

**Final held-out CV-WAPE = 14.80 ± 0.64** (3-fold rolling-origin CV, leakage-free
`series_mean`). vs `seasonal_mean` (CV ~33; public LB 34.41) → ~55% reduction. vs
experiment-3.2's single-split 16.53 → cumulative −1.73. Leaderboard pending submission.

Builds directly on experiment-3.2's direct-regression model. Three principled steps deliver
the gains:

| step | what | CV-WAPE | Δ |
|---|---|---|---|
| §4  | leakage-free 3-fold CV baseline (= experiment-3.2 model honestly evaluated) | 17.27 ± 1.22 | — |
| §7  | + MAE objective + `series_id` as LightGBM categorical | **15.39 ± 0.59** | −1.89 |
| §8  | + temporal context (lags / leads / rolling of top domain features) | **14.80 ± 0.64** | −0.58 |

The cumulative arc (CV-WAPE): 17.27 → 14.80 = **−2.47** (~14% relative). Returns are
clearly diminishing across the three steps — see "Why we stopped" below.

---

## What we already knew from experiment-3.2

The target is approximately a per-hour function of the *known future* domain forecast
features (`queue_pressure_forecast` OOS r² ~47%, `network_pressure_forecast` ~43%, etc.).
`validation_input.csv` has no `target` column, so target-lags don't exist at inference —
the right model maps known covariates at time *t* directly to `target[t]`. That gave WAPE
16.53 on a single held-out split (LightGBM, L2 objective). Experiment 4 starts here and
asks: how much further can this go without changing the framing?

---

## §4 — Rolling-origin CV with leakage-free `series_mean`

3 expanding-window folds, each test block = 336h. `series_mean` (per-series training mean)
is recomputed per fold from the train portion only, closing the ~8% peek in the original
single-split estimate.

The trustworthy baseline is **CV-WAPE 17.27 ± 1.22** — about 0.75 worse than the
single-split 16.53, exactly as predicted (negligible optimism given target stationarity).
Fold 1 is structurally harder (also worst for `seasonal_mean`), pushing std up.

---

## §5 — Residual diagnosis (where the error lives)

OOF residuals broken down three ways revealed two big levers:

| signal | pattern | implication |
|---|---|---|
| Per-series WAPE 8.5 → 28 (19-point spread) | The global model averages over very different per-series relationships | Add `series_id` as a native LightGBM categorical (Step 5) |
| Hour-of-day 14 → 26 ; QP-quintile 28 → 14 (monotonic) | Partly a WAPE-mechanism artifact (low-y → small denominator); also consistent with L2 under-fitting low-y rows | Switch objective from L2 to MAE / Poisson / Tweedie (Step 3) |
| Mean bias −0.31 (~3% under-prediction) | L2 + long-tailed positive target | Also addressed by MAE / Poisson |

Not noise-floor territory — real structure left to capture.

---

## §7 — Step 3 + 5: objective × `series_id` categorical

Tested the **4 × 2 grid** (objective ∈ {L2, MAE, Poisson, Tweedie} × `series_id` ∈ {no, yes})
on the same CV.

| objective | series_id=no | series_id=yes |
|---|---|---|
| L2      | 17.27 ± 1.22 | 16.64 ± 1.48 |
| **MAE** | 16.19 ± 0.47 | **15.39 ± 0.59** |
| Poisson | 17.21 ± 1.00 | 16.23 ± 1.09 |
| Tweedie | 17.17 ± 0.98 | 16.55 ± 1.01 |

**Decomposition:**
- **MAE objective** alone gives −1.08 over L2.
- **`series_id` categorical** gives −0.6 to −1.0 on top of *any* objective.
- Poisson/Tweedie also beat L2 but less than MAE — the lesson is "stop using squared error,"
  not "the target is Poisson-distributed."
- MAE has *much* lower std (0.47 vs L2's 1.22) and shrinks the fold-1 hardness gap.

Sanity check: the `L2 / no` cell reproduced §4's 17.27 exactly.

---

## §8 — Step 4: leakage-free temporal context

`validation_input.csv` provides the full domain trajectory across the 336h horizon, so
**lags AND leads** of domain features are all known at inference time. (Target lags remain
unavailable.)

Added on top of the §7 winner:
- **Lags** at t−1, t−24 and **leads** at t+1, t+24 for the top 5 domain features
  (`queue_pressure_forecast`, `network_pressure_forecast`, `workload_intensity`,
  `shock_risk`, `event_load_forecast`) → 20 features.
- **Rolling mean + std** over 6h and 24h windows for the top 3 → 12 features.

Total 32 new features → 52 total. NaN at series edges left in place — LightGBM handles
missing natively.

| fold | §7 | §8 | Δ |
|---|---|---|---|
| 0 | 14.78 | 14.08 | −0.70 |
| 1 | 16.18 | 15.64 | −0.54 |
| 2 | 15.20 | 14.68 | −0.52 |
| **mean** | **15.39** | **14.80** | **−0.58** |

All three folds improved; the gain is broad, not concentrated on one fold.

For the submission, the refit concatenates `train.csv` + `validation_input.csv` **before**
computing temporal features, so val-input rows' lags reach back into training data and
late-train rows' leads reach into val_input — consistent with the inference setup.

---

## §8 residual re-diagnosis

| pattern | §5 (L2 baseline) | §8 (final) |
|---|---|---|
| Overall OOF WAPE | 17.17 | **14.75** |
| Hour-of-day min / max | 14.06 / 25.61 (h1) | 11.69 / 21.39 (h1) |
| Per-series best / median / worst | 8.55 / 16.93 / 27.99 | 7.55 / 14.63 / 25.96 |
| Worst-5 series | `unit_073`, `_058`, `_022`, `_066`, `_056` | `unit_073`, `_058`, `_066`, `_056`, `_039` |
| QP-quintile lowest → highest | 28.04 → 13.97 | 23.43 → 11.63 |
| Mean bias | −0.31 | −0.48 |

Improvement is broad. Three residual patterns *persist* and explain what's left:

- **QP-quintile pattern is largely a WAPE-mechanism artifact.** Low `queue_pressure` → low
  target → small denominator → inflated per-row WAPE. The model isn't "failing" on low-y
  rows in absolute terms; WAPE just over-weights them. Further modeling won't move this.
- **The `unit_073` cluster** (worst-5) remains. These series likely have either genuinely
  higher noise or feature→target mappings that diverge more from the global average than
  `series_id` + `series_mean` captures. Per-series secondary models could help, at the cost
  of significant complexity and overfit risk.
- **Bias slightly worse** (−0.48 vs −0.31): MAE finds the conditional *median*; for a
  right-skewed target the median sits below the mean, and sharper features make this more
  pronounced. Cheap to correct post-hoc if pursued.

---

## Why we stopped at 14.80

The diminishing-returns curve became explicit: −1.89 from §7, −0.58 from §8. The remaining
residual structure breaks down roughly as:

- **~half WAPE-arithmetic artifact** (the QP-quintile / low-y / late-night rows) — not
  fixable by modeling.
- **~half per-series specific noise or dynamics** (the worst-5 cluster) — would require
  per-series secondary models with their own overfit risk.

Realistic ceiling with one more pass (Step 6 — ensemble + light tuning + bias correction):
~14.3–14.5. Worth ~0.3–0.5 WAPE; not transformative. Better to bank the cumulative −2.47
WAPE gain and lock in the leaderboard transfer before further iteration.

---

## Method summary

- **Model**: LightGBM with `n_estimators=600, learning_rate=0.03, num_leaves=63,
  subsample=0.8, colsample_bytree=0.8, min_child_samples=50, reg_lambda=1.0`,
  `objective='regression_l1'` (MAE), `categorical_feature=['sid_code']`.
- **Features (52 total)**: 11 domain forecasts (ffill imputed within series) + 7 time
  encodings + `series_mean` (per-series train mean) + `sid_code` (stable train+val
  series-ID code) + 32 temporal features (lag/lead/rolling of top domain features).
- **Validation**: 3-fold rolling-origin CV (expanding window, 336h test blocks). All
  evaluations reported as CV mean ± std.
- **Submission**: refit on full `train.csv` + `validation_input.csv` (concatenated for
  temporal-feature computation only), predict per-timestamp, clamp at 0, align to
  forecast index.

---

## Is this leakage?

Same answer as experiment-3.2: no. The validation targets are held by Kaggle (not in any
file we load — `validation_input.csv` has 24 columns vs train's 25). The 14.80 estimate is a
backtest on rolling 14-day blocks inside `train.csv`, with each fold's `series_mean`
recomputed from that fold's train portion (no peeking). Temporal features for the
submission are computed on a train+val_input concatenation only because val's domain
features are *provided inputs* — using `queue_pressure[t+24]` as a feature when predicting
`target[t]` is the intended task, not leakage of the answer. See
[`../experiment-3.2/README.md`](../experiment-3.2/README.md) "Is this leakage?" section for
the full treatment.

---

## Files

- [`kp-experiment-4.ipynb`](kp-experiment-4.ipynb) — standalone notebook (no PyTorch, no
  autoregression). Runs top-to-bottom on Kaggle.
- [`../experiment-3.1/`](../experiment-3.1/) — original LSTM notebook + the Section 6b
  diagnostics that revealed the LSTM was the wrong tool.
- [`../experiment-3.2/`](../experiment-3.2/) — direct-regression baseline (L2 LightGBM,
  single-split WAPE 16.53) + its README.

---

## Possible further gains (not pursued)

- **Bias correction** (post-hoc optimal additive or multiplicative shift fit on OOF):
  0.1–0.3 WAPE.
- **Ensemble** (blend MAE-GBM + Poisson-GBM + linear, possibly different seeds): 0.2–0.4.
- **Per-series secondary models** targeting the worst-5 cluster: variable, overfit-prone.
- **Light hyperparameter tuning** under the CV: 0.1–0.3.
- **CatBoost** (better native categorical handling for `series_id`): plausibly modest.

Combined realistic ceiling: ~14.3–14.5 CV-WAPE.
