# Experiment 5 — Push CV-WAPE below 14.80

**Starting point:** experiment-4 best config (LightGBM-MAE + `sid_code` + temporal
context features) → CV-WAPE **14.80 ± 0.64** on the same 3-fold rolling-origin CV.

**Goal:** stack four orthogonal improvements identified in
[`../APPROACH.md`](../APPROACH.md) §10. Realistic target: **14.3 – 14.5 CV-WAPE**.

---

## Method overview

| § | Step | What | Expected ΔWAPE |
|---|---|---|---:|
| §4 | Baseline | Reproduce exp-4 config (sanity check; must land ~14.80) | — |
| §5 | Tuning | Random 8-config grid over `lr`, `num_leaves`, `min_child_samples`, `n_estimators` | −0.1 to −0.3 |
| §6 | Bias correction | Per-series additive shift, fit via leave-one-fold-out on OOF residuals | −0.1 to −0.3 |
| §7 | Multi-objective ensemble | Blend MAE + Poisson + Tweedie GBMs + Ridge with OOF-tuned weights | −0.2 to −0.4 |
| §8 | Multi-seed averaging | 5 random seeds of the MAE-GBM, simple mean of predictions | −0.05 to −0.15 |
| §9 | Final model | Refit on full train+val, apply every accepted improvement, save CSV | — |
| §10 | Summary | Cumulative WAPE arc + decision table | — |

Each section has a **decision rule** (e.g., "accept only if Δ > half-std") so that
improvements smaller than CV noise don't get committed to the final model.

---

## Why these four improvements (and not others)?

From `APPROACH.md` §10:

- **Hyperparameter tuning** — the exp-4 config was reasonable but never tuned under CV.
- **Bias correction** — MAE training finds the conditional *median*; for a right-skewed
  target (skew = 0.98) the median sits below the mean. Per-series additive shift is the
  cheapest possible correction.
- **Multi-objective ensemble** — exp-4 §7 found L2/MAE/Poisson/Tweedie all landed in
  similar WAPE ranges *with different residual structure*. Blending extracts diversity.
- **Multi-seed averaging** — reduces variance from LightGBM's subsampling stochasticity.

**Explicitly NOT pursued:**

- Per-series secondary models for the worst-5 cluster. APPROACH.md flags this as
  "variable, overfit-prone." Without strong evidence that specific series have a
  fundamentally different feature→target mapping, the overfit risk outweighs the
  expected gain.
- New feature engineering. The §8 analysis in `../data-analysis.ipynb` showed the top
  3 features have mutual correlation > 0.65 — most of the covariate signal is already
  captured. Diminishing returns.

---

## Notebook organization (vs experiment-4)

Experiment-4 mixed grid search, residual diagnostics, ablation, and submission refit in
the same notebook with shifting variable names. Experiment-5 fixes this:

- **§0** holds *all* configuration (paths, seeds, hyperparameters, grids). One place to edit.
- Each section: **markdown rationale → code → markdown decision**. No mixing of concerns.
- The submission refit lives in §9 only — analysis sections never touch `validation_input.csv`.
- §10 produces a single comparison table + WAPE arc plot.

---

## Files

- [`kp-experiment-5.ipynb`](kp-experiment-5.ipynb) — the main notebook (imputation = `ffill().fillna(0)`).
  Runs top-to-bottom on Kaggle (T4 GPU not required) or locally with the data at `data/`.
- [`kp-experiment-5-1.ipynb`](kp-experiment-5-1.ipynb) — exp-5.1 correctness delta. Identical pipeline
  except `impute_domain_features` uses `ffill().bfill()` within series. See §"Experiment 5.1" below.
- `validation_predictions_5.csv` — submission CSV from exp-5 §9.
- `validation_predictions_5_1.csv` — submission CSV from exp-5.1 (same pipeline, fixed imputation).
- This `README.md`.

---

## Reproducibility

- All random seeds set in §0 (`SEED = 42`) and used consistently in §1.
- The CV fold split function in §3 matches experiment-4 exactly, so the §4 sanity check
  should produce the same WAPE as exp-4 §8 (14.80 ± 0.64) — verify before trusting anything later.
- `series_mean` is recomputed per fold (leakage-free) inside `fold_split`.
- Temporal features for the submission are computed on `train + val_input` concatenated
  in §9, so lag/lead features cross the train→val boundary correctly.

---

## How to run

On Kaggle:
```python
# In §0 cell, swap to Kaggle paths:
DATA_DIR   = Path("/kaggle/input/datasets/xitij27/dlam-bonus-project-dataset/data")
OUTPUT_DIR = Path("/kaggle/working")
```

Locally:
```bash
pip install -r ../requirements.txt lightgbm scikit-learn scipy
jupyter notebook experiment-5/kp-experiment-5.ipynb
```

Approximate runtime on CPU:
- §4 baseline: ~1 min
- §5 tuning: ~5-10 min (8 configs × 3 folds)
- §6 bias correction: trivial (arithmetic only)
- §7 ensemble: ~3 min (12 model fits + optimization)
- §8 multi-seed: ~5 min (15 model fits)
- §9 final refit: ~2 min

Total: ~15-20 min.

---

## Results — actual CV and leaderboard

**CV-WAPE arc (3-fold rolling-origin CV):**

| Step | CV-WAPE | Δ vs baseline | Decision |
|---|---|---|---|
| §4. Baseline (exp-4 reproduce) | 14.798 ± 0.629 | — | — |
| §5. + Hyperparameter tuning | 14.775 | −0.023 | ❌ rejected (within noise) |
| §6. + Bias correction | 15.748 | **+0.950** | ❌ rejected (**made things worse**) |
| §7. + Multi-objective ensemble | 14.767 | −0.031 | ❌ rejected (within noise) |
| §8. + Multi-seed averaging | **14.726** | −0.072 | ✅ accepted |

**Public leaderboard:** experiment-5 → WAPE **14.923** (rank 1).

Comparison against previous submissions:
- experiment-5: 14.923 (this work)
- experiment-4.1: 15.033
- experiment-4: 15.476
- experiment-3: 17.828
- seasonal_mean baseline: 34.409

**CV → leaderboard drift:** +0.20 WAPE (14.73 → 14.92). Much tighter than exp-4's drift of +0.68 — the more conservative final pipeline transferred more honestly to held-out data.

---

## Experiment 5.1 — imputation-fix delta

`kp-experiment-5-1.ipynb` re-runs the entire pipeline with one change: `impute_domain_features`
uses `ffill().bfill()` (sorted by `series_id, timestamp`) instead of `ffill().fillna(0)`.
Rationale comes from `../data-analysis-2.ipynb` §2-4: gap lengths in the 10 missing-prone
forecast features are uniformly ≤ 4h (mean ≈ 1.05h), missingness is approximately MCAR
across columns and hour-of-day, so the rare leading-NaN rows are much better served by
the next valid value than by zero (which is far outside the typical range for these features).

**Side-by-side CV-WAPE under the same 3-fold split:**

| Step | exp-5 (ffill+0) | exp-5.1 (ffill+bfill) | Δ |
|---|---:|---:|---:|
| §4. Baseline | 14.798 | 14.782 | −0.016 |
| §5. + Hyperparameter tuning | 14.775 | 14.770 | −0.005 |
| §6. + Bias correction (rejected) | 15.748 | 15.709 | −0.039 |
| §7. + Multi-objective ensemble | 14.767 | 14.751 | −0.016 |
| §8. + Multi-seed averaging (accepted) | **14.726** | **14.728** | +0.002 |

Every step lands within ±0.04 WAPE of its exp-5 counterpart — far inside the fold std
(±0.62). The fix is **harmless and slightly cleaner** but does not change a single
qualitative decision: tuning / bias / ensemble all still fail their thresholds, multi-seed
still passes. Fold 1 is still the structural blocker (15.53 vs 15.58 — essentially identical),
which rules out imputation as the cause of its difficulty.

**Why it matters:** this is the cleanest possible falsification test for the imputation
choice. If `fillna(0)` had been masking signal for the leading-NaN rows, fixing it would
have produced a measurable drop. It didn't — so the leading-NaN rows are too rare to
matter, exactly as the gap-length analysis predicted. Adopt `ffill().bfill()` going forward
as the canonical imputation, but don't expect WAPE gain from it.

`validation_predictions_5_1.csv` was not submitted to the leaderboard; the CV evidence
above already settles it.

---

## Post-mortem — what went wrong and why

### Mistake 1 — bias correction made things worse (severity: high)

Every single fold got worse with the per-series mean-residual correction:

| Fold | Uncorrected | Corrected | Δ |
|---|---|---|---|
| 0 | 14.094 | 14.475 | +0.381 |
| 1 | 15.621 | 16.900 | +1.279 |
| 2 | 14.678 | 15.869 | +1.192 |

**Why it failed (conceptual error):** MAE training already finds the conditional
**median**. The median of OOF residuals is therefore ≈ 0 *by construction*. Adding the
**mean** of residuals shifts predictions away from the median toward the mean of a
right-skewed distribution — for per-series WAPE, this is a coin flip per row, and on
Fold 1 (where residual variance is highest) it hurts a lot more than it helps.

**What should have been done instead:**
- **Multiplicative scaling per series**: find scalar `s_i` for each series that
  minimizes WAPE on OOF (`s_i · pred`). No shape assumption.
- **Isotonic regression per series**: data-driven monotone mapping `pred → target` on
  OOF data.
- **No correction at all** — MAE is already median-calibrated.

### Mistake 2 — hyperparameter search was too shallow (severity: medium)

Best config gave only 14.775 vs 14.798 baseline (Δ = 0.023, within CV noise).

What was tuned: `lr`, `num_leaves`, `min_child_samples`, `n_estimators` (8 random configs out of 54).

What was NOT tuned: `reg_alpha`, `reg_lambda`, `subsample`, `colsample_bytree`,
`bagging_freq`, `feature_fraction_bynode`, `min_data_in_leaf`, `min_gain_to_split`.

**Realistic upside:** 0.2–0.4 WAPE with a proper Optuna run (50–100 trials, broader
parameter space, separate optima per objective family).

### Mistake 3 — the ensemble was internally redundant (severity: medium)

OOF-tuned blend weights:
- MAE: 0.85
- Poisson: 0.12
- Tweedie: **0.00**
- Ridge: 0.03

Tweedie got zero weight; Ridge got essentially zero. The cause is exactly the
multicollinearity ceiling identified in `../data-analysis.ipynb` §8: when four models
train on the **same 52 features** with the **same algorithm family** (or one closely
related Ridge), their errors are highly correlated and the blend collapses to ≈ MAE alone.

**What real ensemble diversity would require:**
- Different feature subsets per model
- Genuinely different model families (CatBoost, neural network with embeddings)
- Different temporal aggregations / lookback windows
- Different objective functions (Huber, Quantile at 0.4 / 0.5 / 0.6)

### Mistake 4 — Fold 1's structural difficulty was never addressed (severity: high)

Per-fold WAPE across every step of the experiment:

| Step | Fold 0 | Fold 1 | Fold 2 |
|---|---|---|---|
| Baseline | 14.09 | **15.62** | 14.68 |
| Tuned | 14.09 | 15.58 | 14.65 |
| Ensemble | 14.05 | **15.60** | 14.65 |
| Multi-seed | 14.04 | **15.53** | 14.61 |

Fold 1 (2023-06-02 → 2023-06-15) is consistently ~1.5 WAPE worse than Folds 0 and 2.
None of the four improvements specifically targeted this. **A method that brings
Fold 1 down from 15.5 to 14.5 alone would drop CV mean by ~0.3 WAPE** — bigger than
any of the four improvements we tried.

This window may contain a holiday, regime shift, or covariate spike that the global
model's features can't capture. Without specifically engineering for it, no amount of
general tuning will close the gap.

### Mistake 5 (mild) — only the smallest improvement actually worked

Multi-seed averaging gave Δ = 0.072 — within the predicted 0.05–0.15 range. It was
the only improvement that survived its decision check. This is technically a success,
but tells us all three of our "tier B" improvements (tuning, bias correction, ensemble)
either need rework or are dead-ends with the current feature set.

---

## What this experiment validated

Positive findings worth keeping:

1. **The measurement system is reliable.** CV-to-leaderboard drift is +0.20, well
   within the CV std (0.63). When we measure an improvement on CV, it transfers
   honestly to the leaderboard. We are not training-on-test or cheating ourselves.

2. **We are near a feature-set ceiling.** With multicollinear top 3 features and a
   global LightGBM, gains < 0.1 are statistically noise. Real improvements require
   **new signal types** (different features) or **targeted fixes** (Fold 1 / worst
   series), not more tuning of the same setup.

3. **Decision rules saved us from worse outcomes.** The bias correction would have
   been silently applied without the half-std threshold and the explicit OOF
   evaluation, costing ~1 WAPE on the leaderboard. The decision-rule discipline is
   worth keeping in future experiments.

4. **Notebook organization works.** All config in §0, single-purpose sections, isolated
   submission refit — running and revisiting this notebook is far easier than exp-4.

---

## Recommendations for experiment 6

Ranked by **expected ΔWAPE / effort**:

### Tier A — high impact (target each: 0.2+ WAPE)

1. **Fix bias correction** with multiplicative per-series scaling or isotonic
   regression. The concept is sound; only the implementation was wrong. Expected gain:
   **0.1 – 0.3**.

2. **Investigate Fold 1 explicitly.** Identify what makes 2023-06-02 → 2023-06-15
   structurally harder (holidays, covariate distribution shifts, residual spike
   patterns). Engineer features that flag it — even a binary "is_anomalous_window"
   flag could help if there's a clear signal. Expected gain: **0.2 – 0.5**.

3. **Aggressive Optuna tuning.** 50–100 trials, full LightGBM parameter space,
   separate optima per objective. Expected gain: **0.2 – 0.4**.

### Tier B — medium impact (target each: 0.1 WAPE)

4. **Add CatBoost to the ensemble.** Genuinely different categorical handling
   (ordered target statistics) → genuinely different errors → real ensemble
   diversity. Expected gain: **0.1 – 0.2**.

5. **Lag-168 of covariates.** Only lag-1 and lag-24 were tested. Pooled covariate
   ACF (from the data analysis) suggests lag-168 may carry independent signal for
   features with weak weekly autocorrelation in the covariate itself. Expected gain:
   **0.05 – 0.15**.

6. **Polynomial / interaction features.** Explicit `queue_pressure ×
   workload_intensity`, `queue_pressure²` for the top correlated features. Trees find
   interactions through splits but explicit features can sharpen them. Expected gain:
   **0.1 – 0.2**.

### Tier C — low impact / high risk (skip unless A & B exhausted)

7. **Per-series specialist models** for worst-5 series. Reserve for after Fold 1 fix.
8. **More seeds (10–15) + averaging.** Marginal extra gain over 5 seeds.
9. **Stacking** with a linear meta-learner over base-model OOF predictions.

### Suggested experiment 6 plan

If I had to pick one path:

1. **§6.1** — implement bias correction correctly (multiplicative + isotonic, choose by CV)
2. **§6.2** — Fold-1 root-cause analysis + engineered feature(s)
3. **§6.3** — Optuna sweep
4. **§6.4** — CatBoost added to ensemble
5. **§6.5** — Refit + submission

Realistic target: **14.0 – 14.4 CV-WAPE / 14.3 – 14.7 leaderboard.**

If those four don't get below 14.4 CV, the dataset has been mined out at this feature
set and the next move is fundamentally new features (interaction, lag-168, etc.) or a
fundamentally different model class.
