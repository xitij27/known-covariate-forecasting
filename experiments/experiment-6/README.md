# Experiment 6 — XGBoost as a Diversity Arm

**CV-WAPE = 14.728** (identical to exp-5.1 — the LGBM+XGB blend was rejected).
**Public leaderboard WAPE = 14.927656** (identical to exp-5.1 to five decimal places).

This notebook tested whether XGBoost-MAE could provide *algorithmic* diversity on top of
the LightGBM-MAE pipeline from experiment-5.1. The hypothesis was that depth-wise
(XGBoost) vs leaf-wise (LightGBM) tree growth would produce **decorrelated residuals**,
so a convex blend of the two would beat either single model.

**It did not.** OOF residuals of LGBM-MAE and XGB-MAE on this dataset are correlated at
**r = 0.9955**. The 2-way blend gain (Δ = +0.048 WAPE) is just below the 0.05 accept
threshold, and a 5-way blend with three additional arms (Poisson + Tweedie + Ridge)
adds only +0.016 more. Multi-seed averaging on LGBM-MAE was the only improvement that
survived its accept rule — which is exactly the same improvement exp-5.1 already had.

This is a **useful negative result.** The single number `r = 0.9955` rules out
algorithm-diversity ensembles as a path forward on this dataset and points to where the
real bottleneck is: the feature set is largely saturated, and ~14.8 WAPE is close to
the floor for any tree-GBM on these inputs.

---

## Hypothesis and motivation

Experiment-5 §7 tested a 4-way LightGBM ensemble (MAE + Poisson + Tweedie + Ridge) and
**rejected it** (Δ = +0.031 WAPE, within noise). The diagnosis was that three of the
four arms came from the same library; the residuals were too correlated to gain
anything from blending.

The natural follow-up was to introduce a **different algorithm** — keeping the loss
and features identical to isolate the algorithm's contribution. XGBoost was the
obvious candidate because:

- It is the most mature library that grows trees **depth-wise**, where LightGBM grows
  **leaf-wise**. The two policies select different splits at each node and
  *in principle* converge to functions of the inputs that disagree on edge cases.
- It supports `reg:absoluteerror` (MAE), so the loss matches LightGBM exactly. Any
  residual difference must come from the algorithm, not the objective.
- It has native categorical handling (`enable_categorical=True, tree_method="hist"`),
  so `sid_code` is treated the same way as in LightGBM.

Same features (52 columns), same CV (3-fold rolling-origin, 336h test per fold), same
imputation (`ffill().bfill()` within series, from `data-analysis-2.ipynb`). The only
variable was the library.

---

## Plan vs reality

| § | Step | Expected ΔWAPE | Observed | Verdict |
|---|------|---------------:|---------:|---------|
| §4 | Reproduce LGBM-MAE baseline | — (= 14.78) | 14.782 ± 0.616 | ✅ pipeline reproduces exp-5.1 exactly |
| §5 | XGB-MAE solo | unclear | 14.799 ± 0.603 | ≈ LGBM (within noise, Δ = +0.017) |
| §6 | XGB hyperparameter grid | −0 to −0.1 | best tuned 14.809 (worse than default) | ⚠️ rejected — defaults already near-optimal |
| §7 | **LGBM+XGB 2-way blend** | −0.1 to −0.3 | 14.733 (Δ +0.048 vs best solo) | ⚠️ rejected — just below 0.05 threshold |
| §8 | + Poisson + Tweedie + Ridge (5-way) | −0 to −0.2 | 14.718 (Δ +0.016 vs §7) | ⚠️ rejected — gain within noise |
| §9 | Multi-seed averaging on winner | −0.05 | 14.728 (Δ +0.053 vs §4) | ✅ accepted |
| §10 | Refit + submit | — | LB 14.927656 | identical to exp-5.1 |

The only adopted change vs exp-5.1 was the same multi-seed averaging exp-5.1 already
had. The winning composition is `WINNER_KIND = "lgb_only"`. XGBoost contributes
**nothing** to the final submission.

---

## The key finding: r(residual_LGBM, residual_XGB) = 0.9955

This single number is the headline result. From cell 16 of the notebook:

```
Pearson correlation of OOF residuals (LGBM vs XGB): 0.9955
  - r > 0.95  → models are essentially identical, blend won't help.
  - r ~ 0.7-0.9 → meaningful diversity, blend likely helps.
  - r < 0.5   → very strong diversity (rare on same features).
```

Three implications:

### 1. Depth-wise vs leaf-wise tree growth does not produce meaningful diversity here

Whatever the textbook difference between the two growth heuristics, by the time the
model has 600 trees of depth 6 (≈ 64 leaves) trained on 410k rows × 52 features, the
distinction is washed out. Both algorithms converge to nearly identical functions of
the same inputs.

This is consistent with the practical observation that LightGBM and XGBoost typically
land within ±0.5% of each other on tabular regression benchmarks when configured
comparably.

### 2. The signal in the data is largely captured

When two models with different inductive biases produce 99.55%-correlated residuals,
what is left in the residuals is either pure noise or structure that **neither** model's
inductive bias can extract. WAPE ~14.8 is therefore likely close to the achievable
floor for any tree-based regressor on this feature set.

### 3. The upper bound on ensemble gain matches the observed gain

Ambiguity decomposition (Krogh & Vedelsby 1995): the variance reduction from blending
two models is bounded above by `(1 − r) × variance_term`. With `r = 0.9955` the bound
is ~0.45% of the variance — exactly the order of the observed +0.048 WAPE gain. The
blend is performing at its theoretical maximum; that maximum is simply small.

This is also why the 5-way blend in §8 gained another +0.016: the three extra arms
(Poisson, Tweedie, Ridge) have *some* genuine diversity from LGBM-MAE/XGB-MAE — they
use different loss functions or are linear — but the dominant LGBM+XGB pair caps the
total available gain.

---

## Per-fold breakdown

| fold | days | LGBM-MAE | XGB-MAE | 2-way blend | 5-way blend |
|-----:|:----:|---------:|--------:|------------:|------------:|
| 0    | 139–152 | 14.082 | 14.125 | 14.055 | — |
| 1    | 153–166 | 15.581 | 15.589 | 15.522 | — |
| 2    | 167–180 | 14.683 | 14.683 | 14.623 | — |
| mean |       | **14.782** | **14.799** | **14.733** | **14.718** |

**Fold 1 (test days 153–166) is consistently ~0.8 WAPE worse than the other two folds**
across every model tested — LGBM, XGB, Poisson, Tweedie, Ridge. This was also true in
exp-5. A focused investigation of *what specifically is hard about days 153–166*
would likely be more productive than additional ensemble work.

---

## Blend weights from §8 (illustrative)

Optimal weights when the SLSQP optimizer was given all five arms:

| arm | weight |
|---|---:|
| LGBM-MAE | 0.3603 |
| **XGB-MAE** | **0.5280** |
| LGBM-Poisson | 0.0618 |
| LGBM-Tweedie | 0.0245 |
| Ridge | 0.0254 |

Two observations:

- The optimizer assigned XGB-MAE the **largest** weight (0.528) despite XGB being
  marginally worse solo (14.799 vs LGBM's 14.782). This is the optimizer finding tiny
  diversity gains by leaning into XGB. But the absolute gain is in the noise.
- Poisson + Tweedie + Ridge together get only ~11% of the weight, and contribute
  ~0.016 WAPE — most of the 5-way gain over the 2-way comes from these three arms,
  not from any deeper LGBM/XGB recombination.

---

## Leaderboard outcome

| Model | LB WAPE | LB MAE | LB Rank |
|---|---:|---:|---:|
| experiment-5 (old imputation) | 14.923378 | 1.642725 | #1 |
| chronos-2-native-train | 14.927777 | 1.643209 | #2 |
| experiment-6 | 14.927656 | 1.643196 | #3 |

`experiment-6` and `experiment-5.1` produce **identical predictions to 5 decimal
places**. This is expected: the winning composition (`lgb_only` + 5-seed averaging)
uses the same seeds, same features, same imputation, and the same LightGBM library
version as exp-5.1. The CSVs are bit-for-bit equivalent.

The marginal gap to exp-5 (Δ = +0.004 WAPE) traces to the imputation change
(`fillna(0)` → `ffill().bfill()`) verified harmless in exp-5.1. Not a real
regression — well within fold-noise (±0.6).

---

## Why this experiment was worth running

Negative results split into two categories:

- **Low-value** — the experiment was poorly designed and we cannot tell what we
  learned from the failure.
- **High-value** — the experiment was well-targeted and the failure mode is itself
  informative.

This is the second kind. The number `r = 0.9955` is now hard evidence for the
following claims:

- Algorithm-diversity blends *within tree-GBM families* are unlikely to move WAPE on
  this dataset.
- The exp-5 rejection of the 4-way blend was correct for this same underlying reason
  — not because the implementation was wrong.
- Trying yet-another tree-GBM library (CatBoost, sklearn HistGradientBoosting) is
  unlikely to beat ~14.78. We now have evidence, not a guess.

The notebook also confirmed that the imputation fix from exp-5.1 (`ffill().bfill()`)
is stable across a different downstream model (XGBoost), since both models reproduced
their expected solo CV-WAPE.

---

## What can be done better

Ranked by expected return, with honest caveats. The recurring theme: future gains
must come from inductive biases that are **not** tree-based regression on the current
feature set.

### 1. A non-tree model family (highest expected return)

A model that is structurally different from trees is the most promising remaining
avenue, because the diversity-blend hypothesis only failed *within* the tree family.
Candidates:

- **Small MLP** (2–3 hidden layers, ~200 units, dropout, BatchNorm). Smooth
  nonlinearities replace trees' piecewise-constant outputs. On the same 52 features,
  an MLP should produce more genuinely decorrelated residuals — that is the testable
  claim. Train with MAE loss to match.
- **CatBoost with target encoding for `series_id`** — uses an *ordered boosting*
  framework specifically designed to reduce target leakage in categorical encoding
  and may extract per-series structure trees miss.

Expected gain if a blend works: 0.1–0.3 WAPE. Risk: a well-trained MLP on the same
features might just rediscover the tree-fit signal and end up at r > 0.9 again — in
which case the bottleneck is the features, not the model class.

**Recommended test:** train MLP-MAE in `experiment-7`, compute residual correlation
with LGBM-MAE OOF. If r < 0.85, build the blend; if r > 0.95, document and move on.

### 2. Per-series investigation of fold 1's difficulty

Fold 1 (test days 153–166) is consistently 0.8 WAPE worse than folds 0 and 2 across
every model tested. If a handful of series account for most of that excess error,
isolating them is more productive than further global model work.

**Action items:**

- Add a per-(fold × series) WAPE table to `data-analysis.ipynb` and look at the tail.
- Inspect whether fold 1's hard series have a distribution shift or a regime change
  in any covariate during days 153–166.
- Consider per-series corrections only for the worst-N series, with strict hold-out
  validation. APPROACH.md flagged this as overfit-prone but the upside is largest here.

### 3. Sequence-aware model on the residuals

The current pipeline is per-hour and treats temporal context only through explicit
lag/lead features. A 1D-CNN or small RNN trained to predict the *residuals* of the
LGBM-MAE prediction from a window of recent timestamps could pick up temporal
structure that pointwise regression misses.

**Caveat:** if the residuals had exploitable temporal structure, the lag/lead
features in exp-4 §8 (which include `lag1`, `lag24`, `lead1`, `lead24` and rolling
means/stds) would already have captured it. So this is a long shot, but cheap to
implement.

### 4. Bigger multi-seed averaging

Increase `SEEDS_ENSEMBLE` from 5 to 15–20. Each additional seed contributes
diminishing variance reduction; expect ~0.02–0.04 WAPE.

**Action:** edit `SEEDS_ENSEMBLE` in §0 of the notebook and re-run §9 + §10.

### 5. Feature engineering with caution

The top three features have 65%+ mutual correlation (`data-analysis.ipynb` §8), so
adding more *similar* features hits diminishing returns. Two angles that haven't
been tried:

- **Explicit interaction features** (e.g. `queue_pressure × workload_intensity`).
  Trees can represent interactions internally via deep splits — but they need depth
  to do it, and with `max_depth = 6` we cap the interaction order. Explicit
  interactions may move the floor by ~0.05 WAPE.
- **Longer-horizon lags** (lag 168 = one week of the top covariates), if seasonal
  structure persists beyond 24h.

Lower expected return than (1)–(3) but cheap to test.

### 6. Stricter CV (5-fold instead of 3-fold)

Several decisions in this notebook landed at the edge of their accept thresholds —
notably the 2-way blend at +0.048 vs a 0.05 threshold. With 5 folds, both the mean
gain estimate and the std estimate would be more reliable, and borderline decisions
would be less seed-dependent. Cost: ~70% more runtime.

### 7. Accept the WAPE floor

Given `r = 0.9955` and that the public leaderboard is now within 0.01 WAPE of itself
across the top three models, the dataset's signal is probably near-saturated by the
current pipeline. Declaring victory at ~14.93 LB-WAPE and moving on is a defensible
decision if the goal is a single solid submission rather than exhaustive
experimentation.

---

## Reproducibility

- **Notebook:** [`kp-experiment-6.ipynb`](kp-experiment-6.ipynb). Runs top-to-bottom on
  Kaggle CPU in roughly 20–25 minutes. All configuration is in §0; section-by-section
  narrative is in the markdown cells; decisions are made explicit via accept-thresholds
  at the end of each section.
- **Submission:** [`validation_predictions_6.csv`](validation_predictions_6.csv) —
  written by §10. 32,256 rows, one per (`series_id`, `timestamp`) in
  `forecast_index_validation.csv`. Predictions clamped at 0.
- **Dependencies:** `lightgbm==4.6.0`, `xgboost==3.2.0`, `scikit-learn`, `scipy`,
  `pandas`, `numpy`, `matplotlib`. Kaggle's default Python 3 environment provides all
  of these.
- **Imputation:** uses the `ffill().bfill()` recipe verified in
  [`../data-analysis-2.ipynb`](../data-analysis-2.ipynb) §2-5 and adopted in
  [`../experiment-5/kp-experiment-5-1.ipynb`](../experiment-5/kp-experiment-5-1.ipynb).
- **CV folds:** 3-fold rolling-origin, 336h test per fold, expanding training window.
  Identical to exp-4, exp-5, and exp-5.1.
- **Seeds:** `SEED = 42` everywhere except multi-seed averaging which uses
  `[42, 101, 202, 303, 404]`.
