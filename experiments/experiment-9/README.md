# Phase 9 — Targeting the error where it actually lives

Phase 7 closed with two leaderboard submissions that lead on WAPE (ensemble
14.68, gated MLP 14.75). Phase 8 (TiDE) confirmed that swapping the architecture
on the fixed feature set does not move the number (CV-WAPE 16.37). Phase 9 changes
strategy: instead of new architectures, **attack the specific pockets where the
remaining error is concentrated**, identified by mining the shipped blend OOF.

The full program is in [`../IMPROVEMENT-FRONTIERS.md`](../IMPROVEMENT-FRONTIERS.md).
The motivating diagnostic (on `experiment-7/oof_predictions_7b_blend.csv`):

- **Top 5% of rows by |error| = 36% of the total WAPE numerator.**
- Those rows are spikes the MAE models **under-predict** (mean residual on the
  top-5%-target rows: ~+5.8 in target units).
- Per-series WAPE spans 7.1 → 25.9.
- Cross-series residual synchrony is only 3.5% → global/cross-sectional features
  are **not** a lever (recorded as a dead end).

| Notebook | Frontier | Result | Verdict |
|---|---|---|---|
| [`kp-experiment-9a-spike-loss.ipynb`](kp-experiment-9a-spike-loss.ipynb) | 1 — spike tail | global loss/target treatments reject on WAPE | see §1 |
| [`kp-experiment-9b-conditional-lift.ipynb`](kp-experiment-9b-conditional-lift.ipynb) | 1 — spike tail | conditional lift **does not survive nested CV** (+0.018, fragile) | see §2 — **Frontier 1 closed for WAPE** |
| [`kp-experiment-9d-chronos-finetune.ipynb`](kp-experiment-9d-chronos-finetune.ipynb) | 2 — decorrelation (Chronos-2) | zero-shot 35 → fine-tuned **18.7** (2000 steps), but **r vs MLP 0.90** (correlated) | see §3 — **Frontier 2 closed** |

---

## §1. The spike problem — `kp-experiment-9a-spike-loss.ipynb`

**Hypothesis.** 36% of WAPE lives in spike rows the MAE objective under-predicts
(it finds the conditional median; the target is right-skewed). A loss/target change
that represents the tail better should reduce error there. Tested on the fast
LightGBM anchor (the cheapest testbed; ports to the MLP if it works), 3-fold
rolling-origin CV, seed-averaged over [42, 101, 202].

**Two caveats registered up front** (and confirmed by the result):
1. WAPE = scaled MAE → its per-row optimum is the **median** (q0.5), the same
   optimum MAE already targets. A higher quantile (q0.65) is therefore *not*
   theoretically WAPE-optimal; it's an empirical bet that compensates the model's
   tail under-fit.
2. The median is preserved under `log1p`, so log-MAE also targets the median in
   principle; any benefit comes from changed loss geometry + the convex `expm1`
   back-transform.

### Results — four arms vs the LGBM-MAE anchor

| arm | objective / space | CV-WAPE | Δ vs anchor | spike-WAPE | spike resid | MSE | r vs anchor |
|---|---|---|---|---|---|---|---|
| `mae` (anchor) | L1 / raw | **14.738 ± 0.62** | — | 23.52 | +5.84 | 8.854 | 1.000 |
| `logmae` | L1 / `log1p` | 14.769 | −0.031 (noise) | 23.15 | +5.71 | 8.844 | 0.998 |
| `pinball65` | quantile α=0.65 / raw | 15.015 | −0.277 | 21.65 | +5.18 | **8.630** | 0.995 |
| `logpinball65` | quantile α=0.65 / `log1p` | 15.136 | −0.398 | **21.49** | **+5.07** | 8.695 | 0.993 |

**Reading the table:**

- **The mechanism is real.** Every treatment lifts the spike rows — spike-WAPE
  drops up to −2.0 (23.52 → 21.49) and the spike under-prediction shrinks from
  +5.84 to +5.07. The diagnosis was correct.
- **But it loses on overall WAPE.** Lifting the 5% tail taxes the 95% bulk more
  than it gains, because WAPE rewards the median. `logmae` is tied within noise;
  both quantile arms regress. **No global treatment clears the 0.05 floor.** Exactly
  as the pre-registered theory predicted.
- **All arms r ≥ 0.993 vs the anchor** — same family, no decorrelation, no straight
  blend value (best 2-way blend: 14.724, +0.014, inside noise).
- **MSE is the one place a global arm helps:** `pinball65` cuts MSE −2.5%
  (8.854 → 8.630) at +0.28 WAPE. Doesn't close the Chronos L2 gap (~7.72) and costs
  WAPE, so it is **not** the WAPE submission — but it is a candidate arm for an
  *L2-targeted* submission if Overall rank is graded.

### §1.x — The conditional blend (post-hoc probe on the 9a OOF)

The global treatments fail because they shift *every* row. The natural refinement:
apply the high-quantile arm **only to rows the model already predicts high** (a
spike proxy), keep MAE elsewhere. Probed directly on `oof_predictions_9a_spike.csv`:

| recipe | CV-WAPE | Δ vs anchor |
|---|---|---|
| anchor | 14.738 | — |
| conditional `pinball65`, applied on **top 10% predicted rows** | **14.676** | **+0.062** |
| conditional `logpinball65`, applied on top 2% predicted rows | 14.685 | +0.052 |

Both clear the 0.05 floor: the bulk stays at the WAPE-optimal median, only the
suspected-spike rows get lifted.

**Honest caveat — not confirmed here.** The threshold (top-10%) was selected on the
same OOF it was scored on (in-sample over 6 candidate fractions). Confirmed in §2.

### Verdict and progression

- **Frontier 1 as a *global* loss lever: rejected** — clean, mechanism-backed
  negative (WAPE rewards the median; lifting the 5% tail taxes the 95% bulk).
- **Frontier 1 as a *conditional* lever: looked like a lead in-sample → tested in §2.**
- **L2 side-result:** keep `pinball65` on file as an MSE-arm candidate for the
  Overall-rank / Chronos question, distinct from the WAPE track.

---

## §2. Conditional lift, honestly validated — `kp-experiment-9b-conditional-lift.ipynb`

Pure post-processing analysis on the 9a OOF (no retraining, no GPU). Re-measured the
conditional spike-lift with **nested, out-of-fold threshold selection**: for each
test fold, the lift fraction and threshold are chosen on the *other* folds only.

| arm | in-sample (9a) | honest nested-CV | Δ vs anchor | chosen fraction / fold |
|---|---|---|---|---|
| anchor (`pred_mae`) | 14.738 | 14.738 | — | — |
| `pinball65` lift | 14.676 (+0.062) | **14.720** | **+0.018** | [0.02, 0.15, 0.15] |
| `logpinball65` lift | 14.685 (+0.052) | **14.733** | **+0.005** | [0.02, 0.10, 0.10] |

**Verdict — Frontier 1 closed for WAPE.** The in-sample +0.062 collapses to +0.018
out-of-fold, below the 0.05 floor, and the chosen fraction is unstable across folds
(a fragile-hyperparameter signature). The 9a in-sample number was selection optimism.
No post-hoc spike treatment — global or conditional — produces a shippable WAPE gain
on this dataset. The conclusion is **metric-level**: WAPE's per-row optimum is the
median, the spikes are 5% of rows, and point-forecast manipulation around them does
not pay. Recorded as a closed avenue.

The §4 cross-check (does the DL/MLP model under-predict the same spike rows?) remains
the one useful forward pointer: if yes, the only principled way left to touch the tail
is *inside* a model that estimates the conditional distribution, not a post-hoc patch.

---

## §3. Chronos-2 foundation model — `kp-experiment-9d-chronos-finetune.ipynb` (+ 9c zero-shot)

Frontier 2: the one lever 9a/9b didn't touch — **genuinely decorrelated signal**.
Within the tree family, residuals correlate at r = 0.9955 (exp-6); the hope was that
a pretrained foundation model, a fundamentally different inductive bias, would produce
errors uncorrelated with the MLP and thus a useful stacking arm.

### 9c — zero-shot (diagnostic, two false starts then fixed)

- **Environment:** native `chronos-forecasting` install needs care on Kaggle — the
  default install clobbered CUDA runtime libs, and a `--no-deps` workaround then left
  an incompatible `transformers`, loading the model with **random weights** (flat
  forecasts, std 0.008). Fixed by pinning `transformers>=4.41,<5` + `einops>=0.7`; a
  sine-wave sanity test confirms the weights load.
- **Result:** working zero-shot Chronos-2 (history-only) ≈ **35 WAPE** — about the
  seasonal-mean baseline. Covariates did not rescue zero-shot. Confirms the signal is
  in the covariates and zero-shot foundation models are weak at a 336h horizon.

### 9d — covariate fine-tune (native `Chronos2Pipeline.fit()`)

- **Method:** AutoGluon (the documented covariate-finetune path) proved
  **env-incompatible** with Kaggle's numpy-2.x base (needs numpy<2 → breaks
  pandas/scipy ABI). Pivoted to the **undocumented native `fit()`** in
  chronos-forecasting 2.2 — full fine-tune, per-fold rolling-origin, 11 known
  covariates, `CONTEXT_LENGTH=1024`, 500 steps, both T4s.

| metric | Chronos-2 FT (500 steps) | Chronos-2 FT (2000 steps) | gated-MLP | 14.47 ensemble |
|---|---|---|---|---|
| CV-WAPE | 19.29 | **18.71** (17.2 / 20.7 / 18.2) | 14.59 | 14.47 |
| MSE | 10.96 | 10.39 | 8.75 | — |
| r vs gated-MLP | 0.891 | **0.900** | — | — |
| best blend (MLP+Chronos) | 14.589 (+0.004, noise) | ≈14.59 (noise) | — | — |

**Verdict — Frontier 2 closed (longer-training check done).** Quadrupling steps
(500 → 2000) moved solo WAPE only 19.3 → 18.7 (training loss plateaued ~0.50) and
**raised** r from 0.891 to 0.900 — the predicted mechanism: as Chronos fits the
covariates better it converges toward the same function the MLP learned, so it gets
*more* correlated, not less. It is both worse than the MLP and correlated; the blend
hands it ~2.5% weight for a noise-level move, still below the 14.47 ensemble. No
stacking value, and more training cannot create the decorrelation it would need.

**One nugget for the writeup:** Chronos *wins lead-1h* (19.1 vs MLP 25.7) and loses
every longer-horizon bucket — a sequence/foundation model captures the immediate next
step better than the tabular MLP (echoes exp-7a's seq2seq lead-1h win). But it is 288
of 96,768 rows (0.3%), and exp-7b already showed horizon-conditional blending isn't
worth the plumbing. Recorded, not pursued.

### The headline finding of Phase 9

> Across three different model classes — tree-GBMs (r = 0.9955 between LGBM and
> XGBoost), and a pretrained foundation model fine-tuned with covariates (r = 0.891
> vs the MLP) — **every covariate model converges to the same function.** The ceiling
> on this dataset is the *feature set*, not the model class. Architectural novelty
> (TiDE, PatchTST-class, Chronos-2) does not break it.

This redirects any further work to **new information** (Frontier 4 — temporal/regime
features) or to accepting ~14.5–14.7 as the dataset's WAPE floor for this feature set.
Frontier 3 (per-series heterogeneity) remains untried; triage first (low-target
WAPE-arithmetic artifact vs genuine hardness).

---

## Files in this folder

```
experiment-9/
├── README.md                              ← this file
├── kp-experiment-9a-spike-loss.ipynb      ← Frontier 1: spike loss/target arms (scaffold)
├── experiment-9a.ipynb                     ← executed run with outputs (consider renaming to the kp- form for consistency)
├── oof_predictions_9a_spike.csv           ← OOF for all 4 arms (feeds blend / conditional probes)
└── arm_summary_9a_spike.csv               ← per-arm metric summary
```
</content>
