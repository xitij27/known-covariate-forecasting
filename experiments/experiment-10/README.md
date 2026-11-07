# Phase 10 — Feature engineering: the "new information" lever

Phase 9 closed with the headline finding that across tree-GBMs (r = 0.9955) and a
covariate-fine-tuned foundation model (r = 0.90), **every covariate model
converges to the same function — the ceiling is the feature set, not the model
class.** This phase acts on that: the only lever left that can move the floor is
**new information in the features**.

Guiding principle (from the convergence finding): a model that represents linear
combinations (GBM, MLP) gains ~nothing from a *linear* recombination of correlated
inputs — it already computes those internally. So the candidate blocks are ordered
by how much **genuinely new** information they add, not by intuition.

| Notebook | What | Result | Verdict |
|---|---|---|---|
| [`kp-experiment-10a-feature-engineering.ipynb`](kp-experiment-10a-feature-engineering.ipynb) | Additive ablation of 3 FE blocks on the LGBM testbed | only block C helps (+0.038), under the 0.05 floor | see §1 |
| [`kp-experiment-10b-dynamics-submit.ipynb`](kp-experiment-10b-dynamics-submit.ipynb) | Expanded block C on the LGBM arm + blend with saved MLP | results not persisted | see §2 |
| [`kp-experiment-10c-mlp-dynamics-submit.ipynb`](kp-experiment-10c-mlp-dynamics-submit.ipynb) | Block C **inside** the gated MLP (dominant arm) | results not persisted | see §2 |

---

## §1. Additive ablation — `kp-experiment-10a-feature-engineering.ipynb`

Three feature blocks tested additively on the LGBM-MAE testbed (fast, CPU; ports
to the MLP if a block wins), 3-fold rolling-origin CV, each gated against the
**0.05 WAPE floor** vs the 52-feature baseline:

- **A — stress factor:** PLS (supervised) + PCA composites of the 5 collinear
  covariates. Fit per-fold (PLS uses the target).
- **B — interactions:** per-row products/ratios (causal, pointwise).
- **C — dynamics:** backward `shift`/`diff` velocities (causal).

### Results (`ablation_10a_features.csv`)

| config | fold0 | fold1 | fold2 | mean | Δ vs base |
|---|---|---|---|---|---|
| baseline (52 feat) | 14.045 | 15.549 | 14.619 | **14.738** | — |
| + A stress factor | 14.066 | 15.576 | 14.744 | 14.795 | **−0.057** |
| + B interactions | 14.072 | 15.567 | 14.667 | 14.769 | **−0.031** |
| + C dynamics | 14.010 | 15.533 | 14.557 | **14.700** | **+0.038** |
| + B + C | 14.021 | 15.531 | 14.616 | 14.723 | +0.015 |
| + all (A+B+C) | 14.045 | 15.512 | 14.691 | 14.749 | −0.011 |

**Reading the table.** Exactly as the linear-recombination argument predicted: the
linear blocks **A and B regress** (they add nothing the model didn't already
compute, only noise/overfit surface). **Only block C (temporal dynamics) helps,
and it helps on all three folds** — but at +0.038 it sits just under the 0.05
noise floor. It is the single most promising feature direction found, so it gets a
fair, expanded retry in 10b/10c.

---

## §2. Giving block C a fair chance — `10b` (LGBM) and `10c` (MLP)

- **10b** expands block C (more velocities, acceleration, multi-span EWMA, regime +
  spike flags), re-checks it on CV, then builds a leaderboard submission: C only
  moves the LGBM arm, so it is put on the LGBM arm and blended with the
  already-saved gated-MLP validation predictions (the same 14.68 ensemble, now with
  a C-augmented LGBM arm). CPU-only, no MLP retrain. Pre-registered expectation: C's
  gain is diluted by the LGBM arm's ~0.285 blend weight, so the ensemble likely
  lands ~14.66–14.68.
- **10c** puts expanded block C **inside the gated MLP** — the 71.5% arm and best
  pure-DL model (14.59 CV / 14.75 LB), where it can actually matter. Architecture,
  gates, and training are verbatim from 7e; the only change is the added block-C
  features (the per-covariate gate auto-attaches each C feature to its parent by
  name prefix). Run as a paired ablation (no-C vs +C, same folds/seeds), then a
  full-train submission. GPU required.

> **Status / honesty note.** 10b and 10c are scaffolds whose **executed results
> were not persisted** — the notebooks were saved with outputs cleared and no
> submission/validation CSVs were written to this folder. The recorded conclusion
> of the phase therefore rests on the 10a ablation (block C is the only live
> feature lever, and it is sub-floor on the LGBM testbed). The 10b/10c numbers, if
> rerun, belong here.

---

## Files

```
experiment-10/
├── README.md                                       ← this file
├── kp-experiment-10a-feature-engineering.ipynb     ← 3-block additive ablation (has outputs)
├── kp-experiment-10b-dynamics-submit.ipynb         ← expanded-C LGBM arm + blend (outputs cleared)
├── kp-experiment-10c-mlp-dynamics-submit.ipynb     ← block C inside the gated MLP (outputs cleared)
└── ablation_10a_features.csv                        ← per-fold ablation results (§1 table)
```
