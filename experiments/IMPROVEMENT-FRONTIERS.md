# Improvement Frontiers — Where the Remaining Error Actually Lives

A planning document for post-Phase-8 work. After 8 experiments the pipeline
plateaued at **CV-WAPE ~14.4 (blend) / LB 14.68**, and swapping architectures
(TiDE, PatchTST-class, seq2seq LSTM) did **not** move it. This document records
*why* — the ceiling is a **feature/information ceiling and a heterogeneity
ceiling, not an architecture ceiling** — and lays out the concrete, unexplored
frontiers that target where the error is actually concentrated.

Maintain the project's existing discipline: **pre-register decision thresholds,
report per-fold breakdowns, gate ensemble arms on residual r < 0.85, accept a
gain only if it clears the ~0.05 WAPE noise floor.**

---

## The diagnostic that reframes everything

Computed on the shipped blend OOF (`experiment-7/oof_predictions_7b_blend.csv`,
96,768 rows, overall WAPE 14.42):

| Finding | Number | Implication |
|---|---|---|
| Top 5% of rows by \|error\| | **36.2%** of total WAPE numerator | Half of WAPE lives in 10% of rows (52.4%) |
| Top 1% of rows by \|error\| | 10.0% of total WAPE numerator | Extreme concentration |
| Mean residual, overall | **−0.485** (under-predict) | Systematic median-collapse |
| Mean residual, top-5% highest-target rows | **−5.60** | Model badly under-shoots spikes |
| Per-series WAPE spread | **7.1 → 25.9** (3.6×) | One global model is badly averaged |
| Worst 20 series share of total error | 27.1% | Heterogeneity is real but spread, not 5 outliers |
| Cross-series residual synchrony | **3.5%** | Global/cross-sectional features are NOT a lever |

**Two pockets hold the addressable error: the spike tail and per-series
heterogeneity.** Neither was seriously attacked across experiments 1–8.

**One negative result (saves effort):** cross-series "common-shock-at-time-t"
features explain only 3.5% of residual variance. Do **not** invest in
cross-sectional/global-panel feature engineering.

---

## Frontier 1 — The spike problem (highest leverage)

> **STATUS — CLOSED for WAPE (experiments 9a + 9b, see `experiment-9/README.md`):**
> *Global* loss/target treatments rejected (lift the 5% tail, tax the 95% bulk;
> WAPE rewards the median). The *conditional* lift looked promising in-sample
> (14.676, +0.062) but **collapsed to +0.018 under honest nested-CV** (9b), below
> the 0.05 floor and with an unstable threshold — i.e. selection optimism. No
> post-hoc spike treatment yields a shippable WAPE gain. Metric-level result, not
> model-level. **Remaining value:** `pinball65` cut MSE −2.5% (an L2/Overall-rank
> arm candidate only); and if a tail fix is ever wanted it must live *inside* a
> model that estimates the conditional distribution, not as a post-hoc patch.

**Evidence:** Top 5% of rows = 36% of the WAPE numerator, and those rows are
under-predicted by ~5.6 on average. The MAE objective finds the conditional
median; the right-skewed target means it can't reach the peaks.

**Why the prior negative result does NOT close this door:** experiment-7h found
"bias correction hurts WAPE," but that tested a *blanket additive shift* — the
wrong tool. A uniform +0.5 hurts the many already-correct rows. The fix must be
**conditional**: lift predictions only on rows headed for a spike. This pocket
helps **WAPE and the L2 metrics simultaneously** (the same big-error rows drive
both), so it also addresses the Overall-rank gap to Chronos.

**Attacks, in order of effort:**

1. **Train on `log1p(target)`.** Modeling in log space makes the loss care about
   *relative* error, which stops the median-collapse on a skewed target. Cheapest
   experiment, highest chance of a real WAPE move. Never tried in exp 1–8.
2. **Asymmetric / quantile (pinball) loss arm** at q ≈ 0.6–0.7. Penalizes
   under-prediction harder than over-prediction; the right-skew means the
   WAPE-optimal point sits above the MAE median. Blend this arm in.
3. **Two-stage spike model.** Classifier ("is this row about to spike") + a
   magnitude head, or a dedicated high-target regressor applied only where it
   predicts high. Heaviest; go here if 1–2 underdeliver.

**Pre-registered test:** WAPE on the top-5%-target subset AND overall WAPE,
per-fold. Accept if overall WAPE improves > 0.05 without the bulk regressing.

**Expected payoff:** the largest of all frontiers — 36% of the numerator is in
scope. Realistic −0.2 to −0.5 WAPE if log-target alone works; more if combined
with a spike arm. Also closes part of the L2 gap.

**First experiment: 9a** — retrain best model on `log1p(target)` + a q0.65
pinball arm, on the existing CV/OOF harness.

---

## Frontier 2 — Stack on Chronos (exploit the model beating you)

> **STATUS — CLOSED (experiments 9c + 9d, see `experiment-9/README.md` §3):**
> Zero-shot Chronos-2 ≈ 35 WAPE (signal is in covariates, weak at 336h). Native
> covariate **fine-tune** (after AutoGluon proved env-incompatible) reached **18.7**
> CV-WAPE at 2000 steps — honest but well above the MLP — with **r vs MLP = 0.90**
> (correlated). Longer training is a dead end for stacking: 500 → 2000 steps moved
> solo WAPE only 19.3 → 18.7 while r *rose* 0.891 → 0.900 (better covariate fit →
> converges toward the MLP's function). Best blend ≈ 14.59 (noise), below the 14.47
> ensemble. **Headline finding:** across tree-GBMs (r=0.9955) and a fine-tuned
> foundation model (r=0.90), all covariate models converge to the same function — the
> ceiling is the **feature set, not the model class**. Remaining levers: Frontier 4
> (new features) or accept ~14.5–14.7 as the floor. (Minor: Chronos wins lead-1h by
> 6.6 WAPE — sequence-model edge at the immediate step — but it's 0.3% of rows and
> horizon-conditional blending was already rejected in exp-7b.)

**Evidence:** Chronos-2-native-optimized leads Overall via L2 metrics (MSE +21%,
RMSE +10%, MAPE +10%) while losing WAPE (14.81 vs your 14.68). It captures
something your 52 features don't — almost certainly the extremes (Frontier 1).
Within your own family, residuals correlate at r = 0.99; a pretrained foundation
model is the one source near-guaranteed to be decorrelated (r ≪ 0.85).

**Approach:** Instead of competing with Chronos, **consume it**:
- Generate Chronos-2 OOF + validation predictions (zero-shot first; fine-tune if
  competitive).
- Add Chronos's prediction as an **input feature** to the blend (stacking), and/or
  as an extra **blend arm** with OOF-tuned weight.

**Pre-registered test:** residual r of Chronos vs your blend (expect ≪ 0.85);
ensemble WAPE + full L2 panel, gated on the 0.05 WAPE floor for the WAPE arm and
on MSE for the Overall-rank question.

**Expected payoff:** most likely thing to break the within-family saturation.
Strong candidate to win Overall rank, not just WAPE.

**Experiment: 9b.**

---

## Frontier 3 — Per-series heterogeneity (7→26 WAPE spread)

**Evidence:** per-series WAPE ranges 7.1 (unit_069) to 25.9 (unit_073), 3.6×.
Worst offenders: unit_073, _058, _066, _056, _039, _071, _003, _079, _031, _061.
The worst 20 series carry 27% of total error.

**Triage FIRST (cheap, ~20 min, do before building anything):** are the worst
series *low-mean-target* series (small WAPE denominator → inflated WAPE, largely
irreducible per the data-analysis low-target/WAPE-arithmetic blind spot) or
genuinely high-variance/hard? This determines how much of the 3.6× spread is even
addressable. Split the worst-20 into "fixable hard" vs "WAPE-arithmetic artifact."

**Attacks for the genuinely-hard series:**
- **Per-cluster models / mixture-of-experts.** Cluster the 96 units by covariate
  profile or target dynamics; train per-cluster. Un-averages the global model.
- **Native per-series embedding + per-series output scaling** (what DeepAR /
  TSMixerx do natively). Your MLP only had `series_mean` + `sid_code` — a learned
  embedding with per-series scaling is a richer mechanism.
- **Per-series residual model on features** (NOT a per-series constant — that
  failed in exp-5 §6 / exp-7h). Fit a small model on OOF residuals using the
  covariates, per cluster.

**Pre-registered test:** worst-20-series WAPE improves while the other 76 do not
regress; overall WAPE clears 0.05.

**Expected payoff:** moderate-to-high if much of the spread is genuine; capped if
triage shows it's mostly low-denominator artifact. **Run triage before committing.**

**Experiment: 9c (triage) → 9d (modeling, conditional on triage).**

---

## Frontier 4 — New temporal / regime features (the ceiling-breaker)

**Evidence:** within-family saturation (r = 0.99) is *feature-bound* — new
architectures on the fixed 52-feature set converge to the same function. The way
past it is new *inputs*. But per the 3.5% synchrony result, the right inputs are
temporal/interaction, NOT cross-sectional.

**Candidate features (none tried in exp 1–8 beyond lag/lead [1,24] + simple rolling):**
- **Spike-predictive interactions** — e.g. `shock_risk × queue_pressure`, to give
  the model a handle on Frontier 1's rows.
- **Covariate velocity / acceleration** — first and second diffs of top covariates;
  "hours since last covariate spike."
- **Weekly lag-168 + EWMA** of top covariates (only lag/lead [1,24] tried so far).
- **Regime / changepoint indicators** — likely the cause of the universally-hard
  fold 1 (days 153–166); a regime flag could lift that whole window.

**Pre-registered test:** add to the best model, 3-fold CV, accept > 0.05 WAPE.
Watch fold-1 specifically for the regime features.

**Expected payoff:** uncertain but this is the only path that adds genuinely new
information rather than re-modeling existing signal. Pairs naturally with
Frontier 1 (the interaction/velocity features target spikes).

**Experiment: 9e.**

---

## The fold-1 thread (cross-cutting, flagged repeatedly, never executed)

Every model scores ~0.8 WAPE worse on fold 1 (days 153–166). The per-series view
shows error is spread, not concentrated in a few series — pointing to a
**temporal/regime** cause, not a per-series one. Best addressed via Frontier 4's
regime/changepoint indicators. Add a per-(fold × series) WAPE table to localize
whether fold-1 excess is uniform or spiky.

---

## Recommended experiment order

| # | Experiment | Frontier | Status / Effort | Why this order |
|---|---|---|---|---|
| 9a | `log1p` + q0.65 pinball arms (global) | 1 (spikes) | **DONE** — global rejected; conditional lead | Biggest pocket; cheap test; mechanism confirmed |
| 9b | Conditional spike blend, nested threshold | 1 (spikes) | **NEXT** / Low | Confirm the +0.062 in-sample lead out-of-fold before shipping |
| 9c | Chronos-2 stacking / blend arm | 2 (decorrelation) | Medium | Most likely to break r=0.99 saturation; Overall-rank play |
| 9d | Worst-series triage (artifact vs genuine) | 3 (heterogeneity) | Low | 20-min gate before any per-series modeling |
| 9e | Per-cluster / per-series embedding model | 3 | Medium-High | Conditional on 9d showing genuine hardness |
| 9f | Spike-interaction + velocity + regime features | 4 + fold-1 | Medium | Only path adding new information; targets spikes + fold-1 |

**Stop rule (unchanged):** stop when the next expected improvement is smaller than
the CV standard deviation (~0.6), or when a frontier's pre-registered test fails.

---

## What NOT to spend time on (evidenced dead ends)

- **More architectures on the fixed feature set** — TiDE (16.37), seq2seq LSTM
  (16.30), tree-GBM diversity (r=0.9955) all confirm the signal is feature-bound,
  not architecture-bound.
- **Cross-sectional / global-panel features** — 3.5% residual synchrony.
- **Blanket bias correction** (additive shift, global or per-series) — failed in
  exp-5 §6 and exp-7h; the spike fix must be conditional (Frontier 1), not a shift.
</content>
</invoke>
