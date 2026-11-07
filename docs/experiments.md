# Experiment log

Thirteen experiments, May to August 2026. Every score is WAPE, lower is better. The notebooks
and notes behind each one are in [`experiments/`](../experiments/).

- **CV**: 3-fold rolling-origin cross-validation on the last 3 × 336 hours of `train.csv`.
- **LB**: the course's public validation leaderboard, scored against hidden labels.

Reference points: naive last value LB 48.10, seasonal mean LB 34.41, best entry on the
leaderboard (another team) 13.05.

## The arc in five phases

### 1. Sequence models that never beat the baseline (experiments 1 to 3.1)

The task looked like classic time-series forecasting, so the first model was an LSTM that
predicted 24 hours at a time and was rolled forward 14 times, feeding its own predictions back
in as lagged targets.

- **Experiment 1** scored LB 37.16, worse than the seasonal-mean baseline (34.41), while
  looking competitive locally. The local check was wrong: `random_split` over sliding windows
  mixed past and future windows between train and validation.
- **Experiment 2** fixed the split (chronological). Training loss fell, validation loss rose
  from the first epoch. Regularisation changed nothing.
- **Experiment 3.1** added the domain covariates and an L1 loss. Same divergence.

The diagnostic that ended this phase: a plain linear regression on the covariates, no
sequence model at all, scored WAPE 21 on the same held-out split, beating every LSTM variant
and the seasonal mean (31).

### 2. The reframe: direct regression on known covariates (experiments 3.2 and 4)

`validation_input.csv` contains every covariate for the 336 forecast hours and no target. So
target lags do not exist at prediction time, and an autoregressive rollout has to invent
them. The covariates for the forecast block, on the other hand, are given. The target turned
out to be largely a per-hour function of them (`queue_pressure_forecast` alone explains ~47%
of its variance out of sample).

| Step | CV WAPE |
|---|---|
| LightGBM, L2 loss, single held-out split | 16.53 |
| Same model under honest 3-fold CV | 17.27 ± 1.22 |
| + MAE objective (matches WAPE) + `series_id` as a categorical | 15.39 ± 0.59 |
| + covariate lags, leads and rolling statistics | **14.80 ± 0.64** |

Swapping L2 for MAE alone was worth 1.08. On the leaderboard this model scored 15.48.

### 3. Squeezing the trees (experiments 5 and 6)

Tuning, per-series bias correction, multi-objective ensembles and seed averaging, each behind
a pre-registered accept rule (for example, accept only if the gain beats half a CV standard
deviation). CV 14.73, LB 14.92.

XGBoost was tried as a diversity arm and rejected: its out-of-fold residuals correlated with
LightGBM's at **r = 0.9955**. Two boosters on the same features were the same function.

### 4. Neural networks on the same inputs (experiments 7 to 10)

Reframed again as sequence-to-sequence: a window of covariates maps to a window of targets.

| Model | CV | LB | Verdict |
|---|---|---|---|
| LightGBM + gated MLP blend (7c) | 14.47 | **14.68** | our best entry until August |
| Gated MLP alone (7f) | 14.59 | 14.75 | |
| LSTM seq2seq (7a) | | 16.30 | |
| TiDE (8) | 16.37 | | rejected |
| Spike-weighted losses (9a, 9b) | | | rejected, gains did not survive nested CV |
| Chronos-2, fine-tuned (9d) | 18.7 | | rejected; residual r = 0.90 with the MLP |
| Extra feature blocks (10a) | | | rejected, best block gained 0.038, under the 0.05 floor |

Every model on these inputs converged to nearly the same predictions. The working conclusion
was a feature-set ceiling.

### 5. Encoder-decoders and the Temporal Fusion Transformer (experiments 11 to 13)

| Model | CV | LB |
|---|---|---|
| BiGRU encoder-decoder with cross-attention, 5 seeds (11a) | 14.20 ± 0.60 | 14.77 |
| BiLSTM version (11b) | 14.41 ± 0.61 | |
| Temporal Fusion Transformer, 1 seed (12a) | 14.72 ± 0.34 | **14.358** |
| Same TFT, 5 seeds averaged (13a, the shipped model) | | 14.479 |

The single-seed 14.358 is the best score of the project, but that run did not save its
weights. The shipped model is the 5-seed ensemble, whose checkpoints are the ones in this
repository's release.

## What the project taught about evaluation

### The CV protocol trap

| Model | Early-stops on | CV | LB | CV to LB |
|---|---|---|---|---|
| BiGRU (11) | the fold block it then reports | 14.20 | 14.77 | +0.57 |
| TFT (12) | the 336 h before that block | 14.72 | 14.36 | -0.36 |

Selecting the epoch on the block being scored made the GRU's CV optimistic. The TFT used the
honest protocol and also lost 336 hours of training data per fold, which made its CV
pessimistic. CV ranked the TFT last; the leaderboard ranked it first. It was rejected at
first on a comparison that the experiment's own notes flagged as invalid. A blend fitted on
the mixed-protocol out-of-fold predictions gave the TFT weight 0.0 for the same reason.

Rule since then: never compare CV numbers produced under different early-stopping protocols.
`src/loadforecast/cv.py` only offers the honest one for early-stopping models.

### Training windows must reach the forecast origin

Experiment 11's final fit reused a dataset cut at fold 0's boundary, so the model never
trained on windows near the end of the history and read an unfamiliar encoder region at
prediction time. Fixing only that moved the leaderboard from **17.91 to 14.77**. A later
version of the window filter anchored the stride grid at time 0, which drops the last window
whenever the stride does not divide the last start; a 5-minute end-to-end run with a
different stride caught it. Both are now assertions in `training_dataset` and a test.

### Seed noise is as large as the effects being measured

| Difference | Mean absolute difference in predictions |
|---|---|
| Same seed, same config, trained twice (GPU non-determinism) | 0.403 |
| 1 seed versus the mean of 5 seeds | 0.375 |
| Mean versus median of the 5 seeds | 0.103 |

So the +0.12 WAPE between the single-seed and 5-seed submissions cannot be attributed to
seed averaging. The project's decision threshold was 0.05 WAPE; a single leaderboard
submission cannot resolve that. Measuring the TFT's seed-to-seed spread on one fold
(about 2.2 GPU hours) was the next planned step and was not run before the deadline.

### Where the remaining error sits

- In the blend's out-of-fold errors, the top 5% of rows by absolute error carry 36% of the
  WAPE numerator: load spikes that
  MAE-trained models under-predict, because MAE targets the conditional median.
- Per-series WAPE ranges from about 7 to 26.
- The TFT's variable-selection weights put `network_pressure_forecast` (0.173) and
  `queue_pressure_forecast` (0.147) first and the target history at only 3.1%: it does use
  the covariates, and still converges to the GRU's function (residual r = 0.973).

## If I started again

1. Read what the inference inputs contain before choosing a model family.
2. Fit a linear model on the obvious features first; it answers whether the signal exists.
3. Cross-validate from the start, with the scored block untouched by model selection.
4. Match the loss to the metric.
5. Break residuals down by series, hour and regime before tuning anything.
6. Measure seed noise before running ablations smaller than it.
