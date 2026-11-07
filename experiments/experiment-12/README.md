# Experiment 12 — Temporal Fusion Transformer

## Status

| Sub-experiment | Status | CV WAPE | LB WAPE | Verdict |
|---|---|---|---|---|
| 12a — TFT | Complete | 14.722 ± 0.339 | **14.358 — best of any entry (rank 57)** | **ACCEPTED** |
| 12b — Blend | Complete | 13.988 (OOF) | not submitted | weights biased — fit on mixed-protocol OOF, gave TFT 0.0 |

> **CORRECTION (2026-08-17).** This README originally recorded TFT as rejected on a CV
> comparison against exp-11's 14.203. That comparison was invalid — the two used different
> early-stopping protocols, which the README itself flagged. The leaderboard inverted the
> ordering: TFT **14.358** vs GRU **14.773**, and TFT has the best MAE (1.5805) of any of our
> submissions. The blend-arm and spike-tail gates below were computed from the same
> mixed-protocol OOF and are equally suspect.

**Headline:** TFT is rejected. It is the strongest evidence yet that the ceiling is in the
feature set, not the model — because unlike every earlier candidate, TFT demonstrably *does*
exploit the known future covariates (`target_history` = 3.1% of encoder importance, against
the 42–49% that got TiDE rejected) and still converges to the same function
(r = 0.973 vs the GRU).

**Actionable byproduct:** §12 produced `validation_predictions_12b_blend.csv` — a GRU + 7c
blend at **13.988 OOF**, the best number in the project, and the exp-11c blend that had been
pending since June. Cross-fitted weights give 13.989, so it is not overfit. Submit it.

Reference points: exp-11a GRU **14.203 ± 0.599** CV / **14.773** LB · exp-7c blend 14.47 CV /
**14.682** LB (our best, rank 27) · leaderboard best 13.053.

---

## Results

### Verdict: rejected on all three pre-registered gates

| gate | threshold | result | |
|---|---|---|---|
| ship solo | beat 14.203 by > 0.05 | **14.722 ± 0.339** | fail |
| ship as blend arm | residual r < 0.85 | **r = 0.9731** vs GRU | fail (saturated) |
| fix the spike tail | improve on −5.60 | **−5.68** | fail |

§12 settled the blend question beyond argument: the simplex optimiser assigned TFT **exactly
0.0 weight**. Its per-series error profile is also near-identical to the GRU's — worst series
`unit_073` at 25.42 (GRU 25.47), best `unit_069` at 7.31 (GRU 7.01).

Fold breakdown: 14.569 / 15.110 / 14.486. Fold 1 hardest, as for every model in this project.
CV std **0.339** — the most fold-stable model here, against the GRU's 0.599.

**Protocol caveat, stated for honesty not as an excuse:** `EARLY_STOP_MODE="inner"` costs TFT
336 h of training data per fold and never selects on the block it is scored on. exp-11's
14.203 does both the other way. The two numbers are not like-for-like and the gap between the
protocols was never priced. It would not change the verdict — r = 0.973 is protocol-independent.

### §7 probe: no structural choice mattered

| arm | WAPE | best_epoch | windows |
|---|---|---|---|
| crop | 17.047 | 10 | 9,984 |
| base | 17.243 | 8 | 9,312 |
| noncausal | 17.384 | 6 | 9,312 |
| quantile | 17.661 | 7 | 9,984 |

Winner over runner-up: 0.196, inside the ~0.3 a single window resolves. Two specific results:

- **Non-causal attention buys nothing.** `noncausal` is *worse* than `base` (17.384 vs 17.243)
  on an identical fixed encoder. This was the single most novel argument for TFT here — that
  decoder step *t* could see a covariate spike at *t+5*. Not supported.
- **The quantile head hurts the point forecast** (+0.418 vs crop). It was the only mechanism
  `IMPROVEMENT-FRONTIERS.md` left open for the spike tail.

The `windows` column is the check that this comparison is trustworthy: 9,984 vs 9,312 is a 7%
spread. In run 1, before the index-dedupe fix, the same comparison was **12× apart**.

### §10 variable selection — the one result worth keeping

`target_history` is **3.1%** of encoder importance (TiDE, rejected in exp-8: 42–49%). Encoder
importance is led by `network_pressure_forecast` (0.173) and `queue_pressure_forecast` (0.147),
matching the r² of 43%/47% from the EDA. Static: `series_id` 0.582, `target_center` 0.214.

So the failure is **not** that the model ignores the covariates. It uses exactly the ones the
EDA identifies, in the right order, and still lands at 14.7. That is a much sharper statement
of the feature ceiling than "another architecture also scored ~14.5".

### The 25–72h error bump is a WAPE artifact, not a pocket

Every model shows error peaking at lead 25–72h then recovering (gru 12.74 → 16.28 → 13.43).
Non-monotonic, and it looked like the last addressable pocket. It is not:

- All three fold test blocks start on a **Friday**, and folds are 336 h = **exactly 14 days**
  apart — so `lead_h` and day-of-week are perfectly confounded. The 25–72h bucket *is*
  Saturday+Sunday, 100% of it.
- **MAE is flat** across buckets: 1.403 / 1.472 / 1.454 / 1.536. The numerator barely moves.
- **`mean_target` collapses at the weekend**: 11.015 → 9.039 → 10.821.

Saturday has MAE 1.470 and WAPE 16.14; Monday has MAE 1.472 and WAPE 12.87. Identical absolute
error, 3.3 WAPE apart — purely the denominator. This is the small-denominator problem
`APPROACH.md` §1.2 already warns about, and it does **not** distort the global metric (which
pools numerator and denominator); it only distorts per-bucket breakdowns.

An earlier estimate of "~0.28 WAPE addressable here" was wrong: it assumed the bucket could be
brought down to its neighbours' level, but on MAE it is already there. **Frontier closed, no
experiment warranted.** Per-bucket diagnostics on this dataset should use MAE, not WAPE.

### Blending is also exhausted

Nine distinct model predictions exist across the repo. A simplex over all of them:

| blend | in-sample | cross-fitted (honest) |
|---|---|---|
| gru + 7c | 13.988 | **13.989** |
| all 9 arms | 13.983 | **14.012** |

Adding seven arms gains 0.005 in-sample — below the 0.05 floor — and *loses* out-of-sample.
The 2-arm blend is optimal and its weights are stable.

---

## Why TFT, given that TiDE was rejected

Experiment 8 rejected TiDE and `IMPROVEMENT-FRONTIERS.md` concluded the ceiling is
"feature-bound, not architecture-bound" (residual r = 0.99 across tree-GBMs, r = 0.90 for a
fine-tuned Chronos-2). Experiment 11 then beat everything with a GRU + attention +
augmentation, which shows that conclusion was at least partly premature — as its own README
says. So "another architecture" is not by itself an argument.

Three things here are **not** just architecture-swapping. They are the reason to run this.

### 1. The training objective finally matches the metric

exp-11 minimises `F.l1_loss(pred, tgt)` where `tgt` is **per-series z-scored**
(`kp-experiment-11a-gru.ipynb`, `train_one_epoch`). That objective weights every series
equally. WAPE does not — it is a single global ratio, `Σ|err| / Σ|actual|`, so a series with
twice the mean target contributes twice the weight.

pytorch-forecasting rescales predictions to the original scale *before* applying the loss,
so `MAE()` is exactly proportional to WAPE. This was verified against the library source and
then empirically on data (`§5` of the notebook asserts it at runtime — if it ever stops
holding, the notebook stops).

This is a mechanism, not a hope: every model in this project so far has optimised something
slightly different from the metric it is scored on.

### 2. Non-causal decoder attention

`causal_attention=False`. Every future covariate is known for the whole 336h window, so
letting decoder step *t* attend to steps *t+1 … 336* leaks **no target information** — the
decoder inputs contain covariates only. It lets the model see an incoming covariate spike
*before* it arrives.

exp-11's decoder is a unidirectional GRU whose cross-attention queries the *encoder* only.
It structurally cannot do this. Given that 36% of the WAPE numerator sits in the top 5% of
rows and the model under-shoots spikes by ~5.6 on average, "see the spike coming" is aimed
straight at the largest identified error pocket.

### 3. It answers the exp-8 diagnostic natively

TiDE was rejected partly because its feature importances showed `target_history` dominating
at 0.42–0.49 — it leaned autoregressive instead of using the future covariates that are the
whole point of this dataset. TFT reports encoder and decoder variable-selection weights
separately (§10). Whatever the WAPE outcome, this produces a direct, quotable answer to the
question exp-8 raised, which is report material either way.

The augmentation that drove exp-11's single largest gain (−1.53 WAPE on fold 0) is
reproduced via `min_encoder_length=168, max_encoder_length=336, randomize_length=(0.2, 0.05)`
— verified to crop the encoder while holding the decoder fixed at the full 336h horizon.

---

## Pre-registered decision rules

Registered before the first run, per project discipline.

| Question | Test | Threshold |
|---|---|---|
| Does TFT ship as a solo model? | 3-fold CV WAPE vs exp-11a GRU | beat **14.203** by > 0.05 |
| Does TFT ship as a blend arm? | residual `r` vs 11a GRU OOF | `r < 0.85` strong, `0.85–0.95` usable, `> 0.95` saturated |
| Does the blend ship? | OOF-optimised blend WAPE | beat **14.47** (exp-7c OOF) by > 0.05 |
| Does non-causal attention help? | §7 probe `noncausal` vs `base` | directional only; single window resolves ≳ 0.3 |
| Does the distribution head help the tail? | §9 top-5%-target WAPE + mean residual | improve on mean residual −5.60 without the bulk regressing |

**Noise floor: 0.05 WAPE.** CV std across folds is ~0.6, so a single-fold difference below
~0.3 is not evidence. Stop rule unchanged: stop when the next expected improvement is
smaller than the CV standard deviation, or when a pre-registered test fails.

---

## Protocol differences from exp-11 — read before comparing numbers

Fold boundaries are identical (train `[0..3312)` / `[0..3648)` / `[0..3984)`, each with a
336h test block), **but early stopping is not.**

exp-11 early-stops on the fold test block itself, then reports that block's WAPE as the CV
score. That is selection on the evaluation set, and it is part of why its CV–LB gap is 0.570.

`EARLY_STOP_MODE="inner"` (the default here) holds out the 336h *before* each test block for
early stopping, so nothing in the test block influences model selection. This is honest, and
it costs 336h of training data per fold — **it will read as a worse number for the same model
quality.**

| mode | early stops on | training data | comparability |
|---|---|---|---|
| `inner` (default) | `[end-336, end)` | `[0, end-336)` | honest; conservative vs exp-11 |
| `fold` | the fold test block | `[0, end)` | like-for-like vs 14.203; optimistic |

**Recommendation:** run `inner` and treat the leaderboard as the arbiter. If a head-to-head
number against 14.203 is needed for the writeup, run one fold in `fold` mode and report both,
labelled.

---

## Running it: Kaggle, T4 ×2

Set the accelerator to **`GPU T4 ×2`**.

### Step 1: `PREFLIGHT = True` (do this first)

The default in §0. It runs **every stage end-to-end** — dispatcher, per-GPU pinning, fp16 +
`mask_bias`, peak memory, parquet/npz round-trips, OOF assembly, submission asserts, blend,
§13 tuning — at the **real architecture, encoder length, batch size and precision**.

What it cuts, and why each matters for wall clock:

| Cut | Effect |
|---|---|
| **16 of 96 series** | The big one. `TimeSeriesDataSet` enumerates *every* window before filtering — 318,048 rows per fold at 96 series (measured in run 1) — and that index is rebuilt in each of the ~12 worker processes. Per-step GPU cost and memory depend on batch × sequence length, **not** series count, so fp16 and peak-memory numbers stay representative. |
| 1 epoch, stride 168 | Training becomes a handful of steps per job |
| 1 seed | 3 CV jobs instead of 9 |
| §13 trials cut to 2 | Enough to exercise the ask/tell loop and prove both GPUs are fed |

Even so, expect **~8–15 minutes**, not five: ~12 jobs each pay process startup (importing
torch + lightning is ~20 s) plus dataset-index construction, and the initial
`pip install pytorch-forecasting` costs a couple of minutes on top. The floor here is
per-job overhead, not training.

§6 still prints a **valid budget projection for the real run** — `ms/step` is measured at the
real architecture so it transfers, and the real window count is recomputed analytically from
the pre-override stride and full series count.

Outputs use `..._preflight` names so a toy run can never overwrite real results, and its WAPE
numbers are meaningless by design. If it completes, set `PREFLIGHT = False` and run for real.

### Step 2: the real run

The workload is 13 independent runs (4 probe arms +
3 folds × 3 seeds), which is embarrassingly parallel, so the notebook keeps a **job queue and
runs one job per GPU concurrently** as separate processes with `CUDA_VISIBLE_DEVICES` pinned.
A GPU picks up the next queued job the moment its current one exits.

Why not `Trainer(devices=2, strategy="ddp_notebook")`:

- DDP in an interactive kernel is fork-based and fragile;
- `predict()` under DDP shards and re-orders outputs — exactly the thing that must stay
  aligned to build OOF frames;
- at ~300 steps/epoch, gradient-sync overhead eats much of the gain;
- process isolation gives every run a clean CUDA context (no cross-run fragmentation, no
  leaked Lightning state), and a crashed run takes down one job, not the kernel.

Expect ~1.7–1.9× rather than a clean 2×: two jobs share ~4 vCPU, so `N_WORKERS_PER_JOB = 1`
keeps the dataloaders from starving the GPUs.

**Where a GPU still idles.** Process-parallelism can only fill both cards when there are ≥2
jobs queued, so the tail of each stage is single-GPU by nature: §8 has 9 jobs (5 waves, the
last one alone), §11 has 3. That is a few minutes of idle, not worth engineering around.
Two cases *were* worth fixing:

- **§10 (interpretation)** used to train its own model — a full extra run, single-GPU, while
  the other card sat idle. It is an exact duplicate of the last fold's first seed, so §8 now
  keeps that one checkpoint and §10 loads it. Saves a whole training run (~0.5–1 h). It falls
  back to training if the checkpoint is missing, e.g. when §8 was skipped.
- **§13 (tuning)** used to call `optimize_hyperparameters`, whose study loop is sequential
  and single-GPU. It now drives Optuna in ask/tell mode and dispatches each round of proposals
  across both cards, so no GPU idles there either.

### Budget — measured, not estimated

Run 1 (a full preflight on Kaggle, 2× T4) gave the real numbers, and they were far worse than
my CPU extrapolation:

| | predicted | **measured** |
|---|---|---|
| ms/step @ batch 32 | 88–184 | **970** |
| params | ~825k | **963,376** |
| windows per fold | 221,280 | **318,048** |

The extrapolation was wrong because TFT does many small matmuls across 672 timesteps — it is
launch/memory-bound, not FLOP-bound, so a T4 gave only ~2.3× over 8 CPU threads. §6 now
sweeps `BATCH_PROBE = [32, 64, 128]` and reports **ms/sample and peak GPU memory** for each,
because a larger batch is often nearly free per sample for this shape.

Run 1 also exposed the cost driver. With `min_encoder_length < max_encoder_length`,
pytorch-forecasting emits **one index row per allowed encoder length** — measured at ~12 rows
per decoder start. Filtering on `time_idx_first_prediction` kept every variant, so the dataset
was ~12× larger than the stride implies, and arms with different `min_encoder` were not
compute-comparable. `build_train_ds` now keeps **one row per (series, decoder start)** and
lets `randomize_length` do the cropping — verified to still sample encoder lengths
`[336, 168, 312, 336, 171, …]`, so the augmentation is untouched.

After that fix, at 30 ms/sample and stride 24 (~10–13k windows/fold):

| | per run | 9 CV runs | wall clock on 2 GPUs |
|---|---|---|---|
| 60 epochs | ~5.7 h | ~51 GPU-h | ~26 h |
| 30 epochs | ~2.9 h | ~26 GPU-h | ~13 h |
| 30 epochs, `SEEDS=[42]` | ~2.9 h | **~8.6 GPU-h** | **~4.4 h** |

Kaggle allows 30 GPU-h/week and caps a session at 12 h. **60 epochs × 3 seeds does not fit.**
§6 now computes total GPU-hours and prints a hard STOP with concrete cuts when the projection
exceeds quota. Recommended first real run: take §6's fastest batch size, `MAX_EPOCHS = 30`,
`SEEDS = [42]`; add seeds only for a config that already looks like a winner.

### Prior-experiment CSVs

The five files §9 and §12 need are staged in **`experiment-12/kaggle_priors/`** (31 MB) —
upload that folder as a single Kaggle dataset. **The dataset name does not matter**: the
notebook resolves them by filename anywhere under `/kaggle/input`, because hardcoded
`/kaggle/input/exp11/...` paths silently skipped the gate in run 1.

Note `oof_predictions_7b_blend.csv` carries `pred_lgbm`, `pred_mlp`, `pred_seq` **and**
`pred_blend`. Taking the first `pred*` column compares against a *component* rather than the
shipped blend, so `pick_pred_col` prefers `pred_blend` and refuses to guess when ambiguous.

Hyperparameter tuning (§13) is **off by default** and is not where the leverage is: the §7
arms test structural choices worth ~0.3–1.5 WAPE each, while `hidden_size` 64 vs 96 is worth
a fraction of that. Tune only if the structural probe has converged and quota remains.

§13 drives Optuna in **ask-and-tell** mode: the TPE sampler proposes `N_GPUS` configs at a
time from the notebook (cheap, CPU-side), the §5 dispatcher trains them **concurrently, one
per GPU**, and the WAPEs are told back to the sampler. TPE's sample efficiency, both cards
busy, scored on the real metric. `TUNE_SAMPLER = "random"` drops the optuna dependency.

It deliberately does not use `pytorch_forecasting`'s `optimize_hyperparameters`, which an
earlier version of this notebook did:

| | `optimize_hyperparameters` | §13 as written |
|---|---|---|
| GPUs | one — the study loop is sequential | **both**, via the dispatcher |
| objective | `val_loss` | **fold WAPE**, computed exactly as §7/§8 do |
| training path | its own internal Trainer | the same worker used everywhere else |
| extra installs | optuna + optuna-integration + statsmodels + tensorboardX | optuna only |
| silent-default traps | `loss`, `causal_attention` and `mask_bias` must each be hand-forwarded through `**kwargs` or it tunes a different model than the one you configured | none — config goes through `make_job` |

That last row is not hypothetical: not forwarding them meant tuning a QuantileLoss, causal
model with a `-1e9` mask bias under fp16, while the notebook was configured for MAE,
non-causal.

Either way, a single 32k-row window resolves ≳0.3 WAPE; differences below that are noise.
Take the region of the space that does well, not the single best row, and confirm any change
on the full CV in §8 before adopting it.

---

## Configuration — and where it differs from the proposed table

| Parameter | Proposed | Used | Why |
|---|---|---|---|
| Prediction length | 336 | 336 | fixed by the task |
| Encoder length | 336 | **168–336, randomised** | reproduces exp-11's crop augmentation, its largest single gain |
| Hidden size | 96 | 96 | 825k params — same order as exp-11's GRU (623k) |
| Hidden continuous size | 32 | 32 | — |
| Attention heads | 4 | 4 | 96/4 = 24 per head |
| LSTM layers | 1 | 1 | 2 only if underfitting shows |
| Dropout | 0.15 | 0.15 | — |
| Causal attention | False | **False — and probed against True** | the single most consequential flag here (§7) |
| Share variable networks | False | False | — |
| Loss | P50 / MAE | **MAE primary, QuantileLoss probed** | MAE is exactly WAPE-aligned once rescaled |
| Optimizer | Adam → AdamW | AdamW | `weight_decay` is a top-level kwarg, **not** `optimizer_params` — the latter raises `TypeError` |
| Learning rate | 1e-3 | 1e-3 | LR finder skipped: it is known to over-suggest on TFT, and 1e-3 is the library default |
| Weight decay | 0 | 1e-4 | — |
| Gradient clipping | 0.5 | 0.5 | — |
| Batch size | 32 | 32 | 672-step sequences; larger risks OOM on a T4 |
| Gradient accumulation | 4 | 4 | effective batch 128 |
| Epochs | max 80 | **60** | budget; §6 tells you if even this is too many |
| Early stopping patience | 12 | 10 | budget |
| Training stride | 24h | 24h | correct, and load-bearing — see above |
| Precision | bf16/fp16 | auto-detected | **T4 is Turing: no bf16.** Falls back to `16-mixed` |
| Seeds | 3 (5 final) | 3 | 5 only if the CV justifies the quota |

### Two landmines found while verifying

1. **`mask_bias` must be `-inf` under mixed precision.** TFT's attention mask bias defaults to
   `-1e9`, which overflows in fp16 and produces NaN losses. The library docstring says to set
   `-float("inf")`; the notebook does this automatically whenever precision is not `32-true`.
   On a T4 this is not optional.
2. **`weight_decay` goes at the top level**, not in `optimizer_params` — pytorch-forecasting
   already forwards `self.hparams.weight_decay` to AdamW, so passing it both ways raises
   `TypeError: got multiple values for keyword argument 'weight_decay'`.

---

## What run 1 confirmed on real hardware

A full preflight ran end-to-end on 2× T4 with **no errors**, which validates everything the
local tests structurally cannot reach:

- **fp16 works** — the `mask_bias = -inf` fix held; no NaN losses at `16-mixed` on Turing.
- **Both premises hold on real data**: loss on the ORIGINAL scale (2.0242 reported vs 2.0242
  original vs 0.3877 normalised), and `causal_attention` changes the forward pass (Δ 0.0286).
- **Dual-GPU dispatch works**: jobs pinned per card, queue drains correctly, no failures.
- **exp-11 Bug 1 guard passed**: last training target 4319, encoder gap exactly 336 h.
- **Submission is sane**: 32,256 rows, mean 10.39 vs training mean 9.91.
- **The headline diagnostic already landed**: `target_history` is **1.5%** of encoder
  importance, against TiDE's 0.42–0.49 that got it rejected in exp-8. TFT is genuinely using
  the covariates, not leaning autoregressive. Encoder importance is led by
  `network_pressure_forecast` (0.337) and `queue_pressure_forecast` (0.105) — consistent with
  the r² = 43%/47% from the EDA.

Two problems it exposed, both now fixed: the budget blowout above, and a **confounded probe**
— `crop` "won" by +1.393 WAPE but ran 16.4 min against `base`'s 1.8 min, because variable
encoder length inflated its window count ~12×. It may have been winning on data volume, not
on the augmentation. The probe table now reports `windows` and `steps_ep` per arm and prints
an explicit warning when arms saw different amounts of data.

Also fixed: residuals are now `pred − target` (negative = under-prediction), matching
`IMPROVEMENT-FRONTIERS.md`; run 1 reported `+5.84` against a quoted `−5.60` with no note that
the conventions were opposite. And §9 now prints a loud warning instead of three quiet
`[skip]` lines when the prior OOF files are missing — in run 1 that silently skipped the
residual-correlation gate *and* left §12 with a single blend arm.

## Verification

`test_notebook.py` is a regression test for the notebook. It executes **every code cell in
order** against synthetic data (4 series × 4,320 + 336 hours, real column names, NaN gaps) on
CPU with two simulated devices, then asserts hard invariants on everything produced. Fake
data, no GPU, no Kaggle quota.

```bash
python test_notebook.py          # PREFLIGHT path: 1 seed, stride 168, tuning on
python test_notebook.py --full   # real path: multi-seed, stride 24, tuning skipped
```

Both modes currently pass — 15/15 cells, ~40 assertions, ~4 min each. **Re-run both after any
edit to the notebook.** Needs `pytorch-forecasting==1.8.0`, optuna, optuna-integration,
statsmodels, tensorboardX, matplotlib, scipy, pyarrow.

What it asserts, beyond "nothing raised":

| Area | Checks |
|---|---|
| OOF | row count `3×96×336`, exact repo column layout, no NaN/inf, all folds present, equal rows per fold, `lead_h` 1–336, no duplicates, **each fold's window starts exactly on the pre-registered boundary** |
| Submission | row count, column layout, all finite, **covers every row of the forecast index**, timestamp range matches |
| exp-11 Bug 1 | final fit's last training target `== T_TRAIN-1`, encoder gap `≤` horizon |
| Premises | loss confirmed on the ORIGINAL target scale; `causal_attention` confirmed to change the forward pass |
| Dispatcher | one result per job, wall-clock stamps present, and **overlapping intervals proving jobs really ran concurrently** |
| Multi-seed | CV job count `== folds × seeds`; `EXP_ID` matches the mode (preflight output can't overwrite real output) |
| Probe/tuning | 4 arms each scored; tuning table written when on, correctly absent when off |
| Importance | all three scopes present, normalised to 1.0 per scope |
| Blend | written, correct row count, finite |
| Logs | no traceback in any worker log |

### Bugs this caught

| Bug | Consequence if shipped |
|---|---|
| **Stride anchored to absolute time 0** | `.filter(time_idx % stride == 0)` drops the final training window whenever the stride doesn't divide `max_start`. `STRIDE=24` happens to divide 3984, so the real config was fine — but PREFLIGHT's 168 was not, and any future stride change would silently reintroduce **exp-11's Bug 1** (LB 17.9 → 14.8). Now anchored to the *last* valid window, so it holds for any stride, with an in-worker assert. |
| `weight_decay` in `optimizer_params` | `TypeError` at `trainer.fit` |
| `LearningRateMonitor` without a logger | `MisconfigurationException` at `trainer.fit` |
| §13 (old `optimize_hyperparameters` path) missing `optuna-integration`, `statsmodels`, `tensorboardX` | three successive `ImportError`s, each surfacing only after the previous was fixed. That path has since been replaced by the ask/tell loop, which needs optuna alone. |
| Worker docstring escaping | generated worker file was not valid Python |

## What has been verified, and what has not

**Executed end-to-end** (both modes of `test_notebook.py`): every cell, the worker/dispatcher
path, job queueing, per-GPU pinning, `.npz` marshalling, all four probe arms, 3-fold CV with
multi-seed averaging, interpretation, final fit, submission, blend, and both tuning cells.

**Verified statically** as well: every cell compiles; every name resolves; the generated
worker compiles and its names resolve; the notebook↔worker contract is exact (worker reads 32
`cfg` keys, `CFG` defines exactly those 32 — nothing missing, nothing unused); and every
`make_job(**overrides)` key maps to a real `CFG` key.

**Still not exercised anywhere** — all of it environment-specific, and all of it covered by
running `PREFLIGHT = True` on Kaggle first:

- **fp16 and the `mask_bias` fix.** The tests run `32-true` on CPU. This is the one landmine
  that can still bite, and PREFLIGHT hits it at the real `hidden_size=96`.
- **Real CUDA and two physical GPUs.** The dispatcher was tested with two concurrent CPU
  processes, which exercises the queueing logic but not `CUDA_VISIBLE_DEVICES` pinning.
- **Peak GPU memory** at encoder 336 + decoder 336, batch 32. §6 reports it.
- **Kaggle's package resolution.** `pytorch-forecasting` is pinned to 1.8.0, the version
  everything was verified against; if the install cell has to fetch it, restart the kernel.
- **Accuracy.** Synthetic data with a planted linear signal says nothing about WAPE. There is
  zero evidence here about whether TFT beats 14.203.

Everything runs on Kaggle. Nothing in this experiment requires a local GPU.

---

## Files

```
experiment-12/
├── README.md
├── kp-experiment-12a-tft.ipynb    — §0 config … §13 optional tuning
├── probe_12a_tft.csv              — §7 arm comparison               (on run)
├── tuning_12a_tft.csv             — §13 tuning trials               (on run, optional)
├── oof_predictions_12a_tft.csv    — 3-fold OOF, repo-standard layout (on run)
├── variable_importance_12a_tft.csv— §10 encoder/decoder VSN weights  (on run)
└── validation_predictions_12a_tft.csv — submission                   (on run)
```

Notebook layout: §0 config · §1 env · §2 data → parquet · §3 CV + job config ·
§4 writes the worker script · §5 dispatcher · §6 throughput + premise gate · §7 config probe ·
§8 full CV · §9 diagnostics · §10 variable selection · §11 final fit + submission ·
§12 blend · §13 optional hyperparameter tuning (both GPUs) · §14 package results.

Stages are individually switchable (`RUN_THROUGHPUT_PROBE`, `RUN_CONFIG_PROBE`,
`RUN_FULL_CV`, `RUN_FINAL_FIT`, `RUN_TUNING`). **Run §6 first and read the projection**
before committing to §7/§8.

### Outputs

Intermediates (job specs, worker logs, per-job results, checkpoints, the panel parquet) go to
**`/kaggle/temp/run/`**, which is scratch and is *not* persisted as notebook output — so the
output browser never fills with ~50 files across four subfolders. (If `/kaggle/temp` is
missing, they fall back under `OUTPUT_DIR`.)

§14 then writes exactly two things to `/kaggle/working`, alongside the result CSVs:

- **`run_report_<EXP_ID>.md`** — a self-contained digest of the entire run: environment,
  config, premise checks, throughput, probe table, per-fold and per-job CV, per-series and
  per-lead WAPE, spike-tail stats, residual correlations vs prior models, variable
  importance, final-fit guards, submission stats, tuning, job timings and achieved GPU
  speedup. ~6 KB on a toy run, ~15–25 KB on a real one — small enough to open in the browser
  or paste straight into a chat for analysis.
- **`experiment12_<EXP_ID>_<UTC>.zip`** — one download with the report, every result CSV, and
  the whole run tree (job specs, logs, result JSONs, raw `.npz` predictions, worker script).
  Checkpoints excluded by default (~10 MB each, useless for analysis); set `INCLUDE_CKPT` to
  keep them. `PRUNE_RUN_DIR` deletes the scratch tree afterwards — off by default so a failed
  run keeps its logs.

Preflight outputs carry a `_preflight` suffix throughout — including the blend — so a toy run
can never overwrite real results. Re-running a stage reuses nothing; delete the matching
results for a clean repeat.
