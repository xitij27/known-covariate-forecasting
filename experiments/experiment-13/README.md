# Experiment 13 — improve the TFT

*Stage 1 complete (no improvement). Stage 1b is next and blocks stages 2–4.*

Written 2026-08-17. **TFT is our best model: LB WAPE 14.358, rank 57**, beating the previous
best (`lgbm_mlp_ensemble`, 14.682) by 0.324, with the best MAE of any of our entries (1.5805).
exp-13 improves *that* model.

Read `CLAUDE.md` §"The CV protocol trap" and `PITFALLS.md` §A7 first.

## Why TFT and not the GRU

An earlier draft proposed searching on the GRU because it is ~30× cheaper per run, then
transferring the winner to TFT. **Our own data says transfer does not hold:**

| intervention | GRU | TFT |
|---|---|---|
| subsequence-crop augmentation | 15.372 → 13.840 = **−1.53** | 17.243 → 17.047 = **−0.196** |

Same augmentation, ~8× different effect. A search on the GRU measures what helps the GRU.
Compute is also the weaker constraint: quota resets weekly and checkpoints make runs resumable
across sessions, so a multi-week TFT campaign is affordable. Optimise the model that is
actually winning.

## Status

| Stage | What | Cost | Expected | Status |
|---|---|---|---|---|
| 0 | equal-weight blend probe | free | unknown | not started |
| 1 | multi-seed final fit | 5.84 GPU-h | −0.1 to −0.2 | **DONE — LB 14.479, no improvement** |
| **1b** | **measure the seed-to-seed noise floor** | ~2.2 GPU-h | — | **NEXT — blocks stage 2** |
| 2 | the two missing augmentations | ~8 GPU-h | −0.1 to −0.5 | blocked on 1b |
| 3 | cosine schedule + longer training | ~8 GPU-h | −0.05 to −0.2 | blocked on 1b |
| 4 | capacity / encoder length | ~10 GPU-h | unknown | conditional |

Spent so far: **5.84 GPU-h** (stage 1). Stage 1b is 2.2 GPU-h and must run before stages 2–4
are worth costing — see "Consequence" below.

---

## Stage 1 results — no improvement, and it exposed a bigger problem

| entry | MAE | WAPE | MAPE | SMAPE |
|---|---|---|---|---|
| `tft` (1 seed) | **1.5805** | **14.358** | 19.321 | 15.830 |
| `tft-run3` (5-seed mean) | 1.5939 | 14.479 | **18.543** | **15.682** |

Seed averaging cost **+0.121 WAPE**. Note the split: every *absolute* metric (MAE, MSE, RMSE,
WAPE) got worse while both *relative* ones (MAPE, SMAPE) improved. Averaging shifted the level
up (+0.222) and pulled the peaks in (max 39.50 → 36.60), moving away from the conditional
median that L1 rewards.

### But the result is confounded with run-to-run noise

Run 3's **seed-42** model and the submission that scored 14.358 use the same seed, config and
data. They differ by **mean |diff| 0.403**. TFT training is non-deterministic (cuDNN atomics /
non-deterministic kernels), so "seed 42" is not a fixed model.

| difference being measured | magnitude |
|---|---|
| same seed, two runs (pure noise) | **0.403** |
| 1-seed vs 5-seed mean (the intervention) | 0.375 |
| mean vs median aggregation | 0.103 |

**The intervention is the same size as the noise.** The +0.121 WAPE cannot be attributed to
averaging — 14.358 may have been a favourable draw, and 14.479 the more reliable estimate of
what this config scores.

### Consequence: the plan is blocked until the noise floor is known

The project's decision threshold is **0.05 WAPE**. Single-submission comparisons on this model
cannot resolve anything near that. An augmentation worth 0.1–0.2 would be invisible.

**Stage 1b (next):** 3 seeds on **one CV fold** (fold 0, 10 epochs, ~45 min per seed ≈
**2.2 GPU-h**) to measure TFT's seed-to-seed WAPE spread directly. exp-11 measured the GRU's at
0.183 on fold 0; TFT's is unknown.

That number decides how stages 2–4 must be run:
- spread ≲ 0.1 → single-submission LB probes are usable
- spread ≳ 0.2 → **nothing below ~0.4 is measurable by one submission**; stages 2–4 need
  multi-seed CV, roughly tripling their cost, or must be dropped

### What stage 1 did deliver

- The first model checkpoints this project has kept (5 × 12.5 MB).
- A validated cost model: predicted **5.7 GPU-h / 3.4 h wall**, actual **5.84 / 3.52** — within
  2.5%, the first accurate projection here.
- `validation_predictions_13b_tft_seedmedian.csv` — the 5 seeds re-aggregated by **median**
  instead of mean, built for free from the saved `.npz` files. Worth one submission slot as a
  *third sample* to bracket the noise, not as a fix: the mean-vs-median difference (0.103) is
  4× below the noise floor.

---

## THE PROTOCOL LOCK

All stages use **`EARLY_STOP_MODE = "inner"`**, matching exp-12. Never compare CV against
exp-11's 14.203 — that number early-stops on the block it reports and is ~0.57 optimistic.
Within exp-13 all comparisons are like-for-like, so CV deltas are valid *relative to exp-12's
14.722*, which is the only baseline that counts here.

---

## Stage 0 — equal-weight blend probe (free, no GPU)

`0.333 × tft + 0.333 × gru + 0.333 × 7c` from the three existing submission CSVs. Equal weights
avoid the mixed-protocol bias that gave TFT weight 0.0 in `12b_blend`.

**Pre-registered:** beats 14.358 → keep blending in scope. Otherwise TFT solo is the ship and
later stages only need to beat 14.358.

---

## Stage 1 — multi-seed final fit (DONE — see results above)

**The 14.358 submission averaged ONE seed** (run 4 report: `seeds averaged: 1`). exp-11's GRU
averaged **five**. Seed averaging is the most reliable variance reduction available and needs
no new code — `SEEDS` is already a config list and §11 already averages across it.

exp-11 measured a fold-0 seed spread of 0.183 for the GRU; averaging 5 seeds typically
recovers 0.1–0.2 WAPE. This is the cheapest expected gain in the plan.

- `SEEDS = [42, 7, 13, 99, 1234]`, `FINAL_EPOCHS_OVERRIDE = 10` (the CV median from run 4).
- Run **§11 only** — `RUN_THROUGHPUT_PROBE/CONFIG_PROBE/FULL_CV = False`. ~1.2 GPU-h per seed.
- **Save every checkpoint** (`save_ckpt=True`, already wired to `OUTPUT_DIR/models`).

**Pre-registered:** submit the 5-seed average. Accept if LB beats 14.358 at all — seed
averaging cannot overfit, so any improvement is real.

**Outcome: 14.479, rejected.** And the pre-registration was itself flawed — "averaging cannot
overfit" is true, but it does not follow that any *change* is signal. The measured run-to-run
noise (0.403) turned out to be the same size as the intervention, so the comparison could not
have decided anything either way. Effect size should have been compared against the noise floor
*before* the run, which is what stage 1b now establishes.

---

## Stage 2 — the two missing augmentations (~8 GPU-h)

Augmentation gave the GRU **−1.53**, the largest single gain in this project. TFT has only one
of the three:

| exp-11 augmentation | in exp-12's TFT | how to add |
|---|---|---|
| subsequence crop | **present** (`randomize_length=(0.2,0.05)`) | — |
| **feature dropout** (p=0.1 per covariate column, train only) | **absent** | zero random covariate columns in the encoder input |
| **hard-series oversample** (3× on worst-20) | **absent** | `WeightedRandomSampler` via `to_dataloader(batch_sampler=...)`, weights from the exp-12 per-series OOF WAPE |

Note the crop arm only gave TFT −0.196, so do **not** assume the other two will give −1.53
either. They are worth testing because they are cheap, proven on this dataset, and absent.

**Arms** (3 folds × 2 seeds each, `inner` protocol):

| arm | config |
|---|---|
| B0 | exp-12 config = baseline (CV 14.722) |
| B1 | + feature dropout |
| B2 | + hard-series oversample |
| B3 | + both |

**Pre-registered:** accept an arm only if CV beats B0 by > **0.05** with consistent per-fold
direction. Single-fold differences under **0.3** are not evidence. Winner goes to a 5-seed
final fit and a submission.

**Hard-series list:** rebuild it by **share of the global error numerator**, not individual
WAPE — `unit_025, _094, _052, _070, _093` are major contributors that the old worst-20 list
omits, and only 3 of 10 overlap. See `diagnostics/per_series_triage.py`.

---

## Stage 3 — cosine schedule + longer training (~8 GPU-h)

exp-12 ran with **no LR schedule** (only `reduce_on_plateau_patience=4`); exp-11's GRU used
`CosineAnnealingLR(T_max=50)`. `best_epoch` was 8–11 at flat LR, so "train longer" alone is
answered — but longer training *with annealing* reaches a different optimum and was never
tested.

| arm | config |
|---|---|
| C0 | winner of stage 2 |
| C1 | + `CosineAnnealingLR(T_max=MAX_EPOCHS, eta_min=1e-5)`, `MAX_EPOCHS=60`, `PATIENCE=15` |
| C2 | C1 + one-cycle warmup |

Use **checkpoint resumption** so a 60-epoch run can span sessions: pass `ckpt_path=` to
`trainer.fit` and persist to `OUTPUT_DIR/models` after every run. This is the mechanism that
makes long training practical against the 12 h session cap.

**Pre-registered:** > 0.05 over C0, per-fold consistent.

---

## Stage 4 — capacity / encoder length (conditional, ~10 GPU-h)

Only if stages 1–3 leave quota and are still improving. `hidden_size` 96 → 128/160,
`encoder_length` 336 → 672, `lstm_layers` 1 → 2. §13's ask-and-tell Optuna loop already
dispatches these across both GPUs.

Note: batch size is **not** a lever (26.9 ms/sample flat across 32/64/128, measured twice).

---

## Notebook build rules

Every line is a bug that actually happened in exp-12. Full write-ups in `PITFALLS.md`; the
authoring checklist is at the end of that file.

1. **One protocol** — `EARLY_STOP_MODE = "inner"` everywhere.
2. **Save the model** for any run whose predictions get submitted, to the *downloadable* dir.
   For early-stopped runs **copy** `best_model_path`; `trainer.save_checkpoint()` writes the
   last epoch, not the best.
3. **Resolve input files by filename**, never by hardcoded `/kaggle/input/<slug>/`.
4. **Never take the first `pred*` column** — `oof_predictions_7b_blend.csv` has four.
5. **`residual = pred − target`**, one convention per notebook.
6. **Per-bucket diagnostics in MAE, not WAPE** — fold blocks are 14 days apart and all start
   Friday, so lead buckets are confounded with day-of-week.
7. **Rank series by share of the global error numerator**, not individual WAPE.
8. **`PREFLIGHT = True` first**, every output suffixed — including blend files.
9. **Budget from measured `ms/sample`** on the target device; never extrapolate CPU→GPU.
10. **Reuse checkpoints instead of retraining** (exp-12 §10 originally trained a duplicate).
11. **After every codegen string-replace, assert the string changed** — a passing
    `assert old in s` followed by writing unchanged text is a silent no-op that happened four
    times in exp-12.

## Verification before any GPU time

Copy `experiment-12/test_notebook.py` and adapt it: it executes every cell against synthetic
data on CPU with two simulated devices and asserts ~60 invariants. Run **both** modes after any
edit. Reading code did not find the bugs that mattered in exp-12; running it did.
