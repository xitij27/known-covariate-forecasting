# PITFALLS.md — failures, root causes, and what now prevents them

A record of every real failure hit while building `experiment-12/` (TFT on Kaggle T4 ×2), so
none of it costs time twice.

**The meta-lesson, stated once:** *reading code finds fewer bugs than running it.* Every bug
below that actually mattered was found by executing something — the end-to-end test suite, a
compile check, or a real Kaggle run. The ones found by reading were mostly cosmetic. When a
change is expensive to get wrong, run it against synthetic data before running it for real.

Categories: [silent-wrongness](#a-silent-wrongness) · [crashes](#b-crashes) ·
[waste](#c-waste) · [estimation](#d-estimation-failures) · [process](#e-process-failures)

---

## A. Silent wrongness

The dangerous class: no error, plausible output, wrong conclusions.

### A1. Stride anchored to absolute time 0 — reintroduces exp-11's most expensive bug

**Symptom:** none. Training completes, CV looks normal.
**Cause:** `.filter(time_idx_first_prediction % stride == 0)` anchors the window grid to time 0.
The last valid decoder start is 3984; `STRIDE=24` divides it, so the real config was fine by
luck. Any stride that doesn't divide it silently drops the final training window, so the final
model never trains on data reaching the forecast origin — exp-11 Bug 1, which cost **LB 17.912
→ 14.773**.
**Fix:** anchor the grid to the *last* valid window and step backwards, so the final window
survives for any stride. Plus an in-worker assert that the last window ends exactly at
`train_end`.
**Caught by:** the end-to-end test, because PREFLIGHT used stride 168 (which doesn't divide
3984). Last training target came out 4199 instead of 4319.
**Prevented by:** `test_notebook.py` asserts `last_train_target == T_TRAIN - 1` and
`encoder gap ≤ horizon`.

### A2. `TimeSeriesDataSet` emits one index row per encoder length

**Symptom:** probe arms with `min_encoder < max_encoder` took 16.4 min against 1.8 min for
fixed-encoder arms, and the dataset was ~12× larger than the stride implied.
**Cause:** with `min_encoder_length < max_encoder_length`, pytorch-forecasting enumerates
**one index row per allowed encoder length** — measured at ~12 rows per decoder start.
Filtering on `time_idx_first_prediction` keeps *every* variant.
**Consequences, all silent:** (1) the stride does not mean what it says; (2) arms with
different `min_encoder` are not compute-comparable, so the `crop` arm's +1.393 WAPE "win" was
confounded with ~12× more gradient steps; (3) the real-run budget blew up ~12×.
**Fix:** keep **one row per (series, decoder start)** — the longest — and let
`randomize_length` do the cropping. Verified the augmentation survives: encoder lengths still
sample `[336, 168, 312, 336, 171, …]`.
**Caught by:** reading run 1's per-arm wall-clock times, then reproducing it in isolation.
**Prevented by:** an exact window-count assertion in `test_notebook.py`, plus the probe table
now reporting `windows`/`steps_ep` per arm and warning when arms saw different data volumes.

### A3. `optimize_hyperparameters` silently substitutes TFT defaults

**Symptom:** none. The study runs and reports a "best" config.
**Cause:** it builds the model itself and takes library defaults for anything not forwarded
through `**kwargs`. Not forwarding meant tuning a **QuantileLoss, causal** model with a
**−1e9 mask bias under fp16** — while the notebook was configured for MAE, non-causal. The
`val_loss` values were not comparable to anything else in the notebook.
**Fix:** removed that code path entirely. §13 now drives Optuna in ask-and-tell mode over the
same worker used everywhere else, scored on fold WAPE. Config flows through `make_job`, so
there is nothing to forget to forward.
**Caught by:** checking the function signature to answer a user question — i.e. by luck.

### A4. Residual sign convention flipped

**Symptom:** run 1 printed `mean residual +5.84` next to the quoted reference `−5.60`, as if
the model behaved oppositely to prior work.
**Cause:** the code computed `target − pred`; `IMPROVEMENT-FRONTIERS.md` uses `pred − target`
(negative = under-prediction). Both describe the same under-shooting, with opposite signs.
**Fix:** standardised on `pred − target` everywhere, with the convention stated inline.
**Prevented by:** the convention is now written down in `CLAUDE.md`.

### A5. Preflight could overwrite real results

**Symptom:** none until it destroyed something.
**Cause:** every output was suffixed `_preflight` *except* the blend submission, which wrote to
a bare `validation_predictions_12b_blend.csv`. A 5-minute toy run could silently clobber a real
submission.
**Caught by:** listing the output directory after a test run.
**Prevented by:** the suffix is applied to the blend too; the harness asserts `EXP_ID` matches
the mode.

### A6. Missing prior files skipped a pre-registered gate quietly

**Symptom:** three tidy `[skip]` lines in run 1's output.
**Cause:** the prior experiments' OOF CSVs weren't attached, so the residual-correlation
computation was skipped — and that correlation is the **pre-registered gate** for shipping TFT
as a blend arm. §12 also silently fell back to a single arm and produced no blend.
**Fix:** a loud, boxed warning explaining exactly what was not computed and why it matters.
**Lesson:** a skipped check should be as visible as a failed one.

## A8. Measuring effects smaller than the noise floor

**Symptom:** exp-13 averaged 5 seeds instead of 1 and scored **14.479** vs the single seed's
**14.358** — apparently a 0.121 regression. Recorded as "seed averaging hurts".

**Reality:** two runs with the **same seed, config and data** differ by **mean |diff| 0.403** in
their predictions. TFT training is non-deterministic (cuDNN atomics, non-deterministic kernels);
`seed_everything()` does not pin the model.

| difference | magnitude |
|---|---|
| same seed, two runs (noise) | **0.403** |
| 1-seed vs 5-seed mean (the intervention) | 0.375 |
| mean vs median aggregation | 0.103 |

The intervention was the same size as the noise, so **the experiment could not have decided
anything** — in either direction. 14.358 may simply have been a favourable draw.

**Cost:** 5.84 GPU-h and one submission slot on an unanswerable question.

**Prevention:** before running an experiment, state its expected effect and compare it to the
measured noise floor. If the effect is not comfortably larger, the design must change
(multi-seed CV, ~3× cost) or the experiment should not run. Measure the floor once — exp-13
stage 1b — and reuse it.

**Related, worth knowing:** seed averaging is *not* free on an L1 metric. MAE-trained models
target the conditional **median**; averaging pulls toward the mean and shrinks peaks. In exp-13
every absolute metric (MAE/MSE/RMSE/WAPE) worsened while MAPE and SMAPE improved — the
signature of that shift. Median aggregation is the matched operation.

---

## A7. Comparing CV across early-stopping protocols — inverted the final verdict

**Symptom:** TFT's CV (14.722) ranked it last; exp-11's GRU (14.203) ranked first. TFT was
recorded as rejected.
**Reality:** on the leaderboard TFT scored **14.358** and the GRU **14.773**.

| | early-stops on | CV | LB | CV→LB |
|---|---|---|---|---|
| GRU (exp-11) | the fold test block it then reports | 14.203 | 14.773 | **+0.570** |
| TFT (exp-12) | an inner block; test block untouched | 14.722 | **14.358** | **−0.364** |

**Cause:** exp-11 selects on the block it scores (optimistic); exp-12 also trains on 336 h less
data per fold (pessimistic). ~0.9 WAPE of relative distortion. The incomparability *was*
documented in the exp-12 README — and then the comparison was used anyway.

**Second-order damage:** blend weights fit on that mixed-protocol OOF gave the best model
weight **0.0**, so the shipped blend excluded it.

**Prevention:** one `EARLY_STOP_MODE` per comparison, recorded next to every OOF file. Never
fit blend weights across protocols. When protocols differ, the leaderboard is the only
arbiter.

**Tell:** a model trained on original-scale MAE targets the conditional **median**, which on a
right-skewed target sits below the mean (TFT predicted 10.175 vs GRU 10.614). It should win
MAE/WAPE and look ordinary on MSE/RMSE — which is exactly what the leaderboard showed, and was
visible before the CV comparison was trusted.

---

## B. Crashes

Cheap when caught locally, expensive when they happen 20 minutes into a GPU session.

| # | Failure | Root cause | Caught by |
|---|---|---|---|
| B1 | `TypeError: got multiple values for 'weight_decay'` | `weight_decay` is a top-level `from_dataset` kwarg; pytorch-forecasting already forwards it to the optimizer. Passing it via `optimizer_params` duplicates it. | end-to-end run |
| B2 | `MisconfigurationException` from `LearningRateMonitor` | it requires the `Trainer` to have a logger; ours had `logger=False` | end-to-end run |
| B3 | Generated worker was invalid Python | the worker source is embedded in an `r"""…"""` string; a `"""` docstring inside terminated it early. **Hit twice** — the second time while adding `StepTimer`. | compile check |
| B4 | Three successive `ImportError`s in the Optuna cell | `optimize_hyperparameters` needs `optuna`, `optuna-integration` (since optuna 3.3), `statsmodels` **and** `tensorboardX` — none pulled in by pytorch-forecasting. Each only appeared after the previous was fixed. | end-to-end run |
| B5 | fp16 NaN losses (latent) | TFT's `mask_bias` defaults to `-1e9`, which overflows in fp16. A T4 is Turing — no bf16 — so `16-mixed` is forced. | reading the docstring; confirmed safe only by run 1 |
| B6 | Batch sweep would abort the whole notebook | `run_jobs` raised on any failure, so an OOM at batch 128 — plausible at 672 timesteps — would take down the entire preflight | pre-run review |

---

## C. Waste

Not wrong, just expensive.

### C1. §10 retrained a model that already existed

The interpretation cell trained its own model on the last fold — byte-identical in config to a
CV job that had just run and been discarded. **A full extra training run (~0.5–1 h) with the
second GPU idle.** Fixed: §8 keeps exactly that one checkpoint; §10 loads it. In run 1 this
single job took 16.8 min of a preflight.

### C2. Preflight cut epochs but not dataset size

Run 1's "preflight" took ~1.5 h. Epochs and seeds were reduced, but every job still built a
318,048-row window index across all 96 series, and rebuilt it in each of ~12 worker processes.
When training is only ~80 steps, **index construction and process startup are the runtime**.
Fixed: preflight subsets to 16 series — per-step cost and memory depend on batch × sequence
length, *not* series count, so fp16 and memory figures stay representative.

### C3. Forcing the sequential tuner on during preflight

Preflight forced `RUN_OPTUNA = True`, which is sequential, single-GPU, and pip-installs four
packages first. Removed from the preflight path.

### C4. Intermediates written to the Kaggle output directory

~50 files across four subfolders in `/kaggle/working`, making downloads painful. First fix
sent them to `/kaggle/temp` — but **that directory does not exist on every Kaggle image**, and
in run 2 the fallback silently put them back in the output dir. Now: `/kaggle/temp`, else
`/tmp/tft_scratch`, and only as a last resort the output dir. A test asserts `run/` never
appears under the output directory. §14 emits one zip plus a compact markdown report.

**Lesson:** a fallback that silently restores the behaviour you were fixing is not a fallback.

### C5. Timing measured through a validation pass, without CUDA sync

ms/step was derived from `sec_per_epoch / steps`, which includes validation — and CUDA is
async, so CPU-side timing understates GPU work. In preflight there'd be ~2 steps per epoch, so
validation would dominate entirely. Fixed: a `StepTimer` callback that synchronises CUDA around
each training step, attached **only** to warmup jobs (syncing every step would tax real
training); a test asserts other jobs stay unsynced.

---

## D. Estimation failures

Every one of these was a confident number with no measured basis.

| Claim | Reality | Factor |
|---|---|---|
| 88–184 ms/step on a T4 | **970 ms/step** | 5–10× |
| preflight "~5 minutes" | ~1.5 h (run 1) | ~18× |
| 221,280 windows/fold | 318,048 | 1.4× |
| ~825k parameters | 963,376 | 1.2× |

**Root cause of the big one:** extrapolating GPU cost from CPU timings. TFT does many small
matmuls across 672 timesteps, so it is **launch/memory-bound, not FLOP-bound** — a T4 gave only
~2.3× over 8 CPU threads, while the extrapolation assumed 12–25×.

### D5. "Bigger batches are nearly free" — plausible, and wrong

I predicted TFT would be launch-bound (many small matmuls over 672 timesteps), so a larger
batch would cost little per sample, and built a batch sweep on that premise. **Measured:
26.91 / 26.22 / 26.25 ms/sample at batch 32 / 64 / 128** — a 1.03× spread. It is compute-bound.
Peak memory scaled linearly too (1.68 / 3.33 / 6.61 GiB), so memory was never the constraint.

The sweep was still worth building — it *disproved* the hypothesis cheaply and permanently. The
error was stating the hypothesis as guidance in the README before measuring it.

**Rules now:** never extrapolate GPU cost from CPU for this model class; measure ms/step on the
target device before sizing anything; label any unmeasured number as a guess; and budget in
GPU-hours against the 30/week cap *before* starting a run.

---

## E. Process failures

### E1. Drip-feeding fixes instead of running the thing

The worst of these. Small bugs surfaced one at a time across several turns, which is corrosive
to trust and impossible to plan around. The cause was reviewing code instead of executing it.
**Fix:** `experiment-12/test_notebook.py` — executes every cell against synthetic data on CPU
with two simulated devices and asserts ~60 invariants. Run **both** modes after any notebook
edit. It has since caught A1, B4, and E3.

### E2. Trusting a documentation summary on a load-bearing claim

The docs said the loss is computed on the **normalised** target scale. It is computed on the
**original** scale (measured: reported `val_loss` 2.2105 = original-scale MAE 2.2105 vs
normalised 0.9054). That fact is the entire objective/metric-alignment argument for using TFT.
Had it gone unchecked, the experiment's stated rationale would have been wrong.
**Rule:** verify load-bearing library semantics by execution, not by documentation.

### E3. A patch that asserted a match but never applied it

A script did `assert old in s` and then wrote `s` unchanged — the `s = s.replace(old, new)`
line was missing. The assertion passed, so it looked successful, and the field it was supposed
to add was simply absent.
**Caught by:** the harness, which asserted the new field existed.
**Rule:** assert on the *post-condition* (the new content is present), not just the
pre-condition.

### E4b. E3 recurred — twice in one script, undetected for two runs

After documenting E3 (assert-without-replace), I wrote it **twice more** in the same patch
script. Both no-oped silently: the projection kept dividing by the *fastest* batch instead of
the configured one, and the quota STOP banner never landed — so run 2 printed the old
"`>1.5 h per run`" warning while a 60 GPU-h plan sailed past unflagged.

Writing a rule down does not enforce it. The fix is mechanical: the patch helper now asserts
the anchor exists, asserts the string actually **changed**, and asserts required tokens are
present afterwards. A rule that depends on remembering is not a control.

### E4. A static checker that was itself wrong

A name-resolution checker reported ~300 undefined names — all false positives, because it
validated uses *before* binding the same cell's own names. A broken checker is worse than none:
it produces noise that hides real signal.
**Rule:** validate the validator on a known-good input before believing its output.

### E5. Comparing experiment arms that weren't compute-matched

See A2. `crop` beat `base` by +1.393 WAPE while getting ~12× more gradient steps. **Any
comparison between configurations must be matched on training compute**, or the result measures
data volume rather than the thing under test.

---

## Checklist for AUTHORING a notebook

Every line is a bug that actually happened.

**Patching / codegen**
- [ ] After every string replace, assert the string **changed** and the new token is present.
      `assert old in s` then writing `s` unchanged is a silent no-op — it happened **four
      times**, twice after it was documented here.
- [ ] No `"""` inside a `r"""..."""` generated-source block. Use `#` comments. Hit twice.
- [ ] Compile every cell *and* any generated script after editing.
- [ ] Validate a static checker on known-good input before trusting its output.

**Correctness**
- [ ] One residual convention per notebook: `pred − target`.
- [ ] One `EARLY_STOP_MODE` per comparison (A7).
- [ ] Resolve input files by filename, not hardcoded mount paths.
- [ ] Never take the first `pred*` column; refuse to guess when ambiguous.
- [ ] Stride grids anchor to the **last** valid window (A1).
- [ ] Dedupe the window index to one row per (series, decoder start) (A2).
- [ ] Preflight outputs suffixed — *including* blend files (A5).
- [ ] Skipped checks must be as loud as failed ones (A6).

**Artifacts**
- [ ] Any run whose predictions are submitted saves a checkpoint to the **downloadable** dir.
- [ ] Copy `best_model_path` for early-stopped runs; `save_checkpoint()` saves the last epoch.
- [ ] Intermediates to scratch, with a fallback that is not the output dir.
- [ ] Select jobs by `kind`, not by sort order (the report read the wrong warmup job).

**Measurement**
- [ ] Time steps with CUDA synchronised, excluding the validation pass.
- [ ] Project budgets from measured `ms/sample` on the target device; never CPU→GPU.
- [ ] Average window counts across folds; fold 0 understates by ~20%.
- [ ] Per-bucket diagnostics use MAE; WAPE's denominator moves with the weekend.
- [ ] Rank series by share of the global error numerator, not individual WAPE.
- [ ] State the expected effect size and compare it to the measured noise floor **before**
      running. TFT's same-seed rerun noise is ~0.4 in prediction units (A8).

## Checklist before any expensive run

1. `python experiment-12/test_notebook.py` **and** `--full` — both green.
2. `PREFLIGHT = True` on Kaggle first. It catches what no local test can: fp16, real CUDA, two
   physical GPUs, peak memory.
3. Read §6's projection. Total GPU-hours against the 30/week cap, and per-run against the 12 h
   session cap. If it prints `STOP`, change §0 before continuing.
4. Check the probe table's `windows` column — arms must have seen comparable data.
5. Confirm prior-experiment OOF files are attached, or the blend gate silently does not run.
