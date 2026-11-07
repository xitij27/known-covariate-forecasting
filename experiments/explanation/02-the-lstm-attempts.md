# The LSTM attempts (experiments 1, 2, 3)

I spent three experiments trying to make an LSTM work on this data. None of them did. Here is the play-by-play, and what I should have noticed earlier.

## Experiment 1 — Baseline LSTM

I built a standard sequence model: a 168-hour lookback window → predict 24 hours ahead → then roll forward 14 times to cover the full 336-hour forecast window. The features I used:

- 7 time encodings (hour, day-of-week, day-of-year in sin/cos, weekend flag)
- 8 target-lag features (lag 1, 2, 3, 6, 12, 24, 48, 168)
- 6 rolling stats of the target (mean and std over 6h, 24h, 168h windows)

A 2-layer LSTM with hidden size 256, dropout 0.1. Huber loss.

I split training and validation by taking random slices of the sliding windows. Local WAPE came out 27.25, beating the seasonal_mean baseline of 31. Felt good. Submitted.

**Leaderboard: WAPE 37.16. Worse than the baseline.**

What went wrong? I had used `torch.utils.data.random_split` on the pooled set of sliding windows. That meant windows from later in a series could land in training while earlier windows from the same series landed in validation. The model was essentially seeing the future of a series during training. The local val score was fake — it measured how well the model interpolated within a series, not how well it forecast forward in time.

Lesson: **for time series, the train/val split has to respect time direction.**

## Experiment 2 — Chronological split

I rewrote the split. For each series, the last 336 hours went to validation, everything earlier went to training. Now the model was being honestly tested.

The training curve looked like this:

```
Epoch  1: train=0.217  val=0.254  ← best
Epoch  2: train=0.170  val=0.291
Epoch  6: train=0.065  val=0.342
```

Training loss kept dropping. Validation loss diverged from epoch 1 and just kept rising. Classic overfitting shape. I tried regularization:

- Bumped dropout to 0.3
- Lowered learning rate to 3e-4
- Smaller hidden size (128)
- Bigger batch size (64)

```
Epoch  1: train=0.533  val=0.568  ← best
Epoch  6: train=0.416  val=0.645
```

Same pattern. No regularization combination changed it. At this point I was frustrated and not sure what was wrong. The model seemed to have plenty of capacity, but val never improved past epoch 1.

This is when I decided to step back and look at the data more carefully — which led to the analysis I described in `01-understanding-the-problem.md`. My working theory was: maybe the model just needs more signal. It's trying to predict from target lags alone, but the target is driven by forecast features it doesn't see.

## Experiment 3 — Add the domain features

I added the 11 forecast features to the input. Now the LSTM saw `queue_pressure_forecast` and friends directly. I also did the following:

- Imputed the NaN gaps with forward-fill within each series, then zero-fill any leading NaNs.
- Built a proper rolling forecast that looks up the val period's domain feature values from `validation_input.csv` instead of leaving them at zero or computing them from predictions.
- Switched from Huber to L1 loss. Reasoning: WAPE is essentially a weighted absolute error, so training with L1 should align better with the metric than Huber.
- Dropped `nominal_capacity` and `trend` (zero predictive signal).

This was a much more thoughtful experiment than experiment 1 or 2. I expected it to work.

```
Epoch  1: train=0.485  val=0.548  ← best
Epoch  4: train=0.327  val=0.647
Epoch  8: train=0.246  val=0.640
```

Same. Shape. As. Experiment 2. Training loss dropped steadily, val loss diverged from epoch 1.

That was the moment I realized something deeper was going on. If `queue_pressure_forecast` really had a 45% correlation with the target, adding it should have *bent the val curve down at least a little.* It didn't move at all. So either the features didn't actually have that signal out-of-sample, or my val loss was measuring the wrong thing — or both.

I needed a sanity check. That's what set up the pivot.
