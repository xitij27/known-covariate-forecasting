# What I learned

Roughly in order of importance.

## 1. Validate the problem framing before tuning the model

The LSTM didn't work for three experiments because the problem isn't a sequence problem. No amount of regularization, more features, or different loss functions was going to fix that — they were all addressing symptoms of a wrong framing. The fix was a complete change of approach.

I should have asked, much earlier: *what kind of problem is this, given what's in the inference inputs?*

## 2. Look at what's in the validation inputs

The most important observation in this project was that `validation_input.csv` has no `target` column. That single fact tells you:

- Target lags don't exist at inference time.
- Therefore any model that depends on target lags has to fabricate them.
- Therefore an autoregressive rollout is fighting the deployment shape.

I had access to this information from day one. I just didn't think about what it meant.

## 3. Try the simplest model first

A linear regression on the right features (WAPE 21) was a more useful experiment than three LSTM variations. Not because linear regression is good — it's not — but because it answers a more fundamental question: *do my features have signal at all?*

If a linear model on your features doesn't beat the baseline, no neural network will. If it does, you've at least proven the signal exists, and you can decide whether to chase the remaining gap with a fancier model.

## 4. Cross-validate when the data allows it

A single held-out split gave me WAPE 16.53. The honest rolling-origin CV gave me 17.27 ± 1.22. The difference is real — a 14-day window can be lucky or unlucky. The CV mean is what I should have been tracking from the start.

Bonus: CV with multiple folds also gives you a *standard deviation*, which is incredibly useful for telling whether an improvement is real or noise.

## 5. Match the objective to the metric

WAPE is essentially a weighted absolute error. Training with L2 loss (squared error) is structurally misaligned: it pulls the model toward fitting big-value rows at the expense of small-value rows.

Switching to MAE was the single biggest improvement I made (−1.08 WAPE on its own). It should have been the first thing I tried, not the eighth.

## 6. Distinguish real model error from metric artifacts

I noticed that WAPE inflated for low queue_pressure rows. At first I thought "the model is failing on these rows." But it isn't really — WAPE has a small denominator there, so even small absolute errors look like big percentage errors. The model is doing fine in absolute terms.

Knowing this stopped me from chasing improvements that wouldn't have helped.

## 7. Residual analysis is the highest-value diagnostic

The single most informative step in experiment 4 was breaking the out-of-fold error down by hour, by series, by feature regime. That table told me exactly which levers had room (per-series structure, objective alignment) and which were dead ends (low-target rows are mostly a WAPE-math issue).

Without that breakdown I would have spent more time on hyperparameter tuning and less on the actual signals.

## 8. Per-series structure mattered more than I expected

A 19-point WAPE spread across series is a flashing red light. One global model averaging over 96 different units was leaving real per-series gains on the table. Adding `series_id` as a categorical feature was a cheap, big win.

## 9. Diminishing returns are real — recognize when to stop

The improvements went: −1.89 from objective + per-series, −0.59 from temporal context. The next pass (ensembling, light tuning, bias correction) was estimated at 0.3–0.5 WAPE, against significant added complexity. I chose to stop and ship.

Recognizing this saved time that could have been spent on the wrong things.

## 10. Tabular ≠ deep learning

LightGBM handled the mixed-scale features (some at scale 10, some at 0.01), the 96 categorical series, the missing values, and the modest data volume per series — all without much tuning. Neural networks would have needed normalization, embeddings, imputation, and careful regularization to do the same job, for at most a marginal gain.

The lesson isn't "neural networks are bad." It's: **match the tool to the data shape.** For tabular regression with heterogeneous features and modest per-series data, GBMs are the better default.

---

## If I started over

If I had to redo this project from scratch:

1. Spend the first day understanding the data and the inference setup, not modeling.
2. **Specifically, check what's in the validation input file.** If it has all the features but no target, the problem is a regression on covariates, not a sequence forecast.
3. Try a linear regression on the obvious features. Get a number. Be honest about it.
4. If the number is good, try a GBM next. If the number is bad, that tells me the features need more work, not the model.
5. Cross-validate from the start. Single-split numbers are unreliable.
6. Match the loss to the metric.
7. Look at residuals before tuning hyperparameters.
8. Stop when returns flatten.

The LSTM journey wasn't entirely wasted — I learned a lot from it. But most of what I learned was about *why* it wasn't the right tool. If I'd seen that earlier, I could have spent that time on the actual model.
