# The pivot — realizing the LSTM was the wrong tool

After experiment 3 didn't work, I added a diagnostic section to the notebook. The idea was simple: test the data analysis claims directly, without any LSTM in the way.

## Diagnostic A — are the features actually predictive?

For each domain feature, I computed the out-of-sample correlation with the target on the held-out last 14 days of training. None of the regressions below were trained on that chunk. If the data analysis numbers reproduced on data the model hadn't touched, the features were real. If they collapsed, the data analysis was misleading.

Results:

| feature | claimed r² | out-of-sample r² (pooled) | (within-series) |
|---|---|---|---|
| `queue_pressure_forecast` | 45% | 47% | 54% |
| `network_pressure_forecast` | 43% | 43% | 50% |
| `workload_intensity` | 29% | 30% | 38% |
| `shock_risk` | 25% | 18% | 24% |

The features were real. They explained most of the target on data the model hadn't seen.

## Diagnostic B — can a trivial model use them?

If those features are real, I shouldn't need an LSTM to exploit them. So I fit a plain **linear regression** on the 11 forecast features alone, trained on the first ~166 days, evaluated on the held-out last 14 days. WAPE.

```
seasonal_mean (baseline)       WAPE = 31.40
linear / domain features       WAPE = 21.00
linear / domain + time         WAPE = 21.12
lightgbm / domain + time       WAPE = 21.20
```

A linear regression. No autoregression, no sliding windows, no rollout. **WAPE 21**, beating the baseline by a third. And my LSTM, with all its complexity, hadn't beaten the baseline at all.

That was a slap-in-the-face moment. The features could carry me to a great score with the simplest possible model. The LSTM wasn't a better model that needed more tuning — it was the wrong kind of model entirely.

## Why the LSTM kept failing

Once I could see this, the rest fell into place. The LSTM had three things working against it:

1. **The training loss was teacher-forced.** Every training window fed the model real lags of the target. With a real `lag_24` and real `lag_168`, the autocorrelation alone was almost enough to predict 24 hours ahead. The model overfit that easy signal and ignored the harder structure (the domain features).

2. **The competition metric is autoregressive, not teacher-forced.** At inference, I had to roll forward 14 times over a 336-hour horizon. After the first 24 hours, the "target lags" I was feeding the model were the model's *own predictions*. Errors compounded. The whole point of having the lag features collapsed across the rollout.

3. **The killer fact:** `validation_input.csv` has no `target` column. **There are no real target lags at inference time, ever.** Every lag feature the LSTM relied on was synthetic — fabricated from its own predictions. The rollout was an architectural patch papered over a fundamental mismatch.

So the model was being trained on a problem (teacher-forced 24h prediction) that wasn't the deployment task (336h autoregressive rollout on a horizon where target lags don't exist).

The deeper insight: the `validation_input.csv` having no target column isn't just a missing column. It's telling me what kind of problem this is.

## What kind of problem this actually is

The data gives me, for every hour in the forecast window, the values of `queue_pressure_forecast`, `network_pressure_forecast`, and the rest. Those features are *inputs I already have*, not things I need to forecast. They cover the full 14-day horizon.

So the problem is:

> Given the known covariates at time *t*, predict `target[t]`.

That's a per-hour regression on the provided forecast features. Not a sequence forecasting problem. There is no past-to-future inference needed at all — the future is in the input already.

This reframing changes everything. The LSTM was solving an autoregressive problem that didn't exist. A regression on the features that *do* exist at inference is the natural fit.

That's what I built next.
