# Experiment 2 — Fixing the Chronological Train/Val Split

Kaggle notebook: https://www.kaggle.com/code/xitij27/kp-experiment-2

---

## What Experiment 1 Got Wrong

Experiment 1 used `torch.utils.data.random_split` to divide the sliding-window dataset into train and validation subsets:

```python
# experiment-1 — BROKEN
full_dataset = SlidingWindowDataset(feat_df, FEATURE_COLS)
n_val   = max(1, int(len(full_dataset) * VAL_FRAC))   # 10 %
n_train = len(full_dataset) - n_val
train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED)
)
```

The `SlidingWindowDataset` pools windows from all 96 series into one flat list before any split happens.
A random draw from that list will mix windows from different points in time across different series.
Concretely, a window extracted from, say, hour 4000 of `unit_000` (late in the series) can end up in
`train_dataset`, while a window from hour 200 of the same series (early) ends up in `val_dataset`.

The model therefore trains on observations from the **future** of a series and is evaluated on
observations from its **past**. This is temporal leakage: the validation loss looks better than it
really is because the model has already seen data that chronologically comes after the held-out window.

### Why This Is Especially Harmful Here

The dataset is a 180-day multivariate hourly time series with clear daily and weekly seasonality.
Late-period windows carry the same periodic patterns as early-period ones, so the model can exploit
that regularity to score low validation loss without having learned to genuinely forecast.
The leaderboard revealed the true cost: experiment-1 achieved WAPE = 37.16 on the public leaderboard,
**worse** than the `seasonal_mean` baseline (WAPE = 34.41), despite appearing competitive locally.

---

## The Fix — Chronological Split by Series

The correct approach is to hold out the **last N timesteps of every series** and train only on the
earlier portion. This matches how the baseline evaluation code already works and guarantees that
every training window is strictly earlier in time than every validation window.

```python
# experiment-2 — FIXED
train_rows, val_rows = [], []
for sid, grp in feat_df.groupby("series_id", sort=False):
    grp = grp.sort_values("timestamp")
    train_rows.append(grp.iloc[:-EVAL_HORIZON])   # all but last 336 hours
    val_rows.append(grp.iloc[-EVAL_HORIZON:])      # last 336 hours per series

train_feat_df = pd.concat(train_rows, ignore_index=True)
val_feat_df   = pd.concat(val_rows,   ignore_index=True)

train_dataset = SlidingWindowDataset(train_feat_df, FEATURE_COLS)
val_dataset   = SlidingWindowDataset(val_feat_df,   FEATURE_COLS)
```

`EVAL_HORIZON = 336` is already set in the Config section — it matches the competition's required
forecast horizon, so the held-out window is the same size as what the model must predict at submission time.

### Why `shuffle=True` on the Train Loader Is Still Safe

`SlidingWindowDataset` pre-computes all `(X, y)` pairs before any DataLoader is created.
Each sample already encodes its full temporal context (a 168-hour lookback window and a 24-hour
target window). Shuffling only changes the **order in which complete, self-contained windows are
fed to the optimiser** — it does not reorder timesteps inside any window.
This is standard practice for sliding-window time series training and is beneficial for SGD
convergence. The val loader keeps `shuffle=False` for reproducibility.

---

## What Changed and What Stayed the Same

| Component | Experiment 1 | Experiment 2 |
|---|---|---|
| Train/val split | `random_split` — temporal leakage | Chronological per-series split |
| Val horizon | Random 10 % of all windows | Last 336 hours of every series |
| Train loader shuffle | `True` | `True` (unchanged — still correct) |
| Val loader shuffle | `False` | `False` (unchanged) |
| Feature set | 21 time + lag + rolling features | Identical |
| Model | LSTM (hidden=256, 2 layers) | Identical |
| Loss | HuberLoss (δ=1.0) | Identical |
| Optimiser | AdamW, lr=1e-3, cosine schedule | Identical |
| Epochs | 100 | 100 |

Everything except the split strategy is held constant so the effect of fixing the leakage
is isolated and measurable.

---

## How the Split Affects Dataset Sizes

With `feat_df` having ~4,152 usable rows per series after dropping the leading NaN rows
from lag_168 (168 rows dropped from 4,320 total):

| Subset | Rows per series | Sliding windows per series | Total windows (96 series) |
|---|---|---|---|
| Train | ~3,816 | ~3,625 | ~347,950 |
| Val | 336 | 145 | 13,920 |

The train set is slightly larger than experiment-1's random split (342,231 windows) because
the chronological approach does not waste any early-period data on validation.
The val set is smaller but represents a **true out-of-time evaluation** matching the
competition's test structure.

---

## Training Results — What Actually Happened

### Observed training curve (first 10 epochs)

```
Epoch   1/100  train=0.21650  val=0.25432  lr=1.00e-03  ← best
Epoch   2/100  train=0.16957  val=0.29051  lr=9.99e-04
Epoch   3/100  train=0.12661  val=0.33036  lr=9.98e-04
Epoch   4/100  train=0.09335  val=0.33112  lr=9.96e-04
Epoch   5/100  train=0.07500  val=0.33988  lr=9.94e-04
Epoch   6/100  train=0.06515  val=0.34151  lr=9.91e-04
Epoch   7/100  train=0.05994  val=0.34050  lr=9.88e-04
Epoch   8/100  train=0.05460  val=0.32503  lr=9.84e-04
Epoch   9/100  train=0.05183  val=0.32751  lr=9.80e-04
Epoch  10/100  train=0.04917  val=0.32991  lr=9.76e-04
```

Train loss drops steadily. Val loss **immediately rises after epoch 1 and never recovers**.
The best checkpoint is epoch 1 — the model is not learning to generalise to the held-out future window.

### Diagnosis — Why the Model Overfit Instantly

**This is not a bug in the split.** The honest chronological val set is working correctly — it is
now exposing a real generalisation failure that the random split was hiding.

Three compounding factors caused immediate overfitting:

**1. Model capacity is too large for the true distribution shift.**
The LSTM has 818,200 parameters (hidden=256, 2 layers). The val set contains only 145 windows per
series (13,920 total). With a 24× ratio of training windows to val windows, the model memorises
training-period patterns and achieves low train loss at the cost of failing on the out-of-time window.

**2. Learning rate is too aggressive.**
`lr=1e-3` with a cosine schedule causes large parameter updates in the first few epochs. The model
races to minimise training loss before the optimiser has a chance to find weights that also generalise.
In experiment-1 this was masked: the random val set shared the same temporal distribution as train,
so aggressive updates happened to reduce both losses together.

**3. The val window represents a genuine distribution shift.**
All 13,920 val windows are drawn from the final 336 hours of each series — the most recent portion
of the data. The model is trained on the first ~3,816 hours and must predict the last 336. Any slow
drift or regime change in the data (trend, seasonality amplitude, etc.) makes this harder than
randomly sampled in-distribution validation.

**Comparison to experiment-1:** In experiment-1 the val loss tracked train loss all the way down to
`0.01141` over 100 epochs, which appeared excellent. That was entirely an artefact of leakage — the
val set contained windows from throughout the time range, so it shared the same distribution as train.
The model was never asked to forecast anything it hadn't already seen.

---

## What Needs to Change

The chronological split is correct and must be kept. The model and training recipe need to be
adjusted to match the harder, honest evaluation.

### Required fixes

**Reduce model capacity** (Section 8):
```python
# before
model = LSTMForecaster(n_features=N_FEATURES, hidden=256, n_layers=2, dropout=0.1)

# after
model = LSTMForecaster(n_features=N_FEATURES, hidden=128, n_layers=2, dropout=0.3)
```
Halving hidden size cuts parameters from 818k to ~270k. Tripling dropout forces regularisation.

**Reduce batch size** (Section 1 Config):
```python
BATCH_SIZE = 64   # was 256
```
Large batches (256) converge to sharp minima in the loss landscape that generalise poorly — this
is a well-documented effect known as the generalisation gap. Smaller batches introduce gradient
noise that acts as implicit regularisation and pushes the optimiser toward flatter, wider minima.
With only 13,920 val windows, batch size 256 also produces only ~54 val batches per epoch,
making the val loss estimate coarse. Reducing to 64 gives ~217 val batches, a much more stable signal.
The T4 GPU can handle batch size 64 comfortably; training will be slower per epoch but converge better.

**Lower the learning rate and strengthen weight decay** (Section 1 Config):
```python
LR           = 3e-4   # was 1e-3
WEIGHT_DECAY = 1e-3   # was 1e-4
```
A slower learning rate lets the optimiser find flatter minima that generalise better across the
time boundary. Stronger L2 decay directly penalises large weights that overfit training patterns.

**Switch loss to L1** (Section 9):
```python
criterion = nn.L1Loss()   # was HuberLoss(delta=1.0)
```
The competition metric is WAPE — a weighted sum of absolute errors. Training with L1 aligns the
optimisation objective directly with what the leaderboard measures.

### Expected behaviour after fixes

With the above changes, val loss should decrease alongside train loss for the first 30–60 epochs
before plateauing or gently diverging. The best checkpoint should be saved somewhere in that range,
not at epoch 1. Train loss will not fall as fast, but the final model will generalise to the
out-of-time forecast window.

These changes are carried forward into **experiment-3**, which tests the full corrected recipe.

---

## Leaderboard Context

| Submission | WAPE (leaderboard) | Notes |
|---|---|---|
| seasonal_mean (baseline) | 34.408848 | Target to beat |
| experiment-1 | 37.158410 | Worse than baseline — leakage inflated local score |
| **experiment-2** | *(not submitted — overfit from epoch 1)* | Split fixed, but model/LR not adjusted |

---

## Key Takeaways

> **`random_split` on a pooled multi-series sliding-window dataset is a form of temporal leakage.**
> The fix is always a chronological split: hold out the latest timesteps of every series for
> validation and train strictly on the earlier portion.

> **A val loss that tracks train loss perfectly is a red flag, not a success.**
> Experiment-1's smoothly decreasing val curve was caused by leakage, not learning.
> Experiment-2's immediately rising val curve is honest — and points directly at what to fix next.
