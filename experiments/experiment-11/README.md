# Experiment 11 — GRU & LSTM Encoder-Decoder with Cross-Attention

## Status

| Sub-experiment | Status | CV WAPE | LB WAPE | Submission name |
|---|---|---|---|---|
| 11a — GRU | Complete | **14.203 ± 0.599** | **14.773** | bigru_attn_aug_multiseed2 |
| 11b — LSTM | Complete | 14.411 ± 0.609 | not submitted standalone | — |
| 11c — Blend | **Complete** (computed in exp-12 §12) | **13.988** (OOF) | pending | `validation_predictions_12b_blend.csv` |

Current leaderboard best (our team): **14.682** (`lgbm_mlp_ensemble`, rank 27).
Leaderboard overall best: **13.053** (external).

---

## Motivation

The professor explicitly required LSTM and GRU architectures. The "feature ceiling" conclusion
from experiments 6–10 was premature — the leaderboard best (13.05) is substantially below our
own best (14.68), meaning addressable signal remains. This experiment builds a proper
encoder-decoder RNN with the following improvements over exp-7a-seq2seq (which scored 16.30 LB):

| Dimension | exp-7a-seq2seq | Experiment 11 |
|---|---|---|
| Encoder | Bi-LSTM, 336h lookback, no attention | Bi-GRU / Bi-LSTM, 168h base lookback |
| Attention | None — final hidden state only | Bahdanau cross-attention (MultiheadAttention, 4 heads) |
| Decoder input | Future covariates only | Future covariates + velocity features + series embedding |
| Velocity features | None | First backward diff of top-3 domain covariates (block-C) |
| Augmentation | None | Feature dropout + subsequence crop + hard-series oversample |
| Cell comparison | LSTM only | GRU (11a) vs LSTM (11b) explicitly compared |
| Seeds | 3 | 5 |
| Ablation | None | 4-variant structured ablation (11a only) |

### Augmentations from aug/dlinear branch

Teammate branch (`fabianzwinge/multivariate_timeseries_forecasting`, aug/dlinear) tested DLinear
+ 1-layer LSTM with 10 augmentation strategies. Their absolute WAPEs (~29) are not comparable
because they used autoregressive rollout. Three augmentation ideas transfer directly:

| Augmentation | Their finding | Adopted as |
|---|---|---|
| Feature dropout (p=0.1, per covariate column) | Best single aug for LSTM | Applied to encoder inputs at training time only |
| Subsequence crop: lookback ~ Uniform[168, 336] | Second-best; reduces sensitivity to lookback length | Per-sample random lookback, zero-pad to 336h |
| Hard-series oversample: 3× weight on worst series | Moderate tail gain | 3× for worst-20 series by OOF WAPE |

---

## Architecture

### Shared design (11a GRU and 11b LSTM identical except cell type)

```
Input at each encoder step (38-dim):
  past_target_norm  (1)
  domain_covariates (11)   — workload_intensity, demand_forecast, upstream_quality_forecast,
                              promotion_intensity, shock_risk, maintenance_known,
                              unit_reliability_forecast, queue_pressure_forecast,
                              network_pressure_forecast, event_load_forecast,
                              service_irregularity_risk_forecast
  time_features     (7)    — hour_sin/cos, dow_sin/cos, doy_sin/cos, is_weekend
  velocity_features (3)    — first backward diff of queue_pressure, network_pressure, shock_risk
  series_embedding  (16)   — Embedding(96, 16), one learned vector per unit

Encoder: Bidirectional GRU (11a) / LSTM (11b)
  Layers:    2
  Hidden:    128 per direction → 256 per step
  Lookback:  168h base (crop augmentation exposes 168–336h during training)
  Output:    enc_out (B, L, 256), final hidden h_enc (4, B, 128)

Decoder initialisation:
  GRU:  h_dec0 = Linear(256→128)(cat[h_fwd, h_bwd])          — projected to (1, B, 128)
  LSTM: h_dec0 = Linear(256→128)(cat[h_fwd, h_bwd])
        c_dec0 = Linear(256→128)(cat[c_fwd, c_bwd])           — separate cell state projection

Input at each decoder step (37-dim):
  future_domain_covariates (11)
  future_velocity_features  (3)   — stitched from last training step at boundary
  time_features             (7)
  series_embedding          (16)

Decoder: Unidirectional GRU (11a) / LSTM (11b)
  Layers:  1
  Hidden:  128
  Horizon: 336 steps (full future window, no autoregressive rollout)

Cross-attention (Bahdanau via nn.MultiheadAttention):
  Queries: dec_out          (B, H, 128)
  Keys:    proj_enc(enc_out)(B, L, 128)   — Linear(256→128, no bias)
  Values:  same as keys
  Output:  context          (B, H, 128)

Head: Linear(128→1) applied to (dec_out + context)  →  (B, H) predictions
```

**Parameter count:** 622,977 (GRU) / ~680,000 (LSTM, adds cell state projections).

**Velocity computation:**
- Training encoder: `vel[t] = cov[t] - cov[t-1]`, causal, `vel[0] = 0`. Precomputed per series.
- Training decoder: same array, sliced at `dec_start`. No per-sample sklearn calls.
- Validation decoder: `vel[0] = val_cov[0] - train_cov[-1]` (stitched across boundary).
- Normalised with `RobustScaler` fit on training-fold covariates.

---

## Training configuration (final, after speedup fixes)

```
Optimizer:          AdamW, lr=1e-3, weight_decay=1e-4
LR schedule:        CosineAnnealingLR, T_max=50, eta_min=1e-5
Loss:               MAE (L1Loss)
Batch size:         512
Stride:             24h (one window per day)
Epochs:             50 max, early stop patience=7 on CV fold WAPE
Seeds:              5 — [42, 7, 13, 99, 1234]; predictions averaged
Gradient clip:      max_norm=1.0
RNN dropout:        0.2 (between layers only)
Mixed precision:    torch.cuda.amp.GradScaler (AMP, CUDA only)
DataLoader:         num_workers=4, pin_memory=True, persistent_workers=True
torch.compile:      enabled (try/except; ~20% speedup on PyTorch 2.x)
```

**CV protocol:** 3-fold rolling-origin. Fold boundaries:
- Fold 0: train [0..3312), test [3312..3648)
- Fold 1: train [0..3648), test [3648..3984)
- Fold 2: train [0..3984), test [3984..4320)

**Final training (§11):** Normalizers fit on all 4320 training hours.
Dataset built with `end_idx_override=4320` so windows extend up to start=3624
(enc_end=3960, dec_end=4296). At inference, encoder reads hours 4152–4320.

---

## Bugs found and fixed

### Bug 1 — fold_k=0 in final training (CRITICAL)

**Symptom:** 11a first LB submission scored **17.912 WAPE** despite CV of 14.313.

**Root cause:** `Seq2SeqDataset(fold_k=0)` uses `fold_train_end_idx(0) = 3312` as `end_idx`.
This gives `max_start = 3312 - 336 - 336 = 2640`. The model's encoder never saw hours beyond
2952 during training. At inference the encoder reads hours 4152–4320 — a 1200h extrapolation
into unseen territory.

**Fix:** Added `end_idx_override=T_PER_SERIES` (= 4320) to the final training dataset call.
Max encoder end during training advances to 3960. Distribution gap at inference reduced from
~1200h to ~192h.

**Impact:** LB improved from **17.912 → 14.773** on resubmission.

### Bug 2 — per-sample sklearn call in `__getitem__` (performance)

**Symptom:** GPU utilisation was low; each training step bottlenecked on CPU.

**Root cause:** The original dataset called `vel_scaler.transform()` inside `__getitem__` for
every sample — a small sklearn call per sample per epoch inside the DataLoader worker.

**Fix:** Precompute the full normalised velocity array for each series in `__init__`. `__getitem__`
only slices `arr["vel"][dec_start:dec_end]`. Equivalent result, zero per-sample sklearn calls.

### Bug 3 — GPU underutilisation (performance)

**Causes:** `num_workers=0` (GPU starved waiting on CPU), `BATCH_SIZE=256` (small batches),
`STRIDE=6` (4× larger dataset than needed), `PATIENCE=10` (wasted epochs).

**Fixes:** `N_WORKERS=4`, `BATCH_SIZE=512`, `STRIDE=24`, `PATIENCE=7`.

**Combined speedup:** ~4× per epoch, ~3.5× end-to-end per fold×seed run
(600–800s → 100–235s per fold×seed).

---

## 11a — GRU Results

### Ablation (fold 0, seed 42, 50 epochs max)

| Config | aug | velocity | WAPE (fold 0) | Time |
|---|---|---|---|---|
| base | ✗ | ✗ | 15.372 | 114s |
| +velocity | ✗ | ✓ | 15.279 | 98s |
| +aug | ✓ | ✗ | 13.840 | 189s |
| **+full** | **✓** | **✓** | **13.838** | 192s |

**Winner: `+full`** (use_aug=True, use_velocity=True) by 0.002 WAPE margin over `+aug`.

**Interpretation:** Augmentation drives the improvement (+1.53 WAPE over base). Velocity alone
gives 0.09 WAPE improvement, which is within noise at a single fold×seed. The 0.002 gap between
`+aug` and `+full` is not meaningful — the `use_velocity` flag was a dead variable in
`train_one_config` (not forwarded to the dataset); the model used velocity in all configs.
Both `+aug` and `+full` ran identical models. The CV ran with `use_aug=True, use_velocity=True`.

### Full CV (3 folds × 5 seeds)

| Fold | WAPE | Seed spread |
|---|---|---|
| 0 | 13.572 | 0.183 |
| 1 | 15.009 | 0.279 |
| 2 | 14.029 | 0.109 |
| **mean** | **14.203** | |
| **std** | **0.599** | |

Per-seed fold-0 results: 13.748, 13.777, 13.831, 13.837, 13.931.

**Residual correlation vs exp-7c OOF:** r = **0.9684** (limited diversity).

### Per-series WAPE (worst 10 / best 5)

| Series | WAPE |
|---|---|
| unit_073 | 25.47 |
| unit_058 | 21.72 |
| unit_056 | 20.88 |
| unit_066 | 20.85 |
| unit_071 | 19.66 |
| unit_079 | 19.36 |
| unit_039 | 18.86 |
| unit_003 | 18.76 |
| unit_022 | 18.58 |
| unit_028 | 18.05 |
| ... | |
| unit_043 | 9.35 |
| unit_015 | 8.81 |
| unit_069 | 7.01 |

Overall: min=7.01, max=25.47, mean=14.52.

### Leaderboard submission

| Run | CV WAPE | LB WAPE | Notes |
|---|---|---|---|
| First run (fold_k=0 bug) | 14.313 | 17.912 | fold_k=0 without end_idx_override |
| **Second run (fixed)** | **14.203** | **14.773** | end_idx_override=T_PER_SERIES |

CV-LB gap (fixed run): **0.570** WAPE. Within normal range; attributable to residual
distribution shift at the training/inference boundary (~192h encoder gap) and fold-1
structural difficulty.

Current leaderboard position: **rank 36** (vs our best at rank 27 = 14.682).
GRU alone trails current best by **0.091 WAPE**.

---

## 11b — LSTM Results

### Full CV (3 folds × 5 seeds)

| Fold | WAPE | Seed spread |
|---|---|---|
| 0 | 13.817 | 0.168 |
| 1 | 15.248 | — |
| 2 | 14.169 | — |
| **mean** | **14.411** | |
| **std** | **0.609** | |

Per-seed fold-0 results: 14.065, 14.080, 14.096, 14.186, 14.233.

**Residual correlation vs exp-7c OOF:** r = **0.9720** (limited diversity).

### GRU vs LSTM comparison

| Metric | GRU (11a) | LSTM (11b) | Delta |
|---|---|---|---|
| CV mean WAPE | **14.203** | 14.411 | +0.208 |
| CV std | **0.599** | 0.609 | +0.010 |
| Fold 0 | **13.572** | 13.817 | +0.245 |
| Fold 1 | **15.009** | 15.248 | +0.239 |
| Fold 2 | **14.029** | 14.169 | +0.140 |
| r vs 7c | **0.9684** | 0.9720 | LSTM more correlated |

**Conclusion:** GRU outperforms LSTM consistently across all three folds. LSTM is also more
correlated with 7c, meaning it adds less diversity to any blend. GRU is the preferred model.

---

## 11c — Blend (complete — run inside exp-12 §12)

**Result: OOF WAPE 13.988**, weights `gru 0.6194 / 7c 0.3806`, on all 96,768 OOF rows.
Cross-fitting the weights (fit on two folds, score the held-out fold) gives 13.989 — the
weights are stable, not overfit. The submission is
`experiment-12/run 4 files/validation_predictions_12b_blend.csv`.

This is the best OOF number in the project: gru solo 14.152, 7c solo 14.419, blend 13.988
(−0.164 on the better component). The prediction below was accurate — a 2-way GRU+7c blend
won, and TFT (added as a third arm) received exactly 0.0 weight.

### Original plan (kept for the record)

**Inputs:**
- `oof_predictions_11a_gru.csv` — GRU OOF (CV WAPE 14.203)
- `oof_predictions_11b_lstm.csv` — LSTM OOF (CV WAPE 14.411)
- `oof_predictions_7b_blend.csv` — exp-7c OOF

**Method:** `scipy.optimize.minimize` on unit simplex, minimising OOF WAPE.

**Combinations to test:** 3-way (GRU+LSTM+7c), GRU+7c, LSTM+7c.

**Expected outcome:** GRU+7c 2-way will likely win. LSTM weight expected near zero given
r(LSTM, 7c) = 0.972 and LSTM's weaker standalone CV. With r(GRU, 7c) = 0.9684, blend benefit
over GRU alone is limited — estimated 0.1–0.2 WAPE improvement on OOF, translating to an
expected LB around **14.55–14.65**, which should beat the current best (14.682).

---

## Structural observations across all models

**Fold 1 is consistently the hardest fold** across every experiment in this project. GRU scores
15.009 and LSTM scores 15.248 on fold 1, vs 13.5–13.8 on fold 0 and 14.0–14.2 on fold 2. This
is not a model artifact — it appears in LGBM, MLP, seq2seq, and attention-based models alike.
The test window for fold 1 (hours 3648–3984) covers a period with harder-to-predict dynamics.

**unit_073, unit_058, unit_056, unit_066** are consistently the hardest series. Hard-series
oversampling (3× weight) improved their representation but did not close the gap to the median
series. These series likely have structural properties (high volatility or weaker covariate
correlation) that cannot be remedied by sampling alone.

**Augmentation is the single largest driver.** +aug over base gives −1.53 WAPE on fold 0
(15.372 → 13.840). Cross-attention and shorter lookback contribute but are hard to isolate
without a separate ablation since they were part of the base config.

---

## Files

```
experiment-11/
├── README.md
├── kp-experiment-11a-gru.ipynb          — GRU ablation + full CV + §11 final training
├── kp-experiment-11b-lstm.ipynb         — LSTM full CV + §9 final training
├── kp-experiment-11c-blend.ipynb        — OOF-optimised blend + submission CSV
├── ablation_11a_gru.csv                 — ablation results (4 configs × fold 0)
├── oof_predictions_11a_gru.csv          — GRU OOF across all 3 folds
├── oof_predictions_11b_lstm.csv         — LSTM OOF across all 3 folds
└── validation_predictions_11c.csv       — final submission (pending 11c run)
```

Inputs from prior experiments:
- `experiment-7/oof_predictions_7b_blend.csv` — 7c OOF for residual correlation and blending
- `experiment-7/val_components_7c.csv` — 7c validation predictions for 11c blend
