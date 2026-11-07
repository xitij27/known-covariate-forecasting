# Phase 8 — TiDE (Time-series Dense Encoder), proper-tier training

Phase 7 shipped two submissions that **lead on WAPE** (ensemble 14.68, gated MLP
14.75) but trail on the L2 metrics (MSE/RMSE/MAPE). Phase 8 asks one question:
*can a different architecture close the L2 gap and challenge for Overall rank 1?*
The model under test is **TiDE** (Das et al. 2023, *Long-term Forecasting with
TiDE*) — a pure-MLP seq2seq encoder, 7.88M params.

## What this run is

A **proper-tier retrain** that replaces an earlier diagnostic-tier run (CV-WAPE
16.4) suspected of under-fitting — all 9 (fold × seed) runs had converged to
inner-val L1 in a narrow 0.305–0.326 band, the classic shallow-attractor
signature for a 7.88M-param model fed only ~276K samples. This version scales
training into line with model capacity (**~3.7M samples seen, 13×**):

| Hyperparameter | Diagnostic v1 | Proper v2 (this) |
|---|---|---|
| `WINDOWS_PER_SERIES_PER_EPOCH` | 96 | **384** |
| `EPOCHS` | 30 | **100** |
| early-stop patience | 5 | **15** |
| `DEC_PER_STEP` | 8 | **16** |
| `N_DEC_BLOCKS` | 2 | **3** |
| `DROPOUT` | 0.2 | **0.3** |
| `BATCH` | 64 | **128** |
| LR schedule | cosine `T_max=30` | **5-epoch warmup → cosine `T_max=100`** |

3-fold rolling-origin CV × 3 seeds [42, 101, 202]. LGBM-MAE anchor in the same
notebook for reference.

**Pre-registered decision:** beat the 7e gated-MLP WAPE (14.59) **and** beat its
OOF MSE (8.30) to be a real rank-1 candidate; `r < 0.85` vs the MLP would make
TiDE a blend candidate even if solo is worse.

## Result — scaling did not move it

| fold | WAPE | rows |
|---|---|---|
| 0 | 16.35 | 32,256 |
| 1 | 17.43 | 32,256 |
| 2 | 15.47 | 32,256 |
| **overall** | **16.37** | 96,768 |

(computed from `oof_predictions_8a_tide.csv`.)

The proper-tier retrain lands at **CV-WAPE 16.37 — statistically identical to the
under-trained v1 (16.4).** The 13× increase in training scale moved the number by
noise. v1 was therefore **not** under-trained in any way that scaling fixes; TiDE
on this fixed feature set simply sits ~1.8 WAPE above the gated MLP. It does not
clear the decision bar on either WAPE or MSE.

**Feature importance** (`feature_importance_8a_tide.csv`, `past_proj` column
norms) confirms the v1 diagnosis persists: `target_history` is the single
strongest input (~0.42–0.49 across folds/seeds), with `queue_pressure_forecast`
and `network_pressure_forecast` next — i.e. TiDE leans autoregressive rather than
fully exploiting the future covariates.

## Verdict

**TiDE rejected.** Swapping in a fundamentally different architecture on the fixed
feature set does not close the L2 gap and is worse on WAPE. This is the result
that motivates the **Phase 9 headline** — *the ceiling is the feature set, not the
model class* — and redirects work away from architecture search. See
[`../experiment-9/README.md`](../experiment-9/README.md) and
[`../IMPROVEMENT-FRONTIERS.md`](../IMPROVEMENT-FRONTIERS.md).

## Files

```
experiment-8/
├── README.md                          ← this file
├── kp-experiment-8a-tide.ipynb        ← TiDE proper-tier CV (scaffold; outputs cleared)
└── feature_importance_8a_tide.csv     ← per-(fold,seed) past_proj column norms
```

`oof_predictions_8a_tide.csv` (the OOF used for the WAPE/MSE/r numbers above) is a
large regenerable artifact and is git-ignored.
