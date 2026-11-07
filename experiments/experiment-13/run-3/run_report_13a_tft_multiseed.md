# Run report — 13a_tft_multiseed

`EXP_ID=13a_tft_multiseed` · generated 2026-08-17 10:36 UTC · **PREFLIGHT=False**

## Environment

- python `3.12.13` · torch `2.10.0+cu128` · lightning `2.6.5` · pytorch-forecasting `1.8.0`
- GPUs visible: 2 — Tesla T4
- jobs run: 8 · total GPU time 5.84 h
- wall clock across all jobs: 3.52 h (speedup 1.66x)

## Configuration

- `min_encoder` = `168`
- `max_encoder` = `336`
- `randomize_length` = `[0.2, 0.05]`
- `stride` = `24`
- `hidden_size` = `96`
- `hidden_continuous_size` = `32`
- `attention_heads` = `4`
- `lstm_layers` = `1`
- `dropout` = `0.15`
- `causal` = `False`
- `share_vsn` = `False`
- `loss_name` = `mae`
- `learning_rate` = `0.001`
- `weight_decay` = `0.0001`
- `optimizer` = `adamw`
- `grad_clip` = `0.5`
- `batch_size` = `32`
- `accum` = `4`
- `max_epochs` = `60`
- `patience` = `10`
- `EARLY_STOP_MODE` = `inner` · `SEEDS` = `[42, 7, 13, 99, 1234]` · `N_FOLDS` = 3 · `EVAL_HORIZON` = 336
- model parameters: 963,376

## Premise checks & throughput

- loss on ORIGINAL target scale: **True** (original 2.7477 vs normalised 0.5261, reported val_loss 2.7477)
- causal vs non-causal mean |Δprediction|: **0.03379**
- 312 steps/epoch · 27.3 s per measured epoch · 9,984 train windows

## §7 config probe

*(config probe not run)*

## §8 cross-validation

*(cross-validation not run)*

## §9 diagnostics

*(cross-validation not run)*

## §10 variable selection

*(interpretation not run)*

## §11 final fit & submission

- last training decoder target: **4319** (must be 4319)
- last training encoder end: 3984 → inference encoder gap **336h** (structural minimum 336h)
- epochs (fixed, from CV median): 10 · seeds averaged: 5

Submission `validation_predictions_13a_tft_multiseed.csv` — 32,256 rows, 2023-06-30 00:00:00 → 2023-07-13 23:00:00
```
         prediction
count  32256.000000
mean      10.397417
std        5.266882
min        0.942969
25%        6.406250
50%        9.587500
75%       13.600000
max       36.599998
```

## §13 tuning

*(tuning not run)*

## Job timings

```
            tag  minutes  epochs  sec/epoch       wape
1     final_s13     69.5      10      415.9        NaN
4     final_s99     69.5      10      415.9        NaN
3      final_s7     69.4      10      415.7        NaN
2     final_s42     69.4      10      415.7        NaN
0   final_s1234     69.2      10      414.8        NaN
5  warmup_bs128      1.6       1       89.3  31.214735
7   warmup_bs64      0.9       1       47.7  31.984521
6   warmup_bs32      0.6       1       27.3  32.721391
```