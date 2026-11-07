# Experiment 12a — TFT run report

`EXP_ID=12a_tft` · generated 2026-08-16 23:08 UTC · **PREFLIGHT=False**

## Environment

- python `3.12.13` · torch `2.10.0+cu128` · lightning `2.6.5` · pytorch-forecasting `1.8.0`
- GPUs visible: 2 — Tesla T4
- jobs run: 11 · total GPU time 11.45 h
- wall clock across all jobs: 7.18 h (speedup 1.59x)

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
- `max_epochs` = `30`
- `patience` = `10`
- `EARLY_STOP_MODE` = `inner` · `SEEDS` = `[42]` · `N_FOLDS` = 3 · `EVAL_HORIZON` = 336
- model parameters: 963,376

## Premise checks & throughput

- loss on ORIGINAL target scale: **None** (original nan vs normalised nan, reported val_loss nan)
- causal vs non-causal mean |Δprediction|: **nan**
- 78 steps/epoch · 89.4 s per measured epoch · 9,984 train windows

## §7 config probe

```
         arm       wape  best_epoch  val_loss    minutes  windows  steps_ep
0       crop  17.047467          10  1.431543  91.072884     9984       312
1       base  17.243321           8  1.447989  76.163862     9312       291
2  noncausal  17.383893           6  1.459794  67.753194     9312       291
3   quantile  17.660774           7  0.847682  77.563405     9984       312
```

## §8 cross-validation

```
           wape  mean_resid     rows
fold                                
0     14.569186    0.852717  32256.0
1     15.110226    0.504285  32256.0
2     14.485609    0.493111  32256.0
```

**CV WAPE 14.722 ± 0.339** (pooled 14.701, 96,768 rows)

Reference: exp-11a GRU 14.203 ± 0.599 · exp-7c blend 14.47 · our LB best 14.682

Per job:
```
         tag  seed       wape  best_epoch  epochs_run  minutes
0  cv_f0_s42    42  14.569186          11          21     95.4
1  cv_f1_s42    42  15.110226          10          20    103.3
2  cv_f2_s42    42  14.485609           8          18    103.2
```

## §9 diagnostics

Worst 15 / best 5 series (exp-11a GRU spread: 7.01 → 25.47):
```
                wape
series_id           
unit_073   25.415656
unit_066   22.447689
unit_058   22.001903
unit_056   20.668001
unit_003   19.718452
unit_071   19.509670
unit_039   19.472572
unit_063   19.230977
unit_079   18.941651
unit_061   18.834749
unit_022   18.693025
unit_021   18.665238
unit_031   18.616918
unit_011   18.137408
unit_040   18.039709
unit_013   10.458327
unit_091   10.289004
unit_015    9.189264
unit_043    9.004048
unit_069    7.307572
```

By forecast lead:
```
               wape
lead_h             
1-24h     13.405314
25-72h    16.563303
73-168h   14.423901
169-336h  14.603649
```

Top-5% target rows: WAPE 23.37, mean residual -5.68 (exp-7b blend reference: −5.60)

Overall mean residual -0.617

Residual correlation vs prior models:
- vs `gru`: r = **-0.9731** [STRONG] (96,768 rows)
- vs `7c`: r = **-0.9716** [STRONG] (96,768 rows)

## §10 variable selection


**static**
```
          variable  importance
51       series_id    0.581925
52   target_center    0.214033
53    target_scale    0.154605
54  encoder_length    0.049438
```

**encoder**
```
                              variable  importance
25           network_pressure_forecast    0.172600
26             queue_pressure_forecast    0.147112
27  service_irregularity_risk_forecast    0.071369
28           unit_reliability_forecast    0.064732
29                   maintenance_known    0.058529
30                             dow_cat    0.044436
31                          shock_risk    0.040482
32       network_pressure_forecast_vel    0.038359
33                     demand_forecast    0.037923
34                              target    0.031179
```

**decoder**
```
                    variable  importance
0  network_pressure_forecast    0.123062
1  unit_reliability_forecast    0.103366
2        event_load_forecast    0.092153
3            demand_forecast    0.069905
4          maintenance_known    0.066007
5                 is_weekend    0.058704
6                   hour_cos    0.058371
7                    dow_cat    0.058158
8         workload_intensity    0.051439
9    queue_pressure_forecast    0.035462
```

`target_history` share of encoder importance: **0.031** (TiDE, rejected in exp-8: 0.42–0.49)

## §11 final fit & submission

- last training decoder target: **4319** (must be 4319)
- last training encoder end: 3984 → inference encoder gap **336h** (structural minimum 336h)
- epochs (fixed, from CV median): 10 · seeds averaged: 1

Submission `validation_predictions_12a_tft.csv` — 32,256 rows, 2023-06-30 00:00:00 → 2023-07-13 23:00:00
```
         prediction
count  32256.000000
mean      10.175216
std        5.286324
min        0.796875
25%        6.156250
50%        9.375000
75%       13.375000
max       39.500000
```

Blend submission written: `validation_predictions_12b_blend.csv` (32,256 rows)

## §13 tuning

*(tuning not run)*

## Job timings

```
                tag  minutes  epochs  sec/epoch       wape
1         cv_f1_s42    103.3      20      309.3  15.110226
2         cv_f2_s42    103.2      18      343.6  14.485609
0         cv_f0_s42     95.4      21      272.3  14.569186
5        probe_crop     91.1      20      272.8  17.047467
7    probe_quantile     77.6      17      273.3  17.660774
4        probe_base     76.2      18      253.4  17.243321
3         final_s42     69.2      10      414.6        NaN
6   probe_noncausal     67.8      16      253.6  17.383893
8      warmup_bs128      1.6       1       89.4  31.219830
10      warmup_bs64      1.0       1       48.7  32.001668
9       warmup_bs32      0.6       1       28.9  32.721391
```