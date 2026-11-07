# Experiment archive

The notebooks and notes from the 13 experiments, kept as they were run on Kaggle between May
and August 2026. The cleaned-up code is in [`src/loadforecast/`](../src/loadforecast/), and
the story is in [`docs/experiments.md`](../docs/experiments.md).

> [!NOTE]
> These files are the record, not a runnable pipeline. Paths point at Kaggle inputs
> (`/kaggle/input/...`). Checkpoints, prediction CSVs and out-of-fold files were left out to
> keep the repository small, so links to them from the notes do not resolve. The 5 final
> checkpoints are in the [v1.0.0 release](https://github.com/xitij27/known-covariate-forecasting/releases/tag/v1.0.0).

| Folder | Experiment | Result |
|---|---|---|
| [`experiment-1/`](experiment-1/) | LSTM, 24-hour autoregressive rollout | LB 37.16, worse than the seasonal mean (34.41) |
| [`experiment-2/`](experiment-2/) | Chronological train/validation split | Training loss falls while validation loss rises |
| [`experiment-3.1/`](experiment-3.1/) | Covariates and L1 loss; the diagnostic that ended the LSTM phase | Linear regression 21.0 against seasonal 31.4 |
| [`experiment-3.2/`](experiment-3.2/) | Direct regression on the known covariates | LightGBM 16.53 on one split |
| [`experiment-4/`](experiment-4/) | Honest CV, MAE loss, `series_id`, covariate context | CV 14.80 ± 0.64, LB 15.48 |
| [`experiment-5/`](experiment-5/) | Tuning, bias correction, ensembles, seeds | CV 14.73, LB 14.92 |
| [`experiment-6/`](experiment-6/) | XGBoost as a diversity arm | Rejected: residual r = 0.9955 |
| [`experiment-7/`](experiment-7/) | Seq2seq, gated MLP, blends | Blend LB 14.68, gated MLP LB 14.75 |
| [`experiment-8/`](experiment-8/) | TiDE | Rejected: CV 16.37 |
| [`experiment-9/`](experiment-9/) | Spike-weighted losses, Chronos-2 | Rejected: Chronos-2 fine-tuned CV 18.7 |
| [`experiment-10/`](experiment-10/) | Feature-engineering blocks | Rejected: best block gained 0.038 |
| [`experiment-11/`](experiment-11/) | BiGRU / BiLSTM encoder-decoder with cross-attention | CV 14.20, LB 14.77 after the final-fit window fix |
| [`experiment-12/`](experiment-12/) | Temporal Fusion Transformer | CV 14.72 ± 0.34, LB 14.36 (one seed) |
| [`experiment-13/`](experiment-13/) | 5-seed final fit; [`run-3/`](experiment-13/run-3/) trained the released checkpoints | LB 14.48 |

Notes written along the way:

- [`APPROACH.md`](APPROACH.md): the plan, rewritten as the findings came in.
- [`PITFALLS.md`](PITFALLS.md): every failure hit while building the TFT pipeline, its cause,
  and what now prevents it.
- [`IMPROVEMENT-FRONTIERS.md`](IMPROVEMENT-FRONTIERS.md): where the remaining error sits, and
  what was tried against it.
- [`explanation/`](explanation/): a first-person account of experiments 1 to 4, from the
  LSTMs to the reframe.
