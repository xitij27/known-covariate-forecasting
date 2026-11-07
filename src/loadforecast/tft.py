"""Temporal Fusion Transformer, trained and run with pytorch-forecasting (experiments 12-13).

The model reads up to 336 hours of history (target and covariates) and the known covariates
for the 336-hour block ahead, and predicts that whole block in one pass. It is trained with
MAE on the original target scale, which matches the WAPE metric.

The shipped model is five of these trained with different seeds on the full history for a
fixed 10 epochs (the median best epoch from cross-validation), with predictions averaged.

Invariants are asserted, not assumed. Two of them cost this project the most:
  * Training windows must reach the end of the history. A final fit whose windows stopped
    ~1200 h early read an unfamiliar encoder region at prediction time: leaderboard
    WAPE 17.91 -> 14.77 on the fix alone (experiment 11, "Bug 1").
  * No training target may sit at or beyond the training cut-off.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("LIGHTNING_DISABLE_APP_TIPS", "1")

import torch  # noqa: E402
from lightning.pytorch import Trainer, seed_everything  # noqa: E402
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint  # noqa: E402
from pytorch_forecasting import (  # noqa: E402
    GroupNormalizer,
    TemporalFusionTransformer,
    TimeSeriesDataSet,
)
from pytorch_forecasting.metrics import MAE  # noqa: E402

from .cv import N_FOLDS, fold_bounds  # noqa: E402
from .data import DOMAIN_COLS, EVAL_HORIZON, Split  # noqa: E402
from .features import (  # noqa: E402
    TIME_REAL_COLS,
    add_time_features,
    add_velocities,
    impute_covariates,
)
from .metrics import wape  # noqa: E402

for _name in ("lightning", "lightning.pytorch", "lightning_fabric", "pytorch_lightning"):
    logging.getLogger(_name).setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Attribute '(loss|logging_metrics)' is an instance.*")
warnings.filterwarnings("ignore", message=".*does not have many workers.*")
warnings.filterwarnings("ignore", message=".*isinstance\\(treespec, LeafSpec\\).*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# The three covariates with the strongest signal also enter as first differences.
VELOCITY_COLS = ["queue_pressure_forecast", "network_pressure_forecast", "shock_risk"]
TIME_CAT_COLS = ["hour_cat", "dow_cat"]


@dataclass
class TFTConfig:
    """Defaults are the configuration of the shipped ensemble (experiment 13, run 3)."""

    horizon: int = EVAL_HORIZON
    min_encoder: int = 168
    max_encoder: int = 336
    # pytorch-forecasting's encoder-length augmentation: with probability 0.2, crop the
    # encoder (at least 5% of max length). Recorded as the "crop" arm in experiment 12.
    randomize_length: tuple[float, float] | None = (0.2, 0.05)
    stride: int = 24  # one training window per series per day
    hidden_size: int = 96
    hidden_continuous_size: int = 32
    attention_heads: int = 4
    lstm_layers: int = 1
    dropout: float = 0.15
    # Non-causal: decoder step t may attend to later decoder steps. Legitimate because every
    # decoder input is a known covariate; no target enters the decoder.
    causal_attention: bool = False
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 0.5
    batch_size: int = 32
    accumulate: int = 4  # effective batch 128
    eval_batch: int = 32
    max_epochs: int = 60
    patience: int = 10
    final_epochs: int = 10  # median best epoch across the CV folds
    seeds: list[int] = field(default_factory=lambda: [42, 7, 13, 99, 1234])
    num_workers: int = 0


def accelerator() -> tuple[str, str]:
    """(accelerator, precision). fp16 on a GPU without bf16 (the T4s used), fp32 on CPU."""
    if torch.cuda.is_available():
        return "gpu", "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
    return "cpu", "32-true"


@contextmanager
def _quiet():
    if os.environ.get("LOADFORECAST_VERBOSE"):
        yield
        return
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        yield


# ---------------------------------------------------------------------------------------------
# Data


def prepare_panel(split: Split) -> tuple[pd.DataFrame, int]:
    """One hourly grid per series over history + forecast block, with a shared `time_idx`.

    The forecast rows carry a placeholder target of 0.0. It is never a label: the decoder
    target is ignored at prediction time and no training window reaches into that block
    (asserted in `training_dataset`). Returns the panel and `t_train`, the first forecast hour.
    """
    history = add_time_features(
        impute_covariates(split.history, DOMAIN_COLS, backfill=True), categorical=True
    )
    future = add_time_features(
        impute_covariates(split.future, DOMAIN_COLS, backfill=True), categorical=True
    )
    panel = pd.concat(
        [history.assign(is_future=False), future.assign(is_future=True, target=0.0)],
        ignore_index=True,
    ).sort_values(["series_id", "timestamp"], ignore_index=True)

    t0 = panel["timestamp"].min()
    panel["time_idx"] = ((panel["timestamp"] - t0).dt.total_seconds() // 3600).astype(int)
    panel, _ = add_velocities(panel, VELOCITY_COLS)
    panel["series_id"] = panel["series_id"].astype(str)
    panel["target"] = panel["target"].astype(np.float32)

    counts = panel.groupby("series_id").size()
    if counts.nunique() != 1:
        raise ValueError(f"ragged panel: series lengths {sorted(counts.unique())}")
    gaps = panel.groupby("series_id")["time_idx"].apply(lambda s: s.diff().dropna().ne(1).sum())
    if int(gaps.sum()):
        raise ValueError("history and forecast block do not form a contiguous hourly grid")
    if panel[DOMAIN_COLS].isna().any().any():
        raise ValueError("NaNs survived covariate imputation")
    t_train = int(panel.loc[~panel["is_future"], "time_idx"].max()) + 1
    return panel, t_train


def _known_reals() -> list[str]:
    return ["time_idx"] + DOMAIN_COLS + [f"{c}_vel" for c in VELOCITY_COLS] + TIME_REAL_COLS


def training_dataset(panel: pd.DataFrame, cfg: TFTConfig, train_end: int) -> TimeSeriesDataSet:
    """Windows whose targets all lie before `train_end`, one per series per `stride` hours."""
    randomize = tuple(cfg.randomize_length) if cfg.randomize_length else None
    if cfg.min_encoder == cfg.max_encoder:
        randomize = None
    ds = TimeSeriesDataSet(
        panel[panel["time_idx"] < train_end],
        time_idx="time_idx",
        target="target",
        group_ids=["series_id"],
        min_encoder_length=cfg.min_encoder,
        max_encoder_length=cfg.max_encoder,
        min_prediction_length=cfg.horizon,
        max_prediction_length=cfg.horizon,
        static_categoricals=["series_id"],
        time_varying_known_categoricals=TIME_CAT_COLS,
        time_varying_known_reals=_known_reals(),
        time_varying_unknown_reals=["target"],
        target_normalizer=GroupNormalizer(groups=["series_id"], transformation=None),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=cfg.min_encoder != cfg.max_encoder,
        randomize_length=randomize,
        allow_missing_timesteps=False,
    )
    if cfg.stride > 1:
        # Anchor the stride grid on the LAST valid window and step backwards, so the window
        # ending at `train_end` always survives. Also keep one row per decoder start: with
        # min_encoder < max_encoder the library emits one row per allowed encoder length
        # (~12x as many windows); randomize_length already does the cropping per sample.
        decoded = ds.decoded_index.reset_index(drop=True)
        length = ds.index["sequence_length"].to_numpy()
        starts = decoded["time_idx_first_prediction"].to_numpy()
        keyed = pd.DataFrame({"g": decoded["series_id"].astype(str), "t": starts, "s": length})
        longest = keyed.groupby(["g", "t"])["s"].transform("max").to_numpy() == length
        on_grid = ((starts.max() - starts) % cfg.stride) == 0
        chosen = keyed.loc[longest & on_grid]
        keep = np.zeros(len(keyed), dtype=bool)
        keep[chosen.index[~chosen.duplicated(["g", "t"])]] = True
        ds = ds.filter(lambda _index: keep)

    last_target = int(ds.decoded_index["time_idx_first_prediction"].max()) + cfg.horizon - 1
    if last_target != train_end - 1:
        raise AssertionError(
            f"training windows end at {last_target}, not at the cut-off {train_end - 1}"
        )
    return ds


def prediction_dataset(
    train_ds: TimeSeriesDataSet, panel: pd.DataFrame, decoder_start: int, n_series: int
) -> TimeSeriesDataSet:
    """One window per series whose decoder starts at `decoder_start`."""
    ds = TimeSeriesDataSet.from_dataset(
        train_ds,
        panel[panel["time_idx"] < decoder_start + train_ds.max_prediction_length],
        predict=True,
        stop_randomization=True,
    )
    starts = ds.decoded_index["time_idx_first_prediction"].unique()
    if len(starts) != 1 or int(starts[0]) != decoder_start:
        raise AssertionError(f"prediction window misaligned: {starts} != {decoder_start}")
    if len(ds) != n_series:
        raise AssertionError(f"expected {n_series} prediction windows, got {len(ds)}")
    return ds


def _loader(ds: TimeSeriesDataSet, cfg: TFTConfig, *, train: bool):
    return ds.to_dataloader(
        train=train,
        batch_size=cfg.batch_size if train else cfg.eval_batch,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )


# ---------------------------------------------------------------------------------------------
# Model


def build_model(train_ds: TimeSeriesDataSet, cfg: TFTConfig) -> TemporalFusionTransformer:
    _, precision = accelerator()
    return TemporalFusionTransformer.from_dataset(
        train_ds,
        hidden_size=cfg.hidden_size,
        hidden_continuous_size=cfg.hidden_continuous_size,
        attention_head_size=cfg.attention_heads,
        lstm_layers=cfg.lstm_layers,
        dropout=cfg.dropout,
        causal_attention=cfg.causal_attention,
        share_single_variable_networks=False,
        loss=MAE(),
        learning_rate=cfg.learning_rate,
        optimizer="adamw",
        weight_decay=cfg.weight_decay,
        # The library default (-1e9) does not fit in fp16, so mixed precision masks with -inf.
        mask_bias=-float("inf") if precision != "32-true" else -1e9,
        log_interval=-1,
        reduce_on_plateau_patience=4,
    )


def fit(
    panel: pd.DataFrame,
    cfg: TFTConfig,
    *,
    train_end: int,
    seed: int,
    epochs: int | None = None,
    early_stop_start: int | None = None,
    checkpoint_path: Path | None = None,
) -> tuple[TemporalFusionTransformer, TimeSeriesDataSet, dict]:
    """Train one model on windows ending before `train_end`.

    Either a fixed number of `epochs` (the final fit, so training reaches the forecast origin)
    or early stopping on the window starting at `early_stop_start` (cross-validation).
    """
    if (epochs is None) == (early_stop_start is None):
        raise ValueError("give exactly one of `epochs` or `early_stop_start`")
    accel, precision = accelerator()
    n_series = panel["series_id"].nunique()
    train_ds = training_dataset(panel, cfg, train_end)
    seed_everything(seed, workers=True, verbose=False)
    model = build_model(train_ds, cfg)

    callbacks, best = [], None
    val_loader = None
    if early_stop_start is not None:
        es_ds = prediction_dataset(train_ds, panel, early_stop_start, n_series)
        val_loader = _loader(es_ds, cfg, train=False)
        best = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,
                               dirpath=checkpoint_path.parent if checkpoint_path
                               else tempfile.mkdtemp(prefix="tft-ckpt-"))
        callbacks += [best, EarlyStopping(monitor="val_loss", min_delta=1e-4,
                                          patience=cfg.patience, mode="min")]
    trainer = Trainer(
        max_epochs=epochs or cfg.max_epochs,
        accelerator=accel,
        devices=1,
        precision=precision,
        gradient_clip_val=cfg.grad_clip,
        accumulate_grad_batches=cfg.accumulate,
        callbacks=callbacks,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=best is not None,
    )
    trainer.fit(model, _loader(train_ds, cfg, train=True), val_loader)

    info = {"seed": seed, "train_end": train_end, "windows": len(train_ds),
            "epochs_run": trainer.current_epoch}
    if best is not None and best.best_model_path:
        model = TemporalFusionTransformer.load_from_checkpoint(best.best_model_path)
        info["best_val_loss"] = float(best.best_model_score)
        info["best_epoch"] = max(trainer.current_epoch - cfg.patience, 1)
        if checkpoint_path is None:
            Path(best.best_model_path).unlink(missing_ok=True)
        else:
            Path(best.best_model_path).replace(checkpoint_path)
    elif checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(checkpoint_path))
    return model, train_ds, info


def predict_block(
    models: list[TemporalFusionTransformer],
    train_ds: TimeSeriesDataSet,
    panel: pd.DataFrame,
    decoder_start: int,
    cfg: TFTConfig,
) -> pd.DataFrame:
    """Mean prediction of `models` for the block starting at `decoder_start`, long format."""
    accel, precision = accelerator()
    n_series = panel["series_id"].nunique()
    loader = _loader(prediction_dataset(train_ds, panel, decoder_start, n_series), cfg, train=False)
    outputs, index = [], None
    for model in models:
        with _quiet():
            out = model.predict(
                loader,
                mode="prediction",
                return_index=True,
                trainer_kwargs=dict(accelerator=accel, devices=1, precision=precision,
                                    logger=False, enable_progress_bar=False),
            )
        outputs.append(out.output.cpu().float().numpy())
        index = out.index
    mean = np.mean(outputs, axis=0)
    horizon = mean.shape[1]
    frames = [
        pd.DataFrame({
            "series_id": str(index["series_id"].iloc[i]),
            "time_idx": np.arange(int(index["time_idx"].iloc[i]),
                                  int(index["time_idx"].iloc[i]) + horizon),
            "prediction": mean[i].astype(np.float64),
        })
        for i in range(len(index))
    ]
    return pd.concat(frames, ignore_index=True).merge(
        panel[["series_id", "time_idx", "timestamp"]], on=["series_id", "time_idx"], how="left"
    )[["series_id", "timestamp", "prediction"]]


# ---------------------------------------------------------------------------------------------
# Workflows


def cross_validate(split: Split, cfg: TFTConfig, *, seeds: list[int] | None = None,
                   n_folds: int = N_FOLDS) -> tuple[list[float], pd.DataFrame, list[dict]]:
    """Honest rolling-origin CV: early-stop on the block before each scored block."""
    panel, t_train = prepare_panel(split)
    truth = panel.loc[~panel["is_future"], ["series_id", "time_idx", "timestamp", "target"]]
    scores, oof, log = [], [], []
    for k in range(n_folds):
        b = fold_bounds(t_train, k, n_folds=n_folds, horizon=cfg.horizon, inner=True)
        models, train_ds = [], None
        for seed in seeds or cfg.seeds[:1]:
            model, train_ds, info = fit(panel, cfg, train_end=b.train_end, seed=seed,
                                        early_stop_start=b.early_stop_start)
            models.append(model)
            log.append({"fold": k, **info})
        pred = predict_block(models, train_ds, panel, b.score_start, cfg)
        scored = pred.merge(truth, on=["series_id", "timestamp"], how="left")
        scores.append(wape(scored["target"], scored["prediction"]))
        oof.append(scored.assign(fold=k))
        print(f"  tft fold {k}: WAPE {scores[-1]:.3f}", flush=True)
    print(f"  tft CV WAPE {np.mean(scores):.3f} +/- {np.std(scores):.3f}", flush=True)
    return scores, pd.concat(oof, ignore_index=True), log


def train_final(split: Split, cfg: TFTConfig, out_dir: Path) -> Path:
    """Fit one model per seed on the full history and write a checkpoint bundle."""
    panel, t_train = prepare_panel(split)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for seed in cfg.seeds:
        path = out_dir / "checkpoints" / f"final_s{seed}.ckpt"
        _, _, info = fit(panel, cfg, train_end=t_train, seed=seed, epochs=cfg.final_epochs,
                         checkpoint_path=path)
        print(f"  seed {seed}: {info['windows']} windows, {info['epochs_run']} epochs", flush=True)
        files.append(str(path.relative_to(out_dir)))
    bundle = out_dir / "checkpoint.json"
    bundle.write_text(json.dumps({"checkpoint_files": files, "seeds": cfg.seeds,
                                  "aggregation": "mean", "config": asdict(cfg)}, indent=2))
    return bundle


def load_bundle(bundle: Path) -> list[Path]:
    """Checkpoint paths listed in a bundle (JSON; the original archive named it checkpoint.pt)."""
    spec = json.loads(Path(bundle).read_text())
    paths = [(Path(bundle).parent / rel).resolve() for rel in spec["checkpoint_files"]]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"checkpoints listed in {bundle} are missing: {missing}")
    return paths


def forecast(split: Split, bundle: Path, cfg: TFTConfig | None = None) -> pd.DataFrame:
    """Predict the forecast index with the seed ensemble in `bundle`."""
    cfg = cfg or TFTConfig()
    seed_everything(42, workers=True, verbose=False)
    panel, t_train = prepare_panel(split)
    reference = training_dataset(panel, cfg, t_train)
    models = [TemporalFusionTransformer.load_from_checkpoint(str(p)) for p in load_bundle(bundle)]
    pred = predict_block(models, reference, panel, t_train, cfg)
    out = split.index[["series_id", "timestamp"]].merge(pred, on=["series_id", "timestamp"],
                                                        how="left")
    if out["prediction"].isna().any():
        raise ValueError(f"{out['prediction'].isna().sum()} forecast-index rows got no prediction")
    if not np.isfinite(out["prediction"]).all():
        raise ValueError("non-finite predictions")
    return out
