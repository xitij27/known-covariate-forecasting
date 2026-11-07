"""Rolling-origin cross-validation on the training history.

Three expanding-window folds, each scored on a 336-hour block, mirroring the 336-hour
validation and test blocks:

    fold 0: train [0 .. T-3H)   score [T-3H .. T-2H)
    fold 1: train [0 .. T-2H)   score [T-2H .. T-H)
    fold 2: train [0 .. T-H)    score [T-H  .. T)

Models that early-stop need a block to stop on. That block must not be the one being scored:
early-stopping on the scored block made one model's CV 0.57 WAPE too optimistic in this project
and inverted a model ranking (see docs/experiments.md, "The CV protocol trap"). `inner_bounds`
therefore holds out the 336 hours *before* each scored block for early stopping.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import EVAL_HORIZON, Split, holdout_split
from .metrics import wape

N_FOLDS = 3


@dataclass(frozen=True)
class FoldBounds:
    """Hour offsets on one series' clock, all end-exclusive."""

    train_end: int
    early_stop_start: int | None
    score_start: int
    score_end: int


def fold_bounds(
    t_train: int, k: int, *, n_folds: int = N_FOLDS, horizon: int = EVAL_HORIZON, inner: bool
) -> FoldBounds:
    """Bounds for fold k on a history of `t_train` hours.

    `inner=True`: train stops one horizon earlier and that horizon is the early-stop block.
    `inner=False`: no early-stop block (for models trained a fixed number of rounds).
    """
    score_start = t_train - (n_folds - k) * horizon
    if score_start <= (2 if inner else 1) * horizon:
        raise ValueError(f"history of {t_train}h is too short for {n_folds} folds of {horizon}h")
    if inner:
        return FoldBounds(score_start - horizon, score_start - horizon, score_start,
                          score_start + horizon)
    return FoldBounds(score_start, None, score_start, score_start + horizon)


def folds(
    history: pd.DataFrame, *, n_folds: int = N_FOLDS, horizon: int = EVAL_HORIZON
) -> Iterator[tuple[int, Split]]:
    """Yield (k, split) per fold: history up to the scored block, and that block's covariates."""
    for k in range(n_folds):
        yield k, holdout_split(history, end_offset=(n_folds - k) * horizon, horizon=horizon)


def evaluate(
    history: pd.DataFrame,
    forecast: Callable[[Split], pd.DataFrame],
    *,
    n_folds: int = N_FOLDS,
    horizon: int = EVAL_HORIZON,
    name: str = "model",
    verbose: bool = True,
) -> tuple[list[float], pd.DataFrame]:
    """Score `forecast(split) -> frame[series_id, timestamp, prediction]` on every fold.

    Returns the per-fold WAPEs and the out-of-fold predictions joined to the actuals.
    """
    scores, oof = [], []
    for k, split in folds(history, n_folds=n_folds, horizon=horizon):
        pred = forecast(split)
        scored = split.index.merge(pred, on=["series_id", "timestamp"], how="left")
        if scored["prediction"].isna().any():
            missing = int(scored["prediction"].isna().sum())
            raise ValueError(f"fold {k}: {missing} rows have no prediction")
        scores.append(wape(scored["target"], scored["prediction"]))
        oof.append(scored.assign(fold=k))
        if verbose:
            print(f"  {name} fold {k}: WAPE {scores[-1]:.3f}", flush=True)
    if verbose:
        print(f"  {name} CV WAPE {np.mean(scores):.3f} +/- {np.std(scores):.3f}", flush=True)
    return scores, pd.concat(oof, ignore_index=True)
