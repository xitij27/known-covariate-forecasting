"""LightGBM direct regression on the known covariates (experiment 4).

Each hour is one row: the covariates at that hour and around it go in, the target at that
hour comes out. No target lags are used, because the forecast block has no targets.

What moved the CV WAPE, from the experiment log:
    L2 objective, plain features            17.27
    + MAE objective, series_id categorical  15.39   (match the loss to WAPE; per-series level)
    + covariate lags, leads and rolling     14.80
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import DOMAIN_COLS, Split
from .features import TIME_REAL_COLS, add_covariate_context, add_time_features, impute_covariates

# Ranked by out-of-sample r^2 with the target in the EDA: 47%, 43%, 30%, 18%, 14%.
CONTEXT_COLS = [
    "queue_pressure_forecast",
    "network_pressure_forecast",
    "workload_intensity",
    "shock_risk",
    "event_load_forecast",
]
ROLLING_COLS = CONTEXT_COLS[:3]


@dataclass
class LGBMConfig:
    objective: str = "regression_l1"
    use_series_id: bool = True
    use_context: bool = True
    seed: int = 42
    params: dict = field(
        default_factory=lambda: dict(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=50,
            reg_lambda=1.0,
        )
    )


def build_frame(split: Split, cfg: LGBMConfig) -> tuple[pd.DataFrame, list[str]]:
    """History and forecast block stacked, with features computed across the seam.

    Computing on the stacked frame lets the first forecast hours see covariate lags from the
    history and the last history hours see leads into the forecast block, as at prediction time.
    """
    future = split.future.assign(target=np.nan, _future=True)
    frame = pd.concat([split.history.assign(_future=False), future], ignore_index=True)
    frame = add_time_features(impute_covariates(frame, DOMAIN_COLS, backfill=False))

    series_mean = split.history.groupby("series_id")["target"].mean().rename("series_mean")
    frame = frame.join(series_mean, on="series_id")
    codes = {sid: i for i, sid in enumerate(sorted(split.history["series_id"].unique()))}
    frame["sid_code"] = frame["series_id"].map(codes).astype("int32")

    features = DOMAIN_COLS + TIME_REAL_COLS + ["series_mean"]
    if cfg.use_context:
        frame, context = add_covariate_context(frame, CONTEXT_COLS, ROLLING_COLS)
        features += context
    if cfg.use_series_id:
        features.append("sid_code")
    return frame, features


def forecast(split: Split, cfg: LGBMConfig | None = None) -> pd.DataFrame:
    """Fit on the history and predict every row of the forecast index."""
    cfg = cfg or LGBMConfig()
    frame, features = build_frame(split, cfg)
    train = frame[~frame["_future"]]
    future = frame[frame["_future"]].copy()

    model = lgb.LGBMRegressor(
        objective=cfg.objective, random_state=cfg.seed, n_jobs=-1, verbosity=-1, **cfg.params
    )
    fit_kw = {"categorical_feature": ["sid_code"]} if cfg.use_series_id else {}
    model.fit(train[features], train["target"].to_numpy(float), **fit_kw)

    future["prediction"] = np.maximum(model.predict(future[features]), 0.0)
    out = split.index[["series_id", "timestamp"]].merge(
        future[["series_id", "timestamp", "prediction"]], on=["series_id", "timestamp"], how="left"
    )
    if out["prediction"].isna().any():
        raise ValueError(f"{out['prediction'].isna().sum()} forecast-index rows got no prediction")
    return out
