"""Feature engineering shared by the models.

Everything here is computed from timestamps and the known covariates only. The target never
enters a feature: at prediction time there are no target values for the forecast block.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIME_REAL_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_weekend"]


def impute_covariates(df: pd.DataFrame, cols: list[str], *, backfill: bool) -> pd.DataFrame:
    """Forward-fill gaps in each series' covariates.

    Leading gaps are back-filled (`backfill=True`, the TFT recipe) or set to zero (the
    LightGBM recipe). Both are kept exactly as they were run so the recorded scores reproduce.
    """
    df = df.sort_values(["series_id", "timestamp"]).copy()
    grouped = df.groupby("series_id", group_keys=False)[cols]
    if backfill:
        df[cols] = grouped.transform(lambda x: x.ffill().bfill())
    else:
        df[cols] = grouped.transform(lambda x: x.ffill().fillna(0))
    return df


def add_time_features(df: pd.DataFrame, *, categorical: bool = False) -> pd.DataFrame:
    """Cyclical hour, weekday and day-of-year encodings, plus a weekend flag."""
    df = df.copy()
    t = pd.to_datetime(df["timestamp"])
    df["hour_sin"] = np.sin(2 * np.pi * t.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * t.dt.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * t.dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * t.dt.dayofweek / 7)
    df["doy_sin"] = np.sin(2 * np.pi * t.dt.dayofyear / 365)
    df["doy_cos"] = np.cos(2 * np.pi * t.dt.dayofyear / 365)
    df["is_weekend"] = (t.dt.dayofweek >= 5).astype(float)
    if categorical:
        # The TFT also gets hour and weekday as categoricals, so they receive learned
        # embeddings and take part in variable selection as discrete variables.
        df["hour_cat"] = t.dt.hour.astype(str)
        df["dow_cat"] = t.dt.dayofweek.astype(str)
    return df


def add_covariate_context(
    df: pd.DataFrame,
    lag_cols: list[str],
    roll_cols: list[str],
    *,
    lags: tuple[int, ...] = (1, 24),
    leads: tuple[int, ...] = (1, 24),
    windows: tuple[int, ...] = (6, 24),
) -> tuple[pd.DataFrame, list[str]]:
    """Lags, leads and trailing rolling statistics of covariates, per series.

    Leads are legitimate here: the covariates for the whole forecast block are provided, so
    `queue_pressure_forecast` at t+24 is known when predicting the target at t. Rows near the
    end of the provided block get NaN leads, exactly as they would at prediction time.
    """
    df = df.sort_values(["series_id", "timestamp"]).copy()
    new: list[str] = []
    for col in lag_cols:
        grouped = df.groupby("series_id", sort=False)[col]
        for k in lags:
            df[f"{col}_lag{k}"] = grouped.shift(k)
            new.append(f"{col}_lag{k}")
        for k in leads:
            df[f"{col}_lead{k}"] = grouped.shift(-k)
            new.append(f"{col}_lead{k}")
    for col in roll_cols:
        grouped = df.groupby("series_id", sort=False)[col]
        for w in windows:
            rolling = grouped.rolling(w, min_periods=1)
            df[f"{col}_rmean{w}"] = rolling.mean().reset_index(level=0, drop=True)
            df[f"{col}_rstd{w}"] = rolling.std().reset_index(level=0, drop=True)
            new += [f"{col}_rmean{w}", f"{col}_rstd{w}"]
    return df, new


def add_velocities(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """First difference of each covariate per series (zero for a series' first row)."""
    df = df.copy()
    names = []
    for col in cols:
        df[f"{col}_vel"] = df.groupby("series_id")[col].diff().fillna(0.0)
        names.append(f"{col}_vel")
    return df, names
