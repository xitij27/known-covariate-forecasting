"""Reference forecasts: repeat the last value, or average by weekday and hour."""

from __future__ import annotations

import pandas as pd

from .data import Split


def naive_last_value(split: Split) -> pd.DataFrame:
    """Repeat each series' last observed target across the whole block."""
    last = (
        split.history.sort_values("timestamp")
        .groupby("series_id", as_index=False)["target"]
        .last()
        .rename(columns={"target": "prediction"})
    )
    return split.index[["series_id", "timestamp"]].merge(last, on="series_id", how="left")


def seasonal_mean(split: Split) -> pd.DataFrame:
    """Per-series mean of the target for each (weekday, hour), falling back to the series mean."""

    def keyed(frame: pd.DataFrame) -> pd.DataFrame:
        t = pd.to_datetime(frame["timestamp"])
        return frame.assign(_dow=t.dt.dayofweek, _hour=t.dt.hour)

    history = keyed(split.history)
    profile = (
        history.groupby(["series_id", "_dow", "_hour"], as_index=False)["target"]
        .mean()
        .rename(columns={"target": "prediction"})
    )
    series_mean = history.groupby("series_id")["target"].mean().rename("_series_mean")
    out = keyed(split.index[["series_id", "timestamp"]]).merge(
        profile, on=["series_id", "_dow", "_hour"], how="left"
    )
    out = out.join(series_mean, on="series_id")
    out["prediction"] = out["prediction"].fillna(out["_series_mean"])
    return out[["series_id", "timestamp", "prediction"]]
