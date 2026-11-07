"""A small synthetic panel shaped like the real data, so tests need no download."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from loadforecast.data import DOMAIN_COLS, Split


def make_panel(n_series: int = 3, hours: int = 24 * 40, seed: int = 0) -> pd.DataFrame:
    """Hourly series whose target is driven by two covariates plus a daily cycle."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2023-01-01", periods=hours, freq="h")
    frames = []
    for i in range(n_series):
        f = pd.DataFrame({"series_id": f"unit_{i:03d}", "timestamp": t})
        for col in DOMAIN_COLS:
            f[col] = rng.normal(size=hours).cumsum() * 0.05 + rng.normal(size=hours) * 0.1
        daily = 2 * np.sin(2 * np.pi * t.hour / 24)
        f["target"] = (
            5 + i + 3 * f["queue_pressure_forecast"] + 2 * f["network_pressure_forecast"] + daily
            + rng.normal(scale=0.3, size=hours)
        ).clip(lower=0.1)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def split_last(panel: pd.DataFrame, horizon: int) -> Split:
    """History = all but the last `horizon` hours per series; future = those hours."""
    position = panel.groupby("series_id").cumcount()
    length = panel.groupby("series_id")["timestamp"].transform("size")
    future = panel[position >= length - horizon]
    return Split(
        history=panel[position < length - horizon].reset_index(drop=True),
        future=future.drop(columns="target").reset_index(drop=True),
        index=future[["series_id", "timestamp", "target"]].reset_index(drop=True),
    )


@pytest.fixture
def panel() -> pd.DataFrame:
    return make_panel()
