"""The leaderboard metrics. Lower is better for all of them; WAPE is the primary one."""

from __future__ import annotations

import numpy as np


def _arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)


def wape(y_true, y_pred, eps: float = 1e-8) -> float:
    """Weighted absolute percentage error: total absolute error over total actuals, in %."""
    y_true, y_pred = _arrays(y_true, y_pred)
    return float(np.abs(y_true - y_pred).sum() / (np.abs(y_true).sum() + eps) * 100)


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _arrays(y_true, y_pred)
    return float(np.abs(y_true - y_pred).mean())


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _arrays(y_true, y_pred)
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true, y_pred = _arrays(y_true, y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(2 * np.abs(y_pred - y_true) / denominator) * 100)


def report(y_true, y_pred) -> dict[str, float]:
    return {
        "wape": wape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
