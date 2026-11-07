"""Dataset download, loading and the column groups every model shares."""

from __future__ import annotations

import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

HF_DATASET = "AIML-TUDA/dlam-ts-project-data-2026"
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main"
DATA_FILES = (
    "train.csv",
    "validation_input.csv",
    "forecast_index_validation.csv",
    "metadata.json",
)

EVAL_HORIZON = 336  # hours per series in the validation and test blocks (14 days)

# The 11 covariates that are provided for the forecast window too. They are inputs at
# prediction time, not quantities to forecast. The EDA dropped `staffing_forecast`,
# `throughput_disruption_risk_forecast` and `nominal_capacity` (each r^2 < 1% with the target)
# and `trend` (the target is stationary over the 180 days).
DOMAIN_COLS = [
    "workload_intensity",
    "demand_forecast",
    "upstream_quality_forecast",
    "promotion_intensity",
    "shock_risk",
    "maintenance_known",
    "unit_reliability_forecast",
    "queue_pressure_forecast",
    "network_pressure_forecast",
    "event_load_forecast",
    "service_irregularity_risk_forecast",
]


def download(data_dir: Path, *, force: bool = False) -> Path:
    """Download the public dataset files from Hugging Face into `data_dir`.

    The dataset licence is "other", so the files are fetched here rather than shipped in
    this repository.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in DATA_FILES:
        dest = data_dir / name
        if dest.exists() and dest.stat().st_size > 1000 and not force:
            print(f"  have      {dest}")
            continue
        url = f"{HF_BASE_URL}/{name}"
        print(f"  download  {url}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
        tmp.replace(dest)
    return data_dir


@dataclass
class Split:
    """The observed history plus the block to forecast.

    `history` has targets. `future` has the same covariates for the forecast block and no
    target column. `index` lists the exact rows to predict.
    """

    history: pd.DataFrame
    future: pd.DataFrame
    index: pd.DataFrame


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    frame["series_id"] = frame["series_id"].astype(str)
    return frame


def load_validation(data_dir: Path) -> Split:
    """Load train.csv as history and the public validation block as the future."""
    split = Split(
        history=_read(data_dir / "train.csv"),
        future=_read(data_dir / "validation_input.csv"),
        index=_read(data_dir / "forecast_index_validation.csv"),
    )
    if "target" in split.future.columns:
        raise ValueError("validation_input.csv must not contain target values")
    return split


def load_input_dir(input_dir: Path) -> Split:
    """Load a grader-style input directory: train.csv plus a test or validation block."""
    for stem in ("test", "validation"):
        future = input_dir / f"{stem}_input.csv"
        index = input_dir / f"forecast_index_{stem}.csv"
        if future.exists() and index.exists():
            split = Split(_read(input_dir / "train.csv"), _read(future), _read(index))
            if "target" in split.future.columns:
                raise ValueError(f"{future.name} must not contain target values")
            return split
    raise FileNotFoundError(
        f"{input_dir} needs train.csv plus test_input.csv + forecast_index_test.csv "
        "or validation_input.csv + forecast_index_validation.csv"
    )


def holdout_split(history: pd.DataFrame, end_offset: int, horizon: int = EVAL_HORIZON) -> Split:
    """Cut a backtest out of the history.

    Per series, the last `end_offset` hours are removed from the history, and the first
    `horizon` of those become the block to forecast. The returned `future` has its target
    column dropped; `index` keeps it so the backtest can be scored.
    """
    history = history.sort_values(["series_id", "timestamp"])
    position = history.groupby("series_id").cumcount()
    length = history.groupby("series_id")["timestamp"].transform("size")
    cut = length - end_offset
    past = history[position < cut]
    block = history[(position >= cut) & (position < cut + horizon)]
    return Split(
        history=past.reset_index(drop=True),
        future=block.drop(columns="target").reset_index(drop=True),
        index=block[["series_id", "timestamp", "target"]].reset_index(drop=True),
    )
