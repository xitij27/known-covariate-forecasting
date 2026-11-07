"""Draw the README figures, each in a light and a dark variant.

    python scripts/make_figures.py --data-dir data --oof outputs/oof_lgbm.csv

`--oof` is the out-of-fold file written by `loadforecast cv --model lgbm --oof ...`.
The leaderboard numbers are the scores the course's validation leaderboard returned; the
validation labels are hidden, so they cannot be recomputed here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from loadforecast import baselines, data  # noqa: E402
from loadforecast.cv import folds  # noqa: E402
from loadforecast.metrics import wape  # noqa: E402

# Public validation leaderboard, WAPE (lower is better).
LEADERBOARD = [
    ("Naive last value", 48.10),
    ("Seasonal mean (weekday × hour)", 34.41),
    ("LightGBM, MAE + context features", 15.48),
    ("BiGRU encoder-decoder, 5 seeds", 14.773),
    ("Gated MLP", 14.75),
    ("LightGBM + gated MLP blend", 14.682),
    ("TFT, 5-seed ensemble (shipped)", 14.479),
]

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", blue="#2a78d6", orange="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", blue="#3987e5", orange="#d95926"),
}


def _style(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelsize=9, length=0)


def leaderboard(out: Path, t: dict) -> None:
    rows = LEADERBOARD[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150, facecolor=t["surface"])
    _style(ax, t)
    names = [n for n, _ in rows]
    values = [v for _, v in rows]
    ax.barh(names, values, height=0.55, color=t["blue"])
    for i, v in enumerate(values):
        ax.text(v + 0.6, i, f"{v:.2f}", va="center", fontsize=9, color=t["ink"])
    ax.set_xlim(0, 54)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", colors=t["ink2"], labelsize=9.5)
    fig.suptitle("Public validation leaderboard, WAPE (lower is better)",
                 x=0.015, ha="left", fontsize=11, color=t["ink"])
    fig.tight_layout()
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)


def backtest(out: Path, t: dict, split_history: pd.DataFrame, oof: pd.DataFrame) -> None:
    fold = oof[oof["fold"] == 2]
    per_series = fold.groupby("series_id").apply(
        lambda g: wape(g["target"], g["prediction"]), include_groups=False
    )
    # a typical series, not the best one: the one closest to the median WAPE
    sid = (per_series - per_series.median()).abs().idxmin()
    _, split = list(folds(split_history))[2]
    seasonal = baselines.seasonal_mean(split)
    s = fold[fold["series_id"] == sid].sort_values("timestamp")
    sm = seasonal[seasonal["series_id"] == sid].sort_values("timestamp")

    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=150, facecolor=t["surface"])
    _style(ax, t)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.plot(s["timestamp"], s["target"], color=t["ink2"], linewidth=1.4, label="Actual")
    ax.plot(sm["timestamp"], sm["prediction"], color=t["orange"], linewidth=1.6,
            label=f"Seasonal mean (WAPE {wape(s['target'], sm['prediction']):.1f})")
    ax.plot(s["timestamp"], s["prediction"], color=t["blue"], linewidth=1.6,
            label=f"LightGBM on known covariates (WAPE {per_series[sid]:.1f})")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False, fontsize=8.5,
              labelcolor=t["ink2"], handlelength=1.6)
    ax.set_title(f"14-day backtest, {sid} (last 336 h of the training data)", loc="left",
                 fontsize=11, color=t["ink"], pad=22)
    ax.set_ylabel("Load index", color=t["muted"], fontsize=9)
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--oof", type=Path, default=Path("outputs/oof_lgbm.csv"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    history = data.load_validation(args.data_dir).history
    oof = pd.read_csv(args.oof, parse_dates=["timestamp"])
    oof["series_id"] = oof["series_id"].astype(str)
    for mode, theme in THEMES.items():
        leaderboard(args.out / f"leaderboard-{mode}.png", theme)
        backtest(args.out / f"backtest-{mode}.png", theme, history, oof)
        print(f"  wrote {args.out}/*-{mode}.png")


if __name__ == "__main__":
    main()
