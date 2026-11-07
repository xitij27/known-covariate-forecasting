"""Command line: `loadforecast <command> --help` for each command."""

from __future__ import annotations

import argparse
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

from . import baselines, cv, data, lgbm

RELEASE_URL = (
    "https://github.com/xitij27/known-covariate-forecasting/releases/download/"
    "v1.0.0/tft-5seed-checkpoints.tar.gz"
)


def _download(args) -> None:
    data.download(args.data_dir, force=args.force)


def _fetch_model(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    archive = args.out / "tft-5seed-checkpoints.tar.gz"
    print(f"  download  {args.url}")
    urllib.request.urlretrieve(args.url, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(args.out, filter="data")
    archive.unlink()
    print(f"  bundle    {args.out / 'checkpoint.json'}")


def _cv(args) -> None:
    split = data.load_validation(args.data_dir)
    results = {}
    if args.model in ("baselines", "all"):
        for name, fn in [("naive", baselines.naive_last_value),
                         ("seasonal_mean", baselines.seasonal_mean)]:
            results[name], _ = cv.evaluate(split.history, fn, name=name)
    if args.model in ("lgbm", "all"):
        results["lgbm"], oof = cv.evaluate(split.history, lgbm.forecast, name="lgbm")
        if args.oof:
            oof.to_csv(args.oof, index=False)
    if args.model == "tft":
        from . import tft

        cfg = tft.TFTConfig(max_epochs=args.max_epochs)
        results["tft"], oof, log = tft.cross_validate(split, cfg, seeds=args.seeds)
        print(json.dumps(log, indent=1))
        if args.oof:
            oof.to_csv(args.oof, index=False)
    print("\nCV WAPE (3 folds x 336 h, lower is better)")
    for name, scores in results.items():
        folds = " / ".join(f"{s:.2f}" for s in scores)
        print(f"  {name:<14} {np.mean(scores):6.2f} +/- {np.std(scores):.2f}   ({folds})")


def _fit(args) -> None:
    from . import tft

    cfg = tft.TFTConfig(final_epochs=args.epochs, seeds=args.seeds)
    bundle = tft.train_final(data.load_validation(args.data_dir), cfg, args.out)
    print(f"  bundle    {bundle}")


def _predict(args) -> None:
    split = data.load_input_dir(args.input_dir)
    if args.model == "lgbm":
        pred = lgbm.forecast(split)
    elif args.model == "seasonal_mean":
        pred = baselines.seasonal_mean(split)
    else:
        from . import tft

        pred = tft.forecast(split, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pred[["series_id", "timestamp", "prediction"]].to_csv(args.output, index=False)
    print(f"  wrote     {args.output} ({len(pred):,} rows)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="loadforecast", description=__doc__)
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("download", help="fetch the public dataset from Hugging Face")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=_download)

    p = sub.add_parser("fetch-model", help="fetch the trained 5-seed TFT checkpoints")
    p.add_argument("--out", type=Path, default=Path("models/tft"))
    p.add_argument("--url", default=RELEASE_URL)
    p.set_defaults(func=_fetch_model)

    p = sub.add_parser("cv", help="rolling-origin cross-validation on train.csv")
    p.add_argument("--model", choices=["baselines", "lgbm", "all", "tft"], default="all")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--seeds", type=int, nargs="+", default=[42], help="tft only")
    p.add_argument("--max-epochs", type=int, default=60, help="tft only")
    p.add_argument("--oof", type=Path, help="write out-of-fold predictions to this CSV")
    p.set_defaults(func=_cv)

    p = sub.add_parser("fit-tft", help="train the TFT seed ensemble on the full history")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("models/tft"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 13, 99, 1234])
    p.set_defaults(func=_fit)

    p = sub.add_parser("predict", help="write predictions for a forecast index")
    p.add_argument("--model", choices=["tft", "lgbm", "seasonal_mean"], default="tft")
    p.add_argument("--input-dir", type=Path, default=Path("data"),
                   help="train.csv plus validation_input.csv or test_input.csv and its index")
    p.add_argument("--checkpoint", type=Path, default=Path("models/tft/checkpoint.json"))
    p.add_argument("--output", type=Path, default=Path("outputs/predictions.csv"))
    p.set_defaults(func=_predict)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
