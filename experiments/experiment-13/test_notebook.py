"""Regression test for kp-experiment-13a-tft-multiseed.ipynb (stage 1).

Runs EVERY code cell of the notebook in order against synthetic data on CPU with two
simulated devices, then asserts hard invariants on the artifacts produced. Fake data, no
GPU, no Kaggle quota -- this is a test suite, not the experiment.

    python test_notebook.py            # PREFLIGHT path
    python test_notebook.py --full     # the REAL stage-1 config (multi-seed, §11 only)

Stage 1 runs §11 only: §7/§8/§9/§10/§12 are off and their artifacts are asserted ABSENT.
The invariant that matters is that the submission is the MEAN of every seed's predictions --
shipping a single seed again would silently defeat the whole experiment.

Needs: torch (cpu), pytorch-forecasting==1.8.0, optuna, optuna-integration, statsmodels,
tensorboardX, matplotlib, scipy, pyarrow.

Re-run both modes after ANY edit to the notebook.

Runs EVERY code cell of the notebook in order with PREFLIGHT=True (the same code path the
user runs first on Kaggle), then asserts hard invariants on the artifacts produced.

This is a test suite, not the experiment: fake data, no GPU, no Kaggle quota.
"""
import json
import os
import re
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
SCRATCH = Path(os.environ.get("TFT_TEST_DIR", "/tmp/tft_notebook_test"))
SCRATCH.mkdir(parents=True, exist_ok=True)
DATA = SCRATCH / "fakedata"
OUT = SCRATCH / "nbout"
NB = HERE / "kp-experiment-13a-tft-multiseed.ipynb"

DOMAIN = [
    "workload_intensity", "demand_forecast", "upstream_quality_forecast",
    "promotion_intensity", "shock_risk", "maintenance_known",
    "unit_reliability_forecast", "queue_pressure_forecast",
    "network_pressure_forecast", "event_load_forecast",
    "service_irregularity_risk_forecast",
]
N_SERIES, T_TRAIN, HORIZON, N_FOLDS = 4, 4320, 336, 3
FAILURES = []


def check(cond, msg):
    if cond:
        print(f"    PASS  {msg}")
    else:
        print(f"    FAIL  {msg}")
        FAILURES.append(msg)


def build_fake():
    if DATA.exists():
        shutil.rmtree(DATA)
    DATA.mkdir(parents=True)
    rng = np.random.default_rng(0)
    T = T_TRAIN + HORIZON
    ts = pd.date_range("2023-01-01", periods=T, freq="h")
    tr, va = [], []
    for s in range(N_SERIES):
        cov = rng.normal(size=(T, len(DOMAIN))).cumsum(0) / 40.0
        scale = 6 + 8 * rng.random()
        tgt = scale * (1 + 0.3 * np.sin(2 * np.pi * np.arange(T) / 24)
                       + 0.4 * cov[:, 7] + 0.1 * rng.normal(size=T))
        d = pd.DataFrame(cov, columns=DOMAIN)
        d["series_id"] = f"unit_{s:03d}"
        d["timestamp"] = ts
        for c in DOMAIN[:3]:                       # exercise the imputation path
            d.loc[rng.random(T) < 0.04, c] = np.nan
        d["target"] = np.clip(tgt, 0.01, None).astype(np.float32)
        tr.append(d[d.timestamp < ts[T_TRAIN]])
        va.append(d[d.timestamp >= ts[T_TRAIN]].drop(columns=["target"]))
    pd.concat(tr, ignore_index=True).to_csv(DATA / "train.csv", index=False)
    v = pd.concat(va, ignore_index=True)
    v.to_csv(DATA / "validation_input.csv", index=False)
    v[["series_id", "timestamp"]].to_csv(DATA / "forecast_index_validation.csv", index=False)
    print(f"  synthetic data: {N_SERIES} series x {T} hours")


PATCHES = [
    (r'DATA_DIR   = Path\("/kaggle/input.*?"\)', f'DATA_DIR   = Path("{DATA}")'),
    (r'OUTPUT_DIR = Path\("/kaggle/working"\)', f'OUTPUT_DIR = Path("{OUT}")'),
    # shrink the model so CPU can do it; PREFLIGHT still shrinks epochs/stride/seeds on top
    (r'^HIDDEN_SIZE             = 96$', 'HIDDEN_SIZE             = 16'),
    (r'^HIDDEN_CONTINUOUS_SIZE  = 32$', 'HIDDEN_CONTINUOUS_SIZE  = 8'),
    (r'^ATTENTION_HEADS         = 4$', 'ATTENTION_HEADS         = 2'),
    (r'^MAX_ENCODER      = 336$', 'MAX_ENCODER      = 96'),
    (r'^MIN_ENCODER      = 168 ', 'MIN_ENCODER      = 48 '),
    (r'^STRIDE   = 168 ', 'STRIDE   = 336 '),
    # CPU, but keep TWO workers so dispatcher concurrency is genuinely exercised
    (r'assert torch\.cuda\.is_available\(\).*', 'pass'),
    (r'N_GPUS = torch\.cuda\.device_count\(\)', 'N_GPUS = 2'),
    (r'for i in range\(N_GPUS\):\n    print\(f"GPU \{i\}: \{torch\.cuda\.get_device_name\(i\)\}"\)', 'pass'),
    (r'torch\.cuda\.is_bf16_supported\(\)', 'False'),
    # worker source (literal text inside the cell that writes it)
    (r'precision = "bf16-mixed" if False else "16-mixed"', 'precision = "32-true"'),
    (r'torch\.cuda\.get_device_name\(0\)', '"cpu"'),
    (r'accelerator="gpu", devices=1, precision=precision', 'accelerator="cpu", devices=1'),
    (r'accelerator="gpu", devices=1, logger=False', 'accelerator="cpu", devices=1, logger=False'),
    (r'pin_memory=True', 'pin_memory=False'),
    (r'\.to\("cuda:0"\)', ''),
    # priors are resolved by filename; point the search at the fabricated dir
    (r'PRIOR_SEARCH_ROOTS = \[.*?\]', f'PRIOR_SEARCH_ROOTS = [Path("{SCRATCH / "priors"}")]'),
]


FULL_MODE = "--full" in sys.argv

FULL_PATCHES = [
    (r'^PREFLIGHT = True$', 'PREFLIGHT = False'),
    (r'^MAX_EPOCHS    = 60$', 'MAX_EPOCHS    = 1'),
    (r'^PATIENCE      = 10$', 'PATIENCE      = 1'),
    (r'^SEEDS        = \[42, 7, 13, 99, 1234\].*$', 'SEEDS        = [42, 7, 13]'),
    # exercise the real stride-24 grid
    (r'^STRIDE   = 336 ', 'STRIDE   = 24 '),
    # force the over-quota STOP branch: on the real config it WILL fire, and a print block
    # that has never executed is a print block that can still NameError.
    (r'if total_gpu_h > 30 or total_wall_h > 12:', 'if True:'),
]


def patch(src):
    for pat, rep in PATCHES:
        src = re.sub(pat, rep, src, flags=re.MULTILINE)
    if FULL_MODE:
        for pat, rep in FULL_PATCHES:
            src = re.sub(pat, rep, src, flags=re.MULTILINE)
    return src


def make_fake_blend_inputs(ns):
    """After CV, fabricate prior-experiment OOF/val files so §12's merge path really runs."""
    # Called before §9 (OOF exists) and again before §12 (submission exists too), so each
    # half is built only once its source is available.
    d = SCRATCH / "priors"
    d.mkdir(parents=True, exist_ok=True)
    if not (ns.get("OOF_OUT") and Path(ns["OOF_OUT"]).exists()):
        return d
    oof = pd.read_csv(ns["OOF_OUT"], parse_dates=["timestamp"])
    have_sub = ns.get("VAL_OUT") and Path(ns["VAL_OUT"]).exists()
    sub = pd.read_csv(ns["VAL_OUT"], parse_dates=["timestamp"]) if have_sub else None
    rng = np.random.default_rng(1)
    for name in ("gru", "7c"):
        o = oof[["series_id", "timestamp", "target"]].copy()
        # mirror the real 7b file: several pred_* columns, only one of which is the blend
        o["pred_lgbm"] = oof["pred_tft"] * (1 + rng.normal(0, .08, len(oof)))
        o["pred_mlp"] = oof["pred_tft"] * (1 + rng.normal(0, .08, len(oof)))
        o["pred_blend"] = oof["pred_tft"] * (1 + rng.normal(0, .05, len(oof)))
        o["lead_h"], o["fold"] = oof["lead_h"], oof["fold"]
        fn = {"gru": "oof_predictions_11a_gru.csv", "7c": "oof_predictions_7b_blend.csv"}[name]
        o.to_csv(d / fn, index=False)
        if sub is None:
            continue
        v = sub[["series_id", "timestamp"]].copy()
        v["prediction"] = sub["prediction"] * (1 + rng.normal(0, .05, len(sub)))
        vfn = {"gru": "validation_predictions_11a_gru.csv",
               "7c": "validation_predictions_7c.csv"}[name]
        v.to_csv(d / vfn, index=False)
    print(f"  fabricated prior-model files in {d}")
    return d


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    build_fake()

    nb = json.loads(NB.read_text())
    cells = [(i, ''.join(c["source"])) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
    import matplotlib
    matplotlib.use("Agg")
    ns = {"display": lambda *a, **k: None, "__name__": "__main__"}

    t0 = time.time()
    for n, (i, src) in enumerate(cells):
        s = patch(src)
        # after the CV cell writes OOF, and before the blend cell, plant prior-model files
        if "BLEND_SOURCES" in s:
            make_fake_blend_inputs(ns)
        if "residual correlation vs prior models" in s:
            make_fake_blend_inputs(ns)
        print(f"\n{'='*72}\nCELL {i}  ({n+1}/{len(cells)})   t+{time.time()-t0:.0f}s\n{'='*72}", flush=True)
        try:
            exec(compile(s, f"cell_{i}", "exec"), ns)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n>>> CELL {i} RAISED {type(e).__name__}: {e}")
            FAILURES.append(f"cell {i} raised {type(e).__name__}: {e}")
            print("    (continuing so all failures surface in one pass)")

    print(f"\n\n{'#'*72}\n# ASSERTIONS\n{'#'*72}")
    verify(ns)

    print(f"\n{'#'*72}")
    if FAILURES:
        print(f"# {len(FAILURES)} FAILURE(S)")
        for f in FAILURES:
            print(f"#   - {f}")
        sys.exit(1)
    print(f"# ALL CHECKS PASSED  ({time.time()-t0:.0f}s)")


def verify(ns):
    run = Path(ns["RUN_DIR"]) if ns.get("RUN_DIR") else OUT / "run"
    ehz = HORIZON
    recs = [json.loads(q.read_text()) for q in sorted((run / "results").glob("*.json"))]
    fin = sorted([r for r in recs if r["tag"].startswith("final_")], key=lambda r: r["tag"])

    print("\n  [stage 1: disabled stages produced nothing]")
    check(not Path(ns["OOF_OUT"]).exists(), "no OOF file (RUN_FULL_CV=False)")
    check(not Path(ns["PROBE_OUT"]).exists(), "no probe table (RUN_CONFIG_PROBE=False)")
    check(not [r for r in recs if r["tag"].startswith("cv_")], "no CV jobs ran")
    check(not [r for r in recs if r["tag"] == "interp"], "no interpretation job ran")
    check(not list(OUT.glob("*12b_blend*")), "no blend file (RUN_BLEND=False)")

    print("\n  [THE stage-1 invariant: submission == mean of ALL seeds]")
    seeds_cfg = sorted(ns["SEEDS"])
    check(len(fin) == len(seeds_cfg), f"one final job per seed: {len(fin)} vs {len(seeds_cfg)}")
    check(sorted(r["seed"] for r in fin) == seeds_cfg, f"seeds used: {[r['seed'] for r in fin]}")

    per = [np.load(run / "results" / f"{r['tag']}.npz")["preds"] for r in fin]
    z0 = np.load(run / "results" / f"{fin[0]['tag']}.npz")
    sid_order = z0["series_id"].astype(str)
    if len(per) > 1:
        spread = float(np.abs(per[0] - per[1]).mean())
        check(spread > 0, f"seeds give DIFFERENT predictions (mean |s0-s1| = {spread:.4f})")

    expect = np.mean(per, axis=0)
    exp_long = pd.concat(
        [pd.DataFrame({"series_id": sid_order[i], "k": np.arange(expect.shape[1]),
                       "expect": expect[i]}) for i in range(len(sid_order))],
        ignore_index=True)
    sub = pd.read_csv(ns["VAL_OUT"], parse_dates=["timestamp"])
    got = (sub.sort_values(["series_id", "timestamp"])
              .assign(k=lambda d: d.groupby("series_id").cumcount()))
    mg = got.merge(exp_long, on=["series_id", "k"])
    check(len(mg) == len(sub), f"every submission row matched a seed prediction ({len(mg)})")
    dev = float((mg.prediction - mg.expect).abs().max())
    check(dev < 1e-4, f"submission IS the mean of all {len(per)} seeds (max dev {dev:.2e})")
    if len(per) > 1:
        solo = float((mg.prediction.values - per[0].ravel()).__abs__().max())
        check(solo > 1e-6, f"submission is NOT silently one seed (dev vs seed0 {solo:.4f})")

    print("\n  [submission integrity]")
    idx = pd.read_csv(DATA / "forecast_index_validation.csv", parse_dates=["timestamp"])
    check(len(sub) == N_SERIES * ehz, f"row count {len(sub)} == {N_SERIES*ehz}")
    check(list(sub.columns) == ["series_id", "timestamp", "prediction"], "column layout")
    check(np.isfinite(sub["prediction"]).all(), "all predictions finite")
    m = idx.merge(sub, on=["series_id", "timestamp"], how="left")
    check(m["prediction"].notna().all(), "covers every row of the forecast index")

    print("\n  [exp-11 Bug 1 guard]")
    check(all(r["last_train_target"] == T_TRAIN - 1 for r in fin),
          f"every final fit reaches the forecast origin: {[r['last_train_target'] for r in fin]}")
    check(all(T_TRAIN - r["last_train_encoder_end"] <= ehz for r in fin),
          "encoder gap <= horizon for every seed")
    check(all(r["epochs_run"] == ns["FINAL_EPOCHS_OVERRIDE"] for r in fin),
          f"fixed epochs honoured: {[r['epochs_run'] for r in fin]} == {ns['FINAL_EPOCHS_OVERRIDE']}")

    print("\n  [checkpoints persist — one per seed]")
    mdir = Path(ns["MODEL_DIR"])
    cks = sorted(mdir.glob("final_*.ckpt"))
    check(len(cks) == len(seeds_cfg), f"{len(cks)} checkpoints for {len(seeds_cfg)} seeds")
    check(str(mdir).startswith(str(OUT)), "checkpoints are in the DOWNLOADABLE output dir")
    check(all(r.get("checkpoint") for r in fin), "results record their checkpoint path")

    print("\n  [dispatcher / concurrency]")
    check(all("started_at" in r for r in recs), "results carry wall-clock stamps")
    laps = sum(1 for i in range(len(recs)) for j in range(i + 1, len(recs))
               if recs[i]["started_at"] < recs[j]["finished_at"]
               and recs[j]["started_at"] < recs[i]["finished_at"])
    check(laps > 0, f"jobs ran concurrently across GPUs ({laps} overlapping pairs)")
    logs = list((run / "logs").glob("*.log"))
    tb = [q.name for q in logs if "Traceback" in q.read_text()]
    check(not tb, f"no tracebacks in worker logs{'' if not tb else ': ' + str(tb)}")

    print("\n  [packaging]")
    rep = OUT / f"run_report_{ns.get('EXP_ID','')}.md"
    check(rep.exists(), f"run report written ({rep.name})")
    if rep.exists():
        txt = rep.read_text()
        check("section failed:" not in txt, "no section of the report errored")
        check("cross-validation not run" in txt, "report states CV was not run")
    zips = list(OUT.glob("results_*.zip"))
    check(len(zips) == 1, f"exactly one bundle written ({[z.name for z in zips]})")
    check(not (OUT / "run").exists(), "intermediates stay out of the output dir")


if __name__ == "__main__":
    main()
