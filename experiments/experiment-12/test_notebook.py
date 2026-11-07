"""Regression test for kp-experiment-12a-tft.ipynb.

Runs EVERY code cell of the notebook in order against synthetic data on CPU with two
simulated devices, then asserts hard invariants on the artifacts produced. Fake data, no
GPU, no Kaggle quota -- this is a test suite, not the experiment.

    python test_notebook.py            # PREFLIGHT path (1 seed, stride 168, tuning on)
    python test_notebook.py --full     # real path (multi-seed, stride 24, tuning off)

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
NB = HERE / "kp-experiment-12a-tft.ipynb"

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
    (r'^SEEDS        = \[42, 7, 13\]', 'SEEDS        = [42, 7]'),
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
    ehz, nf = HORIZON, N_FOLDS

    print("\n  [artifacts]")
    for key in ("OOF_OUT", "VAL_OUT", "PROBE_OUT", "IMPORTANCE_OUT"):
        p = ns.get(key)
        check(p is not None and Path(p).exists(), f"{key} written ({Path(p).name if p else '?'})")

    print("\n  [OOF integrity]")
    if ns.get("OOF_OUT") and Path(ns["OOF_OUT"]).exists():
        oof = pd.read_csv(ns["OOF_OUT"], parse_dates=["timestamp"])
        check(len(oof) == nf * N_SERIES * ehz, f"row count {len(oof)} == {nf*N_SERIES*ehz}")
        check(list(oof.columns) == ["series_id", "timestamp", "target", "pred_tft", "lead_h", "fold"],
              f"column layout matches repo standard: {list(oof.columns)}")
        check(oof["pred_tft"].notna().all() and np.isfinite(oof["pred_tft"]).all(), "no NaN/inf predictions")
        check(oof["target"].notna().all(), "no NaN targets (join succeeded)")
        check(sorted(oof["fold"].unique()) == list(range(nf)), "all folds present")
        check(oof.groupby("fold").size().nunique() == 1, "equal rows per fold")
        check(set(oof["lead_h"]) == set(range(1, ehz + 1)), "lead_h spans 1..336")
        check(not oof.duplicated(["series_id", "timestamp", "fold"]).any(), "no duplicate rows")
        # fold windows must be exactly the pre-registered blocks
        t0 = pd.Timestamp("2023-01-01")
        ok = True
        for k in range(nf):
            g = oof[oof.fold == k]
            start = int((g["timestamp"].min() - t0).total_seconds() // 3600)
            expect = T_TRAIN - (nf - k) * ehz
            ok &= start == expect
        check(ok, "each fold's window starts exactly at the pre-registered boundary")

    print("\n  [submission integrity]")
    if ns.get("VAL_OUT") and Path(ns["VAL_OUT"]).exists():
        sub = pd.read_csv(ns["VAL_OUT"], parse_dates=["timestamp"])
        idx = pd.read_csv(DATA / "forecast_index_validation.csv", parse_dates=["timestamp"])
        check(len(sub) == N_SERIES * ehz, f"row count {len(sub)} == {N_SERIES*ehz}")
        check(list(sub.columns) == ["series_id", "timestamp", "prediction"], "column layout")
        check(np.isfinite(sub["prediction"]).all(), "all predictions finite")
        m = idx.merge(sub, on=["series_id", "timestamp"], how="left")
        check(m["prediction"].notna().all(), "covers every row of the forecast index")
        check(sub["timestamp"].min() == idx["timestamp"].min()
              and sub["timestamp"].max() == idx["timestamp"].max(), "timestamp range matches index")

    print("\n  [leakage guards fired]")
    logs = list((run / "logs").glob("*.log"))
    check(len(logs) > 0, f"{len(logs)} worker logs written")
    tb = [p.name for p in logs if "Traceback" in p.read_text()]
    check(not tb, f"no tracebacks in any worker log{'' if not tb else ': ' + str(tb)}")

    print("\n  [dispatcher / concurrency]")
    res = sorted((run / "results").glob("*.json"))
    check(len(res) > 0, f"{len(res)} job results written")
    recs = [json.loads(p.read_text()) for p in res]
    check(all("started_at" in r for r in recs), "all results carry wall-clock stamps")
    overlaps = 0
    for a in range(len(recs)):
        for b in range(a + 1, len(recs)):
            x, y = recs[a], recs[b]
            if x["started_at"] < y["finished_at"] and y["started_at"] < x["finished_at"]:
                overlaps += 1
    check(overlaps > 0, f"jobs genuinely ran concurrently ({overlaps} overlapping pairs)")

    print("\n  [final fit reached the forecast origin]")
    fin = [r for r in recs if r["tag"].startswith("final_")]
    check(len(fin) > 0, f"{len(fin)} final-fit job(s)")
    if fin:
        r = fin[0]
        check(r["last_train_target"] == T_TRAIN - 1,
              f"last training target {r['last_train_target']} == {T_TRAIN-1} (exp-11 Bug 1 guard)")
        check(T_TRAIN - r["last_train_encoder_end"] <= ehz,
              f"encoder gap {T_TRAIN - r['last_train_encoder_end']}h <= horizon")

    print("\n  [preflight cost controls]")
    warm2 = [r for r in recs if r["tag"].startswith("warmup_")]
    if warm2 and not FULL_MODE:
        check(max(r["epochs_run"] for r in warm2) <= 1, "warmup runs 1 epoch")
    # index dedupe: one window per (series, decoder start)
    cvr = sorted([r for r in recs if r["tag"].startswith("cv_f0")], key=lambda r: r["tag"])
    if cvr:
        # exact expectation: one window per (series, decoder start), starts on the stride grid
        # from MIN_ENCODER up to train_end - horizon.
        te = cvr[0]["train_end"]
        max_start, min_start = te - ns["EVAL_HORIZON"], ns["MIN_ENCODER"]
        stride = ns["STRIDE"]
        expect = ns["N_SERIES"] * ((max_start - min_start) // stride + 1)
        got = cvr[0]["n_train_windows"]
        check(got == expect,
              f"window count {got} == one-per-start at stride {stride} ({expect}) "
              f"-- no per-encoder-length duplication")
    cvj2 = [r for r in recs if r["tag"].startswith("cv_")]
    if cvj2 and not FULL_MODE:
        check(max(r["epochs_run"] for r in cvj2) <= 1,
              "preflight CV jobs run at most 1 epoch")

    print("\n  [§10 reuses the §8 checkpoint]")
    tags = {r["tag"] for r in recs}
    check("interp" not in tags,
          "no redundant 'interp' training job (§10 reused the §8 checkpoint)")
    cks = sorted((run / "ckpt").glob("*.ckpt"))
    check(len(cks) == 1 and cks[0].name.startswith("cv_f"),
          f"exactly one checkpoint kept, from CV: {[q.name for q in cks]}")

    print("\n  [multi-seed averaging]")
    cvj = [r for r in recs if r["tag"].startswith("cv_")]
    seeds = sorted({r["seed"] for r in cvj})
    folds = sorted({r["tag"].split("_")[1] for r in cvj})
    check(len(cvj) == len(seeds) * len(folds),
          f"{len(cvj)} CV jobs == {len(folds)} folds x {len(seeds)} seeds {seeds}")
    check(ns.get("EXP_ID", "").endswith("_preflight") != FULL_MODE,
          f"EXP_ID matches mode: {ns.get('EXP_ID')}")

    print("\n  [premise checks ran]")
    warm = [r for r in recs if r["tag"].startswith("warmup_")]
    check(len(warm) >= 3, f"{len(warm)} warmup jobs (batch-size sweep)")
    check(len({r["batch_size"] for r in warm}) == len(warm),
          f"warmup swept distinct batch sizes: {sorted(r['batch_size'] for r in warm)}")
    check(all("peak_gpu_gib" in r for r in recs), "every job reports peak GPU memory")
    check(all(r.get("timed_steps", 0) > 0 for r in warm),
          f"warmup jobs recorded per-step timings ({[r.get('timed_steps') for r in warm]})")
    check(all(r.get("ms_per_step") for r in warm), "warmup jobs report ms_per_step")
    nonwarm = [r for r in recs if not r["tag"].startswith("warmup_")]
    check(all(r.get("timed_steps", 0) == 0 for r in nonwarm),
          "non-warmup jobs are not slowed by per-step CUDA syncs")
    prem = [r for r in warm if r.get("kind") == "throughput"]
    check(len(prem) == 1, f"exactly one job runs the premise checks ({[r['tag'] for r in prem]})")
    if prem:
        check(prem[0].get("loss_on_original_scale") is True, "loss verified on the ORIGINAL target scale")
        check(prem[0].get("causal_delta", 0) > 0, "causal_attention verified to change the forward pass")

    print("\n  [probe / tuning tables]")
    if ns.get("PROBE_OUT") and Path(ns["PROBE_OUT"]).exists():
        pr = pd.read_csv(ns["PROBE_OUT"])
        check(len(pr) == 4, f"probe has 4 arms ({list(pr['arm'])})")
        check(pr["wape"].notna().all(), "every probe arm produced a WAPE")
    tun = OUT / f"tuning_{ns.get('EXP_ID','')}.csv"
    if FULL_MODE:
        check(not tun.exists(), "§13a correctly skipped when RUN_TUNING is False")
    else:
        check(tun.exists(), f"§13a tuning table written ({tun.name})")
        if tun.exists():
            t = pd.read_csv(tun)
            check(len(t) >= 2 and t["wape"].notna().all(), f"tuning has {len(t)} scored trials")
        check({"hidden_size", "dropout", "learning_rate"} <= set(t.columns),
              "tuning table records the sampled hyperparameters")
        tj = [r for r in recs if r["tag"].startswith("tune_")]
        pairs = sum(1 for a in range(len(tj)) for b in range(a + 1, len(tj))
                    if tj[a]["started_at"] < tj[b]["finished_at"]
                    and tj[b]["started_at"] < tj[a]["finished_at"])
        check(pairs > 0, f"tuning trials ran concurrently across GPUs ({pairs} overlapping pairs)")

    print("\n  [variable importance]")
    if ns.get("IMPORTANCE_OUT") and Path(ns["IMPORTANCE_OUT"]).exists():
        imp = pd.read_csv(ns["IMPORTANCE_OUT"])
        check(set(imp["scope"]) == {"static", "encoder", "decoder"}, "all three scopes present")
        sums = imp.groupby("scope")["importance"].sum()
        check(np.allclose(sums.values, 1.0, atol=1e-4), f"importances normalised per scope: {dict(sums.round(4))}")

    print("\n  [model checkpoints persist]")
    mdir = Path(ns["MODEL_DIR"]) if ns.get("MODEL_DIR") else OUT / "models"
    cks = sorted(mdir.glob("*.ckpt")) if mdir.exists() else []
    names = [c.name for c in cks]
    check(any(n.startswith("final_") for n in names),
          f"final-fit checkpoint saved (the submission model): {names}")
    check(any(n.startswith("cv_f") for n in names),
          f"interpretation checkpoint saved: {names}")
    check(mdir.exists() and str(mdir).startswith(str(OUT)),
          "checkpoints live in the DOWNLOADABLE output dir, not scratch")
    fin = [r for r in recs if r["tag"].startswith("final_")]
    check(all(r.get("checkpoint") for r in fin),
          "final-fit results record their checkpoint path")

    print("\n  [packaging]")
    rep = OUT / f"run_report_{ns.get('EXP_ID','')}.md"
    check(rep.exists(), f"run report written ({rep.name})")
    if rep.exists():
        txt = rep.read_text()
        for heading in ("Environment", "Configuration", "Premise checks", "cross-validation",
                        "diagnostics", "variable selection", "final fit", "Job timings"):
            check(heading in txt, f"report contains '{heading}' section")
        check("section failed:" not in txt, "no section of the report errored")
        # priors are built from the TFT predictions + noise, so correlations MUST be
        # positive. A negative r means a sign flip between §9 and the report.
        check("r = **-" not in txt,
              "report residual correlations are positive (no sign flip vs §9)")
        import re as _re
        rs = [float(x) for x in _re.findall(r"r = \*\*(-?[0-9.]+)\*\*", txt)]
        check(bool(rs) and all(v > 0.5 for v in rs),
              f"report r values agree with §9 in sign and magnitude: {rs}")
        check(len(txt) > 2000, f"report is substantive ({len(txt)/1024:.0f} KB)")
    zips = list(OUT.glob("experiment12_*.zip"))
    check(len(zips) == 1, f"exactly one result bundle written ({[z.name for z in zips]})")
    check(not (OUT / "run").exists(),
          "intermediates stay OUT of the downloadable output dir")
    if zips:
        import zipfile
        with zipfile.ZipFile(zips[0]) as z:
            names = z.namelist()
        check(any(n.startswith("outputs/") and n.endswith(".csv") for n in names),
              "bundle contains result CSVs")
        check(any(n.startswith("run/logs/") for n in names), "bundle contains worker logs")
        check(any(n.startswith("run/results/") for n in names), "bundle contains job results")
        check(any(n.endswith("tft_worker.py") for n in names), "bundle contains the worker script")
        check(not any(n.endswith(".ckpt") for n in names), "checkpoints excluded by default")
        check(zips[0].stat().st_size < 60 * 2**20, f"bundle is shareable ({zips[0].stat().st_size/2**20:.1f} MB)")

    print("\n  [prior-file resolution]")
    rep = OUT / f"run_report_{ns.get('EXP_ID','')}.md"
    if rep.exists():
        txt = rep.read_text()
        check("r = " in txt or "residual correlation" in txt.lower(),
              "report contains residual correlations against priors")

    print("\n  [blend]")
    bl = OUT / f"validation_predictions_12b_blend{'' if FULL_MODE else '_preflight'}.csv"
    check(bl.exists(), "blend submission written")
    if bl.exists():
        b = pd.read_csv(bl)
        check(len(b) == N_SERIES * ehz, f"blend row count {len(b)}")
        check(np.isfinite(b["prediction"]).all(), "blend predictions finite")


if __name__ == "__main__":
    main()
