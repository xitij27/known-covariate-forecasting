import numpy as np
import pandas as pd
import pytest

from conftest import split_last
from loadforecast import baselines, cv, lgbm
from loadforecast.features import add_covariate_context
from loadforecast.metrics import wape


def test_wape():
    assert wape([1, 2, 3], [1, 2, 3]) == 0
    assert wape([10, 10], [12, 8]) == pytest.approx(20.0)


def test_leads_and_lags_stay_inside_each_series(panel):
    frame, _ = add_covariate_context(panel, ["shock_risk"], [], lags=(1,), leads=(1,))
    for _, g in frame.groupby("series_id"):
        assert np.isnan(g["shock_risk_lag1"].iloc[0])
        assert np.isnan(g["shock_risk_lead1"].iloc[-1])
        np.testing.assert_allclose(g["shock_risk_lead1"].iloc[:-1], g["shock_risk"].iloc[1:])


def test_naive_repeats_the_last_value(panel):
    split = split_last(panel, 24)
    pred = baselines.naive_last_value(split)
    last = split.history.groupby("series_id")["target"].last()
    assert (pred.groupby("series_id")["prediction"].nunique() == 1).all()
    np.testing.assert_allclose(pred.groupby("series_id")["prediction"].first(), last)


def test_seasonal_mean_recovers_a_weekly_profile():
    t = pd.date_range("2023-01-02", periods=24 * 7 * 4, freq="h")
    profile = (t.dayofweek * 24 + t.hour).to_numpy(float)
    panel = pd.DataFrame({"series_id": "unit_000", "timestamp": t, "target": profile})
    split = split_last(panel, 24 * 7)
    pred = baselines.seasonal_mean(split)
    np.testing.assert_allclose(pred["prediction"], split.index["target"])


def test_lgbm_covers_the_index_and_uses_the_covariates(panel):
    split = split_last(panel, 48)
    cfg = lgbm.LGBMConfig(params=dict(n_estimators=100, learning_rate=0.1, num_leaves=15))
    pred = lgbm.forecast(split, cfg)
    assert len(pred) == len(split.index)
    assert pred["prediction"].notna().all() and (pred["prediction"] >= 0).all()
    # the synthetic target is driven by covariates, which the seasonal profile cannot see
    seasonal = baselines.seasonal_mean(split)
    assert wape(split.index["target"], pred["prediction"]) < wape(
        split.index["target"], seasonal["prediction"]
    )


def test_cv_returns_one_score_per_fold(panel):
    scores, oof = cv.evaluate(panel, baselines.seasonal_mean, horizon=48, verbose=False)
    assert len(scores) == 3
    assert sorted(oof["fold"].unique()) == [0, 1, 2]
