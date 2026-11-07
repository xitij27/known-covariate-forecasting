"""End-to-end TFT on a tiny panel: train two seeds, write a bundle, predict from it."""

import pytest

pytest.importorskip("pytorch_forecasting")

from conftest import split_last  # noqa: E402
from loadforecast import tft  # noqa: E402

pytestmark = pytest.mark.tft

TINY = dict(horizon=24, min_encoder=24, max_encoder=48, stride=24, hidden_size=8,
            hidden_continuous_size=4, attention_heads=1, batch_size=8, accumulate=1,
            eval_batch=8, final_epochs=1, max_epochs=2, patience=1, seeds=[1, 2])


@pytest.mark.parametrize("stride", [24, 168])
def test_training_windows_reach_the_cutoff_and_not_beyond(panel, stride):
    # 168 does not divide the last decoder start; a grid anchored at time 0 would drop the
    # final window, which is the bug that cost experiment 11 three WAPE points.
    split = split_last(panel, 24)
    data, t_train = tft.prepare_panel(split)
    ds = tft.training_dataset(data, tft.TFTConfig(**{**TINY, "stride": stride}), t_train)
    last_target = int(ds.decoded_index["time_idx_first_prediction"].max()) + 24 - 1
    assert last_target == t_train - 1


def test_fit_bundle_predict(panel, tmp_path):
    split = split_last(panel, 24)
    cfg = tft.TFTConfig(**TINY)
    bundle = tft.train_final(split, cfg, tmp_path)
    assert len(tft.load_bundle(bundle)) == 2
    pred = tft.forecast(split, bundle, cfg)
    assert len(pred) == len(split.index)
    assert pred["prediction"].notna().all()


def test_cross_validation_runs(panel):
    scores, oof, log = tft.cross_validate(split_last(panel, 24), tft.TFTConfig(**TINY), seeds=[1])
    assert len(scores) == 3 and all(s > 0 for s in scores)
    assert all(entry["train_end"] < 24 * 39 for entry in log)
