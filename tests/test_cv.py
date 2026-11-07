import pytest

from loadforecast.cv import fold_bounds, folds


def test_inner_protocol_never_selects_on_the_scored_block():
    for k in range(3):
        b = fold_bounds(4320, k, inner=True)
        assert b.train_end == b.early_stop_start
        # the early-stop block ends exactly where the scored block starts
        assert b.early_stop_start + 336 == b.score_start
        assert b.score_end - b.score_start == 336


def test_folds_tile_the_end_of_the_history():
    bounds = [fold_bounds(4320, k, inner=False) for k in range(3)]
    assert [b.score_start for b in bounds] == [3312, 3648, 3984]
    assert bounds[-1].score_end == 4320
    assert all(b.train_end == b.score_start for b in bounds)


def test_too_short_history_is_rejected():
    with pytest.raises(ValueError):
        fold_bounds(336 * 3, 0, inner=True)


def test_fold_splits_have_no_overlap_and_no_target_in_future(panel):
    for _, split in folds(panel, horizon=48):
        assert "target" not in split.future.columns
        last_seen = split.history.groupby("series_id")["timestamp"].max()
        first_scored = split.index.groupby("series_id")["timestamp"].min()
        assert (first_scored > last_seen).all()
        assert (split.index.groupby("series_id").size() == 48).all()
