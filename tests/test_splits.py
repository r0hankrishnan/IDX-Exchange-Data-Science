import pandas as pd
import pytest

from idx_exchange_data_science import dataset_a as da
from idx_exchange_data_science.splits import MonthSplit, PRICE_QUANTILES, split, validation_months


def months(frame: pd.DataFrame) -> set[str]:
    return set(frame[da.DATE_COL].dt.to_period("M").astype(str))


def test_split_uses_only_the_months_before_evaluation(clean_df):
    train, evaluation = split(clean_df, "2024-08", train_months=3, trim=False)
    assert months(train) == {"2024-05", "2024-06", "2024-07"}
    assert months(evaluation) == {"2024-08"}


def test_gap_skips_the_months_before_evaluation(clean_df):
    train, evaluation = split(clean_df, "2024-08", train_months=3, gap=2, trim=False)
    assert months(train) == {"2024-03", "2024-04", "2024-05"}
    assert months(evaluation) == {"2024-08"}


def test_split_needs_enough_history(clean_df):
    with pytest.raises(ValueError):
        split(clean_df, "2024-03", train_months=6)


def test_price_limits_come_from_train_only(clean_df):
    _, before = split(clean_df, "2024-08", 3)
    in_august = clean_df[da.DATE_COL].dt.to_period("M") == pd.Period("2024-08")
    shifted = clean_df.copy()
    shifted.loc[in_august, da.TARGET] *= 1000  # wild evaluation prices must not move the limits
    train, after = split(shifted, "2024-08", 3)

    low, high = split(clean_df, "2024-08", 3, trim=False)[0][da.TARGET].quantile(PRICE_QUANTILES)
    assert train[da.TARGET].between(low, high).all()
    assert len(before) > 0 and len(after) == 0  # every inflated price is above the training limit


def test_validation_months_stop_before_the_test_month():
    assert [str(m) for m in validation_months("2026-04", 3)] == ["2026-01", "2026-02", "2026-03"]


def test_month_split_matches_split_and_never_trains_on_the_evaluation_month(clean_df):
    cv = MonthSplit(["2024-06", "2024-07"], train_months=3)
    folds = list(cv.split(clean_df, clean_df[da.TARGET]))
    assert len(folds) == cv.get_n_splits() == 2
    for month, (train_rows, eval_rows) in zip(cv.months, folds):
        train, evaluation = split(clean_df, month, 3)
        assert clean_df.index[train_rows].equals(train.index)
        assert clean_df.index[eval_rows].equals(evaluation.index)
        assert max(train[da.DATE_COL].dt.to_period("M")) < month
