"""Time-based splits: test on one calendar month, train on the X months right before it."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import pandas as pd

DATE_COL = "CloseDate"


def close_month(df: pd.DataFrame) -> pd.Series:
    return df[DATE_COL].dt.to_period("M")


def time_split(df: pd.DataFrame, test_month: str | pd.Period, train_months: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test = `test_month`; train = the `train_months` months immediately before it."""
    test_month = pd.Period(test_month, freq="M")
    month = close_month(df)
    first_train = test_month - train_months
    if first_train < month.min():
        raise ValueError(f"Not enough history: {train_months} months before {test_month} starts at {first_train}, "
                         f"data starts at {month.min()}")
    train = df[(month >= first_train) & (month < test_month)]
    test = df[month == test_month]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def validation_months(test_month: str | pd.Period, n_months: int = 1) -> list[pd.Period]:
    """The `n_months` months right before `test_month`, oldest first.

    Use these (not the test month) to choose the window length X. n_months=1 is the
    "hold out the month before test" approach; n_months>1 is a rolling backtest.
    """
    test_month = pd.Period(test_month, freq="M")
    return list(pd.period_range(test_month - n_months, test_month - 1, freq="M"))


def window_splits(df: pd.DataFrame, eval_months: Iterable[str | pd.Period],
                  windows: Iterable[int]) -> Iterator[tuple[int, pd.Period, pd.DataFrame, pd.DataFrame]]:
    """Yield (train_months, eval_month, train, eval) for every window x eval month.

    Fit preprocessing and the model on each `train` separately, so nothing from `eval` leaks in.
    """
    for x in windows:
        for month in eval_months:
            month = pd.Period(month, freq="M")
            train, evaluation = time_split(df, month, x)
            yield x, month, train, evaluation

