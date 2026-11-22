"""Time-based splits: evaluate on one calendar month, train on the X months right before it.

ClosePrice outliers are trimmed from both sides with the 0.5% / 99.5% percentiles of the
training window (a Dataset A rule), so the limits never come from the evaluation month.

    train, evaluation = split(df, "2026-03", train_months=18)

MonthSplit does the same for several months and works as `cv=` in sklearn's
cross_validate or RandomizedSearchCV:

    cv = MonthSplit(validation_months("2026-04", 6), train_months=18)
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
import pandas as pd

from .cleaning import DATE_COL, TARGET

PRICE_QUANTILES = (0.005, 0.995)


def validation_months(test_month: str | pd.Period, n_months: int) -> list[pd.Period]:
    """The `n_months` months right before `test_month`, oldest first. The test month itself is never used."""
    test_month = pd.Period(test_month, freq="M")
    return list(pd.period_range(test_month - n_months, test_month - 1, freq="M"))


def month_masks(dates: pd.Series, prices: pd.Series, month: str | pd.Period, train_months: int,
                gap: int = 0, trim: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks for the training rows and the evaluation rows.

    Train = the `train_months` months ending `gap` months before `month`. gap=2 with month
    2026-03 trains on months up to 2025-12, as if the prediction were made 2 months earlier.
    """
    month = pd.Period(month, freq="M")
    close_month = dates.dt.to_period("M")
    last_train = month - gap
    first_train = last_train - train_months
    if first_train < close_month.min():
        raise ValueError(f"Not enough history: {train_months} months before {month} starts at {first_train}, "
                         f"data starts at {close_month.min()}")
    train = (close_month >= first_train) & (close_month < last_train)
    evaluation = close_month == month
    if trim:
        low, high = prices[train].quantile(PRICE_QUANTILES)
        in_range = prices.between(low, high)
        train, evaluation = train & in_range, evaluation & in_range
    return train.to_numpy(), evaluation.to_numpy()


def split(df: pd.DataFrame, month: str | pd.Period, train_months: int,
          gap: int = 0, trim: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, evaluation) frames for one month. Rows keep their index in `df`."""
    train, evaluation = month_masks(df[DATE_COL], df[TARGET], month, train_months, gap, trim)
    return df[train], df[evaluation]


class MonthSplit:
    """sklearn CV splitter: one fold per month in `months`, trained on the `train_months` before it.

    split(X, y) needs X to have the CloseDate column and y to be ClosePrice in dollars
    (the trim limits are in dollars, even for a log-target model).
    """

    def __init__(self, months: Iterable[str | pd.Period], train_months: int, gap: int = 0):
        self.months = [pd.Period(m, freq="M") for m in months]
        self.train_months = train_months
        self.gap = gap

    def split(self, X: pd.DataFrame, y: pd.Series, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for month in self.months:
            train, evaluation = month_masks(X[DATE_COL], y, month, self.train_months, self.gap)
            yield np.flatnonzero(train), np.flatnonzero(evaluation)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return len(self.months)
