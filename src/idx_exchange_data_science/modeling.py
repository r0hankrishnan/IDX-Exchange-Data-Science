"""Build ClosePrice models and compare them on a rolling backtest over validation months.

    models = {
        "Linear Regression": linear_baseline(),
        "Random Forest": build_model(random_forest(), make_preprocessor(numeric=["DistrictSEDPct"])),
    }
    scores, predictions = backtest(models, df, validation_months("2026-04", 6), train_months=18)
    gain_summary(scores, reference="Linear Regression")

Every model is a Pipeline whose first step picks its own columns, so models are fit on the
whole data frame. The test month is never scored.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .cleaning import TARGET
from .preprocessing import make_preprocessor
from .splits import MonthSplit, split


def build_model(estimator: BaseEstimator, preprocessor: BaseEstimator | None = None, *,
                scale: bool = False, log_target: bool = False) -> BaseEstimator:
    """Preprocessor -> (scaler) -> estimator. With log_target, fits on log1p(price) and predicts dollars."""
    steps = [preprocessor if preprocessor is not None else make_preprocessor()]
    if scale:
        steps.append(StandardScaler().set_output(transform="pandas"))
    model = make_pipeline(*steps, estimator)
    if log_target:
        return TransformedTargetRegressor(model, func=np.log1p, inverse_func=np.expm1)
    return model


def random_forest() -> RandomForestRegressor:
    """The forest used since week 5: 200 trees, leaves of 2 or more, a third of the features per split."""
    return RandomForestRegressor(n_estimators=200, min_samples_leaf=2, max_features=1 / 3, n_jobs=-1, random_state=0)


def tuned_xgboost(**overrides) -> XGBRegressor:
    """XGBoost with the settings from the random search in 06 (tuning month 2025-09).

    The team's unified pipeline doc lists nearly the same settings (learning rate 0.1).
    Keyword arguments override any setting.
    """
    settings = dict(tree_method="hist", n_estimators=600, learning_rate=0.092, max_depth=8, min_child_weight=1,
                    subsample=1.0, colsample_bytree=0.5, n_jobs=-1, random_state=0)
    return XGBRegressor(**{**settings, **overrides})


def linear_baseline(numeric: Iterable[str] = (), categorical: Iterable[str] = ()) -> BaseEstimator:
    """The week 4 baseline: linear regression with log target and log size features."""
    return build_model(LinearRegression(), make_preprocessor(numeric, categorical, log_features=True),
                       scale=True, log_target=True)


def final_estimator(model: BaseEstimator) -> BaseEstimator:
    """The fitted last step (e.g. the forest), whether or not the model is wrapped for a log target."""
    pipeline = model.regressor_ if isinstance(model, TransformedTargetRegressor) else model
    return pipeline[-1]


def median_ape(y_true, y_pred) -> float:
    """Median absolute percentage error: the decision metric (a typical home's error)."""
    return np.median(np.abs(np.asarray(y_pred) / np.asarray(y_true) - 1))


def within_10pct(y_true, y_pred) -> float:
    """Share of homes predicted within 10% of the sale price."""
    return np.mean(np.abs(np.asarray(y_pred) / np.asarray(y_true) - 1) < 0.1)


# Dollar-scale metrics, name -> function(y_true, y_pred). mape is the *mean* absolute % error.
METRICS = {
    "r2": r2_score,
    "mae": mean_absolute_error,
    "mape": mean_absolute_percentage_error,
    "median_ape": median_ape,
    "within_10pct": within_10pct,
}


def score(y_true, y_pred) -> dict[str, float]:
    return {name: float(metric(y_true, y_pred)) for name, metric in METRICS.items()}


def backtest(models: dict[str, BaseEstimator], df: pd.DataFrame, months: Iterable[str | pd.Period],
             train_months: int, gap: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a fresh copy of each model on the `train_months` before each month, then score it on that month.

    Returns the scores (one row per model and month) and the predictions (one column per
    model, indexed like the evaluated rows of `df`). Scores on the evaluated month are named
    after the metric (r2, median_ape, ...). The same metrics on the model's own training rows
    are prefixed with train_, to show overfitting. Like sklearn's return_train_score, these are
    in-sample, so they are optimistic.
    """
    cv = MonthSplit(months, train_months, gap)
    y = df[TARGET]
    scores, predictions = [], []
    for month, (train_rows, eval_rows) in zip(cv.months, cv.split(df, y)):
        train, evaluation = df.iloc[train_rows], df.iloc[eval_rows]
        month_predictions = pd.DataFrame(index=evaluation.index)
        for name, model in models.items():
            fitted = clone(model).fit(train, train[TARGET])
            month_predictions[name] = fitted.predict(evaluation)
            train_scores = score(train[TARGET], fitted.predict(train))
            scores.append({"model": name, "month": str(month), "n_train": len(train),
                           **score(evaluation[TARGET], month_predictions[name]),
                           **{f"train_{metric}": value for metric, value in train_scores.items()}})
        predictions.append(month_predictions)
    return pd.DataFrame(scores), pd.concat(predictions)


def fit_window(model: BaseEstimator, df: pd.DataFrame, month: str | pd.Period, train_months: int,
               gap: int = 0) -> BaseEstimator:
    """A fresh copy of `model` fit on the `train_months` before `month` (e.g. to look at its coefficients)."""
    train, _ = split(df, month, train_months, gap)
    return clone(model).fit(train, train[TARGET])


def feature_importances(model: BaseEstimator) -> pd.Series:
    """A fitted tree model's feature importances, labeled with the preprocessed column names."""
    estimator = final_estimator(model)
    return pd.Series(estimator.feature_importances_, index=estimator.feature_names_in_)


def median_ape_by(predictions: pd.DataFrame, actual: pd.Series, groups: pd.Series) -> pd.DataFrame:
    """Median absolute % error of each prediction column within each group (e.g. price decile), in percent."""
    pct_error = predictions.sub(actual, axis=0).abs().div(actual, axis=0) * 100
    return pct_error.groupby(groups, observed=True).median()


def gain_summary(scores: pd.DataFrame, reference: str, metric: str = "r2") -> pd.DataFrame:
    """Each model's month-by-month change in `metric` from `reference`: mean, spread, and months improved.

    For a metric where lower is better (mae, mape, median_ape), a negative gain is an improvement.
    """
    by_month = scores.pivot(index="month", columns="model", values=metric)
    gain = by_month.drop(columns=reference).sub(by_month[reference], axis=0)
    improved = gain < 0 if metric in ("mae", "mape", "median_ape") else gain > 0
    return pd.DataFrame({"mean gain": gain.mean(), "std gain": gain.std(),
                         f"months improved (of {len(gain)})": improved.sum()})
