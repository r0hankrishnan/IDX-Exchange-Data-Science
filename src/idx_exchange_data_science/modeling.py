"""Build, backtest, and score ClosePrice models on time-based splits.

    models = {
        "linear": build_model(LinearRegression(), scale=True),
        "forest": build_model(RandomForestRegressor(), log_target=True),
    }
    results = backtest(models, df, validation_months("2026-04", 3), windows=[1, 3, 6, 12])
    best = best_windows(results)                # chosen X per model
    test_results = score_on_test(models, df, "2026-04", best)

Swap the preprocessing by passing `preprocessor=` to build_model, or the split rule
by passing `split=` to backtest / score_on_test.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from . import dataset_a

Split = Callable[[pd.DataFrame, pd.Period, int], tuple[pd.DataFrame, pd.DataFrame]]


def build_model(
    estimator: BaseEstimator,
    *,
    scale: bool = False,
    log_target: bool = False,
    log_features: bool = False,
    preprocessor: BaseEstimator | None = None,
) -> BaseEstimator:
    """Preprocessor -> (scaler) -> estimator. With log_target, fits on log1p(price) and predicts dollars."""
    steps = [preprocessor or dataset_a.DatasetAPreprocessor(log_features=log_features)]
    if scale:
        steps.append(StandardScaler().set_output(transform="pandas"))
    model = make_pipeline(*steps, estimator)
    if log_target:
        return TransformedTargetRegressor(model, func=np.log1p, inverse_func=np.expm1)
    return model


def fitted_pipeline(model: BaseEstimator) -> Pipeline:
    """The fitted Pipeline inside a model, whether or not it is wrapped for a log target."""
    return model.regressor_ if isinstance(model, TransformedTargetRegressor) else model


def score(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }


def fit_and_score(model: BaseEstimator, train: pd.DataFrame, test: pd.DataFrame,
                  features: list[str] = dataset_a.FEATURES, target: str = dataset_a.TARGET):
    """Fit a fresh copy of `model` on train. Returns the fitted model, train scores, and test scores."""
    model = clone(model).fit(train[features], train[target])
    train_scores = score(train[target], model.predict(train[features]))
    test_scores = score(test[target], model.predict(test[features]))
    return model, train_scores, test_scores


def backtest(models: dict[str, BaseEstimator], df: pd.DataFrame, eval_months: Iterable[str | pd.Period],
             windows: Iterable[int], split: Split = dataset_a.split) -> pd.DataFrame:
    """Score every model for every training window X on every evaluation month."""
    rows = []
    for x in windows:
        for month in eval_months:
            train, evaluation = split(df, pd.Period(month, freq="M"), x)
            for name, model in models.items():
                _, _, scores = fit_and_score(model, train, evaluation)
                rows.append({"model": name, "X": x, "month": str(month), "n_train": len(train), **scores})
    return pd.DataFrame(rows)


def best_windows(results: pd.DataFrame, metric: str = "r2") -> dict[str, int]:
    """For each model, the X with the best mean `metric` across evaluation months."""
    mean = results.groupby(["model", "X"])[metric].mean()
    best = mean.groupby("model").idxmin() if metric in ("mae", "mape") else mean.groupby("model").idxmax()
    return {model: x for model, x in best}


def score_on_test(models: dict[str, BaseEstimator], df: pd.DataFrame, test_month: str | pd.Period,
                  windows: dict[str, int], split: Split = dataset_a.split):
    """Fit each model on its chosen window and score it once on the test month.

    Returns a results table (train and test scores) and the fitted models.
    """
    rows, fitted = [], {}
    for name, model in models.items():
        train, test = split(df, pd.Period(test_month, freq="M"), windows[name])
        fitted[name], train_scores, test_scores = fit_and_score(model, train, test)
        rows.append({
            "model": name, "X": windows[name], "n_train": len(train), "n_test": len(test),
            "train_r2": train_scores["r2"],
            **{f"test_{k}": v for k, v in test_scores.items()},
        })
    return pd.DataFrame(rows), fitted
