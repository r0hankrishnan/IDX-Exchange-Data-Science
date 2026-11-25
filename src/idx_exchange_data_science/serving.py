"""The final model: defined once, trained on the latest data, saved, and loaded by the dashboard.

    uv run python -m idx_exchange_data_science.serving   # train on the latest 18 months and save

The model is chosen in the notebooks on the validation months. This module only serves the
winner. To change it, edit `final_model()` and retrain.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import TransformedTargetRegressor

from . import features as ft
from .cleaning import DATE_COL, TARGET
from .modeling import backtest, build_model, score, tuned_xgboost
from .paths import MODELS_DIR
from .preprocessing import make_preprocessor
from .splits import split, validation_months

MODEL_PATH = MODELS_DIR / "final_model.joblib"
TRAIN_MONTHS = 18
N_VALIDATION_MONTHS = 6
# Price ranges: the central 80%, 90% and 95% of validation homes' actual / predicted ratios.
RANGE_LEVELS = (0.80, 0.90, 0.95)
NUMERIC = ft.DISTRICT_NUMERIC_COLS + ft.HOA_COLS + ft.FLOORING_COLS + ft.COMP_COLS + ft.COMP_PRICE_COLS
CATEGORICAL = ft.DISTRICT_COLS


def final_model() -> BaseEstimator:
    """Model C, the best single model in 07_final_model_selection (validation MdAPE 7.02%, lower than the earlier
    served model in 6 of 6 months): XGBoost with the settings tuned in 06, on Dataset A + school districts + HOA +
    flooring + comparable sales + their median sale price, fit on log price."""
    return build_model(tuned_xgboost(), make_preprocessor(NUMERIC, CATEGORICAL), log_target=True)


def train_final_model(df: pd.DataFrame, model: BaseEstimator | None = None, train_months: int = TRAIN_MONTHS,
                      n_validation_months: int = N_VALIDATION_MONTHS) -> dict:
    """Fit the model on the `train_months` up to and including the latest month in `df`.

    Also measures how it does on unseen homes: the same model refit before each of the
    `n_validation_months` before the latest month (the usual validation backtest). Those
    out-of-sample errors give its accuracy and its price ranges: for each level in RANGE_LEVELS,
    the central share of the actual / predicted ratios (split conformal, not a bootstrap).
    """
    model = model if model is not None else final_model()
    latest = df[DATE_COL].dt.to_period("M").max()
    train, _ = split(df, latest + 1, train_months)
    fitted = clone(model).fit(train, train[TARGET])

    months = validation_months(latest, n_validation_months)
    _, predictions = backtest({"final": model}, df, months, train_months)
    actual = df.loc[predictions.index, TARGET]
    ratio = actual / predictions["final"]
    return {
        "model": fitted,
        "train_first": train[DATE_COL].min(),
        "train_last": train[DATE_COL].max(),
        "n_train": len(train),
        "validation_months": [str(m) for m in months],
        "validation": score(actual, predictions["final"]),
        "ranges": {level: (ratio.quantile((1 - level) / 2), ratio.quantile((1 + level) / 2)) for level in RANGE_LEVELS},
    }


def input_columns(model: BaseEstimator) -> set[str]:
    """The columns a fitted model reads (its preprocessor picks them from the full table)."""
    pipeline = model.regressor_ if isinstance(model, TransformedTargetRegressor) else model
    used = [cols for _, transformer, cols in pipeline[0].transformers_ if transformer != "drop"]
    return {col for cols in used for col in cols}


def save(bundle: dict, path: str | Path = MODEL_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load(path: str | Path = MODEL_PATH) -> dict:
    return joblib.load(path)


if __name__ == "__main__":
    bundle = train_final_model(ft.build_table())
    save(bundle)
    v = bundle["validation"]
    print(f"Trained on {bundle['n_train']:,} sales, {bundle['train_first']:%Y-%m} to {bundle['train_last']:%Y-%m}. "
          f"Validation ({bundle['validation_months'][0]} to {bundle['validation_months'][-1]}): R² {v['r2']:.4f}, "
          f"median APE {v['median_ape']:.1%}, within 10% {v['within_10pct']:.1%}. "
          f"Ranges (x estimate): { {k: np.round(v, 3).tolist() for k, v in bundle['ranges'].items()} }. Saved to {MODEL_PATH}")
