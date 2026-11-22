"""Predict ClosePrice for California single-family homes before they are listed (CRMLS sales).

Modules, in the order the data flows through them:

    paths          where the data, results and saved models live
    cleaning       Dataset A: fixed, row-by-row cleaning rules (no fitting)
    features       build_table(): Dataset A + school districts, HOA, flooring, comparable sales, ...
    splits         time-based splits: train on the X months before each evaluated month
    preprocessing  make_preprocessor(): fitted imputation and target encoding (sklearn ColumnTransformer)
    modeling       model builders, backtest() over validation months, score(), gain_summary()
    serving        the one final model: train_final_model(), save(), load()

The most used names are importable from the package itself:

    from idx_exchange_data_science import build_table, build_model, make_preprocessor, backtest
"""

from .cleaning import DATE_COL, FEATURES, ID_COL, TARGET, clean, load_clean, load_raw
from .features import build_table
from .modeling import backtest, build_model, gain_summary, linear_baseline, random_forest, score, tuned_xgboost
from .preprocessing import make_preprocessor
from .splits import MonthSplit, split, validation_months

__all__ = [
    "DATE_COL", "FEATURES", "ID_COL", "TARGET", "clean", "load_clean", "load_raw",
    "build_table",
    "backtest", "build_model", "gain_summary", "linear_baseline", "random_forest", "score", "tuned_xgboost",
    "make_preprocessor",
    "MonthSplit", "split", "validation_months",
]
