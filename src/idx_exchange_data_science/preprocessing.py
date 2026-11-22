"""Fitted preprocessing: imputation, missing flags and target encoding, as one sklearn ColumnTransformer.

It is the first step of every model Pipeline, so everything it learns comes from the training rows:

    model = make_pipeline(make_preprocessor(numeric=["DistrictSEDPct"]), RandomForestRegressor())
    model.fit(train, train[TARGET])
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, TargetEncoder

from .cleaning import BINARY_COLS, CATEGORICAL_COLS, GEO_COLS, LOG_COLS, MEDIAN_COLS, MODE_COLS, UNKNOWN


def make_preprocessor(numeric: Iterable[str] = (), categorical: Iterable[str] = (),
                      log_features: bool = False) -> ColumnTransformer:
    """Impute, flag, and target-encode the Dataset A features, plus any extra columns.

    The first step of a model Pipeline, so everything is learned from the training rows.
    It picks its columns by name and drops the rest (ClosePrice, CloseDate, ...), so a
    model can be fit on the whole data frame.

    numeric / categorical: columns beyond FEATURES, median-imputed or target-encoded like
    the Dataset A ones (e.g. the columns from features.py).
    log_features: log1p LivingArea and LotSizeSquareFeet (experiment for linear models).

    The target encoder is cross-fitted over 5 folds of the training rows, so a row's own
    price never goes into its encoding. It encodes whatever target the model is fit on:
    with a log target, the encodings are means of log price.
    """
    median_cols = MEDIAN_COLS + list(numeric)
    if log_features:
        median_cols = [c for c in median_cols if c not in LOG_COLS]
    categorical_cols = CATEGORICAL_COLS + list(categorical)

    steps = [
        ("median", SimpleImputer(strategy="median"), median_cols),
        ("mode", SimpleImputer(strategy="most_frequent"), MODE_COLS + BINARY_COLS),
        ("missing", MissingIndicator(features="all"), BINARY_COLS),
        ("geo", GeoImputer(), GEO_COLS + ["PostalCode", "CountyOrParish"]),
        ("target", make_pipeline(
            FunctionTransformer(fill_unknown, feature_names_out="one-to-one"),
            TargetEncoder(target_type="continuous", cv=KFold(n_splits=5, shuffle=True, random_state=0)),
        ), categorical_cols),
    ]
    if log_features:
        log = make_pipeline(SimpleImputer(strategy="median"),
                            FunctionTransformer(np.log1p, feature_names_out="one-to-one"))
        steps.insert(1, ("log", log, LOG_COLS))
    return ColumnTransformer(steps, verbose_feature_names_out=False).set_output(transform="pandas")


def fill_unknown(categories: pd.DataFrame) -> pd.DataFrame:
    return categories.astype("object").fillna(UNKNOWN)


class GeoImputer(TransformerMixin, BaseEstimator):
    """Fill missing Latitude/Longitude with the training median of the same ZIP, then county, then overall.

    Takes the coordinates plus PostalCode and CountyOrParish; returns the coordinates only.
    """

    def fit(self, X: pd.DataFrame, y=None) -> GeoImputer:
        self.by_zip_ = X.groupby("PostalCode")[GEO_COLS].median()
        self.by_county_ = X.groupby("CountyOrParish")[GEO_COLS].median()
        self.overall_ = X[GEO_COLS].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[GEO_COLS].copy()
        for col in GEO_COLS:
            out[col] = (
                out[col]
                .fillna(X["PostalCode"].map(self.by_zip_[col]))
                .fillna(X["CountyOrParish"].map(self.by_county_[col]))
                .fillna(self.overall_[col])
            )
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(GEO_COLS, dtype=object)
