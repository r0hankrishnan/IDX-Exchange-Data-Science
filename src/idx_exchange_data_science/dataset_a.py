"""Dataset A: the team's agreed preprocessing (see preprocessing-procedures.md).

Fixed rules run once on the whole dataset and produce the cleaned CSV. Anything
learned from data is fit on the training window only: the ClosePrice trim (in
splits.py), and imputation and target encoding (make_preprocessor).

    df = clean(load_raw("data/intermediate/CRMLSSoldFiltered.csv"))
    train, evaluation = splits.split(df, "2026-03", train_months=18)

    model = make_pipeline(make_preprocessor(), LinearRegression())
    model.fit(train, train[TARGET])
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, TargetEncoder

TARGET = "ClosePrice"
DATE_COL = "CloseDate"
ID_COL = "ListingKey"

MAX_MISSING_SHARE = 0.8

# Values outside these inclusive ranges become NaN and are imputed later.
VALID_RANGES = {
    "BedroomsTotal": (1, 20),
    "BathroomsTotalInteger": (1, 20),
    "GarageSpaces": (0, 20),
    "ParkingTotal": (0, 20),
    "Latitude": (32.5, 42.1),
    "Longitude": (-124.5, -114.0),
}
SQFT_PER_ACRE = 43_560

MEDIAN_COLS = ["LivingArea", "PropertyAge", "LotSizeSquareFeet"]
MODE_COLS = ["BedroomsTotal", "BathroomsTotalInteger", "GarageSpaces", "ParkingTotal"]
BINARY_COLS = ["AttachedGarageYN", "FireplaceYN", "NewConstructionYN", "PoolPrivateYN", "ViewYN"]
GEO_COLS = ["Latitude", "Longitude"]
CATEGORICAL_COLS = ["City", "CountyOrParish", "PostalCode"]
LOG_COLS = ["LivingArea", "LotSizeSquareFeet"]
UNKNOWN = "Unknown"

FEATURES = MEDIAN_COLS + MODE_COLS + BINARY_COLS + GEO_COLS + CATEGORICAL_COLS


# =============================================================================
# Fixed cleaning rules (no fitting, safe to run before the split)
# =============================================================================


def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])


def keep_scope(df: pd.DataFrame) -> pd.DataFrame:
    """California, Residential, SingleFamilyResidence."""
    keep = (
        (df["StateOrProvince"] == "CA")
        & (df["PropertyType"] == "Residential")
        & (df["PropertySubType"] == "SingleFamilyResidence")
    )
    return df[keep]


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        CloseDate=pd.to_datetime(df["CloseDate"], errors="coerce"),
        ListingContractDate=pd.to_datetime(df["ListingContractDate"], errors="coerce"),
    )


def drop_invalid_prices(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[TARGET] > 0]


def drop_closed_before_listed(df: pd.DataFrame) -> pd.DataFrame:
    return df[~(df["CloseDate"] < df["ListingContractDate"])]


def normalize_address(address: pd.Series) -> pd.Series:
    """Lowercase, trim, collapse spaces. Addresses with no street number become NaN."""
    normalized = address.str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
    return normalized.where(normalized.str.contains(r"\d", na=False))


def drop_duplicate_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Same address and close date = same sale; keep the first. Rows without a valid address are dropped."""
    address = normalize_address(df["UnparsedAddress"])
    df = df[address.notna()]
    return df[~df.assign(address=address).duplicated(["address", "CloseDate"])]


def drop_nonpositive_living_area(df: pd.DataFrame) -> pd.DataFrame:
    return df[~(df["LivingArea"] <= 0)]


def fix_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize formats, null out implausible values, and derive PropertyAge."""
    out = df.copy()
    out["PostalCode"] = out["PostalCode"].astype("string").str.extract(r"^(\d{5})", expand=False)
    for col, (low, high) in VALID_RANGES.items():
        out[col] = out[col].where(out[col].between(low, high))

    lot_sqft = out["LotSizeSquareFeet"].where(out["LotSizeSquareFeet"] > 0)
    lot_from_acres = out["LotSizeAcres"].where(out["LotSizeAcres"] > 0) * SQFT_PER_ACRE
    out["LotSizeSquareFeet"] = lot_sqft.fillna(lot_from_acres)

    out["PropertyAge"] = out["CloseDate"].dt.year - out["YearBuilt"]
    for col in BINARY_COLS:
        out[col] = out[col].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
    return out


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the id, date, target, and the agreed features.

    The agreed rule drops columns over 80% missing. That excludes every raw column
    that isn't on the feature list, so here it is only a check on the list itself.
    """
    too_sparse = [c for c in FEATURES if df[c].isna().mean() > MAX_MISSING_SHARE]
    if too_sparse:
        raise ValueError(f"Over {MAX_MISSING_SHARE:.0%} missing, remove from FEATURES: {too_sparse}")
    return df[[ID_COL, DATE_COL, TARGET, *FEATURES]].reset_index(drop=True)


CLEANING_STEPS: list[Callable[[pd.DataFrame], pd.DataFrame]] = [
    keep_scope,
    parse_dates,
    drop_invalid_prices,
    drop_closed_before_listed,
    drop_duplicate_sales,
    drop_nonpositive_living_area,
    fix_fields,
    select_columns,
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    for step in CLEANING_STEPS:
        df = step(df)
    return df


def load_clean(path: str | Path) -> pd.DataFrame:
    """Read a cleaned CSV written by `clean(...).to_csv(...)`."""
    return pd.read_csv(path, parse_dates=[DATE_COL], dtype={"PostalCode": "string", "City": "string"})


# =============================================================================
# Fitted preprocessing
# =============================================================================


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
