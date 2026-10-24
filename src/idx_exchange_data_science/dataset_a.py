"""Dataset A: the team's agreed preprocessing (see preprocessing-procedures.md).

Fixed rules run once on the whole dataset and produce the cleaned CSV. Anything
learned from data (ClosePrice trimming, imputation, target encoding) is fit on the
training window only.

    df = clean(load_raw("data/intermediate/CRMLSSoldFiltered.csv"))
    train, test = split(df, test_month="2026-04", train_months=6)

    model = make_pipeline(DatasetAPreprocessor(), LinearRegression())
    model.fit(train[FEATURES], train[TARGET])
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder

from .splits import time_split

TARGET = "ClosePrice"
DATE_COL = "CloseDate"
ID_COL = "ListingKey"

PRICE_QUANTILES = (0.005, 0.995)
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


def clean(df: pd.DataFrame, steps: Iterable[Callable] = CLEANING_STEPS) -> pd.DataFrame:
    for step in steps:
        df = step(df)
    return df


def load_clean(path: str | Path) -> pd.DataFrame:
    """Read a cleaned CSV written by `clean(...).to_csv(...)`."""
    return pd.read_csv(path, parse_dates=[DATE_COL], dtype={"PostalCode": "string", "City": "string"})


# =============================================================================
# Split with training-derived ClosePrice limits
# =============================================================================


def price_limits(train: pd.DataFrame, quantiles: tuple[float, float] = PRICE_QUANTILES) -> tuple[float, float]:
    low, high = train[TARGET].quantile(quantiles)
    return low, high


def split(df: pd.DataFrame, test_month: str | pd.Period, train_months: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time split, then drop ClosePrice outliers from both sides using limits from the training window."""
    train, test = time_split(df, test_month, train_months)
    low, high = price_limits(train)
    return train[train[TARGET].between(low, high)], test[test[TARGET].between(low, high)]


# =============================================================================
# Fitted preprocessing
# =============================================================================


class DatasetAPreprocessor(TransformerMixin, BaseEstimator):
    """Impute, flag, and target-encode the Dataset A features. Learns everything from the training rows.

    Works as the first step of an sklearn Pipeline. The target encoder is cross-fitted
    (5 folds) on training rows, so a row's own price never goes into its encoding.
    It encodes whatever target the pipeline is fit on, so with a log target the
    encodings are means of log price.

    log_features: log1p LivingArea and LotSizeSquareFeet (experiment for linear models).
    """

    def __init__(self, log_features: bool = False, random_state: int = 0):
        self.log_features = log_features
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> DatasetAPreprocessor:
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.medians_ = X[MEDIAN_COLS].median()
        self.modes_ = X[MODE_COLS + BINARY_COLS].mode().iloc[0]
        self.geo_by_zip_ = X.groupby("PostalCode")[GEO_COLS].median()
        self.geo_by_county_ = X.groupby("CountyOrParish")[GEO_COLS].median()
        self.geo_overall_ = X[GEO_COLS].median()

        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        self.encoder_ = TargetEncoder(target_type="continuous", cv=cv)
        encoded = self.encoder_.fit_transform(self._categories(X), y)
        return self._finish(X, encoded)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._finish(X, self.encoder_.transform(self._categories(X)))

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.feature_names_out_)

    @staticmethod
    def _categories(X: pd.DataFrame) -> pd.DataFrame:
        return X[CATEGORICAL_COLS].astype("object").fillna(UNKNOWN)

    def _finish(self, X: pd.DataFrame, encoded: np.ndarray) -> pd.DataFrame:
        out = X[MEDIAN_COLS + MODE_COLS + BINARY_COLS + GEO_COLS].copy()
        for col in BINARY_COLS:
            out[f"{col}_missing"] = out[col].isna().astype(float)

        out[MEDIAN_COLS] = out[MEDIAN_COLS].fillna(self.medians_)
        out[MODE_COLS + BINARY_COLS] = out[MODE_COLS + BINARY_COLS].fillna(self.modes_)
        for col in GEO_COLS:
            out[col] = (
                out[col]
                .fillna(X["PostalCode"].map(self.geo_by_zip_[col]))
                .fillna(X["CountyOrParish"].map(self.geo_by_county_[col]))
                .fillna(self.geo_overall_[col])
            )
        if self.log_features:
            out[LOG_COLS] = np.log1p(out[LOG_COLS])

        for i, col in enumerate(CATEGORICAL_COLS):
            out[f"{col}_te"] = encoded[:, i]
        self.feature_names_out_ = list(out.columns)
        return out
