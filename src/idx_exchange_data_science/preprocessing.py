"""Preprocessing for the CRMLS single-family AVM (predict ClosePrice before listing).

Three stages, each chainable:

1. Stateless cleaning. Plain DataFrame -> DataFrame functions, safe to run on the
   full dataset before splitting because nothing is learned from the data.

       clean_df = raw.pipe(parse_dates).pipe(drop_duplicate_listings).pipe(...)
       clean_df = clean(raw)  # same thing, all steps in order

2. Time-based split. Test = one calendar month, train = the X months right before it.

       train, test = time_split(clean_df, test_month="2026-04", train_months=6)

   To compare window lengths without touching the test month, loop over
   window_splits(clean_df, validation_months("2026-04", n_months=6), windows=[1, 3, 6, 12]).

3. Fitted preprocessing. Imputation values, encoders, and the scaler are learned on
   the training window only, then applied to the test month.

       pre = Preprocessor(scale=True)
       train_p = pre.fit_transform(train)
       test_p = pre.transform(test)
       X_train, y_train = train_p[pre.feature_names_], train_p[LOG_TARGET]

Everything the team is likely to argue about (column lists, plausibility ranges,
imputation strategy per column) is in the CONFIG block below.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

# =============================================================================
# CONFIG
# =============================================================================

TARGET = "ClosePrice"
LOG_TARGET = "LogClosePrice"
DATE_COL = "CloseDate"
ID_COL = "ListingKey"

DATE_COLS = [DATE_COL, "ListingContractDate", "PurchaseContractDate", "ContractStatusChangeDate"]

# Row filters for target typos (see 01_exploration 3.1). ListPrice is only used here
# to sanity-check the target. It is dropped before modeling (leakage).
PRICE_RANGE = (50_000, 50_000_000)
SALE_TO_LIST_RANGE = (0.5, 2.0)
KEEP_STATES = ["CA"]

# Values outside these ranges are set to NaN and then imputed like other missing
# values, rather than dropping the row. Bounds are inclusive.
PLAUSIBLE_RANGES = {
    "LivingArea": (300, 15_000),
    "BedroomsTotal": (1, 15),
    "BathroomsTotalInteger": (1, 15),
    "MainLevelBedrooms": (0, 15),
    "LotSizeSquareFeet": (100, 43_560 * 100),  # up to 100 acres
    "GarageSpaces": (0, 20),
    "ParkingTotal": (0, 30),
    "AssociationFeeMonthly": (0, 5_000),
    "YearBuilt": (1850, 2027),
    "Latitude": (32.4, 42.1),  # rough California bounding box
    "Longitude": (-124.5, -114.1),
}

FEE_TO_MONTHLY = {"Monthly": 1.0, "Quarterly": 1 / 3, "SemiAnnually": 1 / 6, "Annually": 1 / 12}

BOOL_COLS = ["NewConstructionYN", "AttachedGarageYN", "PoolPrivateYN", "ViewYN", "FireplaceYN"]

# Columns kept after cleaning. Everything else in the filtered file is dropped
# (leakage, PII, ids, near-empty, redundant lot-size units, free text).
KEEP_COLS = [
    ID_COL, DATE_COL, TARGET,
    # numeric
    "LivingArea", "BedroomsTotal", "BathroomsTotalInteger", "MainLevelBedrooms",
    "LotSizeSquareFeet", "PropertyAge", "GarageSpaces", "ParkingTotal",
    "AssociationFeeMonthly", "Latitude", "Longitude",
    # yes/no
    *BOOL_COLS, "BasementYN",
    # categorical
    "CountyOrParish", "LevelsPrimary", "PostalCode", "City", "MLSAreaMajor", "HighSchoolDistrict",
]

# --- Missing-value strategy, per column (applied in Preprocessor, fit on train) ---
# Median, no flag: under 0.2% missing, too rare for missingness to carry signal.
IMPUTE_MEDIAN = ["LivingArea", "BedroomsTotal", "BathroomsTotalInteger", "ParkingTotal", "PropertyAge"]
# Median + flag: enough missing that "not reported" may mean something.
IMPUTE_MEDIAN_FLAG = ["LotSizeSquareFeet", "GarageSpaces", "MainLevelBedrooms"]
# Zero + flag: a blank fee most likely means no HOA.
IMPUTE_ZERO_FLAG = ["AssociationFeeMonthly"]
# Mode + flag: in EDA the NaN group of these had very different median prices.
IMPUTE_MODE_FLAG = BOOL_COLS
# Zero, no flag: only "True" is ever recorded, so blank means "no basement".
IMPUTE_ZERO = ["BasementYN"]
# Coordinates: median of the same PostalCode, then County, then overall (+ flag).
IMPUTE_GEO = ["Latitude", "Longitude"]

# --- Encoding ---
MISSING_CATEGORY = "Missing"
ONE_HOT_COLS = ["CountyOrParish", "LevelsPrimary"]  # nominal, few levels
ONE_HOT_MIN_FREQUENCY = 200  # rarer levels are grouped into one "infrequent" column
TARGET_ENCODE_COLS = ["PostalCode", "City", "MLSAreaMajor", "HighSchoolDistrict"]  # high cardinality

# --- Scaling ---
LOG_COLS = ["LivingArea", "LotSizeSquareFeet", "AssociationFeeMonthly"]  # heavy right skew


# =============================================================================
# Stage 1: stateless cleaning
# =============================================================================


def load_filtered(path: str | Path) -> pd.DataFrame:
    """Read the scope-filtered CRMLS file (Residential / SingleFamilyResidence)."""
    df = pd.read_csv(path, low_memory=False)
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(**{c: pd.to_datetime(df[c], errors="coerce") for c in DATE_COLS if c in df})


def drop_duplicate_listings(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ListingKey, keeping the most recent close (overlapping monthly exports)."""
    return df.sort_values(DATE_COL).drop_duplicates(ID_COL, keep="last")


def drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose target is likely a typo and out-of-state rows."""
    sale_to_list = df[TARGET] / df["ListPrice"].replace(0, np.nan)
    keep = (
        df[TARGET].between(*PRICE_RANGE)
        & sale_to_list.between(*SALE_TO_LIST_RANGE)
        & df["StateOrProvince"].isin(KEEP_STATES)
    )
    return df[keep]


def standardize_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize formats and derive the cleaned versions of a few raw fields."""
    levels = df["Levels"].str.split(",").str[0]
    fee_monthly = df["AssociationFee"] * df["AssociationFeeFrequency"].map(FEE_TO_MONTHLY).fillna(1.0)
    bools = {c: df[c].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}) for c in [*BOOL_COLS, "BasementYN"]}
    return df.assign(
        PostalCode=df["PostalCode"].astype("string").str[:5],
        LevelsPrimary=levels,  # "Two,MultiSplit" -> "Two"
        AssociationFeeMonthly=fee_monthly,  # missing frequency assumed monthly
        **bools,
    )


def nullify_implausible(df: pd.DataFrame) -> pd.DataFrame:
    """Set out-of-range values to NaN so they're imputed instead of trusted."""
    out = df.copy()
    for col, (lo, hi) in PLAUSIBLE_RANGES.items():
        out[col] = out[col].where(out[col].between(lo, hi))
    out["MainLevelBedrooms"] = out["MainLevelBedrooms"].where(out["MainLevelBedrooms"] <= out["BedroomsTotal"])
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    age = (df[DATE_COL].dt.year - df["YearBuilt"]).clip(lower=0)
    return df.assign(PropertyAge=age)


def select_columns(df: pd.DataFrame, keep: Iterable[str] = KEEP_COLS) -> pd.DataFrame:
    return df[list(keep)].reset_index(drop=True)


CLEANING_STEPS: list[Callable[[pd.DataFrame], pd.DataFrame]] = [
    parse_dates,
    drop_duplicate_listings,
    drop_invalid_rows,
    standardize_fields,
    nullify_implausible,
    add_features,
    select_columns,
]


def clean(df: pd.DataFrame, steps: Iterable[Callable] = CLEANING_STEPS) -> pd.DataFrame:
    """Run every stateless cleaning step in order."""
    for step in steps:
        df = df.pipe(step)
    return df


# =============================================================================
# Stage 2: time-based split
# =============================================================================


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

    Fit a fresh Preprocessor on each `train`, so nothing from `eval` leaks in:

        for x, month, train, val in window_splits(df, validation_months("2026-04", 6), [1, 3, 6, 12]):
            pre = Preprocessor(scale=...)
            train_p, val_p = pre.fit_transform(train), pre.transform(val)
            ...  # fit and score a model
    """
    for x in windows:
        for month in eval_months:
            month = pd.Period(month, freq="M")
            train, evaluation = time_split(df, month, x)
            yield x, month, train, evaluation


# =============================================================================
# Stage 3: fitted preprocessing (learned on the training window only)
# =============================================================================


class Preprocessor:
    """Impute, encode, and (optionally) scale. Fit on train, then transform test.

    Use `fit_transform` on the training window rather than `fit(...).transform(...)`:
    the target encoder cross-fits on training rows so each row's encoding doesn't
    include its own price.
    """

    def __init__(self, scale: bool = True, random_state: int = 0):
        self.scale = scale
        self.random_state = random_state

    # --- public API -------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> Preprocessor:
        self.fit_transform(df)
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._fit_imputation(df)
        out = self._impute(df)
        out = self._log(out)

        self.onehot_ = OneHotEncoder(
            min_frequency=ONE_HOT_MIN_FREQUENCY, handle_unknown="infrequent_if_exist", sparse_output=False
        ).fit(out[ONE_HOT_COLS])
        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        self.target_encoder_ = TargetEncoder(target_type="continuous", cv=cv)
        te = self.target_encoder_.fit_transform(out[TARGET_ENCODE_COLS], np.log(df[TARGET]))
        out = self._encode(out, te)

        self.numeric_cols_ = self._continuous_cols(out)
        if self.scale:
            self.scaler_ = StandardScaler().fit(out[self.numeric_cols_])
            out = self._scale(out)
        self.feature_names_ = [c for c in out.columns if c not in (ID_COL, DATE_COL, TARGET, LOG_TARGET)]
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._log(self._impute(df))
        out = self._encode(out, self.target_encoder_.transform(out[TARGET_ENCODE_COLS]))
        if self.scale:
            out = self._scale(out)
        return out

    # --- steps --------------------------------------------------------------

    def _fit_imputation(self, df: pd.DataFrame) -> None:
        self.medians_ = df[IMPUTE_MEDIAN + IMPUTE_MEDIAN_FLAG].median()
        self.modes_ = df[IMPUTE_MODE_FLAG].mode().iloc[0]
        self.geo_global_ = df[IMPUTE_GEO].median()
        self.geo_by_zip_ = df.groupby("PostalCode")[IMPUTE_GEO].median()
        self.geo_by_county_ = df.groupby("CountyOrParish")[IMPUTE_GEO].median()

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        flagged = IMPUTE_MEDIAN_FLAG + IMPUTE_ZERO_FLAG + IMPUTE_MODE_FLAG
        flags = {f"{c}_was_missing": out[c].isna().astype(float) for c in flagged}
        flags["Coords_was_missing"] = out[IMPUTE_GEO].isna().any(axis=1).astype(float)

        for col in IMPUTE_MEDIAN + IMPUTE_MEDIAN_FLAG:
            out[col] = out[col].fillna(self.medians_[col])
        for col in IMPUTE_ZERO_FLAG + IMPUTE_ZERO:
            out[col] = out[col].fillna(0.0)
        for col in IMPUTE_MODE_FLAG:
            out[col] = out[col].fillna(self.modes_[col])
        for col in IMPUTE_GEO:
            out[col] = (
                out[col]
                .fillna(out["PostalCode"].map(self.geo_by_zip_[col]))
                .fillna(out["CountyOrParish"].map(self.geo_by_county_[col]))
                .fillna(self.geo_global_[col])
            )
        for col in ONE_HOT_COLS + TARGET_ENCODE_COLS:
            out[col] = out[col].astype("string").fillna(MISSING_CATEGORY)
        return out.assign(**flags)

    @staticmethod
    def _log(df: pd.DataFrame) -> pd.DataFrame:
        logged = {c: np.log1p(df[c]) for c in LOG_COLS}
        if TARGET in df:
            logged[LOG_TARGET] = np.log(df[TARGET])
        return df.assign(**logged)

    def _encode(self, df: pd.DataFrame, te: np.ndarray) -> pd.DataFrame:
        onehot = pd.DataFrame(
            self.onehot_.transform(df[ONE_HOT_COLS]),
            columns=self.onehot_.get_feature_names_out(),
            index=df.index,
        )
        te = pd.DataFrame(te, columns=[f"{c}_te" for c in TARGET_ENCODE_COLS], index=df.index)
        return pd.concat([df.drop(columns=ONE_HOT_COLS + TARGET_ENCODE_COLS), onehot, te], axis=1)

    def _continuous_cols(self, df: pd.DataFrame) -> list[str]:
        """Columns to scale: everything numeric that isn't a 0/1 indicator."""
        skip = {ID_COL, TARGET, LOG_TARGET, *BOOL_COLS, *IMPUTE_ZERO}
        return [
            c for c in df.select_dtypes("number").columns
            if c not in skip and not c.endswith("_was_missing") and not set(df[c].unique()) <= {0.0, 1.0}
        ]

    def _scale(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.numeric_cols_] = self.scaler_.transform(df[self.numeric_cols_])
        return out
