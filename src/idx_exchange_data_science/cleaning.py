"""Dataset A cleaning: the team's fixed rules (see preprocessing-procedures.md).

These rules look at one row at a time and learn nothing from the data, so they run once on
the whole dataset, before any split:

    df = clean(load_raw(FILTERED_PATH))   # notebooks/02b_preprocessing_dataset_a.ipynb
    df.to_csv(DATASET_A_PATH, index=False)
    df = load_clean(DATASET_A_PATH)

Anything learned from data (the ClosePrice trim, imputation, encoding) is fit on the training
window only, in splits.py and preprocessing.py.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "ClosePrice"
DATE_COL = "CloseDate"
ID_COL = "ListingKey"

MAX_MISSING_SHARE = 0.8

SQFT_PER_ACRE = 43_560

# Values outside these inclusive ranges become NaN and are imputed later (the team's unified clean_source).
VALID_RANGES = {
    "LivingArea": (300, 15_000),
    "YearBuilt": (1850, 2027),
    "BedroomsTotal": (1, 20),
    "BathroomsTotalInteger": (1, 20),
    "LotSizeSquareFeet": (100, 100 * SQFT_PER_ACRE),
    "GarageSpaces": (0, 20),
    "ParkingTotal": (0, 30),
}
# A coordinate outside these bounds makes the whole pair missing.
CA_BOUNDS = {"Latitude": (32.4, 42.1), "Longitude": (-124.5, -114.1)}
BOOLEAN_VALUES = {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0, "Yes": 1.0, "No": 0.0, "Y": 1.0, "N": 0.0}

MEDIAN_COLS = ["LivingArea", "PropertyAge", "LotSizeSquareFeet"]
MODE_COLS = ["BedroomsTotal", "BathroomsTotalInteger", "GarageSpaces", "ParkingTotal"]
BINARY_COLS = ["AttachedGarageYN", "FireplaceYN", "NewConstructionYN", "PoolPrivateYN", "ViewYN"]
GEO_COLS = ["Latitude", "Longitude"]
CATEGORICAL_COLS = ["City", "CountyOrParish", "PostalCode"]
LOG_COLS = ["LivingArea", "LotSizeSquareFeet"]
UNKNOWN = "Unknown"

FEATURES = MEDIAN_COLS + MODE_COLS + BINARY_COLS + GEO_COLS + CATEGORICAL_COLS


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


def drop_duplicate_listings(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ListingKey: the record with the latest CloseDate.

    Overlapping monthly files can hold the same listing twice with different close dates (a corrected
    record). Without this step, an earlier copy can be in training and the later copy in validation.
    """
    latest = df.sort_values("CloseDate", kind="stable", na_position="first").duplicated(ID_COL, keep="last")
    return df[~latest.reindex(df.index)]


def drop_duplicate_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Same address and close date = same sale; keep the first. Rows without a valid address are dropped."""
    address = normalize_address(df["UnparsedAddress"])
    df = df[address.notna()]
    return df[~df.assign(address=address).duplicated(["address", "CloseDate"])]


def drop_nonpositive_living_area(df: pd.DataFrame) -> pd.DataFrame:
    return df[~(df["LivingArea"] <= 0)]


def fix_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize formats, null out implausible values, and derive PropertyAge (as in the team's clean_source)."""
    out = df.copy()
    for col, (low, high) in VALID_RANGES.items():
        out[col] = out[col].where(out[col].between(low, high))

    out["PostalCode"] = out["PostalCode"].astype("string").str.strip().str.extract(r"^(\d{5})", expand=False)
    for col in ("City", "CountyOrParish"):
        text = out[col].astype("string").str.strip()
        out[col] = text.mask(text == "")

    outside = pd.Series(False, index=out.index)
    for col, (low, high) in CA_BOUNDS.items():
        outside |= out[col].notna() & ~out[col].between(low, high)
    out.loc[outside, GEO_COLS] = np.nan

    lot_sqft = out["LotSizeSquareFeet"].where(out["LotSizeSquareFeet"] > 0)
    lot_from_acres = out["LotSizeAcres"].where(out["LotSizeAcres"] > 0) * SQFT_PER_ACRE
    out["LotSizeSquareFeet"] = lot_sqft.fillna(lot_from_acres)

    for col in BINARY_COLS:
        out[col] = out[col].map(BOOLEAN_VALUES)
    age = out["CloseDate"].dt.year - out["YearBuilt"]
    out["PropertyAge"] = age.where(age >= 0)
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
    drop_duplicate_listings,
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
