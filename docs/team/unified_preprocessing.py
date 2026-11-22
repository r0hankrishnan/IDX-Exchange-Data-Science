"""Shared cleaning and fold-fitted preprocessing for combined sale data.

This handles source normalization and generic model preprocessing.
Derived features are in unified_feature_engineering.py.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

TARGET = "ClosePrice"
DATE = "CloseDate"
ID = "ListingKey"
SQFT_PER_ACRE = 43_560

BOOL_COLUMNS = [
    "AttachedGarageYN", "FireplaceYN", "NewConstructionYN", "PoolPrivateYN", "ViewYN"
]
COORD_COLUMNS = ["Latitude", "Longitude"]
CA_BOUNDS = {"Latitude": (32.4, 42.1), "Longitude": (-124.5, -114.1)}


def clean_source(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply fixed, row-local cleaning and return ``(cleaned, removal_log)``.

    Fields not required by the selected model are preserved for later feature
    engineering. ClosePrice must be positive; date-order filtering only applies
    when both dates are available. Do not apply target quantile trimming here.
    """
    out = df.drop(columns=[c for c in df if c.startswith("Unnamed")], errors="ignore").copy()
    log: list[dict] = []

    def keep(mask: pd.Series, label: str) -> None:
        nonlocal out
        before = len(out)
        out = out.loc[mask.fillna(False)].copy()
        removed = before - len(out)
        log.append({"step": label, "before": before, "removed": removed,
                    "removed_pct": round(100 * removed / before, 4) if before else 0.0,
                    "remaining": len(out)})

    # Keep this scope filter conditional so already-filtered monthly files work.
    scope = pd.Series(True, index=out.index)
    if "StateOrProvince" in out:
        scope &= out["StateOrProvince"].eq("CA")
    if "PropertyType" in out:
        scope &= out["PropertyType"].eq("Residential")
    if "PropertySubType" in out:
        scope &= out["PropertySubType"].eq("SingleFamilyResidence")
    keep(scope, "California single-family scope")

    for col in (DATE, "ListingContractDate", "PurchaseContractDate", "ContractStatusChangeDate"):
        if col in out:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    if TARGET not in out:
        raise KeyError(f"Missing target column {TARGET!r}")
    out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce")
    keep(out[TARGET].notna() & out[TARGET].gt(0), "Missing or nonpositive ClosePrice")

    if DATE in out and "ListingContractDate" in out:
        invalid_order = out[DATE].notna() & out["ListingContractDate"].notna() & (
            out[DATE] < out["ListingContractDate"]
        )
        keep(~invalid_order, "CloseDate before ListingContractDate")

    # Monthly exports can overlap. ListingKey is preferred; otherwise use a
    # normalized address and close date. Rows with unusable addresses survive.
    if ID in out and out[ID].notna().any():
        out = out.sort_values(DATE, na_position="first") if DATE in out else out
        duplicate = out[ID].notna() & out[ID].duplicated(keep="last")
    elif "UnparsedAddress" in out and DATE in out:
        addr = out["UnparsedAddress"].astype("string").str.lower().str.strip()
        addr = addr.str.replace(r"\s+", " ", regex=True)
        valid = addr.notna() & addr.ne("") & addr.str.contains(r"\d", na=False)
        duplicate = valid & out.assign(_addr=addr).duplicated(["_addr", DATE], keep="first")
    else:
        duplicate = pd.Series(False, index=out.index)
    keep(~duplicate, "Duplicate listing/sale")

    # Normalize numeric and categorical source fields while preserving originals.
    numeric_candidates = [
        "LivingArea", "YearBuilt", "LotSizeSquareFeet", "LotSizeAcres",
        "BedroomsTotal", "BathroomsTotalInteger", "GarageSpaces", "ParkingTotal",
        "Latitude", "Longitude", "ListPrice", "AssociationFee", TARGET,
    ]
    for col in numeric_candidates:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Implausible feature values become missing and are handled by the fitted
    # imputer; they do not remove otherwise usable sale records.
    valid_ranges = {
        "LivingArea": (300, 15_000), "YearBuilt": (1850, 2027),
        "BedroomsTotal": (1, 20), "BathroomsTotalInteger": (1, 20),
        "LotSizeSquareFeet": (100, 43_560 * 100), "GarageSpaces": (0, 20),
        "ParkingTotal": (0, 30),
    }
    for col, (low, high) in valid_ranges.items():
        if col in out:
            out[col] = out[col].where(out[col].between(low, high))

    if "PostalCode" in out:
        out["PostalCode"] = (
            out["PostalCode"].astype("string").str.strip()
            .str.extract(r"^(\d{5})", expand=False)
        )
    for col in ("City", "CountyOrParish", "SchoolDistrictUnified",
                "ElementarySchoolDistrict", "HighSchoolDistrict", "MLSAreaMajor",
                "SubdivisionName"):
        if col in out:
            text = out[col].astype("string").str.strip()
            out[col] = text.mask(text.eq(""))

    # Invalid coordinates are treated as one missing pair.
    if set(COORD_COLUMNS).issubset(out.columns):
        bad = pd.Series(False, index=out.index)
        for col, bounds in CA_BOUNDS.items():
            bad |= out[col].notna() & ~out[col].between(*bounds)
        out.loc[bad, COORD_COLUMNS] = np.nan

    # Repair lot square footage from acreage, retaining valid square footage.
    if {"LotSizeSquareFeet", "LotSizeAcres"}.issubset(out.columns):
        sqft = out["LotSizeSquareFeet"].where(out["LotSizeSquareFeet"] > 0)
        acres_sqft = out["LotSizeAcres"].where(out["LotSizeAcres"] > 0) * SQFT_PER_ACRE
        out["LotSizeSquareFeet"] = sqft.fillna(acres_sqft)

    # Standardize common yes/no representations to 1, 0, or NaN.
    bool_map = {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0,
                "Yes": 1.0, "No": 0.0, "Y": 1.0, "N": 0.0, 1: 1.0, 0: 0.0}
    for col in BOOL_COLUMNS:
        if col in out:
            out[col] = out[col].map(bool_map)

    if DATE in out and "YearBuilt" in out:
        age = out[DATE].dt.year - out["YearBuilt"]
        out["PropertyAge"] = age.where(age >= 0)

    return out.reset_index(drop=True), pd.DataFrame(log)


def add_missing_flags(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Add explicit missingness indicators before model imputation."""
    out = df.copy()
    for col in columns:
        if col in out:
            out[f"{col}_missing"] = out[col].isna().astype("int8")
    return out


def split_by_month(
    df: pd.DataFrame, validation_month: str, train_months: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the immediately preceding calendar months and one validation month."""
    if train_months < 1:
        raise ValueError("train_months must be positive")
    month = pd.to_datetime(df[DATE], errors="coerce").dt.to_period("M")
    val_month = pd.Period(validation_month, freq="M")
    train_mask = (month >= val_month - train_months) & (month < val_month)
    return df.loc[train_mask].copy(), df.loc[month == val_month].copy()


def build_tree_preprocessor(
    numeric_columns: Iterable[str],
    categorical_columns: Iterable[str],
    mode_columns: Iterable[str] = (),
    quantile_columns: Iterable[str] = (),
    numeric_scaling: str = "none",
    categorical_encoding: str = "target",
) -> ColumnTransformer:
    """Create train-fitted imputers and cross-fitted target encoding for trees.

    Numeric scaling is either ``none`` or train-fitted quantile-to-normal for
    the selected skewed columns. Standard and min-max scaling are omitted.
    Target encoding is the default categorical method; ordinal encoding remains
    an optional sensitivity check. TargetEncoder receives the model target,
    including log target if wrapped in TransformedTargetRegressor.
    """
    numeric_columns = list(numeric_columns)
    categorical_columns = list(categorical_columns)
    mode_columns = [c for c in mode_columns if c in numeric_columns]
    quantile_columns = [c for c in quantile_columns if c in numeric_columns and c not in mode_columns]
    continuous_columns = [c for c in numeric_columns if c not in mode_columns]
    raw_continuous = [c for c in continuous_columns if c not in quantile_columns]
    transformers = []
    if raw_continuous:
        transformers.append(("numeric", SimpleImputer(strategy="median", add_indicator=True), raw_continuous))
    if quantile_columns:
        from sklearn.preprocessing import QuantileTransformer
        quantile = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("quantile_gaussian", QuantileTransformer(
                n_quantiles=1000, output_distribution="normal", subsample=10_000,
                random_state=42,
            )),
        ])
        if numeric_scaling != "quantile":
            raise ValueError("quantile_columns require numeric_scaling='quantile'")
        transformers.append(("quantile", quantile, quantile_columns))
    elif numeric_scaling not in {"none", "quantile"}:
        raise ValueError("numeric_scaling must be 'none' or 'quantile'")
    if mode_columns:
        transformers.append(("mode", SimpleImputer(strategy="most_frequent"), mode_columns))
    if categorical_encoding == "target":
        from sklearn.preprocessing import TargetEncoder
        encoder = TargetEncoder(target_type="continuous", cv=5, shuffle=True, random_state=42)
    elif categorical_encoding == "ordinal":
        from sklearn.preprocessing import OrdinalEncoder
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    else:
        raise ValueError("categorical_encoding must be 'target' or 'ordinal'")
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encode", encoder),
    ])
    if categorical_columns:
        transformers.append(("categorical", cat, categorical_columns))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)

