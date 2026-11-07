"""Features added on top of Dataset A, built once on the whole cleaned dataset:

    df = build_table()   # Dataset A + school districts + listing date + comparable sales

Each column only uses information available when the home is listed, so computing it
before the split doesn't leak anything into training:
- School districts: a spatial join on the home's coordinates (a fixed rule).
- Listing date: ListingContractDate, which Dataset A drops, read back from the filtered file.
- Comparable sales: prices of nearby sales that closed before the home's listing month.

California has three kinds of district: Unified (K-12), Elementary and High. Elementary
and High districts overlap, so each home is assigned the district that serves its
elementary grades and the one that serves its high school grades. In unified areas
these are the same district.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from .dataset_a import load_clean

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATASET_A_PATH = DATA_DIR / "processed" / "CRMLSSoldDatasetA.csv"
FILTERED_PATH = DATA_DIR / "intermediate" / "CRMLSSoldFiltered.csv"
DISTRICTS_PATH = DATA_DIR / "external" / "school_districts_2425" / "DistrictAreas2425.shp"

DISTRICT_COLS = ["ElementaryDistrict", "HighDistrict"]
# Share of socioeconomically disadvantaged students in the elementary district.
DISTRICT_NUMERIC_COLS = ["DistrictSEDPct"]

# Prices on a log scale: trees don't care, and the linear model is fit on log price.
COMP_COLS = ["LogCompPricePerSqft", "LogCompImpliedPrice", "CompDistanceKm"]
COMP_K = 10
COMP_LOOKBACK_MONTHS = 12
# Homes with no listing date are assumed to list this many months before close (the median).
DEFAULT_MONTHS_TO_CLOSE = 2
EARTH_RADIUS_KM = 6371

LISTING_DATE = "ListingContractDate"
# Listing dates this long before close are most likely stale (relisted homes).
MAX_MONTHS_TO_CLOSE = 24


def load_school_districts(path: str | Path) -> gpd.GeoDataFrame:
    """CDE School District Areas shapefile, reduced to what the join needs, in lat/lon.

    District names repeat across counties, so each district is labeled "Name (County)".
    """
    districts = gpd.read_file(path)
    return gpd.GeoDataFrame({
        "District": districts["DistrictNa"] + " (" + districts["CountyName"] + ")",
        "DistrictTy": districts["DistrictTy"],
        "SEDpct": districts["SEDpct"],
    }, geometry=districts.geometry).to_crs("EPSG:4326")


def add_school_districts(df: pd.DataFrame, districts: gpd.GeoDataFrame) -> pd.DataFrame:
    """Spatially join each home's coordinates to its elementary and high school districts.

    Homes with missing coordinates, or outside every district, get NaN (encoded as Unknown).
    """
    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]), index=df.index, crs="EPSG:4326"
    )
    matches = gpd.sjoin(points[~points.geometry.is_empty], districts, predicate="within")

    def serving(own_type: str) -> pd.DataFrame:
        """Per home, the district of `own_type` if there is one, otherwise the unified district."""
        candidates = matches[matches["DistrictTy"].isin([own_type, "Unified"])]
        candidates = candidates.assign(fallback=candidates["DistrictTy"] != own_type)
        first = candidates.sort_values("fallback", kind="stable")
        return first[~first.index.duplicated()].reindex(df.index)

    elementary, high = serving("Elementary"), serving("High")
    return df.assign(
        ElementaryDistrict=elementary["District"].astype("string"),
        HighDistrict=high["District"].astype("string"),
        DistrictSEDPct=elementary["SEDpct"].astype(float),
    )


def load_listing_dates(path: str | Path) -> pd.DataFrame:
    """ListingKey -> ListingContractDate from the filtered CRMLS file (dropped from Dataset A)."""
    dates = pd.read_csv(path, usecols=["ListingKey", LISTING_DATE], parse_dates=[LISTING_DATE])
    return dates.drop_duplicates("ListingKey")


def add_listing_date(df: pd.DataFrame, listing_dates: pd.DataFrame) -> pd.DataFrame:
    """Add ListingContractDate and MonthsToClose (calendar months from listing to close).

    Both are NaN when the listing date is unknown, after the close, or implausibly old.
    """
    out = df.merge(listing_dates, on="ListingKey", how="left", validate="many_to_one")
    months_to_close = (out["CloseDate"].dt.year - out[LISTING_DATE].dt.year) * 12 + (
        out["CloseDate"].dt.month - out[LISTING_DATE].dt.month)
    plausible = months_to_close.between(0, MAX_MONTHS_TO_CLOSE)
    out[LISTING_DATE] = out[LISTING_DATE].where(plausible)
    out["MonthsToClose"] = months_to_close.where(plausible)
    return out


def add_comparable_sales(df: pd.DataFrame, k: int = COMP_K,
                         lookback_months: int = COMP_LOOKBACK_MONTHS) -> pd.DataFrame:
    """Price of the k nearest sales that closed in the `lookback_months` before the home's listing month.

    LogCompPricePerSqft: log of the median $/sq ft of those sales. LogCompImpliedPrice: log of
    that times the home's living area. CompDistanceKm: their mean distance, i.e. how local
    the comparison is.
    The home's own sale is never a comparable, because it closes after it lists.
    NaN when the home has no coordinates or fewer than k earlier sales exist.
    """
    close_month = df["CloseDate"].dt.to_period("M")
    listing_month = df[LISTING_DATE].dt.to_period("M").fillna(close_month - DEFAULT_MONTHS_TO_CLOSE)
    has_coords = df[["Latitude", "Longitude"]].notna().all(axis=1)
    price_per_sqft = df["ClosePrice"] / df["LivingArea"].where(df["LivingArea"] > 0)
    coords = np.radians(df[["Latitude", "Longitude"]].to_numpy())

    comp_ppsf = np.full(len(df), np.nan)
    comp_distance = np.full(len(df), np.nan)
    for month in listing_month.dropna().unique():
        earlier = (close_month < month) & (close_month >= month - lookback_months)
        pool = (earlier & has_coords & price_per_sqft.notna()).to_numpy()
        homes = ((listing_month == month) & has_coords).to_numpy()
        if pool.sum() < k or not homes.any():
            continue
        distance, nearest = BallTree(coords[pool], metric="haversine").query(coords[homes], k=k)
        comp_ppsf[homes] = np.median(price_per_sqft.to_numpy()[pool][nearest], axis=1)
        comp_distance[homes] = distance.mean(axis=1) * EARTH_RADIUS_KM

    return df.assign(LogCompPricePerSqft=np.log(comp_ppsf),
                     LogCompImpliedPrice=np.log(comp_ppsf * df["LivingArea"].where(df["LivingArea"] > 0)),
                     CompDistanceKm=comp_distance)


def build_table() -> pd.DataFrame:
    """Dataset A plus school districts, listing date and comparable sales."""
    df = load_clean(DATASET_A_PATH)
    df = add_school_districts(df, load_school_districts(DISTRICTS_PATH))
    df = add_listing_date(df, load_listing_dates(FILTERED_PATH))
    return add_comparable_sales(df)
