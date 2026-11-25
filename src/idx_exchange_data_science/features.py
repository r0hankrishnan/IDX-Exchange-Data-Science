"""Features added on top of Dataset A, built once on the whole cleaned dataset:

    df = build_table()   # Dataset A + every feature group below, ~15 s

Each column only uses information known when the home is listed, and each is a fixed rule
(nothing is learned from prices across rows), so computing it before the split doesn't leak
anything into training. Imputation and encoding happen later, in the model's preprocessor.

Feature groups (pass the column lists to `make_preprocessor(numeric=..., categorical=...)`):
- School districts (DISTRICT_COLS, DISTRICT_NUMERIC_COLS): a spatial join on the home's
  coordinates. California has Unified (K-12), Elementary and High districts; Elementary and
  High overlap, so each home gets the district serving its elementary grades and the one
  serving its high school grades (the same district in unified areas).
- MLS district (MLS_DISTRICT_COLS): the high school district as typed into the listing. The
  MLS elementary district column is empty in this data.
- Room ratios (RATIO_COLS): the team's default ratio group (bedrooms per bathroom, square feet per
  bedroom, log of lot size over living area), as in the unified feature engineering script.
- HOA (HOA_COLS): monthly HOA fee, whether there is one, and whether the fee is missing.
- Flooring (FLOORING_COLS): one 0/1 column per common flooring type, plus "missing".
- Comparable sales (COMP_COLS): prices of the 10 nearest sales that closed before the listing month.
  COMP_PRICE_COLS: their median sale price, without the size adjustment.
- Similar comparable sales (SIMILAR_COLS): the same, but the 10 most similar homes among the 50 nearest
  (living area, year built, bedrooms, bathrooms, lot size and distance; see SIMILARITY_UNITS).
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from .cleaning import load_clean
from .paths import DATASET_A_PATH, DISTRICTS_PATH, FILTERED_PATH

DISTRICT_COLS = ["ElementaryDistrict", "HighDistrict"]
# Share of socioeconomically disadvantaged students in the elementary district.
DISTRICT_NUMERIC_COLS = ["DistrictSEDPct"]
MLS_DISTRICT_COLS = ["MLSHighSchoolDistrict"]

RATIO_COLS = ["BedBathRatio", "SqftPerBedroom", "LogLotToLivingArea"]

HOA_COLS = ["HOAFeeMonthly", "HasHOA", "HOAFeeMissing"]
# Fees are quoted per period; a blank frequency is taken as monthly.
FEES_PER_MONTH = {"Monthly": 1.0, "Quarterly": 1 / 3, "SemiAnnually": 1 / 6, "Annually": 1 / 12}

FLOORING_TYPES = ["Wood", "Tile", "Carpet", "Laminate", "Vinyl", "SeeRemarks"]
FLOORING_COLS = [f"Flooring{t}" for t in FLOORING_TYPES] + ["FlooringMissing"]

# Raw listing columns that Dataset A drops, read back from the filtered file by ListingKey.
LISTING_DETAIL_COLS = ["AssociationFee", "AssociationFeeFrequency", "Flooring", "HighSchoolDistrict"]

# Prices on a log scale: trees don't care, and the linear model is fit on log price.
COMP_COLS = ["LogCompPricePerSqft", "LogCompImpliedPrice", "CompDistanceKm"]
COMP_PRICE_COLS = ["LogCompMedianPrice"]  # median sale price of the same 10 sales (no size adjustment)
COMP_K = 10
COMP_LOOKBACK_MONTHS = 12

# Similar comparable sales: from the SIMILAR_POOL nearest earlier sales, the COMP_K most similar homes.
SIMILAR_COLS = ["LogSimilarPricePerSqft", "LogSimilarImpliedPrice", "SimilarDistanceKm", "SimilarMatchScore"]
SIMILAR_POOL = 50
# One unit of difference for each attribute, from appraisal rules of thumb (fixed in advance, not fit to the data).
# A candidate's score is the sum of its differences in these units; lower is more similar.
SIMILARITY_UNITS = {
    "living area": np.log(1.25),  # 25% larger or smaller
    "year built": 20,             # about one construction era
    "bedrooms": 1,
    "bathrooms": 1,
    "lot size": np.log(2),        # twice or half the lot
    "distance (km)": 1,
}
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


def load_listing_details(path: str | Path = FILTERED_PATH) -> pd.DataFrame:
    """ListingKey -> ListingContractDate and the LISTING_DETAIL_COLS, from the filtered CRMLS file."""
    details = pd.read_csv(path, usecols=["ListingKey", LISTING_DATE, *LISTING_DETAIL_COLS],
                          parse_dates=[LISTING_DATE], low_memory=False)
    return details.drop_duplicates("ListingKey")


def add_listing_date(df: pd.DataFrame, listing_dates: pd.DataFrame) -> pd.DataFrame:
    """Merge in `listing_dates` by ListingKey and add MonthsToClose (calendar months from listing to close).

    ListingContractDate and MonthsToClose are NaN when the listing date is unknown, after the
    close, or implausibly old. Other columns of `listing_dates` come along unchanged.
    """
    out = df.merge(listing_dates, on="ListingKey", how="left", validate="many_to_one")
    months_to_close = (out["CloseDate"].dt.year - out[LISTING_DATE].dt.year) * 12 + (
        out["CloseDate"].dt.month - out[LISTING_DATE].dt.month)
    plausible = months_to_close.between(0, MAX_MONTHS_TO_CLOSE)
    out[LISTING_DATE] = out[LISTING_DATE].where(plausible)
    out["MonthsToClose"] = months_to_close.where(plausible)
    return out


def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """RATIO_COLS, as in the team's unified script. NaN when a denominator is missing or 0 or less."""
    def ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        return numerator / denominator.where(denominator > 0)

    lot_to_living = ratio(df["LotSizeSquareFeet"], df["LivingArea"])
    return df.assign(BedBathRatio=ratio(df["BedroomsTotal"], df["BathroomsTotalInteger"]),
                     SqftPerBedroom=ratio(df["LivingArea"], df["BedroomsTotal"]),
                     LogLotToLivingArea=np.log(lot_to_living.where(lot_to_living > 0)))


def add_hoa(df: pd.DataFrame) -> pd.DataFrame:
    """HOA_COLS from AssociationFee and AssociationFeeFrequency (dropped afterwards).

    HOAFeeMonthly stays NaN when the fee is missing (imputed later); HasHOA is 0 then.
    """
    per_month = df["AssociationFeeFrequency"].map(FEES_PER_MONTH).fillna(1.0)
    fee = pd.to_numeric(df["AssociationFee"], errors="coerce") * per_month
    return df.drop(columns=["AssociationFee", "AssociationFeeFrequency"]).assign(
        HOAFeeMonthly=fee, HasHOA=fee.gt(0).astype(float), HOAFeeMissing=fee.isna().astype(float))


def add_flooring(df: pd.DataFrame) -> pd.DataFrame:
    """FLOORING_COLS from the comma-separated Flooring field (dropped afterwards). All 0 when it is missing."""
    types = df["Flooring"].fillna("").str.split(",").map(lambda parts: {p.strip() for p in parts})
    flags = {f"Flooring{t}": types.map(lambda found, t=t: t in found).astype(float) for t in FLOORING_TYPES}
    return df.drop(columns="Flooring").assign(**flags, FlooringMissing=df["Flooring"].isna().astype(float))


def add_mls_district(df: pd.DataFrame) -> pd.DataFrame:
    """MLSHighSchoolDistrict: the HighSchoolDistrict text from the listing, blanks as missing."""
    district = df["HighSchoolDistrict"].astype("string").str.strip()
    return df.drop(columns="HighSchoolDistrict").assign(MLSHighSchoolDistrict=district.mask(district == ""))


def listing_months(df: pd.DataFrame) -> pd.Series:
    """Each home's listing month. Homes with no listing date are assumed to list 2 months before close."""
    close_month = df["CloseDate"].dt.to_period("M")
    return df[LISTING_DATE].dt.to_period("M").fillna(close_month - DEFAULT_MONTHS_TO_CLOSE)


def price_per_sqft(df: pd.DataFrame) -> pd.Series:
    return df["ClosePrice"] / df["LivingArea"].where(df["LivingArea"] > 0)


def has_coordinates(df: pd.DataFrame) -> pd.Series:
    return df[["Latitude", "Longitude"]].notna().all(axis=1)


def comparable_pool(df: pd.DataFrame, month: pd.Period, lookback_months: int = COMP_LOOKBACK_MONTHS) -> pd.Series:
    """The sales a home listed in `month` can use as comparables: closed in the `lookback_months`
    before it, with coordinates and a price per square foot."""
    close_month = df["CloseDate"].dt.to_period("M")
    earlier = (close_month < month) & (close_month >= month - lookback_months)
    return earlier & has_coordinates(df) & price_per_sqft(df).notna()


def add_comparable_sales(df: pd.DataFrame, k: int = COMP_K,
                         lookback_months: int = COMP_LOOKBACK_MONTHS) -> pd.DataFrame:
    """Price of the k nearest sales that closed in the `lookback_months` before the home's listing month.

    LogCompPricePerSqft: log of the median $/sq ft of those sales. LogCompImpliedPrice: log of
    that times the home's living area. CompDistanceKm: their mean distance, i.e. how local
    the comparison is.
    The home's own sale is never a comparable, because it closes after it lists.
    NaN when the home has no coordinates or fewer than k earlier sales exist.
    """
    listing_month = listing_months(df)
    has_coords = has_coordinates(df)
    ppsf = price_per_sqft(df).to_numpy()
    coords = np.radians(df[["Latitude", "Longitude"]].to_numpy())

    price = df["ClosePrice"].to_numpy()

    comp_ppsf = np.full(len(df), np.nan)
    comp_price = np.full(len(df), np.nan)
    comp_distance = np.full(len(df), np.nan)
    for month in listing_month.dropna().unique():
        pool = comparable_pool(df, month, lookback_months).to_numpy()
        homes = ((listing_month == month) & has_coords).to_numpy()
        if pool.sum() < k or not homes.any():
            continue
        distance, nearest = BallTree(coords[pool], metric="haversine").query(coords[homes], k=k)
        comp_ppsf[homes] = np.median(ppsf[pool][nearest], axis=1)
        comp_price[homes] = np.median(price[pool][nearest], axis=1)
        comp_distance[homes] = distance.mean(axis=1) * EARTH_RADIUS_KM

    return df.assign(LogCompPricePerSqft=np.log(comp_ppsf),
                     LogCompImpliedPrice=np.log(comp_ppsf * df["LivingArea"].where(df["LivingArea"] > 0)),
                     CompDistanceKm=comp_distance, LogCompMedianPrice=np.log(comp_price))


def similarity_attributes(df: pd.DataFrame) -> np.ndarray:
    """Each home's attributes in SIMILARITY_UNITS, so a difference of 1 is one step (distance is added separately)."""
    units = SIMILARITY_UNITS
    year_built = df["CloseDate"].dt.year - df["PropertyAge"]
    return np.column_stack([
        np.log(df["LivingArea"].where(df["LivingArea"] > 0)) / units["living area"],
        year_built / units["year built"],
        df["BedroomsTotal"] / units["bedrooms"],
        df["BathroomsTotalInteger"] / units["bathrooms"],
        np.log(df["LotSizeSquareFeet"].where(df["LotSizeSquareFeet"] > 0)) / units["lot size"],
    ])


def add_similar_sales(df: pd.DataFrame, k: int = COMP_K, pool_size: int = SIMILAR_POOL,
                      lookback_months: int = COMP_LOOKBACK_MONTHS) -> pd.DataFrame:
    """Like add_comparable_sales, but the k sales are the most similar of the `pool_size` nearest earlier sales.

    Similarity: the sum of the differences in SIMILARITY_UNITS (living area, year built, bedrooms, bathrooms,
    lot size and distance). An attribute that is missing for either home adds nothing.
    Adds SIMILAR_COLS: log median $/sq ft of the k sales, log of that times the home's living area, their mean
    distance, and their mean score (how good the matches are). The same timing rule as add_comparable_sales
    applies, so only sales that closed before the listing month are used.
    """
    listing_month = listing_months(df)
    has_coords = has_coordinates(df)
    ppsf = price_per_sqft(df).to_numpy()
    coords = np.radians(df[["Latitude", "Longitude"]].to_numpy())
    attributes = similarity_attributes(df)

    similar_ppsf, similar_distance, match_score = (np.full(len(df), np.nan) for _ in range(3))
    for month in listing_month.dropna().unique():
        pool = comparable_pool(df, month, lookback_months).to_numpy()
        homes = ((listing_month == month) & has_coords).to_numpy()
        if pool.sum() < k or not homes.any():
            continue
        tree = BallTree(coords[pool], metric="haversine")
        distance, nearest = tree.query(coords[homes], k=min(pool_size, pool.sum()))
        km = distance * EARTH_RADIUS_KM
        differences = np.abs(attributes[pool][nearest] - attributes[homes][:, None, :])
        score = np.nansum(differences, axis=2) + km / SIMILARITY_UNITS["distance (km)"]
        best = np.argsort(score, axis=1, kind="stable")[:, :k]
        chosen = np.take_along_axis(nearest, best, axis=1)
        similar_ppsf[homes] = np.median(ppsf[pool][chosen], axis=1)
        similar_distance[homes] = np.take_along_axis(km, best, axis=1).mean(axis=1)
        match_score[homes] = np.take_along_axis(score, best, axis=1).mean(axis=1)

    return df.assign(LogSimilarPricePerSqft=np.log(similar_ppsf),
                     LogSimilarImpliedPrice=np.log(similar_ppsf * df["LivingArea"].where(df["LivingArea"] > 0)),
                     SimilarDistanceKm=similar_distance, SimilarMatchScore=match_score)


def find_comparables(sales: pd.DataFrame, home: pd.DataFrame, k: int = COMP_K,
                     lookback_months: int = COMP_LOOKBACK_MONTHS) -> pd.DataFrame:
    """The k `sales` behind the comparable-sales features of `home` (a one-row frame), nearest first.

    `home` does not have to be in `sales`; it needs Latitude, Longitude, CloseDate and ListingContractDate.
    Adds DistanceKm and PricePerSqft. Empty when the home has no coordinates or too few earlier sales.
    """
    pool = sales[comparable_pool(sales, listing_months(home).iloc[0], lookback_months)]
    if not has_coordinates(home).iloc[0] or len(pool) < k:
        return pool.iloc[:0].assign(DistanceKm=[], PricePerSqft=[])
    tree = BallTree(np.radians(pool[["Latitude", "Longitude"]].to_numpy()), metric="haversine")
    distance, nearest = tree.query(np.radians(home[["Latitude", "Longitude"]].to_numpy()), k=k)
    comps = pool.iloc[nearest[0]]
    return comps.assign(DistanceKm=distance[0] * EARTH_RADIUS_KM, PricePerSqft=price_per_sqft(comps))


def comparable_features(comps: pd.DataFrame, living_area: float) -> dict[str, float]:
    """COMP_COLS and COMP_PRICE_COLS for one home from its comparables (the output of find_comparables)."""
    if comps.empty:
        return dict.fromkeys(COMP_COLS + COMP_PRICE_COLS, np.nan)
    ppsf = comps["PricePerSqft"].median()
    area = living_area if living_area and living_area > 0 else np.nan
    return {"LogCompPricePerSqft": np.log(ppsf), "LogCompImpliedPrice": np.log(ppsf * area),
            "CompDistanceKm": comps["DistanceKm"].mean(), "LogCompMedianPrice": np.log(comps["ClosePrice"].median())}


def build_table() -> pd.DataFrame:
    """Dataset A plus listing date and every feature group (see the module docstring)."""
    df = load_clean(DATASET_A_PATH)
    df = add_school_districts(df, load_school_districts(DISTRICTS_PATH))
    df = add_listing_date(df, load_listing_details(FILTERED_PATH))
    df = add_mls_district(add_flooring(add_hoa(add_ratios(df))))
    return add_similar_sales(add_comparable_sales(df))
