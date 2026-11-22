from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0088
SQFT_PER_ACRE = 43_560
COMP_FEATURES = [
    "comp_ppsf_median", "comp_ppsf_mean", "comp_km_kth", "comp_count",
    "sales_within_1km", "comp_implied_value",
]
MARKET_FEATURES = [
    "market_zip_3m", "market_county_3m", "market_county_12m", "market_county_change"
]
RATIO_COLS = ["BedBathRatio", "SqftPerBedroom", "LogLotToLivingArea"]
DISTRICT_COLS = ["ElementaryDistrict", "HighDistrict"]
DISTRICT_NUMERIC_COLS = ["DistrictSEDPct"]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = pd.to_numeric(denominator, errors="coerce")
    num = pd.to_numeric(numerator, errors="coerce")
    return num.div(den.where(den > 0))


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the canonical ratio features in the uploaded ``features.py``."""
    out = df.copy()
    if {"BedroomsTotal", "BathroomsTotalInteger"} <= set(out.columns):
        out["BedBathRatio"] = _safe_ratio(out["BedroomsTotal"], out["BathroomsTotalInteger"])
    if {"LivingArea", "BedroomsTotal"} <= set(out.columns):
        out["SqftPerBedroom"] = _safe_ratio(out["LivingArea"], out["BedroomsTotal"])
    if {"LotSizeSquareFeet", "LivingArea"} <= set(out.columns):
        lot_to_living = _safe_ratio(out["LotSizeSquareFeet"], out["LivingArea"])
        out["LogLotToLivingArea"] = np.log(lot_to_living.where(lot_to_living > 0))

    return out


def add_candidate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add extra candidates from the feature-ablation notebook, if available."""
    out = df.copy()
    if {"LotSizeSquareFeet", "LivingArea"} <= set(out.columns):
        yard = pd.to_numeric(out["LotSizeSquareFeet"], errors="coerce") - pd.to_numeric(
            out["LivingArea"], errors="coerce")
        out["YardSqft"] = yard.where(yard >= 0)
        out["LotCoverage"] = _safe_ratio(out["LivingArea"], out["LotSizeSquareFeet"])
    if {"BedroomsTotal", "BathroomsTotalInteger"} <= set(out.columns):
        out["BathsPerBed"] = _safe_ratio(out["BathroomsTotalInteger"], out["BedroomsTotal"])
    if {"MainLevelBedrooms", "BedroomsTotal"} <= set(out.columns):
        out["MainLevelShare"] = _safe_ratio(out["MainLevelBedrooms"], out["BedroomsTotal"])
    if "Stories" in out:
        out["StoriesFilled"] = pd.to_numeric(out["Stories"], errors="coerce")
    if "SubdivisionName" in out:
        sub = out["SubdivisionName"].astype("string").str.strip()
        out["InSubdivision"] = (sub.notna() & sub.ne("") & sub.str.lower().ne("nan")).astype("int8")
    return out


def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compatibility name matching the uploaded ``features.py`` API."""
    return add_basic_features(df)


def add_fee_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive monthly HOA fee, presence, and missingness fields."""
    out = df.copy()
    if "AssociationFeeMonthly" in out:
        fee = pd.to_numeric(out["AssociationFeeMonthly"], errors="coerce")
    elif "AssociationFee" in out:
        raw = pd.to_numeric(out["AssociationFee"], errors="coerce")
        frequency = out.get("AssociationFeeFrequency", pd.Series(index=out.index, dtype="object"))
        multiplier = frequency.map({"Monthly": 1.0, "Quarterly": 1/3,
                                   "SemiAnnually": 1/6, "Annually": 1/12}).fillna(1.0)
        fee = raw * multiplier
    else:
        return out
    out["AssociationFee_missing"] = fee.isna().astype("int8")
    out["HasHOA"] = fee.gt(0).astype("int8")
    out["fee_monthly"] = fee
    return out


def load_school_districts(path) -> pd.DataFrame:
    """Load the California CDE district shapefile used in ``features.py``."""
    import geopandas as gpd

    districts = gpd.read_file(path)
    result = gpd.GeoDataFrame({
        "District": districts["DistrictNa"].astype("string") + " (" + districts["CountyName"].astype("string") + ")",
        "DistrictTy": districts["DistrictTy"],
        "SEDpct": pd.to_numeric(districts["SEDpct"], errors="coerce"),
    }, geometry=districts.geometry, crs=districts.crs)
    if result.crs is None:
        raise ValueError("School district file must declare its CRS")
    return result.to_crs("EPSG:4326")


def add_school_districts(df: pd.DataFrame, districts) -> pd.DataFrame:
    """Compatibility name matching the uploaded ``features.py`` API."""
    return add_school_district_features(df, districts)


def add_school_district_features(df: pd.DataFrame, boundaries) -> pd.DataFrame:
    """District names are qualified with county during loading, then each property
    receives an elementary-served district, high-served district, and the
    elementary district's SED percentage. Specialized districts take priority;
    Unified is used as the fallback for both school levels.
    """
    import geopandas as gpd

    needed = {"Latitude", "Longitude"}
    if not needed <= set(df.columns):
        raise KeyError(f"School join requires {sorted(needed)}")
    out = df.copy()
    out["ElementaryDistrict"] = pd.NA
    out["HighDistrict"] = pd.NA
    out["DistrictSEDPct"] = np.nan
    valid = out["Latitude"].notna() & out["Longitude"].notna()
    if not valid.any():
        return out
    positions = np.flatnonzero(valid.to_numpy())
    points = gpd.GeoDataFrame({"_row_id": positions}, geometry=gpd.points_from_xy(
        out.loc[valid, "Longitude"], out.loc[valid, "Latitude"]), crs="EPSG:4326")
    boundary = boundaries.copy()
    if boundary.crs is None:
        raise ValueError("School district boundary data must declare its CRS")
    matches = gpd.sjoin(points, boundary.to_crs("EPSG:4326"), how="left", predicate="within")

    def serving(own_type: str) -> pd.DataFrame:
        candidates = matches.loc[matches["DistrictTy"].isin([own_type, "Unified"])].copy()
        candidates["_fallback"] = candidates["DistrictTy"].ne(own_type)
        candidates = candidates.sort_values(["_row_id", "_fallback"], kind="stable")
        return candidates.drop_duplicates("_row_id").set_index("_row_id").reindex(positions)

    elementary, high = serving("Elementary"), serving("High")
    out.iloc[positions, out.columns.get_loc("ElementaryDistrict")] = elementary["District"].to_numpy()
    out.iloc[positions, out.columns.get_loc("HighDistrict")] = high["District"].to_numpy()
    out.iloc[positions, out.columns.get_loc("DistrictSEDPct")] = pd.to_numeric(
        elementary["SEDpct"], errors="coerce").to_numpy()
    return out


def fit_coordinate_medians(reference: pd.DataFrame) -> dict:
    """Learn ZIP, county, and overall coordinate medians from reference rows."""
    required = {"Latitude", "Longitude", "PostalCode", "CountyOrParish"}
    missing = required - set(reference.columns)
    if missing:
        raise KeyError(f"Coordinate imputation is missing columns: {sorted(missing)}")
    coords = reference[["Latitude", "Longitude"]].apply(pd.to_numeric, errors="coerce")
    geo = reference[["PostalCode", "CountyOrParish"]].copy()
    geo[["Latitude", "Longitude"]] = coords
    return {
        "zip": geo.groupby("PostalCode", dropna=True)[["Latitude", "Longitude"]].median(),
        "county": geo.groupby("CountyOrParish", dropna=True)[["Latitude", "Longitude"]].median(),
        "overall": coords.median(),
    }


def impute_coordinates(df: pd.DataFrame, medians: dict) -> pd.DataFrame:
    """Apply the training-derived ZIP -> county -> overall coordinate fallback."""
    out = df.copy()
    for col in ("Latitude", "Longitude"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}_missing"] = out[col].isna().astype("int8")
        out[col] = (
            out[col]
            .fillna(out["PostalCode"].map(medians["zip"][col]))
            .fillna(out["CountyOrParish"].map(medians["county"][col]))
            .fillna(medians["overall"][col])
        )
    return out


def add_geo_features(
    query: pd.DataFrame,
    history: pd.DataFrame,
    *,
    anchors: dict[str, tuple[float, float]] | None = None,
    n_clusters: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """Add fold-fitted spatial clusters and optional anchor distances.

    Anchor coordinates are ``(latitude, longitude)`` and must be supplied by
    the project config. The clusterer is fit on reference/training coordinates
    only; validation and test rows only receive predictions.
    """
    from sklearn.cluster import MiniBatchKMeans

    out = query.copy()
    ref = history[["Latitude", "Longitude"]].apply(pd.to_numeric, errors="coerce").dropna()
    coords = out[["Latitude", "Longitude"]].apply(pd.to_numeric, errors="coerce")
    valid = coords.notna().all(axis=1)
    if len(ref):
        count = max(1, min(int(n_clusters), len(ref)))
        clusterer = MiniBatchKMeans(n_clusters=count, n_init=3, random_state=random_state)
        clusterer.fit(ref.to_numpy(dtype=float))
        out["geo_cluster"] = pd.Series(pd.NA, index=out.index, dtype="string")
        out.loc[valid, "geo_cluster"] = clusterer.predict(coords.loc[valid].to_numpy(dtype=float)).astype(str)
    if anchors:
        lat1 = np.deg2rad(coords["Latitude"].to_numpy(dtype=float))
        lon1 = np.deg2rad(coords["Longitude"].to_numpy(dtype=float))
        for name, (lat2_deg, lon2_deg) in anchors.items():
            lat2, lon2 = np.deg2rad(lat2_deg), np.deg2rad(lon2_deg)
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            out[f"km_{name}"] = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
            out.loc[~valid, f"km_{name}"] = np.nan
    return out


def _month_series(df: pd.DataFrame) -> pd.PeriodIndex:
    if "CloseDate" in df:
        return pd.to_datetime(df["CloseDate"], errors="coerce").dt.to_period("M")
    if "SourceMonth" in df:
        return pd.PeriodIndex(df["SourceMonth"].astype(str), freq="M")
    raise KeyError("Comparable features require SourceMonth or CloseDate")


def add_comparable_features(
    query: pd.DataFrame,
    history: pd.DataFrame,
    *,
    lookback_months: int = 6,
    radius_km: float = 1.0,
    max_comps: int = 20,
) -> pd.DataFrame:
    """Candidate local comparable-sale features using prior-month sales.

    For each query month, history is restricted to the previous six complete
    calendar months (default) before the spatial search. This avoids using the
    subject sale or same-month/future validation labels. Tune lookback, radius,
    and neighbor count using training folds only.
    """
    if lookback_months < 1 or radius_km <= 0 or max_comps < 1:
        raise ValueError("lookback_months, radius_km, and max_comps must be positive")
    required = {"Latitude", "Longitude", "LivingArea", "ClosePrice"}
    missing = required - set(history.columns)
    if missing:
        raise KeyError(f"Comparable history is missing columns: {sorted(missing)}")
    q = query.copy()
    h = history.copy()
    q_month = _month_series(q)
    h_month = _month_series(h)
    hist_num = h[["Latitude", "Longitude", "LivingArea", "ClosePrice"]].apply(
        pd.to_numeric, errors="coerce"
    )
    h["_ppsf"] = hist_num["ClosePrice"].div(hist_num["LivingArea"].where(hist_num["LivingArea"] > 0))
    h["_month"] = h_month
    h = h.loc[
        h["_month"].notna() & h["_ppsf"].gt(0)
        & h["Latitude"].notna() & h["Longitude"].notna()
    ].copy()

    records = {name: np.full(len(q), np.nan, dtype=float) for name in COMP_FEATURES}
    qcoords = q[["Latitude", "Longitude"]].apply(pd.to_numeric, errors="coerce").to_numpy()
    qliving = pd.to_numeric(q.get("LivingArea", pd.Series(np.nan, index=q.index)), errors="coerce").to_numpy()
    radius_rad = radius_km / EARTH_RADIUS_KM
    # One tree per query month; the eligible history window is fixed for that month.
    for month in q_month.dropna().unique():
        positions = np.flatnonzero((q_month == month).to_numpy())
        start_month = month - lookback_months
        eligible = h.loc[(h["_month"] >= start_month) & (h["_month"] < month)]
        valid_pos = positions[np.isfinite(qcoords[positions]).all(axis=1)]
        if eligible.empty or not len(valid_pos):
            continue
        hcoords = np.deg2rad(eligible[["Latitude", "Longitude"]].to_numpy(dtype=float))
        tree = BallTree(hcoords, metric="haversine")
        qrad = np.deg2rad(qcoords[valid_pos])
        distance_lists, index_lists = tree.query_radius(
            qrad, r=radius_rad, return_distance=True, sort_results=True
        )
        ppsf = eligible["_ppsf"].to_numpy(dtype=float)
        for row_pos, dists, inds in zip(valid_pos, distance_lists, index_lists):
            if not len(inds):
                records["comp_count"][row_pos] = 0.0
                records["sales_within_1km"][row_pos] = 0.0
                continue
            chosen = inds[:max_comps]
            vals = ppsf[chosen]
            km = dists * EARTH_RADIUS_KM
            records["comp_ppsf_median"][row_pos] = float(np.median(vals))
            records["comp_ppsf_mean"][row_pos] = float(np.mean(vals))
            records["comp_km_kth"][row_pos] = float(km[min(max_comps, len(km)) - 1])
            records["comp_count"][row_pos] = float(len(chosen))
            records["sales_within_1km"][row_pos] = float(len(inds))
            if np.isfinite(qliving[row_pos]) and qliving[row_pos] > 0:
                records["comp_implied_value"][row_pos] = float(np.median(vals) * qliving[row_pos])
    for name, values in records.items():
        q[name] = values
    return q


def add_market_trend_features(
    query: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Candidate prior-month median price-per-square-foot market trends.
    The three- and twelve-month values use completed months before each query
    month. ``market_county_change`` is the percent change in county median
    price-per-square-foot between the most recent three months and the prior
    three months. Validation rows use training history only when passed a
    fold's training rows as ``history``. 
    """
    required = {"ClosePrice", "LivingArea", "CountyOrParish"}
    if not required <= set(history.columns):
        raise KeyError(f"Market trend history requires {sorted(required)}")
    q = query.copy()
    q_month = _month_series(q)
    h = history.copy()
    h["_month"] = _month_series(h)
    price = pd.to_numeric(h["ClosePrice"], errors="coerce")
    area = pd.to_numeric(h["LivingArea"], errors="coerce")
    h["_ppsf"] = price.div(area.where(area > 0))
    h = h.loc[h["_ppsf"].gt(0) & h["_month"].notna()].copy()
    for col in MARKET_FEATURES:
        q[col] = np.nan

    for month in q_month.dropna().unique():
        pos = np.flatnonzero((q_month == month).to_numpy())
        recent3 = h.loc[(h["_month"] >= month - 3) & (h["_month"] < month)]
        recent12 = h.loc[(h["_month"] >= month - 12) & (h["_month"] < month)]
        previous3 = h.loc[(h["_month"] >= month - 6) & (h["_month"] < month - 3)]
        query_rows = q.iloc[pos]
        if "PostalCode" in q and "PostalCode" in recent3:
            by_zip = recent3.groupby("PostalCode", dropna=True)["_ppsf"].median()
            q.iloc[pos, q.columns.get_loc("market_zip_3m")] = query_rows["PostalCode"].map(by_zip).to_numpy()
        by_county_3 = recent3.groupby("CountyOrParish", dropna=True)["_ppsf"].median()
        by_county_12 = recent12.groupby("CountyOrParish", dropna=True)["_ppsf"].median()
        by_county_prev = previous3.groupby("CountyOrParish", dropna=True)["_ppsf"].median()
        current = query_rows["CountyOrParish"].map(by_county_3)
        longer = query_rows["CountyOrParish"].map(by_county_12)
        previous = query_rows["CountyOrParish"].map(by_county_prev)
        q.iloc[pos, q.columns.get_loc("market_county_3m")] = current.to_numpy()
        q.iloc[pos, q.columns.get_loc("market_county_12m")] = longer.to_numpy()
        change = current.div(previous.where(previous > 0)) - 1.0
        q.iloc[pos, q.columns.get_loc("market_county_change")] = change.to_numpy()
    return q


def engineer_features(
    query: pd.DataFrame,
    *,
    history: pd.DataFrame | None = None,
    school_boundaries=None,
    anchors: dict[str, tuple[float, float]] | None = None,
    n_geo_clusters: int = 100,
    include_comps: bool = True,
) -> pd.DataFrame:
    """Apply the shared feature recipe; pass fold-local history for comps."""
    reference = query if history is None else history
    coord_medians = fit_coordinate_medians(reference)
    history_ready = impute_coordinates(reference, coord_medians)
    out = impute_coordinates(query, coord_medians)
    for col in ("AttachedGarageYN", "FireplaceYN", "NewConstructionYN", "PoolPrivateYN", "ViewYN"):
        if col in out:
            out[f"{col}_missing"] = out[col].isna().astype("int8")
    out = add_fee_features(add_candidate_columns(add_basic_features(out)))
    out = add_geo_features(out, history_ready, anchors=anchors, n_clusters=n_geo_clusters)
    if school_boundaries is not None:
        out = add_school_district_features(out, school_boundaries)
    if include_comps:
        if history is None:
            raise ValueError("Pass history explicitly to build comparable-sale features")
        out = add_comparable_features(out, history_ready)
        out = add_market_trend_features(out, history_ready)
    return out

