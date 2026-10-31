"""Week 6 features added on top of Dataset A. Both steps are fixed rules (nothing is
learned from prices), so they can run on the whole cleaned dataset before the split.

    districts = load_school_districts("data/external/school_districts_2425/DistrictAreas2425.shp")
    df = add_school_districts(add_ratios(df), districts)

    model = build_model(RandomForestRegressor(), preprocessor=DatasetAPreprocessor(
        extra_numeric=RATIO_COLS + DISTRICT_NUMERIC_COLS, extra_categorical=DISTRICT_COLS))

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

RATIO_COLS = ["BedBathRatio", "SqftPerBedroom", "LogLotToLivingArea"]
DISTRICT_COLS = ["ElementaryDistrict", "HighDistrict"]
# Share of socioeconomically disadvantaged students in the elementary district.
DISTRICT_NUMERIC_COLS = ["DistrictSEDPct"]


def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Ratios of Dataset A columns. NaN where an input is missing (imputed later on train)."""
    return df.assign(
        BedBathRatio=df["BedroomsTotal"] / df["BathroomsTotalInteger"],
        SqftPerBedroom=df["LivingArea"] / df["BedroomsTotal"],
        LogLotToLivingArea=np.log(df["LotSizeSquareFeet"] / df["LivingArea"]),
    )


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
