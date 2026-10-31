import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from idx_exchange_data_science import dataset_a as da
from idx_exchange_data_science import features as ft


def toy_districts() -> gpd.GeoDataFrame:
    """A unified district on the west half; an elementary and an overlapping high district on the east."""
    return gpd.GeoDataFrame({
        "District": ["West Unified (A)", "East Elementary (A)", "East High (A)"],
        "DistrictTy": ["Unified", "Elementary", "High"],
        "SEDpct": [20.0, 60.0, 50.0],
    }, geometry=[box(-119, 34, -118, 35), box(-118, 34, -117, 35), box(-118, 34, -117, 35)], crs="EPSG:4326")


def test_add_ratios():
    df = pd.DataFrame({"BedroomsTotal": [3.0, np.nan], "BathroomsTotalInteger": [2.0, 2.0],
                       "LivingArea": [1500.0, 2000.0], "LotSizeSquareFeet": [6000.0, 2000.0]})
    out = ft.add_ratios(df)
    assert out["BedBathRatio"].iloc[0] == 1.5
    assert out["SqftPerBedroom"].iloc[0] == 500
    assert np.isclose(out["LogLotToLivingArea"].iloc[0], np.log(4))
    assert out["LogLotToLivingArea"].iloc[1] == 0
    assert out[["BedBathRatio", "SqftPerBedroom"]].iloc[1].isna().all()


def test_school_districts_pick_the_serving_district():
    homes = pd.DataFrame({"Latitude": [34.5, 34.5, 34.5, np.nan],
                          "Longitude": [-118.5, -117.5, -110.0, -118.5]}, index=[10, 11, 12, 13])
    out = ft.add_school_districts(homes, toy_districts())

    assert out.loc[10, "ElementaryDistrict"] == out.loc[10, "HighDistrict"] == "West Unified (A)"
    assert out.loc[11, "ElementaryDistrict"] == "East Elementary (A)"
    assert out.loc[11, "HighDistrict"] == "East High (A)"
    assert out.loc[11, "DistrictSEDPct"] == 60.0  # from the elementary district
    assert out.loc[[12, 13], ft.DISTRICT_COLS].isna().all().all()  # outside every district / no coordinates
    assert out.index.tolist() == homes.index.tolist()


def test_elementary_district_wins_over_overlapping_unified():
    districts = pd.concat([toy_districts(), gpd.GeoDataFrame(
        {"District": ["Big Unified (A)"], "DistrictTy": ["Unified"], "SEDpct": [10.0]},
        geometry=[box(-120, 33, -116, 36)], crs="EPSG:4326")], ignore_index=True)
    out = ft.add_school_districts(pd.DataFrame({"Latitude": [34.5], "Longitude": [-117.5]}), districts)
    assert out["ElementaryDistrict"].item() == "East Elementary (A)"
    assert out["HighDistrict"].item() == "East High (A)"


def preprocessor() -> da.DatasetAPreprocessor:
    return da.DatasetAPreprocessor(extra_numeric=ft.RATIO_COLS + ft.DISTRICT_NUMERIC_COLS,
                                   extra_categorical=ft.DISTRICT_COLS)


def with_week6_features(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = ft.add_school_districts(ft.add_ratios(clean_df), toy_districts())
    return df, da.FEATURES + ft.RATIO_COLS + ft.DISTRICT_NUMERIC_COLS + ft.DISTRICT_COLS


def test_extra_columns_are_imputed_and_encoded_from_train(clean_df):
    clean_df.loc[clean_df.index[-5:], "BedroomsTotal"] = np.nan  # in the last (test) month
    df, features = with_week6_features(clean_df)
    train, test = da.split(df, "2024-08", 3)
    pre = preprocessor()
    X_train = pre.fit_transform(train[features], train[da.TARGET])
    X_test = pre.transform(test[features])

    assert not X_train.isna().any().any() and not X_test.isna().any().any()
    assert {"BedBathRatio", "DistrictSEDPct", "ElementaryDistrict_te", "HighDistrict_te"} <= set(X_test.columns)
    missing = test["BedBathRatio"].isna()
    assert missing.any()
    assert (X_test.loc[missing, "BedBathRatio"] == train["BedBathRatio"].median()).all()


def test_extra_columns_keep_test_rows_independent(clean_df):
    df, features = with_week6_features(clean_df)
    train, test = da.split(df, "2024-08", 3)
    pre = preprocessor().fit(train[features], train[da.TARGET])
    pd.testing.assert_frame_equal(pre.transform(test[features]).head(1), pre.transform(test[features].head(1)))


def test_no_extra_columns_matches_dataset_a(clean_df):
    train, test = da.split(clean_df, "2024-08", 3)
    plain = da.DatasetAPreprocessor().fit(train[da.FEATURES], train[da.TARGET])
    assert len(plain.get_feature_names_out()) == len(da.FEATURES) + len(da.BINARY_COLS)
