import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from idx_exchange_data_science import dataset_a as da
from idx_exchange_data_science import features as ft
from idx_exchange_data_science.splits import split


def toy_districts() -> gpd.GeoDataFrame:
    """A unified district on the west half; an elementary and an overlapping high district on the east."""
    return gpd.GeoDataFrame({
        "District": ["West Unified (A)", "East Elementary (A)", "East High (A)"],
        "DistrictTy": ["Unified", "Elementary", "High"],
        "SEDpct": [20.0, 60.0, 50.0],
    }, geometry=[box(-119, 34, -118, 35), box(-118, 34, -117, 35), box(-118, 34, -117, 35)], crs="EPSG:4326")


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


def with_districts(clean_df: pd.DataFrame) -> pd.DataFrame:
    df = ft.add_school_districts(clean_df, toy_districts())
    df.loc[df.index[-5:], "DistrictSEDPct"] = np.nan  # in the last (evaluation) month
    return df


def test_extra_columns_are_imputed_and_encoded_from_train(clean_df):
    train, test = split(with_districts(clean_df), "2024-08", 3)
    pre = da.make_preprocessor(numeric=ft.DISTRICT_NUMERIC_COLS, categorical=ft.DISTRICT_COLS)
    X_train = pre.fit_transform(train, train[da.TARGET])
    X_test = pre.transform(test)

    assert not X_train.isna().any().any() and not X_test.isna().any().any()
    assert {"DistrictSEDPct", "ElementaryDistrict", "HighDistrict"} <= set(X_test.columns)
    missing = test["DistrictSEDPct"].isna()
    assert missing.any()
    assert (X_test.loc[missing, "DistrictSEDPct"] == train["DistrictSEDPct"].median()).all()


def test_extra_columns_keep_test_rows_independent(clean_df):
    train, test = split(with_districts(clean_df), "2024-08", 3)
    pre = da.make_preprocessor(numeric=ft.DISTRICT_NUMERIC_COLS, categorical=ft.DISTRICT_COLS)
    pre.fit(train, train[da.TARGET])
    pd.testing.assert_frame_equal(pre.transform(test).head(1), pre.transform(test.head(1)))


# --- listing date ---------------------------------------------

def sales() -> pd.DataFrame:
    return pd.DataFrame({
        "ListingKey": [1, 2, 3, 4],
        "CloseDate": pd.to_datetime(["2024-05-10", "2024-05-20", "2024-05-25", "2024-05-30"]),
        "PostalCode": pd.array(["90001", "90001", "90002", None], dtype="string"),
    })


def test_listing_date_drops_implausible_gaps():
    dates = pd.DataFrame({"ListingKey": [1, 2, 3],
                          ft.LISTING_DATE: pd.to_datetime(["2024-03-15", "2021-01-01", "2024-06-01"])})
    out = ft.add_listing_date(sales(), dates)
    assert out[ft.LISTING_DATE].notna().tolist() == [True, False, False, False]  # too old, after close, unknown
    assert out["MonthsToClose"].iloc[0] == 2
    assert out["MonthsToClose"].iloc[1:].isna().all()


# --- comparable sales -------------------------------------------------------------

def comp_sales() -> pd.DataFrame:
    """Five homes on a line of longitude; the last one lists in 2024-06 and closes in 2024-08."""
    return pd.DataFrame({
        "Latitude": [34.0] * 5,
        "Longitude": [-118.00, -118.01, -118.02, -118.50, -118.005],
        "LivingArea": [1000.0, 1000.0, 1000.0, 1000.0, 2000.0],
        "ClosePrice": [400_000.0, 500_000.0, 600_000.0, 900_000.0, 1_000_000.0],
        "CloseDate": pd.to_datetime(["2024-03-10", "2024-04-10", "2024-05-10", "2024-05-20", "2024-08-10"]),
        ft.LISTING_DATE: pd.to_datetime(["2024-02-01", "2024-03-01", "2024-04-01", "2024-04-01", "2024-06-15"]),
    })


def test_comparables_are_nearest_earlier_sales():
    out = ft.add_comparable_sales(comp_sales(), k=2)
    home = out.iloc[4]
    assert np.isclose(home["LogCompPricePerSqft"], np.log(450))  # median of the two closest: 400 and 500 per sq ft
    assert np.isclose(home["LogCompImpliedPrice"], np.log(450 * 2000))
    assert 0 < home["CompDistanceKm"] < 2


def test_comparables_never_include_sales_after_listing():
    df = comp_sales()
    df.loc[4, ft.LISTING_DATE] = pd.Timestamp("2024-04-15")  # listed before the 2024-04 and 2024-05 sales closed
    out = ft.add_comparable_sales(df, k=1)
    assert np.isclose(out.iloc[4]["LogCompPricePerSqft"], np.log(400))  # only the 2024-03 sale is earlier


def test_comparables_respect_lookback_and_missing_inputs():
    df = comp_sales()
    out = ft.add_comparable_sales(df, k=2, lookback_months=1)  # only 2024-05 sales count for a 2024-06 listing
    assert np.isclose(out.iloc[4]["LogCompPricePerSqft"], np.log(750))  # 600 and 900 per sq ft
    assert np.isnan(ft.add_comparable_sales(df, k=10).iloc[4]["LogCompPricePerSqft"])  # not enough earlier sales

    df.loc[4, "Latitude"] = np.nan
    assert ft.add_comparable_sales(df, k=2)[ft.COMP_COLS].iloc[4].isna().all()


def test_find_comparables_matches_the_feature():
    df = comp_sales()
    home = ft.add_comparable_sales(df, k=2).iloc[4]
    comps = ft.find_comparables(df, 4, k=2)
    assert comps.index.tolist() == [0, 1]  # nearest first
    assert np.isclose(np.log(comps["PricePerSqft"].median()), home["LogCompPricePerSqft"])
    assert np.isclose(comps["DistanceKm"].mean(), home["CompDistanceKm"])
    assert ft.find_comparables(df, 4, k=10).empty  # not enough earlier sales


def test_missing_listing_date_assumes_two_months_before_close():
    df = comp_sales()
    df.loc[4, ft.LISTING_DATE] = pd.NaT  # closes 2024-08, so treated as listed in 2024-06
    assert np.isclose(ft.add_comparable_sales(df, k=2).iloc[4]["LogCompPricePerSqft"], np.log(450))
