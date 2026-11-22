import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from idx_exchange_data_science import cleaning as cl
from idx_exchange_data_science.preprocessing import make_preprocessor
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
    pre = make_preprocessor(numeric=ft.DISTRICT_NUMERIC_COLS, categorical=ft.DISTRICT_COLS)
    X_train = pre.fit_transform(train, train[cl.TARGET])
    X_test = pre.transform(test)

    assert not X_train.isna().any().any() and not X_test.isna().any().any()
    assert {"DistrictSEDPct", "ElementaryDistrict", "HighDistrict"} <= set(X_test.columns)
    missing = test["DistrictSEDPct"].isna()
    assert missing.any()
    assert (X_test.loc[missing, "DistrictSEDPct"] == train["DistrictSEDPct"].median()).all()


def test_extra_columns_keep_test_rows_independent(clean_df):
    train, test = split(with_districts(clean_df), "2024-08", 3)
    pre = make_preprocessor(numeric=ft.DISTRICT_NUMERIC_COLS, categorical=ft.DISTRICT_COLS)
    pre.fit(train, train[cl.TARGET])
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
    comps = ft.find_comparables(df, df.loc[[4]], k=2)
    assert comps.index.tolist() == [0, 1]  # nearest first
    features = ft.comparable_features(comps, df.loc[4, "LivingArea"])
    for col in ft.COMP_COLS:
        assert np.isclose(features[col], home[col])
    assert ft.find_comparables(df, df.loc[[4]], k=10).empty  # not enough earlier sales
    assert np.isnan(ft.comparable_features(comps.iloc[:0], 1000.0)["LogCompPricePerSqft"])


def test_find_comparables_works_for_a_home_not_in_the_sales():
    df = comp_sales()
    new_home = df.loc[[4]].assign(Longitude=-118.019)  # next to the 2024-05 sale
    assert ft.find_comparables(df.drop(index=4), new_home, k=1).index.tolist() == [2]


def test_missing_listing_date_assumes_two_months_before_close():
    df = comp_sales()
    df.loc[4, ft.LISTING_DATE] = pd.NaT  # closes 2024-08, so treated as listed in 2024-06
    assert np.isclose(ft.add_comparable_sales(df, k=2).iloc[4]["LogCompPricePerSqft"], np.log(450))


# --- HOA, flooring and MLS district ------------------------------------------------

def test_hoa_fee_is_monthly_with_flags():
    df = pd.DataFrame({"AssociationFee": [300.0, 1200.0, 0.0, np.nan, 50.0],
                       "AssociationFeeFrequency": ["Quarterly", "Annually", "Monthly", None, None]})
    out = ft.add_hoa(df)
    assert out["HOAFeeMonthly"].iloc[:3].tolist() == [100.0, 100.0, 0.0]
    assert out["HOAFeeMonthly"].iloc[4] == 50.0  # blank frequency taken as monthly
    assert out["HasHOA"].tolist() == [1.0, 1.0, 0.0, 0.0, 1.0]
    assert out["HOAFeeMissing"].tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]
    assert np.isnan(out["HOAFeeMonthly"].iloc[3])  # imputed later, from training rows
    assert list(out.columns) == ft.HOA_COLS


def test_flooring_flags_match_whole_types_only():
    df = pd.DataFrame({"Flooring": ["Carpet,Tile", " Wood , Vinyl", "SeeRemarks", None, "Stone"]})
    out = ft.add_flooring(df)
    assert out["FlooringTile"].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert out["FlooringWood"].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert out["FlooringSeeRemarks"].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert out["FlooringMissing"].tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]
    assert out.iloc[4].sum() == 0  # a type not on the list sets no flag
    assert list(out.columns) == ft.FLOORING_COLS


def test_mls_district_blanks_are_missing():
    out = ft.add_mls_district(pd.DataFrame({"HighSchoolDistrict": [" Irvine Unified ", "", None]}))
    assert out["MLSHighSchoolDistrict"].iloc[0] == "Irvine Unified"
    assert out["MLSHighSchoolDistrict"].iloc[1:].isna().all()


# --- similar comparable sales -------------------------------------------------------

def similar_sales() -> pd.DataFrame:
    """Three earlier sales and one home (index 3, 3,000 sq ft) listed after them, all on one line of latitude.

    Sales 0 and 1 are the nearest but much smaller; sale 2 is farther (~2 km) but the same size as the home.
    """
    return pd.DataFrame({
        "Latitude": [34.0] * 4,
        "Longitude": [-118.005, -118.006, -118.022, -118.000],
        "LivingArea": [1000.0, 1000.0, 3000.0, 3000.0],
        "LotSizeSquareFeet": [6000.0] * 4,
        "PropertyAge": [30.0] * 4,
        "BedroomsTotal": [3.0, 3.0, 4.0, 4.0],
        "BathroomsTotalInteger": [2.0, 2.0, 3.0, 3.0],
        "ClosePrice": [500_000.0, 520_000.0, 1_800_000.0, 2_000_000.0],
        "CloseDate": pd.to_datetime(["2024-03-10", "2024-03-12", "2024-03-15", "2024-08-10"]),
        ft.LISTING_DATE: pd.to_datetime(["2024-02-01", "2024-02-01", "2024-02-01", "2024-06-15"]),
    })


def test_similar_sales_prefer_a_similar_home_over_a_nearer_one():
    out = ft.add_similar_sales(similar_sales(), k=1)
    home = out.iloc[3]
    assert np.isclose(home["LogSimilarPricePerSqft"], np.log(1_800_000 / 3000))  # sale 2, not the nearer sale 0
    assert np.isclose(home["LogSimilarImpliedPrice"], np.log(1_800_000))
    assert 1.5 < home["SimilarDistanceKm"] < 2.5
    nearest = ft.add_comparable_sales(similar_sales(), k=1).iloc[3]
    assert np.isclose(nearest["LogCompPricePerSqft"], np.log(500))  # the plain feature takes the nearest sale


def test_similar_sales_equal_nearest_sales_when_homes_are_alike():
    """With identical attributes only distance counts, so the result equals add_comparable_sales (same timing rule)."""
    df = comp_sales().assign(LotSizeSquareFeet=6000.0, PropertyAge=30.0, BedroomsTotal=3.0, BathroomsTotalInteger=2.0)
    df["LivingArea"] = 1000.0
    similar, nearest = ft.add_similar_sales(df, k=2).iloc[4], ft.add_comparable_sales(df, k=2).iloc[4]
    assert np.isclose(similar["LogSimilarPricePerSqft"], nearest["LogCompPricePerSqft"])
    assert np.isclose(similar["SimilarDistanceKm"], nearest["CompDistanceKm"])
    assert np.isclose(nearest["LogCompMedianPrice"], np.log(450_000))  # median of the 400K and 500K sales


def test_similar_sales_ignore_missing_attributes():
    df = similar_sales()
    df.loc[3, ["BedroomsTotal", "PropertyAge"]] = np.nan
    home = ft.add_similar_sales(df, k=1).iloc[3]
    assert np.isclose(home["LogSimilarPricePerSqft"], np.log(600))  # still matched on living area
    assert np.isfinite(home["SimilarMatchScore"])
