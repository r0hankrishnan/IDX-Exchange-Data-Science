import numpy as np
import pandas as pd
import pytest

from idx_exchange_data_science import dataset_a as da
from idx_exchange_data_science.splits import close_month, time_split


def raw_rows(**overrides) -> pd.DataFrame:
    """Two valid raw rows; keyword arguments override columns (lists of 2 values)."""
    rows = pd.DataFrame({
        "StateOrProvince": ["CA", "CA"],
        "PropertyType": ["Residential", "Residential"],
        "PropertySubType": ["SingleFamilyResidence", "SingleFamilyResidence"],
        "UnparsedAddress": ["1 Main St", "2 Oak Ave"],
        "CloseDate": ["2024-03-01", "2024-03-02"],
        "ListingContractDate": ["2024-01-01", "2024-01-01"],
        "ClosePrice": [500_000.0, 600_000.0],
        "ListingKey": [1, 2],
        "LivingArea": [1500.0, 1800.0],
        "YearBuilt": [1990.0, 2000.0],
        "LotSizeSquareFeet": [6000.0, 7000.0],
        "LotSizeAcres": [0.14, 0.16],
        "BedroomsTotal": [3.0, 4.0],
        "BathroomsTotalInteger": [2.0, 2.0],
        "GarageSpaces": [2.0, 2.0],
        "ParkingTotal": [2.0, 2.0],
        "Latitude": [34.0, 34.1],
        "Longitude": [-118.0, -118.1],
        "City": ["A", "B"],
        "CountyOrParish": ["Los Angeles", "Los Angeles"],
        "PostalCode": ["90001", "90002"],
        **{col: [True, False] for col in da.BINARY_COLS},
    })
    return rows.assign(**overrides)


# --- fixed cleaning rules ---------------------------------------------------

@pytest.mark.parametrize("overrides", [
    {"StateOrProvince": ["CA", "NV"]},
    {"PropertySubType": ["SingleFamilyResidence", "Condominium"]},
    {"ClosePrice": [500_000.0, 0.0]},
    {"ListingContractDate": ["2024-01-01", "2024-04-01"]},  # closed before listed
    {"LivingArea": [1500.0, -5.0]},
    {"UnparsedAddress": ["1 Main St", " Oak Ave"]},  # no street number
    {"UnparsedAddress": ["1 Main St", None]},
])
def test_clean_drops_invalid_row(overrides):
    out = da.clean(raw_rows(**overrides))
    assert out[da.ID_COL].tolist() == [1]


def test_duplicate_sale_keeps_first_after_normalizing_address():
    rows = raw_rows(UnparsedAddress=["1 Main St", "  1  MAIN st "], CloseDate=["2024-03-01", "2024-03-01"])
    assert da.clean(rows)[da.ID_COL].tolist() == [1]


def test_same_address_different_date_is_kept():
    rows = raw_rows(UnparsedAddress=["1 Main St", "1 Main St"])
    assert len(da.clean(rows)) == 2


def test_fix_fields():
    rows = raw_rows(
        BedroomsTotal=[0.0, 25.0],
        Latitude=[50.0, 34.0],
        LotSizeSquareFeet=[0.0, np.nan],
        LotSizeAcres=[0.5, np.nan],
        PostalCode=["94513-9999", "ABCDE"],
    )
    out = da.fix_fields(da.parse_dates(rows))
    assert out["BedroomsTotal"].isna().all()
    assert out["Latitude"].isna().tolist() == [True, False]
    assert out["LotSizeSquareFeet"].iloc[0] == 0.5 * da.SQFT_PER_ACRE
    assert np.isnan(out["LotSizeSquareFeet"].iloc[1])
    assert out["PostalCode"].iloc[0] == "94513" and pd.isna(out["PostalCode"].iloc[1])
    assert out["PropertyAge"].tolist() == [34, 24]
    assert out["ViewYN"].tolist() == [1.0, 0.0]


def test_mostly_empty_feature_is_rejected():
    with pytest.raises(ValueError, match="GarageSpaces"):
        da.clean(raw_rows(GarageSpaces=[np.nan, np.nan]))


# --- splits ------------------------------------------------------------------

def test_time_split_uses_only_the_months_before_test(clean_df):
    train, test = time_split(clean_df, "2024-08", train_months=3)
    assert set(close_month(train).astype(str)) == {"2024-05", "2024-06", "2024-07"}
    assert set(close_month(test).astype(str)) == {"2024-08"}


def test_time_split_needs_enough_history(clean_df):
    with pytest.raises(ValueError):
        time_split(clean_df, "2024-03", train_months=6)


def test_price_limits_come_from_train_only(clean_df):
    _, test_before = da.split(clean_df, "2024-08", 3)
    in_test = close_month(clean_df) == pd.Period("2024-08")
    shifted = clean_df.copy()
    shifted.loc[in_test, da.TARGET] *= 1000  # wild test prices must not move the limits
    train_after, test_after = da.split(shifted, "2024-08", 3)

    low, high = da.price_limits(time_split(clean_df, "2024-08", 3)[0])
    assert train_after[da.TARGET].between(low, high).all()
    assert len(test_before) > 0 and len(test_after) == 0  # every inflated test price is above the train limit


# --- fitted preprocessing ----------------------------------------------------

def test_preprocessor_fills_everything_from_train(clean_df):
    train, test = da.split(clean_df, "2024-08", 3)
    pre = da.DatasetAPreprocessor()
    X_train = pre.fit_transform(train[da.FEATURES], train[da.TARGET])
    X_test = pre.transform(test[da.FEATURES])

    assert not X_train.isna().any().any() and not X_test.isna().any().any()
    assert list(X_test.columns) == list(pre.get_feature_names_out())
    missing_area = test["LivingArea"].isna()
    assert (X_test.loc[missing_area, "LivingArea"] == train["LivingArea"].median()).all()


def test_coordinates_fall_back_from_zip_to_county_to_overall(clean_df):
    train = clean_df.dropna(subset=["Latitude"])
    pre = da.DatasetAPreprocessor().fit(train[da.FEATURES], train[da.TARGET])
    new = train[da.FEATURES].head(3).copy()
    new["Latitude"] = np.nan
    new["PostalCode"] = ["90001", "99999", "99999"]
    new["CountyOrParish"] = ["Los Angeles", "Los Angeles", "Nowhere"]

    lat = pre.transform(new)["Latitude"]
    assert lat.iloc[0] == train.loc[train["PostalCode"] == "90001", "Latitude"].median()
    assert lat.iloc[1] == train["Latitude"].median()  # one county, so county median = overall
    assert lat.iloc[2] == train["Latitude"].median()


def test_transform_rows_are_independent_of_other_test_rows(clean_df):
    """A test row's features must not depend on what else is in the test set."""
    train, test = da.split(clean_df, "2024-08", 3)
    pre = da.DatasetAPreprocessor().fit(train[da.FEATURES], train[da.TARGET])
    full = pre.transform(test[da.FEATURES])
    alone = pre.transform(test[da.FEATURES].head(1))
    pd.testing.assert_frame_equal(full.head(1), alone)


def test_unseen_category_gets_a_finite_encoding(clean_df):
    train, test = da.split(clean_df, "2024-08", 3)
    pre = da.DatasetAPreprocessor().fit(train[da.FEATURES], train[da.TARGET])
    test = test.assign(PostalCode="00000", City="Nowhere")
    assert np.isfinite(pre.transform(test[da.FEATURES])[["PostalCode_te", "City_te"]]).all().all()
