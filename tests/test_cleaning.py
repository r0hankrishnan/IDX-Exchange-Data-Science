import numpy as np
import pandas as pd
import pytest

from idx_exchange_data_science import cleaning as cl


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
        **{col: [True, False] for col in cl.BINARY_COLS},
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
    out = cl.clean(raw_rows(**overrides))
    assert out[cl.ID_COL].tolist() == [1]


def test_duplicate_sale_keeps_first_after_normalizing_address():
    rows = raw_rows(UnparsedAddress=["1 Main St", "  1  MAIN st "], CloseDate=["2024-03-01", "2024-03-01"])
    assert cl.clean(rows)[cl.ID_COL].tolist() == [1]


def test_same_address_different_date_is_kept():
    rows = raw_rows(UnparsedAddress=["1 Main St", "1 Main St"])
    assert len(cl.clean(rows)) == 2


def test_same_listing_twice_keeps_the_latest_close():
    rows = raw_rows(ListingKey=[7, 7], CloseDate=["2024-05-01", "2024-03-01"], ClosePrice=[510_000.0, 500_000.0])
    out = cl.clean(rows)
    assert len(out) == 1 and out["ClosePrice"].iloc[0] == 510_000.0


def test_listing_keys_are_unique_after_cleaning():
    rows = pd.concat([raw_rows(), raw_rows(CloseDate=["2024-04-01", "2024-04-02"])], ignore_index=True)
    assert cl.clean(rows)[cl.ID_COL].is_unique


def test_fix_fields():
    rows = raw_rows(
        BedroomsTotal=[0.0, 25.0],
        Latitude=[50.0, 34.0],
        LotSizeSquareFeet=[0.0, np.nan],
        LotSizeAcres=[0.5, np.nan],
        PostalCode=["94513-9999", "ABCDE"],
    )
    out = cl.fix_fields(cl.parse_dates(rows))
    assert out["BedroomsTotal"].isna().all()
    assert out["Latitude"].isna().tolist() == [True, False]
    assert out["LotSizeSquareFeet"].iloc[0] == 0.5 * cl.SQFT_PER_ACRE
    assert np.isnan(out["LotSizeSquareFeet"].iloc[1])
    assert out["PostalCode"].iloc[0] == "94513" and pd.isna(out["PostalCode"].iloc[1])
    assert out["PropertyAge"].tolist() == [34, 24]
    assert out["ViewYN"].tolist() == [1.0, 0.0]


def test_mostly_empty_feature_is_rejected():
    with pytest.raises(ValueError, match="GarageSpaces"):
        cl.clean(raw_rows(GarageSpaces=[np.nan, np.nan]))


def test_team_value_rules():
    rows = raw_rows(LivingArea=[200.0, 1800.0], YearBuilt=[2030.0, 1800.0], LotSizeSquareFeet=[50.0, 7000.0],
                    LotSizeAcres=[np.nan, 0.16], ParkingTotal=[25.0, 31.0], Longitude=[-118.0, -110.0],
                    City=["  A ", "   "])
    out = cl.fix_fields(cl.parse_dates(rows))
    assert out["LivingArea"].isna().tolist() == [True, False]
    assert out["PropertyAge"].isna().all()  # YearBuilt out of range
    assert np.isnan(out["LotSizeSquareFeet"].iloc[0])
    assert out["ParkingTotal"].isna().tolist() == [False, True]
    assert out["Latitude"].isna().tolist() == [False, True]  # the whole pair is missing
    assert out["City"].iloc[0] == "A" and pd.isna(out["City"].iloc[1])


def test_negative_property_age_is_missing():
    out = cl.fix_fields(cl.parse_dates(raw_rows(YearBuilt=[2025.0, 2000.0])))
    assert out["PropertyAge"].isna().tolist() == [True, False]
