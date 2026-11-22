import numpy as np
import pandas as pd
import pytest

from idx_exchange_data_science import cleaning as cl


@pytest.fixture
def clean_df() -> pd.DataFrame:
    """A small Dataset A-shaped frame: 8 months x 60 sales, price driven by living area and ZIP."""
    rng = np.random.default_rng(0)
    n = 480
    zips = rng.choice(["90001", "90002", "90003"], n)
    living_area = rng.uniform(800, 4000, n)
    df = pd.DataFrame({
        cl.ID_COL: np.arange(n),
        cl.DATE_COL: pd.to_datetime("2024-01-01") + pd.to_timedelta(np.repeat(np.arange(8) * 31, 60) + 5, "D"),
        "LivingArea": living_area,
        "PropertyAge": rng.integers(0, 100, n).astype(float),
        "LotSizeSquareFeet": rng.uniform(2000, 20000, n),
        "BedroomsTotal": rng.integers(1, 6, n).astype(float),
        "BathroomsTotalInteger": rng.integers(1, 4, n).astype(float),
        "GarageSpaces": rng.integers(0, 3, n).astype(float),
        "ParkingTotal": rng.integers(0, 5, n).astype(float),
        "Latitude": 34 + rng.normal(0, 0.1, n),
        "Longitude": -118 + rng.normal(0, 0.1, n),
        "City": pd.Series(zips).map({"90001": "A", "90002": "B", "90003": "C"}),
        "CountyOrParish": "Los Angeles",
        "PostalCode": zips,
    })
    for col in cl.BINARY_COLS:
        df[col] = rng.integers(0, 2, n).astype(float)
    zip_premium = df["PostalCode"].map({"90001": 1.0, "90002": 1.5, "90003": 2.0})
    df[cl.TARGET] = 300 * living_area * zip_premium * rng.lognormal(0, 0.1, n)

    # sprinkle missing values over the features the preprocessor imputes
    for col in ["LivingArea", "BedroomsTotal", "Latitude", "Longitude", "PoolPrivateYN", "City"]:
        df.loc[rng.choice(n, 20, replace=False), col] = np.nan
    return df
