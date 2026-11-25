import numpy as np
import pandas as pd

from idx_exchange_data_science import cleaning as cl
from idx_exchange_data_science.preprocessing import make_preprocessor
from idx_exchange_data_science.splits import split

N_FLAGS = len(cl.MEDIAN_COLS) + len(cl.BINARY_COLS) + len(cl.GEO_COLS)


def fitted_preprocessor(train: pd.DataFrame, **kwargs):
    return make_preprocessor(**kwargs).fit(train, train[cl.TARGET])


def test_preprocessor_fills_everything_from_train(clean_df):
    train, test = split(clean_df, "2024-08", 3)
    pre = make_preprocessor()
    X_train = pre.fit_transform(train, train[cl.TARGET])
    X_test = pre.transform(test)

    assert not X_train.isna().any().any() and not X_test.isna().any().any()
    assert list(X_test.columns) == list(pre.get_feature_names_out())
    assert len(X_test.columns) == len(cl.FEATURES) + N_FLAGS  # features + missing flags
    missing_area = test["LivingArea"].isna()
    assert (X_test.loc[missing_area, "LivingArea"] == train["LivingArea"].median()).all()


def test_preprocessor_never_sees_price_or_date(clean_df):
    train, test = split(clean_df, "2024-08", 3)
    X_test = fitted_preprocessor(train).transform(test)
    assert cl.TARGET not in X_test.columns and cl.DATE_COL not in X_test.columns


def test_missing_flags_exist_even_without_missing_values(clean_df):
    """The column set must not depend on which values happen to be missing in a training window."""
    train, _ = split(clean_df.fillna({col: 1.0 for col in cl.BINARY_COLS + cl.MEDIAN_COLS + cl.GEO_COLS}), "2024-08", 3)
    assert len(fitted_preprocessor(train).get_feature_names_out()) == len(cl.FEATURES) + N_FLAGS


def test_missing_coordinates_are_flagged(clean_df):
    train, test = split(clean_df, "2024-08", 3)
    X_test = fitted_preprocessor(train).transform(test)
    assert (X_test["missingindicator_Latitude"] == test["Latitude"].isna()).all()


def test_log_features_are_logged_after_imputation(clean_df):
    train, test = split(clean_df, "2024-08", 3)
    X_test = fitted_preprocessor(train, log_features=True).transform(test)
    missing_area = test["LivingArea"].isna()
    assert np.allclose(X_test.loc[~missing_area, "LivingArea"], np.log1p(test.loc[~missing_area, "LivingArea"]))
    assert (X_test.loc[missing_area, "LivingArea"] == np.log1p(train["LivingArea"].median())).all()


def test_coordinates_fall_back_from_zip_to_county_to_overall(clean_df):
    train = clean_df.dropna(subset=["Latitude"])
    pre = fitted_preprocessor(train)
    new = train.head(3).copy()
    new["Latitude"] = np.nan
    new["PostalCode"] = ["90001", "99999", "99999"]
    new["CountyOrParish"] = ["Los Angeles", "Los Angeles", "Nowhere"]

    lat = pre.transform(new)["Latitude"]
    assert lat.iloc[0] == train.loc[train["PostalCode"] == "90001", "Latitude"].median()
    assert lat.iloc[1] == train["Latitude"].median()  # one county, so county median = overall
    assert lat.iloc[2] == train["Latitude"].median()


def test_transform_rows_are_independent_of_other_test_rows(clean_df):
    """A test row's features must not depend on what else is in the test set."""
    train, test = split(clean_df, "2024-08", 3)
    pre = fitted_preprocessor(train)
    pd.testing.assert_frame_equal(pre.transform(test).head(1), pre.transform(test.head(1)))


def test_target_encoding_is_cross_fitted_on_train(clean_df):
    """In training, a row's encoding comes from the other folds, so it differs from the plain group mean."""
    train, _ = split(clean_df, "2024-08", 3)
    pre = make_preprocessor()
    cross_fitted = pre.fit_transform(train, train[cl.TARGET])["PostalCode"]
    refit = pre.transform(train)["PostalCode"]
    assert not np.allclose(cross_fitted, refit)


def test_unseen_category_gets_a_finite_encoding(clean_df):
    train, test = split(clean_df, "2024-08", 3)
    test = test.assign(PostalCode="00000", City="Nowhere")
    assert np.isfinite(fitted_preprocessor(train).transform(test)[["PostalCode", "City"]]).all().all()
