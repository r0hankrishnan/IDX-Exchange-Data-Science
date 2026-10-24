import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from idx_exchange_data_science import dataset_a as da
from idx_exchange_data_science.modeling import backtest, best_windows, build_model, fit_and_score, score_on_test
from idx_exchange_data_science.splits import close_month, time_split

MODELS = {
    "linear": build_model(LinearRegression(), scale=True),
    "linear log": build_model(LinearRegression(), scale=True, log_target=True, log_features=True),
    "tree": build_model(DecisionTreeRegressor(min_samples_leaf=5, random_state=0)),
    "forest log": build_model(RandomForestRegressor(n_estimators=10, random_state=0), log_target=True),
}


@pytest.mark.parametrize("name", MODELS)
def test_test_prices_never_reach_the_model(clean_df, name):
    """Predictions for the test month must not change when its ClosePrice values change."""
    train, test = da.split(clean_df, "2024-08", 3)
    model, _, _ = fit_and_score(MODELS[name], train, test)
    scrambled = test.assign(**{da.TARGET: test[da.TARGET].sample(frac=1, random_state=1).to_numpy()})
    np.testing.assert_allclose(model.predict(test[da.FEATURES]), model.predict(scrambled[da.FEATURES]))


@pytest.mark.parametrize("name", MODELS)
def test_models_learn_something(clean_df, name):
    train, test = da.split(clean_df, "2024-08", 3)
    _, _, scores = fit_and_score(MODELS[name], train, test)
    assert scores["r2"] > 0.5


def test_log_target_predicts_dollars(clean_df):
    train, test = da.split(clean_df, "2024-08", 3)
    model, _, _ = fit_and_score(MODELS["linear log"], train, test)
    ratio = model.predict(test[da.FEATURES]) / test[da.TARGET]
    assert 0.5 < np.median(ratio) < 2


def test_fit_and_score_does_not_modify_the_given_model(clean_df):
    train, test = da.split(clean_df, "2024-08", 3)
    fit_and_score(MODELS["linear"], train, test)
    assert not hasattr(MODELS["linear"][-1], "coef_")


def test_backtest_never_trains_on_the_evaluation_month(clean_df):
    seen = []

    def recording_split(df, month, train_months):
        train, evaluation = da.split(df, month, train_months)
        seen.append((month, set(close_month(train)), set(close_month(evaluation))))
        return train, evaluation

    results = backtest({"linear": MODELS["linear"]}, clean_df, ["2024-06", "2024-07"], [1, 3], split=recording_split)
    assert len(results) == 4
    for month, train_months, eval_months in seen:
        assert eval_months == {month} and max(train_months) < month


def test_best_windows_picks_best_mean_score():
    results = pd.DataFrame({
        "model": ["a", "a", "a", "a"], "X": [1, 1, 3, 3], "r2": [0.5, 0.7, 0.8, 0.8], "mae": [5, 5, 1, 9],
    })
    assert best_windows(results) == {"a": 3}
    assert best_windows(results, metric="mae") == {"a": 1}


def test_score_on_test_uses_each_models_window(clean_df):
    table, fitted = score_on_test(
        {"short": MODELS["linear"], "long": MODELS["linear"]}, clean_df, "2024-08", {"short": 1, "long": 6},
    )
    rows = table.set_index("model")["n_train"]
    assert rows["short"] < rows["long"]
    assert rows["long"] <= len(time_split(clean_df, "2024-08", 6)[0])
    assert set(fitted) == {"short", "long"}
