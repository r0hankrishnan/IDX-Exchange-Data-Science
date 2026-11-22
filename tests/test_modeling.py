import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeRegressor

from idx_exchange_data_science import cleaning as cl
from idx_exchange_data_science.modeling import (
    backtest, build_model, final_estimator, fit_window, gain_summary, linear_baseline, median_ape_by, score,
)
from idx_exchange_data_science.splits import MonthSplit, split

MODELS = {
    "linear": build_model(LinearRegression(), scale=True),
    "linear log": linear_baseline(),
    "tree": build_model(DecisionTreeRegressor(min_samples_leaf=5, random_state=0)),
    "forest log": build_model(RandomForestRegressor(n_estimators=10, random_state=0), log_target=True),
}


@pytest.mark.parametrize("name", MODELS)
def test_evaluation_prices_never_reach_the_model(clean_df, name):
    """Predictions must not change when the evaluated rows' ClosePrice values change."""
    model = fit_window(MODELS[name], clean_df, "2024-08", 3)
    _, evaluation = split(clean_df, "2024-08", 3)
    scrambled = evaluation.assign(**{cl.TARGET: evaluation[cl.TARGET].sample(frac=1, random_state=1).to_numpy()})
    np.testing.assert_allclose(model.predict(evaluation), model.predict(scrambled))


@pytest.mark.parametrize("name", MODELS)
def test_models_learn_something(clean_df, name):
    scores, _ = backtest({name: MODELS[name]}, clean_df, ["2024-08"], 3)
    assert scores["r2"].item() > 0.5


def test_log_target_predicts_dollars(clean_df):
    _, predictions = backtest({"linear log": MODELS["linear log"]}, clean_df, ["2024-08"], 3)
    ratio = predictions["linear log"] / clean_df.loc[predictions.index, cl.TARGET]
    assert 0.5 < ratio.median() < 2


def test_backtest_does_not_modify_the_given_model(clean_df):
    backtest({"linear": MODELS["linear"]}, clean_df, ["2024-08"], 3)
    assert not hasattr(MODELS["linear"][-1], "coef_")


def test_backtest_scores_each_model_and_month_on_the_same_rows(clean_df):
    scores, predictions = backtest({"linear": MODELS["linear"], "tree": MODELS["tree"]}, clean_df,
                                   ["2024-06", "2024-07"], 3)
    assert len(scores) == 4 and list(predictions.columns) == ["linear", "tree"]
    _, june = split(clean_df, "2024-06", 3)
    _, july = split(clean_df, "2024-07", 3)
    assert predictions.index.equals(june.index.append(july.index))
    june_scores = score(june[cl.TARGET], predictions.loc[june.index, "tree"])
    assert scores.query("model == 'tree' and month == '2024-06'")["r2"].item() == pytest.approx(june_scores["r2"])


def test_backtest_matches_cross_validate(clean_df):
    """MonthSplit is a regular sklearn splitter, so cross_validate gives the same scores, train scores included."""
    scores, _ = backtest({"linear": MODELS["linear"]}, clean_df, ["2024-06", "2024-07"], 3)
    cv = cross_validate(MODELS["linear"], clean_df, clean_df[cl.TARGET], scoring="r2", return_train_score=True,
                        cv=MonthSplit(["2024-06", "2024-07"], 3))
    np.testing.assert_allclose(scores["r2"], cv["test_score"])
    np.testing.assert_allclose(scores["train_r2"], cv["train_score"])


def test_score():
    scores = score([100.0, 200.0, 400.0], [110.0, 200.0, 300.0])
    assert scores["mape"] == pytest.approx((0.1 + 0 + 0.25) / 3)
    assert scores["median_ape"] == pytest.approx(0.1)
    assert scores["within_10pct"] == pytest.approx(1 / 3)  # 0.1 itself is not within 10%


def test_final_estimator_sees_through_the_log_target(clean_df):
    model = fit_window(MODELS["forest log"], clean_df, "2024-08", 3)
    forest = final_estimator(model)
    assert len(forest.feature_importances_) == len(forest.feature_names_in_)


def test_gain_summary():
    scores = pd.DataFrame({
        "model": ["ref", "new"] * 3, "month": ["a", "a", "b", "b", "c", "c"],
        "r2": [0.80, 0.82, 0.80, 0.79, 0.80, 0.81], "mae": [10, 9, 10, 11, 10, 9],
    })
    r2 = gain_summary(scores, "ref").loc["new"]
    assert r2["mean gain"] == pytest.approx((0.02 - 0.01 + 0.01) / 3)
    assert r2["months improved (of 3)"] == 2
    assert gain_summary(scores, "ref", metric="mae").loc["new", "months improved (of 3)"] == 2


def test_median_ape_by_group():
    actual = pd.Series([100.0, 100.0, 100.0, 200.0])
    predictions = pd.DataFrame({"a": [110.0, 90.0, 130.0, 200.0]})
    out = median_ape_by(predictions, actual, pd.Series(["low", "low", "low", "high"]))
    assert out.loc["low", "a"] == pytest.approx(10.0) and out.loc["high", "a"] == 0
