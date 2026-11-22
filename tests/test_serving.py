from sklearn.tree import DecisionTreeRegressor

from idx_exchange_data_science import cleaning as cl
from idx_exchange_data_science import serving
from idx_exchange_data_science.modeling import build_model


def test_final_model_trains_on_the_latest_months(clean_df, tmp_path):
    model = build_model(DecisionTreeRegressor(min_samples_leaf=5, random_state=0))
    bundle = serving.train_final_model(clean_df, model, train_months=3, n_validation_months=2)

    latest = clean_df[cl.DATE_COL].max()
    assert bundle["train_last"].to_period("M") == latest.to_period("M")  # includes the latest month
    assert bundle["train_first"].to_period("M") == latest.to_period("M") - 2
    assert bundle["validation_months"] == ["2024-06", "2024-07"]  # the months before the latest
    ranges = [bundle["ranges"][level] for level in serving.RANGE_LEVELS]
    assert all(low < 1 < high for low, high in ranges)
    assert ranges[0][0] >= ranges[1][0] >= ranges[2][0] and ranges[0][1] <= ranges[1][1] <= ranges[2][1]  # nested
    assert "LivingArea" in serving.input_columns(bundle["model"]) and cl.TARGET not in serving.input_columns(bundle["model"])

    serving.save(bundle, tmp_path / "model.joblib")
    loaded = serving.load(tmp_path / "model.joblib")
    assert (loaded["model"].predict(clean_df) == bundle["model"].predict(clean_df)).all()
