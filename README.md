# IDX Exchange Data Science

This project predicts the sale price (`ClosePrice`) of a California single-family home. The prediction is made
before the home goes on the market. Thus, the model uses only data that is known before the listing date.

## Dataset source

The data comes from CRMLS (California Regional Multiple Listing Service). The IDX Exchange program supplied it as
monthly files of closed sales, from January 2024 to April 2026.

| Stage | Rows | File |
|---|---|---|
| Monthly exports, all property types | 615,707 | `data/raw/` (28 files), combined in `data/intermediate/CRMLSSoldCombined.csv` |
| Residential, single-family homes | 309,735 | `data/intermediate/CRMLSSoldFiltered.csv` |
| Dataset A (after the team cleaning rules) | 309,019 | `data/processed/CRMLSSoldDatasetA.csv` |

Two external sources add features:

- **School districts:** [California School District Areas 2024-25](https://data.ca.gov/dataset/california-school-district-areas-2024-25)
  (California Department of Education, shapefile).
- **Listing details:** the listing date, HOA fee, flooring and MLS school district come from the filtered CRMLS file.

The `data/` folder is not in git. Each user must get a copy of the data files.

## Preprocessing

Preprocessing has two parts. The team agreed on the rules in `preprocessing-procedures.md`.

**1. Fixed cleaning rules (Dataset A).** These rules examine one row at a time. They learn nothing from the data.
Thus, they run one time on the full dataset, before the split.

- Keep California, Residential, single-family homes.
- Remove rows with a missing or zero `ClosePrice`.
- Remove rows that closed before the listing date.
- Remove duplicate sales (same normalized address and close date). Remove rows without a valid address.
- Remove rows with a living area of zero or less.
- Set values outside a valid range to missing (for example, more than 20 bedrooms, or coordinates outside California).
- Calculate the lot size from acres when the square-foot value is missing. Calculate `PropertyAge` from `YearBuilt`.

**2. Fitted preprocessing.** These steps learn values from data. Thus, they are fit on the training rows only,
inside each model pipeline (scikit-learn `ColumnTransformer`).

- Remove the top and bottom 0.5% of `ClosePrice`. The limits come from the training window.
- Fill missing numbers with the training median or mode. Add a flag for each missing yes/no value.
- Fill missing coordinates with the median of the same ZIP code, then the county, then all homes.
- Replace each city, county, ZIP code and school district with its mean training price (cross-fitted target
  encoding).

**Features added to Dataset A** (`features.build_table()`):

| Group | Columns | Source |
|---|---|---|
| School districts | Elementary district, high school district, share of disadvantaged students | Spatial join of the coordinates to the district boundaries |
| HOA | Monthly HOA fee, HOA yes/no, fee missing yes/no | `AssociationFee`, `AssociationFeeFrequency` |
| Flooring | Wood, tile, carpet, laminate, vinyl, "see remarks", flooring missing | `Flooring` |
| Comparable sales | Median price per square foot of the 10 nearest sales, that price times the living area, mean distance | Sales that closed in the 12 months before the listing month |

## Evaluation method

- **Time-based splits.** For each month, the model trains on the 18 months before that month.
- **Validation months:** 2025-10 to 2026-03. All decisions about models, features and settings use these 6 months.
- **Test month:** 2026-04. We score the test month one time, for the final model only.
- **Decision metric:** the median absolute percentage error (MdAPE). The goal is an accurate estimate for a typical
  home. R² and the dollar error give the most weight to expensive homes. MdAPE gives the same weight to each home.
  (Notebooks `03` to `06` used R² to make decisions.)
- **Decision rule:** a change is accepted if the mean validation MdAPE decreases and the MdAPE is lower in at least
  5 of the 6 months.
- **Other metrics:** R², the mean absolute error (MAE) in dollars, and the share of homes with a prediction within
  10% of the sale price.

## Models tested

All values are means over the 6 validation months.

| Model | Features | MdAPE | R² | MAE | Within 10% |
|---|---|---|---|---|---|
| Linear regression (log price, log sizes) | Dataset A | 11.96% | 0.8007 | $207.0K | 42.9% |
| Decision tree | Dataset A | 9.39% | 0.8508 | $182.4K | 52.3% |
| Random forest | Dataset A | 7.85% | 0.8897 | $153.3K | 59.0% |
| Random forest | + school districts | 7.74% | 0.8923 | $151.2K | 59.7% |
| Random forest | + school districts + comparable sales | 7.86% | 0.8989 | $149.8K | 59.3% |
| Random forest, log price | + school districts + HOA + flooring | 7.53% | 0.8896 | $148.4K | 60.8% |
| XGBoost | + school districts + comparable sales | 7.73% | 0.9083 | $143.8K | 60.1% |
| XGBoost | + school districts + HOA + flooring + comparable sales | 7.57% | 0.9122 | $140.2K | 61.0% |
| **XGBoost, log price** | **+ school districts + HOA + flooring + comparable sales** | **7.23%** | **0.9132** | **$136.0K** | **62.7%** |

Other tests:

- **Features that did not help:** room ratios, ZIP listing supply, listing month and size tiers.
- **Other models:** LightGBM, CatBoost and Kumo Tabular (a pretrained model from NVIDIA). We tested them on one
  validation month only, in notebook `06`.
- **Team settings:** the team's XGBoost settings, from the team's saved model, were 0.0005 to 0.0017 R² lower than
  our settings for every feature set.

`docs/findings.md` gives the details of each experiment.

## Best result

The best model is **XGBoost on Dataset A + school districts + HOA + flooring + comparable sales, fit on log price**
(notebook `07_final_model_selection`, 30 candidates, selected by MdAPE).

- The median home has a 7.23% error (MdAPE). 62.7% of homes are within 10% of the sale price.
- Validation R² is 0.9132, and MAE is $136.0K.
- Against the random forest from week 6, MdAPE is 0.63 points lower, and the model is better in all 6 months.

**Test month (2026-04, scored one time after the selection, notebook `08_test_month`):**

| Metric | Validation (mean of 6 months) | Test (2026-04, 12,309 sales) |
|---|---|---|
| MdAPE | 7.23% | 7.40% |
| R² | 0.9132 | 0.9152 |
| MAE | $136.0K | $143.9K |
| Within 10% | 62.7% | 62.1% |

The 80%, 90% and 95% price ranges contained 78.1%, 89.1% and 94.6% of the test sale prices.

## How to run the code again

Use [uv](https://docs.astral.sh/uv/). The file `uv.lock` pins the Python version and all packages.

1. Install the packages:

   ```bash
   uv sync
   ```

2. Run the tests. The tests use small synthetic data, so they do not need the data files.

   ```bash
   uv run pytest
   ```

3. Put the data files in these locations (`src/idx_exchange_data_science/paths.py` lists all paths):
   - `data/intermediate/CRMLSSoldFiltered.csv`
   - `data/external/school_districts_2425/DistrictAreas2425.shp` (unzip the download from the link above)

4. Make Dataset A:

   ```bash
   uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace notebooks/02b_preprocessing_dataset_a.ipynb
   ```

5. Run the model notebooks. Each notebook writes its results table to `results/`.

   ```bash
   for nb in 03_baseline_model 04_model_comparison 05_feature_engineering 06_advanced_models 07_final_model_selection; do
     uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace notebooks/$nb.ipynb
   done
   ```

   Notebook `04` needs approximately 8 minutes. Notebook `07` needs approximately 13 minutes. Notebook `06` needs a
   GPU for Kumo Tabular.

| Notebook | Question | Results table |
|---|---|---|
| `01_exploration` | What does the data contain? | none |
| `02_preprocessing` | Individual week 3 preprocessing | `data/processed/CRMLSSoldCleaned.csv` |
| `02b_preprocessing_dataset_a` | Team preprocessing (Dataset A) | `data/processed/CRMLSSoldDatasetA.csv` |
| `03_baseline_model` | Linear regression baseline | `results/baseline_results.csv` |
| `04_model_comparison` | Linear model, tree or forest, and the training window length | `results/model_comparison.csv` |
| `05_feature_engineering` | Room ratios and school districts | `results/feature_comparison.csv` |
| `06_advanced_models` | Gradient boosting and a pretrained tabular model | `results/advanced_models_comparison.csv` |
| `07_final_model_selection` | Which feature set and model is best | `results/final_model_selection.csv` |
| `08_test_month` | Score the selected model one time on the test month | `results/test_month.csv` |

The folder `notebooks/experiments/` contains side experiments. Their tables, and the month-by-month results of all
notebooks, are in `results/archive/`.

## How to start the app

1. Train the final model and save it to `data/models/final_model.joblib` (approximately 30 seconds):

   ```bash
   uv run python -m idx_exchange_data_science.serving
   ```

2. Start the app:

   ```bash
   uv run streamlit run app/dashboard.py
   ```

3. Open the URL that Streamlit shows (usually http://localhost:8501).

Use the app in one of two ways:

- **Search:** type part of an address, city or ZIP code in the search bar. Then select a home from the list.
- **Enter details:** select **Enter details**, then enter a ZIP code and the details of a home.

For each home, the app shows:

- The estimate, with 80%, 90% and 95% price ranges.
- The sale record (for a home in the data).
- The comparable sales, on a map and in a table.
- The features that had the most effect on the estimate.

To see a new estimate, open **Adjust the home details** and change a value. Select **Model** to see the model
details and the validation accuracy.

**Price ranges:** the ranges are not bootstrap intervals. The final model predicts each validation month with
training data from earlier months only. The 80% range contains the middle 80% of the ratios of sale price to
estimate. Thus, 80% of the homes that the model had not seen sold inside their 80% range.

The app uses the model that `serving.final_model()` defines: the best model from the table above (XGBoost on Dataset A + school districts + HOA + flooring + comparable sales, fit on log price). The model is trained on the latest 18 months of data (November 2024 to April 2026).

## Code structure

The package `src/idx_exchange_data_science/` contains all shared code. The modules are in the order of the data
flow:

| Module | Purpose |
|---|---|
| `paths` | Gives the location of the data, results and model files |
| `cleaning` | Applies the Dataset A cleaning rules |
| `features` | Adds the feature groups (`build_table()`) |
| `splits` | Makes the time-based training and evaluation splits |
| `preprocessing` | Makes the fitted preprocessing step (`make_preprocessor()`) |
| `modeling` | Builds models, runs the validation backtest, and calculates the metrics |
| `serving` | Trains, saves and loads the final model |

Example:

```python
from idx_exchange_data_science import backtest, build_model, build_table, make_preprocessor, tuned_xgboost, validation_months
from idx_exchange_data_science import features as ft

df = build_table()
model = build_model(tuned_xgboost(), make_preprocessor(numeric=ft.COMP_COLS, categorical=ft.DISTRICT_COLS))
scores, predictions = backtest({"xgboost": model}, df, validation_months("2026-04", 6), train_months=18)
```
