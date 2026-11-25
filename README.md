# IDX Exchange Data Science

This project predicts the sale price (`ClosePrice`) of a California single-family home before the home goes on the
market. The model uses only data that is known before the listing date.

## Data

The data is CRMLS (California Regional Multiple Listing Service) closed sales from January 2024 to April 2026. The
IDX Exchange program supplied it as monthly files.

| Stage | Rows | File |
|---|---|---|
| Monthly exports, all property types | 615,707 | `data/raw/` (28 files) |
| California single-family homes | 309,735 | `data/intermediate/CRMLSSoldFiltered.csv` |
| Dataset A (after the team cleaning rules) | 308,786 | `data/processed/CRMLSSoldDatasetA.csv` |

The school district boundaries come from
[California School District Areas 2024-25](https://data.ca.gov/dataset/california-school-district-areas-2024-25).
The `data/` folder is not in git, so each user must get a copy of the data files.

## Preprocessing

**Fixed cleaning rules** (`cleaning.py`) look at one row at a time, so they run one time on the full data. They follow
the team's `clean_source` (`docs/team/`) and `preprocessing-procedures.md`:

- Keep California single-family homes with a positive `ClosePrice` that did not close before the listing date.
- Keep one row for each `ListingKey`. Also remove duplicate sales (same address and close date), rows without a valid
  address and rows with a living area of 0 or less.
- Set values outside the valid ranges to missing. Calculate the lot size from acres and `PropertyAge` from `YearBuilt`.

**Fitted steps** (`splits.py`, `preprocessing.py`) learn values from data, so they use the training months only:

- Remove the top and bottom 0.5% of `ClosePrice`.
- Fill missing numbers with the median or the mode. Add a missing flag for each continuous field, yes/no field and
  coordinate. Fill missing coordinates from the ZIP code, then the county.
- Encode city, county, ZIP code and school district with cross-fitted target encoding.

## Features

`features.build_table()` adds these groups to Dataset A:

| Group | Columns |
|---|---|
| School districts | Elementary and high school district, and the share of disadvantaged students (spatial join of the coordinates) |
| HOA | Monthly fee, HOA yes/no, fee missing |
| Flooring | Wood, tile, carpet, laminate, vinyl, "see remarks", flooring missing |
| Comparable sales | The 10 nearest sales that closed in the 12 months before the listing month: their median price per square foot, that value times the living area, their mean distance and their median sale price (Model C) |

## Evaluation

- Each validation month (2025-10 to 2026-03) uses a model trained on the 18 months before it.
- The test month (2026-04) is scored only after the model is selected.
- The decision metric is the median absolute percentage error (MdAPE), because it shows the error for a typical home.
  A change is accepted if it decreases the mean MdAPE and decreases the MdAPE in at least 5 of the 6 months.
- We also report R², the mean absolute error (MAE) and the share of homes within 10% of the sale price.

## Results

Means over the 6 validation months:

| Model | MdAPE | R² | MAE | Within 10% |
|---|---|---|---|---|
| Linear regression, log price (notebook `03`) | 11.96% | 0.8007 | $207.0K | 42.9% |
| Random forest + school districts + comparable sales, dollar price | 7.85% | 0.8994 | $149.7K | 59.3% |
| XGBoost + school districts + HOA + flooring + comparable sales, log price | 7.21% | 0.9133 | $136.0K | 62.8% |
| **Model C: the same + median comparable price (served)** | **7.02%** | **0.9144** | **$133.8K** | **63.6%** |

- Model C is better than the model above it in all 6 validation months. The app uses Model C (from 2026-11-25).
- The test month (2026-04, 12,309 sales) was scored one time, for the earlier model: MdAPE 7.32%, R² 0.9129, MAE
  $144.0K, 62.1% within 10%. Its 80%, 90% and 95% price ranges contained 78.2%, 89.2% and 94.7% of the sale prices.
- Model C was made after that score. Its test-month MdAPE is 7.17% (R² 0.9165), but this is a second look at the
  test month, so the selection uses the validation months only.
- The linear regression row comes from the data before the Dataset A fix of 2026-11-22. All other rows use the fixed
  data (notebook `07`).

`docs/findings.md` gives the details of each experiment, including the features and models that did not help.

## How to run the code

Use [uv](https://docs.astral.sh/uv/). The file `uv.lock` pins the Python version and all packages.

1. Install the packages and run the tests (the tests do not need the data files):

   ```bash
   uv sync
   uv run pytest
   ```

2. Put the data files in these locations (`src/idx_exchange_data_science/paths.py` lists all paths):
   - `data/intermediate/CRMLSSoldFiltered.csv`
   - `data/external/school_districts_2425/DistrictAreas2425.shp`

3. Make Dataset A, then run the model notebooks. Each notebook writes its results table to `results/`.

   ```bash
   for nb in 02b_preprocessing_dataset_a 03_baseline_model 04_model_comparison 05_feature_engineering 06_advanced_models 07_final_model_selection; do
     uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace notebooks/$nb.ipynb
   done
   ```

   Notebook `04` needs approximately 8 minutes and notebook `07` approximately 13 minutes. Notebook `06` needs a GPU.
   Run `08_test_month` only for a final model, because the test month is scored one time.

| Notebook | Question |
|---|---|
| `01_exploration` | What does the data contain? |
| `02_preprocessing`, `02b_preprocessing_dataset_a` | Individual and team preprocessing (Dataset A) |
| `03_baseline_model` | Linear regression baseline |
| `04_model_comparison` | Linear model, tree or forest, and the training window length |
| `05_feature_engineering` | Room ratios and school districts |
| `06_advanced_models` | Gradient boosting and a pretrained tabular model |
| `07_final_model_selection` | Which feature set and model is best |
| `08_test_month` | The score of the selected model on the test month |

Side experiments are in `notebooks/experiments/`. Their tables are in `results/archive/`.

## How to start the app

```bash
uv run python -m idx_exchange_data_science.serving   # train and save the final model (approximately 30 seconds)
uv run streamlit run app/dashboard.py
```

Then open the URL that Streamlit shows (usually http://localhost:8501). Search for an address, city or ZIP code, or
select **Enter details** to enter a home. For each home, the app shows:

- The estimate, with 80%, 90% and 95% price ranges. A range contains the stated share of the validation sale prices.
- The sale record and the comparable sales, on a map and in a table.
- The features that had the most effect on the estimate.

The app uses Model C, trained on the latest 18 months (November 2024 to April 2026). The first start after a data
change builds the sales table (approximately 30 seconds) and saves it to `data/processed/app_table.parquet`. Later
starts read the saved table in less than 1 second.

## Code structure

The package `src/idx_exchange_data_science/` contains all shared code, in the order of the data flow:

| Module | Purpose |
|---|---|
| `paths` | The location of the data, results and model files |
| `cleaning` | The Dataset A cleaning rules |
| `features` | The feature groups (`build_table()`) |
| `splits` | The time-based training and evaluation splits |
| `preprocessing` | The fitted preprocessing step (`make_preprocessor()`) |
| `modeling` | The models, the validation backtest and the metrics |
| `serving` | Trains, saves and loads the final model |
