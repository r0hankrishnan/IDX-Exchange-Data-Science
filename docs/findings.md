# Findings log

This file records the result of each experiment. The newest results are at the top.

Unless a section gives a different method, each value is a mean over the 6 validation months (2025-10 to
2026-03). Each month uses a model trained on the 18 months before it. The test month (2026-04) was scored for the
selected model in notebook `08`.

## Dataset A fix and full rerun (2026-11-22)

**Problem.** The team's duplicate rule (same address and close date) did not find 233 sales that occur two times with
the same `ListingKey` and different close dates (overlapping monthly files; 75% have the same price). An earlier copy
of a sale could be in training and the later copy in validation: 51 of 59,411 validation sales and 6 April sales had
this problem.

**Fix.** Dataset A now follows the team's unified cleaning (`docs/team/unified_preprocessing.py`, `clean_source`):

- One row for each `ListingKey` (the latest close date), before the address rule. Dataset A: 308,786 rows (309,019
  before).
- The team's value rules: living area 300 to 15,000 sq ft, year built 1850 to 2027, lot size 100 sq ft to 100 acres,
  parking up to 30, California bounds for the coordinate pair, negative property age to missing, trimmed text.
- Missing flags for all median-imputed columns and the coordinates (before: yes/no fields only).
- Our extra removals stay: rows without a valid address, duplicate sales by address and close date, and living area
  of 0 or less. All feature values now match `clean_source` for the shared rows.

**Rerun.** All 34 candidates of `07` were fit again. One candidate is new: the team's room ratios (section 7).

| Candidate | MdAPE | R² | MAE | Within 10% | Training MdAPE |
|---|---|---|---|---|---|
| Ensemble: top 3 (C, C + ratios, plain + similar comps) | 6.88% | 0.9176 | $131.5K | 64.4% | n/a |
| **C: districts + HOA + flooring + comps + comp median price** | **7.02%** | 0.9144 | $133.8K | 63.6% | 5.7% |
| C + room ratios | 7.07% | 0.9145 | $134.1K | 63.4% | 5.7% |
| B: plain + similar comps | 7.11% | 0.9142 | $135.1K | 63.2% | 5.7% |
| A: similar comps only | 7.15% | 0.9109 | $136.4K | 63.2% | 5.8% |
| Served model: districts + HOA + flooring + comps | 7.21% | 0.9133 | $136.0K | 62.8% | 5.9% |
| Reference: random forest, districts + comps, dollar price | 7.85% | 0.8994 | $149.7K | 59.3% | n/a |

All rows except the reference are XGBoost (06 settings) on log price.

- **C against the served model:** 0.19 points lower, in 6 of 6 months. B passes (5 of 6). **A no longer passes**
  (4 of 6; it passed in 6 of 6 before the fix).
- **Room ratios do not help:** C + ratios is lower than C in only 1 of 6 months.
- **Top-3 ensemble:** 6.88%, lower than C in all 6 months. C + ratios is a member because the member rule is fixed.
- **School districts and MLS district:** still a tie (7.21% and 7.20%, 3 of 6 months).
- **Test month, rescored** with the same served model and the fixed data: MdAPE 7.32% (first score 7.40%), R² 0.9129
  (0.9152), 62.1% within 10%. Range coverage 78.2% / 89.2% / 94.7%. The model and settings did not change, so the
  rescore is not a second selection.
- **Notebooks `03` to `06` and `experiments/05b` to `05e`** used the Dataset A before the fix. Their results are not
  run again. Less than 1% of the values changed.

The sections below that are dated 2026-11-21 use the data before the fix. Their conclusions hold, except where this
section gives a different result.

## Similar comparable sales (notebook `07`, section 6, 2026-11-21)

Plain comparable sales are the 10 nearest earlier sales. Similar comparable sales are the 10 most similar homes among
the 50 nearest earlier sales. The similarity units were fixed in advance from appraisal rules of thumb: 25% of living
area, 20 years of age, 1 bedroom, 1 bathroom, a factor of 2 in lot size and 1 km (`features.SIMILARITY_UNITS`).
Three candidates, all XGBoost (06 settings) on log price with districts, HOA and flooring, against the served model
(MdAPE 7.23%):

| Candidate | MdAPE | Months lower (of 6) | R² |
|---|---|---|---|
| **C: plain comps + their median sale price** | **7.07%** | **6** | 0.9144 |
| B: plain + similar comps | 7.13% | 6 | 0.9144 |
| A: similar comps instead of plain comps | 7.15% | 6 | 0.9112 |

- All three pass the decision rule. C is the best.
- Before any model, the similar comparables are a better estimate than the plain ones (median error 9.7% against
  13.2%), but the gain in the model is smaller, because the model already adjusts for the home's own attributes.
- The top-3 ensemble (C, B, A; log-scale mean) has an MdAPE of 6.93%, lower than C in all 6 months, and R² 0.9171.
  Its members are more different than the earlier top 3 (correlation of log errors 0.92 to 0.96).
- `build_table()` takes approximately 56 seconds now (15 seconds before), because of the search over 50 sales.
- These candidates were added after the test month was scored, so the test month cannot check them independently.

## Test month (notebook `08_test_month`, 2026-11-21)

The selected model, trained on 2024-10 to 2026-03, scored one time on 2026-04 (12,309 sales).

- MdAPE 7.40% (validation 7.23%), R² 0.9152 (0.9132), MAE $143.9K ($136.0K), 62.1% within 10% (62.7%).
- The price ranges from the validation errors contained 78.1% (80% range), 89.1% (90%) and 94.6% (95%) of the test
  sale prices: close to the target, slightly narrow.
- The error increases with the distance to the comparable sales, as in validation (7.0% within 0.5 km, 15.8% at more
  than 5 km).

## The worst errors (notebook `experiments/07b_worst_errors`, 2026-11-21)

Validation predictions of the selected model (59,411 homes, 2025-10 to 2026-03, out of sample). The test month is
not used.

- **The worst 5% have an error of 31.4% or more** and give 23% of the total error. 81% of them are over-estimates.
- **The main cause is something the model cannot see.** These homes sold near their own list price (median 96%), but
  at 0.72x the price per sq ft of their comparable sales. Typical causes: poor condition, a teardown, a forced sale or
  a land lease. No field before listing shows this.
- **Rural homes have larger errors.** The median error is 6.9% when the comparable sales are within 0.5 km (80% of
  homes) and 15.1% when they are more than 5 km away. For lots above 10 acres, it is 18.0%.
- **Data problems are rare:** approximately 6 sales outside California, 6 mobile-home spaces, and 27 sales with a ZIP
  code outside California (most are California homes with an incorrect ZIP code).
- **Recommended:** price ranges by comparable-distance group; cleaning rules for the team to agree (remove sales
  outside California and mobile-home spaces, set invalid ZIP codes to missing). No new feature is available for the
  condition of a home before listing.

## Final model selection (notebook `07`, 2026-11-21)

We tested 5 feature sets, 3 models and 2 targets (dollar price and log price): 30 candidates in total. The decision
metric is the median absolute percentage error (MdAPE). The reference was the best model from week 6 (random forest
+ school districts + comparable sales, dollar price, MdAPE 7.86%).

**Result:** the best candidate is XGBoost (06 settings) on Dataset A + school districts + HOA + flooring +
comparable sales, fit on log price. The first run of `07` (R² as the decision metric, 15 candidates) selected the
same model.

| Candidate | MdAPE | R² | Within 10% | Training MdAPE | Months better than the reference |
|---|---|---|---|---|---|
| **Best: XGBoost (06), districts + HOA + flooring + comps, log price** | **7.23%** | 0.9132 | 62.7% | 5.9% | 6 of 6 |
| Same, MLS district | 7.24% | 0.9134 | 62.6% | 5.9% | 6 of 6 |
| Same, team XGBoost settings | 7.32% | 0.9127 | 62.4% | 6.3% | 6 of 6 |
| XGBoost (06), districts + HOA + flooring, log price | 7.32% | 0.9073 | 62.4% | 6.0% | 6 of 6 |
| XGBoost (06), districts + comps, log price | 7.42% | 0.9094 | 61.7% | 6.1% | 6 of 6 |
| Random forest, districts + HOA + flooring, log price | 7.53% | 0.8896 | 60.8% | 4.2% | 6 of 6 |
| Reference: random forest, districts + comps, dollar price | 7.86% | 0.8989 | 59.3% | n/a | n/a |

- **Log price:** this is the largest gain in MdAPE. It decreases MdAPE by 0.3 to 0.5 points for each XGBoost
  candidate, and by approximately 0.1 points for each random forest. The 10 best candidates are all XGBoost on log
  price.
- **Comparable sales:** with HOA and flooring in the set, comparable sales decrease MdAPE from 7.32% to 7.23%.
- **HOA and flooring:** by MdAPE, these help with or without comparable sales. For the random forest without
  comparable sales, they decrease MdAPE from 7.86% to 7.62% in all 6 months. By R², the same change made the random
  forest worse. Thus, the decision metric changes this result.
- **Settings:** the 06 XGBoost settings are better than the team settings for all feature sets and targets.
- **District source:** the spatial districts and the MLS district give the same result (a tie, 3 of 6 months). We
  keep the spatial districts. They are available for 99% of homes. The MLS district is missing for 26%.
- **Overfitting:** the best candidate has a training MdAPE of 5.9% and a validation MdAPE of 7.2%. The random
  forest has a larger difference (4.2% against 7.5%). Training scores are in-sample, so they are optimistic.
- **Every other close candidate** has a higher MdAPE than the best candidate in at least 5 of 6 months.
- **Ensembles (equal-weight mean of the log predictions, members selected by fixed rules):**
  - Top 3 candidates: MdAPE 7.16% (-0.07 points against the best candidate, lower in 5 of 6 months), R² 0.9153
    (higher in all 6 months), 63.2% within 10%.
  - Best XGBoost + best random forest: MdAPE 7.17% (-0.06 points, 5 of 6 months), but R² 0.9081 (lower in all 6
    months), because the forest has larger errors for the most expensive homes.
  - The top-3 members make very similar errors (correlation of log errors 0.97). The random forest is more different
    (0.88). The members were selected on the validation months, so part of the small gain can be selection bias.
- **Price bands:** the median error of the best candidate increases with the estimated price, from 5.6% to 6.6% in
  the three lowest fifths to 8.5% (fourth fifth) and 10.3% (highest fifth, estimates above approximately $1.55M).
  No other candidate is best in a specific band. A model selected separately for each band (selected on 2025-10 to
  2025-12, scored on 2026-01 to 2026-03) had an MdAPE of 7.25%, against 7.23% for the best candidate. Thus, we do not
  route by price band.

## Review of the team unified pipeline (2026-11-21)

We examined `docs/team/` (two Python files and a Word document) and the team's saved model
(`data/models/teammate_xgboost_school_hoa_flooring.joblib`).

- **Reported result:** the team's best model is tuned XGBoost + school districts + HOA + flooring. The team
  reported a validation R² of 0.909 and a test R² of 0.904. The team used a different split and different cleaning.
  For example, the team's random forest has an R² of 0.8826, and our random forest has 0.8897. Thus, the team
  values and our values are not comparable.
- **Settings:** the saved model and the document do not agree.
  - The saved model uses depth 7, learning rate 0.1, 600 trees, no column sampling, seed 42 and a dollar target.
  - The document gives depth 8 and column sampling 0.5, which is almost the same as our 06 settings. The document
    also gives a log-price target.
  - The document gives `objective="r2"`. This value is not an XGBoost objective. The saved model uses squared error.
- **Districts:** the saved model uses the MLS district text columns, not the spatial join. In our data, the MLS
  elementary district column is 100% empty.
- **Use:** the saved model cannot make predictions in this project. It needs the team's `idx_scripts` package.
- **Team experiments:** in the team results, comparable sales gave the largest single gain for XGBoost (+0.003 R²).
  HOA gave +0.0006 and flooring gave +0.0013. The team did not test HOA and flooring together with comparable sales.

## Week 7: advanced models (notebook `06`)

This notebook used single months to save time. We tuned the settings on 2025-09 and compared the models on 2026-03.
The features were the week 6 best.

- **XGBoost:** the tuned settings (depth 8, 600 trees, learning rate 0.092) gave the best R² and MAE: 0.914 against
  0.907 for the random forest. XGBoost trains in approximately 1 second, and the random forest needs 18 seconds.
- **Tuning months:** tuning in `06` used one month (2025-09). The protocol gives 3 tuning months (2025-07 to 2025-09).
- **Other gradient boosting models:** LightGBM had an R² of 0.910, and CatBoost had 0.911. CatBoost selected the
  maximum value of its search range. The mean of CatBoost and the random forest had an R² of 0.913.
- **Kumo Tabular:** this pretrained model from NVIDIA does no training on our data. It had the best percentage error:
  median APE 7.2% against 7.7%, and 62.7% of homes within 10%. Its R² was 0.908. Its 10% to 90% ranges contained
  78% of the sale prices, close to the 80% target.
  - Kumo Tabular needs logged size features and scaled inputs.
  - It must predict 500 rows at a time on the Mac GPU. Batches of 2,000 rows stopped.
- **SHAP:** for CatBoost, the comparable-sales price, ZIP code, coordinates and city have the most effect. Living area
  has a small effect, because the comparable-sales price already includes it.
- **macOS:** import LightGBM, XGBoost and CatBoost before torch. If you do not, the gradient boosting fits stop the
  kernel (two OpenMP runtimes).

## Week 6: features (notebook `05` and `notebooks/experiments/`)

A feature is accepted only if the random forest R² increases in most of the 6 months.

- **Comparable sales (`05e`):** random forest + school districts + comparable sales had an R² of 0.8989.
  - This was better than school districts alone (0.8923) in all 6 months.
  - The gain was mostly in expensive homes. The median APE was slightly worse in all 6 months.
- **School districts (`05`):** the random forest with school districts had an R² of 0.8923. This was better than
  Dataset A alone (0.8897) in all 6 months.
- **Room ratios (`05`):** the ratios (for example, bedrooms per bathroom) made the model worse in all months. We do
  not use them.
- **Pre-listing cutoff (`05b`):** we stopped the training data at the listing month of each home. This decreased R²
  by 0.002 or less. Thus, the standard split is satisfactory.
- **Listing supply and season (`05c`):** both made the model slightly worse (−0.0006 to −0.0015). We do not use them.
- **Size tiers (`05d`):** separate models for each size tier did not help the random forest. For linear regression,
  the tiers increased R² by 0.03. Thus, linear regression is a good test for data questions (timing, leakage) only.
  Do not use it to test model structure or most features.
- **Leakage check for comparable sales:**
  - No home uses a sale that closed after its listing month.
  - When we skip the most recent month of sales, the score changes very little.
  - In production, this feature needs a live feed of sold homes.
  - If the model sets list prices, its predictions become future comparable sales. This would increase the
    correlation but not the accuracy. Monitor the error over time.
- **Metric names:** "mape" in the results tables is the **mean** absolute percentage error. A small number of large
  errors make it high: the worst 5% of homes give 25% of the total. The median APE and the share within 10% are the
  usual metrics for home valuation models.
- **Missing raw columns:**
  - TaxAnnualAmount, WaterfrontYN, FireplacesTotal, AboveGradeFinishedArea and CoveredSpaces are 100% missing.
  - Levels is 9% missing, AssociationFee 30%, MLSAreaMajor 14% and MainLevelBedrooms 40%. We do not use
    MainLevelBedrooms.

## Ideas not tested yet

- **Raw columns:** Levels (use "Unknown" for missing values) and MLSAreaMajor (target encoding).
- **Zillow ZIP home value index (ZHVI):** use the value from before the listing month.
  `https://files.zillowstatic.com/research/public_csvs/zhvi/Zip_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv`
- **Distance to the coast:** use the Natural Earth coastline.
  `https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_coastline.geojson`
- **Wildfire hazard zones:** CAL FIRE publishes these zones. We did not find the polygon file. Download it manually.
- **Census tract median income.**
- **The worst 5% of errors:** look for what these homes have in common.
  - Bad data: add a cleaning rule.
  - Data that the model does not have (a view, a teardown, a new build): add a feature.
  - Areas with few nearby sales: give a wider price range.
  - Nothing in common: the errors are noise.
  - A separate model for these homes is not possible, because you know which homes they are only after the sale.
- **Price ranges by group (Mondrian conformal prediction):**
  1. Put the validation homes into groups by a value that is known before listing (for example, `CompDistanceKm`:
     less than 2 km, 2 to 5 km, 5 to 10 km, more than 10 km).
  2. Calculate the 90th percentile of the absolute percentage error for each group.
  3. Give the range of the home's group with each prediction.

  The app now uses one range for all homes. Other methods are quantile regression forests or a second model that
  predicts the size of the error.
