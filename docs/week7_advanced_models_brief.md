# Week 7 brief: advanced models (notebook 06)

> **Update (2026-11-21):** notebook `07_final_model_selection` tested XGBoost on all 6 validation months. XGBoost
> was better than the random forest in all 6 months. See `docs/findings.md`.

**Summary:** tuned XGBoost has the best R² and dollar error: 0.914, against 0.907 for the random forest (our best
model at that time). Kumo Tabular from NVIDIA has the lowest percentage error, but it did no training on our data.
Both results come from one validation month. Thus, neither model replaces the random forest yet.

## 1. Question and method

Can a stronger model do better than our best model? Our best model is a **random forest on Dataset A + school
districts + comparable sales** (notebook 05e).

- **Same inputs for all models:** all models use the same features and the same preprocessing. Imputation and target
  encoding are fit again on each training window. Each model trains on the 18 months before the month that it
  predicts.
- **Test month:** we did not use the test month (2026-04). We compare models on validation data only. Thus, the test
  month stays a fair final check.
- **Two separate months, to make the run shorter:**
  - **Tuning month, 2025-09.** We selected the settings of each gradient boosting model by its R² on this month.
  - **Comparison month, 2026-03.** We scored all models on this month. Tuning did not use this month, so the tuned
    models have no unfair advantage.
- **Noise:** R² usually changes by approximately ±0.005 from one month to the next. A smaller difference does not
  show a real improvement.

## 2. Models

| Model | Description |
|---|---|
| Linear regression | The baseline from week 4. It has one weight for each feature. |
| Random forest | Our best model at that time. It calculates the mean of hundreds of deep trees. |
| LightGBM, XGBoost, CatBoost | Gradient boosting. Each new small tree corrects the errors of the trees before it. These are the three main libraries. |
| Kumo Tabular (NVIDIA) | A pretrained model for tables. It does no training on our data. It reads 30,000 example homes with their prices, and it predicts new homes from these examples. |
| Ensemble | The mean of the random forest and the best gradient boosting model on the tuning month (CatBoost). |

**Tuning (light, as the brief asks):** we used a random search over the number of trees, the learning rate, the tree
depth and some regularization settings.

| Model | Settings tested | Selected settings | Tuning R² |
|---|---|---|---|
| CatBoost | 10 | depth 8, 1,500 trees, learning rate 0.09 | 0.911 |
| XGBoost | 20 | depth 8, 600 trees, learning rate 0.09 | 0.910 |
| LightGBM | 15 | 63 leaves, depth 8, 600 trees, learning rate 0.08 | 0.910 |

The three models have almost the same tuning score. CatBoost selected the maximum depth and the maximum number of
trees in the search. A wider search can possibly improve it.

## 3. Results on 2026-03

| Model | R² | MAE | Median % error | Mean % error | Within 10% of price |
|---|---|---|---|---|---|
| Linear regression (baseline) | 0.850 | $177.0K | 9.8% | 13.4% | 50.6% |
| Random forest (best model at that time) | 0.907 | $147.2K | 7.7% | 11.4% | 60.4% |
| LightGBM | 0.910 | $145.7K | 7.9% | 11.4% | 59.4% |
| CatBoost | 0.911 | $144.7K | 7.9% | 11.3% | 59.9% |
| Mean of CatBoost and random forest | 0.913 | $142.9K | 7.6% | 11.1% | 61.5% |
| **XGBoost** | **0.914** | **$142.4K** | 7.6% | 11.1% | 61.2% |
| **Kumo Tabular** | 0.908 | **$142.4K** | **7.2%** | **10.7%** | **62.7%** |

- **Gradient boosting:** all three gradient boosting models are better than the random forest. Only the XGBoost gain
  (+0.008 R², $4.8K less MAE) is clearly larger than the usual noise between months.
- **Speed:** XGBoost trains in approximately 1 second. The random forest needs 18 seconds.
- **Kumo Tabular:** it has the best value for all percentage metrics. Its R² is almost the same as the random forest.

**Why R² and percentage error do not agree:** expensive homes have the most effect on R² and on the dollar error. For
an expensive home, a 10% error is $300K or more. The percentage error gives the same weight to each home. Thus, it
shows the accuracy for a typical home.

## 4. Where each model is best

We put the 2026-03 homes into 10 price groups of equal size (deciles). Then we compared the median percentage error
in each group.

- **Kumo Tabular:** it is the most accurate model in 8 of the 10 groups.
  - Its largest gain is for the least expensive homes: 7.4% against 8.4% for the random forest.
  - It is weakest for the most expensive group: 12.2% against 11.8%.
- **CatBoost:** it is better than the random forest only in the top three groups (homes above approximately $1.3M).
  This is the source of its R² gain.

**Kumo Tabular also gives a price range.** We checked if the ranges are correct:

| Range | Homes with a sale price in the range | Target |
|---|---|---|
| 25% to 75% | 48% | 50% |
| 10% to 90% | 78% | 80% |
| 5% to 95% | 89% | 90% |

The ranges are well calibrated. A typical 10% to 90% range is 28% of the price, approximately $240K for an $850K
home. The other models do not give a range.

## 5. What has the most effect on the predictions (SHAP)

SHAP divides each prediction into one dollar value for each feature. The sum of these values is the prediction. We
used SHAP on CatBoost (the best gradient boosting model on the tuning month) for 2,000 homes from 2026-03.

| Feature | Mean effect on a prediction |
|---|---|
| Comparable-sales price | $226K |
| ZIP code (encoded) | $175K |
| Latitude, longitude | approximately $141K each |
| City (encoded) | $93K |
| Bathrooms | $74K |
| Lot size | $66K |

- **Comparable sales and location have the most effect.** Appraisers also use these to price homes.
- **Living area has a small effect ($30K)** only because the comparable-sales price already includes it. That feature
  is the price per square foot of the nearby homes, multiplied by the living area of the home.
- **School districts add little to ZIP code and city** (approximately $24K).
- **The waterfall plot in the notebook shows one home.** The model predicted $2.32M. The home sold for $2.47M.

## 6. Limits and next steps

**Limits**

- **One month only:** we did not do the usual 6-month check, to save time. Thus, these results are not decisions.
- **Fixed ensemble partner:** the ensemble used CatBoost, because CatBoost had the best tuning score. If we change it
  to XGBoost now, we select it with the validation month. That is not fair.
- **Kumo Tabular:**
  - It needs a GPU. It runs on the Mac GPU.
  - It needs logged and standardized inputs.
  - It must predict in batches of 500 homes. Batches of 2,000 homes stopped.
  - It used only 30K of the 189K training homes.

**Next steps**

1. Test XGBoost on all 6 validation months. This takes approximately 1 to 2 minutes. If XGBoost is better in at
   least 5 of 6 months, it replaces the random forest. (Done in notebook `07`.)
2. Calculate the mean of XGBoost and Kumo Tabular. They are strong on different homes (expensive and typical), so
   the mean can possibly be better than both.
3. Use the Kumo Tabular ranges, or a similar method, to give a price range with each estimate.

## 7. Possible questions from the team

**Why are there no test values? The brief asks for them.**
Our protocol does not use the test month until we select the final model. If we score it each week, we start to tune
to it. Then it is not a fair check.

**Is +0.008 R² a real improvement?**
Possibly. It is larger than the usual noise between months (approximately ±0.005), but it comes from one month. A
6-month test gives the answer. (Notebook `07` confirmed it.)

**How can a model with no training on our data compete?**
Kumo Tabular was pretrained on synthetic tables only. Thus, it learned how to learn from examples. It reads 30K of our
homes as examples. Then it predicts new homes by comparison, with no change to its weights.

**Does any data come from the future?**
No. Each model uses only the 18 months before the month that it predicts. The Kumo examples come from the same months.
Notebooks 05b and 05e checked that comparable sales use only sales that closed before the listing month.

**Can we explain the predictions?**
For the tree models, yes. SHAP gives the exact dollar effect of each feature. Kumo Tabular is a neural network. It
needs slower, approximate methods such as KernelSHAP or LIME. We did not use these methods yet.
