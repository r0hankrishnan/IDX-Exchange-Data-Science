# Week 7 brief: advanced models (notebook 06)

**Bottom line:** Tuned XGBoost is the strongest model on R² and dollar error: 0.914 vs 0.907 for our current best, the random forest. NVIDIA's Kumo Tabular was never trained on our data, yet it has the lowest percentage error. Both results come from one validation month, so neither replaces the forest yet.

## 1. The question and how we tested it

Can a stronger model beat our current best? That is a **random forest using Dataset A plus school districts and comparable sales** (notebook 05e).

- **Same inputs for every model:** the same features and the same preprocessing. Imputation and target encoding are refit on each training window, and every model trains on the 18 months before the month it predicts.
- **The test month (2026-04) is never touched.** We compare models only on validation data, so the test month stays an honest final check.
- **Two separate months, to keep the run short:**
  - **Tuning month, 2025-09.** Each boosted model's settings were chosen by R² on this month.
  - **Comparison month, 2026-03.** All models are scored here. Tuning never saw it, so tuned models get no unfair edge.
- **Noise level:** R² normally moves about ±0.005 from month to month. Differences smaller than that are not conclusive.

## 2. The models

| Model | What it is, in one line |
|---|---|
| Linear regression | Baseline from week 4. One weight per feature. |
| Random forest | Current best. Hundreds of deep trees averaged together. |
| LightGBM, XGBoost, CatBoost | Gradient boosting. Small trees are added one at a time, each correcting the errors of the trees before it. These are the three main libraries. |
| Kumo Tabular (NVIDIA) | A pretrained "foundation model" for tables. It is not trained on our data. We show it 30,000 example homes with their prices, and it predicts new homes from those examples in one pass. |
| Ensemble | The average of the forest and the best boosted model on the tuning month, which was CatBoost. |

**Tuning (light, as the brief asked):** a random search over the number of trees, learning rate and tree depth, plus a few regularization settings.

| Model | Settings tried | Chosen settings | Tuning R² |
|---|---|---|---|
| CatBoost | 10 | depth 8, 1,500 trees, learning rate 0.09 | 0.911 |
| XGBoost | 20 | depth 8, 600 trees, learning rate 0.09 | 0.910 |
| LightGBM | 15 | 63 leaves, depth 8, 600 trees, learning rate 0.08 | 0.910 |

All three tuned to almost the same score. CatBoost chose the largest depth and tree count allowed, so a wider search might help it.

## 3. Results on 2026-03

| Model | R² | Mean $ error (MAE) | Median % error | Mean % error | Within 10% of price |
|---|---|---|---|---|---|
| Linear regression (baseline) | 0.850 | $177.0K | 9.8% | 13.4% | 50.6% |
| Random forest (current best) | 0.907 | $147.2K | 7.7% | 11.4% | 60.4% |
| LightGBM | 0.910 | $145.7K | 7.9% | 11.4% | 59.4% |
| CatBoost | 0.911 | $144.7K | 7.9% | 11.3% | 59.9% |
| CatBoost + forest average | 0.913 | $142.9K | 7.6% | 11.1% | 61.5% |
| **XGBoost** | **0.914** | **$142.4K** | 7.6% | 11.1% | 61.2% |
| **Kumo Tabular** | 0.908 | **$142.4K** | **7.2%** | **10.7%** | **62.7%** |

- **All three boosted models beat the forest.** Only XGBoost's gain (+0.008 R², $4.8K lower MAE) is clearly larger than normal month-to-month noise.
- **XGBoost is also fast.** It trains in about 1 second, compared with the forest's 18.
- **Kumo Tabular is best on every percentage measure.** Its R² only matches the forest's.

**Why R² and % error disagree.** R² and dollar error are dominated by expensive homes, where a 10% miss is $300K or more. Percentage error treats every home equally, so it reflects the typical home.

## 4. Where each model wins

We split the 2026-03 homes into 10 equal-sized price groups (deciles) and compared median % error within each group.

- **Kumo Tabular is most accurate in 8 of the 10 groups.** Its biggest gains are on the cheapest homes: 7.4% vs 8.4% for the forest. It is weakest on the most expensive group: 12.2% vs 11.8%.
- **CatBoost beats the forest only in the top three groups** (homes above about $1.3M). That is where its R² gain comes from.

**Kumo Tabular also gives a price range, not just a number.** We checked whether the ranges hold up:

| Range | Homes that sold inside it | Target |
|---|---|---|
| 25%–75% | 48% | 50% |
| 10%–90% | 78% | 80% |
| 5%–95% | 89% | 90% |

The ranges are well calibrated. A typical 10%–90% range is 28% of the price wide, about $240K for an $850K home. None of the other models give a range.

## 5. What drives the predictions (SHAP)

SHAP splits each prediction into dollar contributions, one per feature, that add up to the final price. We ran it on CatBoost (the best boosted model on the tuning month) for 2,000 homes from 2026-03.

| Feature | Average effect on a prediction |
|---|---|
| Comparable-sales implied price | $226K |
| ZIP code (encoded) | $175K |
| Latitude / longitude | about $141K each |
| City (encoded) | $93K |
| Bathrooms | $74K |
| Lot size | $66K |

- **Comparable sales and location drive the model.** This matches how appraisers price homes.
- **Living area looks small ($30K) only because it is already built into the comparable-sales price.** That feature is the neighbours' price per square foot times this home's living area.
- **School districts add little on top of ZIP and city** (about $24K).
- **The waterfall plot in the notebook walks through one home.** The model predicted $2.32M; the home sold for $2.47M.

## 6. Caveats and next steps

**Caveats**
- **One month only.** The usual 6-month rolling check was skipped to save compute, so these are promising results, not decisions.
- **The ensemble partner was fixed in advance.** It used CatBoost because CatBoost won the tuning month. Swapping in XGBoost now would mean choosing based on the validation month.
- **Kumo Tabular needs a GPU and some care to run.** It runs on the Mac's GPU, needs log-scaled and standardized inputs, and has to predict in batches of 500 (2,000 stalled). It also saw only 30K of the 189K training homes.

**Next steps**
1. Run XGBoost on the full 6-month rolling validation. It takes about 1–2 minutes. If it wins in at least 5 of 6 months, it replaces the forest.
2. Average XGBoost and Kumo Tabular. They are strong on different homes (expensive vs typical), so the average may beat both.
3. Use Kumo Tabular's ranges, or a similar method, to report price ranges with the estimate.

## 7. Questions the team may ask

**Why no test-set numbers? The brief asks for them.**
Our protocol keeps the test month untouched until the final model is chosen. If we scored it every week, we would end up tuning to it, and it would stop being an honest check.

**Is +0.008 R² a real improvement?**
It might be. It is above normal month-to-month noise (about ±0.005), but it comes from one month. The 6-month rerun settles it.

**How can a model that was never trained on our data compete?**
Kumo Tabular was pretrained only on synthetic tables, so it learned how to learn from examples. It reads 30K of our homes as examples and predicts by analogy, without updating its weights.

**Is anything leaking from the future?**
No. Every model sees only the 18 months before the month it predicts. Kumo's examples come from those same months. Earlier notebooks (05b, 05e) checked that comparable sales use only sales that closed before the listing month.

**Can we explain the predictions?**
Yes for the tree models: SHAP gives exact per-feature dollar contributions. Kumo Tabular is a neural network, so it would need slower, approximate methods such as KernelSHAP or LIME. We haven't run those yet.
