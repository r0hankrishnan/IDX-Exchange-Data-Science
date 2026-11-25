# Final presentation: modeling approach (draft)

## Plan

- **Length:** 6 slides, approximately 6 minutes. Remove slide 6 or merge it into slide 5 if the time is shorter.
- **Main message:** we predict the price before listing, with a fair time-based test, and comparable sales are the
  feature that matters most. The best model has a median error of approximately 7%.
- **Before this section:** a teammate covers the data and the preprocessing. Do not explain the cleaning again.
- **After this section:** the app demo, or the conclusion.
- **Numbers to remember:** 7.02% (Model C, served in the app), 7.21% (the model before it), 7.32% (that model on the
  test month), 18 months of training, 6 validation months.

## Slides and script

### Slide 1: The question and the rules

**On the slide**

- Title: "Predicting the price before the listing"
- Three short lines: "Only data known before listing" | "Train on the past, test on the future" | "Judged by the
  error for a typical home (MdAPE)"
- Visual: one house icon with a calendar. The listing date is marked. Everything after it is grey.

**Script**

> Our goal was to estimate what a home will sell for before it goes on the market. That sets two rules for the model.
> First, it can use only information that exists before the listing date, so no list price and no sale details.
> Second, we test it the way it would be used: we train on past months and predict a later month. We judge the models
> by the median percentage error, because it tells you how far off we are for a typical home. R² gives the most weight
> to the most expensive homes, and we did not want a few $10 million sales to decide our model.

### Slide 2: How we tested the models

**On the slide**

- Visual: a timeline from January 2024 to April 2026. Six rows, one for each validation month (October 2025 to March
  2026). In each row, a blue bar covers the 18 training months and an orange box marks the predicted month. April 2026
  is a separate grey box labeled "Test month, used one time".
- One line under it: "A change is accepted only if the error is lower on average and in at least 5 of the 6 months."

**Script**

> Every model went through the same test. For each of six months, from October 2025 to March 2026, we trained on the
> 18 months before it and predicted that month. All the fitted steps, like filling missing values and encoding ZIP
> codes, learn only from those training months. To accept a change, it had to lower the average error and also win in
> at least five of the six months, so one lucky month cannot carry it. We kept April 2026 locked away and used it one
> time, at the end, as the final check.

### Slide 3: From a baseline to gradient boosting

**On the slide**

- Visual: a horizontal bar chart of validation MdAPE, from top to bottom: linear regression 11.96%, decision tree
  9.39%, random forest 7.85%, XGBoost (log price, + HOA and flooring) 7.21%, Model C 7.02%. Lower is better. The bars
  for the last two are a different color.
- Footnote: "The linear regression and decision tree values come from the earlier data version."

**Script**

> We started with a linear regression on log price as a baseline. Its median error was about 12%. A single decision
> tree cut that to about 9.4%, and a random forest to 7.9%, because house prices depend on combinations of features,
> like size in a specific neighborhood, that a linear model cannot capture. In week 6, we tried gradient boosting
> models and a pretrained tabular model on one month. XGBoost was the strongest, and on all six months it beat the
> random forest every month. Two more changes helped: training on the log of the price, which stops expensive homes
> from dominating the fit, and adding HOA fees and flooring. That gave our served model, at 7.2%.

### Slide 4: Comparable sales, the most important feature

**On the slide**

- Visual: the comparable-sales map from the app (one home and its 10 nearest earlier sales).
- Three short lines: "10 nearest sales that closed in the 12 months before listing" | "Median price per sq ft x living
  area" | "No sale from the same month or later"

**Script**

> The feature that matters most is what an appraiser would use: comparable sales. For each home, we take the ten
> nearest sales that closed in the twelve months before its listing month. From them, we take the median price per
> square foot and multiply it by the home's living area. Because we only use sales that closed before the listing
> month, the home's own sale and anything after it can never leak in, and we have a unit test for that rule. In our
> SHAP analysis, the comparable-sales price and the location features had the largest effect on the predictions.

### Slide 5: Final model and results

**On the slide**

- Visual: the line chart of MdAPE by month for the current model and Model C (from the modelling update report).
- Small table: earlier model 7.21% validation, 7.32% test, R² 0.913. Model C (in the app) 7.02% validation, better
  in 6 of 6 months.

**Script**

> Our last improvement was to add one more number from the same ten sales: their median sale price. It does not depend
> on square footage, so it still works when the living area of a home or its neighbors is wrong in the listing. We call
> this Model C, and it lowered the error from 7.2% to 7.0%, in every one of the six months, so it is the model in our
> app. On the April test month, the model before it had a median error of 7.3%, almost the same as in validation.
> That tells us the validation results were not too optimistic. In practice, half of the homes are priced within about 7% of their sale price, and about
> 63% are within 10%.

### Slide 6: Where the model is weak, and next steps

**On the slide**

- Two columns.
  - "Weak spots": "Expensive homes: 10% median error in the top fifth" | "Few nearby sales: 14% error when the
    comparables are more than 5 km away" | "Condition is not in the data"
  - "Next steps": "Price ranges by comparable distance" | "Market trend features" | "Tune the settings for MdAPE"

**Script**

> The model is weakest in two places. For the most expensive fifth of homes, the median error is about 10%. And when
> the nearest comparable sales are more than five kilometers away, the error doubles to about 14%. When we looked at
> the worst errors, most were homes that sold far below their neighbors, which usually means condition, and condition
> is not in the data before listing. Next, we would make the price ranges wider for homes with distant comparables,
> test the team's market trend features, and retune the XGBoost settings for this error metric, since we tuned them
> for R².

## Questions to prepare for

**Why not use the test month to choose between the models?**
If we use it to choose, it stops being a fair check. We used it one time, for the model that we selected first.

**Is there any data leakage?**
Each model trains only on months before the month it predicts. The comparable sales use only sales that closed
before the listing month. The cleaning keeps one row for each listing, so the same sale cannot be in training and in
validation.

**Why not the average of three models? It had a lower error (6.88%).**
It needs three models and two sets of comparable sales for each home. The members were also picked on the same months
that scored it, so 6.88% is slightly optimistic.

**How did Model C do on the test month?** (backup, only if asked)
7.17%, against 7.32% for the served model. This is a second look at the test month, because we built Model C after
the first test score, so we do not use it to make the decision.

**Why MdAPE and not R²?**
R² is mostly decided by the most expensive homes. A buyer or seller of a typical home cares about the percentage
error for a home like theirs.
