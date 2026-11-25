# IDX Exchange data preprocessing (Dataset A)

Dataset A is the cleaned dataset that all team members use to compare modeling results. The code is in
`src/idx_exchange_data_science/cleaning.py` (fixed rules) and `preprocessing.py` (fitted steps).

**Status values:**

- **Discussed:** the team agreed on the rule during a Zoom meeting.
- **Pending:** the team must make a decision.
- **Experiment:** an optional modeling comparison.

**Important:** put the fixed cleaning rules and the training-derived ClosePrice filter in the cleaned CSV files. Fit
all imputation, clipping, scaling and encoding inside a pipeline. Fit them again in each cross-validation fold.

The rules come from the best practices document. The team comments are quoted as written.

## 1. General data cleaning and target

| Item | Rule | Status |
|---|---|---|
| State | Keep only the records with `StateOrProvince == "CA"`. | Discussed |
| Property type | Keep only the records with the types Residential and SingleFamilyResidence. | Discussed |
| Duplicates | Records with the same valid address and close date are duplicates. Keep the first record. Remove the records with a missing or invalid address. | Discussed |
| Duplicate listings | Before the address rule, keep one record for each `ListingKey`: the record with the latest CloseDate. Added on 2026-11-22: 233 records shared a `ListingKey` with a later record (overlapping monthly files). Without this rule, an earlier copy of a sale can be in the training data and the later copy in the validation data. | Added |
| ClosePrice validity | Remove the records with a missing ClosePrice or a ClosePrice of 0 or less. | Discussed |
| ClosePrice outliers | Calculate the 0.5th and 99.5th percentiles on the training data. Apply the same limits to the validation and test data. | Discussed |
| Date consistency | Remove the records with a CloseDate before the ListingContractDate. | Discussed |
| ClosePrice log target | Keep the original ClosePrice. During modeling, compare the original target with log1p(ClosePrice). Do not use either price column as an input. Convert log predictions back to dollars before scoring. | Experiment |
| High-missingness features | Remove the columns with more than 80% missing values (for example, BasementYN, BuildingAreaTotal and the school columns). Apply this fixed rule to the training, validation and test data. | Discussed |

Team comments:

- **Duplicates.**
  - Jehad: "Transactions with the same valid address and closing date are treated as duplicates, retaining the first
    occurrence. Missing or invalid addresses are excluded from duplicate matching."
  - Emily: "Same rule (same address + same CloseDate, missing addresses not matched), with the address normalized
    first (lowercase, trimmed, repeated spaces collapsed). Retained the last, but I can retain the first if needed."
  - Soonbee: "If two records have the same address and date but different prices, flag them for review instead of
    automatically merging."
  - Stephanie: "Transactions with the same valid address and closing date are treated as duplicates, retaining the
    first occurrence. Missing or invalid addresses are dropped"
- **ClosePrice log target.** Soonbee: "Reason to keep as an experiment - log price can reduce the effect of extreme
  luxury sales, but it may underpredict expensive properties after conversion back to dollars."
- **High-missingness features.**
  - Emily: "I don't use a missingness threshold. Instead I keep an explicit list of 38 columns and drop the rest (66
    dropped in train, 62 in val/test, because the monthly files have different columns)"
  - Stephanie: "Originally used 90%, but OK with switching to 80%"

## 2. Property characteristics

| Item | Rule | Status |
|---|---|---|
| LivingArea | Remove the records with a LivingArea of 0 or less. Fill missing values with the median. | Discussed |
| LivingArea transformation | For linear regression, compare the model with and without a log transformation. | Experiment |
| Bedroom and bathroom counts that do not agree with the living area | Select a range. Do not use this rule for Dataset A. You can use it for a Dataset B. | Not for Dataset A |
| BedroomsTotal | The valid range is 1 to 20. Set other values to missing. Fill missing values with the mode. | Discussed |
| BathroomsTotalInteger | The valid range is 1 to 20. Set other values to missing. Fill missing values with the mode. | Discussed |
| GarageSpaces | The valid range is 0 to 20. Set other values to missing. Fill missing values with the mode. | Discussed |
| ParkingTotal | The valid range is 0 to 20. Set other values to missing. Fill missing values with the mode. | Discussed |
| Stories | Remove this feature. | Discussed |
| YearBuilt, PropertyAge | Calculate PropertyAge from CloseDate.year and YearBuilt. Fill missing values with the median. | Discussed |

Team comments:

- **BedroomsTotal.**
  - Jehad: '"BedroomsTotal" range (1, 20) and implausible values are replaced with missing values and imputed with
    the mode'
  - Emily: "No fixed range yet, only the 70 sqft per bedroom check. Missing values use the train median (3) plus a
    BedroomsTotal_missing flag."
- **BathroomsTotalInteger.**
  - Jehad: "range (1, 20) and implausible values are replaced with missing values and imputed with the mode."
  - Emily: "No fixed range. More than 4 baths per bedroom is set to missing. Missing values use the train median (2)
    plus a BathroomsTotalInteger_missing flag: 52 train / 1 val / 0 test filled."
  - Soonbee: "Recommended status: Adopted/Discussed. Zero, negative, or above 20: convert to missing? For ordinary
    single-family properties, zero bathrooms most likely represents missing or incorrect reporting?"
- **GarageSpaces.**
  - Jehad: "range (0, 20) and implausible values are replaced with missing values and imputed with the mode."
  - Soonbee: "Recommended status: Adopted/Discussed. Valid range: 0–20,Add GarageSpaces_missing, Impute with the
    training median or mode (garage capacity may be a meaningful California housing amenity and is more specific
    than total parking.)"
- **ParkingTotal.**
  - Jehad: "range (0, 20) and implausible values are replaced with missing values and imputed with the mode."
  - Soonbee: "Recommended status: Experiment. ParkingTotal, GarageSpaces, and CoveredSpaces overlap. Including all of
    them can create redundancy, especially for linear regression. The upper bound of 100 is too high for ordinary
    single-family properties. Use a more defensible provisional range such as 0–20, then inspect the training
    distribution."
- **Stories.**
  - Jehad: "range (1, None) and implausible values are replaced with missing values and imputed with the mode."
  - Soonbee: "Recommended status: Experiment: (1,10), Zero, negative, and above 10: convert to missing, Compare with
    the categorical Levels field, stories can influence accessibility, layout, and land use, but Stories and Levels
    may describe nearly the same thing, so we should select the more reliable field."
- **YearBuilt, PropertyAge.**
  - Jehad: "I think adding PropertyAge is a good idea."
  - Emily: "Done as AgeAtSale = CloseDate.year − YearBuilt. Missing YearBuilt uses the train median (1976) plus a
    YearBuilt_missing fla"
  - Soonbee: "Recommended status: Adopted/Discussed. Reason: property age is more interpretable and remains
    meaningful as the calendar changes."

## 3. Lot size characteristics

| Item | Rule | Status |
|---|---|---|
| LotSizeSquareFeet | Set values of 0 or less to missing. If LotSizeAcres is available, use it to calculate a missing LotSizeSquareFeet (1 acre = 43,560 sq ft). | Discussed |
| LotSizeAcres | After you calculate the missing LotSizeSquareFeet values, remove LotSizeAcres. | Discussed |
| Missing lot size | After the calculation from LotSizeAcres, fill the remaining missing values with the median of LotSizeSquareFeet, inside the pipeline. | Discussed |
| Lot size log transformation | Compare the log-transformed lot size with the original values. | Experiment |

Team comments:

- **LotSizeSquareFeet.** Jehad: "Zero and negative values are treated as missing. Missing LotSizeSquareFeet values
  are recovered using LotSizeAcres if not missing(1 acre = 43,560 sq ft)."
- **Missing lot size.** Jehad: "imputed missing values after recovering records using LotSizeAcres, with median of
  LotSizeSquareFeet."

## 4. Geographic characteristics

| Item | Rule | Status |
|---|---|---|
| PostalCode format | Change the ZIP code to a five-digit string when possible. | Discussed |
| Latitude, longitude validity | Set coordinates outside the approximate California limits to missing. The limits are latitude 32.5 to 42.1 and longitude −124.5 to −114.0. | Discussed |
| Coordinate imputation | Fill missing coordinates with the median coordinates of the same PostalCode. Fit these medians on the training fold. | Discussed |
| Coordinate fallback | If the ZIP code median is not available, use the county median. Fill the remaining missing coordinates with the overall median latitude and longitude. | Discussed |

Team comments:

- **PostalCode format.** Jehad: "standardized to a five-digit string format"
- **Latitude, longitude validity.** Jehad: "Proposed bounds: lat 32.5–42.1, lon −124.5 to −114.0."
- **Coordinate imputation and fallback.** Jehad: "I agree with this approach"

## 5. Binary features

| Item | Rule | Status |
|---|---|---|
| AttachedGarageYN, FireplaceYN, NewConstructionYN, PoolPrivateYN, ViewYN | For each binary feature, add a missing-value indicator (1 = missing, 0 = not missing). Then fill the missing values with the mode of the training data. Apply the same training mode to the validation and test data. | Discussed |

A missing value does not always mean False. Confirm this convention for each field.

## 6. Categorical characteristics

| Item | Rule | Status |
|---|---|---|
| City | Replace missing values with "Unknown". Use target encoding with 5-fold cross-fitting. | Discussed |
| CountyOrParish | Replace missing values with "Unknown". Use target encoding with 5-fold cross-fitting. | Discussed |
| PostalCode | Replace missing values with "Unknown". Use target encoding with 5-fold cross-fitting. | Discussed |
| Flooring | Do not use this feature in Dataset A. | Not for Dataset A |
| Other features | Add other features here. | Pending |

Team comments:

- **City.** Jehad: 'standardized to a five-digit string format and replaced missing values with "Unknown" and then I
  used target encoding with 5-fold cross fitting.'
- **CountyOrParish and PostalCode.** Jehad: 'replaced missing values with "Unknown" and then I used target encoding
  with 5-fold cross fitting.'

**Note:** Flooring is not part of Dataset A. `features.build_table()` adds flooring and HOA as feature groups
after the Dataset A cleaning. Notebook `07` tests these groups.
