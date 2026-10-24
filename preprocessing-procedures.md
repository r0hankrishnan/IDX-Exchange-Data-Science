IDX Exchange Date Preprocessing 
Dataset A is the cleaned dataset that every team member uses to compare modeling results.
Discussed—>reported as agreed during Zoom
Pending—>decision required
Experiment—>optional modeling comparison.
Important: Fixed cleaning rules and training-derived ClosePrice filtering belong in the cleaned CSVs. Feature-derived imputation, clipping, scaling and encoding must be fitted inside a pipeline, including within each cross-validation fold.
From best practices doc:
 
1. General data cleaning and target
Feature 
Cleaning or Preprocessing Rule
Status
Notes / Team Comments
State Filtering 
Keep only records where StateOrProvince == ‘CA’.
Discussed


Property type
Only records classified as Residential and SingleFamilyResidence are retained.
Discussed


Duplicates
Remove duplicate sale records using an agreed identification rule.
Discussed

(Transactions with the same valid address and closing date are treated as duplicates, retaining the first occurrence. Missing or invalid addresses are dropped)


Jehad: Transactions with the same valid address and closing date are treated as duplicates, retaining the first occurrence. Missing or invalid addresses are excluded from duplicate matching.
 Emily: Same rule (same address + same CloseDate, missing addresses not matched), with the address normalized first (lowercase, trimmed, repeated spaces collapsed). Retained the last, but I can retain the first if needed.
Soonbee: If two records have the same address and date but different prices, flag them for review instead of automatically merging.
Stephanie: Transactions with the same valid address and closing date are treated as duplicates, retaining the first occurrence. Missing or invalid addresses are dropped
ClosePrice validity
Remove records with missing ClosePrice or ClosePrice ≤ 0.
Discussed


ClosePrice outliers
Use training-derived 0.5th and 99.5th percentile thresholds. 
Apply frozen limits to validation/test.
Discussed


Date consistency
Remove records with CloseDate earlier than ListingContractDate.
Discussed


ClosePrice log target
Keep original ClosePrice, compare original target with log1p(ClosePrice) during modeling.
Experiment
Exclude both price columns from input X.
Inverse-transform log predictions before dollar-scale.
Soonbee: Reason to keep as an experiment - log price can reduce the effect of extreme luxury sales, but it may underpredict expensive properties after conversion back to dollars.


High-missingness features
Exclude BasementYN and BuildingAreaTotal and school-related columns have (80-100%missing values)so drop cols with > 80% missingness. Apply this fixed feature exclusion to training, validation, and test set
Discussed 
 (drop cols with > 80% missing values.)
Emily: I don't use a missingness threshold. Instead I keep an explicit list of 38 columns and drop the rest (66 dropped in train, 62 in val/test, because the monthly files have different columns)
Stephanie: Originally used 90%, but OK with switching to 80%












2. Property characteristics
Feature 
Cleaning or Preprocessing Rule
Status
Notes / Team Comments
LivingArea
Records with zero or negative LivingArea are removed. Use median imputation for missing values.
Discussed 


LivingArea transformation
Compare with/without log transformation for Linear Regression.
Experiment


bedroom/bathroom counts inconsistent with living area 
Choose range 
Not for our
Dataset A . You can apply this to Dataset B if you have one.


BedroomsTotal
Agree on plausible range and whether missing values use median or mode.
Discussed 

Range (1,20) and implausible values are replaced with missing values and imputed with the mode
 
Jehad: "BedroomsTotal" range (1, 20) and implausible values are replaced with missing values and imputed with the mode
Emily: No fixed range yet, only the 70 sqft per bedroom check. Missing values use the train median (3) plus a BedroomsTotal_missing flag.
BathroomsTotalInteger
Agree on plausible range, including whether minimum of 1 is justified and whether missing values use median or mode.
Discussed

(Range (1, 20) and implausible values are replaced with missing values and imputed with the mode.


Jehad: range (1, 20) and implausible values are replaced with missing values and imputed with the mode.
Emily: No fixed range. More than 4 baths per bedroom is set to missing. Missing values use the train median (2) plus a BathroomsTotalInteger_missing flag: 52 train / 1 val / 0 test filled.
Soonbee: Recommended status: Adopted/Discussed. Zero, negative, or above 20: convert to missing? For ordinary single-family properties, zero bathrooms most likely represents missing or incorrect reporting?
GarageSpaces 
Agree on validity rules and missing-value strategy.
Discussed

Range (0, 20) and implausible values are replaced with missing values and imputed with the mode.
Jehad:  range (0, 20) and implausible values are replaced with missing values and imputed with the mode.
Soonbee:Recommended status: Adopted/Discussed. Valid range: 0–20,Add GarageSpaces_missing, Impute with the training median or mode (garage capacity may be a meaningful California housing amenity and is more specific than total parking.)
ParkingTotal
Agree on validity rules and missing-value strategy.
Discussed

Range (0, 20) and implausible values are replaced with missing values and imputed with the mode.


Jehad: range (0, 20) and implausible values are replaced with missing values and imputed with the mode.
Soonbee: Recommended status: Experiment. ParkingTotal, GarageSpaces, and CoveredSpaces overlap. Including all of them can create redundancy, especially for linear regression. The upper bound of 100 is too high for ordinary single-family properties. Use a more defensible provisional range such as 0–20, then inspect the training distribution.


Stories 
Agree on validity rules and missing-value strategy.
Remove this feature 
 Jehad: range (1, None) and implausible values are replaced with missing values and imputed with the mode.
Soonbee: Recommended status: Experiment: (1,10), Zero, negative, and above 10: convert to missing, Zero, negative, and above 10: convert to missing, Compare with the categorical Levels field, stories can influence accessibility, layout, and land use, but Stories and Levels may describe nearly the same thing, so we should select the more reliable field.
YearBuilt / PropertyAge
Create a new feature using YearBuilt and CloseDate.year and for missing values use median 
Create a new feature using YearBuilt and CloseDate.year and for missing values use median 
Jehad: I think adding PropertyAge is a good idea.
Emily: Done as AgeAtSale = CloseDate.year − YearBuilt. Missing YearBuilt uses the train median (1976) plus a YearBuilt_missing fla
Soonbee: Recommended status: Adopted/Discussed. Reason: property age is more interpretable and remains meaningful as the calendar changes.

 
3. Lot size characteristics

Feature 
Cleaning or Preprocessing Rule
Status
Notes / Team Comments
LotSizeSquareFeet
Compare existing plausibility and cleaning approaches.
Zero and negative values are treated as missing. Missing LotSizeSquareFeet values are recovered using LotSizeAcres  if not missing(1 acre = 43,560 sq ft).


Jehad: Zero and negative values are treated as missing. Missing LotSizeSquareFeet values are recovered using LotSizeAcres  if not missing(1 acre = 43,560 sq ft).
LotSizeAcres
Decide whether to retain both acres and square feet or derive one from the other.
Drop this after recovering missing LotSizeSquareFeet vales


Missing lot size
Proposed median imputation inside pipeline.
imputed missing values after recovering records with median of LotSizeSquareFeet.


Jehad: imputed missing values after recovering records using LotSizeAcres,  with median of LotSizeSquareFeet.


Lot size log transform
Optionally compare log-transformed lot size with original.
Experiment
 

 
4. Geographic characteristics
Feature 
Cleaning or Preprocessing Rule
Status
Notes / Team Comments
PostalCode formatting
Standardize ZIP code to a consistent five-digit string when possible.
Discussed
Jehad: standardized to a five-digit string format 
Latitude / Longitude validity
Mark invalid coordinates outside agreed approximate California bounds as missing.
Discussed 
Jehad: Proposed bounds: lat 32.5–42.1, lon −124.5 to −114.0.
Coordinate imputation
Proposed median coordinate lookup by PostalCode, fitted on training fold.
Discussed 
Jehad: I agree with this approach
Coordinate fallback
Proposed county-level median coordinates when ZIP lookup unavailable and any remaining missing coordinates will be filled using overall medians of lat and lon.
Discussed 
Jehad: I agree with this approach





5. Binary features
Feature 
Cleaning or Preprocessing Rule
Status
Notes / Team Comments
AttachedGarageYN, FireplaceYN, NewConstructionYN, PoolPrivateYN, ViewYN
For each binary feature, add a separate missingness indicator (1 = missing, 0 = non-missing), then fill missing values with the mode calculated from the training data. Apply the same training-derived mode to validation and test.
Discussed
Missing does not necessarily mean False. Confirm this convention for every field.

 


6. Categorical characteristics
Feature 
Cleaning or Preprocessing Rule
Status
Notes / Team Comments
City 
Choose a consistent categorical encoding strategy.
Target encoding
Jehad: standardized to a five-digit string format and replaced missing values with "Unknown" and then I used target encoding with 5-fold cross fitting.
CountyOrParish


Target encoding 
Jehad: replaced missing values with "Unknown" and then I used target encoding with 5-fold cross fitting.
PostalCode


Target encoding 
Jehad: replaced missing values with "Unknown" and then I used target encoding with 5-fold cross fitting.
Flooring
Decide whether to use this feature
Not for Dataset A


Other features???
Please add them here
Pending
 


