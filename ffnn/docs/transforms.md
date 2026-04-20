# Transforms

### log1p + z-score

Applied to features that are right-skewed, where raw values would create an uneven gradient landscape and cause the optimizer to weight large-magnitude samples disproportionately.

**`households`** — A raw block-level count variable. Count distributions in census-style data are characteristically right-skewed, with a small number of very large blocks pulling the mean upward. Log compression brings the distribution closer to symmetric before standardization.

**`rooms_per_household`** — An engineered ratio, but with outlier exposure on the high end (range roughly 3–10+ in typical blocks). The ratio transformation already tames the raw count skew substantially, but log1p provides additional robustness against high-end outlier blocks.

**`population_per_household`** — Similar reasoning to rooms_per_household. Generally well-behaved for residential blocks, but institutional blocks (dormitories, care facilities) could push this value high. Log1p compresses those outliers without discarding the observations.

**`median_house_value` (target)** — MSE in log space penalizes relative errors rather than absolute ones, which is more appropriate for a price target spanning a wide range.

### z-score only

Applied to features that are already bounded or have compact, near-symmetric distributions where log compression would either be unnecessary or actively harmful.

**`longitude`, `latitude`** — Bounded geographic coordinates spanning California's extent. Roughly uniform distribution. No compression needed; z-score alone establishes a consistent scale.

**`housing_median_age`** — Bounded between 1 and 52 (hard cap in this dataset). Distribution is near-uniform. Z-score sufficient.

**`median_income`** — Already expressed in units of $10,000, which compresses the natural income skew considerably (training set mean ~3.87, std ~1.90). Residual skew is moderate. Z-score directly is the default recommendation; a log1p pass is defensible and should be validated empirically if pursued.

**`bedrooms_per_room`** — A proportion bounded in (0, 1) by construction. A log transform is inappropriate here: values near zero would be mapped to large negative numbers, distorting the distribution rather than normalizing it. Z-score on the raw proportion is correct.

### No transform

**`inland`, `island`, `near_bay`, `near_ocean`** — Binary OHE dummies. Scaling binary indicators offers no benefit to gradient-based optimization and distorts their interpretation. These pass through unchanged in both scripts.

### Serialization

**Feature scaler** — The fitted StandardScaler must be saved at the end of the training script and loaded at the start of the test script. The test script never recomputes these statistics from test data.

**Target transform parameters** — The training target mean and standard deviation (computed after the log1p pass on `median_house_value`) must be saved separately. These are required to invert the output transform at inference time.

### Inversion requirement

The model is trained to predict in transformed space (log1p then z-scored). At test time, the raw model output must be inverted through both transforms in reverse order before computing any interpretable error metrics:

1. Undo z-score: multiply by training target std, add training target mean
2. Undo log1p: exponentiate, subtract one
