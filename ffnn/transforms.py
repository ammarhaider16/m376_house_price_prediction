import numpy as np
import pandas as pd

LOG1P_FEATURES = [
    "households",
    "rooms_per_household",
    "population_per_household",
]

ZSCORE_FEATURES = [
    "households",
    "rooms_per_household",
    "population_per_household",
    "median_income",
    "longitude",
    "latitude",
    "housing_median_age",
    "bedrooms_per_room",
]

PASSTHROUGH_FEATURES = [
    "inland",
    "island",
    "near_bay",
    "near_ocean",
]

TARGET_COL = "median_house_value"

def apply_log1p_inputs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in LOG1P_FEATURES:
        df[col] = np.log1p(df[col])
    return df

def apply_log1p_target(series: pd.Series) -> pd.Series:
    return np.log1p(series)

def invert_target_transform(values: np.ndarray, target_mean: float, target_std: float) -> np.ndarray:
    """
    values:      np.array of model output values (in transformed space)
    target_mean: mean of log1p(target) computed on training set
    target_std:  std  of log1p(target) computed on training set
    """
    unscaled = values * target_std + target_mean
    return np.expm1(unscaled) # predicted house values in US dollars