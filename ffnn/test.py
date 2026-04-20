import json
import pickle
import numpy as np
import pandas as pd
import os
import torch
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error,
)
from datetime import datetime

from ffnn.config import TrainConfig
from ffnn.transforms import apply_log1p_inputs, invert_target_transform, ZSCORE_FEATURES, TARGET_COL
from ffnn.model import FFNN

TEST_PATH      = "test.csv"
ARTIFACTS_PATH = "ffnn/artifacts"
RESULTS_PATH   = "ffnn/results"

def load_scaler(model_id: str):
    with open(f"{ARTIFACTS_PATH}/{model_id}_feature_scaler.pkl", "rb") as f:
        return pickle.load(f)

def load_target_params(model_id: str) -> dict:
    with open(f"{ARTIFACTS_PATH}/{model_id}_target_params.json") as f:
        return json.load(f)

def load_model(cfg: TrainConfig, model_id: str, checkpoint: str = "best") -> FFNN:
    model = FFNN(cfg.layers_fn())
    weights_path = f"{ARTIFACTS_PATH}/{model_id}_{checkpoint}.pt"
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()
    return model

def transform_inputs(df: pd.DataFrame, scaler) -> pd.DataFrame:
    df = apply_log1p_inputs(df)
    df[ZSCORE_FEATURES] = scaler.transform(df[ZSCORE_FEATURES])
    return df

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse":      root_mean_squared_error(y_true, y_pred),
        "r2":        r2_score(y_true, y_pred),
        "mape":      mean_absolute_percentage_error(y_true, y_pred) * 100,
        "mean_ae":   mean_absolute_error(y_true, y_pred),
        "median_ae": median_absolute_error(y_true, y_pred),
    }

def run_test(configs: list[TrainConfig], checkpoint: str = "best", save_predictions: bool = False):
    os.makedirs(RESULTS_PATH, exist_ok=True)
    df_test = pd.read_csv(TEST_PATH)
    y_true  = df_test[TARGET_COL].values.astype(np.float32)

    all_results = {}
    for cfg in configs:
        model_id = cfg.get_model_id()
        print(f"[{model_id}] running evaluation...")
        # load artifacts
        scaler        = load_scaler(model_id)
        target_params = load_target_params(model_id)
        model         = load_model(cfg, model_id, checkpoint)
        # transform inputs
        df_features = df_test.drop(columns=[TARGET_COL])
        df_features = transform_inputs(df_features, scaler)
        # run inference
        features_tensor = torch.tensor(df_features.values, dtype=torch.float32)
        with torch.no_grad():
            raw_outputs = model(features_tensor).squeeze(1).numpy()
        # invert target transform to get dollar predictions
        y_pred = invert_target_transform(raw_outputs, target_params["mean"], target_params["std"])
        # compute and store metrics
        metrics = compute_metrics(y_true, y_pred)
        all_results[model_id] = metrics
        print(f"  RMSE={metrics['rmse']:.2f} R2={metrics['r2']:.4f}  MAPE={metrics['mape']:.2f}% MAE={metrics['mean_ae']:.2f} MedianAE={metrics['median_ae']:.2f}")

        if save_predictions:
            preds_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
            preds_df.to_csv(f"{RESULTS_PATH}/{model_id}.csv", index=False)

    # save results
    results_df = pd.DataFrame.from_dict(all_results, orient="index")
    results_df.index.name = "model_id"
    results_df = results_df.sort_values("r2", ascending=False)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = f"{RESULTS_PATH}/test_results_{timestamp}.csv"
    results_df.to_csv(results_path)
    print(f"\nresults saved to {results_path}")

    return results_df
