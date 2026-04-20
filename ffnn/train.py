import json
import pickle
import numpy as np
import pandas as pd
import os
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ffnn.transforms import apply_log1p_inputs, apply_log1p_target, ZSCORE_FEATURES, TARGET_COL
from ffnn.config import TrainConfig
from ffnn.model import FFNN

TRAIN_PATH    = "train.csv"
ARTIFACTS_PATH = "ffnn/artifacts"

def get_dataloader(df: pd.DataFrame, batch_size: int, shuffle: bool) -> DataLoader:
    features = torch.tensor(df.drop(TARGET_COL, axis=1).values, dtype=torch.float32)
    targets = torch.tensor(df[TARGET_COL].values, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(features, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def fit_and_transform_inputs(df_train: pd.DataFrame, model_id: str) -> tuple[StandardScaler, pd.DataFrame]:
    df = apply_log1p_inputs(df_train)
    scaler = StandardScaler()
    df[ZSCORE_FEATURES] = scaler.fit_transform(df[ZSCORE_FEATURES])
    with open(f"{ARTIFACTS_PATH}/{model_id}_feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    return scaler, df

def apply_transform_inputs(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    df = apply_log1p_inputs(df)
    df[ZSCORE_FEATURES] = scaler.transform(df[ZSCORE_FEATURES])
    return df

def fit_and_transform_target(series_train: pd.Series, model_id: str) -> tuple[dict, pd.Series]:
    log_target = apply_log1p_target(series_train)
    target_mean = float(log_target.mean())
    target_std  = float(log_target.std())
    series_transformed = (log_target - target_mean) / target_std
    target_params = {"mean": target_mean, "std": target_std}
    with open(f"{ARTIFACTS_PATH}/{model_id}_target_params.json", "w") as f:
        json.dump(target_params, f)
    return target_params, series_transformed

def apply_transform_target(series: pd.Series, target_params: dict) -> pd.Series:
    log_target = apply_log1p_target(series)
    return (log_target - target_params["mean"]) / target_params["std"]

def run_train(cfg: TrainConfig):
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)
    model_id = cfg.get_model_id()

    # train/val split
    df = pd.read_csv(TRAIN_PATH)
    df_train, df_val = train_test_split(df, test_size=cfg.val_size, random_state=42)

    # fit and apply transforms
    scaler, df_train = fit_and_transform_inputs(df_train, model_id)
    target_params, series_transformed = fit_and_transform_target(df_train[TARGET_COL], model_id)
    df_train[TARGET_COL] = series_transformed
    df_val = apply_transform_inputs(df_val, scaler)
    df_val[TARGET_COL] = apply_transform_target(df_val[TARGET_COL], target_params)

    # dataloaders
    train_loader = get_dataloader(df_train, cfg.batch_size, shuffle=True)
    val_loader   = get_dataloader(df_val,   cfg.batch_size, shuffle=False)

    # model, loss, optimizer
    model     = FFNN(cfg.layers_fn())
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_loss = float("inf")
    for epoch in range(cfg.num_epochs):
        # training
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"[{cfg.model_name}] Epoch {epoch+1}/{cfg.num_epochs}", leave=True)
        for features, targets in progress_bar:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_postfix(batch_loss=f"{loss.item():.4f}")
        train_loss /= len(train_loader)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                outputs = model(features)
                val_loss += criterion(outputs, targets).item()
        val_loss /= len(val_loader)
        print(f"[{cfg.model_name}] Epoch {epoch+1:03d}/{cfg.num_epochs} | train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt = f"{model_id}_best.pt"
            torch.save(model.state_dict(), f"{ARTIFACTS_PATH}/{best_ckpt}")
            print(f"  --> best checkpoint updated ({best_ckpt})")

    # save final weights
    final_ckpt = f"{model_id}_final.pt"
    torch.save(model.state_dict(), f"{ARTIFACTS_PATH}/{final_ckpt}")
    print(f"[{cfg.model_name}] final weights saved ({final_ckpt})")
    print(f"[{cfg.model_name}] training complete. best val_loss={best_val_loss:.6f}")

    # save config
    config_path = f"{ARTIFACTS_PATH}/{model_id}_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "model_name":   cfg.model_name,
            "lr":           cfg.lr,
            "weight_decay": cfg.weight_decay,
            "batch_size":   cfg.batch_size,
            "num_epochs":   cfg.num_epochs,
            "val_size":     cfg.val_size,
            "best_val_loss": best_val_loss,
        }, f, indent=2)

