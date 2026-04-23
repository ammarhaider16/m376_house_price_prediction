from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error,
)
import pandas as pd

def compute_metrics(y_true, y_pred) -> dict:
    return {
        "rmse":      root_mean_squared_error(y_true, y_pred),
        "r2":        r2_score(y_true, y_pred),
        "mape":      mean_absolute_percentage_error(y_true, y_pred) * 100,
        "mean_ae":   mean_absolute_error(y_true, y_pred),
        "median_ae": median_absolute_error(y_true, y_pred),
    }

def main():
    metrics = []
    
    regression_train_results_df = pd.read_csv("regression/regression_y_outputs.csv")
    train_metrics = {"model_id": "linear_regression_train"}
    train_metrics.update(compute_metrics(regression_train_results_df["y_true"], regression_train_results_df["y_pred"]))
    metrics.append(train_metrics)

    regression_test_results_df = pd.read_csv("regression/regression_test_y_outputs.csv")
    test_metrics = {"model_id": "linear_regression_test"}
    test_metrics.update(compute_metrics(regression_test_results_df["y_true"], regression_test_results_df["y_pred"]))
    metrics.append(test_metrics)
    
    results_df = pd.DataFrame(metrics)
    results_df.to_csv("regression/metrics.csv", index=False)

if __name__ == "__main__":
    main()

