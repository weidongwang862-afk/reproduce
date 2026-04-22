import numpy as np


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    eps = 1e-8
    mape = float(np.mean(np.abs((y_pred - y_true) / np.maximum(np.abs(y_true), eps))))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {
        "MAE": mae,
        "MAPE": mape,
        "RMSE": rmse,
        "R2": r2,
    }