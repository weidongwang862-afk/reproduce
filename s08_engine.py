from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from s08_dataset import SequenceDataset
from s08_metrics import calc_metrics
from s08_utils import ensure_dir, save_json


def build_loss(loss_name: str):
    loss_name = loss_name.lower()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "l1":
        return nn.L1Loss()
    raise ValueError(f"不支持的 loss_name: {loss_name}")


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for xb, _ in loader:
        xb = xb.to(device)
        pred = model(xb).cpu().numpy()
        preds.append(pred)
    return np.concatenate(preds, axis=0)


def evaluate_split(model, X, y, batch_size, device):
    ds = SequenceDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    pred = predict(model, loader, device)
    metrics = calc_metrics(y, pred)
    return metrics, pred


def train_one_model(
    model,
    model_name: str,
    data_dict: Dict[str, Any],
    cfg,
    device,
    out_dir: Path,
) -> Tuple[dict, pd.DataFrame]:
    ensure_dir(out_dir)

    train_ds = SequenceDataset(data_dict["X_train"], data_dict["y_train"])
    val_ds = SequenceDataset(data_dict["X_val"], data_dict["y_val"])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    criterion = build_loss(cfg.loss_name)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    model = model.to(device)

    best_val_mae = float("inf")
    best_state = None
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_train = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(xb)
            n_train += len(xb)

        train_loss = train_loss_sum / max(n_train, 1)

        val_metrics, _ = evaluate_split(
            model, data_dict["X_val"], data_dict["y_val"], cfg.batch_size, device
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_MAE": val_metrics["MAE"],
            "val_MAPE": val_metrics["MAPE"],
            "val_RMSE": val_metrics["RMSE"],
            "val_R2": val_metrics["R2"],
        })

        if val_metrics["MAE"] < best_val_mae:
            best_val_mae = val_metrics["MAE"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 50 == 0:
            print(
                f"[{model_name}] Epoch {epoch:4d} | "
                f"train_loss={train_loss:.6f} | "
                f"val_MAE={val_metrics['MAE']:.6f} | "
                f"val_RMSE={val_metrics['RMSE']:.6f} | "
                f"val_R2={val_metrics['R2']:.6f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    val_result, val_pred = evaluate_split(
        model, data_dict["X_val"], data_dict["y_val"], cfg.batch_size, device
    )
    b6_result, b6_pred = evaluate_split(
        model, data_dict["X_test_B0006"], data_dict["y_test_B0006"], cfg.batch_size, device
    )
    b7_result, b7_pred = evaluate_split(
        model, data_dict["X_test_B0007"], data_dict["y_test_B0007"], cfg.batch_size, device
    )

    torch.save(model.state_dict(), out_dir / f"{model_name}_best.pt")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({
        "y_true": data_dict["y_val"].reshape(-1),
        "y_pred": val_pred.reshape(-1),
    }).to_csv(out_dir / "pred_val.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({
        "y_true": data_dict["y_test_B0006"].reshape(-1),
        "y_pred": b6_pred.reshape(-1),
    }).to_csv(out_dir / "pred_B0006.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({
        "y_true": data_dict["y_test_B0007"].reshape(-1),
        "y_pred": b7_pred.reshape(-1),
    }).to_csv(out_dir / "pred_B0007.csv", index=False, encoding="utf-8-sig")

    summary = {
        "model": model_name,
        "validation": val_result,
        "test_B0006": b6_result,
        "test_B0007": b7_result,
    }
    save_json(summary, out_dir / "summary.json")

    return summary, pd.DataFrame(history)