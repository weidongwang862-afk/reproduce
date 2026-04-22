from pathlib import Path
import json
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (N, L, 1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class TransformerRegressor(nn.Module):
    """
    基于之前诊断结果，固定采用 last token readout。
    """
    def __init__(self, input_dim: int = 1, d_model: int = 32, dim_ff: int = 64, nhead: int = 2, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=512)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=0.0,
            activation="relu",
            batch_first=True,
            norm_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)   # (B, L, 32)
        x = self.pos_encoder(x)
        x = self.encoder(x)      # (B, L, 32)
        feat = x[:, -1, :]       # last token readout
        y = self.head(feat)
        return y


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

    return {"MAE": mae, "MAPE": mape, "RMSE": rmse, "R2": r2}


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    for xb, _ in loader:
        xb = xb.to(device)
        preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)


def evaluate_model(model: nn.Module, X: np.ndarray, y: np.ndarray, batch_size: int, device: torch.device) -> dict:
    ds = SequenceDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    pred = predict(model, loader, device)
    return calc_metrics(y, pred)


def main():
    set_seed(42)

    data_path = Path("NASA_model_input_step7b_seq8.npz")
    if not data_path.exists():
        raise FileNotFoundError(f"未找到 {data_path}")

    data = np.load(data_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test_B0006 = data["X_test_B0006"]
    y_test_B0006 = data["y_test_B0006"]
    X_test_B0007 = data["X_test_B0007"]
    y_test_B0007 = data["y_test_B0007"]

    epochs = 1000
    lr = 1e-3
    batch_size = 16
    patience = 100
    min_lr = 1e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)
    print("X_train =", X_train.shape, "y_train =", y_train.shape)
    print("X_val   =", X_val.shape, "y_val   =", y_val.shape)
    print("X_test_B0006 =", X_test_B0006.shape, "y_test_B0006 =", y_test_B0006.shape)
    print("X_test_B0007 =", X_test_B0007.shape, "y_test_B0007 =", y_test_B0007.shape)

    train_ds = SequenceDataset(X_train, y_train)
    val_ds = SequenceDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = TransformerRegressor(
        input_dim=1,
        d_model=32,
        dim_ff=64,
        nhead=2,
        num_layers=2
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, min_lr=min_lr
    )

    best_val_mae = np.inf
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * len(xb)
            n_train += len(xb)

        train_loss = train_loss_sum / max(n_train, 1)

        val_pred = predict(model, val_loader, device)
        val_metrics = calc_metrics(y_val, val_pred)
        scheduler.step(val_metrics["MAE"])
        current_lr = optimizer.param_groups[0]["lr"]

        if val_metrics["MAE"] < best_val_mae:
            best_val_mae = val_metrics["MAE"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        history.append({
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "val_MAE": val_metrics["MAE"],
            "val_MAPE": val_metrics["MAPE"],
            "val_RMSE": val_metrics["RMSE"],
            "val_R2": val_metrics["R2"],
        })

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"Epoch {epoch:4d} | lr={current_lr:.6f} | "
                f"train_loss={train_loss:.6f} | val_MAE={val_metrics['MAE']:.6f} | "
                f"val_RMSE={val_metrics['RMSE']:.6f} | val_R2={val_metrics['R2']:.6f}"
            )

        if bad_epochs >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_result = evaluate_model(model, X_val, y_val, batch_size, device)
    test_B0006_result = evaluate_model(model, X_test_B0006, y_test_B0006, batch_size, device)
    test_B0007_result = evaluate_model(model, X_test_B0007, y_test_B0007, batch_size, device)

    print("\n===== Final Results (Best Val Model) =====")
    print("Validation:", val_result)
    print("Test B0006:", test_B0006_result)
    print("Test B0007:", test_B0007_result)

    out_dir = Path("results_transformer_nasa_seq8")
    out_dir.mkdir(exist_ok=True)

    torch.save(model.state_dict(), out_dir / "transformer_best.pt")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False, encoding="utf-8-sig")

    summary = {
        "model": "Transformer_last",
        "data_path": str(data_path),
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "validation": val_result,
        "test_B0006": test_B0006_result,
        "test_B0007": test_B0007_result,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n已保存模型与结果到: {out_dir.resolve()}")


if __name__ == "__main__":
    main()