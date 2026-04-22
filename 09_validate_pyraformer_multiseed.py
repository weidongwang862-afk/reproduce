from pathlib import Path
import json
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (N, L, 1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class CoarserScaleConstruction(nn.Module):
    def __init__(self, d_model: int, num_levels: int = 3):
        super().__init__()
        self.down_blocks = nn.ModuleList([
            nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=3,
                stride=2,
                padding=1
            )
            for _ in range(num_levels)
        ])
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor):
        # x: (B, L, D)
        scales = [x]
        cur = x.transpose(1, 2)  # (B, D, L)

        for block in self.down_blocks:
            cur = block(cur)
            cur = self.act(cur)
            scales.append(cur.transpose(1, 2))  # (B, L_i, D)

        return scales


class PyramidalAttentionBlock(nn.Module):
    def __init__(self, d_model: int = 32, nhead: int = 2, dim_ff: int = 64, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class PyraformerRegressor(nn.Module):
    """
    当前固定为 last_fine 读出
    """
    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 32,
        dim_ff: int = 64,
        nhead: int = 2,
        num_blocks: int = 2,
        num_pyramid_levels: int = 3,
        readout: str = "last_fine"
    ):
        super().__init__()
        self.readout = readout

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=2048)

        self.cscm = CoarserScaleConstruction(
            d_model=d_model,
            num_levels=num_pyramid_levels
        )

        self.scale_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            for _ in range(num_pyramid_levels + 1)
        ])

        self.blocks = nn.ModuleList([
            PyramidalAttentionBlock(
                d_model=d_model,
                nhead=nhead,
                dim_ff=dim_ff,
                dropout=0.0
            )
            for _ in range(num_blocks)
        ])

        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, 1)
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        scales = self.cscm(x)

        tokens = []
        lengths = []
        for i, s in enumerate(scales):
            s = s + self.scale_embeddings[i]
            s = self.pos_encoder(s)
            tokens.append(s)
            lengths.append(s.size(1))

        x_cat = torch.cat(tokens, dim=1)

        for block in self.blocks:
            x_cat = block(x_cat)

        if self.readout == "mean":
            feat = x_cat.mean(dim=1)
        elif self.readout == "last_all":
            feat = x_cat[:, -1, :]
        elif self.readout == "last_fine":
            fine_len = lengths[0]
            feat = x_cat[:, fine_len - 1, :]
        else:
            raise ValueError(f"未知 readout: {self.readout}")

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

    return {
        "MAE": mae,
        "MAPE": mape,
        "RMSE": rmse,
        "R2": r2
    }


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    for xb, _ in loader:
        xb = xb.to(device)
        yb = model(xb).cpu().numpy()
        preds.append(yb)
    return np.concatenate(preds, axis=0)


def evaluate_model(model: nn.Module, X: np.ndarray, y: np.ndarray, batch_size: int, device: torch.device) -> dict:
    ds = SequenceDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    pred = predict(model, loader, device)
    return calc_metrics(y, pred)


def summarize_records(records, prefix):
    rows = []
    for r in records:
        row = {"seed": r["seed"]}
        for k, v in r[prefix].items():
            row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    summary = {}
    for metric in ["MAE", "MAPE", "RMSE", "R2"]:
        summary[metric] = {
            "mean": float(df[metric].mean()),
            "std": float(df[metric].std(ddof=0)),
            "min": float(df[metric].min()),
            "max": float(df[metric].max()),
        }
    return df, summary


def run_one_seed(seed, data_dict, device):
    set_seed(seed)

    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_val = data_dict["X_val"]
    y_val = data_dict["y_val"]
    X_test_B0006 = data_dict["X_test_B0006"]
    y_test_B0006 = data_dict["y_test_B0006"]
    X_test_B0007 = data_dict["X_test_B0007"]
    y_test_B0007 = data_dict["y_test_B0007"]

    epochs = 500
    lr = 1e-3
    batch_size = 16
    weight_decay = 1e-4
    patience = 80
    min_lr = 1e-5

    train_ds = SequenceDataset(X_train, y_train)
    val_ds = SequenceDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = PyraformerRegressor(
        input_dim=1,
        d_model=32,
        dim_ff=64,
        nhead=2,
        num_blocks=2,
        num_pyramid_levels=3,
        readout="last_fine"
    ).to(device)

    criterion = nn.SmoothL1Loss(beta=0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, min_lr=min_lr
    )

    best_val_mae = np.inf
    best_state = None
    history = []
    bad_epochs = 0

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
            "val_R2": val_metrics["R2"]
        })

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"[seed={seed}] Epoch {epoch:4d} | "
                f"lr={current_lr:.6f} | "
                f"train_loss={train_loss:.6f} | "
                f"val_MAE={val_metrics['MAE']:.6f} | "
                f"val_RMSE={val_metrics['RMSE']:.6f} | "
                f"val_R2={val_metrics['R2']:.6f}"
            )

        if bad_epochs >= patience:
            print(f"[seed={seed}] Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_result = evaluate_model(model, X_val, y_val, batch_size, device)
    test_B0006_result = evaluate_model(model, X_test_B0006, y_test_B0006, batch_size, device)
    test_B0007_result = evaluate_model(model, X_test_B0007, y_test_B0007, batch_size, device)

    print(f"\n===== Seed {seed} Final Results =====")
    print("Validation:", val_result)
    print("Test B0006:", test_B0006_result)
    print("Test B0007:", test_B0007_result)

    return {
        "seed": seed,
        "validation": val_result,
        "test_B0006": test_B0006_result,
        "test_B0007": test_B0007_result,
        "history": history,
    }


def main():
    # 这里改成你当前“论文区间 + seq8”对应的输入文件名
    data_path = Path("NASA_model_input_step7b_seq8.npz")
    if not data_path.exists():
        raise FileNotFoundError(f"未找到 {data_path}")

    data = np.load(data_path)
    data_dict = {
        "X_train": data["X_train"],
        "y_train": data["y_train"],
        "X_val": data["X_val"],
        "y_val": data["y_val"],
        "X_test_B0006": data["X_test_B0006"],
        "y_test_B0006": data["y_test_B0006"],
        "X_test_B0007": data["X_test_B0007"],
        "y_test_B0007": data["y_test_B0007"],
    }

    print("X_train =", data_dict["X_train"].shape, "y_train =", data_dict["y_train"].shape)
    print("X_val   =", data_dict["X_val"].shape, "y_val   =", data_dict["y_val"].shape)
    print("X_test_B0006 =", data_dict["X_test_B0006"].shape, "y_test_B0006 =", data_dict["y_test_B0006"].shape)
    print("X_test_B0007 =", data_dict["X_test_B0007"].shape, "y_test_B0007 =", data_dict["y_test_B0007"].shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    seeds = [1, 2, 3, 4, 5]
    all_results = []

    out_dir = Path("results_pyraformer_multiseed")
    out_dir.mkdir(exist_ok=True)

    for seed in seeds:
        result = run_one_seed(seed, data_dict, device)
        all_results.append(result)

        pd.DataFrame(result["history"]).to_csv(
            out_dir / f"history_seed_{seed}.csv",
            index=False,
            encoding="utf-8-sig"
        )

    val_df, val_summary = summarize_records(all_results, "validation")
    b6_df, b6_summary = summarize_records(all_results, "test_B0006")
    b7_df, b7_summary = summarize_records(all_results, "test_B0007")

    val_df.to_csv(out_dir / "validation_all_seeds.csv", index=False, encoding="utf-8-sig")
    b6_df.to_csv(out_dir / "test_B0006_all_seeds.csv", index=False, encoding="utf-8-sig")
    b7_df.to_csv(out_dir / "test_B0007_all_seeds.csv", index=False, encoding="utf-8-sig")

    summary = {
        "model": "Pyraformer_last_fine_multiseed",
        "seeds": seeds,
        "validation_summary": val_summary,
        "test_B0006_summary": b6_summary,
        "test_B0007_summary": b7_summary
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== Multi-seed Summary =====")
    print("Validation:", json.dumps(val_summary, ensure_ascii=False, indent=2))
    print("Test B0006:", json.dumps(b6_summary, ensure_ascii=False, indent=2))
    print("Test B0007:", json.dumps(b7_summary, ensure_ascii=False, indent=2))

    print(f"\n已保存到: {out_dir.resolve()}")


if __name__ == "__main__":
    main()