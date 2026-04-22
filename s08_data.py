from pathlib import Path
from typing import Dict, Any

import numpy as np


def load_npz_data(data_path: Path) -> Dict[str, Any]:
    if not data_path.exists():
        raise FileNotFoundError(f"未找到数据文件: {data_path}")

    data = np.load(data_path)

    out = {
        "X_train": data["X_train"],
        "y_train": data["y_train"],
        "X_val": data["X_val"],
        "y_val": data["y_val"],
        "X_test_B0006": data["X_test_B0006"],
        "y_test_B0006": data["y_test_B0006"],
        "X_test_B0007": data["X_test_B0007"],
        "y_test_B0007": data["y_test_B0007"],
    }

    optional_keys = ["mean", "std", "v_low", "v_high", "window_size"]
    for k in optional_keys:
        if k in data:
            out[k] = data[k]

    return out


def sanity_check(data_dict: Dict[str, Any]) -> None:
    for k, v in data_dict.items():
        if isinstance(v, np.ndarray):
            print(f"{k}: shape={v.shape}, dtype={v.dtype}")

    for key in ["X_train", "X_val", "X_test_B0006", "X_test_B0007"]:
        arr = data_dict[key]
        if np.isnan(arr).any():
            raise ValueError(f"{key} 中存在 NaN")

    for key in ["y_train", "y_val", "y_test_B0006", "y_test_B0007"]:
        arr = data_dict[key]
        if np.isnan(arr).any():
            raise ValueError(f"{key} 中存在 NaN")
        print(f"{key}: min={arr.min():.6f}, max={arr.max():.6f}")