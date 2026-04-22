from pathlib import Path
import numpy as np
import pandas as pd


def clean_monotonic_curve(v_grid: np.ndarray, t_grid: np.ndarray):
    """
    清理并排序电压-时间曲线，保证可插值
    """
    v = np.asarray(v_grid, dtype=float).reshape(-1)
    t = np.asarray(t_grid, dtype=float).reshape(-1)

    mask = np.isfinite(v) & np.isfinite(t)
    v = v[mask]
    t = t[mask]

    if len(v) < 2:
        return None, None

    order = np.argsort(v)
    v = v[order]
    t = t[order]

    # 电压去重，保留最后一个对应时间
    v_unique = []
    t_unique = []
    last_v = None
    for vi, ti in zip(v, t):
        if last_v is None or abs(vi - last_v) > 1e-12:
            v_unique.append(vi)
            t_unique.append(ti)
            last_v = vi
        else:
            t_unique[-1] = ti

    v = np.asarray(v_unique, dtype=float)
    t = np.asarray(t_unique, dtype=float)

    if len(v) < 2:
        return None, None

    return v, t


def extract_finegrid_cumtime_features(
    proc_df: pd.DataFrame,
    v_low: float,
    v_high: float,
    n_points: int = 17,
):
    """
    版本 C1-b:
    在 global interval 内取固定细电压网格，构造“去起点偏移的累计时间序列”
    x = [0, t(v1)-t(v0), ..., t(v_{M})-t(v0)]
    """
    voltage_points = np.linspace(v_low, v_high, n_points, dtype=float)

    rows = []

    for _, row in proc_df.iterrows():
        v_grid = np.asarray(row["v_grid"], dtype=float)
        t_grid = np.asarray(row["t_grid"], dtype=float)

        v, t = clean_monotonic_curve(v_grid, t_grid)
        if v is None:
            continue

        # 必须完整覆盖 global interval
        if v.min() > v_low or v.max() < v_high:
            continue

        t_points = np.interp(voltage_points, v, t)

        if not np.all(np.isfinite(t_points)):
            continue

        # 去起点偏移
        t_rel = t_points - t_points[0]

        if np.any(t_rel < -1e-9):
            continue

        one = {
            "battery_id": row["battery_id"],
            "charge_op_index": int(row["charge_op_index"]),
            "discharge_op_index": int(row["discharge_op_index"]),
            "discharge_index": int(row["discharge_index"]),
            "capacity": float(row["capacity"]),
            "C0": float(row["C0"]),
            "soh": float(row["soh"]),
        }

        for i in range(n_points):
            one[f"ft_{i+1}"] = float(t_rel[i])

        rows.append(one)

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values("discharge_index").reset_index(drop=True)
    return out_df, voltage_points


def build_event_dataset(
    df_feat: pd.DataFrame,
    n_points: int = 17,
    label_mode: str = "current",
):
    """
    current:
        当前事件的 fine-grid 累计时间序列 -> 当前事件 SOH
    next:
        当前事件的 fine-grid 累计时间序列 -> 下一事件 SOH
    """
    feat_cols = [f"ft_{i+1}" for i in range(n_points)]
    X_all = df_feat[feat_cols].to_numpy(dtype=np.float32)
    y_all = df_feat["soh"].to_numpy(dtype=np.float32)

    if label_mode == "current":
        X = X_all
        y = y_all
        meta = df_feat[["battery_id", "discharge_index", "soh"]].copy()
        meta["label_index"] = meta["discharge_index"]
        meta["label_soh"] = meta["soh"]

    elif label_mode == "next":
        if len(df_feat) < 2:
            raise ValueError("样本不足，无法构造 next 标签")
        X = X_all[:-1]
        y = y_all[1:]
        meta = df_feat.iloc[:-1][["battery_id", "discharge_index", "soh"]].copy()
        meta["label_index"] = df_feat.iloc[1:]["discharge_index"].to_numpy()
        meta["label_soh"] = y

    else:
        raise ValueError("label_mode 必须是 'current' 或 'next'")

    return X.astype(np.float32), y.astype(np.float32), meta.reset_index(drop=True)


def standardize_by_train(X_train, X_val, X_test1, X_test2):
    """
    只用训练集统计量做按位置标准化
    """
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0

    X_train_std = (X_train - mean) / std
    X_val_std = (X_val - mean) / std
    X_test1_std = (X_test1 - mean) / std
    X_test2_std = (X_test2 - mean) / std

    return X_train_std, X_val_std, X_test1_std, X_test2_std, mean, std


def main():
    # 你们当前 NASA 搜索得到的共享区间
    v_low = 3.804
    v_high = 4.054

    # fine-grid 节点数
    n_points = 17

    # 第一轮固定做 current
    label_mode = "current"

    # 读取 step4 输出
    proc_05 = pd.read_pickle("B0005_chargeproc_step4.pkl")
    proc_06 = pd.read_pickle("B0006_chargeproc_step4.pkl")
    proc_07 = pd.read_pickle("B0007_chargeproc_step4.pkl")

    # 1) 构造 fine-grid 累计时间序列特征
    feat_05, voltage_points = extract_finegrid_cumtime_features(
        proc_05, v_low=v_low, v_high=v_high, n_points=n_points
    )
    feat_06, _ = extract_finegrid_cumtime_features(
        proc_06, v_low=v_low, v_high=v_high, n_points=n_points
    )
    feat_07, _ = extract_finegrid_cumtime_features(
        proc_07, v_low=v_low, v_high=v_high, n_points=n_points
    )

    print("===== Fine-grid Feature Counts =====")
    print("B0005 特征样本数 =", len(feat_05))
    print("B0006 特征样本数 =", len(feat_06))
    print("B0007 特征样本数 =", len(feat_07))
    print("voltage_points =", np.round(voltage_points, 6))

    # 2) B0005 前30%/后70%
    n05 = len(feat_05)
    n05_train = int(np.floor(n05 * 0.30))

    feat_05_train = feat_05.iloc[:n05_train].reset_index(drop=True)
    feat_05_val = feat_05.iloc[n05_train:].reset_index(drop=True)

    print("\n===== Raw Event Split =====")
    print("B0005 train raw =", len(feat_05_train))
    print("B0005 val raw   =", len(feat_05_val))
    print("B0006 raw       =", len(feat_06))
    print("B0007 raw       =", len(feat_07))

    # 3) 构造数据集
    X_train, y_train, meta_train = build_event_dataset(
        feat_05_train, n_points=n_points, label_mode=label_mode
    )
    X_val, y_val, meta_val = build_event_dataset(
        feat_05_val, n_points=n_points, label_mode=label_mode
    )
    X_test_B0006, y_test_B0006, meta_test_06 = build_event_dataset(
        feat_06, n_points=n_points, label_mode=label_mode
    )
    X_test_B0007, y_test_B0007, meta_test_07 = build_event_dataset(
        feat_07, n_points=n_points, label_mode=label_mode
    )

    print("\n===== Dataset Shapes =====")
    print("X_train =", X_train.shape, "y_train =", y_train.shape)
    print("X_val   =", X_val.shape, "y_val   =", y_val.shape)
    print("X_test_B0006 =", X_test_B0006.shape, "y_test_B0006 =", y_test_B0006.shape)
    print("X_test_B0007 =", X_test_B0007.shape, "y_test_B0007 =", y_test_B0007.shape)

    # 看一下前几个样本
    print("\n===== Preview: first 3 training samples =====")
    for i in range(min(3, len(X_train))):
        print(f"[{i}] y={y_train[i]:.6f}, x={np.round(X_train[i], 4).tolist()}")

    # 4) 标准化
    X_train_std, X_val_std, X_test_B0006_std, X_test_B0007_std, mean, std = standardize_by_train(
        X_train, X_val, X_test_B0006, X_test_B0007
    )

    # 5) 保存特征表
    feat_05.to_csv("NASA_finegrid17_features_B0005_step7d.csv", index=False, encoding="utf-8-sig")
    feat_06.to_csv("NASA_finegrid17_features_B0006_step7d.csv", index=False, encoding="utf-8-sig")
    feat_07.to_csv("NASA_finegrid17_features_B0007_step7d.csv", index=False, encoding="utf-8-sig")

    meta_train.to_csv("NASA_meta_train_step7d_finegrid17_current.csv", index=False, encoding="utf-8-sig")
    meta_val.to_csv("NASA_meta_val_step7d_finegrid17_current.csv", index=False, encoding="utf-8-sig")
    meta_test_06.to_csv("NASA_meta_test_B0006_step7d_finegrid17_current.csv", index=False, encoding="utf-8-sig")
    meta_test_07.to_csv("NASA_meta_test_B0007_step7d_finegrid17_current.csv", index=False, encoding="utf-8-sig")

    # 6) 保存模型输入
    out_name = "NASA_model_input_step7d_finegrid17_current.npz"
    np.savez_compressed(
        out_name,
        X_train=X_train_std.astype(np.float32),
        y_train=y_train.astype(np.float32),
        X_val=X_val_std.astype(np.float32),
        y_val=y_val.astype(np.float32),
        X_test_B0006=X_test_B0006_std.astype(np.float32),
        y_test_B0006=y_test_B0006.astype(np.float32),
        X_test_B0007=X_test_B0007_std.astype(np.float32),
        y_test_B0007=y_test_B0007.astype(np.float32),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        v_low=np.array([v_low], dtype=np.float32),
        v_high=np.array([v_high], dtype=np.float32),
        voltage_points=voltage_points.astype(np.float32),
        n_points=np.array([n_points], dtype=np.int32),
    )

    print(f"\n已保存: {out_name}")
    print("已保存: NASA_finegrid17_features_B0005_step7d.csv")
    print("已保存: NASA_finegrid17_features_B0006_step7d.csv")
    print("已保存: NASA_finegrid17_features_B0007_step7d.csv")


if __name__ == "__main__":
    main()