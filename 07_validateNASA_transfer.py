from pathlib import Path
import numpy as np
import pandas as pd


def extract_scalar_ccct(proc_df, v_low=3.804, v_high=4.054):
    """
    从 step4 的 proc_df 中，为每条样本提取一个共享区间 CCCT 标量
    CCCT = t(v_high) - t(v_low)
    """
    rows = []

    for _, row in proc_df.iterrows():
        v_grid = np.asarray(row["v_grid"], dtype=float)
        t_grid = np.asarray(row["t_grid"], dtype=float)

        i0 = int(np.argmin(np.abs(v_grid - v_low)))
        i1 = int(np.argmin(np.abs(v_grid - v_high)))

        if i1 <= i0:
            continue

        t0 = float(t_grid[i0])
        t1 = float(t_grid[i1])

        if (not np.isfinite(t0)) or (not np.isfinite(t1)):
            continue

        ccct = t1 - t0
        if (not np.isfinite(ccct)) or (ccct < 0):
            continue

        rows.append({
            "battery_id": row["battery_id"],
            "charge_op_index": int(row["charge_op_index"]),
            "discharge_op_index": int(row["discharge_op_index"]),
            "discharge_index": int(row["discharge_index"]),
            "capacity": float(row["capacity"]),
            "C0": float(row["C0"]),
            "soh": float(row["soh"]),
            "ccct": float(ccct),
        })

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values("discharge_index").reset_index(drop=True)
    return out_df


def build_windows_from_series(df, window_size=8, label_mode="last"):
    """
    把单标量 CCCT 序列构造成滑动窗口输入
    label_mode:
        - "last":  标签取窗口最后一个点的 SOH
        - "next":  标签取窗口后一个点的 SOH（这里先不用）
    """
    ccct = df["ccct"].to_numpy(dtype=np.float32)
    soh = df["soh"].to_numpy(dtype=np.float32)

    X_list = []
    y_list = []
    meta_rows = []

    if label_mode == "last":
        max_start = len(df) - window_size + 1
        for s in range(max_start):
            e = s + window_size
            x = ccct[s:e]                 # 长度 8
            y = soh[e - 1]                # 窗口末端 SOH

            X_list.append(x)
            y_list.append(y)
            meta_rows.append({
                "battery_id": df.loc[e - 1, "battery_id"],
                "window_start_discharge_index": int(df.loc[s, "discharge_index"]),
                "window_end_discharge_index": int(df.loc[e - 1, "discharge_index"]),
                "label_soh": float(y),
            })

    elif label_mode == "next":
        max_start = len(df) - window_size
        for s in range(max_start):
            e = s + window_size
            x = ccct[s:e]
            y = soh[e]                    # 下一时刻 SOH

            X_list.append(x)
            y_list.append(y)
            meta_rows.append({
                "battery_id": df.loc[e, "battery_id"],
                "window_start_discharge_index": int(df.loc[s, "discharge_index"]),
                "window_end_discharge_index": int(df.loc[e - 1, "discharge_index"]),
                "label_soh": float(y),
            })
    else:
        raise ValueError("label_mode 只能是 'last' 或 'next'")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    meta_df = pd.DataFrame(meta_rows)

    return X, y, meta_df


def standardize_by_train(X_train, X_val, X_test1, X_test2):
    """
    只用训练集统计量做标准化，避免信息泄漏
    """
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0

    X_train_std = (X_train - mean) / std
    X_val_std = (X_val - mean) / std
    X_test1_std = (X_test1 - mean) / std
    X_test2_std = (X_test2 - mean) / std

    return X_train_std, X_val_std, X_test1_std, X_test2_std, mean, std


if __name__ == "__main__":
    # 当前 NASA 共享区间
    v_low = 3.832
    v_high = 4.082

    # 论文补充材料里窗口长度提到 8，这里先按 8 做
    window_size = 8
    label_mode = "last"   # 先默认窗口末端 SOH 为标签

    # 读取 step4 输出
    proc_05 = pd.read_pickle("B0005_chargeproc_step4.pkl")
    proc_06 = pd.read_pickle("B0006_chargeproc_step4.pkl")
    proc_07 = pd.read_pickle("B0007_chargeproc_step4.pkl")

    # 1) 每条样本先提一个共享区间 CCCT 标量
    seq_05 = extract_scalar_ccct(proc_05, v_low=v_low, v_high=v_high)
    seq_06 = extract_scalar_ccct(proc_06, v_low=v_low, v_high=v_high)
    seq_07 = extract_scalar_ccct(proc_07, v_low=v_low, v_high=v_high)

    print("B0005 标量样本数 =", len(seq_05))
    print("B0006 标量样本数 =", len(seq_06))
    print("B0007 标量样本数 =", len(seq_07))

    # 2) B0005 按 raw sequence 前30%/后70% 划分，再各自做窗口
    n05 = len(seq_05)
    n05_train = int(np.floor(n05 * 0.30))

    seq_05_train = seq_05.iloc[:n05_train].reset_index(drop=True)
    seq_05_val = seq_05.iloc[n05_train:].reset_index(drop=True)

    print("\n===== 原始序列划分 =====")
    print("B0005 train raw =", len(seq_05_train))
    print("B0005 val raw   =", len(seq_05_val))
    print("B0006 raw       =", len(seq_06))
    print("B0007 raw       =", len(seq_07))

    # 3) 滑窗构造
    X_train, y_train, meta_train = build_windows_from_series(
        seq_05_train, window_size=window_size, label_mode=label_mode
    )
    X_val, y_val, meta_val = build_windows_from_series(
        seq_05_val, window_size=window_size, label_mode=label_mode
    )
    X_test_B0006, y_test_B0006, meta_test_06 = build_windows_from_series(
        seq_06, window_size=window_size, label_mode=label_mode
    )
    X_test_B0007, y_test_B0007, meta_test_07 = build_windows_from_series(
        seq_07, window_size=window_size, label_mode=label_mode
    )

    print("\n===== 滑窗结果 =====")
    print("X_train =", X_train.shape, "y_train =", y_train.shape)
    print("X_val   =", X_val.shape, "y_val   =", y_val.shape)
    print("X_test_B0006 =", X_test_B0006.shape, "y_test_B0006 =", y_test_B0006.shape)
    print("X_test_B0007 =", X_test_B0007.shape, "y_test_B0007 =", y_test_B0007.shape)

    # 4) 训练集统计量标准化
    X_train_std, X_val_std, X_test_B0006_std, X_test_B0007_std, mean, std = standardize_by_train(
        X_train, X_val, X_test_B0006, X_test_B0007
    )

    # 5) 保存中间表
    seq_05.to_csv("NASA_scalar_ccct_B0005_step7b.csv", index=False, encoding="utf-8-sig")
    seq_06.to_csv("NASA_scalar_ccct_B0006_step7b.csv", index=False, encoding="utf-8-sig")
    seq_07.to_csv("NASA_scalar_ccct_B0007_step7b.csv", index=False, encoding="utf-8-sig")

    meta_train.to_csv("NASA_meta_train_step7b.csv", index=False, encoding="utf-8-sig")
    meta_val.to_csv("NASA_meta_val_step7b.csv", index=False, encoding="utf-8-sig")
    meta_test_06.to_csv("NASA_meta_test_B0006_step7b.csv", index=False, encoding="utf-8-sig")
    meta_test_07.to_csv("NASA_meta_test_B0007_step7b.csv", index=False, encoding="utf-8-sig")

    # 6) 保存模型输入
    np.savez_compressed(
        "NASA_model_input_step7b_seq8.npz",
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
        window_size=np.array([window_size], dtype=np.int32),
    )

    print("\n已保存: NASA_model_input_step7b_seq8.npz")
    print("已保存: NASA_scalar_ccct_B0005_step7b.csv")
    print("已保存: NASA_scalar_ccct_B0006_step7b.csv")
    print("已保存: NASA_scalar_ccct_B0007_step7b.csv")
    print("已保存: NASA_meta_train_step7b.csv")
    print("已保存: NASA_meta_val_step7b.csv")
    print("已保存: NASA_meta_test_B0006_step7b.csv")
    print("已保存: NASA_meta_test_B0007_step7b.csv")