from pathlib import Path
import numpy as np
import pandas as pd


def compute_pcc_vectorized(X, y):
    """
    X: (n_samples, n_features)
    y: (n_samples,)
    返回每一列与 y 的 Pearson 相关系数
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if X.shape[0] != len(y):
        raise ValueError("X 的样本数与 y 长度不一致")

    y_mean = y.mean()
    y_center = y - y_mean
    y_std = np.sqrt(np.sum(y_center ** 2))

    X_mean = X.mean(axis=0)
    X_center = X - X_mean
    X_std = np.sqrt(np.sum(X_center ** 2, axis=0))

    denom = X_std * y_std
    numer = np.sum(X_center * y_center[:, None], axis=0)

    pcc = np.full(X.shape[1], np.nan, dtype=np.float64)
    valid = denom > 0
    pcc[valid] = numer[valid] / denom[valid]

    return pcc


def compute_single_battery_pcc(npz_path, meta_csv_path, interval_csv_path, save=True):
    npz_path = Path(npz_path)
    meta_csv_path = Path(meta_csv_path)
    interval_csv_path = Path(interval_csv_path)

    data = np.load(npz_path)
    ccct_matrix = data["ccct_matrix"]

    meta_df = pd.read_csv(meta_csv_path, encoding="utf-8-sig")
    interval_df = pd.read_csv(interval_csv_path, encoding="utf-8-sig")

# NASA 先按论文的失效阈值截断
    keep = meta_df["capacity"] >= 1.4

    meta_df = meta_df.loc[keep].reset_index(drop=True)
    ccct_matrix = ccct_matrix[keep.to_numpy(), :]

    y = meta_df["soh"].to_numpy(dtype=np.float64)
    pcc = compute_pcc_vectorized(ccct_matrix, y)

    out_df = interval_df.copy()
    out_df["pcc"] = pcc

    if save:
        battery_id = str(meta_df.loc[0, "battery_id"])
        out_path = npz_path.with_name(f"{battery_id}_pcc_step6.csv")
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"已保存: {out_path}")

    return out_df


def build_avg_pcc_for_nasa(df1, df2, name1="B0005", name2="B0006", save=True,
                           tol_avg=5e-4, min_single_pcc=0.99):
    merged = df1.copy()
    merged = merged.rename(columns={"pcc": f"pcc_{name1}"})
    merged[f"pcc_{name2}"] = df2["pcc"].values
    merged["pcc_avg"] = (merged[f"pcc_{name1}"] + merged[f"pcc_{name2}"]) / 2.0
    merged["pcc_min"] = merged[[f"pcc_{name1}", f"pcc_{name2}"]].min(axis=1)

    # 1. 先找最高平均 PCC
    pcc_max = merged["pcc_avg"].max()

    # 2. 保留“接近最高点”的高相关平台
    candidates = merged[
        (merged["pcc_avg"] >= pcc_max - tol_avg) &
        (merged[f"pcc_{name1}"] >= min_single_pcc) &
        (merged[f"pcc_{name2}"] >= min_single_pcc)
    ].copy()

    # 如果条件太严，一个都没有，就只保留接近最高 avg 的区间
    if len(candidates) == 0:
        candidates = merged[
            merged["pcc_avg"] >= pcc_max - tol_avg
        ].copy()

    # 3. 在高相关平台里优先选更宽的区间，再看平均 PCC
    candidates = candidates.sort_values(
        ["dv", "pcc_avg", "pcc_min", "v_high"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    best_row = candidates.iloc[0].copy()

    if save:
        out_path = Path(f"NASA_{name1}_{name2}_avgpcc_step6.csv")
        merged.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"已保存: {out_path}")

    return merged, candidates, best_row


if __name__ == "__main__":
    # 只用 B0005 和 B0006 搜索 NASA 共享区间
    df_05 = compute_single_battery_pcc(
        npz_path="B0005_ccct_step5.npz",
        meta_csv_path="B0005_samplemeta_step5.csv",
        interval_csv_path="B0005_intervals_step5.csv",
        save=True
    )

    df_06 = compute_single_battery_pcc(
        npz_path="B0006_ccct_step5.npz",
        meta_csv_path="B0006_samplemeta_step5.csv",
        interval_csv_path="B0006_intervals_step5.csv",
        save=True
    )

    avg_df, cand_df, best_row = build_avg_pcc_for_nasa(
    df_05, df_06,
    name1="B0005",
    name2="B0006",
    save=True,
    tol_avg=5e-4,
    min_single_pcc=0.99
)
    
    print("\n===== NASA B0005 + B0006 共享区间搜索结果 =====")
    print("最优 interval_id =", int(best_row["interval_id"]))
    print("v_low =", float(best_row["v_low"]))
    print("v_high =", float(best_row["v_high"]))
    print("dv =", float(best_row["dv"]))
    print("pcc_B0005 =", float(best_row["pcc_B0005"]))
    print("pcc_B0006 =", float(best_row["pcc_B0006"]))
    print("pcc_avg =", float(best_row["pcc_avg"]))
    print("高相关平台候选数 =", len(cand_df))
    print("\n高相关平台中最宽的前10个区间：")
    print(
    cand_df[["interval_id", "v_low", "v_high", "dv", "pcc_B0005", "pcc_B0006", "pcc_avg"]]
    .head(10)
)
    print("\nTop 10 区间：")
    print(
        avg_df.sort_values("pcc_avg", ascending=False)[
            ["interval_id", "v_low", "v_high", "dv", "pcc_B0005", "pcc_B0006", "pcc_avg"]
        ].head(10)
    )