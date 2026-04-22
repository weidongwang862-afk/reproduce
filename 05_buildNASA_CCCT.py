from pathlib import Path
import numpy as np
import pandas as pd


def build_candidate_intervals(v_grid, min_width=0.01, max_width=0.25, step=0.001):
    v_grid = np.asarray(v_grid, dtype=float)
    n = len(v_grid)

    min_step = int(round(min_width / step))   # 10
    max_step = int(round(max_width / step))   # 250

    rows = []
    i_idx = []
    j_idx = []

    for i in range(n):
        for d in range(min_step, max_step + 1):
            j = i + d
            if j >= n:
                break

            dv = float(v_grid[j] - v_grid[i])

            i_idx.append(i)
            j_idx.append(j)
            rows.append({
                "interval_id": len(rows),
                "v_low": float(v_grid[i]),
                "v_high": float(v_grid[j]),
                "dv": dv
            })

    interval_df = pd.DataFrame(rows)
    return interval_df, np.array(i_idx, dtype=np.int32), np.array(j_idx, dtype=np.int32)


def build_ccct_features_from_proc(pkl_path, save=True):
    """
    第五步：从 step4 的预处理结果中，计算所有候选区间的 CCCT
    """
    pkl_path = Path(pkl_path)
    proc_df = pd.read_pickle(pkl_path)

    if len(proc_df) == 0:
        raise ValueError(f"{pkl_path.name} 中没有有效样本")

    # 默认所有样本的 v_grid 相同，取第一条
    v_grid = np.asarray(proc_df.iloc[0]["v_grid"], dtype=float)

    interval_df, i_idx, j_idx = build_candidate_intervals(
        v_grid=v_grid,
        min_width=0.01,
        max_width=0.25
    )

    n_samples = len(proc_df)
    n_intervals = len(interval_df)

    ccct_matrix = np.full((n_samples, n_intervals), np.nan, dtype=np.float32)

    sample_meta = proc_df[[
        "battery_id",
        "charge_op_index",
        "discharge_op_index",
        "discharge_index",
        "capacity",
        "C0",
        "soh"
    ]].copy().reset_index(drop=True)

    for k, row in proc_df.iterrows():
        t_grid = np.asarray(row["t_grid"], dtype=float)

        # 对每个候选区间计算 CCCT = t(v_high) - t(v_low)
        ccct = t_grid[j_idx] - t_grid[i_idx]
        ccct_matrix[k, :] = ccct.astype(np.float32)

    if save:
        battery_id = str(sample_meta.loc[0, "battery_id"])

        interval_csv = pkl_path.with_name(f"{battery_id}_intervals_step5.csv")
        meta_csv = pkl_path.with_name(f"{battery_id}_samplemeta_step5.csv")
        npz_file = pkl_path.with_name(f"{battery_id}_ccct_step5.npz")

        interval_df.to_csv(interval_csv, index=False, encoding="utf-8-sig")
        sample_meta.to_csv(meta_csv, index=False, encoding="utf-8-sig")

        np.savez_compressed(
            npz_file,
            ccct_matrix=ccct_matrix,
            v_grid=v_grid.astype(np.float32),
            i_idx=i_idx,
            j_idx=j_idx,
            v_low=interval_df["v_low"].to_numpy(dtype=np.float32),
            v_high=interval_df["v_high"].to_numpy(dtype=np.float32),
            dv=interval_df["dv"].to_numpy(dtype=np.float32),
        )

        print(f"已保存: {interval_csv}")
        print(f"已保存: {meta_csv}")
        print(f"已保存: {npz_file}")

    return interval_df, sample_meta, ccct_matrix


if __name__ == "__main__":
    files = [
        "B0005_chargeproc_step4.pkl",
        "B0006_chargeproc_step4.pkl",
        "B0007_chargeproc_step4.pkl",
    ]

    for f in files:
        interval_df, sample_meta, ccct_matrix = build_ccct_features_from_proc(f, save=True)

        print(f"\n===== {Path(f).stem} =====")
        print("样本数 =", len(sample_meta))
        print("候选区间数 =", len(interval_df))
        print("CCCT矩阵形状 =", ccct_matrix.shape)
        print(interval_df.head())
        print(sample_meta.head())