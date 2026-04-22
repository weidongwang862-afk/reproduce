from pathlib import Path
import pandas as pd


def build_soh_label_from_discharge(csv_path, save=True):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    df = df.sort_values("op_index").reset_index(drop=True)

    if "capacity" not in df.columns:
        raise ValueError(f"{csv_path.name} 中没有 capacity 列")

    df = df[df["capacity"].notna()].copy().reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(f"{csv_path.name} 中没有有效的 capacity 数据")

    df["discharge_index"] = range(1, len(df) + 1)

    C0 = float(df.loc[0, "capacity"])
    df["C0"] = C0
    df["soh"] = df["capacity"] / C0

    out_cols = [
        "battery_id",
        "op_index",
        "discharge_index",
        "ambient_temperature",
        "start_time_vec",
        "capacity",
        "C0",
        "soh",
        "n_points",
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    label_df = df[out_cols].copy()

    if save:
        out_path = csv_path.with_name(csv_path.name.replace("_discharge_step1.csv", "_label_step2.csv"))
        label_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"已保存: {out_path}")

    return label_df


if __name__ == "__main__":
    files = [
        "B0005_discharge_step1.csv",
        "B0006_discharge_step1.csv",
        "B0007_discharge_step1.csv",
    ]

    for f in files:
        label_df = build_soh_label_from_discharge(f, save=True)
        print(f"\n===== {f} =====")
        print(label_df.head())
        print(label_df.tail())
        print("初始容量 C0 =", label_df['C0'].iloc[0])
        print("最小 SOH =", label_df['soh'].min())