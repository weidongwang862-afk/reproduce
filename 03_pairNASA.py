from pathlib import Path
import importlib.util
import pandas as pd
import numpy as np

# 01_loadNASA.py 以数字开头，不能用常规 import，改为按路径动态导入
_step1_path = Path(__file__).resolve().parent / "01_loadNASA.py"
_spec = importlib.util.spec_from_file_location("loadNASA_step1", _step1_path)
_step1_module = importlib.util.module_from_spec(_spec)
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_step1_module)
load_nasa_battery_step1 = _step1_module.load_nasa_battery_step1


def _resolve_existing_path(p: Path, extra_candidates=None) -> Path:
    p = Path(p)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()

    project_root = Path(__file__).resolve().parent.parent
    candidates = [project_root / p]
    if extra_candidates:
        candidates.extend(extra_candidates)

    for c in candidates:
        if c.exists():
            return c.resolve()
    return p


def build_charge_label_pairs(mat_path, label_csv_path, save=True):
    """
    第三步：把每条 charge 和其后尚未使用的第一条 discharge 的 SOH 标签严格一对一配对
    """
    mat_path = _resolve_existing_path(
        Path(mat_path),
        extra_candidates=[
            Path(__file__).resolve().parent.parent / "Dataset" / "BatteryAgingARC-FY08Q4" / Path(mat_path).name,
            Path(__file__).resolve().parent.parent / "Dataset" / Path(mat_path).name,
        ],
    )
    label_csv_path = _resolve_existing_path(Path(label_csv_path))

    battery_id, meta_df, charge_records, _ = load_nasa_battery_step1(mat_path)
    label_df = pd.read_csv(label_csv_path, encoding="utf-8-sig")
    label_df = label_df.sort_values("op_index").reset_index(drop=True)

    # 建立 discharge 索引映射，方便按 op_index 取标签
    discharge_op_list = label_df["op_index"].tolist()
    discharge_map = {int(row["op_index"]): row for _, row in label_df.iterrows()}

    # charge 按 op_index 排序，确保按原始顺序配对
    charge_records = sorted(charge_records, key=lambda x: x["op_index"])

    pair_rows = []
    j = 0  # discharge 指针

    for rec in charge_records:
        charge_op = int(rec["op_index"])

        # 跳过所有已经在 charge 前面的 discharge
        while j < len(discharge_op_list) and discharge_op_list[j] <= charge_op:
            j += 1

        # 没有可配对的 discharge，就结束
        if j >= len(discharge_op_list):
            break

        # 取“后面尚未使用的第一条 discharge”
        discharge_op = int(discharge_op_list[j])
        row_label = discharge_map[discharge_op]

        time_arr = rec["time"]
        v_arr = rec["voltage"]

        pair_rows.append({
            "battery_id": battery_id,
            "charge_op_index": charge_op,
            "discharge_op_index": discharge_op,
            "discharge_index": int(row_label["discharge_index"]),
            "capacity": float(row_label["capacity"]),
            "C0": float(row_label["C0"]),
            "soh": float(row_label["soh"]),
            "charge_n_points": int(len(time_arr)),
            "charge_t_end": float(time_arr[-1]) if len(time_arr) > 0 else np.nan,
            "charge_v_min": float(np.min(v_arr)) if len(v_arr) > 0 else np.nan,
            "charge_v_max": float(np.max(v_arr)) if len(v_arr) > 0 else np.nan,
        })

        # 这条 discharge 已经用过，指针后移，保证一对一
        j += 1

    pair_df = pd.DataFrame(pair_rows)

    # 排序和一致性检查
    pair_df = pair_df.sort_values(["charge_op_index", "discharge_op_index"]).reset_index(drop=True)

    # 关键校验：一条 discharge 只能出现一次
    if pair_df["discharge_op_index"].duplicated().any():
        dup_vals = pair_df.loc[pair_df["discharge_op_index"].duplicated(), "discharge_op_index"].tolist()
        raise ValueError(f"{battery_id} 配对失败：存在重复使用的 discharge_op_index: {dup_vals}")

    # 关键校验：必须满足 charge 在前，discharge 在后
    bad_order = pair_df[pair_df["discharge_op_index"] <= pair_df["charge_op_index"]]
    if len(bad_order) > 0:
        raise ValueError(f"{battery_id} 配对失败：存在 discharge_op_index <= charge_op_index 的样本")

    if save:
        out_path = label_csv_path.with_name(label_csv_path.name.replace("_label_step2.csv", "_pair_step3.csv"))
        pair_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"已保存: {out_path}")

    return pair_df


if __name__ == "__main__":
    files = [
        (r"Dataset/BatteryAgingARC-FY08Q4/B0005.mat", r"B0005_label_step2.csv"),
        (r"Dataset/BatteryAgingARC-FY08Q4/B0006.mat", r"B0006_label_step2.csv"),
        (r"Dataset/BatteryAgingARC-FY08Q4/B0007.mat", r"B0007_label_step2.csv"),
    ]

    for mat_file, label_file in files:
        pair_df = build_charge_label_pairs(mat_file, label_file, save=True)
        print(f"\n===== {Path(mat_file).stem} =====")
        print(pair_df.head())
        print(pair_df.tail())
        print("样本数 =", len(pair_df))
        print("重复 discharge 数 =", pair_df["discharge_op_index"].duplicated().sum())