from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat


def _to_list(x):
    if isinstance(x, np.ndarray):
        return x.flatten().tolist()
    return [x]


def _to_1d_float(x):
    arr = np.asarray(x).squeeze()
    if arr.size == 0:
        return np.array([], dtype=float)
    return arr.astype(float).reshape(-1)


def _safe_scalar(x, default=np.nan):
    try:
        arr = np.asarray(x).squeeze()
        if arr.size == 0:
            return default
        return float(arr)
    except Exception:
        return default


def _safe_time_vec(x):
    try:
        arr = np.asarray(x).squeeze()
        if arr.size == 0:
            return []
        return arr.astype(int).tolist()
    except Exception:
        return []


def load_nasa_battery_step1(mat_path):
    """
    第一步：只负责读取 NASA 原始 .mat，并整理成统一结构
    返回：
        battery_id
        
        meta_df
        charge_records
        discharge_df
    """
    mat_path = Path(mat_path)
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    # 自动寻找顶层电池变量，比如 B0005
    valid_keys = [k for k in mat.keys() if not k.startswith("__")]
    if len(valid_keys) == 1:
        battery_id = valid_keys[0]
    else:
        # 优先找和文件名同名的变量
        stem = mat_path.stem
        if stem in mat:
            battery_id = stem
        else:
            raise KeyError(f"无法唯一确定顶层变量名，候选有: {valid_keys}")

    battery = mat[battery_id]

    if not hasattr(battery, "cycle"):
        raise ValueError(f"{battery_id} 中没有找到 cycle 字段")

    cycles = _to_list(battery.cycle)

    meta_rows = []
    charge_records = []
    discharge_rows = []

    for op_index, cyc in enumerate(cycles):
        ctype = str(getattr(cyc, "type", "")).strip().lower()
        ambient_temp = _safe_scalar(getattr(cyc, "ambient_temperature", np.nan))
        start_time_vec = _safe_time_vec(getattr(cyc, "time", []))

        meta_rows.append({
            "battery_id": battery_id,
            "op_index": op_index,
            "type": ctype,
            "ambient_temperature": ambient_temp,
            "start_time_vec": start_time_vec,
        })

        data = getattr(cyc, "data", None)
        if data is None:
            continue

        if ctype == "charge":
            record = {
                "battery_id": battery_id,
                "op_index": op_index,
                "ambient_temperature": ambient_temp,
                "start_time_vec": start_time_vec,
                "time": _to_1d_float(getattr(data, "Time", [])),
                "voltage": _to_1d_float(getattr(data, "Voltage_measured", [])),
                "current": _to_1d_float(getattr(data, "Current_measured", [])),
                "temperature": _to_1d_float(getattr(data, "Temperature_measured", [])),
                "current_charge": _to_1d_float(getattr(data, "Current_charge", [])),
                "voltage_charge": _to_1d_float(getattr(data, "Voltage_charge", [])),
            }
            charge_records.append(record)

        elif ctype == "discharge":
            row = {
                "battery_id": battery_id,
                "op_index": op_index,
                "ambient_temperature": ambient_temp,
                "start_time_vec": start_time_vec,
                "capacity": _safe_scalar(getattr(data, "Capacity", np.nan)),
                "n_points": len(_to_1d_float(getattr(data, "Time", []))),
            }
            discharge_rows.append(row)

        # impedance 在第一步先不展开，只在 meta_df 里保留类型信息

    meta_df = pd.DataFrame(meta_rows)
    discharge_df = pd.DataFrame(discharge_rows)

    return battery_id, meta_df, charge_records, discharge_df


def print_step1_summary(battery_id, meta_df, charge_records, discharge_df):
    print(f"\n===== {battery_id} 第一步读取结果 =====")
    print("各类型 operation 数量：")
    print(meta_df["type"].value_counts(dropna=False))
    print("\ncharge 条数：", len(charge_records))
    print("discharge 条数：", len(discharge_df))

    if len(charge_records) > 0:
        first_charge = charge_records[0]
        print("\n第一条 charge 检查：")
        print("op_index =", first_charge["op_index"])
        print("len(time) =", len(first_charge["time"]))
        print("len(voltage) =", len(first_charge["voltage"]))
        print("len(current) =", len(first_charge["current"]))
        print("len(temperature) =", len(first_charge["temperature"]))
        if len(first_charge["time"]) > 0:
            print("time range =", first_charge["time"][0], "->", first_charge["time"][-1])
        if len(first_charge["voltage"]) > 0:
            print("voltage range =", first_charge["voltage"].min(), "->", first_charge["voltage"].max())

    if len(discharge_df) > 0:
        print("\n前几条 discharge capacity：")
        print(discharge_df[["op_index", "capacity"]].head())


if __name__ == "__main__":
    mat_file = r"E:\Reproduce\Dataset\BatteryAgingARC-FY08Q4\B0005.mat"   # 改成你的路径
    battery_id, meta_df, charge_records, discharge_df = load_nasa_battery_step1(mat_file)
    print_step1_summary(battery_id, meta_df, charge_records, discharge_df)

    # 可选保存
    meta_df.to_csv(f"{battery_id}_meta_step1.csv", index=False, encoding="utf-8-sig")
    discharge_df.to_csv(f"{battery_id}_discharge_step1.csv", index=False, encoding="utf-8-sig")

    print(meta_df.head(10))
    print(discharge_df.head(10))