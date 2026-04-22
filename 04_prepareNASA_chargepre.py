from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# 动态导入 01_loadNASA.py
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


def _smooth_voltage(v, window=11, polyorder=2):
    v = np.asarray(v, dtype=float).reshape(-1)
    n = len(v)

    if n < 5:
        return v.copy()

    w = min(window, n if n % 2 == 1 else n - 1)
    if w < 5:
        return v.copy()

    p = min(polyorder, w - 1)

    try:
        return savgol_filter(v, window_length=w, polyorder=p, mode="interp")
    except Exception:
        return v.copy()



def preprocess_one_charge_curve(time_arr, voltage_arr, current_charge_arr, v_grid):
    """
    只在 CC 段上处理充电曲线，并建立 t(v) 插值关系
    返回字段保持和 step5 兼容
    """
    t = np.asarray(time_arr, dtype=float).reshape(-1)
    v = np.asarray(voltage_arr, dtype=float).reshape(-1)
    ic = np.asarray(current_charge_arr, dtype=float).reshape(-1)

    n_raw = len(t)

    # 去掉非有限值
    mask = np.isfinite(t) & np.isfinite(v) & np.isfinite(ic)
    t = t[mask]
    v = v[mask]
    ic = ic[mask]

    if len(t) < 5:
        return {"ok": False, "reason": "too_few_points"}

    # 按时间排序
    order = np.argsort(t)
    t = t[order]
    v = v[order]
    ic = ic[order]

    # 去掉重复时间点
    _, uniq_idx = np.unique(t, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    t = t[uniq_idx]
    v = v[uniq_idx]
    ic = ic[uniq_idx]

    if len(t) < 5:
        return {"ok": False, "reason": "too_few_unique_time"}

    # 1) 只保留 CC 段
    # NASA 充电协议是 1.5A CC 到 4.2V，再进入 CV
    # 这里用宽松阈值截出 CC 段
    cc_mask = np.abs(ic) >= 1.40

    if cc_mask.sum() < 5:
        return {"ok": False, "reason": "too_few_cc_points"}

    t_cc = t[cc_mask]
    v_cc = v[cc_mask]

    if len(t_cc) < 5:
        return {"ok": False, "reason": "too_few_cc_points"}

    # 2) 平滑
    v_cc_smooth = _smooth_voltage(v_cc, window=11, polyorder=2)

    # 为了兼容旧字段，保留一个单调版本
    v_cc_mono = np.maximum.accumulate(v_cc_smooth)
    v_unique, unique_idx = np.unique(v_cc_mono, return_index=True)
    t_unique = t_cc[unique_idx]

    if len(v_unique) < 2:
        return {"ok": False, "reason": "too_few_unique_voltage"}

    # 3) 用“第一次越过电压阈值”的方式求 t(v)
    vmin = float(np.nanmin(v_cc_smooth))
    vmax = float(np.nanmax(v_cc_smooth))
    tol_v = 0.001
    t_grid = np.full_like(v_grid, np.nan, dtype=float)

    for m, vg in enumerate(v_grid):
        # 不在真实覆盖范围内，不插值
        if vg < (vmin - tol_v) or vg > (vmax + tol_v):
            continue

        hit = np.where(v_cc_smooth >= vg)[0]
        if len(hit) == 0:
            continue

        k = int(hit[0])

        if k == 0:
            t_grid[m] = t_cc[0]
        else:
            v1, v2 = v_cc_smooth[k - 1], v_cc_smooth[k]
            t1, t2 = t_cc[k - 1], t_cc[k]

            if np.isclose(v2, v1):
                t_grid[m] = t2
            else:
                t_grid[m] = t1 + (vg - v1) * (t2 - t1) / (v2 - v1)

    n_valid_grid = int(np.isfinite(t_grid).sum())
    if n_valid_grid == 0:
        return {"ok": False, "reason": "no_valid_grid"}

    has_full_38_42 = bool((vmin <= 3.8 + tol_v) and (vmax >= 4.2 - tol_v))

    return {
        "ok": True,

        # 与后续 step5 兼容
        "time_raw": t_cc,
        "voltage_raw": v_cc,
        "voltage_smooth": v_cc_smooth,
        "voltage_mono": v_cc_mono,
        "v_unique": v_unique,
        "t_unique": t_unique,
        "v_grid": v_grid,
        "t_grid": t_grid,

        # QC
        "n_raw": int(n_raw),
        "n_valid": int(len(t_cc)),
        "n_unique_voltage": int(len(v_unique)),
        "v_min_proc": vmin,
        "v_max_proc": vmax,
        "has_full_38_42": has_full_38_42,
        "n_valid_grid": n_valid_grid,
    }


def prepare_charge_curves_for_one_battery(mat_path, pair_csv_path, save=True):
    mat_path = _resolve_existing_path(
        Path(mat_path),
        extra_candidates=[
            Path(__file__).resolve().parent.parent / "Dataset" / "BatteryAgingARC-FY08Q4" / Path(mat_path).name,
            Path(__file__).resolve().parent.parent / "Dataset" / Path(mat_path).name,
        ],
    )
    pair_csv_path = _resolve_existing_path(Path(pair_csv_path))

    battery_id, meta_df, charge_records, _ = load_nasa_battery_step1(mat_path)

    pair_df = pd.read_csv(pair_csv_path, encoding="utf-8-sig")
    pair_df = pair_df.sort_values("charge_op_index").reset_index(drop=True)

    charge_map = {int(rec["op_index"]): rec for rec in charge_records}

    # 统一电压网格：3.800 ~ 4.200，步长 0.001
    v_grid = np.round(np.arange(3.8, 4.2001, 0.001), 4)

    proc_rows = []
    qc_rows = []

    for _, row in pair_df.iterrows():
        charge_op = int(row["charge_op_index"])
        rec = charge_map.get(charge_op, None)

        if rec is None:
            qc_rows.append({
                "battery_id": battery_id,
                "charge_op_index": charge_op,
                "discharge_op_index": int(row["discharge_op_index"]),
                "discharge_index": int(row["discharge_index"]),
                "soh": float(row["soh"]),
                "capacity": float(row["capacity"]),
                "ok": False,
                "reason": "charge_record_not_found"
            })
            continue
                # 先做一个最小物理合理性过滤，去掉明显坏样本
        t_raw = np.asarray(rec["time"], dtype=float).reshape(-1)
        v_raw = np.asarray(rec["voltage"], dtype=float).reshape(-1)
        ic_raw = np.asarray(rec["current_charge"], dtype=float).reshape(-1)

        mask_raw = np.isfinite(t_raw) & np.isfinite(v_raw) & np.isfinite(ic_raw)
        t_raw = t_raw[mask_raw]
        v_raw = v_raw[mask_raw]
        ic_raw = ic_raw[mask_raw]

        if len(t_raw) < 10:
            qc_rows.append({
                "battery_id": battery_id,
                "charge_op_index": charge_op,
                "discharge_op_index": int(row["discharge_op_index"]),
                "discharge_index": int(row["discharge_index"]),
                "soh": float(row["soh"]),
                "capacity": float(row["capacity"]),
                "ok": False,
                "reason": "raw_too_few_points"
            })
            continue

        vmin_raw = float(np.nanmin(v_raw))
        vmax_raw = float(np.nanmax(v_raw))
        t_end_raw = float(np.nanmax(t_raw))
        imax_raw = float(np.nanmax(np.abs(ic_raw)))

        # 只拦明显离谱的样本
        bad_raw = (
            (vmin_raw < 2.0) or
            (vmax_raw > 4.5) or
            (t_end_raw < 1000) or
            (imax_raw < 0.5)
        )

        if bad_raw:
            qc_rows.append({
                "battery_id": battery_id,
                "charge_op_index": charge_op,
                "discharge_op_index": int(row["discharge_op_index"]),
                "discharge_index": int(row["discharge_index"]),
                "soh": float(row["soh"]),
                "capacity": float(row["capacity"]),
                "ok": False,
                "reason": "raw_outlier"
                
            })
            continue
        proc = preprocess_one_charge_curve(
            rec["time"],
            rec["voltage"],
            rec["current_charge"],
            v_grid
        )

        qc = {
            "battery_id": battery_id,
            "charge_op_index": charge_op,
            "discharge_op_index": int(row["discharge_op_index"]),
            "discharge_index": int(row["discharge_index"]),
            "soh": float(row["soh"]),
            "capacity": float(row["capacity"]),
            "ok": proc["ok"],
            "reason": proc.get("reason", ""),
        }
        c0_val = float(row["C0"]) if "C0" in row.index else float(row["capacity"] / row["soh"])
        if proc["ok"]:
            proc_rows.append({
                "battery_id": battery_id,
                "charge_op_index": charge_op,
                "discharge_op_index": int(row["discharge_op_index"]),
                "discharge_index": int(row["discharge_index"]),
                "soh": float(row["soh"]),
                "capacity": float(row["capacity"]),
                "C0": c0_val,
                "time_raw": proc["time_raw"],
                "voltage_raw": proc["voltage_raw"],
                "voltage_smooth": proc["voltage_smooth"],
                "voltage_mono": proc["voltage_mono"],
                "v_unique": proc["v_unique"],
                "t_unique": proc["t_unique"],
                "v_grid": proc["v_grid"],
                "t_grid": proc["t_grid"],
            })

            qc.update({
                "n_raw": proc["n_raw"],
                "n_valid": proc["n_valid"],
                "n_unique_voltage": proc["n_unique_voltage"],
                "v_min_proc": proc["v_min_proc"],
                "v_max_proc": proc["v_max_proc"],
                "has_full_38_42": proc["has_full_38_42"],
                "n_valid_grid": proc["n_valid_grid"],
            })

        qc_rows.append(qc)

    proc_df = pd.DataFrame(proc_rows)
    qc_df = pd.DataFrame(qc_rows)

    if save:
        base_name = pair_csv_path.name
        if base_name.endswith("_pairprev_step3.csv"):
            stem = base_name.replace("_pairprev_step3.csv", "")
        elif base_name.endswith("_pair_step3.csv"):
            stem = base_name.replace("_pair_step3.csv", "")
        else:
            raise ValueError(f"无法识别 pair 文件名: {base_name}")

        out_pkl = pair_csv_path.with_name(f"{stem}_chargeproc_step4.pkl")
        out_qc = pair_csv_path.with_name(f"{stem}_chargeqc_step4.csv")

        proc_df.to_pickle(out_pkl)
        qc_df.to_csv(out_qc, index=False, encoding="utf-8-sig")

        print(f"已保存: {out_pkl}")
        print(f"已保存: {out_qc}")
    return proc_df, qc_df


if __name__ == "__main__":
    files = [
    (r"Dataset/BatteryAgingARC-FY08Q4/B0005.mat", r"B0005_pairprev_step3.csv"),
    (r"Dataset/BatteryAgingARC-FY08Q4/B0006.mat", r"B0006_pairprev_step3.csv"),
    (r"Dataset/BatteryAgingARC-FY08Q4/B0007.mat", r"B0007_pairprev_step3.csv"),
]

    for mat_file, pair_file in files:
        proc_df, qc_df = prepare_charge_curves_for_one_battery(mat_file, pair_file, save=True)

        print(f"\n===== {Path(mat_file).stem} =====")
        print("处理成功样本数 =", len(proc_df))
        print("QC 总样本数 =", len(qc_df))
        print("完整覆盖 3.8-4.2V 的样本数 =", int(qc_df["has_full_38_42"].eq(True).sum()))
        print(qc_df.head())
        if Path(mat_file).stem == "B0006":
            print("\nB0006 覆盖统计：")
            print(qc_df["has_full_38_42"].value_counts(dropna=False))
            print("\nB0006 失败原因统计：")
            print(qc_df["reason"].value_counts(dropna=False).head(20))

            bad = qc_df[(qc_df["ok"] == True) & (qc_df["has_full_38_42"] != True)].copy()
            print("\nB0006 不完整覆盖样本前10行：")
            print(bad[["charge_op_index", "v_min_proc", "v_max_proc", "n_valid_grid"]].head(10))