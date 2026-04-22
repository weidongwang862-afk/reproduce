import pandas as pd

from s08_run_single_model import run, build_model
from s08_config import get_config
from s08_data import load_npz_data, sanity_check
from s08_engine import train_one_model
from s08_utils import get_device, set_seed, ensure_dir


def main():
    cfg = get_config()
    set_seed(cfg.seed)

    data_dict = load_npz_data(cfg.data_path)
    sanity_check(data_dict)

    device = get_device(cfg.device)
    ensure_dir(cfg.output_root)

    model_names = ["lstm", "cnnlstm", "transformer"]
    rows = []

    for model_name in model_names:
        print(f"\n========== Running {model_name} ==========")
        model = build_model(model_name, cfg)
        out_dir = cfg.output_root / model_name
        summary, _ = train_one_model(
            model=model,
            model_name=model_name,
            data_dict=data_dict,
            cfg=cfg,
            device=device,
            out_dir=out_dir,
        )

        row = {
            "model": model_name,
            "val_MAE": summary["validation"]["MAE"],
            "val_RMSE": summary["validation"]["RMSE"],
            "val_R2": summary["validation"]["R2"],
            "B0006_MAE": summary["test_B0006"]["MAE"],
            "B0006_RMSE": summary["test_B0006"]["RMSE"],
            "B0006_R2": summary["test_B0006"]["R2"],
            "B0007_MAE": summary["test_B0007"]["MAE"],
            "B0007_RMSE": summary["test_B0007"]["RMSE"],
            "B0007_R2": summary["test_B0007"]["R2"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(cfg.output_root / "compare_baselines.csv", index=False, encoding="utf-8-sig")
    print("\n已保存 compare_baselines.csv")


if __name__ == "__main__":
    main()