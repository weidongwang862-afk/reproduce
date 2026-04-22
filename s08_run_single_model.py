from pathlib import Path

from s08_config import get_config
from s08_data import load_npz_data, sanity_check
from s08_engine import train_one_model
from s08_model_lstm import LSTMRegressor
from s08_model_cnnlstm import CNNLSTMRegressor
from s08_model_transformer import TransformerRegressor
from s08_model_pyraformer import PyraformerRegressor
from s08_utils import get_device, set_seed, ensure_dir


def build_model(model_name: str, cfg):
    model_name = model_name.lower()

    if model_name == "lstm":
        return LSTMRegressor(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.lstm_hidden_dim,
            num_layers=cfg.lstm_num_layers,
            output_dim=cfg.output_dim,
        )

    if model_name == "cnnlstm":
        return CNNLSTMRegressor(
            input_dim=cfg.input_dim,
            conv_out_channels=cfg.cnn_out_channels,
            conv_kernel_size=cfg.cnn_kernel_size,
            pool_size=cfg.cnn_pool_size,
            hidden_dim=cfg.lstm_hidden_dim,
            num_layers=cfg.lstm_num_layers,
            output_dim=cfg.output_dim,
        )

    if model_name == "transformer":
        return TransformerRegressor(
            input_dim=cfg.input_dim,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_ff=cfg.dim_ff,
            num_blocks=cfg.num_blocks,
            output_dim=cfg.output_dim,
        )

    if model_name == "pyraformer":
        return PyraformerRegressor(
            input_dim=cfg.input_dim,
            d_model=cfg.d_model,
            dim_ff=cfg.dim_ff,
            nhead=cfg.nhead,
            num_blocks=cfg.num_blocks,
            num_levels=cfg.pyraformer_num_levels,
            output_dim=cfg.output_dim,
        )

    raise ValueError(f"未知模型名: {model_name}")


def run(model_name: str):
    cfg = get_config()
    set_seed(cfg.seed)

    data_dict = load_npz_data(cfg.data_path)
    sanity_check(data_dict)

    device = get_device(cfg.device)
    print("device =", device)

    model = build_model(model_name, cfg)
    out_dir = cfg.output_root / model_name
    ensure_dir(out_dir)

    summary, _ = train_one_model(
        model=model,
        model_name=model_name,
        data_dict=data_dict,
        cfg=cfg,
        device=device,
        out_dir=out_dir,
    )

    print("\n===== Final Summary =====")
    print(summary)


if __name__ == "__main__":
    # 先手动改这里调试
    run("lstm")