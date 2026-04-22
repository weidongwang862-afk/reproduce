from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class TrainConfig:
    data_path = Path("NASA_model_input_step7d_finegrid17_current.npz")
    output_root: Path = Path("results_s08")

    seed: int = 42
    multi_seeds: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])

    epochs: int = 1000
    lr: float = 0.01
    batch_size: int = 128
    weight_decay: float = 0.0

    use_scheduler: bool = False
    use_early_stopping: bool = False
    patience: int = 100

    loss_name: str = "mse"
    device: str = "auto"

    input_dim: int = 1
    seq_len: int = 8
    output_dim: int = 1

    d_model: int = 32
    dim_ff: int = 64
    nhead: int = 2
    num_blocks: int = 2

    lstm_hidden_dim: int = 64
    lstm_num_layers: int = 2

    cnn_out_channels: int = 8
    cnn_kernel_size: int = 8
    cnn_pool_size: int = 2

    pyraformer_num_levels: int = 3
    pyraformer_readout: str = "last_fine"


def get_config() -> TrainConfig:
    return TrainConfig()