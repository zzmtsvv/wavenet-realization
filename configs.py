from dataclasses import dataclass
import torch


@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min : int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0
    pad_value: float = -11.5129251


@dataclass
class train_config:
    random_seed: int = 42
    device_str: str = 'cuda' if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    dataset_directory = 'LJSpeech-1.1'
    path2csv = "LJSpeech-1.1/metadata.csv"
    num_epochs = 5
    batch_size = 2
    lr = 3e-4
    model_in_channels = 256
    model_out_channels = 256
    model_condition_channels = 80
    model_upsample_factor = 256
    n_classes = 256
    mu_law_value: float = 256
    max_grad_norm = 1.0
    weight_decay = 1e-3
    num_workers = 2
    weights_path = 'model.pth'
