import torch
from torch.utils.data import Dataset
from configs import MelSpectrogramConfig, train_config
from blocks import MelSpectrogram
import pandas as pd
import torchaudio
from pathlib import Path
import random


class LJSpeechDataset(Dataset):
    def __init__(
        self,
        path2csv: str = train_config.path2csv,
        mel_spec_config = MelSpectrogramConfig(),
        dataset_root: str = f"./{train_config.dataset_directory}",
        max_time_samples: int = 256 * 76,
        train: bool = True) -> None:

        super().__init__()
        
        self.path2csv = path2csv
        self.mel_spec_config = mel_spec_config
        self.dataset_root = Path(dataset_root)
        self.max_time_samples = max_time_samples
        self.train = train

        self.featurizer = MelSpectrogram(mel_spec_config)

        df = pd.read_csv(
            path2csv,
            names=['id', 'gt', 'gt_letters_only'],
            sep='|'
        )
        df = df.dropna()
        self.names = list(df['id'])
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, index):
        wav, sr = torchaudio.load(self.dataset_root / f'wavs/{self.names[index]}.wav')
        wav.squeeze_()

        if self.train:
            start_sample = random.randint(0, wav.size(-1) - self.max_time_samples)
            wav = wav[start_sample: start_sample + self.max_time_samples]
        return wav

