from networks import WaveNet
from utils import MuLaw, OneHot
from blocks import MelSpectrogram
from configs import train_config
import torch
from torch import nn
from dataset import LJSpeechDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.amp import autocast


class AverageMeter:
    def __init__(self) -> None:
        self.reset()
    
    def reset(self):
        self.val = 0
        self.average = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.average = self.sum / self.count


class Trainer:
    def __init__(self) -> None:

        self.model = WaveNet(
            in_channels=train_config.model_in_channels,
            out_channels=train_config.model_out_channels,
            condition_channels=train_config.model_condition_channels,
            upsample_factor=train_config.model_upsample_factor
        ).to(train_config.device)
        
        self.mu_law = MuLaw()
        self.one_hot = OneHot()
        self.mel_specer = MelSpectrogram().to(train_config.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
        self.criterion = nn.CrossEntropyLoss().to(train_config.device)

        self.train_dataset = LJSpeechDataset()
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=train_config.batch_size)
        self.val_dataset = LJSpeechDataset(train=False)
    
    def fit(self):
        for epoch in range(1, train_config.num_epochs + 1):

            train_epoch_loss = AverageMeter()
            self.model.train()

            for i, samples in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):
                self.optimizer.zero_grad()

                samples = samples.to(train_config.device)
                
                with autocast(device_type=train_config.device_str):
                    mels = self.mel_specer(samples)[..., :-1]
                    categorical_samples = self.mu_law(samples).unsqueeze(1)
                    one_hot_samples = self.one_hot(categorical_samples)
                    prediction = self.model(one_hot_samples, mels.transpose(-1, -2))
                    loss = self.criterion(prediction[:, :, :-1], categorical_samples.squeeze(1)[:, 1:])
                
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), train_config.max_grad_norm)
                self.optimizer.step()

                if not (i % 50):
                    train_epoch_loss.update(loss.item())
                    print(f"Loss: {loss.item()}")

        self.save()

    def save(self):
        torch.save(self.model.state_dict(), train_config.weights_path)
