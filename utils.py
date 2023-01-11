import torch
from torch import nn
import numpy as np
import random
import os
from configs import train_config


def seed_everything(seed: int = train_config.random_seed) -> None:
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class OneHot(nn.Module):
    def __init__(self, n_class: int = train_config.n_classes) -> None:
        super().__init__()

        self.n_class = n_class
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3
        return self.encode(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.zeros(x.size(0), self.n_class, x.size(-1), device=x.device)
        output.scatter_(1, x, 1)
        return output

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(x, dim=1)


class MuLaw(nn.Module):
    def __init__(self, mu: float = train_config.mu_law_value) -> None:
        super().__init__()
        self.register_buffer('mu', torch.FloatTensor([mu - 1]))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, -1 + 1e-5, 1 - 1e-5)
        
        mu_law = torch.sign(x) * torch.log1p(self.mu * torch.abs(x)) / torch.log1p(self.mu)
        mu_law = (mu_law + 1) / 2  # [-1; 1] to [0; 1]
        
        return torch.floor(mu_law * self.mu + 0.5).long()

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        out = (x.float() / self.mu) * 2 - 1
        out = (torch.sign(out) / self.mu) * ((1 + self.mu) ** torch.abs(out) - 1)
        return out

