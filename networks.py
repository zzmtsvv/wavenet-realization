import torch
from torch import nn
from torch.nn import functional as F
from utils import OneHot, MuLaw
from configs import MelSpectrogramConfig
from blocks import CondWaveNetBlock
from tqdm.auto import tqdm


class CondNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        hop_size: int) -> None:

        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hop_size = hop_size

        self.net = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.upsampler = nn.Upsample(scale_factor=hop_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.input_size

        self.net.flatten_parameters()
        out, _ = self.net(x)
        out.transpose_(-1, -2)

        return self.upsampler(out)


class WaveNet(nn.Module):
    def __init__(
        self,
        config = MelSpectrogramConfig(),
        in_channels: int = 64,
        out_channels: int = 64,
        gate_channels: int = 64,
        residual_channels: int = 64,
        skip_channels: int = 64,
        head_channels: int = 64,
        condition_channels: int = 64,
        kernel_size: int = 2,
        dilation_cycles: int = 3,
        dilation_depth: int = 10,
        upsample_factor: int = 480,
        ) -> None:
        super().__init__()

        self.config = config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gate_channels = gate_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.head_channels = head_channels
        self.condition_channels = condition_channels
        self.kernel_size = kernel_size
        self.dilation_cycles = dilation_cycles
        self.dilation_depth = dilation_depth
        self.upsample_factor = upsample_factor

        self.cond = CondNet(config.n_mels, condition_channels, upsample_factor)
        self.stem = nn.Conv1d(in_channels, residual_channels, kernel_size=1)

        self.blocks = nn.ModuleList([
            CondWaveNetBlock(residual_channels, gate_channels, condition_channels,
                             skip_channels, kernel_size, 2 ** (i % dilation_depth))
            for i in range(dilation_cycles * dilation_depth)
        ])
        self.blocks[-1].residual_conv.requires_grad_(False)

        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(skip_channels, head_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(head_channels, out_channels, kernel_size=1))
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        condition = self.cond(condition)

        return self.forward_prop(x, condition)
    
    def forward_prop(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        stem_output = self.stem(x)
        accumulation = 0
        residual_output = stem_output
        for block in self.blocks:
            residual_output, skip_output = block(residual_output, condition)
            accumulation = accumulation + skip_output

        output = self.head(accumulation)
        return output
    
    def generate(self, condition: torch.Tensor, verbose: bool = True) -> torch.Tensor:
        mu_law = MuLaw().to(condition.device)

        compressed_samples = self.naive_generate(condition, verbose)

        return mu_law.decode(compressed_samples)

    @property
    def num_parameters(self) -> int:
        return sum([p.numel() for p in self.parameters()])

    @property
    def receptive_field(self) -> int:
        dilations =  [2 ** (i % self.dilation_depth) for i in range(self.dilation_cycles * self.dilation_depth)]
        receptive_field = (self.kernel_size - 1) * sum(dilations) + 1
        return receptive_field

    @torch.no_grad()
    def naive_generate(self, condition: torch.Tensor, verbose: bool) -> torch.Tensor:
        one_hot = OneHot()

        required_num_samples = condition.shape[1] * self.upsample_factor
        generated_samples = torch.Tensor(1, 1, self.receptive_field + required_num_samples).fill_(self.in_channels // 2).to(condition.device)

        condition = F.pad(self.cond(condition), (self.receptive_field, 0), mode='replicate')
        
        iterator = range(required_num_samples)
        if verbose:
            iterator = tqdm(iterator)
        
        for i in iterator:
            current_condition = condition[:, :, i:i + self.receptive_field]
            current_samples = generated_samples[:, :, i:i + self.receptive_field]
            current_one_hot_samples = one_hot(current_samples.long())

            current_output = self.forward_prop(current_one_hot_samples, current_condition)
            last_logits = current_output[:, :, -1].squeeze()

            samples = torch.distributions.Categorical(logits=last_logits)
            new_sample = samples.sample(torch.Size([1]))
            generated_samples[:, :, i + self.receptive_field] = new_sample
        
        return generated_samples.squeeze()[-required_num_samples:]

