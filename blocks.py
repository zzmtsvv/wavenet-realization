import torch
from torch import nn
from configs import MelSpectrogramConfig
from librosa.filters import mel
from torchaudio import transforms


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True):
        
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=bias)
        
        padding_size = (kernel_size - 1) * dilation
        self.zero_padding = nn.ConstantPad1d(padding=(padding_size, 0), value=0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded_input = self.zero_padding(x)
        return super().forward(padded_input)


class GatedConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int
    ) -> None:
        super().__init__()

        self.filter_conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.gate_conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        filter_out = self.filter_conv(x)
        gate_out = self.gate_conv(x)
        return torch.tanh(filter_out) * torch.sigmoid(gate_out)


class ConditionalGatedConv1d(GatedConv1d):
    def __init__(self, in_channels: int, out_channels: int, conditional_in_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__(in_channels, out_channels, kernel_size, dilation)

        self.conditional_conv1d = nn.Conv1d(
            in_channels=conditional_in_channels,
            out_channels=out_channels * 2,
            kernel_size=1)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) == condition.size(-1)

        filter_out = self.filter_conv(x)
        gate_out = self.gate_conv(x)

        cond_filter, cond_gate = torch.chunk(self.conditional_conv1d(condition), 2, dim=1)

        return torch.tanh(filter_out + cond_filter) * torch.sigmoid(gate_out + cond_gate)


class CondWaveNetBlock(nn.Module):
    def __init__(
        self,
        gated_in_channels: int,
        gated_out_channels: int,
        conditional_in_channels: int,
        skip_out_channels: int,
        kernel_size: int,
        dilation: int) -> None:

        super().__init__()

        self.gated_conv = ConditionalGatedConv1d(
            in_channels=gated_in_channels,
            out_channels=gated_out_channels,
            conditional_in_channels=conditional_in_channels,
            kernel_size=kernel_size,
            dilation=dilation)
        
        self.skip_conv = nn.Conv1d(gated_out_channels, skip_out_channels, kernel_size=1)
        self.residual_conv = nn.Conv1d(gated_out_channels, gated_in_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        gated_output = self.gated_conv(x, condition)

        residual_output = self.residual_conv(gated_output) + x
        skip_output = self.skip_conv(gated_output)

        return residual_output, skip_output


class MelSpectrogram(nn.Module):
    def __init__(self, config = MelSpectrogramConfig()) -> None:
        super().__init__()

        self.config = config

        self.mel_spectrogram = transforms.MelSpectrogram(
            sample_rate=config.sr,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels)
        self.mel_spectrogram.spectrogram.power = config.power

        mel_basis = mel(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.Tensor(mel_basis))
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        mel = self.mel_spectrogram(audio).clamp(min=1e-5).log()
        return mel
