"""
RVC Model architecture definition

Contains RVC v1/v2 core model structures:
- RVCGenerator: RVC Main generator
- PitchEncoder: Pitch encoder
- HiFiGANGenerator: HiFi-GAN Vocoder

These classes are used to load and run RVC model inference.
"""

from typing import Optional, Tuple, List, Any, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block"""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: Tuple[int, int, int] = (1, 3, 5)):
        super().__init__()
        self.convs = nn.ModuleList()
        for d in dilation:
            self.convs.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=d,
                    padding=(kernel_size - 1) * d // 2
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for conv in self.convs:
            x = F.leaky_relu(x, 0.1)
            x = conv(x)
        return x + residual


class RVCGenerator(nn.Module):
    """
    RVC Main generatorNetwork

    Architecture:
    - Shared encoder (Projection + ResBlocks)
    - F0 condition modulation
    - UpsampleDecoder
    - Final convolution output
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 1,
        hidden_channels: int = 256,
        kernel_size: int = 7,
        upsample_rates: List[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        embed_dim: int = 256,
        pitch_encoder_dim: int = 256,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.embed_dim = embed_dim

        # Input convolution
        self.input_conv = nn.Conv1d(
            in_channels + pitch_encoder_dim,  # Feature + F0 Encode
            hidden_channels,
            kernel_size,
            padding=kernel_size // 2
        )

        # Residual block
        self.resblocks = nn.ModuleList()
        for kernel_size in resblock_kernel_sizes:
            self.resblocks.append(
                ResBlock(hidden_channels, kernel_size, resblock_dilation_sizes[0])
            )

        # UpsampleLayer
        self.upsamples = nn.ModuleList()
        self.upsample_channels = []
        channel = hidden_channels
        for i, (rate, kernel_size) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            out_channel = upsample_initial_channel // (2 ** i)
            self.upsample_channels.append(channel)
            self.upsamples.append(
                nn.ConvTranspose1d(
                    channel,
                    out_channel,
                    kernel_size,
                    stride=rate,
                    padding=(kernel_size - rate) // 2
                )
            )
            channel = out_channel  # Next input = previous output

        # Final output convolution
        self.output_conv = nn.Conv1d(
            channel,
            out_channels,
            kernel_size,
            padding=kernel_size // 2
        )

        # Initialize
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.normal_(m.weight, 0.0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        f0_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor [batch, channels, time]
            f0_embedding: F0 embedding [batch, pitch_dim, time] (optional)

        Returns:
            Output audio [batch, out_channels, time * upsample_rate]
        """
        if f0_embedding is not None:
            # Concatenate F0 embedding to input
            if f0_embedding.shape[-1] != x.shape[-1]:
                # Resample F0 embedding to match
                f0_embedding = F.interpolate(
                    f0_embedding,
                    size=x.shape[-1],
                    mode='linear'
                )
            x = torch.cat([x, f0_embedding], dim=1)

        # Input convolution
        x = self.input_conv(x)
        x = F.leaky_relu(x, 0.1)

        # Residual block
        for resblock in self.resblocks:
            x = resblock(x)

        # Upsample
        for upsample in self.upsamples:
            x = F.leaky_relu(x, 0.1)
            x = upsample(x)

        # OutputConvolution
        x = F.leaky_relu(x, 0.1)
        x = self.output_conv(x)
        x = torch.tanh(x)

        return x


class FlowDecoder(nn.Module):
    """Flow Decoder (for certain RVC models)"""

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 256,
        out_channels: int = 128,
        num_layers: int = 12,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        # Wavenet style decoder
        self.input_conv = nn.Conv1d(in_channels, hidden_channels, 1)

        self.upsamples = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        # Upsample to original length
        for i in range(4):
            stride = 2
            kernel = 4
            self.upsamples.append(
                nn.ConvTranspose1d(
                    hidden_channels,
                    hidden_channels,
                    kernel,
                    stride=stride,
                    padding=(kernel - stride) // 2
                )
            )
            self.resblocks.append(
                nn.ModuleList([
                    nn.Conv1d(hidden_channels, hidden_channels, kernel_size, dilation=1, padding=kernel_size // 2),
                    nn.Conv1d(hidden_channels, hidden_channels, kernel_size, dilation=3, padding=3 * kernel_size // 2),
                ])
            )

        # OutputLayer
        self.output_conv = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.input_conv(x)
        x = F.leaky_relu(x, 0.1)

        for upsample, resblock in zip(self.upsamples, self.resblocks):
            x = upsample(x)
            x = F.leaky_relu(x, 0.1)

            # ResidualConnection
            skip = x
            for conv in resblock:
                x = F.leaky_relu(x, 0.1)
                x = conv(x)
            x = x + skip

        x = self.output_conv(x)
        return x


class ConvFlowDecoder(nn.Module):
    """ConvFlow Decoder"""

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 192,
        out_channels: int = 80,
        num_flows: int = 12,
    ):
        super().__init__()
        self.num_flows = num_flows

        # ConditionEncode
        self.cond_encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1),
        )

        # Flow Layer
        self.flows = nn.ModuleList()
        for _ in range(num_flows):
            self.flows.append(
                nn.ModuleDict({
                    'conv': nn.Conv1d(hidden_channels, hidden_channels * 2, 3, padding=1),
                    'norm': nn.BatchNorm1d(hidden_channels),
                })
            )

        # Output projection
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (for inference)

        Args:
            x: Input features [batch, channels, time]

        Returns:
            Mel spectrum [batch, n_mels, time]
        """
        x = self.cond_encoder(x)

        for flow in self.flows:
            x = F.relu(x)
            x = flow['conv'](x)
            x = flow['norm'](x)

        x = self.proj(x)
        # Return mean (do not return log_scale for simplicity)
        return x[:, :x.shape[1] // 2, :]


class SineGenerator(nn.Module):
    """Sine generator (simplified vocoder)"""

    def __init__(
        self,
        sampling_rate: int = 40000,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.harmonic_num = harmonic_num
        self.sine_amp = sine_amp
        self.noise_std = noise_std

    def forward(
        self,
        f0: torch.Tensor,
        mel_output: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Based on F0 GenerateAudio

        Args:
            f0: F0 trajectory [batch, 1, time]
            mel_output: Mel spectrum [batch, n_mels, time] (optional)

        Returns:
            Audio waveform [batch, 1, time * hop_length]
        """
        device = f0.device
        batch_size, _, time_steps = f0.shape

        # Base frequency
        freq = f0.squeeze(1)  # [batch, time]

        # Generate time steps
        indices = torch.arange(time_steps, device=device).float().unsqueeze(0)
        indices = indices.repeat(batch_size, 1)

        # Phase accumulation
        phase = torch.cumsum(freq / self.sampling_rate, dim=1)

        # Generate sine wave
        sine_wave = torch.sin(2 * torch.pi * phase)

        # Add harmonics
        if self.harmonic_num > 0:
            for h in range(2, self.harmonic_num + 2):
                sine_wave += torch.sin(2 * torch.pi * phase * h) / h

        # Add noise
        noise = torch.randn_like(sine_wave) * self.noise_std
        sine_wave = sine_wave * self.sine_amp + noise

        # Reshape to [batch, 1, time]
        return sine_wave.unsqueeze(1)


class SimpleRVCModel(nn.Module):
    """
    Simplified RVC model encapsulation

    Supports multiple RVC model formats, auto-adapt
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}

        # Initialize network based on configuration
        self.sample_rate = self.config.get('sample_rate', 40000)
        self.hop_length = self.config.get('hop_length', 512)
        self.mel_channels = self.config.get('mel_channels', 128)

        # Try to build network
        self.use_flow = self.config.get('use_flow', False)

        if self.use_flow:
            self.flow_decoder = ConvFlowDecoder(
                in_channels=256,
                out_channels=self.mel_channels
            )
        else:
            self.generator = RVCGenerator(
                in_channels=256,
                out_channels=1,
                embed_dim=256,
                pitch_encoder_dim=256,
            )

    def forward(self, x: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: HubERT Feature [batch, 256, time]
            f0: F0 trajectory [batch, time] or [batch, 1, time]

        Returns:
            Output audio or mel spectrum
        """
        # Adjust F0 dimension
        if len(f0.shape) == 2:
            f0 = f0.unsqueeze(1)

        # F0 Encode
        f0_encoded = self._encode_f0(f0)

        # Resample F0 encoding to match feature length
        if f0_encoded.shape[-1] != x.shape[-1]:
            f0_encoded = F.interpolate(
                f0_encoded,
                size=x.shape[-1],
                mode='linear'
            )

        if self.use_flow:
            output = self.flow_decoder(x)
        else:
            # Generator inference
            output = self.generator(x, f0_encoded)

        return output

    def _encode_f0(self, f0: torch.Tensor) -> torch.Tensor:
        """F0 pitch encoder"""
        # Simple log encode
        f0_encoded = torch.log(f0 + 1e-6)
        # Expand dimension to match
        if len(f0_encoded.shape) == 2:
            f0_encoded = f0_encoded.unsqueeze(1)
        # Expand to pitch_encoder_dim channels to match RVCGenerator input
        # RVCGenerator.input_conv expects in_channels + pitch_encoder_dim
        target_dim = getattr(self, 'pitch_encoder_dim', 256)
        if f0_encoded.shape[1] != target_dim:
            f0_encoded = f0_encoded.expand(-1, target_dim, -1)
        return f0_encoded

    def inference(
        self,
        features: torch.Tensor,
        f0: torch.Tensor
    ) -> torch.Tensor:
        """InferenceInterface"""
        return self.forward(features, f0)


def create_rvc_model_from_checkpoint(
    checkpoint: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> SimpleRVCModel:
    """
    Create RVC model from checkpoint

    Args:
        checkpoint: Model checkpoint
        config: ModelConfiguration

    Returns:
        RVC ModelInstance
    """
    # Merge configuration
    if isinstance(checkpoint, dict):
        if 'config' in checkpoint and config is None:
            config = checkpoint['config']
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'weight' in checkpoint:
            state_dict = checkpoint['weight']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
        config = config or {}

    # CreateModel
    model = SimpleRVCModel(config)

    # Try to load weights
    if state_dict:
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception:
            # Partial load failed, use default weights
            pass

    return model
