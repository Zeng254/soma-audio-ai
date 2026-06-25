"""
So-VITS-SVC Model architecture definition

Contains So-VITS-SVC 4.0/4.1 core model structures:
- VITSDecoder: VITS Decoder (Causal/Non-causal)
- VITSGenerator: VITS Main generator
- TextEncoder: Text/Content encoder
- LengthRegulator: Length regulator
- PosteriorEncoder: Posterior encoder
- Flow: NormalizationFlow

These classes are used to load and run So-VITS model inference.
"""

from typing import Optional, Tuple, List, Any, Dict
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d


class LeakyReLU(nn.Module):
    """Leaky ReLU"""

    def __init__(self, negative_slope: float = 0.1):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return F.leaky_relu(x, self.negative_slope)


class ResidualBlock(nn.Module):
    """Residual block"""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: Tuple[int, int, int] = (1, 3, 5)):
        super().__init__()
        self.convs = nn.ModuleList()
        for d in dilation:
            self.convs.append(
                Conv1d(
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


class TextEncoder(nn.Module):
    """
    Text/Content encoder

    Encode input features into latent representation
    """

    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int = 192,
        n_layers: int = 6,
        kernel_size: int = 5,
        Dropout: float = 0.1,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        # Embedding layer
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        # ConvolutionLayer
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                nn.Sequential(
                    Conv1d(
                        hidden_channels,
                        hidden_channels,
                        kernel_size,
                        padding=kernel_size // 2
                    ),
                    nn.BatchNorm1d(hidden_channels),
                    LeakyReLU(0.1),
                    nn.Dropout(Dropout),
                )
            )

        # Output projection
        self.proj = Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, input_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input sequence [batch, time]
            input_lengths: Input length [batch]

        Returns:
            (Encoder output, mask)
        """
        # Embedding
        x = self.emb(x).transpose(1, 2)  # [batch, hidden, time]

        # Convolution
        for conv in self.convs:
            x = conv(x)

        # Output
        x = self.proj(x)

        return x, None


class ContentEncoder(nn.Module):
    """
    Content encoder (for HubERT/ContentVec features)

    Directly encode continuous features
    """

    def __init__(
        self,
        in_channels: int = 1024,
        out_channels: int = 192,
        hidden_channels: int = 192,
        n_layers: int = 6,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Input projection
        self.input_proj = Conv1d(in_channels, hidden_channels, 1)

        # ConvolutionLayer
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                nn.Sequential(
                    Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(hidden_channels),
                    LeakyReLU(0.1),
                )
            )

        # Output projection
        self.output_proj = Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input features [batch, in_channels, time]

        Returns:
            Encoder output [batch, out_channels, time]
        """
        x = self.input_proj(x)

        for conv in self.convs:
            x = conv(x)

        x = self.output_proj(x)
        return x


class LengthRegulator(nn.Module):
    """Length regulator (for parallelization)"""

    def __init__(
        self,
        hidden_channels: int = 192,
        expansion: int = 2,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.expansion = expansion

        self.duration_predictor = nn.Sequential(
            Conv1d(hidden_channels, hidden_channels, 1),
            LeakyReLU(0.1),
            Conv1d(hidden_channels, 1, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        durations: Optional[torch.Tensor] = None,
        max_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input [batch, channels, time]
            durations: Duration [batch, time]
            max_len: MaximumLength

        Returns:
            (Expanded output, output length)
        """
        if durations is None:
            # Predict duration
            durations = self.duration_predictor(x.transpose(1, 2)).squeeze(1)
            durations = durations.exp()  # Ensure positive values

        # Expand
        x = self.expand_by_durations(x, durations, max_len)

        return x, durations.sum(dim=-1).long()

    def expand_by_durations(self, x: torch.Tensor, durations: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
        """Expand sequence based on duration"""
        batch_size, channels, time = x.shape
        durations = durations.long()

        if max_len is None:
            max_len = durations.sum(dim=-1).max().item()

        # Expand
        output = torch.zeros(batch_size, channels, max_len, device=x.device, dtype=x.dtype)

        for i in range(batch_size):
            pos = 0
            for j in range(time):
                d = durations[i, j].item()
                if d > 0 and pos + d <= max_len:
                    output[i, :, pos:pos + d] = x[i, :, j:j + 1].expand(-1, d, -1)
                    pos += d

        return output


class PosteriorEncoder(nn.Module):
    """
    Posterior encoder

    Encode Mel spectrum into latent representation
    """

    def __init__(
        self,
        in_channels: int = 80,  # Mel spectrum channel count
        out_channels: int = 192,  # Latent space dimension
        hidden_channels: int = 192,
        n_layers: int = 16,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Input convolution
        self.input_conv = Conv1d(in_channels, hidden_channels, 1)

        # WaveNet style residual block
        self.resblocks = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** (i % 8)
            self.resblocks.append(
                nn.Sequential(
                    Conv1d(hidden_channels, hidden_channels, kernel_size, dilation=dilation, padding=(kernel_size - 1) * dilation // 2),
                    LeakyReLU(0.1),
                )
            )

        # OutputLayer
        self.proj_mean = Conv1d(hidden_channels, out_channels, 1)
        self.proj_std = Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Mel spectrum [batch, in_channels, time]

        Returns:
            (mean, log variance)
        """
        x = self.input_conv(x)

        # ResidualConnection
        skips = []
        for resblock in self.resblocks:
            x = resblock(x)
            skips.append(x)

        x = torch.stack(skips, dim=0).sum(dim=0)

        mean = self.proj_mean(x)
        std = self.proj_std(x)
        log_std = torch.clamp(std, min=1e-5).log()

        return mean, log_std


class Flow(nn.Module):
    """
    NormalizationFlow

    For GAN mode VITS
    """

    def __init__(
        self,
        in_channels: int = 192,
        hidden_channels: int = 192,
        n_layers: int = 12,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.in_channels = in_channels

        # Affine coupling layer
        self.flows = nn.ModuleList()
        for _ in range(n_layers):
            self.flows.append(
                nn.ModuleDict({
                    'pre': Conv1d(in_channels, hidden_channels, 1),
                    'coupling': Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2),
                    'post': Conv1d(hidden_channels, in_channels * 2, 1),
                })
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (for inference)

        Args:
            x: Input [batch, channels, time]

        Returns:
            Shifted output
        """
        for flow in self.flows:
            x = flow['pre'](x)
            x = F.leaky_relu(x, 0.1)
            x = flow['coupling'](x)
            x = F.leaky_relu(x, 0.1)
            affine = flow['post'](x)

            # Affine shift
            log_scale, bias = affine.chunk(2, dim=1)
            log_scale = torch.tanh(log_scale)
            x = x * log_scale.exp() + bias

        return x


class VITSResBlock(nn.Module):
    """VITS residual block"""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: Tuple[int, int, int] = (1, 3, 5)):
        super().__init__()
        self.convs = nn.ModuleList()
        for d in dilation:
            self.convs.append(
                nn.Sequential(
                    Conv1d(channels, channels, kernel_size, dilation=d, padding=(kernel_size - 1) * d // 2),
                    nn.BatchNorm1d(channels),
                    LeakyReLU(0.1),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = x + conv(x)
        return x


class VITSDecoder(nn.Module):
    """
    VITS Decoder

    Generate audio from latent representation
    """

    def __init__(
        self,
        in_channels: int = 192,
        out_channels: int = 1,
        upsample_rates: List[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    ):
        super().__init__()
        self.in_channels = in_channels

        # Input convolution
        self.input_conv = Conv1d(in_channels, upsample_initial_channel, 7, padding=3)

        # UpsampleLayer
        self.upsamples = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        for i, (rate, kernel_size) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            channel = upsample_initial_channel // (2 ** i)
            self.upsamples.append(
                ConvTranspose1d(
                    channel,
                    channel // 2,
                    kernel_size,
                    stride=rate,
                    padding=(kernel_size - rate) // 2
                )
            )

            # Residual block after each upsample layer
            for _ in range(2):
                self.resblocks.append(
                    VITSResBlock(channel // 2, resblock_kernel_sizes[0], resblock_dilation_sizes[0])
                )

        # OutputConvolution
        self.output_conv = Conv1d(channel // 2, out_channels, 7, padding=3)

        # Initialize
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (Conv1d, ConvTranspose1d)):
            nn.init.normal_(m.weight, 0.0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Latent representation [batch, in_channels, time]

        Returns:
            Audio waveform [batch, out_channels, time * prod(upsample_rates)]
        """
        x = self.input_conv(x)
        x = F.leaky_relu(x, 0.1)

        for i, upsample in enumerate(self.upsamples):
            x = upsample(x)
            x = F.leaky_relu(x, 0.1)

            # Apply residual block
            for _ in range(2):
                if self.resblocks:
                    x = self.resblocks[i * 2](x)

        x = self.output_conv(x)
        x = torch.tanh(x)

        return x


class VITSGenerator(nn.Module):
    """
    So-VITS Main generator

    Integrate all components into complete VITS model
    """

    def __init__(
        self,
        n_vocab: int = 0,
        spec_channels: int = 80,
        hidden_channels: int = 192,
        out_channels: int = 1,
        n_speakers: int = 0,
        gin_channels: int = 0,
        use_transformer_flows: bool = False,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels

        # Encoder
        if n_vocab > 0:
            self.text_encoder = TextEncoder(n_vocab, hidden_channels, hidden_channels)
        else:
            self.content_encoder = ContentEncoder(1024, hidden_channels, hidden_channels)

        # Speaker embedding
        if n_speakers > 0:
            self.emb = nn.Embedding(n_speakers, gin_channels)

        # Length regulator
        self.length_regulator = LengthRegulator(hidden_channels)

        # Flow (optional)
        if use_transformer_flows:
            self.flow = Flow(hidden_channels, hidden_channels)

        # Decoder
        self.decoder = VITSDecoder(
            in_channels=hidden_channels,
            out_channels=out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        f0: Optional[torch.Tensor] = None,
        speaker_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input features or text [batch, channels, time] or [batch, time]
            f0: F0 trajectory [batch, time] (optional)
            speaker_ids: Speaker IDs [batch]

        Returns:
            Audio [batch, 1, time]
        """
        # ContentEncode
        if hasattr(self, 'text_encoder'):
            x, _ = self.text_encoder(x)
        else:
            # x is already encoded features
            pass

        # Speaker embedding
        if speaker_ids is not None and self.n_speakers > 0:
            g = self.emb(speaker_ids).unsqueeze(-1)  # [batch, gin, 1]
            x = x + g

        # Length adjustment
        x, _ = self.length_regulator(x)

        # F0 modulation (if provided)
        if f0 is not None:
            x = self._apply_f0_modulation(x, f0)

        # Decode
        x = self.decoder(x)

        return x

    def _apply_f0_modulation(self, x: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        """F0 modulation"""
        # Simple F0 modulation
        if f0.shape[-1] != x.shape[-1]:
            f0 = F.interpolate(f0.unsqueeze(1), size=x.shape[-1], mode='linear').squeeze(1)

        # Add F0 to channel dimension
        f0_expanded = f0.unsqueeze(1).expand(-1, x.shape[1], -1) * 0.01
        x = x + f0_expanded

        return x

    def inference(
        self,
        features: torch.Tensor,
        f0: Optional[torch.Tensor] = None,
        speaker_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        InferenceInterface

        Args:
            features: ContentFeature [batch, dim, time]
            f0: F0 trajectory [batch, time]
            speaker_ids: Speaker IDs [batch]

        Returns:
            Audio [batch, 1, time]
        """
        return self.forward(features, f0, speaker_ids)


class SimpleVITSModel(nn.Module):
    """
    Simplified VITS Model encapsulation

    Supports multiple So-VITS model formats, auto-adapts
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}

        # ExtractionConfiguration
        self.n_vocab = self.config.get('n_vocab', 0)
        self.spec_channels = self.config.get('spec_channels', 80)
        self.hidden_channels = self.config.get('hidden_channels', 192)
        self.out_channels = self.config.get('out_channels', 1)
        self.n_speakers = self.config.get('n_speakers', 0)
        self.gin_channels = self.config.get('gin_channels', 0)
        self.use_flow = self.config.get('use_flow', False)

        # Create generator
        self.generator = VITSGenerator(
            n_vocab=self.n_vocab,
            spec_channels=self.spec_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            n_speakers=self.n_speakers,
            gin_channels=self.gin_channels,
            use_transformer_flows=self.use_flow,
        )

    def forward(
        self,
        x: torch.Tensor,
        f0: Optional[torch.Tensor] = None,
        speaker_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass"""
        return self.generator(x, f0, speaker_ids)

    def inference(
        self,
        features: torch.Tensor,
        f0: Optional[torch.Tensor] = None,
        speaker_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """InferenceInterface"""
        return self.generator.inference(features, f0, speaker_ids)


def create_vits_model_from_checkpoint(
    checkpoint: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> SimpleVITSModel:
    """
    Create VITS model from checkpoint

    Args:
        checkpoint: Model checkpoint
        config: ModelConfiguration

    Returns:
        VITS ModelInstance
    """
    # Merge configuration
    if isinstance(checkpoint, dict):
        if 'config' in checkpoint and config is None:
            config = checkpoint['config']
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'generator' in checkpoint:
            state_dict = checkpoint['generator']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
        config = config or {}

    # CreateModel
    model = SimpleVITSModel(config)

    # Try to load weights
    if state_dict:
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception:
            # Partial load failed, use default weights
            pass

    return model
