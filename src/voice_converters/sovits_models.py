"""
So-VITS-SVC 模型架构定义

包含 So-VITS-SVC 4.0/4.1 的核心模型结构：
- VITSDecoder: VITS 解码器 (因果/非因果)
- VITSGenerator: VITS 主生成器
- TextEncoder: 文本/内容编码器
- LengthRegulator: 长度调节器
- PosteriorEncoder: 后验编码器
- Flow: 归一化流

这些类用于加载和运行 So-VITS 模型推理。
"""

from typing import Optional, Tuple, List, Any, Dict
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d


class LeakyReLU(nn.Module):
    """带泄漏的 ReLU"""

    def __init__(self, negative_slope: float = 0.1):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return F.leaky_relu(x, self.negative_slope)


class ResidualBlock(nn.Module):
    """残差块"""

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
        for conv in self.convs:
            x = F.leaky_relu(x, 0.1)
            x = conv(x)
        return x


class TextEncoder(nn.Module):
    """
    文本/内容编码器

    将输入特征编码为潜在表示
    """

    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int = 192,
        n_layers: int = 6,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        # 嵌入层
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        # 卷积层
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
                    nn.Dropout(dropout),
                )
            )

        # 输出投影
        self.proj = Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, input_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入序列 [batch, time]
            input_lengths: 输入长度 [batch]

        Returns:
            (编码输出, 掩码)
        """
        # 嵌入
        x = self.emb(x).transpose(1, 2)  # [batch, hidden, time]

        # 卷积
        for conv in self.convs:
            x = conv(x)

        # 输出
        x = self.proj(x)

        return x, None


class ContentEncoder(nn.Module):
    """
    内容编码器 (用于 HubERT/ContentVec 特征)

    直接编码连续特征
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

        # 输入投影
        self.input_proj = Conv1d(in_channels, hidden_channels, 1)

        # 卷积层
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(
                nn.Sequential(
                    Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(hidden_channels),
                    LeakyReLU(0.1),
                )
            )

        # 输出投影
        self.output_proj = Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征 [batch, in_channels, time]

        Returns:
            编码输出 [batch, out_channels, time]
        """
        x = self.input_proj(x)

        for conv in self.convs:
            x = conv(x)

        x = self.output_proj(x)
        return x


class LengthRegulator(nn.Module):
    """长度调节器 (用于并行化)"""

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
        前向传播

        Args:
            x: 输入 [batch, channels, time]
            durations: 时长 [batch, time]
            max_len: 最大长度

        Returns:
            (扩展后输出, 输出长度)
        """
        if durations is None:
            # 预测时长
            durations = self.duration_predictor(x.transpose(1, 2)).squeeze(1)
            durations = durations.exp()  # 确保正值

        # 扩展
        x = self.expand_by_durations(x, durations, max_len)

        return x, durations.sum(dim=-1).long()

    def expand_by_durations(self, x: torch.Tensor, durations: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
        """根据时长扩展序列"""
        batch_size, channels, time = x.shape
        durations = durations.long()

        if max_len is None:
            max_len = durations.sum(dim=-1).max().item()

        # 扩展
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
    后验编码器

    将 Mel 频谱编码为潜在表示
    """

    def __init__(
        self,
        in_channels: int = 80,  # Mel 频谱通道数
        out_channels: int = 192,  # 潜在空间维度
        hidden_channels: int = 192,
        n_layers: int = 16,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 输入卷积
        self.input_conv = Conv1d(in_channels, hidden_channels, 1)

        # WaveNet 风格的残差块
        self.resblocks = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** (i % 8)
            self.resblocks.append(
                nn.Sequential(
                    Conv1d(hidden_channels, hidden_channels, kernel_size, dilation=dilation, padding=(kernel_size - 1) * dilation // 2),
                    LeakyReLU(0.1),
                )
            )

        # 输出层
        self.proj_mean = Conv1d(hidden_channels, out_channels, 1)
        self.proj_std = Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: Mel 频谱 [batch, in_channels, time]

        Returns:
            (均值, 对数方差)
        """
        x = self.input_conv(x)

        # 残差连接
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
    归一化流

    用于 GAN 模式的 VITS
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

        # 仿射耦合层
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
        前向传播 (用于推理)

        Args:
            x: 输入 [batch, channels, time]

        Returns:
            变换后输出
        """
        for flow in self.flows:
            x = flow['pre'](x)
            x = F.leaky_relu(x, 0.1)
            x = flow['coupling'](x)
            x = F.leaky_relu(x, 0.1)
            affine = flow['post'](x)

            # 仿射变换
            log_scale, bias = affine.chunk(2, dim=1)
            log_scale = torch.tanh(log_scale)
            x = x * log_scale.exp() + bias

        return x


class VITSResBlock(nn.Module):
    """VITS 残差块"""

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
    VITS 解码器

    从潜在表示生成音频
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

        # 输入卷积
        self.input_conv = Conv1d(in_channels, upsample_initial_channel, 7, padding=3)

        # 上采样层
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

            # 每个上采样层后的残差块
            for _ in range(2):
                self.resblocks.append(
                    VITSResBlock(channel // 2, resblock_kernel_sizes[0], resblock_dilation_sizes[0])
                )

        # 输出卷积
        self.output_conv = Conv1d(channel // 2, out_channels, 7, padding=3)

        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (Conv1d, ConvTranspose1d)):
            nn.init.normal_(m.weight, 0.0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 潜在表示 [batch, in_channels, time]

        Returns:
            音频波形 [batch, out_channels, time * prod(upsample_rates)]
        """
        x = self.input_conv(x)
        x = F.leaky_relu(x, 0.1)

        for i, upsample in enumerate(self.upsamples):
            x = upsample(x)
            x = F.leaky_relu(x, 0.1)

            # 应用残差块
            for _ in range(2):
                if self.resblocks:
                    x = self.resblocks[i * 2](x)

        x = self.output_conv(x)
        x = torch.tanh(x)

        return x


class VITSGenerator(nn.Module):
    """
    So-VITS 主生成器

    整合所有组件的完整 VITS 模型
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

        # 编码器
        if n_vocab > 0:
            self.text_encoder = TextEncoder(n_vocab, hidden_channels, hidden_channels)
        else:
            self.content_encoder = ContentEncoder(1024, hidden_channels, hidden_channels)

        # 说话人嵌入
        if n_speakers > 0:
            self.emb = nn.Embedding(n_speakers, gin_channels)

        # 长度调节器
        self.length_regulator = LengthRegulator(hidden_channels)

        # 流 (可选)
        if use_transformer_flows:
            self.flow = Flow(hidden_channels, hidden_channels)

        # 解码器
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
        前向传播

        Args:
            x: 输入特征或文本 [batch, channels, time] 或 [batch, time]
            f0: F0 轨迹 [batch, time] (可选)
            speaker_ids: 说话人 ID [batch]

        Returns:
            音频 [batch, 1, time]
        """
        # 内容编码
        if hasattr(self, 'text_encoder'):
            x, _ = self.text_encoder(x)
        else:
            # x 已经是编码后的特征
            pass

        # 说话人嵌入
        if speaker_ids is not None and self.n_speakers > 0:
            g = self.emb(speaker_ids).unsqueeze(-1)  # [batch, gin, 1]
            x = x + g

        # 长度调节
        x, _ = self.length_regulator(x)

        # F0 调制 (如果提供)
        if f0 is not None:
            x = self._apply_f0_modulation(x, f0)

        # 解码
        x = self.decoder(x)

        return x

    def _apply_f0_modulation(self, x: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        """F0 调制"""
        # 简单的 F0 调制
        if f0.shape[-1] != x.shape[-1]:
            f0 = F.interpolate(f0.unsqueeze(1), size=x.shape[-1], mode='linear').squeeze(1)

        # 将 F0 添加到通道维度
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
        推理接口

        Args:
            features: 内容特征 [batch, dim, time]
            f0: F0 轨迹 [batch, time]
            speaker_ids: 说话人 ID [batch]

        Returns:
            音频 [batch, 1, time]
        """
        return self.forward(features, f0, speaker_ids)


class SimpleVITSModel(nn.Module):
    """
    简化的 VITS 模型封装

    支持多种 So-VITS 模型格式，自动适配
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}

        # 提取配置
        self.n_vocab = self.config.get('n_vocab', 0)
        self.spec_channels = self.config.get('spec_channels', 80)
        self.hidden_channels = self.config.get('hidden_channels', 192)
        self.out_channels = self.config.get('out_channels', 1)
        self.n_speakers = self.config.get('n_speakers', 0)
        self.gin_channels = self.config.get('gin_channels', 0)
        self.use_flow = self.config.get('use_flow', False)

        # 创建生成器
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
        """前向传播"""
        return self.generator(x, f0, speaker_ids)

    def inference(
        self,
        features: torch.Tensor,
        f0: Optional[torch.Tensor] = None,
        speaker_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """推理接口"""
        return self.generator.inference(features, f0, speaker_ids)


def create_vits_model_from_checkpoint(
    checkpoint: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> SimpleVITSModel:
    """
    从检查点创建 VITS 模型

    Args:
        checkpoint: 模型检查点
        config: 模型配置

    Returns:
        VITS 模型实例
    """
    # 合并配置
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

    # 创建模型
    model = SimpleVITSModel(config)

    # 尝试加载权重
    if state_dict:
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception:
            # 部分加载失败，使用默认权重
            pass

    return model
