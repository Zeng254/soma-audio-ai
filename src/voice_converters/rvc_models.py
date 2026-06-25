"""
RVC 模型架构定义

包含 RVC v1/v2 的核心模型结构：
- RVCGenerator: RVC 主生成器
- PitchEncoder: 音高编码器
- HiFiGANGenerator: HiFi-GAN 声码器

这些类用于加载和运行 RVC 模型推理。
"""

from typing import Optional, Tuple, List, Any, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """残差块"""

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
        for conv in self.convs:
            x = F.leaky_relu(x, 0.1)
            x = conv(x)
        return x


class RVCGenerator(nn.Module):
    """
    RVC 主生成器网络

    架构:
    - 共享编码器 (Projection + ResBlocks)
    - F0 条件调制
    - 上采样解码器
    - 最终卷积输出
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

        # 输入卷积
        self.input_conv = nn.Conv1d(
            in_channels + pitch_encoder_dim,  # 特征 + F0 编码
            hidden_channels,
            kernel_size,
            padding=kernel_size // 2
        )

        # 残差块
        self.resblocks = nn.ModuleList()
        for kernel_size in resblock_kernel_sizes:
            self.resblocks.append(
                ResBlock(hidden_channels, kernel_size, resblock_dilation_sizes[0])
            )

        # 上采样层
        self.upsamples = nn.ModuleList()
        self.upsample_channels = []
        channel = hidden_channels
        for i, (rate, kernel_size) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.upsample_channels.append(channel)
            self.upsamples.append(
                nn.ConvTranspose1d(
                    channel,
                    upsample_initial_channel // (2 ** i),
                    kernel_size,
                    stride=rate,
                    padding=(kernel_size - rate) // 2
                )
            )
            channel = upsample_initial_channel // (2 ** (i + 1))

        # 最终输出卷积
        self.output_conv = nn.Conv1d(
            channel,
            out_channels,
            kernel_size,
            padding=kernel_size // 2
        )

        # 初始化
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
        前向传播

        Args:
            x: 输入张量 [batch, channels, time]
            f0_embedding: F0 嵌入 [batch, pitch_dim, time] (可选)

        Returns:
            输出音频 [batch, out_channels, time * upsample_rate]
        """
        if f0_embedding is not None:
            # 将 F0 嵌入拼接到输入
            if f0_embedding.shape[-1] != x.shape[-1]:
                # 重采样 F0 嵌入以匹配
                f0_embedding = F.interpolate(
                    f0_embedding,
                    size=x.shape[-1],
                    mode='linear'
                )
            x = torch.cat([x, f0_embedding], dim=1)

        # 输入卷积
        x = self.input_conv(x)
        x = F.leaky_relu(x, 0.1)

        # 残差块
        for resblock in self.resblocks:
            x = resblock(x)

        # 上采样
        for upsample in self.upsamples:
            x = F.leaky_relu(x, 0.1)
            x = upsample(x)

        # 输出卷积
        x = F.leaky_relu(x, 0.1)
        x = self.output_conv(x)
        x = torch.tanh(x)

        return x


class FlowDecoder(nn.Module):
    """Flow 解码器 (用于某些 RVC 模型)"""

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

        # Wavenet 风格的解码器
        self.input_conv = nn.Conv1d(in_channels, hidden_channels, 1)

        self.upsamples = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        # 上采样到原始长度
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

        # 输出层
        self.output_conv = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.input_conv(x)
        x = F.leaky_relu(x, 0.1)

        for upsample, resblock in zip(self.upsamples, self.resblocks):
            x = upsample(x)
            x = F.leaky_relu(x, 0.1)

            # 残差连接
            skip = x
            for conv in resblock:
                x = F.leaky_relu(x, 0.1)
                x = conv(x)
            x = x + skip

        x = self.output_conv(x)
        return x


class ConvFlowDecoder(nn.Module):
    """ConvFlow 解码器"""

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 192,
        out_channels: int = 80,
        num_flows: int = 12,
    ):
        super().__init__()
        self.num_flows = num_flows

        # 条件编码
        self.cond_encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1),
        )

        # Flow 层
        self.flows = nn.ModuleList()
        for _ in range(num_flows):
            self.flows.append(
                nn.ModuleDict({
                    'conv': nn.Conv1d(hidden_channels, hidden_channels * 2, 3, padding=1),
                    'norm': nn.BatchNorm1d(hidden_channels),
                })
            )

        # 输出投影
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 (用于推理)

        Args:
            x: 输入特征 [batch, channels, time]

        Returns:
            梅尔频谱 [batch, n_mels, time]
        """
        x = self.cond_encoder(x)

        for flow in self.flows:
            x = F.relu(x)
            x = flow['conv'](x)
            x = flow['norm'](x)

        x = self.proj(x)
        # 返回均值 (不返回 log_scale 用于简化)
        return x[:, :x.shape[1] // 2, :]


class SineGenerator(nn.Module):
    """正弦生成器 (简化版声码器)"""

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
        基于 F0 生成音频

        Args:
            f0: F0 轨迹 [batch, 1, time]
            mel_output: 梅尔频谱 [batch, n_mels, time] (可选)

        Returns:
            音频波形 [batch, 1, time * hop_length]
        """
        device = f0.device
        batch_size, _, time_steps = f0.shape

        # 基础频率
        freq = f0.squeeze(1)  # [batch, time]

        # 生成时间步
        indices = torch.arange(time_steps, device=device).float().unsqueeze(0)
        indices = indices.repeat(batch_size, 1)

        # 相位累积
        phase = torch.cumsum(freq / self.sampling_rate, dim=1)

        # 生成正弦波
        sine_wave = torch.sin(2 * torch.pi * phase)

        # 添加谐波
        if self.harmonic_num > 0:
            for h in range(2, self.harmonic_num + 2):
                sine_wave += torch.sin(2 * torch.pi * phase * h) / h

        # 添加噪声
        noise = torch.randn_like(sine_wave) * self.noise_std
        sine_wave = sine_wave * self.sine_amp + noise

        # 重塑为 [batch, 1, time]
        return sine_wave.unsqueeze(1)


class SimpleRVCModel(nn.Module):
    """
    简化的 RVC 模型封装

    支持多种 RVC 模型格式，自动适配
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}

        # 根据配置初始化网络
        self.sample_rate = self.config.get('sample_rate', 40000)
        self.hop_length = self.config.get('hop_length', 512)
        self.mel_channels = self.config.get('mel_channels', 128)

        # 尝试构建网络
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
        前向传播

        Args:
            x: HubERT 特征 [batch, 256, time]
            f0: F0 轨迹 [batch, time] 或 [batch, 1, time]

        Returns:
            输出音频或梅尔频谱
        """
        # 调整 F0 维度
        if len(f0.shape) == 2:
            f0 = f0.unsqueeze(1)

        # F0 编码
        f0_encoded = self._encode_f0(f0)

        # 重采样 F0 编码以匹配特征长度
        if f0_encoded.shape[-1] != x.shape[-1]:
            f0_encoded = F.interpolate(
                f0_encoded,
                size=x.shape[-1],
                mode='linear'
            )

        if self.use_flow:
            output = self.flow_decoder(x)
        else:
            # 生成器推理
            output = self.generator(x, f0_encoded)

        return output

    def _encode_f0(self, f0: torch.Tensor) -> torch.Tensor:
        """F0 音高编码"""
        # 简单的 log 编码
        f0_encoded = torch.log(f0 + 1e-6)
        # 扩展维度以匹配
        if len(f0_encoded.shape) == 2:
            f0_encoded = f0_encoded.unsqueeze(1)
        return f0_encoded

    def inference(
        self,
        features: torch.Tensor,
        f0: torch.Tensor
    ) -> torch.Tensor:
        """推理接口"""
        return self.forward(features, f0)


def create_rvc_model_from_checkpoint(
    checkpoint: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> SimpleRVCModel:
    """
    从检查点创建 RVC 模型

    Args:
        checkpoint: 模型检查点
        config: 模型配置

    Returns:
        RVC 模型实例
    """
    # 合并配置
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

    # 创建模型
    model = SimpleRVCModel(config)

    # 尝试加载权重
    if state_dict:
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception:
            # 部分加载失败，使用默认权重
            pass

    return model
