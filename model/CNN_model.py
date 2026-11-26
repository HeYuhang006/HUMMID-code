import torch
import torch.nn as nn
from einops import rearrange
from . import Modulate
import math

# CNN基础块
class Conv3dLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        return self.activation(self.conv(x))

# 纯卷积块（替代注意力块）
class CNNBlock(nn.Module):
    """纯卷积块，不包含注意力机制"""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = Conv3dLeakyReLU(channels, channels, kernel_size=kernel_size)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.norm = nn.InstanceNorm3d(channels)
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.norm(x)
        x = x + residual  # 残差连接
        return self.activation(x)

# 下采样块（纯卷积版本）
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.out_channels = out_channels
        self.conv1 = Conv3dLeakyReLU(in_channels, out_channels)
        # 使用CNNBlock替代注意力
        self.cnn_block = CNNBlock(out_channels)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.cnn_block(x)  # 纯卷积块
        skip = x
        x = self.maxpool(x)
        return x, skip

# 纯卷积编码器
class CNNEncoder(nn.Module):
    def __init__(self, in_channels=10, base_channels=48):
        super(CNNEncoder, self).__init__()
        self.down1 = Downsample(in_channels, base_channels)
        self.down2 = Downsample(base_channels, base_channels * 2)
        self.down3 = Downsample(base_channels * 2, base_channels * 4)
        self.down4 = Downsample(base_channels * 4, base_channels * 8)

        self.bottleneck = nn.Sequential(
            Conv3dLeakyReLU(base_channels * 8, base_channels * 16),
            CNNBlock(base_channels * 16),  # 纯卷积块
            Conv3dLeakyReLU(base_channels * 16, base_channels * 16)
        )
        
    def forward(self, x):
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)
        x = self.bottleneck(x)
        return x, [skip1, skip2, skip3, skip4]

# 上采样块（纯卷积版本）
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, kernel_size=2, stride=2):
        super(UpsampleBlock, self).__init__()
        self.skip_channels = skip_channels
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride)
        self.conv = nn.Sequential(
            nn.Conv3d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            CNNBlock(out_channels)  # 纯卷积块替代CBAM
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# 解码器
class Decoder(nn.Module):
    def __init__(self, base_channels=48, out_channels=1):
        super(Decoder, self).__init__()
        self.decoder_blocks = nn.ModuleList([
            UpsampleBlock(base_channels * 16, base_channels * 8, base_channels * 8),
            UpsampleBlock(base_channels * 8, base_channels * 4, base_channels * 4),
            UpsampleBlock(base_channels * 4, base_channels * 2, base_channels * 2),
            UpsampleBlock(base_channels * 2, base_channels, base_channels)
        ])
        self.final_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)
        
    def forward(self, x, skips):
        skips = list(reversed(skips))
        up_features = []
        for i, block in enumerate(self.decoder_blocks):
            x = block(x, skips[i])
            up_features.append(x)
        return self.final_conv(x), up_features

# 纯卷积U-Net（不带Transformer和注意力）
class PureConvUNet(nn.Module):
    """纯卷积版本的U-Net，没有Transformer和注意力机制"""
    def __init__(self, in_channels=10, out_channels=1, base_channels=48):
        super().__init__()
        self.encoder = CNNEncoder(
            in_channels=in_channels,
            base_channels=base_channels
        )
        self.decoder = Decoder(
            base_channels=base_channels,
            out_channels=out_channels
        )
        
    def forward(self, x, loss_weights):
        x = x.permute(0, 4, 1, 2, 3)
        loss_weights = loss_weights.permute(0, 4, 1, 2, 3)
        
        # 编码器
        encoded, skips = self.encoder(x)
        
        # 解码器
        out, up_features = self.decoder(encoded, skips)
        
        out = torch.cat([out, loss_weights], dim=1)
        out = out.permute(0, 2, 3, 4, 1)
        return out
    
    def mul_scale_features(self, x):
        """获取多尺度特征（用于辅助任务）"""
        x = x.permute(0, 4, 1, 2, 3)
        z, skips = self.encoder(x)
        skip1, skip2, skip3, skip4 = skips
        y, up_features = self.decoder(z, skips)
        
        return [z, skip1, skip2, skip3, y, 
                up_features[0], up_features[1], up_features[2]]

