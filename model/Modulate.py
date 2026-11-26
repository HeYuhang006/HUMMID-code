import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
#%%
# SE-Net
class SE_3D(nn.Module):
    def __init__(self, in_channels, ratio):
        super(SE_3D, self).__init__()
        # 使用三维的自适应平均池化
        self.squeeze = nn.AdaptiveAvgPool3d((1, 1, 1))
        # 使用三维卷积，输入通道数为 in_channels，输出通道数为 in_channels // ratio
        self.compress = nn.Conv3d(in_channels, in_channels // ratio, kernel_size=1)
        # 使用三维卷积，输入通道数为 in_channels // ratio，输出通道数为 in_channels
        self.excitation = nn.Conv3d(in_channels // ratio, in_channels, kernel_size=1)

    def forward(self, x):
        # x 的形状为 (batch_size, channels, depth, height, width)
        out = self.squeeze(x)  # (batch_size, channels, 1, 1, 1)
        out = self.compress(out)  # (batch_size, channels//ratio, 1, 1, 1)
        out = F.leaky_relu(out, negative_slope=0.01)  # 激活函数
        out = self.excitation(out)  # (batch_size, channels, 1, 1, 1)
        out = F.sigmoid(out)  # 激活函数将权重归一化到 [0,1]
        return x * out  # 将权重广播到输入张量的每个位置，形状保持一致
#%%
# Spatial Attention Block Channel Attention Block
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

