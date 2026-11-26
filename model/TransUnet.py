# Net1: output
import torch
import torch.nn as nn
from einops import rearrange
from . import Modulate
import math
# CNN
class Conv3dLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        return self.activation(self.conv(x))

# downsample blok
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.out_channels = out_channels
        self.conv1 = Conv3dLeakyReLU(in_channels, out_channels)
        # self.conv2 = Conv3dLeakyReLU(out_channels, out_channels)
        self.cbam = Modulate.CBAM(out_channels)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)
    def forward(self, x):
        a = x
        b = self.out_channels
        x = self.conv1(x)
        # x = self.conv2(x)
        x = self.cbam(x)
        skip = x
        x = self.maxpool(x)
        return x, skip
class CNNEncoder(nn.Module):
    def __init__(self, in_channels=10, base_channels=48):
        super(CNNEncoder, self).__init__()
        self.down1 = Downsample(in_channels, base_channels)
        self.down2 = Downsample(base_channels, base_channels * 2)
        self.down3 = Downsample(base_channels * 2, base_channels * 4)
        self.down4 = Downsample(base_channels * 4, base_channels * 8)

        self.bottleneck = nn.Sequential(
            Conv3dLeakyReLU(base_channels * 8, base_channels * 16),
            Conv3dLeakyReLU(base_channels * 16, base_channels * 16)
        )
    def forward(self, x):

        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)
        x = self.bottleneck(x)
        return x, [skip1, skip2, skip3, skip4]
class TransformerBlock(nn.Module):
    def __init__(self, base_channels = 48, num_heads = 8, mlp_ratio = 4, dropout = 0.1):
        super(TransformerBlock, self).__init__()
        embed_dim = base_channels * 16
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim = embed_dim,
            num_heads = num_heads,
            dropout = dropout,
            batch_first = True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        attn_output, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x +attn_output
        mlp_output = self.mlp(self.norm2(x))
        x = x + mlp_output
        return x
class PositionEncoder3D(nn.Module):
    def __init__(self, grid_size, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(
            torch.zeros(1, grid_size[0] * grid_size[1] * grid_size[2], embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        return x + self.pos_embed

class DynamicPositionEncoder3D(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
    def get_1d_sincos_pos_embed(self, length, embed_dim):
        """生成一维正弦/余弦位置编码"""
        if embed_dim % 2 != 0:
            raise ValueError("Embed dimension must be even for sincos pos_embed")
        
        pos = torch.arange(length, dtype=torch.float)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pos_embed = torch.zeros(length, embed_dim)
        pos_embed[:, 0::2] = torch.sin(pos.unsqueeze(1) * div_term)
        pos_embed[:, 1::2] = torch.cos(pos.unsqueeze(1) * div_term)
        return pos_embed

    def forward(self, x, grid_size):
        """动态生成3D位置编码
        Args:
            x: 输入张量 [batch, seq_len, embed_dim]
            grid_size: 特征图尺寸元组 (d, h, w)
        """
        d, h, w = grid_size
        total_elements = d * h * w
        
        # 生成三维位置编码
        pos_d = self.get_1d_sincos_pos_embed(d, self.embed_dim)  # [d, C]
        pos_h = self.get_1d_sincos_pos_embed(h, self.embed_dim)  # [h, C]
        pos_w = self.get_1d_sincos_pos_embed(w, self.embed_dim)  # [w, C]
        
        # 扩展维度并相加
        pos_d = pos_d.view(d, 1, 1, self.embed_dim)
        pos_h = pos_h.view(1, h, 1, self.embed_dim)
        pos_w = pos_w.view(1, 1, w, self.embed_dim)
        pos_3d = (pos_d + pos_h + pos_w).view(total_elements, self.embed_dim)  # [d*h*w, C]
        
        # 添加位置编码
        return x + pos_3d.to(x.device)
class TransformerEncoder(nn.Module):
    def __init__(self, base_channels = 48, num_heads = 8, depth = 6, mlp_ratio = 4, dropout = 0.1):
        super(TransformerEncoder, self).__init__()
        embed_dim = base_channels*16
        self.layers = nn.ModuleList([
            TransformerBlock(base_channels = base_channels,num_heads = num_heads, mlp_ratio = mlp_ratio, dropout = dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
# full Encoder
class Encoder(nn.Module):
    def __init__(self,in_channels, base_channels, num_heads = 8, mlp_ratio = 4, dropout = 0.1, depth = 6):
        super(Encoder, self).__init__()
        embed_dim = base_channels*16
        self.grid_size = (4, 4, 4)
        self.CNN_encoder = CNNEncoder(
            in_channels=in_channels,
            base_channels=base_channels
        )
        # self.position_encoder = PositionEncoder3D(
        #     grid_size = self.grid_size,
        #     embed_dim = embed_dim
        # )
        self.position_encoder = DynamicPositionEncoder3D(embed_dim=embed_dim)
        self.TransformerEncoder = TransformerEncoder(
            base_channels = base_channels,
            num_heads = num_heads,
            mlp_ratio = mlp_ratio,
            depth = depth
        )
        self.output_projection = nn.Conv3d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        features, skips = self.CNN_encoder(x)
        tokens = rearrange(features, 'b c d h w -> b (d h w) c')
        d, h, w = features.shape[2], features.shape[3], features.shape[4]
        grid_size = (d, h, w)

        tokens = self.position_encoder(tokens, grid_size)
        encoded_tokens = self.TransformerEncoder(tokens)
        encoded_features = rearrange(
            encoded_tokens,
            'b (d h w) c -> b c d h w',
            d=self.grid_size[0],
            h=self.grid_size[1],
            w=self.grid_size[2]
        )
        return self.output_projection(encoded_features), skips

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, kernel_size=2, stride=2):
        super(UpsampleBlock, self).__init__()
        self.skip_channels = skip_channels
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride)
        self.conv = nn.Sequential(
            nn.Conv3d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class Decoder(nn.Module):
    def __init__(self, base_channels = 48, out_channels = 1):
        super(Decoder, self).__init__()
        self.decoder_blocks = nn.ModuleList([
            UpsampleBlock(base_channels * 16, base_channels * 8, base_channels * 8), # 4x4x4 -> 8x8x8
            UpsampleBlock(base_channels * 8, base_channels * 4, base_channels * 4),  # 8x8x8 -> 16x16x16
            UpsampleBlock(base_channels * 4, base_channels * 2, base_channels * 2),  # 16x16x16 -> 32x32x32
            UpsampleBlock(base_channels * 2, base_channels, base_channels)  # 32x32x32 -> 64x64x64
        ])
        self.final_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)
    def forward(self, x, skips):
        skips = list(reversed(skips))
        up_features = []
        for i, block in enumerate(self.decoder_blocks):
            x = block(x, skips[i])
            up_features.append(x)
        return self.final_conv(x), up_features
class TransUNet(nn.Module):
    def __init__(self, in_channels=10, out_channels=1, base_channels=48, num_heads=8, depth=6):
        super().__init__()
        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            num_heads=num_heads,
            depth=depth
        )
        self.decoder = Decoder(
            base_channels=base_channels,
            out_channels=out_channels
        )
    def mul_scale_features(self, x):
        # encoder 3 times downsample
        x = x.permute(0, 4, 1, 2, 3)
        z, skips= self.encoder.CNN_encoder(x)
        skip1, skip2, skip3 = skips
        y, up_features = self.decoder(z, skips)
        return [z, skip1, skip2, skip3, y, up_features[0], up_features[1], up_features[2]]
    def forward(self, x, loss_weights):
        x = x.permute(0, 4, 1, 2, 3)
        loss_weights = loss_weights.permute(0, 4, 1, 2, 3)
        encoded, skips = self.encoder(x)
        out ,up_features = self.decoder(encoded, skips)
        out = torch.cat([out, loss_weights], dim=1)
        out = out.permute(0, 2, 3, 4, 1)
        return out

# class TransWNet(nn.Module):
#     def __init__(self, in_channels=11, out_channels=1, base_channels=48, num_heads=8, depth=6):
#         super().__init__()
#         self.T1w_Generator_Net = TransUNet(in_channels=in_channels, out_channels=1, base_channels=base_channels)
#         self.DWI_Generator_Net = TransUNet(in_channels=8, out_channels=7,base_channels=base_channels)
#         self.T1w_generator = self.T1w_Generator_Net.forward
#         self.DWI_generator = self.DWI_Generator_Net.forward
#     def forward(self, x, loss_weights):
#         t1w_mask = self.T1w_generator(x, loss_weights)
#         # t1w_mask = t1w_mask.permute(0, 4, 1, 2, 3)
#         t1w = t1w_mask[:,:,:,:,:1]
#         # x: ADC1, ADC2, ADC3, Mean, B0, DWI1, DWI2, DWI3, DWI4, DWI5, DWI6
#         dwi = x[:,:,:,:,4:]
#         # dwi = dwi.permute(0, 4, 1, 2, 3)
#         # loss_weights = loss_weights.permute(0, 4, 1, 2, 3)
#         # B0, DWI1, DWI2, DWI3, DWI4, DWI5, DWI6, Tw
#         dwi_input = torch.cat([dwi, t1w], dim=4)
#         dwi_mask = self.DWI_generator(dwi_input, loss_weights)
#         # t1w_mask = t1w_mask.permute(0, 2, 3, 4, 1)
#         # dwi_mask = dwi_mask.permute(0, 2, 3, 4, 1)
#         return t1w_mask, dwi_mask

