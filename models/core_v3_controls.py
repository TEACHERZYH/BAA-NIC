"""Frozen causal controls for core-v3 encoder and decoder experiments."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .baa_nic import LATENT_CHANNELS


class DenseResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.block(x))


def _dense_downsample_stage(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
        DenseResidualBlock(out_channels),
    )


class DenseControlEncoder(nn.Module):
    """Channel-matched control replacing every depthwise convolution by dense convolution."""

    def __init__(self, latent_channels: int = LATENT_CHANNELS) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        self.stage1 = nn.Sequential(DenseResidualBlock(32), DenseResidualBlock(32))
        self.stage2 = _dense_downsample_stage(32, 48)
        self.stage3 = _dense_downsample_stage(48, 64)
        self.stage4 = _dense_downsample_stage(64, 96)
        self.projection = nn.Conv2d(96, latent_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.projection(x)


class _LightUpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))


class LightServerDecoder(nn.Module):
    """Symmetric-scale decoder control with every parameter on the executed path."""

    def __init__(self, latent_channels: int = LATENT_CHANNELS, num_groups: int = 4) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(latent_channels + num_groups, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
        )
        self.up1 = _LightUpsampleBlock(96, 72)
        self.up2 = _LightUpsampleBlock(72, 48)
        self.up3 = _LightUpsampleBlock(48, 32)
        self.up4 = _LightUpsampleBlock(32, 22)
        self.to_rgb = nn.Sequential(nn.Conv2d(22, 3, kernel_size=3, padding=1), nn.Sigmoid())

    def forward(self, masked_latent: Tensor, group_indicator: Tensor, output_size: tuple[int, int]) -> Tensor:
        x = self.fuse(torch.cat([masked_latent, group_indicator], dim=1))
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        if x.shape[-2:] != output_size:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return self.to_rgb(x)
