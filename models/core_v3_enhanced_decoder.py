"""Deterministic same-bitstream server decoder for the core-v3 extension."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Per-pixel channel normalization used by restoration transformers."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias


class DecoderCondition(nn.Module):
    """Embed quantizer and received-group state already present at the receiver."""

    def __init__(self, condition_dim: int = 64) -> None:
        super().__init__()
        self.q_embedding = nn.Embedding(4, condition_dim)
        self.prefix_embedding = nn.Embedding(5, condition_dim)
        self.fuse = nn.Sequential(
            nn.Linear(condition_dim * 2, condition_dim * 2),
            nn.SiLU(inplace=True),
            nn.Linear(condition_dim * 2, condition_dim),
        )

    @staticmethod
    def _q_tensor(q_index: int | Tensor, batch: int, device: torch.device) -> Tensor:
        if isinstance(q_index, Tensor):
            values = q_index.to(device=device, dtype=torch.long).flatten()
            if values.numel() == 1:
                values = values.expand(batch)
            if values.numel() != batch:
                raise ValueError("q_index tensor must have one value or one value per sample")
        else:
            values = torch.full((batch,), int(q_index), dtype=torch.long, device=device)
        if bool(((values < 0) | (values > 3)).any()):
            raise ValueError("q_index must be in [0, 3]")
        return values

    def forward(self, q_index: int | Tensor, group_indicator: Tensor) -> Tensor:
        batch = group_indicator.shape[0]
        q_values = self._q_tensor(q_index, batch, group_indicator.device)
        prefix_values = group_indicator.float().mean(dim=(-2, -1)).sum(dim=1).round().long().clamp(0, 4)
        return self.fuse(torch.cat([self.q_embedding(q_values), self.prefix_embedding(prefix_values)], dim=1))


class ChannelAttention(nn.Module):
    """Restormer-style transposed attention with spatially linear memory growth."""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        self.channels = channels
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_dw = nn.Conv2d(
            channels * 3,
            channels * 3,
            kernel_size=3,
            padding=1,
            groups=channels * 3,
            bias=False,
        )
        self.project = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        q, k, v = self.qkv_dw(self.qkv(x)).chunk(3, dim=1)
        head_channels = channels // self.heads
        q = q.reshape(batch, self.heads, head_channels, height * width)
        k = k.reshape(batch, self.heads, head_channels, height * width)
        v = v.reshape(batch, self.heads, head_channels, height * width)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attention = (q @ k.transpose(-2, -1)) * self.temperature
        attention = attention.softmax(dim=-1)
        output = (attention @ v).reshape(batch, channels, height, width)
        return self.project(output)


class GatedFeedForward(nn.Module):
    def __init__(self, channels: int, expansion: float) -> None:
        super().__init__()
        hidden = int(round(channels * expansion))
        self.project_in = nn.Conv2d(channels, hidden * 2, kernel_size=1, bias=False)
        self.depthwise = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            kernel_size=3,
            padding=1,
            groups=hidden * 2,
            bias=False,
        )
        self.project_out = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        content, gate = self.depthwise(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(content) * gate)


class _ConditionModulation(nn.Module):
    def __init__(self, condition_dim: int, channels: int, branches: int) -> None:
        super().__init__()
        self.channels = channels
        self.branches = branches
        self.projection = nn.Sequential(
            nn.SiLU(inplace=False),
            nn.Linear(condition_dim, branches * channels * 2),
        )
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(self, condition: Tensor) -> tuple[tuple[Tensor, Tensor], ...]:
        values = self.projection(condition).reshape(condition.shape[0], self.branches, 2, self.channels)
        output: list[tuple[Tensor, Tensor]] = []
        for branch in range(self.branches):
            scale = values[:, branch, 0, :, None, None]
            shift = values[:, branch, 1, :, None, None]
            output.append((scale, shift))
        return tuple(output)


class ConditionedRestorationBlock(nn.Module):
    def __init__(self, channels: int, heads: int, condition_dim: int, expansion: float) -> None:
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.attention = ChannelAttention(channels, heads)
        self.norm2 = LayerNorm2d(channels)
        self.feed_forward = GatedFeedForward(channels, expansion)
        self.modulation = _ConditionModulation(condition_dim, channels, branches=2)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        (scale1, shift1), (scale2, shift2) = self.modulation(condition)
        normalized = self.norm1(x) * (1.0 + scale1) + shift1
        x = x + self.attention(normalized)
        normalized = self.norm2(x) * (1.0 + scale2) + shift2
        return x + self.feed_forward(normalized)


class ConditionedConvBlock(nn.Module):
    """Memory-bounded full-resolution refinement block."""

    def __init__(self, channels: int, condition_dim: int, expansion: float) -> None:
        super().__init__()
        hidden = int(round(channels * expansion))
        self.norm = LayerNorm2d(channels)
        self.project_in = nn.Conv2d(channels, hidden * 2, kernel_size=1, bias=False)
        self.depthwise = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            kernel_size=5,
            padding=2,
            groups=hidden * 2,
            bias=False,
        )
        self.project_out = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.modulation = _ConditionModulation(condition_dim, channels, branches=1)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        ((scale, shift),) = self.modulation(condition)
        normalized = self.norm(x) * (1.0 + scale) + shift
        content, gate = self.depthwise(self.project_in(normalized)).chunk(2, dim=1)
        return x + self.project_out(F.gelu(content) * gate)


class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: Tensor) -> Tensor:
        return self.shuffle(self.projection(x))


class EnhancedServerDecoder(nn.Module):
    """Conditioned latent correction plus multi-scale deterministic synthesis."""

    def __init__(
        self,
        latent_channels: int = 192,
        num_groups: int = 4,
        condition_dim: int = 64,
        stage_widths: tuple[int, ...] = (192, 160, 128, 96, 64),
        stage_blocks: tuple[int, ...] = (3, 2, 2, 1, 1),
        stage_heads: tuple[int, ...] = (6, 5, 4, 3, 2),
        transformer_stage_count: int = 3,
        ffn_expansion: float = 2.0,
        residual_logit_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if not (len(stage_widths) == len(stage_blocks) == len(stage_heads) == 5):
            raise ValueError("the enhanced decoder requires five synthesis stages")
        self.residual_logit_scale = float(residual_logit_scale)
        self.condition = DecoderCondition(condition_dim)
        self.fuse = nn.Conv2d(latent_channels + num_groups, stage_widths[0], kernel_size=3, padding=1)
        self.stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for stage_index, (channels, blocks, heads) in enumerate(
            zip(stage_widths, stage_blocks, stage_heads)
        ):
            block_type: type[nn.Module]
            if stage_index < transformer_stage_count:
                modules = [
                    ConditionedRestorationBlock(channels, heads, condition_dim, ffn_expansion)
                    for _ in range(blocks)
                ]
            else:
                modules = [
                    ConditionedConvBlock(channels, condition_dim, ffn_expansion)
                    for _ in range(blocks)
                ]
            self.stages.append(nn.ModuleList(modules))
            if stage_index + 1 < len(stage_widths):
                self.upsamples.append(PixelShuffleUpsample(channels, stage_widths[stage_index + 1]))
        final_channels = stage_widths[-1]
        self.base_head = nn.Conv2d(final_channels, 3, kernel_size=3, padding=1)
        self.residual_head = nn.Sequential(
            nn.Conv2d(final_channels + 3, final_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(final_channels, 3, kernel_size=3, padding=1),
        )
        self.confidence_head = nn.Sequential(
            nn.Conv2d(final_channels + 3, final_channels // 2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(final_channels // 2, 1, kernel_size=3, padding=1),
        )

    def forward(
        self,
        masked_latent: Tensor,
        group_indicator: Tensor,
        q_index: int | Tensor,
        output_size: tuple[int, int],
        return_aux: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        if masked_latent.shape[0] != group_indicator.shape[0]:
            raise ValueError("latent and group indicator batch sizes differ")
        condition = self.condition(q_index, group_indicator)
        x = self.fuse(torch.cat([masked_latent, group_indicator], dim=1))
        for stage_index, blocks in enumerate(self.stages):
            for block in blocks:
                x = block(x, condition)
            if stage_index < len(self.upsamples):
                x = self.upsamples[stage_index](x)
        if x.shape[-2:] != output_size:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)

        base_logits = self.base_head(x)
        base = torch.sigmoid(base_logits)
        high_frequency = base - F.avg_pool2d(base, kernel_size=3, stride=1, padding=1)
        refinement_input = torch.cat([x, high_frequency], dim=1)
        residual_logits = self.residual_head(refinement_input)
        confidence = torch.sigmoid(self.confidence_head(refinement_input))
        reconstruction = torch.sigmoid(
            base_logits + self.residual_logit_scale * confidence * residual_logits
        )
        if return_aux:
            return {
                "reconstruction": reconstruction,
                "base_reconstruction": base,
                "residual_logits": residual_logits,
                "confidence": confidence,
                "condition": condition,
            }
        return reconstruction


def build_enhanced_server_decoder(config: dict[str, Any]) -> EnhancedServerDecoder:
    architecture = config["architecture"]
    return EnhancedServerDecoder(
        latent_channels=int(architecture["latent_channels"]),
        num_groups=int(architecture["group_count"]),
        condition_dim=int(architecture["condition_dim"]),
        stage_widths=tuple(int(value) for value in architecture["stage_widths"]),
        stage_blocks=tuple(int(value) for value in architecture["stage_blocks"]),
        stage_heads=tuple(int(value) for value in architecture["stage_heads"]),
        transformer_stage_count=int(architecture["transformer_stage_count"]),
        ffn_expansion=float(architecture["ffn_expansion"]),
        residual_logit_scale=float(architecture["residual_logit_scale"]),
    )
