"""Minimal BAA-NIC PyTorch model skeleton.

This module implements the paper-facing engineering contract. By default it
uses the original differentiable rate proxy; when requested it can attach
CompressAI entropy models for learned likelihood-based bpp estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .baa_nic_bitstream import (
    BitstreamHeader,
    ParsedSegment,
    assemble_full_stream,
    pack_segment,
    parse_stream,
)


ACTIVE_RATIOS = (0.25, 0.50, 0.75, 1.00)
GROUP_CHANNELS = 48
LATENT_CHANNELS = GROUP_CHANNELS * 4
Q_STEPS = (1.0, 2.0**0.5, 2.0, 2.0**1.5)
ENTROPY_MODEL_TYPES = ("factorized", "hyperprior", "group_hyperprior")


class DepthwiseSeparableResidualBlock(nn.Module):
    """Depthwise separable residual block used by the lightweight UAV encoder."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x + self.block(x))


def _downsample_stage(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
        DepthwiseSeparableResidualBlock(out_channels),
    )


class LightweightEncoder(nn.Module):
    """Light UAV-side encoder with /16 latent downsampling."""

    def __init__(self, latent_channels: int = LATENT_CHANNELS) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        self.stage1 = nn.Sequential(DepthwiseSeparableResidualBlock(32), DepthwiseSeparableResidualBlock(32))
        self.stage2 = _downsample_stage(32, 48)
        self.stage3 = _downsample_stage(48, 64)
        self.stage4 = _downsample_stage(64, 96)
        self.projection = nn.Conv2d(96, latent_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.projection(x)


class LatentGrouper(nn.Module):
    """Build masks for four progressive 48-channel latent groups."""

    def __init__(self, group_channels: int = GROUP_CHANNELS, num_groups: int = 4) -> None:
        super().__init__()
        self.group_channels = group_channels
        self.num_groups = num_groups
        self.total_channels = group_channels * num_groups

    def active_groups(self, active_ratio: float) -> int:
        if active_ratio not in ACTIVE_RATIOS:
            raise ValueError(f"active_ratio must be one of {ACTIVE_RATIOS}, got {active_ratio}")
        return int(round(active_ratio * self.num_groups))

    def mask(self, latent: Tensor, active_ratio: float) -> Tensor:
        groups = self.active_groups(active_ratio)
        return self.mask_from_group_ids(latent, range(1, groups + 1))

    def mask_from_group_ids(self, latent: Tensor, group_ids: Iterable[int]) -> Tensor:
        mask = latent.new_zeros((1, self.total_channels, 1, 1))
        for group_id in group_ids:
            if group_id not in range(1, self.num_groups + 1):
                raise ValueError(f"group_id must be in [1, {self.num_groups}], got {group_id}")
            start = (group_id - 1) * self.group_channels
            mask[:, start : start + self.group_channels] = 1.0
        return mask.expand(latent.shape[0], -1, latent.shape[2], latent.shape[3])

    def group_indicator(self, latent: Tensor, active_ratio: float) -> Tensor:
        groups = self.active_groups(active_ratio)
        return self.group_indicator_from_ids(latent, range(1, groups + 1))

    def group_indicator_from_ids(self, latent: Tensor, group_ids: Iterable[int]) -> Tensor:
        indicator = latent.new_zeros((latent.shape[0], self.num_groups, latent.shape[2], latent.shape[3]))
        for group_id in group_ids:
            if group_id not in range(1, self.num_groups + 1):
                raise ValueError(f"group_id must be in [1, {self.num_groups}], got {group_id}")
            indicator[:, group_id - 1] = 1.0
        return indicator


@dataclass(frozen=True)
class BandwidthPolicy:
    active_ratio: float
    q_index: int
    enhancement_layers: int
    target_bpp: float
    reason: str


class RuleBasedBandwidthController:
    """Rule controller matching scripts/preview_bandwidth_controller.py."""

    def choose(self, estimated_bandwidth: float, packet_loss: float = 0.0) -> BandwidthPolicy:
        if packet_loss >= 0.15 or estimated_bandwidth < 0.9:
            return BandwidthPolicy(0.25, 3, 0, 0.05, "severe_bandwidth_or_loss")
        if packet_loss >= 0.08 or estimated_bandwidth < 2.0:
            return BandwidthPolicy(0.50, 2, 1, 0.10, "low_bandwidth_or_loss")
        if estimated_bandwidth < 4.0:
            return BandwidthPolicy(0.75, 1, 2, 0.20, "medium_bandwidth")
        return BandwidthPolicy(1.00, 0, 3, 0.40, "high_bandwidth")

    def __call__(self, link_state: dict[str, Any] | None) -> BandwidthPolicy:
        link_state = link_state or {}
        bandwidth = float(
            link_state.get("estimated_bandwidth_mbps", link_state.get("estimated_bandwidth", 4.0))
        )
        packet_loss = float(link_state.get("packet_loss", link_state.get("packet_loss_rate", 0.0)))
        return self.choose(estimated_bandwidth=bandwidth, packet_loss=packet_loss)


class _UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            DepthwiseSeparableResidualBlock(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.block(x)


class ServerDecoder(nn.Module):
    """Server-heavy decoder with masked latent fusion and active-group indicators."""

    def __init__(self, latent_channels: int = LATENT_CHANNELS, num_groups: int = 4) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(latent_channels + num_groups, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True),
            DepthwiseSeparableResidualBlock(192),
        )
        self.up1 = _UpsampleBlock(192, 128)
        self.up2 = _UpsampleBlock(128, 96)
        self.up3 = _UpsampleBlock(96, 64)
        self.up4 = _UpsampleBlock(64, 32)
        self.to_rgb = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, masked_latent: Tensor, group_indicator: Tensor, output_size: tuple[int, int]) -> Tensor:
        x = torch.cat([masked_latent, group_indicator], dim=1)
        x = self.fuse(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        if x.shape[-2:] != output_size:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return self.to_rgb(x)


class BAANICHyperpriorEntropyModel(nn.Module):
    """Mean-scale hyperprior for the selected BAA-NIC latent prefix.

    This is still an engineering-stage entropy path: it provides learned
    likelihood bpp for training/evaluation, while arithmetic bitstream I/O is a
    follow-up step.
    """

    def __init__(self, latent_channels: int = LATENT_CHANNELS, hyper_channels: int = 128) -> None:
        super().__init__()
        try:
            from compressai.entropy_models import EntropyBottleneck, GaussianConditional
        except ImportError as exc:
            raise RuntimeError("compressai is required for entropy_model_type='hyperprior'") from exc

        self.entropy_bottleneck = EntropyBottleneck(hyper_channels)
        self.gaussian_conditional = GaussianConditional(None)
        self.hyper_analysis = nn.Sequential(
            nn.Conv2d(latent_channels, hyper_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hyper_channels, hyper_channels, kernel_size=5, stride=2, padding=2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hyper_channels, hyper_channels, kernel_size=5, stride=2, padding=2),
        )
        self.hyper_synthesis = nn.Sequential(
            nn.ConvTranspose2d(hyper_channels, hyper_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(hyper_channels, hyper_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hyper_channels, latent_channels * 2, kernel_size=3, padding=1),
        )

    def forward(self, masked_latent: Tensor, active_mask: Tensor, image_size: tuple[int, int]) -> dict[str, Tensor]:
        z = self.hyper_analysis(masked_latent.abs())
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.hyper_synthesis(z_hat)
        if gaussian_params.shape[-2:] != masked_latent.shape[-2:]:
            gaussian_params = F.interpolate(gaussian_params, size=masked_latent.shape[-2:], mode="bilinear", align_corners=False)
        scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
        scales_hat = F.softplus(scales_hat) + 0.11
        y_hat, y_likelihoods = self.gaussian_conditional(masked_latent, scales_hat, means=means_hat)
        y_hat = y_hat * active_mask
        pixels = float(masked_latent.shape[0] * image_size[0] * image_size[1])
        y_bits = -torch.log2(y_likelihoods.clamp_min(1e-9)) * active_mask
        z_bits = -torch.log2(z_likelihoods.clamp_min(1e-9))
        bpp = (y_bits.sum() + z_bits.sum()) / pixels
        return {
            "latent_hat": y_hat,
            "bpp": bpp,
            "y_likelihoods": y_likelihoods,
            "z_likelihoods": z_likelihoods,
            "scales_hat": scales_hat,
            "means_hat": means_hat,
        }

    def aux_loss(self) -> Tensor:
        return self.entropy_bottleneck.loss()

    def update(self, force: bool = False) -> bool:
        updated = bool(self.entropy_bottleneck.update(force=force))
        if hasattr(self.gaussian_conditional, "update_scale_table"):
            try:
                updated = bool(self.gaussian_conditional.update_scale_table(force=force)) or updated
            except TypeError:
                from compressai.models.google import get_scale_table

                updated = bool(self.gaussian_conditional.update_scale_table(get_scale_table(), force=force)) or updated
        return updated


class BAANICGroupHyperpriorEntropyModel(nn.Module):
    """Shared practical hyperprior independently coding each 48-channel group."""

    def __init__(self, group_channels: int = GROUP_CHANNELS, hyper_channels: int = 64) -> None:
        super().__init__()
        try:
            from compressai.entropy_models import EntropyBottleneck, GaussianConditional
        except ImportError as exc:
            raise RuntimeError("compressai is required for entropy_model_type='group_hyperprior'") from exc

        self.group_channels = group_channels
        self.hyper_channels = hyper_channels
        self.entropy_bottleneck = EntropyBottleneck(hyper_channels)
        self.gaussian_conditional = GaussianConditional(None)
        self.hyper_analysis = nn.Sequential(
            nn.Conv2d(group_channels, hyper_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hyper_channels, hyper_channels, kernel_size=5, stride=2, padding=2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hyper_channels, hyper_channels, kernel_size=5, stride=2, padding=2),
        )
        self.hyper_synthesis = nn.Sequential(
            nn.ConvTranspose2d(
                hyper_channels,
                hyper_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(
                hyper_channels,
                hyper_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(hyper_channels, group_channels * 2, kernel_size=3, padding=1),
        )

    @staticmethod
    def q_step(q_index: int) -> float:
        if q_index not in range(len(Q_STEPS)):
            raise ValueError(f"q_index must be in [0, {len(Q_STEPS) - 1}], got {q_index}")
        return Q_STEPS[q_index]

    def _distribution_parameters(self, z_hat: Tensor, y_shape: tuple[int, int]) -> tuple[Tensor, Tensor]:
        gaussian_params = self.hyper_synthesis(z_hat)
        if gaussian_params.shape[-2:] != y_shape:
            gaussian_params = F.interpolate(gaussian_params, size=y_shape, mode="bilinear", align_corners=False)
        scales_hat, means_hat = gaussian_params.chunk(2, dim=1)
        return F.softplus(scales_hat) + 0.11, means_hat

    def forward_group(self, group_latent: Tensor, q_index: int, image_size: tuple[int, int]) -> dict[str, Tensor]:
        q_step = self.q_step(q_index)
        scaled = group_latent / q_step
        z = self.hyper_analysis(scaled.abs())
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat, means_hat = self._distribution_parameters(z_hat, scaled.shape[-2:])
        scaled_hat, y_likelihoods = self.gaussian_conditional(scaled, scales_hat, means=means_hat)
        pixels = float(group_latent.shape[0] * image_size[0] * image_size[1])
        y_bits = -torch.log2(y_likelihoods.clamp_min(1e-9)).sum()
        z_bits = -torch.log2(z_likelihoods.clamp_min(1e-9)).sum()
        return {
            "latent_hat": scaled_hat * q_step,
            "bpp": (y_bits + z_bits) / pixels,
            "y_likelihoods": y_likelihoods,
            "z_likelihoods": z_likelihoods,
        }

    def forward(
        self,
        latent: Tensor,
        active_group_ids: Iterable[int],
        q_index: int,
        image_size: tuple[int, int],
    ) -> dict[str, Tensor]:
        requested = tuple(int(group_id) for group_id in active_group_ids)
        if len(set(requested)) != len(requested):
            raise ValueError("active_group_ids contains duplicates")
        latent_hat = torch.zeros_like(latent)
        y_likelihoods = torch.ones_like(latent)
        z_likelihoods: list[Tensor] = []
        bpp = latent.new_tensor(0.0)
        for group_id in requested:
            if group_id not in range(1, 5):
                raise ValueError(f"Invalid group ID: {group_id}")
            start = (group_id - 1) * self.group_channels
            group = latent[:, start : start + self.group_channels]
            result = self.forward_group(group, q_index=q_index, image_size=image_size)
            latent_hat[:, start : start + self.group_channels] = result["latent_hat"]
            y_likelihoods[:, start : start + self.group_channels] = result["y_likelihoods"]
            z_likelihoods.append(result["z_likelihoods"])
            bpp = bpp + result["bpp"]
        empty_z = latent.new_empty((latent.shape[0], 0, 0, 0))
        return {
            "latent_hat": latent_hat,
            "bpp": bpp,
            "y_likelihoods": y_likelihoods,
            "z_likelihoods": torch.cat(z_likelihoods, dim=1) if z_likelihoods else empty_z,
        }

    def compress_group_with_latent(self, group_latent: Tensor, q_index: int, group_id: int) -> tuple[bytes, Tensor]:
        if group_latent.shape[0] != 1:
            raise ValueError("Practical BAA-NIC compression currently requires batch size 1")
        q_step = self.q_step(q_index)
        scaled = group_latent / q_step
        z = self.hyper_analysis(scaled.abs())
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.shape[-2:])
        scales_hat, means_hat = self._distribution_parameters(z_hat, scaled.shape[-2:])
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(scaled, indexes, means=means_hat)
        scaled_hat = self.gaussian_conditional.quantize(scaled, "dequantize", means=means_hat)
        if len(z_strings) != 1 or len(y_strings) != 1:
            raise RuntimeError("Expected one entropy string per group for batch size 1")
        return (
            pack_segment(
                group_id=group_id,
                z_string=z_strings[0],
                y_string=y_strings[0],
                z_shape=(int(z.shape[-2]), int(z.shape[-1])),
                y_shape=(int(scaled.shape[-2]), int(scaled.shape[-1])),
            ),
            scaled_hat * q_step,
        )

    def compress_group(self, group_latent: Tensor, q_index: int, group_id: int) -> bytes:
        segment, _ = self.compress_group_with_latent(group_latent, q_index, group_id)
        return segment

    def decompress_group(self, segment: ParsedSegment, q_index: int) -> Tensor:
        q_step = self.q_step(q_index)
        z_hat = self.entropy_bottleneck.decompress([segment.z_string], segment.z_shape)
        scales_hat, means_hat = self._distribution_parameters(z_hat, segment.y_shape)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        scaled_hat = self.gaussian_conditional.decompress(
            [segment.y_string],
            indexes,
            dtype=means_hat.dtype,
            means=means_hat,
        )
        return scaled_hat * q_step

    def aux_loss(self) -> Tensor:
        return self.entropy_bottleneck.loss()

    def update(self, force: bool = False) -> bool:
        from compressai.models.google import get_scale_table

        updated = bool(self.entropy_bottleneck.update(force=force))
        scale_table = get_scale_table().to(next(self.parameters()).device)
        updated = bool(self.gaussian_conditional.update_scale_table(scale_table, force=force)) or updated
        return updated


class BAANICModel(nn.Module):
    """End-to-end BAA-NIC skeleton for wiring training and evaluation."""

    def __init__(self, use_entropy_model: bool = False, entropy_model_type: str = "factorized") -> None:
        super().__init__()
        self.encoder = LightweightEncoder()
        self.grouper = LatentGrouper()
        self.decoder = ServerDecoder()
        self.controller = RuleBasedBandwidthController()
        self.use_entropy_model = use_entropy_model
        if entropy_model_type not in ENTROPY_MODEL_TYPES:
            raise ValueError(f"entropy_model_type must be one of {ENTROPY_MODEL_TYPES}, got {entropy_model_type}")
        self.entropy_model_type = entropy_model_type if use_entropy_model else "none"
        self.entropy_bottleneck = self._build_entropy_bottleneck() if self.entropy_model_type == "factorized" else None
        self.hyperprior = BAANICHyperpriorEntropyModel() if self.entropy_model_type == "hyperprior" else None
        self.group_hyperprior = (
            BAANICGroupHyperpriorEntropyModel() if self.entropy_model_type == "group_hyperprior" else None
        )
        self.codec_checkpoint_sha256 = "0" * 64
        self.codec_contract_sha256 = "0" * 64

    def _build_entropy_bottleneck(self) -> nn.Module:
        try:
            from compressai.entropy_models import EntropyBottleneck
        except ImportError as exc:
            raise RuntimeError("compressai is required when use_entropy_model=True") from exc
        return EntropyBottleneck(LATENT_CHANNELS)

    def aux_loss(self) -> Tensor:
        if self.group_hyperprior is not None:
            return self.group_hyperprior.aux_loss()
        if self.hyperprior is not None:
            return self.hyperprior.aux_loss()
        if self.entropy_bottleneck is None:
            return next(self.parameters()).new_tensor(0.0)
        return self.entropy_bottleneck.loss()

    def update_entropy_bottleneck(self, force: bool = False) -> bool:
        if self.group_hyperprior is not None:
            return self.group_hyperprior.update(force=force)
        if self.hyperprior is not None:
            return self.hyperprior.update(force=force)
        if self.entropy_bottleneck is None:
            return False
        return bool(self.entropy_bottleneck.update(force=force))

    def set_codec_metadata(self, checkpoint_sha256: str, contract_sha256: str) -> None:
        if len(checkpoint_sha256) != 64 or len(contract_sha256) != 64:
            raise ValueError("Codec metadata requires 64-character SHA256 values")
        self.codec_checkpoint_sha256 = checkpoint_sha256.lower()
        self.codec_contract_sha256 = contract_sha256.lower()

    def forward(
        self,
        x: Tensor,
        active_ratio: float | None = None,
        q_index: int | None = None,
        link_state: dict[str, Any] | None = None,
    ) -> dict[str, Tensor | float | int | str]:
        policy: BandwidthPolicy | None = None
        if active_ratio is None or q_index is None:
            policy = self.controller(link_state)
            active_ratio = policy.active_ratio if active_ratio is None else active_ratio
            q_index = policy.q_index if q_index is None else q_index

        latent = self.encoder(x)
        active_mask = self.grouper.mask(latent, float(active_ratio))
        masked_latent = latent * active_mask
        transmitted_latent = masked_latent
        likelihoods: Tensor | None = None
        z_likelihoods: Tensor | None = None
        rate_notes = "RATE_PROXY_TARGET_BPP_NOT_ENTROPY"

        if self.group_hyperprior is not None:
            active_groups = self.grouper.active_groups(float(active_ratio))
            entropy_output = self.group_hyperprior(
                latent,
                active_group_ids=range(1, active_groups + 1),
                q_index=int(q_index),
                image_size=x.shape[-2:],
            )
            transmitted_latent = entropy_output["latent_hat"]
            likelihoods = entropy_output["y_likelihoods"]
            z_likelihoods = entropy_output["z_likelihoods"]
            rate_proxy = entropy_output["bpp"]
            rate_notes = "SHARED_GROUP_HYPERPRIOR_LIKELIHOOD_BPP"
        elif self.hyperprior is not None:
            entropy_output = self.hyperprior(masked_latent, active_mask, image_size=x.shape[-2:])
            transmitted_latent = entropy_output["latent_hat"] * active_mask
            likelihoods = entropy_output["y_likelihoods"]
            z_likelihoods = entropy_output["z_likelihoods"]
            rate_proxy = entropy_output["bpp"]
            rate_notes = "LEARNED_HYPERPRIOR_BPP_ESTIMATE"
        elif self.entropy_bottleneck is not None:
            quantized_latent, likelihoods = self.entropy_bottleneck(masked_latent)
            transmitted_latent = quantized_latent * active_mask
            likelihood_bits = -torch.log2(likelihoods.clamp_min(1e-9)) * active_mask
            rate_proxy = likelihood_bits.sum() / float(x.shape[0] * x.shape[-2] * x.shape[-1])
            rate_notes = "LEARNED_ENTROPY_BOTTLENECK_BPP_ESTIMATE"
        else:
            # Proxy only: replace with learned entropy hyperprior before final results.
            active_fraction = active_mask[:, :, :1, :1].mean()
            rate_proxy = active_fraction + 0.001 * masked_latent.abs().mean()

        group_indicator = self.grouper.group_indicator(latent, float(active_ratio))
        reconstruction = self.decoder(transmitted_latent, group_indicator, output_size=x.shape[-2:])

        return {
            "reconstruction": reconstruction,
            "latent": latent,
            "masked_latent": transmitted_latent,
            "pre_entropy_masked_latent": masked_latent,
            "active_mask": active_mask,
            "active_ratio": float(active_ratio),
            "q_index": int(q_index),
            "rate_proxy": rate_proxy,
            "entropy_likelihoods": likelihoods if likelihoods is not None else masked_latent.new_empty(0),
            "hyper_likelihoods": z_likelihoods if z_likelihoods is not None else masked_latent.new_empty(0),
            "rate_notes": rate_notes,
            "controller_reason": policy.reason if policy else "manual",
        }

    @torch.no_grad()
    def compress(
        self,
        x: Tensor,
        q_index: int,
        checkpoint_sha256: str | None = None,
        contract_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Encode one image into a full four-segment practical bitstream."""

        if self.group_hyperprior is None:
            raise RuntimeError("compress requires entropy_model_type='group_hyperprior'")
        if self.training:
            raise RuntimeError("Call model.eval() before practical compression")
        if x.ndim != 4 or x.shape[0] != 1 or x.shape[1] != 3:
            raise ValueError("compress expects an NCHW tensor with shape 1x3xHxW")
        self.group_hyperprior.q_step(int(q_index))

        height, width = int(x.shape[-2]), int(x.shape[-1])
        pad_bottom = (-height) % 16
        pad_right = (-width) % 16
        padded = F.pad(x, (0, pad_right, 0, pad_bottom))
        latent = self.encoder(padded)
        segments: list[bytes] = []
        quantized_groups: list[Tensor] = []
        for group_index, group in enumerate(latent.split(GROUP_CHANNELS, dim=1), start=1):
            segment, quantized_group = self.group_hyperprior.compress_group_with_latent(
                group, int(q_index), group_index
            )
            segments.append(segment)
            quantized_groups.append(quantized_group)

        header = BitstreamHeader(
            height=height,
            width=width,
            padding=(0, pad_bottom, 0, pad_right),
            q_index=int(q_index),
            checkpoint_sha256=checkpoint_sha256 or self.codec_checkpoint_sha256,
            contract_sha256=contract_sha256 or self.codec_contract_sha256,
            segment_lengths=tuple(len(segment) for segment in segments),
            latent_shape=(int(latent.shape[-2]), int(latent.shape[-1])),
        )
        bitstream = assemble_full_stream(header, segments)
        return {
            "bitstream": bitstream,
            "segments": tuple(segments),
            "q_index": int(q_index),
            "input_size": (height, width),
            "latent_shape": header.latent_shape,
            "segment_lengths": header.segment_lengths,
            "masked_latent": torch.cat(quantized_groups, dim=1),
        }

    @torch.no_grad()
    def decompress_latent(self, bitstream: bytes | dict[str, Any]) -> dict[str, Any]:
        """Entropy-decode a stream without executing a server decoder."""
        if self.group_hyperprior is None:
            raise RuntimeError("decompress_latent requires entropy_model_type='group_hyperprior'")
        if self.training:
            raise RuntimeError("Call model.eval() before practical entropy decoding")
        raw = bitstream.get("bitstream") if isinstance(bitstream, dict) else bitstream
        if not isinstance(raw, (bytes, bytearray)):
            raise TypeError("bitstream must be bytes or a compress() result")
        header, segments = parse_stream(bytes(raw))
        device = next(self.parameters()).device
        latent = torch.zeros(
            (1, LATENT_CHANNELS, header.latent_shape[0], header.latent_shape[1]),
            dtype=next(self.parameters()).dtype,
            device=device,
        )
        present_group_ids: list[int] = []
        for segment in segments:
            group = self.group_hyperprior.decompress_group(segment, header.q_index)
            if tuple(group.shape[-2:]) != header.latent_shape:
                raise ValueError(f"Decoded group {segment.group_id} has an invalid latent shape")
            start = (segment.group_id - 1) * GROUP_CHANNELS
            latent[:, start : start + GROUP_CHANNELS] = group
            present_group_ids.append(segment.group_id)

        indicator = self.grouper.group_indicator_from_ids(latent, present_group_ids)
        return {
            "masked_latent": latent,
            "group_indicator": indicator,
            "present_group_ids": tuple(present_group_ids),
            "active_ratio": len(present_group_ids) / 4.0,
            "q_index": header.q_index,
            "header": header,
            "bitstream_bytes": len(raw),
        }

    @torch.no_grad()
    def decompress(self, bitstream: bytes | dict[str, Any]) -> dict[str, Any]:
        """Decode any valid full, prefix, reordered, or subset stream."""

        decoded = self.decompress_latent(bitstream)
        header = decoded["header"]
        pad_top, pad_bottom, pad_left, pad_right = header.padding
        padded_height = header.height + pad_top + pad_bottom
        padded_width = header.width + pad_left + pad_right
        reconstruction = self.decoder(
            decoded["masked_latent"],
            decoded["group_indicator"],
            output_size=(padded_height, padded_width),
        )
        reconstruction = reconstruction[
            :,
            :,
            pad_top : pad_top + header.height,
            pad_left : pad_left + header.width,
        ]
        return decoded | {
            "reconstruction": reconstruction.clamp(0.0, 1.0),
        }
