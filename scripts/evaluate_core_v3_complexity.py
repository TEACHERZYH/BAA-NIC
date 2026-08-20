"""Measure the frozen core-v3 sender/receiver complexity and latency protocol."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import BAANICModel, DenseControlEncoder, LightServerDecoder, ServerDecoder
from models.baa_nic import GROUP_CHANNELS
from models.baa_nic_bitstream import pack_segment, parse_stream
from models.core_v3_controller import ActionCalibration, CoreV3BandwidthController, WorkerAvailability
from core_v3_contract import (
    contract_sha256,
    file_sha256,
    files_sha256,
    load_contract,
    sender_state_dict_sha256,
    state_dict_sha256,
)
from core_v3_data import CoreV3FixedTileDataset
from core_v3_training import configure_practical_determinism


FIELDS = [
    "contract_version", "contract_sha256", "code_hash", "checkpoint_sha256", "manifest_sha256",
    "model", "component", "side", "device", "image_id", "repeat", "threads", "params",
    "checkpoint_bytes", "macs_per_pixel", "gflops_512", "peak_memory_bytes", "latency_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--heavy-decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--light-decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visdrone-root", type=Path, required=True)
    parser.add_argument("--devices", default="cuda,cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--timing-images", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=50)
    parser.add_argument("--development-limit", type=int, default=256)
    return parser.parse_args()


def parameter_count(modules: Iterable[Any]) -> int:
    seen: set[int] = set()
    total = 0
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                total += parameter.numel()
    return total


def serialized_state_bytes(named_modules: Iterable[tuple[str, Any]]) -> int:
    import torch

    state: dict[str, Any] = {}
    for prefix, module in named_modules:
        state.update({f"{prefix}.{key}": value.detach().cpu() for key, value in module.state_dict().items()})
    buffer = BytesIO()
    torch.save(state, buffer)
    return buffer.tell()


def composite_hash(sender_hash: str, decoder: Any) -> str:
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(sender_hash))
    digest.update(bytes.fromhex(state_dict_sha256(decoder)))
    return digest.hexdigest()


def macs_for_call(module: Any, *inputs: Any) -> int:
    from torch import nn

    macs = 0
    handles = []

    def hook(layer: Any, layer_inputs: tuple[Any, ...], output: Any) -> None:
        nonlocal macs
        if isinstance(layer, nn.Conv2d):
            batch, out_channels, out_h, out_w = output.shape
            kernel_h, kernel_w = layer.kernel_size
            macs += (
                batch * out_channels * out_h * out_w * (layer.in_channels // layer.groups) * kernel_h * kernel_w
            )
        elif isinstance(layer, nn.ConvTranspose2d):
            batch, in_channels, in_h, in_w = layer_inputs[0].shape
            kernel_h, kernel_w = layer.kernel_size
            macs += (
                batch * in_channels * in_h * in_w * (layer.out_channels // layer.groups) * kernel_h * kernel_w
            )
        elif isinstance(layer, nn.Linear):
            macs += layer_inputs[0].numel() // layer.in_features * layer.in_features * layer.out_features

    for layer in module.modules():
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            handles.append(layer.register_forward_hook(hook))
    module.eval()
    with __import__("torch").inference_mode():
        module(*inputs)
    for handle in handles:
        handle.remove()
    return int(macs)


def timed_call(fn: Callable[[], Any], device: Any, host_bound: bool = False) -> tuple[Any, float, int]:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        if host_bound:
            started = time.perf_counter()
            output = fn()
            torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
        else:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = fn()
            end.record()
            torch.cuda.synchronize(device)
            elapsed_ms = float(start.elapsed_time(end))
        peak = max(0, int(torch.cuda.max_memory_allocated(device) - baseline))
        return output, elapsed_ms, peak
    started = time.perf_counter()
    output = fn()
    return output, (time.perf_counter() - started) * 1000.0, 0


def accumulate(target: dict[str, list[float]], name: str, elapsed_ms: float, peak: int) -> None:
    target[f"{name}:latency"].append(float(elapsed_ms))
    target[f"{name}:peak"].append(float(peak))


def collapse(measurements: dict[str, list[float]]) -> dict[str, tuple[float, int]]:
    names = {key.rsplit(":", 1)[0] for key in measurements}
    return {
        name: (
            float(sum(measurements.get(f"{name}:latency", []))),
            int(max(measurements.get(f"{name}:peak", [0.0]))),
        )
        for name in names
    }


def load_joint(path: Path, device: Any) -> BAANICModel:
    import torch

    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("kind") != "core_v3_joint":
        raise RuntimeError("Expected a core_v3_joint checkpoint")
    model = BAANICModel(use_entropy_model=True, entropy_model_type="group_hyperprior").to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    model.update_entropy_bottleneck(force=True)
    return model


def load_decoder(path: Path, kind: str, device: Any) -> Any:
    import torch

    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("kind") != f"core_v3_decoder_{kind}":
        raise RuntimeError(f"Expected a core_v3_decoder_{kind} checkpoint")
    decoder = (ServerDecoder() if kind == "heavy" else LightServerDecoder()).to(device)
    decoder.load_state_dict(payload["decoder"])
    decoder.eval()
    return decoder


def controller_for_timing() -> CoreV3BandwidthController:
    actions = []
    for q_index in range(4):
        for prefix_k in range(1, 5):
            actions.append(
                ActionCalibration(
                    q_index=q_index,
                    prefix_k=prefix_k,
                    utility=1.0 if (q_index, prefix_k) == (1, 4) else 0.5 - 0.01 * q_index + 0.001 * prefix_k,
                    payload_p90_bytes=1000.0 * prefix_k,
                    payload_mean_bytes=900.0 * prefix_k,
                    encode_p95_s=0.01,
                    encode_mean_s=0.008,
                    decode_p95_s=0.01,
                    decode_mean_s=0.008,
                )
            )
    return CoreV3BandwidthController(
        actions,
        loss_bin_edges=(0.0, 0.05, 0.15, 1.0),
        goodput_lower_quantiles=(0.8, 0.75, 0.7),
        global_goodput_lower_quantile=0.75,
        minimum_bin_samples=(100, 100, 100),
        risk_aware=True,
    )


def measure_controller(controller: CoreV3BandwidthController, lengths: tuple[int, ...], device: Any) -> tuple[float, int]:
    cumulative = {index: 144 + sum(lengths[:index]) for index in range(1, 5)}

    def run() -> None:
        decision = controller.select_preencode(
            estimated_goodput_bps=50_000_000.0,
            packet_loss_estimate=0.01,
            frame_arrival_s=0.0,
            deadline_s=1.0,
            workers=WorkerAvailability(0.0, 0.0, 0.0),
        )
        transmitted = controller.select_postencode(decision, cumulative)
        if (transmitted.q_index, transmitted.transmitted_prefix_k) != (1, 4):
            raise RuntimeError("Timing controller did not select the frozen q1/full action")

    _, elapsed_ms, peak = timed_call(run, device, host_bound=True)
    return elapsed_ms, peak


def instrument_baa_sender(model: BAANICModel, image: Any, q_index: int, device: Any) -> dict[str, tuple[float, int]]:
    entropy = model.group_hyperprior
    if entropy is None:
        raise RuntimeError("Missing group hyperprior")
    measurements: dict[str, list[float]] = defaultdict(list)
    latent, ms, peak = timed_call(lambda: model.encoder(image), device)
    accumulate(measurements, "analysis_transform", ms, peak)
    q_step = entropy.q_step(q_index)
    for group_id, group in enumerate(latent.split(GROUP_CHANNELS, dim=1), start=1):
        scaled, ms, peak = timed_call(lambda group=group: group / q_step, device)
        accumulate(measurements, "q_scaling", ms, peak)
        z, ms, peak = timed_call(lambda scaled=scaled: entropy.hyper_analysis(scaled.abs()), device)
        accumulate(measurements, "group_hyper_analysis", ms, peak)

        def encode_hyper(z=z):
            strings = entropy.entropy_bottleneck.compress(z)
            return strings, entropy.entropy_bottleneck.decompress(strings, z.shape[-2:])

        (z_strings, z_hat), ms, peak = timed_call(encode_hyper, device, host_bound=True)
        accumulate(measurements, "entropy_encode", ms, peak)

        def predict(z_hat=z_hat, scaled=scaled):
            scales, means = entropy._distribution_parameters(z_hat, scaled.shape[-2:])
            return scales, means, entropy.gaussian_conditional.build_indexes(scales)

        (scales_hat, means_hat, indexes), ms, peak = timed_call(predict, device)
        accumulate(measurements, "sender_probability_prediction", ms, peak)

        def encode_main(scaled=scaled, indexes=indexes, means_hat=means_hat):
            strings = entropy.gaussian_conditional.compress(scaled, indexes, means=means_hat)
            quantized = entropy.gaussian_conditional.quantize(scaled, "dequantize", means=means_hat)
            return strings, quantized

        (y_strings, _), ms, peak = timed_call(encode_main, device, host_bound=True)
        accumulate(measurements, "entropy_encode", ms, peak)
        _, ms, peak = timed_call(
            lambda: pack_segment(
                group_id,
                z_strings[0],
                y_strings[0],
                (int(z.shape[-2]), int(z.shape[-1])),
                (int(scaled.shape[-2]), int(scaled.shape[-1])),
            ),
            device,
            host_bound=True,
        )
        accumulate(measurements, "packing", ms, peak)
    return collapse(measurements)


def instrument_baa_receiver(model: BAANICModel, stream: bytes, decoder: Any, device: Any) -> dict[str, tuple[float, int]]:
    import torch

    entropy = model.group_hyperprior
    if entropy is None:
        raise RuntimeError("Missing group hyperprior")
    measurements: dict[str, list[float]] = defaultdict(list)
    (header, segments), ms, peak = timed_call(lambda: parse_stream(stream), device, host_bound=True)
    accumulate(measurements, "unpacking", ms, peak)
    latent = torch.zeros((1, 192, *header.latent_shape), device=device)
    present: list[int] = []
    for segment in segments:
        z_hat, ms, peak = timed_call(
            lambda segment=segment: entropy.entropy_bottleneck.decompress([segment.z_string], segment.z_shape),
            device,
            host_bound=True,
        )
        accumulate(measurements, "entropy_decode", ms, peak)

        def predict(z_hat=z_hat, segment=segment):
            scales, means = entropy._distribution_parameters(z_hat, segment.y_shape)
            return means, entropy.gaussian_conditional.build_indexes(scales)

        (means_hat, indexes), ms, peak = timed_call(predict, device)
        accumulate(measurements, "receiver_probability_prediction", ms, peak)

        def decode_main(segment=segment, indexes=indexes, means_hat=means_hat):
            return entropy.gaussian_conditional.decompress(
                [segment.y_string], indexes, dtype=means_hat.dtype, means=means_hat
            )

        scaled_hat, ms, peak = timed_call(decode_main, device, host_bound=True)
        accumulate(measurements, "entropy_decode", ms, peak)
        group, ms, peak = timed_call(lambda scaled_hat=scaled_hat: scaled_hat * entropy.q_step(header.q_index), device)
        accumulate(measurements, "q_rescaling", ms, peak)
        start = (segment.group_id - 1) * GROUP_CHANNELS
        latent[:, start : start + GROUP_CHANNELS] = group
        present.append(segment.group_id)
    indicator = model.grouper.group_indicator_from_ids(latent, present)
    _, ms, peak = timed_call(lambda: decoder(latent, indicator, output_size=(header.height, header.width)), device)
    accumulate(measurements, "decoder", ms, peak)
    return collapse(measurements)


def instrument_anchor_sender(model: Any, image: Any, device: Any) -> dict[str, tuple[float, int]]:
    measurements: dict[str, list[float]] = defaultdict(list)
    y, ms, peak = timed_call(lambda: model.g_a(image), device)
    accumulate(measurements, "analysis_transform", ms, peak)
    z, ms, peak = timed_call(lambda: model.h_a(y.abs()), device)
    accumulate(measurements, "hyper_analysis", ms, peak)

    def encode_hyper():
        strings = model.entropy_bottleneck.compress(z)
        return strings, model.entropy_bottleneck.decompress(strings, z.shape[-2:])

    (z_strings, z_hat), ms, peak = timed_call(encode_hyper, device, host_bound=True)
    accumulate(measurements, "entropy_encode", ms, peak)

    def predict():
        scales = model.h_s(z_hat)
        return scales, model.gaussian_conditional.build_indexes(scales)

    (_, indexes), ms, peak = timed_call(predict, device)
    accumulate(measurements, "sender_probability_prediction", ms, peak)
    _, ms, peak = timed_call(
        lambda: model.gaussian_conditional.compress(y, indexes), device, host_bound=True
    )
    accumulate(measurements, "entropy_encode", ms, peak)
    return collapse(measurements)


def instrument_anchor_receiver(model: Any, output: dict[str, Any], device: Any) -> dict[str, tuple[float, int]]:
    measurements: dict[str, list[float]] = defaultdict(list)
    strings, shape = output["strings"], output["shape"]
    z_hat, ms, peak = timed_call(
        lambda: model.entropy_bottleneck.decompress(strings[1], shape), device, host_bound=True
    )
    accumulate(measurements, "entropy_decode", ms, peak)

    def predict():
        scales = model.h_s(z_hat)
        return model.gaussian_conditional.build_indexes(scales)

    indexes, ms, peak = timed_call(predict, device)
    accumulate(measurements, "receiver_probability_prediction", ms, peak)
    y_hat, ms, peak = timed_call(
        lambda: model.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype),
        device,
        host_bound=True,
    )
    accumulate(measurements, "entropy_decode", ms, peak)
    _, ms, peak = timed_call(lambda: model.g_s(y_hat).clamp_(0, 1), device)
    accumulate(measurements, "decoder", ms, peak)
    return collapse(measurements)


def entropy_bytes(output: dict[str, Any]) -> int:
    return sum(len(item) for group in output["strings"] for item in group)


def anchor_quality_selection(model: BAANICModel, dataset: Any, limit: int, device: Any) -> dict[str, Any]:
    import torch
    from compressai.zoo import bmshj2018_hyperprior
    from pytorch_msssim import ms_ssim

    count = min(limit, len(dataset))
    baa_bpp: list[float] = []
    with torch.inference_mode():
        for index in range(count):
            image = dataset[index]["image"].unsqueeze(0).to(device)
            output = model.compress(image, q_index=1)
            baa_bpp.append(len(output["bitstream"]) * 8.0 / (image.shape[-2] * image.shape[-1]))
    baa_median = float(np.median(baa_bpp))
    candidates: list[dict[str, float | int]] = []
    for quality in range(1, 9):
        anchor = bmshj2018_hyperprior(quality=quality, pretrained=True).to(device).eval()
        anchor.update(force=True)
        bpps: list[float] = []
        with torch.inference_mode():
            for index in range(count):
                image = dataset[index]["image"].unsqueeze(0).to(device)
                output = anchor.compress(image)
                bpps.append(entropy_bytes(output) * 8.0 / (image.shape[-2] * image.shape[-1]))
        median = float(np.median(bpps))
        candidates.append(
            {
                "quality": quality,
                "median_bpp": median,
                "relative_bpp_difference": abs(median - baa_median) / max(baa_median, 1e-12),
            }
        )
        del anchor
        if device.type == "cuda":
            torch.cuda.empty_cache()
    selected = min(candidates, key=lambda item: (float(item["relative_bpp_difference"]), int(item["quality"])))
    anchor = bmshj2018_hyperprior(quality=int(selected["quality"]), pretrained=True).to(device).eval()
    anchor.update(force=True)
    selected_bpp: list[float] = []
    psnr_values: list[float] = []
    msssim_values: list[float] = []
    with torch.inference_mode():
        for index in range(count):
            image = dataset[index]["image"].unsqueeze(0).to(device)
            encoded = anchor.compress(image)
            reconstruction = anchor.decompress(encoded["strings"], encoded["shape"])["x_hat"]
            mse = float((reconstruction - image).square().mean().clamp_min(1e-12).cpu())
            selected_bpp.append(entropy_bytes(encoded) * 8.0 / (image.shape[-2] * image.shape[-1]))
            psnr_values.append(-10.0 * math.log10(mse))
            msssim_values.append(float(ms_ssim(reconstruction.float(), image.float(), data_range=1.0).cpu()))
    return {
        "baa_q1_full_median_bpp": baa_median,
        "candidates": candidates,
        "selected_quality": int(selected["quality"]),
        "selected_relative_bpp_difference": float(selected["relative_bpp_difference"]),
        "within_pre_registered_10pct_tolerance": float(selected["relative_bpp_difference"]) <= 0.10,
        "selected_anchor_median_bpp": float(np.median(selected_bpp)),
        "selected_anchor_mean_psnr": statistics.fmean(psnr_values),
        "selected_anchor_mean_ms_ssim": statistics.fmean(msssim_values),
    }


def build_stats(model: BAANICModel, dense: BAANICModel, heavy: Any, light: Any, anchor: Any, device: Any) -> dict[tuple[str, str], dict[str, int]]:
    import torch

    image = torch.zeros((1, 3, 512, 512), device=device)
    latent = model.encoder(image)
    indicator = torch.ones((1, 4, 32, 32), device=device)
    sender_modules = [model.encoder, model.group_hyperprior]
    dense_modules = [dense.encoder, dense.group_hyperprior]
    entropy = model.group_hyperprior
    if entropy is None or dense.group_hyperprior is None:
        raise RuntimeError("Missing group hyperprior")
    encoder_macs = macs_for_call(model.encoder, image)
    dense_encoder_macs = macs_for_call(dense.encoder, image)
    entropy_macs = macs_for_call(entropy, latent, range(1, 5), 1, (512, 512))
    heavy_macs = macs_for_call(heavy, latent, indicator, (512, 512))
    light_macs = macs_for_call(light, latent, indicator, (512, 512))
    anchor_y = anchor.g_a(image)
    anchor_z = anchor.h_a(anchor_y.abs())
    anchor_ga_macs = macs_for_call(anchor.g_a, image)
    anchor_ha_macs = macs_for_call(anchor.h_a, anchor_y.abs())
    anchor_hs_macs = macs_for_call(anchor.h_s, anchor_z)
    anchor_gs_macs = macs_for_call(anchor.g_s, anchor_y)

    def stats(modules: list[Any], macs: int, names: list[tuple[str, Any]]) -> dict[str, int]:
        return {
            "params": parameter_count(modules),
            "checkpoint_bytes": serialized_state_bytes(names),
            "macs": int(macs),
        }

    zero = {"params": 0, "checkpoint_bytes": 0, "macs": 0}
    values: dict[tuple[str, str], dict[str, int]] = {}
    values[("baa_nic_depthwise", "analysis_transform")] = stats(
        [model.encoder], encoder_macs, [("encoder", model.encoder)]
    )
    values[("baa_nic_depthwise", "group_hyper_analysis")] = stats(
        [entropy.hyper_analysis], macs_for_call(entropy.hyper_analysis, latent[:, :48].abs()),
        [("hyper_analysis", entropy.hyper_analysis)],
    )
    values[("baa_nic_depthwise", "sender_probability_prediction")] = stats(
        [entropy.hyper_synthesis], entropy_macs - 4 * values[("baa_nic_depthwise", "group_hyper_analysis")]["macs"],
        [("hyper_synthesis", entropy.hyper_synthesis)],
    )
    values[("baa_nic_depthwise", "entropy_encode")] = stats(
        [entropy.entropy_bottleneck, entropy.gaussian_conditional], 0,
        [("entropy_bottleneck", entropy.entropy_bottleneck), ("gaussian_conditional", entropy.gaussian_conditional)],
    )
    for component in ("q_scaling", "packing", "controller"):
        values[("baa_nic_depthwise", component)] = dict(zero)
    values[("baa_nic_depthwise", "full_sender")] = stats(
        sender_modules, encoder_macs + entropy_macs,
        [("encoder", model.encoder), ("group_hyperprior", entropy)],
    )
    values[("dense_conv_control", "analysis_transform")] = stats(
        [dense.encoder], dense_encoder_macs, [("encoder", dense.encoder)]
    )
    values[("dense_conv_control", "full_sender")] = stats(
        dense_modules, dense_encoder_macs + entropy_macs,
        [("encoder", dense.encoder), ("group_hyperprior", dense.group_hyperprior)],
    )
    receiver_entropy_modules = [entropy.entropy_bottleneck, entropy.gaussian_conditional, entropy.hyper_synthesis]
    receiver_entropy_macs = values[("baa_nic_depthwise", "sender_probability_prediction")]["macs"]
    for model_name, decoder, decoder_macs in (
        ("baa_nic_server_heavy", heavy, heavy_macs),
        ("baa_nic_symmetric_light", light, light_macs),
    ):
        values[(model_name, "receiver_probability_prediction")] = stats(
            [entropy.hyper_synthesis], receiver_entropy_macs, [("hyper_synthesis", entropy.hyper_synthesis)]
        )
        values[(model_name, "entropy_decode")] = stats(
            [entropy.entropy_bottleneck, entropy.gaussian_conditional], 0,
            [("entropy_bottleneck", entropy.entropy_bottleneck), ("gaussian_conditional", entropy.gaussian_conditional)],
        )
        for component in ("q_rescaling", "unpacking"):
            values[(model_name, component)] = dict(zero)
        values[(model_name, "decoder")] = stats([decoder], decoder_macs, [("decoder", decoder)])
        values[(model_name, "full_receiver")] = stats(
            receiver_entropy_modules + [decoder], receiver_entropy_macs + decoder_macs,
            [
                ("entropy_bottleneck", entropy.entropy_bottleneck),
                ("gaussian_conditional", entropy.gaussian_conditional),
                ("hyper_synthesis", entropy.hyper_synthesis),
                ("decoder", decoder),
            ],
        )
    anchor_name = f"bmshj2018_hyperprior_q{getattr(anchor, 'quality', 'selected')}"
    anchor_sender_modules = [anchor.g_a, anchor.h_a, anchor.h_s, anchor.entropy_bottleneck, anchor.gaussian_conditional]
    anchor_receiver_modules = [anchor.h_s, anchor.entropy_bottleneck, anchor.gaussian_conditional, anchor.g_s]
    for component, modules, macs, names in (
        ("analysis_transform", [anchor.g_a], anchor_ga_macs, [("g_a", anchor.g_a)]),
        ("hyper_analysis", [anchor.h_a], anchor_ha_macs, [("h_a", anchor.h_a)]),
        ("sender_probability_prediction", [anchor.h_s], anchor_hs_macs, [("h_s", anchor.h_s)]),
        ("entropy_encode", [anchor.entropy_bottleneck, anchor.gaussian_conditional], 0,
         [("entropy_bottleneck", anchor.entropy_bottleneck), ("gaussian_conditional", anchor.gaussian_conditional)]),
        ("full_sender", anchor_sender_modules, anchor_ga_macs + anchor_ha_macs + anchor_hs_macs,
         [("g_a", anchor.g_a), ("h_a", anchor.h_a), ("h_s", anchor.h_s),
          ("entropy_bottleneck", anchor.entropy_bottleneck), ("gaussian_conditional", anchor.gaussian_conditional)]),
        ("receiver_probability_prediction", [anchor.h_s], anchor_hs_macs, [("h_s", anchor.h_s)]),
        ("entropy_decode", [anchor.entropy_bottleneck, anchor.gaussian_conditional], 0,
         [("entropy_bottleneck", anchor.entropy_bottleneck), ("gaussian_conditional", anchor.gaussian_conditional)]),
        ("decoder", [anchor.g_s], anchor_gs_macs, [("g_s", anchor.g_s)]),
        ("full_receiver", anchor_receiver_modules, anchor_hs_macs + anchor_gs_macs,
         [("h_s", anchor.h_s), ("entropy_bottleneck", anchor.entropy_bottleneck),
          ("gaussian_conditional", anchor.gaussian_conditional), ("g_s", anchor.g_s)]),
    ):
        values[(anchor_name, component)] = stats(modules, macs, names)
    return values


def write_measurement(
    writer: Any,
    common: dict[str, Any],
    stats: dict[tuple[str, str], dict[str, int]],
    model_name: str,
    seed: int,
    component: str,
    side: str,
    device_name: str,
    image_id: str,
    repeat: int,
    threads: int,
    elapsed_ms: float,
    peak: int,
) -> None:
    values = stats[(model_name, component)]
    pixels = 512 * 512
    writer.writerow(
        common
        | {
            "model": f"{model_name}_seed{seed}",
            "component": component,
            "side": side,
            "device": device_name,
            "image_id": image_id,
            "repeat": repeat,
            "threads": threads,
            "params": values["params"],
            "checkpoint_bytes": values["checkpoint_bytes"],
            "macs_per_pixel": f"{values['macs'] / pixels:.8f}",
            "gflops_512": f"{2.0 * values['macs'] / 1e9:.8f}",
            "peak_memory_bytes": peak,
            "latency_ms": f"{elapsed_ms:.8f}",
        }
    )


def main() -> None:
    import torch
    from compressai.zoo import bmshj2018_hyperprior

    args = parse_args()
    configure_practical_determinism(args.seed)
    contract = load_contract()
    if args.seed not in contract["training"]["model_seeds"]:
        raise ValueError("Seed is outside the frozen contract")
    if (args.timing_images, args.repeats, args.warmups) != (100, 5, 50):
        raise ValueError("Formal E1 timing is frozen at 100 images, five repeats and 50 warmups")
    if args.development_limit != 256:
        raise ValueError("Formal CompressAI operating-point selection uses all 256 development tiles")
    torch.set_num_threads(args.threads)
    manifests = {
        "development": ROOT / "data" / "core_v3_visdrone_development.csv",
        "timing": ROOT / "data" / "core_v3_visdrone_test.csv",
    }
    development = CoreV3FixedTileDataset(
        manifests["development"], limit=args.development_limit, data_root=args.visdrone_root
    )
    timing = CoreV3FixedTileDataset(manifests["timing"], limit=args.timing_images, data_root=args.visdrone_root)
    code_hash = files_sha256(
        [
            ROOT / "models" / "baa_nic.py",
            ROOT / "models" / "baa_nic_bitstream.py",
            ROOT / "models" / "core_v3_controls.py",
            ROOT / "models" / "core_v3_controller.py",
            ROOT / "scripts" / "core_v3_training.py",
            Path(__file__).resolve(),
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "core_v3_complexity_breakdown.csv"
    metadata: dict[str, Any] = {
        "status": "running",
        "contract_version": contract["contract_version"],
        "contract_sha256": contract_sha256(),
        "seed": args.seed,
        "timing_protocol": {
            "images": args.timing_images,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "batch_size": 1,
            "precision": "fp32",
            "threads": args.threads,
            "gpu_module_timer": "cuda_event_with_synchronization",
            "host_bound_timer": "perf_counter_with_cuda_synchronization",
            "peak_memory": "cuda_peak_allocated_delta_from_pre_call_baseline",
        },
        "anchor_byte_accounting": "sum_of_official_compressai_entropy_strings_without_container_overhead",
        "devices": {},
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        frozen_anchor_selection: dict[str, Any] | None = None
        for requested_device in [item.strip() for item in args.devices.split(",") if item.strip()]:
            if requested_device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            device = torch.device(requested_device)
            device_name = "cpu" if device.type == "cpu" else f"cuda:{torch.cuda.get_device_name(device)}"
            model = load_joint(args.checkpoint, device)
            heavy = load_decoder(args.heavy_decoder_checkpoint, "heavy", device)
            light = load_decoder(args.light_decoder_checkpoint, "light", device)
            model.decoder = heavy
            sender_hash = sender_state_dict_sha256(model)
            model.set_codec_metadata(sender_hash, contract_sha256())
            dense = load_joint(args.checkpoint, device)
            torch.manual_seed(20260715 + args.seed)
            dense.encoder = DenseControlEncoder().to(device).eval()
            dense_hash = sender_state_dict_sha256(dense)
            dense.set_codec_metadata(dense_hash, contract_sha256())
            if frozen_anchor_selection is None:
                frozen_anchor_selection = anchor_quality_selection(
                    model, development, args.development_limit, device
                )
            selection = frozen_anchor_selection
            selected_quality = int(selection["selected_quality"])
            anchor = bmshj2018_hyperprior(quality=selected_quality, pretrained=True).to(device).eval()
            anchor.quality = selected_quality
            anchor.update(force=True)
            anchor_name = f"bmshj2018_hyperprior_q{selected_quality}"
            anchor_hash = state_dict_sha256(anchor)
            heavy_hash = state_dict_sha256(heavy)
            light_hash = state_dict_sha256(light)
            stats = build_stats(model, dense, heavy, light, anchor, device)
            controller = controller_for_timing()
            checkpoint_hashes = {
                "baa_nic_depthwise": sender_hash,
                "dense_conv_control": dense_hash,
                "baa_nic_server_heavy": composite_hash(sender_hash, heavy),
                "baa_nic_symmetric_light": composite_hash(sender_hash, light),
                anchor_name: anchor_hash,
            }
            common_base = {
                "contract_version": contract["contract_version"],
                "contract_sha256": contract_sha256(),
                "code_hash": code_hash,
                "manifest_sha256": file_sha256(manifests["timing"]),
            }

            warm_image = timing[0]["image"].unsqueeze(0).to(device)
            for _ in range(args.warmups):
                baa_stream = model.compress(warm_image, q_index=1)
                model.decompress(baa_stream)
                decoded = model.decompress_latent(baa_stream)
                light(decoded["masked_latent"], decoded["group_indicator"], output_size=(512, 512))
                dense.compress(warm_image, q_index=1)
                anchor_stream = anchor.compress(warm_image)
                anchor.decompress(anchor_stream["strings"], anchor_stream["shape"])
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            rows_written = 0
            with torch.inference_mode():
                for image_index in range(len(timing)):
                    sample = timing[image_index]
                    image = sample["image"].unsqueeze(0).to(device)
                    image_id = sample["metadata"]["tile_id"]
                    for repeat in range(1, args.repeats + 1):
                        baa_components = instrument_baa_sender(model, image, 1, device)
                        baa_output, full_ms, full_peak = timed_call(
                            lambda: model.compress(image, q_index=1), device, host_bound=True
                        )
                        control_ms, control_peak = measure_controller(
                            controller, tuple(baa_output["segment_lengths"]), device
                        )
                        baa_components["controller"] = (control_ms, control_peak)
                        baa_components["full_sender"] = (full_ms + control_ms, max(full_peak, control_peak))
                        for component, (elapsed_ms, peak) in baa_components.items():
                            write_measurement(
                                writer,
                                common_base | {"checkpoint_sha256": checkpoint_hashes["baa_nic_depthwise"]},
                                stats,
                                "baa_nic_depthwise",
                                args.seed,
                                component,
                                "sender",
                                device_name,
                                image_id,
                                repeat,
                                args.threads,
                                elapsed_ms,
                                peak,
                            )
                            rows_written += 1

                        for decoder_name, decoder in (
                            ("baa_nic_server_heavy", heavy),
                            ("baa_nic_symmetric_light", light),
                        ):
                            components = instrument_baa_receiver(
                                model, baa_output["bitstream"], decoder, device
                            )
                            original_decoder = model.decoder
                            model.decoder = decoder
                            _, full_ms, full_peak = timed_call(
                                lambda: model.decompress(baa_output), device, host_bound=True
                            )
                            model.decoder = original_decoder
                            components["full_receiver"] = (full_ms, full_peak)
                            for component, (elapsed_ms, peak) in components.items():
                                write_measurement(
                                    writer,
                                    common_base | {"checkpoint_sha256": checkpoint_hashes[decoder_name]},
                                    stats,
                                    decoder_name,
                                    args.seed,
                                    component,
                                    "receiver",
                                    device_name,
                                    image_id,
                                    repeat,
                                    args.threads,
                                    elapsed_ms,
                                    peak,
                                )
                                rows_written += 1

                        _, dense_analysis_ms, dense_analysis_peak = timed_call(
                            lambda: dense.encoder(image), device
                        )
                        dense_output, dense_full_ms, dense_full_peak = timed_call(
                            lambda: dense.compress(image, q_index=1), device, host_bound=True
                        )
                        dense_control_ms, dense_control_peak = measure_controller(
                            controller, tuple(dense_output["segment_lengths"]), device
                        )
                        write_measurement(
                            writer,
                            common_base | {"checkpoint_sha256": checkpoint_hashes["dense_conv_control"]},
                            stats,
                            "dense_conv_control",
                            args.seed,
                            "analysis_transform",
                            "sender",
                            device_name,
                            image_id,
                            repeat,
                            args.threads,
                            dense_analysis_ms,
                            dense_analysis_peak,
                        )
                        rows_written += 1
                        write_measurement(
                            writer,
                            common_base | {"checkpoint_sha256": checkpoint_hashes["dense_conv_control"]},
                            stats,
                            "dense_conv_control",
                            args.seed,
                            "full_sender",
                            "sender",
                            device_name,
                            image_id,
                            repeat,
                            args.threads,
                            dense_full_ms + dense_control_ms,
                            max(dense_full_peak, dense_control_peak),
                        )
                        rows_written += 1

                        anchor_components = instrument_anchor_sender(anchor, image, device)
                        anchor_output, anchor_full_ms, anchor_full_peak = timed_call(
                            lambda: anchor.compress(image), device, host_bound=True
                        )
                        anchor_components["full_sender"] = (anchor_full_ms, anchor_full_peak)
                        for component, (elapsed_ms, peak) in anchor_components.items():
                            write_measurement(
                                writer,
                                common_base | {"checkpoint_sha256": checkpoint_hashes[anchor_name]},
                                stats,
                                anchor_name,
                                args.seed,
                                component,
                                "sender",
                                device_name,
                                image_id,
                                repeat,
                                args.threads,
                                elapsed_ms,
                                peak,
                            )
                            rows_written += 1
                        anchor_receiver = instrument_anchor_receiver(anchor, anchor_output, device)
                        _, anchor_decode_ms, anchor_decode_peak = timed_call(
                            lambda: anchor.decompress(anchor_output["strings"], anchor_output["shape"]),
                            device,
                            host_bound=True,
                        )
                        anchor_receiver["full_receiver"] = (anchor_decode_ms, anchor_decode_peak)
                        for component, (elapsed_ms, peak) in anchor_receiver.items():
                            write_measurement(
                                writer,
                                common_base | {"checkpoint_sha256": checkpoint_hashes[anchor_name]},
                                stats,
                                anchor_name,
                                args.seed,
                                component,
                                "receiver",
                                device_name,
                                image_id,
                                repeat,
                                args.threads,
                                elapsed_ms,
                                peak,
                            )
                            rows_written += 1
                    if (image_index + 1) % 10 == 0:
                        handle.flush()
                        print(f"E1: {device_name} {image_index + 1}/{len(timing)}", flush=True)

            metadata["devices"][device_name] = {
                "rows": rows_written,
                "anchor_selection": selection,
                "sender_state_dict_sha256": sender_hash,
                "dense_sender_state_dict_sha256": dense_hash,
                "heavy_decoder_state_dict_sha256": heavy_hash,
                "light_decoder_state_dict_sha256": light_hash,
                "anchor_state_dict_sha256": anchor_hash,
                "complexity_ratios": {
                    "sender_params": stats[("baa_nic_depthwise", "full_sender")]["params"],
                    "full_codec_unique_params": parameter_count(
                        [model.encoder, model.group_hyperprior, heavy]
                    ),
                    "sender_to_full_codec_params": (
                        stats[("baa_nic_depthwise", "full_sender")]["params"]
                        / parameter_count([model.encoder, model.group_hyperprior, heavy])
                    ),
                    "sender_macs": stats[("baa_nic_depthwise", "full_sender")]["macs"],
                    "full_codec_macs": (
                        stats[("baa_nic_depthwise", "full_sender")]["macs"]
                        + stats[("baa_nic_server_heavy", "full_receiver")]["macs"]
                    ),
                    "sender_to_full_codec_macs": (
                        stats[("baa_nic_depthwise", "full_sender")]["macs"]
                        / (
                            stats[("baa_nic_depthwise", "full_sender")]["macs"]
                            + stats[("baa_nic_server_heavy", "full_receiver")]["macs"]
                        )
                    ),
                },
            }
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            del model, dense, heavy, light, anchor
            if device.type == "cuda":
                torch.cuda.empty_cache()

    metadata["status"] = "pass"
    metadata["output_sha256"] = file_sha256(output_path)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: core-v3 E1 seed={args.seed} output={output_path}")


if __name__ == "__main__":
    main()
