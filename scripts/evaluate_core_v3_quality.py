"""Evaluate same-bitstream decoder asymmetry and all 24 progressive group orders."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any
import zlib

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import LightServerDecoder, ServerDecoder
from models.baa_nic_bitstream import HEADER_SIZE, parse_stream, prefix_stream, subset_stream
from core_v3_contract import contract_sha256, file_sha256, files_sha256, load_contract, state_dict_sha256
from core_v3_data import CoreV3FixedTileDataset


ASYMMETRY_FIELDS = [
    "contract_version", "contract_sha256", "code_hash", "checkpoint_sha256", "manifest_sha256",
    "dataset", "seed", "image_id", "q_index", "prefix_k", "decoder", "bitstream_sha256", "bytes",
    "bpp", "psnr", "ms_ssim", "lpips", "decoder_params", "decoder_macs", "latency_ms",
]
ORDERING_FIELDS = [
    "contract_version", "contract_sha256", "code_hash", "checkpoint_sha256", "manifest_sha256",
    "dataset", "seed", "image_id", "q_index", "ordering_id", "budget_percent", "received_group_mask",
    "cumulative_bytes", "bpp", "psnr", "ms_ssim", "lpips", "auc", "violation_flags",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--heavy-checkpoint", type=Path, required=True)
    parser.add_argument("--light-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visdrone-root", type=Path, required=True)
    parser.add_argument("--uavid-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_decoder(path: Path, kind: str, device: Any) -> Any:
    import torch

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("kind") != f"core_v3_decoder_{kind}":
        raise RuntimeError(f"Invalid {kind} decoder checkpoint")
    decoder = (ServerDecoder() if kind == "heavy" else LightServerDecoder()).to(device)
    decoder.load_state_dict(checkpoint["decoder"])
    decoder.eval()
    return decoder


def composite_hash(sender_hash: str, decoder: Any) -> str:
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(sender_hash))
    digest.update(bytes.fromhex(state_dict_sha256(decoder)))
    return digest.hexdigest()


def decode_cache_row(row: tuple[Any, ...]) -> tuple[bytes, Any]:
    import torch

    stream, stream_hash, latent_zlib, latent_hash, latent_shape = row
    stream = bytes(stream)
    if hashlib.sha256(stream).hexdigest() != stream_hash:
        raise RuntimeError("Evaluation stream hash mismatch")
    latent_bytes = zlib.decompress(latent_zlib)
    if hashlib.sha256(latent_bytes).hexdigest() != latent_hash:
        raise RuntimeError("Evaluation latent hash mismatch")
    shape = tuple(int(value) for value in json.loads(latent_shape))
    latent = torch.from_numpy(np.frombuffer(latent_bytes, dtype=np.float32).reshape(shape).copy())
    return stream, latent


def mask_text(group_ids: tuple[int, ...] | list[int]) -> str:
    present = set(group_ids)
    return "".join("1" if group_id in present else "0" for group_id in range(1, 5))


def masked_batch(full_latent: Any, masks: list[int], device: Any) -> tuple[Any, Any]:
    import torch

    latents: list[Any] = []
    indicators: list[Any] = []
    for mask in masks:
        latent = full_latent[0].clone()
        indicator = torch.zeros((4, latent.shape[-2], latent.shape[-1]), dtype=torch.float32)
        for group_id in range(1, 5):
            if mask & (1 << (group_id - 1)):
                indicator[group_id - 1] = 1.0
            else:
                latent[(group_id - 1) * 48 : group_id * 48] = 0.0
        latents.append(latent)
        indicators.append(indicator)
    return torch.stack(latents).to(device), torch.stack(indicators).to(device)


def quality_batch(reconstructions: Any, target: Any, lpips_model: Any) -> list[dict[str, float]]:
    import torch
    from pytorch_msssim import ms_ssim

    targets = target.expand(reconstructions.shape[0], -1, -1, -1)
    mse = (reconstructions - targets).square().flatten(1).mean(1).clamp_min(1e-12)
    psnr = -10.0 * torch.log10(mse)
    msssim = ms_ssim(reconstructions.float(), targets.float(), data_range=1.0, size_average=False)
    lpips_values = lpips_model(reconstructions.float() * 2.0 - 1.0, targets.float() * 2.0 - 1.0).flatten()
    return [
        {
            "psnr": float(psnr[index].cpu()),
            "ms_ssim": float(msssim[index].cpu()),
            "lpips": float(lpips_values[index].cpu()),
        }
        for index in range(reconstructions.shape[0])
    ]


def decoder_latency(decoder: Any, latent: Any, indicator: Any, output_size: tuple[int, int], device: Any) -> float:
    import torch

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        decoder(latent, indicator, output_size=output_size)
        end.record()
        torch.cuda.synchronize(device)
        return float(start.elapsed_time(end))
    started = time.perf_counter()
    decoder(latent, indicator, output_size=output_size)
    return (time.perf_counter() - started) * 1000.0


def trajectory_auc(cumulative_bytes: list[int], qualities: list[float], full_bytes: int) -> float:
    boundaries = [0.0] + [min(100.0, value * 100.0 / full_bytes) for value in cumulative_bytes]
    states = [0.0] + qualities
    area = 0.0
    for index in range(4):
        area += states[index] * max(0.0, boundaries[index + 1] - boundaries[index])
    return area / 100.0


def main() -> None:
    import lpips
    import torch

    args = parse_args()
    contract = load_contract()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    heavy = load_decoder(args.heavy_checkpoint, "heavy", device)
    light = load_decoder(args.light_checkpoint, "light", device)
    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    connection = sqlite3.connect(f"file:{args.cache.as_posix()}?mode=ro", uri=True)
    metadata = {key: json.loads(value) for key, value in connection.execute("SELECT key, value FROM metadata")}
    if metadata.get("contract_sha256") != contract_sha256():
        raise RuntimeError("Evaluation cache belongs to another contract")
    evaluation_meta = metadata.get("evaluation_cache")
    if not evaluation_meta or evaluation_meta.get("status") != "pass":
        raise RuntimeError("Evaluation cache is incomplete")
    if evaluation_meta.get("latent_storage_dtype") != "float32_exact_entropy_decode":
        raise RuntimeError("Quality evaluation requires exact float32 entropy-decoded latent storage")
    sender_hash = metadata["sender_state_dict_sha256"]
    heavy_hash = composite_hash(sender_hash, heavy)
    light_hash = composite_hash(sender_hash, light)

    architecture = json.loads((ROOT / "runs" / "core_v3_local" / "architecture_freeze.json").read_text(encoding="utf-8"))
    profiles = architecture["profiles"]
    decoder_stats = {
        "heavy": (profiles["server_heavy_decoder"]["params"], profiles["server_heavy_decoder"]["macs"]),
        "light": (profiles["symmetric_light_decoder"]["params"], profiles["symmetric_light_decoder"]["macs"]),
    }
    manifests = {
        "visdrone": ROOT / "data" / "core_v3_visdrone_test.csv",
        "uavid": ROOT / "data" / "core_v3_uavid_test.csv",
    }
    datasets = {
        "visdrone": CoreV3FixedTileDataset(manifests["visdrone"], data_root=args.visdrone_root),
        "uavid": CoreV3FixedTileDataset(manifests["uavid"], data_root=args.uavid_root),
    }
    code_hash = files_sha256(
        [
            ROOT / "models" / "baa_nic.py",
            ROOT / "models" / "core_v3_controls.py",
            ROOT / "models" / "baa_nic_bitstream.py",
            ROOT / "scripts" / "core_v3_data.py",
            Path(__file__).resolve(),
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    asymmetry_path = args.output_dir / "core_v3_decoder_asymmetry.csv"
    ordering_path = args.output_dir / "core_v3_progressive_ordering.csv"
    asymmetry_handle = asymmetry_path.open("w", encoding="utf-8", newline="")
    ordering_handle = ordering_path.open("w", encoding="utf-8", newline="")
    asymmetry_writer = csv.DictWriter(asymmetry_handle, fieldnames=ASYMMETRY_FIELDS)
    ordering_writer = csv.DictWriter(ordering_handle, fieldnames=ORDERING_FIELDS)
    asymmetry_writer.writeheader()
    ordering_writer.writeheader()
    asymmetry_rows = 0
    ordering_rows = 0
    permutations = list(itertools.permutations((1, 2, 3, 4)))
    ordered_masks = [1, 3, 7, 15]

    with torch.inference_mode():
        for dataset_name, dataset in datasets.items():
            manifest_hash = file_sha256(manifests[dataset_name])
            for image_index in range(len(dataset)):
                sample = dataset[image_index]
                target = sample["image"].unsqueeze(0).to(device)
                image_id = sample["metadata"]["tile_id"]
                pixels = int(target.shape[-2] * target.shape[-1])
                for q_index in contract["model"]["q_indices"]:
                    row = connection.execute(
                        """
                        SELECT stream, stream_sha256, latent_zlib, latent_sha256, latent_shape
                        FROM evaluation_entries WHERE dataset=? AND image_index=? AND q_index=?
                        """,
                        (dataset_name, image_index, int(q_index)),
                    ).fetchone()
                    if row is None:
                        raise RuntimeError("Missing evaluation-cache entry")
                    stream, full_latent = decode_cache_row(row)
                    parse_stream(stream)

                    masks = list(range(1, 16))
                    latent_batch, indicator_batch = masked_batch(full_latent, masks, device)
                    heavy_reconstruction = heavy(latent_batch, indicator_batch, output_size=target.shape[-2:])
                    heavy_quality_list = quality_batch(heavy_reconstruction, target, lpips_model)
                    heavy_quality = {mask: quality for mask, quality in zip(masks, heavy_quality_list)}

                    ordered_latents, ordered_indicators = masked_batch(full_latent, ordered_masks, device)
                    light_reconstruction = light(
                        ordered_latents, ordered_indicators, output_size=target.shape[-2:]
                    )
                    light_quality_list = quality_batch(light_reconstruction, target, lpips_model)
                    light_quality = {mask: quality for mask, quality in zip(ordered_masks, light_quality_list)}

                    for prefix_k, mask in enumerate(ordered_masks, start=1):
                        prefix = prefix_stream(stream, prefix_k)
                        for decoder_name, decoder, quality, checkpoint_hash in (
                            ("heavy", heavy, heavy_quality[mask], heavy_hash),
                            ("light", light, light_quality[mask], light_hash),
                        ):
                            latents, indicators = masked_batch(full_latent, [mask], device)
                            latency_ms = decoder_latency(
                                decoder, latents, indicators, tuple(target.shape[-2:]), device
                            )
                            params, macs = decoder_stats[decoder_name]
                            asymmetry_writer.writerow(
                                {
                                    "contract_version": contract["contract_version"],
                                    "contract_sha256": contract_sha256(),
                                    "code_hash": code_hash,
                                    "checkpoint_sha256": checkpoint_hash,
                                    "manifest_sha256": manifest_hash,
                                    "dataset": dataset_name,
                                    "seed": args.seed,
                                    "image_id": image_id,
                                    "q_index": q_index,
                                    "prefix_k": prefix_k,
                                    "decoder": decoder_name,
                                    "bitstream_sha256": hashlib.sha256(prefix).hexdigest(),
                                    "bytes": len(prefix),
                                    "bpp": f"{len(prefix) * 8.0 / pixels:.10f}",
                                    "psnr": f"{quality['psnr']:.8f}",
                                    "ms_ssim": f"{quality['ms_ssim']:.10f}",
                                    "lpips": f"{quality['lpips']:.10f}",
                                    "decoder_params": params,
                                    "decoder_macs": macs,
                                    "latency_ms": f"{latency_ms:.8f}",
                                }
                            )
                            asymmetry_rows += 1

                    for ordering in permutations:
                        cumulative: list[int] = []
                        qualities: list[dict[str, float]] = []
                        masks_for_order: list[int] = []
                        for prefix_k in range(1, 5):
                            group_ids = ordering[:prefix_k]
                            subset = subset_stream(stream, group_ids)
                            mask = sum(1 << (group_id - 1) for group_id in group_ids)
                            cumulative.append(len(subset))
                            masks_for_order.append(mask)
                            qualities.append(heavy_quality[mask])
                        auc = trajectory_auc(cumulative, [value["ms_ssim"] for value in qualities], len(stream))
                        msssim_violations = sum(
                            qualities[index + 1]["ms_ssim"] < qualities[index]["ms_ssim"] - 1e-4
                            for index in range(3)
                        )
                        psnr_violations = sum(
                            qualities[index + 1]["psnr"] < qualities[index]["psnr"] - 0.01
                            for index in range(3)
                        )
                        lpips_violations = sum(
                            qualities[index + 1]["lpips"] > qualities[index]["lpips"] + 1e-4
                            for index in range(3)
                        )
                        violation_flags = (
                            f"msssim:{msssim_violations};psnr:{psnr_violations};lpips:{lpips_violations}"
                        )
                        for budget_percent in (25, 50, 75, 100):
                            budget_bytes = len(stream) * budget_percent / 100.0
                            completed = [index for index, value in enumerate(cumulative) if value <= budget_bytes + 1e-9]
                            if completed:
                                state_index = completed[-1]
                                received_mask = masks_for_order[state_index]
                                used_bytes = cumulative[state_index]
                                quality = qualities[state_index]
                            else:
                                received_mask = 0
                                used_bytes = min(HEADER_SIZE, int(math.floor(budget_bytes)))
                                quality = {"psnr": 0.0, "ms_ssim": 0.0, "lpips": 1.0}
                            ordering_writer.writerow(
                                {
                                    "contract_version": contract["contract_version"],
                                    "contract_sha256": contract_sha256(),
                                    "code_hash": code_hash,
                                    "checkpoint_sha256": heavy_hash,
                                    "manifest_sha256": manifest_hash,
                                    "dataset": dataset_name,
                                    "seed": args.seed,
                                    "image_id": image_id,
                                    "q_index": q_index,
                                    "ordering_id": "-".join(str(value) for value in ordering),
                                    "budget_percent": budget_percent,
                                    "received_group_mask": mask_text(
                                        [group_id for group_id in range(1, 5) if received_mask & (1 << (group_id - 1))]
                                    ),
                                    "cumulative_bytes": used_bytes,
                                    "bpp": f"{used_bytes * 8.0 / pixels:.10f}",
                                    "psnr": f"{quality['psnr']:.8f}",
                                    "ms_ssim": f"{quality['ms_ssim']:.10f}",
                                    "lpips": f"{quality['lpips']:.10f}",
                                    "auc": f"{auc:.10f}",
                                    "violation_flags": violation_flags,
                                }
                            )
                            ordering_rows += 1
                if (image_index + 1) % 32 == 0:
                    asymmetry_handle.flush()
                    ordering_handle.flush()
                    print(f"QUALITY: {dataset_name} {image_index + 1}/{len(dataset)}", flush=True)
    asymmetry_handle.close()
    ordering_handle.close()
    connection.close()
    output = {
        "status": "pass",
        "contract_version": contract["contract_version"],
        "contract_sha256": contract_sha256(),
        "seed": args.seed,
        "sender_state_dict_sha256": sender_hash,
        "heavy_composite_sha256": heavy_hash,
        "light_composite_sha256": light_hash,
        "decoder_asymmetry_rows": asymmetry_rows,
        "progressive_ordering_rows": ordering_rows,
        "all_24_orders": len(permutations) == 24,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: quality seed={args.seed} asymmetry={asymmetry_rows} ordering={ordering_rows}")


if __name__ == "__main__":
    main()
