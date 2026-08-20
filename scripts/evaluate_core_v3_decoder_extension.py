"""Evaluate enhanced and frozen-heavy decoders on byte-identical core-v3 streams."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import ServerDecoder, build_enhanced_server_decoder
from models.baa_nic_bitstream import parse_stream, prefix_stream
from core_v3_contract import (
    contract_sha256,
    file_sha256,
    files_sha256,
    load_contract,
    seed_expansion_sha256,
    state_dict_sha256,
)
from core_v3_data import CoreV3FixedTileDataset
from core_v3_decoder_extension import (
    extension_contract_sha256,
    load_extension_contract,
    registered_extension_seeds,
)
from evaluate_core_v3_quality import decode_cache_row, masked_batch, quality_batch


FIELDS = [
    "extension_version",
    "extension_contract_sha256",
    "parent_contract_sha256",
    "code_hash",
    "checkpoint_sha256",
    "manifest_sha256",
    "dataset",
    "seed",
    "image_id",
    "q_index",
    "prefix_k",
    "decoder",
    "bitstream_sha256",
    "bytes",
    "bpp",
    "psnr",
    "ms_ssim",
    "lpips",
    "decoder_params",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--enhanced-checkpoint", type=Path, required=True)
    parser.add_argument("--heavy-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visdrone-root", type=Path, required=True)
    parser.add_argument("--uavid-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prefix-batch-size", type=int, default=2)
    parser.add_argument("--datasets", default="visdrone,uavid")
    return parser.parse_args()


def composite_hash(sender_hash: str, decoder: Any) -> str:
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(sender_hash))
    digest.update(bytes.fromhex(state_dict_sha256(decoder)))
    return digest.hexdigest()


def load_decoders(
    args: argparse.Namespace, device: Any, extension: dict[str, Any]
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    import torch

    enhanced_checkpoint = torch.load(args.enhanced_checkpoint, map_location=device, weights_only=False)
    if enhanced_checkpoint.get("kind") != "core_v3_decoder_enhanced_r1":
        raise RuntimeError("Invalid enhanced decoder checkpoint")
    if enhanced_checkpoint.get("extension_contract_sha256") != extension_contract_sha256():
        raise RuntimeError("Enhanced decoder checkpoint belongs to another extension contract")
    enhanced = build_enhanced_server_decoder(extension).to(device)
    enhanced.load_state_dict(enhanced_checkpoint["decoder"])
    enhanced.eval()

    heavy_checkpoint = torch.load(args.heavy_checkpoint, map_location=device, weights_only=False)
    if heavy_checkpoint.get("kind") != "core_v3_decoder_heavy":
        raise RuntimeError("Invalid frozen heavy decoder checkpoint")
    heavy = ServerDecoder().to(device)
    heavy.load_state_dict(heavy_checkpoint["decoder"])
    heavy.eval()
    return enhanced, heavy, enhanced_checkpoint["metadata"], heavy_checkpoint["metadata"]


def main() -> None:
    import lpips
    import torch

    args = parse_args()
    if args.prefix_batch_size not in (1, 2, 4):
        raise ValueError("prefix batch size must be 1, 2, or 4")
    parent = load_contract()
    extension = load_extension_contract()
    if args.seed not in registered_extension_seeds(extension):
        raise ValueError("Unregistered model seed")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    enhanced, heavy, enhanced_metadata, heavy_metadata = load_decoders(args, device, extension)
    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    connection = sqlite3.connect(f"file:{args.cache.as_posix()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    metadata = {key: json.loads(value) for key, value in connection.execute("SELECT key, value FROM metadata")}
    if metadata.get("contract_sha256") != contract_sha256():
        raise RuntimeError("Evaluation cache belongs to another parent contract")
    evaluation_meta = metadata.get("evaluation_cache")
    if not evaluation_meta or evaluation_meta.get("status") != "pass":
        raise RuntimeError("Evaluation cache is incomplete")
    if evaluation_meta.get("latent_storage_dtype") != "float32_exact_entropy_decode":
        raise RuntimeError("Evaluation requires exact float32 entropy-decoded latents")
    if int(metadata.get("seed")) != args.seed:
        raise RuntimeError("Evaluation cache belongs to another seed")
    sender_hash = metadata["sender_state_dict_sha256"]
    for name, checkpoint_metadata in (
        ("enhanced", enhanced_metadata),
        ("heavy", heavy_metadata),
    ):
        if checkpoint_metadata.get("sender_state_dict_sha256") != sender_hash:
            raise RuntimeError(f"{name} decoder was not trained from this sender cache")
        if int(checkpoint_metadata.get("seed")) != args.seed:
            raise RuntimeError(f"{name} decoder checkpoint belongs to another seed")
    enhanced_hash = composite_hash(sender_hash, enhanced)
    heavy_hash = composite_hash(sender_hash, heavy)
    decoder_params = {
        "enhanced_r1": sum(parameter.numel() for parameter in enhanced.parameters()),
        "heavy": sum(parameter.numel() for parameter in heavy.parameters()),
    }

    all_manifests = {
        "visdrone": ROOT / "data" / "core_v3_visdrone_test.csv",
        "uavid": ROOT / "data" / "core_v3_uavid_test.csv",
    }
    requested = [value.strip() for value in args.datasets.split(",") if value.strip()]
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("datasets must contain one or more unique names")
    unknown = sorted(set(requested) - set(all_manifests))
    if unknown:
        raise ValueError(f"Unknown evaluation datasets: {unknown}")
    manifests = {name: all_manifests[name] for name in requested}
    roots = {"visdrone": args.visdrone_root, "uavid": args.uavid_root}
    datasets = {
        name: CoreV3FixedTileDataset(manifest, data_root=roots[name])
        for name, manifest in manifests.items()
    }
    code_hash = files_sha256(
        [
            ROOT / "models" / "core_v3_enhanced_decoder.py",
            ROOT / "models" / "baa_nic.py",
            ROOT / "models" / "baa_nic_bitstream.py",
            ROOT / "scripts" / "core_v3_data.py",
            Path(__file__).resolve(),
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "core_v3_decoder_extension.csv"
    rows = 0
    ordered_masks = [1, 3, 7, 15]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        with torch.inference_mode():
            for dataset_name, dataset in datasets.items():
                manifest_hash = file_sha256(manifests[dataset_name])
                for image_index in range(len(dataset)):
                    sample = dataset[image_index]
                    target = sample["image"].unsqueeze(0).to(device)
                    image_id = sample["metadata"]["tile_id"]
                    pixels = int(target.shape[-2] * target.shape[-1])
                    for q_index in extension["q_indices"]:
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
                        qualities: dict[str, dict[int, dict[str, float]]] = {
                            "enhanced_r1": {},
                            "heavy": {},
                        }
                        for start in range(0, 4, args.prefix_batch_size):
                            masks = ordered_masks[start : start + args.prefix_batch_size]
                            latents, indicators = masked_batch(full_latent, masks, device)
                            enhanced_output = enhanced(
                                latents,
                                indicators,
                                q_index=int(q_index),
                                output_size=tuple(target.shape[-2:]),
                            )
                            heavy_output = heavy(latents, indicators, output_size=tuple(target.shape[-2:]))
                            for mask, quality in zip(
                                masks, quality_batch(enhanced_output, target, lpips_model)
                            ):
                                qualities["enhanced_r1"][mask] = quality
                            for mask, quality in zip(masks, quality_batch(heavy_output, target, lpips_model)):
                                qualities["heavy"][mask] = quality

                        for prefix_k, mask in enumerate(ordered_masks, start=1):
                            prefix = prefix_stream(stream, prefix_k)
                            prefix_hash = hashlib.sha256(prefix).hexdigest()
                            for decoder_name, checkpoint_hash in (
                                ("enhanced_r1", enhanced_hash),
                                ("heavy", heavy_hash),
                            ):
                                quality = qualities[decoder_name][mask]
                                writer.writerow(
                                    {
                                        "extension_version": extension["extension_version"],
                                        "extension_contract_sha256": extension_contract_sha256(),
                                        "parent_contract_sha256": contract_sha256(),
                                        "code_hash": code_hash,
                                        "checkpoint_sha256": checkpoint_hash,
                                        "manifest_sha256": manifest_hash,
                                        "dataset": dataset_name,
                                        "seed": args.seed,
                                        "image_id": image_id,
                                        "q_index": q_index,
                                        "prefix_k": prefix_k,
                                        "decoder": decoder_name,
                                        "bitstream_sha256": prefix_hash,
                                        "bytes": len(prefix),
                                        "bpp": f"{len(prefix) * 8.0 / pixels:.10f}",
                                        "psnr": f"{quality['psnr']:.8f}",
                                        "ms_ssim": f"{quality['ms_ssim']:.10f}",
                                        "lpips": f"{quality['lpips']:.10f}",
                                        "decoder_params": decoder_params[decoder_name],
                                    }
                                )
                                rows += 1
                    if (image_index + 1) % 32 == 0:
                        handle.flush()
                        print(f"EXTENSION_QUALITY: {dataset_name} {image_index + 1}/{len(dataset)}", flush=True)
    connection.close()
    expected_rows = sum(len(dataset) for dataset in datasets.values()) * 4 * 4 * 2
    if rows != expected_rows:
        raise RuntimeError(f"Unexpected decoder extension row count: {rows} != {expected_rows}")
    metadata_output = {
        "status": "pass",
        "extension_version": extension["extension_version"],
        "extension_contract_sha256": extension_contract_sha256(),
        "parent_contract_version": parent["contract_version"],
        "parent_contract_sha256": contract_sha256(),
        "seed": args.seed,
        "sender_state_dict_sha256": sender_hash,
        "enhanced_composite_sha256": enhanced_hash,
        "heavy_composite_sha256": heavy_hash,
        "rows": rows,
        "byte_identical_by_construction": True,
    }
    expansion_hash = seed_expansion_sha256()
    if expansion_hash is not None:
        metadata_output["seed_expansion_sha256"] = expansion_hash
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata_output, indent=2) + "\n", encoding="utf-8"
    )
    print(f"PASS: decoder extension quality seed={args.seed} rows={rows}")


if __name__ == "__main__":
    main()
