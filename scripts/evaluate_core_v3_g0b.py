"""Run the frozen 32+32-image practical-bitstream evidence gate for one seed."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import BAANICModel
from models.baa_nic_bitstream import parse_stream, prefix_stream, subset_stream
from core_v3_contract import contract_sha256, file_sha256, files_sha256, load_contract, sender_state_dict_sha256
from core_v3_data import CoreV3FixedTileDataset
from core_v3_training import configure_practical_determinism


ROUNDTRIP_FIELDS = [
    "contract_version", "contract_sha256", "code_hash", "checkpoint_sha256", "manifest_sha256",
    "dataset", "seed", "image_id", "q_index", "subset_mask", "prefix_k", "bytes", "bpp",
    "decode_ok", "crc_ok", "byte_prefix_equal", "reconstruction_sha256",
]
OPERATING_FIELDS = [
    "contract_version", "contract_sha256", "code_hash", "checkpoint_sha256", "manifest_sha256",
    "dataset", "seed", "q_index", "prefix_k", "images", "median_bytes", "median_bpp", "stage",
    "evidence_gate_applicable",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visdrone-root", type=Path, required=True)
    parser.add_argument("--uavid-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--images-per-dataset", type=int, default=32)
    return parser.parse_args()


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reconstruction_sha256(tensor: Any) -> str:
    import torch

    quantized = tensor.detach().clamp(0.0, 1.0).mul(255.0).round().to(dtype=torch.uint8)
    return hashlib.sha256(quantized.cpu().contiguous().numpy().tobytes()).hexdigest()


def mask_text(mask_value: int) -> str:
    return "".join("1" if mask_value & (1 << index) else "0" for index in range(4))


def main() -> None:
    import torch

    args = parse_args()
    configure_practical_determinism(args.seed)
    contract = load_contract()
    if args.seed not in [int(value) for value in contract["training"]["model_seeds"]]:
        raise ValueError("Unregistered model seed")
    if args.images_per_dataset != 32:
        raise ValueError("G0b requires exactly 32 images per dataset")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("kind") != "core_v3_joint":
        raise ValueError("G0b requires a core-v3 joint checkpoint")
    if checkpoint["metadata"]["contract_sha256"] != contract_sha256():
        raise RuntimeError("Checkpoint belongs to another frozen contract")
    model = BAANICModel(use_entropy_model=True, entropy_model_type="group_hyperprior").to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.update_entropy_bottleneck(force=True)
    checkpoint_hash = sender_state_dict_sha256(model)
    model.set_codec_metadata(checkpoint_hash, contract_sha256())

    manifests = {
        "visdrone": ROOT / "data" / "core_v3_visdrone_test.csv",
        "uavid": ROOT / "data" / "core_v3_uavid_test.csv",
    }
    roots = {"visdrone": args.visdrone_root, "uavid": args.uavid_root}
    code_hash = files_sha256(
        [
            ROOT / "models" / "baa_nic.py",
            ROOT / "models" / "baa_nic_bitstream.py",
            ROOT / "scripts" / "core_v3_data.py",
            ROOT / "scripts" / "core_v3_training.py",
            Path(__file__).resolve(),
        ]
    )
    roundtrip_rows: list[dict[str, Any]] = []
    prefix_values: dict[tuple[str, int, int], list[tuple[int, float]]] = {}
    q_stream_hashes: dict[tuple[str, str], set[str]] = {}

    for dataset_name, manifest in manifests.items():
        manifest_hash = file_sha256(manifest)
        dataset = CoreV3FixedTileDataset(
            manifest, limit=args.images_per_dataset, data_root=roots[dataset_name]
        )
        for sample in dataset:
            image = sample["image"].unsqueeze(0).to(device)
            image_id = sample["metadata"]["tile_id"]
            image_key = (dataset_name, image_id)
            q_stream_hashes[image_key] = set()
            pixels = int(image.shape[-2] * image.shape[-1])
            for q_index in contract["model"]["q_indices"]:
                encoded = model.compress(image, q_index=int(q_index))
                repeated = model.compress(image, q_index=int(q_index))
                full_stream = encoded["bitstream"]
                if full_stream != repeated["bitstream"]:
                    raise RuntimeError(f"Non-deterministic stream for {image_key}, q={q_index}")
                header, segments = parse_stream(full_stream)
                if [segment.group_id for segment in segments] != [1, 2, 3, 4]:
                    raise RuntimeError("Full stream group order differs from the frozen contract")
                if header.checkpoint_sha256 != checkpoint_hash or header.contract_sha256 != contract_sha256():
                    raise RuntimeError("Bitstream provenance mismatch")
                decoded_full = model.decompress_latent(full_stream)["masked_latent"]
                if not torch.equal(encoded["masked_latent"], decoded_full):
                    raise RuntimeError("Encoded latent differs from independent entropy decode")
                q_stream_hashes[image_key].add(hashlib.sha256(full_stream).hexdigest())

                previous_prefix_bytes = 0
                for mask_value in range(1, 16):
                    group_ids = [index + 1 for index in range(4) if mask_value & (1 << index)]
                    is_prefix = group_ids == list(range(1, len(group_ids) + 1))
                    if is_prefix:
                        stream = prefix_stream(full_stream, len(group_ids))
                        prefix_k = len(group_ids)
                        byte_prefix_equal = stream == full_stream[: len(stream)]
                        if len(stream) < previous_prefix_bytes:
                            raise RuntimeError("Ordered prefix bytes are not monotonic")
                        previous_prefix_bytes = len(stream)
                    else:
                        stream = subset_stream(full_stream, group_ids)
                        prefix_k = 0
                        byte_prefix_equal = False
                    decoded = model.decompress(stream)
                    if tuple(decoded["present_group_ids"]) != tuple(group_ids):
                        raise RuntimeError("Decoded group IDs differ from requested subset")
                    bpp = len(stream) * 8.0 / pixels
                    roundtrip_rows.append(
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
                            "subset_mask": mask_text(mask_value),
                            "prefix_k": prefix_k,
                            "bytes": len(stream),
                            "bpp": f"{bpp:.10f}",
                            "decode_ok": "true",
                            "crc_ok": "true",
                            "byte_prefix_equal": str(byte_prefix_equal).lower(),
                            "reconstruction_sha256": reconstruction_sha256(decoded["reconstruction"]),
                        }
                    )
                    if is_prefix:
                        prefix_values.setdefault((dataset_name, int(q_index), prefix_k), []).append(
                            (len(stream), bpp)
                        )
            if len(q_stream_hashes[image_key]) != 4:
                raise RuntimeError(f"q_index did not produce four distinct streams for {image_key}")

    keys = [
        (row["dataset"], row["seed"], row["image_id"], row["q_index"], row["subset_mask"])
        for row in roundtrip_rows
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate G0b pairing keys")
    operating_rows: list[dict[str, Any]] = []
    medians: dict[tuple[str, int, int], float] = {}
    for (dataset_name, q_index, prefix_k), values in sorted(prefix_values.items()):
        median_bytes = statistics.median(value[0] for value in values)
        median_bpp = statistics.median(value[1] for value in values)
        medians[(dataset_name, q_index, prefix_k)] = median_bpp
        operating_rows.append(
            {
                "contract_version": contract["contract_version"],
                "contract_sha256": contract_sha256(),
                "code_hash": code_hash,
                "checkpoint_sha256": checkpoint_hash,
                "manifest_sha256": file_sha256(manifests[dataset_name]),
                "dataset": dataset_name,
                "seed": args.seed,
                "q_index": q_index,
                "prefix_k": prefix_k,
                "images": len(values),
                "median_bytes": f"{median_bytes:.6f}",
                "median_bpp": f"{median_bpp:.10f}",
                "stage": "G0b_evidence",
                "evidence_gate_applicable": "true",
            }
        )
    q_separation: dict[str, dict[str, list[float]]] = {}
    for dataset_name in manifests:
        q_separation[dataset_name] = {}
        for prefix_k in range(1, 5):
            values = [medians[(dataset_name, q_index, prefix_k)] for q_index in range(4)]
            relative_drops = [
                (values[index] - values[index + 1]) / max(values[index], 1e-12) for index in range(3)
            ]
            if any(values[index + 1] > values[index] for index in range(3)):
                raise RuntimeError(f"Median bpp is not monotonic for {dataset_name}, prefix={prefix_k}")
            if sum(drop > 0.05 for drop in relative_drops) < 3:
                raise RuntimeError(f"Fewer than three >5% q operating-point separations for {dataset_name}, prefix={prefix_k}")
            q_separation[dataset_name][str(prefix_k)] = relative_drops

    args.output_dir.mkdir(parents=True, exist_ok=True)
    roundtrip_path = args.output_dir / "core_v3_bitstream_roundtrip.csv"
    operating_path = args.output_dir / "core_v3_operating_points.csv"
    write_csv(roundtrip_path, ROUNDTRIP_FIELDS, roundtrip_rows)
    write_csv(operating_path, OPERATING_FIELDS, operating_rows)
    metadata = {
        "status": "pass",
        "gate": "G0b_evidence",
        "contract_version": contract["contract_version"],
        "contract_sha256": contract_sha256(),
        "code_hash": code_hash,
        "checkpoint_sha256": checkpoint_hash,
        "seed": args.seed,
        "device": str(device),
        "images_per_dataset": args.images_per_dataset,
        "roundtrip_rows": len(roundtrip_rows),
        "operating_point_rows": len(operating_rows),
        "q_relative_bpp_drops": q_separation,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: G0b seed={args.seed} roundtrip_rows={len(roundtrip_rows)} operating_rows={len(operating_rows)}")


if __name__ == "__main__":
    main()
