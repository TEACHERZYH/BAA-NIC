from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import BAANICModel
from models.baa_nic_bitstream import parse_stream, prefix_stream, subset_stream
from core_v3_contract import contract_sha256, file_sha256, files_sha256, load_contract, sender_state_dict_sha256
from core_v3_data import CoreV3FixedTileDataset
from core_v3_training import configure_practical_determinism


ROUNDTRIP_FIELDS = [
    "contract_version",
    "contract_sha256",
    "code_hash",
    "checkpoint_sha256",
    "manifest_sha256",
    "dataset",
    "seed",
    "image_id",
    "q_index",
    "subset_mask",
    "prefix_k",
    "bytes",
    "bpp",
    "decode_ok",
    "crc_ok",
    "byte_prefix_equal",
    "reconstruction_sha256",
]

OPERATING_FIELDS = [
    "contract_version",
    "contract_sha256",
    "code_hash",
    "checkpoint_sha256",
    "manifest_sha256",
    "dataset",
    "seed",
    "q_index",
    "prefix_k",
    "images",
    "median_bytes",
    "median_bpp",
    "stage",
    "evidence_gate_applicable",
]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def reconstruction_sha256(tensor: Any) -> str:
    quantized = tensor.detach().clamp(0.0, 1.0).mul(255.0).round().to(dtype=__import__("torch").uint8)
    return hashlib.sha256(quantized.cpu().contiguous().numpy().tobytes()).hexdigest()


def mask_text(mask_value: int) -> str:
    return "".join("1" if mask_value & (1 << index) else "0" for index in range(4))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--images_per_dataset", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "runs" / "core_v3_bitstream")
    args = parser.parse_args()

    import torch

    contract = load_contract()
    contract_hash = contract_sha256()
    architecture_freeze_path = ROOT / "runs" / "core_v3_local" / "architecture_freeze.json"
    architecture_freeze = json.loads(architecture_freeze_path.read_text(encoding="utf-8"))
    if architecture_freeze.get("status") != "pass":
        raise RuntimeError("Core-v3 architecture freeze has not passed")
    if architecture_freeze.get("contract_sha256") != contract_hash:
        raise RuntimeError("Architecture freeze and experiment contract hashes differ")
    architecture_freeze_hash = file_sha256(architecture_freeze_path)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    configure_practical_determinism(args.seed)

    model = BAANICModel(use_entropy_model=True, entropy_model_type="group_hyperprior").to(device)
    model.eval()
    entropy_updated = model.update_entropy_bottleneck(force=True)
    checkpoint_hash = sender_state_dict_sha256(model)
    model.set_codec_metadata(checkpoint_hash, contract_hash)
    code_hash = files_sha256(
        [
            ROOT / "models" / "baa_nic.py",
            ROOT / "models" / "baa_nic_bitstream.py",
            ROOT / "scripts" / "core_v3_contract.py",
            ROOT / "scripts" / "core_v3_data.py",
            ROOT / "scripts" / "core_v3_training.py",
            Path(__file__),
        ]
    )

    manifests = {
        "VisDrone": ROOT / "data" / "core_v3_visdrone_development.csv",
        "UAVid": ROOT / "data" / "core_v3_uavid_test.csv",
    }
    roundtrip_rows: list[dict[str, Any]] = []
    prefix_values: dict[tuple[str, int, int], list[tuple[int, float]]] = {}
    full_hashes: dict[tuple[str, str], set[str]] = {}

    for dataset_name, manifest in manifests.items():
        manifest_hash = file_sha256(manifest)
        dataset = CoreV3FixedTileDataset(manifest, limit=args.images_per_dataset)
        for sample in dataset:
            image = sample["image"].unsqueeze(0).to(device)
            metadata = sample["metadata"]
            image_key = (dataset_name, metadata["tile_id"])
            full_hashes[image_key] = set()
            for q_index in contract["model"]["q_indices"]:
                compressed = model.compress(image, q_index=q_index)
                repeated = model.compress(image, q_index=q_index)
                full_stream = compressed["bitstream"]
                if full_stream != repeated["bitstream"]:
                    raise RuntimeError(f"Non-deterministic bitstream for {image_key}, q={q_index}")
                header, full_segments = parse_stream(full_stream)
                if [segment.group_id for segment in full_segments] != [1, 2, 3, 4]:
                    raise RuntimeError("Full stream is not in physical group order")
                if header.checkpoint_sha256 != checkpoint_hash or header.contract_sha256 != contract_hash:
                    raise RuntimeError("Bitstream provenance hash mismatch")
                decoded_full_latent = model.decompress_latent(full_stream)["masked_latent"]
                if not torch.equal(compressed["masked_latent"], decoded_full_latent):
                    raise RuntimeError("Encoder-returned quantized latent differs from independent bitstream decode")
                full_hashes[image_key].add(hashlib.sha256(full_stream).hexdigest())

                prefix_lengths: list[int] = []
                for mask_value in range(1, 16):
                    group_ids = [index + 1 for index in range(4) if mask_value & (1 << index)]
                    is_prefix = group_ids == list(range(1, len(group_ids) + 1))
                    prefix_k = len(group_ids) if is_prefix else ""
                    if is_prefix:
                        stream = prefix_stream(full_stream, len(group_ids))
                        byte_prefix_equal: str | bool = stream == full_stream[: len(stream)]
                        prefix_lengths.append(len(stream))
                    else:
                        stream = subset_stream(full_stream, group_ids)
                        byte_prefix_equal = "not_applicable"

                    parsed_header, parsed_segments = parse_stream(stream)
                    if parsed_header.q_index != q_index:
                        raise RuntimeError("Decoded q_index differs from encoded q_index")
                    if [segment.group_id for segment in parsed_segments] != group_ids:
                        raise RuntimeError("Subset stream group IDs changed during parsing")
                    decoded = model.decompress(stream)
                    reconstruction = decoded["reconstruction"]
                    decode_ok = tuple(reconstruction.shape) == tuple(image.shape) and bool(torch.isfinite(reconstruction).all())
                    if not decode_ok:
                        raise RuntimeError("Invalid reconstruction in G0a")
                    bpp = len(stream) * 8.0 / float(image.shape[-2] * image.shape[-1])
                    roundtrip_rows.append(
                        {
                            "contract_version": contract["contract_version"],
                            "contract_sha256": contract_hash,
                            "code_hash": code_hash,
                            "checkpoint_sha256": checkpoint_hash,
                            "manifest_sha256": manifest_hash,
                            "dataset": dataset_name,
                            "seed": args.seed,
                            "image_id": metadata["tile_id"],
                            "q_index": q_index,
                            "subset_mask": mask_text(mask_value),
                            "prefix_k": prefix_k,
                            "bytes": len(stream),
                            "bpp": f"{bpp:.10f}",
                            "decode_ok": "true",
                            "crc_ok": "true",
                            "byte_prefix_equal": str(byte_prefix_equal).lower(),
                            "reconstruction_sha256": reconstruction_sha256(reconstruction),
                        }
                    )
                    if is_prefix:
                        prefix_values.setdefault((dataset_name, q_index, len(group_ids)), []).append((len(stream), bpp))
                if prefix_lengths != sorted(prefix_lengths):
                    raise RuntimeError("Ordered prefix bytes are not monotonic")
            if len(full_hashes[image_key]) != len(contract["model"]["q_indices"]):
                raise RuntimeError(f"q_index did not produce four distinct streams for {image_key}")

    expected_rows = len(manifests) * args.images_per_dataset * 4 * 15
    if len(roundtrip_rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} roundtrip rows, got {len(roundtrip_rows)}")
    keys = {
        (row["dataset"], row["seed"], row["image_id"], row["q_index"], row["subset_mask"])
        for row in roundtrip_rows
    }
    if len(keys) != len(roundtrip_rows):
        raise RuntimeError("Duplicate G0a pairing keys")

    operating_rows: list[dict[str, Any]] = []
    manifest_hashes = {name: file_sha256(path) for name, path in manifests.items()}
    for (dataset_name, q_index, prefix_k), values in sorted(prefix_values.items()):
        operating_rows.append(
            {
                "contract_version": contract["contract_version"],
                "contract_sha256": contract_hash,
                "code_hash": code_hash,
                "checkpoint_sha256": checkpoint_hash,
                "manifest_sha256": manifest_hashes[dataset_name],
                "dataset": dataset_name,
                "seed": args.seed,
                "q_index": q_index,
                "prefix_k": prefix_k,
                "images": len(values),
                "median_bytes": f"{float(np.median([value[0] for value in values])):.3f}",
                "median_bpp": f"{float(np.median([value[1] for value in values])):.10f}",
                "stage": "G0a_structural",
                "evidence_gate_applicable": "false",
            }
        )

    write_csv(ROOT / "data" / "core_v3_bitstream_roundtrip.csv", ROUNDTRIP_FIELDS, roundtrip_rows)
    write_csv(ROOT / "data" / "core_v3_operating_points.csv", OPERATING_FIELDS, operating_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": "pass",
        "gate": "G0a_structural",
        "contract_version": contract["contract_version"],
        "contract_sha256": contract_hash,
        "code_hash": code_hash,
        "checkpoint_sha256": checkpoint_hash,
        "result_schema_sha256": file_sha256(ROOT / "configs" / "core_v3_result_schemas.json"),
        "architecture_freeze_sha256": architecture_freeze_hash,
        "device": str(device),
        "entropy_updated": entropy_updated,
        "datasets": {name: args.images_per_dataset for name in manifests},
        "roundtrip_rows": len(roundtrip_rows),
        "operating_point_rows": len(operating_rows),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: G0a bitstream roundtrip rows={len(roundtrip_rows)} operating_rows={len(operating_rows)}")


if __name__ == "__main__":
    main()
