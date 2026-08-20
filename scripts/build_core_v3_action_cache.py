"""Attach action-specific quality and latency records to a core-v3 evaluation cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import BAANICModel, ServerDecoder
from models.baa_nic_bitstream import prefix_stream
from core_v3_contract import (
    contract_sha256,
    file_sha256,
    files_sha256,
    load_contract,
    sender_state_dict_sha256,
    state_dict_sha256,
)
from core_v3_data import CoreV3FixedTileDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--heavy-decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--visdrone-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def synchronized_wall_call(fn: Any, device: Any) -> tuple[Any, float]:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return output, (time.perf_counter() - started) * 1000.0


def load_models(args: argparse.Namespace, device: Any) -> tuple[Any, Any, str, str]:
    import torch

    sender_payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if sender_payload.get("kind") != "core_v3_joint":
        raise RuntimeError("Expected a core_v3_joint checkpoint")
    if sender_payload["metadata"]["contract_sha256"] != contract_sha256():
        raise RuntimeError("Sender checkpoint belongs to another contract")
    model = BAANICModel(use_entropy_model=True, entropy_model_type="group_hyperprior").to(device)
    model.load_state_dict(sender_payload["model"])
    model.eval()
    model.update_entropy_bottleneck(force=True)
    sender_hash = sender_state_dict_sha256(model)
    model.set_codec_metadata(sender_hash, contract_sha256())

    decoder_payload = torch.load(args.heavy_decoder_checkpoint, map_location=device, weights_only=False)
    if decoder_payload.get("kind") != "core_v3_decoder_heavy":
        raise RuntimeError("Expected a core_v3_decoder_heavy checkpoint")
    decoder = ServerDecoder().to(device)
    decoder.load_state_dict(decoder_payload["decoder"])
    decoder.eval()
    decoder_hash = state_dict_sha256(decoder)
    return model, decoder, sender_hash, decoder_hash


def main() -> None:
    import torch
    from pytorch_msssim import ms_ssim

    args = parse_args()
    contract = load_contract()
    if args.seed not in contract["training"]["model_seeds"]:
        raise ValueError("Seed is outside the frozen contract")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model, decoder, sender_hash, decoder_hash = load_models(args, device)
    connection = sqlite3.connect(args.cache)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    metadata = {key: json.loads(value) for key, value in connection.execute("SELECT key, value FROM metadata")}
    if metadata.get("contract_sha256") != contract_sha256():
        raise RuntimeError("Evaluation cache belongs to another contract")
    if metadata.get("sender_state_dict_sha256") != sender_hash:
        raise RuntimeError("Evaluation cache and sender checkpoint differ")
    if metadata.get("evaluation_cache", {}).get("status") != "pass":
        raise RuntimeError("Evaluation cache is incomplete")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS action_entries (
            dataset TEXT NOT NULL,
            image_index INTEGER NOT NULL,
            image_id TEXT NOT NULL,
            sequence_id TEXT NOT NULL,
            q_index INTEGER NOT NULL,
            prefix_k INTEGER NOT NULL,
            stream_sha256 TEXT NOT NULL,
            payload_bytes INTEGER NOT NULL,
            network_bytes INTEGER NOT NULL,
            bpp REAL NOT NULL,
            psnr REAL NOT NULL,
            ms_ssim REAL NOT NULL,
            encode_ms REAL NOT NULL,
            entropy_decode_ms REAL NOT NULL,
            decoder_ms REAL NOT NULL,
            decode_ms REAL NOT NULL,
            PRIMARY KEY (dataset, image_index, q_index, prefix_k)
        )
        """
    )
    manifests = {
        "development": ROOT / "data" / "core_v3_visdrone_development.csv",
        "visdrone": ROOT / "data" / "core_v3_visdrone_test.csv",
    }
    datasets = {
        name: CoreV3FixedTileDataset(manifest, data_root=args.visdrone_root)
        for name, manifest in manifests.items()
    }
    existing = {
        (str(dataset), int(image_index), int(q_index), int(prefix_k))
        for dataset, image_index, q_index, prefix_k in connection.execute(
            "SELECT dataset, image_index, q_index, prefix_k FROM action_entries"
        )
    }
    code_hash = files_sha256(
        [
            ROOT / "models" / "baa_nic.py",
            ROOT / "models" / "baa_nic_bitstream.py",
            ROOT / "models" / "core_v3_controls.py",
            ROOT / "scripts" / "core_v3_data.py",
            Path(__file__).resolve(),
        ]
    )
    inserted = 0
    with torch.inference_mode():
        for dataset_name, dataset in datasets.items():
            for image_index in range(len(dataset)):
                sample = dataset[image_index]
                target = sample["image"].unsqueeze(0).to(device)
                pixels = int(target.shape[-2] * target.shape[-1])
                for q_index in contract["model"]["q_indices"]:
                    row = connection.execute(
                        """
                        SELECT stream, stream_sha256, encode_ms
                        FROM evaluation_entries WHERE dataset=? AND image_index=? AND q_index=?
                        """,
                        (dataset_name, image_index, int(q_index)),
                    ).fetchone()
                    if row is None:
                        raise RuntimeError("Missing evaluation entry for action cache")
                    full_stream, full_hash, encode_ms = bytes(row[0]), str(row[1]), float(row[2])
                    if hashlib.sha256(full_stream).hexdigest() != full_hash:
                        raise RuntimeError("Evaluation stream hash mismatch")
                    for prefix_k in range(1, 5):
                        key = (dataset_name, image_index, int(q_index), prefix_k)
                        if key in existing:
                            continue
                        stream = prefix_stream(full_stream, prefix_k)
                        decoded, entropy_decode_ms = synchronized_wall_call(
                            lambda stream=stream: model.decompress_latent(stream), device
                        )
                        reconstruction, decoder_ms = synchronized_wall_call(
                            lambda decoded=decoded: decoder(
                                decoded["masked_latent"],
                                decoded["group_indicator"],
                                output_size=tuple(target.shape[-2:]),
                            ).clamp(0.0, 1.0),
                            device,
                        )
                        mse = float((reconstruction - target).square().mean().clamp_min(1e-12).cpu())
                        psnr = -10.0 * math.log10(mse)
                        quality = float(
                            ms_ssim(reconstruction.float(), target.float(), data_range=1.0).cpu()
                        )
                        payload_bytes = len(stream)
                        packets = (payload_bytes + 1459) // 1460
                        connection.execute(
                            """
                            INSERT INTO action_entries(
                                dataset, image_index, image_id, sequence_id, q_index, prefix_k,
                                stream_sha256, payload_bytes, network_bytes, bpp, psnr, ms_ssim,
                                encode_ms, entropy_decode_ms, decoder_ms, decode_ms
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                dataset_name,
                                image_index,
                                sample["metadata"]["tile_id"],
                                sample["metadata"].get("sequence_id", "unknown"),
                                int(q_index),
                                prefix_k,
                                hashlib.sha256(stream).hexdigest(),
                                payload_bytes,
                                payload_bytes + packets * 40,
                                payload_bytes * 8.0 / pixels,
                                psnr,
                                quality,
                                encode_ms,
                                entropy_decode_ms,
                                decoder_ms,
                                entropy_decode_ms + decoder_ms,
                            ),
                        )
                        inserted += 1
                if (image_index + 1) % 16 == 0:
                    connection.commit()
                    print(f"ACTION_CACHE: {dataset_name} {image_index + 1}/{len(dataset)}", flush=True)
    expected = sum(len(dataset) for dataset in datasets.values()) * 16
    final_count = int(connection.execute("SELECT COUNT(*) FROM action_entries").fetchone()[0])
    if final_count != expected:
        raise RuntimeError(f"Action cache incomplete: {final_count} != {expected}")
    summary = {
        "status": "pass",
        "contract_version": contract["contract_version"],
        "contract_sha256": contract_sha256(),
        "seed": args.seed,
        "sender_state_dict_sha256": sender_hash,
        "heavy_decoder_state_dict_sha256": decoder_hash,
        "code_hash": code_hash,
        "manifest_sha256": {name: file_sha256(path) for name, path in manifests.items()},
        "rows": final_count,
        "new_rows": inserted,
        "timer": "synchronized_wall_clock_including_entropy_host_work",
    }
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        ("action_cache", json.dumps(summary, sort_keys=True)),
    )
    connection.commit()
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.close()
    sidecar = args.cache.with_suffix(args.cache.suffix + ".action_metadata.json")
    summary["cache_sha256"] = file_sha256(args.cache)
    sidecar.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: action cache seed={args.seed} rows={final_count} new={inserted}")


if __name__ == "__main__":
    main()
