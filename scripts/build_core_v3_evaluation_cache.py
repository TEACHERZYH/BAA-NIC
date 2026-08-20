"""Extend a frozen sender cache with development and test practical bitstreams."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
import zlib


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import BAANICModel
from core_v3_contract import contract_sha256, file_sha256, load_contract, sender_state_dict_sha256
from core_v3_data import CoreV3FixedTileDataset
from core_v3_training import configure_practical_determinism


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--visdrone-root", type=Path, required=True)
    parser.add_argument("--uavid-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--commit-interval", type=int, default=100)
    parser.add_argument("--datasets", default="development,visdrone,uavid")
    return parser.parse_args()


def main() -> None:
    import numpy as np
    import torch

    args = parse_args()
    configure_practical_determinism(args.seed)
    contract = load_contract()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("kind") != "core_v3_joint" or checkpoint["metadata"]["contract_sha256"] != contract_sha256():
        raise RuntimeError("Invalid core-v3 sender checkpoint")
    model = BAANICModel(use_entropy_model=True, entropy_model_type="group_hyperprior").to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.update_entropy_bottleneck(force=True)
    sender_hash = sender_state_dict_sha256(model)
    model.set_codec_metadata(sender_hash, contract_sha256())

    all_manifests = {
        "development": ROOT / "data" / "core_v3_visdrone_development.csv",
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
    all_roots = {
        "development": args.visdrone_root,
        "visdrone": args.visdrone_root,
        "uavid": args.uavid_root,
    }
    roots = {name: all_roots[name] for name in requested}
    datasets = {
        name: CoreV3FixedTileDataset(manifest, data_root=roots[name]) for name, manifest in manifests.items()
    }

    connection = sqlite3.connect(args.cache)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS evaluation_entries (
            dataset TEXT NOT NULL,
            image_index INTEGER NOT NULL,
            image_id TEXT NOT NULL,
            sequence_id TEXT NOT NULL,
            q_index INTEGER NOT NULL,
            stream BLOB NOT NULL,
            stream_sha256 TEXT NOT NULL,
            stream_bytes INTEGER NOT NULL,
            latent_zlib BLOB NOT NULL,
            latent_sha256 TEXT NOT NULL,
            latent_shape TEXT NOT NULL,
            encode_ms REAL NOT NULL,
            entropy_decode_ms REAL NOT NULL,
            PRIMARY KEY (dataset, image_index, q_index)
        )
        """
    )
    metadata = {key: json.loads(value) for key, value in connection.execute("SELECT key, value FROM metadata")}
    if metadata.get("contract_sha256") != contract_sha256() or metadata.get("sender_state_dict_sha256") != sender_hash:
        raise RuntimeError("Evaluation cache and sender checkpoint provenance differ")
    if metadata.get("latent_storage_dtype") != "float32_exact_entropy_decode":
        raise RuntimeError("Evaluation cache requires exact float32 entropy-decoded latent storage")
    existing = {
        (str(dataset), int(image_index), int(q_index))
        for dataset, image_index, q_index in connection.execute(
            "SELECT dataset, image_index, q_index FROM evaluation_entries"
        )
    }
    required = {
        (dataset_name, image_index, int(q_index))
        for dataset_name, dataset in datasets.items()
        for image_index in range(len(dataset))
        for q_index in contract["model"]["q_indices"]
    }
    pending = sorted(required - existing)
    started = time.perf_counter()
    for completed, (dataset_name, image_index, q_index) in enumerate(pending, start=1):
        sample = datasets[dataset_name][image_index]
        image = sample["image"].unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        encode_started = time.perf_counter()
        encoded = model.compress(image, q_index=q_index)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        encode_ms = (time.perf_counter() - encode_started) * 1000.0
        stream = encoded["bitstream"]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        decode_started = time.perf_counter()
        decoded = model.decompress_latent(stream)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        entropy_decode_ms = (time.perf_counter() - decode_started) * 1000.0
        if not torch.equal(encoded["masked_latent"], decoded["masked_latent"]):
            raise RuntimeError("Cached latent differs from independent practical entropy decode")
        latent = encoded["masked_latent"].detach().cpu().to(torch.float32).contiguous().numpy()
        latent_bytes = latent.tobytes()
        connection.execute(
            """
            INSERT OR REPLACE INTO evaluation_entries(
                dataset, image_index, image_id, sequence_id, q_index, stream, stream_sha256,
                stream_bytes, latent_zlib, latent_sha256, latent_shape, encode_ms, entropy_decode_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_name,
                image_index,
                sample["metadata"]["tile_id"],
                sample["metadata"].get("sequence_id", "unknown"),
                q_index,
                sqlite3.Binary(stream),
                hashlib.sha256(stream).hexdigest(),
                len(stream),
                sqlite3.Binary(zlib.compress(latent_bytes, level=1)),
                hashlib.sha256(latent_bytes).hexdigest(),
                json.dumps(list(latent.shape)),
                encode_ms,
                entropy_decode_ms,
            ),
        )
        if completed % args.commit_interval == 0:
            connection.commit()
            elapsed = time.perf_counter() - started
            print(f"EVAL_CACHE: {completed}/{len(pending)} ({completed / max(elapsed, 1e-9):.3f} pair/s)", flush=True)
    connection.commit()
    final = {
        (str(dataset), int(image_index), int(q_index))
        for dataset, image_index, q_index in connection.execute(
            "SELECT dataset, image_index, q_index FROM evaluation_entries"
        )
    }
    if required - final:
        raise RuntimeError(f"Evaluation cache is missing {len(required - final)} entries")
    summary = {
        "status": "pass",
        "contract_version": contract["contract_version"],
        "contract_sha256": contract_sha256(),
        "seed": args.seed,
        "sender_state_dict_sha256": sender_hash,
        "manifest_sha256": {name: file_sha256(path) for name, path in manifests.items()},
        "required_entries": len(required),
        "entries": len(final),
        "new_entries": len(pending),
        "latent_storage_dtype": "float32_exact_entropy_decode",
        "practical_determinism": "strict",
        "wall_time_seconds": time.perf_counter() - started,
    }
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        ("evaluation_cache", json.dumps(summary, sort_keys=True)),
    )
    connection.commit()
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.close()
    sidecar = args.cache.with_suffix(args.cache.suffix + ".evaluation.json")
    sidecar.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: evaluation cache seed={args.seed} entries={len(final)} new={len(pending)}")


if __name__ == "__main__":
    main()
