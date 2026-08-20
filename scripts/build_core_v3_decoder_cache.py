"""Build the byte-identical sender cache consumed by both decoder-only branches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time
import zlib

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import BAANICModel
from core_v3_contract import (
    contract_sha256,
    file_sha256,
    load_contract,
    registered_model_seeds,
    sender_state_dict_sha256,
)
from core_v3_data import CoreV3CropBankDataset, CoreV3FixedTileDataset
from core_v3_training import (
    action_for_step,
    configure_practical_determinism,
    decoder_source_indices,
    provenance_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=("smoke", "benchmark", "train"), default="train")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--commit-interval", type=int, default=25)
    return parser.parse_args()


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS entries (
            source_index INTEGER NOT NULL,
            q_index INTEGER NOT NULL,
            source_id TEXT NOT NULL,
            crop_seed TEXT NOT NULL,
            stream BLOB NOT NULL,
            stream_sha256 TEXT NOT NULL,
            stream_bytes INTEGER NOT NULL,
            latent_zlib BLOB NOT NULL,
            latent_sha256 TEXT NOT NULL,
            latent_shape TEXT NOT NULL,
            PRIMARY KEY (source_index, q_index)
        );
        CREATE TABLE IF NOT EXISTS validation_entries (
            image_index INTEGER NOT NULL,
            q_index INTEGER NOT NULL,
            image_id TEXT NOT NULL,
            stream BLOB NOT NULL,
            stream_sha256 TEXT NOT NULL,
            stream_bytes INTEGER NOT NULL,
            latent_zlib BLOB NOT NULL,
            latent_sha256 TEXT NOT NULL,
            latent_shape TEXT NOT NULL,
            PRIMARY KEY (image_index, q_index)
        );
        """
    )


def encoded_payload(model: object, image: object, q_index: int) -> dict[str, object]:
    import torch

    encoded = model.compress(image, q_index=q_index)
    stream = encoded["bitstream"]
    latent_array = encoded["masked_latent"].detach().cpu().to(torch.float32).contiguous().numpy()
    latent_bytes = latent_array.tobytes()
    return {
        "stream": stream,
        "stream_sha256": hashlib.sha256(stream).hexdigest(),
        "stream_bytes": len(stream),
        "latent_zlib": zlib.compress(latent_bytes, level=1),
        "latent_sha256": hashlib.sha256(latent_bytes).hexdigest(),
        "latent_shape": json.dumps(list(latent_array.shape)),
    }


def main() -> None:
    import torch

    args = parse_args()
    contract = load_contract()
    if args.seed not in registered_model_seeds(contract):
        raise ValueError("Unregistered model seed")
    if args.mode == "benchmark" and args.steps != 500:
        raise ValueError("The frozen decoder benchmark is exactly 500 steps")
    if args.mode == "smoke" and (args.steps is None or not 1 <= args.steps <= 5):
        raise ValueError("Smoke cache mode requires --steps in [1, 5]")
    if args.mode == "train" and args.steps is not None:
        raise ValueError("Formal decoder caching uses the contract maximum")
    steps = int(args.steps) if args.mode != "train" else int(contract["training"]["decoder_only"]["maximum_steps"])

    configure_practical_determinism(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    train_manifest = ROOT / "data" / "core_v3_visdrone_train.csv"
    development_manifest = ROOT / "data" / "core_v3_visdrone_development.csv"
    dataset = CoreV3CropBankDataset(train_manifest, seed=args.seed, data_root=args.data_root)
    development_dataset = CoreV3FixedTileDataset(development_manifest, data_root=args.data_root)
    effective_batch = int(contract["training"]["effective_batch_size"])
    required_pairs: set[tuple[int, int]] = set()
    for step in range(steps):
        q_index, _ = action_for_step(args.seed, step, decoder_only=True)
        for source_index in decoder_source_indices(args.seed, step, effective_batch, len(dataset)):
            required_pairs.add((source_index, q_index))

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("kind") != "core_v3_joint":
        raise ValueError("Sender checkpoint is not a core-v3 joint checkpoint")
    if checkpoint["metadata"]["contract_sha256"] != contract_sha256():
        raise RuntimeError("Sender checkpoint belongs to another frozen contract")
    model = BAANICModel(use_entropy_model=True, entropy_model_type="group_hyperprior").to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.update_entropy_bottleneck(force=True)
    sender_hash = sender_state_dict_sha256(model)
    model.set_codec_metadata(sender_hash, contract_sha256())

    provenance = provenance_payload(
        [
            ROOT / "models" / "baa_nic.py",
            ROOT / "models" / "baa_nic_bitstream.py",
            ROOT / "scripts" / "core_v3_data.py",
            ROOT / "scripts" / "core_v3_training.py",
            Path(__file__).resolve(),
        ],
        [train_manifest, development_manifest],
    )
    cache_metadata = {
        **provenance,
        "status": "building",
        "mode": args.mode,
        "seed": args.seed,
        "steps": steps,
        "effective_batch_size": effective_batch,
        "source_count": len(dataset),
        "required_pairs": len(required_pairs),
        "required_validation_pairs": len(development_dataset) * 4 if args.mode == "train" else 0,
        "sender_checkpoint": str(args.checkpoint),
        "sender_state_dict_sha256": sender_hash,
        "latent_storage_dtype": "float32_exact_entropy_decode",
        "crop_bank": "one_deterministic_random_512_crop_per_frozen_training_source",
        "started_unix": time.time(),
    }

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(args.cache)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    create_schema(connection)
    existing_metadata = dict(connection.execute("SELECT key, value FROM metadata"))
    if existing_metadata:
        existing_sender = json.loads(existing_metadata["sender_state_dict_sha256"])
        existing_contract = json.loads(existing_metadata["contract_sha256"])
        existing_dtype = json.loads(existing_metadata.get("latent_storage_dtype", '"unknown"'))
        if (
            existing_sender != sender_hash
            or existing_contract != contract_sha256()
            or existing_dtype != "float32_exact_entropy_decode"
        ):
            raise RuntimeError("Existing decoder cache belongs to another sender, contract, or latent dtype")
    for key, value in cache_metadata.items():
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            (key, json.dumps(value, sort_keys=True)),
        )
    connection.commit()

    existing = {
        (int(source_index), int(q_index))
        for source_index, q_index in connection.execute("SELECT source_index, q_index FROM entries")
    }
    pending = sorted(required_pairs - existing)
    started = time.perf_counter()
    for completed, (source_index, q_index) in enumerate(pending, start=1):
        sample = dataset[source_index]
        image = sample["image"].unsqueeze(0).to(device)
        payload = encoded_payload(model, image, q_index)
        connection.execute(
            """
            INSERT OR REPLACE INTO entries(
                source_index, q_index, source_id, crop_seed, stream, stream_sha256,
                stream_bytes, latent_zlib, latent_sha256, latent_shape
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_index,
                q_index,
                sample["metadata"]["source_id"],
                str(sample["metadata"]["crop_seed"]),
                sqlite3.Binary(payload["stream"]),
                payload["stream_sha256"],
                payload["stream_bytes"],
                sqlite3.Binary(payload["latent_zlib"]),
                payload["latent_sha256"],
                payload["latent_shape"],
            ),
        )
        if completed % args.commit_interval == 0:
            connection.commit()
            elapsed = time.perf_counter() - started
            print(f"CACHE: {completed}/{len(pending)} new pairs ({completed / max(elapsed, 1e-9):.3f} pair/s)", flush=True)
    connection.commit()
    final_pairs = {
        (int(source_index), int(q_index))
        for source_index, q_index in connection.execute("SELECT source_index, q_index FROM entries")
    }
    final_count = len(final_pairs)
    missing_pairs = required_pairs - final_pairs
    if missing_pairs:
        raise RuntimeError(f"Decoder cache is missing {len(missing_pairs)} required entries")

    new_validation_entries = 0
    if args.mode == "train":
        existing_validation = {
            (int(image_index), int(q_index))
            for image_index, q_index in connection.execute("SELECT image_index, q_index FROM validation_entries")
        }
        required_validation = {
            (image_index, q_index)
            for image_index in range(len(development_dataset))
            for q_index in contract["model"]["q_indices"]
        }
        for completed, (image_index, q_index) in enumerate(sorted(required_validation - existing_validation), start=1):
            sample = development_dataset[image_index]
            image = sample["image"].unsqueeze(0).to(device)
            payload = encoded_payload(model, image, int(q_index))
            connection.execute(
                """
                INSERT OR REPLACE INTO validation_entries(
                    image_index, q_index, image_id, stream, stream_sha256, stream_bytes,
                    latent_zlib, latent_sha256, latent_shape
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    image_index,
                    int(q_index),
                    sample["metadata"]["tile_id"],
                    sqlite3.Binary(payload["stream"]),
                    payload["stream_sha256"],
                    payload["stream_bytes"],
                    sqlite3.Binary(payload["latent_zlib"]),
                    payload["latent_sha256"],
                    payload["latent_shape"],
                ),
            )
            new_validation_entries = completed
            if completed % args.commit_interval == 0:
                connection.commit()
                print(f"VALIDATION_CACHE: {completed}/{len(required_validation - existing_validation)}", flush=True)
        connection.commit()
        final_validation = {
            (int(image_index), int(q_index))
            for image_index, q_index in connection.execute("SELECT image_index, q_index FROM validation_entries")
        }
        if required_validation - final_validation:
            raise RuntimeError("Decoder validation cache is incomplete")
    cache_metadata.update(
        {
            "status": "pass",
            "entries": final_count,
            "new_entries": len(pending),
            "validation_entries": int(connection.execute("SELECT COUNT(*) FROM validation_entries").fetchone()[0]),
            "new_validation_entries": new_validation_entries,
            "build_wall_time_seconds": time.perf_counter() - started,
            "missing_pairs": 0,
            "finished_unix": time.time(),
        }
    )
    for key, value in cache_metadata.items():
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            (key, json.dumps(value, sort_keys=True)),
        )
    connection.commit()
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.close()
    sidecar = args.cache.with_suffix(args.cache.suffix + ".metadata.json")
    cache_metadata["cache_sha256"] = file_sha256(args.cache)
    sidecar.write_text(json.dumps(cache_metadata, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: decoder cache entries={final_count} new={len(pending)}")


if __name__ == "__main__":
    main()
