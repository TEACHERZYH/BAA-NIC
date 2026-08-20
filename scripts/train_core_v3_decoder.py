"""Frozen heavy/light decoder-only training over a shared practical sender cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
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
from models.baa_nic_bitstream import prefix_stream
from core_v3_contract import contract_sha256, load_contract, registered_model_seeds, state_dict_sha256
from core_v3_data import CoreV3CropBankDataset, CoreV3FixedTileDataset
from core_v3_training import (
    action_for_step,
    atomic_torch_save,
    configure_determinism,
    decoder_source_indices,
    learning_rate,
    provenance_payload,
    set_optimizer_lr,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--decoder", choices=("heavy", "light"), required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=("smoke", "benchmark", "train"), default="train")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log-interval", type=int, default=25)
    return parser.parse_args()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def load_cache_metadata(connection: sqlite3.Connection) -> dict[str, Any]:
    return {key: json.loads(value) for key, value in connection.execute("SELECT key, value FROM metadata")}


def decode_latent_row(row: tuple[Any, ...]) -> tuple[np.ndarray, bytes, str, str]:
    source_id, crop_seed, stream, stream_hash, latent_zlib, latent_hash, latent_shape = row
    stream = bytes(stream)
    if hashlib.sha256(stream).hexdigest() != stream_hash:
        raise RuntimeError("Cached practical stream hash mismatch")
    latent_bytes = zlib.decompress(latent_zlib)
    if hashlib.sha256(latent_bytes).hexdigest() != latent_hash:
        raise RuntimeError("Cached latent hash mismatch")
    shape = tuple(int(value) for value in json.loads(latent_shape))
    latent = np.frombuffer(latent_bytes, dtype=np.float32).reshape(shape).copy()
    return latent, stream, str(source_id), str(crop_seed)


def training_sample(
    connection: sqlite3.Connection,
    dataset: CoreV3CropBankDataset,
    source_index: int,
    q_index: int,
    prefix_k: int,
) -> tuple[Any, Any, Any, str]:
    import torch

    row = connection.execute(
        """
        SELECT source_id, crop_seed, stream, stream_sha256, latent_zlib, latent_sha256, latent_shape
        FROM entries WHERE source_index=? AND q_index=?
        """,
        (source_index, q_index),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"Missing decoder cache entry source={source_index}, q={q_index}")
    latent, stream, source_id, crop_seed = decode_latent_row(row)
    sample = dataset[source_index]
    if sample["metadata"]["source_id"] != source_id or str(sample["metadata"]["crop_seed"]) != crop_seed:
        raise RuntimeError("Cached sender output and deterministic crop metadata differ")
    prefix = prefix_stream(stream, prefix_k)
    full_latent = torch.from_numpy(latent[0].astype(np.float32, copy=False))
    latent_tensor = full_latent.clone()
    if prefix_k < 4:
        latent_tensor[prefix_k * 48 :] = 0.0
    return sample["image"], latent_tensor, full_latent, hashlib.sha256(prefix).hexdigest()


def indicator_tensor(batch: int, prefix_k: int, height: int, width: int, device: Any) -> Any:
    import torch

    indicator = torch.zeros((batch, 4, height, width), dtype=torch.float32, device=device)
    indicator[:, :prefix_k] = 1.0
    return indicator


def validation_metrics(
    decoder: Any,
    connection: sqlite3.Connection,
    dataset: CoreV3FixedTileDataset,
    limit: int,
    device: Any,
    micro_batch_size: int,
    amp_enabled: bool,
) -> dict[str, float]:
    import torch
    import torch.nn.functional as F
    from pytorch_msssim import ms_ssim

    was_training = decoder.training
    decoder.eval()
    total_msssim = 0.0
    total_loss = 0.0
    observations = 0
    with torch.inference_mode():
        for q_index in load_contract()["model"]["q_indices"]:
            for prefix_k in range(1, 5):
                for start in range(0, limit, micro_batch_size):
                    images: list[Any] = []
                    latents: list[Any] = []
                    for image_index in range(start, min(start + micro_batch_size, limit)):
                        row = connection.execute(
                            """
                            SELECT image_id, '', stream, stream_sha256, latent_zlib, latent_sha256, latent_shape
                            FROM validation_entries WHERE image_index=? AND q_index=?
                            """,
                            (image_index, int(q_index)),
                        ).fetchone()
                        if row is None:
                            raise RuntimeError("Formal decoder validation cache is incomplete")
                        latent, stream, image_id, _ = decode_latent_row(row)
                        sample = dataset[image_index]
                        if sample["metadata"]["tile_id"] != image_id:
                            raise RuntimeError("Validation cache image ID mismatch")
                        prefix_stream(stream, prefix_k)
                        latent_tensor = torch.from_numpy(latent[0].astype(np.float32, copy=False))
                        if prefix_k < 4:
                            latent_tensor[prefix_k * 48 :] = 0.0
                        images.append(sample["image"])
                        latents.append(latent_tensor)
                    image_batch = torch.stack(images).to(device)
                    latent_batch = torch.stack(latents).to(device)
                    indicator = indicator_tensor(
                        len(images), prefix_k, latent_batch.shape[-2], latent_batch.shape[-1], device
                    )
                    with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                        reconstruction = decoder(latent_batch, indicator, output_size=image_batch.shape[-2:])
                        l1 = F.l1_loss(reconstruction, image_batch)
                        mse = F.mse_loss(reconstruction, image_batch)
                    score = ms_ssim(reconstruction.float(), image_batch.float(), data_range=1.0, size_average=False)
                    count = len(images)
                    total_msssim += float(score.sum().cpu())
                    total_loss += float((l1 + mse).detach().cpu()) * count
                    observations += count
    if was_training:
        decoder.train()
    return {
        "mean_ms_ssim": total_msssim / observations,
        "mean_l1_plus_mse": total_loss / observations,
        "observations": float(observations),
    }


def checkpoint_payload(
    decoder: Any,
    optimizer: Any,
    scaler: Any,
    step: int,
    best_msssim: float,
    stale_full_validations: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "kind": f"core_v3_decoder_{metadata['decoder']}",
        "step": step,
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_full_development_ms_ssim": best_msssim,
        "stale_full_validations": stale_full_validations,
        "metadata": metadata,
    }


def main() -> None:
    import torch
    import torch.nn.functional as F

    args = parse_args()
    contract = load_contract()
    if args.seed not in registered_model_seeds(contract):
        raise ValueError("Unregistered model seed")
    if args.mode == "benchmark" and args.steps != 500:
        raise ValueError("The frozen decoder benchmark is exactly 500 steps")
    if args.mode == "smoke" and (args.steps is None or not 1 <= args.steps <= 5):
        raise ValueError("Smoke mode requires --steps in [1, 5]")
    if args.mode == "train" and args.steps is not None:
        raise ValueError("Formal decoder training uses the contract maximum")

    decoder_contract = contract["training"]["decoder_only"]
    run_steps = int(args.steps) if args.mode != "train" else int(decoder_contract["maximum_steps"])
    effective_batch = int(contract["training"]["effective_batch_size"])
    if effective_batch % args.micro_batch_size:
        raise ValueError("micro batch size must divide the frozen effective batch size")
    accumulation_steps = effective_batch // args.micro_batch_size
    decoder_seed = args.seed + int(decoder_contract["decoder_seed_offset"])
    configure_determinism(decoder_seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    amp_enabled = bool(contract["training"]["amp"] and device.type == "cuda")

    connection = sqlite3.connect(f"file:{args.cache.as_posix()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    cache_metadata = load_cache_metadata(connection)
    if cache_metadata.get("status") != "pass" or cache_metadata.get("contract_sha256") != contract_sha256():
        raise RuntimeError("Decoder cache is incomplete or belongs to another contract")
    if int(cache_metadata.get("seed")) != args.seed:
        raise RuntimeError("Decoder cache belongs to another model seed")

    train_manifest = ROOT / "data" / "core_v3_visdrone_train.csv"
    development_manifest = ROOT / "data" / "core_v3_visdrone_development.csv"
    crop_bank = CoreV3CropBankDataset(train_manifest, seed=args.seed, data_root=args.data_root)
    development = CoreV3FixedTileDataset(development_manifest, data_root=args.data_root)
    decoder = (ServerDecoder() if args.decoder == "heavy" else LightServerDecoder()).to(device)
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=float(decoder_contract["initial_lr"]),
        weight_decay=float(decoder_contract["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    provenance = provenance_payload(
        [
            ROOT / "models" / "baa_nic.py",
            ROOT / "models" / "core_v3_controls.py",
            ROOT / "models" / "baa_nic_bitstream.py",
            ROOT / "scripts" / "core_v3_data.py",
            ROOT / "scripts" / "core_v3_training.py",
            Path(__file__).resolve(),
        ],
        [train_manifest, development_manifest],
    )
    run_metadata = {
        **provenance,
        "status": "running",
        "mode": args.mode,
        "seed": args.seed,
        "decoder_seed": decoder_seed,
        "decoder": args.decoder,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cache": str(args.cache),
        "cache_sha256": cache_metadata.get("cache_sha256"),
        "sender_state_dict_sha256": cache_metadata["sender_state_dict_sha256"],
        "effective_batch_size": effective_batch,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "run_steps": run_steps,
        "started_unix": time.time(),
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.run_dir / "metadata.json"
    log_path = args.run_dir / "train.jsonl"
    metadata_path.write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")

    start_step = 0
    best_msssim = float("-inf")
    stale_full_validations = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("kind") != f"core_v3_decoder_{args.decoder}":
            raise ValueError("Resume checkpoint is for another decoder branch")
        if checkpoint["metadata"]["contract_sha256"] != contract_sha256():
            raise RuntimeError("Resume checkpoint belongs to another contract")
        decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint["step"])
        best_msssim = float(checkpoint["best_full_development_ms_ssim"])
        stale_full_validations = int(checkpoint["stale_full_validations"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    stopped_early = False
    for step in range(start_step, run_steps):
        decoder.train()
        q_index, prefix_k = action_for_step(args.seed, step, decoder_only=True)
        source_indices = decoder_source_indices(args.seed, step, effective_batch, len(crop_bank))
        lr = learning_rate(
            step,
            int(decoder_contract["maximum_steps"]),
            0,
            float(decoder_contract["initial_lr"]),
            float(decoder_contract["minimum_lr"]),
        )
        set_optimizer_lr(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)
        metrics = {"loss": 0.0, "l1": 0.0, "mse": 0.0, "cross_rate": 0.0}
        stream_hashes: list[str] = []
        for micro_start in range(0, effective_batch, args.micro_batch_size):
            images: list[Any] = []
            latents: list[Any] = []
            full_latents: list[Any] = []
            for source_index in source_indices[micro_start : micro_start + args.micro_batch_size]:
                image, latent, full_latent, stream_hash = training_sample(
                    connection, crop_bank, source_index, q_index, prefix_k
                )
                images.append(image)
                latents.append(latent)
                full_latents.append(full_latent)
                stream_hashes.append(stream_hash)
            image_batch = torch.stack(images).to(device)
            latent_batch = torch.stack(latents).to(device)
            indicator = indicator_tensor(
                len(images), prefix_k, latent_batch.shape[-2], latent_batch.shape[-1], device
            )
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                reconstruction = decoder(latent_batch, indicator, output_size=image_batch.shape[-2:])
                l1 = F.l1_loss(reconstruction, image_batch)
                mse = F.mse_loss(reconstruction, image_batch)
                if prefix_k == 4:
                    cross_rate = reconstruction.new_zeros(())
                else:
                    full_latent = torch.stack(full_latents).to(device)
                    full_indicator = indicator_tensor(
                        len(images), 4, latent_batch.shape[-2], latent_batch.shape[-1], device
                    )
                    with torch.no_grad():
                        full_reference = decoder(
                            full_latent, full_indicator, output_size=image_batch.shape[-2:]
                        ).detach()
                    cross_rate = F.l1_loss(reconstruction, full_reference)
                loss = l1 + mse + float(contract["training"]["loss"]["cross_rate_weight"]) * cross_rate
                scaled_loss = loss / accumulation_steps
            scaler.scale(scaled_loss).backward()
            for name, value in (("loss", loss), ("l1", l1), ("mse", mse), ("cross_rate", cross_rate)):
                metrics[name] += float(value.detach().cpu()) / accumulation_steps
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0).detach().cpu())
        scaler.step(optimizer)
        scaler.update()

        completed_step = step + 1
        if completed_step % args.log_interval == 0 or completed_step == 1 or completed_step == run_steps:
            elapsed = time.perf_counter() - started
            append_jsonl(
                log_path,
                {
                    "event": "train",
                    "step": completed_step,
                    "q_index": q_index,
                    "prefix_k": prefix_k,
                    "lr": lr,
                    "grad_norm": grad_norm,
                    "batch_prefix_stream_sha256": hashlib.sha256("".join(stream_hashes).encode("ascii")).hexdigest(),
                    "steps_per_second": (completed_step - start_step) / max(elapsed, 1e-9),
                    **metrics,
                },
            )

        if args.mode == "train" and completed_step % int(decoder_contract["monitor_validation_interval"]) == 0:
            result = validation_metrics(
                decoder, connection, development, 64, device, args.micro_batch_size, amp_enabled
            )
            append_jsonl(log_path, {"event": "monitor_validation", "step": completed_step, **result})

        if args.mode == "train" and completed_step % int(decoder_contract["full_validation_interval"]) == 0:
            result = validation_metrics(
                decoder, connection, development, len(development), device, args.micro_batch_size, amp_enabled
            )
            improved = result["mean_ms_ssim"] > best_msssim
            if improved:
                best_msssim = result["mean_ms_ssim"]
                stale_full_validations = 0
                atomic_torch_save(
                    checkpoint_payload(
                        decoder, optimizer, scaler, completed_step, best_msssim, stale_full_validations, run_metadata
                    ),
                    args.run_dir / "best.pt",
                )
            else:
                stale_full_validations += 1
            append_jsonl(
                log_path,
                {"event": "full_validation", "step": completed_step, "improved": improved, **result},
            )
            if completed_step >= int(decoder_contract["minimum_steps"]) and stale_full_validations >= int(
                decoder_contract["full_validation_patience"]
            ):
                stopped_early = True

        if completed_step % 500 == 0 or completed_step == run_steps or stopped_early:
            atomic_torch_save(
                checkpoint_payload(
                    decoder, optimizer, scaler, completed_step, best_msssim, stale_full_validations, run_metadata
                ),
                args.run_dir / "last.pt",
            )
        if stopped_early:
            break

    elapsed = time.perf_counter() - started
    run_metadata.update(
        {
            "status": "pass",
            "stop_reason": "early_stopping" if stopped_early else "requested_steps_complete",
            "final_step": completed_step,
            "wall_time_seconds": elapsed,
            "steps_per_second": (completed_step - start_step) / max(elapsed, 1e-9),
            "peak_allocated_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "decoder_state_dict_sha256": state_dict_sha256(decoder),
            "finished_unix": time.time(),
        }
    )
    metadata_path.write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
    if args.mode == "benchmark":
        (args.run_dir / "benchmark.json").write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
    connection.close()
    print(
        f"PASS: core-v3 {args.decoder} decoder {args.mode} seed={args.seed} "
        f"steps={completed_step - start_step} throughput={run_metadata['steps_per_second']:.4f} step/s"
    )


if __name__ == "__main__":
    main()
