"""Frozen joint encoder, entropy-model, and temporary-decoder training entrypoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import BAANICModel
from core_v3_contract import load_contract, registered_model_seeds, state_dict_sha256
from core_v3_data import CoreV3FixedTileDataset, CoreV3TrainingDataset
from core_v3_training import (
    atomic_torch_save,
    action_for_step,
    collate_core_v3,
    configure_determinism,
    learning_rate,
    provenance_payload,
    reconstruction_objective,
    set_optimizer_lr,
    split_entropy_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
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


def evaluate_joint(
    model: Any,
    manifest: Path,
    data_root: Path | None,
    device: Any,
    batch_size: int,
    workers: int,
    amp_enabled: bool,
) -> dict[str, float]:
    import torch
    from torch.utils.data import DataLoader

    dataset = CoreV3FixedTileDataset(manifest, data_root=data_root)
    total = {name: 0.0 for name in ("loss", "l1", "mse", "rate", "cross_rate")}
    observations = 0
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for q_index in load_contract()["model"]["q_indices"]:
            for prefix_k in range(1, 5):
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=workers,
                    pin_memory=device.type == "cuda",
                    collate_fn=collate_core_v3,
                )
                for batch in loader:
                    images = batch["image"].to(device, non_blocking=True)
                    with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                        result = reconstruction_objective(model, images, int(q_index), prefix_k, include_rate=True)
                    count = int(images.shape[0])
                    observations += count
                    for name in total:
                        total[name] += float(result[name].detach().cpu()) * count
    if was_training:
        model.train()
    return {name: value / observations for name, value in total.items()} | {"observations": float(observations)}


def checkpoint_payload(
    model: Any,
    optimizer: Any,
    aux_optimizer: Any,
    scaler: Any,
    step: int,
    best_loss: float,
    stale_full_validations: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "kind": "core_v3_joint",
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "aux_optimizer": aux_optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_full_development_loss": best_loss,
        "stale_full_validations": stale_full_validations,
        "metadata": metadata,
    }


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    args = parse_args()
    contract = load_contract()
    registered_seeds = registered_model_seeds(contract)
    if args.seed not in registered_seeds:
        raise ValueError(f"Seed must be one of {registered_seeds}")
    if args.mode == "benchmark" and args.steps != 500:
        raise ValueError("The frozen remote benchmark is exactly 500 steps")
    if args.mode == "smoke" and (args.steps is None or not 1 <= args.steps <= 5):
        raise ValueError("Smoke mode requires --steps in [1, 5]")
    if args.mode == "train" and args.steps is not None:
        raise ValueError("Formal training uses the contract maximum; --steps is not allowed")

    configure_determinism(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    joint = contract["training"]["joint"]
    effective_batch = int(contract["training"]["effective_batch_size"])
    if effective_batch % args.micro_batch_size:
        raise ValueError("micro batch size must divide the frozen effective batch size")
    accumulation_steps = effective_batch // args.micro_batch_size
    run_steps = int(args.steps) if args.mode != "train" else int(joint["maximum_steps"])
    amp_enabled = bool(contract["training"]["amp"] and device.type == "cuda")

    train_manifest = ROOT / "data" / "core_v3_visdrone_train.csv"
    monitor_manifest = ROOT / "data" / "core_v3_visdrone_development_monitor.csv"
    full_manifest = ROOT / "data" / "core_v3_visdrone_development.csv"
    code_paths = [
        ROOT / "models" / "baa_nic.py",
        ROOT / "models" / "baa_nic_bitstream.py",
        ROOT / "scripts" / "core_v3_data.py",
        ROOT / "scripts" / "core_v3_training.py",
        Path(__file__).resolve(),
    ]
    provenance = provenance_payload(code_paths, [train_manifest, monitor_manifest, full_manifest])
    run_metadata = {
        **provenance,
        "status": "running",
        "mode": args.mode,
        "seed": args.seed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "effective_batch_size": effective_batch,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "run_steps": run_steps,
        "data_root": str(args.data_root) if args.data_root else None,
        "started_unix": time.time(),
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.run_dir / "metadata.json"
    metadata_path.write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
    log_path = args.run_dir / "train.jsonl"

    model = BAANICModel(use_entropy_model=True, entropy_model_type="group_hyperprior").to(device)
    main_parameters, aux_parameters = split_entropy_parameters(model)
    optimizer = torch.optim.AdamW(
        main_parameters,
        lr=float(joint["initial_lr"]),
        weight_decay=float(joint["weight_decay"]),
    )
    aux_optimizer = torch.optim.Adam(aux_parameters, lr=float(joint["aux_lr"]))
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    start_step = 0
    best_loss = float("inf")
    stale_full_validations = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("kind") != "core_v3_joint":
            raise ValueError("Resume checkpoint is not a core-v3 joint checkpoint")
        if checkpoint["metadata"]["contract_sha256"] != provenance["contract_sha256"]:
            raise RuntimeError("Resume checkpoint belongs to another frozen contract")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint["step"])
        best_loss = float(checkpoint["best_full_development_loss"])
        stale_full_validations = int(checkpoint["stale_full_validations"])
    if start_step >= run_steps:
        raise ValueError("Resume checkpoint has already reached the requested run length")

    dataset = CoreV3TrainingDataset(
        train_manifest,
        seed=args.seed,
        total_samples=(run_steps - start_step) * effective_batch,
        start_sample=start_step * effective_batch,
        data_root=args.data_root,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
        collate_fn=collate_core_v3,
    )
    iterator = iter(loader)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()
    stopped_early = False

    for step in range(start_step, run_steps):
        model.train()
        q_index, prefix_k = action_for_step(args.seed, step, decoder_only=False)
        lr = learning_rate(
            step,
            int(joint["maximum_steps"]),
            int(joint["lr_warmup_steps"]),
            float(joint["initial_lr"]),
            float(joint["minimum_lr"]),
        )
        set_optimizer_lr(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)
        metrics = {name: 0.0 for name in ("loss", "l1", "mse", "rate", "cross_rate")}
        for _ in range(accumulation_steps):
            batch = next(iterator)
            images = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                result = reconstruction_objective(model, images, q_index, prefix_k, include_rate=True)
                scaled_loss = result["loss"] / accumulation_steps
            scaler.scale(scaled_loss).backward()
            for name in metrics:
                metrics[name] += float(result[name].detach().cpu()) / accumulation_steps
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(main_parameters, max_norm=1.0).detach().cpu())
        scaler.step(optimizer)
        scaler.update()

        aux_optimizer.zero_grad(set_to_none=True)
        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        completed_step = step + 1
        if completed_step % args.log_interval == 0 or completed_step == 1 or completed_step == run_steps:
            elapsed = time.perf_counter() - training_started
            append_jsonl(
                log_path,
                {
                    "event": "train",
                    "step": completed_step,
                    "q_index": q_index,
                    "prefix_k": prefix_k,
                    "lr": lr,
                    "aux_loss": float(aux_loss.detach().cpu()),
                    "grad_norm": grad_norm,
                    "steps_per_second": (completed_step - start_step) / max(elapsed, 1e-9),
                    **metrics,
                },
            )

        if args.mode == "train" and completed_step % int(joint["monitor_validation_interval"]) == 0:
            validation = evaluate_joint(
                model,
                monitor_manifest,
                args.data_root,
                device,
                args.micro_batch_size,
                args.workers,
                amp_enabled,
            )
            append_jsonl(log_path, {"event": "monitor_validation", "step": completed_step, **validation})

        if args.mode == "train" and completed_step % int(joint["full_validation_interval"]) == 0:
            validation = evaluate_joint(
                model,
                full_manifest,
                args.data_root,
                device,
                args.micro_batch_size,
                args.workers,
                amp_enabled,
            )
            improved = validation["loss"] < best_loss
            if improved:
                best_loss = validation["loss"]
                stale_full_validations = 0
                atomic_torch_save(
                    checkpoint_payload(
                        model, optimizer, aux_optimizer, scaler, completed_step, best_loss, stale_full_validations, run_metadata
                    ),
                    args.run_dir / "best.pt",
                )
            else:
                stale_full_validations += 1
            append_jsonl(
                log_path,
                {"event": "full_validation", "step": completed_step, "improved": improved, **validation},
            )
            if completed_step >= int(joint["minimum_steps"]) and stale_full_validations >= int(
                joint["full_validation_patience"]
            ):
                stopped_early = True

        if completed_step % 1000 == 0 or completed_step == run_steps or stopped_early:
            atomic_torch_save(
                checkpoint_payload(
                    model, optimizer, aux_optimizer, scaler, completed_step, best_loss, stale_full_validations, run_metadata
                ),
                args.run_dir / "last.pt",
            )
        if stopped_early:
            break

    elapsed = time.perf_counter() - training_started
    final_step = completed_step
    final_checkpoint = args.run_dir / "last.pt"
    final_hash = state_dict_sha256(model)
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    run_metadata.update(
        {
            "status": "pass",
            "stop_reason": "early_stopping" if stopped_early else "requested_steps_complete",
            "final_step": final_step,
            "wall_time_seconds": elapsed,
            "steps_per_second": (final_step - start_step) / max(elapsed, 1e-9),
            "peak_allocated_memory_bytes": peak_memory,
            "final_state_dict_sha256": final_hash,
            "final_checkpoint": str(final_checkpoint),
            "finished_unix": time.time(),
        }
    )
    metadata_path.write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
    if args.mode == "benchmark":
        (args.run_dir / "benchmark.json").write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"PASS: core-v3 joint {args.mode} seed={args.seed} steps={final_step - start_step} "
        f"throughput={run_metadata['steps_per_second']:.4f} step/s"
    )


if __name__ == "__main__":
    main()
