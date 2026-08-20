"""Measure enhanced versus frozen-heavy decoder compute on one received latent."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import ServerDecoder, build_enhanced_server_decoder
from core_v3_contract import state_dict_sha256
from core_v3_decoder_extension import (
    extension_contract_sha256,
    load_extension_contract,
    registered_extension_seeds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--enhanced-checkpoint", type=Path, required=True)
    parser.add_argument("--heavy-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[position]


def profile_flops(call: Callable[[], Any], device: Any) -> int:
    import torch
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, with_flops=True, record_shapes=False) as result:
        call()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return int(sum(int(event.flops or 0) for event in result.key_averages()))


def measure(call: Callable[[], Any], device: Any, warmup: int, repeats: int) -> dict[str, Any]:
    import torch

    for _ in range(warmup):
        call()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    timings: list[float] = []
    for _ in range(repeats):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            call()
            end.record()
            torch.cuda.synchronize(device)
            timings.append(float(start.elapsed_time(end)))
        else:
            import time

            started = time.perf_counter()
            call()
            timings.append((time.perf_counter() - started) * 1000.0)
    return {
        "latency_median_ms": statistics.median(timings),
        "latency_p95_ms": percentile(timings, 0.95),
        "peak_allocated_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else ""
        ),
    }


def main() -> None:
    import torch

    args = parse_args()
    extension = load_extension_contract()
    if args.seed not in registered_extension_seeds(extension):
        raise ValueError("Unregistered seed")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    enhanced_checkpoint = torch.load(args.enhanced_checkpoint, map_location=device, weights_only=False)
    heavy_checkpoint = torch.load(args.heavy_checkpoint, map_location=device, weights_only=False)
    if enhanced_checkpoint.get("kind") != "core_v3_decoder_enhanced_r1":
        raise RuntimeError("Invalid enhanced checkpoint")
    if enhanced_checkpoint.get("extension_contract_sha256") != extension_contract_sha256():
        raise RuntimeError("Enhanced checkpoint contract mismatch")
    if heavy_checkpoint.get("kind") != "core_v3_decoder_heavy":
        raise RuntimeError("Invalid heavy checkpoint")
    enhanced_sender = enhanced_checkpoint.get("metadata", {}).get("sender_state_dict_sha256")
    heavy_sender = heavy_checkpoint.get("metadata", {}).get("sender_state_dict_sha256")
    if enhanced_sender != heavy_sender:
        raise RuntimeError("Decoder checkpoints do not belong to the same sender")

    enhanced = build_enhanced_server_decoder(extension).to(device).eval()
    enhanced.load_state_dict(enhanced_checkpoint["decoder"])
    heavy = ServerDecoder().to(device).eval()
    heavy.load_state_dict(heavy_checkpoint["decoder"])
    latent = torch.zeros((1, 192, 32, 32), device=device)
    indicator = torch.ones((1, 4, 32, 32), device=device)
    calls = {
        "enhanced_r1": lambda: enhanced(latent, indicator, q_index=1, output_size=(512, 512)),
        "heavy": lambda: heavy(latent, indicator, output_size=(512, 512)),
    }
    modules = {"enhanced_r1": enhanced, "heavy": heavy}
    checkpoints = {"enhanced_r1": args.enhanced_checkpoint, "heavy": args.heavy_checkpoint}
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for name, call in calls.items():
            stats = measure(call, device, args.warmup, args.repeats)
            flops = profile_flops(call, device)
            rows.append(
                {
                    "extension_version": extension["extension_version"],
                    "extension_contract_sha256": extension_contract_sha256(),
                    "seed": args.seed,
                    "decoder": name,
                    "device": str(device),
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
                    "params": sum(parameter.numel() for parameter in modules[name].parameters()),
                    "checkpoint_bytes": checkpoints[name].stat().st_size,
                    "state_dict_sha256": state_dict_sha256(modules[name]),
                    "profiled_flops_512": flops,
                    "profiled_gflops_512": f"{flops / 1e9:.8f}",
                    **stats,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"PASS: decoder extension complexity seed={args.seed} rows={len(rows)}")


if __name__ == "__main__":
    main()
