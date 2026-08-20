"""Generate the frozen paired development and test traces for core-v3 E4."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_v3_contract import contract_sha256, file_sha256, files_sha256, load_contract


PROFILE_NAMES = ("smooth_ar", "drop_recovery", "burst_loss")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def rng_for(seed: int, profile_index: int, stage_index: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, profile_index, stage_index])))


def log_ar(
    rng: np.random.Generator,
    frames: int,
    mean_mbps: float,
    phi: float,
    sigma: float,
    minimum_mbps: float,
    maximum_mbps: float,
) -> np.ndarray:
    values = np.empty(frames, dtype=np.float64)
    log_mean = math.log(mean_mbps)
    state = log_mean
    for index in range(frames):
        if index:
            state = log_mean + phi * (state - log_mean) + sigma * float(rng.normal())
        values[index] = min(max(math.exp(state), minimum_mbps), maximum_mbps)
    return values


def generate_profile(
    profile: str,
    config: dict[str, Any],
    seed: int,
    profile_index: int,
    stage_index: int,
    frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = rng_for(seed, profile_index, stage_index)
    loss = np.zeros(frames, dtype=np.float64)
    if profile == "smooth_ar":
        raw = log_ar(
            rng,
            frames,
            config["mean_mbps"],
            config["phi"],
            config["innovation_sigma"],
            config["minimum_mbps"],
            config["maximum_mbps"],
        )
    elif profile == "drop_recovery":
        raw = np.empty(frames, dtype=np.float64)
        high = True
        for index in range(frames):
            if index:
                if high and rng.random() < config["high_to_low_probability"]:
                    high = False
                elif not high and rng.random() < config["low_to_high_probability"]:
                    high = True
            mean = config["high_mean_mbps"] if high else config["low_mean_mbps"]
            sigma = config["high_log_sigma"] if high else config["low_log_sigma"]
            raw[index] = min(
                max(math.exp(math.log(mean) + sigma * float(rng.normal())), config["minimum_mbps"]),
                config["maximum_mbps"],
            )
    elif profile == "burst_loss":
        raw = log_ar(
            rng,
            frames,
            config["raw_mean_mbps"],
            config["raw_phi"],
            config["raw_innovation_sigma"],
            0.1,
            8.0,
        )
        bad = False
        for index in range(frames):
            if index:
                if not bad and rng.random() < config["good_to_bad_probability"]:
                    bad = True
                elif bad and rng.random() < config["bad_to_good_probability"]:
                    bad = False
            lower, upper = config["bad_loss_range"] if bad else config["good_loss_range"]
            loss[index] = rng.uniform(lower, upper)
    else:
        raise ValueError(profile)
    effective = raw * (1.0 - loss) * 1_000_000.0
    return raw * 1_000_000.0, loss, effective


def generate_stage(
    contract: dict[str, Any],
    stage: str,
    realizations: int,
    seed_start: int,
    initial_goodput: float | None,
) -> dict[str, np.ndarray]:
    network = contract["network_replay"]
    frames = int(network["frames_per_realization"])
    true_goodput = np.empty((3, realizations, frames), dtype=np.float64)
    raw_capacity = np.empty_like(true_goodput)
    packet_loss = np.empty_like(true_goodput)
    for profile_index, profile in enumerate(PROFILE_NAMES):
        for realization in range(realizations):
            raw, loss, effective = generate_profile(
                profile,
                network["profiles"][profile],
                seed_start + realization,
                profile_index,
                0 if stage == "development" else 1,
                frames,
            )
            raw_capacity[profile_index, realization] = raw
            packet_loss[profile_index, realization] = loss
            true_goodput[profile_index, realization] = effective
    if initial_goodput is None:
        initial_goodput = float(np.median(true_goodput))
    estimated = np.empty_like(true_goodput)
    loss_estimate = np.empty_like(true_goodput)
    sigma = float(network["estimator_log_noise_sigma"])
    alpha = float(network["packet_loss_ewma_alpha"])
    for profile_index in range(3):
        for realization in range(realizations):
            noise_rng = rng_for(seed_start + realization, profile_index, 10 if stage == "development" else 11)
            estimated[profile_index, realization, 0] = initial_goodput
            loss_estimate[profile_index, realization, 0] = 0.0
            for frame in range(1, frames):
                estimated[profile_index, realization, frame] = true_goodput[
                    profile_index, realization, frame - 1
                ] * math.exp(sigma * float(noise_rng.normal()))
                loss_estimate[profile_index, realization, frame] = (
                    alpha * packet_loss[profile_index, realization, frame - 1]
                    + (1.0 - alpha) * loss_estimate[profile_index, realization, frame - 1]
                )
    image_count = 256 if stage == "development" else 512
    image_indices = np.empty((realizations, frames), dtype=np.int16)
    for realization in range(realizations):
        seed = seed_start + realization
        first = np.random.Generator(np.random.PCG64(seed)).permutation(image_count)
        if stage == "development":
            second = np.random.Generator(np.random.PCG64(seed + 10_000)).permutation(image_count)
            sequence = np.concatenate([first, second[: frames - image_count]])
        else:
            sequence = first[:frames]
        image_indices[realization] = sequence
    return {
        "true_goodput": true_goodput.astype(np.float32),
        "raw_capacity": raw_capacity.astype(np.float32),
        "packet_loss": packet_loss.astype(np.float32),
        "estimated_goodput": estimated.astype(np.float32),
        "packet_loss_estimate": loss_estimate.astype(np.float32),
        "image_indices": image_indices,
    }


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def main() -> None:
    args = parse_args()
    contract = load_contract()
    network = contract["network_replay"]
    if tuple(network["profiles"].keys()) != PROFILE_NAMES:
        raise RuntimeError("Profile order differs from the frozen generator")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    development = generate_stage(
        contract,
        "development",
        int(network["development_realizations_per_profile"]),
        int(network["development_seed_start"]),
        None,
    )
    initial_goodput = float(np.median(development["true_goodput"]))
    test = generate_stage(
        contract,
        "test",
        int(network["test_realizations_per_profile"]),
        int(network["test_seed_start"]),
        initial_goodput,
    )
    dev_path = args.output_dir / "development_traces.npz"
    test_path = args.output_dir / "test_traces.npz"
    save_npz(dev_path, development)
    save_npz(test_path, test)

    ratios = development["true_goodput"].astype(np.float64) / np.maximum(
        development["estimated_goodput"].astype(np.float64), 1.0
    )
    loss_estimate = development["packet_loss_estimate"].astype(np.float64)
    bin_edges = np.asarray(network["packet_loss_calibration_bins"], dtype=np.float64)
    counts: list[int] = []
    quantiles: list[float] = []
    for index in range(3):
        selected = ratios[(loss_estimate >= bin_edges[index]) & (loss_estimate < bin_edges[index + 1])]
        counts.append(int(selected.size))
        quantiles.append(float(np.quantile(selected, network["controller_goodput_lower_quantile"])))
    severe_thresholds = {
        profile: float(
            np.quantile(
                development["true_goodput"][profile_index],
                network["severe_link_development_quantile"],
            )
        )
        for profile_index, profile in enumerate(PROFILE_NAMES)
    }
    config_hash = files_sha256([ROOT / "configs" / "core_v3_experiment_contract.json", Path(__file__).resolve()])
    metadata = {
        "status": "pass",
        "contract_version": contract["contract_version"],
        "contract_sha256": contract_sha256(),
        "code_hash": config_hash,
        "profiles": list(PROFILE_NAMES),
        "initial_goodput_bps": initial_goodput,
        "goodput_ratio_global_q10": float(
            np.quantile(ratios, network["controller_goodput_lower_quantile"])
        ),
        "goodput_ratio_bin_q10": quantiles,
        "goodput_ratio_bin_counts": counts,
        "severe_link_threshold_bps": severe_thresholds,
        "development_trace_sha256": file_sha256(dev_path),
        "test_trace_sha256": file_sha256(test_path),
        "trace_config_sha256": hashlib.sha256(
            json.dumps(network, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"PASS: traces development={development['true_goodput'].shape} test={test['true_goodput'].shape}",
        flush=True,
    )


if __name__ == "__main__":
    main()
