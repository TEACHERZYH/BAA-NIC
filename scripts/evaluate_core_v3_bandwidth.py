"""Run the frozen causal core-v3 bandwidth replay from practical action caches."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.core_v3_controller import (
    ActionCalibration,
    CoreV3BandwidthController,
    ControllerDecision,
    WorkerAvailability,
)
from core_v3_contract import contract_sha256, file_sha256, files_sha256, load_contract


PROFILE_NAMES = ("smooth_ar", "drop_recovery", "burst_loss")
DETAIL_FIELDS = [
    "contract_version", "contract_sha256", "code_hash", "checkpoint_sha256", "manifest_sha256",
    "model_seed", "profile", "realization_id", "deadline_ms", "frame_id", "policy", "image_id",
    "true_goodput", "estimated_goodput", "packet_loss", "encoder_available_ms",
    "transmitter_available_ms", "decoder_available_ms", "wire_budget_bytes", "q_index",
    "nominal_prefix", "requested_prefix", "delivered_prefix", "payload_bytes", "network_bytes",
    "encoder_wait_ms", "encode_ms", "transmission_wait_ms", "transmission_ms", "decoder_wait_ms",
    "decode_ms", "delivered", "ms_ssim", "u_frame", "calibration_sha256", "trace_config_sha256",
]
SUMMARY_FIELDS = [
    "contract_version", "contract_sha256", "code_hash", "checkpoint_sha256", "manifest_sha256",
    "model_seed", "profile", "realization_id", "deadline_ms", "policy", "frames",
    "delivered_ratio", "severe_link_delivery", "mean_u_frame", "mean_overshoot",
    "deadline_violation_ratio", "encoder_wait_mean_ms", "transmission_wait_mean_ms",
    "decoder_wait_mean_ms", "total_wait_p95_ms", "total_wait_p99_ms", "total_wait_p999_ms",
    "end_to_end_mean_ms", "end_to_end_p95_ms", "end_to_end_p99_ms", "unique_pixel_throughput",
    "delivered_psnr", "delivered_ms_ssim", "policy_switch_rate", "active_ratio_variation",
    "calibration_sha256", "trace_config_sha256",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--openjpeg-cache", type=Path, required=True)
    parser.add_argument("--traces-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--smoke-realizations", type=int, default=2)
    parser.add_argument("--omit-detail", action="store_true")
    return parser.parse_args()


@dataclass
class WorkerState:
    encoder_s: float = 0.0
    transmitter_s: float = 0.0
    decoder_s: float = 0.0


@dataclass
class FrameResult:
    state: WorkerState
    q_index: int
    nominal_prefix: int
    requested_prefix: int
    delivered_prefix: int
    wire_budget_bytes: float
    predicted_network_bytes: int
    payload_bytes: int
    network_bytes: int
    encoder_available_ms: float
    transmitter_available_ms: float
    decoder_available_ms: float
    encoder_wait_ms: float
    encode_ms: float
    transmission_wait_ms: float
    transmission_ms: float
    decoder_wait_ms: float
    decode_ms: float
    delivered: int
    psnr: float
    ms_ssim: float
    u_frame: float


@dataclass
class ActionTable:
    name: str
    image_ids: list[str]
    payload: np.ndarray
    network: np.ndarray
    psnr: np.ndarray
    msssim: np.ndarray
    encode_ms: np.ndarray
    decode_ms: np.ndarray
    q_count: int
    checkpoint_sha256: str
    cache_sha256: str

    @property
    def image_count(self) -> int:
        return self.payload.shape[0]


def combined_hash(*values: str) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def load_action_table(path: Path, dataset: str, name: str, expected_q_count: int) -> tuple[ActionTable, dict[str, Any]]:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    metadata = {key: json.loads(value) for key, value in connection.execute("SELECT key, value FROM metadata")}
    action_meta = metadata.get("action_cache")
    if not action_meta or action_meta.get("status") != "pass":
        raise RuntimeError(f"Action cache is incomplete: {path}")
    rows = list(
        connection.execute(
            """
            SELECT image_index, image_id, q_index, prefix_k, payload_bytes, network_bytes,
                   psnr, ms_ssim, encode_ms, decode_ms
            FROM action_entries WHERE dataset=? ORDER BY image_index, q_index, prefix_k
            """,
            (dataset,),
        )
    )
    connection.close()
    if not rows:
        raise RuntimeError(f"No {dataset} action entries in {path}")
    image_count = max(int(row[0]) for row in rows) + 1
    shape = (image_count, expected_q_count, 4)
    arrays = [np.full(shape, np.nan, dtype=np.float64) for _ in range(6)]
    payload, network, psnr, msssim, encode_ms, decode_ms = arrays
    image_ids = [""] * image_count
    for row in rows:
        image_index, image_id, q_index, prefix_k = int(row[0]), str(row[1]), int(row[2]), int(row[3])
        if q_index >= expected_q_count:
            raise RuntimeError("Action cache has an unexpected q_index")
        image_ids[image_index] = image_id
        index = (image_index, q_index, prefix_k - 1)
        payload[index], network[index], psnr[index], msssim[index], encode_ms[index], decode_ms[index] = map(
            float, row[4:]
        )
    if any(not image_id for image_id in image_ids) or any(not np.isfinite(array).all() for array in arrays):
        raise RuntimeError(f"Incomplete action tensor in {path}")
    if name == "baa":
        checkpoint_hash = combined_hash(
            action_meta["sender_state_dict_sha256"], action_meta["heavy_decoder_state_dict_sha256"]
        )
    else:
        checkpoint_hash = str(action_meta["codec_sha256"])
    return (
        ActionTable(
            name=name,
            image_ids=image_ids,
            payload=payload,
            network=network,
            psnr=psnr,
            msssim=msssim,
            encode_ms=encode_ms,
            decode_ms=decode_ms,
            q_count=expected_q_count,
            checkpoint_sha256=checkpoint_hash,
            cache_sha256=file_sha256(path),
        ),
        metadata,
    )


def calibrate_actions(table: ActionTable) -> list[ActionCalibration]:
    actions: list[ActionCalibration] = []
    for q_index in range(table.q_count):
        encode_values = table.encode_ms[:, q_index, 0] / 1000.0
        for prefix_k in range(1, 5):
            index = (slice(None), q_index, prefix_k - 1)
            actions.append(
                ActionCalibration(
                    q_index=q_index,
                    prefix_k=prefix_k,
                    utility=float(np.mean(table.msssim[index])),
                    payload_p90_bytes=float(np.quantile(table.payload[index], 0.90)),
                    payload_mean_bytes=float(np.mean(table.payload[index])),
                    encode_p95_s=float(np.quantile(encode_values, 0.95)),
                    encode_mean_s=float(np.mean(encode_values)),
                    decode_p95_s=float(np.quantile(table.decode_ms[index] / 1000.0, 0.95)),
                    decode_mean_s=float(np.mean(table.decode_ms[index] / 1000.0)),
                )
            )
    return actions


def network_bytes(payload: float) -> int:
    value = max(0, int(round(payload)))
    return value + ((value + 1459) // 1460) * 40 if value else 0


def simulate_action(
    table: ActionTable,
    image_index: int,
    q_index: int,
    nominal_prefix: int,
    requested_prefix: int,
    state: WorkerState,
    arrival_s: float,
    deadline_s: float,
    true_goodput_bps: float,
    wire_budget_bytes: float,
    predicted_network_bytes: int,
) -> FrameResult:
    before = replace(state)
    deadline_at = arrival_s + deadline_s
    encoder_start = max(arrival_s, state.encoder_s)
    encoder_wait_ms = (encoder_start - arrival_s) * 1000.0
    encode_ms = float(table.encode_ms[image_index, q_index, 0])
    encoder_done = encoder_start + encode_ms / 1000.0
    state.encoder_s = encoder_done
    payload_bytes = int(table.payload[image_index, q_index, requested_prefix - 1]) if requested_prefix else 0
    selected_network = int(table.network[image_index, q_index, requested_prefix - 1]) if requested_prefix else 0
    transmitter_wait_ms = 0.0
    transmission_ms = 0.0
    decoder_wait_ms = 0.0
    decode_ms = 0.0
    delivered_prefix = 0
    delivered_psnr = 0.0
    delivered_quality = 0.0
    if requested_prefix > 0 and encoder_done < deadline_at:
        transmitter_start = max(encoder_done, state.transmitter_s)
        transmitter_wait_ms = max(0.0, min(transmitter_start, deadline_at) - encoder_done) * 1000.0
        if transmitter_start < deadline_at:
            requested_finish = transmitter_start + selected_network * 8.0 / max(true_goodput_bps, 1.0)
            transmitter_done = min(requested_finish, deadline_at)
            transmission_ms = (transmitter_done - transmitter_start) * 1000.0
            state.transmitter_s = transmitter_done
            for prefix_k in range(1, requested_prefix + 1):
                prefix_network = float(table.network[image_index, q_index, prefix_k - 1])
                prefix_finish = transmitter_start + prefix_network * 8.0 / max(true_goodput_bps, 1.0)
                candidate_decode_ms = float(table.decode_ms[image_index, q_index, prefix_k - 1])
                candidate_decode_start = max(prefix_finish, state.decoder_s)
                if prefix_finish <= deadline_at and candidate_decode_start + candidate_decode_ms / 1000.0 <= deadline_at:
                    delivered_prefix = prefix_k
            if delivered_prefix:
                delivered_finish = transmitter_start + float(
                    table.network[image_index, q_index, delivered_prefix - 1]
                ) * 8.0 / max(true_goodput_bps, 1.0)
                decoder_start = max(delivered_finish, state.decoder_s)
                decoder_wait_ms = (decoder_start - delivered_finish) * 1000.0
                decode_ms = float(table.decode_ms[image_index, q_index, delivered_prefix - 1])
                state.decoder_s = decoder_start + decode_ms / 1000.0
                delivered_psnr = float(table.psnr[image_index, q_index, delivered_prefix - 1])
                delivered_quality = float(table.msssim[image_index, q_index, delivered_prefix - 1])
    delivered = int(delivered_prefix > 0)
    return FrameResult(
        state=state,
        q_index=q_index,
        nominal_prefix=nominal_prefix,
        requested_prefix=requested_prefix,
        delivered_prefix=delivered_prefix,
        wire_budget_bytes=float(max(0.0, wire_budget_bytes)),
        predicted_network_bytes=max(0, int(predicted_network_bytes)),
        payload_bytes=payload_bytes,
        network_bytes=selected_network,
        encoder_available_ms=before.encoder_s * 1000.0,
        transmitter_available_ms=before.transmitter_s * 1000.0,
        decoder_available_ms=before.decoder_s * 1000.0,
        encoder_wait_ms=encoder_wait_ms,
        encode_ms=encode_ms,
        transmission_wait_ms=transmitter_wait_ms,
        transmission_ms=transmission_ms,
        decoder_wait_ms=decoder_wait_ms,
        decode_ms=decode_ms,
        delivered=delivered,
        psnr=delivered_psnr,
        ms_ssim=delivered_quality,
        u_frame=delivered_quality if delivered else 0.0,
    )


def fixed_action_scores(
    table: ActionTable,
    traces: dict[str, np.ndarray],
    deadlines_ms: Iterable[int],
    realization_limit: int,
) -> dict[tuple[int, int], float]:
    scores: dict[tuple[int, int], float] = {}
    fps = 4.0
    for q_index in range(table.q_count):
        for prefix_k in range(1, 5):
            total = 0.0
            observations = 0
            for profile_index in range(3):
                for realization in range(realization_limit):
                    sequence = traces["image_indices"][realization]
                    goodput = traces["true_goodput"][profile_index, realization]
                    for deadline_ms in deadlines_ms:
                        state = WorkerState()
                        for frame_id, image_index in enumerate(sequence):
                            result = simulate_action(
                                table,
                                int(image_index),
                                q_index,
                                prefix_k,
                                prefix_k,
                                state,
                                frame_id / fps,
                                deadline_ms / 1000.0,
                                float(goodput[frame_id]),
                                0.0,
                                int(np.mean(table.network[:, q_index, prefix_k - 1])),
                            )
                            total += result.u_frame
                            observations += 1
            scores[(q_index, prefix_k)] = total / max(observations, 1)
    return scores


def select_fixed_actions(scores: dict[tuple[int, int], float], actions: list[ActionCalibration]) -> dict[str, tuple[int, int]]:
    by_action = {(action.q_index, action.prefix_k): action for action in actions}

    def key(item: tuple[int, int]) -> tuple[float, int, int, int, int, int]:
        action = by_action[item]
        return (
            -scores[item],
            network_bytes(action.payload_mean_bytes),
            action.prefix_k,
            -action.q_index,
            action.q_index,
            action.prefix_k,
        )

    selected: dict[str, tuple[int, int]] = {}
    for prefix_k in range(1, 5):
        candidates = [(q_index, prefix_k) for q_index in range(4)]
        selected[f"fixed-r{prefix_k / 4.0:.2f}"] = min(candidates, key=key)
    selected["best-fixed"] = min(scores, key=key)
    return selected


def select_openjpeg(
    actions: list[ActionCalibration],
    estimated_goodput: float,
    arrival_s: float,
    deadline_s: float,
    state: WorkerState,
    actual_payload: dict[int, int],
) -> tuple[int, int, float, int]:
    feasible: list[tuple[ActionCalibration, int, float]] = []
    for action in actions:
        encoder_done = max(arrival_s, state.encoder_s) + action.encode_mean_s
        transmitter_start = max(encoder_done, state.transmitter_s)
        last_transmission = arrival_s + deadline_s - action.decode_mean_s
        budget = max(0.0, estimated_goodput * (last_transmission - transmitter_start) / 8.0)
        predicted = network_bytes(action.payload_mean_bytes)
        if state.decoder_s <= last_transmission and predicted <= budget:
            feasible.append((action, predicted, budget))
    if feasible:
        action, predicted, budget = min(
            feasible,
            key=lambda item: (-item[0].utility, item[1], item[0].prefix_k, 0, 0, item[0].prefix_k),
        )
    else:
        action = actions[0]
        encoder_done = max(arrival_s, state.encoder_s) + action.encode_mean_s
        transmitter_start = max(encoder_done, state.transmitter_s)
        budget = max(
            0.0,
            estimated_goodput * (arrival_s + deadline_s - action.decode_mean_s - transmitter_start) / 8.0,
        )
        predicted = network_bytes(action.payload_mean_bytes)
    requested = 0
    for prefix_k in range(1, action.prefix_k + 1):
        if network_bytes(actual_payload[prefix_k]) <= budget:
            requested = prefix_k
    return action.prefix_k, requested, budget, predicted


def action_for_policy(
    policy: str,
    table: ActionTable,
    image_index: int,
    state: WorkerState,
    arrival_s: float,
    deadline_s: float,
    true_goodput: float,
    estimated_goodput: float,
    packet_loss_estimate: float,
    risk_controller: CoreV3BandwidthController,
    greedy_controller: CoreV3BandwidthController,
    fixed_actions: dict[str, tuple[int, int]],
    calibrations: dict[tuple[int, int], ActionCalibration],
) -> FrameResult:
    if policy in fixed_actions:
        q_index, prefix_k = fixed_actions[policy]
        predicted = network_bytes(calibrations[(q_index, prefix_k)].payload_mean_bytes)
        return simulate_action(
            table, image_index, q_index, prefix_k, prefix_k, state, arrival_s, deadline_s,
            true_goodput, 0.0, predicted,
        )
    if policy in ("BAA-controller", "capacity-greedy"):
        controller = risk_controller if policy == "BAA-controller" else greedy_controller
        decision = controller.select_preencode(
            estimated_goodput,
            packet_loss_estimate,
            arrival_s,
            deadline_s,
            WorkerAvailability(state.encoder_s, state.transmitter_s, state.decoder_s),
        )
        cumulative = {
            prefix_k: int(table.payload[image_index, decision.q_index, prefix_k - 1])
            for prefix_k in range(1, 5)
        }
        transmission = controller.select_postencode(decision, cumulative)
        if transmission.transmitted_prefix_k:
            action = calibrations[(decision.q_index, transmission.transmitted_prefix_k)]
            predicted_network = network_bytes(
                action.payload_p90_bytes if policy == "BAA-controller" else action.payload_mean_bytes
            )
        else:
            predicted_network = 0
        return simulate_action(
            table,
            image_index,
            decision.q_index,
            decision.nominal_prefix_k,
            transmission.transmitted_prefix_k,
            state,
            arrival_s,
            deadline_s,
            true_goodput,
            decision.wire_budget_bytes,
            predicted_network,
        )
    if policy == "prefix-oracle":
        candidates: list[FrameResult] = []
        for q_index in range(4):
            for prefix_k in range(1, 5):
                candidate_state = replace(state)
                candidates.append(
                    simulate_action(
                        table,
                        image_index,
                        q_index,
                        prefix_k,
                        prefix_k,
                        candidate_state,
                        arrival_s,
                        deadline_s,
                        true_goodput,
                        0.0,
                        network_bytes(calibrations[(q_index, prefix_k)].payload_mean_bytes),
                    )
                )
        chosen = max(
            candidates,
            key=lambda item: (
                item.u_frame,
                -item.network_bytes,
                -item.requested_prefix,
                item.q_index,
                -item.q_index,
                -item.requested_prefix,
            ),
        )
        state.encoder_s = chosen.state.encoder_s
        state.transmitter_s = chosen.state.transmitter_s
        state.decoder_s = chosen.state.decoder_s
        chosen.state = state
        return chosen
    raise ValueError(policy)


def summary_row(
    results: list[FrameResult],
    common: dict[str, Any],
    policy: str,
    profile: str,
    realization: int,
    deadline_ms: int,
    true_goodput: np.ndarray,
    severe_threshold: float,
    duration_s: float,
) -> dict[str, Any]:
    delivered = np.asarray([item.delivered for item in results], dtype=np.float64)
    severe = true_goodput <= severe_threshold
    waits = np.asarray(
        [item.encoder_wait_ms + item.transmission_wait_ms + item.decoder_wait_ms for item in results]
    )
    end_to_end = np.asarray(
        [
            item.encoder_wait_ms + item.encode_ms + item.transmission_wait_ms + item.transmission_ms
            + item.decoder_wait_ms + item.decode_ms
            for item in results
        ]
    )
    overshoots = [
        max(0.0, item.network_bytes - item.predicted_network_bytes) / max(1, item.predicted_network_bytes)
        if item.requested_prefix else 0.0
        for item in results
    ]
    violations = [
        float(item.requested_prefix > 0 and item.delivered_prefix < item.requested_prefix) for item in results
    ]
    actions = [(item.q_index, item.requested_prefix) for item in results]
    switches = sum(actions[index] != actions[index - 1] for index in range(1, len(actions)))
    variations = [
        abs(actions[index][1] - actions[index - 1][1]) / 4.0 for index in range(1, len(actions))
    ]
    delivered_psnr = [item.psnr for item in results if item.delivered]
    delivered_quality = [item.ms_ssim for item in results if item.delivered]
    return common | {
        "profile": profile,
        "realization_id": realization,
        "deadline_ms": deadline_ms,
        "policy": policy,
        "frames": len(results),
        "delivered_ratio": f"{float(np.mean(delivered)):.10f}",
        "severe_link_delivery": f"{float(np.mean(delivered[severe])) if np.any(severe) else 0.0:.10f}",
        "mean_u_frame": f"{float(np.mean([item.u_frame for item in results])):.10f}",
        "mean_overshoot": f"{float(np.mean(overshoots)):.10f}",
        "deadline_violation_ratio": f"{float(np.mean(violations)):.10f}",
        "encoder_wait_mean_ms": f"{float(np.mean([item.encoder_wait_ms for item in results])):.8f}",
        "transmission_wait_mean_ms": f"{float(np.mean([item.transmission_wait_ms for item in results])):.8f}",
        "decoder_wait_mean_ms": f"{float(np.mean([item.decoder_wait_ms for item in results])):.8f}",
        "total_wait_p95_ms": f"{float(np.quantile(waits, 0.95)):.8f}",
        "total_wait_p99_ms": f"{float(np.quantile(waits, 0.99)):.8f}",
        "total_wait_p999_ms": f"{float(np.quantile(waits, 0.999)):.8f}",
        "end_to_end_mean_ms": f"{float(np.mean(end_to_end)):.8f}",
        "end_to_end_p95_ms": f"{float(np.quantile(end_to_end, 0.95)):.8f}",
        "end_to_end_p99_ms": f"{float(np.quantile(end_to_end, 0.99)):.8f}",
        "unique_pixel_throughput": f"{float(np.sum(delivered) * 512 * 512 / duration_s):.8f}",
        "delivered_psnr": f"{float(np.mean(delivered_psnr)) if delivered_psnr else 0.0:.8f}",
        "delivered_ms_ssim": f"{float(np.mean(delivered_quality)) if delivered_quality else 0.0:.10f}",
        "policy_switch_rate": f"{switches / max(1, len(actions) - 1):.10f}",
        "active_ratio_variation": f"{float(np.mean(variations)) if variations else 0.0:.10f}",
    }


def main() -> None:
    args = parse_args()
    contract = load_contract()
    network = contract["network_replay"]
    if args.seed not in contract["training"]["model_seeds"]:
        raise ValueError("Seed is outside the frozen contract")
    if args.mode == "formal" and args.omit_detail:
        raise ValueError("Formal E4 requires the frozen per-frame detail artifact")
    dev_baa, baa_meta = load_action_table(args.cache, "development", "baa", 4)
    test_baa, _ = load_action_table(args.cache, "visdrone", "baa", 4)
    dev_openjpeg, openjpeg_meta = load_action_table(args.openjpeg_cache, "development", "openjpeg", 1)
    test_openjpeg, _ = load_action_table(args.openjpeg_cache, "visdrone", "openjpeg", 1)
    trace_meta = json.loads((args.traces_dir / "metadata.json").read_text(encoding="utf-8"))
    if trace_meta["contract_sha256"] != contract_sha256():
        raise RuntimeError("Traces belong to another contract")
    development = dict(np.load(args.traces_dir / "development_traces.npz"))
    test = dict(np.load(args.traces_dir / "test_traces.npz"))
    expected_dev = int(network["development_realizations_per_profile"])
    expected_test = int(network["test_realizations_per_profile"])
    if args.mode == "formal":
        dev_limit, test_limit = expected_dev, expected_test
    else:
        dev_limit = min(args.smoke_realizations, expected_dev)
        test_limit = min(args.smoke_realizations, expected_test)
    baa_actions = calibrate_actions(dev_baa)
    openjpeg_actions = calibrate_actions(dev_openjpeg)
    by_action = {(action.q_index, action.prefix_k): action for action in baa_actions}
    fixed_scores = fixed_action_scores(
        dev_baa, development, network["deadlines_ms"], dev_limit
    )
    fixed_actions = select_fixed_actions(fixed_scores, baa_actions)
    risk_controller = CoreV3BandwidthController(
        baa_actions,
        tuple(float(value) for value in network["packet_loss_calibration_bins"]),
        tuple(float(value) for value in trace_meta["goodput_ratio_bin_q10"]),
        float(trace_meta["goodput_ratio_global_q10"]),
        tuple(int(value) for value in trace_meta["goodput_ratio_bin_counts"]),
        risk_aware=True,
    )
    greedy_controller = CoreV3BandwidthController(
        baa_actions,
        tuple(float(value) for value in network["packet_loss_calibration_bins"]),
        tuple(float(value) for value in trace_meta["goodput_ratio_bin_q10"]),
        float(trace_meta["goodput_ratio_global_q10"]),
        tuple(int(value) for value in trace_meta["goodput_ratio_bin_counts"]),
        risk_aware=False,
    )
    calibration_payload = {
        "contract_version": contract["contract_version"],
        "contract_sha256": contract_sha256(),
        "seed": args.seed,
        "baa_cache_sha256": dev_baa.cache_sha256,
        "openjpeg_cache_sha256": dev_openjpeg.cache_sha256,
        "trace_config_sha256": trace_meta["trace_config_sha256"],
        "fixed_scores": {f"q{q}_k{k}": score for (q, k), score in fixed_scores.items()},
        "fixed_actions": {key: list(value) for key, value in fixed_actions.items()},
        "baa_actions": [action.__dict__ for action in baa_actions],
        "openjpeg_actions": [action.__dict__ for action in openjpeg_actions],
    }
    calibration_sha = hashlib.sha256(
        json.dumps(calibration_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    calibration_payload["calibration_sha256"] = calibration_sha
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "calibration.json").write_text(
        json.dumps(calibration_payload, indent=2) + "\n", encoding="utf-8"
    )
    code_hash = files_sha256(
        [
            ROOT / "models" / "core_v3_controller.py",
            ROOT / "scripts" / "generate_core_v3_traces.py",
            Path(__file__).resolve(),
        ]
    )
    manifest_hash = file_sha256(ROOT / "data" / "core_v3_visdrone_test.csv")
    common = {
        "contract_version": contract["contract_version"],
        "contract_sha256": contract_sha256(),
        "code_hash": code_hash,
        "checkpoint_sha256": test_baa.checkpoint_sha256,
        "manifest_sha256": manifest_hash,
        "model_seed": args.seed,
        "calibration_sha256": calibration_sha,
        "trace_config_sha256": trace_meta["trace_config_sha256"],
    }
    detail_path = args.output_dir / "core_v3_bandwidth_detail.csv"
    summary_path = args.output_dir / "core_v3_bandwidth_summary.csv"
    detail_handle = None if args.omit_detail else detail_path.open("w", encoding="utf-8", newline="")
    detail_writer = None
    if detail_handle is not None:
        detail_writer = csv.DictWriter(detail_handle, fieldnames=DETAIL_FIELDS)
        detail_writer.writeheader()
    summary_handle = summary_path.open("w", encoding="utf-8", newline="")
    summary_writer = csv.DictWriter(summary_handle, fieldnames=SUMMARY_FIELDS)
    summary_writer.writeheader()
    policies = [
        "BAA-controller", "capacity-greedy", "fixed-r0.25", "fixed-r0.50", "fixed-r0.75",
        "fixed-r1.00", "best-fixed", "prefix-oracle", "openjpeg-capacity-greedy",
    ]
    fps = float(network["frames_per_second"])
    frames = int(network["frames_per_realization"])
    detail_rows = 0
    summary_rows = 0
    for profile_index, profile in enumerate(PROFILE_NAMES):
        severe_threshold = float(trace_meta["severe_link_threshold_bps"][profile])
        for realization in range(test_limit):
            sequence = test["image_indices"][realization]
            for deadline_ms in network["deadlines_ms"]:
                for policy in policies:
                    state = WorkerState()
                    results: list[FrameResult] = []
                    for frame_id, image_index_value in enumerate(sequence):
                        image_index = int(image_index_value)
                        arrival_s = frame_id / fps
                        true_goodput = float(test["true_goodput"][profile_index, realization, frame_id])
                        estimated_goodput = float(test["estimated_goodput"][profile_index, realization, frame_id])
                        packet_loss = float(test["packet_loss"][profile_index, realization, frame_id])
                        packet_loss_estimate = float(
                            test["packet_loss_estimate"][profile_index, realization, frame_id]
                        )
                        if policy == "openjpeg-capacity-greedy":
                            actual_payload = {
                                prefix_k: int(test_openjpeg.payload[image_index, 0, prefix_k - 1])
                                for prefix_k in range(1, 5)
                            }
                            nominal, requested, budget, predicted = select_openjpeg(
                                openjpeg_actions,
                                estimated_goodput,
                                arrival_s,
                                deadline_ms / 1000.0,
                                state,
                                actual_payload,
                            )
                            predicted = (
                                network_bytes(openjpeg_actions[requested - 1].payload_mean_bytes)
                                if requested
                                else 0
                            )
                            result = simulate_action(
                                test_openjpeg,
                                image_index,
                                0,
                                nominal,
                                requested,
                                state,
                                arrival_s,
                                deadline_ms / 1000.0,
                                true_goodput,
                                budget,
                                predicted,
                            )
                            row_common = common | {"checkpoint_sha256": test_openjpeg.checkpoint_sha256}
                        else:
                            result = action_for_policy(
                                policy,
                                test_baa,
                                image_index,
                                state,
                                arrival_s,
                                deadline_ms / 1000.0,
                                true_goodput,
                                estimated_goodput,
                                packet_loss_estimate,
                                risk_controller,
                                greedy_controller,
                                fixed_actions,
                                by_action,
                            )
                            row_common = common
                        results.append(result)
                        if detail_writer is not None:
                            detail_writer.writerow(
                                row_common
                                | {
                                    "profile": profile,
                                    "realization_id": realization,
                                    "deadline_ms": deadline_ms,
                                    "frame_id": frame_id,
                                    "policy": policy,
                                    "image_id": (
                                        test_openjpeg.image_ids[image_index]
                                        if policy == "openjpeg-capacity-greedy"
                                        else test_baa.image_ids[image_index]
                                    ),
                                    "true_goodput": f"{true_goodput:.6f}",
                                    "estimated_goodput": f"{estimated_goodput:.6f}",
                                    "packet_loss": f"{packet_loss:.10f}",
                                    "encoder_available_ms": f"{result.encoder_available_ms:.8f}",
                                    "transmitter_available_ms": f"{result.transmitter_available_ms:.8f}",
                                    "decoder_available_ms": f"{result.decoder_available_ms:.8f}",
                                    "wire_budget_bytes": f"{result.wire_budget_bytes:.8f}",
                                    "q_index": result.q_index,
                                    "nominal_prefix": result.nominal_prefix,
                                    "requested_prefix": result.requested_prefix,
                                    "delivered_prefix": result.delivered_prefix,
                                    "payload_bytes": result.payload_bytes,
                                    "network_bytes": result.network_bytes,
                                    "encoder_wait_ms": f"{result.encoder_wait_ms:.8f}",
                                    "encode_ms": f"{result.encode_ms:.8f}",
                                    "transmission_wait_ms": f"{result.transmission_wait_ms:.8f}",
                                    "transmission_ms": f"{result.transmission_ms:.8f}",
                                    "decoder_wait_ms": f"{result.decoder_wait_ms:.8f}",
                                    "decode_ms": f"{result.decode_ms:.8f}",
                                    "delivered": result.delivered,
                                    "ms_ssim": f"{result.ms_ssim:.10f}",
                                    "u_frame": f"{result.u_frame:.10f}",
                                }
                            )
                            detail_rows += 1
                    summary_writer.writerow(
                        summary_row(
                            results,
                            row_common,
                            policy,
                            profile,
                            realization,
                            int(deadline_ms),
                            test["true_goodput"][profile_index, realization],
                            severe_threshold,
                            frames / fps,
                        )
                    )
                    summary_rows += 1
            if (realization + 1) % 10 == 0:
                summary_handle.flush()
                if detail_handle is not None:
                    detail_handle.flush()
                print(f"E4: {profile} {realization + 1}/{test_limit}", flush=True)
    summary_handle.close()
    if detail_handle is not None:
        detail_handle.close()
    expected_summary = 3 * test_limit * len(network["deadlines_ms"]) * len(policies)
    expected_detail = expected_summary * frames if not args.omit_detail else 0
    if summary_rows != expected_summary or detail_rows != expected_detail:
        raise RuntimeError("Bandwidth replay row count mismatch")
    metadata = {
        "status": "pass",
        "contract_version": contract["contract_version"],
        "contract_sha256": contract_sha256(),
        "seed": args.seed,
        "mode": args.mode,
        "development_realizations": dev_limit,
        "test_realizations": test_limit,
        "policies": policies,
        "summary_rows": summary_rows,
        "detail_rows": detail_rows,
        "fixed_actions": {key: list(value) for key, value in fixed_actions.items()},
        "calibration_sha256": calibration_sha,
        "summary_sha256": file_sha256(summary_path),
        "detail_sha256": file_sha256(detail_path) if detail_handle is not None else None,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"PASS: bandwidth seed={args.seed} summary={summary_rows} detail={detail_rows}")


if __name__ == "__main__":
    main()
