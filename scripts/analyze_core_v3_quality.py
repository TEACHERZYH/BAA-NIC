"""Merge core-v3 E1-E3 outputs and apply the frozen hierarchical gates."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import re
import shutil
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_v3_contract import contract_sha256, file_sha256, files_sha256, load_contract
from core_v3_data import read_core_manifest


SEEDS = (17, 29, 43)
ORDERINGS = tuple("-".join(map(str, values)) for values in itertools.permutations((1, 2, 3, 4)))
ORDERED = "1-2-3-4"
REVERSE = "4-3-2-1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--tables-dir", type=Path, default=ROOT / "tables")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "core_v3_analysis")
    return parser.parse_args()


def merge_csv(inputs: list[Path], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    header: str | None = None
    with output.open("w", encoding="utf-8", newline="") as target:
        for path in inputs:
            with path.open("r", encoding="utf-8", newline="") as source:
                current_header = source.readline()
                if header is None:
                    header = current_header
                    target.write(current_header)
                elif current_header != header:
                    raise RuntimeError(f"CSV header mismatch: {path}")
                for line in source:
                    target.write(line)
                    rows += 1
    return rows


def bootstrap(values: np.ndarray, iterations: int, seed: int) -> dict[str, float]:
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError("Bootstrap expects a finite seed x unit matrix with at least two seeds")
    rng = np.random.Generator(np.random.PCG64(seed))
    distribution = np.empty(iterations, dtype=np.float64)
    seed_count = values.shape[0]
    units = values.shape[1]
    for index in range(iterations):
        selected_seeds = rng.integers(0, seed_count, size=seed_count)
        seed_means = []
        for selected_seed in selected_seeds:
            selected_units = rng.integers(0, units, size=units)
            seed_means.append(float(np.mean(values[selected_seed, selected_units])))
        distribution[index] = float(np.mean(seed_means))
    point = float(np.mean(np.mean(values, axis=1)))
    lower, upper = np.quantile(distribution, (0.025, 0.975))
    left = (np.count_nonzero(distribution <= 0.0) + 1) / (iterations + 1)
    right = (np.count_nonzero(distribution >= 0.0) + 1) / (iterations + 1)
    return {
        "effect": point,
        "ci_low": float(lower),
        "ci_high": float(upper),
        "raw_p": min(1.0, 2.0 * min(left, right)),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def cluster_means(values: np.ndarray, clusters: list[str]) -> np.ndarray:
    names = sorted(set(clusters))
    return np.stack(
        [
            np.asarray(
                [np.mean(values[seed_index, np.asarray(clusters) == name]) for name in names],
                dtype=np.float64,
            )
            for seed_index in range(values.shape[0])
        ]
    )


def read_manifests() -> tuple[dict[str, list[str]], dict[str, dict[str, int]], list[str]]:
    paths = {
        "visdrone": ROOT / "data" / "core_v3_visdrone_test.csv",
        "uavid": ROOT / "data" / "core_v3_uavid_test.csv",
    }
    ids: dict[str, list[str]] = {}
    indexes: dict[str, dict[str, int]] = {}
    uavid_clusters: list[str] = []
    for dataset, path in paths.items():
        rows = read_core_manifest(path)
        ids[dataset] = [row["tile_id"] for row in rows]
        indexes[dataset] = {image_id: index for index, image_id in enumerate(ids[dataset])}
        if dataset == "uavid":
            uavid_clusters = [row["sequence_id"] for row in rows]
    return ids, indexes, uavid_clusters


def analyze_e2(
    path: Path,
    ids: dict[str, list[str]],
    indexes: dict[str, dict[str, int]],
    uavid_clusters: list[str],
    data_dir: Path,
    tables_dir: Path,
    iterations: int,
) -> dict[str, Any]:
    arrays = {
        dataset: {
            decoder: np.full((3, len(ids[dataset]), 4, 4), np.nan, dtype=np.float64)
            for decoder in ("heavy", "light")
        }
        for dataset in ids
    }
    byte_records: dict[tuple[str, int, str, int, int], tuple[int, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            dataset = row["dataset"]
            seed_index = SEEDS.index(int(row["seed"]))
            image_index = indexes[dataset][row["image_id"]]
            q_index = int(row["q_index"])
            prefix_index = int(row["prefix_k"]) - 1
            decoder = row["decoder"]
            arrays[dataset][decoder][seed_index, image_index, q_index, prefix_index] = float(row["ms_ssim"])
            key = (dataset, int(row["seed"]), row["image_id"], q_index, prefix_index)
            value = (int(row["bytes"]), row["bitstream_sha256"])
            prior = byte_records.setdefault(key, value)
            if prior != value:
                raise RuntimeError(f"Heavy/light bitstreams differ at {key}")
    if any(not np.isfinite(value).all() for dataset in arrays.values() for value in dataset.values()):
        raise RuntimeError("E2 matrix is incomplete")
    per_image_path = data_dir / "core_v3_decoder_asymmetry_per_image.csv"
    per_image_fields = ["dataset", "seed", "image_id", "prefix_k", "mean_q_ms_ssim_delta"]
    with per_image_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_image_fields)
        writer.writeheader()
        for dataset in ids:
            differences = arrays[dataset]["heavy"] - arrays[dataset]["light"]
            for seed_index, seed in enumerate(SEEDS):
                for image_index, image_id in enumerate(ids[dataset]):
                    for prefix_index in range(4):
                        writer.writerow(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "image_id": image_id,
                                "prefix_k": prefix_index + 1,
                                "mean_q_ms_ssim_delta": f"{float(np.mean(differences[seed_index, image_index, :, prefix_index])):.10f}",
                            }
                        )
    visdrone_stats = []
    visdrone_delta = arrays["visdrone"]["heavy"] - arrays["visdrone"]["light"]
    for prefix_index in range(4):
        values = np.mean(visdrone_delta[:, :, :, prefix_index], axis=2)
        result = bootstrap(values, iterations, 3200 + prefix_index)
        result["prefix_k"] = prefix_index + 1
        result["seed_effects"] = [float(np.mean(values[index])) for index in range(3)]
        visdrone_stats.append(result)
    adjusted = holm_adjust([item["raw_p"] for item in visdrone_stats])
    for item, value in zip(visdrone_stats, adjusted):
        item["holm_p"] = value
        item["gate_pass"] = (
            item["ci_low"] > 0.0 and value < 0.05 and all(effect >= 0.0 for effect in item["seed_effects"])
        )
    uavid_delta = arrays["uavid"]["heavy"] - arrays["uavid"]["light"]
    uavid_per_image = np.mean(uavid_delta, axis=(2, 3))
    uavid_result = bootstrap(cluster_means(uavid_per_image, uavid_clusters), iterations, 3210)
    e2_gate = len([item for item in visdrone_stats if item["gate_pass"]]) >= 3 and uavid_result["effect"] >= 0.0
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_path = tables_dir / "core_v3_decoder_asymmetry.csv"
    fields = ["dataset", "prefix_k", "effect", "ci_low", "ci_high", "raw_p", "holm_p", "seed_effects", "gate_pass"]
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in visdrone_stats:
            writer.writerow(
                {
                    "dataset": "visdrone",
                    "prefix_k": item["prefix_k"],
                    "effect": f"{item['effect']:.10f}",
                    "ci_low": f"{item['ci_low']:.10f}",
                    "ci_high": f"{item['ci_high']:.10f}",
                    "raw_p": f"{item['raw_p']:.10f}",
                    "holm_p": f"{item['holm_p']:.10f}",
                    "seed_effects": ";".join(f"{value:.10f}" for value in item["seed_effects"]),
                    "gate_pass": int(item["gate_pass"]),
                }
            )
        writer.writerow(
            {
                "dataset": "uavid",
                "prefix_k": "aggregate",
                "effect": f"{uavid_result['effect']:.10f}",
                "ci_low": f"{uavid_result['ci_low']:.10f}",
                "ci_high": f"{uavid_result['ci_high']:.10f}",
                "raw_p": f"{uavid_result['raw_p']:.10f}",
                "holm_p": f"{uavid_result['raw_p']:.10f}",
                "seed_effects": ";".join(f"{float(np.mean(uavid_per_image[index])):.10f}" for index in range(3)),
                "gate_pass": int(uavid_result["effect"] >= 0.0),
            }
        )
    return {
        "gate_pass": e2_gate,
        "bitstream_byte_equality": True,
        "visdrone_prefixes_passed": sum(item["gate_pass"] for item in visdrone_stats),
        "visdrone": visdrone_stats,
        "uavid": uavid_result,
        "bytes": byte_records,
    }


def parse_msssim_violations(value: str) -> int:
    match = re.search(r"msssim:(\d+)", value)
    if not match:
        raise ValueError(value)
    return int(match.group(1))


def analyze_e3(
    path: Path,
    ids: dict[str, list[str]],
    indexes: dict[str, dict[str, int]],
    uavid_clusters: list[str],
    e2: dict[str, Any],
    data_dir: Path,
    iterations: int,
) -> dict[str, Any]:
    order_index = {value: index for index, value in enumerate(ORDERINGS)}
    arrays = {
        dataset: {
            "auc": np.full((3, len(ids[dataset]), 4, 24), np.nan, dtype=np.float64),
            "quality": np.full((3, len(ids[dataset]), 4, 24, 4), np.nan, dtype=np.float64),
            "violations": np.zeros((3, len(ids[dataset]), 4), dtype=np.int16),
        }
        for dataset in ids
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            dataset = row["dataset"]
            seed_index = SEEDS.index(int(row["seed"]))
            image_index = indexes[dataset][row["image_id"]]
            q_index = int(row["q_index"])
            ordering_index = order_index[row["ordering_id"]]
            budget_index = (int(row["budget_percent"]) // 25) - 1
            arrays[dataset]["auc"][seed_index, image_index, q_index, ordering_index] = float(row["auc"])
            arrays[dataset]["quality"][seed_index, image_index, q_index, ordering_index, budget_index] = float(
                row["ms_ssim"]
            )
            if row["ordering_id"] == ORDERED and budget_index == 3:
                arrays[dataset]["violations"][seed_index, image_index, q_index] = parse_msssim_violations(
                    row["violation_flags"]
                )
    if any(
        not np.isfinite(dataset[key]).all()
        for dataset in arrays.values()
        for key in ("auc", "quality")
    ):
        raise RuntimeError("E3 matrix is incomplete")
    ordered_index = order_index[ORDERED]
    reverse_index = order_index[REVERSE]
    results: dict[str, Any] = {}
    output_rows: list[dict[str, Any]] = []
    for dataset in ("visdrone", "uavid"):
        auc = arrays[dataset]["auc"]
        ordered_auc = auc[..., ordered_index]
        alternative_auc = np.delete(auc, ordered_index, axis=3)
        delta_alt = np.mean(ordered_auc - np.median(alternative_auc, axis=3), axis=2)
        delta_reverse = np.mean(ordered_auc - auc[..., reverse_index], axis=2)
        if dataset == "uavid":
            alt_values = cluster_means(delta_alt, uavid_clusters)
            reverse_values = cluster_means(delta_reverse, uavid_clusters)
        else:
            alt_values = delta_alt
            reverse_values = delta_reverse
        alt_result = bootstrap(alt_values, iterations, 3300 if dataset == "visdrone" else 3310)
        reverse_result = bootstrap(reverse_values, iterations, 3301 if dataset == "visdrone" else 3311)
        ranks = []
        for seed_index in range(3):
            order_means = np.mean(auc[seed_index], axis=(0, 1))
            ranks.append(1 + int(np.count_nonzero(order_means > order_means[ordered_index])))
        results[dataset] = {"alternative": alt_result, "reverse": reverse_result, "ordered_ranks": ranks}
        output_rows.extend(
            [
                {"dataset": dataset, "metric": "ordered_minus_alternative_auc", **alt_result},
                {"dataset": dataset, "metric": "ordered_minus_reverse_auc", **reverse_result},
            ]
        )
    quality = arrays["visdrone"]["quality"]
    budget_stats = []
    for budget_index, budget in enumerate((25, 50, 75, 100)):
        ordered_quality = quality[..., ordered_index, budget_index]
        alternatives = np.delete(quality[..., budget_index], ordered_index, axis=3)
        values = np.mean(ordered_quality - np.median(alternatives, axis=3), axis=2)
        result = bootstrap(values, iterations, 3320 + budget_index)
        result["budget_percent"] = budget
        budget_stats.append(result)
    adjusted = holm_adjust([item["raw_p"] for item in budget_stats])
    for item, value in zip(budget_stats, adjusted):
        item["holm_p"] = value
        item["gate_pass"] = item["ci_low"] > 0.0 and value < 0.05
        output_rows.append(
            {
                "dataset": "visdrone",
                "metric": f"budget_{item['budget_percent']}_ordered_minus_alternative",
                **item,
            }
        )
    violations = int(np.sum(arrays["visdrone"]["violations"]))
    transitions = 3 * len(ids["visdrone"]) * 4 * 3
    violation_rate = violations / transitions
    byte_monotonic = True
    grouped_bytes: dict[tuple[str, int, str, int], list[int]] = {}
    for (dataset, seed, image_id, q_index, prefix_index), (byte_count, _) in e2["bytes"].items():
        if dataset != "visdrone":
            continue
        grouped_bytes.setdefault((dataset, seed, image_id, q_index), [0, 0, 0, 0])[prefix_index] = byte_count
    for values in grouped_bytes.values():
        byte_monotonic &= all(values[index] < values[index + 1] for index in range(3))
    vis = results["visdrone"]
    uav = results["uavid"]
    gate = (
        byte_monotonic
        and violation_rate <= 0.01
        and vis["alternative"]["ci_low"] > 0.0
        and sum(rank <= 6 for rank in vis["ordered_ranks"]) >= 2
        and sum(item["gate_pass"] for item in budget_stats) >= 3
        and vis["reverse"]["ci_low"] >= 0.0
        and uav["alternative"]["effect"] >= 0.0
        and uav["reverse"]["effect"] >= 0.0
    )
    monotonicity_path = data_dir / "core_v3_progressive_monotonicity.csv"
    fields = ["dataset", "metric", "effect", "ci_low", "ci_high", "raw_p", "holm_p", "gate_pass"]
    with monotonicity_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "metric": row["metric"],
                    "effect": f"{row['effect']:.10f}",
                    "ci_low": f"{row['ci_low']:.10f}",
                    "ci_high": f"{row['ci_high']:.10f}",
                    "raw_p": f"{row['raw_p']:.10f}",
                    "holm_p": f"{row.get('holm_p', row['raw_p']):.10f}",
                    "gate_pass": int(row.get("gate_pass", row["effect"] >= 0.0)),
                }
            )
        writer.writerow(
            {
                "dataset": "visdrone",
                "metric": "ordered_msssim_monotonic_violation_rate",
                "effect": f"{violation_rate:.10f}",
                "ci_low": f"{violation_rate:.10f}",
                "ci_high": f"{violation_rate:.10f}",
                "raw_p": "1.0000000000",
                "holm_p": "1.0000000000",
                "gate_pass": int(violation_rate <= 0.01),
            }
        )
    return {
        "gate_pass": gate,
        "byte_strictly_monotonic": bool(byte_monotonic),
        "msssim_violation_rate": violation_rate,
        "visdrone": vis,
        "uavid": uav,
        "budget_tests": budget_stats,
    }


def analyze_e1(inputs: list[Path], tables_dir: Path) -> dict[str, Any]:
    gates = []
    latency: dict[tuple[str, str, str, str], list[float]] = {}
    static: dict[tuple[int, str, str], dict[str, float]] = {}
    for seed, path in zip(SEEDS, inputs):
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                base_model = re.sub(r"_seed\d+$", "", row["model"])
                key = (seed, base_model, row["component"])
                static.setdefault(
                    key,
                    {
                        "params": float(row["params"]),
                        "macs_per_pixel": float(row["macs_per_pixel"]),
                        "gflops_512": float(row["gflops_512"]),
                        "checkpoint_bytes": float(row["checkpoint_bytes"]),
                    },
                )
                latency.setdefault((base_model, row["component"], row["side"], row["device"]), []).append(
                    float(row["latency_ms"])
                )
        baa_analysis = static[(seed, "baa_nic_depthwise", "analysis_transform")]
        dense_analysis = static[(seed, "dense_conv_control", "analysis_transform")]
        baa_sender = static[(seed, "baa_nic_depthwise", "full_sender")]
        anchor_models = sorted(
            model for current_seed, model, component in static if current_seed == seed and component == "full_sender"
            and model.startswith("bmshj2018_hyperprior_q")
        )
        if len(anchor_models) != 1:
            raise RuntimeError("Expected one selected CompressAI anchor per seed")
        anchor_sender = static[(seed, anchor_models[0], "full_sender")]
        components = {
            component for current_seed, model, component in static if current_seed == seed and model == "baa_nic_depthwise"
        }
        gates.append(
            {
                "seed": seed,
                "analysis_params_reduction": 1.0 - baa_analysis["params"] / dense_analysis["params"],
                "analysis_macs_reduction": 1.0 - baa_analysis["macs_per_pixel"] / dense_analysis["macs_per_pixel"],
                "sender_params_below_anchor": baa_sender["params"] < anchor_sender["params"],
                "sender_macs_below_anchor": baa_sender["macs_per_pixel"] < anchor_sender["macs_per_pixel"],
                "entropy_and_packing_timed": {"entropy_encode", "packing", "full_sender"}.issubset(components),
            }
        )
    table_path = tables_dir / "core_v3_lightweight_asymmetry.csv"
    fields = [
        "model", "component", "side", "device", "samples", "params", "checkpoint_bytes",
        "macs_per_pixel", "gflops_512", "latency_mean_ms", "latency_std_ms", "latency_p50_ms",
        "latency_p95_ms", "latency_ci_low_ms", "latency_ci_high_ms",
    ]
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, values in sorted(latency.items()):
            model, component, side, device = key
            array = np.asarray(values, dtype=np.float64)
            matching = [value for (seed, current_model, current_component), value in static.items() if current_model == model and current_component == component]
            reference = matching[0]
            ci = 1.96 * float(np.std(array, ddof=1)) / math.sqrt(len(array)) if len(array) > 1 else 0.0
            writer.writerow(
                {
                    "model": model,
                    "component": component,
                    "side": side,
                    "device": device,
                    "samples": len(array),
                    "params": int(reference["params"]),
                    "checkpoint_bytes": int(reference["checkpoint_bytes"]),
                    "macs_per_pixel": f"{reference['macs_per_pixel']:.8f}",
                    "gflops_512": f"{reference['gflops_512']:.8f}",
                    "latency_mean_ms": f"{float(np.mean(array)):.8f}",
                    "latency_std_ms": f"{float(np.std(array, ddof=1)):.8f}",
                    "latency_p50_ms": f"{float(np.quantile(array, 0.50)):.8f}",
                    "latency_p95_ms": f"{float(np.quantile(array, 0.95)):.8f}",
                    "latency_ci_low_ms": f"{float(np.mean(array) - ci):.8f}",
                    "latency_ci_high_ms": f"{float(np.mean(array) + ci):.8f}",
                }
            )
    gate = all(
        item["analysis_params_reduction"] >= 0.5
        and item["analysis_macs_reduction"] >= 0.5
        and item["sender_params_below_anchor"]
        and item["sender_macs_below_anchor"]
        and item["entropy_and_packing_timed"]
        for item in gates
    )
    return {"gate_pass": gate, "seeds": gates}


def main() -> None:
    args = parse_args()
    contract = load_contract()
    iterations = int(contract["statistics"]["bootstrap_iterations"])
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    complexity_inputs = [
        args.runs_root / "core_v3_complexity" / f"seed_{seed}" / "core_v3_complexity_breakdown.csv"
        for seed in SEEDS
    ]
    asymmetry_inputs = [
        args.runs_root / "core_v3_quality" / f"seed_{seed}" / "core_v3_decoder_asymmetry.csv"
        for seed in SEEDS
    ]
    ordering_inputs = [
        args.runs_root / "core_v3_quality" / f"seed_{seed}" / "core_v3_progressive_ordering.csv"
        for seed in SEEDS
    ]
    roundtrip_inputs = [
        args.runs_root / "core_v3_g0b" / f"seed_{seed}" / "core_v3_bitstream_roundtrip.csv"
        for seed in SEEDS
    ]
    operating_inputs = [
        args.runs_root / "core_v3_g0b" / f"seed_{seed}" / "core_v3_operating_points.csv"
        for seed in SEEDS
    ]
    for path in complexity_inputs + asymmetry_inputs + ordering_inputs + roundtrip_inputs + operating_inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    merged = {
        "complexity": args.data_dir / "core_v3_complexity_breakdown.csv",
        "asymmetry": args.data_dir / "core_v3_decoder_asymmetry.csv",
        "ordering": args.data_dir / "core_v3_progressive_ordering.csv",
        "roundtrip": args.data_dir / "core_v3_bitstream_roundtrip.csv",
        "operating": args.data_dir / "core_v3_operating_points.csv",
    }
    row_counts = {
        "complexity": merge_csv(complexity_inputs, merged["complexity"]),
        "asymmetry": merge_csv(asymmetry_inputs, merged["asymmetry"]),
        "ordering": merge_csv(ordering_inputs, merged["ordering"]),
        "roundtrip": merge_csv(roundtrip_inputs, merged["roundtrip"]),
        "operating": merge_csv(operating_inputs, merged["operating"]),
    }
    ids, indexes, uavid_clusters = read_manifests()
    e1 = analyze_e1(complexity_inputs, args.tables_dir)
    e2 = analyze_e2(
        merged["asymmetry"], ids, indexes, uavid_clusters, args.data_dir, args.tables_dir, iterations
    )
    e3 = analyze_e3(
        merged["ordering"], ids, indexes, uavid_clusters, e2, args.data_dir, iterations
    )
    e2.pop("bytes")
    result = {
        "status": "pass",
        "contract_version": contract["contract_version"],
        "contract_sha256": contract_sha256(),
        "code_hash": files_sha256([Path(__file__).resolve()]),
        "row_counts": row_counts,
        "E1": e1,
        "E2": e2,
        "E3": e3,
        "all_gates_pass": e1["gate_pass"] and e2["gate_pass"] and e3["gate_pass"],
        "outputs": {key: {"path": str(path), "sha256": file_sha256(path)} for key, path in merged.items()},
    }
    (args.output_dir / "quality_gate_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"PASS: core-v3 analysis E1={e1['gate_pass']} E2={e2['gate_pass']} E3={e3['gate_pass']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
