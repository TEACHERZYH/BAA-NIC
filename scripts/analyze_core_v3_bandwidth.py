"""Merge core-v3 E4 artifacts and evaluate the frozen G4 statistical gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_v3_contract import contract_sha256, file_sha256, files_sha256, load_contract


SEEDS = (17, 29, 43)
PROFILES = ("smooth_ar", "drop_recovery", "burst_loss")
STAT_FIELDS = [
    "contract_version", "contract_sha256", "code_hash", "checkpoint_sha256", "manifest_sha256",
    "profile", "deadline_ms", "policy", "comparator", "metric", "effect_size", "ci_low",
    "ci_high", "raw_p", "holm_p", "noninferiority_margin", "gate_pass",
]
METRICS = (
    "mean_u_frame", "total_wait_p99_ms", "mean_overshoot", "deadline_violation_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "core_v3_analysis")
    return parser.parse_args()


def merge_csv(inputs: list[Path], output: Path, count_rows: bool) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    header: str | None = None
    rows = 0
    with output.open("w", encoding="utf-8", newline="") as target:
        for path in inputs:
            with path.open("r", encoding="utf-8", newline="") as source:
                current = source.readline()
                if header is None:
                    header = current
                    target.write(current)
                elif current != header:
                    raise RuntimeError(f"CSV header mismatch: {path}")
                if count_rows:
                    for line in source:
                        target.write(line)
                        rows += 1
                else:
                    shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
    return rows


def bootstrap_difference(values: np.ndarray, iterations: int, seed: int) -> tuple[dict[str, float], np.ndarray]:
    if values.ndim != 2 or values.shape[0] != 3 or not np.isfinite(values).all():
        raise ValueError("Expected a finite 3 x N paired-difference matrix")
    rng = np.random.Generator(np.random.PCG64(seed))
    units = values.shape[1]
    distribution = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        selected_seeds = rng.integers(0, 3, size=3)
        means = []
        for selected_seed in selected_seeds:
            selected_units = rng.integers(0, units, size=units)
            means.append(float(np.mean(values[selected_seed, selected_units])))
        distribution[index] = float(np.mean(means))
    lower, upper = np.quantile(distribution, (0.025, 0.975))
    left = (np.count_nonzero(distribution <= 0.0) + 1) / (iterations + 1)
    right = (np.count_nonzero(distribution >= 0.0) + 1) / (iterations + 1)
    return (
        {
            "effect_size": float(np.mean(np.mean(values, axis=1))),
            "ci_low": float(lower),
            "ci_high": float(upper),
            "raw_p": min(1.0, 2.0 * min(left, right)),
        },
        distribution,
    )


def holm_adjust(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    adjusted = [1.0] * len(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted


def load_summary(path: Path, expected_realizations: int) -> dict[tuple[str, int, str, str], np.ndarray]:
    output: dict[tuple[str, int, str, str], np.ndarray] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key_base = (row["profile"], int(row["deadline_ms"]), row["policy"])
            realization = int(row["realization_id"])
            for metric in METRICS:
                key = key_base + (metric,)
                array = output.setdefault(key, np.full(expected_realizations, np.nan, dtype=np.float64))
                array[realization] = float(row[metric])
    if any(not np.isfinite(array).all() for array in output.values()):
        raise RuntimeError(f"Incomplete summary matrix: {path}")
    return output


def matrix(
    summaries: list[dict[tuple[str, int, str, str], np.ndarray]],
    profile: str,
    deadline: int,
    policy: str,
    metric: str,
) -> np.ndarray:
    return np.stack([summary[(profile, deadline, policy, metric)] for summary in summaries])


def aggregate_matrix(
    summaries: list[dict[tuple[str, int, str, str], np.ndarray]],
    policy: str,
    metric: str,
    deadlines: list[int],
) -> np.ndarray:
    return np.mean(
        np.stack(
            [matrix(summaries, profile, deadline, policy, metric) for profile in PROFILES for deadline in deadlines],
            axis=0,
        ),
        axis=0,
    )


def combined_checkpoint_hash(metadata_paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in metadata_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        calibration = json.loads((path.parent / "calibration.json").read_text(encoding="utf-8"))
        digest.update(bytes.fromhex(calibration["baa_cache_sha256"]))
        digest.update(bytes.fromhex(payload["calibration_sha256"]))
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    contract = load_contract()
    network = contract["network_replay"]
    iterations = int(contract["statistics"]["bootstrap_iterations"])
    expected = int(network["test_realizations_per_profile"])
    deadlines = [int(value) for value in network["deadlines_ms"]]
    summary_inputs = [
        args.runs_root / "core_v3_bandwidth" / f"seed_{seed}" / "core_v3_bandwidth_summary.csv"
        for seed in SEEDS
    ]
    detail_inputs = [
        args.runs_root / "core_v3_bandwidth" / f"seed_{seed}" / "core_v3_bandwidth_detail.csv"
        for seed in SEEDS
    ]
    metadata_paths = [path.parent / "metadata.json" for path in summary_inputs]
    for path in summary_inputs + detail_inputs + metadata_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.data_dir / "core_v3_bandwidth_summary.csv"
    detail_path = args.data_dir / "core_v3_bandwidth_detail.csv"
    summary_rows = merge_csv(summary_inputs, summary_path, count_rows=True)
    merge_csv(detail_inputs, detail_path, count_rows=False)
    detail_rows = sum(json.loads(path.read_text(encoding="utf-8"))["detail_rows"] for path in metadata_paths)
    summaries = [load_summary(path, expected) for path in summary_inputs]
    common = {
        "contract_version": contract["contract_version"],
        "contract_sha256": contract_sha256(),
        "code_hash": files_sha256([Path(__file__).resolve()]),
        "checkpoint_sha256": combined_checkpoint_hash(metadata_paths),
        "manifest_sha256": file_sha256(ROOT / "data" / "core_v3_visdrone_test.csv"),
    }
    rows: list[dict[str, Any]] = []
    u_rows: dict[tuple[str, int, str], dict[str, Any]] = {}
    for comparator_index, comparator in enumerate(("best-fixed", "capacity-greedy")):
        for deadline in deadlines:
            group_rows = []
            for profile_index, profile in enumerate(PROFILES):
                difference = (
                    matrix(summaries, profile, deadline, "BAA-controller", "mean_u_frame")
                    - matrix(summaries, profile, deadline, comparator, "mean_u_frame")
                )
                result, _ = bootstrap_difference(
                    difference, iterations, 4100 + comparator_index * 100 + deadline + profile_index
                )
                row = common | {
                    "profile": profile,
                    "deadline_ms": deadline,
                    "policy": "BAA-controller",
                    "comparator": comparator,
                    "metric": "mean_u_frame",
                    **result,
                    "noninferiority_margin": "",
                }
                group_rows.append(row)
                u_rows[(profile, deadline, comparator)] = row
            adjusted = holm_adjust([row["raw_p"] for row in group_rows])
            for row, value in zip(group_rows, adjusted):
                row["holm_p"] = value
                row["gate_pass"] = row["ci_low"] > 0.0 and value < 0.05
                rows.append(row)
    aggregate_rows = []
    for comparator_index, comparator in enumerate(("best-fixed", "capacity-greedy")):
        difference = (
            aggregate_matrix(summaries, "BAA-controller", "mean_u_frame", deadlines)
            - aggregate_matrix(summaries, comparator, "mean_u_frame", deadlines)
        )
        result, _ = bootstrap_difference(difference, iterations, 4500 + comparator_index)
        row = common | {
            "profile": "ALL",
            "deadline_ms": 0,
            "policy": "BAA-controller",
            "comparator": comparator,
            "metric": "mean_u_frame_equal_profile_deadline_aggregate",
            **result,
            "holm_p": result["raw_p"],
            "noninferiority_margin": "",
            "gate_pass": result["ci_low"] >= 0.0,
        }
        aggregate_rows.append(row)
        rows.append(row)
    noninferiority = []
    for metric_index, (metric, margin_kind) in enumerate(
        (
            ("total_wait_p99_ms", "waiting"),
            ("mean_overshoot", "proportion"),
            ("deadline_violation_ratio", "proportion"),
        )
    ):
        baa = aggregate_matrix(summaries, "BAA-controller", metric, deadlines)
        comparator = aggregate_matrix(summaries, "capacity-greedy", metric, deadlines)
        difference = baa - comparator
        result, distribution = bootstrap_difference(difference, iterations, 4600 + metric_index)
        margin = max(5.0, 0.05 * float(np.mean(comparator))) if margin_kind == "waiting" else 0.01
        one_sided_p = (np.count_nonzero(distribution >= margin) + 1) / (iterations + 1)
        row = common | {
            "profile": "ALL",
            "deadline_ms": 0,
            "policy": "BAA-controller",
            "comparator": "capacity-greedy",
            "metric": metric,
            **result,
            "raw_p": float(one_sided_p),
            "noninferiority_margin": margin,
            "gate_pass": result["ci_high"] < margin,
        }
        noninferiority.append(row)
    adjusted_noninferiority = holm_adjust([row["raw_p"] for row in noninferiority])
    for row, value in zip(noninferiority, adjusted_noninferiority):
        row["holm_p"] = value
        rows.append(row)
    primary_deadline = int(network["primary_deadline_ms"])
    best_wins = sum(
        u_rows[(profile, primary_deadline, "best-fixed")]["gate_pass"] for profile in PROFILES
    )
    greedy_wins = sum(
        u_rows[(profile, primary_deadline, "capacity-greedy")]["gate_pass"] for profile in PROFILES
    )
    gate = (
        best_wins >= 2
        and greedy_wins >= 1
        and all(row["gate_pass"] for row in aggregate_rows)
        and all(row["gate_pass"] for row in noninferiority)
    )
    statistics_path = args.data_dir / "core_v3_bandwidth_statistics.csv"
    with statistics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STAT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        f"{row[key]:.10f}"
                        if key in {"effect_size", "ci_low", "ci_high", "raw_p", "holm_p"}
                        else int(row[key]) if key == "gate_pass" else row[key]
                    )
                    for key in STAT_FIELDS
                }
            )
    audit = {
        "status": "pass",
        "contract_version": contract["contract_version"],
        "contract_sha256": contract_sha256(),
        "summary_rows": summary_rows,
        "detail_rows": detail_rows,
        "primary_deadline_ms": primary_deadline,
        "best_fixed_profile_wins": best_wins,
        "capacity_greedy_profile_wins": greedy_wins,
        "aggregate_gate_pass": all(row["gate_pass"] for row in aggregate_rows),
        "noninferiority_gate_pass": all(row["gate_pass"] for row in noninferiority),
        "G4_gate_pass": gate,
        "outputs": {
            "summary": {"path": str(summary_path), "sha256": file_sha256(summary_path)},
            "detail": {"path": str(detail_path), "sha256": file_sha256(detail_path)},
            "statistics": {"path": str(statistics_path), "sha256": file_sha256(statistics_path)},
        },
    }
    (args.output_dir / "bandwidth_gate_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(f"PASS: bandwidth analysis G4={gate} summary={summary_rows} detail={detail_rows}")


if __name__ == "__main__":
    main()
