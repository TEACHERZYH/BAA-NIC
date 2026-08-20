"""Paired hierarchical analysis for the deterministic decoder extension."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyze_core_v3_quality import bootstrap, cluster_means, holm_adjust
from core_v3_contract import seed_expansion_sha256
from core_v3_decoder_extension import (
    extension_contract_sha256,
    load_extension_contract,
    registered_extension_seeds,
)


METRICS = ("ms_ssim", "psnr", "lpips")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "core_v3_decoder_extension")
    parser.add_argument("--datasets", default="visdrone,uavid")
    return parser.parse_args()


def uavid_clusters() -> dict[str, str]:
    with (ROOT / "data" / "core_v3_uavid_test.csv").open(encoding="utf-8", newline="") as handle:
        return {row["tile_id"]: row["sequence_id"] for row in csv.DictReader(handle)}


def combine_inputs(inputs: list[Path], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    header: str | None = None
    rows = 0
    with output.open("w", encoding="utf-8", newline="") as target:
        for path in inputs:
            with path.open(encoding="utf-8", newline="") as source:
                current = source.readline()
                if header is None:
                    header = current
                    target.write(current)
                elif current != header:
                    raise RuntimeError(f"CSV header mismatch: {path}")
                for line in source:
                    target.write(line)
                    rows += 1
    return rows


def load_pairs(
    inputs: list[Path],
    seeds: tuple[int, ...],
) -> tuple[dict[tuple[Any, ...], dict[str, dict[str, Any]]], int]:
    pairs: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    rows = 0
    for seed, path in zip(seeds, inputs):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if int(row["seed"]) != seed:
                    raise RuntimeError(f"Seed mismatch in {path}")
                key = (
                    row["dataset"],
                    seed,
                    row["image_id"],
                    int(row["q_index"]),
                    int(row["prefix_k"]),
                )
                decoder = row["decoder"]
                if decoder not in ("enhanced_r1", "heavy") or decoder in pairs.setdefault(key, {}):
                    raise RuntimeError(f"Invalid or duplicate decoder row: {key} {decoder}")
                pairs[key][decoder] = row
                rows += 1
    for key, decoders in pairs.items():
        if set(decoders) != {"enhanced_r1", "heavy"}:
            raise RuntimeError(f"Incomplete decoder pair: {key}")
        enhanced = decoders["enhanced_r1"]
        heavy = decoders["heavy"]
        if (enhanced["bitstream_sha256"], enhanced["bytes"], enhanced["bpp"]) != (
            heavy["bitstream_sha256"],
            heavy["bytes"],
            heavy["bpp"],
        ):
            raise RuntimeError(f"Decoder pair does not use byte-identical input: {key}")
    return pairs, rows


def build_delta_arrays(
    pairs: dict[tuple[Any, ...], dict[str, dict[str, Any]]],
    datasets: list[str],
    seeds: tuple[int, ...],
) -> tuple[dict[tuple[str, str, int], tuple[np.ndarray, list[str]]], list[dict[str, Any]]]:
    accum: dict[tuple[str, int, str, int, str], list[float]] = {}
    for (dataset, seed, image_id, _, prefix_k), decoders in pairs.items():
        for metric in METRICS:
            delta = float(decoders["enhanced_r1"][metric]) - float(decoders["heavy"][metric])
            accum.setdefault((dataset, seed, image_id, prefix_k, metric), []).append(delta)
    for key, values in accum.items():
        if len(values) != 4:
            raise RuntimeError(f"Expected four q-index observations for {key}")

    arrays: dict[tuple[str, str, int], tuple[np.ndarray, list[str]]] = {}
    per_image: list[dict[str, Any]] = []
    for dataset in datasets:
        image_ids = sorted(
            {image_id for current_dataset, _, image_id, _, _ in accum if current_dataset == dataset}
        )
        for metric in METRICS:
            for prefix_k in range(1, 5):
                values = np.empty((len(seeds), len(image_ids)), dtype=np.float64)
                for seed_index, seed in enumerate(seeds):
                    for image_index, image_id in enumerate(image_ids):
                        delta = float(np.mean(accum[(dataset, seed, image_id, prefix_k, metric)]))
                        values[seed_index, image_index] = delta
                        per_image.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "image_id": image_id,
                                "prefix_k": prefix_k,
                                "metric": metric,
                                "mean_q_delta_enhanced_minus_heavy": delta,
                            }
                        )
                arrays[(dataset, metric, prefix_k)] = (values, image_ids)
    return arrays, per_image


def main() -> None:
    args = parse_args()
    contract = load_extension_contract()
    seeds = tuple(registered_extension_seeds(contract))
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    if not datasets or len(set(datasets)) != len(datasets):
        raise ValueError("datasets must contain one or more unique names")
    unknown = sorted(set(datasets) - {"visdrone", "uavid"})
    if unknown:
        raise ValueError(f"Unknown analysis datasets: {unknown}")
    if "visdrone" not in datasets:
        raise ValueError("The confirmatory decoder-extension gate requires VisDrone")
    inputs = [
        args.runs_root / "core_v3_decoder_extension_quality" / f"seed_{seed}" / "core_v3_decoder_extension.csv"
        for seed in seeds
    ]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    expansion_hash = seed_expansion_sha256()
    base_seeds = {int(value) for value in contract["model_seeds"]}
    for seed in seeds:
        if seed in base_seeds:
            continue
        metadata_path = (
            args.runs_root
            / "core_v3_decoder_extension_quality"
            / f"seed_{seed}"
            / "metadata.json"
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if expansion_hash is None or metadata.get("seed_expansion_sha256") != expansion_hash:
            raise RuntimeError(f"Missing seed-expansion provenance for seed {seed}")
    combined_path = args.data_dir / "core_v3_decoder_extension.csv"
    combined_rows = combine_inputs(inputs, combined_path)
    pairs, loaded_rows = load_pairs(inputs, seeds)
    if loaded_rows != combined_rows:
        raise RuntimeError("Combined and analyzed row counts differ")
    arrays, per_image = build_delta_arrays(pairs, datasets, seeds)
    clusters = uavid_clusters() if "uavid" in datasets else {}
    iterations = int(contract["confirmatory_endpoint"]["bootstrap_iterations"])

    summary_rows: list[dict[str, Any]] = []
    visdrone_msssim: list[dict[str, Any]] = []
    for dataset in datasets:
        for metric in METRICS:
            for prefix_k in range(1, 5):
                values, image_ids = arrays[(dataset, metric, prefix_k)]
                units = values
                unit_name = "source_image"
                if dataset == "uavid":
                    units = cluster_means(values, [clusters[image_id] for image_id in image_ids])
                    unit_name = "sequence_cluster"
                stats = bootstrap(
                    units,
                    iterations,
                    seed=20260717 + prefix_k + (100 if dataset == "uavid" else 0) + METRICS.index(metric) * 10,
                )
                result = {
                    "dataset": dataset,
                    "metric": metric,
                    "prefix_k": prefix_k,
                    **stats,
                    "holm_p": "",
                    "seed_effects": [float(np.mean(units[index])) for index in range(len(seeds))],
                    "resampling_unit": unit_name,
                    "gate_pass": "",
                }
                summary_rows.append(result)
                if dataset == "visdrone" and metric == "ms_ssim":
                    visdrone_msssim.append(result)

    adjusted = holm_adjust([float(row["raw_p"]) for row in visdrone_msssim])
    for row, holm_p in zip(visdrone_msssim, adjusted):
        row["holm_p"] = holm_p
        row["gate_pass"] = bool(
            float(row["ci_low"]) > 0.0
            and holm_p < 0.05
            and all(effect >= 0.0 for effect in row["seed_effects"])
        )
    positive_prefixes = sum(bool(row["gate_pass"]) for row in visdrone_msssim)
    gate_pass = positive_prefixes >= int(
        contract["confirmatory_endpoint"]["minimum_positive_prefixes"]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_image_path = args.data_dir / "core_v3_decoder_extension_per_image.csv"
    with per_image_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "dataset",
            "seed",
            "image_id",
            "prefix_k",
            "metric",
            "mean_q_delta_enhanced_minus_heavy",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in per_image:
            writer.writerow({**row, "mean_q_delta_enhanced_minus_heavy": f"{row['mean_q_delta_enhanced_minus_heavy']:.10f}"})

    summary_path = args.data_dir / "core_v3_decoder_extension_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "dataset",
            "metric",
            "prefix_k",
            "effect",
            "ci_low",
            "ci_high",
            "raw_p",
            "holm_p",
            "seed_effects",
            "resampling_unit",
            "gate_pass",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    **row,
                    "effect": f"{float(row['effect']):.10f}",
                    "ci_low": f"{float(row['ci_low']):.10f}",
                    "ci_high": f"{float(row['ci_high']):.10f}",
                    "raw_p": f"{float(row['raw_p']):.10f}",
                    "holm_p": "" if row["holm_p"] == "" else f"{float(row['holm_p']):.10f}",
                    "seed_effects": ";".join(f"{value:.10f}" for value in row["seed_effects"]),
                }
            )

    audit = {
        "status": "pass" if gate_pass else "fail",
        "extension_version": contract["extension_version"],
        "extension_contract_sha256": extension_contract_sha256(),
        "seed_expansion_sha256": expansion_hash,
        "model_seeds": list(seeds),
        "byte_identical_pairs": True,
        "combined_rows": combined_rows,
        "paired_actions": len(pairs),
        "visdrone_positive_prefixes": positive_prefixes,
        "required_positive_prefixes": contract["confirmatory_endpoint"]["minimum_positive_prefixes"],
        "confirmatory_gate_pass": gate_pass,
        "visdrone_ms_ssim": visdrone_msssim,
    }
    (args.output_dir / "gate_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Core-v3 Decoder Extension Gate Audit",
        "",
        f"- Extension: `{contract['extension_version']}`",
        f"- Same-bitstream pairs: `PASS` ({len(pairs)} actions)",
        f"- VisDrone positive prefixes: `{positive_prefixes}/4`",
        f"- Confirmatory gate: `{'PASS' if gate_pass else 'FAIL'}`",
        "",
        "| Prefix | MS-SSIM effect | 95% CI | Holm p | Seed effects | Gate |",
        "|---:|---:|---:|---:|---|---|",
    ]
    for row in visdrone_msssim:
        lines.append(
            f"| {row['prefix_k']} | {row['effect']:.6f} | "
            f"[{row['ci_low']:.6f}, {row['ci_high']:.6f}] | {row['holm_p']:.6g} | "
            f"{'; '.join(f'{value:.6f}' for value in row['seed_effects'])} | "
            f"{'PASS' if row['gate_pass'] else 'FAIL'} |"
        )
    (args.output_dir / "gate_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"PASS: analyzed decoder extension rows={combined_rows} "
        f"confirmatory_gate={'PASS' if gate_pass else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
