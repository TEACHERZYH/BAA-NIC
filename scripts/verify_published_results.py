"""Verify the compact, manuscript-facing BAA-NIC evidence release."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXPECTED_CONTRACTS = {
    "core_v3_experiment_contract.json": "82de33fe6659bd741d774251e01e7602e4460a32770790dc336b7a7ff199b26c",
    "core_v3_decoder_extension_contract.json": "1148399a39b4a24357c253da6c624875c2262039061da9cf9f83bdf34197e05a",
    "core_v3_decoder_five_seed_addendum.json": "c3233ba9535c7f7ef7bba6f1eebc81744bb31d7db6db5b2257cff1dbfc7390ac",
}
EXPECTED_RESULT_HASHES = {
    "core_v3_bitstream_roundtrip.csv": "624815c5b703b80d7b22862966d8fa75ab5361918afa247b4d226f6b3b653bc2",
    "core_v3_operating_points.csv": "095a7bc40acecd620b7d79c07722c728aeb8c01a36628be4a78760cc994f560a",
    "core_v3_bandwidth_statistics.csv": "df48223d112e24f7693c6153e7d74e1e55f30a6dc16c0beb760ec1f5294f0352",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_contracts() -> None:
    for name, expected in EXPECTED_CONTRACTS.items():
        path = ROOT / "configs" / name
        require(path.is_file(), f"missing contract: {name}")
        require(sha256(path) == expected, f"contract hash mismatch: {name}")


def verify_bitstreams() -> None:
    for name, expected in EXPECTED_RESULT_HASHES.items():
        path = RESULTS / name
        require(path.is_file(), f"missing result: {name}")
        require(sha256(path) == expected, f"result hash mismatch: {name}")

    rows = read_csv(RESULTS / "core_v3_bitstream_roundtrip.csv")
    require(len(rows) == 11520, "round-trip row count must be 11,520")
    require(all(row["decode_ok"].lower() == "true" for row in rows), "independent decode failure")
    require(all(row["crc_ok"].lower() == "true" for row in rows), "CRC failure")
    prefixes = {"1000", "1100", "1110", "1111"}
    prefix_rows = [row for row in rows if row["subset_mask"] in prefixes]
    require(len(prefix_rows) == 3072, "cumulative-prefix row count must be 3,072")
    require(
        all(row["byte_prefix_equal"].lower() == "true" for row in prefix_rows),
        "cumulative stream is not byte-identical to its full-stream prefix",
    )

    operating = read_csv(RESULTS / "core_v3_operating_points.csv")
    require(len(operating) == 96, "operating-point row count must be 96")
    expected_actions = {(str(q), str(k)) for q in range(4) for k in range(1, 5)}
    for dataset in ("visdrone", "uavid"):
        for seed in ("17", "29", "43"):
            actions = {
                (row["q_index"], row["prefix_k"])
                for row in operating
                if row["dataset"] == dataset and row["seed"] == seed
            }
            require(actions == expected_actions, f"incomplete 16-action coverage: {dataset}/seed-{seed}")


def verify_sender_and_decoders() -> None:
    rows = read_csv(RESULTS / "core_v3_lightweight_asymmetry.csv")
    index = {
        (row["model"], row["component"], row["device"]): row
        for row in rows
    }
    baa_analysis = index[("baa_nic_depthwise", "analysis_transform", "cpu")]
    dense_analysis = index[("dense_conv_control", "analysis_transform", "cpu")]
    baa_sender = index[("baa_nic_depthwise", "full_sender", "cpu")]
    require(int(baa_sender["params"]) == 634800, "complete-sender parameter count mismatch")
    require(abs(float(baa_sender["gflops_512"]) - 2.63998669) < 1e-8, "complete-sender GFLOPs mismatch")
    param_reduction = 1.0 - int(baa_analysis["params"]) / int(dense_analysis["params"])
    mac_reduction = 1.0 - float(baa_analysis["gflops_512"]) / float(dense_analysis["gflops_512"])
    require(abs(param_reduction - 0.5315692241) < 1e-8, "analysis parameter reduction mismatch")
    require(abs(mac_reduction - 0.7073317308) < 1e-8, "analysis MAC reduction mismatch")

    asymmetry = read_csv(RESULTS / "core_v3_decoder_asymmetry.csv")
    require(len(asymmetry) == 5, "base-decoder summary must contain five tests")
    require(all(row["gate_pass"] == "1" for row in asymmetry), "base decoder gate failure")
    require(all(float(row["ci_low"]) > 0.0 for row in asymmetry), "base decoder CI crosses zero")

    extension = json.loads((RESULTS / "decoder_extension_5seed_gate_audit.json").read_text(encoding="utf-8"))
    require(extension["model_seeds"] == [17, 29, 43, 59, 71], "five-seed set mismatch")
    require(extension["byte_identical_pairs"] is True, "extension byte identity failed")
    require(extension["confirmatory_gate_pass"] is True, "enhanced decoder confirmatory gate failed")
    require(extension["visdrone_positive_prefixes"] == 4, "enhanced decoder prefix gate failed")


def verify_scope_boundaries() -> None:
    quality = json.loads((RESULTS / "quality_gate_audit.json").read_text(encoding="utf-8"))
    bandwidth = json.loads((RESULTS / "bandwidth_gate_audit.json").read_text(encoding="utf-8"))
    require(quality["E1"]["gate_pass"] is True, "E1 sender gate failed")
    require(quality["E2"]["gate_pass"] is True, "E2 byte-identical decoder gate failed")
    require(quality["E2"]["bitstream_byte_equality"] is True, "E2 byte equality failed")
    require(quality["E3"]["byte_strictly_monotonic"] is True, "prefix byte monotonicity failed")
    require(quality["E3"]["gate_pass"] is False, "published E3 boundary changed unexpectedly")
    require(bandwidth["detail_rows"] == 72900000, "causal replay must contain 72.9M rows")
    require(bandwidth["summary_rows"] == 243000, "causal replay summary count mismatch")
    require(bandwidth["noninferiority_gate_pass"] is True, "operational safety boundary failed")
    require(bandwidth["G4_gate_pass"] is False, "published controller-superiority boundary changed")


def main() -> None:
    verify_contracts()
    verify_bitstreams()
    verify_sender_and_decoders()
    verify_scope_boundaries()
    print("PASS: compact BAA-NIC evidence matches the frozen contracts and manuscript claim scope")


if __name__ == "__main__":
    main()
