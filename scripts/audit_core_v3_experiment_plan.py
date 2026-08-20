from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "experiments" / "minimum_sufficient_experiment_plan_literature_aligned_20260715.md"
CONTRACT = ROOT / "configs" / "core_v3_experiment_contract.json"
ROOT_PLAN = ROOT / "experiment_plan.md"
OLD_PLAN = ROOT / "experiments" / "minimum_sufficient_experiment_plan_20260715.md"


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> None:
    plan = PLAN.read_text(encoding="utf-8")
    root_plan = ROOT_PLAN.read_text(encoding="utf-8")
    old_plan = OLD_PLAN.read_text(encoding="utf-8")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    failures: list[str] = []

    model = contract["model"]
    training = contract["training"]
    network = contract["network_replay"]
    statistics = contract["statistics"]

    require(contract["contract_version"] == "core_v3_20260715_r3", "unexpected contract version", failures)

    require(model["latent_groups"] == 4, "latent_groups must equal 4", failures)
    require(model["group_channels"] == 48, "group_channels must equal 48", failures)
    require(len(model["q_indices"]) == 4, "exactly four q indices are required", failures)
    require(len(model["active_ratios"]) == 4, "exactly four prefixes are required", failures)
    require(
        all(a < b for a, b in zip(model["q_steps"], model["q_steps"][1:])),
        "q steps must be strictly increasing",
        failures,
    )
    require(
        len(model["q_indices"]) * len(model["active_ratios"]) == 16,
        "action space must contain 16 q-prefix pairs",
        failures,
    )
    require(model["enhancement_layers"] == 0, "enhancement layers must remain disabled", failures)
    require(model["decoder_complexity_executed_paths_only"], "decoder complexity padding is not prohibited", failures)

    bitstream = contract["bitstream"]
    require(bitstream["header_bytes"] > 0, "header size must be positive", failures)
    require(bitstream["magic_field_bytes"] == 8, "magic field must use eight bytes", failures)
    require(len(bytes.fromhex(bitstream["magic_bytes_hex"])) == bitstream["magic_field_bytes"], "magic bytes do not match field size", failures)
    require(bitstream["segment_miniheader_bytes"] > 0, "segment header must be positive", failures)
    require(not bitstream["whole_stream_checksum"], "whole-stream checksum breaks truncation", failures)

    data = contract["data"]
    require(data["source_id_ordering"] == "utf8_bytewise_sorted_normalized_posix_relative_path", "data source ordering is not deterministic", failures)
    require(data["shuffle_algorithm"] == "numpy_pcg64_permutation", "data split shuffle is not frozen", failures)

    joint = training["joint"]
    decoder = training["decoder_only"]
    require(joint["minimum_steps"] <= joint["maximum_steps"], "invalid joint step limits", failures)
    require(decoder["minimum_steps"] <= decoder["maximum_steps"], "invalid decoder step limits", failures)
    require(
        joint["full_validation_interval"] % joint["monitor_validation_interval"] == 0,
        "joint validation intervals must align",
        failures,
    )
    require(
        decoder["full_validation_interval"] % decoder["monitor_validation_interval"] == 0,
        "decoder validation intervals must align",
        failures,
    )
    require(len(set(training["model_seeds"])) == 3, "three unique model seeds are required", failures)
    for key in ["roi_weight", "edge_weight", "teacher_weight", "detector_weight", "perceptual_weight"]:
        require(training["loss"][key] == 0.0, f"{key} must remain zero", failures)

    require(
        network["network_payload_bytes"] + network["packet_overhead_bytes"] == network["mtu_bytes"],
        "network payload and overhead must sum to MTU",
        failures,
    )
    require(network["development_seed_start"] != network["test_seed_start"], "trace seeds overlap", failures)
    require(network["test_realizations_per_profile"] == 1000, "test realizations must equal 1000", failures)
    require(set(network["profiles"]) == {"smooth_ar", "drop_recovery", "burst_loss"}, "profile set drifted", failures)
    require(statistics["bootstrap_iterations"] == 10000, "bootstrap count must equal 10000", failures)
    require(statistics["multiple_comparison_correction"] == "holm", "Holm correction is required", failures)
    require(statistics["c3b_aggregate_gate_requires_nonnegative_ci_lower"], "aggregate C3b gate must use a CI", failures)
    require("per_image_full_stream_byte_fraction" in statistics["c3a_primary_metric"], "C3a AUC byte normalization is ambiguous", failures)
    require("test_statistics" in network["controller_forbidden_test_information"], "controller test leakage guard is incomplete", failures)
    require("causal_worker_availability" in network["controller_test_information"], "controller queue information timing is undefined", failures)
    require(not network["controller_profile_label_available"], "controller may not observe the synthetic profile label", failures)
    require("development_tiles" in network["development_image_sampling"], "development traces do not isolate development images", failures)
    require("test_tiles" in network["test_image_sampling"], "test trace image source is undefined", failures)
    require(network["require_action_specific_development_and_test_latency_caches"], "E4 latency cache coverage is incomplete", failures)
    require(network["packetization_formula"] == "network_bytes=payload_bytes+ceil(payload_bytes/1460)*40", "packetization formula drifted", failures)
    require(network["controller_no_feasible_action_fallback"].startswith("encode_once_at_q3"), "controller has no deterministic empty-budget fallback", failures)

    required_plan_tokens = [
        "configs/core_v3_experiment_contract.json",
        "G0a structural smoke",
        "G0b evidence gate",
        "all-24-orderings",
        "hierarchical paired bootstrap",
        "capacity-greedy",
        "U_frame=delivered*MS-SSIM",
        "受控 stochastic trace replay",
        "sender/full-codec",
        "参数量和 MACs 只统计实际参与至少一个 prefix 解码的执行路径",
        "controller 在测试时不得访问当前/未来帧图像",
        "hierarchical-bootstrap 95% CI 下界均须不小于 0",
        "不低于第 6 名的 AUC rank",
        "base 仍不满足预算时置 `k=0`",
        "development/test 图像绝不交叉",
        "action-specific encode/decode latency cache",
    ]
    for token in required_plan_tokens:
        require(token in plan, f"plan is missing required token: {token}", failures)

    forbidden_plan_tokens = [
        "fixed-random-order",
        "random prefixes",
        "full codec 的 30%",
        "drop 采用预注册惩罚",
        "P0 通过前不得启动三种子训练",
        "ordered/reverse/random progressive trajectories",
    ]
    for token in forbidden_plan_tokens:
        require(token not in plan, f"stale plan token remains: {token}", failures)

    require(not any(line.endswith((" ", "\t")) for line in plan.splitlines()), "plan has trailing whitespace", failures)
    require("core_v3_20260715_r3" in root_plan, "root plan does not point to the current contract", failures)
    require("historical" in root_plan.lower() and "non-executable" in root_plan.lower(), "root plan does not quarantine historical material", failures)
    require("不得作为后续实验入口" in old_plan, "superseded minimum plan is not quarantined", failures)

    if failures:
        print(f"FAIL: {len(failures)} core-v3 plan issue(s)")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("PASS: core-v3 experiment plan and contract are internally consistent")


if __name__ == "__main__":
    main()
