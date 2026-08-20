from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import ActionCalibration, CoreV3BandwidthController, WorkerAvailability
from core_v3_contract import contract_sha256, load_contract


def actions() -> list[ActionCalibration]:
    result: list[ActionCalibration] = []
    for q_index in range(4):
        for prefix_k in range(1, 5):
            payload = (5 - q_index) * prefix_k * 1200.0
            result.append(
                ActionCalibration(
                    q_index=q_index,
                    prefix_k=prefix_k,
                    utility=0.45 + 0.08 * prefix_k - 0.02 * q_index,
                    payload_p90_bytes=payload * 1.1,
                    payload_mean_bytes=payload,
                    encode_p95_s=0.014,
                    encode_mean_s=0.010,
                    decode_p95_s=0.020 + 0.004 * prefix_k,
                    decode_mean_s=0.015 + 0.003 * prefix_k,
                )
            )
    return result


def main() -> None:
    contract = load_contract()
    kwargs = {
        "actions": actions(),
        "loss_bin_edges": (0.0, 0.05, 0.15, 1.0),
        "goodput_lower_quantiles": (0.82, 0.70, 0.55),
        "global_goodput_lower_quantile": 0.68,
        "minimum_bin_samples": (120, 120, 120),
    }
    risk = CoreV3BandwidthController(**kwargs, risk_aware=True)
    greedy = CoreV3BandwidthController(**kwargs, risk_aware=False)
    workers = WorkerAvailability(encoder_s=0.0, transmitter_s=0.0, decoder_s=0.0)
    risk_decision = risk.select_preencode(2_000_000.0, 0.12, 0.0, 0.25, workers)
    greedy_decision = greedy.select_preencode(2_000_000.0, 0.12, 0.0, 0.25, workers)
    if risk_decision.predicted_network_bytes > risk_decision.wire_budget_bytes:
        raise RuntimeError("Risk controller selected an infeasible calibrated action")
    if risk_decision.wire_budget_bytes >= greedy_decision.wire_budget_bytes:
        raise RuntimeError("Risk margin did not reduce the wire budget")
    actual = {1: 7000, 2: 14000, 3: 21000, 4: 28000}
    post = risk.select_postencode(risk_decision, actual)
    if post.network_bytes > post.wire_budget_bytes:
        raise RuntimeError("Post-encode prefix exceeds the frozen wire budget")

    blocked_workers = WorkerAvailability(encoder_s=1.0, transmitter_s=1.0, decoder_s=1.0)
    fallback = risk.select_preencode(1000.0, 0.4, 0.0, 0.1, blocked_workers)
    no_send = risk.select_postencode(fallback, actual)
    if (fallback.q_index, fallback.nominal_prefix_k, no_send.transmitted_prefix_k, no_send.network_bytes) != (3, 1, 0, 0):
        raise RuntimeError("No-feasible-action fallback differs from the contract")

    output = ROOT / "runs" / "core_v3_local" / "controller_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "pass",
        "contract_version": contract["contract_version"],
        "contract_sha256": contract_sha256(),
        "risk_decision": risk_decision.__dict__,
        "greedy_decision": greedy_decision.__dict__,
        "postencode_decision": post.__dict__,
        "fallback_decision": fallback.__dict__,
        "no_send_decision": no_send.__dict__,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("PASS: causal risk-aware and capacity-greedy controller smoke")


if __name__ == "__main__":
    main()
