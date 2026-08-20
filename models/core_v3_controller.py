"""Causal risk-aware controller used by the frozen core-v3 replay protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ActionCalibration:
    q_index: int
    prefix_k: int
    utility: float
    payload_p90_bytes: float
    payload_mean_bytes: float
    encode_p95_s: float
    encode_mean_s: float
    decode_p95_s: float
    decode_mean_s: float


@dataclass(frozen=True)
class WorkerAvailability:
    encoder_s: float
    transmitter_s: float
    decoder_s: float


@dataclass(frozen=True)
class ControllerDecision:
    q_index: int
    nominal_prefix_k: int
    wire_budget_bytes: float
    predicted_network_bytes: int
    fallback: bool
    risk_aware: bool


@dataclass(frozen=True)
class TransmissionDecision:
    q_index: int
    nominal_prefix_k: int
    transmitted_prefix_k: int
    payload_bytes: int
    network_bytes: int
    wire_budget_bytes: float
    fallback: bool
    risk_aware: bool


class CoreV3BandwidthController:
    def __init__(
        self,
        actions: Iterable[ActionCalibration],
        loss_bin_edges: tuple[float, float, float, float],
        goodput_lower_quantiles: tuple[float, float, float],
        global_goodput_lower_quantile: float,
        minimum_bin_samples: tuple[int, int, int],
        risk_aware: bool = True,
    ) -> None:
        self.actions = tuple(actions)
        if {(action.q_index, action.prefix_k) for action in self.actions} != {
            (q_index, prefix_k) for q_index in range(4) for prefix_k in range(1, 5)
        }:
            raise ValueError("Controller calibration must contain all 16 q-prefix actions")
        if loss_bin_edges != (0.0, 0.05, 0.15, 1.0):
            raise ValueError("Unexpected packet-loss calibration bins")
        if len(goodput_lower_quantiles) != 3 or len(minimum_bin_samples) != 3:
            raise ValueError("Three packet-loss calibration bins are required")
        self.loss_bin_edges = loss_bin_edges
        self.goodput_lower_quantiles = goodput_lower_quantiles
        self.global_goodput_lower_quantile = float(global_goodput_lower_quantile)
        self.minimum_bin_samples = minimum_bin_samples
        self.risk_aware = risk_aware
        self.by_action = {(action.q_index, action.prefix_k): action for action in self.actions}

    @staticmethod
    def network_bytes(payload_bytes: float) -> int:
        payload = max(0, int(round(payload_bytes)))
        packets = (payload + 1459) // 1460 if payload else 0
        return payload + packets * 40

    def _goodput_factor(self, packet_loss_estimate: float) -> float:
        if not self.risk_aware:
            return 1.0
        value = min(max(float(packet_loss_estimate), 0.0), 1.0)
        bin_index = 2
        for index in range(3):
            if self.loss_bin_edges[index] <= value < self.loss_bin_edges[index + 1]:
                bin_index = index
                break
        if self.minimum_bin_samples[bin_index] < 30:
            return self.global_goodput_lower_quantile
        return self.goodput_lower_quantiles[bin_index]

    def _budget_for_action(
        self,
        action: ActionCalibration,
        estimated_goodput_bps: float,
        packet_loss_estimate: float,
        frame_arrival_s: float,
        deadline_s: float,
        workers: WorkerAvailability,
    ) -> float:
        encode_s = action.encode_p95_s if self.risk_aware else action.encode_mean_s
        decode_s = action.decode_p95_s if self.risk_aware else action.decode_mean_s
        encoder_done = max(frame_arrival_s, workers.encoder_s) + encode_s
        transmitter_start = max(encoder_done, workers.transmitter_s)
        deadline_at = frame_arrival_s + deadline_s
        last_transmission_time = deadline_at - decode_s
        if workers.decoder_s > last_transmission_time:
            return 0.0
        effective_goodput = max(0.0, estimated_goodput_bps) * self._goodput_factor(packet_loss_estimate)
        return max(0.0, effective_goodput * (last_transmission_time - transmitter_start) / 8.0)

    @staticmethod
    def _selection_key(action: ActionCalibration, network_bytes: int) -> tuple[float, int, int, int, int, int]:
        return (-action.utility, network_bytes, action.prefix_k, -action.q_index, action.q_index, action.prefix_k)

    def select_preencode(
        self,
        estimated_goodput_bps: float,
        packet_loss_estimate: float,
        frame_arrival_s: float,
        deadline_s: float,
        workers: WorkerAvailability,
    ) -> ControllerDecision:
        feasible: list[tuple[ActionCalibration, int, float]] = []
        for action in self.actions:
            payload = action.payload_p90_bytes if self.risk_aware else action.payload_mean_bytes
            candidate_network_bytes = self.network_bytes(payload)
            budget = self._budget_for_action(
                action,
                estimated_goodput_bps,
                packet_loss_estimate,
                frame_arrival_s,
                deadline_s,
                workers,
            )
            if candidate_network_bytes <= budget:
                feasible.append((action, candidate_network_bytes, budget))
        if not feasible:
            fallback_action = self.by_action[(3, 1)]
            budget = self._budget_for_action(
                fallback_action,
                estimated_goodput_bps,
                packet_loss_estimate,
                frame_arrival_s,
                deadline_s,
                workers,
            )
            payload = fallback_action.payload_p90_bytes if self.risk_aware else fallback_action.payload_mean_bytes
            return ControllerDecision(
                q_index=3,
                nominal_prefix_k=1,
                wire_budget_bytes=budget,
                predicted_network_bytes=self.network_bytes(payload),
                fallback=True,
                risk_aware=self.risk_aware,
            )
        action, predicted_network_bytes, budget = min(
            feasible,
            key=lambda item: self._selection_key(item[0], item[1]),
        )
        return ControllerDecision(
            q_index=action.q_index,
            nominal_prefix_k=action.prefix_k,
            wire_budget_bytes=budget,
            predicted_network_bytes=predicted_network_bytes,
            fallback=False,
            risk_aware=self.risk_aware,
        )

    def select_postencode(
        self,
        decision: ControllerDecision,
        cumulative_payload_bytes: dict[int, int],
    ) -> TransmissionDecision:
        candidates: list[tuple[ActionCalibration, int, int]] = []
        for prefix_k in range(1, decision.nominal_prefix_k + 1):
            payload = int(cumulative_payload_bytes[prefix_k])
            network_bytes = self.network_bytes(payload)
            if network_bytes <= decision.wire_budget_bytes:
                candidates.append((self.by_action[(decision.q_index, prefix_k)], payload, network_bytes))
        if not candidates:
            return TransmissionDecision(
                q_index=decision.q_index,
                nominal_prefix_k=decision.nominal_prefix_k,
                transmitted_prefix_k=0,
                payload_bytes=0,
                network_bytes=0,
                wire_budget_bytes=decision.wire_budget_bytes,
                fallback=decision.fallback,
                risk_aware=decision.risk_aware,
            )
        action, payload, network_bytes = min(
            candidates,
            key=lambda item: self._selection_key(item[0], item[2]),
        )
        return TransmissionDecision(
            q_index=decision.q_index,
            nominal_prefix_k=decision.nominal_prefix_k,
            transmitted_prefix_k=action.prefix_k,
            payload_bytes=payload,
            network_bytes=network_bytes,
            wire_budget_bytes=decision.wire_budget_bytes,
            fallback=decision.fallback,
            risk_aware=decision.risk_aware,
        )
