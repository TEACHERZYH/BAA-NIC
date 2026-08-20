"""Contract helpers for the versioned core-v3 decoder extension."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from core_v3_contract import contract_sha256, load_seed_expansion


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs" / "core_v3_decoder_extension_contract.json"


def load_extension_contract() -> dict[str, Any]:
    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if payload["parent_contract_sha256"] != contract_sha256():
        raise RuntimeError("Decoder extension parent contract does not match the active core-v3 contract")
    if payload["scope"] != "deterministic_same_bitstream_server_decoder_extension":
        raise RuntimeError("Unexpected decoder extension scope")
    return payload


def extension_contract_sha256() -> str:
    return hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()


def registered_extension_seeds(contract: dict[str, Any] | None = None) -> list[int]:
    contract = load_extension_contract() if contract is None else contract
    base = [int(value) for value in contract["model_seeds"]]
    expansion = load_seed_expansion()
    if expansion is None:
        return base
    if expansion.get("decoder_extension_contract_sha256") != extension_contract_sha256():
        raise RuntimeError("Seed expansion belongs to another decoder extension")
    if [int(value) for value in expansion["existing_model_seeds"]] != base:
        raise RuntimeError("Decoder seed expansion does not preserve base seeds")
    return [int(value) for value in expansion["combined_model_seeds"]]
