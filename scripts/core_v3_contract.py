"""Shared loading and provenance helpers for the frozen core-v3 contract."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs" / "core_v3_experiment_contract.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contract_sha256() -> str:
    return file_sha256(CONTRACT_PATH)


def files_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def load_contract() -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contract.get("contract_version") != "core_v3_20260715_r3":
        raise RuntimeError(f"Unexpected core-v3 contract version: {contract.get('contract_version')}")
    return contract


def seed_expansion_path() -> Path | None:
    value = os.environ.get("CORE_V3_SEED_EXPANSION_PATH")
    return Path(value).expanduser().resolve() if value else None


def load_seed_expansion() -> dict[str, Any] | None:
    path = seed_expansion_path()
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "user_authorized_frozen_seed_expansion":
        raise RuntimeError("Seed expansion is not frozen and authorized")
    if payload.get("parent_contract_sha256") != contract_sha256():
        raise RuntimeError("Seed expansion belongs to another core-v3 contract")
    return payload


def seed_expansion_sha256() -> str | None:
    path = seed_expansion_path()
    return file_sha256(path) if path is not None else None


def registered_model_seeds(contract: dict[str, Any] | None = None) -> list[int]:
    contract = load_contract() if contract is None else contract
    base = [int(value) for value in contract["training"]["model_seeds"]]
    expansion = load_seed_expansion()
    if expansion is None:
        return base
    if [int(value) for value in expansion["existing_model_seeds"]] != base:
        raise RuntimeError("Seed expansion does not preserve the frozen base seeds")
    combined = [int(value) for value in expansion["combined_model_seeds"]]
    if combined != base + [int(value) for value in expansion["additional_model_seeds"]]:
        raise RuntimeError("Seed expansion order is inconsistent")
    if len(set(combined)) != len(combined):
        raise RuntimeError("Seed expansion contains duplicate seeds")
    return combined


def state_dict_sha256(module: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def sender_state_dict_sha256(model: Any) -> str:
    """Hash only parameters/buffers required to create and entropy-decode a stream."""

    digest = hashlib.sha256()
    prefixes = ("encoder.", "group_hyperprior.")
    selected = [(name, tensor) for name, tensor in model.state_dict().items() if name.startswith(prefixes)]
    if not selected:
        raise RuntimeError("No sender state found for core-v3 checkpoint hashing")
    for name, tensor in sorted(selected):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()
