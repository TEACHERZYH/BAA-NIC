"""Shared deterministic training utilities for the frozen core-v3 contract."""

from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np

from core_v3_contract import ROOT, contract_sha256, file_sha256, load_contract


ACTION_SCHEDULE_PATH = Path(
    os.environ.get(
        "CORE_V3_ACTION_SCHEDULE_PATH",
        ROOT / "configs" / "core_v3_action_schedules.json",
    )
).expanduser().resolve()
_DECODER_PERMUTATIONS: dict[tuple[int, int, int], np.ndarray] = {}


def configure_determinism(seed: int) -> None:
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def configure_practical_determinism(seed: int) -> None:
    """Require repeatable CUDA kernels for practical entropy bitstreams."""

    configure_determinism(seed)
    import torch

    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def load_seed_schedule(seed: int) -> dict[str, Any]:
    payload = json.loads(ACTION_SCHEDULE_PATH.read_text(encoding="utf-8"))
    if payload.get("contract_sha256") != contract_sha256():
        raise RuntimeError("Action schedule and core-v3 contract hashes differ")
    from core_v3_contract import seed_expansion_sha256

    expansion_hash = seed_expansion_sha256()
    if expansion_hash is not None and payload.get("seed_expansion_sha256") != expansion_hash:
        raise RuntimeError("Action schedule and seed expansion hashes differ")
    try:
        return payload["schedules"][str(seed)]
    except KeyError as exc:
        raise ValueError(f"Seed {seed} is not registered in the core-v3 action schedule") from exc


def action_for_step(seed: int, step: int, decoder_only: bool = False) -> tuple[int, int]:
    schedule = load_seed_schedule(seed)
    if not decoder_only and step < int(schedule["warmup_steps"]):
        action = schedule["warmup_action"]
    else:
        offset = step if decoder_only else step - int(schedule["warmup_steps"])
        cycle = schedule["periodic_cycle"]
        action = cycle[offset % len(cycle)]
    return int(action["q_index"]), int(action["prefix_k"])


def decoder_source_indices(seed: int, step: int, effective_batch: int, source_count: int) -> list[int]:
    """Return the pre-generated source-image minibatch shared by both decoders."""

    indices: list[int] = []
    for global_index in range(step * effective_batch, (step + 1) * effective_batch):
        cycle, offset = divmod(global_index, source_count)
        key = (int(seed), int(cycle), int(source_count))
        if key not in _DECODER_PERMUTATIONS:
            digest = hashlib.sha256(f"{seed}:decoder-source-cycle:{cycle}".encode("utf-8")).digest()
            cycle_seed = int.from_bytes(digest[:8], "little", signed=False)
            _DECODER_PERMUTATIONS[key] = np.random.Generator(np.random.PCG64(cycle_seed)).permutation(source_count)
        permutation = _DECODER_PERMUTATIONS[key]
        indices.append(int(permutation[offset]))
    return indices


def split_entropy_parameters(model: Any) -> tuple[list[Any], list[Any]]:
    named_parameters = list(model.named_parameters())
    auxiliary = [parameter for name, parameter in named_parameters if name.endswith(".quantiles")]
    auxiliary_ids = {id(parameter) for parameter in auxiliary}
    main = [parameter for _, parameter in named_parameters if id(parameter) not in auxiliary_ids]
    if not main or not auxiliary:
        raise RuntimeError("Expected both main and entropy-quantile parameter sets")
    return main, auxiliary


def learning_rate(step: int, maximum_steps: int, warmup_steps: int, initial_lr: float, minimum_lr: float) -> float:
    completed = step + 1
    if completed <= warmup_steps:
        return initial_lr * completed / max(1, warmup_steps)
    progress = min(1.0, (completed - warmup_steps) / max(1, maximum_steps - warmup_steps))
    return minimum_lr + 0.5 * (initial_lr - minimum_lr) * (1.0 + math.cos(math.pi * progress))


def set_optimizer_lr(optimizer: Any, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def reconstruction_objective(
    model: Any,
    images: Any,
    q_index: int,
    prefix_k: int,
    include_rate: bool,
) -> dict[str, Any]:
    import torch.nn.functional as F

    contract = load_contract()
    weights = contract["training"]["loss"]
    active_ratio = prefix_k / 4.0
    output = model(images, active_ratio=active_ratio, q_index=q_index)
    reconstruction = output["reconstruction"]
    l1 = F.l1_loss(reconstruction, images)
    mse = F.mse_loss(reconstruction, images)
    if prefix_k == 4:
        cross_rate = reconstruction.new_zeros(())
    else:
        with __import__("torch").no_grad():
            full_reference = model(images, active_ratio=1.0, q_index=q_index)["reconstruction"].detach()
        cross_rate = F.l1_loss(reconstruction, full_reference)
    rate = output["rate_proxy"] if include_rate else reconstruction.new_zeros(())
    loss = (
        float(weights["l1_weight"]) * l1
        + float(weights["mse_weight"]) * mse
        + float(weights["cross_rate_weight"]) * cross_rate
        + (float(weights["likelihood_bpp_weight"]) * rate if include_rate else 0.0)
    )
    return {
        "loss": loss,
        "l1": l1,
        "mse": mse,
        "rate": rate,
        "cross_rate": cross_rate,
        "reconstruction": reconstruction,
    }


def collate_core_v3(samples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    import torch

    rows = list(samples)
    return {
        "image": torch.stack([sample["image"] for sample in rows], dim=0),
        "metadata": [sample["metadata"] for sample in rows],
    }


def provenance_payload(code_paths: list[Path], manifest_paths: list[Path]) -> dict[str, Any]:
    from core_v3_contract import files_sha256, seed_expansion_sha256

    contract = load_contract()
    architecture_path = ROOT / "runs" / "core_v3_local" / "architecture_freeze.json"
    architecture = json.loads(architecture_path.read_text(encoding="utf-8"))
    if architecture.get("status") != "pass" or architecture.get("contract_sha256") != contract_sha256():
        raise RuntimeError("Architecture freeze is absent, failed, or belongs to another contract")
    payload = {
        "contract_version": contract["contract_version"],
        "contract_sha256": contract_sha256(),
        "code_hash": files_sha256(code_paths),
        "manifest_sha256": {path.name: file_sha256(path) for path in manifest_paths},
        "action_schedule_sha256": file_sha256(ACTION_SCHEDULE_PATH),
        "architecture_freeze_sha256": file_sha256(architecture_path),
    }
    expansion_hash = seed_expansion_sha256()
    if expansion_hash is not None:
        payload["seed_expansion_sha256"] = expansion_hash
    return payload


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
